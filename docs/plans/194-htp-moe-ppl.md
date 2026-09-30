# 194 — `htp_moe_ppl`: the fastest NPU end-to-end decode, judged by decode PPL and a benchmark

Issue: dlwlzzero/nntrainer#194 (p1). Branch `htp_moe_ppl` (cut from `htp_moe`
@ `c34fc445`). User decision 2026-09-30: **this branch drops the
bit-preserving rule**; accuracy is judged only by decode PPL and a benchmark
score; the goal is the fastest NPU decode end to end. `htp_moe` keeps the
bit rule and is not touched by this plan.

Start state: the two-session E2E path of #132 Part B on
`origin/htp/132-partb-e3` @ `69251151` (doc `docs/measurements/132-part-b-e2e.md`,
plan `docs/plans/132-part-b-two-session-e2e.md`), set_e5f as read on
`R3CY10WM83Y`, G = 64, lanes 6,3:

| side | kind, ms/token | sum |
|---|---|---|
| S1 | router 0.80, MoE 10.46 (0.475 a round; the isolated in-app round is 0.43, rule 43) | 11.26 |
| S2 | RMSNORM 0.40, FC 6.06, CONV 0.52, QK 0.10, ROPE 0.03, ATTN_M1 0.61, ADD 0.13, DENSE_FFN 2.24, LM_HEAD 3.49 | 13.58 |
| walls | S1 22.5, S2 26.0 (the sessions strictly alternate: each waits while the other computes); the ARM's round trip 29.8; the tok/s cell 29.4 → **34.0 ms a token**; A (hybrid) 53.5–54 | |

So the token is 24.8 ms of kernel time inside 34.0 ms: **9.2 ms are not
kernels** (1.2 ms of hop latency between the walls and the kernel sums,
3.8 ms between S2's wall and the ARM's round trip, 4.2 ms between the round
trip and the tok/s cell). The bytes are fixed by rule 42: MoE 484 MB, FCs
255, lm_head 147 = 886 MB a token; rule 44's ceiling ≈ 70 GB/s → 12.7 ms →
79 tok/s.

## 1. Goal and gate

**Acceptance (issue, made measurable).** Decode on the NPU end to end
(`NNTR_HTP_E2E=1`, one dspqueue packet per session per token) at
**≥ 50 tok/s at G 64 / 512 / 1024**, prompt 512, cool start, mirrored
against the same sitting's A; then as far above as the DRAM ceiling allows
(§3.4 says where that is). Read from the `decode tok/s` line of the runner
(BENCHMARK's "decode tok/s, NPU, gen 64 / 512 / 1024" row gains an
`htp_moe_ppl` line).

**Accuracy gate of this branch** (replaces contract §1's bit-identity and
"text identical" on `htp_moe_ppl` only):

| # | check | how | pass |
|---|---|---|---|
| P1 | **decode PPL, pooled** | `NNTR_PPL_DECODE` forced on A's own continuation (A self writes `cont_pXX.ids`, the variant reads it), the 8-prompt set `docs/measurements/prompts/` (prompt512 + bitset-02..08) at G = 256; pooled nll = Σ nll_sum / Σ tokens over the 8 runs (`[PPL] decode … nll_sum=` per run, 2048 scored steps); PPL = exp(nll) | `PPL_E ≤ 1.02 × PPL_A` (at A's ≈ 1.24 that is ≤ +0.020 nats a token; #134's paired 2·SE at 512 tokens was 0.0036, so the band is ≈ 5× the noise) |
| P2 | **decode PPL, per prompt** | the same 8 runs, each prompt alone | `PPL_E(p) ≤ 1.05 × PPL_A(p)` for every p |
| P3 | **benchmark: 40-question multiple choice scored from the logits** | `docs/measurements/prompts/mc-40.tsv` (new: 40 four-option factual / arithmetic / reading questions written for this repo, answer letters A–D), each rendered as `<question>\n<A..D options>\nAnswer:` and tokenized on the workstation with `tokenizers` (`$NNTR_MODEL_DIR/hf/tokenizer.json`); one app run per question with `NNTR_PPL_DECODE=q<n>.ids` (ids = the prompt's last token, then the answer letter's token), G = 1, and the new `NNTR_PPL_DECODE_ALTS=<A,B,C,D ids>` print (§3.5) that adds `alts=<id:logit,…>` to the step line; `tools/htp/mc_score.py` (new) picks the argmax over the four letters: **score = questions right / 40**, plus Σ nll of the right letter | `score_E ≥ score_A − 2` and `Σnll_E ≤ 1.05 × Σnll_A`; A's own score is recorded (a homemade set: the point is a systematic break — wrong routing, a dead head — not a 1 % drift, which P1 sees) |
| P4 | **loops** | `tools/htp/loop_check.py --prompt <p> A_pXX.log E_pXX.log` on the 8 G = 256 free-run texts | a loop on a prompt where A has none = **fail** (rule 45 (2)); A's own loops (p02 p03 p05 p06 p08 at G = 256) do not count |
| text | recorded, not gated | A's and E's G = 64 texts pasted in the handoff; the user reads them | — |
| standing | prefill ≥ −5 % of A (the prefill path is unchanged until §3.3's VTCM step, where it is a cell); `calls/token = 1.00`; timeouts 0, stale 0, `id_mismatch = 0`; no leak (S1 ceiling 3840 after every run, an A sanity after every E run) | |

The gate is read **per sitting against that sitting's A**, on the variant's
G = 256 PPL runs, never on a tok/s run. A variant that fails P1–P4 is not
carried, whatever its tok/s. Null check each sitting: A forced on its own
ids ≡ A self to 17 digits (the #134 rule).

## 2. Where it lives

Verified on `origin/htp/132-partb-e3` @ `69251151` unless marked *(ppl)* =
`htp_moe_ppl` @ `c34fc445` (the same file where the e3 diff does not touch it).

**The token, its waits and the ARM side.**
* `nntrainer/tensor/htp_backend/hmx/hexkl_token.c:121-140` `tk_take`: spins
  `spin_us` (default 0, `NNTR_HTP_E2E_SPIN_US`, `htp_compute_ops.cpp:3022`)
  then `tk_sleep` = `qurt_timer_sleep(HEXKL_TOKEN_POLL_US)` (`hexkl_token.h:76`,
  20 µs) with one read per wake; `:189-260` the S2 role, `:263-300` the S1
  role (`hexkl_token_serve`).
* `htp_compute_ops.cpp:3402` `token_us` (the ARM's round trip, printed a
  token), `:3531` the ARM's blocking dspqueue read, `:2833` the token-driver
  stretch, `:1844` `runStretchOp` (the per-op hooks the app still walks:
  `Applications/CausalLM/layers/htp_decode_hook.h:33-58`, every layer's
  forward still runs its node and returns at the hook).
* `Applications/CausalLM/models/causal_lm.cpp:704` `NNTR_OP_TIME=1` (the
  decode loop's split per node), `:411-450` `NNTR_PPL_DECODE`, `:741-749`
  the step line (target, nll, top1), `:819` the summary line.

**The FC set (S2).**
* `nntrainer/tensor/htp_backend/hvx/hvx_q4_gemv_f32.c:248-303`
  `q4m1_group_hvx`: per 32 columns × one 32-block, 8 × `vrmpyacc` then the
  exact product split and the Boldo–Melquiond sum (≈ 85 packets a step,
  `132-cpu-exact-fc-lmhead.md` §0.3: ISS 164 pcycles a step); `:131`
  `hvx_q4m1_prep` (the CPU-order Q8 quantizer, ≈ 14.7 k pcycles a call on
  silicon, rule 53); `:305` `hvx_q4m1_gemv_groups`.
* `test/htp/nntr_hvx_fc_q4.c:240-270` the per-lane DMA (`fc_dma_start` /
  `fc_dma_wait`, double-buffered), `:271-305` `fc_lane`, `:306` the run,
  `:359` the graph entry, `:380` `fc_q4m1_f32(h, variant, lanes, …)` —
  `variant` bit 16 = VTCM feed, bit 17 = L2 scratch (the IDL, `nntr_hvx.idl:675-679`).
* `nntrainer/tensor/htp_backend/hmx/hexkl_graph.c:257-316` `graph_op_fc`,
  `graph_op_dense_ffn` (up, gate, `hvx_swiglu_cpu_f32`, requantize, down),
  `graph_op_lm_head` (8 slices into `g->logits`, ban, `hvx_argmax_first_f32`).

**The small kinds.** `hvx/hvx_m1_ops_f32.c:48-62` the RMSNORM scale (16
scalar `sffma` chains, #164's CPU order), `:116` `hvx_conv_gate_m1_f32`
(scalar `sffma` taps, rule 55), `:246-282` the router (32 scalar `sffma`
chains, `expf_bionic_det`), `:283-300` the SwiGLU over the pool with the
scalar IEEE divide. `hvx/hvx_attn_m1_f32.c:305-607` P1 / P2 / P3 units
(#170 round 3; #146's O1 constant-shape and O3 unit split are the non-exact
options).

**The MoE feed (S1) and VTCM.** `hmx/hexkl_mm_u8i4_moe.c:567-592` the M=1
schedule (two `gu_bytes` slabs = 7.3 MiB; downs in a freed slab), `:837-846`
slab offsets, `:1123-1127` the fit rule `2 × gu_bytes ≤ arena`, `:1440-1490`
the per-expert pushes. `test/htp/hvx_add_f32.c:53-104` `nntr_hvx_open` →
`hexkl_micro_hw_init` takes the whole VTCM for S1's lifetime; S2 opens lite
with `vtcm_kib=0` (rule 50). Plan 132 §3.1 has the query-wrap cap
(`-Wl,--wrap=compute_resource_query_VTCM`) and the half-slab ring, designed
and unmeasured.

**Consumers that move with a changed contract.** IDL `test/htp/nntr_hvx.idl`:
**no new entry** for §3.1–§3.2 (the native FC rides `fc_q4m1_f32`'s
`variant` word; the graph kernels are chosen by a description flag, see
§3.5); §3.3 adds none either (the cap is a skel build variant). `generate_stub.sh`
+ `test/htp/build.sh` after any DSP change (rule 3). `HtpComputeOps`: the
E2E flag words and the per-kind profile line only. Quantizer format tag
(`nntr_quantize_stream`) and the loader check: **unchanged** — every lever
here changes arithmetic, not bytes. `NNTR_HTP_PROFILE` per-kind tables and
`tools/htp_fc_report.py`: unchanged kinds; the graph line gains `hop_us`
and the ARM line `arm_us` (§3.1 L0). The app: `causal_lm.cpp` gains
`NNTR_PPL_DECODE_ALTS` (§3.5); `tools/htp/mc_score.py` and
`docs/measurements/prompts/mc-40.tsv` + its `.ids` are new.

## 3. Design

### 3.1 The levers, ranked by ms/token (G = 64 unless stated; E5f is the base)

| # | lever | today | after | Δ ms/token | what it frees / how; confidence |
|---|---|---|---|---|---|
| **L0** | **the ARM's 8 ms outside S2's wall** (34.0 − 26.0): the blocking dspqueue read's wake-up, the embedding row, the 228-node walk with a hook per node, the sampler / tokenizer / print | 8.0 | ≤ 2.0 | **−5 to −7** | *Measure first* (`NNTR_OP_TIME=1` on the E build + `arm_us` brackets around the packet post / read / append). Expected fix: the decode step calls the token driver from the top of the model's step (one call, the layer walk skipped for the resident stretch as `runStretchOp`'s first hook already keeps the row), and the ARM's read of S2's response spins ≤ 200 µs before it blocks (the token is ≈ 20 ms; the ARM is otherwise idle). Not a kernel change; no PPL effect. The 4.2 ms between the round trip and the tok/s cell is the app's per-token path — A pays its own inside its 9.9 ms CPU share, so ≈ 2.5 ms of it is inherent and the rest is the E-only walk. Confidence: the split is unmeasured; the range covers "half of it is inherent" |
| **L1** | **FC / DENSE_FFN / LM_HEAD accumulate in native qf32** (the FMA emulation goes; `vrmpyacc` int32 per block, `Q6_Vqf32_vmpyacc` of `isum` by `d_w·d_a` per block, one `Vsf_equals_Vqf32` a group) and a **vector Q8 quantizer** (no CPU order: `vmax` amax, `vmpy` by 127/amax, `vcvt`) | 11.79 (402 MB at 34 GB/s: compute-bound, §0.3 of Part A) | 8.0 (L2-fed at ≈ 50 GB/s; 7.7 at K=2048's measured 52; 10.0 if K=7168 stays at 40) | **−3.8** (−1.8 … −4.1) | ≈ 85 → ≈ 12 packets a step, so the FC becomes DMA-bound; the lane ladder must be re-read (6 lanes at K=7168 was slower than 3 *because* of compute). Quantizer 14.7 k → ≈ 3 k pcycles × 74 calls ≈ −0.45 ms (inside the 11.79). PPL: the CPU's own Q8_0 × Q4_0 dot in a different summation order and one rounding per block instead of per FMA — below the quantizer's own noise; P1 expected within +0.2 % |
| **L2** | **router in vector f32**: `vrmpy`-free f32 dot over 2048 × 32 on HVX (qf32 accumulate), vector `exp` (the ATTN_M1 exp16 table), top-4 on 32 lanes | 0.80 (36 µs a round: 32 scalar chains + `expf_bionic_det`) | 0.15 | **−0.65** | Non-exact routing: a flipped expert on a near-tie is the one visible risk; P1/P2 read it (rule 45's D1 lost 0.9 % PPL from norm-induced flips — this is the same class, so the router keeps f32 and the gate stays tight) |
| **L3** | **RMSNORM / QK_NORM sum-of-squares, CONV1D_GATE taps and the SwiGLU on HVX** (`Q6_Vqf32` chains; the divide as `vrecip` + one Newton step) | 0.40 + 0.52 + (SwiGLU inside 2.24) | 0.12 + 0.12 | **−0.7** | #164 / rule 55 bought bit-identity with scalar chains; this branch does not need it. The norm sits before every quantizer, so its rounding is what P1 sees first |
| **L4** | **deadline hop wait**: each side knows the other's expected duration (S1's round ≈ 0.45 ms, S2's stretch ≈ 0.5); `tk_take` sleeps until expected − 40 µs, then spins with `pause` — a bounded spin (≤ 50 µs a hop, 44 hops ≈ 2 ms of one thread a token) instead of 20 µs sleeps whose granularity is the QuRT tick | 1.2 (26.0 − 24.8) | 0.4 | **−0.8** | Rule 56 forbids the *unbounded* spin (it stole a pool lane for the whole other side's compute); a spin that starts at the deadline steals ≤ 5 %. Cell: `hop_us` per side in the profile line, poll 5 / 10 / 20 µs beside it |
| **L5** | **MoE round at its isolated in-app rate** (0.475 → 0.43): why the E2E round is 10 % slower than the hybrid's is unknown — S2's sleeping thread, DVFS votes of two PDs, or the l2fetch lead cold after a 0.5 ms idle | 10.46 | 9.5 | **−1.0** | Measure: the MoE's `dsp` / `mm` per round inside E vs A; try the DCVS vote from both sessions (`HAP_power_set` on S2 too) and a `l2fetch` of the first expert's slab issued when S1 receives the ping. No arithmetic change |
| **L6** | **S1 / S2 VTCM share** (S1 4 MiB, S2 4 MiB) so S2's FC is VTCM-fed (57 GB/s, `src_bypass`) and S2 **prefetches the next layer's first FC slabs during S1's MoE window** (≤ 2.5 MiB a layer into VTCM, 22 layers ≈ 55 MB at the 13 GB/s rule 44 leaves beside the MoE: free) | FC at ≈ 50 (after L1) | FC at 57; 55 MB off the critical path | **−0.7 −1.0** | Decode-only MoE feed need: the feed is DMA-bound, so bytes in flight only cover the DMA issue latency; a **4-chunk ring of gate_up quarters (4 × 0.92 MiB) + downs into freed quarters = 3.7 MiB**, the chunk waits *inside* the pool run (each lane polls its chunk's descriptor, as `fc_dma_wait` does) so there are no extra fork / joins (plan 132 §3.1 priced those at 20–40 µs each). **Prefill**: the HMX layout needs ≥ 6720 KiB in S1 and VTCM is sized at `hw_init` for the session's life, so with 4 MiB the M>1 MoE runs a K-split (two half-K passes, f32 partial sums — no longer bit-identical, which this branch allows) at a prefill cost the −5 % gate reads; if it fails, S1 keeps 8 MiB, S2 stays L2-fed and the prefetch uses S2's 2 MiB L2 scratch instead (≈ −0.6). Query-wrap cap as plan 132 §3.1 |
| **L7** | **non-exact attention** (#146 O1 constant-shape inner kernel, O3 unit balance, exp16 without the CPU order, fp16 P·V) | 0.61 (G=64), ≈ 1.0 (G=1024, 6 × 314 k pcycles) | 0.4 / 0.6 | **−0.2 / −0.4** | The only G-dependent term; #170 round 3 already took most of it. Third in order because it is small and its PPL effect (fp16 P·V) is the least predictable |
| L8 | lm_head: fp16 logits, early top-k | 3.49 | — | **0** beyond L1 | The lm_head is 147 MB of weights a token; the logits (512 KiB) are 0.3 % of that and never leave the DSP except under `NNTR_PPL_DECODE`. Top-k / vocab pruning would change the PPL's softmax: not allowed. Rejected as a lever |
| L9 | fp16 activations | — | — | **0** | At M=1 an activation is 8–28 KiB; the FC input is Q8 already. fp16 buys nothing in ms; taken only inside L7 where it removes a conversion |
| L10 | LUT GEMV (T-MAN, W4A16 via `vlut16`) instead of `vrmpy` | — | — | **0** | T-MAN's own ceiling is the same DMA→TCM 59 GB/s we read (rule 43: 57); our `vrmpy` path already hides compute under the DMA at M=1 (the MoE round: ≈ 240 µs compute under 440 µs of DMA). A LUT only helps where compute is the wall (our exact FC was — L1 removes that) or below 4 bits. Rejected; revisit only if a 2–3-bit weight format is ever adopted |
| L11 | CPU as a second reader (one expert of four on the CPU, or the lm_head) | — | — | ≤ 0 today | Rule 44: the MoE takes 57 of 70 GB/s; the CPU's 13 GB/s share reads a 121 MB expert slice in 9.3 ms > the 8.5 ms the DSP needs for all four. Only pays once the DSP-side floor is reached and only for bytes the DSP would otherwise read *serially* with nothing overlapping — that is L6's prefetch, done on the DSP. Contract §3.2 (Q11) keeps it a user decision; not in this plan |

**Order:** L0 and L1 are independent and largest, and L1 is fully
host-gateable (a scalar spec of the *new* arithmetic + SNR against the f32
reference); they go first, in one sitting. L2–L4 are small kernels with the
same host proxy. L5 and L6 need the device to say anything. L7 last.

### 3.2 Rejected alternative for the FC

Keep the exact kernel and hand-schedule it (plan 132 §3.5's "−1.0 toward
57 GB/s"): the ISS said ≥ 76 packets a step remain even in the native-asm
form that does not run on v79 (rule 53), so the exact FC is compute-bound
at ≈ 8 ms at best — the byte floor is 7.05. The native accumulate removes
the wall instead of shaving it, and the PPL gate is what this branch bought
to be allowed to.

### 3.3 Contract §2 and doc 45 §3 check

Three walls: unchanged in S1 (L5 tunes the feed, L6 re-rings it: the DMA
stays hidden behind compute, the scoreboard in `moe_layer_host_check.c`
proves the order). Arena budget: S1 3840 MiB, S2 ≤ 512 MiB, heaps capped
(rules 50, 54: S1's arena mapped before S2 opens). No CPU fallback for
`QS4CX_WH`: unchanged. `_det` before every quantizer: **the rule changes
meaning here** — every op before a quantizer still ships with a scalar spec
in `test/htp/host/`, but the spec describes the *new* numerics (qf32
accumulate, vector exp) and the host check holds the HVX kernel to it
bit-exactly, while a second check holds the spec against the f32 reference
at an SNR floor (≥ 60 dB for FC / lm_head outputs, ≥ 50 for norms, routing
identical on the fixture rows). Rule 53: `Q6_Vsf_*` / `Q6_Vqf32_*`
intrinsics only, never the asm `.sf` forms.

### 3.4 Cost model (ms/token) and where the DRAM ceiling sits

| stage | G=64 | G=512 | G=1024 | tok/s (64 / 1024) |
|---|---|---|---|---|
| E5f as read | 34.0 | 32.4 | 32.5 | 29.4 / 30.8 |
| + L0 (−5, range −3…−7) | 29.0 | 27.4 | 27.5 | 34.5 / 36.4 |
| + L1 (−3.8) | 25.2 | 23.6 | 23.7 | 39.7 / 42.2 |
| + L2 + L3 + L4 (−2.15) | 23.05 | 21.45 | 21.55 | 43.4 / 46.4 |
| + L5 (−1.0) | 22.05 | 20.45 | 20.55 | 45.4 / 48.7 |
| + L6 (−1.7) | 20.35 | 18.75 | 18.85 | **49.1 / 53.1** |
| + L7 (−0.2 / −0.4) | 20.15 | 18.55 | 18.45 | **49.6 / 54.2** |
| L0 at its optimistic end (−7) | 18.15 | 16.55 | 16.45 | 55.1 / 60.8 |

The G=512 / 1024 columns start 1.5 ms below G=64 because E5f's own G=64
cell carries the first token's 55 ms over 64 tokens and the attention term
is flat (0.61 → ≈ 1.0 ms across G). Read plainly: **≥ 50 at every G needs
L0 at or above its mid estimate plus L1 and the small levers; at L0's low
end the plan lands at ≈ 47–49 at G=64 and 50–52 at G ≥ 512.** The plan
therefore measures L0's split first (sitting 1) and re-writes this table
with the reading before L5–L7 are built.

**Ceiling.** With the alternation, the DRAM-bound stretch is MoE 484 MB +
S2's 402 MB read serially: at 57 GB/s = 15.5 ms; plus the compute-only
terms (router 0.15, norms / small 0.5, attention 0.4–0.6, hops 0.4, ARM
≥ 1.5) ≈ 3.0 → **18.5 ms = 54 tok/s is this design's floor without
overlap.** L6's prefetch moves ≈ 55 MB (≈ 1 ms) off the serial path; a
deeper overlap needs on-chip room that does not exist (S2's VTCM + L2 ≈ 4–6
MiB a layer against 11.6 MB of FC bytes a layer), so **≈ 56–58 tok/s is
where the two-session alternation stops**, and 79 (886 MB at 70 GB/s) is
reached only if the MoE and the FC stream *concurrently* — impossible by
data dependence within a token. The gap between 58 and 79 is the price of
the layer-serial dependence, not of this code.

### 3.5 Switches and the benchmark plumbing

* Every lever is a bit in the graph description's flag word (S2 kinds) or
  `hexkl_moe_flags` (S1), read from `NNTR_HTP_PPL_LEVERS=<mask>` (default:
  all on once merged; a handoff runs `A` = the hybrid, `E0` = mask 0 = the
  bit-identical base, `E<n>` = the sitting's mask). One build, one skel per
  sitting; no lever needs its own binary except L6's cap (a skel variant).
* `NNTR_PPL_DECODE_ALTS=<id,…>` (≤ 8 ids): the step line gains
  `alts=<id:logit,…>` from the logits the app already holds under
  `NNTR_PPL_DECODE` (`causal_lm.cpp:741-749`); ≈ 20 lines, off by default,
  never set in a tok/s run.
* `tools/htp/mc_score.py`: reads the 40 logs, prints `MC <label>
  right=<n>/40 nll_sum=<x>` and the per-question letter; `mc-40.tsv` holds
  question, options, answer; `tools/htp/mc_ids.py` renders and tokenizes
  (prompt text + the two-id file) once on the workstation.
* Host PPL proxy: `run_inproc_e2e.sh` already prints `E2E ppl-decode hd64
  off= on= delta=` and the lfm25-shaped fixture's `min_snr_db`; with the
  levers on, the gate is `delta ≤ 2 %` on hd64 and `min_snr_db ≥ 30` on
  lfm25 (the fixtures are random-weight tiny models: they prove the
  pipeline and the numerics' order of magnitude, not the 8B's PPL — that
  is a device reading, stated as such).

## 4. Steps (each ends in a rung of `.claude/skills/hexagon-gates`)

* **S0. Base.** Rebase `htp/132-partb-e3` @ `69251151` (with set_e5g's
  teardown order and the `S1Ceiling` cell) onto `htp_moe_ppl` as PR 1; add
  the lever mask, `NNTR_PPL_DECODE_ALTS`, `mc-40.tsv` + ids, `mc_score.py`,
  the `hop_us` / `arm_us` brackets, the `NNTR_OP_TIME=1` split on the E
  path. Gate: rung 1 (`ALL CHECKS PASS`, `INPROC E2E PASS`, `*Lfm2Moe*`
  6/6, `mc_score.py --self-test`), rung 2 (skel md5), rung 3.
* **S1. L1 on the host.** `hvx_q4_gemv_f32.c` gains `q4m1_group_native`
  (qf32 accumulate) and `hvx_q4m1_prep_vec`; spec `q4_gemv_native_det.h`
  (scalar model of the qf32 chain: per-block `isum × d`, f32 sum) and the
  SNR check against the f32 GEMV in `q4_gemv_host_check.c`; the graph ops
  pick the kernel by the flag. Gate: rung 1 (`Q4 NATIVE bit-exact vs spec
  n/n`, `snr_db ≥ 60`, `E2E ppl-decode hd64 … delta ≤ 2 %`), rung 2, rung 3
  (`HvxFcQ4.NativeMatchesSpec` gtest for the device canary).
* **S2. Device sitting 1 (unavoidable)** — `docs/measurements/194-sitting-1.md`,
  ≈ 100 min, reboot first, zone0 ≤ 35 °C before each G block, S1 ceiling
  read after every run, an A sanity after every E run. Variants:

  | variant | what | cells |
  |---|---|---|
  | **A** | hybrid, nothing set (unchanged reference) | prompt 512, G 64 / 512 / 1024 ×2 mirrored `A E0 E1 E1 E0 A`; 8 prompts G=256 self (ids) + text; mc-40 |
  | **E0** | `NNTR_HTP_E2E=1`, mask 0 (the bit-identical base) | tok/s at each G; `NNTR_OP_TIME=1` + `arm_us` once per G (**the L0 split**); `hop_us`; 8 prompts forced (the gate's null: PPL ≡ A); mc-40 (≡ A) |
  | **E1** | E0 + L1 | tok/s; per-kind line; FC lane ladder 3 / 6 at K 7168 (`NNTR_HTP_FC_LANES`); 8 prompts forced → P1 / P2 / P4; mc-40 → P3; text |

  Reads: the L0 split (decides S3's design), L1's Δ and the lane ladder,
  the first PPL of a non-exact kernel. **Stop rules:** `0x8000040e`;
  `LEAK` (ceiling < 3840); `AEE_EEXPIRED`; E1 fails P1–P4 → L1's numerics
  go back to S1 (a per-block f32 sum instead of qf32 across the group is
  the fallback) before anything else is built.
* **S3. L0, L2, L3, L4 on the host.** L0 per the sitting's split (the
  one-call decode step; the ARM's bounded spin); the vector router / norms
  / conv / SwiGLU with specs + SNR checks; the deadline wait in
  `hexkl_token.c` with `token_host_check.c` proving no lost post at spin
  bounds 0 / 50 / 200 µs (10 000 tokens, timeouts 0). Gate: rung 1 (each
  spec `bit-exact n/n`, SNR floors, `TOKEN DRIVER … timeouts=0`, `E2E
  tokens fwd==off` on the fixtures now `expected_mismatch` allowed with
  `min_snr_db ≥ 30`), rung 2, rung 3.
* **S4. Device sitting 2** — `194-sitting-2.md`, ≈ 100 min. Variants
  **A**, **E1** (the anchor from sitting 1), **E2** = E1 + L0 + L2 + L3 +
  L4, **E2s** = E2 with the hop spin bound at 200 µs (rule 56's check).
  Cells as sitting 1 plus `hop_us`, `arm_us`. Re-write §3.4 with the
  readings. Stop: E2 fails P1–P4 → the lever whose spec has the lowest SNR
  is masked off and the sitting's E2 is re-read as the anchor.
* **S5. L5 and L6 on the host.** The quarter ring in
  `hexkl_mm_u8i4_moe.c` under a flag, the scoreboard
  (`moe_layer_host_check.c`: every chunk consumed after its wait, every
  reuse after its join); the cap skel (`NNTR_VTCM_CAP_KB=4096`, plan 132
  §3.1); S2's prefetch lane (issue the next layer's first per-lane slabs
  into VTCM when S2 posts its ping; the FC's lane 0 starts from them); the
  prefill K-split under 4 MiB. Gate: rung 1 (`ALL CHECKS PASS`, `*Lfm2Moe*`
  6/6 with the ring on, `INPROC E2E PASS`), rung 2 ×2 skels, rung 3.
* **S6. Device sitting 3** — `194-sitting-3.md`, ≈ 120 min. Variants **A**,
  **E2** (anchor), **E3** = E2 + L5 + L6 (cap skel, ring, VTCM-fed FC,
  prefetch off), **E3p** = E3 + prefetch. Cells: tok/s at each G, the MoE
  round `dsp` / `mm`, `s2 vtcm_kib`, FC GB/s, **prefill** tok/s of E3 vs A
  (the −5 % gate on the K-split), P1–P4, text. Stop: prefill < −5 % → L6
  reverts to L2-fed + L2-scratch prefetch (a flag, same skel set) and E3 is
  re-read that way in the same sitting.
* **S7. L7** per #146 O1 / O3 with specs at SNR ≥ 40 dB (fp16 P·V);
  sitting 4 (**A**, **E3**, **E4**) only if §3.4's re-written table still
  needs it at G=1024. Gate: rung 1–3.
* **S8. Fold and default.** When a variant reads ≥ 50 at every G with
  P1–P4 held in one sitting: `NNTR_HTP_E2E` and the lever mask become the
  default on `htp_moe_ppl` (contract §12's flip rule, with this branch's
  gate in place of bit identity), BENCHMARK / LEDGER rows, the issue to
  `state:measured`.

**Method, every sitting** (unchanged from #132 / #170): prompt 512, G 64 /
512 / 1024, mirrored A/B with A first and last, thermal checkpoints (zone0
before each block, ≤ 35 °C to start), reboot first, `md5.txt` on both ends,
the S1 ceiling read after every run, an A sanity after every E run, the 8
prompts at G=256 with A self then every variant forced, `loop_check.py`
per prompt, mc-40 for A and every variant, texts pasted for the user.

## 5. Risks (host vs device)

* **L0 is a reading, not a design, until sitting 1.** If the 8 ms is mostly
  the app's inherent per-token path (tokenizer, print, sampler), the
  reachable Δ is ≈ −3 and §3.4 lands at 47–49 at G=64: the table says so
  and S8 does not flip a default that misses 50 at any G.
* **PPL on the 8B is a device-only number.** The host proxies (specs, SNR,
  the tiny fixtures' PPL delta) bound the numerics; a routing flip on the
  real model's near-ties (rule 45's D1: +0.9 % PPL and new loops from the
  norms alone) shows only on the phone. Hence the per-sitting order: the
  largest, most predictable levers first (L1), the router (L2) with its own
  variant bit so it can be masked off alone.
* **Rule 56 in L4.** A bounded spin that starts early (a misestimated
  deadline) is the unbounded spin again; `hop_us` and the MoE round's
  `dsp` in the same profile line show it, and E2s reads the bound.
* **DMA rate / DVFS / thermal drift.** Every Δ is read inside one sitting
  against its A, mirrored, cool start; the E-vs-A prefill cell of sitting 3
  is the only prefill reading that can move and it is a gate.
* **Stale skel and two binaries.** One tree per sitting, md5s in the
  handoff, `HvxFcQ4.NativeMatchesSpec` as the canary beside
  `MatchesSpecBitExact` (E0 must still be bit-identical: a broken E0 means
  a broken build, not a lever).
* **Address space and the leak** (rules 50, 54): S1's arena before S2's
  open; heaps capped; the ceiling cell after every run; `LEAK` stops the
  sitting. L6's cap changes S1's `config_off` and the ring's VTCM offsets
  — the scoreboard proves the layout on the host, the device proves the
  VTCM window (plan 132 §5's first two risks stand).
* **The prefill K-split under L6** is the one lever that touches M>1 and
  the −5 % prefill gate; its fallback (L2-fed + L2-scratch prefetch) is in
  the same skel set so the sitting answers it either way.

## 6. Docs to update

* **`docs/htp_moe/BENCHMARK.md`**: an `htp_moe_ppl` block — the decode
  row's `htp_moe_ppl` line (A / E0 / E1 / E2 / E3 per sitting, G 64 / 512 /
  1024, prefill, `calls/token`, `hop_us`, `arm_us`), the per-kind lines per
  variant, the gate table (pooled PPL, per-prompt max ratio, mc-40 score,
  loops) per variant, the FC lane ladder, L6's VTCM cells (`vtcm_kib`, MoE
  round at 4 MiB, prefill Δ), artifact rows.
* **`docs/htp_moe/LEDGER.md`**: rules — *what the ARM spends outside S2's
  wall in the E2E path and what removed it*; *the native-qf32 FC's rate on
  silicon and the lane ladder once compute is gone*; *the PPL cost of each
  non-exact kernel (router, norms, FC) on the 8B*; *the deadline hop wait
  vs rule 56*; *the M=1 feed at 4 MiB (quarter ring) and the prefill
  K-split's cost*. Open items: a `htp_moe_ppl` section beside the
  bit-preserving ones so the two tracks do not share verdict rows; ㉘ / ⑨
  get a pointer to this branch's outcome. Contract §12: an `htp_moe_ppl`
  row stating this branch's gate.
