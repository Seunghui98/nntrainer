# Plan 261: decode levers after the first prefill + decode E2E sitting

Issue #261 (p1). Host-only ranking of the one-PD Gemma-4 26B decode token's
non-weight levers from #234 P4 (`docs/measurements/234-p4-fcwh-decode.md`,
G 512 F16, S25 v79, dummy, prompt 447, C 16) and #260 step 1b
(`260-step1b-4415-c2ec4e90c.md`, pure #4415, real QS4CX file). Tags: [M-P4]
/ [M-1b] measured there, [G] arithmetic from measured rates, [E] estimate.
Tree `htp_decode` @ `f882a0d05`; #4415 at `41bd65e04`.

## 1. Goal and gate

Contract §1: decode ≈ 40 tok/s at G 512 on the real file against the same
sitting's hybrid A, G 64 / 512 / 1024, cool start. Gates per lever in §2;
standing: prefill ≥ −5 % of A (recorded), tokens == A under plan 130 §3.5
for levers 1 / 2 / 4 / 5, decode PPL ≤ +2 % of A + text + user approval for 3.

The token at G 512 [M-P4]: **62.9 ms** = DSP wall 56.45 (ATTN_M1 13.74, MOE
11.39, FC 10.65, LM_HEAD 8.18, DENSE_FFN 5.35, ROUTER_TOPK 4.04, norms /
RoPE / ADD 2.86) + **6.3 ms outside the token call** (`arm token_ms` 56.61
vs 1000 / 15.885). The outside part, from the `NNTR_OP_TIME=1` run
(`optime_F16_G64.log`, steady state = (sum − max) / 63 per node type):
`logit_softcapping` **1.27**, stale hook-less layer walk ≈ 1.1
(`scalar_multiply` 0.24, `lfm2_moe` prologue 0.23, norms 0.30, `addition`
0.13, `fully_connected` 0.085, `activation` 0.08, rest 0.07), ≈ 3.9
unattributed (sampler, tokenizer, embedding, the walk's dispatch). The 1 MiB logits hand-back is
**already off** on a greedy run (`causal_lm.cpp:793`: logits only with
`NNTR_PPL_DECODE`, sampling, a processor or a batch; P4's `id_checked=0`).

## 2. Per-lever arithmetic, ranked by ms per day

| # | lever | ms before → after | gain | days | ms / day | gate |
|---|---|---:|---:|---:|---:|---|
| 1 | **ROUTER_TOPK: prefetch the weight block** | 4.04 → ≤ 1.0–1.5 | −2.5…−3.0 [G] | 1 | **3.0** | bit-identical |
| 2 | **`logit_softcapping` skipped at a resident row, no logits wanted** | 1.27 → 0 | −1.27 [M-P4] | 0.5 | **2.5** | bit-identical (PPL path untouched) |
| 3a | **ATTN_M1 softmax vectorised over positions, fp16 KV kept** | slope 17 → ≈ 8 µs/pos | −4…−6 at G 512 [E] | 2–3 | ≈ 2.0 | D2: PPL ≤ +2 %, text |
| 4 | **MoE 4-slab feed** (plan 229 §3.3) | 11.4 → ≈ 9.4 (floor 7.0) | −2, bound −3.4 [E] | 2 | 1.0 | bit-identical (MoE dumps) |
| 3b | **int8 KV masters on the DSP + `vrmpy`** (after 3a) | ≈ 8 → ≈ 4 µs/pos; heap 440 → 220 MiB | −4…−5 more [E] | 4–5 | ≈ 1.0 | D2 as 3a; rule 71's C lever |
| 5 | **hook-less CPU layers skipped** (㉞ b) | 1.1 → ≈ 0.5 | −0.5…−0.6 [M-P4] | 1 | 0.5 | bit-identical |

**First two after the E sitting: 1 and 2** — 1.5 days, −4.3 ms, both
bit-identical, no IDL change: 62.9 → 58.6 ms ≈ **17.1 tok/s** (+7 %) on the
P4 base. Then 3a, 4, 3b, 5. All rows: ≈ 46 ms → ≈ 21.7 tok/s [G/E] on 4-bit
FCs; with plan 229 §8.1's weight levers (−8 FC 2-bit, −1 / −4.6 head) ≈ 33–37
ms → **27–30 tok/s**. 40 (25 ms) is out of this list's reach: what is left is
MOE at its floor (7), FC + head at 2 bits (≈ 12), norms 2.9, ATTN ≈ 4, the
ARM's 3.9 unattributed — the next plan is those, not more bytes.

### 2.1 Lever 1 — router (`hvx_router_softmax_topk_f32`)

`router pcyc/op=281 879` at `mhz=2094` [M-P4] = 134.6 µs × 30 = 4.04 ms for
30 × [2816 × 128] f32 = 43 MB → **11 GB/s**. Per k-step of `router_sm_lane`
(`hvx_m1_ops_f32.c:362-374`): 281 879 / 2816 = **100 cycles** = one DDR line
miss: the lane walks column block `b` at a 512 B stride (`col + k * Ep`), a
new 128 B line per step, 4 lanes of 6, no prefetch — plain DSP heap f32
(`hexkl_graph.c:36-38`), no DMA, no `l2fetch`. The LFM sigmoid router beside
it has the fix: `ROUTER_PF_ROWS` `Q6_l2fetch_AP` of the next 16 KiB +
`ROUTER_L1_AHEAD` L1 hints (`:244-300`; ISS 47.7 k → 15.7 k, #132 S3).
Target: the same two hints in `router_sm_lane` (rows `k + 128 … k + 256` of
all four blocks are one 2D fetch of 512 B rows), bits unchanged (hints only;
the chain order stays the CPU's `m1_router_softmax_det`). Floor: the chain's
latency 2816 × 2 dependent sf ops ≈ 0.5–0.7 ms a token; at the DDR rate
43 MB / 44–52 GB/s = 0.8–1.0 ms. Rejected: f16 / bf16 storage — changes the
product unless the file's tensor is bf16-exact (check the converter first).

### 2.2 Lever 2 — softcap

`LogitSoftCappingLayer::incremental_forwarding` (`Applications/CausalLM/layers/logit_softcapping.cpp`)
runs 262 144 `tanh` on the stale lm_head row every token: 1.27 ms [M-P4]. The DSP's LM_HEAD does
not cap (`gemma4_moe_causallm.cpp:155-159`, `shape.softcap = 0`) and its
argmax on raw logits is the capped argmax (tanh monotonic) — that ponytail.
Change: skip the layer when `htpDecodeRowResident(from)` and no logits are
wanted — a `decode_logits_wanted()` getter beside `set_decode_logits`
(`compute_ops.h:403` family, `htp_compute_ops.cpp:4666`, `:7084`; hook in
`htp_decode_hook.h:126-140`). With logits wanted (`NNTR_PPL_DECODE`,
sampling) the layer caps the returned logits as today: PPL numbers unmoved.
Rejected: capping on the DSP (`eps_bits` + `hvx_softcap_f32`,
`hexkl_graph.c:538-551`) — shifts the PPL path's logits by the HVX-vs-CPU
`tanh` and buys nothing on the id path.

### 2.3 Lever 3 — attention: what the slope says, what #4415 has

ATTN_M1 9.56 / 13.74 / 17.73 ms at mean context 479 / 703 / 959 [M-P4] →
**17.0 µs per position**, intercept ≈ 1.4 ms (30 layers × 3 pool runs); per
layer 0.57 µs / position at head_dim 256 against LFM's 1.05 at head_dim 64
(㉗). The cost does **not** scale with head_dim, and the kernel reads 158 MB
in 13.7 ms = 11.5 GB/s: neither byte- nor FMA-bound — it is the per-position
softmax (the exp table gather, rule 58: 387 k of the LFM op; the ET
transpose; the serial fp16 sum; the divide), all forced by bit-identity with
the CPU's fp16 sequence (`attn_m1_det.h`). So "bytes ½ → −5…−7 ms" does not
follow: halving the KV bytes leaves the slope. **#4415 shows it [M-1b]: q8
decode 3.02 vs fp16 3.50 tok/s (−14 %) at 447 G 64.**

At #4415: `attn_q2_step` (`nntr_hvx.idl:1167`, `hexkl_attn_q2.c`) is the
HMX a16 / kv8 row-blocked kernel at n_q = 1 — a 64-row block for one row
(wall 1 again), its own FastRPC call per layer with the ARM quantising Q to
u16, the HMX lock and a VTCM plan S1's feed holds: **not usable inside the
token**, and why their q8 decode is slower than their fp16. `attn_q_decode`
(`idl:1104`, HVX `vrmpy` over grouped masters `[hd/4][rows][4]`) is the
right shape but **head_dim ≤ 128** and rejects the fixed-scale plain masters
(`hexkl_kv_q.h` `plain_masters`, `1845d675d`); Gemma is 256 / 512. Reusable:
`hexkl_kv_q.h`'s scalar quantiser / index helpers (plain C, host-tested) and
its `vrmpy` master layouts; `hvx_softmax_q` is 64-row-tile shaped, not M = 1;
doc 60's "K/V straight to int8 masters on the DSP" is the append we need.

Design (D2 kernel, its own spec `attn_m1q_det.h` + host check): **3a**
keeps the fp16 cache and the scores, replaces the softmax with
an hf polynomial exp over 64 positions per vector (no gather, no ET
transpose; vector sum, one `hvx_hf_div16` per head) — [E] slope 17 → ≈ 8
µs/pos, −4…−6 ms at G 512, −8 at G 1024. **3b** quantises the appended row
to int8 on the DSP after RoPE (per-(row, head) scale + colsum; the seed path
`decode_kv_seed_fp32` `htp_compute_ops.cpp:2416` quantised at seed), stores
K^T `[hd/4][rows][4]` / V `[rows/4][hd][4]`, scores and PV by `vrmpy`: byte
floor 112 KB / pos at 50 GB/s = 2.2 µs → slope ≈ 4 µs [E], heap 440 → 220 MiB
(rule 71: C 40 → ≈ 60 on the S25). 3b changes the registration
(`nntr_hvx_attn_m1_register` gains a kind) → IDL + `generate_stub.sh` +
`HtpComputeOps` + the hd64 fixture lines; 3a changes no IDL. Rejected: int8
KV first (the issue's order) — by the slope it moves ≈ 1 ms until the softmax
is restructured, and it costs the IDL change.

### 2.4 Lever 4 — MoE 4-slab feed

MOE 11.39 ms = 30 × 0.373 [M-P4], 10.4 net of misses; per expert 1.52 MB in
43 µs = **35 GB/s** vs the WH FC path's 52 (plan 229 §8.1) → floor 7.0 ms. The M = 1 ring stages gate_up of expert i in slab
i & 1 and waits at the join (`hexkl_mm_u8i4_moe.c:734-770`, `moe_m1_gu_off`
`:1093`, feed condition `:1536-1538`: 2 × feed_gu ≤ arena); at 2 bits a
Gemma slab pair is 1.98 MB of the 8 MiB, so four slabs (i & 3, push two
experts ahead) fit with the downs packed as today. Expected −2 ms [E] (the
exposed per-expert DMA ramp); −3.4 is the floor. Gate: `dma_replay` /
`dma_trace` host checks on the new schedule, `moe_layer_host_check`
bit-identical, MoE dumps `bit_identical=1`; PUSH / DMA_KB accounting
unchanged (`:1107-1112`).

### 2.5 Lever 5 — hook-less layers

Of the 1.1 ms [M-P4], `scalar_multiply` + `lfm2_moe` prologue + `activation`
≈ 0.55 are per-layer skips (`htpDecodeRowResident`, as `dense_ffn_layer.cpp:130`);
the rest is the walk's dispatch over ≈ 1 000 nodes, which only a
network-level jump to the lm_head node removes (2 d for ≈ 1 ms, after 1–4).

## 3. Where it lives (verified `path:line`)

* Router: `nntrainer/tensor/htp_backend/hvx/hvx_m1_ops_f32.c:362-374`,
  `:244-300` (the pattern), `hmx/hexkl_graph.c:377-392`; checks
  `test/htp/host/{m1_ops,graph}_host_check.c`. No IDL / format / loader change.
* Softcap: §2.2's files; no DSP change. `tools/htp_fc_report.py` untouched.
* MoE feed: `hmx/hexkl_mm_u8i4_moe.c:1089-1140`, `:1536-1548`; no IDL.
* Attention: `hvx/hvx_attn_m1_f32.c`, `hvx_attn_m1_hf.h`, `attn_m1_det.h`
  (kept), `hexkl_graph.c:349` (calls `hvx_attn_m1_forward`, **no `_prof`**),
  `test/htp/nntr_hvx_attn_m1.c`; 3b only: `test/htp/nntr_hvx.idl` + stub,
  `htp_compute_ops.cpp` registration, `test/htp/host/attn_m1_host_check.c`.

## 4. Steps (each ends in a `.claude/skills/hexagon-gates` rung)

* **S0 — read the E sitting (#260 rev. 2; no code).** What it must print,
  per cell (A / E × 447 / 1023 × G 64 / 512 / 1024; further device reads use
  the user's p512 / p1024): the per-kind line (ROUTER_TOPK flat over G and
  context → lever 1; ATTN_M1 at three G → slope / intercept on the real file
  for 3a / 3b), the `graph moe pcyc/op … router pcyc/op` + `mhz` line,
  `token driver: pool` (misses/token, lever 4's net), `token driver: close …
  id_checked=` (0 ⇒ the hand-back is already off: lever 2's premise),
  `generation:` tok/s beside `arm token_ms` (the outside-the-call ms), and
  **one extra `NNTR_OP_TIME=1` run at 447 G 512** (not a tok/s cell): the
  `logit_softcapping` node line, §1's by-type aggregate, the `[OP-TIME]
  step` line. ATTN_M1 phase words cannot come from the E sitting (no `_prof`
  in the graph op): ask #260's implementer for a `NNTR_HTP_PROFILE`-gated
  `_prof` call + stderr line, else it is S3's first commit. Gate: the
  numbers in a #261 comment.
* **S1 — router prefetch** (1 d): rung 1 (`run_host_checks.sh` `ALL CHECKS
  PASS`, router lines bit-identical), rung 2 v79 + v81, rung 3.
* **S2 — softcap skip** (0.5 d): rung 1 (`run_inproc_e2e.sh` lines unchanged,
  `E2E ppl-decode self==forced`), rung 3. **Device (unavoidable, with S1):**
  A = the E sitting's E cell unchanged, B = S1 + S2, prompts 512 / 1024, G 64
  / 512 / 1024, cool start, one sitting; gate: tokens B == A, ROUTER_TOPK
  ≤ 1.5 ms, outside-the-call −1.2 ms, the PPL run equal to the digit.
* **S3 — attention 3a** (2–3 d): spec + host check (`attn_m1_cases.h` SNR vs
  the fp16 spec, gemma64 E2E ≥ 30 dB), rung 2 / 3; device C = 3a, the +2 %
  PPL rule + text on the real file, user approval.
* **S4 — MoE 4-slab** (2 d): rung 1–3; device D = S4, MoE dumps `bit_identical=1`.
* **S5 — attention 3b** (4–5 d) after S3's reading: IDL + stub + registration
  + fixture, its own sitting. **S6 — lever 5** last.

## 5. Risks

DVFS: `mhz` on the per-kind line tells cycles from clock (P4: 1993 / 2094);
the table carries it. Thermal: B follows A per G block, cool start (plan 260
§7.3). Stale skel: S1 / S4 change DSP sources (rung 2 both arches, md5 in
the table); S2 does not. Address space: S4 adds no heap (VTCM 3.96 of 8
MiB), 3a nothing, 3b halves the KV heap. DMA rate: lever 4 assumes the expert
feed reaches the FC path's 52 GB/s — one G 64 `NNTR_HTP_PROFILE=2` MoE stage
table reads it first. The router's 100 cycles / step is a silicon reading
(rule 58): S1's target is read only on the device. Real-file routing ≠ the
dummy's — lever 4's net is re-read in S0.

## 6. Docs to update

BENCHMARK.md Gemma goal row: lever cells B / C / D beside the E sitting's E.
LEDGER: ㉘'s router / outside-the-wall candidates → rows 1 / 2 / 5; ㉞ b / c
measured (1.27 + 1.1 ms); ㉗ gains the Gemma slope (17 µs / pos); a rule if
S1 confirms "heap f32 weights without `l2fetch` read one line per step".
Plan 229 §8.1's table gets rows 1–5; contract §1 Speed: ≈ 27–30 reachable.
