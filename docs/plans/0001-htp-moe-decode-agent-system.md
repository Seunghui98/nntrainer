# 0001 — Agent system and contract for the HTP MoE decode project

Status: agreed 2026-09-21 (grill-me session, 4 rounds after the PR head
moved). This is the ground truth for how the `htp_moe` work is organised.
Every agent definition under `.claude/` links here instead of restating it.
It supersedes `docs/plans/0000-agent-system-and-env.md` on the `hvx_impl`
branch, which stays there, frozen, as the record of the Qwen3 HVX line.

## 1. Goal

Make LFM2.5-8B-A1B **decode** on the Hexagon HTP of a Galaxy S25 Ultra
(Snapdragon 8 Elite, v79) at least **2× faster than the upstream PR runs it
today, and faster than the same phone's CPU**.

| item | value | source |
|---|---|---|
| Model | LFM2.5-8B-A1B: 24 layers = 18 conv + 6 attention, 22 MoE FFN layers (32 experts, top-4), hidden 2048, vocab 128000, tied lm_head | `config.json` of `LiquidAI/LFM2.5-8B-A1B` |
| "Now", NPU (MoE FFN on HTP, `htp_moe` head with the M=1 GEMV default since PR #108, the D192 loop/lead default since PR #115, the **VTCM DMA feed default since PR #118**, the **dspqueue MoE transport default since PR #151** and the **DSP L2 bypass on the MoE weight DMA, default since PR #159 / `21169f6e`**) | decode **53.97 / 52.16 / 51.41 tok/s** at gen 64 / 512 / 1024 on a **cool start** (each G block at zone0 ≤ 35 °C), prefill 426–580 tok/s (means 528 / 534 / 503) — **record sitting 2026-09-30** (`R3CY10WM83Y`, 02:48–02:53 KST, `htp_moe` @ `90d88e2b`, nothing set, A twice per G, `record-2026-09-30-cool-a.md`): the mirrored cool control that LEDGER rule 52 asked for; r1 / r2 53.51 / 54.42, 54.75 / 49.56, 51.54 / 51.29, text r1 = r2. **Warm-start reading of the same default, the previous "now": 51.82 / 50.61 / 47.36** (#158 B; cool over warm +4.1 / +3.1 / +8.6 %) — a warm G=1024 still reads 47–49 (rule 52). Until the record sitting: decode **51.82 / 50.61 / 47.36 tok/s** at gen 64 / 512 / 1024, prefill 405–509 tok/s (means, temperature-bound; A of the same sitting 403–507) — **#158 B, cycle 22** (`R3CY10WM83Y`, 2026-09-29, `158-dma-bypass.md`, set from `0106622f`): `src_bypass=1` on the arena expert weight DMA, against its own A (the L2 path) 38.61 / 37.40 / 36.17 → **+34.2 / +35.3 / +30.9 %**, MoE dumps `bit_identical=1`, decode nll lines equal, text ≡ A on 8 prompts; made the default by the user (`21169f6e`, unset = `applied=0x703e1 … dma_bypass=1`) — the device check of the unset banner is done (cycle 23: every A log of 2026-09-29/30 prints `applied=0x703e1 … dma_bypass=1 source=default` with nothing set). **Cycle 23 (2026-09-29/30): unchanged**; the same unit's cool-start A cells that day read 52–55 / 51–54 / 50–52 (G=1024: 52.36, 50.90, 51.26, 50.86, 51.59, 51.40) and a hot sitting 47.88 / 49.50 / 45.86, so the G=1024 ≥ 50 question is temperature-bound and moves only with a mirrored cool control sitting (LEDGER rule 52). Moved without a control sitting on #141 Q's grounds (bit-identical, flip = unset value only). Until cycle 22: decode **39.40 / 37.77 / 36.53** (G=1024 = #90 A, 2026-09-29, the first G=1024 cell on the dspq default); prefill 460–553 tok/s (means) — **#141 Q, cycle 21** (`R3CY10WM83Y`, 2026-09-28, `4049dd15`): the 22 M==1 MoE calls through dspqueue, measured by env on the committed `07fb1938` against its own FastRPC A (36.78 / 35.98 → **+7.1 / +5.0 %**), MoE dumps `bit_identical=1`, text ≡ A on 8 prompts; made the default by the user (PR #151 `08de92b1`, flip `c73384d2`, device banner checked). It moves the "now" without a control sitting because it is bit-identical to A and the flip changes only the unset value of the switch (BENCHMARK Method, cycle 21); the next sitting's A (`dspq: on` once) fills G=1024. Until cycle 21: decode **35.97 / 35.96 / 35.17 tok/s** at gen 64 / 512 / 1024, prefill 438–502 tok/s (prompt 512; zone0 53–61 °C) | #141 Q (cycle 21, `141-dspq-moe.md` @ `4049dd15`); until then #120 variant A, `R3CY10WM83Y`, 2026-09-23 (`7ded3ce9`): `htp_moe` @ `d79c0efe` with nothing set (`applied=0x303e1 … feed=vtcm source=default`) — the first sitting whose control is the feed default, so it replaces #117 A as BENCHMARK.md's Method required (cycle 16); its A0 (`FEED=0`) 28.62 at G=64 puts the feed at **+25.7 %** as a control; its B (the upstream PR #4327 sync, PR #121) 37.68 / 36.29 / 35.10 (+4.7 / +0.9 / −0.2 %), prefill +12.4 / +10.2 / +6.2 %, text identical, M==1 `dsp` 711.7 vs 711.3. Earlier "now": **28.22 / 27.89 / 27.23** — #117 variant A, second (cooled) sitting, `R3CY10WM83Y`, 2026-09-23 (`6412bfc9`) — the first sitting whose control is the D192 default, so it replaces #100 A (27.85 / 27.09 / 26.45, `R3CY205ZMND`, `4bf1fdac`) as rule 33 required; cross-unit it sits +1.3 / +3.0 / +3.0 % above #100 A, inside the 6–8.6 % unit band and not a reading. Its same-sitting A0 (pre-#115 cell) 27.56 / 26.78 / 26.20 puts D192 at **+2.41 / +4.16 / +3.91 %** (#113 confirmed). **Same sitting, variant B — the VTCM DMA feed (PR #118, #117): 37.38 / 36.49 / 35.01 (+32.5 / +30.9 / +28.6 %)**, text identical 36/36 over two sittings; it became the "now" in cycle 16 through #120's A (above). Earlier: #105 A 27.74 / 26.88 / 24.83 (`da41340b`), #88 B 24.50 / 23.80 / 23.52 (`3f9fa38d`), #94 sitting 2's 17.72 / 17.03 / 17.30 (`dad0f476`), the PR author's 20.8 / 523 (doc 49 §1, prompt 444) |
| "Now", CPU (all Q4_0, 8 threads) | decode **52.18 / 51.31 / 50.12 tok/s**, prefill 280–348 tok/s on `R3CY10WM83Y` (record sitting 2026-09-30, cool start per G, `record-2026-09-30-cool-a.md`) — the same unit and protocol as the NPU "now", which is **+3.4 / +1.7 / +2.6 %** above it; other unit (`R3CY205ZMND`): 52.43 / 49.22 / 48.31, prefill 268–340 | record sitting 2026-09-30; before it #94 sitting 2 (#88 had no CPU cell); replaces the PR author's 48 / 334 |
| Levers measured | (1) M=1 HVX GEMV (PR #86, switch on): **18.83 / 18.33 / 17.59** (+6.2 / +7.7 / +1.7 % vs 17.72 / 17.03 / 17.30), text identical — #94 sitting 2 variant C (LEDGER ⑯); (2) wall 3, size-class staging + 5 ms poll (PR #103, merged): **+30.9 / +41.4 / +42.9 %** vs its own A 18.72 / 16.83 / 16.46, transport 401 → 88 µs/call, text identical — #88 (LEDGER ⑦, closed); (1) + (2) together = #105's A above (not yet read against GEMV off in one sitting, #101 / PR #108); (3) GEMV compute side (PR #107 → PR #115, one-row loop + `l2fetch` lead): #105's gate failed (best B3 `mm` −4.3 %); #113 measured the whole (loop × lead) matrix — the lead harms the four-row loop 2× and helps only the one-row loop, to 192 KB (LEDGER rule 31); best cell **D192** `mm` **937.0 (−6.3 %)**, gate 840 missed, decode **+4.11 / +3.29 / +4.26 %** vs its A 27.44 / 26.83 / 25.93, text identical, prefill within 0.08 % on M>1 `dsp`. **Landed as the default by user decision 2026-09-23 (rule 33; PR #115 merged `c45d4433`, cycle 13)**; "now" moved to it in cycle 15 (#117's A, above); (4) **M=1 GEMV feed from VTCM by DMA** (PR #118, #117: each expert's `wh_bytes` staged one expert ahead at the certified `f2` shape, one-row loop reads VTCM): M==1 `mm` **931.9 → 676.4 µs (−27 %)**, gate ≤ 760 **passed twice**, in-situ engine 32.3–34.4 GB/s, transport 190 → 92 µs/call (LEDGER rule 35), decode **+32.5 / +30.9 / +28.6 %** (and +25.2 / +35.7 / +23.2 in the warm sitting), text identical 36/36, M>1 `dsp` ±0.04 %, `bit_identical yes` — on `R3CY10WM83Y`, whose DMA anchor reads 37.3 GB/s vs `R3CY205ZMND`'s 31.2 (rule 34: there the `mm` would read ≈ 805, the decode win ≈ +10 %). **Default since PR #118 (`d79c0efe`, cycle 15); confirmed as a sitting's control by #120 A / A0: +25.7 % at G=64, `mm` 677.2 vs 934.3 (cycle 16)** | #94 s2 `dad0f476`; #88 `3f9fa38d`; #105 `da41340b`; #113 `d03c8929`; #117 `6412bfc9` |
| **Goal** | decode **≥ 50 tok/s** at every measured generation length (**record sitting 2026-09-30: met at all three G on a cool start — 53.97 / 52.16 / 51.41 (`R3CY10WM83Y`, `90d88e2b`, nothing set; G=1024 51.54 / 51.29); the warm-start G=1024 of the same default still reads 47–49 (rule 52), so the goal holds under the stated cool-start condition; **"above the CPU" read on this unit the same night: CPU `q40` 52.18 / 51.31 / 50.12 under the same cool protocol → NPU +3.4 / +1.7 / +2.6 %, met, narrowly** (G=512 margin inside one run's spread). Cycle 23: row of record unchanged — met at gen 64 / 512 (51.82 / 50.61), 1.056× short at gen 1024 (47.36, #158 B); the day's cool A cells read ≥ 50 at every G (G=1024 50.9–52.4 in six sittings) and < 50 hot (45.9–49.1) — temperature-bound, a mirrored cool sitting decides (LEDGER rule 52); #162's two CPU-side levers void (rule 51); the end-to-end track (PR #169) has RMSNORM / QK_NORM (#164) and ATTN_M1 (#170) bit-identical on silicon, ATTN_M1 at 250 k / 374 k pcyc/op vs gates 210 k / 350 k, and the FC set as the open decision (D of #132, #178: two-session path projects ≈ 45–46 tok/s); "above the CPU" not yet read on this unit**; cycle 22: met at gen 64 / 512 on `R3CY10WM83Y` (51.82 / 50.61), 1.056× short at gen 1024 (50 / 47.36), #158 B — #162; earlier 1.32× the NPU "now" at gen 512 — 50 / 37.77, #141 Q; 1.39× on #120 A, 1.79× on #117 A; above the CPU) | user decision Q1 (ii) |
| Physical ceiling | **Corrected in cycle 21 (LEDGER rule 42): the NPU path reads ≈ 886 MB/token (MoE `QS4CX_WH` 484 + Q4_0 FCs 255 + lm_head 147) and the CPU path ≈ 947; the CPU's 52.4 tok/s is ≈ 50 GB/s effective (its single-reader rate is 67.9, LEDGER §2 ④), so it does not sit on a 38 GB/s ceiling. On the NPU path with the MoE on the DSP at its in-app 33 GB/s (rule 41, 14.7 ms) and the 402 MB rest on the CPU at 38–68 GB/s (5.9–10.6 ms): 20.6–25.3 ms ⇒ 40–49 tok/s.** Original: 730 MB of weights per token ÷ 34–38 GB/s measured DDR rate = 19–21 ms ⇒ **48–52 tok/s**. The CPU already sits on it. **Per unit** (LEDGER rule 34): the MoE weights alone (484 MB/token) cost 13.0 ms at `R3CY10WM83Y`'s 37.3 GB/s DMA bound and 15.5 at `R3CY205ZMND`'s 31.2; with the ≈ 6.5 ms byte floor of the rest that is 45–51 tok/s | doc 48 §1; #117 cycle-15 budget |
| Prefill gate | never below **−5 % of the current NPU prefill** (≈ 523–532 tok/s, i.e. ≥ 497) | user decision Q12 |
| Accuracy gate | doc 45 §3.4, all three: (a) kernel bit-identical to its scalar spec, (b) real-model diff (`NNTR_L2_DIFF`) = 0, (c) generated text identical to the CPU run of the same weights. **Amended 2026-09-28 for the ⑨ handoffs (per-token entry, #130 and its successors):** (c) is read as two columns together — text identical to the CPU `q40` run (y/n, first differing token) **and** the decode-side PPL (`NNTR_PPL_DECODE`, #134) of every variant forced on variant A's own greedy continuation, **read against A — the switch-off run of the same NPU model in the same sitting** (amended again 2026-09-28: not the CPU `q40` run, whose different weights alone put it ≈ 4.9 % away, LEDGER ⑱) — **plus a user approval step** (the text column is never dropped in favour of the PPL: both are read, every time): the filled handoff pastes every variant's generated text next to A's, the user reads them and marks `text approved: y/n` per variant; a variant whose text differs is neither pass nor fail until the user approves it, and the supervisor folds only approved rows. A decode PPL above A's by more than the plan's threshold (default +2 %) fails regardless of approval; the CPU `q40` PPL, when a sitting has it, is recorded as information about the NPU model's quantization (#110), never as this gate's reference | user decision Q4; amended 2026-09-28 |

The "now" numbers were replaced on 2026-09-22 (cycle 5) by #94 sitting 2,
the first sitting with all 12 control cells on one binary set (§4.3's
intent; #77 had measured the same cells a day earlier on the same unit:
NPU 18.2–21.3, CPU 46.4–54.1, prefill 403–541 — all in BENCHMARK.md).
On the same day (cycle 7) #88's variant B replaced the NPU row: it is the
code of `htp_moe` @ `ad714de7` measured against its own same-sitting A
(same unit); the CPU row stays #94 sitting 2's, since #88 ran no CPU
cell. In cycle 9 #105's variant A replaced it again: the same `htp_moe`
code with the M=1 GEMV switched on, which PR #108 makes the default. That
variant was the control of its own sitting, not a lever under test. In
cycle 11 #100's variant A replaced it once more — the same code with the
GEMV now on by default (PR #112 is test-only), reproducing #105 A within
+0.4 / +0.8 / +6.5 % on the same unit. In cycle 15 #117's variant A
(second sitting, `R3CY10WM83Y` — the second unit) replaced it: the same
code with the D192 default of PR #115, measured as its own sitting's
control, and its variant B (the VTCM feed) became the next default (PR #118, `d79c0efe`). In cycle 16 #120's
variant A replaced it: the same code with the feed on by default,
measured as its own sitting's control on the same unit (35.97 / 35.96 /
35.17, reproducing #117 B within −3.8 / −1.5 / +0.5 %). In cycle 21
#141's variant Q replaced it (39.40 / 37.77 at G 64 / 512; G=1024 not
re-read): the dspqueue transport, bit-identical to its same-sitting A and
made the default by PR #151 — the first "now" taken from a lever cell,
because a bit-identical change leaves a control sitting nothing to
confirm but the drift the A/B already controls. In cycle 22 #158's variant
B replaced it on the same grounds (51.82 / 50.61 / 47.36): the DSP L2
bypass on the MoE weight DMA, bit-identical to its same-sitting A, made
the default by `21169f6e` (PR #159); the unset-banner device check is
still pending. They are read against the sitting they came from: one unit drifts up to
±9 % (CPU) / −16 % (NPU tok/s) between sittings while its DSP profile
columns move ≤ 4 % (LEDGER rules 9, 20), and two S25 Ultra units differed
by 6–8.6 % on one binary in `hvx_impl`. Hence every verdict is a
same-sitting A/B (§4.2) and no unit is named (decision 2026-09-22).

2× the CPU (96 tok/s) is **not** a goal: it needs 70 GB/s from a 38 GB/s
memory system. Anything that raises the ceiling itself (fewer bytes per
token, two DDR readers) is a separate track (§3.3).

### 1.1 Measurement definition

Prompt fixed at 512 tokens (the PR used 444; we fix 512 so the prefill
column is comparable across handoffs). Generation length **64 / 512 /
1024** tokens. Report, per run: prefill tok/s, decode tok/s as the median
over the whole generation and separately over the last 64 tokens, peak RSS,
and the accuracy columns. `NNTR_NUM_THREADS=8` for both CPU and NPU runs
(doc 49 §1 measured both that way). Never read tok/s from a `--profile`
build (it inflates prefill by 83 %, doc 46 §43.4).

## 2. Facts we build on (do not re-derive)

From the PR's own device work (`docs/htp_attention/44`–`50`):

* **Why NPU decode is slow — three walls, all three must fall** (doc 48 §3).
  Per MoE call at M=1: host 1.91 ms = DSP 1.35 + transport 0.57.
  1. HMX computes 64-row tiles for 1 row: 1.03 ms of the 1.35. Fix: an HVX
     GEMV for M ≤ 6 reading the WH tiles in VTCM directly (int32 sums are
     order-independent, so it can be gated bit-identical to the HMX path).
  2. Weight DMA runs at 18 GB/s where the arena probe reached 38. Cause
     unknown (descriptor shape / engine count / DVFS) — measurement C of
     doc 48 §5 decides.
  3. FastRPC transport 0.57 ms × 22 layers = 12.5 ms/token. Fix: dspqueue
     or a resident DSP worker; measurement B of doc 48 §5 decides which.
  Fixing all three gives MoE ≈ 15 ms/token; the CPU's MoE share is 19.
* **Then the ARM remainder** (FC, attention, norm, lm_head ≈ 19 ms/token,
  not yet measured — measurement A of doc 48 §5) must shrink to a few ms,
  which means one FastRPC call per token with the whole layer stack
  resident on the DSP (doc 45 Phase E), M=1 shape first (§3.1).
* **Wall 2 is not a descriptor-list problem** (#100, 2026-09-23): against
  the tag-validated per-call ceiling `c_star` the traced 46-descriptor
  M=1 list runs at 1.19×, i.e. *faster*. Wall 2's "4–7×" (rule 11) and
  "2.7×" (rule 19) were ratios against an isolated `DMA_PROBE` that no
  validated path reproduces (LEDGER rule 28). What remains of wall 2 at
  M=1 is the read rate itself — DMA ≈ 31.6 GB/s, the GEMV's direct HVX
  arena read 21–27 — and that read is bound by DDR *latency*, not
  bandwidth (rule 26). So the lever there is the `l2fetch` lead, not a
  better descriptor shape.
* **Moving an FC to the HTP is not a lever by itself**: conv in_proj on
  the HTP saved 13 ms of 848 in prefill because the round trip (4.9 ms)
  is larger than the HMX work (2.0) (doc 50 §3.4). Removing round trips is
  the lever.
* **Weights already live in DDR the DSP can see**: all 1408 expert
  weights (3.7 GiB, `QS4CX_WH` = per-channel int4 pre-baked into HMX tile
  order by the quantizer) sit in an ION arena of 256 MiB chunks, 3840 MiB
  mapped, 0.09 ms per registration; DSP heap ≈ 182 MiB remains (doc 46
  §41, §48). There is no residency wall left for the MoE weights; the
  32-bit DSP address space is the limit for anything else we add.
* **Two weight formats, two model files** (doc 46 §35, §48.7):
  `QS4CX_WH` has **no CPU fallback** (`moe_htp_layers` must be empty, a
  wrong layout yields plausible wrong text, not an error); plain `Q4_0`
  is the CPU control. `NNTR_MOE_HTP_DECODE=1` is a measurement-only switch
  on the non-WH `QS4CX` model.
* **Runtime switches** live in `nntr_config.json`: `moe_engine`,
  `moe_htp_layers`, `conv_in_proj_engine`, `conv_out_proj_engine`,
  `attn_proj_engine`, `dense_ffn_engine` and their `*_htp_layers` lists.
  Instrumentation: `NNTR_HTP_PROFILE=2|3` (per-stage DSP breakdown; 3 =
  5× repeat, min), `NNTR_M0_PROFILE=1` (ARM side of the MoE layer),
  `--profile` build (per-node TYPE totals).
* **Traps** (doc 46 §48.7): rebuild the skel (`test/htp/build.sh`) whenever
  the IDL or DSP sources change — `build_android.sh` never does, and a
  stale skel fails with `AEE_EBADPARM (0x8000040E)`; `build_android.sh
  --clean` silently drops `-Denable-htp`; never push the link-time stub
  `libcdsprpc.so` to the phone; read `min`, not `avg`, from profiles.

## 3. Approach and order

### 3.1 Decode first (user decision Q3)

Doc 45's phase order (A MoE → B conv → C attention → D orchestration →
E decode) is prefill-centric. We invert it:

1. **Measure before touching code**: doc 48 §5's A / B / C plus the
   two-reader DDR probe, in the first handoff (§4.3).
2. **The three walls**, in the order the measurements dictate (DMA path,
   M=1 HVX GEMV, transport).
3. **One FastRPC call per token**: M=1 kernels for the rest of the layer
   (RMSNorm, conv1d + gating, RoPE, attention at M=1, dense FFN, lm_head)
   and the per-token orchestration entry. M=1 kernels are HVX, not HMX,
   and small; they may coexist with the prefill kernels as a second path
   (llama.cpp does the same).
4. Prefill-shape residency (doc 45 B/C/D) afterwards.

Every step carries the prefill gate (§1) because the DMA ring, the worker
pool and the weight layout are shared with the prefill path.

### 3.2 NPU-only first (user decision Q10)

Decode runs entirely on the DSP until the NPU-only ceiling is reached.
CPU+NPU expert splitting (the CPU takes one of the four active experts
per layer while the DSP takes three, synchronised once per layer) is a
**later** track. Its only justification is a two-reader DDR bandwidth
above one reader's 38 GB/s, which the first handoff's probe measures.

**Open decision (Q11, user, later):** the rule for when NPU-only stops
and the split begins. Proposed, not decided: after the three walls and the
one-call-per-token step, if the measured decode is < 50 tok/s **and** the
two-reader probe exceeded 45 GB/s.

### 3.3 Separate tracks, filed as issues when the time comes

Raising the ceiling (fewer bytes per token), prefill residency, CPU+NPU
split, upstreaming.

## 4. Machines and the measurement loop

### 4.1 Workstation (all agent work, all build gates)

Ubuntu, 8 cores, 30 GB RAM, 822 GB free under `/`. `source tools/htp/env.sh`
sets everything below; agents put that line first in every shell.

| thing | where | note |
|---|---|---|
| Hexagon SDK | `/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1` (toolv19, hexagon-clang 19.0.04) | the PR was developed on 6.4.0.2; 6.3.0.0 is also installed and not used |
| HexKL 1.0 beta.2 | `~/Qualcomm/hexkl-1.0-beta.2/hexkl_addon` (`HEXKL_ROOT`), `HEXKL_SDK_VER=6.4.0.1`: `lib/6.4.0.1/hexagon_toolv19_v79/libhexkl_micro.a` for the skel, `lib/6.4.0.1/armv8_android26/libsdkl.so` for the app | `~/Qualcomm/hexkl_addon` is the `hvx_impl`-style symlink view of the same package; `test/htp/build.sh` and `build_android.sh --htp` take the package root |
| Android NDK | `~/android-ndk-r30` (`ANDROID_NDK`; `.bashrc` exports it for interactive shells only) | the PR expects r26d; r30 compatibility is verified by the first app build (§9 rung 3) |
| Host build | `build/` (`meson setup build -Denable-transformer=true -Denable-tflite-backbone=false -Denable-tflite-interpreter=false`; no `flatc` here) | host gtests, `nntr_quantize_stream`, `run_host_checks.sh` |
| Models | `/local/mnt/workspace/models/lfm2.5-8b-a1b/{hf,fp32,q40,q40-qs4cx-wh}` (§11) | `NNTR_MODEL_DIR` |
| clang-format-14 | `~/.local/bin/clang-format-14` | AGENTS.md rule, changed lines only |
| adb | `/usr/bin/adb` | **Since 2026-10-02 no device is attached and agents do not run adb** (§12, 2026-10-02 row): a task that needs silicon numbers files its handoff as an issue / issue comment (`needs-user` + `state:needs-measurement`, full path and commands) and the user runs it from a device-farm session. Between 2026-09-30 and 2026-10-01 agents ran the sittings themselves (`adb -s <serial>`, serial named by the issue or the user, handoff doc as the record, text approval the user's); that row applies again only when the user attaches a unit and says so. **Gemma (2026-10-02, user):** the Gemma sittings (#201 S5 / S6) run on a Galaxy S26 Ultra attached to this workstation via adb (§4.2, §12 row); who runs that adb (agent or user) is not yet decided |

**No Hexagon simulator in this project** (user decision 2026-09-21). Kernel
correctness is decided by host scalar specs and bit-identity checks (gate 1)
and by the device (gate 4); nothing in between.

### 4.2 Device (user only)

Galaxy S25 Ultra — **any unit** (user decision 2026-09-22; two units,
`R3CY205ZMND` and `R3CY10WM83Y`, exist and the PR author used
`R5KL30G6MLT`). A handoff never names a serial; the filled handoff
records the serial that ran (`adb devices`), BENCHMARK.md tags every row
with it, and results are compared only inside one sitting (A/B); when two
sittings happen to share a unit the same-unit drift is noted (LEDGER
rule 13). A task that needs silicon numbers ends in a **measurement
handoff** (`docs/measurements/<issue#>-<slug>.md`, template in
`.claude/skills/hexagon-handoff`) run in one sitting — by the user from a
device-farm session since 2026-10-02 (the handoff is also filed as an issue
or issue comment with `needs-user` + `state:needs-measurement`, §12), by the
agent on `htp_decode` between 2026-09-30 and 2026-10-01 (§4.1 adb row), by
the user before that. Since 2026-10-01 the unit is a Galaxy S26 Ultra (v81
skel, #204) when a product unit exists; the S25 rows stay that device's
column. Its serial is given by the user when attached.

**Device scope since 2026-10-02 (user):** two routes by model.
**LFM2.5-8B-A1B** keeps the S25 / device-farm handoff route of the
2026-10-02 row (the user runs the filled handoff from a farm session);
#222's config-of-record sitting is the closing LFM2.5 sitting, after which
the LFM2.5 table is closed out. **Gemma (#201 S5 / S6)** sittings run on a
Galaxy S26 Ultra attached to this workstation via adb; whether an agent or
the user drives that adb is the user's call (the 2026-10-01 row allowed
agents on `R5KL20NFRCK`; it is not yet extended to this unit). Results are
read inside one sitting as before; the S26 Ultra is a new BENCHMARK column,
not a continuation of the S25 one.

Handoff rules (user decisions Q3, Q13, Q18, Q19):

* Every handoff is a **full-model end-to-end inference** command set
  (`nntrainer_causallm` on the phone with the real 8B model), never a
  kernel microbenchmark alone. Microbenchmarks may be added next to it.
* **Control run first, same sitting**: the unchanged reference binary
  (variant A) runs before any variant, and every result is read as an A/B
  inside that sitting (the PR's decode moved 20.8 → 16.5 between sessions
  with no code cause found, doc 50 §3.4; `hvx_impl` #53 measured ±5 %).
* At most 4 variants per handoff; prebuilt artifacts with md5 and commit;
  expected log lines; an empty result table with the reference numbers
  in it; estimated minutes at the top. Measurements happen on demand.
* Accuracy columns (text identical to CPU y/n, `NNTR_L2_DIFF`) are
  mandatory even on a speed sweep.

### 4.3 First handoff (issue: the tracker's first child)

1. A control: PR head unchanged, CPU `Q4_0` and NPU `QS4CX_WH`, prompt
   512, generation 64 / 512 / 1024, each twice → replaces the provisional
   "now" (§1).
2. Doc 48 §5's A (`--profile` decode breakdown of the ARM remainder), B
   (`NNTR_HTP_PROFILE=3` transport floor), C (arena DMA probe: linear vs
   2D descriptors, 1–4 engines, bus vote).
3. The two-reader DDR probe: CPU and DSP streaming disjoint buffers at
   once, aggregate GB/s.

## 5. Repository, branches, upstream

* Fork `dlwlzzero/nntrainer`. Integration branch **`htp_moe`**, created from
  upstream PR nntrainer/nntrainer#4327's head `2ce38d65` (branch
  `claude/htp-lfm2-moe-ffn` on `Seunghui98/nntrainer`, 2026-09-21) and
  **frozen there**. The supervisor reports new commits on the PR each
  cycle; merging them is the user's decision (Q16).
* Work branches `htp/<issue#>-<slug>`, PRs into `htp_moe`, linear history
  (rebase, never merge commits — the static check rejects empty bodies).
* `hvx_impl` is frozen: no cycles, no deletion. Its issues are closed or
  labelled out of the queue (§7).
* **Upstream-shaped** (Q14): follow AGENTS.md (DCO sign-off, `[component]`
  subjects, clang-format-14, no `subprojects/` edits, cross-platform),
  keep the PR's file structure (`nntrainer/tensor/htp_backend/**`,
  `test/htp/**`, `Applications/CausalLM/**`), and keep **kernel/app
  commits separate from agent-system commits** (`.claude/**`,
  `docs/plans/**`, `docs/measurements/**`, `docs/htp_moe/**`) so the former
  can be cherry-picked upstream.
* Everything is written in **English** (Q17): plans, handoffs, benchmark,
  agent files, commit messages, code comments. The PR's own
  `docs/htp_attention/*` stay as they are (mostly Korean) and are read,
  not edited.
* Commits: `git commit -s` in the user's name plus
  the `Co-Authored-By:` trailer of the model that wrote the commit (the
  implementer and guide writer run on Opus 5.5:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`); one topic per
  commit; body ≥ 8 words; new `.c/.h/.cpp/.py` files carry a doxygen
  `@file`/`@brief` header.
* GitHub writes go through the claude.ai GitHub MCP or the `gh` CLI
  (logged in as the owner); both are acceptable.

## 6. Roles

One Claude Code session on the workstation is the orchestrator; it runs
`/hexagon-cycle`, which spawns the roles below as subagents. State lives
in issue labels; agents talk through issues, plans, handoffs and PRs.

| Role | May write | Must not |
|---|---|---|
| `hexagon-supervisor` | issues and labels, `docs/htp_moe/BENCHMARK.md`, `docs/htp_moe/LEDGER.md` | commit source code |
| `hexagon-planner` | `docs/plans/<n>-<slug>.md`; may run host builds and host checks for exploration | edit source |
| `hexagon-implementer` | source on a `htp/<issue#>-<slug>` branch, handoff docs, PRs into `htp_moe` | push to `htp_moe`, force-push, edit `.github/workflows` |
| `hexagon-guide-writer` | `docs/htp_moe/guide/*.html` (English, self-contained, **decode only**) | anything else |

Supervision scope: `nntrainer/tensor/htp_backend/**`, `nntrainer/htp_context.cpp`,
`test/htp/**`, `Applications/CausalLM/models/lfm2*/**`,
`Applications/CausalLM/layers/**` (only where the LFM2 decode path passes),
`tools/htp/**`, `tools/htp_*`, `docs/htp_moe/**`, `docs/plans/**`,
`docs/measurements/**`. Anything in nntrainer core outside these paths
(`nntrainer/tensor/float_tensor.cpp`'s dispatch, `compute_ops.h`, the
graph executor) is allowed **when the plan names it**, because the
one-call-per-token step cannot avoid it, and is otherwise an issue.

## 7. Issue state machine

Labels: every issue carries `hexagon`, one `state:*`, one `prio:*`.

```
state:needs-plan → state:planned → state:in-progress → state:review → (closed)
                                        │
                                        ├→ state:needs-measurement → state:measured → state:in-progress
                                        │
                                        └→ needs-user  (a decision or action only the user can take)
```

At most one `state:in-progress` issue. `state:measured` is processed
first. A merged PR closes its issue; the user merges.

Queue reset (Q6, Q12 of the session): one tracker issue for this project;
the `hvx_impl` issues #65 #69 #70 #71 (the old port plan) are closed as
superseded by the tracker; the Qwen3-only issues (#26 #41 #57 #58 #59 #60,
#37 #38 #40, #51) are closed as "frozen with hvx_impl" and stay
searchable. The supervisor lists only open `hexagon` issues, so nothing
else needs a new label.

## 8. One cycle (`/hexagon-cycle`)

1. Supervisor: process `state:measured` (update BENCHMARK.md rows and
   goal column, LEDGER.md rules, hand the issue back or close it); close
   issues whose PR merged; report new commits on upstream PR #4327; keep
   at least two `needs-plan`/`planned` issues, derived from the goal
   distance, the three walls, and LEDGER.md's open items.
2. Planner: top `state:needs-plan` → plan file → `state:planned`.
3. Implementer: top `state:planned` (or a handed-back `measured`) →
   gates → PR (`state:review`) or handoff (`state:needs-measurement`).
4. Guide writer: on `--guide`, or when a PR merged or a handoff was filled
   since the guide's last commit.
5. The cycle ends with the user's to-do list.

## 9. Verification gates (summary; commands in `.claude/skills/hexagon-gates`)

0. `clang-format-14` on changed C/C++ files.
1. **Host** (seconds to minutes): `ninja -C build`; the cpu-backend
   `*qs4cx*` gtests; `unittest_causallm_models --gtest_filter='*Lfm2Moe*'`
   (FP32 and Q4_0 differential on the tiny fixture);
   `test/htp/host/run_host_checks.sh` (scalar stand-ins for the kernel loop
   structure and worker pool); `tools/htp_syntax_check.sh`.
   A new DSP kernel ships with a scalar spec and a host check that proves
   bit-identity (doc 45 §3.3/§3.4).
2. **DSP skel** (`test/htp/build.sh`, v79, `-Wall -Werror`), md5 recorded.
3. **Android app + device gtest binary** (`Applications/CausalLM/build_android.sh --htp`,
   `ndk-build unittest_hvx_mm_u8i4`), md5 recorded. `readelf -d
   libnntrainer.so` lists `libsdkl.so` and `libcdsprpc.so`.
4. **Device** (user, handoff). Performance verdicts come only from filled
   handoffs.

## 10. Documentation locations

* Contract: this file. Plans: `docs/plans/`. Measurements: `docs/measurements/`.
* Goals and results table: `docs/htp_moe/BENCHMARK.md`.
* Rules learned on silicon, design verdicts, open items: `docs/htp_moe/LEDGER.md`.
* Beginner guide (decode only): `docs/htp_moe/guide/*.html`.
* The PR's design ledger `docs/htp_attention/*` is read-only history.

## 11. Weights

`/local/mnt/workspace/models/lfm2.5-8b-a1b/`:

| dir | content | how |
|---|---|---|
| `hf/` | `LiquidAI/LFM2.5-8B-A1B` (bf16 safetensors, 17.0 GB, 2 shards) | `hf download` |
| `fp32/` | `nntr_lfm2_8b_a1b_fp32.bin` (≈ 31 GB) + `config.json`, tokenizer files, an authored `nntr_config.json` | `Applications/CausalLM/res/lfm2_moe/lfm2-8b-a1b/weight_converter.py --model_path hf/` |
| `q40/` | all-Q4_0 model, the **CPU control** | `nntr_quantize_stream fp32/ -o q40/ --fc_dtype Q4_0 --embd_dtype Q4_0 --lmhead_dtype Q4_0 --isa ARM` |
| `q40-qs4cx-wh/` | FC/embed/lm_head Q4_0, MoE `QS4CX_WH`, the **NPU model** | same plus `--moe_dtype QS4CX_WH`; `nntr_config.json` gets `"moe_engine": "htp"`, `"moe_htp_layers": ""` |

Always `--isa ARM`; the `_ARM.bin` suffix must be present (an x86 packing
runs silently wrong on the phone). md5s of the produced `.bin` files are
recorded in BENCHMARK.md's artifact section once built.

## 12. Decisions log

| date | decision |
|---|---|
| 2026-10-02 | **Gemma device measurements run on a Galaxy S26 Ultra attached to the workstation via adb (user).** The Gemma sittings (#201 S5 / S6: Gemma-4-26B-A4B end to end on the NPU, then its levers) happen on an S26 Ultra connected to the workstation, not through the device farm; the S25 / device-farm handoff route of the row below stays for LFM2.5, whose closing sitting is #222. The S26 Ultra gets its own BENCHMARK column (the S25 column is frozen; the #208 developer-unit appendix is not its baseline). **Open: who runs adb on that S26** — the 2026-10-01 row allowed agents on `R5KL20NFRCK`; until the user says so for this unit, agents do not run adb (§4.1) and Gemma sittings are filed as handoffs |
| 2026-10-02 | **The LFM2.5 table is closed out on the #222 config of record, then the project moves to Gemma (user, on #222).** The proposed `nntr_config.json` is adopted in full (`conv_block_engine`, `dense_ffn_engine`, `attn_proj_engine` = `htp`, `init_seq_len 1024`, the pure fixes), overriding plan 222's "adopt conv_block only"; the PPL cost of `attn_proj` / `dense_ffn` (doc 51: +4.5 / +4.2 %) is accepted and re-read in the closing sitting. Where the engine keys conflict with the decode-side code they are resolved so that they take effect at prefill without breaking either decode path (PR #223: the one-layer `dense_ffn` form now asks `htpDecodeRowResident`; host qs4cx converter fix; decode unchanged on both paths). The closing sitting (`docs/measurements/222-config-refresh.md`, PR #223, S25 via the farm) is being run by the user; the LFM2.5 prefill / decode rows of record move only through its mirrored old-config / new-config A pair, and no LFM2.5 number is folded before PR #223 merges. Same day: PR #218 merged as `238a280b7` — `NNTR_MOE_FADVISE` env-only, default **not flipped** (prefill −7 to −18 %, rule 62); #216 closed |
| 2026-10-02 | **No device on the workstation: agents do not run adb; every device measurement is a filed handoff the user runs from a session on the device farm (user).** The S25 Ultra `R3CY10WM83Y` is disconnected from the workstation. From this date a task that needs silicon numbers ends in a handoff (`docs/measurements/<n>-*.md` on the branch: variants, staged set + md5s, the run script, expected log lines, empty tables) **and** a GitHub issue or an issue comment on the task issue that carries the full handoff path and the commands, labelled `needs-user` + `state:needs-measurement`; the user runs it from another session connected to the device farm, fills the tables, and the supervisor folds them (§4.2 rule set unchanged: control first, A/B inside one sitting, md5 gate, text approval by the user). This replaces the 2026-10-01 "agents run adb" row and the §4.1 adb row for as long as no unit is attached; when a unit is attached again the user says so and the 2026-10-01 row applies. The #216 lever's two sittings (23:06–23:36 KST on 2026-10-01) ran before the disconnect and are folded in cycle 30 |
| 2026-10-01 | **S26 optimization is deferred until a product unit; the two-PD path is deleted (user).** The #208 sitting ran on `R3CY70LV96T`, a developer / engineering S26 (userdebug `S948USQU1AZAB`): the one-PD E2E runs on v81, texts == A 20 / 20, ceiling 3840 — recorded in `204-s26-rebaseline.md`, with its tables as an appendix marked "developer unit, not representative"; no rule, no BENCHMARK column and no "now" come from it (LEDGER cycle 28). The two-PD E2E path (`NNTR_HTP_E2E_PDS` = 2, S2 with the FC set) is to be removed — one PD is the design (rule 59); the removal plan is the next issue |
| 2026-10-01 | **Upstream PR #4327 is no longer watched (user).** The supervisor stops reporting its commits; §Upstream in LEDGER is frozen at `f923bf29`. Same day: PRs #206 (S26 stack port, #204), #202 and #203 (#201 S1: the expert pool inside the per-token E2E entry, one PD) merged into `htp_decode`; #207 (farm S25 sitting) closed as folded; #208 (S26 re-baseline) waits for a farm S26 |
| 2026-10-01 | **The S26 Ultra replaces the S25 for `htp_decode`; agents run adb (user, 2026-09-30 / 10-01).** The S26 stack (#168 ELF-arch guard, #177 `NNTR_MOE_DMA_QUEUES`, #185 DQ schedule; 7 code commits, not C3 / DQR) is ported onto `htp_decode` by #204, so `htp_decode` builds the v81 skel (`HEX_ARCH=v81`, `ARCH OK (V81)`); the queue default stays 1. From #201 on, agents run the device sittings with `adb -s <serial>` (the §4.1 row; this replaces the 2026-09-29 `htp_moe_v81`-only exception), keep the handoff as the record (reboot, cool start, md5s, stop rules), and leave text approval to the user. The S25 rows (`R3CY10WM83Y`) stay that device's column |
| 2026-09-30 | **New base branch `htp_decode` (user, re-plan of the project).** Cut from `htp_first_version` @ `d4a898430` and pushed to `origin`: `htp_moe` + #132 Part B (the two-session E2E) + upstream nntrainer/nntrainer#4383 @ `5a84c05d` (MoE expert streaming from flash, "FSU") + upstream #4343 @ `50ed1916` (fp16 KV attention) + upstream main @ `aad932ce` + plan 194 and the E1 commits of #194. Purpose: review the whole decode-side NPU end-to-end structure (the `NNTR_HTP_E2E` path) with FSU in the tree and optimize it; references named by the user: upstream PR #4296 (Gemma 4 on the CPU), QNN, T-MAN. Wherever §5, §6 and §10 say `htp_moe` as the docs / PR target, read `htp_decode`. The tracks are runtime switches (`NNTR_HTP_E2E`, `NNTR_HTP_PPL_LEVERS`), not branches: `htp_moe_ppl` and `htp_moe_v81` are retired (the L0–L4 levers stay on `htp/194-s3`, the S26 stack on `htp/168` / `177` / `185` / `187-*`, kept as local `archive/*` tags), and lever / S26 work does not continue until re-planned. First item: #201 (`state:needs-plan`, p0), whose first deliverable is a plan (structure review → measured per-stage breakdown → ranked levers). Speed target, accuracy rule, FSU control, prefill gate and device scope for this base are **not decided** — #201 lists them as open questions |
| 2026-09-29 | **Decode NPU end-to-end is the main track (user).** Every decode op moves to the HTP, the per-token entry ends at one call per token (no CPU round trips), and kernels are optimized to reach ≥ 50 tok/s at gen 64 / 512 / 1024, with the bit-preserving rule unchanged (text ≡ the switch-off run; every resident kind bit-identical to the Android CPU path, verified per op with a CPU-vs-HTP shadow on the same input; a passing PPL with a new repetition loop is a failure, LEDGER rule 45). The 2026-09-28 row's "not extended" is lifted for the resident path under this rule. Order: (1) RMSNORM / QK_NORM bit-identical (#164, verified on silicon 2026-09-29: 392/392 rows, 1920/1920 heads, logits == A), (2) the M=1 FCs, router, final norm + lm_head bit-identical (#132 PR 2), (3) one call per token (#132), (4) the bit-identical ATTN_M1 made fast (≈ 1.4 ms/layer on silicon vs ≈ 0.15 needed), (5) the small kernels (norm ≈ 0.4 ms/token on the DSP vs 0.1 on the CPU). The CPU-hybrid default stays the product path until the end-to-end path is faster and bit-identical; hybrid-only levers (#162 prefetch) are secondary. Same day: the user asked for a review (not a build) of splitting the four decode experts two on the CPU / two on the NPU (#157, PR #161 `NNTR_MOE_HTP_SPLIT=k`) |
| 2026-09-29 | **Targets after rule 42 (bytes per token ≈ 886 MB): (a) + (b).** (a) bit-preserving levers as decided on 2026-09-28 (#150 next); (b) CPU + DSP reading weights concurrently to raise the aggregate bandwidth — first measure whether the two readers add up on this phone (#90, raised to p1; HeteroLLM, SOSP 2025, reports one processor 40–45 GB/s and GPU+NPU ≈ 60 GB/s on Snapdragon 8 Gen 3), and under the bit-preserving rule only as overlapped reads (the CPU prefetching the next layer's weights while the DSP runs the MoE), not as split arithmetic. (c) (fewer bytes / non-bit-identical paths) is not opened yet: the user is willing to accept changed text if the outputs stay sensible (no meta sentences, no repetition loops) but wants the accuracy of such paths improved first. Same day: PR #148 (#146, ATTN_M1 O1 + O3) merged; the resident path stays off by default |
| 2026-09-28 | **Direction change (user, after the #134/#132 sitting): accuracy = the generated text identical to the switch-off baseline, and speed must come without changing a single bit of it.** Only bit-preserving levers are pursued: (1) the existing 22 MoE calls through dspqueue (#141 step 2, re-scoped), (2) the MoE feed-engine schedule (LEDGER ㉓), (3) CPU-side threading that keeps every reduction order (LEDGER ⑰), (4) later, a CPU-exact Q4_0 FC on the DSP. Gate for every such change: text identical to A over a multi-prompt set **and** `NNTR_HTP_DUMP` MoE dumps `bit_identical=1` against A (bit-identical dumps make the text identical for any prompt). The per-token resident path (`NNTR_HTP_FORWARD`, #130/#132) stays in the tree, off by default, and is not extended: #132 PR 2 and #146 are parked; its decode PPL (#134) stays as a tool. The 2026-09-28 PPL / approval rule applies only to changes that cannot be bit-identical |
| 2026-09-28 | **Three decisions (cycle 20 close):** (1) PR #143's worker-pool fix is **accepted without its own prefill A/A0** (the pre-#143 set would be a fifth variant after #145's IDL change; #136's post-fix switch-off prefill read 568.9 tok/s, above the 438–502 band). (2) Upstream PR #4327's commits after `4ae1ebd7` (head `bcfc1ac5`, the HVX integer MoE epilogue, prefill-only, default off, IDL +13) are **held, not merged**. (3) Issue #142 (Gemma4 kernels) is **not** part of this project: no `hexagon` label, the supervisor does not track it |
| 2026-09-28 | **Agent models:** `hexagon-implementer` and `hexagon-guide-writer` run on Opus 5.5 (`model: claude-opus-5-5` in `.claude/agents/`); `hexagon-supervisor` and `hexagon-planner` keep the session's model. **Text comparison stays mandatory** in every per-token-entry handoff: the decode PPL adds a column, it never replaces the text column or the user's text approval |
| 2026-09-28 | **Decode-PPL reference = variant A (switch off, same NPU model, same sitting), not the CPU `q40` run.** The CPU model's different weights put its PPL ≈ 4.9 % from the NPU model before any switch is on, which would swamp a +2 % gate; A differs from B/C/D by the switch alone and is bit-identical run to run (#136's dump sitting). Threshold stays +2 %; the text-approval step stays. The CPU `q40` PPL is information for #110 only |
| 2026-09-28 | **Order: #130 PR → #130 device sitting → #132.** The remaining CPU decode ops (M=1 FCs, router + top-k, final norm + lm_head; LEDGER ㉓) become DSP-resident only after #130 is merged and measured; filed as #132 (`state:needs-plan`, p1). hvx_impl's DMA descriptor queue does not reduce the call count (it is weight prefetch, LEDGER §4); only the resident set does |
| 2026-09-28 | **⑨ accuracy column = (a) + (b) + user approval.** For handoffs that run the per-token entry (#130 onward) the accuracy gate keeps the text-identical column (a), adds `NNTR_PPL` vs the CPU `q40` PPL (b), and adds a step where the user reads each variant's generated text and approves it (`text approved: y/n`); differing text is not a fail by itself, an unapproved text is not a pass, a PPL regression fails on its own. Option (c) (CPU NEON `_det` twin) is not taken. Same day: PR #124's Mac container environment reverted (`0feb13cf`); the gates run on the workstation only |
| 2026-09-23 | **A consistent sub-gate decode win with byte-identical text may land as the default** ("1번으로 진행", cycle 12, LEDGER rule 33): #113's D192 (one-row GEMV loop + 192 KB `l2fetch` lead) missed its `mm` ≤ 840 gate at 937.0 but is +4.11 / +3.29 / +4.26 % decode at G 64 / 512 / 1024 with identical text and prefill within 0.08 %, so `HVX_GEMV_M1_ROWS1=1u` and `HVX_GEMV_PF_LEAD_KB=192u` become the defaults on PR #115. Conditions: the win holds at all three G in one sitting, outside the sitting's own A spread, text = A, prefill gate met; the issue's gate still decides whether the issue is done. Same day: PR #107 closed by the user, superseded by PR #115 |
| 2026-09-21 | No simulator in this project: host bit checks + device only |
| 2026-09-22 | **Any device serial.** The same-device requirement (`R3CY10WM83Y` as the anchor unit, per-cell unit ratio) is dropped: any S25 Ultra may run an `htp_moe` handoff, handoffs name no serial, the filled handoff records the serial used, results are compared only inside one sitting (A/B) plus same-unit drift where the unit happens to repeat. The second-unit anchor sitting (#91 → #94 ⑮) is withdrawn, not re-filed. Same day: #94 sitting 2 replaces the provisional "now" (§1); its variant C (M=1 GEMV) is accepted as the LEDGER ⑯ verdict although the device skel was the user's own build of the same sources (rule 14 → rule 21) |
| 2026-09-21 | Base tree = PR #4327 head (Q1 b); model LFM2.5-8B-A1B; goal ≥ 50 tok/s decode (Q1 ii); decode on the NPU is the product condition, CPU decode stays the control (Q2); decode-first order (Q3); NPU-only first, CPU+NPU later (Q10); Q11 rule deferred to the user; prompt 512 + gen 64/512/1024 (Q6'); prefill gate −5 % of NPU now (Q12); control run per sitting (Q13); upstream-shaped commits (Q14); hvx_impl frozen (Q15); htp_moe frozen at the PR head, merges by user (Q16); English + decode-only guide (Q17); first handoff bundles control + doc 48 A/B/C + two-reader probe (Q18); on-demand measurements, ≤ 4 variants (Q19) |
