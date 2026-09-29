# 170 — A fast, bit-identical resident ATTN_M1 (fp16 lanes, one-rounding FMA)

Issue: dlwlzzero/nntrainer#170 (p1, contract §12 step (4), decode NPU
end-to-end main track, bit-preserving). Read against `htp_moe` @ `aaafd0c4`
(2026-09-29). Kernel `hvx/hvx_attn_m1_f32.c` (#81, #146 O1 + O3, #152's CPU
order), spec `nntrainer/tensor/attn_m1_det.h` (#152, bit-identical on silicon:
measurement 152 G0–G2).

## 0. Measured before planning (workstation; no device)

**Where the time goes.** From the #152 sitting (`152/logs/gtest_G1_attn.log`,
`ATTN_M1_PHASE`, lane-summed pcycles, 6 lanes, 16 units = 8 kv heads × 2
q-head pairs):

| pos | warm `dsp_us` | cold `dsp_us` | scores | softmax | PV | append | busy_max / mean lane |
|---|---|---|---|---|---|---|---|
| 511 | 731 | 1099 | 4.53 M (56 %) | 0.48 M (6 %) | 3.08 M (38 %) | 20 k | 1.52 / 1.35 M |
| 1023 | 1737 | 2193 | 10.49 M (54 %) | 0.91 M (5 %) | 7.95 M (41 %) | 23 k | 3.64 / 3.22 M |

* **The fp16 emulation is the cost, not the bytes.** 4096·L fused fp16
  multiply-adds per layer (scores 2048·L, PV 2048·L). At L = 1024 that is
  131 k 32-lane `hvx_fma16_sf` calls in 19.35 M lane-pcycles, i.e.
  **24.6 core pcycles per 32-lane FMA** with 6 lanes. The same loop on the
  v79 ISS, one thread, costs 22.75 cycles. The f32 cache (4 MiB per layer at
  L = 1024) is read at 2.4 GB/s, an order under the DDR rate.
* **DDR fetch** is the cold − warm gap: +26 % at pos 1023, +50 % at pos 511.
* **Threads.** 6 lanes of silicon deliver what one ISS thread does
  (24.6 vs 22.75 cycles per FMA). The vector pipe, not the thread count,
  bounds today's kernel. The imbalance (16 units on 6 lanes) adds 13 %. S1
  (§4 step 2) sweeps 1 / 2 / 4 / 6 lanes on silicon to confirm this.
* **In the model it is worse.** The issue's 1.38 ms is **not** from a
  G = 1024 run: `164/logs/prof_RQ.log` is G = 64 (`generation: 64 tokens`,
  `calls=4928` = 64 × 77). So 2,807,171 pcyc/op is at L ≈ 513–576, which is
  1.2× the cold gtest scaled to that L. #164's speed cells give the rest
  indirectly: Q vs A at G = 1024 is 25.9 vs 45.9 tok/s (+16.8 ms/token), and
  the last 64 tokens are 21.45 vs 41.9. Net of the CPU attention they
  replace (≈ 0.35 ms/layer) and 6 extra calls, that is **≈ 3.0 ms/layer at
  mean L ≈ 1024 and ≈ 4.0 ms/layer at L ≈ 1500**, about 1.3–1.5× the cold
  gtest line.
* **CPU reference** (same phone): `mha_core` E2E 1.68–1.87 ms/token for 6
  layers at G = 1024, i.e. ≈ 0.28–0.31 ms/layer. Microbench `MhaM1Phases`
  as_is: 140.6 µs at L = 1024 (kcache 88, softmax 28, vcache 23), 216.6 µs
  at L = 1536 (`162/logs0/mha.log`).

**Why an HVX FMA costs 23 cycles.** `hexagon-clang 19.0.04 -mv79` lowers
*every* IEEE `sf`/`hf` intrinsic to a `qf32`/`qf16` instruction plus a
convert. For example, `Q6_Vsf_vadd_VsfVsf` becomes `v.qf32 = vadd(sf,sf); v.sf
= v.qf32`, and `Q6_Wsf_vcvt_Vhf` becomes a multiply by 1.0 (checked with `-S`).
Without `-mhvx-ieee-fp`, which the skel does not pass, it also **contracts
`Q6_Vsf_vmpy` → `Q6_Vsf_vadd` into a `qf32` multiply-add with no sf rounding
between them**. Today's kernel is safe from this, because fp16 × fp16 is
exact in f32 and `div16` corrects itself. Any new sf chain with inexact
products is not. `hvx_fma16_sf`'s TwoSum + round-to-odd + `rne16` is about
20 such pairs.

**What the ISS says about exact alternatives** (exploration only; contract
§4.1 has no simulator rung, so nothing below is a gate). Probe sources and
data are in the planner's scratchpad and are reproduced by §4 step 1.

| candidate per 64 fp16 lanes | = the CPU's `fmla .8h`? | ISS cycles / 64-lane FMA |
|---|---|---|
| today: `hvx_fma16_sf` on 32 f32 lanes, ×2 | yes (silicon, #152) | 45.5 |
| `Q6_Vhf_vmpyacc_VhfVhfVhf` (native hf) | **no: unfused**, RN11(c + RN11(ab)): matched the unfused model 24,638/24,638 and differed from the fused one on 2,794 | 4.75 |
| widen to sf, RN24 sum, `Vhf_vcvt` (double rounding), + an exact hazard flag (midpoint ∧ inexact) and chain re-do | yes, with the fallback | 34.8 (the flag costs as much as the arithmetic) |
| same, midpoint-only flag | exact only through the re-do, which then fires on 60 % of the score chains (exact ties on real data) | 19.9 + the re-dos |
| **`qfma`: `Wqf32_vmpy(a,b)` + `Wqf32_vmpy(c,1.0)` → `Vqf32_vadd` ×2 → `Vhf_equals_Wqf32`** | **yes on the ISS**: 0 bitwise mismatches on 4,928 real cases from #136's dump. These include all 333 double-rounding hazards found in 12.6 M replayed fma16 ops (hazard rate 2.6e-5/op). Also 0 on 76,800 adversarial near-midpoint cases (1,206 subnormal results) and 0 on 30,528 zero/sign cases. The same sum narrowed through sf failed exactly the 333 | PV shape (4 q heads share each V row): **4.0**. Scores shape: strided Kt 10.9, tiled Kt 7.8, tiled + 2 q heads per Kt load 6.5 |

The native `hf` ops the tree and softmax need (add, sub, mul, ×0.125, `0 + x`,
max) were bitwise equal to `rne16(f32 op)` over every finite fp16 a × 2 random
b (126,976 pairs, signed zeros included).

## 1. Goal and gate

**Goal (issue).** ATTN_M1 per layer at or below the CPU's per-layer cost,
which is ≈ 0.3 ms at G = 1024 E2E (stretch: the 0.14 / 0.22 ms microbench at
L = 1024 / 1536). Keep it bit-identical to the Android fp16 CPU attention
(`attn_m1_det.h`, **arithmetic unchanged**), so that the six attention layers
stop costing ≈ 8–24 ms/token on the end-to-end path.

| # | check | where | pass |
|---|---|---|---|
| G1 | silicon semantics of the new primitives (S1) | new `HvxAttnM1Probe.Semantics` | `bad=0` for: `qfma` on the #136-derived case file + the adversarial families; hf add / sub / mul / max / `0+x` over all finite fp16 × 4 random; ×0.125 on all fp16; `exp16` over all 31,744 fp16 d ≤ 0 (if §3.3's vector exp is kept); hf `div16` on the host check's 7,881 off-by-one + 27,049 tie quotients |
| G2 | host bit identity | `run_host_checks.sh` | `ATTN M1 HF PRIM OK`, `ATTN M1 BIT-IDENTICAL` at L = 1 / 63 / 64 / 65 / 512 / 513 / 1024 / 1536 × pools 0 / 3 / 7, `ATTN M1 PHASES OK`, `ALL CHECKS PASS`. `run_inproc_e2e.sh`: `INPROC E2E PASS`, with **every resident line byte-identical to the base tree's** (the spec did not move, so the kernel output did not either: `fwd-hd64 min_snr_db` stays 37.17, etc.) |
| G3 | silicon bit identity (S2) | `HvxAttnM1.*`, `AttnM1F16Det.*`, `HvxM1Ops.Rope64*` | `out bad=0` at L = 1 / 63 / 64 / 65 / 512 / **513 / 1024 / 1536**; `bad_stats` equal to the reference skel's value at each L (rule 37); `append_chain bad=0`; `AttnM1F16Det` `bad=0` at 513 / 1024 / 1536; rope `bad=0` |
| G4 | CPU-vs-HTP per-op shadow (S2) | `dev/attn-shadow-170` (§4 step 6), `tools/htp/attn_shadow_check.py` | RQ1: every (layer, step, q head) output equal bit for bit to the CPU's fp16 attention of the same row, **1536/1536 heads** (6 × 8 × 32) at G = 8. Logits equal to A's at 8/8 steps |
| G5 | E2E text / nll (S2) | 8 prompts, G = 256; `NNTR_PPL_DECODE` forced on A's continuation | Q1 and RQ1 text ≡ A byte for byte on all 8 prompts. `nll_sum` equal to A's at the printed precision. Q1 tokens ≡ Q0 tokens |
| G6 | speed | level-2 profile `pcyc/op ATTN_M1=`; `PerLayerCost` | in-model `ATTN_M1` ≤ **T64** at G = 64 and ≤ **T1024** at G = 1024. Provisional T64 = 300 k and T1024 = 600 k pcyc (≈ 0.15 / 0.30 ms at 2.03 GHz). The final values are §3.5's model with S1's silicon constants × 1.15, written into the S2 handoff before S2 runs. If that model already exceeds 600 k at G = 1024, **stop after S1** and report |
| standing | every E2E cell | | prefill ≥ −5 % of A (mirrored band). Q1 decode ≥ 0.95 × A at G = 512 / 1024 (read, not the issue's gate) |

The issue closes on G1–G6.

## 2. Where it lives

**Changes (refs @ `aaafd0c4`).**

* `nntrainer/tensor/htp_backend/hvx/hvx_attn_m1_f32.c`: the whole kernel.
  * `hvx_attn_m1_create` `:77-132`: an hf cache, the tiled Kt, `max_seq % 64`.
  * `append_head` `:164-176`: writes hf bits.
  * `attn_body` `:279-431`: scores `:307-339`, softmax `:345-395`, PV
    `:401-427`.
  * Units / pool: `:435-467`. `hvx_attn_m1_forward_prof` `:476-534`: phases
    and barriers.
  * The phase words keep their meaning (`attn_m1_det.h:115-143`, 9 words).
* `hvx/hvx_attn_m1_f32.h`:
  * ctx `:83-95`: `kt` / `v` become `uint16_t *`, and `cache_floats` becomes
    `cache_halves`.
  * Budget note `:32-37`: 48 → **24 MiB** at max_seq 2048.
* **New** `hvx/hvx_attn_m1_hf.h`: `qfma`, the hf tree, the hf `div16`, and
  `exp16` (vector or table, §3.3). One header serves the kernel, the probe
  entry and the host check, so the probe tests the shipped code.
* `nntrainer/tensor/attn_m1_det.h`: **arithmetic unchanged.** Only the
  kernel note `:95-99` changes.
* `test/htp/nntr_hvx.idl`: one debug-only method `attn_m1_probe(in uint32
  op, in uint32 lanes, in uint32 reps, in sequence<uint16> a, in
  sequence<uint16> b, in sequence<uint16> c, rout sequence<uint16> y, rout
  sequence<uint32> prof)`, appended **last**.
  * The `attn_m1_*` signatures `:568-593` do not change: rows stay f32 on
    the wire.
  * Stubs: `test/htp/build.sh` regenerates the skel side, and
    `nntrainer/tensor/htp_backend/generate_stub.sh` the app side. Both are
    rebuilt and md5-recorded.
* `test/htp/nntr_hvx_attn_m1.c`:
  * The probe entry. `AEE_EINVALIDFORMAT` on bad lengths, never
    `AEE_EBADPARM`.
  * The FARF `cache=` at `:52-58` counts halves.
* `test/htp/host/hvx_emu/hvx_hexagon_protos.h` (279 lines, no hf / qf
  today). It gains:
  * hf add / sub / mpy / max as `rne16(f32 op)`;
  * `Wqf32_vmpy_VhfVhf`, `Vqf32_vadd_Vqf32Vqf32` and `Vhf_equals_Wqf32`,
    with qf32 held as a double (exact here: fp16 products and fp16 + product
    sums stay ≤ 53 bits wherever RN53 could matter; the proof goes in the
    comment);
  * `Vh_vsplat_R`, `vshuff` / `vdeal` / `vror`, and the even/odd lane order
    the ISS showed (`Wsf` lo = even lanes).
* `test/htp/host/attn_m1_host_check.c`:
  * The lengths `:904` gain 513 / 1536, and max_seq becomes 2048.
  * A new `ATTN M1 HF PRIM` block.
  * The phase check `:645` stays.
* **New** `tools/htp/attn_fma_cases.py`, which writes the probe's case file
  from `136/dump_attn`: every hazard and inexact-midpoint triple, plus 1 in
  3000.
* `test/unittest/unittest_hvx_attn.cpp`:
  * `MatchesDetSpecBitExact` `:969` gets L += 513 / 1536.
  * `PerLayerCost` `:1106/1138` covers pos 512 / 1023 / 1535.
  * New `HvxAttnM1Probe.{Semantics,Cost}`.
* `test/unittest/unittest_nntrainer_cpu_backend_fp16.cpp`: `AttnM1F16Det`
  attention lengths gain 1536 (`:1043-1113`).
* `nntrainer/tensor/htp_backend/htp_compute_ops.cpp:1741-1745`: the ARM
  banner's `cache=` counts 2 bytes, so it now reads **`cache=24576 KiB`**.
  Every handoff grep follows this.
* `nntrainer/tensor/htp_backend/hmx/hexkl_graph.c:22-24`: comment only
  (48 → 24 MiB). `graph_op_attn_m1` `:175-183` is unchanged.

**Consumers checked, not moving.**

* `HtpComputeOps`: the ATTN_M1 case `:1536-1551` and the seed `:1647-1665`
  pass f32 rows, and the kernel converts them.
* `nntr_quantize_stream`'s format tag and the loader check: no weight format
  change.
* `NNTR_HTP_PROFILE` stage tables and `tools/htp_fc_report.py`: the
  `pcyc/op` line keeps its format.
* `mha_core.cpp:411-416, 585-657` (the hook and the seed): unchanged in the
  PR.
* **#132 PR 2** (`htp/132-exact-fc`, another implementer) also appends to
  the IDL and edits `htp_compute_ops.cpp`. Whichever lands second rebases.
  This plan touches only the banner line there.

## 3. Design

### 3.1 The one-rounding FMA (`qfma`)

The fp16 accumulators, the probabilities and the cache are hf. A fused step
`RN11(c + a·b)` for 64 lanes has four parts:

1. `P = Wqf32_vmpy(a, b)`, the exact product.
2. `C = Wqf32_vmpy(c, 1.0)`, c exactly.
3. `S = Vqf32_vadd(P.lo, C.lo) | (P.hi, C.hi)`.
4. `r = Vhf_equals_Wqf32(S)`.

That is 2 double-vector multiplies + 3 VS ops. Its correctness is **a
silicon property** (G1, S1). The ISS evidence is §0's. Every other fp16 step
of the spec is one native hf op (tree adds, `0 + t`, ×0.125, `s − m`, max).
Each one is written as an explicit intrinsic, so the compiler lowering (§0)
has nothing to contract.

### 3.2 Scores and PV

**Scores.** Kt is stored **tiled**, `[layer][kv][max_seq/64][64 d][64 pos]`
hf: one 8 KiB contiguous tile per 64 positions.

* 8 accumulators run over d = 8 blk + l (the CPU's lanes), with 2 q heads
  per Kt load (#146's O2).
* Then the `faddp` tree, `0 + t` and ×0.125, all in hf.
* This is #146's O1: literal shapes, with the accumulators in registers (the
  `-S` grep must find no `vmem(r29)` in the loops).

**PV.** Lanes = d (64 = one hf vector per V row). 4 q heads share each row
load, and each runs a `qfma` chain over p ascending, as the CPU does.

### 3.3 Softmax in hf, with the sum vectorised across heads

The sum is one sequential fp16 chain per head, over positions. Today's cost
is one scalar chain per head.

* m = hf max; d = hf(s − m).
* e = exp16(d), in one of two forms:
  * **vector** `exp_ps`, with an explicit `qf32` op + `Vsf_equals_Vqf32` at
    every f32 rounding point of the spec (never the compiler's lowering). The
    host already proved the unfused `fx` step equivalent for all fp16 d;
  * *or* today's table, via scalar lookups, if G1's exhaustive `exp16` row
    fails.
* e is written to `[h][p]` and, through an in-register `vshuff` transpose,
  to `[p][h]`.
* One lane then runs the sum as L dependent hf adds over a 32-head vector
  (≈ 1 add per position instead of 32 scalar chains).
* p = hf `div16`: `hvx_div16_sf`'s proof, re-derived on qf32 products in hf
  lanes.

### 3.4 Threads

The work is split into three phases, one pool run each, with the middle
phase serial:

* **P1**: scores, max, d and exp, by (kv head, position range). This is
  balanced over the 6 lanes, since the blocks are independent.
* **P2**: the sum, on the caller.
* **P3**: divide + PV, by q head (32 units).

The O3 idea (units that share one kv slab run together) is kept. The split
is re-chosen after S1's lanes sweep: if silicon throughput is flat past 2
lanes, the lanes buy only fetch overlap.

The KV read is fp16: 2 KiB·L per layer (1 / 2 / 3 MiB at L 513 / 1024 /
1536). #146's O4 `l2fetch` lead is kept on the next Kt tile / V rows, so the
fetch hides behind the compute (doc 45 §3.2).

### 3.5 Cost model

Per layer, in ISS cycles × 1.08 (silicon/ISS on today's loop, §0),
at 2.03 GHz:

* 64·L 64-lane `qfma` (scores 32·L at 4.5–6.5 cycles, PV 32·L at 4.0).
* Softmax, tree and divide ≈ 60·L.
* ≈ 20 k fixed (2 pool runs, P2 setup, append).

| L | today, cold gtest / in-model | model, scores 6.5 → 4.5 | fp16 bytes (hidden by O4 if < compute) | CPU E2E / microbench |
|---|---|---|---|---|
| 513 | 1.10 / 1.38 ms | 0.12 → 0.10 ms | 1.0 MiB ≈ 35 µs | — / 77 µs (split4; as_is 371 is an outlier) |
| 1024 | 2.19 / ≈ 3.0 ms | 0.23 → 0.19 ms | 2.0 MiB ≈ 70 µs | 0.28–0.31 / 0.14 ms |
| 1536 | ≈ 3.3 (extrap.) / ≈ 4.0 ms | 0.33 → 0.28 ms | 3.0 MiB ≈ 105 µs | — / 0.22 ms |

That is ≈ 10–13× today. It meets the 0.3 ms E2E bar at L ≤ 1024, and is
borderline at 1536. It is 1.3–1.5× the CPU microbench. The in-model / cold
factor (1.2–1.5 today) is the open term; S1 measures it.

### 3.6 Rejected

* **Double rounding + an exact hazard flag + chain re-do.** It is exact, but
  34.8 cycles vs 45.5 buys only 1.3×. On v79 each sf op of the exactness test
  is itself a `qf32` op + convert.
* **Native `hf vmpyacc`.** It is unfused, and the unfused variant was already
  rejected by #152 (52–56 dB).
* **Keeping the sf accumulator and rounding in the qf32 domain** (magic add):
  11.5 cycles on the ISS.
* **The f32 cache.** Bytes are not today's bound, but hf lanes need hf
  operands, and the halved cache frees 24 MiB of the ≈ 182 MiB heap.

Contract walls: arena and address space shrink, and there is no CPU
fallback. Doc 45 §3 holds: the `_det` spec is unchanged, and the gates are
host + silicon bit identity plus text / nll.

## 4. Steps

1. **Primitives + probe (host).**
   * `hvx_attn_m1_hf.h`, the `hvx_emu` additions, `ATTN M1 HF PRIM` against
     `attn_m1_det_fma16` / `exp16` / div, and `attn_fma_cases.py`.
   * The probe entry + IDL method + `HvxAttnM1Probe.*`. The Cost ops are:
     today's fma16 loop (the calibration), scores tiled ×1 / ×2 heads, PV ×4
     heads, and a cold-stream fetch of a 3 MiB hf slab with `l2fetch` 0/1.
     Each runs at lanes 1 / 2 / 4 / 6, and prints `pcyc_per_fma64` and `mhz`.
   * `PerLayerCost` pos 1535; L 513 / 1536 in the gtests.

   **Gate:** rung 0, rung 1 (`ALL CHECKS PASS`), rung 2 (`UNDEFINED SYMBOLS
   OK`, skel md5), rung 3 for `unittest_hvx_attn` +
   `unittest_nntrainer_cpu_backend_fp16`.
2. **DEVICE S1 — microbench sitting (unavoidable).** The orchestrator runs it
   on `R3CY10WM83Y` from a `164-run.sh`-style script; ≈ 30 min. The
   handoff is `docs/measurements/170-attn-m1-hf.md` (S1 part).
   * Two run dirs:
     * `s170p` = the probe set (skel + gtests), with the kernel still today's;
     * `s170a` = the unchanged #164 set (`164/set/`, skel `md5.txt`), for
       the E2E cells. Keeping them apart means no stub/skel mismatch.
   * Cells:
     * (a) `HvxAttnM1Probe.Semantics` (G1).
     * (b) `HvxAttnM1Probe.Cost` × lanes.
     * (c) `HvxAttnM1.*` + `PerLayerCost` warm / cold at pos 512 / 1023 /
       1535: today's per-L line.
     * (d) `AttnM1F16Det.*` at 513 / 1024 / 1536.
     * (e) The contract's full-model cells: A (switch off) and Q0-prof
       (`MOE,QK_NORM,ROPE,ATTN_M1`, `NNTR_HTP_PROFILE=2`) at G = 64 and
       1024, prompt 512, ×1 each. This gives today's in-model `pcyc/op` at
       two L ranges, and the in-model / cold factor.
   * Thermal log at each checkpoint.
   * **Stop rules:** any G1 `bad` ≠ 0 on `qfma` → stop and report (the
     fallback is §3.6's 1.3×, a user decision). `0x8000040e` → stale skel.

   **Read:** T64 / T1024 from §3.5 with S1's constants; the unit split
   (§3.4); `l2fetch` on or off; vector vs table `exp16`.
3. **Kernel (host).** §3.2–3.4, the phase words, O4.
   **Gate:** rung 0–2. The `-S` grep shows no vector spills in the scores /
   PV loops.
4. **Consumers + host E2E.** The banner, the FARF, the comments.
   **Gate:** rung 1 incl. `INPROC E2E PASS` with every resident line
   byte-equal to the base tree's (G2).
5. **Rung 3.** Build the app + all three gtest binaries, with md5s, `readelf`
   NEEDED, and the `NNTR_HTP_FORWARD_KINDS` count.
6. **Shadow set.** `dev/attn-shadow-170` = the PR diff + one inert,
   never-merged commit, following `dev/norm-shadow`'s pattern:
   * `NNTR_ATTN_SHADOW=<file>`: after the hook returns 1 in
     `htpDecodeAttention`, the CPU fp16 path runs the same row into scratch.
     It keeps its own cache current and restores `cache_index`. One record
     per call: tag 3, pos, ordinal, row, cpu out, htp out.
   * `tools/htp/attn_shadow_check.py` compares per head, and
     `NNTR_LOGIT_SHADOW` gives the logits.

   **Gate:** rung 3 on that set.
7. **DEVICE S2 — E2E sitting (unavoidable).** ≈ 60 min.
   * **Variants (4):**
     * **A** = the new set, switch off, run first;
     * **Q0** = the #164 set, Q mask;
     * **Q1** = the new set, Q mask;
     * **RQ1** = the new set, `MOE,RMSNORM,QK_NORM,ROPE,ATTN_M1`.
   * Speed: A / Q0 / Q1, full E2E, prompt 512, G = 64 / 512 / 1024 × 2,
     mirrored `A Q0 Q1 | Q1 Q0 A`. Banners `calls/token=28.00` (Q) / `77.00`
     (RQ), and `cache=24576 KiB` in Q1 / RQ1 (49152 in Q0).
   * Also:
     * the G3 gtests;
     * the G4 shadow (RQ1, G = 8);
     * G5: 8 prompts at G = 256 for A / Q1 / RQ1, plus PPL forced on A's
       continuation;
     * G6: Q0-prof / Q1-prof at G = 64 and 1024.
   * `NNTR_NUM_THREADS=8`.
8. **Fold.** The PR goes to `state:review`, with the S2 numbers in the
   handoff.

## 5. Risks (host vs device)

* **Silicon qf32 ≠ ISS qf32** (hvx_impl: the v75/v79 `qf32` conversions
  differ). The whole design rests on it, so S1 tests it first, on real
  hazards, before any kernel work. `hvx_emu` models qf32 as exact and cannot
  see a difference.
* **ISS cycles ≠ silicon.** #164's ISS-set gates missed 4× on silicon (DDR
  latency). Here the loops are compute-bound and today's loop calibrates at
  1.08, but T64 / T1024 are set only from S1's silicon constants, and the
  cold / in-model factor comes from S1 (e).
* **DMA / DDR rate.** ATTN_M1 reads by HVX loads (rule 26: latency). The S1
  fetch probe gives the GB/s with and without `l2fetch`. S2's warm / cold /
  in-model triple shows any unhidden fetch.
* **DVFS / thermal drift.** Every verdict is read inside its own sitting, in
  mirrored order, with the `mhz` word and zone0 per checkpoint. G6 is in
  pcycles, not µs.
* **Stale skel.** The IDL changes. Two run dirs in S1, every log carries the
  skel md5, and `0x8000040e` is a stop.
* **Address space.** The cache falls 48 → 24 MiB. The new scratch is
  hf e `[h][p]` + `[p][h]` (2 × 128 KiB) and there is no VTCM use. The
  banner shows the cache size.
* **Compiler contraction.** Write explicit qf32 / hf intrinsics only, never
  `Q6_Vsf_vmpy`→`vadd` with an inexact product. G1 / G3 catch any slip.

## 6. Docs to update

* **BENCHMARK.md.**
  * A #170 S1 side table: the G1 rows, `pcyc_per_fma64` × lanes, fetch
    GB/s, today's per-L warm / cold / in-model line.
  * S2 rows: A / Q0 / Q1 × G, the in-model `ATTN_M1` pcyc/op at G 64 / 1024,
    text / nll / shadow.
  * The ⑨ budget row: attention ms/token.
* **LEDGER.md.**
  * ㉗: the new per-L cost.
  * Rule candidates, once confirmed on silicon:
    * *v79 hexagon-clang lowers IEEE sf / hf intrinsics to `qf` + convert
      and, without `-mhvx-ieee-fp`, contracts sf mul + add*;
    * *`Vhf_vmpyacc` is unfused; the CPU's `fmla .8h` is `qf32` mpy + add +
      `Vhf_equals_Wqf32`*;
    * *the v79 vector pipe is per core: 6 lanes ≈ 1 thread's throughput*
      (if S1 shows it).
  * Close plan 146 §3.4's fp16-cache item.
