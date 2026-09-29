# Plan 157: split the 4 routed experts of each decode MoE call between the DSP and the CPU, bit-identical

Issue #157 (p1, part of #76, target (b) in its split form, user 2026-09-29,
under the bit-preserving rule). Branch `htp/157-expert-split` off
`origin/htp_moe` @ `35aa1b0d`. This document answers **step 1** (the issue's
question, no device) and plans **step 2**, which only makes sense because the
answer is yes.

## Verdict of step 1

**Overall: YES.** The CPU can reproduce the DSP's M=1 expert arithmetic bit
for bit. The ISS experiment in §0.3 settles it for normal-range values (every
stage 0 mismatches, while an FMA control fails). Four conditions come with
the yes, and none of them needs a DSP change:

1. **Merge order.** The DSP takes a prefix of the active experts in
   ascending expert id and returns its partial sum. The CPU continues that
   sum in the same order (§0.2). With a prefix split, the DSP entry, the IDL
   and the skel stay unchanged.
2. **No contraction on the CPU.** Every f32 step on the CPU is one IEEE op
   with no FMA and no reassociation, written with the `swiglu_det.h`
   barrier discipline. `-ffp-contract=off` alone is not enough: the app
   builds with `-ffast-math` (`Applications/CausalLM/jni/Android.mk:45`),
   and doc 44 §12.5 showed that `#pragma clang fp contract(off)` does not
   reach NEON intrinsics.
3. **Cached weights for the CPU.** The CPU must read the `QS4CX_WH` bytes
   through a cached mapping. Today the arena is uncached on the ARM side
   (`htp_compute_ops.cpp:3882`, `htp_rpcmem.h:127-131`), and the ARM source
   copy is dropped (`releaseArmSource`, `:3609`). That is a cost, not a
   blocker (§3).
4. **Subnormals are not covered.** The one exposure left is values near
   2^-126 (rules 24 / 37). The ISS does not cover them, and neither do the
   device gtests. At the MoE's value ranges they do not occur (§0.1 row 5),
   and the device dump gate of step 2 would catch them if they did.

No step needs a qf32 emulation, so the "YES-with-cost (emulation)" branch does
not apply. The fallback track (c), a non-bit-identical split judged by PPL,
8-prompt text and approval, is **not needed**.

### 0.1 The M=1 path, op by op

This is the default path: `applied=0x303e1`, dspqueue, the VTCM feed, and
`use_m1` at `hexkl_mm_u8i4_moe.c:1087`. Reading the table:

- "IEEE" means plain f32 ops with two roundings where the source writes two
  ops, reproducible on NEON in the same order.
- "Lowered" is what the skel's compiler emits. `test/htp/build.sh:93-96`
  builds with `-mv79 -mhvx -mhvx-length=128B -O3 -fPIC` and **no**
  `-mhvx-ieee-fp`. Clang 19.0.04 lowers every `Q6_Vsf_vmpy/vadd/vsub_VsfVsf`
  to `qf32` ops and folds away the intermediate `.sf = .qf32` converts. For
  example, `hvx_scale_add_rows_f32` becomes
  `v3.qf32 = vmpy(v1.sf,v0.sf); v4.qf32 = vadd(v3.qf32,v2.sf); v5.sf = v4.qf32`,
  and the fused SwiGLU's Horner steps become
  `v26.qf32 = vmpy(v26.qf32,v27.qf32)`.
- The float code is identical with and without `-fPIC`: 0 diff lines over
  the four translation units.
- §0.3 shows that these folded chains give exactly the IEEE two-rounding
  result on normal values.

| # | step (source) | float ops, in order | lowered | verdict |
|---|---|---|---|---|
| 1 | activation row params, `hvx_quant_rows_u8_params` → `quant_row_params_one` (`hvx_quant_u8.c:37-92`), called at `hexkl_mm_u8i4_moe.c:1328` | HVX `vmin`/`vmax` over K, then a lane fold (exact, order-free). Scalar: `rmin = min(min0,0)`, `rmax = max(max0,0)`, `s = (rmax − rmin) / 255.0f` (`:81`), `z = nearbyintf(−rmin / s)` clamped to [0,255] (`:84`). A constant row gets `s = 1`, `z = 0` | Hexagon's inline divide `sfrecipa / sffixupn / sffixupd / sffma`, which is correctly rounded (the internal `sffma` is part of the division algorithm, not a contraction of source ops); libc `nearbyintf` | **YES**, IEEE. The CPU uses a real `fdiv` and `nearbyintf` (RNE) |
| 2 | activation pack, `hvx_quant_pack_u8_ah_rows` → `quant_pack_group4` (`hvx_quant_u8.c:177-200, 247-276`), at `:1363` | `inv = 1.0f / s` (scalar divide, `:267`); then `q = rne(x·inv) + z` saturated to u8, where `rne(v)` = bits(`v + 12582912.0f`) − `0x4B400000` (`hvx_convert.h`), i.e. one mul and one add, each rounded | `vmpy qf32` → `vadd(qf32, sf magic)` → `.sf`, product convert folded | **YES**. The CPU computes `nearbyintf(round(x·inv))` or the same magic add (|v| ≤ 255 ≪ 2^22). Output is u8, so only exactness matters |
| 3 | gate_up and down GEMV, `hvx_gemm_u8i4_wh_col[_nopf]` (`hvx_gemm_u8i4_wh.c:65-84` `gemm_row1`, since `rows1=1` at m=1) | int32 only: `vrmpy` u8 × (16·i4), kt then g then low then high nibble, `>> 4` | no float | **YES**. Exact in any order (\|Σ\| ≤ 255·8·16·2048 < 2^31). The CPU is free to pick its own order |
| 4 | dequant, `dq_row_sf` (`hvx_dequant_i32.c:186-198`, gate/up) and `DQ_TILE_ROW` (`:85-96`, down, `:104`) | `af = (f32)acc`, `zf = (f32)zp`, `cf = (f32)colsum` (all exact, < 2^24); `c = af − zf·cf` (exact: \|zf·cf\| ≤ 255·16384 and \|c\| < 2^24); `r = ((c·s)·ws) + bias`, three roundings, in this order | qf32 chain, converts folded | **YES**, IEEE. The CPU may form `c` in int32 and convert (same value). `bias` is `0.0f` for experts (`htp_compute_ops.cpp:3693`), and `x + 0.0f` keeps `x` for `x ≠ −0` |
| 5 | SwiGLU, `hvx_swiglu_det_sf` (`hvx_swiglu_det.h:110-190`), fused into `hvx_dq_swiglu_worker` (`hvx_dequant_i32.c:200-237`), called from `moe_m1_pair_worker` (`hexkl_mm_u8i4_moe.c:701`) | the `swiglu_det.h` spec: clamp [−88, 85]; `k = rne(x·log2e)`; `r = (x − kl·LN2_HI) − kl·LN2_LO`; 7 Horner steps (mul, then add); exponent add; underflow `k + e(p) ≤ 0 → 0`; `recip_det` (magic seed, 3 Newton steps); `(g·s)·u` | qf32 chain. In this TU the gate value `g` and `r` stay in qf32 across uses (different folding from `hvx_swiglu_f32.c`, the one `HvxSwigluDet` tests on silicon; plan 146 §3.3's caveat) | **YES**, IEEE = `swiglu_det_one` / `swiglu_det_neon` (`nntrainer/tensor/swiglu_det.h:179, 265`), which already exist and are device-gated (`HvxSwigluDet.MatchesScalarBitExact`, `SwigluDetNeon.MatchesScalar`). The ISS confirms this TU's folding too (§0.3). Smallest output: `exp_det` ≥ 1.6e-38, and `(g·s)·u` goes subnormal only for \|g\| ≳ 80 with \|u\| ≲ 1e-3. That is not seen at real gate values, and the dump gate would catch it |
| 6 | down-input requant, `moe_m1_requant_worker` (`hexkl_mm_u8i4_moe.c:712-728`) | rows 1..3 of `gate_f32` memset to 0 (never read for row 0); then steps 1 and 2 over `inter` = 1792 values of row 0 | as 1 and 2 | **YES** |
| 7 | down dequant into `res`, `moe_m1_down_worker` (`:739-776`, `:765`) | as 4 with `rq_scale`/`rq_zp` | as 4 | **YES** |
| 8 | scatter, `hexkl_mm_u8i4_moe.c:1478-1487` → `hvx_scale_add_rows_f32` (`hvx_scale_add_f32.c:34`) | `out_c` memset to +0 (`:1291`); for each active expert **in ascending expert id** (`order[]`, `:1276-1279`), each row: `out = out + round(res·w)`, one mul and one add, each rounded | `vmpy qf32` → `vadd(qf32, sf)` → `.sf` | **YES, order-dependent**: `((((+0 + p_e1) + p_e2) + p_e3) + p_e4)`, where e1 < e2 < e3 < e4 by id and `p = round(res·w)`. On silicon already: doc 46 §33.2a matched a host IEEE two-rounding reference once the reference's FMA was removed |
| 9 | copy-out (`:1491`), transport | bytes | — | exact |

### 0.2 The merge order and the prefix split

The DSP sums the experts in `order[]`, which is ascending expert id over the
experts with `row_count > 0` (`hexkl_mm_u8i4_moe.c:1270-1287`), starting from
+0. Float addition is not associative, so the CPU's experts cannot be summed
separately and added at the end. The one split that needs **no DSP change**
works like this:

- The DSP gets the routing with only its experts. These are the **first k
  active experts in id order**, and the rest have `row_count = 0`, with
  `row_index` / `row_weight` trimmed to match.
- For those experts the DSP computes exactly today's sequence, so its output
  row is the partial sum `S_k = (…(+0 + p_e1)…) + p_ek`.
- The CPU computes `p_e` for e_(k+1)…e_4 during the call. After the DSP
  answers it continues `S = S_k; S = S + p_e(k+1); …` in id order. When
  k = 0, the CPU starts from a memset +0, like the DSP.

Every per-expert quantity depends on the token row and that expert's weights
only. The activation quant is per row, the requant is per expert, and every
column is independent. So a DSP call with fewer active experts produces the
same `p_e` for the experts it keeps.

### 0.3 The experiment that settles it (throwaway; sources in the appendix)

**Setup.** The real skel sources were compiled with the skel's own flags
(`hvx_dequant_i32.c`, `hvx_quant_u8.c`, `hvx_scale_add_f32.c`,
`hvx_gemm_u8i4_wh.c`; `hexagon-clang` 19.0.04, `-mv79 -mhvx
-mhvx-length=128B -O3 -G0`) and run on `hexagon-sim -mv79` (rev `v79na_1`).

- Shape: the M=1 expert pipeline at the LFM2 shape (K 2048, inter 1792,
  N_out 2048), in the call order of §0.1.
- Data: 6 tokens × 4 experts, random WH weights, true column sums,
  `w_scale` ≈ 2–6e-3, bias 0.
- Activation ranges: amplitudes 0.3 / 3 / 30 with 1 % outliers, plus one
  all-positive row (`zp = 0`). Gate values reached \|g\| = 324, so the exp
  clamp path runs.
- Extras: 300 k `a + b·w` scale-add cases (wide-exponent normals and
  tie-heavy products and sums) and 300 k scalar divides.

**Comparison.** Every stage was compared on the host (x86-64,
`-ffp-contract=off`, every op through a volatile) against the plain IEEE spec
of §0.1. The host spec takes its own previous stage's output, so this is also
a chained check.

| stage | mismatches |
|---|---|
| act params (s, zp) | 0 of 12 |
| act u8 | 0 of 12 288 |
| gate_up int32 | 0 of 86 016 |
| dequant + SwiGLU f32 | 0 of 43 008 |
| requant params | 0 of 48 |
| mid u8 | 0 of 43 008 |
| down int32 | 0 of 49 152 |
| down dequant f32 | 0 of 49 152 |
| output row (IEEE mul, add; id order) | **0 of 12 288** |
| output row, FMA instead (control) | 4 476 of 12 288 |
| scale-add sweep | **0 of 300 000** |
| scale-add sweep, FMA (control) | 22 106 of 300 000 |
| scalar f32 divide | 0 of 300 000 |

**What this proves.** The ISS is Qualcomm's instruction-set model, not the
project's `hvx_emu`. Rule 37 concerns `hvx_emu`, which departs from silicon
only at 2^-120…2^-132. The ISS is not a gate in this tree (LEDGER ⑬, "no
simulator", user decision), so it counts here as evidence for a planning
question and nothing more. The proof on silicon is step 2's device gtest and
the MoE dumps.

The ISS result agrees with every silicon result we have:

- the V1 scatter (doc 46 §33.2a)
- `HvxSwigluDet` (doc 44 §12)
- `ATTN_M1 out bad=0` and rmsnorm kinds 0 / 1 / 3 (rule 37), all normal-range

## 1. Goal and gate (step 2)

**Goal.** Decode ≥ 50 tok/s at G 64 / 512 / 1024 (contract §1). Today's
reference is #90's control, 39.70 / 39.18 / 36.53 (`htp_moe` dspq default,
`R3CY10WM83Y`, 2026-09-29).

**Bit identity** comes before anything is measured:

- (a) Host: `run_host_checks.sh` prints `MOE SPLIT BIT-IDENTICAL k=0..4`.
  This is the CPU expert path against the real `hexkl_mm_u8i4_moe.c` M=1
  path on `hvx_emu`, at the LFM2 shape, with `memcmp` on the output row.
- (b) Device gtest `HmxMmU8I4Layer.MoeM1CpuSplitMatchesDsp`:
  `bad_elems=0` for k = 0…4 against the all-DSP call, NEON path on the phone.
- (c) Device E2E: MoE dumps of every split variant `bit_identical=1`
  against A (`tools/htp/htp_dump_eval.py`, all files).

**Standing gates, per variant against A of the same sitting:**

- text identical to A on the 8 prompts (p01–p08). A bit-identical change
  keeps the text, so any difference is a fail.
- decode PPL (`NNTR_PPL_DECODE`) byte-identical to A's `[PPL] decode`
  lines.
- prefill ≥ −5 % of A's prefill (the split is off at M > 1, so expect
  ±noise).
- the log banners of rule 40, plus the new split banner.

**Lever gate.** S2 decode ≥ A + 20 % at G = 512 in the mirrored mean; the
model predicts +30 to +42 %. The BENCHMARK.md cells are the G 64 / 512 / 1024
decode (all tokens and last 64) and prefill rows for A and each split
variant.

## 2. Where it lives

| file | change | consumers that move with it |
|---|---|---|
| `nntrainer/tensor/htp_backend/htp_compute_ops.cpp` `invokeMoeLayer` (`:2881`), `via_dspq` (`:2903`), `dspqCall` (`:2785`; write `:2831`, spin-read `:2841`) | Under `NNTR_MOE_HTP_SPLIT=k` (k = experts on the DSP; unset or 4 = today's path, byte for byte) and only when `via_dspq`: 1. build the trimmed routing (first k active experts by id); 2. `api.write`; 3. run the CPU experts on `ThreadManager::Global().parallel_for` (`nntrainer/utils/thread_manager.h:170`) between write and read; 4. read; 5. continue `S_k` in id order. `dspqCall` splits into send and wait. The FastRPC path (`NNTR_HTP_DSPQ=0`) never splits (it blocks the caller) | `dumpMoeCall` (`:2417`) writes the merged row, so the dumps compare against A as they are. It gains `row_weight` (`_w.f32`), so a host tool can recompute a call |
| same file, arena (`ArenaChunk`/`ArenaEntry` `:3571-3589`, `get_or_register_wh` `:3643`, chunk alloc `:3882`) | Keep a handle → `ArenaEntry` map for the CPU (WH pointer = chunk ARM mapping + `off`, plus `w_scale`, `colsum_w`, `bias`). Chunks become **cached** on the ARM side when the split is enabled (`HTP_RPC_FLAGS_DEFAULT`), with one cache clean per fill before the DSP reads (§3) | `htp_rpcmem.h:125-131` (flag comment); gtests `ArenaMapAndDma`, `ArenaUncachedWriteAfterMap`, `MoeLayerFromArenaMatchesHeap` (`unittest_hvx_mm_u8i4.cpp:1728, 2134, 2208`) run in both modes |
| **new** `nntrainer/tensor/moe_m1_det.h` (next to `swiglu_det.h`, `m1_ops_det.h`) | The CPU expert path as a `_det` spec: `m1_act_params_det`, `m1_act_quant_det` (steps 1 and 2), `m1_wh_gemv_i32` (NEON `sdot` by element on the WH quarter vectors; scalar reference), `m1_dq_det` (step 4), SwiGLU via `swiglu_det()`, `m1_scale_add_det` (step 8). Every f32 op goes through the `swiglu_det_vmul/vadd/vsub` barrier or a volatile | the host check (§4 step 2), the device gtest, `htp_compute_ops.cpp` |
| `test/htp/host/moe_layer_host_check.c` + `run_host_checks.sh` | New case: the real M=1 path on `hvx_emu` against `moe_m1_det.h` for k = 0…4, bit for bit (the existing `reference_layer` at `:447` is tolerance-based and stays) | `hexagon-gates` rung 1 pass line |
| `test/unittest/unittest_hvx_mm_u8i4.cpp` | `MoeM1CpuSplitMatchesDsp` next to `MoeLayerM1GemvMatchesHmx` (`:2386`) | device gtest list in the handoff |
| `nntrainer/utils/op_time*` / `tools/htp/op_time_report.py`, `NNTR_HTP_PROFILE` ARM rows | The `[OP-TIME] moe` line gains `split=k cpu_us= dsp_wait_us= cpu_MB=`, so the report can show the CPU's in-app GB/s next to the DSP's. The DSP stage table (`HTP_MOE_N_STAGES`) is **unchanged** | `tools/htp_fc_report.py` only if it parses the `moe` line (check) |

**Unchanged, stated so nobody moves them.** The IDL `test/htp/nntr_hvx.idl`
and its stub (`generate_stub.sh`) stay as they are, and so does the skel: no
DSP source changes and no rung 2. The quantizer format tag
(`nntr_quantize_stream`) and the model file stay too: the CPU reads the same
WH bytes. The loader check for `QS4CX_WH` also stays: `FloatTensor::dot`
still throws, and the new kernel is reachable only from the split.

LEDGER rule 6 ("no CPU fallback") keeps its meaning. This is not a fallback.
It is a co-executor that is gated bit-identical to the DSP.

## 3. Design

**Chosen: a prefix split at expert granularity, 2:2 fixed.** The CPU kernel
works on the WH tiles directly, and a cached arena gives the CPU its read
path.

- **CPU GEMV.** A WH quarter vector (128 B) holds 32 columns × 4
  consecutive k. A 16-byte NEON load is 4 columns × 4 k, which is exactly
  the operand shape of `vdotq_laneq_s32`. To unpack: `(v << 4) & 0xF0` gives
  16·w for rows 8g..8g+3, and `v & 0xF0` gives it for 8g+4..8g+7. The
  activation goes u8 → s8 by `^ 0x80`, and the sum is corrected exactly by
  `+ 128·16·colsum`, then `>> 4`. It needs FEAT_DotProd only; `vusdotq`
  (I8MM) is an optional variant with identical int32.
- **Streaming.** Threads own contiguous n-tile ranges and walk kt outer, so
  each reads `(nt1 − nt0) × 512` contiguous bytes per kt.
- **Epilogue.** There are three parallel phases per call over both CPU
  experts, like the DSP's A / B / C: pair GEMV + dequant + SwiGLU, then
  requant, then down + dequant. The merge (2 × 2048 adds) runs on the caller.
- **Cost.** 5 NEON ops per 16 B of weights. At 39 GB/s over 8 cores that is
  about 1.5 G ops/s per core, so the kernel is memory-bound.

**Split ratio model.** A call reads 22.03 MB, 5.51 MB per expert (#150, rule
42). The window is `max(DSP_k, CPU_(4−k))`, per call.

| rates (DSP / CPU, GB/s) + fixed per side | k = 2 (2:2) | k = 1 (1:3) | k = 3 |
|---|---|---|---|
| #90 concurrent 31 / 39, +0.03 ms | DSP 0.385 / CPU 0.312 → **0.385 ms** | 0.454 | 0.563 |
| in-app DSP 33 × 0.83 = 27.4 / kernel 32, +0.06 ms | 0.462 / 0.404 → **0.462 ms** | 0.576 | 0.663 |

Balancing by bytes would put 41–45 % on the DSP (1.65–1.8 experts). The
window is a per-call max, so alternating 1:3 and 2:2 by layer is worse than
2:2 every layer, and **2:2 is the choice**. Today a call is ≈ 0.727 ms (16.0
ms MoE wait over 22 calls).

**Expected decode.** The part outside the MoE is taken from #90 A minus a 16.0
ms wait (16.2 at G = 1024): 9.2 / 9.5 / 11.2 ms. Adding 22 × window gives:

| | G 64 | G 512 | G 1024 |
|---|---|---|---|
| today (#90 A) | 39.70 | 39.18 | 36.53 |
| 2:2, #90 rates | **56.6** | **55.6** | **50.9** |
| 2:2, conservative | 51.7 | 50.8 | 46.9 |

G = 1024 misses 50 in the conservative case. Attention grows with position,
and the MoE is not the only term left there.

**Weight access (the real cost of the yes).** Two options:

- *Chosen:* allocate arena chunks cached on the ARM side, and clean once
  after each fill (`DMA_BUF_IOCTL_SYNC`, or fill before `attach`, the order
  doc 46 §32 proved). The weights are written once and never again, so no
  line can go stale after that. RSS, the ARM address space and the DSP's
  32-bit space (rule 8) are all unchanged.
- *Rejected:* keep the ARM source copies (`NNTR_HTP_KEEP_ARM_WEIGHTS=1`,
  `:3609`). It needs no new memory code but adds ≈ 3.7 GB RSS, which is the
  pressure that made `releaseArmSource` necessary in the first place (doc 46
  §36.3).

**Rejected alternative for the split itself: per-expert result rows back
from the DSP** (a merge on the CPU in any interleaved order). It needs an IDL
field, a skel rebuild and a larger dspq response, all to buy an assignment
freedom that the equal-size experts do not need.

**Also rejected: KleidiAI `qai8dxp_qsi4cxp`** (`kleidiai_interface_qai8dxp_qsi4cxp.cpp`).
It uses a different layout, and its own dynamic quantizer rounds differently
from step 1.

**Contract checks:**

- §2 three walls: the transport stays dspq, and one call per layer is
  unchanged.
- Arena budget: unchanged.
- No CPU fallback for `QS4CX_WH`: see §2.
- Doc 45 §3: there is a `_det` spec before every quantizer (steps 1, 2 and
  6 on the CPU mirror the DSP's), with bit-identical and text gates.
  Activation handles and DMA-behind-compute are untouched on the DSP side.

ponytail: the expert-granularity split leaves up to 0.04 ms per call (the
gap between 0.385 and the balanced 0.345). A finer split (down-projection
columns of the boundary expert, bit-identical per column) is the upgrade if
S2's `dsp_wait_us` stays ≫ `cpu_us`.

## 4. Steps

1. **Spec header + scalar path** (`moe_m1_det.h`, scalar only). Gate: rung 1,
   plus a unit case that reproduces §0.3's host spec on its fixture
   (`ninja -C build`, the new gtest `[ PASSED ]`).
2. **Host bit check against the real DSP code.** Add the case to
   `moe_layer_host_check.c`: the M=1 path on `hvx_emu` with the full routing
   versus the prefix routing plus `moe_m1_det.h` continuing the sum, for k =
   0…4, three amplitude sets and an all-positive row. Gate:
   `run_host_checks.sh` → `MOE SPLIT BIT-IDENTICAL k=0..4` and `ALL CHECKS
   PASS`.
3. **NEON GEMV and epilogue** behind the same spec. Gate: the aarch64
   build's `objdump -d` shows **0 `fmla`/`fmls`** in the header's functions
   (`libnntrainer.so`), and `SwigluDetNeon`-style NEON == scalar holds on
   qemu-aarch64 (the doc 44 §12 precedent).
4. **Cached arena.** Add the flag and one clean per fill, active only when
   the split is enabled, so A's path is byte-identical. Gate: rung 1, plus
   the three arena gtests built (the device runs them in step 7).
5. **Split in `invokeMoeLayer`** (`NNTR_MOE_HTP_SPLIT`, banner
   `[HTP] moe split: k=2 cpu_threads=8 arena=cached` once). Gate:
   `run_inproc_e2e.sh` with the split set (scalar path on x86) prints
   `E2E eval split … bit_identical=1` against golden, plus the existing
   `INPROC E2E PASS`.
6. **App build** (rung 3, `build_android.sh --htp`). No skel rebuild: the
   staged skel is A's (`37468a7f…`, the #150/#90 set), and the handoff
   checks its md5.
7. **Device measurement (unavoidable; the handoff).** One sitting, one unit,
   mirrored order, prompt 512, G 64 / 512 / 1024, `NNTR_NUM_THREADS=8`,
   `NNTR_OP_TIME=1` in every cell (inert per #150). Variants (4):
   - **A**: `htp_moe` unchanged (split unset, arena uncached). The reference.
   - **C**: `NNTR_HTP_ARENA_CACHED=1`, split off. Isolates the cache
     attribute: dumps ≡ A, decode within ±1 %, M==1 `mm` within ±2 %.
   - **S2**: `NNTR_MOE_HTP_SPLIT=2` (implies cached). The lever.
   - **S1**: `NNTR_MOE_HTP_SPLIT=1`. The rate-model check (predicts 45–52
     at G 512).

   Before the E2E cells: the device gtests `MoeM1CpuSplitMatchesDsp` and the
   three arena gtests in both modes. After: one level-2 profile per variant
   at G = 64 (not a tok/s cell, rule 15) for the DSP `mm` of a 2-expert and
   a 1-expert call. MoE dumps (`NNTR_HTP_DUMP`) at G = 64 for A / C / S2 /
   S1, plus the 8-prompt text and `NNTR_PPL_DECODE` for all four. Estimated
   device time ≈ 45 min.

## 5. Risks

| risk | how the handoff shows it |
|---|---|
| **ISS ≠ silicon** on some folded qf32 chain in this TU (§0.3 is not a gate) | `MoeM1CpuSplitMatchesDsp` `bad_elems` names the stage (per-stage fields, as `SWIGLU_DET_FIELD`); the dumps name the first differing call |
| **Subnormals / signed zero** (rules 24, 37; `+0` memset vs the CPU's merge start) | same dumps; the gtest adds an exact-zero row and a \|g\| ≥ 85 row |
| **Cache maintenance wrong** (stale lines → the DSP reads old bytes → plausible wrong text, rule 6's failure mode) | variant C alone: dumps ≡ A, arena gtests `checksum_ok=y` |
| **DSP read rate changes** with the cached attribute (SMMU / IO-coherent path) | C's M==1 `mm` and the ring `engine … GB/s` against A |
| **DRAM contention** worse than #90's probe (in-app ring 0.88 ×, rule 41; CPU −36–44 % under the ring) | `[OP-TIME] moe split=2 cpu_us / dsp_wait_us` and the per-variant `mm`; S1 vs S2 checks the model's slope |
| **DVFS / thermal**: the CPU now works through the 16 ms that used to be idle, so clocks may drop over G = 1024 | mirrored order, thermal checkpoints per block, `freq.log` as in #150; S2's last-64 column vs its all-token column |
| **ARM work outside the MoE slows** (the CPU's caches now hold expert bytes) | OP-TIME per-kind rows of S2 vs A (FC, lm_head GB/s) |
| **Stale skel** | none expected (no DSP change); the handoff still checks the skel md5 = A's and `dspq: on` once |
| **Address space** | unchanged (no new mapping; the chunks are already mapped in the 64-bit ARM process) |
| **Thread-pool fork/join** ≈ 10–20 µs × 3 × 22 per token | OP-TIME `cpu_us` minus the byte time |

## 6. Docs to update

- **BENCHMARK.md**:
  - Results rows for #157 A / C / S2 / S1, each at G 64 / 512 / 1024 with
    prefill and decode (all tokens and last 64).
  - If S2 passes and the user makes it the default: the NPU "now" row
    (Method's bit-identical rule, as #141 Q).
  - An Artifacts entry.
- **LEDGER.md**:
  - §3 ⑫ ("CPU+NPU expert split"): answered, step 1 yes, filed as #157.
  - A new rule from §0.3: *"The M=1 MoE expert path as the skel compiles it
    is IEEE f32 with two roundings and no FMA. Clang lowers `Vsf` ops to
    qf32 with the `.sf` converts folded, and on normal values that equals
    the IEEE sequence (ISS, 2026-09-29, 0 of ≈ 1 M; FMA control 7–36 %
    off). The DSP scatter sums experts in ascending id from +0; a split
    keeps bit identity only as a prefix."* Upgrade it to a silicon rule
    after step 7.
  - A §2 verdict row after the sitting.
- **contract §2** (the "no CPU fallback for `QS4CX_WH`" bullet): one line
  noting that the bit-identical split's CPU experts are not a fallback. This
  is the user's wording to approve.

## Appendix: the step-1 experiment (reproduce in ≈ 4 min)

```
source tools/htp/env.sh; T=$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin; B=$PWD/nntrainer/tensor/htp_backend
# in a scratch dir holding common.h, sim_m1.c, host_check.c below
$T/hexagon-clang -mv79 -mhvx -mhvx-length=128B -G0 -O3 -Wall -I. -I$B/.. -I$B -I$B/hvx -I$B/hmx \
  -I "$HEXKL_ROOT/include" -isystem "$HEXAGON_SDK_ROOT/incs" -isystem "$HEXAGON_SDK_ROOT/incs/stddef" \
  sim_m1.c $B/hvx/hvx_dequant_i32.c $B/hvx/hvx_quant_u8.c $B/hvx/hvx_scale_add_f32.c \
  $B/hvx/hvx_gemm_u8i4_wh.c -lm -o sim_m1.elf
$T/hexagon-sim -mv79 --quiet sim_m1.elf          # ≈ 3 min, writes sim_out.bin
gcc -std=c99 -O2 -ffp-contract=off -Wno-unused-function -I$PWD/nntrainer/tensor \
  -o host_check host_check.c -lm && ./host_check  # the table of §0.3
```

<details><summary><code>common.h</code></summary>

```c
#include <stdint.h>
#define K 2048u
#define I 1792u
#define N 2048u
#define E 4u
#define T 6u
#define NOPS 300000u
static uint32_t lcg_s;
static uint32_t lcg(void){ lcg_s = lcg_s*1664525u + 1013904223u; return lcg_s; }
static float urand(void){ return (float)(lcg() >> 8) * (1.0f/16777216.0f); }
static float grand(void){ return (urand()+urand()+urand()+urand()-2.0f)*1.7f; }
```
</details>

<details><summary><code>sim_m1.c</code> (runs on the ISS; calls the skel's own functions in the M=1 order)</summary>

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static void *memalign_(size_t a, size_t n){ void *p=0; return posix_memalign(&p,a,n)?0:p; }
#define memalign memalign_
#include "common.h"
#include "hvx_dequant_i32.h"
#include "hvx_quant_u8.h"
#include "hvx_scale_add_f32.h"
#include "hvx_gemm_u8i4_wh.h"
void hvx_worker_pool_run(hvx_worker_pool *p, hvx_worker_pool_func f, void *c, uint32_t n){(void)p;(void)n;f(1u,0,c);}
void hvx_worker_pool_wait(hvx_worker_pool *p){(void)p;}
static int csum_col(const uint8_t *wh, uint32_t kt_n, uint32_t n_col, uint32_t col){
  int s=0; uint32_t nt=col/32, c=col%32;
  for(uint32_t kt=0;kt<kt_n;++kt){ const uint8_t *t=wh+((size_t)kt*n_col+nt)*512;
    for(uint32_t r=0;r<32;++r){ uint8_t b=t[(r/8)*128+c*4+(r%4)]; int v=((r%8)<4)?(b&15):(b>>4); if(v>7)v-=16; s+=v; } }
  return s;
}
static void wr(FILE*f,const void*p,size_t n){ fwrite(p,1,n,f); }
int main(void){
  FILE *f=fopen("sim_out.bin","wb"); if(!f){printf("no file\n");return 1;}
  lcg_s = 12345u;
  const uint32_t ktg=K/32, ntg=2*I/32, ktd=I/32, ntd=N/32;
  uint8_t *wgu = memalign(128, (size_t)E*ktg*ntg*512), *wdn = memalign(128,(size_t)E*ktd*ntd*512);
  for(size_t i=0;i<(size_t)E*ktg*ntg*512;++i) wgu[i]=(uint8_t)(lcg()>>24);
  for(size_t i=0;i<(size_t)E*ktd*ntd*512;++i) wdn[i]=(uint8_t)(lcg()>>24);
  static int32_t cs_gu[E][2*I], cs_dn[E][N]; static float ws_gu[E][2*I], ws_dn[E][N], b_gu[E][2*I], b_dn[E][N];
  for(uint32_t e=0;e<E;++e){ for(uint32_t c=0;c<2*I;++c){cs_gu[e][c]=csum_col(wgu+(size_t)e*ktg*ntg*512,ktg,ntg,c); ws_gu[e][c]=(0.5f+urand())*0.004f; b_gu[e][c]=0.0f;}
    for(uint32_t c=0;c<N;++c){cs_dn[e][c]=csum_col(wdn+(size_t)e*ktd*ntd*512,ktd,ntd,c); ws_dn[e][c]=(0.5f+urand())*0.004f; b_dn[e][c]=0.0f;} }
  wr(f,wgu,(size_t)E*ktg*ntg*512); wr(f,wdn,(size_t)E*ktd*ntd*512);
  wr(f,cs_gu,sizeof cs_gu); wr(f,cs_dn,sizeof cs_dn); wr(f,ws_gu,sizeof ws_gu); wr(f,ws_dn,sizeof ws_dn);
  float *x = memalign(128, 4*K*4); uint8_t *ah = memalign(2048, 64*K); float sc[64]; int32_t zp[64];
  int32_t *acc = memalign(128, 2*4*32*4); float *gate = memalign(128, 4*I*4); uint8_t *mid = memalign(2048, 64*I);
  float rqs[64]; int32_t rqz[64]; int32_t *tile = memalign(128, 4*32*4); float *res = memalign(128, N*4); float *out = memalign(128,N*4);
  for(uint32_t t=0;t<T;++t){
    float amp = (t%3==0)?0.3f:(t%3==1?3.0f:30.0f);
    for(uint32_t k=0;k<K;++k) x[k]=grand()*amp*((lcg()%97)==0?8.0f:1.0f);
    if(t==5) for(uint32_t k=0;k<K;++k) if(x[k]<0) x[k]=-x[k]; /* all-positive row: rmin=0 */
    float rw[E]; float sum=0; for(uint32_t e=0;e<E;++e){rw[e]=0.05f+urand(); sum+=rw[e];} for(uint32_t e=0;e<E;++e) rw[e]=rw[e]/sum;
    wr(f,x,K*4); wr(f,rw,sizeof rw);
    hvx_quant_rows_u8_params(x,1,64,K,sc,zp,NULL);
    for(int r=1;r<4;++r){sc[r]=sc[0];zp[r]=zp[0];}
    uint32_t rm[4]={0,0,0,0};
    hvx_quant_pack_u8_ah_rows(x,rm,0,1,K,sc,zp,ah);
    wr(f,sc,4); wr(f,zp,4);
    for(uint32_t kt=0;kt<ktg;++kt) wr(f,ah+(size_t)kt*2048,32);
    memset(out,0,N*4);
    for(uint32_t e=0;e<E;++e){
      const uint8_t *g=wgu+(size_t)e*ktg*ntg*512, *d=wdn+(size_t)e*ktd*ntd*512;
      for(uint32_t j=0;j<I/32;++j){
        hvx_gemm_u8i4_wh_col_nopf(ah,1,ktg,g,ntg,j,1,acc);
        hvx_gemm_u8i4_wh_col_nopf(ah,1,ktg,g,ntg,I/32+j,1,acc+128);
        wr(f,acc,32*4); wr(f,acc+128,32*4);
        hvx_dequant_swiglu_acc_tiles_to_f32((const uint8_t*)acc,512,1,j,32,1,sc,zp,cs_gu[e],ws_gu[e],b_gu[e],I,gate,I,NULL);
      }
      wr(f,gate,I*4);
      memset(gate+I,0,3*I*4);
      hvx_quant_rows_u8_params(gate,1,64,I,rqs,rqz,NULL);
      hvx_quant_pack_u8_ah_rows(gate,NULL,0,4,I,rqs,rqz,mid);
      wr(f,rqs,4); wr(f,rqz,4); for(uint32_t kt=0;kt<ktd;++kt) wr(f,mid+(size_t)kt*2048,32);
      for(uint32_t nt=0;nt<ntd;++nt){
        hvx_gemm_u8i4_wh_col_nopf(mid,1,ktd,d,ntd,nt,1,tile);
        wr(f,tile,32*4);
        hvx_dequant_acc_tile_to_f32(tile,32,1,rqs,rqz,cs_dn[e]+nt*32,ws_dn[e]+nt*32,b_dn[e]+nt*32,res+nt*32,N,0);
      }
      wr(f,res,N*4);
      hvx_scale_add_rows_f32(out,res,rw[e],N);
    }
    wr(f,out,N*4);
    printf("token %u done\n",(unsigned)t);
  }
  float *A=memalign(128,NOPS*4),*B=memalign(128,NOPS*4),*W=memalign(128,NOPS*4);
  for(uint32_t i=0;i<NOPS;++i){
    uint32_t m=i%3;
    if(m==0){ int ea=(int)(lcg()%40)-20, eb=(int)(lcg()%40)-20; uint32_t ba=((uint32_t)(127+ea)<<23)|(lcg()&0x7fffff)|((lcg()&1)<<31); uint32_t bb=((uint32_t)(127+eb)<<23)|(lcg()&0x7fffff)|((lcg()&1)<<31); memcpy(&A[i],&ba,4); memcpy(&B[i],&bb,4); W[i]=0.1f+urand(); }
    else if(m==1){ float a=1.0f+(float)(lcg()%4096)*(1.0f/4096.0f); float b=1.0f+(float)(lcg()%4096)*(1.0f/4096.0f); A[i]=(float)(lcg()%1000)*0.001f; B[i]=a; W[i]=b; }
    else { uint32_t ba=(127u<<23)|(lcg()&0x7fffff); float a; memcpy(&a,&ba,4); A[i]=a; B[i]=ldexpf(1.0f+(float)(lcg()%2)*0.5f, -(int)(lcg()%26)); W[i]=1.0f; }
  }
  wr(f,A,NOPS*4); wr(f,B,NOPS*4); wr(f,W,NOPS*4);
  for(uint32_t i=0;i<NOPS;i+=32){ float w=W[i]; hvx_scale_add_rows_f32(A+i,B+i,w,32);}
  wr(f,A,NOPS*4);
  for(uint32_t i=0;i<NOPS;++i){ uint32_t ba=((uint32_t)(100+lcg()%60)<<23)|(lcg()&0x7fffff); uint32_t bb=((uint32_t)(100+lcg()%60)<<23)|(lcg()&0x7fffff); memcpy(&A[i],&ba,4); memcpy(&B[i],&bb,4);}
  wr(f,A,NOPS*4); wr(f,B,NOPS*4);
  for(uint32_t i=0;i<NOPS;++i){ volatile float a=A[i], b=B[i]; W[i]=a/b; }
  wr(f,W,NOPS*4);
  fclose(f); printf("done\n"); return 0;
}
```
</details>

<details><summary><code>host_check.c</code> (the IEEE spec of §0.1; this is what <code>moe_m1_det.h</code> must compute)</summary>

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "swiglu_det.h"
#include "common.h"
static FILE *f;
static void rd(void *p, size_t n){ if(fread(p,1,n,f)!=n){fprintf(stderr,"short read\n");exit(2);} }
static float vmul(float a,float b){volatile float r=a*b;return r;}
static float vadd(float a,float b){volatile float r=a+b;return r;}
static float vsub(float a,float b){volatile float r=a-b;return r;}
static float vdiv(float a,float b){volatile float r=a/b;return r;}
static int32_t rne(float v){ volatile float b=v+12582912.0f; int32_t i; memcpy(&i,(const void*)&b,4); return i-0x4B400000; }
static void qparams(const float *x,uint32_t n,float *s,int32_t *z){
  float mn=x[0],mx=x[0]; for(uint32_t i=1;i<n;++i){ if(x[i]<mn)mn=x[i]; if(x[i]>mx)mx=x[i]; }
  float rmin=mn<0.0f?mn:0.0f, rmax=mx>0.0f?mx:0.0f;
  if(rmin==rmax){*s=1.0f;*z=0;return;}
  float sc=vdiv(vsub(rmax,rmin),255.0f); int32_t zz=(int32_t)nearbyintf(vdiv(-rmin,sc)); if(zz<0)zz=0; if(zz>255)zz=255; *s=sc;*z=zz;
}
static void quant(const float *x,uint32_t n,float s,int32_t z,uint8_t *u){
  float inv=vdiv(1.0f,s);
  for(uint32_t i=0;i<n;++i){ int32_t q=rne(vmul(x[i],inv))+z; if(q<0)q=0; if(q>255)q=255; u[i]=(uint8_t)q; }
}
static int nib(const uint8_t *wh,uint32_t n_col,uint32_t k,uint32_t col){
  uint32_t kt=k/32,r=k%32,nt=col/32,c=col%32; uint8_t b=wh[((size_t)kt*n_col+nt)*512+(r/8)*128+c*4+(r%4)];
  int v=((r%8)<4)?(b&15):(b>>4); return v>7?v-16:v;
}
static float dq(int32_t acc,float s,int32_t zp,int32_t cs,float ws,float b){
  float corr=vsub((float)acc, vmul((float)zp,(float)cs)); return vadd(vmul(vmul(corr,s),ws),b);
}
static uint32_t fb(float x){uint32_t u;memcpy(&u,&x,4);return u;}
static long bad[16]; static long cnt[16]; static float gmax;
#define CMPF(stage,a,b) do{ cnt[stage]++; if(fb(a)!=fb(b)){ if(bad[stage]<3) printf("  stage %d mismatch sim %08x host %08x\n",stage,fb(a),fb(b)); bad[stage]++; } }while(0)
#define CMPI(stage,a,b) do{ cnt[stage]++; if((a)!=(b)){ if(bad[stage]<3) printf("  stage %d mismatch sim %ld host %ld\n",stage,(long)(a),(long)(b)); bad[stage]++; } }while(0)
int main(void){
  f=fopen("sim_out.bin","rb");
  const uint32_t ktg=K/32,ntg=2*I/32,ktd=I/32,ntd=N/32;
  size_t gb=(size_t)E*ktg*ntg*512, db=(size_t)E*ktd*ntd*512;
  uint8_t *wgu=malloc(gb),*wdn=malloc(db); rd(wgu,gb); rd(wdn,db);
  static int32_t cs_gu[E][2*I],cs_dn[E][N]; static float ws_gu[E][2*I],ws_dn[E][N];
  rd(cs_gu,sizeof cs_gu); rd(cs_dn,sizeof cs_dn); rd(ws_gu,sizeof ws_gu); rd(ws_dn,sizeof ws_dn);
  static float x[K],rw[E],gate_s[I],gate_h[I],res_s[N],res_h[N],out_h[N],out_s[N],out_c[N];
  static uint8_t u_s[K],u_h[K],mid_s[I],mid_h[I]; static int32_t acc_s[2*I],accd_s[N];
  for(uint32_t t=0;t<T;++t){
    rd(x,K*4); rd(rw,sizeof rw); float sc_s; int32_t zp_s; rd(&sc_s,4); rd(&zp_s,4);
    for(uint32_t kt=0;kt<ktg;++kt) rd(u_s+kt*32,32);
    float sc_h; int32_t zp_h; qparams(x,K,&sc_h,&zp_h); CMPF(0,sc_s,sc_h); CMPI(0,zp_s,zp_h);
    quant(x,K,sc_h,zp_h,u_h); for(uint32_t k=0;k<K;++k) CMPI(1,u_s[k],u_h[k]);
    memset(out_h,0,sizeof out_h); memset(out_c,0,sizeof out_c);
    for(uint32_t e=0;e<E;++e){
      const uint8_t *g=wgu+(size_t)e*ktg*ntg*512,*d=wdn+(size_t)e*ktd*ntd*512;
      for(uint32_t j=0;j<I/32;++j){ rd(acc_s+j*32,128); rd(acc_s+I+j*32,128); }
      for(uint32_t c=0;c<2*I;++c){ int32_t a=0; for(uint32_t k=0;k<K;++k) a+=(int32_t)u_h[k]*nib(g,ntg,k,c); CMPI(2,acc_s[c],a); }
      rd(gate_s,I*4);
      for(uint32_t c=0;c<I;++c){
        float gg=dq(acc_s[c],sc_h,zp_h,cs_gu[e][c],ws_gu[e][c],0.0f), uu=dq(acc_s[I+c],sc_h,zp_h,cs_gu[e][I+c],ws_gu[e][I+c],0.0f);
        gate_h[c]=swiglu_det_one(gg,uu);
        if(fabsf(gg)>gmax) gmax=fabsf(gg);
        CMPF(3,gate_s[c],gate_h[c]); }
      float rqs_s; int32_t rqz_s; rd(&rqs_s,4); rd(&rqz_s,4); for(uint32_t kt=0;kt<ktd;++kt) rd(mid_s+kt*32,32);
      float rqs_h; int32_t rqz_h; qparams(gate_h,I,&rqs_h,&rqz_h); CMPF(4,rqs_s,rqs_h); CMPI(4,rqz_s,rqz_h);
      quant(gate_h,I,rqs_h,rqz_h,mid_h); for(uint32_t k=0;k<I;++k) CMPI(5,mid_s[k],mid_h[k]);
      for(uint32_t nt=0;nt<ntd;++nt) rd(accd_s+nt*32,128);
      for(uint32_t c=0;c<N;++c){ int32_t a=0; for(uint32_t k=0;k<I;++k) a+=(int32_t)mid_h[k]*nib(d,ntd,k,c); CMPI(6,accd_s[c],a);
        res_h[c]=dq(a,rqs_h,rqz_h,cs_dn[e][c],ws_dn[e][c],0.0f); }
      rd(res_s,N*4); for(uint32_t c=0;c<N;++c) CMPF(7,res_s[c],res_h[c]);
      for(uint32_t c=0;c<N;++c){ out_h[c]=vadd(out_h[c],vmul(res_h[c],rw[e])); out_c[c]=fmaf(res_h[c],rw[e],out_c[c]); }
    }
    rd(out_s,N*4); for(uint32_t c=0;c<N;++c){ CMPF(8,out_s[c],out_h[c]); cnt[9]++; if(fb(out_s[c])!=fb(out_c[c])) bad[9]++; }
    printf("token %u: act scale=%g zp=%d max|gate|=%g\n",t,sc_h,zp_h,gmax);
  }
  static float A[NOPS],B[NOPS],W[NOPS],R[NOPS];
  rd(A,sizeof A); rd(B,sizeof B); rd(W,sizeof W); rd(R,sizeof R);
  for(uint32_t i=0;i<NOPS;++i){ float w=W[i-(i%32)]; float h=vadd(A[i],vmul(B[i],w)); CMPF(10,R[i],h); cnt[11]++; if(fb(R[i])!=fb(fmaf(B[i],w,A[i]))) bad[11]++; }
  rd(A,sizeof A); rd(B,sizeof B); rd(R,sizeof R);
  for(uint32_t i=0;i<NOPS;++i) CMPF(12,R[i],vdiv(A[i],B[i]));
  const char *nm[]={"act qparams(s,zp)","act u8","gate_up int32","dq+swiglu f32","requant params","mid u8","down int32","down dq f32","out row (IEEE mul,add)","out row vs FMA (control)","scale_add sweep","scale_add sweep vs FMA (control)","scalar f32 divide"};
  for(int s=0;s<13;++s) printf("%-34s bad %ld of %ld\n",nm[s],bad[s],cnt[s]);
  return 0;
}
```
</details>
