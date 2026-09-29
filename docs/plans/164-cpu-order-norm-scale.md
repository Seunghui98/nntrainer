# 164 — Bit-identical HTP RMSNORM / QK_NORM: the row scale in the Android CPU's order

Issue: dlwlzzero/nntrainer#164 (p1). Read against `htp_moe` @ `c95c0feb`
(2026-09-29). The user's direction (2026-09-29) is that decode runs end to end
on the NPU and the bit-preserving rule stays (contract §12, 2026-09-28; LEDGER
rule 39 consequence (2)). There is no PPL route. This plan follows #152's
approach (`docs/plans/152-resident-accuracy.md`): the spec is rewritten to the
Android CPU's exact order, the kernel reproduces it, and it is gated on ARM, on
the host and on the DSP.

## 0. Step 2 of the issue, done on the workstation: **proved**

**Inputs.** The inputs are the 2026-09-29 norm-shadow dumps
(`/local/mnt/workspace/htp_moe/norm/d_f_*/norm.bin`, `dev/norm-shadow` @
`5ead9b7f`). The true gamma comes from `hf/model.safetensors` (bf16 → f32):

* the 49 RMSNORM calls in order `layers.{i}.operator_norm`, `layers.{i}.ffn_norm`
  for i = 0 … 23, then `embedding_norm`;
* per attention layer, `q_layernorm` | `k_layernorm` (64 each).

`eps = 1e-5f`.

**The CPU order**, from the aarch64 disassembly of the shipped set's
`libnntrainer.so` (`nntrainer::neon::rms_norm_wrt_width_fp32_intrinsic` @
`0x3712a4`, NDK r30, app built with `-ffast-math`). It is the source's order;
nothing was reassociated or approximated.

1. Four `float32x4` accumulators. `acc_k` lane `l` runs a **fused** `fmla`
   chain over `x[16i + 4k + l]`, i = 0 … W/16 − 1. That is 16 independent
   FMA chains, 128 deep at W = 2048 and 4 deep at W = 64.
2. `faddp v,v,v` then `faddp s`, which gives `h_k = (l0 + l1) + (l2 + l3)`.
3. `s = ((h0 + h1) + h2) + h3`. There is no 4-lane or scalar tail at W = 2048 or 64.
4. `mean = s / W` (`fdiv`), then `d = eps + mean` (`fadd`).
5. `q = fsqrt(d)`, then `r = 1.0f / q` (`fdiv`). These are **two correctly
   rounded steps**. There is no `frsqrte`, and r is not the correctly rounded
   1/sqrt.
6. `y = x·r` (`fmul` by lane), then `multiply_i(gamma)`, so y = fl(fl(x·r)·g).

**Today's HTP spec** (`m1_ops_det.h:35-47`) differs at steps 1–5. It uses a
32-lane non-fused sum, a 5-step tree, and a magic-seed rsqrt with three
Newton steps. Step 6 is the same.

**Result** (`step2.c` in the planner's scratchpad, host, `-ffp-contract=off`):

| check | f_R | f_RQ | f_A (tag 2) | f_Q |
|---|---|---|---|---|
| CPU-order emulation (above, true gamma) == CPU dump, RMSNORM rows (49 calls × 8 steps) | 392/392 | 392/392 | 392/392 | — |
| today's spec `m1_rmsnorm_det` (true gamma) == **HTP dump** | 392/392 | 392/392 | — | — |
| **HTP elementwise stage with r := CPU r** == CPU dump (2048/2048 each) | **392/392** | **392/392** | — | — |
| r_HTP − r_CPU in ulp (−3 … +3) | 1 / 18 / 98 / **185** / 79 / 11 / 0 | 0 / 18 / 80 / **195** / 80 / 16 / 3 | | |
| QK_NORM tag 1: CPU-order emulation at W = 64 == CPU normed q\|k (48 records × 40 heads) | 48/48 | 48/48 | 48/48 | 48/48 |
| QK_NORM heads where today's spec's r ≠ the CPU's r | 923/1920 | 925/1920 | 963/1920 | 961/1920 |

Conclusions:

* The whole RMSNORM difference is the one scalar r. The issue's "88–97 %
  reproduced" was an artefact of back-solving gamma. With the true gamma the
  reproduction is 100 %.
* The device kernel equals its spec on every dumped row, so rule 37's
  emulation gap does not apply here: these are normal-range rows.
* On QK_NORM, about half of the heads carry a different r. They rarely show
  downstream because RoPE rounds q and k to fp16 first (#152), which absorbs a
  1–3-ulp f32 change unless an element sits near an fp16 boundary. That is
  what happened at pos 513, layer 4, head 3.

**The method works on the Hexagon ISA.** This was exploration only; contract
§12 (2026-09-21) keeps the simulator out of the gates. `hexagon-sim -mv79`
(tools 19.0.04) ran a prototype:

* the scalar-core `sffma` for the 16 chains;
* integer correctly rounded sqrt and reciprocal;
* HVX `Vsf` for y = (x·r)·g.

It reproduced the Android CPU output on **784/784** RMSNORM rows (f_R + f_RQ)
and **3840/3840** QK heads (f_Q + f_RQ), and two subnormal-sum probes equal
glibc `fmaf` bit for bit. The integer `sqrt_rn` / `recip_rn` were checked
**exhaustively** on the host:

* `sqrt_rn` against `sqrtf` over all 2 130 706 432 positive normal floats:
  `bad=0`;
* `recip_rn` against `1.0f/x` over every positive normal x whose reciprocal
  is normal: `bad=0`.

## 1. Goal and gate

**Goal.** Resident RMSNORM and QK_NORM produce the Android CPU's bits on every
decode call. The masks `MOE,RMSNORM` and `MOE,QK_NORM,ROPE,ATTN_M1` then
become bit-identical to A.

| # | check | where | pass |
|---|---|---|---|
| G0 | spec == the shipped CPU function | new ARM gtest `RmsNormCpuOrder.*` in `unittest_nntrainer_cpu_backend` (calls the exported `nntrainer::rms_norm_wrt_width_fp32_intrinsic` + gamma multiply of the set's `libnntrainer.so`) | `bad=0` at W = 2048 (1 row) and W = 64 (40 rows), 20 000 random rows over magnitudes 2⁻²⁰ … 2²⁰, plus the ±1e-39 row and the replayed 2026-09-29 rows (§4 step 3) |
| G1 | kernel == spec on silicon | `unittest_hvx_softmax --gtest_filter='HvxM1Ops.*'` | `RmsnormMatchesDetBitExact` / `QkNormMatchesDetBitExact` `bad_y=0 bad_rs=0` on every normal-range kind. The subnormal kind (rule 37, #137) is reported, see step 5 |
| G2 | **norm-shadow: HTP == CPU on every call and step** | rebased `dev/norm-shadow` (§4 step 6), prompt 512, G = 8, forced on A's tokens | tag 0: 392/392 records bit-equal in f_R and f_RQ. Tag 1 (now carrying the DSP's normed q\|k): 48/48 records × 40 heads bit-equal in f_Q and f_RQ. Logits equal to A on all 8 steps in f_R, f_Q, f_RQ |
| G3 | MoE dumps vs A | `NNTR_HTP_DUMP`, G = 4, `tools/htp/htp_dump_eval.py A1 <run>` | `bit_identical=1` for `KINDS=MOE,RMSNORM` and `KINDS=MOE,QK_NORM,ROPE,ATTN_M1` (null check A1 vs A2 = 1) |
| G4 | decode nll and text | `NNTR_PPL_DECODE` lines; 8-prompt set at G = 256 (`docs/measurements/prompts/`) | every `[PPL] decode step=… nll=…` line of R, Q, RQ equals A's to all 17 digits; text ≡ A 8/8 for RQ (`cmp` after the banner strip of measurement 152 §5) |
| G5 | cost | `NNTR_HTP_PROFILE=2` `pcyc/op` line of the RQ run | `RMSNORM ≤ 4 000` and `QK_NORM ≤ 25 000` pcyc/op, i.e. ≤ 0.35 M pcycles/token for 49 + 6 calls (ISS estimate 0.22 M, §3.3) |
| standing | prefill; text | every speed cell | prefill ≥ −5 % of A (mirrored band). Text ≡ the CPU (A) run: G4 |

The issue closes on G0–G5. Decode tok/s of R / Q / RQ is recorded but is not
a gate. It is bound by the resident path's call count (R: `calls/token=71.00`
in `logs/g_R.log`), not by these kernels (§3.3). That count is #162's lever.

## 2. Where it lives

**Changes.**

* `nntrainer/tensor/m1_ops_det.h`:
  * the contract block `:25-47` and the accuracy note `:92-96` now allow one
    fused multiply-add (`fmaf`, IEEE) and integer sqrt / reciprocal. Still no
    libm sqrt and no division;
  * `m1_rsqrt_det` `:143` and `m1_sumsq_det` `:163` are replaced by
    `m1_sumsq_cpu_det` (16 fused chains, the `faddp` order),
    `m1_sqrt_rn_det` and `m1_recip_rn_det` (integers, Appendix);
  * `m1_rmsnorm_chunk_det` `:187` chains them;
  * `m1_rmsnorm_det` `:205` keeps its signature and row-scale semantics.
* `nntrainer/tensor/htp_backend/hvx/hvx_m1_ops_f32.c`:
  * `hvx_rsqrt_det_sf` `:42` and `hvx_reduce_add_sf` `:59` go (no other
    user);
  * in `hvx_rmsnorm_f32` `:68-99` the HVX accumulate / reduce / rsqrt is
    replaced by a scalar r (16 register accumulators through
    `__builtin_HEXAGON_F2_sffma`, which is `fmaf` under `hvx_emu`, plus the
    spec's two integer helpers), splatted;
  * the elementwise loop `:95-97` is unchanged;
  * the file's header comment `:12-23` is updated;
  * `hvx_m1_ops_f32.h:37-47` keeps its signature and documents the order.
* `test/htp/host/m1_ops_host_check.c`:
  * `check_rmsnorm` `:152` gains an independent CPU-order reference written
    from the disassembly (`fmaf`, `sqrtf`, `1.0f/`);
  * strided and edge checks of `m1_sqrt_rn_det` / `m1_recip_rn_det`;
  * four mutants;
  * an optional `--replay <norm.bin> <gamma_rms.f32> <gamma_qk.f32>` mode.
* `test/unittest/unittest_hvx_softmax.cpp:360-464`: the tests already compare
  the device kernel with `m1_rmsnorm_det`, so they follow the spec with no
  edit. `:416` gains the magnitude sweep of G0.
* `test/unittest/unittest_nntrainer_cpu_backend.cpp`: new
  `RmsNormCpuOrder.*` (G0). It is built for Android by `test/jni/Android.mk:763`,
  and on non-aarch64 it calls `GTEST_SKIP` because the host order is AVX2.

**Consumers checked; none moves.**

* **IDL** `test/htp/nntr_hvx.idl:530-538` (`rmsnorm_det_f32`, `row_scale`)
  and `nntr_hvx_small_ops.c:36-55`: same shapes, and `row_scale` is still r.
  No stub regeneration (`generate_stub.sh`) is needed.
* **Graph** `hexkl_graph.c:111-139`: `graph_op_rmsnorm` / `graph_op_qk_norm`
  call the same function with the same arguments. The validator already
  requires a positive normal eps (`htp_graph_desc.h:442-446`), so d is
  always normal. `K % 32` (`:448-450`) implies the CPU loop's `% 16`.
* **`HtpComputeOps`** `htp_compute_ops.cpp:1489-1535` (eps check, gamma
  binding): unchanged. The `NNTR_HTP_PROFILE` graph line `:796-812` already
  prints `pcyc/op` per kind (G5). There is no new stage, so neither the
  stage tables nor `tools/htp_fc_report.py` changes.
* **Quantizer / loader**: `nntr_quantize_stream`'s format tag and the loader
  check are untouched (no weight format change).
* **CPU layers**: `rms_norm.cpp:80-90` and `qkv_layer.cpp:170-184, :278`.
  The CPU path is the reference and is not touched.
* **Host E2E** (`run_inproc_e2e.sh`): the switch-off goldens are unchanged.
  The `fwd` SNR lines compare with the x86 AVX2 order, so they can move
  slightly. Record them against the 30 dB floor and do not re-bless goldens.

## 3. Design

### 3.1 The spec

The spec is §0's CPU order, written as one f32 operation per step (volatile
store helpers). One step is fused: `acc = fmaf(x, x, acc)` over the 16
chains. The spec's `fmaf` is IEEE on every platform that compiles it: glibc,
aarch64 `fmadd`, and Hexagon `fmaf` = `r2 += sfmpy(r0,r1)`, disassembled.

* `mean = s · (1/W)` equals the CPU's `s / W` for a power-of-two W, because
  both are the correctly rounded value of the same real number.
* `q = m1_sqrt_rn_det(d)` and `r = m1_recip_rn_det(q)` are integer-only and
  give exactly RN(√d) and RN(1/q). They do not depend on compiler flags or
  libm.
* Specials, for completeness (the CPU's IEEE results):
  * d = +inf (sum overflow): q = inf, r = +0;
  * NaN: NaN. NaN payloads are out of domain.

### 3.2 The kernel

* Per chunk, the calling thread computes r on the scalar core: 16 `sffma`
  chains held in registers, the 7 adds, one multiply, one add, then the two
  integer helpers.
* HVX then runs the existing `(x·r)·g` loop.
* No VTCM, no pool, no heap: the only new memory is 16 floats in registers.
  The arena and the 32-bit address space are unchanged, and there is no CPU
  fallback. On a v79 skel the recip helper's 64-bit divide links as
  `__hexagon_udivdi3`, which `build.sh:126`'s guard allows.

**Rejected: an exact FMA on HVX.** HVX has no sf FMA. A Dekker split plus a
correctly rounded three-term sum (round-to-odd emulated, as #152's `fma16`)
costs about 20 vector ops per step. The step count is 128 sequential steps per
chain, and the 16 chains fill only half a vector. That is several times
today's whole kernel, and it needs a new exactness proof. The scalar `sffma`
is the hardware's own single-rounding op, already matched bit for bit on 784
rows in the ISS.

Also not taken: norms back on the CPU (against the direction), and a PPL gate
(the user kept the bit-preserving rule).

### 3.3 Cost

These are ISS pcycles (`hexagon-sim -mv79 --timing`, a REP=1 vs REP=101
difference, `-O3`, the prototype of §0). They are an estimate, not a device
number.

| per call | today's kernel | new (prototype) | Δ |
|---|---|---|---|
| RMSNORM, K = 2048 | 2 101 | 2 311 (r: 1 636) | +210 |
| QK_NORM, 32 + 8 heads × 64 | 9 170 | 17 181 (r: ≈ 425 / head) | +8 011 |
| **per token** (49 RMSNORM + 6 QK_NORM) | 158 k | 216 k | **+58 k** |

+58 k pcycles is 29–58 µs at 1–2 GHz, which is 0.15–0.3 % of a 20 ms token.
The QK term is mostly the bit-loop sqrt and the soft 64-bit divide.

Speed ladder if G5 fails:

* (a) Seed the sqrt from today's `m1_rsqrt_det(d)·d` and the reciprocal from
  `sfrecipa`. Then make one exact integer correction: a 32×32→64 product
  compared with the midpoint. That is ≈ 30 cycles each, bit-identical by
  construction, and the exhaustive host check covers it.
* (b) Stop and report.

## 4. Steps

1. **Spec + host check.** Rewrite `m1_ops_det.h` (§3.1). In
   `m1_ops_host_check.c`, add:
   * the independent CPU-order reference (`fmaf` / `sqrtf` / `1.0f/`);
   * `sqrt_rn` / `recip_rn` checks strided over every 251st positive normal,
     plus ±4 ulp around every power of two and every exponent edge;
   * mutants that must be caught:
     * the old 32-lane tree sum;
     * a non-fused multiply-add;
     * `m1_rsqrt_det` in place of sqrt/recip;
     * the reduction order `h0 + (h1 + (h2 + h3))`.

   Run the exhaustive form once (≈ 60 s, `M1_NORM_EXHAUSTIVE=1`) and paste
   its line into the PR.
   Gate: rung 1. `run_host_checks.sh` prints `ALL CHECKS PASS`, with new lines
   `M1 OPS NORM CPU-ORDER OK mutants=4/4` and `M1 OPS SQRT/RECIP RN OK`.
2. **Kernel** (§3.2) on `hvx_emu`.
   Gate: `M1 OPS BIT-IDENTICAL` and `graph_host_check` pass (its RMSNORM /
   QK_NORM references follow the spec). Then rung 2: `test/htp/build.sh`
   prints `UNDEFINED SYMBOLS OK`. Record the skel md5.
3. **Replay on the host.** Run `m1_ops_host_check --replay` on the 2026-09-29
   dumps with the true gammas. Extract the gammas with a 20-line script, kept
   on `dev/norm-shadow`, never merged, as §0 did.
   Gate: `REPLAY rms=784/784 qk_heads=7680/7680` (kernel on `hvx_emu` == CPU
   dump). This step is the host's version of G2.
4. **G0 gtest** `RmsNormCpuOrder.MatchesNeon` (§1). It optionally replays
   `NNTR_NORM_REPLAY=<norm.bin>` with gamma files pushed next to it.
   Gate: builds in rung 3. It runs in the sitting.
5. **Subnormal rows (rule 37).** G1's kind 2 (the ±1e-39 row) runs the new
   scalar r and the unchanged HVX multiply. If the sitting shows `bad_y > 0`
   there while normal kinds pass, the follow-up puts rows whose
   `min |x| < 2⁻¹²⁶` through a scalar `(x·r)·g`. That is flagged by one HVX
   exponent compare per vector, and bit-identity is unaffected by
   construction. Do **not** build it before the device says so: real hidden
   states in the dumps have no subnormals.
6. **Measurement branch** (`dev/norm-shadow`, measurement only, never
   merged). Rebase it on the PR branch. For tag 1, fill `other` with the
   **DSP's** normed q|k:
   * call the `rmsnorm_det_f32` test entry on the same session with chunk 64:
     q heads with the q gamma, then k heads with the k gamma. This is the same
     compiled `hvx_rmsnorm_f32` with the same arguments `graph_op_qk_norm`
     uses (`hexkl_graph.c:134-135`);
   * add a dev-only `HtpComputeOps` wrapper for that call;
   * `row_scale` also gives r per head.

   This makes the HTP's q/k norm observable per head. The in-graph binding is
   then checked downstream by G2's logits and G3.
   Gate: rung 3 of that branch. On the host (`-Dhtp-inproc`), a short
   `NNTR_NORM_SHADOW` run on the hd64 fixture writes tag-1 records with a
   non-zero `other`.
7. **Rung 3** for the PR set and the shadow set. Run `llvm-objdump` on the
   new `libnntrainer.so`: `neon::rms_norm_wrt_width_fp32_intrinsic` still
   shows 4 × `fmla .4s`, `faddp`, `fdiv`, `fsqrt`, `fdiv` (§0), because G0's
   premise is the shipped code.
8. **Device sitting (unavoidable; the orchestrator runs it).** Write
   `docs/measurements/164-cpu-order-norm.md`, about 55 min, with a thermal log
   at each checkpoint. One set: the shadow branch = the PR diff plus the inert
   measurement commit. Its md5s go next to the PR-only md5s, the arrangement
   of 2026-09-29. **Variants (4):** A = switch off (the unchanged reference;
   the changed kernels are not reached); R = `MOE,RMSNORM`;
   Q = `MOE,QK_NORM,ROPE,ATTN_M1`; RQ = `MOE,RMSNORM,QK_NORM,ROPE,ATTN_M1`.
   All use `NNTR_HTP_FORWARD=1`, and every switch-on log shows its
   `graph: init … resident=` banner.
   * (a) gtests G0, G1.
   * (b) Norm shadow (G2): prompt 512, G = 8, forced on A's tokens,
     `NNTR_NORM_SHADOW` + `NNTR_LOGIT_SHADOW` for A, R, Q, RQ. A's
     `logits.bin` must also equal the 2026-09-29 A's bit for bit, which is a
     cross-sitting determinism check that ignores thermal state.
   * (c) Dumps at G = 4 (G3): A1, A2, R, Q.
   * (d) Speed: full E2E, prompt 512, G = 64 / 512 / 1024 × 2, mirrored
     `A R Q RQ | RQ Q R A`. Plus one `NNTR_HTP_PROFILE=2` RQ run at G = 64
     (G5).
   * (e) nll + text (G4): `NNTR_PPL_DECODE` on the 8 prompts at G = 256 for A
     (self, writes `cont_<p>.ids`) and forced for R, Q, RQ. The texts of A and
     RQ are compared with `cmp`, and `tools/htp/loop_check.py` is recorded.

   Stop rules:
   * G1 fails on normal rows: stale skel (`0x8000040e`) or silicon mismatch;
     stop before (b).
   * G0 fails: the CPU is not what §0 read; stop and re-read the disassembly
     of that set.
9. **Fold.** G0–G5 ✓ closes the issue. Otherwise use the ladder in §3.3 for
   G5 and step 5 for rule 37.

## 5. Risks (host vs device)

* **ISS ≠ silicon on tiny values** (rule 37). The r path is scalar `sffma`,
  checked on the ISS and on the host only. G1's subnormal kind and G0's
  ±1e-39 row expose it, and step 5 is the prepared answer.
* **ARM codegen.** A different NDK or flags (`-ffast-math` + `-mrecip`) could
  turn `1/sqrtf` into `frsqrte` steps. Step 7's objdump and G0 catch that
  before E2E.
* **Cost only known on silicon.** The ISS pcycles ignore DDR latency. The
  kernels read ≤ 8 KiB per call from slots that are already hot. G5 reads
  `pcyc/op` in the same skel as E2E. Report pcycles, not µs.
* **DVFS / thermal drift.** Speed cells are compared only inside the sitting,
  in mirrored order with zone0 logged. The bit checks (G2–G4) are immune, and
  (b)'s cross-sitting logits equality is a free determinism check.
* **Stale skel.** Every log carries the skel md5. `0x8000040e` means a stale
  skel. The IDL is unchanged, so an old stub still binds, which is why the md5
  is the only guard.
* **Address space / DMA.** Untouched: no new buffers, and no weight or DMA
  path is involved.
* **Speed expectation.** R / Q / RQ decode stay far under A: 40.9 / 30.2
  vs 52.9 tok/s on 2026-09-29, because of 71+ FastRPC calls per token. This
  plan makes those masks exact, not fast. #162 owns the call count.

## 6. Docs to update

* **`docs/htp_moe/BENCHMARK.md`**:
  * Results: #164 rows for A / R / Q / RQ (G 64 / 512 / 1024, prefill,
    calls/token, text, nll equal y/n);
  * a side table with G0 / G1 lines, G2 record counts, G3 `bit_identical`
    and G5 `pcyc/op`.
* **`docs/htp_moe/LEDGER.md`**:
  * rule 39 amended: RMSNORM's "131–148 dB, DSP nearer to f64" was a
    different r order, not an amplifier-only effect. With the CPU order, the
    norm kinds are exact;
  * a new rule, if the device confirms it: *the Android CPU's RMSNorm scale is
    RN(1/RN(√(((h0+h1)+h2)+h3)/W + eps))) over 16 fused `fmla` chains; a
    resident norm matches only with exactly that r; QK_NORM's r errors hide
    behind the fp16 RoPE rounding until a boundary element flips*;
  * §2: the #164 verdict row;
  * ⑨: RMSNORM / QK_NORM exact.

## Appendix — the integer helpers (verified exhaustively on the host)

```c
/* RN(sqrt(d)), d a positive normal. d = m*2^ee; M = m << s in [2^48, 2^50),
 * ee - s even; q = floor(sqrt(M)) has 25 bits; no ties (M is not an odd
 * square), so RN = (q + 1) >> 1. */
uint32_t u = bits(d); int ee = (int)(u >> 23) - 150;
uint64_t m = (u & 0x7fffff) | 0x800000; int s = ((ee - 25) & 1) ? 26 : 25;
uint64_t rem = m << s, q = 0, bit = 1ull << 48;
while (bit) { if (rem >= q + bit) { rem -= q + bit; q = (q >> 1) + bit; }
              else q >>= 1; bit >>= 2; }
uint32_t r = (uint32_t)((q + 1) >> 1); int E = (ee - s) / 2 + 1;
if (r == 1u << 24) { r >>= 1; ++E; }
return from_bits(((uint32_t)(E + 150) << 23) | (r & 0x7fffff));

/* RN(1/q), q a positive normal with a normal reciprocal. Exact for a power
 * of two; else Q = floor(2^48/mq) has 25 bits, no ties, RN = (Q + 1) >> 1. */
uint32_t u = bits(q); int Eq = (int)(u >> 23) - 150;
uint64_t mq = (u & 0x7fffff) | 0x800000;
if (mq == 0x800000) return from_bits((uint32_t)(-Eq - 23 + 127) << 23);
uint64_t Q = (1ull << 48) / mq; uint32_t r = (uint32_t)((Q + 1) >> 1);
int E = -47 - Eq; if (r == 1u << 24) { r >>= 1; ++E; }
return from_bits(((uint32_t)(E + 150) << 23) | (r & 0x7fffff));
```
