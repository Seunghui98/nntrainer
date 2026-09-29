# 152 — Accuracy of the resident per-token path: a CPU-exact attention stretch

Issue: dlwlzzero/nntrainer#152 (p1, track (c), part of #76). Read against
`htp_moe` @ `df17fcdb` (2026-09-29). Contract `0001` §1 accuracy gate as
amended 2026-09-28; LEDGER rules 37–40, ⑨, ㉗; measurements
`136-forward-text.md` (dump sitting, kernel-level check) and
`134-132-combined.md`.

**Correction to the issue body, measured before planning.** Lever 1 as filed
("store the DSP's decode K/V rows fp16-rounded") moves nothing. On Android the
CPU attention is **fp16 from end to end**, not only its cache. The
`#if ENABLE_FP16 && defined(__ANDROID__)` branch (`Applications/CausalLM/layers/mha_core.cpp:530-559`)
rounds q, k and v to fp16 and runs RoPE, the scores, the softmax and PV in
fp16 arithmetic. At the dumped first decode token (pos 512), the DSP cache
holds the CPU's 512 fp16 prefill rows plus one row of its own. So the K/V
rounding touches 1 of 513 rows. Host replay of #136's `dump_attn` (all six
attention layers, pos 512). Each column is the SNR of an attention output
against an emulation of the Android CPU path: numpy float16, fused FMA
through float64, libm exp.

| layer | today (f32 DSP) | + K/V row fp16 (lever 1 as filed) | + q/k fp16 RoPE | fp16, non-fused `hf` ops | fp16, FMA double-rounded via f32 (outputs ≠ / 2048) | fp16, FMA exact |
|---|---|---|---|---|---|---|
| 0 | 41.0 | 41.0 | 41.1 | 52.7 | 84.0 (62) | ∞ |
| 1 | 43.2 | 43.2 | 43.3 | 55.0 | 82.1 (27) | ∞ |
| 2 | 51.9 | 52.0 | 52.1 | 56.5 | 95.0 (3) | ∞ |
| 3 | 51.7 | 51.7 | 51.8 | 56.4 | 88.8 (11) | ∞ |
| 4 | 48.0 | 48.0 | 48.1 | 55.5 | 89.9 (10) | ∞ |
| 5 | 42.1 | 42.1 | 42.0 | 56.5 | 86.6 (20) | ∞ |

Sanity checks: the f32 reference reproduces the dumped DSP output at
121–129 dB, which confirms the layout and the RoPE table. Layer 0's 41.0 dB
agrees with #136's 40.2 dB at the first MoE input after it.

Per stage, taking fp16 for one stage only gives: scores 41.5, softmax 46.6,
PV 40.6. Taking it for two stages gives 48.4 / 51.0 / 41.0. **Only the whole
fp16 sequence, fused FMAs included, closes the gap.**

This plan therefore makes the resident ROPE + ATTN_M1 stretch reproduce the
Android CPU's attention bit for bit. It is the issue's lever 1 done at every
rounding point, not only at the cache. Lever 2 (the norms) is sized in the
same sitting and planned only if the re-judge needs it (§4 step 9).

## 1. Goal and gate

**Goal.** With the attention kinds resident, the MoE inputs are bit-identical
to A. D's remaining deviation is then only the norms. D is re-judged on the
8 prompts.

| # | check | where | pass |
|---|---|---|---|
| G0 | the spec equals the Android CPU | new ARM gtest `AttnM1F16Det.*` in `unittest_nntrainer_cpu_backend_fp16` (the real exported fp16 functions against `attn_m1_det.h`) | `bad=0` at L = 1 / 2 / 63 / 64 / 65 / 512 / 513 / 1024, random and midpoint-adversarial rows; the exp probe is exhaustive over d ∈ [−17.5, 0] |
| G1 | kernel = spec on silicon | `unittest_hvx_attn --gtest_filter='HvxAttnM1.*'` and `unittest_hvx_softmax --gtest_filter='HvxM1Ops.Rope64*'` | `out bad=0` at every L, `append_chain bad=0`, rope `bad=0`. The subnormal rows of rule 37 must not be worse than the reference skel. None are expected, since every sf op now stays in the f32 normal range (§3) |
| G2 | **attention-only mask bit-identical to A** | `NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=MOE,ROPE,ATTN_M1 NNTR_HTP_DUMP=…`, G = 4, then `tools/htp/htp_dump_eval.py A1 <run>` | `bit_identical=1` on every file of A1's manifest (null check A1 vs A2 = 1). The text at G = 64 on p01 must be ≡ A. **Before** (the old set, same mask): ≈ 40 → 18 dB (#136) |
| G3 | speed | `HvxAttnM1.PerLayerCost` `ATTN_M1_FIELD pos=1023 … dsp_us=`, and E2E decode | `dsp_us ≤ 940` (the pre-#148 kernel that D's 28.16 tok/s carried, #134 ride-along 941–959). **D1 decode ≥ D0 decode** (same sitting, mirrored means) at G = 512 and 1024 |
| G4 | re-judge D1 on the 8 prompts | `tools/htp/loop_check.py` (§4 step 4), `NNTR_PPL_DECODE`, approval | no prompt where D1 loops and A does not; pooled decode PPL of D1 ≤ A × 1.02; user `text approved: y` per prompt |
| standing | prefill; text | every E2E cell | prefill ≥ −5 % of A (mirrored band). Text ≡ A for G2's mask. For D1 the rule-39 gate applies: PPL + text column + approval |

The issue closes on G0–G4. G2 without G4 closes lever 1 and opens lever 2
(§4 step 9).

## 2. Where it lives

**Changes.**

* `nntrainer/tensor/attn_m1_det.h`. The whole file is the spec: the contract
  block `:28-53` and the helpers through `attn_m1_det_forward`. It is rewritten
  to the Android CPU order (§3.1).
* `nntrainer/tensor/m1_ops_det.h:213` (`m1_rope64_det`). Changes to the fp16
  RoPE of `neon_impl_fp16.cpp:2176-2240`.
* `nntrainer/tensor/htp_backend/hvx/hvx_attn_m1_f32.c` changes in these places:
  * `hvx_attn_m1_create` `:64` gains the exp table, built once.
  * `append_head` `:146` rounds the row to fp16.
  * `attn_body` `:247-356` changes all three passes.
  * The profile words `ATTN_M1_PROF_*` keep their meaning.
* `nntrainer/tensor/htp_backend/hvx/hvx_m1_ops_f32.c:107` (`hvx_rope64_f32`).
  Rounds q and k to fp16, then does the fp16 RoPE.
* `nntrainer/tensor/htp_backend/htp_graph_desc.h:432-434`. The ATTN_M1
  head_dim rule narrows to 64. The shipping graph does not change, because
  ROPE already requires 64 and ATTN_M1 requires ROPE.
* `test/htp/host/hvx_emu/hvx_hexagon_protos.h`. Gains the integer intrinsics
  the kernel uses (`vand`/`vor`/`vxor`/`vcmp.eq`/`vcmp.gt` on words, and a
  vector-count shift if used). These are exact by definition, which avoids
  rule 37's trap.
* Host checks:
  * `test/htp/host/attn_m1_host_check.c`: an independent CPU-order reference
    replaces the double-tolerance block (`:38`, `:161`), with new mutants.
  * `test/htp/host/m1_ops_host_check.c:234` (`check_rope`).
  * `run_host_checks.sh:214-232` needs no edit.
* Device gtests:
  * `test/unittest/unittest_hvx_attn.cpp:974` / `:1112`, and
    `unittest_hvx_softmax.cpp:463`, follow the spec. The shapes in
    `RejectsBadShapes` with head_dim ≠ 64 become rejections.
  * New `AttnM1F16Det.*` in `test/unittest/unittest_nntrainer_cpu_backend_fp16.cpp`
    (built by `test/jni/Android.mk:807`, already linked to `libnntrainer.so`).
* New `tools/htp/loop_check.py` (§4 step 4).

**Consumers checked; none moves.**

* **IDL** `test/htp/nntr_hvx.idl:557-590`, stub `generate_stub.sh`: no
  signature changes, because the rows stay f32 on the wire. The stub is not
  regenerated.
* **`HtpComputeOps`** (`htp_compute_ops.cpp:1536-1551` ATTN_M1 case,
  `:1647-1665` `decode_kv_seed_fp32`): unchanged. The seed rows are the
  CPU's fp16 cache widened to f32 (`mha_core.cpp:629-651`), and the append's
  fp16 rounding is the identity on them.
* **`hexkl_graph.c`**: `graph_op_rope` `:142` and `graph_op_attn_m1` `:180`
  are unchanged. `graph_attn_scale` `:169` still passes 0.125, and ×0.125 is
  exact on fp16 values, which is the CPU's `/ sqrt(64.f)`.
* **Quantizer, loader, profile**: `nntr_quantize_stream`'s tag and the loader
  check are untouched (no weight format change). `NNTR_HTP_PROFILE` stage
  tables and `tools/htp_fc_report.py` are untouched (no stage change).
* **Transport**: the resident path keeps FastRPC for its stretches. Rule 40's
  dspqueue is for the MoE-only default and is not touched.

## 3. Design

### 3.1 The CPU's rounding points (Android, `ENABLE_FP16`), as the spec's order

These come from the source. The disassembly of the #134 set's
`libnntrainer.so` (NDK r30) agrees.

1. `Q_step/K_step/V_step.copyData` (`mha_core.cpp:545-547`): f32 → fp16, RNE.
2. RoPE on q and k (`neon_impl_fp16.cpp:2176`). The vector loop covers all 32
   pairs: `out0 = (a·c) − (b·s)`, `out1 = (a·s) + (b·c)`. Each product and each
   add is a separate fp16 op (`fmul/fsub/fadd .8h`; no `fmla` in that loop).
   The fp16 table is `(_FP16)` of the same f32 `calc_trigonometric_vals_dup`
   values the DSP already receives (`mha_core.cpp:1002-1023` vs `:605-622`).
   The K row goes into the cache as fp16, and so does the V row.
3. Scores (`:2008`), per position. Eight fp16 lane accumulators run over
   `d = 8·blk + l` with **fused** `vfmaq_f16`. Then the `faddp` tree
   `((a0+a1)+(a2+a3))+((a4+a5)+(a6+a7))`, then `0 + t`, then `(float)t / 8.0f`
   rounded to fp16.
4. Softmax (`:1490-1555`, the `row == 1` call):
   * m is the max over positions.
   * `d = fp16(s − m)`.
   * `e = fp16(exp_ps(d))`: `exp_f16x8` `:1400` is NEON `exp_ps`
     (`neon_mathfun.hxx:161`). Its `vmlaq_f32` step at `:169` **compiles to
     `fmla` (fused)**, followed by `fcvtzs` truncation.
   * The sum is an fp16 sum, **sequential over positions**.
   * `p = fp16(e / sum)` (`fdiv .8h`, correctly rounded).
5. PV (`:1858`), per q head and per d. Fused `vfmaq_f16` sequential over
   positions from 0, then fp16 out, widened to f32 (`:558`).

### 3.2 The spec (`attn_m1_det.h`, rewritten)

All values are fp16-valued f32. Every step is one IEEE f32 op followed by
`rne16`, or an integer op. There are no `hf` instructions, so there is no
`-mhvx-ieee-fp` and no `Vhf_equals_Vsf` whose v79 rounding would need a spec
(plan 81 §3.1's open item is dissolved).

* `rne16(x)`:
  * For |x| ≥ 2⁻¹⁴, an integer RNE on the bits:
    `(u + 0xFFF + ((u >> 13) & 1)) & ~0x1FFF`, with overflow to ±inf as the CPU does.
  * For |x| < 2⁻¹⁴, `(|x| + 0.5f) − 0.5f` with the sign restored. The grid
    spacing is a uniform 2⁻²⁴, and every operand is a normal f32.
  * Double rounding f32 → fp16 is harmless for add, sub, mul and div of fp16
    values (24 ≥ 2·11 + 2), so those steps are `rne16(f32 op)`.
* `fma16(c, a, b)` (steps 3 and 5). `p = a·b` is exact in f32 (22 bits,
  ≥ 2⁻⁴⁸). `s = c + p` is followed by TwoSum's exact error `err`. If `err ≠ 0`
  and s's last mantissa bit is even, move s one ulp toward `err`, which gives
  round-to-odd at 24 bits. Then `rne16`. This is exactly `RN11(a·b + c)`. The
  column "double-rounded" in the table above is this step without the fix.
* `exp16(d)`: a table over d ∈ [−17.5, 0] (≈ 19.5 k fp16 values; e = 0 below).
  It is built once in `hvx_attn_m1_create` from a scalar twin of `exp_ps`.
  The fused `fx = x·LOG2EF + 0.5` is `(float)((double)x · LOG2EF + 0.5)`. That
  is exact for fp16 x: the product fits in 35 bits and the sum in 52 of
  double's 53. The remaining ops are separate f32 ops. The double helpers are
  `__hexagon_*` imports, which `build.sh:126` allows. The volatile-store
  discipline of the existing `_det` helpers prevents contraction.
* `div16(e, l)`:
  1. `c = rne16(e · r)`, where `r ≈ 1/l` (`recip_det`).
  2. Compare e with the neighbouring midpoints times `l`. Each is 12 bits
     × 11 bits, so it is exact in f32.
  3. Step c by one fp16 ulp if needed. Ties go to even.
  This gives exact RNE without a divider.
* **Order.** The spec's order is the CPU's, including the sequential fp16 sum.
  The 32-lane tree of today's spec goes away.

### 3.3 The kernel

The layout is unchanged: f32 `Kt[d][max_seq]` / `V[pos][d]`, holding
fp16-valued floats.

* Scores keep lanes = positions. The CPU's 8 lane accumulators map to 8
  vector accumulators over `d = 8·blk + l`.
* PV keeps one V row per step.
* The sequential sum and the table lookup run per q head on the unit's
  scalar core, about 1 k ops per head.
* The cost lands in `fma16`: about 12–15 vector ops per 32 lanes against 2
  today.

**Speed ladder (G3).**

* (a) The straight kernel.
* (b) If (a) is over 940 µs: skip the TwoSum/round-to-odd fix for a block
  unless some lane's `s` sat exactly on an fp16 midpoint. RN11(RN24(x)) ≠
  RN11(x) only in that case. Keep a per-block OR of the midpoint flag, and
  recompute a flagged block exactly. This is bit-identical by construction,
  and the host check forces midpoints to prove it.
* (c) Stop and report.

### 3.4 Rejected

* **Lever 1 as filed (round only the DSP's K/V rows).** It measured +0.0 dB
  (table above), because the CPU's fp16 arithmetic, not its cache, is the
  difference.
* **Native `hf` non-fused arithmetic.** It is faster but only 52–56 dB. It
  needs `-mhvx-ieee-fp` and `hf` emulation in `hvx_emu`.
* **The double-rounded FMA.** It gives 82–95 dB with 3–62 differing outputs
  per layer. It is not bit-exact, so it stays a user fallback only if G3's
  ladder fails.
* **An fp16 cache.** It halves bytes, but the kernel is not byte-bound: 4 MiB
  in 387 µs ≈ 11 GB/s (#146). It also needs `hf` widening ops the emulator
  lacks. It stays a later option.

Contract walls: arena and address space are unchanged (48 MiB cache
+ ≈ 78 KiB table). There is no CPU fallback. Doc 45 §3 holds: `_det` spec,
bit identity on host and device, and the text gate.

## 4. Steps

1. **Spec + independent CPU-order reference.** Rewrite `attn_m1_det.h` and
   `m1_rope64_det`. The host check gets a separate reference written from
   `neon_impl_fp16.cpp`:
   * `_Float16` for the non-fused ops;
   * `fma()` in double with an explicit sticky for the fused ones;
   * the `exp_ps` twin.

   Also add midpoint-adversarial rows (v, p pairs chosen so that `c + a·b`
   lands on an fp16 midpoint after RN24). Mutants: without round-to-odd, `hf`
   non-fused, a 32-lane sum, an unfused `fx`, an f32 RoPE. Each must be
   caught.
   Gate: rung 1, `run_host_checks.sh` prints `ALL CHECKS PASS`, with a new
   line `ATTN M1 F16 CPU-ORDER OK mutants=5/5`.
2. **Kernel + rope on `hvx_emu`.**
   Gate: `ATTN M1 BIT-IDENTICAL` at L = 1 / 63 / 64 / 65 / 512 / 1024 × pool
   0 / 3 / 7, `append_chain == bulk`, `M1 OPS BIT-IDENTICAL`, `ATTN M1 PHASES OK`.
   Then rung 2: `test/htp/build.sh` prints `UNDEFINED SYMBOLS OK`, and record
   the skel md5.
3. **ARM gtest `AttnM1F16Det`** (G0). The test chains the real exported
   functions in the order of `mha_core.cpp:821-877`:
   `copyData` → `compute_rotary_emb_value(__fp16)` → `compute_kcaches(_FP16)`
   → `softmax_row_inplace(_FP16)` → `compute_fp16vcache_transposed`. It
   compares against the spec over the L list.
   The exp probe feeds `softmax_row_inplace` with (0, d) column pairs over
   every fp16 d ∈ [−17.5, 0], and compares p₀ and p₁. For d < −7.6 the sum
   is 1, so p₁ = e exactly.
   Gate: builds in rung 3. It runs in the sitting.
4. **`tools/htp/loop_check.py`** (host only). Arguments: `--prompt <file> <log>…`.
   * It extracts the generated text: after the prompt echo, before the
     `=================[ LLM` banner.
   * **L1**: the longest run of consecutive identical sentences with ≥ 3 words.
     Split after `.!?。？！` plus whitespace or a newline, lowercase, collapse
     whitespace, strip trailing punctuation.
   * **L2**: the fraction of word 12-grams starting in the second half of the
     text that already occurred earlier in it.
   * `loop = L1 ≥ 3 or L2 ≥ 0.5`.
   * Output line: `LOOP <label> L1run=<n> L2=<x> loop=0|1`.
   * **A variant fails a prompt only if it loops where A does not.** A itself
     loops at G = 64 on p05 (L1 5, L2 1.00) and p08 (L2 1.00), and on p01
     after ≈ 300 words (L2 0.60 at 300, 0.95 at 431).

   Self-test, measured by the planner on the #134 G=512 logs cut to 200
   words: A 1 / 0.00 → 0, B 7 / 1.00 → 1, D 8 / 1.00 → 1. D is already
   flagged at 100 words (3 / 0.98). Hence the text set runs at **G = 256**.
   Gate: the script reproduces these rows.
5. **Host E2E.** `run_inproc_e2e.sh` prints `INPROC E2E PASS`. The host CPU
   attention is f32 (the fp16 branch is Android-only), so `fwd-hd64
   min_snr_db` cannot rise and may fall. Record it against the 30 dB floor.
   Do not lower the floor.
6. **Rung 3.** Build the app and the gtests. Stage the new set, and next to it
   the unchanged #134 set (`/local/mnt/workspace/htp_moe/134-132/set/`) as
   D0, in its own run dir. Check md5s for both. Run
   `llvm-objdump` on the new `libnntrainer.so`: `softmax_row_inplace<half>`
   still shows the `fmla .4s` + `fcvtzs` pair the twin assumes.
7. **Device sitting (unavoidable; the orchestrator runs it on `R3CY10WM83Y`).**
   Write `docs/measurements/152-resident-accuracy.md`, ≈ 50 min, thermal log
   at each checkpoint.
   * (a) gtests G0 / G1 / G3's `PerLayerCost`.
   * (b) Dumps at G = 4. Diagnostic cells, not speed variants:
     * A1, A2;
     * `MOE,ROPE,ATTN_M1` on the old set (before) and on the new set (G2);
     * new set: `MOE,QK_NORM,ROPE,ATTN_M1`, `MOE,RMSNORM`, `MOE,CONV1D_GATE`,
       `MOE,ADD,ROUTER_TOPK`, D1;
     * old set: D0;
     * each read by `htp_dump_eval.py`: `bit_identical`, `min_snr_db`, `first_diff`.
   * (c) Speed, **3 variants**: A (new set, nothing set), D0 (old set, D
     mask), D1 (new set, D mask). Full E2E, prompt 512, G = 64 / 512 / 1024 ×
     2, mirrored `A D0 D1 | D1 D0 A`. Every switch-on log shows its
     `graph: init … resident=` banner and `calls/token=51.00` (rule 36).
   * (d) Text: 8 prompts at G = 256 for A, D0 and D1, then `loop_check.py`.
     D0 may be dropped if time is short.
   * (e) PPL: `NNTR_PPL_DECODE` per prompt at G = 256: A self (writes
     `cont_<p>.ids`), A forced on p01 (null check), D0, D1. Report pooled and
     per prompt.
   * (f) Paste A / D1 texts for approval.
   Stop rules: if G1 fails on normal rows → stale skel or silicon mismatch;
   stop before (c). If G0 fails → G2 cannot be judged; run (b) for the record
   and stop.
8. **Fold.** G2 ✓ + G3 ✓ + G4 ✓ → the issue closes, and track (c) goes to the
   user with the numbers.
9. **Lever 2, only if G4 fails and D1's `first_diff` is a RMSNORM / QK_NORM
   stretch.** File it as its own issue, sized by (b)'s `MOE,RMSNORM` and
   `MOE,QK_NORM,…` SNRs. The shape of the fix: bit-exact NEON
   `rms_norm_wrt_width_fp32_intrinsic` (`neon_impl.cpp:1888`). That means:
   * fused `vfmaq_f32` (Dekker split + round-to-odd);
   * `hsum` order;
   * `/W`, then `1/sqrtf`, correctly rounded;
   * `x·scale`.

   The CPU's cost is unchanged. G3 fails with G2 passing → ladder step (b),
   then a user decision on the double-rounded fallback.

## 5. Risks (host vs device)

* **Emulation vs silicon on tiny values (rule 37).** By design every sf op
  sees normal-range f32 (fp16 values ≥ 2⁻²⁴, products ≥ 2⁻⁴⁸). The
  fp16-subnormal rounding goes through `+0.5` on normal values. G1's `bad`
  columns show a violation per case.
* **ARM codegen.** The twin assumes one fused step in `exp_ps` and non-fused
  fp16 RoPE. A different NDK or different flags could change either. Step 6's
  objdump and G0 catch it before E2E.
* **Speed is only known on silicon.** The hvx_emu has no timing. G3 reads
  `PerLayerCost` in the same skel as the E2E. A D1 < D0 with `dsp_us ≤ 940`
  would point at the table build or the scalar sum, which the phase words
  `SOFTMAX` / `PV` localise.
* **DVFS and thermal drift.** Every speed number is read inside the sitting.
  The order is mirrored, zone0 is logged at each checkpoint, and the prefill
  gate is read against the mirrored band (#134 fell 564 → 419 with heat).
* **Stale skel / old set.** Two run dirs with two skels. Every log carries
  the skel md5. `0x8000040e` = stale.
* **Address space.** The cache stays at 48 MiB plus a ≈ 78 KiB table, and the
  `attn_m1: registered … cache=49152 KiB` banner must still read the same.
* **The baseline loops itself** (p05, p08 at G = 64; p01 by 300 words). The
  relative loop rule keeps that from failing D. The user should still know
  that "no repetition loops" cannot hold for A on this prompt set.

## 6. Docs to update

* **`docs/htp_moe/BENCHMARK.md`.**
  * Results: rows for #152 A / D0 / D1 (G 64 / 512 / 1024, prefill, PPL,
    loop flags, approval).
  * A #152 side table: the per-kind dump SNR before/after, plus G0/G1/G3
    gtest lines.
  * The accuracy row of Goals (`:143`) gets the verdict.
* **`docs/htp_moe/LEDGER.md`.**
  * ⑨: track (c) status and the lever-1 correction.
  * ㉗: ATTN_M1 cost with the CPU-exact kernel.
  * §2: a #152 verdict row.
  * A new rule if the device confirms it: *the Android CPU attention is fp16
    end to end (q/k/v, RoPE, scores, fused FMAs, sequential fp16 softmax sum,
    PV). Rounding the DSP's K/V alone changes nothing. A resident attention
    matches A only as a whole.*
  * Close plan 81 §3.1's open `sf → hf` item as not needed.
