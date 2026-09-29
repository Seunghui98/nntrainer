# Measurement 132 PR 2: the CPU-exact FC, router, SwiGLU and argmax on silicon

Branch `htp/132-exact-fc`, PR code @ `cd611465`; the device set is built
from `dev/fc-shadow` @ `fec9d999` (= the PR diff plus one inert measurement
commit, never merged). Plan `docs/plans/132-cpu-exact-fc-lmhead.md` Part A,
step A6. Estimated device time: **≈ 55 min** (state + install 4, G0 6, G1 3,
G3 2, G4 + G2 4, speed 14, router profile 2, nll + text 16, checks 2, plus
cool-down waits). Run by the orchestrator:

```
bash /local/mnt/workspace/htp_moe/132/run_132.sh R3CY10WM83Y
```

(a byte-identical copy is `docs/measurements/132-pr2-run.sh` on this
branch). The script logs to `/local/mnt/workspace/htp_moe/132/logs/`,
writes the summary to `logs/sitting.out`, stops only on a missing unit, a
set / device md5 mismatch or `0x8000040e` (stale skel), counts every other
missing expected line (`expectation mismatches: N`) and **reports** the
plan's stop rules (`plan stop rules hit: N`) instead of stopping, so that
G3 and G4 — decision D's inputs — are read in the same sitting whatever G0
and G1 show.

## Why

1. Every op the decode may move to the NPU must produce the Android CPU's
   bits (LEDGER rule 45). PR 2 writes the CPU's order down as specs
   (`q4_gemv_cpu_det.h`, `m1_ops_det.h`) and makes the DSP equal them. The
   host proves spec == an independent model of the CPU and kernel ==
   spec on the emulation and the ISS; this sitting reads the three things
   the host cannot: the spec against **this set's** CPU functions (G0),
   the kernel on silicon (G1), and every op on the model's real
   activations, DSP against CPU (G2).
2. Decision D (plan §3.4) needs two numbers: the exact FC's rate on
   silicon (G3; the CPU does FC + lm_head in 7.4 ms/token) and the DSP
   address space left to the loaded app (G4).

### Variants (one set, switched by env)

| | env (beyond `NNTR_NUM_THREADS=8`) | expected |
|---|---|---|
| **A** (reference, first) | none | `[HTP] dspq: on` once, `dspq: close calls=N served=N bad=0`, no `graph:` line |
| **S** | `NNTR_FC_SHADOW=<file>` | as A; plus `fc.bin` records. The DSP re-runs every M=1 FC / lm_head slice, residual ADD and router on the CPU's own input for the first 8 decode steps (`NNTR_FC_SHADOW_STEPS`, default 8; 0 in the G2 cell) and writes both outputs. **Nothing the CPU computes is replaced**, so S's text and nll must equal A's. Under the shadow the residual adds are built as `residual_add` (`copy(in0)`, `add_i(in1)`: the `addition` layer's bits) to carry the ADD records |
| **P / Pb** (profile only) | `NNTR_HTP_FORWARD=1 …KINDS=MOE,RMSNORM,QK_NORM,ROPE,CONV1D_GATE,ATTN_M1,ADD,ROUTER_TOPK NNTR_HTP_PROFILE=2`; Pb with `ADSP_LIBRARY_PATH=<D>/base` (the `htp_moe` skel, the old HVX router) | `graph: init … resident=RMSNORM\|CONV1D_GATE\|QK_NORM\|ROPE\|ATTN_M1\|ADD\|ROUTER_TOPK\|MOE`, `calls/token=51.00`; the `ROUTER_TOPK=` pcycles/op of the new scalar-chain router against the old one (the code review's finding 3). Never a tok/s cell; the text is not compared |

## Artifacts

Set `/local/mnt/workspace/htp_moe/132/set/`, `md5.txt` inside it:

| file | md5 (device set, `fec9d999`) | PR-only md5 (`04ef2761` = `cd611465` for ARM, not pushed) | built with |
|---|---|---|---|
| `libnntr_hvx_skel.so` | `ce0ff77fc02894d7e039de6874599993` | `f2e9f384f01997026a4655f40457b3dd` | `test/htp/build.sh` (v79, HexKL 6.4.0.1): `UNDEFINED SYMBOLS OK (52 runtime imports)`; `hvx_q4_gemv_f32.c` alone with `-mhvx-ieee-fp`. Not byte-reproducible: the md5 you push is the record |
| `libnntr_hvx_skel_base.so` | `b07eb8a699c7e4fbee6c15cd1cba3bd3` | | `origin/htp_moe` @ `aaafd0c4`, same script (51 imports); Pb only, pushed to `<D>/base/libnntr_hvx_skel.so` |
| `nntrainer_causallm` | `7797dd804309b357e0c0b9fd55d45dbe` | `7797dd804309b357e0c0b9fd55d45dbe` | `build_android.sh --htp --cache` (`jni/libs/arm64-v8a/`) |
| `libcausallm_core.so` | `5ddac9a4f5d4a32d67c0edccd34cdaeb` | `8ea4137b42156500b9d2823e9cf0cb0a` | same; `NNTR_HTP_FORWARD_KINDS` 2 |
| `libnntrainer.so` | `44dfcdad3d77d8d6210a864be9241385` | `ee95ff59c77a35b2a10c35d2e6df674b` | same (`jni/obj/local/arm64-v8a/`; NEEDED `libsdkl.so`, `libcdsprpc.so`; `graph: forward calls` 1; `dspq: on` 1) |
| `libccapi-nntrainer.so` | `02d314e56800a72adc166f64256aca0e` | `02d314e56800a72adc166f64256aca0e` | same |
| `unittest_nntrainer_cpu_backend` | `0106ed65b1faf7dfc0afb346f4518f25` | `ca11415659132ddd647be00fd0bdb2a4` | `test/jni` ndk-build against the set's `libnntrainer.so` (G0) |
| `unittest_hvx_softmax` | `b04269ac26098d3b0c5d377ea6951060` | `b04269ac26098d3b0c5d377ea6951060` | same (G1, G3) |
| `libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | | NDK r30 sysroot |
| `libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | | HexKL 6.4.0.1 `armv8_android26` |
| `prompt512.txt` (p01), `bitset-02-code.txt` … `bitset-08-short.txt` | as `md5.txt` (p01 `fc65c158…`) | | from `164/set/` |
| `tools/fc_shadow_check.py` (workstation) | `fa18d344605ceb01a109bd53e567ec8c` | | `dev/fc-shadow`, G2's reader |
| model `q40-qs4cx-wh`, `tokenizer.json` | on the phone since #100 | | |

`fec9d999`'s ARM tree equals `a031b42e`'s, from which the app binaries
were built (the two differ by a comment in `test/htp/nntr_hvx_fc_q4.c`);
the skel was rebuilt at `fec9d999`. Not staged, never pushed:
`libcdsprpc.so` (rule 5).

Workstation checks done while staging:

* G0's premise: `llvm-objdump` of the set's `libnntrainer.so` /
  `libcausallm_core.so` against the 2026-09-29 `norm/set` the plan read
  (§0): `nntr_gemv_q4_0_4x8_q8_0` (one `fmla` chain), `nntr_quantize_row_q8_0`
  (the two `fdiv`), `neon::swiglu`, `cblas_sgemv` and
  `buildExpertAssignments` disassemble identically (0 differing lines,
  address operands normalized).
* Rung 1: `run_host_checks.sh` (`ALL CHECKS PASS` ×3, `WORKER POOL LANES OK`,
  `Q4 GEMV CPU-ORDER OK mutants=6/6`, `Q4 GEMV BIT-IDENTICAL`, `ROUTER TOPK
  BIT-IDENTICAL`, `M1 OPS BIT-IDENTICAL`), `htp_syntax_check.sh` 0,
  `*Lfm2Moe*` 6 passed, `INPROC E2E PASS` (fixture lines with
  ROUTER_TOPK resident: `d-tiny` 146.32 dB, `d-hd64` 52.92, `d-lfm25`
  133.63, tokens 8/8 each). Rung 2 as above. Rung 3 on the PR head and on
  the dev head: the NEEDED / strings checks and both gtest binaries.
* The dev commit on the host (in-process, hd64 fixture, 8 steps): tokens
  equal to switch-off 8/8, ADD records 14/14 equal; router records 0/14
  and no FC records **by design** (the x86 CPU's sgemv is not the aarch64
  order; the FC hook lives in the ARM backend). The plan's host A4 gate
  "every record equal" cannot hold on x86; G2 is a device reading.
* ISS (`hexagon-sim -mv79 --timing`, one thread, K = 2048, N = 256), every
  variant bit-equal to the spec on 4 rows: pcycles per (column, 32-block)
  step **3.75** HVX native (120 per 32-column vector-step; the planner's
  prototype 124), **4.07** intrinsics, **5.01 / 4.76 / 5.79** sffma with
  8 / 16 / 32 columns in flight. Not a gate (contract §12); G3 reads the
  real ones.

## Steps (the script does all of it)

0. `adb devices`, the unit, `input keyevent 223`, battery / zone0, wait
   until zone0 ≤ 35 °C (also before G1, G3, each G, each prompt).
1. md5 of the workstation set, push to `<C>/s132`, the base skel to
   `s132/base/`, md5 on the device against `md5.txt`, config
   (`do_sample false`, `bad_word_ids [124900]`, `moe_engine htp`).
2. **G0** `unittest_nntrainer_cpu_backend --gtest_filter='Q8QuantCpuOrder.*:Q4GemvCpuOrder.*:SwigluCpuOrder.*:SgemvNCpuOrder.*:ExpfBionic.*'`. Expected:
   `Q8QuantCpuOrder rows=4000 bad_blocks=0`,
   `Q4GemvCpuOrder K=… N=… rows=2000 bad=0 layout=ok` ×5,
   `SwigluCpuOrder n=7168 rows=2000 bad=0`,
   `SgemvNCpuOrder K=2048 E=32 rows=2000 bad_logits=0 bad_sel=0`,
   `ExpfBionic inputs=4278190082 bad=0`, `PASSED ] 5 tests`.
   ExpfBionic prints the first 16 differing inputs; the host's glibc
   corrects two inputs the algorithm misrounds (`0x4202422f`,
   `0xc27c65d9`), so if bionic does too they are the ones listed.
3. **G1** `unittest_hvx_softmax --gtest_filter='HvxFcQ4.MatchesSpecBitExact:HvxFcQ4.SmallOpsMatchSpec:HvxM1Ops.*'`. Expected:
   `FC_Q4_FIELD K=… N=… variant=… feed=… rows=8 bad=0` ×50 (5 shapes ×
   {hvx_native, hvx_intrin, sffma8, sffma16, sffma32} × {direct, vtcm}),
   `FC_Q4_FIELD total bad=0`,
   `FC_Q4_FIELD small ops: q8_quant bad_rows=0 swiglu_cpu bad=0 argmax bad=0`,
   `M1_OPS_FIELD router_topk … bad_logits=0 bad_sel=0 bad_weight=0` ×3,
   and #164's `HvxM1Ops` lines as before.
4. **G3** `--gtest_filter='HvxFcQ4.Rate'` with zone0 before / after:
   `FC_RATE K N variant feed lanes us_per_call quant_us pcyc_per_call
   cyc_per_colstep_lane GBps bad` per cell (6 shapes × 5 variants × 2 feeds
   × 1 / 2 / 4 / 6 lanes), `FC_RATE_PROJ … ms_per_token=… (fc=… lm_head=…)
   quant_ms=… cpu_reference_ms=7.4` per variant, feed and lane count,
   `FC_RATE bad_total=0`. The lm_head is one 16384-row slice × 7.8125.
5. **G4** A, prompt 512, G = 8, `NNTR_HEAP_PROBE=1 NNTR_PPL_DECODE=cont.ids`
   (self; writes A's continuation): `[HTP] arena …` lines and
   `[HTP] heap probe: <N> MiB allocatable in 1 MiB chunks`.
6. **G2** S, same prompt, G = 8, forced on `cont.ids`,
   `NNTR_FC_SHADOW_STEPS=0`; pull `fc.bin`; `fc_shadow_check.py`:
   `FC SHADOW … fc=n/n add=n/n router=n/n` (every record bit-equal, each
   n > 0), per step ≈ 74 FC records (18 × 2 conv + 6 × 4 attention + 2 × 3
   dense FFN + 8 lm_head slices), 48 ADD, 22 router; nll of S == A.
7. Speed and inertness, prompt 512, **A S S A** at G = 64 / 512 / 1024:
   S's text == A's (run 1 of the same G), prefill of S ≥ −5 % of A.
8. Router profile: Pb then P at G = 64 (see Variants).
9. Eight prompts at G = 256: A self (`cont_p0i.ids`), S forced on it (nll
   lines equal to 17 digits), then A and S unforced (text identical).

## Results (fill in)

| cell | expected | got |
|---|---|---|
| G0 quantizer / Q4 GEMV 5 shapes / SwiGLU / sgemv_n + selection / expf 2^32 | all bad=0 | |
| G1 FC 50 cells, small ops, router 3 shapes | all bad=0 | |
| G1 by variant (cells with bad > 0) | 0 each | native: , intrin: , sffma8: , sffma16: , sffma32: |
| G2 records fc / add / router equal | n/n each | |
| G2 nll S == A | y | |
| G3 ms/token at 6 lanes (fc + lm_head), direct / vtcm | reported (CPU 7.4) | native: , intrin: , sffma8: , sffma16: , sffma32: |
| G3 cycles per (column, block) step per lane, 6 lanes, K 2048 N 6144 | reported | |
| G4 heap MiB free / arena mapped | reported (plan: ≈ 100 − 48 heap, 144 arena) | |
| router ROUTER_TOPK pcycles/op, htp_moe skel / this set | reported | |

| variant | gen | run | prefill tok/s | decode tok/s (all) | decode tok/s (last 64) | text = A run 1? |
|---|---|---|---|---|---|---|
| A | 64 | 1 | | | | (reference) |
| S | 64 | 1 | | | | |
| S | 64 | 2 | | | | |
| A | 64 | 2 | | | | |
| … G = 512, 1024 likewise | | | | | | |

| prompt | S nll == A (G = 256, forced) | S text == A (G = 256) |
|---|---|---|
| p01 … p08 | | |

Reference: the NPU "now" is 51.82 / 50.61 / 47.36 decode tok/s (#158 B,
`R3CY10WM83Y`, 2026-09-29); A of this sitting is the reference for every
cell. S's decode speed is not read (the shadow works during its first 8
steps); its prefill is (the shadow does not run at M > 1).

## Notes from the run

<serial, battery, zone0 log, FARF/AEE errors, anything stale>

## What the verdicts mean (plan §4 A7)

* G0 ✓ G1 ✓ G2 ✓: Part A closes. The ROUTER_TOPK kind becomes CPU-exact
  (with #164's norms, the next per-token-entry sitting can read text ≡ A
  for masks that include it). The G3 and G4 tables go to the user as
  decision D.
* G1 ✗ on `hvx_native` only: the IEEE `.sf` instructions are not RN on
  silicon; the intrinsics variant is the fallback (plan §5).
* G0 ✗: the CPU is not what §0 read; the printed differences name the
  function. ExpfBionic ✗ on exactly the two glibc-corrected inputs: the
  port needs those two as exceptions (a two-entry table).

## Sitting 1 as read (2026-09-29 21:13-22:0x, `R3CY10WM83Y`; logs `/local/mnt/workspace/htp_moe/132/logs/`)

Reported by the orchestrator; the numbers below are copied from the logs.

| cell | result |
|---|---|
| G0 (quantizer, Q4 GEMV 5 shapes, SwiGLU, sgemv_n router, expf over all 4 278 190 082 inputs) | all bad=0 |
| G1 `hvx_intrin` | bad=0 in all 10 cells (5 shapes x 2 feeds) |
| G1 `hvx_native`, `sffma8/16/32` | every output `0x0` (`bad=N` per call in every cell): not a rounding difference |
| G1 router (3 shapes), small ops | bad=0 |
| G2 | add 96/96, router 176/176; fc 0/600 (the shadow's `fc_q4m1_f32` variant 0 was `hvx_native`) |
| G3 at 6 lanes, VTCM feed | `hvx_intrin` 7.88 ms/token (fc 5.14 + lm_head 2.74; K 7168 N 2048: 181 µs/call, 45.5 GB/s) vs the CPU's 7.4; direct feed 30.5; `hvx_native` 7.30; sffma 18.4-22.7; quant_ms ≈ 4.8 |
| G4 | heap probe 113 MiB (1 MiB chunks) |
| S vs A | tokens, nll and logits equal; 8 prompts nll and text equal |
| resident ROUTER_TOPK | 427 784 pcycles/op (CPU-order scalar chains) vs 34 100 (the old HVX router) |

The runner's 8-prompt loop wrote its logs as `*_p03` / `*_p04`: `cool()`
reused the loop's `i`. Fixed (`local t z`) in both runners.

## Sitting S3 (after sitting 1)

Branch `htp/132-exact-fc`; set `/local/mnt/workspace/htp_moe/132/set2/`
from `dev/fc-shadow`; runner
`bash /local/mnt/workspace/htp_moe/132/run_132s3.sh R3CY10WM83Y`
(≈ 35 min; copy `docs/measurements/132-pr2-run-s3.sh`); logs
`/local/mnt/workspace/htp_moe/132/logs_s3/`. What changed since sitting 1:

1. **One kernel.** `hvx_q4_gemv_f32.c` is the intrinsics kernel only;
   `hvx_native` and the sffma tail are gone from the entry (`fc_q4m1_f32`'s
   variant word is the feed bit alone), so the shadow's call is the
   intrinsics kernel: G2 must read **fc 600/600**.
2. **Why native / sffma read 0.** Both used the inline-asm IEEE
   `vadd / vsub / vmpy (.sf = .sf op .sf)` instructions (the sffma tail
   for its `d_w * d_a` terms); `hvx_intrin` uses only the `Q6_Vsf_*`
   intrinsics, which the compiler lowers to a `qf32` op and a conversion
   even with `-mhvx-ieee-fp`. #164's norm and the router passed because
   they use the scalar `sffma` and the intrinsics, never the asm form. So
   the difference is not the rounding of any op: it is that instruction
   form, which the v79 ISS executes and the phone returns 0 for.
   `HvxFcQ4.SfProbe` confirms it on silicon: each op on 256 random normal
   pairs and on the operands of the kernel's first failing step (group 0,
   block 0: P1 = F1 * S1, P2 = F2 * S2, uh = P1 + P2), asm form vs
   intrinsics vs scalar sffma, against the host's IEEE result.
3. **Vector quantizer** (`hvx_q4m1_prep`): amax by a word max of the
   sign-cleared bits, d and 1/d by the scalar divide (`sfrecipa` /
   `sffixup`, IEEE RN: `HvxFcQ4.ScalarDivide` sweeps every mantissa of
   both divides on silicon, the ISS sweep is below), q by the 1.5 * 2^23
   magic add (ties to even), bytes by `vpacke`, sums by a rotate-add
   tree; blocks under 2^-100 run the scalar spec. Host:
   `Q8 QUANT HVX rows=3000 bad_rows=0`. ISS, K = 2048: **10 738** pcycles
   per call against the scalar spec's 105 609 (G3 reads the silicon
   `quant_ms`; target < 0.5 ms/token).
4. **Router**: 8 register-resident sffma chains per pool lane (4 lanes at
   E = 32), each lane also computing its experts' sigmoids; the pick is
   serial. ISS, one lane: 41 k pcycles per 8-chain group plus ≈ 4 k of
   sigmoids per lane; ≈ 50 k projected at 4 lanes (target ≤ 60 k). The
   graph's ROUTER_TOPK passes `env->pool`, the test entry the session's.

### S3 artifacts

Set `/local/mnt/workspace/htp_moe/132/set2/` (`md5.txt` inside), built from
`dev/fc-shadow` @ `19af4eea` (PR code `b8037431` + the shadow commit):

| file | md5 | note |
|---|---|---|
| `libnntr_hvx_skel.so` | `eaba476cbec5840004445f4613abcf9a` | `test/htp/build.sh`, `UNDEFINED SYMBOLS OK (52 runtime imports)` |
| `libnntr_hvx_skel_base.so` | `b07eb8a699c7e4fbee6c15cd1cba3bd3` | `htp_moe` @ `aaafd0c4`, the router profile's Pb |
| `libnntrainer.so` | `9e8b91267dfa02bef6ec70e5c3f1ae02` | the new IDL's stub; NEEDED `libsdkl.so`, `libcdsprpc.so` |
| `libcausallm_core.so` / `nntrainer_causallm` / `libccapi-nntrainer.so` | `5ddac9a4…` / `7797dd80…` / `02d314e5…` | unchanged since sitting 1 |
| `unittest_hvx_softmax` | `90507ccd9e23d2b1fc3b4f5d3c94684f` | G1, G3 |
| `unittest_nntrainer_cpu_backend` | `17afcad382e5a8be7b758d367deacfa9` | relinked, not run in S3 |
| `libc++_shared.so`, `libsdkl.so`, prompts | as sitting 1 | |

Workstation, S3 tree: `run_host_checks.sh` (`ALL CHECKS PASS` x3,
`WORKER POOL LANES OK`, `Q8 QUANT HVX BIT-IDENTICAL`, `Q4 GEMV
BIT-IDENTICAL`, `ROUTER TOPK BIT-IDENTICAL`, `GRAPH STRETCH
BIT-IDENTICAL` with ROUTER_TOPK on a real pool), syntax check 0,
`*Lfm2Moe*` 6 passed, `INPROC E2E PASS` (same fixture lines as before);
the ISS sweep of the scalar divide at exponent 0: `a/127 bad=0 1/a bad=0`
over all 8 388 608 mantissas; the kernel's code carries 72 qf32 ops and no
asm IEEE .sf, the quantizer's divides are inline sfrecipa sequences.

### S3 expected lines

* G1: `SF_PROBE … op=intrin_* bad=0/…` and `scalar_sffma bad=0/256`, the
  `asm_ieee_*` lines as they come (the finding), `SF_PROBE
  intrinsics+sffma bad=0`, `SCALAR_DIV divides=50331648 bad=0`,
  `FC_Q4_FIELD … variant=hvx_intrin … rows=8 bad=0` x10, `FC_Q4_FIELD
  total bad=0`, `FC_Q4_FIELD small ops: q8_quant bad_rows=0 swiglu_cpu
  bad=0 argmax bad=0` (the quantizer now on 500 rows), router x3, `PASSED
  ] 5 tests`.
* G3: `FC_RATE_PROJ variant=hvx_intrin …` direct / vtcm x 1/2/4/6 lanes,
  `quant_ms` now the vector quantizer's; `FC_RATE bad_total=0`.
* G2: `FC SHADOW … fc=600/600 add=96/96 router=176/176`; nll S == A.
* Router: `ROUTER_TOPK pcyc/op: htp_moe skel … this set … (target <= 60000)`.
* 8 prompts: S nll == A and S text identical, 32 logs present.

### S3 results (2026-09-30 02:22-02:47 KST, `R3CY10WM83Y`, zone0 27.8-29.7 °C at start; logs `/local/mnt/workspace/htp_moe/132/logs_s3/`; 1 expectation mismatch: the router gate)

| cell | expected | got |
|---|---|---|
| SfProbe `asm_ieee_vadd / vsub / vmpy`, 256 random normal pairs | the finding | **256/256 bad, every result `0x00000000`** (e.g. `431a663d + 3828cf8f`: dsp `00000000`, host `431a6640`) |
| SfProbe asm on the failing step's operands (P1 = F1·S1, P2 = F2·S2) | the finding | vadd / vsub 32/32 bad (`0x00000000`); vmpy 0/32 only because those lanes' exact products are 0 |
| SfProbe `intrin_vadd / vsub / vmpy`, `scalar_sffma` | bad=0 | 0/256 and 0/32 each; `SF_PROBE intrinsics+sffma bad=0` |
| ScalarDivide (6 x 2^23 divides) | bad=0 | `SCALAR_DIV divides=50331648 bad=0` |
| G1 FC 10 cells, quantizer 500 rows, SwiGLU, argmax, router 3 shapes | bad=0 | all bad=0, `PASSED ] 5 tests` |
| G3 at 6 lanes, VTCM feed | reported | **7.844 ms/token** (fc 5.089 + lm_head 2.755), **quant_ms 0.527**; K 7168 N 2048: 173.7 µs/call, 47.6 GB/s; lm_head slice 352.7 µs, 53.5 GB/s |
| G3 direct feed | reported | 30.601 ms/token (quant_ms 0.864); 1 / 2 / 4 lanes VTCM 44.3 / 22.3 / 11.3 |
| G2 | 600/600, 96/96, 176/176 | **fc 600/600, add 96/96, router 176/176**; nll S == A |
| G4 (heap probe, A) | reported | 109 MiB (sitting 1: 113) |
| ROUTER_TOPK pcycles/op | ≤ 60 000 | **65 528** (htp_moe skel 33 755): missed by 9 % |
| 8 prompts nll / text S == A | 8/8, 8/8 | 8/8, 8/8 (p01-p08, 32 logs) |

**Confirmed cause of sitting 1's hvx_native / sffma failure.** The
inline-asm IEEE HVX `.sf` forms (`Vd.sf = vadd / vsub / vmpy(Vu.sf,
Vv.sf)`, assembled with `-mhvx-ieee-fp`) return `0x00000000` in every lane
on this v79 silicon, while the ISS executes them as IEEE ops. The
`Q6_Vsf_*` intrinsics (a qf32 op and a conversion) and the scalar
`sffma` are exact on the same operands. No rounding differs: the asm form
does not execute. `hvx_native` used it for every add, sub and mul, and the
sffma tail for its `d_w * d_a` terms. #164's norm and the router passed
because they never used it. LEDGER rule 53.

Quantizer: 7.0 µs per K = 2048 call on silicon (≈ 14.7 k pcycles at the
≈ 2.1 GHz G3 implies) against 10.7 k on the ISS (x1.37). Scalar spec in
sitting 1: ≈ 60 µs; quant_ms 4.8 → 0.53.

## Sitting S4 (router only, after S3)

The router missed its gate by 9 %. On the ISS the chain loop ran at about
20 cycles per k whether it held 8 or 16 chains: it was bound by the row
load (each k reads a new 128-byte line of the gate weight, an L1 miss),
not by `sffma`. `38dd1626` prefetches the row 8 ahead into L1 (`dcfetch`),
in a loop split so the last 8 rows do not prefetch. ISS, one 8-chain
group at K = 2048: **47.7 k → 13.8 k pcycles** (4 / 16 ahead the same, 32
worse; a clamped prefetch address inside the loop 24.1 k). ISS against
silicon: S3's 65 528 on the phone against the ISS model's ≈ 50 k
projection for the same code (x1.3; the quantizer x1.37); the ISS now
projects ≈ 19 k (13.8 k per lane + 17.4 k / 4 of sigmoids and pick +
dispatch), so ≈ 25-27 k on silicon if the ratio holds. Host checks all
pass (`ROUTER TOPK BIT-IDENTICAL`, `GRAPH STRETCH BIT-IDENTICAL`).

Set `/local/mnt/workspace/htp_moe/132/set3/`, runner
`bash /local/mnt/workspace/htp_moe/132/run_132s4.sh R3CY10WM83Y` (≈ 15 min,
logs `logs_s4/`; copy `docs/measurements/132-pr2-run-s4.sh`): G1 router
cells (3 shapes), the router profile (htp_moe skel vs this set, gate ≤
60 000), and the 8 prompts (S nll and text == A). The set is rebuilt from
the tree rebased on `htp_moe` @ `4eab54ef` (#170's ATTN_M1 work landed
between S3 and S4); md5s in `set3/md5.txt`.

| file (`set3/`, built at `dev/fc-shadow` @ `a8a00d54`) | md5 |
|---|---|
| `libnntr_hvx_skel.so` (`UNDEFINED SYMBOLS OK (52 runtime imports)`, 2 `dcfetch` in `router_lane`) | `63c372ec42abd5a38ff8ab7f34f90917` |
| `libnntr_hvx_skel_base.so` (htp_moe @ `aaafd0c4`, Pb) | `b07eb8a699c7e4fbee6c15cd1cba3bd3` |
| `libnntrainer.so` (NEEDED `libsdkl.so`, `libcdsprpc.so`) | `6d0116785f3aba951671a26288599192` |
| `libcausallm_core.so` / `nntrainer_causallm` / `libccapi-nntrainer.so` | `5ddac9a4…` / `7797dd80…` / `02d314e5…` (no app change upstream) |
| `unittest_hvx_softmax` | `3160d2343655477a67bda33641a56ced` |

Gates on the rebased tree: `run_host_checks.sh` (`ALL CHECKS PASS` x3,
`WORKER POOL LANES OK`, `ROUTER TOPK BIT-IDENTICAL`, `Q8 QUANT HVX
BIT-IDENTICAL`, `Q4 GEMV BIT-IDENTICAL`, `ATTN M1 BIT-IDENTICAL`), syntax
check 0, `*Lfm2Moe*` 6 passed, `INPROC E2E PASS`, skel, app and gtest
builds.

### S4 results (2026-09-30 07:33-07:58 KST, `R3CY10WM83Y`; logs `/local/mnt/workspace/htp_moe/132/logs_s4/`; 1 expectation mismatch: the router gate)

| cell | expected | got |
|---|---|---|
| G1 router 3 shapes | bad=0 | bad=0 (`PASSED ] 1 test`) |
| ROUTER_TOPK pcycles/op (htp_moe skel / this set) | ≤ 60 000 | **33 764 / 62 311** (S3 65 528): **gate missed by 4 %** |
| 8 prompts nll / text S == A | 8/8, 8/8 | 8/8, 8/8 (32 logs) |

**Verdict (orchestrator, 2026-09-30): accepted as Part A's result.** 62 311
pcycles/op is ≈ 30 µs per call, ≈ 0.65 ms/token over 22 layers, inside the
± 1 ms of the E2E budget's 0.5 ms. No further router round now; revisit
only if the E2E budget needs it.

**ISS against silicon.** The L1 prefetch took the ISS model of one 8-chain
group from 47.7 k to 13.8 k pcycles (−60 % of the op), but silicon moved
only −5 % (65 528 → 62 311). So the ISS's L1 miss on each weight row is
not what bounds the loop on silicon. The profile words cannot say what
does: they give one `pcyc/op` per op, with no split inside the router.
Unmeasured candidates are the 4-lane pool fork / join, the serial sigmoid
and pick tail, and the `sffma` chain latency. Measuring them would need
stage timers inside the op. Beside it in the same profile, ATTN_M1 reads
394 375 pcycles/op with this set against 2 451 682 with the base skel:
#170's work, which came in with the rebase onto `4eab54ef`.
