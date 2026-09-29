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
