# Measurement 164: the RMSNORM / QK_NORM row scale in the Android CPU's order, on silicon

Branch `htp/164-cpu-order-norm`, PR code @ `41b03160`; the device set is
built from `dev/norm-shadow` @ `07718beb` (= the PR diff plus the inert
measurement commits, never merged; the arrangement of 2026-09-29; pushed as
`dev/norm-shadow-164`, since the rebased branch cannot replace
`origin/dev/norm-shadow` without a force push). Plan
`docs/plans/164-cpu-order-norm-scale.md` §4 step 8. Estimated device time:
**≈ 55 min** (state + install 4, gtests 4, norm shadow 4, dumps 5, speed
14, nll + text 20, checks 2). Run by the orchestrator:

```
bash /local/mnt/workspace/htp_moe/164/run_164.sh R3CY10WM83Y
```

(a byte-identical copy is `docs/measurements/164-run.sh` on this branch).
The script logs to `/local/mnt/workspace/htp_moe/164/logs/`, writes the
summary to `logs/sitting.out`, stops on the plan's stop rules and counts
every other missing expected line (`expectation mismatches: N`).

## Why

1. The resident RMSNORM and QK_NORM differed from the Android CPU only in
   the row scale r (plan §0: with the true gamma, forcing r to the CPU's
   reproduced 392/392 dumped rows). The kernel now computes r in the CPU's
   order — 16 fused chains on the scalar core (`sffma`), the `faddp`
   reduction, an integer RN sqrt and RN reciprocal — and the host replays
   the 2026-09-29 dumps bit for bit (`REPLAY rms=784/784
   qk_heads=7680/7680`). This sitting reads whether that holds on silicon.
2. Decision: G0–G5 ✓ closes #164 (masks `MOE,RMSNORM` and
   `MOE,QK_NORM,ROPE,ATTN_M1` bit-identical to A). G5 ✗ alone → plan §3.3
   ladder (a). G1 kind 2 (the ±1e-39 row) ✗ with the normal kinds ✓ →
   plan step 5.

### Variants (4, one set, switched by env)

| | env (beyond `NNTR_NUM_THREADS=8`) | expected banner (else **void**, rule 36) |
|---|---|---|
| **A** (reference, first) | none | `[HTP] dspq: on` once, `dspq: close calls=N served=N bad=0`, no `graph:` line |
| **R** | `NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=MOE,RMSNORM` | `graph: init n_ops=228 resident=RMSNORM\|MOE moe_ops=22`, `calls/token=71.00` |
| **Q** | `… KINDS=MOE,QK_NORM,ROPE,ATTN_M1` | `resident=QK_NORM\|ROPE\|ATTN_M1\|MOE`, `calls/token=28.00` |
| **RQ** | `… KINDS=MOE,RMSNORM,QK_NORM,ROPE,ATTN_M1` | `resident=RMSNORM\|QK_NORM\|ROPE\|ATTN_M1\|MOE`, `calls/token=77.00` |

The changed kernels are not reached in A (switch off); with
`NNTR_NORM_SHADOW` set, every variant (A too) also calls the DSP's
`rmsnorm_det_f32` entry once per q and k of each attention layer to fill
the tag-1 record — an observation on the side, it writes nothing the
model reads.

## Artifacts

Set `/local/mnt/workspace/htp_moe/164/set/`, `md5.txt` inside it:

| file | md5 (device set) | PR-only md5 (`41b03160`, not pushed) | built with |
|---|---|---|---|
| `libnntr_hvx_skel.so` | `69416d72edec7ee7eeaef139f54569ee` | `fac85da4987c5917d7c01c07fb452ced` | `test/htp/build.sh` (v79, HexKL 6.4.0.1): `UNDEFINED SYMBOLS OK (51 runtime imports)`; the DSP sources are the same in both trees |
| `nntrainer_causallm` | `291d5805ac300d47b1b9efb2bf9c78e4` | `1dca15740514eb06805b3f44f699065c` | `build_android.sh --htp --cache` (`jni/libs/arm64-v8a/`) |
| `libcausallm_core.so` | `930fe3189f4d1496dd8b416faa822226` | `e8cba7a149d649d474bab165ff0d0851` | same (`NNTR_HTP_FORWARD_KINDS` 2; `NNTR_NORM_SHADOW` 1 in the set, 0 PR-only) |
| `libnntrainer.so` | `067aeb3df6ce6a86344c3ae95c222cc4` | `fbb73614c4d4c10ac34732a30c52e599` | same (`jni/obj/local/arm64-v8a/`; NEEDED `libsdkl.so`, `libcdsprpc.so`; `dspq: on` 1; `graph: forward calls` 1) |
| `libccapi-nntrainer.so` | `1a4452163fcee89cfcab15e171a2ca20` | `0708b2aa727a94c0d30a51e4344e5a49` | same |
| `unittest_nntrainer_cpu_backend` | `e2b2cc384bb1c1c01c6efdc005ada694` | | `test/jni` ndk-build, links the set's `libnntrainer.so` (`RmsNormCpuOrder.*`, G0) |
| `unittest_hvx_softmax` | `43a25ae7694053801588cc7f9eabd5fc` | | same (`HvxM1Ops.*`, G1) |
| `libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | | NDK r30 sysroot |
| `libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | | HexKL 6.4.0.1 (from `norm/set/`) |
| `gamma_rms.f32` / `gamma_qk.f32` | `6698d86bd1d3897ed1d927d2b0fda808` / `9e4e44f2d57a45774f2d390b03786c87` | | `tools/htp/norm_gammas.py hf/model.safetensors` (dev branch): 49 x 2048 and 12 x 64, equal to the planner's |
| `replay_norm.bin` | `69bc505ac1ecd07e1ca6a6364c4150eb` | | `norm/d_f_RQ/norm.bin` of 2026-09-29 (392 tag-0 + 48 tag-1 records) |
| `prompt512.txt` (p01), `bitset-02-code.txt` … `bitset-08-short.txt` | as `docs/measurements/prompts/README.md` (p01 `fc65c158…`) | | from `152/set/` |
| model `q40-qs4cx-wh`, `tokenizer.json` | on the phone since #100 | | |

The set's app binaries were built from the tree at the dev commit before
the last rebase; `git diff` of that tree against `07718beb` lists only
`test/htp/host/m1_ops_host_check.c` and
`test/unittest/unittest_nntrainer_cpu_backend.cpp`, and the gtest was
rebuilt after it.

Not staged, never pushed: `libcdsprpc.so` (rule 5). The skel is not
byte-reproducible: with a rebuilt set the table is void and the md5s you
push are the record.

Workstation checks done while staging:

```
S=/local/mnt/workspace/htp_moe/164/set
(cd $S && LC_ALL=C md5sum -c md5.txt | grep -vc ': OK$')                 # 0
strings $S/libnntrainer.so | grep -c 'graph: forward calls'             # 1
strings $S/libnntrainer.so | grep -c 'dspq: on'                         # 1
strings $S/libcausallm_core.so | grep -c NNTR_HTP_FORWARD_KINDS         # 2 (rule 36)
strings $S/libcausallm_core.so | grep -c NNTR_NORM_SHADOW               # 1 (the dev commit)
strings $S/libnntrainer.so | grep -c 'dev rmsnorm_det_f32'              # 1 (the dev wrapper)
strings $S/nntrainer_causallm | grep -c 'per-layer-type totals'         # 0 (not a profile binary)
find /local/mnt/workspace/htp_moe/164 -name 'libcdsprpc*' | wc -l       # 0
```

Plan step 7 (`llvm-objdump -d -C` of the staged `libnntrainer.so`, NDK
r30): `nntrainer::neon::rms_norm_wrt_width_fp32_intrinsic` @ `0x371494`
runs four `fmla v.4s, x, x` in its 16-wide loop, `faddp v.4s` then `faddp
s` per accumulator, three `fadd s`, `fdiv` by W, `fadd` eps, `fsqrt`,
`fdiv` 1 / q, then `fmul v.4s, v, v3.s[0]` — plan §0's order, no
`frsqrte`. G0's premise holds for this set.

Rebuild recipe (if the set is lost): `git checkout dev/norm-shadow` (the
commits above), `source tools/htp/env.sh`, `export
HEXKL_ROOT=/home/j2z0-lee/Qualcomm/hexkl-1.0-beta.2/hexkl_addon
HEXKL_SDK_VER=6.4.0.1`, `git submodule update --init --depth 1`, copy
`Applications/CausalLM/lib/libtokenizers_android_c.a` from another
worktree, `./test/htp/build.sh`, `(cd Applications/CausalLM &&
./build_android.sh --htp)` (fresh `builddir`: `cd builddir && meson
configure -Dprefix=$PWD/android_build_result && ninja install`, then
`--htp --cache`), the `test/jni` ndk-build of `unittest_hvx_softmax
unittest_nntrainer_cpu_backend` (gates skill rung 3), `libc++_shared.so`
from the NDK sysroot, `python3 tools/htp/norm_gammas.py
<hf/model.safetensors> set/`, `replay_norm.bin` =
`/local/mnt/workspace/htp_moe/norm/d_f_RQ/norm.bin`.

## Steps (what `run_164.sh` does)

0. **Device state.** `adb devices`, screen off, checkpoint t0, wait for
   zone0 ≤ 35 °C.
1. **Install.** Workstation `md5sum -c`, push `set/` to `…/causallm/s164`,
   device `md5sum` diffed against `md5.txt` (stop on a mismatch); the
   config of #134 / #152 (greedy, `bad_word_ids [124900]`,
   `init_seq_len 512`, `moe_engine htp`, `moe_htp_layers ""`) re-applied.
2. **(a) gtests.**
   * G0 `unittest_nntrainer_cpu_backend --gtest_filter='RmsNormCpuOrder.*'`
     with `NNTR_NORM_REPLAY=replay_norm.bin`. Expected:
     `RmsNormCpuOrder W=2048 rows/call=1 subnormal(+-1e-39) bad=0`,
     `RmsNormCpuOrder W=64 rows/call=40 subnormal(+-1e-39) bad=0`,
     `RmsNormCpuOrder rows=41041 bad=0`, `RmsNormCpuOrder replay … rms=392
     bad=0 qk_heads=1920 bad_calls=0`, `[  PASSED  ] 2 tests`.
     **Stop** if `rows=… bad≠0`: the CPU is not what plan §0 read.
   * G1 `unittest_hvx_softmax --gtest_filter='HvxM1Ops.*'`. Expected:
     `M1_OPS_FIELD rmsnorm kind=0|1|3 bad_y=0 bad_row_scale=0 of 2048`,
     `M1_OPS_FIELD rmsnorm sweep 2^-20..2^20 rows=41 bad_y=0
     bad_row_scale=0`, `M1_OPS_FIELD qk_norm heads=32|8 bad_y=0
     bad_row_scale=0`, `M1_OPS_FIELD qk_norm sweep … heads=80 bad_y=0
     bad_row_scale=0`. Kind 2 (±1e-39) is printed and read (rule 37, plan
     step 5); it also sits in `qk_norm heads=32|8` as head 1. **Stop** on a
     normal-row failure (stale skel `0x8000040e` or silicon).
3. **(b) Norm shadow, G2.** Prompt 512, G = 8, `NNTR_NORM_SHADOW` +
   `NNTR_LOGIT_SHADOW` + `NNTR_PPL_DECODE=cont.ids` for A (self, writes
   `cont.ids`), then R, Q, RQ forced on it. Pulled to `164/shadow/`.
   `tools/htp/norm_shadow_check.py` expected:
   * `d_f_A tag0=0/0 tag1_heads=1920/1920 tag1_zero_records=0 tag2=392/392
     logits_equal_steps=8/8` against the **2026-09-29 A** (cross-sitting
     determinism) and
   * `d_f_R tag0=392/392 tag1_heads=1920/1920 … logits_equal_steps=8/8`,
     `d_f_Q tag0=0/0 tag1_heads=1920/1920 … tag2=392/392 …=8/8`,
     `d_f_RQ tag0=392/392 tag1_heads=1920/1920 … =8/8` against this A;
   * every `[PPL] decode step` line of R, Q, RQ equal to A's;
   * the host replay of these dumps (`m1_ops_host_check --replay`, kernel
     on hvx_emu): `REPLAY rms=784/784 qk_heads=7680/7680
     (rms_cpu=784/784)`.
4. **(c) Dumps at G = 4, G3.** `NNTR_HTP_DUMP` for A1, A2, R and Q (the
   last two with `NNTR_HTP_DUMP_ALL=1`), `tools/htp/htp_dump_eval.py A1
   <run>`: `bit_identical=1` for A2 (null check), R and Q.
5. **(d) Speed.** Prompt 512, G = 64 / 512 / 1024, mirrored `A R Q RQ | RQ Q
   R A`, thermal checkpoint per G; then one `NNTR_HTP_PROFILE=2` RQ run at
   G = 64 (G5: the `[HTP-PROFILE]   graph: calls=… pcyc/op: …` line,
   `RMSNORM ≤ 4000`, `QK_NORM ≤ 25000`).
6. **(e) nll + text, G4.** The 8 prompts of `docs/measurements/prompts/`
   at G = 256: A with `NNTR_PPL_DECODE=cont_p0i.ids` (self), A forced once
   on p01 (null check), R / Q / RQ forced; then a plain A and RQ run per
   prompt for the text (`cmp` after measurement 152 §5's strip) and
   `tools/htp/loop_check.py`.
7. **Checks.** Printed to `sitting.out`: nll equality per prompt and
   variant, text A vs RQ, loops, the speed table with a `text=same|DIFF`
   column against A run 1 of the same G, G2 / G3 / G5 lines, thermals.

## Gates (plan 164 §1)

| # | check | pass |
|---|---|---|
| G0 | `RmsNormCpuOrder.*` (step 2) | `bad=0` on 41 041 rows at W = 2048 and 64, the ±1e-39 rows and the replayed 2026-09-29 rows |
| G1 | `HvxM1Ops.*` (step 2) | `bad_y=0 bad_row_scale=0` on kinds 0 / 1 / 3 and both sweeps; kind 2 reported |
| G2 | norm shadow (step 3) | tag 0 392/392 in R and RQ; tag 1 1920/1920 heads in Q and RQ (and A, R); logits 8/8 equal to A in R, Q, RQ; A's logits equal to the 2026-09-29 A's |
| G3 | dumps (step 4) | `bit_identical=1` for R and Q (null check A2 = 1) |
| G4 | step 6 | every `[PPL] decode step` line of R, Q, RQ equal to A's on the 8 prompts; text RQ ≡ A 8/8 |
| G5 | profile (step 5) | `RMSNORM ≤ 4000`, `QK_NORM ≤ 25000` pcyc/op |
| standing | every speed cell | prefill ≥ −5 % of A (mirrored band) |

Decode tok/s of R / Q / RQ is recorded, not gated: the call count (71 / 28
/ 77 per token) bounds it, not these kernels (#162 owns it).

## Results (fill in)

Unit: ______ , date / time: ______ , `MD5 OK`: ___ , `expectation
mismatches:` ___

### Gtests (G0, G1)

| line | value |
|---|---|
| G0 `RmsNormCpuOrder rows=` | |
| G0 subnormal W = 2048 / 64 | |
| G0 replay | |
| G1 rmsnorm kind 0 / 1 / 3 | |
| G1 rmsnorm kind 2 (±1e-39, rule 37) | |
| G1 rmsnorm sweep | |
| G1 qk_norm heads 32 / 8 | |
| G1 qk_norm sweep | |
| G1 overflow row (not gated) | |

### Norm shadow (G2) and dumps (G3)

```
(paste logs/shadow_check.txt, logs/host_replay.txt last line, logs/dump_eval.txt)
```

| run | tag0 | tag1 heads | tag2 | logits = A | nll = A | dump bit_identical |
|---|---|---|---|---|---|---|
| A (vs 2026-09-29 A) | — | | | | (reference) | A2: |
| R | | | — | | | |
| Q | — | | | | | |
| RQ | | | — | | | (not dumped) |

Before (2026-09-29, the old r): tag0 185/392 (R) and 195/392 (RQ); logits
equal 0/8 (R, RQ) and 1/8 (Q).

### Speed (prompt 512; mirrored per G)

| variant | G | run | prefill tok/s | decode tok/s (all) | decode tok/s (last 64) | text = A r1? | calls/token |
|---|---|---|---|---|---|---|---|
| A | 64 | 1 | | | | (reference) | — |
| R | 64 | 1 | | | | | |
| Q | 64 | 1 | | | | | |
| RQ | 64 | 1 | | | | | |
| RQ | 64 | 2 | | | | | |
| Q | 64 | 2 | | | | | |
| R | 64 | 2 | | | | | |
| A | 64 | 2 | | | | | — |
| (same 8 rows for G = 512 and 1024) | | | | | | | |

| G | A mean | R mean (vs A) | Q mean (vs A) | RQ mean (vs A) | prefill band |
|---|---|---|---|---|---|
| 64 | | | | | |
| 512 | | | | | |
| 1024 | | | | | |

G5: `RMSNORM=` ___ `QK_NORM=` ___ pcyc/op (host ISS estimate: 2 311 and
17 181 per call, plan §3.3).

Reference (2026-09-29 norm sitting, G = 64, one run each, old r): A 52.94,
R 40.95, Q 33.81, RQ 30.16 decode tok/s; prefill 576.6 / 575.3 / 517.7 /
547.0.

### nll and text (G4), G = 256, forced on A's continuation

| prompt | tokens | nll R = A | nll Q = A | nll RQ = A | text RQ = A | A loop | RQ loop |
|---|---|---|---|---|---|---|---|
| p01 `prompt512.txt` | 512 | | | | | | |
| p02 `bitset-02-code.txt` | 207 | | | | | | |
| p03 `bitset-03-math.txt` | 117 | | | | | | |
| p04 `bitset-04-korean.txt` | 326 | | | | | | |
| p05 `bitset-05-json.txt` | 207 | | | | | | |
| p06 `bitset-06-dialogue.txt` | 276 | | | | | | |
| p07 `bitset-07-facts.txt` | 402 | | | | | | |
| p08 `bitset-08-short.txt` | 24 | | | | | | |

A forced p01 = A self (null check): ___

### Text approval

A bit-identical text needs no approval; a `DIFFERENT` row fails G4 and is
pasted here in full (A's and RQ's G = 256 text) for the record.

## Notes from the run

<serial, battery, thermals (logs/therm.log), anything void or stale>
