# Measurement 170: the hf one-rounding FMA on silicon, its cost per lane count, and today's attention per L (S1)

Branch `htp/170-attn-fast`, code @ `f949a786` (the set was built from it). Plan
`docs/plans/170-attn-m1-fast-exact.md` §4 step 2 (S1; the S2 part of this
file comes with the kernel, step 7). Estimated device time: **≈ 30 min**
(state + install 4, G1 2, cost 2, today's kernel 5, CPU side 3, full model
8, summary 1, thermal waits extra). Run by the orchestrator on
`R3CY10WM83Y`:

```
bash /local/mnt/workspace/htp_moe/170/s1/run_s1.sh R3CY10WM83Y
```

(a byte-identical copy is `docs/measurements/170-s1-run.sh` on this
branch). The script logs to `/local/mnt/workspace/htp_moe/170/s1/logs/`,
writes the summary to `logs/sitting.out`, stops on the plan's stop rules
and counts every other missing expected line (`expectation mismatches:
N`). Start at zone0 ≤ 35 °C (the script waits for it).

## Why

1. The fast kernel (plan §3) runs every fused fp16 FMA of the spec as a
   qf32 multiply-add narrowed once to hf (`hvx_hf_fma`,
   `nntrainer/tensor/htp_backend/hvx/hvx_attn_m1_hf.h`). The v79 ISS and
   the host emulation say that equals the CPU's `fmla .8h` on every case;
   whether silicon's qf32 does is the question the whole design rests on
   (G1), and it is asked before any kernel is written.
2. The speed gate (G6's T64 / T1024) is set from silicon constants, not
   ISS cycles: the pcycles per 64-lane FMA of the kernel's loop shapes at
   1 / 2 / 4 / 6 lanes, the cold fetch rate with and without `l2fetch`,
   today's per-L line (warm / cold, pos 511 / 1023 / 1535) and today's
   in-model `pcyc/op ATTN_M1` at G = 64 and 1024 (the in-model / cold
   factor).

Decisions that hang on it: any qfma `bad` ≠ 0 → stop, report; the
fallback (plan §3.6, double rounding + hazard flag, 1.3×) is a user
decision. qfma `bad=0` → the model below sets T64 / T1024; a model above
600 k pcyc at G = 1024 → stop after S1 and report (plan §1 G6). exp16
`bad=0` → the vector exp is kept (else the table, plan §3.3). div16
`bad` ≠ 0 → see Notes (it is today's divide too). The lanes sweep sets the
unit split (§3.4); the fetch pair sets `l2fetch` on or off.

### Run dirs (two, so no stub meets a foreign skel)

| dir | what | used by |
|---|---|---|
| `s170p` (`…/causallm/s170p`) | this branch's skel (the new `attn_m1_probe` entry; **the attention kernel is today's**, unchanged since #164) + `unittest_hvx_attn` + `unittest_nntrainer_cpu_backend_fp16` + the case file | cells (a)–(d) |
| `s170a` (`…/causallm/s170a`) | the unchanged #164 set (`164/set/`, same md5s) | cell (e) |

### Cells

| cell | binary / env | expected |
|---|---|---|
| (a) G1 | `unittest_hvx_attn --gtest_filter=HvxAttnM1Probe.Semantics`, `NNTR_ATTN_FMA_CASES=attn_fma_cases.bin` | the 11 `ATTN_M1_PROBE` lines below, each `bad=0` |
| (b) cost | `… HvxAttnM1Probe.Cost` | 24 `ATTN_M1_PROBE_COST` lines (6 ops × lanes 1 / 2 / 4 / 6), `ran` = `lanes` |
| (c) today's kernel | `… HvxAttnM1.*` | `ATTN_M1_FIELD L=<L> bad=0` at L = 1 / 63 / 64 / 65 / 512 / **513 / 1024 / 1536**, `append_chain … bad=0`, 3 warm + 3 cold `ATTN_M1_PHASE` lines (pos 511 / 1023 / 1535); `bad_stats` recorded as the reference for S2's G3 (rule 37) |
| (d) CPU side | `unittest_nntrainer_cpu_backend_fp16 --gtest_filter=AttnM1F16Det.*` | `ATTN_M1_F16 L=513/1024/1536 rope_pos=… out bad=0`, no `FAILED` |
| (e) full model, s170a | **A** (switch off, first) then **Q0-prof** (`NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=MOE,QK_NORM,ROPE,ATTN_M1 NNTR_HTP_PROFILE=2`), prompt 512, G = 64 then 1024, ×1 each, `NNTR_NUM_THREADS=8` | A: `dspq: on` once, `dspq: close calls=N served=N bad=0`, no `graph:` line. Q0: `graph: init n_ops=228 resident=QK_NORM\|ROPE\|ATTN_M1\|MOE moe_ops=22`, `calls/token=28.00`, one `graph: calls=… pcyc/op: … ATTN_M1=…` line (else the row is **void**, rule 36) |

G = 512 is not run: S1 reads no speed verdict (plan §4 step 2); S2 runs
A / Q0 / Q1 at G = 64 / 512 / 1024 × 2, mirrored. No text-approval
section: S1 runs no new arithmetic in the model (s170a is the #164 set,
whose Q text was byte-identical to A's in that sitting); the Q0-prof text
against A is printed for the record.

G1's expected lines (the reference side counted on the host from the
same generators and file; the device prints the same n / hazards /
subnormal because the ARM spec is the same IEEE code):

```
ATTN_M1_PROBE qfma cases n=4878 hazards=333 subnormal=81 bad=0
ATTN_M1_PROBE qfma adversarial n=76800 hazards=8382 subnormal=2300 bad=0
ATTN_M1_PROBE qfma zero_sign n=25600 hazards=0 subnormal=25600 bad=0
ATTN_M1_PROBE hf add n=253568 bad=0
ATTN_M1_PROBE hf sub n=253560 bad=0
ATTN_M1_PROBE hf mul n=228348 bad=0
ATTN_M1_PROBE hf max n=253952 bad=0
ATTN_M1_PROBE hf mul0.125 n=253952 bad=0
ATTN_M1_PROBE hf zero_plus n=253952 bad=0
ATTN_M1_PROBE exp16 n=31745 bad=0
ATTN_M1_PROBE div16 hard n=27049 bad=0
```

`hazards` = triples where rounding c + a·b to f32 first gives a different
fp16 than the fused spec (`amc_is_midpoint_case`): the 333 of the real
replay are all of plan §0's. The div16 set is every quotient e / l (l in
[1, 2048], e in [0, l]) whose `rne16(e · recip_det(l))` is off by one or
which is an exact tie, found on the phone with `swiglu_det_recip`; on
this domain every off-by-one case is also a tie, so the host check's
7,881 + 27,049 are 27,049 quotients.

Stop rules (the script stops): a qfma line missing or with `bad` ≠ 0;
`0x8000040e` in any log (stale skel, rule 3); the device md5s differ
from `md5.txt`.

## Artifacts

Set `/local/mnt/workspace/htp_moe/170/s1/`, `md5.txt` there (paths
relative to it):

| file | md5 | built with |
|---|---|---|
| `s170p/libnntr_hvx_skel.so` | `599ed52db03d52e0e8f5cddac7095986` | `test/htp/build.sh` (v79, HexKL 6.4.0.1): `UNDEFINED SYMBOLS OK (51 runtime imports)`; `hexagon-nm` lists `nntr_hvx_attn_m1_probe` |
| `s170p/unittest_hvx_attn` | `e1655e699d5bba2d11e64fed9440b268` | `test/jni` ndk-build (NDK r30), stub from this IDL; `HvxAttnM1Probe` in its strings |
| `s170p/unittest_nntrainer_cpu_backend_fp16` | `29b114e68d7f32be9e3035f407267dc8` | same, links the set's `libnntrainer.so` |
| `s170p/libnntrainer.so` | `aec06f69492bc91ce1cee4a12864b06d` | `build_android.sh --htp --cache` (`jni/obj/local/arm64-v8a/`; NEEDED `libsdkl.so`, `libcdsprpc.so`; `NNTR_HTP_FORWARD_KINDS` 2 in `libcausallm_core.so`) — only the fp16 gtest loads it |
| `s170p/libccapi-nntrainer.so` | `3856bbde9094ca8286e8a806059274f4` | same |
| `s170p/libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL 6.4.0.1 `armv8_android26` (the NEEDED of `libnntrainer.so`) |
| `s170p/libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 sysroot |
| `s170p/attn_fma_cases.bin` | `4ab75655226064cc3dcba73639f07d7f` | `tools/htp/attn_fma_cases.py /local/mnt/workspace/htp_moe/136/dump_attn` (`ops=12607488 hazards=333 inexact_midpoints=675 written=4878`) |
| `s170a/nntrainer_causallm` | `291d5805ac300d47b1b9efb2bf9c78e4` | the #164 set (`164/set/`, `dev/norm-shadow` @ `07718beb`), copied unchanged |
| `s170a/libcausallm_core.so` | `930fe3189f4d1496dd8b416faa822226` | same |
| `s170a/libnntrainer.so` | `067aeb3df6ce6a86344c3ae95c222cc4` | same |
| `s170a/libccapi-nntrainer.so` | `1a4452163fcee89cfcab15e171a2ca20` | same |
| `s170a/libnntr_hvx_skel.so` | `69416d72edec7ee7eeaef139f54569ee` | same (the #164 skel; its attention kernel is the one s170p carries) |
| `s170a/libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | same |
| `s170a/libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | same |
| `s170a/prompt512.txt` | `fc65c1588dc66dd764c7013fe96cbb75` | same (`docs/measurements/prompts/README.md` p01) |
| model `q40-qs4cx-wh`, `tokenizer.json` | on the phone since #100 | |

Not staged, never pushed: `libcdsprpc.so` (rule 5). The skel is not
byte-reproducible: with a rebuilt set the table is void and the md5s you
push are the record.

Workstation checks done while staging:

```
S=/local/mnt/workspace/htp_moe/170/s1
(cd $S && LC_ALL=C md5sum -c md5.txt | grep -vc ': OK$')                        # 0
strings $S/s170p/unittest_hvx_attn | grep -c HvxAttnM1Probe                     # 67
hexagon-nm $S/s170p/libnntr_hvx_skel.so | grep -c nntr_hvx_attn_m1_probe         # 1
readelf -d $S/s170p/unittest_hvx_attn | grep -c 'libcdsprpc.so\|libc++_shared'   # 2
readelf -d $S/s170p/unittest_nntrainer_cpu_backend_fp16 | grep -c 'libnntrainer\|libccapi'  # 2
diff <(cd 164/set && md5sum <the 8 s170a files>) <(cd $S/s170a && md5sum …)     # empty
find /local/mnt/workspace/htp_moe/170 -name 'libcdsprpc*' | wc -l               # 0
```

Rebuild recipe (if the set is lost): `git checkout htp/170-attn-fast`,
`source tools/htp/env.sh`, `export
HEXKL_ROOT=/home/j2z0-lee/Qualcomm/hexkl-1.0-beta.2/hexkl_addon
HEXKL_SDK_VER=6.4.0.1`, `git submodule update --init --depth 1`, copy
`Applications/CausalLM/lib/libtokenizers_android_c.a` from another
worktree, `./test/htp/build.sh`,
`nntrainer/tensor/htp_backend/generate_stub.sh`, `(cd
Applications/CausalLM && ./build_android.sh --htp)` (fresh `builddir`:
`cd builddir && meson configure -Dprefix=$PWD/android_build_result &&
ninja install`, then `--htp --cache`), the `test/jni` ndk-build of
`unittest_hvx_attn unittest_nntrainer_cpu_backend_fp16` (gates skill
rung 3), `libc++_shared.so` from the NDK sysroot, `python3
tools/htp/attn_fma_cases.py /local/mnt/workspace/htp_moe/136/dump_attn
s170p/attn_fma_cases.bin`; `s170a/` = the app files of `164/set/`.

## Results (S1 ran 2026-09-29 19:36–19:39 on `R3CY10WM83Y`, logs `/local/mnt/workspace/htp_moe/170/s1/logs/`; filled by the implementer from the logs)

Device md5 = `md5.txt` (`MD5 OK`), zone0 25.1 °C at start, 61.4 °C after
the G = 1024 cells. `expectation mismatches: 2`, both expected failures
rather than results (below). No `0x8000040e`.

### (a) G1 — silicon semantics: **pass**

| row | n | hazards | bad |
|---|---|---|---|
| qfma cases (#136 replay) | 4878 | 333 | **0** |
| qfma adversarial | 76800 | 8382 | **0** |
| qfma zero / sign | 25600 | 0 | **0** |
| hf add / sub / mul | 253568 / 253560 / 228348 | | 0 / 0 / 0 |
| hf max / ×0.125 / 0 + x | 253952 each | | 0 / 0 / 0 |
| exp16 (vector) | 31745 | | **0** |
| div16 hard | 27049 | | **0** |

The one `FAILED` in `HvxAttnM1Probe.Semantics` is the n = 63 length check:
the call returned `AEE_ERPC` (`0x80000600`), the FastRPC layer's refusal
before the entry runs (LEDGER rule 38), not the entry's
`AEE_EINVALIDFORMAT`. Fixed in `38f80800` (either code is a refusal).

### (b) cost per 64-lane FMA (pcycles, L2-resident loops) and fetch rate

`pcyc_per_fma64` = wall pcycles / all lanes' FMAs; `lane` = lane-summed
pcycles / FMA at 6 lanes; mhz 2110–2115 throughout.

| op | lanes 1 | 2 | 4 | 6 | lane (6) |
|---|---|---|---|---|---|
| fma16_sf (today) | 45.2 | 23.5 | 11.9 | 8.06 | 45.7 |
| scores1 (one q head per Kt load) | 7.82 | 4.65 | 2.45 | **1.96** | 9.26 |
| scores2 (two per load) | 6.47 | 3.60 | 2.30 | 2.22 | 12.0 |
| pv4 | 6.06 | 3.23 | 1.62 | **1.26** | 6.91 |

| fetch, 3 MiB cold | GB/s lanes 1 | 2 | 4 | 6 |
|---|---|---|---|---|
| no `l2fetch` | 11.4 | 21.7 | 34.9 | **38.1** |
| `l2fetch` lead 32 KiB | 32.9 | 23.0 | 31.7 | 32.5 |

### (c) today's kernel per L (the S2 reference line): bit-identical at all 8 L

| pos | warm `dsp_us` | cold `dsp_us` | cold scores / softmax / pv (Mpcyc, lane-summed) | `bad_stats` (L = pos + 1) |
|---|---|---|---|---|
| 511 | 763 (#152: 731) | 1093 (#152: 1099) | 7.14 / 0.47 / 4.55 | 0 |
| 1023 | 1823 (#152: 1737) | 2185 (#152: 2193) | 14.46 / 0.89 / 9.03 | 0 |
| 1535 | 3236 | 3290 | 21.72 / 1.36 / 13.70 | 0 |

`ATTN_M1_FIELD L=<L> bad=0 bad_stats=0` at L = 1 / 63 / 64 / 65 / 512 /
513 / 1024 / 1536, `append_chain … bad=0`. `HvxAttnM1.RejectsBadShapes`
failed on `AEE_ERPC` as in every sitting since #137 (rule 38; #137 owns
it); the runner's "3 of 4" expectation was wrong, 2 of 4 plus that known
failure is this sitting's normal.

### (d) CPU side: **pass**

`AttnM1F16Det` out bad = 0 at L = 513 / 1024 / 1536, both rope positions
(`[  PASSED  ] 3 tests`).

### (e) full model (s170a = the #164 set)

| variant | G | prefill tok/s | decode tok/s (all) | last 64 | `pcyc/op ATTN_M1` |
|---|---|---|---|---|---|
| A | 64 | 575.3 | 52.37 | 52.37 | — |
| Q0-prof | 64 | 568.9 | 34.10 | 34.10 | **2,469,581** |
| A | 1024 | 559.0 | 51.26 | 50.04 | — |
| Q0-prof | 1024 | 509.5 | 29.09 | 24.61 | **4,547,297** |

In-model / cold = 2,469,581 / (2,306,776 × 544.5 / 512) = **1.01** at
G = 64 and 4,547,297 / 4,612,298 = **0.99** at G = 1024 (cold = append +
pool pcycles of (c)): in the model the kernel runs at its cold gtest
rate. The runner printed "Q0-prof text vs A: DIFF" because its strip kept
the profile's `[HTP-DMA]` lines; with them removed the texts are
identical (the orchestrator's check). The runner's model line printed
"a cost line is missing" (its grep looked for `op=scores2 lanes=6 ` with
a trailing space the line does not have); the model is computed below.

### Read

* **qfma**: bad = 0 everywhere, so the design stands (no fallback).
* **Lane plan**: scores2 stops scaling past 2–4 lanes (lane cost 6.4 →
  12.0), while scores1 and pv4 scale to 6. The kernel uses scores1 (one
  q head per Kt load, 1.96 at 6 lanes) and pv4.
* **`l2fetch`**: off. The plain stream reaches 38 GB/s at 6 lanes, and
  the lead is no better at any lane count.
* **exp16**: vector (bad = 0 exhaustively).
* **T64 / T1024**: S2 section, from these constants.

## Notes from building the set (host / ISS, not device results)

* **The compiler folds explicit qf32 conversions.** `hexagon-clang 19 -O3
  -mv79` turned the exp16 chain's `Q6_Vsf_equals_Vqf32` + next qf32 op into
  `vadd(Vu.qf32, Vv.sf)` (20×) and `vmpy(Vu.qf32, Vv.qf32)` (12×), skipping
  those f32 roundings even though both were written as intrinsics (plan
  §3.3 assumed they would be kept). `hvx_attn_m1_hf.h` now pins each
  rounded value with an empty `asm` (`hvx_hf_pin`); the -S shows no mixed
  qf32 / sf op left in exp16. On the v79 ISS the pinned and unpinned
  exp16 were both bit-exact at all 31,745 d, so the fold was harmless
  there; the pinned form is the spec's rounding by construction. The same
  folding happens inside `hvx_div16_sf` / `hvx_recip_det_sf` (today's
  kernel's divide, reused by `hvx_hf_div16`); both divides were bit-exact
  on the ISS on the 27,049 hard quotients, which silicon has never run
  (attention-level data does not reach them). A div16 `bad` ≠ 0 in G1 is
  therefore also a latent fault of today's kernel.
* `Q6_Vqf32_equals_Vsf` is v81 only; the narrowing multiplies by 1.0 in
  qf32 instead.
* Probe loop shapes (static packets, `-S`): scores1 8 FMAs in 26 packets,
  no spill; pv4 8 FMAs in 24 packets, no spill; scores2 16 FMAs in 42
  packets with 10 stack accesses (16 accumulators + the q splats); today's
  `fma16_sf` loop keeps its accumulators on the stack as the kernel does.
* hvx_emu models qf32 by what its one consumer returns (the exact sum
  rounded to odd for the narrowing to hf), not as a format: it cannot see
  a silicon qf32 that narrows differently, which is what G1 reads.
* Rebase order with #132 PR 2 (`htp/132-exact-fc`, also appends to the
  IDL and edits `htp_compute_ops.cpp`): whichever lands second rebases;
  this branch appends one method after `dspq_stop` and adds one skel
  source, and touches no ARM file in step 1.

## Notes from the run (S1)

zone0 25.1 → 30.1 °C over the gtests, 59.4 / 61.4 °C after the two
full-model pairs; battery 100 %. No FARF / AEE error besides rule 38's
`AEE_ERPC` on the two bad-length checks.

---

# S2: the fp16-lane kernel on silicon (plan 170 step 7)

PR code `htp/170-attn-fast` @ `38f80800` (kernel `2457b311`, consumers
`8d28fab0`); the device set is built from `dev/attn-shadow-170` @
`ca2dd5f8` (= the PR plus one inert measurement commit, never merged,
pushed). Estimated device time: **≈ 55 min** (state + install 4, G3
gtests 7, shadow 4, speed 12, G6 profiles 5, G5 nll + text 20, checks 2,
thermal waits extra). Run by the orchestrator:

```
bash /local/mnt/workspace/htp_moe/170/s2/run_s2.sh R3CY10WM83Y
```

(a byte-identical copy is `docs/measurements/170-s2-run.sh`). Logs go to
`/local/mnt/workspace/htp_moe/170/s2/logs/`, the summary to
`logs/sitting.out`. The script stops on `0x8000040e` or a device md5
mismatch, and counts every other missing expected line.

## Why (S2)

The kernel now runs the spec in fp16 lanes (qfma, exact on silicon in
S1). S2 reads four things: bit identity at L = 1 … 1536 on silicon
(G3); the CPU-vs-HTP attention per (layer, step, q head) inside the model
(G4); text and nll against A (G5); and the in-model `pcyc/op ATTN_M1`
against T64 / T1024 (G6). The issue closes on G1–G6.

## The lane plan and the projected per-layer cost (S1's silicon constants)

S1 showed that the one-head score shape (scores1: 1.96 wall pcycles per
64-lane FMA at 6 lanes) and the four-head PV shape (pv4: 1.26, 6.9 per
lane) scale to 6 lanes, while the two-head score shape does not. The
fetch reaches 38 GB/s at 6 lanes, and `l2fetch` adds nothing. The kernel
(`hvx_attn_m1_f32.c`'s header) runs:

| step | where | units at LFM2.5 | per layer (L positions) |
|---|---|---|---|
| append, q to fp16 | caller | — | ≈ 3 k integer conversions |
| P1 scores | pool | 8 kv × ⌈L/64⌉ tiles (64–192 units: balanced) | 32 L FMAs, scores1 shape; Kt tile read from DDR once per unit (1 KiB·L) |
| max | caller | — | 32 · L/64 hf vmax |
| P2 exp + scatter | pool | as P1 | 32 L exp16 (widen, 2 × 45 explicit sf ops, narrow) + 32 L halfword scatter into ET |
| sum | caller | — | L dependent hf adds, all q heads at once |
| P3 divide + PV | pool | 8 kv × 1 group of 4 q heads (8 units on 6 lanes) | 32 L divides + 32 L FMAs, pv4 shape; V read once (1 KiB·L) |

Kernel inner loops (`-S`, v79): P1 8 FMAs per 22–24 packets (two passes
of four accumulators; one reload of the 1.0 splat constant, no
accumulator spill), P3 4 FMAs per 11 packets, no spill.

Model, in pcycles at 2.11 GHz, 6 lanes:

* scores = 32 L × 1.96 = **62.7 L**
* PV = the busiest lane runs 2 of the 8 units = 8 L FMAs × 6.9 = **55 L**
* exp + scatter ≈ 32 L × 170 packets / 64 / 6 lanes = **14 L**
* divide ≈ 2 units × 4 heads × L × 60 / 64 = **7.5 L**
* max + sum (serial) ≈ **6 L**
* fixed (append, q, three pool runs, output conversion) ≈ **40 k**
* the fp16 fetch = 2 KiB·L at 38 GB/s = **114 L**, of which P1 reads
  half and P3 the other half

Compute is **145 L + 40 k**. With full fetch overlap the total is the
same; with none it is **259 L + 40 k**:

| L | model, overlap … none | µs at 2.11 GHz | today in-model (S1) | CPU (E2E / microbench) |
|---|---|---|---|---|
| 513 | 114 k … 173 k pcyc | 54 … 82 | 2.47 M (G = 64, mean L 544.5) | — / 77 µs |
| 1024 | 188 k … 305 k | 89 … 145 | 4.55 M (G = 1024, mean L 1024.5) | 280–310 / 140 µs |
| 1536 | 263 k … 438 k | 125 … 207 | 6.95 M (cold gtest, pos 1535) | — / 217 µs |

**Gates (G6)**: T = 1.15 × the no-overlap model at the mean L of each run
(544.5 at G = 64, 1024.5 at G = 1024): **T64 = 210 000** and **T1024 =
350 000** pcyc/op. Both are below the plan's provisional 300 k / 600 k.
At T1024 the kernel is ≈ 0.17 ms per layer, under the CPU's 0.28–0.31 ms
end-to-end and above the 0.14 ms microbench.

Not in the model: the v79 ISS, one thread, memory model on, gave 1.39 M
pcyc at L = 1024 (pool 1.39 M: scores 0.44 M, softmax part 0.56 M, PV
0.41 M) and 0.11 M fixed. Divided over 6 lanes that is ≈ 230–300 k,
inside the range above. The ISS is not silicon (#164: 4× off on DDR), so
it cross-checks the order of magnitude and nothing more.

## Variants (4; A first; `NNTR_NUM_THREADS=8`)

| | dir | env | expected banner (else **void**, rule 36) |
|---|---|---|---|
| **A** (reference) | `new` | none | `dspq: on`, `dspq: close calls=N served=N bad=0`, no `graph:` |
| **Q0** | `q0` (#164 set) | `NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=MOE,QK_NORM,ROPE,ATTN_M1` | `resident=QK_NORM\|ROPE\|ATTN_M1\|MOE`, `calls/token=28.00`, `cache=49152 KiB` |
| **Q1** | `new` | same | same banner, `calls/token=28.00`, **`cache=24576 KiB`** |
| **RQ1** | `new` | `… KINDS=MOE,RMSNORM,QK_NORM,ROPE,ATTN_M1` | `resident=RMSNORM\|QK_NORM\|ROPE\|ATTN_M1\|MOE`, `calls/token=77.00`, `cache=24576 KiB` |

`NNTR_ATTN_SHADOW` / `NNTR_LOGIT_SHADOW` are set only in the G4 cells.
Without them the dev commit takes one `getenv` per process and changes
nothing: on the host, the HTP dumps with the shadow on and off are
bit-identical (hd64 fixture, 40/40 files).

## Artifacts (S2)

Set `/local/mnt/workspace/htp_moe/170/s2/`, `md5.txt` there (paths
relative to it):

| file | md5 | built with |
|---|---|---|
| `new/libnntr_hvx_skel.so` | `f673f1a3d9e56e23e0cb3cad6649d759` | `test/htp/build.sh` (v79, HexKL 6.4.0.1) on `ca2dd5f8` (the DSP sources are the PR's): `UNDEFINED SYMBOLS OK (51 runtime imports)` |
| `new/nntrainer_causallm` | `ff4b7d7739d5aad3d85ad79ba05432e8` | `build_android.sh --htp --cache` after `ninja install` in `builddir` |
| `new/libcausallm_core.so` | `233da2d87cfbf5f8450c4b609496122f` | same (`NNTR_HTP_FORWARD_KINDS` 2, `NNTR_ATTN_SHADOW` 1) |
| `new/libnntrainer.so` | `5e2ff8cc902530791a8f7e4e85051f79` | same (`jni/obj/local`; NEEDED `libsdkl.so`, `libcdsprpc.so`; `graph: forward calls` 1; `dspq: on` 1; the banner counts 2 bytes) |
| `new/libccapi-nntrainer.so` | `3856bbde9094ca8286e8a806059274f4` | same |
| `new/unittest_hvx_attn` | `9169e94e25a0b1a68a903fe6cbc0424b` | `test/jni` ndk-build (`HvxAttnM1.*`, `HvxAttnM1Probe.*`) |
| `new/unittest_nntrainer_cpu_backend_fp16` | `997ad7d625e2a8eba92bd7d2af751164` | same (`AttnM1F16Det.*`) |
| `new/unittest_hvx_softmax` | `4b554ada44de07b185eb57905c4b1735` | same (`HvxM1Ops.Rope64*`) |
| `new/libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL 6.4.0.1 |
| `new/libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 sysroot |
| `new/prompt512.txt`, `new/bitset-02 … 08` | as `164/set/` (p01 `fc65c158…`) | the #164 prompts |
| `q0/*` (8 files) | as S1's `s170a` (skel `69416d72…`, app `291d5805…`) | the #164 set, unchanged |
| `attn_shadow_check.py` (not pushed) | `4c88617dfc12f8a8f94260a4e7566ece` | `tools/htp/attn_shadow_check.py` of `ca2dd5f8` |

PR-only build of `38f80800` (not staged): skel `0d12f237…`, app
`ff4b7d77…`, `libcausallm_core.so` `b4fbb005…`, `libnntrainer.so`
`5e2ff8cc…`, the same three gtest md5s. The skel is not
byte-reproducible.

Rebuild recipe: `git checkout dev/attn-shadow-170`, then as S1's recipe:
`./test/htp/build.sh`, `generate_stub.sh`, `(cd builddir && ninja
install)` (`--cache` skips the nntrainer build), `build_android.sh --htp
--cache`, the `test/jni` ndk-build of `unittest_hvx_attn
unittest_nntrainer_cpu_backend_fp16 unittest_hvx_softmax`.

## Cells and expected lines (what `run_s2.sh` checks)

1. **Install**: `md5sum -c`, push `new/` to `…/causallm/s170n` and `q0/`
   to `…/s170q0`, device md5 against `md5.txt` (stop on a mismatch), the
   model config as S1.
2. **G3** (`new`): `ATTN_M1_FIELD L=<L> bad=0 bad_stats=0 of 2048` at L =
   1 / 63 / 64 / 65 / 512 / 513 / 1024 / 1536 (S1's `bad_stats` reference
   is 0 at every L), `append_chain L=65 bad=0`, 6 `ATTN_M1_PHASE` lines
   (pos 511 / 1023 / 1535 warm and cold: the new kernel's per-L line);
   `ATTN_M1_F16 L=513/1024/1536 … out bad=0` ×2 each; `M1_OPS_FIELD
   rope64 pos=… bad=0 of 2560` ×5. `RejectsBadShapes` is expected to
   fail on `AEE_ERPC` (#137) and is not counted.
3. **G4** (prompt 512, G = 8, `NNTR_PPL_DECODE` forced on A's tokens):
   A, Q1, RQ1 with `NNTR_ATTN_SHADOW=d_f_<v>/attn.bin
   NNTR_LOGIT_SHADOW=d_f_<v>/logits.bin`, pulled to `s2/shadow/`;
   `attn_shadow_check.py d_f_A d_f_A d_f_Q1 d_f_RQ1`:
   ```
   ATTN SHADOW d_f_Q1 tag3_heads=1536/1536 records=48 layers=6 positions=8 zero_records=0 logits_equal_steps=8/8
   ATTN SHADOW d_f_RQ1 tag3_heads=1536/1536 records=48 layers=6 positions=8 zero_records=0 logits_equal_steps=8/8
   ```
   and every `[PPL] decode step` line of Q1 / RQ1 equal to A's.
4. **Speed**: A / Q0 / Q1 at G = 64 / 512 / 1024, run 1 in order A Q0 Q1
   and run 2 in order Q1 Q0 A, banners as the table above.
5. **G6**: Q0-prof and Q1-prof (`NNTR_HTP_PROFILE=2`) at G = 64 and 1024;
   Q1's `pcyc/op … ATTN_M1=` ≤ **210 000** (G = 64) and ≤ **350 000**
   (G = 1024).
6. **G5**: 8 prompts at G = 256, `NNTR_PPL_DECODE` (A self, A forced once
   as a null check, Q1 / RQ1 forced on A's ids); free-text runs A / Q0 /
   Q1 / RQ1. Expected: every nll step line of Q1 and RQ1 equal to A's;
   Q1 and RQ1 text = A byte for byte; Q1 text = Q0 text (the tokens do
   not move with the kernel). The strip removes every `[HTP…]` and
   `[PPL]` line (S1's DIFF came from `[HTP-DMA]` lines).

## Results S2 (fill in)

| gate | line | value | pass |
|---|---|---|---|
| G3 | out bad / bad_stats at 8 L; F16Det; rope | | |
| G4 | Q1 / RQ1 tag3_heads, logits_equal_steps | | |
| G5 | nll = A (16 pairs); text Q1 = A, RQ1 = A, Q1 = Q0 (8 prompts each) | | |
| G6 | Q1 `pcyc/op ATTN_M1` G = 64 / 1024 (≤ 210 k / 350 k); Q0 same cells | | |

| variant | G | run | prefill tok/s | decode tok/s (all) | last 64 | text = A | `calls/token` |
|---|---|---|---|---|---|---|---|
| A | 64 / 512 / 1024 | 1, 2 | | | | ref | — |
| Q0 | 64 / 512 / 1024 | 1, 2 | | | | | 28.00 |
| Q1 | 64 / 512 / 1024 | 1, 2 | | | | | 28.00 |

Reference: S1 A 52.37 / 51.26 tok/s at G = 64 / 1024 (cool phone), Q0
34.10 / 29.09. #164: A 45.3 / 47.2, Q 33.9 / 25.4. Standing checks:
prefill ≥ −5 % of A; Q1 decode ≥ 0.95 × A at G = 512 / 1024 is read, not
gated.

| pos | new warm `dsp_us` | new cold `dsp_us` | cold scores / softmax / pv (lane-summed) | S1 (old kernel) cold |
|---|---|---|---|---|
| 511 | | | | 1093 |
| 1023 | | | | 2185 |
| 1535 | | | | 3290 |

## Text approval (per-token-entry handoff)

| variant | decode PPL (forced on A, prompt p01, G = 256) | generated text (p01, G = 256) | text approved (user: y/n) |
|---|---|---|---|
| A | | <paste> | (reference) |
| Q1 | | <paste> | |
| RQ1 | | <paste> | |

## Notes (S2 build)

* **The shadow's QK_NORM detail.** With QK_NORM resident, the stretch
  leaves the qkv layer's Q / K outputs unwritten, so the attention layer's
  input rows are stale. In the dev commit, with the shadow on, the qkv
  layer writes the CPU norm there (equal to the DSP's, #164 G2); the
  stretch reads its own held row, not these outputs. Without this the
  host shadow read a 70 % gap; with it 2e-4, which is the host CPU's f32
  attention against the fp16 kernel. On Android the CPU attention is
  fp16, so bit equality is expected.
* **ET scatter** is scalar (64 halfword stores per head and tile).
  ponytail: an in-register vshuff transpose of the 4-head block would
  cut the stores 4×, worth it if S2's SOFTMAX word is large next to
  SCORES + PV.
* **P3 imbalance**: 8 units on 6 lanes. Splitting two kv heads into head
  pairs would even it out, worth it if BUSY_MAX ≫ (SCORES + SOFTMAX + PV)
  / LANES.
* **Rebase order with #132 PR 2**: this branch appends one IDL method,
  edits the `attn_m1_*` comment block of the IDL, one line of
  `htp_compute_ops.cpp` (the banner) and one skel source line in
  `build.sh`; whichever lands second rebases.

## Notes from the run (S2)

<thermal, first-run page faults, FARF/AEE errors, anything stale>
