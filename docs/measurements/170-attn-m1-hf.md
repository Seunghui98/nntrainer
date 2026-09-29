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

## Results (fill in)

### (a) G1 — silicon semantics

| row | n | hazards | bad | pass |
|---|---|---|---|---|
| qfma cases (#136 replay) | 4878 | 333 | | |
| qfma adversarial | 76800 | 8382 | | |
| qfma zero / sign | 25600 | 0 | | |
| hf add / sub / mul | 253568 / 253560 / 228348 | | | |
| hf max / ×0.125 / 0 + x | 253952 each | | | |
| exp16 (vector) | 31745 | | | |
| div16 hard | 27049 | | | |

### (b) cost per 64-lane FMA (pcycles; L2-resident loops) and fetch rate

`pcyc_per_fma64` = wall pcycles / all lanes' FMAs (the throughput the
model uses); `lane` = lane-summed pcycles / FMAs (#152's measure: 24.6
per 32-lane FMA, i.e. ≈ 49 per 64 lanes, for today's loop at 6 lanes).
ISS reference, one thread: today 45.5, scores1 7.8, scores2 6.5, PV 4.0.

| op | lanes 1 | 2 | 4 | 6 | lane (6) | mhz |
|---|---|---|---|---|---|---|
| fma16_sf (today) | | | | | | |
| scores1 | | | | | | |
| scores2 | | | | | | |
| pv4 | | | | | | |

| fetch, 3 MiB cold | GB/s lanes 1 | 2 | 4 | 6 |
|---|---|---|---|---|
| no `l2fetch` | | | | |
| `l2fetch` lead 32 KiB | | | | |

### (c) today's kernel per L (the S2 reference line)

| pos | warm `dsp_us` | cold `dsp_us` | cold scores / softmax / pv (Mpcyc) | `bad_stats` (L = pos + 1) |
|---|---|---|---|---|
| 511 | (#152: 731) | (#152: 1099) | (#152: 7.21 / 0.49 / 4.52) | |
| 1023 | (#152: 1737) | (#152: 2193) | (#152: 14.52 / 0.92 / 9.06) | |
| 1535 | | | | |

### (d) CPU side

| L | 513 | 1024 | 1536 |
|---|---|---|---|
| `AttnM1F16Det` out bad (rope_pos 0 / L−1) | | | |

### (e) full model (s170a = the #164 set)

| variant | G | prefill tok/s | decode tok/s (all) | last 64 | `pcyc/op ATTN_M1` | text = A |
|---|---|---|---|---|---|---|
| A | 64 | (#164: 565.7) | (#164: 45.33) | | — | ref |
| Q0-prof | 64 | | | | (#164 RQ G=64: 2,807,171) | |
| A | 1024 | (#164: 385.3) | (#164: 47.23) | (#164: 42.11) | — | ref |
| Q0-prof | 1024 | | | | | |

In-model / cold factor = `pcyc/op ATTN_M1` over the cold (append + pool)
pcycles at the matching L (G = 64: mean L 544.5 against pos 511 × 544.5 /
512; G = 1024: mean L 1024.5 against pos 1023).

### Read (the script prints the model; the reader fills the rest)

* **T64 / T1024** (plan §3.5 with S1's constants, × 1.15): cycles per layer
  = 32 L · s + 32 L · p + 60 L + 20 000 with s = scores2 and p = pv4 at 6
  lanes, at mean L 544.5 and 1024.5. The script prints it for scores2 and
  for scores1. Above 600 k at G = 1024 → stop after S1 and report.
* **Unit split** (§3.4): if pcyc_per_fma64 is flat past 2 lanes, the lanes
  buy only fetch overlap.
* **`l2fetch`** on if the lead row beats the plain row at 6 lanes.
* **exp16**: vector if its row is `bad=0`, else the table.

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

## Notes from the run

<thermal, first-run page faults, FARF/AEE errors, anything stale>
