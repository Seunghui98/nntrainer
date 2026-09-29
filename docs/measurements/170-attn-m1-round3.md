# Measurement 170 round 3 (S4): the split softmax, exp16 by the checked table, q by vlut16 — G6 passes at G = 64 and 1024, bit-identical

Branch `htp/170-round3` @ `0a6aea4f` (plan `docs/plans/170-attn-m1-round3.md`,
PR #189's plan; stacks on PR #182 = `htp/170-round2` @ `72c42079`). The new
device set is built from `dev/attn-shadow-170-r3` @ `e1b2eaa9` (= the
branch plus the inert shadow commit `0c1e2202`, cherry-picked unchanged:
same `git patch-id`; pushed, never merged). The reference set `q2` is S3's
`new/` (round 2, `dev/attn-shadow-170-r2` @ `0c1e2202`), copied unchanged.
Estimated device time: **≈ 75 min** (state + install 4, gtest half 25,
shadow 4, speed 13, G6 profiles 5, G5 nll + text 22, checks 2; thermal
waits extra). Run by the orchestrator:

```
bash /local/mnt/workspace/htp_moe/170/s4/run_s4.sh R3CY10WM83Y
```

(`docs/measurements/170-s4-run.sh` is the copy, byte-identical until the
post-sitting `prefill.txt` fix; it sources
`tools/htp/env.sh` between `set +u` and `set -u`). Logs go to
`/local/mnt/workspace/htp_moe/170/s4/logs/`, the summary to
`logs/sitting.out`. The script waits for zone0 ≤ 35 °C, stops on
`0x8000040e`, a device md5 mismatch (at install and at every skel swap) or
any `bad ≠ 0` in the G3 gtests before the E2E half; it counts every other
missing expected line (`expectation mismatches`) and, apart, the G6a
terms over their line (`read, not gated`).

## Why

Round 2 (S3) is bit-identical on silicon and misses G6 narrowly: in-model
`pcyc/op ATTN_M1` 250,289 / 373,642 at G = 64 / 1024 against 210 k /
350 k. Its SOFTMAX word lumped five pieces; the ISS split (plan §0) put
65 % of it in exp16, which is issue-bound. Round 3 looks exp16 up in the
spec's own checked table, builds P1's q operand with a permute instead of
2048 stored splats, and times the five softmax pieces apart. S4 reads the
split on silicon (G6b), decides the lut and the two `l2fetch` leads on
four skels from one source (the winner W becomes the E2E skel and the
fold's default), and asks whether G6 holds with G2–G5 kept. The issue
closes on G6 with G2–G5 held.

## What changed (host and ISS evidence, not device results)

| piece | round 3 | host / ISS check |
|---|---|---|
| phase words | five new words after `CALL_QT`: `EXP` 9, `ET` 10, `MAX` 11, `SUM` 12, `DIV` 13 (`ATTN_M1_PROF_WORDS` 14); `SOFTMAX` keeps its meaning | `ATTN M1 PHASES OK` with the identity `SOFTMAX ≥ EXP + ET + MAX + SUM + DIV` (a mutant that inflates `DIV` fails it); no timer read in any P1 FMA or PV chain loop (`-S`); IDL unchanged (the word count is symbolic) |
| exp16 (P2) | the spec's `attn_m1_det_exp_table` as fp16 bits (37 KiB, built at create), index = 3 integer ops (`hvx_hf_exp16_idx`), one scalar gather loop per unit (`ATTN_M1_EXP_TAB`, default 1) | PRIM row: vector index = `attn_m1_det_exp_index` and `tab[index]` = exp16's bits at every fp16 d ≤ 0 and −inf (31,746, bad = 0); an index off by one fails the row and BIT-IDENTICAL |
| q operand (P1) | one rounded row per q head stored zipped (`vshuffe`), `q[d]`, `q[d+1]` from one `vlut16` with index bytes (d, d+1) (`ATTN_M1_Q_LUT`, default 1); q scratch 256 KiB → 4 KiB | PRIM row: the pair = `vsplat(q[d])` / `vsplat(q[d+1])` on 256,000 lanes of random rows with ±0 and subnormals; P1 loop 11 / 10 packets per 4 FMAs, 2 `vlut16`, 4 vector loads (Kt only), no `vsplat`, no q load |
| `hvx_emu` | `vlut16`, `vsplat_b`, `vadd_b`, `vsub_sat` / `vmin` (uh), `vshuffe` | each diffed against `hexagon-sim -mv79` on iota / spread inputs: 78 dump rows equal (`vlut16` over all 256 index bytes × 15 Rt values) |

`ATTN M1 BIT-IDENTICAL` and `ATTN M1 PHASES OK` at L = 1 / 63 / 64 / 65 /
512 / 513 / 1024 / 1536 × pools 0 / 3 / 7 × shapes (8, 4) / (1, 2) / (2, 3)
/ (1, 8) for the default build, `-DATTN_M1_Q_LUT=0`, `-DATTN_M1_Q_LUT=0
-DATTN_M1_EXP_TAB=0` (= r2w) and `-DATTN_M1_P1_LEAD=0 -DATTN_M1_ET_LEAD=0`;
`ATTN M1 HF PRIM OK`; `ALL CHECKS PASS`, `WORKER POOL LANES OK`.
`run_inproc_e2e.sh`: `INPROC E2E PASS`, all 203 `E2E …` lines byte-equal
to round 2's (`fwd-hd64 min_snr_db=37.17`, `golden*` `bit_identical=1`,
`tokens fwd==off 8/8`). `*qs4cx*` 2 passed, `*Lfm2Moe*` 6 passed,
`tools/htp_syntax_check.sh` exit 0.

**One deviation from the plan (§3.2).** The plan's form — one `vlut16`
per FMA, lo half only, index + 1 within the four chains and + 5 across —
compiled to 15 / 13 packets per 4 FMAs (each `vlut16` alone in its
packet), over the plan's ≤ 14 gate. `vlut16` fills its hi half from the
odd index bytes (the ISS dump), so one permute with bytes (d, d + 1)
gives two splats: 11 / 10 packets, round 2's count. Both forms were
bit-identical on the host; the pair form is what ships.

**ISS reads** (`hexagon-sim -mv79 --timing`, pool NULL, one thread, the
second call at each position, the tree's own phase words; exploration
under contract §12, **not device numbers**; the ISS prices a compute-bound
piece at ≈ 0.8× the 6-lane silicon lane-sum and sees no L2 contention):

| pos 1023 | round 2 + words (= r2w) | + exp table | + vlut16 pair (= r3) |
|---|---|---|---|
| append | 16.1 k | 16.3 k | **6.1 k** |
| scores | 278.6 k | 278.5 k | **222.0 k** |
| softmax | 423.2 k | 281.2 k | 256.2 k |
| · exp | 271.0 k | **133.4 k** | 124.4 k |
| · et | 41.0 k | 36.5 k | 36.5 k |
| · max / sum | 6.0 k / 8.6 k | 5.8 k / 8.8 k | 5.7 k / 8.8 k |
| · div | 89.6 k | 89.8 k | 73.8 k |
| pv | 186.3 k | 186.3 k | 186.3 k |
| pool | 888.4 k | 746.6 k | **665.3 k** |
| output / S hash | `60787324…` / `cc6e9471…` | same | same |

Hashes are equal at pos 511 / 1023 / 1535 in all three columns. The
plan's step-2 read (exp ≤ 130 k) came in at 133 k; the planning prototype
read 115 k with a 12-packet gather against the tree's 13 (and its `div`
bracket carried the two-tile interleave, which round 3 does not take).
`div` moves between builds without any change to it (layout noise, as the
plan's §0 saw in `scores`).

## Variants (4 E2E; A first; `NNTR_NUM_THREADS=8`)

| | dir | env | expected banner (else **void**, rule 36) |
|---|---|---|---|
| **A** (reference) | `new` (+ W) | none | `dspq: on`, `dspq: close calls=N served=N bad=0`, no `graph:` |
| **Q2** (round 2) | `q2` | `NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=MOE,QK_NORM,ROPE,ATTN_M1` | `resident=QK_NORM\|ROPE\|ATTN_M1\|MOE`, `calls/token=28.00`, `cache=24576 KiB` |
| **Q3** (round 3, W) | `new` + W | same | same |
| **RQ3** | `new` + W | `… KINDS=MOE,RMSNORM,QK_NORM,ROPE,ATTN_M1` | `resident=RMSNORM\|QK_NORM\|ROPE\|ATTN_M1\|MOE`, `calls/token=77.00`, `cache=24576 KiB` |

A is the `new` dir with the switch off: the same CPU path as S3's A (no
ARM code differs from round 2's but the inert shadow commit), and G4's
logits reference, so no separate An cell.

## Skels (one source, `HEX_EXTRA_CFLAGS`)

| skel | flags | role |
|---|---|---|
| `r2w` | `-DATTN_M1_Q_LUT=0 -DATTN_M1_EXP_TAB=0` | round 2's arithmetic + the 14 words: the split (G6b) |
| `r3` | (defaults: table, lut, P1 lead, ET lead) | round 3 |
| `r3n` | `-DATTN_M1_P1_LEAD=0` | the P1 next-tile lead read |
| `r3e` | `-DATTN_M1_ET_LEAD=0` | the ET lead read |

**Winner W** (plan §3.3, computed by the script): the lowest cold `pool`
at pos 1023 among `r3`, `r3n`, `r3e`; any within 3 % of that minimum with
fewer leads wins (`r3` has 2, `r3n` / `r3e` 1); between two with the same
lead count, the lower pos-511 cold `pool`. W is the E2E skel and the
fold's default.

## Artifacts

Set `/local/mnt/workspace/htp_moe/170/s4/`, `md5.txt` there (paths
relative to it):

| file | md5 | built with |
|---|---|---|
| `new/libnntr_hvx_skel.r2w.so` | `3118fe5fa91121e94bc85416b64f53e7` | `test/htp/build.sh` on `e1b2eaa9` (DSP sources = `0a6aea4f`'s), `HEX_EXTRA_CFLAGS` as above: `UNDEFINED SYMBOLS OK (51 runtime imports)` |
| `new/libnntr_hvx_skel.r3.so` | `5530d5523dfc1029c7146bbc15bf778a` | same |
| `new/libnntr_hvx_skel.r3n.so` | `edf79474784e652c8e077f144ed00fe0` | same |
| `new/libnntr_hvx_skel.r3e.so` | `d550a9624f1eab7b37a292703d4cda7c` | same |
| `new/nntrainer_causallm` | `494682d8d3cec327763d5fed30ef71bd` | `build_android.sh --htp --cache` after `ninja install` in a fresh `builddir` |
| `new/libcausallm_core.so` | `3f20fe9a5b7253ade4bf425c78cc8dd5` | same (`NNTR_HTP_FORWARD_KINDS` 2, `NNTR_ATTN_SHADOW` 1) |
| `new/libnntrainer.so` | `53509ac57b69476ed85da0a55dea10be` | same (`jni/obj/local`; NEEDED `libsdkl.so`, `libcdsprpc.so`; `graph: forward calls` 1; `dspq: on` 1) |
| `new/libccapi-nntrainer.so` | `6ce0ef9ee32a28eaefa85973609535fe` | same |
| `new/unittest_hvx_attn` | `86d072664e2b119c8ca12ca9db388f77` | `test/jni` ndk-build (`HvxAttnM1.*`, the 14-word `ATTN_M1_PHASE` line) |
| `new/unittest_nntrainer_cpu_backend_fp16` | `90db7e1358249a0f78fd28e436ec4b5e` | same (`AttnM1F16Det.*`) |
| `new/unittest_hvx_softmax` | `4a4919e20ede42479411856364b3f501` | same (`HvxM1Ops.Rope64*`) |
| `q2/*` | as S3's `new/*` (skel `2ceeedd6…`, app `7b62a0aa…`, `unittest_hvx_attn` `5cc92331…`; S3's `md5.txt`) | copied from `/local/mnt/workspace/htp_moe/170/s3/new/` |
| `{new,q2}/libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL 6.4.0.1 |
| `{new,q2}/libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 sysroot |
| `{new,q2}/prompt512.txt`, `bitset-02 … 08` | as S3's (`prompt512` `fc65c158…`) | the #164 prompts |
| `attn_shadow_check.py` (not pushed) | `4c88617dfc12f8a8f94260a4e7566ece` | `tools/htp/attn_shadow_check.py` of `e1b2eaa9` (= S3's) |

`new/` stages no `libnntr_hvx_skel.so`: the script copies the variant
under test to that name on the device and checks its md5 before every
gtest and before the E2E half.

PR-only build of `0a6aea4f` (rung 3, not staged): skel (defaults)
`f318052e696041245c4d4ff608a7019d`, `nntrainer_causallm`
`2bdb7e0c6269707cd47ae38965573bf9`, `libcausallm_core.so`
`f7c700e70436ad31a45c2c0b416238d0`, `libnntrainer.so`
`28f85e2d9f2f0de3fe70e21cd1071f67`, `libccapi-nntrainer.so`
`0a39ff48f38fec60a5c64f408bca37dc`, `unittest_hvx_attn`
`40c31d8207a77e21c07e13bae782297d`, `unittest_hvx_softmax`
`256ea43e5657843dc3d0e8573addc211`,
`unittest_nntrainer_cpu_backend_fp16`
`3ee5e799daa86f32cb40750562f2fd14`; NEEDED `libsdkl.so` +
`libcdsprpc.so`, `NNTR_HTP_FORWARD_KINDS` 2. The binaries differ from
`new/`'s by the worktree path and the dev commit; the skel is not
byte-reproducible across worktrees.

Rebuild recipe (a fresh worktree): `git submodule update --init
--depth 1`; copy `Applications/CausalLM/lib/libtokenizers_android_c.a`;
`source tools/htp/env.sh` with `HEXKL_ROOT=$HOME/Qualcomm/hexkl-1.0-beta.2/hexkl_addon
HEXKL_SDK_VER=6.4.0.1 PATH=$HOME/.local/bin:$ANDROID_NDK:$PATH`;
`(cd Applications/CausalLM && ./build_android.sh --htp)` (fails at the
install on a fresh `builddir`), `(cd builddir && meson configure
-Dprefix=$PWD/android_build_result && ninja install)`,
`(cd Applications/CausalLM && ./build_android.sh --htp --cache)`; the four
skels with `HEX_EXTRA_CFLAGS=… ./test/htp/build.sh` and a copy each; the
`test/jni` ndk-build of `unittest_hvx_attn unittest_hvx_softmax
unittest_nntrainer_cpu_backend_fp16`; `libc++_shared.so` from the NDK
sysroot if `jni/obj/local` lacks it.

## Cells and expected lines (what `run_s4.sh` checks)

1. **Install**: `md5sum -c`, push `new/` to `…/causallm/s170r3n` and `q2/`
   to `…/s170r3q`, device md5 against `md5.txt` (stop on a mismatch), the
   model config as S3 (`do_sample` false, `bad_word_ids [124900]`,
   `moe_engine htp`).
2. **Gtest half** (`HvxAttnM1.PerLayerCost`, 6 `ATTN_M1_PHASE` lines per
   skel: pos 511 / 1023 / 1535 warm and cold):
   (a) `q2` — round 2's 9-word line, this sitting;
   (b) `r2w` — 14 words; **G6b**: cold pos 1023 `exp / softmax` in
   [0.55, 0.75], `softmax ≥ exp + et + max + sum + div`, and r2w's
   `scores + softmax + pv` within 15 % of (a)'s (else the sitting is
   flagged as drifted and continues; S3's line was 1,572 k);
   (c) `r3`, `r3n`, `r3e`; `reads.txt` lines up the lut read (`scores`
   r2w vs r3), the P1-lead read (`scores` r3 vs r3n) and the ET-lead read
   (`softmax` / `et` r3 vs r3e), then `WINNER W=…`;
   (d) on W: `HvxAttnM1.*` (**G3**: `ATTN_M1_FIELD L=<L> bad=0 bad_stats=0
   of 2048` at the 8 L, `append_chain L=65 bad=0`; its cold phase line is
   **G6a**), `AttnM1F16Det.*` `out bad=0` ×2 at 513 / 1024 / 1536, rope64
   `bad=0` ×5. `RejectsBadShapes` fails on `AEE_ERPC` (#137) and is not
   counted. **Stop before the E2E half on any `bad ≠ 0`.**
   **G6a** (read): cold pos 1023 on W: `append` ≤ 8 k, `exp` ≤ 150 k,
   `div` ≤ 120 k, `et` ≤ 55 k, `max + sum` ≤ 20 k, `softmax` ≤ 300 k,
   `scores` ≤ 600 k (best of r3 / r3n / r3e), `pv` ≤ 400 k, `busy_max` ≤
   245 k, `pool` ≤ 275 k, `dsp_us` ≤ 145; pos 511 cold `pool` ≤ 148 k.
3. **G4** (prompt 512, G = 8, forced on A's tokens): A, Q3, RQ3 with the
   shadow dumps; `attn_shadow_check.py d_f_A d_f_A d_f_Q3 d_f_RQ3`:
   ```
   ATTN SHADOW d_f_Q3 tag3_heads=1536/1536 records=48 layers=6 positions=8 zero_records=0 logits_equal_steps=8/8
   ATTN SHADOW d_f_RQ3 tag3_heads=1536/1536 records=48 layers=6 positions=8 zero_records=0 logits_equal_steps=8/8
   ```
   and every `[PPL] decode step` line of Q3 / RQ3 equal to A's.
4. **Speed**: A / Q2 / Q3 at G = 64 / 512 / 1024, run 1 in order A Q2 Q3,
   run 2 in order Q3 Q2 A; banners as above; every cell's text = A run 1
   of its G; prefill of each Q cell within −5 % of A's (mirrored mean,
   `prefill.txt`).
5. **G6**: Q2-prof and Q3-prof (`NNTR_HTP_PROFILE=2`) at G = 64 and 1024;
   Q3's `pcyc/op … ATTN_M1=` ≤ **210 000** (G = 64) and ≤ **350 000**
   (G = 1024).
6. **G5**: 8 prompts at G = 256: A self (writes `cont_p0<i>.ids`), A
   forced once (null check), Q3 / RQ3 forced on A's ids; free text A / Q2
   / Q3 / RQ3. Expected: every nll step line of Q3 and RQ3 equal to A's;
   Q3 and RQ3 text = A byte for byte; Q3 text = Q2 text.

Stop rules: `0x8000040e` (stale skel), a device md5 mismatch (install or
skel swap), `bad ≠ 0` in (d).

## Results S4 (ran 2026-09-30 04:40:48 – 05:07:18 KST on `R3CY10WM83Y`, logs `/local/mnt/workspace/htp_moe/170/s4/logs/`; filled from the logs)

Device md5 = `md5.txt` (`MD5 OK`, and every skel swap's md5 matched).
`expectation mismatches: 1`: G6b's ratio (below). `G6a terms over their
line: 8` (read, not gated). No `0x8000040e`. **W = r3e.** zone0 27.8 °C
at start, 30.8 °C after the gtest half, 52.5 °C after the shadow, 59–69
°C through the speed cells, 64–65 °C through G6 and G5; battery 100 → 91
%. `mhz` 2091–2108 in every phase line. `HvxAttnM1.RejectsBadShapes`
failed on `AEE_ERPC` (0x80000600) as expected (#137, not counted). The
whole sitting took 26.5 min, not 75: the gtest half ran in ≈ 10 s.

`HvxAttnM1.PerLayerCost`, cold pos 1023, pcycles lane-summed over 6 lanes
(q2 = round 2's skel and gtest in this sitting, 9 words):

| skel | append | scores | softmax | exp | et | max | sum | div | pv | busy_max | pool | dsp_us | pos 511 pool |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q2 (round 2) | 35.2 k | 670.1 k | 513.9 k | – | – | – | – | – | 363.4 k | 268.8 k | 307.0 k | 163.1 | 164.5 k |
| r2w | 34.1 k | 680.7 k | 536.0 k | **272.0 k** | 73.4 k | 15.8 k | 10.3 k | 139.7 k | 383.4 k | 279.2 k | 316.1 k | 167.4 | 171.9 k |
| r3 | 24.5 k | **391.6 k** | 643.5 k | **389.4 k** | 73.9 k | 15.8 k | 9.6 k | 132.6 k | 367.0 k | 243.9 k | 283.5 k | 146.5 | 148.7 k |
| r3n | 29.2 k | 648.7 k | 616.6 k | 371.3 k | 73.3 k | 14.5 k | 10.1 k | 129.8 k | 353.4 k | 282.4 k | 318.4 k | 166.6 | 166.9 k |
| **r3e = W** | 28.5 k | 386.9 k | 637.3 k | 386.7 k | 73.2 k | 14.9 k | 9.2 k | 128.2 k | 368.2 k | 242.2 k | **280.9 k** | 148.1 | 152.1 k |

W's own G3 run (`per_l.txt`, the G6a line) read `append` 26.7 k, `scores`
385 k, `softmax` 614 k (`exp` 373 k, `et` 71 k, `max` 13 k, `sum` 9 k,
`div` 123 k), `pv` 384 k, `busy_max` 242 k, `pool` 276 k, `dsp_us` 144.9;
pos 511 `pool` 148.8 k.

| gate | line | value | pass |
|---|---|---|---|
| G6b | r2w `exp / softmax`; pieces ≤ softmax; drift vs q2 | **0.507** (window 0.55–0.75); 511.2 k ≤ 536.0 k; 1,600 k vs q2 1,547 k (+3.4 %) | ratio **no** (see Read), identity yes, no drift |
| G3 | W: `ATTN_M1_FIELD` `bad` / `bad_stats` at the 8 L, `append_chain`, `AttnM1F16Det` 513 / 1024 / 1536, rope64 ×5 | all 0 (22 lines) | **yes** |
| G4 | shadow (G = 8, A reference) | Q3 and RQ3 `tag3_heads=1536/1536 records=48 layers=6 positions=8 zero_records=0 logits_equal_steps=8/8`; nll = A for both | **yes** |
| G5 | nll = A (Q3, RQ3 × p01–p08), A forced = A self; text = A (Q3, RQ3 × 8), Q3 = Q2 (× 8), G = 256 | all equal (16 + 1 nll, 24 text) | **yes** |
| G6 | `pcyc/op ATTN_M1`, Q3 (Q2 same sitting) | G = 64: **193,838** (Q2 254,869, −23.9 %); G = 1024: **313,747** (Q2 379,551, −17.3 %) | **yes**: 0.92× / 0.90× the gates 210 k / 350 k |

G6a (read, W): within its line `pv` 384 k ≤ 400 k, `busy_max` 242 k ≤ 245
k, `dsp_us` 144.9 ≤ 145, `scores` (best of three) 387 k ≤ 600 k; over it
`append` 26.7 k (8 k), `exp` 373 k (150 k), `div` 123 k (120 k), `et` 71 k
(55 k), `softmax` 614 k (300 k), `max + sum` 22.7 k (20 k), `pool` 276 k
(275 k), pos 511 `pool` 148.8 k (148 k).

| variant | G | run 1 prefill / decode / last 64 | run 2 prefill / decode / last 64 | decode mean | text = A |
|---|---|---|---|---|---|
| A | 64 | 560.8 / 53.96 / 53.96 | 503.4 / 53.92 / 53.92 | **53.94** | ref |
| Q2 | 64 | 562.6 / 44.11 / 44.11 | 532.8 / 43.99 / 43.99 | 44.05 | same |
| Q3 | 64 | 544.7 / 44.38 / 44.38 | 555.3 / 44.60 / 44.60 | **44.49** | same |
| A | 512 | 538.9 / 54.57 / 50.16 | 425.2 / 51.70 / 50.24 | **53.13** | ref |
| Q2 | 512 | 541.8 / 45.71 / 44.85 | 422.4 / 45.31 / 44.85 | 45.51 | same |
| Q3 | 512 | 528.9 / 46.36 / 46.04 | 423.5 / 45.71 / 45.75 | **46.04** | same |
| A | 1024 | 416.6 / 50.60 / 48.56 | 369.7 / 50.06 / 48.30 | **50.33** | ref |
| Q2 | 1024 | 426.0 / 44.73 / 43.99 | 425.6 / 43.26 / 39.65 | 43.99 | same |
| Q3 | 1024 | 385.5 / 44.82 / 44.17 | 424.5 / 44.16 / 44.20 | **44.49** | same |

All Q cells `calls/token=28.00` and `cache=24576 KiB`; every cell's text
equals A run 1 of its G. Q3 vs Q2 decode: +1.0 / +1.2 / +1.1 %; Q3 vs A:
−17.5 / −13.3 / −11.6 %. The profile's `dsp` per graph call: Q2 419.1 →
Q3 372.4 µs (G = 64), 421.9 → 419.9 µs (G = 1024). Prefill means against
A's (computed by hand: the script's verdict column printed empty -- a
bare `>` in awk's `printf` arguments is an output redirection; fixed in
`170-s4-run.sh` after the sitting): Q2 +2.9 / +0.0 / +8.3 %, Q3 +3.4 / −1.2 /
+3.0 % (G = 64 / 512 / 1024): within the −5 % band.

## Read S4

* **Both speed gates pass**, bit identity held everywhere (G3, G4, G5):
  in-model `ATTN_M1` 193.8 k / 313.7 k against 210 k / 350 k, from round
  2's 254.9 k / 379.6 k in the same sitting. Decode moved +1 % (6 layers ×
  0.03–0.06 ms out of ≈ 22 ms per token): the resident path is still 28
  DSP calls per token, and Q3 stays 12–18 % under A. **The default stays
  off**; removing the round trips is the end-to-end track (#132), not this
  kernel.
* **What landed: the vlut16 q operand.** Cold `scores` 681 k (r2w) →
  392 k (r3), warm 601 k → 314 k; `append` 34 k → 24–28 k. The P1 lead is
  worth 649 k → 387 k with the lut (r3n vs r3e); the ET lead is neutral
  (r3e within 1 % of r3), so W = r3e and the fold removes it.
* **What did not: the exp table.** On silicon the gather reads **387 k**
  cold (r3), the exp16 compute it replaced **272 k** (r2w): the table is a
  115 k lane-summed loss (≈ 19 k pcycles wall, 9 µs per call). The ISS
  priced the compute right (271 k, 1.0×) and the gather 3× low (124 k):
  six threads issuing scalar loads after HVX stores is what it cannot
  model. `pool` still fell 307 k → 276–281 k because scores fell more.
  **Round 4's first term is `exp`**: a gtest-only cell of W with
  `-DATTN_M1_EXP_TAB=0` (kept for that; host bit-identical) would, if
  exp16 costs there what it cost in r2w, read `softmax` ≈ 510 k and `pool`
  ≈ 257 k (an estimate: 115 k lane-summed / 6 lanes off W's busy lane);
  then `vgather` from a VTCM
  carve-out (plan §3.5).
* **G6b's ratio missed (0.507 vs 0.55–0.75)** because silicon's other
  softmax pieces are larger than the ISS's, not because exp is smaller:
  `exp` 272 k = the ISS's 271 k, but `div` 140 k (ISS 90 k), `et` 73 k
  (ISS 41 k), `max + sum` 26 k (ISS 15 k). The split itself holds (pieces
  511 k ≤ softmax 536 k); the sitting did not drift (+3.4 % vs q2).
* **Round-4 order by size (W, cold pos 1023, lane-summed):** `exp` 373 k,
  `scores` 385 k, `pv` 384 k (at the DDR line, plan §3.5's DMA lever),
  `div` 123 k, `et` 71 k.

## Text approval (per-token-entry handoff)

The p01 text at G = 256 of A, Q2, Q3 and RQ3 is byte-identical (md5 of
the stripped text `e377add566c09b7ef2a0698ed106b8be` for all four, the
same as S2's and S3's); so are p02–p08 (G5). The decode PPL forced on A's
continuation is equal to 17 digits.

| variant | decode PPL (forced on A, p01, G = 256) | generated text (p01, G = 256) | text approved (user: y/n) |
|---|---|---|---|
| A | 1.42156 (nll_sum 90.048889370024341, source=self) | md5 above | (reference) |
| Q2 | (text only) | identical to A (same md5) | |
| Q3 | 1.42156 (nll_sum 90.048889370024341) | identical to A (same md5) | |
| RQ3 | 1.42156 (nll_sum 90.048889370024341) | identical to A (same md5) | |

## Fold (plan step 7, commit `653a619d`)

`ATTN_M1_Q_LUT` and round 2's splat vectors removed (the lut won); the
ET lead removed (W = r3e); the P1 and PV leads kept. The folded default
compiles to r3e's instructions exactly (`-S` diff empty outside debug
info), so S4's skel is the PR's kernel. `ATTN_M1_EXP_TAB` stays, default
1 (the configuration G3–G6 were read on), with the S4 reading beside it;
`-DATTN_M1_EXP_TAB=0` passes the host check. After the fold: `ALL CHECKS
PASS`, `ATTN M1 BIT-IDENTICAL`, `INPROC E2E PASS` with the 203 lines
byte-equal, both skels `UNDEFINED SYMBOLS OK (51 runtime imports)`.

## Notes (S4 build)

* **The gather.** Index vectors of the unit's heads are stored once, one
  scalar loop (8-way unrolled, 13 packets per 8 lookups: two memory slots,
  24 memory ops) gathers into a stack row (2 × 1 KiB of the 32 KiB worker
  stack at MAX_GQA 8), four reloads. If silicon reads `exp` > 150 k with
  six threads gathering, plan §5's fallbacks apply (the per-head form, then
  `vgather` from VTCM in round 4).
* **Heap.** −252 KiB (`qs`) + 37 KiB (`exp_tab`): ≈ 426 KiB of scratch per
  cache at LFM2.5 / 2048, 215 KiB less than round 2. No VTCM, no IDL
  change, `cache=` unchanged. The table build is a one-time ≈ 18.5 k
  scalar exp16 at create.
* **No IDL change**: the skel entry and the IDL name the word count
  symbolically; a round-2 gtest against a round-3 skel (or the reverse)
  gets `AEE_EINVALIDFORMAT` on the phase-word call (`kM1StaleProf`), not a
  hang. Each set's gtest and skel come from one tree.
* The ISS table above is exploration (contract §12), never a device
  number.

## Notes from the run (S4)

`adb -s R3CY10WM83Y` throughout. Thermal and `mhz` as in Results; no
FARF / AEE error besides rule 38's `AEE_ERPC` in `RejectsBadShapes`. The
single expectation mismatch is G6b's ratio. `run_s4.sh` defect for the
next copy: the `prefill.txt` verdict column is empty (mawk parsed the
unparenthesised `d >= -5 ? …` in `printf` as a redirection; the copy in
the repo is fixed, `run_s4.sh` in the set is the one that ran); the
numbers above are from `speed.txt`.
