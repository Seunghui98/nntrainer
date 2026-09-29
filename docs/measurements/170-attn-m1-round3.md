# Measurement 170 round 3 (S4): the split softmax, exp16 by the checked table, q by vlut16 — speed gate G6 on silicon

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

(a byte-identical copy is `docs/measurements/170-s4-run.sh`; it sources
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

## Results S4 (fill in)

Device md5 = `md5.txt`: | expectation mismatches: | G6a over: | zone0 at start: | W =

| skel | cold pos 1023: append / scores / softmax / exp / et / max / sum / div / pv / busy_max / pool / dsp_us / mhz | cold pos 511 pool |
|---|---|---|
| q2 (round 2, 9 words) | | |
| r2w | | |
| r3 | | |
| r3n | | |
| r3e | | |

| gate | line | value | pass |
|---|---|---|---|
| G6b | r2w exp / softmax; pieces ≤ softmax; drift vs q2 | | |
| G3 | W: `ATTN_M1_FIELD` 8 L, `append_chain`, F16Det, rope64 | | |
| G4 | shadow Q3 / RQ3, nll = A | | |
| G5 | nll = A (Q3, RQ3 × 8), A forced = A self; text = A (Q3, RQ3 × 8), Q3 = Q2 (× 8) | | |
| G6 | Q3 `pcyc/op ATTN_M1` (Q2 same sitting) | G 64: / G 1024: | |

| variant | G | run 1 prefill / decode / last 64 | run 2 prefill / decode / last 64 | decode mean | text = A |
|---|---|---|---|---|---|
| A | 64 | | | | ref |
| Q2 | 64 | | | | |
| Q3 | 64 | | | | |
| A | 512 | | | | ref |
| Q2 | 512 | | | | |
| Q3 | 512 | | | | |
| A | 1024 | | | | ref |
| Q2 | 1024 | | | | |
| Q3 | 1024 | | | | |

Reference (S3, same unit): A 55.13 / 50.82 / 51.40, Q2 44.49 / 46.26 /
45.24 decode tok/s at G = 64 / 512 / 1024; Q2 `pcyc/op ATTN_M1` 250,289 /
373,642. Goal ≥ 50 decode.

## Text approval (per-token-entry handoff)

| variant | decode PPL (forced on A, p01, G = 256) | generated text (p01, G = 256) | text approved (user: y/n) |
|---|---|---|---|
| A | | | (reference) |
| Q2 | (text only) | | |
| Q3 | | | |
| RQ3 | | | |

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

<thermal, `mhz` per phase line, FARF / AEE errors, anything stale>
