# Measurement 170 round 2 (S3): the round-2 ATTN_M1 on silicon — bit-identical, in-model cost −39 / −44 %, speed gate missed by 19 / 7 %

Branch `htp/170-round2` @ `e079fec3` (plan `docs/plans/170-attn-m1-round2.md`,
PR #179's plan); the new device set is built from `dev/attn-shadow-170-r2`
@ `0c1e2202` (= the branch plus the inert shadow commit of
`dev/attn-shadow-170`, cherry-picked unchanged; pushed, never merged). The
reference set is `htp_moe` @ `90d88e2b` (PR #176 merged: round 1).
Estimated device time: **≈ 60 min** (state + install 4, probe cells 4,
G3 + G6a gtests 10, shadow 4, speed 13, G6 profiles 5, G5 nll + text 22,
checks 2; thermal waits extra). Run by the orchestrator:

```
bash /local/mnt/workspace/htp_moe/170/s3/run_s3.sh R3CY10WM83Y
```

(a byte-identical copy is `docs/measurements/170-s3-run.sh`). Logs go to
`/local/mnt/workspace/htp_moe/170/s3/logs/`, the summary to
`logs/sitting.out`. The script waits for zone0 ≤ 35 °C, stops on
`0x8000040e` or a device md5 mismatch, counts every other missing
expected line (`expectation mismatches`) and, apart, the G6a terms over
their line (`read, not gated`).

## Why

Round 1 (PR #176) is bit-identical on silicon and misses G6 by ≈ 2×
(`pcyc/op ATTN_M1` 394,501 / 700,449 at G = 64 / 1024 against 210 k /
350 k). S2's phase words put the time in four terms; round 2 moves bytes
and reorders independent work only (every fp16 operation and its order in
`attn_m1_det.h` unchanged). S3 reads whether each term landed (the probe
cells and the per-term phase line, round 1 measured beside it) and whether
G6 now holds with G3–G5 kept. The issue closes on G6 with G2–G5 held.

## What changed (host evidence, not device results)

| term (S2, pos 1023 cold) | round 2 | host / ISS check |
|---|---|---|
| PV 385 k wall: one DDR miss per V row inside the chain | `l2fetch` of the first two 16 KiB V blocks before each chain, then the block two ahead every 128 positions (`ATTN_M1_PV_LEAD`); lanes take ranges of q-head chains in pairs, groups of 4 / 2 (busiest lane 6 chains instead of 8) | chain loops (`-S`): ng 4 = 11 packets / 4 FMAs, ng 2 = 6, as round 1, no spill, no `l2fetch` and no sf op inside |
| append 96 k: scalar q / k / v rounding, 512 scattered stores | `hvx_hf_round_row` (integer ops + the exp16 magic add, sign of zero kept), q as one splat vector per value, the k column merged by P1's last-tile unit | `ATTN M1 HF PRIM`: vector rounding = rne16 at 1.22e9 f32, widening = the fp16 value at 63,488 fp16, bad = 0 |
| softmax: 256 scalar halfword stores per P2 unit into cold ET lines | 4-head `vshuff` transpose + one masked store per position; ET `l2fetch`ed by the caller (`ATTN_M1_ET_LEAD`) | `hvx_emu` moves diffed against `hexagon-sim -mv79`: identical |
| scores: tile fetched on demand | next tile `l2fetch`ed per lane (`ATTN_M1_P1_LEAD`); q from splat vectors | P1 loop 11 packets / 4 FMAs, vector q loads, no stack reload (round 1 11–12 + a reload) |

`ATTN M1 BIT-IDENTICAL` at L = 1 / 63 / 64 / 65 / 512 / 513 / 1024 / 1536 ×
pools 0 / 3 / 7 × shapes (8, 4) / (1, 2) / (2, 3) / (1, 8); `ATTN M1 PHASES
OK`; the kernel on `hexagon-sim -mv79` (pool NULL) byte-equal to the spec at
L = 1 / 65 / 513 / 1024 (exploration, contract §12: not a gate).
`run_inproc_e2e.sh`: `INPROC E2E PASS`, and all 203 `E2E …` lines
byte-equal to the same script on `90d88e2b` (`fwd-hd64 min_snr_db=37.17`,
`golden*` `bit_identical=1`, `tokens fwd==off 8/8`). Rung 1 otherwise:
`*qs4cx*` 2 passed, `*Lfm2Moe*` 6 passed, `ALL CHECKS PASS`, `WORKER POOL
LANES OK`, `tools/htp_syntax_check.sh` exit 0.

## Variants (4; A first; `NNTR_NUM_THREADS=8`)

| | dir | env | expected banner (else **void**, rule 36) |
|---|---|---|---|
| **A** (reference) | `q1` (90d88e2b) | none | `dspq: on`, `dspq: close calls=N served=N bad=0`, no `graph:` |
| **Q1** (round 1) | `q1` | `NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=MOE,QK_NORM,ROPE,ATTN_M1` | `resident=QK_NORM\|ROPE\|ATTN_M1\|MOE`, `calls/token=28.00`, `cache=24576 KiB` |
| **Q2** (round 2) | `new` | same | same |
| **RQ2** | `new` | `… KINDS=MOE,RMSNORM,QK_NORM,ROPE,ATTN_M1` | `resident=RMSNORM\|QK_NORM\|ROPE\|ATTN_M1\|MOE`, `calls/token=77.00`, `cache=24576 KiB` |

G4 needs the shadow commit on both sides of its logits compare, so its
reference cell **An** is the `new` dir with the switch off: the same CPU
path as A (round 2 changes no ARM code; the dev commit takes one `getenv`
per process without `NNTR_ATTN_SHADOW` / `NNTR_LOGIT_SHADOW`). An is a dump
cell, not a speed variant.

## Artifacts

Set `/local/mnt/workspace/htp_moe/170/s3/`, `md5.txt` there (paths
relative to it):

| file | md5 | built with |
|---|---|---|
| `new/libnntr_hvx_skel.so` | `2ceeedd62dbe4f1bb2dccdadb77bb1aa` | `test/htp/build.sh` (v79, HexKL 6.4.0.1) on `0c1e2202` (DSP sources = `e079fec3`'s): `UNDEFINED SYMBOLS OK (51 runtime imports)` |
| `new/nntrainer_causallm` | `7b62a0aac53144ec3f3f09bb75eb2454` | `build_android.sh --htp --cache` after `ninja install` in a fresh `builddir` |
| `new/libcausallm_core.so` | `bd94dd16092c9e74ee52be10f3145515` | same (`NNTR_HTP_FORWARD_KINDS` 2, `NNTR_ATTN_SHADOW` 1) |
| `new/libnntrainer.so` | `ff0bb850160ed8a59c9b7abecd1b3ea2` | same (`jni/obj/local`; NEEDED `libsdkl.so`, `libcdsprpc.so`; `graph: forward calls` 1; `dspq: on` 1) |
| `new/libccapi-nntrainer.so` | `2dfea5234738528a749d62e88646b2e0` | same |
| `new/unittest_hvx_attn` | `5cc923315b378bebdb0642a65dae345c` | `test/jni` ndk-build (`HvxAttnM1.*`, `HvxAttnM1Probe.Cost` with the five new rows) |
| `new/unittest_nntrainer_cpu_backend_fp16` | `a2f7ad2335ac3facd32db3bad945f793` | same (`AttnM1F16Det.*`) |
| `new/unittest_hvx_softmax` | `6a9eb149627c2bbc6ca749618d88b8b7` | same (`HvxM1Ops.Rope64*`) |
| `q1/libnntr_hvx_skel.so` | `4f78badb45ccc2e40d17899d9cf614f6` | `test/htp/build.sh` on `90d88e2b`: `UNDEFINED SYMBOLS OK (51 runtime imports)` (S2's round-1 skel was `f673f1a3…`; not byte-reproducible) |
| `q1/nntrainer_causallm` | `bad89163ce46c4ab5cd38f7a14c91e69` | as `new` on `90d88e2b` |
| `q1/libcausallm_core.so` | `0ccdbc26d4e5cffdd93c6bb99e13aba0` | same (`NNTR_HTP_FORWARD_KINDS` 2) |
| `q1/libnntrainer.so` | `ad0ecbcb82119e031a17770d7dc02e3d` | same (NEEDED `libsdkl.so`, `libcdsprpc.so`; `graph: forward calls` 1; `dspq: on` 1) |
| `q1/libccapi-nntrainer.so` | `5232abe8ca0912ddd6ffc51e3f571faf` | same |
| `q1/unittest_hvx_attn` | `ea0ee789da22b914eaef4ff06aa4c86c` | `test/jni` ndk-build on `90d88e2b` (round 1's `HvxAttnM1.PerLayerCost`) |
| `{new,q1}/libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL 6.4.0.1 |
| `{new,q1}/libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 sysroot |
| `{new,q1}/prompt512.txt`, `bitset-02 … 08` | as S2's (`prompt512` `fc65c158…`) | the #164 prompts |
| `attn_shadow_check.py` (not pushed) | `4c88617dfc12f8a8f94260a4e7566ece` | `tools/htp/attn_shadow_check.py` of `0c1e2202` (= S2's) |

PR-only build of `e079fec3` (rung 3, not staged): skel
`b89a78cd2a407e7df5cf19f3d2a7f364`, `nntrainer_causallm`
`f2823a4ece019b6249c0b77075aebad4`, `libcausallm_core.so`
`6cff9fd1ba8338b017e462f363a2ca3f`, `libnntrainer.so`
`933c69303f135bdafb6d9ad540babe88`, `libccapi-nntrainer.so`
`fe1f54feae1bd242a2846f709a8a1a7e`, `unittest_hvx_attn`
`84a8c4640ad5ed92e08639c30edc8951`, `unittest_hvx_softmax`
`156ebb7edfe8d892acaa7993ebbc66db`,
`unittest_nntrainer_cpu_backend_fp16`
`037e4f072e42efc354a3f3760d469165`; NEEDED `libsdkl.so` +
`libcdsprpc.so`, `NNTR_HTP_FORWARD_KINDS` 2. The binaries differ from
`new/`'s by the worktree path and the dev commit.

Rebuild recipe (per set; a fresh worktree): `git submodule update --init
--depth 1`; copy `Applications/CausalLM/lib/libtokenizers_android_c.a`;
`source tools/htp/env.sh` with `HEXKL_ROOT=$HOME/Qualcomm/hexkl-1.0-beta.2/hexkl_addon
HEXKL_SDK_VER=6.4.0.1 PATH=$HOME/.local/bin:$ANDROID_NDK:$PATH`;
`(cd Applications/CausalLM && ./build_android.sh --htp)` (fails at the
install on a fresh `builddir`), `(cd builddir && meson configure
-Dprefix=$PWD/android_build_result && ninja install)`,
`(cd Applications/CausalLM && ./build_android.sh --htp --cache)`,
`./test/htp/build.sh`, the `test/jni` ndk-build of `unittest_hvx_attn
unittest_hvx_softmax unittest_nntrainer_cpu_backend_fp16`. The skel is not
byte-reproducible.

## Cells and expected lines (what `run_s3.sh` checks)

1. **Install**: `md5sum -c`, push `new/` to `…/causallm/s170r2n` and `q1/`
   to `…/s170r2q`, device md5 against `md5.txt` (stop on a mismatch), the
   model config as S2 (`do_sample` false, `bad_word_ids [124900]`,
   `moe_engine htp`).
2. **Probe cells** (`new`, `HvxAttnM1Probe.Cost`): 4 rows (lanes 1 / 2 /
   4 / 6) each of `pv4`, `pv4_cold`, `pv4_cold_l2f`, `scores1`,
   `scores1_cold`, `scores1_cold_l2f`, `scores1_splat`; `probe_read.txt`
   lines them up per lane count. A lead whose `_l2f` cell reads slower
   than its lead-0 cold cell is turned off in the fold (`-DATTN_M1_<X>_LEAD=0`,
   a second, gtest-only sitting).
3. **G3** (`new`, `HvxAttnM1.*`): `ATTN_M1_FIELD L=<L> bad=0 bad_stats=0 of
   2048` at the 8 L, `append_chain L=65 bad=0`, 6 `ATTN_M1_PHASE` lines;
   `AttnM1F16Det.*` `out bad=0` ×2 at 513 / 1024 / 1536; rope64 `bad=0` ×5.
   `RejectsBadShapes` fails on `AEE_ERPC` (#137) and is not counted.
   **G6a** (read): the cold pos 1023 phase line against `append` ≤ 10 k,
   `scores` ≤ 550 k, `softmax` ≤ 160 k, `pv` ≤ 400 k, `busy_max` ≤ 200 k,
   `pool` ≤ 220 k, `dsp_us` ≤ 110, beside round 1's line from `q1`'s
   `HvxAttnM1.PerLayerCost` in the same sitting.
4. **G4** (prompt 512, G = 8, forced on An's tokens): An, Q2, RQ2 with the
   shadow dumps; `attn_shadow_check.py d_f_An d_f_An d_f_Q2 d_f_RQ2`:
   ```
   ATTN SHADOW d_f_Q2 tag3_heads=1536/1536 records=48 layers=6 positions=8 zero_records=0 logits_equal_steps=8/8
   ATTN SHADOW d_f_RQ2 tag3_heads=1536/1536 records=48 layers=6 positions=8 zero_records=0 logits_equal_steps=8/8
   ```
   and every `[PPL] decode step` line of Q2 / RQ2 equal to An's.
5. **Speed**: A / Q1 / Q2 at G = 64 / 512 / 1024, run 1 in order A Q1 Q2,
   run 2 in order Q2 Q1 A; banners as above; every cell's text = A run 1
   of its G.
6. **G6**: Q1-prof and Q2-prof (`NNTR_HTP_PROFILE=2`) at G = 64 and 1024;
   Q2's `pcyc/op … ATTN_M1=` ≤ **210 000** (G = 64) and ≤ **350 000**
   (G = 1024).
7. **G5**: 8 prompts at G = 256: A self (writes `cont_p0<i>.ids`), A
   forced once (null check), Q2 / RQ2 forced on A's ids; free text A / Q1
   / Q2 / RQ2. Expected: every nll step line of Q2 and RQ2 equal to A's;
   Q2 and RQ2 text = A byte for byte; Q2 text = Q1 text.

Stop rules: `0x8000040e` (stale skel) or a device md5 mismatch.
Standing: every E2E cell's prefill within −5 % of A's (mirrored mean).

## Results S3 (ran 2026-09-29 23:56 – 2026-09-30 00:26 KST on `R3CY10WM83Y`, logs `/local/mnt/workspace/htp_moe/170/s3/logs/`; filled from the logs)

Device md5 = `md5.txt` (`MD5 OK`). `expectation mismatches: 2`: the two
G6 gates, nothing else. `G6a terms over their line: 6` (read, not gated).
No `0x8000040e`. zone0 32.8 °C at start, 51.7 °C after the shadow cells,
58–68 °C through the speed and G5 cells; battery 100 → 90 %.
`HvxAttnM1.RejectsBadShapes` failed on `AEE_ERPC` (0x80000600) as
expected (#137, not counted).

| gate | line | value | pass |
|---|---|---|---|
| G3 | `ATTN_M1_FIELD` `bad` / `bad_stats` at 1 / 63 / 64 / 65 / 512 / 513 / 1024 / 1536; `append_chain`; `AttnM1F16Det` 513 / 1024 / 1536; rope64 ×5 | all 0 | **yes** |
| G4 | shadow (G = 8, An reference) | Q2 `tag3_heads=1536/1536` `logits_equal_steps=8/8`; RQ2 the same (48 records, 6 layers, 8 positions, 0 zero records); nll = An for both | **yes** |
| G5 | nll = A (Q2, RQ2 × p01–p08), A forced = A self; text = A (Q2, RQ2 × 8), Q2 = Q1 (× 8), G = 256 | all equal (16 + 1 nll, 24 text) | **yes** |
| G6 | `pcyc/op ATTN_M1`, Q2 (Q1 same sitting) | G = 64: **250,289** (Q1 411,215, −39 %); G = 1024: **373,642** (Q1 668,528, −44 %) | **no**: 1.19× / 1.07× the gates 210 k / 350 k |

`HvxAttnM1.PerLayerCost`, pos 1023, pcycles lane-summed over 6 lanes
(G6a line; round 1 = the `q1` set in the same sitting):

| term | G6a line | round 2 warm | round 2 cold | round 1 warm | round 1 cold | cold, round 2 / round 1 |
|---|---|---|---|---|---|---|
| append | ≤ 10 k | 33.0 k | 32.9 k | 92.9 k | 95.2 k | 0.35 |
| scores | ≤ 550 k | 636 k | 668 k | 581 k | 698 k | 0.96 (warm 1.10) |
| softmax | ≤ 160 k | 526 k | 518 k | 511 k | 528 k | 0.98 |
| pv | ≤ 400 k | 362 k | **386 k** | 1,098 k | 1,545 k | **0.25** |
| busy_max | ≤ 200 k | 272 k | 278 k | 460 k | 595 k | 0.47 |
| pool | ≤ 220 k | 307 k | 312 k | 493 k | 625 k | 0.50 |
| dsp_us | ≤ 110 | 162 | 165 | 278 | 342 | 0.48 |

Other positions (round 2, cold `dsp_us`): pos 511 90 (round 1 191),
pos 1535 249 (round 1 487). The in-model line sits at 1.20× the cold
gtest pool at G = 1024 (374 k / 312 k) and 1.5× at G = 64 (250 k against
165 k at pos 511, L 512 vs a mean of 544.5).

Probe cells (`probe_read.txt`; wall pcycles per 64-lane FMA, all lanes
together; the lane-summed figure after the slash):

| cell | lanes 1 | 2 | 4 | 6 |
|---|---|---|---|---|
| pv4 warm | 6.28 | 3.35 | 1.68 | 1.26 / 6.9 |
| pv4 cold, no lead | 37.8 | 19.9 | 11.0 | 7.55 / 40.9 |
| pv4 cold, 32 KiB lead | **7.14** | 4.95 | 3.21 | **3.12** / 11.8 |
| scores1 warm (splat vectors) | 11.2 | 6.34 | 3.28 | 2.49 / 12.4 |
| scores1 cold, no lead | 57.0 | 31.7 | 16.8 | 11.8 / 60.2 |
| scores1 cold, next-tile lead | **12.4** | 10.9 | 9.42 | **9.96** / 51.2 |
| scores1, scalar-load splats | 9.11 | 5.16 | 2.80 | 4.19 / 22.5 |

| variant | G | run 1 prefill / decode / last 64 | run 2 prefill / decode / last 64 | decode mean | text = A |
|---|---|---|---|---|---|
| A | 64 | 532.8 / 55.85 / 55.85 | 525.1 / 54.42 / 54.42 | **55.13** | ref |
| Q1 | 64 | 533.3 / 45.20 / 45.20 | 533.9 / 44.23 / 44.23 | 44.71 | same |
| Q2 | 64 | 537.3 / 44.88 / 44.88 | 468.0 / 44.11 / 44.11 | **44.49** | same |
| A | 512 | 507.4 / 48.74 / 52.46 | 462.1 / 52.89 / 49.73 | **50.82** | ref |
| Q1 | 512 | 458.8 / 44.55 / 44.57 | 460.0 / 45.65 / 44.98 | 45.10 | same |
| Q2 | 512 | 463.3 / 46.31 / 46.44 | 463.8 / 46.22 / 46.14 | **46.26** | same |
| A | 1024 | 461.3 / 52.05 / 49.81 | 420.4 / 50.76 / 49.19 | **51.40** | ref |
| Q1 | 1024 | 426.3 / 44.24 / 42.67 | 425.2 / 43.90 / 40.92 | 44.07 | same |
| Q2 | 1024 | 427.0 / 45.37 / 44.29 | 428.8 / 45.11 / 44.72 | **45.24** | same |

All Q cells `calls/token=28.00` and `cache=24576 KiB`; every cell's text
equals A run 1 of its G. Q2 vs Q1 decode: −0.5 / +2.6 / +2.6 %; Q2 vs A:
−19.3 / −9.0 / −12.0 %. The profile's `dsp` per graph call: 420.7 → 371.0
µs (G = 64), 428.6 → 390.1 µs (G = 1024). Prefill means against A's: Q1
+0.9 / −5.2 / −3.4 %, Q2 −5.0 / −4.4 / −2.9 % (G = 64 / 512 / 1024).
Prefill runs no ATTN_M1 code, and Q1 is A's own binary, so the −5 %
cells are run-to-run spread (Q2's run 2 at G = 64 read 468 against 537 in
run 1).

## Read S3

* **Bit identity holds on silicon everywhere** (G3, G4, G5), with the
  masked ET stores from six threads, the vector rounding and the vector
  output: kernel = spec at every L, the model's attention = the CPU's for
  every head, layer and step, nll and text = A on all 8 prompts.
* **G6 missed narrowly**: 250 k / 374 k against 210 k / 350 k (1.19× /
  1.07×), from 411 k / 669 k in the same sitting. The in-model attention
  fell 39–44 %; decode moved +2.6 % at G 512 / 1024 (6 layers × 0.08–0.14 ms
  out of ≈ 22 ms per token). The switch stays off: Q2 is still 9–19 %
  below A.
* **What landed.**
  * **PV** 1,545 k → 386 k cold (0.25×), inside its line. The probe
    shows why: at 6 lanes the 32 KiB window takes the cold pv4 cell from
    7.55 to 3.12 pcycles per FMA (1 lane: 37.8 → 7.14, close to warm
    6.28).
  * **Lane balance**: busy_max 595 k → 278 k; the chain ranges and the
    smaller PV term together.
  * **Append** 95 k → 33 k, still 3.3× its 10 k line.
* **What did not.**
  * **softmax** 528 k → 518 k: unchanged, and now the largest term with
    scores. The word lumps P2's exp16 + ET stores, the caller's max and
    sum, and P3's divides, so S3 cannot say which part costs. The masked
    stores and the ET lead removed nothing visible, so S2's estimate
    that the scalar scatter cost ≈ 2.8 k per P2 unit (an inference, never
    timed) was probably wrong.
  * **scores** 698 k → 668 k cold, and **worse warm** (581 k → 636 k,
    +9.5 %). At 6 lanes the next-tile lead buys little on its cold cell
    (11.8 → 9.96, against 57.0 → 12.4 at one lane: the 6-lane tile fetch
    is at the bus, not at latency). Warm, where the tile is in L2 already,
    the lead, the k column merge (64 vmux + stores per last tile) and
    the splat-vector loads (32 KiB of q per unit read from L2, 4× the
    tile) are pure cost. The splat cell does not settle the source:
    vectors win at 6 lanes (2.49 against 4.19) and lose at one (11.2
    against 9.11).
* **Round 3 must** (none of it changes arithmetic):
  1. **Split the softmax word before acting**: time P2's exp16, P2's ET
     stores, the caller's max, the caller's sum and P3's divides as
     separate words (a measurement change, not a kernel change), then
     cut the largest.
  2. **scores**: compile the P1 lead out (`-DATTN_M1_P1_LEAD=0`) and try
     the round-1 scalar splats against the splat vectors in the kernel's
     own P1 (a gtest-only sitting with two skels); keep whichever reads
     lower warm and cold.
  3. **append** (33 k): most of it is presumably the 2048 splat-vector
     stores (256 KiB). If round 3 drops the splat vectors this goes with
     them; otherwise time the rounding and the stores apart.
  The DMA-into-VTCM lever (plan §3.1, ≈ 45 L) stays unopened until the
  compute terms above are at their lines.

## Text approval (per-token-entry handoff)

The p01 text at G = 256 of A, Q1, Q2 and RQ2 is byte-identical (md5 of
the stripped text `e377add566c09b7ef2a0698ed106b8be` for all four, the
same as S2's); so are p02–p08 (G5). The decode PPL forced on A's
continuation is equal to 17 digits.

| variant | decode PPL (forced on A, p01, G = 256) | generated text (p01, G = 256) | text approved (user: y/n) |
|---|---|---|---|
| A | 1.42156 (nll_sum 90.048889370024341, source=self) | identical to S2's A text (md5 above) | (reference) |
| Q1 | (text only) | identical to A (same md5) | |
| Q2 | 1.42156 (nll_sum 90.048889370024341) | identical to A (same md5) | |
| RQ2 | 1.42156 (nll_sum 90.048889370024341) | identical to A (same md5) | |

## Notes (S3 build)

* **Vector rounding without qf32.** The plan named `hvx_rne16_sf` +
  `hvx_hf_narrow`; the narrowing goes through qf32, which has no -0, so a
  q / k / v that rounds to -0 (|x| < 2^-25, negative) would come back +0
  and could flip the sign of a zero score or output. `hvx_hf_round_row`
  does `hvx_hf_bits_rne`'s steps in integer ops plus one exact multiply
  and the exp16 magic add (the pinned sf ops exp16 was proven with), and
  `hvx_hf_store_sf` widens the output the same way (`hf_float`'s steps).
  Both are in `ATTN M1 HF PRIM` and matched on `hexagon-sim`.
* **No PV group of 3.** hexagon-clang 19 compiled the ng = 3 chain loop
  with one product moved to sf and back to qf32 (`vadd(v.sf, 0)`), an
  instruction sequence S1 never measured on silicon. P3 takes chain pairs
  (even gqa) and groups of 4 / 2 / 1, so LFM2.5 runs only the ng = 4 and
  ng = 2 loops (11 and 6 packets, round 1's code). An odd gqa (not in the
  model) runs 2 + 1 for a remainder of 3.
* **P1 loop.** Two passes of four accumulators, q from splat vectors: 11
  packets per 4 FMAs, 8 vector loads, no stack reload. The single
  8-accumulator pass (S1's `scores1` shape) compiled to 26 packets per 8
  FMAs in the kernel and was not taken.
* **Leads** are compile-time (`ATTN_M1_P1_LEAD`, `ATTN_M1_PV_LEAD`,
  `ATTN_M1_ET_LEAD`, default 1); per thread at most the ET box + one tile
  box (P1) or two V boxes (P3) are queued.
* **Shadow branch.** `dev/attn-shadow-170` is left as S2 used it; the
  round-2 shadow set is a new branch `dev/attn-shadow-170-r2` (the same
  commit cherry-picked without conflict), so no force-push was needed.
* The ISS numbers (one thread, pool NULL, L = 1024): append 13.6 k and
  pool 1.29 M pcycles (round 1 on the same harness: append 113 k, pool
  1.39 M). Not silicon, and the ISS does not see the fetch the leads
  target; S3 measures that.

## Notes from the run (S3)

`adb devices` listed three units (`R3CN80CW3FY`, `R3CY10WM83Y`,
`R5KL20NFRCK`); every command used `-s R3CY10WM83Y`. Thermal: zone0 32.8
°C at start, 38 °C after the gtests, 52 °C after the shadow cells, 58 → 68
°C over the speed cells, 64–66 °C through G6 and G5. `mhz` in the phase
lines 2096–2110. No FARF / AEE error besides rule 38's `AEE_ERPC` in
`RejectsBadShapes`. Both expectation mismatches are the G6 gates.
