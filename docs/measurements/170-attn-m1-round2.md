# Measurement 170 round 2 (S3): the round-2 ATTN_M1 against its speed gate, round 1 in the same sitting

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

## Results (fill in)

Reference (S2, same unit, 2026-09-29): A decode 52.46 / 50.84 / 48.26
tok/s at G = 64 / 512 / 1024; round 1 (Q1 there) 42.13 / 43.71 / 42.31;
round 1 `pcyc/op ATTN_M1` 394,501 / 700,449; round 1 cold pos 1023:
append 96 k, scores 702 k, softmax 526 k, pv 1,542 k, busy_max 590 k,
pool 619 k, `dsp_us` 340. Goal ≥ 50 tok/s decode; the gates 210 k / 350 k.

| gate | line | value | pass |
|---|---|---|---|
| G3 | 8 L × `bad` / `bad_stats`, `append_chain`, F16Det, rope64 | | |
| G4 | Q2 / RQ2 `tag3_heads`, `logits_equal_steps`, nll = An | | |
| G5 | nll = A (Q2, RQ2 × 8), text = A (Q2, RQ2 × 8), Q2 = Q1 (× 8) | | |
| G6 | Q2 `pcyc/op ATTN_M1` G = 64 / 1024 (Q1 same sitting) | | |

| term, cold pos 1023 | line | round 2 | round 1 (same sitting) |
|---|---|---|---|
| append | ≤ 10 k | | |
| scores | ≤ 550 k | | |
| softmax | ≤ 160 k | | |
| pv | ≤ 400 k | | |
| busy_max | ≤ 200 k | | |
| pool | ≤ 220 k | | |
| dsp_us | ≤ 110 | | |

| probe, wall pcyc / 64-lane FMA | lanes 1 | 2 | 4 | 6 |
|---|---|---|---|---|
| pv4 warm / cold / cold + 32 KiB lead | | | | |
| scores1 warm / cold / cold + tile lead | | | | |
| scores1 splat vectors / scalar splats | | | | |

| variant | G | run 1 prefill / decode / last 64 | run 2 prefill / decode / last 64 | decode mean | text = A |
|---|---|---|---|---|---|
| A | 64 | | | | ref |
| Q1 | 64 | | | | |
| Q2 | 64 | | | | |
| A | 512 | | | | ref |
| Q1 | 512 | | | | |
| Q2 | 512 | | | | |
| A | 1024 | | | | ref |
| Q1 | 1024 | | | | |
| Q2 | 1024 | | | | |

skel md5 (device): `new` / `q1` as `md5.txt` (`MD5 OK` in `sitting.out`).

## Text approval (per-token-entry handoff)

| variant | decode PPL (forced on A, p01, G = 256) | generated text (p01, G = 256) | text approved (user: y/n) |
|---|---|---|---|
| A | | <paste> | (reference) |
| Q1 | (text only) | | |
| Q2 | | | |
| RQ2 | | | |

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

<thermal, `mhz`, FARF / AEE errors, anything stale>
