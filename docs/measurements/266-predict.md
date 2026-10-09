# Measurement 266 S2: next-layer route prediction (lever 3), Gemma-4 26B QS4CX, S25 Ultra

Issue #266. Plan: `docs/plans/266-flash-miss-path.md` §3.3 and §4 S2 / S3.
S1 (miss readers) is in `266-miss-readers.md` (PR #271, merged).

**Result: the S2 decision rule fails at both prompts, so S3 was not started
and lever 3 closes at the S2 number.** The rule needs p ≥ 0.7 on at least
20 of 30 layers and at most +25 % net bytes. The guess reaches p ≥ 0.7 on
15 / 29 layers at p512 and 6 / 29 at p1024 (mean p 67.2 % / 61.0 %). A
prefetch of the full guess reads +62 % / +59 % more bytes per token.

**Prompts.** These are the 512 / 1024-token prompts of the S1 and 260 r2
sittings, not the new 1024 / 2048 / 4096 sitting standard. This run
continues the S1 lever measurement, so before and after stay comparable.

## Artifacts

| item | value |
|---|---|
| S2 build | `htp/266-predict` @ `1af7f74d0`. Code = `fdc35a598`: `htp_decode` @ `4c3953bb1` (#271 readers + #272 L0 split / DMA bypass) + the PREDICT instrument |
| first S2 cell (before the rebase) | `233cd3e75` = `htp/266-miss-readers` @ `a415594a6` + the same instrument. Skel `6e2e28ef…`, `libnntrainer.so` `7f99045e…` |
| toolchain | Hexagon SDK 6.4.0.1, HexKL 6.4.0.1, NDK r30 |
| `libnntr_hvx_skel.so` v79 (device) | `f052fcde914004e3456ba6dc7467259a`: `UNDEFINED SYMBOLS OK (68 runtime imports)`, `ARCH OK (V79)` |
| `libnntr_hvx_skel.so` v81 | `9d5a78c605802ec9a170217d5d3658e8`: `UNDEFINED SYMBOLS OK (68 runtime imports)`, `ARCH OK (V81)` |
| after the review fix `21618d478` (guards only; no device run) | skel v79 `2418487b6eec363629e1433620536cb0`, v81 `4482bbf9533c6327e9c84a0ccd948423`, both `UNDEFINED SYMBOLS OK (68 runtime imports)` / `ARCH OK`. For Gemma every guard holds (router in slot 0, out slot 2, every router softmax over 128 experts), so the guesses logged above are what this code logs |
| `nntrainer_causallm` | `8e355a64afb7449b65fad326594984e4` |
| `libcausallm_core.so` | `3a01c6ebae09212784a6a7a823e44e42` (`NNTR_HTP_FORWARD_KINDS` strings = 2) |
| `libnntrainer.so` (`jni/obj/local`) | `e9c7068ac3bd7111b767405f5f3ce4b1`: NEEDED `libsdkl.so` and `libcdsprpc.so`; strings `NNTR_HTP_PREDICT` and `compute_mhz` (#272) |
| `libccapi-nntrainer.so` | `e3f3f7a8f0241ea1e560e001373fbcfa` |
| `libc++_shared.so` / `libsdkl.so` / `unittest_hvx_two_sessions` | `b1586b9b…` / `0ad4e22a…` / `3ecada0e…`, copied on the device from `s266/` |
| device dir | `/data/local/tmp/nntrainer/causallm/s266p/`. Device `md5sum` == local for all 8 files |
| model / configs | `nntr_gemma4_qs4cx_fc_arm.bin`; the 260 r2 sitting's `s260cfg/r2_E_*` configs |
| runner | `266-predict-run.sh` (sources `260-e-run.sh`; lock owner `r266p@…`, `COOL_QUICK`). Cells `s2`, `s2_1024`, `s2_ppl` |
| device | S25 Ultra `R3CY205ZMND` (v79). Sitting 2026-10-09: 11:18–11:19 KST (pre-rebase cell), 12:06–12:22 KST (rebased cells) |

## What S2 measures

Plan §3.3. When `NNTR_HTP_PREDICT=1`, each `ROUTER_TOPK` op first runs the
next `ROUTER_TOPK` op's router (its weights, norm scale and per-expert
scale) on its own input. That input is the un-normed stream after layer L's
attention, which is what layer L's router reads. The guess goes to the out
slot, and the op's own router then rewrites that slot, so the real routing
moves no bit. The guessed top-8 is logged best-first beside the routed ids
and returned in the token response. The ARM writes it as the third field of
the `NNTR_HTP_ROUTE_LOG` line it predicts, and
`tools/moe_expert_cache_sim.py --predict trace` scores it. Layer 0 has no
guess, so 29 layers are scored.

## Cells

Cool start: `COOL_QUICK` (battery ≤ 32.0 °C or a 5 min cap). Every cell
started at battery ≤ 25.6 °C, so none waited. The p1024 cell followed the
p512 cell directly and started warm (cpu/nsp 47.3 °C, zone0 40.3 °C).

| cell | build | start cpu / bat / zone0 °C | end | prefill tok/s | **decode tok/s** | tokens | misses/token | **miss wait ms/token** | rounds | arm_ms/round | pgpgin MiB/miss | router pcyc/op | text == 260 r2 E | nll | peak RSS KiB | S1 ceiling |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 B p512 G512 (ref, `266-miss-readers.md`) | `275cfc06e` | — / 30.0 / 32.1 | — | 142.7 | 4.85 | 169 | 94.96 | 112.8 | 4838 | 4.38 | 2.44 | 225,245 | yes | — | 2,902,636 | 3584 |
| S2 p512 G512, pre-rebase | `233cd3e75` | 33.7 / 26.5 / 30.2 | 53.5 / 27.1 / 45.7 | 150.0 | 4.84 | 169 | 94.96 | 110.0 | 4838 | 4.28 | 2.41 | 453,851 | **yes** | — | 2,899,648 | 3584 |
| **S2 p512 G512** | `fdc35a598` | 29.4 / 24.5 / 26.7 | 51.2 / 25.6 / 42.6 | 147.1 | **4.98** | 169 (`<eos>`) | 94.96 | **115.7** | 4838 | 4.48 | 2.47 | 452,623 | **yes** | — | 2,894,576 | 3584 |
| S1 B p1024 G512 (ref) | `275cfc06e` | 37.6 / 32.6 / 34.9 | — | 191.5 | 4.13 | 512 | 120.81 | 142.2 | 15131 | 5.18 | 2.48 | — | yes | — | 2,946,480 | 3584 |
| **S2 p1024 G512** | `fdc35a598` | 47.3 / 25.6 / 40.3 (warm) | 58.1 / 27.8 / 48.8 | 191.6 | **3.95** | 512 | 120.81 | **161.4** | 15131 | 5.82 | 2.92 | 440,423 | **yes** | — | 2,948,756 | 3584 |
| S2 p512 G64 PPL | `fdc35a598` | 29.0 / 24.4 / 26.3 | 51.2 / 25.1 / 43.4 | 75.7 (PPL) | 4.43 | 64 | 99.42 | 124.8 | 1848 | 4.74 | 2.61 | — | yes | **4.57345** | 2,901,552 | 3584 |

**L0 per-kind wall, ms per token** (#272's `graph per-kind us/token` line,
rebased cells; `compute_mhz` = 2112 in both cells):

| cell | MOE (all) | **MOE net of the wait** | FC | DENSE_FFN | ATTN_M1 | LM_HEAD | ROUTER_TOPK (guess included) | RMSNORM + QK_NORM + ROPE + ADD | ops/wall |
|---|---|---|---|---|---|---|---|---|---|
| S2 p512 G512 | 139.9 | **24.1** | 12.3 | 6.1 | 12.2 | 8.3 | 6.4 | 3.6 | 0.9983 |
| S2 p1024 G512 | 186.0 | **24.5** | 12.4 | 6.3 | 22.2 | 8.3 | 6.3 | 3.6 | 0.9987 |

## Gates

| gate | p512 | p1024 |
|---|---|---|
| token ids == 260 r2 E (generated text byte-equal) | **pass**: 169 / 169 ending on `<eos>`, both builds | **pass**: 512 / 512 |
| routed ids == S1's route log (`cut -d'|' -f1,2`, `cmp`) | **pass** | **pass** |
| nll | **pass**: 4.57345 == 4.57345 (with PREDICT on) | not run. S2 changes no prefill code, and 3.61786 belongs to S3's gate, which was not started |
| misses/token unchanged | **pass**: 94.96 (16 049), rounds 4838 | **pass**: 120.81 (61 855), rounds 15131 |
| memory: peak RSS within +50 MiB of S1's B | **pass**: −7.9 MiB | **pass**: +2.2 MiB |
| S1 ceiling | 3584 after every cell. S1's A and B read the same; not attributed | same |
| router pcyc ≤ 2× (plan §3, lever 3) | 2.01× (452,623 / 225,245): the guess is one more router per layer, ≈ 3.2 ms/token at 2112 MHz | 440,423 |
| host (rung 1) | see below | |
| skel (rung 2) | v79 and v81 `UNDEFINED SYMBOLS OK`, `ARCH OK` | |

## Prediction accuracy per layer (the S2 number)

p is the mean share of the 8 routed experts of layer m that layer m − 1's
guess named.

| layer | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | mean | p ≥ 0.7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p512 % | 34 | 49 | 47 | 49 | 64 | 70 | 71 | 75 | 62 | 74 | 76 | 63 | 52 | 72 | 68 | 63 | 82 | 72 | 73 | 81 | 82 | 87 | 76 | 81 | 83 | 79 | 63 | 63 | 40 | **67.2** | **15 / 29** |
| p1024 % | 35 | 47 | 37 | 49 | 61 | 64 | 70 | 71 | 55 | 65 | 70 | 62 | 55 | 67 | 65 | 48 | 67 | 62 | 55 | 73 | 82 | 80 | 67 | 66 | 74 | 69 | 48 | 64 | 40 | **61.0** | **6 / 29** |

The guess is weakest at the first four layers and the last layer
(34–49 %), and best at layers 17–26 (up to 87 %). The pre-rebase p512 log
is byte-identical to the rebased one (`cmp`): the guess is deterministic
and does not depend on the read path.

## Policy replay: what the guess buys (`--predict`, C = 16)

All rows replay the device's routed ids. Columns:
- **misses/call**: what the DSP posts.
- **waited/call**: the misses not covered by a prefetch that came in time.
- **reads/call**: flash reads (demand + prefetch).

Per MoE call (×30 = per token). Rows:
- **lru**: plain LRU, the app today.
- **protect**: an eviction skips the guessed set of the next layer.
- **prefetch**: after layer L, read the guessed non-resident experts of L + 1 and enter them most recent.
- **prefetch-tail**: the same, entered least recent, so a wrong guess is the next victim.
- **trace:j**: only the first j guessed ids, best first.
- **oracle**: the next layer's real set (p = 1).
- **p<x>**: synthetic accuracy x, with wrong ids drawn uniformly.

**p512 G512** (S1 LRU 60.4 %, Belady 79.1 %):

| mode | guess | misses/call | hit % | waited/call | reads/call | bytes vs LRU |
|---|---|---|---|---|---|---|
| lru | — | 3.16 | 60.4 | 3.16 | 3.16 | — |
| protect | oracle / trace / p0.5–0.9 | 3.15–3.16 | 60.5–60.6 | 3.15–3.16 | 3.15–3.16 | 0 |
| prefetch | trace | 3.71 | 53.6 | 1.46 | 5.06 | +60 % |
| prefetch-tail | trace | 3.17 | 60.3 | **1.27** | 5.11 | **+62 %** |
| prefetch-tail | trace:4 | 3.17 | 60.4 | 2.03 | 3.69 | **+17 %** |
| prefetch-tail | trace:2 | 3.17 | 60.4 | 2.54 | 3.33 | +5 % |
| prefetch-tail | oracle | 3.16 | 60.5 | 0.13 | 3.16 | 0 |
| prefetch-tail | p0.7 (synthetic) | 3.17 | 60.3 | 1.04 | 5.33 | +69 % |

**p1024 G512** (S1 LRU 49.6 %, Belady 74.6 %):

| mode | guess | misses/call | hit % | waited/call | reads/call | bytes vs LRU |
|---|---|---|---|---|---|---|
| lru | — | 4.03 | 49.6 | 4.03 | 4.03 | — |
| protect | oracle / trace | 4.02 | 49.7–49.8 | 4.02 | 4.02 | 0 |
| prefetch | trace | 4.85 | 39.4 | 2.11 | 6.67 | +66 % |
| prefetch-tail | trace | 4.04 | 49.6 | **1.80** | 6.42 | **+59 %** |
| prefetch-tail | trace:4 | 4.03 | 49.6 | 2.66 | 4.78 | **+19 %** |
| prefetch-tail | trace:2 | 4.03 | 49.6 | 3.26 | 4.29 | +6 % |
| prefetch-tail | oracle | 4.03 | 49.7 | 0.16 | 4.03 | 0 |

**Reading.**
- **The protect policy moves the hit rate by 0.1–0.2 points at most**,
  even with an oracle: 60.4 → 60.6 at p512, 49.6 → 49.8 at p1024. LRU's
  victims are about two tokens old, and the next layer's experts are
  younger than that. Eviction policy alone is not where the Belady gap is.
- **A prefetch entered most recent raises the DSP-posted misses.** At
  p512, 3.16 → 3.71: wrong guesses push resident experts out. Entering the
  guesses least recent keeps misses/call at 3.17. If S3 is ever built,
  its "misses/token unchanged by construction" (plan §3.3) holds only with
  tail placement. That is a design fact the plan did not state.
- **The full guess hides 60 % / 55 % of the waited misses** (3.16 → 1.27,
  4.03 → 1.80), but reads 62 % / 59 % more bytes. That is over the +25 %
  rule and, at S1's 2.4 GiB/s in-app rate, about +70 ms of extra flash time
  per token at p512. Every wrong guess is almost always a non-resident
  expert, so the bytes follow 1 − p.
- **The first 4 ids fit the byte rule (+17 % / +19 %)** and hide 36 % /
  34 % of the waited misses. This is outside the plan's rule, which is
  written for the full guess with p ≥ 0.7. It is the only variant here
  that a plan revision could take up. Its effect on tok/s depends on how
  much of the extra 17 % of flash time overlaps the 80 ms of compute, and
  that was not measured.

## Combined "now" reading (coordinator)

On `htp_decode` @ `4c3953bb1` (readers + DMA bypass + L0 split), with the
guess on:
- **p512 G512: 4.98 tok/s, miss wait 115.7 ms.** S1 alone: 4.85 / 112.8.
- **p1024 G512: 3.95 tok/s, miss wait 161.4 ms.** S1 alone: 4.13 / 142.2.

Caveats:
- The guess costs about 3.2 ms/token of the 6.4 ms `ROUTER_TOPK` row. Off,
  p512 would read ≈ 5.06 tok/s at the same wait (arithmetic, not a run).
- The p1024 cell started warm, right after the p512 cell (zone0 40.3 °C
  against S1 B's 34.9 °C). It read 2.92 MiB pgpgin per miss against S1's
  2.48, and its wait is 19 ms above S1's on the same misses. One run; this
  is not attributed to the build.
- **MOE net of the wait is 24.1 / 24.5 ms.** FC is 12.3 ms, DENSE_FFN
  6.1 ms and ATTN_M1 12.2 / 22.2 ms. The wait (115.7 / 161.4 ms) is still
  83 % / 87 % of the MOE row.

## Host gates (rung 1: `fdc35a598`, again on `21618d478` after the review fix)

- `ninja -C build`: built.
- `run_host_checks.sh`, on both trees: `ALL CHECKS PASS` (×4) and `WORKER POOL LANES OK`.
  `GRAPH STRETCH BIT-IDENTICAL … PREDICT on, its guess == router 1's spec`
  passes for both the LFM2 sigmoid router and the Gemma softmax router.
- `htp_syntax_check.sh`: exit 0.
- `*qs4cx*` 3 / 3 and `*Lfm2Moe*` 7 / 7: run on the pre-rebase tree
  (`233cd3e75`). The rebase touched only `htp_dspq_wire.h`'s response
  struct.
- `run_inproc_e2e.sh`, default (`fdc35a598` tree): `INPROC E2E PASS`. Every
  pool line has `bit_identical=1` with S1's `misses=`: hd64 C=2 5, lfm25 C=1 56,
  C=2 14, gemma64 C=3 19, 2bit C=1 / C=2 56 / 13, keys 15, fcwh 11.
- `run_inproc_e2e.sh` with `NNTR_HTP_PREDICT=1 NNTR_HTP_ROUTE_LOG=…`, built
  on the review-fixed sources: `INPROC E2E PASS`, with the same pool lines
  and the same `misses=`. The last process's route log has 28 lines
  (7 tokens × 4 MoE ops). 21 of them carry a guess: layers 1–3, none on
  layer 0. This is the alignment `routeLog` relies on, which the device logs
  show too (29 guessed lines per token, all with 8 ids).
- `tools/moe_expert_cache_sim.py --selftest`: `selftest OK`.

## Rerun

```
source tools/htp/env.sh; ./test/htp/build.sh; (cd Applications/CausalLM && ./build_android.sh --htp --cache)   # after (cd builddir && ninja install)
docs/measurements/266-predict-run.sh s2 <log dir>; docs/measurements/266-predict-run.sh s2_1024 <log dir>; docs/measurements/266-predict-run.sh s2_ppl <log dir>
REF=<260 r2 log dir> docs/measurements/266-predict-run.sh gate <log dir>
python3 tools/moe_expert_cache_sim.py <log dir>/pred_p512_g512.txt --cache 16 --predict trace trace:4 trace:2 oracle p0.7
```
