# Measurement 266: flash miss readers on the one-PD decode (S0 probe + S1 sitting), Gemma-4 26B QS4CX, S25 Ultra

Issue #266, plan `docs/plans/266-flash-miss-path.md` (§4 S0, S1). Reference
for every E row: the 260 r2 sitting, `260-step2-r2-3897963d8.md` (E p512 G512
3.33 tok/s, miss wait 208.5 ms/token; p1024 G512 2.58, 286.9).

## Artifacts

| item | value |
|---|---|
| S1 build | `htp/266-miss-readers` @ `275cfc06e` (`htp_decode` @ `2fde0591f` = `3897963d8` + docs-only merges #265 / #268 / #269, + the probe / sim commit `6be85790a`) |
| toolchain | Hexagon SDK 6.4.0.1, HexKL 6.4.0.1 (`hexkl-1.0-beta.2`), NDK r30, v79 |
| `libnntr_hvx_skel.so` | `4c665097de89238e68d9cdc8731c1f48`, copied from `s260r2/`. S1 changes no DSP source and no IDL (`git diff 3897963d8 -- test/htp nntrainer/tensor/htp_backend` = `htp_compute_ops.cpp` only, ARM side) |
| `nntrainer_causallm` | `131e2b8cd355355578bc21cc1119cc30` |
| `libcausallm_core.so` | `26207ed76611f6c47b8fdb4229a217b2` (`NNTR_HTP_FORWARD_KINDS` strings = 2) |
| `libnntrainer.so` (`jni/obj/local`) | `3f08cde313b465fa94200fa2cfa84cfa` (NEEDED `libsdkl.so`, `libcdsprpc.so`; strings `NNTR_MOE_MISS_READERS`, `NNTR_HTP_ROUTE_LOG`) |
| `libccapi-nntrainer.so` (`jni/obj/local`) | `b4ecec9ca7382e3dd7fc1dd99d5e2497` |
| `libc++_shared.so` / `libsdkl.so` / `unittest_hvx_two_sessions` | `b1586b9b…` / `0ad4e22a…` / `3ecada0e…`, copied from `s260r2/` |
| device dirs | S1 = `/data/local/tmp/nntrainer/causallm/s266/` (device `md5sum` == local for all 8 files); A = `s260r2/`, untouched |
| model / configs | `nntr_gemma4_qs4cx_fc_arm.bin` `1d00c31d…`; the 260 r2 sitting's `s260cfg/r2_E_*` configs, unchanged |
| probe | `tools/htp/pread_probe.c` @ `6be85790a`, NDK r30 `aarch64-linux-android26-clang -O2 … -ldl`; raw output `266-pread-probe.txt` |
| runner | `266-run.sh` (sources `260-e-run.sh`: cells, cool start, logs, S1 ceiling); `run` = the first grid, stopped after 4 cells; `quick` = the cut plan |
| device | S25 Ultra `R3CY205ZMND` (v79), 11.1 GB RAM; every run under `tools/htp/sitting_lock.sh` |
| sitting | 2026-10-09: probe 08:54–08:58 KST, A trace 08:59–09:02, grid 09:28–10:12 (A p512 G512, B, D, A p1024 G512), quick 10:13–10:28 (B p1024 G512, B PPL) |

**Cool starts.**
- The first four cells used the 260 r2 protocol: 180 s, then cpu/nsp ≤ 38 °C, battery ≤ 30.0 °C and zone0 ≤ 35 °C, with a 20 min cap. They started at battery 30.0 °C and zone0 31.8–32.5 °C.
- **Relaxation (user, 2026-10-09):** the user cut the plan to the minimum. The last two cells waited for battery ≤ 32.0 °C or a 5 min cap (`COOL_QUICK=1`), and both hit the cap:
  - B p1024 G512 started at battery 32.6 °C, cpu/nsp 37.6 °C, zone0 34.9 °C;
  - B PPL started at 33.6 / 38.4 / 36.0 °C.
- Both quick-cool cells started warmer than their A. Any bias from that works against B.

**Cut plan (user 2026-10-09).** The plan's grid (A / B / D × p512 / p1024 × G 64 / 512 / 1024, A first and last, a PPL cell per prompt) was stopped after 4 cells. What ran instead:
1. B p512 G512;
2. B p1024 G512;
3. D p512 G512 (it had already run in the grid) and one PPL cell, p512 G64.

No G64 / G1024 speed cells and no A at the end. The 260 r2 sitting's E rows are the A of record. The two A cells of this sitting are a same-day control.

## S0: the pread probe

`pread_probe <file> --pressure 4300 --n 64 --reps 2 --round 3`:
- **Ranges.** Per cell, 64 Gemma-sized experts: gate_up 1 993 728 B and down 1 013 760 B, at seeded random 64 B-aligned offsets, so not 4 KiB-aligned, like the real ones.
- **Rounds.** Rounds of 3 experts, each followed by a barrier; E averages 3.32 misses per round.
- **Pressure.** 4300 MiB of anon memory, touched with non-constant data.
- **Isolation.** `POSIX_FADV_DONTNEED` on the whole file before each cell, and each cell reads its own 1/64 stripe of the expert region.
- **Destination.** The ION ring is `rpcmem_alloc` heap 25, uncached, like the arena; malloc is a control.

Each cell reads mean GiB/s over the requested bytes (rep 0 / rep 1) · ms per round · pgpgin MiB per expert. Requests:
- **slices**: each weight split in T page-aligned slices, experts one after another. This is the old miss path's shape.
- **1m**: 1 MiB jobs.
- **weight**: one whole-weight `pread` per job.
- **expert**: both weights per job.

| dest | mode | request | T=1 | T=2 | T=4 | T=8 |
|---|---|---|---|---|---|---|
| ion | pread | slices | 1.38* · 1.01 ms · 2.12 | 1.71* · 0.82 ms · 1.88 | 1.82* · 0.77 ms · 1.84 | **1.81** (1.81 / 1.82) · 0.77 ms · 1.85 |
| ion | pread | 1m | 1.40 (1.43 / 1.36) · 5.84 ms · 2.02 | 2.12 (2.11 / 2.13) · 3.85 ms · 1.97 | 2.96 (3.00 / 2.93) · 2.75 ms · 1.92 | 3.30 (3.27 / 3.33) · 2.47 ms · 1.91 |
| ion | pread | weight | 1.47 (1.49 / 1.45) · 5.56 ms · 2.08 | 2.46 (2.50 / 2.41) · 3.32 ms · 2.04 | **3.15** (3.16 / 3.14) · 2.59 ms · 2.03 | 3.09 (3.06 / 3.13) · 2.63 ms · 2.04 |
| ion | pread | expert | 1.51 (1.49 / 1.54) · 5.38 ms · 2.01 | 2.12 (2.16 / 2.07) · 3.85 ms · 2.10 | 2.77 (2.70 / 2.85) · 2.94 ms · 2.04 | 2.70 (2.65 / 2.76) · 3.02 ms · 2.01 |
| ion | random | slices | 1.27 (1.19 / 1.36) · 1.10 ms · 1.75 | 1.72 (1.67 / 1.76) · 0.82 ms · 1.84 | 2.00 (1.86 / 2.13) · 0.71 ms · 1.68 | 1.80 (1.72 / 1.88) · 0.78 ms · 1.80 |
| ion | random | 1m | 1.30 (1.30 / 1.30) · 6.29 ms · 1.84 | 2.26 (2.29 / 2.24) · 3.60 ms · 1.69 | 3.13 (3.16 / 3.10) · 2.60 ms · 1.77 | 3.29 (3.30 / 3.29) · 2.48 ms · 1.83 |
| ion | random | weight | 1.33 (1.33 / 1.32) · 6.14 ms · 1.84 | 2.36 (2.45 / 2.27) · 3.46 ms · 1.79 | 3.12 (3.17 / 3.07) · 2.61 ms · 1.79 | 3.04 (2.96 / 3.12) · 2.68 ms · 1.84 |
| ion | random | expert | 1.32 (1.36 / 1.29) · 6.16 ms · 1.86 | 2.10 (2.11 / 2.08) · 3.89 ms · 1.72 | 2.76 (2.69 / 2.84) · 2.95 ms · 1.80 | 2.80 (2.73 / 2.88) · 2.91 ms · 1.79 |
| ion | bounce | slices | 1.80 (1.79 / 1.80) · 0.78 ms · 2.88 | 1.88 (1.87 / 1.89) · 0.74 ms · 2.88 | 1.57 (1.56 / 1.58) · 0.89 ms · 2.90 | 1.49 (1.49 / 1.48) · 0.94 ms · 2.93 |
| ion | bounce | 1m | 1.67 (1.61 / 1.73) · 4.88 ms · 2.88 | 2.70 (2.60 / 2.81) · 3.02 ms · 2.88 | 2.94 (2.93 / 2.95) · 2.78 ms · 2.88 | 2.88 (2.86 / 2.90) · 2.83 ms · 2.88 |
| ion | bounce | weight | 1.87 (1.86 / 1.89) · 4.35 ms · 2.88 | 2.80 (2.68 / 2.91) · 2.92 ms · 2.88 | 2.99 (3.01 / 2.98) · 2.72 ms · 2.88 | 2.92 (2.85 / 2.98) · 2.80 ms · 2.88 |
| ion | bounce | expert | 1.91 (1.85 / 1.98) · 4.27 ms · 2.88 | 2.65 (2.71 / 2.59) · 3.08 ms · 2.88 | 2.96 (3.14 / 2.79) · 2.76 ms · 2.88 | 2.97 (3.08 / 2.86) · 2.75 ms · 2.88 |
| ion | direct (`O_DIRECT` into ION) | expert | | | **`EFAULT` (errno 14) on every read** | |
| malloc | pread | slices | | | 1.90 (1.92 / 1.88) · 0.74 ms · 1.75 | 1.84 (1.82 / 1.85) · 0.76 ms · 1.83 |
| malloc | pread | expert | | | 2.76 (2.69 / 2.82) · 2.96 ms · 1.98 | 2.78 (2.76 / 2.80) · 2.93 ms · 1.98 |
| malloc | bounce | slices | | | 1.76 (1.58 / 1.94) · 0.80 ms · 2.90 | 1.83 (1.99 / 1.68) · 0.77 ms · 2.93 |
| malloc | bounce | expert | | | 3.02 (3.02 / 3.03) · 2.70 ms · 2.88 | 3.00 (3.10 / 2.91) · 2.71 ms · 2.88 |

\* rep 1 only. Rep 0 of these three cells read stripes 0–2, which a warm-up run had just read, and shows pgpgin ≈ 0. Slices "ms" is per weight, not per round of 3. PSI io `some avg10` was 0.1–17.9 over the run. The stripes cycle every 64 cells and the run had 57 cells per rep, so from its 8th cell on rep 1 re-reads the stripes that rep 0's cells read about 50 cells (≈ 10 GB of reads) earlier. Rep 0 and rep 1 agree within 0–12 % in every cell except the starred ones (the decision cells: 3.16 / 3.14, 1.81 / 1.82).

**Anomaly, not resolved.**
- Buffered cells show 1.7–2.1 MiB of pgpgin per 2.87 MiB expert. `mincore` reports 0 pages of the file resident after `DONTNEED`.
- A `dd` re-read of an evicted 200 MiB range gave pgpgin 0, three times, at 1.3–1.4 GB/s.
- `O_DIRECT` cells always show 2.88.
- So on this unit, pgpgin under-counts what a buffered re-read costs, or some cache outside the page cache serves part of it. Where the bytes come from was not found. The rates above are wall time over requested bytes, and the app A/B below is the reading that counts.

**Decisions (plan §4 S0 gate; posted on #266):**
- **Lever 1: in.** One job per weight, 4 readers: 3.15 GiB/s against the old shape's 1.81 (1.74×). 1 MiB jobs at T=8 are 5 % faster and need twice the threads, so they were not taken.
- **Lever 2 (`O_DIRECT` → bounce): out.** It reads 2.99 against 3.15 (0.95×), under the 1.2× rule. `O_DIRECT` straight into ION fails with `EFAULT`, as the plan expected.
- **Lever 4 (`FADV_RANDOM`): out.** It reads 3.12 against 3.15. The load-end `DONTNEED` pass was not built: the plan defaults it to off, and the probe gave no signal for it.

### Hit-rate curve (`NNTR_MOE_TRACE` on A, then E's own route log)

Replayed with `tools/moe_expert_cache_sim.py --cache 16` (the reuse curve and the per-layer line were added in `6be85790a`):

| trace | tokens | LRU hit % (C = 16) | Belady % | same-layer reuse from the previous d tokens: d = 1 / 2 / 3 / 4 / 8 | LRU hit % per layer (min–max) |
|---|---|---|---|---|---|
| A p512 G512 (`NNTR_MOE_TRACE`, s260r2; text == the 260 r2 A; loops after ≈ 150 words) | 512 | 45.3 | 76.5 | 33.9 / 43.3 / 55.3 / 71.2 / 80.3 | 34–55 |
| **E p512 G512** (B's `NNTR_HTP_ROUTE_LOG`) | 169 | **60.4** | **79.1** | 46.8 / 56.7 / 63.1 / 67.6 / 77.9 | 43–70 |
| **E p1024 G512** (B's route log) | 512 | **49.6** | **74.6** | 36.9 / 48.3 / 56.1 / 62.6 / 75.1 | 35–60 |

- Replay against the app: the sim's LRU gives 3.16 × 5070 = 16 021 misses against the app's 16 049 at p512 (−0.17 %), and 4.03 × 15 360 = 61 901 against 61 855 at p1024 (+0.07 %).
- These are not equal by construction. The sim starts from an empty pool, while the app's pool holds the prefill's last experts. The plan's "reproduce 16 049" is therefore met to 0.2 %, not exactly.
- Belady at C = 16 would cut misses about in half, to ≈ 50 / 61 per token. That is the ceiling for any policy or prediction lever (lever 3).

## S1: B (4 miss readers) and D (1 reader) against A

Every cell is E: `NNTR_HTP_E2E=1`, C = 16, the r2_E configs. A = `s260r2/` with nothing set. B = `s266/` with defaults. D = `s266/` with `NNTR_MOE_MISS_READERS=1`.

Columns:
- **DSP non-wait** = DSP wall − miss wait.
- **implied rate** = 2.93 MiB × misses ÷ wait.

| cell | cool start (bat / zone0 °C) | prefill tok/s | **decode tok/s** | tokens | misses/token | **miss wait ms/token** | DSP wall ms | DSP non-wait ms | arm_ms/round | pgpgin MiB/miss | implied rate GiB/s | mhz | text == 260 r2 E | peak RSS KiB | S1 ceiling |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 260 r2 E p512 G512 (ref) | — / 30.6 | 143.8 | 3.33 | 169 | 94.96 | 208.5 | 289.1 | 80.6 | 7.71 | 2.46 | 1.30 | 1180 | — | 2,894,412 | 3840 |
| A p512 G512 | 30.0 / 32.5 | 160.1 | 3.41 | 169 | 94.96 | 200.8 | 281.5 | 80.7 | 7.44 | 2.40 | 1.35 | 1190 | yes | 2,900,380 | 3584 |
| **B p512 G512** | 30.0 / 32.1 | 142.7 | **4.85** | 169 | 94.96 | **112.8** | 194.3 | 81.5 | 4.38 | 2.44 | 2.41 | 1369 | **yes** | 2,902,636 | 3584 |
| D p512 G512 | 30.0 / 32.1 | 160.1 | 4.19 | 169 | 94.96 | 151.7 | 227.4 | 75.7 | 5.69 | 3.55 | 1.79 | 1248 | yes | 2,896,628 | 3584 |
| 260 r2 E p1024 G512 (ref) | — | 191.1 | 2.58 | 512 | 120.81 | 286.9 | 379.2 | 92.4 | 10.08 | 2.56 | 1.20 | 1133 | — | 2,940,116 | 3584 |
| A p1024 G512 | 30.0 / 31.8 | 188.6 | 2.47 | 512 | 120.81 | 306.0 | 397.9 | 91.9 | 10.72 | 2.79 | 1.13 | 1115 | yes | 2,940,624 | 3584 |
| **B p1024 G512** | 32.6 / 34.9 (5 min cap) | 191.5 | **4.13** | 512 | 120.81 | **142.2** | 234.2 | 92.0 | 5.18 | 2.48 | 2.43 | 1333 | **yes** | 2,946,480 | 3584 |
| B p512 G64 PPL | 33.6 / 36.0 (5 min cap) | 67.9 (PPL) | 4.10 | 64 | 99.42 | 133.9 | 214.3 | 80.4 | 5.06 | 2.81 | 2.13 | 1310 | — | 2,902,328 | 3584 |

**nll** (`NNTR_PPL=1`, prompt rows): B p512 **4.57345** == the 260 r2 E 4.57345 == A 4.57345. **Pass.** S1 touches no prefill code, and the prefill path is the one this line scores. No p1024 PPL cell was run (cut plan).

**Gates (plan §1):**

| gate | p512 | p1024 |
|---|---|---|
| token ids == 260 r2 E (generated text byte-equal) | **pass**: B 169 / 169 (`<eos>`), D 169 / 169 | **pass**: B 512 / 512 |
| nll equal | **pass**: 4.57345 | not run (cut) |
| misses/token unchanged | **pass**: 94.96 (16 049) | **pass**: 120.81 (61 855) |
| pgpgin per miss ≤ 2.46 MiB | **pass**: 2.44 | 2.48. Below the same prompt's references (E 2.56, A 2.79); +0.02 over the p512 number read literally |
| memory: `s1_arena_mib` 1408, `mapped_mib` 449, peak RSS ≤ +50 MiB | **pass**: 1408 / 449, +2.2 MiB against A | **pass**: 1408 / 449, +5.7 MiB |
| skel md5 unchanged | **pass**: `4c665097…` | same |
| prefill ≥ 0.95 × A | **0.89** (142.7 / 160.1), one run; D on the same binary 160.05 | **pass**: 1.02 |
| miss wait ≤ 60 ms (issue target) | 112.8: **not met** | 142.2: **not met** |

**Reading.**
- **The miss wait halves at the same misses:**
  - p512: 200.8 → 112.8 ms/token (−44 % against this sitting's A, −46 % against the 260 r2 E);
  - p1024: 306.0 → 142.2 (−54 %; −50 % against E).
- **Decode speed:** 3.41 → 4.85 tok/s (+42 %) at p512 and 2.47 → 4.13 (+68 %) at p1024.
- **Compute and the round count are unchanged.** DSP non-wait stays at 80.7 → 81.5 and 91.9 → 92.0 ms, and the rounds/token are the same (4838 / 15131).
- **The implied in-app rate is 2.41–2.43 GiB/s.** The plan's floor row at 2.5 GiB/s (111 / 142 ms) is where B lands. The probe's 3.15 does not fully carry over to the app, which is under real pressure and pays its own `arm_ms/round` overhead (the answer and the harvest).
- **D separates the two effects.** At p512, whole-weight requests alone (no parallelism) give 200.8 → 151.7 ms, and the 4 readers take that to 112.8. D's pgpgin of 3.55 MiB/miss shows a single sequential `pread` pulling readahead past the weight. The parallel readers do not (2.44).
- **The issue's ≤ 60 ms needs more than read rate.** It needs overlap with compute (lever 3) or fewer bytes per token (plan §0.2): no read-rate lever reaches it at C = 16.
- **Prefill gate at p512 (0.89): not attributed to S1.** No prefill code changed: the readers run only inside `poolAnswer`, and the prefill keeps the sliced `readExpert(use_pool=true)`. D, on the same binary in the same sitting, reads 160.05. The 260 r2 sitting's E prefill spread was 143.8–158.1. This is one run, and its prefill rows have no per-stage breakdown. The p1024 pair reads 1.02.
- The S1 ceiling read 3584 after every cell, A included. As in the 260 r2 sitting, this is not attributed.
- **Thermal:** B's average DSP clock reads higher (1333–1369 MHz against A's 1115–1190). This is the same dilution effect the 260 r2 doc describes: fewer sleeping waits inside the clock average. It is not a different DVFS state.

## Host gates (rung 1, `275cfc06e`)

- `ninja -C build`: built.
- `*Lfm2Moe*`: 7 / 7 passed.
- `*qs4cx*`: 3 / 3 passed.
- `run_host_checks.sh`: `ALL CHECKS PASS`, `WORKER POOL LANES OK`.
- `htp_syntax_check.sh`: exit 0.
- `run_inproc_e2e.sh` (readers = 4): `INPROC E2E PASS`. Its pool lines:
  - `E2E e3 pool C=2 hd64 … bit_identical=1 misses=5`;
  - `C=1 lfm25 … misses=56`;
  - `C=2 lfm25 … misses=14`;
  - `E2E e3 pool C=3 gemma64 … bit_identical=1 misses=19`;
  - `2bit pool C=1 / C=2 … misses=56 / 13`;
  - keys `pool C=2 … misses=15`.
- With `NNTR_MOE_MISS_READERS=1`: see the PR.

## Rerun

```
source tools/htp/env.sh; (cd Applications/CausalLM && ./build_android.sh --htp)   # s266/ set above
docs/measurements/266-run.sh quick <log dir>                                       # B p1024 G512, B PPL
REF=<260 r2 log dir> docs/measurements/266-run.sh gate <log dir>
pread_probe /data/local/tmp/nntrainer/gemma4_26b_qs4cx/nntr_gemma4_qs4cx_fc_arm.bin --pressure 4300 --n 64 --reps 2 --round 3
```
