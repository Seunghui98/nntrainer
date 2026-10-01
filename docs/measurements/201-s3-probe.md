# Measurement 201 S3 probe: the pool-28 miss cost is a boot-proximity effect, not reader placement

Two sittings on `R3CY10WM83Y` (S25, SM-S938N), 2026-10-01 17:27–17:37 KST,
run by the implementer directly (contract §4.1 as amended 2026-10-01).
Both ran **sitting 1's build** (`a2ebef9c9`, the set staged at
`/local/mnt/workspace/htp_moe/201/s3/`, `201-one-pd.md`'s artifact table;
device md5s of `nntrainer_causallm` `c106135d…`, `libnntr_hvx_skel.so`
`31c0a033…`, `libnntrainer.so` `651b23dd…`, `libcausallm_core.so`
`8f532d8a…` matched on both sittings, `MD5 OK`). One PD
(`NNTR_HTP_E2E=1 NNTR_HTP_E2E_PDS=1`), G = 64, `NNTR_HTP_PROFILE=2` on every
run (so **no cell here is a tok/s row of record**, contract §1.1), model
file pre-read (`cat` + `page_cache_evict -1`, resident MiB printed),
ceiling **3840 MiB after every run** of both sittings. **Every text ==
sitting 1's `A_G64_r1`, 20 of 20** (the runners' `text same`; re-checked by
the supervisor with the same `strip` + `cmp` on all 20 logs).

Runners: `201-s3probe-run.sh` (= `201/s3/run_s3probe.sh`, md5
`4963938478f0d689fa1ee820d5a0c357`) and `201-s3disc-run.sh` (=
`201/s3/logs_disc/run_disc.sh`, md5 `582dbdbd9228394bfdcccfaae53c010e`).
Logs: `201/s3/logs_probe/` (`sitting.out`, `summary.txt`, `ceiling.txt`,
per-run `.log` / `.logcat` / `ceil_*.log`) and `201/s3/logs_disc/`
(`sitting.out`, `summary.txt`, per-run `.log` and `.threads` = per-thread
state / utime / stime / cpu / `Cpus_allowed_list` samples every ≈ 0.8 s).

## Why

LEDGER ㉜ / rule 59 b: a pool-28 miss read 3.45–3.79 ms on two S25 units
against 0.5–0.7 at C = 16 / 24, with page cache resident and the prefetch
88 / 88 ahead. The probe's hypothesis (runner header): the prefetch reader
threads idle between the rarer misses, their cores downclock, so pinning
them to the big cores (`NNTR_MOE_PREFETCH_CPUS=6,7`) or using one reader
(`NNTR_MOE_PREFETCH_READERS=1`) changes the miss cost.

## Code facts (`origin/htp_decode` @ `87cc9c4a4`, read after sitting 1)

* The prefetch readers (`startPrefetchReaders`, pinned by
  `NNTR_MOE_PREFETCH_CPUS`) read **only in prefill**: `lfm2_moe_layer.cpp`
  enqueues when `total_tokens > 1`; in decode they wait on a condvar
  (`*.threads`: affinity `0-5,7`, 3–4 utime ticks over a run).
* A one-PD decode miss is read by the **pool server thread**:
  `poolAnswer` → `readExpert(st, use_pool=true)` → `readWeight` →
  `ThreadManager::parallel_for` over `min(compute threads, kExpertReadSlicesMax)`
  = **8 page-aligned slices**, one per compute worker, and the workers are
  hard-pinned **one per core (7, 0, 1, 2, 3, 4, 5)**; the server itself is
  created by main and inherits main's pin, **core 6**
  (`Cpus_allowed_list=6` in every `*.threads`; ≈ 0.3 s utime + 0.8–1.0 s
  stime over 64 tokens), where it spins for the token while main blocks in
  the dspqueue read. `readWeight`'s comment (doc 52 §10.7 / 10.9): the
  nibble `pread` into the uncached ION mapping is 82 % of a miss, capped
  near 4.9 GB/s; 8 slices bought 18 %.
* So the two knobs the probe turned move threads that do **not** run the
  decode miss. The sitting below shows it on silicon as well.

## Sitting 1 — probe (17:27:37–17:30:10 KST; uptime 60.3 s at start)

Order as run. zone0 31.2 °C at the block start; a 60 s cool wait before
runs 7–8 (zone0 56.0 → 33.2). Column `up ≈` is the uptime at the run's
end from the log mtimes (± a few s).

| # | run | setting beyond Q28 / Q24 | up ≈ s | decode tok/s | prefill tok/s | pool misses | miss_wait µs/token | server ms/round | profile file read ms (ms/miss) | resident MiB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Q28 | — | 99 | **32.64** | 538.9 | 121 | 8527.7 | 6.620 | 575.9 (**4.76**) | 4116 |
| 2 | Q28_R1 | `NNTR_MOE_PREFETCH_READERS=1` | 107 | **33.40** | 470.2 | 121 | 7793.1 | 6.081 | 529.0 (**4.37**) | 4116 |
| 3 | Q28_BIG | `NNTR_MOE_PREFETCH_CPUS=6,7` | 115 | 44.38 | 562.0 | 121 | 395.7 | 0.674 | 58.6 (0.48) | 4102 |
| 4 | Q28_BIG1 | `READERS=1 CPUS=7` | 123 | **34.65** | 515.1 | 121 | 6591.6 | 5.198 | 452.3 (**3.74**) | 3989 |
| 5 | Q24 | — (C = 24 control) | 130 | 41.53 | 551.1 | 229 | 1614.5 | 0.998 | 166.7 (0.73) | 4116 |
| 6 | Q24_BIG | `CPUS=6,7` | 138 | 43.10 | 547.0 | 229 | 787.0 | 0.689 | 115.1 (0.50) | 4116 |
| — | cool 60 s | | | | | | | | | |
| 7 | Q28_r2 | — | 206 | 43.87 | 534.4 | 121 | 754.1 | 0.939 | 81.7 (0.67) | 4116 |
| 8 | Q28_BIG_r2 | `CPUS=6,7` | 213 | 44.02 | 547.6 | 121 | 727.0 | 0.921 | 80.1 (0.66) | 3948 |

misses/token 1.89 (C = 28) / 3.58 (C = 24), rounds 87 / 167, `swap rpc
0.0 ms`, `calls/token=1.00`, `pds=1`, close clean on all eight.

**The reading made at the time — "reader-core placement decides the miss
cost" (Q28_BIG 0.48 against Q28 4.76) — was wrong.** It read run 3
against run 1 and did not weigh run 4 (`CPUS=7`, slow again) or runs 7–8
(default and `CPUS=6,7` equal at 0.67 / 0.66). Sitting 2 was run to
discriminate.

## Sitting 2 — discriminator (17:34:34–17:36:44 KST; rebooted, uptime 60.7 s at start)

D = default, B = `NNTR_MOE_PREFETCH_CPUS=6,7`, P0 = `NNTR_MOE_PREFETCH=0`
(the reader threads never start), interleaved × 4, zone0 32.8 °C at the
start, no cool wait inside (12 runs in 130 s).

| # | run | up ≈ s | decode tok/s | prefill tok/s | pool misses / token | miss_wait µs/token | server ms/round | profile file read ms (ms/miss) | reader cpus printed |
|---|---|---|---|---|---|---|---|---|---|
| 1 | D_1 | 99 | **32.08** | 460.8 | 121 / 1.89 | 8926.5 | 6.921 | 602.2 (**4.98**) | {0,1,2,3,4,5} |
| 2 | B_1 | 108 | **32.14** | 430.6 | 121 / 1.89 | 7395.2 | 5.817 | 506.1 (**4.18**) | {6,7} |
| 3 | P0_1 | 117 | **37.85** | 461.7 | 100 / 1.56 | 4026.2 | 3.748 | 312.4 (1.77 †) | — |
| 4 | D_2 | 126 | **32.10** | 471.0 | 121 / 1.89 | 8550.9 | 6.645 | 578.1 (**4.78**) | {0,1,2,3,4,5,7} |
| 5 | B_2 | 134 | 41.48 | 458.4 | 121 / 1.89 | 1574.7 | 1.538 | 133.8 (1.11) | {6,7} |
| 6 | P0_2 | 143 | 41.91 | 443.3 | 100 / 1.56 | 1464.6 | 1.620 | 216.4 (1.22 †) | — |
| 7 | D_3 | 152 | 42.55 | 477.6 | 121 / 1.89 | 1209.7 | 1.267 | 110.2 (0.91) | {0,1,2,3,4,5,7} |
| 8 | B_3 | 161 | 42.08 | 493.7 | 121 / 1.89 | 1497.3 | 1.485 | 129.2 (1.07) | {6,7} |
| 9 | P0_3 | 170 | 40.30 | 484.8 | 100 / 1.56 | 2370.5 | 2.385 | 210.4 (1.19 †) | — |
| 10 | D_4 | 179 | 41.75 | 466.7 | 121 / 1.89 | 1510.2 | 1.497 | 130.3 (1.08) | {0,1,2,3,4,5,7} |
| 11 | B_4 | 188 | 41.50 | 482.1 | 121 / 1.89 | 1592.6 | 1.556 | 135.3 (1.12) | {6,7} |
| 12 | P0_4 | 197 | 42.58 | 501.0 | 100 / 1.56 | 1114.7 | 1.325 | 129.8 (0.73 †) | — |

† P0's profile line counts **177** misses (the 77 prefill misses the
absent prefetch leaves, plus 100 in decode; `swap rpc 1.1 ms`), so its
ms/miss averages prefill and decode reads and is **not comparable** with
D / B's 121-miss figure; P0's decode-side reads are `miss_wait` and
`server ms/round`. P0 also enters decode with a different pool content
(1.56 misses a token, 76 rounds), so its tok/s is not a D / B cell either.

## Reading

1. **Reader placement does not decide the miss cost (refuted).** B_1
   (big-core pin) is as slow as D_1; P0_1, with no reader threads at all,
   is slow-ish (3.75 ms a round); from run 5 on, D, B and P0 read the same
   0.9–1.6 ms a round. `NNTR_MOE_PREFETCH_CPUS` / `_READERS` are not a
   decode lever; the S3 "reader cores" item of plan 201 is **not applied**
   (no code, no PR).
2. **The slow runs cluster right after a boot.** In both sittings every
   slow miss (3.7–5.0 ms) lies at uptime ≲ 130 s (probe runs 1, 2, 4;
   disc runs 1, 2, 4 and the partial run 3), every run after ≈ 130 s is
   fast (0.5–1.6 ms), whatever the setting — with one fast run inside the
   window (probe run 3, Q28_BIG at ≈ 115 s). Each sitting was started
   ≈ 60 s after the reboot, as the handoffs prescribe.
3. **Arena size / pool size is not it either**: the same Q28 (C = 28,
   3.3 GB arena, 13 chunks) reads 0.66–1.1 ms a miss once the boot is
   old, the same as C = 24 (0.50–0.73) — rule 59 b's "3.5–3.8 vs 0.5–0.7"
   compared a C = 28 cell that happened to be near a boot with C = 16 /
   24 cells that were not. Page cache was ruled out before (`201-fsu-e2e.md`;
   here resident 3948–4116 MiB on fast and slow runs alike).
4. **Nothing else in the app slows down in the slow runs**: the ARM
   staging memcpy reads 22–26 GB/s in all 20 runs (12.8 once, on a fast
   run), the M > 1 MoE `dsp` 15.1–16.2 ms a call throughout, prefill
   431–562 tok/s with no split by miss cost, the pinned workers' utime per
   run 55–80 ticks in slow and fast runs alike; the `L0` line puts the
   whole difference in `rt` (D_1 28.9 ms vs D_3 21.0 ms a token, `wake`
   0.15–0.19 both). The cost is inside the miss's `pread` into the ION
   arena: ≈ 5.4 MiB an expert pair (3328 MiB / 616 registered) at ≈ 1.1–1.5 GB/s when slow against ≈ 5–11
   GB/s when fast.
5. **Hypothesis for the cause (unmeasured):** post-boot background work
   (package scans, media / index services, zram / kswapd settling after
   the 4 GB `cat`) occupying one of the eight cores; since every miss is
   8 slices hard-pinned to 8 distinct cores plus the server on core 6, a
   busy core stalls the whole miss (the slice barrier waits for the last
   core). The `*.threads` samples cover only the app's threads, so per-core
   system load was **not** read — that is the next probe, with "2 slices
   on the big cores" and "slices unpinned" as the cheap counter-tests.

## What this changes

* LEDGER ㉜ stays open with its candidate list rewritten (page cache ✗,
  arena size ✗, reader placement ✗; boot proximity ✓ correlated; the
  8-slices-on-all-cores structure as the suspect); rule 61 (protocol: a
  boot's first ≈ 2 min are not measurement time); the pool-28 miss cells
  measured as a boot's first runs are flagged, not corrected:
  `201-one-pd.md` sitting 2's `Q28 G = 512 r1` (already noted cold) and
  `prof_Q28` (3.45), `201-fsu-e2e.md`'s 3.79, and `204-s26-rebaseline.md`'s
  `prof_Q28_q4` / `_q1` (4.69 / 3.99 on the developer S26, after that
  sitting's reboots) — **#208's "≈ 4.7 ms a miss on this phone" is
  unexplained and boot proximity is now a candidate for it.**
* The miss-cost cause is issue #216 (p1, `state:needs-plan`): per-core load
  after boot, then the miss read with 2 slices on the big cores / unpinned.

## Not verified here

* Per-core system load after boot (the hypothesis in 5).
* Whether a warm-up run or a ≥ 5 min wait removes the slow regime on
  purpose (both sittings reached the fast regime by elapsed time only).
* Any G ≥ 512 cell; at G ≥ 512 the 28-pool misses 0.12–0.34 a token, so
  even the slow regime costs 0.5–1.7 ms a token there.
* The prefill gate: profiled runs without an A in the sitting (sitting 1's
  A 538 / 514; here 431–562, read as spread, not as a gate).

## Oddities in the logs

* `summary.txt` of sitting 2 prints `reader cpus {0,1,2,3,4,5}` for D_1
  and `{0,1,2,3,4,5,7}` for D_2–D_4 (the app's own print of the reader
  set): the default reader set excludes the caller's core and was read
  once without core 7 — not followed up; the readers do not run in decode.
* Sitting 1's resident MiB reads 3989 (Q28_BIG1) and 3948 (Q28_BIG_r2)
  against 4116 elsewhere: a part of the file was evicted between the
  `cat` and the run on one slow and one fast run — residency does not
  split the two regimes.
* `convert to registry` varies 235–1495 ms across runs with no relation to
  the miss cost (Q28_BIG1 1495 slow, D_3 739 fast).
