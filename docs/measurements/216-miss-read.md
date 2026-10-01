# Measurement 216: the slow pool miss is a storage read (page cache evicted under memory pressure), not a busy core

One sitting on `R3CY10WM83Y` (S25, SM-S938N), 2026-10-01 21:43–22:01 KST,
run by the implementer directly (agents run adb since 2026-09-30), plan
`docs/plans/216-miss-read-slices.md` §4 steps 1–2. **Sitting 1's build**
(`a2ebef9c9`, the set at `/local/mnt/workspace/htp_moe/201/s3/app`, staged
read-only into a new device dir `s216a`; `s201s3` untouched): device md5s
`nntrainer_causallm` `c106135d…`, `libnntr_hvx_skel.so` `31c0a033…`,
`libnntrainer.so` `651b23dd…`, `libcausallm_core.so` `8f532d8a…` == its
`md5.txt`, `MD5 OK` on all three boots. One PD (`NNTR_HTP_E2E=1
NNTR_HTP_E2E_PDS=1`), `NNTR_MOE_CACHE_EXPERTS=28`, prompt 512, G = 64,
`NNTR_HTP_PROFILE=2` on every run (**no cell here is a tok/s row of
record**, contract §1.1), model file pre-read (`cat` + `page_cache_evict
-1`) as in #201 S3. **Text == sitting 1's `A_G64_r1` 12 / 12**
(`strip` + `cmp`), `calls/token=1.00` 12 / 12, **ceiling 3840 MiB after
every run** (12 / 12).

Runner `216-core-load-run.sh` (reboot → `216-sampler.sh` on the phone from
`sys.boot_completed`, uptime ≈ 16 s, every 0.5 s → runs; md5 below),
fold `216-core-load-report.py`. Logs:
`/local/mnt/workspace/htp_moe/216/logs_core/{b1,b2,b3}/` (`sitting.out`,
per-run `.log`, `core.samples`, `top.frames` (b1, b3), `report.md` = the
fold's full output with the 10 s timeline).

| file | md5 |
|---|---|
| `216-core-load-run.sh` | `a722e9c51f82be4903cc2e58b2d626fd` |
| `216-sampler.sh` (as committed; b1 / b2 ran it without the `pgpgin` / `workingset_refault_file` / meminfo lines, added before b3) | `baf97d58e680fc55422ef8789ea9a70b` |
| `216-core-load-report.py` | `f12806d3972bff11cb7d3dd618de4c8a` |

## Design (as run)

* **b1, b2:** reboot, sampler on, runs at uptime ≈ 60 / 120 / 180 / 300 s
  (b2 without `top` frames, plan §5's perturbation check).
* **b3 (control):** reboot, sampler on, phone idle to uptime 360 s, then
  four runs back to back.
* Per run, the **decode window** is the sample intervals in which the pool
  server thread (the app task pinned to core 6 other than main) gained
  ticks (1.1–2.2 s, = 64 tokens at 30–44 tok/s). **"Other busy"** per core
  = `/proc/stat` busy − the app's own task ticks on that core; the column
  gives the busiest core. All counters are deltas over the decode window.

## Uptime × per-core busy × ms/miss (decode window)

| boot | run | up s | decode tok/s | prefill tok/s | ms/miss | arm_ms/round | miss_wait µs/tok | busiest other core (% busy) | MHz cpu0/6/7 | DDR MHz | kswapd scan/s | PSI mem/io/cpu ms | pgpgin MiB | file refaults |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b1 | U60 | 60 | 30.51 | 538.4 | **5.77** | 8.020 | 10518 | cpu1 13 | 2285/3366/3366 | 3878 | 107052 | 60/**213**/104 | n/s | n/s |
| b1 | U120 | 120 | 41.94 | 538.4 | 1.31 | 1.823 | 1973 | cpu0 7 | 2227/4474/4474 | 4403 | 25286 | 8/31/37 | n/s | n/s |
| b1 | U180 | 181 | 44.08 | 541.8 | 0.65 | 0.903 | 705 | cpu6 10 | 2003/4474/4474 | 4403 | 19789 | 5/0/35 | n/s | n/s |
| b1 | U300 | 300 | 34.93 | 545.8 | **3.72** | 5.168 | 6547 | cpu6 16 | 2227/3757/3757 | 2623 | 99200 | 201/**159**/22 | n/s | n/s |
| b2 | U60 | 60 | 30.87 | 485.3 | **5.32** | 7.399 | 9624 | cpu0 30 (also 1, 4, 5: 26–28) | 2726/3821/3821 | 3432 | 138735 | 73/**296**/219 | n/s | n/s |
| b2 | U120 | 120 | 33.77 | 550.5 | **4.21** | 5.849 | 7484 | cpu0 5 | 2270/3754/3754 | 4224 | 89777 | 12/**220**/39 | n/s | n/s |
| b2 | U180 | 180 | 34.08 | 508.4 | **4.03** | 5.609 | 7156 | cpu6 9 | 2285/3834/3834 | 4224 | 108683 | 12/**201**/30 | n/s | n/s |
| b2 | U300 | 301 | 33.14 | 548.8 | **4.49** | 6.244 | 8010 | cpu0 3 | 2227/3514/3514 | 3513 | 91694 | 9/**209**/25 | n/s | n/s |
| b3 | R1 | 361 | 32.06 | 483.9 | **5.03** | 6.990 | 9027 | cpu6 16 | 2509/4474/4474 | 3878 | 121004 | 25/**258**/79 | **610** | 151527 |
| b3 | R2 | 373 | 44.44 | 563.9 | 0.55 | 0.769 | 524 | cpu6 41 | 2227/4474/4474 | 4224 | 15442 | 1/0/10 | **0** | 1 |
| b3 | R3 | 380 | 40.58 | 508.9 | 1.55 | 2.160 | 2421 | cpu7 12 | 2227/3283/3283 | 3878 | 28309 | 6/20/44 | **61** | 15613 |
| b3 | R4 | 389 | 37.32 | 512.0 | **2.72** | 3.788 | 4667 | cpu1 7 | 2285/3514/3514 | 3168 | 65185 | 85/**92**/33 | **310** | 79271 |

n/s = not sampled on that boot (counter added before b3). misses 121 /
1.89 a token and rounds 87 on all twelve; `pswpout` in the decode window
0 except b1 U60 (19 924 pages). Model file resident before the app
started: 3393–4116 MiB, no relation to the miss cost (b1 U60 4116 slow,
b3 R2 3529 fast).

## Reading

1. **No busy core during the slow misses (hypothesis refuted).** In 11 of
   12 decode windows the busiest non-app core is 3–16 % busy, slow and
   fast alike; the exceptions are b2 U60 (cpu0/1/4/5 26–30 %, slow) and
   b3 R2 (cpu6 41 %, the **fastest** run). Outside the runs the phone sits
   at 0.3 non-app cores from uptime ≈ 30 s on (timelines in `report.md`);
   the post-boot burst (6–7 cores busy) is over before uptime 30 s, i.e.
   before any run of this or the #201 S3 sittings. In the `top -H` frames
   overlapping the windows the one large non-app thread is `kswapd0`
   (10–86 %); next are `dmabuf-deferred-free-worker` 15 % (b1 U60) and
   ≤ 5 % app threads.
2. **The slow misses are storage reads.** b3's counters say it directly:
   ms/miss 5.03 / 2.72 / 1.55 / 0.55 against **610 / 310 / 61 / 0 MiB**
   read from block devices (`pgpgin`) and 151 527 / 79 271 / 15 613 / 1
   file-page refaults inside the decode window; 121 misses × ≈ 5.4 MiB
   ≈ 650 MiB, so R1 read nearly every missed expert from UFS. On all
   three boots the slow windows carry **92–296 ms of PSI io stall** (fast:
   0–31) and ≥ 65 k kswapd scans/s (fast: ≤ 28 k). The model file's pages
   are evicted by kswapd **during the run** (the 3.3 GB ION arena, the
   4.1 GB file and the rest of Android share 11.1 GB), so the page cache
   check before the run (#201 S3 reading 3, `201-fsu-e2e.md`) did not see
   it.
3. **Not boot proximity.** b1 U300 (3.72), all of b2 (4.0–5.3 up to
   uptime 301 s) and b3 R1 after a 6-minute idle (5.03) are slow; b3 R2,
   8 s after R1, is the fastest of the sitting. The #201 S3 sittings'
   "slow first ≈ 2 min" was a correlation of that sitting's memory state,
   not a property of uptime. **Rule 61's cause is wrong; its protocol
   ("wait ≥ 5 min / warm up") does not remove the slow regime** (b3 R1).
4. **DVFS follows, it does not lead:** cpu6/7 average 3.3–3.8 GHz in slow
   windows against 4.47 in the fast ones (read as the read threads
   blocking in IO, not a cause — a 25 % clock gap cannot make a 5–10×
   slower miss); DDR 2.6–4.2 GHz with no split.

## Step 2 not run (the lever does not address the cause)

The plan's lever (the server's slices off the 8-core ThreadManager
barrier, `NNTR_MOE_MISS_SLICES` / `_CPUS`) targets a busy core stalling the
barrier; the measured cause is the file read going to storage, which the
slice shape does not change. Per the task's stop rule no app was built and
no device A/B was run. The code was written and type-checked
(`tools/htp_syntax_check.sh` exit 0; `run_inproc_e2e.sh` with the knobs
unset `INPROC E2E PASS`, the four pool lines `bit_identical=1`), not
committed; the patch is kept at
`/local/mnt/workspace/htp_moe/216/lever-not-built.patch` (270 lines).
Skel: unchanged (no build).

## What this changes

* LEDGER ㉜: cause named — page cache eviction of the model file during
  decode under memory pressure (pgpgin / refault / PSI io); reader
  placement ✗, arena size ✗ (as before), busy core ✗, boot proximity ✗
  (correlate only). Rule 61 to be rewritten: a pool-28 miss cell is only
  readable with the decode window's `pgpgin` (or PSI io) next to it; a
  5-minute wait does not make it fast.
* The fix is a new plan (`state:needs-plan`). Candidates for the planner,
  none measured: keep the non-resident experts' file pages from being
  reclaimed (an `mlock` / `MADV_WILLNEED`-style hold on the expert byte
  ranges, within the memory budget), a smaller arena that leaves the file
  room (C = 24's 0.50–0.73 in #201 S3 may be this effect), or reading the
  miss with `O_DIRECT` so its cost is a stable UFS read rather than a
  bimodal cache hit / miss.

## Not verified here

* `pgpgin` / refaults for b1 and b2 (counter added before b3); their link
  to storage is through PSI io and kswapd only.
* Whether holding the file pages resident removes the slow regime (no
  run with such a hold).
* The plan's slice shapes on silicon (`SLICES=8/2/1`), the prefill gate,
  any G ≥ 512 cell.
* Which process takes the memory each time (lmkd reaper threads appear in
  b3 R4's frames; no per-process RSS was sampled).
