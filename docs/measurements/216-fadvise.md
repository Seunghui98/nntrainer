# Measurement 216 (lever): keep the model file's page cache to the arena's complement — `NNTR_MOE_FADVISE`

Plan `docs/plans/216-miss-read-slices.md` revision 2, §4 steps 0–4. Two
sittings on `R3CY10WM83Y` (S25, SM-S938N, kernel 6.6.77-android15),
2026-10-01, run by the implementer directly (agents run adb since
2026-09-30). Branch `htp/216-fadvise` @ `222a3196c` (the measured code; `b5a5cc281` adds the fd `dup` below), one binary set
for every variant, staged into a new device dir `s216b` (`s201s3`, `s216a`
untouched). Q28 one PD (`NNTR_HTP_E2E=1 NNTR_HTP_E2E_PDS=1
NNTR_MOE_CACHE_EXPERTS=28`), prompt 512, `NNTR_NUM_THREADS=8`, model file
pre-read (`cat` + `page_cache_evict -1`, #201 S3's "warm") before every Q
run. **Text == sitting 1's `A_G64_r1` / `A_G512_r1` (strip + cmp) on 26 /
26 runs**, `calls/token=1.00` and close clean on every Q run, ceiling 3840
after 25 / 26 runs (one 3584, below).

Logs: `/local/mnt/workspace/htp_moe/216/fadvise/logs/{fresh,old}/`
(`sitting.out`, per-run `.log`, `core.samples`, `md5_device.log`); the
design iterations' smoke logs and the two probes in `.../fadvise/probes/`.

| file | md5 |
|---|---|
| `libnntrainer.so` (the change) | `61a2658023773e64690a4fd0aea14cc3` |
| `libnntr_hvx_skel.so` (v79, rebuilt from `htp_decode` @ `03811d8ef`: the DSP sources moved since sitting 1's `a2ebef9c9` with #209 / #210, not with this change) | `9d61aef487d12b0ea11a126513787087` |
| `libcausallm_core.so` | `b72067c7510de7cab10ea9671cd66a71` |
| `nntrainer_causallm` (= sitting 1's) | `c106135d7074b3bd1129a17e0c57cb01` |
| `libccapi-nntrainer.so` / `libc++_shared.so` / `libsdkl.so` | `ad46760c…` / `b1586b9b…` / `0ad4e22a…` |
| `unittest_hvx_two_sessions` / `page_cache_evict` (sitting 1's; IDL unchanged) | `172f55ae…` / `42595651…` |
| `216-fadvise-run.sh` | `f4c8483ed06ebc4e1a81ecc471cba63a` |
| `216-fadvise-sampler.sh` (old boot; the fresh boot ran it before the `R`-line fix, so its `resident in window` is empty) | `da71de7654e7954ce28523d67df4d39c` |
| `216-fadvise-report.py` | `15b658d7b14c825e8646620ea59ba031` |

`md5_device.log` == the staged `md5.txt` on both boots (`MD5 OK`); the
app set rebuilt from the committed source reproduced the same
`libnntrainer.so` md5 between the boots.

After the sittings the review asked for each queued advice to hold its
own `dup` of the fd (a model closed before the worker drains its queue
must not send the advice to a file that reuses the number): that commit
changes `libnntrainer.so` to `a38488a24263ef2f819e82d1c034fbeb` and
nothing else. Re-checked on the old boot's phone (uptime 1186–1208 s, two
A / B pairs, G = 64, `probes/final_*.log`): A 37.49 / 34.13 tok/s at
3.78 / 5.45 ms/round, B 43.27 / 43.21 at 1.15 / 1.10, `pgpgin_mib=` 639.5,
resident after 1 499, text == `A_G64_r1` 4 / 4, `timeouts=0/0`, ceiling
3840 after. The tables below are the `61a26580` binary.

## Step 0: device facts (no build)

| item | value |
|---|---|
| MemTotal | 11 381 316 kB (11 114 MiB) |
| SwapTotal / SwapFree at boot | 4 194 300 kB (zram0, 4 GiB) / 4 194 300 kB |
| `ulimit -l` (shell user) | **64 KiB** — lever (b), `mlock` of the complement (~465 MiB), is not available without root |
| MemAvailable at idle (sampler, uptime 16 s and 55 / 590 s, before any run) | 7 118–7 360 MiB; Cached 3 966–5 060 MiB |
| `/proc/sys/vm/swappiness` | not readable (permission denied) |
| `read_ahead_kb` of the data volume | not readable; measured instead (below) |

## What the code does (and where the plan's assumptions did not hold)

`NNTR_MOE_FADVISE=1`: an expert's two weight ranges (`[off, off +
whBytes(K, N) + 8 N)`) are dropped (`POSIX_FADV_DONTNEED`) once its bytes
are in the arena slot, and asked back (`POSIX_FADV_WILLNEED`) when it
leaves the arena; `=2` drops only. Unset: no call, no thread; bytes, slots
and reads unchanged (host pool lines below). The pool line gains
`pgpgin_mib=`, the driver-on line `fadvise=`.

Measured on the phone with two throwaway probes (`probes/willneed_probe.c`,
`probes/drop_probe.c`, 5.4 MiB ranges of the model file):

1. **One WILLNEED reads at most the read-ahead window of its range**: 1 MiB
   of 5.4 MiB on the S25 (128 KiB on the workstation; the kernel clamps
   `force_page_cache_ra` to the device's window). The plan's single call
   per range left ~4.3 MiB of every victim on storage: smoke B read 685
   MiB in the decode window at 8.5 ms/round. → issued in 128 KiB pieces.
2. **A WILLNEED is not asynchronous here**: 3–8 ms of caller time per
   expert (the pieces are submitted and largely read before it returns);
   **a DONTNEED costs 2–10 ms** per expert (the plan assumed ≈ 0.1 ms).
   Inline they put 1.4 ms/round on the miss path. → one worker thread.
3. A thread per call inherited the pool server's core-6 pin and stretched
   S1's miss wait (2.1 ms/token at arm 0.65 ms/round); one unpinned worker
   took it to 0.9–1.0. A drop of a just-read decode miss on the worker,
   beside the next miss reads, still cost 0.4–0.7 ms/round (A/B/no-drop
   smoke, `smoke6.log`: 1.4–1.9 vs 1.0–1.1) → **a decode miss's pages are
   dropped at the next `poolSync`** (once per prefill; ponytail: until
   then they stay cached, ≤ misses × 5.3 MiB a generation, 640 MiB at
   G = 64 here).

Variants tried in smoke and not kept (each A-interleaved, same boot,
`probes/smoke*.log`): WILLNEED of prefill victims moved to `poolSync` (the
plan's prefill fallback) — prefill unchanged (−6..−10 %), decode worse
(arm 2.6–2.8 ms/round: the 88 asks race token 0's misses); load drops
synchronous on the loader — prefill unchanged, load +0.6 s; drops of the
prefill loads deferred to `poolSync` — decode arm 2.2–2.7; worker at nice
10 / −5 / `SCHED_IDLE` — no change outside run-to-run noise.

## A/B (both boots, the committed code)

**A** = knob unset, **B** = `NNTR_MOE_FADVISE=1`, **D** = `=2`, **H0 / H** =
hybrid A (nothing set) without / with `=1`. ms/miss: the profiled `file
read … ms/miss` on `prof=2` runs, else `arm_ms/round × rounds / misses`
(`*`, the same quantity: the server's time in its answers over the
misses). `app pgpgin` = the new `pgpgin_mib=` field (driver on → close,
system-wide); `win` columns = the sampler over the decode window
(0.5 s samples, so the window's first interval holds the end of the
prefill: its pgpgin includes the prefill's WILLNEEDs and A's `cat`).

**Fresh boot** (reboot 23:06 KST, first run at uptime 60 s):

| run | cell | up s | prefill tok/s | decode tok/s | ms/miss | arm_ms/round | miss_wait us/tok | app pgpgin MiB | win pgpgin MiB | win refault | win PSI io ms | resident after MiB | text |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | A G=64 | 60 | 541.2 | 33.39 | 4.38* | 6.092 | 7808 | 647.1 | 648 | 165838 | 239 | 1347 | same |
| B1 | B G=64 | 72 | 437.6 | 41.53 | 1.07* | 1.491 | 1497 | 643.0 | 2077 | 261035 | 839 | 1252 | same |
| A2 | A G=64 | 83 | 496.1 | 33.70 | 4.01* | 5.582 | 7111 | 636.6 | 796 | 202525 | 875 | 2813 | same |
| B2 | B G=64 | 93 | 451.9 | 41.61 | 1.08* | 1.499 | 1538 | 639.6 | 1114 | 2630 | 11 | 1499 | same |
| A3 | A G=64 | 101 | 461.7 | 39.07 | 1.71* | 2.381 | 2727 | 203.1 | 666 | 169077 | 1030 | 2796 | same |
| B3 | B G=64 | 111 | 447.6 | 41.83 | 1.03* | 1.432 | 1423 | 639.5 | 1185 | 18790 | 55 | 1499 | same |
| A4 | A G=64 prof | 120 | 480.3 | 38.91 | 1.89 | 2.631 | 3093 | 200.8 | 748 | 190099 | 1093 | 2756 | same |
| B4 | B G=64 prof | 130 | 437.2 | 41.26 | **1.13** | 1.569 | 1605 | 639.5 | 1157 | 29812 | 69 | 1499 | same |
| D | D G=64 prof | 139 | 458.4 | 33.72 | 3.87 | 5.382 | 6837 | 642.4 | 1064 | 111991 | 1241 | 1037 | same |
| A512 | A G=512 prof | 149 | 476.7 | 45.77 | 2.73 | 3.409 | 830 | 500.1 | 882 | 225466 | 1115 | 2741 | same |
| B512 | B G=512 prof | 169 | 391.7 | 45.52 | 1.07 | 1.340 | 248 | 925.2 | 1392 | 467 | 10 | 1785 | same |
| H0 | hybrid | 188 | 447.2 | 50.67 | – | – | – | – | – | – | – | 1978 | same |
| H | hybrid, =1 | 202 | 446.4 | 50.96 | – | – | – | – | – | – | – | 2262 | same |

**Old boot** (reboot 23:24 KST, idle to uptime 600 s, sampler on):

| run | cell | up s | prefill tok/s | decode tok/s | ms/miss | arm_ms/round | miss_wait us/tok | app pgpgin MiB | win pgpgin MiB | win refault | win PSI io ms | resident in win (min) | resident after MiB | text |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | A G=64 | 601 | 557.7 | 35.63 | 3.51* | 4.881 | 6161 | 517.9 | 486 | 124229 | 169 | 949 | 951 | same |
| B1 | B G=64 | 611 | 535.0 | 43.33 | 0.86* | 1.202 | 1111 | 639.5 | 1660 | 143214 | 893 | 860 | 1499 | same |
| A2 | A G=64 | 620 | 543.5 | 40.48 | 1.69* | 2.347 | 2696 | 200.7 | 563 | 143514 | 638 | 2059 | 2053 | same |
| B2 | B G=64 | 628 | 504.4 | 42.72 | 0.96* | 1.342 | 1296 | 639.5 | 1391 | 78772 | 638 | 860 | 1499 | same |
| A3 | A G=64 | 637 | 555.3 | 43.66 | 0.68* | 0.945 | 765 | 0.3 | 451 | 92178 | 944 | 3358 | 3358 | same |
| B3 | B G=64 | 647 | 465.9 | 41.97 | **1.13*** | 1.577 | 1621 | 639.7 | 1096 | 280 | 7 | 860 | 1499 | same |
| A4 | A G=64 prof | 654 | 505.4 | 34.90 | 3.51 | 4.876 | 6151 | 502.0 | 636 | 133937 | 154 | 3393 | 3380 | same |
| B4 | B G=64 prof | 662 | 495.6 | 42.81 | 0.92 | 1.285 | 1231 | 639.5 | 1094 | 18 | 5 | 1018 | 1499 | same |
| D | D G=64 prof | 669 | 496.6 | 34.15 | 3.97 | 5.516 | 7023 | 643.3 | 1100 | 94752 | 404 | 822 | 1037 | same |
| A512 | A G=512 prof | 678 | 502.9 | 46.06 | 3.87 | 4.836 | 1221 | 775.2 | 915 | 234038 | 763 | 3196 | 3196 | same |
| B512 | B G=512 prof | 696 | 442.9 | 48.05 | **1.11** | 1.383 | 279 | 925.0 | 1407 | 4480 | 10 | 860 | 1785 | same |
| H0 | hybrid | 714 | 460.4 | 52.24 | – | – | – | – | – | – | – | – | 2292 | same |
| H | hybrid, =1 | 726 | 483.5 | 49.46 | – | – | – | – | – | – | – | – | 2820 | same |

Block means (G = 64, four runs each; tok/s read only inside a block):

| boot | prefill A → B | decode A → B | ms/miss A (range) | ms/miss B (range) |
|---|---|---|---|---|
| fresh | 494.8 → 443.6 (**−10.4 %**) | 36.27 → 41.56 (**+14.6 %**) | 1.71–4.38 | 1.03–1.13 |
| old | 540.5 → 500.2 (**−7.4 %**) | 38.67 → 42.71 (**+10.5 %**) | 0.68–3.51 | 0.86–1.13 |
| G = 512 (one each) | fresh −17.8 %, old −11.9 % | fresh −0.5 %, old +4.3 % | 2.73 / 3.87 | 1.07 / 1.11 |

## Gate (plan §1)

| gate | result |
|---|---|
| 1. ≤ 1.1 ms/miss in every B run | **7 / 10**: 0.86–1.08 on seven, 1.13 (fresh B4, old B3) and 1.11 (old B512) on three. B's spread 0.86–1.13 against A's 0.68–4.38: the bimodal slow regime is gone, a constant ~1 ms/miss is paid instead |
| 1b. `arm_ms/round` ≤ A's fast runs (0.77–0.94) | **fail**: B 1.20–1.58 (A's one fast run here 0.945). The worker's WILLNEED and drop of the previous rounds run beside the next miss reads |
| 2. window `pgpgin` ≤ 11 MiB | **fail by construction** — the plan's own §3 arithmetic: WILLNEED re-reads one expert per eviction, ≈ 10 MiB/token at G = 64; B's `pgpgin_mib=` is 639–643 at G = 64 (121 × 5.29 MiB) and 925 at G = 512 in every run, A's 0.3–775 depending on the regime. What the lever removes is the *synchronous* read under reclaim: B's window PSI io is 5–69 ms on 7 of 10 B windows (638–893 on fresh B1, old B1 / B2, not attributed: the window's first 0.5 s interval also holds the end of the prefill), A's 154–1115 on all ten; window refaults 18–29 812 on those seven B windows vs 92–234 k on every A window |
| 3. text == A, `calls/token=1.00`, close clean, ceiling 3840 | text 26 / 26, calls and close 22 / 22 Q runs; **ceiling 3584 once** (after the fresh boot's H, a hybrid run whose arena closed `unmapped 15/15`; not reproduced in five H / H0 runs after a reboot, all 3840; the hybrid passes neither `readExpert` nor `release_qs4cx_wh_expert`, so not read as this change's, but not explained) |
| 4. prefill ≥ −5 % of the block's A | **fail**: −7.4 % / −10.4 % at G = 64, −12 / −18 % at G = 512. Profiled (A vs `=2`, `probes/prof_*.log`): the M > 1 MoE calls are unchanged (380.1 vs 385.5 ms host), the ~110 ms goes to the CPU side of the prefill — the 2–10 ms drops (and WILLNEEDs) on the worker beside the 8-thread prefill |
| 5. hybrid unchanged | H vs H0: +0.6 % (fresh), −5.3 % (old); five alternating runs after a reboot: 53.0 / 54.3 / 53.0 with `=1` vs 54.6 / 53.3 without — inside the hybrid's run-to-run spread; the path is not touched (non-virtual experts never reach `readExpert`) |
| 6. host | `run_host_checks.sh` `ALL CHECKS PASS` + `WORKER POOL LANES OK`; `htp_syntax_check.sh` type-checks; `*Lfm2Moe*` 6 / 6; `run_inproc_e2e.sh` `INPROC E2E PASS` with the knob unset, `=1` and `=2`, its four pool lines `bit_identical=1` with the same `misses=` (5 / 56 / 14 / 14) under all three |

Host fixture under the knob (the runnable check this change leaves
behind; the fixture is file-backed, so the advice acts there too):
`=2` puts the misses on storage (`arm_ms/round` 0.99–8.66, `pgpgin_mib`
8.0–9.5) where unset reads them from the cache (0.01–0.19, 0.0) and `=1`
asks them back (0.01–0.21, 3.0–4.6 = the WILLNEEDs).

## Reading

1. **The model is confirmed.** D (drop only) is slow on both boots (3.87 /
   3.97 ms/miss, `pgpgin` 642 MiB = every miss from storage), as predicted
   in plan §3: every decode miss is a re-miss of an expert the arena once
   held. With the victims asked back (B) the misses are cache hits — the
   file's resident set falls to the complement (860 MiB in the window,
   1 499 after a G = 64 run with the decode loads still cached) and B's
   windows show little reclaim (kswapd 0–14.8 k scans/s vs A's 31–109 k).
2. **The lever trades the bimodal miss for a constant one.** A's ms/miss
   ranged 0.68–4.38 within one boot (rule 61's regime, both boots); B's
   0.86–1.13 on both. At G = 64 that is +10 % / +15 % decode on the block
   means; at G = 512 (0.34 misses/token) −0.5 % / +4.3 %. Against A's one
   fast run (A3 old, 0.68 ms/miss, 43.66 tok/s) B's runs are 1–4 % slower
   at decode (41.97–43.33).
3. **Its price is the advice's own cost on this kernel**: WILLNEED and
   DONTNEED are 2–10 ms CPU calls per expert, not hints. On the miss path
   that is +0.3–0.6 ms/round over A's fast reads; beside the prefill it is
   −7 to −18 %. Every placement tried (inline, per-call threads, one
   worker at any priority, deferred to the decode start) moves the cost,
   none removed it.
4. **Not flipped.** Per the plan the default stays unset; the user decides
   at review. The prefill gate fails, so the flip would trade ~10 % prefill
   for ~10–15 % decode at G = 64 and parity at G = 512.

## Not verified here

* G = 1024 (the plan's design has G = 64 ×4 and G = 512 ×1 per boot).
* The race the plan predicted (a victim re-missed within one UFS read): no
  per-round read split; B's worst run (1.13) cannot be attributed to it.
* Whether a smaller drop cost exists on this kernel (e.g. `MADV_PAGEOUT`
  / `MADV_COLD` on a mapping, or reading the victim with a plain `pread`
  into a scratch buffer instead of WILLNEED): not tried.
* The host check on the real file (plan step 1's registration-only run):
  the inproc host build capped the arena at 512 MiB, so the preload drop
  of 616 experts was not exercised there; the device's `resident after`
  column is the measurement of the mechanics.
* `docs/measurements/216-remiss-sim.py` (the planner's replay script) is
  not in this branch: it lived in the planning session's scratchpad.
* The ceiling 3584 after the fresh boot's H run: cause unknown.
