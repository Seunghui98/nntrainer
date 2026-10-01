# Plan 216: the decode pool miss read — name the post-boot stall, then take the read off the 8-core barrier

Issue #216 (p1, part of #76 / #201). Base `htp_decode` @ `36dbc21f4`; every
`path:line` below is that tree (PR #215, open, shifts the cited
`htp_compute_ops.cpp` lines but changes none of them — see §2). Source
measurements: `docs/measurements/201-s3-probe.md`, LEDGER rule 61 / ㉜.

## 1. Goal and gate

From the issue, made measurable:

1. **A table uptime × per-core busy % × ms/miss** over a boot's first 5 min,
   one unit, two reboots plus one idle-first control boot, that either
   names the busy core(s) during slow misses or shows none. Posted on
   #216, folded into `201-s3-probe.md` "Not verified here" and ㉜.
2. **The winning miss-read shape, if any:** `[HTP-PROFILE] expert misses …
   (x ms/miss)` (`htp_compute_ops.cpp:859–869`) **≤ 1.1 ms inside a fresh
   boot's first 2 min** on Q28 G = 64 `NNTR_HTP_PROFILE=2`, and
   `arm_ms/round` / `miss_wait_us/token` (`:4173–4182`) not above the
   current shape's on an old boot, same sitting, interleaved. On every run:
   `calls/token=1.00`, `pds=1`, close clean, ceiling 3840 MiB, uptime on
   the run line (rule 61).
3. **Text == A** on every run (`strip` + `cmp` against the sitting's
   `A_G64_r1`, as `201-s3disc-run.sh` does). Bit identity is untouched by
   construction: the bytes land in the same arena slot in the same order;
   the PR states it and the host `run_inproc_e2e.sh` pool line
   (`test/htp/host/run_inproc_e2e.sh:352`, `NNTR_HTP_E2E_PDS=1
   NNTR_MOE_CACHE_EXPERTS=2`) prints `bit_identical=1` under every shape.
4. **Prefill ≥ −5 % of the same sitting's A** on prefill tok/s and the
   M > 1 MoE `dsp` ms/call (the pinned worker pool is prefill's).
5. Host: `run_host_checks.sh` `ALL CHECKS PASS` + `WORKER POOL LANES OK`,
   `tools/htp_syntax_check.sh` exit 0, `run_inproc_e2e.sh` `INPROC E2E
   PASS`, `clang-format-14` on changed lines.

Verdict-only close (step 1 names nothing and step 2 moves nothing) is an
allowed end: rule 61 stays the fix.

## 2. Where it lives (verified on `origin/htp_decode` @ `36dbc21f4`)

**The miss read.**

* `nntrainer/tensor/htp_backend/htp_compute_ops.cpp:5584–5640`
  `readWeight(fd, off, K, N, arena_dst, e, use_pool)`: with `use_pool`,
  `n_slices = min(ThreadManager::getComputeThreadCount(), kExpertReadSlicesMax)`
  (`:5603–5604`; `kExpertReadSlicesMax = 8` at `:6214`) = **8** under
  `NNTR_NUM_THREADS=8`; slice = ceil(nib / 8) rounded up to 4 KiB
  (`:5605–5606`); `tm.parallel_for(0, n_slices, …)` (`:5607–5611`), each
  slice a `preadAll` (`:5278–5295`) into the uncached ION arena; then the
  caller alone `pread`s the scales / column sums tail (`:5617–5620`).
  `readExpert` (`:5387–5399`) calls it twice (gate_up, then down) — **two
  barriers per missed expert pair** (≈ 5.4 MiB).
* Callers of `use_pool=true`: the pool server's `poolAnswer`
  (`:2299–2375`, `readExpert` at `:2347`, under `handle_mutex_`), the
  prefill-side `register_qs4cx_wh_expert_file[s]` (`:2717`, `:2746`, main
  thread). `use_pool=false`: the prefetch readers (`prefetchReaderLoop`
  `:5549–5572`, `readExpert` at `:5563`) — one expert per thread, prefill
  only (`Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.cpp:912–915`
  gates the depth on `total_tokens > 1`).

**Who runs the slices.** `nntrainer/utils/thread_manager.h:170–181`
`parallel_for` → `parallelize` (`:262–315`): takes `execution_mutex_`,
splits `[0, 8)` over `threads_count = workers + 1 = 8` — one index each;
**tid 0 is the calling thread** (`thread_parallelize(0)` at `:311`), tids
1–7 the workers; the join is `wait_worker_threads()` (`:313`,
`thread_manager.cpp:176–`: `SPIN_COUNT = 1e6` yields, then futex) and it
**waits for all 7 workers' check-in whatever their range** — a 2-slice
`parallel_for` would still be an 8-thread barrier. Workers wait for work in
`wait_for_new_command` (`thread_manager.cpp:145–174`, 1e6 yields then
`futex_wait`). Pinning: `ThreadManager::initialize`
(`thread_manager.cpp:71–124`) — `core_map = getCoresByPerformance()`
(`thread_manager_util.cpp:97–175`, cores sorted by `cpuinfo_max_freq`
descending), **main pinned to `core_map[0]`** (`:100–101`), worker *i* to
`core_map[i]` (`:115–121`, `pinSelfToCore` `thread_manager_util.cpp:191`).
On the S25 that is main → 6, workers → 7, 0, 1, 2, 3, 4, 5
(`201/s3/logs_disc/D_1.threads`: tids 13451–13457 `Cpus_allowed_list`
7, 0–5; main 13353 `6`). No env knob turns the affinity off
(`ThreadManagerConfig::enable_affinity = true`, `thread_manager.h:88`, no
`setConfig` caller in the tree).

**The pool server.** `poolArm` (`:2425–2434`) creates
`std::thread(poolServe)` from main **after** `initialize()` pinned main to
core 6, so the server inherits `Cpus_allowed_list=6` (`D_1.threads` tid
14204). `poolServe` (`:2385–2420`) yield-spins on core 6 for the token;
`poolAnswer`'s `readExpert` therefore runs slice 0 on core 6 and slices 1–7
on the pinned workers: **every miss is a barrier over all 8 cores**, twice
per expert pair. Meanwhile main is blocked in the dspqueue read
(`:4337–4347`, `q2.api->read … kDspqTimeoutUs`), one packet a token
(`calls/token=1.00`), so no other `parallel_for` runs in decode — the
workers do nothing else during a token. In prefill the same 7 workers are
every CPU op's `parallel_for` and the prefill misses' slices (`:2746`,
from main); hence the prefill gate.

**Accounting.** `arm_ms/round` = `pool_read_us / pool_rounds`
(`:2368–2370` `read_us` = whole `poolAnswer`: harvest + policy + reads;
printed `:4173–4182`); the `[HTP-PROFILE] expert misses … ms/miss` line is
the same `read_us` through `addExpertLoad` (`:2371–2372`, print
`:859–869`), so "ms/miss" = answer time / loads. `miss_wait_us/token` is
the **DSP's** wait from posting the request to seeing the answer word
(`hmx/hexkl_token.c:227–254` `tk_miss_wait`, `st.miss_us`; returned per
token `test/htp/nntr_hvx_token.c:161`, summed `:4416–4417`); the gap
`miss_wait − rounds × arm_ms` is the server's poll latency.

**PR #215 (`htp/211-one-pd-only`)** touches none of these bodies: its
`htp_compute_ops.cpp` hunks nearest to them are the `poolServe` /
`poolArm` comments (`-2384,7` / `-2422,8`), the close print (`-4180,65`)
and `E2eState` (`-6375,42`). After it, `poolAnswer` sits ≈ 14 lines
earlier and `readWeight` / `kExpertReadSlicesMax` ≈ 148 lines earlier.
Rebase cost: zero conflicts expected; re-run the host gates once on top.

**Consumers that do not move.** No IDL / stub / skel change (the DSP side
is untouched; `libnntr_hvx_skel.so` md5 stays `31c0a033…` of the staged
set), no quantizer tag, no loader check, no new `NNTR_HTP_PROFILE` stage
(the existing `arm_ms/round` and `ms/miss` lines are the readout),
`tools/htp_fc_report.py` unaffected.

## 3. Design

**Measure first (the issue's order).** Step 1 is a sampler beside the runs,
no app change: per-core `/proc/stat` deltas at 0.5 s, `/proc/loadavg`,
`scaling_cur_freq` per core, `pgscan_kswapd` / `pswpout` from
`/proc/vmstat`, and every 5 s the second frame of
`top -b -H -d 1 -n 2 -m 12` (names the thread on a busy core). The
existing `.threads` sampler covered only the app; the system sample is what
is missing. A third boot that idles 5 min before its first run is the
control that separates "uptime" from "the first N runs of a process".

**The lever, one mechanism, three shapes.** Take the pool server's reads
off `ThreadManager`: `readExpert(st, ReadBy)` replaces the `bool use_pool`
with an enum `{kSingle, kWorkers, kServer}`; `kServer` runs the slices on
the server thread plus `k` **helper threads owned by `PoolServer`**
(created in `poolArm` next to the server; affinity set explicitly with
`pinToCpus`, `:6279–6290`, because a thread created there would otherwise
inherit core 6; they yield-spin while `p.active`, condvar-wait otherwise,
like the server — a `ponytail:` comment names the spin as the ceiling).
Two env knobs, parsed like `prefetchKnobs()` (`:6246–6268`):

* `NNTR_MOE_MISS_SLICES=n` — unset → today's `kWorkers` path, byte for
  byte; `1` → the server reads alone (lever c, no barrier at all); `n ≥ 2`
  → server + `n − 1` helpers.
* `NNTR_MOE_MISS_CPUS=list` — the helpers' affinity; unset = all cores
  (lever b, unpinned); `7` → lever a (server on 6 + one helper on 7).

The barrier shrinks from 8 pinned threads to `n` threads that the scheduler
may move; the bytes, the slot and the order are unchanged. Prefill's
misses (`:2746`) keep `kWorkers` (main calls them between CPU ops), so
prefill is untouched by construction. If the A/B names a winner, the PR
flips only the unset value (as #115 / #151 did: bit-identical, prefill
gate, text == A), keeps the knobs, and `201-s3probe-run.sh`'s successor
records the shape on every run line.

**Ranked levers** (gain inside the boot window / risk / code):

| rank | lever | expected gain | risk | code |
|---|---|---|---|---|
| 1 | (b) `SLICES=8`, helpers unpinned | one busy core no longer stalls the read: the kernel places the helper elsewhere; fast-regime rate kept (8 preads, 0.5–1.1 ms) | condvar / spin wake of 7 helpers per barrier (~0.05–0.15 ms if they slept); EAS may stack two helpers on one core | the `PoolServer` helper set, ≈ 50 lines |
| 2 | (a) `SLICES=2 CPUS=7` | barrier over the two prime cores only, where post-boot services are least likely (the busy ones land on 0–5 if EAS behaves) | 2 preads ≈ 1.2 ms/miss by doc 52 §10.9's curve (1 thread 1.37 → 8 threads 1.12 on 3.5 MiB) — may miss the 1.1 gate; core 7 may be the busy one | same mechanism, no extra code |
| 3 | (c) `SLICES=1` | no barrier, no other core; the cleanest **diagnostic**: fast inside the window ⇒ the barrier is the cause | +0.3–0.6 ms/miss on an old boot (≈ −2–4 % decode at G = 64, ≈ 0 at G ≥ 512) — fails "not slower" at G = 64 | 3 lines |
| 4 | (d) read-ahead from the previous token's routing (plan 201 P-D) | hides the miss (ceiling 64 %, doc 53 §7) | needs a predictor (the ARM learns a token's routes only in `poolRefresh`, `:2456`, after the token); wasted reads cost DDR and slots; it hides the slow regime rather than curing it, and 8 pinned cores would still burn | a new design; not this issue |

**Ship first: (b)**, the unpinned helper set — it is the only shape that
promises both halves of the gate (≤ 1.1 inside the window, parity on an
old boot), and (a) / (c) are the same code with other knob values, so the
A/B costs one binary. (c) runs in the same sitting as the diagnostic; (d)
stays on #201.

**Rejected alternative: fewer slices through `ThreadManager`.** A 2-slice
`parallel_for` still wakes and joins all 7 workers (`thread_manager.h:272–313`),
so the 8-core barrier remains; and reusing the prefetch readers' job
queue (`:5549–5572`) was rejected because they exist only after the first
prefill `_begin`, vanish under `NNTR_MOE_PREFETCH=0`, are pinned to
all-but-caller, and their one-expert-per-thread job would need a closure
queue — more change than four private helpers.

## 4. Steps

Each step ends in a rung of `.claude/skills/hexagon-gates`.

1. **Sampler and fold (no app change).** `docs/measurements/216-core-load-run.sh`
   (from `201-s3disc-run.sh`: md5 check, `cat` + `page_cache_evict`,
   Q28 G = 64 `NNTR_HTP_PROFILE=2`, `strip` + `cmp` text, ceiling after
   every run, **uptime stamped before and after each run** and the
   `generation: … ms` line locating the decode window) plus the system
   sampler above writing `core.samples`; `docs/measurements/216-core-load-report.py`
   folds `core.samples` + the run logs into the table (per run: uptime,
   ms/miss, `arm_ms/round`, `miss_wait`, busy % per core over the decode
   window with the app's own threads' ticks subtracted from the `.threads`
   deltas, top thread names). Gate: the script runs on the host against
   the existing `201/s3/logs_disc/*` files (a synthetic `core.samples`
   from the workstation's `/proc/stat`) and prints the table.
   *No gate rung needed — docs/measurements only.*
2. **Device, measurement (unavoidable).** Unit `R3CY10WM83Y`, sitting 1's
   build (`a2ebef9c9`, md5s as in `201-s3-probe.md`); nothing new to push.
   Boot 1 and 2: sampler from uptime 60 s to 330 s, runs at uptime ≈ 60,
   120, 180, 300 s. Boot 3 (control): sampler on, phone idle until uptime
   360 s, then the same four runs back to back. Readout: slow misses
   coincide with a busy core (name it) / with a cpufreq dip / with
   neither; the control says whether an idle 5 min alone ends the regime.
   Post the table on #216. **If no core is ever busy and `SLICES=1` in
   step 5 is not faster in the window either, stop at a verdict-only close.**
3. **Code: `ReadBy` and the helper set** (`htp_compute_ops.cpp` only:
   `readExpert` / `readWeight` signature, `poolAnswer` → `kServer`,
   `PoolServer` helpers + two knobs, the `[HTP] token driver: pool …`
   line gains `read=<shape>`). Gate rung 0 + 1: `clang-format-14`;
   `ninja -C build`; `run_host_checks.sh` (`ALL CHECKS PASS`, `WORKER
   POOL LANES OK`); `tools/htp_syntax_check.sh`; `run_inproc_e2e.sh`
   `INPROC E2E PASS` — and its pool line (`:352`) re-run by hand under
   `NNTR_MOE_MISS_SLICES=1`, `=2 NNTR_MOE_MISS_CPUS=7`, `=8`, each
   `bit_identical=1` with the same `misses=`. That is the one runnable
   check the new code leaves behind (the host build has an 8-thread pool,
   so all three shapes exercise their branch).
4. **App build once** (rung 3, `--cache`). The skel is not rebuilt: its
   device md5 must stay `31c0a033…`; the app set's md5s go on the handoff
   table. Rebase onto `htp_decode` once #215 merges, re-run rung 1.
5. **Device, A/B (unavoidable).** Variants (≤ 4, one binary):
   **A** = knobs unset (today's 8 pinned slices), **B1** = `SLICES=8`
   unpinned, **B2** = `SLICES=2 CPUS=7`, **B3** = `SLICES=1`. Fresh boot:
   from uptime 60 s, interleave A B1 B2 B3 twice (8 runs ≈ 80 s, all
   inside the window; the sampler from step 1 on). Old boot (≥ 5 min or
   the same boot after 300 s): A B1 B2 B3 × 3 interleaved. Prompt 512,
   G = 64 Q28 throughout (the pool's miss rate is highest there, 1.89 a
   token); one A and one winner G = 512 run on the old boot for the
   G ≥ 512 column. Prefill gate from the old-boot block: prefill tok/s and
   M > 1 `dsp` of the winner against A. Text == A on all runs; uptime on
   every line.
6. **PR into `htp_decode`** (`htp/216-miss-read-slices`): the winner as
   the unset value, knobs kept, the diagnostic cells and the step-1 table
   in `docs/measurements/216-miss-read.md`, LEDGER / BENCHMARK rows below.
   Or the verdict-only close with the table folded into `201-s3-probe.md`.

## 5. Risks

* **Sampler perturbation.** The adb-side sampler is itself CPU load on an
  unpinned shell; 0.5 s `/proc/stat` is negligible, the 5 s `top` frame is
  not — its own PID is reported in the table so a "busy core" that is the
  sampler is visible, and boot 2 can run with the `top` frames off.
* **Time alignment.** The app prints no timestamps; the decode window is
  reconstructed as `[run end − generation ms, run end]` from the runner's
  stamps. Misalignment by a second smears but does not invent a busy
  core; the table carries both the window and the whole-run busy %.
* **Thermal / DVFS between sittings.** The probe started at zone0 35.9 °C
  sixty seconds after boot (the boot itself heats the unit); fresh-boot
  and old-boot blocks differ in temperature as well as uptime. The
  per-core `scaling_cur_freq` column and the interleaving inside each
  block keep this visible; the gate reads shapes against A within one
  block, never across blocks.
* **Unpinned helpers on a big.LITTLE scheduler.** EAS may place two
  helpers on one core or on a cold core; the `read=<shape>` run line plus
  the `.threads` sampler (helpers' `cpu` field) show where they ran.
* **Stale skel / wrong set.** No skel change, so a stale skel cannot
  appear as a win; the md5 table on every run guards the app set.
* **Address-space budget.** Unchanged: no new ION, no new slot, C = 28 /
  3.3 GB arena as before (ceiling 3840 after every run, as now).
* **The hypothesis is wrong.** If no core is busy and `SLICES=1` is as
  slow in the window as 8 slices, the stall is in the copy itself
  (page cache → uncached ION) and none of (a)–(c) helps — the plan ends in
  the verdict and the next read is the kernel side (`pread` vs `mmap` +
  `memcpy`, DMA-BUF cached mapping), filed as its own issue.

## 6. Docs to update

* `docs/measurements/201-s3-probe.md`: "Not verified here" → the step-1
  table and the control boot's reading; the step-5 cells if run.
* `docs/htp_moe/LEDGER.md`: ㉜ closed or rewritten with the named cause
  and the shape; rule 61 amended (either "the window no longer costs
  with `read=<shape>`" or "protocol stays"); #208's "≈ 4.7 ms a miss"
  note points at the result; a §2 verdict row for the miss-read shape.
* `docs/htp_moe/BENCHMARK.md`: a Method cycle note (profiled runs, no
  Results row — contract §1.1); if the default flips, the next sitting's A
  carries it as #115 / #151 did.
* `docs/plans/201-htp-decode-e2e-review-gemma-moe.md` P-D row: unchanged,
  referenced from the ranked table.
