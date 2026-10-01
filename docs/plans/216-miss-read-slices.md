# Plan 216 (revision 2): the slow decode pool miss is a UFS read — keep the page cache equal to the arena's complement

Issue #216 (p1, part of #76 / #201). Base `htp_decode` @ `f5f459587`; every
`path:line` below is that tree (PR #215, open, shifts `htp_compute_ops.cpp`
by ≈ 14 lines around `poolAnswer` and ≈ 148 around `readWeight`, bodies
unchanged). Source measurements: `docs/measurements/216-miss-read.md`
(PR #217), logs `/local/mnt/workspace/htp_moe/216/logs_core/{b1,b2,b3}/`,
`201-s3-probe.md`, LEDGER rules 59 b / 61, ㉜.

**Revision 1 refuted.** Revision 1 (commit `2c740606c`) read the slow miss
as an 8-core `parallel_for` barrier stalled by a busy post-boot core and
planned slice shapes (`NNTR_MOE_MISS_SLICES` / `_CPUS`). Step 1's sampler
found no busy core in any slow window (busiest non-app core 3–16 %), and
put the cost on storage instead: in b3 the decode window read **610 / 310 /
61 / 0 MiB** from block devices (`pgpgin`) for **5.03 / 2.72 / 1.55 /
0.55 ms/miss**, with PSI io 92–296 ms in every slow window (0–31 in the
fast ones) and kswapd ≥ 65 k scans/s. Not boot proximity (b2 slow at
uptime 301 s, b3 slow after a 6-min idle and fastest 8 s later). The
slice patch (`/local/mnt/workspace/htp_moe/216/lever-not-built.patch`) is
not used here: it addressed the wrong cause.

## 1. Goal and gate

Named cause: the model file's page cache is evicted by kswapd during the
run, so a pool miss `pread`s from UFS (≈ 4–5 ms for 5.3 MiB) instead of
from the cache (≈ 0.5 ms). The lever must remove the memory pressure that
makes the kernel evict the pages the pool will need. Gate, Q28 one PD
(`NNTR_HTP_E2E=1 NNTR_HTP_E2E_PDS=1 NNTR_MOE_CACHE_EXPERTS=28`), prompt
512, `NNTR_HTP_PROFILE=2`, unit `R3CY10WM83Y`:

1. **≤ 1.1 ms/miss in every run** of a 4-run A/B sequence (B's four runs)
   on a fresh boot (first run at uptime ≈ 60 s) **and** on an old boot
   (≥ 10 min idle) — `[HTP-PROFILE] expert misses … (x ms/miss)`
   (`htp_compute_ops.cpp:859–869`), plus `arm_ms/round` not above A's
   fast runs (0.77–0.94) so the new calls are shown off the critical path.
2. **Decode-window `pgpgin` ≈ 0** (≤ 11 MiB = two experts) in the sampler's
   fold **and** in the app's own new `pgpgin_mib=` field on the pool line
   (§2); `workingset_refault_file` ≈ 0; PSI io ≤ 31 ms.
3. **Text == A** on every run (`strip` + `cmp` against sitting 1's
   `A_G64_r1`, as `216-core-load-run.sh` does); `calls/token=1.00`, close
   clean, ceiling 3840 after every run. Bit identity is untouched by
   construction — `posix_fadvise` changes no byte, the slot, order and
   `pread` are the same — and the host `run_inproc_e2e.sh` pool lines
   (`test/htp/host/run_inproc_e2e.sh:339–352`) print `bit_identical=1`
   with the same `misses=` under the knob.
4. **Prefill ≥ −5 % of the same block's A** (tok/s and the M > 1 MoE
   `dsp`): the lever issues read-aheads during prefill (§3).
5. **Hybrid A unchanged** (nothing set, the product path): one run per boot
   with the knob set; its experts never pass through `readExpert`, so the
   lever is a no-op there by construction (§3).
6. Host: `run_host_checks.sh` `ALL CHECKS PASS` + `WORKER POOL LANES OK`,
   `tools/htp_syntax_check.sh` exit 0, `run_inproc_e2e.sh` `INPROC E2E
   PASS`, `clang-format-14` on changed lines.

## 2. Where it lives (verified on `f5f459587`)

**The memory picture** (S25 SM-S938N, `201/phone_state/device.txt`, the
216 sampler's `M` lines, the run logs):

| item | MiB | source |
|---|---|---|
| MemTotal | 11 114 (11 381 316 kB = 10.85 GiB) | `device.txt` |
| Android before our process (MemTotal − MemAvailable at uptime 16 s) | ≈ 3 740 | b3 `core.samples` first `M` line: MemAvailable 7 374, Cached 4 035 |
| expert arena, Q28: 13 × 256 MiB chunks (616 slots × 5.29 MiB = 3 259 + the 64 MiB grain) | 3 328 | `takeExpertSlot` `:5332–5376`, `expertStride` `:5380–5382`; log `arena chunk 12 … mapped total 3328` |
| FC set + lm_head arena (one PD) | 448 (383.6 attached) | log `s2: fc arena … mapped_mib=448` |
| app RSS (ION mappings excluded: `RSS 599 -> 599` while chunks map) | 766 | log `RSS 766 MB` |
| model file in the page cache after `cat` + preload | 4 116 | runner's `resident … of 4116 MiB` |
| **sum** | **≈ 12 400 > 11 114** | |

≈ 1.3 GiB must leave: per boot the sampler counts 5.3–9.5 M kswapd scans
and 280–394 k pages (1.1–1.5 GB) swapped out to zram (`pswpout`; SwapFree
4 194 300 kB = 4.0 GiB at boot — the task's "12 GB swap" is not what the
sampler shows; `SwapTotal` is read in step 0). The file pages are the
cheapest reclaim, and which survive is the kernel LRU's choice: the `cat`
walks the file head → tail, the preload reads layers 0–18 again, so after
prefill the non-resident experts (layers 0–2, §3) are the *coldest* pages
in the LRU at the moment the arena's 3.3 GB allocation pushes. **Every
resident expert's bytes are held twice** — in its arena slot and in the
page cache — and only the complement (704 − 616 = 88 experts ≈ 465 MiB at
C = 28; 176 ≈ 931 MiB at C = 24) is ever read again. The pool-size
correlation of rule 59 b (C = 16 / 24 cheap, C = 28 dear) is this
arithmetic: the 3.3 GB arena pushes the cache over the edge, C = 24's
2 816 MiB arena does so 512 MiB less often.

**The read path (unchanged by this plan).** `readExpert` (`:5387–5399`) →
`readWeight` (`:5584–5631`): 8 page-aligned `pread` slices through
`ThreadManager::parallel_for` into the uncached ION slot, then the tail;
`preadAll` `:5278–5295`. Callers: the pool server's `poolAnswer`
(`:2299–2383`, `readExpert` at `:2347`), the preload
`register_qs4cx_wh_expert_file` (`:2708–2731`, `at_load=true`), the
prefill miss batch `register_qs4cx_wh_expert_files` (`:2734–2763`), the
prefetch readers (`prefetchReaderLoop` `:5548–5572`). Evictions:
`release_qs4cx_wh_expert` (`:2868–2882`, key only — the slot goes to
`free_expert_slots_`, `experts_` loses the key) from the layer's `release`
lambda (`lfm2_moe_layer.cpp:788–793`, prefill and the hybrid-pool path)
and from `poolAnswer`'s evict callback (`:2322–2330`, decode). The
descriptor with the file range is `ExpertFileDesc{key_gu, key_dn, fd,
off_gu, off_dn, K, inter, N_out}` (`compute_ops.h:435–442`); the fd is the
loader's long-lived `model_file_fd` (`neuralnet.cpp:1070`, `O_RDONLY`),
captured by virtual tensors in `Tensor::read` (`tensor.cpp:1414–1424`). A
weight's bytes in the file are `[off, off + whBytes(K, N) + 8 N)` (nibbles,
N f32 scales, N f32 column sums — what `readWeight` reads at `:5617–5620`).
Virtual experts are never mmapped in the pool path (the layer passes the
tensor's address as a key, `lfm2_moe_layer.cpp:840–844`), so
`POSIX_FADV_DONTNEED` can drop their pages.

**What changes** (all in `nntrainer/tensor/htp_backend/htp_compute_ops.cpp`):

* `readExpert` `:5387–5399`: after both weights are in the slot, with the
  knob on, `posix_fadvise(fd, off, len, POSIX_FADV_DONTNEED)` on both
  ranges.
* `ExpertResident` `:5270–5274` gains the `ExpertFileDesc` (filled in
  `fileRegistered` `:5439–5445`), so `release_qs4cx_wh_expert` `:2868–2882`
  knows the victim's ranges and issues `POSIX_FADV_WILLNEED` on them —
  inline from the layer, **deferred** in `poolAnswer`: the evict callback
  pushes the ranges on a `PoolServer` vector that is flushed after the
  answer word is written (`:2378–2382`), off the DSP's path.
* Knob `NNTR_MOE_FADVISE` parsed like `prefetchKnobs()` (`:6246–6268`):
  unset / `0` = today, byte for byte; `1` = DONTNEED on load + WILLNEED on
  evict (the lever); `2` = DONTNEED only (the diagnostic that shows the
  re-miss cost, §3). Printed on `[HTP] token driver: on …` (`:4046–4050`).
* The pool line `[HTP] token driver: pool misses=… arm_ms/round=…`
  (`:4173–4182`) gains `pgpgin_mib=<Δ /proc/vmstat pgpgin from driver-on
  to close>` (`E2eState` `:6410` gets the baseline; Linux only). From then
  on every pool cell carries its own storage-read count — rule 61's
  requirement, without a sampler.

**Consumers that do not move.** No IDL / stub (`test/htp/nntr_hvx.idl`,
`generate_stub.sh`) or skel change — the DSP is untouched, the staged
`libnntr_hvx_skel.so` md5 `31c0a033…` stays; no quantizer tag
(`nntr_quantize_stream`), no loader check, no `NNTR_HTP_PROFILE` stage
table; `tools/htp_fc_report.py` does not parse the pool line; the 216 fold
(`216-core-load-report.py:63–66`) and runners `grep -o` up to
`arm_ms/round=[0-9.]*`, so the appended field is harmless.
`docs/measurements/216-sampler.sh` / `216-core-load-run.sh` /
`216-core-load-report.py` change as §4 step 3 says.

## 3. Design

**Chosen: (a+) page cache = arena complement.** Drop an expert's file
pages the moment its bytes are in the arena (they are dead there: the DSP
reads the slot), and re-read the victim's pages in the background the
moment it leaves the arena. The cache then holds ≈ the complement
(465 MiB at C = 28) instead of 4 116, the sum in §2 falls to ≈ 8 750 MiB
(≈ 2.3 GiB headroom), kswapd has nothing to reclaim, and a miss is a
cache hit by construction unless its victim was evicted less than one UFS
read (≈ 5 ms) ago.

**Why DONTNEED alone (pure (a)) is not enough — the re-miss arithmetic.**
Replaying the #201 S0 routing trace (`201/s0/logs/moe_trace.txt`, A,
G = 1024, prompt 512) from the device's start state — `preloadExperts`
fills the pool in layer order (`lfm2_moe_layer.cpp:520–533`: layers 0–18
whole, 8 of layer 19), the prefill read-ahead evicts LRU experts outside
the call and outside later layers' resident sets (`:942–960`) — gives at
C = 28 **102 misses at G = 64** (device: 121), **0 first-time loads, 83
re-misses of experts evicted during prefill** (layers 0–2 are the 88
prefill victims; token 0 alone misses 10) **and 19 re-misses of decode
victims**; at G = 512 / 1024: 129 / 132 misses, 84 / 84 prefill victims,
45 / 48 decode victims. At C = 24: 190 / 639 / 1133 misses, 146 / 161 /
162 prefill victims. So after prefill **every decode miss is a re-miss of
an expert the arena once held**: with DONTNEED on load and nothing else,
every miss becomes a UFS read (the slow regime made permanent — the
`NNTR_MOE_FADVISE=2` cell is kept to show exactly that). With WILLNEED at
eviction: the 88 prefill victims are read ahead during the ≈ 1 s prefill
(465 MiB, UFS ≥ 1 GB/s, done before token 0); the decode victims' gap
between eviction and re-miss is p10 1.1 tokens (≈ 25 ms), p50 11 tokens;
**1 of 19 at G = 64 (2 of 48 at G = 1024) falls inside one token** and
pays a partial UFS read once. Script: `scratchpad/remiss2.py` of this
session (to be committed beside the measurement as
`docs/measurements/216-remiss-sim.py` in step 3).

**Ranked levers.**

| rank | lever | arithmetic | risk | code |
|---|---|---|---|---|
| 1 | **(a+)** DONTNEED on load + WILLNEED on evict | cache 4 116 → ≈ 465 MiB; sum 12 400 → 8 750 MiB; misses stay cache hits (0.5 ms); UFS traffic = one expert per eviction (≈ 10 MiB/token at G = 64's 1.89 misses, 1–2 at G ≥ 512) in the background | WILLNEED race when the victim returns within ≈ 5 ms (1 of 19 re-misses at G = 64); prefill pays 88 WILLNEED submissions (≈ 0.1–0.3 ms each ≈ −2 % worst case; fallback below); fadvise cost on the miss path ≈ 0.1 ms (DONTNEED of 1 300 pages) — read in `arm_ms/round` | ≈ 60 lines, one file |
| 2 | (b) pin the complement: `mlock` / `MAP_POPULATE` on the non-resident ranges, re-pinned at every eviction | same memory as (a+) but pinned; no race | needs `RLIMIT_MEMLOCK` ≥ 0.5 GB for the shell user (read `ulimit -l` in step 0; Android's default is small), one VMA per expert, and pinning under pressure moves the kill to lmkd; `MADV_WILLNEED` re-touch per token costs ≈ 88 calls a token | the pinned fallback of (a+) if the race shows in the A/B |
| 3 | (c) C = 24 | arena −512 MiB of a 1.3 GiB shortfall; 2.56–3.58 misses/token at G = 64 (1.2 at G ≥ 512, ten times C = 28's) at 0.5–0.7 ms = 1.3–2.5 ms/token vs C = 28's 1.0 fast / 9.5 slow; `Q24` read 41.5–43.1 against Q28's fast 44.0–44.4 | trades a cure for a milder disease; still under pressure (12 400 − 512 > 11 114), so the slow regime can return | none (knob) |
| 4 | (d) `O_DIRECT` miss read | every miss = UFS ≈ 4–5 ms, deterministic = the slow regime | rejected unless (a+) fails; ION slot alignment is fine | small |

**Rejected alternative: a user-space second-level cache** (keep the
complement in anon memory, copy the victim out of its arena slot before
the overwrite). It is the only design with *no* UFS traffic, but the copy
out reads uncached ION on the ARM (rate unmeasured, likely ≪ 5 GB/s) on
the miss's critical path, the anon pool is zram-swappable, and it is a
new cache with its own policy — ten times the code of (a+) for a race
that the trace says bites once per 64 tokens.

**Hybrid interaction.** The hybrid A (`moe_engine: htp`, nothing set) is
not virtual: the loader reads every expert into the weight pool
(`manager.cpp:474–480`), `get_or_register_wh` (`:5687–`) copies it to the
arena and `releaseArmSource` (`:5653–5673`) `MADV_DONTNEED`s the anon copy.
Nothing in it calls `readExpert` or `release_qs4cx_wh_expert`, so the knob
is a no-op there (gate 5 confirms with a run). The hybrid double-holds too
(3 840 MiB arena + the 4 116 MiB file that nothing reads after load,
`fsu: false` in `nntr_config.json`): a whole-file DONTNEED after the load
would hand Android 4 GB. That is a memory-headroom gain with no decode
read behind it, needs the fd at app level (non-virtual tensors keep no fd,
`tensor.cpp:1421–1424`), and is filed as a follow-up line in LEDGER, not
built here.

**Design rules kept.** No DSP change (doc 45 §3: activation handles, DMA,
`_det` untouched); contract §2's walls and the arena budget unchanged
(C = 28, 3 328 MiB, ceiling 3840); no CPU fallback for `QS4CX_WH`
introduced; the knob's unset value is today's behaviour, and the default
flips only after the A/B, as #115 / #151 did.

## 4. Steps

Each step ends in a rung of `.claude/skills/hexagon-gates`.

0. **Device facts, one adb minute, no build** (goes into the runner's
   header and `216-miss-read.md`): `grep -E 'MemTotal|SwapTotal|SwapFree'
   /proc/meminfo`, `ulimit -l` (decides whether (b) is even possible),
   `cat /proc/sys/vm/swappiness`, `ls /sys/block | grep zram`. *No rung —
   a reading.*
1. **Code** (`htp_compute_ops.cpp` only, §2): the knob; DONTNEED in
   `readExpert`; `ExpertResident` + `fileRegistered` carry the descriptor;
   `release_qs4cx_wh_expert` → WILLNEED, deferred through `PoolServer` in
   `poolAnswer`; `fadvise=` on the driver-on line and `pgpgin_mib=` on the
   pool line. Gate rung 0 + 1: `clang-format-14`; `ninja -C build`;
   `run_host_checks.sh` (`ALL CHECKS PASS`, `WORKER POOL LANES OK`);
   `tools/htp_syntax_check.sh`; `run_inproc_e2e.sh` `INPROC E2E PASS`; then
   its two pool lines (`:339–352`) re-run by hand under
   `NNTR_MOE_FADVISE=1` and `=2`: `bit_identical=1`, the same `misses=`,
   and the new `pgpgin_mib=` field printed. The runnable check the code
   leaves behind: a host registration-only run of the real file
   (`q40-qs4cx-wh`, prompt 16, `NNTR_MOE_CACHE_EXPERTS=28`, G = 1) with
   `tools/htp/page_cache_evict <file> -1` after it — `resident` ≈ 4 116
   MiB with the knob unset, ≈ the complement with `=1` (the host has no
   pressure, so this proves the mechanics, not the gate).
2. **App build once** (rung 3, `--cache`); the skel is not rebuilt (device
   md5 must stay `31c0a033…`); md5s of the app set on the handoff table.
   Rebase onto `htp_decode` once #215 merges and re-run rung 1.
3. **Runners** (`docs/measurements/`, from the 216 set): `216-sampler.sh`
   adds, every 4th sample, `page_cache_evict <model> -1` → a `R <MiB>`
   line (the file's resident MiB *during* the decode window; mincore of
   1 M pages ≈ 0.1–0.3 s on the phone, hence not every sample);
   `216-core-load-report.py` gains `resident min/max in window` and keeps
   `pgpgin MiB` / refaults / PSI io as standard columns next to ms/miss;
   `216-fadvise-run.sh` (from `216-core-load-run.sh`) keeps the `cat` +
   `page_cache_evict -1` pre-read for every variant (A's "warm" definition
   is unchanged; after the lever the pre-read serves only the load) and
   stamps uptime and `fadvise=` on every run line. `216-remiss-sim.py` =
   the replay of §3. *No rung — docs/measurements only; the fold is run
   on b3's logs to show it still parses.*
4. **Device A/B (unavoidable).** One binary, four variants: **A** = knob
   unset (today), **B** = `NNTR_MOE_FADVISE=1`, **D** = `=2` (diagnostic,
   predicted slow), **H** = hybrid A with `=1` (predicted unchanged).
   Two boots, sampler on from `sys.boot_completed`:
   * **Boot 1 (fresh):** from uptime 60 s, `A B A B A B A B` at G = 64,
     then `D` once, then `A B` at G = 512, then `H` once.
   * **Boot 2 (old):** idle to uptime ≥ 600 s, then the same sequence.
   Read per run: ms/miss, `arm_ms/round`, `miss_wait_us/token`, prefill
   tok/s and M > 1 `dsp`, the app's `pgpgin_mib=`, the fold's `pgpgin` /
   refaults / PSI io / kswapd / `resident min/max` for the window, text,
   ceiling, uptime. Gate = §1 on B's eight G = 64 runs and two G = 512
   runs; D is expected at ≈ 4–5 ms/miss with `pgpgin` ≈ 121 × 5.3 MiB
   (if D is *fast*, the model of §3 is wrong and the plan stops for a
   re-read); H within A's spread.
   *If B passes everywhere but prefill fails the −5 %:* move the
   prefill-side WILLNEEDs to one bulk pass at `poolSync` (`:2229–2240`,
   the "a prefill touched the pool" hook: WILLNEED every expert in
   `pool_descs_` not in `experts_`), re-run boot 2's block. *If B shows
   the race (a run with `pgpgin` > 11 MiB and ms/miss > 1.1 while
   `resident` stayed ≥ the complement):* (b) with the `ulimit -l` reading
   from step 0 decides whether pinning is available; otherwise the verdict
   is (a+) with the residual named.
5. **PR into `htp_decode`** (`htp/216-page-cache-complement`): code, the
   runners, `216-miss-read.md` §"A/B" (or `216-fadvise.md`), LEDGER /
   BENCHMARK rows of §6, `bit identity untouched by construction` stated.
   Default flip (unset → `1`) is the user's call at review (precedent
   #115 / #151): the PR ships with unset = today's behaviour unless told
   otherwise.

## 5. Risks

* **Host-vs-device gap is total for the gate.** The workstation has no
  memory pressure; only the phone shows eviction. The host proves bit
  identity, the miss count and the mechanics (`resident` falling to the
  complement); the handoff table's `pgpgin` / `resident` / PSI columns
  per run are what make the device-side effect visible.
* **Readahead race / UFS rate.** WILLNEED is asynchronous; a victim that
  returns within ≈ 5 ms pays the remaining I/O (1 of 19 re-misses at
  G = 64 in the trace). The `pgpgin_mib=` field on every run line bounds
  it (one expert = 5.3 MiB).
* **Prefill-side cost.** 88 WILLNEED submissions and 465 MiB of
  background UFS during a ≈ 1 s prefill; read against the −5 % gate
  inside each block; the `poolSync` bulk pass is the fallback.
* **Other apps / zram.** The lever removes our 3.3 GB of needless cache,
  it does not pin: a foreground app or lmkd can still move the kernel's
  LRU. The sampler's kswapd / `pswpout` / PSI columns say whether a slow
  run is ours; an old boot with other apps resident is the harder case and
  is the second boot of the design.
* **DVFS / thermal.** Slow runs read cpu6/7 at 3.3–3.8 GHz against 4.47 —
  a consequence of I/O waits, and a confound for tok/s; the gate is read
  on ms/miss and `pgpgin`, tok/s is read only within a block (A/B
  interleaved), never across boots.
* **Stale skel / wrong set.** No skel change, so a stale skel cannot
  appear as a win; md5s on every run line.
* **Address-space budget.** Unchanged (no new ION, C = 28, ceiling 3840).
* **The replay is an approximation.** 102 vs the device's 121 misses at
  G = 64 (the read-ahead's exact victims differ); the classification
  (every miss a re-miss, 80 % prefill victims) does not depend on the
  difference. D's cell is the on-device check of that model.

## 6. Docs to update

* `docs/htp_moe/LEDGER.md`: ㉜ → cause named (page-cache eviction under
  the double holding, `pgpgin` ↔ ms/miss) and the lever's verdict; rule 61
  rewritten — the slow regime is memory pressure, not a boot window, the
  5-min wait is dropped, and a pool cell is readable only with its
  `pgpgin_mib=` (or window `pgpgin`) beside it; rule 59 b's "C = 28 dear"
  gets the arithmetic; #208's "≈ 4.7 ms a miss" is the same candidate
  cause; a §2 verdict row for #216; a follow-up line for the hybrid's
  whole-file drop (memory headroom only). If the default flips: rule row
  as #115 / #151.
* `docs/htp_moe/BENCHMARK.md`: Method cycle note (profiled runs, no
  Results row — contract §1.1); the "warm" protocol note gains "`pgpgin`
  per window is a standard column"; if the default flips, the next
  sitting's A carries it.
* `docs/measurements/201-s3-probe.md` "Not verified here" and
  `216-miss-read.md` "What this changes": pointer to the A/B result.
* `docs/plans/201-htp-decode-e2e-review-gemma-moe.md`: the preload-order
  observation from §3 (layers 0–2 cold after every prefill, ≈ 1.5 of the
  1.89 misses/token at G = 64) as a one-line candidate for the policy
  track, not for this issue.
