# Plan 219: the 88-expert complement in a cached ARM tier — a pool miss is a memcpy, the victim is refilled off the token path, the page cache leaves the sum

Issue #219 (p1, part of #76 / #201; follows #216). Base `htp_decode` @
`0f67e263b`; every `path:line` below is that tree. Sources: `216-fadvise.md`
@ `45be8de65` (PR #218, `htp/216-fadvise`, **not merged**), `216-miss-read.md`
(PR #217), `201-s3-probe.md`, LEDGER cycle 30, rules 61 (amended) / 62, ㉜.
Work branch `htp/219-tier` from `htp_decode`; PR into `htp_decode`.

**No device is attached (contract §12, 2026-10-02).** Steps 1–3 are
host-gated and the implementer runs them; step 4 is a handoff the user runs
from a device-farm session, filed as an issue comment (`needs-user` +
`state:needs-measurement`). Host checks come first, by design of §4.

## 0. What the planner found (it changes two of the issue's numbers)

1. **The miss memcpy will not run at the staging rate.** The 22–26 GB/s of
   `201-s3-probe.md` is the staging copy into a *cached* buffer. The expert
   slot is the uncached ION arena, and `readWeight`'s own comment
   (`htp_compute_ops.cpp:5438–5444`) records the store rate into it:
   "capped near 4.9 GB/s by the uncached mapping whatever the thread count:
   8 slices bought 18 %". A 5.3 MiB copy into the slot is therefore
   ≈ 0.3–0.6 ms with the 8-slice `parallel_for`, ≈ 1.1 ms single-threaded,
   not 0.25. The issue's gate (≤ 0.5 ms/miss) is kept as the gate; the
   plan's own estimate straddles it, and §1 says what each outcome means.
2. **Decode-window `pgpgin` cannot be ≈ 0 with a refill from storage.**
   `pgpgin` counts every block read, direct or buffered; each miss refills
   its victim from UFS (5.3 MiB), so a G = 64 run reads ≈ 121 × 5.3 ≈ 640 MiB
   in the window by construction — exactly what #216's B showed (639–643).
   The signal the issue wants ("no eviction-driven storage read on the token
   path") is read instead on `workingset_refault_file` ≈ 0, PSI io low,
   `tier_hits = misses`, `tier_waits ≈ 0`, and `pgpgin_mib ≈ misses × 5.3`
   (the refills, nothing else). §1 states the gate that way; the issue
   comment says so.
3. **The page cache gone, prefill's misses must come from the tier too.**
   Today prefill's ≈ 83 loads (`register_qs4cx_wh_expert_files`, the
   prefetch readers) are page-cache hits because the runner pre-reads the
   file. Once the file's pages are dropped at load and never re-read
   buffered, a prefill load that still `pread`s the file is a UFS read
   (≈ 1.75 ms × 83 ≈ 145 ms on a ≈ 1 s prefill: −10 %, a gate fail by
   construction). So **every** load path takes its bytes from the tier; the
   file is read only by the refill helper, `O_DIRECT`, off both the token
   and the prefill compute path. This deviates from the issue's "nothing
   else changes in prefill" and is the reason.

## 1. Goal and gate

Goal (issue): a decode pool miss costs a memcpy, never a storage read; the
model file's pages leave the memory sum; the victim's bytes return to the
tier off the token path. Q28 one PD (`NNTR_HTP_E2E=1 NNTR_HTP_E2E_PDS=1
NNTR_MOE_CACHE_EXPERTS=28`), prompt 512, S25 (`R3CY10WM83Y` or the farm
unit; one unit per sitting).

| # | check | pass |
|---|---|---|
| G1 | ms/miss | **≤ 0.5 ms on every B run** of a fresh boot (first run at uptime ≈ 60 s) and a ≥ 10-min-old boot — profiled `file read … (x ms/miss)` (`htp_compute_ops.cpp:862–869`) on `prof=2` runs, else `arm_ms/round × rounds / misses` from the pool line (`:4082–4091`). The same binary's A is the control inside each block. **Reading:** 0.5–0.7 on every run with `tier_hits = misses` and refaults ≈ 0 is "the slow regime is gone, the copy is store-bound" — the lever worked, the gate number did not; the user decides as for #218. A single B run > 1.1 is a fail |
| G2 | no eviction-driven read | per B run: `tier_hits = misses`, `tier_waits ≤ 2`, window `workingset_refault_file` ≤ 2 000 (two experts), PSI io ≤ 70 ms, `pgpgin_mib` within ± 11 MiB of `misses × 5.29` (the refills and nothing else), `pswpin = pswpout = 0` in the window; no `fadvise` on the token or prefill path (the one drop is on the loader, printed with its ms) |
| G3 | prefill (standing) | B's prefill tok/s and M > 1 MoE `dsp` **≥ −5 % of the same block's A** (4-run means at G = 64; single pairs at G = 512 / 1024 read with that caveat). The DMA ring, worker pool and weight layout are untouched (stated in the PR) |
| G4 | decode | B's G = 64 block mean ≥ A's fast regime on the unit (≥ 43.7 tok/s on `R3CY10WM83Y`'s block, `216-fadvise.md` old A3); B ≥ A at G = 512 and G = 1024 |
| G5 | bits and text | **text == A r1 of its G on every run** (`strip` + `cmp`, `216-fadvise-run.sh`'s definition); host `run_inproc_e2e.sh` pool lines (`test/htp/host/run_inproc_e2e.sh:431–441`) `bit_identical=1` with the same `misses=` under the knob unset / 1 / 2. Bit identity holds by construction: the tier holds the file's bytes, and the copy writes exactly what `readWeight` writes (nibbles verbatim, scales verbatim, sums by the same `float → int32` cast, §3); stated in the PR |
| G6 | hygiene | `calls/token=1.00`, `close tokens=… timeouts=0 stale=0`, S1 ceiling 3840 after every run; hybrid H vs H0 inside the hybrid's spread (the hybrid's experts are not virtual, so `preloadExperts` returns at `lfm2_moe_layer.cpp:491` and no tier exists — no-op by construction) |
| G7 | memory | `resident` of the model file after load and after every B run ≈ 0 MiB (`page_cache_evict <file> -1`; A's stays 1–4 GiB); the load log's `tier: experts=88 mib=…` ; RSS budget of §3 holds: process anon ≈ 766 + 467 ≈ 1 233 MiB, no `pswpout` |
| G8 | host | `run_host_checks.sh` `ALL CHECKS PASS` + `WORKER POOL LANES OK`; `run_inproc_e2e.sh` `INPROC E2E PASS` with the knob unset, `=1`, `=2`; `tools/htp_syntax_check.sh` exit 0; `*Lfm2Moe*` 6 / 6; `clang-format-14` on changed lines |

## 2. Where it lives (verified on `0f67e263b`)

**The pool's read path today.** `readExpert` (`htp_compute_ops.cpp:5239–5251`)
→ `readWeight` (`:5436–5490`): 8 page-aligned `pread` slices via
`ThreadManager::parallel_for` into the uncached slot (`use_pool`), then the
tail (N f32 scales + N f32 sums → f32 scales, i32 sums written at
`:5472–5486`). Callers: the pool server `poolAnswer` (`:2285–2362`,
`readExpert` at `:2347`; evict callback `:2322–2330` →
`release_qs4cx_wh_expert`), the preload `register_qs4cx_wh_expert_file`
(`:2666–2688`, `at_load=true`, `readExpert` at `:2676`), the prefill miss
batch `register_qs4cx_wh_expert_files` (`:2692–2721`, `:2704`), the prefetch
readers `prefetchReaderLoop` (`:5400–5425`, `readExpert(st, false)` at
`:5415`). Eviction: `release_qs4cx_wh_expert` (`:2826–2839`, key only; the
slot goes to `free_expert_slots_` `:6059`). Residency: `experts_` (`:6060`),
`ExpertResident` (`:5122–5126`), `fileRegistered` (`:5291–5297`). Slots:
`takeExpertSlot` (`:5184–5226`), `expertStride` (`:5232–5234`, page-rounded),
`reserve_qs4cx_wh_expert_slots` (`:2808–2811`, `expert_slots_wanted_`).
Descriptor `ExpertFileDesc{key_gu, key_dn, fd, off_gu, off_dn, K, inter,
N_out}` (`nntrainer/tensor/cpu_backend/compute_ops.h:435–441`); the fd is
the loader's `model_file_fd` (`neuralnet.cpp:1070`, `O_RDONLY`, long-lived).
Weight bytes in the file: `[off, off + whBytes(K, N) + 8 N)`
(`htp_wh_layout.h:60`). Shape here: K 2048, inter 1792, N_out 2048 → gate_up
3.5 MiB + down 1.75 MiB ≈ 5.29 MiB an expert (the arena's 3 328 MiB / 616).

**The load order.** `transformer.cpp:543–549` calls
`Lfm2MoELayer::preloadExperts` per layer at load (`lfm2_moe_layer.cpp:490–538`):
it fills the LRU in layer order until `g_expert_lru.capacity()` (`:517–521`),
so layers 0–18 are whole, 8 of layer 19, and layers 19 (24), 20, 21 are the
88-expert complement. **The first layer whose `need.size() < num_experts`
is the moment the pool is full** — the hook for the tier fill and the drop
(§3). `set_decode_moe_experts` (`:754–771` → `:2204–2212`) runs at the first
*forward*, too late for a load-time fill.

**The memory picture (S25, LEDGER cycle 30, `216-fadvise.md` step 0)**, MiB:
MemTotal 11 114; Android before the app ≈ 3 740; expert arena 3 328 + FC
arena 448 (ION, pinned, outside RSS); app RSS 766; model file in the page
cache 4 116 after `cat` + preload; SwapTotal 4 096 (zram); `ulimit -l`
64 KiB (no `mlock`). Sum today ≈ 12 400 > 11 114.

**What changes.**

| file | change |
|---|---|
| `nntrainer/tensor/cpu_backend/compute_ops.h` (after `:458`) | one virtual, default no-op: `virtual void tier_qs4cx_wh_experts(const std::vector<ExpertFileDesc> &not_preloaded) {}` — "these experts of the layer were not preloaded; hold them". No IDL, no skel, no stub (`test/htp/nntr_hvx.idl`, `generate_stub.sh` untouched); `CpuComputeOps` inherits the no-op |
| `Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.cpp:533–537` | after the acquire in `preloadExperts`: `if (need.size() < num_experts) ops->tier_qs4cx_wh_experts({descs.begin() + need.size(), descs.end()});` (≈ 3 lines) |
| `nntrainer/tensor/htp_backend/htp_compute_ops.cpp` | the tier (§3): `tier_qs4cx_wh_experts` override (fill + drop), `fetchExpert` replacing the three `readExpert` call sites (`:2347`, `:2704`, `:5415`; the preload `:2676` keeps `readExpert`), `release_qs4cx_wh_expert` → refill enqueue, the refill helper thread (start / stop beside `prefetch_readers_`, destructor `:1131–1146`), `writeTail` factored out of `readWeight` `:5472–5486`, knob `tierKnob()` beside `prefetchKnobs()` (`:6102`), `tier=` on the driver-on line (`:3985`), `tier_hits= tier_waits= tier_wait_us= refill_ms= pgpgin_mib=` on the pool line (`:4082–4091`), `E2eState` counters (`:6232–6265`), a `tier:` load line |
| `docs/measurements/219-tier.md`, `219-tier-run.sh`, `219-tier-sampler.sh`, `219-tier-report.py` | the handoff: copied from #218's `216-fadvise-*` (they live on an unmerged branch; copy, do not depend), variants of §4, new columns `tier_hits tier_waits refill_ms refault pswpin` |

`pgpgin_mib=` and `vmstatPgpginKib()` are PR #218's (`htp/216-fadvise`
diff, ≈ 25 lines). The branch carries that hunk verbatim so the field exists
whether #218 is merged or closed; a rebase after a #218 merge drops it.

**Consumers that do not move.** No DSP change: IDL / stub / skel untouched
(the skel is rebuilt once at rung 2 for the md5, because `htp_decode` moved
DSP sources since the staged `9d61aef4…`; same sources for A and B). No
quantizer tag (`nntr_quantize_stream`), no loader check, no
`NNTR_HTP_PROFILE` stage table (`file read … ms/miss` keeps its name — it is
"the pool server's time per miss", a memcpy now; said in the handoff);
`tools/htp_fc_report.py` does not parse the pool line; the runners' `grep -o`
up to `pgpgin_mib=` tolerate appended fields.

## 3. Design

**Chosen: a user-space second-level cache that mirrors the arena's
complement (the issue's mechanism), with every load served from it.**

*Invariant.* Every expert is in exactly one of: the arena (616), the tier
(88 slots), or in flight between them. Tier slots = `n_layers × E −
capacity` = 704 − 616 = 88 (`experts_.size()` is 616 at the first tier call,
`expert_slots_wanted_` names the capacity); a tier slot is 4 KiB-aligned
(`posix_memalign`) and holds the *file image* of both weights as their
enclosing 4 KiB-aligned ranges (`[align_down(off), align_up(off + len))`,
≈ gu 3.5 + dn 1.75 MiB + ≤ 16 KiB), so `O_DIRECT` reads it in one `pread`
per weight. Tier memory: 88 × ≈ 5.31 MiB ≈ **467 MiB**, plain anon.

*Load.* `tier_qs4cx_wh_experts(rest)` (called by layers 19–21 at load): the
first call opens `/proc/self/fd/<d.fd>` once with `O_RDONLY | O_DIRECT`
(buffered fallback with a log line if `open` fails, e.g. tmpfs on a host),
reads each expert of `rest` into a tier slot on the loader thread (88 ×
≈ 5 ms ≈ 0.4–0.5 s of load), then one `posix_fadvise(fd, 0, 0,
POSIX_FADV_DONTNEED)` over the **whole file** (rule 62's cost is per call —
`lru_add_drain_all` and the range walk — so one call per tier call, 3 at
C = 28, not one per expert; `tools/htp/page_cache_evict.c` does the same
call on the device today). Printed: `[HTP] tier: experts=88 mib=467
read_ms=… drop_ms=… direct=1`. After this no `fadvise` runs anywhere.
The `cat` pre-read of the runner is kept for A and B alike (A's "warm"
definition is unchanged; B's drop removes it, measured in `drop_ms`).

*Miss (`fetchExpert(st, use_pool)` in place of `readExpert` on the three
load paths).* Look up `st.d.key_gu` in the tier: **READY** → copy the
nibbles (`use_pool`: the same 8-slice `parallel_for` as `readWeight`, since
the uncached store rate per thread is ≈ 4.9 GB/s; the readers copy
single-threaded as they read today) and `writeTail` (scales verbatim, sums
`static_cast<int32_t>(float)` as `readWeight:5479–5481`), free the tier
slot, `++tier_hits`; **PENDING / IN_FLIGHT** (a victim re-missed before its
refill landed) → promote it to the helper's front, wait on the helper's cv
(`++tier_waits`, `tier_wait_us`), then as READY; **ABSENT** (tier off, or a
bookkeeping hole) → today's `readExpert` from the file (`++tier_reads`,
expected 0 with the knob on). Bit identity: the tier bytes are the file's;
the copy's output is byte for byte `readWeight`'s.

*Eviction → refill.* `release_qs4cx_wh_expert` (from the layer's `release`
lambda `lfm2_moe_layer.cpp:787–792` in prefill, and from `poolAnswer`'s
evict callback) pushes the victim's `ExpertFileDesc` on the helper's queue
as PENDING. One helper thread (unpinned, like #218's worker; started at the
first tier call, stopped in the destructor beside the readers) pops a victim
when a tier slot is free (the slot its load vacated — a round evicts n then
loads n, so a slot frees within the same `poolAnswer`), `pread`s the two
enclosing ranges `O_DIRECT` (`refill_ms` accumulates; ≈ 1.75 ms a cold read
at UFS rate), marks READY, notifies. Decode: ≈ 1.89 victims a token at
G = 64, re-miss gap p10 1.1 tokens (plan 216 rev. 2 replay) → `tier_waits`
expected ≈ 1 in 64 tokens. Prefill: the ≈ 83 victims refill during the
≈ 1 s prefill (440 MiB over UFS, DMA, little CPU) — the prefill gate G3 is
where that shows. `ExpertResident` needs no descriptor: `pool_where_` /
`pool_descs_` are set only at the first forward, so the release path takes
the descriptor from a `key_gu → ExpertFileDesc` map the tier fills at load
(the preload's `register_qs4cx_wh_expert_file` records it too, 1 line).

*Knob.* `NNTR_MOE_TIER`: unset / `0` = today, byte for byte, no thread, no
`fadvise`; `1` = the tier, 8-slice copy on the pool server; `2` = the tier,
single-thread copy (diagnostic: the uncached store rate; if `=2` equals `=1`
the `parallel_for` goes). Default stays unset until the sitting; the flip is
the user's call at review (precedent #115 / #151 / #218).

**RSS budget (the constraint named in the task).** Process anon RSS 766 →
≈ **1 233 MiB** (+467 tier); ION 3 328 + 448 = 3 776 (unchanged, pinned);
Android ≈ 3 740; **sum ≈ 8 750 MiB on 11 114 → ≈ 2 360 MiB headroom**, with
the file's page cache ≈ 0 by construction: the preload's buffered pages are
dropped once at load, the tier fill and every refill are `O_DIRECT` (never
cached), and no load path `pread`s the file buffered any more (§0.3). The
double holding is gone: a resident expert exists once (arena), a non-resident
one once (tier). What can still fill the cache is other apps; kswapd then
has only *their* pages and zram to take — the sampler's `pswpout` /
`pswpin` columns (G2) say whether it took ours.

**Rejected alternative: copy the victim out of its ION slot into the tier
(no storage traffic at all, `pgpgin ≈ 0`).** The read side of an uncached
ION mapping on the ARM is unmeasured and presumably far below 5 GB/s, and
the copy would have to finish before the slot is overwritten — i.e. on the
pool server's path, inside the miss round, adding ≥ 1 ms to each miss. The
file refill costs 5.3 MiB of UFS per miss on a thread nobody waits for. If
G2's refault / PSI columns are clean, `pgpgin ≈ 640 MiB` is the price and
is paid off the path. Also weighed, not built (issue step 0): (c) C = 24
(512 MiB less pressure, still over the sum, 2.5–3.6 misses/token); (d)
`O_DIRECT` on the miss read itself (deterministic ≈ 1.75 ms = slower than
today's fast regime); `MADV_PAGEOUT` on a mapping as a cheaper drop
(unnecessary once the file is dropped once at load and never re-cached);
a cached-ION (`rpcmem`) tier — pinned, unswappable, no DSP mapping needed
— kept as the fallback if G2 shows `pswpin` on the tier's pages.

**Design rules kept.** No DSP change (doc 45 §3: activation handles, DMA
behind compute, `_det` before every quantizer — untouched); contract §2:
three walls untouched, arena budget unchanged (C = 28, 3 328 MiB, ceiling
3840, no new DSP mapping — the tier is ARM-only, so the 32-bit DSP address
space is not touched), no CPU fallback for `QS4CX_WH` (the ABSENT branch is
today's file read into the arena, not a CPU matmul); bit-identical + text
gates. Simplifications marked `ponytail:` in code: one tier fd / one helper
(a second helper is the upgrade if `refill_ms` lags prefill's evictions);
tier size fixed at the first call (one model per process, as
`set_decode_moe_experts` already assumes `lfm2_moe_layer.cpp:756–758`).

## 4. Steps

Each step ends in a rung of `.claude/skills/hexagon-gates`.

1. **Interface and hook** (`compute_ops.h`, `lfm2_moe_layer.cpp:533–537`),
   plus the `pgpgin_mib=` hunk from #218. Rung 0 + 1: `clang-format-14`,
   `ninja -C build`, `*Lfm2Moe*` 6 / 6, `run_host_checks.sh`,
   `tools/htp_syntax_check.sh`, `run_inproc_e2e.sh` `INPROC E2E PASS`
   (no behaviour change yet: a no-op default).
2. **The tier** (`htp_compute_ops.cpp`, §3: fill + drop, `fetchExpert`,
   refill helper, `writeTail`, knob, log fields). Rung 1 again, then the
   pool lines (`run_inproc_e2e.sh:431–441`) re-run by hand under
   `NNTR_MOE_TIER=1` and `=2`: `bit_identical=1`, the same `misses=`
   (5 / 56 / 14 / 14 as in `216-fadvise.md`), and on the log the new
   fields with `tier_hits = misses`, `tier_reads=0`, the `tier:` line with
   `direct=1`. The runnable check this leaves behind: the pool lines under
   the knob, in the script itself (one new `E2E e3 pool tier … tier_hits=N
   bit_identical=1` line per fixture, failing when `tier_reads > 0`).
   Optional, if `$NNTR_MODEL_DIR/q40-qs4cx-wh` is present on the
   workstation: a registration-only run with a small C (the inproc arena
   is capped at 512 MiB, `216-fadvise.md` "not verified") to exercise
   `O_DIRECT` on the real file's offsets; the `tier:` line and
   `page_cache_evict <file> -1` (plain POSIX, builds with `cc`) ≈ 0 MiB
   after load. Not a gate, a mechanics check.
3. **Skel + app build once** (rung 2 for the md5 — `ARCH OK (V79)`,
   `UNDEFINED SYMBOLS OK`; rung 3 `--cache`); md5s of the set into the
   handoff table. Then the handoff doc + runner (`docs/measurements/219-*`,
   from #218's `216-fadvise-*`: two boots, sampler with `R` lines every 4th
   sample and `workingset_refault_file` / `pswpin` / `pswpout` in `V`,
   report with the new columns, `strip` unchanged). The sitting plan:
   * **Boot 1 (fresh, first run at uptime 60 s):** `A B A B A B A B` at
     G = 64 (the 4th pair profiled), `B2` once (G = 64, profiled), `A B` at
     G = 512 (profiled), `A B` at G = 1024, `H0 H`.
   * **Boot 2 (old, idle to uptime ≥ 600 s):** the same sequence.
   Variants (≤ 4): **A** = knob unset (unchanged reference), **B** =
   `NNTR_MOE_TIER=1`, **B2** = `=2`, **H / H0** = hybrid A with / without
   the knob (control pair, not a lever cell). Per run: prefill tok/s and
   M > 1 `dsp`, decode tok/s, ms/miss, `arm_ms/round`, `misses`,
   `tier_hits/waits/reads`, `refill_ms`, `pgpgin_mib`, window refaults /
   PSI io / kswapd / `pswpin` / `pswpout` / `resident min`, `resident
   after`, uptime, text, ceiling, the load's `tier:` line (`drop_ms`).
   Expected: B 0.3–0.7 ms/miss on every run, refaults ≈ 0, `pgpgin_mib ≈
   misses × 5.3`; B2 ≈ 1.1 ms/miss (if B2 ≈ B, drop the slices in the PR).
   **Device measurement unavoidable here**: filed as an issue comment on
   #219 with the doc path, the staged set and md5s, the commands, and the
   empty tables; labels `needs-user` + `state:needs-measurement`. *No
   rung — docs/measurements only; the report is run on #218's b-logs to
   show it still parses.*
4. **Fold (after the user fills the tables):** the supervisor reads G1–G8;
   the PR into `htp_decode` (`htp/219-tier`) carries the code, the handoff
   with its tables, §6's rows. If G3 fails with the refills in prefill, the
   one fallback is to start the helper's queue at the first `poolArm`
   (refills of prefill victims deferred to the decode start; token 0 then
   waits on ≈ 10 of them — read `tier_waits` on that cell); if G2 shows
   `pswpin` on the tier, the cached-ION tier of §3. Either is a second
   handoff, not a guess.

## 5. Risks

* **Host-vs-device gap is total for G1–G4.** The host has no memory
  pressure, a different store rate into its "arena", and ext4 `O_DIRECT`
  semantics; it proves bits, counts and mechanics. The handoff table's
  per-run `pgpgin` / refaults / PSI / `pswpin` / `resident` columns are what
  make the device effect visible, run by run, next to ms/miss.
* **Uncached store rate (§0.1).** The copy may land at 0.5–0.7 ms; B2 vs B
  separates the per-thread cap from the DDR limit. The gate reading in G1
  names the outcome either way.
* **UFS rate and the refill race.** A re-miss inside one refill (p10 gap
  1.1 tokens vs ≈ 2–5 ms refill latency including the queue) waits on the
  helper; `tier_waits` / `tier_wait_us` bound it per run. 83 refills ride
  the prefill's ≈ 1 s (440 MiB at ≥ 0.5 GB/s): G3 is read inside each block.
* **zram.** The tier is anon and swappable (`mlock` 64 KiB). The budget
  leaves ≈ 2.4 GiB; a run with `pswpin > 0` in the window and slow tier
  hits points at it, and the cached-ION tier is the named fallback.
* **DVFS / thermal.** A/B interleaved inside one boot, tok/s read only
  within a block; ms/miss and the refault columns are the primary read.
  Uptime on every run line; zone0 not available on the farm shim — the
  two-boot design is the thermal control there.
* **Stale skel / wrong set.** The skel is rebuilt from the branch head and
  pushed with the set; `md5_device.log == md5.txt` gates every boot. No DSP
  change, so a stale skel cannot appear as a win.
* **Address-space budget.** Unchanged: no new DSP mapping, ceiling 3840
  checked after every run.
* **The drop's cost at load** (one whole-file `DONTNEED` per tier call ×3):
  unmeasured on this kernel; read on `drop_ms`. If it is seconds, the
  upgrade is `O_DIRECT` for the preload reads too (then no drop at all).
* **O_DIRECT availability.** f2fs `/data` supports it; the fallback is
  buffered + the same drop after each refill batch (logged `direct=0`), a
  slower and cache-filling path — a `direct=0` on the device is a stop.
* **Sitting is run by the user.** The runner must be self-contained
  (stage, md5, reboot, sampler, sequence, stop rules, pull) and tested
  for syntax (`bash -n`) and against #218's logs for the report; nothing
  in it may assume a workstation path.

## 6. Docs to update

* `docs/htp_moe/LEDGER.md`: §2 verdict row for #219; ㉜ → the tier's
  verdict (closed, or the residual named); rule 61 gains "a pool cell
  carries `tier_hits/waits` and the window's refaults next to `pgpgin`";
  rule 62 gets the pointer ("the lever that avoids the syscalls: #219");
  a new silicon rule only if measured — the uncached store rate per
  thread / aggregate from B vs B2, and the whole-file `DONTNEED` cost; a
  rule row as #115 / #151 if the default flips.
* `docs/htp_moe/BENCHMARK.md`: one Results row for the sitting (lever cell,
  not of record, as #216's); Method cycle note; if the default flips the
  next sitting's A carries `NNTR_MOE_TIER=1`.
* `docs/measurements/216-fadvise.md` "Not verified here" and
  `201-s3-probe.md`: pointers to `219-tier.md`.
* `docs/plans/201-htp-decode-e2e-review-gemma-moe.md` S3 lever 3: closed by
  #219 (or the residual).
