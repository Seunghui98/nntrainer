# 90 — Two-reader DDR probe, re-purposed for target (b): do CPU and DSP reads add up, and what does a CPU prefetch during the MoE call cost the DSP?

Issue #90 (**p1** since 2026-09-29, tracker #76, LEDGER ④ / ⑫ / rule 12).
Contract `docs/plans/0001-htp-moe-decode-agent-system.md` §3.2 and the
decisions row of 2026-09-29 (targets (a) + (b), `htp_moe` @ `df17fcdb`).
This plan **replaces** the 2026-09-27 version of this file. That version
only made the DSP side a valid DDR stream; this one keeps that part and adds
three things: CPU thread sweeps, an HVX-direct reader, and the
prefetch-overlap cell that the new purpose needs. Base = `origin/htp_moe` @
`df17fcdb`, worktree `/home/j2z0-lee/nntrainer-90`, branch
`htp/90-two-reader`. Every `path:line` below was read there.

**Why (b), in numbers.** A decode token reads ≈ 886 MB (rule 42): 484 MB
of MoE weights on the DSP, which the ring reads at 33 GB/s in the app
(0.88 × the 37.3 anchor, rule 41), plus ≈ 402 MB of Q4_0 FC and lm_head
on the CPU, at ≥ 37–41 GB/s effective. The two readers take turns, so the
DDR sits at one reader's rate. HeteroLLM (SOSP 2025, arXiv 2501.14794,
Snapdragon 8 Gen 3) measured one processor at 40–45 GB/s and GPU + NPU at
≈ 60 GB/s. The achievable peak there was 61.9 GB/s, which is 0.81 of
LPDDR5X-9600's 76.8 GB/s.

**The ceiling on this phone.** On the S25 Ultra (SM8750), the repo's only
peak figure is **≈ 85.3 GB/s**: a 64-bit LPDDR5X-10667 bus, 10 667 MT/s ×
8 B (`docs/plans/100-dma-chunk-list.md:47-50`). That plan calls it spec
arithmetic and says the phone's DRAM grade is not recorded in this repo.
10 667 MT/s is also the fastest LPDDR5X that SM8750 supports. That figure
comes from Qualcomm's product brief ("up to 5.3 GHz"), not from repo docs.
So no DDR read on this phone can exceed 85.3 GB/s whatever the grade. That makes
85.3 the probe's hard INVALID bound.

**Prediction for this sitting.** Scaling HeteroLLM's efficiency
(0.81 × 85.3) gives ≈ 69 GB/s achievable. #77's CPU alone already read
**67.9 GB/s** (on the other unit, `R3CY205ZMND`). So the likely outcome is
that the saturating CPU (8 threads) plus the DSP do **not** add much above
≈ 69–70 GB/s. The lever in (b) is then the headroom a **light** CPU reader
finds beside the DSP's 33–37 GB/s: about 30 GB/s, or ≈ 20 MB in a 700 µs
window. The sitting must confirm or refute this. The plan does not assume
it.

## 1. Goal and gate

The issue's acceptance, re-purposed by the 2026-09-29 comment, made
measurable. Everything below is read from the logs of **one** sitting on
`R3CY10WM83Y`: the probe is run cold, then the E2E control, then the probe
again warm (§4 step 6).

| gate | read from | pass |
|---|---|---|
| **(1) readers alone** | `DDR_CPU threads=t` (t = 1, 2, 4, 8), `DDR_DSP reader=ring`, `DDR_DSP reader=hvx` | every line `valid=y`; `cpu_alone(8)` ≥ 25 GB/s (plan 77 §5 escape); `dsp_ring_alone` in **30–120 GB/s** and within ±20 % of the same run's `DDR_DSP_REF` (the #100 `f2` 20-call cell); `dsp_hvx_alone` in 15–40 GB/s (rule 27 band 21–27) |
| **(1) both at once** | `DDR_TWO_READER reader=ring cpu_threads=t` for t = 1, 2, 4, 8, and `reader=hvx` for t = 2, 8 | `valid=y` on every row; each row prints `cpu_alone cpu_with dsp_alone dsp_with aggregate cpu_loss_pct dsp_loss_pct` |
| **(2) prefetch overlap** | `PREFETCH_OVERLAP size_mib=S touch_threads=T` for S = 4, 10, 20, 32 and T = 1, 2, 7 | every line carries `dmm_pct` (the DSP `mm` slowdown vs `PREFETCH_COLD` of the same S), `staged` (fraction of S the caches still hold at re-read) and `net_ms_per_token`; `touch_late` ≤ 3/64 at S ≤ 10, T ≥ 2 (the touch fits the window) |
| **validity (rule 12)** | every DDR line | the test prints `INVALID` and fails (`EXPECT`) when any DDR rate or aggregate is > **85.3 GB/s** (physical, tighter than the issue's 150), when a ring cell's `checksum_ok=n`, or when the ring stream is > ±20 % from `DDR_DSP_REF`. Cache re-reads (`hot_us`, `reread_us`) are not DDR rates and carry no bound |
| **(3) decision** | the filled table, §3.4 rules P and S | the supervisor writes one verdict per form: P = "build the in-app prefetch variant" / "close"; S = the three numbers for the user, parked behind track (c) |
| **host** | `bash test/htp/host/run_host_checks.sh` | the existing `SKEL REPLAY MATCHES TAG SIMULATOR (16 cells)`, `ALL CHECKS PASS`, `WORKER POOL LANES OK`, plus new `TWO READER CELL SOUND (bytes=528482304 footprint=168 MiB)` and `TWO READER BOUNDS OK (3265.6 INVALID, 85.4 INVALID, 67.9 valid, net=…)`; `tools/htp_syntax_check.sh` rc 0 |
| **skel** | `test/htp/build.sh` | `UNDEFINED SYMBOLS OK`. The skel includes `nntr_moe_dma_plan.h`, so rung 2 runs as a compile proof. **The device keeps the staged skel `37468a7f…`**: the additions are unused `static inline` code, and the IDL is unchanged |
| **gtest** | `ndk-build unittest_hvx_dma_probe` | binary exists; md5 recorded |
| **standing gates** | E2E control A in the same sitting | No model-path code changes. A's decode / prefill / text are recorded as the sitting's anchor. Its text must equal #150 T8's text for the same prompt and G (hash), and its prefill must be within −5 % of #150's A0 |

## 2. Where it lives

| file | what changes |
|---|---|
| `test/unittest/unittest_hvx_dma_probe.cpp` | **`TwoReaderDdr` (`:334-452`) rewritten.** The CPU side reuses `stream_xor` (`:285-315`) and `CpuStreamer` (`:353-390`) and runs t = 1, 2, 4, 8, 1.5 s each; each thread prints the CPU it ended on (`sched_getcpu`). The DSP side drops `dma_probe` (`:401-433`, the 3265 GB/s defect) for two readers. **ring** = `dma_replay` on the `f2` list with `fresh=1`, chunks of 500 calls until Σ`res[0]` ≥ 1.2 s, tag-checked; this is the old plan's §3.1, unchanged. **hvx** = `mm_u8i4_moe_layer_timed` at M = 1 with `moe_set_opts(feed=0, rows1, lead 192)`: the D192 direct HVX arena read, the only direct DSP reader production has, looped ≥ 1.2 s over rotating expert sets. The print keeps #77's prefix fields and appends new ones. It also prints `CPU_CACHE` lines read from `/sys/devices/system/cpu/cpu{0,7}/cache/index*/{level,size,shared_cpu_list}` |
| same file, **new `TEST_F(HvxDmaProbe, PrefetchOverlap)`** | §3.2. It registers 16 expert sets (64 arena handles; the model registers 1408) over the two fixture chunks as `MoeChunkReplay` does (`:487-506`), and sets the app's default cell `moe_set_opts(feed=1, rows1)` (flags as `MoeGemvOpts`, `unittest_hvx_mm_u8i4.cpp:2374-2377`; bits in `hexkl_mm_u8i4_moe.h:235-327`). It asserts the echo and `stage[29]==1`, `stage[30]==1` (`unittest_hvx_mm_u8i4.cpp:2507`), and uses 8 persistent CPU threads with a ring of 10 MiB-class buffers |
| `test/htp/nntr_moe_dma_plan.h` | header-only additions shared by the gtest and the host check. `nntr_two_reader_cell()` returns `f2` = `nntr_moe_dma_cell(12, …)` (`:504-508`) + `fresh=1`, `calls_per_chunk=500`, `chunks_max=8`, `min_us=1 200 000`. `nntr_two_reader_footprint_bytes(n_regions, region_bytes)`. `nntr_ddr_rate_valid(bytes, us, ceiling_gbs)`. `nntr_two_reader_verdict(gbs, ref_gbs, checksum_ok)` (≤ 85.3, ±20 %, tag). `nntr_prefetch_net_us(cold_us, reread_us, dsp_us, dsp_alone_us)` (= `(cold − reread) − (dsp − dsp_alone)`, per layer). Pure functions, no DSP call |
| `test/htp/host/two_reader_host_check.c` (new) + a block in `run_host_checks.sh` after the `dma_replay_host_check` block (`:124-138`) | §3.3. Built like `dma_replay_host_check`: the skel's `nntr_hvx_dma_probe.c` as-is against `replay_stub/` |
| `test/htp/host/replay_stub/hexkl_dma_standin.h` | `replay_stub_dma_run` (`:26-35`) also adds `n × row_size` to `replay_stub_bytes_landed` and marks the 4 KiB source pages it read in a bitmap (`replay_stub_pages`). Host only |
| `test/htp/nntr_hvx_dma_probe.c`, `test/htp/nntr_hvx.idl` | **No change.** `dma_replay` (`:386-606`) already has `fresh` rotation (`replay_src` `:238-246`), qtimer µs (`:586`) and the tag sum (`:603`). `mm_u8i4_moe_layer_timed` / `moe_set_opts` exist (IDL `:331`, `:478`) |
| consumers of a changed contract | **None.** The FastRPC contract is untouched: no stub regeneration, no `HtpComputeOps`, no quantizer format tag (`nntr_quantize_stream`), no loader check, no `NNTR_HTP_PROFILE` stage table, no `tools/htp_fc_report.py`. The IDL diff `07fb1938..df17fcdb` is one comment (`attn_m1_forward`, #146), so a stub built at this branch talks to the staged skel |
| `test/jni/Android.mk` | no change (module `:954-975` already compiles the file with the stub) |
| `docs/measurements/90-two-reader-ddr-probe.md` (new) | the handoff of §4 step 5 |

## 3. Design

### 3.1 Question (1): readers alone and together

**CPU reader.** 512 MiB heap buffer, pre-faulted: 21× the two 12 MiB
cluster L2s plus the SLC. The SLC size is not in repo docs (press figures
for SM8750 say 2 × 12 MiB L2 and 8 MiB SLC); the `CPU_CACHE` lines print
what sysfs says, and the SLC is inferred from §3.2's knee. The reader uses
t = 1, 2, 4, 8 unpinned threads, as the app's pool is unpinned. Each thread
reports `sched_getcpu()` at exit, so a 1- or 2-thread row shows whether it
ran on the 4.32 GHz prime pair (cpu 6–7) or on the 3.53 GHz cluster.

**DSP ring reader** (unchanged from the previous plan's §3.1). This is
exactly the MoE feed: `f2` (8 descriptors, 22 020 096 B per call, strided
into VTCM) through `dma_replay(workers=1, fresh=1)`. It rotates over 32
regions × 5.25 MiB = **168 MiB** of distinct DDR, runs in chunks of 500
calls (≈ 0.3 s each) until ≥ 1.2 s, and is proven landed by `res[12]` ==
`nntr_moe_dma_tag_sum`. The rate is Σbytes / Σqtimer-µs, the same
arithmetic as the `DMA_REPLAY` lines, so the reader compares directly with
the rule-34 anchor (37.3) and rule 41's in-app 0.88×. `DDR_DSP_REF` (one
`f2` 20-call cell) is the in-run tie.

**DSP HVX-direct reader.** This is the real M = 1 MoE call with the feed
off (`feed=0, rows1=1, lead=192`, the pre-#117 default D192). It is
production's direct vector read of the arena, latency-bound at 21–27 GB/s
(rules 26, 27). The loop rotates over 16 registered expert sets on both
chunks (64 regions, **336 MiB**) until ≥ 1.2 s. Rate = 22 020 096 B /
`stage[10]` (`mm`); `stage[0]` (`dsp`) and the duty cycle Σ`dsp` /
window are also printed. It answers whether the vector unit's DDR read
suffers from the CPU as much as the DMA engine does.

**Concurrency.** The CPU streamer starts, the DSP stream runs (blocking
FastRPC calls on the main thread), and the CPU stops when it returns.
`cpu_with` = CPU bytes / window, `dsp_with` = DSP bytes / DSP-side µs.
Row output: `aggregate = cpu_with + dsp_with`, `cpu_loss_pct`,
`dsp_loss_pct`.

#77's `cpu_with` 39.73 came from a **0.04 s** window next to a DSP loop
that was not reading DDR (`:479-482` of its measurement). That makes its
−41 % as unreliable as its DSP side; this sitting replaces both.

Grid: 4 `cpu_alone` + `dsp_ref` + 2 DSP alone + 4 ring pairs + 2 hvx
pairs = **13 cells, ≈ 20 s** of streaming.

### 3.2 Question (2): the bit-preserving form, CPU prefetch during the MoE call

**What it stands for.** While layer l's MoE runs on the DSP (≈ 700 µs in
the app, ≈ 600 µs isolated, rule 41), the ARM main thread spins on the
dspqueue (rule 40) and the pool's workers idle. A worker could load layer
l+1's FC weights: `in_proj` 3 × 2048² Q4_0 ≈ 7.1 MB, plus `out_proj` or
attention q/k/v/o, ≈ 10 MB per layer on average (255 MB / 22, rule 42).
The FC GEMV that follows would then find them in L2 or SLC. No arithmetic
moves, so the result is bit-identical by construction.

**Cell.** A persistent pool of 8 CPU threads (like the app's pool) plus the
main thread that issues the DSP call. There is a ring of R buffers of S MiB,
with R × S ≥ 192 MiB, so a buffer is cold when it comes round again. Per
iteration i, with buffer B = ring[i mod R]:

| cell | phase 1 (window) | phase 2 (timed) |
|---|---|---|
| `PREFETCH_COLD S` | main: one real MoE call (feed = 1, next expert set of 16); no touch | 8 threads re-read B (1/8 slice each) → `cold_us`; the call's `dsp_us` / `mm_us` = **DSP alone** |
| `PREFETCH_HOT S T` | T threads touch B (`stream_xor`); no DSP call | 8 threads re-read B → `hot_us` (the cache ceiling of staging) |
| `PREFETCH_OVERLAP S T` | main: the same MoE call; T threads touch B, released as the call is issued; touch thread j covers the slices of readers k ≡ j (mod T) | 8 threads re-read B → `reread_us`; the call's `dsp_us` / `mm_us` under the touch |

Sweep: S = 4, 10, 20, 32 MiB (the capacity knee: one cluster L2, two, +
SLC) × T = 1, 2, 7 (7 = every worker while the main thread spins).
64 iterations per cell, medians. 4 × (1 + 3 + 3) = 28 cells × 64 × ≈ 1 ms
≈ 2 s.

Printed per `PREFETCH_OVERLAP` line: `dsp_us mm_us mm_alone_us dmm_pct`,
`touch_us touch_gbs touch_late=k/64` (touch still running when the call
returned), `cold_us hot_us reread_us`,
`staged = (cold − reread) / (cold − hot)`, `staged_mib = staged × S`,
`saved_us = cold − reread`, and
`net_ms_per_token = 22 × nntr_prefetch_net_us(…) / 1000`, and `moved=`
(readers whose CPU differs from their touch partner's).

**What it answers.** Three things, all from one table:

* how much the DSP's `mm` loses to a light CPU reader (`dmm_pct`);
* how many bytes the CPU can stage in the window and keep until the
  consumer runs (`staged_mib` vs S; the knee is the usable L2 + SLC);
* whether the net per token is positive.

**Known gap.** The re-read is a plain stream, not the NEON Q4_0 GEMV. The
app's FC GEMV reads at ≈ 37–41 GB/s effective (rule 42), well below the
stream's 67.9, so part of its time is not DDR. `saved_us` is therefore the
**byte** saving, and the in-app variant (rule P's follow-up) measures the
real one with `NNTR_OP_TIME` (#150's timer).

### 3.3 Host check (before any phone time)

`two_reader_host_check.c` runs the skel's `nntr_hvx_dma_replay` on the host
with `nntr_two_reader_cell()` and asserts:

1. **Byte accounting.** For `calls = 24`, bytes landed ==
   `res[2] × res[1]` == 24 × 22 020 096 = 528 482 304.
2. **Footprint.** ≥ 168 MiB of distinct 4 KiB source pages (≥ 121 MiB
   with a 128 MiB arena, `n_regions = 23`), equal to
   `nntr_two_reader_footprint_bytes`.
3. **Tag.** `res[12]` == `nntr_moe_dma_tag_sum` for `calls ∈ {1, 20, 500}`
   with `fresh=1`, and call 500's tag ≠ call 499's (a landing one call late
   is caught).
4. **Bounds (rule 12).** `nntr_ddr_rate_valid` rejects #77's pair
   (85 362 475 008 B / 26 140 µs = 3265.6) and 85.4, and accepts 67.9 and
   37.3. The verdict rejects the ring at `ref × 1.21` and accepts it at
   `ref × 1.19`.
5. **Projection arithmetic.** `nntr_prefetch_net_us(250, 150, 640, 600)`
   = 60, and 22 × 60 / 1000 = 1.32 ms/token.

Pass lines as in §1. The device's timing is not modelled.

### 3.4 Question (3): the decision rules the supervisor applies

Inputs come from the **cold** probe run. The warm one is a check: if the
two disagree by > 10 % on a term, that is noted and the lower-benefit value
is used.

**Rule P: prefetch overlap, bit-preserving, "now".** File the in-app
variant (a pool task that touches layer l+1's FC weight rows, partitioned
like the GEMV that follows, during the MoE wait) **iff all three hold at
S = 10 and the best T**:

* P1: `dmm_pct` ≤ **5 %**. The DSP loses ≤ ≈ 30 µs/layer = 0.7 ms/token
  at the in-app 670 µs `mm`.
* P2: `staged` ≥ **0.5**. Half the bytes survive until the consumer, so
  the capacity is real, not a race.
* P3: `net_ms_per_token` ≥ **1.0**. That is ≈ +3.8 % at 37.77 tok/s,
  outside the ±2 % run spread of a same-sitting A/B.

The in-app variant's own gates are then the usual ones: `bit_identical=1`
MoE dumps, text ≡ A, decode ≥ A at G 64 / 512 / 1024, prefill −5 %.

If P1 fails but the knee is large, the table also gives the T and S at
which P holds, if any. If none holds, (b) closes in its bit-preserving
form with the numbers.

**Rule S: split arithmetic (the CPU computes experts). Parked behind track
(c).** A CPU expert in Q4_0 is not bit-identical to the DSP's `QS4CX_WH`
path, so it is a (c)-type change that the user has not opened. The sitting
only records the three numbers the user will need:

* whether **the readers add up**: `aggregate(8) ≥ 1.15 × max(cpu_alone(8),
  dsp_ring_alone)` (HeteroLLM's pair is ≈ 1.35×);
* the per-layer projection `T0 = 22.02 MB / (0.88 × dsp_ring_alone)` vs
  `T1 = 22.02 MB / (cpu_with(t) + 0.88 × dsp_with(t)) + 14 µs` (dspq sync)
  for t = 4, 8, as `22 × (T0 − T1)` ms/token;
* the contract's Q11 headline `aggregate ≥ 45 GB/s`, kept for continuity.

S is worth tabling to the user only if the readers add up **and** the
projection is ≥ 2 ms/token. Even then it waits for (c) to pass.

### 3.5 Rejected alternatives

* **A new IDL entry for an HVX `vmem` stream over the arena** (the
  "pure" direct reader). It would need an IDL change, stub regeneration and
  a new skel for the sitting, so it could not ride the staged set. Its
  number would also not be a reader production uses: the only direct DSP
  reader is the D192 GEMV, which this plan runs as-is. Rule 27 already
  gives the plain vector band.
* **Linking nntrainer's NEON Q4_0 GEMV into the probe** to time a real
  consumer instead of a stream re-read. It would pull `libnntrainer` into
  a FastRPC probe for a number the in-app variant measures anyway, with
  `NNTR_OP_TIME`, on the real graph.
* **`prfm pldl2keep` instead of loads for the touch.** Prefetch hints can
  be dropped under load and their completion is invisible. Loads are
  measurable and are what a first in-app version would use.
* **Fixing `dma_probe` and keeping it** (as the previous plan also
  rejected): rule 28 already bars it as a ceiling. The replay has a landing
  proof. The legacy "why 3265" diagnostic pair is dropped: it does not
  inform (b).

## 4. Steps

Rungs follow `.claude/skills/hexagon-gates`. Everything runs on the
workstation. The branch touches only
`test/unittest/unittest_hvx_dma_probe.cpp`, `test/htp/nntr_moe_dma_plan.h`,
`test/htp/host/**` and `docs/measurements/90-*.md`. **Implementer budget
≈ 1 h** (steps 0–4), **device ≈ 20 min** (step 5).

| step | what | gate |
|---|---|---|
| 0 | `nntr_moe_dma_plan.h` additions; `hexkl_dma_standin.h` counter and page bitmap; `two_reader_host_check.c`; the `run_host_checks.sh` block | rung 1: `TWO READER CELL SOUND (bytes=528482304 footprint=168 MiB)`, `TWO READER BOUNDS OK (…)`, `SKEL REPLAY MATCHES TAG SIMULATOR (16 cells)` unchanged, `ALL CHECKS PASS`, `WORKER POOL LANES OK` (the baseline at `df17fcdb` passes, run while planning); `clang-format-14` on changed lines |
| 1 | `TwoReaderDdr` rewrite (§3.1) + `PrefetchOverlap` (§3.2). `DmaProbeShapes` and `MoeChunkReplay` untouched: `git diff` has no hunk outside `:334-452` except the new test after it | rung 1: `tools/htp_syntax_check.sh` rc 0; `ninja -C build` green (the gtest is not in meson; the syntax check and step 3 are its compile gates) |
| 2 | `./test/htp/build.sh` as a compile proof (the skel includes the header). **Not pushed**: the sitting uses the staged skel | rung 2: `UNDEFINED SYMBOLS OK (<n> runtime imports)`; `git diff --stat origin/htp_moe -- test/htp/*.c test/htp/nntr_hvx.idl nntrainer/tensor/htp_backend` is empty |
| 3 | `(cd test/jni && $ANDROID_NDK/ndk-build … unittest_hvx_dma_probe -j8)`; copy into `$W/set/`; md5 | rung 3: binary exists; md5 in the handoff |
| 4 | **Stage the set** `W=/local/mnt/workspace/htp_moe/90`: `set/` = the #150 set's `libnntr_hvx_skel.so` (`37468a7f…`), `libc++_shared.so`, `libsdkl.so`, `prompt512.txt`, the #150 T app (`nntrainer_causallm` `db0c4bc3…`, `libnntrainer.so` `178b6e12…`, `libcausallm_core.so` `2b354511…`, `libccapi-nntrainer.so` `e3f0a124…`; code `25c8eb5c` = dspq default + `NNTR_OP_TIME`), and the new `unittest_hvx_dma_probe`; `md5.txt`. Write `docs/measurements/90-two-reader-ddr-probe.md` from `hexagon-handoff` with §4.1 and the empty tables of §4.2 | `LC_ALL=C md5sum -c` 0 mismatches; `strings set/libnntrainer.so \| grep -c 'dspq: on'` = 1; no `libcdsprpc*` in `$W` |
| **5 (device, unavoidable)** | the sitting, §4.1, on `R3CY10WM83Y`, run by whoever holds the phone (the orchestrator, per its 2026-09-29 note) | the §1 gates |
| 6 | PR into `htp_moe`: one `[test]` commit (probe + host check) and one docs commit (plan + handoff + results). The supervisor folds §3.4's verdicts into LEDGER / BENCHMARK (§6) | review |

### 4.1 Handoff (the device step): about 20 min, one variant

The sitting has **one variant**. **A** (control, first) is the #150 T set
unchanged, `NNTR_NUM_THREADS=8 NNTR_OP_TIME=1`. The dspq default gives one
`[HTP] dspq: on` line and the `[OP-TIME]` table at the end: the in-app FC
ms/token is the ceiling for rule P's saving. It runs as a full E2E, prompt
512, G = 64 / 512 / 1024, once each. Every other cell is a probe cell on
the same skel.

```
source tools/htp/env.sh; W=/local/mnt/workspace/htp_moe/90; mkdir -p $W/logs
S=R3CY10WM83Y; D=/data/local/tmp/nntrainer/causallm/s90; M=../models/q40-qs4cx-wh
therm() { adb -s $S shell dumpsys battery | grep -E 'level|temperature'; adb -s $S shell cat /sys/class/thermal/thermal_zone0/temp; }
probe() { adb -s $S shell "cd $D && md5sum libnntr_hvx_skel.so unittest_hvx_dma_probe && \
  LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_dma_probe \
  --gtest_filter='*MoeChunkReplay*:*TwoReaderDdr*:*PrefetchOverlap*'" 2>&1 \
  | tee $W/logs/probe_$1.log | grep -E 'md5|^CPU_CACHE|^DDR_|^PREFETCH_|INVALID|DMA_REPLAY_X name=f2 |PASSED|FAILED'; }
```

0. `adb devices` (record serial); `therm | tee -a $W/logs/therm.log`;
   screen off, charger in, cool start. (1 min)
1. Install: `adb -s $S shell mkdir -p $D`; `adb -s $S push $W/set/. $D/`;
   `chmod 755` the two binaries; the device md5 must equal `md5.txt` line
   by line. Model reused (`nntr_lfm2_8b_a1b_q40_arm.bin` 4316133120 B).
   Re-apply #141's config block (greedy, `bad_word_ids [124900]`,
   `moe_engine htp`). (3 min)
2. **Probe cold**: `probe cold` (≈ 1.5 min: fixture fill ≈ 3 s per test,
   `MoeChunkReplay` gives the rule-30/34 anchor and `f2`, then 13 stream
   cells and 28 prefetch cells). Checks: `grep -c INVALID` = 0;
   `grep -c '^DDR_TWO_READER'` = 6; `grep -c '^PREFETCH_OVERLAP'` = 12;
   `FAILED` absent. `therm`.
3. **A, E2E**: G = 64, 512, 1024 once each, as #141's `run()` with
   `NNTR_OP_TIME=1`. Logs `A_g{64,512,1024}.log`; each shows
   `dspq: on` once and `dspq: close calls=N served=N bad=0`. (≈ 6 min)
4. **Probe warm**: `probe warm`, same checks. `therm`. (≈ 1.5 min)
5. Pull nothing else. Record the serial, battery and temperatures in Notes.

Stop conditions: a skel md5 ≠ `37468a7f…`; any `err=0x8000040e` (stale
skel, rule 3); `DDR_DSP_REF` > ±15 % from the same log's anchor
`DMA_REPLAY workers=1 load=0 pace=0` (note which one drifted, rule 30, and
go on).

### 4.2 Empty tables the handoff carries

**(1) readers**, cold | warm. References: #77 CPU 67.90 (other unit), ring
anchor 37.3, in-app ring 33.0, D192 `mm` 922.7–931.9 → 23.6–23.9 GB/s.

| t | cpu_alone | ring: cpu_with / dsp_with / aggregate / cpu_loss % / dsp_loss % | hvx: cpu_with / dsp_with / aggregate |
|---|---|---|---|
| 1 | | | — |
| 2 | | | |
| 4 | | | — |
| 8 | | | |
| DSP alone | — | ring `dsp_alone` = … (`dsp_ref` …, anchor …) | hvx `dsp_alone` = … |

**(2) prefetch**, cold (warm in a copy). `mm_alone` from `PREFETCH_COLD`.

| S MiB | cold_us | T | hot_us | reread_us | staged / staged_mib | dmm_pct | touch_late | net ms/token |
|---|---|---|---|---|---|---|---|---|
| 4 / 10 / 20 / 32 | | 1 / 2 / 7 | | | | | | |

**A**: decode tok/s (last 64) at G 64 / 512 / 1024, prefill, text hash vs
#150 T8, `[OP-TIME]` FC ms/token at G = 512.

## 5. Risks

| risk | how the handoff table makes it visible |
|---|---|
| **DMA rate differs by unit and by sitting** (rules 30/32/34: 31.2 vs 37.3; ≈ 22 % drift once) | `MoeChunkReplay`'s anchor cell and `DDR_DSP_REF` run in the same log as the stream. Ring numbers are read as ratios to both; > ±20 % from `dsp_ref` is INVALID |
| **DVFS / bus votes.** The DSP stream may raise the DDR clock and the CPU threads the CPU clock; 1–2 thread rows are the most exposed | `cpus=` shows the cluster; t = 1/2/4/8 bracket it; cold vs warm triplets side by side. A > 10 % gap between them is written down, not averaged |
| **Thermal drift across the sitting** | `therm` at 0, after the cold probe, after A, after the warm probe. The probe runs at both ends of the E2E block |
| **Stale skel / stale gtest** | md5 line first in each probe log; `err=0x8000040e` is a stop. The IDL is comment-only different from the staged skel's source (§2) |
| **Address space.** 2 × 256 MiB arena chunks + 64 arena handles + ≤ 704 MiB of CPU heap (512 stream + 192 ring) | The fixture falls back to 128 MiB chunks (`chunk_bytes=` printed; the footprint check accepts 121 MiB, the hvx rotation drops to 10 sets = 210 MiB). The CPU heap is ARM-side and outside the DSP's 32-bit budget |
| **The gtest's MoE window (≈ 600 µs isolated) is shorter than the app's (≈ 700 µs)** | `touch_late` counts overruns. P is read at T where `touch_late` ≈ 0, and the app's longer window can only help |
| **The stream re-read is not the GEMV** (§3.2) | `saved_us` is labelled a byte saving. Rule P's follow-up is an in-app variant with `NNTR_OP_TIME`. A's `[OP-TIME]` FC ms/token in this sitting bounds what P can win |
| **Host-vs-device gap the plan cannot close.** The host stub lands descriptors instantly and models no DDR, cache or contention | The host check proves only bytes, footprint, tag and arithmetic. Every rate is the sitting's |
| **Unpinned threads migrate between phases** | Each prefetch line prints how many reader threads ended on a different CPU from their touch partner (`moved=`); a high `moved` with low `staged` is read as placement, not capacity |

## 6. Docs to update (after the sitting)

* **`docs/htp_moe/BENCHMARK.md`.** Replace the ④ line at `:371` ("DSP
  side 3265 / 3113 … invalid (#90)") with the cold / warm reader table
  (t = 1/2/4/8, ring and hvx, aggregate, losses, unit, skel md5). Add a
  "④b prefetch overlap (#90)" side table: S × T, `dmm_pct`, `staged`,
  net ms/token. Add A's cells as a sitting row, not a new "now" unless it
  is the first dspq-default G=1024 reading (BENCHMARK `:140` has
  (35.17) as a FastRPC placeholder).
* **`docs/htp_moe/LEDGER.md`.**
  * Item ④ (`:1018`, `:1053`) → **measured**, with the S-rule numbers.
  * Item ⑫ (`:1061`): its precondition is now "(c) opened and S's
    projection ≥ 2 ms/token" instead of "④ > 45 GB/s".
  * A new item for rule P's in-app prefetch variant if P holds, otherwise
    a closed line.
  * Rule 12 is amended with the physical bound (85.3 GB/s), and #77's
    `cpu_with` is marked as a 0.04 s window.
  * A new rule if the DSP's ring or HVX read loses more than the CPU under
    contention, or if a light CPU reader costs the DSP ≈ 0.
  * ㉘'s "real 'more readers' lever is ⑫" sentence gets the numbers.
* **Contract §3.2 / Q11**: the supervisor hands the user rule S's three
  numbers. The decision stays the user's.

## 7. Not done here

* No in-app prefetch code (rule P's follow-up issue), no split code, no
  DSP source or IDL change.
* No fix of `dma_probe`. `DmaProbeShapes` stays as is, with rule 28 as
  its caveat.
* No second unit. `R3CY205ZMND`'s ring is 31.2 (rule 34), and a P verdict
  transfers as a ratio, not as µs.
