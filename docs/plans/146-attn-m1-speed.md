# 146 — Speed up the resident m=1 attention (ATTN_M1)

Issue: dlwlzzero/nntrainer#146 (p1, LEDGER ㉗, roadmap step 7b). Read against
`htp_moe` @ `bb845426` (cycle 20, 2026-09-28). Kernel `hvx/hvx_attn_m1_f32.c`,
spec `attn_m1_det.h` (#81, PR #126), wired as the `ATTN_M1` slot of the
per-token entry (#130).

**Two corrections to the issue body, both checked against the code.**
(1) The bytes per layer per token are `2 × n_kv × head_dim × L × 4` =
4096 · L B, so **4.19 MB at pos 1024**. The 2.1 MB in the issue is the pos-512
figure. At about 30 GB/s the f32 byte floor at pos 1023 is **≈ 140 µs, not 70**.
The conclusion still holds: 140 µs is about 8× below the measured ≈ 1090 µs on
the DSP side.
(2) There is no horizontal reduction in the score loop. Lanes are positions
(`Kt [d][max_seq]`), so the only trees are the two five-step rotate trees per
q head (`reduce_max_sf` / `reduce_sum_sf`, `hvx_attn_m1_f32.c:197-208`).
Those trees are negligible.

## 1. Goal and gate

**Goal (issue).** Cut `ATTN_M1` from ≈ 1.05 µs per KV position per layer to
a cost that stops dominating the per-token entry at G 512 / 1024. Keep the
`_det` bit gate.

**Gate, measurable:**

| # | check | where | pass |
|---|---|---|---|
| G1 | `ATTN_M1_FIELD pos=1023 us=` (warm, host-timed, median of 10) | `unittest_hvx_attn --gtest_filter='HvxAttnM1.*'`, `PerLayerCost` (`test/unittest/unittest_hvx_attn.cpp:1036-1066`) | **≤ 400 µs** (today 1183.38, #130's sitting). See the revision rule below |
| G2 | device bit identity | same run, `MatchesDetSpecBitExact` (`:974-1000`) + `AppendChainEqualsBulk` | `out bad=0` at L = 1 / 63 / 64 / 65 / 512 / 1024 and `append_chain bad=0`. `bad_stats` must equal the reference skel's value at each L: today 1 at L = 63 / 512 / 1024 (the subnormal row, rule 37, #137). The plan must add no new mismatch |
| G3 | host bit identity | `bash test/htp/host/run_host_checks.sh` | `ATTN M1 BIT-IDENTICAL` for every shape row (the new rows of §4 step 2 included), pools of 0 / 3 / 7 workers byte-equal, plus `ALL CHECKS PASS`, `WORKER POOL LANES OK` and the new `ATTN M1 PHASES OK` |
| G4 | same-sitting E2E A/B (handoff, §4 step 7) | `nntrainer_causallm`, prompt 512, G 64 / 512 / 1024, ×2 | C's decode above B's at **G 512 and 1024**, outside B's own two-run spread. Decode PPL (`NNTR_PPL_DECODE`, forced on A's continuation) of C ≤ A + 2 %. Text: C's is pasted next to A's and B's and the user marks `text approved`. Because the kernel is bit-identical, **C's tokens must equal B's byte for byte**. A C ≠ B difference is a device bit-identity failure in the model, not a text-approval question. Prefill of C ≥ −5 % of A (ATTN_M1 is not on the prefill path, so none is expected) |
| G5 | in-model cost | one `NNTR_HTP_PROFILE=2` run each of B and C at G=64 (not for tok/s) | `ATTN_M1=` pcycles per op of C ≤ 0.4 × B's (today 1204826 at pos 512–575) |

**Standing gates:** prefill ≥ −5 % of variant A of the same sitting. Text is
compared with A and with the CPU `q40` run, and the user approves it (contract
§1 as amended 2026-09-28, rule 39: the per-token entry cannot be
text-identical to the CPU, so PPL + text + approval decide).

**Revision rule for G1 (decided after the §4 step 4 ride-along, written into
the PR).** G1 is host-timed, so it includes one FastRPC round trip (≈ 82–92 µs
per call in #130's sitting, rule 35). The model never pays that round trip,
because the graph entry calls `hvx_attn_m1_forward` directly
(`hmx/hexkl_graph.c:187`). The new phase line reports `dsp_us`. If
`us − dsp_us` ≥ 100 µs at pos 1023, G1 is restated as `dsp_us ≤ 300` at pos
1023 (the same 400 minus the measured transport).

If the step-4 split shows that scores + PV already read ≥ 20 GB/s (4.19 MB /
(scores+PV time) at pos 1023), the kernel is at its f32 byte floor. In that
case the achieved number becomes the gate, with the reason recorded, and the
fp16 cache (§3.4) is filed as the next issue. The stretch expectation from
§3.2 is ≈ 250–320 µs host-timed.

## 2. Where it lives

| file | what changes | refs (verified @ `bb845426`) |
|---|---|---|
| `nntrainer/tensor/htp_backend/hvx/hvx_attn_m1_f32.c` | phase counters (§3.1). O1: constant-shape inner kernel. O3: `(kv head, q-head pair)` units and per-q-head scratch rows. O4: `l2fetch` of the next Kt strip / V rows | `attn_head` `:211-299` (scores `:227-250`, softmax `:252-274`, PV `:276-298`), `forward_unit` `:302-307`, `hvx_attn_m1_forward` `:309-337` (append `:321-325`, pool run `:335`), `append_head` `:143-156`, `MAX_GQA` / `MAX_HD_VEC` `:57-59`, scratch lane `:223` |
| `nntrainer/tensor/htp_backend/hvx/hvx_attn_m1_f32.h` | header comment: THREADS paragraph (units are no longer one per kv head after O3), the phase-word contract | THREADS `:36-41`, ctx struct `:74-86`, `hvx_attn_m1_forward` `:139-141` (signature **unchanged**) |
| `nntrainer/tensor/attn_m1_det.h` | **arithmetic unchanged** in every step of this plan. Adds one clearly separate block, `ATTN_M1_PROF_*` word indices and `ATTN_M1_PROF_WORDS`. This is the only header the kernel, the skel entry, the host check and the gtest all see (the gtest's include path has `htp_backend/` but not `hvx/`, `test/jni/Android.mk:1018-1020`) | next to `ATTN_M1_DET_LANES` `:75` |
| `test/htp/nntr_hvx_attn_m1.c` | the validator accepts `statsLen ∈ {0, 2·n_q, 2·n_q + ATTN_M1_PROF_WORDS}` and copies the words out | `:99-125`, validator `:113-122` |
| `test/htp/nntr_hvx.idl` | **comment only**: the stats sequence may carry the phase words. The signature is unchanged, so the generated stub/skel must be byte-identical. Check with md5 of `test/htp/generated/*` and `nntrainer/tensor/htp_backend/generated/nntr_hvx_stub.c` (`generate_stub.sh`) before and after | `:580-591` |
| `test/htp/host/attn_m1_host_check.c`, `run_host_checks.sh` | shape rows `(n_kv, gqa, hd)` = (8, 4, 64) [LFM2.5, the specialised path], (1, 2, 64) [the hd64 fixture], (2, 3, 32) [generic path]. Phase-word check. `-I stub` already brings `HAP_perf.h` (monotonic pcycle stub) | shape enum `:68`, create `:184`, forward calls `:221`, `:288-294`, `:323`, `:369-385`; script `:214-228` |
| `test/unittest/unittest_hvx_attn.cpp` | `PerLayerCost` also passes a stats buffer of `2·n_q + ATTN_M1_PROF_WORDS` and prints `ATTN_M1_PHASE …` (median per word). It gains a **cold** line: 6 layers at max_seq 2048, rotating the layer per call so every call's 4.19 MB slab was evicted by the other five, as in the model | `:1036-1066`, `m1_run` `:899-914` |

**Consumers checked and not moving.** `hmx/hexkl_graph.c:180-190` (the
`ATTN_M1` slot passes `stats = NULL`, so it records nothing and the in-model
cost is read through the existing per-kind pcycles `:508-510`, printed by
`htp_compute_ops.cpp:785-790` as `ATTN_M1=`). `HtpComputeOps` does not call
the kernel. `nntr_quantize_stream`'s format tag and the loader check are
untouched, because no weight layout changes (the KV cache is runtime state and
its layout stays `Kt [d][max_seq]` / `V [p][d]`). `NNTR_HTP_PROFILE` stage
tables and `tools/htp_fc_report.py` are untouched. The phase words are a gtest
channel, not a stage.

## 3. Design

### 3.1 First: a per-phase pcycle split in the existing timed path

There is no new IDL method and no new build flavour. `attn_m1_forward`'s
`rout sequence<float> stats` already exists as the debug channel
("production passes it empty and pays nothing"). A caller that passes
`2·n_q + ATTN_M1_PROF_WORDS` floats receives the (m, l) pairs followed by
these uint32 words, bit-copied with `memcpy` so there is no 2^24 float bound:

| word | meaning | taken by |
|---|---|---|
| `APPEND` | pcycles of the k/v append (`:321-325`, the scalar strided scatter into Kt) | caller |
| `POOL` | pcycles from just before `hvx_worker_pool_run` to its return | caller |
| `LANES` | n = min(units, workers + 1) | caller |
| `SCORES` | Σ over heads of the score loop incl. the running vmax | units |
| `SOFTMAX` | Σ of max tree + exp_det + lane sum + recip_det | units |
| `PV` | Σ of PV + `o · r` | units |
| `BUSY_MAX` | max over units of (end − start) | units → caller |
| `START_MAX` | max over units of (start − POOL's t0): dispatch skew | units → caller |
| `CALL_QT` | qtimer ticks (19.2 MHz) of the whole forward, giving `dsp_us` and the effective pcycle clock of that call (DVFS visible) | caller |

Each unit writes only its own slot of a small per-unit array on the caller's
stack. The caller reduces the slots after `pool_run`, so no atomics are
needed. Pool dispatch + merge = `POOL − BUSY_MAX`. Lane imbalance =
`BUSY_MAX / ((SCORES+SOFTMAX+PV)/LANES)`. The timestamps are taken only when
the words are requested (`job->prof != NULL`), so the graph path pays one
predicted branch per phase. The pcycle counter is core-wide, so unit and
caller readings share one clock.

### 3.2 What is likely dominant — static reading of the code and its assembly

I compiled `hvx_attn_m1_f32.c` with the skel's exact flags (`hexagon-clang
19.0.04 -mv79 -mhvx -mhvx-length=128B -O3`, `test/htp/build.sh:93-95`) to
`-S` and counted packets per KV position per kv head (LFM2.5: gqa 4, head_dim
64, nvec 2):

| phase | today | cause in the asm | constant-shape prototype (O1) |
|---|---|---|---|
| scores | ≈ 28.7 packets/pos | `gqa` is a runtime trip count, so `acc[MAX_GQA]` lives **on the stack**: every (d, g) step is `vmem` load acc, `vmpy`, `vadd`, convert, `vmem` store acc (`r18 = add(r29,#1152)` …). Plus a scalar load + `vsplat` of q per (d, g), ≈ 14 packets per d per block | **≈ 13.6**: acc[4] in registers, d unrolled by 4, 26 packets per 16 MACs |
| PV | ≈ 42 packets/pos | `o[MAX_GQA][MAX_HD_VEC]` **on the stack** (`r15 = add(r29,#3200)`), load/store per (p, g, i). The V row is **re-loaded once per g** (the stack stores alias it), ≈ 10 packets per (p, g) | **≈ 13**: o[4][2] in registers, V loaded once per p |
| softmax | ≈ 2 packets/pos | exp_det ≈ 30 ops per vector, 4 vectors per 32 positions, pipelined | ≈ 2 |
| **total** | **≈ 73** | | **≈ 29 (2.5×)** |

The floor is the IEEE multiply. Every `Q6_Vsf_vmpy`/`vadd` pair lowers to
`vmpy(.sf,.sf)→qf32`, `vadd(qf32,.sf)`, `.sf=.qf32`, and the prototype packs
at most one `vmpy` per packet. So 8 + 8 MACs per position gives **≈ 19
packets/pos** as the compute floor.

Cross-check against silicon. 8 units over 6 lanes (`n_hvx = 6`,
`hvx_add_f32.c:113-118`) means lanes 0 and 1 run 2 kv heads each, so the
critical path at pos 1023 is 2 × 73 × 1024 ≈ 150 k packets. The DSP side
measured ≈ 1090 µs (1183 host-timed minus ≈ 90 transport), which is ≈ 1.9 M
pcycles at 1.74 GHz, or **≈ 12.7 pcycles per packet**. Plan 105 read ≈ 6.3
pcycles per packet per lane with 6 lanes busy. The factor-2 excess over pure
issue fits the stack store→load round trips and the Kt misses.

Kt is read as 64 rows × 128 B at a stride of `max_seq × 4` = 4–8 KB per
block, which is prefetch-hostile. The in-model cost (1204826 pcycles at pos
512–575 ≈ 692 µs, ≈ 1.27 µs/pos) is ≈ 45 % above the gtest's warm ≈ 0.88
µs/pos at pos 511. That gap is the cold-cache share. The ride-along's cold
line measures it directly.

**Expected order of the terms (to be confirmed by §3.1):**
1. PV, then scores, both inflated ≈ 2–3× by stack-resident accumulators.
2. Lane imbalance: 8 units on 6 lanes, so the critical path is 2 heads where
   8/6 = 1.33 is possible (−33 %).
3. Memory latency on the strided Kt stream, larger in the model (cold) than
   in the warm gtest.
4. Append (`:321-325`, 512 scalar stores at 4–8 KB stride on the caller
   before the pool starts). Its size is unknown and it is measured on its own.
5. exp_det, recip_det, trees and pool dispatch, each expected < 5 %.

### 3.3 The optimisations, in order of expected gain

Every one keeps **each q head's operation sequence per lane exactly as
`attn_m1_det.h` writes it**, so `attn_m1_det.h`'s arithmetic does not change.
Each is gated by `ATTN M1 BIT-IDENTICAL` on the host (the real source on
`hvx_emu`, pools 0 / 3 / 7) and by `HvxAttnM1.*` on silicon.

One caveat binds all of them. `hvx_emu` emulates the *intrinsics* as IEEE,
but the compiler lowers them to `qf32` chains and may fold conversions
differently in restructured code (the current exp_det already shows
`vmpy(qf32,qf32)` without an intermediate `.sf=` convert). The device gtest
is therefore the only proof. That is why the ride-along (§4 step 4) runs G2
on every candidate skel before any E2E sitting.

* **O1 — constant-shape inner kernel (expected ×2–2.5 on the kernel).**
  `attn_head` becomes an `always_inline` body taking `gqa` / `head_dim` as
  parameters. The dispatcher calls it with literal `(4, 64)`, or `(2, 64)`
  after O3, and falls back to the runtime-shape call for any other shape. The
  compiler then fully unrolls the g / i loops and keeps `acc`, `vmax` and `o`
  in registers. The V row is loaded once per p into locals before the g loop.
  Same operations in the same order per lane, so no spec change. The
  prototype (scratch copy, not committed) compiled `-Werror` clean with no
  vector stack traffic inside either loop.
* **O3 — balance: units = (kv head, q-head pair) (expected −25 % of the
  critical path).** With 16 units on 6 lanes the busiest lane runs 3 half
  heads = 1.5 heads instead of 2. The two units of one kv head are adjacent
  indices, so they run concurrently and share the Kt/V slab in L2. The
  probability scratch is indexed by q head (`hq · max_seq`) instead of by
  lane (`:223`). The size stays `n_kv · gqa · max_seq`, with no new
  allocation. No reduction crosses a unit, so the output is byte-equal at any
  worker count (the host check's 0 / 3 / 7 rows). Only when `gqa` is even:
  for LFM2.5, gqa 4 gives 2 pairs; for the hd64 fixture, gqa 2 gives 1 pair.
  **Rejected alternative:** a three-job phase split, (h, block chunk) for
  scores and exp, then a caller max-merge, then (h, head_dim vector) for
  sum + PV. It reaches 1.33 heads (−33 %), but it costs three pool barriers
  per call plus merge code, for ≈ 8 % more than O3. The sum over blocks must
  stay sequential per lane, which is why a position split cannot touch it.
  That alternative is reconsidered only if §3.1 shows `POOL − BUSY_MAX` below
  ≈ 2 µs per barrier.
* **O4 — `l2fetch` lead on Kt and V (expected: most of the cold − warm gap;
  sized by the cold line).** One 2D `Q6_l2fetch_AP` per few blocks: stride
  `max_seq · 4` (8192 at 2048, inside the 16-bit field), width `128 · nb`,
  height `head_dim`, one strip ahead. V is contiguous, so one 1D fetch of the
  next `nb · 32` rows. This reuses the pattern of
  `hvx_gemm_u8i4_wh_prefetch` (`hvx/hvx_gemm_u8i4_wh.c:32-50`, `#if
  defined(__hexagon__)`, a no-op on the host) and LEDGER rule 26 (DDR latency,
  not bandwidth, bounds direct HVX reads). This is the attention analogue of
  "DMA hidden behind compute" (doc 45 §3.2): ATTN_M1 has no DMA. It is a
  memory hint only, so there is no arithmetic change. **Rejected
  alternative:** a tiled Kt layout `[blk][d][32]`, which would make each
  block 8 KB contiguous. The 2D fetch covers the stride without touching the
  append, the kv_append seeding or the cache-byte comparison in the host
  check.
* **O2 — two position blocks per d pass in the score loop (expected ≈ −10 %
  of the total).** 8 accumulators and 2 Kt loads per d share the 4 q splats,
  so there are half as many scalar loads and `vsplat`s. This is done only if,
  after O1, `SCORES` is still more than 1.3× `PV` (they are equal in MAC
  count).
* **Append (only if `APPEND` ≥ 3 % of the call).** Move `append_head` into
  the unit that owns the kv head, before its score loop. It is the same
  store, done by the thread that then reads it, and it runs in parallel
  across heads.

With O1 + O3 the estimate at pos 1023 is 29 packets × 1.5 heads × 1024 ≈
45 k packets on the critical lane. At 6.3–9 pcycles per packet that is
≈ 160–230 µs DSP, next to the 140 µs f32 byte floor, so O4 decides whether
the reads hide. Host-timed (+ ≈ 90) that gives **≈ 250–320 µs**, inside G1.

In the model, the six-kind entry at G=512 (mean pos ≈ 768) pays ≈ 6 × 768 ×
1.27 µs ≈ 5.9 ms/token for attention today. Cutting that 2.5–3× saves ≈
3.5–4 ms/token (≈ +10–12 % decode vs B). At G=1024 it saves ≈ 5 ms/token
(≈ +14 %). These are expectations, not gates.

### 3.4 The fp16 cache — sized, deferred

It halves the bytes (4.19 → 2.10 MB per layer per token at pos 1024, so a
140 → 70 µs floor) and the heap footprint (48 → 24 MiB at max_seq 2048, so
max_seq 4096 fits in today's 48 MiB). It is worth doing only once §3.1
shows the kernel at ≥ 60 % of the DDR rate. That is not the case today (4.19
MB in ≈ 1090 µs ≈ 3.8 GB/s).

It needs:
* an `sf → hf` RNE step at append, spec-first, with the widening `hf → sf`
  exact in the score and PV loops;
* a device confirmation of the conversion (hvx_impl's rule: `Vhf_equals_Vqf16`
  rounds badly);
* `-mhvx-ieee-fp`, which toolchain 19 requires for fp16 intrinsics and which
  `test/htp/build.sh` does not pass today. Adding it changes the codegen of
  every skel source, which is its own risk.

The values change, so the gate becomes decode PPL, not bit identity to
today's kernel. One upside: the CPU's cache is already fp16 (LEDGER ⑨, the
39 dB attention stretch), so an RNE fp16 cache would move the DSP toward the
CPU's rounding. **Not in this issue.** The supervisor files it when the phase
line says bytes are the bound.

## 4. Steps

Every step is host-gated. Rung numbers are from `.claude/skills/hexagon-gates`.

1. **Phase counters (§3.1).** Changes: the kernel, the `attn_m1_det.h` word
   block, the skel validator, the IDL comment, the host check
   (`ATTN M1 PHASES OK`: every word present, `POOL ≥ BUSY_MAX`, and `out`
   byte-equal with and without the words requested at every L and pool), and
   the gtest (`ATTN_M1_PHASE pos=… warm|cold append= scores= softmax= pv=
   busy_max= start_max= pool= lanes= dsp_us= mhz=` plus the new cold
   `ATTN_M1_FIELD` line).
   **Gates:** rung 0; rung 1 (`ATTN M1 BIT-IDENTICAL`, `ATTN M1 PHASES OK`,
   `ALL CHECKS PASS`); rung 2 (`UNDEFINED SYMBOLS OK`, generated stub md5
   unchanged); rung 3 for `unittest_hvx_attn` only. Save the skel as
   `libnntr_hvx_skel.prof.so`.
2. **O1.** Add the shape rows (8, 4, 64), (1, 2, 64) and (2, 3, 32) to the
   host check. Record in the PR the `-S` grep showing no `vmem(r29`/`r30`
   inside the score and PV loops of the specialised path.
   **Gates:** rung 0–2. Save the skel as `.o1.so`.
3. **O3** on top of O1. **Gates:** rung 0–2, with pools 0 / 3 / 7 byte-equal
   on all three shape rows. Save the skel as `.o13.so`.
4. **DEVICE: ride-along R1 (unavoidable; the first device step, cheap).**
   It joins the next sitting's handoff as a block after its E2E cells:
   `unittest_hvx_attn --gtest_filter='HvxAttnM1.*'` with the skels
   `prof → o1 → o13 → prof` (≈ 3 min each, ≈ 12 min). The repeated `prof`
   reads the drift, so the same skel is at both ends. Then the sitting's own
   skel is restored. It records per skel: G2's `bad` / `bad_stats`, warm and
   cold `ATTN_M1_FIELD`, and the `ATTN_M1_PHASE` lines. `0x8000040e` or a
   rejected stats length means a stale skel.
   **Read:**
   * **(a)** Does the phase ranking match §3.2? If not, re-plan before step 5.
   * **(b)** Keep O3 only if `o13` ≤ 0.9 × `o1` warm.
   * **(c)** O4 if cold/warm ≥ 1.2 or if (scores+PV) per position is above
     1.5× the §3.2 packet estimate.
   * **(d)** O2 per its rule, and append per its rule.
   * **(e)** G1 revision per §1.
5. **O4 / O2 / append as step 4 dictates.** **Gates:** rung 0–2 each.
   O4 is a no-op on the host, so its only device check is G2 in step 7's
   ride-along. Then rung 3 (full app `build_android.sh --htp`, `readelf`
   `NEEDED` lines, the `NNTR_HTP_FORWARD_KINDS` strings count, gtest
   binaries) and the PR into `htp_moe` → `state:review`. The PR carries the
   counters permanently: they are the measurement tool for any later
   attention work.
6. **Handoff** `docs/measurements/146-attn-m1-speed.md` (`hexagon-handoff`
   template, per-token-entry sections), `state:needs-measurement`.
7. **DEVICE: E2E A/B (unavoidable).** Full `nntrainer_causallm`,
   `q40-qs4cx-wh`, prompt 512, G 64 / 512 / 1024, each ×2,
   `NNTR_NUM_THREADS=8`, A first. The CPU `q40` cell is included only if the
   sitting needs it for the text column. Variants:
   * **A** = reference skel (`htp_moe` at the PR's base), switch off. This is
     the control and the PPL reference: its first G=512 run writes
     `cont.ids`.
   * **B** = reference skel, `NNTR_HTP_FORWARD=1` with the entry mask of the
     sitting (six kinds, or D if #132's sitting made D the default; B and C
     use the same mask).
   * **C** = PR skel, same switches as B.
   * **D** is left free. Its default use is the PR skel with the switch off,
     which proves that the skel swap leaves the MoE/prefill path alone.
   Profiles, one run each at G=64 (not tok/s): B-prof and C-prof with
   `NNTR_HTP_PROFILE=2`, reading `ATTN_M1=` pcycles per op (G5). Ride-along
   on C's skel: `HvxAttnM1.*` (G1, G2 final numbers). If C's tokens ≠ B's,
   run one `NNTR_HTP_DUMP_ALL` decode (prompt 16, G 4) on B and on C and
   recompute the ATTN_M1 stretch inputs against `attn_m1_det.h` (rule 39
   (2)). Estimated ≈ 55 min.

## 5. Risks (host vs device) and how the tables expose them

* **Compiler lowering vs `hvx_emu`.** The host proves the intrinsic
  sequence, not the `qf32` code the compiler emits (§3.3 caveat). Exposed
  by G2 per skel in R1 before any E2E time is spent, and by C ≡ B tokens in
  step 7.
* **DMA / DDR rate.** ATTN_M1 has no DMA, but its reads are direct HVX
  loads, bounded by DDR latency (rule 26) and by the unit (31 vs 37 GB/s,
  rule 34). Exposed by the cold line and by `dsp_us` vs the 4.19 MB byte
  count per skel.
* **DVFS and thermal drift between sittings.** Every verdict is read inside
  one sitting. R1 brackets its candidates with `prof` at both ends. The
  `mhz` word (pcycles / qtimer µs) shows a clock change per call. The E2E
  table records battery / zone0 per G as in #130.
* **Stale skel.** A skel without the counters rejects the longer stats
  length, and the gtest names that. `0x8000040e` means the IDL is stale.
  Every row carries the skel md5 as seen on the device.
* **Warm gtest vs cold model.** The existing warm line flatters the kernel
  (it re-runs one layer), which is why the cold line exists and why G5
  reads the in-model pcycles.
* **Address-space budget.** Nothing new on the heap: the scratch keeps its
  size (O3 only re-indexes it), the phase slots are on the caller's stack,
  the cache stays 48 MiB at max_seq 2048 (26 % of ≈ 182 MiB). The fp16
  cache is deferred.
* **Subnormal rows (rule 37, #137).** G2 compares `bad_stats` with the
  reference skel's value at each L, not with 0, so #137's open question
  neither blocks nor hides this plan.

## 6. Docs to update (by the supervisor, from the filled handoff)

* **BENCHMARK.md:** a "#146 side tables" block. It holds R1's per-skel lines
  (warm / cold `ATTN_M1_FIELD`, `ATTN_M1_PHASE`, `bad` / `bad_stats`), the
  E2E rows A / B / C (/ D) with the serial, and B-prof / C-prof `ATTN_M1=`.
  The #130 side table's `ATTN_M1_FIELD` bullet gets a pointer to it.
* **LEDGER.md:**
  * ㉗: the measured split, the kept levers and the new µs/position, moving
    the item to §2 when G1–G5 pass.
  * A rule candidate, if R1 confirms it: *runtime-trip-count
    `HVX_Vector` arrays spill to the stack under hexagon-clang 19 -O3; a
    constant-shape specialisation is worth ×2–3 per kernel*.
  * ⑨'s budget row: attention ms/token at G 512 / 1024.
  * §4's #81 row: `HTP_ATTN_L2FETCH` moves from "not lifted" to "lifted as
    an `l2fetch` lead" if O4 lands.
  * The fp16 cache as an open item with §3.4's sizing.
