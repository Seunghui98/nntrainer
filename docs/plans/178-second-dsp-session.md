# 178 — A second cDSP session for the resident FC set + lm_head (decision D, option (a) of #132 PR 2)

Issue: dlwlzzero/nntrainer#178 (p1). Read against `htp_moe` @ `90d88e2b`
(PR #176 merged) and PR #175's branch `htp/132-exact-fc` @ `e5e579b3`,
whose IDL entries (`q4m1_register`, `fc_q4m1_f32`, `q8_quant_f32`) the
probe reuses. Decision D of `132-cpu-exact-fc-lmhead.md` §3.4: the user
chose option (a), a second session, on 2026-09-29 (contract §12: decode NPU
end to end, bit-preserving).

**What the 2026-09-29 sitting left us** (PR #175's last comment,
`R3CY10WM83Y`): 113 MiB allocatable in the loaded app vs 383 MiB needed
(FC 243 + lm_head 140); MoE arena 3696 MiB of the 3840 mapped (LEDGER
rule 8); exact FC `hvx_intrin` VTCM-fed **181 µs/call at K=7168 N=2048,
45.5 GB/s → 7.88 ms/token** (fc 5.14 + lm_head 2.74) vs CPU 7.4; direct
from DDR 699 µs → 30 ms/token; scalar quantizer +4.8 ms/token and the
scalar router 427 784 pcyc/op ≈ 4.4 ms/token, both ponytails needing HVX
versions before any end-to-end number is worth reading.

**This plan is probe-first.** One device gtest answers the issue's five
questions with exact cells and stop rules (§4). The end-to-end design (§3)
is written from those cells and from measured numbers, so the sitting's
handoff can be read straight into a build / no-build recommendation. No
app change; the default path is untouched.

## 0. What the SDK says (6.4.0.1, read on the workstation)

`incs/remote.h`, `docs/software/ipc/rpc.html` §"Multi-sessions",
`examples/calculator/src/calculator_test.c:315-433`:

* A second session is **reserved**, not merely opened twice:
  `remote_session_control(FASTRPC_RESERVE_NEW_SESSION,
  &remote_rpc_reserve_new_session{domain_name="cdsp", session_name})`
  returns `session_id` (≥ 1; 0 is the default session and cannot be
  reserved) and `effective_domain_id`. Then `FASTRPC_GET_EFFECTIVE_DOMAIN_ID`
  and `FASTRPC_GET_URI` (module URI in, `"...&_dom=cdsp&_session=1"`-style
  URI out, allocate `module_uri_len + 30`), then
  `DSPRPC_CONTROL_UNSIGNED_MODULE` **with the effective domain id** (not
  `CDSP_DOMAIN_ID`), then `nntr_hvx_open(uri2, &h2)`. `remote.h:740`: "only
  2 sessions are supported, session_id 0 and 1"; rpc.html: "Only 2 sessions
  on most devices, 4 from Lanai onwards" (Lanai = SM8650; the S25 Ultra is
  SM8750, so ≥ 2 either way). No sessions left → `AEE_ENOSESSION` (0x73).
* **Every session is its own PD** ("Effective Domain ID is the unique
  identifier representing the session (PD) on DSP"), and rpc.html §"MMUs and
  address spaces": "each process on the DSP runs in its own virtual address
  space … 32-bit … 4 GB". The multi-session section adds "the limitations
  on process virtual address space of 4GB continue to apply" — read as *per
  PD*, which is the whole premise of option (a). **Unverified on this
  device**: probe Q1 is the check.
* All sessions on one DSP must be the same PD type (signed / unsigned); the
  app is unsigned (`htp_backend.cpp:41`), so S2 is unsigned too. Unsigned
  PD services include VTCM, HVX, thread creation and mapping HLOS memory
  (rpc.html §"Unsigned PD available services"); default thread cap 128
  (192 on newer targets), raisable to 256 with `FASTRPC_MAX_THREAD_PARAM`.
* Effective domain ids are what `fastrpc_mmap(domain, fd, …)` and
  `dspqueue_create(domain, …)` take when an application has multiple
  sessions (`remote.h:734`, `dspqueue.h:246`). `dspqueue_create`'s
  `AEE_EBADPARM` covers "too many queues open for the DSP in this process";
  the count is not documented. A multi-domain queue (`dspqueue_request`,
  `DSPQUEUE_CREATE` over a `FASTRPC_CONTEXT_CREATE` context) broadcasts one
  packet to several sessions — not a DSP-to-DSP channel. **There is no
  DSP-PD-to-DSP-PD signalling API**: every hop either goes through the ARM
  or through a shared buffer both PDs poll.
* VTCM (`incs/HAP_compute_res.md` §"VTCM window feature", v79): the
  resource manager can give another *process* the part of VTCM a first
  process allocated-but-did-not-request only when the first process's
  request is one contiguous region smaller than its page. Availability is
  readable with `HAP_query_avail_VTCM(&avail_block_size, &max_page_size,
  &num_pages)` (`HAP_vtcm_mgr.h:46`). Whether `hexkl_micro_hw_init`
  (`hexkl_micro.h:230`, "initializes and powers up HVX, HMX, VTCM, DCVS")
  requests the whole 8 MiB is not documented; the M=1 MoE feed itself uses
  2 × `gu_bytes` ≈ 7.3 MiB of the 8 (`hexkl_mm_u8i4_moe.c:837-847`), so
  **session 1 leaves ≤ 0.7 MiB of VTCM whatever HexKL asks for**. The exact
  FC's VTCM feed needs `2 × gbytes × lanes` = 2 × 129 KB × 6 ≈ 1.5 MiB at
  K=7168 and ≈ 0.45 MiB at K=2048 (`nntr_hvx_fc_q4.c:247-259` on the PR
  branch). So Q4 is the deciding question, not Q1.
* HMX: `hexkl_micro_hmx_lock` is per process; session 2 must not take it
  (it has no HMX work) and today's `nntr_hvx_open` fails the open if the
  lock fails (`hvx_add_f32.c:95-100`). The probe's skel needs the lite open
  of §2.

## 1. Goal and gate

**Acceptance (issue):** a filled handoff with the five answers, and a
recommendation: session split (which ops in which session), hops per token,
and the projected per-token cost of the NPU end-to-end path with the FC set
resident. No change to the default path.

Made measurable — every cell is a `S2_FIELD key=value` line of the new
gtest `unittest_hvx_two_sessions` (§2), pasted into
`docs/measurements/178-second-dsp-session.md` and BENCHMARK.md's #178 side
table:

| Q | cell | pass / stop |
|---|---|---|
| 1 | `s2_reserve_rc`, `s2_session_id`, `s2_effdom`, `s2_open_rc`, `s2_open_us` | `rc=0` and the open ≤ 2 s; `0x73` = no second session on this device → **stop the probe** (Q5 only) |
| 1 | `s1_mmap_mib` (256 MiB steps, expect 3840), `s1_heap_mib` (1 MiB chunks, expect ≈ 113–182), `s2_mmap_mib` with S1 held, `s2_heap_mib` with S1 held, `s2_q4m1_mib` (the FC set's 54 weights in its 5 shapes, plan 132 §1 G0, plus the lm_head in 16k-row slices, registered into S2 heap in 8-slot rotation) | the design needs `s2_mmap_mib + s2_heap_mib ≥ 383 + 48 + 64` (weights, ATTN_M1 cache, activations/slabs); `s2_mmap_mib < 512` = the 4 GiB is per HLOS process → **stop** |
| 2 | `hop_arm_spin_us`, `hop_arm_block_us` (two dspqueues, ARM-brokered, 0 B and 12 KiB payload), `hop_mbox_us` (shared uncached ION word both PDs poll; 0 B and 8 KiB copy + cache clean), `hop_mbox_thread_cost_pct` (S2's FC rate with S1's mailbox thread spinning vs parked) | reported; per token = 44 × hop (§3.2). A hop > 100 µs on both transports means ≥ 4.4 ms/token of hops → flagged in the recommendation |
| 3 | `ddr_s1_alone_gbs` (S1 `dma_replay` bypass, fresh, the #158 probe cell ≈ 69 GB/s), `ddr_s2_alone_gbs` (S2 `fc_q4m1_f32` VTCM- or L2-fed, reps 20, ≈ 45), `ddr_concurrent_aggregate_gbs`, `ddr_sequential_gbs` | reported against rule 44's ≈ 70 ceiling; the design counts on the sequential number only |
| 4 | `s2_vtcm_avail_kib` with S1 open (from S2's lite open), `s2_fc_rate_vtcm_us` if avail ≥ 2 × gbytes × lanes for that lanes count (1..6 tried), `s2_fc_rate_l2_us` (new feed bit: DMA into a 2 MiB DDR heap scratch, `dst_bypass=0`), `s2_fc_rate_direct_us` | the FC set's ms/token from the best feed; **> 13 ms/token (rate < 31 GB/s) = the E2E path cannot beat A** → recommendation (c) of #132 |
| 5 | no device cell; cost estimate in §3.5 | — |
| standing | variant A E2E (unchanged reference), prompt 512, G 64 / 512 / 1024 × 2 | the sitting's control; the probe changes no product code, so the prefill gate (≥ −5 %) and text ≡ A are trivially met and are still recorded (rule 36: an A log must print `applied=0x703e1 … dma_bypass=1` and `dspq: on`) |

There is no bit-identity gate in this issue: the probe computes with
PR #175's kernels, whose `bad=0` cells already exist (G1); the FC probe
cells re-print `bad` against `q4_gemv_cpu_det` as hygiene.

## 2. Where it lives (all verified on `htp_moe` @ `90d88e2b` unless noted)

**Session open, one per process today.**
* `nntrainer/tensor/htp_backend/htp_backend.cpp:41-52`: unsigned-PD control
  on `CDSP_DOMAIN_ID`, then `nntr_hvx_open(nntr_hvx_URI + "&_dom=cdsp")`;
  `:83-96` the poll-QoS control on the handle. `htp_backend.h:47-104`: the
  singleton with `handle()`, `pollUs()`, `atClose()`.
* `test/htp/hvx_add_f32.c:40-104`: the DSP side of `nntr_hvx_open` —
  `qurt_hvx_get_units`, `hexkl_micro_hw_init(&vtcm_base, &vtcm_size, …)`,
  `config_off`, `hexkl_micro_hmx_lock`, `hexkl_micro_hmx_setup_acc_read_int32`,
  the pool sized from the HVX unit count. `test/htp/nntr_hvx_session.h:29-33`
  says in words what the code assumes: one open session at a time,
  `hexkl_micro_hw_init` is a singleton resource.

**Arena registration and mapping.**
* `htp_compute_ops.cpp:3877-3945` `tryChunk`: `rpcmem_alloc` (uncached) →
  `fastrpc_mmap(CDSP_DOMAIN_ID, fd, …, FASTRPC_MAP_FD)` →
  `nntr_hvx_arena_attach(session, fd, size, &dsp_id)`; `:3840`
  `kArenaChunkMax = 256 MiB`; `:3786-3818` `place`; `:3768-3777`
  `ensureArena`. `htp_rpcmem.h:63-70` resolves `rpcmem_*`, `fastrpc_mmap` /
  `fastrpc_munmap` with `dlsym`.
* `test/unittest/unittest_hvx_mm_u8i4.cpp:2981-3070`
  `ArenaCeilingThenHeapHeadroom`: the 256 MiB mapping ladder and the
  heap fill, which Q1's S1 half reuses verbatim; `:2627`
  `nntr_hvx_mem_probe_dsp_heap` (IDL `:401`) for the 1 MiB heap probe.

**dspqueue.**
* `htp_compute_ops.cpp:2620-2760` `DspqMoe` / `dspqCreate`:
  `dspqueue_create(CDSP_DOMAIN_ID, 0, 16 KiB, 4 KiB, …)`, `export`, two
  64 KiB ION buffers mapped with `fastrpc_mmap(CDSP_DOMAIN_ID, …)`,
  `nntr_hvx_dspq_start(session, id, spin_us)`; `:2760-2860` `dspqReady` and
  the packet write. Every `CDSP_DOMAIN_ID` here becomes the session's
  effective domain id in the E2E design.
* `test/htp/nntr_hvx_dspq.c` the DSP thread (`dspqueue_import`, spin then
  block, `HTP_DSPQ_OP_MOE` only); `test/htp/nntr_hvx_dspq_bench.c:48`
  `DSPQ_OP_ECHO` and IDL `:629-632` `dspq_bench_start / stop`, which the
  hop probe reuses on both sessions; `test/unittest/unittest_hvx_dspq_bench.cpp:110-160`
  `run_q` (ARM spin vs block rows, 12 KiB payload with FLUSH/INVALIDATE
  flags) — the per-hop timing loop is this with two queues.

**Per-token entry (the E2E design's host, unchanged in this issue).**
* `htp_compute_ops.cpp:1314-1420` `set_decode_graph_desc`: the resident
  mask, stretches `[s, e)` of resident ops, `first_resident_op_`; `:2512`
  `invokeForward` runs a stretch and returns `resume_at`; `:961-967` the
  `calls/token` line. `htp_graph_desc.h:56-68` the kinds; `:403` the FC op
  rule; `hmx/hexkl_graph.c:235` the kernel table.
* `nntr_hvx_attn_m1.c` / `hvx_attn_m1_f32.c`: the KV cache is 48 MiB of
  DSP heap at max_seq 2048 (LEDGER ⑨), which moves to S2 with the
  attention layers (§3.1).

**Probe pieces on PR #175's branch (`origin/htp/132-exact-fc`)**: IDL
`q4m1_register / q4m1_release / fc_q4m1_f32 / q8_quant_f32` (`nntr_hvx.idl:655-676`
there), `test/htp/nntr_hvx_fc_q4.c` (`FC_Q4_FEED_VTCM = 1 << 16`, per-lane
double-buffered DMA with `src_bypass=1`, `vtcm_per_lane = config_off / lanes`),
`unittest_hvx_softmax.cpp:842` `HvxFcQ4.Rate` (the `FC_RATE` / `FC_RATE_PROJ`
lines). The probe branch stacks on that branch.

**What the probe adds (test-only, no consumer of a changed contract):**
* `test/htp/nntr_hvx.idl`, appended after `argmax_f32` (additive; the
  skel is rebuilt and every device binary of the sitting is built from the
  same IDL — rule 3's `0x8000040e` otherwise):
  `session_info(rout sequence<uint32> res)` — `[hmx_locked, vtcm_size,
  vtcm_avail_kib, vtcm_max_page_kib, heap_free_probe_mib(0 = not run),
  open_path(0 full / 1 lite), hvx_units]`;
  `mailbox_run(in int32 fd, in uint32 bytes, in uint32 role, in uint32 n,
  in uint32 payload, in uint32 spin_us, rout sequence<uint32> res)`.
  `generate_stub.sh` regenerates the stub; `HtpComputeOps` does not call
  either (test entries), so `htp_compute_ops.cpp` is untouched.
  `nntr_quantize_stream`'s format tag, the loader check, the
  `NNTR_HTP_PROFILE` stage tables and `tools/htp_fc_report.py` are not
  touched: no weight format and no app path changes.
* `test/htp/hvx_add_f32.c` `nntr_hvx_open`: the **lite open** — if
  `hexkl_micro_hw_init` or `hexkl_micro_hmx_lock` fails, do not fail the
  open: record `open_path = 1`, `hmx_locked = 0`, acquire the largest
  available VTCM block through `HAP_compute_res_acquire` (size from
  `HAP_query_avail_VTCM`, no HMX, timeout 100 ms; 0 bytes is allowed),
  set `config_off = vtcm_size`, size the pool as before. Every HMX entry
  already reaches the session's VTCM through `s->vtcm_base/size/config_off`;
  the HMX ones gain one `if (!s->hmx_locked) return AEE_EUNSUPPORTED;` at
  the top (`mm_u8i4_*`, `moe_layer*`, `conv_block*`, `graph_*`). **Session
  1's behaviour is unchanged when `hw_init` succeeds**, which it does
  today (rule 21/22 provenance: the sitting's S1 skel is this build; A's
  log shows the usual `[HTP]` banners). `fc_q4m1_f32` gains feed bit 17
  (`FC_Q4_FEED_L2`: the same per-lane double buffer into a 2 MiB DDR heap
  scratch with `dst_bypass = 0`), which is what S2 runs if Q4 reads
  `vtcm_avail_kib < 2 × gbytes × lanes`.
* `test/htp/nntr_hvx_mailbox.c` (new, `@file` / `@brief`): `mailbox_run` —
  `HAP_mmap_get(fd)`, two 128 B-aligned words `ping` / `pong` at the start
  of the buffer, payload region after; role 0 writes `seq` to `ping`
  (after copying `payload` bytes and `qurt_mem_cache_clean(FLUSH)`), then
  spins with `pause` on `pong == seq` (invalidate before each read, with a
  `spin_us` cap that then sleeps 50 µs); role 1 mirrors. Returns
  `[us_total, n_done, n_timeouts, payload_checksum]`. ≈ 120 lines.
* `test/unittest/unittest_hvx_two_sessions.cpp` (new) + a
  `unittest_hvx_two_sessions` target in `test/jni/Android.mk` (copy of the
  `unittest_hvx_fc` block at `:1037-1049`). Fixture: S1 opened exactly as
  the app does (`unittest_hvx_softmax.cpp:99-105`), then the five `TEST_F`s
  of §4 in order; `S2_FIELD` lines; every `remote_session_control` result
  printed in hex. `session_info` on both handles at the start and the end.
* `test/htp/host/`: `mailbox_host_check.c` — the two roles on two
  pthreads over a `malloc`ed buffer with the cache calls stubbed
  (`hvx_scalar_stubs.h`), 10 000 exchanges, `n_timeouts = 0`, checksum
  equal (`run_host_checks.sh` gains the line `MAILBOX OK`). The lite-open
  branch is compiled by `test/htp/build.sh` (rung 2), and exercised on the
  host by `graph_host_check` only as far as the `hmx_locked` guard is
  reachable there (it is not — host stand-ins do not open sessions;
  stated, not claimed).

## 3. Design

### 3.1 The split: S1 = router + MoE (the arena), S2 = everything else

| session | ops (`htp_graph_desc.h` kinds) | memory | hardware |
|---|---|---|---|
| **S1** (today's session, unchanged skel path) | `ROUTER_TOPK`, `MOE` | 3840 MiB arena + the MoE staging; the router's 22 × 256 KB in heap | HMX lock (prefill), the M=1 feed's 7.3 MiB of VTCM, the dspq thread |
| **S2** (new) | `RMSNORM`, `FC` (conv in/out, q/k/v/o), `CONV1D_GATE`, `QK_NORM`, `ROPE`, `ATTN_M1`, `ADD`, `DENSE_FFN` (layers 0–1), `LM_HEAD` + argmax | FC set 243 + lm_head 140 in `Q4M1` (heap or a mapped arena — Q1 says which is available), ATTN_M1 cache 48, RoPE table 0.5, activations | no HMX, ≤ 0.7 MiB VTCM (Q4) or the L2 feed, its own DMA queues, its own dspq thread |

Why this cut: the address-space wall is the FC set + lm_head, nothing else;
the arena is the only reason to keep a session at all; every other kind
has no address-space need and costs a hop if it sits on the far side of
one. With the ROUTER in S1 the S2 → S1 hop carries only the normed
activation (8 KiB) and S1 → S2 carries the MoE output (8 KiB); the
top-4 ids and weights never leave S1.

**Rejected alternative:** S2 = FCs only, small ops stay in S1 (the
smallest change to today's resident set). It costs 4 hops per conv layer
(norm → in_proj → conv → out_proj → add) and 6 per attention layer, ≈ 100
hops/token against 44, for no memory reason. Also rejected: attention in
S1 (its 48 MiB cache fits S1's 113): +2 hops on each of the 6 attention
layers and nothing gained, since S2 has room.

### 3.2 Hops per token and the transport

Per MoE layer (22 of the 24): S2 runs `[RMSNORM … RMSNORM]` up to the
ffn norm, hop to S1 (`ROUTER_TOPK`, `MOE`), hop back to S2 (`ADD`, next
layer). Layers 0–1 (dense FFN) stay inside S2. Token: ARM → S2 (embedding
row in), 44 inner hops, S2 → ARM (token id out). **46 transport events per
token, 44 of them DSP ↔ DSP; today's path has 22.**

Two ways to make a hop, both probed (Q2):

* **ARM-brokered (the design's baseline)**: two dspqueues, one per session
  (`dspqueue_create(effdom_i)`), the same `DspqMoe` object twice, packets =
  today's `forward` arguments (start op, pos, buffer refs). The ARM loop is
  today's `invokeForward` alternating sessions on `resume_at`; the
  activation rides in one ION buffer mapped in both sessions
  (`fastrpc_mmap(effdom1, fd)` and `fastrpc_mmap(effdom2, fd)`; buffer
  refs with FLUSH_SENDER / INVALIDATE_RECIPIENT keep the cache maintenance
  in the framework). Expected cost: one `QSS` round trip per hop, ≈ 17 µs
  (rule 40: F12 87.1, QSS12 16.7, QBS12 28.9) → **44 × 17 ≈ 0.75 ms/token**
  plus the two ARM ends; both DSP threads spin their windows.
* **Mailbox (the upgrade if the probe likes it)**: a shared uncached ION
  page; S1's dspq thread, after finishing a MoE, writes the sequence word
  and spins on S2's; S2's forward thread mirrors. The ARM is out of the 44
  inner hops and issues one packet per token. Expected ≈ 1–5 µs/hop, but
  each session then keeps a HW thread spinning while the other computes
  (6 HW threads on v79, the FC uses 6 lanes): `hop_mbox_thread_cost_pct`
  measures what the spinner takes from the FC. Not built in this issue;
  the probe decides whether the E2E plan lists it.

### 3.3 Memory in S2 and the feed

The FC set goes into S2 as `Q4M1` (PR #175's layout, bit-preserving
reorder at registration): heap if `s2_heap_mib ≥ 450`, else a mapped ION
arena of 256 MiB chunks with the DMA reading from it (the same
`fastrpc_mmap` + `HAP_mmap_get` path as the MoE arena). The weight DMA
feed is per lane, double-buffered, `src_bypass = 1` (the weights are
written once by the CPU before the map and never by the DSP — the same
argument as rule 43); its destination is VTCM if Q4 finds ≥ 2 × gbytes ×
lanes, else the L2 scratch feed (feed bit 17). `_det` specs sit before
every quantizer as in #132 (the activation quantizer `q8_0_quant_cpu_det`
runs in S2 in front of every FC; its HVX version is a prerequisite, see
§3.4).

### 3.4 Projected per-token cost (measured inputs; the sitting fills the blanks)

| term | ms/token | source |
|---|---|---|
| MoE, 22 × 0.418–0.43 | **9.2–9.5** | rule 43 (`dsp` 417.5 µs/call on the bypass default) |
| FC + lm_head, VTCM-fed `hvx_intrin` | **7.9** | G3 (181 µs/call, 45.5 GB/s); **L2-fed: Q4; direct: 30** |
| activation quantizer, HVX version | 0.3 (assumed; today 4.8 scalar) | #132 ponytail; ≈ 53 quantizations of 2048–7168 floats |
| router, multi-chain sffma | 1.0 (assumed; today 4.4) | #132 sitting: 427 784 pcyc/op scalar chain |
| ATTN_M1, 6 layers | **2.0** | #170 ≈ 0.33 ms/layer |
| RMSNORM ×49, QK_NORM, ROPE, CONV1D_GATE, ADD, argmax | 0.9 | contract §12 (norm ≈ 0.4) + the rest assumed 0.5 |
| hops, 44 ARM-brokered + 2 ends | **0.8** (mailbox: ≈ 0.2 + spinner cost) | rule 40; Q2 |
| **total, sequential** | **≈ 22.1–22.4 → 45 tok/s** | |
| A today (hybrid, G=512) | 19.74 → 50.6 tok/s | #158 B |
| CPU today, same work as S2 | FC + lm_head 7.4, attention 1.5, rest 0.6 | rule 46 |

Read plainly: **with every measured number at face value the two-session
E2E path projects ≈ 10 % slower than the hybrid A at G=512**, because the
DSP does the FC set at 7.9 ms where the CPU does it in 7.4 and adds 2 ms
of attention where the CPU's 1.5 grows only with G. What can move it:
(1) Q3 — if S2's FC DMA and S1's MoE ring could overlap, the byte floor is
886 MB / 70 GB/s = 12.7 ms; but the layer order serialises FC → MoE → FC,
so the only overlap is prefetching the next FC's first slabs during the
MoE, bounded by S2's VTCM (≤ 0.7 MiB) — ≤ 0.5 ms; (2) the FC kernel is
compute-bound at 45.5 GB/s (direct feed 699 µs shows DMA is not the limit
at 181), so a hand-scheduled `hvx_intrin` loop toward the 57 GB/s bypass
rate would give ≈ −1.6 ms; (3) at G=1024 the CPU's attention is 1.97 and
the DSP's 2.0 — a wash. So the honest expectation is parity at best, and
the recommendation in the handoff must say whether the user wants the
"NPU end to end" product at parity (§12 says the hybrid stays the product
path "until the end-to-end path is faster and bit-identical").

### 3.5 Fallback if a second session is refused: partial expert residency (cost only)

Shrink the MoE arena to make ≈ 450 MiB of room in S1 (FC set + lm_head +
slack): 3696 → ≈ 3250 MiB = 12 % of the 704 layer-expert slots (5.25 MiB
each) evicted, ≈ 85 experts non-resident. Doc 52's cached-slim (LRU over a
DDR pool) keeps the bytes identical, so text and dumps stay bit-identical;
only the time changes. Per token, top-4 of 32 in 22 layers = 88 expert
uses; with uniform routing 12 % miss → ≈ 10.6 misses × 5.25 MiB = 56 MB
staged through an ION slot per token: a CPU `memcpy` into an uncached
arena slot at ≈ 8–10 GB/s = **5.6–7 ms/token**, plus the DSP-side attach
(0.09 ms per registration) and the cold DMA of the slot. With skewed
routing (an LRU keeping the popular experts) the miss rate might halve:
**≈ 3–7 ms/token on top of §3.4's 22 ms → 25–29 ms → 34–40 tok/s.** It
also needs the CPU to know the routing one layer ahead (it does: the
router runs before the MoE, but on the DSP in the E2E design — so the ids
come back to the ARM, one more hop). Not planned; the estimate goes into
the handoff's recommendation as the cost of option (a)'s absence.

### 3.6 Contract and doc 45 §3 check

* Three walls: the split changes the transport wall only (22 → 46
  events); wall 1 (M=1 GEMV) and wall 2 (DMA rate) are untouched in S1.
* Arena budget: S1's 3840 MiB is not touched by the probe; S2's mapping
  is released at the end of every test (`fastrpc_munmap` with the
  effective domain id, then `nntr_hvx_close(h2)`, then
  `FASTRPC_SESSION_CLOSE` is *not* called — it closes every handle on the
  domain, S1's included).
* No CPU fallback for `QS4CX_WH`: unchanged; the probe never loads a model.
* Activation handles / DMA hidden behind compute / `_det` before every
  quantizer: §3.3; bit-identical + text gates: §1's standing row and the
  E2E plan that follows this one.

## 4. Steps

Branch `htp/178-two-session-probe`, stacked on `origin/htp/132-exact-fc`
(PR #175) so the FC entries exist; rebase onto `htp_moe` when #175
merges. Every step ends in a rung of `.claude/skills/hexagon-gates`.

1. **Mailbox spec + host check.** `nntr_hvx_mailbox.c` with the roles as
   plain C over `volatile uint32_t` words and the cache calls behind
   `hvx_scalar_stubs.h`; `mailbox_host_check.c` runs both roles on two
   pthreads, 10 000 exchanges, payload 8 KiB, checksum equal, `MAILBOX OK`.
   Gate: rung 1 (`ninja -C build`, `run_host_checks.sh` → `ALL CHECKS PASS`
   with the new line; `tools/htp_syntax_check.sh`).
2. **Lite open + `session_info` + feed bit 17 in the skel.** `hvx_add_f32.c`
   per §2; `nntr_hvx_fc_q4.c` gains the L2 scratch feed (a 2 MiB
   `memalign(128)` heap buffer per session, lanes split it as VTCM is
   split; `dst_bypass = 0`); IDL +2 entries; the `hmx_locked` guard on the
   HMX entries. Host: `q4_gemv_host_check` still `BIT-IDENTICAL` (the L2
   feed is a destination change only — the host stand-in copies). Gate:
   rung 1 again, then rung 2 (`test/htp/build.sh` → `UNDEFINED SYMBOLS OK`,
   md5 recorded; `-Wall -Werror` clean with the weak `HAP_compute_res_*`
   symbols declared like `nntr_hvx_dspq.c`'s `#pragma weak`).
3. **The gtest** `unittest_hvx_two_sessions.cpp`, five tests in this order,
   each printing its `S2_FIELD`s and skipping (not failing) when an
   earlier stop rule fired:
   * `Q1_SecondSession`: S1 fixture; S1 ladder to the mapping ceiling
     (`ArenaCeilingThenHeapHeadroom`'s loop, `arena_attach` per chunk so
     the DSP side also holds them) and a 1 MiB heap probe; reserve +
     effective domain + URI + unsigned control + `nntr_hvx_open(uri2)`
     with the elapsed time; `session_info` on both; S2 ladder (256 MiB,
     `fastrpc_mmap(effdom2, …)`, up to 4 GiB); S2 heap probe; the FC set's
     54 weights + lm_head slices registered with `q4m1_register` in 8-slot rotation
     (bytes accepted → `s2_q4m1_mib`); unmap and release S2's, keep S1's
     for the next tests (they are the loaded app's state).
   * `Q2_HopCost`: two dspqueues (`create(effdom_i)`), `dspq_bench_start`
     on both, `run_q`-style loop alternating the queues, 100 warm + 1000
     timed, rows {ARM spin, ARM block} × {0 B, 12 KiB}; then `mailbox_run`
     on two ARM threads (roles 0 / 1, n = 10 000, payload 0 / 8 KiB, spin
     cap 1 ms), per-hop µs = `us_total / (2 n)`; then S2's
     `fc_q4m1_f32` rate (K=7168) with S1's mailbox thread spinning on a
     word that never comes (spin cap = the FC's duration) vs parked →
     `hop_mbox_thread_cost_pct`.
   * `Q3_DdrShare`: S1 `dma_replay` (`MoeChunkReplay`'s bypass + fresh
     schedule over 8 chunks of the held arena, calls = 40) and S2
     `fc_q4m1_f32` (best feed of Q4's pre-read, reps 40) launched from two
     ARM threads within 50 µs of each other (a barrier), each returning its
     own µs; aggregate = Σ bytes / the overlapped window from ARM
     timestamps; then the same two sequentially. Also the CPU-free
     control: both again with the ARM threads pinned to little cores.
   * `Q4_VtcmShare`: `session_info` on S2 → `vtcm_avail_kib`,
     `vtcm_max_page_kib`; `fc_q4m1_f32` on S2 at K=7168 N=2048 and K=2048
     N=6144 with feed ∈ {vtcm if it fits at lanes 1..6, L2, direct}, 3
     timed calls after 1 warm, `bad` against the spec, `FC_RATE_PROJ`-style
     ms/token for the 402 MB set from the best cell.
   * `Q5_Teardown`: close S2 before S1; `session_info` on S1 after, to
     show S1's VTCM and HMX survived S2's lifetime (`hmx_locked = 1`,
     `vtcm_size` unchanged) — the E2E path's precondition.
   Gate: rung 3 (`build_android.sh --htp` for A's app set — unchanged
   product code, so the md5s equal `htp_moe`'s current set when built from
   the same tree; `ndk-build … unittest_hvx_two_sessions unittest_hvx_softmax`,
   md5s recorded; `readelf -d` NEEDED lines).
4. **Handoff** `docs/measurements/178-second-dsp-session.md` (skill
   `hexagon-handoff`), ≈ 35 min. **This is the step where a device is
   unavoidable**; the orchestrator runs it on `R3CY10WM83Y` (the filled
   doc records the serial, the handoff names none). Variants (2 of 4):

   | variant | what | runs |
   |---|---|---|
   | **A** | unchanged reference: `htp_moe` @ `90d88e2b` app set + the sitting's skel (S1 path identical), nothing set | full E2E, prompt 512 (`docs/measurements/77-prompt512.txt`), G = 64 / 512 / 1024 × 2, `A A` per G, zone0 before / after; expect `applied=0x703e1 … dma_bypass=1`, `dspq: on`, `calls/token` absent (switch off) |
   | **P** | `unittest_hvx_two_sessions` on the same skel, `--gtest_filter='TwoSessions.*'`, run twice (cold, then after A's G=1024 pair while warm) | the five tests' `S2_FIELD` lines; `unittest_hvx_softmax --gtest_filter='HvxFcQ4.MatchesSpecBitExact'` once as the stale-skel canary |

   Stop rules on the day: `s2_reserve_rc = 0x73` → run Q5's teardown only
   and note it; `0x8000040e` anywhere → stale skel, stop; `s2_open_us > 2 s`
   or the open hangs → S2's `hw_init` is blocking on VTCM; note the FARF,
   kill with `FASTRPC_REMOTE_PROCESS_KILL` on effdom2 only. Never
   `FASTRPC_SESSION_CLOSE`.
5. **Fold.** The filled table + the §3.4 projection with the blanks filled
   → the recommendation paragraph in the handoff (build the E2E plan as a
   new issue / stop at (c) / fallback §3.5), issue → `state:measured`.

## 5. Risks (host vs device)

* **The second session may be refused or blocked at open, not at reserve.**
  `hexkl_micro_hw_init` inside S2's open may wait on VTCM or the HMX lock
  with an unknown timeout instead of failing fast; the lite open catches
  a failure, not a hang. Mitigation: the elapsed time is a cell, the
  stop rule kills effdom2 only, and A's E2E cells are run before P.
* **VTCM is the real wall (Q4), not address space (Q1).** S1's feed holds
  ≈ 7.3 of 8 MiB; the L2 feed is a guess with no rate yet. If it reads
  < 31 GB/s the whole design fails on speed while every Q1–Q3 cell
  passes; the handoff's recommendation is written to lead with Q4.
* **DMA rate and DVFS.** Rule 44: a second DSP reader redistributes the
  70 GB/s; Q3's concurrent cell will read below the sum. The design counts
  only the sequential cell; the concurrent one is information. DVFS shows
  in the zone0 log and in running P cold and warm; the A pairs bracket it.
* **Thermal drift between sittings.** No cross-sitting comparison is made:
  A is this sitting's control, and every P cell is an absolute rate on
  the same unit and skel within minutes of A.
* **Stale skel / two binaries.** The IDL grows by two entries, so A's app
  and P's gtest must be built from one tree; the `HvxFcQ4` canary and
  `0x8000040e` in any line void the run (rule 3).
* **Address-space budget on the HLOS side.** The driver's per-process
  mapping table (not documented) may cap S1 + S2 mappings; Q1's S2 ladder
  with S1 held at 3840 is exactly that reading. If S2 maps < 512 MiB the
  heap probe still says whether heap-resident weights (383 MiB) fit.
* **Cache coherence of the mailbox.** Two PDs on one core share the L2
  physically, but the DSP's L1D and the ARM's view need explicit
  maintenance; the probe's payload checksum on every exchange makes a
  stale read visible (`n_timeouts`, `payload_checksum`).
* **HW-thread contention.** Two spinning dspq threads plus two pools on
  6 HW threads: `hop_mbox_thread_cost_pct` and the ARM-spin rows show it;
  the E2E plan chooses spin windows from them.
* **Projection inputs that are not measured yet**: the HVX quantizer and
  the multi-chain router (assumed 0.3 and 1.0 ms/token). Both are #132
  follow-ups and are listed as prerequisites of the E2E plan, not of this
  probe.

## 6. Docs to update

* **`docs/htp_moe/BENCHMARK.md`**: a #178 side table with the Q1–Q4 cells
  (S2 reserve / open rc and µs, S1 / S2 mapping and heap MiB, hop µs per
  transport and payload, DDR alone / concurrent / sequential GB/s, S2 VTCM
  KiB and the FC ms/token per feed); the A rows of the sitting (G 64 / 512 /
  1024, prefill, text ≡ A, the banner check); the artifact rows (skel md5,
  gtest md5, app set md5).
* **`docs/htp_moe/LEDGER.md`**: new rules from the cells — *this device
  gives an unsigned app N cDSP sessions; a second session's PD maps X MiB
  beside the first's 3840 (the 4 GiB is per PD / per process)*; *a
  DSP-to-DSP hop costs Y µs ARM-brokered and Z µs through a shared page*;
  *session 2 gets K KiB of VTCM beside the M=1 feed, and the exact FC
  L2-fed reads R GB/s*; *two DSP readers share the 70 GB/s (rule 44
  extended to DSP + DSP)*. Open items: ⑨ / ㉓ get decision D's outcome;
  ㉘ (CPU-exact FC) is updated with §3.4's projection; a new item for the
  E2E two-session plan or for the §3.5 fallback, whichever the
  recommendation picks. §2 verdict row for #178 once the handoff is filled.
