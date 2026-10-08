# 132 — PR 2 Part B: the two-session NPU end-to-end decode (decision D = (1) + (2))

> Superseded by #211 (one PD only, `211-one-pd-only.md`); kept as the record of the design and its measurements.

Issue: dlwlzzero/nntrainer#132 (p1). Decision D (user, 2026-09-30): **(1)**
Part B is the two-session end-to-end design of `178-second-dsp-session.md`
§3 (S1 = router + MoE, S2 = everything else, one call per session per
token, shared-page hops), bit-preserving; **(2)** in parallel, measure what
S1's MoE feed costs if it gives up ≈ 1.5–2 MiB of its VTCM so S2's exact FC
can be VTCM-fed. The hybrid (#158 B) stays the default until the E2E path
is faster *and* bit-identical (contract §12).

Read against `origin/htp/132-exact-fc` @ `3dbe1dd7` (PR #175, unmerged;
the probe branch `origin/htp/178-probe` @ `bf78a8f9` stacks on it) and
`htp_moe` @ `4eab54ef`. Part A (`132-cpu-exact-fc-lmhead.md`) supplies the
specs, the `hvx_intrin` FC kernel, the HVX quantizer, the 4-lane router
and the shadow; this file does not repeat them.

**Facts on silicon this plan builds on** (`docs/measurements/178-second-dsp-session.md`
set2 / set3, `132-pr2-exact-fc.md` sitting 1, #170 round 2, #164):

| fact | value |
|---|---|
| second cDSP session | reserve + lite open rc 0, **22 ms**; per-PD address space (S2 maps 3584 MiB while S1 holds 3840); FC + lm_head **74 weights / 383 MiB** fit in S2 |
| hop | shared-page mailbox **0.33 µs (0 B) / 2.4 µs (8 KiB)**; dspqueue spin 2.8 / 12.7 µs |
| DDR | S1 alone 70 GB/s, S2 alone 20 (L2-fed); concurrent **58.2 + 15.4** (S1 −17 %; the sum is rule 44's ceiling) |
| VTCM | S2 gets **0**: S1's `hexkl_micro_hw_init` holds the 8 MiB; the M=1 feed uses 2 × `gu_bytes` ≈ 7.3 MiB of it |
| exact FC (`hvx_intrin` only) | **7.88 ms/token** VTCM-fed (6 lanes, 45.5 GB/s) / **8.06** L2-fed (best 3 lanes: 26 GB/s at K=7168, 52 at K=2048) / 30 direct |
| HVX quantizer / 4-lane router | ≈ 0.4 ms/token / ≈ 50 k pcyc per op (ISS; sitting S3 measures) |
| ATTN_M1 round 2 | ≈ 0.18 ms/layer at G=1024 (373.6 k pcyc) |
| norms | bit-identical (#164); MoE 22 × ≈ 0.43 ms (rule 43); hybrid A 19.7 ms / 50.6 tok/s (#158), 18.4–19.5 ms on the #178 sittings |

## 1. Goal and gate

**Acceptance (issue, re-scoped 2026-09-29):** decode runs end to end on the
NPU at one call per token, every resident op bit-identical to the Android
CPU path, verified per op with a CPU-vs-HTP shadow on the same input.

Made measurable:

| # | check | where | pass |
|---|---|---|---|
| B0 | **VTCM-share probe** (track 2, §3.1) | new cells in `unittest_hvx_two_sessions` + `HmxMmU8I4Layer.MoeM1GemvFeedVsCompute`, one sitting | reported: S1's M=1 MoE `dsp` µs/call at VTCM cap 8 / 6.5 / 5.8 MiB with today's feed and with the half-slab ring; S2's `vtcm_avail_kib` and VTCM-fed FC ms/token at each cap; concurrent S1 + S2 GB/s with S2 VTCM-fed. **Stop rule** (the break-even, §3.1): the E2E design is VTCM-fed only if `Δdsp × 22 < gain`, else it stays L2-fed |
| B1 | **calls/token = 1.00** | `[HTP] graph: … calls/token=` (`htp_compute_ops.cpp:967`), redefined as ARM → S2 packets per token; host fixtures (`run_inproc_e2e.sh`: tiny, hd64, lfm25) and LFM2.5 on the device | `calls/token=1.00` on all four |
| B2 | **bit-identity per op** | MoE dumps (`NNTR_HTP_DUMP`) `bit_identical=1` against A; the shadows: `dev/fc-shadow` (FC / lm_head / ADD / router), `dev/norm-shadow-164`, `dev/attn-shadow-170-r2`, each on the E2E build's kernels; new DENSE_FFN (SwiGLU) and argmax records in the FC shadow | every record equal, `n > 0` per tag; dumps `=1` |
| B3 | **text and nll** | 8 prompts at G=256: `NNTR_PPL_DECODE` nll lines equal to A to 17 digits; text ≡ A **8/8** | 8/8 both |
| B4 | **speed** | prompt 512, G 64 / 512 / 1024, mirrored against the sitting's A | E ≥ A at every G, and ≥ 50 tok/s; §3.5 says the projection does **not** reach this without the sheds it names |
| standing | prefill ≥ −5 % of A (the E2E build changes nothing at M > 1, but S2's open and the 383 MiB registration add startup, which the prefill cell does not cover: a separate `load_ms` cell); text ≡ A |

## 2. Where it lives (verified on the base worktree unless marked *probe*)

**Session and VTCM (track 2).**
* `test/htp/hvx_add_f32.c:53-104` `nntr_hvx_open`: `hexkl_micro_hw_init(&vtcm_base, &vtcm_size, …)` (`hexkl_micro.h:230`), `config_off` from `hexkl_micro_hmx_config_size()`, `hexkl_micro_hmx_lock`. *probe* (`origin/htp/178-probe`): the lite open `lite_vtcm()` (`HAP_query_avail_VTCM` → `HAP_compute_res_acquire`, 0 bytes allowed) and `session_info`.
* `libhexkl_micro.a` (`hexkl_micro_hw_init.c.obj`) references `compute_resource_query_VTCM`, `compute_resource_attr_set_vtcm_param_v2`, `compute_resource_acquire` as **weak undefined** symbols (`hexagon-llvm-nm`), and `test/htp/build.sh:35,84,114` links it statically into the skel — so `-Wl,--wrap=compute_resource_query_VTCM` on the skel link binds hexkl's query to a shim in the skel (§3.1). HexKL has no VTCM release entry (only `hw_init`, `hmx_lock/unlock`).
* SDK `incs/HAP_compute_res.md` §"VTCM window feature" (v79): a process that requests 6 MiB in an 8 MiB page is window-restricted to its 6 MiB and **another process can acquire the remaining 2 MiB**; the window must be one contiguous region.
* `nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.c:567-592` the M=1 feed's schedule (two `gu_bytes` slabs, downs in a freed slab), `:1119-1127` the fit rule `2 × gu_bytes ≤ arena && 2 × dn_bytes ≤ gu_bytes` (a shape that fails runs the arena read), `:836-846` slab offsets, `:1440-1470` the per-expert pool runs and slab reuse. `test/htp/host/moe_layer_host_check.c` is the schedule's scoreboard.
* `test/htp/nntr_hvx_fc_q4.c:70` `FC_Q4_FEED_VTCM`, `:242-253` `gbytes = K/64 × Q4M1_PAIR_BYTES` (129 KB at K=7168) and `vtcm_per_lane ≥ 2 × gbytes` (1.5 MiB at 6 lanes, 0.77 at 3); *probe*: feed bit 17 (L2 scratch, `s->fc_l2`).
* `test/unittest/unittest_hvx_mm_u8i4.cpp:2504` `MoeM1GemvFeedVsCompute` (the `vtcm` cell's `dsp` / `mm` µs); *probe* `unittest_hvx_two_sessions.cpp:612` Q2, `:725` Q3, `:831` Q4.

**Graph and per-token entry (track 1).**
* `nntrainer/tensor/htp_backend/htp_graph_desc.h:56-68` kinds (wire order, append only), `:265` the weight kinds, `:291-305` validator rules (ADD needs every RMSNORM resident; ROUTER_TOPK needs its MOE), `:403` the FC op rule.
* `nntrainer/tensor/htp_backend/hmx/hexkl_graph.c:235-247` kernel table (`FC`, `DENSE_FFN`, `LM_HEAD` are `NULL`), `:448-521` `hexkl_graph_forward` (runs the maximal resident run from `start_op`, `memcpy` in / out of `HTP_GRAPH_N_SLOTS` slots, returns `resume_at`).
* `nntrainer/tensor/htp_backend/htp_compute_ops.cpp:1314-1420` `set_decode_graph_desc` (`:1342-1351` the ARM rule "ROUTER_TOPK needs ADD"), `:1480-1560` `decode_op_fp32` per kind (param binding, conv-state re-seed, KV seed), `:1568-1640` `runStretchOp` (first / mid / last hook), `:2511-2618` `invokeForward`, `:2620-2760` `DspqMoe` / `dspqCreate` (`:2695 dspqueue_create(CDSP_DOMAIN_ID, …)`), `:3839` `kArenaChunkMax`, `:3877-3945` `tryChunk` (`:3904 fastrpc_mmap(CDSP_DOMAIN_ID, …)`, `nntr_hvx_arena_attach`).
* `nntrainer/tensor/htp_backend/htp_backend.cpp:41-52` the one session (`DSPRPC_CONTROL_UNSIGNED_MODULE` on `CDSP_DOMAIN_ID`, `nntr_hvx_open`). *probe* `unittest_hvx_two_sessions.cpp:355-420` the reserve / `FASTRPC_GET_EFFECTIVE_DOMAIN_ID` / `FASTRPC_GET_URI` / unsigned-control / open sequence to lift.
* `test/htp/nntr_hvx_dspq.c:126-200` the dspq thread (`HTP_DSPQ_OP_MOE` only, spin then block); `nntrainer/tensor/htp_backend/htp_dspq_wire.h` (`HTP_DSPQ_OP_MOE` 1, `_QUIT` 2, 64 KiB buffers). *probe* `test/htp/nntr_hvx_mailbox.c` (ping / pong words on their own 128 B lines, payload after, `qurt_mem_cache_clean` FLUSH before a post and FLUSH_INVALIDATE before a read, spin then 50 µs polls).
* `Applications/CausalLM/layers/htp_decode_hook.h:33-58` the layers' hooks (`rms_norm.cpp`, `conv_block_layer.cpp`, `qkv_layer.cpp`, `mha_core.cpp`, `residual_add.cpp`, `lfm2_moe/lfm2_moe_layer.cpp`); `models/causal_lm.cpp:328-331` the greedy pick (`std::max_element`, first max wins); `llm_util.cpp:87`.
* `test/htp/nntr_hvx.idl:54` `arena_attach`, `:489` `graph_init`, `:606` `graph_set_param`, `:643` `dspq_start`, `:656-682` `q4m1_register` / `q4m1_release` / `fc_q4m1_f32` / `q8_quant_f32` / `swiglu_cpu_f32` / `argmax_f32`; *probe* `session_info`, `mailbox_run`.

**Consumers that move with a changed contract.**
* IDL `test/htp/nntr_hvx.idl` + `generate_stub.sh` (rung 2 rebuilds the skel; rule 3 `0x8000040e` otherwise): new entries §3.3; `q4m1_register`'s heap copy is joined by an arena form.
* `HtpComputeOps` (`htp_compute_ops.cpp`): a second handle, every `CDSP_DOMAIN_ID` at `:2646/:2648/:2695/:2718/:2722/:3904/:3928` becomes the session's effective domain id, the graph description is sent to both sessions with complementary masks, `invokeForward` is replaced by the token driver for the all-resident mask.
* Quantizer format tag (`nntr_quantize_stream`) and the loader check: **unchanged** — the `Q4M1` reorder happens at registration from the `q40-qs4cx-wh` file's `q4_0x4` bytes (plan 132 §2), the same bits.
* `NNTR_HTP_PROFILE`: the graph line `pcyc/op` names `FC`, `DENSE_FFN`, `LM_HEAD`; one line per session (`graph[S1]` / `graph[S2]`) plus a `hops` line (count, µs); `tools/htp_fc_report.py` gains the `q4m1` FC row and the per-session lines. Rule 36's banner set grows by `[HTP] s2: open … mib=… vtcm_kib=…` and `[HTP] token driver: on`.
* The shadows (`dev/*`, never merged) follow the E2E build as measurement commits.

## 3. Design

### 3.1 Track 2 — the VTCM-share probe (gtest only, no app change)

**Mechanism.** HexKL asks the resource manager for the VTCM it is told is
available. The skel wraps that query: `-Wl,--wrap=compute_resource_query_VTCM`
on the skel link (`build.sh`, a variant build `HEX_EXTRA_CFLAGS=-DNNTR_VTCM_CAP_KB=<n>`
+ the wrap flag) and a `__wrap_compute_resource_query_VTCM` in a new
`test/htp/nntr_hvx_vtcm_cap.c` that calls `__real_…` and caps
`avail_block_size` (and the layout's page) at `NNTR_VTCM_CAP_KB`. S1 then
opens with `vtcm_size = cap`, its `config_off` follows, and by the VTCM
window rule S2's lite open finds `8 MiB − cap` (probe Q4's `vtcm_avail_kib`
reads it). No HexKL change, no IDL change, no product code: the default
build has no wrap and is byte-for-byte today's skel path.

**Caps.** 8 MiB (control: the wrap present, cap = total — proves the shim is
inert), **6.5 MiB** (S2 gets 1.5 = the FC's 6-lane need at K=7168),
**5.8 MiB** (S2 gets 2.2 = the FC's need + a 0.7 MiB prefetch slab). The
HMX config block (`hexkl_micro_hmx_config_size()`, read in the probe's
`session_info`) comes off the top of each.

**The feed at a smaller arena.** At 6.5 or 5.8 MiB the fit rule
`2 × gu_bytes ≤ arena` (7.3 MiB at LFM2.5) fails and the M=1 call falls
back to the arena read (21–27 GB/s): that cell is measured too, as the
"no feed" bound, but it is not a candidate. The candidate is a **half-slab
ring**: the gate_up push splits into two halves (`moe_push_gate_up_chunks`'s
chunking already exists for M > 1), a ring of **three half-slabs**
(3 × 1.83 = 5.5 MiB) keeps one whole expert in flight while one half
computes — the same bytes-in-flight as today's two slabs — and a down
(1.75 MiB) lands in one freed half-slab. Cost: one extra pool run per
expert (4 per call) and one more ring wait; the schedule is provable in
`moe_layer_host_check.c`'s scoreboard, so the host gate stays exact. A
flag bit (`hexkl_moe_flags`, `NNTR_MOE_FEED_HALF`) selects it; default off.
The bytes and the int32 sums do not change, so dumps stay `bit_identical=1`.

**Cells (one sitting, `R3CY10WM83Y`-class unit, ≈ 40 min).**

| cell | how | reads |
|---|---|---|
| `moe_dsp_us[cap][feed]` | `MoeM1GemvFeedVsCompute`'s `vtcm` cell (K 2048, I 1792, N 2048, 4 experts, 20 reps, `dsp` and `mm` µs) under skel cap 8 / 6.5 / 5.8, feed = today / half-ring / none | the M=1 slowdown Δ per call |
| `s2_vtcm_kib[cap]`, `s2_fc_vtcm_ms[cap]` | Q4 re-run with S1 open under each cap: `session_info` on S2, then `fc_q4m1_f32` VTCM-fed at lanes 1..6 for K 7168 and 2048, `FC_RATE_PROJ` ms/token | the FC gain at that share |
| `ddr_conc[cap]` | Q3 with S2 VTCM-fed (was L2-fed) | whether S1's −17 % changes with S2's faster reader |
| `app_A_cap` | the app, A (nothing set) on the capped + half-ring skel, prompt 512, G 64 / 512, ×2, dumps, text | the slowdown **in the model** (the number that decides), dumps `=1`, text ≡ A |

**Break-even (computed here, read on the day).** The gain of VTCM-feeding
S2 is `gain = (8.06 − 7.88) + H`, where H is the prefetch of the next
FC's first per-lane slabs during S1's MoE window: ≤ 1.5 MiB / 45 GB/s ≈ 35 µs
per MoE layer, minus what that read takes from S1 on the shared 70 GB/s
(≈ 1.5 MiB / 70 GB/s ≈ 22 µs) → H ≈ 13 µs × 22 ≈ 0.3 ms/token at most. So
**gain ≈ 0.18 + 0.3 ≈ 0.5 ms/token, and the MoE may slow by at most
≈ 22 µs per call (≈ 5 % of 430 µs)**. The 4 ms/token the issue thread once
called a budget does not exist. Expectation, stated so the reading is not
fitted afterwards: four extra fork/joins per call are ≈ 20–40 µs on this
pool (the ponytail note at `hexkl_mm_u8i4_moe.c:589` prices six of them),
so the half-slab ring is expected to sit **at or above** the break-even; the
"no feed" fallback (arena read) is ≈ +300 µs/call and is out. **Stop rule:**
`Δmoe_dsp_us × 22 ≥ 0.18 + H_measured` (with H from `ddr_conc`) → the E2E
design stays L2-fed and S2 opens with 0 VTCM; the probe's cells go to
BENCHMARK either way.

**Rejected:** releasing 1.5 MiB *after* `hw_init` from S1 — HexKL owns the
context and exposes no release; the resource manager's cooperative
release callback would need HexKL to register it. Also rejected: S1
acquiring VTCM itself and calling `hexkl_micro_hmx_lock` without
`hw_init` — `hw_init` also votes HVX / HMX / DCVS power (`hexkl_utils_powerup_*`),
undocumented to separate.

### 3.2 Track 1 — the session split and the token protocol

**Split** (unchanged from plan 178 §3.1): S1 = `ROUTER_TOPK`, `MOE` (the
3840 MiB arena, HMX for prefill, the M=1 feed's VTCM, the router's 5.5 MiB
of params); S2 = `RMSNORM`, `FC` (conv in / out, q / k / v / o), `CONV1D_GATE`,
`QK_NORM`, `ROPE`, `ATTN_M1` (48 MiB cache), `ADD` (slot 0 = the residual),
`DENSE_FFN` (layers 0–1), final `RMSNORM`, `LM_HEAD` + argmax. Both sessions
receive the **same description**; S1's resident mask is `ROUTER_TOPK|MOE`,
S2's the complement, so `hexkl_graph_forward`'s stretch rule (run until
the first non-resident op) already cuts the list at the right places: S2
runs `[0, r1)`, S1 `[r1, r1 + 2)`, S2 `[r1 + 2, r2)`, … The validator's
cross-kind rules become per-session: ROUTER_TOPK needs its MOE (holds in
S1), ADD needs every RMSNORM (holds in S2); the ARM rule "ROUTER_TOPK needs
ADD" (`:1342`) is replaced by "mask1 = ROUTER_TOPK|MOE, mask2 = the rest,
mask1 | mask2 = every kind".

**Hops per token.** 22 MoE layers × (S2 → S1: the normed row, 8 KiB; S1 →
S2: the MoE output, 8 KiB) = **44 mailbox hops** + ARM → S2 (embedding
row in, 8 KiB, a dspqueue packet), ARM → S1 (one "serve 22 rounds at pos"
packet), S2 → ARM (token id, 4 B; logits 512 KiB only under
`NNTR_PPL_DECODE` / the shadow). **One dspqueue packet per session per
token**; the ARM is out of the 44 inner hops. Per-layer-group calls
(ARM-brokered, 44 × 12.7 µs ≈ 0.56 ms + 44 ARM wake-ups) are the
rejected alternative: measured 4–5× the mailbox and they put the ARM's
scheduler in the loop.

**Mailbox protocol** (the probe's `nntr_hvx_mailbox.c`, made a session
object): one ION page (16 KiB: two 128 B sequence lines + one 8 KiB
payload region each way) allocated by the ARM, `fastrpc_mmap`'d into both
effective domains, `HAP_mmap_get` in each PD once at `token_driver_start`.
S2 posts `ping = 2·layer + 1` after writing its row and cleaning the payload
and the line (`qurt_mem_cache_clean` FLUSH); S1 spins on `ping` (flush-
invalidate the line before each read), reads the row (flush-invalidate the
8 KiB), runs `[ROUTER_TOPK, MOE]`, writes the output, posts `pong`; S2
mirrors. Each side spins for the other's expected duration (S1 waits ≈ 0.5
ms while S2 runs a layer, S2 ≈ 0.45 ms while S1 runs the MoE) and then
polls every 50 µs with a 1 s timeout that fails the token call
(`AEE_EEXPIRED`; the ARM turns it into the usual throw — no silent
fallback). **Who polls:** S1's dspq thread (it already spins between MoE
calls, rule 40) and S2's forward thread; both are the sessions' own
FastRPC-created threads, not pool lanes, so the FC's 6 lanes and the MoE's
pool are untouched (`hop_mbox_thread_cost_pct` read ≈ 0). Cache
maintenance is per PD as above; the ARM never touches the page during a
token, so no ARM-side maintenance is needed between hops (the page is
`rpcmem` uncached on the HLOS side, as the probe's).

**Sequence per token** (`pos`): the ARM writes the embedding row (the tied
table's Q4_0 row dequantized on the CPU as today, 8 KiB) into S2's dspq
activation buffer, posts `HTP_DSPQ_OP_TOKEN{pos}` to S1 (its loop: 22 ×
{wait ping, forward 2 ops, post pong}) and to S2 (its loop: forward from
op 0; at each `resume_at < n_ops` post ping, wait pong, copy the MoE output
into the ADD op's input slot, continue at `resume_at + 2`; at the end
argmax → the response). The ARM blocks on S2's response (the dspqueue's
blocking read, no ARM spin: the token takes ≈ 20 ms), reads the id, appends
it, and — under `NNTR_PPL_DECODE` — reads the logits from a 512 KiB ION
buffer S2 fills. S1's response carries its 22 rounds' rc and pcycles.

**Weights into S2 at load.** The 74 slots (66 FCs in five shapes + the
lm_head in 8 slices of 16 384 rows, 383 MiB) go into a **mapped ION arena
on S2** (the same `tryChunk` / `place` / `arena_attach` path as S1's, on
effdom2, 256 MiB chunks), written by the CPU in `Q4M1` order before the
attach, read by the FC's per-lane DMA with `src_bypass = 1` (rule 43's
argument: never written on the DSP). New IDL `q4m1_attach(arena, off, K, N)
→ h` beside `q4m1_register`; `NNTR_HVX_Q4M1_SLOTS` 80 (the probe's). S2's
DSP heap then holds only the ATTN_M1 cache (48 MiB), the graph (≈ 2 MiB),
the L2 scratch (2 MiB) and the slots table: **a session never grows its
heap to the end of its space** (the #178 leak rule), and the app run after
S2's teardown is a sitting step. Startup: 383 MiB of reorder + copy on the
ARM (≈ 0.3–0.5 s at memcpy rates) + 2 × 256 MiB attaches; a `load_ms` cell.

**ARM driver loop.** `HtpBackend` gains `handle2()` / `effdom2()` opened
by the probe's sequence (reserve → effective domain → URI → unsigned
control → `nntr_hvx_open`); `HtpComputeOps` sends the description to both
(`graph_init` ×2), binds params to the session that owns the kind
(gammas, conv, RoPE table, ATTN_M1 cache → S2; router → S1), registers the
FC set into S2 and the MoE set into S1 as today. The per-token entry: with
every kind resident the list is one stretch, so `runStretchOp` already
makes op 0's hook keep the row and the last hook run the call; the last
hook becomes the `LM_HEAD` one (new, in `tie_word_embedding.cpp`'s M=1
path), and `invokeForward` for a stretch that is the whole list calls the
token driver instead of `nntr_hvx_forward`. The CausalLM's sampler takes
the id from the hook's output (`causal_lm.cpp:328-331` skipped when the
hook returned one; the logits are still filled under `NNTR_PPL_DECODE`).
`decode_kv_seed_fp32` and the conv-state re-seed go to S2's handle.

**Prefill** stays as today (HMX in S1, the CPU for the rest); the first
decode token after a prefill re-seeds S2's conv state and KV cache from
the CPU's copies exactly as #130 does now.

### 3.3 IDL changes (additive, after `mailbox_run`)

* `q4m1_attach(in uint32 arena, in uint32 off, in uint32 K, in uint32 N, rout uint32 h)`;
* `token_driver_start(in int32 mbox_fd, in uint32 mbox_bytes, in uint32 role, in uint32 spin_us)` / `token_driver_stop` — role 0 = S2 (main), 1 = S1 (MoE server); maps the page, records the role;
* dspqueue op `HTP_DSPQ_OP_TOKEN` (wire: `{op, seq, flags, pos}`; S2's buffers: act in, logits out (512 KiB) — `HTP_DSPQ_BUF_BYTES` grows for S2's out buffer only; response `{seq, rc, id, n_hops, hop_us, pcyc[S]}`);
* `graph_init` takes the resident mask per session (a word of the description already carries the resident bits; the ARM rewrites them per session before sending).

`FC` / `DENSE_FFN` / `LM_HEAD` graph kernels: `graph_op_fc` = `hvx_q4m1_prep`
(the HVX quantizer, `= q8_0_quant_cpu_det`) + `hvx_q4m1_gemv_groups`
(`hvx_intrin`, per-lane feed: VTCM if the session has ≥ 2 × gbytes × lanes,
else the L2 scratch) per weight handle; q / k / v = three handles, one
quantization; `graph_op_dense_ffn` = up, gate, `swiglu_cpu`, down;
`graph_op_lm_head` = 8 slices + `argmax_first` (logits kept in a heap
buffer for the response). The `_det` specs stand before every quantizer
(doc 45 §3.3): RMSNORM (#164) before every FC, `swiglu_cpu_det` before the
down FC, `CONV1D_GATE` before out_proj — all already resident and exact.

### 3.4 Contract §2 and doc 45 §3 check

* Three walls: wall 1 (M=1 GEMV) and wall 2 (DMA rate) unchanged in S1;
  wall 3 (transport) goes from 22 dspqueue packets to 2 packets + 44
  mailbox hops (≈ 0.16 ms/token). No CPU fallback for `QS4CX_WH`: unchanged.
* Arena budget: S1's 3840 MiB untouched; S2's arena ≤ 512 MiB in its own
  4 GiB space; heap rule as above.
* DMA hidden behind compute: S1 as today; S2's FC feed is double-buffered
  per lane (`nntr_hvx_fc_q4.c:196-212`); the cross-op prefetch (§3.1 H) is
  taken only if track 2 passes.
* Rule 44: the design counts the sequential numbers only; the concurrent
  cell is information. Rule 45: bit-identical + text 8/8, no PPL route.

### 3.5 Per-token budget (ms) and the shortfall, stated honestly

| term | G=64 | G=512 | G=1024 | source; what must still shed |
|---|---|---|---|---|
| MoE, 22 × 0.418–0.43 | 9.2–9.5 | 9.2–9.5 | 9.2–9.5 | rule 43 (57 GB/s in-app; floor at 70 GB/s = 7.6, not a lever we hold) |
| FC + lm_head, L2-fed `hvx_intrin` | 8.06 | 8.06 | 8.06 | #178 Q4; VTCM-fed 7.88 (track 2); the bytes at 57 GB/s are 7.05 → a hand-scheduled inner loop must shed **≈ 1.0** |
| activation quantizer (HVX) | 0.4 | 0.4 | 0.4 | ISS 10.7 k pcyc at K=2048 × 74 calls; **S3 measures** |
| router, 4 lanes | 0.5 | 0.5 | 0.5 | ≈ 50 k pcyc × 22 (scaled from 427 784 → 4.4 ms); **S3 measures**; target ≤ 40 k |
| ATTN_M1 (round 2), 6 layers | 0.3 | 0.6 | 1.1 | 0.18 ms/layer at G=1024; round 3 (#170) owns the rest |
| RMSNORM ×49 + QK_NORM ×6 | 0.4 | 0.4 | 0.4 | #164 G5 (≤ 4 k / 25 k pcyc/op); contract §12's "≈ 0.4 vs 0.1 on the CPU" |
| ROPE, CONV1D_GATE ×18, ADD ×48, SwiGLU ×2, argmax | 0.3–0.5 | 0.3–0.5 | 0.3–0.5 | assumed; the profile line reads it |
| hops: 44 mailbox + 2 packets + 2 responses | 0.16–0.2 | same | same | #178 Q2 |
| ARM per token (embedding row, append, packets) | 0.2–0.3 | same | same | assumed |
| **total** | **19.5–20.4** | **19.8–20.7** | **20.3–21.2** | |
| tok/s | 49–51 | 48–50.5 | 47–49 | |
| A of the #178 sittings (this unit) | 51.3–54.4 → 18.4–19.5 ms | 53.5 → 18.7 | 51.2–52.0 → 19.3–19.5 | |

Read plainly: **the E2E path projects at parity with 50 tok/s and 1–2 ms
behind the sitting's A at every G**, before the two S3 numbers land. To pass
B4 it must shed ≈ 1.5–2.5 ms: the FC loop (−1.0, toward the 57 GB/s byte
floor), ATTN_M1 round 3 (−0.5 at G=1024), the router (−0.2), the VTCM feed
+ prefetch if track 2 passes (−0.5). The MoE term is the same in both
paths and cannot move the comparison. The plan therefore builds the path
to B1–B3 (bit-identical, one call per token) and reads B4 as the
distance; it does not promise B4.

## 4. Steps (each ends in a rung of `.claude/skills/hexagon-gates`)

**Track 2 first** (one PR `htp/132-vtcm-share`, stacked on `htp/178-probe`).

* **T1. The query wrap + cap.** `nntr_hvx_vtcm_cap.c`, `build.sh` variant
  (`HEX_EXTRA_CFLAGS=-DNNTR_VTCM_CAP_KB=… HEX_EXTRA_LDFLAGS=-Wl,--wrap=compute_resource_query_VTCM`),
  three skels copied to `libnntr_hvx_skel.cap{8192,6656,5939}.so`.
  `session_info` prints `vtcm_size`. Gate: rung 2 ×3 (`UNDEFINED SYMBOLS OK`,
  md5s recorded; the default build has no wrap: `nm` shows no `__wrap_`).
* **T2. The half-slab ring.** `hexkl_mm_u8i4_moe.c` M=1 feed with
  `NNTR_MOE_FEED_HALF`; `moe_layer_host_check.c` proves the schedule
  (every slab reuse after its join, every wait before its read). Gate:
  rung 1 (`ALL CHECKS PASS`, `*Lfm2Moe*` 6 passed, `INPROC E2E PASS` with
  `bit_identical=1` under the flag), then rung 2.
* **T3. The cells.** `MoeM1GemvFeedVsCompute` gains the feed flag column;
  `TwoSessions.Q4` / `Q3` re-run under a capped S1 (the runner swaps the
  skel via `ADSP_LIBRARY_PATH` per cap). Gate: rung 3 (both gtests built,
  md5s; the app set unchanged = A's md5s).
* **T4. Device sitting (unavoidable)** — handoff
  `docs/measurements/132-vtcm-share.md`, ≈ 40 min, phone rebooted first,
  an app sanity run after every probe (the #178 rule). Variants (≤ 4):

  | variant | what |
  |---|---|
  | **A** | unchanged reference, default skel: full E2E, prompt 512, G 64 / 512 / 1024 ×2 |
  | **P** | the two gtests under cap 8 / 6.5 / 5.8 × feed today / half / none |
  | **Ah** | A's app on the cap-6.5 + half-ring skel (`NNTR_MOE_FEED_HALF=1`): prompt 512, G 64 / 512 ×2, dumps, text — the in-model Δ |

  Fold: the break-even line of §3.1 with `H_measured`; the verdict
  (VTCM-fed or L2-fed) goes into this plan's §3.2 as one sentence and into
  the LEDGER.

**Track 1** (one PR `htp/132-part-b`, stacked on #175 and #180 until they
merge; the shadows as `dev/*` measurement commits).

* **E1. Kernels and one-session all-resident on the host.** `graph_op_fc`,
  `graph_op_dense_ffn`, `graph_op_lm_head`, `q4m1_attach`, the `LM_HEAD`
  hook, the validator's per-session masks (host stand-ins run one session
  with the full mask). Gate: rung 1 — `run_inproc_e2e.sh` prints
  `calls/token=1` for tiny / hd64 / lfm25, `E2E tokens fwd==off 8/8`,
  `bit_identical=1` on the golden evals; `run_host_checks.sh` `ALL CHECKS
  PASS` (the FC / SwiGLU / argmax specs already there).
* **E2. The mailbox session object and the token driver on the DSP.**
  `nntr_hvx_token.c` (both roles over `hexkl_graph_forward`), the dspq
  `OP_TOKEN`, `token_driver_start/stop`. Host: `token_host_check.c` runs
  S1 and S2 roles on two pthreads over one in-process graph split by mask,
  10 000 tokens on the hd64 fixture, output ≡ the one-session E1 run
  (`TOKEN DRIVER BIT-IDENTICAL`, hops = 44 × tokens, timeouts 0). Gate:
  rung 1, then rung 2 (IDL grew: skel md5).
* **E3. Second session on the ARM.** `HtpBackend::handle2()`, effective
  domains through `tryChunk` / `dspqCreate` / mmaps, the two `graph_init`s,
  param routing, the FC arena on S2, `load_ms`, the teardown order (S2's
  arena unmapped, `nntr_hvx_close(h2)`, never `FASTRPC_SESSION_CLOSE`).
  Gate: rung 3 (`build_android.sh --htp`, NEEDED lines, `strings` count,
  md5s; `ndk-build` of `unittest_hvx_two_sessions` + `unittest_hvx_softmax`).
  Host cannot open two PDs: stated, not claimed.
* **E4. Shadows rebased** on the E2E build: `dev/fc-shadow` (+ DENSE_FFN
  and argmax tags), `dev/norm-shadow-164`, `dev/attn-shadow-170-r2`, and
  `NNTR_HTP_DUMP` for the MoE. Gate: rung 3 for the dev set; the shadow's
  in-process run on hd64 (ADD / router / norm / attn records equal; FC
  records are a device reading, x86 has no aarch64 order).
* **E5. Device sitting (unavoidable)** — handoff
  `docs/measurements/132-part-b-e2e.md`, ≈ 90 min. Variants:

  | variant | what | runs |
  |---|---|---|
  | **A** | unchanged reference (hybrid, switch off) | full E2E, prompt 512, G 64 / 512 / 1024 ×2, mirrored `A E E A`; 8 prompts G=256 with `NNTR_PPL_DECODE` |
  | **E** | `NNTR_HTP_FORWARD=1`, all kinds, two sessions, L2-fed S2 | same cells; `calls/token=1.00`; `NNTR_HTP_PROFILE=2` once per G for the per-op lines; dumps; `load_ms` |
  | **Ev** | E with S2 VTCM-fed (only if T4 passed) | G 64 / 512 / 1024 ×2 |
  | **S** | E + the shadows, G=8 forced on A's ids | B2's records |

  Stop rules: `0x8000040e` (stale skel) anywhere; `s2_open` > 2 s; a token
  call `AEE_EEXPIRED` (a hop timed out: the FARF names the side); the app
  after the sitting fails to register its arena (a mapping leak → reboot,
  note it). A's cells run first and last.
* **E6. Fold.** B1–B3 decide whether the E2E path is *correct*; B4 whether
  it becomes the default. If B1–B3 pass and B4 fails, the issue's residency
  part is done and the sheds of §3.5 are filed as their own items.

## 5. Risks (host vs device)

* **The wrap may not reach HexKL's query.** If `hexkl_micro_hw_init`
  resolves `compute_resource_query_VTCM` at load rather than at the static
  link (it is weak-undefined in the `.a`, so `--wrap` should bind), the cap
  is inert: T4's control cap-8 vs cap-6.5 `vtcm_size` shows it at once;
  fallback = the `attr_set_vtcm_param_v2` wrap (same mechanism, next symbol).
* **The VTCM window may not free the remainder** on this firmware even
  with a smaller request (the SDK text is v79-generic): Q4's
  `vtcm_avail_kib` under the cap is the reading; then track 2 stops and
  the design is L2-fed.
* **The half-slab ring's cost is a pool-run cost the host cannot time.**
  The ISS is out (contract §12); the scoreboard proves the order, T4 the
  µs. Ah's in-model cell, not P's isolated one, is the verdict (rule 11).
* **DMA rate / DVFS / thermal.** Every cell is read inside one sitting
  against its A, mirrored, with zone0 ≤ 35 °C before each block; the
  concurrent DDR cell is information only (rule 44).
* **Stale skel and two binaries.** The IDL grows twice (track 2: none;
  track 1: +3 entries + a dspq op): each sitting's app, skel and gtests
  come from one tree, md5s in the handoff, `HvxFcQ4.MatchesSpecBitExact`
  as the canary.
* **Address-space budget.** S2: 383 MiB arena + 48 MiB cache + ≈ 5 MiB in
  a 3584 MiB-mappable space — no wall; S1 unchanged. The #178 leak came
  from a heap grown to the end: S2's heap stays ≈ 55 MiB by design and the
  post-sitting app run checks it.
* **Hop hazards.** A stale read (cache line not invalidated) shows as a
  wrong token, caught by B2 / B3 (the mailbox's probe words stay in the
  session object as a per-hop check under `NNTR_HTP_PROFILE ≥ 2`); a lost
  post shows as `AEE_EEXPIRED`, never a hang (1 s cap).
* **The two pending S3 numbers** (HVX quantizer, 4-lane router) and the
  assumed small-op sum move the §3.5 total by up to ± 1 ms; the profile
  line in E replaces them with measured pcycles.
* **Text gate at G=256 across 8 prompts** depends on every op's
  bit-identity, including `expf` (bionic; G0 re-run each sitting) and the
  f16-subnormal `d` cases (G1 mutants). One flipped bit anywhere is a text
  fail with no partial credit — B2's per-op records say where.

## 6. Docs to update

* **`docs/htp_moe/BENCHMARK.md`**: a #132 Part B side table — track 2's
  `moe_dsp_us[cap][feed]`, `s2_vtcm_kib[cap]`, `s2_fc_vtcm_ms[cap]`,
  `ddr_conc[cap]`, Ah vs A rows and the break-even line; track 1's A / E /
  Ev rows (G 64 / 512 / 1024, prefill, `calls/token`, `load_ms`, text ≡,
  nll ≡, dumps), the per-op `pcyc/op` lines per session, the B2 record
  counts; artifact rows (skels per cap, app set, gtests).
* **`docs/htp_moe/LEDGER.md`**: new rules — *a skel can cap HexKL's VTCM
  request by wrapping `compute_resource_query_VTCM`, and the v79 VTCM
  window gives (or does not give) the remainder to a second PD*; *the
  M=1 feed at 5.5 MiB (half-slab ring) costs Δ µs/call*; *the two-session
  token protocol: 2 packets + 44 mailbox hops = X ms/token, bit-identical*;
  *the E2E path reads Y tok/s against A — the distance and its terms*.
  Open items: ⑨ and ㉓ (resident set) get Part B's outcome; ㉘ (CPU-exact
  FC) the E cells; ㉚ (#178's decision) closes with D; new items for the
  sheds §3.5 names (FC loop toward 57 GB/s, router ≤ 40 k, ATTN_M1 round 3
  already #170). §2 verdict rows for the two sittings. Contract §12: D's
  row is already there; add the track-2 verdict when read.
