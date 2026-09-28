# Plan 141 step 2 — the 22 per-token MoE calls through dspqueue, arithmetic untouched

Issue #141 (p1), step 2, re-scoped by the user's direction change of
2026-09-28 (contract `0001` decisions table: bit-preserving levers only; the
gate is text identical to A over several prompts **and** `NNTR_HTP_DUMP` MoE
dumps `bit_identical=1` against A). Base: `htp_moe` @ `046e7c7a`, which
includes PR #147 (step 1, the microbench). The plan builds on #147's
merged code:
* the `dspqueue_*` WEAK-import guard and the `dspqueue_` allow-list entry in
  `test/htp/build.sh`;
* `htp_set_latency_qos(h, poll_us)` (`test/unittest/htp_rpc_bench.h`);
* the dlsym `DspqApi` (`test/unittest/unittest_hvx_dspq_bench.cpp:51–78`);
* the DSP pattern in `test/htp/nntr_hvx_dspq_bench.c`.

Step 1 (`134-132-combined.md` on `origin/htp/134-132-sitting`, § Ride-along
#141, `R3CY10WM83Y`): F12 FastRPC 87.1 µs, QSS12 (both sides spin) 16.7,
QBS12 (DSP blocks, ARM spins) 28.9, QBB12 76.9, QSS0 3.7, F12p (F12 with a
spinner parked) 91.1. Rule 35 puts today's M==1 MoE transport at 82–92
µs/call.

## 1. Goal and gate

Goal: the switch-off decode path's 22 M==1 MoE calls per token go
ARM → DSP through one dspqueue request/response pair each instead of one
FastRPC invoke each. The DSP runs **the same C function** the FastRPC method
runs, with the same argument bytes. Expected: transport 82–92 → 17–29
µs/call, i.e. −1.3 to −1.6 ms/token. At A = 36.45 tok/s (27.4 ms/token, #134
A, G=512) that is ≈ 38.3–38.7 tok/s (+5–6 %).

Device gate. Q = `NNTR_HTP_DSPQ=1` against A = switch off, one sitting,
mirrored. All of these must hold:

| # | check | where it is read | pass |
|---|---|---|---|
| G1 | MoE dumps, Q and Q0 against A1, prompt 512, G=64 | `htp_dump_eval.py --label q A1 Q` (+ `q0`, and the null `a2 A1 A2`) | `bit_identical=1` on every file of A1's manifest, prefill and decode calls included. The null A2 must be `=1`, or the sitting cannot judge G1 |
| G2 | text, 8 prompts (§4 steps 6 and 7d), G=64 | byte `cmp` of the generated text, Q vs A | 8/8 identical |
| G3 | text in every tok/s cell | Q and Q0 against A of the same G and run | identical |
| G4 | decode tok/s (all), G 64 and 512, 2 mirrored runs | BENCHMARK.md #141 rows | Q mean > A mean at both G by more than A's own r1/r2 spread. Expected +5 % |
| G5 | mechanism: `NNTR_HTP_PROFILE=2`, G=64, one run each of A and Q | the M==1 bucket line `transport=` | Q ≤ 35 µs/call (A ≈ 82–92). `dsp=` and `mm` within ±3 % of A: the kernel is the same, only its thread changes |
| G6 | prefill tok/s | same cells as G4 | Q ≥ 0.95 × A at each G (the prefill calls stay on FastRPC and the queue does not exist yet while they run) |
| G7 | banners | Q / Q0 logs | `[HTP] dspq: on …` once. At exit `[HTP] dspq: close calls=N served=N bad=0` with N > 0. A variant without the `on` banner is **void**, never "at A's speed" (rule 36) |

Standing gates. Prefill ≥ −5 % of A (G6). Text: the direction change makes
**A** the reference, not the CPU run. The CPU `q40` text column stays
information (rule 39, ⑱). No `NNTR_L2_DIFF` column: no DSP arithmetic
changes, and G1 is the sharper check. No PPL column: that column is for
changes that cannot be bit-identical (decisions table, 2026-09-28).

Host gate: rungs 0–3 of `hexagon-gates`, plus the new in-process lines of
§4 step 4 (`E2E eval dspq-* … bit_identical=1`, `E2E dspq off-path … bit_identical=1`).

Default flip: the switch stays **off by default** after the sitting. Making
it the default is a user decision once G1–G7 pass (rule 33's form).

## 2. Where it lives

Line numbers were verified on `046e7c7a`.

**DSP side**
* `test/htp/nntr_hvx.idl`. Append two methods after `dspq_bench_stop`
  (`nntr_hvx.idl:629`, the last method): `dspq_start(in uint64 queue_id, in
  uint32 spin_us)` and `dspq_stop(rout sequence<uint32> res)`. Existing
  method indices stay as they are.
* **New** `test/htp/nntr_hvx_dspq.c` (≈ 180 lines). It holds the queue
  thread, `nntr_hvx_dspq_start / _stop`, and `nntr_hvx_dspq_shutdown(s)` for
  `close`. The `dspqueue_*` symbols are `#pragma weak`, in the same form as
  #147's `nntr_hvx_dspq_bench.c`: the same five symbols, so the skel's import
  count stays at #147's 51.
* `test/htp/nntr_hvx_session.h:53–73`, the struct: one field,
  `struct nntr_hvx_dspq *dspq; /* NULL = no queue */`.
* `test/htp/hvx_add_f32.c:184` `nntr_hvx_close`: its first statement becomes
  `nntr_hvx_dspq_shutdown(s)`, which runs before `hexkl_graph_free` and
  before the pool and the scratch are freed.
* The entry the thread calls is the existing one, unchanged:
  `test/htp/nntr_hvx_mm_u8i4.c:995` `nntr_hvx_mm_u8i4_moe_layer` and `:1024`
  `_timed`, which lead to `hexkl_mm_u8i4_moe_layer_run`
  (`hmx/hexkl_mm_u8i4_moe.c:1020`).
* `test/htp/build.sh:77` `SRCS` (already lists `nntr_hvx_dspq_bench.c`): add
  `nntr_hvx_dspq.c`.

**Shared wire format**
* **New** `nntrainer/tensor/htp_backend/htp_dspq_wire.h` (C, ≈ 40 lines). It
  holds the request and response layout of §3.1, the op codes, and
  `HTP_DSPQ_MAX_MSG`. The ARM side and the DSP side include this one header,
  which is the precedent `htp_graph_desc.h` set.

**ARM side**
* `nntrainer/tensor/htp_backend/htp_rpcmem.h`: `HtpDspqApi`, the dlsym table
  moved here from `test/unittest/unittest_hvx_dspq_bench.cpp:51–78`. The bench then
  includes it instead of its own copy.
* `nntrainer/tensor/htp_backend/htp_backend.{h,cpp}`:
  * `pollUs()`: the value the constructor already reads from
    `NNTR_HTP_POLL_US` / 5000 (`htp_backend.cpp:85–95`).
  * `atClose(std::function<void()>)`, run by `~HtpBackend`
    (`htp_backend.cpp:114`) before `nntr_hvx_close`. This is the "explicit
    shutdown hook on HtpBackend" that `htp_compute_ops.cpp:3771–3777` names
    as its own upgrade path.
* `nntrainer/tensor/htp_backend/htp_compute_ops.cpp`:
  * `invokeMoeLayer` (`:2604`). Under `invoke_mutex_` (`:2615`), after the
    staging `stagedMemcpy`, the two `nntr_hvx_mm_u8i4_moe_layer[_timed]`
    calls (`:2648–2664`) become one choice: `dspqCall(...)` when
    `M == 1 && kind == 0 && dspqReady(...)`, FastRPC otherwise. The error
    text (`:2667–2694`), `dumpMoeCall("moe_layer", …)` (`:2696`, entry label
    **unchanged**, so the manifests of A and Q are byte-comparable), the
    profile's `addInvokeMoeLayer` (`:2699–2707`) and the level-3 repeat loop
    stay shared.
  * New members next to `act_pool_ / out_pool_` (`:3823`): the queue handle,
    two queue-owned 64 KiB ION buffers (act, out), the sequence counter, the
    state (untried / on / off) and the call count.
  * `HtpProfile`: `addInvokeMoeLayer` takes a `via_dspq` bit. The `staging:`
    line (`:764–776`) then prints `via=dspq msg=<n> B` in place of
    `non-ION in-args=6/<n> B`, and the bucket line keeps `transport = host −
    dsp`, so it means what it meant before.
* The `NNTR_HTP_FORWARD=1` path (`runStretchOp`, `:1238`) is **not** routed.
  The per-token resident path is parked (decisions table) and keeps FastRPC.

**Host in-process build**
* **New** `test/htp/host/inproc/dspqueue_standin.c` (≈ 150 lines, §3.6).
* `nntrainer/tensor/htp_backend/meson.build:74–99` `inproc_sources`: add
  `nntr_hvx_dspq.c` and `dspqueue_standin.c`.
* `test/htp/host/run_inproc_e2e.sh`: the four runs of §4 step 4.

**Consumers checked, no change:**
* The ARM stub is regenerated by meson from the IDL
  (`generate_stub.sh` / `meson.build`). The skel is rebuilt by `build.sh`.
  Both are rebuilt because the IDL changed, and rungs 1–3 cover them.
* No weight format change: the `nntr_quantize_stream` format tag and the
  loader check are untouched.
* The `NNTR_HTP_PROFILE` stage table (`HTP_MOE_N_STAGES`, 31 slots) is
  unchanged. The timed call returns the same slots, now through the
  response message.
* `tools/htp_fc_report.py` reads only the gtests' `FC_STAGE` / `FC_FIELD`
  lines (`:37`), not the MoE profile. No change.
* `tools/htp/htp_dump_eval.py`: no change.

## 3. Design

### 3.1 Packet layout: what rides where

| data | carrier | why |
|---|---|---|
| `op, seq, flags (bit0 = timed), M, K, inter, N_out, n_experts, n_rows` | request **message** header, 9 × u32 | small, per call |
| `h_gate_up[n_e]`, `h_down[n_e]`, `row_count[n_e]` (u32), `row_index[n_rows]` (u32), `row_weight[n_rows]` (f32) | the same message, after the header | at decode n_e = 32 and n_rows = 4: 36 + 384 + 32 = **452 B**. A message is a copy into the queue's shared memory with no cache maintenance. A buffer reference costs a flush plus an invalidate (QSS12 − QSS0 = 13 µs for two 12 KiB refs) |
| activation `M × K` f32 (8 KiB) | buffer ref 0: the queue-owned ION `act` buffer, `fastrpc_mmap(FASTRPC_MAP_FD)` once at creation. Flags `REF \| FLUSH_SENDER \| INVALIDATE_RECIPIENT`, `size = M·K·4` | #147's measured pattern. Cache maintenance covers `size`, not the whole dma-buf (the FastRPC path pays for the whole 64 KiB class, rule 35's "what staging does") |
| output `M × N_out` f32 (8 KiB) | buffer ref 1: the queue-owned `out` buffer, `REF` in the request. The response hands it back with `DEREF \| FLUSH_SENDER \| INVALIDATE_RECIPIENT` | #147's pattern, echo-verified on silicon (`bad=0`, served 4400) |
| response | message `{seq, rc, stage_us[31] when timed}`, ≤ 132 B | the timed call's slots come back without a FastRPC `rout` |

The activation copy into ION stays exactly as it is today: `stagedMemcpy`
into the queue's `act` buffer instead of `stage(act_pool_)`. The FastRPC
staging pools are not touched, so the FastRPC path stays byte-for-byte what
it is. A call whose message would exceed `HTP_DSPQ_MAX_MSG` (4 KiB, i.e.
more than ≈ 300 experts or rows), or whose `act`/`out` exceeds the 64 KiB
buffers, takes FastRPC. That is only a `ponytail:`: every MoE layer of this
model has one shape.

Queue sizes: request 16 KiB and response 4 KiB, set explicitly in
`dspqueue_create`, so a 452 B packet always fits. At most one packet is in
flight (§3.4).

### 3.2 DSP side: same entry, so bit-identical by construction

`dspq_start(queue_id, spin_us)` runs on a FastRPC thread and does the
following:
* returns `AEE_EUNSUPPORTED` if any weak `dspqueue_*` is NULL;
* returns `AEE_EBADSTATE` if `s->dspq` is already set;
* calls `dspqueue_import` with no packet callback, so that blocking reads
  are allowed;
* mallocs a **64 KiB** stack. Lane 0 of the M=1 GEMV runs on the caller's
  stack, and the pool gives its workers 32 KiB for the same units
  (`hvx_worker_pool.c:40–43`); 64 KiB is 2× that. It is 64 KiB of the
  ≈ 182 MiB heap: an address-budget note, not a problem;
* creates one qurt thread at **the calling FastRPC thread's own priority**.
  That is the priority lane 0 runs at today and the one the pool was
  created with (`hvx_worker_pool.c:236`). #147 used +1; this plan uses +0.

The thread loop:
1. Read one packet (§3.3).
2. Check the message: `len == header + 12·n_e + 8·n_rows`, `nb == 2`, both
   ptrs non-NULL, `bufs[0].size == M·K·4`, `bufs[1].size == M·N_out·4`.
   A malformed packet is answered with `rc = AEE_EBADPARM` and counted as
   bad. The ARM never hangs on it.
3. Call `nntr_hvx_mm_u8i4_moe_layer((remote_handle64)s, M, K, inter, N_out,
   h_gu, n_e, h_dn, n_e, row_index, n_rows, row_count, n_e, row_weight,
   n_rows, bufs[0].ptr, M·K, bufs[1].ptr, M·N_out)`, or `_timed` with a
   local `stage_us[HTP_MOE_N_STAGES]` (31) when flags bit0 is set. **The FastRPC skel calls
   this same function, and its own argument checks
   (`check_moe_layer_args`, `check_moe_row_totals`) run unchanged.**
4. Respond `{seq, rc[, stage_us]}` with the two DEREFs.

Why nothing else can differ:
* **Session state.** The thread reads the same `s->weights_u8i4`,
  `vtcm_base`, `quant_pool`, `moe_scratch` and `moe_flags`. At most one
  entry runs at a time (§3.4).
* **DMA.** The thread change does not matter:
  `hexkl_mm_u8i4_moe_layer_run` calls `hexkl_dma_ring_reset()` at the start
  of every call (`hexkl_mm_u8i4_moe.c:1285`) and drains before its output
  copy (`:1491`, `moe_dma_copy`). No DMA chain spans the prefill's FastRPC
  thread and the decode's queue thread.
* **HVX.** The pool's own workers are plain `qurt_thread_create` threads
  that use HVX without `qurt_hvx_lock` (no call in the tree), so a
  non-FastRPC thread running HVX is already how every M=1 call works.
* **HMX.** The default M=1 path (GEMV + VTCM feed) does not touch HMX. The
  HMX loop (`NNTR_MOE_HTP_M1_GEMV=0`) would use the HMX lock that
  `nntr_hvx_open` took on another thread. That is a risk (§5), not part of
  the gate.
* **Reduction order.** It does not depend on which thread is lane 0: the
  pool's job split is by index. PR #143 fixed the job-boundary race that
  #136 exposed.

`dspq_stop(res)`:
* sets the stop flag;
* joins the thread;
* calls `dspqueue_close`;
* frees the stack;
* returns `res = {served, bad, empty_polls, spin_us}`.

`nntr_hvx_dspq_shutdown(s)`, called from `close`, does the same thing
without `res`. It covers an ARM side that never stopped the queue, and the
order of the two singletons' destructors.

### 3.3 Spinning vs blocking, and what each costs

| side | default | switch | cost |
|---|---|---|---|
| ARM, waiting for the response | spin on `read_noblock` up to `HtpBackend::pollUs()` (5000 µs, `NNTR_HTP_POLL_US`), then a blocking `dspqueue_read` with a 5 s timeout | the existing `NNTR_HTP_POLL_US` | **nothing new.** `RPC_POLL_QOS` already makes the FastRPC call spin the model thread for up to the same window. The call's ≈ 0.7 ms is spent spinning either way |
| DSP, waiting for the next request | **hybrid**: spin on `read_noblock` with `pause(#255)` between polls (≈ 0.25 µs, the pool's `hvx_worker_pool_pause`) for up to `spin_us` = **1000 µs** after each response, then a blocking `dspqueue_read` with a 100 ms timeout loop that checks the stop flag | `NNTR_HTP_DSPQ_SPIN_US` (0 = always block, the QBS config) | one DSP hardware thread busy-polls through the gaps between MoE calls. The decode gaps average ≈ 440 µs (9.7 ms outside the call / 22, §2 budget row), so it spins through 21 of the 22 gaps per token. The 22nd gap (lm_head, sampling, several ms) and every idle period end in a blocking read, with no idle cost beyond 10 wake-ups/s. The session's DCVS vote (`nntr_hvx_open`) already holds the clocks up between calls, so the spin adds dynamic power on one hardware thread, not a clock change. Power cannot be read on this setup: the handoff's thermal log is the proxy |

Why this default: QSS12 vs QBS12 is 12 µs × 22 = 0.27 ms/token (≈ 1 %
decode), and a 1 ms window captures it during decode while ending every
other idle period in a block. Always-spin (QSS for the whole process) is
rejected: it holds a spinning hardware thread for the whole process life,
including between generations. F12p (+4 µs on a FastRPC call with a spinner
parked) shows a spinner is not free for other DSP work. In the switch-off
decode there is no other DSP work between the MoE calls. The Q0 variant
(`SPIN_US=0`) measures the power-safe configuration in the same sitting, so
the default can be revisited with numbers.

### 3.4 Lifecycle and ordering

* **Create lazily at the first M==1 MoE call**, under `invoke_mutex_`, only
  when `NNTR_HTP_DSPQ=1`. That call is the first decode token's first
  layer; the prompt's prefill calls have already run on FastRPC. Creation
  does the following, once:
  * resolve `HtpDspqApi`;
  * `dspqueue_create(CDSP_DOMAIN_ID, 0, 16 KiB, 4 KiB, NULL, err_cb, this)`;
  * `dspqueue_export`;
  * allocate two `HtpRpcBuffer(64 KiB)` (they must be ION), each mapped with
    `fastrpc_mmap(FASTRPC_MAP_FD)`;
  * call `nntr_hvx_dspq_start(session, id, spin_us)`;
  * print the banner;
  * register `HtpBackend::atClose(teardown)`.

  The few-ms creation cost lands in the first decode token only.
* **In-flight invariant**: at most one DSP entry at a time.
  `invoke_mutex_` is held from `dspqueue_write` to the matching response.
  Every other `HtpComputeOps` FastRPC entry takes the same mutex
  (`:2227, 2281, 2347, 2507, 2615, 2750, 2822, 2890`). Registration and
  `sendMoeOptsOnce` run on the model thread before the call, in
  `gemm_qs4cx_moe_layer_fp32` (`:1203–1219`). The response's `seq` must
  equal the request's; a mismatch throws.
* **Teardown**: `~HtpBackend` runs the hook, which does:
  1. write `{op = QUIT}`;
  2. `nntr_hvx_dspq_stop(session, res)`;
  3. print `[HTP] dspq: close calls=N served=res[0] bad=res[1]`;
  4. ARM `dspqueue_close`;
  5. `fastrpc_munmap` of the two buffers.

  Only after the hook does `~HtpBackend` call `nntr_hvx_close`. The DSP
  `close` stops a thread that is still alive, as a backstop.
* FastRPC stays for everything else: open, QoS, registration, `moe_set_opts`,
  M>1 (prefill), the dense FFN (`kind 1`), the conv / attention / FC entries,
  the per-token entry, profiles' DMA trace reads, close.

### 3.5 Failure paths and the switch

* `NNTR_HTP_DSPQ` unset or 0: no queue and **no log line**. The code path
  and the logs are today's.
* `NNTR_HTP_DSPQ=1` and any creation step fails, for example:
  * a missing ARM symbol;
  * `create` / `export` fails;
  * a buffer is not ION or `fastrpc_mmap` fails;
  * `dspq_start` returns `AEE_EUNSUPPORTED` (the DSP image has no dspqueue);
  * `dspq_start` returns `0x8000040e` (the skel predates this IDL: the hint
    names `test/htp/build.sh`, rule 3).

  In every such case, undo what was created, print **one** banner
  `[HTP] dspq: off (<step> <err hex>) -- MoE calls stay on FastRPC`, and
  never retry. The run continues on FastRPC.
* After `on`, failures are loud, never a fallback, because the DSP may still
  hold the packet and a FastRPC retry could run the same buffers
  concurrently:
  * a response timeout (5 s), the error callback, or a `seq` mismatch throws
    `runtime_error("dspq: …")`;
  * a kernel `rc ≠ 0` throws the existing FastRPC message with `(via dspq)`
    appended.
* Switches, all read once:
  * `NNTR_HTP_DSPQ` (default off);
  * `NNTR_HTP_DSPQ_SPIN_US` (default 1000; 0 = block);
  * the ARM window reuses `NNTR_HTP_POLL_US`.

### 3.6 Host stand-in (in-process build)

The in-process build has no queue. `dspqueue_standin.c` supplies one:
* exported `dspqueue_create / export / import / close / write / read /
  read_noblock`;
* two fixed rings of 16 packets (4 KiB message + 2 buffer refs each), one
  mutex and two condvars;
* `import(id)` returns the "DSP end" of the same object;
* buffer refs resolve `fd → ptr` through `rpc_standin.c`'s table and return
  `AEE_ENOSUCHMAP` for a buffer that was not `fastrpc_mmap`'d, the device's
  own rule, so the ARM-side mapping step is actually checked;
* `NNTR_INPROC_NO_DSPQ=1` makes `create` fail, which drives the off path.

With this stand-in, the **real** `nntr_hvx_dspq.c` thread runs as a pthread
(`stub/qurt.h`) against the real `HtpComputeOps` marshalling. The host
proves the packing, the unpacking, the validation, the spin→block switch,
teardown and the off path. It cannot prove transport, DSP threads, cache
maintenance or power.

### 3.7 Rejected alternative

**The routing and the 64 handles in a third, pre-registered ION "args"
buffer** (or handles pre-bound per layer on the DSP, with only a layer id
in the packet). It is rejected for three reasons:
* A third buffer reference adds a flush and an invalidate per call. The
  refs are the measured expensive part (13 µs for two).
* Pre-binding adds DSP state that the graph entry already owns for the
  parked path.
* A 452 B message copy is below the resolution of the 16.7 µs round trip.

Also rejected: routing through `hexkl_graph`'s `forward` (the issue body's
original "run stretch [start, resume)" packet). That is the parked
per-token path. The re-scope asks for the switch-off calls with no change
to what runs.

## 4. Steps

1. **Wire header + DSP side.** Add `htp_dspq_wire.h`, `nntr_hvx_dspq.c`, the
   IDL pair, the session field, the `close` hook and the `SRCS` line.
   *Gate: rung 2.* `build.sh` prints `UNDEFINED SYMBOLS OK (51 runtime
   imports)`, the WEAK guard is silent (5 `dspqueue_*`, all WEAK: the same
   set as #147's), and the build is `-Werror` clean. Record the skel md5.
2. **ARM side.**
   * `HtpDspqApi` moves into `htp_rpcmem.h`, and #147's bench uses it.
   * `HtpBackend::pollUs / atClose`.
   * `invokeMoeLayer`'s choice, `dspqReady`, `dspqCall`, the teardown hook
     and the profile's `via` bit.

   *Gate: rung 0 (`clang-format-14`, changed lines) + rung 1 host build.*
3. **Host stand-in.** Add `dspqueue_standin.c` and the meson `inproc_sources`
   entries. *Gate: `ninja -C build_htp_host nntrainer/libnntrainer.so` links
   with `--no-undefined`.*
4. **In-process E2E lines** in `run_inproc_e2e.sh`, each against the
   switch-off dump the script already writes:
   * `NNTR_HTP_DSPQ=1` on the tiny fixture, then
     `E2E eval dspq-tiny … bit_identical=1` vs `dump_htp`;
   * `NNTR_HTP_DSPQ=1 NNTR_HTP_DSPQ_SPIN_US=0` on the lfm25 fixture (prompt
     512), then `E2E eval dspq-lfm25 … bit_identical=1` vs `dump_25off`;
   * `NNTR_HTP_DSPQ=1 NNTR_MOE_HTP_M1_GEMV=0`, then
     `E2E eval dspq-hmx … bit_identical=1` vs `dump_hmx`;
   * `NNTR_INPROC_NO_DSPQ=1 NNTR_HTP_DSPQ=1`, then `E2E dspq off-path
     banner=1 bit_identical=1`.

   Each switch-on log must show `dspq: on` and `close calls=N served=N
   bad=0` with N = the manifest's M==1 call count.
   *Gate: rung 1 in full.*
   * `run_host_checks.sh`: `ALL CHECKS PASS`, `WORKER POOL LANES OK`.
   * `tools/htp_syntax_check.sh` exits 0.
   * `run_inproc_e2e.sh`: every existing line unchanged, the four new
     lines, and `INPROC E2E PASS`.
5. **App + ride-along binaries.** *Gate: rung 3.*
   * `build_android.sh --htp --cache`: both `NEEDED` lines, and
     `strings libcausallm_core.so | grep -c NNTR_HTP_FORWARD_KINDS` ≥ 1.
   * New checks: `strings libnntrainer.so | grep -c 'dspq: on'` = 1, and
     `nm -D libnntrainer.so | grep -c ' U dspqueue_'` = 0, which proves
     dlsym only: a runtime without dspqueue still loads the library.
   * Rebuild #147's `unittest_hvx_dspq` against the moved `HtpDspqApi`.
6. **Prompt set** (docs commit, agent-system side). Add
   `docs/measurements/prompts/` with seven new prompt files and a `README.md`
   listing each file's name, domain, token count (measured with the model's
   `tokenizer.json` on the workstation) and md5. The eighth prompt is
   `docs/measurements/77-prompt512.txt` itself. Every prompt is ≤ 512 tokens
   (`init_seq_len: 512`):

   | file | content | ≈ tokens |
   |---|---|---|
   | `77-prompt512.txt` (p01) | the harbour narrative. It has a near-tie at decode step 1 (`town` p = 0.134, #134), the most sensitive prompt known | 512 |
   | `bitset-02-code.txt` | a buggy Python function + "explain and fix" | ≈ 300 |
   | `bitset-03-math.txt` | a multi-step arithmetic word problem, "step by step" | ≈ 150 |
   | `bitset-04-korean.txt` | a Korean paragraph + a question about it | ≈ 250 |
   | `bitset-05-json.txt` | an instruction to extract fields as JSON | ≈ 200 |
   | `bitset-06-dialogue.txt` | a chat-style multi-turn exchange | ≈ 350 |
   | `bitset-07-facts.txt` | an encyclopedic prompt (history / science) | ≈ 450 |
   | `bitset-08-short.txt` | "List ten …", a short prefill | ≈ 32 |

   The set is the reusable accuracy set for every bit-preserving lever of the
   direction change, not only for #141.
7. **Handoff `docs/measurements/141-dspq-moe.md`** (`hexagon-handoff`
   template). **The device step is unavoidable here**: transport, the DSP
   thread and silicon bit-identity cannot be shown on the host. The
   orchestrator runs it (the phone stays connected). ≈ 25 min.

   Variants (3 of 4, one binary set, environment only):
   * **A** = nothing set, the reference, run first;
   * **Q** = `NNTR_HTP_DSPQ=1`;
   * **Q0** = `NNTR_HTP_DSPQ=1 NNTR_HTP_DSPQ_SPIN_US=0`.

   Clean run dir, and `md5sum` on the device = the staged table. The config
   is #134's (`do_sample: false`, `init_seq_len: 512`, `moe_engine: htp`,
   `moe_htp_layers: ""`), with `NNTR_NUM_THREADS=8`.

   a. **tok/s**, prompt 512 (`77-prompt512.txt`), G = 64 and 512. Order
      `A Q Q0` for run 1, `Q0 Q A` for run 2. Text of each cell compared
      against A (G3). G=1024 is not run: the change is per call, not per
      position, and the user's scope is 64/512.

   b. **Profile** (not tok/s): `NNTR_HTP_PROFILE=2`, G=64, one run of A and
      one of Q. Paste the M==1 bucket line and the `staging:` line (G5).

   c. **Dumps**, prompt 512, G=64: `NNTR_HTP_DUMP=<dir>` for A1, Q, Q0, A2
      in that order. About 200 MB each (22 prefill calls × 8 MiB dominate).
      Pull them, run `htp_dump_eval.py` for `q`, `q0` and the null `a2`
      against A1 (G1), then delete them from the phone.

   d. **Text set**: the 8 prompts of step 6 at G=64, A then Q, one run each.
      A is bit-identical run to run (#136's dump sitting). Compare the texts
      byte for byte (G2). Paste A's p01 text once for the record; no approval
      step is needed when every text is identical. If a text differs, stop
      and file it: under the direction change that is a defect, not an
      approval question.

   e. Record the thermal log (battery / zone0) at the start, after a, after
      c and at the end, as the power proxy for Q vs Q0.

   Sanity before any cell:
   * `adb logcat`-free check: Q's first log shows `[HTP] dspq: on queue=…
     dsp_spin_us=1000 arm_spin_us=5000 buffers=2x65536 ion=y`;
   * Q0's shows `dsp_spin_us=0`;
   * A's shows no `dspq` line.

   Set the issue to `state:needs-measurement` after the PR is up.

## 5. Risks and how the table shows them

* **Stale skel** (rule 3). An old skel on the phone makes `dspq_start`
  return `0x8000040e`, and Q would silently run FastRPC. Guards:
  * the `off (dspq_start 0x8000040e …)` banner;
  * G7 voids the variant;
  * the skel md5 row.
* **The microbench does not transfer.** F12 moved 12 KiB of `add_f32`; the
  MoE call moves 8 + 8 KiB plus a 452 B message, and the DSP thread runs
  ≈ 700 µs of HVX between packets instead of a memcpy. G5 reads the real
  per-call transport next to `dsp` and `mm`. If Q's transport lands well
  above 29 µs, the profile line shows where it went.
* **`dsp` / `mm` move with the thread.** A different priority, or an HVX
  context contended with the spinner (Q), would show as `dsp` ≠ A in G5.
  Q0 against Q separates the spinner's share.
* **Thermal drift inside the sitting.** #134's prefill fell 564 → 419
  through one sitting. The mirrored order (`A Q Q0 / Q0 Q A`) and A's own
  r1/r2 spread in G4 keep that out of the verdict. G6 is read per mirrored
  pair.
* **DVFS**: the session vote holds the clocks. If Q still reads slower than
  predicted, compare the spin window's `empty_polls` in the close line
  between Q and Q0.
* **The DMA rate is not in play.** The weight feed is per call and on the
  same thread as the kernel (`:1285` reset). `mm` in G5 would show a change.
* **HMX from a non-FastRPC thread** (`NNTR_MOE_HTP_M1_GEMV=0` only). The
  host covers the wiring (step 4), but silicon is not in the gate. If
  someone runs that opt-out with the queue and it fails, the failure is a
  loud `rc`, not wrong numbers. Noted in the PR, not measured.
* **Address space**: +64 KiB stack and 2 × 64 KiB ARM-side ION (mapped into
  the DSP's 32-bit space) against 3840 MiB of arena + ≈ 182 MiB of heap.
  Negligible, and noted in the PR per the kernel review list.
* **Dump volume**: about 0.8 GB on the phone for step 7c. The handoff
  deletes it after the pull.

## 6. Docs to update (after the sitting)

* `docs/htp_moe/BENCHMARK.md`:
  * Results: #141 rows A / Q / Q0 × G 64 / 512 × r1 / r2 (prefill, decode
    all, last 64, text = A);
  * a side table with the M==1 `transport` / `dsp` / `mm` for A and Q, the
    G1 eval lines and the 8/8 text line;
  * Artifacts: the #141 set (skel md5 with `UNDEFINED SYMBOLS OK (51 runtime
    imports)`, the app md5s, the prompt set's md5s).
* `docs/htp_moe/LEDGER.md`:
  * ⑦ / wall 3: the M==1 transport per call under dspqueue, against rule
    35's 82–92;
  * §2: a verdict row for #141 step 2;
  * rule 35 corollary or a new rule if the MoE transport disagrees with the
    microbench (F12 → QSS12);
  * ⑨ / ㉓: #132 PR 2 stays parked (decisions table); the transport term of
    the §2 budget row becomes Q's.
* Contract §2 wall 3 ("Fix: dspqueue or a resident DSP worker, measurement B
  decides"): the supervisor records the answer.
* `docs/htp_moe/guide/` (01 run-it: the two new switches): the guide
  writer's follow-up.
