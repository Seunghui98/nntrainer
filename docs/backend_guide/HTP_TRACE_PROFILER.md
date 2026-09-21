<!-- SPDX-License-Identifier: Apache-2.0 -->

# nntr trace: one timeline for CPU, HTP (HMX / HVX / DMA) and QNN

Status: plan + prototype. The viewer and a synthetic sample trace live in
`tools/nntr_trace/`; nothing in the runtime emits a trace yet.

## 0. Why

The HTP work is tuned from numbers that live in three places that do not line
up: host wall-clock prints in CausalLM, `stage_us[]` arrays returned by the
`*_timed` FastRPC entries, and the five `hexkl_probe` accumulators inside
`layer_run`. Each is an aggregate. None of them says *when* something ran, on
*which* engine, or *what else* was running at the same time. The questions
the HTP docs keep asking cannot be answered from aggregates:

- Is HMX idle while HVX dequants, or the other way round (doc 35 §3)?
- Which of 140 FastRPC calls per decode token is transport and which is compute?
- Did this op run on the NPU, or did `HtpContext` silently fall back to CPU?
- Is the 6-thread pool tail-heavy (`tiles >= threads x 4` rule, doc 36 T6)?
- Where is our `mul_op`, the one 20x cycles/element outlier (doc 36 T5)?

QNN answers all of these with one file, the HTP optrace
(`*_chromeTrace_opTrace.json`): a Chrome Trace Event Format file with one
track per hardware unit (DMA at tid 256, HVX threads at 512-519, HMX at 768),
every micro-kernel instance as a span, and per-span cycle counts. We should
produce the same kind of file for our stack, with the host side included.

## 1. What exists today (facts)

Host side, on `main`:

| piece | where | what it gives | gap |
|---|---|---|---|
| `Profiler` + `PROFILE_TIME_*` | `nntrainer/utils/profiler.{h,cpp}` | per-item min/avg/max/sum, text table via `PROFILE_END` | no timestamp, no thread id, aggregated, `time_item_times` unguarded, `end()` erases so no nesting |
| `TimeTracer` / `MemoryTracer` | `nntrainer/utils/tracer.{h,cpp}` (`-Denable-trace`) | ms-relative points, RSS points, tab-separated files | point events only, ms granularity, `traceStart/End` are stubs |
| layer spans | `LayerNode::forwarding` / `incremental_forwarding` (`layer_node.cpp:789`, `:828`) | names `"<name>:forward(<type>)"` | keyed per instance; graph-level spans keyed per *type* overlap them |
| engine per layer | `LayerNode::getComputeEngineType()` -> `cpu / gpu / qnn / htp` | the CPU-vs-NPU track key | no thread-local "current layer", so tensor ops and pool workers cannot attribute themselves |
| threads | `ThreadManager` (`NNTR_NUM_THREADS`), `TaskExecutor` loadPool/UnloadPool (2+2), `ParallelBatch`, KleidiAI std::thread | the host tracks | threads are unnamed (`pthread_setname_np` unused) |
| CausalLM timing | `causal_lm.cpp:603-704` | prefill / generation ms, TPS, peak RSS | no per-token, no TTFT |
| QNN profiling | `qnn_context.cpp:169` (`ProfilingLevel::OFF`), `extractBackendProfilingInfo()` | `QnnProfile` plumbed through `graphExecute` | hard-coded off; never read after execute; output is `ml_logd` only |
| HTP backend | `htp_compute_ops.cpp:32` `get_htp_ops() { return get_cpu_ops(); }` | skeleton | no accelerated op on `main`; sdkl macro-API is opaque |

DSP side, on the HTP branches (`claude/htp-lfm2-moe-ffn`, `htp/attention-handoff`, ...):

| piece | where | what it gives |
|---|---|---|
| `hexkl_probe` | `htp_backend/hmx/hexkl_probe.{h,c}` | 5 global microsecond accumulators (acc_read, acc_copy, dequant, quant, drain), qtimer based, gated by `hexkl_probe_on` |
| `stage_us[]` | `attn_forward_timed`, `mm_u8i8_layer_timed` (`test/htp/nntr_hvx.idl`) | per-call DSP-internal stage totals returned to the host |
| `hvx_worker_pool` | `htp_backend/hvx/hvx_worker_pool.{h,c}` | 6 HVX contexts, fork-join, unit index `i` per worker |
| DMA ring | `hexkl_dma_ring.{h,c}` | `push2d` / `drain` are the natural DMA-lane events |
| reports | `tools/htp_fc_report.py`, `tools/htp_attn_report.py` | parse `FC_FIELD`/`FC_STAGE` marker lines into terminal/HTML bar charts |

The probes are the right measurements in the wrong container: totals in
globals instead of timestamped events in a ring. The plan below keeps every
probe point and changes only what it writes.

## 2. Requirements

1. One `trace.json` per run, Chrome Trace Event Format, so it opens in
   `ui.perfetto.dev`, `chrome://tracing`, and `tools/nntr_trace/viewer.html`.
2. Tracks: host `main`, ThreadManager workers, loadPool / UnloadPool, FastRPC
   seam; DSP `main (caller)`, `HMX`, `HVX-0..5`, `DMA`; counters `VTCM`,
   `DDR bytes`, host `RSS`. QNN layers get a virtual `QNN HTP` process.
3. Every span carries `layer`, `op`, shape (`M,K,N` or `n_query,kv`), and for
   compute spans `engine` and `elems`, so cycles/element is computable.
4. Flow arrows from each host FastRPC call to the DSP entry it triggered.
5. Zero cost when off (macros compile to nothing, `NNTR_TRACE` unset); bounded
   cost when on: fixed-size rings, no allocation on the hot path, dropped
   events counted and reported, no per-tile events unless `level=tile`.
6. The trace says which engine actually ran an op. A fallback to CPU is a
   visible span on the host with `args.fallback="htp_disabled"`, never a
   silent flat line.
7. Works for CPU-only, QNN-layer and HTP-layer models with the same file.

## 3. Architecture

```
 host process                                    cDSP (V79)
 +----------------------------------------+      +-----------------------------------+
 | LayerNode::forwarding  --> layer span  |      | nntr_hvx_* entry  --> entry span  |
 | incremental_inference  --> token span  |      |   layer_run       --> call span   |
 | ThreadManager workers  --> cpu spans   |      |   HMX bursts      --> HMX lane    |
 | HtpComputeOps::*       --> wait span --+-flow-+-> hvx_worker_pool --> HVX-i lanes |
 | FastRPC marshal/return --> seam spans  |      |   dma_ring push   --> DMA lane    |
 | CacheLoader tasks      --> async load  |      |   VTCM arena      --> counter     |
 | QNNGraph::execute      --> QNN span    |      +-------- per-thread rings ---------+
 +--------- per-thread rings ------------+                  | trace_flush (1 RPC / token)
                  |                                          v
                  +---------> trace_sink: merge, clock-map, write trace.json <-------+
```

### 3.1 Host tracer (`nntrainer/utils/trace_sink.{h,cpp}`)

- `NNTR_TRACE_SCOPE(cat, name, args...)`: RAII guard writing
  `{t_start_ns, t_end_ns, tid, cat, name_id, args[4]}` into a `thread_local`
  ring (e.g. 64k entries). `CLOCK_MONOTONIC` on Linux/Android, `QPC` on
  Windows. Names are interned once at registration time (same idea as
  `registerTimeItem`), so the hot path writes integers only.
- `NNTR_TRACE_COUNTER(name, value)` and `NNTR_TRACE_FLOW(id)` for counters and
  arrows.
- A `thread_local` current-layer token, pushed by `LayerNode::forwarding` and
  copied into `ThreadManager::parallel_for` workers and `ParallelBatch`
  threads, so a GEMM slice on `tm-worker-2` still knows it belongs to
  `layer12/q_proj`. This is the one piece missing from the runtime today.
- Hook points (all existing functions, all currently untimed or aggregate):
  `NeuralNetwork::incremental_inference` (token + phase), `LayerNode::forwarding`
  / `incremental_forwarding` (layer; `args.engine` from
  `getComputeEngineType()`), `ThreadManager::parallel_for`, `CacheLoader`
  load/unload (`ph:"b"/"e"` async slices keyed by the TaskExecutor task id),
  `SwapDevice` read/write, `QNNGraph` execute, every `HtpComputeOps` override.
- `Profiler` stays as is; the new listener can also subscribe to it so the
  legacy `PROFILE_TIME_*` sites appear in the trace. Fixing `ProfileEventData`
  to carry a timestamp is a small follow-up, not a prerequisite.

### 3.2 DSP tracer (`htp_backend/hmx/hexkl_trace.{h,c}`, replaces `hexkl_probe`)

- Event: `{u64 t0_ticks, u32 dur_ticks, u16 lane, u16 kind, u32 a0, a1, a2, a3}`
  = 24 bytes. `t` is `HAP_perf_get_qtimer_count()` (19.2 MHz, shared with the
  host-visible `CNTVCT` on Snapdragon, which is what makes clock sync cheap).
  `a0..a3` carry chunk / tile counts / elems, and for kernels the PMU cycle
  delta (`HAP_perf_get_pcycles()`), so cycles/element and cycles/packet do not
  depend on an assumed clock.
- Lanes: `MAIN` (caller thread), `HMX`, `HVX_0..5`, `DMA`. The caller thread
  emits `HMX` events around micro-mm bursts + `acc_read` and `DMA` events
  around `push2d` / `drain`; each pool worker writes its own `HVX_i` ring
  (index = `i` from `hvx_worker_pool_func`), so there are no locks and no
  false sharing. One ring per lane, fixed at open.
- Levels: `off` (today's production path), `stage` (quant, hmx-chunk, dequant,
  softmax band, DMA push/drain, pool fork/join: a few hundred events per
  call), `tile` (per `acc_read` / per micro-mm: the current probe sites, for
  debugging only). `stage` replaces `*_timed`; `tile` replaces `hexkl_probe`.
- IDL additions (`nntr_hvx.idl`):
  `trace_control(in uint32 level)`, `trace_ping(rout uint64 qtimer)`,
  `trace_flush(rout sequence<uint8> events, rout uint32 dropped)`. Flushing
  once per token costs one FastRPC round trip against ~140 the token already
  pays. The `*_timed` entries and their `stage_us` contract are deleted once
  `stage` level lands, with `htp_fc_report.py` moved onto the trace.

### 3.3 Clock mapping

- At `nntr_hvx_open`: 8x `trace_ping`, keep the sample with the smallest host
  RTT, `offset = qtimer_us - (t_send + t_recv) / 2`. Expected error is a few
  µs against RTTs of ~300 µs, and the host `marshal -> DSP entry -> return`
  triple on every call is a continuous re-check: a DSP entry that maps before
  its own marshal ended is drift and is reported in `metadata.clock_sync`.
- Because the qtimer is a constant-rate counter, no per-run frequency fit is
  needed; only the offset.

### 3.4 QNN layers

- Turn on `QNN_PROFILE_LEVEL_DETAILED` behind a layer property, call
  `extractBackendProfilingInfo()` after `graphExecute` (today it runs only
  after `contextCreateFromBinary`) and emit its sub-events as spans under a
  virtual `QNN HTP` process, parented to the host `QNNGraph::execute` span.
- When the vendor optrace is available (`QNN_HTP_PROFILE=optrace`), a
  converter (`tools/nntr_trace/qnn_optrace_to_nntr.py`) rebases it onto our
  timeline so a hybrid model (QNN layers + HexKL layers + CPU layers) is one
  file. Its tid convention is already ours.

### 3.5 Writer

- `trace_sink::write(path)` merges all rings, maps DSP ticks to host µs, and
  writes Chrome JSON: `X` spans, `M` names, `s`/`f` flows, `C` counters, plus
  `metadata` (`model`, `device`, `dsp_clock_mhz`, `hvx_threads`, `level`,
  `clock_sync`, `dropped`, `htp_enabled`). Opt-in via
  `NNTR_TRACE=/data/local/tmp/trace.json[,level=stage]`, or the C API
  `ml_train_model_set_trace()` for the Android app.

## 4. Event schema

| field | value |
|---|---|
| `pid` | 1 host, 2 HTP, 3 QNN HTP |
| `tid` host | 1 main, 11.. ThreadManager workers, 20 FastRPC seam, 30.. loadPool / UnloadPool |
| `tid` DSP | 1 caller, 768 HMX, 512..517 HVX, 256 DMA (QNN optrace convention) |
| `cat` | `host.cpu` `host.wait` `host.rpc` `host.layer` `host.phase` `host.load` `dsp.entry` `dsp.call` `dsp.hmx` `dsp.hvx` `dsp.dma` `dsp.sync` `qnn.op` |
| `args` | `layer`, `op`, `M K N` / `n_query kv head block chunk`, `engine` (`CPU HMX HVX`), `elems`, `cycles`, `bytes`, `units`, `fallback` |
| counters | `VTCM (KB)`, `DDR bytes`, `RSS (MB)` |

## 5. Metrics the viewer derives (and CI can gate on)

| metric | definition | QNN reference (ref_16) |
|---|---|---|
| engine busy % | union of a track's spans / range wall | HMX 9.5 %, HVX 31-78 %, DMA 13 % |
| parallel compression | Σ(HMX + HVX + DMA busy) / DSP wall | 4.1x (ours today: 1.0x) |
| DSP idle | DSP entry time not covered by any lane | 0.09 % |
| transport share | host seam spans / wall | 128 + 39 µs of a 513 µs FC NetRun |
| cycles / element | Σ cycles / Σ elems per kernel name | HMX 0.01-0.06, HVX 0.07-0.10, softmax 0.24, outlier 1.52 |
| cycles / packet | PMU cycles / packets per kernel | 2.0-2.4 healthy, > 6 stalls |
| pool tail | max unit dur / mean unit dur per `pool_run` | `tiles >= threads x 4` rule |
| per-token | token span, TTFT, calls per token, bytes per token | - |

## 6. Phases

| phase | deliverable | gate |
|---|---|---|
| **P0** (this change) | schema, `make_sample_trace.py`, `viewer.html`, this doc; `stage_us_to_trace.py` to lift existing `FC_STAGE` / `attn_forward_timed` logs into the schema | viewer shows the sample; a real `FC_STAGE` log renders as sequential stage spans |
| **P1** host tracer | `trace_sink`, `NNTR_TRACE_SCOPE`, layer/token/worker/loader hooks, writer, `NNTR_TRACE` env | a CPU-only Qwen3-0.6B decode produces a trace with per-token spans; overhead < 1 % at `stage` level |
| **P2** HTP seam | `HtpComputeOps` wait spans with `fallback` args, `trace_ping` clock sync, `trace_flush`, DSP `stage` ring replacing `*_timed` | flows line up: DSP entry inside its host wait span on every call; `htp_fc_report.py` reads the trace instead of markers |
| **P3** DSP lanes | per-worker `HVX_i` rings, HMX and DMA lanes, PMU cycles, `tile` level replacing `hexkl_probe` | parallel compression and cy/elem computed from real events; the doc-35 lane totals reproduce |
| **P4** QNN | detailed `QnnProfile` after execute, optrace converter, `pid 3` | a hybrid model renders CPU, HexKL and QNN spans on one timeline |
| **P5** CI | `nntr_trace summarize trace.json -> metrics.json`, hooked into `benchmark_android.py`; thresholds on compression, idle, transport share, top cy/elem | a regression in any metric fails the device benchmark |

P0 and P1 need no device. P2 onward run on the HTP branches (the kernels do
not exist on `main`) and should land there first.

## 7. Risks and how the design absorbs them

- **Probe overhead in the tile loop.** `hexkl_probe` already showed four
  qtimer reads per tile are visible; `tile` level is debug-only and `stage`
  level emits per chunk (16 tiles), which is ~1 % of a 15 µs chunk.
- **Ring overflow.** Fixed rings drop the newest event and count drops; the
  count is in `metadata` and the viewer shows it. Prefill at seq 1024 with
  `stage` level is ~2k events per layer, well inside a 64k ring.
- **Clock drift.** Constant-rate counter plus per-call re-check; a bad
  mapping is reported, not hidden.
- **Flush payload.** 24 bytes x 60k events = 1.4 MB per token at `tile`
  level; negligible at `stage` level (~100 KB). One extra RPC per token.
- **HMX lane attribution.** HMX events are written by whichever thread issues
  the micro-mm (today the caller). If a dedicated HMX thread arrives
  (doc 35 §4a), the lane id stays `HMX` and only the writer thread changes.
- **sdkl macro-API on `main`.** Nothing below `sdkl_*` is visible; there the
  trace stops at the host wait span. The micro-API session on the HTP
  branches is the one that gets lanes.
- **Overlapping probes.** Once stages overlap (P3), `quant + dequant + ...`
  no longer sums to `dsp_total`, which is exactly the case doc 35 §6 warns
  about. Per-lane unions, not sums, are the reported totals.

## 8. Open questions

1. Is `HAP_perf_get_qtimer_count()` the same counter as the host's
   `CNTVCT_EL0` on the target (it is on documented Snapdragon parts)? If so
   the offset is a constant and `trace_ping` becomes a check, not a
   calibration.
2. Does `HAP_perf_get_pcycles()` count per hardware thread or per PD in an
   unsigned PD? Per-thread is what cycles/packet needs.
3. Where should the Android app write the file: `getFilesDir()` through the
   C API, or `/data/local/tmp` for `adb pull`? Both are one string.
