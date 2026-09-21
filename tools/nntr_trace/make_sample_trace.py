#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# @package make_sample_trace
# @brief Emit a synthetic nntrainer trace.json (Chrome Trace Event Format).
"""Emit a synthetic nntrainer trace.json in Chrome Trace Event Format.

This is the *shape* of the file the real tracer (docs/backend_guide/
HTP_TRACE_PROFILER.md) will produce: one process for the host CPU, one for the
Hexagon cDSP, one track per engine (main thread, ThreadManager workers, FastRPC
seam, DSP caller thread, HMX, HVX worker pool, DMA), flow arrows from every
host FastRPC call to the DSP work it triggered, and counter tracks for VTCM.

Every duration here is scaled from numbers measured on device and written
down in the HTP branch docs (34_fc_measured.md, 35_hmx_hvx_overlap.md,
31_dataflow_as_built.md). They are a stand-in for the real instrumentation,
not a measurement: the file exists so the viewer, the converters and the
metric code can be built and reviewed before the DSP-side ring buffer lands.

Usage
    tools/nntr_trace/make_sample_trace.py                # trace_asbuilt.json
    tools/nntr_trace/make_sample_trace.py --pipelined    # T2 projection
    tools/nntr_trace/make_sample_trace.py --with-warnings
    tools/nntr_trace/make_sample_trace.py -o out.json --layers 4 --tokens 12

The output opens directly in tools/nntr_trace/viewer.html and in
https://ui.perfetto.dev (drag and drop) or chrome://tracing.
"""

import argparse
import json
import random

# --- process / thread ids ---------------------------------------------------
# tids on the DSP side mirror the QNN optrace convention (ref_16 §0) so the
# two traces line up side by side: 256 = DMA, 512.. = HVX threads, 768 = HMX.
PID_HOST = 1
PID_DSP = 2
TID_MAIN = 1
TID_OMP = [11, 12, 13]  # ThreadManager workers (NNTR_NUM_THREADS=4 -> 3 workers)
TID_RPC = 20  # host-side FastRPC seam (marshal, call, unmarshal)
TID_FSU = 30  # weight loader / swap thread
TID_DSP_MAIN = 1  # DSP caller thread: drives HMX, joins the pool
TID_DMA = 256
TID_HVX = [512, 513, 514, 515, 516, 517]  # 6 HVX contexts on V79
TID_HMX = 768

DSP_CLOCK_MHZ = 1200.0  # cycles = us * MHz; the real tracer reads PMU cycles

# Measured references (us). Prefill M=1024, Qwen3-0.6B (hidden 1024, 16 q
# heads, 8 kv heads, head_dim 128).
FASTRPC_M1024_US = 963  # transport per call at M=1024 (34 §2)
FASTRPC_M1_US = 326  # transport per call at M=1 (34 §2)
FC_M1_KERNEL_US = 113  # decode FC kernel (35 §8)
HMX_CHUNK_US = 15.6  # 16-tile chunk: mm + acc_read (35 §5)
DEQ_CHUNK_US = 8.7  # 16-tile chunk dequant (35 §5)

# Transport model (34 §2): a fixed cost plus a per-byte term. The two measured
# points (326 us at M=1, 963 us at M=1024 for a 12 MB payload) give the slope.
TRANSPORT_FIXED_US = 300.0
TRANSPORT_US_PER_MB = 55.0


class Trace:
  def __init__(self, seed=7):
    self.ev = []
    self.flow_id = 0
    self.pool_id = 0
    self.rng = random.Random(seed)

  def meta(self, pid, tid, name, sort=None):
    if tid is None:
      self.ev.append({"ph": "M", "pid": pid, "name": "process_name",
                      "args": {"name": name}})
      if sort is not None:
        self.ev.append({"ph": "M", "pid": pid, "name": "process_sort_index",
                        "args": {"sort_index": sort}})
    else:
      self.ev.append({"ph": "M", "pid": pid, "tid": tid, "name": "thread_name",
                      "args": {"name": name}})
      if sort is not None:
        self.ev.append({"ph": "M", "pid": pid, "tid": tid,
                        "name": "thread_sort_index",
                        "args": {"sort_index": sort}})

  def x(self, pid, tid, name, cat, ts, dur, **args):
    # Round first and return the rounded end, so back-to-back slices never
    # overlap by a rounding residue (which would count as HMX/HVX overlap).
    ts, dur = round(ts, 3), round(dur, 3)
    e = {"ph": "X", "pid": pid, "tid": tid, "name": name, "cat": cat,
         "ts": ts, "dur": dur}
    if args:
      e["args"] = {k: v for k, v in args.items() if v is not None}
    self.ev.append(e)
    return round(ts + dur, 3)

  def counter(self, pid, name, ts, **vals):
    self.ev.append({"ph": "C", "pid": pid, "name": name, "ts": round(ts, 3),
                    "args": vals})

  def flow(self, src, dst):
    """src/dst: (pid, tid, ts). Arrow from a host call into DSP work."""
    self.flow_id += 1
    fid = self.flow_id
    self.ev.append({"ph": "s", "pid": src[0], "tid": src[1], "ts": round(src[2], 3),
                    "id": fid, "name": "fastrpc", "cat": "flow"})
    self.ev.append({"ph": "f", "pid": dst[0], "tid": dst[1], "ts": round(dst[2], 3),
                    "id": fid, "name": "fastrpc", "cat": "flow", "bp": "e"})

  def transport(self, nbytes):
    """Fixed + per-byte FastRPC cost with a deterministic +-8 % jitter."""
    base = TRANSPORT_FIXED_US + TRANSPORT_US_PER_MB * nbytes / 1e6
    return base * (1.0 + self.rng.uniform(-0.08, 0.08))


def hvx_split(tr, name, cat, ts, total_us, n_units, layer, elems, cls,
              **extra):
  """Run `total_us` of HVX work across the pool (caller + workers), fork-join.

  Returns the wall end. Models hvx_worker_pool_run: unit 0 on the caller
  thread, the rest on parked workers, with a small fork/join tax. Every unit
  slice and the pool_run frame share one `pool` id so the viewer can pair
  them (VIEWER_PLAN.md W5).
  """
  tr.pool_id += 1
  pool = tr.pool_id
  n = min(n_units, len(TID_HVX))
  per = total_us / n
  fork = 3.0
  end = ts + fork
  for i in range(n):
    jitter = 1.0 + 0.04 * ((i * 7) % 5) / 5.0  # uneven tails, like real
    if n_units < len(TID_HVX) * 4 and i == n - 1:
      jitter += 0.35  # too few units to absorb the tail (doc 36 T6)
    d = per * jitter
    tr.x(PID_DSP, TID_HVX[i], name, cat, ts + fork, d, layer=layer,
         engine="HVX", unit=i, units=n, elems=int(elems / n), pool=pool,
         **{"class": cls}, **extra)
    end = max(end, ts + fork + d)
  tr.x(PID_DSP, TID_DSP_MAIN, "pool_run " + name, "dsp.sync", ts, end - ts + 2,
       layer=layer, units=n, n_units=n_units, pool=pool)
  return end + 2


def dsp_fc_call(tr, ts, layer, opname, M, K, N, weight_us, pipelined,
                quant_us, dequant_us, n_handles=1):
  """One mm_u8iX_layer call on the DSP: quant -> [DMA | HMX | dequant] chunks."""
  t0 = ts
  tr.counter(PID_DSP, "VTCM (KB)", ts, used=int(256 + M * K / 1024))
  ts = hvx_split(tr, "quant f32->u8 AH", "dsp.hvx", ts, quant_us, K // 32,
                 layer, M * K, "quant", op=opname, stage="quant")
  n_tiles = (M // 64) * (N // 32)
  n_chunks = max(1, n_tiles // 16)
  hmx_us = weight_us / n_chunks
  deq_us = dequant_us / n_chunks
  dma_us = min(hmx_us * 0.6, 12.0)
  hmx_end = ts
  deq_end = ts
  dma_end = ts
  for c in range(n_chunks):
    # DMA the next weight tiles into the VTCM double buffer.
    dma_start = max(dma_end, hmx_end - hmx_us) if c else ts
    dma_end = tr.x(PID_DSP, TID_DMA, "wbuf push2d", "dsp.dma", dma_start, dma_us,
                   layer=layer, op=opname, chunk=c, bytes=16 * 2048,
                   **{"class": "dma"})
    hmx_start = max(hmx_end, dma_end)
    if not pipelined:
      hmx_start = max(hmx_start, deq_end)
    hmx_end = tr.x(PID_DSP, TID_HMX, "micro-mm x32 + acc_read", "dsp.hmx",
                   hmx_start, hmx_us, layer=layer, op=opname, chunk=c,
                   engine="HMX", elems=16 * 64 * 32 * (K // 32) * 2,
                   tiles=16, M=M, K=K, N=N, n_handles=n_handles,
                   **{"class": "matmul"})
    if pipelined:
      # HVX pool dequants chunk c while HMX runs chunk c+1.
      deq_start = max(hmx_end, deq_end)
      deq_end = tr.x(PID_DSP, TID_HVX[1 + (c % 5)], "dequant i32->f32",
                     "dsp.hvx", deq_start, deq_us, layer=layer, op=opname,
                     chunk=c, engine="HVX", elems=16 * 64 * 32, stage="dequant",
                     **{"class": "dequant"})
    else:
      deq_end = tr.x(PID_DSP, TID_HVX[0], "dequant i32->f32", "dsp.hvx",
                     hmx_end, deq_us, layer=layer, op=opname, chunk=c,
                     engine="HVX", elems=16 * 64 * 32, stage="dequant",
                     **{"class": "dequant"})
  end = max(hmx_end, deq_end)
  tr.x(PID_DSP, TID_DSP_MAIN, "dma_ring_drain", "dsp.sync", end, 4, layer=layer)
  end += 4
  tr.x(PID_DSP, TID_DSP_MAIN, "layer_run " + opname, "dsp.call", t0, end - t0,
       layer=layer, op=opname, M=M, K=K, N=N, n_handles=n_handles,
       pipelined=pipelined)
  tr.counter(PID_DSP, "VTCM (KB)", end, used=256)
  return end


def fastrpc(tr, ts, layer, name, in_bytes, out_bytes, body, shape=None):
  """Host FastRPC seam around DSP work `body(ts) -> end`."""
  transport_us = tr.transport(in_bytes + out_bytes)
  half = transport_us / 2.0
  t_marshal = tr.x(PID_HOST, TID_RPC, "marshal " + name, "host.rpc", ts, half,
                   layer=layer, bytes=in_bytes)
  tr.flow((PID_HOST, TID_RPC, ts + half * 0.9), (PID_DSP, TID_DSP_MAIN, t_marshal))
  dsp_end = body(t_marshal)
  tr.x(PID_DSP, TID_DSP_MAIN, "nntr_hvx_" + name, "dsp.entry", t_marshal,
       dsp_end - t_marshal, layer=layer)
  t_ret = tr.x(PID_HOST, TID_RPC, "return " + name, "host.rpc", dsp_end, half,
               layer=layer, bytes=out_bytes)
  tr.x(PID_HOST, TID_MAIN, "wait " + name, "host.wait", ts, t_ret - ts,
       layer=layer, engine="HTP", bytes_in=in_bytes, bytes_out=out_bytes,
       **(shape or {}))
  return t_ret


def cpu_omp(tr, ts, layer, name, total_us, elems, **extra):
  """CPU op split across main + ThreadManager workers (parallel_for)."""
  per = total_us / (1 + len(TID_OMP))
  end = ts
  for i, tid in enumerate([TID_MAIN] + TID_OMP):
    d = per * (1.0 + 0.05 * (i % 2))
    tr.x(PID_HOST, tid, name, "host.cpu", ts, d, layer=layer, engine="CPU",
         elems=elems // 4, **extra)
    end = max(end, ts + d)
  return end + 1.5


def prefill_layer(tr, ts, li, seq, pipelined, fallback):
  layer = "layer%d" % li
  t_layer = ts
  ts = cpu_omp(tr, ts, layer, "rmsnorm (input)", 180, seq * 1024)
  # q/k/v projection: one x3 call sharing the quantized activation.
  ts = fastrpc(tr, ts, layer, "mm_u8i8_layer[q,k,v]", seq * 1024 * 4,
               seq * 4096 * 4,
               lambda t: dsp_fc_call(tr, t, layer, "qkv_proj", seq, 1024, 4096,
                                     1973, pipelined, 720, 1037, 3),
               dict(M=seq, K=1024, N=4096, n_handles=3))
  ts = cpu_omp(tr, ts, layer, "rope (q,k)", 320, seq * 4096)
  ts = tr.x(PID_HOST, TID_MAIN, "kv_append (fp16 pack)", "host.cpu", ts, 210,
            layer=layer, engine="CPU")

  # SDPA: fused attention, one DSP call. HVX 5206 / HMX 1614 measured.
  def sdpa(t):
    t0 = t
    for h in range(2):  # two of 16 heads drawn out, the rest folded below
      t = hvx_split(tr, "q gather+quant", "dsp.hvx", t, 40, 8, layer,
                    seq * 128, "quant", op="sdpa", head=h)
      qk_end = t
      for j in range(4):  # 4 kv blocks of 256
        d = tr.x(PID_DSP, TID_DMA, "Kt_wh block", "dsp.dma", qk_end, 6,
                 layer=layer, op="sdpa", head=h, block=j, **{"class": "dma"})
        qk_end = tr.x(PID_DSP, TID_HMX, "Q.Kt micro-mm x32", "dsp.hmx", d, 12,
                      layer=layer, op="sdpa", head=h, block=j, engine="HMX",
                      elems=64 * 256 * 128, M=64, K=128, N=256,
                      **{"class": "matmul"})
        qk_end = tr.x(PID_DSP, TID_HVX[0], "acc_copy + dequant", "dsp.hvx",
                      qk_end, 55, layer=layer, op="sdpa", head=h, block=j,
                      engine="HVX", elems=64 * 256, stage="acc_copy",
                      **{"class": "elementwise"})
      t = hvx_split(tr, "softmax (band)", "dsp.hvx", qk_end, 150, 8, layer,
                    seq * 256, "softmax", op="sdpa", head=h, stage="softmax")
      for j in range(4):
        t = hvx_split(tr, "P quant", "dsp.hvx", t, 20, 4, layer, 64 * 256,
                      "quant", op="sdpa", head=h, block=j, stage="quant")
        d = tr.x(PID_DSP, TID_DMA, "V_wh block (blocking)", "dsp.dma", t, 6,
                 layer=layer, op="sdpa", head=h, block=j, **{"class": "dma"})
        t = tr.x(PID_DSP, TID_HMX, "P.V micro-mm x32", "dsp.hmx", d, 12,
                 layer=layer, op="sdpa", head=h, block=j, engine="HMX",
                 elems=64 * 256 * 128, M=64, K=256, N=128,
                 **{"class": "matmul"})
        t = tr.x(PID_DSP, TID_HVX[0], "acc_copy + dequant", "dsp.hvx", t, 55,
                 layer=layer, op="sdpa", head=h, block=j, engine="HVX",
                 elems=64 * 128, stage="acc_copy", **{"class": "elementwise"})
    # remaining heads/bands folded into one span so the file stays small
    rest = 6820 - (t - t0)
    tr.x(PID_DSP, TID_HMX, "Q.Kt / P.V (14 heads folded)", "dsp.hmx", t,
         rest * 0.24, layer=layer, op="sdpa", engine="HMX",
         elems=14 * seq * seq * 128 * 2, **{"class": "matmul"})
    t = hvx_split(tr, "softmax/quant/dequant (14 heads folded)", "dsp.hvx",
                  t + rest * 0.24, rest * 0.76 * 0.9, 6, layer, 14 * seq * seq,
                  "softmax", op="sdpa")
    tr.x(PID_DSP, TID_DSP_MAIN, "attn_forward", "dsp.call", t0, t - t0,
         layer=layer, op="sdpa", n_query=seq, kv=seq)
    return t
  ts = fastrpc(tr, ts, layer, "attn_forward", seq * 2048 * 4, seq * 2048 * 4,
               sdpa, dict(n_query=seq, kv=seq))
  ts = fastrpc(tr, ts, layer, "mm_u8i8_layer[o]", seq * 2048 * 4, seq * 1024 * 4,
               lambda t: dsp_fc_call(tr, t, layer, "o_proj", seq, 2048, 1024,
                                     809, pipelined, 640, 905),
               dict(M=seq, K=2048, N=1024))
  ts = cpu_omp(tr, ts, layer, "residual add", 60, seq * 1024)
  ts = cpu_omp(tr, ts, layer, "rmsnorm (post-attn)", 180, seq * 1024)
  if fallback:
    # HtpContext degraded this op to CPU: the tracer marks it, never hides it.
    ts = cpu_omp(tr, ts, layer, "gate_up_proj (CPU fallback)", 9800,
                 seq * 6144, fallback="htp_disabled", M=seq, K=1024, N=6144)
  else:
    ts = fastrpc(tr, ts, layer, "mm_u8i8_layer[gate,up]", seq * 1024 * 4,
                 seq * 6144 * 4,
                 lambda t: dsp_fc_call(tr, t, layer, "gate_up_proj", seq, 1024,
                                       6144, 2960, pipelined, 720, 1555, 2),
                 dict(M=seq, K=1024, N=6144, n_handles=2))
  ts = cpu_omp(tr, ts, layer, "silu * up", 410, seq * 3072)
  ts = fastrpc(tr, ts, layer, "mm_u8i8_layer[down]", seq * 3072 * 4,
               seq * 1024 * 4,
               lambda t: dsp_fc_call(tr, t, layer, "down_proj", seq, 3072, 1024,
                                     1480, pipelined, 1900, 905),
               dict(M=seq, K=3072, N=1024))
  ts = cpu_omp(tr, ts, layer, "residual add", 60, seq * 1024)
  tr.x(PID_HOST, TID_MAIN, layer, "host.layer", t_layer, ts - t_layer,
       layer=layer, phase="prefill", seq=seq)
  return ts


def decode_layer(tr, ts, li, kv, fallback):
  layer = "layer%d" % li
  t_layer = ts
  grow = kv / 1024.0  # attention cost scales with the kv length

  def fc(t, op, K, N, n_handles=1):
    t0 = t
    t = hvx_split(tr, "quant f32->u8 AH", "dsp.hvx", t, 8, 2, layer, K, "quant",
                  op=op)
    d = tr.x(PID_DSP, TID_DMA, "weight tiles", "dsp.dma", t, 30, layer=layer,
             op=op, bytes=K * N, **{"class": "dma"})
    t = tr.x(PID_DSP, TID_HMX, "micro-mm (M=2, 62 rows pad)", "dsp.hmx", d,
             FC_M1_KERNEL_US * 0.55, layer=layer, op=op, engine="HMX",
             elems=K * N * 2, M=1, K=K, N=N, n_handles=n_handles,
             **{"class": "matmul"})
    t = tr.x(PID_DSP, TID_HVX[0], "dequant (64-row pad)", "dsp.hvx", t,
             FC_M1_KERNEL_US * 0.3, layer=layer, op=op, engine="HVX",
             elems=64 * N, **{"class": "dequant"})
    tr.x(PID_DSP, TID_DSP_MAIN, "layer_run " + op, "dsp.call", t0, t - t0,
         layer=layer, op=op, M=1, K=K, N=N, n_handles=n_handles)
    return t

  ts = tr.x(PID_HOST, TID_MAIN, "rmsnorm", "host.cpu", ts, 6, layer=layer,
            engine="CPU")
  ts = fastrpc(tr, ts, layer, "mm_u8i8_layer[q,k,v]", 4096, 16384,
               lambda t: fc(t, "qkv_proj", 1024, 4096, 3),
               dict(M=1, K=1024, N=4096, n_handles=3))
  ts = tr.x(PID_HOST, TID_MAIN, "rope + kv_append", "host.cpu", ts, 14,
            layer=layer, engine="CPU")

  def attn(t):
    t0 = t
    t = hvx_split(tr, "q quant", "dsp.hvx", t, 6, 2, layer, 2048, "quant",
                  op="sdpa")
    d = tr.x(PID_DSP, TID_DMA, "Kt_wh blocks", "dsp.dma", t, 25 * grow,
             layer=layer, op="sdpa", **{"class": "dma"})
    t = tr.x(PID_DSP, TID_HMX, "Q.Kt (M=2 pad 64)", "dsp.hmx", d, 30 * grow,
             layer=layer, op="sdpa", engine="HMX", elems=16 * kv * 128,
             M=1, K=128, N=kv, **{"class": "matmul"})
    t = tr.x(PID_DSP, TID_HVX[0], "acc_copy + dequant", "dsp.hvx", t, 50 * grow,
             layer=layer, op="sdpa", engine="HVX", elems=16 * kv,
             **{"class": "elementwise"})
    t = tr.x(PID_DSP, TID_HVX[0], "softmax", "dsp.hvx", t, 22 * grow,
             layer=layer, op="sdpa", engine="HVX", elems=16 * kv,
             **{"class": "softmax"})
    d = tr.x(PID_DSP, TID_DMA, "V_wh blocks", "dsp.dma", t, 25 * grow,
             layer=layer, op="sdpa", **{"class": "dma"})
    t = tr.x(PID_DSP, TID_HMX, "P.V (M=2 pad 64)", "dsp.hmx", d, 30 * grow,
             layer=layer, op="sdpa", engine="HMX", elems=16 * kv * 128,
             M=1, K=kv, N=128, **{"class": "matmul"})
    t = tr.x(PID_DSP, TID_HVX[0], "acc_copy + dequant", "dsp.hvx", t, 50,
             layer=layer, op="sdpa", engine="HVX", elems=16 * 128,
             **{"class": "elementwise"})
    tr.x(PID_DSP, TID_DSP_MAIN, "attn_forward", "dsp.call", t0, t - t0,
         layer=layer, op="sdpa", n_query=1, kv=kv)
    return t
  ts = fastrpc(tr, ts, layer, "attn_forward", 8192, 8192, attn,
               dict(n_query=1, kv=kv))
  ts = fastrpc(tr, ts, layer, "mm_u8i8_layer[o]", 8192, 4096,
               lambda t: fc(t, "o_proj", 2048, 1024), dict(M=1, K=2048, N=1024))
  ts = tr.x(PID_HOST, TID_MAIN, "residual + rmsnorm", "host.cpu", ts, 9,
            layer=layer, engine="CPU")
  if fallback:
    ts = tr.x(PID_HOST, TID_MAIN, "gate_up_proj (CPU fallback)", "host.cpu",
              ts, 1900, layer=layer, engine="CPU", fallback="htp_disabled",
              M=1, K=1024, N=6144)
  else:
    ts = fastrpc(tr, ts, layer, "mm_u8i8_layer[gate,up]", 4096, 24576,
                 lambda t: fc(t, "gate_up_proj", 1024, 6144, 2),
                 dict(M=1, K=1024, N=6144, n_handles=2))
  ts = tr.x(PID_HOST, TID_MAIN, "silu * up", "host.cpu", ts, 5, layer=layer,
            engine="CPU")
  ts = fastrpc(tr, ts, layer, "mm_u8i8_layer[down]", 12288, 4096,
               lambda t: fc(t, "down_proj", 3072, 1024), dict(M=1, K=3072, N=1024))
  ts = tr.x(PID_HOST, TID_MAIN, "residual", "host.cpu", ts, 3, layer=layer,
            engine="CPU")
  tr.x(PID_HOST, TID_MAIN, layer, "host.layer", t_layer, ts - t_layer,
       layer=layer, phase="decode")
  return ts


def build(n_layers, seq, pipelined, n_tokens, warnings):
  tr = Trace()
  tr.meta(PID_HOST, None, "host (CPU, Cortex-X)", 0)
  tr.meta(PID_DSP, None, "HTP (cDSP V79)", 1)
  tr.meta(PID_HOST, TID_MAIN, "main", 0)
  for i, t in enumerate(TID_OMP):
    tr.meta(PID_HOST, t, "tm-worker-%d" % (i + 1), 1 + i)
  tr.meta(PID_HOST, TID_RPC, "fastrpc seam", 5)
  tr.meta(PID_HOST, TID_FSU, "weight loader", 6)
  tr.meta(PID_DSP, TID_DSP_MAIN, "dsp main (caller)", 0)
  tr.meta(PID_DSP, TID_HMX, "HMX", 1)
  for i, t in enumerate(TID_HVX):
    tr.meta(PID_DSP, t, "HVX-%d" % i, 2 + i)
  tr.meta(PID_DSP, TID_DMA, "DMA", 10)

  fallback_layer = 1 if warnings and n_layers > 1 else -1
  ts = 0.0
  # weight load / WH bake happens before the first forward
  ts = tr.x(PID_HOST, TID_FSU, "load + WH bake (%d layers)" % n_layers,
            "host.load", ts, 12200.0 * 0.3 * n_layers, engine="CPU")
  t_pre = ts
  for li in range(n_layers):
    ts = prefill_layer(tr, ts, li, seq, pipelined, li == fallback_layer)
  ts = cpu_omp(tr, ts, "lm_head", "lm_head Q4_0 GEMV (tied)", 1400, 151936 * 1024)
  tr.x(PID_HOST, TID_MAIN, "prefill (seq=%d)" % seq, "host.phase", t_pre,
       ts - t_pre, phase="prefill", seq=seq, layers=n_layers)
  ts += 200
  for tok in range(n_tokens):
    t_tok = ts
    kv = seq + tok + 1
    for li in range(n_layers):
      ts = decode_layer(tr, ts, li, kv * (1.0 + 0.02 * tok), li == fallback_layer)
    ts = cpu_omp(tr, ts, "lm_head", "lm_head Q4_0 GEMV (tied)", 1400,
                 151936 * 1024)
    ts = tr.x(PID_HOST, TID_MAIN, "sample + detokenize", "host.cpu", ts, 40,
              engine="CPU")
    tr.x(PID_HOST, TID_MAIN, "decode token %d" % (tok + 1), "host.phase", t_tok,
         ts - t_tok, phase="decode", token=tok + 1, kv=kv)
    ts += 30
  md = {
    "tool": "nntr_trace sample (synthetic, scaled from measured docs)",
    "model": "Qwen3-0.6B (subset: %d layers)" % n_layers,
    "device": "S25U / V79 (numbers scaled from 34/35/31 docs)",
    "level": "stage",
    "dsp_clock_mhz": DSP_CLOCK_MHZ,
    "hvx_threads": len(TID_HVX),
    # Measure on device before filling this in; the viewer shows TOPS only
    # while it is null (VIEWER_PLAN.md W6).
    "hmx_peak_tops": None,
    "budgets": {"VTCM (KB)": 8192},
    "pipelined": pipelined,
    "htp_enabled": True,
    "dropped": {"host": 0, "dsp": 0},
    "clock_sync": {"method": "ntp-style ping at nntr_hvx_open",
                   "offset_us": 0, "rtt_us": 0, "violations": 0},
  }
  if warnings:
    md["dropped"] = {"host": 0, "dsp": 137}
    md["clock_sync"]["violations"] = 2
  return {"displayTimeUnit": "ms", "traceEvents": tr.ev, "metadata": md}


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("-o", "--out", default=None)
  ap.add_argument("--layers", type=int, default=2)
  ap.add_argument("--seq", type=int, default=1024)
  ap.add_argument("--tokens", type=int, default=8,
                  help="decode tokens to generate (attention grows with kv)")
  ap.add_argument("--pipelined", action="store_true",
                  help="project the T2 HMX/HVX chunk pipeline instead of the "
                       "serialized as-built loop")
  ap.add_argument("--with-warnings", action="store_true",
                  help="one layer falls back to CPU, and the metadata reports "
                       "dropped DSP records and clock violations")
  a = ap.parse_args()
  out = a.out or ("trace_warnings.json" if a.with_warnings else
                  "trace_pipelined.json" if a.pipelined else
                  "trace_asbuilt.json")
  doc = build(a.layers, a.seq, a.pipelined, a.tokens, a.with_warnings)
  with open(out, "w") as f:
    json.dump(doc, f, separators=(",", ":"))
  n = len(doc["traceEvents"])
  last = max(e.get("ts", 0) + e.get("dur", 0) for e in doc["traceEvents"])
  print("%s: %d events, %.1f ms" % (out, n, last / 1000.0))


if __name__ == "__main__":
  main()
