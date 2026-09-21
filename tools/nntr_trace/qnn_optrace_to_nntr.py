#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# @package qnn_optrace_to_nntr
# @brief Rebase a QNN HTP optrace (chromeTrace_opTrace.json) onto the nntr_trace schema.
"""Rebase a QNN HTP optrace onto the nntr_trace schema so both open together.

QAIRT's HTP optrace (``*_chromeTrace_opTrace.json``) is already Chrome Trace
Event Format with one thread per hardware unit (ref_16 §0: tid 256 = DMA,
512..519 = HVX threads, 768 = HMX, tid 1 = zero-cost view ops). This script
keeps the events, moves them to pid 3, tags them with the ``cat`` / ``args``
keys the viewer's metrics read (``dsp.hmx`` etc., ``engine``, ``elems``,
``cycles``, ``op``, ``class``), and converts timestamps to microseconds.

ASSUMPTIONS TO VERIFY ON A REAL FILE (VIEWER_PLAN.md W9). This was written
against a hand-made fixture (test/data/qnn_mini.json) that follows ref_16's
description, not against a vendor file. Before trusting the output, open a
real optrace and check, then fix the constants below and record what you
found here:

  * ``ts`` / ``dur`` unit: ref_16 quotes the E2E as 5,666,199 HTP cycles at
    1.2 GHz, so timestamps may be cycles. ``--ts-unit cycles`` divides by
    ``--clock-mhz``; ``--ts-unit us`` (default) leaves them alone.
  * where the per-instance cycle count lives (``args.cycles``? ``args["Cycles
    per Packet"]`` is named in ref_16 §3.3, which implies a packet count too).
    SHAPE_KEYS / CYCLE_KEYS below list the candidates tried in order.
  * the tile shape key (``[1,8,32,256]`` style lists) for ``elems``.
  * whether node names carry the graph node (``node_linear_2``) and the kernel
    (``q::ConvLayer_s1.opt``) in one string or in separate fields.

Usage
    tools/nntr_trace/qnn_optrace_to_nntr.py in.json -o out.json
    tools/nntr_trace/qnn_optrace_to_nntr.py in.json -o out.json --ts-unit cycles --clock-mhz 1200
    tools/nntr_trace/qnn_optrace_to_nntr.py in.json -o out.json --keep-views

Stdlib only.
"""

import argparse
import json
import re

PID_QNN = 3
TID_DMA, TID_HVX0, TID_HVX_LAST, TID_HMX, TID_VIEWS = 256, 512, 519, 768, 1
SHAPE_KEYS = ("shape", "tile", "tile_shape", "dims", "output_shape", "out_shape")
CYCLE_KEYS = ("cycles", "Cycles", "cycle", "unit_cycles", "Unit Cycles")
CLASS_RULES = (
  (re.compile(r"conv|matmul|fullyconnected|gemm", re.I), "matmul"),
  (re.compile(r"softmax", re.I), "softmax"),
  (re.compile(r"transpose", re.I), "transpose"),
  (re.compile(r"dequant|dequantize", re.I), "dequant"),
  (re.compile(r"quant|requant|convert_weights", re.I), "quant"),
  (re.compile(r"dma|weights_to_vtcm|spill|fill", re.I), "dma"),
)


def cat_for(tid):
  if tid == TID_HMX:
    return "dsp.hmx", "HMX"
  if TID_HVX0 <= tid <= TID_HVX_LAST:
    return "dsp.hvx", "HVX"
  if tid == TID_DMA:
    return "dsp.dma", None
  if tid == TID_VIEWS:
    return "dsp.sync", None
  return "dsp.sync", None


def class_for(name, tid):
  if tid == TID_DMA:
    return "dma"
  for rx, cls in CLASS_RULES:
    if rx.search(name):
      return cls
  return "elementwise"


def op_for(name):
  """`node_linear_2::q::Reshape` -> node_linear_2; `q::Softmax_Crouton` -> q::Softmax_Crouton."""
  m = re.match(r"^(node_[A-Za-z0-9_]+)", name)
  return m.group(1) if m else name.split(".")[0]


def elems_for(args):
  for k in SHAPE_KEYS:
    v = args.get(k)
    if isinstance(v, str):
      try:
        v = json.loads(v)
      except ValueError:
        continue
    if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
      n = 1
      for x in v:
        n *= int(x)
      return n
  return None


def cycles_for(args):
  for k in CYCLE_KEYS:
    v = args.get(k)
    if isinstance(v, (int, float)):
      return int(v)
    if isinstance(v, str) and v.replace(".", "", 1).isdigit():
      return int(float(v))
  return None


def convert(doc, ts_unit="us", clock_mhz=1200.0, keep_views=False, pid=PID_QNN):
  ev_in = doc.get("traceEvents", doc) if isinstance(doc, dict) else doc
  scale = (1.0 / clock_mhz) if ts_unit == "cycles" else 1.0
  out = [{"ph": "M", "pid": pid, "name": "process_name", "args": {"name": "QNN HTP (optrace)"}},
         {"ph": "M", "pid": pid, "name": "process_sort_index", "args": {"sort_index": 9}}]
  seen_tids = {}
  for e in ev_in:
    ph = e.get("ph")
    if ph == "M":
      if e.get("name") == "thread_name" and "tid" in e:
        seen_tids[e["tid"]] = e["args"]["name"]
      continue
    if ph not in ("X", "B", "E"):
      continue
    tid = e.get("tid", 0)
    if tid == TID_VIEWS and not keep_views:
      continue
    cat, engine = cat_for(tid)
    a = dict(e.get("args") or {})
    a["op"] = op_for(e["name"])
    a["class"] = class_for(e["name"], tid)
    if engine:
      a["engine"] = engine
    n = elems_for(a)
    if n:
      a["elems"] = n
    c = cycles_for(a)
    if c:
      a["cycles"] = c
    x = {"ph": ph, "pid": pid, "tid": tid, "name": e["name"], "cat": cat,
         "ts": round(e["ts"] * scale, 3), "args": a}
    if ph == "X":
      x["dur"] = round(e.get("dur", 0) * scale, 3)
    out.append(x)
    seen_tids.setdefault(tid, None)
  for tid, name in sorted(seen_tids.items()):
    label = name or ("DMA" if tid == TID_DMA else "HMX" if tid == TID_HMX else
                     "HVX-%d" % (tid - TID_HVX0) if TID_HVX0 <= tid <= TID_HVX_LAST else
                     "views" if tid == TID_VIEWS else "tid %d" % tid)
    out.append({"ph": "M", "pid": pid, "tid": tid, "name": "thread_name", "args": {"name": label}})
    out.append({"ph": "M", "pid": pid, "tid": tid, "name": "thread_sort_index",
                "args": {"sort_index": 0 if tid == TID_HMX else 1 + (tid - TID_HVX0) if TID_HVX0 <= tid <= TID_HVX_LAST else 20 if tid == TID_DMA else 30}})
  md = {"tool": "qnn_optrace_to_nntr", "source": "QNN HTP optrace", "dsp_clock_mhz": clock_mhz,
        "hvx_threads": len([t for t in seen_tids if TID_HVX0 <= t <= TID_HVX_LAST]) or 8,
        "ts_unit_in": ts_unit}
  if isinstance(doc, dict) and isinstance(doc.get("metadata"), dict):
    md["qnn_metadata"] = doc["metadata"]
  return {"displayTimeUnit": "ms", "traceEvents": out, "metadata": md}


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("input")
  ap.add_argument("-o", "--out", required=True)
  ap.add_argument("--ts-unit", choices=("us", "cycles"), default="us")
  ap.add_argument("--clock-mhz", type=float, default=1200.0)
  ap.add_argument("--keep-views", action="store_true",
                  help="keep the tid 1 zero-cost view ops as dsp.sync slices")
  ap.add_argument("--pid", type=int, default=PID_QNN)
  a = ap.parse_args()
  with open(a.input) as f:
    doc = json.load(f)
  out = convert(doc, a.ts_unit, a.clock_mhz, a.keep_views, a.pid)
  with open(a.out, "w") as f:
    json.dump(out, f, separators=(",", ":"))
  n = sum(1 for e in out["traceEvents"] if e["ph"] == "X")
  print("%s: %d slices on pid %d" % (a.out, n, a.pid))


if __name__ == "__main__":
  main()
