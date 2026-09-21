#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# @package htp_profile_to_trace
# @brief Lift an NNTR_HTP_PROFILE=2 summary print into the nntr trace schema.
"""Lift the ``[HTP-PROFILE]`` summary a run prints into a trace.json.

``NNTR_HTP_PROFILE=2 ./nntrainer_causallm ...`` prints, at exit, one line per
call shape with the per-call averages:

    [HTP-PROFILE]   K=2048  N=2048  M==1    calls=11264   rows=11264    host=  24431.0 ms ( 2168.9 us/call)  dsp= 1354.5 us/call (62.5%) transport=  814.4 us/call  [quant 14.9 gather 139.1 requant 42.0 swiglu 0.0 dequant 13.1 acc 259.3 drain 105.8+8.1 push 2.5 scatter 0.9 alloc 0.2 stage 4.4 mm 747.7 | rest<=16.5 (0.8% of host) blocks=45056]

Those are averages over thousands of calls, so no timeline exists in them.
This script draws each shape as ONE representative call (host wait, seam
halves, DSP entry, stages back to back on their lanes) so the breakdown can
be looked at in the viewer next to a real trace. For a real per-call
timeline run with ``NNTR_TRACE=/path/trace.json`` instead (htp_trace.h on
the HTP branch), which writes the same schema from measured timestamps.

Usage
    tools/nntr_trace/htp_profile_to_trace.py run.log -o profile.json
    adb shell "... NNTR_HTP_PROFILE=2 ./nntrainer_causallm model" 2>&1 | tee run.log
"""

import argparse
import json
import re

PID_HOST, PID_DSP = 1, 2
TID_MAIN, TID_RPC, TID_REG = 1, 20, 30
TID_DSP_MAIN, TID_DMA, TID_HVX, TID_HMX = 1, 256, 512, 768

ROW = re.compile(
  r"\[HTP-PROFILE\]\s+K=(?P<K>\d+)\s+N=(?P<N>\d+)\s+(?P<shape>M==1|M>1)\s+"
  r"calls=(?P<calls>\d+)\s+rows=(?P<rows>\d+)\s+host=\s*(?P<host_ms>[\d.]+) ms"
  r"\s+\(\s*(?P<host_us>[\d.]+) us/call\)(?:\s+dsp=\s*(?P<dsp_us>[\d.]+) us/call"
  r"\s+\([\d.]+%\)\s+transport=\s*(?P<transport_us>[\d.]+) us/call\s+\[(?P<stages>[^\]]*)\])?")
STAGE = re.compile(r"(?P<name>[a-z_]+) (?P<v>[\d.]+)(?:\+(?P<v2>[\d.]+))?")
REG = re.compile(r"\[HTP-PROFILE\]\s+registration total\s*:\s*([\d.]+) ms")
REG_N = re.compile(r"\[HTP-PROFILE\]\s+weights registered\s*:\s*(\d+)")
LEVEL = re.compile(r"\[HTP-PROFILE\] level=(\d+) qos_mode=(\d+)")

# stage name -> (lane tid, cat, class, label)
LANES = {
  "alloc": (TID_DSP_MAIN, "dsp.sync", "sync", "alloc"),
  "gather": (TID_HVX, "dsp.hvx", "elementwise", "gather rows"),
  "quant": (TID_HVX, "dsp.hvx", "quant", "quant f32->u8 AH"),
  "push": (TID_DMA, "dsp.dma", "dma", "dma push"),
  "mm": (TID_HMX, "dsp.hmx", "matmul", "micro-mm"),
  "drain": (TID_DMA, "dsp.dma", "dma", "dma drain"),
  "drain_dn": (TID_DMA, "dsp.dma", "dma", "dma drain (down)"),
  "acc": (TID_HMX, "dsp.hmx", "matmul", "acc_read + acc_copy"),
  "requant": (TID_HVX, "dsp.hvx", "quant", "requant"),
  "swiglu": (TID_HVX, "dsp.hvx", "elementwise", "swiglu"),
  "dequant": (TID_HVX, "dsp.hvx", "dequant", "dequant i32->f32"),
  "scatter": (TID_HVX, "dsp.hvx", "elementwise", "scatter-add"),
  "stage": (TID_DSP_MAIN, "dsp.sync", "sync", "stage copies"),
  "rest": (TID_DSP_MAIN, "dsp.sync", "sync", "rest (unnamed)"),
}
ORDER = ["alloc", "gather", "quant", "push", "mm", "drain", "acc", "requant",
         "swiglu", "drain_dn", "dequant", "scatter", "stage", "rest"]


def parse(text):
  rows, reg_ms, reg_n, level, qos = [], 0.0, 0, None, None
  for line in text.splitlines():
    m = ROW.search(line)
    if m:
      d = m.groupdict()
      stages = {}
      if d.get("stages"):
        for s in STAGE.finditer(d["stages"]):
          stages[s["name"]] = float(s["v"])
          if s["name"] == "drain" and s["v2"]:
            stages["drain_dn"] = float(s["v2"])
        r = re.search(r"rest<=([\d.]+)", d["stages"])
        if r:
          stages["rest"] = float(r.group(1))
        b = re.search(r"blocks=(\d+)", d["stages"])
        stages["blocks"] = int(b.group(1)) if b else 0
      rows.append({"K": int(d["K"]), "N": int(d["N"]), "decode": d["shape"] == "M==1",
                   "calls": int(d["calls"]), "rows": int(d["rows"]),
                   "host_us": float(d["host_us"]),
                   "dsp_us": float(d["dsp_us"]) if d.get("dsp_us") else 0.0,
                   "transport_us": float(d["transport_us"]) if d.get("transport_us") else 0.0,
                   "stages": stages})
      continue
    m = REG.search(line)
    if m:
      reg_ms = float(m.group(1))
    m = REG_N.search(line)
    if m:
      reg_n = int(m.group(1))
    m = LEVEL.search(line)
    if m:
      level, qos = int(m.group(1)), int(m.group(2))
  return rows, reg_ms, reg_n, level, qos


def build(rows, reg_ms, reg_n, level, qos):
  ev = []

  def meta(pid, tid, name, sort):
    e = {"ph": "M", "pid": pid, "name": "process_name" if tid is None else "thread_name",
         "args": {"name": name}}
    if tid is not None:
      e["tid"] = tid
    ev.append(e)
    s = {"ph": "M", "pid": pid, "name": "process_sort_index" if tid is None else "thread_sort_index",
         "args": {"sort_index": sort}}
    if tid is not None:
      s["tid"] = tid
    ev.append(s)

  def x(pid, tid, name, cat, ts, dur, **args):
    ev.append({"ph": "X", "pid": pid, "tid": tid, "name": name, "cat": cat,
               "ts": round(ts, 3), "dur": round(dur, 3), "args": args})
    return ts + dur

  meta(PID_HOST, None, "host (CPU) -- one representative call per shape", 0)
  meta(PID_DSP, None, "HTP (cDSP, per-call averages)", 1)
  meta(PID_HOST, TID_MAIN, "main", 0)
  meta(PID_HOST, TID_RPC, "fastrpc seam", 5)
  meta(PID_HOST, TID_REG, "weight register", 6)
  meta(PID_DSP, TID_DSP_MAIN, "dsp main (caller)", 0)
  meta(PID_DSP, TID_HMX, "HMX", 1)
  meta(PID_DSP, TID_HVX, "HVX (pool total)", 2)
  meta(PID_DSP, TID_DMA, "DMA", 10)

  ts = 0.0
  if reg_ms:
    ts = x(PID_HOST, TID_REG, "weight registration (%d weights)" % reg_n, "host.load", ts,
           reg_ms * 1000.0, engine="CPU", weights=reg_n) + 200.0
  for r in sorted(rows, key=lambda r: (r["decode"], r["K"], r["N"])):
    M = 1 if r["decode"] else max(1, r["rows"] // max(1, r["calls"]))
    op = "K%d_N%d_%s" % (r["K"], r["N"], "decode" if r["decode"] else "prefill")
    t0 = ts
    half = max(0.0, r["transport_us"] / 2.0)
    t = x(PID_HOST, TID_RPC, "marshal " + op, "host.rpc", ts, half, op=op,
          bytes=M * r["K"] * 4)
    d0, dsp = t, r["dsp_us"]
    tt = d0
    for name in ORDER:
      v = r["stages"].get(name, 0.0)
      if v <= 0:
        continue
      tid, cat, cls, label = LANES[name]
      a = {"op": op, "stage": name, "class": cls, "M": M, "K": r["K"], "N": r["N"],
           "calls_averaged": r["calls"]}
      if tid in (TID_HMX, TID_HVX):
        a["engine"] = "HMX" if tid == TID_HMX else "HVX"
        if name == "mm":
          a["ops"] = a["elems"] = 2 * M * r["K"] * r["N"]
        elif name == "quant":
          a["elems"] = M * r["K"]
        elif name in ("dequant", "scatter"):
          a["elems"] = M * r["N"]
      tt = x(PID_DSP, tid, label, cat, tt, v, **a)
    d1 = max(tt, d0 + dsp)
    x(PID_DSP, TID_DSP_MAIN, "layer call " + op, "dsp.entry", d0, d1 - d0, op=op,
      dsp_total_us=dsp, blocks=r["stages"].get("blocks", 0), M=M, K=r["K"], N=r["N"],
      layout="averages over %d calls, stages laid out back to back" % r["calls"])
    # the return half absorbs whatever host time the DSP total did not cover
    t = x(PID_HOST, TID_RPC, "return " + op, "host.rpc", d1,
          max(half, r["host_us"] - (d1 - t0)), op=op, bytes=M * r["N"] * 4)
    x(PID_HOST, TID_MAIN, "wait " + op, "host.wait", t0, max(t - t0, r["host_us"]), op=op,
      engine="HTP", M=M, K=r["K"], N=r["N"], calls=r["calls"], rows=r["rows"],
      host_us=r["host_us"], dsp_us=dsp, transport_us=r["transport_us"],
      total_host_ms=r["host_us"] * r["calls"] / 1000.0)
    ts = max(t, t0 + r["host_us"]) + 100.0
  return {"displayTimeUnit": "ms", "traceEvents": ev,
          "metadata": {"tool": "htp_profile_to_trace (per-shape averages, one representative call each)",
                       "level": "summary", "profile_level": level, "qos_mode": qos,
                       "dsp_clock_mhz": 1200, "hvx_threads": 6, "htp_enabled": True,
                       "shapes": len(rows)}}


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("log")
  ap.add_argument("-o", "--out", default="htp_profile.json")
  a = ap.parse_args()
  with open(a.log, errors="replace") as f:
    rows, reg_ms, reg_n, level, qos = parse(f.read())
  if not rows:
    raise SystemExit("no [HTP-PROFILE] layer-call rows found in %s" % a.log)
  doc = build(rows, reg_ms, reg_n, level, qos)
  with open(a.out, "w") as f:
    json.dump(doc, f, separators=(",", ":"))
  print("%s: %d shapes, %d events" % (a.out, len(rows), len(doc["traceEvents"])))


if __name__ == "__main__":
  main()
