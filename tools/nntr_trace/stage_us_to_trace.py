#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# @package stage_us_to_trace
# @brief Lift FC_STAGE / ATTN_STAGE marker logs into the nntr trace schema.
"""Lift the existing HTP device-run logs into a trace.json.

The device tests (unittest_hvx_fc.cpp, unittest_hvx_attn.cpp on the HTP
branches) print one ``FC_FIELD`` / ``FC_STAGE`` / ``ATTN_FIELD`` /
``ATTN_STAGE`` marker line per number:

    FC_STAGE path=fc_q_proj_m1024 field=quant_us value=720
    ATTN_STAGE path=forward_pre_kv1024_i8i8 field=softmax_us value=150

Those are per-call *totals* from the ``*_timed`` entries and the Tier 0
probes, so they have durations but no timestamps. This script lays every
path out as one call, stages back to back on the lane that runs them, so the
numbers can be looked at in the timeline viewer and compared with the real
tracer's output once it exists (docs/backend_guide/HTP_TRACE_PROFILER.md P0).
Positions inside a call are therefore synthetic; only the durations and the
host/transport/DSP split are measured.

Usage
    tools/nntr_trace/stage_us_to_trace.py run.log [more.log ...] -o trace.json
"""

import argparse
import json
import re

MARKER = re.compile(
  r"^(?P<kind>FC_FIELD|FC_STAGE|ATTN_FIELD|ATTN_STAGE)\s+path=(?P<path>\S+)\s+"
  r"field=(?P<field>\S+)\s+value=(?P<value>\S+)\s*$")

PID_HOST, PID_DSP = 1, 2
TID_MAIN, TID_RPC = 1, 20
TID_DSP_MAIN, TID_DMA, TID_HVX0, TID_HMX, TID_PROBE = 1, 256, 512, 768, 900

# (field, lane, label) in the order the kernel runs them.
FC_LANES = [("quant_us", TID_HVX0, "quant f32->u8 AH"),
            ("__hmx", TID_HMX, "micro-mm (dsp_total - probes)"),
            ("acc_read_us", TID_HMX, "acc_read (HMX acc -> VTCM)"),
            ("acc_copy_us", TID_HVX0, "acc_copy (VTCM -> DDR)"),
            ("dequant_us", TID_HVX0, "dequant i32->f32"),
            ("drain_us", TID_DMA, "dma drain")]
ATTN_LANES = [("gather_us", TID_HVX0, "q gather"),
              ("qk_us", TID_HMX, "Q.Kt (incl. probes)"),
              ("softmax_us", TID_HVX0, "softmax"),
              ("pv_us", TID_HMX, "P.V (incl. probes)"),
              ("accum_us", TID_HVX0, "o accumulate")]
PROBES = ["acc_read_us", "acc_copy_us", "dequant_us", "quant_us", "drain_us"]


def parse(paths):
  rec = {}
  for p in paths:
    with open(p) as f:
      for line in f:
        m = MARKER.match(line.strip())
        if not m:
          continue
        try:
          v = float(m["value"])
        except ValueError:
          v = m["value"]
        rec.setdefault(m["path"], {})[(m["kind"].split("_")[1], m["field"])] = v
  return rec


def build(rec):
  ev = []
  meta = [("process_name", PID_HOST, None, "host (CPU)"),
          ("process_name", PID_DSP, None, "HTP (cDSP)"),
          ("thread_name", PID_HOST, TID_MAIN, "main"),
          ("thread_name", PID_HOST, TID_RPC, "fastrpc seam"),
          ("thread_name", PID_DSP, TID_DSP_MAIN, "dsp main (caller)"),
          ("thread_name", PID_DSP, TID_HMX, "HMX"),
          ("thread_name", PID_DSP, TID_HVX0, "HVX-0 (pool total)"),
          ("thread_name", PID_DSP, TID_DMA, "DMA"),
          ("thread_name", PID_DSP, TID_PROBE, "Tier 0 probes (totals)")]
  for name, pid, tid, label in meta:
    e = {"ph": "M", "pid": pid, "name": name, "args": {"name": label}}
    if tid is not None:
      e["tid"] = tid
    ev.append(e)

  def x(pid, tid, name, cat, ts, dur, **args):
    ev.append({"ph": "X", "pid": pid, "tid": tid, "name": name, "cat": cat,
               "ts": round(ts, 3), "dur": round(dur, 3), "args": args})
    return ts + dur

  ts = 0.0
  for path in sorted(rec):
    r = rec[path]
    g = lambda f, d=0.0: r.get(("STAGE", f), r.get(("FIELD", f), d))
    wall = g("us_avg_1_10")
    dsp = g("dsp_total_us")
    transport = max(0.0, g("transport_us", wall - dsp))
    if not wall and not dsp:
      continue
    is_attn = path.startswith("forward_")
    lanes = ATTN_LANES if is_attn else FC_LANES
    shape = {k: int(g(k)) for k in ("M", "K", "N") if g(k)}
    t0 = ts
    t = x(PID_HOST, TID_RPC, "marshal " + path, "host.rpc", ts, transport / 2,
          path=path)
    d0 = t
    tt = d0
    stage_sum = sum(g(f) for f, _, _ in lanes if f != "__hmx")
    for f, lane, label in lanes:
      dur = g(f) if f != "__hmx" else max(0.0, dsp - stage_sum)
      if dur <= 0:
        continue
      tt = x(PID_DSP, lane, label, "dsp.hmx" if lane == TID_HMX else
             "dsp.dma" if lane == TID_DMA else "dsp.hvx", tt, dur, path=path,
             stage=f, engine="HMX" if lane == TID_HMX else "HVX", **shape)
    if is_attn:
      pt = d0
      for f in PROBES:
        if g(f) > 0:
          pt = x(PID_DSP, TID_PROBE, f, "dsp.sync", pt, g(f), path=path)
    d1 = max(tt, d0 + dsp)
    x(PID_DSP, TID_DSP_MAIN, path, "dsp.entry", d0, d1 - d0, path=path,
      dsp_total_us=dsp, **shape)
    t = x(PID_HOST, TID_RPC, "return " + path, "host.rpc", d1, transport / 2,
          path=path)
    x(PID_HOST, TID_MAIN, "wait " + path, "host.wait", t0, t - t0, path=path,
      engine="HTP", us_avg_1_10=wall, **shape)
    ts = t + 100.0
  return {"displayTimeUnit": "ms", "traceEvents": ev,
          "metadata": {"tool": "stage_us_to_trace (totals laid out sequentially)",
                       "dsp_clock_mhz": 1200}}


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("logs", nargs="+")
  ap.add_argument("-o", "--out", default="trace_stage_us.json")
  a = ap.parse_args()
  doc = build(parse(a.logs))
  with open(a.out, "w") as f:
    json.dump(doc, f, separators=(",", ":"))
  print("%s: %d events" % (a.out, len(doc["traceEvents"])))


if __name__ == "__main__":
  main()
