#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# @package summarize
# @brief Derive the nntr_trace metrics (schema nntr_trace.metrics.v1) from a trace.json.
"""Derive the nntr_trace metrics from a trace.json, exactly as viewer.html does.

A port of the viewer's metrics / kernels / layers / tokens / transport /
pool-tail / warnings code (VIEWER_PLAN.md W12), kept definition-for-
definition identical so test/test_summarize.py can compare the two outputs
to 1e-6 relative. This is what the P5 device gate runs.

Usage
    tools/nntr_trace/summarize.py trace.json                  # JSON on stdout
    tools/nntr_trace/summarize.py trace.json -o metrics.json
    tools/nntr_trace/summarize.py trace.json --range phase:1  # 2nd phase slice
    tools/nntr_trace/summarize.py trace.json --range 1000:5000  # t0:t1 in us
    tools/nntr_trace/summarize.py trace.json \\
        --fail-if 'compression<1.5' --fail-if 'idle_ratio>0.05'  # exit 1 on breach

Stdlib only.
"""

import argparse
import json
import math
import re
import sys

EPS = 1e-6
LANES = ("dsp.hmx", "dsp.hvx", "dsp.dma")
BUSY_CATS = LANES + ("host.cpu",)
CAT = {
  "host.cpu": "cpu", "host.rpc": "rpc", "host.wait": "wait", "host.layer": "frame",
  "host.phase": "frame", "host.load": "load", "dsp.hmx": "hmx", "dsp.hvx": "hvx",
  "dsp.dma": "dma", "dsp.sync": "sync", "dsp.call": "frame", "dsp.entry": "frame",
  "qnn.op": "cpu",
}


# ---------- model ----------
class Ev(dict):
  """A trace event with the derived fields the viewer attaches (_label, _th...)."""
  __getattr__ = dict.get


def parse_events(doc):
  ev = doc.get("traceEvents", doc) if isinstance(doc, dict) else doc
  procs = {}

  def P(pid):
    return procs.setdefault(pid, {"pid": pid, "name": "pid %s" % pid, "sort": pid,
                                  "threads": {}})

  def TH(pid, tid):
    p = P(pid)
    return p["threads"].setdefault(tid, {"pid": pid, "tid": tid, "name": "tid %s" % tid,
                                         "sort": tid, "ev": [], "depth": 1})

  tmin, tmax = math.inf, -math.inf
  stacks, asyncs = {}, {}

  def push(th, x):
    nonlocal tmin, tmax
    th["ev"].append(x)
    tmin = min(tmin, x["ts"])
    tmax = max(tmax, x["ts"] + x["dur"])

  for e in ev:
    ph = e.get("ph")
    if ph == "M":
      a = e.get("args", {})
      if e["name"] == "process_name":
        P(e["pid"])["name"] = a["name"]
      elif e["name"] == "process_sort_index":
        P(e["pid"])["sort"] = a["sort_index"]
      elif e["name"] == "thread_name":
        TH(e["pid"], e["tid"])["name"] = a["name"]
      elif e["name"] == "thread_sort_index":
        TH(e["pid"], e["tid"])["sort"] = a["sort_index"]
    elif ph == "X":
      push(TH(e["pid"], e["tid"]), Ev(e))
    elif ph in ("B", "E"):
      th = TH(e["pid"], e["tid"])
      st = stacks.setdefault(id(th), [])
      if ph == "B":
        st.append(e)
      elif st:
        b = st.pop()
        push(th, Ev(dict(b, ph="X", dur=e["ts"] - b["ts"])))
    elif ph in ("b", "e"):
      th = TH(e["pid"], e["tid"])
      am = asyncs.setdefault(id(th), {})
      if ph == "b":
        am[e.get("id")] = e
      elif e.get("id") in am:
        b = am.pop(e["id"])
        push(th, Ev(dict(b, ph="X", dur=e["ts"] - b["ts"])))
  if not math.isfinite(tmin):
    tmin = 0.0
  plist = sorted(procs.values(), key=lambda p: p["sort"])
  for p in plist:
    p["tlist"] = sorted(p["threads"].values(), key=lambda t: t["sort"])
    for th in p["tlist"]:
      for e in th["ev"]:
        e["ts"] -= tmin
      th["ev"].sort(key=lambda e: (e["ts"], -e["dur"]))
      st = []
      for i, e in enumerate(th["ev"]):
        while st and st[-1]["ts"] + st[-1]["dur"] <= e["ts"] + EPS:
          st.pop()
        e["_d"] = len(st)
        st.append(e)
        e["_cat"] = CAT.get(e.get("cat"), "sync")
        e["_th"] = th
        a = e.get("args") or {}
        e["_op"] = a.get("op") or e["name"]
        e["_label"] = (a["op"] + ": " + e["name"]
                       if a.get("op") and a["op"] not in e["name"] else e["name"])
  all_ev = [e for p in plist for t in p["tlist"] for e in t["ev"]]
  phases = sorted((e for e in all_ev if e.get("cat") == "host.phase"),
                  key=lambda e: e["ts"])
  return {"procs": plist, "tmin": 0.0, "tmax": tmax - tmin, "all": all_ev,
          "md": (doc.get("metadata") if isinstance(doc, dict) else None) or {},
          "phases": phases}


# ---------- interval math (same as the viewer) ----------
def union(iv):
  out = []
  for x in sorted((x for x in iv if x[1] > x[0]), key=lambda x: x[0]):
    if out and x[0] <= out[-1][1]:
      out[-1][1] = max(out[-1][1], x[1])
    else:
      out.append([x[0], x[1]])
  return out


def length(iv):
  return sum(x[1] - x[0] for x in iv)


def inter(a, b):
  o, i, j = [], 0, 0
  while i < len(a) and j < len(b):
    lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
    if hi > lo + EPS:
      o.append([lo, hi])
    if a[i][1] < b[j][1]:
      i += 1
    else:
      j += 1
  return o


def minus(a, b):
  o = []
  for x0, x1 in a:
    c = x0
    for y0, y1 in b:
      if y1 <= c + EPS:
        continue
      if y0 >= x1 - EPS:
        break
      if y0 > c + EPS:
        o.append([c, y0])
      c = max(c, y1)
    if c < x1 - EPS:
      o.append([c, x1])
  return o


def in_range(e, r):
  return e["ts"] < r["t1"] and e["ts"] + e["dur"] > r["t0"]


def clip(e, r):
  return [max(e["ts"], r["t0"]), min(e["ts"] + e["dur"], r["t1"])]


def spans(pred, r, all_ev):
  return union([clip(e, r) for e in all_ev if pred(e) and in_range(e, r)])


# ---------- metrics ----------
def metrics(r, all_ev):
  wall = r["t1"] - r["t0"]
  hmx = spans(lambda e: e.get("cat") == "dsp.hmx", r, all_ev)
  hvx = spans(lambda e: e.get("cat") == "dsp.hvx", r, all_ev)
  dma = spans(lambda e: e.get("cat") == "dsp.dma", r, all_ev)
  entry = spans(lambda e: e.get("cat") == "dsp.entry", r, all_ev)
  any_ = union(hmx + hvx + dma)
  rpc = spans(lambda e: e.get("cat") == "host.rpc", r, all_ev)
  cpu = spans(lambda e: e.get("cat") == "host.cpu", r, all_ev)
  load = spans(lambda e: e.get("cat") == "host.load", r, all_ev)
  overlap = inter(hmx, hvx)
  idle = minus(entry, any_)
  other = wall - length(union(cpu + rpc + any_ + idle + load))
  occ = []
  seen = []
  for e in all_ev:
    th = e["_th"]
    if any(t is th for t in seen):
      continue
    seen.append(th)
  for th in seen:
    if not any(x.get("cat") in BUSY_CATS for x in th["ev"]):
      continue
    c = next(x for x in th["ev"] if x.get("cat") in BUSY_CATS)
    busy = length(spans(lambda e, th=th: e["_th"] is th and e.get("cat") in BUSY_CATS,
                        r, all_ev)) / wall
    occ.append({"th": th, "cat": c["_cat"], "busy": busy})
  occ.sort(key=lambda o: (o["th"]["pid"], o["th"]["sort"]))
  la = length(any_)
  return {
    "wall": wall, "cpu": length(cpu), "rpc": length(rpc),
    "hmxOnly": length(minus(hmx, hvx)), "hvxOnly": length(minus(hvx, hmx)),
    "overlap": length(overlap), "dmaOnly": length(minus(dma, union(hmx + hvx))),
    "idle": length(idle), "load": length(load), "other": max(0.0, other),
    "any": la, "entry": length(entry),
    "compression": (length(hmx) + length(hvx) + length(dma)) / la if la else 0.0,
    "occ": occ,
  }


def ops_of(e):
  a = e.get("args") or {}
  if a.get("ops"):
    return a["ops"]
  if e.get("cat") in ("dsp.call", "host.wait") and a.get("M") and a.get("K") and a.get("N"):
    return 2 * a["M"] * a["K"] * a["N"] * (a.get("n_handles") or 1)
  return None


def kernels(r, all_ev, md):
  m = {}
  mhz = md.get("dsp_clock_mhz") or 1200
  for e in all_ev:
    a = e.get("args") or {}
    if not (a.get("engine") and a.get("elems") and e.get("cat") in ("dsp.hmx", "dsp.hvx")
            and in_range(e, r)):
      continue
    k = a["engine"] + "|" + e["_label"]
    row = m.setdefault(k, {"engine": a["engine"], "name": e["_label"], "n": 0, "us": 0.0,
                           "elems": 0, "cyc": 0, "ops": 0})
    row["n"] += 1
    row["us"] += e["dur"]
    row["elems"] += a["elems"]
    row["cyc"] += a.get("cycles") or 0
    row["ops"] += ops_of(e) or 0
  out = []
  for k in m.values():
    k["cpe"] = (k["cyc"] or k["us"] * mhz) / k["elems"]
    k["tops"] = k["ops"] / k["us"] / 1e6 if k["ops"] else None
    out.append(k)
  return sorted(out, key=lambda k: -k["us"])


def layer_rows(r, all_ev):
  return [(e, metrics({"t0": max(e["ts"], r["t0"]), "t1": min(e["ts"] + e["dur"], r["t1"])},
                      all_ev))
          for e in all_ev if e.get("cat") == "host.layer" and in_range(e, r)]


def token_rows(all_ev, phases):
  rows = []
  for p in phases:
    r = {"t0": p["ts"], "t1": p["ts"] + p["dur"]}
    m = metrics(r, all_ev)
    waits = [e for e in all_ev if e.get("cat") == "host.wait" and in_range(e, r)]
    a = p.get("args") or {}
    b = sum(((e.get("args") or {}).get("bytes_in") or 0) +
            ((e.get("args") or {}).get("bytes_out") or 0) for e in waits)
    rows.append({"e": p, "name": p["name"], "phase": a.get("phase"), "token": a.get("token"),
                 "wall": p["dur"], "calls": len(waits), "bytes": b, "transport": m["rpc"],
                 "dsp": m["any"], "cpu": m["cpu"],
                 "other": max(0.0, p["dur"] - m["rpc"] - m["any"] - m["cpu"])})
  return rows


def token_summary(rows):
  dec = [x for x in rows if x["phase"] == "decode"]
  pre = next((x for x in rows if x["phase"] == "prefill"), None)

  def med(a):
    if not a:
      return None
    s = sorted(a)
    return s[(len(s) - 1) >> 1]
  return {"ttft_us": pre["wall"] if pre else None,
          "median_token_us": med([x["wall"] for x in dec]),
          "calls_per_token": (sum(x["calls"] for x in dec) / len(dec)) if dec else None,
          "tokens": len(dec)}


def transport_calls(r, all_ev):
  calls = []
  by_thread = {}
  for e in all_ev:
    if e.get("cat") == "host.rpc":
      by_thread.setdefault(id(e["_th"]), []).append(e)
  waits = [e for e in all_ev if e.get("cat") == "host.wait"]
  for evs in by_thread.values():
    open_ = None
    for e in evs:
      if e["name"].startswith("marshal "):
        open_ = e
      elif e["name"].startswith("return ") and open_ and e["name"][7:] == open_["name"][8:]:
        b = ((open_.get("args") or {}).get("bytes") or 0) + ((e.get("args") or {}).get("bytes") or 0)
        us = open_["dur"] + e["dur"]
        if in_range(open_, r) or in_range(e, r):
          wait = next((w for w in waits if w["ts"] <= open_["ts"] + EPS
                       and w["ts"] + w["dur"] >= e["ts"] + e["dur"] - EPS), None)
          calls.append({"name": open_["name"][8:], "marshal": open_, "ret": e, "bytes": b,
                        "us": us, "wait": wait, "resid": 0.0, "outlier": False})
        open_ = None
  n = len(calls)
  fixed = slope = sigma = 0.0
  if n >= 2:
    mx = sum(c["bytes"] for c in calls) / n
    my = sum(c["us"] for c in calls) / n
    sxx = sum((c["bytes"] - mx) ** 2 for c in calls)
    sxy = sum((c["bytes"] - mx) * (c["us"] - my) for c in calls)
    slope = sxy / sxx if sxx > 0 else 0.0
    fixed = my - slope * mx
    for c in calls:
      c["resid"] = c["us"] - (fixed + slope * c["bytes"])
    sigma = math.sqrt(sum(c["resid"] ** 2 for c in calls) / max(1, n - 2))
    for c in calls:
      c["outlier"] = sigma > 0 and abs(c["resid"]) > 2 * sigma
  outliers = sorted((c for c in calls if c["outlier"]), key=lambda c: -abs(c["resid"]))
  return {"calls": calls, "n": n, "fixed_us": fixed, "us_per_mb": slope * 1e6,
          "sigma": sigma, "outliers": outliers}


def pool_tail(r, all_ev, md):
  units = {}
  for e in all_ev:
    a = e.get("args") or {}
    if e.get("cat") == "dsp.hvx" and a.get("pool") is not None:
      units.setdefault(a["pool"], []).append(e)
  runs = []
  for e in all_ev:
    a = e.get("args") or {}
    if not (e.get("cat") == "dsp.sync" and e["name"].startswith("pool_run ")
            and a.get("pool") is not None and in_range(e, r)):
      continue
    u = units.get(a["pool"]) or []
    if not u:
      continue
    durs = [x["dur"] for x in u]
    mx, mean = max(durs), sum(durs) / len(durs)
    runs.append({"e": e, "kind": e["name"][9:], "units": len(u),
                 "n_units": a.get("n_units") or len(u), "max": mx, "mean": mean,
                 "tail": mx / mean, "join_gap": e["dur"] - mx})
  threads = md.get("hvx_threads") or 6
  groups = {}
  for x in runs:
    g = groups.setdefault(x["kind"], {"kind": x["kind"], "runs": 0, "tail_sum": 0.0,
                                      "worst": None, "min_units": math.inf})
    g["runs"] += 1
    g["tail_sum"] += x["tail"]
    g["min_units"] = min(g["min_units"], x["n_units"])
    if g["worst"] is None or x["tail"] > g["worst"]["tail"]:
      g["worst"] = x
  out = []
  for g in groups.values():
    g["mean_tail"] = g["tail_sum"] / g["runs"]
    g["few_units"] = g["min_units"] < threads * 4
    out.append(g)
  return {"runs": runs, "groups": sorted(out, key=lambda g: -g["worst"]["tail"]),
          "rule": threads * 4}


def warnings_(all_ev, md):
  w = []
  fb = [e for e in all_ev if (e.get("args") or {}).get("fallback")]
  dropped = md.get("dropped") or {}
  if md.get("htp_enabled") is False:
    w.append("htp")
  if fb:
    w.append("fallback")
  if (dropped.get("dsp") or 0) + (dropped.get("host") or 0) > 0:
    w.append("dropped")
  if (md.get("clock_sync") or {}).get("violations", 0) > 0:
    w.append("clock")
  return {"items": w, "fallbacks": len(fb), "fallback_us": sum(e["dur"] for e in fb),
          "dropped": dropped, "clock_violations": (md.get("clock_sync") or {}).get("violations") or 0}


def metrics_json(model, r):
  all_ev, md = model["all"], model["md"]
  m = metrics(r, all_ev)
  tok = token_rows(all_ev, model["phases"])
  t = transport_calls(r, all_ev)
  return {
    "schema": "nntr_trace.metrics.v1",
    "range": {"label": r["label"], "t0_us": r["t0"], "t1_us": r["t1"]},
    "wall_us": m["wall"],
    "buckets": {"cpu": m["cpu"], "rpc": m["rpc"], "hmx_only": m["hmxOnly"],
                "overlap": m["overlap"], "hvx_only": m["hvxOnly"], "dma_only": m["dmaOnly"],
                "dsp_idle": m["idle"], "load": m["load"], "other": m["other"]},
    "compression": m["compression"],
    "idle_ratio": m["idle"] / m["entry"] if m["entry"] else 0.0,
    "engines": [{"track": o["th"]["name"], "busy": o["busy"]} for o in m["occ"]],
    "kernels": [{"engine": k["engine"], "name": k["name"], "calls": k["n"],
                 "total_us": k["us"], "mean_us": k["us"] / k["n"], "elems": k["elems"],
                 "cy_per_elem": k["cpe"], "tops": k["tops"]} for k in kernels(r, all_ev, md)],
    "layers": [{"name": e["name"], "phase": (e.get("args") or {}).get("phase"),
                "wall_us": e["dur"], "cpu": lm["cpu"], "rpc": lm["rpc"],
                "hmx_only": lm["hmxOnly"], "overlap": lm["overlap"], "hvx_only": lm["hvxOnly"],
                "dma_only": lm["dmaOnly"], "dsp_idle": lm["idle"]}
               for e, lm in layer_rows(r, all_ev)],
    "warnings": warnings_(all_ev, md),
    "tokens": [{"name": x["name"], "phase": x["phase"], "token": x["token"],
                "wall_us": x["wall"], "calls": x["calls"], "bytes": x["bytes"],
                "transport_us": x["transport"], "dsp_us": x["dsp"], "cpu_us": x["cpu"]}
               for x in tok],
    "token_summary": token_summary(tok),
    "transport": {"n": t["n"], "fixed_us": t["fixed_us"], "us_per_mb": t["us_per_mb"],
                  "sigma_us": t["sigma"],
                  "outliers": [{"name": c["name"], "bytes": c["bytes"], "us": c["us"],
                                "residual": c["resid"]} for c in t["outliers"]]},
    "pool_tail": [{"kind": g["kind"], "runs": g["runs"], "mean_tail": g["mean_tail"],
                   "worst_tail": g["worst"]["tail"], "min_units": g["min_units"],
                   "few_units": g["few_units"]} for g in pool_tail(r, all_ev, md)["groups"]],
    "hmx_peak_tops": md.get("hmx_peak_tops"),
  }


def resolve_range(model, spec):
  if not spec or spec == "all":
    return {"t0": model["tmin"], "t1": model["tmax"], "label": "whole trace"}
  m = re.match(r"^phase:(\d+)$", spec)
  if m:
    p = model["phases"][int(m.group(1))]
    return {"t0": p["ts"], "t1": p["ts"] + p["dur"], "label": p["name"]}
  m = re.match(r"^([\d.]+):([\d.]+)$", spec)
  if m:
    return {"t0": float(m.group(1)), "t1": float(m.group(2)), "label": "selection"}
  raise SystemExit("bad --range %r (all | phase:N | t0:t1)" % spec)


def check_gates(out, gates):
  """--fail-if 'compression<1.5' style expressions over top-level scalars."""
  failed = []
  for g in gates:
    m = re.match(r"^\s*([a-z_.]+)\s*(<=|>=|<|>|==)\s*([\d.]+)\s*$", g)
    if not m:
      raise SystemExit("bad --fail-if %r" % g)
    key, op, val = m.group(1), m.group(2), float(m.group(3))
    cur = out
    for part in key.split("."):
      cur = cur[part]
    hit = {"<": cur < val, "<=": cur <= val, ">": cur > val, ">=": cur >= val,
           "==": cur == val}[op]
    if hit:
      failed.append("%s = %s (%s)" % (key, cur, g))
  return failed


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("trace")
  ap.add_argument("-o", "--out")
  ap.add_argument("--range", default="all", help="all | phase:N | t0:t1 (us)")
  ap.add_argument("--fail-if", action="append", default=[],
                  help="expression over the output, e.g. compression<1.5; exit 1 when true")
  a = ap.parse_args()
  with open(a.trace) as f:
    model = parse_events(json.load(f))
  out = metrics_json(model, resolve_range(model, a.range))
  text = json.dumps(out, indent=1)
  if a.out:
    with open(a.out, "w") as f:
      f.write(text)
  else:
    print(text)
  failed = check_gates(out, a.fail_if)
  for f in failed:
    print("GATE FAILED: " + f, file=sys.stderr)
  sys.exit(1 if failed else 0)


if __name__ == "__main__":
  main()
