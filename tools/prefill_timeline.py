#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prefill timeline from a profile build's [PROFILE] table.

The table (Applications/CausalLM/main.cpp, build_android.sh --profile)
lists every layer node in reverse graph order with avg/min/max/sum over
the run. Prefill is one call per node and the longest one, so the max
column is the prefill time; reversing the rows gives execution order.

  python3 tools/prefill_timeline.py run.log              # every node
  python3 tools/prefill_timeline.py run.log --layer 5    # one layer
  python3 tools/prefill_timeline.py run.log --top 20     # slowest nodes
  python3 tools/prefill_timeline.py run.log --min-ms 1   # hide tiny nodes
"""
import argparse
import re

ROW = re.compile(r"^\s*(\S+):forward\((\w+)\)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)")


def read_rows(path):
    rows, started = [], False
    for line in open(path, errors="ignore"):
        if "[PROFILE] per-layer-type totals" in line:
            started = True
            continue
        if not started:
            continue
        m = ROW.match(line)
        # input nodes (KV cache placeholders) cost nothing and are listed
        # by name, not in graph order
        if m and m[2] != "input" and not m[1].endswith("generated_out_0"):
            rows.append((m[1], m[2], int(m[5]) / 1000.0))
    rows.reverse()  # the table is in reverse graph order
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--layer", type=int, help="only layerN_* nodes")
    ap.add_argument("--top", type=int, help="the N slowest nodes instead")
    ap.add_argument("--min-ms", type=float, default=0.0)
    a = ap.parse_args()

    rows = read_rows(a.log)
    if not rows:
        raise SystemExit("no [PROFILE] table: is this a --profile build's log?")
    total = sum(ms for _, _, ms in rows)
    if a.layer is not None:
        rows = [r for r in rows if r[0].startswith(f"layer{a.layer}_")]
    if a.top:
        rows = sorted(rows, key=lambda r: -r[2])[: a.top]

    print(f"{'#':>4} {'node':44} {'type':18} {'ms':>8} {'cum ms':>8} {'%':>6}")
    cum = 0.0
    for i, (name, typ, ms) in enumerate(rows):
        cum += ms
        if ms < a.min_ms:
            continue
        bar = "#" * int(40 * ms / max(r[2] for r in rows))
        print(f"{i:4d} {name:44.44} {typ:18.18} {ms:8.2f} {cum:8.1f} "
              f"{100 * ms / total:5.1f}% {bar}")
    print(f"shown {cum:.1f} ms of {total:.1f} ms (sum of every node's max)")


if __name__ == "__main__":
    main()
