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
  python3 tools/prefill_timeline.py run.log --by-op      # per op, over layers
  python3 tools/prefill_timeline.py run.log --by-layer   # per layer
"""
import argparse
import collections
import re

LAYER = re.compile(r"^layer(\d+)_(.+)$")


def by_op(rows, total):
    """One line per op name with the layer index stripped (layerN_wq -> wq)."""
    ops = collections.OrderedDict()
    for name, typ, ms in rows:
        m = LAYER.match(name)
        key = (m[2] if m else name, typ)
        ops.setdefault(key, []).append((int(m[1]) if m else -1, ms))
    print(f"{'op':34} {'type':18} {'n':>3} {'avg ms':>8} {'min':>8} "
          f"{'max (layer)':>14} {'sum ms':>9} {'%':>6}")
    for (op, typ), v in sorted(ops.items(), key=lambda kv: -sum(x for _, x in kv[1])):
        s = sum(x for _, x in v)
        lmax, mx = max(v, key=lambda t: t[1])
        where = f"{mx:8.2f} (L{lmax})" if lmax >= 0 else f"{mx:8.2f}     "
        print(f"{op:34.34} {typ:18.18} {len(v):3d} {s / len(v):8.2f} "
              f"{min(x for _, x in v):8.2f} {where:>14} {s:9.1f} "
              f"{100 * s / total:5.1f}% {'#' * int(50 * s / total)}")


def by_layer(rows, total):
    layers = collections.OrderedDict()
    for name, typ, ms in rows:
        m = LAYER.match(name)
        layers.setdefault(int(m[1]) if m else None, []).append(
            (m[2] if m else name, ms))
    sums = [sum(ms for _, ms in v) for k, v in layers.items() if k is not None]
    avg = sum(sums) / len(sums)
    print(f"{'layer':>6} {'ms':>8} {'%':>6} {'vs avg':>7}  heaviest op")
    for k, v in layers.items():
        s = sum(ms for _, ms in v)
        op, ms = max(v, key=lambda t: t[1])
        label = f"L{k}" if k is not None else "other"
        print(f"{label:>6} {s:8.1f} {100 * s / total:5.1f}% {s / avg:6.2f}x  "
              f"{op} {ms:.1f} ms ({100 * ms / s:.0f}%) "
              f"{'#' * int(40 * s / max(sums))}")
    print(f"layer avg {avg:.1f} ms; median "
          f"{sorted(sums)[len(sums) // 2]:.1f} ms over {len(sums)} layers")

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
    ap.add_argument("--by-op", action="store_true",
                    help="per op name over all layers: avg/min/max/sum/%%")
    ap.add_argument("--by-layer", action="store_true",
                    help="per layer: total, %%, vs the layer average")
    a = ap.parse_args()

    rows = read_rows(a.log)
    if not rows:
        raise SystemExit("no [PROFILE] table: is this a --profile build's log?")
    total = sum(ms for _, _, ms in rows)
    if a.by_op or a.by_layer:
        print(f"prefill (sum of node max): {total:.1f} ms")
        if a.by_op:
            by_op(rows, total)
        if a.by_layer:
            by_layer(rows, total)
        return
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
