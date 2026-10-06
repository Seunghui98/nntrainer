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
  ... --config nntr_config.json                          # add a CPU/NPU column

The log does not say where a node ran, so the unit column comes from the
run's nntr_config.json: the engine keys and their *_htp_layers lists, by
the rule the model builders apply (an empty list means every layer). An
"htp" engine runs the prefill (M > 1) on the NPU; decode is not shown
here. Every other node is CPU. The MoE node is NPU though its top-k runs
on the CPU inside it (NNTR_M0_PROFILE splits them).
"""
import argparse
import collections
import json
import re

LAYER = re.compile(r"^layer(\d+)_(.+)$")
ROW = re.compile(r"^\s*(\S+):forward\((\w+)\)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)")

# op name in the graph -> (engine key, layer list key or None). The fused
# names (qkv, ffn, the two residual adds that carry the post norms) follow
# the projection before them, as the model builder gives them its engine;
# the unfused names stay for older graphs.
ENGINE_KEYS = {
    **{op: ("attn_proj_engine", "attn_proj_htp_layers")
       for op in ("wq", "wk", "wv", "qkv", "attention_out",
                  "post_attention_norm")},
    **{op: ("dense_ffn_engine", "dense_ffn_htp_layers")
       for op in ("ffn_gate", "ffn_up", "ffn_down", "ffn", "post_ffn_norm")},
    "sparse_moe": ("moe_engine", "moe_htp_layers"),
    "attention": ("attention_engine", None),
}


def unit_fn(config_path):
    if not config_path:
        return lambda op, layer: "?"
    cfg = json.load(open(config_path))

    def unit(op, layer):
        keys = ENGINE_KEYS.get(op)
        if keys is None or cfg.get(keys[0], "cpu") != "htp":
            return "CPU"
        if keys[1] is None:
            return "NPU"
        ids = {int(t) for t in str(cfg.get(keys[1], "")).split(",") if t.strip()}
        return "NPU" if not ids or layer in ids else "CPU"

    return unit


def read_rows(path, unit):
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
            lm = LAYER.match(m[1])
            u = unit(lm[2], int(lm[1])) if lm else unit(m[1], -1)
            rows.append((m[1], m[2], int(m[5]) / 1000.0, u))
    rows.reverse()  # the table is in reverse graph order
    return rows


def split(rows):
    t = collections.Counter()
    for _, _, ms, u in rows:
        t[u] += ms
    return "  ".join(f"{u} {ms:.1f} ms" for u, ms in sorted(t.items()))


def by_op(rows, total):
    """One line per op name with the layer index stripped (layerN_wq -> wq)."""
    ops = collections.OrderedDict()
    for name, typ, ms, u in rows:
        m = LAYER.match(name)
        key = (m[2] if m else name, typ)
        ops.setdefault(key, []).append((int(m[1]) if m else -1, ms, u))
    print(f"{'op':30} {'type':18} {'unit':>10} {'n':>3} {'avg ms':>8} "
          f"{'min':>8} {'max (layer)':>15} {'sum ms':>9} {'%':>6}")
    for (op, typ), v in sorted(ops.items(), key=lambda kv: -sum(x[1] for x in kv[1])):
        s = sum(x[1] for x in v)
        lmax, mx, _ = max(v, key=lambda t: t[1])
        where = f"{mx:8.2f} (L{lmax})" if lmax >= 0 else f"{mx:8.2f}      "
        units = collections.Counter(x[2] for x in v)
        unit = (next(iter(units)) if len(units) == 1 else
                "/".join(f"{u}{n}" for u, n in sorted(units.items())))
        print(f"{op:30.30} {typ:18.18} {unit:>10} {len(v):3d} {s / len(v):8.2f} "
              f"{min(x[1] for x in v):8.2f} {where:>15} {s:9.1f} "
              f"{100 * s / total:5.1f}% {'#' * int(50 * s / total)}")


def by_layer(rows, total):
    layers = collections.OrderedDict()
    for name, typ, ms, u in rows:
        m = LAYER.match(name)
        layers.setdefault(int(m[1]) if m else None, []).append(
            (m[2] if m else name, ms, u))
    sums = [sum(x[1] for x in v) for k, v in layers.items() if k is not None]
    avg = sum(sums) / len(sums)
    print(f"{'layer':>6} {'ms':>8} {'%':>6} {'vs avg':>7} {'NPU %':>6}  heaviest op")
    for k, v in layers.items():
        s = sum(x[1] for x in v)
        npu = sum(x[1] for x in v if x[2] == "NPU")
        op, ms, u = max(v, key=lambda t: t[1])
        label = f"L{k}" if k is not None else "other"
        print(f"{label:>6} {s:8.1f} {100 * s / total:5.1f}% {s / avg:6.2f}x "
              f"{100 * npu / s:5.0f}%  {op} [{u}] {ms:.1f} ms "
              f"({100 * ms / s:.0f}%) {'#' * int(40 * s / max(sums))}")
    print(f"layer avg {avg:.1f} ms; median "
          f"{sorted(sums)[len(sums) // 2]:.1f} ms over {len(sums)} layers")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--config", help="the run's nntr_config.json (unit column)")
    ap.add_argument("--layer", type=int, help="only layerN_* nodes")
    ap.add_argument("--top", type=int, help="the N slowest nodes instead")
    ap.add_argument("--min-ms", type=float, default=0.0)
    ap.add_argument("--by-op", action="store_true",
                    help="per op name over all layers: avg/min/max/sum/%%")
    ap.add_argument("--by-layer", action="store_true",
                    help="per layer: total, %%, vs the layer average")
    a = ap.parse_args()

    rows = read_rows(a.log, unit_fn(a.config))
    if not rows:
        raise SystemExit("no [PROFILE] table: is this a --profile build's log?")
    total = sum(r[2] for r in rows)
    print(f"prefill (sum of node max): {total:.1f} ms   [{split(rows)}]")
    if a.by_op or a.by_layer:
        if a.by_op:
            by_op(rows, total)
        if a.by_layer:
            by_layer(rows, total)
        return
    if a.layer is not None:
        rows = [r for r in rows if r[0].startswith(f"layer{a.layer}_")]
    if a.top:
        rows = sorted(rows, key=lambda r: -r[2])[: a.top]

    print(f"{'#':>4} {'node':44} {'type':18} {'unit':>4} {'ms':>8} "
          f"{'cum ms':>8} {'%':>6}")
    cum = 0.0
    peak = max(r[2] for r in rows)
    for i, (name, typ, ms, u) in enumerate(rows):
        cum += ms
        if ms < a.min_ms:
            continue
        print(f"{i:4d} {name:44.44} {typ:18.18} {u:>4} {ms:8.2f} {cum:8.1f} "
              f"{100 * ms / total:5.1f}% {'#' * int(40 * ms / peak)}")
    print(f"shown {cum:.1f} ms of {total:.1f} ms (sum of every node's max)")


if __name__ == "__main__":
    main()
