#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
"""
Turn activation dumps into the calibration JSON build_decoder_graphs.py
consumes.

Expected dump layout (any tool that writes raw fp16/fp32 rows works; the
CPU model with NNTR_DUMP_ACT=<dir> or a torch hook script):

  <dir>/layer{i}_{edge}_*.raw      edge in attn_in | attn | ffn_in | ffn_act

Per (layer, edge) the range is the [p_lo, p_hi] percentile over all dumped
values (default 0.01 / 99.99 -- clip the rare outliers instead of wasting the
8-bit range on them). Output:

  {"default": {edge: [lo, hi]}, "layers": {"i": {edge: [lo, hi]}}}
"""

import argparse
import glob
import json
import os
import re

import numpy as np

EDGES = ("attn_in", "attn", "ffn_in", "ffn_act")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--dtype", default="fp16", choices=("fp16", "fp32"))
    ap.add_argument("--p-lo", type=float, default=0.01)
    ap.add_argument("--p-hi", type=float, default=99.99)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dt = np.float16 if args.dtype == "fp16" else np.float32

    per = {}
    pat = re.compile(r"layer(\d+)_(attn_in|attn|ffn_in|ffn_act)_.*\.raw$")
    for fp in glob.glob(os.path.join(args.dir, "*.raw")):
        m = pat.match(os.path.basename(fp))
        if not m:
            continue
        layer, edge = int(m.group(1)), m.group(2)
        a = np.fromfile(fp, dtype=dt).astype(np.float32)
        per.setdefault(layer, {}).setdefault(edge, []).append(a)

    out = {"default": {}, "layers": {}}
    all_edge = {e: [] for e in EDGES}
    for layer in sorted(per):
        out["layers"][str(layer)] = {}
        for edge, chunks in per[layer].items():
            a = np.concatenate(chunks)
            lo, hi = np.percentile(a, [args.p_lo, args.p_hi])
            out["layers"][str(layer)][edge] = [float(lo), float(hi)]
            all_edge[edge].append(a)
            print(f"layer {layer:3d} {edge:8s} n={a.size:9d} "
                  f"[{lo:+.4f}, {hi:+.4f}] absmax={np.abs(a).max():.4f}")
    for edge, chunks in all_edge.items():
        if chunks:
            a = np.concatenate(chunks)
            lo, hi = np.percentile(a, [args.p_lo, args.p_hi])
            out["default"][edge] = [float(lo), float(hi)]
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
