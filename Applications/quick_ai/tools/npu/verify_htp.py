#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
"""
Compare a device (HTP) graph output dump against the exact NumPy reference.

The reference accumulates every FC in int32 with the same int8 weights the
HTP graph carries, so for graph A the only differences allowed are the FP16
output rounding (and the HTP's own final requantization); a large error means
a wrong offset/scale, a transposed weight, or a mismatched bucket -- not
"quantization noise".

Dump the graph IO from the device with NNTR_QNN_DUMP=<dir> (see QNNGraph),
then:

  verify_htp.py --weights npu/weights/layer0.npz --graph attn_in \
      --x  dump/layer0_attn_in_m1.in0.raw  --out dump/layer0_attn_in_m1.out0.raw \
      --M 1 --out-dtype fp16

  verify_htp.py --weights npu/weights/layer0.npz --graph attn_out_ffn \
      --x dump/...in0.raw --resid dump/...in1.raw --out dump/...out0.raw --M 1
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference import ref_attn_in_exact, ref_attn_out_ffn  # noqa: E402
from selftest import load_ql  # noqa: E402

DT = {"fp16": np.float16, "fp32": np.float32, "uint8": np.uint8,
      "uint16": np.uint16, "int32": np.int32}


def load_raw(path, dtype, shape):
    a = np.fromfile(path, dtype=DT[dtype])
    n = int(np.prod(shape))
    assert a.size >= n, f"{path}: {a.size} elems < expected {n}"
    return a[:n].reshape(shape)


def report(tag, got, ref):
    got = got.astype(np.float32)
    ref = ref.astype(np.float32)
    err = np.abs(got - ref)
    print(f"{tag}: max_abs={err.max():.4g} mean_abs={err.mean():.4g} "
          f"ref_absmax={np.abs(ref).max():.4g} "
          f"rel={err.max() / (np.abs(ref).max() + 1e-12):.3e}")
    if np.array_equal(got, ref):
        print(f"{tag}: BIT-EXACT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="weights/layer{i}.npz")
    ap.add_argument("--graph", required=True, choices=("attn_in",
                                                        "attn_out_ffn"))
    ap.add_argument("--M", type=int, required=True, help="bucket / rows")
    ap.add_argument("--x", required=True, help="input 0 raw dump")
    ap.add_argument("--x-dtype", default="uint8", choices=DT.keys())
    ap.add_argument("--resid", help="graph B input 1 (resid) raw dump")
    ap.add_argument("--resid-dtype", default="fp16", choices=DT.keys())
    ap.add_argument("--out", required=True, help="output raw dump (graph A: "
                    "q dump; use --out-k/--out-v for k, v)")
    ap.add_argument("--out-k")
    ap.add_argument("--out-v")
    ap.add_argument("--out-dtype", default="fp16", choices=DT.keys())
    args = ap.parse_args()

    ql = load_ql(args.weights)
    M = args.M
    if args.graph == "attn_in":
        s, zp, bits = ql.act["attn_in"]
        if args.x_dtype in ("uint8", "uint16"):
            xq = load_raw(args.x, args.x_dtype, (M, ql.hidden))
        else:
            from npu_quant import quantize_act
            xq = quantize_act(load_raw(args.x, args.x_dtype, (M, ql.hidden)),
                              s, zp, bits)
        rq, rk, rv = ref_attn_in_exact(ql, xq)
        report("q", load_raw(args.out, args.out_dtype, rq.shape), rq)
        if args.out_k:
            report("k", load_raw(args.out_k, args.out_dtype, rk.shape), rk)
        if args.out_v:
            report("v", load_raw(args.out_v, args.out_dtype, rv.shape), rv)
    else:
        assert args.resid, "--resid required for attn_out_ffn"
        s, zp, bits = ql.act["attn"]
        from npu_quant import dequantize_act
        if args.x_dtype in ("uint8", "uint16"):
            attn = dequantize_act(load_raw(args.x, args.x_dtype,
                                           (M, ql.q_units)), s, zp)
        else:
            attn = load_raw(args.x, args.x_dtype, (M, ql.q_units))
        resid = load_raw(args.resid, args.resid_dtype, (M, ql.hidden))
        ref = ref_attn_out_ffn(ql, attn, resid.astype(np.float32))
        report("out", load_raw(args.out, args.out_dtype, ref.shape), ref)


if __name__ == "__main__":
    main()
