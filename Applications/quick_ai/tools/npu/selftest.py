#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
"""
Self-test for the NPU graph builder: build a tiny synthetic model, run the
ONNX graphs with onnxruntime and compare against the NumPy reference that
uses exact int32 accumulation (the HMX numerics).

  python3 selftest.py            # exit 0 on success
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from npu_quant import quantize_act  # noqa: E402
from reference import (QuantizedLayer, ref_attn_in_exact,  # noqa: E402
                       ref_attn_out_ffn)


def load_ql(npz_path):
    z = np.load(npz_path)
    layer, qu, kv, hidden, inter = [int(v) for v in z["meta"]]
    act = {k[4:]: (float(z[k][0]), int(z[k][1]), int(z[k][2]))
           for k in z.files if k.startswith("act_")}
    return QuantizedLayer(
        layer=layer, q_units=qu, kv_units=kv, hidden=hidden, inter=inter,
        eps=float(z["eps"][0]),
        w_qkv_q=z["w_qkv_q"], w_qkv_s=z["w_qkv_s"], wo_q=z["wo_q"],
        wo_s=z["wo_s"], w_gate_q=z["w_gate_q"], w_gate_s=z["w_gate_s"],
        w_up_q=z["w_up_q"], w_up_s=z["w_up_s"], w_down_q=z["w_down_q"],
        w_down_s=z["w_down_s"], ffn_norm_gamma=z["ffn_norm_gamma"],
        b_qkv=z["b_qkv"] if "b_qkv" in z.files else None, act=act)


def rel_err(a, b):
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


def run(a8_input: bool, a16: str, M: int = 4) -> bool:
    ok = True
    with tempfile.TemporaryDirectory() as td:
        cmd = [sys.executable, os.path.join(HERE, "build_decoder_graphs.py"),
               "--synthetic", "--out", td, "--buckets", str(M)]
        if not a8_input:
            cmd.append("--fp-input")
        if a16:
            cmd += ["--a16", a16]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL)

        assert os.path.exists(os.path.join(td, "npu_graphs.json"))
        assert os.path.exists(os.path.join(td, "convert.sh"))

        rng = np.random.default_rng(1)
        for layer in (0, 1):  # layer 1 has qkv bias in synthetic mode
            ql = load_ql(os.path.join(td, "weights", f"layer{layer}.npz"))
            H, Q = ql.hidden, ql.q_units

            # ---------------- graph A ----------------
            x = (rng.standard_normal((M, H)) * 2.0).astype(np.float32)
            s, zp, bits = ql.act["attn_in"]
            xq = quantize_act(x, s, zp, bits)
            sess = ort.InferenceSession(
                os.path.join(td, "onnx", f"layer{layer}_attn_in_m{M}.onnx"),
                providers=["CPUExecutionProvider"])
            feed = {"x": (xq if a8_input else x)[None]}
            q, k, v = [o[0] for o in sess.run(None, feed)]
            rq, rk, rv = ref_attn_in_exact(ql, xq)
            for name, got, ref in (("q", q, rq), ("k", k, rk), ("v", v, rv)):
                e = rel_err(got, ref)
                good = e < 1e-4  # fp32 dequant math vs exact int32
                ok &= good
                print(f"  L{layer} attn_in[{name}] a8={a8_input} "
                      f"rel_err={e:.2e} {'OK' if good else 'FAIL'}")

            # ---------------- graph B ----------------
            attn = (rng.standard_normal((M, Q)) * 1.5).astype(np.float32)
            resid = (rng.standard_normal((M, H)) * 1.0).astype(np.float32)
            sess = ort.InferenceSession(
                os.path.join(td, "onnx",
                             f"layer{layer}_attn_out_ffn_m{M}.onnx"),
                providers=["CPUExecutionProvider"])
            # graph B's attn input is always float (mha_core has no quantize
            # epilogue); the HTP quantizes it with the static params.
            feed = {"attn": attn[None], "resid": resid[None]}
            out = sess.run(None, feed)[0][0]
            ref = ref_attn_out_ffn(ql, attn, resid)
            e = rel_err(out, ref)
            # two internal requantizations: a one-LSB flip on a boundary
            # value is legal, so the bound is looser than graph A's.
            good = e < 5e-3
            ok &= good
            print(f"  L{layer} attn_out_ffn a8={a8_input} a16='{a16}' "
                  f"rel_err={e:.2e} {'OK' if good else 'FAIL'}")
    return ok


def main():
    ok = True
    print("[selftest] uint8 inputs, all-A8")
    ok &= run(a8_input=True, a16="")
    print("[selftest] float inputs, all-A8")
    ok &= run(a8_input=False, a16="")
    print("[selftest] uint8 inputs, ffn_act in A16")
    ok &= run(a8_input=True, a16="ffn_act")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
