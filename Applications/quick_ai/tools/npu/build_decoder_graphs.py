#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
"""
Build the per-layer NPU (Hexagon HTP / HMX) decoder graphs as QDQ ONNX models
and emit everything the nntrainer runtime needs to bind them.

Per decoder layer i and per token bucket M two graphs are produced:

  layer{i}_attn_in_m{M}       x[1,M,H] (uint8 | fp)  -> q, k, v          (fp)
  layer{i}_attn_out_ffn_m{M}  attn[1,M,Q] (fp), resid[1,M,H] (fp)
                              -> out[1,M,H] (fp)   (= resid + attn_out + ffn)

The FCs are int8 x int8 -> int32 (per-channel weight scale, per-tensor
activation scale) so they land on HMX; everything else (residual add,
RMSNorm, SiLU*up) stays float on the HTP vector unit. The weights are the
SAME int8 bytes nntrainer's per-channel Q8_0 path uses, so device output can
be verified against the CPU kernel (verify_htp.py).

Outputs (in --out):
  onnx/layer{i}_{graph}_m{M}.onnx      one ONNX per (layer, graph, bucket)
  weights/layer{i}.npz                 int8 weights + scales + act params
  npu_graphs.json                      runtime binding (dims, dtypes, quant)
  convert.sh                           qnn-onnx-converter -> model-lib ->
                                       context-binary commands (one .bin/layer)

Usage:
  # real model (HF safetensors + config.json)
  build_decoder_graphs.py --hf-dir /path/to/Qwen3-0.6B --out npu_qwen3 \
      --buckets 1,32,128 --calib calib.json --a16 ffn_act

  # synthetic (no model needed; what selftest.py uses)
  build_decoder_graphs.py --synthetic --out /tmp/npu_syn --buckets 4
"""

import argparse
import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from npu_quant import check_alignment  # noqa: E402
from reference import (QuantizedLayer, SafeTensors, load_calib,  # noqa: E402
                       load_hf_layer, quantize_layer, synthetic_layer)

# uint8 QDQ needs opset 13+; uint16 QuantizeLinear only exists from opset 21
# (ir_version 10). Keep the lowest opset the graph needs so older
# qnn-onnx-converter releases still accept the all-A8 graphs.
OPSET_A8, OPSET_A16 = 17, 21
FLOAT = TensorProto.FLOAT


# ----------------------------------------------------------------------------
# ONNX helpers
# ----------------------------------------------------------------------------

class GraphBuilder:
    def __init__(self, name, opset=OPSET_A8):
        self.name = name
        self.opset = opset
        self.nodes, self.inits, self.inputs, self.outputs = [], [], [], []
        self._n = 0

    def uid(self, base):
        self._n += 1
        return f"{base}_{self._n}"

    def const(self, name, arr):
        self.inits.append(numpy_helper.from_array(np.ascontiguousarray(arr),
                                                  name))
        return name

    def node(self, op, inputs, outputs=None, **attrs):
        if outputs is None:
            outputs = [self.uid(op.lower())]
        self.nodes.append(helper.make_node(op, inputs, outputs, **attrs))
        return outputs[0] if len(outputs) == 1 else outputs

    # -- QDQ primitives ------------------------------------------------------
    def q(self, x, scale, zp, bits, name=None):
        s = self.const(self.uid("s"), np.array(scale, np.float32))
        z = self.const(self.uid("zp"),
                       np.array(zp, np.uint8 if bits == 8 else np.uint16))
        out = [name] if name else None
        return self.node("QuantizeLinear", [x, s, z], out)

    def dq(self, xq, scale, zp, bits):
        s = self.const(self.uid("s"), np.array(scale, np.float32))
        z = self.const(self.uid("zp"),
                       np.array(zp, np.uint8 if bits == 8 else np.uint16))
        return self.node("DequantizeLinear", [xq, s, z])

    def fc(self, x, wq, ws, tag, bias=None):
        """x: float [1,M,K]; wq: int8 [N,K]; ws: fp32 [N]. Returns [1,M,N]."""
        w_init = self.const(f"{tag}_w", np.ascontiguousarray(wq.T))  # [K,N]
        s_init = self.const(f"{tag}_ws", ws.astype(np.float32))
        z_init = self.const(f"{tag}_wz", np.zeros(ws.shape[0], np.int8))
        wf = self.node("DequantizeLinear", [w_init, s_init, z_init], axis=1)
        y = self.node("MatMul", [x, wf])
        if bias is not None:
            b = self.const(f"{tag}_b", bias.astype(np.float32))
            y = self.node("Add", [y, b])
        return y

    def build(self):
        g = helper.make_graph(self.nodes, self.name, self.inputs, self.outputs,
                              initializer=self.inits)
        m = helper.make_model(g,
                              opset_imports=[helper.make_opsetid("", self.opset)])
        m.ir_version = 8 if self.opset <= 19 else 10
        onnx.checker.check_model(m)
        return m


def _io(name, shape, dtype):
    return helper.make_tensor_value_info(name, dtype, shape)


# ----------------------------------------------------------------------------
# Graph A: attn_in
# ----------------------------------------------------------------------------

def _opset_for(ql: QuantizedLayer, edges):
    return OPSET_A16 if any(ql.act[e][2] == 16 for e in edges) else OPSET_A8


def build_attn_in(ql: QuantizedLayer, M: int, a8_input: bool):
    gname = f"layer{ql.layer}_attn_in_m{M}"
    gb = GraphBuilder(gname, _opset_for(ql, ("attn_in",)))
    s, zp, bits = ql.act["attn_in"]
    H = ql.hidden

    if a8_input:
        # nntrainer quantizes in the RMSNorm epilogue (rms_norm out_quant) and
        # hands the uint8 buffer over -- no float transfer, no HTP quantize op.
        gb.inputs.append(_io("x", [1, M, H], TensorProto.UINT8))
        xq = "x"
    else:
        gb.inputs.append(_io("x", [1, M, H], FLOAT))
        xq = gb.q("x", s, zp, bits)
    xf = gb.dq(xq, s, zp, bits)
    y = gb.fc(xf, ql.w_qkv_q, ql.w_qkv_s, "qkv", ql.b_qkv)

    qu, kv = ql.q_units, ql.kv_units
    split = gb.const("qkv_split", np.array([qu, kv, kv], np.int64))
    gb.node("Split", [y, split], ["q", "k", "v"], axis=2)
    for n, w in (("q", qu), ("k", kv), ("v", kv)):
        gb.outputs.append(_io(n, [1, M, w], FLOAT))
    return gname, gb.build(), {
        "inputs": [("x", "UINT8" if a8_input else "FP", [1, 1, M, H],
                    (s, -zp) if a8_input else (1.0, 0))],
        "outputs": [("q", "FP", [1, 1, M, qu], (1.0, 0)),
                    ("k", "FP", [1, 1, M, kv], (1.0, 0)),
                    ("v", "FP", [1, 1, M, kv], (1.0, 0))],
    }


# ----------------------------------------------------------------------------
# Graph B: attn_out_ffn
# ----------------------------------------------------------------------------

def build_attn_out_ffn(ql: QuantizedLayer, M: int, a8_input: bool):
    gname = f"layer{ql.layer}_attn_out_ffn_m{M}"
    gb = GraphBuilder(gname, _opset_for(ql, ("attn", "ffn_in", "ffn_act")))
    H, Q = ql.hidden, ql.q_units

    s_a, zp_a, bits_a = ql.act["attn"]
    if a8_input:
        gb.inputs.append(_io("attn", [1, M, Q], TensorProto.UINT8))
        aq = "attn"
    else:
        gb.inputs.append(_io("attn", [1, M, Q], FLOAT))
        aq = gb.q("attn", s_a, zp_a, bits_a)
    gb.inputs.append(_io("resid", [1, M, H], FLOAT))

    af = gb.dq(aq, s_a, zp_a, bits_a)
    h1 = gb.fc(af, ql.wo_q, ql.wo_s, "wo")
    resid2 = gb.node("Add", ["resid", h1])

    # RMSNorm decomposed (portable to every converter version):
    #   n = x * rsqrt(mean(x^2) + eps) * gamma
    sq = gb.node("Mul", [resid2, resid2])
    if gb.opset >= 18:  # opset 18 moved ReduceMean's axes to an input
        axes = gb.const("rms_axes", np.array([-1], np.int64))
        ms = gb.node("ReduceMean", [sq, axes], keepdims=1)
    else:
        ms = gb.node("ReduceMean", [sq], axes=[-1], keepdims=1)
    eps = gb.const("eps", np.array(ql.eps, np.float32))
    ms_e = gb.node("Add", [ms, eps])
    rs = gb.node("Sqrt", [ms_e])
    inv = gb.node("Reciprocal", [rs])
    nx = gb.node("Mul", [resid2, inv])
    gamma = gb.const("ffn_norm_gamma", ql.ffn_norm_gamma.astype(np.float32))
    n = gb.node("Mul", [nx, gamma])

    s_n, zp_n, bits_n = ql.act["ffn_in"]
    nq = gb.q(n, s_n, zp_n, bits_n)
    nf = gb.dq(nq, s_n, zp_n, bits_n)
    gate = gb.fc(nf, ql.w_gate_q, ql.w_gate_s, "gate")
    up = gb.fc(nf, ql.w_up_q, ql.w_up_s, "up")
    sig = gb.node("Sigmoid", [gate])
    silu = gb.node("Mul", [gate, sig])
    act = gb.node("Mul", [silu, up])

    s_d, zp_d, bits_d = ql.act["ffn_act"]
    dq_ = gb.q(act, s_d, zp_d, bits_d)
    df = gb.dq(dq_, s_d, zp_d, bits_d)
    down = gb.fc(df, ql.w_down_q, ql.w_down_s, "down")
    gb.node("Add", [resid2, down], ["out"])
    gb.outputs.append(_io("out", [1, M, H], FLOAT))

    return gname, gb.build(), {
        "inputs": [("attn", "UINT8" if a8_input else "FP", [1, 1, M, Q],
                    (s_a, -zp_a) if a8_input else (1.0, 0)),
                   ("resid", "FP", [1, 1, M, H], (1.0, 0))],
        "outputs": [("out", "FP", [1, 1, M, H], (1.0, 0))],
    }


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def parse_buckets(s):
    b = sorted({int(x) for x in s.split(",") if x.strip()})
    assert b and b[0] >= 1
    return b


def parse_layers(s, n_layers):
    if s is None:
        return list(range(n_layers))
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [i for i in out if 0 <= i < n_layers]


def write_convert_sh(out_dir, layers, graphs_per_layer, fp_bits):
    lines = [
        "#!/bin/bash",
        "# Generated by build_decoder_graphs.py -- converts every ONNX graph",
        "# into one QNN context binary per decoder layer.",
        "# Requires QNN_SDK_ROOT (and ANDROID_NDK_ROOT for the model libs).",
        "set -e",
        'HERE="$(cd "$(dirname "$0")" && pwd)"',
        ': "${QNN_SDK_ROOT:?set QNN_SDK_ROOT}"',
        'CONV="$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-onnx-converter"',
        'LIBGEN="$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-model-lib-generator"',
        'BINGEN="$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator"',
        'BACKEND="$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so"',
        '# HTP arch of the target SoC (v75 = 8 Gen 3, v79 = 8 Elite). Override:',
        '#   HTP_ARCH=v79 ./convert.sh',
        'HTP_ARCH="${HTP_ARCH:-v75}"',
        'mkdir -p "$HERE/cpp" "$HERE/libs" "$HERE/bin"',
        "",
    ]
    for i in layers:
        names = graphs_per_layer[i]
        lines.append(f"# ---- layer {i} ----")
        for g in names:
            lines.append(
                f'"$CONV" --input_network "$HERE/onnx/{g}.onnx" '
                f'--output_path "$HERE/cpp/{g}.cpp" '
                f'--float_bitwidth {fp_bits} --preserve_io datatype '
                f'--quantization_overrides "$HERE/onnx/{g}.encodings.json" '
                f'2>&1 | tail -2')
            lines.append(
                f'"$LIBGEN" -c "$HERE/cpp/{g}.cpp" -b "$HERE/cpp/{g}.bin" '
                f'-o "$HERE/libs" -t x86_64-linux-clang 2>&1 | tail -1')
        models = ",".join(f'"$HERE/libs/x86_64-linux-clang/lib{g}.so"'
                          for g in names)
        lines.append(
            f'"$BINGEN" --backend "$BACKEND" --model {models} '
            f'--binary_file layer{i} --output_dir "$HERE/bin" '
            f'--config_file "$HERE/htp_backend_ext.json" 2>&1 | tail -1')
        lines.append("")
    lines.append('echo "context binaries in $HERE/bin (one per layer)"')
    with open(os.path.join(out_dir, "convert.sh"), "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(os.path.join(out_dir, "convert.sh"), 0o755)

    # HTP backend extension config: pick the SoC arch + performance mode.
    ext = {
        "backend_extensions": {
            "shared_library_path": "libQnnHtpNetRunExtensions.so",
            "config_file_path": "htp_config.json"}}
    cfg = {
        "graphs": [{"graph_names": [g for i in layers for g in
                                    graphs_per_layer[i]],
                    "vtcm_mb": 8, "O": 3, "fp16_relaxed_precision": 1}],
        "devices": [{"dsp_arch": "${HTP_ARCH}", "cores": [
            {"core_id": 0, "perf_profile": "burst", "rpc_control_latency": 100}]}],
        "context": {"enable_graphs": []}}
    with open(os.path.join(out_dir, "htp_backend_ext.json"), "w") as f:
        json.dump(ext, f, indent=2)
    with open(os.path.join(out_dir, "htp_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def write_encodings(path, ql: QuantizedLayer, io_meta, a8_input):
    """
    AIMET-style quantization_overrides so the converter uses OUR static
    activation params (no calibration data needed) and per-channel weights.
    Only the tensors we care about are listed; unlisted float tensors run in
    float (--float_bitwidth).
    """
    enc = {"version": "0.6.1", "activation_encodings": {},
           "param_encodings": {}}

    def act(name, edge):
        s, zp, bits = ql.act[edge]
        enc["activation_encodings"][name] = [{
            "bitwidth": bits, "dtype": "int", "is_symmetric": "False",
            "max": float((((1 << bits) - 1) - zp) * s), "min": float(-zp * s),
            "offset": -zp, "scale": float(s)}]

    if a8_input:
        for n, _, _, _ in io_meta["inputs"]:
            if n == "x":
                act("x", "attn_in")
            if n == "attn":
                act("attn", "attn")
    for tag, ws in (("qkv", ql.w_qkv_s), ("wo", ql.wo_s), ("gate", ql.w_gate_s),
                    ("up", ql.w_up_s), ("down", ql.w_down_s)):
        enc["param_encodings"][f"{tag}_w"] = [
            {"bitwidth": 8, "dtype": "int", "is_symmetric": "True",
             "max": float(127 * s), "min": float(-128 * s), "offset": -128,
             "scale": float(s)} for s in ws]
    with open(path, "w") as f:
        json.dump(enc, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-dir", help="HF model dir (config.json + safetensors)")
    ap.add_argument("--synthetic", action="store_true",
                    help="tiny random model instead of --hf-dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--buckets", default="1,32,128",
                    help="token-count buckets M (comma list)")
    ap.add_argument("--layers", default=None, help="e.g. 0-3,10")
    ap.add_argument("--calib", default=None, help="calibration ranges JSON")
    ap.add_argument("--a16", default="",
                    help="edges to keep in uint16 (comma list of "
                         "attn_in,attn,ffn_in,ffn_act); default all uint8")
    ap.add_argument("--fp-input", action="store_true",
                    help="graph inputs are float (HTP quantizes) instead of "
                         "uint8 written by nntrainer's rms_norm out_quant")
    ap.add_argument("--fp-bits", type=int, default=16, choices=(16, 32),
                    help="float bitwidth for the non-FC ops / float IO")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "onnx"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "weights"), exist_ok=True)
    buckets = parse_buckets(args.buckets)
    a16 = tuple(e for e in args.a16.split(",") if e)
    calib = load_calib(args.calib)
    a8_input = not args.fp_input

    if args.synthetic:
        cfg = {"hidden_size": 64, "num_attention_heads": 2,
               "num_key_value_heads": 1, "head_dim": 32,
               "intermediate_size": 96, "num_hidden_layers": 2,
               "rms_norm_eps": 1e-6}
        st = None
    else:
        assert args.hf_dir, "--hf-dir or --synthetic"
        with open(os.path.join(args.hf_dir, "config.json")) as f:
            cfg = json.load(f)
        st = SafeTensors(args.hf_dir)

    n_layers = int(cfg["num_hidden_layers"])
    eps = float(cfg.get("rms_norm_eps", 1e-6))
    layers = parse_layers(args.layers, n_layers)

    hidden = int(cfg["hidden_size"])
    heads = int(cfg["num_attention_heads"])
    kv_heads = int(cfg.get("num_key_value_heads", heads))
    head_dim = int(cfg.get("head_dim", hidden // heads))
    check_alignment("hidden_size", hidden)
    check_alignment("q_units", heads * head_dim)
    check_alignment("kv_units", kv_heads * head_dim)
    check_alignment("intermediate_size", int(cfg["intermediate_size"]))

    manifest = {
        "version": 1,
        "model": {"hidden": hidden, "q_units": heads * head_dim,
                  "kv_units": kv_heads * head_dim,
                  "inter": int(cfg["intermediate_size"]), "eps": eps},
        "buckets": buckets,
        "a8_input": a8_input,
        "fp_bits": args.fp_bits,
        "layers": {},
    }
    graphs_per_layer = {}

    for i in layers:
        if st is None:
            p = synthetic_layer(hidden=hidden, q_units=heads * head_dim,
                                kv_units=kv_heads * head_dim,
                                inter=int(cfg["intermediate_size"]), eps=eps,
                                seed=args.seed + i, bias=(i % 2 == 1))
        else:
            p = load_hf_layer(st, i, eps)
        ql = quantize_layer(i, p, calib, a16_edges=a16)
        np.savez(os.path.join(args.out, "weights", f"layer{i}.npz"),
                 **ql.npz_dict())

        lyr = {"bin": f"bin/layer{i}.bin", "graphs": {}}
        names = []
        for M in buckets:
            # graph A's input comes from rms_norm (out_quant epilogue -> uint8
            # unless --fp-input); graph B's `attn` comes from mha_core, which
            # has no quantize epilogue, so it is always float and the HTP
            # quantizes it (Q op) in front of the o_proj FC.
            for builder, key, a8 in ((build_attn_in, "attn_in", a8_input),
                                     (build_attn_out_ffn, "attn_out_ffn",
                                      False)):
                gname, model, meta = builder(ql, M, a8)
                onnx.save(model, os.path.join(args.out, "onnx", gname + ".onnx"))
                write_encodings(os.path.join(args.out, "onnx",
                                             gname + ".encodings.json"),
                                ql, meta, a8_input)
                names.append(gname)
                lyr["graphs"].setdefault(key, {})[str(M)] = {
                    "name": gname,
                    "inputs": [{"name": n, "dtype": d, "dim": dim,
                                "scale": float(sc), "offset": int(off)}
                               for n, d, dim, (sc, off) in meta["inputs"]],
                    "outputs": [{"name": n, "dtype": d, "dim": dim,
                                 "scale": float(sc), "offset": int(off)}
                                for n, d, dim, (sc, off) in meta["outputs"]],
                }
        lyr["act"] = {k: {"scale": v[0], "zero_point": v[1], "bits": v[2]}
                      for k, v in ql.act.items()}
        manifest["layers"][str(i)] = lyr
        graphs_per_layer[i] = names
        print(f"layer {i}: {len(names)} graphs, "
              f"act={ {k: (round(v[0], 6), v[1], v[2]) for k, v in ql.act.items()} }")

    with open(os.path.join(args.out, "npu_graphs.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    write_convert_sh(args.out, layers, graphs_per_layer, args.fp_bits)
    print(f"wrote {args.out}/npu_graphs.json and convert.sh")


if __name__ == "__main__":
    main()
