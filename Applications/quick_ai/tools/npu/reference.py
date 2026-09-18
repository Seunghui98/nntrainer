#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
"""
NumPy reference of the two NPU decoder graphs, with the exact same
quantization points the ONNX/QNN graphs use. Used by selftest.py (against
onnxruntime) and verify_htp.py (against device dumps).

Graph A  ``attn_in``      : x(uint8) -> FC(qkv fused, int8 per-channel) -> q,k,v
Graph B  ``attn_out_ffn`` : attn, resid -> FC(wo) -> +resid -> RMSNorm(gamma)
                            -> Q -> FC(gate), FC(up) -> SiLU*up -> Q -> FC(down)
                            -> +resid -> out
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from npu_quant import (act_qparams, fc_dequant, fc_int32,
                       quantize_act, quantize_weight_per_channel)

# ----------------------------------------------------------------------------
# Calibration ranges
# ----------------------------------------------------------------------------

# Edge names (what the calibration JSON keys are) and a conservative default
# range used when no calibration is available. Real models MUST be calibrated;
# these defaults only make the pipeline runnable.
EDGE_DEFAULTS = {
    "attn_in": (-8.0, 8.0),   # attention_norm output (graph A input)
    "attn": (-8.0, 8.0),      # mha_core output (graph B input)
    "ffn_in": (-8.0, 8.0),    # ffn_norm output (inside graph B)
    "ffn_act": (-16.0, 16.0), # SiLU(gate)*up, down_proj input (inside graph B)
}


def load_calib(path: Optional[str]) -> dict:
    if not path:
        return {"default": {}, "layers": {}}
    with open(path) as f:
        c = json.load(f)
    c.setdefault("default", {})
    c.setdefault("layers", {})
    return c


def edge_range(calib: dict, layer: int, edge: str):
    layers = calib.get("layers", {})
    if str(layer) in layers and edge in layers[str(layer)]:
        lo, hi = layers[str(layer)][edge]
    elif edge in calib.get("default", {}):
        lo, hi = calib["default"][edge]
    else:
        lo, hi = EDGE_DEFAULTS[edge]
    return float(lo), float(hi)


# ----------------------------------------------------------------------------
# Per-layer parameters
# ----------------------------------------------------------------------------

@dataclass
class LayerParams:
    """FP32 weights of one decoder layer, HF orientation [out, in]."""
    wq: np.ndarray
    wk: np.ndarray
    wv: np.ndarray
    wo: np.ndarray
    w_gate: np.ndarray
    w_up: np.ndarray
    w_down: np.ndarray
    ffn_norm_gamma: np.ndarray
    bq: Optional[np.ndarray] = None
    bk: Optional[np.ndarray] = None
    bv: Optional[np.ndarray] = None
    eps: float = 1e-6


@dataclass
class QuantizedLayer:
    """Everything the ONNX builder / reference needs, already quantized."""
    layer: int
    q_units: int
    kv_units: int
    hidden: int
    inter: int
    eps: float
    # weights: int8 [N, K] + per-channel fp32 scale [N]
    w_qkv_q: np.ndarray
    w_qkv_s: np.ndarray
    wo_q: np.ndarray
    wo_s: np.ndarray
    w_gate_q: np.ndarray
    w_gate_s: np.ndarray
    w_up_q: np.ndarray
    w_up_s: np.ndarray
    w_down_q: np.ndarray
    w_down_s: np.ndarray
    ffn_norm_gamma: np.ndarray
    b_qkv: Optional[np.ndarray]
    # activation quant params: edge -> (scale, zp, bits)
    act: Dict[str, tuple] = field(default_factory=dict)

    def npz_dict(self) -> dict:
        d = {
            "w_qkv_q": self.w_qkv_q, "w_qkv_s": self.w_qkv_s,
            "wo_q": self.wo_q, "wo_s": self.wo_s,
            "w_gate_q": self.w_gate_q, "w_gate_s": self.w_gate_s,
            "w_up_q": self.w_up_q, "w_up_s": self.w_up_s,
            "w_down_q": self.w_down_q, "w_down_s": self.w_down_s,
            "ffn_norm_gamma": self.ffn_norm_gamma,
            "meta": np.array([self.layer, self.q_units, self.kv_units,
                              self.hidden, self.inter], dtype=np.int64),
            "eps": np.array([self.eps], dtype=np.float32),
        }
        if self.b_qkv is not None:
            d["b_qkv"] = self.b_qkv
        for k, (s, zp, bits) in self.act.items():
            d[f"act_{k}"] = np.array([s, zp, bits], dtype=np.float64)
        return d


def quantize_layer(layer: int, p: LayerParams, calib: dict,
                   a16_edges=()) -> QuantizedLayer:
    q_units, hidden = p.wq.shape
    kv_units = p.wk.shape[0]
    inter = p.w_gate.shape[0]

    w_qkv = np.concatenate([p.wq, p.wk, p.wv], axis=0).astype(np.float32)
    b_qkv = None
    if p.bq is not None:
        b_qkv = np.concatenate([p.bq, p.bk, p.bv]).astype(np.float32)

    def qw(w):
        return quantize_weight_per_channel(np.asarray(w, np.float32))

    w_qkv_q, w_qkv_s = qw(w_qkv)
    wo_q, wo_s = qw(p.wo)
    wg_q, wg_s = qw(p.w_gate)
    wu_q, wu_s = qw(p.w_up)
    wd_q, wd_s = qw(p.w_down)

    act = {}
    for edge in EDGE_DEFAULTS:
        bits = 16 if edge in a16_edges else 8
        lo, hi = edge_range(calib, layer, edge)
        s, zp = act_qparams(lo, hi, bits)
        act[edge] = (float(s), int(zp), bits)

    return QuantizedLayer(
        layer=layer, q_units=q_units, kv_units=kv_units, hidden=hidden,
        inter=inter, eps=p.eps,
        w_qkv_q=w_qkv_q, w_qkv_s=w_qkv_s, wo_q=wo_q, wo_s=wo_s,
        w_gate_q=wg_q, w_gate_s=wg_s, w_up_q=wu_q, w_up_s=wu_s,
        w_down_q=wd_q, w_down_s=wd_s,
        ffn_norm_gamma=np.asarray(p.ffn_norm_gamma, np.float32),
        b_qkv=b_qkv, act=act)


# ----------------------------------------------------------------------------
# Reference forward passes
# ----------------------------------------------------------------------------

def silu(x):
    return x / (1.0 + np.exp(-x))


def rms_norm(x, gamma, eps):
    ms = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(ms + eps)) * gamma


def ref_attn_in_exact(ql: QuantizedLayer, x_q: np.ndarray):
    """
    Graph A with exact integer accumulation (what HMX does).
    x_q: [M, hidden] uint8 already quantized with ql.act['attn_in'].
    Returns (q, k, v) float32.
    """
    s, zp, _ = ql.act["attn_in"]
    acc = fc_int32(x_q, zp, ql.w_qkv_q)
    y = fc_dequant(acc, s, ql.w_qkv_s)
    if ql.b_qkv is not None:
        y = y + ql.b_qkv[None, :]
    qu, kv = ql.q_units, ql.kv_units
    return y[:, :qu], y[:, qu:qu + kv], y[:, qu + kv:]


def ref_attn_in(ql: QuantizedLayer, x: np.ndarray):
    """Graph A from a float input: quantize (static params) then exact FC."""
    s, zp, bits = ql.act["attn_in"]
    return ref_attn_in_exact(ql, quantize_act(x, s, zp, bits))


def ref_attn_out_ffn(ql: QuantizedLayer, attn: np.ndarray, resid: np.ndarray):
    """
    Graph B. Element-wise math is float32 (HTP runs it in FP16; tolerance is
    the caller's business); the three FCs are exact int32.
    """
    s_a, zp_a, bits_a = ql.act["attn"]
    a_q = quantize_act(attn, s_a, zp_a, bits_a)
    h1 = fc_dequant(fc_int32(a_q, zp_a, ql.wo_q), s_a, ql.wo_s)
    resid2 = resid.astype(np.float32) + h1

    n = rms_norm(resid2, ql.ffn_norm_gamma, ql.eps)
    s_n, zp_n, bits_n = ql.act["ffn_in"]
    n_q = quantize_act(n, s_n, zp_n, bits_n)
    gate = fc_dequant(fc_int32(n_q, zp_n, ql.w_gate_q), s_n, ql.w_gate_s)
    up = fc_dequant(fc_int32(n_q, zp_n, ql.w_up_q), s_n, ql.w_up_s)
    act = silu(gate) * up

    s_d, zp_d, bits_d = ql.act["ffn_act"]
    act_q = quantize_act(act, s_d, zp_d, bits_d)
    down = fc_dequant(fc_int32(act_q, zp_d, ql.w_down_q), s_d, ql.w_down_s)
    return resid2 + down


# ----------------------------------------------------------------------------
# HF safetensors loading (minimal, no external dependency)
# ----------------------------------------------------------------------------

_ST_DTYPES = {
    "F32": (np.float32, 4), "F16": (np.float16, 2), "BF16": (np.uint16, 2),
    "I8": (np.int8, 1), "U8": (np.uint8, 1), "I32": (np.int32, 4),
    "I64": (np.int64, 8),
}


class SafeTensors:
    """Lazy reader for one or more .safetensors files (HF index supported)."""

    def __init__(self, path: str):
        self.files = []
        if os.path.isdir(path):
            idx = os.path.join(path, "model.safetensors.index.json")
            if os.path.exists(idx):
                with open(idx) as f:
                    names = sorted(set(json.load(f)["weight_map"].values()))
                self.files = [os.path.join(path, n) for n in names]
            else:
                self.files = sorted(
                    os.path.join(path, n) for n in os.listdir(path)
                    if n.endswith(".safetensors"))
        else:
            self.files = [path]
        self.index = {}
        for fp in self.files:
            with open(fp, "rb") as f:
                hlen = int.from_bytes(f.read(8), "little")
                header = json.loads(f.read(hlen))
            for k, v in header.items():
                if k == "__metadata__":
                    continue
                self.index[k] = (fp, 8 + hlen, v)

    def keys(self):
        return self.index.keys()

    def __contains__(self, k):
        return k in self.index

    def get(self, key: str) -> np.ndarray:
        fp, base, info = self.index[key]
        dt, _ = _ST_DTYPES[info["dtype"]]
        start, end = info["data_offsets"]
        with open(fp, "rb") as f:
            f.seek(base + start)
            raw = f.read(end - start)
        arr = np.frombuffer(raw, dtype=dt).reshape(info["shape"])
        if info["dtype"] == "BF16":
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        return np.ascontiguousarray(arr.astype(np.float32))


def load_hf_layer(st: SafeTensors, layer: int, eps: float,
                  prefix: str = "model.layers.") -> LayerParams:
    p = f"{prefix}{layer}."

    def opt(k):
        return st.get(k) if k in st else None

    return LayerParams(
        wq=st.get(p + "self_attn.q_proj.weight"),
        wk=st.get(p + "self_attn.k_proj.weight"),
        wv=st.get(p + "self_attn.v_proj.weight"),
        wo=st.get(p + "self_attn.o_proj.weight"),
        w_gate=st.get(p + "mlp.gate_proj.weight"),
        w_up=st.get(p + "mlp.up_proj.weight"),
        w_down=st.get(p + "mlp.down_proj.weight"),
        ffn_norm_gamma=st.get(p + "post_attention_layernorm.weight"),
        bq=opt(p + "self_attn.q_proj.bias"),
        bk=opt(p + "self_attn.k_proj.bias"),
        bv=opt(p + "self_attn.v_proj.bias"),
        eps=eps,
    )


def synthetic_layer(hidden=64, q_units=64, kv_units=32, inter=96,
                    eps=1e-6, seed=0, bias=False) -> LayerParams:
    rng = np.random.default_rng(seed)

    def w(n, k, scale=0.05):
        return (rng.standard_normal((n, k)) * scale).astype(np.float32)

    return LayerParams(
        wq=w(q_units, hidden), wk=w(kv_units, hidden), wv=w(kv_units, hidden),
        wo=w(hidden, q_units), w_gate=w(inter, hidden), w_up=w(inter, hidden),
        w_down=w(hidden, inter),
        ffn_norm_gamma=(1.0 + 0.1 * rng.standard_normal(hidden)).astype(
            np.float32),
        bq=w(q_units, 1)[:, 0] if bias else None,
        bk=w(kv_units, 1)[:, 0] if bias else None,
        bv=w(kv_units, 1)[:, 0] if bias else None,
        eps=eps)
