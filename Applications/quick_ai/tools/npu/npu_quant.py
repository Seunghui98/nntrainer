#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
"""
Quantization primitives shared by the NPU (Hexagon HTP / HMX) offload tools.

The weight quantizer reproduces nntrainer's per-channel Q8_0 scheme
(``__ggml_quantize_q8_0_per_channel`` in ggml_interface.cpp) bit for bit:

* one scale per output channel, ``d = amax(row) / 127``
* the scale is rounded through FP16 (that is the value the runtime sees)
* ``q = round_half_away_from_zero(x / d_fp16)``, saturated to [-128, 127]

Because the CPU fallback (``nntr_gemm_q8ch_*``) and the HTP graph then share
the *same* int8 weight bytes and both accumulate in int32, a per-layer HTP
output can be compared against the CPU kernel exactly (see verify_htp.py).

Activations use asymmetric uint8/uint16 with a static (calibrated) scale and
zero point, which is what QNN's ``scaleOffsetEncoding`` expects:

    real = (q + offset) * scale        (QNN convention, offset = -zero_point)
    q    = clip(round(real / scale) + zero_point, 0, qmax)
"""

import numpy as np


def round_half_away(v: np.ndarray) -> np.ndarray:
    """std::round semantics (ties away from zero); np.rint is ties-to-even."""
    return np.sign(v) * np.floor(np.abs(v) + 0.5)


def fp16_roundtrip(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16).astype(np.float32)


def quantize_weight_per_channel(w: np.ndarray):
    """
    w: [N, K] float32 (N = output channels).
    Returns (q int8 [N, K], scale float32 [N]) matching the C++ quantizer.
    """
    w = np.asarray(w, dtype=np.float32)
    assert w.ndim == 2, "weight must be [out, in]"
    amax = np.abs(w).max(axis=1)
    d = np.where(amax > 0.0, amax / 127.0, 1.0).astype(np.float32)
    d_used = fp16_roundtrip(d)  # exact runtime value (block_q8_0.d is fp16)
    inv = np.where(d_used > 0.0, 1.0 / d_used, 0.0).astype(np.float32)
    q = round_half_away(w * inv[:, None])
    q = np.clip(q, -128, 127).astype(np.int8)
    return q, d_used


def act_qparams(lo: float, hi: float, bits: int = 8):
    """
    Asymmetric quantization parameters for a calibrated range [lo, hi].
    Returns (scale, zero_point). QNN offset is -zero_point.
    The range is widened to include 0 so that padding / residual zeros are
    exactly representable (the same reason the conv path passes pad_q).
    """
    lo = float(min(lo, 0.0))
    hi = float(max(hi, 0.0))
    qmax = (1 << bits) - 1
    scale = (hi - lo) / qmax if hi > lo else 1.0
    zp = int(round(-lo / scale))
    zp = max(0, min(qmax, zp))
    return np.float32(scale), zp


def quantize_act(x: np.ndarray, scale: float, zp: int, bits: int = 8):
    qmax = (1 << bits) - 1
    q = round_half_away(np.asarray(x, np.float32) / np.float32(scale)) + zp
    q = np.clip(q, 0, qmax)
    return q.astype(np.uint8 if bits == 8 else np.uint16)


def dequantize_act(q: np.ndarray, scale: float, zp: int) -> np.ndarray:
    return (q.astype(np.int32) - zp).astype(np.float32) * np.float32(scale)


def fake_quant_act(x: np.ndarray, scale: float, zp: int, bits: int = 8):
    return dequantize_act(quantize_act(x, scale, zp, bits), scale, zp)


def fc_int32(xq: np.ndarray, zp: int, wq: np.ndarray) -> np.ndarray:
    """
    Exact integer GEMM as HMX / nntr_gemm_q8ch compute it:
      acc[m, n] = sum_k (xq[m, k] - zp) * wq[n, k]      (int32)
    xq: [M, K] uint8/uint16, wq: [N, K] int8.
    """
    xi = xq.astype(np.int32) - np.int32(zp)
    return xi @ wq.astype(np.int32).T


def fc_dequant(acc: np.ndarray, x_scale: float, w_scale: np.ndarray):
    """y[m, n] = acc[m, n] * x_scale * w_scale[n] -- one FP32 multiply at the
    end, exactly the per-channel kernel's epilogue."""
    return acc.astype(np.float32) * (np.float32(x_scale) * w_scale)[None, :]


def check_alignment(name: str, n: int, mult: int = 32) -> None:
    if n % mult != 0:
        print(f"[npu_quant] warning: {name}={n} is not a multiple of {mult}; "
              f"HMX tiles are {mult}-wide, expect padding inside the HTP graph")
