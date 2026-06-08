## @file  gguf_to_nntrainer.py
## @brief Convert a native-`gemma4` (Gemma 3n E2B) Q4_0 GGUF file into an
##        nntrainer **safetensors** weight file that the CausalLM runtime
##        (Gemma4CausalLM) loads as a quantized model on x86.
##
## Reference implementation reused from
##   Applications/CausalLM/res/qwen3/qwen3-0.6b/gguf_to_nntrainer.py
##   - GGUF reader, repack_q4_0 (q4_0x8 / q4_0x4), quantize_q4_0,
##     quantize_q6_k, dequant of Q4_0/Q4_1/Q8_0/Q4_K/Q5_K/Q6_K.
## Unlike the qwen3 script (which writes a flat .bin), this writes a
## **nntrainer safetensors** container so the loader matches tensors by NAME.
##
## Quant / layout policy (verified against the nntrainer source):
##   * fully_connected (FC) weights  -> Q4_0 repacked q4_0x8 (x86).
##       nntr does NOT repack at load; the Q4_0 matmul kernel expects x8.
##   * embedding0 (TieWordEmbedding) -> Q6_K, **plain** (non-repacked) row
##       layout. tie_word_embedding.cpp dequantizes each row with
##       dequantize_row_q6_K at stride 210*blocks_per_row -> plain layout.
##   * norms / 1-D / scalar           -> F32 (graph uses packed=false, so the
##       weight dtype follows the FP32 activation type).
##   * per_layer_input_embedding (regular embedding_layer, FC_LAYER_DTYPE=Q4_0)
##       and the small per-layer FCs (inp_gate/proj, per_layer_input_projection)
##       -> Q4_0 repacked q4_0x8 like every other FC.
##
## Matching contract (neuralnet.cpp MODEL_FORMAT_SAFETENSORS load):
##   loader reads ONLY data_offsets from the header and matches by
##   weight->getName(). dtype/shape labels in the header are ignored on load.
##   The raw bytes are read directly into each weight's memory, so the bytes
##   we emit MUST equal that weight's getMemoryBytes() in the resolved dtype.
##   Weight names follow nntr's "<layer_name>:<suffix>" convention:
##     fully_connected      -> ":weight"
##     embedding/tie embed  -> ":Embedding"
##     rms_norm / reshaped  -> ":gamma"
##     scalar_multiply      -> ":scalar_multiplier"
##
## ---------------------------------------------------------------------------
## --fp32 mode (compute-precision isolation experiment):
##   When --fp32 is passed, EVERY mapped tensor is dequantized to fp32 and
##   written as a plain row-major F32 safetensors tensor (dtype "F32"), using
##   the SAME nntr weight names/orientation as the quantized path -- NO q4_0x8
##   repack and NO Q6_K. The values are still Q4_0-precision (dequantized from
##   the Q4_0 GGUF), so the model's logit gap is unchanged; only the COMPUTE
##   precision changes (fp32 kernel vs Q4_0 kernel on identical weight values).
##   Pair with an nntr config whose fc_layer_dtype/embedding_dtype/lmhead_dtype
##   are all "FP32" and model_tensor_type "FP32-FP32" so the graph resolves
##   every weight to plain fp32 (getMemoryBytes() == numel*4).
##
## @author Claude (for shsh1004/nntrainer, branch add/DDTree)

import argparse
import json
import os
import struct
import sys

import numpy as np

# Set by convert() from args.fp32. When True, every byte-producer emits plain
# row-major fp32 (dtype "F32") instead of its quantized/repacked layout.
FP32_MODE = False


# ---------------------------------------------------------------------------
# GGUF file format
# ---------------------------------------------------------------------------
GGUF_MAGIC = 0x46554747  # "GGUF"

GGUF_U8, GGUF_I8 = 0, 1
GGUF_U16, GGUF_I16 = 2, 3
GGUF_U32, GGUF_I32 = 4, 5
GGUF_F32 = 6
GGUF_BOOL = 7
GGUF_STR = 8
GGUF_ARR = 9
GGUF_U64, GGUF_I64 = 10, 11
GGUF_F64 = 12

GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_0 = 2
GGML_Q4_1 = 3
GGML_Q8_0 = 8
GGML_Q4_K = 12
GGML_Q5_K = 13
GGML_Q6_K = 14
GGML_BF16 = 30

GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 30: "BF16",
}

QK4_0 = 32
Q4_0_BLOCK_BYTES = 18
Q4_1_BLOCK_BYTES = 20
Q8_0_BLOCK_BYTES = 34
QK_K = 256
Q4_K_BLOCK_BYTES = 144
Q5_K_BLOCK_BYTES = 176
Q6_K_BLOCK_BYTES = 210


class GGUFReader:
    """Minimal GGUF v2 / v3 reader (numpy-only)."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        self._read_header()
        self._read_metadata()
        self._read_tensor_info()

    def close(self):
        self.f.close()

    def _u32(self):
        return struct.unpack("<I", self.f.read(4))[0]

    def _u64(self):
        return struct.unpack("<Q", self.f.read(8))[0]

    def _read_string(self):
        n = self._u64()
        return self.f.read(n).decode("utf-8", errors="replace")

    def _read_value(self, typ):
        if typ == GGUF_U8:   return struct.unpack("<B", self.f.read(1))[0]
        if typ == GGUF_I8:   return struct.unpack("<b", self.f.read(1))[0]
        if typ == GGUF_U16:  return struct.unpack("<H", self.f.read(2))[0]
        if typ == GGUF_I16:  return struct.unpack("<h", self.f.read(2))[0]
        if typ == GGUF_U32:  return struct.unpack("<I", self.f.read(4))[0]
        if typ == GGUF_I32:  return struct.unpack("<i", self.f.read(4))[0]
        if typ == GGUF_F32:  return struct.unpack("<f", self.f.read(4))[0]
        if typ == GGUF_BOOL: return struct.unpack("<B", self.f.read(1))[0] != 0
        if typ == GGUF_STR:  return self._read_string()
        if typ == GGUF_ARR:
            arr_typ = self._u32()
            n = self._u64()
            return [self._read_value(arr_typ) for _ in range(n)]
        if typ == GGUF_U64:  return struct.unpack("<Q", self.f.read(8))[0]
        if typ == GGUF_I64:  return struct.unpack("<q", self.f.read(8))[0]
        if typ == GGUF_F64:  return struct.unpack("<d", self.f.read(8))[0]
        raise ValueError("unknown GGUF metadata type: %d" % typ)

    def _read_header(self):
        magic = self._u32()
        if magic != GGUF_MAGIC:
            raise ValueError("not a GGUF file (magic=0x%08x): %s" %
                             (magic, self.path))
        self.version = self._u32()
        self.n_tensors = self._u64()
        self.n_kv = self._u64()

    def _read_metadata(self):
        self.metadata = {}
        for _ in range(self.n_kv):
            key = self._read_string()
            typ = self._u32()
            self.metadata[key] = self._read_value(typ)

    def _read_tensor_info(self):
        self.tensors = {}
        for _ in range(self.n_tensors):
            name = self._read_string()
            n_dim = self._u32()
            dims = [self._u64() for _ in range(n_dim)]
            shape = tuple(reversed(dims))  # numpy order: outer first
            typ = self._u32()
            offset = self._u64()
            self.tensors[name] = {"shape": shape, "type": typ, "offset": offset}
        alignment = self.metadata.get("general.alignment", 32)
        pos = self.f.tell()
        pad = (alignment - (pos % alignment)) % alignment
        self.data_start = pos + pad

    def tensor_bytes_size(self, info):
        numel = 1
        for d in info["shape"]:
            numel *= d
        t = info["type"]
        if t == GGML_F32:   return numel * 4
        if t == GGML_F16:   return numel * 2
        if t == GGML_BF16:  return numel * 2
        if t == GGML_Q4_0:  return (numel // QK4_0) * Q4_0_BLOCK_BYTES
        if t == GGML_Q4_1:  return (numel // QK4_0) * Q4_1_BLOCK_BYTES
        if t == GGML_Q8_0:  return (numel // 32) * Q8_0_BLOCK_BYTES
        if t == GGML_Q4_K:  return (numel // QK_K) * Q4_K_BLOCK_BYTES
        if t == GGML_Q5_K:  return (numel // QK_K) * Q5_K_BLOCK_BYTES
        if t == GGML_Q6_K:  return (numel // QK_K) * Q6_K_BLOCK_BYTES
        raise ValueError("unsupported GGML type %d (%s)" %
                         (t, GGML_TYPE_NAMES.get(t, "?")))

    def read_tensor_raw(self, name):
        info = self.tensors[name]
        size = self.tensor_bytes_size(info)
        self.f.seek(self.data_start + info["offset"])
        return self.f.read(size), info


# ---------------------------------------------------------------------------
# Dequant helpers
# ---------------------------------------------------------------------------
def dequant_q4_0(buf, numel):
    nb = numel // QK4_0
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(nb, Q4_0_BLOCK_BYTES)
    d = arr[:, :2].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
    q = arr[:, 2:]
    low = (q & 0x0F).astype(np.int32) - 8
    high = (q >> 4).astype(np.int32) - 8
    out = np.concatenate([low, high], axis=1).astype(np.float32) * d
    return out.reshape(numel)


def dequant_q4_1(buf, numel):
    nb = numel // QK4_0
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(nb, Q4_1_BLOCK_BYTES)
    d = arr[:, :2].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
    m = arr[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
    q = arr[:, 4:]
    low = (q & 0x0F).astype(np.float32)
    high = (q >> 4).astype(np.float32)
    out = np.concatenate([low, high], axis=1) * d + m
    return out.reshape(numel)


def dequant_q8_0(buf, numel):
    nb = numel // 32
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(nb, Q8_0_BLOCK_BYTES)
    d = arr[:, :2].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
    q = arr[:, 2:].copy().view(np.int8).reshape(nb, 32).astype(np.float32)
    return (q * d).reshape(numel)


def _unpack_k4_scales(scales):
    """Vectorized ggml get_scale_min_k4 for all nb blocks at once.
    scales: (nb, 12) uint8. Returns (sc, mn): each (nb, 8) uint8 — the
    per-sub-block scale and min for the 8 sub-blocks j=0..7."""
    s = scales.astype(np.int32)
    sc = np.empty((s.shape[0], 8), dtype=np.int32)
    mn = np.empty((s.shape[0], 8), dtype=np.int32)
    for j in range(8):
        if j < 4:
            sc[:, j] = s[:, j] & 63
            mn[:, j] = s[:, j + 4] & 63
        else:
            sc[:, j] = (s[:, j + 4] & 0xF) | ((s[:, j - 4] >> 6) << 4)
            mn[:, j] = (s[:, j + 4] >> 4) | ((s[:, j] >> 6) << 4)
    return sc, mn


def dequant_q4_k(buf, numel):
    """Vectorized port of ggml dequantize_row_q4_K. block_q4_K = 144 bytes:
    d(f16) dmin(f16) scales[12] qs[128]."""
    nb = numel // QK_K
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(nb, Q4_K_BLOCK_BYTES)
    d = arr[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
    dmin = arr[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
    scales = arr[:, 4:16]
    qs = arr[:, 16:144].astype(np.int32)  # (nb,128)
    sc, mn = _unpack_k4_scales(scales)    # (nb,8)
    out = np.empty((nb, QK_K), dtype=np.float32)
    for j in range(4):  # 4 groups of 64 -> uses sub-blocks 2j, 2j+1
        block = qs[:, j * 32:(j + 1) * 32]
        lo = (block & 0xF).astype(np.float32)
        hi = (block >> 4).astype(np.float32)
        d1 = (d * sc[:, 2 * j:2 * j + 1].astype(np.float32))
        m1 = (dmin * mn[:, 2 * j:2 * j + 1].astype(np.float32))
        d2 = (d * sc[:, 2 * j + 1:2 * j + 2].astype(np.float32))
        m2 = (dmin * mn[:, 2 * j + 1:2 * j + 2].astype(np.float32))
        out[:, j * 64:j * 64 + 32] = d1 * lo - m1
        out[:, j * 64 + 32:j * 64 + 64] = d2 * hi - m2
    return out.reshape(numel)


def dequant_q5_k(buf, numel):
    """Vectorized port of ggml dequantize_row_q5_K. block_q5_K = 176 bytes:
    d(f16) dmin(f16) scales[12] qh[32] qs[128]."""
    nb = numel // QK_K
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(nb, Q5_K_BLOCK_BYTES)
    d = arr[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
    dmin = arr[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
    scales = arr[:, 4:16]
    qh = arr[:, 16:48].astype(np.int32)   # (nb,32)
    qs = arr[:, 48:176].astype(np.int32)  # (nb,128)
    sc, mn = _unpack_k4_scales(scales)
    out = np.empty((nb, QK_K), dtype=np.float32)
    for j in range(4):
        block = qs[:, j * 32:(j + 1) * 32]
        u1 = 1 << (2 * j)
        u2 = 2 << (2 * j)
        lo = (block & 0xF) + np.where((qh & u1) != 0, 16, 0)
        hi = (block >> 4) + np.where((qh & u2) != 0, 16, 0)
        d1 = (d * sc[:, 2 * j:2 * j + 1].astype(np.float32))
        m1 = (dmin * mn[:, 2 * j:2 * j + 1].astype(np.float32))
        d2 = (d * sc[:, 2 * j + 1:2 * j + 2].astype(np.float32))
        m2 = (dmin * mn[:, 2 * j + 1:2 * j + 2].astype(np.float32))
        out[:, j * 64:j * 64 + 32] = d1 * lo.astype(np.float32) - m1
        out[:, j * 64 + 32:j * 64 + 64] = d2 * hi.astype(np.float32) - m2
    return out.reshape(numel)


def dequant_q6_k(buf, numel):
    nb = numel // QK_K
    out = np.empty((nb, QK_K), dtype=np.float32)
    view = memoryview(buf)
    for b in range(nb):
        base = b * Q6_K_BLOCK_BYTES
        ql_all = np.frombuffer(view[base:base + 128], dtype=np.uint8)
        qh_all = np.frombuffer(view[base + 128:base + 192], dtype=np.uint8)
        sc_all = np.frombuffer(view[base + 192:base + 208], dtype=np.int8)
        d = np.frombuffer(
            view[base + 208:base + 210], dtype=np.float16).astype(np.float32)[0]
        y = np.empty(QK_K, dtype=np.float32)
        for half in range(2):
            ql = ql_all[half * 64:(half + 1) * 64]
            qh = qh_all[half * 32:(half + 1) * 32]
            sc = sc_all[half * 8:(half + 1) * 8]
            y_off = half * 128
            for l in range(32):
                is_ = l // 16
                q1 = int(ql[l + 0] & 0xF) | (((int(qh[l]) >> 0) & 3) << 4)
                q2 = int(ql[l + 32] & 0xF) | (((int(qh[l]) >> 2) & 3) << 4)
                q3 = int(ql[l + 0] >> 4) | (((int(qh[l]) >> 4) & 3) << 4)
                q4 = int(ql[l + 32] >> 4) | (((int(qh[l]) >> 6) & 3) << 4)
                y[y_off + l + 0] = d * sc[is_ + 0] * (q1 - 32)
                y[y_off + l + 32] = d * sc[is_ + 2] * (q2 - 32)
                y[y_off + l + 64] = d * sc[is_ + 4] * (q3 - 32)
                y[y_off + l + 96] = d * sc[is_ + 6] * (q4 - 32)
        out[b] = y
    return out.reshape(numel)


# ---------------------------------------------------------------------------
# Quant helpers
# ---------------------------------------------------------------------------
def quantize_q4_0(x):
    flat = x.reshape(-1, QK4_0).astype(np.float32)
    nb = flat.shape[0]
    amax_idx = np.argmax(np.abs(flat), axis=1)
    amax = flat[np.arange(nb), amax_idx]
    d = amax / -8.0
    id_ = np.where(d != 0.0, 1.0 / d, 0.0).astype(np.float32)
    q = np.clip(np.rint(flat * id_[:, None]) + 8.0, 0, 15).astype(np.uint8)
    low = q[:, :16]
    high = q[:, 16:]
    qs = (low | (high << 4)).astype(np.uint8)
    d_fp16 = d.astype(np.float16).view(np.uint16).reshape(nb, 1)
    d_bytes = d_fp16.view(np.uint8).reshape(nb, 2)
    blocks = np.concatenate([d_bytes, qs], axis=1).astype(np.uint8)
    return blocks.tobytes()


def quantize_q6_k(x):
    """Vectorized port of ggml quantize_row_q6_K_ref (no imatrix). Produces
    bytes byte-identical in layout to a GGUF block_q6_K."""
    flat = x.reshape(-1, QK_K).astype(np.float32)
    nb = flat.shape[0]
    sub = flat.reshape(nb, 16, 16)               # 16 sub-blocks of 16 elems
    amax_idx = np.argmax(np.abs(sub), axis=2)    # (nb,16)
    amax = np.take_along_axis(sub, amax_idx[:, :, None], axis=2)[:, :, 0]
    scales = np.where(amax == 0.0, 0.0, amax / -32.0).astype(np.float32)  # (nb,16)

    abs_scales = np.abs(scales)
    max_idx = np.argmax(abs_scales, axis=1)      # (nb,)
    max_scale = scales[np.arange(nb), max_idx]   # (nb,) signed scale at amax
    max_abs = abs_scales[np.arange(nb), max_idx]
    allzero = max_abs == 0.0

    iscale = np.where(allzero, 0.0, -128.0 / np.where(max_scale == 0.0, 1.0,
                                                      max_scale)).astype(np.float32)
    d_super = np.where(iscale == 0.0, 0.0, 1.0 / np.where(iscale == 0.0, 1.0,
                                                          iscale)).astype(np.float32)

    int8_scales = np.clip(np.rint(iscale[:, None] * scales), -128,
                          127).astype(np.int32)   # (nb,16)
    d_sub = d_super[:, None] * int8_scales.astype(np.float32)  # (nb,16)
    inv = np.where(d_sub == 0.0, 0.0, 1.0 / np.where(d_sub == 0.0, 1.0, d_sub))
    q = np.rint(sub * inv[:, :, None]).astype(np.int32) + 32
    q = np.clip(q, 0, 63).astype(np.uint8).reshape(nb, QK_K)  # L, row-major

    out = np.zeros((nb, Q6_K_BLOCK_BYTES), dtype=np.uint8)
    for half in range(2):
        y = half * 128
        qlo = half * 64
        qho = half * 32
        l = np.arange(32)
        q1 = q[:, y + l + 0].astype(np.int32)
        q2 = q[:, y + l + 32].astype(np.int32)
        q3 = q[:, y + l + 64].astype(np.int32)
        q4 = q[:, y + l + 96].astype(np.int32)
        out[:, qlo + l + 0] = ((q1 & 0xF) | ((q3 & 0xF) << 4)).astype(np.uint8)
        out[:, qlo + l + 32] = ((q2 & 0xF) | ((q4 & 0xF) << 4)).astype(np.uint8)
        out[:, 128 + qho + l] = (
            (((q1 >> 4) & 3) << 0) | (((q2 >> 4) & 3) << 2) |
            (((q3 >> 4) & 3) << 4) | (((q4 >> 4) & 3) << 6)).astype(np.uint8)
    out[:, 192:208] = int8_scales.astype(np.int8).view(np.uint8)
    out[:, 208:210] = d_super.astype(np.float16).view(np.uint8).reshape(nb, 2)
    # All-zero blocks -> all zero bytes.
    out[allzero] = 0
    return out.reshape(-1).tobytes()


# ---------------------------------------------------------------------------
# Q4_0 repack into q4_0x8 / q4_0x4 (matches nntrainer nntr_ggml_impl_*)
# ---------------------------------------------------------------------------
def repack_q4_0(raw_q4_0, N, K, interleave):
    assert interleave in (4, 8)
    assert N % interleave == 0, \
        "N=%d must be divisible by interleave %d" % (N, interleave)
    assert K % QK4_0 == 0
    nblocks = K // QK4_0
    src = np.frombuffer(raw_q4_0, dtype=np.uint8).reshape(
        N, nblocks, Q4_0_BLOCK_BYTES)
    d_all = src[:, :, :2]
    qs_all = src[:, :, 2:]

    if interleave == 8:
        out_super = np.empty((N // 8, nblocks, 16 + 128), dtype=np.uint8)
        for g in range(N // 8):
            rows = slice(g * 8, g * 8 + 8)
            d_chunk = d_all[rows].transpose(1, 0, 2).reshape(nblocks, 16)
            out_super[g, :, :16] = d_chunk
            qs_chunk = qs_all[rows]
            dst = np.empty((nblocks, 16, 8), dtype=np.uint8)
            for i in range(16):
                row = i % 8
                off = (i // 8) * 8
                dst[:, i, :] = qs_chunk[row, :, off:off + 8]
            dst = dst.reshape(nblocks, 128)
            dst ^= 0x88
            out_super[g, :, 16:] = dst
        return out_super.tobytes()
    else:
        out_super = np.empty((N // 4, nblocks, 8 + 64), dtype=np.uint8)
        for g in range(N // 4):
            rows = slice(g * 4, g * 4 + 4)
            d_chunk = d_all[rows].transpose(1, 0, 2).reshape(nblocks, 8)
            out_super[g, :, :8] = d_chunk
            qs_chunk = qs_all[rows]
            dst = np.empty((nblocks, 8, 8), dtype=np.uint8)
            for i in range(8):
                row = i % 4
                off = (i // 4) * 8
                dst[:, i, :] = qs_chunk[row, :, off:off + 8]
            dst = dst.reshape(nblocks, 64)
            dst ^= 0x88
            out_super[g, :, 8:] = dst
        return out_super.tobytes()


# ---------------------------------------------------------------------------
# GGUF tensor -> fp32 numpy (shape preserved, GGUF order = [out, in])
# ---------------------------------------------------------------------------
def gguf_to_fp32(reader, name):
    buf, info = reader.read_tensor_raw(name)
    t = info["type"]
    numel = 1
    for d in info["shape"]:
        numel *= d
    if t == GGML_F32:
        arr = np.frombuffer(buf, dtype=np.float32).copy()
    elif t == GGML_F16:
        arr = np.frombuffer(buf, dtype=np.float16).astype(np.float32)
    elif t == GGML_BF16:
        u16 = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32)
        arr = (u16 << 16).view(np.float32)
    elif t == GGML_Q4_0:
        arr = dequant_q4_0(buf, numel)
    elif t == GGML_Q4_1:
        arr = dequant_q4_1(buf, numel)
    elif t == GGML_Q8_0:
        arr = dequant_q8_0(buf, numel)
    elif t == GGML_Q4_K:
        arr = dequant_q4_k(buf, numel)
    elif t == GGML_Q5_K:
        arr = dequant_q5_k(buf, numel)
    elif t == GGML_Q6_K:
        arr = dequant_q6_k(buf, numel)
    else:
        raise ValueError("dequant for type %d (%s) not implemented (%s)" %
                         (t, GGML_TYPE_NAMES.get(t, "?"), name))
    return arr.reshape(info["shape"])


# ---------------------------------------------------------------------------
# Per-tensor byte producers. Each returns (dtype_label, shape, raw_bytes).
# shape is the nntr [b,c,h,w]-flavored shape stored in the header (informational
# only on load); we store the logical [out, in] / [vocab, hidden] / [len] shape.
# ---------------------------------------------------------------------------
def bytes_norm_f32(reader, name, expected_len):
    arr = gguf_to_fp32(reader, name).astype(np.float32).ravel()
    assert arr.size == expected_len, \
        "%s: len %d != expected %d" % (name, arr.size, expected_len)
    return "F32", [int(expected_len)], arr.tobytes()


def bytes_scalar_f32(reader, name):
    arr = gguf_to_fp32(reader, name).astype(np.float32).ravel()
    assert arr.size == 1, "%s: scalar must be 1 element, got %d" % (name, arr.size)
    return "F32", [1], arr.tobytes()


def bytes_fc_q4_0(reader, name, out_features, in_features, interleave):
    """FC weight -> Q4_0 repacked q4_0x8. GGUF stores [out, in]; the repack
    treats raw blocks as (N=out rows, K=in cols), which is exactly the layout
    the nntr Q4_0 matmul kernel expects."""
    if in_features % QK4_0 != 0:
        raise ValueError("%s: in_features=%d not divisible by %d" %
                         (name, in_features, QK4_0))
    if out_features % interleave != 0:
        raise ValueError("%s: out_features=%d not divisible by interleave %d" %
                         (name, out_features, interleave))
    if FP32_MODE:
        arr = gguf_to_fp32(reader, name).astype(np.float32)
        assert arr.shape == (out_features, in_features), \
            "%s shape %s != (%d,%d)" % (name, arr.shape, out_features, in_features)
        # The plain-FP32 nntr FC layer computes input @ weight with weight
        # shaped [in_features, out_features] (fc_layer.cpp weight_dim =
        # (in_dim.width, unit)). GGUF stores FC weights as [out, in], so the
        # FP32 weight must be the TRANSPOSE -> [in, out], row-major. (The Q4_0
        # path below keeps [out, in] because its repacked kernel consumes that
        # orientation directly.)
        arr_t = np.ascontiguousarray(arr.T)  # [in, out]
        return "F32", [int(in_features), int(out_features)], arr_t.tobytes()
    info = reader.tensors[name]
    t = info["type"]
    if t == GGML_Q4_0:
        raw, _ = reader.read_tensor_raw(name)
        expected = (out_features * in_features // QK4_0) * Q4_0_BLOCK_BYTES
        assert len(raw) == expected, \
            "%s: raw Q4_0 size %d != %d" % (name, len(raw), expected)
    else:
        print("  [requant] %s: %s -> Q4_0" % (name, GGML_TYPE_NAMES.get(t, "?")))
        arr = gguf_to_fp32(reader, name)
        assert arr.shape == (out_features, in_features), \
            "%s shape %s != (%d,%d)" % (name, arr.shape, out_features, in_features)
        raw = quantize_q4_0(arr)
    packed = repack_q4_0(raw, out_features, in_features, interleave)
    return "Q4_0", [int(out_features), int(in_features)], packed


def bytes_embedding_q6k(reader, name, vocab, hidden):
    """Embedding -> Q6_K, **plain** row layout (TieWordEmbedding dequantizes
    each row at stride 210*blocks_per_row)."""
    info = reader.tensors[name]
    if FP32_MODE:
        arr = gguf_to_fp32(reader, name).astype(np.float32)
        assert arr.shape == (vocab, hidden), \
            "%s shape %s != (%d,%d)" % (name, arr.shape, vocab, hidden)
        # plain row-major fp32, no Q6_K
        return "F32", [int(vocab), int(hidden)], \
            np.ascontiguousarray(arr).tobytes()
    if hidden % QK_K != 0:
        raise ValueError("%s: hidden=%d not divisible by QK_K=%d" %
                         (name, hidden, QK_K))
    if info["type"] == GGML_Q6_K:
        raw, _ = reader.read_tensor_raw(name)
        return "Q6_K", [int(vocab), int(hidden)], raw
    print("  [requant] %s: %s -> Q6_K" %
          (name, GGML_TYPE_NAMES.get(info["type"], "?")))
    arr = gguf_to_fp32(reader, name)
    assert arr.shape == (vocab, hidden), \
        "%s shape %s != (%d,%d)" % (name, arr.shape, vocab, hidden)
    return "Q6_K", [int(vocab), int(hidden)], quantize_q6_k(arr.reshape(-1, hidden))


# ---------------------------------------------------------------------------
# safetensors writer (matches neuralnet.cpp save / safetensors_util buildHeader)
# ---------------------------------------------------------------------------
def write_safetensors(entries, out_path):
    """entries: list of (name, dtype_label, shape_list, raw_bytes).
    Header = {"__metadata__":{"format":"nntrainer"}, <name>:{...}}, then
    8-byte LE header length, header (padded with spaces to 8), then raw data."""
    offset = 0
    header = {"__metadata__": {"format": "nntrainer"}}
    buffers = []
    for name, dtype, shape, raw in entries:
        n = len(raw)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + n],
        }
        buffers.append(raw)
        offset += n

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad

    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for buf in buffers:
            f.write(buf)
    print("\nWrote %s (header=%d B, data=%.1f MiB, tensors=%d)" %
          (out_path, len(header_bytes), offset / (1024 * 1024), len(entries)))


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def convert(args):
    global FP32_MODE
    FP32_MODE = bool(getattr(args, "fp32", False))
    reader = GGUFReader(args.gguf)
    md = reader.metadata
    interleave = 8 if args.target == "x86" else 4
    if FP32_MODE:
        print("FP32 MODE: every mapped tensor -> plain row-major F32 "
              "(no q4_0x8 repack, no Q6_K)\n")

    arch = md.get("general.architecture", "")
    print("GGUF architecture: %s" % arch)

    n_layers = int(md["gemma4.block_count"])
    hidden = int(md["gemma4.embedding_length"])
    n_heads = int(md["gemma4.attention.head_count"])
    # full-attention head dim = key_length (512); sliding head dim =
    # key_length_swa (256).
    head_dim = int(md.get("gemma4.attention.key_length", hidden // n_heads))
    sliding_kl = int(md.get("gemma4.attention.key_length_swa", 256))
    ple_dim = int(md["gemma4.embedding_length_per_layer_input"])
    n_kv_shared = int(md.get("gemma4.attention.shared_kv_layers", 0))

    vocab = reader.tensors["token_embd.weight"]["shape"][0]
    ple_vocab = reader.tensors["per_layer_token_embd.weight"]["shape"][0]
    ple_total = n_layers * ple_dim  # per_layer_total_dim in the nntr graph

    # sliding_window_pattern: list[bool], True => sliding_attention layer.
    layer_types = md.get("gemma4.attention.sliding_window_pattern", None)
    first_shared = n_layers - n_kv_shared

    print("config: layers=%d hidden=%d vocab=%d n_heads=%d head_dim=%d "
          "sliding_kl=%d ple_dim=%d ple_vocab=%d kv_shared=%d first_shared=%d" %
          (n_layers, hidden, vocab, n_heads, head_dim, sliding_kl, ple_dim,
           ple_vocab, n_kv_shared, first_shared))
    print("target=%s (Q4_0 interleave=%d)\n" % (args.target, interleave))

    entries = []  # (name, dtype, shape, bytes) IN GRAPH ORDER

    def add(name, triple):
        dtype, shape, raw = triple
        entries.append((name, dtype, shape, raw))

    def is_sliding(i):
        if layer_types is not None and i < len(layer_types):
            return bool(layer_types[i])
        return True

    def is_shared(i):
        return n_kv_shared > 0 and i >= first_shared

    def cur_head_dim(i):
        # full-attention layers use the global head dim (key_length); sliding
        # layers use the sliding key length (256).
        return sliding_kl if is_sliding(i) else head_dim

    # ---- top-level embeddings / projections (graph constructModel order) ----
    # embedding0 (TieWordEmbedding) <- token_embd, Q6_K plain.
    add("embedding0:Embedding",
        bytes_embedding_q6k(reader, "token_embd.weight", vocab, hidden))

    # per_layer_input_embedding (regular embedding_layer, EMBEDDING_DTYPE=Q6_K)
    # <- per_layer_token_embd [ple_vocab, ple_total]. Written Q6_K **canonical**
    # (plain row layout, same as embedding0); the EmbeddingLayer dequantizes
    # each row at stride 210*blocks_per_row. GGUF per_layer_token_embd is Q5_K,
    # so this requants Q5_K -> Q6_K canonical.
    add("per_layer_input_embedding:Embedding",
        bytes_embedding_q6k(reader, "per_layer_token_embd.weight",
                            ple_vocab, ple_total))

    # per_layer_input_projection (fully_connected, unit=ple_total, in=hidden)
    # <- per_layer_model_proj [ple_total, hidden].
    add("per_layer_input_projection:weight",
        bytes_fc_q4_0(reader, "per_layer_model_proj.weight",
                      ple_total, hidden, interleave))

    # per_layer_projection_norm (reshaped_rms_norm, feature=ple_dim, F32)
    # <- per_layer_proj_norm [ple_dim].
    add("per_layer_projection_norm:gamma",
        bytes_norm_f32(reader, "per_layer_proj_norm.weight", ple_dim))

    # ---- decoder blocks ----
    for i in range(n_layers):
        g = "blk.%d." % i
        L = "layer%d" % i
        hd = cur_head_dim(i)
        q_units = hd * n_heads
        shared = is_shared(i)

        # attention_norm <- attn_norm
        add("%s_attention_norm:gamma" % L,
            bytes_norm_f32(reader, g + "attn_norm.weight", hidden))

        # wq always present
        add("%s_wq:weight" % L,
            bytes_fc_q4_0(reader, g + "attn_q.weight", q_units, hidden, interleave))
        # q_norm always present (reshaped_rms_norm feature=hd, F32)
        add("%s_q_norm:gamma" % L,
            bytes_norm_f32(reader, g + "attn_q_norm.weight", hd))

        # wk/wv/k_norm only for NON-shared layers (shared reuse source's K/V).
        if not shared:
            kv_units = hd  # num_kv_heads == 1 for gemma4 E2B
            add("%s_wk:weight" % L,
                bytes_fc_q4_0(reader, g + "attn_k.weight", kv_units, hidden, interleave))
            add("%s_k_norm:gamma" % L,
                bytes_norm_f32(reader, g + "attn_k_norm.weight", hd))
            add("%s_wv:weight" % L,
                bytes_fc_q4_0(reader, g + "attn_v.weight", kv_units, hidden, interleave))
            # v_norm has use_gamma=false -> NO weight (graph requests none).

        # attention_out <- attn_output [hidden, q_units]
        add("%s_attention_out:weight" % L,
            bytes_fc_q4_0(reader, g + "attn_output.weight", hidden, q_units, interleave))

        # post_attention_norm <- post_attention_norm
        add("%s_post_attention_norm:gamma" % L,
            bytes_norm_f32(reader, g + "post_attention_norm.weight", hidden))
        # pre_ffn_norm <- ffn_norm
        add("%s_pre_ffn_norm:gamma" % L,
            bytes_norm_f32(reader, g + "ffn_norm.weight", hidden))

        # MLP: shared layers use double-wide intermediate (use_double_wide_mlp).
        ffn_in = reader.tensors[g + "ffn_gate.weight"]["shape"][0]
        add("%s_ffn_gate:weight" % L,
            bytes_fc_q4_0(reader, g + "ffn_gate.weight", ffn_in, hidden, interleave))
        add("%s_ffn_up:weight" % L,
            bytes_fc_q4_0(reader, g + "ffn_up.weight", ffn_in, hidden, interleave))
        add("%s_ffn_down:weight" % L,
            bytes_fc_q4_0(reader, g + "ffn_down.weight", hidden, ffn_in, interleave))

        # post_ffn_norm <- post_ffw_norm
        add("%s_post_ffn_norm:gamma" % L,
            bytes_norm_f32(reader, g + "post_ffw_norm.weight", hidden))

        # per_layer_input_gate (fully_connected, unit=ple_dim, in=hidden)
        # <- inp_gate [ple_dim, hidden]
        add("%s_per_layer_input_gate:weight" % L,
            bytes_fc_q4_0(reader, g + "inp_gate.weight", ple_dim, hidden, interleave))
        # per_layer_input_proj (fully_connected, unit=hidden, in=ple_dim)
        # <- proj [hidden, ple_dim]
        add("%s_per_layer_input_proj:weight" % L,
            bytes_fc_q4_0(reader, g + "proj.weight", hidden, ple_dim, interleave))
        # post_per_layer_input_norm <- post_norm
        add("%s_post_per_layer_input_norm:gamma" % L,
            bytes_norm_f32(reader, g + "post_norm.weight", hidden))
        # layer_scalar (scalar_multiply use_weight=true) <- layer_output_scale
        add("%s_layer_scalar:scalar_multiplier" % L,
            bytes_scalar_f32(reader, g + "layer_output_scale.weight"))

        print("  layer %2d/%d written (sliding=%s shared=%s hd=%d ffn_in=%d)" %
              (i, n_layers, is_sliding(i), shared, hd, ffn_in))

    # ---- output norm <- output_norm ----
    add("output_norm:gamma",
        bytes_norm_f32(reader, "output_norm.weight", hidden))

    # lm_head is tied (TIE_WORD_EMBEDDINGS=true) -> shares embedding0, no weight.

    write_safetensors(entries, args.output)
    reader.close()


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="Convert native-gemma4 (Gemma 3n E2B) Q4_0 GGUF to "
                    "nntrainer safetensors (Q4_0 FC + Q6_K embedding).")
    ap.add_argument(
        "gguf", nargs="?",
        default="/home/shsh1004/gemma4run/gguf/gemma-4-E2B-it-Q4_0.gguf",
        help="Path to the source .gguf file")
    ap.add_argument(
        "-o", "--output", default=None,
        help="Output .safetensors path (default depends on --fp32)")
    ap.add_argument("--target", choices=["x86", "arm"], default="x86",
                    help="Target CPU for Q4_0 repack (x86 -> q4_0x8)")
    ap.add_argument("--fp32", action="store_true",
                    help="Dequantize every mapped tensor to plain row-major "
                         "F32 (no q4_0x8 repack, no Q6_K) for fp32 compute-"
                         "precision isolation.")
    args = ap.parse_args()
    if args.output is None:
        args.output = os.path.join(
            here, "nntr_gemma4_e2b_fp32.safetensors" if args.fp32
            else "nntr_gemma4_e2b_q4.safetensors")
    return args


if __name__ == "__main__":
    sys.exit(convert(parse_args()) or 0)
