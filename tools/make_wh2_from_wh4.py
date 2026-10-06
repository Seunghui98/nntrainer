#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Dummy int2 expert weights for the gemma-4-26B-A4B HTP measurement (doc 57).

Reads a quantized model file written by nntr_quantize_stream with
--moe_dtype QS4CX_WH (every expert weight: WH nibbles, then N f32 scales,
then N f32 column sums) and writes, from the same int4 values:

  --out-wh2  the PACKED file NNTR_MOE_EXPERT_BITS=2 reads: every expert
             weight is its WH2 bytes (2-bit codes, half the nibble bytes)
             followed by the N scales and N column sums, and every other
             tensor follows at its new, earlier offset. The loader counts a
             virtual QS4CX_WH weight at that size under the same switch
             (neuralnet.cpp), so the offsets agree. 7.2 GB for the 12.9 GB
             gemma-4-26B-A4B file.
  --out-wh4  optional: the SAME int2 values in the ordinary WH layout and
             file size, for NNTR_MOE_EXPERT_BITS=4. The two files are then
             the same model, bit for bit after the DSP's expansion, so an
             int4 run on this one and an int2 run on --out-wh2 route
             identically, hit the expert cache identically, and must print
             the same tokens.

Every other tensor is copied unchanged. The int2 values are a coarse
requantization, q2 = clamp(floor((q4 + 2) / 4), -2, 1) with the scale x4, so
the model's output is not meaningful -- this file is for timing.

The expert offsets come from replaying writeGemma4Moe's tensor order with
the dtype sizes quantize_stream.cpp uses; the total is checked against the
file size before anything is written, so a wrong dtype or config is refused
rather than producing a shifted file.

Usage:
  python3 tools/make_wh2_from_wh4.py --config config.json \
      --in nntr_gemma4_q40_arm.bin --out-wh2 nntr_gemma4_wh2.bin \
      [--out-wh4 nntr_gemma4_wh4_int2vals.bin] [--fc-dtype Q4_0] \
      [--embd-dtype Q4_0]
  python3 tools/make_wh2_from_wh4.py --self-test
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np

QK_K_BYTES = {"Q4_K": 144, "Q6_K": 210}


def fc_bytes(k, n, dtype):
    """quantize_stream.cpp's quantizedSize(dtype, rows=n, cols=k)."""
    if dtype == "FP32":
        return 4 * k * n
    if dtype == "Q4_0":
        assert k % 32 == 0
        return n * (k // 32) * 18
    if dtype in QK_K_BYTES:
        assert k % 256 == 0
        return n * (k // 256) * QK_K_BYTES[dtype]
    if dtype == "QS4CX_WH":
        return n * (k // 2) + 8 * n
    raise ValueError("unsupported dtype " + dtype)


def layout(cfg, fc_dtype, embd_dtype):
    """(total bytes, [(offset, K, N) of every expert weight]), in the order
    writeGemma4Moe writes."""
    t = cfg.get("text_config", cfg)
    h = t["hidden_size"]
    off = fc_bytes(h, t["vocab_size"], embd_dtype)  # embedding: rows=vocab
    experts = []
    for li, kind in enumerate(t["layer_types"]):
        sliding = kind == "sliding_attention"
        hd = t["head_dim"] if sliding else t["global_head_dim"]
        k_eq_v = t.get("attention_k_eq_v", False)
        kvh = (t["num_key_value_heads"] if sliding or not k_eq_v
               else t["num_global_key_value_heads"])
        q, kv = t["num_attention_heads"] * hd, kvh * hd
        off += 4 * h  # attention_norm
        off += fc_bytes(h, q, fc_dtype) + 4 * hd  # wq, q_norm
        off += fc_bytes(h, kv, fc_dtype) + 4 * hd  # wk, k_norm
        if not k_eq_v or sliding:
            off += fc_bytes(h, kv, fc_dtype)  # wv
        off += fc_bytes(q, h, fc_dtype)  # attention_out
        off += 8 * h  # post_attention_norm, pre_ffn_norm
        inter = t["intermediate_size"]
        off += 2 * fc_bytes(h, inter, fc_dtype) + fc_bytes(inter, h, fc_dtype)
        off += 12 * h  # post_ffn_norm_1, pre_ffn_norm_2, router_norm
        e, mi = t["num_experts"], t["moe_intermediate_size"]
        off += 4 * h * e + 4 * e  # router, per_expert_scale
        for _ in range(e):
            experts.append((off, h, 2 * mi))
            off += fc_bytes(h, 2 * mi, "QS4CX_WH")
            experts.append((off, mi, h))
            off += fc_bytes(mi, h, "QS4CX_WH")
        off += 8 * h  # post_ffn_norm_2, post_ffn_norm
        if t.get("hidden_size_per_layer_input", 0):
            raise ValueError("per-layer input layers are not replayed here")
        off += 4  # layer_scalar
    off += 4 * h  # output_norm
    if not t.get("tie_word_embeddings", True):
        off += fc_bytes(h, t["vocab_size"], fc_dtype)
    return off, experts


def convert(nib_bytes, scales, k, n):
    """One QS4CX_WH weight -> (wh2 bytes, wh4 bytes of the same int2
    values, scales x4, column sums of the int2 values)."""
    b = np.frombuffer(nib_bytes, np.uint8)
    nib = np.empty(b.size * 2, np.int16)  # slot order: low nibble first
    nib[0::2], nib[1::2] = b & 15, b >> 4
    q4 = np.where(nib > 7, nib - 16, nib)
    q2 = np.clip((q4 + 2) >> 2, -2, 1)  # arithmetic shift = floor
    code = (q2 + 2).astype(np.uint8).reshape(-1, 4)
    wh2 = code[:, 0] | (code[:, 1] << 2) | (code[:, 2] << 4) | (code[:, 3] << 6)
    n4 = (q2 & 15).astype(np.uint8)
    wh4 = n4[0::2] | (n4[1::2] << 4)
    # tiles are kt * n_tiles + nt; within a tile slot s = g*256 + c*8 + j
    # (whSlot), so column c is axis 3 of this view
    tiles = q2.reshape(k // 32, n // 32, 4, 32, 8)
    colsum = tiles.sum(axis=(0, 2, 4), dtype=np.int64).reshape(n)
    colsum = colsum.astype(np.float32)
    return wh2.tobytes(), wh4.tobytes(), (scales * 4.0).astype(np.float32), colsum


def run(args):
    with open(args.config) as f:
        cfg = json.load(f)
    size = os.path.getsize(args.inp)
    total, experts = layout(cfg, args.fc_dtype, args.embd_dtype)
    if total != size:
        sys.exit(f"layout replay gives {total} bytes, the file has {size}: "
                 f"wrong --fc-dtype/--embd-dtype or config. Nothing written.")
    packed = size - sum(k * n // 4 for _, k, n in experts)
    print(f"layout matches the file: {size} bytes, {len(experts)} expert "
          f"weights; packed int2 file will be {packed} bytes")
    out2 = open(args.out_wh2, "wb")
    out4 = None
    if args.out_wh4:
        print("copying ->", args.out_wh4)
        shutil.copyfile(args.inp, args.out_wh4)
        out4 = open(args.out_wh4, "r+b")
    with open(args.inp, "rb") as src:
        pos = 0  # next unread byte of the source

        def copy_through(end):
            nonlocal pos
            while pos < end:
                chunk = src.read(min(64 << 20, end - pos))
                out2.write(chunk)
                pos += len(chunk)

        for i, (off, k, n) in enumerate(experts):
            copy_through(off)
            whb = k * n // 2
            nib = src.read(whb)
            scales = np.frombuffer(src.read(4 * n), np.float32)
            src.seek(4 * n, 1)  # the column sums are recomputed
            pos = off + whb + 8 * n
            wh2, wh4, s4, cs = convert(nib, scales, k, n)
            out2.write(wh2)
            out2.write(s4.tobytes())
            out2.write(cs.tobytes())
            if out4 is not None:
                out4.seek(off)
                out4.write(wh4)
                out4.seek(off + whb)
                out4.write(s4.tobytes())
                out4.write(cs.tobytes())
            if i % 256 == 0:
                print(f"  expert weight {i}/{len(experts)}", flush=True)
        copy_through(size)
    out2.close()
    if out4 is not None:
        out4.close()
    got = os.path.getsize(args.out_wh2)
    if got != packed:
        sys.exit(f"packed file is {got} bytes, expected {packed}")
    print(f"done: {args.out_wh2} = {got} bytes")


def self_test():
    """The repacking on a random tile set: WH2 decoded by the DSP's rule
    equals the WH4 file's nibbles, and the column sums equal a direct sum
    over the WH layout's columns."""
    rng = np.random.default_rng(0)
    k, n = 64, 96
    nib = rng.integers(0, 256, k * n // 2, dtype=np.uint8).tobytes()
    wh2, wh4, s4, cs = convert(nib, np.ones(n, np.float32), k, n)
    b2 = np.frombuffer(wh2, np.uint8)
    # hvx_expand_wh2 / the host stub: code c -> nibble (c + 14) & 15
    codes = np.stack([(b2 >> (2 * q)) & 3 for q in range(4)], 1).reshape(-1)
    nibs = ((codes + 14) & 15).astype(np.uint8)
    exp4 = nibs[0::2] | (nibs[1::2] << 4)
    assert exp4.tobytes() == wh4, "expand(wh2) != wh4"
    # direct column sum: value at (row, col) via whSlot
    vals = np.empty(len(nibs), np.int64)
    vals[:] = np.where(nibs > 7, nibs.astype(np.int64) - 16, nibs)
    ref = np.zeros(n)
    nt_n = n // 32
    for kt in range(k // 32):
        for nt in range(nt_n):
            t = (kt * nt_n + nt) * 1024
            for r in range(32):
                for c in range(32):
                    s = (r // 8) * 256 + c * 8 + (r % 4) * 2 + (r // 4) % 2
                    ref[nt * 32 + c] += vals[t + s]
    assert np.array_equal(ref, cs), "column sums"
    assert np.all(s4 == 4.0)
    # gemma-4-26B-A4B: 30 layers x 128 experts x 2 weights
    cfg = {"hidden_size": 2816, "vocab_size": 262144, "num_attention_heads": 16,
           "head_dim": 256, "global_head_dim": 512, "num_key_value_heads": 8,
           "num_global_key_value_heads": 2, "attention_k_eq_v": True,
           "intermediate_size": 2112, "moe_intermediate_size": 704,
           "num_experts": 128, "tie_word_embeddings": True,
           "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 5}
    total, ex = layout(cfg, "Q4_0", "Q4_0")
    packed = total - sum(k * n // 4 for _, k, n in ex)
    print(f"self-test: repack OK; gemma-4-26B-A4B Q4_0 file = {total} bytes "
          f"({total / 2**20:.0f} MiB), {len(ex)} expert weights; packed int2 "
          f"= {packed} bytes ({packed / 2**20:.0f} MiB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", help="HF config.json (text_config is read)")
    ap.add_argument("--in", dest="inp", help="the QS4CX_WH-expert .bin")
    ap.add_argument("--out-wh2", help="output for NNTR_MOE_EXPERT_BITS=2")
    ap.add_argument("--out-wh4", help="optional: same values, WH layout")
    ap.add_argument("--fc-dtype", default="Q4_0")
    ap.add_argument("--embd-dtype", default="Q4_0")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not (args.config and args.inp and args.out_wh2):
        ap.error("--config, --in and --out-wh2 are required")
    run(args)


if __name__ == "__main__":
    main()
