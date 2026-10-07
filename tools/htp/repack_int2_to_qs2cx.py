#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# @file    repack_int2_to_qs2cx.py
# @brief   Repack the external "int2" Gemma 4 MoE .bin into the QS2CX_WH
#          expert layout htp_decode reads, every other tensor byte for byte
# @author  dlwlzzero <dlwlzzero@gmail.com>
#
# The external converter (#250) writes writeGemma4Moe's tensor order
# (Applications/CausalLM/quantize_stream.cpp) with Q4_0 embedding / FCs and
# 2-bit experts that are QS2CX_WH in all but two things:
#   * codes: slot sl of a 32x32 WH tile (whSlot, htp_wh_layout.h) sits at
#     tile byte sl/4, shift 2*(sl%4), value = code - 2; QS2CX_WH puts it at
#     whCodeByte2(sl) / whCodeShift2(sl) (hvx_expand_i2i4.h);
#   * no palette: QS2CX_WH has 4 bytes between codes and scales.
# So per expert tensor: the codes are permuted inside each 256-byte tile,
# the palette fe ff 00 01 (int4 -2, -1, 0, +1 ascending, so palette[code] =
# code - 2 and no code value changes) is inserted, scale[N] and colsum[N]
# are copied. The permutation, worked from the two formulas: output byte j
# (j < 128) of a tile takes the input byte j/2's nibble j%2 as its low
# nibble and byte 64 + j/2's as its high one; bytes 128..255 the same from
# input bytes 128.. and 192...
#
#   repack_int2_to_qs2cx.py <in.bin> <config.json> <out.bin>
#                           [--skip-verify | --verify-only]
#
# Writes <out.bin>, every *.json beside <in.bin> next to it, and
# nntr_config.json with moe_layer_dtype QS2CX_WH and the new
# model_file_name. Then compiles and runs repack_int2_check.cc, which walks
# both files with this script's segment table and checks every byte:
# copies equal, every expert's whUnpack2 (the loader's own reader) equal
# to the input's sl/4 decode, the input's colsum equal to its decoded
# column sums, the palette, scale and colsum bytes. Streams: one expert
# tensor (about 1 MB) in memory at a time.

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

PALETTE = bytes([0xFE, 0xFF, 0x00, 0x01])
TILE2 = 256  # bytes of one 32x32 tile at 2 bits
LO = bytes(b & 0x0F for b in range(256))
HI = bytes(b >> 4 for b in range(256))
LO_UP = bytes((b & 0x0F) << 4 for b in range(256))
HI_UP = bytes(b & 0xF0 for b in range(256))
FC_BYTES = {"Q4_0": lambda k, n: k * n // 32 * 18, "FP32": lambda k, n: 4 * k * n}


def segments(cfg, nntr):
    """writeGemma4Moe's order as ("C", bytes) copies and ("E", K, N) experts."""
    c = cfg.get("text_config", cfg)
    # ponytail: the dummy's layout -- text-only Gemma 4 MoE, tied head, no
    # per-layer input, embedding / FC in a FC_BYTES dtype, experts as
    # described in the header. A real checkpoint by the same script (user,
    # 2026-10-07) is the same; anything else stops here or fails the size
    # check. Upgrade: read the external converter's tensor table if it ever
    # ships one.
    if c.get("hidden_size_per_layer_input", 0) or not c.get("tie_word_embeddings", True):
        sys.exit("unsupported: per-layer input or untied head")
    fc = FC_BYTES[nntr["fc_layer_dtype"]]
    emb = FC_BYTES[nntr["embedding_dtype"]]
    H, E, I = c["hidden_size"], c["num_experts"], c["intermediate_size"]
    MI = c["moe_intermediate_size"]
    yield ("C", emb(c["vocab_size"], H))
    for t in c["layer_types"][: c["num_hidden_layers"]]:
        sliding = t == "sliding_attention"
        has_wv = sliding or not c.get("attention_k_eq_v")
        hd = c["head_dim"] if sliding else c["global_head_dim"]
        kvh = c["num_key_value_heads"] if has_wv else c["num_global_key_value_heads"]
        qw, kvw = c["num_attention_heads"] * hd, kvh * hd
        n = 4 * H + fc(H, qw) + 4 * hd + fc(H, kvw) + 4 * hd
        n += fc(H, kvw) if has_wv else 0
        n += fc(qw, H) + 8 * H + 2 * fc(H, I) + fc(I, H) + 8 * H
        n += 4 * H * E + 4 * H + 4 * E  # router, router_scale, per-expert scale
        yield ("C", n)
        for _ in range(E):
            yield ("E", H, 2 * MI)  # gate|up fused
            yield ("E", MI, H)  # down
        yield ("C", 8 * H + 4)  # post_ffn_norm_2, post_ffn_norm, layer_scalar
    yield ("C", 4 * H)  # output_norm


def or_bytes(a, b):
    """Bytewise OR of two equal-length strings."""
    return (int.from_bytes(a, "little") | int.from_bytes(b, "little")).to_bytes(len(a), "little")


def repack_codes(x):
    """sl/4 tile codes -> whCodeByte2 tile codes (see the file header)."""
    out = bytearray(len(x))
    for half in (0, 128):
        for i in range(64):
            a, b = x[half + i :: TILE2], x[half + 64 + i :: TILE2]
            out[half + 2 * i :: TILE2] = or_bytes(a.translate(LO), b.translate(LO_UP))
            out[half + 2 * i + 1 :: TILE2] = or_bytes(a.translate(HI), b.translate(HI_UP))
    return out


def copy_n(src, dst, n):
    while n:
        buf = src.read(min(n, 64 << 20))
        if not buf:
            sys.exit("input ended early")
        dst.write(buf)
        n -= len(buf)


def table(segs):
    """Segments with their offsets: (kind, in_off, out_off, a, b)."""
    i = o = 0
    for s in segs:
        if s[0] == "C":
            yield ("C", i, o, s[1], 0)
            i, o = i + s[1], o + s[1]
        else:
            K, N = s[1], s[2]
            if K % 32 or N % 32:
                sys.exit(f"expert {K}x{N} is not whole WH tiles")
            yield ("E", i, o, K, N)
            i, o = i + K * N // 4 + 8 * N, o + K * N // 4 + 4 + 8 * N


def verify(src, out, rows):
    here = os.path.dirname(os.path.abspath(__file__))
    inc = os.path.join(here, "..", "..", "nntrainer", "tensor")
    with tempfile.TemporaryDirectory() as d:
        exe, tab = os.path.join(d, "check"), os.path.join(d, "table.txt")
        with open(tab, "w") as f:
            f.writelines(" ".join(map(str, r)) + "\n" for r in rows)
        subprocess.run(["g++", "-std=c++17", "-O2", "-fopenmp", "-I", inc, "-o", exe,
                        os.path.join(here, "repack_int2_check.cc")], check=True)
        return subprocess.run([exe, src, out, tab]).returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("config")
    p.add_argument("output")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--skip-verify", action="store_true")
    g.add_argument("--verify-only", action="store_true", help="check an existing output")
    a = p.parse_args()

    src_dir = os.path.dirname(os.path.abspath(a.input))
    out_dir = os.path.dirname(os.path.abspath(a.output))
    if src_dir == out_dir:
        sys.exit("output must be in another directory (its nntr_config.json is rewritten)")
    with open(a.config) as f:
        cfg = json.load(f)
    with open(os.path.join(src_dir, "nntr_config.json")) as f:
        nntr = json.load(f)
    rows = list(table(segments(cfg, nntr)))
    last = rows[-1]
    in_size, out_size = last[1] + last[3], last[2] + last[3]
    if os.path.getsize(a.input) != in_size:
        sys.exit(f"input is {os.path.getsize(a.input)} B, the layout says {in_size}")

    if not a.verify_only:
        os.makedirs(out_dir, exist_ok=True)
        with open(a.input, "rb") as fi, open(a.output, "wb") as fo:
            for n, (kind, _, _, k, nn) in enumerate(rows):
                if kind == "C":
                    copy_n(fi, fo, k)
                    continue
                fo.write(repack_codes(fi.read(k * nn // 4)))
                fo.write(PALETTE)
                copy_n(fi, fo, 8 * nn)
                if n % 1000 == 0:
                    print(f"  {n}/{len(rows)} segments", flush=True)
        for name in os.listdir(src_dir):
            if name.endswith(".json") and name != "nntr_config.json":
                shutil.copy(os.path.join(src_dir, name), out_dir)
        nntr["moe_layer_dtype"] = "QS2CX_WH"
        nntr["model_file_name"] = os.path.basename(a.output)
        with open(os.path.join(out_dir, "nntr_config.json"), "w") as f:
            json.dump(nntr, f, indent=2, ensure_ascii=False)
            f.write("\n")

    got = os.path.getsize(a.output)
    experts = sum(r[0] == "E" for r in rows)
    print(f"REPACK in={in_size} out={got} expected={out_size} experts={experts}", flush=True)
    if got != out_size:
        sys.exit("REPACK FAIL output size")
    if not a.skip_verify:
        sys.exit(verify(a.input, a.output, rows))


if __name__ == "__main__":
    main()
