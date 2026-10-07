#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# @file    fc_wh_sidecar_from_q4.py
# @brief   Write the FC WH sidecar (#225, QS4CX_WH/1) of a Gemma 4 MoE .bin
#          whose FC weights exist only as Q4_0 (no f32 source)
# @author  dlwlzzero <dlwlzzero@gmail.com>
#
# nntr_quantize_stream --fc_wh_sidecar writes the sidecar from the f32
# source. The 26B dummy (#250's repack) has no f32 source: its FCs are the
# external converter's ARM Q4_0 (q4_0x4). This tool reads those bytes and
# re-quantizes them per column, through htp_qs4cx_from_q4_0x4 -- the same
# function the HTP backend runs at load when no sidecar serves a weight
# (qs4cxFromModelQ4_0) -- then whPack, so each image is the one the requant
# path would build in memory. The index (names, K, N, fcWhKey of the Q4_0
# bytes, offsets) is the one quantize_stream writes for the same main file.
# The main file is only read.
#
#   fc_wh_sidecar_from_q4.py <model dir> <out sidecar.bin> [--lib <dir>]
#                            [--check <quantize_stream's sidecar>]
#
# <model dir> holds nntr_config.json (fc_layer_dtype Q4_0) and config.json.
# --lib: the host build's nntrainer/ (libnntrainer.so), default build/.
# --check: compare with a sidecar quantize_stream wrote from the f32 source
# for the same main file (the gemma64 fixture quantized with --isa arm):
# the index must be byte-identical, and each image's dequantized weight is
# compared by SNR (two quantizations of one weight, printed). This is the
# tool's runnable check.
# Prints the two nntr_config.json keys to add (fc_wh_file_name,
# fc_wh_format); it does not edit the model's config.
#
# ponytail: text-only Gemma 4 MoE (writeGemma4Moe's order, tied head, no
# per-layer input), ARM Q4_0 FCs, QS4CX_WH / QS2CX_WH experts. Anything else
# stops here or fails the size check. Upgrade: a --fc_wh_from_q4 mode in
# nntr_quantize_stream's own walk.

import argparse
import json
import os
import subprocess
import sys
import tempfile


FC_BYTES = {"Q4_0": lambda k, n: k * n // 32 * 18, "FP32": lambda k, n: 4 * k * n}


def expert(k, n, dtype):
    if dtype == "FP32":  # an f32 source: gate and up side by side, as read
        return 4 * k * n
    if dtype == "QS4CX_WH":
        return k * n // 2 + 8 * n
    if dtype == "QS2CX_WH":
        return k * n // 4 + 4 + 8 * n
    sys.exit(f"unsupported moe_layer_dtype {dtype}")


def fcs(cfg, nntr):
    """writeGemma4Moe's FC weights as (name, offset, K, N); and the size.

    Also walks an f32 source (every dtype "FP32"): run_inproc_e2e.sh's
    gemma64 reference with the sidecar's weights as its FCs."""
    c = cfg.get("text_config", cfg)
    tied = c.get("tie_word_embeddings", cfg.get("tie_word_embeddings", True))
    if c.get("hidden_size_per_layer_input", 0) or not tied:
        sys.exit("unsupported: per-layer input or untied head")
    q4, emb = FC_BYTES[nntr["fc_layer_dtype"]], FC_BYTES[nntr["embedding_dtype"]]
    H, E, I = c["hidden_size"], c["num_experts"], c["intermediate_size"]
    MI, md = c["moe_intermediate_size"], nntr["moe_layer_dtype"]
    off = emb(H, c["vocab_size"])  # embedding
    out = []

    def fc(name, k, n):
        nonlocal off
        out.append((name, off, k, n))
        off += q4(k, n)

    layers = c["layer_types"][: c["num_hidden_layers"]]
    for i, t in enumerate(layers):
        p = f"layer{i}_"
        sliding = t == "sliding_attention"
        has_wv = sliding or not c.get("attention_k_eq_v")
        hd = c["head_dim"] if sliding else c["global_head_dim"]
        kvh = c["num_key_value_heads"] if has_wv else c["num_global_key_value_heads"]
        qw, kvw = c["num_attention_heads"] * hd, kvh * hd
        off += 4 * H  # attention_norm
        fc(p + "wq", H, qw)
        off += 4 * hd  # q_norm
        fc(p + "wk", H, kvw)
        off += 4 * hd  # k_norm
        if has_wv:
            fc(p + "wv", H, kvw)
        fc(p + "attention_out", qw, H)
        off += 8 * H  # post_attention_norm, pre_ffn_norm
        fc(p + "ffn_gate", H, I)
        fc(p + "ffn_up", H, I)
        fc(p + "ffn_down", I, H)
        off += 8 * H  # post_ffn_norm_1, pre_ffn_norm_2
        off += 4 * H * E + 4 * H + 4 * E  # router, router_scale, expert scale
        off += E * (expert(H, 2 * MI, md) + expert(MI, H, md))
        off += 8 * H + 4  # post_ffn_norm_2, post_ffn_norm, layer_scalar
    off += 4 * H  # output_norm
    return out, off


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(here, "..", "..")
    p = argparse.ArgumentParser()
    p.add_argument("model_dir")
    p.add_argument("output")
    p.add_argument("--lib", default=os.path.join(root, "build", "nntrainer"))
    p.add_argument("--check")
    a = p.parse_args()
    with open(os.path.join(a.model_dir, "nntr_config.json")) as f:
        nntr = json.load(f)
    with open(os.path.join(a.model_dir, "config.json")) as f:
        cfg = json.load(f)
    main_bin = os.path.join(a.model_dir, nntr["model_file_name"])
    if nntr["fc_layer_dtype"] != "Q4_0" or nntr["embedding_dtype"] != "Q4_0":
        sys.exit("fc_layer_dtype and embedding_dtype must be Q4_0")
    rows, size = fcs(cfg, nntr)
    if os.path.getsize(main_bin) != size:
        sys.exit(f"{main_bin} is {os.path.getsize(main_bin)} B, the layout says {size}")
    inc = os.path.join(root, "nntrainer", "tensor")
    with tempfile.TemporaryDirectory() as d:
        exe, tab = os.path.join(d, "fcwh"), os.path.join(d, "table.txt")
        with open(tab, "w") as f:
            f.writelines(f"{n} {o} {k} {nn}\n" for n, o, k, nn in rows)
        subprocess.run(["g++", "-std=c++17", "-O2", "-I", inc, "-o", exe,
                        os.path.join(here, "fc_wh_sidecar_from_q4.cc"),
                        "-L", a.lib, "-lnntrainer", "-Wl,-rpath," + os.path.abspath(a.lib)],
                       check=True)
        cmd = [exe, main_bin, tab, a.output] + ([a.check] if a.check else [])
        rc = subprocess.run(cmd).returncode
    if rc == 0:
        print(f'nntr_config.json: "fc_wh_file_name": "{os.path.basename(a.output)}", '
              f'"fc_wh_format": "QS4CX_WH/1"')
    return rc


if __name__ == "__main__":
    sys.exit(main())
