# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>

## @file weight_converter.py
## @brief Weight conversion script for the PE-Lang-L14-448 vision encoder
##        + BERT-small decoder of ScreenAI caption v3.0.0-C62.
##
## Source assets:
##   v3.0.0-C62/PE-Lang-L14-448.pt          bf16, 327 tensors, OpenCLIP names
##   v3.0.0-C62/best/encoder_to_decoder.pt  fp32, 2 tensors (proj weight+bias)
##   v3.0.0-C62/best/decoder/model.safetensors  fp32, 114 tensors, HF Bert
##   pelang_golden/rope_{sin,cos}.npy       [1024,64] static RoPE table
##
## Encoder tensor order MUST match PELangVisionEncoder::constructModel graph
## order (see §3.2 of the support plan). Decoder order is identical to the
## SigLIP2-path BERT decoder (collect_decoder reused verbatim — the layout is
## dimension-agnostic per §1.4).
##
## Plan-vs-code discrepancies (intentional, code-authoritative):
##   * Source `.pt` keys are bare `conv1.weight` / `class_embedding` /
##     `positional_embedding` / `ln_pre.*` / `transformer.resblocks.N.*`
##     (no PE-Lang-specific schema) — direct rename, no timm.
##   * RoPE sin/cos tables are NOT in the source checkpoint (timm builds them
##     from ref_feat_shape/freqs at load time); they come from the golden
##     dump_pelang_golden.py output as static constant tensors.
##   * Conv2D weight is written under :filter (nntrainer conv2d convention),
##     not the brief's :weight name — matching the SigLIP2 converter.
##   * LayerScale gamma written under :gamma (consistent with LN gamma naming),
##     so the quantizer dtype map naturally excludes both.
##   * Tensor count: 7 (header) + 23*18 (per-block) + 2 (proj) = 423. The brief's
##     "4+4+23*15+2=355" formula is an arithmetic slip; the actual count of
##     listed tensors is 423.

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch
from safetensors.torch import load_file

# Reuse the existing BERT-decoder converter's collect_decoder (dimension-
# agnostic rename+transpose — same key layout as the v2.3 BERT-small decoder
# per plan §1.4). The v3.0.0-C62 source uses bare `bert.`/`cls.` prefixes, so
# we prepend `decoder.` to the loaded state dict before handing it over.
_BERT_DEC_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "bert-decoder"
)
sys.path.insert(0, _BERT_DEC_DIR)
import weight_converter as bert_dec_wc  # noqa: E402

SAFETENSORS_DTYPE_MAP = {"float32": "F32"}

ENC_LAYERS = 23
EMBED_DIM = 1024
PROJ_DIM = 512


# ---------------------------------------------------------------------------
# Helpers (same shape as res/siglip2-encoder/weight_converter.py)
# ---------------------------------------------------------------------------

def tensor_to_numpy(tensor, dtype, transpose=False):
    """Convert torch tensor to contiguous numpy array (optionally transposed)."""
    if transpose:
        tensor = tensor.permute(1, 0)
    return np.ascontiguousarray(tensor.detach().cpu().numpy().astype(dtype))


def get_safetensors_output_name(output_name):
    if output_name.endswith(".bin"):
        return output_name[:-4] + ".safetensors"
    if output_name.endswith(".safetensors"):
        return output_name
    return output_name + ".safetensors"


def save_safetensors(weights, output_path, dtype):
    if dtype not in SAFETENSORS_DTYPE_MAP:
        raise ValueError(f"Unsupported safetensors dtype: {dtype}")
    safetensors_dtype = SAFETENSORS_DTYPE_MAP[dtype]
    metadata = {"format": "pt"}

    offset = 0
    tensor_meta = {}
    raw_buffers = []

    for name, arr in weights:
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        nbytes = arr.nbytes
        tensor_meta[name] = {
            "dtype": safetensors_dtype,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        raw_buffers.append(arr.tobytes(order="C"))
        offset += nbytes

    header = {"__metadata__": metadata}
    header.update(tensor_meta)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad

    with open(output_path, "wb") as output_file:
        output_file.write(struct.pack("<Q", len(header_bytes)))
        output_file.write(header_bytes)
        for buffer in raw_buffers:
            output_file.write(buffer)

    print(f"Saved safetensors: {output_path}")
    print(f"Tensor data size: {offset / 1e9:.4f} GB")


def save_bin(weights, output_path):
    with open(output_path, "wb") as f:
        for _name, arr in weights:
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            arr.tofile(f)
    total_bytes = sum(arr.nbytes for _, arr in weights)
    print(f"Saved binary: {output_path}")
    print(f"Tensor data size: {total_bytes / 1e9:.4f} GB")


# ---------------------------------------------------------------------------
# Encoder weight collection
# Order MUST match PELangVisionEncoder::constructModel (Phase 2 graph order).
#
# 1.  pe_patch_conv:filter           [1024,3,14,14]  OIHW, no bias
# 2.  pe_pos_embed:pe_pos_embed      [1,1,1024,1024] = positional_embedding[1:]
# 3.  pe_cls_row:pe_cls_row          [1,1,1,1024]    = class_emb + pos_embed[0]
# 4.  pe_rope_sin:pe_rope_sin        [1,1,1024,64]   static, BF/FP32 from golden
# 5.  pe_rope_cos:pe_rope_cos        [1,1,1024,64]   static, BF/FP32 from golden
# 6.  pe_ln_pre:gamma, beta          [1024] x 2
# for i in 0..22:
#       enc_layer{i}_ln1:gamma, beta           (2)
#       enc_layer{i}_wq:weight, bias           (2)  in_proj_weight[0:1024]
#       enc_layer{i}_wk:weight, bias           (2)  in_proj_weight[1024:2048]
#       enc_layer{i}_wv:weight, bias           (2)  in_proj_weight[2048:3072]
#       enc_layer{i}_out:weight, bias          (2)
#       enc_layer{i}_ls1:gamma                 (1)  LayerScale
#       enc_layer{i}_ln2:gamma, beta           (2)
#       enc_layer{i}_fc1:weight, bias          (2)  mlp.c_fc
#       enc_layer{i}_fc2:weight, bias          (2)  mlp.c_proj
#       enc_layer{i}_ls2:gamma                 (1)  LayerScale
#                                            = 18 per layer
# final:
#       enc_to_dec_proj:weight, bias           (2)
# Total = 7 + 23*18 + 2 = 423
# ---------------------------------------------------------------------------

def collect_encoder(sd_enc, proj_sd, rope_sin, rope_cos, dtype):
    """Return ordered list of (nntr_name, ndarray) for the encoder file."""
    weights = []

    def add(name, tensor, transpose=False):
        arr = tensor_to_numpy(tensor, dtype, transpose=transpose)
        weights.append((name, arr))

    # ----- 1. Patch embed conv (OIHW, no bias; trap §6.5) -----
    # nntrainer conv2d weight name is "filter" (the brief's ":weight" was a
    # typo — verified against res/siglip2-encoder).
    add("pe_patch_conv:filter", sd_enc["conv1.weight"], transpose=False)

    # ----- 2. patch positions (positional_embedding[1:]) -> [1,1,1024,1024] -----
    pos = sd_enc["positional_embedding"]   # [1025, 1024]
    patch_pos = pos[1:].unsqueeze(0).unsqueeze(0)
    add("pe_pos_embed:pe_pos_embed", patch_pos, transpose=False)

    # ----- 3. CLS row = class_embedding + positional_embedding[0] -----
    # Plan §3.2 CLS+pos fuse (maxdiff 0.0). Token 0 is input-independent, so we
    # pre-compute the constant and stuff it into pe_cls_row.
    cls_row = (sd_enc["class_embedding"] + pos[0]).view(1, 1, 1, EMBED_DIM)
    add("pe_cls_row:pe_cls_row", cls_row, transpose=False)

    # ----- 4/5. RoPE static tables (trap §6.7/15: shared across all 23 blocks) -----
    sin_t = torch.from_numpy(
        np.ascontiguousarray(rope_sin.astype(np.float32))
    ).view(1, 1, 1024, 64)
    cos_t = torch.from_numpy(
        np.ascontiguousarray(rope_cos.astype(np.float32))
    ).view(1, 1, 1024, 64)
    add("pe_rope_sin:pe_rope_sin", sin_t, transpose=False)
    add("pe_rope_cos:pe_rope_cos", cos_t, transpose=False)

    # ----- 6. ln_pre gamma/beta -----
    add("pe_ln_pre:gamma", sd_enc["ln_pre.weight"])
    add("pe_ln_pre:beta",  sd_enc["ln_pre.bias"])

    # ----- 7. 23 transformer blocks -----
    for i in range(ENC_LAYERS):
        lp = f"transformer.resblocks.{i}."
        pfx = f"enc_layer{i}"

        # Pre-LN1 (eps set in C++ to 1e-5, trap §6.4)
        add(f"{pfx}_ln1:gamma", sd_enc[f"{lp}ln_1.weight"])
        add(f"{pfx}_ln1:beta",  sd_enc[f"{lp}ln_1.bias"])

        # Fused QKV split (trap §6.8: order is q/k/v at [0:1024]/[1024:2048]/
        # [2048:3072]). Each slice is HF [out=1024, in=1024] -> transpose to
        # nntrainer FC [in, out].
        in_w = sd_enc[f"{lp}attn.in_proj_weight"]   # [3072, 1024]
        in_b = sd_enc[f"{lp}attn.in_proj_bias"]     # [3072]
        add(f"{pfx}_wq:weight", in_w[0:1024],   transpose=True)
        add(f"{pfx}_wq:bias",   in_b[0:1024])
        add(f"{pfx}_wk:weight", in_w[1024:2048], transpose=True)
        add(f"{pfx}_wk:bias",   in_b[1024:2048])
        add(f"{pfx}_wv:weight", in_w[2048:3072], transpose=True)
        add(f"{pfx}_wv:bias",   in_b[2048:3072])

        # Output projection (HF [out, in] -> transpose)
        add(f"{pfx}_out:weight", sd_enc[f"{lp}attn.out_proj.weight"], transpose=True)
        add(f"{pfx}_out:bias",   sd_enc[f"{lp}attn.out_proj.bias"])

        # LayerScale gamma (FP32-only at quant time — Phase 5 trap §6.12)
        add(f"{pfx}_ls1:gamma", sd_enc[f"{lp}ls_1.gamma"])

        # Pre-LN2
        add(f"{pfx}_ln2:gamma", sd_enc[f"{lp}ln_2.weight"])
        add(f"{pfx}_ln2:beta",  sd_enc[f"{lp}ln_2.bias"])

        # MLP fc1 -> erf GELU (NOT tanh_gelu — trap §6.2) -> fc2.
        # HF c_fc [out=4096, in=1024] -> transpose [1024, 4096] for nntrainer FC.
        # HF c_proj [out=1024, in=4096] -> transpose [4096, 1024].
        add(f"{pfx}_fc1:weight", sd_enc[f"{lp}mlp.c_fc.weight"], transpose=True)
        add(f"{pfx}_fc1:bias",   sd_enc[f"{lp}mlp.c_fc.bias"])
        add(f"{pfx}_fc2:weight", sd_enc[f"{lp}mlp.c_proj.weight"], transpose=True)
        add(f"{pfx}_fc2:bias",   sd_enc[f"{lp}mlp.c_proj.bias"])

        # LayerScale gamma
        add(f"{pfx}_ls2:gamma", sd_enc[f"{lp}ls_2.gamma"])

    # ----- 8. Encoder-to-decoder projection [512,1024]+[512] (fp32 source) -----
    add("enc_to_dec_proj:weight", proj_sd["weight"], transpose=True)  # -> [1024,512]
    add("enc_to_dec_proj:bias",   proj_sd["bias"])

    # Graph execution orders independent constant nodes lexicographically:
    # pe_cls_row, pe_pos_embed, pe_rope_cos, pe_rope_sin. Keep the raw binary
    # order aligned with that order; unlike safetensors, .bin has no names.
    weights[1:5] = [weights[2], weights[1], weights[4], weights[3]]
    for offset in range(7, 7 + ENC_LAYERS * 18, 18):
        weights[offset + 2:offset + 6] = weights[offset + 4:offset + 6] + weights[offset + 2:offset + 4]
    return weights


# ---------------------------------------------------------------------------
# Decoder weight collection
#
# Plan §1.4: identical key layout to the v2.3 BERT-small decoder, only the
# dimension values differ. The existing res/bert-decoder/weight_converter.py
# collect_decoder is a pure rename+transpose independent of dimension, so we
# reuse it. The v3.0.0-C62 source safetensors uses bare `bert.`/`cls.` prefixes
# (no `decoder.` wrapper), so we add the prefix before delegating.
# ---------------------------------------------------------------------------

def collect_decoder(sd_dec, dtype):
    sd_remapped = {f"decoder.{k}": v for k, v in sd_dec.items()}
    return bert_dec_wc.collect_decoder(sd_remapped, dtype)


# ---------------------------------------------------------------------------
# Argument parsing and main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the PE-Lang-L14-448 encoder + BERT-small decoder "
        "of ScreenAI caption v3.0.0-C62 to nntrainer weight format."
    )
    parser.add_argument(
        "--encoder_pt",
        default="v3.0.0-C62/PE-Lang-L14-448.pt",
        help="Path to PE-Lang-L14-448.pt (bf16 OpenCLIP state dict)",
    )
    parser.add_argument(
        "--proj_pt",
        default="v3.0.0-C62/best/encoder_to_decoder.pt",
        help="Path to encoder_to_decoder.pt (fp32 projection state dict)",
    )
    parser.add_argument(
        "--decoder_safetensors",
        default="v3.0.0-C62/best/decoder/model.safetensors",
        help="Path to the BERT-small decoder safetensors",
    )
    parser.add_argument(
        "--rope_dir",
        default="/home/leeseunghui/workspace/models/pelang_golden",
        help="Directory containing rope_sin.npy / rope_cos.npy [1024,64] FP32",
    )
    parser.add_argument(
        "--encoder_output",
        default="./nntr_pelang_encoder_fp32.bin",
        help="Output path for encoder+projection weight file",
    )
    parser.add_argument(
        "--decoder_output",
        default="./nntr_pelang_decoder_fp32.bin",
        help="Output path for decoder weight file",
    )
    parser.add_argument(
        "--data_type",
        default="float32",
        choices=["float32"],
        help="Output data type",
    )
    parser.add_argument(
        "--safetensors",
        action="store_true",
        help="Save in safetensors format instead of binary",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = args.data_type

    # ---- Load encoder checkpoint, bf16 -> fp32 (trap §6.9) ----
    print(f"Loading encoder checkpoint: {args.encoder_pt}")
    sd_enc = torch.load(args.encoder_pt, map_location="cpu", weights_only=True)
    assert isinstance(sd_enc, dict), f"unexpected encoder ckpt type {type(sd_enc)}"
    assert len(sd_enc) == 327, f"unexpected encoder tensor count {len(sd_enc)}"
    for k in list(sd_enc.keys()):
        sd_enc[k] = sd_enc[k].to(torch.float32)
    print(f"Encoder loaded (bf16->fp32) — {len(sd_enc)} tensors")

    # ---- Load projection ----
    print(f"Loading projection checkpoint: {args.proj_pt}")
    proj_sd = torch.load(args.proj_pt, map_location="cpu", weights_only=False)
    if hasattr(proj_sd, "state_dict"):
        proj_sd = proj_sd.state_dict()
    assert "weight" in proj_sd and "bias" in proj_sd, \
        f"projection missing weight/bias: keys={list(proj_sd.keys())}"
    print(f"Projection loaded — {len(proj_sd)} tensors")

    # ---- Load RoPE tables (static, from the golden dump) ----
    rope_sin = np.load(os.path.join(args.rope_dir, "rope_sin.npy"))
    rope_cos = np.load(os.path.join(args.rope_dir, "rope_cos.npy"))
    assert rope_sin.shape == (1024, 64), f"rope_sin shape {rope_sin.shape}"
    assert rope_cos.shape == (1024, 64), f"rope_cos shape {rope_cos.shape}"
    print(f"RoPE tables loaded — sin/cos {rope_sin.shape}")

    enc_weights = collect_encoder(sd_enc, proj_sd, rope_sin, rope_cos, dtype)
    print(f"\nEncoder tensors: {len(enc_weights)}")
    expected_enc = 7 + ENC_LAYERS * 18 + 2  # 423
    assert len(enc_weights) == expected_enc, (
        f"ENCODER COUNT MISMATCH: got {len(enc_weights)}, "
        f"expected {expected_enc} (=7 + 23*18 + 2). "
        "Brief's 4+4+23*15+2=355 formula is an arithmetic slip; the actual "
        "count of tensors listed in the spec is 423."
    )

    # ---- Encoder write ----
    if args.safetensors:
        out = get_safetensors_output_name(args.encoder_output)
        save_safetensors(enc_weights, out, dtype)
    else:
        save_bin(enc_weights, args.encoder_output)

    # ---- Decoder ----
    print(f"\nLoading decoder safetensors: {args.decoder_safetensors}")
    sd_dec = load_file(args.decoder_safetensors)
    print(f"Decoder loaded — {len(sd_dec)} tensors (bare bert./cls. prefix)")

    dec_weights = collect_decoder(sd_dec, dtype)
    print(f"\nDecoder tensors: {len(dec_weights)}")
    assert len(dec_weights) == 114, (
        f"DECODER COUNT MISMATCH: got {len(dec_weights)}, expected 114"
    )

    if args.safetensors:
        out = get_safetensors_output_name(args.decoder_output)
        save_safetensors(dec_weights, out, dtype)
    else:
        save_bin(dec_weights, args.decoder_output)

    print("\nDone.")


if __name__ == "__main__":
    main()
