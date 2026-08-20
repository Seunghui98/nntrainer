# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
## @file verify_encoder.py
## @brief PyTorch reference + nntrainer parity check for the SigLIP2 vision
##        encoder. Emits the golden projected encoder output
##        [1,num_patches,decoder_hidden_size] and the preprocessed pixel
##        tensor, and (with --nntr-npy) compares the nntrainer dump
##        (nntr_encoder_hidden.npy from `nntr_causallm --dump-encoder`).
##
## Supports two checkpoint layouts:
##   --ckpt <dir>                        combined VisionEncoderDecoderModel
##                                        checkpoint (encoder + enc_to_dec_proj
##                                        + decoder all under one HF repo dir).
##   --vision_path <dir> --connector_path <file>
##                                        split checkpoint (e.g. the real
##                                        v4.0.0-S1 layout): a standalone
##                                        vision-tower-only HF dir plus a
##                                        separate raw torch.save'd connector
##                                        state_dict with bare "weight"/"bias"
##                                        keys (e.g. best/encoder_to_decoder.pt).
import argparse
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipVisionModel, VisionEncoderDecoderModel


def preprocess(image_path, processor, resample):
    # Goes through the checkpoint's own AutoImageProcessor for resolution and
    # mean/std, but the resample FILTER is forced explicitly (see --resample):
    # this checkpoint's own preprocessor_config.json says "resample": 2
    # (bilinear), but that field is a stale byproduct of extracting just the
    # vision tower — the checkpoint's actual training/eval code resizes with
    # PIL BICUBIC (see MIGRATION_384.md Findings §1). Trusting the processor's
    # own "resample" blindly reproduces that stale value silently; forcing it
    # here and printing what's actually used avoids a plausible-looking wrong
    # number.
    if resample != "auto":
        processor.resample = {
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
        }[resample]
    print(f"[verify_encoder] using resample={processor.resample!r} "
          f"({'forced' if resample != 'auto' else 'from processor config'})")
    img = Image.open(image_path).convert("RGB")
    return processor(images=img, return_tensors="pt").pixel_values


def load_combined(ckpt):
    """Load a combined VisionEncoderDecoderModel checkpoint."""
    model = VisionEncoderDecoderModel.from_pretrained(
        ckpt, attn_implementation="eager"
    ).eval()
    processor = AutoImageProcessor.from_pretrained(ckpt, use_fast=False)

    def encode(pixel):
        with torch.no_grad():
            enc = model.get_encoder()(pixel_values=pixel,
                                     return_dict=True).last_hidden_state
            proj = (
                model.enc_to_dec_proj(enc)
                if getattr(model, "enc_to_dec_proj", None)
                else enc
            )
        return proj

    return processor, encode


def load_split(vision_path, connector_path):
    """Load a standalone vision tower + a separate raw connector state_dict."""
    vision = SiglipVisionModel.from_pretrained(
        vision_path, attn_implementation="eager"
    ).eval()
    processor = AutoImageProcessor.from_pretrained(vision_path, use_fast=False)
    connector_sd = torch.load(connector_path, map_location="cpu")
    conn_weight, conn_bias = connector_sd["weight"], connector_sd["bias"]

    def encode(pixel):
        with torch.no_grad():
            enc = vision(pixel_values=pixel, return_dict=True).last_hidden_state
            proj = torch.nn.functional.linear(enc, conn_weight, conn_bias)
        return proj

    return processor, encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="combined VisionEncoderDecoder HF checkpoint dir")
    ap.add_argument("--vision_path", help="standalone vision-tower HF checkpoint dir "
                    "(split-layout alternative to --ckpt)")
    ap.add_argument("--connector_path", help="raw enc_to_dec_proj connector "
                    "state_dict, e.g. best/encoder_to_decoder.pt (required "
                    "with --vision_path)")
    ap.add_argument("--image", default="sample.png")
    ap.add_argument("--out", default="golden", help="output prefix")
    ap.add_argument("--nntr-npy", help="nntr_encoder_hidden.npy to compare")
    ap.add_argument("--resample", choices=["bicubic", "bilinear", "auto"],
                    default="bicubic",
                    help="resize filter to force (default: bicubic, confirmed "
                    "correct for the real 384px checkpoint per "
                    "MIGRATION_384.md Findings §1 — do not trust the "
                    "checkpoint's own preprocessor_config.json 'resample' "
                    "field, which is stale). 'auto' trusts the processor's "
                    "own config instead.")
    # Float32 activations differ across implementations only by accumulation
    # order (GEMM / softmax / gelu-tanh), so parity is judged by cosine
    # similarity + relative L2, not raw max-abs (which is dominated by a handful
    # of large-magnitude outliers). Defaults are tight enough that any real
    # weight/graph error fails immediately.
    ap.add_argument("--min-cos", type=float, default=0.9999)
    ap.add_argument("--max-rel-l2", type=float, default=1e-2)
    args = ap.parse_args()

    if args.ckpt:
        processor, encode = load_combined(args.ckpt)
    elif args.vision_path and args.connector_path:
        processor, encode = load_split(args.vision_path, args.connector_path)
    else:
        ap.error("pass either --ckpt, or both --vision_path and --connector_path")

    pixel = preprocess(args.image, processor, args.resample)
    proj = encode(pixel)
    golden = proj.cpu().numpy().astype(np.float32)  # [1,num_patches,decoder_hidden_size]

    np.save(args.out + ".encoder_hidden.npy", golden)
    np.save(args.out + ".pixel.npy", pixel.cpu().numpy())
    print("golden encoder_hidden shape:", golden.shape)
    print("saved:", args.out + ".encoder_hidden.npy,", args.out + ".pixel.npy")

    if args.nntr_npy:
        nntr = np.load(args.nntr_npy).astype(np.float64).reshape(-1)
        ref = golden.astype(np.float64).reshape(-1)
        if nntr.shape != ref.shape:
            raise SystemExit(
                f"SHAPE MISMATCH: nntr has {nntr.shape[0]} floats, golden has "
                f"{ref.shape[0]} — check num_patches/decoder_hidden_size "
                "config agree between the PyTorch and nntrainer sides."
            )
        diff = np.abs(ref - nntr)
        cos = float(ref @ nntr / (np.linalg.norm(ref) * np.linalg.norm(nntr)))
        rel_l2 = float(np.linalg.norm(ref - nntr) / np.linalg.norm(ref))
        print(f"max |diff| = {diff.max():.6e}  mean |diff| = {diff.mean():.6e}")
        print(f"cosine = {cos:.8f} (min {args.min_cos})  "
              f"rel L2 = {rel_l2:.6e} (max {args.max_rel_l2})")
        ok = (cos >= args.min_cos) and (rel_l2 <= args.max_rel_l2)
        print("ENCODER PARITY:", "PASS" if ok else "FAIL")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
