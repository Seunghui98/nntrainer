#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# @file   weight_converter.py
# @date   14 July 2026
# @brief  Convert a YOLOv7ReIDtiny (pose + ReID) PyTorch checkpoint into an
#         nntrainer FP32 safetensors, with weight keys that bind by name to the
#         layers built by yolov7_pose_graph.h.
#
#         Pipeline:
#           1. build YOLOv7ReIDtiny, load_state_dict(strict=True), fuse()
#              (folds Conv+BN so every conv becomes a single biased conv).
#           2. map each fused parameter to its nntrainer weight name and layout.
#           3. write <out>/yolov7_pose.safetensors  (FP32).
#
#         Stage 2 (Q8_0 weights) is produced from this FP32 output by
#         nntr_quantize (see the E2E guide), so this script stays FP32-only.
#
# @author Seungbaek Hong <sb92.hong@samsung.com>

import argparse
import os
import sys

import numpy as np
import torch
from safetensors.numpy import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_pose.yolo_backbone import YOLOv7ReIDtiny  # noqa: E402

# Head hyper-parameters (must match yolov7_pose_graph.h).
NKPT = 87
FEATMAP = 10
FLATTEN = FEATMAP * FEATMAP  # 100


def build_model(checkpoint_path, device):
    model = YOLOv7ReIDtiny(
        nc=1, img_size=320, embed_dim=128, widen_factor=1.5, nkpt=NKPT
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state and hasattr(
        state["model"], "state_dict"
    ):
        state = state["model"].state_dict()
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device).float().eval()
    model.fuse()
    return model


def np32(t):
    return t.detach().cpu().float().numpy().astype(np.float32)


def convert(model):
    """Return {nntrainer_weight_name: np.ndarray(float32)}."""
    sd = model.state_dict()
    out = {}
    consumed = set()

    def take(k):
        consumed.add(k)
        return sd[k]

    # self.model = nn.Sequential(backbone, head, head_feat) re-registers the
    # same tensors under "model.0/1/2.*"; skip those aliases and use the
    # canonical backbone/head/head_feat paths.
    for k in list(sd.keys()):
        if k.startswith("model."):
            consumed.add(k)
            continue
        if k in consumed:
            continue

        # ---- fused Conv modules: "<mp>.conv.weight" / "<mp>.conv.bias" -----
        if k.endswith(".conv.weight"):
            mp = k[: -len(".conv.weight")]
            out[mp + ":filter"] = np32(take(k))  # [out,in,kh,kw] (NCHW)
            bk = mp + ".conv.bias"
            if bk in sd:
                out[mp + ":bias"] = np32(take(bk))
            continue

        # ---- head.final_layer: raw nn.Conv2d (no ".conv" wrapper) ----------
        if k == "head.final_layer.weight":
            out["head.final_layer:filter"] = np32(take(k))
            if "head.final_layer.bias" in sd:
                out["head.final_layer:bias"] = np32(take("head.final_layer.bias"))
            continue

        # ---- ScaleNorm scalar g -> per-channel gamma (broadcast) -----------
        if k == "head.mlp.0.g":
            g = float(np32(take(k)).reshape(-1)[0])
            out["head.mlp.0:gamma"] = np.full((FLATTEN,), g, dtype=np.float32)
            continue

        # ---- GAU (RTMCCBlock) ----------------------------------------------
        if k == "head.gau.uv.weight":  # [2e+s, D] -> [D, 2e+s]
            out["head.gau:uv_weight"] = np32(take(k)).T.copy()
            continue
        if k == "head.gau.o.weight":  # [D, e] -> [e, D]
            out["head.gau:o_weight"] = np32(take(k)).T.copy()
            continue
        if k == "head.gau.gamma":
            out["head.gau:gamma"] = np32(take(k))
            continue
        if k == "head.gau.beta":
            out["head.gau:beta"] = np32(take(k))
            continue
        if k == "head.gau.ln.g":
            out["head.gau:ln_g"] = np32(take(k)).reshape(1)
            continue
        if k == "head.gau.res_scale.scale":
            out["head.gau:res_scale"] = np32(take(k))
            continue

        # ---- generic Linear (mlp.1, cls_x, cls_y, head_feat.fc) ------------
        if k.endswith(".weight"):
            mp = k[: -len(".weight")]
            w = np32(take(k))
            if w.ndim == 2:  # Linear [out,in] -> [in,out]
                out[mp + ":weight"] = w.T.copy()
                bk = mp + ".bias"
                if bk in sd:
                    out[mp + ":bias"] = np32(take(bk))
                continue

        # ---- unmatched .bias handled alongside its weight above ------------
        if k.endswith(".bias") and k in consumed:
            continue

    unconsumed = [k for k in sd.keys() if k not in consumed]
    if unconsumed:
        raise RuntimeError(
            "unmapped checkpoint keys (update converter):\n  "
            + "\n  ".join(unconsumed)
        )
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Convert YOLOv7ReIDtiny .pt -> nntrainer FP32 safetensors"
    )
    ap.add_argument("--weights", default="pose_merged_v311.pt")
    ap.add_argument("--output", default="yolov7_pose.safetensors")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--dump-names",
        action="store_true",
        help="print produced nntrainer weight names and exit",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    model = build_model(args.weights, device)
    tensors = convert(model)

    if args.dump_names:
        for name in sorted(tensors):
            print(f"{name}\t{tuple(tensors[name].shape)}")
        print(f"# total {len(tensors)} tensors")
        return

    save_file(tensors, args.output)
    total = sum(t.nbytes for t in tensors.values())
    print(f"[converter] wrote {args.output}")
    print(f"[converter] {len(tensors)} tensors, {total / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    main()
