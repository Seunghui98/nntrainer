#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# @file   make_reference.py
# @brief  Build a YOLOv7ReIDtiny model (random weights unless --weights given),
#         run the fused PyTorch forward on a fixed input, and dump:
#           <out>/pose_merged_random.pt   (checkpoint, when random)
#           <out>/input_320.bin           (NCHW FP32 input, 1x3x320x320)
#           <out>/ref_pose.bin            (FP32 [2*NKPT, SIMCC_BINS])
#           <out>/ref_reid.bin            (FP32 [EMBED_DIM])
#         Used by verify_parity.py to check the nntrainer port end-to-end.

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_pose.yolo_backbone import YOLOv7ReIDtiny  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="", help="optional real .pt")
    ap.add_argument("--out", default=".")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = YOLOv7ReIDtiny(
        nc=1, img_size=320, embed_dim=128, widen_factor=1.5, nkpt=87
    )
    if args.weights:
        state = torch.load(args.weights, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
    else:
        # randomize BN running stats a little so fuse() is non-trivial
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.normal_(0, 0.1)
                m.running_var.uniform_(0.5, 1.5)
        torch.save(model.state_dict(), os.path.join(args.out, "pose_merged_random.pt"))

    model.eval()
    model.fuse()

    x = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        pose, reid = model(x)

    x.numpy().astype(np.float32).tofile(os.path.join(args.out, "input_320.bin"))
    pose[0].numpy().astype(np.float32).tofile(os.path.join(args.out, "ref_pose.bin"))
    reid[0].numpy().astype(np.float32).tofile(os.path.join(args.out, "ref_reid.bin"))
    print("pose", tuple(pose.shape), "reid", tuple(reid.shape))
    print("wrote input_320.bin, ref_pose.bin, ref_reid.bin to", args.out)


if __name__ == "__main__":
    main()
