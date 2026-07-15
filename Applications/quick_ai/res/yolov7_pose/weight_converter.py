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
import importlib.abc
import importlib.machinery
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn
from safetensors.numpy import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_pose.yolo_backbone import YOLOv7ReIDtiny  # noqa: E402

# Head hyper-parameters (must match yolov7_pose_graph.h).
NKPT = 87
FEATMAP = 10
FLATTEN = FEATMAP * FEATMAP  # 100


class _Placeholder(nn.Module):
    """Permissive nn.Module stand-in for an unavailable training-repo class.

    A full torch.save(model) may pickle instance state that re-binds methods
    (e.g. a fused Conv does ``self.forward = self.forward_fuse``); restoring
    such a bound method calls ``getattr(obj, 'forward_fuse')``. Return a no-op
    callable for any attribute the real class would have provided, so the
    (real) parameter/buffer/submodule dicts still deserialize and state_dict()
    recovers the tensors.
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)  # _parameters/_buffers/_modules
        except AttributeError:
            if name.startswith("__") and name.endswith("__"):
                raise
            return lambda *a, **k: None


def _placeholder_class(module_name, cls_name):
    return type(cls_name, (_Placeholder,), {"__module__": module_name})


class _ShimLoader(importlib.abc.Loader):
    """Materialize a module whose every attribute is an nn.Module placeholder."""

    def create_module(self, spec):
        m = types.ModuleType(spec.name)

        def _getattr(name, _m=m):
            cls = _placeholder_class(_m.__name__, name)
            setattr(_m, name, cls)
            return cls

        m.__getattr__ = _getattr
        return m

    def exec_module(self, module):
        pass


class _ShimFinder(importlib.abc.MetaPathFinder):
    """Serve shim modules for training-repo packages absent in this env."""

    def __init__(self, roots):
        self.roots = set(roots)

    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in self.roots:
            return importlib.machinery.ModuleSpec(name, _ShimLoader())
        return None


def _load_checkpoint(checkpoint_path, device):
    """Load a checkpoint saved either as a state_dict or as a full model object.

    A full ``torch.save(model)`` embeds the training repo's class paths (e.g.
    ``models.*``); if that package is absent we install a permissive shim so
    unpickling reconstructs the module tree as nn.Module placeholders, from
    which the (real) state_dict is recovered. Only the tensors matter here.
    """
    installed = []
    while True:
        try:
            obj = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            break
        except ModuleNotFoundError as e:
            root = (e.name or "").split(".")[0]
            if not root or root in installed:
                raise
            sys.meta_path.insert(0, _ShimFinder({root}))
            installed.append(root)
            print(f"[converter] shimming missing package '{root}' for unpickling")

    if hasattr(obj, "state_dict") and not isinstance(obj, dict):
        return obj.state_dict()
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "ema", "weights"):
            v = obj.get(key)
            if hasattr(v, "state_dict") and not isinstance(v, dict):
                return v.state_dict()
            if isinstance(v, dict):
                return v
    return obj


def build_model(checkpoint_path, device):
    model = YOLOv7ReIDtiny(
        nc=1, img_size=320, embed_dim=128, widen_factor=1.5, nkpt=NKPT
    )
    state = _load_checkpoint(checkpoint_path, device)
    # tolerate a "module." prefix on externally-saved state dicts
    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}

    # A checkpoint with no BatchNorm keys was already fused (Conv+BN folded);
    # fuse the reconstruction first so the key sets line up.
    already_fused = not any(".bn." in k for k in state)
    model.eval()
    if already_fused:
        model.fuse()

    missing, unexpected = model.load_state_dict(state, strict=False)
    # Ignore the aliased "model.*" (nn.Sequential) view: those tensors are
    # shared with backbone/head/head_feat and always resolve via the canonical
    # paths, so their absence from either side is benign.
    missing = [k for k in missing if not k.startswith("model.")]
    unexpected = [k for k in unexpected if not k.startswith("model.")]
    if missing or unexpected:
        print(
            f"[converter] load_state_dict: {len(missing)} missing, "
            f"{len(unexpected)} unexpected keys"
        )
        if missing:
            print("  e.g. missing:", missing[:8])
        if unexpected:
            print("  e.g. unexpected:", unexpected[:8])
        raise SystemExit(
            "checkpoint keys do not match the reconstructed model. Compare with "
            "`--inspect` and align models_pose/ (and the weight_converter "
            "mapping) with your model's module hierarchy."
        )
    model.to(device).float()
    if not already_fused:
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
    ap.add_argument(
        "--inspect",
        action="store_true",
        help="print the raw checkpoint's parameter keys+shapes and exit "
        "(does not require matching the reconstructed model)",
    )
    args = ap.parse_args()

    device = torch.device(args.device)

    if args.inspect:
        state = _load_checkpoint(args.weights, device)
        for k in sorted(state):
            v = state[k]
            shape = tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
            print(f"{k}\t{shape}")
        print(f"# total {len(state)} checkpoint entries")
        return

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
