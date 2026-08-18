# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
## @file verify_encoder.py
## @brief Compare an nntrainer PE-Lang-L14-448 encoder/decoder dump against a
##        golden .npy tensor and apply the PELANG_L14_448_SUPPORT_PLAN.md
##        gate thresholds (G2/G3/G4/G5/G6).
##
## Unlike res/siglip2-encoder/verify_encoder.py, this does NOT regenerate the
## golden tensors itself -- the PE-Lang golden set
## (pixel.npy, s0_patch_embed.npy, s1_pos_embed.npy, s2_norm_pre.npy,
## s3_block0.npy, encoder_features.npy, encoder_hidden.npy,
## decoder_init_logits.npy) is produced by the external, timm-dependent
## dump_pelang_golden.py against the v3.0.0-C62 checkpoint (plan Phase 0).
## This script only does the comparison + gate-threshold judgement half, so
## it has no torch/timm dependency and runs against any two .npy files:
##
##   # G2 (per-stage) / G3 (final encoder) parity:
##   python3 verify_encoder.py --gate g3 \
##       --golden pelang_golden/encoder_hidden.npy \
##       --nntr   nntr_encoder_hidden.npy
##
##   # G4 (decoder 1-step) parity -- also checks the expected argmax token:
##   python3 verify_encoder.py --gate g4 --expected-argmax 2048 \
##       --golden pelang_golden/decoder_init_logits.npy \
##       --nntr   nntr_decoder_init_logits.npy
import argparse
import sys

import numpy as np

# (min_cos, max_rel_l2, max_abs_diff) -- max_abs_diff is None when the plan
# doesn't set one (G4/G5/G6 judge by cosine/rel-L2/token-match only).
GATES = {
    "g2": (0.9999, None, 1e-3),  # per-stage tap (s0/s1/s2/s3)
    "g3": (0.9999, 1e-3, None),  # x86 fp32 encoder vs encoder_hidden.npy
    "g4": (0.999, None, None),  # x86 fp32 decoder 1-step logits
    "g5": (0.999, 5e-2, None),  # ARM fp32 encoder (looser -- plan PR#4007 baseline)
    "g6": (0.999, None, None),  # ARM Q8_0 -- caption-level, cosine as a proxy
}


def cos_rel_l2_maxdiff(a: np.ndarray, b: np.ndarray):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: golden {a.shape} vs nntr {b.shape}")
    diff = np.abs(a - b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")
    rel_l2 = float(np.linalg.norm(a - b) / na) if na > 0 else float("nan")
    return cos, rel_l2, float(diff.max()), float(diff.mean())


def main():
    ap = argparse.ArgumentParser(
        description="PE-Lang-L14-448 nntrainer-vs-golden parity check "
        "(PELANG_L14_448_SUPPORT_PLAN.md gates G2-G6)."
    )
    ap.add_argument("--golden", required=True, help="golden .npy (PyTorch/timm reference)")
    ap.add_argument("--nntr", required=True, help="nntrainer-dumped .npy to check")
    ap.add_argument("--gate", choices=sorted(GATES), default="g3",
                    help="which plan gate's thresholds to apply (default: g3)")
    ap.add_argument("--min-cos", type=float, default=None,
                    help="override the gate's cosine threshold")
    ap.add_argument("--max-rel-l2", type=float, default=None,
                    help="override the gate's relative-L2 threshold")
    ap.add_argument("--max-abs-diff", type=float, default=None,
                    help="override the gate's max|diff| threshold")
    ap.add_argument("--expected-argmax", type=int, default=None,
                    help="for G4: also require golden/nntr argmax to equal this "
                         "token id (plan's greedy-decode check)")
    args = ap.parse_args()

    gate_cos, gate_rel_l2, gate_maxdiff = GATES[args.gate]
    min_cos = args.min_cos if args.min_cos is not None else gate_cos
    max_rel_l2 = args.max_rel_l2 if args.max_rel_l2 is not None else gate_rel_l2
    max_abs_diff = args.max_abs_diff if args.max_abs_diff is not None else gate_maxdiff

    golden = np.load(args.golden)
    nntr = np.load(args.nntr)
    cos, rel_l2, maxdiff, meandiff = cos_rel_l2_maxdiff(golden, nntr)

    print(f"golden shape={golden.shape} dtype={golden.dtype}")
    print(f"nntr   shape={nntr.shape} dtype={nntr.dtype}")
    print(f"cosine       = {cos:.10f}  (gate {args.gate}: >= {min_cos})")
    if max_rel_l2 is not None:
        print(f"rel L2       = {rel_l2:.6e}  (gate {args.gate}: <= {max_rel_l2})")
    if max_abs_diff is not None:
        print(f"max |diff|   = {maxdiff:.6e}  (gate {args.gate}: <= {max_abs_diff})")
    print(f"mean |diff|  = {meandiff:.6e}")

    ok = cos >= min_cos
    if max_rel_l2 is not None:
        ok = ok and (rel_l2 <= max_rel_l2)
    if max_abs_diff is not None:
        ok = ok and (maxdiff <= max_abs_diff)

    if args.expected_argmax is not None:
        golden_argmax = int(np.argmax(golden.reshape(-1)))
        nntr_argmax = int(np.argmax(nntr.reshape(-1)))
        argmax_ok = (golden_argmax == args.expected_argmax) and (
            nntr_argmax == args.expected_argmax
        )
        print(f"argmax       = golden {golden_argmax}, nntr {nntr_argmax} "
              f"(expected {args.expected_argmax})")
        ok = ok and argmax_ok

    print(f"[{args.gate.upper()}]", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
