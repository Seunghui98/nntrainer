#!/usr/bin/env python3
"""Phase C parity (multi-block): nntrainer runtime DDTree 32-node tree vs python
ddtree_ref, on the SAME real gemma4 draft logits, for EVERY block of a long
generation (not just block 0).

For each block i the nntrainer runtime dumps the exact [HORIZON, NUM_VOCAB] fp32
draft buffer it fed into buildTree to <prefix>.<i>.bin (+ .<i>.bin.meta.json).
This harness loops over every block, feeds that IDENTICAL buffer into the python
reference build/compile/follow, and asserts every tree field is element-by-element
IDENTICAL to what nntrainer produced for that same block (from blocks.json):
  parents, verify_input_ids(node_tokens), node_depths, verify_position_ids,
  visibility, accepted_indices(acc), next_token.

Both sides consume the IDENTICAL logits, so the tree-build is deterministic and
the comparison must be EXACT. Comparisons are NEVER loosened. Prints one line per
block ([block i] OK / [block i] DIFF field=... idx=... py=... nntr=...) and exits
non-zero if ANY block/field mismatches.

Usage:
  compare_gemma4_tree_all.py <draft_prefix> <blocks.json>
Per-block files are <draft_prefix>.<i>.bin and <draft_prefix>.<i>.bin.meta.json.
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ddtree_ref import (  # noqa: E402
    build_ddtree_tree,
    compile_ddtree_tree,
    follow_verified_tree,
)


def fail(msg: str) -> None:
    print(f"[FATAL] {msg}")
    sys.exit(2)


def first_diff(a, b):
    """Return (index, a_val, b_val) of the first differing element, else None.
    A length mismatch is reported as ('len', len_a, len_b)."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return (i, a[i], b[i])
    if len(a) != len(b):
        return ("len", len(a), len(b))
    return None


def reconstruct_visibility(parents):
    """Deterministic visibility from parents (same recurrence as ddtree_ref):
    visibility[i, :i] = visibility[parent[i], :i]; visibility[i,i]=1.
    parents[0] == -1 (root). Returns a python list-of-lists of 0/1 ints."""
    n = len(parents)
    vis = [[0] * n for _ in range(n)]
    vis[0][0] = 1
    for i in range(1, n):
        p = parents[i]
        for j in range(i):
            vis[i][j] = vis[p][j]
        vis[i][i] = 1
    return vis


def compare_block(i: int, draft_path: str, blk: dict) -> tuple[bool, str]:
    """Run the python reference on block i's dumped logits and compare every
    field element-by-element to nntrainer's record. Returns (ok, message). The
    message is either 'OK' or the FIRST differing field with idx/py/nntr."""
    meta_path = draft_path + ".meta.json"
    if not os.path.exists(draft_path):
        return False, f"missing draft bin {draft_path}"
    if not os.path.exists(meta_path):
        return False, f"missing sidecar {meta_path}"
    meta = json.load(open(meta_path))
    HORIZON = int(meta["HORIZON"])
    NUM_VOCAB = int(meta["NUM_VOCAB"])
    BUDGET = int(meta["BUDGET"])
    root = int(meta["root"])
    pos = int(meta["pos"])
    past_length = int(meta["pastLength"])

    # cross-check meta vs blocks.json
    if int(blk["root"]) != root or int(blk["pos"]) != pos:
        return False, (f"meta root/pos ({root}/{pos}) != blocks.json "
                       f"({blk['root']}/{blk['pos']})")

    # 1) raw fp32 draft logits the runtime fed into buildTree.
    raw = np.fromfile(draft_path, dtype=np.float32)
    expected = HORIZON * NUM_VOCAB
    if raw.size != expected:
        return False, f"logits size {raw.size} != HORIZON*NUM_VOCAB {expected}"
    draft_logits = torch.from_numpy(raw.reshape(HORIZON, NUM_VOCAB))

    # nntrainer record for this block
    nntr_parents = list(blk["parents"])          # len cl, parents[0] == -1
    nntr_node_tokens = list(blk["node_tokens"])  # len cl, [0]==root (verify_input_ids)
    nntr_node_depths = list(blk["node_depths"])  # len cl-1, EXCLUDES root
    nntr_posterior = list(blk["posterior"])      # len cl
    nntr_acc = list(blk["acc"])                  # accepted_indices
    nntr_next = int(blk["next"])                 # next_token
    cl = len(nntr_parents)

    # 2) build the tree from the IDENTICAL logits via the python reference.
    (py_node_token_ids, py_node_depths, py_parents, py_child_maps,
     py_visibility, _) = build_ddtree_tree(draft_logits, budget=BUDGET)
    py_node_token_ids = py_node_token_ids.tolist()
    py_node_depths = py_node_depths.tolist()
    py_current_length = 1 + len(py_node_token_ids)

    # 3) compile -> verify_input_ids / verify_position_ids / mask.
    dtype = torch.float32
    device = torch.device("cpu")
    vii_buf = torch.zeros((1, py_current_length), dtype=torch.long)
    vpi_buf = torch.zeros((1, py_current_length), dtype=torch.long)
    mask_buf = torch.zeros(
        (1, 1, py_current_length, past_length + py_current_length), dtype=dtype)
    tree_vis_buf = torch.zeros(
        (py_current_length, py_current_length), dtype=torch.bool)
    (verify_input_ids, verify_position_ids, _attn, _pl,
     comp_cl) = compile_ddtree_tree(
        root_token_id=torch.tensor(root, dtype=torch.long),
        start=pos,
        node_token_ids=torch.tensor(py_node_token_ids, dtype=torch.long),
        node_depths=torch.tensor(py_node_depths, dtype=torch.long),
        visibility_cpu=py_visibility,
        past_length=past_length,
        dtype=dtype,
        device=device,
        verify_input_ids_buffer=vii_buf,
        verify_position_ids_buffer=vpi_buf,
        attention_mask_buffer=mask_buf,
        tree_visibility_buffer=tree_vis_buf,
        previous_tree_start=0,
        previous_tree_length=0,
    )
    py_verify_input_ids = verify_input_ids[0].tolist()
    py_verify_position_ids = verify_position_ids[0].tolist()

    # 4) follow the verified tree with nntrainer's posterior for this block.
    nntr_posterior_t = torch.tensor([nntr_posterior], dtype=torch.long)
    py_accepted, py_next = follow_verified_tree(py_child_maps, nntr_posterior_t)

    # ---- element-by-element comparison (first failing field wins) ----
    # current_length / node_count
    if not (comp_cl == cl and py_current_length == cl):
        return False, (f"current_length py={py_current_length} "
                       f"compiled={comp_cl} nntr={cl}")

    nntr_expected_pos = [pos] + [pos + d for d in nntr_node_depths]
    nntr_vis = reconstruct_visibility(nntr_parents)
    py_vis_flat = py_visibility.to(torch.int32).flatten().tolist()
    nntr_vis_flat = [v for row in nntr_vis for v in row]

    checks = [
        ("parents", py_parents, nntr_parents),
        ("verify_input_ids", py_verify_input_ids, nntr_node_tokens),
        ("node_depths", py_node_depths, nntr_node_depths),
        ("verify_position_ids", py_verify_position_ids, nntr_expected_pos),
        ("visibility", py_vis_flat, nntr_vis_flat),
        ("accepted_indices", py_accepted, nntr_acc),
    ]
    for name, a, b in checks:
        d = first_diff(list(a), list(b))
        if d is not None:
            if d[0] == "len":
                return False, f"field={name} LENGTH py={d[1]} nntr={d[2]}"
            return False, (f"field={name} idx={d[0]} py={d[1]} nntr={d[2]}")

    if py_next != nntr_next:
        return False, f"field=next_token py={py_next} nntr={nntr_next}"

    return True, "OK"


def main() -> None:
    if len(sys.argv) != 3:
        fail("usage: compare_gemma4_tree_all.py <draft_prefix> <blocks.json>")
    prefix, blocks_path = sys.argv[1], sys.argv[2]

    blocks = json.load(open(blocks_path))
    if not blocks:
        fail("blocks.json is empty")
    n = len(blocks)
    print(f"[info] comparing {n} blocks (prefix={prefix})")

    all_ok = True
    for i in range(n):
        draft_path = f"{prefix}.{i}.bin"
        ok, msg = compare_block(i, draft_path, blocks[i])
        if ok:
            print(f"[block {i}] OK")
        else:
            print(f"[block {i}] DIFF {msg}")
            all_ok = False

    print()
    if all_ok:
        print(f"ALL {n} BLOCKS TREE PARITY OK")
        sys.exit(0)
    else:
        print(f"GEMMA4 MULTI-BLOCK TREE PARITY FAILED ({n} blocks)")
        sys.exit(1)


if __name__ == "__main__":
    main()
