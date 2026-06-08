#!/usr/bin/env python3
"""Phase C parity: nntrainer runtime DDTree 32-node tree vs python ddtree_ref,
on the SAME real gemma4 draft logits.

Feed the nntrainer runtime's OWN block-0 draft logits (dumped by runDDTreeDump
under env NNTR_DDTREE_LOGITS) into the python reference build/compile/follow
functions and assert every tree field is element-by-element IDENTICAL to what
the nntrainer runtime produced for that same block (from blocks.json).

Both sides consume the IDENTICAL logits, so the tree-build is deterministic and
the comparison must be EXACT. Comparisons are NEVER loosened. Exits non-zero on
any mismatch.

Usage:
  compare_gemma4_tree.py <draft0.bin> <blocks.json>
The sidecar <draft0.bin>.meta.json supplies HORIZON/NUM_VOCAB/BUDGET/root/pos.
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import torch

# Import the standalone python reference (pure heap+numpy+torch, no model load).
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
    """Return (index, a_val, b_val) of the first differing element, else None."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return (i, a[i], b[i])
    if len(a) != len(b):
        return ("len", len(a), len(b))
    return None


def report(name: str, a, b) -> bool:
    """Element-by-element exact comparison. Returns True iff identical."""
    d = first_diff(list(a), list(b))
    if d is None:
        print(f"[OK]   {name:24s} len={len(a)}")
        return True
    if d[0] == "len":
        print(f"[DIFF] {name:24s} LENGTH py={d[1]} nntr={d[2]}")
    else:
        print(f"[DIFF] {name:24s} first idx={d[0]} py={d[1]} nntr={d[2]} "
              f"(py_len={len(a)} nntr_len={len(b)})")
    return False


def reconstruct_visibility(parents):
    """Deterministic visibility from parents (same recurrence as build_ddtree_tree
    / ddtree_ref): visibility[i, :i] = visibility[parent[i], :i]; visibility[i,i]=1.
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


def main() -> None:
    if len(sys.argv) != 3:
        fail("usage: compare_gemma4_tree.py <draft0.bin> <blocks.json>")
    draft_path, blocks_path = sys.argv[1], sys.argv[2]
    meta_path = draft_path + ".meta.json"

    if not os.path.exists(meta_path):
        fail(f"missing sidecar {meta_path}")
    meta = json.load(open(meta_path))
    HORIZON = int(meta["HORIZON"])
    NUM_VOCAB = int(meta["NUM_VOCAB"])
    BUDGET = int(meta["BUDGET"])
    root = int(meta["root"])
    pos = int(meta["pos"])
    past_length = int(meta["pastLength"])
    print(f"[meta] HORIZON={HORIZON} NUM_VOCAB={NUM_VOCAB} BUDGET={BUDGET} "
          f"root={root} pos={pos} pastLength={past_length}")

    # 1) Load the raw fp32 draft logits the runtime fed into buildTree.
    raw = np.fromfile(draft_path, dtype=np.float32)
    expected = HORIZON * NUM_VOCAB
    if raw.size != expected:
        fail(f"logits size {raw.size} != HORIZON*NUM_VOCAB {expected}")
    draft_np = raw.reshape(HORIZON, NUM_VOCAB)
    draft_logits = torch.from_numpy(draft_np)

    # 2) Load the nntrainer block-0 record (first JSON object).
    blocks = json.load(open(blocks_path))
    if not blocks:
        fail("blocks.json is empty")
    b0 = blocks[0]
    if int(b0["root"]) != root or int(b0["pos"]) != pos:
        fail(f"block0 root/pos ({b0['root']}/{b0['pos']}) != sidecar "
             f"({root}/{pos})")
    nntr_parents = list(b0["parents"])            # len cl, parents[0] == -1
    nntr_node_tokens = list(b0["node_tokens"])    # len cl, [0] == root (verify_input_ids)
    nntr_node_depths = list(b0["node_depths"])    # len cl-1, EXCLUDES root
    nntr_posterior = list(b0["posterior"])        # len cl
    nntr_acc = list(b0["acc"])                    # accepted_indices
    nntr_next = int(b0["next"])                   # next_token
    cl = len(nntr_parents)

    # 3) Build the tree from the IDENTICAL logits via the python reference.
    (py_node_token_ids, py_node_depths, py_parents, py_child_maps,
     py_visibility, _) = build_ddtree_tree(draft_logits, budget=BUDGET)
    py_node_token_ids = py_node_token_ids.tolist()
    py_node_depths = py_node_depths.tolist()
    py_current_length = 1 + len(py_node_token_ids)

    # 4) Compile (verify_input_ids / verify_position_ids / mask). Match the real
    #    ddtree_ref.compile_ddtree_tree signature: it writes into preallocated
    #    buffers and returns slices.
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

    # 5) Follow the verified tree with the nntrainer posterior (so accept/next
    #    depend only on tree topology, isolating the algorithm from model
    #    numerics — both already consumed the identical logits).
    nntr_posterior_t = torch.tensor([nntr_posterior], dtype=torch.long)
    py_accepted, py_next = follow_verified_tree(py_child_maps, nntr_posterior_t)

    # ---- element-by-element comparison ----
    print()
    all_ok = True

    # current_length / node_count
    cl_ok = (comp_cl == cl and py_current_length == cl)
    print(f"[{'OK' if cl_ok else 'DIFF'}]   {'current_length':24s} "
          f"py={py_current_length} compiled={comp_cl} nntr={cl}")
    all_ok &= cl_ok

    # verify_input_ids (root-prepended node_token_ids) vs nntr node_tokens
    all_ok &= report("verify_input_ids", py_verify_input_ids, nntr_node_tokens)

    # node_depths: nntr excludes root; python node_depths also excludes root.
    all_ok &= report("node_depths", py_node_depths, nntr_node_depths)

    # parents (len cl, parents[0] == -1)
    all_ok &= report("parents", py_parents, nntr_parents)

    # verify_position_ids: nntrainer does not dump it; it is pos + depth with the
    # root at `pos`. Build the expected from nntr (pos + node_depths, root=pos)
    # and require python's compiled position ids to match exactly.
    nntr_expected_pos = [pos] + [pos + d for d in nntr_node_depths]
    all_ok &= report("verify_position_ids", py_verify_position_ids,
                     nntr_expected_pos)

    # visibility: nntrainer does not dump it; reconstruct nntrainer's expected
    # visibility deterministically from its parents and compare (flattened) to
    # python's visibility. This still tests parity (visibility == f(parents)).
    nntr_vis = reconstruct_visibility(nntr_parents)
    py_vis_flat = py_visibility.to(torch.int32).flatten().tolist()
    nntr_vis_flat = [v for row in nntr_vis for v in row]
    all_ok &= report("visibility", py_vis_flat, nntr_vis_flat)

    # accepted_indices vs acc; next_token vs next
    all_ok &= report("accepted_indices", py_accepted, nntr_acc)

    nt_ok = (py_next == nntr_next)
    print(f"[{'OK' if nt_ok else 'DIFF'}]   {'next_token':24s} "
          f"py={py_next} nntr={nntr_next}")
    all_ok &= nt_ok

    print()
    if all_ok:
        print("GEMMA4 TREE PARITY OK")
        sys.exit(0)
    else:
        print("GEMMA4 TREE PARITY FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
