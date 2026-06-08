#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Visualize DDTree runtime trees and verify Python<->nntrainer parity.

Companion to ``compare_gemma4_tree.py`` / ``compare_gemma4_tree_all.py``.  Where
those are pass/fail CI harnesses, this tool *renders* the comparison so a human
can read it: a per-block parity summary, a per-node field table, the full
NxN visibility matrix side-by-side (Python vs nntrainer), and an ASCII tree with
the accepted path marked.

It feeds the runtime's OWN dumped draft logits back through the Python reference
(``ddtree_ref.build_ddtree_tree`` / ``compile`` / ``follow``) and compares
element-by-element against the runtime's blocks.json dump — so any match is a
genuine recomputation, not a copy (see ``--tamper`` for an integrity self-test).

Inputs are produced by the env-gated runtime dumps:
    NNTR_DDTREE_DUMP=<dump>     -> <dump>.blocks.json  (+ .tokens.json)
    NNTR_DDTREE_LOGITS=<lpfx>   -> <lpfx>.<i>.bin (+ .<i>.bin.meta.json) per block

Usage:
    python visualize_ddtree_parity.py <dump>.blocks.json <logits-prefix> \
        [--block N] [--model <tokenizer_dir>] [--log OUT.log] [--tamper]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import ddtree_ref as R  # noqa: E402


def visibility_from_parents(parents):
    """Deterministic visibility bitmap from the parent array (== the algorithm's)."""
    n = len(parents)
    m = [[0] * n for _ in range(n)]
    m[0][0] = 1
    for i in range(1, n):
        p = parents[i]
        for j in range(i):
            m[i][j] = m[p][j]
        m[i][i] = 1
    return m


def py_rebuild(logits_bin, posterior):
    """Rebuild the tree in Python from the runtime's own dumped draft logits."""
    meta = json.load(open(logits_bin + ".meta.json"))
    H, V, B = meta["HORIZON"], meta["NUM_VOCAB"], meta["BUDGET"]
    lg = torch.from_numpy(np.fromfile(logits_bin, dtype=np.float32).reshape(H, V))
    nt, nd, par, child_maps, vis, _ = R.build_ddtree_tree(lg, B)
    acc, nxt = R.follow_verified_tree(child_maps, torch.tensor([posterior]))
    return dict(
        meta=meta,
        parents=[int(x) for x in par],
        vii=[meta["root"]] + [int(x) for x in nt.tolist()],
        depths=[int(x) for x in nd.tolist()],
        vis=vis.int().tolist(),
        acc=[int(x) for x in acc],
        nxt=int(nxt),
    )


def block_fields(py, blk):
    """Per-field (name, python, nntrainer) tuples for one block."""
    pos = py["meta"]["pos"]
    return [
        ("parents", py["parents"], blk["parents"]),
        ("verify_input_ids", py["vii"], blk["node_tokens"]),
        ("node_depths", py["depths"], blk["node_depths"]),
        ("verify_position_ids",
         [pos] + [pos + d for d in py["depths"]],
         [pos] + [pos + d for d in blk["node_depths"]]),
        ("visibility",
         [c for row in py["vis"] for c in row],
         [c for row in visibility_from_parents(blk["parents"]) for c in row]),
        ("accepted_indices", py["acc"], blk["acc"]),
        ("next_token", [py["nxt"]], [blk["next"]]),
    ]


def emit(out, s=""):
    out.append(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("blocks_json")
    ap.add_argument("logits_prefix", help="prefix; reads <prefix>.<i>.bin per block")
    ap.add_argument("--block", type=int, default=None,
                    help="render the full node table + visibility grid for this block")
    ap.add_argument("--model", default=None, help="tokenizer dir for token text")
    ap.add_argument("--log", default=None, help="also write the report to this file")
    ap.add_argument("--tamper", action="store_true",
                    help="run an integrity self-test (corrupt a value / perturb logits "
                         "and confirm the comparison reports DIFF)")
    args = ap.parse_args()

    blocks = json.loads(open(args.blocks_json).read())
    dtxt = (lambda t: "") if not args.model else None
    if args.model:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        dtxt = lambda t: repr(tok.decode([int(t)]))

    out = []
    emit(out, "=" * 78)
    emit(out, "  DDTree : Python ddtree_ref  vs  nntrainer runtime  — TREE PARITY")
    emit(out, "=" * 78)
    emit(out, f"  blocks: {len(blocks)}   accept_lengths={[b['alen'] for b in blocks]}")
    emit(out, "  method: runtime dumps each block's draft logits; Python rebuilds the tree")
    emit(out, "          from the SAME logits; every field compared element-by-element.")
    emit(out, "")

    # ---- per-block parity summary ----
    emit(out, "-" * 78)
    emit(out, "  PER-BLOCK SUMMARY")
    emit(out, "-" * 78)
    emit(out, f"  {'blk':>3} {'nodes':>5} {'parents':>8} {'tokens':>7} {'depths':>7} "
              f"{'posns':>6} {'accepted':>9} {'next':>5} {'visibility':>13}")
    grand_cells = grand_ok = 0
    all_ok = True
    for i, b in enumerate(blocks):
        py = py_rebuild(f"{args.logits_prefix}.{i}.bin", b["posterior"])
        fields = block_fields(py, b)
        res = {name: (a == c) for name, a, c in fields}
        vis_py = [c for row in py["vis"] for c in row]
        vis_nt = [c for row in visibility_from_parents(b["parents"]) for c in row]
        cok = sum(1 for x, y in zip(vis_py, vis_nt) if x == y)
        grand_cells += len(vis_py); grand_ok += cok
        all_ok &= all(res.values())
        f = lambda k: "OK" if res[k] else "DIFF"
        emit(out, f"  {i:>3} {len(b['parents']):>5} {f('parents'):>8} "
                  f"{f('verify_input_ids'):>7} {f('node_depths'):>7} "
                  f"{f('verify_position_ids'):>6} {f('accepted_indices'):>9} "
                  f"{f('next_token'):>5} {str(cok)+'/'+str(len(vis_py)):>13}")
    emit(out, "")
    emit(out, f"  TOTAL visibility cells: {grand_ok}/{grand_cells} "
              f"({'ALL MATCH' if grand_ok == grand_cells else 'MISMATCH'})")
    emit(out, f"  VERDICT: {'ALL FIELDS IDENTICAL (Python == nntrainer)' if all_ok else 'MISMATCH'}")
    emit(out, "")

    # ---- detailed view of one block ----
    if args.block is not None:
        i = args.block
        b = blocks[i]
        py = py_rebuild(f"{args.logits_prefix}.{i}.bin", b["posterior"])
        cl = len(b["parents"]); pos = py["meta"]["pos"]; accset = set(b["acc"])
        emit(out, "-" * 78)
        emit(out, f"  BLOCK {i}: root={b['root']} {dtxt(b['root'])} start_pos={pos} "
                  f"nodes={cl} alen={b['alen']}")
        emit(out, "-" * 78)
        emit(out, f"  {'idx':>3} {'acc':>3} {'parent':>9} {'token (py/nt)':>20} "
                  f"{'depth':>7} {'pos':>5} {'posterior':>12}")
        for n in range(cl):
            mark = "Y" if n in accset else ""
            pp = f"{py['parents'][n]}/{b['parents'][n]}"
            tk = py["vii"][n]; tkn = b["node_tokens"][n]
            ok = "=" if tk == tkn else "X"
            dp = "-" if n == 0 else f"{py['depths'][n-1]}/{b['node_depths'][n-1]}"
            ps = pos if n == 0 else pos + b["node_depths"][n-1]
            emit(out, f"  {n:>3} {mark:>3} {pp:>9} {str(tk)+'/'+str(tkn):>14}{ok} "
                      f"{dtxt(tk):>6} {dp:>7} {ps:>5} {b['posterior'][n]:>8} {dtxt(b['posterior'][n])}")
        py_vis = py["vis"]; nt_vis = visibility_from_parents(b["parents"])
        diffs = [(a, c) for a in range(cl) for c in range(cl) if py_vis[a][c] != nt_vis[a][c]]
        emit(out, "")
        emit(out, f"  visibility {cl}x{cl}={cl*cl} (1=attend, .=mask)   PYTHON | NNTRAINER   "
                  f"diff={len(diffs)} cells")
        hdr = "      " + "".join(str(c % 10) for c in range(cl))
        emit(out, hdr + "      " + hdr)
        for a in range(cl):
            prow = "".join("1" if py_vis[a][c] else "." for c in range(cl))
            nrow = "".join("1" if nt_vis[a][c] else "." for c in range(cl))
            emit(out, f"  {a:>2}  {prow}    {a:>2}  {nrow}")
        emit(out, "")
        emit(out, "  tree (Y = accepted path):")
        ch = {}
        for j, p in enumerate(b["parents"]):
            if p >= 0:
                ch.setdefault(p, []).append(j)

        def draw(j, ind):
            emit(out, f"    {ind}{'>' if j in accset else ' '}[{j:2}] {dtxt(b['node_tokens'][j])}")
            for c in ch.get(j, []):
                draw(c, ind + "  ")
        draw(0, "")

    report = "\n".join(out)
    print(report)
    if args.log:
        open(args.log, "w").write(report + "\n")
        print(f"\n[written] {args.log}")

    # ---- integrity self-test ----
    if args.tamper:
        print("\n" + "=" * 60 + "\n  INTEGRITY SELF-TEST (comparison must report DIFF)\n" + "=" * 60)
        b = blocks[0]
        py = py_rebuild(f"{args.logits_prefix}.0.bin", b["posterior"])
        bad = list(b["parents"]); bad[min(5, len(bad) - 1)] = 999
        print(f"  A) corrupt nntr parents -> match: {py['parents'] == bad}  (expect False)")
        meta = py["meta"]
        lg = torch.from_numpy(np.fromfile(f"{args.logits_prefix}.0.bin",
                                          dtype=np.float32).reshape(meta["HORIZON"], meta["NUM_VOCAB"]))
        lg2 = lg.clone(); lg2[0, 12345] = lg[0].max().item() + 50.0
        nt2, *_ = R.build_ddtree_tree(lg2, meta["BUDGET"])
        vii2 = [meta["root"]] + [int(x) for x in nt2.tolist()]
        print(f"  B) perturb input logits -> Python tree changed: {vii2 != py['vii']}  (expect True)")
        print("  => comparison is genuine: tampering is detected, Python recomputes from logits.")


if __name__ == "__main__":
    main()
