#!/usr/bin/env python3
"""Decode-divergence diagnostic: tree-verify node_logits vs sequential (self-draft)
logits, per accepted node of a DDTree block.

Background. `runDDTreeDump` in causal_lm.cpp runs a SEQUENTIAL single-token greedy
self-draft (its per-step logits, dumped via NNTR_DDTREE_LOGITS, are the
"ground-truth" sequential logits) and then a masked tree-VERIFY forward (its
per-node logits, dumped via NNTR_DDTREE_NODELOGITS, whose per-row argmax is the
`posterior`). For an accepted node that is fed the SAME token at the SAME absolute
position as a sequential draft step, the two logit rows must be ~identical up to
fp/quant noise -- and crucially their ARGMAX must match. When the committed token
differs from greedy, this harness pinpoints whether it is:

  (A) a NEAR-TIE numerical flip -- verify and sequential logits coincide to within
      the fp/quant noise floor, and the two competing tokens are within that noise
      of each other (argmax legitimately flips). Speculative decode is "lossy" only
      on genuine ties; acceptable.
  (B) a real BUG -- verify and sequential produce materially different logits for
      the same token/position (large max_abs_diff, or one token strongly wins
      sequentially but the other strongly wins under verify).

IMPORTANT softcap detail. gemma3/gemma4 apply final_logit_softcapping (=30): the
graph output (hence the SEQUENTIAL draft logits) is softcapped, but the verify dump
in TieWordEmbedding::incremental_forwarding_lmhead writes the RAW lmhead logits
(pre-softcap, computed before the logit_softcapping layer; that layer's apply_rows=1
caps only row 0 anyway). softcap(x)=cap*tanh(x/cap) is strictly monotonic, so it
NEVER changes an argmax -- but to compare logit ROWS we must apply the SAME softcap
to the verify rows first. This harness does that (--softcap, default 30; pass 0 to
disable for non-gemma models whose dump is already final).

Per accepted node it reports: depth, fed token (verify vs the sequential chain),
verify argmax, sequential argmax, max|softcap(verify)-seq| over the full vocab, and
-- on an argmax mismatch -- the two competing tokens' values and gaps in both
forwards. A node fed a DIFFERENT input token than the sequential chain (i.e. it is
already downstream of a flip) is marked INPUT_DIFF and skipped from the noise floor.

Usage:
  compare_verify_vs_sequential.py <dump_prefix> <block_idx> [--softcap V]
Expects <prefix>.blocks.json, <prefix>.node.<block>.bin(+.meta.json),
        <prefix>.draft.<block>.bin(+.meta.json).
"""
from __future__ import annotations
import argparse
import json

import numpy as np


def topk(row, k=5):
    idx = np.argsort(row)[::-1][:k]
    return [(int(i), round(float(row[i]), 4)) for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("block", type=int)
    ap.add_argument("--softcap", type=float, default=30.0,
                    help="final_logit_softcapping value (0 to disable)")
    args = ap.parse_args()
    prefix, block, cap = args.prefix, args.block, args.softcap

    def softcap(x):
        return x if cap <= 0 else cap * np.tanh(x / cap)

    blk = json.load(open(prefix + ".blocks.json"))[block]
    node_tokens = blk["node_tokens"]
    node_depths = [0] + list(blk["node_depths"])  # prepend root depth 0
    draft_am = blk["draft_am"]
    acc = blk["acc"]
    pos = blk["pos"]
    root = blk["root"]
    cl = len(node_tokens)

    nm = json.load(open(f"{prefix}.node.{block}.bin.meta.json"))
    dm = json.load(open(f"{prefix}.draft.{block}.bin.meta.json"))
    V = int(nm["NUM_VOCAB"])
    H = int(dm["HORIZON"])
    node_logits = np.fromfile(
        f"{prefix}.node.{block}.bin", dtype=np.float32).reshape(cl, V)
    draft = np.fromfile(
        f"{prefix}.draft.{block}.bin", dtype=np.float32).reshape(H, V)

    print(f"=== BLOCK {block}: pos={pos} root={root} cl={cl} acc={acc} "
          f"softcap={cap} ===")
    noise = []
    for node in acc:
        dep = node_depths[node]
        if dep >= H:
            continue  # no sequential draft step at this depth
        vtok = node_tokens[node]
        seq_fed = root if dep == 0 else draft_am[dep - 1]
        v_raw = node_logits[node]
        v_cap = softcap(v_raw)
        s_row = draft[dep]
        v_arg, s_arg = int(np.argmax(v_cap)), int(np.argmax(s_row))
        if vtok != seq_fed:
            print(f"node {node} d={dep} fed={vtok} (seq_fed={seq_fed}) "
                  f"INPUT_DIFF (downstream of a flip) -- skipped")
            continue
        md = float(np.max(np.abs(v_cap - s_row)))
        noise.append(md)
        flag = "  <-- ARGMAX MISMATCH" if v_arg != s_arg else ""
        print(f"node {node} d={dep} fed={vtok} v_arg={v_arg} s_arg={s_arg} "
              f"max|softcap(v)-seq|={md:.4f}{flag}")
        if v_arg != s_arg:
            a, b = s_arg, v_arg
            print(f"     token {a} (seq win): softcap(v)={v_cap[a]:+.5f} "
                  f"seq={s_row[a]:+.5f}")
            print(f"     token {b} (verify win): softcap(v)={v_cap[b]:+.5f} "
                  f"seq={s_row[b]:+.5f}")
            print(f"     verify gap(b-a)={v_cap[b]-v_cap[a]:+.5f}  "
                  f"seq gap(a-b)={s_row[a]-s_row[b]:+.5f}")
            print(f"     verify top5: {topk(v_cap)}")
            print(f"     seq    top5: {topk(s_row)}")
    if noise:
        print(f"--- comparable nodes (same input token)={len(noise)} "
              f"noise floor max|softcap(v)-seq|: min={min(noise):.4f} "
              f"max={max(noise):.4f} mean={float(np.mean(noise)):.4f} ---")


if __name__ == "__main__":
    main()
