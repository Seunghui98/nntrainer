#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
@file   op_time_report.py
@date   29 Sep 2026
@brief  [#150] Per-op-kind decode table from the NNTR_OP_TIME=1 lines of
        one nntrainer_causallm log (plan docs/plans/150-cpu-decode.md 3.1)
@author dlwlzzero <dlwlzzero@gmail.com>
@bug    No known bugs except for NYI items

usage: op_time_report.py <log> [--base <log>]

Reads `[OP-TIME] node|step|moe` lines, groups the nodes into op kinds and
prints ms/token, share of the token, MB/token of weights read, GB/s and
avg/min (the per-token mean over the sum of the per-node minima: a kind
whose mean runs far above its best call is waiting, not computing).
`unattributed` is token time minus every timed row; it is what no timer
covers (graph walk, KV bind, PPL scoring, the loop). --base adds a column
with this log's ms/token minus the base log's.

Exit 1 when the log has no step line, the node call counts disagree with
the step token count, or the timed rows exceed the token time by more
than 2 % (a double count). A one-token prefill (a one-token prompt)
reads as a decode call to the timer and trips the first check.
"""
import re
import sys
from collections import OrderedDict

# ponytail: LFM2.5-8B-A1B's router, [2048 x 32] FP32. The lfm2_moe node's
# weights are the experts, which the DSP reads, not the ARM; another model
# needs its own number here. The embedding node reads one row of its
# (tied) table, so it counts 0 bytes.
ROUTER_BYTES = 2048 * 32 * 4

KINDS = [
    "FC conv in_proj", "FC conv out_proj", "FC attn qkv (+q/k norm)",
    "FC attn o", "FC dense FFN (+swiglu)", "conv block (fused)",
    "attention (mha_core)", "conv1d + gate",
    "RMSNorm", "residual add", "MoE CPU part", "MoE wait", "lm_head",
    "embedding", "sampling", "register", "other", "unattributed",
]


def kind_of(name, typ):
    """Op kind of one node, by type first and name suffix second."""
    if typ == "fully_connected":
        if name.endswith("_conv_in_proj"):
            return "FC conv in_proj"
        if name.endswith("_conv_out_proj"):
            return "FC conv out_proj"
        if name.endswith("_attention_out"):
            return "FC attn o"
        if re.search(r"_ffn_(up|gate|down)$", name) or "dense_ffn" in name:
            return "FC dense FFN (+swiglu)"
        return "other"
    if typ in ("dense_ffn", "swiglu"):
        return "FC dense FFN (+swiglu)"
    if typ == "conv_block":
        return "conv block (fused)"
    if typ == "qkv_layer":
        return "FC attn qkv (+q/k norm)"
    if typ == "mha_core":
        return "attention (mha_core)"
    if typ in ("causal_conv1d", "custom_multiply", "split"):
        return "conv1d + gate"
    if typ == "rms_norm":
        return "RMSNorm"
    if typ in ("addition", "residual_add"):
        return "residual add"
    if typ in ("tie_word_embeddings", "embedding_layer", "lm_head"):
        return "embedding" if name.startswith("embedding") else "lm_head"
    return "other"


def fields(line):
    """key=value pairs of one [OP-TIME] line."""
    return dict(kv.split("=", 1) for kv in line.split()[2:] if "=" in kv)


def parse(path):
    """-> (per-kind [us, bytes, min_us] over the run, tokens, token_us)."""
    nodes, steps, moe = [], [], []
    with open(path, errors="replace") as f:
        for line in f:
            m = re.search(r"\[OP-TIME\] (node|step|moe) .*", line)
            if not m:
                continue
            kv = fields(m.group(0))
            {"node": nodes, "step": steps, "moe": moe}[m.group(1)].append(kv)
    if not steps:
        sys.exit(f"{path}: no [OP-TIME] step line (NNTR_OP_TIME=1 unset?)")
    tokens = sum(int(s["tokens"]) for s in steps)
    token_us = sum(float(s["token_us"]) for s in steps)
    rows = OrderedDict((k, [0.0, 0, 0.0]) for k in KINDS)
    for n in nodes:
        if int(n["calls"]) != tokens:
            sys.exit(f"{path}: node {n['name']} calls={n['calls']} != "
                     f"step tokens={tokens}")
        us, mn, wb = float(n["sum_us"]), float(n["min_us"]), int(n["wbytes"])
        if n["type"] in ("lfm2_moe", "lfm2_moe_pool"):
            rows["MoE CPU part"][0] += us
            rows["MoE CPU part"][1] += ROUTER_BYTES
            continue  # no avg/min: its minimum holds the call
        k = kind_of(n["name"], n["type"])
        r = rows[k]
        r[0] += us
        r[1] += 0 if k == "embedding" else wb
        r[2] += mn
    call_us = sum(float(m["call_us"]) for m in moe)
    rows["MoE CPU part"][0] -= call_us
    rows["MoE wait"][0] = call_us
    rows["sampling"][0] = sum(float(s["sample_us"]) for s in steps)
    rows["register"][0] = sum(float(s["register_us"]) for s in steps)
    timed = sum(r[0] for r in rows.values())
    rows["unattributed"][0] = token_us - timed
    if timed > token_us * 1.02:
        sys.exit(f"{path}: timed rows {timed:.0f} us > token {token_us:.0f} "
                 f"us + 2 % (double count)")
    return rows, tokens, token_us


def main(argv):
    if len(argv) not in (2, 4) or (len(argv) == 4 and argv[2] != "--base"):
        sys.exit(__doc__.split("usage: ")[1].split("\n")[0])
    rows, tokens, token_us = parse(argv[1])
    base = parse(argv[3]) if len(argv) == 4 else None
    ms_tok = token_us / tokens / 1e3
    print(f"[OP-TIME] {argv[1]}: tokens={tokens} token={ms_tok:.3f} ms "
          f"({1e3 / ms_tok:.2f} tok/s)")
    head = f"{'kind':<26}{'ms/token':>10}{'share':>8}{'MB/token':>10}" \
        f"{'GB/s':>8}{'avg/min':>9}"
    print(head + ("   d ms vs base" if base else ""))
    for k, (us, wb, mn) in rows.items():
        ms = us / tokens / 1e3
        mb = wb / 1e6
        gbs = f"{mb / ms:8.1f}" if wb and ms > 0 else f"{'-':>8}"
        am = f"{us / tokens / mn:9.2f}" if mn > 0 else f"{'-':>9}"
        line = f"{k:<26}{ms:10.3f}{100 * us / token_us:7.1f}%{mb:10.1f}" \
            f"{gbs}{am}"
        if base:
            line += f"   {ms - base[0][k][0] / base[1] / 1e3:+.3f}"
        print(line)
    print(f"{'total':<26}{ms_tok:10.3f}{100.0:7.1f}%")


if __name__ == "__main__":
    main(sys.argv)
