#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay an NNTR_MOE_TRACE routing trace against expert-cache policies.

Doc 52 section 10.10. The HTP path keeps experts in one pool of slots shared
by every MoE layer (capacity = C x layers), and a layer call needs all of its
routed experts resident at once, so each policy here pins the call's experts
and evicts only outside them -- the constraint the device has. The routing is
the same whatever the cache does, so one trace from a resident run answers
every C and every policy:

  ours     LRU, then recency refreshed from each token's top-(k+5), walked
           back to front -- exactly Lfm2MoELayer + ExpertLru
  lru      LRU without the refresh
  lfu      least frequently used so far, LRU among equals
  random   uniform among the unpinned (seeded)
  belady   evicts the expert whose next use is farthest away: the best any
           policy can do on this trace, the ceiling for policy work

Usage:
  NNTR_MOE_TRACE=/data/local/tmp/moe_trace.txt ./nntrainer_causallm <model>
  python3 tools/moe_expert_cache_sim.py moe_trace.txt --cache 2 4 8 16
  python3 tools/moe_expert_cache_sim.py --selftest

The TPS column is arithmetic, not a measurement: 1000 / (base_ms + layers x
misses_per_call x miss_ms), with base_ms the resident decode token and
miss_ms one synchronous miss, both measured in doc 52 sections 10.7-10.9.
"""
import argparse
import random as _random
import sys
from collections import OrderedDict


def parse(lines):
    """-> list of (layer, tokens, routed experts, top-(k+5) per token)."""
    calls = []
    for n, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            head, routed, ext = line.split("|")
            layer, tokens = (int(v) for v in head.split())
            calls.append((layer, tokens, [int(v) for v in routed.split()],
                          [int(v) for v in ext.split()]))
        except ValueError:
            raise SystemExit(f"trace line {n} is not '<layer> <tokens> | "
                             f"<experts> | <top-k+5>': {line[:80]!r}")
    return calls


def simulate(calls, capacity, policy, seed=0):
    """-> (decode misses, decode calls, prefill misses). Keys are
    (layer, expert). Raises when one call needs more than the pool holds."""
    order = OrderedDict()   # resident keys, least recent first
    freq = {}
    rng = _random.Random(seed)
    next_use = {}
    if policy == "belady":  # positions at which each key is needed
        for i, (layer, _, routed, _) in enumerate(calls):
            for e in routed:
                next_use.setdefault((layer, e), []).append(i)
        ptr = {k: 0 for k in next_use}

    def upcoming(k, i):
        uses = next_use.get(k, [])
        p = ptr[k]
        while p < len(uses) and uses[p] <= i:
            p += 1
        ptr[k] = p
        return uses[p] if p < len(uses) else float("inf")

    dec_miss = dec_calls = pre_miss = 0
    for i, (layer, tokens, routed, ext) in enumerate(calls):
        need = [(layer, e) for e in routed]
        if len(need) > capacity:
            raise ValueError(f"call {i} needs {len(need)} experts, pool "
                             f"holds {capacity}")
        pinned = set(need)
        misses = [k for k in need if k not in order]
        for k in need:
            freq[k] = freq.get(k, 0) + 1
            if k in order:
                order.move_to_end(k)
        excess = len(order) + len(misses) - capacity
        while excess > 0:
            cand = [k for k in order if k not in pinned]
            if policy in ("ours", "lru"):
                victim = cand[0]
            elif policy == "lfu":
                victim = min(cand, key=lambda k: freq[k])  # stable: LRU tie
            elif policy == "random":
                victim = rng.choice(cand)
            elif policy == "belady":
                victim = max(cand, key=lambda k: upcoming(k, i))
            else:
                raise ValueError(policy)
            del order[victim]
            excess -= 1
        for k in misses:
            order[k] = None
        if policy == "ours":
            for e in reversed(ext):
                k = (layer, e)
                if k in order:
                    order.move_to_end(k)
        if tokens == 1:
            dec_calls += 1
            dec_miss += len(misses)
        else:
            pre_miss += len(misses)
    return dec_miss, dec_calls, pre_miss


POLICIES = ["ours", "lru", "lfu", "random", "belady"]


def report(calls, caches, base_ms, miss_ms):
    layers = len({c[0] for c in calls})
    routed = sum(len(c[2]) for c in calls if c[1] == 1)
    decode_calls = sum(1 for c in calls if c[1] == 1)
    per_call = routed / decode_calls if decode_calls else 0
    print(f"trace: {len(calls)} calls, {layers} layers, {decode_calls} decode "
          f"calls routing {per_call:.2f} experts each")
    print(f"{'C':>3} {'slots':>5} {'policy':>7} {'miss/decode call':>17} "
          f"{'hit %':>6} {'prefill misses':>15} {'TPS (arith)':>12}")
    for c in caches:
        cap = c * layers
        for p in POLICIES:
            try:
                dm, dc, pm = simulate(calls, cap, p)
            except ValueError as err:
                print(f"{c:>3} {cap:>5} {p:>7}  -- {err}")
                break
            mpc = dm / dc if dc else 0.0
            hit = 100.0 * (1 - dm / routed) if routed else 0.0
            tps = 1000.0 / (base_ms + layers * mpc * miss_ms)
            print(f"{c:>3} {cap:>5} {p:>7} {mpc:>17.2f} {hit:>6.1f} "
                  f"{pm:>15} {tps:>12.1f}")


def selftest():
    # One layer, capacity 2, single-expert decode calls 0 1 2 0 1 2.
    calls = [(0, 1, [e], [e]) for e in (0, 1, 2, 0, 1, 2)]
    assert simulate(calls, 2, "lru")[0] == 6, "LRU thrashes a cyclic 3 in 2"
    assert simulate(calls, 2, "belady")[0] == 4, "Belady keeps what is next"
    # Pinning: a call's own experts are never the victim.
    calls = [(0, 1, [0, 1], []), (0, 1, [1, 2], [])]
    assert simulate(calls, 2, "lru")[0] == 3  # 0 goes, 1 stays pinned
    try:
        simulate([(0, 1, [0, 1, 2], [])], 2, "lru")
        raise AssertionError("an over-capacity call must raise")
    except ValueError:
        pass
    # The refresh: top-(k+5) walked back to front leaves ext[0] most recent.
    # Resident {0, 1}; call routes 1 with ext [0, 1]; then 2 arrives. Without
    # the refresh 0 is oldest and goes; with it 1 is oldest and goes.
    calls = [(0, 1, [0], []), (0, 1, [1], [0, 1]), (0, 1, [2], []),
             (0, 1, [0], [])]
    assert simulate(calls, 2, "lru")[0] == 4
    assert simulate(calls, 2, "ours")[0] == 3
    # Prefill calls count apart from decode.
    assert simulate([(0, 2, [0, 1], [])], 2, "lru") == (0, 0, 2)
    # Parsing.
    assert parse(["3 1 | 4 7 | 7 4 9\n", "\n"]) == [(3, 1, [4, 7], [7, 4, 9])]
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace", nargs="?")
    ap.add_argument("--cache", type=int, nargs="+", default=[2, 4, 8, 16],
                    help="experts per layer (NNTR_MOE_CACHE_EXPERTS)")
    ap.add_argument("--base-ms", type=float, default=42.0,
                    help="resident decode token, ms (doc 52: 1000/23.8)")
    ap.add_argument("--miss-ms", type=float, default=1.37,
                    help="one synchronous miss, ms (doc 52 section 10.9)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.trace:
        ap.error("a trace file, or --selftest")
    with open(a.trace) as f:
        report(parse(f), a.cache, a.base_ms, a.miss_ms)


if __name__ == "__main__":
    sys.exit(main())
