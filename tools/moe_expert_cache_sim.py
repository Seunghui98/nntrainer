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
  lrfu:H   frequency and recency in one score: each use adds 1, and the
           score halves every H tokens; evicts the lowest (doc 52 10.27)
  lrfu+:H  lrfu, plus half a use for each expert a decode token ranked 5th
           to 9th -- the routing's own guess at what comes next

Usage:
  NNTR_MOE_TRACE=/data/local/tmp/moe_trace.txt ./nntrainer_causallm <model>
  python3 tools/moe_expert_cache_sim.py moe_trace.txt --cache 2 4 8 16
  python3 tools/moe_expert_cache_sim.py --selftest

The TPS column is arithmetic, not a measurement: 1000 / (base_ms + layers x
misses_per_call x miss_ms), with base_ms the resident decode token and
miss_ms one synchronous miss, both measured in doc 52 sections 10.7-10.9.
"""
import argparse
import math
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


def simulate(calls, capacity, policy, seed=0, layers=1):
    """-> (decode misses, decode calls, prefill misses). Keys are
    (layer, expert). Raises when one call needs more than the pool holds.
    lrfu:H / lrfu+:H take H in tokens, layers calls each."""
    order = OrderedDict()   # resident keys, least recent first
    freq = {}
    crf, last = {}, {}      # lrfu: score as of call last[k]
    base, _, hl = policy.partition(":")
    half = float(hl) * layers if hl else 0.0

    def use(k, i, w=1.0):   # lrfu: decay the score to call i, then add w
        crf[k] = w + crf.get(k, 0.0) * 2.0 ** (-(i - last.get(k, i)) / half)
        last[k] = i

    def score(k, i):        # lrfu: log2 of the score decayed to call i
        c = crf.get(k, 0.0)
        return (math.log2(c) if c > 0 else -1e9) - (i - last.get(k, i)) / half
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
            if base in ("lrfu", "lrfu+"):
                use(k, i)
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
            elif base in ("lrfu", "lrfu+"):
                victim = min(cand, key=lambda k: score(k, i))
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
        if base == "lrfu+" and tokens == 1:
            for e in ext[4:9]:
                use((layer, e), i, 0.5)
        if tokens == 1:
            dec_calls += 1
            dec_miss += len(misses)
        else:
            pre_miss += len(misses)
    return dec_miss, dec_calls, pre_miss


POLICIES = ["ours", "lru", "lfu", "random", "belady"]
HALF_LIVES = [2, 8, 32, 128]  # tokens, for lrfu and lrfu+


def report(calls, caches, base_ms, miss_ms, policies):
    layers = len({c[0] for c in calls})
    routed = sum(len(c[2]) for c in calls if c[1] == 1)
    decode_calls = sum(1 for c in calls if c[1] == 1)
    per_call = routed / decode_calls if decode_calls else 0
    print(f"trace: {len(calls)} calls, {layers} layers, {decode_calls} decode "
          f"calls routing {per_call:.2f} experts each")
    print(f"{'C':>3} {'slots':>5} {'policy':>9} {'miss/decode call':>17} "
          f"{'hit %':>6} {'prefill misses':>15} {'TPS (arith)':>12}")
    for c in caches:
        cap = c * layers
        for p in policies:
            try:
                dm, dc, pm = simulate(calls, cap, p, layers=layers)
            except ValueError as err:
                print(f"{c:>3} {cap:>5} {p:>9}  -- {err}")
                break
            mpc = dm / dc if dc else 0.0
            hit = 100.0 * (1 - dm / routed) if routed else 0.0
            tps = 1000.0 / (base_ms + layers * mpc * miss_ms)
            print(f"{c:>3} {cap:>5} {p:>9} {mpc:>17.2f} {hit:>6.1f} "
                  f"{pm:>15} {tps:>12.1f}")
    curve = reuse(calls)
    if curve:
        print("token-to-token reuse (plan 266): share of a decode call's "
              "experts the same layer routed in its previous d decode tokens")
        print("  " + " ".join(f"d={d}:{100.0 * v:5.1f}%"
                              for d, v in curve.items()))
        for c in caches:
            hits = layer_hits(calls, c * layers)
            print(f"  lru per-layer hit % at C={c}: " +
                  " ".join(f"{h:.0f}" for h in hits))
    spec = speculation(calls)
    if spec[0][1]:
        print("decode read-ahead ceiling: share of a token's experts in the "
              "same layer's previous token's top-(4+m)")
        for m, (cov, n, reads) in spec.items():
            calls_n = n / per_call if per_call else 0
            print(f"  top-{4 + m}: {100.0 * cov / n:5.1f}% covered, "
                  f"{reads / calls_n if calls_n else 0:.1f} candidates/call")


def speculation(calls):
    """Decode read-ahead's ceiling (doc 52 section 10.25): for each decode
    call, the share of its routed experts that the SAME layer's previous
    decode token had in its top-(4+m), m = 0..5 -- what reading that
    token's next-ranked experts during the other layers' calls could have
    had resident in time. -> {m: (covered, routed, reads per call)}."""
    prev = {}
    out = {m: [0, 0, 0] for m in range(6)}
    for layer, tokens, routed, ext in calls:
        if tokens != 1:
            continue
        last = prev.get(layer)
        if last is not None:
            for m in range(6):
                window = set(last[:4 + m])
                out[m][0] += sum(1 for e in routed if e in window)
                out[m][1] += len(routed)
                out[m][2] += len(window)
        prev[layer] = ext
    return {m: tuple(v) for m, v in out.items()}


def reuse(calls, depth=8):
    """Plan 266's hit-rate curve: {d: share of decode-routed experts that
    the same layer routed in one of its previous d decode tokens}."""
    hist, hit, n = {}, [0] * (depth + 1), 0
    for layer, tokens, routed, _ in calls:
        if tokens != 1:
            continue
        h = hist.setdefault(layer, [])
        for e in routed:
            for d in range(1, depth + 1):
                if any(e in prev for prev in h[-d:]):
                    hit[d] += 1
        n += len(routed)
        h.append(set(routed))
    return {d: hit[d] / n for d in range(1, depth + 1)} if n else {}


def layer_hits(calls, capacity):
    """LRU decode hit % per layer at capacity slots (plan 266)."""
    order, need, miss = OrderedDict(), {}, {}
    for layer, tokens, routed, _ in calls:
        keys = [(layer, e) for e in routed]
        pinned = set(keys)
        misses = [k for k in keys if k not in order]
        for k in keys:
            if k in order:
                order.move_to_end(k)
        for _ in range(len(order) + len(misses) - capacity):
            del order[next(k for k in order if k not in pinned)]
        for k in misses:
            order[k] = None
        if tokens == 1:
            need[layer] = need.get(layer, 0) + len(keys)
            miss[layer] = miss.get(layer, 0) + len(misses)
    return [100.0 * (1 - miss[m] / need[m]) for m in sorted(need)]


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
    # Speculation: layer 0 routes [1, 2] after a token whose top list was
    # [2, 5, 1, ...]: top-4 covers both.
    spec = speculation([(0, 1, [0, 3], [2, 5, 1, 7, 9, 8, 6, 4, 3]),
                        (0, 1, [1, 2], [])])
    assert spec[0] == (2, 2, 4) and spec[5][0] == 2
    # lrfu: a key used twice outlives one used once more recently, while its
    # score has not halved away; with a short half-life recency wins again.
    calls = [(0, 1, [0], []), (0, 1, [0], []), (0, 1, [1], []),
             (0, 1, [2], []), (0, 1, [0], [])]
    assert simulate(calls, 2, "lrfu:100")[0] == 3  # 1 goes, 0 stays
    assert simulate(calls, 2, "lru")[0] == 4       # 0 goes
    assert simulate(calls, 2, "lrfu:0.1")[0] == 4  # recency again
    # lrfu+: an expert the last token ranked 5th counts as half a use.
    # Token 2 ranks 9 fifth, so when 0 needs a slot lrfu+ keeps 9 over 4.
    ext = [4, 1, 2, 3, 9, 5, 6, 7, 8]
    calls = [(0, 1, [9], []), (0, 1, [4], ext), (0, 1, [0], []),
             (0, 1, [9], [])]
    assert simulate(calls, 2, "lrfu:100")[0] == 4   # 9 went, missed again
    assert simulate(calls, 2, "lrfu+:100")[0] == 3  # 4 went, 9 hits
    # Reuse: layer 0 routes [1, 2] then [2, 3]: d=1 covers 2 of the 4.
    assert reuse([(0, 1, [1, 2], []), (0, 1, [2, 3], [])])[1] == 0.25
    assert layer_hits([(0, 1, [1], []), (0, 1, [1], [])], 1) == [50.0]
    # Parsing.
    assert parse(["3 1 | 4 7 | 7 4 9\n", "\n"]) == [(3, 1, [4, 7], [7, 4, 9])]
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace", nargs="?")
    ap.add_argument("--cache", type=int, nargs="+", default=[2, 4, 8, 16],
                    help="experts per layer (NNTR_MOE_CACHE_EXPERTS)")
    ap.add_argument("--base-ms", type=float, default=40.6,
                    help="resident decode token, ms (doc 52 section 10.28: "
                         "1000/24.61)")
    ap.add_argument("--miss-ms", type=float, default=0.40,
                    help="one warm miss, read + swap, ms (doc 52 section "
                         "10.29: 0.33 read + 0.07 swap)")
    ap.add_argument("--policies", nargs="+",
                    default=POLICIES + [f"{b}:{h}" for b in ("lrfu", "lrfu+")
                                        for h in HALF_LIVES],
                    help="policies to replay; lrfu:H and lrfu+:H take a "
                         "half-life H in tokens")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.trace:
        ap.error("a trace file, or --selftest")
    with open(a.trace) as f:
        report(parse(f), a.cache, a.base_ms, a.miss_ms, a.policies)


if __name__ == "__main__":
    sys.exit(main())
