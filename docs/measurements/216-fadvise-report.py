#!/usr/bin/env python3
##
# @file    216-fadvise-report.py
# @brief   Folds 216-fadvise-run.sh's sitting.out and core.samples into one
#          row per run: ms/miss, tok/s, the app's and the window's pgpgin,
#          refaults, PSI io, kswapd and the model file's resident MiB
#
# The decode window is the sample intervals in which the pool server (the
# app task pinned to core 6 alone, other than main) gained CPU ticks, as in
# PR #217's 216-core-load-report.py; the hybrid runs have no pool server and
# no window. ms/miss: the profiled `file read` figure where the run was
# profiled, else arm_ms/round x rounds / misses (the same quantity: the pool
# server's time in its answers over the misses).
# Usage: 216-fadvise-report.py <logs/<label> dir> [...]
import os
import re
import sys


def samples(path):
    out, cur = [], None
    for line in open(path, errors="replace"):
        t = line.split()
        if not t:
            continue
        if t[0] == "S":
            cur = {"u": float(t[1]), "app": {}, "aff": {}, "vm": {}, "psi": {}, "res": None}
            out.append(cur)
        elif cur is None:
            continue
        elif t[0] == "V":
            cur["vm"] = {k: int(v) for k, v in (x.split("=") for x in t[1:])}
        elif t[0] == "P" and len(t) == 4:
            cur["psi"][t[1] + "_" + t[2]] = int(t[3])
        elif t[0] == "R" and len(t) == 2 and t[1].lstrip("-").isdigit():
            cur["res"] = int(t[1])
        elif t[0] == "A":
            rest = line[line.rindex(")") + 2:].split()
            cur["app"][int(t[1])] = int(rest[11]) + int(rest[12])
        elif t[0] == "G" and len(t) >= 3:
            cur["aff"][int(t[1])] = t[2]
    return out


def runs(d):
    rx = re.compile(r"^(\S+): up ([0-9.]+)-([0-9.]+) \| (.*)$")
    out = []
    for line in open(os.path.join(d, "sitting.out"), errors="replace"):
        m = rx.match(line.strip())
        if not m:
            continue
        rest = m.group(4)

        def f(p, rest=rest):
            mm = re.search(p, rest)
            return mm.group(1) if mm else None
        out.append({"name": m.group(1), "u0": float(m.group(2)), "u1": float(m.group(3)),
                    "cell": f(r"^(\S+ G=\d+ prof=\d)"), "pre": f(r"([0-9.]+) TPS prefill"),
                    "tps": f(r"\| ([0-9.]+) TPS \|"), "prof": f(r"\(([0-9.]+) ms/miss\)"),
                    "arm": f(r"arm_ms/round=([0-9.]+)"), "rounds": f(r"rounds=(\d+)"),
                    "misses": f(r" misses=(\d+)"), "mw": f(r"miss_wait_us/token=([0-9.]+)"),
                    "pg": f(r"pgpgin_mib=([0-9.]+)"), "after": f(r"resident_after (\d+)"),
                    "text": f(r"text (\S+)")})
    return out


def window(ss, r):
    idx = [i for i, s in enumerate(ss) if r["u0"] - 1 <= s["u"] <= r["u1"] + 1]
    if len(idx) < 2:
        return []
    main = min((t for i in idx for t in ss[i]["app"]), default=None)
    srv = {t for i in idx for t, a in ss[i]["aff"].items() if a == "6" and t != main}
    return [(a, b) for a, b in zip(idx, idx[1:])
            if sum(ss[b]["app"].get(t, 0) - ss[a]["app"].get(t, 0) for t in srv) > 0]


def report(d):
    ss = samples(os.path.join(d, "core.samples"))
    print(f"## {os.path.basename(d.rstrip('/'))}\n")
    print("| run | cell | up s | prefill tok/s | decode tok/s | ms/miss | arm_ms/round | miss_wait us/tok "
          "| app pgpgin MiB | win pgpgin MiB | win refault | win PSI io ms | win kswapd scan/s "
          "| resident MiB in win (min-max) | resident after | text |")
    print("|" + "---|" * 16)
    for r in runs(d):
        msm = r["prof"] or (f"{float(r['arm']) * int(r['rounds']) / int(r['misses']):.2f}*"
                            if r["arm"] and r["misses"] and int(r["misses"]) else "-")
        dec = window(ss, r) if r["arm"] else []
        if dec:
            a, b = dec[0][0], dec[-1][1]
            span = max(ss[b]["u"] - ss[a]["u"], 1e-3)
            vm = lambda k: ss[b]["vm"].get(k, 0) - ss[a]["vm"].get(k, 0)
            res = [ss[j]["res"] for j in range(a, b + 1) if ss[j]["res"] is not None]
            w = (f"{vm('pgpgin') / 1024:.0f} | {vm('workingset_refault_file')} "
                 f"| {(ss[b]['psi'].get('io_some', 0) - ss[a]['psi'].get('io_some', 0)) / 1000:.0f} "
                 f"| {vm('pgscan_kswapd') / span:.0f} | "
                 + (f"{min(res)}-{max(res)}" if res else "-"))
        else:
            w = "- | - | - | - | -"
        print(f"| {r['name']} | {r['cell']} | {r['u0']:.0f} | {r['pre']} | {r['tps']} | {msm} "
              f"| {r['arm'] or '-'} | {r['mw'] or '-'} | {r['pg'] or '-'} | {w} | {r['after']} | {r['text']} |")
    print("\n`*` = arm_ms/round x rounds / misses (run not profiled).\n")


for d in sys.argv[1:]:
    report(d)
