#!/usr/bin/env python3
##
# @file    216-core-load-report.py
# @brief   Folds 216-core-load-run.sh's core.samples, run logs and top frames
#          into uptime x per-core busy % x ms/miss (plan 216 §4 step 1)
#
# Per run: the decode window is the sample intervals in which the pool
# server thread (the app task pinned to core 6 other than main) gained CPU
# ticks; per core, "other" busy % = /proc/stat busy minus the app's own
# task ticks attributed to the core each task last ran on. Also a 10 s
# timeline of the whole boot and the busiest non-app threads of the top -H
# frames that overlap each decode window.
# Usage: 216-core-load-report.py <logs_core/<label> dir> [...]
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
            cur = {"u": float(t[1]), "cpu": {}, "app": {}, "aff": {},
                   "freq": [], "bus": [], "vm": {}, "psi": {}, "mem": {}}
            out.append(cur)
        elif cur is None:
            continue
        elif t[0] == "C" and len(t) >= 9:
            v = list(map(int, t[2:10]))
            busy = v[0] + v[1] + v[2] + v[5] + v[6] + v[7]
            cur["cpu"][int(t[1][3:])] = (busy, busy + v[3] + v[4])
        elif t[0] == "F":
            cur["freq"] = [int(x) for x in t[1:]]
        elif t[0] == "B":
            cur["bus"] = [int(x) for x in t[1:]]
        elif t[0] == "V":
            cur["vm"] = {k: int(v) for k, v in (x.split("=") for x in t[1:])}
        elif t[0] == "M":
            cur["mem"] = {k: int(v) // 1024 for k, v in (x.split("=") for x in t[1:])}
        elif t[0] == "P" and len(t) == 4:
            cur["psi"][t[1] + "_" + t[2]] = int(t[3])
        elif t[0] == "A":
            rest = line[line.rindex(")") + 2:].split()
            tid = int(t[1])
            cur["app"][tid] = (int(rest[11]) + int(rest[12]), int(rest[36]))
        elif t[0] == "G" and len(t) >= 3:
            cur["aff"][int(t[1])] = t[2]
    return [s for s in out if len(s["cpu"]) == 8]


def runs(d):
    rx = re.compile(r"^(\S+): up ([0-9.]+)-([0-9.]+) \| (.*)$")
    out = []
    for line in open(os.path.join(d, "sitting.out"), errors="replace"):
        m = rx.match(line.strip())
        if not m:
            continue
        name, u0, u1, rest = m.group(1), float(m.group(2)), float(m.group(3)), m.group(4)
        def f(p):
            mm = re.search(p, rest)
            return mm.group(1) if mm else "-"
        out.append({"name": name, "u0": u0, "u1": u1,
                    "tps": f(r"\| ([0-9.]+) TPS \|"), "pre": f(r"([0-9.]+) TPS prefill"),
                    "msm": f(r"\(([0-9.]+) ms/miss\)"), "arm": f(r"arm_ms/round=([0-9.]+)"),
                    "mw": f(r"miss_wait_us/token=([0-9.]+)"), "text": f(r"text (\S+)")})
    return out


def tops(path):
    frames, cur = [], None
    if not os.path.exists(path):
        return frames
    for line in open(path, errors="replace"):
        if line.startswith("TOP "):
            cur = {"u": float(line.split()[1]), "rows": []}
            frames.append(cur)
        elif cur is not None:
            t = line.split()
            if len(t) >= 12 and t[0].isdigit():
                try:
                    cpu = float(t[8])
                except ValueError:
                    continue
                cur["rows"].append((cpu, " ".join(t[11:])[:48], t[0]))
    return frames


def window(ss, r):
    """Sample index pairs inside the run whose interval the server gained ticks."""
    idx = [i for i, s in enumerate(ss) if r["u0"] - 1 <= s["u"] <= r["u1"] + 1]
    if len(idx) < 2:
        return [], idx
    main = None
    srv = set()
    for i in idx:
        app = ss[i]["app"]
        if app and main is None:
            main = min(app)
        for tid, a in ss[i]["aff"].items():
            if a == "6" and tid != main and tid != min(app or [tid]):
                srv.add(tid)
    dec = []
    for a, b in zip(idx, idx[1:]):
        g = sum(ss[b]["app"].get(t, (0, 0))[0] - ss[a]["app"].get(t, (0, 0))[0]
                for t in srv if t in ss[b]["app"])
        if g > 0:
            dec.append((a, b))
    return dec, idx


def busy(ss, pairs):
    other, app, tot = [0] * 8, [0] * 8, [0] * 8
    for a, b in pairs:
        for c in range(8):
            other[c] += ss[b]["cpu"][c][0] - ss[a]["cpu"][c][0]
            tot[c] += ss[b]["cpu"][c][1] - ss[a]["cpu"][c][1]
        for tid, (tk, cpu) in ss[b]["app"].items():
            d = tk - ss[a]["app"].get(tid, (tk, cpu))[0]
            if 0 <= cpu < 8:
                app[cpu] += d
    o = [max(0, other[c] - app[c]) * 100.0 / tot[c] if tot[c] else 0.0 for c in range(8)]
    p = [app[c] * 100.0 / tot[c] if tot[c] else 0.0 for c in range(8)]
    return o, p


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def report(d):
    ss = samples(os.path.join(d, "core.samples"))
    rs = runs(d)
    tf = tops(os.path.join(d, "top.frames"))
    print(f"## {os.path.basename(d.rstrip('/'))}  ({len(ss)} samples, uptime "
          f"{ss[0]['u']:.0f}-{ss[-1]['u']:.0f} s)\n")
    print("| run | up s | decode tok/s | prefill | ms/miss | arm_ms/round | miss_wait us/tok "
          "| dec win s | other busy % cpu0..7 (decode) | app % cpu0..7 (decode) "
          "| MHz cpu0/6/7 | DDR MHz | kswapd scan/s | pswpout | pgpgin MiB | refault | MemAvail / Cached MiB at decode | PSI mem/io/cpu ms | text |")
    print("|" + "---|" * 19)
    for r in rs:
        dec, idx = window(ss, r)
        if not dec:
            print(f"| {r['name']} | {r['u0']:.0f} | {r['tps']} | {r['pre']} | {r['msm']} | {r['arm']} | {r['mw']} | no window ||||||||||| {r['text']} |")
            continue
        o, p = busy(ss, dec)
        a, b = dec[0][0], dec[-1][1]
        span = ss[b]["u"] - ss[a]["u"]
        inside = [ss[j] for j in range(a + 1, b + 1)]
        fr = [mean([s["freq"][c] for s in inside]) / 1000 for c in (0, 6, 7)]
        ddr = mean([s["bus"][0] for s in inside]) / 1000
        vm = lambda k: (ss[b]["vm"][k] - ss[a]["vm"][k]) if k in ss[a]["vm"] else float("nan")
        ps = lambda k: (ss[b]["psi"].get(k, 0) - ss[a]["psi"].get(k, 0)) / 1000
        print(f"| {r['name']} | {r['u0']:.0f} | {r['tps']} | {r['pre']} | {r['msm']} | {r['arm']} | {r['mw']} "
              f"| {span:.1f} | {' '.join(f'{x:.0f}' for x in o)} | {' '.join(f'{x:.0f}' for x in p)} "
              f"| {fr[0]:.0f}/{fr[1]:.0f}/{fr[2]:.0f} | {ddr:.0f} | {vm('pgscan_kswapd') / max(span, 1e-3):.0f} "
              f"| {vm('pswpout')} | {vm('pgpgin') / 1024:.0f} | {vm('workingset_refault_file')} | {ss[a]['mem'].get('MemAvailable', '-')} / {ss[a]['mem'].get('Cached', '-')} | {ps('memory_some'):.0f}/{ps('io_some'):.0f}/{ps('cpu_some'):.0f} | {r['text']} |")
    print()
    for r in rs:
        dec, _ = window(ss, r)
        if not dec or not tf:
            continue
        a, b = ss[dec[0][0]]["u"], ss[dec[-1][1]]["u"]
        near = [f for f in tf if a - 6 <= f["u"] <= b + 1]
        rows = {}
        for f in near:
            for cpu, name, tid in f["rows"]:
                if "nntrainer" in name or name.startswith("top"):
                    continue
                rows[name] = max(rows.get(name, 0), cpu)
        best = sorted(rows.items(), key=lambda x: -x[1])[:5]
        print(f"* {r['name']} top -H frames at uptime "
              f"{', '.join(f'{f['u']:.0f}' for f in near) or '-'}: "
              + ", ".join(f"{n} {c:.0f}%" for n, c in best))
    print("\nTimeline (10 s bins; other = /proc/stat busy minus the app's ticks, in cores):\n")
    print("| up s | other cores | busiest core (other %) | app cores | MHz cpu0/6/7 | DDR MHz | kswapd scan/s | pswpout/s | PSI mem/io ms/s |")
    print("|---|---|---|---|---|---|---|---|---|")
    t0 = int(ss[0]["u"] // 10 * 10)
    while t0 < ss[-1]["u"]:
        ii = [i for i, s in enumerate(ss) if t0 <= s["u"] < t0 + 10]
        if len(ii) >= 2:
            pairs = list(zip(ii, ii[1:]))
            o, p = busy(ss, pairs)
            a, b = ii[0], ii[-1]
            span = ss[b]["u"] - ss[a]["u"]
            vm = lambda k: (ss[b]["vm"].get(k, 0) - ss[a]["vm"].get(k, 0)) / span
            ps = lambda k: (ss[b]["psi"].get(k, 0) - ss[a]["psi"].get(k, 0)) / 1000 / span
            inside = [ss[j] for j in ii]
            fr = [mean([s["freq"][c] for s in inside]) / 1000 for c in (0, 6, 7)]
            mc = max(range(8), key=lambda c: o[c])
            print(f"| {t0} | {sum(o) / 100:.2f} | cpu{mc} ({o[mc]:.0f}) | {sum(p) / 100:.2f} "
                  f"| {fr[0]:.0f}/{fr[1]:.0f}/{fr[2]:.0f} | {mean([s['bus'][0] for s in inside]) / 1000:.0f} "
                  f"| {vm('pgscan_kswapd'):.0f} | {vm('pswpout'):.0f} | {ps('memory_some'):.0f}/{ps('io_some'):.0f} |")
        t0 += 10
    print()


for d in sys.argv[1:]:
    report(d)
