#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
@file   mc_score.py
@date   30 Sep 2026
@brief  [#194 P3] Score the 40-question multiple choice from the app's
        NNTR_PPL_DECODE step lines (the NNTR_PPL_DECODE_ALTS logits)
@author dlwlzzero <dlwlzzero@gmail.com>

Reads <logdir>/<prefix><id>.log for every question of the tsv; each log's
'[PPL] decode step=1 target=<letter id> nll=<x> top1=<t> alts=<id:logit,..>'
line gives the four letters' logits (mc_ids.py's alts.txt order A B C D)
and the nll of the right letter over the whole vocabulary. The pick is the
first maximum of the four. Prints one line per question and
    MC <label> right=<n>/40 nll_sum=<x> missing=<m>
With --ref <prefix> (the same sitting's A) also plan 194's P3 gate:
    MC GATE <label> vs <ref> right=<n>/<nA> nll_ratio=<r> pass|FAIL
(pass: right >= right_A - 2 and nll_sum <= 1.05 nll_sum_A, nothing
missing). --self-test checks the parser and the gate on made-up logs.

Usage: mc_score.py [--tsv mc-40.tsv] [--ref A_prefix] <label> <logdir> <prefix>
       mc_score.py --self-test
"""

import argparse
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TSV = os.path.join(
    os.path.dirname(os.path.dirname(HERE)), "docs", "measurements", "prompts",
    "mc-40.tsv")
STEP = re.compile(
    r"\[PPL\] decode step=1 target=(\d+) nll=(\S+) top1=\d+ alts=(\S+)")


def questions(tsv):
    rows = [l.rstrip("\n").split("\t") for l in open(tsv, encoding="utf-8")]
    return [(r[0], r[6]) for r in rows[1:]]


def score(qs, logdir, prefix, out=None):
    """(right, nll_sum, missing) over the questions' logs."""
    right, nll_sum, missing = 0, 0.0, 0
    for qid, ans in qs:
        path = os.path.join(logdir, prefix + qid + ".log")
        m = None
        if os.path.exists(path):
            with open(path, errors="replace") as f:
                m = STEP.search(f.read())
        if m is None:
            missing += 1
            if out:
                print("%s missing (%s)" % (qid, path), file=out)
            continue
        logits = [float(p.split(":")[1]) for p in m.group(3).split(",")]
        if len(logits) != 4:
            missing += 1
            continue
        pick = "ABCD"[logits.index(max(logits))]
        nll = float(m.group(2))
        right += pick == ans
        nll_sum += nll
        if out:
            print("%s pick=%s answer=%s %s nll=%.4f" %
                  (qid, pick, ans, "ok" if pick == ans else "WRONG", nll),
                  file=out)
    return right, nll_sum, missing


def gate(r, n, m, ra, na, ma):
    """Plan 194 P3."""
    return m == 0 and ma == 0 and r >= ra - 2 and n <= 1.05 * na


def self_test():
    qs = [("q01", "A"), ("q02", "C"), ("q03", "B")]
    d = tempfile.mkdtemp()

    def log(prefix, qid, alts, nll):
        with open(os.path.join(d, prefix + qid + ".log"), "w") as f:
            f.write("noise\n[PPL] decode step=1 target=334 nll=%s top1=334 "
                    "alts=%s\n[PPL] decode tokens=1\n" % (nll, alts))

    # A: right; right (a C / D tie picks the first, C); wrong
    log("A_", "q01", "334:5,378:1,340:1,388:1", "0.5")
    log("A_", "q02", "334:1,378:1,340:3,388:3", "0.25")  # tie C/D -> C
    log("A_", "q03", "334:9,378:1,340:1,388:1", "2")
    r, n, m = score(qs, d, "A_")
    ok = (r, n, m) == (2, 2.75, 0)
    # E: one fewer right, nll 2 % higher: pass; then a missing log: fail
    log("E_", "q01", "334:5,378:1,340:1,388:1", "0.5")
    log("E_", "q02", "334:1,378:1,340:1,388:3", "0.3")
    log("E_", "q03", "334:9,378:1,340:1,388:1", "2.0")
    re_, ne, me = score(qs, d, "E_")
    ok &= (re_, me) == (1, 0) and gate(re_, ne, me, r, n, m)
    ok &= not gate(re_, n * 1.06, me, r, n, m)
    os.remove(os.path.join(d, "E_q03.log"))
    re_, ne, me = score(qs, d, "E_")
    ok &= me == 1 and not gate(re_, ne, me, r, n, m)
    print("MC SELF-TEST %s" % ("ok" if ok else "FAILED"))
    return 0 if ok else 1


def main():
    if "--self-test" in sys.argv[1:]:
        return self_test()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tsv", default=TSV)
    ap.add_argument("--ref", help="the reference (A) log prefix, same dir")
    ap.add_argument("label")
    ap.add_argument("logdir")
    ap.add_argument("prefix")
    a = ap.parse_args()
    qs = questions(a.tsv)
    r, n, m = score(qs, a.logdir, a.prefix, sys.stdout)
    print("MC %s right=%d/%d nll_sum=%.6f missing=%d" % (a.label, r, len(qs), n, m))
    if a.ref:
        ra, na, ma = score(qs, a.logdir, a.ref)
        ok = gate(r, n, m, ra, na, ma)
        print("MC GATE %s vs %s right=%d/%d nll_ratio=%.4f %s" %
              (a.label, a.ref, r, ra, n / na if na else float("inf"),
               "pass" if ok else "FAIL"))
        return 0 if ok else 1
    return 0 if m == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
