#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# @file    htp_dump_eval.py
# @brief   Compare two NNTR_HTP_DUMP directories file by file: bit identity
#          first, SNR second, the first differing file named
# @author  dlwlzzero <dlwlzzero@gmail.com>
#
# The file list is the REFERENCE directory's: every call in its
# manifest.txt (<name>_in.f32 and <name>_out.f32, in call order) and every
# logits_<step>.f32 it holds. So a reference with only logits (a CPU run,
# which makes no HTP call) compares logits alone, and a reference with the
# manifest compares each MoE call's input and output too. The first file
# that is not identical names the call and the side: an _in file differing
# means the layers BEFORE the call differ (the host's own CPU path), an
# _out file with an identical _in means the HTP path itself.
#
#     htp_dump_eval.py [--label L] <ref_dir> <got_dir>
#
# Prints one line per file and
#     E2E eval <L> files=<n> bit_identical=0|1 min_snr_db=<x> first_diff=<f>
# Exit 0 when every file is byte-identical, 1 when any differs, 2 when a
# file is missing or its length differs (nothing to compare).
#
# SNR = 10 log10(sum(ref^2) / sum((ref - got)^2)), inf when identical.
# Lifted from hvx_impl's hexagon_e2e_test --eval (LEDGER section 4 lift 4).

import argparse
import math
import os
import re
import sys

import numpy as np


def file_list(ref_dir):
    names = []
    manifest = os.path.join(ref_dir, "manifest.txt")
    if os.path.isfile(manifest):
        with open(manifest) as f:
            for line in f:
                parts = line.split()
                if parts:
                    names.append(parts[0] + "_in.f32")
                    names.append(parts[0] + "_out.f32")
    logits = []
    for entry in os.listdir(ref_dir):
        m = re.fullmatch(r"logits_(\d+)\.f32", entry)
        if m:
            logits.append((int(m.group(1)), entry))
    names.extend(entry for _, entry in sorted(logits))
    return names


def snr_db(ref, got):
    err = float(np.sum((ref.astype(np.float64) - got.astype(np.float64)) ** 2))
    sig = float(np.sum(ref.astype(np.float64) ** 2))
    if err == 0.0:
        return math.inf
    if sig == 0.0:
        return -math.inf
    return 10.0 * math.log10(sig / err)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="dump")
    ap.add_argument("ref_dir")
    ap.add_argument("got_dir")
    a = ap.parse_args()

    names = file_list(a.ref_dir)
    if not names:
        print(f"E2E eval {a.label} error=no files in {a.ref_dir}")
        return 2
    # A run that made MORE calls (or steps) than the reference is a
    # difference too, and one the reference's list cannot see. A reference
    # without a manifest (a CPU run) compares logits only, by design.
    ref_has_calls = os.path.isfile(os.path.join(a.ref_dir, "manifest.txt"))
    extra = [n for n in file_list(a.got_dir)
             if n not in names and (ref_has_calls or n.startswith("logits_"))]
    if extra:
        print(f"E2E eval {a.label} error={len(extra)} files beyond the "
              f"reference, first {extra[0]}")
        return 2
    identical = True
    min_snr = math.inf
    first_diff = "-"
    for name in names:
        rp = os.path.join(a.ref_dir, name)
        gp = os.path.join(a.got_dir, name)
        if not os.path.isfile(gp):
            print(f"E2E eval {a.label} file={name} error=missing in {a.got_dir}")
            return 2
        ref = np.fromfile(rp, dtype=np.float32)
        got = np.fromfile(gp, dtype=np.float32)
        if ref.size != got.size:
            print(f"E2E eval {a.label} file={name} error=length {ref.size} vs "
                  f"{got.size}")
            return 2
        same = np.array_equal(ref.view(np.uint32), got.view(np.uint32))
        s = math.inf if same else snr_db(ref, got)
        print(f"E2E eval {a.label} file={name} n={ref.size} "
              f"bit_identical={int(same)} snr_db={s:.2f}")
        if not same:
            if identical:
                first_diff = name
            identical = False
            min_snr = min(min_snr, s)
    print(f"E2E eval {a.label} files={len(names)} bit_identical={int(identical)} "
          f"min_snr_db={min_snr:.2f} first_diff={first_diff}")
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
