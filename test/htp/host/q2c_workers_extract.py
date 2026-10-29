#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Writes q2c_ctx and the q2 calibration workers of nntr_hvx_attn_q.c to one
file, for q2c_calib_check.c to include: the check runs the production code,
not a copy of it."""
import re
import sys


def block(text, start):
    i = text.index("{", start) + 1
    depth = 1
    while depth:
        depth += (text[i] == "{") - (text[i] == "}")
        i += 1
    return i


def main(src, out):
    text = open(src, encoding="utf-8").read()
    end = text.index("} q2c_ctx;") + len("} q2c_ctx;")
    parts = [text[text.rindex("typedef struct {", 0, end):end]]
    for name in ("q2c_worker", "q2c_worker_kd"):
        m = re.search(r"^static void " + name + r"\(", text, re.M)
        if not m:
            raise SystemExit("missing production function " + name)
        parts.append(text[m.start():block(text, m.start())])
    open(out, "w", encoding="utf-8").write("\n\n".join(parts) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
