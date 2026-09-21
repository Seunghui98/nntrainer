#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# @package bundle
# @brief Embed trace.json files into viewer.html for a self-contained report.
"""Embed trace.json files into viewer.html so one file carries the data.

The viewer opens from file:// with no server, but a browser will not let a
file:// page fetch a sibling json, so either pick the file with the Load
button or bake it in here. The result is one HTML file that opens anywhere
(mail, chat, an artifact link) and needs nothing else.

Usage
    tools/nntr_trace/bundle.py -o report.html --trace run1=trace_asbuilt.json
    tools/nntr_trace/bundle.py -o report.html --trace "as built=a.json" \
        --trace "pipelined=b.json"

Stdlib only.
"""

import argparse
import json
import os

MARKER = '<script id="embedded-traces" type="application/json">{}</script>'
HERE = os.path.dirname(os.path.abspath(__file__))


def bundle(viewer_path, traces):
  with open(viewer_path) as f:
    html = f.read()
  if MARKER not in html:
    raise SystemExit("%s has no embedded-traces slot" % viewer_path)
  emb = {}
  for spec in traces:
    name, _, path = spec.partition("=")
    if not path:
      path, name = name, os.path.basename(name)
    with open(path) as f:
      emb[name] = json.load(f)
  payload = json.dumps(emb, separators=(",", ":")).replace("</", "<\\/")
  return html.replace(MARKER, MARKER.replace("{}", payload, 1))


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--viewer", default=os.path.join(HERE, "viewer.html"))
  ap.add_argument("--trace", action="append", default=[],
                  help="name=path (repeatable); a bare path uses its basename")
  ap.add_argument("-o", "--out", required=True)
  a = ap.parse_args()
  out = bundle(a.viewer, a.trace)
  with open(a.out, "w") as f:
    f.write(out)
  print("%s: %.2f MB, %d trace(s)" % (a.out, len(out) / 1e6, len(a.trace)))


if __name__ == "__main__":
  main()
