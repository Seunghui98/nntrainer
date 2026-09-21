#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Generate the samples, bundle them into the viewer, run the headless checks
# and the python unit tests. Needs python3, node with playwright on
# NODE_PATH, and a Chromium binary (CHROMIUM=/path to override).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
FIX="$HERE/fixtures"
mkdir -p "$FIX"

python3 "$ROOT/make_sample_trace.py" -o "$FIX/trace_asbuilt.json"
python3 "$ROOT/make_sample_trace.py" --pipelined -o "$FIX/trace_pipelined.json"
python3 "$ROOT/make_sample_trace.py" --with-warnings -o "$FIX/trace_warnings.json"
python3 "$ROOT/qnn_optrace_to_nntr.py" "$HERE/data/qnn_mini.json" -o "$FIX/qnn_mini_nntr.json" --ts-unit cycles
python3 "$ROOT/bundle.py" -o "$FIX/bundle.html" \
  --trace "sample: as built=$FIX/trace_asbuilt.json" \
  --trace "sample: pipelined=$FIX/trace_pipelined.json" \
  --trace "sample: warnings=$FIX/trace_warnings.json" \
  --trace "qnn: mini fixture=$FIX/qnn_mini_nntr.json"

NODE_PATH="${NODE_PATH:-$(npm root -g)}" node "$HERE/check.js" "$FIX/bundle.html" "$FIX"
python3 -m unittest discover -s "$HERE" -p 'test_*.py'
