#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# [#136] The load loop: six in-process E2E runs side by side, five rounds,
# all six kinds resident on the LFM2.5-shaped fixture (lfm2_moe_tiny_lfm25),
# prompt 16 + 4 greedy tokens. This is the reproducer of the worker pool's
# job pickup race (plan 136 section 0.3): alone, the harness never crashed;
# six at once, 5 of 30 runs segfaulted with a pool worker still inside the
# MoE down phase after hvx_worker_pool_run had returned. So it is a gate,
# not a benchmark: a run counts only if it exits 0 AND prints the same
# E2E gen line as an unloaded run of the same fixture -- the device's
# form of the fault is not a crash but a wrong output row. A run that
# hangs (the barrier underflowing, the old pool never returns) is cut at
# 300 s and counts as failed.
#
# Gate line:  LOAD LOOP <ok>/30 rc=0
# Needs run_inproc_e2e.sh's build (build_htp_host/) and the fixture's
# weights (see the message below); the same env.sh as that script.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
B="${NNTR_INPROC_BUILD:-$ROOT/build_htp_host}"
FIX="$ROOT/test/unittest/models/causallm_reference/lfm2_moe_tiny_lfm25"
GEN="test/unittest/models/causallm_reference/generators/generate_lfm2_moe_reference.py"
ROUNDS=5
PAR=6

: "${HEXAGON_SDK_ROOT:?source tools/htp/env.sh first (the SDK headers)}"
[ -f "$B/build.ninja" ] ||
  { echo "LOAD FAIL no $B: run test/htp/host/run_inproc_e2e.sh once" >&2; exit 1; }
ninja -C "$B" nntrainer/libnntrainer.so \
  Applications/CausalLM/nntr_quantize_stream Applications/CausalLM/htp_e2e_test
if [ ! -f "$FIX/nntr_lfm2_moe_tiny_fp32.bin" ]; then
  echo "LOAD FAIL lfm25 fixture weights missing: run" >&2
  echo "  python3 $GEN --dim 2048 --n-heads 32 --n-kv-heads 8 --head-dim 64 --max-pos 2048 --layer-types conv,conv,attention,conv,attention,conv --num-dense 2 --rope-theta 5000000 --moe-inter 256 --out $FIX" >&2
  echo "  git checkout -- $FIX/" >&2
  exit 1
fi

OUT="$(mktemp -d)"
trap '[ -f "$OUT/.pass" ] && rm -rf "$OUT" || echo "kept: $OUT"' EXIT
Q="$B/Applications/CausalLM/nntr_quantize_stream"
E2E="$B/Applications/CausalLM/htp_e2e_test"
"$Q" "$FIX" -o "$OUT/htp" --fc_dtype Q4_0 --moe_dtype QS4CX_WH > "$OUT/q.log"

run_one() { # run_one <log>
  NNTR_HTP_FORWARD=1 timeout 300 "$E2E" --model "$OUT/htp" --tokenizer "$FIX/tokenizer.json" \
    --prompt 16 --steps 4 --max-seq 2048 --moe-engine htp > "$1" 2>&1
}
run_one "$OUT/ref.log"
ref="$(grep '^E2E gen ' "$OUT/ref.log")"
echo "LOAD ref: $ref"

for round in $(seq 1 $ROUNDS); do
  for j in $(seq 1 $PAR); do
    ( log="$OUT/r${round}_$j.log"
      if run_one "$log" && [ "$(grep '^E2E gen ' "$log" || true)" = "$ref" ]; then
        touch "$log.ok"
      else
        echo "LOAD run $round/$j failed: $(tail -1 "$log")"
      fi ) &
  done
  wait
done
ok=$(ls "$OUT"/r*.log.ok 2> /dev/null | wc -l)
total=$((ROUNDS * PAR))
if [ "$ok" = "$total" ]; then
  touch "$OUT/.pass"
  echo "LOAD LOOP $ok/$total rc=0"
else
  echo "LOAD LOOP $ok/$total rc=1"
  exit 1
fi
