#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# The host E2E gate of the in-process HTP build (#84): the tiny LFM2-MoE
# fixture, prompt 16 + 8 greedy tokens, through the REAL ARM-side HTP path
# (HtpComputeOps, HtpBackend, the MoE call marshalling) into the DSP skel
# compiled for this machine on scalar stand-ins (test/htp/host/inproc,
# standin, hvx_emu, stub). Needs the Hexagon SDK's headers and qaic
# (source tools/htp/env.sh); no device, no HexKL, no simulator.
#
# What it proves: the op-table / session / opts wiring, the marshalling,
# and whole-token bit-identity of every MoE call's input and output
# against a committed golden and between the two M=1 paths. What it
# cannot: tok/s, transport, DMA, the HVX/HMX arithmetic bits (the
# stand-ins are scalar), the 32-bit DSP address budget. Read
# docs/plans/84-host-e2e-inproc.md section 1 before quoting a line.
#
# Gate lines (all five must print):
#   E2E eval golden files=<n> bit_identical=1 ...
#   E2E eval hmx-loop files=<n> bit_identical=1 ...   (NNTR_MOE_HTP_M1_GEMV=0)
#   E2E tokens htp==cpu 8/8
#   E2E eval cpu ... min_snr_db=<x>                    (printed, x >= 60 gated)
#   E2E eval self-test ok
# NNTR_INPROC_GOLDEN=update rewrites test/htp/host/golden/lfm2_moe_tiny
# from this run's HTP dump (deliberate, like reference_logits.json).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
B="${NNTR_INPROC_BUILD:-$ROOT/build_htp_host}"
FIX="$ROOT/test/unittest/models/causallm_reference/lfm2_moe_tiny"
GOLDEN="$HERE/golden/lfm2_moe_tiny"
EVAL="python3 $ROOT/tools/htp/htp_dump_eval.py"
PROMPT=16
STEPS=8

: "${HEXAGON_SDK_ROOT:?source tools/htp/env.sh first (the SDK headers)}"
if [ ! -f "$B/build.ninja" ]; then
  [ -f "$ROOT/nntrainer/tensor/htp_backend/generated/nntr_hvx.h" ] ||
    bash "$ROOT/nntrainer/tensor/htp_backend/generate_stub.sh"
  meson setup "$B" -Denable-transformer=true -Denable-tflite-backbone=false \
    -Denable-tflite-interpreter=false \
    -Dc_args=-Wno-error=missing-include-dirs \
    -Dcpp_args=-Wno-error=missing-include-dirs \
    -Denable-htp=true -Dhtp-inproc=true \
    -Dhexagon-sdk-root="$HEXAGON_SDK_ROOT"
fi
ninja -C "$B" nntrainer/libnntrainer.so \
  Applications/CausalLM/nntr_quantize_stream Applications/CausalLM/htp_e2e_test

if [ ! -f "$FIX/nntr_lfm2_moe_tiny_fp32.bin" ]; then
  echo "E2E FAIL fixture weights missing: run" >&2
  echo "  python3 test/unittest/models/causallm_reference/generators/generate_lfm2_moe_reference.py" >&2
  echo "  git checkout -- test/unittest/models/causallm_reference/lfm2_moe_tiny/" >&2
  exit 1
fi

OUT="$(mktemp -d)"
# Removed on a pass only: on a failure the dumps and logs are what the
# first_diff line points at.
trap '[ -f "$OUT/.pass" ] && rm -rf "$OUT" || echo "kept: $OUT"' EXIT
Q="$B/Applications/CausalLM/nntr_quantize_stream"
E2E="$B/Applications/CausalLM/htp_e2e_test"

# Two quantizations of the one fixture: QS4CX for the CPU control, QS4CX_WH
# (the HMX tile layout, ISA-free bytes) for the HTP model. FC Q4_0 on the
# host ISA: an ARM-packed Q4_0 is wrong on x86 (contract section 11).
"$Q" "$FIX" -o "$OUT/cpu" --fc_dtype Q4_0 --moe_dtype QS4CX > "$OUT/q_cpu.log"
"$Q" "$FIX" -o "$OUT/htp" --fc_dtype Q4_0 --moe_dtype QS4CX_WH > "$OUT/q_htp.log"

run_e2e() { # run_e2e <model dir> <engine> <dump dir> <log>
  "$E2E" --model "$2" --tokenizer "$FIX/tokenizer.json" --prompt $PROMPT \
    --steps $STEPS --moe-engine "$3" --dump "$4" | tee "$5"
  grep -q '^E2E gen ' "$5" || { echo "E2E FAIL no gen line in $1"; exit 1; }
}
echo "== htp (M1_GEMV default)"
run_e2e htp "$OUT/htp" htp "$OUT/dump_htp" "$OUT/htp.log"
echo "== htp, NNTR_MOE_HTP_M1_GEMV=0 (HMX loop at M = 1)"
NNTR_MOE_HTP_M1_GEMV=0 run_e2e hmx "$OUT/htp" htp "$OUT/dump_hmx" "$OUT/hmx.log"
echo "== cpu control (QS4CX on the CPU path)"
run_e2e cpu "$OUT/cpu" cpu "$OUT/dump_cpu" "$OUT/cpu.log"

if [ "${NNTR_INPROC_GOLDEN:-}" = update ]; then
  mkdir -p "$GOLDEN"
  rm -f "$GOLDEN"/*.f32 "$GOLDEN/manifest.txt"
  cp "$OUT"/dump_htp/*.f32 "$OUT/dump_htp/manifest.txt" "$GOLDEN/"
  echo "golden updated: $GOLDEN ($(ls "$GOLDEN"/*.f32 | wc -l) files)"
fi

fail=0
# (b) this run against the committed golden, every call's bytes.
$EVAL --label golden "$GOLDEN" "$OUT/dump_htp" | tail -1 || fail=1
# (a) the HMX loop at M = 1 against the GEMV: the same bytes, at model level.
$EVAL --label hmx-loop "$OUT/dump_htp" "$OUT/dump_hmx" | tail -1 || fail=1
# (c) the CPU control: tokens gated, SNR printed (rule 25: the number is
# the run's, the verdict is the tokens').
htp_gen="$(grep '^E2E gen ' "$OUT/htp.log")"
cpu_gen="$(grep '^E2E gen ' "$OUT/cpu.log")"
same=$(paste <(tr ' ' '\n' <<< "$htp_gen") <(tr ' ' '\n' <<< "$cpu_gen") |
  tail -n +3 | awk '$1==$2{n++} END{print n+0}')
echo "E2E tokens htp==cpu $same/$STEPS"
[ "$same" = "$STEPS" ] || fail=1
cpu_line="$($EVAL --label cpu "$OUT/dump_cpu" "$OUT/dump_htp" | tail -1 || true)"
echo "$cpu_line"
snr="$(sed -n 's/.*min_snr_db=\([-0-9.inf]*\).*/\1/p' <<< "$cpu_line")"
awk -v s="$snr" 'BEGIN{exit !(s == "inf" || s+0 >= 60)}' ||
  { echo "E2E FAIL cpu-vs-htp min_snr_db=$snr < 60 (a wiring fault reads 0-20)"; fail=1; }

# The comparator's own check: identical -> 1; one byte flipped -> 0 with a
# finite SNR and exit 1; a truncated file -> exit 2.
cp -r "$OUT/dump_htp" "$OUT/self"
first="$(head -1 "$OUT/dump_htp/manifest.txt" | awk '{print $1}')_out.f32"
$EVAL --label self-same "$OUT/dump_htp" "$OUT/self" > /dev/null || fail=1
printf '\001' | dd of="$OUT/self/$first" bs=1 seek=1 conv=notrunc 2> /dev/null
rc=0; flip="$($EVAL --label self-flip "$OUT/dump_htp" "$OUT/self" | tail -1)" || rc=$?
[ $rc = 1 ] && grep -q 'bit_identical=0' <<< "$flip" &&
  ! grep -q 'min_snr_db=inf' <<< "$flip" || { echo "self-test: flip not caught ($rc: $flip)"; fail=1; }
truncate -s 4 "$OUT/self/$first"
rc=0; $EVAL --label self-trunc "$OUT/dump_htp" "$OUT/self" > /dev/null || rc=$?
[ $rc = 2 ] || { echo "self-test: truncation exit $rc, want 2"; fail=1; }
[ $fail = 0 ] && echo "E2E eval self-test ok"

if [ $fail = 0 ]; then
  touch "$OUT/.pass"
  echo "INPROC E2E PASS"
else
  echo "INPROC E2E FAIL"
  exit 1
fi
