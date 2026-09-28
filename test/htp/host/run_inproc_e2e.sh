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
# Gate lines (all must print):
#   E2E eval golden files=<n> bit_identical=1 ...
#   E2E eval hmx-loop files=<n> bit_identical=1 ...   (NNTR_MOE_HTP_M1_GEMV=0)
#   E2E tokens htp==cpu 8/8
#   E2E eval cpu ... min_snr_db=<x>                    (printed, x >= 60 gated)
#   E2E eval self-test ok
# and, since #130, the per-token entry with the small ops and the m=1
# attention resident (NNTR_HTP_FORWARD=1; plan 130 section 1 gate 2):
#   E2E fwd tiny kinds=MOE,RMSNORM,CONV1D_GATE calls/token=11
#   E2E eval fwd-tiny ... min_snr_db=<x>               (x >= 30 gated: the two
#                              runs compute the same ops, _det vs the CPU's
#                              rounding; a wiring fault reads 0-20 dB)
#   E2E fwd tiny-all-kinds refused: AEE_ESCHEMENOTSUPPORTED   (head_dim 8)
#   E2E eval golden-hd64 ... bit_identical=1           (the hd64 fixture, off)
#   E2E fwd hd64 kinds=MOE,RMSNORM,QK_NORM,ROPE,CONV1D_GATE,ATTN_M1 calls/token=12
#   E2E eval fwd-hd64 ... min_snr_db=<x>               (x >= 30 gated)
#   E2E tokens fwd==off 8/8 expected_mismatch=0        (section 3.5's policy)
# and, since #136, LFM2.5's per-layer shape (lfm2_moe_tiny_lfm25: hidden
# 2048, 32 / 8 heads, six layers, max_seq 2048) at prompt 512 -- the
# real-shape case that the two small fixtures could not exercise:
#   E2E eval golden-lfm25 ... bit_identical=1         (logits only, off)
#   E2E fwd lfm25 kinds=<all six> calls/token=23.00
#   E2E eval fwd-lfm25 ... min_snr_db=<x> routing_flip=moe_00010 min_snr_db_gated=<y>
#                              (y >= 30 gated: the floor stops at the first
#                              router top-k flip, a near-tie on random
#                              weights; the tokens policy is the verdict there)
#   E2E tokens fwd==off-lfm25 8/8 expected_mismatch=0
# and, since #134, NNTR_PPL_DECODE through the app's CausalLM::run (--run;
# plan 134 section 1 gates 1-4):
#   E2E ppl-decode self==forced tokens=7 identical=1   (step lines to 17
#                              digits and the MoE dumps bit-identical)
#   E2E tokens run==adapter 8/8                        (run() = the loop above)
#   E2E ppl-decode index |dec-(S24-S17)|=<d> ok        (d < 1e-2 nats: a
#                              non-greedy forced continuation against the
#                              prefill NNTR_PPL sums of prompts 24 and 17)
#   E2E ppl-decode hd64 off=<ppl> on=<ppl> delta=<%> top1=7/7   (printed;
#                              gated: finite, and top1 7/7)
# NNTR_INPROC_GOLDEN=update rewrites test/htp/host/golden/lfm2_moe_tiny,
# lfm2_moe_tiny_hd64 and lfm2_moe_tiny_lfm25 from this run's switch-off
# HTP dumps (deliberate, like reference_logits.json).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
B="${NNTR_INPROC_BUILD:-$ROOT/build_htp_host}"
FIX="$ROOT/test/unittest/models/causallm_reference/lfm2_moe_tiny"
FIX64="$ROOT/test/unittest/models/causallm_reference/lfm2_moe_tiny_hd64"
GOLDEN="$HERE/golden/lfm2_moe_tiny"
GOLDEN64="$HERE/golden/lfm2_moe_tiny_hd64"
FIX25="$ROOT/test/unittest/models/causallm_reference/lfm2_moe_tiny_lfm25"
GOLDEN25="$HERE/golden/lfm2_moe_tiny_lfm25"
EVAL="python3 $ROOT/tools/htp/htp_dump_eval.py"
PROMPT=16
STEPS=8
ALL_KINDS=MOE,RMSNORM,QK_NORM,ROPE,CONV1D_GATE,ATTN_M1

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

GEN="test/unittest/models/causallm_reference/generators/generate_lfm2_moe_reference.py"
if [ ! -f "$FIX/nntr_lfm2_moe_tiny_fp32.bin" ]; then
  echo "E2E FAIL fixture weights missing: run" >&2
  echo "  python3 $GEN" >&2
  echo "  git checkout -- test/unittest/models/causallm_reference/lfm2_moe_tiny/" >&2
  exit 1
fi
if [ ! -f "$FIX64/nntr_lfm2_moe_tiny_fp32.bin" ]; then
  echo "E2E FAIL hd64 fixture weights missing: run" >&2
  echo "  python3 $GEN --dim 128 --n-heads 2 --n-kv-heads 1 --head-dim 64 --max-pos 32 --out $FIX64" >&2
  echo "  git checkout -- $FIX64/" >&2
  exit 1
fi
if [ ! -f "$FIX25/nntr_lfm2_moe_tiny_fp32.bin" ]; then
  echo "E2E FAIL lfm25 fixture weights missing: run" >&2
  echo "  python3 $GEN --dim 2048 --n-heads 32 --n-kv-heads 8 --head-dim 64 --max-pos 2048 --layer-types conv,conv,attention,conv,attention,conv --num-dense 2 --rope-theta 5000000 --moe-inter 256 --out $FIX25" >&2
  echo "  git checkout -- $FIX25/" >&2
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

run_e2e() { # run_e2e <label> <model dir> <engine> <dump dir> <log> [args]
  local label=$1 model=$2 engine=$3 dump=$4 log=$5
  shift 5
  # stderr too: the backend's [HTP] graph: lines are the gate's
  "$E2E" --model "$model" --tokenizer "$FIX/tokenizer.json" --prompt $PROMPT \
    --steps $STEPS --moe-engine "$engine" --dump "$dump" "$@" 2>&1 | tee "$log"
  grep -q '^E2E gen ' "$log" || { echo "E2E FAIL no gen line in $label"; exit 1; }
}
calls_per_token() { # the backend's close-time summary line
  sed -n 's/.*forward calls=[0-9]* tokens=[0-9]* calls\/token=\([0-9.]*\).*/\1/p' "$1"
}
echo "== htp (M1_GEMV default)"
run_e2e htp "$OUT/htp" htp "$OUT/dump_htp" "$OUT/htp.log"
echo "== htp, NNTR_MOE_HTP_M1_GEMV=0 (HMX loop at M = 1)"
NNTR_MOE_HTP_M1_GEMV=0 run_e2e hmx "$OUT/htp" htp "$OUT/dump_hmx" "$OUT/hmx.log"
echo "== cpu control (QS4CX on the CPU path)"
run_e2e cpu "$OUT/cpu" cpu "$OUT/dump_cpu" "$OUT/cpu.log"

# [#130] the per-token entry: the hd8 fixture with the small ops resident
# (the attention kinds cannot run at head_dim 8), then all kinds refused,
# then the hd64 fixture with every kind resident against its switch-off run
echo "== htp, NNTR_HTP_FORWARD=1 KINDS=MOE,RMSNORM,CONV1D_GATE (the per-token entry)"
NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=MOE,RMSNORM,CONV1D_GATE \
  run_e2e fwd-tiny "$OUT/htp" htp "$OUT/dump_fwd" "$OUT/fwd.log"
echo "== htp, NNTR_HTP_FORWARD=1 all kinds on head_dim 8 (must be refused)"
rc=0
NNTR_HTP_FORWARD=1 "$E2E" --model "$OUT/htp" --tokenizer "$FIX/tokenizer.json" \
  --prompt $PROMPT --steps $STEPS --moe-engine htp > "$OUT/all.log" 2>&1 || rc=$?
tail -1 "$OUT/all.log"
"$Q" "$FIX64" -o "$OUT/htp64" --fc_dtype Q4_0 --moe_dtype QS4CX_WH > "$OUT/q_htp64.log"
echo "== hd64 htp, switch off"
run_e2e hd64-off "$OUT/htp64" htp "$OUT/dump_64off" "$OUT/64off.log" --max-seq 32
echo "== hd64 htp, NNTR_HTP_FORWARD=1 (all six kinds)"
NNTR_HTP_FORWARD=1 \
  run_e2e hd64-fwd "$OUT/htp64" htp "$OUT/dump_64fwd" "$OUT/64fwd.log" --max-seq 32
# [#136] the real-shape fixture at prompt 512 (the device's prompt length):
# the per-token entry after a long CPU prefill, the KV seed of 512 rows,
# RoPE past 512, the conv state seed, at LFM2.5's widths
"$Q" "$FIX25" -o "$OUT/htp25" --fc_dtype Q4_0 --moe_dtype QS4CX_WH > "$OUT/q_htp25.log"
echo "== lfm25 htp, switch off, prompt 512"
PROMPT=512 run_e2e lfm25-off "$OUT/htp25" htp "$OUT/dump_25off" "$OUT/25off.log" --max-seq 2048
echo "== lfm25 htp, NNTR_HTP_FORWARD=1 (all six kinds), prompt 512"
PROMPT=512 NNTR_HTP_FORWARD=1 \
  run_e2e lfm25-fwd "$OUT/htp25" htp "$OUT/dump_25fwd" "$OUT/25fwd.log" --max-seq 2048

if [ "${NNTR_INPROC_GOLDEN:-}" = update ]; then
  mkdir -p "$GOLDEN" "$GOLDEN64" "$GOLDEN25"
  rm -f "$GOLDEN"/*.f32 "$GOLDEN/manifest.txt" "$GOLDEN64"/*.f32 "$GOLDEN64/manifest.txt" \
    "$GOLDEN25"/*.f32
  cp "$OUT"/dump_htp/*.f32 "$OUT/dump_htp/manifest.txt" "$GOLDEN/"
  cp "$OUT"/dump_64off/*.f32 "$OUT/dump_64off/manifest.txt" "$GOLDEN64/"
  # logits only: a prompt-512 MoE call's input is 4 MiB (README.md there)
  cp "$OUT"/dump_25off/logits_*.f32 "$GOLDEN25/"
  echo "golden updated: $GOLDEN ($(ls "$GOLDEN"/*.f32 | wc -l) files)," \
    "$GOLDEN64 ($(ls "$GOLDEN64"/*.f32 | wc -l) files)," \
    "$GOLDEN25 ($(ls "$GOLDEN25"/*.f32 | wc -l) files)"
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

# (d) [#130] the per-token entry on the hd8 fixture: 11 calls per token
# (plan 130 section 3.5's arithmetic), every MoE call's bytes within the
# rounding band of the switch-off run. The floor's calibration, measured
# on this harness at the first passing run (PR for #130): a wiring fault
# reads 0-20 dB (plan 84 section 1); CONV1D_GATE alone is bit-identical
# and RMSNORM alone 129 dB (f32 rounding, _det vs the CPU intrinsic); the
# attention kinds read 39 dB at worst on the hd64 fixture -- the CPU's
# fp16 KV rows against the DSP's f32 decode rows, amplified by the Q4_0
# and u8 activation quantizers on 128-wide rows -- while the first two
# decode steps stay bit-identical through the whole chain. 30 dB sits 10
# above the fault band and 9 below that observation.
SNR_FLOOR=30
calls="$(calls_per_token "$OUT/fwd.log")"
echo "E2E fwd tiny kinds=MOE,RMSNORM,CONV1D_GATE calls/token=${calls:-none}"
[ "$calls" = 11.00 ] || fail=1
$EVAL --label fwd-tiny --allow-diff --snr-floor $SNR_FLOOR "$OUT/dump_htp" "$OUT/dump_fwd" | tail -1 || fail=1
# (e) all kinds at head_dim 8: the validator's shape rule, through the
# model's real throw (exit 1 with the E2E FAIL line)
if [ $rc = 1 ] && grep -q '^E2E FAIL set_decode_graph_desc: AEE_ESCHEMENOTSUPPORTED' "$OUT/all.log"; then
  echo "E2E fwd tiny-all-kinds refused: AEE_ESCHEMENOTSUPPORTED"
else
  echo "E2E FAIL all kinds at head_dim 8 not refused (rc=$rc)"; fail=1
fi
# (f) the hd64 fixture: its own golden with the switch off; with it on, 12
# calls per token, the SNR floor, the token policy, and the init lines
$EVAL --label golden-hd64 "$GOLDEN64" "$OUT/dump_64off" | tail -1 || fail=1
calls="$(calls_per_token "$OUT/64fwd.log")"
echo "E2E fwd hd64 kinds=$ALL_KINDS calls/token=${calls:-none}"
[ "$calls" = 12.00 ] || fail=1
$EVAL --label fwd-hd64 --allow-diff --snr-floor $SNR_FLOOR "$OUT/dump_64off" "$OUT/dump_64fwd" | tail -1 || fail=1
$EVAL --label 'fwd==off' --tokens-policy "$OUT/dump_64off" "$OUT/dump_64fwd" | tail -1 || fail=1
grep -q '^\[HTP\] graph: init n_ops=30 resident=RMSNORM|CONV1D_GATE|QK_NORM|ROPE|ATTN_M1|MOE ' "$OUT/64fwd.log" ||
  { echo "E2E FAIL hd64: no init line with every kind resident"; fail=1; }
grep -q '^\[HTP\] attn_m1: registered layers=1 kv=1 gqa=2 head_dim=64 max_seq=32 cache=16 KiB' "$OUT/64fwd.log" ||
  { echo "E2E FAIL hd64: no attn_m1 registration line"; fail=1; }
# (g) [#136] the lfm25 fixture: its logits golden with the switch off; with
# it on, 23 calls per token (the stretch count of the six-layer list with
# every kind resident, plan 136 section 0.2), the SNR floor, the token
# policy, and the init lines at LFM2.5's widths
$EVAL --label golden-lfm25 "$GOLDEN25" "$OUT/dump_25off" | tail -1 || fail=1
calls="$(calls_per_token "$OUT/25fwd.log")"
echo "E2E fwd lfm25 kinds=$ALL_KINDS calls/token=${calls:-none}"
[ "$calls" = 23.00 ] || fail=1
$EVAL --label fwd-lfm25 --allow-diff --snr-floor $SNR_FLOOR "$OUT/dump_25off" "$OUT/dump_25fwd" | tail -1 || fail=1
$EVAL --label 'fwd==off-lfm25' --tokens-policy "$OUT/dump_25off" "$OUT/dump_25fwd" | tail -1 || fail=1
grep -q '^\[HTP\] graph: init n_ops=58 resident=RMSNORM|CONV1D_GATE|QK_NORM|ROPE|ATTN_M1|MOE ' "$OUT/25fwd.log" ||
  { echo "E2E FAIL lfm25: no init line with every kind resident"; fail=1; }
grep -q '^\[HTP\] attn_m1: registered layers=2 kv=8 gqa=4 head_dim=64 max_seq=2048 cache=16384 KiB' "$OUT/25fwd.log" ||
  { echo "E2E FAIL lfm25: no attn_m1 registration line"; fail=1; }

# (h) [#134] NNTR_PPL_DECODE on the app's own decode loop (CausalLM::run,
# --run). g1: the self run writes its greedy continuation, the forced run
# reads it back; both score the same logits, so every step line and every
# MoE call match exactly. g2: run() emits the adapter path's tokens. g3:
# forcing a NON-greedy continuation (prompt ids 16..23, all distinct; the
# greedy path is 16 at every step, blind to an off-by-one) equals the
# difference of the prefill NNTR_PPL sums of prompts 24 and 17, which score
# the same seven targets; a shifted target reads O(0.1-1) nats per token
# here, the 6-digit prefill print adds <= 3e-4. The CPU engine: the index
# logic is under test, not the engine. g4: the six-kind entry forced on
# the hd64 switch-off continuation; the numbers are the fixture's, not
# silicon's (39 dB here, 18 dB on the device: plan 134 section 5).
dec_steps() { grep -o '\[PPL\] decode step=.*' "$1" || true; }
dec_field() { # dec_field <log> <field>: from the [PPL] decode tokens= line
  sed -n "s/.*\[PPL\] decode tokens=.* $2=\([^ ]*\).*/\1/p" "$1"
}
echo "== [#134] run() path, NNTR_PPL_DECODE self then forced (hd8, htp)"
NNTR_PPL_DECODE="$OUT/g.ids" run_e2e run-self "$OUT/htp" htp "$OUT/dump_gself" "$OUT/gself.log" --run
NNTR_PPL_DECODE="$OUT/g.ids" run_e2e run-forced "$OUT/htp" htp "$OUT/dump_gforced" "$OUT/gforced.log" --run
g_eval="$($EVAL --label ppl-decode-forced "$OUT/dump_gself" "$OUT/dump_gforced" | tail -1 || true)"
echo "$g_eval"
n="$(dec_steps "$OUT/gself.log" | wc -l)"
if [ "$n" = $((STEPS - 1)) ] && [ "$(dec_field "$OUT/gself.log" source)" = self ] &&
  [ "$(dec_field "$OUT/gforced.log" source)" = file ] &&
  [ "$(dec_steps "$OUT/gself.log")" = "$(dec_steps "$OUT/gforced.log")" ] &&
  grep -q 'bit_identical=1' <<< "$g_eval"; then
  echo "E2E ppl-decode self==forced tokens=$n identical=1"
else
  echo "E2E FAIL ppl-decode self vs forced (steps=$n)"; fail=1
fi
run_gen="$(grep '^E2E gen ' "$OUT/gself.log")"
same=$(paste <(tr ' ' '\n' <<< "$htp_gen") <(tr ' ' '\n' <<< "$run_gen") |
  tail -n +3 | awk '$1==$2{n++} END{print n+0}')
echo "E2E tokens run==adapter $same/$STEPS"
[ "$same" = "$STEPS" ] || fail=1
echo "== [#134] index check: forced non-greedy continuation vs prefill NNTR_PPL (cpu)"
echo "23 30 7 14 21 28 5 12" > "$OUT/idx.ids"
NNTR_PPL_DECODE="$OUT/idx.ids" "$E2E" --model "$OUT/cpu" --tokenizer "$FIX/tokenizer.json" \
  --moe-engine cpu --run --prompt 16 --steps 8 > "$OUT/idx.log" 2>&1 || true
for p in 17 24; do
  NNTR_PPL=1 "$E2E" --model "$OUT/cpu" --tokenizer "$FIX/tokenizer.json" \
    --moe-engine cpu --run --prompt $p --steps 1 > "$OUT/idx$p.log" 2>&1 || true
done
dec="$(dec_field "$OUT/idx.log" nll_sum)"
a17="$(sed -n 's/.*\[PPL\] prompt tokens=16 nll\/token=\([^ ]*\).*/\1/p' "$OUT/idx17.log")"
a24="$(sed -n 's/.*\[PPL\] prompt tokens=23 nll\/token=\([^ ]*\).*/\1/p' "$OUT/idx24.log")"
if [ -n "$dec" ] && [ -n "$a17" ] && [ -n "$a24" ] &&
  grep -q '^E2E gen 23 30 7 14 21 28 5 12$' "$OUT/idx.log"; then
  d="$(awk -v d="$dec" -v a="$a24" -v b="$a17" \
    'BEGIN{x = d - (23 * a - 16 * b); if (x < 0) x = -x; printf "%.2g", x}')"
  if awk -v d="$d" 'BEGIN{exit !(d < 1e-2)}'; then
    echo "E2E ppl-decode index |dec-(S24-S17)|=$d ok"
  else
    echo "E2E FAIL ppl-decode index |dec-(S24-S17)|=$d >= 1e-2"; fail=1
  fi
else
  echo "E2E FAIL ppl-decode index: missing line (dec=$dec a17=$a17 a24=$a24)"; fail=1
fi
echo "== [#134] hd64: switch off self, then all six kinds forced on it"
NNTR_PPL_DECODE="$OUT/h.ids" \
  run_e2e run-64off "$OUT/htp64" htp "$OUT/dump_g64off" "$OUT/g64off.log" --max-seq 32 --run
NNTR_HTP_FORWARD=1 NNTR_PPL_DECODE="$OUT/h.ids" \
  run_e2e run-64fwd "$OUT/htp64" htp "$OUT/dump_g64fwd" "$OUT/g64fwd.log" --max-seq 32 --run
off="$(dec_field "$OUT/g64off.log" ppl)"
on="$(dec_field "$OUT/g64fwd.log" ppl)"
top1="$(dec_field "$OUT/g64fwd.log" top1)"
line="E2E ppl-decode hd64 off=${off:-none} on=${on:-none}"
if [ "$(dec_field "$OUT/g64fwd.log" source)" = file ] &&
  awk -v a="$off" -v b="$on" 'BEGIN{exit !(a + 0 > 0 && b + 0 > 0 && a + 0 < 1e30 && b + 0 < 1e30)}'; then
  echo "$line delta=$(awk -v a="$off" -v b="$on" 'BEGIN{printf "%+.3f%%", (b / a - 1) * 100}') top1=$top1"
  [ "$top1" = "$((STEPS - 1))/$((STEPS - 1))" ] || fail=1
else
  echo "E2E FAIL $line (not finite, or not forced)"; fail=1
fi

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
