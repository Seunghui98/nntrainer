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
# and, since #132, ADD and ROUTER_TOPK resident (logits only: no MoE stretch
# is sole any more, so the switch-on runs dump no decode MoE call):
#   E2E fwd tiny kinds=MOE,RMSNORM,CONV1D_GATE,ADD calls/token=9.00
#   E2E eval add==noadd-tiny ... bit_identical=1       (ADD is IEEE a + b)
#   E2E fwd hd64 kinds=<six>,ADD calls/token=10.00
#   E2E eval add==noadd-hd64 ... bit_identical=1
#   E2E fwd tiny kinds=MOE,RMSNORM,CONV1D_GATE,ADD,ROUTER_TOPK calls/token=7.00
#   E2E fwd hd64 kinds=<six>,ADD,ROUTER_TOPK calls/token=8.00
#   E2E fwd lfm25 kinds=<six>,ADD,ROUTER_TOPK calls/token=15.00
#   E2E eval d-tiny / d-hd64 ... min_snr_db=<x>       (x >= 30 gated, vs off)
#   E2E eval d-lfm25 ... min_snr_db=<x>                (x >= 30 gated, vs the
#                              six-kind run: B's own near-tie routing flip
#                              reads 16 dB against off at logits_2, and D
#                              carries it unchanged)
#   E2E tokens d==off-tiny / -hd64 / -lfm25 8/8 expected_mismatch=0
#   E2E fwd tiny ADD-without-RMSNORM refused: AEE_ENOTALLOWED
# and, since #132 Part B E1, every kind resident in one session (the FC,
# the dense FFN and the tied lm_head on the DSP too: one call per token;
# logits only, the switch-on runs dump no MoE call):
#   E2E fwd hd64 kinds=all calls/token=1.00 q4m1_handles=12
#   E2E fwd lfm25 kinds=all calls/token=1.00 q4m1_handles=23
#   E2E eval e1x86==d-hd64 / -lfm25 ... bit_identical=1 (the quantizer in
#                              x86's id order, NNTR_INPROC_X86_Q8=1: every
#                              logit and prefill MoE call equals D's)
#   E2E eval e1-hd64 / e1-lfm25 ... min_snr_db=<x>     (printed: the
#                              Android CPU's order against x86's D)
#   E2E tokens e1==off-hd64 / -lfm25 8/8 expected_mismatch=0
#   E2E eval e1-feed-l2 ... bit_identical=1            (NNTR_HTP_FC_FEED=l2:
#                              the feed moves no bits)
#   E2E fwd hd64 / lfm25 cpu-fc-skipped=<n> per_token=7 / 14 ok  (E2: the CPU
#                              skips every FC GEMV of a resident row -- the
#                              core FC, qkv and conv block hooks; the
#                              bit_identical lines above hold with it)
#   E2E fwd tiny all-kinds refused: AEE_ESCHEMENOTSUPPORTED   (head_dim 8:
#                              no one-call token on the hd8 fixture)
# and, since #132 Part B E3, the same token on two sessions in this process
# (NNTR_HTP_E2E=1: S1 the router and the experts, S2 the rest on a lite
# open with no VTCM, its FC set in an attached arena, the MoE rounds over
# the mailbox page; one dspqueue packet per session per token). Not the
# device's two PDs -- one address space, no cache maintenance, no transport
# time -- but every ARM-side step of the device path:
#   E2E eval e3==e1-hd64 / -lfm25 ... bit_identical=1  (logits, the Android-
#                              order E1 run of the same model)
#   E2E fwd hd64 / lfm25 e3 calls/token=1.00 hops/token=4.00 / 8.00
#                              timeouts=0/0 id_mismatch=0 ok
#   E2E tokens e3-run==e1-run-lfm25 8/8 id_only=1      (run(): only the id
#                              comes back, the CPU takes it)
#   E2E ppl-decode e3==e1-hd64 steps=7 identical=1     (NNTR_PPL_DECODE
#                              forced on E1's path: the logits come back)
# and, since #141 step 2, the M==1 MoE calls over dspqueue
# (NNTR_HTP_DSPQ=1, inproc/dspqueue_standin.c; plan 141-dspq-moe.md step 4),
# each against the switch-off run's dumps, every call's bytes:
#   E2E eval dspq-tiny ... bit_identical=1
#   E2E eval dspq-lfm25 ... bit_identical=1            (prompt 512, SPIN_US=0)
#   E2E eval dspq-hmx ... bit_identical=1              (NNTR_MOE_HTP_M1_GEMV=0)
#   E2E dspq on-lines tiny/lfm25/hmx calls=<n> ok      (on once, close
#                              calls=N served=N bad=0, N = M==1 MoE calls)
#   E2E dspq off-path banner=1 bit_identical=1         (NNTR_INPROC_NO_DSPQ=1)
# and, since #150, the decode timer NNTR_OP_TIME=1 on the run() path, after
# its op_time_report.py table (plan 150-cpu-decode.md step 1a):
#   E2E eval op-time ... bit_identical=1               (vs the unset self run)
#   E2E op-time inert bit_identical=1 nll=identical unset_lines=0 table=ok
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
# dspqueue is the default since #141: every run below is FastRPC unless it
# sets NNTR_HTP_DSPQ=1 itself, so the switch-off goldens stay the reference.
export NNTR_HTP_DSPQ=0
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
# [#132] ADD, then ADD + ROUTER_TOPK (D), on the three fixtures
D_TINY=MOE,RMSNORM,CONV1D_GATE,ADD
echo "== [#132] htp, NNTR_HTP_FORWARD=1 KINDS=$D_TINY / +ROUTER_TOPK"
NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=$D_TINY \
  run_e2e tiny-add "$OUT/htp" htp "$OUT/dump_add" "$OUT/add.log"
NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=$D_TINY,ROUTER_TOPK \
  run_e2e tiny-d "$OUT/htp" htp "$OUT/dump_d" "$OUT/d.log"
echo "== [#132] hd64 htp, KINDS=<six>,ADD / +ROUTER_TOPK"
NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=$ALL_KINDS,ADD \
  run_e2e hd64-add "$OUT/htp64" htp "$OUT/dump_64add" "$OUT/64add.log" --max-seq 32
NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=$ALL_KINDS,ADD,ROUTER_TOPK \
  run_e2e hd64-d "$OUT/htp64" htp "$OUT/dump_64d" "$OUT/64d.log" --max-seq 32
echo "== [#132] lfm25 htp, KINDS=<six>,ADD,ROUTER_TOPK, prompt 512"
PROMPT=512 NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=$ALL_KINDS,ADD,ROUTER_TOPK \
  run_e2e lfm25-d "$OUT/htp25" htp "$OUT/dump_25d" "$OUT/25d.log" --max-seq 2048
# [#132 Part B, E1] every kind resident: one call per token, VTCM feed and
# L2 feed; the hd8 fixture refused (its attention kinds cannot be resident).
# The lm_head on the DSP is the Q4_0 one, so these runs quantize the tied
# embedding to Q4_0 (LFM2.5's file does; the fixtures default to FP32) and
# carry their own switch-off and D references on that model.
E1_KINDS=RMSNORM,FC,CONV1D_GATE,QK_NORM,ROPE,ATTN_M1,ADD,ROUTER_TOPK,MOE,DENSE_FFN,LM_HEAD
"$Q" "$FIX64" -o "$OUT/htp64q" --fc_dtype Q4_0 --moe_dtype QS4CX_WH \
  --embd_dtype Q4_0 > "$OUT/q_htp64q.log"
"$Q" "$FIX25" -o "$OUT/htp25q" --fc_dtype Q4_0 --moe_dtype QS4CX_WH \
  --embd_dtype Q4_0 > "$OUT/q_htp25q.log"
# NNTR_INPROC_X86_Q8=1 runs the DSP quantizer in x86 nntrainer's id order
# (hvx_q4_gemv_f32.c; the only op where the x86 CPU's Q4_0 FC differs from
# the Android CPU's), so that run is held against D bit for bit; the
# Android-order run (e1a) is the tokens policy and a printed SNR.
echo "== [#132 Part B] hd64 / lfm25 htp, Q4_0 lm_head: off, D, KINDS=all (one call per token)"
for fx in 64 25; do
  if [ $fx = 64 ]; then p=$PROMPT; ms=32; else p=512; ms=2048; fi
  PROMPT=$p run_e2e "q$fx-off" "$OUT/htp${fx}q" htp "$OUT/dump_${fx}qoff" "$OUT/${fx}qoff.log" --max-seq $ms
  PROMPT=$p NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=$ALL_KINDS,ADD,ROUTER_TOPK \
    run_e2e "q$fx-d" "$OUT/htp${fx}q" htp "$OUT/dump_${fx}qd" "$OUT/${fx}qd.log" --max-seq $ms
  PROMPT=$p NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=$E1_KINDS NNTR_INPROC_X86_Q8=1 \
    run_e2e "q$fx-e1" "$OUT/htp${fx}q" htp "$OUT/dump_${fx}e1" "$OUT/${fx}e1.log" --max-seq $ms
  PROMPT=$p NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=$E1_KINDS \
    run_e2e "q$fx-e1a" "$OUT/htp${fx}q" htp "$OUT/dump_${fx}e1a" "$OUT/${fx}e1a.log" --max-seq $ms
done
NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=$E1_KINDS NNTR_HTP_FC_FEED=l2 \
  NNTR_INPROC_X86_Q8=1 \
  run_e2e hd64-e1l2 "$OUT/htp64q" htp "$OUT/dump_64e1l2" "$OUT/64e1l2.log" --max-seq 32
# [#132 Part B E3] the two-session token (NNTR_HTP_E2E=1; its packets ride
# dspqueue, so these runs set NNTR_HTP_DSPQ=1)
echo "== [#132 Part B E3] hd64 / lfm25, NNTR_HTP_E2E=1 (two sessions)"
NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1 \
  run_e2e q64-e3 "$OUT/htp64q" htp "$OUT/dump_64e3" "$OUT/64e3.log" --max-seq 32
PROMPT=512 NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1 \
  run_e2e q25-e3 "$OUT/htp25q" htp "$OUT/dump_25e3" "$OUT/25e3.log" --max-seq 2048
PROMPT=512 NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=$E1_KINDS \
  run_e2e q25-e1run "$OUT/htp25q" htp "$OUT/dump_25e1run" "$OUT/25e1run.log" --max-seq 2048 --run
PROMPT=512 NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1 \
  run_e2e q25-e3run "$OUT/htp25q" htp "$OUT/dump_25e3run" "$OUT/25e3run.log" --max-seq 2048 --run
rm -f "$OUT/e3.ids"
NNTR_PPL_DECODE="$OUT/e3.ids" NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=$E1_KINDS \
  run_e2e q64-e1ppl "$OUT/htp64q" htp "$OUT/dump_64e1ppl" "$OUT/64e1ppl.log" --max-seq 32 --run
NNTR_PPL_DECODE="$OUT/e3.ids" NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1 \
  run_e2e q64-e3ppl "$OUT/htp64q" htp "$OUT/dump_64e3ppl" "$OUT/64e3ppl.log" --max-seq 32 --run
rc_e1=0
NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=$E1_KINDS "$E2E" --model "$OUT/htp" \
  --tokenizer "$FIX/tokenizer.json" --prompt $PROMPT --steps $STEPS \
  --moe-engine htp > "$OUT/tiny_e1.log" 2>&1 || rc_e1=$?
tail -1 "$OUT/tiny_e1.log"
# [#141] the M==1 MoE calls over dspqueue: tiny (DSP spins between calls),
# lfm25 at prompt 512 (DSP always blocks), the HMX loop at M = 1, and a
# runtime without dspqueue (the off banner, FastRPC throughout)
echo "== [#141] htp, NNTR_HTP_DSPQ=1 (tiny / lfm25 SPIN_US=0 / HMX loop / no dspqueue)"
NNTR_HTP_DSPQ=1 run_e2e dspq-tiny "$OUT/htp" htp "$OUT/dump_dspq" "$OUT/dspq.log"
PROMPT=512 NNTR_HTP_DSPQ=1 NNTR_HTP_DSPQ_SPIN_US=0 \
  run_e2e dspq-lfm25 "$OUT/htp25" htp "$OUT/dump_25dspq" "$OUT/25dspq.log" --max-seq 2048
NNTR_HTP_DSPQ=1 NNTR_MOE_HTP_M1_GEMV=0 \
  run_e2e dspq-hmx "$OUT/htp" htp "$OUT/dump_dspqhmx" "$OUT/dspqhmx.log"
NNTR_INPROC_NO_DSPQ=1 NNTR_HTP_DSPQ=1 \
  run_e2e dspq-off "$OUT/htp" htp "$OUT/dump_dspqoff" "$OUT/dspqoff.log"
echo "== [#132] htp, ADD resident without RMSNORM (must be refused)"
rc_add=0
NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=MOE,CONV1D_GATE,ADD "$E2E" \
  --model "$OUT/htp" --tokenizer "$FIX/tokenizer.json" --prompt $PROMPT \
  --steps $STEPS --moe-engine htp > "$OUT/add_nonorm.log" 2>&1 || rc_add=$?
tail -1 "$OUT/add_nonorm.log"

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
grep -q '^\[HTP\] attn_m1: registered layers=2 kv=8 gqa=4 head_dim=64 max_seq=2048 cache=8192 KiB' "$OUT/25fwd.log" ||
  { echo "E2E FAIL lfm25: no attn_m1 registration line"; fail=1; }
# (h) [#132] ADD alone moves no bit (IEEE a + b on both sides) and takes
# two calls per MoE layer off; ROUTER_TOPK takes one more (hd8 11 -> 9 ->
# 7, hd64 12 -> 10 -> 8, lfm25 23 -> 19 -> 15, plan 132 section 0). The
# switch-on runs dump no decode MoE call, so each reference is its
# run's logits alone (a directory with no manifest compares logits only).
# D's SNR reference is the switch-off run, except on lfm25: there B (the
# six kinds) already reads 16 dB against off at the #136 routing flip
# (moe_00010, a near-tie), which the logits-only compare cannot skip, so
# D is gated against B -- the step this PR adds; the tokens policy stays
# against off.
logits_only() { mkdir -p "$2" && cp "$1"/logits_*.f32 "$2/"; }
logits_only "$OUT/dump_fwd" "$OUT/ref_fwd"
logits_only "$OUT/dump_64fwd" "$OUT/ref_64fwd"
logits_only "$OUT/dump_htp" "$OUT/ref_off"
logits_only "$OUT/dump_64off" "$OUT/ref_64off"
logits_only "$OUT/dump_25fwd" "$OUT/ref_25fwd"
calls="$(calls_per_token "$OUT/add.log")"
echo "E2E fwd tiny kinds=$D_TINY calls/token=${calls:-none}"
[ "$calls" = 9.00 ] || fail=1
$EVAL --label add==noadd-tiny "$OUT/ref_fwd" "$OUT/dump_add" | tail -1 || fail=1
calls="$(calls_per_token "$OUT/64add.log")"
echo "E2E fwd hd64 kinds=$ALL_KINDS,ADD calls/token=${calls:-none}"
[ "$calls" = 10.00 ] || fail=1
$EVAL --label add==noadd-hd64 "$OUT/ref_64fwd" "$OUT/dump_64add" | tail -1 || fail=1
for d in "tiny $D_TINY,ROUTER_TOPK d 7.00 ref_off dump_htp" \
  "hd64 $ALL_KINDS,ADD,ROUTER_TOPK 64d 8.00 ref_64off dump_64off" \
  "lfm25 $ALL_KINDS,ADD,ROUTER_TOPK 25d 15.00 ref_25fwd dump_25off"; do
  read -r fx kinds tag want ref off <<< "$d"
  calls="$(calls_per_token "$OUT/$tag.log")"
  echo "E2E fwd $fx kinds=$kinds calls/token=${calls:-none}"
  [ "$calls" = "$want" ] || fail=1
  $EVAL --label "d-$fx" --allow-diff --snr-floor $SNR_FLOOR "$OUT/$ref" "$OUT/dump_$tag" | tail -1 || fail=1
  $EVAL --label "d==off-$fx" --tokens-policy "$OUT/$off" "$OUT/dump_$tag" | tail -1 || fail=1
done
# (h2) [#132 Part B, E1] one call per token with every kind resident, and
# bit-identical to the D run on the same model (the prefill MoE calls and
# every logit) once the quantizer takes x86's id order -- so the FC,
# dense FFN and lm_head are the x86 CPU's bits end to end; the handles the
# model bound; the Android-order run against off by the tokens policy
# (its SNR printed: lfm25 reads ~15 dB at the first decode step, where the
# x86-order run is bit-identical, so the quantizer's id rounding is the
# whole difference); the L2 feed bit-identical to the VTCM feed
for d in "hd64 64e1 12 dump_64qd dump_64qoff" "lfm25 25e1 23 dump_25qd dump_25qoff"; do
  read -r fx tag nh ref off <<< "$d"
  calls="$(calls_per_token "$OUT/$tag.log")"
  handles="$(sed -n 's/^\[HTP\] graph: q4m1 weights=[0-9]* handles=\([0-9]*\) .*/\1/p' "$OUT/$tag.log")"
  echo "E2E fwd $fx kinds=all calls/token=${calls:-none} q4m1_handles=${handles:-none}"
  [ "$calls" = 1.00 ] && [ "$handles" = "$nh" ] || fail=1
  [ "$(calls_per_token "$OUT/${tag}a.log")" = 1.00 ] || fail=1
  $EVAL --label "e1x86==d-$fx" "$OUT/$ref" "$OUT/dump_$tag" | tail -1 || fail=1
  $EVAL --label "e1-$fx" --allow-diff --snr-floor 0 "$OUT/$ref" "$OUT/dump_${tag}a" | tail -1 || fail=1
  $EVAL --label "e1==off-$fx" --tokens-policy "$OUT/$off" "$OUT/dump_${tag}a" | tail -1 || fail=1
done
# [#132 Part B E2] the CPU skips the FC layers of every resident row: per
# decode token hd64 (C A C, layer 0 dense) asks 7 times (two conv blocks,
# qkv, attention_out, the dense FFN's three), lfm25 (C C A C A C, two
# dense) 14; the first decode token after the prefill is a resident row too
for d in "hd64 64e1 7" "lfm25 25e1 14"; do
  read -r fx tag per <<< "$d"
  skipped="$(sed -n 's/^\[HTP\] graph: cpu fc skipped=\([0-9]*\)$/\1/p' "$OUT/$tag.log")"
  toks="$(sed -n 's/.*forward calls=[0-9]* tokens=\([0-9]*\) .*/\1/p' "$OUT/$tag.log")"
  if [ -n "$skipped" ] && [ -n "$toks" ] && [ "$skipped" = $((per * toks)) ]; then
    echo "E2E fwd $fx cpu-fc-skipped=$skipped per_token=$per ok"
  else
    echo "E2E FAIL $fx cpu fc skipped=${skipped:-none} tokens=${toks:-none} (want $per per token)"; fail=1
  fi
done
grep -q '^\[HTP\] graph: q4m1 weights=12 handles=12 feed=l2$' "$OUT/64e1l2.log" ||
  { echo "E2E FAIL hd64 e1 l2: no feed=l2 line"; fail=1; }
$EVAL --label e1-feed-l2 "$OUT/dump_64e1" "$OUT/dump_64e1l2" | tail -1 || fail=1
# (h3) [#132 Part B E3] two sessions: every logit equal to the one-session
# E1 run (Android order) of the same model, one packet per token, 2 hops
# per MoE layer, no timeout, S2's argmax equal to the logits' first
# maximum; run() with only the id back equals E1's run(); the PPL decode
# step lines (17 digits) equal E1's
for d in "hd64 64e3 64e1a 4.00" "lfm25 25e3 25e1a 8.00"; do
  read -r fx tag ref hops <<< "$d"
  $EVAL --label "e3==e1-$fx" "$OUT/dump_$ref" "$OUT/dump_$tag" | tail -1 || fail=1
  close="$(grep '^\[HTP\] token driver: close ' "$OUT/$tag.log" || true)"
  calls="$(calls_per_token "$OUT/$tag.log")"
  if [ "$calls" = 1.00 ] && grep -q " hops/token=$hops " <<< "$close" &&
    grep -q ' timeouts=0/0 stale=0/0 ' <<< "$close" &&
    grep -q ' id_mismatch=0 ' <<< "$close" &&
    grep -q '^\[HTP\] s2: fc arena weights=' "$OUT/$tag.log"; then
    echo "E2E fwd $fx e3 calls/token=$calls hops/token=$hops timeouts=0/0 id_mismatch=0 ok"
  else
    echo "E2E FAIL $fx e3: calls/token=${calls:-none} close=[$close]"; fail=1
  fi
done
e1_gen="$(grep '^E2E gen ' "$OUT/25e1run.log")"
e3_gen="$(grep '^E2E gen ' "$OUT/25e3run.log")"
same=$(paste <(tr ' ' '\n' <<< "$e1_gen") <(tr ' ' '\n' <<< "$e3_gen") |
  tail -n +3 | awk '$1==$2{n++} END{print n+0}')
id_only=0
grep -q '^\[HTP\] token driver: first token .* logits=0$' "$OUT/25e3run.log" && id_only=1
echo "E2E tokens e3-run==e1-run-lfm25 $same/$STEPS id_only=$id_only"
[ "$same" = "$STEPS" ] && [ $id_only = 1 ] || fail=1
e3_steps() { grep -o '\[PPL\] decode step=.*' "$1" || true; }
n="$(e3_steps "$OUT/64e3ppl.log" | wc -l)"
if [ "$n" = $((STEPS - 1)) ] &&
  [ "$(e3_steps "$OUT/64e1ppl.log")" = "$(e3_steps "$OUT/64e3ppl.log")" ]; then
  echo "E2E ppl-decode e3==e1-hd64 steps=$n identical=1"
else
  echo "E2E FAIL ppl-decode e3 vs e1 (steps=$n)"; fail=1
fi
if [ $rc_e1 = 1 ] && grep -q '^E2E FAIL set_decode_graph_desc: AEE_ESCHEMENOTSUPPORTED' "$OUT/tiny_e1.log"; then
  echo "E2E fwd tiny all-kinds refused: AEE_ESCHEMENOTSUPPORTED"
else
  echo "E2E FAIL all kinds at head_dim 8 not refused (rc=$rc_e1)"; fail=1
fi
if [ $rc_add = 1 ] && grep -q '^E2E FAIL set_decode_graph_desc: AEE_ENOTALLOWED' "$OUT/add_nonorm.log"; then
  echo "E2E fwd tiny ADD-without-RMSNORM refused: AEE_ENOTALLOWED"
else
  echo "E2E FAIL ADD without RMSNORM not refused (rc=$rc_add)"; fail=1
fi

# (i) [#141] dspqueue: the same DSP function on the same bytes, so every
# MoE call and every logit is bit-identical to the switch-off run; the on
# line once, and the close line's served = the ARM's calls = the manifest's
# M==1 MoE calls, bad = 0.
m1_calls() { awk '$2 == "moe_layer" && $3 == 1 && $7 == 0' "$1/manifest.txt" | wc -l; }
for d in "tiny dspq dump_htp dump_dspq" "lfm25 25dspq dump_25off dump_25dspq" \
  "hmx dspqhmx dump_hmx dump_dspqhmx"; do
  read -r fx tag ref dump <<< "$d"
  $EVAL --label "dspq-$fx" "$OUT/$ref" "$OUT/$dump" | tail -1 || fail=1
  n="$(m1_calls "$OUT/$dump")"
  if [ "$n" -gt 0 ] && [ "$(grep -c '^\[HTP\] dspq: on ' "$OUT/$tag.log")" = 1 ] &&
    grep -q "^\[HTP\] dspq: close calls=$n served=$n bad=0 " "$OUT/$tag.log"; then
    echo "E2E dspq on-lines $fx calls=$n ok"
  else
    echo "E2E FAIL dspq $fx: on/close lines (manifest M==1 calls=$n)"
    grep '^\[HTP\] dspq' "$OUT/$tag.log" || true
    fail=1
  fi
done
grep -q 'dsp_spin_us=0 ' "$OUT/25dspq.log" ||
  { echo "E2E FAIL dspq lfm25: NNTR_HTP_DSPQ_SPIN_US=0 not applied"; fail=1; }
off_line="$($EVAL --label dspq-off "$OUT/dump_htp" "$OUT/dump_dspqoff" | tail -1 || true)"
banner="$(grep -c '^\[HTP\] dspq: off (create 0x[0-9a-f]*) -- MoE calls stay on FastRPC$' "$OUT/dspqoff.log" || true)"
if [ "$banner" = 1 ] && grep -q 'bit_identical=1' <<< "$off_line" &&
  ! grep -q 'dspq: close' "$OUT/dspqoff.log"; then
  echo "E2E dspq off-path banner=1 bit_identical=1"
else
  echo "E2E FAIL dspq off-path banner=$banner ($off_line)"; fail=1
fi

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
# [#150] NNTR_OP_TIME=1 is inert (same dumps and nll lines as the unset
# self run above) and its log makes a table; the unset log has no line.
echo "== [#150] run() path, NNTR_OP_TIME=1 against the unset self run"
NNTR_OP_TIME=1 NNTR_PPL_DECODE="$OUT/t.ids" \
  run_e2e run-optime "$OUT/htp" htp "$OUT/dump_gtime" "$OUT/gtime.log" --run > /dev/null
t_eval="$($EVAL --label op-time "$OUT/dump_gself" "$OUT/dump_gtime" | tail -1 || true)"
echo "$t_eval"
unset_lines="$(grep -c '\[OP-TIME\]' "$OUT/gself.log" || true)"
if python3 "$ROOT/tools/htp/op_time_report.py" "$OUT/gtime.log" > "$OUT/gtime.txt" &&
  grep -q '^unattributed ' "$OUT/gtime.txt" && [ "$unset_lines" = 0 ] &&
  [ "$(dec_steps "$OUT/gself.log")" = "$(dec_steps "$OUT/gtime.log")" ] &&
  grep -q 'bit_identical=1' <<< "$t_eval"; then
  cat "$OUT/gtime.txt"
  echo "E2E op-time inert bit_identical=1 nll=identical unset_lines=0 table=ok"
else
  cat "$OUT/gtime.txt"
  echo "E2E FAIL op-time (unset_lines=$unset_lines, $t_eval)"; fail=1
fi
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
