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
# and, since #132 Part B E3 (one PD since #211), the one-PD token in this
# process (NNTR_HTP_E2E=1: every kind, the FC set on S1's arena chunks, the
# pool's miss rounds on the page; one dspqueue packet a token). Not the
# device's PD -- one address space, no cache maintenance, no transport
# time -- but every ARM-side step of the device path:
#   E2E eval e3==e1-hd64 / -lfm25 ... bit_identical=1  (logits, the Android-
#                              order E1 run of the same model)
#   E2E fwd hd64 / lfm25 e3 calls/token=1.00 hops/token=0.00
#                              timeouts=0 id_mismatch=0 unmap_fail=0 ok
#   E2E tokens e3-run==e1-run-lfm25 8/8 id_only=1      (run(): only the id
#                              comes back, the CPU takes it)
#   E2E tokens e3-run==e1-run-lfm25-ban 8/8 id_only=1  (bad_word_ids = the
#                              token the unbanned run picks: the DSP's argmax
#                              skips it, LM_BAN, as the CPU's -inf does)
#   E2E ppl-decode e3==e1-hd64 steps=7 identical=1     (NNTR_PPL_DECODE
#                              forced on E1's path: the logits come back)
#   E2E teardown 25off / 25e3 arena chunks unmapped n/n mapped_kib=0
#                              freed_while_mapped=0  (E5i: the S1 arena
#                              released on the DSP, then unmapped, before
#                              the close; hybrid and E2E)
#   E2E arena retry cap=200 refused=1 bit_identical=1  (E5g: a boot whose
#                              last 256 MiB window is taken; the stand-in
#                              refuses maps past 200 MiB and keeps a refused
#                              fd, as the driver did; the MoE chunk halves
#                              to 128 MiB, the FC set's 64 fits beside it
#                              (#211: one PD holds both), and the run
#                              equals the uncapped one)
#   E2E e3 pool C=2 hd64 / C=1 lfm25 / C=2 lfm25 == e3 bit_identical=1
#                              misses=<n> calls/token=1.00 timeouts=0
#                              (plan 201 S1: NNTR_MOE_CACHE_EXPERTS with
#                              NNTR_HTP_E2E=1, the miss rounds served by the
#                              ARM's pool server; logits against the
#                              all-resident E run of the same fixture)
#   E2E e2e pds=2 refused ok   (#211: NNTR_HTP_E2E_PDS is a guard, the
#                              two-PD path is gone)
# and, since plan 201 S4, the Gemma 4 MoE fixture at head_dim 64 / global
# 128 (gemma4_moe_tiny_hd64: two KV cache widths, k = v full layer,
# seeded norms and layer_scalar, the final soft-cap) through #4296's CPU
# model (gemma4_moe, QS4CX experts), the HTP MoE layer alone (lfm2_moe's
# softmax router, QS4CX_WH, GeGLU) and NNTR_HTP_E2E=1 (the list built
# after load, its parameters handed by name), all-resident and pooled:
#   E2E gemma64 tokens off==cpu 8/8 expected_mismatch=0   (the HTP MoE
#                              layer against #4296's CPU one)
#   E2E eval gemma64-off-vs-cpu ... min_snr_db=<x>     (printed)
#   E2E fwd gemma64 e3 calls/token=1.00 attn_caches=2 timeouts=0 ok
#   E2E eval gemma64-e3 ... min_snr_db=<x>             (x >= 20 gated, vs the
#                              off run; measured 27.6, a swapped gamma,
#                              router scale, q | k gamma, up / gate or a
#                              dropped layer_scalar reads 0-10)
#   E2E tokens gemma64 e3==off 8/8 expected_mismatch=0
#   E2E e3 pool C=2 gemma64 == e3 bit_identical=1 misses=<n> calls/token=1.00 timeouts=0
# and, since plan 229 S1 (#4410's QS2CX_WH: 2-bit expert codes and a
# per-tensor palette of four int4 values), the one-PD token on 2-bit
# experts against the 4-bit model restricted to the same four levels
# (QS4CX_WH --moe_palette_g max, the "palette twin"): prefill on the HMX
# path's VTCM expansion, decode on the u8i2 LUT GEMV, the same int32 sums,
# so every MoE call and every logit equal the twin's; and through the pool:
#   E2E 2bit lfm25 e3 == palette-twin bit_identical=1 calls/token=1.00 timeouts=0
#   E2E tokens 2bit==twin-lfm25 8/8 expected_mismatch=0
#   E2E 2bit pool C=1 / C=2 lfm25 == 2bit e3 bit_identical=1 misses=<n> calls/token=1.00 timeouts=0
#   E2E 2bit gemma64 e3 == palette-twin bit_identical=1 calls/token=1.00 timeouts=0
#   E2E tokens 2bit==twin-gemma64 8/8 expected_mismatch=0
#   E2E 2bit pool C=2 gemma64 == 2bit e3 bit_identical=1 misses=<n> calls/token=1.00 timeouts=0
#   E2E keys lfm25 prefill-moved=1 htp_fc_rows=<n> e3==e1 bit_identical=1 pool C=2
#     bit_identical=1 misses=<n> calls/token=1.00 q4m1_handles=23
#     cpu-fc-skipped=<n> per_token=10 ok
#                              (#222: the config of record's engine keys and
#                              init_seq_len 1024 on the lfm25 fixture)
#   E2E keys fcwh-lfm25 handles=<n> arena_kib=<n> heap=0 requant=0 main=same
#     heap-path heap_kib=<n> e3 calls/token=1.00 ok
#   E2E eval fcwh==wh-lfm25 ... bit_identical=1
#   E2E eval fcwh-nokeys==off-lfm25 ... bit_identical=1 ... sidecar=unopened ok
#   E2E keys lfm25-p2x prompt=1024 chunks=512 ... logits bit_identical=1 ok
#                              (#225: the keys reading the FC WH sidecar,
#                              its images in the arena and through the heap
#                              path; P1024 in 512-row prefill chunks)
#   E2E quant fcwh-gemma64 main=same images=<n> ok
#                              (#234 P3: the Gemma 4 MoE writer's sidecar:
#                              main .bin byte-identical to the flagless
#                              run's, one image per graph FC -- 7 a layer,
#                              6 on a full-attention layer with
#                              attention_k_eq_v, which has no _wv)
#   E2E fwd lfm25 fcwh kinds=all calls/token=1.00 q4m1_handles=1 wh_handles=28
#     e3==e1 bit_identical=1 pool C=2 bit_identical=1 misses=<n> ok
#   E2E eval e3fcwh-lfm25-vs-off ... min_snr_db=<x>   (printed, not gated)
#   E2E tokens e3fcwh==off-lfm25 8/8 expected_mismatch=0
#   E2E ppl-decode e3fcwh-lfm25 off=<ppl> wh=<ppl> delta=<%> top1=7/7
#                              (#225 PR 2: the one-PD token's FC and dense
#                              FFN on the sidecar's WH handles, the lm_head
#                              alone Q4M1; NNTR_PPL_DECODE forced on the
#                              hybrid's continuation)
#   E2E fwd gemma64 fcwh kinds=all calls/token=1.00 attn_caches=2
#     q4m1_handles=1 wh_handles=17 ok
#   E2E eval gemma64-fcwh-vs-cpu ... min_snr_db=<x>   (printed, not gated)
#   E2E tokens gemma64-fcwh==off <n>/8 expected_mismatch=<m>   (the policy)
#   E2E eval gemma64x-fcwh-vs-q4m1 ... min_snr_db=<x>  (x >= 20 gated)
#   E2E tokens gemma64x-fcwh==q4m1 8/8 expected_mismatch=0
#   E2E e3 pool C=2 gemma64-fcwh == e3 bit_identical=1 misses=<n> ...
#     moe_dumps==sidecar-less files=<n> bit_identical=1
#                              (#234 P4: the Gemma 4 sidecar model's one-PD
#                              token, opened for the bind with no keyed FC;
#                              gemma64x: the same fixture with int4-exact
#                              FCs, so both sides hold the same weights)
# and, since #194 S1 (htp_moe_ppl), the same token with lever
# L1 (NNTR_HTP_PPL_LEVERS=2: the native FC / DENSE_FFN / LM_HEAD kernels,
# q4_gemv_native_det.h), forced on E1's hd64 path, and on lfm25:
#   E2E ppl-decode hd64 levers=0x2 e0=<ppl> l1=<ppl> delta=<%> ok   (|delta|
#                              <= 2 % gated: a random-weight fixture's PPL,
#                              the pipeline and the numerics' order of
#                              magnitude, not the 8B's)
#   E2E eval l1-lfm25 ... min_snr_db=<x>               (x >= 30 gated, vs E0;
#                              bit_identical=0, else the lever is not wired)
#   E2E levers banner e0=0x0 l1=0x2 ok
#   E2E L0 wake split closes: disp=.. pkt=.. ret=.. clk_resid=..
#   E2E ppl-decode alts steps=7 ok                     (NNTR_PPL_DECODE_ALTS:
#                              the step lines gain the ids' logits)
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
FIXG="$ROOT/test/unittest/models/causallm_reference/gemma4_moe_tiny_hd64"
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
GENG="test/unittest/models/causallm_reference/generators/generate_gemma4_moe_reference.py"
if [ ! -f "$FIXG/nntr_gemma4_moe_tiny_fp32.bin" ]; then
  echo "E2E FAIL gemma hd64 fixture weights missing: run" >&2
  echo "  python3 $GENG --hidden 128 --inter 64 --heads 2 --kv-heads 1 --head-dim 64 --global-head-dim 128 --global-kv-heads 1 --layer-types sliding_attention,sliding_attention,full_attention --max-pos 32 --sliding-window 8 --experts 8 --top-k 2 --moe-inter 32 --final-softcap 30 --random-layer-scalar --random-norms --router-scale 50 --seed 12 --out $FIXG" >&2
  echo "  git checkout -- $FIXG/" >&2
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
logits_only() { mkdir -p "$2" && cp "$1"/logits_*.f32 "$2/"; }
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
# [#132 Part B E3, #211] the one-PD token (NNTR_HTP_E2E=1; its packets
# ride dspqueue, so these runs set NNTR_HTP_DSPQ=1)
echo "== [#132 Part B E3] hd64 / lfm25, NNTR_HTP_E2E=1 (one PD)"
NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1 \
  run_e2e q64-e3 "$OUT/htp64q" htp "$OUT/dump_64e3" "$OUT/64e3.log" --max-seq 32
PROMPT=512 NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1 \
  run_e2e q25-e3 "$OUT/htp25q" htp "$OUT/dump_25e3" "$OUT/25e3.log" --max-seq 2048
PROMPT=512 NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=$E1_KINDS \
  run_e2e q25-e1run "$OUT/htp25q" htp "$OUT/dump_25e1run" "$OUT/25e1run.log" --max-seq 2048 --run
PROMPT=512 NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1 \
  run_e2e q25-e3run "$OUT/htp25q" htp "$OUT/dump_25e3run" "$OUT/25e3run.log" --max-seq 2048 --run
# the same model with the unbanned run's first decode pick as a bad word
cp -a "$OUT/htp25q" "$OUT/htp25qb"
ban="$(sed -n 's/^E2E gen [0-9]* \([0-9]*\).*/\1/p' "$OUT/25e1run.log")"
sed -i "s/\"bad_word_ids\": \[\]/\"bad_word_ids\": [${ban:-0}]/" "$OUT/htp25qb/nntr_config.json"
PROMPT=512 NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=$E1_KINDS \
  run_e2e q25-e1ban "$OUT/htp25qb" htp "$OUT/dump_25e1ban" "$OUT/25e1ban.log" --max-seq 2048 --run
PROMPT=512 NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1 \
  run_e2e q25-e3ban "$OUT/htp25qb" htp "$OUT/dump_25e3ban" "$OUT/25e3ban.log" --max-seq 2048 --run
rm -f "$OUT/e3.ids"
NNTR_PPL_DECODE="$OUT/e3.ids" NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=$E1_KINDS \
  run_e2e q64-e1ppl "$OUT/htp64q" htp "$OUT/dump_64e1ppl" "$OUT/64e1ppl.log" --max-seq 32 --run
NNTR_PPL_DECODE="$OUT/e3.ids" NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1 \
  run_e2e q64-e3ppl "$OUT/htp64q" htp "$OUT/dump_64e3ppl" "$OUT/64e3ppl.log" --max-seq 32 --run
PROMPT=512 NNTR_HTP_PROFILE=1 NNTR_INPROC_MMAP_CAP_MIB=200 NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1 \
  run_e2e q25-e3cap "$OUT/htp25q" htp "$OUT/dump_25e3cap" "$OUT/25e3cap.log" --max-seq 2048
# [plan 201 S1] the expert pool (NNTR_MOE_CACHE_EXPERTS) inside the
# per-token entry: S1's miss rounds served by the ARM's pool server, a pool
# of half the experts (hd64 C=2: 4 of 8; lfm25 C=2: 8 of 16) and a quarter
# (lfm25 C=1), against the all-resident E runs above
NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=2 \
  run_e2e q64-e3pool2 "$OUT/htp64q" htp "$OUT/dump_64e3pool2" "$OUT/64e3pool2.log" --max-seq 32
for c in 1 2; do
  PROMPT=512 NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=$c \
    run_e2e q25-e3pool$c "$OUT/htp25q" htp "$OUT/dump_25e3pool$c" "$OUT/25e3pool$c.log" --max-seq 2048
done
# [plan 201 S4] the Gemma 4 MoE fixture: #4296's CPU model (QS4CX
# experts), the HTP MoE layer alone (QS4CX_WH), the E2E token, and the
# token with a pool of 2 of the 8 experts a layer
"$Q" "$FIXG" -o "$OUT/g64cpu" --fc_dtype Q4_0 --moe_dtype QS4CX \
  --embd_dtype Q4_0 > "$OUT/q_g64cpu.log"
"$Q" "$FIXG" -o "$OUT/g64htp" --fc_dtype Q4_0 --moe_dtype QS4CX_WH \
  --embd_dtype Q4_0 > "$OUT/q_g64htp.log"
# [#234 P3] the same with the FC WH sidecar: checked below; run since P4
"$Q" "$FIXG" -o "$OUT/g64htpw" --fc_dtype Q4_0 --moe_dtype QS4CX_WH \
  --embd_dtype Q4_0 --fc_wh_sidecar > "$OUT/q_g64htpw.log"
run_gemma() { # run_gemma <label> <model> <engine> [env...]: run_e2e on FIXG
  local label=$1 model=$2 engine=$3
  shift 3
  local rc=0
  env "$@" "$E2E" --model "$OUT/$model" --tokenizer "$FIXG/tokenizer.json" \
    --prompt $PROMPT --steps $STEPS --moe-engine "$engine" \
    --dump "$OUT/dump_$label" --max-seq 32 ${G_ARGS:-} > "$OUT/$label.log" 2>&1 || rc=$?
  if [ $rc != 0 ] || ! grep -q '^E2E gen ' "$OUT/$label.log"; then
    echo "E2E FAIL $label rc=$rc"; tail -3 "$OUT/$label.log"; exit 1
  fi
}
run_gemma g64cpu g64cpu cpu
run_gemma g64off g64htp htp
run_gemma g64e3 g64htp htp NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1
run_gemma g64e3pool2 g64htp htp NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=2
# [#234 P4] the sidecar model's one-PD token (FC / DENSE_FFN on its WH
# handles), all experts resident and a pool of 2, and the sidecar-less
# token beside them; --repack as the app's main runs it (repack_weight is
# where the loader hands the sidecar over)
G_ARGS=--repack run_gemma g64e3r g64htp htp NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1
G_ARGS=--repack run_gemma g64e3w g64htpw htp NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1
G_ARGS=--repack run_gemma g64e3wpool2 g64htpw htp NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=2
# and gemma64x, the bind check's fixture: every FC weight int4-exact (per
# column q * 2^e, each 32-row block holding a -8 and the column a 7, so
# Q4_0 and the sidecar's QS4CX_WH hold the same values) and the q / k norm
# gammas at 0.3. The plain fixture's two quantizations read -4 dB apart
# (Q4_0 against f32 FCs reads the same), and its peaked sliding softmax
# turns the WH path's u8 row quantization of q and k into 9.5 dB even on
# int4-exact weights; at 0.3 the WH token reads 31 dB against the Q4M1 one
python3 - "$ROOT/tools/htp" "$FIXG" "$OUT/fixg_x" <<'PY'
import json, math, os, shutil, struct, sys
tools, src, dst = sys.argv[1:4]
sys.path.insert(0, tools)
from fc_wh_sidecar_from_q4 import fcs
shutil.copytree(src, dst)
cfg = json.load(open(src + "/config.json"))
c = cfg.get("text_config", cfg)
b = dst + "/" + json.load(open(src + "/nntr_config.json"))["model_file_name"]
rows, size = fcs(cfg, dict.fromkeys(
    ("fc_layer_dtype", "embedding_dtype", "moe_layer_dtype"), "FP32"))
if os.path.getsize(b) not in (size, size + 4 * c["vocab_size"] * c["hidden_size"]):
    sys.exit("gemma64x: the f32 layout walk disagrees with the file size")
f = open(b, "r+b")
for name, o, K, N in rows:
    f.seek(o)
    w = list(struct.unpack("<%df" % (K * N), f.read(4 * K * N)))
    for n in range(N):
        s = 2.0 ** round(math.log2(max(abs(w[k * N + n]) for k in range(K)) / 8))
        for k in range(K):
            q = -8 if k % 32 == 0 else 7 if k == 1 else round(w[k * N + n] / s)
            w[k * N + n] = max(-8, min(7, q)) * s
    if name.endswith(("_wq", "_wk")):  # q_norm / k_norm follow
        hd = c["head_dim"] if c["layer_types"][int(name[5:name.index("_")])] \
            == "sliding_attention" else c["global_head_dim"]
        w += [0.3] * hd
    f.seek(o)
    f.write(struct.pack("<%df" % len(w), *w))
PY
"$Q" "$OUT/fixg_x" -o "$OUT/g64x" --fc_dtype Q4_0 --moe_dtype QS4CX_WH \
  --embd_dtype Q4_0 > "$OUT/q_g64x.log"
"$Q" "$OUT/fixg_x" -o "$OUT/g64xw" --fc_dtype Q4_0 --moe_dtype QS4CX_WH \
  --embd_dtype Q4_0 --fc_wh_sidecar > "$OUT/q_g64xw.log"
G_ARGS=--repack run_gemma g64xe3 g64x htp NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1
G_ARGS=--repack run_gemma g64xe3w g64xw htp NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1
# [plan 229] QS2CX_WH experts and their 4-bit palette twin (header above)
"$Q" "$FIX25" -o "$OUT/htp25q2" --fc_dtype Q4_0 --moe_dtype QS2CX_WH \
  --embd_dtype Q4_0 > "$OUT/q_htp25q2.log"
"$Q" "$FIX25" -o "$OUT/htp25qp" --fc_dtype Q4_0 --moe_dtype QS4CX_WH \
  --moe_palette_g max --embd_dtype Q4_0 > "$OUT/q_htp25qp.log"
PROMPT=512 NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1 \
  run_e2e q25-e3p "$OUT/htp25qp" htp "$OUT/dump_25e3p" "$OUT/25e3p.log" --max-seq 2048
PROMPT=512 NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1 \
  run_e2e q25-e3q2 "$OUT/htp25q2" htp "$OUT/dump_25e3q2" "$OUT/25e3q2.log" --max-seq 2048
for c in 1 2; do
  PROMPT=512 NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=$c \
    run_e2e q25-e3q2pool$c "$OUT/htp25q2" htp "$OUT/dump_25e3q2pool$c" "$OUT/25e3q2pool$c.log" --max-seq 2048
done
"$Q" "$FIXG" -o "$OUT/g64htp2" --fc_dtype Q4_0 --moe_dtype QS2CX_WH \
  --embd_dtype Q4_0 > "$OUT/q_g64htp2.log"
"$Q" "$FIXG" -o "$OUT/g64htpp" --fc_dtype Q4_0 --moe_dtype QS4CX_WH \
  --moe_palette_g max --embd_dtype Q4_0 > "$OUT/q_g64htpp.log"
run_gemma g64e3p g64htpp htp NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1
run_gemma g64e3q2 g64htp2 htp NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1
run_gemma g64e3q2pool2 g64htp2 htp NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=2
# [#222] the config of record's keys on the same model: conv_block,
# dense_ffn and attn_proj on the HTP (their prefill on the HTP FC kernels,
# load-time registration and warm-ups), init_seq_len 1024; hybrid (off),
# the one-session E1 run and the one-PD run, the last with a pool of C=2
# (8 of the fixture's 16 experts)
cp -a "$OUT/htp25q" "$OUT/htp25k"
python3 - "$OUT/htp25k/nntr_config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(conv_block_engine="htp", dense_ffn_engine="htp",
         attn_proj_engine="htp", init_seq_len=1024)
json.dump(c, open(sys.argv[1], "w"), indent=4)
PY
echo "== [#222] lfm25 with the engine keys: off, KINDS=all, NNTR_HTP_E2E=1, pool C=2"
PROMPT=512 NNTR_HTP_PROFILE=1 run_e2e q25k-off "$OUT/htp25k" htp "$OUT/dump_25koff" "$OUT/25koff.log" --max-seq 2048
PROMPT=512 NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=$E1_KINDS \
  run_e2e q25k-e1a "$OUT/htp25k" htp "$OUT/dump_25ke1a" "$OUT/25ke1a.log" --max-seq 2048
PROMPT=512 NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1 \
  run_e2e q25k-e3 "$OUT/htp25k" htp "$OUT/dump_25ke3" "$OUT/25ke3.log" --max-seq 2048
PROMPT=512 NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=2 \
  run_e2e q25k-e3pool2 "$OUT/htp25k" htp "$OUT/dump_25ke3pool2" "$OUT/25ke3pool2.log" --max-seq 2048
# [#225] the same keys on the model packed with the FC WH sidecar (its main
# file must be htp25q's): hybrid with the images in the arena, the same
# with every image through the heap path (NNTR_HTP_FC_WH_HEAP=1), the
# one-PD run (decode still on Q4M1 in PR 1); then prompt 1024 in 512-row
# prefill chunks (the default) against one call (NNTR_HTP_PREFILL_ROWS=0).
# --repack: the app's load (main.cpp), where the sidecar is opened and the
# FCs registered before the first prefill
"$Q" "$FIX25" -o "$OUT/htp25w" --fc_dtype Q4_0 --moe_dtype QS4CX_WH \
  --embd_dtype Q4_0 --fc_wh_sidecar > "$OUT/q_htp25w.log"
w_main_same=0
cmp -s "$OUT/htp25q/$(sed -n 's/.*"model_file_name": "\(.*\)".*/\1/p' "$OUT/htp25q/nntr_config.json")" \
  "$OUT/htp25w/$(sed -n 's/.*"model_file_name": "\(.*\)".*/\1/p' "$OUT/htp25w/nntr_config.json")" && w_main_same=1
cp -a "$OUT/htp25w" "$OUT/htp25wn" # no keys: the sidecar named, never opened
python3 - "$OUT/htp25w/nntr_config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(conv_block_engine="htp", dense_ffn_engine="htp",
         attn_proj_engine="htp", init_seq_len=1024)
json.dump(c, open(sys.argv[1], "w"), indent=4)
PY
echo "== [#225] lfm25 keys + FC WH sidecar: arena, heap, NNTR_HTP_E2E=1, P1024 chunked / whole"
PROMPT=512 run_e2e q25w-off "$OUT/htp25w" htp "$OUT/dump_25woff" "$OUT/25woff.log" --max-seq 2048 --repack
PROMPT=512 run_e2e q25w-nokeys "$OUT/htp25wn" htp "$OUT/dump_25wn" "$OUT/25wn.log" --max-seq 2048 --repack
PROMPT=512 NNTR_HTP_FC_WH_HEAP=1 \
  run_e2e q25w-heap "$OUT/htp25w" htp "$OUT/dump_25wheap" "$OUT/25wheap.log" --max-seq 2048 --repack
PROMPT=512 NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1 \
  run_e2e q25w-e3 "$OUT/htp25w" htp "$OUT/dump_25we3" "$OUT/25we3.log" --max-seq 2048 --repack
PROMPT=1024 NNTR_HTP_PROFILE=1 \
  run_e2e q25w-p2x "$OUT/htp25w" htp "$OUT/dump_25wp2x" "$OUT/25wp2x.log" --max-seq 2048 --repack
PROMPT=1024 NNTR_HTP_PREFILL_ROWS=0 NNTR_HTP_PROFILE=1 \
  run_e2e q25w-p1x "$OUT/htp25w" htp "$OUT/dump_25wp1x" "$OUT/25wp1x.log" --max-seq 2048 --repack
# [#225 PR 2] the decode FC set on the sidecar's WH handles: the one-session
# E1 and the pool of 8 against the one-PD run above, then NNTR_PPL_DECODE:
# the hybrid writes its continuation, the one-PD run is forced on it
echo "== [#225 PR 2] lfm25 sidecar: KINDS=all, NNTR_HTP_E2E=1 pool C=2, PPL forced"
PROMPT=512 NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=$E1_KINDS \
  run_e2e q25w-e1 "$OUT/htp25w" htp "$OUT/dump_25we1" "$OUT/25we1.log" --max-seq 2048 --repack
PROMPT=512 NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=2 \
  run_e2e q25w-e3pool2 "$OUT/htp25w" htp "$OUT/dump_25we3pool2" "$OUT/25we3pool2.log" --max-seq 2048 --repack
rm -f "$OUT/w.ids"
PROMPT=512 NNTR_PPL_DECODE="$OUT/w.ids" \
  run_e2e q25w-offppl "$OUT/htp25w" htp "$OUT/dump_25woffppl" "$OUT/25woffppl.log" --max-seq 2048 --repack --run
PROMPT=512 NNTR_PPL_DECODE="$OUT/w.ids" NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1 \
  run_e2e q25w-e3ppl "$OUT/htp25w" htp "$OUT/dump_25we3ppl" "$OUT/25we3ppl.log" --max-seq 2048 --repack --run
# [#211] NNTR_HTP_E2E_PDS is a guard: anything but 1 is refused at load
rc_pds=0
NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1 NNTR_HTP_E2E_PDS=2 "$E2E" --model "$OUT/htp64q" \
  --tokenizer "$FIX/tokenizer.json" --prompt $PROMPT --steps $STEPS \
  --moe-engine htp --max-seq 32 > "$OUT/64pds2.log" 2>&1 || rc_pds=$?
tail -1 "$OUT/64pds2.log"
# [#194 S1] lever L1 on the same token
NNTR_PPL_DECODE="$OUT/e3.ids" NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1 NNTR_HTP_PPL_LEVERS=2 \
  run_e2e q64-l1ppl "$OUT/htp64q" htp "$OUT/dump_64l1ppl" "$OUT/64l1ppl.log" --max-seq 32 --run
PROMPT=512 NNTR_HTP_DSPQ=1 NNTR_HTP_E2E=1 NNTR_HTP_PPL_LEVERS=2 \
  run_e2e q25-l1 "$OUT/htp25q" htp "$OUT/dump_25l1" "$OUT/25l1.log" --max-seq 2048
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
for d in "64e3 64e3pool2 hd64 2" "25e3 25e3pool1 lfm25 1" "25e3 25e3pool2 lfm25 2"; do
  set -- $d
  ev="$($EVAL --label "e3pool-$3-C$4" "$OUT/dump_$1" "$OUT/dump_$2" | tail -1 || true)"
  calls="$(calls_per_token "$OUT/$2.log")"
  close="$(grep -o 'token driver: close .*' "$OUT/$2.log")"
  misses="$(sed -n 's/.*token driver: pool misses=\([0-9]*\) .*/\1/p' "$OUT/$2.log")"
  if grep -q 'bit_identical=1' <<< "$ev" && [ "$calls" = 1.00 ] &&
     grep -q ' timeouts=0 stale=0 ' <<< "$close" && [ "${misses:-0}" -gt 0 ]; then
    echo "E2E e3 pool C=$4 $3 == e3 bit_identical=1 misses=$misses calls/token=1.00 timeouts=0"
  else
    echo "E2E FAIL e3 pool C=$4 $3: [$ev] calls/token=${calls:-none} misses=${misses:-none} close=[$close]"; fail=1
  fi
done
# [plan 201 S4] the Gemma lines: the HTP MoE layer against #4296's CPU
# model by tokens (its SNR printed: QS4CX on both, the HTP's quantizer and
# GeGLU against the CPU's); the E2E token one call a token on both
# attention caches, its logits within the floor of the off run and its
# tokens by the policy; the pool bit-identical to the all-resident token.
# The floor: the off run reads 27.6 dB at worst, every hand-over mutant
# 0-10 (plan 201 S4, PR body), 20 sits between.
for d in g64cpu g64off g64e3 g64e3pool2; do logits_only "$OUT/dump_$d" "$OUT/ref_$d"; done
$EVAL --label 'gemma64 off==cpu' --tokens-policy "$OUT/ref_g64cpu" "$OUT/ref_g64off" | tail -1 || fail=1
$EVAL --label gemma64-off-vs-cpu --allow-diff "$OUT/ref_g64cpu" "$OUT/ref_g64off" | tail -1 || true
calls="$(calls_per_token "$OUT/g64e3.log")"
caches="$(grep -c '^\[HTP\] attn_m1: registered ' "$OUT/g64e3.log" || true)"
close="$(grep -o 'token driver: close .*' "$OUT/g64e3.log")"
if [ "$calls" = 1.00 ] && [ "$caches" = 2 ] && grep -q ' timeouts=0 stale=0 ' <<< "$close"; then
  echo "E2E fwd gemma64 e3 calls/token=1.00 attn_caches=2 timeouts=0 ok"
else
  echo "E2E FAIL gemma64 e3: calls/token=${calls:-none} attn_caches=$caches close=[$close]"; fail=1
fi
$EVAL --label gemma64-e3 --allow-diff --snr-floor 20 "$OUT/ref_g64off" "$OUT/ref_g64e3" | tail -1 || fail=1
$EVAL --label 'gemma64 e3==off' --tokens-policy "$OUT/ref_g64off" "$OUT/ref_g64e3" | tail -1 || fail=1
ev="$($EVAL --label e3pool-gemma64-C2 "$OUT/ref_g64e3" "$OUT/ref_g64e3pool2" | tail -1 || true)"
calls="$(calls_per_token "$OUT/g64e3pool2.log")"
close="$(grep -o 'token driver: close .*' "$OUT/g64e3pool2.log")"
misses="$(sed -n 's/.*token driver: pool misses=\([0-9]*\) .*/\1/p' "$OUT/g64e3pool2.log")"
if grep -q 'bit_identical=1' <<< "$ev" && [ "$calls" = 1.00 ] &&
   grep -q ' timeouts=0 stale=0 ' <<< "$close" && [ "${misses:-0}" -gt 0 ]; then
  echo "E2E e3 pool C=2 gemma64 == e3 bit_identical=1 misses=$misses calls/token=1.00 timeouts=0"
else
  echo "E2E FAIL e3 pool C=2 gemma64: [$ev] calls/token=${calls:-none} misses=${misses:-none} close=[$close]"; fail=1
fi
# [#234 P4] the sidecar model's one-PD token: the bind banner (the tied
# lm_head the one Q4M1 handle, every FC part and dense chunk pair a WH
# handle: 2 sliding layers x (q k v o + gate|up, down) + the full layer's
# k = v (no v) = 6 + 6 + 5), one call a token on both caches. Against the
# CPU model (SNR printed) and the off run (tokens by the policy) the plain
# fixture compares two int4 quantizations of its FCs; the gate is
# gemma64x's, same weights on both sides: the WH token against the Q4M1
# token, its logits above 20 dB (a wrong part or chunk reads below 0) and
# its tokens 8/8. Then its prefill MoE dumps are the sidecar-less token's
# (the experts did not move; a Gemma prefill FC stays on the CPU, no
# keys) and its pool of 2 bit-identical to it (logits: the pool splits
# prefill calls)
for d in g64e3r g64e3w g64e3wpool2 g64xe3 g64xe3w; do logits_only "$OUT/dump_$d" "$OUT/ref_$d"; done
gq="$(sed -n 's/^\[HTP\] graph: q4m1 weights=[0-9]* handles=\([0-9]*\) feed=[a-z0-9]* wh_handles=\([0-9]*\)$/\1 \2/p' "$OUT/g64e3w.log")"
read -r g_q4m1 g_wh <<< "${gq:-x x}"
calls="$(calls_per_token "$OUT/g64e3w.log")"
caches="$(grep -c '^\[HTP\] attn_m1: registered ' "$OUT/g64e3w.log" || true)"
close="$(grep -o 'token driver: close .*' "$OUT/g64e3w.log")"
if [ "$g_q4m1" = 1 ] && [ "$g_wh" = 17 ] && [ "$calls" = 1.00 ] &&
   [ "$caches" = 2 ] && grep -q ' timeouts=0 stale=0 ' <<< "$close" &&
   grep -q ' wh_handles=17$' "$OUT/g64xe3w.log"; then
  echo "E2E fwd gemma64 fcwh kinds=all calls/token=1.00 attn_caches=2 q4m1_handles=$g_q4m1 wh_handles=$g_wh ok"
else
  echo "E2E FAIL fwd gemma64 fcwh: bind=[${gq:-none}] calls/token=${calls:-none} attn_caches=$caches close=[$close]"; fail=1
fi
$EVAL --label gemma64-fcwh-vs-cpu --allow-diff "$OUT/ref_g64cpu" "$OUT/ref_g64e3w" | tail -1 || true
$EVAL --label 'gemma64-fcwh==off' --tokens-policy "$OUT/ref_g64off" "$OUT/ref_g64e3w" | tail -1 || fail=1
$EVAL --label gemma64x-fcwh-vs-q4m1 --allow-diff --snr-floor 20 "$OUT/ref_g64xe3" "$OUT/ref_g64xe3w" | tail -1 || fail=1
$EVAL --label 'gemma64x-fcwh==q4m1' --tokens-policy "$OUT/ref_g64xe3" "$OUT/ref_g64xe3w" | tail -1 || fail=1
mkdir -p "$OUT/moe_g64e3r" "$OUT/moe_g64e3w"
for d in g64e3r g64e3w; do
  find "$OUT/dump_$d" -maxdepth 1 -type f ! -name 'logits_*' -exec cp {} "$OUT/moe_$d/" \;
done
em="$($EVAL --label gemma64-fcwh-moe==e3 "$OUT/moe_g64e3r" "$OUT/moe_g64e3w" | tail -1 || true)"
ev="$($EVAL --label e3pool-gemma64-fcwh-C2 "$OUT/ref_g64e3w" "$OUT/ref_g64e3wpool2" | tail -1 || true)"
calls="$(calls_per_token "$OUT/g64e3wpool2.log")"
close="$(grep -o 'token driver: close .*' "$OUT/g64e3wpool2.log")"
misses="$(sed -n 's/.*token driver: pool misses=\([0-9]*\) .*/\1/p' "$OUT/g64e3wpool2.log")"
nm="$(sed -n 's/.* files=\([0-9]*\) .*/\1/p' <<< "$em")"
if grep -q 'bit_identical=1' <<< "$ev" && grep -q 'bit_identical=1' <<< "$em" &&
   [ "${nm:-0}" -gt 0 ] && [ "$calls" = 1.00 ] &&
   grep -q ' timeouts=0 stale=0 ' <<< "$close" && [ "${misses:-0}" -gt 0 ]; then
  echo "E2E e3 pool C=2 gemma64-fcwh == e3 bit_identical=1 misses=$misses calls/token=1.00 timeouts=0 moe_dumps==sidecar-less files=$nm bit_identical=1"
else
  echo "E2E FAIL e3 pool C=2 gemma64-fcwh: pool=[$ev] moe=[$em] calls/token=${calls:-none} misses=${misses:-none} close=[$close]"; fail=1
fi
# [plan 229] the 2-bit lines: every dumped file (the prefill MoE calls'
# inputs and outputs, the logits) of the 2-bit token equals its palette
# twin's, one call a token, no timeout; the pool's logits equal the
# all-resident 2-bit token's with misses served (logits: a pool smaller
# than a prefill layer's experts splits its calls, as the 4-bit gemma64
# line above).
two_bit() { # two_bit <label> <ref run> <2-bit run> <pool runs...>
  local label=$1 ref=$2 got=$3 p ev calls close misses
  shift 3
  ev="$($EVAL --label "2bit-$label" "$OUT/dump_$ref" "$OUT/dump_$got" | tail -1 || true)"
  calls="$(calls_per_token "$OUT/$got.log")"
  close="$(grep -o 'token driver: close .*' "$OUT/$got.log")"
  if grep -q 'bit_identical=1' <<< "$ev" && [ "$calls" = 1.00 ] &&
     grep -q ' timeouts=0 stale=0 ' <<< "$close"; then
    echo "E2E 2bit $label e3 == palette-twin bit_identical=1 calls/token=1.00 timeouts=0"
  else
    echo "E2E FAIL 2bit $label: [$ev] calls/token=${calls:-none} close=[$close]"; fail=1
  fi
  $EVAL --label "2bit==twin-$label" --tokens-policy "$OUT/dump_$ref" "$OUT/dump_$got" | tail -1 || fail=1
  logits_only "$OUT/dump_$got" "$OUT/ref_$got"
  for p in "$@"; do
    logits_only "$OUT/dump_$p" "$OUT/ref_$p"
    ev="$($EVAL --label "2bit-$label-$p" "$OUT/ref_$got" "$OUT/ref_$p" | tail -1 || true)"
    calls="$(calls_per_token "$OUT/$p.log")"
    close="$(grep -o 'token driver: close .*' "$OUT/$p.log")"
    misses="$(sed -n 's/.*token driver: pool misses=\([0-9]*\) .*/\1/p' "$OUT/$p.log")"
    if grep -q 'bit_identical=1' <<< "$ev" && [ "$calls" = 1.00 ] &&
       grep -q ' timeouts=0 stale=0 ' <<< "$close" && [ "${misses:-0}" -gt 0 ]; then
      echo "E2E 2bit pool C=${p##*pool} $label == 2bit e3 bit_identical=1 misses=$misses calls/token=1.00 timeouts=0"
    else
      echo "E2E FAIL 2bit pool $p: [$ev] calls/token=${calls:-none} misses=${misses:-none} close=[$close]"; fail=1
    fi
  done
}
two_bit lfm25 25e3p 25e3q2 25e3q2pool1 25e3q2pool2
two_bit gemma64 g64e3p g64e3q2 g64e3q2pool2
# [#222] with the keys: the prefill moved (step 0's logits differ from the
# keyless run's), and the decode paths hold as without them -- the one-PD
# token equal to E1's of the same model, one call per token, the pool
# equal to the all-resident run, 23 handles (the keys' one-layer forms
# hand the same weights), the CPU's FC work skipped at every resident row
# (conv_block 4, qkv 2, attention_out 2, dense_ffn 2), and E1 against the
# keys' own hybrid run by the tokens policy
mkdir -p "$OUT/k0a" "$OUT/k0b"
cp "$OUT/dump_25qoff/logits_0.f32" "$OUT/k0a/"; cp "$OUT/dump_25koff/logits_0.f32" "$OUT/k0b/"
k0="$($EVAL --label keys-prefill --allow-diff --snr-floor 0 "$OUT/k0a" "$OUT/k0b" | tail -1 || true)"
k3="$($EVAL --label e3==e1-lfm25-keys "$OUT/dump_25ke1a" "$OUT/dump_25ke3" | tail -1 || true)"
kp="$($EVAL --label e3pool-lfm25-keys-C2 "$OUT/dump_25ke3" "$OUT/dump_25ke3pool2" | tail -1 || true)"
kt="$($EVAL --label 'e1==off-lfm25-keys' --tokens-policy "$OUT/dump_25koff" "$OUT/dump_25ke1a" | tail -1 || true)"
echo "$k0"; echo "$k3"; echo "$kp"; echo "$kt"
calls="$(calls_per_token "$OUT/25ke3.log")"
calls_p="$(calls_per_token "$OUT/25ke3pool2.log")"
handles="$(sed -n 's/^\[HTP\] graph: q4m1 weights=[0-9]* handles=\([0-9]*\) .*/\1/p' "$OUT/25ke1a.log")"
skipped="$(sed -n 's/^\[HTP\] graph: cpu fc skipped=\([0-9]*\)$/\1/p' "$OUT/25ke1a.log")"
toks="$(sed -n 's/.*forward calls=[0-9]* tokens=\([0-9]*\) .*/\1/p' "$OUT/25ke1a.log")"
misses="$(sed -n 's/.*token driver: pool misses=\([0-9]*\) .*/\1/p' "$OUT/25ke3pool2.log")"
rows="$(grep -cE 'HTP-PROFILE\]   K=[0-9 ]+N=[0-9 ]+M>1 (conv|dense|FC) +calls=[1-9]' "$OUT/25koff.log" || true)"
if grep -q 'bit_identical=0' <<< "$k0" && grep -q 'bit_identical=1' <<< "$k3" &&
  [ "${rows:-0}" -ge 3 ] &&
  grep -q 'bit_identical=1' <<< "$kp" && grep -q '^E2E tokens' <<< "$kt" && ! grep -q 'unexpected=' <<< "$kt" &&
  [ "$calls" = 1.00 ] && [ "$calls_p" = 1.00 ] && [ "${misses:-0}" -gt 0 ] &&
  [ "$handles" = 23 ] && [ -n "$toks" ] && [ "${skipped:-0}" = $((10 * toks)) ]; then
  echo "E2E keys lfm25 prefill-moved=1 htp_fc_rows=$rows e3==e1 bit_identical=1 pool C=2 bit_identical=1 misses=$misses calls/token=1.00 q4m1_handles=23 cpu-fc-skipped=$skipped per_token=10 ok"
else
  echo "E2E FAIL keys lfm25: htp_fc_rows=${rows:-none} calls/token=${calls:-none}/${calls_p:-none} handles=${handles:-none} skipped=${skipped:-none} tokens=${toks:-none} misses=${misses:-none}"; fail=1
fi
# [#225] (1) the sidecar's banner: every FC prefill weight from the file,
# none re-quantized, nothing on the heap -- and with NNTR_HTP_FC_WH_HEAP=1
# everything on it; (2) fcwh==wh: the images as they lie in the file (the
# arena, the way get_or_register_wh places a WH tensor) against the same
# images through whUnpack and the DSP's bake (the hybrid's overflow), every
# MoE call and the logits; (3) the one-PD run on the sidecar model;
# (3b) no engine key: the sidecar named in the config, never opened, the
# logits those of the keyless model without one (the CPU-only and Aoff
# configs of the handoff); (4) P1024 in two 512-row prefill chunks against
# one call, logits; the chunked run makes more MoE calls, which says it
# chunked; (5) printed, not
# gated: the keys' logits against the keyless hybrid (CPU Q4_0 FCs), the
# sidecar's and the load-time re-quantization's, side by side
wb="$(sed -n 's/^\[HTP\] fc wh: file=.* handles=\([0-9]*\) arena_kib=\([0-9]*\) heap_kib=\([0-9]*\) requant=\([0-9]*\)$/\1 \2 \3 \4/p' "$OUT/25woff.log")"
wh_heap="$(sed -n 's/^\[HTP\] fc wh: file=.* arena_kib=\([0-9]*\) heap_kib=\([0-9]*\) requant=0$/\1 \2/p' "$OUT/25wheap.log")"
read -r w_handles w_arena w_heap w_requant <<< "${wb:-x x x x}"
read -r h_arena h_heap <<< "${wh_heap:-x x}"
calls_w="$(calls_per_token "$OUT/25we3.log")"
if [ "$w_main_same" = 1 ] && [ "$w_heap" = 0 ] && [ "$w_requant" = 0 ] &&
  [ "${w_handles:-0}" -gt 0 ] 2> /dev/null && [ "$h_arena" = 0 ] &&
  [ "$h_heap" = "$w_arena" ] && [ "$calls_w" = 1.00 ]; then
  echo "E2E keys fcwh-lfm25 handles=$w_handles arena_kib=$w_arena heap=0 requant=0 main=same heap-path heap_kib=$h_heap e3 calls/token=1.00 ok"
else
  echo "E2E FAIL keys fcwh-lfm25: banner=[${wb:-none}] heap-run=[${wh_heap:-none}] main_same=$w_main_same calls/token=${calls_w:-none}"; fail=1
fi
$EVAL --label 'fcwh==wh-lfm25' "$OUT/dump_25woff" "$OUT/dump_25wheap" | tail -1 || fail=1
mkdir -p "$OUT/p1x" "$OUT/p2x" "$OUT/lq" "$OUT/lk" "$OUT/lw" "$OUT/lwn"
cp "$OUT"/dump_25wn/logits_*.f32 "$OUT/lwn/"
nk="$($EVAL --label fcwh-nokeys==off-lfm25 "$OUT/lwn" "$OUT/dump_25qoff" | tail -1 || true)"
if grep -q 'bit_identical=1' <<< "$nk" && ! grep -q '^\[HTP\] fc wh:' "$OUT/25wn.log"; then
  echo "$nk sidecar=unopened ok"
else
  echo "E2E FAIL fcwh-nokeys-lfm25: [$nk]"; fail=1
fi
cp "$OUT"/dump_25wp1x/logits_*.f32 "$OUT/p1x/"; cp "$OUT"/dump_25wp2x/logits_*.f32 "$OUT/p2x/"
n1="$(grep -c moe_layer "$OUT/dump_25wp1x/manifest.txt" || true)"
n2="$(grep -c moe_layer "$OUT/dump_25wp2x/manifest.txt" || true)"
p2="$($EVAL --label lfm25-p2x "$OUT/p1x" "$OUT/p2x" | tail -1 || true)"
# [#236] the M>1 FC rows' calls / rows summed: the qkv and o_proj FCs step
# by prefillRows() too, so the chunked run carries the same rows in calls of
# 512 (every FC call of this prompt is 1024 or 512 rows) and the whole run
# in fewer
fc_sum() { sed -n 's/.*M>1 FC *calls=\([0-9]*\) *rows=\([0-9]*\).*/\1 \2/p' "$1" |
  awk '{c+=$1; r+=$2} END {print c+0, r+0}'; }
read -r f2 r2 <<< "$(fc_sum "$OUT/25wp2x.log")"
read -r f1 r1 <<< "$(fc_sum "$OUT/25wp1x.log")"
if grep -q 'bit_identical=1' <<< "$p2" && [ "${n2:-0}" -gt "${n1:-0}" ] &&
  [ "$f1" -gt 0 ] && [ "$f2" -gt "$f1" ] && [ "$f2" = $((r2 / 512)) ] &&
  [ $((r2 % 512)) = 0 ] && [ "$r2" = "$r1" ]; then
  echo "E2E keys lfm25-p2x prompt=1024 chunks=512 moe_calls=$n2 whole=$n1 fc_calls=$f2 whole=$f1 fc_rows=$r2 logits bit_identical=1 ok"
else
  echo "E2E FAIL keys lfm25-p2x: [$p2] moe_calls=$n2 whole=$n1 fc_calls=$f2/$r2 whole=$f1/$r1"; fail=1
fi
# [#234 P3] the Gemma writer's sidecar: the main file untouched, and the
# index names exactly the graph FCs the config's layers hold
gw="$(python3 - "$OUT/g64htp" "$OUT/g64htpw" "$FIXG/config.json" <<'PY'
import filecmp, json, struct, sys
a, b, cfg = sys.argv[1:4]
ca, cb = (json.load(open(d + "/nntr_config.json")) for d in (a, b))
same = filecmp.cmp(a + "/" + ca["model_file_name"],
                   b + "/" + cb["model_file_name"], shallow=False)
c = json.load(open(cfg))
c = c.get("text_config", c)
kv = c.get("attention_k_eq_v", False)
want = []
for i, t in enumerate(c["layer_types"]):
    sfx = ["wq", "wk"] + ([] if kv and t == "full_attention" else ["wv"])
    sfx += ["attention_out", "ffn_gate", "ffn_up", "ffn_down"]
    want += ["layer%d_%s" % (i, s) for s in sfx]
f = open(b + "/" + cb["fc_wh_file_name"], "rb").read()
n = struct.unpack_from("<I", f, 12)[0]
names = [f[16 + 104 * i:16 + 104 * i + 64].rstrip(b"\0").decode()
         for i in range(n)]
ok = (f[:8] == b"NNTRFCWH" and cb.get("fc_wh_format") == "QS4CX_WH/1"
      and "fc_wh_file_name" not in ca and sorted(names) == sorted(want))
print("main=%s images=%d want=%d %s" % ("same" if same else "differs", n,
                                         len(want), "ok" if same and ok else "bad"))
PY
)" || gw="error"
if [ "${gw##* }" = ok ]; then
  echo "E2E quant fcwh-gemma64 ${gw% want=* ok} ok"
else
  echo "E2E FAIL quant fcwh-gemma64: [$gw]"; fail=1
fi
cp "$OUT"/dump_25qoff/logits_*.f32 "$OUT/lq/"; cp "$OUT"/dump_25koff/logits_*.f32 "$OUT/lk/"
cp "$OUT"/dump_25woff/logits_*.f32 "$OUT/lw/"
$EVAL --label fcwh-lfm25-vs-cpu-fc --allow-diff "$OUT/lq" "$OUT/lw" | tail -1 || true
$EVAL --label requant-lfm25-vs-cpu-fc --allow-diff "$OUT/lq" "$OUT/lk" | tail -1 || true
# [#225 PR 2] (6) the one-PD token's FC and dense FFN ops on the sidecar's
# WH handles (HTP_GRAPH_FEED_WH), the lm_head alone Q4M1 (one slice: vocab
# 32): the bind banner's counts on the one-PD and the one-session run, one
# call a token, the one-PD token equal to E1's and to the pool C=2 run's,
# every logit; (7) against the hybrid, whose decode FCs are the CPU's Q4_0:
# the tokens by the policy, and the decode logits' SNR printed, not gated
# -- two int4 quantizations of the fixture's random FCs (per column from
# f32, and Q4_0 blocks with a Q8_0 row; PR 1's prefill line above reads
# 2.5 dB for the same reason), so it is the size of the format change, and
# (6) is the wiring check
wq="$(sed -n 's/^\[HTP\] graph: q4m1 weights=[0-9]* handles=\([0-9]*\) feed=[a-z0-9]* wh_handles=\([0-9]*\)$/\1 \2/p' "$OUT/25we3.log")"
wq1="$(sed -n 's/^\[HTP\] graph: q4m1 weights=[0-9]* handles=\([0-9]*\) feed=[a-z0-9]* wh_handles=\([0-9]*\)$/\1 \2/p' "$OUT/25we1.log")"
read -r w_q4m1 w_wh <<< "${wq:-x x}"
we="$($EVAL --label e3fcwh==e1fcwh-lfm25 "$OUT/dump_25we1" "$OUT/dump_25we3" | tail -1 || true)"
wp="$($EVAL --label e3fcwh-pool-C2-lfm25 "$OUT/dump_25we3" "$OUT/dump_25we3pool2" | tail -1 || true)"
echo "$we"; echo "$wp"
calls_wp="$(calls_per_token "$OUT/25we3pool2.log")"
misses_w="$(sed -n 's/.*token driver: pool misses=\([0-9]*\) .*/\1/p' "$OUT/25we3pool2.log")"
if [ "$w_q4m1" = 1 ] && [ "$w_wh" = 28 ] && [ "$wq1" = "$wq" ] &&
  [ "$calls_w" = 1.00 ] && [ "$calls_wp" = 1.00 ] && [ "${misses_w:-0}" -gt 0 ] &&
  grep -q 'bit_identical=1' <<< "$we" && grep -q 'bit_identical=1' <<< "$wp"; then
  echo "E2E fwd lfm25 fcwh kinds=all calls/token=1.00 q4m1_handles=$w_q4m1 wh_handles=$w_wh e3==e1 bit_identical=1 pool C=2 bit_identical=1 misses=$misses_w ok"
else
  echo "E2E FAIL fwd lfm25 fcwh: bind=[${wq:-none}] e1=[${wq1:-none}] calls/token=${calls_w:-none}/${calls_wp:-none} misses=${misses_w:-none}"; fail=1
fi
mkdir -p "$OUT/lwo" "$OUT/lw3"
cp "$OUT"/dump_25woff/logits_*.f32 "$OUT/lwo/"; cp "$OUT"/dump_25we3/logits_*.f32 "$OUT/lw3/"
$EVAL --label e3fcwh-lfm25-vs-off --allow-diff "$OUT/lwo" "$OUT/lw3" | tail -1 || true
$EVAL --label 'e3fcwh==off-lfm25' --tokens-policy "$OUT/dump_25woff" "$OUT/dump_25we3" | tail -1 || fail=1
if [ $rc_pds = 1 ] &&
  grep -q '^E2E FAIL .*NNTR_HTP_E2E_PDS=2: the two-PD path was removed (#211)' "$OUT/64pds2.log"; then
  echo "E2E e2e pds=2 refused ok"
else
  echo "E2E FAIL NNTR_HTP_E2E_PDS=2 not refused (rc=$rc_pds)"; fail=1
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
# (h3) [#132 Part B E3, #211] one PD: every logit equal to the one-session
# E1 run (Android order) of the same model, one packet per token, no hop,
# no timeout, the DSP's argmax equal to the logits' first maximum, nothing
# left mapped; run() with only the id back equals E1's run(); the PPL
# decode step lines (17 digits) equal E1's
for d in "hd64 64e3 64e1a" "lfm25 25e3 25e1a"; do
  read -r fx tag ref <<< "$d"
  $EVAL --label "e3==e1-$fx" "$OUT/dump_$ref" "$OUT/dump_$tag" | tail -1 || fail=1
  close="$(grep '^\[HTP\] token driver: close ' "$OUT/$tag.log" || true)"
  calls="$(calls_per_token "$OUT/$tag.log")"
  if [ "$calls" = 1.00 ] && grep -q " hops/token=0.00 " <<< "$close" &&
    grep -q ' timeouts=0 stale=0 ' <<< "$close" &&
    grep -q ' id_mismatch=0 ' <<< "$close" &&
    grep -q '^\[HTP\] e2e: fc arena weights=' "$OUT/$tag.log" &&
    grep -q '^\[HTP\] e2e: close .* unmap_fail=0 detach_fail=0 ' "$OUT/$tag.log"; then
    echo "E2E fwd $fx e3 calls/token=$calls hops/token=0.00 timeouts=0 id_mismatch=0 unmap_fail=0 ok"
  else
    echo "E2E FAIL $fx e3: calls/token=${calls:-none} close=[$close]"; fail=1
  fi
done
cap_eval="$($EVAL --label e3cap "$OUT/dump_25e3" "$OUT/dump_25e3cap" | tail -1 || true)"
refused=$(grep -c 'arena: [0-9]* MiB refused' "$OUT/25e3cap.log" || true)
if grep -q 'bit_identical=1' <<< "$cap_eval" && [ "$refused" -ge 1 ]; then
  echo "E2E arena retry cap=200 refused=$refused bit_identical=1"
else
  echo "E2E FAIL arena retry under a cap: refused=$refused [$cap_eval]"; fail=1
fi
# [#132 Part B E5i] every S1 arena chunk unmapped before the session
# closes, on the one-session path and on E's; nothing mapped at exit and no
# buffer freed while mapped (the stand-in refuses an unmap the DSP side
# still holds)
for t in 25off 25e3; do
  un="$(sed -n 's/.*arena: chunks unmapped \([0-9]*\)\/\([0-9]*\) .*/\1 \2/p' "$OUT/$t.log")"
  ex="$(grep -o 'INPROC rpc exit: .*' "$OUT/$t.log" | head -1)"
  # the arena goes after the other hooks (E: after the e2e close line)
  late="$(grep -n 'arena: chunks unmapped\|e2e: close' "$OUT/$t.log" | tail -1)"
  if read -r a b <<< "$un" && [ -n "$a" ] && [ "$a" = "$b" ] && [ "$a" -gt 0 ] &&
    [ "$ex" = "INPROC rpc exit: mapped_kib=0 freed_while_mapped=0" ] &&
    grep -q 'arena: chunks unmapped' <<< "$late"; then
    echo "E2E teardown $t arena chunks unmapped $a/$b mapped_kib=0 freed_while_mapped=0"
  else
    echo "E2E FAIL teardown $t: unmapped=[${un:-none}] [$ex]"; fail=1
  fi
done
for d in "25e1run 25e3run lfm25" "25e1ban 25e3ban lfm25-ban"; do
  read -r r1 r3 label <<< "$d"
  e1_gen="$(grep '^E2E gen ' "$OUT/$r1.log")"
  e3_gen="$(grep '^E2E gen ' "$OUT/$r3.log")"
  same=$(paste <(tr ' ' '\n' <<< "$e1_gen") <(tr ' ' '\n' <<< "$e3_gen") |
    tail -n +3 | awk '$1==$2{n++} END{print n+0}')
  id_only=0
  grep -q '^\[HTP\] token driver: first token .* logits=0$' "$OUT/$r3.log" && id_only=1
  echo "E2E tokens e3-run==e1-run-$label $same/$STEPS id_only=$id_only"
  [ "$same" = "$STEPS" ] && [ $id_only = 1 ] || fail=1
done
[ "$(grep '^E2E gen ' "$OUT/25e1ban.log")" != "$(grep '^E2E gen ' "$OUT/25e1run.log")" ] ||
  { echo "E2E FAIL the ban changed no token (ban=${ban:-none})"; fail=1; }
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
# [#194] NNTR_PPL_DECODE_ALTS=16,23: the forced run's step lines gain
# alts=16:<logit>,23:<logit> and are otherwise the same; 16 is the greedy
# pick at every step here, so its logit is the larger (the index is right)
NNTR_PPL_DECODE="$OUT/g.ids" NNTR_PPL_DECODE_ALTS=16,23 \
  run_e2e run-alts "$OUT/htp" htp "$OUT/dump_galts" "$OUT/galts.log" --run > /dev/null
n_alts="$(dec_steps "$OUT/galts.log" | awk '$NF ~ /^alts=16:[^,]*,23:/ {split(substr($NF, 6), a, "[:,]"); n += (a[2] + 0 >= a[4] + 0)} END {print n + 0}')"
if [ "$n_alts" = $((STEPS - 1)) ] &&
  [ "$(dec_steps "$OUT/galts.log" | sed 's/ alts=.*//')" = "$(dec_steps "$OUT/gforced.log")" ]; then
  echo "E2E ppl-decode alts steps=$n_alts ok"
else
  echo "E2E FAIL ppl-decode alts (steps with alts, 16 >= 23: $n_alts)"; fail=1
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
# [#225 PR 2] (8) the sidecar model's one-PD token (WH FC set) forced on the
# hybrid's continuation (CPU Q4_0 decode FCs): the host shape of the
# device's G4 read, on random weights -- printed; gated: forced, finite
po="$(dec_field "$OUT/25woffppl.log" ppl)"
pw="$(dec_field "$OUT/25we3ppl.log" ppl)"
line="E2E ppl-decode e3fcwh-lfm25 off=${po:-none} wh=${pw:-none}"
if [ "$(dec_field "$OUT/25woffppl.log" source)" = self ] &&
  [ "$(dec_field "$OUT/25we3ppl.log" source)" = file ] &&
  awk -v a="$po" -v b="$pw" 'BEGIN{exit !(a + 0 > 0 && b + 0 > 0 && a + 0 < 1e30 && b + 0 < 1e30)}'; then
  echo "$line delta=$(awk -v a="$po" -v b="$pw" 'BEGIN{printf "%+.3f%%", (b / a - 1) * 100}') top1=$(dec_field "$OUT/25we3ppl.log" top1)"
else
  echo "E2E FAIL $line (not finite, or not forced)"; fail=1
fi

# (j) [#194 S1] lever L1: the hd64 PPL forced on E1's path against E0's
# (the unset run, q64-e3ppl), lfm25's logits against E0's by SNR (and not
# bit-identical: an unwired lever would read 999 dB), the banners
e0="$(dec_field "$OUT/64e3ppl.log" ppl)"
l1="$(dec_field "$OUT/64l1ppl.log" ppl)"
if [ "$(dec_field "$OUT/64l1ppl.log" source)" = file ] &&
  awk -v a="$e0" -v b="$l1" 'BEGIN{d = (b / a - 1) * 100; if (d < 0) d = -d;
    exit !(a + 0 > 0 && b + 0 > 0 && d <= 2)}'; then
  echo "E2E ppl-decode hd64 levers=0x2 e0=$e0 l1=$l1 delta=$(awk -v a="$e0" -v b="$l1" 'BEGIN{printf "%+.3f%%", (b / a - 1) * 100}') ok"
else
  echo "E2E FAIL ppl-decode hd64 levers=0x2 e0=${e0:-none} l1=${l1:-none}"; fail=1
fi
l1_line="$($EVAL --label l1-lfm25 --allow-diff --snr-floor 30 "$OUT/dump_25e3" "$OUT/dump_25l1" | tail -1)" || fail=1
echo "$l1_line"
grep -q 'bit_identical=0' <<< "$l1_line" ||
  { echo "E2E FAIL l1-lfm25 bit-identical to E0: the lever is not wired"; fail=1; }
# [#194 L0] the wake split closes on one clock: disp + the DSP's packet
# time + its wall + ret = the ARM's round trip, to within 50 us a token,
# nothing negative
wk="$(grep -h 'L0 wake us/token' "$OUT/25e3.log" || true)"
if awk -v l="$wk" 'BEGIN{n = split(l, f, "[ =]"); ok = n > 0; for (i = 1; i < n; ++i) {
    if (f[i] == "clk_resid") { r = f[i + 1]; if (r < 0) r = -r; ok = ok && r < 50; seen = 1 }
    if (f[i] == "disp" || f[i] == "pkt" || f[i] == "ret") ok = ok && f[i + 1] + 0 >= 0 }
    exit !(ok && seen)}'; then
  echo "E2E L0 wake split closes: ${wk#*us/token }"
else
  echo "E2E FAIL L0 wake split [$wk]"; fail=1
fi
if grep -q '^\[HTP\] ppl levers=0x0 L1=exact$' "$OUT/64e3.log" &&
  grep -q '^\[HTP\] ppl levers=0x2 L1=native_fc$' "$OUT/25l1.log"; then
  echo "E2E levers banner e0=0x0 l1=0x2 ok"
else
  echo "E2E FAIL levers banner"; fail=1
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
