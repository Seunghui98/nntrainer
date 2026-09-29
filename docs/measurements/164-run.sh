#!/usr/bin/env bash
# Plan 164 step 8 (docs/measurements/164-cpu-order-norm.md on
# htp/164-cpu-order-norm): the RMSNORM / QK_NORM row scale in the Android
# CPU's order, gated on silicon. One set, built from dev/norm-shadow (= the
# PR diff + the inert measurement commit). Variants: A (switch off, first),
# R = MOE,RMSNORM, Q = MOE,QK_NORM,ROPE,ATTN_M1, RQ = both.
# Usage: bash /local/mnt/workspace/htp_moe/164/run_164.sh [serial]   (~55 min)
# Everything is logged to $L; the summary to $L/sitting.out. The script stops
# on the plan's stop rules; every other mismatch is counted and printed.
set -u -o pipefail
WT=/home/j2z0-lee/nntrainer-shadow          # dev/norm-shadow: tools + host check
S=${1:-R3CY10WM83Y}
W=/local/mnt/workspace/htp_moe/164; L=$W/logs; mkdir -p $L $W/dump $W/shadow
OLD=/local/mnt/workspace/htp_moe/norm       # the 2026-09-29 norm-shadow sitting
C=/data/local/tmp/nntrainer/causallm; D=$C/s164; M=../models/q40-qs4cx-wh; DD=/data/local/tmp/s164dump
cd $WT && { set +u; source tools/htp/env.sh > /dev/null 2>&1; set -u; }
exec > >(tee -a $L/sitting.out) 2>&1
MIS=0
stop() { echo "STOP: $*"; exit 1; }
want() { # want <label> <got> <expected>
  if [ "$2" = "$3" ]; then echo "OK  $1 = $2"; else echo "BAD $1: got '$2' want '$3'"; MIS=$((MIS + 1)); fi; }
therm() { echo "$1 $(date +%H:%M:%S) $(adb -s $S shell 'dumpsys battery | grep -E "^  (level|temperature)"; cat /sys/class/thermal/thermal_zone0/temp' | tr -d '\r' | tr '\n' ' ')" | tee -a $L/therm.log; }
zone0() { adb -s $S shell cat /sys/class/thermal/thermal_zone0/temp | tr -d '\r'; }
F="NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS"
env_of() { case $1 in
  A*) echo "";;
  R) echo "$F=MOE,RMSNORM";;
  Q) echo "$F=MOE,QK_NORM,ROPE,ATTN_M1";;
  RQ) echo "$F=MOE,RMSNORM,QK_NORM,ROPE,ATTN_M1";;
esac; }
banner() { case $1 in  # the line a switch-on log must carry (rule 36), and calls/token
  R) echo "graph: init n_ops=228 resident=RMSNORM|MOE moe_ops=22|calls/token=71.00";;
  Q) echo "graph: init n_ops=228 resident=QK_NORM|ROPE|ATTN_M1|MOE moe_ops=22|calls/token=28.00";;
  RQ) echo "graph: init n_ops=228 resident=RMSNORM|QK_NORM|ROPE|ATTN_M1|MOE moe_ops=22|calls/token=77.00";;
esac; }
run() { # run <variant> <G> <log name> <prompt file> [extra env ...]
  local v=$1 g=$2 log=$3 p=$4; shift 4
  adb -s $S shell "cd $D && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $g/' $M/nntr_config.json && \
    grep num_to_generate $M/nntr_config.json && md5sum libnntr_hvx_skel.so && \
    $(env_of $v) $* NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm $M \"\$(cat $p)\"" \
    > $L/$log.log 2>&1
  grep -qi '0x8000040e' $L/$log.log && stop "$log: 0x8000040e (stale skel, rule 3)"
  echo "$log: $(grep -h -E '^(prefill|generation):' $L/$log.log | grep -o '[0-9.]* TPS' | tr '\n' ' ')$(grep -h -o 'calls/token=[0-9.]*' $L/$log.log) $(grep -h -o '\[PPL\] decode tokens=.*' $L/$log.log | cut -c1-90)"
}
vbanner() { # vbanner <variant> <log>: the switch-on banner / A's dspq pair, else void
  local f=$L/$2.log
  if [ "${1#A}" != "$1" ]; then
    want "$2 dspq: on" "$(grep -c 'dspq: on' $f)" 1
    want "$2 dspq close bad=0" "$(grep -c 'dspq: close calls=\([0-9]*\) served=\1 bad=0' $f)" 1
    want "$2 no graph" "$(grep -c 'graph: init' $f)" 0
  else
    local b; b=$(banner $1)
    want "$2 banner" "$(grep -cF "${b%|*}" $f)" 1
    want "$2 ${b#*|}" "$(grep -cF "${b#*|}" $f)" 1
  fi; }
# The generated text of a log: everything before the summary, minus banners
# and the [PPL] lines (measurement 152 section 5's strip).
strip() { sed -n '/^=====/q;p' "$1" | perl -0pe 's/\[HTP\] [^\n]*\n//g; s/\[PPL\] [^\n]*\n//g' |
  grep -v 'moe m1 gemv\|libnntr_hvx_skel\|nntrainer_causallm\|num_to_generate'; }
nll() { grep -o '\[PPL\] decode step=.*' "$1"; }
P="prompt512.txt bitset-02-code.txt bitset-03-math.txt bitset-04-korean.txt bitset-05-json.txt bitset-06-dialogue.txt bitset-07-facts.txt bitset-08-short.txt"

echo "=== 164 sitting  $(date '+%F %T %Z')  unit=$S  wt=$(git -C $WT rev-parse --short HEAD)"
# ---- 0. device state, cool start (1 min)
adb devices | tee $L/devices.log
adb devices | grep -q "^$S[[:space:]]*device" || stop "unit $S not attached"
adb -s $S shell input keyevent 223 || true
therm t0
for i in $(seq 1 40); do z=$(zone0); [ "$z" -le 35000 ] && break
  echo "zone0=$z > 35000, waiting 30 s ($i/40)"; sleep 30; done
echo "start zone0=$(zone0)"

# ---- 1. install and config (3 min)
[ "$(cd $W/set && LC_ALL=C md5sum -c md5.txt | grep -vc ': OK$')" = 0 ] || stop "workstation set differs from set/md5.txt"
adb -s $S shell ls -l $C/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin
adb -s $S shell "rm -rf $D $DD && mkdir -p $D $DD" && adb -s $S push $W/set/. $D/ > /dev/null
adb -s $S shell "rm -f $D/md5.txt; chmod 755 $D/nntrainer_causallm $D/unittest_*"
adb -s $S shell "cd $D && md5sum \$(ls -p | grep -v / | sort)" > $L/md5_device.log
diff <(sort -k2 $W/set/md5.txt | tr -d '\r') <(sort -k2 $L/md5_device.log | tr -d '\r') && echo "MD5 OK" || stop "device md5 differs from md5.txt"
adb -s $S shell "cd $D/$M && \
  sed -i 's/\"do_sample\": true/\"do_sample\": false/' generation_config.json && \
  sed -i 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' nntr_config.json && \
  (grep -q moe_engine nntr_config.json || sed -i 's/\"bad_word_ids\": \[124900\],/\"bad_word_ids\": [124900],\n    \"moe_engine\": \"htp\",/' nntr_config.json) && \
  grep -H do_sample generation_config.json && grep -H -E 'bad_word_ids|num_to_generate|init_seq_len|_engine|_htp_layers' nntr_config.json" \
  | tee $L/config.log
for k in '"do_sample": false' '"bad_word_ids": \[124900\]' '"init_seq_len": 512' '"moe_engine": "htp"' '"moe_htp_layers": ""'; do
  want "config $k" "$(grep -c "$k" $L/config.log)" 1; done

# ---- 2. (a) gtests: G0 (spec == this libnntrainer.so's CPU norm), G1 (kernel == spec) (4 min)
adb -s $S shell "cd $D && NNTR_NORM_REPLAY=$D/replay_norm.bin LD_LIBRARY_PATH=. \
  ./unittest_nntrainer_cpu_backend --gtest_filter='RmsNormCpuOrder.*'" > $L/gtest_G0.log 2>&1
grep -E '^RmsNormCpuOrder|OK \]|FAILED|PASSED|SKIPPED' $L/gtest_G0.log
want "G0 rows bad=0" "$(grep -c '^RmsNormCpuOrder rows=41041 bad=0' $L/gtest_G0.log)" 1
want "G0 subnormal rows bad=0" "$(grep -c '^RmsNormCpuOrder W=.* subnormal(+-1e-39) bad=0' $L/gtest_G0.log)" 2
want "G0 replay" "$(grep -c '^RmsNormCpuOrder replay .* rms=392 bad=0 qk_heads=1920 bad_calls=0' $L/gtest_G0.log)" 1
want "G0 PASSED 2" "$(grep -c 'PASSED  \] 2 tests' $L/gtest_G0.log)" 1
grep -q '^RmsNormCpuOrder rows=41041 bad=0' $L/gtest_G0.log || stop "G0: the spec is not this set's CPU norm (re-read the disassembly, plan step 8)"
adb -s $S shell "cd $D && md5sum libnntr_hvx_skel.so && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  ./unittest_hvx_softmax --gtest_filter='HvxM1Ops.*'" > $L/gtest_G1.log 2>&1
grep -E 'libnntr_hvx|M1_OPS_FIELD (rmsnorm|qk_norm)|OK \]|FAILED|PASSED' $L/gtest_G1.log
for k in 0 1 3; do want "G1 rmsnorm kind=$k" "$(grep -c "M1_OPS_FIELD rmsnorm kind=$k bad_y=0 bad_row_scale=0 of 2048" $L/gtest_G1.log)" 1; done
want "G1 rmsnorm sweep" "$(grep -c 'M1_OPS_FIELD rmsnorm sweep 2^-20..2^20 rows=41 bad_y=0 bad_row_scale=0' $L/gtest_G1.log)" 1
want "G1 qk_norm 32/8 heads" "$(grep -c 'M1_OPS_FIELD qk_norm heads=\(32\|8\) bad_y=0 bad_row_scale=0' $L/gtest_G1.log)" 2
want "G1 qk_norm sweep" "$(grep -c 'M1_OPS_FIELD qk_norm sweep 2^-20..2^20 heads=80 bad_y=0 bad_row_scale=0' $L/gtest_G1.log)" 1
echo "G1 subnormal kind (rule 37, reported; plan step 5): $(grep -h 'M1_OPS_FIELD rmsnorm kind=2' $L/gtest_G1.log)"
[ "$(grep -c 'M1_OPS_FIELD rmsnorm kind=[013] bad_y=0 bad_row_scale=0' $L/gtest_G1.log)" = 3 ] &&
  grep -q 'rmsnorm sweep .* bad_y=0 bad_row_scale=0' $L/gtest_G1.log &&
  grep -q 'qk_norm sweep .* bad_y=0 bad_row_scale=0' $L/gtest_G1.log ||
  stop "G1 fails on normal rows: stale skel or a silicon mismatch (plan step 8)"
therm t1

# ---- 3. (b) norm shadow, G2: prompt 512, G = 8, forced on A's tokens (4 min)
adb -s $S shell "rm -f $D/cont.ids"
for v in A R Q RQ; do
  adb -s $S shell "rm -rf $D/d_f_$v && mkdir -p $D/d_f_$v"
  run $v 8 f_$v prompt512.txt NNTR_NORM_SHADOW=d_f_$v/norm.bin NNTR_LOGIT_SHADOW=d_f_$v/logits.bin NNTR_PPL_DECODE=cont.ids
  vbanner $v f_$v
  rm -rf $W/shadow/d_f_$v && adb -s $S pull $D/d_f_$v $W/shadow/ > /dev/null && adb -s $S shell "rm -rf $D/d_f_$v"
done
want "f_A source=self" "$(grep -c 'decode tokens=8 .* source=self' $L/f_A.log)" 1
python3 tools/htp/norm_shadow_check.py $OLD/d_f_A $W/shadow/d_f_A | tee $L/shadow_check.txt
python3 tools/htp/norm_shadow_check.py $W/shadow/d_f_A $W/shadow/d_f_R $W/shadow/d_f_Q $W/shadow/d_f_RQ | tee -a $L/shadow_check.txt
want "G2 A (logits == 2026-09-29 A, DSP q|k == CPU)" "$(grep -c 'd_f_A tag0=0/0 tag1_heads=1920/1920 tag1_zero_records=0 tag2=392/392 logits_equal_steps=8/8' $L/shadow_check.txt)" 1
want "G2 R" "$(grep -c 'd_f_R tag0=392/392 tag1_heads=1920/1920 tag1_zero_records=0 tag2=0/0 logits_equal_steps=8/8' $L/shadow_check.txt)" 1
want "G2 Q" "$(grep -c 'd_f_Q tag0=0/0 tag1_heads=1920/1920 tag1_zero_records=0 tag2=392/392 logits_equal_steps=8/8' $L/shadow_check.txt)" 1
want "G2 RQ" "$(grep -c 'd_f_RQ tag0=392/392 tag1_heads=1920/1920 tag1_zero_records=0 tag2=0/0 logits_equal_steps=8/8' $L/shadow_check.txt)" 1
for v in R Q RQ; do cmp -s <(nll $L/f_A.log) <(nll $L/f_$v.log) && echo "G2 nll f_$v == f_A" || { echo "BAD G2 nll f_$v != f_A"; MIS=$((MIS + 1)); }; done
# host replay of this sitting's dumps: the kernel on hvx_emu == the CPU rows
B=$WT/nntrainer/tensor/htp_backend
gcc -std=c99 -O2 -ffp-contract=off -I $WT/test/htp/host/hvx_emu -I $B/.. -I $B/hvx -o $L/m1_ops_host_check \
  $WT/test/htp/host/m1_ops_host_check.c $B/hvx/hvx_m1_ops_f32.c $B/hvx/hvx_conv_gate_f32.c -lm &&
  $L/m1_ops_host_check --replay $W/set/gamma_rms.f32 $W/set/gamma_qk.f32 $W/shadow/d_f_{A,R,Q,RQ}/norm.bin | tee $L/host_replay.txt
want "host replay" "$(grep -c '^REPLAY rms=784/784 qk_heads=7680/7680 (rms_cpu=784/784)$' $L/host_replay.txt)" 1
therm t2

# ---- 4. (c) MoE dumps at G = 4, G3 (5 min)
dump() { # dump <name> <variant> [env ...]
  local n=$1 v=$2; shift 2
  adb -s $S shell "rm -rf $DD/$n && mkdir -p $DD/$n"
  run $v 4 dump_$n prompt512.txt NNTR_HTP_DUMP=$DD/$n "$@"
  rm -rf $W/dump/$n && adb -s $S pull $DD/$n $W/dump/$n > /dev/null && adb -s $S shell "rm -rf $DD/$n"
}
dump A1 A; dump A2 A; dump R R NNTR_HTP_DUMP_ALL=1; dump Q Q NNTR_HTP_DUMP_ALL=1
E="python3 tools/htp/htp_dump_eval.py"
for n in A2 R Q; do $E --label $n $W/dump/A1 $W/dump/$n | tail -1; done | tee $L/dump_eval.txt
for n in A2 R Q; do want "G3 dump $n bit_identical" "$(grep -c "^E2E eval $n .*bit_identical=1" $L/dump_eval.txt)" 1; done
therm t3

# ---- 5. (d) speed, prompt 512, mirrored A R Q RQ | RQ Q R A (14 min), G5 profile
for g in 64 512 1024; do
  for v in A R Q RQ; do run $v $g ${v}_G${g}_r1 prompt512.txt; vbanner $v ${v}_G${g}_r1; done
  for v in RQ Q R A; do run $v $g ${v}_G${g}_r2 prompt512.txt; vbanner $v ${v}_G${g}_r2; done
  therm t4_G$g
done
run RQ 64 prof_RQ prompt512.txt NNTR_HTP_PROFILE=2; vbanner RQ prof_RQ
grep -h 'graph: calls=' $L/prof_RQ.log | tee $L/g5.txt
rp=$(grep -o ' RMSNORM=[0-9]*' $L/g5.txt | cut -d= -f2); qp=$(grep -o ' QK_NORM=[0-9]*' $L/g5.txt | cut -d= -f2)
echo "G5 RMSNORM pcyc/op=${rp:-?} (<= 4000)  QK_NORM pcyc/op=${qp:-?} (<= 25000)"
want "G5 RMSNORM <= 4000" "$([ -n "$rp" ] && [ "$rp" -le 4000 ] && echo y || echo n)" y
want "G5 QK_NORM <= 25000" "$([ -n "$qp" ] && [ "$qp" -le 25000 ] && echo y || echo n)" y

# ---- 6. (e) G4: nll forced on A's continuation, text A vs RQ, 8 prompts at G = 256 (16 min)
i=0; for p in $P; do i=$((i+1))
  adb -s $S shell "rm -f $D/cont_p0$i.ids"
  run A 256 ppl_A_p0$i $p NNTR_PPL_DECODE=cont_p0$i.ids               # A's own continuation (source=self)
  [ $i = 1 ] && run A 256 ppl_Af_p01 $p NNTR_PPL_DECODE=cont_p01.ids   # null check: A forced == A self
  for v in R Q RQ; do run $v 256 ppl_${v}_p0$i $p NNTR_PPL_DECODE=cont_p0$i.ids; vbanner $v ppl_${v}_p0$i; done
  run A 256 text_A_p0$i $p; run RQ 256 text_RQ_p0$i $p; vbanner RQ text_RQ_p0$i
done
therm t5

# ---- 7. checks and summary (2 min)
echo "--- G4 nll (every [PPL] decode step line equal to A's, 17 digits)"
cmp -s <(nll $L/ppl_A_p01.log) <(nll $L/ppl_Af_p01.log) && echo "null check A forced == A self" || { echo "BAD A forced != A self"; MIS=$((MIS + 1)); }
for i in 1 2 3 4 5 6 7 8; do for v in R Q RQ; do
  if cmp -s <(nll $L/ppl_A_p0$i.log) <(nll $L/ppl_${v}_p0$i.log); then echo "p0$i $v nll == A"; else echo "BAD p0$i $v nll != A"; MIS=$((MIS + 1)); fi
done; done
echo "--- G4 text RQ vs A (G = 256)"
for i in 1 2 3 4 5 6 7 8; do
  if cmp -s <(strip $L/text_A_p0$i.log) <(strip $L/text_RQ_p0$i.log); then echo "p0$i text identical"; else echo "BAD p0$i text DIFFERENT"; MIS=$((MIS + 1)); fi
done
i=0; for p in $P; do i=$((i+1))
  python3 tools/htp/loop_check.py --prompt $W/set/$p $L/text_A_p0$i.log $L/text_RQ_p0$i.log
done | tee $L/loop_check.txt
grep -H '\[PPL\] decode tokens' $L/ppl_*.log | sed 's/.*logs\///' > $L/ppl.txt
echo "--- speed (prefill / decode / last 64 TPS; text vs A run 1 of the same G)"
for g in 64 512 1024; do for r in r1 r2; do for v in A R Q RQ; do f=$L/${v}_G${g}_$r.log
  t=$(cmp -s <(strip $L/A_G${g}_r1.log) <(strip $f) && echo same || echo DIFF)
  echo "G=$g $v $r: $(grep -h -E '^(prefill|generation|generation\(last 64\)):' $f | grep -o '[0-9.]* TPS' | tr '\n' ' ')text=$t $(grep -h -o 'calls/token=[0-9.]*' $f)"
done; done; done | tee $L/speed.txt
echo "--- G2"; cat $L/shadow_check.txt
echo "--- G3"; cat $L/dump_eval.txt
echo "--- G5"; cat $L/g5.txt
echo "--- therm"; cat $L/therm.log
echo "=== done $(date '+%F %T %Z')  expectation mismatches: $MIS (0 = every expected line seen)"
