#!/usr/bin/env bash
# Plan 170 round 2 step 6, sitting S3 (docs/measurements/170-attn-m1-round2.md,
# on htp/170-round2): the round-2 ATTN_M1 on silicon -- the lead probe cells,
# bit identity (G3), the per-term phase line (G6a), the CPU-vs-HTP attention
# shadow (G4), text and nll (G5), the in-model pcyc/op against T64 / T1024
# (G6) and decode tok/s, with round 1 (the merged kernel) in the same sitting.
# Two run dirs so no stub meets a foreign skel:
#   new (…/causallm/s170r2n)  dev/attn-shadow-170-r2 = htp/170-round2 + the
#                             inert shadow commit
#   q1  (…/causallm/s170r2q)  htp_moe @ 90d88e2b (PR #176 merged: round 1)
# Variants: A = q1, switch off (first); Q1 = q1, Q mask (round 1, the
# same-sitting reference); Q2 = new, Q mask; RQ2 = new,
# MOE,RMSNORM,QK_NORM,ROPE,ATTN_M1. G4's logits reference is the new dir
# with the switch off (An: the shadow needs the dev commit; no ARM code
# differs from A's but that inert commit).
# Usage: bash /local/mnt/workspace/htp_moe/170/s3/run_s3.sh [serial]   (~60 min)
# Everything is logged to $L, the summary to $L/sitting.out. Stops on the
# plan's stop rules (0x8000040e, device md5); every other missing expected
# line is counted and printed. The G6a terms are read lines, counted apart.
set -u -o pipefail
WT=/home/j2z0-lee/nntrainer-170q           # env.sh only
S=${1:-R3CY10WM83Y}
W=/local/mnt/workspace/htp_moe/170/s3; L=$W/logs; mkdir -p $L $W/shadow
C=/data/local/tmp/nntrainer/causallm; DN=$C/s170r2n; DQ=$C/s170r2q; M=../models/q40-qs4cx-wh
T64=210000; T1024=350000                   # G6 gates, pcyc/op (plan 170 round 2 section 1)
cd $WT && { set +u; source tools/htp/env.sh > /dev/null 2>&1; set -u; }
exec > >(tee -a $L/sitting.out) 2>&1
MIS=0; OVER=0
stop() { echo "STOP: $*"; therm stop; exit 1; }
want() { # want <label> <got> <expected>
  if [ "$2" = "$3" ]; then echo "OK  $1 = $2"; else echo "BAD $1: got '$2' want '$3'"; MIS=$((MIS + 1)); fi; }
therm() { echo "$1 $(date +%H:%M:%S) $(adb -s $S shell 'dumpsys battery | grep -E "^  (level|temperature)"; cat /sys/class/thermal/thermal_zone0/temp' | tr -d '\r' | tr '\n' ' ')" | tee -a $L/therm.log; }
zone0() { adb -s $S shell cat /sys/class/thermal/thermal_zone0/temp | tr -d '\r'; }
stale() { grep -qi '0x8000040e' "$1" && stop "$(basename $1): 0x8000040e (stale skel, rule 3)"; true; }
F="NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS"
dir_of() { case $1 in A|Q1) echo $DQ;; *) echo $DN;; esac; }
env_of() { case $1 in
  A|An) echo "";;
  Q1|Q2) echo "$F=MOE,QK_NORM,ROPE,ATTN_M1";;
  RQ2) echo "$F=MOE,RMSNORM,QK_NORM,ROPE,ATTN_M1";;
esac; }
run() { # run <variant> <G> <log name> <prompt file> [extra env ...]
  local v=$1 g=$2 log=$3 p=$4 d; d=$(dir_of $1); shift 4
  adb -s $S shell "cd $d && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $g/' $M/nntr_config.json && \
    grep num_to_generate $M/nntr_config.json && md5sum libnntr_hvx_skel.so && \
    $(env_of $v) $* NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm $M \"\$(cat $p)\"" > $L/$log.log 2>&1
  stale $L/$log.log
  echo "$log: $(grep -h -E '^(prefill|generation):' $L/$log.log | grep -o '[0-9.]* TPS' | tr '\n' ' ')$(grep -h -o 'calls/token=[0-9.]*' $L/$log.log) $(grep -h -o '\[PPL\] decode tokens=.*' $L/$log.log | cut -c1-90)"
}
vbanner() { # vbanner <variant> <log>: the switch-on banner / A's dspq pair, else void (rule 36)
  local f=$L/$2.log
  case $1 in
  A|An)
    want "$2 dspq: on" "$(grep -c 'dspq: on' $f)" 1
    want "$2 dspq close bad=0" "$(grep -c 'dspq: close calls=\([0-9]*\) served=\1 bad=0' $f)" 1
    want "$2 no graph" "$(grep -c 'graph: init' $f)" 0;;
  Q1|Q2)
    want "$2 banner" "$(grep -cF 'graph: init n_ops=228 resident=QK_NORM|ROPE|ATTN_M1|MOE moe_ops=22' $f)" 1
    want "$2 calls/token=28.00" "$(grep -cF 'calls/token=28.00' $f)" 1
    want "$2 cache=24576 KiB" "$(grep -cF 'max_seq=2048 cache=24576 KiB' $f)" 1;;
  RQ2)
    want "$2 banner" "$(grep -cF 'graph: init n_ops=228 resident=RMSNORM|QK_NORM|ROPE|ATTN_M1|MOE moe_ops=22' $f)" 1
    want "$2 calls/token=77.00" "$(grep -cF 'calls/token=77.00' $f)" 1
    want "$2 cache=24576 KiB" "$(grep -cF 'max_seq=2048 cache=24576 KiB' $f)" 1;;
  esac; }
# The generated text of a log: everything before the summary, minus every
# [HTP…] line (the level-2 profile prints [HTP-DMA] ones) and [PPL] lines.
strip() { sed -n '/^=====/q;p' "$1" | perl -0pe 's/\[HTP[^\]\n]*\] [^\n]*\n//g; s/\[PPL\] [^\n]*\n//g' |
  grep -v 'moe m1 gemv\|libnntr_hvx_skel\|nntrainer_causallm\|num_to_generate'; }
nll() { grep -o '\[PPL\] decode step=.*' "$1"; }
P="prompt512.txt bitset-02-code.txt bitset-03-math.txt bitset-04-korean.txt bitset-05-json.txt bitset-06-dialogue.txt bitset-07-facts.txt bitset-08-short.txt"

echo "=== 170 S3 sitting  $(date '+%F %T %Z')  unit=$S  wt=$(git -C $WT rev-parse --short HEAD)"
# ---- 0. device state, cool start (1 min)
adb devices | tee $L/devices.log
adb devices | grep -q "^$S[[:space:]]*device" || stop "unit $S not attached"
adb -s $S shell input keyevent 223 || true
therm t0
for i in $(seq 1 40); do z=$(zone0); [ "$z" -le 35000 ] && break
  echo "zone0=$z > 35000, waiting 30 s ($i/40)"; sleep 30; done
echo "start zone0=$(zone0)"

# ---- 1. install both run dirs and the model config (3 min)
[ "$(cd $W && LC_ALL=C md5sum -c md5.txt | grep -vc ': OK$')" = 0 ] || stop "workstation sets differ from md5.txt"
adb -s $S shell ls -l $C/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin
adb -s $S shell "rm -rf $DN $DQ && mkdir -p $DN $DQ"
adb -s $S push $W/new/. $DN/ > /dev/null && adb -s $S push $W/q1/. $DQ/ > /dev/null
adb -s $S shell "chmod 755 $DN/unittest_* $DN/nntrainer_causallm $DQ/unittest_* $DQ/nntrainer_causallm"
for d in new q1; do
  dd=$DN; [ $d = q1 ] && dd=$DQ
  adb -s $S shell "cd $dd && md5sum \$(ls -p | grep -v / | sort)" | tr -d '\r' | awk -v d=$d '{print $1"  "d"/"$2}'
done > $L/md5_device.log
diff <(grep -E '^[0-9a-f]{32}  (new|q1)/' $W/md5.txt | sort -k2) <(sort -k2 $L/md5_device.log) && echo "MD5 OK" || stop "device md5 differs from md5.txt"
adb -s $S shell "cd $DN/$M && \
  sed -i 's/\"do_sample\": true/\"do_sample\": false/' generation_config.json && \
  sed -i 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' nntr_config.json && \
  (grep -q moe_engine nntr_config.json || sed -i 's/\"bad_word_ids\": \[124900\],/\"bad_word_ids\": [124900],\n    \"moe_engine\": \"htp\",/' nntr_config.json) && \
  grep -H do_sample generation_config.json && grep -H -E 'bad_word_ids|num_to_generate|init_seq_len|_engine|_htp_layers|max_seq_len' nntr_config.json" \
  | tee $L/config.log   # one model dir, shared by both run dirs through ../models
for k in '"do_sample": false' '"bad_word_ids": \[124900\]' '"init_seq_len": 512' '"moe_engine": "htp"' '"moe_htp_layers": ""'; do
  want "config $k" "$(grep -c "$k" $L/config.log)" 1; done
therm t1

gt() { # gt <log> <dir> <binary> <filter>: one gtest run
  adb -s $S shell "cd $2 && md5sum libnntr_hvx_skel.so $3 && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./$3 --gtest_filter='$4'" > $L/$1.log 2>&1
  stale $L/$1.log
  grep -h -E '^\[  (PASSED|FAILED) ' $L/$1.log | sed "s/^/$1: /"
}
# ---- 2. microbench: the lead probe cells (new set, 4 min)
gt probe $DN unittest_hvx_attn 'HvxAttnM1Probe.Cost'
want "probe: no FAILED" "$(grep -c '^\[  FAILED' $L/probe.log)" 0
for o in pv4 pv4_cold pv4_cold_l2f scores1 scores1_cold scores1_cold_l2f scores1_splat; do
  want "probe $o rows (lanes 1/2/4/6)" "$(grep -c "^ATTN_M1_PROBE_COST op=$o lanes=" $L/probe.log)" 4; done
echo "--- probe: wall pcyc per 64-lane FMA; each lead's cold cell against its warm op (warm / cold / cold+lead)" | tee $L/probe_read.txt
for n in 1 2 4 6; do
  pf() { grep "^ATTN_M1_PROBE_COST op=$1 lanes=$n " $L/probe.log | grep -o 'pcyc_per_fma64=[0-9.e+-]*' | cut -d= -f2; }
  echo "lanes=$n pv4: warm=$(pf pv4) cold=$(pf pv4_cold) cold_l2f=$(pf pv4_cold_l2f) | scores1: warm=$(pf scores1) cold=$(pf scores1_cold) cold_l2f=$(pf scores1_cold_l2f) splat=$(pf scores1_splat)"
done | tee -a $L/probe_read.txt
therm t2

# ---- 3. G3 + G6a: bit identity and the per-term phase line (new), round 1's line (q1) (10 min)
gt g3_attn $DN unittest_hvx_attn 'HvxAttnM1.*'
for n in 1 63 64 65 512 513 1024 1536; do
  want "G3 L=$n out bad=0" "$(grep -c "^ATTN_M1_FIELD L=$n bad=0 bad_stats=[0-9]* of 2048" $L/g3_attn.log)" 1
  want "G3 L=$n bad_stats=0" "$(grep -c "^ATTN_M1_FIELD L=$n bad=[0-9]* bad_stats=0 of 2048" $L/g3_attn.log)" 1
done
want "G3 append_chain bad=0" "$(grep -c '^ATTN_M1_FIELD append_chain L=65 bad=0' $L/g3_attn.log)" 1
grep -h -E '^ATTN_M1_(FIELD (L=|(cold )?pos)|PHASE)' $L/g3_attn.log | tee $L/per_l.txt
want "PHASE lines (3 warm + 3 cold)" "$(grep -c '^ATTN_M1_PHASE pos=\(511\|1023\|1535\) \(warm\|cold\)' $L/per_l.txt)" 6
gt q1_cost $DQ unittest_hvx_attn 'HvxAttnM1.PerLayerCost'
grep -h -E '^ATTN_M1_(FIELD (cold )?pos|PHASE)' $L/q1_cost.log | tee $L/per_l_q1.txt
want "Q1 PHASE lines (3 warm + 3 cold)" "$(grep -c '^ATTN_M1_PHASE pos=\(511\|1023\|1535\) \(warm\|cold\)' $L/per_l_q1.txt)" 6
echo "--- G6a (read, not gated): cold pos 1023, round 2 vs its line (round 1 in the same sitting)"
g6a() { # g6a <term> <limit>
  local v q; v=$(grep '^ATTN_M1_PHASE pos=1023 cold' $L/per_l.txt | grep -o " $1=[0-9.]*" | cut -d= -f2)
  q=$(grep '^ATTN_M1_PHASE pos=1023 cold' $L/per_l_q1.txt | grep -o " $1=[0-9.]*" | cut -d= -f2)
  if [ -n "$v" ] && awk -v a="$v" -v b="$2" 'BEGIN{exit !(a <= b)}'; then echo "G6a $1=$v <= $2 (round 1: ${q:-?})"
  else echo "G6a OVER $1=${v:-?} > $2 (round 1: ${q:-?})"; OVER=$((OVER + 1)); fi; }
g6a append 10000; g6a scores 550000; g6a softmax 160000; g6a pv 400000; g6a busy_max 200000; g6a pool 220000; g6a dsp_us 110
gt g3_f16det $DN unittest_nntrainer_cpu_backend_fp16 'AttnM1F16Det.*'
for n in 513 1024 1536; do
  want "G3 AttnM1F16Det L=$n (2 rope positions) bad=0" "$(grep -c "^ATTN_M1_F16 L=$n rope_pos=[0-9]* out bad=0 " $L/g3_f16det.log)" 2; done
want "G3 AttnM1F16Det no FAILED" "$(grep -c 'FAILED' $L/g3_f16det.log)" 0
gt g3_rope $DN unittest_hvx_softmax 'HvxM1Ops.Rope64*'
want "G3 rope64 bad=0 (5 positions)" "$(grep -c '^M1_OPS_FIELD rope64 pos=[0-9]* bad=0 of 2560' $L/g3_rope.log)" 5
therm t3

# ---- 4. G4: the attention shadow, prompt 512, G = 8, forced on An's tokens (4 min)
adb -s $S shell "rm -f $DN/cont.ids"
for v in An Q2 RQ2; do
  adb -s $S shell "rm -rf $DN/d_f_$v && mkdir -p $DN/d_f_$v"
  run $v 8 f_$v prompt512.txt NNTR_ATTN_SHADOW=d_f_$v/attn.bin NNTR_LOGIT_SHADOW=d_f_$v/logits.bin NNTR_PPL_DECODE=cont.ids
  vbanner $v f_$v
  rm -rf $W/shadow/d_f_$v && adb -s $S pull $DN/d_f_$v $W/shadow/ > /dev/null && adb -s $S shell "rm -rf $DN/d_f_$v"
done
want "f_An source=self" "$(grep -c 'decode tokens=8 .* source=self' $L/f_An.log)" 1
python3 $W/attn_shadow_check.py $W/shadow/d_f_An $W/shadow/d_f_An $W/shadow/d_f_Q2 $W/shadow/d_f_RQ2 | tee $L/shadow_check.txt
want "G4 An (no hook, no record; logits == itself)" "$(grep -c 'd_f_An tag3_heads=0/0 records=0 .*logits_equal_steps=8/8' $L/shadow_check.txt)" 1
want "G4 Q2" "$(grep -c 'd_f_Q2 tag3_heads=1536/1536 records=48 layers=6 positions=8 zero_records=0 logits_equal_steps=8/8' $L/shadow_check.txt)" 1
want "G4 RQ2" "$(grep -c 'd_f_RQ2 tag3_heads=1536/1536 records=48 layers=6 positions=8 zero_records=0 logits_equal_steps=8/8' $L/shadow_check.txt)" 1
for v in Q2 RQ2; do cmp -s <(nll $L/f_An.log) <(nll $L/f_$v.log) && echo "G4 nll f_$v == f_An" || { echo "BAD G4 nll f_$v != f_An"; MIS=$((MIS + 1)); }; done
therm t4

# ---- 5. speed, prompt 512, mirrored A Q1 Q2 | Q2 Q1 A (13 min)
for g in 64 512 1024; do
  for v in A Q1 Q2; do run $v $g ${v}_G${g}_r1 prompt512.txt; vbanner $v ${v}_G${g}_r1; done
  for v in Q2 Q1 A; do run $v $g ${v}_G${g}_r2 prompt512.txt; vbanner $v ${v}_G${g}_r2; done
  therm t5_G$g
done

# ---- 6. G6: the in-model pcyc/op, Q1-prof / Q2-prof at G = 64 and 1024 (5 min)
for g in 64 1024; do
  run Q1 $g Q1p_G$g prompt512.txt NNTR_HTP_PROFILE=2; vbanner Q1 Q1p_G$g
  run Q2 $g Q2p_G$g prompt512.txt NNTR_HTP_PROFILE=2; vbanner Q2 Q2p_G$g
done
grep -H 'graph: calls=' $L/Q1p_G64.log $L/Q2p_G64.log $L/Q1p_G1024.log $L/Q2p_G1024.log | sed 's/.*logs\///' | tee $L/g6.txt
a64=$(grep '^Q2p_G64' $L/g6.txt | grep -o ' ATTN_M1=[0-9]*' | cut -d= -f2)
a1024=$(grep '^Q2p_G1024' $L/g6.txt | grep -o ' ATTN_M1=[0-9]*' | cut -d= -f2)
echo "G6 Q2 ATTN_M1 pcyc/op G=64 ${a64:-?} (<= $T64)  G=1024 ${a1024:-?} (<= $T1024)"
want "G6 G=64 <= $T64" "$([ -n "$a64" ] && [ "$a64" -le $T64 ] && echo y || echo n)" y
want "G6 G=1024 <= $T1024" "$([ -n "$a1024" ] && [ "$a1024" -le $T1024 ] && echo y || echo n)" y
therm t6

# ---- 7. G5: 8 prompts at G = 256, nll forced on A's continuation, text (22 min)
i=0; for p in $P; do i=$((i+1))
  adb -s $S shell "rm -f $DQ/cont_p0$i.ids $DN/cont_p0$i.ids"
  run A 256 ppl_A_p0$i $p NNTR_PPL_DECODE=cont_p0$i.ids                 # A's own continuation (source=self)
  [ $i = 1 ] && run A 256 ppl_Af_p01 $p NNTR_PPL_DECODE=cont_p01.ids     # null check: A forced == A self
  adb -s $S shell "cp $DQ/cont_p0$i.ids $DN/cont_p0$i.ids"
  for v in Q2 RQ2; do run $v 256 ppl_${v}_p0$i $p NNTR_PPL_DECODE=cont_p0$i.ids; vbanner $v ppl_${v}_p0$i; done
  for v in A Q1 Q2 RQ2; do run $v 256 text_${v}_p0$i $p; [ $v = A ] || vbanner $v text_${v}_p0$i; done
done
therm t7

# ---- 8. checks and summary (2 min)
echo "--- G5 nll (every [PPL] decode step line equal to A's, 17 digits)"
want "ppl_A_p01 source=self" "$(grep -c 'decode tokens=256 .* source=self' $L/ppl_A_p01.log)" 1
cmp -s <(nll $L/ppl_A_p01.log) <(nll $L/ppl_Af_p01.log) && echo "null check A forced == A self" || { echo "BAD A forced != A self"; MIS=$((MIS + 1)); }
for i in 1 2 3 4 5 6 7 8; do for v in Q2 RQ2; do
  if cmp -s <(nll $L/ppl_A_p0$i.log) <(nll $L/ppl_${v}_p0$i.log); then echo "p0$i $v nll == A"; else echo "BAD p0$i $v nll != A"; MIS=$((MIS + 1)); fi
done; done
echo "--- G5 text (G = 256): Q2 and RQ2 vs A, Q2 vs Q1"
for i in 1 2 3 4 5 6 7 8; do
  for v in Q2 RQ2; do
    if cmp -s <(strip $L/text_A_p0$i.log) <(strip $L/text_${v}_p0$i.log); then echo "p0$i $v text == A"; else echo "BAD p0$i $v text != A"; MIS=$((MIS + 1)); fi
  done
  if cmp -s <(strip $L/text_Q1_p0$i.log) <(strip $L/text_Q2_p0$i.log); then echo "p0$i Q2 text == Q1"; else echo "BAD p0$i Q2 text != Q1"; MIS=$((MIS + 1)); fi
done
grep -H '\[PPL\] decode tokens' $L/ppl_*.log | sed 's/.*logs\///' > $L/ppl.txt
echo "--- speed (prefill / decode / last 64 TPS; text vs A run 1 of the same G)"
for g in 64 512 1024; do for r in r1 r2; do for v in A Q1 Q2; do f=$L/${v}_G${g}_$r.log
  t=$(cmp -s <(strip $L/A_G${g}_r1.log) <(strip $f) && echo same || echo DIFF)
  echo "G=$g $v $r: $(grep -h -E '^(prefill|generation|generation\(last 64\)):' $f | grep -o '[0-9.]* TPS' | tr '\n' ' ')text=$t $(grep -h -o 'calls/token=[0-9.]*' $f)"
done; done; done | tee $L/speed.txt
want "speed: every cell's text == A run 1 of its G" "$(grep -c 'text=DIFF' $L/speed.txt)" 0
echo "--- probe"; cat $L/probe_read.txt
echo "--- G3 per L (round 2)"; cat $L/per_l.txt
echo "--- round 1 per L (q1, same sitting)"; cat $L/per_l_q1.txt
echo "--- G4"; cat $L/shadow_check.txt
echo "--- G6"; cat $L/g6.txt
echo "--- therm"; cat $L/therm.log
echo "=== done $(date '+%F %T %Z')  expectation mismatches: $MIS (0 = every expected line seen)  G6a terms over their line: $OVER (read, not gated)"
