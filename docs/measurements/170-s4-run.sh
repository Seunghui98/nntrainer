#!/usr/bin/env bash
# Plan 170 round 3 step 6, sitting S4 (docs/measurements/170-attn-m1-round3.md,
# on htp/170-round3): the round-3 ATTN_M1 on silicon -- the softmax split
# (G6b) on the round-2-arithmetic skel r2w, the lut / P1-lead / ET-lead reads
# on four skels from one source and the winner W (plan section 3.3), bit
# identity on W (G3), the per-term line (G6a), then with W installed the
# CPU-vs-HTP attention shadow (G4), decode tok/s, the in-model pcyc/op
# against T64 / T1024 (G6) and text / nll (G5), with round 2 in the same
# sitting. Two run dirs so no stub meets a foreign skel:
#   new (…/causallm/s170r3n)  dev/attn-shadow-170-r3 = htp/170-round3 + the
#                             inert shadow commit; four skels
#                             libnntr_hvx_skel.{r2w,r3,r3n,r3e}.so, the
#                             active one copied to libnntr_hvx_skel.so and
#                             its md5 checked before every gtest and the E2E
#                             half
#   q2  (…/causallm/s170r3q)  S3's new/ set: round 2's app + skel
# Variants: A = new, switch off (first; also the shadow's logits reference:
# no ARM code differs from round 2's but the inert commit); Q2 = q2, Q mask;
# Q3 = new + W, Q mask; RQ3 = new + W, MOE,RMSNORM,QK_NORM,ROPE,ATTN_M1.
# Usage: bash /local/mnt/workspace/htp_moe/170/s4/run_s4.sh [serial]   (~75 min)
# Everything is logged to $L, the summary to $L/sitting.out. Stops on the
# plan's stop rules (0x8000040e, a device md5 mismatch, any bad != 0 in the
# G3 gtests before the E2E half); every other missing expected line is
# counted and printed. The G6a terms are read lines, counted apart.
set -u -o pipefail
WT=/home/j2z0-lee/nntrainer-170r3i          # env.sh only
S=${1:-R3CY10WM83Y}
W=/local/mnt/workspace/htp_moe/170/s4; L=$W/logs; mkdir -p $L $W/shadow
C=/data/local/tmp/nntrainer/causallm; DN=$C/s170r3n; DQ=$C/s170r3q; M=../models/q40-qs4cx-wh
T64=210000; T1024=350000                    # G6 gates, pcyc/op (plan 170 round 3 section 1)
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
dir_of() { case $1 in Q2) echo $DQ;; *) echo $DN;; esac; }
env_of() { case $1 in
  A) echo "";;
  Q2|Q3) echo "$F=MOE,QK_NORM,ROPE,ATTN_M1";;
  RQ3) echo "$F=MOE,RMSNORM,QK_NORM,ROPE,ATTN_M1";;
esac; }
md5_of() { awk -v f="$1" '$2 == f {print $1}' $W/md5.txt; }
swap() { # swap <skel variant>: make it the new dir's active skel, md5 checked
  local got; adb -s $S shell "cd $DN && cp libnntr_hvx_skel.$1.so libnntr_hvx_skel.so"
  got=$(adb -s $S shell "md5sum $DN/libnntr_hvx_skel.so" | tr -d '\r' | cut -d' ' -f1)
  [ "$got" = "$(md5_of new/libnntr_hvx_skel.$1.so)" ] || stop "active skel md5 $got is not $1's"
  echo "skel $1 active ($got)"; }
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
  A)
    want "$2 dspq: on" "$(grep -c 'dspq: on' $f)" 1
    want "$2 dspq close bad=0" "$(grep -c 'dspq: close calls=\([0-9]*\) served=\1 bad=0' $f)" 1
    want "$2 no graph" "$(grep -c 'graph: init' $f)" 0;;
  Q2|Q3)
    want "$2 banner" "$(grep -cF 'graph: init n_ops=228 resident=QK_NORM|ROPE|ATTN_M1|MOE moe_ops=22' $f)" 1
    want "$2 calls/token=28.00" "$(grep -cF 'calls/token=28.00' $f)" 1
    want "$2 cache=24576 KiB" "$(grep -cF 'max_seq=2048 cache=24576 KiB' $f)" 1;;
  RQ3)
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

echo "=== 170 S4 sitting  $(date '+%F %T %Z')  unit=$S  wt=$(git -C $WT rev-parse --short HEAD)"
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
adb -s $S push $W/new/. $DN/ > /dev/null && adb -s $S push $W/q2/. $DQ/ > /dev/null
adb -s $S shell "chmod 755 $DN/unittest_* $DN/nntrainer_causallm $DQ/unittest_* $DQ/nntrainer_causallm"
for d in new q2; do
  dd=$DN; [ $d = q2 ] && dd=$DQ
  adb -s $S shell "cd $dd && md5sum \$(ls -p | grep -v / | sort)" | tr -d '\r' | awk -v d=$d '{print $1"  "d"/"$2}'
done > $L/md5_device.log
diff <(grep -E '^[0-9a-f]{32}  (new|q2)/' $W/md5.txt | sort -k2) <(sort -k2 $L/md5_device.log) && echo "MD5 OK" || stop "device md5 differs from md5.txt"
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
PH='^ATTN_M1_PHASE pos=\(511\|1023\|1535\) \(warm\|cold\)'
# word <file> <pos> <warm|cold> <term>: one term of a phase line
word() { grep "^ATTN_M1_PHASE pos=$2 $3" "$1" | grep -o " $4=[0-9.]*" | cut -d= -f2; }
# ---- 2. gtest half (25 min): the split, the four skels, the winner, G3
# (a) round 2's line (q2, 9 words), this sitting
gt cost_q2 $DQ unittest_hvx_attn 'HvxAttnM1.PerLayerCost'
grep -h -E '^ATTN_M1_(FIELD (cold )?pos|PHASE)' $L/cost_q2.log | tee $L/per_l_q2.txt
want "q2 PHASE lines (3 warm + 3 cold)" "$(grep -c "$PH" $L/per_l_q2.txt)" 6
# (b) r2w: round 2's arithmetic with the 14 words -- G6b; (c) r3, r3n, r3e
for v in r2w r3 r3n r3e; do
  swap $v
  gt cost_$v $DN unittest_hvx_attn 'HvxAttnM1.PerLayerCost'
  grep -h -E '^ATTN_M1_(FIELD (cold )?pos|PHASE)' $L/cost_$v.log | tee $L/per_l_$v.txt
  want "$v PHASE lines (3 warm + 3 cold)" "$(grep -c "$PH" $L/per_l_$v.txt)" 6
  want "$v 14 words (exp= .. div=)" "$(grep "$PH" $L/per_l_$v.txt | grep -c ' exp=[0-9]* et=[0-9]* max=[0-9]* sum=[0-9]* div=[0-9]* ')" 6
done
therm t2
echo "--- G6b: the split on r2w (cold pos 1023)"
ge() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a >= b)}'; }
e=$(word $L/per_l_r2w.txt 1023 cold exp); sm=$(word $L/per_l_r2w.txt 1023 cold softmax)
pieces=$(for t in exp et max sum div; do word $L/per_l_r2w.txt 1023 cold $t; done | awk '{s += $1} END {print s}')
r=$(awk -v a="${e:-0}" -v b="${sm:-1}" 'BEGIN{printf "%.3f", a / b}')
echo "G6b r2w exp=$e softmax=$sm exp/softmax=$r (0.55 .. 0.75; ISS 0.65) pieces=$pieces"
want "G6b exp/softmax in [0.55, 0.75]" "$(awk -v x=$r 'BEGIN{print (x >= 0.55 && x <= 0.75) ? "y" : "n"}')" y
want "G6b softmax >= exp + et + max + sum + div" "$(ge "${sm:-0}" "${pieces:-1}" && echo y || echo n)" y
tot() { for t in scores softmax pv; do word $1 1023 cold $t; done | awk '{s += $1} END {print s}'; }
tw=$(tot $L/per_l_r2w.txt); tq=$(tot $L/per_l_q2.txt)
echo "G6b drift: r2w scores+softmax+pv=$tw, q2 (this sitting) $tq, S3's Q2 line 1572000 (668 k + 518 k + 386 k)"
if awk -v a="$tw" -v b="$tq" 'BEGIN{d = a / b - 1; exit !(d <= 0.15 && d >= -0.15)}'; then echo "G6b r2w within 15 % of q2"
else echo "FLAG G6b r2w more than 15 % from q2: drifted sitting (flag, continue)"; MIS=$((MIS + 1)); fi
echo "--- the three reads (cold pos 1023): lut = scores r2w vs r3; P1 lead = scores r3 vs r3n; ET lead = softmax / et r3 vs r3e"
for v in r2w r3 r3n r3e; do
  echo "$v: $(for t in append scores softmax exp et div pv busy_max pool dsp_us; do printf '%s=%s ' $t "$(word $L/per_l_$v.txt 1023 cold $t)"; done)| pos 511 pool=$(word $L/per_l_$v.txt 511 cold pool)"
done | tee $L/reads.txt
# The winner (plan section 3.3): min cold pool at pos 1023 among r3, r3n,
# r3e; within 3 % of that min the fewer leads win (r3 has 2, r3n and r3e 1);
# between two with the same lead count, the lower pos 511 cold pool.
WIN=$(for v in r3 r3n r3e; do
  n=2; [ $v = r3 ] || n=1
  echo "$v $(word $L/per_l_$v.txt 1023 cold pool) $(word $L/per_l_$v.txt 511 cold pool) $n"
done | awk '{v[NR] = $1; p[NR] = $2; q[NR] = $3; n[NR] = $4; if (NR == 1 || $2 < m) m = $2}
  END {b = 0; for (i = 1; i <= NR; ++i) if (p[i] != "" && p[i] <= 1.03 * m)
         if (!b || n[i] < n[b] || (n[i] == n[b] && q[i] < q[b])) b = i;
       print b ? v[b] : "r3"}')
echo "WINNER W=$WIN (min cold pool at pos 1023, 3 % band -> fewer leads, pos 511 tie-break)" | tee -a $L/reads.txt
swap $WIN
# (d) G3 + G6a on W, then F16Det and rope
gt g3_attn $DN unittest_hvx_attn 'HvxAttnM1.*'
for n in 1 63 64 65 512 513 1024 1536; do
  want "G3 L=$n out bad=0" "$(grep -c "^ATTN_M1_FIELD L=$n bad=0 bad_stats=[0-9]* of 2048" $L/g3_attn.log)" 1
  want "G3 L=$n bad_stats=0" "$(grep -c "^ATTN_M1_FIELD L=$n bad=[0-9]* bad_stats=0 of 2048" $L/g3_attn.log)" 1
done
want "G3 append_chain bad=0" "$(grep -c '^ATTN_M1_FIELD append_chain L=65 bad=0' $L/g3_attn.log)" 1
grep -h -E '^ATTN_M1_(FIELD (L=|(cold )?pos)|PHASE)' $L/g3_attn.log | tee $L/per_l.txt
want "PHASE lines (3 warm + 3 cold)" "$(grep -c "$PH" $L/per_l.txt)" 6
echo "--- G6a (read, not gated): cold pos 1023 on W=$WIN, against round 2 (q2) of this sitting"
g6a() { # g6a <term> <limit> [pos]
  local v q; v=$(word $L/per_l.txt ${3:-1023} cold $1); q=$(word $L/per_l_q2.txt ${3:-1023} cold $1)
  if [ -n "$v" ] && awk -v a="$v" -v b="$2" 'BEGIN{exit !(a <= b)}'; then echo "G6a ${3:-1023} $1=$v <= $2 (round 2: ${q:-n/a})"
  else echo "G6a OVER ${3:-1023} $1=${v:-?} > $2 (round 2: ${q:-n/a})"; OVER=$((OVER + 1)); fi; }
g6a append 8000; g6a exp 150000; g6a div 120000; g6a et 55000; g6a softmax 300000; g6a pv 400000
g6a busy_max 245000; g6a pool 275000; g6a dsp_us 145; g6a pool 148000 511
ms=$(for t in max sum; do word $L/per_l.txt 1023 cold $t; done | awk '{s += $1; n++} END {print n == 2 ? s : ""}')
if [ -n "$ms" ] && [ "$ms" -le 20000 ]; then echo "G6a 1023 max+sum=$ms <= 20000"; else echo "G6a OVER 1023 max+sum=${ms:-?} > 20000"; OVER=$((OVER + 1)); fi
bs=$(for v in r3 r3n r3e; do word $L/per_l_$v.txt 1023 cold scores; done | sort -n | head -1)
if [ -n "$bs" ] && [ "$bs" -le 600000 ]; then echo "G6a 1023 scores (best of r3 / r3n / r3e)=$bs <= 600000"; else echo "G6a OVER scores best=${bs:-?} > 600000"; OVER=$((OVER + 1)); fi
gt g3_f16det $DN unittest_nntrainer_cpu_backend_fp16 'AttnM1F16Det.*'
for n in 513 1024 1536; do
  want "G3 AttnM1F16Det L=$n (2 rope positions) bad=0" "$(grep -c "^ATTN_M1_F16 L=$n rope_pos=[0-9]* out bad=0 " $L/g3_f16det.log)" 2; done
want "G3 AttnM1F16Det no FAILED" "$(grep -c 'FAILED' $L/g3_f16det.log)" 0
gt g3_rope $DN unittest_hvx_softmax 'HvxM1Ops.Rope64*'
want "G3 rope64 bad=0 (5 positions)" "$(grep -c '^M1_OPS_FIELD rope64 pos=[0-9]* bad=0 of 2560' $L/g3_rope.log)" 5
therm t3
# Stop before the E2E half on any bad != 0 in (d) (plan step 6).
nb=$(cat $L/g3_attn.log $L/g3_f16det.log $L/g3_rope.log | grep -E '^(ATTN_M1_FIELD|ATTN_M1_F16|M1_OPS_FIELD)' | grep -E ' (out )?bad(_stats)?=[1-9]' | wc -l)
[ "$nb" = 0 ] || stop "G3: $nb lines with bad != 0 on W=$WIN (see g3_*.log)"

# ---- 3. G4: the attention shadow, prompt 512, G = 8, forced on A's tokens (4 min)
adb -s $S shell "rm -f $DN/cont.ids"
for v in A Q3 RQ3; do
  adb -s $S shell "rm -rf $DN/d_f_$v && mkdir -p $DN/d_f_$v"
  run $v 8 f_$v prompt512.txt NNTR_ATTN_SHADOW=d_f_$v/attn.bin NNTR_LOGIT_SHADOW=d_f_$v/logits.bin NNTR_PPL_DECODE=cont.ids
  vbanner $v f_$v
  rm -rf $W/shadow/d_f_$v && adb -s $S pull $DN/d_f_$v $W/shadow/ > /dev/null && adb -s $S shell "rm -rf $DN/d_f_$v"
done
want "f_A source=self" "$(grep -c 'decode tokens=8 .* source=self' $L/f_A.log)" 1
python3 $W/attn_shadow_check.py $W/shadow/d_f_A $W/shadow/d_f_A $W/shadow/d_f_Q3 $W/shadow/d_f_RQ3 | tee $L/shadow_check.txt
want "G4 A (no hook, no record; logits == itself)" "$(grep -c 'd_f_A tag3_heads=0/0 records=0 .*logits_equal_steps=8/8' $L/shadow_check.txt)" 1
want "G4 Q3" "$(grep -c 'd_f_Q3 tag3_heads=1536/1536 records=48 layers=6 positions=8 zero_records=0 logits_equal_steps=8/8' $L/shadow_check.txt)" 1
want "G4 RQ3" "$(grep -c 'd_f_RQ3 tag3_heads=1536/1536 records=48 layers=6 positions=8 zero_records=0 logits_equal_steps=8/8' $L/shadow_check.txt)" 1
for v in Q3 RQ3; do cmp -s <(nll $L/f_A.log) <(nll $L/f_$v.log) && echo "G4 nll f_$v == f_A" || { echo "BAD G4 nll f_$v != f_A"; MIS=$((MIS + 1)); }; done
therm t4

# ---- 4. speed, prompt 512, mirrored A Q2 Q3 | Q3 Q2 A (13 min)
for g in 64 512 1024; do
  for v in A Q2 Q3; do run $v $g ${v}_G${g}_r1 prompt512.txt; vbanner $v ${v}_G${g}_r1; done
  for v in Q3 Q2 A; do run $v $g ${v}_G${g}_r2 prompt512.txt; vbanner $v ${v}_G${g}_r2; done
  therm t5_G$g
done

# ---- 5. G6: the in-model pcyc/op, Q2-prof / Q3-prof at G = 64 and 1024 (5 min)
for g in 64 1024; do
  run Q2 $g Q2p_G$g prompt512.txt NNTR_HTP_PROFILE=2; vbanner Q2 Q2p_G$g
  run Q3 $g Q3p_G$g prompt512.txt NNTR_HTP_PROFILE=2; vbanner Q3 Q3p_G$g
done
grep -H 'graph: calls=' $L/Q2p_G64.log $L/Q3p_G64.log $L/Q2p_G1024.log $L/Q3p_G1024.log | sed 's/.*logs\///' | tee $L/g6.txt
a64=$(grep '^Q3p_G64' $L/g6.txt | grep -o ' ATTN_M1=[0-9]*' | cut -d= -f2)
a1024=$(grep '^Q3p_G1024' $L/g6.txt | grep -o ' ATTN_M1=[0-9]*' | cut -d= -f2)
echo "G6 Q3 ATTN_M1 pcyc/op G=64 ${a64:-?} (<= $T64)  G=1024 ${a1024:-?} (<= $T1024)"
want "G6 G=64 <= $T64" "$([ -n "$a64" ] && [ "$a64" -le $T64 ] && echo y || echo n)" y
want "G6 G=1024 <= $T1024" "$([ -n "$a1024" ] && [ "$a1024" -le $T1024 ] && echo y || echo n)" y
therm t6

# ---- 6. G5: 8 prompts at G = 256, nll forced on A's continuation, text (22 min)
i=0; for p in $P; do i=$((i+1))
  adb -s $S shell "rm -f $DN/cont_p0$i.ids"
  run A 256 ppl_A_p0$i $p NNTR_PPL_DECODE=cont_p0$i.ids                 # A's own continuation (source=self)
  [ $i = 1 ] && run A 256 ppl_Af_p01 $p NNTR_PPL_DECODE=cont_p01.ids     # null check: A forced == A self
  for v in Q3 RQ3; do run $v 256 ppl_${v}_p0$i $p NNTR_PPL_DECODE=cont_p0$i.ids; vbanner $v ppl_${v}_p0$i; done
  for v in A Q2 Q3 RQ3; do run $v 256 text_${v}_p0$i $p; [ $v = A ] || vbanner $v text_${v}_p0$i; done
done
therm t7

# ---- 7. checks and summary (2 min)
echo "--- G5 nll (every [PPL] decode step line equal to A's, 17 digits)"
want "ppl_A_p01 source=self" "$(grep -c 'decode tokens=256 .* source=self' $L/ppl_A_p01.log)" 1
cmp -s <(nll $L/ppl_A_p01.log) <(nll $L/ppl_Af_p01.log) && echo "null check A forced == A self" || { echo "BAD A forced != A self"; MIS=$((MIS + 1)); }
for i in 1 2 3 4 5 6 7 8; do for v in Q3 RQ3; do
  if cmp -s <(nll $L/ppl_A_p0$i.log) <(nll $L/ppl_${v}_p0$i.log); then echo "p0$i $v nll == A"; else echo "BAD p0$i $v nll != A"; MIS=$((MIS + 1)); fi
done; done
echo "--- G5 text (G = 256): Q3 and RQ3 vs A, Q3 vs Q2"
for i in 1 2 3 4 5 6 7 8; do
  for v in Q3 RQ3; do
    if cmp -s <(strip $L/text_A_p0$i.log) <(strip $L/text_${v}_p0$i.log); then echo "p0$i $v text == A"; else echo "BAD p0$i $v text != A"; MIS=$((MIS + 1)); fi
  done
  if cmp -s <(strip $L/text_Q2_p0$i.log) <(strip $L/text_Q3_p0$i.log); then echo "p0$i Q3 text == Q2"; else echo "BAD p0$i Q3 text != Q2"; MIS=$((MIS + 1)); fi
done
grep -H '\[PPL\] decode tokens' $L/ppl_*.log | sed 's/.*logs\///' > $L/ppl.txt
echo "--- speed (prefill / decode / last 64 TPS; text vs A run 1 of the same G)"
for g in 64 512 1024; do for r in r1 r2; do for v in A Q2 Q3; do f=$L/${v}_G${g}_$r.log
  t=$(cmp -s <(strip $L/A_G${g}_r1.log) <(strip $f) && echo same || echo DIFF)
  echo "G=$g $v $r: $(grep -h -E '^(prefill|generation|generation\(last 64\)):' $f | grep -o '[0-9.]* TPS' | tr '\n' ' ')text=$t $(grep -h -o 'calls/token=[0-9.]*' $f)"
done; done; done | tee $L/speed.txt
want "speed: every cell's text == A run 1 of its G" "$(grep -c 'text=DIFF' $L/speed.txt)" 0
echo "--- standing: prefill mean of each Q cell against A's (mirrored runs), -5 % band"
for g in 64 512 1024; do
  pa=$(grep "^G=$g A " $L/speed.txt | awk '{s += $4} END {print s / NR}')
  for v in Q2 Q3; do
    pv=$(grep "^G=$g $v " $L/speed.txt | awk '{s += $4} END {print s / NR}')
    echo "G=$g $v prefill $pv vs A $pa: $(awk -v a=$pv -v b=$pa 'BEGIN{d = 100 * (a / b - 1); printf "%+.1f %% %s", d, d >= -5 ? "ok" : "BELOW -5 %"}')"
  done
done | tee $L/prefill.txt
echo "--- reads (the split and the three A/B reads)"; cat $L/reads.txt
echo "--- G3 per L (W=$WIN)"; cat $L/per_l.txt
echo "--- round 2 per L (q2, same sitting)"; cat $L/per_l_q2.txt
echo "--- G4"; cat $L/shadow_check.txt
echo "--- G6"; cat $L/g6.txt
echo "--- therm"; cat $L/therm.log
echo "=== done $(date '+%F %T %Z')  W=$WIN  expectation mismatches: $MIS (0 = every expected line seen)  G6a terms over their line: $OVER (read, not gated)"
