#!/usr/bin/env bash
# Plan 132 PR 2, sitting S4 (docs/measurements/132-pr2-exact-fc.md, section
# "Sitting S4", on htp/132-exact-fc): the router after its L1 prefetch
# (37a36e44 after the rebase on htp_moe @ 4eab54ef). Set
# /local/mnt/workspace/htp_moe/132/set3/, all of it built at dev/fc-shadow
# @ a8a00d54 (the PR code + the inert shadow commit).
# What it re-reads:
#   G1  the router cells (HvxM1Ops.RouterTopkMatchesDetBitExact, 3 shapes)
#   the resident ROUTER_TOPK's pcycles/op, htp_moe skel vs this set (<= 60000)
#   8 prompts at G = 256: S nll and text == A (the router is in S's shadow)
# Usage: bash /local/mnt/workspace/htp_moe/132/run_132s4.sh [serial]   (~15 min)
# Logs to $L, summary to $L/sitting.out. Stops on a missing unit, an md5
# mismatch or a stale skel; every other missing expected line is counted.
set -u -o pipefail
S=${1:-R3CY10WM83Y}
W=/local/mnt/workspace/htp_moe/132; L=$W/logs_s4; mkdir -p $L
C=/data/local/tmp/nntrainer/causallm; D=$C/s132c; M=../models/q40-qs4cx-wh
{ set +u; source /home/j2z0-lee/nntrainer-132x/tools/htp/env.sh > /dev/null 2>&1; set -u; }
exec > >(tee -a $L/sitting.out) 2>&1
MIS=0
stop() { echo "STOP: $*"; exit 1; }
want() { # want <label> <got> <expected>
  if [ "$2" = "$3" ]; then echo "OK  $1 = $2"; else echo "BAD $1: got '$2' want '$3'"; MIS=$((MIS + 1)); fi; }
therm() { echo "$1 $(date +%H:%M:%S) $(adb -s $S shell 'dumpsys battery | grep -E "^  (level|temperature)"; cat /sys/class/thermal/thermal_zone0/temp' | tr -d '\r' | tr '\n' ' ')" | tee -a $L/therm.log; }
zone0() { adb -s $S shell cat /sys/class/thermal/thermal_zone0/temp | tr -d '\r'; }
cool() { local t z; for t in $(seq 1 40); do z=$(zone0); [ "$z" -le 35000 ] && break
  echo "zone0=$z > 35000, waiting 30 s ($t/40)"; sleep 30; done; echo "zone0=$(zone0)"; }
F="NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=MOE,RMSNORM,QK_NORM,ROPE,CONV1D_GATE,ATTN_M1,ADD,ROUTER_TOPK NNTR_HTP_PROFILE=2"
env_of() { case $1 in
  A*) echo "";;
  S*) echo "NNTR_FC_SHADOW=$D/shadow_scratch.bin";;
  Pb) echo "$F ADSP_LIBRARY_PATH=$D/base";;   # profile only: the htp_moe skel (the old HVX router)
  P) echo "$F";;                              # profile only: this set's skel
esac; }
run() { # run <variant> <G> <log name> <prompt file> [extra env ...]
  local v=$1 g=$2 log=$3 p=$4; shift 4
  adb -s $S shell "cd $D && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $g/' $M/nntr_config.json && \
    grep num_to_generate $M/nntr_config.json && md5sum libnntr_hvx_skel.so base/libnntr_hvx_skel.so && \
    LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. $(env_of $v) $* NNTR_NUM_THREADS=8 \
    ./nntrainer_causallm $M \"\$(cat $p)\"" \
    > $L/$log.log 2>&1
  grep -qi '0x8000040e' $L/$log.log && stop "$log: 0x8000040e (stale skel, rule 3)"
  grep -q 'dev/fc-shadow' $L/$log.log && { echo "BAD $log: dev/fc-shadow error: $(grep -m1 'dev/fc-shadow' $L/$log.log)"; MIS=$((MIS + 1)); }
  echo "$log: $(grep -h -E '^(prefill|generation):' $L/$log.log | grep -o '[0-9.]* TPS' | tr '\n' ' ')$(grep -h -o '\[PPL\] decode tokens=.*' $L/$log.log | cut -c1-90)"
}
dspq() { local f=$L/$1.log
  want "$1 dspq: on" "$(grep -c 'dspq: on' $f)" 1
  want "$1 dspq close bad=0" "$(grep -c 'dspq: close calls=\([0-9]*\) served=\1 bad=0' $f)" 1
  want "$1 no graph" "$(grep -c 'graph: init' $f)" 0; }
strip() { sed -n '/^=====/q;p' "$1" | perl -0pe 's/\[HTP\] [^\n]*\n//g; s/\[PPL\] [^\n]*\n//g; s/\[FC-SHADOW\] [^\n]*\n//g' |
  grep -v 'moe m1 gemv\|libnntr_hvx_skel\|nntrainer_causallm\|num_to_generate'; }
nll() { grep -o '\[PPL\] decode step=.*' "$1"; }
P="prompt512.txt bitset-02-code.txt bitset-03-math.txt bitset-04-korean.txt bitset-05-json.txt bitset-06-dialogue.txt bitset-07-facts.txt bitset-08-short.txt"

echo "=== 132 PR 2 sitting S4  $(date '+%F %T %Z')  unit=$S"
# ---- 0. device state, cool start
adb devices | tee $L/devices.log
adb devices | grep -q "^$S[[:space:]]*device" || stop "unit $S not attached"
adb -s $S shell input keyevent 223 || true
therm t0; cool

# ---- 1. install and config (2 min)
[ "$(cd $W/set3 && LC_ALL=C md5sum -c md5.txt | grep -vc ': OK$')" = 0 ] || stop "workstation set3 differs from set3/md5.txt"
adb -s $S shell "rm -rf $D && mkdir -p $D" && adb -s $S push $W/set3/. $D/ > /dev/null
adb -s $S shell "rm -f $D/md5.txt; chmod 755 $D/nntrainer_causallm $D/unittest_*"
adb -s $S shell "mkdir -p $D/base && mv $D/libnntr_hvx_skel_base.so $D/base/libnntr_hvx_skel.so"
adb -s $S shell "cd $D && md5sum \$(ls -p | grep -v / | sort) && cd base && md5sum libnntr_hvx_skel.so | sed 's/libnntr_hvx_skel.so/libnntr_hvx_skel_base.so/'" > $L/md5_device.log
diff <(sort -k2 $W/set3/md5.txt | tr -d '\r') <(sort -k2 $L/md5_device.log | tr -d '\r') && echo "MD5 OK" || stop "device md5 differs from md5.txt"
adb -s $S shell "cd $D/$M && \
  sed -i 's/\"do_sample\": true/\"do_sample\": false/' generation_config.json && \
  sed -i 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' nntr_config.json && \
  (grep -q moe_engine nntr_config.json || sed -i 's/\"bad_word_ids\": \[124900\],/\"bad_word_ids\": [124900],\n    \"moe_engine\": \"htp\",/' nntr_config.json) && \
  grep -H do_sample generation_config.json && grep -H -E 'bad_word_ids|num_to_generate|init_seq_len|_engine|_htp_layers' nntr_config.json" \
  | tee $L/config.log
for k in '"do_sample": false' '"bad_word_ids": \[124900\]' '"init_seq_len": 512' '"moe_engine": "htp"' '"moe_htp_layers": ""'; do
  want "config $k" "$(grep -c "$k" $L/config.log)" 1; done

# ---- 2. G1 router cells (1 min)
adb -s $S shell "cd $D && md5sum libnntr_hvx_skel.so && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  ./unittest_hvx_softmax --gtest_filter='HvxM1Ops.RouterTopkMatchesDetBitExact'" > $L/gtest_G1.log 2>&1
grep -E 'libnntr_hvx|M1_OPS_FIELD router|OK \]|FAILED|PASSED' $L/gtest_G1.log
want "G1 router 3 shapes" "$(grep -c 'M1_OPS_FIELD router_topk .* bad_logits=0 bad_sel=0 bad_weight=0' $L/gtest_G1.log)" 3
want "G1 PASSED 1" "$(grep -c 'PASSED  \] 1 test' $L/gtest_G1.log)" 1
therm t1; cool

# ---- 3. the resident router's cost, htp_moe skel vs this set (2 min; profile only)
for v in Pb P; do run $v 64 prof_$v prompt512.txt
  want "prof_$v graph banner" "$(grep -c 'graph: init .*resident=RMSNORM|CONV1D_GATE|QK_NORM|ROPE|ATTN_M1|ADD|ROUTER_TOPK|MOE' $L/prof_$v.log)" 1
  want "prof_$v calls/token=51.00" "$(grep -c 'calls/token=51.00' $L/prof_$v.log)" 1
  grep -h 'graph: calls=' $L/prof_$v.log | sed "s/^/$v /" | tee -a $L/router_prof.txt
done
rp=$(grep -o '^P .* ROUTER_TOPK=[0-9]*' $L/router_prof.txt | grep -o '[0-9]*$')
echo "ROUTER_TOPK pcyc/op: htp_moe skel $(grep -o '^Pb .* ROUTER_TOPK=[0-9]*' $L/router_prof.txt | grep -o '[0-9]*$')  this set ${rp:-?} (target <= 60000; S3 65528)" | tee -a $L/router_prof.txt
want "router <= 60000 pcyc/op" "$([ -n "$rp" ] && [ "$rp" -le 60000 ] && echo y || echo n)" y
therm t2

# ---- 4. nll and text over 8 prompts at G = 256: A self, S forced on A, A / S text (~10 min)
n=0; for p in $P; do n=$((n+1)); cool
  adb -s $S shell "rm -f $D/cont_p0$n.ids"
  run A 256 ppl_A_p0$n $p NNTR_PPL_DECODE=cont_p0$n.ids
  run S 256 ppl_S_p0$n $p NNTR_PPL_DECODE=cont_p0$n.ids
  run A 256 text_A_p0$n $p; run S 256 text_S_p0$n $p
done
therm t3
for i in 1 2 3 4 5 6 7 8; do
  if cmp -s <(nll $L/ppl_A_p0$i.log) <(nll $L/ppl_S_p0$i.log); then echo "p0$i S nll == A"; else echo "BAD p0$i S nll != A"; MIS=$((MIS + 1)); fi
  if cmp -s <(strip $L/text_A_p0$i.log) <(strip $L/text_S_p0$i.log); then echo "p0$i S text identical"; else echo "BAD p0$i S text DIFFERENT"; MIS=$((MIS + 1)); fi
done
want "8 prompts x 4 logs" "$(ls $L/ppl_A_p0?.log $L/ppl_S_p0?.log $L/text_A_p0?.log $L/text_S_p0?.log 2>/dev/null | wc -l)" 32
echo "--- router"; tail -1 $L/router_prof.txt
echo "--- therm"; cat $L/therm.log
echo "=== done $(date '+%F %T %Z')  expectation mismatches: $MIS"
