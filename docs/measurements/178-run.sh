#!/usr/bin/env bash
# Plan 178 step 4 (docs/measurements/178-second-dsp-session.md on
# htp/178-probe): a second cDSP session beside the loaded app's.
# Variants: A = the unchanged reference (app set + this set's skel, nothing
# set; the probe changes no product code) at G = 64 / 512 / 1024 x 2, and
# P = unittest_hvx_two_sessions (TwoSessions.*: Q1-Q5, S2_FIELD lines) run
# cold (after A's G 64 / 512 pairs, cooled) and warm (right after A's
# G = 1024 pair). HvxFcQ4.MatchesSpecBitExact first, as the stale-skel canary.
# Usage: bash /local/mnt/workspace/htp_moe/178/run_178.sh [serial]   (~35 min)
# Logs to $L, summary to $L/sitting.out. Stops on a stale skel, a missing
# unit or an md5 mismatch; every other missing expected line is counted
# (expectation mismatches: N); the plan's stop rules (S2_STOP lines) are
# reported, not enforced. No adb from the agent: the orchestrator runs it.
set -u -o pipefail
S=${1:-R3CY10WM83Y}
W=/local/mnt/workspace/htp_moe/178; L=$W/logs; mkdir -p $L
C=/data/local/tmp/nntrainer/causallm; D=$C/s178; M=../models/q40-qs4cx-wh
{ set +u; source /home/j2z0-lee/nntrainer-178p/tools/htp/env.sh > /dev/null 2>&1; set -u; }
exec > >(tee -a $L/sitting.out) 2>&1
MIS=0; RULE=0
stop() { echo "STOP: $*"; exit 1; }
want() { # want <label> <got> <expected>
  if [ "$2" = "$3" ]; then echo "OK  $1 = $2"; else echo "BAD $1: got '$2' want '$3'"; MIS=$((MIS + 1)); fi; }
therm() { echo "$1 $(date +%H:%M:%S) $(adb -s $S shell 'dumpsys battery | grep -E "^  (level|temperature)"; cat /sys/class/thermal/thermal_zone0/temp' | tr -d '\r' | tr '\n' ' ')" | tee -a $L/therm.log; }
zone0() { adb -s $S shell cat /sys/class/thermal/thermal_zone0/temp | tr -d '\r'; }
cool() { for i in $(seq 1 40); do z=$(zone0); [ "$z" -le 35000 ] && break
  echo "zone0=$z > 35000, waiting 30 s ($i/40)"; sleep 30; done; echo "zone0=$(zone0)"; }
run() { # run <G> <log name>: A, prompt 512, nothing set
  local g=$1 log=$2
  adb -s $S shell "cd $D && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $g/' $M/nntr_config.json && \
    grep num_to_generate $M/nntr_config.json && md5sum libnntr_hvx_skel.so && \
    LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. NNTR_NUM_THREADS=8 \
    ./nntrainer_causallm $M \"\$(cat prompt512.txt)\"" > $L/$log.log 2>&1
  grep -qi '0x8000040e' $L/$log.log && stop "$log: 0x8000040e (stale skel, rule 3)"
  echo "$log: $(grep -h -E '^(prefill|generation):' $L/$log.log | grep -o '[0-9.]* TPS' | tr '\n' ' ')"
  # rule 36: the A banner
  want "$log applied=0x703e1 dma_bypass=1" "$(grep -c 'moe m1 gemv: on (applied=0x703e1).* dma_bypass=1' $L/$log.log)" 1
  want "$log dspq: on" "$(grep -c 'dspq: on' $L/$log.log)" 1
  want "$log dspq close bad=0" "$(grep -c 'dspq: close calls=\([0-9]*\) served=\1 bad=0' $L/$log.log)" 1
  want "$log no graph (calls/token absent)" "$(grep -c 'calls/token' $L/$log.log)" 0
}
probe() { # probe <log name>: TwoSessions.* (Q1-Q5), FARF of the opens from logcat
  adb -s $S logcat -c || true
  echo "$1 zone0 before: $(zone0)"
  adb -s $S shell "cd $D && md5sum libnntr_hvx_skel.so unittest_hvx_two_sessions && \
    LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. timeout 600 ./unittest_hvx_two_sessions \
    --gtest_filter='TwoSessions.*'" > $L/$1.log 2>&1
  echo "$1 zone0 after: $(zone0)"
  adb -s $S logcat -d 2>/dev/null | grep -E 'nntr_hvx_open|lite|hexkl_micro|AEE|fastrpc' | tail -200 > $L/$1.logcat || true
  grep -qi '0x8000040e' $L/$1.log && stop "$1: 0x8000040e (stale skel, rule 3)"
  grep -E '^S2_(FIELD|STOP)|OK \]|FAILED|SKIPPED|PASSED' $L/$1.log
  want "$1 five tests ran" "$(grep -cE '^\[ *(OK|FAILED|SKIPPED) *\] TwoSessions\.Q[1-5]_' $L/$1.log)" 5
  grep -h '^S2_STOP' $L/$1.log | while read -r l; do echo "PLAN STOP RULE HIT (reported, sitting continues): $l"; done
  RULE=$((RULE + $(grep -c '^S2_STOP' $L/$1.log)))
}
strip() { sed -n '/^=====/q;p' "$1" | perl -0pe 's/\[HTP\] [^\n]*\n//g' |
  grep -v 'moe m1 gemv\|libnntr_hvx_skel\|nntrainer_causallm\|num_to_generate'; }

echo "=== 178 sitting  $(date '+%F %T %Z')  unit=$S"
# ---- 0. device state, cool start (1-5 min)
adb devices | tee $L/devices.log
adb devices | grep -q "^$S[[:space:]]*device" || stop "unit $S not attached"
adb -s $S shell input keyevent 223 || true
therm t0; cool

# ---- 1. install and config (3 min)
[ "$(cd $W/set && LC_ALL=C md5sum -c md5.txt | grep -vc ': OK$')" = 0 ] || stop "workstation set differs from set/md5.txt"
adb -s $S shell ls -l $C/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin
adb -s $S shell "rm -rf $D && mkdir -p $D" && adb -s $S push $W/set/. $D/ > /dev/null
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

# ---- 2. stale-skel canary: the exact FC == spec on this skel (2 min)
adb -s $S shell "cd $D && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  ./unittest_hvx_softmax --gtest_filter='HvxFcQ4.MatchesSpecBitExact'" > $L/canary.log 2>&1
grep -qi '0x8000040e' $L/canary.log && stop "canary: 0x8000040e (stale skel, rule 3)"
grep -E 'FC_Q4_FIELD total|OK \]|FAILED' $L/canary.log
want "canary FC total bad=0" "$(grep -c '^FC_Q4_FIELD total bad=0$' $L/canary.log)" 1
therm t1; cool

# ---- 3. A at G = 64 and 512, twice each (6 min)
for g in 64 512; do run $g A_G${g}_r1; run $g A_G${g}_r2; therm t2_G$g; done

# ---- 4. P cold (2 min)
cool; probe P_cold; therm t3

# ---- 5. A at G = 1024 twice, then P warm at once (5 min)
cool; run 1024 A_G1024_r1; run 1024 A_G1024_r2; therm t4
probe P_warm; therm t5

# ---- 6. summary
echo "--- A speed (prefill / decode / last 64 TPS; text vs run 1 of the same G)"
for g in 64 512 1024; do for r in r1 r2; do f=$L/A_G${g}_$r.log
  t=$(cmp -s <(strip $L/A_G${g}_r1.log) <(strip $f) && echo same || echo DIFF)
  echo "G=$g A $r: $(grep -h -E '^(prefill|generation|generation\(last 64\)):' $f | grep -o '[0-9.]* TPS' | tr '\n' ' ')text=$t"
done; done > $L/speed.txt
cat $L/speed.txt
want "A r2 text == r1 at every G" "$(grep -c 'text=DIFF' $L/speed.txt)" 0
for p in P_cold P_warm; do echo "--- $p"; grep -h -E '^S2_(FIELD|STOP)' $L/$p.log; done
echo "--- therm"; cat $L/therm.log
echo "=== done $(date '+%F %T %Z')  expectation mismatches: $MIS  plan stop rules hit: $RULE"
