#!/usr/bin/env bash
##
# @file    201-s2-run.sh
# @brief   Plan 201 S2 (two-PD half) device sitting: the expert pool inside
#          the per-token E2E entry (P2) against A and E0 (handoff
#          docs/measurements/201-fsu-e2e.md)
#
# Staged by 201-s2-stage.sh as run_s2.sh; run from the stage directory's copy:
#   bash /local/mnt/workspace/htp_moe/201/s2/run_s2.sh [serial, default the S25]
# Variants (one binary set, env only):
#   A     hybrid, nothing set (the unchanged reference)
#   E0    NNTR_HTP_E2E=1 (two sessions, all experts resident)
#   P<C>  NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=<C>, the model file pre-read
#         (warm; its resident MiB printed before the run)
#   P16c  P16 with page_cache_evict on the model file every 20 ms (cold)
# G = 64: cool, A E0 P28 P24 P16, cool, P16 P24 P28 E0 A. G = 512 / 1024:
# cool, A E0 Pbest P28, cool, P28 Pbest E0 A (Pbest: the fastest of P16 /
# P24 / P28 at G = 64, r1 + r2; P24 when that is P28). Then at G = 64: P16c,
# one NNTR_HTP_PROFILE=2 run each of P28 / P24 / P16 (read ms a miss), and
# P32 (every expert in the pool: the no-miss control). S1's ceiling after
# every run. RESUMABLE per run (logs/done/<run>); after any STOP reboot the
# phone and run the same command again.
set -u -o pipefail
S=${1:-R3CY10WM83Y} # the S25; never the first `adb devices` entry (a Note20 is attached too)
W=$(cd "$(dirname "$0")" && pwd); L=$W/logs; mkdir -p $L/done
C=/data/local/tmp/nntrainer/causallm; D=$C/s201s2; M=../models/q40-qs4cx-wh
MF=$M/nntr_lfm2_8b_a1b_q40_arm.bin
AD="adb -s $S"
exec > >(tee -a $L/sitting.out) 2>&1
MIS=0
stop() { echo "STOP: $*"; echo "  -> reboot the phone and run the same command again: it resumes at the first unfinished run"; exit 1; }
want() { if [ "$2" = "$3" ]; then echo "OK  $1 = $2"; else echo "BAD $1: got '$2' want '$3'"; MIS=$((MIS + 1)); fi; }
therm() { echo "$1 $(date +%H:%M:%S) $($AD shell 'dumpsys battery | grep -E "^  (level|temperature)"; cat /sys/class/thermal/thermal_zone0/temp' | tr -d '\r' | tr '\n' ' ')" | tee -a $L/therm.log; }
zone0() { $AD shell cat /sys/class/thermal/thermal_zone0/temp | tr -d '\r'; }
cool() { local i z; for i in $(seq 1 40); do z=$(zone0); [ "$z" -le 35000 ] && break
    echo "zone0=$z > 35000, waiting 30 s ($i/40)"; sleep 30; done; echo "block start zone0=$(zone0)"; }
ceil() {
  $AD shell "cd $D && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_two_sessions \
    --gtest_filter=TwoSessions.S1Ceiling" > $L/ceil_$1.log 2>&1
  local c; c=$(grep -o 'CEILING s1_mmap_mib=[0-9]*' $L/ceil_$1.log | cut -d= -f2)
  echo "ceiling after $1: ${c:-?} MiB" | tee -a $L/ceiling.txt
  [ "${c:-0}" -ge 3840 ] || stop "LEAK: S1 ceiling ${c:-?} MiB after $1"; }
strip() { sed -n '/^=====/q;p' "$1" | perl -0pe 's/\[HTP[^\]\n]*\] [^\n]*\n//g; s/\[PPL\] [^\n]*\n//g' |
  grep -v 'moe m1 gemv\|libnntr_hvx_skel\|nntrainer_causallm\|num_to_generate'; }
run() { # run <A|E0|P<C>|P16c> <G> <log> [env ...]
  local v=$1 g=$2 log=$3 e="" pre="" post=""; shift 3
  [ -f $L/done/$log ] && { echo "$log: done earlier, skipped"; return 0; }
  case $v in
    A) ;;
    E0) e="NNTR_HTP_E2E=1" ;;
    P16c) e="NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=16"
          pre="(./page_cache_evict $MF 20 & echo \$! > evict.pid) &&"
          post="; kill \$(cat evict.pid); rm -f evict.pid" ;;
    P*) e="NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=${v#P}"
        pre="cat $MF > /dev/null && ./page_cache_evict $MF -1 &&" ;;
  esac
  $AD logcat -c
  $AD shell "cd $D && sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $g/' $M/nntr_config.json && \
    md5sum libnntr_hvx_skel.so && $pre $e $* NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm $M \"\$(cat prompt512.txt)\" $post" > $L/$log.log 2>&1
  $AD logcat -d | grep -iE 'adsprpc|fastrpc|nntr_hvx|token' > $L/$log.logcat || true
  grep -qi '0x8000040e' $L/$log.log && stop "$log: 0x8000040e (stale skel or stub)"
  grep -q FATAL $L/$log.log && stop "$log: $(grep -m1 FATAL $L/$log.log | cut -c1-200)"
  grep -q 'token driver: token .* failed\|token driver: .* transport failed' $L/$log.log && stop "$log: a token failed ($(grep -m1 'token driver: token' $L/$log.log | cut -c1-160))"
  grep -q '^generation:' $L/$log.log || stop "$log cannot generate (a LEAK? reboot; logcat in $log.logcat)"
  echo "$log: $(grep -h -E '^(prefill|generation|generation\(last 64\)):' $L/$log.log | grep -o '[0-9.]* TPS' | tr '\n' ' ')$(grep -h -o 'calls/token=[0-9.]*' $L/$log.log) $(grep -h -o 'resident [0-9]* MiB' $L/$log.log)"
  case $v in
    A)
      want "$log hybrid banners (dspq on, no s2)" "$(grep -c '\[HTP\] dspq: on' $L/$log.log)/$(grep -c 's2: open' $L/$log.log)" 1/0 ;;
    *)
      want "$log levers banner" "$(grep -c '^\[HTP\] ppl levers=0x0 L1=exact$' $L/$log.log)" 1
      want "$log s2 close: no unmap / detach refused" "$(grep -c 's2: close .* unmap_fail=0 detach_fail=0 ' $L/$log.log)" 1
      want "$log close clean" "$(grep -c 'token driver: close tokens=[0-9]* hops/token=44.00 .* timeouts=0/0 stale=0/0 .* id_mismatch=0 ' $L/$log.log)" 1
      want "$log calls/token=1.00" "$(grep -cF 'calls/token=1.00' $L/$log.log)" 1
      [ $v = E0 ] || grep -h 'token driver: pool\|token driver: L0 us/token' $L/$log.log | sed 's/^/    /' ;;
  esac
  touch $L/done/$log
  ceil $log; }
mean() { grep -h '^generation:' "$@" | grep -o '[0-9.]* TPS' | awk '{s += $1; n++} END {printf "%.2f", n ? s / n : 0}'; }

echo "=== 201 S2 $(date '+%F %T %Z') unit=$S (done: $(ls $L/done | wc -l) runs)"
$AD get-state > /dev/null 2>&1 || stop "unit $S not attached"
echo "uptime (reboot first): $($AD shell cat /proc/uptime | tr -d '\r')"
$AD shell input keyevent 223 || true
therm t0
[ "$(cd $W && LC_ALL=C md5sum -c md5.txt | grep -vc ': OK$')" = 0 ] || stop "workstation set differs from md5.txt"
$AD shell "mkdir -p $D" && $AD push $W/app/. $D/ > /dev/null
$AD shell "chmod 755 $D/nntrainer_causallm $D/unittest_* $D/page_cache_evict"
$AD shell "cd $D && md5sum $(grep -o '  app/.*' $W/md5.txt | sed 's|  app/||' | tr '\n' ' ')" | tr -d '\r' | awk '{print $1"  app/"$2}' > $L/md5_device.log
diff <(grep -E '^[0-9a-f]{32}  app/' $W/md5.txt | sort -k2) <(sort -k2 $L/md5_device.log) && echo "MD5 OK" || stop "device md5 differs"
$AD shell "cd $D/$M && sed -i 's/\"do_sample\": true/\"do_sample\": false/' generation_config.json && \
  sed -i 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' nntr_config.json && \
  (grep -q moe_engine nntr_config.json || sed -i 's/\"bad_word_ids\": \[124900\],/\"bad_word_ids\": [124900],\n    \"moe_engine\": \"htp\",/' nntr_config.json) && \
  grep -H do_sample generation_config.json && grep -H -E 'bad_word_ids|init_seq_len|moe_engine' nntr_config.json" | tee $L/config.log
[ -f $L/done/ceil_start ] || { ceil start; touch $L/done/ceil_start; }

if [ ! -f $L/done/evict_check ]; then
  cool; run A 64 warmup
  r=$($AD shell "cd $D && ./page_cache_evict $MF" | tr -d '\r')
  echo "evict check: $r" | tee $L/evict_check.txt
  set -- $r # resident <a> -> <b> MiB of <c> MiB
  [ "${2:-0}" -gt 0 ] && [ $(( ${4:-1} * 100 )) -lt "${7:-0}" ] ||
    { echo "BAD evict check: the file's pages did not leave the page cache"; MIS=$((MIS + 1)); }
  touch $L/done/evict_check
fi

echo "--- G=64 $(date +%H:%M:%S)"; cool
for v in A E0 P28 P24 P16; do run $v 64 ${v}_G64_r1; done
cool
for v in P16 P24 P28 E0 A; do run $v 64 ${v}_G64_r2; done
therm t_G64
best=P16; bt=0
for v in P16 P24 P28; do t=$(mean $L/${v}_G64_r1.log $L/${v}_G64_r2.log)
  echo "G=64 mean decode $v: $t"; awk -v a=$t -v b=$bt 'BEGIN {exit !(a > b)}' && { best=$v; bt=$t; }; done
[ $best = P28 ] && best=P24
echo "Pbest=$best (with P28 at G 512 / 1024)" | tee $L/best.txt
for g in 512 1024; do
  echo "--- G=$g $(date +%H:%M:%S)"; cool
  for v in A E0 $best P28; do run $v $g ${v}_G${g}_r1; done
  cool
  for v in P28 $best E0 A; do run $v $g ${v}_G${g}_r2; done
  therm t_G$g
done
echo "--- G=64 extras: cold, profiles, the no-miss control"; cool
run P16c 64 P16c_G64
for v in P28 P24 P16; do run $v 64 prof_$v NNTR_HTP_PROFILE=2; done
run P32 64 P32_G64
therm t_end

echo "--- speed (prefill / decode / last 64 TPS; text vs A r1 of the same G)"
for f in $L/[AEP]*_G*_r[12].log $L/P16c_G64.log $L/P32_G64.log; do b=$(basename $f .log); g=${b#*_G}; g=${g%%_*}
  t=$(cmp -s <(strip $L/A_G${g}_r1.log) <(strip $f) && echo same || echo DIFF)
  echo "$b: $(grep -h -E '^(prefill|generation|generation\(last 64\)):' $f | grep -o '[0-9.]* TPS' | tr '\n' ' ')text=$t"
done | tee $L/speed.txt
want "every text == A r1 of its G (E0 and the pool are A's bits)" "$(grep -c 'text=DIFF' $L/speed.txt)" 0
echo "--- pool lines (misses, miss wait, the server's time a round; L0)"
for f in $L/P*_G*.log $L/prof_*.log; do echo "$(basename $f .log):"; grep -h 'token driver: pool\|L0 us/token\|expert cache misses\|resident [0-9]* MiB' $f | sed 's/^/  /'; done | tee $L/pool.txt
echo "--- E0 / P per session (G = 512 r1)"; for v in E0 P28 $best; do echo "$v"; grep -h -E 'per-kind pcyc|moe pcyc/round|L0 us/token|L0 wake' $L/${v}_G512_r1.log | sed 's/^/  /'; done
echo "--- evict check"; cat $L/evict_check.txt
echo "--- ceiling"; sort -t: -k2 -n $L/ceiling.txt | head -2
echo "--- therm"; cat $L/therm.log
echo "=== done $(date '+%F %T %Z') expectation mismatches (this invocation): $MIS"
