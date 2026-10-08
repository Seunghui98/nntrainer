#!/usr/bin/env bash
##
# @file    201-s0-run.sh
# @brief   Plan 201 S0 device sitting: the expert pool on the hybrid path,
#          against A and E0 (handoff docs/measurements/201-pool-baseline.md)
#
# Staged by 201-s0-stage.sh as run_s0.sh; run from the stage directory's copy:
#   bash /local/mnt/workspace/htp_moe/201/s0/run_s0.sh [serial]
# Variants (one binary set, env only):
#   A    hybrid, nothing set (the unchanged reference)
#   E0   NNTR_HTP_E2E=1 (two sessions, all experts resident)
#   F28  A + NNTR_MOE_CACHE_EXPERTS=28, model file pre-read (warm)
#   F16w A + NNTR_MOE_CACHE_EXPERTS=16, model file pre-read (warm)
#   F16c A + NNTR_MOE_CACHE_EXPERTS=16, page_cache_evict on the model file
#        every 20 ms for the whole run (cold: every miss read from flash)
# Per G in 64 512 1024: zone0 <= 35 C, then A E0 F28 F16w F16c F16c F16w F28
# E0 A. Then G = 64 profile runs (NNTR_HTP_PROFILE=2) of F28 / F16w / F16c,
# and one A run at G = 1024 with NNTR_MOE_TRACE (pulled, replayed with
# tools/moe_expert_cache_sim.py on the workstation). S1's mapping ceiling
# after every run. RESUMABLE per run: a finished run leaves logs/done/<run>;
# after any STOP (LEAK, 0x8000040e, a failed token, FATAL) reboot the phone
# and run the same command again.
set -u -o pipefail
S=${1:-R3CY10WM83Y} # the S25; never the first `adb devices` entry (a Note20 is attached too)
W=$(cd "$(dirname "$0")" && pwd); L=$W/logs; mkdir -p $L/done
C=/data/local/tmp/nntrainer/causallm; D=$C/s201s0; M=../models/q40-qs4cx-wh
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
run() { # run <A|E0|F28|F16w|F16c> <G> <log> [env ...]
  local v=$1 g=$2 log=$3 e="" pre="" post=""; shift 3
  [ -f $L/done/$log ] && { echo "$log: done earlier, skipped"; return 0; }
  case $v in
    E0) e="NNTR_HTP_E2E=1" ;;
    F28) e="NNTR_MOE_CACHE_EXPERTS=28"; pre="cat $MF > /dev/null &&" ;;
    F16w) e="NNTR_MOE_CACHE_EXPERTS=16"; pre="cat $MF > /dev/null &&" ;;
    F16c) e="NNTR_MOE_CACHE_EXPERTS=16"; pre="(./page_cache_evict $MF 20 & echo \$! > evict.pid) &&"
          post="; kill \$(cat evict.pid); rm -f evict.pid" ;;
  esac
  $AD logcat -c
  $AD shell "cd $D && sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $g/' $M/nntr_config.json && \
    md5sum libnntr_hvx_skel.so && $pre $e $* NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm $M \"\$(cat prompt512.txt)\" $post" > $L/$log.log 2>&1
  $AD logcat -d | grep -iE 'adsprpc|fastrpc|nntr_hvx|token' > $L/$log.logcat || true
  grep -qi '0x8000040e' $L/$log.log && stop "$log: 0x8000040e (stale skel or stub)"
  grep -q FATAL $L/$log.log && stop "$log: $(grep -m1 FATAL $L/$log.log | cut -c1-200)"
  grep -q 'token driver: token .* failed' $L/$log.log && stop "$log: a token failed (AEE_EEXPIRED = a hop timed out)"
  grep -q '^generation:' $L/$log.log || stop "$log cannot generate (a LEAK? reboot; logcat in $log.logcat)"
  echo "$log: $(grep -h -E '^(prefill|generation|generation\(last 64\)):' $L/$log.log | grep -o '[0-9.]* TPS' | tr '\n' ' ')$(grep -h -o 'peak memory: [0-9]* KB' $L/$log.log) $(grep -h -o 'calls/token=[0-9.]*' $L/$log.log)"
  case $v in
    E0)
      want "$log levers banner" "$(grep -c '^\[HTP\] ppl levers=0x0 L1=exact$' $L/$log.log)" 1
      want "$log s2 close: no unmap / detach refused" "$(grep -c 's2: close .* unmap_fail=0 detach_fail=0 ' $L/$log.log)" 1
      want "$log s2 arena after S1 (3840)" "$(grep -c 's2: fc arena weights=67 handles=74 .* s1_arena_mib=3840' $L/$log.log)" 1
      want "$log close clean" "$(grep -c 'token driver: close tokens=[0-9]* hops/token=44.00 .* timeouts=0/0 stale=0/0 .* id_mismatch=0 ' $L/$log.log)" 1
      want "$log calls/token=1.00" "$(grep -cF 'calls/token=1.00' $L/$log.log)" 1 ;;
    *)
      want "$log hybrid banners (dspq on, no s2)" "$(grep -c '\[HTP\] dspq: on' $L/$log.log)/$(grep -c 's2: open' $L/$log.log)" 1/0 ;;
  esac
  touch $L/done/$log
  ceil $log; }

echo "=== 201 S0 $(date '+%F %T %Z') unit=$S (done: $(ls $L/done | wc -l) runs)"
$AD get-state > /dev/null 2>&1 || stop "no device attached"
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
  # the model file's own pages (mincore), not /proc/meminfo's Cached: the
  # file is a symlink and the phone never holds all of it (S0: 2613 of 4116)
  r=$($AD shell "cd $D && ./page_cache_evict $MF" | tr -d '\r')
  echo "evict check: $r" | tee $L/evict_check.txt
  set -- $r # resident <a> -> <b> MiB of <c> MiB
  [ "${2:-0}" -gt 0 ] && [ $(( ${4:-1} * 100 )) -lt "${7:-0}" ] ||
    { echo "BAD evict check: the file's pages did not leave the page cache"; MIS=$((MIS + 1)); }
  touch $L/done/evict_check
fi

for g in 64 512 1024; do
  n=0; for r in r1 r2; do for v in A E0 F28 F16w F16c; do [ -f $L/done/${v}_G${g}_$r ] && n=$((n + 1)); done; done
  [ $n = 10 ] && { echo "--- G=$g done earlier"; continue; }
  echo "--- G=$g $(date +%H:%M:%S)"; cool
  for v in A E0 F28 F16w F16c; do run $v $g ${v}_G${g}_r1; done
  for v in F16c F16w F28 E0 A; do run $v $g ${v}_G${g}_r2; done
  therm t_G$g
done

echo "--- profiles (G = 64, NNTR_HTP_PROFILE=2; not for tok/s)"; cool
for v in F28 F16w F16c; do run $v 64 prof_$v NNTR_HTP_PROFILE=2; done

echo "--- routing trace (A, G = 1024)"
if [ ! -f $L/done/trace_A_G1024 ]; then
  $AD shell "rm -f $D/moe_trace.txt"
  run A 1024 trace_A_G1024 NNTR_MOE_TRACE=moe_trace.txt
  $AD pull $D/moe_trace.txt $L/moe_trace.txt > /dev/null || stop "no moe_trace.txt on the device"
fi
therm t_end

echo "--- speed (prefill / decode / last 64 TPS; peak RSS; text vs A r1 of the same G)"
for g in 64 512 1024; do for r in r1 r2; do for v in A E0 F28 F16w F16c; do f=$L/${v}_G${g}_$r.log
  t=$(cmp -s <(strip $L/A_G${g}_r1.log) <(strip $f) && echo same || echo DIFF)
  echo "G=$g $v $r: $(grep -h -E '^(prefill|generation|generation\(last 64\)):' $f | grep -o '[0-9.]* TPS' | tr '\n' ' ')$(grep -h -o 'peak memory: [0-9]* KB' $f) text=$t"
done; done; done | tee $L/speed.txt
want "every text == A r1 of its G (the pool and E0 are A's bits)" "$(grep -c 'text=DIFF' $L/speed.txt)" 0
echo "--- E0 per session and L0 (G = 512 r1)"; grep -h -E 'per-kind pcyc|moe pcyc/round|token driver: close|L0 us/token|L0 wake' $L/E0_G512_r1.log | sed 's/^/  /'
echo "--- pool profiles"; for v in F28 F16w F16c; do echo "$v:"; grep -h -E 'expert cache misses|expert misses|arena chunk .*mapped total|peak memory|^generation:' $L/prof_$v.log | tail -6 | sed 's/^/  /'; done
echo "--- hit rate per C (simulator on the trace; 22 layers)"
python3 $W/tools/moe_expert_cache_sim.py $L/moe_trace.txt --cache 8 12 16 20 24 28 32 --policies ours belady 2>&1 | tee $L/sim.txt
echo "--- evict check"; cat $L/evict_check.txt
echo "--- ceiling"; sort -t: -k2 -n $L/ceiling.txt | head -2
echo "--- therm"; cat $L/therm.log
echo "=== done $(date '+%F %T %Z') expectation mismatches (this invocation): $MIS"
