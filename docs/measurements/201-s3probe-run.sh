#!/usr/bin/env bash
##
# @file    run_s3probe.sh
# @brief   #201 S3 probe: why a pool-28 miss costs 3.5-3.8 ms when a pool-24
#          miss costs 0.7 ms (one PD, same build as sitting 1 = a2ebef9c9).
#          Hypothesis under test: the reader threads idle longer between the
#          rarer misses and their cores downclock / sleep; pinning the readers
#          to the big cores (6,7) or using one reader changes the miss cost.
# Usage:   bash run_s3probe.sh <serial>   (G = 64, NNTR_HTP_PROFILE=2 on every run)
# Variants (all one PD, NNTR_HTP_E2E=1 NNTR_HTP_E2E_PDS=1, model pre-read):
#   Q28        pool 28, readers default (4, every core but the caller's)
#   Q28_R1     + NNTR_MOE_PREFETCH_READERS=1
#   Q28_BIG    + NNTR_MOE_PREFETCH_CPUS=6,7
#   Q28_BIG1   + NNTR_MOE_PREFETCH_READERS=1 NNTR_MOE_PREFETCH_CPUS=7
#   Q24        pool 24, readers default (the control: cheap misses in sitting 1)
#   Q24_BIG    + NNTR_MOE_PREFETCH_CPUS=6,7
# Reads: tok/s, pool misses / miss wait / server ms a round, the profile's
# "expert cache misses ... ms/miss", text == sitting 1's A_G64_r1, ceiling.
set -u -o pipefail
S=${1:?usage: run_s3probe.sh <serial>}
W=$(cd "$(dirname "$0")" && pwd); L=$W/logs_probe; mkdir -p $L/done
C=/data/local/tmp/nntrainer/causallm; D=$C/s201s3; M=../models/q40-qs4cx-wh
MF=$M/nntr_lfm2_8b_a1b_q40_arm.bin
AD="adb -s $S"
exec > >(tee -a $L/sitting.out) 2>&1
MIS=0
stop() { echo "STOP: $*"; exit 1; }
want() { if [ "$2" = "$3" ]; then echo "OK  $1 = $2"; else echo "BAD $1: got '$2' want '$3'"; MIS=$((MIS + 1)); fi; }
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
  grep -v 'moe m1 gemv\|libnntr_hvx_skel\|nntrainer_causallm\|num_to_generate\|^resident [0-9]* MiB\|HTP-PROFILE'; }
run() { # run <log> <pool C> [env ...]
  local log=$1 pc=$2; shift 2
  [ -f $L/done/$log ] && { echo "$log: done earlier, skipped"; return 0; }
  $AD logcat -c
  $AD shell "cd $D && sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": 64/' $M/nntr_config.json && \
    cat $MF > /dev/null && ./page_cache_evict $MF -1 && \
    NNTR_HTP_E2E=1 NNTR_HTP_E2E_PDS=1 NNTR_MOE_CACHE_EXPERTS=$pc NNTR_HTP_PROFILE=2 $* \
    NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm $M \"\$(cat prompt512.txt)\"" > $L/$log.log 2>&1
  $AD logcat -d | grep -iE 'adsprpc|fastrpc|nntr_hvx|token' > $L/$log.logcat || true
  grep -qi '0x8000040e' $L/$log.log && stop "$log: 0x8000040e (stale skel or stub)"
  grep -q FATAL $L/$log.log && stop "$log: $(grep -m1 FATAL $L/$log.log | cut -c1-200)"
  grep -q '^generation:' $L/$log.log || stop "$log cannot generate (logcat in $log.logcat)"
  echo "$log: $(grep -h -E '^(prefill|generation):' $L/$log.log | grep -o '[0-9.]* TPS' | tr '\n' ' ')$(grep -h -o 'calls/token=[0-9.]*' $L/$log.log) $(grep -h -o 'resident [0-9]* MiB' $L/$log.log)"
  grep -h 'token driver: pool\|expert cache misses' $L/$log.log | sed 's/^/    /' | cut -c1-200
  want "$log one PD" "$(grep -c 'token driver: on .* pds=1$' $L/$log.log)" 1
  want "$log close clean" "$(grep -c 'token driver: close tokens=[0-9]* hops/token=0.00 .* timeouts=0/0 stale=0/0 .* id_mismatch=0 ' $L/$log.log)" 1
  want "$log text == sitting 1 A_G64_r1" "$(cmp -s <(strip $W/logs/A_G64_r1.log) <(strip $L/$log.log) && echo same || echo DIFF)" same
  touch $L/done/$log
  ceil $log; }

echo "=== 201 S3 probe $(date '+%F %T %Z') unit=$S (done: $(ls $L/done | wc -l) runs)"
$AD get-state > /dev/null 2>&1 || stop "unit $S not attached"
echo "uptime (reboot first): $($AD shell cat /proc/uptime | tr -d '\r')"
$AD shell "cd $D && md5sum nntrainer_causallm libnntr_hvx_skel.so libnntrainer.so libcausallm_core.so" | tr -d '\r' | awk '{print $1"  app/"$2}' > $L/md5_device.log
diff <(grep -E 'app/(nntrainer_causallm|libnntr_hvx_skel.so|libnntrainer.so|libcausallm_core.so)$' $W/md5.txt | sort -k2) <(sort -k2 $L/md5_device.log) && echo "MD5 OK (sitting 1 build)" || stop "device app differs from sitting 1's md5.txt"
[ -f $L/done/ceil_start ] || { ceil start; touch $L/done/ceil_start; }
cool
run Q28      28
run Q28_R1   28 NNTR_MOE_PREFETCH_READERS=1
run Q28_BIG  28 NNTR_MOE_PREFETCH_CPUS=6,7
run Q28_BIG1 28 NNTR_MOE_PREFETCH_READERS=1 NNTR_MOE_PREFETCH_CPUS=7
run Q24      24
run Q24_BIG  24 NNTR_MOE_PREFETCH_CPUS=6,7
cool
run Q28_r2   28
run Q28_BIG_r2 28 NNTR_MOE_PREFETCH_CPUS=6,7
echo "--- summary (decode tok/s | misses/token | miss_wait us/token | server ms/round | profile ms/miss)"
for f in Q28 Q28_R1 Q28_BIG Q28_BIG1 Q24 Q24_BIG Q28_r2 Q28_BIG_r2; do
  echo "$f: $(grep -h '^generation:' $L/$f.log | grep -o '[0-9.]* TPS') | $(grep -h -o 'misses/token=[0-9.]* miss_wait_us/token=[0-9.]* rounds=[0-9]* arm_ms/round=[0-9.]*' $L/$f.log) | $(grep -h -o 'expert cache misses: [0-9]*, file read [0-9.]* ms ([0-9.]* ms/miss)' $L/$f.log)"
done | tee $L/summary.txt
echo "--- ceiling"; cat $L/ceiling.txt
echo "=== done $(date '+%F %T %Z') expectation mismatches: $MIS"
