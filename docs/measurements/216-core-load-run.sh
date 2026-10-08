#!/usr/bin/env bash
##
# @file    216-core-load-run.sh
# @brief   #216 step 1: per-core system load across a boot's first 5 min,
#          beside Q28 G = 64 profiled runs (plan 216 §4 steps 1-2)
#
# Reboots the phone, starts 216-sampler.sh at once and (unless TOP=0) a
# `top -H` frame every 5 s, then runs Q28 G = 64 NNTR_HTP_PROFILE=2 at uptime
# 60 / 120 / 180 / 300 s ("boot" mode) or idles to 360 s and runs four back
# to back ("control" mode). Sitting 1's build (#201 S3, a2ebef9c9), staged
# read-only from $A1 into $C/s216a. Text checked against sitting 1's A_G64_r1;
# uptime stamped before and after every run; ceiling after every run.
# Usage: bash 216-core-load-run.sh <serial> <label> [boot|control]
set -u -o pipefail
S=${1:?serial}; LB=${2:?label}; MODE=${3:-boot}
H=$(cd "$(dirname "$0")" && pwd)
W=/local/mnt/workspace/htp_moe/216; L=$W/logs_core/$LB; mkdir -p $L
A1=/local/mnt/workspace/htp_moe/201/s3/app; REF=/local/mnt/workspace/htp_moe/201/s3/logs/A_G64_r1.log
C=/data/local/tmp/nntrainer/causallm; D=$C/s216a; M=../models/q40-qs4cx-wh
MF=$M/nntr_lfm2_8b_a1b_q40_arm.bin; AD="adb -s $S"
exec > >(tee -a $L/sitting.out) 2>&1
up() { $AD shell cat /proc/uptime | tr -d '\r' | cut -d' ' -f1; }
strip() { sed -n '/^=====/q;p' "$1" | perl -0pe 's/\[HTP[^\]\n]*\] [^\n]*\n//g; s/\[PPL\] [^\n]*\n//g' |
  grep -v 'moe m1 gemv\|libnntr_hvx_skel\|nntrainer_causallm\|num_to_generate\|^resident [0-9]* MiB\|HTP-PROFILE'; }
# stage (idempotent): sitting 1's app set + the sampler, md5 checked on the device
$AD shell mkdir -p $D
for f in nntrainer_causallm libnntr_hvx_skel.so libnntrainer.so libcausallm_core.so libccapi-nntrainer.so \
         libc++_shared.so libsdkl.so page_cache_evict prompt512.txt unittest_hvx_two_sessions; do
  [ "$($AD shell md5sum $D/$f 2>/dev/null | cut -d' ' -f1)" = "$(md5sum < $A1/$f | cut -d' ' -f1)" ] || $AD push $A1/$f $D/ >/dev/null
done
$AD push $H/216-sampler.sh $D/sampler.sh >/dev/null; $AD shell chmod 755 $D/page_cache_evict $D/nntrainer_causallm $D/unittest_hvx_two_sessions
$AD shell "cd $D && md5sum nntrainer_causallm libnntr_hvx_skel.so libnntrainer.so libcausallm_core.so" | tr -d '\r' | awk '{print $1"  app/"$2}' | sort -k2 > $L/md5_device.log
diff <(grep -E 'app/(nntrainer_causallm|libnntr_hvx_skel.so|libnntrainer.so|libcausallm_core.so)$' $A1/../md5.txt | sort -k2) $L/md5_device.log && echo "MD5 OK" || { echo STOP md5; exit 1; }
$AD shell "sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": 64/' $C/models/q40-qs4cx-wh/nntr_config.json"
echo "=== $LB $MODE reboot $(date '+%F %T')"
$AD reboot; $AD wait-for-device
until [ "$($AD shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" = 1 ]; do sleep 1; done
echo "boot_completed uptime $(up)"
$AD shell "rm -f $D/core.samples; nohup sh $D/sampler.sh $D/core.samples 0.5 >/dev/null 2>&1 &"
if [ "${TOP:-1}" = 1 ]; then
  $AD shell 'while :; do read u x < /proc/uptime; echo "TOP $u"; top -b -H -d 1 -n 2 -m 12 | tail -13; sleep 4; done' > $L/top.frames 2>/dev/null &
  TP=$!
fi
wait_up() { while [ "$(up | cut -d. -f1)" -lt $1 ]; do sleep 1; done; }
run() { local log=$1
  local u0; u0=$(up)
  $AD shell "cd $D && cat $MF > /dev/null && ./page_cache_evict $MF -1 && \
    NNTR_HTP_E2E=1 NNTR_HTP_E2E_PDS=1 NNTR_MOE_CACHE_EXPERTS=28 NNTR_HTP_PROFILE=2 \
    NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm $M \"\$(cat prompt512.txt)\"" > $L/$log.log 2>&1
  local u1; u1=$(up)
  local t; t=$(cmp -s <(strip $REF) <(strip $L/$log.log) && echo same || echo DIFF)
  echo "$log: up $u0-$u1 | $(grep -h '^prefill:' $L/$log.log | grep -o '[0-9.]* TPS') prefill | $(grep -h '^generation:' $L/$log.log | grep -o '[0-9.]* TPS') | $(grep -h -o 'misses=[0-9]* misses/token=[0-9.]* miss_wait_us/token=[0-9.]* rounds=[0-9]* arm_ms/round=[0-9.]*' $L/$log.log) | $(grep -h -o 'file read [0-9.]* ms ([0-9.]* ms/miss)' $L/$log.log) | $(grep -h -o 'calls/token=[0-9.]*' $L/$log.log) | text $t"
  echo "  ceiling $($AD shell "cd $D && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_two_sessions --gtest_filter=TwoSessions.S1Ceiling" 2>&1 | grep -o 'CEILING s1_mmap_mib=[0-9]*')"
}
if [ "$MODE" = control ]; then
  wait_up 360; for i in 1 2 3 4; do run R$i; done
else
  for u in 60 120 180 300; do wait_up $u; run U$u; done
fi
$AD shell touch $D/core.samples.stop; sleep 1; [ -n "${TP:-}" ] && kill $TP 2>/dev/null
$AD pull $D/core.samples $L/core.samples >/dev/null
echo "=== $LB done $(date '+%F %T') uptime $(up)"
