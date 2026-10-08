#!/usr/bin/env bash
# #201 S3 discriminator: does the prefetch readers' placement decide the decode miss cost?
# Same build as sitting 1 (a2ebef9c9) in s201s3 (read only). Alternates
# D (default), B (NNTR_MOE_PREFETCH_CPUS=6,7), P0 (NNTR_MOE_PREFETCH=0: readers never start), x4.
# Samples every thread's CPU / affinity / utime during each run.
set -u -o pipefail
S=${1:?serial}; W=/local/mnt/workspace/htp_moe/201/s3; L=$W/logs_disc; mkdir -p $L
C=/data/local/tmp/nntrainer/causallm; D=$C/s201s3; M=../models/q40-qs4cx-wh
MF=$M/nntr_lfm2_8b_a1b_q40_arm.bin; AD="adb -s $S"
exec > >(tee -a $L/sitting.out) 2>&1
zone0() { $AD shell cat /sys/class/thermal/thermal_zone0/temp | tr -d '\r'; }
cool() { local i z; for i in $(seq 1 40); do z=$(zone0); [ "$z" -le 35000 ] && break; sleep 30; done; echo "block start zone0=$(zone0)"; }
strip() { sed -n '/^=====/q;p' "$1" | perl -0pe 's/\[HTP[^\]\n]*\] [^\n]*\n//g; s/\[PPL\] [^\n]*\n//g' |
  grep -v 'moe m1 gemv\|libnntr_hvx_skel\|nntrainer_causallm\|num_to_generate\|^resident [0-9]* MiB\|HTP-PROFILE'; }
sampler() { # threads: tid comm-ish state cpu(field39) utime(14) + affinity, every 0.3 s
  $AD shell 'p=""; while [ -z "$p" ]; do p=$(pidof nntrainer_causallm); done; \
    while [ -d /proc/$p ]; do echo "T $(date +%s.%N)"; for t in /proc/$p/task/*; do \
      s=$(cat $t/stat 2>/dev/null) || continue; a=$(grep Cpus_allowed_list $t/status 2>/dev/null | cut -f2); \
      echo "$s" | awk -v a="$a" "{print \$1, \$3, \$14, \$15, \$39, a}"; done; sleep 0.3; done' > $L/$1.threads 2>/dev/null; }
run() { local log=$1; shift
  sampler $log & local sp=$!
  $AD shell "cd $D && cat $MF > /dev/null && ./page_cache_evict $MF -1 && \
    NNTR_HTP_E2E=1 NNTR_HTP_E2E_PDS=1 NNTR_MOE_CACHE_EXPERTS=28 NNTR_HTP_PROFILE=2 $* \
    NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm $M \"\$(cat prompt512.txt)\"" > $L/$log.log 2>&1
  wait $sp
  local t; t=$(cmp -s <(strip $W/logs/A_G64_r1.log) <(strip $L/$log.log) && echo same || echo DIFF)
  echo "$log: $(grep -h '^generation:' $L/$log.log | grep -o '[0-9.]* TPS') | $(grep -h -o 'miss_wait_us/token=[0-9.]* rounds=[0-9]* arm_ms/round=[0-9.]*' $L/$log.log) | $(grep -h -o 'file read [0-9.]* ms ([0-9.]* ms/miss)' $L/$log.log) | $(grep -h -o 'caller cpus.*' $L/$log.log) | text $t"
}
echo "=== disc $(date '+%F %T') uptime $($AD shell cat /proc/uptime | tr -d '\r')"
$AD shell "cd $D && md5sum nntrainer_causallm libnntr_hvx_skel.so libnntrainer.so libcausallm_core.so" | tr -d '\r' | awk '{print $1"  app/"$2}' | sort -k2 > $L/md5_device.log
diff <(grep -E 'app/(nntrainer_causallm|libnntr_hvx_skel.so|libnntrainer.so|libcausallm_core.so)$' $W/md5.txt | sort -k2) $L/md5_device.log && echo "MD5 OK" || { echo STOP md5; exit 1; }
$AD shell "cd $D && sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": 64/' $M/nntr_config.json"
cool
for i in 1 2 3 4; do
  run D_$i
  run B_$i NNTR_MOE_PREFETCH_CPUS=6,7
  run P0_$i NNTR_MOE_PREFETCH=0
done
$AD shell "cd $D && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_two_sessions --gtest_filter=TwoSessions.S1Ceiling" 2>&1 | grep -o 'CEILING s1_mmap_mib=[0-9]*'
echo "=== done $(date '+%F %T')"
