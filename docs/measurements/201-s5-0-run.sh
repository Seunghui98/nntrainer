#!/usr/bin/env bash
##
# @file    201-s5-0-run.sh
# @brief   Plan 201 S5-0 sitting (Gemma-4 26B-A4B DUMMY file, QS2CX_WH
#          experts): the C sweep of the one-PD E2E path (handoff
#          docs/measurements/201-s5-0-gemma-dummy.md)
#
# Run from the workstation with the phone on USB, after the app set is
# pushed to $D (md5 checked) and the model dir holds the config copy:
#   bash docs/measurements/201-s5-0-run.sh <serial> <log dir> <cmax> ["<G list>"] [run tag]
# Per G in 64 / 512 / 1024: cool (zone0 <= 35 C), A16 (hybrid, C = 16),
# E<cmax>, E32, E16, E8 (NNTR_HTP_E2E=1, one PD); G = 512 adds E16 r2.
# S1's ceiling after every run; /proc/vmstat pgpgin / workingset_refault
# around every run. RESUMABLE per run (<log dir>/done/<run>).
set -u -o pipefail
S=${1:?serial}; L=${2:?log dir}; CMAX=${3:?cmax}; GS=${4:-64 512 1024}; R=${5:-r1}
mkdir -p "$L/done"
AD="adb -s $S"
C=/data/local/tmp/nntrainer/causallm; D=$C/s201s5; M=../models/gemma4_26b
exec > >(tee -a "$L/sweep.out") 2>&1
stop() { echo "STOP: $*"; exit 1; }
zone0() { $AD shell cat /sys/class/thermal/thermal_zone0/temp | tr -d '\r'; }
cool() { local i z; for i in $(seq 1 40); do z=$(zone0); [ "$z" -le 35000 ] && break
    echo "zone0=$z > 35000, waiting 30 s ($i/40)"; sleep 30; done
  echo "block start $(date +%H:%M:%S) zone0=$(zone0)"; }
vm() { $AD shell "grep -E '^(pgpgin|workingset_refault_file) ' /proc/vmstat" | tr -d '\r' | awk '{printf "%s=%s ", $1, $2}'; }
ceil() {
  $AD shell "cd $D && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_two_sessions \
    --gtest_filter=TwoSessions.S1Ceiling" > "$L/ceil_$1.log" 2>&1
  local c; c=$(grep -o 'CEILING s1_mmap_mib=[0-9]*' "$L/ceil_$1.log" | cut -d= -f2)
  echo "ceiling after $1: ${c:-?} MiB" | tee -a "$L/ceiling.txt"
  [ "${c:-0}" -ge 3840 ] || stop "LEAK: S1 ceiling ${c:-?} MiB after $1"; }
run() { # run <A16|E<C>> <G> <name>
  local v=$1 g=$2 n=$3 e
  [ -f "$L/done/$n" ] && { echo "$n: done earlier, skipped"; return 0; }
  case $v in
    A*) e="NNTR_MOE_CACHE_EXPERTS=${v#A}" ;;
    E*) e="NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=${v#E}" ;;
  esac
  $AD shell "sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $g/' $D/$M/nntr_config.json"
  local z0 v0 v1; z0=$(zone0); v0=$(vm)
  $AD shell "cd $D && $e NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm $M" > "$L/$n.log" 2>&1
  v1=$(vm)
  echo "vm_before $v0" >> "$L/$n.log"; echo "vm_after $v1" >> "$L/$n.log"; echo "zone0_start $z0" >> "$L/$n.log"
  grep -q '^generation:' "$L/$n.log" || stop "$n cannot generate: $(grep -m1 -E 'FATAL|Abort' "$L/$n.log" | cut -c1-200)"
  echo "$n: $(grep -h -E '^(prefill|generation):' "$L/$n.log" | grep -o '[0-9.]* TPS' | tr '\n' ' ')$(grep -h -o 'calls/token=[0-9.]*' "$L/$n.log") $(grep -h -o 'misses/token=[0-9.]*' "$L/$n.log") $(grep -h -o '^peak memory: [0-9]*' "$L/$n.log") zone0=$z0"
  touch "$L/done/$n"
  ceil "$n"; }

echo "=== 201 S5-0 sweep $(date '+%F %T %Z') unit=$S cmax=$CMAX uptime=$($AD shell cat /proc/uptime | tr -d '\r')"
for g in $GS; do
  echo "--- G=$g"; cool
  run A16 $g A16_G${g}_$R
  for c in $CMAX 32 16 8; do run E$c $g E${c}_G${g}_$R; done
  [ $g = 512 ] && [ $R = r1 ] && run E16 $g E16_G512_r2
done
echo "=== done $(date '+%F %T %Z')"
