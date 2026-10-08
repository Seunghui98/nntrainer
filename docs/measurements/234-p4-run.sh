#!/usr/bin/env bash
##
# @file    234-p4-run.sh
# @brief   #234 P4 sitting (Gemma-4 26B-A4B DUMMY, QS2CX_WH experts): the
#          one-PD decode with its FC / DENSE_FFN ops on the FC WH sidecar
#          (F<C>) against the hybrid (A<C>) and the sidecar-less one-PD
#          token (E<C>) (handoff docs/measurements/234-p4-fcwh-decode.md)
#
# 201-s5-0-run.sh's runner with one more variant. Run from the workstation
# with the phone on USB, after the app set is pushed to $D (md5 checked),
# ../models/gemma4_26b holds the original config.json and
# ../models/gemma4_26b_fcwh links its files and adds the sidecar and its
# own nntr_config.json (fc_wh_file_name / fc_wh_format):
#   bash docs/measurements/234-p4-run.sh <serial> <log dir> ["<G list>"] [run tag] [C]
# Per G: cool (zone0 <= 35 C), A<C> (hybrid), F<C> (NNTR_HTP_E2E=1, the
# sidecar model), E<C> (NNTR_HTP_E2E=1, no sidecar); G = 512 r1 adds F<C>
# r2. S1's ceiling after every run; /proc/vmstat pgpgin /
# workingset_refault around every run. RESUMABLE per run (<log dir>/done/<run>).
set -u -o pipefail
S=${1:?serial}; L=${2:?log dir}; GS=${3:-512 64 1024}; R=${4:-r1}; CC=${5:-16}
mkdir -p "$L/done"
AD="adb -s $S"
C=/data/local/tmp/nntrainer/causallm; D=$C/s234p4
M=../models/gemma4_26b; MW=../models/gemma4_26b_fcwh
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
run() { # run <A<C>|E<C>|F<C>> <G> <name>
  local v=$1 g=$2 n=$3 e m=$M
  [ -f "$L/done/$n" ] && { echo "$n: done earlier, skipped"; return 0; }
  case $v in
    A*) e="NNTR_MOE_CACHE_EXPERTS=${v#A}" ;;
    E*) e="NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=${v#E}" ;;
    F*) e="NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=${v#F}"; m=$MW ;;
  esac
  $AD shell "sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $g/' $D/$m/nntr_config.json"
  local z0 v0 v1; z0=$(zone0); v0=$(vm)
  $AD shell "cd $D && $e NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm $m" > "$L/$n.log" 2>&1
  v1=$(vm)
  echo "vm_before $v0" >> "$L/$n.log"; echo "vm_after $v1" >> "$L/$n.log"; echo "zone0_start $z0" >> "$L/$n.log"
  grep -q '^generation:' "$L/$n.log" || { echo "$n cannot generate: $(grep -m1 -E 'FATAL|Abort|ERROR' "$L/$n.log" | cut -c1-240)"; touch "$L/done/$n"; ceil "$n"; return 1; }
  echo "$n: $(grep -h -E '^(prefill|generation):' "$L/$n.log" | grep -o '[0-9.]* TPS' | tr '\n' ' ')$(grep -h -o 'calls/token=[0-9.]*' "$L/$n.log") $(grep -h -o 'misses/token=[0-9.]*' "$L/$n.log") $(grep -h -o 'wh_handles=[0-9]*' "$L/$n.log") $(grep -h -o '^peak memory: [0-9]*' "$L/$n.log") zone0=$z0"
  touch "$L/done/$n"
  ceil "$n"; }

echo "=== 234 P4 sweep $(date '+%F %T %Z') unit=$S C=$CC G=[$GS] uptime=$($AD shell cat /proc/uptime | tr -d '\r')"
for g in $GS; do
  echo "--- G=$g"; cool
  run A$CC $g A${CC}_G${g}_$R
  run F$CC $g F${CC}_G${g}_$R
  run E$CC $g E${CC}_G${g}_$R
  [ $g = 512 ] && [ $R = r1 ] && run F$CC $g F${CC}_G512_r2
done
echo "=== done $(date '+%F %T %Z')"
