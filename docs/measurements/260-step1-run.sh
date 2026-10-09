#!/usr/bin/env bash
##
# @file    260-step1-run.sh
# @brief   #260 step 1: pure upstream #4415 @ 86bb496b6 on the S25 (R3CY205ZMND),
#          the delivered QS4CX 26B file, their hybrid / all-NPU config
#
# usage: 260-step1-run.sh OUTDIR CELL [ENV=VAL ...]
#   CELL is a directory under /data/local/tmp/nntrainer/s260cfg (p447_g64, ...).
#   Waits for no other nntrainer_causallm, 180 s, then until the hottest
#   cpu*/nsp* zone <= 38 C and the battery <= 30.0 C (doc 57 section 7.5;
#   gives up after 20 min and records the temperatures), then runs once.
#   Writes OUTDIR/CELL[.tag].log with the temps before / after and the
#   device md5 of the binary set.
set -u
OUT=$1; CELL=$2; shift 2
ENVS="$*"
TAG=${TAG:-}
SER=R3CY205ZMND
D=${BIN:-/data/local/tmp/nntrainer/causallm/s260p}
C=/data/local/tmp/nntrainer/s260cfg/$CELL
mkdir -p "$OUT"
LOG="$OUT/$CELL${TAG:+.$TAG}.log"
A() { adb -s $SER "$@"; }
temps() {
  A shell 'm=0; for z in /sys/class/thermal/thermal_zone*; do case "$(cat $z/type)" in cpu*|nsp*) t=$(cat $z/temp); [ "$t" -gt "$m" ] && m=$t;; esac; done; b=$(dumpsys battery | grep "  temperature" | tr -dc 0-9); z0=$(cat /sys/class/thermal/thermal_zone0/temp); echo "$m $b $z0"'
}
while A shell 'ps -A' | grep -q 'nntrainer_causall[m]'; do sleep 10; done
sleep "${COOL_S:-180}"
for i in $(seq 120); do
  read -r soc bat z0 <<<"$(temps)"
  [ "$soc" -le 38000 ] && [ "$bat" -le 300 ] && break
  sleep 10
done
{
  echo "CELL $CELL env=[$ENVS] start soc=$soc bat=$bat zone0=$z0 $(date -Is)"
  A shell "cd $D && md5sum nntrainer_causallm libcausallm_core.so libnntrainer.so libccapi-nntrainer.so libnntr_hvx_skel.so && md5sum $C/nntr_config.json"
  A logcat -c
  A shell "cd $D && env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. NNTR_NUM_THREADS=8 $ENVS \
    /system/bin/time -v ./nntrainer_causallm $C" 2>&1
  echo "RC=$?"
  read -r soc bat z0 <<<"$(temps)"
  echo "END soc=$soc bat=$bat zone0=$z0 $(date -Is)"
  echo "== logcat (mha_core / HTP)"
  A logcat -d | grep -aE 'mha_core|HTP|hexkl|nntr_hvx' | tail -200
} > "$LOG" 2>&1
grep -aE "prefill:|generation:|peak memory|nll|ppl|RC=" "$LOG"
