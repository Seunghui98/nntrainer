#!/usr/bin/env bash
##
# @file    261-softcap-attn-run.sh
# @brief   #261 levers 2 / 3a, sitting 1: the ATTN_M1 phase words on silicon
#          (P = htp_decode @ 4c3953bb1 + the #261 phase-word channel) at the
#          267 sitting's E cells, Gemma-4 26B QS4CX, C 16, S25 R3CY205ZMND
#
# Reuses 260-e-run.sh's machinery (cell, temps, the S1-ceiling gtest, the
# staged r2_E_* configs, prompts 512 / 1024 kept for comparability with the
# 266 / 267 sittings) from R260 (PR #265's branch, default ~/nntrainer-260doc).
#
# usage:
#   261-softcap-attn-run.sh wait <stage dir>   poll until the sitting lock
#                                  is ours, then push the stage dir (jni/...
#                                  + libnntr_hvx_skel.so) to s261p/
#   261-softcap-attn-run.sh one <log dir> <prompt> <G> [suffix] [extra env]
#                                  one P cell (lock held)
#   261-softcap-attn-run.sh release
set -u -o pipefail
R260=${R260:-$HOME/nntrainer-260doc/docs/measurements}
# shellcheck disable=SC1090
source <(sed '/^case "${1:-}" in/,$d' "$R260/260-e-run.sh")
HERE=$R260
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CR=/data/local/tmp/nntrainer/causallm
P=$CR/s261p
export SITTING_OWNER=${SITTING_OWNER:-r261b@$(hostname)}

push_stage() {
  local st=${1:?stage dir} f
  $AD shell "mkdir -p $P && cd $P && cp $CR/s260r2/libc++_shared.so $CR/s260r2/libsdkl.so $CR/s260r2/unittest_hvx_two_sessions ."
  for f in jni/libs/arm64-v8a/nntrainer_causallm jni/libs/arm64-v8a/libcausallm_core.so \
    jni/obj/local/arm64-v8a/libnntrainer.so jni/obj/local/arm64-v8a/libccapi-nntrainer.so \
    libnntr_hvx_skel.so; do
    $AD push "$st/$f" "$P/" >/dev/null
  done
  $AD shell "chmod 755 $P/nntrainer_causallm"
  $AD shell "cd $P && md5sum nntrainer_causallm libcausallm_core.so libnntrainer.so libccapi-nntrainer.so libnntr_hvx_skel.so" | tr -d '\r'
}

cool() { # relaxed (user, 2026-10-09): no other run, then battery <= 32.0 C
  # or a 5 minute cap; the start temperatures are logged
  local i soc bat z0
  while $AD shell 'ps -A' | grep -q 'nntrainer_causall[m]'; do sleep 10; done
  for i in $(seq 30); do
    read -r soc bat z0 <<<"$(temps)"
    [ "$bat" -le 320 ] && break
    sleep 10
  done
  echo "$soc $bat $z0"
}

wait_lock() {
  until (cd "$ROOT" && tools/htp/sitting_lock.sh take $SER "261 attn phase words" 60); do
    sleep 60
  done
  push_stage "${1:?stage dir}"
}

one() { # one <log dir> <prompt> <G> [suffix] [extra env]
  L=${1:?log dir}
  mkdir -p "$L/done"
  CLADDER=16 cell "$P" "$L" E "$2" "$3" "P${4:+_$4}" "${5:-}"
}

case "${1:-}" in
wait) shift; wait_lock "$@" ;;
one) shift; one "$@" ;;
release) (cd "$ROOT" && tools/htp/sitting_lock.sh release $SER) ;;
*) sed -n '2,20p' "$0"; exit 1 ;;
esac
