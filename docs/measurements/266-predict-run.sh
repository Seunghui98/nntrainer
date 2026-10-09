#!/usr/bin/env bash
##
# @file    266-predict-run.sh
# @brief   #266 S2 / S3 cells: the next-layer route prediction (lever 3) on
#          the one-PD decode, Gemma-4 26B QS4CX file, S25 R3CY205ZMND
#
# Plan: docs/plans/266-flash-miss-path.md section 4 S2 / S3. Every cell is E
# (NNTR_HTP_E2E=1, C = 16, the 260 r2 sitting's r2_E_* configs, prompts 512
# / 1024 as in the S1 sitting so before / after compare); the install is
# s266p/ (this branch). Cells, cool start, logs and the S1-ceiling column
# come from 260-e-run.sh (sourced); COOL_QUICK (battery <= 32 C or 5 min).
#
# usage:
#   266-predict-run.sh <cell> <log dir>   one cell, takes and releases the
#                                         lock (SITTING_OWNER r266p@host)
#     s2      E p512 G512, NNTR_HTP_PREDICT=1, route log with the guesses
#     s2_1024 the same at p1024 G512
#     s2_ppl  E p512 G64 NNTR_PPL=1 with PREDICT on: the nll gate
#   266-predict-run.sh gate <log dir>     266-run.sh's table (REF=<260 r2
#                                         log dir>)
set -u -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=260-e-run.sh
source "$HERE/260-e-run.sh"
BP=/data/local/tmp/nntrainer/causallm/s266p
RL=/data/local/tmp/nntrainer/r266p
export SITTING_OWNER=${SITTING_OWNER:-r266p@$(hostname)}

one() {
  local what=${1:?cell} L=${2:?log dir}
  mkdir -p "$L/done"
  exec > >(tee -a "$L/sweep.out") 2>&1
  (cd "$ROOT" && tools/htp/sitting_lock.sh take $SER "266 $what" 20) || exit 1
  trap '(cd "$ROOT" && tools/htp/sitting_lock.sh release $SER)' EXIT
  export CLADDER=16 COOL_QUICK=1
  $AD shell mkdir -p $RL
  echo "=== 266 $what $(date '+%F %T %Z')"
  case "$what" in
  s2) cell $BP "$L" E 512 512 S2 \
    "NNTR_HTP_PREDICT=1 NNTR_HTP_ROUTE_LOG=$RL/pred_p512_g512.txt"
    $AD pull $RL/pred_p512_g512.txt "$L/" ;;
  s2_1024) cell $BP "$L" E 1024 512 S2 \
    "NNTR_HTP_PREDICT=1 NNTR_HTP_ROUTE_LOG=$RL/pred_p1024_g512.txt"
    $AD pull $RL/pred_p1024_g512.txt "$L/" ;;
  s2_ppl) cell $BP "$L" E 512 64 S2_ppl "NNTR_PPL=1 NNTR_HTP_PREDICT=1" ;;
  *) echo "unknown cell $what"; exit 1 ;;
  esac
  echo "=== done $(date '+%F %T %Z')"
}

case "${1:-}" in
gate) shift; "$HERE/266-run.sh" gate "$@" ;;
s2 | s2_1024 | s2_ppl) one "$@" ;;
*) sed -n '2,20p' "$0"; exit 1 ;;
esac
