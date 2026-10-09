#!/usr/bin/env bash
##
# @file    282-run.sh
# @brief   #282 sitting: the 2-bit Gemma-4 26B file with the experts pinned
#          in RAM (NNTR_MOE_PIN), the C ladder, the miss-cost and kernel
#          steps, and the prompt x G grid; S25 R3CY205ZMND
#
# One cell per call, so every number can be relayed as it lands. Cells,
# cool start (COOL_QUICK: battery <= 32 C or 5 min), the S1-ceiling column
# and the logs come from 260-e-run.sh (sourced); configs are 276-run.sh's
# (q8 KV: s276q8/, the grid's prompt + G rule: s282cfg/).
#
# usage:
#   282-run.sh one <log dir> <bin dir> <cfg dir> <E|A> <prompt> <G> <C> <tag> [env...]
#                 one cell, named <E|A>_p<prompt>_g<G>_<tag>
#   282-run.sh sum <log dir>     the 260-e-run.sh table
set -u -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=260-e-run.sh
source "$HERE/260-e-run.sh"
DM=/data/local/tmp/nntrainer/gemma4_26b_ternary_fcqs4cx
export COOL_QUICK=1

case "${1:-}" in
one)
  L=$2 b=$3 DC=$4 v=$5 p=$6 g=$7 c=$8 tag=$9
  shift 9
  mkdir -p "$L/done"
  CLADDER=$c cell "$b" "$L" "$v" "$p" "$g" "$tag" "$*" ;;
sum) TOK=$2/model/tokenizer.json; sum "$2" ;;
*) sed -n 2,20p "$0"; exit 1 ;;
esac
