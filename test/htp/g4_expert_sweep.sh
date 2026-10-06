#!/system/bin/sh
# SPDX-License-Identifier: Apache-2.0
#
# Expert-loading sweep for gemma-4-26B-A4B (doc 57 section 6): how fast the
# experts get from flash into the arena, as a function of the knobs that
# already exist, with the run-to-run noise controlled. Runs ON THE DEVICE:
#
#   cd /data/local/tmp/nntrainer/causallm && sh g4_expert_sweep.sh <model dir> [bits] [reps] [rest_s]
#
# e.g. sh g4_expert_sweep.sh models/gemma4-26b-a4b-int2-wh 2 2 60
#
# One line per run: prefill ms, decode TPS, read-ahead exposed wait, reader
# ms/expert, misses and ms/miss, and the CPU temperature before the run.
# Every configuration is run `reps` times, configurations alternating, with
# `rest_s` seconds between runs so heat from one run does not land on the
# next. Logs: /data/local/tmp/g4s/<tag>_<rep>.{out,err}.
set -u
DIR=${1:?model dir}
BITS=${2:-2}
REPS=${3:-2}
REST=${4:-60}
cd /data/local/tmp/nntrainer/causallm || exit 1
L=/data/local/tmp/g4s
mkdir -p $L

temp() { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo -; }

run() { # tag then env assignments
  tag=$1; shift
  echo "=== $tag rep=$REP temp=$(temp) env: $*"
  env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. NNTR_NUM_THREADS=8 \
    NNTR_MOE_EXPERT_BITS=$BITS NNTR_HTP_PROFILE=2 "$@" \
    ./nntrainer_causallm "$DIR" > $L/${tag}_$REP.out 2> $L/${tag}_$REP.err
  echo "exit=$?"
  grep -hE "^prefill:|^generation:" $L/${tag}_$REP.out
  grep -hE "expert prefetch:|prefetch readers|expert cache misses" $L/${tag}_$REP.err | head -3
  sleep $REST
}

# Configurations: the baseline first and last, so drift shows. Edit freely.
CFGS="
base      NNTR_MOE_CACHE_EXPERTS=16
readers2  NNTR_MOE_CACHE_EXPERTS=16 NNTR_MOE_PREFETCH_READERS=2
readers6  NNTR_MOE_CACHE_EXPERTS=16 NNTR_MOE_PREFETCH_READERS=6
readers8  NNTR_MOE_CACHE_EXPERTS=16 NNTR_MOE_PREFETCH_READERS=8
threads6  NNTR_MOE_CACHE_EXPERTS=16 NNTR_NUM_THREADS=6 NNTR_MOE_PREFETCH_READERS=6
depth1    NNTR_MOE_CACHE_EXPERTS=16 NNTR_MOE_PREFETCH=1
c32       NNTR_MOE_CACHE_EXPERTS=32
nopf      NNTR_MOE_CACHE_EXPERTS=16 NNTR_MOE_PREFETCH=0
base_end  NNTR_MOE_CACHE_EXPERTS=16
"
for REP in $(seq 1 $REPS); do
  echo "$CFGS" | while read -r tag envs; do
    [ -z "$tag" ] && continue
    # shellcheck disable=SC2086
    run $tag $envs
  done
done
echo "=== done"
