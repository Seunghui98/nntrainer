#!/system/bin/sh
# SPDX-License-Identifier: Apache-2.0
#
# gemma-4-26B-A4B, int4 vs int2 expert weights, read-ahead off vs on
# (doc 57 section 4). Runs ON THE DEVICE, from the app directory:
#
#   cd /data/local/tmp/nntrainer/causallm && sh g4_int2_bench.sh [setup|quick]
#
# Needs two model directories made by doc 57 section 3:
#   models/g4-i4same  the int2 VALUES in the WH layout  (NNTR_MOE_EXPERT_BITS=4)
#   models/g4-i2      the same values as WH2            (NNTR_MOE_EXPERT_BITS=2)
# They are one model after the DSP's expansion, so every int4/int2 pair
# below must print the same text; the script compares it.
#
# Each configuration runs twice: "cold" drops the page cache first when
# that is allowed (root), and says so either way; "warm" follows at once.
# Logs: /data/local/tmp/g4b/<name>_<pass>.{out,err}. "quick" skips the
# cold passes and the C=32 run.
set -u
cd /data/local/tmp/nntrainer/causallm || exit 1
L=/data/local/tmp/g4b
mkdir -p $L
QUICK=${1:-}

# "setup": the two model directories, every file of the existing int4 one
# but its .bin, with model_file_name pointed at the dummy file and 128
# tokens to generate. The .bin files are pushed into them afterwards.
if [ "$QUICK" = setup ]; then
  BASE=models/gemma4-26b-a4b-qs4cx-wh
  for pair in "g4-i2 nntr_gemma4_wh2.bin" "g4-i4same nntr_gemma4_i4same.bin"; do
    set -- $pair
    mkdir -p models/$1
    for f in $BASE/*; do
      case $f in *.bin) ;; *) cp "$f" models/$1/ ;; esac
    done
    sed -i -e "s/\"model_file_name\": *\"[^\"]*\"/\"model_file_name\": \"$2\"/" \
      -e "s/\"num_to_generate\": *[0-9]*/\"num_to_generate\": 128/" \
      models/$1/nntr_config.json
    echo "=== models/$1: $(grep -oE '"model_file_name": *"[^"]*"|"num_to_generate": *[0-9]+' models/$1/nntr_config.json | tr '\n' ' ')"
  done
  df -h /data | tail -1
  exit 0
fi

text_md5() { # the generated text: after the prompt, before the summary
  awk 'index($0,"=================[ LLM"){exit} f{print} index($0,"<|turn>model"){f=1}' "$1" |
    md5sum | cut -c1-8
}

run() { # name dir bits C prefetch(on|off) profile pass
  name=$1 dir=$2 bits=$3 c=$4 pf=$5 prof=$6 pass=$7
  dc=-
  if [ "$pass" = cold ]; then
    sync
    if echo 3 > /proc/sys/vm/drop_caches 2>/dev/null; then dc=dropped; else dc=NOT_dropped; fi
  fi
  if [ "$pf" = off ]; then PFV=0; else PFV=; fi
  echo "=== $name pass=$pass bits=$bits C=$c prefetch=$pf profile=$prof pagecache=$dc"
  env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. NNTR_NUM_THREADS=8 \
    NNTR_MOE_EXPERT_BITS=$bits NNTR_MOE_CACHE_EXPERTS=$c \
    ${PFV:+NNTR_MOE_PREFETCH=$PFV} NNTR_HTP_PROFILE=$prof \
    ./nntrainer_causallm "$dir" > $L/${name}_$pass.out 2> $L/${name}_$pass.err
  echo "exit=$? text_md5=$(text_md5 $L/${name}_$pass.out)"
  grep -hE "^prefill:|^generation:|^peak memory|arena chunk [0-9]+:" $L/${name}_$pass.out | tail -4
  grep -hE "expert cache misses|expert prefetch:|expert prefetch progress|weight DMA|int2 expand|FATAL|rror" \
    $L/${name}_$pass.err $L/${name}_$pass.out | head -8
}

for cfg in \
  "i4_pf  models/g4-i4same 4 16 on" \
  "i2_pf  models/g4-i2     2 16 on" \
  "i4_nopf models/g4-i4same 4 16 off" \
  "i2_nopf models/g4-i2    2 16 off" \
  "i2_c32 models/g4-i2     2 32 on"; do
  set -- $cfg
  [ -n "$QUICK" ] && [ "$1" = i2_c32 ] && continue
  [ -z "$QUICK" ] && run $1 $2 $3 $4 $5 1 cold
  run $1 $2 $3 $4 $5 1 warm
done
# The per-stage breakdown, once each (level 2 times every DSP stage).
run i4_prof models/g4-i4same 4 16 on 2 warm
grep -hE "K=2816 N=2816|dsp=" $L/i4_prof_warm.err | head -6
run i2_prof models/g4-i2 2 16 on 2 warm
grep -hE "K=2816 N=2816|dsp=" $L/i2_prof_warm.err | head -6
echo "=== same text int4/int2 (pf, nopf):" \
  "$(text_md5 $L/i4_pf_warm.out) $(text_md5 $L/i2_pf_warm.out)" \
  "$(text_md5 $L/i4_nopf_warm.out) $(text_md5 $L/i2_nopf_warm.out)"
echo "=== done"
