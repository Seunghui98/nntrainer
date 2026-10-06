#!/system/bin/sh
# SPDX-License-Identifier: Apache-2.0
#
# int4 vs int2 on the same device, the numbers doc 57 section 7 needs to
# split the expert path into hidden and exposed parts: flash read (reader
# ms/expert, exposed read-ahead wait, miss ms/miss), DDR->VTCM DMA (KB/call,
# drain = exposed wait on a weight), expansion (worker us). Runs ON THE
# DEVICE:
#
#   cd /data/local/tmp/nntrainer/causallm && \
#     sh g4_io_dma_probe.sh <int4 model dir> <int2 model dir> [rest_s]
#
# Each of int4/int2 x read-ahead on/off runs twice; the first run of each
# pair warms the page cache and CPU clocks and is printed but marked warmup.
# One summary line per run at the end (also /data/local/tmp/g4p/summary.txt).
set -u
D4=${1:?int4 model dir}
D2=${2:?int2 model dir}
REST=${3:-45}
cd /data/local/tmp/nntrainer/causallm || exit 1
L=/data/local/tmp/g4p
mkdir -p $L
: > $L/summary.txt


run() { # tag bits dir prefetch(on|off) rep
  tag=$1 bits=$2 dir=$3 pf=$4 rep=$5
  if [ "$pf" = off ]; then PFV=0; else PFV=; fi
  o=$L/${tag}_$rep.out; e=$L/${tag}_$rep.err
  echo "=== $tag rep=$rep temp=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo -)"
  env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. NNTR_NUM_THREADS=8 \
    NNTR_MOE_EXPERT_BITS=$bits NNTR_MOE_CACHE_EXPERTS=16 \
    ${PFV:+NNTR_MOE_PREFETCH=$PFV} NNTR_HTP_PROFILE=2 \
    ./nntrainer_causallm "$dir" > $o 2> $e
  rc=$?
  pre=$(grep -oE 'prefill: [0-9]+ tokens, [0-9]+ ms' $o | grep -oE '[0-9]+ ms' | grep -oE '[0-9]+')
  gen=$(grep -oE 'generation: [0-9]+ tokens, [0-9]+ ms, [0-9.]+ TPS' $o | grep -oE '[0-9.]+ TPS' | grep -oE '[0-9.]+')
  # prefill (M>1) and decode (M==1) layer-call lines
  p_line=$(grep -E 'K=2816 +N=2816 +M>1' $e | head -1)
  d_line=$(grep -E 'K=2816 +N=2816 +M==1' $e | head -1)
  p_dsp=$(echo "$p_line" | grep -oE 'dsp=[ 0-9.]+' | grep -oE '[0-9.]+')
  p_drain=$(echo "$p_line" | grep -oE 'drain [0-9.]+\+[0-9.]+' | cut -d' ' -f2)
  p_mm=$(echo "$p_line" | grep -oE 'mm [0-9.]+' | head -1 | cut -d' ' -f2)
  d_dsp=$(echo "$d_line" | grep -oE 'dsp=[ 0-9.]+' | grep -oE '[0-9.]+')
  d_drain=$(echo "$d_line" | grep -oE 'drain [0-9.]+\+[0-9.]+' | cut -d' ' -f2)
  d_mm=$(echo "$d_line" | grep -oE 'mm [0-9.]+' | head -1 | cut -d' ' -f2)
  dma_p=$(grep -E 'weight DMA' $e | head -1 | grep -oE '[0-9]+ KB/call' | grep -oE '[0-9]+')
  dma_d=$(grep -E 'weight DMA' $e | sed -n 2p | grep -oE '[0-9]+ KB/call' | grep -oE '[0-9]+')
  xp_p=$(grep -E 'weight DMA' $e | head -1 | grep -oE 'expand \(worker\) [0-9.]+' | grep -oE '[0-9.]+$')
  xp_d=$(grep -E 'weight DMA' $e | sed -n 2p | grep -oE 'expand \(worker\) [0-9.]+' | grep -oE '[0-9.]+$')
  miss=$(grep -E 'expert cache misses:' $e | head -1 | grep -oE 'misses: [0-9]+' | grep -oE '[0-9]+')
  mpm=$(grep -E 'expert cache misses:' $e | head -1 | grep -oE '\([0-9.]+ ms/miss\)' | head -1 | grep -oE '[0-9.]+')
  pfn=$(grep -E 'expert prefetch:' $e | head -1 | grep -oE '[0-9]+ experts' | grep -oE '[0-9]+')
  pfw=$(grep -E 'expert prefetch:' $e | head -1 | grep -oE 'exposed wait [0-9.]+' | grep -oE '[0-9.]+')
  rdr=$(grep -E 'prefetch readers:' $e | head -1 | grep -oE '[0-9.]+ ms/expert' | grep -oE '[0-9.]+')
  s="$tag rep=$rep rc=$rc prefill_ms=${pre:--} decode_tps=${gen:--} | read: prefetch_n=${pfn:--} exposed_ms=${pfw:--} reader_ms=${rdr:--} miss=${miss:--} ms/miss=${mpm:--} | prefill call: dsp=${p_dsp:--} drain=${p_drain:--} mm=${p_mm:--} dma_kb=${dma_p:--} expand=${xp_p:--} | decode call: dsp=${d_dsp:--} drain=${d_drain:--} mm=${d_mm:--} dma_kb=${dma_d:--} expand=${xp_d:--}"
  [ "$rep" = 1 ] && s="$s (warmup)"
  echo "$s" | tee -a $L/summary.txt
  sleep $REST
}

for rep in 1 2; do
  run i4_pf    4 "$D4" on  $rep
  run i2_pf    2 "$D2" on  $rep
  run i4_nopf  4 "$D4" off $rep
  run i2_nopf  2 "$D2" off $rep
done
echo "=== summary"
cat $L/summary.txt
