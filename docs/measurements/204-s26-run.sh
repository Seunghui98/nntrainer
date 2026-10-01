#!/usr/bin/env bash
##
# @file    204-s26-run.sh
# @brief   Plan 204 step 7: the S26 (v81) re-baseline sitting on htp_decode,
#          A / E0 / P28 / Q28 at NNTR_MOE_DMA_QUEUES=4 and unset (= 1)
#          (handoff docs/measurements/204-s26-rebaseline.md)
#
# Staged by 204-s26-stage.sh as run_s26.sh; run from the stage directory's copy:
#   bash /local/mnt/workspace/htp_moe/204/s26/run_s26.sh <serial> [pool C]
# The serial is required (no default: the old runners defaulted to the S25).
# pool C (default 28) names the pool size of P / Q; use 24 when Q28 cannot
# load (fastrpc_mmap on the S26's PD space, plan 204 §5) and say so.
# Variants (one binary set, env only; the queue count is a column):
#   A     hybrid, nothing set (the unchanged reference)
#   E0    NNTR_HTP_E2E=1 (two PDs, all experts resident)
#   P<C>  E0 + NNTR_MOE_CACHE_EXPERTS=<C> (two PDs, the model file pre-read)
#   Q<C>  P<C> + NNTR_HTP_E2E_PDS=1 (one PD: the FC set and the pool in S1)
#   CPU   the q40 model, MoE on the CPU (text control, read, not gated)
# Order: the five device gtests (G4), the ceiling, then
#   G 64 q4: cool, A E0 P Q, cool, Q P E0 A
#   G 64 q1: cool, A E0 P Q, cool, Q P E0 A
#   G 512 q4: cool, A E0 P Q
#   G 64 NNTR_HTP_PROFILE=2: Q at q4 and q1 (dmaq=, DMA_FIRST); CPU G 64.
# S1's ceiling after every run must not fall below the start's. RESUMABLE
# per run (logs/done/<run>); after any STOP reboot the phone and run the
# same command again.
set -u -o pipefail
S=${1:?usage: run_s26.sh <serial> [pool C]  (the serial of the S26 from adb devices; no default)}
PC=${2:-28}
W=$(cd "$(dirname "$0")" && pwd); L=$W/logs; mkdir -p $L/done
C=/data/local/tmp/nntrainer/causallm; D=$C/s204; M=../models/q40-qs4cx-wh; MC=../models/q40
MF=$M/nntr_lfm2_8b_a1b_q40_arm.bin
AD="adb -s $S"
Q4="NNTR_MOE_DMA_QUEUES=4"
exec > >(tee -a $L/sitting.out) 2>&1
MIS=0
stop() { echo "STOP: $*"; echo "  -> reboot the phone and run the same command again: it resumes at the first unfinished run"; exit 1; }
want() { if [ "$2" = "$3" ]; then echo "OK  $1 = $2"; else echo "BAD $1: got '$2' want '$3'"; MIS=$((MIS + 1)); fi; }
therm() { echo "$1 $(date +%H:%M:%S) $($AD shell 'dumpsys battery | grep -E "^  (level|temperature)"; cat /sys/class/thermal/thermal_zone0/temp' | tr -d '\r' | tr '\n' ' ')" | tee -a $L/therm.log; }
zone0() { $AD shell cat /sys/class/thermal/thermal_zone0/temp | tr -d '\r'; }
cool() { local i z; for i in $(seq 1 40); do z=$(zone0); [ "$z" -le 35000 ] && break
    echo "zone0=$z > 35000, waiting 30 s ($i/40)"; sleep 30; done; echo "block start zone0=$(zone0)"; }
ceil() {
  $AD shell "cd $D && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_two_sessions \
    --gtest_filter=TwoSessions.S1Ceiling" > $L/ceil_$1.log 2>&1
  local c; c=$(grep -o 'CEILING s1_mmap_mib=[0-9]*' $L/ceil_$1.log | cut -d= -f2)
  echo "ceiling after $1: ${c:-?} MiB" | tee -a $L/ceiling.txt
  [ "$1" = start ] && { echo "${c:-0}" > $L/ceil0; return 0; }
  [ "${c:-0}" -ge "$(cat $L/ceil0)" ] || stop "LEAK: S1 ceiling ${c:-?} MiB after $1, $(cat $L/ceil0) at start"; }
strip() { sed -n '/^=====/q;p' "$1" | perl -0pe 's/\[HTP[^\]\n]*\] [^\n]*\n//g; s/\[PPL\] [^\n]*\n//g' |
  grep -v 'moe m1 gemv\|libnntr_hvx_skel\|nntrainer_causallm\|num_to_generate\|^resident [0-9]* MiB'; }
run() { # run <A|E0|P<C>|Q<C>|CPU> <G> <q 4|1> <log> [env ...]
  local v=$1 g=$2 q=$3 log=$4 e="" pre="" m=$M; shift 4
  [ -f $L/done/$log ] && { echo "$log: done earlier, skipped"; return 0; }
  case $v in
    A) ;;
    CPU) m=$MC ;;
    E0) e="NNTR_HTP_E2E=1" ;;
    P*) e="NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=${v#P}"
        pre="cat $MF > /dev/null && ./page_cache_evict $MF -1 &&" ;;
    Q*) e="NNTR_HTP_E2E=1 NNTR_HTP_E2E_PDS=1 NNTR_MOE_CACHE_EXPERTS=${v#Q}"
        pre="cat $MF > /dev/null && ./page_cache_evict $MF -1 &&" ;;
  esac
  [ "$q" = 4 ] && e="$e $Q4"
  $AD logcat -c
  $AD shell "cd $D && sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $g/' $m/nntr_config.json && \
    md5sum libnntr_hvx_skel.so && $pre $e $* NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm $m \"\$(cat prompt512.txt)\"" > $L/$log.log 2>&1
  $AD logcat -d | grep -iE 'adsprpc|fastrpc|nntr_hvx|token' > $L/$log.logcat || true
  grep -qi '0x8000040e' $L/$log.log && stop "$log: 0x8000040e (stale skel or stub)"
  grep -q FATAL $L/$log.log && stop "$log: $(grep -m1 FATAL $L/$log.log | cut -c1-200)"
  grep -q 'token driver: token .* failed\|token driver: .* transport failed' $L/$log.log && stop "$log: a token failed ($(grep -m1 'token driver: token' $L/$log.log | cut -c1-160))"
  grep -q '^generation:' $L/$log.log || stop "$log cannot generate (a LEAK, or Q$PC does not load: rerun with pool C 24; logcat in $log.logcat)"
  echo "$log: $(grep -h -E '^(prefill|generation|generation\(last 64\)):' $L/$log.log | grep -o '[0-9.]* TPS' | tr '\n' ' ')$(grep -h -o 'calls/token=[0-9.]*' $L/$log.log) $(grep -h -o 'resident [0-9]* MiB' $L/$log.log)"
  if [ $v = CPU ]; then
    want "$log no HTP banner" "$(grep -c 'moe m1 gemv\|dspq: on' $L/$log.log)" 0
  else
    local ap=0x703e1; [ "$q" = 4 ] && ap=0x1f03e1
    want "$log queues banner" "$(grep -c "moe m1 gemv: on (applied=$ap) .* dma_q=$q " $L/$log.log)" 1
  fi
  case $v in
    A)
      want "$log hybrid banners (dspq on, no s2)" "$(grep -c '\[HTP\] dspq: on' $L/$log.log)/$(grep -c 's2: open' $L/$log.log)" 1/0
      want "$log dspq bad=0" "$(grep -c 'dspq: close .* bad=0' $L/$log.log)" 1 ;;
    CPU) ;;
    *)
      want "$log levers banner" "$(grep -c '^\[HTP\] ppl levers=0x0 L1=exact$' $L/$log.log)" 1
      want "$log s2 close: no unmap / detach refused" "$(grep -c 's2: close .* unmap_fail=0 detach_fail=0 ' $L/$log.log)" 1
      local hops=44.00; [ "${v#Q}" != "$v" ] && hops=0.00
      want "$log close clean" "$(grep -c "token driver: close tokens=[0-9]* hops/token=$hops .* timeouts=0/0 stale=0/0 .* id_mismatch=0 " $L/$log.log)" 1
      [ "${v#Q}" != "$v" ] && want "$log one PD" "$(grep -c 'token driver: on .* pds=1$' $L/$log.log)" 1
      want "$log calls/token=1.00" "$(grep -cF 'calls/token=1.00' $L/$log.log)" 1
      [ $v = E0 ] || grep -h 'token driver: pool\|token driver: L0 us/token' $L/$log.log | sed 's/^/    /' ;;
  esac
  touch $L/done/$log
  [ $v = CPU ] || ceil $log; }

echo "=== 204 S26 $(date '+%F %T %Z') unit=$S pool C=$PC (done: $(ls $L/done | wc -l) runs)"
$AD get-state > /dev/null 2>&1 || stop "unit $S not attached"
echo "model: $($AD shell getprop ro.product.model | tr -d '\r') soc: $($AD shell getprop ro.soc.model | tr -d '\r')"
echo "uptime (reboot first): $($AD shell cat /proc/uptime | tr -d '\r')"
$AD shell input keyevent 223 || true
therm t0
[ "$(cd $W && LC_ALL=C md5sum -c md5.txt | grep -vc ': OK$')" = 0 ] || stop "workstation set differs from md5.txt"
$AD shell "mkdir -p $D" && $AD push $W/app/. $D/ > /dev/null
$AD shell "chmod 755 $D/nntrainer_causallm $D/unittest_* $D/page_cache_evict"
$AD shell "cd $D && md5sum $(grep -o '  app/.*' $W/md5.txt | sed 's|  app/||' | tr '\n' ' ')" | tr -d '\r' | awk '{print $1"  app/"$2}' > $L/md5_device.log
diff <(grep -E '^[0-9a-f]{32}  app/' $W/md5.txt | sort -k2) <(sort -k2 $L/md5_device.log) && echo "MD5 OK" || stop "device md5 differs"
$AD shell "md5sum $C/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin $C/models/q40/nntr_lfm2_8b_a1b_q40_arm.bin" | tr -d '\r' | tee $L/md5_models.log
for m in $M $MC; do
  $AD shell "cd $D/$m && sed -i 's/\"do_sample\": true/\"do_sample\": false/' generation_config.json && \
    sed -i 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' nntr_config.json && \
    grep -H do_sample generation_config.json && grep -H -E 'bad_word_ids|init_seq_len|moe_engine' nntr_config.json" | tee -a $L/config.log
done
$AD shell "cd $D/$M && (grep -q moe_engine nntr_config.json || sed -i 's/\"bad_word_ids\": \[124900\],/\"bad_word_ids\": [124900],\n    \"moe_engine\": \"htp\",/' nntr_config.json) && grep -H moe_engine nntr_config.json" | tee -a $L/config.log

echo "--- G4: device gtests on the v81 skel"
for t in unittest_hvx_mm_u8i4 unittest_hvx_softmax unittest_hvx_attn unittest_hvx_fc unittest_hvx_two_sessions; do
  [ -f $L/done/gt_$t ] && { echo "$t: done earlier, skipped"; continue; }
  $AD shell "cd $D && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./$t" > $L/gt_$t.log 2>&1
  grep -qi '0x8000040e' $L/gt_$t.log && stop "$t: 0x8000040e (stale skel or stub)"
  want "$t passed / failed" "$(grep -c '^\[  PASSED  \]' $L/gt_$t.log)/$(grep -c '^\[  FAILED  \]' $L/gt_$t.log)" 1/0
  grep -h '^\[  PASSED  \]\|^\[  FAILED  \]\|^\[  SKIPPED \]' $L/gt_$t.log | sed 's/^/    /'
  touch $L/done/gt_$t
done
[ -f $L/done/ceil_start ] || { ceil start; touch $L/done/ceil_start; }

for q in 4 1; do
  echo "--- G=64 q$q $(date +%H:%M:%S)"; cool
  for v in A E0 P$PC Q$PC; do run $v 64 $q ${v}_G64_q${q}_r1; done
  cool
  for v in Q$PC P$PC E0 A; do run $v 64 $q ${v}_G64_q${q}_r2; done
  therm t_G64_q$q
done
echo "--- G=512 q4 $(date +%H:%M:%S)"; cool
for v in A E0 P$PC Q$PC; do run $v 512 4 ${v}_G512_q4_r1; done
therm t_G512
echo "--- G=64 profiles (dmaq=, DMA_FIRST) and the CPU control"; cool
for q in 4 1; do run Q$PC 64 $q prof_Q${PC}_q$q NNTR_HTP_PROFILE=2; done
run CPU 64 1 CPU_G64
ceil end
therm t_end

echo "--- speed (prefill / decode / last 64 TPS; text vs A q4 r1 of the same G)"
for f in $L/[AEPQ]*_G*_q[14]_r[12].log $L/CPU_G64.log; do b=$(basename $f .log); g=${b#*_G}; g=${g%%_*}
  t=$(cmp -s <(strip $L/A_G${g}_q4_r1.log) <(strip $f) && echo same || echo DIFF)
  echo "$b: $(grep -h -E '^(prefill|generation|generation\(last 64\)):' $f | grep -o '[0-9.]* TPS' | tr '\n' ' ')text=$t"
done | tee $L/speed.txt
want "every NPU text == A q4 r1 of its G (E0, P, Q and both queue counts are A's bits)" "$(grep -v '^CPU' $L/speed.txt | grep -c 'text=DIFF')" 0
echo "--- profiles: feed / dmaq= and the first DMA wait"
for f in $L/prof_*.log; do echo "$(basename $f .log):"; grep -h -E 'dmaq=|DMA_FIRST|dma first' $f | sed 's/^/  /'; done | tee $L/prof.txt
echo "--- pool lines"
for f in $L/[PQ]*_G*.log; do echo "$(basename $f .log):"; grep -h 'token driver: pool\|L0 us/token\|expert cache misses\|resident [0-9]* MiB' $f | sed 's/^/  /'; done | tee $L/pool.txt
echo "--- ceiling"; cat $L/ceiling.txt
echo "--- therm"; cat $L/therm.log
echo "=== done $(date '+%F %T %Z') expectation mismatches (this invocation): $MIS"
