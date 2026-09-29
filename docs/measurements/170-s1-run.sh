#!/usr/bin/env bash
# Plan 170 step 2, sitting S1 (docs/measurements/170-attn-m1-hf.md on
# htp/170-attn-fast): does the qf32 multiply-add narrowed once to hf equal
# the CPU's fused fp16 FMA on silicon (G1), what a 64-lane FMA costs per
# lane count, the cold fetch rate, today's per-L attention line and its
# in-model pcyc/op. Two run dirs so no stub meets a foreign skel:
#   s170p  this branch's skel + gtests (the probe; the kernel is today's)
#   s170a  the unchanged #164 set, for the full-model cells (A first)
# Usage: bash /local/mnt/workspace/htp_moe/170/s1/run_s1.sh [serial]   (~30 min)
# Everything is logged to $L, the summary to $L/sitting.out. Stops on the
# plan's stop rules (a qfma bad != 0, 0x8000040e); every other missing
# expected line is counted and printed.
set -u -o pipefail
WT=/home/j2z0-lee/nntrainer-170            # htp/170-attn-fast (env.sh only)
S=${1:-R3CY10WM83Y}
W=/local/mnt/workspace/htp_moe/170/s1; L=$W/logs; mkdir -p $L
C=/data/local/tmp/nntrainer/causallm; DP=$C/s170p; DA=$C/s170a; M=../models/q40-qs4cx-wh
cd $WT && { set +u; source tools/htp/env.sh > /dev/null 2>&1; set -u; }
exec > >(tee -a $L/sitting.out) 2>&1
MIS=0
stop() { echo "STOP: $*"; therm stop; exit 1; }
want() { # want <label> <got> <expected>
  if [ "$2" = "$3" ]; then echo "OK  $1 = $2"; else echo "BAD $1: got '$2' want '$3'"; MIS=$((MIS + 1)); fi; }
therm() { echo "$1 $(date +%H:%M:%S) $(adb -s $S shell 'dumpsys battery | grep -E "^  (level|temperature)"; cat /sys/class/thermal/thermal_zone0/temp' | tr -d '\r' | tr '\n' ' ')" | tee -a $L/therm.log; }
zone0() { adb -s $S shell cat /sys/class/thermal/thermal_zone0/temp | tr -d '\r'; }
stale() { grep -qi '0x8000040e' "$1" && stop "$(basename $1): 0x8000040e (stale skel, rule 3)"; true; }
gt() { # gt <log> <binary> <filter> [env ...]: one gtest run in s170p
  local log=$1 bin=$2 f=$3; shift 3
  adb -s $S shell "cd $DP && md5sum libnntr_hvx_skel.so $bin && $* LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./$bin --gtest_filter='$f'" > $L/$log.log 2>&1
  stale $L/$log.log
  grep -h -E '^\[  (PASSED|FAILED) |^\[  SKIPPED' $L/$log.log | sed "s/^/$log: /"
}
F="NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=MOE,QK_NORM,ROPE,ATTN_M1"
run() { # run <variant A|Q> <G> <log> [extra env ...]: nntrainer_causallm in s170a, prompt 512
  local v=$1 g=$2 log=$3; shift 3
  local e=""; [ "$v" = Q ] && e="$F"
  adb -s $S shell "cd $DA && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $g/' $M/nntr_config.json && \
    grep num_to_generate $M/nntr_config.json && md5sum libnntr_hvx_skel.so && \
    $e $* NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm $M \"\$(cat prompt512.txt)\"" > $L/$log.log 2>&1
  stale $L/$log.log
  echo "$log: $(grep -h -E '^(prefill|generation|generation\(last 64\)):' $L/$log.log | grep -o '[0-9.]* TPS' | tr '\n' ' ')$(grep -h -o 'calls/token=[0-9.]*' $L/$log.log)"
}
vbanner() { # vbanner <A|Q> <log>: the switch-on banner / A's dspq pair, else void (rule 36)
  local f=$L/$2.log
  if [ "$1" = A ]; then
    want "$2 dspq: on" "$(grep -c 'dspq: on' $f)" 1
    want "$2 dspq close bad=0" "$(grep -c 'dspq: close calls=\([0-9]*\) served=\1 bad=0' $f)" 1
    want "$2 no graph" "$(grep -c 'graph: init' $f)" 0
  else
    want "$2 banner" "$(grep -cF 'graph: init n_ops=228 resident=QK_NORM|ROPE|ATTN_M1|MOE moe_ops=22' $f)" 1
    want "$2 calls/token=28.00" "$(grep -cF 'calls/token=28.00' $f)" 1
  fi; }
strip() { sed -n '/^=====/q;p' "$1" | perl -0pe 's/\[HTP\] [^\n]*\n//g; s/\[PPL\] [^\n]*\n//g' |
  grep -v 'moe m1 gemv\|libnntr_hvx_skel\|nntrainer_causallm\|num_to_generate'; }

echo "=== 170 S1 sitting  $(date '+%F %T %Z')  unit=$S  wt=$(git -C $WT rev-parse --short HEAD)"
# ---- 0. device state, cool start (1 min)
adb devices | tee $L/devices.log
adb devices | grep -q "^$S[[:space:]]*device" || stop "unit $S not attached"
adb -s $S shell input keyevent 223 || true
therm t0
for i in $(seq 1 40); do z=$(zone0); [ "$z" -le 35000 ] && break
  echo "zone0=$z > 35000, waiting 30 s ($i/40)"; sleep 30; done
echo "start zone0=$(zone0)"

# ---- 1. install both run dirs and the model config (3 min)
[ "$(cd $W && LC_ALL=C md5sum -c md5.txt | grep -vc ': OK$')" = 0 ] || stop "workstation sets differ from md5.txt"
adb -s $S shell ls -l $C/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin
adb -s $S shell "rm -rf $DP $DA && mkdir -p $DP $DA"
adb -s $S push $W/s170p/. $DP/ > /dev/null && adb -s $S push $W/s170a/. $DA/ > /dev/null
adb -s $S shell "chmod 755 $DP/unittest_* $DA/nntrainer_causallm"
for d in s170p s170a; do
  adb -s $S shell "cd $C/$d && md5sum \$(ls -p | grep -v / | sort)" | tr -d '\r' | awk -v d=$d '{print $1"  "d"/"$2}'
done > $L/md5_device.log
diff <(sort -k2 $W/md5.txt) <(sort -k2 $L/md5_device.log) && echo "MD5 OK" || stop "device md5 differs from md5.txt"
adb -s $S shell "cd $DA/$M && \
  sed -i 's/\"do_sample\": true/\"do_sample\": false/' generation_config.json && \
  sed -i 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' nntr_config.json && \
  (grep -q moe_engine nntr_config.json || sed -i 's/\"bad_word_ids\": \[124900\],/\"bad_word_ids\": [124900],\n    \"moe_engine\": \"htp\",/' nntr_config.json) && \
  grep -H do_sample generation_config.json && grep -H -E 'bad_word_ids|num_to_generate|init_seq_len|_engine|_htp_layers' nntr_config.json" \
  | tee $L/config.log
for k in '"do_sample": false' '"bad_word_ids": \[124900\]' '"init_seq_len": 512' '"moe_engine": "htp"' '"moe_htp_layers": ""'; do
  want "config $k" "$(grep -c "$k" $L/config.log)" 1; done
therm t1

# ---- 2. (a) G1: hvx_attn_m1_hf.h on silicon == attn_m1_det.h (2 min)
gt g1_semantics unittest_hvx_attn 'HvxAttnM1Probe.Semantics' NNTR_ATTN_FMA_CASES=$DP/attn_fma_cases.bin
grep -h '^ATTN_M1_PROBE ' $L/g1_semantics.log | tee $L/g1.txt
while read -r line; do want "G1 ${line% bad=0}" "$(grep -cxF "$line" $L/g1.txt)" 1; done <<'EOF'
ATTN_M1_PROBE qfma cases n=4878 hazards=333 subnormal=81 bad=0
ATTN_M1_PROBE qfma adversarial n=76800 hazards=8382 subnormal=2300 bad=0
ATTN_M1_PROBE qfma zero_sign n=25600 hazards=0 subnormal=25600 bad=0
ATTN_M1_PROBE hf add n=253568 bad=0
ATTN_M1_PROBE hf sub n=253560 bad=0
ATTN_M1_PROBE hf mul n=228348 bad=0
ATTN_M1_PROBE hf max n=253952 bad=0
ATTN_M1_PROBE hf mul0.125 n=253952 bad=0
ATTN_M1_PROBE hf zero_plus n=253952 bad=0
ATTN_M1_PROBE exp16 n=31745 bad=0
ATTN_M1_PROBE div16 hard n=27049 bad=0
EOF
want "G1 PASSED 1" "$(grep -c 'PASSED  \] 1 test' $L/g1_semantics.log)" 1
[ "$(grep -c '^ATTN_M1_PROBE qfma .* bad=0$' $L/g1.txt)" = 3 ] ||
  stop "G1: the qf32 FMA is not the fused fp16 FMA on silicon (or did not run): plan step 2's stop rule -- report, the fallback (plan 3.6, 1.3x) is a user decision"
therm t2

# ---- 3. (b) cost x lanes, fetch rate (2 min)
gt cost unittest_hvx_attn 'HvxAttnM1Probe.Cost'
grep -h '^ATTN_M1_PROBE_COST ' $L/cost.log | tee $L/cost.txt
want "cost lines" "$(grep -c '^ATTN_M1_PROBE_COST ' $L/cost.txt)" 24
want "cost ran == lanes" "$(grep -c -E 'lanes=([0-9]+) ran=\1 ' $L/cost.txt)" 24
therm t3

# ---- 4. (c) today's kernel: bit identity at 8 lengths, per-L warm / cold (5 min)
gt g3_attn unittest_hvx_attn 'HvxAttnM1.*'
# out bad=0 is the gate; bad_stats is recorded as the reference skel's
# value at each L (rule 37: the (m, l) pair may differ on tiny values)
for n in 1 63 64 65 512 513 1024 1536; do
  want "HvxAttnM1 L=$n out bad=0" "$(grep -c "^ATTN_M1_FIELD L=$n bad=0 bad_stats=[0-9]* of 2048" $L/g3_attn.log)" 1; done
grep -h -o '^ATTN_M1_FIELD L=[0-9]* bad=[0-9]* bad_stats=[0-9]*' $L/g3_attn.log | tee $L/bad_stats_ref.txt
want "append_chain bad=0" "$(grep -c '^ATTN_M1_FIELD append_chain L=65 bad=0' $L/g3_attn.log)" 1
grep -h -E '^ATTN_M1_(FIELD (cold )?pos|PHASE)' $L/g3_attn.log | tee $L/per_l.txt
want "PHASE lines (3 warm + 3 cold)" "$(grep -c '^ATTN_M1_PHASE pos=\(511\|1023\|1535\) \(warm\|cold\)' $L/per_l.txt)" 6
want "HvxAttnM1 3 of 4 tests pass (+ PerLayerCost prints)" "$(grep -c '^\[       OK \] HvxAttnM1\.\(RejectsBadShapes\|AppendChainEqualsBulk\|PerLayerCost\)' $L/g3_attn.log)" 3
therm t4

# ---- 5. (d) the CPU side: AttnM1F16Det.* at L up to 1536 (3 min)
gt f16det unittest_nntrainer_cpu_backend_fp16 'AttnM1F16Det.*'
for n in 513 1024 1536; do
  want "AttnM1F16Det L=$n (2 rope positions) bad=0" "$(grep -c "^ATTN_M1_F16 L=$n rope_pos=[0-9]* out bad=0 " $L/f16det.log)" 2; done
want "AttnM1F16Det no FAILED" "$(grep -c 'FAILED' $L/f16det.log)" 0
therm t5

# ---- 6. (e) full model from s170a (the #164 set): A first, then Q0 with the level-2 profile (8 min)
for g in 64 1024; do
  run A $g A_G$g; vbanner A A_G$g
  run Q $g Qp_G$g NNTR_HTP_PROFILE=2; vbanner Q Qp_G$g
  therm t6_G$g
done
grep -H 'graph: calls=' $L/Qp_G64.log $L/Qp_G1024.log | sed 's/.*logs\///' | tee $L/inmodel.txt
want "in-model profile lines" "$(grep -c 'ATTN_M1=' $L/inmodel.txt)" 2

# ---- 7. summary (1 min)
echo "--- G1"; cat $L/g1.txt
echo "--- cost (pcyc_per_fma64: wall pcycles per 64-lane FMA of all lanes together)"; cat $L/cost.txt
echo "--- today's per-L line"; cat $L/per_l.txt
echo "--- in-model"; cat $L/inmodel.txt
for g in 64 1024; do
  t=$(cmp -s <(strip $L/A_G$g.log) <(strip $L/Qp_G$g.log) && echo same || echo DIFF)
  echo "G=$g A: $(grep -h -E '^(prefill|generation|generation\(last 64\)):' $L/A_G$g.log | grep -o '[0-9.]* TPS' | tr '\n' ' ')| Q0-prof text vs A: $t"
done
# plan 3.5 with S1's silicon constants: cycles/layer = 32 L s + 32 L p + 60 L + 20000,
# s = scores2 and p = pv4 at 6 lanes; T = that x 1.15 at mean L 544.5 (G 64) and 1024.5 (G 1024)
c6() { grep "op=$1 lanes=6 " $L/cost.txt | grep -o 'pcyc_per_fma64=[0-9.e+-]*' | cut -d= -f2; }
s2=$(c6 scores2); s1=$(c6 scores1); pv=$(c6 pv4)
python3 - "$s2" "$s1" "$pv" <<'EOF'
import sys
try:
    s2, s1, pv = (float(x) for x in sys.argv[1:4])
except ValueError:
    sys.exit('model: a cost line is missing')
for name, s in (('scores2', s2), ('scores1', s1)):
    t = {g: 1.15 * (32 * L * s + 32 * L * pv + 60 * L + 20000)
         for g, L in ((64, 544.5), (1024, 1024.5))}
    print('model (%s, pv4) T64=%.0f T1024=%.0f pcyc/op -> %s' %
          (name, t[64], t[1024], 'within 600k at G=1024' if t[1024] <= 600000
           else 'ABOVE 600k at G=1024: plan 1 G6 says stop after S1 and report'))
EOF
echo "--- therm"; cat $L/therm.log
echo "=== done $(date '+%F %T %Z')  expectation mismatches: $MIS (0 = every expected line seen)"
