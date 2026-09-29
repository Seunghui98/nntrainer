#!/usr/bin/env bash
# Plan 132 PR 2 step A6 (docs/measurements/132-pr2-exact-fc.md on
# htp/132-exact-fc): the CPU-exact FC, router, SwiGLU and argmax on silicon.
# One set, built from dev/fc-shadow (= the PR diff + the inert measurement
# commit). Variants: A (switch off, the reference, first) and S = A +
# NNTR_FC_SHADOW (the DSP re-runs every M=1 FC / lm_head slice, residual ADD
# and router on the CPU's own input for the first 8 decode steps and records
# both outputs; nothing the CPU computes is replaced, so S must equal A).
# Usage: bash /local/mnt/workspace/htp_moe/132/run_132.sh [serial]   (~55 min)
# Logs to $L, summary to $L/sitting.out. Stops on a stale skel or a missing
# unit; every other missing expected line is counted (expectation
# mismatches: N) and the plan's stop rules are reported, not enforced, so
# that G3 / G4 (decision D's inputs) are still read in the same sitting.
set -u -o pipefail
S=${1:-R3CY10WM83Y}
W=/local/mnt/workspace/htp_moe/132; L=$W/logs; mkdir -p $L $W/shadow
C=/data/local/tmp/nntrainer/causallm; D=$C/s132; M=../models/q40-qs4cx-wh
{ set +u; source /home/j2z0-lee/nntrainer-132x/tools/htp/env.sh > /dev/null 2>&1; set -u; }
exec > >(tee -a $L/sitting.out) 2>&1
MIS=0; RULE=0
stop() { echo "STOP: $*"; exit 1; }
rule() { echo "PLAN STOP RULE HIT (reported, sitting continues): $*"; RULE=$((RULE + 1)); }
want() { # want <label> <got> <expected>
  if [ "$2" = "$3" ]; then echo "OK  $1 = $2"; else echo "BAD $1: got '$2' want '$3'"; MIS=$((MIS + 1)); fi; }
therm() { echo "$1 $(date +%H:%M:%S) $(adb -s $S shell 'dumpsys battery | grep -E "^  (level|temperature)"; cat /sys/class/thermal/thermal_zone0/temp' | tr -d '\r' | tr '\n' ' ')" | tee -a $L/therm.log; }
zone0() { adb -s $S shell cat /sys/class/thermal/thermal_zone0/temp | tr -d '\r'; }
cool() { local t z; for t in $(seq 1 40); do z=$(zone0); [ "$z" -le 35000 ] && break
  echo "zone0=$z > 35000, waiting 30 s ($t/40)"; sleep 30; done; echo "zone0=$(zone0)"; }
F="NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=MOE,RMSNORM,QK_NORM,ROPE,CONV1D_GATE,ATTN_M1,ADD,ROUTER_TOPK NNTR_HTP_PROFILE=2"
env_of() { case $1 in
  A*) echo "";;
  S*) echo "NNTR_FC_SHADOW=$D/shadow_scratch.bin";;
  Pb) echo "$F ADSP_LIBRARY_PATH=$D/base";;   # profile only: the htp_moe skel (the old router)
  P) echo "$F";;                              # profile only: this set's skel (the CPU-order router)
esac; }
run() { # run <variant> <G> <log name> <prompt file> [extra env ...]
  local v=$1 g=$2 log=$3 p=$4; shift 4
  adb -s $S shell "cd $D && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $g/' $M/nntr_config.json && \
    grep num_to_generate $M/nntr_config.json && md5sum libnntr_hvx_skel.so base/libnntr_hvx_skel.so && \
    LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. $(env_of $v) $* NNTR_NUM_THREADS=8 \
    ./nntrainer_causallm $M \"\$(cat $p)\"" \
    > $L/$log.log 2>&1
  grep -qi '0x8000040e' $L/$log.log && stop "$log: 0x8000040e (stale skel, rule 3)"
  grep -q 'dev/fc-shadow' $L/$log.log && { echo "BAD $log: dev/fc-shadow error: $(grep -m1 'dev/fc-shadow' $L/$log.log)"; MIS=$((MIS + 1)); }
  echo "$log: $(grep -h -E '^(prefill|generation):' $L/$log.log | grep -o '[0-9.]* TPS' | tr '\n' ' ')$(grep -h -o '\[PPL\] decode tokens=.*' $L/$log.log | cut -c1-90)"
}
dspq() { # A and S are both switch-off runs: the MoE on the dspqueue, no graph (rule 36)
  local f=$L/$1.log
  want "$1 dspq: on" "$(grep -c 'dspq: on' $f)" 1
  want "$1 dspq close bad=0" "$(grep -c 'dspq: close calls=\([0-9]*\) served=\1 bad=0' $f)" 1
  want "$1 no graph" "$(grep -c 'graph: init' $f)" 0; }
strip() { sed -n '/^=====/q;p' "$1" | perl -0pe 's/\[HTP\] [^\n]*\n//g; s/\[PPL\] [^\n]*\n//g; s/\[FC-SHADOW\] [^\n]*\n//g' |
  grep -v 'moe m1 gemv\|libnntr_hvx_skel\|nntrainer_causallm\|num_to_generate'; }
nll() { grep -o '\[PPL\] decode step=.*' "$1"; }
P="prompt512.txt bitset-02-code.txt bitset-03-math.txt bitset-04-korean.txt bitset-05-json.txt bitset-06-dialogue.txt bitset-07-facts.txt bitset-08-short.txt"

echo "=== 132 PR 2 sitting  $(date '+%F %T %Z')  unit=$S"
# ---- 0. device state, cool start (1-5 min)
adb devices | tee $L/devices.log
adb devices | grep -q "^$S[[:space:]]*device" || stop "unit $S not attached"
adb -s $S shell input keyevent 223 || true
therm t0; cool

# ---- 1. install and config (3 min)
[ "$(cd $W/set && LC_ALL=C md5sum -c md5.txt | grep -vc ': OK$')" = 0 ] || stop "workstation set differs from set/md5.txt"
adb -s $S shell ls -l $C/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin
adb -s $S shell "rm -rf $D && mkdir -p $D" && adb -s $S push $W/set/. $D/ > /dev/null
adb -s $S shell "rm -f $D/md5.txt; chmod 755 $D/nntrainer_causallm $D/unittest_*"
adb -s $S shell "mkdir -p $D/base && mv $D/libnntr_hvx_skel_base.so $D/base/libnntr_hvx_skel.so"
adb -s $S shell "cd $D && md5sum \$(ls -p | grep -v / | sort) && cd base && md5sum libnntr_hvx_skel.so | sed 's/libnntr_hvx_skel.so/libnntr_hvx_skel_base.so/'" > $L/md5_device.log
diff <(sort -k2 $W/set/md5.txt | tr -d '\r') <(sort -k2 $L/md5_device.log | tr -d '\r') && echo "MD5 OK" || stop "device md5 differs from md5.txt"
adb -s $S shell "cd $D/$M && \
  sed -i 's/\"do_sample\": true/\"do_sample\": false/' generation_config.json && \
  sed -i 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' nntr_config.json && \
  (grep -q moe_engine nntr_config.json || sed -i 's/\"bad_word_ids\": \[124900\],/\"bad_word_ids\": [124900],\n    \"moe_engine\": \"htp\",/' nntr_config.json) && \
  grep -H do_sample generation_config.json && grep -H -E 'bad_word_ids|num_to_generate|init_seq_len|_engine|_htp_layers' nntr_config.json" \
  | tee $L/config.log
for k in '"do_sample": false' '"bad_word_ids": \[124900\]' '"init_seq_len": 512' '"moe_engine": "htp"' '"moe_htp_layers": ""'; do
  want "config $k" "$(grep -c "$k" $L/config.log)" 1; done

# ---- 2. G0: the specs == this set's CPU (6 min; ExpfBionic sweeps all 2^32 on 8 threads)
adb -s $S shell "cd $D && LD_LIBRARY_PATH=. ./unittest_nntrainer_cpu_backend \
  --gtest_filter='Q8QuantCpuOrder.*:Q4GemvCpuOrder.*:SwigluCpuOrder.*:SgemvNCpuOrder.*:ExpfBionic.*'" > $L/gtest_G0.log 2>&1
grep -E '^(Q8QuantCpuOrder|Q4GemvCpuOrder|SwigluCpuOrder|SgemvNCpuOrder|ExpfBionic)|OK \]|FAILED|PASSED|SKIPPED' $L/gtest_G0.log
want "G0 quantizer" "$(grep -c '^Q8QuantCpuOrder rows=4000 bad_blocks=0$' $L/gtest_G0.log)" 1
want "G0 Q4 GEMV 5 shapes" "$(grep -c '^Q4GemvCpuOrder K=[0-9]* N=[0-9]* rows=2000 bad=0 layout=ok$' $L/gtest_G0.log)" 5
want "G0 swiglu" "$(grep -c '^SwigluCpuOrder n=7168 rows=2000 bad=0$' $L/gtest_G0.log)" 1
want "G0 sgemv_n + selection" "$(grep -c '^SgemvNCpuOrder K=2048 E=32 rows=2000 bad_logits=0 bad_sel=0 ' $L/gtest_G0.log)" 1
want "G0 expf all 2^32" "$(grep -c '^ExpfBionic inputs=4278190082 bad=0$' $L/gtest_G0.log)" 1
want "G0 PASSED 5" "$(grep -c 'PASSED  \] 5 tests' $L/gtest_G0.log)" 1
grep -q 'PASSED  \] 5 tests' $L/gtest_G0.log || rule "G0 fails: the CPU is not what plan section 0 read (re-read that set)"
therm t1; cool

# ---- 3. G1: kernel == spec on silicon (3 min)
adb -s $S shell "cd $D && md5sum libnntr_hvx_skel.so && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  ./unittest_hvx_softmax --gtest_filter='HvxFcQ4.MatchesSpecBitExact:HvxFcQ4.SmallOpsMatchSpec:HvxM1Ops.*'" > $L/gtest_G1.log 2>&1
grep -E 'libnntr_hvx|FC_Q4_FIELD .*bad=[1-9]|FC_Q4_FIELD (total|small)|M1_OPS_FIELD router|OK \]|FAILED|PASSED' $L/gtest_G1.log
want "G1 FC 5 shapes x 5 variants x 2 feeds bad=0" "$(grep -c '^FC_Q4_FIELD K=.* rows=8 bad=0$' $L/gtest_G1.log)" 50
want "G1 FC total" "$(grep -c '^FC_Q4_FIELD total bad=0$' $L/gtest_G1.log)" 1
want "G1 small ops" "$(grep -c '^FC_Q4_FIELD small ops: q8_quant bad_rows=0 swiglu_cpu bad=0 argmax bad=0$' $L/gtest_G1.log)" 1
want "G1 router 3 shapes" "$(grep -c 'M1_OPS_FIELD router_topk .* bad_logits=0 bad_sel=0 bad_weight=0' $L/gtest_G1.log)" 3
echo "G1 by variant (bad rows over the 10 shape x feed cells):"
for v in hvx_native hvx_intrin sffma8 sffma16 sffma32; do
  echo "  $v: $(grep "variant=$v " $L/gtest_G1.log | grep -vc 'bad=0$') cells with bad > 0"; done
grep -q '^FC_Q4_FIELD total bad=0$' $L/gtest_G1.log ||
  rule "G1 fails: stale skel (0x8000040e) or a sf / sffma op not RN on silicon; if only hvx_native is bad, the next sitting uses the intrinsics"
therm t2; cool

# ---- 4. G3: the exact GEMV's rate (2 min), zone0 before / after
echo "G3 zone0 before: $(zone0)"
adb -s $S shell "cd $D && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  ./unittest_hvx_softmax --gtest_filter='HvxFcQ4.Rate'" > $L/gtest_G3.log 2>&1
echo "G3 zone0 after: $(zone0)"
grep -E '^FC_RATE_PROJ .*lanes=6|^FC_RATE bad_total|OK \]|FAILED' $L/gtest_G3.log
want "G3 outputs bit-equal" "$(grep -c '^FC_RATE bad_total=0$' $L/gtest_G3.log)" 1
therm t3; cool

# ---- 5. G4 (heap probe, arena) and A's own continuation, prompt 512, G = 8 (1 min)
adb -s $S shell "rm -f $D/cont.ids"
run A 8 f_A prompt512.txt NNTR_HEAP_PROBE=1 NNTR_PPL_DECODE=cont.ids; dspq f_A
want "f_A source=self" "$(grep -c 'decode tokens=8 .* source=self' $L/f_A.log)" 1
grep -h -E '\[HTP\] (arena|heap probe)' $L/f_A.log | tee $L/g4.txt
want "G4 heap probe" "$(grep -c '\[HTP\] heap probe: [0-9]* MiB' $L/g4.txt)" 1

# ---- 6. G2: S at G = 8 forced on A's tokens, every record DSP == CPU (3 min)
adb -s $S shell "rm -rf $D/d_S && mkdir -p $D/d_S"
run S 8 f_S prompt512.txt NNTR_FC_SHADOW=$D/d_S/fc.bin NNTR_FC_SHADOW_STEPS=0 NNTR_PPL_DECODE=cont.ids; dspq f_S
rm -rf $W/shadow/d_S && adb -s $S pull $D/d_S $W/shadow/ > /dev/null && adb -s $S shell "rm -rf $D/d_S"
python3 $W/tools/fc_shadow_check.py $W/shadow/d_S/fc.bin | tee $L/shadow_check.txt
want "G2 every record DSP == CPU" "$(grep -c '^FC SHADOW .* fc=\([0-9]*\)/\1 add=\([0-9]*\)/\2 router=\([0-9]*\)/\3 ' $L/shadow_check.txt)" 1
cmp -s <(nll $L/f_A.log) <(nll $L/f_S.log) && echo "OK  G2 nll f_S == f_A" || { echo "BAD G2 nll f_S != f_A"; MIS=$((MIS + 1)); }
therm t4

# ---- 7. inertness and speed: prompt 512, A S S A at G = 64 / 512 / 1024 (14 min)
for g in 64 512 1024; do cool
  run A $g A_G${g}_r1 prompt512.txt; dspq A_G${g}_r1
  run S $g S_G${g}_r1 prompt512.txt; dspq S_G${g}_r1
  run S $g S_G${g}_r2 prompt512.txt; dspq S_G${g}_r2
  run A $g A_G${g}_r2 prompt512.txt; dspq A_G${g}_r2
  therm t5_G$g
done

# ---- 7b. the resident router's cost, old vs new skel (review finding 3):
# the D mask of the 134+132 sitting, NNTR_HTP_PROFILE=2, G = 64; profile
# only, never a tok/s cell; the text is not compared (RMSNORM etc. resident)
cool
for v in Pb P; do run $v 64 prof_$v prompt512.txt
  want "prof_$v graph banner" "$(grep -c 'graph: init .*resident=RMSNORM|CONV1D_GATE|QK_NORM|ROPE|ATTN_M1|ADD|ROUTER_TOPK|MOE' $L/prof_$v.log)" 1
  want "prof_$v calls/token=51.00" "$(grep -c 'calls/token=51.00' $L/prof_$v.log)" 1
  grep -h 'graph: calls=' $L/prof_$v.log | sed "s/^/$v /" | tee -a $L/router_prof.txt
done
echo "ROUTER_TOPK pcyc/op: htp_moe skel $(grep -o '^Pb .* ROUTER_TOPK=[0-9]*' $L/router_prof.txt | grep -o '[0-9]*$')  this set $(grep -o '^P .* ROUTER_TOPK=[0-9]*' $L/router_prof.txt | grep -o '[0-9]*$')" | tee -a $L/router_prof.txt

# ---- 8. nll and text over 8 prompts at G = 256: A self, S forced on A, A / S text (16 min)
i=0; for p in $P; do i=$((i+1)); cool
  adb -s $S shell "rm -f $D/cont_p0$i.ids"
  run A 256 ppl_A_p0$i $p NNTR_PPL_DECODE=cont_p0$i.ids
  run S 256 ppl_S_p0$i $p NNTR_PPL_DECODE=cont_p0$i.ids
  run A 256 text_A_p0$i $p; run S 256 text_S_p0$i $p
done
therm t6

# ---- 9. checks and summary
echo "--- nll (every [PPL] decode step line of S equal to A's, 17 digits)"
for i in 1 2 3 4 5 6 7 8; do
  if cmp -s <(nll $L/ppl_A_p0$i.log) <(nll $L/ppl_S_p0$i.log); then echo "p0$i S nll == A"; else echo "BAD p0$i S nll != A"; MIS=$((MIS + 1)); fi
  if cmp -s <(strip $L/text_A_p0$i.log) <(strip $L/text_S_p0$i.log); then echo "p0$i S text identical"; else echo "BAD p0$i S text DIFFERENT"; MIS=$((MIS + 1)); fi
done
echo "--- speed (prefill / decode / last 64 TPS; text vs A run 1 of the same G)"
for g in 64 512 1024; do for r in r1 r2; do for v in A S; do f=$L/${v}_G${g}_$r.log
  t=$(cmp -s <(strip $L/A_G${g}_r1.log) <(strip $f) && echo same || echo DIFF)
  echo "G=$g $v $r: $(grep -h -E '^(prefill|generation|generation\(last 64\)):' $f | grep -o '[0-9.]* TPS' | tr '\n' ' ')text=$t"
done; done; done > $L/speed.txt
cat $L/speed.txt
want "speed cells with S text == A" "$(grep -c 'text=DIFF' $L/speed.txt)" 0
grep -h '\[PPL\] decode tokens' $L/ppl_*.log > $L/ppl.txt
echo "--- G2"; cat $L/shadow_check.txt
echo "--- G3 (ms per token projected, 6 lanes; the CPU is 7.4)"; grep -h '^FC_RATE_PROJ' $L/gtest_G3.log
echo "--- G4"; cat $L/g4.txt
echo "--- router (resident, pcyc/op)"; tail -1 $L/router_prof.txt
echo "--- therm"; cat $L/therm.log
echo "=== done $(date '+%F %T %Z')  expectation mismatches: $MIS  plan stop rules hit: $RULE"
