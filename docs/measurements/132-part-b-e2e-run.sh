#!/usr/bin/env bash
# Plan 132 Part B step E5 (docs/plans/132-part-b-two-session-e2e.md on
# htp/132-partb-plan; handoff docs/measurements/132-part-b-e2e.md on
# htp/132-partb-e3): the two-session NPU end-to-end decode on silicon.
# REBOOT THE PHONE FIRST (#178: a leaked cDSP mapping survives processes;
# only a reboot clears it). Two run dirs so no stub meets a foreign skel:
#   e3 (…/causallm/s132e5)   htp/132-partb-e3: A (nothing set) and
#                            E (NNTR_HTP_E2E=1: S1 router + MoE, S2 the rest)
#   sh (…/causallm/s132e5s)  dev/e2e-shadow-132 = e3 + the norm / attn / FC
#                            shadows (same DSP sources: its skel md5 must
#                            equal e3's)
# Variants: A = e3, switch off (first and last of every block); E = e3,
# NNTR_HTP_E2E=1; S = sh, NNTR_HTP_FORWARD=1 (#130's six kinds) + the
# shadows: per op, the E build's DSP kernels on the CPU's own input (norm,
# q|k norm, attention, FC via S2's L2 feed at 3 lanes, ADD, router,
# SwiGLU, argmax); S0 = sh, switch off (S's logits reference); Ev = sh,
# NNTR_HTP_E2E=1: the E2E token's logits per step against S0's (the layer
# shadows do not fire there: the whole token is on the DSP).
# Every E run opens and tears down a second session, so each is followed
# by an A run that must register its arena and generate (the #178 rule);
# a missing generation line prints LEAK and stops.
# set_e5b (after set_e5, 06:13): S2 now opens after S1's MoE arena is
# mapped (set_e5's E runs died at S1 3584 MiB with S2 opened and mapped
# first); each E / Ev run keeps its logcat's fastrpc lines ($L/<log>.logcat).
# Usage: bash /local/mnt/workspace/htp_moe/132/set_e5b/run_e5.sh [serial]  (~90 min)
# Everything is logged to $L, the summary to $L/sitting.out.
set -u -o pipefail
WT=/home/j2z0-lee/nntrainer-132e3           # env.sh only
S=${1:-R3CY10WM83Y}
W=/local/mnt/workspace/htp_moe/132/set_e5b; L=$W/logs; mkdir -p $L $W/shadow $W/dumps
C=/data/local/tmp/nntrainer/causallm; DE=$C/s132e5; DS=$C/s132e5s; M=../models/q40-qs4cx-wh
cd $WT && { set +u; source tools/htp/env.sh > /dev/null 2>&1; set -u; }
exec > >(tee -a $L/sitting.out) 2>&1
MIS=0
stop() { echo "STOP: $*"; therm stop; exit 1; }
want() { # want <label> <got> <expected>
  if [ "$2" = "$3" ]; then echo "OK  $1 = $2"; else echo "BAD $1: got '$2' want '$3'"; MIS=$((MIS + 1)); fi; }
therm() { echo "$1 $(date +%H:%M:%S) $(adb -s $S shell 'dumpsys battery | grep -E "^  (level|temperature)"; cat /sys/class/thermal/thermal_zone0/temp' | tr -d '\r' | tr '\n' ' ')" | tee -a $L/therm.log; }
zone0() { adb -s $S shell cat /sys/class/thermal/thermal_zone0/temp | tr -d '\r'; }
cool() { # zone0 <= 35 C before a block
  local i z; for i in $(seq 1 40); do z=$(zone0); [ "$z" -le 35000 ] && break
    echo "zone0=$z > 35000, waiting 30 s ($i/40)"; sleep 30; done; echo "block start zone0=$(zone0)"; }
stale() { grep -qi '0x8000040e' "$1" && stop "$(basename $1): 0x8000040e (stale skel, rule 3)"; true; }
dir_of() { case $1 in S|S0|Ev) echo $DS;; *) echo $DE;; esac; }
env_of() { case $1 in
  A|S0) echo "";;
  E|Ev) echo "NNTR_HTP_E2E=1";;
  S) echo "NNTR_HTP_FORWARD=1";;
esac; }
run() { # run <variant> <G> <log name> <prompt file> [extra env ...]
  local v=$1 g=$2 log=$3 p=$4 d; d=$(dir_of $1); shift 4
  case $v in E|Ev) adb -s $S logcat -c;; esac
  adb -s $S shell "cd $d && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $g/' $M/nntr_config.json && \
    grep num_to_generate $M/nntr_config.json && md5sum libnntr_hvx_skel.so && \
    $(env_of $v) $* NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm $M \"\$(cat $p)\"" > $L/$log.log 2>&1
  stale $L/$log.log
  echo "$log: $(grep -h -E '^(prefill|generation):' $L/$log.log | grep -o '[0-9.]* TPS' | tr '\n' ' ')$(grep -h -o 'calls/token=[0-9.]*' $L/$log.log) $(grep -h -o '\[PPL\] decode tokens=.*' $L/$log.log | cut -c1-90)"
  case $v in E|Ev)
    adb -s $S logcat -d | grep -iE 'adsprpc|fastrpc|apps_mem|remote_mmap|mmap|nntr_hvx|HAP_' > $L/$log.logcat || true
    e2e_checks $log;; esac
}
e2e_checks() { # the stop rules of plan step E5 on one E / Ev log
  local f=$L/$1.log ms
  grep -q 'token driver: token .* failed' $f && stop "$1: $(grep -m1 -o 'token driver: token .* failed[^(]*' $f) (AEE_EEXPIRED = a hop timed out; the FARF names the side)"
  grep -q 's2: open FAILED' $f && stop "$1: $(grep -m1 's2: open FAILED' $f)"
  grep -q 'FATAL' $f && stop "$1: $(grep -m1 'FATAL' $f | cut -c1-240) (logcat: $L/$1.logcat)"
  ms=$(grep -m1 -o 'open_ms=[0-9.]*' $f | cut -d= -f2)
  [ -n "$ms" ] && awk -v m="$ms" 'BEGIN{exit !(m > 2000)}' && stop "$1: s2 open_ms=$ms > 2000"
  true; }
sanity() { # sanity <log name>: A at G = 8 after an E run; the app must register its arena and generate
  run A 8 $1 prompt512.txt
  grep -q '^generation:' $L/$1.log || stop "LEAK: A cannot generate after an E run ($1; reboot the phone, see its logcat)"; }
vbanner() { # vbanner <variant> <log>: the variant's banners, else the log is void (rule 36)
  local f=$L/$2.log
  case $1 in
  A|S0)
    want "$2 moe m1 gemv 0x703e1" "$(grep -c 'moe m1 gemv: on (applied=0x703e1)' $f)" 1
    want "$2 dspq: on" "$(grep -c '\[HTP\] dspq: on' $f)" 1
    want "$2 dspq close bad=0" "$(grep -c 'dspq: close calls=\([0-9]*\) served=\1 bad=0' $f)" 1
    want "$2 no s2 / graph" "$(grep -c 's2: open\|graph: init' $f)" 0;;
  E|Ev)
    want "$2 s2 open lite, 0 KiB VTCM" "$(grep -c 's2: open s1_effdom=[0-9]* s2_session=[0-9]* s2_effdom=[0-9]* open_ms=[0-9.]* info_rc=0x0 open_path=1 hmx=0 vtcm_kib=0 ' $f)" 1
    want "$2 s2 fc arena 67 weights, 74 handles, feed=l2" "$(grep -c 's2: fc arena weights=67 handles=74 attach_mib=[0-9.]* chunks=[0-9]* mapped_mib=[0-9]* feed=l2 load_ms=[0-9.]* s1_arena_mib=3840$' $f)" 1
    want "$2 q4m1 banner feed=l2" "$(grep -c 'graph: q4m1 weights=67 handles=74 feed=l2$' $f)" 1
    want "$2 graph init all kinds" "$(grep -cF 'graph: init n_ops=228 resident=RMSNORM|FC|CONV1D_GATE|QK_NORM|ROPE|ATTN_M1|ADD|ROUTER_TOPK|MOE|DENSE_FFN|LM_HEAD moe_ops=22' $f)" 1
    want "$2 attn cache on S2" "$(grep -cF 'max_seq=2048 cache=24576 KiB' $f)" 1
    want "$2 dspq[S2]: on" "$(grep -c '\[HTP\] dspq\[S2\]: on .* domain=' $f)" 1
    want "$2 token driver on, 22 rounds, 44 hops" "$(grep -c 'token driver: on s1_effdom=[0-9]* s2_effdom=[0-9]* mbox=65536 spin_us=[0-9]* rounds=22 hops/token=44 ' $f)" 1
    want "$2 close hops/token=44.00, no timeout / stale, ids agree" "$(grep -c 'token driver: close tokens=[0-9]* hops/token=44.00 .* timeouts=0/0 stale=0/0 .* id_mismatch=0 stop_err=0x0/0x0' $f)" 1
    want "$2 calls/token=1.00" "$(grep -cF 'calls/token=1.00' $f)" 1
    want "$2 dspq[S2] close bad=0" "$(grep -c 'dspq\[S2\]: close calls=\([0-9]*\) served=\1 bad=0' $f)" 1;;
  S)
    want "$2 graph init six kinds" "$(grep -cF 'graph: init n_ops=228 resident=RMSNORM|CONV1D_GATE|QK_NORM|ROPE|ATTN_M1|MOE moe_ops=22' $f)" 1
    want "$2 no s2" "$(grep -c 's2: open' $f)" 0;;
  esac; }
strip() { sed -n '/^=====/q;p' "$1" | perl -0pe 's/\[HTP[^\]\n]*\] [^\n]*\n//g; s/\[PPL\] [^\n]*\n//g; s/\[FC-SHADOW\] [^\n]*\n//g' |
  grep -v 'moe m1 gemv\|libnntr_hvx_skel\|nntrainer_causallm\|num_to_generate'; }
nll() { grep -o '\[PPL\] decode step=.*' "$1"; }
startup() { # e2e time - run() total: the process's load + init, ms
  local e t; e=$(grep -o '\[e2e time\]: [0-9]*' "$1" | grep -o '[0-9]*$'); t=$(grep -o '^total: [0-9]*' "$1" | grep -o '[0-9]*$')
  [ -n "$e" ] && [ -n "$t" ] && echo $((e - t)) || echo "?"; }
P="prompt512.txt bitset-02-code.txt bitset-03-math.txt bitset-04-korean.txt bitset-05-json.txt bitset-06-dialogue.txt bitset-07-facts.txt bitset-08-short.txt"

echo "=== 132 Part B E5 sitting  $(date '+%F %T %Z')  unit=$S  wt=$(git -C $WT rev-parse --short HEAD)"
# ---- 0. device state, cool start (1 min)
adb devices | tee $L/devices.log
adb devices | grep -q "^$S[[:space:]]*device" || stop "unit $S not attached"
echo "uptime (reboot first): $(adb -s $S shell cat /proc/uptime | tr -d '\r')"
adb -s $S shell input keyevent 223 || true
therm t0; cool

# ---- 1. install both run dirs and the model config (3 min)
[ "$(cd $W && LC_ALL=C md5sum -c md5.txt | grep -vc ': OK$')" = 0 ] || stop "workstation set differs from md5.txt"
want "sh skel == e3 skel (no DSP source differs)" "$(awk '$2=="e3/libnntr_hvx_skel.so"{a=$1} $2=="sh/libnntr_hvx_skel.so"{b=$1} END{print (a==b)?"y":"n"}' $W/md5.txt)" y
adb -s $S shell ls -l $C/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin
adb -s $S shell "rm -rf $DE $DS && mkdir -p $DE $DS"
adb -s $S push $W/e3/. $DE/ > /dev/null && adb -s $S push $W/sh/. $DS/ > /dev/null
adb -s $S shell "chmod 755 $DE/unittest_* $DE/nntrainer_causallm $DS/nntrainer_causallm"
for d in e3 sh; do
  dd=$DE; [ $d = sh ] && dd=$DS
  adb -s $S shell "cd $dd && md5sum \$(ls -p | grep -v / | sort)" | tr -d '\r' | awk -v d=$d '{print $1"  "d"/"$2}'
done > $L/md5_device.log
diff <(grep -E '^[0-9a-f]{32}  (e3|sh)/' $W/md5.txt | sort -k2) <(sort -k2 $L/md5_device.log) && echo "MD5 OK" || stop "device md5 differs from md5.txt"
adb -s $S shell "cd $DE/$M && \
  sed -i 's/\"do_sample\": true/\"do_sample\": false/' generation_config.json && \
  sed -i 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' nntr_config.json && \
  (grep -q moe_engine nntr_config.json || sed -i 's/\"bad_word_ids\": \[124900\],/\"bad_word_ids\": [124900],\n    \"moe_engine\": \"htp\",/' nntr_config.json) && \
  grep -H do_sample generation_config.json && grep -H -E 'bad_word_ids|num_to_generate|init_seq_len|_engine|_htp_layers|max_seq_len' nntr_config.json" \
  | tee $L/config.log   # one model dir, shared by both run dirs through ../models
for k in '"do_sample": false' '"bad_word_ids": \[124900\]' '"init_seq_len": 512' '"moe_engine": "htp"' '"moe_htp_layers": ""'; do
  want "config $k" "$(grep -c "$k" $L/config.log)" 1; done
therm t1

# ---- 2. canary: the exact FC's gtest on this skel (1 min); a stale skel stops here
adb -s $S shell "cd $DE && md5sum libnntr_hvx_skel.so unittest_hvx_softmax && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  ./unittest_hvx_softmax --gtest_filter='HvxFcQ4.MatchesSpecBitExact'" > $L/canary.log 2>&1
stale $L/canary.log
want "canary FC_Q4_FIELD total bad=0" "$(grep -c '^FC_Q4_FIELD total bad=0$' $L/canary.log)" 1
want "canary PASSED" "$(grep -c '^\[  PASSED  \] 1 test' $L/canary.log)" 1

# ---- 3. A first, then the dumps and the startup cell, prompt 512, G = 8 (4 min)
cool
run A 8 A_first prompt512.txt; vbanner A A_first
grep -q '^generation:' $L/A_first.log || stop "A cannot generate before any E run: the phone was not rebooted, or the arena does not fit"
for v in A E; do
  adb -s $S shell "rm -rf $DE/d_$v && mkdir -p $DE/d_$v"
  run $v 8 dump_$v prompt512.txt NNTR_HTP_DUMP=d_$v; vbanner $v dump_$v
  rm -rf $W/dumps/d_$v && adb -s $S pull $DE/d_$v $W/dumps/ > /dev/null && adb -s $S shell "rm -rf $DE/d_$v"
done
sanity sanity_dump
# E's dump holds the prefill MoE calls only (the decode MoE runs inside S1
# with no ARM call): E is the reference, so A's same calls are compared
n=$(wc -l < $W/dumps/d_E/manifest.txt)
rm -rf $W/dumps/d_A_pre && mkdir -p $W/dumps/d_A_pre && head -n $n $W/dumps/d_A/manifest.txt > $W/dumps/d_A_pre/manifest.txt
for c in $(awk '{print $1}' $W/dumps/d_A_pre/manifest.txt); do cp $W/dumps/d_A/${c}_in.f32 $W/dumps/d_A/${c}_out.f32 $W/dumps/d_A_pre/; done
want "E's dump = A's prefill calls ($n)" "$(cmp -s <(cut -d' ' -f2- $W/dumps/d_E/manifest.txt) <(cut -d' ' -f2- $W/dumps/d_A_pre/manifest.txt) && echo y || echo n)" y
python3 $W/tools/htp_dump_eval.py --label moe-dump-E==A $W/dumps/d_E $W/dumps/d_A_pre | tail -1 | tee $L/dump_eval.txt
want "MoE dumps E == A (prefill calls)" "$(grep -c 'bit_identical=1' $L/dump_eval.txt)" 1
echo "startup_ms (e2e time - run total): A $(startup $L/dump_A.log)  E $(startup $L/dump_E.log)  | E $(grep -h -o 'open_ms=[0-9.]*' $L/dump_E.log) $(grep -h -o 'attach_mib=[0-9.]*' $L/dump_E.log) $(grep -h -o 'load_ms=[0-9.]*' $L/dump_E.log)" | tee $L/startup.txt
therm t3

# ---- 4. shadows, prompt 512, G = 8, forced on A's own continuation (8 min)
cool
adb -s $S shell "rm -f $DE/cont8.ids $DS/cont8.ids"
run A 8 f_A prompt512.txt NNTR_PPL_DECODE=cont8.ids; vbanner A f_A
want "f_A source=self" "$(grep -c 'decode tokens=8 .* source=self' $L/f_A.log)" 1
adb -s $S shell "cp $DE/cont8.ids $DS/cont8.ids"
for v in S0 S Ev; do
  adb -s $S shell "rm -rf $DS/d_$v && mkdir -p $DS/d_$v"
  case $v in
    S) sh_env="NNTR_FC_SHADOW=d_S/fc.bin NNTR_FC_SHADOW_STEPS=0 NNTR_NORM_SHADOW=d_S/norm.bin NNTR_ATTN_SHADOW=d_S/attn.bin";;
    *) sh_env="";;
  esac
  run $v 8 f_$v prompt512.txt $sh_env NNTR_LOGIT_SHADOW=d_$v/logits.bin NNTR_PPL_DECODE=cont8.ids
  vbanner $v f_$v
  grep -q 'dev/fc-shadow' $L/f_$v.log && { echo "BAD f_$v: $(grep -m1 'dev/fc-shadow' $L/f_$v.log)"; MIS=$((MIS + 1)); }
  rm -rf $W/shadow/d_$v && adb -s $S pull $DS/d_$v $W/shadow/ > /dev/null && adb -s $S shell "rm -rf $DS/d_$v"
done
sanity sanity_shadow
for v in S0 S Ev; do cmp -s <(nll $L/f_A.log) <(nll $L/f_$v.log) && echo "B3 nll f_$v == f_A (8 steps)" || { echo "BAD nll f_$v != f_A"; MIS=$((MIS + 1)); }; done
python3 $W/tools/fc_shadow_check.py $W/shadow/d_S/fc.bin | tee $L/fc_shadow.txt
want "B2 FC shadow: every fc / add / router / swiglu / argmax record equal, n > 0" "$(python3 $W/tools/fc_shadow_check.py $W/shadow/d_S/fc.bin > /dev/null && echo y || echo n)" y
python3 $W/tools/norm_shadow_check.py $W/shadow/d_S0 $W/shadow/d_S | tee $L/norm_shadow.txt
want "B2 norm shadow S (tag0, tag1 heads all equal, logits 8/8 == S0)" "$(grep -c '^NORM SHADOW d_S tag0=\([0-9]*\)/\1 tag1_heads=\([0-9]*\)/\2 tag1_zero_records=0 tag2=0/0 logits_equal_steps=8/8' $L/norm_shadow.txt)" 1
python3 $W/tools/attn_shadow_check.py $W/shadow/d_S0 $W/shadow/d_S | tee $L/attn_shadow.txt
want "B2 attn shadow S" "$(grep -c '^ATTN SHADOW d_S tag3_heads=\([0-9]*\)/\1 records=48 layers=6 positions=8 zero_records=0 logits_equal_steps=8/8' $L/attn_shadow.txt)" 1
cmp -s $W/shadow/d_S0/logits.bin $W/shadow/d_Ev/logits.bin && echo "B2 Ev: the E2E token's logits == S0's, every step" || { echo "BAD Ev logits != S0"; MIS=$((MIS + 1)); }
want "Ev logits = 8 steps" "$(stat -c %s $W/shadow/d_Ev/logits.bin)" $((8 * 128000 * 4))
therm t4

# ---- 5. speed, prompt 512, G 64 / 512 / 1024, mirrored A E | E A (15 min)
for g in 64 512 1024; do
  cool
  run A $g A_G${g}_r1 prompt512.txt; vbanner A A_G${g}_r1
  run E $g E_G${g}_r1 prompt512.txt; vbanner E E_G${g}_r1
  run E $g E_G${g}_r2 prompt512.txt; vbanner E E_G${g}_r2
  run A $g A_G${g}_r2 prompt512.txt; vbanner A A_G${g}_r2
  grep -q '^generation:' $L/A_G${g}_r2.log || stop "LEAK: A_G${g}_r2 cannot generate after E"
  want "E_G${g}_r1 id-only (bad_word_ids ride LM_BAN)" "$(grep -c 'token driver: first token .* logits=0$' $L/E_G${g}_r1.log)" 1
  therm t5_G$g
done

# ---- 6. text and nll: 8 prompts at G = 256, E forced on A's continuation (45 min)
i=0; for p in $P; do i=$((i+1))
  [ $((i % 2)) = 1 ] && cool
  adb -s $S shell "rm -f $DE/cont_p0$i.ids"
  run A 256 ppl_A_p0$i $p NNTR_PPL_DECODE=cont_p0$i.ids                 # A's own continuation
  [ $i = 1 ] && run A 256 ppl_Af_p01 $p NNTR_PPL_DECODE=cont_p01.ids     # null check: A forced == A self
  run E 256 ppl_E_p0$i $p NNTR_PPL_DECODE=cont_p0$i.ids; vbanner E ppl_E_p0$i
  run A 256 text_A_p0$i $p
  grep -q '^generation:' $L/text_A_p0$i.log || stop "LEAK: text_A_p0$i cannot generate after E"
  run E 256 text_E_p0$i $p; vbanner E text_E_p0$i
done
sanity sanity_last
therm t6

# ---- 7. checks and summary (1 min)
echo "--- B3 nll (every [PPL] decode step line equal to A's, 17 digits)"
want "ppl_A_p01 source=self" "$(grep -c 'decode tokens=256 .* source=self' $L/ppl_A_p01.log)" 1
cmp -s <(nll $L/ppl_A_p01.log) <(nll $L/ppl_Af_p01.log) && echo "null check A forced == A self" || { echo "BAD A forced != A self"; MIS=$((MIS + 1)); }
for i in 1 2 3 4 5 6 7 8; do
  if cmp -s <(nll $L/ppl_A_p0$i.log) <(nll $L/ppl_E_p0$i.log); then echo "p0$i E nll == A"; else echo "BAD p0$i E nll != A"; MIS=$((MIS + 1)); fi
done | tee $L/nll.txt
want "B3 nll 8/8" "$(grep -c 'E nll == A' $L/nll.txt)" 8
echo "--- B3 text (G = 256): E vs A"
for i in 1 2 3 4 5 6 7 8; do
  if cmp -s <(strip $L/text_A_p0$i.log) <(strip $L/text_E_p0$i.log); then echo "p0$i E text == A"; else echo "BAD p0$i E text != A"; MIS=$((MIS + 1)); fi
done | tee $L/text.txt
want "B3 text 8/8" "$(grep -c 'E text == A' $L/text.txt)" 8
grep -H '\[PPL\] decode tokens' $L/ppl_*.log | sed 's/.*logs\///' > $L/ppl.txt
echo "--- B4 speed (prefill / decode / last 64 TPS; text vs A run 1 of the same G)"
for g in 64 512 1024; do for r in r1 r2; do for v in A E; do f=$L/${v}_G${g}_$r.log
  t=$(cmp -s <(strip $L/A_G${g}_r1.log) <(strip $f) && echo same || echo DIFF)
  echo "G=$g $v $r: $(grep -h -E '^(prefill|generation|generation\(last 64\)):' $f | grep -o '[0-9.]* TPS' | tr '\n' ' ')text=$t $(grep -h -o 'calls/token=[0-9.]*' $f) startup_ms=$(startup $f)"
done; done; done | tee $L/speed.txt
want "speed: every cell's text == A run 1 of its G" "$(grep -c 'text=DIFF' $L/speed.txt)" 0
echo "--- per session (E runs): pcyc / wait per token, hops"
grep -h -E 'graph\[S[12]\]:|token driver: close' $L/E_G*_r*.log | sed 's/^/  /' | tee $L/sessions.txt
echo "--- standing: prefill mean of E against A's (mirrored runs), -5 % band"
for g in 64 512 1024; do
  pa=$(grep "^G=$g A " $L/speed.txt | awk '{s += $4} END {print s / NR}')
  pv=$(grep "^G=$g E " $L/speed.txt | awk '{s += $4} END {print s / NR}')
  echo "G=$g E prefill $pv vs A $pa: $(awk -v a=$pv -v b=$pa 'BEGIN{d = 100 * (a / b - 1); printf "%+.1f %% %s", d, (d >= -5 ? "ok" : "BELOW -5 %")}')"
done | tee $L/prefill.txt
echo "--- startup (load_ms cell)"; cat $L/startup.txt
echo "--- B2"; cat $L/dump_eval.txt $L/fc_shadow.txt $L/norm_shadow.txt $L/attn_shadow.txt
echo "--- therm"; cat $L/therm.log
echo "=== done $(date '+%F %T %Z')  expectation mismatches: $MIS (0 = every expected line seen)"
