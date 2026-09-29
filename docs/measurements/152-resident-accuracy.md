# Measurement 152: a CPU-exact resident attention — gtests, dumps, speed, 8-prompt re-judge

Branch `htp/152-resident-accuracy`, code @ `5e11c102`: the skel and the
three gtests below were built from that tree, the app set from `632873e0`
(no source the app compiles changed after it: `git diff --stat 632873e0
5e11c102 -- nntrainer Applications` lists only DSP files and
`attn_m1_det.h`, which the app does not include). `7ffaa8d2` adds
`tools/htp/loop_check.py`. Plan
`docs/plans/152-resident-accuracy.md` §4 step 7. Estimated device time:
**≈ 55 min** (state + install 4, gtests 5, dumps 8, speed 12, text + PPL
21, checks 4; the plan said ≈ 50 — D0's 16 text / PPL runs are the part
to drop if time is short). Run
by the orchestrator on the workstation (the phone stays on USB).

## Why

1. The resident attention stretch (ROPE + ATTN_M1) now reproduces the
   Android CPU's fp16 attention bit for bit — fp16 q/k/v, fp16 RoPE, fused
   `vfmaq_f16` scores and PV, `exp_ps`, the sequential fp16 sum, `fdiv
   .8h` — instead of computing it in f32 (41–52 dB from the CPU, plan 152's
   table). This sitting reads whether that holds on silicon (G0: the spec
   equals the phone's CPU functions; G1: the kernel equals the spec; G2:
   the attention-only mask is bit-identical to A at every MoE input), what
   it costs (G3: the kernel is ≈ 9× the vector ops of the f32 one), and
   whether D with the new attention passes the re-judge on 8 prompts (G4:
   no new loops, pooled PPL ≤ A × 1.02, the user's approval).
2. Decision: G2 ✓ + G3 ✓ + G4 ✓ closes #152 (track (c) goes to the user
   with the numbers); G2 ✓ without G4 closes lever 1 and opens lever 2
   (the norms, plan §4 step 9); G3 ✗ with G2 ✓ → the speed ladder (b).

### Variants (3 for speed; one new binary set switched by env, plus the old set)

| | set / run dir | env (beyond `NNTR_NUM_THREADS=8`) | expected banner |
|---|---|---|---|
| **A** (reference, first) | new, `s152` | none | `[HTP] dspq: on …` **once**, and `[HTP] dspq: close calls=N served=N bad=0` (rule 40); no `graph:` line |
| **D0** | old #134/#132 set, `s152d0` | `KD` below | `[HTP] graph: init n_ops=228 resident=RMSNORM\|CONV1D_GATE\|QK_NORM\|ROPE\|ATTN_M1\|ADD\|ROUTER_TOPK\|MOE moe_ops=22`, `attn_m1: registered … cache=49152 KiB`, `calls/token=51.00` |
| **D1** | new, `s152` | `KD` below | the same three lines |

`KD="NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=MOE,RMSNORM,QK_NORM,ROPE,CONV1D_GATE,ATTN_M1,ADD,ROUTER_TOPK"`.
D0 is #134's D (its 28.16 tok/s at G = 512 carried the pre-#148 kernel,
941–959 µs at pos 1023). A switch-on log without its three lines, or A
without its one `dspq: on`, is **void** (rules 36, 40).
`0x8000040e` anywhere = a stale skel for that dir (rule 3): stop, fix the
push.

## Artifacts

New set `/local/mnt/workspace/htp_moe/152/set/`, `md5.txt` next to it:

| file | md5 | built with |
|---|---|---|
| `libnntr_hvx_skel.so` | `f51125995a129afa2a00e610775c9e30` | `test/htp/build.sh` (v79, HexKL 6.4.0.1): `UNDEFINED SYMBOLS OK (51 runtime imports)` |
| `nntrainer_causallm` | `03f4e3c32519e58692d183b861f3ca42` | `build_android.sh --htp --cache` (`jni/libs/arm64-v8a/`) |
| `libcausallm_core.so` | `ffc2b166602a0a4ba318b0603e715ad2` | same (`NNTR_HTP_FORWARD_KINDS` count 2, `NNTR_PPL_DECODE` present) |
| `libnntrainer.so` | `fd764a268d163244507f68bb58331119` | same (`jni/obj/local/arm64-v8a/`; NEEDED `libsdkl.so`, `libcdsprpc.so`; `dspq: on` 1; `U dspqueue_` 0) |
| `libccapi-nntrainer.so` | `8599efcf0a501eeaded611a8faf722ed` | same (`jni/obj/local/arm64-v8a/`) |
| `unittest_nntrainer_cpu_backend_fp16` | `e358c08fdd8e141a9246f9a11940b620` | `test/jni` ndk-build (links the `libnntrainer.so` above: `AttnM1F16Det.*`, G0) |
| `unittest_hvx_attn` | `deea69837a1c40b70817cc8bf0159e4b` | same (`HvxAttnM1.*`, G1 + G3) |
| `unittest_hvx_softmax` | `6ad3026e84fe5a0f34aead9d69a7b24c` | same (`HvxM1Ops.Rope64*`, G1) |
| `libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 sysroot |
| `libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL 6.4.0.1 (from `134-132/set/`) |
| `prompt512.txt` (p01), `bitset-02-code.txt` … `bitset-08-short.txt` (p02–p08) | as `docs/measurements/prompts/README.md` (p01 `fc65c158…`) | from `141b/set/`; 512 / 207 / 117 / 326 / 207 / 276 / 402 / 24 tokens |
| model `q40-qs4cx-wh`, `tokenizer.json` | on the phone since #100 | |

Old set (D0), unchanged: `/local/mnt/workspace/htp_moe/134-132/set/` with
its `../md5.txt` (skel `0c2d5b00…`, app `b6adb4f5…`, `libnntrainer.so`
`bd5abd80…`, `unittest_hvx_softmax` `25bec7df…`; see
`134-132-combined.md`). Not staged, never pushed: `libcdsprpc.so` (rule 5).

Workstation sanity before pushing (done while staging; repeat if rebuilt):

```
W=/local/mnt/workspace/htp_moe/152; W0=/local/mnt/workspace/htp_moe/134-132
(cd $W/set && LC_ALL=C md5sum -c ../md5.txt | grep -vc ': OK$')           # 0
(cd $W0/set && LC_ALL=C md5sum -c ../md5.txt | grep -vc ': OK$')          # 0
strings $W/set/libnntrainer.so | grep -c 'dspq: on'                       # 1
strings $W/set/libcausallm_core.so | grep -c NNTR_HTP_FORWARD_KINDS       # 2 (rule 36)
strings $W/set/nntrainer_causallm | grep -c 'per-layer-type totals'       # 0 (not a profile binary)
find $W $W0 -name 'libcdsprpc*' | wc -l                                   # 0
```

Step 6 of the plan, done on the workstation (`llvm-objdump -d` of the
staged `libnntrainer.so`, NDK r30): `neon::softmax_row_inplace<half>` has
six inlined `exp_ps`, each one fused `fmla .4s` (the `vmlaq` step) and two
`fcvtzs`, every other step a separate `fmul/fadd/fsub .4s`, then `fadd
.8h` (the sum) and `fdiv .8h`; `neon::compute_kcaches(half…)` one fused
`fmla .8h`, the `faddp .8h` tree, a scalar fp16 `fadd h` (`0 + t`) and the
`/ 8` in f32; `neon::compute_rotary_emb_value(…half…)`'s 8-wide loop is
`fmul / fsub / fadd .8h` (not fused; its fused scalar tail never runs at
head_dim 64); `neon::compute_fp16vcache_transposed` a fused by-element
`fmla .8h`. That is what `attn_m1_det.h` models. (The host check also
showed the fused `fx` step does not matter for any fp16 input: the
`unfused_fx` mutant is equivalent, so a different NDK fusing or not
fusing it would not move a bit.)

Rebuild recipe if `$W` is not on the measuring workstation: `git checkout
5e11c102`, `source tools/htp/env.sh`, `export HEXKL_ROOT=…` explicitly,
`git submodule update --init --depth 1`, copy
`Applications/CausalLM/lib/libtokenizers_android_c.a` from another
worktree, `./test/htp/build.sh`, `(cd Applications/CausalLM &&
./build_android.sh --htp)` (on a fresh `builddir`: `cd builddir && meson
configure -Dprefix=$PWD/android_build_result && ninja install`, then
`--htp --cache`; after any change under `nntrainer/` run `ninja -C builddir
&& ninja -C builddir install` first), then the `test/jni` ndk-build of
`unittest_hvx_attn unittest_hvx_softmax
unittest_nntrainer_cpu_backend_fp16` (gates skill rung 3);
`libc++_shared.so` from the NDK sysroot. The skel is not
byte-reproducible: with a rebuilt set the table above is void and the
md5s you push are the record.

## Steps (workstation, phone on USB)

Shell setup once. Every `adb` names the serial. Two clean run dirs: `s152`
(new set) and `s152d0` (old set); the model dir is shared.

```
cd /home/j2z0-lee/nntrainer-152 && git fetch -q && git checkout htp/152-resident-accuracy && source tools/htp/env.sh
W=/local/mnt/workspace/htp_moe/152; W0=/local/mnt/workspace/htp_moe/134-132
mkdir -p $W/logs $W/dump
S=R3CY10WM83Y      # adb devices: record the serial you use under Notes
C=/data/local/tmp/nntrainer/causallm; D=$C/s152; D0=$C/s152d0; M=../models/q40-qs4cx-wh; DD=/data/local/tmp/s152dump
KD="NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=MOE,RMSNORM,QK_NORM,ROPE,CONV1D_GATE,ATTN_M1,ADD,ROUTER_TOPK"
therm() { adb -s $S shell dumpsys battery | grep -E 'level|temperature'; adb -s $S shell cat /sys/class/thermal/thermal_zone0/temp; }
run() { # run <dir> <G> <log name> <prompt file> [env ...]
  local dir=$1 g=$2 log=$3 p=$4; shift 4
  adb -s $S shell "cd $dir && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $g/' $M/nntr_config.json && \
    grep num_to_generate $M/nntr_config.json && md5sum libnntr_hvx_skel.so && \
    $* NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm $M \"\$(cat $p)\"" \
    2>&1 | tee $W/logs/$log.log | grep -E '^(prefill|generation|total|peak memory)|dspq: (on|off|close)|graph: (init|forward)|attn_m1: registered|\[PPL\] decode tokens|libnntr_hvx_skel|0x8000040e'
}
# The generated text of a log: everything before the summary, minus banners
# and the per-step [PPL] lines (each with its own newline).
strip() { sed -n '/^=====/q;p' "$1" | perl -0pe 's/\[HTP\] [^\n]*\n//g; s/\[PPL\] [^\n]*\n//g' |
  grep -v 'moe m1 gemv\|libnntr_hvx_skel\|nntrainer_causallm\|num_to_generate'; }
P="prompt512.txt bitset-02-code.txt bitset-03-math.txt bitset-04-korean.txt bitset-05-json.txt bitset-06-dialogue.txt bitset-07-facts.txt bitset-08-short.txt"
```

### 0. Device state (1 min)

```
adb devices                           # the serial listed; note any other
therm | tee -a $W/logs/therm.log      # checkpoint 0 (battery %, °C·10, zone0 m°C)
```

Screen off, charger in, cool start.

### 1. Install (≈ 3 min) — model reused

```
adb -s $S shell ls -l $C/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin   # 4316133120
adb -s $S shell "rm -rf $D $D0 $DD && mkdir -p $D $D0 $DD"
adb -s $S push $W/set/. $D/
adb -s $S push $W0/set/. $D0/
adb -s $S shell "chmod 755 $D/nntrainer_causallm $D/unittest_* $D0/nntrainer_causallm $D0/unittest_*"
adb -s $S shell "cd $D && md5sum *"  | tee $W/logs/md5_device.log
adb -s $S shell "cd $D0 && md5sum *" | tee $W/logs/md5_device_d0.log
diff <(sort -k2 $W/md5.txt  | tr -d '\r') <(sort -k2 $W/logs/md5_device.log    | tr -d '\r') && echo MD5 OK
diff <(sort -k2 $W0/md5.txt | tr -d '\r') <(sort -k2 $W/logs/md5_device_d0.log | tr -d '\r') && echo MD5 D0 OK
```

Config (as #134 / #136 / #141: greedy, `init_seq_len: 512`, `moe_engine:
htp`, `moe_htp_layers: ""`), re-applied unconditionally:

```
adb -s $S shell "cd $D/$M && \
  sed -i 's/\"do_sample\": true/\"do_sample\": false/' generation_config.json && \
  sed -i 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' nntr_config.json && \
  (grep -q moe_engine nntr_config.json || sed -i 's/\"bad_word_ids\": \[124900\],/\"bad_word_ids\": [124900],\n    \"moe_engine\": \"htp\",/' nntr_config.json) && \
  grep -H do_sample generation_config.json && grep -H -E 'bad_word_ids|num_to_generate|init_seq_len|_engine|_htp_layers' nntr_config.json"
```

Expected echo: `do_sample": false`, `bad_word_ids": [124900]`,
`init_seq_len": 512`, `moe_engine": "htp"`, `moe_htp_layers": ""`.

### 2. Device gtests first (≈ 5 min) — plan step 7(a): G0, G1, G3

```
adb -s $S shell "cd $D && LD_LIBRARY_PATH=. ./unittest_nntrainer_cpu_backend_fp16 --gtest_filter='AttnM1F16Det.*'" 2>&1 \
  | tee $W/logs/gtest_G0.log | grep -E 'ATTN_M1_F16|OK \]|FAILED|PASSED'
adb -s $S shell "cd $D && md5sum libnntr_hvx_skel.so && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_attn --gtest_filter='HvxAttnM1.*'" 2>&1 \
  | tee $W/logs/gtest_G1_attn.log | grep -E 'md5|libnntr|ATTN_M1_(FIELD|PHASE)|OK \]|FAILED|PASSED'
adb -s $S shell "cd $D && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_softmax --gtest_filter='HvxM1Ops.Rope64*'" 2>&1 \
  | tee $W/logs/gtest_G1_rope.log | grep -E 'M1_OPS_FIELD rope|OK \]|FAILED|PASSED'
adb -s $S shell "cd $D0 && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_softmax --gtest_filter='HvxM1Ops.Rope64*'" 2>&1 \
  | tee $W/logs/gtest_rope_ref_d0.log | grep -E 'M1_OPS_FIELD rope|OK \]|FAILED|PASSED'   # the old skel's rule-37 reference
therm | tee -a $W/logs/therm.log      # checkpoint 1
```

Expected:
* G0: `ATTN_M1_F16 rope64 pos=… bad=0 of 2560` at 0 / 1 / 511 / 1023 / 4095;
  `ATTN_M1_F16 L=… rope_pos=… out bad=0 of 2048 pv_cases=… score_cases=…`
  at L = 1, 2, 63, 64, 65, 512, 513, 1024, each at rope_pos 0 and L − 1
  (pv_cases ≈ 400–800 and score_cases = 2(L − 3) at L ≥ 63 with rope_pos 0);
  `ATTN_M1_F16 exp probe d<=0 n=31744 bad_p0=0 bad_p1=0 bad_in[-17.5,0]=0`;
  `[  PASSED  ] 3 tests`.
* G1: `ATTN_M1_FIELD L=… bad=0 bad_stats=0 of 2048 pv_cases=… score_cases=…`
  at L = 1 … 1024; `ATTN_M1_FIELD append_chain L=65 bad=0`;
  `RejectsBadShapes` — rule 38 may answer `AEE_ERPC` (`0x80000600`) for
  the q-one-head-short case as on #130's skel; the two new head_dim 32 /
  128 register cases must read `AEE_EINVALIDFORMAT + 0x80000400`. Rope:
  `M1_OPS_FIELD rope64 pos=… bad=0` at every position (the old skel's
  reference printed `bad=10…18` at pos ≥ 1 on #130 — rule 37; the new
  path rounds those tiny values to fp16 zero on entry, so none are
  expected here).
* G3: `ATTN_M1_PHASE pos=1023 warm … dsp_us=…` (and `cold`), with the
  `scores= softmax= pv=` split.

**Stop rules (plan §4 step 7).** G0 `bad` ≠ 0 → the spec is not the phone's
CPU: G2 cannot be judged; run step 3 for the record, then stop. G1 `bad` ≠ 0
on a normal row → stale skel or a silicon mismatch: stop before step 4.
G3 is read, not a stop.

### 3. Dumps at G = 4, prompt 512 (≈ 8 min, a few hundred MB) — plan step 7(b)

Diagnostic cells, not speed variants. Switch-on runs dump every stretch
(`NNTR_HTP_DUMP_ALL=1`); every run is read against A1's manifest.

```
dump() { # dump <name> <dir> [env ...]
  local v=$1 dir=$2; shift 2
  adb -s $S shell "rm -rf $DD/$v && mkdir -p $DD/$v"
  run $dir 4 dump_$v prompt512.txt NNTR_HTP_DUMP=$DD/$v "$@"
  rm -rf $W/dump/$v && adb -s $S pull $DD/$v $W/dump/$v > /dev/null && adb -s $S shell "rm -rf $DD/$v"
}
F="NNTR_HTP_FORWARD=1 NNTR_HTP_DUMP_ALL=1 NNTR_HTP_FORWARD_KINDS"
dump A1 $D; dump A2 $D
dump N_attn  $D  $F=MOE,ROPE,ATTN_M1            # G2
dump N_qk    $D  $F=MOE,QK_NORM,ROPE,ATTN_M1
dump N_rms   $D  $F=MOE,RMSNORM
dump N_conv  $D  $F=MOE,CONV1D_GATE
dump N_route $D  $F=MOE,ROUTER_TOPK
dump D1      $D  $F=MOE,RMSNORM,QK_NORM,ROPE,CONV1D_GATE,ATTN_M1,ADD,ROUTER_TOPK
dump A0      $D0                                  # the old set's switch-off: must equal A1
dump O_attn  $D0 $F=MOE,ROPE,ATTN_M1              # before (#136: 40 -> 18 dB)
dump D0      $D0 $F=MOE,RMSNORM,QK_NORM,ROPE,CONV1D_GATE,ATTN_M1,ADD,ROUTER_TOPK
therm | tee -a $W/logs/therm.log      # checkpoint 2
E="python3 tools/htp/htp_dump_eval.py"
for v in A2 A0 N_attn N_qk N_rms N_conv N_route D1 O_attn D0; do $E --label $v $W/dump/A1 $W/dump/$v | tail -1; done | tee $W/logs/dump_eval.txt
```

Expected: `A2` and `A0` `bit_identical=1` (the null checks: A is stable run
to run and across the two sets; if `A0` is not, read the old-set rows
against `A0` instead and say so); **`N_attn` `bit_identical=1` (G2)**;
`O_attn` ≈ 40 → 18 dB as #136; the others name their `min_snr_db` and
`first_diff`. The plan's `MOE,ADD,ROUTER_TOPK` cell is replaced by
`MOE,ROUTER_TOPK`: a resident ADD needs every RMSNORM resident, so that
mask is refused at `set_decode_graph_desc` (`AEE_ENOTALLOWED`) and aborts
the app; ADD's own share is D1 − the rest.

### 4. Speed, prompt 512 (≈ 12 min) — plan step 7(c), G3

```
for g in 64 512 1024; do
  run $D  $g A_G${g}_r1  prompt512.txt
  run $D0 $g D0_G${g}_r1 prompt512.txt $KD
  run $D  $g D1_G${g}_r1 prompt512.txt $KD
  run $D  $g D1_G${g}_r2 prompt512.txt $KD
  run $D0 $g D0_G${g}_r2 prompt512.txt $KD
  run $D  $g A_G${g}_r2  prompt512.txt
  therm | tee -a $W/logs/therm.log    # checkpoints 3, 4, 5
done
```

Expected in every log: `prefill: 512 tokens, … TPS`, `generation: <G>
tokens, … TPS`, `generation(last 64): 64 tokens, … TPS`, `peak memory`, no
`[HTP-PROFILE]`; A its `dspq: on` once and `dspq: close calls=N served=N
bad=0`; D0 / D1 their three lines and `calls/token=51.00`.

### 5. Text and PPL on the 8 prompts at G = 256 (≈ 21 min) — plan step 7(d)(e)

```
i=0; for p in $P; do i=$((i+1))
  run $D  256 text_A_p0$i  $p
  run $D0 256 text_D0_p0$i $p $KD         # D0 may be dropped if time is short (plan)
  run $D  256 text_D1_p0$i $p $KD
done
therm | tee -a $W/logs/therm.log          # checkpoint 6
i=0; for p in $P; do i=$((i+1))
  adb -s $S shell "rm -f $D/cont_p0$i.ids $D0/cont_p0$i.ids"
  run $D 256 ppl_A_self_p0$i $p NNTR_PPL_DECODE=cont_p0$i.ids      # writes A's own continuation (source=self)
  adb -s $S shell "cp $D/cont_p0$i.ids $D0/"
  [ $i = 1 ] && run $D 256 ppl_A_forced_p01 $p NNTR_PPL_DECODE=cont_p01.ids   # null check: nll_sum = self's
  run $D0 256 ppl_D0_p0$i $p NNTR_PPL_DECODE=cont_p0$i.ids $KD
  run $D  256 ppl_D1_p0$i $p NNTR_PPL_DECODE=cont_p0$i.ids $KD
done
run $D 64 text_N_attn_G64_p01 prompt512.txt NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=MOE,ROPE,ATTN_M1   # G2's text cell
therm | tee -a $W/logs/therm.log          # checkpoint 7
```

Every `prefill:` line shows the prompt's count (512 / 207 / 117 / 326 / 207
/ 276 / 402 / 24). Every PPL log ends in `[PPL] decode tokens=256 … top1=…/256
source=self|file nll_sum=…`.

### 6. Checks on the workstation (≈ 4 min)

```
grep -c 'dspq: on' $W/logs/A_G*_r*.log $W/logs/text_A_*.log                          # 1 each (rule 40)
grep -h 'dspq: close' $W/logs/A_G*_r*.log | sed 's/empty_polls=.*//' | sort | uniq -c  # served = calls, bad=0
grep -L 'calls/token=51.00' $W/logs/D[01]_G*_r*.log $W/logs/text_D[01]_*.log         # nothing (rule 36)
grep -h 'attn_m1: registered' $W/logs/D1_G64_r1.log $W/logs/D0_G64_r1.log            # cache=49152 KiB both
grep -l 'HTP-PROFILE\|0x8000040e\|dspq: off' $W/logs/*.log                          # nothing
# G2 text cell
cmp -s <(strip $W/logs/A_G64_r1.log) <(strip $W/logs/text_N_attn_G64_p01.log) && echo 'G2 text identical' || echo 'G2 text DIFFERENT'
# loops (plan step 4's rule: a variant fails a prompt only where it loops and A does not)
i=0; for p in $P; do i=$((i+1))
  python3 tools/htp/loop_check.py --prompt $W/set/$p $W/logs/text_A_p0$i.log $W/logs/text_D0_p0$i.log $W/logs/text_D1_p0$i.log
done | tee $W/logs/loop_check.txt
# PPL per prompt and pooled (exp of the summed nll over the summed tokens)
grep -H '\[PPL\] decode tokens' $W/logs/ppl_*.log | sed 's/.*logs\///' | tee $W/logs/ppl.txt
for v in A_self D0 D1; do cat $W/logs/ppl_${v}_p0?.log | grep '\[PPL\] decode tokens' |
  sed -E 's/.*tokens=([0-9]+).*nll_sum=([0-9.e+-]+).*/\1 \2/' |
  awk -v v=$v '{n+=$1; s+=$2} END {printf "POOLED %s tokens=%d nll/token=%.6f ppl=%.6f\n", v, n, s/n, exp(s/n)}'; done | tee -a $W/logs/ppl.txt
cat $W/logs/therm.log
```

Then fill the tables below, paste A's and D1's texts for approval, commit
this file on the branch, push, and set #152 to `state:measured`.

## Gates (plan 152 §1)

| # | check | pass |
|---|---|---|
| G0 | `AttnM1F16Det.*` (step 2) | `bad=0` on every rope, attention and exp-probe line |
| G1 | `HvxAttnM1.*`, `HvxM1Ops.Rope64*` (step 2) | `out bad=0` and `bad_stats=0` at every L, `append_chain bad=0`, rope `bad=0`; the tiny-value rows no worse than the old skel's reference |
| G2 | `N_attn` dump eval (step 3), text cell (step 6) | `bit_identical=1` on every file of A1's manifest (null checks `A2`, `A0` = 1); the G = 64 p01 text ≡ A |
| G3 | `ATTN_M1_PHASE pos=1023 warm … dsp_us=` (step 2); decode (step 4) | `dsp_us ≤ 940`; **D1 decode ≥ D0 decode** (mirrored means) at G = 512 and 1024 |
| G4 | step 5–6 | no prompt where D1 loops and A does not; pooled decode PPL of D1 ≤ A × 1.02; user `text approved: y` per prompt |
| standing | every speed cell | prefill ≥ −5 % of A (mirrored band); A's PPL forced p01 = self (null check) |

## Results (fill in)

Unit: ______ , date / time: ______ , device `md5sum` = the tables: ___ / ___

### Device gtests (G0, G1, G3)

| line | value |
|---|---|
| G0 rope64 (5 positions) | bad = |
| G0 attention, rope_pos 0, L 1 / 2 / 63 / 64 / 65 / 512 / 513 / 1024 | bad = |
| G0 attention, rope_pos L − 1, same L | bad = |
| G0 exp probe | n=31744 bad_p0= bad_p1= bad_in[-17.5,0]= |
| G1 `ATTN_M1_FIELD L=… bad / bad_stats` (1 … 1024) | |
| G1 append_chain | bad = |
| G1 `RejectsBadShapes` (head_dim 32 / 128 lines) | |
| G1 rope64 new skel / old skel reference | bad = … / bad = … |
| G3 `ATTN_M1_PHASE pos=1023 warm` | dsp_us= scores= softmax= pv= busy_max= lanes= mhz= |
| G3 `ATTN_M1_PHASE pos=1023 cold` | dsp_us= |
| G3 `ATTN_M1_FIELD pos=1023 us=` (host-timed) | |

Reference (#134 ride-along, `134-132-combined.md`): pos 1023 warm
`dsp_us` 941–959 (the pre-#148 kernel D0 carries) and 386.6 (#148's O13,
f32). Host hint (hvx_emu, not timing): the new kernel runs 7.2× the old
one's emulated work at L = 1024 (96.7 vs 13.4 ms per call, one lane).

### Dumps (G2 and the per-kind table)

```
(paste dump_eval.txt: A2, A0, N_attn, N_qk, N_rms, N_conv, N_route, D1, O_attn, D0)
```

| run | set | mask | bit_identical | min_snr_db | first_diff |
|---|---|---|---|---|---|
| A2 | new | off | | | |
| A0 | old | off | | | |
| O_attn (before) | old | MOE,ROPE,ATTN_M1 | | | |
| **N_attn (G2)** | new | MOE,ROPE,ATTN_M1 | | | |
| N_qk | new | MOE,QK_NORM,ROPE,ATTN_M1 | | | |
| N_rms | new | MOE,RMSNORM | | | |
| N_conv | new | MOE,CONV1D_GATE | | | |
| N_route | new | MOE,ROUTER_TOPK | | | |
| D0 | old | D | | | |
| D1 | new | D | | | |

### Speed (prompt 512; mirrored per G: A D0 D1 run 1, D1 D0 A run 2)

| variant | G | run | prefill tok/s | decode tok/s (all) | decode tok/s (last 64) | peak RSS KB | text = A? | banner (dspq / calls/token) |
|---|---|---|---|---|---|---|---|---|
| A | 64 | 1 | | | | | (reference) | |
| D0 | 64 | 1 | | | | | | |
| D1 | 64 | 1 | | | | | | |
| D1 | 64 | 2 | | | | | | |
| D0 | 64 | 2 | | | | | | |
| A | 64 | 2 | | | | | (reference) | |
| A | 512 | 1 | | | | | (reference) | |
| D0 | 512 | 1 | | | | | | |
| D1 | 512 | 1 | | | | | | |
| D1 | 512 | 2 | | | | | | |
| D0 | 512 | 2 | | | | | | |
| A | 512 | 2 | | | | | (reference) | |
| A | 1024 | 1 | | | | | (reference) | |
| D0 | 1024 | 1 | | | | | | |
| D1 | 1024 | 1 | | | | | | |
| D1 | 1024 | 2 | | | | | | |
| D0 | 1024 | 2 | | | | | | |
| A | 1024 | 2 | | | | | (reference) | |

| G | A mean | D0 mean (vs A) | D1 mean (vs A) | D1 vs D0 | prefill band (A r1/r2) | G3 (D1 ≥ D0) |
|---|---|---|---|---|---|---|
| 64 | | | | | | (read) |
| 512 | | | | | | |
| 1024 | | | | | | |

Reference: #141 sitting (A = the dspqueue path, the "now"): decode 39.40
(G = 64) / 37.77 (G = 512); #134 sitting: D 28.61 / 28.16, A (FastRPC)
36.90 / 36.45; CPU now 52.43 / 49.22 (#94 s2). Goal ≥ 50 decode, prefill
≥ −5 % of this sitting's A.

### Re-judge (G4): loops and decode PPL at G = 256, forced on A's own continuation

| prompt | tokens | A loop (L1 / L2) | D0 loop | D1 loop | A ppl (self) | D0 ppl | D1 ppl | D1 fails (loops where A does not)? |
|---|---|---|---|---|---|---|---|---|
| p01 `prompt512.txt` | 512 | | | | | | | |
| p02 `bitset-02-code.txt` | 207 | | | | | | | |
| p03 `bitset-03-math.txt` | 117 | | | | | | | |
| p04 `bitset-04-korean.txt` | 326 | | | | | | | |
| p05 `bitset-05-json.txt` | 207 | | | | | | | |
| p06 `bitset-06-dialogue.txt` | 276 | | | | | | | |
| p07 `bitset-07-facts.txt` | 402 | | | | | | | |
| p08 `bitset-08-short.txt` | 24 | | | | | | | |
| **pooled** | | | | | | | | D1 / A = ___ (gate ≤ 1.02) |

A forced p01 = A self (null check): nll_sum ___ = ___ .
Known (plan step 4): A itself loops at G = 64 on p05 and p08 and on p01
after ≈ 300 words, so a D1 loop on those counts only where A's does not.

### Text approval (G = 256, the generated part of `strip text_<v>_p0<i>.log`)

| prompt | A | D1 | text approved (user: y/n) |
|---|---|---|---|
| p01 | <paste> | <paste> | |
| p02 | <paste> | <paste> | |
| p03 | <paste> | <paste> | |
| p04 | <paste> | <paste> | |
| p05 | <paste> | <paste> | |
| p06 | <paste> | <paste> | |
| p07 | <paste> | <paste> | |
| p08 | <paste> | <paste> | |

D0's texts stay in `$W/logs/` (not for approval; D0 is the before).

## Host evidence this sitting rests on (workstation, not device numbers)

* `run_host_checks.sh`: `ALL CHECKS PASS`, `WORKER POOL LANES OK`, `ATTN M1
  F16 CPU-ORDER OK mutants=4/4 equivalent=1` (the spec equals an
  independent `_Float16` model of the CPU on exp at all 31745 fp16 d ≤ 0,
  RoPE, attention at L 1 … 1024 with adversarial rows; `no_rto`,
  `hf_nonfused`, `sum32`, `rope_f32` caught; `unfused_fx` equivalent),
  `ATTN M1 PRIM rne16 / fma16 / div16 … bad=0` (div16 on every quotient of
  the domain that is hard: 7881 off-by-one products and 27049 exact ties),
  `ATTN M1 BIT-IDENTICAL`, `ATTN M1 PHASES OK`, `M1 OPS BIT-IDENTICAL`,
  `GRAPH CHECKS PASS`.
* `run_inproc_e2e.sh`: `INPROC E2E PASS`. The host CPU attention is f32
  (the fp16 branch is Android-only), so the resident lines against the
  host switch-off moved as the plan expected — base tree → this tree:
  `fwd-hd64 min_snr_db` 39.18 → **37.17** (floor 30), `fwd-lfm25
  min_snr_db_gated` 31.86 → 30.52, `d-hd64` 60.09 → 52.92, `d-lfm25`
  54.37 → 131.68, `ppl-decode hd64 on` 7.14144 → 7.14078 (off 7.14184);
  every default-path line (`golden`, `hmx-loop`, `dspq-*`, tokens 8/8)
  unchanged.

## Notes from the run

Thermal checkpoints (0 start, 1 after the gtests, 2 after the dumps, 3–5
after each G, 6 after the texts, 7 after the PPL runs):

```
(paste therm.log)
```

FARF / AEE errors, anything stale, the serial used.
