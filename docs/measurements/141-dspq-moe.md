# Measurement 141 (step 2): the 22 M==1 MoE calls through dspqueue, bit for bit

Branch `htp/141-dspq-moe`, code @ `07fb1938` (the set below was built from
that tree; the later commits on the branch are docs only). Plan
`docs/plans/141-dspq-moe.md`. Estimated device time: **≈ 25 min**. Run by the
orchestrator on the workstation (the phone stays on USB).

## Why

1. Transport: #147's microbench put one FastRPC round trip at 87 µs and a
   dspqueue round trip at 17–29 µs. This sitting reads whether the model's
   22 M==1 MoE calls per token get that saving (G5: ≤ 35 µs/call in the
   profile, against A's ≈ 82–92) and what it is worth in decode tok/s
   (G4, expected ≈ +5 %).
2. Accuracy, under the user's 2026-09-28 direction change: the change must
   not move a single bit. The DSP runs the FastRPC method's own function on
   the same bytes, so the MoE dumps must be `bit_identical=1` against A
   (G1) and the text identical to A on all 8 prompts of the new set (G2)
   and in every tok/s cell (G3). Any differing text is a defect, not an
   approval question.

### Variants (3, one binary set, environment only)

| | env (beyond `NNTR_NUM_THREADS=8`) | expected banner |
|---|---|---|
| **A** (reference, first) | none | **no** `dspq` line |
| **Q** | `NNTR_HTP_DSPQ=1` | `[HTP] dspq: on queue=0x… dsp_spin_us=1000 arm_spin_us=5000 buffers=2x65536 ion=y` once; at exit `[HTP] dspq: close calls=N served=N bad=0 …`, N > 0 |
| **Q0** | `NNTR_HTP_DSPQ=1 NNTR_HTP_DSPQ_SPIN_US=0` | the same with `dsp_spin_us=0` |

A Q / Q0 log without its `on` line (or with `[HTP] dspq: off (…)`) is
**void**, never "at A's speed" (rule 36). `dspq: off (dspq_start … 0x8000040e)`
means the pushed skel is not the staged one (rule 3): stop, fix the push.

## Artifacts (`/local/mnt/workspace/htp_moe/141b/set/`, `md5.txt` next to it)

| file | md5 | built with |
|---|---|---|
| `libnntr_hvx_skel.so` | `37468a7fdbf2e469589849598ca860ff` | `test/htp/build.sh` (v79, HexKL 6.4.0.1): `UNDEFINED SYMBOLS OK (51 runtime imports)`, 5 `dspqueue_*` all WEAK |
| `nntrainer_causallm` | `b1be9a062ae702af9e52b45510093826` | `build_android.sh --htp --cache` (`jni/libs/arm64-v8a/`) |
| `libcausallm_core.so` | `8204bf909354798c965b177ace694d15` | same (`NNTR_HTP_FORWARD_KINDS` count 2) |
| `libnntrainer.so` | `de7e2b8c425d67e7f941117a5e719ec7` | same (`jni/obj/local/arm64-v8a/`; NEEDED `libsdkl.so`, `libcdsprpc.so`; `dspq: on` count 1; `U dspqueue_` count 0) |
| `libccapi-nntrainer.so` | `36fc2231eb7947ee93835309091670a8` | same (`jni/obj/local/arm64-v8a/`) |
| `libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 (from `134-132/set/`) |
| `libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL 6.4.0.1 (from `134-132/set/`) |
| `unittest_hvx_dspq` | `de694f7d721a509a09790929bf2e44e0` | #147's bench on the moved `HtpDspqApi` (optional step 7f) |
| `prompt512.txt` (p01) | `fc65c1588dc66dd764c7013fe96cbb75` | `docs/measurements/77-prompt512.txt`, 512 tokens |
| `bitset-02-code.txt` … `bitset-08-short.txt` (p02–p08) | see `docs/measurements/prompts/README.md` | 207 / 117 / 326 / 207 / 276 / 402 / 24 tokens |
| model `q40-qs4cx-wh`, `tokenizer.json` | `7b7867fab5…` / `7b8067a5…` | on the phone since #100 |

Not staged, never pushed: `builddir/.../libcdsprpc.so` (rule 5).

Workstation sanity before pushing:

```
W=/local/mnt/workspace/htp_moe/141b
(cd $W/set && LC_ALL=C md5sum -c ../md5.txt | grep -vc ': OK$')           # 0
strings $W/set/libnntrainer.so | grep -c 'dspq: on'                       # 1
nm -D $W/set/libnntrainer.so | grep -c ' U dspqueue_'                     # 0 (dlsym only)
strings $W/set/libcausallm_core.so | grep -c NNTR_HTP_FORWARD_KINDS       # 2 (rule 36)
strings $W/set/nntrainer_causallm | grep -c 'per-layer-type totals'       # 0 (not a profile binary)
find $W -name 'libcdsprpc*' | wc -l                                       # 0
```

Rebuild recipe if `$W` is not on the measuring workstation: `git checkout
07fb1938`, `source tools/htp/env.sh`, `export HEXKL_ROOT=…` explicitly,
`git submodule update --init --depth 1`, copy
`Applications/CausalLM/lib/libtokenizers_android_c.a` from another worktree,
`./test/htp/build.sh`, `(cd Applications/CausalLM && ./build_android.sh --htp)`
(on a fresh `builddir`: `cd builddir && meson configure
-Dprefix=$PWD/android_build_result && ninja install`, then `--htp --cache`);
`libc++_shared.so` from the NDK sysroot if missing. `--cache` does not
rebuild `libnntrainer.so`: after any source change run `(cd builddir && ninja
install)` first, or the app set keeps the old library (it cost one rebuild
while staging this set). The skel is not byte-reproducible (two builds of
the same tree gave different md5s here), so a rebuilt set never matches the
table: record the md5s you push; with a rebuilt set the table above is void
and yours is the record.

## Steps (workstation, phone on USB)

Shell setup once. Every `adb` names the serial (a second device may be on
USB). Clean run dir `s141`: the old `causallm/` dir holds a foreign
`libc++_shared.so`.

```
cd /home/j2z0-lee/nntrainer-141b && git fetch -q && git checkout htp/141-dspq-moe && source tools/htp/env.sh
W=/local/mnt/workspace/htp_moe/141b; mkdir -p $W/logs $W/dump
S=R3CY10WM83Y      # adb devices: record the serial you use under Notes
D=/data/local/tmp/nntrainer/causallm/s141; M=../models/q40-qs4cx-wh; DD=/data/local/tmp/s141dump
therm() { adb -s $S shell dumpsys battery | grep -E 'level|temperature'; adb -s $S shell cat /sys/class/thermal/thermal_zone0/temp; }
env_of() { case $1 in Q) echo "NNTR_HTP_DSPQ=1";; Q0) echo "NNTR_HTP_DSPQ=1 NNTR_HTP_DSPQ_SPIN_US=0";; *) echo "";; esac; }
run() { # run <A|Q|Q0> <G> <log name> <prompt file> [extra env]
  adb -s $S shell "cd $D && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $2/' $M/nntr_config.json && \
    grep num_to_generate $M/nntr_config.json && md5sum libnntr_hvx_skel.so && \
    $(env_of $1) $5 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm $M \"\$(cat $4)\"" \
    2>&1 | tee $W/logs/$3.log | grep -E '^(prefill|generation|total|peak memory)|dspq:|moe m1 gemv|libnntr_hvx_skel'
}
# The generated text of a log: everything before the summary, minus the
# banners. The dspq on line is printed at the first decode call, so it can
# land inside the streamed text; perl removes it with its own newline.
strip() { sed -n '/^=====/q;p' "$1" | perl -0pe 's/\[HTP\] dspq: [^\n]*\n//g' |
  grep -v 'moe m1 gemv\|libnntr_hvx_skel\|nntrainer_causallm\|num_to_generate'; }
```

### 0. Device state (1 min)

```
adb devices                           # R3CY10WM83Y listed; note any other serial
therm | tee -a $W/logs/therm.log      # checkpoint 0 (battery %, °C·10, zone0 m°C)
```

Screen off, charger in, cool start.

### 1. Install (≈ 2 min) — model reused

```
adb -s $S shell ls -l /data/local/tmp/nntrainer/causallm/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin   # 4316133120
adb -s $S shell mkdir -p $D $DD
adb -s $S push $W/set/. $D/
adb -s $S shell "chmod 755 $D/nntrainer_causallm $D/unittest_hvx_dspq"
adb -s $S shell "cd $D && md5sum *" | tee $W/logs/md5_device.log     # must equal ../md5.txt, line by line
diff <(sort -k2 $W/md5.txt | tr -d '\r') <(sort -k2 $W/logs/md5_device.log | tr -d '\r') && echo MD5 OK
```

Config (as #134 / #136: greedy, `init_seq_len: 512`, `moe_engine: htp`,
`moe_htp_layers: ""`), re-applied unconditionally:

```
adb -s $S shell "cd $D/$M && \
  sed -i 's/\"do_sample\": true/\"do_sample\": false/' generation_config.json && \
  sed -i 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' nntr_config.json && \
  (grep -q moe_engine nntr_config.json || sed -i 's/\"bad_word_ids\": \[124900\],/\"bad_word_ids\": [124900],\n    \"moe_engine\": \"htp\",/' nntr_config.json) && \
  grep -H do_sample generation_config.json && grep -H -E 'bad_word_ids|num_to_generate|init_seq_len|_engine|_htp_layers' nntr_config.json"
```

Expected echo: `do_sample": false`, `bad_word_ids": [124900]`,
`init_seq_len": 512`, `moe_engine": "htp"`, `moe_htp_layers": ""`, no other
`_engine` key.

### 2. Sanity, one short run each (≈ 1 min)

```
run A 8 sanity_A prompt512.txt; run Q 8 sanity_Q prompt512.txt; run Q0 8 sanity_Q0 prompt512.txt
grep -c 'dspq' $W/logs/sanity_A.log                                               # 0
grep -h 'dspq:' $W/logs/sanity_Q.log $W/logs/sanity_Q0.log
```

Expected: A no `dspq` line; Q `[HTP] dspq: on queue=0x… dsp_spin_us=1000
arm_spin_us=5000 buffers=2x65536 ion=y` and `[HTP] dspq: close calls=N
served=N bad=0 …` with N a multiple of 22 (22 per decode step); Q0 the
same with `dsp_spin_us=0`. Anything else: stop and record it (G7).

### 3. tok/s, prompt 512 (≈ 8 min) — mirrored per G (step 7a of the plan)

```
for g in 64 512; do
  run A  $g A_G${g}_r1  prompt512.txt; run Q $g Q_G${g}_r1 prompt512.txt; run Q0 $g Q0_G${g}_r1 prompt512.txt
  run Q0 $g Q0_G${g}_r2 prompt512.txt; run Q $g Q_G${g}_r2 prompt512.txt; run A  $g A_G${g}_r2  prompt512.txt
  therm | tee -a $W/logs/therm.log    # checkpoints 1, 2
done
```

Expected in every log: `prefill: 512 tokens, … TPS`, `generation: <G>
tokens, … TPS`, `generation(last 64): 64 tokens, … TPS`, `peak memory`,
one `moe m1 gemv` banner, no `[HTP-PROFILE]`; the Q / Q0 banners as in
step 2, `calls` a multiple of 22 that grows with G.

### 4. Profile, not tok/s (≈ 1 min) — step 7b

```
run A 64 prof_A prompt512.txt NNTR_HTP_PROFILE=2
run Q 64 prof_Q prompt512.txt NNTR_HTP_PROFILE=2
grep -hE 'level=|K=2048  N=2048  M==1|staging:' $W/logs/prof_A.log $W/logs/prof_Q.log | cut -c1-400
```

Expected: `level=2 qos_mode=2` (`qos_mode=1` voids the profile), the M==1
row `K=2048  N=2048  M==1  calls=<n> … dsp= … transport= … mm …` (#120's
profile read n = 1408 at G = 64), and its `staging:` line — A
`non-ION in-args=6/…`, Q `via=dspq <n>/<n> calls msg=452 B`. Q's close
line must show `calls=<n>` as well.

### 5. Dumps, prompt 512, G = 64 (≈ 6 min, ≈ 0.8 GB on the phone) — step 7c

```
for v in A1 Q Q0 A2; do
  adb -s $S shell "rm -rf $DD/$v && mkdir -p $DD/$v"
  run ${v%[12]} 64 dump_$v prompt512.txt NNTR_HTP_DUMP=$DD/$v
  rm -rf $W/dump/$v && adb -s $S pull $DD/$v $W/dump/$v > /dev/null && adb -s $S shell "rm -rf $DD/$v"
done
therm | tee -a $W/logs/therm.log      # checkpoint 3
E="python3 tools/htp/htp_dump_eval.py"
$E --label a2 $W/dump/A1 $W/dump/A2 | tail -1      # null check: must be bit_identical=1
$E --label q  $W/dump/A1 $W/dump/Q  | tail -1      # G1
$E --label q0 $W/dump/A1 $W/dump/Q0 | tail -1      # G1
wc -l $W/dump/A1/manifest.txt                       # 22 prefill rows + the decode rows (A1 = Q = Q0)
awk '$2=="moe_layer" && $3==1 && $7==0' $W/dump/Q/manifest.txt | wc -l   # = dump_Q.log's close calls=
```

`a2` not `=1` means the sitting cannot judge G1 (A is not bit-stable run to
run): record it and go on. `q` / `q0` `bit_identical=0` is a defect: record
the `first_diff` and stop before step 6.

### 6. Text set, 8 prompts at G = 64 (≈ 7 min) — step 7d

```
P="prompt512.txt bitset-02-code.txt bitset-03-math.txt bitset-04-korean.txt bitset-05-json.txt bitset-06-dialogue.txt bitset-07-facts.txt bitset-08-short.txt"
i=0; for p in $P; do i=$((i+1)); run A 64 text_A_p0$i $p; run Q 64 text_Q_p0$i $p; done
therm | tee -a $W/logs/therm.log      # checkpoint 4
for i in 1 2 3 4 5 6 7 8; do printf 'p0%s: ' $i
  cmp -s <(strip $W/logs/text_A_p0$i.log) <(strip $W/logs/text_Q_p0$i.log) && echo identical || echo DIFFERENT; done
```

Every `prefill:` line must show the prompt's count from the prompt README
(512 / 207 / 117 / 326 / 207 / 276 / 402 / 24; a different count means the
app tokenized differently — record it, the A-vs-Q comparison still stands).

### 7. Checks on the workstation (1 min)

```
grep -c 'dspq' $W/logs/A_G*_r*.log $W/logs/text_A_*.log $W/logs/dump_A?.log                      # 0 each
grep -h 'dspq: on' $W/logs/Q_G*_r*.log $W/logs/Q0_G*_r*.log $W/logs/text_Q_*.log | sed 's/queue=0x[0-9a-f]*//' | sort | uniq -c
grep -h 'dspq: close' $W/logs/Q*_G*_r*.log $W/logs/text_Q_*.log $W/logs/dump_Q*.log                # served = calls, bad=0
grep -l 'dspq: off' $W/logs/*.log                                                                 # nothing
grep -l 'HTP-PROFILE' $W/logs/[AQ]*_G*_r*.log                                                     # nothing
for v in Q Q0; do for g in 64 512; do for r in 1 2; do printf '%s G=%s r%s vs A: ' $v $g $r
  cmp -s <(strip $W/logs/A_G${g}_r$r.log) <(strip $W/logs/${v}_G${g}_r$r.log) && echo identical || echo DIFFERENT
done; done; done                                                                                  # G3
cat $W/logs/therm.log
```

7f (optional, 1 min, not a gate): #147's bench on the moved `HtpDspqApi`,
`adb -s $S shell "cd $D && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_dspq" 2>&1 | tee $W/logs/gtest_dspq.log | grep -E 'DSPQ_BENCH|PASSED|FAILED'`;
expected rows as #147's (`bad=0`, `verdict … rule=adopt`).

Then fill the tables below, commit this file on the branch, push, and set
#141 to `state:measured`.

## Gates (plan 141-dspq-moe.md section 1)

| # | check | pass |
|---|---|---|
| G1 | `q` and `q0` dump evals vs A1 (step 5) | `bit_identical=1` on every file of A1's manifest; the null `a2` must be `=1` |
| G2 | text, 8 prompts, G = 64, Q vs A (step 6) | 8/8 identical |
| G3 | text in every tok/s cell, Q and Q0 vs A of the same G and run (step 7) | 8/8 identical |
| G4 | decode tok/s (all), G 64 and 512, mirrored means | Q mean > A mean at both G by more than A's own r1/r2 spread (expected ≈ +5 %) |
| G5 | M==1 row of the level-2 profile (step 4) | Q `transport=` ≤ 35 µs/call (A ≈ 82–92); Q `dsp=` and `mm` within ±3 % of A's |
| G6 | prefill tok/s, per mirrored pair | Q ≥ 0.95 × A at each G |
| G7 | banners (steps 2, 7) | `dspq: on` once per Q / Q0 log; `close calls=N served=N bad=0`, N > 0; A no `dspq` line |

The default stays **off** after this sitting; flipping it is a user
decision once G1–G7 pass.

## Results (fill in)

Unit: ______ , date / time: ______ , device `md5sum` = the table: ___

### tok/s (prompt 512; mirrored: A Q Q0 run 1, Q0 Q A run 2)

| variant | G | run | prefill tok/s | decode tok/s (all) | decode tok/s (last 64) | peak RSS KB | text = A? | dspq close line |
|---|---|---|---|---|---|---|---|---|
| A | 64 | 1 | | | | | (reference) | (none) |
| Q | 64 | 1 | | | | | | |
| Q0 | 64 | 1 | | | | | | |
| Q0 | 64 | 2 | | | | | | |
| Q | 64 | 2 | | | | | | |
| A | 64 | 2 | | | | | (reference) | (none) |
| A | 512 | 1 | | | | | (reference) | (none) |
| Q | 512 | 1 | | | | | | |
| Q0 | 512 | 1 | | | | | | |
| Q0 | 512 | 2 | | | | | | |
| Q | 512 | 2 | | | | | | |
| A | 512 | 2 | | | | | (reference) | (none) |

Reference (#134 sitting, same unit, A = this tree's switch-off path):
decode 36.90 (G=64) / 36.45 (G=512) tok/s, prefill 536–565 / 419–541;
CPU now 52.43 / 49.22 (#94 s2). Goal ≥ 50, prefill ≥ −5 % of this sitting's A.
Expected here: Q ≈ +5 % decode (−1.3 to −1.6 ms/token of 27.4).

| G | A mean | Q mean (vs A) | Q0 mean (vs A) | A r1/r2 spread | G4 | G6 |
|---|---|---|---|---|---|---|
| 64 | | | | | | |
| 512 | | | | | | |

### Profile, M==1 row (G5)

| variant | calls | host us/call | dsp us/call | transport us/call | mm | staging line |
|---|---|---|---|---|---|---|
| A | | | | | | |
| Q | | | | | | |

### Dumps (G1)

```
(paste the three htp_dump_eval lines: a2, q, q0)
```

### Text set (G2), G = 64

| prompt | tokens (prefill line) | Q = A? |
|---|---|---|
| p01 `prompt512.txt` | | |
| p02 `bitset-02-code.txt` | | |
| p03 `bitset-03-math.txt` | | |
| p04 `bitset-04-korean.txt` | | |
| p05 `bitset-05-json.txt` | | |
| p06 `bitset-06-dialogue.txt` | | |
| p07 `bitset-07-facts.txt` | | |
| p08 `bitset-08-short.txt` | | |

A's p01 text, for the record (no approval step when every text is identical):

```
(paste strip text_A_p01.log's generated part)
```

## Notes from the run

Thermal checkpoints (0 start, 1 after G=64, 2 after G=512, 3 after the
dumps, 4 after the text set) — the Q vs Q0 power proxy:

```
(paste therm.log)
```

Close-line `empty_polls` of Q vs Q0 (the DSP spin window's cost), FARF /
AEE errors, anything stale.

## Results

Run 2026-09-28 21:2x–21:5x KST by the orchestrator (user's request), unit
`R3CY10WM83Y` (SM-S938N; a second, unauthorized device `R3CN80CW3FY` was on
USB and never addressed). Device `md5sum` = `md5.txt` (MD5 OK). Thermal
(battery °C·10 / zone0 m°C): 265 / 27400 start, 316 / 58700 after G=64,
338 / 59000 after G=512, 337 / 45900 after the dumps, 343 / 56000 after the
text set.

### Gates

| # | result | pass |
|---|---|---|
| G1 | `a2 files=2862 bit_identical=1`; `q files=2862 bit_identical=1`; `q0 files=2862 bit_identical=1` (manifest 1431 rows) | **yes** |
| G2 | 8/8 prompts identical, Q vs A at G=64 (prefill counts 512 / 207 / 117 / 326 / 207 / 276 / 402 / 24 as the README) | **yes** |
| G3 | 8/8 tok/s cells identical, Q and Q0 vs A | **yes** |
| G4 | see the tok/s table: Q − A = +2.62 (G=64) and +1.79 tok/s (G=512), A's own r1/r2 spread 0.04 and 1.51 | **yes** |
| G5 | M==1 row: A `transport=87.9 µs/call`, `dsp=709.4`, `mm 674.9`; Q `transport=14.1`, `dsp=700.0` (−1.3 %), `mm 673.7` (−0.2 %); Q staging `via=dspq 1408/1408 calls msg=452 B` | **yes** |
| — | every Q / Q0 log: `dspq: on` once, `dspq: close calls=N served=N bad=0 stop_err=0x0`, N = 176 / 1408 / 11264 at G = 8 / 64 / 512; no `dspq: off`; no `dspq` line in any A log | **yes** |

### tok/s (prompt 512; mirrored A Q Q0 | Q0 Q A)

| variant | G | prefill r1 / r2 | decode r1 / r2 | decode mean | vs A |
|---|---|---|---|---|---|
| A | 64 | 559.0 / 540.1 | 36.80 / 36.76 | **36.78** | — |
| Q | 64 | 560.8 / 545.3 | 39.48 / 39.31 | **39.40** | **+7.1 %** |
| Q0 | 64 | 537.8 / 542.4 | 38.16 / 38.93 | 38.55 | +4.8 % |
| A | 512 | 528.4 / 406.7 | 36.74 / 35.23 | **35.98** | — |
| Q | 512 | 483.5 / 436.5 | 38.55 / 36.99 | **37.77** | **+5.0 %** |
| Q0 | 512 | 460.0 / 420.4 | 36.47 / 35.81 | 36.14 | +0.4 % |

Q (DSP spins up to 1 ms between calls) takes the whole transport win; Q0
(DSP blocks at once) keeps only part of it at G=64 and none at G=512. Prefill
is untouched by construction (M>1 stays on FastRPC) and follows the phone's
temperature in every variant.

Profiles (not tok/s): `prof_A` decode 34.92, `prof_Q` 38.23 tok/s.
