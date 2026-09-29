# Measurement 158: the MoE weight DMA with `src_bypass=1` in the app (A vs `NNTR_MOE_DMA_BYPASS=1`)

Branch `htp/158-dma-settings`, code @ `0106622f` (the set below was built
from that tree; later commits on the branch are docs only). Issue #158,
PR #159. Estimated device time: **≈ 30 min**. Run by the orchestrator on
the workstation (the phone stays on USB).

## Why

1. PR #159's probe read DDR→VTCM DMA at **69.3 GB/s with the descriptor's
   `src_bypass=1` vs 37.3 GB/s without it** (fresh 168 MiB, tag-validated,
   two runs, `R3CY10WM83Y`). The in-app feed runs at 32–34 GB/s (#117 / #120
   ring line) and the MoE wait is 16.1 of 26.2 ms per token (#150). This
   sitting checks whether the gain survives in the app: M==1 `mm`, the ring
   line `engine … GB/s`, decode tok/s, and the M>1 (prefill) row. The knob
   covers both paths.
2. The bit must not change a byte. B must give `bit_identical=1` MoE dumps
   against A, byte-identical `[PPL] decode` lines, and the same text on the
   8 prompts. If those gates pass and decode is up at every G, a follow-up
   PR makes the bit the default.

Rough prediction, not a gate: 22 MB per call at ≈ 60 GB/s ≈ 370 µs, with
the ≈ 240 µs of arithmetic hidden under it. That gives M==1 `mm` ≈ 400 µs
(A ≈ 675), ≈ 6 ms less per token, and decode ≈ 45–50 tok/s at G=512
(A ≈ 38).

### Variants (2; one binary set, one skel)

| | env (beyond `NNTR_NUM_THREADS=8`) | expected banner (once per log) |
|---|---|---|
| **A** (reference, first) | none | `[HTP] moe m1 gemv: on (applied=0x303e1) lead=192KB rows1=1 feed=vtcm dma_bypass=0 source=default` |
| **B** | `NNTR_MOE_DMA_BYPASS=1` | `[HTP] moe m1 gemv: on (applied=0x703e1) lead=192KB rows1=1 feed=vtcm dma_bypass=1 source=default` |

With the knob off, A runs the `htp_moe` decode path byte for byte. The only
other change is `NNTR_OP_TIME` (#150 / PR #153, merged into this branch),
which is off unless it is set, and inert when on (#150: dumps
`bit_identical=1`, nll identical). Every log of both variants must print
`[HTP] dspq: on queue=0x… dsp_spin_us=1000 arm_spin_us=5000
buffers=2x65536 ion=y` **once**, and `[HTP] dspq: close calls=N served=N
bad=0 …` with N = 22 × G (rule 40). The following logs are **void**
(rule 36): a log without the `on` line, a log with `dspq: off (…)`, and a
B log whose banner reads `dma_bypass=0` or whose `applied` word is not
`0x703e1`. A skel that predates bit 18 makes B throw `nntr_hvx_moe_set_opts
failed … applied=0x303e1`. That means a stale skel: stop.

What the bit covers: every expert weight descriptor of a MoE call whose
slot is arena-backed. That is the M=1 feed's 8 whole-matrix pushes and the
M>1 HMX path's gate_up / down chunks. It does not cover the activation
blocks, the two staging copies, the conv block, or the dense FC path.

Rule 15: `NNTR_HTP_PROFILE` is set only in step 4, never in a tok/s cell.
Rule 1: the set is not a `--profile` binary.

## Artifacts (`/local/mnt/workspace/htp_moe/158b/`, `md5.txt` next to them)

| file | md5 | built with |
|---|---|---|
| `libnntr_hvx_skel.so` | `3af9b127759b433382b8922ae6651fb5` | `test/htp/build.sh` @ `0106622f` (v79, HexKL 6.4.0.1; `UNDEFINED SYMBOLS OK (51 runtime imports)`). New: bit 18 in `HEXKL_MOE_FLAGS_KNOWN`. The skel md5 is not reproducible build to build (three builds of the same tree gave three md5s), so check this file's md5, not a rebuild's |
| `nntrainer_causallm` | `9159fd9e5f40f7832b42e083c6e44303` | `build_android.sh --htp`, then `--htp --cache` after `ninja install` @ `0106622f` (`jni/libs/arm64-v8a/`) |
| `libcausallm_core.so` | `feb911c3de0971951d9de3e081b50cc7` | same (`NNTR_HTP_FORWARD_KINDS` count 2, `OP-TIME` count 2) |
| `libnntrainer.so` | `9d8553404b41a720c6a1f2bd023cf2bd` | same (`jni/obj/local/arm64-v8a/`; NEEDED `libsdkl.so`, `libcdsprpc.so`; `dspq: on` 1, `dma_bypass=` 1, `NNTR_MOE_DMA_BYPASS` 1, `OP-TIME` 1, `graph: forward calls` 1) |
| `libccapi-nntrainer.so` | `3b5ac11b187aa1c9dfce511bac0b4228` | same (`jni/obj/local/arm64-v8a/`) |
| `libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 (from `150/set/`) |
| `libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL 6.4.0.1 (from `150/set/`) |
| `prompt512.txt` (p01) | `fc65c1588dc66dd764c7013fe96cbb75` | `docs/measurements/77-prompt512.txt`, 512 tokens |
| `bitset-02-code.txt` … `bitset-08-short.txt` (p02–p08) | see `md5.txt` / `docs/measurements/prompts/README.md` | 207 / 117 / 326 / 207 / 276 / 402 / 24 tokens (from `141b/set/`) |
| model `q40-qs4cx-wh`, `tokenizer.json` | `7b7867fab5…` / `7b8067a5…` | on the phone since #100 |

Not staged, never pushed: `builddir/.../libcdsprpc.so` (rule 5).

Workstation sanity before pushing:

```
W=/local/mnt/workspace/htp_moe/158b
(cd $W && LC_ALL=C md5sum -c md5.txt | grep -vc ': OK$')                    # 0
strings $W/libnntrainer.so | grep -c 'dspq: on'                              # 1
strings $W/libnntrainer.so | grep -c 'dma_bypass='                           # 1
strings $W/libcausallm_core.so | grep -c NNTR_HTP_FORWARD_KINDS              # 2 (rule 36)
strings $W/nntrainer_causallm | grep -c 'per-layer-type totals'              # 0 (no profile binary)
find $W -name 'libcdsprpc*' | wc -l                                          # 0
```

Rebuild recipe if `$W` is not on the measuring workstation: `git checkout
0106622f`, `source tools/htp/env.sh`, `export HEXKL_ROOT=…` explicitly,
`git submodule update --init --depth 1`, copy
`Applications/CausalLM/lib/libtokenizers_android_c.a` from another worktree,
`./test/htp/build.sh`, `(cd Applications/CausalLM && ./build_android.sh
--htp)`. On a fresh `builddir`, first run `cd builddir && meson configure
-Dprefix=$PWD/android_build_result && ninja install`, then `--htp --cache`.
Take `libc++_shared.so` / `libsdkl.so` from `150/set/`. A rebuilt set never
matches the table: record the md5s you push.

## Steps (workstation, phone on USB)

Shell setup once. Every `adb` names the serial, because a second device may
be on USB. The run dir is `s158` and is cleaned first.

```
cd /home/j2z0-lee/nntrainer-158 && git fetch -q && git checkout htp/158-dma-settings && source tools/htp/env.sh
W=/local/mnt/workspace/htp_moe/158b; L=$W/logs; mkdir -p $L $W/dump
S=R3CY10WM83Y      # adb devices: record the serial you use under Notes
D=/data/local/tmp/nntrainer/causallm/s158; M=/data/local/tmp/nntrainer/causallm/models/q40-qs4cx-wh; DD=/data/local/tmp/s158dump
therm() { adb -s $S shell dumpsys battery | grep -E 'level|temperature'; adb -s $S shell cat /sys/class/thermal/thermal_zone0/temp; }
env_of() { case $1 in A*) echo "";; B) echo "NNTR_MOE_DMA_BYPASS=1";; esac; }
run() { # run <A|A1|A2|B> <G> <log name> <prompt file> [extra env]
  adb -s $S shell "cd $D && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $2/' $M/nntr_config.json && \
    grep num_to_generate $M/nntr_config.json && md5sum libnntr_hvx_skel.so nntrainer_causallm libnntrainer.so && \
    NNTR_NUM_THREADS=8 $(env_of $1) $5 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm $M \"\$(cat $D/$4)\"" \
    2>&1 | tee $L/$3.log | grep -E '^(prefill|generation|total|peak memory)|dspq:|moe m1 gemv|pinning|\[OP-TIME\] (step|moe)|libnntr_hvx_skel|nntrainer_causallm$|libnntrainer.so$'
}
# The generated text of a log: everything before the summary, minus the
# banners and any [OP-TIME] / [PPL] line stderr put inside the streamed text.
strip() { sed -n '/^=====/q;p' "$1" | perl -0pe 's/\[HTP\] dspq: [^\n]*\n//g; s/\[OP-TIME\][^\n]*\n//g; s/\[PPL\][^\n]*\n//g' |
  grep -v 'moe m1 gemv\|libnntr_hvx_skel\|nntrainer_causallm\|libnntrainer.so\|num_to_generate'; }
```

### 0. Device state (1 min)

```
adb devices                           # note every serial listed
therm | tee -a $L/therm.log           # checkpoint 0 (battery %, °C·10, zone0 m°C)
```

Screen off, charger in, cool start (the probe's 69 GB/s was read from a
25 °C start).

### 1. Install (≈ 2 min). The model is reused.

```
adb -s $S shell ls -l $M/nntr_lfm2_8b_a1b_q40_arm.bin                        # 4316133120
adb -s $S shell "rm -rf $D $DD && mkdir -p $D $DD"
adb -s $S push $(cd $W && ls -p | grep -v / | grep -v md5.txt | sed "s|^|$W/|") $D/ > /dev/null   # the 15 files of md5.txt
adb -s $S shell "chmod 755 $D/nntrainer_causallm"
adb -s $S shell "cd $D && md5sum \$(ls -p | grep -v / | sort)" | tee $L/md5_device.log
diff <(sort -k2 $W/md5.txt | tr -d '\r') <(sort -k2 $L/md5_device.log | tr -d '\r') && echo MD5 OK
```

Config, as in #141 / #150: greedy, `init_seq_len: 512`, `moe_engine: htp`,
`moe_htp_layers: ""`. Re-apply it unconditionally:

```
adb -s $S shell "cd $M && \
  sed -i 's/\"do_sample\": true/\"do_sample\": false/' generation_config.json && \
  sed -i 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' nntr_config.json && \
  (grep -q moe_engine nntr_config.json || sed -i 's/\"bad_word_ids\": \[124900\],/\"bad_word_ids\": [124900],\n    \"moe_engine\": \"htp\",/' nntr_config.json) && \
  grep -H do_sample generation_config.json && grep -H -E 'bad_word_ids|num_to_generate|init_seq_len|_engine|_htp_layers' nntr_config.json"
```

Expected echo: `do_sample": false`, `bad_word_ids": [124900]`,
`init_seq_len": 512`, `moe_engine": "htp"`, `moe_htp_layers": ""`, no other
`_engine` key.

### 2. Sanity at G=8 (≈ 1 min)

```
run A 8 sanity_A prompt512.txt; run B 8 sanity_B prompt512.txt
grep -h 'moe m1 gemv' $L/sanity_A.log $L/sanity_B.log
grep -c 'dspq: on' $L/sanity_A.log $L/sanity_B.log                          # 1 each
```

Expected: A's banner reads `applied=0x303e1 … dma_bypass=0 source=default`
and B's reads `applied=0x703e1 … dma_bypass=1 source=default`. Both show
`dspq: close calls=176 served=176 bad=0`. A B run that throws
`nntr_hvx_moe_set_opts failed` means a stale skel (the device md5 is not
`3af9b127…`): stop.

### 3. tok/s, prompt 512, G 64 / 512 / 1024, mirrored per G (≈ 8 min)

`NNTR_OP_TIME=1` rides in every cell (inert per #150), so step 6 gets the
per-op tables of A and B from the same runs.

```
for g in 64 512 1024; do
  run A $g A_G${g}_r1 prompt512.txt NNTR_OP_TIME=1; run B $g B_G${g}_r1 prompt512.txt NNTR_OP_TIME=1
  run B $g B_G${g}_r2 prompt512.txt NNTR_OP_TIME=1; run A $g A_G${g}_r2 prompt512.txt NNTR_OP_TIME=1
  therm | tee -a $L/therm.log         # checkpoints 1, 2, 3
done
```

Expected in every log: `prefill: 512 tokens, … TPS`, `generation: <G>
tokens, … TPS`, `generation(last 64): 64 tokens, … TPS`, `peak memory`, one
`moe m1 gemv` banner as in step 2, `dspq: on` once, `dspq: close calls=22·G
served=22·G bad=0`, `[OP-TIME] step tokens=<G> …` and `[OP-TIME] moe
calls=22·G …`, and no `[HTP-PROFILE]`.

### 4. Level-2 profile, not tok/s (≈ 1 min)

```
run A 64 prof_A prompt512.txt NNTR_HTP_PROFILE=2
run B 64 prof_B prompt512.txt NNTR_HTP_PROFILE=2
for v in A B; do echo "== $v"; grep -E 'level=|  M==1  |  M>1  |weight DMA:|DMA ring:' $L/prof_$v.log | grep -v 'M>1 \(dense\|conv\|FC\)' | cut -c1-420; done
```

Expected: `level=2 qos_mode=2`. `qos_mode=1` voids the profile. The
`K=2048  N=2048  M==1` row is `calls=1408 … blocks=0 m1_gemv=1408/1408
feed=1408/1408`, and under it `weight DMA: 21504 KB/call …` and `DMA ring:
desc=10/call … busy=…..… us -> engine …..… GB/s …`. A's `mm` should read
≈ 675 µs and its engine ≈ 32–34 GB/s (#120 / #141). **B's engine bracket
and `mm` are the in-app answer.** A B engine above 85.3 GB/s is a
bracket artefact, not a rate: record it as such. The `M>1` row (the 22
prefill calls, `m1_gemv=0/22 feed=0/22`) gives the prefill side: its
`dsp=`, `mm` and `weight DMA: … GB/s` for A vs B. The ring line prints
`busy=n/a` above 512 pushes.

### 5. Dumps (G=64) and decode nll lines (G=512), prompt 512 (≈ 7 min)

Dumps take ≈ 0.8 GB per variant on the phone. They are pulled and deleted
one at a time.

```
for v in A1 B A2; do
  adb -s $S shell "rm -rf $DD/$v && mkdir -p $DD/$v"
  run ${v%[12]} 64 dump_$v prompt512.txt NNTR_HTP_DUMP=$DD/$v
  rm -rf $W/dump/$v && adb -s $S pull $DD/$v $W/dump/$v > /dev/null && adb -s $S shell "rm -rf $DD/$v"
done
E="python3 tools/htp/htp_dump_eval.py"
$E --label a2 $W/dump/A1 $W/dump/A2 | tail -1      # null check: must be bit_identical=1
$E --label b  $W/dump/A1 $W/dump/B  | tail -1      # the gate: bit_identical=1
# PPL: A's first run writes its own continuation; B and A once more are forced on it.
adb -s $S shell "rm -f $DD/cont.ids"
run A 512 ppl_A  prompt512.txt NNTR_PPL_DECODE=$DD/cont.ids
run B 512 ppl_B  prompt512.txt NNTR_PPL_DECODE=$DD/cont.ids
run A 512 ppl_A2 prompt512.txt NNTR_PPL_DECODE=$DD/cont.ids
therm | tee -a $L/therm.log         # checkpoint 4
nll() { grep -o '\[PPL\] decode step=.*' "$1"; }
nll $L/ppl_A.log | wc -l                                                     # one line per decode step (≈ 512)
cmp <(nll $L/ppl_A.log) <(nll $L/ppl_A2.log) && echo "nll A2 == A"
cmp <(nll $L/ppl_A.log) <(nll $L/ppl_B.log)  && echo "nll B == A"
grep -h '\[PPL\] decode tokens=' $L/ppl_A.log $L/ppl_B.log $L/ppl_A2.log
```

If `a2` is not `=1`, the sitting cannot judge the bit gate, because A is
not bit-stable run to run. Record it and go on. If `b` prints
`bit_identical=0`, or an nll line differs, the knob is defective. Record
the `first_diff` and the first differing line, then run step 6 anyway,
but do not fold the tok/s.

### 6. Text set, 8 prompts at G=64 (≈ 6 min)

```
P="prompt512.txt bitset-02-code.txt bitset-03-math.txt bitset-04-korean.txt bitset-05-json.txt bitset-06-dialogue.txt bitset-07-facts.txt bitset-08-short.txt"
i=0; for p in $P; do i=$((i+1)); run A 64 text_A_p0$i $p; run B 64 text_B_p0$i $p; done
therm | tee -a $L/therm.log         # checkpoint 5
for i in 1 2 3 4 5 6 7 8; do printf 'p0%s: ' $i
  cmp -s <(strip $L/text_A_p0$i.log) <(strip $L/text_B_p0$i.log) && echo identical || echo DIFFERENT; done
```

Every `prefill:` line must show the prompt's token count from the prompt
README (512 / 207 / 117 / 326 / 207 / 276 / 402 / 24).

### 7. Reports and checks on the workstation (1 min)

```
R="python3 tools/htp/op_time_report.py"
for g in 512 1024; do for r in 1 2; do $R $L/A_G${g}_r$r.log; $R $L/B_G${g}_r$r.log --base $L/A_G${g}_r$r.log; done; done | tee $L/report.txt
grep -H -E '^(prefill|generation)' $L/[AB]_G*_r?.log
grep -h 'moe m1 gemv' $L/*.log | sort | uniq -c               # two kinds only: 0x303e1 … dma_bypass=0 (A*), 0x703e1 … dma_bypass=1 (B*)
grep -L 'dma_bypass=1' $L/B_*.log $L/*_B.log $L/text_B_*.log  # nothing
grep -l 'dma_bypass=1' $L/A_*.log $L/*_A*.log $L/text_A_*.log # nothing
grep -c 'dspq: on' $L/*.log | grep -v ':1$'                    # nothing (rule 40: once in every log)
grep -l 'dspq: off' $L/*.log                                   # nothing
grep -h 'dspq: close' $L/[AB]_G*_r?.log | awk '{print $4, $5, $6}' | sort | uniq -c   # calls=served, bad=0
grep -l 'HTP-PROFILE' $L/[AB]_G*_r?.log                        # nothing
grep -h 'pinning' $L/*.log                                     # nothing, else record it
for g in 64 512 1024; do for r in 1 2; do printf 'G=%s r%s text B vs A: ' $g $r
  cmp -s <(strip $L/A_G${g}_r$r.log) <(strip $L/B_G${g}_r$r.log) && echo identical || echo DIFFERENT; done; done
cat $L/therm.log
```

Each report must print a table whose `total` equals the log's token time,
and exit 0. Then fill the tables below, commit this file on the branch,
push, and set #158 to `state:measured`.

## Gates

| # | check | pass |
|---|---|---|
| H1 | banners (steps 2, 7) | A: `applied=0x303e1 … dma_bypass=0 source=default`; B: `applied=0x703e1 … dma_bypass=1`; `dspq: on` once in every log; `close calls=N served=N bad=0`, N = 22 × G |
| H2 | MoE dumps (step 5) | B vs A1 `bit_identical=1` on every file; the null A2 vs A1 `=1` |
| H3 | decode nll lines (step 5) | `[PPL] decode step=…` byte-identical, B vs A and A2 vs A |
| H4 | text (steps 3, 6) | B ≡ A on all 8 prompts and in every tok/s cell |
| H5 | prefill (step 3) | B's mean prefill ≥ −5 % of A's at every G |
| H6 | decode (step 3), mirrored means | the lever: B vs A per G. Up ≥ 5 % at every G together with H1–H5 → a follow-up PR flips the default. Within ±5 % → #158 closes with the numbers |
| H7 | profile (step 4), read with H6 | B's M==1 ring `engine` bracket vs A's (the probe predicts ≈ 55–69 vs ≈ 33), `mm` vs A's; the M>1 row's `dsp` / `weight DMA` GB/s vs A's |

## Results (fill in)

Unit, date, KST window: …; device `md5sum` = `md5.txt`? …; thermal
checkpoints (battery °C·10 / zone0 m°C): 0 …, 1 …, 2 …, 3 …, 4 …, 5 ….

### tok/s (prompt 512; mirrored A B | B A per G)

| variant | G | prefill r1 / r2 | decode (all) r1 / r2 | decode mean | vs A | last 64 r1 / r2 | peak RSS | text ≡ A |
|---|---|---|---|---|---|---|---|---|
| A | 64 | | | | — | | | (reference) |
| B | 64 | | | | | | | |
| A | 512 | | | | — | | | (reference) |
| B | 512 | | | | | | | |
| A | 1024 | | | | — | | | (reference) |
| B | 1024 | | | | | | | |

Reference (not this sitting): #141 Q (the dspq default, `R3CY10WM83Y`)
39.40 / 37.77 decode at G 64 / 512; #150 A0 37.48 at G=512, prefill
455–540; #150 per-op: MoE wait 16.08 of 26.22 ms per token. Goal ≥ 50.

### Profile (step 4, G=64, level 2)

| row | field | A | B |
|---|---|---|---|
| M==1 | `dsp` µs/call | | |
| M==1 | `mm` µs/call | | |
| M==1 | `transport` µs/call | | |
| M==1 | `weight DMA:` first / average GB/s | | |
| M==1 | `DMA ring:` `busy` bracket µs, `engine` bracket GB/s | | |
| M==1 | `first expert ready at` / `last issue at` µs | | |
| M>1 | `dsp` µs/call | | |
| M>1 | `mm` µs/call | | |
| M>1 | `weight DMA:` GB/s | | |

### Accuracy (step 5, step 6)

| check | result |
|---|---|
| dumps A2 vs A1 (null) | |
| dumps B vs A1 | |
| nll A2 == A / B == A (lines) | |
| `[PPL] decode tokens=` A / B / A2 | |
| text p01–p08, B vs A | |
| text in the 12 tok/s cells | |

### Per-op decode (step 7, `op_time_report.py`, G=512 r1; B with `--base` A)

```
(paste A_G512_r1 and B_G512_r1 --base A_G512_r1)
```

## Notes from the run

<serials seen, thermal, anything stale or void>

## Decisions recorded with this handoff (implementer)

* **One knob for both paths.** The M>1 (prefill) HMX path's weight chunks
  honour `NNTR_MOE_DMA_BYPASS` too. Its weights are the same arena bytes,
  read once per call, so the L2 has nothing to offer them. The M>1
  profile row and the prefill column (H5) read that side. If B's prefill
  regresses while decode gains, the follow-up splits the bit (an M==1-only
  variant is a one-line change in `hexkl_mm_u8i4_moe_layer_run`).
* **Coherency.** `src_bypass` only skips the DSP L2. A bypassing read is
  stale only if the L2 holds a dirty line of the source. The bit is set
  only for arena-backed (borrowed) slots. The CPU fills those before
  `arena_attach`, and nothing on the DSP ever writes them. The GEMV arena
  read and `l2fetch` leave clean lines only. So no cache maintenance is
  needed. Heap slots (DSP `memcpy`), activation blocks (DSP-packed) and
  the staging copies keep `src_bypass=0`. **#157's cached ARM arena** is
  a separate matter: the ARM caches are not the DSP L2, so its clean per
  fill is needed with or without this bit. The bypass neither helps nor
  replaces it. If #157 ever relied on I/O coherency instead of an
  explicit clean, its variant C would have to be re-read under B.
* **No change to the in-situ chunk plan or the probe's host checks.**
  `nntr_moe_dma_plan.h` is test-only and already carries
  `NNTR_MOE_DMA_SRC_BYPASS`. The production path's bit is held by
  `moe_layer_host_check` (the bypassed bytes equal the arena weights'
  bytes exactly, on the M>1 path and on 80 M=1 cells, all output
  byte-compared) and by `moe_opts_host_check` (the word and the echo mask).

## Filled results (orchestrator, 2026-09-29 12:33–12:49 KST, `R3CY10WM83Y`)

Device md5 = md5.txt for all 15 staged files (the only diff line is an extra
`sitting.out` the orchestrator wrote into `$W` before the push — not an
artifact). Banners: A `applied=0x303e1 … dma_bypass=0`, B `applied=0x703e1 …
dma_bypass=1`; every log `dspq: on` once and `dspq: close calls=22·G
served=22·G bad=0`. Thermal: 33.1 °C / 39.7 at start, 34.8 / 61.7 at the end.

| G | A decode r1 / r2 | B decode r1 / r2 | A mean | B mean | B vs A | prefill A r1/r2 · B r1/r2 |
|---|---|---|---|---|---|---|
| 64 | 38.67 / 38.55 | 51.78 / 51.86 | 38.61 | **51.82** | **+34.2 %** | 512.5 / 502.0 · 518.2 / 500.5 |
| 512 | 38.10 / 36.69 | 50.66 / 50.56 | 37.40 | **50.61** | **+35.3 %** | 501.5 / 422.4 · 425.2 / 421.7 |
| 1024 | 36.69 / 35.65 | 47.56 / 47.15 | 36.17 | **47.36** | **+30.9 %** | 422.4 / 383.8 · 426.0 / 385.0 |

Prefill follows the phone's temperature in both variants (mirrored pairs
agree within ±1 % except the first cell); level-2 M>1 `dsp` 15127 → 14724
µs/call (−2.7 %).

**Profile (G=64, level 2).** M==1: `dsp` **726.0 → 417.5 µs/call (−42.5 %)**,
transport 12.6 / 13.1 µs; ring `engine 31.5..33.4 → 56.3..57.9 GB/s`,
`first expert ready at 123 → 77 µs`. M>1 ring `engine 16.9..26.7 →
25.9..73.6 GB/s`.

**Accuracy.** MoE dumps (G=64): `a2 files=2862 bit_identical=1`, **`b files=2862
bit_identical=1`**; decode nll lines (G=512): A2 == A, **B == A**; text
identical on **8/8 prompts** and in **6/6 tok/s cells**.

**Per-op (G=512 r1, B vs A):** MoE wait 16.07 → **9.85 ms** (−6.22), token
26.0 → **19.74 ms**; the CPU rows unchanged within noise.

**Verdict:** bit-preserving, all gates pass; decode ≥ 50 tok/s at G 64 / 512
on this unit for the first time. Making `NNTR_MOE_DMA_BYPASS=1` the default is
the user's decision (rule 33).
