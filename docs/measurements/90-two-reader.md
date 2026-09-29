# Measurement 90: do CPU and DSP DDR reads add up, and what does a CPU prefetch during the MoE call cost the DSP?

Branch `htp/90-two-reader`, probe code @ `cf6fba14` (`[test]` commit; later
commits on the branch are docs only). Plan
`docs/plans/90-two-reader-ddr-probe.md` §4.1–4.2. Estimated device time:
**≈ 20 min** (≈ 15 min straight after #150's sitting, see "Back to back
with #150"). One variant plus probe cells, one sitting, one unit.

## Why

1. Target (b): the 886 MB a decode token reads are split between the DSP
   ring (484 MB of MoE weights, ≈ 33 GB/s in the app) and the CPU (≈ 402
   MB of Q4_0 FC + lm_head), and the two take turns. This sitting measures
   whether the two readers add up when they run at once (the split form,
   rule S, parked behind track (c)) and what a light CPU touch of the next
   layer's FC weights during the MoE call costs the DSP's `mm` and how much
   of it is still cached when the consumer reads it (the bit-preserving
   form, rule P).
2. The supervisor applies plan §3.4 to the filled tables: P = "build the
   in-app prefetch variant" iff at S = 10 MiB and the best T `dmm_pct` ≤ 5,
   `staged` ≥ 0.5 and `net_ms_per_token` ≥ 1.0; S = three numbers for the
   user.

The probe is `unittest_hvx_dma_probe` rebuilt from this branch; it rides
the staged #150 skel (no DSP source or IDL change). The E2E control A is
the #150 T set unchanged, so A also ties this sitting to #150's.

### Variant

| | set on the phone | env (beyond `NNTR_NUM_THREADS=8`) | expected banners |
|---|---|---|---|
| **A** (control, first E2E) | `$D/` = the #150 T set (code `56ba0835`, dspq default on) | `NNTR_OP_TIME=1` | `[HTP] dspq: on …` once; `[HTP] dspq: close calls=N served=N bad=0 …` with N = 22 × G; `[OP-TIME] step` / `moe` / `node` lines at the end |

A log without the `dspq: on` line, or with `dspq: off (…)`, is **void**
(rules 36 / 40), never read as "at A's speed". `dspq: off (dspq_start …
0x8000040e)` or `err=0x8000040e` in the probe means a stale skel (rule 3):
stop. No cell sets `NNTR_HTP_PROFILE` (rule 15). No `--profile` binary
(rule 1).

## Artifacts (`/local/mnt/workspace/htp_moe/90/set/`, `md5.txt` next to it)

| file | md5 | built with |
|---|---|---|
| `unittest_hvx_dma_probe` | `f0e448eed87cd0f7c442462dcf6a9e5f` | this branch @ `cf6fba14`: `(cd test/jni && $ANDROID_NDK/ndk-build … unittest_hvx_dma_probe -j8)`, NDK r30, stub regenerated from `test/htp/nntr_hvx.idl` by `test/htp/build.sh` (IDL unchanged since `df17fcdb`; the diff to the staged skel's `07fb1938` IDL is one comment) |
| `libnntr_hvx_skel.so` | `37468a7fdbf2e469589849598ca860ff` | #141b's (`07fb1938`), from `150/set/`. **Not rebuilt**: the branch's skel sources are unchanged; `nntr_moe_dma_plan.h` gained unused `static inline` helpers only |
| `nntrainer_causallm` | `db0c4bc3129ef04ca9ec17d88d3092b7` | #150 T set, `build_android.sh --htp --cache` @ `56ba0835`, from `150/set/` |
| `libcausallm_core.so` | `44b60ecdc2a773048155864298aaea03` | same |
| `libnntrainer.so` | `178b6e126e3a2aac8d1d83ad2f658250` | same (`dspq: on` count 1) |
| `libccapi-nntrainer.so` | `e3f0a1243fbf28d8109e019e092288fc` | same |
| `libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 |
| `libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL 6.4.0.1 |
| `prompt512.txt` | `fc65c1588dc66dd764c7013fe96cbb75` | `docs/measurements/77-prompt512.txt`, 512 tokens |
| model `q40-qs4cx-wh` (`nntr_lfm2_8b_a1b_q40_arm.bin`, 4316133120 B), `tokenizer.json` | on the phone since #100 | NPU model |

Workstation sanity before pushing (done at staging: 0 / 1 / 0):

```
W=/local/mnt/workspace/htp_moe/90
(cd $W/set && LC_ALL=C md5sum -c ../md5.txt | grep -vc ': OK$')    # 0
strings $W/set/libnntrainer.so | grep -c 'dspq: on'                 # 1
find $W -name 'libcdsprpc*' | wc -l                                 # 0
```

Rebuild recipe for the probe if `$W` is not on the measuring workstation:
`git checkout cf6fba14`, `source tools/htp/env.sh`, `export HEXKL_ROOT=…`
explicitly, `git submodule update --init --depth 1 subprojects/googletest`,
`ln -sfn $PWD/subprojects/googletest/googletest test/jni/googletest`,
`./test/htp/build.sh` (regenerates `test/htp/generated/nntr_hvx_stub.c`),
then a `builddir/jni/arm64-v8a/` holding any `libnntrainer.so` /
`libccapi-nntrainer.so` (the module does not link them, `Android.mk` only
requires the prebuilt files to exist; this one was built with copies from
`/home/j2z0-lee/nntrainer-150/builddir/jni/arm64-v8a/`), and the ndk-build
line above. The app set is `/local/mnt/workspace/htp_moe/150/set/` (rebuild
recipe in `150-cpu-decode.md` on `htp/150-cpu-decode`). A rebuilt binary
never matches the table: record the md5s you push.

## Back to back with #150

Both sittings use the clean run dir `/data/local/tmp/nntrainer/causallm/s150`
and the same unit. They **can run back to back in either order**:

* The #90 set is the #150 T set (byte-identical, same md5s) plus
  `unittest_hvx_dma_probe`. **After #150**: skip the `rm -rf` in step 1,
  push only the probe (and `prompt512.txt` is already there), then check
  the device md5s. **Before #150**: #150's step 1 wipes `$D` and pushes its
  own set (with `a0/`); nothing of #90 is left behind that it reads.
* The model config (`num_to_generate`) is set by every run; the #141 config
  block is re-applied here unconditionally.
* Thermal: the cold probe must start cool. After #150 (≈ 25 min of load),
  let `zone0` drop to within ≈ 3 °C of #150's checkpoint 0 before step 2,
  and write both readings under Notes. If it does not, run anyway and mark
  the cold probe "warm start".

## Steps (workstation, phone on USB)

Shell setup once. Every `adb` names the serial. Logs go to `$W/logs/`.

```
cd /home/j2z0-lee/nntrainer-90 && git fetch -q && git checkout htp/90-two-reader && source tools/htp/env.sh
W=/local/mnt/workspace/htp_moe/90; L=$W/logs; mkdir -p $L
S=R3CY10WM83Y      # adb devices: record the serial you use under Notes
D=/data/local/tmp/nntrainer/causallm/s150; M=/data/local/tmp/nntrainer/causallm/models/q40-qs4cx-wh
therm() { adb -s $S shell dumpsys battery | grep -E 'level|temperature'; adb -s $S shell cat /sys/class/thermal/thermal_zone0/temp; }
probe() { # probe <cold|warm>
  adb -s $S shell "cd $D && md5sum libnntr_hvx_skel.so unittest_hvx_dma_probe && \
    LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_dma_probe \
    --gtest_filter='*MoeChunkReplay*:*TwoReaderDdr*:*PrefetchOverlap*'" 2>&1 \
    | tee $L/probe_$1.log | grep -E 'md5|libnntr_hvx_skel|unittest_hvx_dma_probe$|^CPU_CACHE|^DDR_|^PREFETCH_|INVALID|DMA_REPLAY workers=1 load=0 pace=0|DMA_REPLAY_X name=f2 |PASSED|FAILED'; }
run() { # run <G> <log name>
  adb -s $S shell "cd $D && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $1/' $M/nntr_config.json && \
    grep num_to_generate $M/nntr_config.json && md5sum libnntr_hvx_skel.so nntrainer_causallm libnntrainer.so && \
    NNTR_NUM_THREADS=8 NNTR_OP_TIME=1 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=$D \
    ./nntrainer_causallm $M \"\$(cat $D/prompt512.txt)\"" \
    2>&1 | tee $L/$2.log | grep -E '^(prefill|generation|total|peak memory)|dspq:|moe m1 gemv|\[OP-TIME\] (step|moe)|libnntr_hvx_skel|nntrainer_causallm$|libnntrainer.so$'
}
# The generated text of a log (as #150): everything before the summary,
# minus the banners and any [OP-TIME] line stderr put inside the text.
strip() { sed -n '/^=====/q;p' "$1" | perl -0pe 's/\[HTP\] dspq: [^\n]*\n//g; s/\[OP-TIME\][^\n]*\n//g' |
  grep -v 'moe m1 gemv\|libnntr_hvx_skel\|nntrainer_causallm\|libnntrainer.so\|num_to_generate'; }
```

### 0. Device state (1 min)

```
adb devices                           # note every serial listed
therm | tee -a $L/therm.log           # checkpoint 0 (battery %, °C·10, zone0 m°C)
```

Screen off, charger in, cool start.

### 1. Install (≈ 3 min; ≈ 1 min after #150) — model reused

```
adb -s $S shell ls -l $M/nntr_lfm2_8b_a1b_q40_arm.bin                 # 4316133120
# fresh (not after #150):
adb -s $S shell "rm -rf $D && mkdir -p $D" && adb -s $S push $W/set/. $D/
# after #150 instead: adb -s $S push $W/set/unittest_hvx_dma_probe $D/
adb -s $S shell "chmod 755 $D/nntrainer_causallm $D/unittest_hvx_dma_probe"
adb -s $S shell "cd $D && md5sum $(cd $W/set && ls | tr '\n' ' ')" | tee $L/md5_device.log
diff <(sort -k2 $W/md5.txt | tr -d '\r') <(sort -k2 $L/md5_device.log | tr -d '\r') && echo MD5 OK
```

Config (as #141 / #150: greedy, `bad_word_ids [124900]`, `moe_engine: htp`),
re-applied unconditionally:

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

### 2. Probe, cold (≈ 1.5 min)

```
probe cold
grep -c INVALID $L/probe_cold.log                  # 0
grep -c '^DDR_TWO_READER' $L/probe_cold.log        # 6
grep -c '^DDR_CPU' $L/probe_cold.log               # 10 (4 alone + 6 with)
grep -c '^DDR_DSP ' $L/probe_cold.log              # 2 (reader=ring, reader=hvx)
grep -c '^PREFETCH_COLD' $L/probe_cold.log         # 4
grep -c '^PREFETCH_HOT' $L/probe_cold.log          # 12
grep -c '^PREFETCH_OVERLAP' $L/probe_cold.log      # 12
grep -E '\[  (PASSED|FAILED) ' $L/probe_cold.log   # PASSED 3 tests, no FAILED
therm | tee -a $L/therm.log                        # checkpoint 1
```

Tests run in the binary's order: TwoReaderDdr (≈ 25 s), PrefetchOverlap
(≈ 10 s), MoeChunkReplay (≈ 10 s), each after a ≈ 3 s fixture fill. The
first md5 lines must show the skel `37468a7f…` and the probe `f0e448ee…`.

Expected lines (numbers illustrative):

```
CPU_CACHE cpu=0 index=2 level=2 type=Unified size=12288K shared_cpu_list=0-5
DDR_CPU threads=8 phase=alone bytes=… s=1.50 gbs=… valid=y cpus=…
DDR_DSP_REF name=f2 fresh=0 calls=20 us=… bytes_per_call=22020096 gbs=… checksum_ok=y
DDR_DSP reader=ring bytes=… us=… gbs=… checksum_ok=y ref_gbs=… regions=32 footprint_mib=168 chunk_bytes=268435456 valid=y
DDR_DSP reader=hvx bytes=… us=… gbs=… checksum_ok=y dsp_us=… window_us=… duty=… sets=16 footprint_mib=336 chunk_bytes=268435456 valid=y
DDR_TWO_READER cpu_alone=… dsp_alone=… cpu_with=… dsp_with=… aggregate=… cpu_threads=8 chunk_bytes=268435456 reader=ring cpu_loss_pct=… dsp_loss_pct=… checksum_ok=y valid=y
PREFETCH_CONFIG readers=8 workers=7 ring_mib=200 sets=16 opts=0x303e1 chunk_bytes=268435456
PREFETCH_COLD size_mib=10 bufs=20 dsp_us=… mm_us=… cold_us=… cold_gbs=…
PREFETCH_HOT size_mib=10 touch_threads=2 touch_us=… hot_us=… moved=…
PREFETCH_OVERLAP size_mib=10 touch_threads=2 dsp_us=… mm_us=… mm_alone_us=… dmm_pct=… touch_us=… touch_gbs=… touch_late=k/64 cold_us=… hot_us=… reread_us=… staged=… staged_mib=… saved_us=… net_ms_per_token=… moved=…
```

`chunk_bytes=134217728` means the fixture fell back to 128 MiB chunks:
then `regions=23 footprint_mib=120`, `sets=10 footprint_mib=210`; record
it, the lines stay valid. **Stop conditions**: a skel md5 ≠ `37468a7f…`;
any `err=0x8000040e`. **Note and go on**: `DDR_DSP_REF` more than ±15 %
from the same log's anchor `DMA_REPLAY workers=1 load=0 pace=0 …` (rule
30: write which one drifted).

### 3. A, full E2E, prompt 512, G = 64 / 512 / 1024 (≈ 6 min)

```
for g in 64 512 1024; do run $g A_g$g; done
grep -c 'dspq: on' $L/A_g*.log                                   # 1 each (rule 40)
grep -h 'dspq: close' $L/A_g*.log                                # calls=1408/11264/22528 served=same bad=0
grep -l 'dspq: off\|HTP-PROFILE' $L/A_g*.log                     # nothing
python3 tools/htp/op_time_report.py $L/A_g512.log | tee $L/report_A_g512.txt   # exit 0
therm | tee -a $L/therm.log                                      # checkpoint 2
```

Expected in every log: `prefill: 512 tokens, … TPS`, `generation: G
tokens, … TPS`, `generation(last 64): 64 tokens, … TPS`, `peak memory`,
one `moe m1 gemv` banner, the md5sum lines showing `37468a7f…`,
`db0c4bc3…`, `178b6e12…`, and `[OP-TIME] step tokens=G …` /
`[OP-TIME] moe calls=22×G …` at the end. `op_time_report.py` is on
`htp/150-cpu-decode` (not on this branch): run it from
`/home/j2z0-lee/nntrainer-150` if this checkout lacks it.

Text against #150 (plan §1 standing gate), if #150's sitting has run:

```
L150=/local/mnt/workspace/htp_moe/150/logs1b
cmp -s <(strip $L/A_g512.log) <(strip $L150/T8_G512_r1.log) && echo "A g512 == #150 T8" || echo DIFFERENT
strip $L/A_g512.log | md5sum; strip $L150/T8_G512_r1.log | md5sum
grep -h '^prefill:' $L/A_g*.log $L150/A0_G512_r?.log           # A within -5 % of #150 A0's mean
```

If #150 has not run yet, record `strip $L/A_g{64,512,1024}.log | md5sum`
for the supervisor to compare later.

### 4. Probe, warm (≈ 1.5 min)

```
probe warm
# the same seven checks as step 2 on $L/probe_warm.log
therm | tee -a $L/therm.log                        # checkpoint 3
```

### 5. Finish

Pull nothing else. Record the serial, battery and temperatures under
Notes, fill the tables below, commit this file on the branch, push, and
set #90 to `state:measured`.

## Gates (plan §1)

| # | read from | pass |
|---|---|---|
| G1 readers alone | `DDR_CPU … phase=alone`, `DDR_DSP reader=ring`, `DDR_DSP reader=hvx` | every line `valid=y`; `cpu_alone(8)` ≥ 25 GB/s; ring in 30–120 GB/s and within ±20 % of `DDR_DSP_REF` (the line's own validity); hvx in 15–40 GB/s (rule 27 band 21–27) |
| G2 both at once | 6 × `DDR_TWO_READER` | `valid=y` on every row |
| G3 prefetch | 12 × `PREFETCH_OVERLAP` | every line present; `touch_late` ≤ 3/64 at S ≤ 10, T ≥ 2 |
| G4 validity (rule 12) | whole log | `grep -c INVALID` = 0, no `FAILED` (an INVALID line fails its test by `EXPECT`) |
| G5 standing | A logs | `dspq: on` once each; text at G=512 ≡ #150 T8 (hash); prefill within −5 % of #150 A0 |

## Results

Run 2026-09-29 11:07–11:10 KST by the orchestrator, unit `R3CY10WM83Y`,
right after #150's sitting in the same `s150` dir (only the probe pushed;
device md5 = md5.txt, MD5 OK). Cool start (battery 26.3 °C, zone0 27.8 °C);
zone0 52.9 °C after the cold probe, 63.3 °C after the E2E cells. Logs:
`/local/mnt/workspace/htp_moe/90/logs/`.

**`MoeChunkReplay` FAILED with `checksum_ok=n`**: expected — this probe
binary predates PR #154 (#99's fix, measured `checksum_ok=y` the same
morning). TwoReaderDdr and PrefetchOverlap PASSED; `INVALID` count 0.

### (1) Readers alone and together (cold probe)

| reader | alone GB/s | with the other | aggregate |
|---|---|---|---|
| CPU 1 / 2 / 4 / 8 threads | 63.0 / 68.1 / 69.4 / 68.3 | — | — |
| DSP DMA ring (`f2`, fresh, 168 MiB) | 37.3 (ref `f2` 40.3) | — | — |
| DSP HVX direct (M=1 MoE, feed off) | 24.8 | — | — |
| CPU 1 / 2 / 4 / 8 thr + ring | — | CPU 40.3 / 40.6 / 38.9 / 38.5, ring 31.0 / 30.9 / 31.3 / 31.0 | **71.2 / 71.5 / 70.2 / 69.5** |
| CPU 2 / 8 thr + HVX | — | CPU 48.7 / 49.3, HVX 15.9 / 17.3 | 64.6 / 66.6 |

Warm probe: CPU + ring aggregate 63.9–70.2 GB/s, same shape.

**Reading.** The phone's DRAM tops out at **≈ 70 GB/s** for any mix of
readers. The CPU alone already reaches 63–69 GB/s; adding the DSP ring
raises the total by only ≈ 2–3 % (the CPU loses 36–44 %, the ring 16–17 %).
So (b) does not add bandwidth on top of a CPU that already streams; it
**redistributes** it. What it can buy is using the ≈ 70 GB/s during the
MoE window, when today only the DSP reads (at ≈ 30–33 GB/s).

### (2) Prefetch overlap (the bit-preserving form of (b))

| S MiB | threads | DSP `mm` change cold / warm | net ms/token cold / warm |
|---|---|---|---|
| 4 | 1 / 2 / 7 | +0.7 / −0.5 / −0.9 % · +0.3 / +0.3 / −0.5 % | +0.72 / +0.91 / +1.23 · +0.88 / +0.95 / +1.39 |
| 10 | 1 / 2 / 7 | +5.5 / +7.6 / +6.4 % · +1.5 / +0.8 / +1.5 % | −0.09 / −0.37 / +0.72 · +0.82 / +1.08 / +1.69 |
| 20 | 1 / 2 / 7 | +18.4 / +17.4 / +15.0 % (cold) | −1.96 / −1.85 / −0.58 |
| 32 | 1 / 2 / 7 | +31.2 / +30.5 / +28.3 % (cold) | −4.16 / −3.69 / −3.30 |

Rule P (S=10: DSP slowdown ≤ 5 %, retention ≥ 0.5, net ≥ 1.0 ms/token):
**fails cold, passes warm with 2 or 7 threads.** S=4 is positive in every
cell (+0.7 to +1.4 ms/token, DSP unchanged). Up to ≈ 4–10 MiB per MoE call
fits in the CPU caches without slowing the DSP; beyond that it costs more
than it saves.

### (3) Full E2E control A (NNTR_OP_TIME=1, dspq default)

| G | prefill tok/s | decode tok/s | dspq close |
|---|---|---|---|
| 64 | 576.6 | **39.70** | 1408 / 1408, bad=0 |
| 512 | 552.3 | **39.18** | 11264 / 11264, bad=0 |
| 1024 | 561.4 | **36.53** | 22528 / 22528, bad=0 |

Every log `dspq: on` once (rule 40). Text A G=512 = #150 T8 run 1 (identical).
Per-op at G=1024: MoE wait 16.2 of 27.4 ms (59 %), attention 1.97 ms (grows
with position), FC + lm_head 7.9 ms at 47–61 GB/s. **First G=1024 cell on
the dspq default: 36.53 tok/s** (fills the "now" G=1024, previously 35.17
from #120).
 (fill in)

Unit, date, KST window: …; device `md5sum` = `md5.txt`? …; thermal
checkpoints (battery °C·10 / zone0 m°C): 0 …, 1 …, 2 …, 3 ….
`CPU_CACHE` (cpu0 / cpu7, L2 / L3 sizes and sharing): ….
`chunk_bytes`: …; `MoeChunkReplay` anchor `DMA_REPLAY workers=1 load=0
pace=0` gbs cold / warm: … / …; `DMA_REPLAY_X name=f2` gbs: … / ….

### (1) Readers, cold | warm (GB/s)

References: #77 CPU 67.90 (other unit `R3CY205ZMND`), ring anchor 37.3
(rule 34), in-app ring 33.0 (0.88 ×, rule 41), D192 `mm` 922.7–931.9 µs →
23.6–23.9 GB/s. Physical ceiling 85.3.

| t | cpu_alone (cpus=) | ring: cpu_with / dsp_with / aggregate / cpu_loss % / dsp_loss % | hvx: cpu_with / dsp_with / aggregate / cpu_loss % / dsp_loss % |
|---|---|---|---|
| 1 | | | — |
| 2 | | | |
| 4 | | | — |
| 8 | | | |
| DSP alone | — | ring `dsp_alone` = … (`dsp_ref` …, anchor …) | hvx `dsp_alone` = … (duty …) |

### (2) Prefetch, cold (copy the table for warm). `mm_alone` from `PREFETCH_COLD`

| S MiB | cold_us | mm_alone_us | T | hot_us | reread_us | staged / staged_mib | dmm_pct | touch_us / touch_late | net ms/token | moved |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | | | 1 | | | | | | | |
| 4 | | | 2 | | | | | | | |
| 4 | | | 7 | | | | | | | |
| 10 | | | 1 | | | | | | | |
| 10 | | | 2 | | | | | | | |
| 10 | | | 7 | | | | | | | |
| 20 | | | 1 | | | | | | | |
| 20 | | | 2 | | | | | | | |
| 20 | | | 7 | | | | | | | |
| 32 | | | 1 | | | | | | | |
| 32 | | | 2 | | | | | | | |
| 32 | | | 7 | | | | | | | |

### (3) A, full E2E (prompt 512, `NNTR_NUM_THREADS=8 NNTR_OP_TIME=1`)

| variant | gen | run | prefill tok/s | decode tok/s (all) | decode tok/s (last 64) | peak RSS | text ≡ #150 T8 (G=512) / hash | skel md5 (device) | `dspq: close` |
|---|---|---|---|---|---|---|---|---|---|
| A | 64 | 1 | | | | | | | |
| A | 512 | 1 | | | | | | | |
| A | 1024 | 1 | | | | | | | |

Reference (#141 sitting, same unit, Q = today's default, no `OP-TIME`):
decode 39.40 at G=64, 38.55 / 36.99 (mean 37.77) at G=512, prefill
483.5 / 436.5; G=1024: (35.17) FastRPC placeholder only (BENCHMARK).
#150 A0 / T8 at G=512 from its own sitting when filled. Goal ≥ 50 decode,
prefill ≥ −5 % of A.

`[OP-TIME]` at G=512 (paste `report_A_g512.txt`'s FC kinds + lm_head
rows): FC ms/token = … (the ceiling of what rule P can save), MoE wait
ms/token = …, `[OP-TIME] moe` call µs/call = ….

### Gates

| # | result | pass |
|---|---|---|
| G1 | | |
| G2 | | |
| G3 | | |
| G4 | | |
| G5 | | |

## Notes from the run

Serial, battery, zone0 at each checkpoint, cold-start temperature (and
#150's checkpoint 0 if run after it), `chunk_bytes`, a second device on
USB, FARF / AEE errors, anything stale.
