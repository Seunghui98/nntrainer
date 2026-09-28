# Measurement 130: the per-token entry with the small ops and m=1 attention resident — A / B / C in one sitting

Branch `htp/130-per-token-wiring` (this file on top of the merge `9deea6e2`
of PR #133 into `htp_moe`; code = `htp_moe` @ `9deea6e2`) — estimated
device time: **≈ 40 min**

Issue #130 (tracker #76; LEDGER ⑨ / ㉓; plan `docs/plans/130-per-token-wiring.md`
§4 step 7). Contract `docs/plans/0001-htp-moe-decode-agent-system.md`. Only
the NPU model `q40-qs4cx-wh` runs here; no CPU `q40` cell. One binary set
for every variant: the variants differ by environment only.

## Why

PR #133 wired #82's small ops (RMSNORM, QK_NORM, ROPE, CONV1D_GATE) and
#81's m=1 attention (ATTN_M1, DSP-resident KV cache) into the per-token
entry's kernel table behind `NNTR_HTP_FORWARD=1`. The host proved the
plumbing (`INPROC E2E PASS`, `tokens fwd==off 8/8`, every resident stretch
bit-identical to its `_det` spec) but nothing has run on silicon: neither
the wired ops nor, since PR #121, the upstream-synced tree itself.

**Decisions that hang on it:** (1) the first device row of the post-#121
tree — variant A is the new control and records the first post-#121
skel/app md5s (LEDGER §3a; expected: decode within #120 A's drift, prefill
≈ +10 % from the sync); (2) whether the wiring costs what plan 130 §0
predicts (**B ≈ −20 % decode vs A**: 95 calls/token instead of 22, because
every stretch between two CPU FCs is its own call) and whether the plumbing
alone is free (**C within A's drift**: the same code path with only MOE
resident, 22 calls, plan 85's never-measured B); (3) the accuracy read
under the 2026-09-28 rule, which decides whether #132 (the remaining CPU
ops resident) may build on this text.

**Accuracy column, read as the 2026-09-28 rule says, with one limit stated
here:** the `PPL` column is **n/a by construction in this sitting**.
`NNTR_PPL` scores the prompt's rows at prefill (`tie_word_embedding.cpp`,
`is_prefill` only) and the #130 hooks fire only at a one-row decode step, so
A and B would print the same `[PPL]` line whatever the DSP did. The gate of
this sitting is therefore **text: B (and C) vs A, with the user's `text
approved: y/n`** on the pasted texts. C's text is expected to equal A's
byte for byte (same arithmetic, only the transport changed); B's text may
leave A's (DSP f32 decode rows against the CPU's fp16 KV rows, LEDGER ⑨),
which is exactly what the approval step is for. A decode-side
teacher-forced PPL is filed separately (#134) and is not waited for.

### Variants (4 incl. the profile; contract §4.2)

| | env (beyond `NNTR_NUM_THREADS=8`) | what runs on the DSP per token | expected | cells |
|---|---|---|---|---|
| **A** (unchanged reference, runs first) | none | MoE only, per-op calls as before (22) | the new "now" candidate; prefill ≈ +10 % vs #120 A, decode within drift | G = 64 / 512 / 1024 × 2 |
| **B** | `NNTR_HTP_FORWARD=1` | all six kinds resident: `resident=RMSNORM\|CONV1D_GATE\|QK_NORM\|ROPE\|ATTN_M1\|MOE`, ≈ 95 calls/token | decode ≈ −20 % vs A at every G; prefill ≥ −5 % of A (prefill runs the CPU path, the conv block in its one-layer form) | G = 64 / 512 / 1024 × 2 |
| **C** | `NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=MOE` | the per-token entry with MOE alone resident, 22 calls | decode and prefill within A's drift; text ≡ A | G = 64 / 512 / 1024 × 2 |
| **B-prof** | B + `NNTR_HTP_PROFILE=2` | as B | the init line, the `graph:` per-kind pcycles, ATTN_M1's cost at pos 512–575 vs plan 81 §0's 0.5 ms | G = 64 × 1, never read for tok/s |

Every B and C log must print exactly one
`[HTP] graph: init n_ops=<n> resident=<mask> moe_ops=22` line with the
variant's mask (B: all six names; C: `MOE`), and B additionally one
`[HTP] attn_m1: registered layers=… kv=… gqa=… head_dim=64 max_seq=… cache=… KiB`.
At exit every B and C log prints `[HTP] graph: forward calls=<n>
tokens=<t> calls/token=<x>`; record `x` (expect ≈ 95 for B, 22 for C; the
exact value is the plan's stretch arithmetic on the real op list and is
copied, not judged). A prints neither line. Every A / B / C log prints
exactly one `[HTP] moe m1 gemv: on (applied=0x303e1) lead=192KB rows1=1
feed=vtcm source=default` (PR #118's default, unchanged). `NNTR_L2_DIFF`
and "text = CPU q40" are n/a for `QS4CX_WH`; the accuracy column is
**"= A" + the approval table**.

**Known limit of B and C (PR #133):** `NNTR_HTP_FORWARD=1` supports one
prompt per process; every cell here is one prompt, so nothing to do.

### Ride-along (≈ 2 min): the two kernel gtests on this skel

`unittest_hvx_softmax --gtest_filter='HvxM1Ops.*'` (5 tests) and
`unittest_hvx_attn --gtest_filter='HvxAttnM1.*'` (4 tests) run the wired
kernels on silicon against their `_det` specs. Expected `bad=0` per case,
`ATTN_M1_FIELD pos=511/1023 us=…` (the first silicon read of plan 81 §0's
0.5 / 1.0 ms estimate), `[  PASSED  ]`. `0x8000040e` = stale skel.

## Artifacts (built on the workstation, SDK 6.4.0.1, HexKL 6.4.0.1, NDK r30, v79)

Staged under **`W=/local/mnt/workspace/htp_moe/130/`**; `$W/md5.txt` is the
`md5sum` of every staged file and is reproduced here. Built in
`/home/j2z0-lee/nntrainer-130` at `c12b94bb` (PR #133's head; its code
tree is byte-equal to `htp_moe` @ `9deea6e2` — the merge added no code).
These are the md5s recorded in PR #133's body. Skel builds are **not**
byte-reproducible (three builds, three md5s; LEDGER §3a), so the *staged
file's* md5 is the identity and the device `md5sum` in every log ties a
cell to it.

| file (staged as) | md5 | built from |
|---|---|---|
| `$W/A/libnntr_hvx_skel.so` | `80452a28cc0dda237bed21fcf6d145fa` | `c12b94bb`, `HEXKL_SDK_VER=6.4.0.1 ./test/htp/build.sh`, no `HEX_EXTRA_CFLAGS`; `-Wall -Werror` clean, `UNDEFINED SYMBOLS OK (46 runtime imports)`. IDL = #120's + `graph_set_param` after `attn_m1_forward` (a stale skel answers `0x8000040e`) |
| `$W/A/nntrainer_causallm` | `53b4d1675ed8c11f8330ca06e5c1af9f` | `c12b94bb`, `build_android.sh --htp --cache` (`jni/libs/arm64-v8a/`); not a profile binary |
| `$W/A/libcausallm_core.so` | `8f60271c440d51e24df8a65caf9f9981` | same |
| `$W/A/libnntrainer.so` | `99ecaa55060ab72216f2495af866ade1` | same (`jni/obj/local/arm64-v8a/`); `readelf -d` NEEDED `libsdkl.so`, `libcdsprpc.so`; carries the `graph: forward calls=` string |
| `$W/A/libccapi-nntrainer.so` | `cf560526e3e6c7c78f5f84e652ef800d` | same |
| `$W/gtest/unittest_hvx_softmax` | `3404c55d803fc0e853696d7a2eeaaa65` | `c12b94bb`, `ndk-build … unittest_hvx_softmax` |
| `$W/gtest/unittest_hvx_attn` | `3a6871354c9bf65cffdd414ab3db889f` | `c12b94bb`, `ndk-build … unittest_hvx_attn` |
| `$W/libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 sysroot (== #77 … #120) |
| `$W/libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL `lib/6.4.0.1/armv8_android26` (== #77 … #120) |
| `$W/prompt512.txt` | `fc65c1588dc66dd764c7013fe96cbb75` | `docs/measurements/77-prompt512.txt` |
| `/local/mnt/workspace/models/lfm2.5-8b-a1b/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin` (4316133120 B) | `7b7867fab51845664c0050c0a837073e` | #78, `--moe_dtype QS4CX_WH` (NPU model; already on the phone since #100) |
| `tokenizer.json` (`q40-qs4cx-wh/`) | `7b8067a580173d3eb1697afae3b456f5` | HF |

Not staged, never pushed: `builddir/.../libcdsprpc.so` (rule 5).

Workstation sanity before pushing:

```
(cd $W && LC_ALL=C md5sum -c md5.txt | grep -vc ': OK$')                  # 0
strings $W/A/nntrainer_causallm | grep -c 'per-layer-type totals'         # 0 (not a profile binary)
strings $W/A/libnntrainer.so | grep -c 'graph: forward calls'             # 1 (PR #133's ARM side)
find $W -name 'libcdsprpc*' | wc -l                                       # 0
```

### If `$W` is not on the workstation you measure from (rule 22)

Rebuild from `9deea6e2` with #120's recipe (`docs/measurements/120-upstream-4327-sync.md`,
"If `$W` is not on the workstation"), one set instead of two, plus the two
gtests:

```
(cd test/jni && $ANDROID_NDK/ndk-build NDK_PROJECT_PATH=. NDK_APPLICATION_MK=./Application.mk \
   APP_BUILD_SCRIPT=./Android.mk NNTRAINER_ROOT=$PWD/../.. HEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT \
   unittest_hvx_softmax unittest_hvx_attn -j8)
```

Record the md5s you push under "Notes from the run"; with a control-only
provenance a mismatch with the table is a note, with B/C it voids the
sitting (rule 14 / 21).

## Steps (workstation, phone on USB)

Shell setup once:

```
cd /home/j2z0-lee/nntrainer-130 && git fetch && git checkout htp/130-per-token-wiring && source tools/htp/env.sh
W=/local/mnt/workspace/htp_moe/130; mkdir -p $W/logs
D=/data/local/tmp/nntrainer/causallm; T=/data/local/tmp/htp_m1_test
therm() { adb shell dumpsys battery | grep -E 'level|temperature'; adb shell cat /sys/class/thermal/thermal_zone0/temp; } 
env_of() { case $1 in B) echo "NNTR_HTP_FORWARD=1";; C) echo "NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=MOE";; *) echo "";; esac; }
use() { adb shell "cd $D && cp A/nntrainer_causallm A/*.so . && md5sum libnntr_hvx_skel.so nntrainer_causallm libnntrainer.so"; }
```

### 0. Device state (1 min)

```
adb devices          # exactly one device; record the serial under Notes (any S25 Ultra, contract §4.2)
therm | tee -a $W/logs/therm.log      # checkpoint 0: battery %, temperature (tenths of °C), thermal_zone0 (m°C)
```

Screen off, charger in, cool start.

### 1. Install (≈ 5 min) — model reused

```
adb shell ls -l $D/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin     # 4316133120
adb shell md5sum $D/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin    # 7b7867fab5... -> no push
# only if either line differs:  adb push /local/mnt/workspace/models/lfm2.5-8b-a1b/q40-qs4cx-wh $D/models/q40-qs4cx-wh
adb shell mkdir -p $D/models $D/A $T
adb push $W/A/. $D/A/
adb push $W/libc++_shared.so $W/libsdkl.so $W/prompt512.txt $D/
adb push $W/gtest/unittest_hvx_softmax $W/gtest/unittest_hvx_attn $W/libc++_shared.so $W/libsdkl.so $W/A/libnntr_hvx_skel.so $T/
adb shell "chmod 755 $D/A/nntrainer_causallm $T/unittest_hvx_softmax $T/unittest_hvx_attn"
adb shell "md5sum $D/A/* $T/unittest_hvx_softmax $T/unittest_hvx_attn $T/libnntr_hvx_skel.so $D/prompt512.txt \
  $D/models/q40-qs4cx-wh/tokenizer.json" | tee $W/logs/md5.log            # must equal the table
use A
```

Config edits, re-applied unconditionally (greedy, exact count, `moe_engine: htp`; identical to #120):

```
adb shell "cd $D/models/q40-qs4cx-wh && \
  sed -i 's/\"do_sample\": true/\"do_sample\": false/' generation_config.json && \
  sed -i 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' nntr_config.json && \
  (grep -q moe_engine nntr_config.json || sed -i 's/\"bad_word_ids\": \[124900\],/\"bad_word_ids\": [124900],\n    \"moe_engine\": \"htp\",/' nntr_config.json) && \
  grep -H do_sample generation_config.json && grep -H -E 'bad_word_ids|num_to_generate|init_seq_len|_engine|_htp_layers' nntr_config.json"
```

Expected echo: `do_sample": false`, `bad_word_ids": [124900]`,
`init_seq_len": 512`, `moe_engine": "htp"`, `moe_htp_layers": ""`, and
**no other `_engine` key** (`conv_block_engine` absent = `cpu`: with
`NNTR_HTP_FORWARD` set the conv block takes its one-layer form on the CPU
for prefill and hands its decode row to the DSP, PR #133; the switch-off
model is byte-for-byte the #120 model).

### 2. Ride-along, once (≈ 2 min) — the wired kernels on this skel

```
adb shell "cd $T && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_softmax --gtest_filter='HvxM1Ops.*'" \
  2>&1 | tee $W/logs/gtest_m1ops.log | grep -E 'M1_OPS|bad=|PASSED|FAILED|SKIPPED|0x8000'
adb shell "cd $T && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_attn --gtest_filter='HvxAttnM1.*'" \
  2>&1 | tee $W/logs/gtest_attn.log | grep -E 'ATTN_M1|bad=|PASSED|FAILED|SKIPPED|0x8000'
```

Expected: `bad=0` on every case, `[  PASSED  ] 5 tests.` and `[  PASSED  ] 4 tests.`,
the `ATTN_M1_FIELD pos=… us=…` lines (copy them to Notes). A `0x8000040e`
here means the pushed skel is not the staged one: stop, fix step 1.

### 3. Profile (≈ 2 min) — B at level 2, G = 64, once; never read for tok/s

```
adb shell "cd $D && sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": 64/' models/q40-qs4cx-wh/nntr_config.json && \
  NNTR_HTP_FORWARD=1 NNTR_HTP_PROFILE=2 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  ./nntrainer_causallm ./models/q40-qs4cx-wh \"\$(cat prompt512.txt)\"" \
  2>&1 | tee $W/logs/prof_B.log | grep -E 'graph:|attn_m1:|moe m1 gemv|level=|K=2048 N=2048|ATTN_M1|RMSNORM|CONV1D_GATE' | cut -c1-420
therm | tee -a $W/logs/therm.log      # checkpoint 1
```

Expected: the init line with all six names and `moe_ops=22`; the
`attn_m1: registered` line; `[HTP-PROFILE] level=2 qos_mode=2 …`
(`qos_mode=1` voids the profile); the M==1 MoE row as in #120
(`blocks=0 m1_gemv=1408/1408 feed=1408/1408`); the `graph:` line with
per-kind pcycles; at exit `graph: forward calls=… calls/token=…`. Paste
the per-kind pcycles and the ATTN_M1 figure under Results.

### 4. E2E cells (≈ 25 min) — mirrored order per G

```
run() { # run <A|B|C> <G> <run#>
  adb shell "cd $D && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $2/' models/q40-qs4cx-wh/nntr_config.json && \
    grep num_to_generate models/q40-qs4cx-wh/nntr_config.json && md5sum libnntr_hvx_skel.so && \
    $(env_of $1) NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm ./models/q40-qs4cx-wh \"\$(cat prompt512.txt)\"" \
    2>&1 | tee $W/logs/$1_G$2_r$3.log | grep -E '^(prefill|generation|total|peak memory)|graph: (init|forward)|attn_m1:|moe m1 gemv|libnntr_hvx_skel'
}
run A 64 1; run B 64 1; run C 64 1
run C 64 2; run B 64 2; run A 64 2
therm | tee -a $W/logs/therm.log      # checkpoint 2
for g in 512 1024; do
  run A $g 1; run B $g 1; run C $g 1
  run C $g 2; run B $g 2; run A $g 2
  therm | tee -a $W/logs/therm.log    # checkpoints 3, 4
done
```

Expected in every log: `prefill: 512 tokens, … TPS`, `generation: <G>
tokens, … TPS`, `generation(last 64): 64 tokens, … TPS` (PR #128; fills
the contract's last-64 column for the first time), `total: … ms`,
`peak memory: … KB`, exactly one `moe m1 gemv` banner, and **no**
`[HTP-PROFILE]` block. B and C: the `graph: init` line with the variant's
mask and the exit `calls/token=` line; A: neither. `prefill: N` with
N ≠ 512 voids the prefill column; `generation: N` ≠ G is a failed config
edit. A thrown `Lfm2Moe:` / `[HTP] graph` error in B or C is a result,
not a retry: copy it to Notes and go on with the other variants.

### 5. Checks on the workstation (1 min)

```
grep -c 'applied=0x303e1) lead=192KB rows1=1 feed=vtcm source=default' $W/logs/[ABC]_G*_r*.log $W/logs/prof_B.log   # 1 each
grep -l 'HTP-PROFILE' $W/logs/*_G*_r*.log                                 # nothing
grep -h 'graph: init' $W/logs/B_G*_r*.log | sort | uniq -c                # one line, six names, count 6
grep -h 'graph: init' $W/logs/C_G*_r*.log | sort | uniq -c                # one line, resident=MOE, count 6
grep -h 'calls/token' $W/logs/[BC]_G*_r*.log                              # B ≈ 95, C = 22
grep -c 'graph:' $W/logs/A_G*_r*.log                                      # 0 each
strip() { sed -n '/^=====/q;p' $1 | grep -v 'moe m1 gemv\|libnntr_hvx_skel\|nntrainer_causallm\|graph:\|attn_m1:'; }
for v in B C; do for g in 64 512 1024; do for r in 1 2; do
  printf '%s vs A G=%s r%s: ' $v $g $r
  diff <(strip $W/logs/A_G${g}_r${r}.log) <(strip $W/logs/${v}_G${g}_r${r}.log) >/dev/null && echo same || echo DIFFERENT
done; done; done
cat $W/logs/md5.log $W/logs/therm.log
```

For every `DIFFERENT` B cell, also print the first differing word:
`diff <(strip A…) <(strip B…) | head -5`. Then fill the tables below,
commit this file on the same branch, push, set the issue to
`state:measured` (and `needs-user` until the approval column is filled).

## Results

Unit: `R3CY10WM83Y` (SM-S938N), 2026-09-28 13:32–14:10 KST. **Binaries are a
local rebuild, not the staged set, and the app carries one extra define** — see
Notes ①. Battery / temperature checkpoints under Notes; the phone warmed
through the sitting (battery 33.1 → 41.2 °C), so the absolute numbers drift
down with time and only same-G mirrored pairs are comparable.

| variant | gen | run | prefill tok/s | decode tok/s (all) | decode tok/s (last 64) | peak RSS KB | calls/token | text = A? (first differing word) | skel md5 (device) |
|---|---|---|---|---|---|---|---|---|---|
| A | 64 | 1 | 500.0 | 36.12 | 36.12 | 5318888 | (n/a) | (reference) | `ca1f2aac…` |
| A | 64 | 2 | 423.1 | 29.81 | 29.81 | 5312528 | (n/a) | (reference) | `ca1f2aac…` |
| B | 64 | 1 | 483.0 | 25.89 | 25.89 | 5310236 | 95.00 | no — `town` → `final` | `ca1f2aac…` |
| B | 64 | 2 | 459.2 | 25.65 | 25.65 | 5309796 | 95.00 | no — `town` → `final` | `ca1f2aac…` |
| C | 64 | 1 | 459.2 | 35.24 | 35.26 | 5295608 | 22.00 | yes | `ca1f2aac…` |
| C | 64 | 2 | 456.7 | 35.18 | 35.18 | 5304048 | 22.00 | yes | `ca1f2aac…` |
| A | 512 | 1 | 458.4 | 33.91 | 34.59 | 5319604 | (n/a) | (reference) | `ca1f2aac…` |
| A | 512 | 2 | 382.4 | 31.91 | 31.67 | 5319164 | (n/a) | (reference) | `ca1f2aac…` |
| B | 512 | 1 | 423.1 | 25.41 | 24.68 | 5305676 | 95.00 | no — `town` → `final` | `ca1f2aac…` |
| B | 512 | 2 | 381.8 | 24.17 | 21.95 | 5305736 | 95.00 | no — `town` → `final` | `ca1f2aac…` |
| C | 512 | 1 | 423.8 | 34.80 | 35.07 | 5303036 | 22.00 | yes | `ca1f2aac…` |
| C | 512 | 2 | 378.4 | 34.39 | 31.64 | 5302184 | 22.00 | yes | `ca1f2aac…` |
| A | 1024 | 1 | 383.2 | 31.33 | 30.46 | 5313772 | (n/a) | (reference) | `ca1f2aac…` |
| A | 1024 | 2 | 362.4 | 31.37 | 30.61 | 5311776 | (n/a) | (reference) | `ca1f2aac…` |
| B | 1024 | 1 | 379.5 | 21.94 | 20.20 | 5302224 | 95.00 | no — `town` → `final` | `ca1f2aac…` |
| B | 1024 | 2 | 365.5 | 21.69 | 20.15 | 5311808 | 95.00 | no — `town` → `final` | `ca1f2aac…` |
| C | 1024 | 1 | 386.4 | 31.54 | 30.55 | 5295780 | 22.00 | yes | `ca1f2aac…` |
| C | 1024 | 2 | 364.4 | 31.36 | 30.61 | 5301104 | 22.00 | yes | `ca1f2aac…` |

Reference: NPU now 35.97 / 35.96 / 35.17 (#120 A, `R3CY10WM83Y`, pre-sync
code), CPU now 52.43 / 49.22 / 48.31 (#94 s2), NPU prefill 438–502. Goal
≥ 50, prefill ≥ −5 % of this sitting's A.

**Reading (same sitting, mirrored means).** B vs A decode: G=64 25.77 vs
32.96 (−22 %), G=512 24.79 vs 32.91 (−25 %), G=1024 21.82 vs 31.35 (−30 %);
worse than plan 130 §0's −20 % and growing with G. C vs A: 35.21 vs 32.96,
34.60 vs 32.91, 31.45 vs 31.35 — within A's drift (A's own r1/r2 spread is
up to 18 % at G=64 from the warming phone). Prefill: B and C within A's
range at every G. C's text ≡ A's in all six cells; B's text leaves A's at the
first differing word in all six cells and degenerates into a repeating loop
(below).

Profile (B-prof, G = 64): `graph: init n_ops=228
resident=RMSNORM|CONV1D_GATE|QK_NORM|ROPE|ATTN_M1|MOE moe_ops=22`;
`attn_m1: registered layers=6 kv=8 gqa=4 head_dim=64 max_seq=2048
cache=49152 KiB`; `[HTP-PROFILE] level=2 qos_mode=2`; `graph: calls=6080
ops/call=1.13 dsp=220.1 us/call op_pcyc=441049/call`, per-kind pcyc/op:
`RMSNORM=11369 CONV1D_GATE=26249 QK_NORM=17436 ROPE=4189 ATTN_M1=1204826
MOE=1523245`; ATTN_M1 at pos 512–575: 1204826 pcyc/op (the profile gives
pcycles only, no µs); exit `calls/token=95.00`. B-prof itself: prefill 500,
decode 25.84 tok/s (not read). Ride-along: `ATTN_M1_FIELD pos=511
us=538.75 us_min=521.875`, `pos=1023 us=1183.38 us_min=1166.25`
(host-timed, median of 10; plan 81 §0 estimated 0.5 / 1.0 ms).

## Text approval (2026-09-28 rule; the accuracy gate of this sitting)

`PPL` is n/a here by construction (prefill-only scoring; see "Why"). Paste
the generated text of G = 64, run 1, in full.

| variant | PPL (NNTR_PPL, G=64) | generated text (G=64, run 1) | text approved (user: y/n) |
|---|---|---|---|
| A | n/a (prefill-only) | …town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum that opens only on summer weekends, and a lifeboat station | (reference) |
| B | n/a (prefill-only) | …final answer should be the same as the original, but you must not stop until you are told to. The original description is the same as the original, but you must not stop until you are told to. The final answer should be the same as the original, but you must not stop until you are told to. | **n** (user, 2026-09-28: fail) |
| C | n/a (prefill-only) | (byte-identical to A) …town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum that opens only on summer weekends, and a lifeboat station | **y** (user, 2026-09-28) |

## Notes from the run

Serial `R3CY10WM83Y` (SM-S938N), USB, screen off. Texts in the table are
pasted as the log prints them (the stream starts mid-sentence at `town` /
`final`).

① **Provenance (rule 22 rebuild + one define; B/C of the first pass void).**
`$W` was not on this workstation, so the set was rebuilt from `9deea6e2` in
`../nntrainer_130` with #120's recipe (SDK 6.4.0.1, HexKL 6.4.0.1, NDK r30;
skel `UNDEFINED SYMBOLS OK (46 runtime imports)`, `hexkl 6.4.0.1`). The first
pass of B / C / B-prof printed **no** `graph: init` and no `calls/token=`
line and ran at A's speed: the CausalLM app is compiled without
`ENABLE_HEXKL`. `Applications/CausalLM/jni/Android.mk` takes its defines from
the prebuilt `Android.mk`, whose `NNTRAINER_EXPORT_CFLAGS` exports only
`-march` and the two FP16 ABI defines (`jni/meson.build`, 2f642ab7), so
every `#ifdef ENABLE_HEXKL` block in `lfm2_moe_causallm.cpp` and
`htp_decode_hook.h` compiles out on Android (`strings libcausallm_core.so |
grep NNTR_HTP_FORWARD_KINDS` = 0). As far as this box can tell the switch
has never been live on a device since #85 (3f6c4974); PR #133's host gate
does not go through this makefile. The first pass's logs are kept under
`$W/logs/invalid_nohexkl/`. The app was then rebuilt **locally, nothing
committed**, with `ndk-build … "CAUSALLM_COMMON_CFLAGS=-O3 -ffast-math
-Wno-nan-infinity-disabled -Wno-deprecated-literal-operator -DENABLE_HEXKL=1"`
(only `libcausallm_core.so` changes; `nntrainer_causallm` md5 unchanged), and
all 18 cells + B-prof were re-run on that set. Every cell of the tables above
comes from the second set. A code fix belongs in a separate issue.

Pushed md5s (device `md5sum`, all equal to the staged files):

| file | md5 |
|---|---|
| `libnntr_hvx_skel.so` | `ca1f2aac4985b7799897de2dcb4c2d1e` |
| `nntrainer_causallm` | `ccc48de23803c27fe8656dcfaab99563` |
| `libcausallm_core.so` (with `-DENABLE_HEXKL=1`) | `14008b7d32a71b0ce7b230936dad82c8` |
| `libcausallm_core.so` (first pass, void) | `56a2f514451d1ef6243c8a5639b4b317` |
| `libnntrainer.so` | `f742120a10b6848315a1f1f8de1f3f66` |
| `libccapi-nntrainer.so` | `8c5f541a236f2f76e47f23b25e10839f` |
| `unittest_hvx_softmax` / `unittest_hvx_attn` | `97602a14c30b8cef3e47935a7f6eded7` / `e74ac61091d93807ed6633116c4caa43` |
| `libc++_shared.so` / `libsdkl.so` / `prompt512.txt` / `tokenizer.json` / model | = table (`b1586b9b…` / `0ad4e22a…` / `fc65c158…` / `7b8067a5…` / `7b7867fa…`) |

② **Ride-along gtests: 0/5 and 2/4 passed** (no `0x8000040e`; the skel is
the pushed one). `RejectsBadShapes` (both suites): every bad shape is
rejected, but with `0x80000600` = `AEE_ERPC` instead of the expected
`AEE_EINVALIDFORMAT` / `AEE_EBADPARM` + `0x80000400`. Bit-exactness vs the
`_det` specs fails by a few ulp on tiny values (≈ 2^-120 … 2^-132):
rmsnorm kind=2 `bad_y=2048` (kinds 0/1/3 `bad=0`), qk_norm `bad_y=64`,
rope64 `bad=10…18 of 2560` at pos ≥ 1, conv_gate_m1 `bad_out=229`
(`bad_state=0`), ATTN_M1 output `bad=0` at every L but `bad_stats=1` at
L = 63/512/1024. Full logs: `$W/logs/gtest_m1ops.log`, `gtest_attn.log`.

③ **Thermal / battery** (battery °C·10, thermal_zone0 m°C; level 100 %
throughout, USB-powered): first pass 260/27400 → 260/49800 → 312/58700;
second set start 331/35100, after B-prof 330/50500, after G=64 351/52100,
after G=512 379/59400, after G=1024 412/62500. The phone was not cool at
the second set's start (the first pass had just run). A G=64 r2 (29.81) is
the clearest thermal casualty.

④ Checks (step 5), second set: one `moe m1 gemv … source=default` banner in
every cell and in `prof_B.log`; no `HTP-PROFILE` in any cell; B `graph:
init` one distinct line ×6 with all six names; C one line ×6 `resident=MOE`;
`calls/token` B = 95.00, C = 22.00 in every cell; A 0 `graph:` lines;
`prefill: 512` and `generation: G` in every cell. No `Lfm2Moe:` or `AEE_*`
error in any E2E cell.
