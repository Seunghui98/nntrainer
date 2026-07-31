# Probe Runbook — running the HTP probes on device

Everything here is on branch **`claude/hexkl-mha-hmx-optimization-6ycsx0`**.

Three probes, answering three questions that the
[lm_head plan](09_lmhead_u8i4_plan.md) is gated on:

| probe | side | question | §|
| :-- | :-- | :-- | :-- |
| `hexkl_gemv_probe` | host (ARM) | does `sdkl_npu_mm_u8i4_i32` accept an unaligned `n_row`? | [09](09_lmhead_u8i4_plan.md) §4 |
| `hexkl_pin_probe` | host (ARM) | how much NPU memory can stay resident? | [09](09_lmhead_u8i4_plan.md) §3 |
| `hexkl_decode_probe` | host (ARM) | how much of a decode token is weight staging rather than matmul? | [09](09_lmhead_u8i4_plan.md) §2 |
| `hexkl_layout_probe` | DSP | what do RM / AH / WH actually permute to? | [08](08_attention_hmx_design.md) §3 |

Run them in that order. `hexkl_gemv_probe small` is seconds and decides the
most; `hexkl_gemv_probe lmhead` needs the pin budget to exist, so run the pin
probe in between.

---

## 0. Prerequisites

### 0.1 A beta2 SDK, and knowing where its library lives

beta2 inserts the Hexagon SDK version between `lib/` and the ABI directory,
which beta1 did not:

```bash
ls $HEXKL_SDK_ROOT/lib                 # e.g. 6.4.0.2  6.5.0.1  6.6.0.0 ...
ls $HEXKL_SDK_ROOT/lib/6.4.0.2         # e.g. armv8_android26  armv9_android26 ...
```

Pick one and export the part between `lib/` and `libsdkl.so`:

```bash
export HEXKL_LIB_SUBDIR=6.4.0.2/armv8_android26
```

The probes link `libsdkl.so` through this (`test/unittest/jni_htp/Android.mk`).
Leaving it unset keeps the beta1 layout (`lib/<abi>/`) and will not find beta2.

### 0.2 The Hexagon SDK headers

`remote.h` is a **Hexagon SDK** header, not a HexKL one, and beta2's `sdkl.h`
includes it. When the addon is installed inside an SDK tree
(`$HEXAGON_SDK_ROOT/<ver>/addons/hexkl_addon`) it is two levels up, which is
what the build used to assume. A standalone unzip of the beta2 package has no
SDK above it, and the build fails with:

```
fatal error: 'remote.h' file not found
```

Find it:

```bash
find $HEXKL_SDK_ROOT -name remote.h        # does the addon ship its own incs/?
ls $HEXAGON_SDK_ROOT/incs/remote.h         # or is an SDK installed separately?
```

`Android.mk` looks in `$(HEXKL_ADDON_ROOT)/incs`, then `$(HEXAGON_SDK_ROOT)/incs`,
then the in-SDK-tree path, so if either of those two hits nothing needs setting.
Otherwise point it at whichever directory contains `remote.h`:

```bash
export HEXKL_INCS_DIR=/path/to/hexagon_sdk/incs
```

If neither exists, the Hexagon SDK has to be installed — same `qpm-cli` route as
HexKL. Match the version to `HEXKL_LIB_SUBDIR` (6.4.0.2 here) so the FastRPC
headers agree with the library.

### 0.3 The matching CDSP skeleton

`libsdkl.so` is only the host half; the DSP half is a skeleton `.so` that
FastRPC loads, and it has to be **beta2's, for your device's Hexagon version**:

```bash
find $HEXKL_SDK_ROOT -name '*skel*.so'
```

`hexkl_gemv_probe` prints `sdkl_npu_get_hw_info` before anything else, and
`hex_arch_version` bits [7:0] are the ISA version — `0x75` is V79's
predecessor V75, `0x79` is V79. Match the skeleton to that.

> **Mixing revisions is the most likely way to get a confusing result.** A
> beta1 skeleton under a beta2 `libsdkl.so` is not a configuration anyone
> tested. Use a separate directory so the existing beta1 setup at
> `/data/local/tmp` stays intact:
>
> ```bash
> adb shell mkdir -p /data/local/tmp/hexkl_beta2
> ```

### 0.4 What "PASS" is worth

`hexkl_gemv_probe` prints the CDSP version string (`1_0_56_beta.2_HEXAGON_V73`
style). **If it does not say `beta.2`, the run proves nothing about the beta2
claim** — you measured beta1 with a beta2 header.

---

## 1. `hexkl_gemv_probe` — the unaligned-`n_row` question

### Build

```bash
git pull
export HEXKL_LIB_SUBDIR=6.4.0.2/armv8_android26

ndk-build -C test/unittest/jni_htp NDK_PROJECT_PATH=. \
  APP_BUILD_SCRIPT=Android.mk NDK_APPLICATION_MK=Application.mk \
  hexkl_gemv_probe -j$(nproc)
```

Two things about that invocation are load-bearing:

- **`-C test/unittest/jni_htp` plus the explicit `NDK_PROJECT_PATH` /
  `APP_BUILD_SCRIPT` / `NDK_APPLICATION_MK`.** The folder is not named `jni`,
  so ndk-build will not find any of it by convention — and run from the repo
  root it picks up `test/unittest/jni/Android.mk` instead, which is a different
  build entirely.
- **The module name at the end.** Without it ndk-build resolves every module in
  the file, including the `libnntrainer.so` and `libccapi-nntrainer.so`
  prebuilts, which need a meson build that these probes do not.

> **Clean `obj/` and `libs/` when switching HexKL revisions.** Two prebuilts in
> `Android.mk` have a source file named `libsdkl.so` — the addon's, and the
> armv9 copy meson leaves in `builddir/jni/arm64-v8a/` — so ndk-build gives
> them the same install target and says `overriding recipe for target
> .../libsdkl.so`. On a clean tree the addon's wins, which is what you want,
> but a copy left over from a previous build is not replaced. The symptom is a
> link failure naming beta2-only symbols:
>
> ```
> ld.lld: error: undefined symbol: sdkl_npu_get_hw_info
> ld.lld: error: undefined symbol: sdkl_cpu_i4_rm_to_i4_wh
> ```
>
> That is a beta1 library under beta2 sources. Fix:
>
> ```bash
> rm -rf test/unittest/jni_htp/obj test/unittest/jni_htp/libs
> ```

### Deploy

```bash
D=/data/local/tmp/hexkl_beta2

adb push $HEXKL_SDK_ROOT/lib/$HEXKL_LIB_SUBDIR/libsdkl.so $D/
adb push <the beta2 skel .so for your arch>                $D/
# push the executable from obj/, NOT libs/ -- see the caveat below
adb push test/unittest/jni_htp/obj/local/arm64-v8a/hexkl_gemv_probe $D/
adb shell chmod +x $D/hexkl_gemv_probe
```

> **Push executables from `obj/`, not `libs/`.** ndk-build links into
> `obj/local/arm64-v8a/` and installs a *stripped* copy into
> `libs/arm64-v8a/`, but that install step does not reliably refresh on
> incremental rebuilds — `libs/` can hold old code while `obj/` has your
> latest. The `.so` deps in `libs/` refresh normally.

### Run

```bash
adb shell "cd /data/local/tmp/hexkl_beta2 && \
  LD_LIBRARY_PATH=/data/local/tmp/hexkl_beta2 \
  ADSP_LIBRARY_PATH=/data/local/tmp/hexkl_beta2 \
  ./hexkl_gemv_probe small"
```

### Reading it

Six shapes run. The first, `M=64 N=32 K=32`, is the aligned control — **if that
one fails, something is wrong with the setup, not with the claim.** Stop and fix
the setup before reading anything else.

Each case prints two findings:

```
  --- M=1 N=32 K=32   (Mp would be 64, the lm_head row count)
      result correct (all 32 elements = 32), 143 us
      rows 1..63 untouched (2016 poisoned elements intact)
        -> A can be allocated at M rows: 0.0 MB instead of 0.0 MB
```

| line | means |
| :-- | :-- |
| `n_row=1 REJECTED, rc=…` | the padding is required. Keep `Mp`. Plan §4 is dead. |
| `WRONG: n/m elements differ` | worse than rejection — it runs and lies. Keep `Mp`. |
| `result correct` | the beta2 claim holds at this shape. |
| `rows 1..63 untouched` | `A` can shrink to `M` rows — 37.1 MiB → 594 KB for lm_head. |
| `rows 1..63 WERE written` | `A` must stay at `Mp` rows, but `M` can still be passed. |

The probe never bets on the answer: `A` is always allocated at the padded size
and poison-filled, so a "rows were written" outcome is observed safely rather
than as a heap overrun.

`M=1 N=100 K=32` is the odd one — `N` off a 32 boundary. `hexkl_mm.cpp:700` and
`quantize_qint4_weight` both throw on `N % 32 != 0`; if that case passes, those
throws are over-restrictive. It changes nothing for lm_head (151936 is 4748
tiles exactly) but it matters for any other vocab size.

### Then the real shape

```bash
adb shell "cd /data/local/tmp/hexkl_beta2 && \
  LD_LIBRARY_PATH=/data/local/tmp/hexkl_beta2 \
  ADSP_LIBRARY_PATH=/data/local/tmp/hexkl_beta2 \
  ./hexkl_gemv_probe lmhead"
```

`N=151936, K=1024` at `M=1` and at `M=64`, so the two timings can be compared.
Allocates roughly 150 MB — if it fails to allocate, that is §2 answering, not
this test. **Run §2 first.**

If the two timings are close, the kernel pads internally regardless and what
`M=1` buys is host memory, not kernel time. That is still the 37 MiB
accumulator, the `memset` and the pool thrash, so it is still worth doing — but
it tells you not to expect the matmul itself to get faster.

---

## 2. `hexkl_pin_probe` — the residency budget

```bash
ndk-build -C test/unittest/jni_htp NDK_PROJECT_PATH=. \
  APP_BUILD_SCRIPT=Android.mk NDK_APPLICATION_MK=Application.mk \
  hexkl_pin_probe -j$(nproc)

adb push test/unittest/jni_htp/obj/local/arm64-v8a/hexkl_pin_probe $D/
adb shell chmod +x $D/hexkl_pin_probe

adb shell "cd $D && LD_LIBRARY_PATH=$D ADSP_LIBRARY_PATH=$D \
  ./hexkl_pin_probe sweep"
adb shell "cd $D && LD_LIBRARY_PATH=$D ADSP_LIBRARY_PATH=$D \
  ./hexkl_pin_probe total"
```

`sweep` is the largest single block; `total` is the cumulative budget, which is
the one that governs the design (the 28 decoder weights want 210 MiB of their
own at `fc_layer_dtype=QINT4_HTP`). See
[09](09_lmhead_u8i4_plan.md) §3 for how to read the milestones.

Run it **alone** — a failure mode is a SIGSEGV inside SDKL, and folding it into
another binary makes that impossible to attribute.

---

## 3. Cross-checking with the addon's own micro example

The macro kernel is what matters, but the addon ships a micro-API u8i4 example
with a scalar C reference and a bit-exact check already wired up, so it is a
cheap second opinion — and it exercises a different library
(`libhexkl_micro.a`, DSP-side) than `libsdkl.so`.

```bash
ls $HEXKL_SDK_ROOT/examples                    # find the u8i4 micro example
cp -r $HEXKL_SDK_ROOT/examples/<u8i4_example> ~/gemv_micro
cd ~/gemv_micro
```

Change one line in the source:

```c
#define N_ROW   (64U)      →      #define N_ROW   (1U)
```

Leave `N_COL` and `N_INNER` at 128. The example's own loop arithmetic assumes
they are multiples of 32 — `row_tiles_in_A = A_cols / HEXKL_HMX_INT8_BLOCK_N_INNER`
truncates, and the column loop runs `col < W_cols` at a stride of 32 — so
unaligned `N` or `K` breaks the *example*, not necessarily the hardware. The row
loop (`row < A_rows`, with the copy helpers taking the real `input_rows` /
`output_rows`) is the one that is written to tolerate a partial tile.

```bash
./build.sh
./run_android.sh          # or ./run_simulator.sh
```

`hexkl_test_vector_check_i32` reporting bit-exact means the HMX primitives
handle a single row. That does not prove `sdkl_npu_mm_u8i4_i32` does — but if
micro passes and macro fails, the gap is in the macro wrapper, which is a
concrete thing to raise with Qualcomm rather than a shrug.

While you are in that example, two things in it are worth reading for the
attention work ([08](08_attention_hmx_design.md) §5, Option B):

- the int32 accumulator is read into **VTCM** (`hexkl_micro_hmx_acc_read_int32`
  → `result_offset`) and only `output_rows` reach DDR via
  `hexkl_micro_hmx_copy_32b_to_submatrix`. The 64-row tile is an HMX-internal
  shape, not DDR traffic.
- `hexkl_micro_hmx_copy_submatrix_to_8b_activation` is followed by the comment
  `// No layout needed` — for 8-bit activations the submatrix copy already
  lands them in AH form, so there is no separate `rm_to_ah` cost on that path.
  Whether the fp16 example says the same is worth checking.

> **A naming trap in that file.** The scalar reference
> `matmul_u8i4(..., int32_t *matX, const uint8_t *matA, ...)` uses X for the
> output and A for the input; `hexkl_micro_matmul_u8i4_i32(..., int32_t *matA,
> uint8_t *matX, ...)` uses the opposite (and matches SDKL's A=output
> convention), while *inside* it the locals go back to `X_rows = A_rows`. Read
> the call sites, not the parameter names.

---

## 3.5 `hexkl_decode_probe` — where a decode token actually goes

`hexkl_mm.cpp:487` copies the whole WH weight from host into NPU scratch on
every decode call. For qwen3-0.6b at fp16 that is 30 MiB per layer and 840 MiB
per generated token across 28 layers — the same order as the ~102 ms/token that
9.8 TPS implies. Neither branch instruments `shgemm_f32f16_f32`, so this
measures it from outside instead.

```bash
ndk-build -C test/unittest/jni_htp NDK_PROJECT_PATH=. \
  APP_BUILD_SCRIPT=Android.mk NDK_APPLICATION_MK=Application.mk \
  hexkl_decode_probe -j$(nproc)
adb push test/unittest/jni_htp/obj/local/arm64-v8a/hexkl_decode_probe $D/
adb shell chmod +x $D/hexkl_decode_probe

adb shell "cd $D && LD_LIBRARY_PATH=$D ADSP_LIBRARY_PATH=$D \
  ./hexkl_decode_probe layer"
adb shell "cd $D && LD_LIBRARY_PATH=$D ADSP_LIBRARY_PATH=$D \
  ./hexkl_decode_probe full"
```

It runs qwen3-0.6b's seven FC shapes at `M = 1` two ways — staged (a host WH
weight memcpy'd into NPU scratch before each matmul, what the code does today)
and resident (the weight allocated in NPU memory once, which
[§2](#2-hexkl_pin_probe--the-residency-budget) showed there is room for) — and
reports staging and matmul time separately.

`layer` uses one layer's 30 MiB and multiplies by 28. `full` builds all 28
layers, 840 MiB of host buffers and as much again in NPU memory. **Prefer
`full`**: `layer` re-reads the same 30 MiB every iteration, which is far kinder
to cache than a real token's walk through 840 MiB, so it flatters the staged
case.

What the output decides:

| | |
| :-- | :-- |
| staging ≫ matmul | the decode bottleneck is memory movement, not compute. Making FC weights resident is worth more than anything in [09](09_lmhead_u8i4_plan.md), and the pin probe already showed the budget is there. |
| staging ≈ matmul or less | the 881 MB/token hypothesis is wrong; decode time is elsewhere and the lm_head plan stays the main thread. |

If fewer weights fit in NPU memory than were asked for, the probe says so and
skips the resident half — which is its own answer for that configuration.

---

## 4. `hexkl_layout_probe` — DSP side

Not part of `Android.mk`; nntrainer has no Hexagon toolchain path. Build it
inside the addon's examples tree, which already carries the cross-compile
scripts. See `test/unittest/jni_htp/hexagon/README.md`.

Run it before any `mm_f16`-based attention plan: it prints `hmx_fp16_rate`, and
**a zero there means the device has no HMX fp16**, which invalidates that whole
direction before anything else is worth measuring.

---

## 5. What to do with the results

Paste the output into [09](09_lmhead_u8i4_plan.md) §3 and §4 and adjust the
plan. Specifically:

- `gemv small` all-pass + "rows untouched" → §4 fix (1) is on; the accumulator
  is 594 KB and §9 step 3b goes ahead.
- `gemv small` rejects `M=1` → §4 is dead; the `Mp = 64` wrapper stays, lm_head
  on HMX moves 111 MiB per token against CPU int4's 75, and §10's CPU path
  becomes the likely answer.
- `pin total ≥ 285 MiB` → the whole model can be resident, which is worth far
  more than lm_head (see [09](09_lmhead_u8i4_plan.md) §2 on the decode-time
  weight staging).
- `pin sweep < 74 MiB` → stop; go to §10.
