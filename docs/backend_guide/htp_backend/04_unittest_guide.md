# HTP Unittest Guide and Results

Deploying cross-compiled HTP unittest binaries to a target device (e.g. Galaxy S25 Ultra) for remote verification, and recording results.

## 1. Building and Running Unittests

### Prerequisites and Build

The HTP unittest binaries are **not** built by meson/ninja: the top-level `meson.build` skips `subdir('test')` on `platform=android` (`test is not supported in android build, test skipped`), so `ninja -C <builddir> test/unittest/...` fails with `unknown target`. They are instead built with **ndk-build** from `test/unittest/jni_htp/Android.mk`, which links against the prebuilt `libnntrainer.so` / `libccapi-nntrainer.so` / `libsdkl.so` produced by the library build.

**1. Set the cross-build environment variables** (see [02 §1](02_build_and_run.md)):

```bash
export HEXKL_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1/addons/hexkl_addon
export ANDROID_NDK=/opt/android-ndk-r26d
export PATH=$ANDROID_NDK:$PATH
```

**2. Build the HTP library** ([02 §2](02_build_and_run.md) golden path). This produces the prebuilt `.so`s under `builddir/jni/arm64-v8a/` that `jni_htp/Android.mk` links against:

```bash
./tools/package_android.sh \
  --arm-arch=armv8.2-a \
  -Denable-htp=true \
  -Dhexkl-sdk-root=$HEXKL_SDK_ROOT \
  -Dhexkl-lib-subdir=armv8_android26 \
  -Dmmap-read=false \
  -Dwerror=false
```

**3. Build the test binaries with ndk-build.** `jni_htp/` uses a flat layout (`Android.mk` + `Application.mk` directly in the folder), so point ndk-build at them explicitly:

```bash
ndk-build -C test/unittest/jni_htp \
  NDK_PROJECT_PATH=. \
  APP_BUILD_SCRIPT=Android.mk \
  NDK_APPLICATION_MK=Application.mk \
  -j$(nproc)
```

Output (binaries + all runtime `.so` dependencies) lands in `test/unittest/jni_htp/libs/arm64-v8a/`:
- `unittest_nntrainer_htp_kernels`, `unittest_nntrainer_htp_backend`
- `libnntrainer.so`, `libccapi-nntrainer.so`, `libsdkl.so`, `libc++_shared.so`

To build a single binary, append its module name (e.g. `unittest_nntrainer_htp_kernels`) to the command.

### Deploy via ADB
Target device: **Galaxy S25 Ultra** (SM-S938N)

The ndk-build output dir contains the binaries and every runtime `.so` they link against, so push `libs/arm64-v8a/` to `/data/local/tmp`.

```bash
# 1. Push the runtime .so deps (libnntrainer/libccapi-nntrainer/libsdkl/libc++_shared)
adb push test/unittest/jni_htp/libs/arm64-v8a/. /data/local/tmp/

# 2. Push the freshly-linked executables from obj/ (see caveat below)
adb push test/unittest/jni_htp/obj/local/arm64-v8a/unittest_nntrainer_htp_kernels /data/local/tmp/
adb push test/unittest/jni_htp/obj/local/arm64-v8a/unittest_nntrainer_htp_backend /data/local/tmp/

# [Note] The CDSP device skeleton for HTP (libhexkl_skel.so, V79) must also already be in /data/local/tmp.
```

> **Caveat — push executables from `obj/`, not `libs/`, on incremental rebuilds.** ndk-build links each executable in `obj/local/arm64-v8a/` and installs a *stripped* copy into `libs/arm64-v8a/`, but the strip-install step does **not** reliably refresh on incremental rebuilds — the `libs/` copy can stay stale (old code) while `obj/` has your latest changes. Always push the executable from `obj/local/arm64-v8a/` (unstripped, larger, but current). The `.so` deps in `libs/` are refreshed normally.

### Run Remotely
Inject the shared-library path (`LD_LIBRARY_PATH`) and the CDSP skeleton linker path (`ADSP_LIBRARY_PATH`):

```bash
# Run the kernel-level tests
adb shell \
  "cd /data/local/tmp && \
   LD_LIBRARY_PATH=/data/local/tmp ADSP_LIBRARY_PATH=/data/local/tmp \
   ./unittest_nntrainer_htp_kernels"

# Run the nntrainer API-level integration tests
adb shell \
  "cd /data/local/tmp && \
   LD_LIBRARY_PATH=/data/local/tmp ADSP_LIBRARY_PATH=/data/local/tmp \
   ./unittest_nntrainer_htp_backend"
```

Add `--gtest_filter=<Suite>.<Test>` to run a single test:

```bash
adb shell \
  "cd /data/local/tmp && \
   LD_LIBRARY_PATH=/data/local/tmp ADSP_LIBRARY_PATH=/data/local/tmp \
   ./unittest_nntrainer_htp_kernels \
   --gtest_filter=HtpKernelTest.Accuracy_f32f16_f32_Prefill"
```

### Run with Diagnostic Env Toggles
Inject these `NNTR_HTP_*` env vars into the run command to closely verify the WH cache and prefill matmul path.

```bash
# Force lookupPrefillWH to miss, using the known-good transient RM→WH path (bisection)
adb shell \
  "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ADSP_LIBRARY_PATH=/data/local/tmp \
   NNTR_HTP_DISABLE_PREBAKED_WH=1 ./unittest_nntrainer_htp_kernels --gtest_color=no"

```

`NNTR_HTP_DISABLE_PREBAKED_WH` is the only runtime toggle the backend reads.
Correctness of the pre-baked path against the transient one is asserted by
`PrefillWHRegistry_UsedByShgemmMatchesTransient` and
`OfflineWH_ConversionIsDeterministicAndByteIdentical` rather than by a
diagnostic env var, so there is nothing to enable by hand.

## 2. Test Source File Composition

Two test binaries verify the HTP backend at different levels.

| Binary | Source File | Role | Test Suites |
| :--- | :--- | :--- | :--- |
| `unittest_nntrainer_htp_kernels` | `test/unittest/unittest_nntrainer_htp_kernels.cpp` | Calls the sdkl C API directly — measures kernel numerical accuracy and latency for the kernels the backend actually uses (`f32f16_f32` prefill, `u8i8_i32`) plus WH cache/registry behaviour | `HtpKernelTest` **10** |
| `unittest_nntrainer_htp_backend` | `test/unittest/unittest_nntrainer_htp_backend.cpp` | Integration tests at the nntrainer ComputeOps API level — shgemm/QINT8 dispatch plus the WH trailer codec, prefill-WH registry/loader, offline-bake gate, and backend lifecycle | `HtpShgemmTest`(5) + `HtpU8i8Test`(3) + `HtpFallbackTest`(2) + `HtpDispatchTest`(2) + `WHTrailerCodec`(2) + `WHTrailerLoad`(2) + `HtpPrefillWH`(1) + `WHBakeGate`(1) + `HtpBackendLifecycle`(1), **19 tests, 9 suites** |

NPU-dependent tests auto-skip via `GTEST_SKIP()` when `HtpBackend::global().enabled() == false`. Tests that run even without an NPU: `HtpFallbackTest` and the entire `WHTrailerCodec` / `WHTrailerLoad` / `HtpPrefillWH` / `WHBakeGate` / `HtpBackendLifecycle` group — they exercise host-side WH-trailer/registry/bake/lifecycle logic rather than the NPU.

## 3. Tests per File

### `unittest_nntrainer_htp_kernels` — `HtpKernelTest` (10)
- Accuracy_u8i8_i32
- Perf_u8i8_i32
- Padding_NonMultipleOf32_f32f16_f32
- Accuracy_f32f16_f32_Prefill
- Accuracy_f32f16_f32_Prefill_TransBFalse_KNLayout
- PrefillWHResidency_ReusesCacheAcrossCalls
- PrefillWHResidency_PinsMultipleNeverEvicts
- PrefillWHRegistry_UsedByShgemmMatchesTransient
- ScratchReuse_MixedShapesStayCorrect
- OfflineWH_ConversionIsDeterministicAndByteIdentical

### `unittest_nntrainer_htp_backend` — 9 suites (19)

**HtpShgemmTest** (5)
- AccuracyVsCpu
- ZeroWeightGivesZero
- AlphaBetaHandling
- BetaNonZeroThrows
- NNotAlignedThrows

**HtpFallbackTest** (2) — runs even without NPU
- SupportsShgemmTracksBackendState
- CpuOpsNeverAdvertisesShgemm

**HtpU8i8Test** (3)
- Accuracy_VsCpu
- ZpCorrApplied
- AlignmentGuard_NNot32

**HtpDispatchTest** (2)
- RoutesToHtp_WhenMAligned
- PadsAndRunsHtp_WhenMMisaligned

**WHTrailerCodec** (2) — host-side, runs without NPU
- RoundTripsEntries
- ReturnsFalseOnPlainData

**WHTrailerLoad** (2) — host-side, runs without NPU
- RegisterThenLookupReturnsBytes
- InferenceModeLoadRegistersWH

**HtpPrefillWH** (1) — host-side, runs without NPU
- disableToggleForcesMiss

**WHBakeGate** (1) — host-side, runs without NPU
- RespectsLayerDtypeMapOverride

**HtpBackendLifecycle** (1) — runs everywhere
- npuAliveTracksEnabled

## 4. Test Results Record

Record results in the tables below after running the commands in §1. (Legend: ✅ PASS / ❌ FAIL / ⏭️ SKIPPED — accuracy/perf tests are expected to SKIP when the NPU isn't up)

Last run: **2026-07-13, Galaxy S25 Ultra (ADB `R3CY205ZMND`, V79 skel)**. Kernels: **12 passed, 2 skipped** (default run; pool probes opt-in). Backend: **18 passed**.

### 4.1 `unittest_nntrainer_htp_kernels` — `HtpKernelTest` (10)

| Test | Result | Time (ms) | Notes |
| :--- | :---: | :---: | :--- |
| Accuracy_u8i8_i32 | ✅ | 58 | |
| Perf_u8i8_i32 | ✅ | 12981 | all shapes SKIP (M not mult. of 64) — perf sweep no-op, test passes |
| Padding_NonMultipleOf32_f32f16_f32 | ✅ | 1 | |
| Accuracy_f32f16_f32_Prefill | ✅ | 325 | |
| Accuracy_f32f16_f32_Prefill_TransBFalse_KNLayout | ✅ | 466 | |
| PrefillWHResidency_ReusesCacheAcrossCalls | ✅ | 33 | |
| PrefillWHResidency_PinsMultipleNeverEvicts | ✅ | 45 | |
| PrefillWHRegistry_UsedByShgemmMatchesTransient | ✅ | 16 | |
| ScratchReuse_MixedShapesStayCorrect | ✅ | 115 | |
| OfflineWH_ConversionIsDeterministicAndByteIdentical | ✅ | 217 | |

### 4.2 `unittest_nntrainer_htp_backend` — 9 suites (19)

| Suite | Test | Result | Time (ms) | Notes |
| :--- | :--- | :---: | :---: | :--- |
| HtpShgemmTest | AccuracyVsCpu | ✅ | 72 | |
| HtpShgemmTest | ZeroWeightGivesZero | — | — | |
| HtpShgemmTest | AlphaBetaHandling | ✅ | 0 | |
| HtpShgemmTest | BetaNonZeroThrows | ✅ | 0 | |
| HtpShgemmTest | NNotAlignedThrows | ✅ | 0 | |
| HtpFallbackTest | SupportsShgemmTracksBackendState | ✅ | 0 | Runs even without NPU |
| HtpFallbackTest | CpuOpsNeverAdvertisesShgemm | ✅ | 0 | Runs even without NPU |
| HtpU8i8Test | Accuracy_VsCpu | ✅ | 2 | |
| HtpU8i8Test | ZpCorrApplied | ✅ | 4 | |
| HtpU8i8Test | AlignmentGuard_NNot32 | ✅ | 0 | |
| HtpDispatchTest | RoutesToHtp_WhenMAligned | ✅ | 3 | |
| HtpDispatchTest | PadsAndRunsHtp_WhenMMisaligned | ✅ | 2 | |
| WHTrailerCodec | RoundTripsEntries | ✅ | 0 | Host-side, runs without NPU |
| WHTrailerCodec | ReturnsFalseOnPlainData | ✅ | 0 | Host-side, runs without NPU |
| WHTrailerLoad | RegisterThenLookupReturnsBytes | ✅ | 0 | Host-side, runs without NPU |
| WHTrailerLoad | InferenceModeLoadRegistersWH | ✅ | 7 | Host-side, runs without NPU |
| HtpPrefillWH | disableToggleForcesMiss | ✅ | 0 | Host-side, runs without NPU |
| WHBakeGate | RespectsLayerDtypeMapOverride | ✅ | 1 | Host-side, runs without NPU |
| HtpBackendLifecycle | npuAliveTracksEnabled | ✅ | 0 | Runs everywhere |
