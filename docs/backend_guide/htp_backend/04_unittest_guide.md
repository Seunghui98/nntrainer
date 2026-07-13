# HTP Unittest Guide and Results

Deploying cross-compiled HTP unittest binaries to a target device (e.g. Galaxy S25 Ultra) for remote verification, and recording results.

## 1. Building and Running Unittests

### Build
To avoid rebuilding the entire module, compile only the changed test binary:

```bash
ninja -C build_android test/unittest/unittest_nntrainer_htp_kernels
ninja -C build_android test/unittest/unittest_nntrainer_htp_backend
```

### Deploy via ADB
Target device: **Galaxy S25 Ultra** (SM-S938N)

Push the cross-compiled binaries and the SDK's runtime libraries directly to `/data/local/tmp` on the device.

```bash
# 1. Push the compiled unittest binaries
adb push build_android/test/unittest/unittest_nntrainer_htp_kernels /data/local/tmp/
adb push build_android/test/unittest/unittest_nntrainer_htp_backend /data/local/tmp/

# 2. Push the HexKL prebuilt shared library
adb push $HEXKL_SDK_ROOT/lib/armv8_android26/libsdkl.so /data/local/tmp/

# [Note] The CDSP device skeleton for HTP (libhexkl_skel.so, V79) must also already be in /data/local/tmp.
```

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
adb -s R3CY205ZMND shell \
  "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ADSP_LIBRARY_PATH=/data/local/tmp \
   NNTR_HTP_DISABLE_PREBAKED_WH=1 ./unittest_nntrainer_htp_kernels --gtest_color=no"

# Recompute the transient conversion on every pre-baked hit and memcmp it (per-weight OK/MISMATCH log)
adb -s R3CY205ZMND shell \
  "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ADSP_LIBRARY_PATH=/data/local/tmp \
   NNTR_HTP_VERIFY_PREBAKED_WH=1 ./unittest_nntrainer_htp_backend --gtest_color=no"

# Diff live NPU output against the CPU reference
adb -s R3CY205ZMND shell \
  "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ADSP_LIBRARY_PATH=/data/local/tmp \
   NNTR_HTP_VERIFY_PREFILL_MM=1 ./unittest_nntrainer_htp_kernels --gtest_color=no"
```

## 2. Test Source File Composition

Two precision-level test suites verify the HTP backend.

| Binary | Source File | Role | Test Suites |
| :--- | :--- | :--- | :--- |
| `unittest_nntrainer_htp_kernels` | `test/unittest/unittest_nntrainer_htp_kernels.cpp` | Calls the sdkl C API directly — measures kernel numerical accuracy and latency | `HtpKernelTest` **18** |
| `unittest_nntrainer_htp_backend` | `test/unittest/unittest_nntrainer_htp_backend.cpp` | Integration test at the nntrainer ComputeOps API level — verifies QINT8 dispatch | `HtpShgemmTest`(6) + `HtpFallbackTest`(2) + `HtpU8i8Test`(3) + `HtpDispatchTest`(4), **15 tests, 4 suites** |

NPU-dependent tests auto-skip via `GTEST_SKIP()` when `HtpBackend::global().enabled() == false`. `HtpFallbackTest` and part of `HtpDispatchTest` (the `*FallbackDisabled` tests) verify the fallback path and run even without an NPU.

`test/unittest/jni_htp/sdkl_rm_to_wh_i8_probe.cpp` is a standalone diagnostic binary that isolates `sdkl_cpu_rm_to_wh_i8_inplace()`'s behavior per buffer kind — used for debugging QINT8 weight-layout conversion issues.

```bash
# Build, push to device, and run (presets: small / qwen_attn / qwen_ffn_up / qwen_ffn_down)
adb -s R3CY205ZMND shell \
  "cd /data/local/tmp && \
   LD_LIBRARY_PATH=/data/local/tmp ADSP_LIBRARY_PATH=/data/local/tmp \
   ./sdkl_rm_to_wh_i8_probe --preset qwen_attn --buf all"
```

## 3. Tests per File

### `unittest_nntrainer_htp_kernels` — `HtpKernelTest` (18)
- Accuracy_f16f16_f16
- Accuracy_f16
- Accuracy_u8i8_i32
- Accuracy_u8i4_i32
- Accuracy_mm_tensor_f16
- Constraint_MisalignedRejected_f16f16
- Perf_f16f16_f16
- Perf_f16
- Perf_u8i8_i32
- Perf_u8i4_i32
- Padding_NonMultipleOf32_f32f16_f32
- Accuracy_f32f16_f32_Prefill
- PrefillWHResidency_ReusesCacheAcrossCalls
- PrefillWHResidency_PinsMultipleNeverEvicts
- Perf_f32f16_f32_Prefill
- PhaseTiming_TransientPrefillBreakdown
- PoolProbe_MeasureMaxResidentBytes
- PoolProbe_MeasureMaxSustainedPinBytes

### `unittest_nntrainer_htp_backend` — 4 suites (15)

**HtpShgemmTest** (6)
- AccuracyVsCpu
- AlphaBetaHandling
- BetaNonZeroThrows
- MNotAlignedThrows
- EdgeCase_MinValidShape
- EdgeCase_SingleRow

**HtpFallbackTest** (2)
- SupportsShgemmTracksBackendState
- CpuOpsNeverAdvertisesShgemm

**HtpU8i8Test** (3)
- Accuracy_VsCpu
- ZpCorrApplied
- AlignmentGuard_NNot32

**HtpDispatchTest** (4)
- RoutesToHtp_WhenMAligned
- PadsAndRunsHtp_WhenMMisaligned
- RoutesAttentionToCpu_WhenAttentionFallbackDisabled (runs even without NPU)
- RoutesFfnToCpu_WhenFfnFallbackDisabled (runs even without NPU)

## 4. Test Results Record

Record results in the tables below after running the commands in §1. (Legend: ✅ PASS / ❌ FAIL / ⏭️ SKIPPED — accuracy/perf tests are expected to SKIP when the NPU isn't up)

### 4.1 `unittest_nntrainer_htp_kernels` — `HtpKernelTest` (18)

| Test | Result | Time (ms) | Notes |
| :--- | :---: | :---: | :--- |
| Accuracy_f16f16_f16 | | | |
| Accuracy_f16 | | | |
| Accuracy_u8i8_i32 | | | |
| Accuracy_u8i4_i32 | | | |
| Accuracy_mm_tensor_f16 | | | |
| Constraint_MisalignedRejected_f16f16 | | | |
| Perf_f16f16_f16 | | | |
| Perf_f16 | | | |
| Perf_u8i8_i32 | | | |
| Perf_u8i4_i32 | | | |
| Padding_NonMultipleOf32_f32f16_f32 | | | |
| Accuracy_f32f16_f32_Prefill | | | |
| PrefillWHResidency_ReusesCacheAcrossCalls | | | |
| PrefillWHResidency_PinsMultipleNeverEvicts | | | |
| Perf_f32f16_f32_Prefill | | | |
| PhaseTiming_TransientPrefillBreakdown | | | |
| PoolProbe_MeasureMaxResidentBytes | | | |
| PoolProbe_MeasureMaxSustainedPinBytes | | | |

### 4.2 `unittest_nntrainer_htp_backend` — 4 suites (15)

| Suite | Test | Result | Time (ms) | Notes |
| :--- | :--- | :---: | :---: | :--- |
| HtpShgemmTest | AccuracyVsCpu | | | |
| HtpShgemmTest | AlphaBetaHandling | | | |
| HtpShgemmTest | BetaNonZeroThrows | | | |
| HtpShgemmTest | MNotAlignedThrows | | | |
| HtpShgemmTest | EdgeCase_MinValidShape | | | |
| HtpShgemmTest | EdgeCase_SingleRow | | | |
| HtpFallbackTest | SupportsShgemmTracksBackendState | | | |
| HtpFallbackTest | CpuOpsNeverAdvertisesShgemm | | | |
| HtpU8i8Test | Accuracy_VsCpu | | | |
| HtpU8i8Test | ZpCorrApplied | | | |
| HtpU8i8Test | AlignmentGuard_NNot32 | | | |
| HtpDispatchTest | RoutesToHtp_WhenMAligned | | | |
| HtpDispatchTest | PadsAndRunsHtp_WhenMMisaligned | | | |
| HtpDispatchTest | RoutesAttentionToCpu_WhenAttentionFallbackDisabled | | | Runs even without NPU |
| HtpDispatchTest | RoutesFfnToCpu_WhenFfnFallbackDisabled | | | Runs even without NPU |
