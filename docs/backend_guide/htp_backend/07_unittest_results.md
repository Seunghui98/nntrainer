# 실제 단말 Unittest 실행 결과 양식

이 문서는 HTP 백엔드 연동 모듈 및 커널 테스트가 단말에서 통과되었을 때 나타나야 하는 표준 보고서 및 콘솔 출력 양식입니다.

---

## 1. `unittest_nntrainer_htp_kernels` — 커널 레벨 테스트

`HtpKernelTest` 스위트: sdkl C API 직접 호출 정확도 및 성능 측정.

```
[==========] Running 18 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 18 tests from HtpKernelTest
[ RUN      ] HtpKernelTest.Accuracy_f16f16_f16
[       OK ] HtpKernelTest.Accuracy_f16f16_f16 ([ms] ms)
[ RUN      ] HtpKernelTest.Accuracy_f16
[       OK ] HtpKernelTest.Accuracy_f16 ([ms] ms)
[ RUN      ] HtpKernelTest.Accuracy_u8i8_i32
[       OK ] HtpKernelTest.Accuracy_u8i8_i32 ([ms] ms)
[ RUN      ] HtpKernelTest.Accuracy_u8i4_i32
[       OK ] HtpKernelTest.Accuracy_u8i4_i32 ([ms] ms)
[ RUN      ] HtpKernelTest.Accuracy_mm_tensor_f16
[       OK ] HtpKernelTest.Accuracy_mm_tensor_f16 ([ms] ms)
[ RUN      ] HtpKernelTest.Constraint_MisalignedRejected_f16f16
[       OK ] HtpKernelTest.Constraint_MisalignedRejected_f16f16 ([ms] ms)
[ RUN      ] HtpKernelTest.Perf_f16f16_f16
[       OK ] HtpKernelTest.Perf_f16f16_f16 ([ms] ms)
[ RUN      ] HtpKernelTest.Perf_f16
[       OK ] HtpKernelTest.Perf_f16 ([ms] ms)
[ RUN      ] HtpKernelTest.Perf_u8i8_i32
[       OK ] HtpKernelTest.Perf_u8i8_i32 ([ms] ms)
[ RUN      ] HtpKernelTest.Perf_u8i4_i32
[       OK ] HtpKernelTest.Perf_u8i4_i32 ([ms] ms)
[ RUN      ] HtpKernelTest.Padding_NonMultipleOf32_f32f16_f32
[       OK ] HtpKernelTest.Padding_NonMultipleOf32_f32f16_f32 ([ms] ms)
[ RUN      ] HtpKernelTest.Accuracy_f32f16_f32_Prefill
[       OK ] HtpKernelTest.Accuracy_f32f16_f32_Prefill ([ms] ms)
[ RUN      ] HtpKernelTest.PrefillWHResidency_ReusesCacheAcrossCalls
[       OK ] HtpKernelTest.PrefillWHResidency_ReusesCacheAcrossCalls ([ms] ms)
[ RUN      ] HtpKernelTest.PrefillWHResidency_PinsMultipleNeverEvicts
[       OK ] HtpKernelTest.PrefillWHResidency_PinsMultipleNeverEvicts ([ms] ms)
[ RUN      ] HtpKernelTest.Perf_f32f16_f32_Prefill
[       OK ] HtpKernelTest.Perf_f32f16_f32_Prefill ([ms] ms)
[ RUN      ] HtpKernelTest.PhaseTiming_TransientPrefillBreakdown
[       OK ] HtpKernelTest.PhaseTiming_TransientPrefillBreakdown ([ms] ms)
[ RUN      ] HtpKernelTest.PoolProbe_MeasureMaxResidentBytes
[       OK ] HtpKernelTest.PoolProbe_MeasureMaxResidentBytes ([ms] ms)
[ RUN      ] HtpKernelTest.PoolProbe_MeasureMaxSustainedPinBytes
[       OK ] HtpKernelTest.PoolProbe_MeasureMaxSustainedPinBytes ([ms] ms)
[----------] 18 tests from HtpKernelTest ([ms] ms total)
[==========] 18 tests from 1 test suite ran.
[  PASSED  ] 18 tests.
```

NPU 없는 환경(ADSP_LIBRARY_PATH 미설정 등)에서는 accuracy/perf 테스트가 `[ SKIPPED ]`로 표시됩니다.

---

## 2. `unittest_nntrainer_htp_backend` — nntrainer API 통합 테스트

4개 스위트: FP16 GEMM, Fallback, QINT8 GEMM, 라우팅 디스패치.

```
[==========] Running 13 tests from 4 test suites.
[----------] 6 tests from HtpShgemmTest
[ RUN      ] HtpShgemmTest.AccuracyVsCpu
[       OK ] HtpShgemmTest.AccuracyVsCpu ([ms] ms)
[ RUN      ] HtpShgemmTest.AlphaBetaHandling
[       OK ] HtpShgemmTest.AlphaBetaHandling ([ms] ms)
[ RUN      ] HtpShgemmTest.BetaNonZeroThrows
[       OK ] HtpShgemmTest.BetaNonZeroThrows ([ms] ms)
[ RUN      ] HtpShgemmTest.MNotAlignedThrows
[       OK ] HtpShgemmTest.MNotAlignedThrows ([ms] ms)
[ RUN      ] HtpShgemmTest.EdgeCase_MinValidShape
[       OK ] HtpShgemmTest.EdgeCase_MinValidShape ([ms] ms)
[ RUN      ] HtpShgemmTest.EdgeCase_SingleRow
[       OK ] HtpShgemmTest.EdgeCase_SingleRow ([ms] ms)
[----------] 6 tests from HtpShgemmTest ([ms] ms total)

[----------] 2 tests from HtpFallbackTest
[ RUN      ] HtpFallbackTest.SupportsShgemmTracksBackendState
[       OK ] HtpFallbackTest.SupportsShgemmTracksBackendState ([ms] ms)
[ RUN      ] HtpFallbackTest.CpuOpsNeverAdvertisesShgemm
[       OK ] HtpFallbackTest.CpuOpsNeverAdvertisesShgemm ([ms] ms)
[----------] 2 tests from HtpFallbackTest ([ms] ms total)

[----------] 3 tests from HtpU8i8Test
[ RUN      ] HtpU8i8Test.Accuracy_VsCpu
[       OK ] HtpU8i8Test.Accuracy_VsCpu ([ms] ms)
[ RUN      ] HtpU8i8Test.ZpCorrApplied
[       OK ] HtpU8i8Test.ZpCorrApplied ([ms] ms)
[ RUN      ] HtpU8i8Test.AlignmentGuard_NNot32
[       OK ] HtpU8i8Test.AlignmentGuard_NNot32 ([ms] ms)
[----------] 3 tests from HtpU8i8Test ([ms] ms total)

[----------] 2 tests from HtpDispatchTest
[ RUN      ] HtpDispatchTest.RoutesToHtp_WhenMAligned
[       OK ] HtpDispatchTest.RoutesToHtp_WhenMAligned ([ms] ms)
[ RUN      ] HtpDispatchTest.PadsAndRunsHtp_WhenMMisaligned
[       OK ] HtpDispatchTest.PadsAndRunsHtp_WhenMMisaligned ([ms] ms)
[----------] 2 tests from HtpDispatchTest ([ms] ms total)

[==========] 13 tests from 4 test suites ran.
[  PASSED  ] 13 tests.
```

`HtpDispatchTest.RoutesAttentionToCpu_WhenAttentionFallbackDisabled` / `RoutesFfnToCpu_WhenFfnFallbackDisabled`는 NPU 없는 환경에서도 실행되는 fallback 경로 검증 테스트입니다.

---

## 3. gtest XML 결과 스켈레톤 (`--gtest_output=xml`)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<testsuites tests="31" failures="0" disabled="0" errors="0" time="[TBD]" name="AllTests">
  <!-- unittest_nntrainer_htp_kernels -->
  <testsuite name="HtpKernelTest" tests="18" failures="0" disabled="0" errors="0" time="[TBD]">
    <testcase name="Accuracy_f16f16_f16" status="run" result="completed" time="[TBD]" classname="HtpKernelTest" />
    <testcase name="Accuracy_f16" status="run" result="completed" time="[TBD]" classname="HtpKernelTest" />
    <testcase name="Accuracy_u8i8_i32" status="run" result="completed" time="[TBD]" classname="HtpKernelTest" />
    <testcase name="Accuracy_u8i4_i32" status="run" result="completed" time="[TBD]" classname="HtpKernelTest" />
    <testcase name="Accuracy_mm_tensor_f16" status="run" result="completed" time="[TBD]" classname="HtpKernelTest" />
    <testcase name="Constraint_MisalignedRejected_f16f16" status="run" result="completed" time="[TBD]" classname="HtpKernelTest" />
    <testcase name="Perf_f16f16_f16" status="run" result="completed" time="[TBD]" classname="HtpKernelTest" />
    <testcase name="Perf_f16" status="run" result="completed" time="[TBD]" classname="HtpKernelTest" />
    <testcase name="Perf_u8i8_i32" status="run" result="completed" time="[TBD]" classname="HtpKernelTest" />
    <testcase name="Perf_u8i4_i32" status="run" result="completed" time="[TBD]" classname="HtpKernelTest" />
    <testcase name="Padding_NonMultipleOf32_f32f16_f32" status="run" result="completed" time="[TBD]" classname="HtpKernelTest" />
    <testcase name="Accuracy_f32f16_f32_Prefill" status="run" result="completed" time="[TBD]" classname="HtpKernelTest" />
    <testcase name="PrefillWHResidency_ReusesCacheAcrossCalls" status="run" result="completed" time="[TBD]" classname="HtpKernelTest" />
    <testcase name="PrefillWHResidency_PinsMultipleNeverEvicts" status="run" result="completed" time="[TBD]" classname="HtpKernelTest" />
    <testcase name="Perf_f32f16_f32_Prefill" status="run" result="completed" time="[TBD]" classname="HtpKernelTest" />
    <testcase name="PhaseTiming_TransientPrefillBreakdown" status="run" result="completed" time="[TBD]" classname="HtpKernelTest" />
    <testcase name="PoolProbe_MeasureMaxResidentBytes" status="run" result="completed" time="[TBD]" classname="HtpKernelTest" />
    <testcase name="PoolProbe_MeasureMaxSustainedPinBytes" status="run" result="completed" time="[TBD]" classname="HtpKernelTest" />
  </testsuite>
  <!-- unittest_nntrainer_htp_backend -->
  <testsuite name="HtpShgemmTest" tests="6" failures="0" disabled="0" errors="0" time="[TBD]">
    <testcase name="AccuracyVsCpu" status="run" result="completed" time="[TBD]" classname="HtpShgemmTest" />
    <testcase name="AlphaBetaHandling" status="run" result="completed" time="[TBD]" classname="HtpShgemmTest" />
    <testcase name="BetaNonZeroThrows" status="run" result="completed" time="[TBD]" classname="HtpShgemmTest" />
    <testcase name="MNotAlignedThrows" status="run" result="completed" time="[TBD]" classname="HtpShgemmTest" />
    <testcase name="EdgeCase_MinValidShape" status="run" result="completed" time="[TBD]" classname="HtpShgemmTest" />
    <testcase name="EdgeCase_SingleRow" status="run" result="completed" time="[TBD]" classname="HtpShgemmTest" />
  </testsuite>
  <testsuite name="HtpFallbackTest" tests="2" failures="0" disabled="0" errors="0" time="[TBD]">
    <testcase name="SupportsShgemmTracksBackendState" status="run" result="completed" time="[TBD]" classname="HtpFallbackTest" />
    <testcase name="CpuOpsNeverAdvertisesShgemm" status="run" result="completed" time="[TBD]" classname="HtpFallbackTest" />
  </testsuite>
  <testsuite name="HtpU8i8Test" tests="3" failures="0" disabled="0" errors="0" time="[TBD]">
    <testcase name="Accuracy_VsCpu" status="run" result="completed" time="[TBD]" classname="HtpU8i8Test" />
    <testcase name="ZpCorrApplied" status="run" result="completed" time="[TBD]" classname="HtpU8i8Test" />
    <testcase name="AlignmentGuard_NNot32" status="run" result="completed" time="[TBD]" classname="HtpU8i8Test" />
  </testsuite>
  <testsuite name="HtpDispatchTest" tests="2" failures="0" disabled="0" errors="0" time="[TBD]">
    <testcase name="RoutesToHtp_WhenMAligned" status="run" result="completed" time="[TBD]" classname="HtpDispatchTest" />
    <testcase name="PadsAndRunsHtp_WhenMMisaligned" status="run" result="completed" time="[TBD]" classname="HtpDispatchTest" />
  </testsuite>
</testsuites>
```
