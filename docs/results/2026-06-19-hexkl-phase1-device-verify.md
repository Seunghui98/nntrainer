# HexKL Phase 1 디바이스 검증 결과

날짜: 2026-06-19  
브랜치: `hexkl_integration`  
기기: Galaxy S25 Ultra · ADB ID `R3CY205ZMND`  
빌드: NDK r26d · `armv8.2-a+fp16+dotprod+i8mm` (이유: 아래 참조)

---

## 1. 빌드 환경 — 발견된 이슈 및 수정

### 1.1 submodule 미초기화
- `subprojects/iniparser/`, `subprojects/googletest/` 가 비어 있음
- `git submodule update --init` 으로 해결

### 1.2 `sdkl.h` include 경로 누락 (NDK 빌드)
- `jni/Android.mk.in`의 sdkl 모듈에 `LOCAL_EXPORT_C_INCLUDES`가 없었음
- `jni/meson.build`에 `MESON_HEXKL_INCLUDE_DIR` 추가, `Android.mk.in` 패치

### 1.3 KleidiAI SVE 소스 누락 (NDK 빌드)
- x86 호스트 gcc가 `-march=armv8.2-a+fp16+sve+i8mm` has_argument 체크에서 `false` 반환
- armv9.2-a 빌드임에도 SVE 소스(`kai_common_sve_asm.S`, 3개 SVE matmul `.c/.S`)가 `LOCAL_SRC_FILES`에서 빠짐
- `jni/meson.build`에서 `arm_march_flag.contains('sve')` 조건부로 SVE 소스 명시 추가

### 1.4 `nntrainer::_FP16` namespace 오류 (테스트 파일)
- `test/unittest/unittest_nntrainer_htp_backend.cpp`에서 `nntrainer::_FP16` 사용
- `_FP16`은 `tensor_dim.h`에서 정의된 전처리 매크로 (`__fp16` 또는 `_Float16`)이므로 namespace 지정 불가
- x86 빌드에서는 `ENABLE_HEXKL` 미정의로 컴파일 안 됐기 때문에 발견되지 않음
- 6곳 모두 `_FP16`으로 수정

### 1.5 armv9.2-a → armv8.2-a 변경
- 기기 `/proc/cpuinfo`에 `sve` 없음 — 커널이 SVE를 userspace에 노출하지 않음
- armv9.2-a로 빌드한 바이너리 실행 시 `kai_get_sve_vector_length_u8()`(`cntb` 명령어)에서 SIGILL 발생
- armv8.2-a+fp16+dotprod+i8mm로 재빌드하여 해결
- **HTP NPU 연산은 FastRPC로 DSP에 위임되므로 ARM SVE와 무관**

### 1.6 NPU skel 미배포
- `libhexkl_skel.so` (Hexagon v79용)를 기기 `/data/local/tmp/`에 push 필요
- `ADSP_LIBRARY_PATH=/data/local/tmp` 환경변수 설정 필요

---

## 2. 테스트 결과

### 2.1 NPU 없는 환경 (ADSP_LIBRARY_PATH 미설정)

```
[==========] 7 tests from 2 test suites ran.
[  PASSED  ] 2 tests.
[  SKIPPED ] 5 tests.
  HtpShgemmTest.AccuracyVsCpu        [ SKIPPED ] — NPU not available
  HtpShgemmTest.AlphaBetaHandling    [ SKIPPED ] — NPU not available
  HtpShgemmTest.BetaNonZeroThrows    [ SKIPPED ] — NPU not available
  HtpShgemmTest.EdgeCase_SingleElement [ SKIPPED ] — NPU not available
  HtpShgemmTest.EdgeCase_SingleRow   [ SKIPPED ] — NPU not available
  HtpFallbackTest.SupportsShgemmTracksBackendState [ OK ]
  HtpFallbackTest.CpuOpsNeverAdvertisesShgemm      [ OK ]
```

핸드오프 예측(5 SKIP + 2 OK)과 정확히 일치. fallback 로직 정상.

### 2.2 NPU 활성화 (ADSP_LIBRARY_PATH=/data/local/tmp + skel push)

```
[  PASSED  ] 3 tests.
[  FAILED  ] 3 tests.
  HtpShgemmTest.BetaNonZeroThrows          [ OK ]  ← beta≠0 throw 정상
  HtpFallbackTest.SupportsShgemmTracksBackendState [ OK ]
  HtpFallbackTest.CpuOpsNeverAdvertisesShgemm      [ OK ]

  HtpShgemmTest.AccuracyVsCpu              [ FAILED ] — WH 변환 제약 위반
  HtpShgemmTest.AlphaBetaHandling          [ FAILED ] — WH 변환 제약 위반
  HtpShgemmTest.EdgeCase_SingleElement     [ FAILED ] — WH 변환 제약 위반
  HtpShgemmTest.EdgeCase_SingleRow         [ SIGILL ]
```

---

## 3. 발견된 버그 — Phase 2 수정 필요

### Bug-1: `sdkl_cpu_rm_to_wh_f16_inplace` — n_row must be multiple of 32

```
[SDKL][ERROR] sdkl_cpu_rm_to_wh_f16_inplace: n_row must be multiple of 32, got 4
```

- **발생 조건**: `shgemm_f32f16_f32()` 내부의 WH(Winograd Hash) weight 변환 단계
- **원인**: `sdkl_cpu_rm_to_wh_f16_inplace(B, N, K)` 호출 시 N이 32의 배수여야 하는 제약
- **영향**: AccuracyVsCpu (M=4, N=4, K=8), AlphaBetaHandling (M=8, N=8, K=64), EdgeCase_SingleElement (1×1×1) 실패
- **수정 위치**: `nntrainer/tensor/htp_backend/hmx_ops/hexkl_mm.cpp`
  - N < 32 또는 N % 32 ≠ 0 인 경우 CPU fallback으로 전환하거나 예외를 던져야 함
  - 또는 N을 32의 배수로 패딩 후 변환

### Bug-2: EdgeCase_SingleRow (M=1, N=64, K=128) — SIGILL

- N=64, K=128 모두 32의 배수이므로 WH 제약은 통과해야 함
- SIGILL 원인 미확정 (디버그 심볼 필요)
- 가능성: NEON/i8mm 명령어 alignment 이슈, 또는 KleidiAI M=1 케이스 처리 버그

---

## 4. 완료 기준 달성 여부

| 기준 | 결과 |
|------|------|
| NPU 없는 환경: 2 OK + 5 SKIP | ✅ 달성 |
| NPU 있는 환경: 7/7 PASS | ❌ 3 FAIL + 1 SIGILL |
| htp_phase1_result.xml 확인 | ✅ pull 완료 |

---

## 5. 다음 단계

1. **Phase 1.5 버그픽스** (Phase 2 진입 전 선행 필요):
   - `hexkl_mm.cpp`: N % 32 ≠ 0 케이스 처리 (fallback 또는 패딩)
   - `EdgeCase_SingleRow` SIGILL 원인 규명 (디버그 빌드 + `addr2line`)
2. **NPU 배포 절차 공식화**: `ADSP_LIBRARY_PATH` 설정 + skel push를 `tools/run_device_test.sh`에 추가
3. Phase 2: WH weight 캐시 (`hexkl_mm.cpp:171`)
