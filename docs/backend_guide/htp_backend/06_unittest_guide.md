# 유닛 테스트 구성 및 단말 실행 가이드

교차 컴파일된 유닛 테스트 바이너리를 실제 대상 단말(Galaxy S25 Ultra 등)에 이식하여 원격 검증하는 절차를 안내합니다.

## 1. 테스트 소스 파일 구성

HTP 백엔드 검증을 위해 두 종류의 정밀 레벨 테스트 스위트가 동작합니다.

| 바이너리 | 소스 파일 | 역할 |
| :--- | :--- | :--- |
| `unittest_nntrainer_htp_kernels` | `test/unittest/unittest_nntrainer_htp_kernels.cpp` | sdkl C API 직접 호출 — 커널 수치 정확도 및 처리 레이턴시 측정 |
| `unittest_nntrainer_htp_backend` | `test/unittest/unittest_nntrainer_htp_backend.cpp` | nntrainer ComputeOps API 레벨 통합 테스트 — Fallback 제어 흐름, QINT8 디스패치 검증 |

NPU 의존 테스트는 `HtpBackend::global().enabled()` == false 일 때 `GTEST_SKIP()`으로 자동 건너뜁니다.

## 2. ADB를 통한 필수 바이너리 배포법

대상 단말: **Galaxy S25 Ultra** (SM-S938N, ADB `R3CY205ZMND`)

단말의 `/data/local/tmp` 디렉토리에 교차 컴파일 결과 바이너리와 연동 SDK 필수 동적 라이브러리를 직접 배포합니다.

```bash
# 1. 컴파일 완료된 유닛테스트 바이너리 전송
adb -s R3CY205ZMND push build_android/test/unittest/unittest_nntrainer_htp_kernels /data/local/tmp/
adb -s R3CY205ZMND push build_android/test/unittest/unittest_nntrainer_htp_backend /data/local/tmp/

# 2. HexKL 프리빌트 동적 공유 라이브러리 전송
#    주의: armv9_android26/libsdkl.so는 SM8750(S25 Ultra)에서 SIGILL 발생 — armv8 버전 사용
adb -s R3CY205ZMND push $HEXKL_SDK_ROOT/lib/armv8_android26/libsdkl.so /data/local/tmp/

# [주의] HTP 구동을 위한 CDSP 기기 스켈레톤(libhexkl_skel.so, V79용)도 반드시 /data/local/tmp에 존재해야 합니다.
```

### armv8 vs armv9 libsdkl.so

`armv9_android26/libsdkl.so`는 SM8750(Snapdragon 8 Elite) 단말에서 프로세스 시작 시 **SIGILL**이 발생합니다 (init 경로에서 미활성화된 SME 명령어 사용). `armv8_android26/libsdkl.so`로 대체하면 V79 DSP 세션이 정상 초기화됩니다.

## 3. 단말 원격 실행 명령어 스크립트

물리 공유 라이브러리 참조 경로(`LD_LIBRARY_PATH`) 및 CDSP 스켈레톤 링커 경로(`ADSP_LIBRARY_PATH`)를 아래와 같이 주입합니다.

```bash
# 커널 레벨 테스트 실행
adb -s R3CY205ZMND shell \
  "cd /data/local/tmp && \
   LD_LIBRARY_PATH=/data/local/tmp ADSP_LIBRARY_PATH=/data/local/tmp \
   ./unittest_nntrainer_htp_kernels --gtest_color=no"

# nntrainer API 레벨 통합 테스트 실행
adb -s R3CY205ZMND shell \
  "cd /data/local/tmp && \
   LD_LIBRARY_PATH=/data/local/tmp ADSP_LIBRARY_PATH=/data/local/tmp \
   ./unittest_nntrainer_htp_backend --gtest_color=no"
```

특정 테스트만 실행하려면 `--gtest_filter=<Suite>.<Test>` 추가:

```bash
adb -s R3CY205ZMND shell \
  "cd /data/local/tmp && \
   LD_LIBRARY_PATH=/data/local/tmp ADSP_LIBRARY_PATH=/data/local/tmp \
   ./unittest_nntrainer_htp_kernels \
   --gtest_filter=HtpKernelTest.Accuracy_f32f16_f32_Prefill --gtest_color=no"
```

## 4. sdkl_rm_to_wh_i8_probe (진단 도구)

`test/unittest/jni_htp/sdkl_rm_to_wh_i8_probe.cpp`는 `sdkl_cpu_rm_to_wh_i8_inplace()` 버퍼 종류별 동작을 격리 검증하는 독립 바이너리입니다. QINT8 가중치 레이아웃 변환 이슈 디버깅에 사용합니다.

```bash
# 빌드 후 단말에 push하고 실행 (presets: small / qwen_attn / qwen_ffn_up / qwen_ffn_down)
adb -s R3CY205ZMND shell \
  "cd /data/local/tmp && \
   LD_LIBRARY_PATH=/data/local/tmp ADSP_LIBRARY_PATH=/data/local/tmp \
   ./sdkl_rm_to_wh_i8_probe --preset qwen_attn --buf all"
```
