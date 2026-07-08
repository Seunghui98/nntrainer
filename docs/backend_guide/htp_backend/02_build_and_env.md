# HexKL 빌드 및 환경 구성

HTP 백엔드를 활성화하여 안드로이드 기기용 바이너리를 컴파일하는 방법과 필요 환경변수를 기술합니다.

## 1. 필수 환경변수 설정
빌드 실행 전에 아래의 환경변수를 호스트 머신 장치에 설정해야 합니다.

```bash
# Hexagon SDK 내 HexKL Addon 라이브러리 경로 (libsdkl.so 및 헤더 포함)
export HEXKL_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1/addons/hexkl_addon

# Android NDK 경로 설정
export ANDROID_NDK=/opt/android-ndk-r26d
export PATH=$ANDROID_NDK:$PATH
```

## 2. Android 크로스 컴파일 실행 명령어
Galaxy S25 Ultra (SM-S938N, Snapdragon 8 Elite, `armv9.2-a`) 타겟 빌드는 다음 쉘 스크립트를 사용하여 활성화합니다.

```bash
./tools/package_android.sh \
  --arm-arch=armv9.2-a \
  -Denable-htp=true \
  -Dhexkl-sdk-root=$HEXKL_SDK_ROOT \
  -Dhexkl-lib-subdir=armv9_android26 \
  -Dmmap-read=false \
  -Dwerror=false
```

### 주요 Meson 빌드 플래그 요약
- `enable-htp=true`: HexKL/HTP NPU 연동 코드 활성화 및 컴파일 플래그 (`ENABLE_HEXKL`) 주입.
- `hexkl-lib-subdir=armv9_android26`: 프리빌트 `libsdkl.so` 배포 파일의 아키텍처 디렉토리 타겟 지정.
- `enable-fp16=true`: HTP 가속에 필수적인 반정밀도(FP16) 타입 구동 옵션 강제 연동.

### int8-only 구성 및 헤더 설치 (2026-07-07, 88779af0)

`ENABLE_FP16=0`(즉 `enable-fp16=false`)이면서 `ENABLE_HEXKL=1`인 **int8-only 구성**도 정상 빌드됩니다. `shgemm_u8i8_i32`의 `cleanup()`이 `ENABLE_FP16` 없이도 컴파일되므로, `npuFreeIfAlive` 헬퍼가 `ENABLE_HEXKL` 블록 상단(`ENABLE_FP16`보다 앞)에 정의되어 있고, `<remote.h>`도 함께 hoist되어 `sdkl.h`가 요구하는 `CDSP_DOMAIN_ID`/`CDSP1_DOMAIN_ID`를 제공합니다.

또한 `wh_trailer.h`가 `enable-htp` 헤더 install 블록(`nntrainer/meson.build`)에 `htp_context.h`와 함께 추가되었습니다. `layer_devel.h`(설치되는 public 헤더)가 `ENABLE_HEXKL`에서 `<wh_trailer.h>`를 include하므로, 설치된 헤더로 빌드하는 consumer(예: `Applications/CausalLM` ndk-build)가 "file not found" 오류 없이 빌드됩니다.

### armv9 libsdkl.so SIGILL 이슈 (2026-06-25 확인)

`armv9_android26/libsdkl.so`를 SM8750(SM-S938N, Snapdragon 8 Elite) 단말에서 실행하면 프로세스 init 경로(`libsdkl.so+0x14eb0`)에서 **SIGILL**이 발생합니다. 원인: armv9/SME 명령어가 해당 프로세스 컨텍스트에서 활성화되지 않은 상태.

**현재 우회책**: 배포 시 `armv8_android26/libsdkl.so` (497,312 bytes) 사용.
- HTP NPU 세션(`sdkl_npu_initialize`)은 armv8 userspace + V79 DSP 스켈레톤 조합으로 정상 초기화됨
- 빌드 플래그는 `armv9_android26`으로 유지하고, **ADB push 단계에서만 `armv8_android26/libsdkl.so`로 교체**

```bash
# 배포 시 armv8 버전 사용
adb -s R3CY205ZMND push $HEXKL_SDK_ROOT/lib/armv8_android26/libsdkl.so /data/local/tmp/
```

## 3. 단일 유닛 테스트 신속 빌드 팁
전체 모듈 빌드 타임을 단축하기 위해 변경 사항이 발생한 단일 테스트 바이너리만 고속 타겟 컴파일 하는 ninja 명령입니다.

```bash
ninja -C build_android test/unittest/unittest_nntrainer_htp_kernels
ninja -C build_android test/unittest/unittest_nntrainer_htp_backend
```
