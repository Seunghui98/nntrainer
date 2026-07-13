# HexKL 빌드 및 환경 구성 — 호스트 빌드부터 단말 qwen3-0.6b 실행까지

HTP 백엔드를 활성화해 안드로이드 기기용 바이너리를 빌드하고, 단말에서 실제 qwen3-0.6b(WH-baked FP16) 추론을 실행하기까지의 end-to-end 흐름을 기술합니다. 문제 발생 시 [8. Troubleshooting](#8-troubleshooting)을 참고하세요.

## 1. 필수 환경변수 설정

빌드 실행 전에 아래 환경변수를 호스트 머신에 설정합니다.

```bash
# Hexagon SDK 내 HexKL Addon 라이브러리 경로 (libsdkl.so 및 헤더 포함)
export HEXKL_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1/addons/hexkl_addon

# Android NDK 경로 설정
export ANDROID_NDK=/opt/android-ndk-r26d
export PATH=$ANDROID_NDK:$PATH
```

> 이 환경변수는 §2 라이브러리 빌드뿐 아니라 §4 `install_android.sh`에서도 사용됩니다 — install 스크립트가 meson-info에서 `hexkl-sdk-root`/`hexkl-lib-subdir`를 읽어 push할 `libsdkl.so`를 찾습니다.

## 2. nntrainer HTP 라이브러리 빌드

Galaxy S25 Ultra (SM-S938N, Snapdragon 8 Elite) 타겟 빌드는 다음 쉘 스크립트로 활성화합니다.

```bash
./tools/package_android.sh \
  --arm-arch=armv8.2-a \
  -Denable-htp=true \
  -Dhexkl-sdk-root=$HEXKL_SDK_ROOT \
  -Dhexkl-lib-subdir=armv8_android26 \
  -Dmmap-read=false \
  -Dwerror=false
```

산출물: `builddir/android_build_result/lib/arm64-v8a/libnntrainer.so` (HTP 경로 활성). 이 `libnntrainer.so`가 다음 단계에서 CausalLM 앱에 그대로 링크됩니다.

### 주요 Meson 빌드 플래그 요약
- `enable-htp=true`: HexKL/HTP NPU 연동 코드 활성화 및 컴파일 플래그(`ENABLE_HEXKL`) 주입.
- `hexkl-lib-subdir=armv8_android26`: 프리빌트 `libsdkl.so` 배포 파일의 아키텍처 디렉토리 지정. meson 기본값과 일치하며, SIGILL을 피합니다(§8-1 참고).
- `arm-arch=armv8.2-a`: 단말 CPU는 armv9.2를 지원하지만, 현재 소스에는 `ARM_ARCH_ARMV9_2A`/`ENABLE_SVE2`에 걸린 코드 경로가 없고 armv9.2-a march 플래그 자체도 `+nosve+nosve2`로 SVE/SVE2 codegen을 비활성화하므로, armv8.2-a로 빌드해도 오늘 기준 기능적으로 동일합니다.
- `enable-fp16=true`: HTP 가속에 필수인 반정밀도(FP16) 타입 활성화(`package_android.sh`가 기본 주입).

## 3. CausalLM 앱 빌드 (`nntrainer_causallm`)

단말에서 LLM을 실행하려면 §2에서 만든 HTP `libnntrainer.so`를 링크하는 CausalLM 앱을 빌드해야 합니다.

```bash
cd Applications/CausalLM
./build_android.sh --cache
```

- **`--cache`는 HTP 실행에 필수입니다.** `Applications/CausalLM/jni/Android.mk`는 nntrainer를 소스에서 재컴파일하지 않고 `builddir/android_build_result/lib/arm64-v8a/libnntrainer.so`를 **prebuilt(`PREBUILT_SHARED_LIBRARY`)로 링크**합니다. `--cache` 없이 실행하면 `build_android.sh`가 `builddir`를 지우고 `./tools/package_android.sh`를 **HTP 플래그 없이** 다시 호출해 HTP가 빠진 `libnntrainer.so`를 만듭니다. 따라서 반드시 **(§2) HTP 빌드 → (§3) `--cache`로 재사용** 순서를 지키세요.
- 산출물(`Applications/CausalLM/jni/libs/arm64-v8a/`): `nntrainer_causallm`(메인 실행파일), `nntr_quantize`(WH-bake 도구), `nntr_safetensors_info`, `libcausallm_core.so`.
- **경로 하드코딩 주의:** `nntr_quantize` 타깃은 `jni/Android.mk`에서 `HEXKL_ADDON_ROOT`(기본 `/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1/addons/hexkl_addon`)와 `armv8_android26/libsdkl.so` 경로를 하드코딩하고 `-DENABLE_HEXKL=1`로 컴파일합니다. 다른 호스트에서는 `jni/Android.mk`의 이 값을 수정해야 합니다.

## 4. 단말 배포 (`install_android.sh`)

빌드 산출물과 HTP 런타임 라이브러리를 단말로 push하고 on-device 실행 스크립트를 생성합니다.

```bash
./install_android.sh
```

- 설치 경로: `/data/local/tmp/nntrainer/causallm/`. push 대상: `nntrainer_causallm`, `nntr_quantize`, `libcausallm_core.so`, `libnntrainer.so`, `libccapi-nntrainer.so`, `libc++_shared.so`, `libsdkl.so`, `libhexkl_skel.so`(V79 스켈레톤).
- 생성되는 on-device 스크립트: `run_causallm.sh`, `run_quantize.sh`, `run_safetensors_info.sh` (모두 `LD_LIBRARY_PATH`/`ADSP_LIBRARY_PATH`를 설치 경로로 설정하고, 인자를 `"$@"`로 전달 — §8-4 참고).
- `install_android.sh`는 meson 옵션 `hexkl-lib-subdir` 값을 그대로 읽어 push하며, §2에서 이미 `armv8_android26`으로 설정했으므로 별도 조치 없이 올바른 armv8 라이브러리가 자동으로 push됩니다(armv9로 빌드했을 때의 SIGILL 이슈는 §8-1 참고).

## 5. qwen3-0.6b 모델 준비 (WH-baked FP16)

WH-baked 경로는 FP32 nntrainer 가중치를 받아 **단말에서** `--fc_dtype FP16_WH`로 bake하여, RM→WH 레이아웃 변환을 로드 시점에 미리 끝낸 `WHF1` 트레일러 통합 bin을 만듭니다. 그러면 최초 prefill부터 rm_to_wh 재계산 없이 pinned WH 포인터를 즉시 씁니다(bake 파이프라인 상세: [05 §6](05_operators_and_cache.md), 실측 A/B: [08 §6](08_e2e_performance_results.md)).

### (a) FP32 nntrainer `.bin` 확보

`Applications/CausalLM/res/qwen3/qwen3-0.6b/`에 이미 있는 `nntr_qwen3_0.6b_fp32.bin` + `nntr_config.json` + `tokenizer.json`을 사용합니다. GGUF에서 새로 변환하려면 같은 디렉토리의 스크립트를 씁니다:

> **⚠️ bake 입력 config는 FP32 bin을 정직하게 기술해야 합니다.** `nntr_quantize`는 소스 `nntr_config.json`에 선언된 dtype으로 `.bin`을 로드하므로, FP32 파일을 `"model_tensor_type": "FP16-FP32"` / `"fc_layer_dtype": "FP16"`처럼 잘못 선언하면 loader가 4바이트 가중치를 2바이트로 읽어 **가중치가 통째로 깨진 채 로드→재저장**됩니다(구조는 멀쩡하지만 값이 garbage, WH 트레일러 0개 → `[HTP] Registered 0/0`). 소스 config는 반드시 `"model_tensor_type": "FP32-FP32"`, `"fc_layer_dtype": "FP32"`여야 합니다. (`nntr_quantize`는 소스가 `FP32-FP32`가 아니면 에러로 중단합니다 — `quantize.cpp`.)

```bash
python3 Applications/CausalLM/res/qwen3/qwen3-0.6b/gguf_to_nntrainer.py \
  /path/to/qwen3-0.6b.gguf \
  -o nntr_qwen3_0.6b_fp32.bin --target arm --emit-nntr-config
```

### (b) 모델 dir push 후 on-device WH-bake

WH 변환에는 sdkl 런타임이 필요하므로 bake는 **단말에서** 실행합니다(`nntr_quantize`는 armv8 `libsdkl.so`에 링크되어 있고, 그 라이브러리는 §4에서 이미 push됨).

```bash
# 모델 파일을 단말로 push
adb push Applications/CausalLM/res/qwen3/qwen3-0.6b \
  /data/local/tmp/nntrainer/causallm/models/

# 단말에서 WH-bake (FP16_WH는 DataType이 아니라 도구 지시자 — setWHBakeRequested(true))
adb shell \
  "/data/local/tmp/nntrainer/causallm/run_quantize.sh \
   models/qwen3-0.6b --fc_dtype FP16_WH \
   --output_bin nntr_qwen3_0.6b_fp16_wh.bin"
```

bake가 끝나면 모델 dir에 `WHF1` 트레일러가 붙은 `nntr_qwen3_0.6b_fp16_wh.bin`과 `nntr_config_quantized.json`이 생성됩니다.

- **rename 불필요 (자동 로드):** `nntr_config_quantized.json`은 dtype·`model_file_name`·`compute_engine`(=`htp`, WH bake는 HTP 전용이라 자동 설정)이 실행에 맞게 채워져 나옵니다. 실행 시 앱(`main.cpp`)이 이 파일이 있으면 **자동 우선 로드**하고 없으면 `nntr_config.json`으로 폴백하므로, `nntr_config.json`으로 옮겨 쓸 필요가 없습니다(`Using config: …` 로그로 확인). 소스 `nntr_config.json`은 보존되어 재-bake에도 안전합니다.
- **확인만:** `tokenizer_file`이 단말 실제 경로와 일치해야 합니다 — res 디렉토리 기본값이 다른 경로를 가리키면 push한 위치로 수정하세요. (구버전 `nntr_quantize`로 만든 config는 `compute_engine`이 소스 값 그대로 `cpu`일 수 있으니, 그때만 `htp`로 고치세요.)

## 6. 실행 및 성공 확인

```bash
adb shell \
  "/data/local/tmp/nntrainer/causallm/run_causallm.sh \
   /data/local/tmp/nntrainer/causallm/models/qwen3-0.6b \
   'The capital of France is'"
```

> **프롬프트에 공백이 있으면 반드시 작은따옴표로 감싸세요.** `install_android.sh`가 만든 실행 스크립트가 인자를 `"$@"`로 전달하지 않는 구버전이라면 공백에서 잘려 첫 단어만 모델에 들어갑니다(증상: `prefill: 1 tokens`, 프롬프트 에코가 첫 단어만). 최신 스크립트는 `"$@"`로 고쳐졌으니 `install_android.sh`를 다시 실행해 스크립트를 재생성하거나, 임시로 바이너리를 직접 실행하세요(§8-4).

성공 신호:

- **출력이 프롬프트에 대해 의미 있는 문장**이어야 합니다 — 가장 먼저 볼 correctness 신호입니다. 예: `The capital of France is` → ` Paris. The capital of Italy is Rome. ...`. 무의미 토큰이 반복되면 §8-5(깨진 bake)를 의심하세요.
- stdout에 **`Using config: .../nntr_config_quantized.json`** 출력 → 앱이 bake 산출 config를 자동 로드함(`main.cpp`). `.../nntr_config.json`이 찍히면 quantized config가 없다는 뜻이니 bake 산출물 위치를 확인하세요.
- 로드 로그에 **`[HTP] Registered 196/196 pre-baked WH weights`** 출력 → 전체 FC 가중치가 WH로 등록됨. **이 로그는 stdout이 아니라 Android logcat(`ml_logi`)에 찍히므로** `adb logcat`으로 확인하세요. (RM-only bin이면 이 로그가 없고 transient/pin-once 경로로 폴백합니다.)
- **prefill 시간이 수십 ms 수준**(WH-baked 기준). 참고 실측: RM-only 3178 ms → WH-baked 96 ms (약 33x, [08 §6](08_e2e_performance_results.md)).
- decode(M=1)는 설계상 CPU NEON `hsgemv` 고정이라 WH-bake와 무관하게 동일합니다.

실패 신호 (원인 → §8):

| 증상 | 진단 |
|------|------|
| 출력이 정상 문장인데 `prefill: 1 tokens`, 프롬프트 에코가 첫 단어만 | 실행 스크립트 `$@` 인용 누락으로 프롬프트가 잘림 → §8-4 |
| 출력이 garbage(반복/무의미 토큰) + logcat에 `[HTP] Registered 0/0 pre-baked WH weights` + prefill이 수십 ms가 아니라 수 초 | 소스 config dtype 불일치로 만들어진 깨진 bake → §8-5 |

## 7. 단일 유닛 테스트 신속 빌드 팁

전체 모듈 빌드 시간을 줄이기 위해 변경된 단일 테스트 바이너리만 고속 타겟 컴파일하는 ninja 명령입니다.

```bash
ninja -C build_android test/unittest/unittest_nntrainer_htp_kernels
ninja -C build_android test/unittest/unittest_nntrainer_htp_backend
```

## 8. Troubleshooting

### 8-1. armv9 libsdkl.so SIGILL (2026-06-25 확인)

`armv9_android26/libsdkl.so`를 SM8750(SM-S938N, Snapdragon 8 Elite) 단말에서 실행하면 프로세스 init 경로(`libsdkl.so+0x14eb0`)에서 **SIGILL**이 발생합니다. 원인: armv9/SME 명령어가 해당 프로세스 컨텍스트에서 활성화되지 않은 상태.

현재 golden path(§2)는 `-Dhexkl-lib-subdir=armv8_android26`(meson 기본값)으로 빌드하여 이 문제를 처음부터 피합니다 — `install_android.sh`가 처음부터 armv8 라이브러리를 push하고, `sdkl_npu_initialize()`가 armv8 userspace + V79 DSP 스켈레톤 조합으로 정상 초기화됩니다. 만약 이 문제를 재현/디버깅할 목적으로 일부러 `hexkl-lib-subdir=armv9_android26`으로 빌드했다면, 설치 후 아래처럼 armv8 `.so`를 수동으로 재push해야 합니다:

```bash
adb push $HEXKL_SDK_ROOT/lib/armv8_android26/libsdkl.so \
  /data/local/tmp/nntrainer/causallm/
```

### 8-2. int8-only 구성 및 헤더 설치 (2026-07-07, 88779af0)

`ENABLE_FP16=0`(즉 `enable-fp16=false`)이면서 `ENABLE_HEXKL=1`인 **int8-only 구성**도 정상 빌드됩니다. `shgemm_u8i8_i32`의 `cleanup()`이 `ENABLE_FP16` 없이도 컴파일되므로, `npuFreeIfAlive` 헬퍼가 `ENABLE_HEXKL` 블록 상단(`ENABLE_FP16`보다 앞)에 정의되어 있고, `<remote.h>`도 함께 hoist되어 `sdkl.h`가 요구하는 `CDSP_DOMAIN_ID`/`CDSP1_DOMAIN_ID`를 제공합니다.

또한 `wh_trailer.h`가 `enable-htp` 헤더 install 블록(`nntrainer/meson.build`)에 `htp_context.h`와 함께 추가되었습니다. `layer_devel.h`(설치되는 public 헤더)가 `ENABLE_HEXKL`에서 `<wh_trailer.h>`를 include하므로, 설치된 헤더로 빌드하는 consumer(예: `Applications/CausalLM` ndk-build)가 "file not found" 오류 없이 빌드됩니다.

### 8-3. HTP 초기화 실패 시 동작

`sdkl_npu_initialize()`가 실패하면 HTP 백엔드는 `enabled()==false`가 되고 모든 `supports_*()`가 `false`를 반환해 **CPU로 무중단 폴백**합니다(설계: [01 소개](01_introduction.md)). 즉 libsdkl/스켈레톤 문제로 NPU가 뜨지 않아도 앱은 CPU 경로로 계속 실행됩니다 — HTP 활성 여부는 §6의 `[HTP] Registered ...` 로그와 prefill 시간으로 판별하세요.

### 8-4. 프롬프트가 첫 단어로 잘림 (`$@` 인용 누락, 2026-07-13 확인)

증상: 프롬프트를 `'The capital of France is'`처럼 넘겨도 모델은 `The`만 받는다. `prefill: 1 tokens`로 찍히고, 프롬프트 에코도 첫 단어만 나온다. M=1이라 NPU prefill(M>1) 경로가 아예 안 타므로 WH fast path도 발동하지 않는다.

원인: `install_android.sh`가 생성하던 on-device 실행 스크립트가 `./nntrainer_causallm $@`처럼 **따옴표 없이** 인자를 전달했다. `adb shell "... 'a b c'"`로 넘긴 한 개의 인자를, 스크립트가 `$@`를 재분리해 공백마다 쪼갠 뒤 바이너리에 전달하고, `main.cpp`는 `argv[2]`(=첫 단어)만 프롬프트로 쓴다.

수정: `install_android.sh`가 이제 `./nntrainer_causallm "$@"`(및 `run_quantize.sh`/`run_safetensors_info.sh`도 동일)로 생성한다. 기존 단말 스크립트는 `install_android.sh` 재실행으로 재생성하거나, 아래처럼 바이너리를 직접 호출한다:

```bash
adb shell \
  "cd /data/local/tmp/nntrainer/causallm && \
   LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. NNTR_NUM_THREADS=4 \
   ./nntrainer_causallm \
     /data/local/tmp/nntrainer/causallm/models/qwen3-0.6b \
     'The capital of France is'"
```

### 8-5. 깨진 출력 + `[HTP] Registered 0/0` (소스 config dtype 불일치, 2026-07-13 확인)

증상: 프롬프트는 제대로 들어가는데(`prefill: 5 tokens`) 출력이 garbage(무의미 토큰 반복)이고, logcat에 `[HTP] Registered 0/0 pre-baked WH weights`가 찍히며 prefill이 수십 ms가 아니라 수 초다.

원인: WH-bake의 **입력** `nntr_config.json`이 FP32 bin을 `"model_tensor_type": "FP16-FP32"` / `"fc_layer_dtype": "FP16"`로 잘못 선언한 채 bake했다. `nntr_quantize`가 FP32 파일을 FP16으로 오해해 로드→재저장하면서 가중치가 통째로 깨지고, WH 엔트리도 0개인 빈 트레일러가 붙는다. 이 깨진 bin은 HTP뿐 아니라 **CPU 엔진(`compute_engine: cpu`)에서도 동일하게 garbage**이므로 HTP 백엔드 문제가 아니다.

진단(bisection):
- 같은 bin을 `compute_engine: cpu`로 돌려도 동일 garbage → 모델/bake 문제(백엔드 아님).
- 원본 FP32 bin을 `"model_tensor_type": "FP32-FP32"` / `"fc_layer_dtype": "FP32"` config로 돌리면 정상 출력 → 소스 데이터는 정상.

수정: 소스 config를 `FP32-FP32`/`FP32`로 바로잡고 재-bake(§5). 최신 `nntr_quantize`는 소스가 `FP32-FP32`가 아니면 경고가 아니라 **에러로 중단**한다(`quantize.cpp`). 재-bake 후에는 `[HTP] Registered 196/196` + prefill ~96 ms + 정상 출력이 확인된다.
