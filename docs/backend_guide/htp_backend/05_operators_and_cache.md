# 연동 커널 및 WHCache 메커니즘

HTP 가속 연산의 코어를 담당하는 저수준 Matrix Multiplication 커널 및 버퍼 관리 전략을 기술합니다.

## 1. ComputeOps 바인딩 및 커널

`HtpComputeOps`는 `CpuComputeOps`를 상속하며 NPU 전용 커널 두 종을 재정의합니다.

| 메서드 | SDKL API | 데이터 타입 | 조건 |
| :--- | :--- | :--- | :--- |
| `shgemm` | `sdkl_npu_mm_f32f16_f32` | FP32 입력 × FP16 가중치 → FP32 출력 | `ENABLE_FP16` |
| `shgemm_u8i8` | `sdkl_npu_mm_u8i8_i32` | U8 입력 × I8 가중치(WH) → I32 → FP32 출력 | 항상 |

두 커널 모두 `supports_shgemm()` / `supports_shgemm_u8i8()`가 `HtpBackend::global().enabled()`를 반환하므로 NPU 미초기화 시 자동으로 CPU Fallback됩니다.

## 2. RM → WH 레이아웃 트랜스폼

Qualcomm HMX 연산 타일은 가중치 행렬이 WH(Width-Height) 레이아웃이어야 병렬 실행됩니다. CPU에서 적재된 가중치는 Row-Major 형식이므로 NPU에 전달하기 전 `sdkl_cpu_rm_to_wh_f16_inplace`(FP16) 또는 사전 변환된 WH 버퍼를 사용합니다.

- **FP16 GEMM**: `shgemm_f32f16_f32` 내부에서 RM→WH 변환 수행 (인플레이스)
- **QINT8 GEMM**: `shgemm_u8i8_i32`는 이미 WH로 변환된 `B_wh` 포인터를 받으므로 추가 변환 없음 (오프라인 양자화 단계에서 사전 변환)

## 3. FP16 GEMM 버퍼 관리: 경로별 전략

`shgemm_f32f16_f32` 내부에서 M 크기에 따라 두 경로로 분기합니다.

### 3-1. Prefill (M > 1) — PrefillWHCache (pin-once) + Transient Fallback

`g_prefill_wh` (`PrefillWHCache`)를 통해 WH 레이아웃 변환 비용을 최초 1회만 지불합니다.

| 상태 | 처리 | W_npu 수명 |
| :--- | :--- | :--- |
| 캐시 **히트** (동일 포인터 + N,K) | 즉시 pinned WH 포인터 반환 | 프로세스 수명 내내 유지 |
| 캐시 **미스** + `total_bytes + new_bytes ≤ 48 MB` | alloc + memcpy + rm_to_wh + pin | 프로세스 수명 내내 유지 |
| 캐시 **미스** + 캡 초과 또는 alloc 실패 | pin 안 함, `nullptr` 반환 → transient 경로 | 1회 호출 후 즉시 free |

**상수:** `PREFILL_WH_PIN_MAX_BYTES = 48 MB` (hexkl_mm.cpp:41)

Qwen3-0.6B 기준 48 MB 내에 pin 가능한 가중치는 레이어 1-3의 FC 3종×3레이어 = **9개(48 MB)**. 나머지 레이어(4-28)는 `nullptr`을 받아 transient 경로를 사용합니다.

### 3-2. Decode (M = 1) — CPU NEON 고정 (g_wh_cache 미사용)

**Decode(M=1)는 NPU shgemm 경로에 진입하지 않습니다.** `float_tensor.cpp::dotFloat32Float16`에서 M==1 조건을 먼저 검사하여 CPU NEON `hsgemv`로 분기하므로 `shgemm_f32f16_f32`가 호출되지 않습니다.

- `g_wh_cache` (FIFO WHCache, 64 MB 캡, `hexkl_mm.cpp`)는 소스에 존재하지만 **실제로 채워지지 않습니다.**
- decode → CPU 고정 라우팅은 메모리 대역폭 제한 구간에서 CPU NEON이 경쟁력 있고, NPU FastRPC 왕복 오버헤드가 불리하기 때문입니다.

## 4. QINT8 GEMM 파이프라인 (`shgemm_u8i8_i32`)

QINT8 경로는 WHCache를 사용하지 않습니다. 모든 버퍼(X_npu, C_npu, W_npu)는 호출마다 할당 후 즉시 해제됩니다.

```
입력 A (FP32, M×K)
  ↓  per-tensor 동적 양자화
X_u8 (U8, Mp×K)  [act_scale = max_abs(A) / 127, zp = 128]
  ↓  sdkl_npu_mm_u8i8_i32
C_i32 (I32, Mp×N)
  ↓  역양자화
C (FP32, M×N)  [C[m,n] = act_scale × wt_scale[n] × (C_i32[m,n] − zp_corr[n])]
```

- `B_wh`: 오프라인 양자화 단계에서 이미 WH 레이아웃으로 저장된 I8 가중치
- `wt_scale` (FP32, 크기 N): 출력 채널별 per-channel 가중치 스케일
- `zp_corr` (I32, 크기 N): zero-point 보정 벡터 (미리 계산됨)
- M은 내부에서 64 단위로 패딩(Mp = ⌈M/64⌉×64); 출력은 실제 M 행만 반환

## 5. 라우팅 결정 (float_tensor.cpp)

| 조건 | 실행 경로 | 이유 |
| :--- | :--- | :--- |
| M == 1 (decode) | CPU `hsgemv` | 메모리 대역 제한 구간; CPU NEON FP16이 경쟁력 있음. shgemm 진입 전에 분기 |
| M > 1 && N%32==0 && `supports_shgemm()` && PrefillWHCache 히트 | NPU `shgemm_f32f16_f32` (pinned WH) | pin-once 경로 — rm_to_wh 비용 없음 |
| M > 1 && N%32==0 && `supports_shgemm()` && PrefillWHCache 미스 | NPU `shgemm_f32f16_f32` (transient W_npu) | 캡 초과 또는 최초 호출 — pin 공간 부족 시 WH 매 호출 재계산 |
| M > 1 && (N%32≠0 또는 NPU 미가동) | CPU `shgemm` fallback | N 정렬 위반 시 SDK 예외 방지 |

## 6. 오프라인 WH Bake 파이프라인 (Integrated .bin)

3-1의 pin-once 캐시는 프로세스 최초 호출 시점에 48 MB 캡 내에서만 rm_to_wh 비용을 상각합니다. 배포 전에 WH 변환을 오프라인으로 1회 수행해 두면 그 캡을 넘어서는 가중치까지도 첫 prefill 호출부터 rm_to_wh 비용 없이 NPU에 바로 투입할 수 있습니다.

파이프라인 흐름:

0. **전제조건 — 소스는 진짜 FP32여야 함.** `nntr_quantize`는 소스 `nntr_config.json`에 선언된 dtype으로 `.bin`을 로드하므로, 입력 config는 `"model_tensor_type": "FP32-FP32"` / `"fc_layer_dtype": "FP32"`로 FP32 bin을 정직하게 기술해야 합니다. `FP16-FP32`처럼 잘못 선언하면 loader가 4바이트 가중치를 2바이트로 읽어 가중치가 깨진 채 재저장되고, WH 수집도 안 돼 **빈 트레일러(`Registered 0/0`) + garbage 출력**이 됩니다. `nntr_quantize`는 소스가 `FP32-FP32`가 아니면 **에러로 중단**합니다(`quantize.cpp`; 이전에는 경고만 하고 진행해 깨진 모델을 생성했음). 증상/진단은 [02 §8-5](02_build_and_env.md) 참조.
1. **Bake (오프라인, `nntr_quantize --fc_dtype FP16_WH`)** — `--fc_dtype FP16_WH`는 `DataType`이 아니라 도구 자체의 지시자입니다. 인자 파싱 단계에서 `nntrainer::setWHBakeRequested(true)`를 호출하고 이후 흐름은 일반 FP16 저장(`fc_dtype_str = "FP16"`)으로 진행됩니다. 저장 과정에서 `g_wh_collector`가 FC 레이어별 RM→WH 변환 결과를 `WHTrailerEntry{name, N, K, wh_bytes}` 형태로 수집합니다.
   - **Weight-layout 버그**: FC 계열 레이어(`fc_layer.cpp`, `shared_fully_connected_layer.cpp`)는 가중치를 **`[K,N]` row-major**로 저장하고 `dot(..., trans=false, trans_in=false)`(즉 `TransB=false`)로 호출한다. 과거 NPU shgemm 경로와 오프라인 WH-bake는 반대인 `[N,K]`를 가정해 모든 FC/QKV/FFN 가중치의 바이트 레이아웃을 잘못 라벨링했고, 그 결과 **NPU 출력이 전부 garbage**였다(CPU는 `ldb`로 레이아웃을 존중해 항상 정상). pre-baked 경로와 transient 경로가 같은 잘못된 가정에서 바이트를 유도했기에 둘 다 동일한 garbage를 냈다.
   - **수정(`8c1c1fa5`)**: `TransB==false`일 때 WH-bake 전에 실제 transpose를 수행한다. `hexkl_mm.cpp`에 `copyForWHBake()` 헬퍼를 추가해 transient·pin-cache RM→WH 경로에 연결하고, `layer_devel.h`는 오프라인 bake 전에 `Tensor::transpose("0:2:1")`(Q4_0/QS4CX와 동일 패턴)를 적용한다.
2. **Trailer 기록 (writer, save 경로)** — 기존 RM FP16 바이너리 바이트열은 그대로 유지되고, 파일 끝에 `WH_TRAILER_MAGIC = "WHF1"`으로 식별되는 트레일러가 추가로 append됩니다 (`writeWHTrailer`). 트레일러를 모르는 구버전 로더는 동일 RM 바이너리만 읽으므로 파일 포맷은 하위 호환입니다.
3. **Load-time registration (loader, `neuralnet.cpp`)** — 모델 로드 시 `readWHTrailer`가 트레일러 존재 여부를 먼저 확인하고, 존재하면 각 엔트리를 이름 키로 `registerPrefillWH(ptr, N, K, wh_bytes)`에 등록합니다. 이 등록은 `ExecutionMode::TRAIN`뿐 아니라 실제 CausalLM 앱이 항상 사용하는 `ExecutionMode::INFERENCE` 로드 경로에서도 수행됩니다.
4. **Prefill fast path (consumer, `shgemm_f32f16_f32`)** — 호출 시 `lookupPrefillWH(ptr, N, K)`로 사전 등록된 WH 포인터를 먼저 조회합니다. 히트하면 rm_to_wh 변환 없이 즉시 NPU에 투입되고, 미스(트레일러 없음/등록 실패)면 3-1의 pin-once/transient 경로로 안전하게 폴백합니다 — 어느 경우에도 throw하지 않습니다.
5. **Decode는 변경 없음** — decode(M=1)는 여전히 CPU NEON `hsgemv` 고정 경로이며 WH 트레일러/레지스트리와 무관합니다 (2절, 3-2절 참조).

실측(2026-07-06, Qwen3-0.6B, S25 Ultra): 트레일러가 포함된 integrated bin 로드 시 `[HTP] Registered 196/196 pre-baked WH weights` 로그로 전 FC 가중치 등록을 확인했으며, 첫 prefill 호출부터 대부분의 레이어가 즉시 fast path를 탄 결과 prefill 시간이 3178 ms → 96 ms로 감소했습니다 (자세한 A/B 수치는 `08_e2e_performance_results.md` 6절 참조).

## 7. 진단 env 토글

| 토글 | 효과 |
| :--- | :--- |
| `NNTR_HTP_DISABLE_PREBAKED_WH=1` | `lookupPrefillWH`를 강제 miss시켜 known-good transient RM→WH 경로 사용 (escape hatch / bisection) |
| `NNTR_HTP_VERIFY_PREBAKED_WH=1` | pre-baked hit마다 transient 변환을 재계산해 등록 바이트와 memcmp; 가중치별 OK/MISMATCH를 stderr에 로그 |
| `NNTR_HTP_VERIFY_PREFILL_MM=1` | 실입력에서 live NPU 출력을 CPU 레퍼런스와 diff (상시 진단) |
