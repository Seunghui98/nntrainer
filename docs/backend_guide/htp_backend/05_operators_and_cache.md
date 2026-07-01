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
