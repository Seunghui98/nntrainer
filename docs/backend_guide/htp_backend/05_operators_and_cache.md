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

### 3-1. Prefill (M > 1) — Transient Alloc 경로

- `sdkl_npu_alloc(W_npu)` → RM→WH 변환 → 연산 → `sdkl_npu_free(W_npu)` 1회성 처리
- WHCache를 **완전히 스킵**: LLM prefill에서 동일 가중치가 1번만 참조되므로 캐싱 이득이 없음
- 최대 NPU DMA 점유: 호출당 `N×K×2 bytes` (FP16)로 제한됨

### 3-2. Decode (M = 1) — WHCache 경로

디코드 단계에서는 동일 가중치가 토큰 생성 루프 전체에 걸쳐 반복 참조됩니다.

- **용량 상한**: `WH_CACHE_MAX_BYTES = 64 MB`로 고정 — NPU DMA 자원 고갈 방지
- **키**: 가중치 포인터 `B`(기존 CPU 버퍼 주소)
- **캐시 미스**: NPU 버퍼 할당 + RM→WH 변환 + 삽입
- **캐시 히트**: 이미 WH로 변환된 `W_npu` 재사용 — 할당·변환 비용 0
- **FIFO 만료(Eviction)**: 삽입 순서(`std::vector<const void*> order`)를 추적하여 용량 초과 시 가장 오래된 엔트리를 `sdkl_npu_free`로 제거
- **종료 처리**: `WHCache` 소멸자에서 잔여 엔트리 전량 해제

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
| M == 1 (decode) | CPU `hsgemv` | 메모리 대역 제한 구간; CPU NEON FP16이 경쟁력 있음 |
| M > 1 && N%32==0 && `supports_shgemm()` | NPU `shgemm_f32f16_f32` (transient) | HMX 타일 병렬성 유효 |
| M > 1 && (N%32≠0 또는 NPU 미가동) | CPU `shgemm` fallback | N 정렬 위반 시 SDK 예외 방지 |
