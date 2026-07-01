# HTP 메모리 할당 및 물리 버퍼 매핑

Hexagon Tensor Processor는 가상 주소가 아닌 물리 버퍼(ION/RPC 메모리)를 통해 하드웨어 장치와 직접 DMA 교환합니다. 이 문서는 현재 활성 할당 전략과 NPU DMA 풀 예산을 기술합니다.

---

## 1. HtpMemAllocator — 현재 비활성화

`HtpMemAllocator` 클래스(`htp_mem_allocator.h` / `htp_mem_allocator.cpp`)는 `sdkl_npu_alloc`/`sdkl_npu_free`를 `MemAllocator` 인터페이스로 감싼 구현체입니다. 파일은 소스트리에 존재하지만 **현재 초기화 경로에서 등록되지 않습니다.**

`htp_context.cpp::initialize()` 에서의 주석이 이유를 명시합니다:

> _"Use the default CPU allocator for model tensors (weights, activations). shgemm_f32f16_f32 stages data into NPU-accessible buffers per-call via sdkl_npu_alloc; **allocating all model tensors with HtpMemAllocator would exhaust the NPU DMA pool** on a full transformer model."_

모든 모델 텐서(가중치, 활성화값)를 NPU DMA 풀에서 할당하면 트랜스포머 전체 모델 규모에서 풀이 고갈됩니다. 따라서 기본 CPU allocator(일반 가상 메모리)를 사용하고, NPU DMA 버퍼는 커널 호출 단위로만 할당·해제합니다.

---

## 2. NPU DMA 풀 예산 (실측, 2026-07-02)

| 측정 방식 | 결과 | 비고 |
| :--- | :--- | :--- |
| Transient probe (`PoolProbe_MeasureMaxResidentBytes`): 4 MB 청크, 전부 해제 후 측정 | **4000 MB** | pin-once 산정에 사용 불가 |
| Sustained-pin probe (`PoolProbe_MeasureMaxSustainedPinBytes`): 4/6/6 MB cycling, 해제 없이 84개 연속 할당 | **448 MB** | 루프 상한(84) 도달, 할당기 미실패 |
| 커널 실행 중 실효 pin 예산 | **~48 MB** | `sdkl_npu_mm_f32f16_f32` 첫 호출 시 내부 스크래치 ~400 MB 영구 점유 |

`sdkl_npu_mm_f32f16_f32`는 첫 호출 시 내부 스크래치 버퍼로 ~400 MB를 영구 할당(세션 수명 동안 해제 안 됨)합니다. 따라서 애플리케이션에서 pin-once로 사용할 수 있는 실효 예산은 B_raw(448 MB)가 아닌 **B_effective ≈ 48 MB**입니다. cap=512 MB로 올려 warmup-ON 테스트 시 커널 스크래치 할당 실패 → SDKL 내부 상태 오염 → SIGSEGV로 확인됨 (task-4b-report.md 참조).

---

## 3. 활성 할당 전략: per-call staging buffer

`shgemm_f32f16_f32` 내부에서 NPU GEMM 호출마다 staging buffer를 직접 할당·해제합니다.

| 버퍼 | 크기 | 수명 |
| :--- | :--- | :--- |
| `X_npu` (FP32 입력, A) | `Mp × K × 4 bytes` (`Mp` = M을 32의 배수로 올림) | 1회 커널 호출 후 즉시 free |
| `A_npu` (FP32 출력, C) | `Mp × N × 4 bytes` | 1회 커널 호출 후 즉시 free |
| `W_npu` (FP16 가중치 WH, prefill) | `N × K × 2 bytes` | PrefillWHCache 히트 시 pinned; 캐시 미스 + 캡 초과 시 transient |

`X_npu`와 `A_npu`는 항상 per-call transient입니다. `W_npu`만 PrefillWHCache 전략이 적용됩니다.

---

## 4. PrefillWHCache — pin-once 잔류 메커니즘

`g_prefill_wh` (`PrefillWHCache` 타입, `hexkl_mm.cpp` 내부) 는 prefill(M>1) 경로에서 WH 레이아웃 변환 비용을 최초 1회만 지불하도록 설계된 영구 캐시입니다.

```cpp
struct PrefillWHCache {
  std::map<const void*, WHEntry> cache;  // key = CPU-side weight pointer
  size_t total_bytes = 0;
  std::mutex mtx;
  ~PrefillWHCache() { /* sdkl_npu_free all entries */ }
} g_prefill_wh;
```

`getOrCreatePrefillWH(B, N, K)` 동작:

| 상태 | 처리 | 반환 |
| :--- | :--- | :--- |
| 캐시 히트 (B 포인터 + N,K 일치) | 즉시 반환 | 기존 pinned WH 포인터 |
| 캐시 미스 + `total_bytes + new_bytes ≤ PREFILL_WH_PIN_MAX_BYTES` | `sdkl_npu_alloc` + memcpy + `sdkl_cpu_rm_to_wh_f16_inplace` + pin | 새 WH 포인터 |
| 캐시 미스 + 캡 초과 또는 alloc 실패 | 할당 안 함, 에러 무시 | `nullptr` → 호출자가 transient 경로 사용 |

**상수:** `PREFILL_WH_PIN_MAX_BYTES = 48 MB` (`hexkl_mm.cpp:41`)

Qwen3-0.6B 기준 48 MB 내에 핀 가능한 가중치: q_proj(4 MB)×3 + gate_up_proj(6 MB)×3 + down_proj(6 MB)×3 = **48 MB (레이어 1-3, 총 9개)**. 레이어 4-28의 가중치는 transient 경로로 처리됩니다.

캐시는 **프로세스 수명 동안 유지**됩니다 (eviction 없음). 프로세스 종료 시 소멸자에서 전체 pin 해제.
