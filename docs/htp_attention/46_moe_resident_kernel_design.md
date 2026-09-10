# 46 — Phase A: MoE FFN 상주 커널 설계 (코드 전 레이아웃 문서)

**선행**: `45_whole_model_on_htp_plan.md` (계획, Gate 0/0b 통과), `44_moe_ffn_bottleneck_map.md` §14–16 (측정치).
**이 문서가 존재하는 이유**: 45 §3.2 — L2 1차 시도는 VTCM 레이아웃을 M 전체로 잡아 `AEE_ENOMEMORY`(0x80000402)로 expert 6에서 죽었다(43 §7). 레이아웃은 **코드 전에 산술로 닫는다.**

---

## 1. 무엇을 대체하는가

| | 지금 (레이어당) | 이 커널 |
|---|---|---|
| FastRPC 호출 | 64 (`mm_u8i4_gate_up_swiglu` + `mm_u8i4_layer_u8in`) × 32 | **1** |
| 와이어 | 36 MB | **4.5 MB** (act f32 3.6 in + out f32 3.6… §6 참조) |
| gather / route / scatter | ARM (6.3 ms) | **DSP 안** |
| act 양자화 | expert마다 (1776행) | **레이어당 한 번** (444행) |
| weight DMA | 호출마다 drain 노출 | **expert 간 파이프라인** |

---

## 2. 인터페이스 — 커널은 포인터를 받고 위치를 모른다

45 §3.1이 요구한 것: Phase D에서 activation이 DSP에 상주해도 커널을 다시 쓰지 않는다. **해법은 커널 시그니처를 IDL과 분리하는 것이다.**

```c
/* hexkl_mm_u8i4_moe.c — act/out이 어디 있는지 모른다 */
int hexkl_mm_u8i4_moe_layer_run(
    hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base, uint32_t vtcm_size,
    uint32_t config_off,
    uint32_t M, uint32_t K, uint32_t inter, uint32_t N_out,
    uint32_t n_experts,
    const uint32_t *h_gate_up,   /* [n_experts] */
    const uint32_t *h_down,      /* [n_experts] */
    const uint32_t *row_index,   /* [n_rows] 토큰 인덱스, expert 순으로 연속 */
    const uint32_t *row_count,   /* [n_experts] */
    const float    *row_weight,  /* [n_rows] 라우팅 가중치 */
    const float    *act_f32,     /* [M x K]     — DDR 어디든 */
    float          *out_f32,     /* [M x N_out] — DDR 어디든, 0으로 초기화됨 */
    hvx_worker_pool *pool);
```

skel 래퍼 둘이 같은 커널을 부른다:

| 래퍼 | 시기 | act/out |
|---|---|---|
| `nntr_hvx_mm_u8i4_moe_layer` (IDL `sequence<float>`) | **Phase A, 지금** | FastRPC가 옮긴 ION 버퍼 |
| `nntr_hvx_moe_layer_resident` (IDL `uint64` 주소) | Phase D | DSP-side DDR, 옮기지 않음 |

**커널은 한 벌이고 Phase D에서 다시 쓰지 않는다.** 이것이 45 §3.1의 요구를 만족시키는 최소 형태다.

---

## 3. VTCM 예산 — 산술로 닫음

M=444, K=2048, inter=1792, N_out=2048, 64행 블록.

| 영역 | 계산 | 크기 |
|---|---|---:|
| act u8 (한 블록) | `(K/32) × ACT_TILE_BYTES` = 64 × 2048 | 128 KB |
| **A: gate_up WH** | `(2048/32) × (3584/32) × 512` | 3584 KB |
| **B: down WH** | `(1792/32) × (2048/32) × 512` | 1792 KB |
| gate f32 + up f32 | `64 × 1792 × 4 × 2` | 896 KB |
| mid u8 AH | `(1792/32) × 2048` | 112 KB |
| result tile | `ACC_TILE_BYTES` | 8 KB |
| **합** | | **6.37 MB** |
| 여유 (arena ~8.3 MB) | | **1.93 MB** |

**weight를 이중 버퍼링하지 않는다.** 그것이 이 설계의 핵심이다 — 2×(3584+1792) = 10.5 MB로 arena를 넘긴다. 대신 §4의 스케줄이 버퍼 하나씩을 시간차로 재사용한다.

---

## 4. 파이프라인 — 왜 이중 버퍼가 필요 없는가

expert `e`의 down을 계산하는 동안 **A(gate_up)는 이미 쓸모가 없다.** 거기로 `e+1`을 당긴다.

```
expert e:
  [A에 gate_up[e] 있음 — e-1의 down mm 중에 당겨둠]
  gate_up mm  (185 us)  ──┬── 이 동안 down[e] → B      (54 us)  ✓ 숨음
  swiglu_det + requant    │
  down mm     ( 92 us)  ──┴── 이 동안 gate_up[e+1] → A (108 us) ✗ 16 us 노출
```

| | µs/expert | ms/layer |
|---|---:|---:|
| down DMA (54) vs gate_up mm (185) | 0 | 0 |
| gate_up DMA (108) vs down mm (92) | 16 | **0.51** |

지금은 drain이 **3.31 ms/layer**(44 §14.3)다. → **−2.8 ms.**

`hexkl_dma_ring`은 그대로 쓴다. `push2d`로 다음 weight를 밀고 `drain()`은 그것을 **소비하기 직전**에만 부른다 — D1이 한 호출 안에서 한 것을 expert 간으로 확장한 것이다(44 §14.5의 하드 요구사항).

---

## 5. DSP 안에서의 gather / route / scatter

```
act_u8[M][K]  ← 레이어당 한 번 양자화 (444행, VTCM 블록 단위)
for e in 0..n_experts-1:
  rows = row_index[base[e] .. base[e]+row_count[e]]
  for blk in 64행 단위:
    A_act ← rows의 act_u8 행들을 VTCM으로 gather      (인덱스 복사, ARM 왕복 없음)
    gate_up mm → dequant → swiglu_det → requant → mid u8
    down mm → result f32 [64 x 2048]
    for r in blk: out_f32[row_index[r]] += result[r] * row_weight[r]   ← route+scatter 합침
```

**route와 scatter가 한 번의 통과로 합쳐진다** — 지금은 ARM에서 `multiply_i` 1776회 + `add_i` 1776회로 텐서 객체를 3552개 만든다(44 §16.3 B2/B3).

`out_f32`는 커널이 0으로 초기화한다 (지금 ARM의 `output.setZero()`가 하는 일). expert들이 같은 토큰 행에 누적하므로 **읽기-수정-쓰기**이고, 레이어당 트래픽은 14.5 MB × 2 = 29 MB ≈ 0.85 ms.

---

## 6. 와이어 — Phase A에서 왜 4.5 MB가 아니라 7.2 MB인가

45 §2는 "u8 in 0.9 + f32 out 3.6 = 4.5 MB"라 했다. **Phase A에서는 act가 f32로 들어온다** (양자화가 DSP로 옮겨갔으므로): 3.6 in + 3.6 out = **7.2 MB**.

| | 와이어 | transport (1.67 GB/s) |
|---|---:|---:|
| 지금 | 36 MB | 17.0 ms |
| **Phase A** | **7.2 MB** | **≈4.3 ms** |
| Phase D (act 상주) | 0 | 0 |

45 §16의 "transport 2.5"는 Phase A 단독으로는 **4.3**이다. 차이 1.8 ms는 Phase D가 가져간다. 정정.

---

## 7. 검증 — 세 단계, 각각 게이트

| # | 무엇 | 게이트 |
|---|---|---|
| V1 | `MoeLayerMatchesTwoCallReference` (신규, `unittest_hvx_mm_u8i4`) — 합성 weight/act로 배칭 커널 vs 현행 64호출 경로 | **비트 동일** (`memcmp`). SNR 아님 — 44 §13.3 규칙 5 |
| V2 | `NNTR_L2_DIFF` 확장 — 실모델 weight/act로 같은 비교 | 32/32 `max_abs_err=0` |
| V3 | 전체 모델 실행 | 텍스트가 CPU와 동일, `layer calls total` 측정 |

V1이 비트 동일할 수 있는 이유: 같은 양자화기, 같은 `swiglu_det`, 같은 HMX 커널. **다른 것은 호출 구조뿐이다.** 비트가 다르면 그 자체가 버그다.

한 가지 예외: act 양자화가 expert별(1776행 중 M_e행씩)에서 레이어별(444행)로 바뀌므로 **행별 scale/zp는 같지만 어느 행이 어느 블록에 속하는지가 달라진다.** 양자화는 행별로 독립이므로(`hvx_quant_rows_u8_params`가 행마다 min/max) 값은 같아야 한다 — V1이 이것도 검증한다.

---

## 8. 구현 순서

| | 산출물 | 기기 |
|---|---|:-:|
| A-1 | `hexkl_mm_u8i4_moe.c` 커널 + `hexkl_mm_u8i4_moe.h` | — |
| A-2 | IDL `mm_u8i4_moe_layer` + skel 래퍼 | — |
| A-3 | V1 테스트 | 1 |
| A-4 | `HtpComputeOps::gemm_qs4cx_moe_layer_fp32` + `Lfm2MoELayer` 배선 | — |
| A-5 | V2 + V3 | 1–2 |
| A-6 | P2 dequant 프로브 (같은 커널 안에서) | 1 |

**A-1의 첫 줄을 쓰기 전에 이 문서의 §3 산술이 맞아야 한다.** 위 표의 합 6.37 MB는 `hexkl_mm_u8i4_moe_layout()`이 런타임에 다시 계산하고 `vtcm_size`와 비교해 `AEE_ENOMEMORY`를 반환한다 — L2 1차가 죽은 자리에 이번엔 검사가 있다.
