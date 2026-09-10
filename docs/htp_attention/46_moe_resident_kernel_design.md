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

---

## 9. V1 결과 (2026-09-10, 기기)

```
U8I4_FIELD path=moe_layer field=bad_elems value=0 of 409600
[       OK ] HmxMmU8I4Layer.MoeLayerMatchesTwoCallReference (937 ms)
[  PASSED  ] 21 tests.
```

**배칭 커널이 기존 64호출 경로와 비트 동일하다.** M=200, 4 experts (70/0/64/33행 — 2블록·빈 expert·블록 경계·행 중복), 실제 HMX/HVX 위에서 409,600개 전부 일치.

§7의 V1 게이트 통과. A-4(모델 배선)로 간다.

기기 없이 먼저 확인했던 것들이 그대로 성립했다: 호스트 구조 검사(worst_rel=0), VTCM 6.37 MB, skel 시그니처, 인자 검증 9종. **기기가 새로 확인해준 것은 하나 — 실물 HMX/HVX 산술에서도 두 경로가 같은 비트를 낸다는 것.**

`RegistryCapacity`는 report-only로 바꿨다. 1.89 GB라는 답이 이미 나왔고 설계가 ION으로 옮겨갔으므로(45 §8–9), 폐기하기로 한 경로에 ≥4.3 GB를 단언하면 매 실행 빨간불이 되고 진짜 실패를 흘려보내게 된다.

---

## 10. A-5 1차 측정 (2026-09-10, 기기) — 배칭은 되고, **내가 uncached 메모리로 되돌아갔다**

```
K=2048 N=2048 M>1  calls=1  rows=444  host=146.9 ms  dsp=144279 us  transport=2650 us
  [quant 17347  swiglu 1911  dequant 5841  acc 104766  drain 5560 | mm<=8854]
layer calls total : 146.9 ms      (직전 fused: 40.1 ms)
```

경로는 맞다 — `calls=1`, `rows=444`, **transport 16.95 → 2.65 ms** (예측 4.3보다 좋다). 그런데 전체는 **3.7배 느려졌다.**

### 10.1 원인: `out_f32`와 `act_f32`는 uncached다

FastRPC 버퍼는 rpcmem, 즉 **uncached DDR**이다 (44 §7.2). 내 커널은 거기에 대고:

| | 무엇 | 측정 |
|---|---|---:|
| scatter | `out_f32[row][c] += res * w`, **스칼라 read-modify-write** 3.6M 원소 | ~102 ms |
| gather | `act_f32`에서 행 복사, top-4라 같은 바이트를 4번 = 14.5 MB **uncached read** | 17.3 (quant 포함) |

원소당 ~28 ns. **Q1이 깨진 축과 같다** (44 §10.1: "128B 연속 쓰기가 2048B stride의 흩어진 쓰기로 뒤집혔다"). 예전 ARM 경로는 캐시된 메모리에 scatter했으므로 이 비용이 없었다.

§11.5의 규칙이 지목하는 바로 그것 — **어느 메모리를 어떤 패턴으로 접근하는가** — 인데, 그 규칙을 문서에 써놓고 커널에서 어겼다.

### 10.2 조치

uncached 버퍼를 **정확히 한 번씩, 순차적으로만** 만진다:

```
시작:  act_f32(ION) → act_c(캐시 힙)  한 번, 3.6 MB 순차
중간:  gather도 scatter도 전부 캐시 힙에서
끝:    out_c(캐시 힙) → out_f32(ION)  한 번, 3.6 MB 순차
```

DSP 힙은 3.75 GB 여유가 있으므로(45 §9) 7.2 MB는 문제가 아니다.

### 10.3 프로파일도 고쳤다 — 진단을 가리고 있었다

`addInvokeMoeLayer`가 `ACC_READ`와 `SCATTER`를 한 버킷에 합쳤다. "같은 종류의 일"이라 묶었는데, 그 결과 104.8 ms가 둘 중 어느 쪽인지 **프로파일이 답하지 못했다.** 이제 `scatter` 컬럼이 따로 있고, staging 복사도 거기 들어간다. `mm` 잔차 계산에서도 빼야 한다 — 안 그러면 scatter 시간이 행렬곱으로 보고된다.

### 10.4 아직 안 갈린 것

- **drain 5.56 ms** (예측 0.51). expert 간 파이프라인이 안 숨고 있다. 다만 scatter가 104 ms를 먹는 상태에서는 타이밍이 왜곡되므로, §10.2 수정 후 다시 본다.
- **생성 텍스트**: 512 토큰 상한에 걸려 `<think>` 안에서 잘렸다. 내용은 정연하고 과제에 맞지만, 직전 정상 실행은 473 토큰에서 요약을 내고 끝났다. **다른 궤적인지 발산인지 V2(`NNTR_L2_DIFF`)가 답한다.**

---

## 11. A-5 2차: scatter를 HVX로 (2026-09-10) — 154.0 → **63.5 ms**, 텍스트 정상

```
calls=1 rows=444 host=63.5 ms transport=2352
  [quant 18461 swiglu 1931 dequant 6486 acc 2794 drain 5966 scatter 14339 | mm<=11186]
```

생성 텍스트가 **정상 3문장 요약(473 토큰)** — §10.4에서 열어둔 항목 하나가 닫혔다. 512 상한에 걸렸던 건 궤적 차이였고 발산이 아니었다.

### 11.1 fused 40.1과의 분해

| 단계 | fused | 배칭 | |
|---|---:|---:|---|
| **transport** | 16.95 | **2.35** | **−14.6** ✓ |
| **quant (+gather)** | 3.57 | **18.46** | +14.9 |
| **scatter (+staging)** | ~2.1 | **14.34** | +12.2 |
| drain | 3.31 | 5.97 | +2.7 |
| 나머지 | 19.25 | 22.4 | +3.1 |

**transport는 설계대로 이겼고, gather와 scatter가 그걸 다 먹었다.** 둘 다 내가 단일 스레드로 짰다 — 같은 파일의 `hvx_swiglu_inplace_f32`는 3.18M 원소를 워커풀로 1.9 ms에 하는데, 내 scatter는 3.64M을 11 ms에 했다.

### 11.2 조치: 둘 다 워커풀로

행 단위로 나눈다. gather는 목적지가 서로 겹치지 않는다. scatter는 **한 블록 안에서는 `row_index`가 서로 다르므로**(토큰은 k개의 *서로 다른* expert를 고른다) 안전하다 — 블록 간·expert 간에는 겹치므로 분할은 블록 안에서만 하고 블록은 순차로 둔다.

### 11.3 그러다 `hvx_worker_pool_run`의 실제 버그를 찾았다

호스트 체크가 2368개 중 2240개 불일치로 즉시 잡았다. 원인은 커널이 아니라 **풀의 degenerate 분기**였다:

```c
if (!pool || pool->n_workers == 0 || n_units <= 1) {
  func(n_units == 0 ? 1u : n_units, 0, ctx);   /* <- n_units개 중 0번 슬라이스 */
```

`func(n_units, 0, ctx)`는 "한 스레드로 전부"가 아니라 **"n_units개 워커 중 0번"**이고, 따라서 **1/n_units만 하고 나머지를 조용히 버린다.** 43 §7의 L1 정정 행이 정확히 이것 때문에 쓰였다 — 테스트가 NULL을 "직렬 레퍼런스"로 쓰다가 1/n만 구워진 weight 이미지와 비교했고, L1이 잘못 되돌려졌다. 그때 함정은 문서에만 적고 **코드는 그대로 뒀다.** 이번에 `func(1u, 0, ctx)`로 고쳤다 — NULL을 넘긴 모든 호출자가 의도한 동작이다.

내 스텁이 그 NULL 분기를 그대로 베껴서 같은 실수를 반복했고, 호스트 체크가 그걸 잡았다. **문서에 적힌 함정은 코드에서 제거해야 한다는 사례.**

---

## 12. A-5 3차: 워커풀 (2026-09-10) — 63.5 → **47.9 ms**

```
calls=1 rows=444 host=47.9 ms transport=3324
  [quant 5503 swiglu 1935 dequant 6164 acc 2739 drain 5388 scatter 9170 | mm<=13688]
```

| 단계 | fused 40.1 | 2차 63.5 | **3차 47.9** |
|---|---:|---:|---:|
| transport | 16.95 | 2.35 | 3.32 |
| **quant(+gather)** | 3.57 | 18.46 | **5.50** |
| **scatter(+staging)** | ~2.1 | 14.34 | **9.17** |
| drain | 3.31 | 5.97 | 5.39 |
| mm≤ (잔차) | 8.87 | 11.19 | 13.69 |

풀이 gather를 18.46 → 5.50으로 줄였다. scatter는 14.34 → 9.17인데 그중 **≈3.6 ms가 staging memcpy**(`act_f32`를 uncached에서 1.8, `out_f32`로 1.8)이므로 본체는 ≈5.6.

### 12.1 벽시계는 반대로 갔다 — 그리고 그건 HTP 밖이다

| | 2차 | 3차 |
|---|---:|---:|
| layer calls total | 63.5 | **47.9** |
| prefill | 2957 | 3675 |
| prefill − registration − layer | **1345** | **2004** |

HTP 밖이 +660 ms다. `convert to registry`가 5.49 → 6.61 ms/weight(+20%)이고 그건 순수 ARM 연산이므로 이번 실행의 CPU가 느렸다는 신호지만, 660 전부를 설명하지는 못한다. `prefill − registration`은 지금까지 1408–2052로 흔들렸다 — §13.3 규칙 4의 "비교 불가" 영역이고, **우리 변경을 재는 숫자는 `layer calls total`이다.**

### 12.2 fused 40.1까지 남은 +7.8 ms

| | |
|---|---:|
| scatter + staging | +7.1 |
| mm 잔차 | +4.8 |
| drain | +2.1 |
| transport | −13.6 |

### 12.3 다음 — gather를 없앤다

activation을 **레이어당 한 번** u8로 양자화해두면:
- 스캔이 1776행 → **444행**
- 블록별 gather가 f32 14.5 MB → u8 **3.6 MB** (4배)
- `act_c` staging 복사(1.8 ms)가 사라진다 — 양자화 패스가 `act_f32`를 순차로 한 번 읽으면 끝

행별 양자화는 독립이므로 scale/zp는 행을 어떻게 묶든 같고, 따라서 **비트 동일이 유지된다.** A-1에서 "비교 대상이 바뀐다"며 미뤄뒀지만 V1이 통과한 지금은 안전하다.

필요한 것: `hvx_quant_rows_u8_rm`(행-major u8로 양자화)와 `hvx_pack_ah_from_u8_rows`(행 인덱스로 골라 AH 타일로 재배치). 예상 quant 5.50 → ≈1.5, staging −1.8.

**mm 잔차 13.69**(실측 mm은 8.87)는 별도로 열어둔다 — 4.8 ms가 어느 단계에도 안 잡힌다. 프로브를 하나 더 넣기 전에는 추측하지 않는다.
