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

## 13. A-5 4차: quantize-once (2026-09-10) — 47.9 → **42.7 ms**

기기 실행. 텍스트 정상(473토큰 요약), prefill 2931 ms / 151.484 TPS(지금까지 최고),
decode 31.4056 TPS, registration 1534.1 ms.

```
K=2048 N=2048 M>1 calls=1 rows=444 host= 42.7 ms dsp=40432 transport=2291
  [quant 5537 swiglu 1926 dequant 6005 acc 2794 drain 5466 scatter 8066
   | mm<=10638 (24.9% of host)]
layer calls total : 42.7 ms
```

`prefill − registration − layer`는 1354로, 12.1의 2004에서 정상 범위(1345–1408)로
돌아왔다. 그 660 ms는 역시 우리 코드가 아니었다.

### 13.1 예측은 맞았고 이유는 틀렸다

측정 전에 적어둔 예측은 `quant 5.50 → ≈1.5`, `layer calls total ≈43`이었다.
총합은 42.7로 왔지만 **quant는 5503 → 5537로 움직이지 않았다.**

| 단계 | 3차 | 4차 | Δ |
|---|---:|---:|---:|
| quant | 5503 | 5537 | **+0.03** |
| scatter(+staging) | 9170 | 8066 | −1.10 |
| mm 잔차 | 13688 | 10638 | −3.05 |
| dequant | 6141 | 6005 | −0.14 |
| drain | 5124 | 5466 | +0.34 |
| acc | 2792 | 2794 | 0 |
| swiglu | 1934 | 1926 | 0 |
| **layer calls total** | **47.9** | **42.7** | **−5.2** |

−5.2 ms는 예측한 자리에서 오지 않았다. 이건 그냥 넘길 결과가 아니다 — 문서 44
§11.5의 "세는 논증은 믿지 마라"가 정확히 이 모양이다. **합이 맞았다는 사실이 각
항이 맞았다는 뜻은 아니다.**

읽을 수 있는 가설은 두 개뿐이고, 지금 프로파일로는 **어느 쪽도 확정할 수 없다**:

1. quant 버킷이 실제로는 세 가지를 담고 있었다 — 레이어 전체 activation 양자화,
   블록별 `hvx_gather_ah_u8` + scale/zp 슬라이스, SwiGLU 출력의 블록별 재양자화.
   재구조화가 첫 번째를 5.5 → 1.5로 줄였더라도 두 번째가 새로 들어왔으므로
   **버킷 합은 그대로일 수 있다.**
2. mm 잔차가 −3.05 움직인 것은 잔차이기 때문이다 — 이름 없는 시간이 전부 여기
   모인다. 3차의 mm 13688은 실측 mm 8870 대비 +4.8이었고, 그 미설명분이 줄었다.

1번이 맞다면 재구조화는 성공했고 비용이 이름만 바꿔 앉은 것이다. 2번이 맞다면
재구조화는 quant에 영향이 없었고 이득은 다른 데서 왔다. **버킷 하나로는 답이 안
나온다** — ACC_READ와 SCATTER를 한 칸에 넣었다가 104.8 ms를 놓쳤던 것과 같은 실수다.

### 13.2 그래서 버킷을 쪼갰다 (코드 완료, 미측정)

`HEXKL_PROBE_QUANT` 하나를 셋으로:

| 슬롯 | 재는 것 |
|---|---|
| `HEXKL_PROBE_QUANT` | 레이어당 1회 activation 양자화 (`hvx_quant_rows_u8_params` + `hvx_quant_pack_u8_ah`) |
| `HEXKL_PROBE_GATHER` | `hvx_gather_ah_u8` + 블록별 scale/zp 슬라이스 |
| `HEXKL_PROBE_REQUANT` | SwiGLU 출력의 블록별 양자화 (재구조화가 손대지 않은 부분) |

동시에 **`scatter`에서 `stage`를 떼어냈다.** 지금까지 `scatter_us`는 라우팅
곱셈+누적과 FastRPC 버퍼 staging 복사를 같이 담고 있었는데, 13.3의 변경이 건드리는
건 정확히 후자뿐이라 붙어 있으면 효과를 읽을 수 없다.

`MOE_T_GATHER`/`MOE_T_REQUANT`가 붙어 `MOE_N_STAGES`가 9 → 11이 됐다(§15.3에서 12). 즉
**skel을 다시 빌드해야 한다** — `./test/htp/build.sh`. 안 하면 `0x8000040E`
(EBADPARM)로 죽는다. `build_android.sh`는 skel을 빌드하지 않는다.

### 13.3 staging 복사를 DMA로 (코드 완료, 미측정)

`scatter 8066` 안에는 staging memcpy가 약 3.6 ms 들어 있다(3.6 MB in + 3.6 MB out,
uncached rpcmem ↔ cached heap). 스칼라 코어가 uncached DDR을 읽는 건 이 하드웨어가
가진 가장 느린 동작이고, **대량 이동용 엔진이 바로 옆에 놀고 있다.**

`moe_dma_copy()`가 `hexkl_dma_ring_push2d`로 옮긴다. 두 지점 다 DMA가 비어 있다:
activation 복사는 전문가 루프 **전**(0번 전문가의 gate_up만 in-flight),
출력 복사는 루프 **후**(아무것도 없음). `hexkl_dma_ring_drain()`이 pending 전부를
기다리는 제약과 충돌하지 않는 유일한 두 자리다.

### 13.4 fused 40.1까지 남은 +2.6 ms

| | fused 40.1 | 배칭 42.7 | Δ |
|---|---:|---:|---:|
| **transport** | 16.95 | **2.29** | **−14.7** |
| scatter + staging | ~2.1 | 8.07 | +6.0 |
| drain | 3.31 | 5.47 | +2.2 |
| quant | 3.57 | 5.54 | +2.0 |
| mm 잔차 | 8.87 | 10.64 | +1.8 |

**설계가 약속한 것(transport −14.7)은 그대로 왔다.** 다 못 챙긴 이유는 ARM에서
DSP로 옮긴 일(gather·scatter·staging)이 DSP에서 더 비싸기 때문이다. DSP는 HMX
행렬곱에서 빠른 것이지 대량 메모리 이동에서 빠른 게 아니다 — 이게 이 단계에서
배운 것이고, 남은 항목들(13.3의 DMA, A-6의 dequant)은 전부 같은 성질이다.

`drain 5.47`은 설계가 예측한 0.51의 10배다. 전문가 간 weight DMA 파이프라인이
**숨지 않고 있다.** 원인 후보는 (a) 앞 전문가의 계산이 다음 weight를 끌어올 만큼
길지 않다, (b) `drain()`이 pending 전부를 기다리므로 필요 없는 대기까지 한다.
13.3이 activation/출력 복사를 같은 링에 얹으므로 **이 숫자를 (b) 관점에서 다시
봐야 한다** — 다음 측정에서 drain이 오르면 (b)가 원인이라는 증거다.

## 14. 인계 (2026-09-10) — 다음 세션이 여기서 시작한다

### 14.1 지금 상태

| | |
|---|---|
| 브랜치 | `claude/lfm2-moe-ffn-hexkl-2ivn5v` |
| 기기 최고 기록 | **42.7 ms** (§13). 이겨야 할 fused 경로는 **40.1 ms** |
| 정확도 | V1 비트 동일(`bad_elems=0 of 409600`), 실모델 텍스트 473토큰 정상 |
| 커밋됐지만 **기기에서 안 잰 것** | §19의 R0 (프로브 3종, 전부 측정 전용) |

**커밋됐지만 안 잰 것이 두 개 있다는 게 이 인계의 핵심이다.** 둘 중 프로브 분할은
동작을 바꾸지 않는 측정 전용이고, DMA만 기능 변경이다 — 그래서 한 번의 기기 실행이
"DMA가 먹혔나"와 "quant 5537의 정체가 뭔가"를 동시에 답한다.

### 14.2 다음 세션의 첫 세 줄

0. **IDL이 바뀌었으면 `bash nntrainer/tensor/htp_backend/generate_stub.sh` 먼저.**
   생성물이 **두 벌**이다 — `test/htp/generated/`(skel, `build.sh`가 만듦)와
   `nntrainer/tensor/htp_backend/generated/`(ARM 클라이언트, `generate_stub.sh`가
   만듦). `build.sh`만 돌리면 **skel은 새것, ARM 스텁은 옛것**이 되고, ARM 빌드가
   새 심볼에서 깨진다. 그걸 못 보고 옛 바이너리로 돌리면 EBADPARM이다 —
   2026-09-14에 이렇게 한 사이클을 잃었다. meson이 이제 IDL보다 낡은 스텁을
   거부한다.
1. `./test/htp/build.sh` — **skel을 반드시 다시 빌드한다.** `MOE_N_STAGES`가
   9 → 15로 늘었다. 안 하면 `nntr_hvx_mm_u8i4_moe_layer_timed failed:
   err=-2147482610` (0x8000040E, EBADPARM). 이 사이클을 이미 한 번 잃었다.
2. `./build_android.sh` — **`--htp` 플래그는 없다.** 인자 파서가 모르는 옵션에
   `exit 1`을 하므로 그대로 치면 빌드가 시작도 안 된다(이 문서가 한 번 틀리게
   적었다). HTP는 플래그가 아니라 **`builddir`에 이미 박혀 있는 meson 옵션**이다:
   `package_android.sh`는 `builddir`가 있으면 `meson configure` + `--wipe`로
   기존 옵션을 보존하고, 없으면 `-Denable-htp` 없이 새로 만든다. 따라서
   **`--clean`을 쓰면 안 된다** — builddir가 지워지고 HTP가 조용히 꺼진다.
   확인: `meson configure builddir | grep -i htp` 가 `enable-htp true`여야 한다.
   (skel은 이 스크립트가 안 만든다 — 1번이 만든다.)
3. `NNTR_HTP_PROFILE=2`로 모델 실행 → `layer calls total`과 새 칼럼
   `[quant … gather … requant … stage …]`을 본다.

### 14.3 그 측정이 답하는 것

| 관측 | 해석 | 다음 |
|---|---|---|
| `quant ≈1.5`, `gather ≈2.5`, `requant ≈1.5` | §13.1 가설 1. 재구조화는 먹혔고 비용이 gather로 이름만 옮겼다 | gather를 줄인다 — 지금은 행마다 `k_tiles`번 32바이트 복사다. 행 인덱스로 DMA 2D descriptor를 쓰거나, 아예 gather 없이 전문가별 행을 양자화 단계에서 바로 AH에 배치 |
| `quant ≈5.5`, `gather ≈0` | 가설 2. 재구조화가 quant에 영향이 없었다 | `hvx_quant_pack_u8_ah` 자체가 병목 — 내부 프로브 필요 |
| `stage`가 3.6 → ≈0.5 | §13.3 DMA 성공 | scatter 본체 ~4.5 ms가 다음 표적 |
| `stage`는 줄었는데 `drain`이 올랐다 | §13.4의 (b) — `drain()`이 pending 전부를 기다린다 | 링을 분리하거나 `wait_idx`로 필요한 것만 기다린다 |

### 14.4 남은 Phase A 항목 (우선순위)

- **A-6 / dequant 6.0 ms.** 1.75 G elem/s인데 이론은 25.6 G/s — 15배. 아직 프로브
  안 넣었다. 문서 44 §16의 P2.
- **mm 잔차.** 10638인데 실측 mm은 ~8870. §13.2의 분할이 일부를 흡수할 수 있다.
- **drain 5.47 vs 설계 0.51.** §13.4.
- **V2**: `NNTR_L2_DIFF`를 배치 호출까지 확장(§7의 V1/V3는 끝, V2만 남음).
- **§15.3 실험**: `nntr_config.json`의 `moe_htp_layers`를 `"2"` → `"5"`. 코드 변경
  없음. HTP 레이어의 추가 gather 비용이 HTP 고유인지 registration 여파인지 가른다.
- **P4 / ION arena** (문서 45 §8.4): WH 바이트를 파일로 한 번 굽고 ION에 mmap.
  전체 모델 상주(메모리)와 prefill에서 registration 1.5 s 제거, 둘 다에 필요.

그 뒤는 문서 45: Gate 1(M1 블록별 CPU 분해) → Phase B(conv) → C(attention)
→ D(레이어 오케스트레이션) → E(decode).

### 14.5 바뀌지 않는 규칙

- **기기에서 재기 전에는 "확인했다"고 쓰지 않는다.** 이 문서에서 §13.2·§13.3이
  "코드 완료, 미측정"인 이유다.
- 한 버킷이 두 가지를 담으면 진단이 불가능하다. ACC_READ+SCATTER에서 104.8 ms를
  놓쳤고, QUANT에서 §13.1을 놓쳤다. 세 번은 없어야 한다.
- 세는 논증(문서 44 §11.5)은 못 믿는다. 지배하는 건 메모리 접근 패턴과 대기다.
- decode < 30 TPS면 온도 게이트 — transport/host/벽시계는 비교 불가, DSP 내부
  단계만 유효하다.

## 15. 전문가 32개의 산술 — 이 커널이 무엇에 묶여 있는가 (2026-09-10)

"experts가 32개 맞냐"는 질문에서 나온 정리. 세 가지가 독립적으로 확인해준다.

| | |
|---|---|
| 전문가 1개 | gate_up 2048×3584 + down 1792×2048 = **11.01M 파라미터** |
| × 32 × 22 MoE 레이어 | **7.75B** → embedding·attention·conv 포함 ≈8.4B = 이름의 "8B" |
| top-4만 계산 | 4 × 11.01M × 22 = **969M** = 이름의 "**A1B**" |
| VTCM 6.37 MB (§3) | gate_up int4 3.5 MiB + down 1.75 MiB = 전문가 하나치 **5.25 MiB** + 작업버퍼 |

top-4가 16개 중 4개였다면 활성은 같아도 총량이 4B라 이름이 안 맞는다. **32이어야 8B다.**

### 15.1 그래서 무엇에 묶여 있나 — 가중치 스트림이 고정비다

레이어당 전문가 가중치 = 32 × 5.25 MiB = **176 MB (int4)**. 그런데 444토큰 × top-4 =
**1776 라우팅 행**이 32개 전문가에 흩어지므로 **전문가를 하나도 안 건너뛴다.**
176 MB를 통째로 DMA로 끈다.

**MoE의 전제("8B 저장, 1B 계산")가 prefill에서는 성립하지 않는다.** 안 쓰는 전문가를
건너뛸 수 있어야 이득인데, 배치가 크면 전부 쓴다. `drain 5.47 ms`가 그 고정비의
안 숨은 부분이고, 우리가 협상할 수 없는 바닥이다.

뒤집으면 **이게 배칭이 그렇게 크게 먹힌 이유**이기도 하다: 176 MB는 토큰 수와
무관하게 고정이므로 토큰이 2배면 토큰당 가중치 비용은 절반이다. decode(M==1)는
정반대 — 전문가 4개만 = 22 MB인데 64행 타일에 1행만 들어간다.

### 15.2 두 번째: 55.5행 vs 64행 타일

```
1776 라우팅 행 ÷ 32 전문가 = 55.5행/전문가
BR = HEXKL_HMX_INT8_BLOCK_N_ROW = 64
```

55.5는 64 **바로 아래**라 가장 나쁜 자리다. `for (mb = 0; mb < n_e; mb += BR)`이므로
70행짜리 전문가는 두 번째 블록을 90% 비운 채 돌리고, **HMX는 `m_blk`과 무관하게 타일
64행을 전부 계산한다** (타일 루프에 `m_blk`이 없다 — HVX 후처리만 `m_blk`을 쓴다).

| | 블록 | 발행 MAC | 유효 19.55 G 대비 |
|---|---:|---:|---:|
| 전부 ≤64 | 32 | 22.5 G | +15% |
| 절반이 넘침 | 48 | 33.8 G | +73% |

**우리는 블록을 몇 개 발행하는지 한 번도 안 세어봤다.** mm의 40%가 이 둘 사이에 있는데
프로파일이 두 경우를 구분하지 못한다.

### 15.3 그래서 `HEXKL_PROBE_BLOCKS`를 넣었다 (코드 완료, 미측정)

시간이 아니라 **개수**다 — `HEXKL_PROBE_ACC_STRIDE`와 같은 성격. 프로파일에
`blocks=<n>`으로 붙는다. `MOE_N_STAGES` 11 → **12**.

읽는 법:
- `blocks ≈ 32` → 라우팅이 고르다. mm 잔차는 타일 낭비가 아니라 딴 데 있다.
- `blocks ≈ 48+` → 타일 낭비가 mm의 3분의 1이다. 전문가별 행을 64의 배수로 맞추도록
  블록을 자르는 방식(전문가 경계를 넘어 묶기)이 다음 표적이 된다.

§13.2·§13.3과 같은 skel 재빌드 사이클에 얹히므로 **기기 실행 한 번이 세 가지를 답한다.**

## 16. A-5 5차 (2026-09-14, 기기) — 42.7 → **37.2 ms**, fused 40.1을 처음 넘었다

```
prefill: 444 tokens, 4392 ms, 101.093 TPS      ← 온도 게이트 (아래)
generation: 473 tokens, 37999 ms, 12.4477 TPS
registration total : 2296.9 ms
K=2048 N=2048 M>1 calls=1 rows=444 host=37.2 ms dsp=36054 transport=1146
  [quant 610 gather 3623 requant 1254 swiglu 1927 dequant 5314 acc 2768
   drain 5016 scatter 1431 stage 389 | mm<=13722 (36.9%) blocks=43]
layer calls total : 37.2 ms
```

**온도 게이트.** decode 12.4 TPS → ARM이 스로틀 상태. 그런데 DSP 단계는 안 움직였다
(swiglu 1927 vs 1926, acc 2768 vs 2794) — **DSP 클록은 멀쩡했다.** 따라서 DSP 분해는
유효하고, prefill 4392·registration 2297·벽시계는 비교 불가. 이 문서에서 4차 → 5차를
비교하는 숫자는 DSP 단계뿐이다.

텍스트: 473토큰 생성, 정상 실행들과 같은 길이. 본문은 못 봤으므로 **정상이라고
확인한 것은 아니다.** 다음 실행에서 본문 또는 `NNTR_L2_DIFF`로 닫는다.

### 16.1 새 칼럼 셋이 전부 답을 냈다

| 질문 (§14.3) | 답 |
|---|---|
| quant 5537의 정체 | **quant 610 + gather 3623 + requant 1254.** §13.1 가설 1이 맞았다 — quantize-once는 먹혔고(5.5 → 0.6) 비용이 gather로 이름만 옮겼다 |
| staging DMA | **stage 389.** scatter+stage 8066 → 1820, **−6.2 ms.** 내가 "staging 3.6 / scatter 4.5"로 나눠 짐작한 건 틀렸다 — 실제는 대략 6.6 / 1.4였다. 세는 논증이 또 빗나갔고, 버킷을 쪼갠 게 맞았다 |
| blocks | **43.** 32 전문가 중 11개가 64행을 넘었다. 타일 활용 1776 / (43×64) = **64.5%.** 발행 MAC 30.3 G vs 유효 19.55 G |

### 16.2 4차 → 5차 (DSP 단계만)

| 단계 | 4차 | 5차 | Δ |
|---|---:|---:|---:|
| scatter + stage | 8066 | 1820 | **−6246** |
| dequant | 6005 | 5314 | −691 |
| drain | 5466 | 5016 | −450 |
| quant+gather+requant | 5537 | 5487 | −50 |
| swiglu / acc | 1926 / 2794 | 1927 / 2768 | 0 |
| **mm 잔차** | 10638 | **13722** | **+3084** |
| DSP 합계 | 40432 | 36054 | −4378 |

**mm 잔차 +3.1 ms는 설명이 안 된다.** 라우팅은 결정적이고 같은 프롬프트이므로 블록 수는
양쪽 다 43이다. 잔차 안에 있는 것은 HMX 발행 루프, `acc_clear`, `moe_push_weight`,
루프 오버헤드뿐이고 이번 변경은 그 어느 것도 건드리지 않았다. 후보: (a) 잔차가
"이름 없는 시간 전부"라서 이전엔 scatter 버킷에 가려져 있던 무언가가 드러났다,
(b) `push2d`가 링 상태에 따라 블록한다. **프로브 없이는 못 가른다** (§17 A5).

### 16.3 천장 — 이 커널만으로는 CPU의 1.9배가 끝이다

43블록에서 못 줄이는 바닥 = mm 13.7 + acc 2.8 = **16.5 ms.** 나머지 19.5 ms는 전부
포장(quant/gather/requant/dequant/swiglu/drain/scatter/stage)이다. 포장을 0으로 만들어도
16.5 → CPU 31.8 대비 **1.9×.** 문서 44 §16이 말한 "MoE FFN만으로는 1.26×"가 배칭으로
1.9×까지 올라온 것이지, "훨씬 빠르게"(2~3×)는 여전히 문서 45(전체 상주)의 몫이다.
이 커널은 그 Phase A이고, 아래 §17은 전부 그쪽으로 이월된다.

## 17. 다음 최적화 — 순위·예상·순서

효과는 ms/레이어, 근거는 §16의 실측. "예상"은 §11.5 규칙대로 **예측이지 약속이 아니다.**

### A. 커널 내부 (37.2 → ≈25 예상)

| # | 항목 | 지금 | 예상 | 어떻게 | 비트동일 | 크기 |
|---|---|---:|---:|---|---|---|
| **A5** | **mm 잔차 프로브** | 13722 | — | HMX 발행 루프에 `HEXKL_PROBE_MM`. 교란을 감수한다 — 13.7이 HMX 시간인지 딴 것인지 이걸로만 안다. **A2의 중첩이 값어치가 있는지도 이게 정한다** | 측정만 | 극소 |
| **A4** | **drain 대역폭 프로브** | 5016 | — | push 바이트와 push→완료 시간 → 유효 GB/s. 전문가당 노출 157 µs가 대역폭(rpcmem 11 GB/s?)인지 순서(gate_up[e+1]이 늦게 나감)인지 가른다 | 측정만 | 소 |
| **A1** | **gather 제거** | 3623 | ≈1000 | `hvx_quant_pack_u8_ah`에 row map을 주어 **전문가 순서·64행 패딩으로 바로 팩** (2752 슬롯 × 2 KB = 5.6 MB heap). 블록당 act는 128 KB 연속 DMA 한 번 — 이전 블록 down 뒤에 숨는다 | ✓ 행별 양자화는 위치 무관 | 소 |
| **A2** | **dequant 병렬 + HMX/HVX 중첩** | 5314 | ≈1500 | 지금은 **싱글코어**, 64×32 타일당 1회 × 176/블록 × 43. 풀 워커 3개가 논다. ① acc 타일 112개를 896 KB VTCM에 모아 풀로 한 번에 (1.9 MB 여유 안) ② 워커가 nt를 dequant하는 동안 메인이 nt+1 HMX 발행 (문서 35 패턴) | ✓ | 중 — 레이아웃 변경 |
| **A3** | **에필로그 융합** dequant→SwiGLU | 5314+1927 | ≈3000 | gate f32 쓰기 + swiglu 읽기/쓰기 = 블록당 ~1.8 MB VTCM 왕복. gate/up 열쌍을 dequant하면서 바로 SwiGLU → 3584 대신 1792 f32만 쓴다. requant는 행 max 스캔이 필요하므로 분리 유지 | ✓ (V1로 증명) | 중상 — A2 뒤 |
| A6 | acc read | 2768 | — | A2 ②에 흡수 (타일 비동기 읽기). 단독으로는 구조적 | | |
| A7 | blocks 43 | — | — | **커널이 못 고친다.** Σceil(n_e/64)은 라우팅이 정한다. 레버는 prefill 청크 크기뿐 → 문서 45 Phase D | | |

합: A1 −2.6, A2 −3.8, A3 −2, A4가 순서 문제면 −3 → **≈25–28 ms**, CPU 31.8을 레이어
단위로 처음 이긴다. 1.1–1.3×. 그게 이 커널의 한계 근처다 (§16.3).

### B. 커널 밖, prefill 안 — 이 페이지에서 가장 큰 숫자

| # | 항목 | 지금 | 예상 | 어떻게 |
|---|---|---:|---:|---|
| **B1** | **P4 / ION arena** | **2297 ms** | ≈0 | `convert to registry` 547 + `register FastRPC` 1597 = 176 MB를 110 MB/s로 매 실행. **22층이면 50초.** WH 바이트를 양자화 단계에서 파일로 굽고, 로드 때 ION에 mmap, 핸들로 등록. Gate 0의 1.89 GB 레지스트리 천장도 같이 없어진다 |

`layer calls total`엔 안 잡히지만 prefill 벽시계에선 레이어 37 ms의 **62배**다.
문서 45의 모든 Phase가 이걸 전제한다.

### C. 구조 — 문서 45 그대로

Phase B(conv 18) → C(attention 6) → D(오케스트레이션) → E(decode). §16.3의 이유로
2~3×는 여기서만 나온다.

### 17.1 실행 순서 (기기 실행 단위)

| 실행 | 넣는 것 | 기능 변경 | 답하는 것 |
|---|---|---|---|
| **R1** | A5 + A4 프로브 + **A1** | A1 하나 | mm 13.7의 정체, drain의 원인, gather −2.6 실현 여부. 프로브는 측정 전용이라 나빠지면 원인은 A1로 특정 |
| R2 | A2 | dequant 배치+풀(+중첩) | −3.8 실현 여부, HMX/HVX 중첩이 되는지 |
| R3 | A3 | 융합 | V1 비트동일 + −2 |
| R4 | A4 수정 (R1 결과에 따라) | | drain |
| 병렬 | B1 | 별도 트랙 | registration 0 |

R1은 skel 재빌드 필요 (`MOE_N_STAGES` 12 → 13+). 규칙은 §14.5 그대로.

## 18. prefill TPS 결산 (2026-09-14) — §16.3의 1.9×는 틀렸다

### 18.1 정정

§16.3은 "바닥 16.5 ms vs CPU 31.8 = 1.9×"라고 적었다. **범위가 안 맞는 비교였다.**
16.5는 `layer calls total`(DSP 호출 하나)이고 31.8은 CPU의 **레이어 전체**다. HTP
쪽에는 그 위에 ARM staging 1.0 + FastRPC dispatch 1.2 + ARM 비-ffn이 더 붙는다.

기준선은 문서 44 §15.1의 P3 측정이다 — **같은 forward pass 안에서** HTP 레이어(2번)와
CPU 레이어(4–7번 평균)를 나란히 잰 유일한 숫자이므로 온도 문제가 없다:

```
CPU MoE 레이어 = 31.805 ms  (비-ffn 5.549 + ffn 26.256)
```

같은 범위로 맞추면:

| | 레이어 ms | vs CPU 31.8 |
|---|---:|---:|
| 이론 바닥 (§16.3이 1.9×라고 한 것) | 20.8–23.7 | **1.34–1.53×** |

batched 경로의 ARM 비-ffn은 **안 재봤다** — P3는 fused 경로를 쟀고(그때 ARM gather가
6.17 ms), 지금은 gather·scatter가 DSP로 갔다. 2.135(setup/router/topk만 남음) ~
5.0 사이로 두고 폭으로 표기한다. **M0을 batched 경로에서 다시 재야 닫힌다.**

### 18.2 prefill TPS — 22층 전부 HTP, registration은 B1이 제거했다고 가정

```
CPU: 31.805 × 22 = 700 ms + 나머지 890 ms = prefill 1590 ms → 444/1.590 = 279 TPS
```

| 시나리오 | 레이어 ms | prefill TPS | **vs CPU 279** |
|---|---:|---:|---:|
| **오늘 (측정 37.2)** | 41.5–44.4 | 238–246 | **0.85–0.88×** |
| A1+A2+A3 후 (예상) | 30.8–33.7 | 272–283 | **0.97–1.01×** |
| 이론 바닥 (포장 0) | 20.8–23.7 | 315–329 | **1.13–1.18×** |
| MoE FFN = 0 (물리적 불가) | — | 423–450 | **1.52–1.61×** |

세 가지가 바로 읽힌다:

1. **오늘은 아직 CPU보다 느리다.** 22층을 다 올려도 0.85–0.88×. `layer calls total
   37.2 < fused 40.1`은 **HTP 내부 비교**였고 CPU 비교선은 따로다. §16에서 "처음
   넘었다"고 쓴 것은 fused를 넘은 것이지 CPU를 넘은 것이 아니다.
2. **A1+A2+A3를 다 해도 동률이다.** 0.97–1.01×. 이 세 레버는 prefill TPS를 CPU 위로
   올리지 못한다.
3. **Amdahl 천장이 1.6×다.** MoE FFN을 0으로 만들어도 나머지 890 ms(prefill의 **56%**)가
   남는다.

**민감도.** 문서 44 §16은 MoE를 841 ms(53%)로 적었는데 P3에서 유도하면 700 ms(44%)다.
841 쪽이면 천장 2.1×, 이론 바닥 1.32×로 올라간다. 어느 쪽이든 **"A만으로는 동률,
이론 바닥에서 1.1–1.3×"**는 안 바뀐다. (이 불일치 자체도 열어둔다 — 같은 실행에서
두 숫자를 다시 뽑아야 한다.)

### 18.3 그래서 목표를 어디에 두나

prefill의 56%가 MoE 밖이므로 **문서 45(conv 18 · attention 6 · 오케스트레이션)가
유일한 길**이다. 다만: 모든 블록에서 1.5×를 내면 prefill 1.5×, 2~3×를 원하면 모든
블록에서 2~3×가 나와야 한다. **conv/attention이 DSP에서 얼마에 도는지 한 번도 안
쟀으므로 그 숫자는 아직 근거가 없다.**

**현재 근거로 말할 수 있는 것: prefill TPS 1.2–1.5×가 현실적 목표. 2× 이상은 미근거.**

그리고 end-to-end 벽시계는 이번 실행에서 prefill이 42.4초 중 4.4초(10.4%)이므로
prefill 1.5×는 전체 **1.03×**다. 벽시계를 움직이려면 decode이고, decode는 문서 43
§7에서 대역폭 바운드로 닫혔다 — 커널이 아니라 토큰당 읽는 바이트를 줄이는 구조 변경이
필요하다. 이건 이 문서의 범위 밖이다.


## 19. R0 — 게이트 프로브 (2026-09-14, 코드 완료·미측정)

§18의 결론에 따라 순서를 바꿨다. 기능 변경 A1(gather 제거)은 **빼고**, 측정 전용
세 가지만 넣는다. 한 번의 기기 실행이 "나머지 계획이 값어치가 있는가"를 답한다.

### 19.1 세 프로브

| | 무엇 | 왜 지금 |
|---|---|---|
| **A5** `HEXKL_PROBE_MM` | HMX 발행 루프(acc_clear + k-타일 mm)를 **차분으로** 잰다 — n-타일 루프 전체 시간에서 이미 프로브된 ACC_READ·DEQUANT의 증가분을 뺀다. 112타일 루프에 클록 읽기 2회(타일마다면 224회) | mm은 지금까지 **잔차**였고, 잔차는 이름 없는 것을 전부 흡수한다. 같은 43블록에서 10638 → 13722로 3.1 ms 움직였는데 그 사이 변경이 잔차 내용물을 안 건드렸다(§16.2). 이름을 붙여야 남는 게 진짜 남는 것이 된다 |
| **A4** `DMA_KB` + `DMA_FIRST` | 링에 밀어넣은 KB 총량, 그리고 **첫 weight drain**의 µs — 빈 링에 gate_up 하나(3584 KB)만 떠 있는 유일한 지점이라 파이프라인이 아니라 전송을 잰다 | **계획 전체의 게이트.** 레이어당 176 MB를 mm 뒤에 숨기려면 12.8 GB/s가 필요하다. 6 GB/s 근처면 레이어는 대역폭 바운드 ~30 ms이고 **A1–A3를 다 해도 안 내려간다** |
| **M0** `ffn` 타이머 | `tryMoeLayerOnAccelerator` 호출을 감싼다 | 여기에 타이머가 **없었다**. batched 경로에서 ARM gather·route·scatter는 DSP로 가서 0을 읽고 ffn도 0이었으므로 레이어 벽시계를 설명하는 단계가 하나도 없었다 — §18.1이 ARM 잔여를 2.1–5.0 범위로 쓴 이유 |

프로파일 출력이 바뀐다: `mm<=` 가 **`mm <측정값> | rest<=<잔차>`** 가 되고,
줄 하나가 추가된다:

```
[HTP-PROFILE]     weight DMA: N KB/call, first 3584 KB took T us = X GB/s;
                  averaged over the call Y GB/s
```

암산용: **GB/s ≈ 3670 / T(µs).**

### 19.2 이미 나와 있는 단서

5차 실행의 `stage 389 µs`는 staging DMA로 **7.28 MB**(act 3.64 in + out 3.64)를
옮긴 시간이다 → **18.7 GB/s**. 필요한 12.8을 넘는다. 다만 그건 **DDR→DDR**이고
weight는 **DDR→VTCM**이라 경로가 다르다. A4가 확정한다. 게이트가 통과할 쪽에
무게가 실린다.

### 19.3 읽는 법

| 관측 | 결론 | 다음 |
|---|---|---|
| `first` ≲ 290 µs (≥12.6 GB/s) | 대역폭은 바닥이 아니다 | A1 → A2 → A3 예정대로 |
| `first` ≳ 600 µs (≲6 GB/s) | **레이어가 대역폭 바운드** | A1–A3 취소, B1(ION arena)로 직행 — 경로 자체가 바뀐다 |
| `mm` ≈ 13700, `rest` ≈ 0 | 잔차의 정체는 HMX 발행이었다 | A2의 HMX/HVX 중첩이 최대 레버 |
| `mm` ≪ 13700, `rest` 큼 | 이름 없는 것이 따로 있다 | `moe_push_weight`(push2d 블로킹?)부터 의심 |
| M0 `ffn` vs 레이어 벽시계 | §18.1의 2.1–5.0이 숫자가 된다 | prefill TPS 표의 폭이 닫힌다 |

### 19.4 검증

호스트 체크에 weight DMA 바이트 수 검사를 추가했다 — DMA 스텁이 즉시 완료하므로
**빠뜨린 push는 결과로 드러나지 않는다**(이른 push와 달리). 이 카운터가 유일한
방어선이고, 동시에 기기에서 GB/s의 분자이므로 틀리면 계획의 게이트를 조용히
잘못 읽게 된다. `12 KB over 4 experts (want 12)` 통과.

`MOE_N_STAGES` 12 → **15**. `./test/htp/build.sh` 필수.

## 20. B1a — 구운 WH 바이트 캐시 (2026-09-14, 코드 완료·미측정)

§18.2가 registration 2297 ms를 제거했다고 **가정**한 표를 냈다. 이건 그 가정을
실제로 만드는 작업이다.

### 20.1 먼저: 1597 ms는 마샬링이 아니다

프로파일이 `register FastRPC 1597.0 ms (24.95 ms/weight)`로 부르는 것을 처음엔
7.34 MB 전송 비용으로 읽었다. **아니다** — FastRPC 호출은 동기라 저 시간에
**DSP의 RM→WH bake가 포함**된다. `hexkl_weight_u8i4_register`의 주석과 문서 43
§2가 이미 그렇게 적고 있었다: gate_up 하나가 **7168개의 독립 512바이트 타일**
재배치이고 weight당 34–48 ms.

**그래서 ION으로 옮기는 것만으로는 안 줄어든다.** 줄이려면 bake를 다시 하지
않아야 하고, bake 결과는 결정적이므로(43 §7 FNV-1a) **파일에 캐시할 수 있다.**

| | 지금 | 캐시 히트 |
|---|---:|---:|
| convert to registry | 8.54 ms | **0** (scale·colsum이 파일에 있다) |
| register FastRPC | 24.95 ms | bake 없음, 페이로드 절반 |
| 와이어 페이로드 | 7.34 MB (int4를 int8 컨테이너에) | **3.5 MB** (WH는 바이트당 int4 2개) |

### 20.2 무엇을 만들었나

**DSP** (`hexkl_mm_u8i4_dma.c`) — 기존 register를 셋으로 쪼갰다:
`hexkl_weight_u8i4_check`(모양·바이트수), `_free_slot`, `_fill_slot`(할당+복사).
그 위에 두 진입점:

- `hexkl_weight_u8i4_register_baked(...)` — bake 없이 슬롯에 복사만. VTCM도
  워커풀도 안 받는다. `wh_len`을 **검증한다** — 이 바이트는 파일에서 왔고, 길이가
  틀리면 크래시가 아니라 **조용히 틀린 행렬곱**이 된다.
- `hexkl_weight_u8i4_export(...)` — 구운 바이트를 도로 읽어낸다.

**IDL/skel** — `weight_register_u8i4_baked`, `weight_bake_export`.

**ARM** (`htp_weight_cache.h`, 신규) — 디렉터리 하나짜리 캐시.
`NNTR_HTP_WEIGHT_CACHE`가 가리키지 않으면 **꺼져 있다** (캐시 파일이 어디 쌓일지는
운영자가 정할 일이지 코드가 만들어낼 기본값이 아니다).

파일: 헤더(magic·version·K·N·wh_len·**소스 해시**) + WH + w_scale + colsum + bias.
임시 파일에 쓰고 rename한다.

**소스** 바이트를 해시하는 것이 핵심이다 — 그래야 히트가 bake뿐 아니라
`htp_qs4cx_from_packed`까지 건너뛴다. 그리고 그게 stale 파일을 "조용히 틀린 결과"가
아니라 "미스"로 만든다.

### 20.3 왜 이게 §8.4의 ION 아레나가 아닌가

문서 45 §8.4의 최종형은 **파일 → ION mmap → DSP가 직접 읽음**이고, 그쪽의 추가
이득은 *메모리*다(ARM 사본 + DSP 사본 → 한 벌, Gate 0의 1.89 GB 벽). B1a는 그
작업의 **부분집합**이다 — 파일 포맷과 "이미 구운 것을 등록하는 경로"는 아레나에도
그대로 필요하다. 지금 기기 없이 테스트 가능하고 2297 ms를 오늘 걷어낸다.
**B1a는 B1 전체가 아니다.** 22층 상주는 여전히 아레나가 필요하다.

### 20.4 측정 방법 — R0을 오염시키지 않는다

캐시가 기본 꺼짐인 이유가 이것이기도 하다. **같은 빌드로 두 번 돌린다:**

```bash
# 1) R0 게이트 — 캐시 없음. §19.3 표대로 읽는다.
NNTR_HTP_PROFILE=2 <run>

# 2) B1a — 캐시 켬. 첫 실행은 굽고 쓴다(= 1번과 같은 시간), 두 번째부터 히트.
mkdir -p /data/local/tmp/whcache
NNTR_HTP_WEIGHT_CACHE=/data/local/tmp/whcache NNTR_HTP_PROFILE=2 <run>   # 씀
NNTR_HTP_WEIGHT_CACHE=/data/local/tmp/whcache NNTR_HTP_PROFILE=2 <run>   # 읽음
```

3번째 실행에서 `convert to registry`가 **0에 가깝고** `registration total`이
2297에서 크게 떨어져야 한다. 안 떨어지면 히트를 못 한 것이고, 그때 볼 것은 캐시
디렉터리에 `wh_2048x3584_*.bin`이 생겼는지다.

**텍스트가 정상이어야 한다** — 캐시가 틀린 바이트를 먹였다면 거기서 드러난다.

### 20.5 검증

`test/htp/host/weight_cache_host_check.cpp` — 라운드트립 바이트 동일, 그리고
거절들: 소스 해시 불일치, K 불일치, N 불일치, 파일 없음, **잘린 파일**. 전부
통과. 거절 쪽이 본체다 — 이 바이트는 DSP가 곱하는 weight가 된다.

Q4_0 경로(`get_or_register`)는 캐시를 안 탄다. ponytail: 이 모델의 MoE weight는
QS4CX로 가고(문서 44 §70), Q4_0 소스의 해시 범위를 확정하려면 그 포맷을 한 번 더
파야 한다. 올리려면 `get_or_register`에 같은 (ptr, len) 쌍을 넘기면 된다.

## 21. 외부 비교 — EStream (arXiv 2609.06551), 2026-09-14

**"EStream: Fast and Memory-Efficient MoE Prefill through Expert Virtualization on
Mobile NPUs"** (Zhang, Zheng, Wu, Huang, Wu). 평가 모델에 **LFM2.5**가 들어 있다 —
OLMoE, DeepSeek-V2-Lite와 함께, 256–4096 토큰, Snapdragon 8 Elite Gen 5.

이 세션의 샌드박스에서 arxiv와 모든 미러가 차단되어 **본문 표는 못 읽었다.** 아래는
초록·검색 스니펫·llama.cpp 소스에서 확인된 것만이다. "500 TPS"와 "16×16×16"은 확인
못 했다 — 접근 가능한 텍스트엔 비율(2.25–27.57× TTFT)만 있다.

### 21.1 이 논문이 아닌 것 (그리고 §21.1의 첫 판은 틀렸다)

llama.cpp Hexagon 커널 논문은 아니다. 그러나 **"llama.cpp의 HTP는 HVX 전용이고
HMX를 행렬곱에 안 쓴다"고 쓴 첫 판은 틀렸다.** 그건 웹 요약을 옮긴 것이고 소스를
읽지 않았다. 실제로 읽으니 `matmul-ops.c`에 HMX가 141군데 나온다. 정정은 §22.

### 21.2 EStream의 네 가지와 우리 위치

| EStream | 우리 | 차이 |
|---|---|---|
| **컴파일된 expert 그래프 하나**, 호출 시 토큰·weight 주소 바인딩. 패딩 없음, CPU 폴백 없음, 전부 NPU | 레이어당 FastRPC 1회, `row_index/row_count/row_weight`로 전문가 루프 전체를 DSP에서 (§2). 라우팅은 ARM | **패딩**: 우리는 HMX 64행 타일 → blocks=43, 활용 64.5% (§16.1). 그쪽은 (HVX면) 임의 M. **라우팅**: 그쪽은 NPU |
| **Expert virtualization**: expert 풀은 UFS, 고정 크기 NPU 아레나로 그룹 단위 페이징, 로딩을 계산 뒤에 숨김. 메모리는 모델이 아니라 아레나 크기 | 문서 45 §8.4 ION 아레나 = **같은 설계**, 한 단계 안쪽(DDR→VTCM)만 구현됨(`hexkl_dma_ring`). B1a는 그 부분집합 | 그쪽은 UFS→DDR 계층이 하나 더 있고 **자동**이다. 우리 Gate 0(1.89 GB 벽)과 RSS 8.4 GB가 정확히 이게 없어서 생긴 문제 |
| **하드웨어 인지 설정 알고리즘** — UFS·NPU 파이프라인 크기를 자동으로, 로딩-계산 중첩 최대화 | 손으로 고름 | R0의 `DMA_FIRST`(GB/s)가 그 알고리즘의 입력값이다 |
| QNN 그래프 **아래**: 스칼라 제어 스레드 + DMA/HVX/HMX 동시 스케줄 + 소프트웨어 VTCM | **같은 층이다** — HexKL micro API | 여기선 차이 없음 |

### 21.3 그쪽 숫자를 읽을 때 주의

속도 비율은 "각 설정에서 **완주한** 비-오프로딩 베이스라인 중 가장 빠른 것" 대비다.
7B–16B MoE를 폰에서 돌리면 베이스라인 다수가 메모리로 죽거나 CPU다 — 27.57×의
윗단은 그 효과다. 우리 CPU 기준선 279 TPS(§18.2)도 같은 종류의 숫자다. **절대값
없이는 우리와 직접 비교가 안 된다.**

### 21.4 적용 가능한 것, 순서대로

| | 무엇 | 우리 코드에서 | 의존 |
|---|---|---|---|
| **A** | 아레나 + 그룹 페이징 | 문서 45 §8.4 그대로. B1a의 파일 포맷 위에 mmap + 오프셋 등록. 등록 0, 메모리 한 벌 | B1a |
| **B** | 로딩 완전 은닉 (drain 5.0 → ≈0) | 전문가 1개가 아니라 **그룹**을 프리페치. VTCM 8.3 MB에 전문가 2개(5.25 MiB)가 안 들어가므로 그룹은 DDR 아레나에, VTCM은 전문가 단위 스테이징 — 즉 지금 구조에서 **노출 5 ms의 원인이 대역폭인지 순서인지**가 먼저 (R0) | A, R0 |
| **C** | 라우팅을 NPU로 | router dot + top-k(ARM 2.1 ms)와 row_index 전송 제거. 레이어 간 activation 상주(문서 45 Phase D)의 전제 | — |
| **D** | 패딩 없는 expert 계산 | 두 길: (i) 64행 꽉 찬 타일은 HMX, **꼬리 블록만 HVX** — u8·i4 HVX dot 커널이 필요한데 **우리 트리에 없다**(확인). llama.cpp의 `tiled_vec_dot_q4_0_32x1/32x2`가 참고 설계. (ii) 전문가들의 꼬리를 한 타일에 묶기 — weight가 달라 HMX로는 불가 | 상한 = mm의 ~35% (43 → 28블록 상당, ≈4.8 ms) |
| E | 설정 자동화 | (DMA GB/s, HMX 속도, VTCM)으로 프리페치 깊이 결정. 나중 | R0 |

### 21.5 결론

**논문은 문서 45의 방향을 대체하지 않고 확인한다** — 상주 아레나, 전부 NPU, CPU 폴백
없음. 우리 커널 층(HMX·DMA 링·결정적 SwiGLU)은 이미 그쪽이 말하는 "QNN 아래"에 있다.
그쪽 영역에 닿으려면 A(아레나) → C(라우팅+레이어 간 상주) → D(패딩). 그리고 그쪽
비율의 절반은 베이스라인 쪽 사정이다.


## 22. llama.cpp HTP의 HMX 경로 — 소스 확인 (2026-09-14)

`ggml/src/ggml-hexagon/htp/` 를 직접 받아 읽었다. **fp16 HMX 경로가 있다.**

### 22.1 그쪽 구조

`matmul-ops.h`:

```c
#define HTP_MM_HMX_TILE_N_COLS 32
#define HTP_MM_HMX_TILE_N_ROWS 32
#define HTP_MM_HMX_TILE_SIZE   (32 * 32 * sizeof(__fp16)) // 2048 bytes
#define HTP_MM_HMX_TILE_N_ELMS 1024
#define HTP_MM_HMX_MIN_NROWS   4
#define HTP_MM_HMX_COST_W_DEQUANT 3 // 양자화 weight 로딩/디퀀트 비용 패널티
#define HTP_MM_HMX_COST_A_CONVERT 2 // activation 로딩/변환 비용 패널티
enum { ..., HTP_MM_KERNEL_HMX_2D, HTP_MM_KERNEL_HMX_F16_BATCHED,
       HTP_MM_KERNEL_HVX_F16_F16_VTCM, ... HTP_MM_KERNEL_HVX_QUANT_ROW, ... };
```

**HMX는 fp16×fp16→fp16이다.** 양자화 weight(Q4_0/Q4_1/IQ4_NL/MXFP4/Q8_0)는
DDR에서는 양자화된 채로 오고 **VTCM에서 타일 단위로 fp16으로 풀린다**
(`dequantize_tiled_weight_chunk_to_fp16_tiles`, 타입별
`dequantize_tiled_worker_loop_*`). 그 디퀀트는 HVX 작업이고 **비동기 큐로 HMX와
겹친다** (`hmx-queue.c`, `hmx_matmul_job_t job_slots[2]` 더블 버퍼,
`hmx_queue_push` / `hmx_queue_pop`).

꼬리 처리도 있다:
```c
const uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);
const uint32_t n_rows_tiled  = (n_rows / HTP_MM_HMX_TILE_N_ROWS) * HTP_MM_HMX_TILE_N_ROWS;
```

### 22.2 우리와의 대조

| | llama.cpp | 우리 |
|---|---|---|
| HMX 모드 | **fp16 × fp16 → fp16** | **u8 × i4 → i32** |
| 타일 | 32×32 fp16 (2048 B) | acc 64×32 i32, weight 타일 512 B |
| weight | 양자화된 채 DDR→VTCM, **타일마다 fp16 디퀀트** | int4 그대로 HMX에 |
| activation | f32→f16 **변환만** | **u8 양자화** (scale/zp/colsum) |
| 디퀀트 위치 | **입력측**(weight), HVX, HMX와 비동기 중첩 | **출력측**(i32→f32), 직렬 |
| 커널 선택 | **solver + 비용 모델** | 고정 |
| 꼬리 | `n_rows_tiled` / 나머지 분리 | 없음 — 64로 패딩 |

### 22.3 우리에게 옮길 수 있는 것

**① 비동기 HMX 작업 큐 — 가장 직접적이고, 이미 우리 A2다.**
`hmx-queue.c`가 동작하는 참조 설계다. 우리 `dequant 5314 + swiglu 1927 + requant
1254 = 8.5 ms`가 원리상 `mm 13722` 뒤에 숨을 수 있다. 지금은 전부 직렬이고 워커
3개가 논다(§17 A2). **이건 그냥 한다.**

**② 꼬리 분리.** `blocks=43`의 패딩 낭비(mm의 ~35%, §16.1)를 겨냥. 우리 타일이
64행이라 그쪽 32행보다 낭비가 크다 — 70행 전문가가 우리는 128슬롯(55%), 32행
타일이면 96슬롯(73%). 꼬리를 HVX로 빼려면 u8·i4 HVX dot이 필요한데 **우리 트리에
없다**(§21.4 D).

**③ 커널 선택 solver.** M이 작으면 HVX, 크면 HMX. 우리 decode(M=1)가 정확히
그 경우다 — 지금은 `M > 1` 게이트로 CPU에 떨어진다.

### 22.4 fp16 HMX로 갈아타는 건? — 아마 아니다

솔깃하다: activation u8 양자화가 사라지면 `quant 610 + requant 1254 + dequant
5314 = 7.2 ms`가 **구조적으로** 없어진다. 그리고 우리는 이미 자산이 있다 —
문서 14가 `hexkl_micro_hmx_mm_f16`을 기기에서 검증했고 prefill 43–65×,
**fp16 누산 정확도 max rel err 3e-4 @ K=1024**(누산기가 naive fp16 반올림보다
정밀하다).

그런데 산수가 반대로 간다:

- **int8 HMX는 fp16 HMX보다 빠르다** (같은 실리콘에서 보통 2–4×). 7.2 ms를 아끼려고
  `mm 13.7`을 2배로 만들면 순손실이다.
- **VTCM이 안 맞는다.** gate_up이 int4로 3.5 MiB인데 fp16이면 14 MiB. 아레나
  6.37 MB에 안 들어간다 → llama.cpp처럼 **타일 단위 디퀀트**가 강제된다. 큰 재설계.
- **V1 비트 동일성이 깨진다.** 지금 경로와 수치가 달라지므로 검증을 다시 세워야 한다.

**결정은 R0이 한다.** `HEXKL_PROBE_MM`이 int8 HMX의 실제 발행 속도를 처음 알려주고,
그 값이 있어야 "fp16으로 바꾸면 mm이 얼마나 느려지나"를 계산할 수 있다. 지금은
추측이다 — 그래서 여기서 멈춘다.

### 22.5 §21.5 수정

§21.5는 "우리 커널 층은 이미 그쪽이 말하는 QNN 아래에 있다"고 했고 그건 맞다.
다만 **llama.cpp도 거기 있다** — 첫 판이 쓴 "HVX 전용"은 틀렸다. 순서(A 아레나 →
C 라우팅 → D 패딩)는 그대로고, 여기에 **①(HMX/HVX 비동기 중첩)이 A2로 이미 있었다**는
것이 확인됐을 뿐이다.

## 23. R0 결과 (2026-09-14, 기기) — 게이트 통과, mm은 8.4, drain은 대역폭이 아니다

```
prefill: 444 tokens, 7528 ms, 58.98 TPS        ← 온도 게이트
generation: 473 tokens, 46594 ms, 10.15 TPS    ← 깊은 스로틀
registration total : 2656.5 ms  (alloc+other 706.8 — 5차의 153)
K=2048 N=2048 M>1 calls=1 rows=444 host=33.8 ms dsp=31592 transport=2164
  [quant 561 gather 3271 requant 1209 swiglu 1927 dequant 4705 acc 2750
   drain 4912 scatter 1294 stage 367 mm 8395 | rest<=2201 blocks=43]
weight DMA: 172032 KB/call, first 3584 KB took 116 us = 31.6 GB/s;
            averaged over the call 5.6 GB/s
layer calls total : 33.8 ms
```

**온도 게이트.** decode 10.15 TPS — 5차(12.4)보다 더 깊은 스로틀. prefill 7528, registration
2656(`alloc+other` 706 vs 153), transport 2164은 전부 ARM 쪽이고 **비교 불가.**
"성능이 떨어졌다"는 이 숫자들이고, 코드가 아니다. 이번 실행은 **기능 변경이 0**이다 —
R0은 프로브뿐이고 B1a는 `NNTR_HTP_WEIGHT_CACHE`가 없어 꺼져 있었다(convert 522 ms가
돌았다).

텍스트 본문: **또 못 봤다.** 6차째 미확인.

### 23.1 세 프로브의 답

| 프로브 | 답 | 의미 |
|---|---|---|
| **A4 `first`** | **116 µs = 31.6 GB/s** | **게이트 통과, 압도적으로.** 필요 12.8의 2.5배. 대역폭은 바닥이 아니다. A1–A3 살아있다 |
| **A4 평균** | **5.6 GB/s** (172 MB / 31.6 ms) | DMA 엔진이 **호출의 82%를 논다.** 그런데 drain은 4.9 ms → **파이프라인이 겹치지 않고 있다** (§23.3) |
| **A5 `mm`** | **8395 µs** | HMX 발행은 8.4 ms. 문서 44 §16의 fused 경로 실측 8.87과 일치. **"mm≤13722"는 HMX 8.4 + 이름 없는 5.3이었다.** 이제 `rest` 2201 |
| M0 | 미출력 | `NNTR_M0_PROFILE=1`이 안 켜졌다. 다음에 |

### 23.2 5차 → 6차, 기능 변경 0인데 DSP가 −4.5 ms

| 단계 | 5차 | 6차 | Δ | 성격 |
|---|---:|---:|---:|---|
| swiglu | 1927 | **1927** | **0** | HVX 계산 바운드 |
| acc | 2768 | 2750 | −18 | VTCM |
| quant / requant | 610 / 1254 | 561 / 1209 | −49 / −45 | |
| **dequant** | 5314 | 4705 | **−609** | DDR 접촉 |
| **gather** | 3623 | 3271 | **−352** | DDR 읽기 |
| scatter | 1431 | 1294 | −137 | DDR 쓰기 |
| drain | 5016 | 4912 | −104 | |
| mm+rest | 13722 | 10596 | −3126 | 5차는 잔차라 직접 비교 불가 |
| **DSP** | **36054** | **31592** | **−4462** | |

**계산 바운드(swiglu)는 정확히 같고 메모리 접촉 단계만 빨라졌다.** DSP 클록은 같았고
DDR 경합이 덜했다는 뜻이다 — ARM이 더 깊이 스로틀돼서(10.2 TPS) ARM 쪽 메모리
트래픽이 줄었을 가능성. 추측이지만 방향은 맞다.

**교훈: DSP 시간의 실행 간 변동이 ±4 ms다.** A1(−2.6), A3(−2) 같은 변경은 **단일
실행으로 검증이 안 된다.** 같은 실행 안에서 A/B를 하거나(레이어 호출을 N번 반복해
min/median), 3회 이상 돌려야 한다. §24.1.

### 23.3 drain 4.9 ms의 정체 — 파이프라인이 직렬로 무너져 있다

DMA는 31.6 GB/s를 내는데 평균 5.6이다. 전문가당 노출 drain = 4912 / 32 = **153 µs.**
겹침이 전혀 없을 때의 비용 = gate_up 116 + down 58 = 174 µs. **거의 안 겹친다.**

구조상 겹쳐야 한다: gate_up[i+1]은 전문가 i 마지막 블록의 swiglu+requant+down+scatter
(≈200 µs) 뒤에 숨을 수 있고 116 µs면 된다. 그런데 안 숨는다. 후보:

- (a) **VTCM 포트 경합**: `first` 116 µs는 VTCM이 놀 때 잰 것. HMX·HVX가 VTCM을 두드리는
  동안 DMA→VTCM 쓰기가 느려진다면 116이 300이 될 수 있다
- (b) `hexkl_dma_ring_push2d`가 dmlink 체인 상태에 따라 블록한다 (`rest` 2201의 일부?)
- (c) 두 drain(top: gate_up, mb==0: down) 중 하나가 거의 전부다

**한 버킷에 두 drain이 있다.** §13.1과 같은 실수. 쪼갠다 — §24 R1.

### 23.4 R0 이후의 바닥

DSP 31.6 = mm 8.4 + drain 4.9 + dequant 4.7 + gather 3.3 + acc 2.75 + rest 2.2
+ swiglu 1.9 + scatter 1.3 + requant 1.2 + quant 0.6 + stage 0.4

못 줄이는 것: **mm 8.4 + acc 2.75 = 11.1 ms.** 나머지 20.5는 전부 숨기거나 없앨 수 있는
성격이다. 레이어로 환산(staging 1 + dispatch 1.2 + 비-ffn 2.1 추가) ≈ **15.4 ms →
CPU 31.8 대비 2.1×.** §18.2의 1.53×보다 올라간 이유는 mm이 13.7이 아니라 8.4였기
때문이다.

## 24. 계획 v3 — R0 결과 반영

### 24.1 측정 방법부터 고친다

±4 ms 변동 위에서 −2 ms 변경을 검증하려면 **한 실행 안에서 비교**해야 한다.
`NNTR_HTP_PROFILE=3`: MoE 레이어 호출을 같은 입력으로 **5회 반복**하고 min/median을
찍는다. 온도·DDR 경합이 같은 조건이라 실행 간 변동이 사라진다. 결과에 영향 없음
(출력은 첫 호출 것). 코드 ≈30줄. **R1에 넣는다.**

### 24.2 우선순위 — ms × 확신도

| # | 항목 | 지금 | 목표 | 근거 | 확신 |
|---|---|---:|---:|---|---|
| **R1-a** | **drain 쪼개기** (top / mb==0) + `push2d` 시간 | 4912 | — | §23.3 (a)/(b)/(c)를 가른다 | 측정 |
| **R1-b** | **A2: dequant·swiglu·requant를 HMX 뒤에** | 7841 | ≈1500 | llama.cpp `hmx-queue.c` 패턴. 워커 3개가 HMX 동안 논다. mm 8.4 ms가 숨길 공간 | ●●○ |
| **R1-c** | **A1: gather 제거** | 3271 | ≈1000 | 전문가 순서로 바로 팩 | ●●● 비트동일 |
| R2 | drain 수정 (R1-a 결과대로) | 4912 | ≈500 | (a)면 push 시점을 더 앞으로 + f32 버퍼 축소로 down 이중버퍼; (b)면 링 수정 | R1-a 뒤 |
| R2 | `rest` 2201 프로브 | 2201 | — | push_weight·setup·free 중 무엇인지 | 측정 |
| R3 | A3 융합 dequant→SwiGLU | — | — | A2 뒤 | |
| 병렬 | **B1a 측정** — 캐시 켜고 2회 | reg 2656 | ≈300 | 코드 있음, 미측정 | |

**R1 = 24.1 + R1-a + R1-b + R1-c.** 기능 변경은 R1-b, R1-c 둘. 24.1의 반복 측정이
있어서 둘의 효과를 같은 실행에서 읽을 수 있다.

예상: DSP 31.6 − 6.3 (A2) − 2.3 (A1) ≈ **23 ms**, drain 수정 후 ≈ **18.6**, 레이어 ≈ 23
→ CPU 31.8 대비 **1.4×.** 여기까지가 커널 안이고, 그 위는 문서 45다.

### 24.3 반드시

**텍스트 본문.** 6회째 못 봤다. 다음 실행은 생성 앞 5줄을 같이 붙인다. B1a 캐시
실행에서는 필수 — 틀린 바이트를 먹였는지 거기서만 드러난다.

## 25. R1 구현 (2026-09-14, 코드 완료·미측정)

§24의 계획 네 항목. 기능 변경은 둘(R1-b, R1-c)이고 **둘 다 비트 동일**이라 호스트
체크가 `mismatches=0 of 2368`로 확인한다. 나머지 둘은 측정 전용.

### 25.1 24.1 — 같은 실행 안에서 5회 (`NNTR_HTP_PROFILE=3`)

§23.2에서 **기능 변경 0인데 DSP가 4.5 ms 움직였다.** A1(−2.3 예상)·A3(−2)가 그
잡음에 묻힌다. `invokeMoeLayer`가 같은 입력으로 5회 부르고 **가장 빠른 회차의
`elapsed`와 stage 배열을 통째로** 채택한다. 출력은 입력이 같으므로 안 바뀐다.
level 2 이하에서는 1회 — 기존 동작 그대로.

### 25.2 R1-a — drain 쪼개기 + push 계측 (측정 전용)

`HEXKL_PROBE_DRAIN`이 전문가당 두 번의 drain을 다 담고 있었다. gate_up은 전문가
하나 앞서 push되고 down은 gate_up 행렬곱 하나 앞서 push되므로 **숨을 수 있는 양이
다르다.** 쪼갠다: `DRAIN`(gate_up) / `DRAIN_DN`(down). 그리고 `push2d` 자체를
`HEXKL_PROBE_PUSH`로 잰다 — dmlink 체인에 얹는 호출이 블록하는지가 §23.3의 (b)다.

프로파일: `drain 4912` → **`drain A+B push C`**.

### 25.3 R1-b — 누산 타일을 모아 풀로 dequant

바뀐 것: n-타일 루프가 `L.acc_tiles`개씩 끊어 돌고, 각 `acc_read`는 자기 슬롯에
쌓인다. 배치가 끝나면 `hvx_dequant_acc_tiles_to_f32`가 **타일 단위로 워커풀에
쪼개** 한 번에 처리한다.

| | 전 | 후 |
|---|---|---|
| dequant 실행 | 타일마다 1회, **호출 스레드만** | 배치마다 1회, **풀 전체** |
| 블록당 풀 fork | 0 (dequant는 풀을 안 썼다) | 2 |
| 블록당 dequant 호출 | 176 | 2 |

VTCM: `result_off`가 8 KB → `acc_tiles × 8 KB`. LFM2에서 112타일 = 896 KiB,
아레나 **6.37 → 7.23 MiB**. 안 맞으면 `fits`가 자동으로 줄이고 1이면 예전 동작과
같다 — 거절하지 않는다.

gate/up 경계를 배치가 가로지를 수 있어 목적지를 `dst_a`/`dst_b` + `split`로 받는다.
down은 목적지가 하나라 `split = N_out`, `dst_b = NULL`.

**ponytail**: 이건 *풀 활용*이지 *HMX/HVX 중첩*이 아니다. 진짜 중첩(워커가 타일 nt를
dequant하는 동안 메인이 nt+1을 발행)은 `hvx_worker_pool_run`이 fork-join이라 비동기
dispatch가 필요하고, 검증된 프리미티브를 건드리는 별개 작업이다. llama.cpp의
`hmx-queue.c`가 그 참조 설계(§22.3 ①). 상한은 dequant 4705 → mm 8395 뒤에 완전히
숨는 것이고, 이번 변경의 상한은 4705/워커수다.

### 25.4 R1-c — gather를 inner 타일로 쪼갬

3.6 MB를 3271 µs에 옮겼다 = **1.1 GB/s.** 순수 복사치고 말이 안 된다. 원인은
볼륨이 아니라 패턴이었다: 행별로 쪼개니 워커마다 2048바이트 간격으로 32바이트씩
쓴다 — 블록의 128 KB가 **4096개의 흩어진 store**로 나간다.

inner 타일로 쪼개면 타일당 목적지가 **연속 2048바이트 한 덩어리**이고, 유닛 수가
항상 `kt_n`(=64)이라 **한 행짜리 꼬리 블록에서도 워커 4개가 다 일한다** (전에는
`n_rows`가 유닛이라 워커 셋이 놀았다).

`memset`도 고쳤다: 전엔 128 KB 전부를 호출 스레드가 밀었는데, 덮어쓸 행은 밀 필요가
없다. 이제 `n_rows` 이후만 민다.

### 25.5 슬롯

`MOE_N_STAGES` 15 → **17** (DRAIN_DN, PUSH). skel 재빌드 + **ARM 스텁 재생성**
둘 다 필요하다(§14.2의 0번).

### 25.6 측정

```bash
bash nntrainer/tensor/htp_backend/generate_stub.sh
HEXKL_SDK_VER=6.4.0.2 ./test/htp/build.sh
./build_android.sh && ./Applications/CausalLM/install_android.sh
adb push test/htp/build/libnntr_hvx_skel.so "$DEVICE_DIR/"
NNTR_HTP_PROFILE=3 NNTR_M0_PROFILE=1 <run>
```

볼 것:

| | 기대 | 아니면 |
|---|---|---|
| `dequant` | 4705 → **≈1500** | 풀이 VTCM 포트에서 막힌다 — 중첩(§25.3 ponytail)이 유일한 길 |
| `gather` | 3271 → **≈1200** | 병목이 destination store가 아니라 scattered source read다 |
| `drain A+B` | 합 4912의 분해 | A가 대부분이면 gate_up prefetch 거리가 부족, B면 down |
| `push` | ≈0 | 크면 §23.3 (b) — `push2d`가 블록한다 |
| 반복 5회 | stage가 안정 | `NNTR_HTP_PROFILE=3`이 안 먹었거나 첫 회차가 항상 느리다 |

**텍스트 본문 앞 5줄** — 7회째 미확인이다. R1-b/R1-c가 비트 동일이라 호스트에서는
증명됐지만 기기에서는 아니다.

## 26. R1 결과 (2026-09-14, 기기) — 33.8 → **29.0 ms**, 그리고 가설 하나가 틀렸다

```
prefill 118.5 TPS / decode 33.86 TPS   ← 식은 기기, 이번엔 비교 가능
K=2048 N=2048 M>1 calls=1 rows=444 host=29.0 ms dsp=26639 transport=2359
  [quant 590 gather 3156 requant 1254 swiglu 1910 dequant 940 acc 2799
   drain 3513+1748 push 13 scatter 1387 stage 362 mm 8051
   | rest<=916 (3.2%) blocks=43]
weight DMA: first 3584 KB took 136 us = 27.0 GB/s; 호출 평균 6.6 GB/s
```

**텍스트 정상 — 7회 만에 처음 본문을 확인했다.** 3문장 요약이 원문 내용을 다 담고
있다. R1-b/R1-c의 비트 동일이 기기에서도 성립한다.

### 26.1 단계별

| 단계 | 6차 | 7차 | Δ | DSP 비중 |
|---|---:|---:|---:|---:|
| mm | 8395 | 8051 | −344 | **30.2%** |
| drain | 4912 | **3513+1748** | +349 | **19.7%** |
| gather | 3271 | **3156** | **−115** | 11.8% |
| acc | 2750 | 2799 | +49 | 10.5% |
| swiglu | 1927 | 1910 | −17 | 7.2% |
| scatter | 1294 | 1387 | +93 | 5.2% |
| requant | 1209 | 1254 | +45 | 4.7% |
| **dequant** | 4705 | **940** | **−3765** | 3.5% |
| rest | 2201 | **916** | −1285 | 3.4% |
| quant / stage / push | 561/367/0 | 590/362/**13** | | 3.6% |
| **DSP** | 31592 | **26639** | **−4953** | |

### 26.2 R1-b는 예측을 넘겼다

dequant **4705 → 940, 5.0배.** 예상은 ≈1500이었다. 타일마다 호출 스레드 혼자 돌던
것을 배치로 묶어 풀에 쪼갠 것뿐인데, 워커 수(4) 이상이 나왔다 — 호출 오버헤드가
계산만큼 컸다는 뜻이다(블록당 176회 → 2회). `rest`도 2201 → 916으로 떨어졌는데
그 안에 있던 루프 오버헤드가 같이 사라진 것으로 보인다.

### 26.3 R1-c는 **실패했다** — 내 가설이 틀렸다

gather **3271 → 3156, −3.5%.** §25.4에서 "행별로 쪼개면 흩어진 store가 문제"라고
썼고 목적지를 연속 2048바이트로 바꿨는데 **거의 안 움직였다.**

**따라서 병목은 destination store가 아니라 scattered source read다.** 고정 kt에
대해 64개 행을 읽으면 주소가 `(t/64)*kt_n*2048`씩 튄다 — 3.6 MB 버퍼 전체에 흩어진
**32바이트 읽기 64개**다. 목적지를 아무리 정리해도 읽기 쪽이 그대로면 안 변한다.

§11.5의 규칙이 또 맞았다: 메커니즘을 짐작으로 고르면 절반은 틀린다. 이번엔 고친
코드가 더 나쁘지도 않고 유닛 분배는 실제로 개선됐으므로 되돌리지 않는다.

### 26.4 push는 13 µs — §23.3 (b) 기각

`push2d`는 블록하지 않는다. 남은 후보는 **(a) VTCM 포트 경합**이다. 산수가 맞는다:

```
전문가당 drain: gate_up 110 us, down 55 us
3584 KB @ 27.0 GB/s(유휴 실측) = 136 us
gate_up drain 110 = 전송 시간의 83%  → prefetch가 거의 아무것도 못 숨긴다
부하 중 유효 대역폭 ≈ 3584KB / (136+110) = 15 GB/s  (유휴의 55%)
```

push 시점 뒤에 남는 일은 그 블록의 swiglu+requant+down mm+scatter ≈ 173 µs로,
136 µs 전송을 **덮고도 남아야 한다.** 안 덮인다는 건 HMX·HVX가 VTCM을 두드리는
동안 DMA가 절반 속도로 떨어진다는 뜻이다.

### 26.5 남은 바닥

```
mm 8051 + acc 2799 = 10.9 ms   (HMX 발행 + 벤더 acc_read)
```

나머지 15.8 ms가 대상이다. 레이어 환산(staging 1.1 + dispatch 1.2 + 비-ffn 2.1)
≈ **15.3 ms → CPU 31.8 대비 2.1×**.

## 27. 계획 v4 — 남은 것

### 27.1 순위 (DSP 26.6 ms 기준)

| # | 대상 | 지금 | 목표 | 방법 | 확신 |
|---|---|---:|---:|---|---|
| **1** | **drain** | 5261 | ≈1000 | **WH bake 순서를 nt-major로.** 지금 타일 인덱스가 `kt*n_col+nt`라 nt 하나를 계산하려면 버퍼 전체가 필요하다. `nt*k_tiles+kt`로 바꾸면 nt=0의 64타일(32 KB)만 도착해도 시작할 수 있고, **전송의 1/112 뒤에 계산이 붙는다.** bake 순서만 바뀌고 산술은 그대로 | ●●○ |
| **2** | **gather** | 3156 | **0** | **전문가 순서로 바로 팩.** quant/pack이 2752 슬롯에 직접 쓰면 gather 자체가 없어진다. quant 590 → ≈2400(행 4배)이지만 gather 3156이 사라져 **순 −1.3 ms**, 패스 하나가 통째로 빠진다. §26.3이 "source read가 문제"라고 확정했으므로 이게 유일한 구조적 해법 | ●●● |
| **3** | **A3 융합** dequant→SwiGLU | 940+1910 | ≈1200 | gate/up f32를 VTCM에 쓰고 다시 읽는 왕복(블록당 1.8 MB)을 없앤다. dequant가 940으로 줄어 이득이 작아졌지만 swiglu 1910은 그대로 | ●●○ |
| 4 | scatter+requant | 2641 | ≈1800 | 둘 다 이미 풀. 남은 건 f32 왕복 | ●○○ |
| 5 | mm | 8051 | ≈6000 | blocks 43 → 32은 라우팅이 정한다. prefill 청크를 키우는 것 외에 커널 안에 레버 없음 | ●○○ |
| 6 | acc 2799 | — | — | 벤더 `acc_read`. 타일당 1회가 최소 | — |
| — | **B1a registration** | **1825** | ≈300 | **코드 있음, 아직 안 쟀다.** 캐시 켜고 2회 돌리면 끝 | 측정만 |
| — | transport 2359 | | | ARM FastRPC 마샬링. 문서 45 Phase D | |

### 27.2 R2 = 1 + 2

둘 다 기능 변경이고 서로 다른 버킷(drain vs gather)이라 한 실행에서 따로 읽힌다.
예상 DSP 26.6 − 4.3 − 1.3 ≈ **21 ms**, 레이어 ≈ 24.5 → CPU 대비 **1.3×**.

nt-major bake는 **B1a 캐시 파일을 무효화**한다 — `HtpWeightCacheHeader.version`을
올려서 옛 파일이 자동으로 미스가 되게 한다(그러라고 넣은 필드다).

### 27.3 그 다음은 커널 밖

§26.5의 바닥 10.9 ms는 HMX와 벤더 코드다. 거기 닿으면 **문서 45**(conv 18 ·
attention 6 · 레이어 간 상주)만 남는다. prefill TPS로 보면 §18.2 그대로 —
MoE FFN이 0이어도 1.6×가 천장이다.

## 28. R2 구현 (2026-09-14, 코드 완료·미측정)

§27.2의 둘. 기능 변경 둘 다 **비트 동일** — 호스트 체크 `mismatches=0 of 2368`.

### 28.1 R2-2 — gather 제거 (3156 → 0)

§26.3이 "병목은 scattered **source read**"라고 확정했으므로 gather를 튜닝하지 않고
**없앴다.**

`hvx_quant_pack_u8_ah_mapped`가 생겼다: 목적지 행 d가 `row_map[d]` 소스 행에서
온다. pack worker는 4행을 한 벡터로 묶는데 `vin0..3`이 독립 포인터라 map은
간접참조 4번이 전부다. scale/zp는 **목적지 행 기준**으로 받는다 — 전문가 4개에
뽑힌 토큰은 슬롯 4개를 차지하고 자기 양자화 파라미터를 그대로 반복한다.

커널은 `slot_row`/`slot_scale`/`slot_zp`(2752 슬롯)를 만들고 pack을 한 번 부른다.
블록의 64행이 이미 연속이므로 gather는 `hvx_copy_ah_block`(평평한 128 KB 풀 복사)로
바뀌었다. 패딩 슬롯은 행 0을 가리키고, 그 누산 결과는 `m_blk`이 막아 절대 안 읽힌다.

대가: pack이 444행이 아니라 1776행을 읽는다 (quant 590 → ≈2400 예상). gather 3156이
사라지므로 **순 −1.3 ms**이고 패스가 하나 없어진다.

### 28.2 R2-1 — gate_up을 청크로 스트리밍 (drain 5261 → ?)

**bake 순서는 안 바꿨다.** WH 타일이 `kt*n_col + nt`이므로 n-타일 열 한 묶음은
`cn*512`바이트가 `n_col*512` 간격으로 `k_tiles`번 반복되는 것 — **정확히 2D 전송
하나**다. 목적지 레이아웃이 같아서 행렬곱 인덱싱도 그대로다. 비-MoE 경로 5곳을
건드리지 않는다.

링에 `hexkl_dma_ring_next_idx()` / `hexkl_dma_ring_wait(idx)`를 노출했다 —
`wait_idx`는 이미 static으로 있었다. `drain()`은 **pending 전부**를 기다리므로
파이프라인된 호출자가 일부러 띄워둔 prefetch까지 기다린다. 청크 대기는 그걸 표현할
수 없어서 필요했다.

전문가 루프:

```
gate_up[0]을 4청크로 push (idx 기록)
for i:
    down[i] push (1청크)
    for mb:
        for c in 4청크:
            if mb==0: wait(gu_idx[c])   ← 첫 블록만, 이후는 상주
            matmul + acc_read + 배치 dequant
        if mb==0: wait(dn_idx)
        if last_block: gate_up[i+1]을 4청크로 push
```

`L.acc_tiles`가 **32로 캡**되어 청크 크기 겸 dequant 배치 크기가 됐다. 결과:
gate_up 4청크, down 2청크, 레이어당 링 디스크립터 **194개 < 256**. acc 스테이징이
896 → 262 KB로 줄어 아레나 **7.23 → 6.61 MiB**.

예상: 첫 청크 대기 136/4 = 34 µs, 나머지는 계산 뒤에 숨는다. 전문가당 110 → ≈34,
**−2.4 ms**.

### 28.3 측정

`MOE_N_STAGES`는 **17 그대로** — 새 슬롯 없음. 그래도 IDL은 안 바뀌었지만
**스텁 재생성은 습관으로 돌리는 게 안전하다**(§14.2).

| 볼 것 | 기대 |
|---|---|
| `gather` | 3156 → **≈0** (평평한 복사만 남음) |
| `quant` | 590 → **≈2400** (행 4배) |
| `drain A+B` | 3513+1748 → **≈1100+900** |
| `push` | 13 → ≈50 (디스크립터 194개) |
| **layer** | 29.0 → **≈25** |


## 29. R2 결과 (2026-09-14, 기기) — 29.0 → **27.0 ms**, 그리고 **우리는 prefill의 0.8%를 깎고 있다**

```
prefill 128.845 TPS (3446 ms) / decode 35.18 TPS   ← 식은 기기, 텍스트 정상(2회 연속)
registration total : 1662.4 ms
host=27.0 ms dsp=24700 transport=2283
  [quant 1913 gather 1843 requant 1234 swiglu 1902 dequant 1398 acc 2784
   drain 1039+1602 push 23 scatter 1414 stage 447 mm 8082
   | rest<=1019 blocks=43]
```

### 29.1 먼저: prefill TPS가 왜 안 움직이나

| prefill 3446 ms | ms | % |
|---|---:|---:|
| **registration** | **1662.4** | **48.2%** |
| 나머지 ARM (21개 CPU MoE + conv 18 + attn 6 + norm + lm_head) | 1755.5 | 50.9% |
| **HTP 레이어 호출 (레이어 2 하나)** | **27.0** | **0.8%** |

**레이어 호출을 0으로 만들어도 prefill은 +0.8%다.** §18.2가 Amdahl로 말한 것이
실측으로 확인됐고, 실제로는 그보다 더 나쁘다 — 22개 MoE 레이어 중 **1개만** HTP에
있기 때문이다.

TPS를 움직이는 것은 둘뿐이다:

| | 지금 | 후 | prefill TPS |
|---|---:|---:|---|
| **B1a 등록 캐시** (코드 있음, **여전히 미측정**) | 1662 | ≈300 | **128.8 → 213 (+65%)** |
| 22개 MoE 전부 HTP | 21개 CPU | | 1755 중 ~670이 MoE |

**커널 최적화와 prefill TPS는 이 시점에서 목표가 갈라졌다.** 커널 작업은 "레이어
하나를 CPU보다 빠르게"에 유효하고(문서 45 Phase A의 전제), prefill TPS에는 B1a와
레이어 확장이 유효하다.

### 29.2 R2는 반쪽 성공

| | 7차 | 8차 | Δ | |
|---|---:|---:|---:|---|
| **drain gate_up** | 3513 | **1039** | **−2474** | ✅ 청크 스트리밍 |
| gather | 3156 | 1843 | −1313 | ⚠️ 0이 목표 |
| quant | 590 | 1913 | +1323 | 예상대로 |
| dequant | 940 | 1398 | +458 | acc 배치 112→32의 대가 |
| drain down | 1748 | 1602 | −146 | ⚠️ down은 청크를 안 했었다 |
| **DSP** | 26639 | **24700** | −1939 | |

**`quant + gather`가 3746 → 3756, 무승부다.** gather를 없앤 게 아니라 128 KB 평평한
복사로 바꿨는데, 5.5 MB를 1843 µs = **3.1 GB/s**로 옮기고 있었다 — 목적지가 VTCM인데
코어로 쓰고 있었다.

그리고 프로파일의 `first 3584 KB ... 114.7 GB/s`는 **내 버그였다.** 첫 대기가 이제
전체가 아니라 첫 청크(1024 KB)인데 라벨이 하드코딩이었다. 실제는 ~33 GB/s.

### 29.3 R2b (코드 완료·미측정)

1. **블록 복사를 DMA로** — `hexkl_dma_ring_push2d` + 대상 인덱스 대기. 1843 → ≈200 기대
2. **down도 청크로** — gate_up과 같은 이유. 1602 → ≈500 기대
3. **`DMA_FIRST_KB` 슬롯 추가** — 대기가 덮은 바이트 수를 같이 보고해서 GB/s가
   더 이상 거짓말하지 않게. `MOE_N_STAGES` 17 → **18**

예상 DSP 24700 − 1600 − 1100 ≈ **22 ms**, 레이어 ≈ 24.4.

### 29.4 다음에 잴 것 — **B1a를 먼저**

```bash
mkdir -p /data/local/tmp/whcache
NNTR_HTP_WEIGHT_CACHE=/data/local/tmp/whcache NNTR_HTP_PROFILE=2 <run>   # 굽고 씀
NNTR_HTP_WEIGHT_CACHE=/data/local/tmp/whcache NNTR_HTP_PROFILE=2 <run>   # 히트
```

`registration total`이 1662 → 300 근처로 떨어져야 하고, 그게 **prefill TPS를
128.8 → 213으로 올리는 유일한 이미-구현된 레버**다. 텍스트도 봐야 한다 — 캐시가
틀린 바이트를 먹였으면 거기서만 드러난다.

## 30. B1a 측정 (2026-09-14, 기기) — prefill **160.8 TPS**, 하지만 레이어가 회귀했다

```
1회차(캐시 씀): prefill 98.9 TPS,  registration 2234.7,  layer 32.9
2회차(히트)   : prefill 160.8 TPS, registration  974.8,  layer 47.9
직전 베이스라인: prefill 128.8 TPS, registration 1662.4,  layer 27.0
```

### 30.1 캐시는 동작한다

| | 베이스라인 | 히트 | |
|---|---:|---:|---|
| convert to registry | 510.9 | **0.0** | ARM 변환 건너뜀 |
| register FastRPC | 1134.1 | **392.1** | DSP bake 건너뜀 (19.55 → 6.13 ms/weight) |
| alloc + other | 17.5 | **582.7** | **캐시 파일 178 MB를 0.31 GB/s로 fread** |
| registration total | 1662.4 | **974.8** | |

`974.8 = 392 + 583`으로 완전히 설명된다. **prefill 160.8 TPS는 최고 기록**(+25%)이지만
예상 213에는 못 미쳤고, 이유가 둘이다.

### 30.2 무엇을 캐싱하는가 (한 번 더 못 박는다)

**weight다. activation이 아니다.** 두 텐서가 다른 레이아웃, 다른 경로를 탄다:

| | **WH** | **AH** |
|---|---|---|
| 대상 | weight (`w_i4_rm`) | activation (`act_f32`) |
| 변환 | `hexkl_micro_hmx_rm_to_wh_i4` | `hvx_quant_rows_u8_params` + `_pack_u8_ah_mapped` |
| 언제 | `weight_register` — 모델 로드 1회 | `layer_run` — **호출마다** |
| 캐시 | ✅ | ❌ |

activation을 캐싱하면 입력이 바뀔 때 틀린 답이 나온다. 그래서 안 한다 — `quant 1816`과
`gather 1697`이 매 호출 지불하는 그 비용이다.

### 30.3 회귀 둘 — 원인 확정과 미확정

**① `rpcmem/ION buffer : 0/64`** — 캐시 경로에서 `HtpRpcBuffer` 대신 `std::vector`를
썼다. 3.5 MB 페이로드가 일반 힙에서 넘어가 FastRPC가 호출마다 pin+map했다. **내 누락.**

**② `layer calls total` 27.0 → 47.9, 전부 `rest`(1019 → 23888)에** — 커널 코드는
같고 이름 붙은 단계도 전부 같다(mm 8086, quant 1816, acc 2788...). **DSP에 이름 없는
24 ms가 생겼다.** `rest`에 있는 건 `layer_run` 자신의 malloc/free(호출당 12.8 MB)와
셋업뿐이다. 캐시 경로가 DSP 힙을 다르게 남겼을 가능성이 높지만 **추측하지 않는다.**

### 30.4 R3 (코드 완료·미측정)

1. **캐시를 ION 버퍼로 직접 읽는다** — `load`가 벡터가 아니라 caller의 버퍼를 받는다.
   복사 두 번(fread→vector, FastRPC pin+copy)이 한 번(read→ION, zero-copy)이 된다.
   호스트 체크에 **short buffer 거절**을 추가했다: 이 용량 검사가 잘못 잡은 rpcmem
   할당과 오버런 사이에 서 있는 유일한 것이다.
2. **`HEXKL_PROBE_ALLOC`** — `layer_run`의 malloc/free를 잰다. ②가 여기 있으면
   숫자가 말해준다. `MOE_N_STAGES` 18 → **19**.

## 31. 남은 최적화 — 전체 목록

### 31.1 prefill TPS를 움직이는 것 (prefill 2761 ms 기준)

| # | 항목 | 지금 | 목표 | 상태 |
|---|---|---:|---:|---|
| **P1** | **`rest` 24 ms 회귀** | layer 47.9 | 27.0 복귀 | **R3-2 프로브 대기** |
| **P2** | 캐시 파일 읽기 | 583 ms | ≈150 | **R3-1 완료, 미측정** |
| **P3** | **22개 MoE 레이어 전부 HTP** | 21개 CPU | | `moe_htp_layers` 확장. registration이 싸진 지금 처음 가능 |
| **P4** | FastRPC 등록 | 392 ms | ≈0 | 문서 45 §8.4 ION 아레나 — 파일을 mmap해 DSP가 직접 읽고, 등록은 핸들만 |
| P5 | 나머지 ARM 1755 ms | | | conv 18 + attn 6 → 문서 45 Phase B/C |

### 31.2 레이어 시간을 줄이는 것 (DSP 24.7 ms 기준, 회귀 제외)

| # | 항목 | 지금 | 목표 | 근거 |
|---|---|---:|---:|---|
| **L1** | 블록 복사 DMA | 1697 | ≈200 | R2b 완료·미측정 |
| **L2** | down 청크 | 1659 | ≈500 | R2b 완료·미측정 |
| **L3** | A3 dequant→SwiGLU 융합 | 1358+1906 | ≈1800 | 블록당 1.8 MB VTCM 왕복 제거 |
| **L4** | quant 1816 | | ≈1200 | 슬롯 2752개를 f32에서 읽는다. 소스 행 재사용이 캐시에 안 남는 듯 |
| L5 | scatter+requant 2556 | | ≈1800 | 이미 풀. f32 왕복만 남음 |
| **바닥** | **mm 8086 + acc 2788 = 10.9 ms** | | — | HMX + 벤더 코드 |

### 31.3 못 고치는 것

- **blocks=43** — 라우팅이 정한다. prefill 청크를 키우는 것 외에 커널 레버 없음
- **decode** — 문서 43 §7에서 대역폭 바운드로 닫혔다. 토큰당 484 MB를 읽는 구조를
  바꿔야 하고 커널 밖이다

## 32. B1a 폐기 → 변환된 모델 + ION 아레나 (2026-09-14, 설계)

### 32.1 왜 폐기하나

B1a는 "런타임 사이드 캐시"였다: 첫 실행이 굽고 weight마다 파일을 쓰고, 다음 실행이
해시로 찾아 읽는다. 1개 레이어에서는 동작했다(prefill 128.8 → 160.8 TPS). 그러나:

| 문제 | |
|---|---|
| **디스크 중복** | 레이어당 178 MB. **22개면 +3.9 GB**, 모델 4.3 GB 옆에. 안 확장된다 |
| 여전히 복사 | 파일 → ION(583 ms) → FastRPC → DSP malloc+memcpy. 등록 975 ms |
| 첫 실행 | 전액 지불 + 178 MB 쓰기 |
| 부가 복잡도 | weight마다 해시·버전·atomic write·용량 검사 |

지적이 맞았다: **weight는 상수이므로 "캐시"가 아니라 "변환된 모델"이어야 한다.**
양자화처럼 한 번 만드는 산출물이고, 로드는 읽어서 매핑하는 것뿐이어야 한다.

### 32.2 왜 PC에서 못 굽나

`hexkl_micro_hmx_rm_to_wh_i4`는 `libhexkl_micro.a`(hexagon_toolv19_v79) 안에만 있다.
WH 타일 레이아웃을 아는 코드가 ARM/x86에 없다. 따라서 변환은 **기기에서 1회** 돈다 —
`weight_bake_export`가 정확히 그 도구다. 호스트 구현(레이아웃 역산)은 가능하지만 독점
레이아웃이고 SDK 버전마다 깨질 수 있어 뒤로 둔다.

### 32.3 설계

```
지금:  [.bin QS4CX] → ARM 변환 → FastRPC 7.3MB × 64 → DSP bake → DSP malloc 복사
                     547 ms       (bake 포함) 1134 ms          → 등록 1662 ms

목표:  [.bin] + [model.htp: WH·scale·colsum·bias 전부, 인덱스 포함]
              → ARM: rpcmem_alloc(≤2GB 조각) + 파일 읽기 + rpcmem_to_fd
              → DSP: HAP_mmap(fd) 한 번 → 아레나 VA
              → weight마다 IDL 호출 하나(K, N, 조각, 오프셋 4개) — 페이로드 0
              → 슬롯이 아레나 안을 가리킨다. malloc 없음, memcpy 없음, bake 없음
              → 커널 변경 없음: hexkl_dma_ring은 포인터를 받을 뿐이다
```

`model.htp`는 **캐시가 아니다**: 해시 없음, 첫 실행 페널티 없음, weight별 파일 없음.
헤더에 모델 식별자 + 포맷 버전만 둔다 — 다른 체크포인트의 사이드카를 거절하는 건
신뢰 경계의 입력 검증이지 캐시 로직이 아니다.

`.bin` 자체에 넣는 것(사용자 제안의 문자 그대로)은 `nntr_quantize` 출력 포맷과
로더를 바꾸는 일이라 **사이드카를 먼저** 하고 그 다음 접는다. 런타임 이득은 같다.

### 32.4 미증명 — Gate 0c

**"DSP가 ARM ION 버퍼를 영구 매핑하고 거기서 DMA로 VTCM에 끌어온다"**는 이 트리에서
한 번도 안 해봤다. `rpcmem_to_fd`도 `HAP_mmap`도 없다(§31 확인). Gate 0b는 *할당
크기*만 증명했다. llama.cpp가 `HAP_mmap`으로 같은 하드웨어에서 하고 있으므로 될
가능성이 높지만, **될 것 같다는 이유로 설계 전체를 얹지 않는다.**

프로브: `rpcmem_alloc` → `rpcmem_to_fd` → IDL `arena_map(fd, bytes)` → DSP `HAP_mmap`
→ 3.5 MB를 VTCM으로 DMA → 시간 → 27–33 GB/s와 비교. `HAP_munmap`도. 모델 경로
변경 없음, `mem_probe_dsp_heap`과 같은 종류다.

### 32.5 순서

| # | 항목 | 내용 | 없애는 것 | 의존 |
|---|---|---|---|---|
| **A0** | **Gate 0c** | **통과 (§32.11)** | — | — |
| **A1** | 변환 모드 | `nntrainer_causallm --convert-htp <dir>`: 로드 → MoE weight마다 register(bake) → `weight_bake_export` → `model.htp`에 append, 인덱스 기록. 기기에서 1회 | — | — |
| **A2** | 아레나 로드 | `model.htp` 있으면: ION 조각 할당(≤2 GB) → 읽기 → `arena_map` → weight마다 `weight_register_arena(K,N,chunk,off×4)`. 슬롯은 아레나 포인터 | 등록 1662 → **≈0.2 s** (파일 읽기 178 MB) | A0 |
| **A3** | 삭제 | `HtpWeightCache`, `register_baked`, `NNTR_HTP_WEIGHT_CACHE`, 그 호스트 체크. `weight_bake_export`는 A1이 쓴다 | 코드 −400줄 | A2 |
| **A4** | **22개 전부 HTP** | `moe_htp_layers` 확장. 아레나 3.9 GB, ION 조각 2개(Gate 0b: 총 6 GB OK) | ARM MoE 21개분 ≈670 ms | A2 |
| A5 | ARM 사본 해제 | 아레나에 있는 weight의 QS4CX Tensor 사본을 안 만든다 — 8.4 GB RSS의 절반 | 메모리 −3.9 GB | A4, 로더 작업 |

**A4 뒤 prefill 예상**: 2761 − 975(등록) + 200 − 670(CPU MoE → HTP 22×27) + 594 ≈
**1900 ms → 234 TPS**, CPU 279 대비 아직 0.84×. §18.2 그대로 — 나머지 1755 ms의
절반(conv·attention)이 문서 45 Phase B/C다.

### 32.6 커널 항목 — 그대로, 측정 대기

| | 항목 | 지금 | 목표 | 상태 |
|---|---|---:|---:|---|
| L1 | 블록 복사 DMA | 1697 | ≈200 | 코드 완료 |
| L2 | down 청크 | 1659 | ≈500 | 코드 완료 |
| — | `alloc` 프로브 (24 ms 회귀) | rest 23888 | — | 코드 완료. **아레나로 가면 weight가 DSP 힙을 안 쓰므로 힙 단편화 가설은 구조적으로 사라진다** — 그래도 잰다 |
| L3 | dequant→SwiGLU 융합 | 3264 | ≈1800 | |
| L4 | quant | 1816 | ≈1200 | |
| L5 | scatter+requant | 2556 | ≈1800 | |
| 바닥 | mm 8086 + acc 2788 | 10.9 ms | — | HMX + 벤더 |

대기 중인 빌드는 이걸 잰다. **`NNTR_HTP_WEIGHT_CACHE` 없이** 돌린다 — 캐시 칼럼은
이제 의미가 없다.

### 32.7 A0 / Gate 0c — 구현 완료, 측정 대기

`arena_probe(fd, bytes, dma_bytes, res[6])`. 호스트가 `rpcmem_alloc` 한 3.5 MB(=
gate_up 하나치)에 패턴을 쓰고 `rpcmem_to_fd`로 fd를 얻어 넘긴다. DSP는 `HAP_mmap`
으로 한 번 매핑하고, 1 MB를 VTCM으로 DMA 하고, 체크섬을 내고, `HAP_munmap` 한다.

`res = [mapped, map_us, dma_us, dma_bytes, checksum, unmap_ok]`

**방어 셋:**

1. **`HAP_mem.h`가 이 트리 첫 사용**이라 `__has_include`로 감쌌다. 없는 SDK면
   `AEE_EUNSUPPORTED`를 돌려준다 — **컴파일이 안 되는 프로브는 "아니오"라고 답하는
   프로브보다 나쁘다.**
2. **`HAP_mmap` / `HAP_munmap`을 쓴다** (`HAP_mmap_get`/`_put`이 아니라). 전자가 더
   오래됐고 널리 있다. 링크가 이 심볼에서 깨지면 후자 쌍으로 바꾸면 되고 나머지는
   그대로 — 코드 주석에 적어뒀다. `prot`은 매크로 대신 숫자로 썼다(누락된 매크로가
   빌드를 깨는 걸 막기 위해).
3. **체크섬을 검사한다.** 매핑이 성공하고 0으로 읽히면 "아주 좋은 대역폭"처럼 보인다.
   호스트가 쓴 패턴과 대조하는 것만이 유일한 assert다 — 나머지는 §8의 두 프로브처럼
   **보고만 한다**(못 하는 기기는 계획에 대한 사실이지 깨진 빌드가 아니다).

`rpcmem_to_fd`는 `HtpRpcMemApi`에 선택적으로 dlsym 된다. 없는 libcdsprpc.so면
`HtpRpcBuffer::fd()`가 −1을 주고 테스트는 skip 한다.

**판정:**

| 관측 | 결론 |
|---|---|
| `mapped=yes`, `checksum_ok=yes`, `dma_gbs` ≈ 27–33 | **통과.** A1–A4 그대로 진행 |
| `dma_gbs` ≪ 27 | 매핑은 되는데 느리다. 아레나는 되지만 weight 스트리밍 예산을 다시 계산해야 한다 |
| `err=EUNSUPPORTED` | SDK에 `HAP_mem.h`가 없다. 다른 매핑 API를 찾거나 A 재설계 |
| `mapped=no` / `checksum_ok=no` | **Gate 0c 실패.** §32.5 전체 재설계 |

### 32.8 Gate 0c 1차 (2026-09-14, 기기) — `HAP_mmap`이 null, fd가 붙어있지 않았다

```
U8I4_FIELD path=arena field=rpcmem_to_fd value=yes
U8I4_FIELD path=arena field=fd           value=14
U8I4_FIELD path=arena field=err          value=0x80000402   ← AEE_ENOMEMORY
```

`0x80000402`는 내가 `HAP_mmap` 실패에 붙인 코드다. 읽히는 것:

| | |
|---|---|
| `HAP_mem.h` | **있다** (없었으면 `EUNSUPPORTED`) |
| `HAP_mmap` 링크 | **된다** (빌드 통과) |
| `rpcmem_to_fd` | **있다**, fd 14를 줬다 |
| `HAP_mmap(NULL, bytes, RW, 0, fd, 0)` | **null 반환** |

**호스트 fd 번호는 FastRPC 세션에 붙기 전까지 DSP에게 의미가 없다.** rpcmem 버퍼가
IDL `sequence`로 넘어갈 때는 프레임워크가 그 호출 동안만 매핑해준다 — 프로파일의
"ION은 pinned+mapped per call이 아니다"가 그 얘기고, **호출이 끝나면 사라진다.**
영구 매핑에는 `fastrpc_mmap`이 필요하고, 1차 프로브에 그게 없었다.

**2차 (코드 완료, 미측정):**
1. ARM이 `fastrpc_mmap(CDSP, fd, buf, 0, bytes, FASTRPC_MAP_FD)`로 먼저 붙인다.
   dlsym으로 찾는다 — 없는 libcdsprpc.so는 링크 에러가 아니라 보고할 사실이다
2. DSP가 **null과 MAP_FAILED를 구분해서 보고**한다(`res[6]`). null은 "이 fd를
   모른다"(= 안 붙었다), MAP_FAILED는 "알지만 거절했다" — 1차는 이 둘을 못 갈랐다
3. 끝나면 `fastrpc_munmap`

이건 **가설 하나짜리 변경**이다: 빠진 게 attach라면 통과하고, 아니면 `hap_mmap`
필드가 null인지 failed인지로 다음 후보가 갈린다.

### 32.9 Gate 0c 2차 — attach는 됐다, `HAP_mmap`이 MAP_FAILED

```
fastrpc_mmap    = yes      fastrpc_mmap_rc   = 0x00000000   ← attach 성공
attached        = yes      fastrpc_munmap_rc = 0x00000000
hap_mmap        = failed   ← null 아님. DSP가 fd를 알고서 거절했다
err             = 0x80000402
```

1차의 가설(attach 누락)이 맞았고 고쳤다. 이제 실패가 **null → MAP_FAILED**로 바뀌었다 —
1차에 둘을 구분해두지 않았으면 이 진전을 못 봤을 것이다.

**원인: `flags = 0`.** POSIX `mmap`은 `MAP_SHARED`나 `MAP_PRIVATE` 중 **정확히 하나**를
요구하고 0은 `EINVAL`이다. 매크로 누락으로 빌드가 깨지는 걸 피하려고 숫자를 쓴 결정이
여기서 비용을 냈다 — prot은 맞게 찍었는데 flags를 안 찍었다.

**3차 (코드 완료·미측정):** 값을 하나 더 찍지 않고 **가능한 조합을 전부 시도하고 어느
것이 통과했는지 보고한다.** 기기 실행 한 번으로 끝난다.

| # | prot | flags | |
|---|---|---|---|
| 1 | rw | shared | 가장 유력 |
| 2 | r | shared | 읽기만으로 충분하다면 |
| 3 | rw | private | |
| 4 | r | private | |
| 5 | rw | 0 | 2차가 쓴 것. 통과하면 진단이 틀린 것이므로 남겨둔다 |

`res[6]` = 통과한 번호(0이면 전부 실패), `res[7]` = 조합당 2비트로 null(1)/MAP_FAILED(2).

**이 프로브가 세 번 만에 여기까지 온 방식이 요점이다.** 매번 하나씩 좁혔고, 각
단계에서 "무엇이 실패했나"가 아니라 **"어떻게 실패했나"**를 남겨둔 덕에 다음 가설이
나왔다. 1차: 헤더·심볼 존재 확인. 2차: attach가 원인. 3차: flags.

### 32.10 Gate 0c 3차 (2026-09-14, 기기) — 전부 MAP_FAILED, 그리고 `HAP_mmap_get`도 거부

```
hap_mmap_accepted   = none        hap_mmap_fail_mask = 0x2aa   ← 5조합 전부 MAP_FAILED, null은 0
hap_mmap_get_rc     = 0x80000448  hap_mmap_get_paddr_lo = 0x0  ← 4차 전반부: getter도 거부
fastrpc_mmap_rc     = 0x0         attached = yes
```

`0x2aa` = 조합당 2비트가 전부 `2`(MAP_FAILED). 3차의 flags 가설은 **틀렸다** — 어떤 flags를
줘도 같은 답이다. 프로브의 판독("MAP_FAILED = 알고서 거절")은 맞았는데, 그 다음 추론이
한 단계 더 필요했다: DSP가 fd를 "안다"는 것과 그 fd가 **`HAP_mmap`으로 매핑할 수 있게
태그돼 있다**는 것은 다르다.

4차 전반부에서 `HAP_mmap_get`(fd, &va, &paddr)을 먼저 시도하게 했고, 시그니처를
`(fd, void**, int* size)`로 잘못 짚어 빌드가 멈췄다 — 헤더는 `uint64 *paddr`다. 고쳐서
돌리니 **getter도 `0x80000448`** 로 거부. 즉 DSP 쪽 호출은 둘 다 아니고, 문제는 **호스트가
붙이는 방식**이다.

**원인: attach flag `0`.** 테스트가 `kMapFd = 0`을 `FASTRPC_MAP_FD`라고 주석 달고 넘겼는데,
`remote.h`의 `enum fastrpc_map_flags`에서 **0은 `FASTRPC_MAP_STATIC`** — 버퍼를 *호출
인자로 넘길 때* 드라이버가 만드는 고정 원격 주소 매핑이고, fd에 태그되지 않는다.
`FASTRPC_MAP_FD`는 **2**이고, 그 주석이 정확히 "DSP에서 `HAP_mmap_get()`/`HAP_mmap_put()`
으로 주소를 얻는다"이다. 2차에서 `flags=0`이 틀렸던 것과 같은 종류의 실수를 **다른
함수에서 한 번 더** 했다 — 매크로 대신 숫자를 쓴 곳마다.

**4차 (코드 완료·미측정):** 테스트가 이미 `<remote.h>`를 include하므로 `CDSP_DOMAIN_ID`와
`FASTRPC_MAP_FD`를 그대로 쓰고, 보낸 값을 `map_domain`/`map_flag`로 찍는다. DSP 쪽은
`HAP_mmap_get` 우선, 5조합 폴백 유지. IDL 무변경.

| 결과 | 뜻 |
|---|---|
| `hap_mmap_accepted=mmap_get`, `paddr_lo≠0`, `checksum_ok=yes` | **Gate 0c 통과** → A1 |
| `mmap_get` 통과, `checksum_ok=no` | 매핑은 됐는데 CPU 캐시가 안 내려감. `FASTRPC_MAP_FD`는 캐시 유지가 사용자 책임 — 패턴을 attach *전에* 쓰므로 attach 시 dma_buf map이 내려줄 것으로 봤다. 아니면 `rpcmem_sync_cache` 또는 uncached alloc |
| `hap_mmap_get_rc` 여전히 `0x80000448` | `FASTRPC_MAP_FD_DELAYED`(3) 시도. 그것도 아니면 이 SDK/커널 조합이 fd 태그 매핑을 지원하지 않는 것 — §32.5 재설계 |

**남는 교훈**: 숫자로 쓴 상수는 세 곳(`HAP_mmap` flags, `fastrpc_mmap` flags, 그리고
2차의 prot) 모두에서 비용을 냈다. 헤더가 있으면 헤더를 쓴다. 헤더가 없어서 빌드가 깨지는
것이 틀린 숫자로 기기 사이클을 태우는 것보다 싸다.

**RegistryCapacity 부수 결과**: `released=0 of 516`, `first_release_err=0x27`. `0x27`은
`hexkl_weight_u8i4_release`가 돌려줄 수 있는 두 값(0, `0x8000040e`) 어느 쪽도 아니므로
호출이 함수에 도달하지 못했다 — `0x8000040d`(`AEE_EBADSTATE`)로 멈춘 PD의 전송 오류.
테스트는 cap에서 멈춘 경우에만 release를 assert하도록 바꿨다.

### 32.11 Gate 0c 4차 (2026-09-14, 기기) — **통과**

```
map_domain=3  map_flag=2  fastrpc_mmap_rc=0x0
hap_mmap_accepted=mmap_get  hap_mmap_get_rc=0x0  hap_mmap_get_paddr_lo=0xfee00000
map_us=1   dma_us=35 (1 MiB)   dma_gbs=29.96   checksum_ok=yes   unmapped=yes
```

| 확인된 것 | 숫자 | 뜻 |
|---|---|---|
| 호스트 rpcmem 버퍼가 DSP 주소공간에 붙는다 | `paddr=0xfee00000` | `fastrpc_mmap(FASTRPC_MAP_FD=2)` + `HAP_mmap_get` |
| 매핑 비용 | 1 µs | 세션당 한 번, 무시 |
| 아레나 → VTCM DMA | **29.96 GB/s** | DSP 힙 가중치(27–33)와 동일. 아레나라서 느려지는 것 없음 |
| 데이터 | `checksum_ok` | attach 전에 cached ION에 쓴 패턴이 그대로 읽힘 |

네 번 걸린 이유는 전부 **헤더 대신 숫자를 쓴 상수**였다: `HAP_mmap` flags(2차),
`HAP_mmap_get` 시그니처(4차 전반), `fastrpc_mmap` flag `0`=`FASTRPC_MAP_STATIC`(3차).
헤더가 있으면 헤더를 쓴다.

**프로브가 증명한 순서**: cached ION → 쓰기 → attach → DSP 읽기.
§34의 설계는 **uncached ION → attach → 쓰기 → DSP 읽기**를 쓴다. 그 순서는 아직
미증명이다 (§34.6 #1).

## 33. V1이 깨졌다 → 해결 (2026-09-14, 기기) — 원인은 레퍼런스의 FMA

`run_u8i4_layer_on_device.sh`가 드디어 제대로 빌드된 실행에서:

```
[ FAILED ] HmxMmU8I4Layer.MoeLayerMatchesTwoCallReference (918 ms)
U8I4_FIELD path=moe_layer field=bad_elems value=20097 of 409600   (4.9%)
  first at 0 (row 0 col 0): got 0x1.9e697ep+12  want 0x1.9e698p+12
```

**`0x9E697E` vs `0x9E6980` = 정확히 1 ULP.** 이게 진단의 전부다:

- **DMA 경쟁이 아니다.** 미완료 전송을 읽었다면 쓰레기 값이 나온다. R2-1 청크 스트리밍을
  제일 의심했는데 **틀렸다**
- **부동소수점 연산 순서/입력이 미세하게 다르다.** 4.9%가 마지막 비트에서만 갈린다

### 33.1 배제한 것

| 후보 | 배제 근거 |
|---|---|
| quantize-once가 scale/zp를 바꿨나 (§12.3의 내 단언) | `quant_rows_worker`가 행마다 독립. **단언이 맞았다** — 검증함 |
| acc 타일 스테이징 간격 8192가 작나 | `TILE_N = 64×32 = 2048` int32 = 정확히 8192 B. `hexkl_acc_layout_get`이 `base + 63*stride + 31 < TILE_N`를 전 원소 검사하므로 넘칠 수 없다 |
| `dq_tiles_ctx` 초기화자 순서 | 필드 순서와 일치 확인 |
| 배치가 gate/up 경계를 가로지를 때 | `split` 판정이 타일마다 c0로 갈라 정확 |
| `res_f32` 별칭 | 배치 여부와 무관하게 같은 패턴 |
| acc 레이아웃 캐시 | 세션당 1회, 첫 호출자가 정한다 — 전과 동일 |

### 33.2a 원인 — 레퍼런스의 FMA였다 (2026-09-14, 기기 확인)

**커널이 아니라 테스트 레퍼런스가 DSP보다 정확한 산술을 하고 있었다.**

| | |
|---|---|
| 레퍼런스 (ARM, 호스트) | `dst[c] += src[c] * w;` — 한 문장 → clang이 `-ffp-contract=on` 기본값으로 FMLA 융합, **반올림 1회** |
| 커널 (`hvx_scale_add_rows_f32`) | `Q6_Vsf_vadd_VsfVsf(d, Q6_Vsf_vmpy_VsfVsf(s, wv))` — **반올림 2회** |

첫 기여에서는 `dst`가 0이라 `fma(s,w,0) == round(s*w)`로 양쪽이 같다. **두 번째
기여부터** 갈린다. 그래서 오차가 multi-expert 행에만 몰렸고 scatter/배칭 문제처럼
보였다.

라우팅을 바꿔가며 두 번 재면 배칭이 아니란 게 드러난다:

| 라우팅 | 기여 ≥2인 행 | 그 원소 | `bad_elems` | 비율 |
|---|---|---|---|---|
| 복원추출 (원래) | 34 | 69632 | 20097 | 28.9% |
| 비복원추출 | 38 | 77824 | 20709 | 26.6% |

행 수는 바뀌는데 비율은 그대로고, 기여 1개인 행은 양쪽 다 전부 일치한다.

**수정**: 레퍼런스에서 곱을 별도 문장으로 빼 두 번 반올림하게 했다. 기기 결과
`bad_elems=0`, `max_ulp=0`, `bad_rows=0 of 127 routed`. **V1 그린.**

**프로덕션 함의**: scatter를 ARM에서 DSP로 옮기면 top-k 누적의 반올림이 1회에서
2회로 바뀐다. 1 ULP라 토큰 선택은 안 바뀌지만 ARM 경로와 비트 단위로 같지는 않다.

#### 반증된 가설 (기록용)

처음엔 **≥3 expert 행의 결합법칙**이라고 봤다. 기여 3개 이상인 행이 6.00%,
관측이 4.91%라 82%로 맞아떨어졌다. **우연이었다.** 라우팅을 비복원추출로 바꾸니
그 상한이 1.00%로 6배 줄었는데 오차는 5.06%로 그대로였다 — 그게 반증이다.
숫자가 맞아떨어지는 것과 원인인 것은 다르다. 상한을 **움직여서** 오차가 따라오는지
봐야 한다.

같이 나온 진짜 문제 하나는 별개로 고쳤다: 테스트가 `row_index`를 복원추출로
만들어 expert 0 안에서만 같은 행이 13번 중복됐고, `moe_scatter_worker`가 주석에
명시한 전제(한 expert 블록 안에서 row_index는 전부 다르다)를 어겨 두 워커가 같은
출력 행을 동시에 read-modify-write 했다. 이번 실패의 원인은 아니었지만 실제
레이스였다.

#### 남은 진단 필드

다시 깨지면 한 번 돌려서 위치를 짚으라고 붙여놨다:

| 필드 | 읽는 법 |
|---|---|
| `bad_on_single_expert_rows > 0` | matmul이나 quantizer — 모든 행이 지나는 경로 |
| `bad_rows` 가 작고 `worst_row_bad_cols` 가 크다 | 특정 행의 누적 |
| `bad_rows` ≈ routed, `worst_row_bad_cols` 작다 | 전 행에 얇게 — 산술 정밀도 차이 |
| `max_ulp = 1` | 반올림 차이. 2 이상이면 값 자체가 다르다 |

### 33.2 bisect 계획 (쓰지 않았다 — 33.2a가 답했다)

읽어서 나올 버그가 아니다. 기능 변경 커밋이 여섯이고, 한 테스트만 돌리면 1초다.

| | 커밋 | 무엇 |
|---|---|---|
| 1 | `00208b7` | quantize-once (이 이후 V1을 **기기에서 안 쟀다** — 여기가 이미 깨졌을 수 있다) |
| 2 | `caad187` | staging DMA |
| 3 | **`da0ab81`** | **R1: 풀 dequant + gather k-타일 분할** |
| 4 | **`3bc2a1d`** | **R2: gather 제거(슬롯 팩) + gate_up 청크** |
| 5 | **`d8f3d3c`** | **R2b: 블록 DMA + down 청크** |
| 6 | `2acee32` | R3 (등록 경로만 — 커널 무관) |

이분 탐색: **3번(`da0ab81`)부터** 본다. 통과하면 4~5, 실패하면 1~2.

```bash
git checkout <sha>
source $HEXAGON_SDK_ROOT/setup_sdk_env.source
HEXKL_SDK_VER=6.4.0.2 ./test/htp/build.sh
(cd test/jni && $ANDROID_NDK/ndk-build NDK_PROJECT_PATH=. \
   NDK_APPLICATION_MK=./Application.mk APP_BUILD_SCRIPT=./Android.mk \
   NNTRAINER_ROOT=$PWD/../.. HEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT \
   unittest_hvx_mm_u8i4)
adb push test/htp/build/libnntr_hvx_skel.so test/jni/obj/local/arm64-v8a/unittest_hvx_mm_u8i4 \
   /data/local/tmp/htp_u8i4_layer_test/
adb shell "cd /data/local/tmp/htp_u8i4_layer_test && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
   ./unittest_hvx_mm_u8i4 --gtest_filter='*MoeLayerMatchesTwoCall*'" | grep bad_elems
```

### 33.3 교훈

**V1은 §13 이후 기기에서 한 번도 안 돌았다.** 그 사이 커널이 여섯 번 바뀌었고
매번 "호스트 체크 비트 동일"로 넘어갔다. 호스트 스텁은 HMX도 HVX도 진짜 DMA도
모델링하지 않는다 — **§7이 V1을 만든 이유가 정확히 그것인데, 그 뒤로 안 썼다.**

모델 텍스트가 정상이었던 것도 이걸 못 잡았다: 1 ULP 차이는 473토큰 생성에서
같은 토큰을 고르게 한다. **정확성 게이트는 텍스트가 아니라 V1이다.**

→ `run_u8i4_layer_on_device.sh`를 커널 변경마다 돌린다. 1초짜리 필터가 있다.

## 34. 아레나 구현 설계 — 인계 (2026-09-14)

A1·A2·A3를 하나로 합친다. 별도의 변환 모드도, 캐시 파일을 DSP 힙으로 다시 복사하는
경로도 없다. **DSP 힙에 가중치를 상주시키는 경로 자체를 없앤다.**

### 34.1 한 줄

> `NNTR_HTP_WEIGHT_CACHE` 디렉터리가 변환된 모델이다. 호스트는 그 파일들을 **uncached
> rpcmem 아레나**에 읽어 넣고, 아레나를 세션당 한 번 DSP에 매핑하고, 가중치는
> **아레나 오프셋**으로 등록한다. 파일이 없는 가중치는 그 자리에서 bake → export →
> 파일 기록 → 아레나에 기록 → 아레나로 등록 → **힙 사본 즉시 해제**.

결과: DSP 힙에는 어느 순간에도 bake 중인 가중치 하나만 있다. 1.89 GB 천장(Gate 0)이
모든 경로에서 사라지고, A4(22개 레이어 전부 HTP)는 설정 한 줄이 된다.

### 34.2 데이터 흐름

```
get_or_register_qs4cx(ptr, scale, K, N)
  │ pointer cache hit → handle
  │ ensureArena()          ← 프로세스당 한 번: 디렉터리의 파일 전부 → 아레나, 매핑
  │ hash = FNV(src bytes) ^ FNV(scale)
  ├─ arena_index[path(K,N,hash)] 있음  →  register_arena(chunk.dsp_id, off)  → handle   [hit: RPC 1회, ~0.1 ms]
  └─ 없음:
       convert(QS4CX→int8 RM) → register(bake, DSP 힙) = hh
       export(hh) → wh bytes (host)
       store(file)                                      ← 다음 실행이 hit
       place(wh.size) → (chunk, off); memcpy(arena+off, wh)   ← uncached라 flush 불필요
       register_arena(...) = ah;  release(hh)          ← 힙 비움
       handle = ah
```

### 34.3 이미 커밋된 것 (`[htp] WIP: arena registration, DSP side and host headers`)

**DSP** (`test/htp/nntr_hvx.idl`, `nntr_hvx_mm_u8i4.c`, `nntr_hvx_session.h`,
`hvx_add_f32.c`, `hmx/hexkl_mm_u8i4_dma.[ch]`) — skel은 단독으로 빌드된다:

| 호출 | 시그니처(호스트 스텁) | 계약 |
|---|---|---|
| `nntr_hvx_arena_attach` | `(h, int32 fd, uint32 bytes, uint32 *arena)` | `HAP_mmap_get(fd)`. 최대 `NNTR_HVX_MAX_ARENAS=8`. 실패: `AEE_ENOMEMORY`(자리 없음) 또는 HAP rc |
| `nntr_hvx_arena_detach` | `(h, uint32 arena)` | 그 아레나를 빌리는 슬롯이 하나라도 살아있으면 **`AEE_EBADSTATE`** |
| `nntr_hvx_weight_register_u8i4_arena` | `(h, K, N, arena, wh_off, w_scale, N, colsum_w, N, bias, N, uint32 *w_handle)` | `wh_off + (K/32)(N/32)·512 ≤ bytes` (DSP가 검사), `wh_off % 512 == 0`. 슬롯은 `borrowed=1`, release가 free하지 않음 |
| `nntr_hvx_close` | — | 슬롯 해제 후 모든 아레나 `HAP_mmap_put` |
| ~~`weight_register_u8i4_baked`~~ | **삭제됨** | baked bytes가 돌아오는 길은 아레나뿐 |
| `nntr_hvx_weight_bake_export` | 유지 | miss 경로가 파일과 아레나를 채우는 데 쓴다 |

**호스트 헤더**:

| 파일 | 추가/변경 |
|---|---|
| `htp_rpcmem.h` | `HtpRpcMemApi::get().mmap(domain, fd, addr, offset, len, flags)`, `.munmap(domain, fd, addr, len)` (dlsym, 없을 수 있음). `HtpRpcBuffer(bytes, flags)`; `HTP_RPC_FLAGS_UNCACHED = 0`; `.fd()` |
| `htp_weight_cache.h` | 파일 형식 그대로. `load()` 삭제 → `listFiles()`, `readHeader(FILE*, hdr&)`, `readPayload(FILE*, hdr, uint8_t *wh, vec&, vec&, vec&)` (payload를 **호출자 포인터** = 아레나로 직접 읽는다). `store()`, `path()`, `whBytes()`, `htpWeightHash()` 유지 |
| `run_u8i4_layer_on_device.sh`, `bisect_moe_v1.sh` | `generate_stub.sh` 매번 실행 (IDL 변경 → 두 스텁) |

### 34.4 `htp_compute_ops.cpp` 포팅 — **구현 완료·미측정**

지울 것: `registerFromCache`, `exportToCache`, 그리고 `get_or_register_qs4cx` 안의
`wc.load`/`register_u8i4_baked` 호출.

넣을 것 (모두 `HtpComputeOps`의 private, `handle_mutex_` 아래):

```cpp
struct ArenaChunk { std::unique_ptr<HtpRpcBuffer> buf; uint32_t dsp_id; size_t used; };
struct ArenaEntry { uint32_t chunk, off; std::vector<float> w_scale, bias; std::vector<int32_t> colsum; };
std::vector<ArenaChunk> arena_chunks_;
std::unordered_map<std::string, ArenaEntry> arena_index_;   // key = wc.path(K,N,hash) — 이미 유일
enum { ARENA_UNTRIED, ARENA_ON, ARENA_OFF } arena_state_ = ARENA_UNTRIED;
```

`ensureArena(session)` — 첫 호출에서 한 번:
1. `!wc.enabled() || !api.to_fd || !api.mmap` → `ARENA_OFF`, 끝. (힙 경로가 그대로 남아 있어 동작은 한다 — 천장만 돌아온다.)
2. `files = wc.listFiles()`; 각 파일: `fopen` → `readHeader` 실패면 skip → `place(hdr.wh_len)` → `readPayload(f, hdr, chunk.data()+off, ...)` 실패면 index에 넣지 않음(바이트는 버려짐, `used`는 그대로) → `arena_index_[path] = entry`.
3. `ARENA_ON`. 파일이 0개여도 ON — chunk는 첫 miss에서 만든다.
4. 로드 시간과 GB/s를 프로파일 줄로 찍는다 (§34.6 #3).

`place(bytes) → (chunk, off)`: `off = align_up(chunk.used, 4096)`; 안 들어가면 `newChunk(need)`:
`size = clamp(align_up(need, 64 MiB), 256 MiB, 1 GiB)`; 초기 로드에서 `need` = 아직 못 넣은 파일들의 합, miss에서 `need` = `wh_len`.
`newChunk`: `HtpRpcBuffer(size, HTP_RPC_FLAGS_UNCACHED)` → `!isIon()` 이면 `ARENA_OFF` →
`fd = buf.fd()` → `api.mmap(CDSP_DOMAIN_ID, fd, data, 0, size, FASTRPC_MAP_FD)` (둘 다
`<remote.h>` 이름으로, 숫자 금지) → `nntr_hvx_arena_attach(session, fd, size, &dsp_id)`.
어느 단계든 실패 → `ARENA_OFF`, 호출자는 힙 경로.

`get_or_register_qs4cx`는 §34.2 그대로. 세부:
- `register_locked`는 지금처럼 `handle_cache_`에 힙 핸들을 넣는다. 아레나로 바꾼 뒤 **덮어쓴다** (`handle_cache_[ptr] = ah`).
- miss에서 `export`·`store`·`place`·`register_arena` 중 하나라도 실패하면 **힙 핸들을 그대로 돌려준다** — 지금과 같은 동작, 천장만 있다.
- hit: `profile.addRegister(total, /*convert_us=*/0, rpc_us, /*ion=*/true)`.
- `get_or_register_q4_0x4`(Q4_0 경로)는 같은 꼬리를 태울 수 있으면 태우고, 아니면 힙 그대로 둔다. MoE는 qs4cx 경로다.

해제: 호스트에서 `fastrpc_munmap`은 **하지 않는다** — `ponytail:` DSP `close()`가
`HAP_mmap_put`을 하고, ION은 프로세스 종료 시 커널이 회수한다. `HtpBackend`와 ops
싱글톤의 소멸 순서가 정해져 있지 않아 close 뒤에 munmap을 부르면 죽은 세션을 만진다.

### 34.5 테스트 — **구현 완료·미측정** (`test/unittest/unittest_hvx_mm_u8i4.cpp`)

1. **`ArenaUncachedWriteAfterMap`** (`path=arena_uncached`): `rpcmem_alloc(25, 0 /*UNCACHED*/, …)` → `fastrpc_mmap(FASTRPC_MAP_FD)` → **그 다음** 패턴 쓰기 → `arena_probe` → `checksum_ok=yes`. §34.2가 실제로 하는 순서 그대로다. 이게 `no`면 §34 전체가 틀린 것이고, 그 땐 "cached + attach 전 쓰기"(증명됨)로 후퇴: 초기 로드는 그대로 가능하고 miss는 파일만 쓰고 힙에 남긴다.
2. **`MoeLayerFromArenaMatchesHeap`** (`path=arena_moe`): expert 4개의 gate_up/down을 bake로 등록(힙) → 각각 `bake_export` → uncached 아레나 하나에 4096 정렬로 memcpy → `fastrpc_mmap` → `arena_attach` → `register_arena` ×8 → 같은 입력으로 `moe_layer`를 힙 핸들/아레나 핸들로 각각 실행 → **`memcmp == 0`** (같은 바이트, 같은 커널이므로 1 ULP도 허용 안 함). 이어서 핸들이 살아있는 채 `arena_detach` → `AEE_EBADSTATE` 확인 → release ×8 → `arena_detach` → `AEE_SUCCESS`.

두 번째가 §33이 요구하는 "runnable check"다. 첫 번째는 gate다.

#### 구현된 것 (커밋 `[htp] Register MoE weights out of a DSP-mapped arena`)

| | |
|---|---|
| `ensureArena(session)` | 첫 호출 1회. `wc.enabled() && to_fd && fastrpc_mmap` 아니면 `ARENA_OFF`(힙 경로 그대로). 헤더를 **먼저 전부 읽어** 총량을 구한 뒤 청크를 잡는다 — 그래야 256 MB씩 늘리다 8개 한도에 걸리지 않는다. 로드 GB/s를 찍는다 |
| `place(bytes, want)` | 4 KB 정렬 bump. `want`는 "뒤에 더 올 양"이라 청크가 한 weight가 아니라 실행 전체에 맞게 잡힌다 |
| `newChunk` | `clamp(align_up(want, 64 MiB), 256 MiB, 1 GiB)` uncached ION → `fastrpc_mmap(CDSP_DOMAIN_ID, fd, …, FASTRPC_MAP_FD)` → `arena_attach`. 어디서 실패하든 `false` → 호출자는 힙 |
| `registerFromArena` | `e.K/e.N`이 요청 shape와 다르면 거부 — 다른 모델의 파일이 섞인 디렉터리를 막는다. `t_begin=0`이면 프로파일에 안 센다(miss에서 `register_locked`가 이미 셌으므로 중복 방지) |
| `moveToArena` | `bake_export`를 **아레나로 직접** → 그 바이트에서 파일 기록 → `register_arena`. 실패하면 `kNoHandle`, 호출자는 힙 핸들 유지 |
| `get_or_register_qs4cx` | hit → `register_arena` 1회. miss → convert·bake·export·store·place·register_arena·**힙 release**, `handle_cache_` 덮어쓰기 |

`fastrpc_munmap`은 호스트에서 안 부른다 (`ponytail` 주석): DSP `close()`가 `HAP_mmap_put`
하고 ION은 프로세스 종료 시 커널이 회수한다. 싱글톤 소멸 순서가 고정이 아니라 닫힌 세션을
만질 수 있다. 모델을 load/unload 반복하는 프로세스면 모델당 아레나 하나가 샌다 — 고치려면
`HtpBackend`가 세션을 닫기 **전에** 도는 shutdown 훅이 필요하다.

### 34.6 미증명·위험 — 각각 어떻게 답하는지

| # | 항목 | 답하는 방법 |
|---|---|---|
| 1 | **uncached ION + 매핑 후 쓰기**를 DSP DMA가 본다 | §34.5 #1. 프로브는 cached·매핑 전만 증명했다 |
| 2 | `rpcmem_alloc(…, flags=0)`이 ION을 준다(`isIon`) | 같은 테스트의 `alloc` 필드 |
| 3 | 파일 읽기 속도. B1a에서 ION으로 읽을 때 **0.31 GB/s** (178 MB에 583 ms). 3.9 GB면 12 s+ | `ensureArena`가 찍는 로드 GB/s. 느리면: 파일을 `mmap`해서 memcpy, 또는 파일 하나로 합치기 |
| 4 | 해시 비용: FNV-1a 바이트 루프, weight당 ~1.8 MB, 1408개 | 프로파일 `register` 총합. 느리면 ops seam에 텐서 이름을 태워 이름으로 키잉 (`ponytail`) |
| 5 | uncached 1 GiB 청크 ×4의 RSS | A5가 ARM QS4CX 사본을 떨어뜨리기 전까진 +3.9 GB. 측정 |
| 6 | `FASTRPC_MAP_FD`는 DSP 캐시 유지도 사용자 책임 | 같은 오프셋에 **다시 쓰는 일이 없다**(bump allocator)면 DSP에 stale line이 생길 길이 없다. 재기록을 넣게 되면 이 전제가 깨진다 |
| 7 | `HEXKL_MM_U8I4_MAX_WEIGHTS = 2048` ≥ 22×64 = 1408 | 충분. 22개 넘는 모델이면 상수 |

### 34.7 그 다음 (A4, A5)

- **A4**: §34.5 둘 다 통과하고 실제 모델이 layer 2에서 아레나로 도는 것을 보면
  `moe_htp_layers`를 22개 전부로. 첫 실행은 전부 miss(bake+export, 레이어당 ~1.6 s,
  총 ~35 s, 힙은 한 개씩) — 이게 변환이다. 두 번째 실행부터 hit. §32.5 예상 234 TPS.
- **A5**: 아레나에 있는 weight의 ARM QS4CX 사본을 안 만든다. 로더 작업, 별도.

### 34.8 다음 세션 프롬프트

> `docs/htp_attention/46_moe_resident_kernel_design.md` §34를 읽고 `htp_compute_ops.cpp`
> 포팅(§34.4)과 테스트 둘(§34.5)을 구현해줘. WIP 커밋(`[htp] WIP: arena registration`)
> 위에서 시작하고, 그 커밋이 지운 심볼(`register_u8i4_baked`, `HtpWeightCache::load`,
> `registerFromCache`, `exportToCache`)이 더 이상 참조되지 않게. 상수는 `<remote.h>`
> 이름으로만. 다 되면 `./test/htp/run_u8i4_layer_on_device.sh` 명령과 §34.6 #1의
> 두 가지 결과 각각에서 뭘 할지 알려줘. 기기 확인 전엔 "확인했다"고 하지 마.
