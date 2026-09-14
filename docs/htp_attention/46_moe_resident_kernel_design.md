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
