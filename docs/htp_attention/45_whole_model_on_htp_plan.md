# 45 — 목표 "CPU보다 훨씬 빠르게": 모델 전체를 HTP에 상주시키는 계획

**선행**: `44_moe_ffn_bottleneck_map.md` §15–16 (P3 결과, 레버 목록, MoE FFN만의 천장 1.2×).
**이 문서가 대체하는 것**: 44 §13.4/§16.6의 순서. MoE FFN 커널(P1)은 이제 독립 최적화가 아니라 이 계획의 **부품**이다.

---

## 0. 왜 계획의 축이 바뀌는가 — 숫자 하나

44 §16.5: MoE FFN에 가능한 모든 레버를 적용해도 **레이어 1.4×, 전체 prefill 1.2×, decode 0×**. 이유는 구조적이다: HMX 행렬곱은 이미 CPU FFN보다 3배 빠른데(8.9 vs 26.3 ms), 그 주변의 **포장 — transport 17.0, dequant 5.7, drain 3.7, quant 3.6 = 30 ms**가 연산의 3.4배다. 포장은 "값이 ARM에서 f32로 들어와 f32로 나간다"는 구조에서 나온다. 레이어 하나를 아무리 잘 만들어도 그 구조 안에서는 포장의 절반 이상이 남는다.

**"훨씬"(2× 이상)은 값이 레이어 사이에 DSP에 머물 때만 나온다.** 그러면:
- transport가 레이어당 0 (지금 17 ms, 전체 포장의 57%)
- dispatch/staging 0
- 양자화는 행렬곱 입력마다 한 번만 (HMX가 u8을 요구하므로 불가피), f32 왕복 재양자화는 소멸
- **decode도 달라진다** — 지금 HTP decode가 CPU와 같은 25.8 TPS인 이유의 상당 부분이 토큰당 64번의 FastRPC(≈26 ms)다. 한 토큰에 한 호출이면 DDR 한계(≈55 TPS)에 접근할 여지가 생긴다 (●○○, §6)

---

## 1. 모델 구조와 레이어당 필요한 DSP 연산

LFM2.5-8B-A1B, 실제 config.json (2026-09-10 확인): hidden 2048, 24 레이어 = **18 conv + 6 full_attention** (attention은 layer 2, 6, 10, 14, 18, 21), dense FFN 2 (intermediate **7168**) + MoE 22 (1792 × 32 experts, top-4), attention 32 heads / **8 kv heads (GQA 4) / head_dim 64**, conv L=3, max_seq_len 1024. FC weight는 **Q4_0**(HTP 경로는 `htp_qs4cx_from_q4_0x4`로 즉석 변환 — 상주 설계에서는 QS4CX로 재양자화해 한 번만), MoE는 QS4CX.

| 블록 | 연산 | DSP 커널 상태 | 신규 작업 |
|---|---|---|---|
| 공통 | RMSNorm | **없음** | HVX, ~50줄. **행렬곱 앞이므로 `_det`판 필요** (§4.3) |
| 공통 | residual add | `add_f32` 있음 | — |
| **conv (×18?)** | in_proj FC 2048→6144 | `mm_u8i4_layer` 있음 | weight를 QS4CX로 재양자화 (지금 Q4_0 → 즉석 변환 경로) |
| | split 3 + gate_a⊙gate_c | 없음 | HVX elementwise, 트리비얼 |
| | causal depthwise conv1d (L=3) | **없음** | HVX, ~100줄 + decode용 state cache |
| | gate_b ⊙ conv_out | 없음 | 위와 동일. **out_proj 앞이므로 `_det`** |
| | out_proj FC 2048→2048 | 있음 | — |
| **attention (×6?)** | wq/wk/wv/wo FC | 있음 | QS4CX 재양자화 |
| | q_norm/k_norm (head_dim RMS) | 없음 | RMSNorm 변형 |
| | RoPE | **없음** | HVX, ~80줄 |
| | attention core (GQA, causal) | **`attn_forward` 있음.** 실측(Qwen3-0.6B 형상, 8 kv × gqa 2 × 128 — LFM2와 토큰당 MAC 동일): 128-tok 청크 prefill **4.7 ms/layer @kv512, 6.0 @kv1024**, decode 0.56–0.74 ms/layer. **정확도 게이트가 합성 데이터 고정 허용치(I4에서 rel_err 1.83 통과)** — L2를 통과시킨 것과 같은 종류 | **모델 통합 없음.** `mha_core.cpp`에 dispatch 없음. KV cache를 DSP에 (`attn_kv_append` 있음). **실모델 차분 게이트 필수** |
| **dense FFN (×2)** | gate_up→SwiGLU→down | `mm_u8i4_gate_up_swiglu` + `_u8in` 있음 (= expert 1개) | — |
| **MoE FFN (×22)** | 라우팅 + 32 expert | 44 P1 — **미구현** | **DSP 상주 activation을 읽고 쓰는 인터페이스로 설계** (§3) |
| 외곽 | embedding, lm_head | CPU 유지 | prefill의 lm_head는 마지막 토큰만 |
| 오케스트레이션 | 레이어 전체를 한 호출로 | **없음** | `lfm2_layer_forward` IDL (§3) |

**결론**: 행렬곱과 attention은 있다. 없는 것은 **작은 HVX 연산 5개(norm, conv1d, gating, RoPE, q/k norm), MoE 배칭, 그리고 전부를 잇는 오케스트레이션**이다.

---

## 2. 천장 추정 — 측정된 처리량으로 (●○○, 그러나 근거 있음)

측정값: HMX 2.55 TMAC/s (44 §5), dequant 1.75 G elem/s (44 §15.5; P2 후 목표 ~10), DDR 34 GB/s, 444 tokens.

| 블록 | MAC | mm | 포장 (quant/dequant/acc/drain) | 기타 | **합** |
|---|---:|---:|---:|---:|---:|
| MoE FFN (상주, P1+P2) | 3.2 G × … | 8.9 | quant 0.9 + dequant 1.0 + acc 2.8 + drain 0.5 | swiglu 1.9 | **≈16** |
| MoE FFN (상주, P1만) | | 8.9 | 0.9 + 5.7 + 2.8 + 0.5 | 1.9 | ≈20.6 |
| conv 블록 | 7.5 G | 2.9 | ≈1.3 | conv1d+gating 0.3 | **≈4.5** |
| attention 블록 | 2.3 G + attn | 0.9 | ≈1.0 | attn_forward **8–16** (실측 4.7 ms/128-tok 청크 @kv512; 444 tok = 4청크, 청크당 고정비 ≈3.4 ms가 한 호출로 합쳐지면 하한) | **≈10–18** |
| norm + residual | | | | 0.2 | 0.2 |

레이어 평균 (18 conv, 6 attn, 22 MoE, 2 dense): 연산자 블록 (18×4.5 + 6×14)/24 ≈ **6.9** + FFN ≈15 + 0.2 ≈ **22 ms** × 24 = **530 ms**.
+ 레이어당 FastRPC 1회 0.4 × 24 = 10 + 입출력 transport 1회 + embedding/lm_head(CPU) ≈ 40 → **prefill ≈ 580 ms**. (첫 추정 520은 attention core를 2 ms로 잡은 것 — 실측 반영해 정정.)

| | prefill | vs CPU ≈1320 ms |
|---|---:|---:|
| 전부 달성 | ≈580 | **2.3×** |
| P2 실패 (dequant 그대로) | ≈680 | 1.9× |
| P2 실패 + drain 파이프라인 실패 | ≈760 | 1.7× |
| **44 §16 (MoE FFN만)** | ≈1150 | 1.15× |

**1.7~2.3×가 이 계획의 도달 범위다.** 44 §16.5의 1.2×와의 차이 전부가 "값이 DSP에 머문다"에서 나온다.

---

## 3. 설계 원칙 — 지금 만드는 것이 나중에 버려지지 않게

### 3.1 activation은 DSP-side 버퍼 핸들로 오간다
모든 신규 IDL 엔트리는 f32 포인터가 아니라 **DSP-side activation 핸들**(rpcmem 버퍼 id + offset)을 받고 돌려준다. residual stream 444×2048 f32 = 3.6 MB는 VTCM(8 MB, weight 이중 버퍼 5.5 MB와 공존 불가)이 아니라 **DSP가 DMA로 접근하는 DDR**에 산다. DSP-side DDR 왕복은 3.6 MB / 34 GB/s ≈ 0.2 ms — FastRPC transport 17 ms의 1/80.

오늘의 모델에 즉시 쓰기 위해 **f32-in/f32-out 래퍼**를 따로 둔다: 래퍼가 f32를 핸들로 올리고 → 상주 커널 → 핸들을 f32로 내린다. 래퍼는 Phase D에서 사라지고 커널은 그대로 남는다.

### 3.2 weight DMA는 연산 뒤에 숨긴다 — 예외 없이
44 §14: u8in 경로에서 quant가 사라지자 DMA 5.18 ms가 통째로 노출됐다. 상주 구조에서는 quant가 대부분 사라지므로 **모든 커널이 다음 weight를 현재 mm 중에 당겨야 한다.** 이건 최적화가 아니라 커널의 계약이다.

### 3.3 양자화기 앞의 연산은 전부 `_det`
44 §3/§12: 두 근사가 u8 경계 하나를 뒤집으면 22 레이어 뒤 토큰이 바뀐다. 이 계획에서 양자화기 앞에 오는 연산: **RMSNorm(모든 FC 앞), conv 블록의 gate_b⊙conv_out(out_proj 앞), SwiGLU(down 앞, 완료), q/k norm + RoPE(attention 앞)**. 각각 `swiglu_det.h`와 같은 패턴 — 하나의 스펙, 스칼라/NEON/HVX 세 구현, 비트 비교 게이트. 새 패턴이 아니라 **있는 패턴의 반복**이다.

CPU 레퍼런스 경로도 같은 `_det`를 써야 한다 (A1 단계 A가 증명한 것: 모델은 ≤1 ULP 변화를 견딘다).

### 3.4 게이트는 SNR이 아니라 비트 동일 + 텍스트
각 Phase의 완료 조건: (a) 커널이 스칼라 스펙과 비트 동일, (b) `NNTR_L2_DIFF`류의 실모델 차분 0, (c) 생성 텍스트가 CPU와 동일. 셋 다.

---

## 4. Phase와 게이트 — 비용은 기기 실행 횟수, 기간은 1인 기준

### Gate 0 — DSP가 모델 전체를 담을 수 있는가 (1일, 실행 1) — **계획을 죽일 수 있는 유일한 게이트, 그래서 첫 번째**
4-bit 모델 ≈ 4.3 GB의 weight가 DSP-side 메모리(rpcmem/ION)에 상주해야 한다. 확인할 것: (a) rpcmem이 그만큼 할당되는가, (b) **DSP의 주소 공간이 4 GB를 넘는가** (Hexagon 사용자 PD의 VA 한계), (c) `HEXKL_MM_U8I4_MAX_WEIGHTS=512`를 ~1700으로 올려도 레지스트리가 견디는가. 방법: 더미 weight를 실패할 때까지 등록하는 테스트. **실패하면** weight를 레이어마다 스트리밍해야 하고(레이어당 ~180 MB FastRPC ≈ 90 ms) 이 계획은 성립하지 않는다 — 그때는 더 작은 모델이거나 다른 설계다.

### Gate 1 — M1: CPU prefill의 나머지 47%가 어디에 있는가 (2일, 실행 1)
M0/P3와 같은 계측을 conv 블록 / attention 블록 / dense FFN / norm / embedding+lm_head에 넣는다. §2의 추정을 실측으로 바꾸고 **Phase B와 C의 순서**를 정한다.

### Phase A — MoE FFN 상주 커널 = P1 (6–8주, 실행 4–5) — **−28 ms/layer가 오늘 모델에 바로 들어온다**
`moe_ffn_resident(act_handle_in, act_handle_out, routing[], handles[])`: activation 1회 u8 양자화(VTCM) → expert마다 row gather → gate_up → SwiGLU_det → requant → down → route weight 곱해 누적. weight DMA를 expert 간 파이프라인(§3.2). 44 §16의 P1 항목 전부 + P2 프로브. **f32 래퍼로 오늘 모델에 연결**해 57.5 → ≈29 ms를 먼저 확인한다. 게이트: `NNTR_L2_DIFF` 32/32 = 0, 텍스트 동일.

### Phase B — conv 블록 상주 (3–4주, 실행 3)
RMSNorm_det, gating, conv1d(state cache 포함), in_proj/out_proj. `conv_block_resident`. 게이트: 호스트 스칼라 모델과 비트 동일, 실모델 차분 0. Gate 1 결과에 따라 C와 순서 교환.

### Phase C — attention 블록 상주 (3–4주, 실행 3)
있는 `attn_forward` 통합 + RoPE + q/k norm_det + wq/wk/wv/wo + KV cache DSP 상주. `attn_block_resident`. 게이트: `mha_htp_host_model` 대비 비트 동일, 실모델 차분 0.

### Phase D — 레이어 오케스트레이션 + 등록 캐시 (3주, 실행 3) — **"훨씬"이 여기서 나타난다**
`lfm2_layer_forward`: norm → (conv|attn) → residual → norm → (dense|MoE) → residual, activation은 핸들. 이어서 N 레이어를 한 호출로. **P4 등록 캐시 필수**(24 레이어 × 1.5 s = 36 s → bake 결과를 파일로, 로드 ≈3 s). 여기까지 와야 transport가 사라지고 §2의 숫자가 측정된다. 게이트: prefill 텍스트 동일, **CPU 대비 ≥2×**.

### Phase E — decode (2–3주, 실행 2)
토큰당 한 호출로 24 레이어. 먼저 **M2**: FastRPC 없는 DSP 내부 decode 레이어 비용 측정 → §6의 가설 확인. 게이트: decode TPS ≥ CPU × 1.5.

**합계 ≈ 4–5개월.** Phase A 끝(≈2개월)에 중간 성과 −28 ms/layer, Phase D 끝(≈4개월)에 2×.

---

## 5. 위험 — 확률과 대응

| 위험 | 확률 | 영향 | 대응 |
|---|:-:|---|---|
| **Gate 0 실패** — DSP가 4.3 GB를 못 담음 | 25% | **계획 무효** | 첫날 확인. 실패 시 더 작은 모델로 같은 계획, 또는 중단 |
| `_det` 연산이 5개로 늘어 각각 A1급 검증 필요 | 확실 | 각 1–2주 | 패턴은 완성됨(`swiglu_det.h`). 반복이지 발명이 아님 |
| P2 실패 (dequant가 출력 메모리 문제라 커널로 못 고침) | 40% | 2.5× → 2.1× | 상주 구조에서는 출력이 DSP DDR이라 문제 자체가 바뀜. Phase A에서 프로브 |
| drain 파이프라인이 VTCM 예산에 안 맞음 | 30% | −3.2 ms 상실 | 레이아웃 계산을 코드 전에 문서로 |
| 지속 부하 스로틀 — DSP가 CPU보다 먼저 느려짐 | 50% | 2.5× → ? | 44 §13.3 규칙 4. **긴 실행(1000+ 토큰)에서 재측정** 필수 |
| attn_forward 통합이 예상보다 큼 (KV cache 생명주기) | 40% | +2주 | 커널은 검증됨. 통합만 |
| 등록 캐시 없이 36 s 로드 | 확실 | 사용 불가 | P4는 Phase D의 일부, 선택 아님 |

---

## 6. decode에 대한 정정 (●○○)

44 §16.4 C4는 "decode 0×, 확정"이라 했다. 그 확정(25.8 vs 25.7)은 **레이어당 64번 FastRPC를 하는 지금 구조**에서 잰 것이다. 토큰당 64 × 0.4 ms = 26 ms의 RPC 고정비가 decode 시간 ≈39 ms의 2/3다. 한 토큰에 한 호출이면 그 26 ms가 0.4로 줄고, 남는 것은 DDR 스트리밍(active 1.2B × 0.5 B = 0.6 GB / 34 GB/s ≈ 18 ms → ≈55 TPS)이다. M=1에서 HMX 활용률이 1/64이지만 MAC 자체가 작아 병목이 아니다. **가설이다** — Phase E의 M2가 확인한다. 맞으면 decode 25 → 40–50 TPS, **1.6–2×**.

---

## 7. 시작 전 확인 — 상태

1. ✅ **실제 config.json** — §1에 반영. 18+6 가정이 맞았다.
2. ✅ **attention 실측** — §1/§2에 반영. 2 ms 추정은 7배 낙관이었고, 천장이 2.5× → 2.3×로 내려왔다.
3. **Gate 0** — `unittest_hvx_mm_u8i4`의 `RegistryCapacity` (아래).
4. **Gate 1** — `NNTR_M1_PROFILE` (아래).
