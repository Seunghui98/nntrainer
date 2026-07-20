# ScreenAI: SigLIP2/BERT가 ONNX Runtime보다 느린 원인 분석

대상 브랜치: `claude/weight-q8-quantization-hr0r4o-pr4055` (HEAD `7cedc66`)

측정치 (S26U):

| Runtime | YOLOv11m | SigLIP2 | BERT decoder |
|---|---|---|---|
| Quick.AI (nntrainer) | 370.5 ms | **53.2 ms** | **8.6 ms** |
| ONNX Runtime | 661.1 ms | 50.9 ms | 7.1 ms |

- SigLIP2 격차: **+2.3 ms (+4.5%)**
- BERT 격차: **+1.5 ms (+21%)**

## TL;DR

질문이 "커널 단 차이냐, per-channel vs per-block 차이냐"인데, 결론은 **둘 다이지만
비중이 다르다**:

1. **SigLIP2 (+4.5%)**: GEMM 자체는 이미 MLAS int8과 경쟁 가능한 수준
   (b9bef77 커밋 메시지의 벤치 결과와 일치). 남은 격차는
   (a) **attention이 GEMM이 아닌 per-row dot fallback으로 도는 것**,
   (b) **activation의 Q8_0 online 양자화가 직렬 구간인 것**,
   (c) per-block 포맷이 inner loop에 강제하는 **32개마다의 scale fold**,
   (d) ORT의 그래프 융합(Attention/SkipLayerNorm/BiasGelu) 대비 **per-op 실행 +
   barrier 오버헤드**의 합.
2. **BERT (+21%)**: 모델이 극단적으로 작아서(dim 256, 4 layers) **커널 시간보다
   런타임 오버헤드(per-op parallel_for barrier, 레이어 dispatch)** 비중이 크고,
   FLOPs를 지배하는 **lm_head GEMV (30522×256)가 메모리 바운드**라 Q8_0의
   +6.25% weight bytes와 per-block fold가 그대로 격차로 나타난다.

per-channel vs per-block의 본질적 차이는 **메타데이터 크기(+6.25%)보다,
per-block이 int32 누적 체인을 32개 단위로 끊어서 커널 inner loop에 FP fold를
강제하는 것**이다. 즉 "포맷 차이가 커널 차이를 만든다" — 두 가설은 독립이 아니다.

---

## 1. nntrainer의 W8A32 실행 경로

### 1.1 FC (GEMM) 경로

```
FloatTensor::dotQnK (nntrainer/tensor/float_tensor.cpp:1036)
  → gemm_q8_0_fp32 → nntrainer::gemm_q8_0 (arm_compute_backend.cpp:393)
  → __ggml_q8_0_4x4_q8_0_GEMM (ggml_interface_bs_threadpool.cpp:845)
      M==1  → nntr_quantize_row_q8_0 + nntr_gemv_q8_0x4_q8_0
      M>=4  → nntr_quantize_mat_q8_0_4x8 (직렬) + nntr_gemm_q8_0x4_q8_0x4 (병렬)
```

- weight: offline repack된 q8_0x4 interleave (`__ggml_repack_q8_0_to_q8_0_4`),
  블록당 `d`(fp16 scale 1개) + int8 32개 = **34B / 32 weights = 8.5 bit/weight**.
- activation: 매 GEMM 호출마다 online으로 block_q8_0로 양자화 (per-32 amax 계산).
- GEMM 커널 (`nntr_ggml_impl_neon.cpp:2739` `nntr_gemm_q8_0x4_q8_0x4`):
  16(M)×4(N) 타일, SMMLA(`vmmlaq_s32`), weight 레지스터를 4개 act super-block이
  공유 (16x reuse). 구조적으로 잘 짜인 커널이다.

### 1.2 Attention 경로

`gemm_attention()`(flash/GEMM attention)은 의존하는 FP16 마이크로커널
(`hgemm_f16xf16_f32_fmlal`)이 **선언만 있고 정의가 없어서 모든 빌드에서 강제
비활성화** (mha_core.cpp:229-235, 커밋 475ecdd). 따라서 인코더 prefill(196토큰)도
디코드용 per-row dot 경로로 돈다:

```
compute_kcaches (per-row Q·K dot) → softmax_triangle → compute_fp16vcache_transposed (AXPY)
```

### 1.3 스레딩

- `ThreadManager::parallel_for`: **호출마다 정적 균등 분할 + futex 기반
  wake/barrier**. GEMM은 N(컬럼) 방향으로만 분할.
- activation 양자화는 parallel 구간 **밖에서 caller 스레드가 직렬 수행**
  (c33ead7에서 병렬화 시도 → f42d968에서 revert됨).

## 2. ONNX Runtime의 대응 경로 (비교 기준)

W8A32로 dynamic-quantize된 ONNX 모델에서 ORT는:

- **weight: per-channel(정확히는 per-column) int8 scale**, activation:
  **per-tensor dynamic quantize** (`DynamicQuantizeLinear` 또는 fused
  `DynamicQuantizeMatMul`).
- MLAS QGEMM: SDOT/i8mm 커널로 **K 전체를 int32로 누적한 뒤 epilogue에서 단 한 번**
  `acc × (a_scale × w_scale[col]) + bias`로 복원. inner loop는 load+SMMLA뿐.
- transformer 그래프 최적화: **Attention(fused MHA, packed sgemm), SkipLayerNorm,
  BiasGelu 융합** — elementwise 노드와 스케줄링 오버헤드가 크게 줄어든다.
- 스핀 대기 스레드풀 + 비용 모델 기반 분할.

참고로 ORT의 메모리 796MB (vs 359MB)는 ORT가 상당 부분을 fp32로 들고 실행하고
있음을 시사한다(초기화 시 dequant 또는 fp32 그래프 잔존). 즉 ORT 쪽 숫자는
"극한 최적화된 int8"이라기보다 "융합이 잘 된 MLAS 실행"에 가깝고, 그런데도
nntrainer가 근소하게 지는 것은 GEMM ALU가 아니라 **주변부**가 원인이라는 방증이다.

## 3. 원인별 상세 분석

### 3.1 per-block(Q8_0)이 커널에 강제하는 비용 — "포맷이 곧 커널 차이"

`nntr_gemm_q8_0x4_q8_0x4`의 K-블록(32개) 1회 반복, act super-block 1개 기준:

| 항목 | μop 수 |
|---|---|
| SMMLA (`vmmlaq_s32`) | 16 (= 512 MAC) |
| act load (`vld1q_s8`) | 8 |
| weight load (4 SB로 분할 상환) | ~2 |
| **per-block fold**: `vcombine`×4 + `vcvtq_f32_s32`×4 + `vmulq_n_f32`×4 + `vfmaq_f32`×4 + fp16 scale 변환 ~2 | **~18** |

per-block scale 때문에 int32 누적기(acc00..acc11)를 **32개 K마다 리셋하고 FP로
fold**해야 하므로(neon_impl.cpp의 `fold4`, 2773행), inner loop에서 SMMLA 대비
비-SMMLA SIMD μop이 약 60~70% 추가된다. per-channel 포맷이면 이 fold 전체가 K-loop
밖 epilogue 1회로 사라져 **inner loop μop이 이론상 ~1.5-1.7× 감소**한다
(fold ops가 SMMLA와 같은 SIMD 포트를 경쟁하므로 실효 이득은 그보다 작지만 유의미).

추가 비용:

- weight 스트림 **+6.25%** (블록당 fp16 scale 2B). GEMM(M=196)에선 캐시 재사용으로
  희석되지만, **GEMV(M=1, BERT)는 weight-스트리밍 바운드라 거의 그대로 시간에 반영**.
- 출력 저장이 `store4`(2790행)에서 tmp 배열 경유 **스칼라 4회 store** — 부차적.

단, per-block의 반대급부는 정확도다. Q8_0(per-32 group)은 per-tensor dynamic
activation + per-channel weight보다 outlier에 강하다. ORT와 같은 스킴으로 바꾸면
속도는 얻지만 SigLIP2/BERT의 정확도 마진 재검증이 필요하다.

### 3.2 activation online 양자화가 직렬 구간 (Amdahl)

`__ggml_q8_0_4x4_q8_0_GEMM` (bs_threadpool.cpp:887-893): M=196이면 49개의
`nntr_quantize_mat_q8_0_4x8` 호출을 **caller 스레드 혼자** 수행하고, 그동안
워커 풀은 논다. 인코더 1회당 양자화해야 하는 activation은
`12층 × 196 × (768×4 + 768 + 3072) ≈ 16.3M float` (Q/K/V/O 입력, FFN up/down 입력).
게다가 `nntr_quantize_mat_q8_0_4x8`(neon_impl.cpp:1423)는 byte 저장을
`vgetq_lane_s32` 스칼라 추출로 하는 느린 패킹이다. 대략 1-3ms 수준의 순수 직렬
구간으로 추정되며 53ms 중 상당 지분이다.

c33ead7이 정확히 이걸 병렬화했으나 f42d968로 revert됨 — revert 사유가 기능 문제가
아니라면(정황상 super-row당 일감이 작아 parallel_for 오버헤드가 이득을 상쇄),
**행 단위가 아니라 큰 청크 단위(스레드당 M4/T super-row 연속 구간) 분할**로
재시도할 가치가 있다. ORT는 per-tensor scale이라 reduction 1회 + 단순 scale 패스로
이 비용 자체가 작고 병렬화도 쉽다.

### 3.3 Attention이 GEMM이 아닌 per-row dot fallback — 인코더 최대의 커널 격차

SigLIP2 prefill의 attention 연산량은 `12층 × 2(QK+AV) × 196×196×768 ≈ 1.4 GFLOP`
(전체 ~34 GFLOP의 4%). 그런데 이 4%가 다음 커널로 실행된다:

- **Q·K** (`neon_impl_fp16.cpp:1935`): query row당 K row를
  `load_fp16_4_to_chunk`로 **fp16→fp32 변환해 tmp buffer 경유** — 같은 K row가
  query 수(196)만큼 **반복 변환**된다. dot은 **단일 float32x4 누적기**의 의존
  체인(FMA latency에 묶임, ILP 없음)이고, 출력마다 `sum / sqrt(head_dim)`으로
  **나눗셈+sqrt를 매 원소마다** 수행한다.
- **A·V** (`neon_impl_fp16.cpp:1853`): 누적을 **fp16으로**(`vfmaq_f16`) 하고
  (196개 확률 합산 시 정밀도도 손해), 누적기를 **호출마다 std::vector heap
  할당**한다. 이 함수가 query row × layer마다 불린다.

같은 연산을 ORT는 fused Attention 노드에서 **packed sgemm(다중 누적기, 블록킹)**
으로 처리한다. FLOPs 대비 효율 차이가 5-10×라서, 이 4%의 FLOPs가 nntrainer에선
수 ms(추정 3-8ms), ORT에선 ~1ms 수준이 된다. **+2.3ms 격차의 최대 단일 후보.**
`gemm_attention` 경로가 이미 코드에 있으나 FP16 커널 미구현으로 죽어 있는 상태
(475ecdd)라는 점이 아깝다.

### 3.4 per-op 실행 + barrier 오버헤드, 미융합 elementwise

nntrainer는 residual add, bias add, LayerNorm, GELU, softmax가 전부 개별 노드/
개별 `parallel_for`다. `parallel_for` 1회는 futex wake → 정적 분할 → barrier
(thread_manager.cpp:120-137)로, 모바일 SoC에서 회당 수~수십 µs. 인코더는 층당
GEMM 6회 + LN 2회 + GELU + softmax + residual 2회 + attention 커널 3회 ≈ 15회
× 12층 ≈ **180회 dispatch/이미지**. b9bef77이 LN(9-pass 텐서 op 체인 → fused
3-pass)과 GELU(ARM에서 NEON 계산 후 스칼라로 덮어쓰던 버그!)를 고쳐 큰 것은
제거됐지만, ORT의 SkipLayerNorm/BiasGelu 융합 대비 pass 수와 barrier 수는 여전히
많다.

### 3.5 BERT decoder (+21%)의 구성

BD_DIM=256, 4층, 4 head, FFN 1024 (bert_decoder.h:64-77) — 커널 하나하나가
마이크로초 단위라 **고정 오버헤드가 지배**한다:

1. **per-op dispatch**: 토큰당 self QKV/O + cross Q/O + FFN 2 + LN 3 + attention
   커널 6 ≈ 층당 ~15회 × 4층 + lm_head ≈ **~60회 parallel_for**/토큰. 회당 10µs만
   잡아도 0.6ms — 격차 1.5ms의 절반 가까이가 여기서 나올 수 있다.
2. **lm_head GEMV**: 30522×256, Q8_0로 **~8.3MB/토큰 스트리밍** — 메모리 바운드
   GEMV에서 per-block의 +6.25% bytes와 per-32 fold(`nntr_gemv_q8_0x4_q8_0`,
   neon_impl.cpp:2953 — 블록마다 vpaddq+cvt+fma fold)가 그대로 반영된다.
   ORT의 per-column int8(8.0 bit) + 순수 SDOT inner loop 대비 열세.
3. **cross-attention 병렬성**: enc_len=196을 도는 attention 커널이 KV head 수(4)
   로만 병렬화되는 decode 경로(mha_core.cpp:635-649) — 스레드 풀의 절반이 논다.

### 3.6 격차 요인 정리

| 요인 | 성격 | SigLIP2 기여 | BERT 기여 |
|---|---|---|---|
| attention per-row dot fallback (gemm_attention 비활성) | 커널 | **높음 (수 ms)** | 중간 |
| activation Q8_0 직렬 양자화 | 커널/스케줄링 (per-block 유발) | 높음 (1-3ms) | 낮음 |
| per-block fold가 GEMM/GEMV inner loop에 강제하는 μop | **포맷→커널** | 중간 | 중간 (lm_head) |
| weight +6.25% bytes | 포맷 | 낮음 | 중간 (GEMV) |
| per-op barrier/dispatch, 미융합 elementwise | 런타임 | 중간 | **높음** |

## 4. "per-channel vs per-block?"에 대한 직접 답변

- 순수한 "포맷 메타데이터" 차이(+6.25% bytes)만으로는 4.5%/21% 격차를 설명 못 한다.
- 그러나 per-block은 **커널 구조를 바꾼다**: int32 누적을 32-K마다 끊고 FP fold를
  삽입해야 하므로, ORT MLAS(per-channel, full-K int32 누적 + epilogue 1회) 대비
  inner loop가 원리적으로 무거워진다. 이건 구현을 아무리 잘해도 포맷이 남기는
  하한이다.
- 다만 **현재 측정된 격차의 주범은 포맷이 아니라 GEMM 바깥**이다: attention
  fallback, 직렬 activation 양자화, per-op 오버헤드. GEMM 커널 자체는 이미
  MLAS와 대등하다는 벤치가 커밋 로그(b9bef77)에 있다.

## 5. 성능 개선 백로그 (우선순위 리스트)

예상 회수치는 코드 구조 기반 오더 추정이며 실측으로 검증 필요. 표기:
효과 = 예상 회수 시간, 난이도 = 구현+검증 비용.

### P0 — 즉시 착수 권장 (격차의 대부분)

| # | 항목 | 대상 | 예상 효과 | 난이도 | 근거 코드 |
|---|---|---|---|---|---|
| 1 | **레이어별 실측 프로파일 계측** — 아래 모든 추정치를 실측으로 교체. simpleperf + per-layer timer | 공통 | (전제 작업) | 하 | `performance_metrics.h` |
| 2 | **attention GEMM화**: 미정의 `hgemm_f16xf16_f32_fmlal` 호출을 기존 `shgemm`/`custom_hgemm` 분기로 대체하고 `NNTR_ENABLE_GEMM_ATTENTION` 활성화. 인코더 prefill(196토큰)과 BERT cross-attn이 대상 | SigLIP2, BERT | 인코더 2~5ms, BERT 0.2~0.5ms | 중 | mha_core.cpp:229, 1058, 1168 |
| 3 | **activation Q8_0 양자화 병렬화 재시도**: revert(f42d968)된 row 단위 분할 대신 스레드당 연속 super-row 청크(M4/T개)로 분할해 dispatch 오버헤드 회피 | SigLIP2 | 1~3ms | 하 | bs_threadpool.cpp:887-893 |
| 4 | **`parallel_for` 소규모 일감 cutoff**: 현재 범위≥2면 무조건 futex wake+barrier. 총 일감 추정치가 임계 미만이면 main thread 직행 | BERT(주), 공통 | BERT 0.3~1ms | 하 | thread_manager.h:156-168 |
| 5 | **lm_head per-channel int8 실험**: N=30522 GEMV는 메모리 바운드라 +6.25% bytes와 per-32 fold가 그대로 시간. 출력층은 per-block 정확도 이득이 작아 포맷 전환 리스크 최소. 포맷 기여분 실측 분리도 겸함 | BERT | 토큰당 0.2~0.5ms | 중 | lm_head.cpp:154, neon_impl.cpp:2953 |

### P1 — 다음 단계

| # | 항목 | 대상 | 예상 효과 | 난이도 | 근거 코드 |
|---|---|---|---|---|---|
| 6 | **fallback attention 커널 개선** (P0-2를 안 하거나 보완으로): (a) K row fp16→fp32를 query마다 반복 변환하는 것 제거(층당 1회 사전 변환), (b) 단일 누적기 → 4개 언롤, (c) `/sqrt(d)` 나눗셈을 사전 곱으로, (d) AV fp32 누적 + per-call heap 할당 제거 | SigLIP2, BERT | 인코더 1~3ms | 중 | neon_impl_fp16.cpp:1935-1997, 1853-1932 |
| 7 | **`nntr_quantize_mat_q8_0_4x8` 패킹 개선**: byte 저장이 `vgetq_lane_s32` 스칼라 추출 — `vqmovn`+`vtbl` 계열 벡터 셔플로 교체 | SigLIP2 | 0.5~1ms | 중 | neon_impl.cpp:1423-1520 |
| 8 | **BERT decode attention 2차원 분할**: KV head 4개로만 병렬화 → head×row 분할로 풀 활용 | BERT | 0.1~0.3ms | 하 | mha_core.cpp:635-649 |
| 9 | **elementwise 융합**: bias+GELU, residual+LayerNorm 융합 노드 (ORT의 BiasGelu/SkipLayerNorm 대응). barrier 수와 메모리 pass 동시 감소 | 공통 | 인코더 0.5~1.5ms, BERT 0.2~0.5ms | 상 | layer graph 전반 |
| 10 | **정적 균등 분할 → 청크+atomic 카운터 동적 스케줄링**: big.LITTLE에서 균등 컬럼 분할은 가장 느린 코어가 barrier를 지배 | 공통 | 가변 | 중 | bs_threadpool.cpp:863-868, thread_manager.cpp |

### P2 — 실험/장기

| # | 항목 | 대상 | 비고 |
|---|---|---|---|
| 11 | per-channel(또는 K-chunk scale) 포맷 전면 실험: inner-loop fold 제거로 GEMM ~10-30% 이득 가능하나 정확도 재검증 필수. Q8_0 유지 시에도 블록 128/256 확대 실험 가치 | SigLIP2, BERT | 정확도-속도 트레이드오프 |
| 12 | `store4` 스칼라 store 제거 (tmp 배열 경유 4회 scalar → `vst1q_f32` 직행) | GEMM 공통 | 소폭 | neon_impl.cpp:2790-2795 |
| 13 | 커널 내 per-call heap 할당 제거 (`std::vector`/`new float[]`): softmax_row_inplace, compute_kcaches tmp, AV 누적기 | 공통 | 소폭 + jitter 감소 | neon_impl.cpp:1681, neon_impl_fp16.cpp:1876, 1939 |
| 14 | fmlal 기반 `hgemm_f16xf16_f32` 전용 커널 구현 (P0-2의 상위 호환) | SigLIP2, BERT | P0-2 대비 추가 이득은 실측 후 판단 |

### 정리

- SigLIP2 격차 +2.3ms: P0-2(attention) + P0-3(양자화 병렬화)만으로 역전 가능 전망.
- BERT 격차 +1.5ms: P0-4(cutoff) + P0-5(lm_head) + P1-8이 주 타깃.
- 포맷(per-block) 기인분은 P0-5 실험으로 실측 분리 후 P2-11 확대 여부 결정.

## 부록: 수치 가정

- SigLIP2 ViT-B/16, 224px → 196 tokens, hidden 768, FFN 3072, 12층, ~33 GFLOP.
- BERT decoder: dim 256, 4층, vocab 30522, enc_len 196.
- μop 카운트는 `nntr_ggml_impl_neon.cpp`의 커널 소스 기준 정적 계수이며 실측이
  아님. 기여도 추정(ms)은 오더 추정치로, 실측 프로파일로 대체되어야 한다.
