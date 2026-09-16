# 48 — decode 35 → 10 → 15.5 TPS: 무엇을 잃었고, 돌아가려면 무엇이 필요한가 (2026-09-16)

QS4CX(비-WH) 모델에서 decode는 ~35 TPS였다. QS4CX_WH(오프라인 WH bake, ION 아레나)로
바꾸자 10 TPS로 떨어졌고, 문서 47의 변경들로 15.5까지 돌아왔다. 이 문서는 그 격차를
숫자로 가르고, 35로 돌아가는 조건을 적는다. 기기 확인이 안 된 항목은 그렇게 표시했다.

## 1. 35 TPS의 정체 — 커널이 아니라 DDR 대역폭

QS4CX 모델에서 decode는 **22층 전부 CPU**였다. `tryMoeLayerOnAccelerator`의
`total_tokens <= 1 && !weights_wh` 게이트(문서 46 §37.3)가 M==1을 ARM으로 보낸다. HTP는
prefill에만 쓰였다. 토큰 하나가 읽는 바이트:

| | MB/token |
|---|---:|
| MoE 22층 × top-4 × (gate_up 3.67 + down 1.83) | 484 |
| conv 18층 (in_proj 6.3 + out_proj 2.1) + attn 6층 (q/k/v/o ≈ 2.5) | ≈ 170 |
| lm_head Q4_0 65536 × 2048 | 75 |
| **합** | **≈ 730** |

28.6 ms/token(35 TPS)이면 **25.5 GB/s** — ARM 4스레드가 DDR에서 뽑는 속도 그대로다.
decode는 어느 쪽에서 돌든 *바이트 수 ÷ 대역폭*이고, 35 TPS는 그 산술의 값이다.

**정정:** 문서 46 §47.1은 "CPU MoE 1.29 ms/layer"라고 썼는데, 28.6 ms 전체를 MoE로만
나눈 값이다. 730 MB 중 MoE는 484이므로 CPU MoE는 **≈ 0.88 ms/layer** (19 ms/token)다.
HTP가 넘어야 할 선은 1.29가 아니라 0.88이다.

## 2. WH로 넘어가며 잃은 것 — 셋이 곱해졌다

| 토큰당 ms | 35 TPS (QS4CX, CPU) | 10 TPS (WH, §42 첫 실행) | **15.5 TPS (지금)** |
|---|---:|---:|---:|
| MoE 22층 | ≈ 19 (CPU, 25 GB/s) | 59.7 (HTP 2.71/콜) | **42.1** (HTP 1.913/콜) |
| lm_head | ≈ 3 (repack된 Q4_0) | **25.7** | ≈ 3.4 (twin — `--profile`로 미확인) |
| FC·attn·norm 등 ARM | ≈ 7 | ≈ 14 | ≈ 19 (잔차, **미측정**) |
| **합** | **28.6** | **99** | **64.3** |

- **① MoE가 HTP로 갔다.** WH엔 CPU 커널이 없다(`FloatTensor::dot`이 throw, §35.5) — 선택이
  아니라 강제. 콜 1.91 ms는 CPU 0.88의 2.2배.
- **② lm_head 25.7 ms.** 묶인 임베딩이 repack 없는 정규 Q4_0으로 나와 행 단위 dot을
  탔다(§44). `6cfcc30` blocked twin이 고쳤고, decode 11.8 → 15.5의 대부분이 이것과 §49.2의
  DMA 순서다. **twin의 효과는 `--profile` 노드 표(`output_of_causallm`)로 아직 안 봤다.**
- **③ 나머지 ARM ≈ 19.** §43의 TYPE 합계(54 s 실행)를 토큰당으로 나누면 FC 9.4, mha 3.3,
  기타 2 ≈ 15에 미귀속 ~4. 순수 CPU 때의 7보다 크다 — 같은 가중치, 같은 코어인데. §45가
  본 big.LITTLE 과분할이 후보지만 **이 실행에선 안 쟀다.**

## 3. HTP MoE 콜 1.91 ms의 해부 — 벽이 셋이고, 하나만 고치면 소용없다

```
host 1913 = dsp 1348 + transport 565
dsp 1348  = HMX 1029 (mm 770 + acc 259)   ← 1행 계산에 64행 타일. 63/64가 빈 행
          + drain 173 (노출 DMA)
          + quant 43 + requant 42 + dequant 23 + 기타 ~40
DMA 실측  = first 1024 KB 58 us = 18 GB/s, 콜 평균 16.3   ← Gate 0c의 38.8이 아니다
```

| 벽 | 지금 | 없애면 | 이것만 없애면 |
|---|---:|---|---|
| **HMX 64행 타일** | 1.03 ms | HVX GEMV(M=1): 5.5 MB/expert를 4스레드 `vrmpy`로 ≈ 40 us → 4 expert 0.16 | DSP는 DMA 바닥 21.5 MB / 18 GB/s = **1.19** — 지금 1.35와 거의 같다 |
| **DMA 18 GB/s** | 1.19 ms 바닥 | 38 GB/s면 0.57 | HMX 1.03이 그대로 바닥 |
| **transport 0.57** | 22 × 0.57 = **12.5 ms/token** | dspqueue / 상주 워커로 ≤ 0.1 | DSP를 0으로 만들어도 12.5 — CPU MoE 19의 2/3 |

셋 다 하면 콜 ≈ 0.57 + 0.1 ≈ **0.7 ms → MoE 15 ms/token**, CPU의 19보다 빠르다.
둘만 하면 CPU에 진다. 이것이 §49.3의 갈림길이 아직 안 정해진 이유다.

## 4. 35 TPS 복구 산술

```
MoE 15 (세 벽 다 넘음) + lm_head 3.4 + ARM 나머지 ≈ 10 (FC 스레드 정리) ≈ 28.4 ms ≈ 35 TPS
```

가능은 하다. 조건은 **세 벽 전부 + ARM 정리**다. 한 개라도 빠지면 20대 TPS다.

## 5. 측정 — 방향을 정하는 셋

| | 무엇 | 코드 | 정하는 것 |
|---|---|---|---|
| **A** | `--profile` 빌드 decode: `output_of_causallm`(twin, ≈3.4 기대), `fully_connected`, `mha_core`, 기타 TYPE 합계 | 0 | ②·③의 실제 값. FC가 9.4면 §45의 스레드 과분할부터 |
| **B** | `NNTR_HTP_PROFILE=3` (같은 입력 5회, 최소값) | 0 | transport 565가 300 이하로 떨어지면 콜 사이 wake/클록 → 상주 워커. 안 떨어지면 마샬링/캐시 유지보수 → prebound + 버퍼 정리 |
| **C** | 아레나 프로브 확장 — 고립 상태에서 (i) 선형 1 MB, (ii) 2D 16 KB×64 @ 56 KB 스트라이드(가중치 청크 모양), (iii) 8 KB×64, (iv) 워커 2~4개가 각자 `dmstart` (엔진이 스레드별이면 병렬로 2배). 버스 투표는 별도 skel로 on/off | 작음 | DMA 18 vs 38.8의 원인: 디스크립터 모양 / 엔진 수 / DVFS. 문서 46 §50.7의 투표 실험은 짝 청크와 겹쳐 판독 불가였다 |

C가 결정적이다: HVX GEMV는 DMA가 30+가 아니면 0.15 ms짜리 이득이라 **C 결과 전에 만들지
않는다.**

## 6. 코드 순서 (측정 뒤)

1. DMA 경로 — C가 가리키는 것(모양 / 다중 엔진 / 투표)
2. HVX GEMV (M ≤ 6): `hexkl_micro` 없이 VTCM의 WH 타일을 직접 읽는 u8×i4 내적. u8×i4 int32
   합은 순서 무관 정확이라 HMX 경로와 **int32 누산기 비트동일** 검사가 가능하다
3. transport — B가 가리키는 것. 상주 워커면 Hexagon SDK `dspqueue`(ARM↔DSP 저지연 큐)가
   직접 만드는 것보다 싸다

## 7. 대안 — WH 타일을 읽는 CPU dot

Q4_0 dot에 타일 순열 하나를 얹은 커널. prefill은 HTP 그대로, decode만 CPU로 → 예전
35 TPS 경로가 그대로 돌아온다. 가장 확실하고 싸다. 다만 "decode에 CPU를 쓰면 안 된다"가
**제품 조건**(NPU 필수)이면 값이 0이고, **WH라서 못 쓴다**는 뜻이면 이게 답이다.
어느 쪽인지 정해져야 §6의 우선순위가 정해진다.

## 8. 기록

- 이 문서의 CPU 숫자(0.88 ms/layer, 730 MB/token, 25.5 GB/s)는 §29의 35.18 TPS와 모델
  shape에서 **유도**한 것이지 같은 실행에서 잰 것이 아니다. 문서 46 §49.5 Run C(QS4CX +
  `NNTR_MOE_HTP_DECODE=1`)가 같은 실행에서 재는 방법이다.
- HTP 숫자는 2026-09-15의 `808ad92` skel 실행(문서 47 §11.1).
