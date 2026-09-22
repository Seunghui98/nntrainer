# 51 — 블록 단위 콜: dense FFN 융합(구현)과 conv 블록 상주(설계) (2026-09-21, dense 코드 완료·기기 미측정)

문서 50이 닫은 결론: **행렬곱 하나를 보내면 포장(고정 0.4 ms + 캐시 유지 + 스테이징 + quant/dequant)이
곱셈보다 크다.** in_proj 콜 5.25 ms 중 HMX는 1.49(29%), wq는 16%. 이기는 방식은 하나 — 행렬곱
여러 개와 그 사이 원소 연산을 **한 콜**에 묶어 포장을 한 번만 내는 것. MoE가 콜당 39 GFLOP으로
그렇게 이겼다. 이 문서는 그 방식을 나머지 두 블록에 적용한다.

## 1. dense FFN — 구현됨: "4개 expert짜리 MoE 콜"

layer 0·1의 dense FFN은 up·gate [2048×7168], down [7168×2048]. 문서 50 §3.5에서 FC 3콜(up, gate,
down)로 보냈더니 콜 3개 + 중간값 25 MB 왕복 + ARM SwiGLU로 층당 17.7 ms — ARM(≈20)과 본전이었다.

**새 커널 없음.** 관찰: intermediate 7168을 1792씩 4조각으로 자르면 조각 하나가 정확히 이 모델의
**expert 형상**(gate_up [2048×3584], down [1792×2048])이고, MoE 레이어 커널은 이미

- expert마다 gate_up → SwiGLU → down을 VTCM 안에서 돌리고(중간값이 칩을 안 건넘),
- `out[row] += w · res[row]`로 expert 출력을 **누적**하며(`memset` 후 scatter-add),
- 에필로그 숨기기·팩 백그라운드·DMA 프리페치가 전부 들어 있다(문서 47).

그러니 dense FFN = **"expert 4개가 모든 행을 가중치 1.0으로 받는 MoE 콜"**이다. 조각 c의 down 출력을
더하면 그게 곧 intermediate 전체에 대한 합이다.

```
row_index  = [0..M-1] × 4        row_count = {M, M, M, M}        row_weight = 1.0
h_gu[c] = WH( [gate[:, c·w..] | up[:, c·w..]] )   [K × 2w]   colsum = 열 슬라이스 (K 전체 합)
h_dn[c] = WH( down[c·w.., :] )                    [w × N]    colsum = 그 w행에 대해서만 다시 계산
```

down 조각의 colsum을 다시 계산하는 이유: 커널의 zero-point 보정 `acc − zp·colsum`이 **그 콜의 K**
(= w행)에 대한 합을 원한다. gate_up 조각은 열 슬라이스라 K가 그대로다.

### 1.1 코드

| 어디 | 무엇 |
|---|---|
| `layers/dense_ffn_layer.{h,cpp}` (신규, type `dense_ffn`) | up·gate·down **세 가중치를 파일 순서 그대로** 가진 레이어. M>1이고 ops에 융합 콜이 있으면 한 콜, 아니면(decode) CPU: dot·dot·`swiglu_det`·dot — `createMlp`의 세 레이어와 바이트 동일 |
| `compute_ops.h` | `supports/gemm_q4_0_dense_ffn_fp32`, `register_q4_0_dense_ffn` |
| `htp_compute_ops.cpp` | `get_or_register_dense`: Q4_0x4 세 개 → int8 RM → 조각별 gate_up/down 조립 → `registerRm`(아레나 우선, 힙 폴백; 50 §3.3의 조각 등록을 헬퍼로 뺌) → `invokeMoeLayer` |
| `transformer.cpp createMlp` | `dense_ffn_engine: htp`면 FC 3개 + swiglu 대신 `dense_ffn` 하나 |
| `transformer.cpp repack_weight` | 아레나 매핑 뒤 등록 + M=512 워밍업 (FC와 같은 이유) |

가중치 42 MiB(2층 × 4쌍 × 5.25) — in_proj를 끄면 아레나 슬랙 144에 들어간다. in_proj까지 켜면
108 + 42 = 150 > 144 → 일부 힙(≈100 여유). 프로파일에서는 MoE 행(`K=2048 N=2048 M>1`)에 합산돼
보인다 — calls가 22 + 2 + 워밍업.

### 1.2 기대

MoE 콜의 블록당 비용(252 us) × (444/64 → 7블록 × 4조각 = 28블록) ≈ 7.1 + 콜 고정 ≈1.5 → **dsp ≈8.6
ms**, host ≈9.7, 스테이징 0.9 → **≈10.5 ms/층**. ARM ≈20, FC 3콜 17.7. 2층 **−19 ms**. 크지 않다 —
dense가 2층뿐이라서다. 값어치는 **conv 블록(18층)에 같은 논리를 쓰기 전의 검증**이다: 커널 무변경으로
"블록 한 콜"이 텍스트를 지키고 예측대로 나오는지.

게이트: 텍스트가 dense를 CPU에 둔 실행과 같은가(재양자화가 두 층 늘었다), `NNTR_HTP_PROFILE=2`의
MoE 행 calls가 25(22+2+1)인가, decode 불변인가.

### 1.3 첫 기기 실행 (2026-09-21) — 돈다; 콜 비용은 아직 분리 안 됨

config C(`conv_in_proj` + `dense_ffn`), 8스레드, PROFILE=2, 2회:

```
prefill 896 / 888 ms   decode 20.63 / 20.55 TPS   등록 1478 = MoE 1408 + in_proj 54 + dense 16 (전부 아레나)
  K=2048 N=2048 M>1  calls=26  rows=11680  blocks=1126     ← 22 MoE + 2 dense + 워밍업 2; 블록 +91 (dense 56 + 워밍업 32)
                     host 15722–16347 us/call  transport 1371–1945
  N=6144 in_proj     19콜  host 4044–4147   transport 851–964
  M==1 MoE decode    host 1509  transport 157–159                  ← 50 §3.6 수정 유지
```

- **융합 콜이 돈다**: calls 26, blocks +91, decode 불변. 텍스트는 또 바뀌었다(세 번째 변형; 3문장
  요약은 정상). 재양자화 층이 늘 때마다 토큰이 흔들리는 것이라, "텍스트 동일" 게이트는 NPU 경로에
  양자화 지점을 더할 때는 쓸 수 없다 — **로짓 차·perplexity 게이트**가 필요하다(§3).
- **dense 콜 비용은 이 실행에선 못 뗀다**: MoE 행(K=2048 N=2048)에 합산된다. MoE 콜을 15.7~16.3으로
  잡으면 dense 3콜(워밍업 포함)이 34~48 ms → **11~16 ms/콜**, 기대 10.5. 폭이 커서 판정 불가 →
  프로파일에 `M>1 dense` 행을 따로 뒀다(같은 형상이라 키만으로는 못 가른다). 다음 실행부터 나온다.
- prefill 888~896: all-on(943)보다 −50이지만 in_proj만 켠 835(스테이징 수정 전)보다 느리다. 수정 후
  기대(≈795 − dense 15 ≈ 780)보다 **+110** — 50 §3.7의 "ARM 잔여가 09-16보다 느리다"가 그대로다.
  decode도 합은 20.6으로 09-16과 같지만 구성이 다르다(MoE 콜 −8.6 ms/token, ARM +9). **A 실행 없이는
  못 가른다.** → 50 §3.8의 A가 갈랐다: 오늘 기기 자체가 921~1013이고, C는 그보다 −33.
- 3회째(PROFILE=2, 951 ms, decode 20.53): MoE 행이 여전히 calls=26으로 합산 — 바이너리가 행 분리 커밋
  이전이다. in_proj 콜 4.35 ms(dsp 3.21, transport 1.13 — 앞 실행 0.85~0.96보다 높다, 드리프트),
  decode 콜 1.51/transport 157 불변. dense 콜은 여전히 12~21 ms 범위 추정 → **재빌드 뒤 B로 다시**.

### 1.4 B 실행 — dense 콜 10.4 ms, 기대 10.5 그대로 (2026-09-21)

config B(`dense_ffn_engine`만), 8스레드, PROFILE=2, 재빌드 후 1회:

```
prefill 1000 ms   decode 20.50 TPS   등록 1424 = MoE 1408 + dense 16 (전부 아레나)
  M>1 dense   calls=3  rows=1400 (444×2 + 워밍업 512)  blocks=88 (28+28+32)
              host 10428 us/call  dsp 10015 (96%)  transport 413
              [mm 5605  acc 1915  dequant 931  requant 534  stage 356  quant 272  gather 162  scatter 83]
  M>1 MoE     calls=23  host 17029  dsp 14998  transport 2031        ← 오늘 드리프트(앞 실행 15.7~16.3)
  M==1 MoE    host 1513  transport 158                                ← 불변
```

| | 기대 (§1.2) | 측정 |
|---|---:|---:|
| dsp / 콜 | 8.6 | 10.0 |
| host / 콜 | 9.7 | 10.4 |
| 블록당 mm | — | 191 us (MoE 행 8637/45.2 = 191, **같다**) |
| 블록당 dsp 전체 | 252 | 342 (MoE 행 332) |

- **"블록 한 콜" 논리가 예측대로 나온다.** 블록당 비용이 MoE 콜과 같고(mm 191, 전체 ≈340), dense 콜 =
  28블록 × 340 + 고정 0.5 ≈ 10.0. §1.2의 252는 mm+acc만 센 값이었고 dequant·requant·stage를 더하면 340.
  커널 무변경으로 다른 블록을 실어도 비용이 블록 수에 비례한다 — conv 블록(§2) 산술의 근거.
- **손익: 층당 ARM ≈19.5(39.1 GFLOP @ 2 TFLOPS) → HTP 10.4 + 스테이징 0.4 = −8.7, 2층 −17 ms.** 벽시계
  (1000)는 오늘 A의 921~1013 안이라 보이지 않는다 — §1.2가 예고한 대로 dense는 2층뿐이라 작다.
- transport 413은 in_proj 콜(850~1130)의 절반 — 활성 3.6 MB만 오가서다. MoE 콜 2031은 gather/scatter의
  row_index까지 실어서.
- 텍스트 네 번째 변형(A와 다르고 C와도 다름). 재양자화 지점 2층 추가 = 토큰 변화, §3의 로짓 게이트 필요.
- decode 불변 — dense 레이어의 M=1 CPU 경로가 세 FC와 바이트 동일하다는 것의 실측 확인.

**판정: dense 융합은 유지한다.** 이득은 −17이지만 비용이 예측과 맞는 것이 conv 블록 −115의 전제였고,
그 전제가 섰다.

## 2. conv 블록 상주 — 설계 (§2.7에 구현이 어떻게 달라졌는지)

### 2.1 무엇을 한 콜로

```
x ─rms_norm─▶ in_proj [T×2048]×[2048×6144] ─split─▶ a, b, c
                                            a⊙c ─conv1d(L=3, depthwise, causal)─▶ y
                                            b⊙y ─out_proj [T×2048]×[2048×2048]─▶ o
x + o ─▶ 다음 층
```

콜 입력: norm 뒤 x [T×2048] f32 (3.6 MB). 콜 출력: o [T×2048] f32 (3.6 MB). 중간값 a·b·c·y(각 T×2048)는
VTCM에 산다. rms_norm과 residual add는 ARM에 둔다(원소 연산, 0.1 ms; 넣으면 −0.3 ms·층이지만
`_det` 판이 필요해 Phase D로).

### 2.2 VTCM 배치 (8 MiB)

in_proj 가중치 6 MiB + out_proj 2 MiB = 8 MiB — **둘 다 상주는 불가.** 시간차 재사용:

```
offset   크기      내용
0        128 KB    act   64행 × 2048 u8 (AH 타일)
128 KB   2 MiB     W 슬롯 A  ← in_proj 조각 0/2 (N 2048씩 3조각) → 나중에 out_proj
2.1 MiB  2 MiB     W 슬롯 B  ← in_proj 조각 1
4.1 MiB  1.5 MiB   a, b, c   64행 × 2048 × f32 × 3
5.6 MiB  512 KB    y / gated 64행 × 2048 f32 (a⊙c → conv1d → ⊙b, 제자리)
6.1 MiB  128 KB    mid       64행 × 2048 u8 (out_proj 입력, requant)
6.2 MiB  512 KB    staging   acc 2벌 (256 KB × 2)
6.7 MiB  512 KB    res       out_proj 결과 64행 × 2048 f32
합 ≈7.2 MiB
```

in_proj를 **N 조각 3개(2048씩)**로 슬롯 A/B에 번갈아 DMA(문서 50의 조각과 같은 단위), 조각 c의
결과가 곧 a/b/c다. 3조각이 끝나면 슬롯 A에 out_proj 2 MiB. 블록(64행)마다 이 순서를 돌면 가중치
DMA가 블록마다 8 MiB — **행 블록당 8 MiB DMA는 7블록 × 8 = 56 MB/층, 38 GB/s에 1.5 ms** — 숨길 수
있지만 무겁다. 대안: **M 전체를 먼저 in_proj(조각별로 전 블록 순회) → conv1d → out_proj** 순으로 돌면
가중치 DMA는 층당 8 MiB로 끝나지만 중간값 a·b·c가 T×2048×3×4 = 10.9 MB라 VTCM에 안 들어간다 →
중간값을 **u8로 requant해 DDR(DSP 힙) 왕복**(3 × 0.9 MB) 하거나, conv1d의 캐주얼 L=3 특성을 써서
**64행 블록 + 이전 2행만 유지**하면 된다. 후자가 맞다: conv1d(L=3)는 행 t가 t−1, t−2만 보므로
블록 경계에 2행 겹침만 두면 블록 단위 스트리밍이 성립한다. 그러면 순서는 블록마다
`in_proj(3조각) → gating → conv1d(2행 캐리) → gating → out_proj`, 가중치는 블록마다 다시 DMA.
**블록당 8 MiB DMA 1.5 ms vs 블록 계산 ≈(7168+2048... ) 타일 ≈ 0.35 ms** — DMA가 4배 길다. 안 된다.

그래서 **가중치 상주 + 활성화 스트리밍**으로 뒤집는다: in_proj 조각 c(2 MiB)를 올린 채 **모든 블록**을
돌려 a/b/c 중 하나를 u8(requant)로 DSP 힙에 내려놓는다(T×2048 u8 = 0.9 MB ×3). 3조각 뒤 out_proj를
올리고, 블록마다 힙에서 a·b·c 64행씩 읽어 gating → conv1d(2행 캐리) → gating → requant → out_proj →
res → out. 가중치 DMA 층당 8 MiB(0.2 ms), 활성화 왕복 층당 ≈5.4 MB(DSP 힙, 캐시드, 0.15 ms).
정밀도: a·b·c를 u8로 내리는 것이 새 양자화 지점이다 — **gating(a⊙c)은 f32에서 하고 그 결과와 b를
u8로** 내리면 지점이 둘(gated, b). `_det` SwiGLU와 같은 급의 게이트가 필요하다.

### 2.3 새 HVX 연산 (전부 원소 연산, ~50줄씩)

| 연산 | 형상 | 비고 |
|---|---|---|
| split 3 + a⊙c | [64×6144] → [64×2048] | dequant 에필로그에 융합 가능 (열 범위가 곧 a/b/c) |
| conv1d L=3 depthwise causal | [64×2048], 가중치 [3×2048] f32 | 블록 경계 2행 캐리, decode는 state 2행 |
| b⊙y | [64×2048] | |
| requant (f32 → u8 행별) | 이미 있음 (`hvx_quant_u8`) | |

### 2.4 콜 하나의 파이프라인 (T=444, 7블록)

```
Phase 1  in_proj 조각 c=0..2 (가중치 2 MiB 상주):  블록 b=0..6: HMX 64×[2048×2048] → dequant(+a⊙c 융합, c==2에서) → u8 → 힙
Phase 2  out_proj (2 MiB 상주):                    블록 b=0..6: 힙에서 gated,b → conv1d → ⊙b → requant → HMX → dequant → out
```

HMX 타일: in_proj 7168 + out_proj 2048 = 9216/블록 × 7 = 64.5K 타일 × 18.9 ns = **1.2 ms**; acc_read
(192+64)×7×0.37 = 0.66; 원소 연산 ≈0.5; quant/requant ≈0.6; 전송 3.6 MB×2 → 0.9 + 스테이징 0.9 →
**콜 ≈4.8 ms/층**. 지금(in_proj HTP 5.2 + out_proj·conv·gating ARM ≈6) ≈11.2 → **−6.4 × 18 ≈ −115 ms**.
CPU 원본(in_proj+out_proj+conv 등 ≈11.3)과 비교해도 같은 폭.

### 2.5 게이트와 순서

1. **호스트 참조 구현**(C, 비트 동일 판) + 유닛테스트 — conv1d·gating·requant 3개. 1주.
2. **skel 커널** `nntr_hvx_conv_block` (IDL 추가, 가중치 핸들 2개 + conv 가중치 f32 6 K) — 문서 46의
   MoE 커널 구조(워커 풀, staging 2벌, DMA 링)를 그대로 재사용. 2주.
3. **앱 배선**: `conv_block` 레이어(in_proj·conv·out_proj 가중치 3개, 파일 순서 유지) + `conv_block_engine`
   config + 로드 등록·워밍업 — §1의 `dense_ffn`과 같은 골격. 3일.
4. 게이트: 텍스트 동일(u8 지점 둘), decode 불변(conv state 2행은 ARM 경로가 그대로 가짐 — decode는 CPU),
   `NNTR_HTP_PROFILE=2` 새 행 `conv_block` host ≤ 5 ms.

**decode**: M=1은 CPU 그대로(conv state 캐시가 ARM 레이어에 있다). 상주 커널은 prefill 전용.

### 2.6 안 하는 것

- rms_norm·residual을 콜에 넣기 — `_det` norm이 필요, 이득 0.3 ms/층. Phase D.
- MoE와 conv 블록을 한 콜로(레이어 전체 상주) — 문서 45 Phase D. 이 문서의 두 블록이 그 전 단계다.

### 2.7 구현 (2026-09-22) — 코드 완료, 호스트 체크 통과, 기기 미측정

§2.2의 두 번째 안(가중치 상주 + 활성화 스트리밍)을 그대로 짓되, 중간값을 u8로 내리지 않는다. 열
슬라이스 3개(a, b, c)를 **두 개씩** 올리면 되기 때문이다:

```
Phase 1  슬롯 A = W_a, 슬롯 B = W_c (2 MiB씩):  블록 b=0..6: HMX a·c 타일을 쌍으로 → dq(a)·dq(c)를 f32로 곱해 g [M×C] → DSP 힙
Phase 2  슬롯 A = W_b, 슬롯 B = W_out:          블록 b=0..6: HMX b → dq(b)를 VTCM z에 → z *= conv1d(g)[행 mb−2..] → requant → HMX out_proj → dq → out
```

- **양자화 지점은 x→u8, (b⊙y)→u8 둘뿐** — in_proj·out_proj를 따로 HTP에 보낼 때와 같다. §2.2가 걱정한
  "a·c를 u8로 내리는 새 지점"이 없다. g는 f32 3.6 MB로 힙을 한 번 쓰고 한 번 읽는다.
- **conv1d는 HVX 원소 연산**: `z *= (w0·g[t] + w1·g[t−1]) + w2·g[t−2]`, FMA 없이 곱 3·합 2 순서 고정
  (`hvx_conv_gate_f32.c`). 블록 경계 캐리가 없다 — g 전체가 힙에 있으니 앞 두 행을 그냥 읽는다.
- **decode 상태**: 콜이 g의 마지막 2행을 `state_f32`로 돌려주고 레이어가 conv_state에 넣는다. decode(M=1)는
  CPU 커널(`causal_depthwise_conv1d_k3_decode`) 그대로 — 기존 6개 레이어 체인과 바이트 동일.
- 가중치 DMA 층당 8 MiB 한 번. 블록당 HMX 타일 = a 64 + c 64 + b 64 + out 64 = 256 n-tiles × 64 k-tiles,
  §2.4의 9216과 같다. VTCM 합 5.27 MiB (staging 32 타일 2벌 포함).
- MoE 커널의 스크래치·DMA 푸시·백그라운드 pack·MM 타이머를 `hexkl_moe_*`로 내보내 그대로 쓴다. 블록당
  비용이 §1.4의 340 us와 같은 급이면 콜 ≈ 7블록 × 4 타일군... 계산으로 §2.4의 **≈4.8 ms**가 기대값.

| 어디 | 무엇 |
|---|---|
| `hmx/hexkl_conv_block.{h,c}` | 레이아웃 + 두 단계 루프. `hexkl_moe_scratch` 공유 |
| `hvx/hvx_dequant_i32.{h,c}` | `hvx_dq_mul_worker`: 두 가중치의 타일 쌍 dequant·곱 (a⊙c) |
| `hvx/hvx_conv_gate_f32.{h,c}` | z *= conv1d(g) |
| `test/htp/nntr_hvx.idl`, `nntr_hvx_mm_u8i4.c` | `mm_u8i4_conv_block(_timed)`; stage_us는 MoE 콜과 같은 열, SWIGLU 열 = conv gate |
| `test/htp/host/conv_block_host_check.c` | 스칼라 참조와 비트 동일 (M=150, K=64, C=544, N=1056; M=1; 잘못된 핸들 형상; LFM2 형상 레이아웃). stub은 `hvx_scalar_stubs.c`로 빼서 MoE 체크와 공유 |
| `htp_compute_ops.cpp` | `get_or_register_conv`: in_proj를 `get_or_register_fc`로 등록하면 K=2048에서 슬라이스가 정확히 2048열 = a, b, c (ponytail: 다른 형상은 직접 슬라이스 필요). 프로파일 행 `M>1 conv` |
| `layers/conv_block_layer.{h,cpp}` (type `conv_block`) | in_proj·conv·out_proj 세 가중치를 파일 순서로. M>1이고 ops가 있으면 한 콜, 아니면 CPU: dot → a⊙c → conv(k3 / decode) → ⊙b → dot |
| `lfm2_causallm.cpp` | `conv_block_engine` / `conv_block_htp_layers`; 켜지면 conv_norm → `conv_block` → residual → ffn |
| `transformer.cpp repack_weight` | 아레나 매핑 뒤 등록 + M=512 워밍업 |

**측정 config (D)** — 스위치는 conv 블록과 dense만. `conv_in_proj_engine`·`conv_out_proj_engine`·
`attn_proj_engine`은 넣지 않는다 (conv 블록이 켜진 층에서는 어차피 무시되고, attention 층의 것은 손해):

```json
"conv_block_engine": "htp",
"dense_ffn_engine": "htp"
```

IDL이 바뀌었으니 **세 가지를 다시 빌드**한다: `nntrainer/tensor/htp_backend/generate_stub.sh`(ARM stub),
`test/htp/build.sh`(DSP skel → `libnntr_hvx_skel.so` push), 앱. 하나라도 빠지면 첫 콜이 AEE_EBADPARM.

게이트(§2.5의 4): (1) 텍스트가 3문장 요약으로 정상인가 — 토큰은 바뀐다(양자화 지점 추가). (2) decode 불변.
(3) `NNTR_HTP_PROFILE=2`의 `M>1 conv` 행: calls 19 (18 + 워밍업), host ≤ 5 ms/콜이 목표, SWIGLU 열이
conv gate 시간. (4) prefill: 같은 날 A 대비 **−100 안팎**이면 §2.4의 산술대로.

### 2.8 D 실행 — conv 콜 4.9 ms, 기대 4.8; prefill 732~741 (2026-09-22)

config D(`conv_block_engine` + `dense_ffn_engine`), 8스레드, PROFILE=2, 2회:

```
prefill 741 / 732 ms  (599 / 607 TPS)   decode 20.64 / 20.96 TPS   등록 1496 = MoE 1408 + conv 72 + dense 16
  M>1 conv    calls=19  rows=8504  blocks=134 (18×7 + 워밍업 8)
              host 4902 / 4894   dsp 4333 / 4354 (88%)   transport 569 / 541
              [mm 2120  acc 671  swiglu(=conv gate) 534  stage 240  quant 223  gather 200  requant 137  dequant 79  drain 0.7+77]
  M>1 dense   host 10246 / 10367   transport 337 / 442
  M>1 MoE     host 16858 / 17191   dsp 15012 / 14997   transport 1846 / 2194
  M==1 MoE    host 1507 / 1501   transport 152 / 148          ← 불변
```

| | 기대 (§2.4) | 측정 |
|---|---:|---:|
| 콜 host | 4.8 | **4.9** |
| HMX 타일 | 64.5K × 18.9 ns = 1.2 | mm 2.12 (114.7K 타일 × 18.5 ns — §2.4가 a·c·b·out 중 둘을 빠뜨렸다; 타일당은 천장) |
| acc_read | 0.66 | 0.67 |
| 원소 연산 | 0.5 | 0.53 (gate) + 0.14 (requant) |
| 전송+스테이징 | 0.9 + 0.9 | 0.55 + 0.24 |
| prefill | C 888~896 − 115 ≈ 775 | **732~741** |

- **돈다, 산술대로.** 게이트 4개 통과: 3문장 요약 정상(토큰은 또 바뀜), decode 불변, calls 19, host < 5 ms.
- 같은 날 C(888~896) 대비 **−150**, 오늘 A(921~1013) 대비 −180~−280, 09-16의 848 대비 −110. §2.4의
  −115보다 큰 것은 C가 in_proj 콜 4.35 × 18 = 78을 이미 내고 있었고 그것까지 conv 콜에 흡수됐기 때문.
- conv 콜 안에서 HMX가 노는 곳: gate 534 + requant 137 + dequant 꼬리 79 = **750 us/콜 (17%)** — 전부
  동기 구간. 블록 간 파이프라인(다음 블록의 b 행렬곱·게이트·requant를 이번 블록의 out_proj 아래에; z·mid
  2벌, VTCM +0.6 MiB)으로 숨길 수 있다 → 18콜 × 0.75 = **−13 ms**.
- **MoE 콜의 transport가 이상하다**: 같은 커널·같은 크기 버퍼인 dense 콜이 337~442인데 MoE 콜은
  1846~2194. 22콜 × ≈1.5 = **33 ms**가 설명 없이 나간다. 차이는 콜 길이(15 vs 10 ms)와 DMA량(165 vs
  21 MB)뿐. 가설: `RPC_POLL_QOS`의 latency=100 us 이후 인터럽트 대기로 떨어지는 경로가 긴 콜에서
  느리다. 실험: (a) 짧은 프롬프트(~200 토큰, MoE 콜 ≈7 ms)에서 transport가 dense 수준으로 떨어지는지,
  (b) `htp_backend.cpp`의 latency를 10000으로.
- 이제 prefill의 구성: HTP 콜 host 합 ≈ 481 (MoE 372 + conv 88 + dense 21) + 스테이징 memcpy ≈18 +
  **ARM 잔여 ≈ 236** (attention 6층의 q/k/v/o·core, norm, router, lm_head). ARM 잔여가 다음 미지수.

### 2.9 정확도 — 첫 DIFF는 커널을 증명했고, 텍스트는 지표가 아니다 (2026-09-22)

D의 텍스트가 A보다 헐거워 보여 `NNTR_CONV_BLOCK_DIFF`(두 경로 다 돌려 층별 SNR)와 `_SHADOW`(모델엔
CPU 경로 값)를 넣고 돌렸다:

```
DIFF   : 18층 중 16층 SNR 146~154 dB, 2층 80~86 dB (layer 3, 9);  state 전 층 정확히 동일 (999)
SHADOW : 16층 147~156, 1층 80.6 (layer 17);  state 동일.  텍스트는 D와 다른 변형, decode 20.6
```

- **150 dB는 "같은 수를 두 번 계산했다"는 뜻이다.** 참조 경로가 진짜 CPU가 아니었다: 이 레이어의
  `engine=htp`라 `dot()`이 in_proj·out_proj를 HTP FC 콜로 보냈고(프로파일에 `N=6144` 18콜, MoE 행
  +18콜이 그 증거), 참조 = HTP FC 둘 + CPU 원소연산 = 융합 콜과 같은 양자화 지점. 그래서 이 비교가
  증명한 것은 **커널이 맞다**는 것이다 — 두 경로의 차이는 dequant·gate의 f32 연산 순서뿐이고, g(state)는
  비트 동일.
- 80 dB짜리 층은 u8 경계 뒤집힘이다: z의 한 원소가 f32 반올림 차이로 양자화 경계를 넘으면 그 원소 오차가
  행 범위의 1/255 → 콜 SNR 80. 실행마다 다른 층에서 나는 것도 그래서다(SHADOW는 하류 입력이 달라진다).
- **텍스트는 정확도 지표가 아니다.** SHADOW는 150 dB(상대 1e-7) 차이의 값을 넣었는데도 토큰이 바뀌었다.
  greedy 512 토큰은 반올림 잡음에도 갈라진다. 문서 50·51의 "텍스트 변형" 관찰 전부, 그리고 attn_proj·
  conv_out_proj를 "정확도 위험"으로 뺀 판단도 같은 근거 위에 있었다 → 전부 재평가 대상.
- 고침: compare 모드의 참조는 `nntrainer::gemm_q4_0`을 직접 부른다(A config의 FC 경로와 같은 함수).
  다음 DIFF는 융합 콜 vs 진짜 CPU(ggml Q8_0 활성 양자화)의 SNR을 준다 — 양자화 지점 두 개(x→u8, z→u8)의
  실제 비용. 기대는 30~45 dB.
- 게이트로 남는 것: **로짓 기준**. 같은 프롬프트의 prefill 로짓(444×V)을 두 config에서 덤프해 위치별
  argmax 일치율과 평균 KL을 본다. 텍스트 대신 이것으로 판정한다.

### 2.10 진짜 CPU 참조: 10~17 dB — 손실이 진짜다 (2026-09-22)

참조를 `nntrainer::gemm_q4_0`(A config의 FC 경로)으로 바꾼 DIFF:

```
out  10.6 ~ 16.7 dB (layer 0이 16.7, 나머지 10.6~13.5)     state(g) 15.6 ~ 27.4 dB
```

- 11 dB = RMS 오차 28%. 반올림이 아니라 양자화 손실이고, **g(state)부터 이미 16~27 dB**라 in_proj 한 단
  (x→u8 + int4 열당 가중치)에서 대부분이 난다. out_proj 한 단이 더해져 11.
- 후보는 둘. (1) 활성화 per-row u8: 행 하나의 2048값을 min~max 255단계로 — outlier 채널이 있으면 보통 값은
  몇 단계밖에 못 받는다. (2) 가중치 레시피: 파일의 Q4_0(32개마다 스케일)을 로드 시 열당 int4로 **다시**
  양자화한다(`htp_qs4cx_from_q4_0x4`, 헤더 자체가 "real requantization"이라 적어 뒀다). 이중 양자화이고
  MoE 전문가(오프라인 qs4cx)는 이 단계가 없다.
- 이 손실은 conv 블록 커널의 것이 아니다. in_proj·out_proj·q/k/v/o·dense를 HTP로 보낸 모든 FC 경로가
  같은 레시피라 같은 손실을 안고 있었고, 지금까지 아무도 CPU 대비 SNR을 재지 않았다(§2.9). **HTP FC
  경로 전체의 문제**로 격상.
- 가르기: DIFF에 "활성화만" 참조를 더했다 — CPU 경로에 x와 y만 DSP의 per-row u8 규칙으로 가짜 양자화
  (가중치는 Q4_0 그대로). 그 SNR이 HTP와 비슷하면 활성화가 원인, 훨씬 높으면 소거법으로 가중치 레시피.
  행별 outlier 비율(max|x|/rms)도 같이 찍는다.
- 고칠 길, 원인별:
  - 활성화: SmoothQuant식 채널 스케일 — x의 채널 c를 1/s_c로, 가중치 행 c를 s_c로. 가중치는 어차피 로드
    시 다시 양자화하니 행 스케일을 접는 건 공짜고, x쪽은 conv_norm의 gamma에 접으면 된다. s_c는 prefill
    한 번의 채널 max 통계.
  - 가중치: 로드 시 변환을 버리고 **오프라인 qs4cx 양자화**(전문가와 같은 도구, f32에서 직접) — 모델 파일
    변경. 또는 u8i8 커널(`hexkl_mm_u8i8_dma.c`)로 int8 열당 가중치 — 가중치 2배라 conv 블록의 상주
    배치(2 MiB 슬롯 2개)는 깨진다.

### 2.11 갈렸다: 가중치 레시피가 16~18 dB를 먹는다 (2026-09-22)

```
            HTP (실측)         act-u8 only (CPU, 가중치 Q4_0 그대로)      outlier max/rms
out         10.6 ~ 16.7        26.5 ~ 30.8                              x 4~14   y 10~34
g (state)   15.6 ~ 27.4        33.5 ~ 51.5
```

- 활성화 두 지점만의 비용은 out 27~31 dB. HTP는 11~17. **차이 16 dB가 가중치 레시피** — 파일의 Q4_0
  (32개마다 스케일)을 열당 int4 하나로 다시 양자화하는 `htp_qs4cx_from_q4_0x4`. 이 함수의 규칙은
  scale = 15/(rmax−rmin), 반올림, [−8, 7] 클램프 — 영점 없이 범위 스케일을 쓰므로 치우친 열은 한쪽이
  잘린다(KleidiAI qs4cx 관례 그대로).
- 활성화 outlier 비율(max/rms) 10~14는 per-row u8이 보통 값에 ~20단계를 주는 정도 — 30 dB 바닥의 원인.
  두 번째 수리 대상.
- **주소 공간이 int8을 막는다**: 열당 int8은 그 손실을 거의 없애지만(Q4_0 격자를 int8이 거의 정확히
  담는다) conv 가중치가 144 → 288 MiB. 아레나 슬랙 144 + 힙 100 = 244 MiB 안에 안 들어간다(32비트 DSP
  주소 공간, 50 §3.2). in_proj만 int8(216)도 넘친다.
- 그래서 실제 후보는 int4를 유지한 채 격자를 더 잘 쓰는 쪽: SmoothQuant(α=½) — 입력 채널 k를 1/s_k,
  가중치 행 k를 s_k. 가중치는 어차피 로드 시 다시 양자화하니 접는 비용이 없고, x쪽은 콜 안에서 채널
  곱 한 번(또는 conv_norm gamma — 단, decode의 CPU in_proj는 원본 Q4_0라 gamma는 못 건드린다 → 커널 안).
- 짓기 전에 잰다: DIFF에 f32 에뮬레이션을 더했다 — 현재 규칙 / 대칭 int4 / smooth+int4 / int8(상한 참조),
  각각 act u8과 함께. "현재 규칙" 행이 실측 11과 맞으면 나머지 행을 믿는다.

### 2.12 에뮬레이션 1차: SmoothQuant 탈락, int8이 전부 회복, 격자가 원인 (2026-09-22)

```
                cur int4/ch   sym int4/ch   smooth(α=½)+int4   int8/ch      HTP 실측    act-u8 only
out (18층)      10.6~16.7     9.0~15.2      8.6~18.1           26.3~30.6    10.6~16.7   26.5~30.8
g               16.8~27.4     15.3~25.7     15.4~24.1          33.1~48.5    15.6~27.4   33.5~51.5
```

- **에뮬레이션이 맞다**: `cur` 행이 실측과 0.1 dB 이내(out 18층 전부).
- **SmoothQuant는 반대 방향이다.** 활성화 난이도를 가중치로 옮기는데 병목이 가중치라 더 나빠진다(−1~−3).
- **sym은 −1.5**: 15/(rmax−rmin) 규칙은 원인이 아니다.
- **int8은 act-u8 only와 같다**: 가중치 손실 16 dB는 전부 "채널당 스케일 하나인 int4 격자"의 것. Q4_0의
  32-블록 스케일이 담던 K 방향 변화를 채널당 스케일 하나가 못 따라간다.
- 다음 에뮬레이션(int4 유지): (1) **행 평탄화** — 가중치 입력 채널 k를 m_k = max|w[:,k]|^α로 나누고 x 채널
  k에 곱한다. SmoothQuant의 정반대 방향으로, 활성화 쪽 14 dB 여유를 쓴다. 가중치 통계만 쓰니 보정 불필요,
  로드 시 변환에 공짜로 접힌다. (2) **K-그룹 int4** — 그룹(256/128)마다 스케일. DSP에서는 그룹마다 별도
  핸들 + acc_read → 행렬곱 단계 비용 ×2~3.

### 2.13 에뮬레이션 2차: 행 평탄화 탈락; 그룹 int4는 규칙을 잘못 써서 다시 (2026-09-22)

```
            cur        rowflat α=1   rowflat α=½   blk256 (cur 규칙)   blk128 (cur 규칙)   int8
out         10.6~16.7  7.7~15.3      10.0~16.3     11.7~19.3           10.5~20.5           26.3~30.6
```

- **행 평탄화 탈락**: α=1은 −3~−4, α=½는 ≈−0.5. K 방향 변화를 활성화로 옮기면 per-row u8이 그만큼 더
  잃는다 — 활성화의 14 dB 여유는 그런 식으로는 못 쓴다.
- **그룹 int4가 그룹이 작을수록 나빠졌다** — 물리적으로 불가능하니 에뮬레이션의 잘못이다: 로드 시 규칙
  (15/(rmax−rmin), 영점 없음)을 32~256개짜리 그룹에 쓰면 min/max가 비대칭이라 한쪽이 잘린다. 채널
  2048개에서는 대칭에 가까워 안 보이던 것. Q4_0은 대칭(max|w| 기준). 그룹 후보는 대칭 규칙으로 다시 재고,
  **B=32 행이 act-u8-only(≈28)와 같아야 검산이 된다.**

### 2.14 에뮬레이션 3차: 참조가 Q4_0라서 SNR은 여기까지 — 게이트는 perplexity로 (2026-09-22)

```
            cur int4/ch   blk32(대칭)   blk64   blk128   blk256   int8/ch    act-u8 only
out         10.6~16.7     15.9~19.0    14.3~17.6  13.2~16.4  11.8~17.0  26.3~30.6  26.5~30.8
```

- **blk32가 act-u8-only(≈28)로 안 돌아온다.** 같은 32블록이라도 격자가 다르면(Q4_0은 step=max/8, 에뮬은
  max/7) 값이 격자 사이에 떨어져 int4 잡음(≈17~19 dB)만큼 어긋난다. 참조 자체가 Q4_0 양자화 모델이라,
  **Q4_0 격자를 정확히 담는 레시피(int8) 말고는 어떤 재양자화도 이 바닥 아래로 못 내려간다.**
- 그러면 "HTP vs CPU 16 dB"의 구성은: ~11 dB는 "서로 다른 int4 격자"의 잡음(모델 품질과 무관), ~5 dB는
  채널당 격자가 32블록보다 거친 몫, ~2 dB는 이중 양자화. 레시피끼리의 상대 비교는 유효하다(같은 바닥):
  blk256 +1~2, blk64 +4~5, 행 평탄화·SmoothQuant는 −. 그룹 int4는 DSP에서 그룹 수만큼 acc_read가 늘어
  (G=8이면 acc 0.67 → 5.4 ms/콜) 얻는 것보다 잃는다.
- **SNR-vs-Q4_0로는 더 못 가른다.** 모델 품질은 f32 원본 대비인데 기기엔 f32가 없다. 참조 없이 재는 지표가
  perplexity다: 프롬프트 토큰의 teacher-forced NLL을 prefill에서 잰다.
- `NNTR_PPL=1`: 앱이 프롬프트 id를 tied lm_head에 넘기고, lm_head가 prefill의 모든 행(마지막 행뿐 아니라)
  로짓을 스크래치에 계산해 −log softmax[다음 토큰]을 합산 → `[PPL] prompt tokens=N nll/token=… ppl=…`.
  행당 ~7 ms(어휘 65536 GEMV), 프롬프트당 몇 초. 모델 출력(마지막 행)은 건드리지 않는다.
- 판정: 같은 프롬프트로 A(CPU Q4_0)와 D(conv 블록 + dense HTP)의 ppl. 같은 급이면 채널당 int4는 모델 품질을
  못 해친 것이고 지금 레시피로 간다. D가 뚜렷이 나쁘면 이중 양자화를 없애는 오프라인 qs4cx(전문가와 같은
  도구)가 첫 수. 444 토큰 하나는 작으니 긴 텍스트(1000+ 토큰) 프롬프트로 한 번 더.

### 2.15 perplexity: conv 블록 HTP 58.5 vs CPU 59.4 — 손실 없음 (2026-09-22)

```
D  (conv 블록 HTP + dense HTP)            ppl 58.51   nll/token 4.069   443 토큰
D + SHADOW (conv 블록 CPU 값, dense HTP)  ppl 59.41   nll/token 4.085
```

- HTP 쪽이 1.5% 낮다. 표본 하나(프롬프트 1개)라 "더 좋다"는 아니지만, **"깨진다"는 증거는 없다.** 두 경로는
  같은 가중치의 서로 다른 int4 양자화(Q4_0 32블록 vs 채널당)이고 어느 쪽도 원본에 더 가까울 이유가 없다.
  §2.10의 11 dB는 두 격자가 서로 다르다는 것이었지 품질 손실이 아니었다.
- 이로써 §2.9~2.14의 결론이 닫힌다: 텍스트 인상은 지표가 아니고(§2.9), SNR-vs-Q4_0는 int4 잡음 바닥에서
  멈추며(§2.14), 판정은 ppl로 한다. **conv 블록 HTP 경로는 그대로 간다.** attn_proj·conv_out_proj를
  "정확도 위험"으로 뺐던 판단도 같은 방법으로 다시 볼 수 있다.
- 남은 것: 순수 A(스위치 없음)의 ppl은 기록용으로 한 번, 그리고 1000토큰 이상 긴 텍스트로 한 번. DIFF의
  레시피 에뮬레이션은 역할을 다했다 — 코드는 남기되 기본은 꺼져 있다.

정리(2026-09-22): §2.10~2.13의 레시피 에뮬레이션(act-u8-only 참조, 4~6개 가중치 레시피, outlier 비율)은
역할을 다해 코드에서 뺐다. 남는 것은 `NNTR_CONV_BLOCK_DIFF`/`_SHADOW`(CPU 대비 SNR, MoE의 L2_DIFF와 같은
급의 판별기)와 `NNTR_PPL`(판정 지표)이다. 이 절들의 숫자는 이 문서에만 남는다.

### 2.16 Phase 2 파이프라인 (코드, 호스트 체크 통과) + poll 창 실험 1차 (2026-09-22)

- **conv 커널 Phase 2를 한 블록 앞서 가게 바꿨다.** 블록 n의 b 행렬곱·dequant(z[n&1]) 뒤에 그 블록의
  gate·requant를 **백그라운드 레인**에 넣고(16행 단위 gate 4개 + requant 1개, 레인이 순서를 보장), 블록
  n−1의 out_proj를 그 아래에서 돌린다. z·mid·requant 파라미터 2벌(VTCM 5.27 → 5.9 MiB). 동기 구간이던
  gate 0.53 + requant 0.14 + dequant 꼬리 0.08 = 0.75 ms/콜이 HMX 아래로 들어간다 → 기대 −0.6 ms/콜,
  18콜 −11 ms. SWIGLU 열은 이제 워커 시간(숨은 일)이고, 노출은 REQUANT 열(wait_bg)로 읽는다.
- `NNTR_HTP_POLL_US=10000`은 드라이버가 **거부**했다(`qos_mode=1` = PM QoS로 후퇴). PM 모드의 MoE transport
  2012 us — 가설 검증이 아니다. 다음은 1000, 5000.
- 이 실행의 텍스트는 이전 D 실행들과 바이트 동일하다(결정적). "정확도가 깨졌다"는 인상은 §2.15의 ppl이
  이미 답한 것과 같은 출력을 다시 본 것이다.

### 2.17 A/D/SHADOW의 ppl 분해 — 비용은 conv가 아니라 dense다 (2026-09-22)

같은 프롬프트(443 토큰), `NNTR_PPL=1`:

| config | conv 블록 | dense FFN | ppl | nll/token |
|---|---|---|---:|---:|
| A | CPU | CPU | **57.00** | 4.043 |
| D + SHADOW | CPU | HTP | **59.41** | 4.085 |
| D | HTP | HTP | **58.51** | 4.069 |

차분으로 읽으면:

```
dense FFN을 HTP로      : 57.00 → 59.41   +2.41  (+4.2%)
conv 블록을 HTP로      : 59.41 → 58.51   −0.90  (−1.5%)
합 (A → D)             : +1.51  (+2.6%)
```

- **conv 블록은 비용이 아니다.** 켜면 오히려 내려간다(§2.15가 본 것이 이 차분이다). 두 경로가 같은
  가중치의 다른 int4 격자라 어느 쪽이 원본에 가까운지는 우연이고, conv는 운 좋은 쪽에 떨어졌다.
- **dense FFN이 +2.4를 낸다.** dense도 같은 로드 시 재양자화(`htp_qs4cx_from_q4_0x4`)를 쓰지만 형상이
  다르다: down은 [7168 × 2048]을 1792행 청크로 잘라 **청크마다 열당 스케일 하나**를 다시 잡는다. K가
  7168에서 1792로 줄어도 스케일 수는 그대로라, 잘린 청크의 열이 원래 열보다 범위가 좁으면 격자가
  어긋난다. 그리고 이 열들은 SwiGLU 출력을 받는 쪽이라 outlier가 많다.
- **판정: dense는 끈다.** §1.4에서 dense 융합의 이득은 −17 ms(오늘 기기 편차 92 ms 안)였고, 비용이
  ppl +4.2%다. conv 블록의 −150 ms는 그대로 남는다. 켜고 싶으면 먼저 오프라인 qs4cx(전문가와 같은
  도구, f32에서 직접)로 dense 가중치를 다시 뽑아 이중 양자화를 없애야 한다.
- 확인용 한 실행: `conv_block_engine`만 켠 config의 ppl이 57 근처인가. 그러면 분해가 맞고 D 대신
  그 config가 기본이 된다.

### 2.18 conv만 켠 config — ppl 57.12 vs A 57.00, 그리고 파이프라인이 먹었다 (2026-09-22)

`conv_block_engine`만, `NNTR_PPL=1`, PROFILE=2 (등록 1480 = MoE 1408 + conv 72):

```
ppl 57.1179  nll/token 4.045        (A 57.0017 / 4.043,  D 58.5066 / 4.069)
M>1 conv  calls=19  host 4652  dsp 4002  transport 650
          [mm 2061  acc 682  swiglu(hidden) 3255  dequant 389  gather 287  stage 255  quant 227  requant 0.3]
```

- **정확도 닫힘.** conv만 켜면 A와 0.2% 차이(57.12 vs 57.00). §2.17의 분해가 맞았다: 비용은 dense의
  것이었고 conv 블록은 공짜다. **기본 config는 `conv_block_engine: "htp"` 하나다.**
- **Phase 2 파이프라인이 먹었다**: dsp 4361 → 4002 (−359 us/콜), host 4875 → 4652 (−223). requant 대기가
  139 → 0.3으로 사라졌고(완전히 숨었다), gate는 SWIGLU 열에 3255 us의 **워커 시간**으로 나온다 — 3 워커가
  HMX 아래에서 돌린 숨은 일이고, 콜 벽시계 4.0 ms보다 크다는 것이 숨었다는 증거다. 18콜 −6.5 ms.
- 대신 dequant가 79 → 389로 올랐다. 백그라운드 레인이 워커를 잡고 있는 동안 전경 에필로그가 늦게
  집히는 것 — MoE 커널이 tail 경로에서 본 것과 같은 현상(문서 47 §21.1)이고, 그쪽은 그래서 껐다. 여기서는
  순이득(−359)이 남아 유지한다.
- 프로파일 고침: conv 행의 SWIGLU는 워커 시간이라 mm 잔차에서 빼면 안 된다(빼서 `rest` −3197이 나왔다).
  kind==2에서만 제외하고 열 이름에 `(hidden)`을 붙인다.

### 2.19 정정: dense가 비싼 이유는 청크가 아니라 down의 깊이다 (2026-09-22)

§2.17에서 "dense는 청크마다 열당 스케일을 새로 잡는다"고 썼는데 코드가 그렇지 않다.
`get_or_register_dense`는 `htp_qs4cx_from_q4_0x4`를 **전체 [7168 × 2048]에 한 번** 돌리고, 청크는 이미
양자화된 바이트를 행으로 자르면서 colsum만 다시 더한다(스케일과 int4 값은 공유). 청크는 양자화 중립이다.

원인은 **재양자화가 덮는 K의 깊이**다. 열당 스케일 하나가 커버하는 값의 수:

| 가중치 | K | Q4_0의 열당 스케일 수 | 변환 후 | 비율 |
|---|---:|---:|---:|---:|
| conv in_proj | 2048 | 64 | 1 | 64:1 |
| conv out_proj | 2048 | 64 | 1 | 64:1 |
| dense up/gate | 2048 | 64 | 1 | 64:1 |
| **dense down** | **7168** | **224** | **1** | **224:1** |

dense down만 224:1이고 나머지는 64:1이다. 게다가 down의 입력은 SwiGLU 출력이라 outlier가 많다.
MoE 전문가의 down은 K=1792이고 애초에 오프라인에서 그 깊이로 열당 int4가 정해졌으니(`htp_qs4cx_from_packed`
= 비트 재배치, 재양자화 아님) 이 손실이 없다. **원래 hexkl 경로와 dense의 차이는 커널이 아니라 이 한 가지다.**

되살리는 길도 그래서 하나다: dense 가중치를 오프라인에서 f32로부터 qs4cx로 뽑는다(전문가와 같은 도구).
그러면 이중 양자화가 사라지고, 열당 스케일이 224:1을 덮는 문제도 양자화기가 f32를 보고 정하므로 남는
손실은 전문가와 같은 급이 된다.

### 2.20 poll 창: 5000 us에서 prefill 716, decode 23.7 TPS (2026-09-22)

conv만 켠 config, 세 실행 모두 `qos_mode=2`:

| POLL_US | prefill | decode | MoE transport | conv transport | decode 콜 transport |
|---:|---:|---:|---:|---:|---:|
| 100 (이전 기본) | 748 / 594 TPS | 20.61 | 2190 | 650 | 158 |
| 1000 | 737 / 602 | 23.22 | 1648 | 511 | 126 |
| **5000** | **716 / 620** | **23.75** | **1131** | **321** | **83** |

- **가설이 맞았다.** 100 us 뒤 인터럽트 대기로 떨어지는 경로가 콜마다 0.3~1 ms를 먹고 있었다. 5000에서
  MoE 콜 −1.06 × 22 = −23, conv −0.33 × 18 = −6 → prefill −32. decode 콜 −75 us × 22 = −1.65 ms/token →
  **20.6 → 23.7 TPS (+15%)**. 10000은 드라이버가 거부(§2.16). 기본값 5000으로 바꿈; 5000~10000 사이의
  상한은 아직 안 찾았다(9000, 8000 순으로 한 번씩).
- 비용: 대기 중 코어 하나가 최대 5 ms 스핀. prefill·decode 모두 콜이 동기라 그 코어는 어차피 노는
  코어다. 발열은 이후 긴 실행에서 본다.
- MoE 콜 transport가 여전히 conv의 3.5배(1131 vs 321)다. 같은 버퍼 크기, 같은 poll 창. 남은 차이는
  MoE 콜만 넘기는 힙 배열 셋(row_index·row_weight 7 KB씩, row_count·핸들 128 B)이다 — ION이 아니라
  FastRPC가 매 콜 복사·매핑한다. ION 스테이징 버퍼로 옮겨 재는 것이 다음 실험(맞으면 −0.5~0.8 × 22).
- 오늘 A(921~1013) 대비 **−205~−297**, 09-16의 848 대비 −132. 세 실행의 텍스트는 바이트 동일(§2.18의
  ppl 57.12 config).

### 2.21 전부 켬: prefill 618, ppl 62.1 — attn proj는 −88, dense는 −10에 같은 ppl 값 (2026-09-22)

`conv_block_engine` + `dense_ffn_engine` + `attn_proj_engine`, poll 5000, `-qs4cx-wh` 디렉터리, 등록 1520:

```
prefill 642 / 618 ms (692 / 718 TPS)   decode 24.6 TPS   ppl 62.09  (A 57.00, conv만 57.12, D 58.51, dense만 59.41)
  K=2048 N=512  M>1       calls=12  host 843/콜   dsp 571  transport 272   [quant 299 dequant 75 acc 43]     ← k, v
  K=2048 N=2048 M>1       calls=36  host 11140/콜 dsp 10176 transport 964  ← MoE 22 + q/o 12 + 워밍업 2가 한 행에 섞임
  M>1 dense               calls=3   host 10677/콜 transport 605
  M>1 conv                calls=19  host 4493/콜  dsp 3999  transport 494  (conv만 켠 실행에선 321)
  M==1 (decode MoE)       calls=11264  host 1417/콜  transport 86
  arm staging memcpy 28.1 ms (634.7 MB)
```

- **conv만(716) 대비 −98.** q/k/v/o가 CPU에서 80~115(문서 50 §2 추정) → HTP 34(q/o ≈24 + k/v 10)라
  ≈ −88, dense ≈ −10. 문서 50 §3.7의 "전부 켬 +22"는 poll 100에서 24콜의 transport가 이득을 먹은 것이고,
  poll 5000에서 뒤집혔다.
- **ppl 62.09 = A +8.9%.** dense 몫 +4.2%(§2.17)와 attn proj 몫 ≈ +4.5%가 얹힌다. 텍스트(3문장 요약)는
  세 실행 동일하고 정상. **거래 조건이 다르다**: attn proj는 −88 ms에 +4.5%, dense는 −10 ms에 +4.2% —
  dense는 끄는 것이 맞다(conv + attn = 예상 prefill ≈ 630, ppl ≈ 59.6). 사용자 판단은 "문장 정상이면 OK".
- **프로파일 결함**: `addInvoke`(일반 FC)가 kind 0으로 MoE 행과 같은 키를 써서 q/o(2048×2048)가 MoE 행에
  섞였다. kind 3 `M>1 FC`로 분리 — 다음 실행부터 q/o 콜당 비용이 그대로 읽힌다.
- **k/v 콜 843 us의 구성**: quant 299 + transport 272 + dsp의 나머지 ≈ 270. 연산이 아니라 고정비다. q/k/v
  셋이 같은 x(normed residual, 3.6 MB)를 각자 양자화해서 각자 전송한다 — 한 콜(N=3072)로 묶으면 콜당
  quant·transport·스테이징 두 벌이 사라진다(≈ −1.2 ms × 6). attention 블록 콜(§2.20 4번)의 첫 걸음.
- **conv transport 321 → 494, dense 605.** 같은 커널·같은 poll인데 등록 1480 → 1520, 콜 수 +24에서 올랐다.
  문서 50 §3.6의 "스테이징 크기 비례 캐시 유지비"로는 설명이 안 된다(크기 동일). 열이거나 FastRPC 핸들
  수 비례 — 식힌 뒤 conv만 켠 실행 한 번이면 가른다.
- prefill 618의 구성(콜당 × 콜 수): MoE ≈340 (55%) · conv 81 · q/o ≈24 · dense 21 · k/v 10 · 스테이징 28 ·
  **ARM 잔여 ≈ 115~140 (≈20%)**. ARM 잔여는 attention core 6층(444² 어텐션 ≈ 10 GFLOP f32 → 그것만으로
  ≈100)이 대부분일 것 — 다른 담당자 몫.

### 2.22 MoE 커널: down을 한 블록 뒤로 — 노출된 requant·마지막 에필로그를 HMX 아래로 (2026-09-22, 코드, 호스트 체크 통과)

§2.21 4번의 MoE transport가 아니라 2번(커널 안의 노출 2.2 ms/콜)부터. 전경 레인은 한 번에 한 잡이고, 잡은
자기 submit과 자기를 거두는 wait 사이에 HMX 배치 발행이 있어야만 숨는다(문서 47 §14). 블록 순서
GU(n)→DN(n)에서는 블록마다 세 잡이 숨을 곳이 없었다: 마지막 gate_up 에필로그(requant가 gate_off 전체를
필요로 해서), requant 자체(동기, 풀 왕복 2번), 마지막 down 에필로그 + 그 뒤의 scatter.

바꾼 것 — `hexkl_mm_u8i4_moe.c`의 블록 루프 전체:

- **발행 순서 `GU(0) rq(0) | GU(1) DN(0) | GU(2) DN(1) | … | DN(N−1)`.** DN(n−1)의 배치가 GU(n)의 마지막
  에필로그와 rq(n)을 덮고, GU(n+1)의 첫 배치가 DN(n−1)의 마지막을 덮는다.
- **rq(n)은 DN(n−1)의 첫 에필로그 잡 안에, scatter는 모든 down 에필로그 안에** (`moe_dn_worker`: 타일 하나
  dequant → 그 32열을 바로 out_c에 scatter-add; 라운드로빈으로 requant 유닛 4개(16행씩)도 같은 잡에). 블록당
  잡 수 = 배치 수. 노출은 콜의 양 끝(rq(0) 앞의 에필로그 하나, 마지막 down 에필로그 하나)뿐.
- 그러려면: mid 2벌(+112 KB VTCM, 레이아웃 7.03 MB), mid 행 파라미터 2벌, down[e]는 GU(e) 앞이 아니라
  DN(prev) 발행 뒤에 push(안 그러면 DN(prev)가 읽는 버퍼 위에 떨어진다), 스테이징 패리티를 gate_up·down
  배치에 걸쳐 하나로. gate_off는 1벌로 충분 — rq(n)은 GU(n+1)의 첫 에필로그 submit 전에 거둬진다.
- 비트 동일: 타일의 32열은 그 타일만 쓰고, 원소마다 expert마다 곱셈-덧셈 한 번이 블록 순서대로 —
  행 분할이 만들던 바이트 그대로. 호스트 체크 `MOE KERNEL MATCHES REFERENCE`(mismatches 0), 블록 5, DMA 12 KB.
- 기대: requant 0.73~1.0 + dequant 꼬리 ~0.4 + scatter 0.24 → **−1.2~1.5 ms/콜 × 22 ≈ −30**. REQUANT 열은
  이제 "rq(n)을 실은 잡의 노출"이다. 실측이 답: MoE 행의 requant/dequant/scatter가 0 근처로 가야 한다.
  DN 에필로그가 배치보다 길어지면(scatter가 타일당 64번의 1벡터 호출이라) 대신 REQUANT가 오른다 — 그때는
  타일 모양 scale-add 하나(ponytail 주석).

### 2.23 q/k/v를 한 콜로: `qkv_layer`에 q_norm·k_norm까지 접음 (2026-09-22, 코드)

§2.21 3번. k/v 콜 843 us 중 quant 299 + transport 272가 고정비였고, q/k/v 셋이 같은 x(3.6 MB)를 각자
양자화해 각자 보냈다. 한 콜이면 6층 × ≈1.2 ms.

- **레이어**: 이미 있던 `qkv_layer`(q/k/v 가중치 셋, `Tensor::dot`의 벡터 오버로드 → `gemm_q4_0_batch_fp32`
  한 콜)에 `feature_size`(head_dim)·`epsilon`을 더해 **q_norm·k_norm을 접었다.** 파일 순서가 q, q_norm, k,
  k_norm, v라서 셋만 묶으면 로더가 gamma를 k 가중치로 읽는다 — 그래서 gamma 둘도 이 레이어의 가중치다
  (FP32, `ReshapedRMSNormLayer`와 같은 요청·같은 `rms_norm_wrt_width_fp32_intrinsic` + gamma 곱).
  출력 셋: q_normed, k_normed, v. 원본이 `__restrict`라 in-place가 안 되어 q_raw/k_raw 텐서 둘을 둔다.
- **LFM2 attention은 엔진과 무관하게 이 레이어를 쓴다**(fully_connected 3 + reshaped_rms_norm 2 → 1). CPU
  경로는 같은 GEMM 셋 + 같은 norm 커널이라 A config도 그대로이고, 호스트 유닛 테스트
  (`unittest_causallm_lfm2`, `_lfm2_moe`)가 이 경로를 탄다. decode(M=1)는 벡터 dot이 CPU로 떨어지므로 불변.
- **HTP 쪽** `gemm_q4_0_batch_fp32`: M=1 전용이던 것을 prefill 모양으로 — 행 청크(fcMaxRows)와 가중치별
  목적지(`invokeLayer(..., &N, &dsts)`; 힙 out_cat + 두 번째 복사 5 MB/콜 제거). 프로파일 행은 `K=2048
  N=3072 M>1 FC`로 따로 찍힌다. 로드 시 등록은 transformer.cpp의 FC 분기에 `qkv_layer` 타입 추가.
- 기대: k/v 12콜(10 ms) + q 12콜(≈24) → 6콜 ≈ 6 × (2.0 + 0.55) ≈ 15 → **−15~−19 ms**. transport 한 번,
  quant 한 번, HMX는 같은 타일 수.
- **attention 블록 콜(§2.20 4번)의 첫 걸음**: 이 레이어가 그 콜의 호스트 쪽 자리다 — core 담당자의 커널이
  오면 `qkv_layer`의 HTP 경로가 q/k/v 대신 attention 출력을 받아오면 된다.

### 2.24 실측: 585 ms, ppl 62.09 그대로 — 두 변경 합 −33 (2026-09-22)

전부 켬 config, poll 5000, §2.22 + §2.23 적용:

```
prefill 586 / 585 ms (758 TPS)   decode 24.2 TPS   ppl 62.0916 (§2.21과 소수점까지 동일 -- 산술 불변 확인)
  M>1 MoE   calls=23  host 14448  dsp 13822  transport 627   [mm 9102 acc 2954 stage 379 drain 251+53 quant 269 dequant 243 rest 190 alloc 147 gather 84 requant 62 scatter 36]
  M>1 dense calls=3   host 9560   dsp 9118   transport 442   [requant 57 dequant 286  -- 529/923에서]
  M>1 conv  calls=19  host 4356   dsp 3928   transport 427
  M>1 FC N=3072 (qkv) calls=6  host 2278  dsp 1808  transport 471  [quant 338 dequant 403 acc 239 drain 82 | mm ≈746]
  M>1 FC N=2048 (o)   calls=7  host 1724  dsp 1347  transport 377  [quant 342 dequant 268 acc 163 drain 59 | mm ≈515]
  staging memcpy 25.5 ms (593 MB)
```

- **§2.22 MoE 파이프라인**: requant 727 → 62, dequant 689 → 243, scatter 그대로 → 콜당 ≈ −1.1 ms(기대 −1.2~1.5).
  dense도 같은 커널이라 −1.1. MoE transport가 964 → 627로 내려온 것은 설명 없음(콜이 짧아진 것과 같이 움직임).
- **§2.23 qkv**: k/v 12콜 10 ms + q 12콜 ≈24 → N=3072 6콜 13.7 ms, **−20**. dsp 1808 중 quant 338 + dequant 403
  + acc 239 + drain 82 = 1.06 ms가 HMX 밖 — FC 커널은 에필로그 파이프라인이 없다.
- 구성(585): MoE 22 × 14.45 = **318 (54%)** · conv 78 · dense 19 · qkv 14 · o_proj 10 · 스테이징 25 · **ARM 잔여 ≈ 120**.
- **MoE 콜의 정체는 HMX 활용률이다.** mm 9.1 + acc 3.0 = 12.1 ms/콜 = 14.45의 84%, 그리고 둘 다 블록 수에
  비례한다: 1776 행(444 × top-4)에 HMX 블록 47 = 3008 행 슬롯 → **59%**. expert마다 64행 블록 하나는
  차니까(평균 55행) 나머지는 패딩이고, HMX는 행 수와 무관하게 64행을 계산한다. M=444에서는 구조적 상한;
  100%면 콜 ≈ 9.5, prefill ≈ −110. 프롬프트가 길수록 저절로 좋아진다(1024 토큰 → expert당 128행 = 꽉 찬 2블록).
  HVX 꼬리(≤16행)는 행당 ≈13 us라 19행 위로는 HMX가 이기므로 꼬리 문턱을 올리는 것도 답이 아니다.
- 남은 소프트웨어 지렛대는 전부 작다: gate_up[e+1] 청크를 GU(e)의 배치 c 발행 직후 청크 c 자리에 push하면
  drain 251 → ~50 (−4); FC 커널의 quant 340(단일 스레드로 보임)·dequant 노출 (−5); MoE transport 627 vs FC 400
  (−4); 스테이징 25(레이어 입출력을 ION에 두면 0, 텐서 할당자 훅 필요, 중간 규모).

### 2.25 §2.24의 1·2·3 (2026-09-22, 코드, 호스트 체크 통과)

세 개를 한 커밋에. 전부 산술 불변이라 ppl 62.09 그대로여야 한다.

1. **gate_up[e+1]을 청크 단위로 일찍** (`hexkl_mm_u8i4_moe.c`): GU(e)의 배치 gb가 청크 gb의 열을 마지막으로
   읽으므로, 배치 gb 발행 직후 그 자리에 다음 expert의 청크 gb를 push. 통째로 마지막 배치 뒤에 밀면 3.5 MB가
   DN(n−1)의 배치 둘 아래밖에 못 숨어 DRAIN 251 us. 이제 다음 블록을 GU 전에 알아야 해서 `moe_blk_next`가
   루프 머리로. 활성화 블록은 청크들 뒤에 줄을 서므로 GATHER가 마지막 청크 쌍의 꼬리를 포함할 수 있다.
   기대: drain 251 → ~50, −4 ms.
2. **FC 커널 에필로그를 풀에** (`hexkl_mm_u8i4_dma.c` `hexkl_mm_u8i4_layer_run`): 결과 타일 하나 대신 스테이징
   2벌 × acc_tiles(남는 VTCM으로 결정, ≤32, ≥1), 배치 발행 → wait → `hvx_dq_tiles_worker` submit — MoE
   커널과 같은 모양. `accumulate`(attention의 P·V)와 acc 레이아웃 불가 경로는 동기 그대로. 스테이징 패리티는
   행 블록·핸들에 걸쳐 하나. 끝(과 에러 경로)에서 마지막 잡을 거둔다 — out_cat이 RPC 반환 전에 완성돼야
   한다. **quant 340은 그대로다**: 풀 안 쓴다는 추측이 틀렸다(`s->quant_pool`로 이미 분할). 기대: dequant
   268/403 → ~50, −4 ms. 새 호스트 체크 `fc_layer_host_check.c`(핸들 셋 N=1056/512/512, M=150·1, accumulate,
   타일 2개짜리 아레나, 1개는 거부) 비트 동일.
3. **MoE 콜의 작은 시퀀스 다섯을 rpcmem에서** (`htp_compute_ops.cpp` `invokeMoeLayer`): 핸들·row_index·
   row_count·row_weight를 `moe_args_buf_` 하나에 복사해 넘김. 힙에서 넘기면 드라이버가 매 콜 pin+map.
   기대: transport 627 → ~400, −4 ms.

측정: 전부 켬 config 그대로, 헤드라인 2회 + PROFILE=2 NNTR_PPL=1 1회. MoE 행 drain·transport, FC 행 dequant를
본다. 셋 합 기대 ≈ **−12**.

### 2.26 §2.25 실측: 1은 손해, 3은 무효, 2만 남김 — 641/626, 그리고 poll 창의 정체 (2026-09-22)

```
prefill 641 / 626 (585에서 +45)   decode 22.2 (24.2에서 -2)   ppl 62.0916 그대로
  MoE   host 14822 (+374)  gather 84 -> 747  drain 251+53 -> 18+42  transport 627 -> 605
  M==1  host 1612 (+170)   gather 128 -> 388
  FC N=2048  1724 -> 1480  dequant 268 -> 12
  FC N=3072  2278 -> 1920  dequant 403 -> 8
```

- **1(청크 일찍 push) 되돌림.** 활성화 블록이 링에서 3.5 MB 뒤에 줄을 서게 됐고, HMX 옆에서 DMA는 유휴
  33이 아니라 **~12 GB/s**(프로파일의 "averaged over the call")라 down 배치 둘 동안 못 내린다. GATHER +663이
  DRAIN −244를 삼켰다; decode도 같은 이유로 +170 us/콜. 활성화 슬롯을 2벌로 하면 되지만 상한이 −0.2 ms/콜이라
  안 한다. 커널 주석에 남김.
- **3(rpcmem 인자) 되돌림.** 627 → 605, 노이즈. 가설이 틀렸다. **MoE transport가 FC보다 200 us 큰 진짜 이유는
  콜 길이**: poll 창 5 ms를 넘는 콜(MoE 14 ms, dense 9.6)은 인터럽트 대기로 떨어져 깨어나는 비용을 내고,
  창 안에 끝나는 콜(FC 1.5, conv 4.3)은 안 낸다. 표가 정확히 그렇게 갈린다(MoE 605·dense 499 vs FC 349~442·
  conv 393). 남은 지렛대는 poll 상한(10000 거부, §2.16) — `NNTR_HTP_POLL_US=9000`, 안 되면 8000, 7000: 받아들여져
  `qos_mode=2`면 MoE transport가 ~400으로 내려오고 22 + 2콜 × ~0.2 = **−5 ms**.
- **2(FC 에필로그 풀) 유지**: 콜당 −244/−358 us, 12콜 **−3.6 ms**, 기대와 일치.

### 2.27 2만 남긴 실측 616/611, poll 9000은 받아들여지지만 무효 — 기기 편차가 남은 지렛대보다 크다 (2026-09-22)

```
(a) 2만: prefill 616 / 611   decode 24.3 / 24.2
(b) + NNTR_HTP_POLL_US=9000, PROFILE=2 (PPL 없음): 673   qos_mode=2 (받아들여짐)
    MoE host 14469  transport 596 (605/627에서)   FC N=2048 1524 (1724) dequant 7.6   FC N=3072 2006 (2278) dequant 6.0
    conv 4431 (4356)  dense 9627 (9560)  등록 4593 ms (2841)  스테이징 20.9 GB/s (24.2)
```

- **2는 산 대로 이겼다**: FC 콜당 −200/−272 us, 12콜 ≈ −3 ms.
- **헤드라인 +26(585 → 611)은 HTP 콜 밖이다.** 콜별 host 합: 585 실행 439 ms, 이번 442 — 같다. 등록 2.9 → 4.6 s,
  스테이징 24.2 → 20.9 GB/s, conv·dense 콜도 +1~2%: ARM·DDR 쪽이 이번 세션에 느리다(열, DVFS). ARM 잔여
  121 → 139. 같은 날 같은 세션이 아니면 **±25 ms는 기기 편차**이고, 남은 HTP 지렛대(각 ≤5 ms)는 그 아래다 —
  앞으로 그런 것은 `NNTR_HTP_PROFILE=3`(콜 안 5회 반복 최솟값)의 콜당 수치로만 판정한다.
- **poll 9000: 받아들여지지만(qos_mode=2) MoE transport 605 → 596.** §2.26의 "콜 길이가 창을 넘어서"라는
  설명도 틀렸다 — 창을 5 → 9 ms로 늘려도 14 ms 콜의 transport가 그대로다. 드라이버가 상한을 조용히 자르거나,
  200 us의 정체가 다른 것(콜 길이에 비례하는 캐시 유지?)이다. 기본값은 5000 그대로(스핀 코어 시간이 덜 든다).
  MoE transport는 미해결로 남긴다: 22콜 × 0.2 = 4 ms짜리 미지수.
- 여기서 HTP 쪽 소프트웨어는 멈춘다. 남은 것은 구조적이다: MoE HMX 활용률 59%(M=444, §2.24), ARM 잔여
  ~120~140(attention core, 다른 담당), 스테이징 25~30(레이어 입출력을 ION에, 텐서 할당자 훅).
