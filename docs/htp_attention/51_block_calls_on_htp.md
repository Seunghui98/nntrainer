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
