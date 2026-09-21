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
  못 가른다.**

## 2. conv 블록 상주 — 설계

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
