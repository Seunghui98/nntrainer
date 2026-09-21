# 50 — FC projection을 HTP로: conv in_proj와 attention q/k/v/o (2026-09-18, 코드 완료·기기 미측정)

문서 49가 닫은 뒤 남은 prefill의 큰 덩어리는 ARM의 `fully_connected`다 (47 §16: 472–482 ms,
프로파일 빌드). 이 문서는 그중 두 묶음을 기존 FC→HTP 경로로 보내는 스위치와, 그 결과를
읽는 법이다. **기계는 새로 만들지 않았다** — 있는 것을 켰다.

## 1. 무엇을 켰나

| 스위치 (`nntr_config.json`) | 대상 | 층 | 형상 (M=444) | GFLOP |
|---|---|---:|---|---:|
| `"conv_in_proj_engine": "htp"` | `layerN_conv_in_proj` | 18 | [444×2048]×[2048×6144] | 201 |
| `"attn_proj_engine": "htp"` | `layerN_wq/_wk/_wv/_attention_out` | 6 | q/o: ×[2048×2048], k/v: ×[2048×512] | 56 |

`conv_in_proj_htp_layers` / `attn_proj_htp_layers`는 `moe_htp_layers`와 같은 층 목록이다
(빈 값 = 전부). 둘 다 기본값 `cpu`라 기존 config는 그대로 돈다.

§3.4 뒤에 나머지 FC에도 같은 스위치를 달았다 (2026-09-21):

| 스위치 | 대상 | 층 | 형상 (M=444) | GFLOP | WH 바이트 |
|---|---|---:|---|---:|---:|
| `"conv_out_proj_engine": "htp"` | `layerN_conv_out_proj` | 18 | ×[2048×2048] | 67 | 36 MiB |
| `"dense_ffn_engine": "htp"` | `layerN_ffn_up/_ffn_gate/_ffn_down` | 2 | up·gate ×[2048×7168], down ×[7168×2048] | 52 | 42 MiB |

각각 `*_htp_layers` 목록이 있다. dense FFN의 SwiGLU는 ARM에 남는다(up·gate 결과가 ARM으로 왔다가
down으로 다시 간다 — 융합 경로 `gemm_qs4cx_fused_swiglu`는 QS4CX 가중치를 받으므로 Q4_0 파일에는
안 맞는다). down은 K=7168이라 활성화 3.1 MiB가 VTCM에서 조각 더블버퍼(4 MiB)와 함께 빠듯하고
M=1024면 7 MiB로 안 들어간다 → `fcMaxRows(K)`가 콜을 행 단위로 나눈다 (K=7168: 512행/콜, K=2048:
1,920행/콜 — 이걸로 §3.1의 "m_pad ≈ 1,900 천장"도 사라진다. 콜 하나가 더 붙을 때마다 0.4 ms).

**기대는 작다** (§3.4의 in_proj 교훈, ARM ≈2 TFLOPS 기준): out_proj 18층 ARM ≈75 vs HTP ≈72 →
**≈0**; dense FFN 2층 ARM ≈65 vs HTP ≈32 → **−30**; q/k/v/o **−10~−30**. 합 **−40~−60 ms** 정도.
켜는 이유는 실측으로 표를 채우기 위해서지 배수를 기대해서가 아니다. 메모리: 아레나 슬랙에 in_proj
108 뒤 ≈36 남음 → out_proj 일부, 나머지(q/k/v/o 30 + dense 42 + out_proj 잔여)는 힙 ≈100 → **빠듯**.
`ENOMEMORY`면 `*_htp_layers`로 그룹별 층 수를 줄인다.

동작: `fully_connected`에 `engine=htp`가 붙으면 `FloatTensor::dot`(`float_tensor.cpp:1031`)이
M>1에서 `gemm_q4_0_accel_fp32`(FastRPC 1콜, `mm_u8i4_layer`)로, M==1에서는 CPU Q4_0으로
간다 (`accelerates_q4_0_at_m1()==false`). **decode는 안 바뀐다.** 가중치는 파일의 Q4_0x4
그대로이고, 로드 때 `htp_qs4cx_from_q4_0x4`로 변환해 DSP 힙에 등록한다.

코드 (커밋 하나):

- `compute_ops.h` / `htp_compute_ops.cpp`: `register_q4_0_weight(data, K, N)` — MoE의
  `register_qs4cx_weight`의 Q4_0 짝. 로드 때 등록해 첫 prefill이 24~42개 등록을 안 물게.
- `transformer.cpp repack_weight`: `fully_connected`의 Q4_0 가중치를 위 함수로 등록하고,
  첫 등록 뒤 M=512 워밍업 콜 1회(스테이징 버퍼 성장·첫 페이지 접촉을 prefill 밖으로 —
  47 §20.1 레버 6과 같은 이유).
- `lfm2_causallm.{h,cpp}`: 키 4개 파싱, 다섯 FC에 `engine`. `parseLayerIdList`를
  `lfm2_moe_causallm.cpp`에서 여기로 옮겨 공유.

## 2. 왜 in_proj가 먼저인가 — 콜당 포장이 고정비다

콜 하나의 왕복(FastRPC 고정 ≈0.4 ms + 페이로드 40 us/MB + ARM 스테이징 memcpy 0.125 ms/MB
+ DSP 활성화 quant ≈0.55 ms)은 **N에 거의 안 늘고 계산은 N에 비례**한다. 문서 34의 단가로
M=444, K=2048에서:

| 콜 | GFLOP | ARM (ms, FLOP 기준~프로파일 기준) | HTP 계산 | HTP 포장 | HTP 합 | 손익 |
|---|---:|---:|---:|---:|---:|---:|
| wk / wv (N=512) | 0.93 | 0.8–1.8 | 0.24 | 1.7 | 1.9 | **+1.1 ~ +0.1 손해** |
| wq / wo (N=2048) | 3.72 | 3.3–4.7 | 0.95 | 2.2 | 3.1 | −0.2 ~ −1.6 |
| conv_in_proj (N=6144) | 11.2 | 9.8–11.8 | 2.86 | 3.3 | 6.2 | **−3.6 ~ −5.6** |

손익분기: ARM에서 ≈2 ms 넘는 행렬곱만 이긴다 → 이 K에서 **N ≥ 약 1,300**. k/v는 따로
보내면 지고, q/o는 노이즈 안이고, in_proj는 확실히 이긴다.

기대 (prefill 848 ms 기준, 비프로파일):

| | ARM에서 빠짐 | HTP에 더해짐 | 순 |
|---|---:|---:|---:|
| in_proj 18층 | 152–212 | ≈110 | **−65 ~ −100** |
| q/k/v/o 6층 | 80–115 | 47–62 | −20 ~ −70 (k/v 몫은 0 또는 손해) |

두 숫자 모두 추정이다 — ARM 쪽 비프로파일 값을 안 쟀다. §4가 그걸 잰다.

## 3. 메모리 — 들어간다, 여유는 적다

DSP 32비트 주소공간(46 §41): 아레나 3840 MiB 매핑(MoE 3696 사용), 힙 ≈182 MiB(스크래치
12.8 사용). Q4_0 경로의 등록(`register_locked`)은 **힙**으로 간다.

| | WH 바이트 | 힙 누적 |
|---|---:|---:|
| q/k/v/o 6층 | 30 MiB | 43 |
| conv_in_proj 18층 | 113 MiB | 156 (둘 다 켜면) |

156 < ≈169 여유. 등록이 `AEE_ENOMEMORY`로 실패하면 로드 때 `runtime_error`로 죽는다
(`nntr_hvx_weight_register_u8i4 failed`). 그러면 `*_htp_layers`로 층 수를 줄이거나,
in_proj를 아레나 슬랙(144 MiB)에 넣는 코드(§6)를 앞당긴다. `conv_out_proj`(36 MiB)까지는
어느 쪽으로도 안 들어간다 — 47 F2 판정 그대로.

### 3.1 첫 기기 실행 — VTCM이 먼저 막았다 (2026-09-21)

```
[!] FATAL ERROR: Failed to repack weights: nntr_hvx_mm_u8i4_layer_timed failed: err=-2147482622
[HTP-PROFILE]   weights registered : 1   (convert 45.1 ms, register 39.9 ms)
```

`-2147482622 = 0x80000402 = AEE_ENOMEMORY`, 등록은 됐고(1개, 88.6 ms) **로드 워밍업 콜**에서
났다 — 첫 prefill이 아니라 로드에서 잡힌 것이 워밍업의 값어치다. 힙이 아니라 **VTCM**이다:
`hexkl_mm_u8i4_layer_run`(`hexkl_mm_u8i4_dma.c:361`)은 VTCM을

```
활성화 전체 (m_pad × K) | 가장 넓은 핸들의 WH 바이트 × 2 (더블버퍼) | result 타일 8 KB | config
```

로 잡는다. in_proj는 64 k타일 × 192 n타일 × 512 B = **6 MiB**, 두 벌이면 12 MiB > 8 MiB.
등록 검사(`hexkl_weight_u8i4_check`)는 한 벌(6 ≤ 8)만 보므로 통과했다. q/o(2 MiB)와
k/v(0.5 MiB)는 들어간다 — **막힌 건 in_proj뿐**이다. 문서 34의 FC 측정이 전부 K=1024,
N≤2048(≤1 MiB)이었던 이유이기도 하다.

**수정(호스트만, skel 무변경)**: `get_or_register_fc`가 `fcSliceCols(K)` = 2 MiB 조각(K=2048에서
N=2048)보다 넓은 가중치를 열 방향으로 잘라 핸들 여러 개로 등록하고(변환은 한 번, 조각마다
컬럼 복사·scale·colsum 슬라이스), `gemm_q4_0_accel_fp32`는 그 핸들들을 **한 콜**에 보낸다.
커널은 핸들 i+1의 가중치를 핸들 i의 행렬곱 뒤에 프리페치하므로(34 §4 C) 조각 수만큼의
DMA가 거의 숨는다. 활성화는 콜당 한 번만 양자화된다. 출력은 커널이 **핸들별 [M×N_i] 블록**으로
쓰므로(`out_off += M·N`) `copyOut`이 행마다 제자리에 되돌린다 — 어차피 하던 out 복사와 같은
바이트다.

같은 사실이 드러낸 것 하나: `gemm_q4_0_batch_fp32`/`gemm_qs4cx_batch_fp32`는 출력을
row-major(stride = ΣN)로 읽고 있었다. M=1(decode, 유일한 호출 형상)에서는 두 배치가 같아
지금까지 맞았고, M>1이면 틀렸을 것이다. 블록 단위로 고쳤다 — decode 바이트는 안 바뀐다.

**남는 천장**: 조각 2 MiB × 2 = 4 MiB를 빼면 활성화에 ≈4 MiB → K=2048에서 **m_pad ≈ 1,900행**.
`max_seq_len 2048`에서 num_to_generate 512를 빼면 prefill 최대 1,536이라 닿지 않지만, 더 긴
프롬프트는 FC 커널이 MoE 커널처럼 64행 블록을 돌아야 한다 (`fcSliceCols`의 ponytail).

### 3.2 두 번째 실행 — 조각은 돌았고, 주소공간이 막았다 (2026-09-21)

```
[HTP-PROFILE]   K=2048  N=6144  M>1  calls=1  rows=512  host=3.9 ms  dsp=3617 us  transport=315 us
                [quant 400  dequant 920  acc 552  | rest 1745 (mm; FC 경로는 mm 프로브가 없다)]
...
[HTP] arena chunk 13: 256 MiB, mapped total 3584 MiB
[HTP] arena: 256 MiB refused ... 128 refused ... 64 refused
[!] FATAL ERROR: ... cannot register a 2048x3584 weight (3 MiB) ... mapped=3584 MiB in 14 chunks
```

**조각 콜은 됐다.** M=512 워밍업이 host 3.9 ms(dsp 3.6 + transport 0.3)에 돌았다 — §2의 추정
(계산 2.9 + 포장 3.3)보다 transport가 훨씬 싸다. M=444로 환산하면 콜 ≈3.4 + 스테이징 ≈1.1 →
**≈4.5 ms**, ARM 9.8~11.8 대비 층당 −5~−7, 18층 **−95~−130 ms** 기대로 올라간다.

**막은 것은 DSP 주소공간이다.** 아레나가 14청크(3584 MiB)에서 멈췄다 — 혼자 돌 때는 15청크
3840까지 간다(46 §41). 그 사이 달라진 건 FC 가중치가 **그래프 순서로** DSP 힙에 등록된 것뿐이다:
conv 층(in_proj 6 MiB, 힙)이 MoE 층(아레나 청크)과 번갈아 온다. 46 §41의 모델 — 청크 매핑은
크기 정렬(256 MiB 경계)이고 힙과 같은 커서를 쓴다 — 대로면, 청크 사이에 힙이 자랄 때마다 힙
끝에서 다음 256 MiB 경계까지가 버려진다. 등록된 힙 ≈100 MiB로 ≈400 MiB가 사라진 셈이다.

**수정**: `repack_weight`가 FC 등록을 모아 두었다가 **레이어 순회가 끝난 뒤**, 즉 15청크가 전부
매핑된 뒤에 한다. 그러면 46 §41이 잰 배치(3840 매핑 + 힙 182 MiB) 그대로이고, FC 143 + 스크래치
12.8 = **156 < 182**. 여유 26 MiB. 힙 성장 단위가 그보다 굵으면 여기서도 `ENOMEMORY`가 날 수
있다 — 그때는 in_proj를 아레나 슬랙(144 MiB)으로 (§6), 또는 `conv_in_proj_htp_layers`로 층을 줄인다.

같이 고친 것: 조각 등록의 프로파일 시계가 변환 뒤에 시작해 `alloc + other`가 음수로 넘쳤다
(`18446744073709464.0 ms`). 첫 조각의 시계를 변환 앞에서 시작한다.

### 3.3 세 번째 실행 — 아레나는 찼고, 힙은 100 MiB뿐이었다 (2026-09-21)

```
[HTP] arena chunk 14: 256 MiB, mapped total 3840 MiB          ← 15청크, 46 §41 그대로
[!] FATAL ERROR: ... nntr_hvx_weight_register_u8i4 failed: err=-2147482622
[HTP-PROFILE]   weights registered : 1450                       ← MoE 1408 + FC 조각 42
```

순서 수정은 맞았다 — 아레나가 3840까지 갔다. 그 뒤 FC 조각이 **42개(84 MiB)** 힙에 들어가고
43번째에서 `ENOMEMORY`. 스크래치 12.8까지 **힙 ≈100 MiB** — 46 §41의 프로브가 깨끗한 프로세스에서
잰 182와 다르다(앱은 ION 스테이징 버퍼 등도 DSP에 매핑돼 있다). FC 143은 힙에 안 들어간다.

**수정**: 조각을 **아레나의 남은 자리**에 먼저 넣는다. 매핑된 3840 중 MoE가 3696이라 ≈144 MiB가
비어 있고(청크 꼬리 + 마지막 청크), `place()`의 스캔 부분만 떼어낸 `placeExisting`이 **이미
매핑된 청크 안에서만** 자리를 찾는다 — 새 청크는 남은 힙 등록이 쓸 주소공간을 먹으므로 안 만든다.
아레나는 WH 바이트를 원하므로 호스트에서 `whPack`(오프라인 양자화기와 같은 함수, 유닛테스트
`WhPackReferenceMatchesDspBake`가 바이트 단위로 지킨다)으로 굽고, 캐시된 버퍼에 구운 뒤 통째로
memcpy한다(uncached 청크에 nibble RMW를 하면 기어간다). 자리가 없으면 힙으로 떨어진다.

예산: in_proj 54조각 108 MiB → 아레나 ≈144 (마지막 청크 ≈112 + 꼬리들; 꼬리는 down 1.75 MiB가
채워 2 MiB 조각이 못 들어갈 수 있다 → 실제 ≈120~144). q/k/v/o 30 MiB → 아레나 나머지 또는 힙
100. **둘 다 들어간다, 여유 10~40 MiB.** 로드 로그의 `[HTP-PROFILE] rpcmem/ION buffer` 줄과
등록 수로 어디에 들어갔는지 읽을 수 있다.

### 3.4 네 번째 실행 — 돌았다. 그리고 이득은 −13 ms다 (2026-09-21)

`NNTR_NUM_THREADS=8`, 444 토큰, in_proj 18층 HTP (attn_proj는 아직 CPU):

```
PROFILE=0   prefill 835 ms  531.7 TPS   decode 16.45 TPS
PROFILE=2   prefill 834 ms  532.4 TPS   decode 15.23 TPS
  K=2048 N=6144 M>1  calls=19 rows=8504  host=3985 us/call  dsp=3203  transport=782
                      [quant 357  dequant 794  acc 501  drain 58  | mm(rest) 1492]
  K=2048 N=2048 M>1  calls=23  host=17297  dsp=14917  transport=2380      ← MoE, 전과 같음
  weights registered 1462 (MoE 1408 + 조각 54, 전부 아레나)
```

**돌아간다**: 18층 전부 NPU(19콜 = 18 × 444행 + 워밍업 512행), 텍스트 CPU 실행과 동일, decode에
FC 형상 없음(M=1은 CPU). 콜은 host 4.0 ms — §2의 추정 6.2보다 싸다.

**그런데 prefill은 848 → 835, −13 ms.** §2가 기대한 −65~−100의 1/5이다. 산술은 이렇다:

| | ms |
|---|---:|
| HTP in_proj 18층: 콜 3.98 × 18 = 72 + 스테이징 14.5 MB × 18 / 15 GB/s = 17 | **≈ 89** |
| ARM in_proj 18층 (역산: 89 − 13) | **≈ 100** |

즉 ARM in_proj는 프로파일 빌드가 말한 152~212가 아니라 **≈100 ms** — ggml Q4_0 GEMM이
N=6144짜리 넓은 행렬에서는 11.2 GFLOP를 5.6 ms(≈2 TFLOPS)에 한다. §2의 ARM 값은 프로파일
빌드 팽창을 그대로 믿은 것이고, §4의 0단계(비프로파일 ARM 측정)를 건너뛴 대가다. **문서 45의
결론 그대로다: 포장이 계산보다 크면 옮겨도 남는 게 없다** — 콜 4.9 ms 중 HMX+acc는 2.0.

decode 16.5 vs 이전 20.8은 이 변경과 무관해 보인다(FC decode는 같은 CPU 커널, MoE decode 콜
1.90 → 2.17 ms는 transport 553 → 814 us가 오른 것 = 호스트 쪽) — 같은 세션에서 스위치 없는
A 실행이 아직 없어 발열인지 다른 것인지 못 가른다. **A를 먼저 돌린다.**

### 3.5 다섯 번째 실행 — 전부 켬: 형상별 콜은 잡혔고, 벽시계는 더 나빠졌다 (2026-09-21)

네 스위치 전부 `htp`, 8스레드, PROFILE=2, 앞선 실행 직후(식히지 않음):

```
prefill 945 ms 469.8 TPS   decode 11.6 TPS   등록 1528 (MoE 1408 + FC 120), 로드 +2.9 s
  N=512   (wk/wv)        12콜  host 1750 us  dsp  629  transport 1121  [quant 364 dequant 58 acc 56]
  N=2048  M>1            53콜  = MoE 23 + wq/wo 12 + out_proj 18 → FC 몫 ≈ (488−398)/30 = 3.0 ms/콜
  N=6144  (in_proj)      19콜  host 4591     dsp 3251  transport 1339   (§3.4에선 3985 / 782)
  N=7168  (up/gate)       4콜  host 4254     dsp 3609  transport  645  [quant 372 dequant 908 acc 582]
  K=7168 N=2048 (down)    2콜  host 4076     dsp 3351  transport  726  [quant 1253 dequant 279 acc 193]
  M==1 MoE decode       9416콜 host 2993     dsp 1360  transport 1633   (09-16: 1910 / 553)
```

**형상별 판정 (콜 host + 스테이징 vs ARM ≈2 TFLOPS):**

| 그룹 | HTP 실측 | ARM 추정 | 순 |
|---|---:|---:|---:|
| wk/wv N=512 ×12 | 21 + 0.5 = **22** | 10–22 | **손해~본전** (§2 예측대로) |
| wq/wo/out_proj N=2048 ×30 | 90 + 27 = **117** | ≈122 | **본전** |
| dense up/gate ×4 + down ×2 | 25 + 6.5 = **32** | 39–65 | −7 ~ −33 |
| in_proj ×18 | 83 + 17 = **100** | ≈100 | 0 (§3.4의 89가 100으로 — transport가 올랐다) |

콜 하나하나는 예측 범위 안이다. down의 K=7168 quant 1.25 ms가 콜의 37% — 활성화 u8 변환이 K에
비례하고 FC 커널은 그걸 숨기지 못한다.

**그런데 prefill 835 → 945, decode 16.5 → 11.6.** 두 가지가 겹쳐 있다:

1. **transport가 실행마다 오른다.** MoE decode 콜의 dsp는 1.36 ms로 09-16과 같은데 transport는
   553 → 814 → 1633 us. in_proj도 782 → 1339. transport = ARM 쪽(FastRPC 스택 + 캐시 유지)이다.
   같은 날 4번 연속 50초짜리 실행 뒤라 **발열로 ARM이 느려진 것**이 첫 후보이고, 등록 핸들 수
   (1408 → 1528)나 힙 사용량에 FastRPC 콜 비용이 비례하는 것이 둘째 후보다. 스위치 없는 A 실행을
   **식힌 뒤** 돌려야 가른다 — 아직 없다.
2. **텍스트가 바뀌었다.** 요약 3문장이 §3.4까지의 실행과 다르다(더 짧은 think, 다른 문장, 428토큰에서
   `<|im_end|>`로 정상 종료). 재양자화 층이 늘어난 결과이고, softmax 앞의 q/k/v가 가장 의심스럽다.
   **정확도 게이트 실패** — attn_proj는 QS4CX 오프라인 양자화(§6) 없이는 끄는 것이 맞다.

### 3.6 transport의 정체 — 스테이징 버퍼 크기에 비례하는 콜당 캐시 유지비 (2026-09-21)

§3.5의 "전부 켬"을 한 번 더 돌렸다: prefill 996, decode 11.66, MoE decode 콜 transport **1626 us**
(직전 1633). 7 us 차이 — 발열이면 이렇게 같을 수 없다. 형상이 불변인 MoE decode 콜의 transport를
설정별로 놓으면:

| HTP에 올린 것 | 스테이징 act / out (가장 큰 형상에 맞춰 자란 크기) | decode transport | decode TPS |
|---|---|---:|---:|
| MoE만 (09-16) | 3.6 / 3.6 MB | 553 us | 20.8 |
| + in_proj (out 10.9 MB) | 3.6 / 10.9 | 814 | 16.5 |
| + dense (act·out 12.7 MB) | 12.7 / 12.7 | 1633, 1626 | 11.6, 11.7 |

증가분 +7.3 MB → +261 us, +18.2 MB → +1080 us = **36~59 us/MB**, 캐시 플러시 속도다.
메커니즘: `invokeLayer`·MoE 콜 전부가 `act_buf_`/`out_buf_` **한 쌍**을 공유하고 `ensureCapacity`가
가장 큰 형상에 맞춰 키웠다. ARM 캐시드 ION 버퍼는 FastRPC 드라이버가 콜마다 clean/invalidate를
하는데, 그 범위가 **넘긴 바이트가 아니라 dma-buf 전체**다. 8 KB를 쓰는 decode 콜이 25 MB를
플러시했고, prefill의 82콜 전부가 같은 세금을 냈다(in_proj 콜 transport 782 → 1496).

**수정**: 크기 클래스별 버퍼(`StagingPool`/`stage`, 64 KiB부터 2배씩). decode는 64 KiB 쌍, MoE
prefill은 4 MiB, in_proj·dense는 16 MiB. 기대: decode transport 553 근처로 → **≈20 TPS 복구**,
prefill 콜 82개 × 0.3~0.7 ms → **−40~−60 ms** (835 기준으로 in_proj·dense의 진짜 값이 그때 보인다).
ION 합계 ≈40 MB(이전 25). 이 세금은 in_proj 이전에도 있었다 — 3.6 MB 쌍의 553 us 중 ≈260이
그것이니, MoE만 돌 때도 decode 콜당 ≈0.25 ms(22층 5 ms/token)는 돌아온다.

### 3.7 크기 클래스 적용 — decode 복구, prefill은 FC가 손해라는 것이 남았다 (2026-09-21)

전부 켠 채(등록 1528) 같은 명령:

```
prefill 943 ms 470.8 TPS   decode 20.48 TPS (직전 11.66)
  M==1 MoE decode  host 1513 us  dsp 1353  transport  161   (직전 2985 / 1359 / 1626; 09-16: 1910 / 553)
  N=512  (wk/wv)   host 1029     transport  456   (직전 2115 / 1470)
  N=2048 M>1       host 8364     transport 1101   (직전 9302 / 2011)   ← MoE 23 + FC 30 합산
  N=6144 (in_proj) host 4299     transport 1092   (직전 4762 / 1496)
  N=7168 / K=7168  host 4164 / 3944                (직전 4250 / 4066)
```

**§3.6의 진단이 맞았다.** 형상이 불변인 MoE decode 콜의 transport가 1626 → **161 us**, 09-16의
553보다도 낮다(3.6 MB 쌍 → 64 KiB 쌍). decode 11.7 → 20.5 TPS. prefill 콜 82개 전부 transport가
0.4~1 ms씩 내려 prefill 996 → 943.

**남은 것 둘:**

1. **prefill 943은 여전히 in_proj만 켠 835보다 느리다.** HTP에 올린 FC의 host 합 ≈204 ms + 스테이징
   ≈45 = **≈250 ms**인데, 그 FC들이 ARM에서 걸리던 시간은 §3.4 방식으로 역산하면 ≈115~210. 즉
   **k/v·q/o·out_proj는 손해, dense는 본전~소폭 이득**. FC를 HTP로 보내는 건 여기서 닫는다 —
   남기는 건 §3.6의 스테이징 수정(MoE만 돌려도 decode 콜 0.39 ms, prefill 콜 1.3 ms를 돌려준다)과
   실측 표다.
2. **ARM 쪽이 09-16보다 느리다.** decode 48.8 ms/token = MoE 33.3 + ARM 15.5인데 09-16은 48.1 = 42.0 +
   ≈6. prefill도 HTP 콜 합을 빼면 ARM 잔여 ≈324 (in_proj만 켰을 때 366, 거기서 FC ≈115~210이 빠졌어야
   한다). 발열이거나, `engine=htp`인 FC 레이어의 **decode CPU 경로**가 cpu 엔진의 것과 다른 것이다
   (같은 `CpuComputeOps::gemm_q4_0_fp32`로 읽히지만 실측이 없다). **스위치 전부 끈 A 실행**이 가른다:
   decode가 ≈25 TPS(1/(33.3+6))로 나오면 FC 스위치가 ARM decode를 늦춘 것이고, 20.5면 발열/환경이다.

### 3.8 A 실행 — 오늘의 기준선은 848이 아니라 921~1013이다 (2026-09-21)

스위치 없음(MoE만), 8스레드, PROFILE=0, 연속 2회:

```
prefill 921 ms 482 TPS / 1013 ms 438 TPS      decode 21.18 / 21.09 TPS      텍스트 2회 동일
```

| | 09-16 | 오늘 A | 뜻 |
|---|---:|---:|---|
| prefill | 848 (3회 최솟값) | **921 / 1013** | 기기가 오늘 +73~+165 느리다. 연속 2회가 10% 벌어진다 |
| decode | 20.8 | **21.1** | §3.6 수정 후에도 거의 그대로 — 콜 transport −0.39 ms×22가 벽시계엔 안 보인다 |

**§3.7·§1.3의 "ARM이 느리다"는 스위치 탓이 아니다** — 스위치 없는 A도 같은 만큼 느리다. 오늘 기기
(발열·DVFS)의 상태이고, 09-16의 848은 오늘 비교 기준으로 못 쓴다. 같은 날 A/B/C만 유효하다.

같은 날 비교:

| config | prefill (2회 최솟값) | vs A |
|---|---:|---:|
| A: MoE만 | 921 | — |
| C: + in_proj + dense 융합 | 888 | **−33** |
| 전부 켬 (§3.7) | 943 | +22 |

−33은 기대(in_proj −13, dense −15~−20)와 맞지만 A 자체의 편차(92 ms)보다 작다. **FC 단위 효과는
오늘 기기의 노이즈 안에 있다** — 그래서 판정은 벽시계가 아니라 프로파일의 콜 행(`N=6144`, `M>1 dense`)으로
한다. decode는 A 21.1 vs C 20.6 — 4회 일관된 0.5 TPS 차(≈1.2 ms/token)인데, C의 FC decode 경로는
같은 CPU 커널이라 설명이 없다. 노이즈로 두고, 다시 보이면 조사한다.

텍스트: A는 2회 동일하고, in_proj만 켠 실행·C와는 다르다. **양자화 지점이 바뀔 때마다 토큰이 바뀐다**는
것을 A가 확인한다 — 게이트는 로짓 차로 바꿔야 한다.

## 4. 측정 — 실행 순서와 읽을 것

config는 문서 49 §6의 NPU config(`moe_engine: htp`)에 키만 더한다. 프롬프트·`num_to_generate`
같게. 프로파일 빌드는 TPS를 왜곡하니(46 §48.7) **헤드라인은 비프로파일 3회 최솟값**.

```
0. 기준: 스위치 없음                        → prefill ms (3회 최솟값)        = 848 근처여야 함
1. "conv_in_proj_engine": "htp"              → prefill, decode TPS, 텍스트
2. 1 + "attn_proj_engine": "htp"             → 같은 것
3. (선택) attn만                             → q/k/v/o 단독 몫
```

각 실행에서:

- **텍스트가 CPU 실행과 같은가.** Q4_0→QS4CX 즉석 재양자화는 오차가 더 크다(mean_abs_err
  0.0451 vs 직접 QS4CX 0.0333, `htp_compute_ops.cpp`). q/k/v는 softmax 앞이라 텍스트가
  바뀔 수 있다. 바뀌면 §6의 QS4CX 오프라인 양자화로.
- **decode TPS가 20.8 근처인가.** M=1은 CPU여야 하니 안 움직여야 한다. 움직이면 게이트가
  샌 것이다.
- `NNTR_HTP_PROFILE=2`: 형상별 행에 `M=444 K=2048 N=6144`(in_proj), `N=2048`(q/o),
  `N=512`(k/v)가 새로 생긴다. 행마다 **wall과 dsp의 차 = transport** — §2의 포장 추정
  (1.7~3.3 ms/콜)을 실측으로 바꾼다. `register` 합계는 로드에 있어야 하고 prefill 안에
  없어야 한다.
- `--profile` 빌드 + `num_to_generate: 1` 1회: `[PROFILE]`의 `layerN_conv_in_proj`·`_wq`·
  `_wk`·`_wv`·`_attention_out` 행이 HTP 콜 시간으로 바뀐다. 0단계와 나란히 놓으면 ARM에서
  빠진 양이 층별로 나온다.

보고할 것: 0/1/2의 prefill 최솟값 3개, decode TPS, 텍스트 동일 여부, PROFILE=2의 새 형상
행(wall/dsp/calls), 로드 시간 변화(등록 24~42개가 로드로 옮겨왔으니 몇 백 ms 늘어야 정상).

## 5. 판정

| 결과 | 뜻 | 다음 |
|---|---|---|
| 1단계 −65 이상 | 포장 추정이 맞다 | in_proj 유지, 2단계 |
| 1단계 −30 미만 | transport가 추정의 2배 이상 — PROFILE=2 행이 말해준다 | 포장을 줄이는 쪽(§6 u8 활성화, 또는 문서 45 Phase D) |
| 2단계가 1단계보다 느리거나 같다 | k/v가 진 것 | `attn_proj_htp_layers`로 빼거나 §6 qkv 묶음 |
| 텍스트 다름 | 재양자화 오차 | §6 QS4CX 오프라인 |
| 로드에서 ENOMEMORY | 힙 여유 추정이 틀림 | §3 |

## 6. 다음 손잡이 (이 커밋에 없음)

- **qkv 한 콜**: 가중치 3개를 들고 `Tensor::dot(vector)` 한 번 부르는 작은 레이어
  (`gemm_q4_0_batch_fp32`가 이미 있다). 포장을 층당 4→2회. 기대 −1.5~−2.5 ms/층.
- **QS4CX 오프라인**: `quantize.cpp buildLayerDtypeMap`에 이름 4개(또는 in_proj)만 QS4CX로
  쓰는 플래그 → `dotQs4cx` → prefill HTP, decode KleidiAI. 오차 26% 감소, 로드 변환 없음.
  FC 레이어가 텐서별 dtype을 어떻게 받는지 확인 필요.
- **in_proj를 아레나 슬랙에**: 변환 뒤 bump-alloc + `registerFromArena`. 힙 대신 아레나
  144 MiB. 힙이 모자랄 때만.
- **왕복 자체를 없애기**: 문서 45 Phase C/D. 포장이 층당 1콜로 줄고 mha_core까지 빠진다.
  "훨씬"은 여기서 나온다.
