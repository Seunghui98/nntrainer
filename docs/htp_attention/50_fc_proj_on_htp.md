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
