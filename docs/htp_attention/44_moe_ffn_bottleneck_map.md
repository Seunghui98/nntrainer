# 44 — MoE FFN on HTP: 병목 지도와 설계 방향

**대상**: LFM2.5-8B-A1B MoE FFN, Hexagon V79 HTP, 기기 R3CY10WM83Y
**근거**: `NNTR_HTP_PROFILE=2` (layer 0만 HTP, 32 experts × 64 FastRPC 호출, 444-token prefill) 여섯 번 실행 중 `mm<=` 컬럼이 있는 마지막 실행. `layer calls total`은 여섯 번 모두 41.5–42.6 ms (±1%)로 안정. 등록과 decode는 기기 drift로 ±40% 흔들리므로 단발 비교에 쓰지 않음.
**선행 문서**: `43_moe_ffn_measured_next_levers.md` §7의 2026-09-09/10 행 (L2 근본 원인, `NNTR_L2_DIFF`/`NNTR_L2_SHADOW` 결과). 이 문서는 그 결과를 하나의 설계로 묶은 것.
**시각화**: 같은 내용의 인터랙티브 버전 — https://claude.ai/code/artifact/1071c03a-d855-4d47-84ab-126eac81b31e

---

## 0. 한 줄 결론

| | ms/layer |
|---|---|
| 두 번 dot HTP (L2 off) | 55.6 (43 §1) |
| fused L2 (split-call) | **42.9 측정 + ≈21 미계측 ARM ≈ 64** |
| **CPU (M0 계측, bracketed 38.2/38.8)** | **38** |

fused가 두 번 dot보다 25% 빠른 건 맞지만 둘 다 CPU보다 느리다. **가속기가 존재하는 이유인 행렬곱(`mm`)은 벽시계의 14%**뿐이고, 나머지는 바이트를 옮기고 타입을 바꾸는 일이며, 그중 가장 큰 덩어리는 **64번의 FastRPC 왕복**이다.

---

## 1. 레이어 하나, 42.9 ms의 분해 (32 experts 합산)

| 단계 | ms/layer | 비중 | 정체 | 줄일 수 있나 |
|---|---:|---:|---|---|
| **transport** (FastRPC 왕복) | **14.76** | **34%** | host − dsp. 페이로드 expert당 ~1.1 MB, 레이어당 36 MB | **배칭으로 8×** |
| **mm** (HMX 행렬곱, 잔차 상한) | 8.83 | 21% | DSP 총시간 − 이름 붙은 단계 전부. 상한값 | 배칭이 config 상각 |
| drain (weight DMA 대기) | 5.13 | 12% | 176 MB / 34 GB/s = DDR 하한. **CPU도 똑같이 냄** | 불가 |
| dequant (i32→f32) | 4.90 | 11% | 2.3 G elem/s — 32-lane HVX치고 느림 | HVX 튜닝, 프로브 먼저 |
| quant (act f32→u8) | 3.15 | 7% | 1.3 G elem/s — 역시 느림. gate_up만 | 배칭으로 4.6× 적게 |
| acc (HMX accumulator read) | 2.95 | 7% | 벤더 코드 | 불가 |
| swiglu (DSP SwiGLU) | 1.81 | 4% | VTCM 안에서 silu(gate)·up | 유지 |
| ARM staging memcpy (측정) | 1.4 | 3% | 27.8 MB, 21.1 GB/s | 배칭으로 u8 한 번 |
| **합 (프로파일)** | **42.9** | | | |
| 미계측 ARM 쪽 | ≈21 | | 43 §4.3의 M0 계측(HTP 레이어 벽시계 68 vs 프로파일 47.3)에서 온 **추정치**. 이번 실행에서 재지 않음 | **P3이 재는 항목** |
| **벽시계 (추정)** | **≈64** | | | |

`mm`은 DSP 총시간에서 이름 붙은 단계를 전부 뺀 잔차라 상한값이다. DSP에 프로브를 넣으면 재는 대상이 흔들리므로 빼기가 정직한 방법이다 (`htp_compute_ops.cpp`의 `mm<=` 컬럼).

---

## 2. 호출 하나의 해부 — gate_up vs down (µs / call)

| 단계 | gate_up (K=2048 N=3584) | down (K=1792 N=2048) | 비고 |
|---|---:|---:|---|
| host (벽시계) | 884.3 | 413.9 | FastRPC 진입부터 반환까지 |
| &nbsp;&nbsp;transport | 272.1 | 189.2 | 450 KB f32 in / 114 KB u8 out (gate_up); 114 KB in / 450 KB out (down) |
| &nbsp;&nbsp;dsp | 612.2 | 224.8 | DSP 자체 시계 |
| &nbsp;&nbsp;&nbsp;&nbsp;quant | 98.5 | 0 | 64×2048 f32→u8 |
| &nbsp;&nbsp;&nbsp;&nbsp;swiglu | 56.6 | 0 | |
| &nbsp;&nbsp;&nbsp;&nbsp;dequant | 100.5 | 52.7 | 64×3584 i32→f32 |
| &nbsp;&nbsp;&nbsp;&nbsp;acc | 59.3 | 33.0 | |
| &nbsp;&nbsp;&nbsp;&nbsp;drain | 111.9 | 48.4 | 3.67 MB + 1.84 MB weight / expert |
| &nbsp;&nbsp;&nbsp;&nbsp;**mm (잔차 ≤)** | **185.3** | **90.7** | M=55→64 패딩, weight 바이트당 128 MAC |

**Qwen3-0.6B q_proj가 빨랐던 이유가 이 표에 있다**: 그 측정(`34_fc_measured.md`)은 M=1024, 한 번 호출, DSP-only 시간이었다. 여기는 M≈55, 64번 호출, FastRPC 포함이고, weight 원소 하나가 1024번이 아니라 64번만 재사용된다. M은 라우팅(444 토큰 × top-4 / 32 experts)이 정하는 값이라 커널로 못 늘린다 — 호출당 고정비를 상각하는 유일한 길은 **호출 수를 줄이는 것**이다.

---

## 3. 정확도 — 왜 깨지고, 어떻게 fusion을 살리는가

### 3.1 두 경로의 실제 차이

| | 동작 (L2 off) | 깨짐 (L2 on, split-call) |
|---|---|---|
| gate_up | HTP `mm_u8i4_layer` → f32 반환 | HTP, VTCM 안 |
| **SwiGLU** | **ARM NEON `exp_ps` + `vdivq_f32`** | **DSP HVX `hvx_exp_sf`(qf32, 7차) + NR 역수** |
| down | HTP `mm_u8i4_layer` (내부 u8 양자화) | HTP `mm_u8i4_layer_u8in` (u8 그대로) |

**동작하는 경로도 HTP를 두 번 타고 u8 양자화를 두 번 겪는다.** `float_tensor.cpp:1087`의 `dotQs4cx`가 `supports_gemm_qs4cx_accel_fp32()`를 가장 먼저 확인하고 HTP는 `true`를 돌려주므로, QS4CX weight는 `kFusedSwigluEnabled`와 무관하게 HTP로 간다. (`mobile_e2e_run_guide.md` §3-1b의 "QS4CX는 KleidiAI CPU 경로"는 task #17로 QS4CX HTP 훅이 생기기 **전** 문구다 — 지금은 틀렸다.) 같은 양자화 함수(`hvx_quant_rows_u8_params`/`hvx_quant_pack_u8_ah`, 소스로 확인), 같은 weight. 다른 건 **SwiGLU 한 줄뿐**이다.

### 3.2 기기에서 확정된 메커니즘 (43 §7 2026-09-09/10 행 요약)

1. 두 SwiGLU는 스펙상 둘 다 정확(~1e-6)하지만 **서로 다른 근사**다.
2. 그 차이가 u8 양자화 경계에 걸린 원소 하나를 가끔 옆 레벨로 민다 — `NNTR_L2_DIFF`: 32 호출 중 5개에서 `total_flips=1/1792`, `scale_diff` ≈ 1e-5 % (1 ULP), 나머지 27개는 142 dB (float32 노이즈).
3. down 행렬곱이 그 원소를 K=1792 전체에 곱해 퍼뜨린다. SNR 67.6–79.8 dB의 분산은 flip 개수(항상 1)가 아니라 **어느 K 인덱스가 갈렸는가**(그 인덱스의 down weight 크기)로 설명된다.
4. `NNTR_L2_SHADOW` (fused 커널을 전부 실행하되 모델에는 레퍼런스 값을 넘김) → 정상 512-token 요약. **부수효과가 아니라 값이다.**
5. NaN(`hvx_recip_qf32` seed 발산, `exp_top` 88→85 수정)은 실재하는 잠재 버그였으나 이 실패의 원인은 아니었다 — `NNTR_L2_CHECK`로 non-finite 0개 확인. 수정과 `SwigluSurvivesExtremeNegativeGate`는 그대로 유지.
6. 기각된 가설 두 개: outlier-row 양자화(정상 호출의 `call_max_span` 0.17–2.91이 나쁜 호출의 0.30–0.54를 완전히 덮음), 행 전체 scale 불일치(`scale_diff` ≈ 0).

### 3.3 "SiLU는 CPU에서만"이 아니다

문제는 *어디서* 계산하느냐가 아니라 *두 쪽이 서로 다른 근사를 쓴다*는 것이다. 그리고 HMX의 activation 포트는 하드웨어적으로 8-bit 고정(IDL: "HMX's activation port is always 8-bit")이라 "중간값을 int16으로 넓히자"는 이 하드웨어에서 선택지가 아니다.

**해법: 양쪽이 비트 단위로 같은 알고리즘을 돌리게 한다.** NEON과 HVX 둘 다 순수 IEEE f32 곱셈/덧셈만으로(FMA 없이, qf32 없이, 나눗셈 없이) 같은 다항식과 같은 NR 역수를 같은 순서로 계산하면, 같은 입력에 같은 비트가 나오고 flip은 정확히 0이 된다. 결정적(deterministic) 크로스플랫폼 수학의 표준 기법이고, 원소별 연산이라 비용은 expert당 56 KB — 무시할 수준이다.

지금 두 구현(`neon_mathfun.h` `exp_ps`, `hvx_exp_f32.h` `hvx_exp_sf`)은 range-reduction 상수(`ln2_hi=0.693359375`, `ln2_lo=-2.12194440e-4`)를 이미 공유한다. 갈리는 지점은 셋뿐: 다항식 차수(5 vs 7), 산술 포맷(IEEE+FMA vs qf32), 나눗셈(`vdivq` vs NR). ARM 쪽이 `vdivq` 대신 NR을 쓰게 되면서 CPU 경로도 ≤1 ULP 바뀌지만, 모델이 `std::exp`(shadow 실행)와 `exp_ps` 둘 다에서 정상 동작한 만큼 허용 범위 안이다. 검증은 기기 gtest에서 두 구현의 출력 바이트를 `memcmp` — **SNR이 아니라 비트 동일성**.

---

## 4. 개선 항목 — 효과순

| # | 항목 | 효과 | 비고 |
|---|---|---|---|
| **A1** | **비트 동일 SwiGLU** — NEON/HVX 공통 결정적 알고리즘 (§3.3) | flip 5/32 → **0**, fused == two-dot | **P1의 선행조건** (배칭은 SwiGLU가 DSP에 있어야만 가능) |
| **P3** | **미계측 ARM ≈21 ms 실측** — `NNTR_M0_PROFILE` 확장, 라우터/top-k/gather/scatter/뷰 생성/디스패치 분리 | **HTP-vs-CPU 판정 확정** | 코드 ≈0. §5 참조 |
| **P1** | **MoE 배칭** — 64 호출 → 1. 활성화를 u8로 한 번(0.9 MB) + 라우팅 테이블만 보내고, 라우팅 weight 곱-합까지 DSP에서 끝낸 뒤 444×2048 f32(3.6 MB)만 반환. IDL 엔트리 하나: `mm_u8i4_moe_layer(handles[], row_index[], row_counts[], act_ah, …)`. 검증된 u8in 경로 재사용 | transport −12, quant −2.5, staging −1.1 → **≈ −15 ms** | 와이어 36 → 4.5 MB, HMX config 상각 |
| P2 | dequant HVX 튜닝 — 2.3 G elem/s는 느림. 두 패스 구조거나 worker-pool 오버헤드 의심 | 4.9 → ≈2.5, **≈ −2.4 ms** | 프로브 하나로 확인 후 |
| P4 | 등록 bake 캐시 + ION 매핑 — convert 452 ms + register 1217 ms를 **레이어마다**. 22 레이어면 40 s 기동, `HEXKL_MM_U8I4_MAX_WEIGHTS=512`로 8 레이어까지만. WH bake 결과는 결정적 바이트라 한 번 구워 저장하면 다음 로드는 memcpy | 1.8 s → ≈0.1 s / layer | 성능과 무관하게 **전체 모델 실행의 전제** |
| — | **건드리지 않을 것**: drain (DDR 하한, CPU도 냄) · acc (벤더) · decode를 HTP로 (M=1은 weight-bound, 25.8 vs 25.7 TPS로 문서 확정) · quant를 ARM으로 (300–420 µs/call로 이미 측정된 손해) | | 근거 있는 제외 |

---

## 5. 어디까지 갈 수 있나

| 단계 | HTP 프로파일 | ARM 쪽 | 벽시계 | vs CPU 38 |
|---|---:|---:|---:|---|
| 지금 | 42.9 | ≈21 | ≈64 | 1.7× 느림 |
| A1 + P1 + P2 | ≈23 | ≈21 (CPU와 공유되는 비용이면) | ≈44 | 비슷 — **못 이김** |
| A1 + P1 + P2 | ≈23 | ≈3 (P1이 같이 걷어내는 HTP 고유 비용이면) | ≈26 | **1.5× 빠름** |

두 줄 중 어느 쪽인지는 **P3 하나로 결정**된다. (§8의 커널 내부 항목 D1/DQ1/Q1까지 넣은 갱신 추정은 §8.4 — 프로파일 ≈23이 아니라 **≈16.6**까지 내려간다.) 그래서 순서는 **A1 (정확도 선행조건, 작음) → P3 (측정, 거의 공짜) → P1 (큰 작업, P3 결과로 가치 확정)**. P1을 P3 전에 시작하는 건 결과가 판정을 못 바꾸는 큰 작업을 먼저 하는 것이라 권하지 않는다.

---

## 6. 이 문서가 쓰인 계측 (모두 env-gated, 기본 off)

| env | 위치 | 하는 일 |
|---|---|---|
| `NNTR_HTP_PROFILE=2` | `htp_compute_ops.cpp` `HtpProfile` | 단계별 DSP µs + `mm<=` 잔차 + `arm staging memcpy` 줄 |
| `NNTR_L2_CHECK=1` | `l2CheckFinite` | gate_up의 행별 requant scale과 down 출력에서 non-finite 카운트 |
| `NNTR_L2_DIFF=1` | `HtpComputeOps::l2Diff` | split-call vs 레퍼런스(두 번 `mm_u8i4_layer` + 호스트 SwiGLU)를 실제 weight/activation으로 expert마다 SNR, worst row, `call_max_span`, `bin_flips`/`total_flips`/`block_flips`, `scale_diff` |
| `NNTR_L2_SHADOW=1` | 같은 함수 | fused 커널을 전부 실행하되 모델에는 레퍼런스 값을 넘김 — 값 vs 부수효과 판별 |
| `NNTR_M0_PROFILE=1` | `Lfm2MoELayer` | 레이어 벽시계 (P3의 출발점) |


---

## 7. 코드에서 확인된 병목의 원인 (2026-09-10, 소스 읽음)

§1의 숫자만으로는 "왜 느린가"까지는 안 나온다. 커널을 읽어 다섯 가지를 확정했다. 전부 `nntrainer/tensor/htp_backend/` 기준.

### 7.1 drain — weight DMA가 HMX와 **전혀 겹치지 않는다**

`hmx/hexkl_mm_u8i4_dma.c:800-803` (gate_up_swiglu_run), 같은 패턴이 `:302`, `:336`, `:420`, `:552` (layer_run / u8in / fused):

```c
hexkl_dma_ring_push2d(vtcm_base + w_off, h_gu->wh_bytes, ...);  // weight 3.67 MB 전체
hexkl_dma_ring_drain();                                          // 다 올 때까지 블로킹
// ... 그 다음에야 HMX 루프 시작
```

weight 3.67 MB를 VTCM으로 **전부 받고 나서** 첫 HMX를 발행한다. 그 111.9 µs 동안 HMX는 논다. gate_up의 mm이 185 µs, DMA가 112 µs이므로 겹치면 **완전히 숨는다** (down: mm 91 vs DMA 48, 역시 숨음). 링(`hexkl_dma_ring`)은 이미 descriptor 여러 개를 outstanding으로 잡을 수 있다 — 쓰지 않고 있을 뿐이다.

weight 타일 배치는 `w_off + (kt*n1_tiles + nt)*512` — k-major. nt 루프가 바깥이므로 N-타일 하나의 k-타일 64개는 stride `n1_tiles*512`로 흩어져 있고, 2D DMA 한 descriptor로 모을 수 있다 (64 rows × 512 B = 32 KB/N-tile, 112개). 깊이 2–4의 프리페치면 충분하다.

### 7.2 quant — 실제 HVX 일은 ~5 µs, 나머지 ~90 µs는 오버헤드

`hvx/hvx_quant_u8.c`. 호출 하나(gate_up 입력 64×2048)에 드는 것:

| 항목 | 줄 | 비용 추정 |
|---|---|---|
| `hvx_worker_pool_run` ×2 (params, pack) | `:120`, `:265` | 디스패치당 ~25 µs (7.4 참조) |
| `memset(out_ah, 0, m_pad*k)` | `:228` | 128 KB. 유효 행은 바로 덮어쓰이므로 **패딩 행 9개(18 KB)만** 지우면 된다 |
| `malloc` ×2 + `free` ×2 (vinv, vz) | `:252-253` | 호출마다 |
| 실제 min/max 스캔 + pack | | 113K floats / 32 lanes ≈ 3.5K + 14K vector ops ≈ **5 µs (5 워커)** |

측정 98.5 µs = 입력 quant(~50) + 출력 requant(~50). 출력 requant는 `out_ah`가 **FastRPC 출력 버퍼(ION, uncached DDR)** 라서 memset도 pack의 흩어진 store도 모두 uncached 쓰기다 — VTCM에 staging 후 DMA 한 번이 맞다.

### 7.3 dequant — 단일 스레드, N-타일마다 112번 호출, 행마다 스칼라 splat

`hmx/hexkl_mm_u8i4_dma.c:849` → `hvx/hvx_dequant_i32.c` `hvx_dequant_acc_tile_to_f32`. **pool을 쓰지 않는다.** 블록당 112번 호출되고, 호출마다 64행 × (act_scale splat + act_zp splat + load + 5 ops + store). 64×112 = 7168 row-tile × ~10 ops ≈ 72K ops ≈ **72 µs 단일 스레드** — 측정 100.5와 맞는다.

블록 하나의 accumulator 전체(64×3584 i32 = 917 KB)는 VTCM에 들어간다 (현재 사용 5.6 MB / 8 MB). `result_off`를 타일마다 전진시켜 112 타일을 다 읽어둔 뒤 **pool로 행 분할 한 번**에 dequant 하면 7.2K vector ops / 5 워커 ≈ 2 µs + 디스패치 1회. 행별 splat(vs/vz 64개씩)도 블록당 한 번만 만들면 된다.

### 7.4 worker pool — 잠자는 워커를 futex로 깨우는 비용이 일보다 크다

`hvx/hvx_worker_pool.c:72` `qurt_futex_wait` / `:191` `qurt_futex_wake`. 디스패치마다 워커 5개가 커널 스케줄러를 거쳐 깨어난다 — 수 µs~수십 µs. 호출당 디스패치 5회(입력 quant 2, swiglu 1, 출력 requant 2) × ~25 µs ≈ **125 µs = DSP 시간 612 µs의 20%**. 4 µs짜리 일을 5개 코어에 나누는 게 단일 스레드 18 µs보다 느리다.

swiglu 56.6 µs도 같은 구조다: 115K elements × ~30 ops ≈ 108 µs 단일 → 22 µs (5 워커) + 디스패치 ~30 = 52. 수학은 실재하고 오버헤드가 절반이다.

### 7.5 FastRPC — 64번 왕복, 활성화를 f32로 32번 재전송

§1·§4 P1. expert별 450 KB f32 in + 114 KB u8 out + 114 KB in + 450 KB out ≈ 1.1 MB × 32 = 36 MB/layer. 32 experts가 받는 행은 같은 444×2048 활성화의 순열이므로 u8로 한 번(0.9 MB)이면 된다.

---

## 8. 최적화 계획

### 8.1 원칙

- **측정 가능한 순서로.** Phase A는 API를 안 바꾸고 커널 내부만 고치므로 **지금 동작하는 two-dot 경로(L2 off)에서 바로 잰다.** 정확도 리스크 0.
- **A1이 P1 앞.** 배칭은 SwiGLU가 DSP에 있어야만 가능하고, DSP SwiGLU는 A1 없이는 문장을 깨뜨린다.
- **P3이 P1 앞.** §5. P1은 큰 작업이고, ARM ≈21 ms의 정체가 P1의 가치를 정한다.
- 한 번에 하나, 43 §6.3의 accept/reject 그대로: 생성 텍스트 일치 + 노린 단계가 예측만큼 움직였는가 + bracketed CPU control.

### 8.2 항목

| # | 항목 | 무엇을 | 근거 | 기대 (ms/layer) | 리스크 |
|---|---|---|---|---|---|
| **D1** | drain 파이프라이닝 | N-타일 열 단위 2D DMA descriptor(32 KB) 깊이 2–4 프리페치, 첫 타일 도착 즉시 HMX 발행. `hexkl_dma_ring`의 outstanding 기능 사용. 4개 call site 공통 | 7.1 | 5.13 → **≈0.5 (−4.6)** | DMA 완료 폴링이 HMX 발행 루프에 끼어드는 비용 — 타일당 1 µs vs 1.65 µs라 여유 있음 |
| **DQ1** | dequant 블록화 | 112 타일 acc를 VTCM에 모두 읽은 뒤 pool 행 분할 1회. 행별 vs/vz splat 블록당 1회 생성 | 7.3 | 4.90 → **≈1.2 (−3.7)** | VTCM +917 KB (5.6 → 6.5 MB, config 영역 아래) |
| **Q1** | quant 오버헤드 제거 | params+pack을 **한 디스패치**로 융합, `malloc` → pool ctx의 고정 VTCM scratch, memset은 패딩 행만, 출력 requant는 VTCM staging → DMA 1회 | 7.2 | 3.15 → **≈1.2 (−2.0)** | 없음 |
| **W1** | pool 하이브리드 대기 | 워커가 `futex_wait` 전에 N µs spin (`pause`). FastRPC 호출 중엔 DSP가 전용이라 spin 비용 없음; 호출 사이엔 futex로 떨어져 전력 보존. 작은 job은 인라인 임계값 | 7.4 | swiglu 1.81 → ≈1.0 (−0.8); Q1/DQ1의 디스패치 비용도 여기서 사라짐 | spin 시간 튜닝 필요 |
| **A1** | 비트 동일 SwiGLU | §3.3. `exp_ps` 다항식 + NR 역수를 순수 IEEE f32로 NEON/HVX 동일 구현, 기기 gtest `memcmp` | 43 §7 | flip 5/32 → **0** | ARM 경로 ≤1 ULP 변화 (허용 범위 확인됨) |
| **P3** | ARM ≈21 ms 실측 | `NNTR_M0_PROFILE` 확장: 라우터 softmax/top-k, 토큰 gather, expert별 텐서 뷰 생성, dispatch, scatter-add 분리 타이머 | §5 | **판정** | 없음 |
| **P1** | MoE 배칭 | IDL `mm_u8i4_moe_layer(handles[64], row_index[], row_counts[], act_ah, act_scale, act_zp, routing_w, out_f32)`. DSP 안에서 expert 루프: gather(u8 memcpy 113 KB) → gate_up HMX → dequant → SwiGLU → requant → down HMX → dequant → ×routing_w → scatter-add. expert e+1의 weight를 e의 계산 뒤에 프리페치 | 7.5 | transport 14.76 → ≈2.0, staging 1.4 → 0.3, 입력 quant 32× → 1×, HMX config 상각 → **≈ −15** | **VTCM 예산**: gate_up W 3.67 + down W 1.84 + act 0.9 + gate/up/acc scratch ≈1.5 = 7.9 MB — 8 MB에 빠듯. 2-region ping-pong(gate_up 계산 중 down W 프리페치, down 계산 중 다음 gate_up W 프리페치) 또는 gate_up W를 N 절반씩 스트리밍 |
| P4 | 등록 bake 캐시 | §4 | | 1.8 s → ≈0.1 s /layer | 전체 모델 실행의 전제 |

### 8.3 실행 순서

```
Phase A  (커널 내부, two-dot 경로에서 측정, 정확도 리스크 0)
  W1 → Q1 → DQ1 → D1        각각 한 번씩 프로파일, 노린 컬럼이 움직였는지 확인
                              42.9 → ≈31.8

Phase B  (정확도)
  A1                         기기 gtest memcmp == 0 → kFusedSwigluEnabled=true → 텍스트 일치

Phase C  (판정)
  P3                         ARM ≈21이 HTP 고유인지 CPU 공유인지

Phase D  (P3이 "HTP 고유"라고 답했을 때만)
  P1                         31.8 → ≈16.6, ARM ≈21 → ≈3
  P4                         전체 모델 기동
```

W1을 먼저 하는 이유: Q1/DQ1/swiglu 셋 다 디스패치 비용을 안고 있어서, W1 하나가 셋의 측정 기준선을 바꾼다. W1 없이 Q1을 재면 Q1의 효과를 과소평가한다.

### 8.4 갱신된 도달 추정

| 단계 | transport | mm | drain | dequant | quant | acc | swiglu | staging | **프로파일** | ARM | **벽시계** | vs CPU 38 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 지금 | 14.76 | 8.83 | 5.13 | 4.90 | 3.15 | 2.95 | 1.81 | 1.4 | **42.9** | ≈21 | ≈64 | 1.7× 느림 |
| Phase A | 14.76 | 8.83 | 0.5 | 1.2 | 1.2 | 2.95 | 1.0 | 1.4 | **≈31.8** | ≈21 | ≈53 | 1.4× 느림 |
| + P1 | 2.0 | 8.3 | 0.5 | 1.2 | 0.4 | 2.95 | 1.0 | 0.3 | **≈16.6** | ≈3 (HTP 고유일 때) | **≈20** | **1.9× 빠름** |
| + P1 | | | | | | | | | ≈16.6 | ≈21 (CPU 공유일 때) | ≈38 | 동률 |

§0의 `mm`-only 천장 8.83에 대해 ≈16.6은 1.9배 — acc(벤더 2.95)와 drain 잔여를 빼면 남는 오버헤드는 ≈4 ms다. 그 이상은 커널이 아니라 라우팅 구조(M 자체)의 문제다.

**모든 기대값은 소스와 §1 측정에서 유도한 추정이며, 기기에서 잰 것이 아니다.** Phase A의 각 항목이 끝날 때마다 §1 표를 갱신한다.

---

## 9. Phase A, 구현됨 (2026-09-10) — **미측정**

§8.3의 Phase A 네 항목이 브랜치에 들어갔다. **기기에서 빌드된 적도 측정된 적도 없다.** 각 커밋은 stub 헤더(`hexagon_types` / `hvx_hexagon_protos` / `hexkl_micro` / `qurt` / `HAP_perf`)를 상대로 gcc 문법·타입 검사만 통과했다 — 호출 지점 배선 실수는 잡지만 그 이상은 아니다.

| # | 커밋 | 실제로 한 것 | 계획과 다른 점 |
|---|---|---|---|
| W1 | `0506834` | 워커가 `qurt_futex_wait` 전에 최대 400 µs spin (`HAP_perf` qtimer로 예산 측정, 64 pause마다 확인) | 계획대로. `hvx_worker_pool_run`의 무조건 `futex_wake`는 남겨둠 — 없애려면 sleeper 카운트가 필요하고, 그건 상수의 ponytail 주석에 업그레이드 경로로 적어둠 |
| Q1 | `950119c` | params+pack 한 디스패치. **분할 축을 k-타일 → 4행 그룹으로** 바꾼 것이 융합을 가능하게 함(행의 scale/zp는 그 행에만 의존 → 배리어 불필요). malloc 2개 제거(축이 바뀌어 행별 벡터가 지역변수가 됨), memset은 `[m_valid, m_pad)`만, `k % 32` 가드 추가 | 계획엔 없던 **읽기 지역성 개선**이 따라옴: 예전엔 params가 전 행을 훑고 pack이 다시 훑어 2×512 KB 콜드 리드였는데, 이제 한 행의 두 패스가 연속 |
| DQ1 | `035b12d` | `act_scale[m]`/`act_zp[m]` splat을 타일 루프 밖으로. 행 블록당 1회(`hvx_dequant_prepare_rows`), 예전엔 타일마다 = 블록당 14,336회 | **계획의 배칭 버전은 안 함.** 112개 acc 타일을 VTCM에 모아 한 디스패치로 dequant하려면 ~917 KB가 필요한데 `fused_run`의 6-region 레이아웃이 못 내주고, 커널 3개의 HMX 발행 루프를 뜯어야 함. 그 가치는 W1이 디스패치 비용을 깎은 뒤의 프로파일이 정할 문제 |
| D1 | `7b125c2` | weight DMA를 **activation quant보다 먼저 발행**하고 뒤에서 대기. quant는 weight가 필요 없으므로 `min(quant, dma)`만큼 공짜로 겹침 | **계획의 청크 파이프라인은 안 함.** descriptor 하나만 기다리는 프리미티브가 필요한데, 그게 안전한지는 `dmlink`가 체인 끝까지 가서 멈춘 DMA 엔진을 깨우는지에 달림. `hexkl_dma_ring_drain()`의 주석("engine idle after drain -- the next push must dmstart")은 저자가 **안 깨운다**고 결론냈음을 뜻하고, 그러면 순진한 wait-one은 간헐적 행이나 조용한 stale weight가 된다. 매뉴얼과 기기가 필요한 질문이지 여기서 추측할 것이 아님 |

**D1의 남은 절반, 다음 사람을 위해:** weight는 k-major(`tile = kt*n_col_tiles + nt`)라 **열 구간도 2D descriptor 하나**로 표현된다 — src/dst stride `n_col_tiles*512`, row size `(c1-c0)*512`, rows `k_tiles`. 따라서 **push+drain만으로 만든 G-way 청크 프롤로그**가 이미 전송의 1/G만 노출시킨다. per-descriptor 대기는 그 너머로 갈 때만 필요하다.

**손대지 않은 것:** `fused_run`(휴면 중이고, gate_up drain이 down weight 프리페치도 겸하고 있어 같은 재배치가 다른 변경이 됨), `hexkl_mm_u8i8_dma.c`의 D1(ring reset이 quant 뒤에 있어 그것도 옮겨야 하고, MoE 경로가 아님).

### 9.1 측정 방법

**`kFusedSwigluEnabled`를 `true`로 두고 재는 것을 권한다** — §1·§2의 모든 숫자가 그 구성에서 나왔으므로 컬럼이 직접 비교된다. 문장이 깨지는 것은 이 측정에 상관없다(성능만 본다). 출하 경로 숫자는 A1 이후 `false`로 따로 재면 된다.

한 번의 실행으로 네 항목의 귀속이 갈린다 — 각각이 움직여야 할 컬럼이 다르기 때문:

| 항목 | 움직여야 할 컬럼 | §1 기준 | 예상 |
|---|---|---|---|
| W1 | `quant` `swiglu` `dequant` 전부 (디스패치 비용) | — | 셋 다 내려감 |
| Q1 | `quant` | 3.15 ms | ≈1.2 |
| DQ1 | `dequant` | 4.90 ms | ≈2.9 |
| D1 | `drain` | 5.13 ms | ≈3.5 (quant가 가려주는 만큼만) |
| — | `mm` `acc` | 8.83 / 2.95 | **안 움직여야 정상** |

`mm`이나 `acc`가 움직였다면 무언가 잘못된 것이다 — 넷 중 어느 것도 HMX 발행이나 accumulator 읽기를 건드리지 않았다.

기대 합계: 프로파일 42.9 → **≈33**. §8.4의 Phase A 예측(≈31.8)보다 조금 높은데, DQ1과 D1이 계획의 절반씩만 들어갔기 때문이다.


---

## 10. Phase A 측정 결과 (2026-09-10, 기기) — **넷 중 하나만 살아남음**

§9.1 프로토콜대로 `kFusedSwigluEnabled=true`, `NNTR_HTP_PROFILE=2` 한 번 실행.

| 단계 (gate_up, µs/call) | §2 기준 | Phase A | 판정 |
|---|---:|---:|---|
| **drain** | 111.9 | **22.8** | **D1 성공, −80%** |
| **quant** | 98.5 | **295.5** | **Q1 회귀, +200%** |
| dequant | 100.5 | 99.2 | DQ1 효과 없음 |
| swiglu | 56.6 | 54.9 | — |
| `mm` (control) | 185.3 | 196.2 | +6% |
| `acc` (control) | 59.3 | 58.3 | −2% |
| **layer calls total** | **42.9 ms** | **50.7 ms** | 순 악화 |

**control이 판정을 가능하게 했다.** 이 실행에서 기기는 심하게 스로틀 중이었다 — decode 14.7 TPS(최고 38.7 대비 38%)이고 decode에는 HTP가 전혀 없다. 그런데 drift라면 모든 단계가 함께 올라가야 하는데 `mm`·`acc`·`dequant`는 안 움직였다. 따라서 quant의 3배는 기기가 아니라 코드다. §8.2에 `mm`/`acc`를 control로 지정해둔 것이 이 한 번의 실행으로 네 항목을 갈라낸 이유다.

### 10.1 Q1이 왜 회귀했나 — 쓰기 패턴이 뒤집혔다

분할 축을 k-타일 → 4행 그룹으로 옮긴 것이 원인이다.

| | k-타일 분할 (원래) | 행 그룹 분할 (Q1) |
|---|---|---|
| 워커의 쓰기 | 한 2048B 타일 안을 `r0*32`로 **연속** | kt마다 → **2048B stride로 64번 흩뿌림** |

그리고 `gate_up_swiglu_run`의 **출력** requant destination `out_ah`는 FastRPC 출력 버퍼, 즉 **uncached DDR**이다 — §7.2가 이미 그렇게 적어놓았는데도 축을 바꿨다. uncached DDR에 128B씩 2 KB 간격으로 흩뿌리면 write combining이 깨진다. 근거로 내세운 "읽기 지역성 개선"은 성립하지 않았다: 입력(`act_f32`)도 uncached DDR이라 두 번째 패스가 warm해질 캐시가 애초에 없다.

### 10.2 DQ1은 왜 효과가 없었나 — 명령어를 세고 비용을 안 쟀다

행별 splat 2개를 16 KB 스크래치로부터의 **벡터 로드** 2개로 바꾼 것이다. splat은 명령어 하나짜리 레지스터 연산이다. "타일 dequant 연산 수의 40%"라는 추정은 명령어 개수만 세고 가중치를 주지 않은 것이고, 레지스터 연산을 메모리 트래픽으로 바꾼 셈이라 잘해야 본전이다.

### 10.3 조치

- **D1 유지** — 깨끗한 트리 위에 재적용, 측정값을 주석에 박아둠 (`cb05ec0`)
- **Q1·DQ1 전부 롤백** — memset 축소도 포함. 128 KB 순차 memset 하나를 32B짜리 576번으로 쪼갠 것도 같은 실수의 축소판이고, 그것 역시 단독 측정된 적 없다
- **W1 유지, 판정 보류** — quant가 오염돼 읽을 수 없었다. 다음 측정에서 나온다

### 10.4 이 실행이 가르쳐준 것

§8.2의 기대값 세 개 중 둘이 틀렸고, **둘 다 같은 방식으로 틀렸다: 명령어나 바이트 개수를 세고 그것이 시간에 비례한다고 가정했다.** 이 커널에서 실제로 시간을 지배하는 것은 **어느 메모리에 어떤 패턴으로 접근하는가**다 — uncached DDR인지 VTCM인지, 연속인지 흩어지는지. D1이 성공한 이유도 같은 축에서 설명된다: 그것만이 유일하게 "일을 줄이자"가 아니라 "대기를 다른 일 뒤에 숨기자"였다.

다음 항목을 평가할 때 물어야 할 것: **이 변경이 접근하는 메모리의 종류나 패턴을 바꾸는가?** 바꾼다면 명령어 수 논증은 무효다.


---

## 11. 롤백 확인 측정 (2026-09-10, 2차) — Phase A는 **넷 중 하나**로 끝났다

| gate_up (µs/call) | §2 기준 | Phase A | **롤백 후** |
|---|---:|---:|---:|
| quant | 98.5 | 295.5 | **99.3** |
| drain | 111.9 | 22.8 | **67.9** |
| swiglu | 56.6 | 54.9 | 55.0 |
| dequant | 100.5 | 99.2 | 91.4 |
| acc | 59.3 | 58.3 | 56.8 |
| mm | 185.3 | 196.2 | 186.3 |
| **dsp 합** | **612.2** | 726.8 | **556.8** |
| transport | 272.1 | 353.5 | 543.4 |

`quant`가 99.3으로 정확히 복귀했다 — Q1이 원인이었다는 증명이다.

### 11.1 D1의 정직한 값은 22.8이 아니라 67.9다

1차의 22.8은 **Q1의 회귀가 quant를 295 µs로 부풀려 DMA 전체를 덮어버린 결과**였다. quant가 정상으로 돌아오자 67.9가 남았다. D1의 실제 값은 **111.9 → 67.9 (−39%, −44 µs/call, 레이어당 −1.4 ms)**.

예상은 "min(quant, dma) ≈ 99 µs를 가림"이었는데 44만 가렸다. 원인 추정: quant도 `act_f32` 512 KB를 DDR에서 읽으므로 **weight DMA와 같은 DDR 대역폭을 놓고 경쟁**한다. 둘 다 DDR을 두드리면 완전히 병렬이 되지 않는다. (추정이며 측정되지 않았다.)

### 11.2 W1도 효과 없음 — 세 번째 같은 실수

`quant 99.3 vs 98.5`, `swiglu 55.0 vs 56.6`, `dequant 91.4 vs 100.5`. §7.4는 futex wake를 25 µs로 잡고 호출당 5 dispatch를 곱해 "DSP 시간의 20%"라고 했다. 입력 quant만 해도 dispatch 2회이므로 ~50 µs가 보여야 하는데 **0.8 µs**다.

Q1·DQ1과 동일하게 **이벤트를 세고 시간이 따라온다고 가정한** 논증이었다. 3전 3패. spin은 전력을 쓰고(워커 5개 × 최대 400 µs) 측정 가능한 이득이 없으므로 되돌렸다.

### 11.3 이 실행에서 신뢰할 수 있는 것과 없는 것

기기가 계속 심하게 스로틀 중이다 — **decode 13.6 TPS**(최고 38.7 대비 35%, decode에는 HTP가 전혀 없음), prefill 68.7 TPS(최고 146 대비 47%), `convert to registry` 1007.8 ms(이전 350–743).

- **신뢰 가능**: DSP 내부 단계(`quant` `swiglu` `dequant` `acc` `mm` `drain`, `dsp` 합). DSP는 별도 클럭 도메인이고, 이 값들이 전부 §2 기준선 근처로 돌아왔다는 사실 자체가 그 증거다.
- **신뢰 불가**: `transport`(272 → 543), `host`, `layer calls total`(50.7). transport는 ARM↔DSP 왕복이라 **ARM 클럭과 스케줄링에 직접 의존**한다. 제 변경은 FastRPC를 건드리지 않았는데 2배가 됐다.

**앞으로 프로파일을 읽을 때의 규칙**: `decode TPS`를 스로틀 지표로 먼저 본다. 30 미만이면 `transport`/`host`/`total`은 비교에 쓰지 않고 DSP 내부 단계만 본다.

### 11.4 Phase A 최종

| 항목 | 결과 |
|---|---|
| **D1** | **유지** — drain −39%, DSP 합 612.2 → 556.8 (−9%) |
| Q1 | 롤백 (quant +200%) |
| DQ1 | 롤백 (효과 없음) |
| W1 | 롤백 (효과 없음, 전력 비용) |

§8.4의 "Phase A ≈31.8 ms" 예측은 틀렸다. 실제로 얻은 것은 DSP 시간 −9%뿐이고, 벽시계 효과는 기기가 식은 뒤에 다시 재야 한다.

### 11.5 이제 항목을 고르는 기준

§10.4의 규칙이 세 번 연속 맞았다. 강화해서 다시 적는다:

> **이벤트나 바이트를 세어 "N번 × 개당 T = 총 N·T"로 만든 논증은 이 커널에서 세 번 다 틀렸다.** 그 형태의 추정은 이제 근거로 인정하지 않는다. 실제로 시간을 지배하는 것은 (a) 어느 메모리를 어떤 패턴으로 접근하는가, (b) 무엇이 무엇을 기다리는가다.

이 기준으로 남은 항목을 다시 보면:

| 항목 | 이 기준에서 | 판정 |
|---|---|---|
| **P1 (배칭)** | 와이어 바이트를 36 MB → 4.5 MB로 **줄이는 것 자체**가 목적. 개수 곱하기 논증이 아니라 전송량 자체의 감소 | **안전, 다음 순위** |
| **P4 (등록 캐시)** | DSP WH bake를 **아예 안 하는 것**. 1.8 s/layer가 0.1로 — 없애는 것이지 빠르게 하는 것이 아님 | **안전** |
| P2 (dequant 튜닝) | DQ1과 같은 함정. 프로브로 **어디서 시간이 가는지 먼저 재기 전엔 금지** | 보류 |
| A1 (비트 동일 SwiGLU) | 성능 항목이 아니라 정확도 항목. 기준 무관 | 유효 |


---

## 12. A1 — 결정적 SwiGLU, **전제 검증까지 포함** (2026-09-10, 미측정)

§3.3의 "양쪽이 비트 단위로 같은 알고리즘을 돌리게 한다"를 구현했다. 다만 **아직 어느 경로에도 연결하지 않았다** — 이 접근 전체가 기대는 전제 하나가 검증되지 않았기 때문이다.

### 12.1 전제

`hvx_exp_f32.h`의 주석:

> "on real HVX hardware, chained Vsf multiply-adds lose precision that qf32 retains"

이 문장은 두 가지로 읽힌다.
1. **"f32는 매 연산마다 반올림하고 확장 포맷은 안 한다"** — 당연한 관찰. 그렇다면 Vsf는 IEEE이고 비트 동일이 가능하다.
2. **"Vsf는 IEEE 정확 반올림이 아니다"** — 그렇다면 이 하드웨어에서 비트 동일은 **원리적으로 불가능**하고 A1 전체가 죽는다.

Phase A에서 세 번 틀린 뒤라 추측하지 않는다. **구현과 함께 이 전제를 직접 재는 테스트를 넣었다.**

### 12.2 들어간 것

| 파일 | 내용 |
|---|---|
| `hvx/hvx_swiglu_det.h` | 결정적 exp/역수/SwiGLU. **plain Vsf만** — FMA 없음, qf32 없음, 나눗셈 없음. 상수와 **연산 순서**가 계약의 일부 |
| `test/htp/nntr_hvx.idl` | `swiglu_det_f32(gate, up, rout res, rout exp_out, rout recip_out)` — 중간값 둘을 같이 반환 |
| `test/htp/nntr_hvx_softmax.c` | skel 엔트리. `exp_f32`가 softmax에서 분리된 것과 같은 이유 |
| `test/unittest/unittest_hvx_softmax.cpp` | `HvxSwigluDet.MatchesScalarBitExact` — 스칼라 레퍼런스와 **비트 비교**, 단계별 카운트 |

알고리즘 요약 (전문은 헤더 주석):
- exp: `x`를 [−88, 85]로 클램프 → `k = rne(x·log2e)` → `r = (x − k·ln2_hi) − k·ln2_lo` → 7차 Horner (`Σ rⁿ/n!`) → 지수 필드에 `k` 주입 → underflow 가드
- 역수: 매직 시드 `0x7EF311C2 − bits(d)` + Newton-Raphson 3회, 전부 plain f32
- SwiGLU: `(g · recip(1 + exp(−g))) · u`

정확도는 타협이 아니다: `|r| ≤ ln2/2`라 7차 절단 오차는 `r⁸/8! ≤ 5.2e-9`(f32 eps 1.2e-7보다 훨씬 작음), NR 3회는 시드의 ~6%를 ~1.7e-10로 만든다. 둘 다 마지막 비트 아래다.

스칼라 레퍼런스는 매 연산을 `volatile`을 거쳐 저장한다 — 컴파일러가 곱셈과 덧셈을 FMA로 합치면 DSP가 두 번 반올림하는 곳에서 한 번만 반올림하게 되고, 그 불일치는 하드웨어가 아니라 컴파일러의 것이 된다.

### 12.3 아직 연결하지 않은 이유

Vsf가 IEEE가 아니라면 `hvx_exp_det_sf`는 지금의 qf32판 `hvx_exp_sf`보다 **덜 정확하다.** fused 경로가 휴면 중이라 사용자 영향은 없지만, 검증 전에 더 나쁜 구현을 넣어두는 것은 희망에 거는 것이다. **테스트가 게이트다.**

- 통과하면 → `hvx_swiglu_f32.c`를 det판으로 전환하고, ARM `nntrainer::neon::swiglu`도 같은 스펙으로 다시 쓴다(그때 CPU 경로가 ≤1 ULP 바뀌므로 별도 텍스트 검증 필요)
- 실패하면 → 단계별 카운트가 exp인지 역수인지 마지막 곱인지 말해준다. exp/역수가 둘 다 틀리면 Vsf가 IEEE가 아니라는 뜻이고, **A1은 이 하드웨어에서 불가능**하다. 그 경우 fused 경로는 영구히 포기하거나, 정확도 기준 자체를 재검토해야 한다

### 12.4 돌리는 법

skel과 gtest 둘 다 재빌드가 필요하다 (새 IDL 엔트리).

```bash
source ~/workspace/Hexagon_SDK/6.4.0.2/setup_sdk_env.source
cd ~/workspace/nntrainer
bash nntrainer/tensor/htp_backend/generate_stub.sh      # 새 엔트리 반영
HEXKL_ROOT=~/workspace/hxkl-beta2/hexkl_addon ./test/htp/build.sh
./test/htp/run_u8i4_layer_on_device.sh                  # skel push + gtest 빌드/실행
```

`unittest_hvx_softmax`의 `HvxSwigluDet.MatchesScalarBitExact` 결과와 `SWIGLU_DET_FIELD bad_exp=.. bad_recip=.. bad_out=..` 줄이 답이다. 기기 스로틀과 무관한 정확도 테스트라 지금 상태에서도 유효하다.

### 12.4 게이트 결과 (2026-09-10, 기기)

```
SWIGLU_DET_FIELD bad_exp=5 bad_recip=0 bad_out=0 of 8192
```

**전제는 통과했다.** `hvx_recip_det_sf`는 Vsf 곱셈 3회 + 뺄셈 3회 + 곱셈 3회의 9단 체인인데 8192개가 전부 비트 일치했다. Vsf가 IEEE 정확 반올림이 아니었다면 불가능한 결과다. §12.1의 두 해석 중 **1번이 맞다** — `hvx_exp_f32.h`의 문장은 "f32는 매 단계 반올림한다"는 평범한 관찰이었다.

`bad_exp=5`는 산술 포맷 차이가 아니라 **HVX 쪽 술어 하나의 오프바이원**이었다. 헤더의 스펙과 스칼라 레퍼런스는 `k + exp_field(p) <= 0`인데 HVX 코드만 `Q6_Q_vcmp_gt_VwVw(zero, sum)`, 즉 `< 0`으로 짜여 있었다. 호스트에서 재현:

| exp 인자 | k | p_exp | k+p_exp | ref (`<=0`) | HVX (`<0`) |
|---:|---:|---:|---:|---|---|
| −87 | −126 | 127 | 1 | 1.64581e-38 | 1.64581e-38 |
| **−88** | −127 | 127 | **0** | **0** | **3.54261e-40** |

`k+p_exp == 0`이면 비트 덧셈 결과의 지수 필드가 0이라 참값(6.05e-39)의 1/17짜리 쓰레기가 나온다 — 가드가 발동하는 게 맞고, **레퍼런스가 맞고 HVX가 틀렸다.** 테스트 입력 중 exp 인자가 정확히 −88이 되는 것은 `i=64..67`(클램프)과 `i=68`(g=88, 클램프 없이 딱 −88) — 정확히 5개.

`bad_recip=0`/`bad_out=0`인 이유도 같은 사실에서 나온다: `1 + 3.5e-40`은 f32에서 정확히 `1.0f`라 그 차이가 역수 단계에서 흡수된다. 즉 이 버그는 지금은 무해하지만, 다른 입력 분포에서는 그렇지 않다.

### 12.5 ARM 절반, 그리고 실제 위험했던 것

A1은 양쪽이 같은 스펙을 돌려야 성립하므로 ARM 구현이 필요하다. `nntrainer::neon::swiglu`는 **건드리지 않는다** — 모든 모델이 쓰는 공용 API라 blast radius가 넓고, 그 자체로는 아무 문제가 없다. 대신 `nntrainer/tensor/htp_backend/swiglu_det.h`에 스펙의 호스트 구현을 두고, 테스트도 자기 사본을 버리고 이 헤더를 쓰게 했다. **사본이 세 벌이면 드리프트가 난다 — §12.4의 버그가 바로 그것이다.**

진짜 위험은 다른 데 있었다. **`#pragma clang fp contract(off)`는 이 커널을 보호하지 못한다.** `vmulq_f32`/`vaddq_f32`는 곱셈과 덧셈이 `arm_neon.h` 안에 텍스트로 쓰여 있어서 블록 스코프 프래그마가 덮지 못한다. aarch64 `-O3 -ffp-contract=fast`로 빌드하면 프래그마가 있으나 없으나 **fmla 12개가 똑같이** 나온다. 그리고 `Applications/CausalLM/jni/Android.mk:41`은 **`-ffast-math`**로 빌드한다 — 가정이 아니라 실제 설정이다.

작동하는 것은 각 결과를 벡터 레지스터에 묶는 빈 asm(`__asm__("" : "+w"(r))`)이다. 곱셈에만 걸었을 때 `-ffast-math`에서 fmla 하나가 살아남았는데, `y*(2 − d·y)`가 `2y − d·y·y`로 분배되기 때문이었다. 그래서 f32 곱셈·덧셈·뺄셈 **전부**를 감쌌다.

세지 않고 확인했다 (§11.5의 규칙 그대로 — 이건 소스의 성질이 아니라 컴파일러의 성질이다): aarch64 gcc 13과 clang 18, `-O3` / `-ffp-contract=fast` / `-ffast-math` / `-Ofast` 네 설정 전부에서 fmla 0개, 그리고 게이트 테스트와 같은 8192개 입력에서 NEON과 스칼라가 **비트 동일**(qemu-aarch64 실행 — 에뮬레이션이지 이 기기가 아니다). 기기 확인은 `SwigluDetNeon.MatchesScalar`가 한다.

### 12.6 게이트 통과 (2026-09-10, 기기)

```
SWIGLU_DET_FIELD      bad_exp=0 bad_recip=0 bad_out=0 of 8192
SWIGLU_DET_NEON_FIELD bad=0 of 8192
```

DSP == 스칼라, NEON == 스칼라 ⇒ **DSP == NEON**, 실기기에서. A1의 전제와 구현이 모두 확인됐다. 술어 수정(`< 0` → `<= 0`)이 정확히 그 5개를 잡았고 다른 것은 건드리지 않았다.

### 12.7 배선 — 그리고 SwiGLU가 셋이었다는 사실

배선하며 드러난 것: 이 코드베이스에는 **서로 다른 SwiGLU 구현이 셋** 돌고 있었다.

| 위치 | 구현 | 쓰이는 곳 |
|---|---|---|
| `neon_impl.cpp` `nntrainer::swiglu` | `exp_ps`(5차) + `vdivq_f32` | 모델의 실제 ARM 경로 |
| `htp_compute_ops.cpp:745` | `std::exp` (스칼라) | `NNTR_L2_DIFF`의 레퍼런스 |
| `hvx_swiglu_f32.c` | `hvx_exp_sf`(qf32, 7차) + NR 역수 | fused DSP 경로 |

**`total_flips`가 잘못된 것을 재고 있었다.** l2Diff는 fused DSP를 `std::exp` 레퍼런스와 비교했는데, 모델이 실제로 쓰는 것은 `exp_ps`다. 0이 아닌 값이 나와도 "fused가 모델의 실제 경로와 어긋났다"인지 "둘 다 libm과 어긋났다"인지 구분되지 않았다.

배선한 곳 네 군데:

| 파일 | 변경 |
|---|---|
| `hvx/hvx_swiglu_f32.c` | `hvx_swiglu_det_sf`로 전환. qf32 역수와 `exp_top` 클램프 삭제(det판이 자체 클램프를 가짐). 스칼라 tail도 `expf` → `swiglu_det_one` — tail만 다른 근사를 쓰면 그 열에서 발산이 되살아난다 |
| `htp_compute_ops.cpp` | l2Diff 레퍼런스 → `swiglu_det_one`. 이제 `total_flips == 0`이 정확히 필요한 성질이다 |
| `lfm2_moe_layer.cpp` | `nntrainer::swiglu` 3곳 → `swiglu_det`. **`neon::swiglu` 자체는 건드리지 않는다** — 모든 모델이 쓰고 그 자체는 문제가 없다 |
| `jni/Android.mk`, `test/htp/build.sh` | 공용 헤더 하나를 위한 include 경로 |

`kFusedSwigluEnabled`는 **아직 `false`**다.

### 12.8 두 단계로 재는 이유

이번 push는 기본 실행에서 딱 하나를 바꾼다: **모델의 ARM SwiGLU 스펙.** fused는 꺼져 있으니 DSP 변경은 잠자고, l2Diff는 env-gated라 잠잔다.

**단계 A (지금 잴 것)** — fused off, 그냥 실행. 텍스트가 여전히 정상 3문장 요약이어야 한다. 이것이 격리하는 질문: *모델이 `exp_ps` 대신 det SwiGLU를 견디는가?* det는 `exp_ps`와 ≤1 ULP 다르고, 22 레이어 greedy decoding에서 그 정도가 토큰을 바꿀 수 있다는 것이 바로 이 조사 전체의 교훈이다. 그러니 가정하지 않고 잰다.

- 정상이면 → 단계 B
- 깨지면 → 모델이 그만큼 민감하다는 뜻이고, 그 자체가 답이다. A1은 fused를 살리지 못하고 되돌린다

**단계 B** — `kFusedSwigluEnabled = true`, `NNTR_L2_DIFF=1`. **판정은 텍스트가 아니라 `total_flips`다**: 32 호출 전부 0이어야 한다(지금은 5개에서 1). 텍스트는 그 다음 확인이지 기준이 아니다 — 텍스트가 우연히 맞을 수도 있고, 그때 우리는 아무것도 배우지 못한다.

---

## 13. 갱신된 계획 (2026-09-10, A1 게이트 통과 후) — §4/§8을 대체

### 13.1 먼저 천장부터 — "CPU보다 훨씬"이 어디까지 가능한가

측정된 숫자만으로 계산한다.

**MoE FFN 하나, ms/layer:**

| 단계 | 지금 (fused, §1) | D1 후 (측정) | + P1 배칭 | + P2 dequant | 근거 |
|---|---:|---:|---:|---:|---|
| transport | 14.76 | 14.76 | **2.5** | 2.5 | 와이어 36 → 4.5 MB (u8 in 0.9 + f32 out 3.6), 호출 64 → 1 |
| mm (HMX) | 8.83 | 8.83 | 8.0 | 8.0 | config 상각. M=55→64 패딩 14%는 라우팅이 정하는 값이라 못 줄임 |
| drain (weight DMA) | 5.13 | **1.16** | **0.5** | 0.5 | 176 MB / 34 GB/s = 5.1 ms인데 mm 8.8 ms **뒤에 숨길 수 있다** (D1의 일반화: expert e+1의 weight를 e의 mm 중에 당김). D1 후 값은 식은 기기에서 재측정한 것 — 43 §7 2026-09-10 단계 A 행 |
| dequant | 4.90 | 4.90 | 4.90 | **2.0** | 2.3 G elem/s는 HVX 32-lane치고 4~5배 느림. 프로브 먼저 |
| quant | 3.15 | 3.15 | **0.7** | 0.7 | 1776행(expert별 gather) → 444행 한 번 |
| acc | 2.95 | 2.95 | 2.95 | 2.95 | 벤더 코드 |
| swiglu | 1.81 | 1.81 | 1.81 | 1.81 | det판 비용 미측정, qf32판과 비슷할 것 |
| staging | 1.4 | 1.4 | 0.3 | 0.3 | u8 한 번 |
| **프로파일 합** | **42.9** | **38.9** | **19.3** | **16.4** | 42.9는 D1 이전이자 온도 불명. 같은 온도의 fused 기준은 단계 B가 준다 |
| 미계측 ARM | ≈21 | ≈21 | **3 또는 21** | 3 또는 21 | **P3이 정한다** |
| **벽시계** | ≈64 | ≈60 | **22 또는 40** | **19 또는 37** | |
| vs CPU 38 | 1.7× 느림 | 1.6× 느림 | **1.7× 빠름** 또는 비슷 | **2.0× 빠름** 또는 비슷 | |

**MoE FFN만 HTP로 보내면 레이어당 최대 ≈1.7×다.** 그 이상은 이 커널 안에 없다 — mm 8.0 + acc 2.95는 벤더/실리콘 바닥이고, 그 둘만으로 11 ms다.

**전체 prefill로 환산하면 (Amdahl):** CPU prefill ≈1590 ms 중 MoE FFN이 841 ms (53%, M0 측정). MoE FFN을 22 ms/layer로 만들면 22 레이어 = 484 ms → prefill 1233 ms → **1.29×**. 이상적으로 12 ms까지 밀어도 264 ms → 1013 ms → **1.57×**.

**decode는 이 계획으로 전혀 안 빨라진다.** M=1은 weight 대역폭 바운드라 HTP 25.8 vs CPU 25.7 TPS로 문서 확정(43 §7). DDR은 어느 쪽에서 읽어도 DDR이다. 최근 13~17 TPS는 기기 스로틀이지 코드가 아니다.

**결론: "CPU보다 훨씬 빠르게"(2~3×)는 MoE FFN 최적화만으로는 도달 불가능하다.** 나머지 47%(attention, dense FFN, norm)도 HTP로 가야 하고, 그러면 activation이 레이어 사이에 DSP에 머물러 transport 자체가 사라진다. 그것이 §13.4의 Phase 5이고, 이 문서 밖(00~40번 문서, HTP attention)의 작업이다. **이 사실을 먼저 받아들이고 순서를 정한다.**

### 13.2 지금까지 실제로 남은 것 — 정직한 결산

| 항목 | 결과 | 레이어당 |
|---|---|---|
| D1 weight DMA 선행 | **생존** (4회 확인) | **−3.97 ms** (식은 기기에서 정정; 스로틀 상태에서 −1.6으로 과소평가했었다) |
| Q1 / DQ1 / W1 | 회귀·무효 → 되돌림 | 0 |
| A1 결정적 SwiGLU | 게이트 통과, 단계 A/B 측정 대기 | 0 (정확도; P1의 전제) |
| P3 / P1 / P2 / P4 | **시작 안 함** | — |

성능에 기여한 건 D1 하나(−1.6 ms, 4%). 시간의 대부분은 (a) 세어서 만든 논증 셋의 실패, (b) L2 정확도 근본 원인, (c) A1에 갔다. (b)와 (c)는 P1의 전제라 회피 불가능했지만, **P3(코드 ≈0)을 먼저 했어야 했다** — §5에서 그렇게 적어놓고 안 지켰다.

### 13.3 이번 라운드가 바꾼 규칙

§11.5의 카운팅 규칙에 더해서:

1. **한 실행에 한 변수.** fused 플래그 + det 배선이 같이 움직인 실행은 아무것도 답하지 못했다(§12.8). 플래그 상태는 실행 전에 `grep`으로 확인하고, 프로파일의 `swiglu` 컬럼으로 사후 검증한다.
2. **컴파일러는 소스의 일부다.** `#pragma clang fp contract(off)`가 안 통했고 `-ffast-math`가 실제 빌드 플래그였다(§12.5). 비트 동일이 필요한 커널은 emitted asm을 세고 에뮬레이터로 돌려본다.
3. **빌드 papercut은 즉시 고친다.** 잘못된 기본 경로(`build.sh`)와 복사된 include 목록(`Android.mk`)이 한 세션에서 측정 세 번을 날렸다. 그 비용은 실제 작업과 같다.
4. **온도 게이트:** decode < 30 TPS면 transport/host/벽시계는 비교 불가, DSP 내부 단계만 유효. 세 실행에서 dsp는 612→560→558로 안정, transport는 272→2686→527로 요동.
5. **SNR은 양자화 앞에서 무의미하다.** 양자화기에 들어가는 값은 비트 비교로만 게이트한다.

### 13.4 순서 — 각 단계의 비용은 "기기 실행 횟수"로 센다

**Phase 0 — A1 닫기** (코드 0, 실행 2)
- 단계 A: fused off, det ARM. 텍스트 정상 → 통과.
- 단계 B: fused on, `NNTR_L2_DIFF=1`. **`total_flips` 32/32 = 0** → 통과. fused 경로 부활(−13 ms vs 두 번 dot).
- 실패 시: 모델이 1 ULP에 민감. fused 포기, P1은 "DSP에서 SwiGLU" 대신 "u8 왕복 2회" 설계로 바뀌고 이득이 절반이 된다.

**Phase 1 — P3: ARM 21 ms 실측** (코드 ≈30줄, 실행 1) — **P1 전에 반드시.**
- `NNTR_M0_PROFILE`을 확장: 라우터 matmul / top-k / 토큰→expert gather / 뷰 생성 / HTP dispatch 진입 전후 / scatter-add / 라우팅 weight 곱. 레이어 벽시계와의 차이가 0에 가까워질 때까지 쪼갠다.
- 판정: HTP 고유 비용(gather/scatter/뷰/dispatch)이 ≥15 ms면 P1이 그걸 같이 걷어내고 **1.5×**가 나온다. CPU와 공유하는 비용(라우터/top-k)이 대부분이면 P1 후에도 **비슷**이고, 그때는 Phase 5로 바로 간다.

**Phase 2 — P1: MoE 배칭, 64 호출 → 1** (코드 큼, 실행 3~4)
- IDL 하나: `mm_u8i4_moe_layer(handles_gate_up[32], handles_down[32], row_index[1776], row_count[32], route_weight[1776], act_u8_ah[444], out_f32[444×2048])`.
- DSP 안에서: activation을 **한 번** u8 양자화 → expert마다 row gather → gate_up → SwiGLU(det) → requant → down → route_weight 곱해서 out에 누적. 검증된 `u8in` 경로 재사용.
- **weight DMA 파이프라인** = D1의 일반화: expert e의 mm 동안 e+1의 gate_up(3.67 MB)을, e의 down 동안 e+1의 down(1.84 MB)을 당긴다. VTCM 8 MB 예산: activation u8 0.9 + out f32 3.6은 DDR에 두고 행 블록만 올린다; weight는 1.5 세트 이중 버퍼. **레이아웃 계산을 코드 전에 문서로 먼저.**
- 예상 −19 ms (transport −12, quant −2.5, drain −3, staging −1.1, config 상각 −0.8). §13.1 표.
- 게이트: 기존 `NNTR_L2_DIFF`를 배칭 호출에 맞게 확장 — 레퍼런스는 지금의 64-호출 경로, 비트 비교.

**Phase 3 — P2: dequant** (코드 작음, 실행 2)
- 프로브 먼저: 두 패스인지, worker 분할 오버헤드인지, i32→f32 변환 자체인지 `HEXKL_PROBE`로 쪼갠다. **원인 모르고 튜닝하지 않는다** (DQ1의 교훈).
- 예상 −2.9 ms.

**Phase 4 — P4: 등록 캐시** (코드 중간, 실행 1)
- 성능 아님. 22 레이어 = 60 s 기동, `HEXKL_MM_U8I4_MAX_WEIGHTS=512`로 8 레이어까지만. WH bake 결과는 결정적 바이트라 한 번 구워 파일로. **전체 모델을 돌리려면 필수**이므로 Phase 2 결과가 좋으면 즉시.

**Phase 5 — "훨씬"으로 가는 길: 나머지 47%** (별도 로드맵)
- attention → HTP (00~40번 문서의 작업, 이 모델에 미통합), dense FFN → HTP, norm → HTP.
- 그러면 residual stream(444×2048 f32 = 3.6 MB)이 레이어 사이에 DSP에 머물고 **transport가 레이어당 0으로** 간다. §13.1의 2.5 ms도 사라지고, 무엇보다 ARM 21 ms의 HTP 고유 부분이 구조적으로 없어진다.
- 이것 없이는 prefill 1.3~1.6×가 한계다.

### 13.5 하지 않을 것 (근거 있는 제외, 갱신)

| 항목 | 이유 |
|---|---|
| "N번 × T"로 정당화되는 커널 내부 미세 최적화 | 3/3 실패. 메모리 접근 패턴 변경이나 대기 숨기기만 |
| ARM 쪽 양자화 | 300–420 µs/call 측정된 손해 |
| decode를 HTP로 | M=1은 DDR 바운드, 25.8 vs 25.7 |
| `neon::swiglu` 전역 변경 | 모든 모델이 씀. MoE 경로만 `swiglu_det` |
| 스로틀된 기기에서 transport로 판단 | §13.3 규칙 4 |
| FastRPC async로 호출 겹치기 | P1이 1 호출로 만들면 겹칠 대상이 없다. Phase 5에서 재검토 |

### 13.6 다음 한 걸음

Phase 0 단계 A의 출력(fused off, `swiglu 0.0`, 텍스트). 그 다음 단계 B. 그 사이에 P3 계측 코드를 쓴다 — 기기가 필요 없는 작업이라 병렬로 간다.

---

## 14. A1 닫힘, 그리고 fused가 −13이 아니라 −2인 이유 (2026-09-10, 단계 B)

### 14.1 결과

```
[L2-DIFF] ... snr=999.00 dB max_abs_err=0   × 32/32
```

fused 출력이 레퍼런스(두 번 `mm_u8i4_layer` + 호스트 `swiglu_det`)와 **완전히 같다.** flip 0이 아니라 값 자체가 같다. 텍스트는 단계 A와 동일한 473-token 요약. `kFusedSwigluEnabled = true`로 커밋. **L2 정확도 문제는 여기서 끝난다** — 근본 원인(두 근사의 u8 경계 flip)과 해법(양쪽이 같은 스펙을 비트 동일하게)이 실기기에서 닫혔다.

### 14.2 장부 (µs/call, 단계 B − 단계 A, 둘 다 식은 기기)

| | gate_up | down | 레이어당 |
|---|---:|---:|---:|
| transport | −106.6 | −64.6 | **−5.5 ms** |
| quant | +28.1 (requant) | −69.1 (u8in) | −1.3 |
| swiglu | +60.0 | 0 | +1.9 (미계측 ARM → 계측 DSP로 이동) |
| dequant | +18.9 | +3.4 | +0.7 |
| **drain** | **+33.0** | **+46.5** | **+2.5** |
| mm, acc | −0.7 | −10.6 | −0.4 |
| staging | | | −1.3 |
| 합 | | | ≈ −3 (측정: 43.1 → 41.1) |

### 14.3 근본 원인: D1과 u8in의 충돌

D1은 weight DMA를 activation quant 뒤에 숨긴다. fused down은 u8in이라 quant가 없다 → 숨길 곳이 없다. **down drain 50.3 µs = 1.84 MB / 34 GB/s = 54 µs, 전체 DMA 시간.** 구조적이다. gate_up의 65.4는 fused 커널이 첫 64행 블록의 quant 뒤에만 drain해서 M이 클수록 덜 숨기는 것으로 설명된다.

**확정 (깨끗한 실행, `NNTR_L2_DIFF` 없음, 같은 온도):** `layer calls total` **40.1 ms**, gate_up drain **56.5**, down drain **47.0**, gate_up dequant **107.9**. l2Diff 오염은 레이어당 ≈1.0 ms였고 나머지는 전부 구조적이다.

| 레이어당 drain | 두 번 dot+D1 | fused+D1 | 이론상 전체 DMA |
|---|---:|---:|---:|
| | **1.16 ms** (78% 숨김) | **3.31 ms** (36% 숨김) | 5.18 ms |

fused의 순이득은 **43.1 → 40.1 = −3.0 ms**이고, 여기에 swiglu 1.9 ms가 미계측 ARM에서 계측 DSP 컬럼으로 옮겨온 것이라 벽시계 이득은 ≈−4.9 ms다. **노출된 2.15 ms를 파이프라이닝으로 되찾으면 −5.2 ms**가 된다. dequant +11.9(96.0 → 107.9)는 남은 미설명 항목 — fused가 VTCM에 더 많이 상주시키므로 dequant가 다른 메모리를 건드리는 것으로 의심되나 프로브 전까지는 가설이다. P2에서 같이 본다.

### 14.4 −13은 이중 계산이었다

43 §1의 55.6(두 번 dot)은 D1 이전 값이다. D1이 두 번 dot에서 같은 DMA-숨기기 이득을 먼저 가져갔으므로(55.6 → 43.1), fused가 그 위에 얹는 이득은 작다. fused가 나쁜 게 아니라 기대치가 D1의 이득을 두 번 셌다.

### 14.5 해법 — P1의 설계 요구사항으로 흡수

호출 간 prefetch(IDL 힌트, VTCM 5.5 MB 동시 상주)로 drain −3.4 ms를 따로 회복할 수 있지만, P1이 만들어지면 버려지는 부분집합이다. **P1으로 간다.** 이번 발견은 P1의 하드 요구사항이다: *expert e의 mm 동안 e+1의 gate_up·down weight를 당겨야 하며, 그러지 않으면 레이어당 5.1 ms가 그대로 노출된다.* P1 후 프로파일 예상: 41.1 − 3.4(drain) − 12(transport) − 2.7(quant) − 0.9(staging) ≈ **22 ms/layer**.

**부수 발견**: 같은 `hvx_quant_rows_u8` 함수가 VTCM 입력(fused requant, M×1792)에서 28.1 µs, uncached DDR 입력(두 번 dot down quant, 같은 크기)에서 69.1 µs. 2.5배. Q1이 깨진 축과 같다. P1에서 activation을 DSP에 올려 VTCM에서 양자화하면 gate_up의 83.5도 같은 비율로 떨어진다 — §13.1의 quant 0.7 추정을 뒷받침한다.

---

## 15. P3 측정 결과 (2026-09-10) — §4.3의 "미계측 ARM 21 ms"는 틀렸다

### 15.1 단계별 (µs/layer, HTP=layer[0], CPU=layer[4..7] 평균, 같은 실행)

| 단계 | HTP | CPU | 차이 |
|---|---:|---:|---:|
| setup / router / topk / wksp | 2,135 | 2,219 | −84 |
| **gather** | **6,166** | **1,103** | **+5,063** |
| route / scatter / other | 3,383 | 2,227 | +1,156 |
| **비-ffn 합** | **11,684** | **5,549** | **+6,135** |
| **ffn** (HTP는 등록 1,543,700 제외) | **45,816** | **26,256** | **+19,560** |
| **레이어** | **57,500** | **31,805** | **+25,695** |

**계측 건전성**: `ffn` 45.8 ms vs `NNTR_HTP_PROFILE`의 layer calls 43.4 + staging 1.2 = 44.6. 차이 1.2 ms가 FastRPC dispatch. 두 독립 계측이 맞물린다.

### 15.2 뒤집힌 것 셋

1. **"미계측 ARM ≈21 ms"는 존재하지 않는다.** 비-ffn 전체가 11.7 ms이고 CPU도 5.5 ms를 낸다. HTP 고유 초과분은 **6.1 ms**. 43 §4.3의 21 ms 잔차는 등록 시간이 섞인 값이었다 — **잔차로 얻은 숫자를 3주간 계획의 축으로 썼다.**
2. **격차의 76%(19.6/25.7)는 FFN 안에 있다.** ARM 주변부가 아니라 FastRPC 왕복(17.0)과 DSP 파이프라인이다.
3. **CPU 기준선은 38이 아니라 31.8 ms**다 (식은 기기). 목표가 더 높다.

### 15.3 gather +5.1 ms — 두 가설, 아직 안 갈림

`prefill_token_input`은 평범한 `nntrainer::Tensor`이고 `htp_rpcmem`은 weight 등록 스크래치 전용이라 **"목적지가 uncached ION"은 틀렸다** (소스 확인). 남은 설명: layer[0]은 **등록 직후**라 64 weight ≈185 MB를 rpcmem에 복사한 뒤 캐시·TLB가 밀린 상태이고, gather가 그 뒤 첫 대용량 접근이다. route/scatter의 +1.16 ms도 같은 후유증으로 설명된다.

**가르는 실험, 코드 0줄**: `nntr_config.json`의 `moe_htp_layers`를 `0` → `5`. gather가 여전히 6000대면 HTP 고유, 1300대면 등록 후유증이고 **P4가 같이 고친다.** 한 관측에서 메커니즘을 단정하지 않는다 (§11.5).

### 15.4 도달 지점 (§13.1 대체)

FFN 45.8 = transport 17.0 + DSP 26.5 + staging 1.2 + dispatch 1.2

| | ms |
|---|---:|
| 지금 (레이어 벽시계) | 57.5 |
| P1: transport 17.0→2.5, quant 3.6→0.9, drain 3.7→0.5, dispatch+staging, gather DSP로 | **29.9** |
| P2: dequant 5.7→1.0 | **25.2** |
| **vs CPU 31.8** | **1.26×** |

**MoE FFN만으로는 1.26배가 천장이다.** 못 줄이는 바닥이 mm 8.87 + acc 2.76 = **11.6 ms**인데 CPU FFN 전체가 26.3 ms다. **HMX 행렬곱 자체는 CPU보다 3.0배 빠르다** — 포장(quant 3.6 / dequant 5.7 / drain 3.7 / transport 17.0) 30 ms가 연산 8.9 ms의 3.4배인 것이 전부다. (첫 기록은 gate_up의 mm만 세어 5.9/4.5배로 적었다 — 정정.) 포장을 없애려면 값이 레이어 사이에 DSP에 머물러야 하고 그것이 Phase 5(§13.4)다. 2~3배를 원하면 Phase 5 말고 길이 없다.

### 15.5 dequant는 이론 대비 15배 느리다 — P2를 앞으로

1776×3584 + 1776×2048 = **9.99M 원소를 5.7 ms** → 1.75 G elem/s. HVX 32 lane × ~800 MHz = 25.6 G/s 이론 대비 **15배**. i32→f32 스케일 곱 하나짜리 연산이 이럴 수 없다. 원인이 하나일 가능성이 높고 작업량이 작으므로 **P2를 P1 앞으로 올린다.** 프로브 먼저, 튜닝은 그 다음 (DQ1의 교훈).

### 15.6 순서 (§13.4 대체)

| | 작업 | 코드 | 실행 | 예상 |
|---|---|---|---|---|
| 1 | `moe_htp_layers=5` 대조 — gather의 정체 | 0 | 1 | 판정 |
| 2 | **P2 dequant 프로브 → 튜닝** | 작음 | 2 | **−4.7** |
| 3 | P1 배칭 (weight DMA 파이프라인 필수, §14.5) | 큼 | 3~4 | −22.6 |
| 4 | P4 등록 캐시 | 중간 | 1 | 전체 모델 전제 |
| 5 | Phase 5 — 나머지 47%를 HTP로 | 별도 | | **2~3배의 유일한 길** |


---

## 16. 최적화 레버 전체 목록 — 예상 절감치와 신뢰도 (2026-09-10, P3 이후)

모든 숫자는 식은 기기의 실측(§14–15)에서 나온 것. **신뢰도**: ●●● 메커니즘이 기기에서 증명됨 / ●●○ 측정된 상한이 있고 메커니즘은 합리적 / ●○○ 프로브 전 추정. §11.5의 규칙에 따라 "N번 × T"로 만든 숫자는 ●○○ 이하로만 둔다.

### 16.1 출발점 (ms/layer, 444 tokens)

| | HTP | CPU |
|---|---:|---:|
| ffn | 45.8 | 26.3 |
| ├ transport | 16.95 | — |
| ├ DSP: mm 8.87 · dequant 5.70 · drain 3.68 · quant 3.57 · acc 2.76 · swiglu 1.92 | 26.49 | — |
| └ staging 1.2 + dispatch 1.2 | 2.4 | — |
| 비-ffn (공유 5.5 + HTP 초과 6.1) | 11.7 | 5.5 |
| **레이어** | **57.5** | **31.8** |

### 16.2 A. MoE FFN 커널 — 이 문서의 범위

| # | 레버 | 대상 | 지금 → 후 | **절감** | 신뢰도 | 근거 · 조건 |
|---|---|---|---:|---:|:-:|---|
| A1 | ✅ 결정적 SwiGLU (fused 부활) | — | 43.1 → 40.1 | **−3.0** (벽시계 −4.9) | ●●● | §14, 32/32 `max_abs_err=0` |
| A2 | ✅ D1 weight DMA 선행 | drain | 5.13 → 1.16 | **−4.0** | ●●● | §7 43, 4회 확인. fused에선 절반 되돌아옴(§14.3) |
| **A3** | **P1 배칭 64 호출 → 1** | transport | 16.95 → 2.5 | **−14.5** | ●●● | 와이어 36 → 4.5 MB. transport는 바이트에 비례함이 세 실행에서 확인 |
| | | dispatch | 1.2 → 0.05 | −1.15 | ●●● | 호출 수 |
| | | staging | 1.2 → 0.3 | −0.9 | ●●○ | u8 한 번 |
| | | quant | 3.57 → 0.9 | −2.7 | ●●○ | 1776 → 444행(4×), VTCM에서 (§14.5: 2.5×) |
| | | drain | 3.68 → 0.5 | −3.2 | ●●○ | expert 간 파이프라인. **설계 요구사항**(§14.5), 안 넣으면 5.18 노출 |
| | | gather | 5.1 → 0 | −5.1 | ●○○ | row index로 DSP 안에서. **§15.3 실험 결과에 따라 P4 몫일 수 있음** |
| | | HMX config 상각 | 8.87 → ~8.0 | −0.9 | ●○○ | 추정 |
| | | **A3 합** | | **−28.5** | | |
| **A4** | **P2 dequant** | dequant | 5.70 → 1.0 | **−4.7** | ●○○ | 이론 대비 15배 느림(§15.5). 프로브 먼저 |
| A5 | P4 등록 캐시 | 기동 | 1544 ms/layer → ~100 | 레이어당 0 / **기동 34 s → 2 s** | ●●○ | WH bake는 결정적 바이트. **전체 모델 실행의 전제.** gather 5.1이 등록 후유증이면 이것이 고침 |
| A6 | down 출력 fp16 | transport | 3.6 → 1.8 MB | −0.5 | ●●○ | P1 후. residual add가 fp16 받아야 함 |
| A7 | M 패딩 55.5 → 64 | mm | 13% 낭비 | **0** | — | 라우팅이 정함. 불가 |
| A8 | acc (HMX accumulator read) | acc | 2.76 | **0** | — | 벤더 |

**A 합계 (A3+A4+A6): −33.7 → 57.5 → 23.8 ms. vs CPU 31.8 = 1.34×.** gather가 P4 몫이면 A3에서 −5.1이 빠지고 P4가 같은 만큼 걷어내므로 합계는 같다.

### 16.3 B. ARM 쪽 비-ffn — CPU 경로도 같이 빨라짐

| # | 레버 | 지금 | **절감** | 신뢰도 | 비고 |
|---|---|---:|---:|:-:|---|
| B1 | 워크스페이스 텐서 4개를 forward마다 새로 할당 → 한 번만 | wksp 1.3 | **−1.2** | ●●● | `forwarding()`/`incremental_forwarding()` 지역 변수. 코드 작음 |
| B2 | route: 토큰마다 `Tensor` 만들어 `multiply_i` (1776회) → 한 번의 strided 커널, 또는 P1에서 DSP로 | 0.97 | −0.9 | ●●○ | P1이 흡수 |
| B3 | scatter: 토큰마다 `Tensor` 만들어 `add_i` (1776회) → strided 커널 | 1.15 | −0.8 | ●●○ | |
| B4 | gather: 같은 패턴 | 1.1 | −0.6 | ●●○ | P1이 흡수 |
| B5 | gather HTP 초과 5.1 / route·scatter 초과 1.2 | 6.3 | −6.3 | ●○○ | **§15.3 실험 대기.** 등록 후유증이면 P4, HTP 고유면 P1 |

**B는 CPU 경로도 같은 만큼 빨라지므로 HTP/CPU 비율은 거의 안 움직인다.** 절대 시간은 준다. B1은 공짜라 먼저 한다.

### 16.4 C. 이 커널 밖 — "훨씬"이 사는 곳

| # | 레버 | **절감** | 신뢰도 | 비고 |
|---|---|---|:-:|---|
| C1 | attention → HTP | prefill의 나머지 47% 중 큰 몫 | ●○○ | 00~40번 문서의 작업, 이 모델에 미통합 |
| C2 | dense FFN(layer 0–1), norm → HTP | | ●○○ | |
| C3 | **residual stream을 레이어 사이에 DSP에 유지** | transport → 0 (모든 레이어), quant/dequant 대부분 소멸 | ●○○ | C1+C2의 결과. 포장 30 ms가 여기서 사라진다 |
| C4 | decode → HTP | **0** | ●●● | M=1 DDR 바운드, 25.8 vs 25.7 확정. 어떤 계획으로도 안 빨라짐 |

### 16.5 도달 지점 — 정직한 표

| 단계 | HTP ms/layer | vs CPU 31.8 | **전체 prefill** (MoE FFN=53%, Amdahl) |
|---|---:|---:|---:|
| 지금 | 57.5 | 0.55× | 0.7× (느림) |
| + A3 P1 | 29.0 | 1.10× | 1.05× |
| + A4 P2 | 24.3 | 1.31× | 1.13× |
| + A6, B1 (CPU도 −1.2) | 22.6 vs 30.6 | 1.35× | 1.15× |
| **A 전부 = MoE FFN 천장** | **≈22** | **≈1.4×** | **≈1.2×** |
| + C (Phase 5) | | | **2× 이상 가능** — 유일한 길 |

**decode: 0×, 어느 단계에서도.**

### 16.6 순서

1. **B1** (코드 20줄, 공짜) + **§15.3 실험** (config 한 줄) — 같은 실행
2. **A4 P2** 프로브 → 튜닝 (−4.7, 작업 작음)
3. **A3 P1** (−28.5, 큰 작업, drain 파이프라인 필수)
4. **A5 P4** (전체 모델 실행 전제)
5. **C** — 별도 로드맵. "훨씬"을 원하면 여기.


---

**→ 목표가 "CPU보다 훨씬 빠르게"라면 이 문서의 §13.4/§16.6 순서는 더 이상 계획이 아니다. `45_whole_model_on_htp_plan.md`가 대체한다.** MoE FFN P1은 그 계획의 Phase A이고, 인터페이스가 바뀐다(§3.1: f32 포인터가 아니라 DSP-side 핸들).
