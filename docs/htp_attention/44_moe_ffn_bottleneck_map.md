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
