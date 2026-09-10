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

두 줄 중 어느 쪽인지는 **P3 하나로 결정**된다. 그래서 순서는 **A1 (정확도 선행조건, 작음) → P3 (측정, 거의 공짜) → P1 (큰 작업, P3 결과로 가치 확정)**. P1을 P3 전에 시작하는 건 결과가 판정을 못 바꾸는 큰 작업을 먼저 하는 것이라 권하지 않는다.

---

## 6. 이 문서가 쓰인 계측 (모두 env-gated, 기본 off)

| env | 위치 | 하는 일 |
|---|---|---|
| `NNTR_HTP_PROFILE=2` | `htp_compute_ops.cpp` `HtpProfile` | 단계별 DSP µs + `mm<=` 잔차 + `arm staging memcpy` 줄 |
| `NNTR_L2_CHECK=1` | `l2CheckFinite` | gate_up의 행별 requant scale과 down 출력에서 non-finite 카운트 |
| `NNTR_L2_DIFF=1` | `HtpComputeOps::l2Diff` | split-call vs 레퍼런스(두 번 `mm_u8i4_layer` + 호스트 SwiGLU)를 실제 weight/activation으로 expert마다 SNR, worst row, `call_max_span`, `bin_flips`/`total_flips`/`block_flips`, `scale_diff` |
| `NNTR_L2_SHADOW=1` | 같은 함수 | fused 커널을 전부 실행하되 모델에는 레퍼런스 값을 넘김 — 값 vs 부수효과 판별 |
| `NNTR_M0_PROFILE=1` | `Lfm2MoELayer` | 레이어 벽시계 (P3의 출발점) |
