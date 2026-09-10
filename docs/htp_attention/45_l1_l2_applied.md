<!-- SPDX-License-Identifier: Apache-2.0 -->

# 45 — L1 / L2: 실제로 적용한 최적화와 현재 상태

**목적**: `43_moe_ffn_measured_next_levers.md` §5가 랭킹한 레버 중 **L1**과 **L2**
두 개에 대해, *무엇을 왜 어떻게 고쳤고, 기기에서 어떤 숫자가 나왔고, 지금 켜져
있는지 꺼져 있는지*만 한자리에 모은 문서. 근거 수치와 측정 로그는 43 §1·§7,
설계 맥락은 `44_moe_ffn_bottleneck_map.md`에 있고 여기서 중복하지 않는다.

**대상**: LFM2.5-8B-A1B MoE FFN, Hexagon V79 HTP, Galaxy S25 Ultra `R3CY10WM83Y`,
`--fc_dtype Q4_0 --moe_dtype QS4CX --isa ARM`, 444-token prefill / 512 generated,
`moe_engine=htp`, `moe_htp_layers="2"` (22개 MoE 레이어 중 1개).

---

## 0. 한 줄 요약

| | 무엇 | 상태 | 효과 |
|---|---|---|---|
| **L1** | DSP WH bake를 HVX 워커 풀로 병렬화 | **적용, 켜져 있음** | register FastRPC 46.86 → **18.90 ms/weight**, 레이어 등록 3819.6 → **1674.7 ms** (~2.3×), 바이트 단위 동일 |
| **L2** | gate_up → SwiGLU → down을 한 DSP 호출로 융합 | **구현됨, 꺼져 있음** (`kFusedSwigluEnabled = false`) | 속도는 **25% 이득** (55.6 → 41.5–42.6 ms/layer) 인데 **정확도**에서 실패 — 원인은 규명 완료, 코드 결함 아님 |

L2가 꺼져 있는 이유는 성능이 아니다. 융합 경로는 실제로 더 빠르다. 꺼져 있는
이유는 §2.4의 u8 양자화 경계 flip이고, 그 해법(A1, 비트 동일 SwiGLU)은 아직
구현되지 않았다.

두 레버 모두 **`moe_engine=htp`일 때만** 관계된다. L2 플래그와 무관하게 QS4CX
weight는 이미 HTP로 간다 (`float_tensor.cpp`의 `dotQs4cx`가
`supports_gemm_qs4cx_accel_fp32()`를 먼저 확인한다 — 44 §3.1).

---

## 1. L1 — 등록(registration) bake의 워커 풀 병렬화

### 1.1 무엇이 문제였나

HTP가 weight를 쓰려면 먼저 RM(row-major) int4 바이트를 HMX가 읽는 WH 타일
배치로 **구워서**(bake) DSP 쪽에 상주시켜야 한다. 이 일은
`hexkl_weight_u8i4_register`(`nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_dma.c:83`)가
`k_tiles × n_tiles`를 돌며 `hexkl_micro_hmx_rm_to_wh_i4`를 호출해서 한다.

gate_up weight 하나가 `k_tiles=64 × n_tiles=112 = 7168` 타일, 타일당 ~6.6 µs.
**34–48 ms/weight**, 한 레이어(64 weights)에 **3.7 s**. 22개 레이어 전체로는
~81 s 기동 — 성능 이전에 전체 모델을 켤 수 없게 만드는 항목이었다 (43 §2).

그런데 그 루프는 **단일 스레드**였고, 같은 세션이 이미 5-워커 HVX 풀
(`qurt_hvx_get_units()=0x600` → `n_hvx=6`, `pool_workers=5`)을 만들어 레이어
호출마다 넘기고 있었다. bake만 그 풀을 안 받고 있었다.

### 1.2 무엇을 했나

타일 `t` 하나는 `w_i4_rm`의 서로 겹치지 않는 영역을 읽어 VTCM `t*512`의 서로
겹치지 않는 오프셋에 쓴다 — 즉 **완전히 독립적**이다. 그래서:

- `hexkl_weight_u8i4_register`에 `hvx_worker_pool *pool` 인자를 추가하고
  (`hexkl_mm_u8i4_dma.h:71`), `nntr_hvx_weight_register_u8i4`
  (`test/htp/nntr_hvx_mm_u8i4.c:118`)에서 세션의 기존 풀을 그대로 흘려보낸다.
- 루프 본체를 `hexkl_bake_u8i4_worker` + `hexkl_bake_u8i4_ctx`로 빼내
  `hvx_worker_pool_run(pool, ..., k_tiles * n_tiles)` 한 번으로 대체.
  분할은 `lo = n_tiles*i/n_threads`, `hi = n_tiles*(i+1)/n_threads`.
- `ctx.err`의 레이스는 의도적으로 허용한다: 벤더 bake의 입력이 타일 간
  shape·정렬이 동일해서 **전부 실패하거나 전부 성공**하고, 어느 워커의 store가
  살아남든 등록은 어차피 에러를 보고한다.

건드린 파일: `hmx/hexkl_mm_u8i4_dma.{c,h}`, `test/htp/nntr_hvx_mm_u8i4.c`
(+ 같은 시그니처를 부르는 `hmx/hexkl_attn_dtype.c:29`). **skel 재빌드 필요.**

### 1.3 기기 숫자

| | 이전 | 이후 |
|---|---:|---:|
| register FastRPC (첫 적용, 43 §7 2026-09-08) | 34–48 ms/weight | **19.1–19.4** |
| register FastRPC (복원 후 재측정, 2026-09-09) | 46.86 ms/weight | **18.90** |
| 등록 총합 (1 레이어 = 64 weights) | 3692 / 3819.6 ms | **1591–1636 / 1674.7 ms** |
| 전체 모델 기동 추정 | ~81 s | ~35 s |

정확도 게이트는 SNR이 아니라 **바이트 동일성**으로 잡았다: 시리얼 bake와 N-워커
bake가 같은 matmul 출력을 내는지 FNV-1a로 비교 — `0x198748e597cf4105`
(M=64/K=2048/N=3584) 양쪽 동일, `unittest_hvx_mm_u8i4` 16/16.

`prefill_ms − registration_ms`는 1774–1837 ms로 CPU 컨트롤 밴드(1544–1771) 안에
있었다 — L1은 **등록만** 줄이고 레이어 호출 시간은 건드리지 않으므로 예상대로다.

### 1.4 이상치(~12 ms/weight)에 못 미친 이유

5 워커면 34–48 → ~12를 기대했지만 19에서 멈췄다. Amdahl: bake **자체**는 5–6×로
스케일했고, 남은 것은 weight당 직렬 잔여분 — `malloc`, 1.8–3.7 MB WH memcpy
(VTCM scratch → DSP heap), FastRPC 고정비. 그 잔여분을 없애는 것은 L1이 아니라
**P4(bake 캐시)** 항목이다 (44 §4): WH bake 결과는 결정적 바이트라 한 번 구워
파일로 저장하면 다음 로드는 memcpy가 된다. 등록은 L1 이후에도 레이어당
convert 452 ms + register 1217 ms이고, `HEXKL_MM_U8I4_MAX_WEIGHTS = 512` 때문에
8 레이어까지만 등록 가능하다는 사실은 그대로다.

### 1.5 같이 들어간 등록 쪽 최적화 (L1은 아니지만 같은 항목을 공격)

`htp_qs4cx_from_*`의 ARM 쪽 transpose가 캐시에 적대적인 store 패턴이었다.
64×64 타일 transpose로 바꿔서 **convert 25.40 → 9.91 ms/weight** (호스트 2.2–3.1×),
출력은 비트 동일. 43 §7 2026-09-08 첫 행.

### 1.6 함정 — 읽고 가라

L1은 한 번 **잘못 revert됐다가 복원**됐다. FNV-1a 해시 비교에서 "5-워커 bake"와
"시리얼 레퍼런스"가 어긋나서 HMX 락 손상으로 읽었는데, 진짜 원인은 그
**레퍼런스 쪽**이었다:

```c
hvx_worker_pool_run(NULL, func, ctx, n_units);   // ← "시리얼"이 아니다
// NULL 분기는 func(n_units, 0, ctx) — "전부 한 스레드에서"가 아니라
// "n_units개 스레드가 있다고 치고 그중 워커 0의 1/n_units 슬라이스만"
// n_units = n_tiles이면 lo=0, hi=1 → 타일 0 하나만 굽고 나머지는 stale VTCM
```

**교훈**: `hvx_worker_pool_run`에 NULL을 넘기는 것이 안전한 것은 그 호출부의
`n_units <= 1`일 때뿐이다. 함수의 한 줄 요약이 아니라 호출부를 보고 판단할 것.
어떤 것의 "시리얼 레퍼런스"로 NULL 풀을 쓰지 말 것.

---

## 2. L2 — gate_up → SwiGLU → down 융합

### 2.1 노린 것

expert 하나당 HTP를 두 번 타면서 왕복 한 번 + quant 한 번 + dequant 한 번이
그냥 버려진다. 43 §5의 산술 (prefill, M≈55):

| | 지금 (two-dot) | 융합 |
|---|---|---|
| 호출 | 2 | 1 |
| 와이어 | 450 + 787 + 394 + 450 = **2081 KB** | 450 in + 450 out = **900 KB** |
| transport | 556.8 + 423.3 = 980 µs | ~450 µs |
| quant / dequant | ×2 / 94.1+54.7 | ×1 / ×1 |
| **host** | **1738 µs** | **~1075 µs → 1.6×** |

VTCM도 통과한다: gate_up WH 3.67 + down WH 1.84 + act 0.128 + u8 중간값 0.128 +
result 0.008 ≈ **5.8 MB / 8.3 MB**, 싱글 버퍼로 들어간다.

### 2.2 무엇을 만들었나 — 구현 두 개

| | 시도 1 (one-call) | 시도 2 (split-call) |
|---|---|---|
| DSP 진입점 | `hexkl_mm_u8i4_fused_run` — 6-region VTCM 레이아웃, weight 두 개를 한 호출에서 | `hexkl_mm_u8i4_gate_up_swiglu_run` (weight 하나: gate_up matmul → SwiGLU → u8 AH로 requantize) → 이미 검증된 `mm_u8i4_layer_u8in`이 down 처리 |
| 왜 | 43 §5 그대로 | 시도 1보다 VTCM 레이아웃이 작고 들여다보기 쉽게 |
| 테스트 | `FusedSwigluMatchesTwoCallReference` 77–139 dB (M∈{55,64,100,128}) | `GateUpSwigluMatchesHostIntermediate` 37.5–37.7 dB (단일 홉 바닥 23.5 dB 대비), `GateUpSwigluPlusU8InMatchesTwoCallReference` 138.9–140.8 dB, M∈{11,55,64,100,128,138,200} 안정 |

두 구현 **모두 트리에 남아 있고 dormant**. ARM 쪽 진입점은 둘 다 동일한
`gemm_qs4cx_fused_swiglu_fp32` (`compute_ops.{h,cpp}`, `htp_compute_ops.cpp`),
스위치는 `Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.cpp:432`의
`constexpr bool kFusedSwigluEnabled = false`.

관련 파일: `hmx/hexkl_mm_u8i4_dma.{c,h}`, `hvx/hvx_swiglu_f32.c`,
`test/htp/nntr_hvx.idl`, `test/htp/nntr_hvx_mm_u8i4.c`,
`test/unittest/unittest_hvx_mm_u8i4.cpp`.

### 2.3 성능은 나왔다

split-call 경로: **41.5–42.6 ms/layer** vs two-dot HTP의 55.6 → **25% 빠름**
(6번 실행, ±1%). 같은 실행에서 새로 넣은 `mm<=` 잔차 컬럼이 HMX 행렬곱을
host의 **21.0%**로 못박았다 (gate_up ≤185.3 µs, down ≤90.7 µs). 나머지는
transport 37%, 포맷 변환 33%, weight DMA 13%.

**다만 그래도 CPU보다 느리다**: 융합 42.9 프로파일 + 미계측 ARM ≈21 ≈ **64 ms**
vs CPU M0 계측 **38 ms**. 이건 L2의 실패가 아니라 구조적 결론이고, 44가 그
지도다.

### 2.4 왜 꺼져 있나 — 정확도

**증상**: 유닛 테스트는 전부 통과하는데 실제 모델이 깨진다. 같은 프로세스,
back-to-back: `moe_engine=cpu` → 정상 3문장 요약 512 토큰,
`moe_engine=htp` + L2 on → **"Could you please provide the text you would like
summarized?" 206 토큰** — 모델이 자기 프롬프트를 잃는다. 구현 두 개가 **동일하게**
실패했다.

원인 추적은 네 단계로 진행했고 전부 43 §7에 원문이 있다:

1. **ARM 쪽 배선 무죄.** `NNTR_L2_DEBUG`로 실제 실행 중 디스패치 인자를 찍었다 —
   K=2048, N=[3584, 2048], expert별 M 11–138 (유닛 테스트가 이미 스윕한 범위),
   포인터 전부 구분되고 일관됨.
2. **NaN 가설 — 실재하는 버그였지만 이 실패의 원인은 아니었다.**
   `hvx_swiglu_f32.c`의 `hvx_recip_qf32`가 Newton-Raphson 시드를
   `0x7EF311C2u - bits(a)`로 만든다. 언사인드 뺄셈이라 `bits(a) > 0x7EF311C2`면
   래핑해서 NaN이 된다. 호출부는 `a = 1 + exp(min(-gate, 88.0f))`이고, 실제 안전
   경계는 `-gate <= 87.977863`(bisection) / `87.977861`(해석적) — 즉
   **`exp_top = 88.0f` 클램프 자체가 NaN 구역**이었다. 클램프를 **85.0f**로 내렸다
   (`hvx_swiglu_f32.c:105`): 상대오차 `1.55e-8`(스펙의 1/65), NaN 경계까지 ULP
   3650만 개 여유, 버리는 항은 `85·exp(−85) = 1.03e-35`로 u8 한 스텝의 3.8e32분의
   1 — 호스트 레퍼런스도 그 구간은 정확히 0이다. 기존 SwiGLU 테스트 셋 전부가
   `fill_deterministic × fill_deterministic`으로 gate를 만들어 이 범위에 **한 번도
   들어간 적이 없어서** 놓쳤고, 그래서 bias 오버라이드로 gate를 −200/−300/−400까지
   미는 `SwigluSurvivesExtremeNegativeGate`를 새로 넣었다.
   → 기기에서 `NNTR_L2_CHECK`로 확인: **non-finite 0개인데 텍스트는 여전히 틀림.**
   NaN 가설 사망. 수정과 테스트는 그 자체로 옳으므로 그대로 유지.
3. **진짜 원인 — u8 양자화 경계 flip.** `NNTR_L2_DIFF`(`HtpComputeOps::l2Diff`)로
   *이 모델의 실제 등록된 weight 바이트*와 *실제 캡처된 activation*을 두 경로에
   나란히 통과시켰다. 한 forward의 32 호출 중 5개가 SNR 67.6–79.8 dB(나머지 27개는
   142 dB = float32 노이즈). 세 라운드로 후보를 하나씩 죽였다:
   - outlier-row 양자화? → 깨끗한 호출의 `call_max_span`(0.17–2.91)이 문제 호출의
     범위(0.30–0.54)를 완전히 덮음. **기각.**
   - 양자화 함수가 두 경로에서 다른가? → 소스 확인, `hvx_quant_rows_u8_params` /
     `hvx_quant_pack_u8_ah` **동일 함수**. 기각.
   - 행 전체 scale 불일치? → 호스트에서 양자화 공식을 재구현해 독립적으로 뽑은
     scale이 기기 것과 `scale_diff = 0.0000%`. **기각.**
   - 남은 것: **`total_flips = 1/1792`, 문제 호출 다섯 개 전부.** 원소 딱 하나가
     u8 레벨 하나만큼 옆으로 넘어간다. SNR이 67.6–79.8로 16배나 벌어지는 건
     flip 개수가 아니라 **1792개 중 어느 K 인덱스가 넘었는가**로 설명된다 — down
     matmul이 K=1792 전체를 합하므로 그 인덱스의 down weight 크기가 곧 영향력이다.
4. **값이냐 부수효과냐.** `NNTR_L2_SHADOW=1`은 융합 커널을 **전부 실행하되**
   (모든 버퍼 접근, 모든 락) 모델에는 레퍼런스 값을 넘긴다. → 정상 512-토큰 요약.
   **메모리 aliasing이나 workspace 손상이 아니라 커널이 낸 값 자체가 원인.**

**메커니즘, 끝까지**: 기기 SwiGLU가 호스트 `expf`와 스펙 범위(~1e-6) 안에서
다르다 → 약 16%의 호출에서 원소 하나가 자기 행의 u8 bin 경계를 반대편으로 넘는다
→ 그 K 인덱스의 down weight 크기만큼 출력 행 전체가 틀어진다 → 22개 레이어를
greedy decoding으로 지나며 다른 토큰 스트림으로 증폭.

**이건 코드 결함이 아니다.** 독립적으로 계산된 두 float 파이프라인이 같은 행별
affine u8 양자화기를 먹이는 한, 어떤 원소는 언젠가 한 레벨 어긋난다. 그리고
HMX의 activation 포트는 하드웨어적으로 8-bit 고정(IDL: "HMX's activation port is
always 8-bit")이라 중간값을 int16으로 넓히는 도피처가 이 하드웨어에는 없다.

### 2.5 L2를 살리려면 — A1

**양쪽이 비트 단위로 같은 알고리즘을 돌게 한다** (44 §3.3). NEON(`neon_mathfun.h`
`exp_ps`)과 HVX(`hvx_exp_f32.h` `hvx_exp_sf`)는 이미 range-reduction 상수
(`ln2_hi = 0.693359375`, `ln2_lo = -2.12194440e-4`)를 공유한다. 갈리는 지점은 셋뿐:

| | NEON | HVX |
|---|---|---|
| 다항식 차수 | 5 | 7 |
| 산술 포맷 | IEEE f32 + FMA | qf32 |
| 나눗셈 | `vdivq_f32` | NR 역수 |

셋을 맞춰 순수 IEEE f32 곱셈/덧셈만으로(FMA 없이, qf32 없이, 나눗셈 없이) 같은
다항식과 같은 NR 역수를 같은 순서로 돌리면 flip은 **정확히 0**이 된다.
원소별 연산이라 비용은 expert당 56 KB 수준. ARM 경로도 ≤1 ULP 바뀌지만, 모델이
`std::exp`(shadow 실행)와 `exp_ps` 양쪽에서 정상 동작했으므로 허용 범위 안이다.

**검증은 SNR이 아니라 `memcmp`** — 기기 gtest에서 두 구현의 출력 바이트가
같아야 한다. 그 다음에야 `kFusedSwigluEnabled = true`로 돌리고 43 §6.3의
accept/reject(생성 텍스트 일치 + bracketed CPU control)를 다시 돌린다.

A1은 성능 레버가 아니라 **P1(MoE 배칭)의 선행조건**이기도 하다: 배칭은 SwiGLU가
DSP에 있어야만 가능하고, DSP SwiGLU는 A1 없이는 문장을 깨뜨린다.

---

## 3. 남겨둔 계측 (전부 env-gated, 기본 off)

L1/L2 작업 중 만든 것들. 다시 손대는 사람이 처음부터 만들 필요 없다.

| env | 위치 | 하는 일 |
|---|---|---|
| `NNTR_HTP_PROFILE=2` | `htp_compute_ops.cpp` `HtpProfile` | 단계별 DSP µs + `mm<=` 잔차 + `arm staging memcpy` |
| `NNTR_L2_CHECK=1` | `l2CheckFinite` | gate_up 행별 requant scale과 down 출력의 non-finite 카운트 |
| `NNTR_L2_DIFF=1` | `HtpComputeOps::l2Diff` | 실제 weight/activation으로 split-call vs 레퍼런스: expert별 SNR, worst row, `call_max_span`, `bin_flips`/`total_flips`/`block_flips`, `scale_diff` |
| `NNTR_L2_SHADOW=1` | 같은 함수 | 융합 커널을 전부 실행하되 모델에는 레퍼런스 값을 전달 — 값 vs 부수효과 판별 |
| `NNTR_M0_PROFILE=1` | `Lfm2MoELayer` | 레이어 벽시계 |

---

## 4. 다시 손대는 사람을 위한 체크리스트

**L1을 건드린다면**
- `hvx_worker_pool_run(NULL, ...)`을 "시리얼"로 쓰지 말 것 (§1.6).
- 정확도 게이트는 SNR이 아니라 FNV-1a 바이트 동일성
  (`0x198748e597cf4105`, M=64/K=2048/N=3584).
- 등록을 더 줄이려면 L1이 아니라 P4(bake 캐시)다 — 남은 19 ms/weight는 대부분
  malloc + WH memcpy + RPC 고정비다.
- skel 재빌드 필요 (43 §6.1의 4단계 전부, 순서대로).

**L2를 켜려 한다면**
- **네 번째 구현을 쓰지 말 것.** 커널 산술은 세 번 검증됐고 문제가 아니다.
- 순서는 고정: A1(비트 동일 SwiGLU, `memcmp == 0`) → `kFusedSwigluEnabled = true`
  → 실제 모델 텍스트 일치 → bracketed CPU control (cpu → htp → cpu).
- `exp_top = 85.0f`와 `SwigluSurvivesExtremeNegativeGate`는 이 실패와 무관하게
  옳다. 되돌리지 말 것.
- 유닛 테스트만으로는 이 실패를 못 잡는다. `fill_deterministic` 합성 데이터가
  모든 통과 게이트와 모든 실패 실행에 공통인 유일한 변수였다. 실제 weight +
  실제 activation으로 `NNTR_L2_DIFF`를 돌리는 것이 진짜 게이트다.
- 이 기기의 run-to-run drift는 대부분의 효과보다 크다 (43 §4). 단발 비교 금지.

**둘 다에 해당**
- 결과는 pass든 fail이든 43 §7에 추가한다. 실패 기록이 더 값지다 — §7 여러 행이
  실패다.
