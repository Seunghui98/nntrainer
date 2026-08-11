<!-- SPDX-License-Identifier: Apache-2.0 -->

# 40 -- RoPE on HVX, uint8 in / uint8 out -- the task, self-contained

`00_START_HERE.md`와 `01_working_style.md`를 먼저 읽으세요. 이 파일이 태스크
본문입니다. **열린 결정은 없습니다** -- §2의 표에 이미 정해진 것들이 있고,
그중 어느 것도 다시 열지 마세요.

§10에 구현 에이전트에게 그대로 붙여넣을 프롬프트가 있습니다.

---

## 0. 범위

**만드는 것:** u8 액티베이션을 받아 u8 액티베이션을 내놓는 RoPE 커널 하나와,
그것을 디바이스에서 게이트하는 것까지.

**범위 밖 -- 손대지 마세요:**

| 항목 | 이유 |
| :-- | :-- |
| `attn_forward` / `attn_kv_append` 융합 | 다음 태스크. 이 커널이 device에서 green이 된 뒤 |
| AH 타일 레이아웃 출력 | 융합 태스크의 것. 이번 출력은 **평면 row-major u8** |
| K 경로 | K는 u8이 아님 -- §1.5 |
| calibration (`s_out` 산출) | 다음 태스크. 이번엔 `s_out`을 **파라미터로 받기만** 함 |
| DSP에서 sin/cos 계산 | cos/sin은 **입력으로 받음** |
| yarn / mrope / vision / NORMAL-mode rope | nntrainer에 없음 |

한 태스크 = 한 세션입니다 (`12_prompt_kit.md` §1).

---

## 1. 계약 -- 이게 정답의 정의입니다. 재유도 금지

### 1.1 기호

| | |
| :-- | :-- |
| `qa`, `qb` | 입력 u8 코드. 쌍은 `(w+k, w+k+half)`, `half = dim/2` |
| `s_in[m]`, `zp_in[m]` | row `m`의 입력 양자화. `x_real = (q - zp_in) * s_in` |
| `s_out`, `zp_out` | 출력 양자화 (전체 공통 스칼라). `y = round(y_real/s_out) + zp_out` |
| `cosQ[m][k]`, `sinQ[m][k]` | **Q15 int16**. 실수값 = `cosQ / 32768` |

`half`개짜리 테이블입니다. CPU 캐시(`mha_core.cpp:871`)는 뒷절반에 같은 값을
복제해 두지만 **아무도 안 읽습니다** -- NEON 구현도 `k < half_`만 씁니다.
새 커널도 앞절반만 받습니다.

### 1.2 산술

nntrainer의 회전은 NeoX half-rotation입니다
(`nntrainer/tensor/cpu_backend/arm/neon_impl.cpp:2470`,
`compute_rotary_emb_value`). row 하나(`width` 원소)를 `dim` 스텝으로 훑고,
각 세그먼트 `w` 안에서 `k in [0, half)`:

```
t0 = (qa - zp_in[m]) * cosQ[m][k] - (qb - zp_in[m]) * sinQ[m][k]
t1 = (qa - zp_in[m]) * sinQ[m][k] + (qb - zp_in[m]) * cosQ[m][k]

r  = t * (s_in[m] / (s_out * 32768))
y  = clamp_u8( rne(r) + zp_out )
```

`rne` = round-to-nearest-**even**. 임의 선택이 아닙니다:
`hvx_convert.h`의 `hvx_sf_to_w_rne`가 RNE이고, 그 파일의 주석이
**"Host references that compare against this must use nearbyint, not round"**
라고 못박아 뒀습니다. 호스트 레퍼런스는 `std::nearbyint`(기본 반올림 모드)를
쓰세요. `std::round`를 쓰면 절반 지점에서 어긋납니다.

`s_in[m] / (s_out * 32768)`은 **row당 한 번** 계산해 루프 밖에 둡니다.

### 1.3 왜 `zp`를 먼저 빼나 (접지 않는다)

전개하면 `zp_in*(cosQ - sinQ)`를 bias로 접을 수 있지만, **우리 양자화는
per-row**(`hvx_quant_rows_u8_params`가 `zp[m]`을 냄)라 컴파일 타임 상수가
되지 않습니다. 벡터 명령 수도 같습니다 (빼기 2 vs 곱 2 + 더하기 2).
**빼세요.** 접기는 per-tensor 정적 양자화로 갈 때 하는 것이고, 이 태스크가
아닙니다.

### 1.4 partial rotary는 커널이 몰라도 됩니다

`rope_partial_rotary_factor != 1`은 `_compute_proportional_parameters`
(`mha_core.cpp:959`)가 **`thetas[i] = 0`** 으로 처리합니다. 그러면
`cos = attention_scaling`, `sin = 0` -- 즉 그 lane은 항등 회전입니다.
**커널에 "회전 안 하는 채널" 분기를 만들지 마세요.** 테이블이 이미 처리합니다.

한 가지 결과: Q15에서 `cos = 1`은 `32768`인데 int16에 안 들어가므로
`32767`(= 0.99997)이 최대입니다. 항등 lane이 정확한 항등이 아니라 3e-5만큼
줄어듭니다. 재양자화 LSB에 비해 무시할 수 있지만, **테스트에서 "항등이면
출력 == 입력"을 기대하지 마세요** -- 호스트 레퍼런스와 비교하세요.

### 1.5 K는 이 커널을 쓰지 않습니다

| | 입력 | 출력이 되어야 하는 것 |
| :-- | :-- | :-- |
| **Q** | u8 액티베이션 (HMX activation port) | **u8** -- 이 태스크 |
| **K** | fp16 KV row | i4/i8 **대칭, per-kv-position** (`hexkl_kv_quant.h`) |

K는 HMX의 **weight slot**으로 갑니다. u8이 아닙니다. 이 커널을 K에 갖다
쓰면 안 되고, 이 태스크에서 K 경로를 건드리지도 마세요.

---

## 2. 이미 결정된 것 -- 다시 열지 말 것

| 항목 | 결정 | 근거 |
| :-- | :-- | :-- |
| 내부 산술 | **f32** | `\|q-zp\| <= 255`, `\|cosQ\| <= 32768` -> 곱이 최대 8.36e6 < 2^24라 f32에서 **각 곱이 exact**. 트리의 `hvx_convert.h` / `hvx_quant_u8.c` 헬퍼를 그대로 재사용 |
| bit-parity (QNN 레퍼런스와 비트 동일) | **요구사항 아님** | 이 출력을 소비하는 것은 QNN 그래프가 아니라 우리 HMX activation port. 계약(u8 + scale/zp)만 맞으면 됨 |
| cos/sin dtype | **Q15 int16** | half-step이 곧 각도 오차: u8 3.9e-3 rad vs Q15 3.0e-5 rad. `\|ds\| <= eps*\|\|q\|\|*\|\|k\|\|`이므로 eps가 그대로 score 상대 오차 상한. 테이블은 position당 `half`개뿐이라 Q15가 공짜 |
| `s_out` / `zp_out` | **호출자가 준다** | 모드 플래그 만들지 말 것. `sqrt(2)*s_in`을 넘길지 calibration 값을 넘길지는 호출자 문제 |
| `n_saturated` 출력 | **필수** | `s_out`이 작으면 예외도 로그도 없이 상위 몇 %가 잘리고 softmax가 증폭. 이게 유일한 계기판 |
| 출력 레이아웃 | **평면 row-major u8** | AH 타일은 융합 태스크의 것 |

### 2.1 이 계약이 얼마를 내는지 -- 알고 시작하세요

u8->u8은 재양자화 반올림을 **한 번** 추가합니다. 피할 수 없습니다 (입력이
u8이고 출력이 u8이면 op 안에 반올림이 한 번 더 생기는 건 계약의 정의).

```
RMS(요소 오차) = 0.289 LSB = 0.289 * 2/255 * amax = amax의 0.23%
신호 RMS 대비 (amax ~ 4 sigma 가정) = 0.9%

score: s = sum_i q_i k_i  (head_dim = 128)
  RMS(ds)/RMS(s) = sqrt(2) * sigma_d / sigma     <- head_dim으로 안 줄어듦
                                                    분자/분모가 같은 sqrt(128)
```

| | q 양자화 라운드 | 상대 score 오차 |
| :-- | :--: | --: |
| 오늘 (f32 q -> u8 한 번) | 1 | ~1.3% |
| u8->u8 rope | 2 | **~1.8%** |

**1.4배.** 새 자릿수가 아니라 이미 내는 것의 1.4배입니다. §6이 이 표를
device 실측으로 채우라고 요구합니다.

---

## 3. 만들 것

### 3.1 호스트 스칼라 레퍼런스 -- 여기서 검증 가능한 유일한 부분

`test/unittest/mha_htp_host_model.{h,cpp}`에 추가 (새 파일 만들지 마세요):

```cpp
/**
 * @brief §1.2 exactly, in int64 + double. This defines the answer.
 *
 * Rounding is std::nearbyint (round-to-nearest-even) because the kernel's
 * hvx_sf_to_w_rne is RNE -- hvx_convert.h says so explicitly. std::round
 * disagrees at the half-way point.
 *
 * @param n_saturated  counts elements whose pre-clamp code left [0, 255]
 */
void rope_u8_ref(const uint8_t *x, uint8_t *y, uint32_t n_rows,
                 uint32_t width, uint32_t dim,
                 const float *s_in, const int32_t *zp_in,
                 const int16_t *cos_q15, const int16_t *sin_q15,
                 float s_out, int32_t zp_out, uint32_t *n_saturated);
```

`test/unittest/unittest_mha_htp_host_model.cpp`에 이 레퍼런스의 자체 검사를
넣으세요:
- `cosQ = 32767, sinQ = 0`, `s_out = s_in`, `zp_out = zp_in` -> 출력이 입력과
  최대 1 LSB 차이 (§1.4의 0.99997 때문에 정확히 같지는 않음)
- `s_out`을 일부러 작게 -> `n_saturated > 0`
- **회전 항등식**: 같은 입력에 각도 `theta`를 두 번 적용한 것과 `2*theta`를
  한 번 적용한 것이 양자화 오차 범위 안에서 일치

### 3.2 HVX 커널

`nntrainer/tensor/htp_backend/hvx/hvx_rope_u8.{c,h}` (신규 2파일).
`.c`입니다 -- skel은 `hexagon-clang`으로 **C**로 빌드되고 libnntrainer도 C++
런타임도 링크되지 않습니다 (`00_START_HERE.md` §5의 defect 1이 정확히 그
실수였습니다). `nntrainer::` 심볼을 하나라도 부르면 DSP에 못 들어갑니다.

```c
/**
 * @brief RoPE with the uint8 activation contract. y may alias x.
 *
 * Row-range shape mirrors hvx_softmax_rows_f32: rows are independent, so a
 * worker pool can hand each thread its own range later without this
 * changing.
 *
 * @param x          n_rows * width uint8, row-major. No alignment requirement
 * @param y          same shape; may be the same pointer as x
 * @param s_in,zp_in one entry per row (index m, not m - m_first)
 * @param cos_q15    [m_last][dim/2] int16, Q15. Row m at cos_q15 + m*(dim/2)
 * @param n_saturated  incremented, not assigned -- caller zeroes it
 */
void hvx_rope_u8_rows(const uint8_t *x, uint8_t *y,
                      uint32_t m_first, uint32_t m_last,
                      uint32_t width, uint32_t dim,
                      const float *s_in, const int32_t *zp_in,
                      const int16_t *cos_q15, const int16_t *sin_q15,
                      float s_out, int32_t zp_out, uint32_t *n_saturated);
```

**broadcast(per-tensor)는 커널이 모릅니다.** IDL 계층에서 길이 1 시퀀스를
`n_rows`로 펼쳐 넘기세요. 커널은 항상 per-row 배열을 봅니다.

파이프라인:

```
u8 로드 -> Q6_Wuh_vunpack_Vub -> (uh -> w) -> Q6_Vsf_equals_Vw  ... f32(정수값)
int16 cos/sin 로드 -> (h -> w) -> Q6_Vsf_equals_Vw              ... f32(정수값)
a = Q6_Vsf_vsub_VsfVsf(af, vzp)          /* (qa - zp_in), exact */
ac = Q6_Vqf32_vmpy_VsfVsf(a, vc);  as = ...;  bc = ...;  bs = ...
t0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(ac, bs))
t1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(as, bc))
vq = Q6_Vw_vadd_VwVw(hvx_sf_to_w_rne(Q6_Vsf_vmpy_VsfVsf(t, vinv)), vzp_out)
    /* vinv[m] = s_in[m] / (s_out * 32768.0f), row당 한 번 */
saturation 카운트: 포화 전 vq 를 [0,255]와 비교
vh01 = Q6_Vh_vpack_VwVw_sat(vq1, vq0);  vh23 = ...
vb   = Q6_Vub_vpack_VhVh_sat(vh23, vh01)
```

마지막 세 줄은 `hvx_quant_u8.c:211-213`에 이미 있는 체인 그대로입니다. 같은
파일 `:237`이 **`Q6_Vub_vpack_VhVh_sat`의 [0,255] 포화가 명시적 clamp와
byte-identical**임을 이미 확인해 뒀으니 clamp를 따로 넣지 마세요.

### 3.3 IDL + skel

`test/htp/nntr_hvx.idl`에 추가:

```
// RoPE with the QNN activation contract: QUInt8 in, QUInt8 out.
// NOT a production path on its own -- a per-layer round trip costs ~404 us
// of FastRPC against a few microseconds of arithmetic. This exists to gate
// the kernel and to measure prefill. The production path is the fusion into
// attn_forward, which is a separate task.
//
// s_in/zp_in have length n_rows, or length 1 meaning per-tensor.
// cos/sin are Q15 int16 (scale 1/32768), [n_rows][dim/2] -- the HALF table.
// n_saturated is not diagnostics: nonzero means s_out is wrong and the
// result is silently clipped.
AEEResult rope_u8(in uint32 n_rows, in uint32 width, in uint32 dim,
                  in sequence<uint8> x,
                  in sequence<float> s_in, in sequence<int32> zp_in,
                  in sequence<int16> cos_q15, in sequence<int16> sin_q15,
                  in float s_out, in int32 zp_out,
                  rout sequence<uint8> y, rout uint32 n_saturated);
```

`test/htp/nntr_hvx_rope.c` (신규): `nntr_hvx_softmax.c`를 본뜨세요 --
`nntr_hvx_session *`를 확인하고, 길이를 전부 검증하고 (`FARF(ERROR, ...)` +
`AEE_EBADPARM`), broadcast를 펼치고, 커널을 부르고 끝.

`x`는 `in` 버퍼라 FastRPC가 read-only로 매핑할 수 있습니다.
in-place를 테스트하려면 `y`로 먼저 복사한 뒤 `y` 위에서 도세요 --
`nntr_hvx_softmax.c`의 `softmax_blocked`가 같은 이유로 그렇게 합니다.

### 3.4 device gtest

`test/unittest/unittest_hvx_rope.cpp` (신규).
`unittest_hvx_softmax.cpp`의 fixture를 복사하세요 -- unsigned PD를
`remote_session_control`로 켜는 부분과 `kDspOffset` 처리가 거기 있습니다.

### 3.5 빌드 배선

- `test/htp/build.sh`: `$SRCS`에 `nntr_hvx_rope.c`와
  `$BACKEND/hvx/hvx_rope_u8.c` 추가
- `test/jni/Android.mk`: `unittest_hvx_rope` 모듈. `unittest_hvx_add` 블록
  (`:906-927`)을 복사해서 소스 파일명만 바꾸면 됩니다
- `test/htp/run_u8i4_layer_on_device.sh`: ndk-build 타깃 목록과 push/run에
  `unittest_hvx_rope` 한 줄씩 추가 (새 스크립트 만들지 마세요)
- `test/unittest/meson.build`: 호스트 테스트는 기존
  `unittest_mha_htp_host_model` 타깃에 들어가므로 **변경 없음**

---

## 4. 트리에 이미 있는 것 -- 쓰세요, 다시 쓰지 마세요

| 필요한 것 | 있는 곳 |
| :-- | :-- |
| f32 -> u8 양자화 + 포화 pack 체인 | `hvx_quant_u8.c:192-213` |
| f32 <-> int32 변환, f32 splat | `hvx_convert.h` (`hvx_sf_to_w_rne`, `hvx_splat_sf`), `hvx_dequant_i32.c:51`의 `Q6_Vsf_equals_Vw` |
| tail 스테이징 / in-vector reduction | `hvx_softmax_util.h` (`load_tail_sf`, `store_tail_sf`) |
| 회전 산술의 벡터 형태 | llama.cpp `ggml/src/ggml-hexagon/htp/rope-ops.c`의 `hvx_rope_neox_f32_aa` |
| u8 -> u16 위닝 | `Q6_Wuh_vunpack_Vub` -- llama.cpp `hmx-mm-kernels-tiled.h:341` |
| CPU 정답 | `nntrainer::compute_rotary_emb_value` (`neon_impl.cpp:2470`) |
| cos/sin이 만들어지는 곳 | `mha_core.cpp:871` `precompute_freqs` -> `calc_trigonometric_vals_dup` |
| skel entry point 형태 | `test/htp/nntr_hvx_softmax.c` |
| device fixture | `test/unittest/unittest_hvx_softmax.cpp` |

---

## 5. 알려진 함정 5개

1. **`Q6_Wuh_vunpack_Vub`의 even/odd lane 순서를 확인하지 않고 믿지 마세요.**
   이 문서를 쓴 환경에서 Hexagon 컴파일을 못 했습니다. 회전은 lane index가
   곧 `k`라, 순서가 틀리면 **에러 없이 잘못된 각도**를 씁니다. §6의 lane
   테스트가 이것만 잡으라고 있는 것입니다.
2. **정렬 없음.** FastRPC 버퍼는 벡터 정렬 보장이 없습니다. `HVX_UVector`를
   쓰세요 (`hvx_softmax_f32.c:34`가 같은 이유로 그렇게 합니다).
3. **tail.** `dim = 96`이면 `half = 48`, `48 % 32 = 16`이 남습니다.
   `load_tail_sf`/`store_tail_sf`를 재사용하세요.
4. **aliasing.** `y == x`가 정상 경로입니다. `a`, `b`를 둘 다 읽은 뒤에
   쓰면 안전한데, **4-row 묶음 pack을 하면 쓰기 시점이 뒤로 밀립니다.**
   그 경우 다시 확인하세요.
5. **`s_in`/`zp_in` 인덱스는 `m`이지 `m - m_first`가 아닙니다.** row-range
   커널에서 가장 흔한 off-by-range 버그입니다.

---

## 6. Acceptance -- 고정입니다. 바꾸지 마세요

### 6.1 호스트 (DSP 없이)

`meson test unittest_mha_htp_host_model --print-errorlogs` 통과, §3.1의 세
자체 검사 포함.

### 6.2 디바이스 -- `unittest_hvx_rope`

| 항목 | 기준 |
| :-- | :-- |
| **정확도** | `rope_u8_ref`와 **최대 +-1 LSB**, 그리고 **불일치 원소 비율을 출력**. (f32 내부 산술과 double 레퍼런스의 반올림 경로가 달라 완전 일치는 기대하지 않음) |
| **lane 순서** | `cos_q15[k] = k * 256`, `sin_q15[k] = 0`으로 두면 lane마다 배율이 달라져 어떤 순열도 결과를 바꿉니다. 레퍼런스와 일치해야 함. **이 케이스는 필수입니다** |
| **saturation** | 넉넉한 `s_out`에서 `n_saturated == 0`. 일부러 작은 `s_out`(예: `s_in/8`)에서 `n_saturated > 0`이고 그 값이 레퍼런스와 **정확히 일치** |
| **범위** | 회전 후/전 `amax` 비율을 출력. `sqrt(2)` 상한 안이어야 함 (회전은 쌍마다 직교변환이라 `a^2+b^2` 보존, 등호는 `a=b, 45도`) |
| **오차 예산** | §2.1 표를 실측으로 채워 출력: 요소 RMS, 상대 score 오차 (u8 입력만 vs u8->u8) |
| **in-place** | `y == x` 결과가 별도 버퍼 결과와 **bitwise 동일** |
| **broadcast** | `s_in` 길이 1과 길이 `n_rows` 양쪽, 같은 값이면 결과 동일 |
| **shape 매트릭스** | `dim` {64, 96, 128} x `width/dim` {1, 8, 16} x `n_rows` {1, 7, 32, 1024} |
| **zp_in** | {0, 128, 255} 및 row마다 다른 값 |
| 성능 | `ROPE_FIELD path=... field=us value=...`로 출력. **threshold assert 금지** -- thermal 때문에 flaky합니다 (`13_htp_pr_plan.md` T6와 같은 이유) |

`dim = 96`은 tail을, `n_rows = 1`은 decode를, `1024`는 prefill을 봅니다.
매트릭스를 줄이지 마세요.

### 6.3 오차 진단표 -- bound를 올리는 대신 이걸로 진단하세요

| 증상 | 원인 |
| :-- | :-- |
| lane 테스트만 실패 | §5-1. `Q6_Wuh_vunpack_Vub` 순서 |
| `dim=96`에서만 실패 | tail (§5-3) |
| `n_rows` 큰 경우만 실패 | `s_in`/`zp_in` 인덱스 (§5-5) |
| in-place만 실패 | §5-4 |
| 전 케이스에서 균일하게 +-2 LSB 이상 | `rne` 대신 `round`를 썼거나 `vinv`에서 32768을 빠뜨림 |
| `n_saturated`가 레퍼런스와 다름 | 포화 **후**의 값을 세고 있음. 포화 전 `vq`를 세야 함 |

---

## 7. 빌드와 실행

디바이스: Galaxy S25 Ultra (`R3CY10WM83Y`), V79.

```bash
# 0. 환경 (한 번). setup_sdk_env.source 는 이 머신에서 실패하니 고치지 말고 손으로.
export HEXAGON_SDK_ROOT=~/workspace/Hexagon_SDK/6.4.0.2
export DEFAULT_HEXAGON_TOOLS_ROOT=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.04
export HEXKL_ROOT=~/workspace/hxkl-beta2/hexkl_addon   # beta2! addons/ 밑은 beta1
export HEXKL_SDK_VER=6.4.0.2
export ANDROID_NDK=~/workspace/android-ndk-r26d
git submodule sync && git submodule update --init --depth 1

# 1. 호스트 먼저 -- DSP 없이 잡히는 건 여기서 다 잡는다
meson build -Denable-transformer=true && ninja -C build
cd build && meson test unittest_mha_htp_host_model --print-errorlogs

# 2. DSP skel
cd test/htp && bash build.sh          # -Werror에서 warning 0이어야 함

# 3+4. ARM 빌드 + push + run: 스크립트가 다 합니다
bash test/htp/run_u8i4_layer_on_device.sh
```

`build`를 쓰세요. 체크인된 `builddir`는 안드로이드 크로스 빌드용이라 호스트
테스트가 안 돕니다.

**이미 물린 함정 4개** (다시 물리지 마세요):
- `NNTRAINER_ROOT`를 ndk-build에 **명시**할 것. 이 개발 머신들의 프로필이 다른
  체크아웃을 가리키는 값을 export 하고 있고 Android.mk의 `ifndef` 기본값이
  조용히 거기에 집니다
- `test/jni/googletest` 심볼릭 링크는 아무도 안 만들어 줍니다 (스크립트가 함)
- `subprojects/iniparser`가 빈 wrap-git placeholder일 수 있습니다
- `qaic`가 `unexpected "o" / expecting "in", "rout" or "inrout"`로 거부하면
  스칼라 out에 `out`을 쓴 것입니다. 이 qaic 버전은 스칼라에도 `rout`만 받습니다

**Hexagon SDK나 디바이스가 없으면**: §6.1(호스트)까지만 하고, §6.2를
"실행하지 못했다"고 **그대로 쓰세요.** 추정해서 초록색으로 보고하지 마세요.
이 브랜치는 이미 그 실패를 한 번 겪었습니다 (`00_START_HERE.md` §5).

---

## 8. RULE ZERO -- 어기면 테스트가 초록이어도 실패입니다

1. 레퍼런스를 고치지 마세요: `nntrainer::compute_rotary_emb_value`,
   `calc_trigonometric_vals_dup`, `hvx_quant_u8.c`, `hvx_convert.h`,
   `hvx_softmax_util.h`. 테스트가 실패하면 **새 코드가 틀린 것**입니다.
2. §6의 허용오차, shape, 매트릭스를 바꾸지 마세요. 코드가 존재하기 전에
   고정된 것들입니다.
3. HVX/HexKL/QuRT API를 발명하지 마세요. 모든 호출은 이 트리가 이미
   include하는 헤더에 있어야 합니다. 없으면 **멈추고 보고하세요**, 그럴듯한
   이름을 쓰지 말고.
4. `nntrainer/tensor/htp_backend/` 아래 기존 파일을 고치지 마세요 -- 전부
   device-verified이고 이 태스크는 어느 것도 바꿀 필요가 없습니다. 필요하다고
   생각되면 멈추고 **어느 줄이 왜인지** 보고하세요.
5. 실패한 테스트를 bound를 낮추거나 케이스를 빼서 "고치지" 마세요.
   멈추고 전체 출력과 함께 보고하세요.
6. §0의 범위 밖을 시작하지 마세요. 읽다가 발견한 문제는 고치지 말고 최종
   보고서에 쓰세요.
7. `git commit -s`, `Co-authored-by:` 트레일러, `[<component>] <subject>`,
   바뀐 줄에 `clang-format-14` (`AGENTS.md`).

---

## 9. 보고 형식

- 실행한 **모든 명령의 완전한 출력**을 붙이세요. 요약 금지, 발췌 금지,
  "all tests passed" 금지. 길어도 붙이세요.
- 실패한 실행도 붙이세요. **실패한 실행이 숨겨진 깨끗한 보고서가 이
  태스크의 최악의 결과입니다.**
- `git diff --stat`의 줄 수.
- §6.2의 출력 표 전체 (오차 예산, amax 비율, `n_saturated`).
- 디바이스에서 돌리지 못했으면 그 사실을 첫 줄에 쓰세요.

---

## 10. 붙여넣을 프롬프트

```
You are implementing one task for nntrainer's Hexagon HTP backend.

BEFORE WRITING ANY CODE:
1. Read docs/htp_attention/00_START_HERE.md in full.
2. Read docs/htp_attention/01_working_style.md in full.
3. Read docs/htp_attention/40_rope_u8_task.md in full. That is your task.
4. Then, in your first reply and before writing code, do three things:
   a) restate §1.2's arithmetic in your own words, including why the
      rounding must be nearbyint and not round;
   b) state why the K path cannot use this kernel (§1.5);
   c) list the files you intend to create or modify, and stop for one round
      if that list includes anything §0 or RULE ZERO told you not to touch.

Work through §3 in order: the host scalar reference and its meson test
first, because that is the only part you can verify without a device. Then
the kernel, then the IDL and skel, then the device test.

RULE ZERO is §8 of the task file. Read it as written; it is not negotiable.
In particular: do not invent an HVX intrinsic. §5-1 warns that
Q6_Wuh_vunpack_Vub's lane order was never compiled by the author of that
document -- confirm it against the SDK header, and if the lane test in §6.2
fails, that is the first place to look.

If you do not have the Hexagon SDK or a device, do §6.1 and say plainly in
your first line of the report that §6.2 was not run. Do not report a device
result you did not observe.

Report per §9: complete output of every command, nothing summarised.
```

한 태스크 = 한 세션. 이 태스크가 device에서 green이 되면 다음은
`attn_forward` 융합(회전을 `hvx_quant_pack_u8_ah`의 AH 타일 쓰기 앞에
붙이기)이고, 그건 별도 세션에서 별도 태스크 문서로 주세요.
