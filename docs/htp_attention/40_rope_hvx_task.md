<!-- SPDX-License-Identifier: Apache-2.0 -->

# 40 -- RoPE on HVX, uint8 in / uint8 out

이 문서 하나만 읽고 작업을 시작할 수 있게 씁니다. 선행 읽기는
`00_START_HERE.md`(무엇이 이미 측정되었는지)와 `01_working_style.md`(어떻게
일하는지)뿐입니다. 참조 구현은 llama.cpp
`ggml/src/ggml-hexagon/htp/rope-ops.c`, QNN 동작 근거는
`ref_16_qnn_optrace_analysis.md`(이하 ref_16)입니다.

아직 구현 안 됨. 이 문서는 계획입니다.

**요구사항 (고정):** QNN과 같은 계약을 써야 합니다 -- **액티베이션 입력이
`QUInt8`, 출력도 `QUInt8`.** 이 문서는 그 계약을 전제로 설계합니다. f32로
회전하는 안은 §5(K 경로)를 빼고는 채택하지 않습니다.

**개정 이력:** 1판은 f32 회전, 2판은 QNN이 u8을 쓰는 이유 분석. 3판(이 문서)은
u8->u8이 요구사항으로 확정되어 그에 맞춰 전체를 다시 씀. 이전 판의 f32 설계는
§5의 K 경로에만 남았습니다.

---

## 0. 결론부터

**(1) u8->u8 계약의 산술은 완전히 결정되어 있습니다** -- §1이 그 식이고,
zero-point 접기와 정수 multiplier/shift가 ref_16 §2.1/§2.2의 QNN 형태와
동형입니다. 여기엔 자유도가 없습니다.

**(2) 진짜 설계 결정은 산술이 아니라 `s_out` / `zp_out`을 무엇으로 두느냐
입니다** (§2). 회전은 직교변환이라 `amax_out <= sqrt(2) * amax_in`이 tight
상한이고, 여기서 세 가지 모드가 갈립니다. **saturation 카운터는 옵션이
아니라 필수 출력입니다** -- 이 값이 잘못됐을 때 조용히 틀리는 유일한
계기판입니다.

**(3) u8->u8은 양자화 라운드를 하나 더 냅니다. 피할 수 없습니다.**
입력이 u8이고 출력도 u8이면 op 내부에 반올림이 정확히 한 번 더 생깁니다.
크기는 유도됩니다 (§3): QK^T score의 상대 오차가 **~1.3% -> ~1.8%**,
즉 **1.4배**. 새로운 자릿수의 오차가 아니라 이미 내고 있는 것의 1.4배입니다.
이 숫자를 PR 본문에 쓰세요.

**(4) 커널 내부 산술은 f32로 하는 것이 수치적으로 정수와 동등합니다** (§4).
`(q - zp)` 는 [-255, 255], Q15 cos/sin과의 곱은 최대 8.4e6 < 2^24이라
**f32에서 각 곱이 정확(exact)** 합니다. 정수로 가야 하는 경우는 딱 하나,
**QNN 레퍼런스와 bit-parity가 요구될 때**입니다. 그건 결정해서 알려주세요.

**(5) 독립 FastRPC op으로는 decode에서 여전히 못 씁니다.** u8 I/O가
transport를 4배 줄이지만(prefill 1024 Q 기준 16.8 MB -> 4.2 MB) **~404 us/call
고정비는 그대로**입니다. layer당 한 번이면 28 x 404 us = 11.3 ms/token.
harness와 prefill 실험까지가 독립 op의 용도이고, 프로덕션은 §7 Stage 3의
융합입니다.

---

## 1. 계약 -- 정확한 식

### 1.1 기호

| | |
| :-- | :-- |
| `qa`, `qb` | 입력 u8 코드. 쌍은 `(w+k, w+k+half)`, `half = dim/2` (NeoX half-rotation) |
| `s_in`, `zp_in` | 입력 양자화. `x_real = (q - zp_in) * s_in` |
| `s_out`, `zp_out` | 출력 양자화. `y = round(y_real / s_out) + zp_out`, clamp [0,255] |
| `cosQ[k]`, `sinQ[k]` | **Q15 int16**, 실수값 = `cosQ[k] / 32768` (§3.3) |

nntrainer의 회전 계약은 `compute_rotary_emb_value`
(`nntrainer/tensor/cpu_backend/arm/neon_impl.cpp:2470`)와 같습니다:

```
out0_real = a_real*cos - b_real*sin
out1_real = a_real*sin + b_real*cos
```

`dim`은 `head_dim`, `width`는 row에 head가 이어 붙은 길이입니다. head 경계 =
dim 경계라 커널은 head를 몰라도 됩니다.

### 1.2 정수 형태 -- QNN §2.1/§2.2와 동형

```
t0 = (qa - zp_in)*cosQ[k] - (qb - zp_in)*sinQ[k]        # int32, exact
t1 = (qa - zp_in)*sinQ[k] + (qb - zp_in)*cosQ[k]

y0 = clamp_u8( rne(t0 * M) >> S  + zp_out )
y1 = clamp_u8( rne(t1 * M) >> S  + zp_out )
```

여기서 `(M, S)`는 실수 배율 `s_in / (s_out * 32768)`를 정수
multiplier + shift로 표현한 것입니다. **호출당 한 번** 계산하고 루프 밖에
둡니다 -- ref_16 §2.1의 `scale_convert` / "스케일 준비 체인은 타일 루프
바깥에서 1회만"과 같은 구조입니다.

### 1.3 zero-point 접기 (선택, QNN §2.2와 동형)

전개하면:

```
t0 = qa*cosQ[k] - qb*sinQ[k] - zp_in*(cosQ[k] - sinQ[k])
                               ^^^^^^^^^^^^^^^^^^^^^^^^ bias0[k]
t1 = qa*sinQ[k] + qb*cosQ[k] - zp_in*(sinQ[k] + cosQ[k])
                               ^^^^^^^^^^^^^^^^^^^^^^^^ bias1[k]
```

**주의: 우리 액티베이션 양자화는 per-row입니다** (`hvx_quant_rows_u8_params`가
`scale[m]`, `zp[m]`을 냄). 그래서 `bias`는 `(m, k)`에 의존하고, QNN처럼
그래프 컴파일 타임에 접어 둘 수 없습니다. 대신 `d[k] = cosQ[k] - sinQ[k]`,
`e[k] = sinQ[k] + cosQ[k]`를 position당 한 번 만들고 row마다 `zp_in[m]`를
곱하면 됩니다.

**권장: 접지 말고 `(q - zp_in)`을 먼저 빼세요.** 벡터 명령 수가 같고
(빼기 2회 vs 곱하기 2회 + 더하기 2회), 틀릴 여지가 훨씬 적습니다. 접기는
per-tensor 정적 양자화(QNN 형태)로 갈 때 의미가 생기는 최적화입니다.
그때 `bias`가 진짜 상수가 됩니다. 그 시점에 하세요.

---

## 2. `s_out` / `zp_out` -- 이 문서의 진짜 설계 결정

### 2.1 범위 상한, 유도

회전은 각 쌍 `(a, b)`에 대한 직교변환이라 `a^2 + b^2`를 보존합니다:

```
|out0| = |a cos - b sin| <= sqrt(a^2 + b^2) <= sqrt(2) * max(|a|, |b|)
=> amax_out <= sqrt(2) * amax_in            (tight: a=b, theta=45도에서 등호)
```

**최대 0.5비트.** 회전 때문에 u8 범위가 무너지는 일은 구조적으로 없습니다.

### 2.2 세 가지 모드

| 모드 | `s_out` | 패스 | 손실 | 언제 |
| :-- | :-- | :--: | :-- | :-- |
| **A. static** | 호출자가 calibration에서 준 값 | 1 | calibration 품질에 달림 | **기본. QNN 계약 그대로** |
| **B. sqrt(2) 유도** | `sqrt(2) * s_in`, `zp_out = 128` | 1 | 무조건 0.5비트 | calibration 데이터가 없을 때의 안전한 기본값 |
| **C. dynamic** | 회전 결과의 실제 per-row 범위 | 2 | 최소 | 우리 기존 per-row 동적 양자화와 같은 정확도가 필요할 때 |

모드 C는 회전 결과를 f32 스크래치에 한 번 쓰고 범위를 잡은 뒤 양자화합니다.
`hvx_quant_rows_u8_params` + `hvx_quant_pack_u8_ah`의 2패스 구조와 정확히
같은 모양이고, 실제로 **그 두 함수를 재사용하면 새로 쓸 코드가 없습니다**
(rung 2).

**세 모드를 다 만들지 마세요.** IDL은 A만 노출하고(`s_out`, `zp_out`을
파라미터로 받음), B는 호출자가 `sqrt(2)*s_in`을 넘기면 되는 것이고, C는
Stage 3 융합에서만 의미가 있습니다. 커널에 모드 플래그를 넣지 마세요.

### 2.3 saturation 카운터 -- 필수

```c
rout uint32 n_saturated;   /* clamp가 실제로 물린 원소 수 */
```

**옵션이 아닙니다.** `s_out`이 작게 잡히면 결과는 예외 없이, 로그 없이,
그냥 조용히 틀립니다 -- 상위 몇 %가 잘려 나가고 softmax가 그걸 증폭합니다.
카운터가 0이 아니면 그건 성능 튜닝 대상이 아니라 **calibration 버그**입니다.
`01_working_style.md`의 "hardware/calibration 노브는 없애지 말 것"이 정확히
이런 것입니다.

### 2.4 calibration 레시피 (모드 A)

1. calibration 프롬프트 셋으로 prefill을 돌리되, **기존 ARM RoPE 경로**로
   post-RoPE 실수값을 뽑습니다 (`apply_rotary_emb_tensor_v2`의 출력).
2. 텐서별(또는 layer/head별) `amax`를 모읍니다. 상위 0.1% 절단은 **하지
   마세요** -- §2.3의 이유로 clip은 여기서 비쌉니다.
3. `s_out = amax / 127.5`, `zp_out = 128`.
   post-RMSNorm 액티베이션은 대체로 zero-mean이라 대칭이 맞습니다. 아니면
   `zp_out = round(-min/s_out)`으로 비대칭을 쓰세요.
4. **검증:** device 테스트에서 `n_saturated == 0`.

### 2.5 탈출구 하나 -- 이 텐서만 int16

§3의 오차가 실제 모델에서 too much로 나오면, **RoPE 출력만** int16으로
올리는 게 정석 대응입니다. 바이트가 2배지만 재양자화 오차가 256배 줄고,
QNN도 민감한 텐서에 같은 짓을 합니다. `s_out`/`zp_out` 인터페이스는 그대로고
clamp 상한만 바뀝니다. **지금 만들지는 마세요** -- §3의 숫자가 device에서
나온 다음에 결정할 노브이고, 이 문단은 그 노브가 존재한다는 기록입니다.

---

## 3. 정확도 예산 -- 정직하게

### 3.1 무엇이 추가되나

u8->u8 계약이 추가하는 것은 **재양자화 반올림 한 번**입니다. 그게 전부이고,
피할 수 없습니다 (입력이 u8이고 출력이 u8이면 op 안에 반올림이 한 번 더
생기는 건 계약의 정의입니다).

균등 반올림 오차: RMS = `0.289 LSB`. 대칭 범위 `+-amax`를 255단계로 나누면
`LSB = 2*amax/255`이므로

```
RMS(요소 오차) = 0.289 * 2/255 * amax = 0.0023 * amax = amax의 0.23%
```

액티베이션이 대략 `amax ~ 4 sigma`인 정규 분포라면 신호 RMS 대비
`0.0023 * 4 = 0.9%`.

### 3.2 score에 미치는 영향 -- 평균화되지 않는다

`s = sum_i q_i k_i` (head_dim = 128). 오차가 독립이라 하면

```
RMS(ds) = sqrt(2*128) * sigma_d * sigma          (random walk)
RMS(s)  = sqrt(128) * sigma^2
ds/s    = sqrt(2) * sigma_d / sigma
```

**head_dim으로 나눠 줄어들지 않습니다** -- 분자와 분모가 같은 `sqrt(128)`을
갖기 때문입니다. 이걸 "128개 더하니까 평균화된다"고 착각하지 마세요.

| | q의 양자화 라운드 | `sigma_d/sigma` | 상대 score 오차 |
| :-- | :--: | --: | --: |
| 오늘 (f32 q -> u8 한 번) | 1 | 0.9% | **~1.3%** |
| u8->u8 rope | 2 | 1.27% | **~1.8%** |

**1.4배.** 새로운 자릿수가 아니라 이미 내고 있는 것의 1.4배입니다.
device에서 실측으로 이 표를 채우는 것이 Stage 1 게이트의 산출물입니다
(§10).

### 3.3 cos/sin은 u8이 아니라 Q15 int16

`cos, sin` in [-1, 1]에서 half-step이 곧 각도 오차 상한 `eps`입니다:

| 표현 | `eps` |
| :-- | --: |
| u8 symmetric | 2/255/2 = **3.9e-3 rad** (~0.22도) |
| **Q15 (int16)** | 1/32768 = **3.0e-5 rad** |

score 영향은 바로 묶입니다. `s = sum_p |q_p||k_p| cos(dtheta_p + phi_p)`에서
`phi_p`를 `eps_p` (|eps_p| <= eps) 만큼 흔들면

```
|ds| = |sum_p |q_p||k_p| sin(...) eps_p| <= eps * sum_p |q_p||k_p| <= eps*||q||*||k||
```

(Cauchy-Schwarz), 그리고 `|s| <= ||q||*||k||`이므로 **`eps`가 그대로 상대
오차 상한**입니다. u8이면 0.4% -- §3.2의 1.8% 예산에 견줘 무시할 수 없는
크기입니다. Q15면 0.003%로 완전히 사라집니다.

테이블은 position당 `half` 개(head_dim 128이면 64개)뿐입니다. **Q15로
가세요. 공짜입니다.** (ref_16 §2의 dtype 분포는 액티베이션/가중치/bias만
나누므로 QNN이 cos/sin에 무엇을 쓰는지는 트레이스에서 알 수 없습니다.
우리가 u8을 고를 이유는 없습니다.)

---

## 4. 커널

`nntrainer/tensor/htp_backend/hvx/hvx_rope_u8.{c,h}` (신규 2파일).

### 4.1 내부 산술: f32로 하되, 정수와 동등하다는 것을 보이고 간다

```
|q - zp_in| <= 255,  |cosQ| <= 32768
=> |곱| <= 8.36e6 < 2^24 = 1.677e7
```

**f32 mantissa가 24비트이므로 각 곱이 exact입니다.** 두 곱의 차 `t0`는
최대 1.67e7로 2^24를 살짝 넘어 최대 1 ulp(=2)를 잃을 수 있는데, 최종
재양자화의 LSB가 `32768 * s_out/s_in` 규모라 **비교가 안 되게 작습니다.**

그래서:

| | |
| :-- | :-- |
| **기본: f32 내부 산술** | 트리의 `hvx_convert.h` / `hvx_quant_u8.c` 헬퍼를 그대로 재사용. lane interleave 문제 없음. 수치적으로 정수와 동등 |
| **정수(int16) 내부로 가야 하는 경우** | (a) **QNN 레퍼런스와 bit-parity가 요구될 때** -- 반올림 경로가 다르므로 f32는 실격입니다. (b) 측정 결과 커널이 compute bound일 때 -- int16은 벡터당 64 lane, f32는 32 lane |

**(a)는 지금 결정해야 하는 항목입니다.** bit-parity가 요구사항이면 f32 경로를
만들지 말고 처음부터 정수로 가세요. ref_16 §3.3의 elementwise 기준선
(0.07 cy/elem = 8-bit peak의 10~15%, 즉 **메모리 바운드**)을 보면 (b)로
정수가 필요해질 가능성은 낮습니다.

### 4.2 벡터 파이프라인 (f32 내부)

들어가는 쪽 -- u8을 넓히기:

```c
HVX_VectorPair uh = Q6_Wuh_vunpack_Vub(vb);   /* ub -> uh 쌍 */
/* uh -> w -> sf.  int32 -> f32 는 Q6_Vsf_equals_Vw (hvx_dequant_i32.c:51) */
```

> **컴파일 확인 필요:** `Q6_Wuh_vunpack_Vub`는 llama.cpp
> `hmx-mm-kernels-tiled.h:341`에서 확인한 이름이고, **even/odd lane 순서는
> 이 문서를 쓴 환경에서 컴파일해 보지 못했습니다.** 회전은 lane index가
> 곧 `k`이므로 순서가 틀리면 조용히 잘못된 각도를 씁니다. SDK 헤더로
> 확인하고, §10의 "lane 순서" 테스트를 반드시 넣으세요.

회전 (§1.2, `LANES = 32`):

```c
HVX_Vector a  = Q6_Vsf_vsub_VsfVsf(af, vzp);   /* (qa - zp_in) as f32 */
HVX_Vector b  = Q6_Vsf_vsub_VsfVsf(bf, vzp);
HVX_Vector ac = Q6_Vqf32_vmpy_VsfVsf(a, vc);
HVX_Vector as = Q6_Vqf32_vmpy_VsfVsf(a, vs);
HVX_Vector bc = Q6_Vqf32_vmpy_VsfVsf(b, vc);
HVX_Vector bs = Q6_Vqf32_vmpy_VsfVsf(b, vs);
HVX_Vector t0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(ac, bs));
HVX_Vector t1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(as, bc));
```

나가는 쪽 -- **`hvx_quant_u8.c:192-213`의 체인을 그대로 씁니다**:

```c
vq = Q6_Vw_vadd_VwVw(hvx_sf_to_w_rne(Q6_Vsf_vmpy_VsfVsf(t0, vinv)), vzp_out);
/* 4개 모아서 */
HVX_Vector vh01 = Q6_Vh_vpack_VwVw_sat(vq1, vq0);
HVX_Vector vh23 = Q6_Vh_vpack_VwVw_sat(vq3, vq2);
HVX_Vector vb   = Q6_Vub_vpack_VhVh_sat(vh23, vh01);
```

`hvx_quant_u8.c:237`이 이미 확인해 둔 사실: **`Q6_Vub_vpack_VhVh_sat`의
[0,255] 포화는 명시적 clamp와 byte-identical**입니다. 따로 clamp를 넣지
마세요 -- 대신 §2.3의 카운터를 위해 **포화 전 `vq`에 대해** 비교/누적을
하세요 (`Q6_Q_vcmp_gt_VwVw` + population count).

### 4.3 주의 6가지

1. **정렬 없음.** `HVX_UVector`를 쓰세요 (`hvx_softmax_f32.c:34`가 같은 이유).
2. **tail (`half % 32 != 0`).** head_dim 80/96 모델이 있습니다.
   `hvx_softmax_util.h`의 `load_tail_sf`/`store_tail_sf`를 **재사용**하세요.
   u8용 tail은 정렬된 스택 버퍼 스테이징 -- 같은 관용구입니다.
3. **`out == in` aliasing.** u8 in-place도 정상 경로가 되게 하세요.
   `vy`를 쓰기 전에 `a`, `b`를 둘 다 읽으면 안전 -- 위 순서가 이미 그렇습니다.
   **단, 4-row 묶음 pack을 하면 쓰기 시점이 밀리므로 다시 확인해야 합니다.**
4. **`dim < width`인 partial rotary** (`rope_partial_rotary_factor != 1`):
   `dim` 뒤 채널은 **회전 없이 그대로 통과하되 재양자화는 거쳐야 합니다**
   (`s_in != s_out`이면 코드가 바뀜). f32 경로의 단순 memcpy와 다릅니다 --
   여기서 틀리기 쉽습니다.
5. **per-row vs per-tensor.** IDL은 `s_in`/`zp_in`을 길이 `n_rows`의
   sequence로 받고, **길이 1이면 broadcast**로 정의하세요. 전자가 우리 파이프
   라인(`hvx_quant_rows_u8_params`), 후자가 QNN 계약입니다. 한 줄로 둘 다 됩니다.
6. `hvx_worker_pool`은 Stage 1에서 쓰지 마세요. row가 독립이라 나중에 붙이는
   게 한 줄입니다.

---

## 5. K 경로는 u8->u8이 아닙니다 -- 비대칭을 먼저 알고 시작하세요

| | 입력 | RoPE 출력이 되어야 하는 것 |
| :-- | :-- | :-- |
| **Q** | u8 액티베이션 (HMX activation port) | **u8** -- 이 문서의 계약 |
| **K** | fp16 KV row | **i4/i8 대칭, per-kv-position** (`hexkl_kv_quant.h`) |

K는 HMX의 **weight slot**으로 들어가고, `hexkl_kvq_pack_kt_block`이 kv
position마다 head_dim 위의 amax로 대칭 양자화합니다. **u8이 아닙니다.**
u8->u8 커널을 K에 갖다 쓰면 안 됩니다.

K의 올바른 자리는 여전히 `hexkl_kv_quant.c:129~`의 fp16 디코드 안입니다:

```
row_f16[d] --kvq_fp16_to_fp32--> float v --amax--> scale --quantize--> int8
                                 ^^^^^^^ 여기서 회전. 쌍의 짝 (d, d+half)이 같은 row 안
```

position은 `kv_from + r`로 이미 압니다. 추가 로드/스토어 0. **회전 후에
scale을 잡는 것이 옳습니다** -- §2.1의 sqrt(2) 때문에 회전 전 scale을
재사용하면 최대 0.5비트를 잘라먹습니다.

주의: 이 함수는 fp16을 **두 번 디코드**합니다 (amax 패스 + quant 패스,
`.c:175` 주석이 "simplicity over avoiding re-decode"라고 밝힘). 회전도 두 번
하거나, 한 번만 디코드하도록 고치거나 -- **둘 다 정답이지만 고르고 이유를
적으세요.**

**V는 건드리지 않습니다** -- rope가 없고, 변환은 이미 kv_append 안에 있습니다.

---

## 6. IDL

```
// RoPE with the QNN activation contract: QUInt8 in, QUInt8 out.
// NOT a production path on its own -- a per-layer round trip costs
// ~404 us of FastRPC against a few microseconds of arithmetic
// (doc 40 §0-5). This exists to gate the kernel and to measure prefill.
// The production path is the fusion in doc 40 §7 Stage 3.
//
// s_in/zp_in have length n_rows, or length 1 meaning per-tensor
// (doc 40 §4.3-5). cos/sin are Q15 int16, scale 1/32768 (doc 40 §3.3),
// [n_rows][dim/2] -- the HALF table only, the CPU cache's duplicated
// second half is never read.
//
// n_saturated is not diagnostics-only: a nonzero value means s_out is
// wrong and the result is silently clipped (doc 40 §2.3).
AEEResult rope_u8(in uint32 n_rows, in uint32 width, in uint32 dim,
                  in sequence<uint8> x,
                  in sequence<float> s_in, in sequence<int32> zp_in,
                  in sequence<int16> cos_q15, in sequence<int16> sin_q15,
                  in float s_out, in int32 zp_out,
                  rout sequence<uint8> y, rout uint32 n_saturated);
```

`qaic`가 `unexpected "o" / expecting "in", "rout" or "inrout"`로 거부하면
스칼라 out에 `out`을 쓴 것입니다. 이 qaic 버전은 스칼라에도 `rout`만 받습니다.

---

## 7. 단계

### Stage 0 -- 코드보다 먼저: 측정과 결정 두 가지

1. **ARM RoPE 실측.** `apply_rotary_emb_tensor_v2` 세 호출을 감싸 prefill
   1024 / decode 1 토큰의 us를 `ROPE_FIELD path=... field=us value=...`로
   찍으세요. **측정된 RoPE 비용은 이 프로젝트에 아직 없습니다** -- 만들지
   말고 재세요. 이게 Stage 3 이득의 상한이고 doc 34/35 표에 들어갈 유일한
   근거입니다.
2. **bit-parity 결정** (§4.1a). QNN 레퍼런스와 비트 단위로 맞아야 하나요?
   Yes면 f32 내부 산술을 만들지 말고 처음부터 정수로 갑니다.

### Stage 1 -- 커널 + harness + device 게이트

- `hvx_rope_u8.{c,h}`
- 호스트 스칼라 레퍼런스: **int64 정확 산술**로 §1.2를 그대로 구현.
  이게 정답의 정의입니다.
- IDL `rope_u8` + `test/htp/nntr_hvx_rope.c` (`nntr_hvx_softmax.c`를 본뜸)
- `unittest_hvx_rope.cpp` + `Android.mk` + device run

게이트는 §10.

### Stage 2 -- calibration과 `s_out`

§2.4의 레시피를 `tools/`에 스크립트로 넣고, layer별 `s_out` 표를 만듭니다.
**게이트: device 테스트에서 `n_saturated == 0`.** 이 단계 없이 Stage 3으로
가면 §2.3의 조용한 clip을 그대로 프로덕션에 넣게 됩니다.

### Stage 3 -- 융합 (프로덕션 경로)

**Q:** RoPE 출력이 HMX activation port로 바로 가야 하므로, 융합 형태는
`rope + AH tile pack`입니다 -- 평면 row-major u8을 내고 나서 다시 레이아웃
패스를 돌면 §0-5에서 아낀 것을 도로 뱉습니다. `hvx_quant_pack_u8_ah`가 이미
AH 타일(64x32, 2048 B stride)을 씁니다. 회전을 그 앞에 붙이세요.
`attn_forward`에 `in uint32 rope_h` 하나 추가.

**K:** §5. `hexkl_kvq_pack_kt_block`의 fp16 디코드 안. `attn_kv_append`에
`in uint32 rope_h` 하나 추가. **K rows는 여전히 fp16으로 건너오므로**
transport 바이트는 그대로입니다.

**반드시 결정해야 하는 seam 하나.** ARM KV cache(`b_cache_key_step`)는 지금
**post-RoPE** fp16을 담고 CPU fallback이 그걸 씁니다. DSP가 rope를 하면
shadow가 pre-RoPE가 됩니다:

- (a) shadow를 pre-RoPE로 바꾸고 CPU fallback도 고친다 -- A/B 비교가 죽음
- (b) ARM rope를 CPU cache용으로 남긴다 -- 이득이 사라짐. 임시로만
- (c) **권장:** shadow는 pre-RoPE, CPU 비교가 필요할 때만 테스트 경로에서
  ARM이 자기 사본에 rope 적용. A/B 유지 + 프로덕션 중복 없음

### Stage 4 -- (선택) cos/sin을 DSP에서

`thetas[half]` + `attention_scaling`만 세션에 올리고 DSP가
`cos(p*thetas[i])`를 계산 (llama.cpp `hvx-sin-cos.h`).

**우선순위 낮습니다.** Q15 테이블은 prefill 1024에서 256 KB/call이고, 없애는
대가로 실제 위험을 삽니다: `p*thetas[i]`는 `p`가 40,960까지 가고
`thetas[0]=1.0`이라 인자가 최대 ~4e4 rad, f32에서 그 크기의 ulp가 ~0.004 rad
-- **§3.3이 u8 cos/sin을 탈락시킨 그 크기와 같습니다.** 하려면 게이트는
"position p in {0, 1, 127, 1023, 4095, 32767}별 max abs err 표 출력"이고,
표가 나쁘면 버리세요. **Stage 1~3은 여기에 의존하지 않습니다.**

---

## 8. 파일 목록

| 파일 | 상태 | Stage |
| :-- | :-- | :-- |
| `nntrainer/tensor/htp_backend/hvx/hvx_rope_u8.h/.c` | 신규 | 1 |
| `test/htp/nntr_hvx_rope.c` | 신규 | 1 |
| `test/htp/nntr_hvx.idl` | `rope_u8` 추가 | 1 |
| `test/htp/build.sh` | `$SRCS`에 위 2개 `.c` 추가 | 1 |
| `test/unittest/unittest_hvx_rope.cpp` | 신규 (device gtest) | 1 |
| `test/jni/Android.mk` | `unittest_hvx_rope` (`unittest_hvx_add` 블록 복사) | 1 |
| `test/htp/run_u8i4_layer_on_device.sh` | `unittest_hvx_rope` 한 줄 추가 | 1 |
| `tools/rope_calibrate.py` | 신규 (§2.4) | 2 |
| `hexkl_attn_u8.c` (Q, AH pack 앞), `hexkl_kv_quant.c` (K), IDL `attn_*` | 수정 | 3 |
| `Applications/CausalLM/layers/mha_core.cpp` | rope 호출 조건부화 | 3 |

`.c`이지 `.cpp`가 아닙니다. skel은 `hexagon-clang`으로 **C**로 빌드되고
libnntrainer도 C++ 런타임도 링크되지 않습니다 (`00_START_HERE.md` §5의
defect 1이 정확히 그 실수였습니다). `nntrainer::` 심볼을 하나라도 부르면
DSP에 못 들어갑니다.

---

## 9. 안드로이드 기기에서 빌드하고 e2e 검증하기

디바이스: Galaxy S25 Ultra (`R3CY10WM83Y`), V79. 전부 이 트리에서 이미 동작이
확인된 절차입니다 (`13_htp_pr_plan.md` §2/§3a,
`test/htp/run_u8i4_layer_on_device.sh`).

### 9.0 환경 (한 번)

```bash
export HEXAGON_SDK_ROOT=~/workspace/Hexagon_SDK/6.4.0.2
export DEFAULT_HEXAGON_TOOLS_ROOT=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.04
export HEXKL_ROOT=~/workspace/hxkl-beta2/hexkl_addon   # beta2! addons/ 밑은 beta1
export HEXKL_SDK_VER=6.4.0.2
export ANDROID_NDK=~/workspace/android-ndk-r26d
git submodule sync && git submodule update --init --depth 1
```

`setup_sdk_env.source`는 이 머신에서 "missed components"로 실패합니다. 고치지
말고 위 두 변수를 손으로 넣으세요.

### 9.1 호스트 먼저 -- DSP 없이 잡히는 것을 여기서 잡는다

§1.2의 int64 스칼라 레퍼런스, §3의 오차 예산, §2.4의 calibration은 전부
호스트에서 확인됩니다. `mha_htp_host_model.cpp` 옆에 두고:

```bash
meson build -Denable-transformer=true
ninja -C build
cd build && meson test unittest_mha_htp_host_model --print-errorlogs
```

`build`를 쓰세요. 체크인된 `builddir`는 안드로이드 크로스 빌드용입니다.

### 9.2 DSP skel

```bash
cd test/htp && bash build.sh
# -> test/htp/build/libnntr_hvx_skel.so  (v79, hexkl 6.4.0.2), -Werror에서 warning 0
```

**§4.2의 `Q6_Wuh_vunpack_Vub` lane 순서는 여기서 처음 확인됩니다.** 컴파일이
통과한다고 순서가 맞는 건 아니니, §10의 lane 순서 테스트로 확인하세요.

### 9.3 ARM 쪽 -- libnntrainer.so 먼저, 그 다음 gtest

```bash
cd <repo> && ANDROID_NDK=$ANDROID_NDK PATH=$ANDROID_NDK:$PATH \
  ./tools/package_android.sh --arm-arch=armv8.2-a -Dwerror=false
# -> builddir/jni/arm64-v8a/libnntrainer.so

ln -sfn "$PWD/subprojects/googletest/googletest" test/jni/googletest   # 아무도 안 만들어 줌
cd test/jni && "$ANDROID_NDK/ndk-build" \
  NDK_PROJECT_PATH=. NDK_APPLICATION_MK=./Application.mk \
  APP_BUILD_SCRIPT=./Android.mk \
  NNTRAINER_ROOT="<repo>" HEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT" \
  unittest_hvx_rope
# -> test/jni/obj/local/arm64-v8a/unittest_hvx_rope   (libs/ 아님!)
```

`NNTRAINER_ROOT`를 **반드시 명시**하세요. 이 개발 머신들의 프로필이 다른
체크아웃을 가리키는 `NNTRAINER_ROOT`를 export 하고 있고, Android.mk의
`ifndef` 기본값이 조용히 거기에 집니다. `subprojects/iniparser`가 빈
wrap-git placeholder면 `git submodule update --init subprojects/iniparser`.

### 9.4 push + run

```bash
D=/data/local/tmp/htp_rope_test
adb shell mkdir -p $D
adb push test/htp/build/libnntr_hvx_skel.so $D/
adb push test/jni/obj/local/arm64-v8a/unittest_hvx_rope $D/
adb push $ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so $D/
adb shell "cd $D && chmod +x unittest_hvx_rope && \
  LD_LIBRARY_PATH=$D ADSP_LIBRARY_PATH=$D ./unittest_hvx_rope" | tee /tmp/hvx_rope_device_run.log
```

`ADSP_LIBRARY_PATH`가 skel 디렉터리를 가리켜야 DSP가 로드합니다. unsigned PD는
테스트 안에서 `remote_session_control`로 켭니다 -- `unittest_hvx_softmax.cpp`의
fixture를 그대로 복사하세요.

**스크립트로 만드세요.** `run_u8i4_layer_on_device.sh`가 skel 빌드 ->
libnntrainer -> ndk-build -> push -> run -> 요약까지 하는 검증된 형태이고,
`unittest_hvx_rope`를 그 목록에 한 줄 추가하는 게 새 스크립트보다 짧습니다.

### 9.5 e2e -- 모델 레벨에서 무엇을 비교하나

여기까지는 커널 테스트입니다. Stage 3이 끝난 뒤에야 e2e가 성립합니다:

1. **같은 프롬프트, 같은 seed, greedy decoding**으로 CPU 전용 실행과 HTP
   실행의 **생성 토큰 열이 같은지.** 유일한 진짜 e2e 판정입니다.
   §3의 1.8%가 여기서 토큰을 바꾸는지가 실제 질문입니다. 바뀌면 §2.5(int16
   승격)가 답이지 tolerance 완화가 답이 아닙니다.
2. 다르면 어디서 갈렸는지: `attn_scores_debug`(IDL에 이미 있음)로 layer 0의
   S band를 뽑아 ARM `compute_kcaches_fp32_reference`와 비교.
   - **rope 산술이 틀렸으면 첫 토큰부터** 틀림
   - **position 계산이 틀렸으면 position이 커질수록** 벌어짐
   - **`s_out`이 작으면 큰 값에서만** 틀림 (`n_saturated`가 같이 0이 아님)

   세 패턴을 구분해서 보고하세요.
3. prefill 512 / 1024에서 layer당 us를 찍어 doc 34/35 표에 RoPE 행 추가.
   Stage 0의 측정이 before, 이게 after입니다.

```bash
adb shell "cd $D && LD_LIBRARY_PATH=$D ADSP_LIBRARY_PATH=$D \
  ./nntrainer_causallm --model qwen3-0.6b --prompt-file p.txt --greedy --max-new 64" \
  | tee /tmp/rope_htp.log
diff <(grep '^TOKEN ' /tmp/rope_cpu.log) <(grep '^TOKEN ' /tmp/rope_htp.log)
```

---

## 10. Acceptance

| 항목 | 기준 |
| :-- | :-- |
| **정확도, 정수 내부** | 호스트 int64 레퍼런스(§1.2)와 **bit-exact** |
| **정확도, f32 내부** | 호스트 int64 레퍼런스와 **최대 +-1 LSB**, 그리고 **불일치 원소 비율을 출력**. bit-parity가 요구사항이면 이 경로는 실격 (§4.1a) |
| **lane 순서** | `x[m][k] = k` 인 램프 입력 + `cosQ=32767, sinQ=0`(항등 회전)에 대해 출력이 입력과 같아야 함. §4.2의 `Q6_Wuh_vunpack_Vub` even/odd 순서를 잡는 유일한 테스트 |
| **saturation** | calibration된 `s_out`에서 `n_saturated == 0`. 일부러 작은 `s_out`을 주면 0이 아니어야 함 (카운터가 실제로 동작하는지) |
| **범위** | 회전 후/전 `amax` 비율을 출력. §2.1의 `sqrt(2)` 상한 안인지 확인 |
| **오차 예산** | §3.2 표를 device 실측으로 채워 출력: 요소 RMS, 상대 score 오차 (오늘 vs u8->u8) |
| **Q15 테이블** | f32 테이블 대비 각도 오차 표 출력. §3.3의 3.0e-5 rad 확인 |
| **in-place** | `y == x` 결과가 별도 버퍼 결과와 **bitwise 동일** |
| **shape** | `dim` {64, 96, 128} x `width/dim` {1, 8, 16} x `n_rows` {1, 7, 32, 1024} |
| **per-tensor / per-row** | `s_in` 길이 1(broadcast)과 길이 `n_rows` 양쪽 |
| **partial rotary** | `dim < width`에서 회전 안 하는 채널도 **재양자화는 거쳤는지** (§4.3-4) |
| Stage 3 정확도 | doc 30 §4의 `(w_k, w_v)` 허용오차 표를 그대로 |
| Stage 3 e2e | greedy 토큰 열이 CPU와 동일 |
| 성능 | before/after를 `ROPE_FIELD path=... field=us value=...`로 출력. **threshold assert 금지** (thermal 때문에 flaky) |

허용오차를 올려서 통과시키지 마세요. 어긋나면 보고할 finding입니다.

---

## 11. 하지 말 것

1. **`n_saturated`를 빼지 말 것** (§2.3). 이게 없으면 잘못된 `s_out`은
   조용히 틀립니다.
2. **u8->u8 커널을 K 경로에 쓰지 말 것** (§5). K는 i4/i8 대칭 weight slot입니다.
3. **cos/sin을 u8로 담지 말 것** (§3.3). Q15.
4. **회전 전 scale을 회전 후에 재사용하지 말 것** (§2.1). 최대 0.5비트 손실.
5. **`s_out` 모드를 커널 플래그로 만들지 말 것** (§2.2). IDL은 `s_out`을
   받기만 하고, 무엇을 넘길지는 호출자의 문제입니다.
6. **평면 u8을 낸 뒤 별도 레이아웃 패스를 돌지 말 것** (§7 Stage 3 Q).
   AH 타일로 바로 쓰세요.
7. `nntrainer/tensor/htp_backend/` 아래 기존 파일은 device-verified입니다.
   Stage 3에서 `hexkl_kv_quant.c` / `hexkl_attn_u8.c`를 건드릴 때는 **그
   변경만으로 device 재검증**을 돌리세요.
8. yarn / mrope / vision / NORMAL-mode rope 포팅 금지 -- nntrainer의 rope
   scaling 3종은 전부 `thetas[half]` + `attention_scaling` 두 개로
   환원됩니다 (`mha_core.cpp:871`, `calc_trigonometric_vals_dup`).
9. 새 파일은 `.c`. `nntrainer::` 심볼 금지 (§8).
10. `hvx_softmax_util.h`의 tail/reduce 헬퍼를 다시 쓰지 말 것 -- 재사용.
11. **"QNN의 rope는 X us"라고 쓰지 말 것** -- ref_16에 없습니다. §7.1의
    `node_mul_4`는 rope가 아니라 GQA `repeat_kv`이고(§9.1), doc 36 §2가
    8,011,003 unit-cycles(34.7%)로 인용하는 그 노드입니다.
12. "inspection으로 확인함"을 검증이라고 보고하지 말 것.
13. 커밋: `git commit -s`, `Co-authored-by:` 트레일러,
    `[<component>] <subject>`, 바뀐 줄에 `clang-format-14`.

---

## 12. 작업 순서 요약

```
0a. ARM RoPE 실측 (prefill 1024 / decode 1)              <- 코드보다 먼저
0b. bit-parity 요구사항 결정 (§4.1a)                      <- 내부 산술을 가름
1.  호스트 int64 스칼라 레퍼런스 + Q15 오차 표 + meson 테스트
2.  hvx_rope_u8.{c,h}
3.  IDL rope_u8 + nntr_hvx_rope.c + build.sh
4.  unittest_hvx_rope.cpp + Android.mk + device run       <- Stage 1 게이트
5.  calibration 스크립트 + layer별 s_out + n_saturated==0 <- Stage 2 게이트
6.  hexkl_attn_u8.c: 회전을 AH pack 앞에 융합 (Q)          <- Stage 3
7.  hexkl_kv_quant.c: 회전을 fp16 디코드 안에 (K, §5)      <- Stage 3
8.  mha_core.cpp seam (§7의 (c)) + e2e 토큰 열 비교
9.  doc 34/35 표에 RoPE 행 추가 (0a가 before, 8이 after)
--- 여기까지가 지금 할 일 ---
10. (선택) Stage 4: on-DSP sin/cos + 오차 표
```

4번까지가 한 PR, 5~9번이 다음 PR입니다. 10번은 게이트 표가 나쁘면 버립니다.
