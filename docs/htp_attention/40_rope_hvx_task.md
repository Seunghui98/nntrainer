<!-- SPDX-License-Identifier: Apache-2.0 -->

# 40 -- RoPE를 HVX로: 설계, 단계, 안드로이드 e2e 검증

이 문서 하나만 읽고 작업을 시작할 수 있게 씁니다. 선행 읽기는
`00_START_HERE.md`(무엇이 이미 측정되었는지)와 `01_working_style.md`(어떻게
일하는지)뿐입니다. 참조 구현은 llama.cpp `ggml/src/ggml-hexagon/htp/rope-ops.c`
이고, QNN의 실제 동작 근거는 `ref_16_qnn_optrace_analysis.md`(이하 ref_16)
입니다.

아직 구현 안 됨. 이 문서는 계획입니다.

**개정 이력:** 초판은 "f32로 회전한다"만 썼습니다. QNN이 RoPE를 uint8
액티베이션 위에서 돌린다는 지적을 받고 ref_16을 다시 읽어 §2를 새로 쓰고
Stage 3/4를 갈랐습니다. 결론이 바뀌었으니 §2를 먼저 읽으세요.

---

## 0. 결론부터

**(1) RoPE를 독립된 FastRPC op으로 만들면 오늘 기준 순손해입니다.**

| | |
| :-- | :-- |
| FastRPC 고정 비용 | **~404 us/call** (`00_START_HERE.md` §2, `ref_15` §3) |
| decode 한 토큰의 RoPE 원소 수 (Qwen3-0.6B, layer 1개) | Q 2,048 + K 1,024 = **3,072 floats** |

layer마다 rope 호출을 하나 더 만들면 28 x 404 us = **11.3 ms/token**을
고정비로 냅니다. 그 대가로 옮기는 일감은 ARM NEON에서 마이크로초 단위입니다.

**(2) QNN이 RoPE를 uint8로 도는 건 사실이고, 그 이유는 정확도나 속도가 아니라
그래프 구조입니다.** QNN에서 RoPE는 u8 노드 두 개 사이에 낀 독립 그래프
노드라서 입력도 출력도 u8일 수밖에 없습니다. **우리는 그 제약을 갖지
않습니다** -- 우리 경로에는 RoPE 자리에 op 경계가 없습니다. 근거와 유도는 §2.

**(3) 그래서 옳은 형태는 "u8 rope 커널"이 아니라 "이미 도는 단계 안으로 접기"
입니다.** 접을 자리가 시점마다 다릅니다:

| 시점 | 접을 자리 | 회전 도메인 |
| :-- | :-- | :-- |
| **지금 (T1 이전)** | `q_gather` memcpy (Q), `hexkl_kvq_pack_kt_block`의 fp16 디코드 (K) | **f32** -- 그 자리에 f32가 이미 살아 있음 |
| **T1 이후** | producer의 requantize 단계 (`(int32_acc + bias) * mult >> shift`) | **int32 -> u8**, 회전이 requantize에 합성됨 |

두 경우 모두 **추가 메모리 트래픽 0, 추가 양자화 단계 0**입니다. QNN의 별도
u8 mul 노드보다 낫습니다.

**(4) 회전을 정수 도메인에서 하게 되는 순간, cos/sin은 u8이 아니라 Q15(int16)
여야 합니다.** u8 cos/sin은 각도 오차 상한 3.9e-3 rad, Q15는 3.0e-5 rad.
score 오차는 `|ds| <= eps * ||q|| * ||k||`로 바로 묶입니다 (§2.4). 테이블은
position당 half개(head_dim=128이면 64개)뿐이라 Q15가 공짜입니다.

---

## 1. 지금 RoPE가 어디서 무엇을 하고 있나 (재유도 금지, 코드에서 읽은 것)

### 1.1 호출 지점 3개

`Applications/CausalLM/layers/mha_core.cpp`
`one_batch_incremental_forwarding` 안:

| 줄 | 대상 | 하는 일 |
| :-- | :-- | :-- |
| `:725` | K | **rope + f32->fp16 변환 + KV cache 슬롯에 쓰기**, 한 패스에 |
| `:730` | V | 변환만 (`convert_only=true`), rope 없음 |
| `:747` | Q | in-place rope, f32 유지 |

### 1.2 회전 계약 -- NeoX half-rotation

`nntrainer::compute_rotary_emb_value(width, dim, half_, inout, output, cos_, sin_, only_convert)`
(`nntrainer/tensor/cpu_backend/arm/neon_impl.cpp:2470`, fallback 동일):

```
row 하나(width floats)를 dim 스텝으로 훑고, 각 dim 세그먼트 w 안에서
  k in [0, half_):
    a = inout[w+k]        b = inout[w+k+half_]
    out[w+k]        = a*cos_[k] - b*sin_[k]
    out[w+k+half_]  = a*sin_[k] + b*cos_[k]
```

llama.cpp `HTP_ROPE_TYPE_NEOX`와 **같은 pairing**입니다. `rope_basic`
(인접 쌍) 은 nntrainer에 존재하지 않습니다 -- 포팅하지 마세요 (§3).

`dim`은 `head_dim`, `width`는 한 row에 head가 이어 붙은 길이(Q는
`nHq*head_dim`, K는 `nHkv*head_dim`)입니다. head 경계 = dim 경계라서 커널은
head를 몰라도 됩니다. `rope_partial_rotary_factor != 1`이면 `dim < head_dim`이
되어 나머지 채널은 그대로 복사됩니다.

### 1.3 cos/sin 테이블 -- 설계의 지렛대

`precompute_freqs`(`mha_core.cpp:871`) -> `calc_trigonometric_vals_dup`:

```
cos_[pos][i] = cos(pos * thetas[i]) * attention_scaling      i in [0, half_)
sin_[pos][i] = sin(pos * thetas[i]) * attention_scaling
그리고 [i + half_]에 같은 값을 복제해 둠
```

**(a) 복제된 뒷절반은 아무도 안 읽습니다.** NEON 구현도 `k < half_`에서만
로드합니다. 복제는 다른 함수를 위한 것입니다. 새 커널은 **`half_`개만 받는
계약**으로 가세요. 헤더에 이유를 적으세요.

**(b) rope scaling 3종(`default`/`yarn`/`proportional`)이 전부
`thetas[half_]` + `attention_scaling` 스칼라 두 개로 환원됩니다.**
position에 독립입니다. 그래서 llama.cpp가 DSP로 들고 간
`rope_yarn_one`/`corr_dims`/`ext_factor`를 **우리는 포팅할 이유가 없습니다.**

### 1.4 얼마나 벌 수 있나 -- 측정 안 됨, 형태만

**측정된 RoPE 비용은 이 프로젝트에 없습니다.** 만들지 마세요. 원소 수만
shape에서 나옵니다 (Qwen3-0.6B: head_dim 128, nHq 16, nHkv 8):

| | Q | K | V(변환만) | layer 합 |
| :-- | --: | --: | --: | --: |
| decode (n_query=1) | 2,048 | 1,024 | 1,024 | **4 K floats** |
| prefill (n_query=1024) | 2.10 M | 1.05 M | 1.05 M | **4.2 M floats** |

**decode에서는 무의미, prefill에서만 의미가 있습니다.**

**첫 번째 할 일은 코드가 아니라 측정입니다** (`CLAUDE.md` 습관 1):
`apply_rotary_emb_tensor_v2` 세 호출을 감싸 prefill 1024 / decode 1 토큰의
실측 us를 `ROPE_FIELD path=... field=us value=...`로 찍으세요. 그 숫자가
Stage 3의 기대 이득 상한이고, doc 34/35 표에 들어갈 수 있는 유일한 근거입니다.

---

## 2. QNN은 RoPE를 uint8로 돈다 -- 확인된 것, 그게 뜻하는 것

### 2.1 트레이스가 실제로 말하는 것

ref_16에서 직접 읽은 것만:

| 근거 | ref_16 |
| :-- | :-- |
| 액티베이션 dtype 분포: `QUInt8` **1,312개** / 가중치 `QInt8` 100 / bias·누산 `QInt32`,`Int32` 532 | §2 |
| **"데이터 경로에 float dequant/requant 커널이 하나도 없습니다"** | §2.1 |
| HMX 출력이 `(int32_acc + folded_bias) * multiplier >> shift` 한 번으로 uint8이 됨 | §2.1 |
| RoPE `rotate_half`의 `slice`/`cat` 4노드가 소멸, 인접 op 안에서 in-place `q::Concat`(0 cycle) | §1.4 |
| RoPE는 Q/K 체인 위에 있음: `linear -> RMSNorm -> transpose -> RoPE -> QK^T` | §7.2 |

**그래서 QNN의 RoPE = u8 텐서 위의 mul 2개 + add 1개이고, slice/concat은
공짜입니다.** 지적하신 대로입니다.

**정직하게 모르는 것 두 가지, 여기서 못 메웁니다:**

1. **RoPE 자체의 사이클 수를 이 트레이스에서 뽑을 수 없습니다.** §7.1의
   `node_mul_4`(span 77,695~4,500,706)는 rope가 아니라 **GQA
   `repeat_kv`** 입니다 -- §9.1이 "최대 병목"으로 따로 다루고, doc 36 §2가
   8,011,003 unit-cycles(34.7%)로 인용하는 그 노드입니다. rope의 mul은
   항목화되어 있지 않습니다. **"QNN의 rope는 X us"라고 쓰지 마세요.**
2. **cos/sin 상수 테이블의 dtype이 u8인지 int16인지 트레이스에 없습니다.**
   §2의 분포는 액티베이션/가중치/bias만 나눕니다. 아래 §2.4는 이걸
   가정하지 않고, 우리가 무엇을 골라야 하는지만 유도합니다.

### 2.2 왜 u8인가 -- 이유는 그래프 구조지 성능이 아니다

QNN에서 RoPE는 **독립된 그래프 노드**입니다. 앞(RMSNorm)도 뒤(QK^T)도 u8
텐서를 주고받는 노드이고, §2.1이 말하듯 데이터 경로에 float 커널이 아예
없습니다. 그러니 RoPE의 입력은 u8이고 출력도 u8이어야 합니다. **선택이 아니라
귀결입니다.**

거꾸로 말하면: **op 경계가 없는 곳에서는 u8일 이유가 없습니다.** QNN은 실제로
그렇게 합니다 -- §1.1의 RMSNorm 융합은 `x^2`, `mean`, `rsqrt` 같은 중간
텐서를 "메모리에 쓰지 않는 것이 핵심 이득"이라고 명시하고, 그 중간값들은 u8이
아닙니다. 융합 안쪽은 자유롭고, 경계에서만 u8입니다.

**우리 RoPE 자리에는 경계가 없습니다.** §2.3이 그 자리를 정확히 짚습니다.

### 2.3 우리 경로에 f32가 살아 있는 정확한 두 지점

RoPE를 DSP로 옮긴다는 건 새 op을 만드는 게 아니라 **이미 도는 루프에 6줄을
끼우는 것**입니다. 끼울 자리는 코드에서 확인됩니다:

**K -- `hexkl_kv_quant.c:129~`**

```
row_f16[d] --kvq_fp16_to_fp32--> float v --amax--> scale --quantize--> int8
                                 ^^^^^^^
                                 여기. 쌍의 짝(d, d+half)이 같은 row 안에 있음
```

position은 `kv_from + r`로 이미 압니다. 회전을 여기 넣으면 추가 로드/스토어가
없습니다. 주의: 이 함수는 **fp16을 두 번 디코드**합니다(amax 패스 + quant
패스, `.c:175`의 주석이 "simplicity over avoiding re-decode"라고 밝힘). 회전도
두 번 하거나, 이 김에 한 번만 디코드하도록 고치거나 -- **둘 다 정답이지만
고르고 이유를 적으세요.**

**Q -- `hexkl_attn_u8.c:406`**

```
memcpy(ctx->q_gather + m*head_dim, <q의 해당 head 행>, ...)
^^^^^^ 이 memcpy를 "회전하며 쓰기"로 바꾸면 끝
```

이 자리가 중요한 이유가 하나 더 있습니다. Q의 u8 양자화는 **2패스**입니다
(`hvx_quant_rows_u8_params`로 scale/zp를 구하고, `hvx_quant_pack_u8_ah`가 x를
다시 읽음). scale/zp는 **회전된 값**에서 나와야 하므로, 회전은 반드시 params
패스보다 앞이어야 합니다. gather memcpy가 정확히 그 앞입니다.

**V는 건드리지 않습니다** -- rope가 없고, 변환은 이미 kv_append 안에 있습니다.

### 2.4 회전을 정수로 하면 무엇이 달라지나 -- 유도

**(a) 범위는 sqrt(2)배까지만 늘어난다.** 회전은 각 쌍 `(a, b)`에 대한
직교변환이라 `a^2 + b^2`를 보존합니다. 따라서
`|out| <= sqrt(a^2 + b^2) <= sqrt(2) * max(|a|, |b|)`, 즉

```
amax_after <= sqrt(2) * amax_before        (tight)
```

**최대 0.5비트**입니다. 회전 후에 다시 u8로 담는 것이 범위 때문에 깨지는
일은 없습니다. (우리 K 양자화는 kv position마다 head_dim 위의 amax를 잡으므로
-- `hexkl_kv_quant.h` -- **회전 후에 scale을 잡는 것이 그냥 옳습니다.** 회전
전 scale을 재사용하려 들지 마세요.)

**(b) u8->u8 회전은 양자화 라운드를 하나 더 낸다.** 입력이 이미 u8(255단계,
amax 대비 ~0.4% 오차)인데 출력을 다시 u8로 담으면 반올림이 한 번 더 붙습니다.
**f32 창 안에서 회전하면 이 라운드가 아예 없습니다** -- 양자화는 원래 있던
그 한 번뿐입니다. 이게 §0(3)이 "QNN의 별도 u8 mul 노드보다 낫다"고 쓴 이유이고,
QNN이 그렇게 못 하는 건 §2.2의 그래프 제약 때문입니다.

**(c) cos/sin의 dtype이 지배 오차항이 된다.** `cos, sin` in [-1, 1]:

| 표현 | half-step = 각도 오차 상한 eps |
| :-- | --: |
| u8 symmetric | 2/255 / 2 = **3.9e-3 rad** (~0.22도) |
| **Q15 (int16)** | 1/32768 = **3.0e-5 rad** |

score에 미치는 영향은 바로 묶입니다. `s = sum_p |q_p||k_p| cos(dtheta_p +
phi_p)`에서 각 `phi_p`를 `eps_p` (|eps_p| <= eps) 만큼 흔들면

```
|ds| = |sum_p |q_p||k_p| sin(...) eps_p| <= eps * sum_p |q_p||k_p| <= eps * ||q|| * ||k||
```

(Cauchy-Schwarz). `|s| <= ||q||*||k||`이므로 **eps가 그대로 score의 상대
오차 상한**입니다. u8이면 0.4%, Q15면 0.003%. softmax가 logit 오차를
증폭한다는 걸 생각하면 0.4%는 "괜찮다"고 말할 수 있는 숫자가 아닙니다.

테이블은 position당 `half_`개(head_dim 128이면 64개)뿐입니다. **Q15로 가세요.
공짜입니다.** QNN이 u8을 쓰는지 int16을 쓰는지는 모르지만(§2.1), 우리가
u8을 고를 이유는 없습니다.

### 2.5 그래서 T1 이후에는 -- 회전을 requantize에 합성한다

doc 36 §3 T1 item 3(**u8-in / u8-out endpoint**, 블록 12.9 ms 중 ~4.4 ms
가치)이 랜딩하면 `attn_forward`의 `q`가 **u8로 도착합니다.** 그게 T1의
요점입니다(FastRPC 페이로드도 4배 작아짐). 그러면 §2.3의 **Q쪽 f32 창이
사라집니다.** K쪽 창은 kv_append가 fp16을 디코드하는 한 남습니다.

여기서 QNN처럼 "u8 rope 패스"를 따로 만드는 건 우리에겐 후퇴입니다. **RoPE는
액티베이션에 대해 선형**이므로 producer의 requantize에 합성됩니다:

```
지금 producer가 이미 하는 일:   u8_out[c] = (acc[c] + bias[c]) * mult >> shift
회전을 합성한 형태:             t0 = acc[c]*cosQ15 - acc[c+half]*sinQ15   (int64 중간)
                                t1 = acc[c]*sinQ15 + acc[c+half]*cosQ15
                                u8_out[c]      = (t0 >> 15) ... * mult >> shift
                                u8_out[c+half] = (t1 >> 15) ... * mult >> shift
```

`acc[c]`와 `acc[c+half]`는 **같은 producer의 두 출력 채널**이라 동시에 살아
있습니다. 회전이 **이미 도는 단계 안의 정수 곱 2개**가 되고, 별도 패스도
별도 양자화 라운드도 없습니다.

**여기서 반드시 확인할 제약 하나:** Qwen3는 RoPE 앞에 `q_norm`/`k_norm`이
있습니다 (ref_16 §7.2: `linear -> RMSNorm -> transpose -> RoPE`). RMSNorm은
비선형이라 **회전을 projection의 누산기까지 되접을 수 없습니다.** 합성 대상은
**RMSNorm의 출력 requantize** 단계입니다 -- QNN이 `rmsnorm_8_normalize.tcm`을
"정규화 + 재양자화"로 분해해 둔 바로 그 단계(§1.1)와 같은 모양입니다. q/k norm이
없는 모델이라면 합성 대상이 projection의 requantize가 됩니다.

**T1이 이 문서의 f32 설계가 가진 명시된 천장입니다.** Stage 3 코드에
`ponytail:` 주석으로 그렇게 적으세요:

```
ponytail: rotation runs in f32 because a f32 window exists here today
(the gather memcpy / the fp16 decode). T1 (doc 36 §3, u8-in/u8-out
endpoints) removes that window on the Q side. Upgrade path: compose the
rotation into the producer's requantize -- doc 40 §2.5.
```

---

## 3. llama.cpp `rope-ops.c`에서 가져올 것 / 가져오지 말 것

| llama.cpp의 것 | 우리 | 이유 |
| :-- | :-- | :-- |
| `hvx_rope_neox_f32_aa`의 **산술 구조** (qf32 mpy 4 -> sub/add 2) | **가져온다** | pairing이 동일. §4가 이걸 우리 계약으로 다시 쓴 것 |
| `hvx-base.h`의 `hvx_vec_f32_to_f16(_shuff)` / `hvx_vec_f16_to_f32` | **가져온다** | K/V cache가 fp16. `Q6_Vhf_equals_Wqf32` / `Q6_Wsf_vmpy_VhfVhf` 조합은 다시 찾을 필요 없음 |
| `hvx-sin-cos.h` (Chebyshev sin/cos) + `hvx-floor.h` | **거의 안 가져온다** | §5 Stage 2 참고. Q15 테이블이면 transport가 반이라 이득이 더 줄어듦 |
| spad 더블버퍼 + row 프리페치 루프 | **가져오지 않는다** | `hexkl_dma_ring`이 있고 device-verified. rung 2 |
| 인터리브 theta cache + `Q6_W_vdeal_VVR` | **가져오지 않는다** | 우리 cos/sin은 이미 분리된 배열(§1.3). deal이 통째로 사라짐 |
| `rope_basic_f32` (NORMAL mode) | **가져오지 않는다** | nntrainer에 그 모드 없음. YAGNI |
| mrope / imrope / vision, `sections[4]` | **가져오지 않는다** | CausalLM 경로에 없음. YAGNI |
| `rope_yarn_one`, `corr_dims`, `ext_factor`, `beta_*` | **가져오지 않는다** | §1.3(b): 전부 ARM의 `thetas`에 접혀 있음 |
| `fastdiv` 4D (ne0..ne3) 인덱싱 | **가져오지 않는다** | 우리 텐서는 row-major 2D |

한 줄: **가져오는 건 안쪽 루프 6줄과 fp16 변환 헬퍼 2개뿐입니다.**

---

## 4. 커널 설계

`nntrainer/tensor/htp_backend/hvx/hvx_rope_f32.{c,h}` (신규 2파일).
`hvx_softmax_f32.h`의 row-range 시그니처를 따릅니다.

```c
/* out may alias in.  cos_/sin_ are [m_last][half] -- HALF only, see §1.3(a).
   width % dim == 0, dim % 2 == 0, half = dim/2. */
void hvx_rope_neox_rows_f32(const float *in, float *out,
                            uint32_t m_first, uint32_t m_last,
                            uint32_t width, uint32_t dim,
                            const float *cos_half, const float *sin_half);

/* Same rotation, fp16 (uint16 container) output -- the K path's shape. */
void hvx_rope_neox_rows_f32_to_f16(const float *in, uint16_t *out,
                                   uint32_t m_first, uint32_t m_last,
                                   uint32_t width, uint32_t dim,
                                   const float *cos_half,
                                   const float *sin_half);
```

안쪽 루프 (`half`축으로만 벡터화, `LANES=32`):

```c
for (v = 0; v < half / LANES; ++v) {
  HVX_Vector a = vx[v];                       /* in[w + k]        */
  HVX_Vector b = vx_hi[v];                    /* in[w + k + half] */
  HVX_Vector c = vc[v], s = vs[v];
  HVX_Vector ac = Q6_Vqf32_vmpy_VsfVsf(a, c);
  HVX_Vector as = Q6_Vqf32_vmpy_VsfVsf(a, s);
  HVX_Vector bc = Q6_Vqf32_vmpy_VsfVsf(b, c);
  HVX_Vector bs = Q6_Vqf32_vmpy_VsfVsf(b, s);
  vy[v]    = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(ac, bs));
  vy_hi[v] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(as, bc));
}
```

주의 6가지 -- 전부 이 트리에 선례가 있습니다:

1. **정렬 없음.** `HVX_UVector`를 쓰세요 (`hvx_softmax_f32.c:34`가 같은 이유).
2. **tail (`half % 32 != 0`).** head_dim 80/96 모델이 있습니다.
   `hvx_softmax_util.h`의 `load_tail_sf`/`store_tail_sf`를 **재사용**하세요.
3. **`out == in` aliasing이 정상 경로입니다** (Q in-place, `:747`).
   `vy[v]`를 쓰기 전에 `a`, `b`를 둘 다 읽으면 안전 -- 위 루프가 이미 그렇습니다.
4. **`dim < width`인 partial rotary**: `dim` 뒤 채널은 복사. in-place면 no-op.
5. **fp16 변환 반올림.** ARM은 `vcvt_f16_f32`(RNE), DSP는
   `Q6_Vhf_equals_Wqf32`. 같은 RNE로 알려져 있지만 **가정하지 말고 Stage 1
   게이트에서 exact-match 개수를 세서 찍으세요.**
6. **cos/sin 인터페이스는 f32로 두되, 값은 Q15에서 온 것일 수 있게.** §2.4가
   정수 경로에서 Q15를 요구하는데, Stage 1의 f32 커널은 f32 테이블을 받습니다.
   host가 Q15로 보내고 DSP가 f32로 펼치는 형태를 **Stage 1부터** 쓰면
   transport가 반이 되고 Stage 4로 갈 때 테이블 포맷이 안 바뀝니다.
   테이블 dtype을 IDL에 `int16`으로 박고 스케일은 `1/32768` 고정.

`hvx_worker_pool`은 Stage 1에서 쓰지 마세요. row가 독립이라 나중에 붙이는 게
한 줄입니다.

---

## 5. 단계

### Stage 1 -- 커널 + 정확도 harness (게이트용, 프로덕션 아님)

cos/sin은 host가 보냅니다 (Q15 int16, §4-6). DSP는 회전만 합니다.

```
// Accuracy/perf harness for the HVX RoPE kernel. NOT a production path:
// a per-layer rope round trip costs ~404 us of FastRPC against a few
// microseconds of arithmetic (doc 40 §0). The production path is the
// fusion in attn_kv_append / attn_forward (doc 40 §2.3).
// cos/sin arrive as Q15 int16 (scale 1/32768) -- doc 40 §2.4.
AEEResult rope_f32(in uint32 n_rows, in uint32 width, in uint32 dim,
                   in sequence<float> x,
                   in sequence<int16> cos_q15, in sequence<int16> sin_q15,
                   rout sequence<float> y);

AEEResult rope_f32_to_f16(in uint32 n_rows, in uint32 width, in uint32 dim,
                          in sequence<float> x,
                          in sequence<int16> cos_q15,
                          in sequence<int16> sin_q15,
                          rout sequence<uint16> y);
```

skel: `test/htp/nntr_hvx_rope.c` (`nntr_hvx_softmax.c`를 본뜸).

**게이트** (`unittest_hvx_rope`, device):
- **같은 Q15 테이블을 먹인** ARM `compute_rotary_emb_value` 결과 대비
  max abs err **<= 1e-6**. (f32 테이블과 비교하면 Q15 양자화 오차가 섞여
  커널 버그와 구분이 안 됩니다. **비교 기준을 Q15로 통일하세요.**)
- **별도로**, Q15 테이블 자체가 f32 테이블 대비 얼마나 어긋나는지 표로 출력
  -- 이게 §2.4의 3.0e-5 rad 주장을 실제 데이터로 확인하는 유일한 자리입니다
- fp16 경로: ARM 결과와 bit-exact 비율을 **출력**(실패 기준 아님)
- `y == x` (in-place) 결과가 별도 버퍼 결과와 **bitwise 동일**
- shape: `dim` {64, 96, 128} x `width/dim` {1, 8, 16} x `n_rows` {1, 7, 32, 1024}

### Stage 2 -- cos/sin을 DSP에서 (선택, 우선순위 낮음)

`thetas[half]` + `attention_scaling`만 세션에 올리고 DSP가
`cos(p*thetas[i])`를 Chebyshev로 계산 (llama.cpp `hvx-sin-cos.h`).

**초판보다 매력이 더 떨어졌습니다.** Q15 테이블이면 prefill 1024에서
`1024 x 64 x 2 x 2 B = 256 KB`/call이고, 없애는 대가로 실제 위험을 삽니다:
`p * thetas[i]`는 `p`가 40,960까지 가고 `thetas[0] = 1.0`이라 인자가 최대
~4e4 rad입니다. f32에서 그 크기의 ulp는 ~0.004 rad -- **§2.4가 u8 cos/sin을
탈락시킨 그 크기와 같습니다.** 즉 인자 축소를 f32로 하면 Q15로 얻은 정밀도를
그대로 반납합니다.

**하려면 게이트는 "Stage 1과 같다"가 아니라
"position p in {0, 1, 127, 1023, 4095, 32767}별 max abs err 표를 출력한다"
입니다.** 표가 나쁘면 Stage 2는 버리세요. **Stage 3/4는 Stage 2에 의존하지
않습니다.**

### Stage 3 -- f32 창 안으로 융합 (지금의 프로덕션 경로)

§2.3의 두 자리에 넣습니다. IDL 변경은 `attn_kv_append`와 `attn_forward`에
`in uint32 rope_h`(0이면 rope 안 함) 하나씩 추가하는 것뿐이고, **K rows는
여전히 fp16으로 건너옵니다** -- pre-RoPE 값일 뿐이라 transport 바이트는
그대로입니다.

`rope_register(in sequence<int16> cos_q15_table, in sequence<int16>
sin_q15_table, ...)` 로 테이블을 세션에 등록할지, call마다 보낼지는
`n_query`에 달렸습니다. decode는 1행(256 B)이라 per-call이 맞고, prefill은
256 KB라 등록이 낫습니다. **둘 다 만들지 말고 per-call로 먼저 하고
측정 후에 결정하세요.**

**반드시 결정해야 하는 seam 하나.** ARM KV cache(`b_cache_key_step`)는 지금
**post-RoPE** fp16을 담고 CPU fallback이 그걸 씁니다. DSP가 rope를 하면
shadow가 pre-RoPE가 됩니다:

- (a) shadow를 pre-RoPE로 바꾸고 CPU fallback도 고친다 -- A/B 비교가 죽음
- (b) ARM rope를 CPU cache용으로 남긴다 -- 이득이 사라짐. 임시로만
- (c) **권장:** shadow는 pre-RoPE, CPU 비교가 필요할 때만 테스트 경로에서
  ARM이 자기 사본에 rope 적용. A/B 유지 + 프로덕션 중복 없음

**게이트:** 같은 pre-RoPE K/Q에 대해 `attn_forward(rope_h != 0)`의 결과가
ARM이 rope를 적용한 뒤 `attn_forward(rope_h == 0)`을 부른 결과와 일치.
허용오차는 doc 30 §4의 `(w_k, w_v)` 표를 **그대로** 씁니다 -- rope는 quantize
앞단이므로 오차 예산은 이미 그 표가 정의한 것입니다.

### Stage 4 -- T1 이후: 회전을 requantize에 합성 (아직 시작하지 말 것)

§2.5. **T1(doc 36 §3)이 랜딩하기 전에는 만들 대상 자체가 없습니다.**
지금 할 일은 코드가 아니라 Stage 3에 `ponytail:` 주석을 정확히 남기는 것
(§2.5의 문안 그대로)과, T1 작업자가 이 문서 §2.5를 읽게 doc 36 §3 T1 항목에서
여기를 링크하는 것입니다.

Stage 4를 실제로 할 때의 게이트: **Stage 3의 결과와 비교**합니다. 정수 합성이
f32 회전 대비 얼마나 어긋나는지가 유일한 관심사이고, §2.4(a)의 sqrt(2) 범위
확대가 실제로 관측되는지도 같이 찍으세요 (관측 amax 비율 히스토그램).

---

## 6. 파일 목록

| 파일 | 상태 | Stage |
| :-- | :-- | :-- |
| `nntrainer/tensor/htp_backend/hvx/hvx_rope_f32.h/.c` | 신규 | 1 |
| `test/htp/nntr_hvx_rope.c` | 신규 | 1 |
| `test/htp/nntr_hvx.idl` | `rope_f32`, `rope_f32_to_f16` 추가 | 1 |
| `test/htp/build.sh` | `$SRCS`에 위 2개 `.c` 추가 | 1 |
| `test/unittest/unittest_hvx_rope.cpp` | 신규 (device gtest) | 1 |
| `test/jni/Android.mk` | `unittest_hvx_rope` 모듈 (`unittest_hvx_add` 블록 복사) | 1 |
| `test/htp/run_u8i4_layer_on_device.sh` | `unittest_hvx_rope` 한 줄 추가 | 1 |
| `hexkl_kv_quant.c` (K), `hexkl_attn_u8.c:406` (Q), IDL `attn_*` | 수정 | 3 |
| `Applications/CausalLM/layers/mha_core.cpp` | rope 호출 조건부화 | 3 |

`.c`이지 `.cpp`가 아닙니다. skel은 `hexagon-clang`으로 **C**로 빌드되고
libnntrainer도 C++ 런타임도 링크되지 않습니다 (`00_START_HERE.md` §5의
defect 1이 정확히 그 실수였습니다). `nntrainer::` 심볼을 하나라도 부르면
DSP에 못 들어갑니다.

---

## 7. 안드로이드 기기에서 빌드하고 e2e 검증하기

디바이스: Galaxy S25 Ultra (`R3CY10WM83Y`), V79. 전부 이 트리에서 이미 동작이
확인된 절차입니다 (`13_htp_pr_plan.md` §2/§3a,
`test/htp/run_u8i4_layer_on_device.sh`).

### 7.0 환경 (한 번)

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

### 7.1 호스트 먼저 -- DSP 없이 잡히는 버그를 여기서 잡는다

Stage 1의 산술과 §2.4의 Q15 오차 표는 스칼라로 먼저 확인됩니다.
`mha_htp_host_model.cpp` 옆에 스칼라 rope 레퍼런스를 두고:

```bash
meson build -Denable-transformer=true
ninja -C build
cd build && meson test unittest_mha_htp_host_model --print-errorlogs
```

`build`를 쓰세요. 체크인된 `builddir`는 안드로이드 크로스 빌드용입니다.

### 7.2 DSP skel

```bash
cd test/htp && bash build.sh
# -> test/htp/build/libnntr_hvx_skel.so  (v79, hexkl 6.4.0.2), -Werror에서 warning 0
```

`qaic`가 `unexpected "o" / expecting "in", "rout" or "inrout"`로 거부하면
스칼라 out 파라미터에 `out`을 쓴 것입니다. 이 qaic 버전은 스칼라에도 `rout`만
받습니다.

### 7.3 ARM 쪽 -- libnntrainer.so 먼저, 그 다음 gtest

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

### 7.4 push + run

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

### 7.5 e2e -- 모델 레벨에서 무엇을 비교하나

여기까지는 커널 테스트입니다. Stage 3이 끝난 뒤에야 e2e가 성립합니다:

1. **같은 프롬프트, 같은 seed, greedy decoding**으로 CPU 전용 실행과 HTP
   실행의 **생성 토큰 열이 같은지.** 유일한 진짜 e2e 판정입니다.
2. 다르면 어디서 갈렸는지: `attn_scores_debug`(IDL에 이미 있음)로 layer 0의
   S band를 뽑아 ARM `compute_kcaches_fp32_reference`와 비교. **rope 산술이
   틀렸으면 첫 토큰부터** 틀리고, **position 계산이 틀렸으면 position이
   커질수록** 벌어집니다. 두 패턴을 구분해서 보고하세요.
3. prefill 512 / 1024에서 layer당 us를 찍어 doc 34/35 표에 RoPE 행 추가.
   §1.4의 "먼저 측정"이 before, 이게 after입니다.

```bash
adb shell "cd $D && LD_LIBRARY_PATH=$D ADSP_LIBRARY_PATH=$D \
  ./nntrainer_causallm --model qwen3-0.6b --prompt-file p.txt --greedy --max-new 64" \
  | tee /tmp/rope_htp.log
diff <(grep '^TOKEN ' /tmp/rope_cpu.log) <(grep '^TOKEN ' /tmp/rope_htp.log)
```

---

## 8. Acceptance

| 항목 | 기준 |
| :-- | :-- |
| Stage 1 f32 | **같은 Q15 테이블을 먹인** `compute_rotary_emb_value` 대비 max abs err **<= 1e-6** |
| Q15 테이블 자체 | f32 테이블 대비 각도 오차를 **표로 출력**. §2.4의 3.0e-5 rad를 확인하는 자리 |
| Stage 1 fp16 | ARM 대비 bit-exact 비율을 **출력**(실패 기준 아님), max abs err <= 1e-3 |
| Stage 1 in-place | `y==x` 결과가 별도 버퍼 결과와 **bitwise 동일** |
| Stage 1 shape | `dim` {64, 96, 128} x `width/dim` {1, 8, 16} x `n_rows` {1, 7, 32, 1024} |
| Stage 2 (하면) | position {0, 1, 127, 1023, 4095, 32767}별 max abs err **표 출력**. 표 없이 머지 금지 |
| Stage 3 정확도 | doc 30 §4의 `(w_k, w_v)` 허용오차 표를 그대로 |
| Stage 3 범위 | 회전 후/전 amax 비율을 출력. §2.4(a)의 sqrt(2) 상한 안인지 확인 |
| Stage 3 e2e | greedy 토큰 열이 CPU와 동일 |
| 성능 | before/after를 `ROPE_FIELD path=... field=us value=...`로 출력. **threshold assert 금지** (thermal 때문에 flaky) |

허용오차를 올려서 통과시키지 마세요. 어긋나면 보고할 finding입니다. 1e-3쯤의
f32 오차는 tolerance 문제가 아니라 pairing/tail/index 버그입니다.

---

## 9. 하지 말 것

1. **rope를 독립 FastRPC op으로 프로덕션에 연결하지 말 것** (§0-1).
   harness entry는 harness라고 IDL 주석에 쓸 것.
2. **QNN이 u8로 하니까 우리도 u8로 하자, 로 가지 말 것** (§2.2). QNN은
   그래프 경계 때문에 강제된 것이고 우리는 그 경계가 없습니다. T1 이후에도
   답은 "별도 u8 패스"가 아니라 "requantize에 합성"(§2.5)입니다.
3. **cos/sin을 u8로 담지 말 것** (§2.4c). Q15.
4. `nntrainer/tensor/htp_backend/` 아래 기존 파일은 device-verified입니다.
   Stage 3에서 `hexkl_kv_quant.c` / `hexkl_attn_u8.c`를 건드릴 때는 **그
   변경만으로 device 재검증**을 돌리세요.
5. yarn / mrope / vision / NORMAL-mode rope 포팅 금지 (§3).
6. 새 파일은 `.c`. `nntrainer::` 심볼 금지 (§6).
7. `hvx_softmax_util.h`의 tail/reduce 헬퍼를 다시 쓰지 말 것 -- 재사용.
8. "QNN의 rope는 X us"라고 쓰지 말 것 -- 이 트레이스에 없습니다 (§2.1).
9. "inspection으로 확인함"을 검증이라고 보고하지 말 것.
10. 커밋: `git commit -s`, `Co-authored-by:` 트레일러,
    `[<component>] <subject>`, 바뀐 줄에 `clang-format-14`.

---

## 10. 작업 순서 요약

```
0.  ARM RoPE 실측 (prefill 1024 / decode 1)             <- 코드보다 먼저
1.  hvx_rope_f32.{c,h} + 호스트 스칼라 레퍼런스 + Q15 오차 표 + meson 테스트
2.  IDL 2개 (cos/sin은 int16 Q15) + nntr_hvx_rope.c + build.sh
3.  unittest_hvx_rope.cpp + Android.mk + device run       <- Stage 1 게이트
4.  hexkl_attn_u8.c:406의 q_gather memcpy를 회전으로     <- Stage 3 (Q)
5.  hexkl_kv_quant.c의 fp16 디코드 안에 회전            <- Stage 3 (K)
6.  mha_core.cpp seam (§5의 (c)) + e2e 토큰 열 비교
7.  doc 34/35 표에 RoPE 행 추가 (0번 before, 6번 after)
--- 여기까지가 지금 할 일 ---
8.  (선택) Stage 2: rope_register + on-DSP sin/cos + 오차 표
9.  T1 랜딩 후: Stage 4 (회전을 requantize에 합성, §2.5)
```

3번까지가 한 PR, 4~7번이 다음 PR입니다. 8번은 게이트 표가 나쁘면 버립니다.
9번은 T1 없이는 시작하지 마세요.
