<!-- SPDX-License-Identifier: Apache-2.0 -->

# 40 -- RoPE를 HVX로: 설계, 단계, 안드로이드 e2e 검증

이 문서 하나만 읽고 작업을 시작할 수 있게 씁니다. 선행 읽기는
`00_START_HERE.md`(무엇이 이미 측정되었는지)와 `01_working_style.md`(어떻게
일하는지)뿐입니다. 참조 구현은 llama.cpp `ggml/src/ggml-hexagon/htp/rope-ops.c`
이고, **어디를 가져오고 어디를 가져오지 않는지는 §3에 표로 있습니다.**

아직 구현 안 됨. 이 문서는 계획입니다.

---

## 0. 결론부터 -- 순서를 바꿔야 합니다

**RoPE를 "독립된 FastRPC op"으로 만들면 오늘 기준 순손해입니다.** 이것이 이
문서의 가장 중요한 한 줄이고, 아래 두 숫자에서 바로 나옵니다.

| | |
| :-- | :-- |
| FastRPC 고정 비용 | **~404 us/call** (`00_START_HERE.md` §2, `ref_15` §3) |
| decode 한 토큰의 RoPE 원소 수 (Qwen3-0.6B, layer 1개) | Q 2,048 + K 1,024 = **3,072 floats** |

decode에서 layer마다 rope 호출을 하나 더 만들면 28 layer x 404 us = **11.3
ms/token**을 고정비로 냅니다. 그 대가로 옮기는 일감은 layer당 3 K 원소, ARM
NEON에서 마이크로초 단위입니다. 두 자릿수 손해입니다.

그래서 이 작업의 형태는 **"새 op 추가"가 아니라 "이미 있는 DSP 호출 안으로
접어 넣기(fusion)"** 여야 합니다. 다행히 접어 넣을 자리가 **이미 정확히 두
군데** 있고, 둘 다 rope가 필요로 하는 f32 값을 **이미 레지스터에 들고 있습니다**
(§5 Stage 3). 그 자리에 넣으면 추가 메모리 트래픽 0, 추가 transport 0,
추가 round trip 0입니다.

정리:

1. **Stage 1** -- 커널(`hvx_rope_neox_f32`)과 정확도 harness entry를 만든다.
   이건 게이트용이지 프로덕션 경로가 아니다. 그렇게 문서에 쓴다.
2. **Stage 2** -- cos/sin을 DSP에서 계산해 transport를 없앤다. (선택,
   Stage 3의 전제는 아님)
3. **Stage 3** -- `attn_kv_append`(K)와 `attn_forward`(Q) 안으로 접는다.
   **여기가 실제 이득이 나는 유일한 지점.**

Stage 1만 하고 멈추면 "커널은 있는데 아무도 안 쓰는" 상태가 됩니다. 그건 실패가
아니라 **의도된 중간 단계**이고, Stage 3 없이 프로덕션에 연결하지 말라는 뜻입니다.

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

K 경로가 세 가지 일을 한 번에 한다는 게 중요합니다. RoPE를 DSP로 옮긴다는 건
**저 융합을 푸는 게 아니라 통째로 옮기는 것**이어야 합니다.

### 1.2 회전 계약 -- NeoX half-rotation

`nntrainer::compute_rotary_emb_value(width, dim, half_, inout, output, cos_, sin_, only_convert)`
(`nntrainer/tensor/cpu_backend/arm/neon_impl.cpp:2470`, fallback도 동일):

```
row 하나(width floats)를 dim 스텝으로 훑고, 각 dim 세그먼트 w 안에서
  k in [0, half_):
    a = inout[w+k]        b = inout[w+k+half_]
    out[w+k]        = a*cos_[k] - b*sin_[k]
    out[w+k+half_]  = a*sin_[k] + b*cos_[k]
```

llama.cpp의 `HTP_ROPE_TYPE_NEOX`와 **같은 pairing**입니다. `rope_basic`
(인접 쌍 `(2i, 2i+1)`) 은 nntrainer에 존재하지 않습니다 -- 포팅하지 마세요(§3).

`dim`은 `head_dim`이고 `width`는 한 row에 여러 head가 이어 붙은 길이입니다
(Q는 `nHq*head_dim`, K는 `nHkv*head_dim`). 그래서 head 경계 = dim 경계이고,
커널은 head를 몰라도 됩니다. `rope_partial_rotary_factor`가 1이 아니면
`dim < head_dim`이 되어 나머지 채널은 그대로 복사됩니다 -- llama.cpp의
`if (rctx->n_dims < ne0) hvx_copy_f32_uu(...)`와 같은 처리가 필요합니다.

### 1.3 cos/sin 테이블 -- 여기가 설계의 지렛대

`precompute_freqs`(`mha_core.cpp:871`)는 position마다 한 줄씩 만들어 캐시합니다:

```
calc_trigonometric_vals_dup(half_, thetas, cos_[pos], sin_[pos], pos, attention_scaling)
  -> cos_[pos][i] = cos(pos * thetas[i]) * attention_scaling      i in [0, half_)
     sin_[pos][i] = sin(pos * thetas[i]) * attention_scaling
     그리고 [i + half_]에 같은 값을 복제해 둠
```

두 가지가 따라옵니다.

**(a) 복제된 뒷절반은 아무도 안 읽습니다.** NEON 구현도 `k < half_`에서만
`cos_[k]`를 로드합니다. 복제는 다른 함수(`compute_rotary_embedding_value`)를
위한 것입니다. **새 HVX 커널은 `half_`개만 받는 계약으로 갑니다** -- 테이블
바이트가 절반이 되고, 이건 CPU 시그니처와의 의도된 차이이니 헤더에 이유를
적으세요.

**(b) rope scaling 3종(`default` / `yarn` / `proportional`)이 전부
`thetas[half_]` + `attention_scaling` 스칼라 두 개로 환원됩니다.**
`_compute_yarn_parameters` 등은 `thetas`만 다르게 채우고, mscale은
`attention_scaling`에 들어갑니다. position에 독립입니다.

(b)는 llama.cpp가 `rope_yarn_one` / `corr_dims` / `ext_factor` 전부를 DSP에
들고 간 이유를 **우리는 갖지 않는다**는 뜻입니다. yarn을 DSP에 포팅할 필요가
없습니다. ARM이 이미 맞게 계산해서 캐시까지 해 둔 `thetas` 64개(head_dim=128
기준)를 세션당 한 번 올리면 끝입니다.

### 1.4 얼마나 벌 수 있나 -- 측정 안 됨, 형태만

**측정된 RoPE 비용은 이 프로젝트에 없습니다.** 만들지 마세요. 대신 원소 수는
shape에서 바로 나옵니다 (Qwen3-0.6B: head_dim 128, nHq 16, nHkv 8):

| | Q | K | V(변환만) | layer 합 |
| :-- | --: | --: | --: | --: |
| decode (n_query=1) | 2,048 | 1,024 | 1,024 | **4 K floats** |
| prefill (n_query=1024) | 2.10 M | 1.05 M | 1.05 M | **4.2 M floats** |

**decode에서는 아무 의미도 없고, prefill에서만 의미가 있습니다.** prefill
layer당 ~16.8 MB read + ~12.6 MB write이고, 이건 doc 35 §2가 "op 경계마다
DDR f32 왕복"이라고 부른 것과 정확히 같은 종류의 비용입니다.

**첫 번째 할 일은 코드가 아니라 측정입니다** (`CLAUDE.md`의 습관 1):
`apply_rotary_emb_tensor_v2` 세 호출을 감싸서 prefill 1024 / decode 1
토큰에서의 실측 us를 `ROPE_FIELD path=... field=us value=...` 형태로
찍으세요. 그 숫자가 Stage 3의 기대 이득 상한입니다. 그게 doc 34/35의 표에
들어갈 수 있는 유일한 정직한 근거이고, 없으면 이 작업 전체가 "QNN이 하니까
우리도"가 됩니다.

---

## 2. 왜 굳이 하나 -- 진짜 이유 하나, 가짜 이유 하나

**진짜:** doc 35 §1이 지적한 비교 불일치. QNN의 "attention layer"는 qkv
projection + RoPE + SDPA + o_proj 전부이고 우리 숫자는 SDPA 코어뿐입니다.
`hexkl_attn_u8.h:139,:203`이 Q/K를 "post-RoPE"로 문서화하고 있는 게 그
증거입니다. RoPE가 DSP로 들어가면 doc 35 §3의 세 번째 레버(block 융합,
doc 36 §5의 T3)에 필요한 조각이 하나 채워집니다. **이 작업의 가치는 rope
자체가 아니라 T3의 선행 조각이라는 데 있습니다.**

**가짜:** "rope 연산이 비싸서". 안 비쌉니다 (§1.4). 이 이유로 정당화하는 문장을
PR 본문에 쓰지 마세요.

---

## 3. llama.cpp `rope-ops.c`에서 가져올 것 / 가져오지 말 것

| llama.cpp의 것 | 우리 | 이유 |
| :-- | :-- | :-- |
| `hvx_rope_neox_f32_aa`의 **산술 구조** (qf32 mpy 4개 -> sub/add 2개) | **가져온다** | pairing이 우리와 동일. 아래 §4가 이걸 우리 계약으로 다시 쓴 것 |
| `hvx-base.h`의 `hvx_vec_f32_to_f16(_shuff)` / `hvx_vec_f16_to_f32` | **가져온다** | K/V cache가 fp16. `Q6_Vhf_equals_Wqf32` / `Q6_Wsf_vmpy_VhfVhf` 조합은 우리가 다시 찾을 필요 없는 것 |
| `hvx-sin-cos.h` (Chebyshev sin/cos) + `hvx-floor.h` | **Stage 2에서만** | Stage 1은 cos/sin을 host가 준다 |
| spad 더블버퍼 + row 프리페치 루프 | **가져오지 않는다** | 우리에겐 `hexkl_dma_ring`이 있고 device-verified. rung 2 |
| theta cache를 `[cos,sin,cos,sin,...]` 인터리브로 두고 `Q6_W_vdeal_VVR`로 가르는 것 | **가져오지 않는다** | 우리 cos/sin은 **이미 분리된 배열**(§1.3). deal이 통째로 사라짐 -- 벡터당 명령 2개 절약 |
| `rope_basic_f32` (NORMAL mode, 인접 쌍) | **가져오지 않는다** | nntrainer에 그 모드가 없음. YAGNI |
| mrope / imrope / vision, `sections[4]` | **가져오지 않는다** | CausalLM 경로에 vision rope 없음. YAGNI |
| `rope_yarn_one`, `corr_dims`, `ext_factor`, `beta_fast/slow` | **가져오지 않는다** | §1.3(b): 전부 ARM의 `thetas`에 이미 접혀 있음 |
| `fastdiv` 4D (ne0..ne3) 인덱싱 | **가져오지 않는다** | 우리 텐서는 row-major 2D (`n_rows` x `width`) |

한 줄 요약: **가져오는 건 안쪽 루프 6줄과 fp16 변환 헬퍼 2개뿐이고, 나머지는
우리 트리에 이미 더 나은 게 있거나 우리 모델에 해당 사항이 없습니다.**

---

## 4. 커널 설계

`nntrainer/tensor/htp_backend/hvx/hvx_rope_f32.{c,h}` (신규, 2개 파일).
`hvx_softmax_f32.h`의 row-range 시그니처를 그대로 따릅니다 -- 그래야
`hvx_worker_pool`이 스레드마다 row 구간을 던지는 게 지금 코드와 같은 모양이
됩니다.

```c
/* out may alias in.  cos_/sin_ are [m_last][half] -- HALF only, see §1.3(a).
   width % dim == 0, dim % 2 == 0, half = dim/2. */
void hvx_rope_neox_rows_f32(const float *in, float *out,
                            uint32_t m_first, uint32_t m_last,
                            uint32_t width, uint32_t dim,
                            const float *cos_half, const float *sin_half);

/* Same rotation, fp16 (uint16 container) output. This is the K path:
   rope and the f32->fp16 conversion in one pass, matching mha_core.cpp:725. */
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

주의할 점 5가지 -- 전부 이 트리에 선례가 있습니다:

1. **정렬 없음.** FastRPC 버퍼도 KV cache도 벡터 정렬 보장이 없습니다.
   `HVX_UVector`를 쓰세요 (`hvx_softmax_f32.c:34`가 같은 이유로 그렇게 합니다).
2. **tail (`half % 32 != 0`).** `half=64`(head_dim 128)나 `half=32`은 딱
   떨어지지만 head_dim 80/96 모델이 있습니다. `hvx_softmax_util.h`의
   `load_tail_sf` / `store_tail_sf`를 **재사용**하세요. 새로 쓰지 마세요.
3. **`out == in` aliasing이 정상 경로입니다** (Q는 in-place, `:747`).
   `vy[v]`를 쓰기 전에 `a`, `b`를 둘 다 읽어 두면 안전합니다 -- 위 루프가 이미
   그렇습니다. 헤더에 명시하세요.
4. **`dim < width`인 partial rotary.** `dim` 뒤 나머지 채널은 그대로 복사.
   in-place면 복사 자체가 no-op이니 `in != out`일 때만 하세요.
5. **fp16 변환의 반올림.** ARM 쪽은 `vcvt_f16_f32`(RNE), DSP 쪽은
   `Q6_Vhf_equals_Wqf32`입니다. 같은 RNE로 알려져 있지만 **가정하지 말고
   Stage 1 게이트에서 exact-match 개수를 세서 찍으세요.** 어긋나면 그건
   커널 버그가 아니라 계약 차이이고, 그 사실을 아는 채로 tolerance를
   정해야 합니다.

`hvx_worker_pool`은 Stage 1에서는 쓰지 마세요. row가 독립이라 나중에 붙이는 게
한 줄이고, PHASE B가 그랬듯 필요해진 다음에 붙이는 편이 낫습니다.

---

## 5. 단계

### Stage 1 -- 커널 + 정확도 harness (게이트용, 프로덕션 아님)

cos/sin은 **host가 계산해서 보냅니다.** ARM의 `freqs_fp32->cos[pos]`를 그대로
씁니다. 즉 DSP는 회전만 합니다.

IDL 추가 (`test/htp/nntr_hvx.idl`):

```
// Accuracy/perf harness for the HVX RoPE kernel. NOT a production path:
// a per-layer rope round trip costs ~404 us of FastRPC against a few
// microseconds of arithmetic (doc 40 §0). The production path is the
// fusion in attn_kv_append / attn_forward.
AEEResult rope_f32(in uint32 n_rows, in uint32 width, in uint32 dim,
                   in sequence<float> x,
                   in sequence<float> cos_half, in sequence<float> sin_half,
                   rout sequence<float> y);

// Same rotation, fp16 out -- the K path's shape (mha_core.cpp:725).
AEEResult rope_f32_to_f16(in uint32 n_rows, in uint32 width, in uint32 dim,
                          in sequence<float> x,
                          in sequence<float> cos_half,
                          in sequence<float> sin_half,
                          rout sequence<uint16> y);
```

skel: `test/htp/nntr_hvx_rope.c` (`nntr_hvx_softmax.c`를 그대로 본뜸 --
`nntr_hvx_session *`를 확인하고, 길이를 검증하고, 커널을 부르고 끝).

**게이트.** `unittest_hvx_rope`가 device에서:
- `y`가 `nntrainer::compute_rotary_emb_value`의 결과와 **max abs err <= 1e-6**
  (입력 cos/sin이 동일하므로 이 정도가 나와야 정상입니다. 1e-3 같은 게
  나오면 그건 tolerance를 올릴 일이 아니라 pairing이나 tail이 틀린 겁니다)
- fp16 경로: ARM `compute_rotary_emb_value(..., out_uint16, ...)` 결과와
  **bit-exact 비율을 세서 출력**. 100%가 아니어도 실패는 아니지만 숫자는 찍습니다
- shape 매트릭스: `dim` in {64, 128}, `width/dim` in {1, 8, 16},
  `n_rows` in {1, 7, 32, 1024}, `half % 32 != 0`을 만드는 `dim=96` 한 줄 포함
- `y == x` (in-place) 결과가 별도 버퍼 결과와 **bitwise 동일**

### Stage 2 -- cos/sin을 DSP에서 (transport 제거, 선택)

prefill 1024에서 cos/sin 테이블은 `1024 x 64 x 2 x 4 B = 512 KB`/call입니다.
없앨 수 있습니다. §1.3(b) 덕분에 필요한 건 `thetas[half]`와
`attention_scaling` 뿐입니다:

```
AEEResult rope_register(in sequence<float> thetas, in float attention_scaling,
                        rout uint32 h);
AEEResult rope_release(in uint32 h);
```

DSP는 position `p`에 대해 `cos(p * thetas[i]) * attention_scaling`을
llama.cpp `hvx-sin-cos.h`의 Chebyshev로 계산합니다.

**여기엔 실제 정확도 위험이 있습니다.** `p * thetas[i]`는 `p`가 40,960까지
가고 `thetas[0] = 1.0`이므로 인자가 최대 ~4x10^4 rad입니다. f32에서 그 크기의
ulp는 ~0.004 rad이라 `n = floor(x/pi + 0.5); y = x - n*pi` 인자 축소가 f32로는
정확하지 않습니다. **ARM `std::cos`도 같은 f32 인자를 받지만 축소는 내부에서
더 정확하게 합니다.** 그래서 Stage 2는 Stage 1과 자동으로 어긋납니다.

그러니 Stage 2의 게이트는 "Stage 1과 같다"가 아니라 **"position별 max abs
err를 p in {0, 1, 127, 1023, 4095, 32767}에서 표로 찍고, 커지는 게 보이면
그 지점을 명시한다"** 입니다. 그 표 없이 Stage 2를 프로덕션에 넣지 마세요.
표가 나쁘면 Stage 2는 버리고 Stage 1의 host 테이블을 유지하면 됩니다 --
Stage 3은 Stage 2에 의존하지 않습니다.

`ponytail:` 주석으로 천장을 적어 두세요: "host table, 512 KB/call at
prefill 1024; upgrade path = rope_register + on-DSP sin/cos, blocked on the
large-argument reduction error table in doc 40 §5".

### Stage 3 -- 융합 (여기가 전부)

**K 경로.** `hexkl_kvq_pack_kt_block`(`hexkl_kv_quant.c:129~`)은 이미
`kvq_fp16_to_fp32`로 원소마다 f32를 만들어 놓고 quantize합니다. 회전을
**그 f32가 살아 있는 동안** 끼워 넣으면 추가 메모리 트래픽이 **0**입니다.
쌍의 짝(`d`와 `d + half`)이 같은 row 안에 있으므로 접근 패턴도 안 바뀝니다.
position은 `kv_from + r`로 이미 알고 있습니다.

`attn_kv_append`에 `in uint32 rope_h`(0이면 rope 안 함)만 추가하면 IDL
변경은 그것으로 끝입니다. K rows는 **여전히 fp16**으로 건너오고 -- pre-RoPE
값일 뿐 -- transport 바이트는 그대로입니다.

**Q 경로.** `attn_forward`가 받는 `q`는 이미 f32이고, `hvx_quant_pack_u8_ah`가
DDR에서 읽어 VTCM에 u8 AH tile로 씁니다 (doc 31 §4 항목 6). 그 읽기와 쓰기
사이에 회전을 넣습니다. 역시 추가 트래픽 0. `attn_forward`에 `in uint32
rope_h`를 추가하고, position은 `kv_from + i`로 이미 계산 가능합니다.

**V는 건드리지 않습니다** -- rope가 없고 변환만 하는데, 그 변환은 이미
kv_append 안에 있습니다.

**여기서 반드시 결정해야 하는 seam 하나.** ARM쪽 KV cache
(`b_cache_key_step`)는 지금 **post-RoPE** fp16을 담고 있고, CPU attention
fallback이 그걸 씁니다. DSP가 rope를 하면 shadow cache는 pre-RoPE가 됩니다.
둘 중 하나를 고르고 그 이유를 코드에 적으세요:

- (a) shadow를 pre-RoPE로 바꾸고 **CPU fallback 경로를 같이 고친다** --
  DSP 경로가 유일한 정답이 되고 A/B 비교가 불가능해짐
- (b) ARM rope를 CPU cache용으로 남겨 둔다 -- 이득이 사라짐. 임시로만 허용
- (c) **권장:** shadow는 pre-RoPE로 두고, CPU 비교가 필요할 때만 ARM이
  자기 사본에 rope를 적용한다 (테스트 전용 경로). A/B가 살아 있고 프로덕션
  경로엔 중복 작업이 없음

**게이트.** `unittest_hvx_attn`을 확장해서, 같은 pre-RoPE K/Q에 대해
`attn_forward(rope_h != 0)`의 결과가 ARM이 rope를 적용한 뒤
`attn_forward(rope_h == 0)`을 부른 결과와 일치하는지 봅니다. 허용오차는 doc 30
§4의 `(w_k, w_v)` 표를 **그대로** 씁니다 -- rope는 quantize 앞단이므로 오차
예산은 이미 그 표가 정의한 것입니다.

---

## 6. 파일 목록

| 파일 | 상태 | Stage |
| :-- | :-- | :-- |
| `nntrainer/tensor/htp_backend/hvx/hvx_rope_f32.h/.c` | 신규 | 1 |
| `test/htp/nntr_hvx_rope.c` | 신규 | 1 |
| `test/htp/nntr_hvx.idl` | `rope_f32`, `rope_f32_to_f16` 추가 | 1 |
| `test/htp/build.sh` | `$SRCS`에 위 2개 `.c` 추가 | 1 |
| `test/unittest/unittest_hvx_rope.cpp` | 신규 (device gtest) | 1 |
| `test/jni/Android.mk` | `unittest_hvx_rope` 모듈 추가 (`unittest_hvx_add` 블록 복사) | 1 |
| `test/htp/run_rope_on_device.sh` | 신규 (또는 기존 스크립트에 4e/4 단계 추가) | 1 |
| `nntrainer/tensor/htp_backend/hvx/hvx_sin_cos_f32.h` | 신규 (llama.cpp 이식) | 2 |
| `hexkl_kv_quant.c`, `hexkl_attn_u8.c`, IDL의 `attn_*` | 수정 | 3 |
| `Applications/CausalLM/layers/mha_core.cpp` | rope 호출 조건부화 | 3 |

`.c`이지 `.cpp`가 아닙니다. skel은 `hexagon-clang`으로 **C**로 빌드되고
libnntrainer도 C++ 런타임도 링크되지 않습니다 (`00_START_HERE.md` §5의
defect 1이 정확히 그 실수였습니다). `nntrainer::` 이름을 하나라도 부르면
DSP에 못 들어갑니다.

---

## 7. 안드로이드 기기에서 빌드하고 e2e 검증하기

디바이스: Galaxy S25 Ultra (`R3CY10WM83Y`), V79. 아래는 전부 이 트리에서 이미
동작이 확인된 절차입니다 (`13_htp_pr_plan.md` §2/§3a,
`test/htp/run_u8i4_layer_on_device.sh`).

### 7.0 환경 (한 번)

```bash
export HEXAGON_SDK_ROOT=~/workspace/Hexagon_SDK/6.4.0.2
export DEFAULT_HEXAGON_TOOLS_ROOT=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.04
export HEXKL_ROOT=~/workspace/hxkl-beta2/hexkl_addon   # beta2! addons/ 밑의 건 beta1
export HEXKL_SDK_VER=6.4.0.2
export ANDROID_NDK=~/workspace/android-ndk-r26d
git submodule sync && git submodule update --init --depth 1
```

`setup_sdk_env.source`는 이 머신에서 "missed components"로 실패합니다. 고치지
말고 위 두 변수를 손으로 넣으세요 (§3a).

### 7.1 호스트 먼저 -- DSP 없이 잡히는 버그를 여기서 잡는다

Stage 1의 산술은 스칼라로 먼저 확인할 수 있습니다. `mha_htp_host_model.cpp`
옆에 스칼라 rope 레퍼런스를 두고 meson 테스트로 돌리세요:

```bash
meson build -Denable-transformer=true
ninja -C build
cd build && meson test unittest_mha_htp_host_model --print-errorlogs
```

`build`를 쓰세요. 체크인된 `builddir`는 안드로이드 크로스 빌드용이라 호스트
테스트가 안 돕니다.

### 7.2 DSP skel

```bash
cd test/htp && bash build.sh
# -> test/htp/build/libnntr_hvx_skel.so  (v79, hexkl 6.4.0.2), -Werror에서 warning 0
```

`qaic`가 IDL을 `unexpected "o" / expecting "in", "rout" or "inrout"`로 거부하면
스칼라 out 파라미터에 `out`을 쓴 것입니다. 이 qaic 버전은 스칼라에도 `rout`만
받습니다 (§3a에 이미 물린 적 있는 함정).

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

`NNTRAINER_ROOT`는 **반드시 명시**하세요. 이 개발 머신들의 프로필이 다른
체크아웃을 가리키는 `NNTRAINER_ROOT`를 export 하고 있고, Android.mk의
`ifndef` 기본값이 조용히 거기에 집니다 (§3b에서 실제로 물림).
`subprojects/iniparser`가 빈 wrap-git placeholder면 `git submodule update
--init subprojects/iniparser`.

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

`ADSP_LIBRARY_PATH`가 skel이 있는 디렉터리를 가리켜야 DSP가 로드합니다.
unsigned PD는 테스트 안에서 `remote_session_control`로 켭니다
(`unittest_hvx_softmax.cpp`의 fixture를 그대로 복사하세요).

**스크립트로 만드세요.** `test/htp/run_u8i4_layer_on_device.sh`가 skel 빌드 ->
libnntrainer -> ndk-build -> push -> run -> 요약까지 전부 하는 검증된 형태이고,
`unittest_hvx_rope`를 그 목록에 한 줄 추가하는 게 새 스크립트를 쓰는 것보다
짧습니다.

### 7.5 e2e -- 모델 레벨에서 무엇을 비교하나

여기까지는 커널 테스트입니다. Stage 3이 끝난 뒤에야 e2e가 성립하고, 비교
대상은 이것입니다:

1. **같은 프롬프트, 같은 seed, greedy decoding**으로 CPU 전용 실행과 HTP 실행의
   **생성 토큰 열이 같은지.** 이게 유일한 진짜 e2e 판정입니다.
2. 다르면 어디서 갈렸는지: `attn_scores_debug`(IDL에 이미 있음)로 layer 0의
   S band를 뽑아 ARM 쪽 `compute_kcaches_fp32_reference`와 비교. rope가
   틀렸으면 S가 **첫 토큰부터** 틀리고, position이 틀렸으면 **position이
   커질수록** 벌어집니다. 두 패턴을 구분해서 보고하세요.
3. prefill 512 / 1024에서 layer당 us를 찍어 doc 34/35 표에 RoPE 행을 추가.
   §1.4의 "먼저 측정"이 여기서 before, 이게 after입니다.

명령 예 (CausalLM 앱 기준 -- 실제 실행 인자는 앱의 README를 따르세요):

```bash
adb shell "cd $D && LD_LIBRARY_PATH=$D ADSP_LIBRARY_PATH=$D \
  ./nntrainer_causallm --model qwen3-0.6b --prompt-file p.txt --greedy --max-new 64" \
  | tee /tmp/rope_htp.log
# CPU 기준선은 같은 바이너리에서 HTP를 끄고 한 번 더
diff <(grep '^TOKEN ' /tmp/rope_cpu.log) <(grep '^TOKEN ' /tmp/rope_htp.log)
```

---

## 8. Acceptance

| 항목 | 기준 |
| :-- | :-- |
| Stage 1 f32 | vs `compute_rotary_emb_value`, max abs err **<= 1e-6** |
| Stage 1 fp16 | bit-exact 비율을 **출력** (실패 기준 아님), max abs err <= 1e-3 |
| Stage 1 in-place | `y==x` 결과가 별도 버퍼 결과와 **bitwise 동일** |
| Stage 1 shape 매트릭스 | `dim` {64, 96, 128} x `width/dim` {1, 8, 16} x `n_rows` {1, 7, 32, 1024} |
| Stage 2 | position {0, 1, 127, 1023, 4095, 32767}별 max abs err **표를 출력**. 표 없이 머지 금지 |
| Stage 3 정확도 | doc 30 §4의 `(w_k, w_v)` 허용오차 표를 그대로 |
| Stage 3 e2e | greedy 토큰 열이 CPU와 동일 |
| 성능 | before/after를 `ROPE_FIELD path=... field=us value=...`로 출력. **threshold assert 금지** (thermal 때문에 flaky -- `13_htp_pr_plan.md` T6와 같은 이유) |

허용오차를 올려서 통과시키지 마세요. 어긋나면 그건 보고할 finding입니다.
1e-3쯤의 f32 오차는 tolerance 문제가 아니라 pairing/tail/index 버그입니다.

---

## 9. 하지 말 것

1. **rope를 독립 FastRPC op으로 프로덕션에 연결하지 말 것** (§0).
   harness entry는 harness라고 IDL 주석에 쓸 것.
2. `nntrainer/tensor/htp_backend/` 아래 기존 파일은 device-verified입니다.
   Stage 3에서 `hexkl_kv_quant.c` / `hexkl_attn_u8.c`를 건드릴 때는
   **그 변경만으로 device 재검증**을 돌리세요. 다른 이유로 같이 고치지 마세요.
3. yarn / mrope / vision / NORMAL-mode rope를 포팅하지 말 것 (§3).
4. 새 파일은 `.c`. `nntrainer::` 심볼 금지 (§6).
5. `hvx_softmax_util.h`의 tail/reduce 헬퍼를 다시 쓰지 말 것 -- 재사용.
6. "inspection으로 확인함"을 검증이라고 보고하지 말 것. 디바이스에서 못
   돌렸으면 못 돌렸다고 쓰세요.
7. 커밋: `git commit -s`, `Co-authored-by:` 트레일러,
   `[<component>] <subject>`, 바뀐 줄에 `clang-format-14`.

---

## 10. 작업 순서 요약

```
0.  ARM RoPE 실측 (prefill 1024 / decode 1)          <- 코드보다 먼저
1.  hvx_rope_f32.{c,h} + host 스칼라 레퍼런스 + meson 테스트
2.  IDL 2개 + nntr_hvx_rope.c + build.sh
3.  unittest_hvx_rope.cpp + Android.mk + device run   <- Stage 1 게이트
4.  (선택) rope_register + on-DSP sin/cos + 오차 표    <- Stage 2
5.  attn_kv_append(K) 융합 + device 재검증
6.  attn_forward(Q) 융합 + device 재검증               <- Stage 3
7.  mha_core.cpp seam (§5의 (c)) + e2e 토큰 열 비교
8.  doc 34/35 표에 RoPE 행 추가 (0번의 before, 7번의 after)
```

3번까지가 한 PR, 5~7번이 다음 PR입니다. 4번은 게이트 표가 나쁘면 버립니다.
