# PE-Lang-L14-448 encoder 지원 계획 (ScreenAI caption v3.0.0-C62)

> 대상: 구현 에이전트. 이 문서만 읽고 착수할 수 있도록 검증된 사실 / 파일 경로 / 수용 기준을 모두 담았다.
> 작성 근거는 전부 **실측·실행 검증**했다 (timm 1.0.22 모델 빌드, 체크포인트 덤프, 디바이스 벤치 3회 평균).

---

## 0. 결정 사항 (사용자 확정)

| 항목 | 결정 |
| :-- | :-- |
| Base 브랜치 | `origin/claude/weight-q8-quantization-hr0r4o-pr4055` (Quick.AI 서브모듈이 pin한 `3d1eb6850`이 있는 브랜치). 새 브랜치 `feature/pelang-l14-448-encoder`. |
| 디코딩 | **1단계 greedy** 로 수치 parity + 성능 비교 완료 → **2단계 beam=3 / no_repeat_ngram=3 / length_penalty=1.0 / max_length=77** 추가 |
| SigLIP2 경로 | **유지**. PE-Lang을 별도 model 클래스로 추가하고 `config.json`의 `architectures` / `encoder_backend` 로 분기 |
| 양자화 | 기존과 동일 **weight Q8_0 / activation FP32** (`model_tensor_type: "Q8_0-FP32"`) — 아래 §3.5의 patch conv 예외 1건 제외 |

---

## 1. 검증된 아키텍처 사실

### 1.1 소스 자산

```
/home/leeseunghui/workspace/v3.0.0-C62/
├── PE-Lang-L14-448.pt        556MB  bf16, 327 tensors, OpenCLIP 네이밍 (transformer.resblocks.*)
├── best/
│   ├── model_config.json     image_size=448, include_cls_token=true, max_len=77, mean=std=0.5
│   ├── encoder_to_decoder.pt weight [512,1024] + bias [512]  (fp32)
│   ├── decoder/model.safetensors  126MB, 114 tensors, 표준 BertLMHeadModel
│   └── tokenizer/tokenizer.json
└── code/pelang_caption_model.py   PELangCaptionModel 정의 (참조용)
```

### 1.2 encoder = timm `vit_pe_lang_large_patch14_448` (EVA 계열)

timm 1.0.22 에서 실제 모델을 빌드해 확인한 정의:

```python
patch_size=14, embed_dim=1024, depth=23, num_heads=16, mlp_ratio=4.0,
attn_type='rope', class_token=True, use_rot_pos_emb=True,
ref_feat_shape=(32,32), rope_grid_offset=1., rope_grid_indexing='xy',
use_pre_transformer_norm=True,      # ln_pre 있음
use_post_transformer_norm=False,    # norm = Identity  ← ln_post 없음
use_fc_norm=False, init_values=0.1, # LayerScale (gamma_1/gamma_2)
norm_layer=LayerNorm(eps=1e-5)
```

forward (검증 완료, `timm.models.eva.Eva.forward_features`):

```
x = Conv2d(3→1024, k=14, s=14, bias=False)(pixels)      # [1,1024,32,32] → [1,1024(N),1024(C)]
x = concat([cls_token, x], dim=1)                        # N: 1024 → 1025
x = x + pos_embed                                        # [1,1025,1024]
x = LayerNorm(eps=1e-5)(x)                               # ln_pre / norm_pre
for i in range(23):
    x = x + gamma_1[i] * Attn(LN(x), rope)               # fused qkv[3072] w/ bias, proj[1024] w/ bias
    x = x + gamma_2[i] * MLP(LN(x))                      # fc1[4096] → GELU(erf) → fc2[1024], 둘 다 bias
# norm = Identity  → 출력이 곧 마지막 블록 출력
features = x                                             # [1,1025,1024]
hidden   = Linear(1024→512)(features)                    # encoder_to_decoder → [1,1025,512]
```

**SigLIP2 대비 차이 (구현에 직접 영향)**

| 항목 | SigLIP2 (현행) | PE-Lang (신규) |
| :-- | :-- | :-- |
| image / patch | 224 / 16 | **448 / 14** |
| tokens | 196 | **1025** (CLS 포함, `include_cls_token=true`) |
| depth / dim / heads | 12 / 768 / 12 | **23 / 1024 / 16** (head_dim 64) |
| FFN | 3072, `tanh_gelu` | **4096, `gelu` (erf)** |
| patch conv bias | 있음 | **없음** (`disable_bias=true`) |
| CLS token | 없음 | **있음** (learned, pos_embed[0] 과 fuse 가능 — §3.2) |
| pre-norm | 없음 | **ln_pre 있음** (patch+pos 직후) |
| post-norm | `post_ln` 있음 | **없음 (Identity)** |
| LayerScale | 없음 | **gamma_1 / gamma_2 (per-layer, [1024])** |
| position | learned abs만 | learned abs **+ 2D axial RoPE** |
| Q/K/V | 분리된 3 FC | fused `in_proj` [3072,1024] → **converter에서 3분할** |
| LN eps | 1e-6 | **1e-5** |
| proj 출력 | 768→256 | **1024→512** |

### 1.3 RoPE — 런타임 삼각함수 불필요 (검증 완료)

`rope.get_embed()` 는 입력과 무관한 **정적 상수 텐서** `[1024, 128]` = `cat(sin, cos)` (각 `[1024, 64]`).
`rope_mixed=False` 이므로 **23개 블록이 같은 테이블을 공유**한다.

적용 규칙 (`apply_rot_embed_cat(..., half=False)` = **interleaved**):

```
# q,k: [B, 16, 1025, 64].  prefix token(=CLS, index 0)에는 적용하지 않는다.
x  = q[:, :, 1:, :]                       # [B,16,1024,64]
rot(x)[..., 2i]   = -x[..., 2i+1]
rot(x)[..., 2i+1] =  x[..., 2i]
out = x * cos + rot(x) * sin              # cos/sin 은 [1024,64], head 축으로 broadcast
```

수식 동치성은 timm 원본과 **maxdiff 0.0** 으로 확인했다.
→ 구현체는 `rope_sin` / `rope_cos` 를 **weight 파일에 그대로 실어서** 정적 테이블로 로드하면 된다.

### 1.4 decoder = BERT-small, 구조 동일 / 치수만 변경

`best/decoder/config.json` 은 기존 `res/bert-decoder/config.json` 과 **키 구성이 완전히 동일**하고 값만 다르다.
safetensors 키도 기존 converter 가 다루는 `bert.encoder.layer.N.{attention,crossattention,intermediate,output}` /
`cls.predictions.*` 그대로다 (114 tensors 확인).

| 항목 | 현행 (hardcode) | 신규 |
| :-- | :-- | :-- |
| hidden_size | 256 | **512** |
| num_attention_heads | 4 | **8** (head_dim 64 동일) |
| intermediate_size | 1024 | **2048** |
| num_hidden_layers | 4 | 4 (동일) |
| vocab / max_pos / type_vocab | 30522 / 512 / 2 | 동일 |
| layer_norm_eps | 1e-12 | 동일 |
| cross-attn K/V 길이 (`BD_ENC_LEN`) | 196 | **1025** |
| tie_word_embeddings | true | true |

→ **새 아키텍처가 아니라 `BertDecoder` 의 하드코딩 상수를 config 주입으로 바꾸는 작업**이다.

### 1.5 전처리

`direct_resize_448` — PIL **BICUBIC** 로 (448,448) 리사이즈 후 `x/255`, `(x-0.5)/0.5`.
현행 `siglip2LoadAndPreprocessImage` 는 PIL **BILINEAR** 고정소수점 재현 구현이다.
**BICUBIC 용 계수/정수 경로를 새로 넣어야 한다** (§3.2, PIL `Resample.c` 의 `bicubic_filter`, `a=-0.5`, support=2.0,
PRECISION_BITS=22 동일). 이 부분은 과거에 1 LSB 차이가 토큰을 뒤집은 전력이 있으니 bit-exact 를 목표로 한다.

---

## 2. 실측 베이스라인 (SM-S938N / SM8750, 3회 평균, 2026-08-18)

```
# nntrainer (Quick.AI native, 현행 SigLIP2 v2.3 Q8_0)
YOLO detect  403.9 ms | SigLIP2 enc  56.5 ms (196 tok) | BERT dec  10.2 ms (10 tok, 992 TPS)
e2e 893.3 ms (모델 로드 포함) | compute-only 470.5 ms | peak RSS 350 MB

# ONNX Runtime (w8a16_w8a32)
YOLO detect  804.7 ms | SigLIP2 enc  57.9 ms          | BERT dec   7.7 ms (8 tok, 1040 TPS)
e2e 1372.6 ms (로드 493 ms 포함) | compute-only 879.6 ms | peak RSS 583 MB
```

재현 명령:
```bash
cd ~/workspace/screenai_onnx_compare
./run_screenai_vtt_quickai.sh <img> 0.25 0.5 "" 3
./onnx_vtt_pipeline.sh w8a16_w8a32 <img> "" 3
```

**현재 우위는 e2e 1.54× (거의 전부 YOLO). encoder 는 사실상 동률(56.5 vs 57.9).**

### 2.1 연산량 변화 — 이 과제의 핵심 리스크

| | MAC / image |
| :-- | --: |
| SigLIP2-B/16-224 (12L, d=768, N=196) | 17.4 G |
| **PE-Lang-L14-448 (23L, d=1024, N=1025)** | **346 G** |

**약 20배.** 현재 실효 처리율(≈308 GMAC/s)을 그대로 유지해도 encoder 1회 ≈ **1.1 초**.
ONNX 도 같은 벽을 만나므로 "더 빨라야 한다"는 요구는 **상대 비교**로는 달성 가능하지만,
encoder 가 동률인 현 상태로는 절대 시간이 e2e를 지배하며 우위 폭이 얇아진다. §4가 이 문제를 다룬다.

내역 (per layer): qkvo 4.30 G / MLP 8.60 G / attention 2.15 G → attention 비중 14%, MLP 57%.

---

## 3. 작업 분해

브랜치: `feature/pelang-l14-448-encoder` (base `origin/claude/weight-q8-quantization-hr0r4o-pr4055`)

### Phase 0 — 준비 (완료됨, 재실행 불필요)

Golden 텐서가 이미 생성되어 있다: **`/home/leeseunghui/workspace/models/pelang_golden/`**

| 파일 | 내용 |
| :-- | :-- |
| `pixel.npy` | `[1,3,448,448]` PyTorch 전처리 결과 (전처리 격리 검증용) |
| `s0_patch_embed.npy` | `[1,1024,1024]` conv 직후 |
| `s1_pos_embed.npy` | `[1,1025,1024]` cls concat + pos add 후 |
| `s2_norm_pre.npy` | `[1,1025,1024]` ln_pre 후 |
| `s3_block0.npy` | `[1,1025,1024]` block 0 후 (rope/LayerScale 검증 핵심) |
| `encoder_features.npy` | `[1,1025,1024]` 최종 encoder 출력 |
| `encoder_hidden.npy` | `[1,1025,512]` enc_to_dec_proj 후 |
| `decoder_init_logits.npy` | `[1,1,30522]` token 101 1-step, argmax=**2048** |
| `rope_embed/​sin/​cos.npy` | `[1024,128]` / `[1024,64]` ×2 |
| `dump_pelang_golden.py` | 재생성 스크립트 (다른 이미지로 재생성 가능) |

입력 이미지: `screenaivttprobe-v2.1.0-s02-pytorch/test_images/synth_screenshot_with_photo.png`
(timm 1.0.22 필요: `pip install timm==1.0.22`. 체크포인트 로드 시 missing/unexpected 모두 0 확인)

### Phase 1 — weight converter

**신규** `Applications/CausalLM/res/pelang-encoder/weight_converter.py`
(기존 `res/siglip2-encoder/weight_converter.py` 의 `save_safetensors` / `save_bin` 헬퍼 재사용)

* 입력: `--encoder_pt PE-Lang-L14-448.pt`, `--proj_pt best/encoder_to_decoder.pt`,
  `--decoder_safetensors best/decoder/model.safetensors`
* `.pt` 는 `torch.load(..., weights_only=False)` → **bf16 → float32** 캐스팅
* timm 의존 없음: `checkpoint_filter_fn` 이 **순수 rename + reshape 임을 325/327 exact-match 로 확인**했다.
  나머지 2개는 `class_embedding (1024,) → (1,1,1024)`, `positional_embedding (1025,1024) → (1,1025,1024)` 뿐.
* 출력 텐서 순서 = nntrainer 그래프의 layer 생성 순서와 **정확히 일치**해야 한다:

```
pe_patch_conv:weight                        # [1024, 3,14,14] OIHW, bias 없음
pe_pos_embed:weight                         # [1,1,1024,1024]  = positional_embedding[1:]
pe_cls_row:weight                           # [1,1,1,1024]     = class_embedding + positional_embedding[0]
pe_rope_sin:weight / pe_rope_cos:weight     # [1,1,1024,64] 각각
pe_ln_pre:gamma, beta
for i in 0..22:
  enc_layer{i}_ln1:gamma, beta
  enc_layer{i}_wq:weight, bias              # in_proj_weight[0:1024]   / in_proj_bias[0:1024]
  enc_layer{i}_wk:weight, bias              # in_proj_weight[1024:2048]
  enc_layer{i}_wv:weight, bias              # in_proj_weight[2048:3072]
  enc_layer{i}_out:weight, bias             # attn.out_proj
  enc_layer{i}_ls1:weight                   # ls_1.gamma  [1024]
  enc_layer{i}_ln2:gamma, beta
  enc_layer{i}_fc1:weight, bias             # mlp.c_fc
  enc_layer{i}_fc2:weight, bias             # mlp.c_proj
  enc_layer{i}_ls2:weight                   # ls_2.gamma
enc_to_dec_proj:weight, bias                # [512,1024] / [512]
```

`--safetensors` 플래그로 safetensors / bin 둘 다 낼 수 있게 유지 (기존 컨버터와 동일 인터페이스).
decoder 는 기존 `res/bert-decoder/weight_converter.py` 의 `collect_decoder` 를
`hidden_size` / `num_heads` / `intermediate_size` 파라미터화해서 재사용한다 (키 구조 동일 — §1.4).

**self-check**: 변환 후 텐서 수 = `4 + 4 + 23*15 + 2` 를 assert (기존 컨버터의 `== 199` assert 패턴 그대로).

### Phase 2 — PE-Lang encoder model

**신규** `Applications/CausalLM/models/pelang/pelang_vision_encoder.{h,cpp}` + `meson.build`
(`models/siglip2/siglip2_vision_encoder.cpp` 를 출발점으로. `Applications/CausalLM/models/meson.build`,
`jni/Android.mk`, `quantize.cpp` 팩토리 등록에 추가)

그래프:

```
input0 [1,3,448,448]
 └ conv2d "pe_patch_conv" k=14 s=14 valid, disable_bias=true, weight_dtype=FP32   ← §3.5 주의
 └ reshape → [1,1024,1024] → permute(1,3,2) → [1,1,1024,1024]
 └ addition( weight "pe_pos_embed" [1,1,1024,1024] )        # = positional_embedding[1:]
 └ concat(axis=2, [ weight "pe_cls_row" [1,1,1,1024] , 위 결과 ]) → [1,1,1025,1024]
 └ layer_normalization "pe_ln_pre" axis=3 eps=1e-5 packed=false
 └ 23 × block
 └ fully_connected "enc_to_dec_proj" unit=512, bias
출력 [1,1025,512]
```

**CLS + pos fuse (검증 완료, maxdiff 0.0)**
`x = cat([cls, patches]) + pos_embed` 이므로 다음 두 항으로 분해된다:
* token 0 = `class_embedding + positional_embedding[0]` — **입력과 무관한 상수** → `pe_cls_row` 로 미리 합산
* token 1..1024 = `patch_out + positional_embedding[1:]` → `pe_pos_embed` 로 addition

→ **순서가 중요**하다. `addition(pe_pos_embed)` 을 patch 토큰 1024개에 **먼저** 적용하고,
그 뒤에 상수 `pe_cls_row` 를 `concat` 으로 앞에 붙인다. 반대로 하면 pos[0]이 두 번 더해진다.
런타임 cls 브로드캐스트 로직은 필요 없다. `concat` 레이어는 nntrainer 코어에 이미 있다
(`nntrainer/layers/concat_layer.h`, type `"concat"`).

block(i):

```
h  = layer_normalization(x, eps=1e-5)                        enc_layer{i}_ln1
q  = fc(h,1024,bias) / k = fc(h,1024,bias) / v = fc(h,1024,bias)
q,k = pe_rope(q), pe_rope(k)                                 ← 신규 커스텀 레이어 (§3.3)
c  = mha_core(q,k,v) num_heads=16 num_heads_kv=16 is_causal=false use_rope=false
o  = fc(c,1024,bias)                                         enc_layer{i}_out
x  = x + ls1 ⊙ o                                             ← LayerScale (§3.4)
h2 = layer_normalization(x, eps=1e-5)                        enc_layer{i}_ln2
h2 = fc(h2,4096,bias) → activation "gelu" → fc(h2,1024,bias)
x  = x + ls2 ⊙ h2
```

`activation="gelu"` 는 코어에 이미 있고 ARM NEON 다항식 경로도 존재한다
(`ACT_GELU`, `gelu_v2`; 커밋 `ab1312fdd` 에서 ARM 이중계산 버그 수정됨). **`tanh_gelu` 를 쓰면 안 된다.**

`setupParameters` 는 `config.json` / `nntr_config.json` 에서 읽는다 —
`hidden_size 1024 / intermediate_size 4096 / num_hidden_layers 23 / num_attention_heads 16 /
layer_norm_eps 1e-5 / image_size 448 / patch_size 14 / num_patches 1024 / include_cls_token true`.
`NUM_PATCHES+1 = 1025` 가 seq_len 이며 `mha_core` 의 `max_timestep` 도 1025.

### Phase 3 — 신규 커스텀 레이어 2종

`Applications/CausalLM/layers/` 에 추가하고 `layers/meson.build` + `jni/Android.mk` 에 등록.
등록은 encoder 의 `registerCustomLayers()` 에서.

**(a) `pe_rope`** — 정적 테이블 RoPE
* 입력 `[1,1,1025,1024]` (= 16 head × 64), weight = `sin[1024,64]`, `cos[1024,64]` (그래프 전체에서 공유)
* prefix 토큰 수 `num_prefix_tokens=1` → row 0 은 **그대로 통과**
* row 1..1024, head h, 페어 `(2i, 2i+1)`:
  `out[2i] = x[2i]*cos[i2] - x[2i+1]*sin[i2]`, `out[2i+1] = x[2i+1]*cos[i2] + x[2i]*sin[i2]`
  (`cos`/`sin` 은 interleaved 로 이미 중복 저장되어 있으므로 index 는 채널과 1:1)
* NEON: `float32x4x2_t` deinterleave(`vld2q_f32`) → fma 2회 → `vst2q_f32`. 64ch=16 lane-pair, 완전 정렬.
* 비용은 전체의 0.05% 미만이지만 **메모리 트래픽**이므로 in-place 로 쓰는 것이 좋다.

**(b) `layer_scale`** — per-channel gamma 곱 + residual
* `out = residual + gamma ⊙ input`, gamma `[1024]`
* 별도 `mul` + `addition` 2노드 대신 1노드로 합치면 1025×1024 버퍼 왕복 1회를 줄인다.
  23층 × 2 = 46회 × 4 MB = **184 MB 의 불필요한 트래픽 제거**.
* NEON: `vfmaq_f32(residual, input, gamma_bcast)` 단순 루프 + 스레딩.

> 두 레이어 모두 **assert 기반 self-check 유닛테스트** 를 남길 것
> (`layers/unittest_*` 패턴 — `unittest_embedding_sidecar_lut.cpp` 참고).
> `pe_rope` 는 `rope_sin/cos.npy` 와 `s3_block0.npy` 로 검증 가능.

### Phase 4 — decoder 일반화

`Applications/CausalLM/models/bert_decoder/bert_decoder.{h,cpp}`

`BD_DIM=256 / BD_NUM_HEADS=4 / BD_INTERMEDIATE_SIZE=1024 / BD_ENC_LEN=196` 하드코딩을
**config 주입으로 교체**한다. 기존 v2.3 경로가 깨지지 않도록 현재 값은 기본값으로 남긴다.

* `nntr_config.json`: `decoder_hidden_size`, `decoder_num_heads`, `decoder_intermediate_size`, `enc_len`
* `prefillCrossCache()` 의 `BD_ENC_LEN` 루프 → 런타임 값
* **주의**: cross K/V 캐시가 196→1025 로 5.2배. 레이어당 `1025 × 512 × 2` 이므로
  fp16 저장 시 4 layer × 2 MB = 8 MB. 할당 경로가 `BD_ENC_LEN` 상수를 컴파일 타임으로 쓰고 있지 않은지 확인.
* Q8_0 제약 `width % 32 == 0`: 512 / 2048 모두 OK.

### Phase 5 — quantize / config

`Applications/CausalLM/quantize.cpp`

* 팩토리에 `PELangVisionEncoder` 등록 (기존 `Siglip2VisionEncoder` / `BertDecoder` 옆)
* `buildEncoderLayerDtypeMap` 를 `architecture` 로 분기하거나 prefix 목록을 파라미터화:
  PE-Lang 은 layer 당 `_wq,_wk,_wv,_out,_fc1,_fc2` (SigLIP2 와 동일) + `enc_to_dec_proj`
* **`_ls1` / `_ls2` / `pe_rope_sin` / `pe_rope_cos` / `pe_cls_pos` 는 반드시 FP32 유지** (dtype map 에 넣지 말 것)
* `buildDecoderLayerDtypeMap` 은 그대로 (레이어 이름 동일)

**⚠️ patch conv 는 Q8_0 불가 (신규 제약)**
Q8_0 은 K 축 블록 32 이므로 `CRS % 32 == 0` 이 필요하다.
SigLIP2: `3×16×16 = 768 = 32×24` ✅ → 기존에 `patch_embed_dtype: "Q8_0"` 로 양자화하고 있었다.
PE-Lang: `3×14×14 = **588** = 32×18 + 12` ❌.
→ `patch_embed_dtype` 는 **FP32 고정**. 크기는 `1024×588×4 = 2.4 MB` 로 무시 가능.
  `buildEncoderLayerDtypeMap` 의 `if (fc_dtype == Q8_0) dtype_map["patch_embed_conv"] = fc_dtype;`
  분기가 PE-Lang 에는 적용되지 않도록 막아야 한다. (안 막으면 조용히 깨지거나 로드 실패)

**신규 res 디렉토리** `Applications/CausalLM/res/pelang-encoder/`
`config.json` / `nntr_config.json` / `generation_config.json` / `weight_converter.py` / `verify_encoder.py`.

`nntr_config.json` 예상값:
```json
{
  "model_type": "ScreenAICaption",
  "encoder_backend": "pelang_l14_448",
  "model_tensor_type": "Q8_0-FP32",
  "fc_layer_dtype": "Q8_0", "embedding_dtype": "Q8_0", "lmhead_dtype": "Q8_0",
  "patch_embed_dtype": "FP32",
  "img_size": 448, "patch_size": 14, "num_patches": 1024, "include_cls_token": true,
  "encoder_hidden_size": 1024, "encoder_num_layers": 23, "encoder_num_heads": 16,
  "encoder_intermediate_size": 4096, "encoder_layer_norm_eps": 1e-5,
  "decoder_hidden_size": 512, "decoder_num_heads": 8, "decoder_intermediate_size": 2048,
  "enc_len": 1025,
  "num_to_generate": 77, "max_seq_len": 96,
  "tokenizer_file": "tokenizer.json"
}
```

예상 산출물 크기 (Q8_0 ≈ 1.0625 byte/param): encoder ≈ **322 MB**, decoder ≈ **28 MB**.
현행 92 MB / 13 MB 대비 3.5배 → §4.1 의 로드 전략이 필수가 된다.

### Phase 6 — Quick.AI 통합

`/home/leeseunghui/workspace/Quick.AI`

* `src/models/screen_ai/screenai_caption.{h,cpp}` — encoder 를 `encoder_backend` 로 분기해 생성
  (`Siglip2VisionEncoder` | `PELangVisionEncoder`). 인터페이스(`encode()` → `vector<float>`)는 동일하게 유지.
* `src/models/screen_ai/screen_ai.cpp` — §4.1 의 **인스턴스 재사용** 수정
* `nntrainer` 서브모듈을 새 브랜치 커밋으로 bump, `git add nntrainer` 커밋
* 모델 디렉토리 `src/res/screen_ai_v30_c62_q80/` 신설
* 빌드/설치:
  ```bash
  cd ~/workspace/Quick.AI
  ./build.sh --platform=android --target=src,api      # 첫 빌드는 --clean 없이 캐시 재사용 주의
  ./install_android.sh
  ```
  > **알려진 함정**: nntrainer 코어(`layer_normalization_layer.cpp` 등)를 건드린 뒤에는
  > 캐시 재사용 빌드가 stale 코어를 링크해 런타임에 deprecated 경고/크래시가 난다.
  > 코어 수정이 있으면 `--clean` 으로 한 번 전체 재빌드할 것.

### Phase 7 — ONNX 비교

`~/workspace/screenai_onnx_compare/onnx_vtt_pipeline.sh` 는 지금 외부에서 받은
`encoder.onnx / decoder_init.onnx / decoder_step.onnx` 를 쓴다. PE-Lang 은 그 산출물이 없으므로
`export_patched.py` 를 확장해 `v3.0.0-C62` 에서 직접 export 해야 한다.

* `torch.onnx.export` 로 encoder(`PELangCaptionModel.encode`) / decoder_init / decoder_step 3분할
* `make_w8a16_w8a32.py` 로 동일 양자화 프리셋 적용 (caption = W8 / activation FP32)
* `CAP_MODEL_SRC_DIR` 를 새 export 디렉토리로 지정해 기존 스크립트 흐름 그대로 사용
* greedy 로 양쪽 조건을 맞춘 뒤 3회 이상 평균 비교

---

## 4. 성능 계획

**목표: PE-Lang 로 바꾼 뒤에도 e2e 에서 ONNX 대비 우위 유지.** 우선순위 순.

### 4.1 [최우선] caption 인스턴스 재사용 — 구조적 win

현재 `screen_ai.cpp:194-204` 의 `captionOne()` 은 **caption 대상마다** `ScreenAICaption` 을 새로 만들고
`initialize()` + `load_weight("")` 를 다시 부른다 (주석: "ScreenAICaption is memory-planned for a single run").

SigLIP2(92 MB) 에서도 e2e 893 ms 중 로드/셋업이 423 ms 였다.
**PE-Lang(322 MB) 에서는 대상 하나 늘 때마다 ~1.5 초의 순수 가중치 재로드가 붙는다.**
`boxes=N` 이면 `(N+1) × (encode + full reload)`.

→ 그래프를 한 번만 build/load 하고 이미지만 바꿔 재실행할 수 있게 고친다.
"single-run memory plan" 제약이 실제로 무엇인지부터 확인할 것 —
`Siglip2VisionEncoder::encode()` 는 이미 `incremental_inference` 를 반복 호출 가능한 형태다.
제약이 decoder KV 캐시 쪽이라면 **encoder 만 분리해 재사용**해도 이득의 대부분을 얻는다.
*이 항목 하나가 다른 모든 커널 최적화보다 크다.*

### 4.2 [높음] attention prefill — N=1025 경로

`layers/mha_core.cpp` 의 `use_gemm_attention` (blocked QK/AV + FP16 softmax, 커밋 1046~1197 라인 부근)
경로가 커밋 `475ecdda8 [CausalLM] Disable the unlinkable gemm_attention fast path everywhere` 에서
전면 비활성화되어 있다. 현재는 `compute_kcaches` 의 row-wise 경로를 탄다.

* N=196 에서는 차이가 작았지만 N=1025 에서는 attention 이 **49.5 GMAC (전체의 14%)** 이고
  QK 결과 버퍼가 head 당 `1025×1025×4B = 4.2 MB` → 캐시 밖. 블로킹 여부가 크게 갈린다.
* 먼저 **프로파일로 attention 실측 비중을 확인**한 뒤, 필요하면 gemm_attention 을 비인과(non-causal)
  prefill 한정으로 되살린다. "unlinkable" 사유(심볼 누락으로 추정)를 먼저 규명할 것.

### 4.3 [높음] Q8_0 GEMM — M=1025 프리필

MLP 가 전체의 57%. `q8_0x4` interleaved + SMMLA 경로는 이미 있고 ORT MLAS 와 동률이다.
M=196 → 1025 로 커지면서 새로 유효해지는 것:

* **activation 양자화(FP32→int8 패킹)의 스레딩.** 커밋 `c33ead71d
  [nntrainer] Thread the activation packing in the interleaved Q8_0 GEMM` 이 `f42d968f0` 로
  **revert 되어 있다.** M=196 에서는 이득이 없어 되돌린 것으로 보이나, M=1025 · K=4096 에서는
  패킹량이 5배라 재평가 가치가 있다. revert 사유를 확인하고 A/B 로 판단할 것.
* M 타일링/L2 블로킹: `1025 × 4096 × 4B = 16.8 MB` 활성화가 L2 를 넘는다. K-패널 재사용 순서 점검.

### 4.4 [중간] elementwise 트래픽 축소

N=1025, C=1024 기준 버퍼 1개 = 4.2 MB. 층당 왕복 횟수가 곧 대역폭이다.

* `layer_scale` 융합 (§3.3b) — 46회 × 4.2 MB 절감
* GELU: `1025×4096 × 23 = 96 M` 원소. 이미 NEON 이지만 **FC2 와 융합**(fused-FFN 경로가 있으면 재사용)하면
  16.8 MB 왕복 23회를 줄인다. 커밋 `ab1312fdd` 가 언급한 "fused-FFN paths" 확인.
* LayerNorm: `1025×1024 × 47 = 49 M` 원소. ARM fp16 빌드에서 generic fp32 경로 강제
  (`cf606a78a`) 가 걸려 있으니 그 경로가 벡터화되어 있는지 확인.

### 4.5 [중간] 메모리

encoder Q8_0 322 MB + decoder 28 MB + YOLO 21 MB + 활성화. 현행 peak 350 MB → **700 MB+ 예상**.
ONNX 는 583 MB 였으므로 여기서 지면 아프다. §4.1 의 인스턴스 재사용이 여기에도 직접 기여한다.
`fsu` (weight streaming) 는 최후 수단 — 지연이 늘어난다.

### 4.6 측정 방법

`repeat_perf_test.py` / `benchmarks/` 의 기존 계측을 쓰고,
`per_caption` metrics 에 **encoder 단계별(patch/attn/mlp) 타이밍**을 임시로 추가해
어디가 지배적인지 먼저 확인한 뒤 최적화 순서를 정할 것. **추측으로 커널부터 만지지 말 것.**

---

## 5. 검증 기준 (단계별 게이트)

각 게이트를 통과하지 못하면 다음 단계로 넘어가지 않는다.

| # | 항목 | 기준 |
| :-- | :-- | :-- |
| G1 | converter | 텐서 수 assert 통과, dtype F32, 이름/순서 = 그래프 순서 |
| G2 | x86 fp32 stage parity | `encodePixels(pixel.npy)` 결과가 각 tap 과 일치 — `s0/s1/s2/s3` **cosine > 0.9999, maxdiff < 1e-3** |
| G3 | x86 fp32 encoder | `encoder_hidden.npy` 대비 **cosine > 0.9999, rel-L2 < 1e-3** |
| G4 | x86 fp32 decoder | `encoder_hidden.npy` 입력 1-step logits **argmax == 2048**, cosine > 0.999 |
| G5 | ARM fp32 | 동일 항목 **cosine > 0.999, rel-L2 < 5e-2** (기존 PR#4007 이 ARM 에서 0.9993 / 3.8e-2 였음) |
| G6 | ARM Q8_0 | caption 문자열이 x86 fp32 golden 과 **의미상 동일**, greedy 토큰열 diff ≤ 1 token |
| G7 | 전처리 | C++ BICUBIC 리사이즈가 PIL `Image.BICUBIC` 과 **픽셀 0 diff** (uint8 기준) |
| G8 | e2e 성능 | `run_screenai_vtt_quickai.sh` 3회 평균 e2e < `onnx_vtt_pipeline.sh w8a16_w8a32` 3회 평균 e2e |
| G9 | peak RSS | ONNX 대비 우위 유지 (현행 350 vs 583 MB) |
| G10 | 회귀 | 기존 SigLIP2 v2.3 경로가 그대로 동작 (`CAP_MODEL_DIR=models/screen_ai_v23_q80` 로 재실행) |
| G11 | 빌드 | x86 meson + Android ndk 둘 다 경고 없이 성공, `install_android.sh` 후 디바이스 실행 |

**2단계 (beam)**: beam=3 / no_repeat_ngram=3 / length_penalty=1.0 / max_length=77 적용 후
`run_caption.py --device cpu` 의 캡션 문자열과 **완전 일치**를 목표로 한다.

---

## 6. 함정 목록

1. **patch conv Q8_0 불가** — `CRS=588`, 32 로 안 나눠떨어진다. `patch_embed_dtype` FP32 고정. (§3.5)
2. **`tanh_gelu` 아님** — PE-Lang 은 erf GELU. SigLIP2 코드를 복사하면 조용히 틀린다.
3. **post-norm 없음** — SigLIP2 의 `post_ln` 을 지우지 않으면 출력이 어긋난다. timm `norm` 은 Identity.
4. **LN eps 1e-5** (SigLIP2 는 1e-6).
5. **patch conv bias 없음** (`disable_bias=true`).
6. **RoPE 는 CLS 토큰(row 0)에 적용하지 않는다.** `num_prefix_tokens=1`.
7. **RoPE 는 interleaved** (`half=False`). LLM 흔한 rotate-half 와 다르다. mha_core 의 기존 `use_rope`
   경로를 그대로 쓰면 안 된다 (`use_rope=false` 로 두고 별도 레이어).
8. **fused qkv 분할 순서는 q, k, v** (`in_proj_weight` 의 0:1024 / 1024:2048 / 2048:3072).
9. **bf16 → fp32 캐스팅**을 컨버터에서 반드시 할 것. 체크포인트 전체가 bf16 이다.
10. **`include_cls_token=true`** 이므로 1024 가 아니라 **1025** 토큰이 decoder cross-attn 으로 간다.
11. **cross K/V 캐시 5.2배** — `BD_ENC_LEN` 이 상수로 박혀 있는 할당 지점 전부 확인.
12. **`_ls1/_ls2/rope/pos` 를 양자화하지 말 것** — dtype map 에 이름이 걸리지 않는지 확인.
13. **nntrainer 코어 수정 후 캐시 빌드 금지** — stale 링크로 런타임에 깨진다.
14. **Quick.AI 는 절대경로로 실행할 것** — `run_screenai_vtt_quickai.sh` 주석대로, `cd` + 상대경로로
    호출하면 캡션이 깨지고 peak RSS 가 2배가 된다 (디바이스 A/B 확인된 기존 이슈).
15. **rope 테이블은 23 블록 공유** (`rope_mixed=False`). 층마다 복사본을 만들면 메모리 낭비.

---

## 7. 산출물

* nntrainer 브랜치 `feature/pelang-l14-448-encoder`
  * `Applications/CausalLM/models/pelang/` (encoder)
  * `Applications/CausalLM/layers/pe_rope.{h,cpp}`, `layer_scale.{h,cpp}` (+ 유닛테스트)
  * `Applications/CausalLM/res/pelang-encoder/` (config, converter, verify)
  * `bert_decoder` 치수 일반화, `quantize.cpp` 확장
* Quick.AI 브랜치: `screenai_caption` backend 분기, 인스턴스 재사용, 서브모듈 bump
* `screenai_onnx_compare`: PE-Lang ONNX export + 비교 결과표
* 성능 리포트: 단계별 타이밍 / ONNX 대비표 / 최적화 전후 A/B
