# PE-Lang-L14-448 Encoder nntrainer 포팅 — 핸드오프 문서

> 작성일: 2026-08-18 | 브랜치: `feature/pelang-l14-448-encoder`
> 작업자: Sisyphus (GLM 5.2) | 계획서: `PELANG_L14_448_SUPPORT_PLAN.md`

---

## 1. 완료된 작업 (Phase 1-4 코드 전부 작성됨, 빌드 성공, 런타임 블로커 1건)

### Phase 1 (G1 PASS) — Weight Converter

**파일:** `Applications/CausalLM/res/pelang-encoder/weight_converter.py`

- 소스: `v3.0.0-C62/PE-Lang-L14-448.pt` (bf16, 327 tensors, OpenCLIP 네이밍)
- bf16→fp32 변환, timm 의존 없이 순수 rename+reshape
- RoPE sin/cos 테이블은 checkpoint에 없음 → golden dump에서 로드 (`pelang_golden/rope_{sin,cos}.npy`, [1024,64] FP32)
- CLS+pos fuse: `pe_cls_row = class_embedding + positional_embedding[0]` 미리 합산 (§3.2)
- fused in_proj_weight [3072,1024] → q/k/v 3분할 (trap §6.8)
- Decoder는 `res/bert-decoder/weight_converter.py::collect_decoder` 재사용 (§1.4 — 키 구조 동일, 치수만 다름). 소스 safetensors가 bare `bert.`/`cls.` prefix라 `decoder.` prepend 후 위임.
- **산출물:** encoder 423 tensors (1.17 GB fp32), decoder 114 tensors (132 MB fp32)
- G1 검증: 텐서 수 assert 통과, dtype 전부 F32, 이름/순서 = 그래프 생성 순서 일치

**계획서 vs 코드 차이 (보고):**
- 계횡서의 assert 공식 `4+4+23*15+2=355`는 산술 오류. 실제 텐서 수 = `7 + 23*18 + 2 = 423` (헤더 7 + 블록당 18 + 프로젝션 2). 코드 기준으로 assert 423 사용.
- 계획서가 `pe_patch_conv:weight`라고 쓴 것은 nntrainer Conv2D 키(`:filter`)와 불일치. SigLIP2 컨버터 코드 기준으로 `:filter` 사용.

### Phase 3 — 커스텀 레이어 2종

**파일:**
- `Applications/CausalLM/layers/pe_rope.h` / `pe_rope.cpp`
- `Applications/CausalLM/layers/layer_scale.h` / `layer_scale.cpp`

**pe_rope** — 정적 테이블 interleaved RoPE:
- 입력 3개: [0] q/k `[B,1,SEQ,C]`, [1] sin `[1,1,1024,64]`, [2] cos `[1,1,1024,64]`
- CLS 토큰(row 0)은 identity 통과 (trap §6.6)
- patch 행(row 1..1024)은 interleaved 회전: `out[2i] = x[2i]*c - x[2i+1]*s; out[2i+1] = x[2i+1]*c + x[2i]*s` (trap §6.7 — rotate_half 아님)
- 프로퍼티: `num_prefix_tokens`(default 1), `num_heads`(default 16), `head_dim`(default 64)
- 0-weight 레이어 (sin/cos는 input으로 들어옴)
- NEON 최적화는 미룸 (§4.4 프로파일링 후, trap §6.13)

**layer_scale** — fused residual + per-channel scale:
- 입력 2개: [0] input (sub-block 출력), [1] residual (skip connection)
- weight 1개: gamma `[C]` (FP32 고정 — trap §6.12)
- `out = residual + gamma[c] * input[r,c]` — 스칼라 루프 (NEON vfmaq는 Phase 5)
- 프로퍼티: `num_channels`(default 0, optional — 입력 width에서 자동 추론)

**빌드 등록:**
- `layers/meson.build`: shared_library `causallm_pe_rope` / `causallm_layer_scale` + dep 추가
- `Applications/CausalLM/meson.build`: `causallm_layer_dependencies`에 두 dep 추가
- `jni/Android.mk`: 두 .cpp를 메인 lib + quantize lib 양쪽 LOCAL_SRC_FILES에 추가

### Phase 2 — PE-Lang Encoder Model

**파일:** `Applications/CausalLM/models/pelang/pelang_vision_encoder.h` / `.cpp` / `meson.build`

- `PELangVisionEncoder : virtual public Transformer` (SigLIP2와 동일 패턴)
- `ENC_TO_DEC_DIM = 512`, `ENC_NUM_TOKENS = 1025`
- 그래프 순서 (converter와 1:1 매칭):
  1. `conv2d "pe_patch_conv"` k=14 s=14 valid, **disable_bias=true**, weight_dtype=FP32 (trap §6.1/6.5)
  2. reshape → permute → `[1,1,1024,1024]`
  3. `addition(pe_pos_embed)` — patch 토큰에 pos_embed 더함
  4. `concat(axis=2, [pe_cls_row, h])` — CLS를 앞에 붙임 → `[1,1,1025,1024]`
  5. `layer_normalization "pe_ln_pre"` eps=1e-5 (trap §6.4)
  6. `weight "pe_rope_sin"` / `"pe_rope_cos"` — 정적 상수 테이블 (23블록 공유, trap §6.15)
  7. 23× block: ln1→q/k/v FC→pe_rope(q/k)→mha_core(non-causal,use_rope=false)→out FC→layer_scale(ls1,residual)→ln2→fc1→gelu(erf)→fc2→layer_scale(ls2,residual)
  8. `enc_to_dec_proj` FC unit=512
  - **NO post_ln** (trap §6.3 — timm norm=Identity)
  - **gelu NOT tanh_gelu** (trap §6.2)
- `registerCustomLayers()` override: base 호출 후 `pe_rope`/`layer_scale`를 `AppContext`에 registerFactory
- `encode(path)`는 stub (BICUBIC은 G7/Phase 6), `encodePixels()`만 구현 (G2/G3용)
- `setupParameters()`: config.json + nntr_config.json에서 모든 치수 읽음

**등록:**
- `models/meson.build`: `subdir('pelang')` 추가
- `quantize.cpp`: `#include "pelang/pelang_vision_encoder.h"` + factory `registerModel("PELangVisionEncoder", ...)`
- `main.cpp`: `#include` 추가, `--dump-encoder` handler를 `encoder_backend` config로 분기 (`siglip2` | `pelang_l14_448`)
- `jni/Android.mk`: `models/pelang` include path + `pelang_vision_encoder.cpp` src (메인 lib + quantize lib 양쪽)

### Phase 4 — BertDecoder 일반화

**파일:** `Applications/CausalLM/models/bert_decoder/bert_decoder.h` / `.cpp` (수정)

- `BD_DIM`, `BD_NUM_HEADS`, `BD_INTERMEDIATE_SIZE`, `BD_ENC_LEN`을 `static constexpr int` → 비정적 멤버 `int`로 변경 (디폴트값 = v2.3 BERT-small: 256/4/1024/196)
- config-driven ctor 추가: `explicit BertDecoder(const json &nntr_cfg)` — delegating ctor로 v2.3 디폴트 설정 후 nntr_cfg 키로 4개 치수 override
- `bert_decoder.cpp`에서 `196` 리터럴 4곳 → `BD_ENC_LEN`으로 교체 (trap §6.11)
- `encoder_hidden({1, 1, BD_ENC_LEN, static_cast<unsigned int>(BD_DIM)})` — narrowing 방지
- `quantize.cpp` factory: `BertDecoder()` → `BertDecoder(nntr_cfg)` 사용
- `main.cpp --decoder-init-parity`: `nntr_config.json`에서 `enc_len`/`decoder_hidden_size` 읽어 ENC_COUNT 계산 (하드코딩 196*256 제거)
- G10 회귀 안전: 디폴트 ctor가 v2.3 값을 그대로 사용, 기존 SigLIP2 res는 수정 없이 동작

### res 파일

**`res/pelang-encoder/`:**
- `config.json` — encoder 구조 (hidden=1024, layers=23, heads=16, patch=14, image=448, eps=1e-5)
- `nntr_config.json` — `encoder_backend: "pelang_l14_448"`, encoder/decoder 치수, `encoder_model_file_name`
- `generation_config.json` — greedy 설정 (beam=1, max_length=77)
- `nntr_pelang_encoder_fp32.bin` — 1.17 GB encoder 가중치
- `golden.encoder_hidden.npy` — symlink → `pelang_golden/encoder_hidden.npy`

**`res/pelang-decoder/`:**
- `config.json` — decoder 구조 (hidden=512, heads=8, inter=2048, layers=4)
- `nntr_config.json` — `decoder_hidden_size: 512`, `enc_len: 1025`, `model_file_name`
- `generation_config.json`
- `nntr_pelang_decoder_fp32.bin` — 132 MB decoder 가중치
- `golden.encoder_hidden.npy` — 같은 symlink

---

## 2. 현재 블로커 — `model->compile()` 메모리 플래닝 실패

### 증상
```
[PE-Lang DBG] enc_to_dec_proj done, returning
[!] FATAL ERROR (--dump-encoder): Creating shared tensor of size bigger than tensor memory.
```

### 분석
- 그래프 생성(`constructModel()`)은 전부 성공 — 23블록 + enc_to_dec_proj까지 DBG 프린트 모두 출력됨
- 에러 발생 위치: `Transformer::initialize()` 안의 `model->compile(x, y, INFERENCE)` 호출 (transformer.cpp:223)
- 에러 메시지 출처: `nntrainer/tensor/tensor_base.cpp:228` — `getSharedDataTensor()`에서 `dim_.getDataLen() + offset > dim.getDataLen()` 조건
- **0 layers + rope weight layers 제거 테스트**: `free(): invalid size` (heap corruption) → 블록/rope가 아니라 **patch embed 구조 자체**가 원인
- **23 layers + rope weight layers 제거 테스트**: `[TensorDim] Trying to assign value <=0 to tensor dim` → rope_stash가 비어있어 pe_rope에 빈 텐서가 들어가서 차원 에러
- 결론: 원래 에러("shared tensor size bigger")는 rope weight layers와 블록 구조가 **조합될 때** 발생. 0-layer에서 rope를 빼면 다른 에러(heap corruption)로 바뀜 → 근본 원인은 patch embed의 weight/concat 구조가 메모리 플래너와 충돌

### 의심 원인 (최종 정리)

 SigLIP2와 PE-Lang의 patch embed 차이점 중 하나가 플래너를 깨뜨림:
1. **conv2d disable_bias=true** — SigLIP2는 bias 있음. bias 없는 conv의 출력 텐서 할당이 다를 수 있음
2. **concat(axis=2)으로 [1,1,1,1024] + [1,1,1024,1024] 합치기** — SigLIP2는 concat 없이 addition만 사용. concat이 메모리 플래너에서 출력 크기 계산을 잘못할 수 있음
3. **reshape target_shape이 3값** (`"1:1024:1024"`) — SigLIP2도 3값 (`"1:768:196"`)이라 동일. 차이 없음
4. **weight layer 3개** (pos_embed + cls_row + rope_sin/cos) — SigLIP2는 1개(pos_embed). 여러 weight layer의 출력이 그래프에서 서로 다른 차원을 가지면 플래너가 aliasing 시도 시 크기 불일치

### 디버깅 방법 (다음 에이전트용)
1. **0-layer + rope 제거 + concat 제거**: concat을 addition으로 교체 (CLS를 pos_embed에 포함시키거나 brute-force add) → heap corruption 해결 여부
2. **0-layer + rope 제거 + conv bias 추가**: `disable_bias="false"`로 변경 → heap corruption 해결 여부
3. **0-layer + rope 제거 + weight layer 1개만**: pos_embed만 남기고 cls_row 제거 (CLS를 별도 처리)
4. 위 3개 테스트로 원인 좁힌 후, nntrainer 코어 수정 또는 그래프 구조 회피

---

## 3. 남은 작업

### 즉시 (블로커 해결 후)
| 단계 | 내용 | 게이트 |
|------|------|--------|
| 블로커 해결 | `model->compile()` 메모리 플래닝 에러 원인 파악 + 수정 | — |
| G2 | `encodePixels(pixel.npy)` 결과 vs s0/s1/s2/s3 단계별 golden — cos>0.9999, maxdiff<1e-3 | G2 |
| G3 | `encoder_hidden.npy` 대비 cos>0.9999, rel-L2<1e-3 | G3 |
| G4 | `--decoder-init-parity` 1-step argmax==2048, cos>0.999 | G4 |
| **STOP** | G4 통과 후 보고 — ARM/양자화/성능은 Agent B 스코프 | — |

### G4 이후 (Agent B — Phase 5-7)
- Phase 5: quantize.cpp dtype map (patch_conv FP32 고정, ls1/ls2/rope/pos 제외)
- Phase 6: Quick.AI 통합 (encoder_backend 분기, 인스턴스 재사용, 서브모듈 bump)
- Phase 7: ONNX export + 비교 (3회 평균, G8 e2e < ONNX)
- G5: ARM fp32 parity
- G6: ARM Q8_0 caption
- G7: C++ BICUBIC bit-exact (픽셀 0 diff)
- G8-G11: 성능/메모리/회귀/빌드

### Phase 3c (블로커 해결 후 언제든)
- `layers/unittest_pe_rope.cpp` — rope_sin/cos.npy + s3_block0.npy로 검증
- `layers/unittest_layer_scale.cpp` — gamma 곱 + residual 검증
- 패턴: `unittest_embedding_sidecar_lut.cpp` 참고

---

## 4. 주요 결정 사항 & 유의점

### pe_rope 레이어 분리 결정 (유저 지적에 대한 응답)
유저가 "mha_core가 이미 rope 계산+캐싱을 하는데 왘 별도 레이어?"라고 지적. 조사 결과:
- mha_core의 `apply_rotary_emb_tensor_v2`는 **split-half** 방식 (`i0=w+k, i1=w+k+half`)
- PE-Lang은 **interleaved** 방식 (`i0=w+2k, i1=w+2k+1`) — trap §6.7
- PE-Lang의 RoPE 테이블은 **2D axial RoPE** (`ref_feat_shape=(32,32)`)로 timm이 빌드 타임에 생성, theta 단독으로는 재생 불가
- 결론: mha_core 확장(interleaved 분기 + 외부 테이블 주입)은 Qwen3/Gemma 등 다른 모델이 의존하는 shared 코드를 건드림 → 회귀 리스크. 작은 전용 레이어가 더 안전한 diff. **pe_rope 유지 결정.**

### 디버그 프린트
`pelang_vision_encoder.cpp`에 `[PE-Lang DBG]` 프린트가 다수 남아 있음. 블로커 해결 후 전부 제거 필요.

### ponytail:` 주석
코드에 `ponytail:` 마커 주석이 있음 (BICUBIC deferral, NEON deferral 등). 이 intentional — ponytail 모드 규칙에 따른 known-ceiling 표시.

### 코드 리뷰 후보
- `pe_rope.cpp`의 `runRotate` — CLS 행 identity 복사 루프 + interleaved 회전 루프. NEON vld2q 패스는 Phase 5.
- `layer_scale.cpp`의 `runScale` — scalar 루프. NEON vfmaq 패스는 Phase 5.
- `pelang_vision_encoder.cpp`의 `encode(path)` stub — BICUBIC은 Phase 6에서 구현.

---

## 5. 빌드 & 실행 명령

```bash
# x86 fp32 빌드 (이미 성공)
cd ~/workspace/nntrainer
meson setup --reconfigure build -Denable-transformer=true -Denable-fp16=false \
  -Dthread-backend=omp -Denable-app=true -Denable-test=false
ninja -C build Applications/CausalLM/nntr_causallm

# 런타임 (LD_LIBRARY_PATH 필요)
export LD_LIBRARY_PATH=$PWD/build/nntrainer:$PWD/build/Applications/CausalLM/layers:$PWD/build/Applications/CausalLM/models/gpt_oss:$PWD/build/Applications/CausalLM/models/qwen3_moe:$PWD/build/Applications/CausalLM/models/qwen3_slim_moe:$PWD/build/Applications/CausalLM/models/gpt_oss_cached_slim:$PWD/build/Applications/CausalLM/models/qwen3_cached_slim_moe:$LD_LIBRARY_PATH

# G2/G3: encoder parity
./build/Applications/CausalLM/nntr_causallm --dump-encoder \
  Applications/CausalLM/res/pelang-encoder \
  --input-pixels ~/workspace/models/pelang_golden/pixel.npy

# G4: decoder parity
./build/Applications/CausalLM/nntr_causallm --decoder-init-parity \
  Applications/CausalLM/res/pelang-decoder

# Python에서 cos/maxdiff 비교
python3 -c "
import numpy as np
a = np.load('nntr_encoder_hidden.npy').astype(np.float64).ravel()
b = np.load('~/workspace/models/pelang_golden/encoder_hidden.npy').astype(np.float64).ravel()
cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
rel_l2 = float(np.linalg.norm(a - b) / np.linalg.norm(b))
print(f'cos={cos:.8f} rel_l2={rel_l2:.6e} maxdiff={np.abs(a-b).max():.6e}')
"
```

---

## 6. 파일 목록 (전부)

### 신규 파일
| 파일 | 설명 |
|------|------|
| `Applications/CausalLM/res/pelang-encoder/weight_converter.py` | encoder+decoder 가중치 변환 |
| `Applications/CausalLM/res/pelang-encoder/config.json` | encoder 구조 |
| `Applications/CausalLM/res/pelang-encoder/nntr_config.json` | 런타임 config |
| `Applications/CausalLM/res/pelang-encoder/generation_config.json` | 생성 설정 |
| `Applications/CausalLM/res/pelang-encoder/nntr_pelang_encoder_fp32.bin` | 1.17 GB 가중치 |
| `Applications/CausalLM/res/pelang-decoder/config.json` | decoder 구조 |
| `Applications/CausalLM/res/pelang-decoder/nntr_config.json` | 런타임 config |
| `Applications/CausalLM/res/pelang-decoder/generation_config.json` | 생성 설정 |
| `Applications/CausalLM/res/pelang-decoder/nntr_pelang_decoder_fp32.bin` | 132 MB 가중치 |
| `Applications/CausalLM/layers/pe_rope.h` | RoPE 레이어 헤더 |
| `Applications/CausalLM/layers/pe_rope.cpp` | RoPE 레이어 구현 |
| `Applications/CausalLM/layers/layer_scale.h` | LayerScale 레이어 헤더 |
| `Applications/CausalLM/layers/layer_scale.cpp` | LayerScale 레이어 구현 |
| `Applications/CausalLM/models/pelang/pelang_vision_encoder.h` | encoder 모델 헤더 |
| `Applications/CausalLM/models/pelang/pelang_vision_encoder.cpp` | encoder 모델 구현 |
| `Applications/CausalLM/models/pelang/meson.build` | 빌드 설정 |

### 수정 파일
| 파일 | 수정 내용 |
|------|-----------|
| `Applications/CausalLM/models/meson.build` | `subdir('pelang')` 추가 |
| `Applications/CausalLM/layers/meson.build` | pe_rope/layer_scale shared_library + dep 등록 |
| `Applications/CausalLM/meson.build` | causallm_layer_dependencies에 두 dep 추가 |
| `Applications/CausalLM/quantize.cpp` | PELangVisionEncoder include + factory 등록, BertDecoder config ctor 사용 |
| `Applications/CausalLM/main.cpp` | PELangVisionEncoder include, --dump-encoder를 encoder_backend 분기, --decoder-init-parity를 config-driven ENC_COUNT |
| `Applications/CausalLM/jni/Android.mk` | pe_rope/layer_scale/pelang src + include path 추가 (메인 lib + quantize lib 양쪽) |
| `Applications/CausalLM/models/bert_decoder/bert_decoder.h` | BD_DIM/HEADS/INTER/ENC_LEN을 비정적 멤버화, config ctor 선언 |
| `Applications/CausalLM/models/bert_decoder/bert_decoder.cpp` | config ctor 구현, 196 리터럴 → BD_ENC_LEN |

---

## 7. 함정 목록 체크리스트 (§6, 15개)

| # | 함정 | 처리 |
|---|------|------|
| 1 | patch conv Q8_0 불가 (CRS=588) | ✅ PATCH_EMBED_DTYPE="FP32" 강제 |
| 2 | erf GELU not tanh_gelu | ✅ activation="gelu" 사용 |
| 3 | post_ln 없음 | ✅ constructModel에 post_ln 없음 |
| 4 | LN eps 1e-5 | ✅ NORM_EPS = 1e-5f |
| 5 | patch conv bias 없음 | ✅ disable_bias="true" |
| 6 | RoPE CLS 미적용 | ✅ pe_rope의 num_prefix_tokens=1, row 0 identity |
| 7 | RoPE interleaved | ✅ pe_rope에서 interleaved 구현, mha_core use_rope=false |
| 8 | fused qkv 분할 순서 q/k/v | ✅ converter에서 in_proj[0:1024]/[1024:2048]/[2048:3072] |
| 9 | bf16→fp32 | ✅ converter에서 .to(torch.float32) |
| 10 | include_cls_token → 1025 토큰 | ✅ NUM_TOKENS=1025, concat으로 CLS 추가 |
| 11 | cross KV 캐시 5.2배 | ✅ BD_ENC_LEN을 config 주입, 196 리터럴 제거 |
| 12 | ls/rope/pos 양자화 금지 | ⏳ Phase 5 (Agent B) — dtype map에서 제외 필요 |
| 13 | 코어 수정 후 캐시 빌드 금지 | ⏳ Phase 6 주의사항 |
| 14 | Quick.AI 절대경로 실행 | ⏳ Phase 6 주의사항 |
| 15 | rope 테이블 23블록 공유 | ✅ weight layer 1개 생성, rope_stash_sin_/cos_로 23블록에 전달 |
