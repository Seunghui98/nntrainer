# 55. Gemma-4 26B-A4B를 모바일에서 — expert 스트리밍 + MM 연산 HTP (과제 + 기록)

작성 2026-09-29 · 브랜치 `claude/epic-hopper-occf31` (출발점 `claude/eager-keller-f91z9o` @ 7a8e260) · 인계 문서 `54_gemma4_moe_htp_handoff.md`

이 문서는 54의 Phase 0 산출물이다. 계산과 호스트 검사만 했다. **기기에서는 아무것도 돌리지 않았다.** 아래 숫자 가운데 기기에서 잰 것은 LFM2 값(53)을 인용한 것뿐이고, Gemma 값은 전부 산술(기기 미측정)이다.

단위: 메모리는 MiB/GiB(2^20/2^30), flash에서 읽는 바이트와 대역폭은 GB, GB/s(10^9). 한 열 안에서 섞지 않는다.

## 1. 과제

54 §1 그대로다. G1: 물리 메모리 예산 안에서 CPU 참조와 같은 ppl. G2: FFN(expert, dense MLP)과 projection의 행렬곱은 HTP. G3: 53의 최적화를 그대로 가져간다. 단계는 54 §4를 따른다(0 조사·예산 → 1 CPU 참조 → 2 모델 파일 → 3 MoE HTP → 4 projection·dense HTP → 5 튜닝).

## 2. config (54 §2 표)

출처는 두 가지다.
- `config.json`: 사용자가 HF `google/gemma-4-26B-A4B`에서 복사해 붙여 준 것. 이 환경에서는 `huggingface.co`가 네트워크 정책으로 막혀 있어 원본을 직접 받지 못했다.
- 동작 확인용 소스: transformers `main` @ `f5af3202d63d9bb7578a41f0041c9040071e7345`의 `models/gemma4/modeling_gemma4.py`, `configuration_gemma4.py` (config의 `transformers_version`은 `5.5.0.dev0`). 코드는 GitHub에서 받았다.

| 필드 | 값 (`text_config`) | 의미 / 판정 |
|---|---|---|
| `num_hidden_layers` | 30 | 층 30개. `enable_moe_block`이 켜져 있으면 **모든 층이 MoE**다(`Gemma4TextDecoderLayer`는 층 번호로 가르지 않는다) |
| `layer_types` | sliding 5개 + full 1개 반복, full은 5·11·17·23·29층 | sliding 25층, full 5층 |
| `sliding_window` | 1024 | |
| `enable_moe_block` | true | |
| `num_experts` | 128 | 층당 expert 128개 (LFM2는 32) |
| `top_k_experts` | 8 | 토큰당 8개 (LFM2는 4) |
| `moe_intermediate_size` | 704 | **필드 이름이 54 §2의 `expert_intermediate_size`와 다르다.** 이 이름이 맞다 (`Gemma4TextExperts`가 읽는 이름) |
| expert 형상 | gate_up `[128, 1408, 2816]`, down `[128, 2816, 704]` | 3D 파라미터 하나에 expert 128개가 들어 있다. 양자화기에서 expert별로 잘라야 한다 |
| dense MLP 동시 존재 | **있다** (`intermediate_size` 2112) | 모든 층에서 `self.mlp`(dense)와 experts가 **병렬로** 돌고, 각각 norm을 거친 뒤 더한다. `use_double_wide_mlp: false`, `num_kv_shared_layers: 0`이라 2배 폭은 해당 없음 |
| `hidden_size` | 2816 | |
| `num_attention_heads` | 16 | |
| `head_dim` / `num_key_value_heads` (sliding) | 256 / 8 | q 4096, k·v 2048 |
| `global_head_dim` / `num_global_key_value_heads` (full) | 512 / 2 | q 8192, k 1024 |
| `attention_k_eq_v` | true | full 층은 `v_proj`가 없고 V = K(v_norm만 따로, scale 없음). sliding 층은 `v_proj`가 있다 |
| `hidden_activation` | `gelu_pytorch_tanh` | expert와 dense MLP 모두 `act(gate) * up` = **GeGLU** (§5 R1) |
| `hidden_size_per_layer_input` | 0 | per-layer input 없음 |
| `num_kv_shared_layers` | 0 | KV 공유 없음 |
| `final_logit_softcapping` | 30.0 | |
| `vocab_size` / `tie_word_embeddings` | 262144 / true | |
| `rope_parameters` | full: proportional, partial 0.25, θ 1e6 · sliding: default, θ 1e4 | 작은 gemma4와 같은 형식 |
| router | `proj` 2816→128, softmax(fp32) → top-8 → 합 1로 정규화 → `per_expert_scale` 곱 | router 입력은 **attention 뒤 잔차(`residual`)**를 자기 norm·`scale`·`hidden_size^-0.5`로 처리한 값. expert 입력은 따로 `pre_feedforward_layernorm_2`를 거친다 (Phase 1 구현 주의) |

### 2.1 레포 gemma4 구현이 이미 다루는 것 / 못 다루는 것

`Applications/CausalLM/models/gemma4/gemma4_causallm.cpp` (작은 텍스트 모델용)를 읽고 판정했다.

| 항목 | 상태 |
|---|---|
| `final_logit_softcapping`, `global_head_dim`, `num_global_key_value_heads`, `attention_k_eq_v`, sliding/full, tie embedding | 설정을 읽는다 (118~168행) |
| `hidden_size_per_layer_input: 0` | **로드 시 예외.** 141행 `NNTR_THROW_IF(... == 0)`. Phase 1에서 0이면 per-layer input 경로를 빼도록 고쳐야 한다 |
| MoE 블록 (router, experts, dense와의 합, norm 3개 추가) | 없음 (54 §2 참고 그대로) |
| KV cache | 층마다 `max_seq_len` 전체를 FP16으로 잡는다 (220행). sliding 층도 창 1024가 아니라 전체 길이 |

## 3. 크기

### 3.1 계산식 검증

아레나 칸 크기와 청크 배치는 코드에서 읽었다.
- 칸 = `expertStride(K, 2·inter) + expertStride(inter, N)`, `expertStride = K·N/2 + 8·N`을 4 KiB로 올림 (`htp_compute_ops.cpp` 2586·2624행)
- 청크는 최대 256 MiB, 마지막 청크는 남은 칸 수만큼을 64 MiB 단위로 올림 (`takeExpertSlot`, `newChunk`)

같은 식으로 LFM2 값을 다시 만들면 53 §3.2와 **정확히 같다**: 칸 5.293 MiB, 청크당 48칸, C=8/12/16/상주 아레나 960/1408/1920/3776 MiB. cold 바닥도 22 × 24 × 5.55 MB = 2.93 GB → 0.98 s로 53 §5.6과 같다. 아래 Gemma 숫자는 이 식을 그대로 썼다.

### 3.2 Gemma-4 26B-A4B

| 항목 | 값 | 근거 |
|---|---|---|
| MoE 층 수 × 층당 expert | 30 × 128 = 3840개 | config |
| expert 1개 형상 | gate_up K=2816, N=1408 · down K=704, N=2816 | config |
| expert 1개 크기 (int4 WH + 스케일·colsum, 4 KiB 올림) | WH 2,973,696 B + 33,792 B → 칸 **3,010,560 B = 2.871 MiB** | §3.1 식 |
| 256 MiB 청크당 칸 수 | 89 | |
| expert 전체 | 22.84 B 파라미터, 칸 기준 **11.56 GB = 10.77 GiB** | |
| 비 expert 파라미터 | attention 1.110 B + dense MLP 0.535 B + router 0.011 B + embedding 0.738 B = 2.39 B | 층별 형상 합 |
| 전체 / 활성 파라미터 (텍스트) | 25.23 B / 3.82 B | "26B-A4B"와 맞다 (나머지는 vision) |
| 비 expert 가중치 바이트 | attention·dense·router Q4_0 931 MB + embedding Q6_K 606 MB = **1.43 GiB** | `res/gemma4/nntr_config.json`의 dtype(fc Q4_0, embedding Q6_K, tie로 1벌) 가정 |
| KV cache (FP16, 층마다 `max_seq_len`) | 1024: 220 MiB · 2048: 440 MiB · 4096: 880 MiB | §2.1. 이하 표는 2048 기준 |
| 앱 RSS 추정 | 가중치 1.43 + KV 0.43 = **1.86 GiB + 런타임 오버헤드** | **기기 미측정.** LFM2는 실측 873 MiB |
| 상주(C=128) 시 물리 메모리 | 1.86 + 아레나 10.81 = **12.67 GiB** | 모바일에서 불가능. DSP 주소공간(§5 R5)으로도 불가능 |
| projection·dense를 HTP로 보낼 때 int4 WH | 785 MiB | Phase 4. DSP 주소공간을 expert 아레나와 나눠 쓴다 |
| MoE 커널 VTCM 레이아웃 (호스트 실행) | expert: rc=0, **4.45 MiB**, 청크 gate_up 2개·down 3개 (상한 16) | §5 R3 |

## 4. 메모리 예산표 (C별)

C = 층당 캐시 칸 수. 풀은 30 × C칸을 30층이 공유한다. prefill 중 읽을 바이트 = 30 × (128 − C) × 칸 (444토큰 prefill이면 층마다 128개가 전부 닿는다고 본 상한, 53 §5.6 방식). cold 바닥 = 그 바이트 ÷ 3.0 GB/s(53 §5.5의 이 기기 실측 flash 대역폭). 물리 합계 = RSS 추정 1.86 GiB + 아레나. **표 전체가 산술이고 기기 미측정이다.**

| C | 칸 | 아레나 MiB (청크) | 물리 합계 GiB | expert handle | prefill 중 읽을 바이트 | cold 바닥 | 비고 |
|---|---|---|---|---|---|---|---|
| 4 | 120 | 384 (2) | 2.24 | 240 | 11.20 GB | 3.73 s | 풀 120 < 128. 층 콜을 나눠야 함(R2) |
| **5** | 150 | 448 (2) | 2.30 | 300 | 11.11 GB | 3.70 s | 나누지 않는 최소 C. 선읽기 여유 22칸 |
| 8 | 240 | 704 (3) | 2.55 | 480 | 10.84 GB | 3.61 s | |
| 12 | 360 | 1088 (5) | 2.92 | 720 | 10.48 GB | 3.49 s | |
| 16 | 480 | 1408 (6) | 3.24 | 960 | 10.12 GB | 3.37 s | |
| 24 | 720 | 2112 (9) | 3.92 | 1440 | 9.39 GB | 3.13 s | |
| 29 | 870 | 2560 (10) | 4.36 | 1740 | 8.94 GB | 2.98 s | FC handle ~300개면 handle 상한(R4) |
| 32 | 960 | 2816 (11) | 4.61 | 1920 | 8.67 GB | 2.89 s | 상수를 바꾸지 않으면 FC와 함께 못 씀(R4) |
| 64 | 1920 | 5568 (22) | 7.30 | 3840 | 5.78 GB | 1.93 s | DSP 주소공간 초과(R5) |
| 128 (상주) | 3840 | 11072 (44) | 12.67 | 7680 | 0 | 0 | 불가능 |

C의 상한은 다음과 같다.
- DSP 주소공간(R5): expert 아레나만 쓰면 **C ≤ 44**, projection·dense WH 785 MiB까지 같이 두면 **C ≤ 34**.
- handle(R4): FC가 CPU에 남으면 **C ≤ 34**, FC handle이 ~300개면 **C ≤ 29**.

decode 참고치 (히트율은 기기 trace가 있어야 알 수 있어 비워 둔다):
- 토큰 1개가 쓰는 expert = 30층 × 8 = 240개 = 723 MB.
- 미스 1개를 flash에서 읽으면 3.01 MB ÷ 3.0 GB/s ≈ **1.0 ms**.
- 미스율 10%p마다 토큰당 **+24 ms**.

### 4.1 LFM2와 다른 점: "warm"이 사라진다

LFM2의 warm 성능과 0.35 ms 미스(53 §5.3)는 OS page cache가 4.3 GB 모델 파일을 통째로 들고 있어서 나온 값이다. Gemma의 expert 부분은 11.56 GB라, 아레나와 앱을 뺀 나머지 RAM에 page cache로 다 남을 수 없다. 그래서 다음과 같이 **예상한다(산술, 기기 미측정).**
- prefill은 매번 cold 바닥에 가깝다. C ≤ 32 구간에서 444토큰 prefill의 바닥은 2.9~3.7 s다. DSP 계산은 활성 expert MAC이 LFM2와 비슷(토큰당 47.6 M vs 44.0 M)하므로 flash 쪽이 지배할 것이다.
- decode 미스는 page cache가 아니라 flash에서 읽힌다(1.0 ms/미스, LFM2 warm의 약 3배).
- 이 판정은 Phase 3의 첫 기기 실행에서 `NNTR_HTP_PROFILE=2`의 미스 읽기 시간으로 확인한다.

## 5. 위험 R1~R7 판정

| # | 위험 | 판정 | 근거 |
|---|---|---|---|
| R1 | 활성화 함수 | **해당** | `hidden_activation: gelu_pytorch_tanh`, expert와 dense MLP 모두 `act(gate) * up`(GeGLU). MoE 커널 에필로그는 SwiGLU 고정 → tanh-GELU 에필로그를 추가하고 호스트 참조 비교를 붙여야 한다 |
| R2 | 층당 expert가 32보다 많음 | **해당, C ≥ 5면 split 불필요** | 128개, prefill 한 콜이 층의 128칸을 동시에 잡는다. 풀 30C ≥ 128 → C ≥ 5면 split 없이 된다. C ≤ 4일 때만 split이 필요하고, 그때는 `MoeLayerSplitMatchesWhole`(ppl 62.09 → 63.03 원인)부터 닫는다. MoE 경로 코드에 expert 수 고정 상한은 없다(grep) |
| R3 | 형상 제약 | **비해당** | K=2816, inter=704, N=2816, 2·inter=1408 모두 32의 배수. `hexkl_mm_u8i4_moe_layout`을 호스트에서 VTCM 8300 KiB로 돌려 rc=0, 4.45 MiB(LFM2 7.03 MiB보다 작음), 청크 gate_up 2개·down 3개 ≤ 16. dense MLP(inter 2112)는 한 번에는 rc=ENOMEMORY지만 51처럼 inter를 704씩 3조각으로 나누면 expert와 같은 형상이 된다 |
| R4 | handle 수 (최대 2048) | **해당 (C ≥ 30 부근)** | expert handle = 2 × 30C. FC가 CPU에 남으면 C ≤ 34, projection·dense를 HTP로 보내 FC handle이 ~300개가 되면 C ≤ 29. 넘으면 `HEXKL_MM_U8I4_MAX_WEIGHTS`를 올린다(DSP 정적 메모리 ~48 B/칸) |
| R5 | 아레나·DSP 주소공간 | **해당 (상주 불가, C ≤ 34~44)** | 실측 최대 매핑은 3840 MiB(15청크, 46 §41·50 §3.3), 앱 힙 여유는 ~100 MiB. 상주 11072 MiB는 불가능하다. rpcmem 크기 int 한도는 256 MiB 청크라 비해당. 첫 실행은 작은 C로 |
| R6 | cold prefill = flash 바닥 | **해당 (지배적)** | §4: C ≤ 32에서 prefill 중 8.7~11.1 GB를 읽어 바닥이 2.9~3.7 s다. §4.1 때문에 warm 경로도 기대하기 어렵다. C로 줄일 수 있는 폭이 작다(C 5 → 32에서 −0.8 s) |
| R7 | attention 차원 | **해당 (계획대로 CPU 유지)** | full 층 `global_head_dim 512`, sliding `head_dim 256`. attention 본체는 초기 범위 밖이라 CPU. q/k/v/o projection은 형상 제약이 없어 Phase 4 대상 |

### 5.1 R 목록 밖에서 새로 보인 것

| 항목 | 내용 | 언제 |
|---|---|---|
| per-layer input 0 | 레포 gemma4가 0이면 예외를 던진다(§2.1) | Phase 1 |
| router 입력 | expert 입력과 다른 norm 경로(§2 표) | Phase 1 |
| lm_head | 262144 × 2816, Q6_K 606 MB. 토큰마다 738 M MAC을 CPU에서 한다. decode 비용이 클 수 있다 | Phase 1에서 CPU 시간 분해로 확인 |
| KV cache | sliding 층도 `max_seq_len` 전체를 잡아 2048에서 440 MiB | 예산에 포함. 창 크기로 줄이는 건 필요해질 때 |

## 6. 결정과 구현 계획 (2026-09-29, 사용자 합의)

**예산 3.2 GiB → C=16 기본.** §4 표에서 C=16은 물리 3.24 GiB(아레나 1408 MiB, 6청크, expert handle 960), cold prefill 바닥 3.37 s다. C는 환경변수로 바꿀 수 있어야 한다(§6.3).

### 6.1 레포에 이미 있는 것 (Phase 0에서 추가로 확인)

| 있는 것 | 위치 | 판정 |
|---|---|---|
| Gemma4 MoE 양자화 경로 | `quantize_stream.cpp` `Gemma4MoePlan`/`writeGemma4Moe` (fb55f88, PR #4255 포팅) | 텐서 순서가 "아직 없는 그래프"를 가정한 초안이다: expert가 gate/up/down **분리**, `fc_dtype`으로 양자화, router_scale [hidden]을 따로 둔다. HTP 커널은 fused gate_up WH를 원하므로 LFM2 식(`_gate_up`, `_down`, `moe_dtype`)으로 바꾼다. 그래프도 변환기도 아직 없어 순서는 자유롭게 정할 수 있다 |
| Gemma4 변환기 | `res/gemma4/weight_converter.py` | MoE 없음. full 층 `attention_k_eq_v`(v_proj 없음)도 안 다룬다. 확장한다 |
| Gemma4 모델 클래스 | `models/gemma4/gemma4_causallm.cpp` | MoE 없음. `hidden_size_per_layer_input == 0`이면 예외. 확장한다 |
| MoE 층(CPU + HTP + 가상 expert·LRU·선읽기) | `models/lfm2_moe/lfm2_moe_layer.cpp` `Lfm2MoELayer` | 53의 최적화가 전부 이 층과 `htp_compute_ops.cpp`에 있다. **복사하지 않고 이 층을 그대로 쓴다** |
| 활성화 | `props::MoEActivation` → `acti_func` | CPU는 `tanh_gelu`가 이미 있다(`ACT_TANH_GELU`). HTP 에필로그만 SwiGLU 고정 |
| C 손잡이 | `NNTR_MOE_CACHE_EXPERTS` (`expertCacheFromEnv`) | 이미 환경변수. 미설정 = 상주인데 Gemma4는 상주가 불가능하므로 기본값을 모델 설정에서 받도록 한다 |

### 6.2 LFM2 층을 Gemma4가 쓰기 위한 차이 (router만 다르다)

| | LFM2 (`Lfm2MoELayer` 현재) | Gemma4 (`Gemma4TextRouter`) |
|---|---|---|
| router 입력 | expert 입력과 같은 텐서 | **다른 텐서**: `residual`을 norm(scale 없음)·`scale`·`hidden^-0.5` 한 것. expert 입력은 `pre_feedforward_layernorm_2(residual)` |
| 점수 | sigmoid | softmax(fp32) |
| top-k 선택 | sigmoid + `expert_bias` 로 선택, 가중치는 bias 없는 sigmoid | softmax 그대로 top-8 |
| 가중치 정규화 | norm_topk_prob, `routed_scaling_factor` | 합 1로 정규화 후 `per_expert_scale[e]` 곱 |
| [num_experts] 벡터 | `expert_bias` (선택에만 더함) | `per_expert_scale` (가중치에 곱함) |
| expert 계산 | `act(gate)·up → down`, row_weight로 scatter | 같음 (활성화만 GELU) |

설계: `Lfm2MoELayer`에 속성 `router_type = sigmoid_bias | softmax_scale` (기본 `sigmoid_bias`, LFM2 불변)과 **선택적 두 번째 입력**(router 입력)을 추가한다. `[num_experts]` 가중치 슬롯은 그대로 두고 `router_type`에 따라 bias/scale로 읽는다 → 파일 순서가 LFM2와 같아진다(router [hidden,E], [E], expert별 gate_up, down). router의 norm+scale은 층 밖의 `rms_norm` 한 개로 만들고 gamma에 `scale · hidden^-0.5`를 변환기에서 접어 넣는다(Gemma4RMSNorm은 `x·rsqrt(mean+eps)·weight`라 nntrainer `rms_norm`과 식이 같다). `per_expert_scale`은 호스트에서 row_weight에 곱하므로 DSP 커널은 건드리지 않는다.

ponytail: 54 §3의 "공통 부분을 모델 무관한 곳으로 이동"은 하지 않는다. 복사가 없으므로 옮길 이유가 없다. 이름이 `lfm2_moe`인 채로 Gemma4가 쓰는 것이 한계이고, 세 번째 모델이 오면 `moe_layer`로 이름만 바꾸는 무동작 커밋이 업그레이드 경로다.

### 6.3 C 손잡이

- `nntr_config.json`에 `moe_cache_experts` 키(Gemma4 26B의 기본 설정 파일에 16). 모델 클래스가 읽어 층 속성 `cache_experts`로 넘긴다.
- **환경변수 `NNTR_MOE_CACHE_EXPERTS`가 있으면 그것이 이긴다**(지금 `expertCacheFromEnv` 그대로). 키도 변수도 없으면 지금처럼 0(상주) — LFM2 동작 불변.
- 다른 손잡이(`NNTR_MOE_PREFETCH`, `_READERS`, `NNTR_MOE_SPLIT`, `NNTR_HTP_PROFILE`, `NNTR_MOE_TRACE`)는 그대로 쓴다.

### 6.4 단계별 커밋

각 커밋은 `git commit -s`, `[component] 제목`, 한 주제. 코드 이동과 동작 변경을 섞지 않는다.

**Phase 1 — CPU 참조 (PC x86에서 판정)**

| # | 커밋 | 내용 | 검사 |
|---|---|---|---|
| 1a | `[CausalLM/MoE] Router type and optional router input on the MoE layer` | `router_type` 속성, 입력 1개 또는 2개 허용(2개면 `input[1]`이 router 입력), `softmax_scale` 경로: softmax → top-k → 합 1 정규화 → `per_expert_scale` 곱. `buildExpertAssignments`에 분기 하나 | x86 gtest 1개: 작은 E·k에서 스칼라 참조와 배정·가중치 비교. LFM2 기본 경로는 코드가 안 바뀌므로 기존 테스트로 충분 |
| 1b | `[CausalLM/Gemma4] MoE block when enable_moe_block; per-layer input optional` | `setupParameters`: `enable_moe_block`, `num_experts`, `top_k_experts`, `moe_intermediate_size`, `moe_layer_dtype`, `moe_engine`, `moe_htp_layers`, `moe_cache_experts`. `hidden_size_per_layer_input == 0`이면 per-layer input 그래프 전체를 뺀다. 블록: `pre_ffn_norm → createMlp(dense)` 뒤에 `post_ffn_norm_1(mlp)`, `router_norm(post_attention)`, `pre_ffn_norm_2(post_attention)`, `moe({pre_ffn_norm_2, router_norm})`, `post_ffn_norm_2`, `addition` → 기존 `post_ffn_norm` → 잔차 합. `registerCustomLayers`가 `Lfm2MoELayer`를 등록(gemma4 meson에 `lfm2_moe_layer_dep`) | 그래프 이름이 1d의 텐서 순서와 일치하는지: 1d의 dry-run 크기 = 파일 크기 |
| 1c | `[CausalLM/Gemma4] weight_converter: MoE tensors, K==V full layers` | 층마다 `post_feedforward_layernorm_1`, `router.scale · hidden^-0.5`(→ `_router_norm` gamma), `pre_feedforward_layernorm_2`, `router.proj.weight`ᵀ [hidden,E], `router.per_expert_scale` [E], expert e마다 `experts.gate_up_proj[e]`ᵀ [hidden, 2·inter](HF chunk 순서 gate‖up 그대로), `experts.down_proj[e]`ᵀ [inter, hidden], `post_feedforward_layernorm_2`. full 층에서 `attention_k_eq_v`면 `v_proj` 생략. per-layer input 0이면 생략. safetensors 경로는 expert별 `get_slice`로 한 expert씩 읽는다 | 텐서 수·바이트 수를 출력하고 §3의 파라미터 수(25.23 B)와 비교 |
| 1d | `[CausalLM] quantize_stream: Gemma4 MoE in the LFM2 expert layout` | `writeGemma4Moe`를 1b·1c 순서로: `_router_norm` FP32, router FP32, `_per_expert_scale` FP32, expert별 `_gate_up`·`_down`을 `quant.moe_dtype`으로. `--moe_dtype` 도움말 문구 갱신. `moe_layer_dtype`·`moe_cache_experts`를 출력 config에 쓴다 | 호스트 dry-run: 예상 바이트 = FP32 .bin 크기 |

Phase 1 판정(사용자 PC): 변환 → `--fc_dtype Q4_0 --moe_dtype Q4_0 --embd_dtype Q6_K` 양자화 → x86 `nntrainer_causallm`에 `NNTR_PPL=1`로 ppl 1개와 짧은 출력. HF bf16과 같은 프롬프트의 top-1 토큰 비교(`debug_dump_logits`). **기기 CPU 참조는 불가능하다**(expert 상주 11.5 GB). PC 조건: FP32 .bin **약 101 GB**(25.23 B × 4) 디스크, Q4_0 로드 RAM 약 14 GB.

**Phase 2 — HTP용 모델 파일**

같은 양자화기에 `--moe_dtype QS4CX_WH`. 파일 ≈ 11.56 GB(expert) + 1.5 GB ≈ **13.1 GB**. 로더가 읽는지만 본다(기기, C=16). 로드 로그의 `[HTP] arena chunk` 줄로 청크 6개·1408 MiB 확인.

**Phase 3 — MoE를 HTP로 (cached-slim)**

| # | 커밋 | 내용 | 검사 |
|---|---|---|---|
| 3a | `[HTP/MoE] GeGLU epilogue: act flag through the MoE layer call` | IDL `mm_u8i4_moe_layer`/`_timed`에 `in uint32 act`(0 silu, 1 gelu_tanh) → stub·skel 재생성. `hexkl_mm_u8i4_moe_layer_run` 인자, `hvx_dq_swiglu_job.act`, tail 경로 `hvx_swiglu_inplace_f32`도 같은 플래그. HVX 식: gelu_tanh(x) = x·σ(t), t = 2·√(2/π)·(x + 0.044715·x³) — 벡터 곱 3개 뒤 **기존 silu의 σ(det exp + recip) 그대로** 재사용. 호스트 스텁(`hvx_scalar_stubs.c`)과 `moe_layer_host_check.c`에 gelu 참조 비교 추가. 층은 `MoEActivation`(`tanh_gelu`)을 플래그로 바꿔 넘긴다 | `run_host_checks.sh` 통과; 기기 ppl로 최종 판정. ponytail: CPU `tanh_gelu`와 비트 동일은 보장하지 않는다(44의 SwiGLU는 L2 실패 뒤 비트 동일로 갔다). ppl이 어긋나면 `swiglu_det.h`처럼 `gelu_det` 쌍을 만드는 것이 업그레이드 경로 |
| 3b | `[CausalLM/MoE] Expert cache size from nntr_config; the env var overrides` | §6.3 | LFM2: 키 없음 → 동작 불변(ppl 62.0916 재확인은 기기) |
| 3c | 기기 가이드(54 §6.3 형식) | C=16 기본 → C=8 비교. 순서: 로드 확인 → `NNTR_PPL=1` ppl → `NNTR_HTP_PROFILE=2` 분해. **첫 실행은 C=5로**(R5) | 판정: C=16과 C=8의 ppl이 같음(스트리밍 정확성), PC CPU 참조와 같은 수준. 기대치(산술): cold prefill ≈ 3.4 s, 미스 1개 ≈ 1.0 ms, decode는 히트율 미측정이라 예측 안 함 |

**Phase 4 — projection·dense FFN을 HTP로**

- 엔진 키 `attn_proj_engine`, `dense_ffn_engine`(+`_htp_layers`)을 gemma4에 연결(lfm2 340~356행 방식). q/k/v/o: K=2816에서 N=4096/2048/8192/1024, o_proj K=4096·8192 → N=2816. dense FFN은 51 방식으로 MoE 커널에 inter 2112를 704×3 조각으로(§5 R3).
- **예산 주의(산술):** FC WH 785 MiB가 아레나에 들어가면 C=16 기준 물리 3.24 → **약 4.0 GiB**로 예산을 넘는다. CPU Q4_0 복사본이 해제되지 않으면 그렇다. 선택지: (a) C=8로 내리고 FC 켬(2.55 + 0.77 = 3.32 GiB), (b) FC 중 이득이 큰 것만(50 §2: N ≥ 1300), (c) CPU 복사본 해제 구현. Phase 3 측정 뒤 결정.
- DSP 주소공간: 1408 + 785 = 2193 MiB < 3840. handle 960 + ~300 < 2048.

**Phase 5 — 튜닝·정리:** C 스윕(5/8/16), `NNTR_MOE_TRACE` + `tools/moe_expert_cache_sim.py`, 53 형식 결과 문서(56).

### 6.5 열린 질문 (진행 중 확인)

- PC x86 백엔드가 QS4CX(WH 아님)를 CPU에서 돌리는지. 되면 HTP와 같은 양자화의 CPU 참조 ppl을 얻을 수 있다. 안 되면 참조는 Q4_0이고, LFM2처럼 Q4_0↔QS4CX 차이(50.6 vs 62.1)를 감안해 비교한다.
- 101 GB FP32 중간 파일이 PC에 부담이면, 변환기가 bf16을 쓰고 양자화기가 읽는 경로가 다음 손잡이다(지금은 만들지 않는다).

## 10. 측정 기록

### 10.1 Phase 0 (2026-09-29): 호스트 계산만, 기기 미측정

- 조건: 호스트 x86. `hexkl_mm_u8i4_moe_layout`을 `test/htp/host/stub`로 빌드해 실행했다. 크기 계산은 §3.1 식으로 했고, LFM2 53 §3.2·§5.6 값과 일치하는 것을 먼저 확인했다.
- 결과: §3, §4, §5.
- 판정: C ≥ 5면 기존 층 콜 구조 그대로 된다. 새로 만들어야 하는 커널 작업은 GeGLU 에필로그(R1) 하나다. 성능은 flash 대역폭이 지배할 것으로 예상된다(R6, §4.1).
- 다음: 메모리 예산 결정 → Phase 1.

### 10.2 Phase 1 코드 (2026-09-29): 호스트 검사 통과, 기기·실제 체크포인트 미측정

커밋 4개 (`claude/epic-hopper-occf31`): 11fefbb 층 `router_type`·router 입력·`cache_experts` → 1cffe5e Gemma4 MoE 블록·K==V·per-layer input 선택 → 228a026 변환기 → f714503 양자화기.

- 호스트 x86 빌드(`meson build -Denable-transformer=true`, 이 환경에서 처음 빌드): `unittest_causallm_models` **85/85**, `run_host_checks.sh` 3/3.
- 새 테스트 `unittest_causallm_gemma4_moe`(26B config 형상의 tiny 모델): 스케일 0이면 logits가 dense와 같음(softmax_scale 경로), 그리고 **가중치 저장 순서**를 컴파일된 그래프에서 읽어 변환기·양자화기 순서와 대조. 실제 순서는 생성 순서와 달랐다: `post_ffn_norm_1, pre_ffn_norm_2, router_norm, sparse_moe, post_ffn_norm_2` (router_norm이 pre_ffn_norm_2 뒤). 변환기·양자화기를 이 순서로 맞췄다.
- 미확인: 실제 체크포인트 변환(HF 접근 불가), 양자화기 dry-run 바이트 수, ppl. 기대 FP32 `.bin` 크기 = 25,233,141,790 파라미터 × 4 + tied lm head 재기록 = **103,885,357,176 B (103.9 GB)**; 양자화기는 뒤의 embedding 중복을 버린다.
- 다음: 사용자 PC에서 변환 → Q4_0 양자화 → x86 ppl (가이드는 세션 대화 기록).

### 10.3 Phase 2 파일, Phase 3a GeGLU 에필로그 (2026-10-01): PC·호스트 확인, 기기 미측정

- **Phase 2 (사용자 PC):** `weight_converter.py --safetensors`로 변환(소스 96,256 MiB = 25,233,141,790 × 4 B, 산술과 일치) → `nntr_quantize_stream --fc_dtype Q4_0 --moe_dtype QS4CX_WH --embd_dtype Q4_0 --isa ARM` → `nntr_gemma4_q40_arm.bin` **12,336 MiB**(산술 12,339). 양자화기가 `.safetensors`를 직접 읽게 했다(9dd2c76; 헤더 뒤는 .bin 순서). dry-run 바이트 대조 통과 = 변환기·양자화기·그래프의 텐서 순서가 맞는다.
- **Phase 3a 코드:** `act` 플래그(0 silu, 1 gelu_tanh)를 층 `MoEActivation` → `ComputeOps::gemm_qs4cx_moe_layer_fp32(…, gelu)` → IDL `mm_u8i4_moe_layer(…, act, …)` → skel → `hexkl_mm_u8i4_moe_layer_run(…, act, …)` → `hvx_dq_swiglu_job.act`로 끼웠다. HVX `hvx_geglu_det_sf` = x·σ(x(C0 + C1x²))로 기존 det exp·recip 재사용(벡터 곱 3 + 합 1 추가), 호스트 쌍 `geglu_det_one`(스칼라). CPU 경로도 이제 `moe_activation`을 따른다(전에는 속성과 무관하게 SwiGLU였다). **IDL이 바뀌어 stub(`generate_stub.sh`)과 skel(`test/htp/build.sh`)을 다시 만들어야 한다**; 옛 skel이면 첫 콜이 AEE_EBADPARM.
- 호스트 검사: `moe_layer_host_check`에 GeGLU 케이스(참조와 0 불일치), 항등식 x·σ(2y) = 0.5x(1+tanh y) 스윕(상대 1.3e-6, 참조는 double — f32 tanh는 y < −5에서 1+tanh가 상쇄돼 7%까지 틀린다), act=7 거부. 3/3 통과. `htp_compute_ops.cpp`는 IDL에서 생성한 헤더로 문법 검사만(호스트에 SDK 없음).
- 교훈 하나: 처음 참조를 교과서 tanh 형으로 쓰자 u8 재양자화 뒤 2개 원소가 9e-5 어긋났다 — 44의 L2와 같은 메커니즘. 참조는 커널과 같은 식으로, 항등식은 양자화 전에 따로 본다.
- ponytail: ARM NEON GeGLU 없음(CPU expert는 x86 참조 전용). dense FFN(Phase 4)의 HTP 경로는 아직 silu 고정(`invokeMoeLayer` kind=1에 glu=0).

### 10.4 첫 기기 실행 (2026-10-01): 돌아가지만 출력이 깨짐 — CPU 그래프는 HF와 일치

- 기기(S25 Ultra, V79): C=5, `gemma4-26b-a4b-qs4cx-wh`(12,336 MiB). 로드·아레나·prefill·decode 모두 **돌아간다**(skel·stub 재빌드 뒤; 처음엔 옛 skel로 `AEE_EBADPARM`, 그다음 `use_bidirectional_attention: "vision"` 문자열 파싱 예외 → 3fde104로 수정). 토크나이저가 LFM2 것이어서 한 번 깨졌고(`<|startoftext|>`), Gemma 토크나이저·`<bos><|turn>` 템플릿으로 바꾼 뒤에도 **출력은 무의미한 토큰**("SAH TEL X.A RE…"). 긴 프롬프트(약 1000토큰 + 256 생성)에서는 "Creating shared tensor of size bigger than tensor memory"로 죽음(1024 경계 의심, 별도).
- 호스트에서 가른 것: tiny Gemma4-MoE(26B 형상: 128→4 expert, top-2, K==V, global kv 2, per-layer input 0)를 transformers 5.18.0으로 돌린 HF 참조와 nntrainer 그래프가 **FP32에서 일치**(`Gemma4MoeDifferentialTest`, 06862c3). 즉 router·expert·norm 접기·K==V·가중치 파일 순서·변환기·양자화기 순서는 맞다. 깨진 출력의 원인은 **기기 쪽**이다: (a) HTP MoE 커널의 GeGLU 에필로그 또는 Gemma 형상(K 2816, inter 704)에서의 커널 동작, (b) ARM 커널(attention head 512·GQA 8, Q4_0 ARM 레이아웃)의 이 형상 처리, (c) QS4CX_WH 파일 자체. 다음 판정 수단: 기기 gtest `HvxSwigluDet.GegluMatchesScalarBitExact`(GeGLU HVX 비트 일치), PC x86 Q4_0 CPU 실행(ARM/HTP 무관 참조).
- 부수 발견: x86 AVX2 rotary 커널의 fp16 꼬리 저장이 head_dim 8(half 4)에서 8 lane을 써 16폭 KV cache의 마지막 행을 넘친다(tiny fixture는 `max_seq_len` 16으로 회피; 실제 head_dim은 16의 배수라 꼬리를 안 탄다). `NetworkGraph::getTensor`의 `unordered_map::at` 예외는 잡혀서 무해.

### 10.5 잡음의 원인: FP32 중간 파일의 MoE 순서가 양자화기와 다르다 (2026-10-01, 호스트에서 확정, 기기 미측정)

- 조건: PC 호스트만. 기기도 DSP도 쓰지 않았다. 대상은 §10.3이 만든 `res/gemma4_26ba4b/nntr_gemma4_fp32.safetensors`(100,933,907,208 B)와 그것을 양자화한 `q40/nntr_gemma4_q40_arm.bin`(12,935,608,440 B, 기기에 올라가 있는 바로 그 파일).
- 방법: 파일 헤더의 텐서 이름·형상·오프셋을 payload 순서로 뽑아 `quantize_stream.cpp`의 `writeGemma4Moe`가 **위치로** 읽는 순서와 한 항목씩 맞춰 걸었다(8277 항목). 어긋나는 첫 지점과, 어긋난 뒤 각 읽기가 실제로 집어 가는 바이트를 numpy로 확인했다.

| | 파일에 있는 순서 | 양자화기가 읽는 순서 | 결과 |
|---|---|---|---|
| 14번까지 | attention_norm … pre_ffn_norm_2 | 같음 | 일치 (attention·dense MLP·norm 전부 정상) |
| 15 | `sparse_moe:router` [2816,128] | `_router_norm` 2816 | **router 행렬의 앞 22행이 router norm gamma가 된다** (mean 0.0000 std 0.0185). 진짜 gamma = `router.scale · H^-0.5` (mean 0.6041)는 어디에도 안 들어간다 |
| 16 | `sparse_moe:router_scale` [2816] | `_router` 2816×128 | **router 행렬이 22행 밀리고 꼬리 2816 float이 생 `router.scale`**(mean 32.05)로 채워진다 → 라우팅이 전부 다른 값 |
| 17 | `router_per_expert_scale` [128] | `_per_expert_scale` 128 | 우연히 일치(앞의 밀림이 정확히 2816 float이라 여기서 맞아떨어진다) |
| expert e | `expert_gate_e`[2816,704] + `expert_up_e`[2816,704] | `_gate_up` [2816,1408] 한 개 | **융합 행 k의 up 절반 = gate 행 2k+1**(numpy로 비트 일치 확인, up 행 0과는 불일치) → 128 expert × 30층의 gate·up이 모두 섞인다 |
| expert e | `expert_down_e` [704,2816] | `_down` [704,2816] | 일치 |

- 왜 아무것도 안 터졌나: **층당 float 수가 양쪽 모두 814,094,977로 같다.** §10.3이 "dry-run 바이트 대조 통과 = 텐서 순서가 맞는다"고 적은 근거가 바로 이 숫자이고, 이 숫자는 순서를 볼 수 없다. 누적 밀림도 없어서 층 경계마다 다시 맞는다.
- 판정: **잡음의 원인은 기기가 아니라 PC 변환 경로다.** 파일을 쓴 변환기는 레포의 `res/gemma4/weight_converter.py`(228a026)가 아니다 — 그쪽은 expert를 융합 `expert_gate_up_{e}`로, router norm을 `scale·H^-0.5` 접어 router **앞에** 쓴다. 레포 이력·다른 체크아웃에 그 이름(`sparse_moe:router_scale`, `expert_gate_N`)을 쓰는 변환기는 없다(55 §6.1이 "초안은 gate/up 분리, router_scale 따로"라고 적어 둔 그 초안과 일치). HTP MoE 커널(GeGLU·Gemma 형상), ARM attention, QS4CX_WH 파일 포맷은 이 증상에 대해 **용의자가 아니다**. tiny 모델 호스트 테스트가 통과한 이유도 같다: `generate_gemma4_moe_reference.py`가 fixture를 직접 융합 순서로 써서 변환기를 거치지 않는다.
- 고친 것:
  1. `[CausalLM] quantize_stream: one tensor read is one tensor the converter wrote` — `.safetensors` 입력의 헤더를 읽어 두고, 읽는 텐서 하나가 변환기가 쓴 텐서 하나와 바이트 수까지 같은지 dry run에서 본다. 이 파일에 대해 실제로 `tensor 15: layer0_router_norm 11264 B vs layer0_sparse_moe:router 1441792 B`로 거부한다(확인). 바이트 총합이 같아도 순서는 틀릴 수 있다는 것이 이 가드가 막는 유일한 실패다.
  2. FP32 파일 재배치(일회용 스크립트, 레포에 넣지 않음): router scale을 `·H^-0.5` 접어 router 앞으로, expert마다 gate|up을 **열 방향**으로 융합. 8277 텐서·25,233,141,790 float이 `writeGemma4Moe`의 읽기 순서와 한 항목씩 일치하는 것을 쓰기 전에 확인했다.
- 남은 가정 하나: 파일의 `expert_gate_N`이 HF `gate_up_proj`의 **앞** 청크(gate)라는 것. HF 관례와 이름이 그렇지만 체크포인트가 PC에 더 없어 대조할 수 없다. 재양자화 뒤에도 잡음이면 재배치에서 gate/up을 바꿔 다시 양자화하는 것이 한 줄 실험이다.
- 다음 순서: 재양자화 → 기기 push → `NNTR_PPL=1`. 판정 기준은 nll/token이 한 자리, 생성이 영어 문장. 그래도 틀리면 §6.1의 `NNTR_MOE_DIFF=8`(층 입력 통계 + MoE SNR을 한 실행에서) → `NNTR_MOE_SHADOW=1`. 가이드는 `56_gemma4_accuracy_handoff.md` §9.

