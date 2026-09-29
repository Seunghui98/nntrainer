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

## 6. 다음 (Phase 1)

메모리 예산이 정해지면 C 범위가 정해진다. Phase 1은 C와 무관하게 CPU 참조부터 시작한다.
1. gemma4에 MoE 블록(router, experts, dense와의 합)을 CPU로 구현하고 per-layer input 0을 허용한다.
2. ppl 기준값 1개와 짧은 출력을 얻는다.

## 10. 측정 기록

### 10.1 Phase 0 (2026-09-29): 호스트 계산만, 기기 미측정

- 조건: 호스트 x86. `hexkl_mm_u8i4_moe_layout`을 `test/htp/host/stub`로 빌드해 실행했다. 크기 계산은 §3.1 식으로 했고, LFM2 53 §3.2·§5.6 값과 일치하는 것을 먼저 확인했다.
- 결과: §3, §4, §5.
- 판정: C ≥ 5면 기존 층 콜 구조 그대로 된다. 새로 만들어야 하는 커널 작업은 GeGLU 에필로그(R1) 하나다. 성능은 flash 대역폭이 지배할 것으로 예상된다(R6, §4.1).
- 다음: 메모리 예산 결정 → Phase 1.
