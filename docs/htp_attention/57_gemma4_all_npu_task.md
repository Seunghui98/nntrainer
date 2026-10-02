# 57 — Gemma-4 26B-A4B: embedding만 빼고 전부 NPU로 (작업 문서)

작성: 2026-10-02. 기준 커밋: `claude/gemma4-accuracy-diff` @ e8d2604.
이 문서만 읽고 작업을 시작할 수 있도록 썼다. 근거가 되는 측정은 55 §10.5–10.15에 있고, 여기서는 그 결과만 옮긴다.

## 0. 한 줄 요약

PR 4385(`htp_first_version`)에는 HTP에서 쓸 수 있는 부품이 이미 들어 있다.

- LFM2.5 MoE를 NPU에서 전부 돌리는 데 쓴 HVX 소형 연산(RMSNorm, RoPE, router 등, 주로 M=1)
- HMX/HVX flash attention: prefill은 `attn_f16_prefill`, decode는 `hvx_attn_decode_f16`
- `sdpa_fp16_kvcache` ComputeOps와 rpcmem KV cache

이 브랜치의 Gemma-4 작업(45커밋)을 PR 4385 위로 옮긴다. 그다음 Gemma-4 그래프에서 아직 CPU에 남은 연산을 위 부품으로 HTP에 올린다. 남기는 것은 embedding뿐이다.
**prefill부터 한다.** 측정된 병목 순서는 §3에 있다. decode는 그다음이다.

## 1. 출발점 — 브랜치 만들기

```bash
cd ~/workspace/nntrainer
git fetch https://github.com/nntrainer/nntrainer.git pull/4385/head:pr4385
git fetch origin claude/gemma4-accuracy-diff
git switch -c <새 작업 브랜치> pr4385
# 이 브랜치의 Gemma-4 커밋(7a8e260 이후 45개)을 옮긴다
git log --oneline --reverse 7a8e2609e..origin/claude/gemma4-accuracy-diff
git cherry-pick 7a8e2609e..origin/claude/gemma4-accuracy-diff
```

- 공통 조상은 `7a8e260`(doc 54)이다. 그 뒤로 PR 4385에는 507커밋, 이 브랜치에는 45커밋이 있다.
- 충돌이 예상되는 파일: `models/gemma4/gemma4_causallm.{h,cpp}`, `models/lfm2_moe/lfm2_moe_layer.cpp`, `models/transformer.cpp`, `htp_backend/htp_compute_ops.cpp`, `quantize_stream.cpp`, `layers/mha_core.cpp`. PR 4385도 이 파일들을 고쳤다.
  - 충돌은 **PR 4385 쪽 구조를 기준으로** 풀고, 이 브랜치의 기능을 그 위에 다시 얹는다.
- `CLAUDE.md`와 `docs/htp_attention/*`는 작업 참고용 문서다. **upstream PR 브랜치에 넣지 않는다.** PR을 올릴 때는 코드 커밋만 골라 따로 브랜치를 만든다.
- cherry-pick을 끝낸 뒤 host 검증을 먼저 통과시킨다(§6.1). 그다음 이 문서의 기준선(§2)을 **새 브랜치에서 다시 잰다.** PR 4385에서 MoE 경로(dspqueue, DMA L2 bypass 등)가 바뀌었으므로 숫자가 달라질 수 있다.

옮겨 올 이 브랜치의 기능. `[Docs]` 커밋은 문서라서 PR 브랜치에서는 뺀다.

| 기능 | 커밋 |
|---|---|
| MoE 층의 router type(`softmax_scale`), router 입력, `cache_experts` | 11fefbb |
| Gemma-4 MoE 블록, K == V, per-layer input 생략 가능 | 1cffe5e, 3fde104 |
| weight converter, 양자화기의 Gemma-4 MoE expert 배치, safetensors 입력 | 228a026, f714503, 9dd2c76 |
| HTP MoE GeGLU epilogue (`act` 플래그)와 device 테스트 | 757e819, 1dcb7e0, 3bc7499, 906d05e, 0dcb278 |
| Gemma-4 MoE tiny 모델 HF 대조 테스트 | 06862c3, d02cd45 |
| `whUnpack`과 round-trip 테스트 | 7dbd876 |
| `NNTR_MOE_DIFF` / `NNTR_MOE_SHADOW` (층별 MoE f32 대조) | a7c527a |
| 양자화기의 텐서 순서 검사 (노이즈 출력의 원인을 막음) | f9c09ea |
| QS4CX `pack()`을 ARM에서만 | 89b426f |
| q/k/v/o, dense MLP의 engine 키 | 42d1ac6 |
| QS4CX FC slice와 로드 시 등록, 양자화기 허용 | 66c41a5, 69db273, 7becfdc |
| `tools/prefill_timeline.py` (층별 timeline, CPU/NPU 열) | 93c2a6f, f33ed21, e8d2604 |
| (넣지 않음) MSE scale과 그 revert: 둘이 서로 상쇄된다 | d652e57, 2212121 |

## 2. 지금 기준선 (기기 R3CY10WM83Y, S25 Ultra, V79)

모델: `--fc_dtype Q4_0 --moe_dtype QS4CX_WH --embd_dtype Q4_0 --lmhead_dtype Q4_0 --isa ARM`, 12,935,608,440 B.
조건: 512토큰 요약 prompt, C=16.

| 설정 | prefill | decode (512토큰 생성) | 텍스트 |
|---|---|---|---|
| 전부 NPU (MoE, q/k/v/o, dense MLP를 htp), 일반 빌드 | 3986 ms, **128.5 TPS** | 3.41 TPS | 정상 요약 |
| MoE만 NPU (§10.13) | 4.62 s, 110.7 TPS | 3.65 TPS (32토큰 생성) | 정상 요약 |
| 전부 CPU (PR 4296 앱, Q4_0 expert flash offload) | 10.87 s, 47.1 TPS | 3.26 TPS | 정상 요약 |

정확도 기준(446토큰 prompt, `NNTR_PPL=1`, prefill nll/token):

| 설정 | nll |
|---|---|
| MoE만 NPU, FC는 CPU Q4_0 | **4.556** ← 기준 |
| 전부 NPU, FC는 로드 시 Q4_0→QS4CX 재양자화 | 5.095 (+0.54, 대부분 q/k/v/o) |
| 전부 NPU, FC는 오프라인 QS4CX | 4.510. 단, decode가 CPU QS4CX로 바뀌어 512 prompt에서 반복 루프 → 되돌림 (§10.15) |
| 전부 CPU, ARM Q4_0 (PR 4296) | 4.458 |

C(층당 상주 expert 수) sweep 결과(§10.14, 전부 NPU, 512토큰): C=8이 decode 4.51 TPS로 가장 빠르다. C=16은 3.39 TPS다. C≥32는 DSP 주소 공간이 모자라(ENOMEMORY) 로드에 실패한다. **HTP로 옮기는 연산이 늘면 DSP heap 사용이 늘어 C 상한이 내려간다. 단계마다 C=16 로드가 성공하는지 확인한다.**

## 3. 512토큰 prefill 병목 — 측정 (프로파일 빌드, 전부 NPU, 4820–4959 ms)

`tools/prefill_timeline.py <log> --config <nntr_config.json> --by-op --by-layer`로 만든 표다.

| 연산 | 연산기 | 합계 | 비중 | 층 평균 | 비고 |
|---|---|---|---|---|---|
| attention 본체 `mha_core` | **CPU FP16** | 2067–2138 ms | 43% | 69–71 ms | L0 330 ms, L5 670–685 ms(첫 sliding 층, 첫 full 층). 나머지 층은 25–70 ms |
| MoE `sparse_moe` | NPU (router·top-k는 CPU) | 1193–1259 ms | 25% | 40–42 ms | DSP 29.6 ms/층: mm 15.6, acc 6.7, requant 2.7. router(CPU) 191 ms/30층 |
| q/k/v/o FC | NPU | 608–644 ms | 13% | 층당 약 21 ms | 호출당 transport 0.5–2.9 ms, quant 0.5–1.0 ms |
| dense MLP gate/up/down | NPU | 358 ms | 7% | 층당 약 12 ms | |
| scalar_multiply (`layer_scalar`, `q_scaled`) | CPU | 148–173 ms | 3.4% | | |
| rms_norm류 (층당 9개) | CPU | 약 300 ms | 6% | | |
| add, GeLU, GeGLU multiply | CPU | 약 120 ms | 2.5% | | |
| ARM staging memcpy (HTP 입출력) | CPU | 240 ms | — | | `[HTP-PROFILE] arm staging` |
| lm_head (Q4_0), softcap | CPU | 24 ms | 0.5% | | lm_head는 마지막 행만 |

- 512토큰에서 expert 선읽기는 3360개 전부 제때 끝났다(기다린 시간 0 ms). **flash는 아직 병목이 아니다.** 하지만 prefill 한 번이 약 10 GB를 읽으므로 3.0 GB/s 기준 약 3.3 s가 바닥이다. CPU 몫을 줄이면 이 바닥이 드러난다(§10.11). 단계마다 `expert prefetch ... exposed wait`를 같이 본다.
- L0/L5의 1 s는 RoPE 표 때문이다. `MHACoreLayer::precompute_freqs`가 `max_position_embeddings` 262144 위치분 cos/sin을 위치마다 `std::vector`로 만든다. 실제로 쓰는 위치는 `max_seq_len`(2048) 미만이다. **정확도와 무관한 순수 낭비다.**

## 4. CPU에 남은 연산 → 옮길 곳 (PR 4385 부품과의 대응)

Gemma-4는 LFM2와 다른 점이 많다. 아래 "Gemma 특이점"은 PR 4385 커널이 지원하는지 **먼저 확인한다.** 확인하지 않고 "될 것"이라고 쓰지 않는다.

| CPU 연산 | PR 4385에서 쓸 것 | Gemma 특이점 (지원 여부 확인) |
|---|---|---|
| attention 본체 (prefill) | `sdpa_fp16_kvcache` → `attn_f16_prefill` (HMX flash attention), rpcmem KV cache (`alloc_shared`) | head_dim 256(sliding)과 **512**(full). plan 20의 측정은 hd 128이다. sliding window 1024. full 층은 `attention_k_eq_v`(V = K 투영, v_proj 없음, k_norm 전)이고 kv head 2. 커널이 1/sqrt(hd)를 곱하는데 Gemma는 scaling=1.0이라, 지금은 `q_scaled`(×sqrt(hd))로 상쇄한다(`gemma4_causallm.cpp:624`) |
| attention 본체 (decode) | `hvx_attn_decode_f16` (n_q<5, **hd≤128**), resident `ATTN_M1` | hd 256/512는 조건 밖 → 커널 확장 또는 HMX 경로 |
| RoPE | `hvx_m1_ops_f32` ROPE는 **head_dim 64 전용**(`mha_core.cpp` `htpDecodeAttention`의 `head_dim == 64` 조건) | sliding: theta 1e4, default, hd 256. full: theta 1e6, **proportional, partial_rotary_factor 0.25**, hd 512 |
| RMSNorm (층당 9개: attention_norm, q_norm, k_norm, v_norm, post_attention_norm, pre_ffn_norm, post_ffn_norm_1, pre_ffn_norm_2, router_norm, post_ffn_norm_2, post_ffn_norm) | `hvx_m1_ops_f32` RMSNorm (whole row / per head) — M=1 | prefill(M>1) 버전 필요 여부. v_norm은 gamma 없는 RMSNorm |
| router (2816→128 FP32) + top-k | `hvx_router_topk_f32`, `ROUTER_TOPK` op (#132) | Gemma는 `softmax_scale` router(softmax 뒤 top-8, `per_expert_scale` 곱하기). LFM2는 sigmoid+bias |
| GeLU(tanh)·GeGLU multiply | — (MoE 커널에는 GeGLU epilogue가 있다, `act` 플래그) | dense MLP는 FC 3개에 CPU GeGLU. dense FFN 융합 경로(`register_q4_0_dense_ffn`)는 **SwiGLU 전용인지** 확인 |
| residual add, `ffn_sum` | `residual_add` 레이어, `ADD` op | |
| `layer_scalar`, `q_scaled` | — | `q_scaled`는 q_norm gamma에 미리 곱해 둘 수 있다(값이 같다). `layer_scalar`는 층 출력 전체에 곱한다 |
| final norm, lm_head, `logit_softcapping`(30.0) | `132-cpu-exact-fc-lmhead` plan의 LM_HEAD op | lm_head 262144×2816 Q4_0, 마지막 행만 |
| embedding | **CPU에 남긴다** | |

## 5. 권장 순서 (각 단계: 측정 → 변경 → 측정)

| # | 작업 | 근거 / 예상 효과 | 정확도 게이트 |
|---|---|---|---|
| 1 | RoPE 표를 `max_seq_len`까지만 생성 | L0+L5 약 1.0 s 중 대부분(예상) | nll 4.556 그대로, 텍스트 동일 |
| 2 | q/k/v를 한 호출로, gate+up을 한 호출로 (같은 입력을 쓰는 FC끼리. LFM2 `qkv_layer`, `gemm_q4_0_batch_fp32` 재사용) | 층당 호출 7→4, transport와 quant 3회 절약. 예상 −0.15~0.2 s | 값 동일(같은 커널) |
| 3 | prefill attention 본체를 HMX `attn_f16_prefill`로 | 0층과 5층을 빼고도 약 1.1 s. hd 256/512 지원이 관건 | FP16 수준. nll Δ ≤ 0.01 |
| 4 | 소형 연산(RMSNorm, add, scalar, GeGLU)을 HVX로. 가능하면 앞뒤 HTP 호출에 붙여 왕복 제거 | 약 0.57 s + staging 0.24 s | bit-identical 또는 nll Δ ≤ 0.005 |
| 5 | router를 MoE HTP 호출에 합치기 | 0.19 s, CPU-DSP 왕복 감소 | top-k 선택 동일 여부 |
| 6 | MoE 블록을 64행에서 32행으로 (512토큰에서 expert당 평균 32행이라 블록이 반쯤 빈다) | mm+acc 22 ms/층 중 약 8–10 ms (예상) | 값 동일 |
| 7 | decode 쪽: hd 256/512 decode attention, M=1 소형 연산을 resident graph(`hexkl_graph`)에 | decode 3.4 TPS. expert miss의 flash 읽기와 함께 분석 | 8-prompt text set (`loop_check.py`) |

- **측정하기 전에 가설로 코드를 바꾸지 않는다**(56 §8).
- 예상 효과는 모두 산술값이다. 기기에서 확인한 값만 "측정"이라고 쓴다.
- FC를 HTP로 보낼 때 생기는 정확도 손실(+0.54)은 형식 문제다(§10.12, §10.15). 이 작업의 범위가 아니다. 다른 파트에서 오차가 적은 양자화 binary를 받기로 했다.

## 6. 검증

### 6.1 host (PC)

```bash
cd ~/workspace/nntrainer
meson build -Denable-transformer=true   # 처음 한 번
ninja -C build
cd build && meson test unittest_causallm_models --print-errorlogs; cd ..
bash test/htp/host/run_host_checks.sh
```

- e8d2604 기준으로 `unittest_causallm_models`는 98/100 통과한다. 실패하는 `Lfm2DifferentialTest.Q40CloseToFP32Reference`와 `Lfm2MoeDifferentialTest.Q40MatchesHFReference`는 **이 브랜치의 변경 전에도 실패했다**(원인은 조사하지 않았다). PR 4385 위에서 다시 확인한다.
- HTP 백엔드(`htp_compute_ops.cpp`)는 x86에서 컴파일되지 않는다. Android 빌드가 컴파일 검사를 대신한다.

### 6.2 정확도 (기기)

- 446토큰 prompt의 `NNTR_PPL=1` nll(§2 표)로 본다. config는 `docs/...`가 아니라 기기 디렉터리에 있다(§7.3).
- 512토큰 prompt로 512토큰을 생성해 텍스트를 확인한다. 정상 출력은 "The small harbour town of Ardley, located where a river meets the sea, has a rich history of fishing and farming. ..."이다.
  - prefill nll이 좋아도 decode가 반복 루프에 빠진 사례가 있다(§10.15). 텍스트도 반드시 본다.
- 층 단위로 의심되면 `NNTR_MOE_DIFF=<rows>`(MoE f32 대조, SNR 출력)를 쓴다. 같은 방식으로 새 HTP 연산에도 CPU 대조를 붙이는 것을 권한다.

## 7. 기기 측정 가이드

### 7.1 환경

- 기기 `R3CY10WM83Y`(S25 Ultra, Hexagon V79). 모든 adb 명령에 `-s R3CY10WM83Y`를 붙인다.
- SDK와 도구 경로:

```bash
export HEXAGON_SDK_ROOT=$HOME/workspace/Hexagon_SDK/6.4.0.2
export HEXKL_ROOT=$HOME/workspace/hxkl-beta2/hexkl_addon
export ANDROID_NDK=$HOME/workspace/android-ndk-r26d
```

### 7.2 빌드와 설치

```bash
cd ~/workspace/nntrainer/Applications/CausalLM
./build_android.sh --htp            # 일반 빌드, 약 15분. --cache를 쓰지 않는다
./build_android.sh --htp --profile  # 층별 시간표 빌드
```

- `--cache`는 설치된 옛 헤더(`builddir/android_build_result/include`)를 그대로 쓴다. 새 헤더 함수가 없다는 오류가 날 수 있다.
- **IDL이나 DSP 코드를 바꿨으면** stub, skel, 앱을 모두 다시 빌드하고 `libnntr_hvx_skel.so`도 push한다. 빌드 스크립트가 "this does NOT rebuild libnntr_hvx_skel.so"라고 알려 준다. skel 빌드는 `test/htp/build.sh`다.
- `install_android.sh`는 `jni/libs/arm64-v8a/libc++_shared.so`가 없으면 아무것도 올리지 않고 멈춘다. 그럴 때는 바뀐 파일만 직접 push한다:

```bash
D=/data/local/tmp/nntrainer/causallm
adb -s R3CY10WM83Y push jni/libs/arm64-v8a/nntrainer_causallm $D/
adb -s R3CY10WM83Y push jni/libs/arm64-v8a/libcausallm_core.so $D/
adb -s R3CY10WM83Y push ../../builddir/android_build_result/lib/arm64-v8a/libnntrainer.so $D/
adb -s R3CY10WM83Y push ../../builddir/android_build_result/lib/arm64-v8a/libccapi-nntrainer.so $D/
```

- 프로파일 빌드는 일반 바이너리를 덮어쓰지 않도록 `/data/local/tmp/nntrainer/causallm_prof`에 올린다(그 디렉터리에 `$D/*.so`를 먼저 복사해 둔다). 실행할 때 `ADSP_LIBRARY_PATH=/data/local/tmp/nntrainer/causallm`을 주어 skel은 일반 디렉터리의 것을 쓴다.

### 7.3 기기의 모델과 설정

| 경로 (`/data/local/tmp/nntrainer/causallm/models/`) | 내용 |
|---|---|
| `gemma4-26b-a4b-qs4cx-wh/` | 모델 본체 `nntr_gemma4_q40_arm.bin`(12,935,608,440 B), tokenizer, config. 전부 NPU 설정 |
| `g4-npu-512`, `g4-npu-1024` | 전부 NPU, 512/1024 prompt, 생성 512 (bin은 symlink) |
| `g4-prof-npu`, `g4-prof-moe` | 프로파일용. 512 prompt, 생성 8, 전부 NPU / MoE만 NPU |
| `g4-cpu-512`, `g4-cpu-1024` | 전부 CPU 비교용. **PR 4296 앱** `/data/local/tmp/nntrainer/causallm_pr4296`, 모델 `gemma4-26b-a4b-q40-arm` |
| `g4-fcq-htp`, `g4-fcq-cpu`, `g4-moe-*`, `g4-diag` 등 | 옛 실험용. 일부는 지운 bin을 가리킨다 |

- 기기 저장 공간이 약 2–15 GB뿐이다. 새 모델을 올리려면 옛 bin을 먼저 지워야 한다. PC `/`도 약 7–20 GB만 남아 있다.
- PC의 원본과 양자화 결과:
  - FP32 원본: `Applications/CausalLM/res/gemma4_26ba4b/nntr_gemma4_fp32_fixed.safetensors`. 입력 디렉터리는 symlink를 모아 둔 `res/gemma4_26ba4b_fixed`다.
  - 양자화 결과: `res/gemma4_26ba4b/q40_fixed/`
  - 양자화 명령(약 15분):

```bash
./build/Applications/CausalLM/nntr_quantize_stream Applications/CausalLM/res/gemma4_26ba4b_fixed \
  -o Applications/CausalLM/res/gemma4_26ba4b/q40_fixed --output_bin nntr_gemma4_q40_arm.bin \
  --fc_dtype Q4_0 --moe_dtype QS4CX_WH --embd_dtype Q4_0 --lmhead_dtype Q4_0 --isa ARM
```

  `--config`는 쓰지 않는다. lm_head dtype과 output_bin 이름이 엉뚱하게 바뀐다.

### 7.4 실행과 측정

```bash
adb -s R3CY10WM83Y shell "cd /data/local/tmp/nntrainer/causallm && \
  LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. NNTR_NUM_THREADS=8 NNTR_HTP_PROFILE=1 \
  ./nntrainer_causallm /data/local/tmp/nntrainer/causallm/models/g4-npu-512" 2>&1 | tee run.log
grep -aE 'prefill:|generation:|peak memory|registration total|expert prefetch' run.log
```

| 환경 변수 | 뜻 |
|---|---|
| `NNTR_MOE_CACHE_EXPERTS=<C>` | 층당 상주 expert 수. config의 `moe_cache_experts`보다 우선한다 |
| `NNTR_HTP_PROFILE=1` / `2` | HTP 호출, 등록, expert miss, prefetch 통계. 2는 호출 형태별 DSP 분해(quant, mm, acc 등) |
| `NNTR_M0_PROFILE=1` | MoE 층마다 router, top-k, ffn 시간 |
| `NNTR_PPL=1` | prompt의 nll/token. prefill에서 lm_head가 모든 행을 채점하므로 prefill 시간이 늘어난다 |
| `NNTR_MOE_DIFF=<rows>` / `NNTR_MOE_SHADOW=1` | MoE를 층별로 f32와 대조. SHADOW는 출력을 f32로 바꾼다 |

층별 표:

```bash
adb -s R3CY10WM83Y pull /data/local/tmp/nntrainer/causallm/models/g4-prof-npu/nntr_config.json prof_cfg.json
python3 tools/prefill_timeline.py prof.log --config prof_cfg.json --by-op --by-layer
python3 tools/prefill_timeline.py prof.log --config prof_cfg.json --layer 5
```

### 7.5 측정 규칙 (이 세션이 비용을 치르고 배운 것)

- **한 번에 하나만 돌린다.** 다른 `nntrainer_causallm`(사용자의 CPU 벤치마크 포함)과 겹치면 miss당 flash 읽기가 1.5 ms에서 7 ms로 느려져 숫자가 무효가 된다. 시작 전에 `adb shell 'ps -A | grep nntrainer_causall[m]'`로 확인한다.
- **식힌 뒤 잰다.** 512토큰을 생성하면 SoC가 약 58 °C까지 오른다. 실행 사이에 최소 180 s를 쉬고, CPU·NSP 센서 최고값이 38 °C 이하, 배터리(`dumpsys battery`)가 30.0 °C 이하가 될 때까지 기다린다. scratchpad의 `prof_run.sh`의 `cool()`과 같은 방식이다.
- `pgrep -f`로 기다리는 루프는 자기 자신의 명령줄과 일치해서 끝나지 않는다. `grep 'nntrainer_causall[m]'`처럼 쓴다.
- PC에서 PR 4296 같은 다른 worktree의 바이너리를 돌릴 때는 `env -u LD_LIBRARY_PATH`를 쓴다. 셸의 `LD_LIBRARY_PATH`가 메인 저장소의 라이브러리를 잡는다.
- config 함정:
  - `skip_prefill: true`이면 PPL을 채점할 위치가 없다.
  - prompt 길이가 `init_seq_len`과 같으면 첫 생성 토큰 등록을 건너뛰는 분기를 탄다. 1024 prompt는 `init_seq_len` 2048로 둔다.
- 기기에서 돌리지 못했거나 확인하지 못한 것은 **"기기 미측정"**이라고 쓴다.

## 8. 작업 규칙

- 저장소 규칙은 `AGENTS.md`를 따른다.
  - `git commit -s`, 에이전트가 쓴 커밋에는 `Co-Authored-By:` trailer를 붙인다.
  - 제목은 `[<component>] <subject>`. history의 component를 따른다: `[HTP]`, `[HTP/MoE]`, `[CausalLM]`, `[CausalLM/Gemma4]`, `[Tensor]`, `[Tools]`, `[Docs]`.
  - 한 커밋에 한 주제. clang-format-14는 바꾼 줄에만 적용한다(`git diff -U0 | clang-format-diff-14 -p1`).
  - `subprojects/`는 고치지 않는다. 크로스 플랫폼을 유지한다(`#ifndef _WIN32` 등).
- 본문 형식은 history의 예(42d1ac6, 66c41a5)를 따른다. 무엇이 문제였는지, 무엇을 바꿨는지, 무엇을 측정했는지(Host:/Device: 줄, 숫자)를 쓴다.
- **모델 이름을 커밋과 코드 주석에 쓰지 않는다.** 단, 기존 component 태그와 문서(`docs/htp_attention`)는 예외다.
- ponytail 작업 방식(`01_working_style.md`, CLAUDE.md):
  - 이미 있는 것을 재사용한다. 특히 PR 4385의 커널, ComputeOps 진입점, 테스트 하네스.
  - 요청받지 않은 추상화는 만들지 않는다.
  - 의도적으로 줄인 부분에는 `ponytail:` 주석으로 한계와 확장 경로를 남긴다.
  - 사소하지 않은 로직에는 실행 가능한 검사를 하나 남긴다. HTP 커널이면 host check나 기기 gtest로 CPU 결과와 대조한다.
- 측정 → 변경 → 측정 순서를 지킨다. 결과는 이 문서 아래 §9에 날짜와 커밋을 붙여 추가한다.
- 답변은 한국어로 쓴다.

## 9. 결과 기록

(여기에 단계별 결과를 추가한다: 날짜, 커밋, 조건, prefill/decode, nll, 텍스트, `prefill_timeline` 상위 표.)
