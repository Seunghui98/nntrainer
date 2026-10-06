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
- **식힌 뒤 잰다.** 512토큰을 생성하면 SoC가 약 58 °C까지 오른다. 실행 사이에 최소 180 s를 쉬고, CPU·NSP 센서 최고값이 38 °C 이하, 배터리(`dumpsys battery`)가 30.0 °C 이하가 될 때까지 기다린다. 쓰던 함수는 아래와 같다(20분이 지나도 안 식으면 그대로 진행하고 온도를 기록한다).

```bash
export ANDROID_SERIAL=R3CY10WM83Y
temps() { adb shell 'm=0; for z in /sys/class/thermal/thermal_zone*; do case "$(cat $z/type)" in cpu*|nsp*) t=$(cat $z/temp); [ "$t" -gt "$m" ] && m=$t;; esac; done; b=$(dumpsys battery | grep "  temperature" | tr -dc 0-9); echo "$m $b"'; }
cool() { while adb shell 'ps -A' | grep -q 'nntrainer_causall[m]'; do sleep 10; done; sleep 180
         for i in $(seq 120); do read soc bat <<<"$(temps)"; [ "$soc" -le 38000 ] && [ "$bat" -le 300 ] && break; sleep 10; done
         echo "start soc=$soc bat=$bat"; }
```
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

### 9.0 0단계·1단계 (2026-10-02): 브랜치 세움, host 검증 통과, 기기 미측정

- 브랜치: `claude/zealous-bell-a2pot9` = pr4385(`htp_first_version` @ d4a8984) + 46커밋(코드 24, 테스트 보강 1, `[Docs]` 21). d652e57·2212121은 §1대로 뺐다. 공통 조상 7a8e260.
- 충돌 2건을 PR 4385 구조로 풀었다.
  - e23f947e (757e819의 GeGLU `act`): 커널 `hexkl_mm_u8i4_moe_layer_run`의 시그니처는 PR 4385의 `flags` 그대로 두고 `HEXKL_MOE_FLAG_GELU_TANH`(bit 19)를 더했다. skel이 IDL의 `act`(0 silu, 1 gelu_tanh)를 그 비트로 바꿔 `moe_set_opts` 워드에 OR 한다. dspqueue 패킷(`htp_dspq_wire.h`)에는 `HTP_DSPQ_FLAG_GELU`. M=1 GEMV 경로(`moe_m1_pair_worker`)도 같은 `hvx_dequant_swiglu_acc_tiles_to_f32`를 쓰므로 GeGLU를 받는다. host 체크: GeGLU 층 0 mismatch(swiglu_det.h의 `geglu_det_one` 대조), identity 1.2e-6.
  - 1b79a200 (a7c527a의 `NNTR_MOE_DIFF`): include 줄만 충돌.
- 6f7fa8d3: PR 4385가 7a8e260 뒤에 추가한 기기 gtest 호출 4곳(`unittest_hvx_mm_u8i4.cpp` 2, `unittest_hvx_dma_probe.cpp` 2)에 `act` 인자를 넣었다. Android 빌드는 이 환경에 없어 IDL 순서로만 맞췄다.
- host (x86, 시스템 OpenBLAS, meson 1.5.2): `run_host_checks.sh` 끝까지 통과, `htp_syntax_check.sh` 통과, `unittest_causallm_models` **98 passed / 0 failed / 3 skipped**(101개; `Lfm2MoeDifferentialTest` 3개는 lfm2_moe_tiny fixture 가중치를 torch 없이 만들 수 없어 skip. e8d2604 기준에서 실패하던 Q40 두 테스트가 여기서는 skip이라 그 원인은 아직 모른다). Gemma4·Gemma4Moe differential 테스트는 통과.
- **기기 미측정.** IDL이 바뀌었으므로 기기에서는 stub·skel·앱을 모두 다시 빌드하고 `libnntr_hvx_skel.so`를 push해야 한다(§7.2). §2 기준선은 이 브랜치에서 다시 재야 한다.

### 9.1 §5-1, §5-3, §5-2, §5-4, §5-8 (2026-10-06): prefill의 CPU 몫을 HTP 호출 안으로, 기기 미측정

브랜치 `claude/zealous-bell-a2pot9`, pr4385 위 59커밋. 전부 host에서만 검증했다(`unittest_causallm_models` 98 passed / 3 skipped, `run_host_checks.sh`, `htp_syntax_check.sh`). **기기 미측정.**

| 단계 | 커밋 | 내용 | 기대 효과(산술, 57 §3 기준) |
|---|---|---|---|
| §5-1 | `[CausalLM] mha_core: build the RoPE table to max_timestep` | RoPE 표를 262144 → `max_timestep`(2048) 위치만 | L0·L5의 약 1.0 s |
| §5-3a | (코드 변경 없음) | Gemma4 builder가 이미 `createAttentionCore`를 거침. config에 `"attention_engine": "htp"`를 주면 sliding 25층(hd 256)이 `attn_f16_prefill`로 | attention 본체 2.1 s 중 sliding 층 몫 |
| §5-3b | `[HTP] attn_f16: take head_dim up to 512` | plan·ComputeOps의 hd 상한 256 → 512. full 5층(hd 512)도 HMX. 기기 gtest `PrefillWideHeadsTwoKvHeads` 추가 | 커널에 host stand-in이 없어 **기기 gtest가 게이트** |
| §5-2 | `[CausalLM] dense_ffn: a GeGLU gate ...`, `[CausalLM] qkv_layer: a gamma-less v norm ...`, `[CausalLM/Gemma4] q/k/v and the dense MLP as one layer each` | q/k/v(+q_norm·k_norm·v_norm·√hd) → `qkv_layer` 1호출, gate/up/GeGLU/down → `dense_ffn` 1호출(GeGLU는 DSP epilogue). 가중치 순서는 파일과 동일 | 층당 FC 호출 7 → 2, CPU GeLU·multiply·q_scaled 제거 |
| §5-4 | `[HTP] hvx_rmsnorm_rows_f32 ...`, `[HTP] Layer calls that carry the block's RMSNorms`, `[CausalLM] in_norm / out_norm on the fused layers ...`, `[CausalLM/Gemma4] Fold seven of the block's nine norms ...` | IDL `mm_u8i4_layer_norm`·`mm_u8i4_moe_layer_norm`: pre/post RMSNorm을 호출 안으로. attention_norm→qkv, pre_ffn_norm·post_ffn_norm_1→dense_ffn, pre_ffn_norm_2·router_norm·post_ffn_norm_2→MoE 층. q/k/v per-head norm도 호출 안 | 층당 CPU norm 9개 중 7개 제거(약 230 ms) |
| §5-8 | `[HTP] QS4CX weights on the fused calls ...`, `[CausalLM] The fused layers take QS4CX weights, at decode too` | `--fc_dtype QS4CX` 파일을 재양자화 없이 등록. decode M=1 FC도 HTP(`accelerates_qs4cx_at_m1`). 융합 호출을 512열 정렬 슬라이스로(2816×4096이 VTCM 안에) | §10.15의 nll 4.510, CPU QS4CX decode 경로(반복 루프) 제거 |

CPU에 남는 것(prefill): embedding, post_attention_norm, post_ffn_norm, 잔차 add 2개, ffn_sum, layer_scalar, RoPE(`apply_rotary_emb_tensor_v2`, attention 호출 전), router(2816→128 + softmax top-8), lm_head·softcap. 다음은 §5-5(router)와 post_attention_norm을 attention_out 층으로 접는 것. → 블록 epilogue는 §9.3에서 처리.

주의:
- IDL이 바뀌었다. stub·skel·앱을 모두 다시 빌드하고 `libnntr_hvx_skel.so`를 push한다.
- `fcSliceCols`가 512열 배수로 내림한다(head 경계). hd가 512를 나누지 않는 모델은 별도 정렬이 필요하다.
- `attention_out`의 post_attention_norm은 §9.3에서 `residual_add`로 옮겼다.
- 기기 실행 가이드는 §9.2.

### 9.3 §5-4 나머지: 블록 epilogue(post norm·잔차 add·ffn_sum·layer_scalar) 한 호출 (2026-10-06), 기기 미측정

attention_out 뒤의 `post_attention_norm`+add, FFN 뒤의 `ffn_sum`+`post_ffn_norm`+add+`layer_scalar`(층당 CPU 연산 6개)를 `residual_add` 층 둘로 바꿨다. `residual_add`가 `in_norm`(gamma 가중치), `use_weight`(scalar 가중치), 세 번째 입력(MoE 출력)을 받아 `out = scale·(resid + rmsnorm(x[+x2])·gamma)`를 prefill에서 HTP 한 호출(`rmsnorm_add_f32`, `hvx_rmsnorm_add_f32` 커널: 행마다 2회 스트리밍, scratch 없음)로 보낸다. decode(M=1)는 CPU(호출 1회가 2816 float 연산보다 비싸다).

- 왜 attention_out FC 호출 안에 접지 않았나: FC는 512열 슬라이스로 쪼개져 각 슬라이스가 별도 DSP 호출이라, 전체 행(2816) RMSNorm을 슬라이스 안에서 끝낼 수 없다. 별도 호출(층당 2회, 호출당 약 0.1–0.2 ms 전송 예상)이 가장 짧은 경로. 입력 staging memcpy는 여전히 ARM(`[HTP-PROFILE] arm staging`).
- 가중치 순서 불변: 층 이름 `_post_attention_norm`(gamma), `_post_ffn_norm`(gamma, scalar)을 그대로 쓴다. `hidden_size_per_layer_input > 0`이면 scalar는 per-layer 경로 뒤에 남는다(26B-A4B는 0).
- 검증: `rmsnorm_rows_host_check`(RMSNORM ROWS OK, epilogue 3 shape + 거부 1), `unittest_causallm_models`, `htp_syntax_check.sh`. 기기: profile에서 K=N=2816(또는 K=5632) FC 행으로 보인다.
- CPU에 남는 것(prefill): embedding, RoPE, router, lm_head·softcap. → router는 §9.4.

### 9.4 §5-5 router 로짓을 DSP로 (2026-10-06), 기기 미측정

MoE 층의 router(`router_norm` + 2816×128 f32 dot, CPU 191 ms/30층)를 prefill에서 DSP 호출 `router_logits_f32`로 옮겼다. 커널 `hvx_router_rows_f32`: 가중치 256행 chunk(128 KiB, L2 상주)를 x의 모든 행에 재사용, 행 4개씩 qf32 누적, chunk 사이는 f32. router_norm은 호출 안(`norm_rows_in`). softmax·top-8·가중치 정규화는 CPU에 그대로 둔다(선택 규칙이 CPU와 완전히 같고, 512×128이라 작다). expert 선읽기(flash)가 top-k 결과를 먼저 알아야 하므로 MoE 본 호출과 한 호출로 합치지 않았다(합치면 선읽기가 호출 뒤로 밀린다).

- 게이트: M>1, E%32==0, E≤128, FP32. decode(M=1)는 CPU dot(호출 1회 비용 > 2816×128 MAC).
- 검증: `router_rows_host_check`(ROUTER ROWS OK, worst_rel 2.6e-7, 4 shape + 거부 3), `swap_host_check`가 skel 파일을 새 entry와 함께 링크, `htp_syntax_check.sh`, `unittest_causallm_models`(CPU 경로).
- ponytail: 가중치 1.4 MB가 호출마다 FastRPC로 넘어간다(층당 prefill 1회). profile에서 K=2816 N=128 행으로 보인다.
- (2026-10-06 추가) softmax·top-8·가중치 정규화도 같은 호출 안으로: `hvx_router_topk_rows_f32`(기존 `hvx_softmax_rows_f32` 재사용 + 첫 최대값 스캔, 동률이면 낮은 index = CPU comparator). host check에서 CPU 선택과 **mismatch 0**(정수 로짓의 정확한 동률 포함), 가중치 worst_rel 2.8e-7. 층은 sel/weight로 `expert_assignments`와 prefetch 힌트(top-8+5)를 바로 만든다. sigmoid router(LFM2)는 기존대로 로짓만.
- CPU에 남는 것(prefill): embedding, RoPE(attention 호출 전 q/k 회전), lm_head·softcap. → RoPE는 §9.5.

### 9.5 RoPE를 qkv 호출의 post 단계로 (2026-10-06), 기기 미측정

`mha_core`가 attention 호출 전에 CPU로 돌리던 q/k RoPE(`apply_rotary_emb_tensor_v2`, 층당 q 512×4096 + k 512×2048 f32)를 `qkv_layer`로 옮겼다. `qkv_layer`에 `rope`, `rope_theta`, `rope_scaling_type`(default·proportional), `rope_partial_rotary_factor`, `max_timestep` 속성; 표는 mha_core의 `precompute_freqs`와 같은 수식(`calc_trigonometric_vals_dup`, 같은 float/double 연산 순서)으로 [pos][cos|sin] 행을 만들고 같은 shape의 층끼리 공유한다(sliding hd 256: 4 MB, full hd 512: 8 MB). HTP: `mm_u8i4_layer_norm`에 `rope_hd/rope_handles/rope_cs`가 붙어 per-head norm 뒤에 q·k 슬라이스를 `hvx_rope_rows_f32`로 회전(행마다 표 2·hd float, 512행 full 층이면 2 MB 전송). CPU: 같은 표로 `compute_rotary_emb_value`. Gemma4 builder는 attention core에 `use_rope=false`를 주므로 core는 k를 fp16 캐시로 변환만 한다.

- 검증: `rope_rows_host_check`(CPU 커널 연산 순서와 비트 일치, hd 256·512(partial 0.25)·64, 거부 3), `swap_host_check`(확장된 entry 링크), `htp_syntax_check.sh`, `unittest_causallm_models`(Gemma4 golden logits가 CPU 경로의 RoPE 이동을 검증).
- KV 공유 층(`createSharedAttention`)은 q만 FC+norm+scalar로 만들고 core의 RoPE를 그대로 쓴다(prefill을 건너뛰는 층).
- ponytail: `rope_scaling_factor`는 1.0 가정(yarn은 거부 → core의 use_rope를 켠다).
- CPU에 남는 것(prefill): embedding, lm_head·softcap, ARM staging memcpy.

### 9.2 기기 실행 가이드 (이 브랜치)

```bash
cd ~/workspace/nntrainer && git fetch origin claude/zealous-bell-a2pot9 && git checkout claude/zealous-bell-a2pot9
export HEXAGON_SDK_ROOT=$HOME/workspace/Hexagon_SDK/6.4.0.2 HEXKL_ROOT=$HOME/workspace/hxkl-beta2/hexkl_addon ANDROID_NDK=$HOME/workspace/android-ndk-r26d
./test/htp/build.sh                                   # skel (IDL 변경): UNDEFINED SYMBOLS OK
(cd Applications/CausalLM && ./build_android.sh --htp) # stub 재생성 + 앱. --cache 금지
export ANDROID_SERIAL=R3CY10WM83Y; D=/data/local/tmp/nntrainer/causallm
adb push test/htp/build/libnntr_hvx_skel.so $D/
adb push Applications/CausalLM/jni/libs/arm64-v8a/{nntrainer_causallm,libcausallm_core.so} $D/
adb push builddir/android_build_result/lib/arm64-v8a/{libnntrainer.so,libccapi-nntrainer.so} $D/
```

모델: FC를 QS4CX로 다시 양자화한다(§7.3 명령에서 `--fc_dtype QS4CX`, 약 15분, 12.2 GB). 기존 Q4_0 bin도 로드되지만 FC가 로드 시 재양자화 경로를 탄다.

config(`g4-npu-512/nntr_config.json`)에 더할 키:

```json
"attention_engine": "htp",
"attn_proj_engine": "htp",
"dense_ffn_engine": "htp",
"moe_engine": "htp",
"fc_layer_dtype": "QS4CX",
"model_tensor_type": "QS4CX-FP32"
```

측정 순서(§7.5의 `cool` 뒤, 한 번에 하나):
1. 기기 gtest `unittest_hvx_attn_f16 --gtest_filter='*PrefillWideHeads*'` (hd 512 커널 게이트).
2. `g4-npu-512` 일반 빌드: prefill/decode TPS, 512토큰 텍스트.
3. 446토큰 `NNTR_PPL=1`: nll (기준 4.556; QS4CX FC면 4.510 근처 기대).
4. `--profile` 빌드로 `g4-prof-npu` → `prefill_timeline.py --config nntr_config.json --by-op --by-layer`. 이 브랜치의 그래프 이름(`qkv`, `ffn`, `post_attention_norm`, `post_ffn_norm`, `attention`)을 CPU/NPU 열로 가른다.
5. 각 단계를 가르려면 config 키를 하나씩 켠다: `attention_engine`만(§5-3), 거기에 engine 키 셋(§5-2·5-4·9.3·9.5), 마지막에 QS4CX 파일(§5-8). router(§9.4)는 `moe_engine`을 따른다.

§9.3–9.5 이후 `[HTP-PROFILE]` 표에서 새로 보이는 행: FC 행 `K=2816 N=2816`(post_attention_norm의 epilogue), `K=5632 N=2816`(post_ffn_norm: dense+MoE 두 addend), `K=2816 N=128`(router). RoPE는 qkv 호출 안이라 행이 따로 없다(호출 시간에 포함). 기대 효과(산술): §3의 CPU 몫 중 norm·add·scalar 약 420 ms + router 191 ms + RoPE(§3에서 attention 본체에 섞여 있던 몫)가 DSP 호출로 바뀐다. 호출당 전송 0.1–0.2 ms × 층당 3회 × 30층 ≈ 10–20 ms가 새 비용. **기기 미측정.**

정확도 게이트: nll이 4.51±0.02 밖이면 §9.3(epilogue)·§9.5(RoPE)를 config로는 끌 수 없으므로 `NNTR_MOE_DIFF`처럼 층별 대조가 필요하다. 가장 빠른 분리: `attn_proj_engine: cpu`(qkv·attention_out·post_attention_norm이 CPU로, RoPE도 CPU 경로) → nll이 돌아오면 §9.5/9.3-attention 쪽, 아니면 `dense_ffn_engine: cpu`로 §9.3-ffn 쪽.

### 9.6 지금 기준 층별 실행 위치 (prefill, 2026-10-06 코드 기준, 기기 미측정)

config에 engine 키 넷을 모두 `htp`로 준 경우. 층당 DSP 호출 8회(qkv, attention, attention_out, post_attention_norm, ffn, router, MoE, post_ffn_norm; dense만 있는 층은 6회).

| 그래프 노드 | 하는 일 | 어디서 | 비고 |
|---|---|---|---|
| `embedding0` | 토큰 임베딩, ×√hidden | **CPU** | 남긴다 |
| `layer{i}_qkv` | attention_norm → q/k/v 투영(QS4CX) → q_norm·k_norm·v_norm·√hd → RoPE | NPU 1호출 | `mm_u8i4_layer_norm`(+rope) |
| `layer{i}_attention` | HMX flash attention(hd 256/512); k/v fp16 캐시 변환은 CPU memcpy급 | NPU | `attn_f16_prefill` |
| `layer{i}_attention_out` | o 투영(QS4CX) | NPU 1호출 | 512열 슬라이스 |
| `layer{i}_post_attention_norm` | post_attention_norm + 잔차 add | NPU 1호출 | `rmsnorm_add_f32`; decode(M=1)는 CPU |
| `layer{i}_ffn` | pre_ffn_norm → gate/up(QS4CX) → GeGLU → down → post_ffn_norm_1 | NPU 1호출 | dense FFN 호출 |
| `layer{i}_sparse_moe` router | pre_ffn_norm_2·router_norm → 2816×128 로짓 → softmax·top-8·가중치 | NPU 1호출 | expert_assignments 구성만 CPU |
| `layer{i}_sparse_moe` experts | 128 expert 중 8, GeGLU, post_ffn_norm_2 | NPU 1호출 | `mm_u8i4_moe_layer_norm`, flash 선읽기 |
| `layer{i}_post_ffn_norm` | dense+MoE 합 → post_ffn_norm → 잔차 add → layer_scalar | NPU 1호출 | `rmsnorm_add_f32`(x2); decode는 CPU |
| KV 공유 층(마지막 N층) | prefill을 건너뜀(skip_prefill) | — | decode 전용; q는 FC+norm+scalar, RoPE는 core |
| `output_norm`, `lm_head`, softcap | 마지막 행만 | **CPU** | lm_head 370 MB 등록 비용 → 기기 측정 뒤 결정 |
| ARM staging memcpy | 호출마다 입력/출력 복사 | **CPU** | `[HTP-PROFILE] arm staging` |


