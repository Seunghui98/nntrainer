# 56. 인계: Gemma-4 26B-A4B가 기기에서 돌아가지만 출력이 잡음이다 — 어느 연산인지 찾기

작성 2026-10-01 · 브랜치 `claude/epic-hopper-occf31` @ 0dcb278 · 과제 문서 `55_gemma4_moe_htp_task.md`(§10.4까지) · 원 인계 `54_gemma4_moe_htp_handoff.md`

이 문서만 읽고 새 세션이 이어받을 수 있게 쓴다. **지금 상태, 확인된 것, 확인 안 된 것, 증상의 숫자, 용의자와 각각을 가르는 수단, 다음에 만들려던 도구, 일하는 규칙** 순서다.

## 1. 한 줄 요약

26B 모델이 기기(Galaxy S25 Ultra, V79 HTP)에서 **끝까지 돈다**(로드·아레나·prefill·decode). 그러나 출력은 무의미한 토큰이고 `[PPL] nll/token=13.26`(ppl 576,570)으로 **균등분포(ln 262144 = 12.48)보다 나쁘다** = logits가 잡음이다. 한편 같은 그래프·변환기·양자화기 순서를 쓴 **tiny 모델은 호스트 x86에서 HF transformers와 FP32로 일치**한다. 따라서 범인은 **기기에서만 도는 코드**에 있다: HTP MoE 커널(새 GeGLU 에필로그, Gemma 형상), ARM 커널(attention head 512·GQA 8, Q4_0 ARM 레이아웃의 FC·임베딩·lm_head), 또는 QS4CX_WH 파일.

## 2. 지금 어디까지 왔나 (커밋 순)

| 커밋 | 내용 |
|---|---|
| 8a620af, 7511fcc | 55 §1~§6: config, 메모리 예산(3.2 GiB → C=16), 단계 계획 |
| 11fefbb | `Lfm2MoELayer`에 `router_type`(sigmoid_bias / softmax_scale), 선택적 두 번째 입력(router 입력), `cache_experts` 속성(env `NNTR_MOE_CACHE_EXPERTS` 우선) |
| 1cffe5e | Gemma4 모델: `enable_moe_block`이면 dense MLP 옆에 MoE 블록(norm 3개 + `lfm2_moe` 층), full 층 `attention_k_eq_v`(V = raw K 투영), `hidden_size_per_layer_input == 0` 허용, `moe_engine`/`moe_htp_layers`/`moe_cache_experts` 키, `res/gemma4/gemma4-26b-a4b/nntr_config.json` |
| 228a026 | `res/gemma4/weight_converter.py`: MoE 텐서(router gamma에 `scale·H^-0.5` 접기, expert별 fused gate_up·down, lazy slice), K==V 층 v_proj 생략 |
| f714503, 9dd2c76 | `quantize_stream.cpp`: expert를 `--moe_dtype`(QS4CX_WH)로 fused `_gate_up`/`_down`; `.safetensors` 입력 직접 읽기 |
| 757e819 | **HTP GeGLU 에필로그**: `act` 플래그(0 silu, 1 gelu_tanh)를 층 → `ComputeOps::gemm_qs4cx_moe_layer_fp32(…, gelu)` → IDL `mm_u8i4_moe_layer(…, act, …)` → skel → `hexkl_mm_u8i4_moe_layer_run(…, act, …)` → `hvx_dq_swiglu_job.act`. HVX `hvx_geglu_det_sf` = x·σ(x(C0+C1x²)), 호스트 쌍 `geglu_det_one`. CPU 경로도 `moe_activation`을 따름(전에는 항상 SwiGLU) |
| 3fde104 | `use_bidirectional_attention: "vision"`(문자열) 파싱 — 기기 로드 예외 수정 |
| 06862c3, d02cd45 | **HF 참조 차분 테스트** `Gemma4MoeDifferentialTest`(tiny, 26B 형상) + 생성기 `generate_gemma4_moe_reference.py` — 호스트 통과 |
| 3bc7499, 906d05e | 기기 gtest `HvxSwigluDet.GegluMatchesScalarBitExact`(HVX GeGLU 비트 일치; IDL `swiglu_det_f32`에 `act` 추가) |
| 0dcb278 | 기기 gtest `MoeLayerMatchesTwoCallReference`에 Gemma 형상(K 2816, inter 704, N 2816) 추가 |

호스트 상태: `meson build`(x86) 전체 빌드, `unittest_causallm_models` **86/86**, `test/htp/host/run_host_checks.sh` **3/3**.

## 3. 자산 (경로 그대로)

| 것 | 위치 |
|---|---|
| HF 체크포인트(safetensors) | PC `~/workspace/nntrainer/Applications/CausalLM/res/gemma4_26ba4b/hf/` |
| 변환 FP32 | PC `…/gemma4_26ba4b/nntr_gemma4_fp32.safetensors` (96,256 MiB; 산술 25,233,141,790×4 B와 일치) |
| 기기용 양자화 | PC `…/gemma4_26ba4b/q40/nntr_gemma4_q40_arm.bin` (12,336 MiB; `--fc_dtype Q4_0 --moe_dtype QS4CX_WH --embd_dtype Q4_0 --isa ARM`) |
| x86 참조용 양자화 | PC `…/gemma4_26ba4b/q40_x86/` (`--moe_dtype Q4_0 --isa X86`; 만들었는지 미확인) |
| 기기 모델 디렉터리 | `/data/local/tmp/nntrainer/causallm/models/gemma4-26b-a4b-qs4cx-wh/` (config.json, tokenizer.json=Gemma, nntr_config.json: `moe_engine htp`, `moe_cache_experts 16`, `moe_layer_dtype QS4CX_WH`, sample_input은 `<bos><|turn>user … <turn|>\n<|turn>model\n` 형식의 논문 초록 요약 프롬프트 ≈446 토큰) |
| 기기 바이너리 | `/data/local/tmp/nntrainer/causallm/{nntrainer_causallm, lib*.so, libnntr_hvx_skel.so}` — **IDL이 바뀌었으므로 stub(`generate_stub.sh`)·skel(`test/htp/build.sh`)·앱(`build_android.sh --htp`)을 같은 커밋에서 다시 만들어 짝을 맞춰야 한다**. skel 빌드는 `DEFAULT_HEXAGON_TOOLS_ROOT`가 필요(사용자 쉘에서 `setup_sdk_env.source`가 "already setup"이라며 안 채우는 일이 있었다 → `export DEFAULT_HEXAGON_TOOLS_ROOT=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/<ver>`) |
| 기기 실행 | `cd /data/local/tmp/nntrainer/causallm && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. NNTR_NUM_THREADS=8 NNTR_MOE_CACHE_EXPERTS=5 NNTR_PPL=1 ./nntrainer_causallm ./models/gemma4-26b-a4b-qs4cx-wh` |
| 기기 gtest | `bash test/htp/run_u8i4_layer_on_device.sh` (skel + ARM gtest 빌드 → push → 실행; `HEXAGON_SDK_ROOT HEXKL_ROOT ANDROID_NDK DEFAULT_HEXAGON_TOOLS_ROOT` 필요) |
| 호스트 HF 차분 | `build/Applications/CausalLM/unittest_causallm_models --gtest_filter='Gemma4MoeDifferential*'`; fixture 재생성 `python3 test/unittest/models/causallm_reference/generators/generate_gemma4_moe_reference.py` (torch + transformers ≥ 5.x; `--no-k-eq-v`, `--global-kv N` 변형 있음) |

## 4. 증상의 숫자 (2026-10-01, C=5, 첫 정상 완주)

```
[PPL] prompt tokens=446 nll/token=13.2649 ppl=576570
prefill: 447 tokens, 14867 ms, 30.07 TPS
generation: 512 tokens, 135092 ms, 3.79 TPS
peak memory: 3222772 KB
```
생성 예: `SAH TEL X.A RE, NUB IN / SITA RE / INING DRAMER …`. 그 전 실행에서는 LFM2 토크나이저가 들어가 있었고(`<|startoftext|>`), Gemma 토크나이저로 바꾼 뒤에도 같은 수준의 잡음이다. 긴 프롬프트(≈1000 토큰 + 256 생성)에서는 끝 무렵 `Creating shared tensor of size bigger than tensor memory`로 죽는다(1024 경계 의심, §7).

랜덤 logits(표준편차 σ)의 기대 nll ≈ ln V + σ²/2이므로 13.26은 σ≈1.25의 잡음과 일치한다. 한 층이 조금 틀린 게 아니라 **어느 한 연산이 완전히 틀렸거나 가중치·그래프가 맞물리지 않는** 종류다.

## 5. 확인된 것 / 안 된 것

**확인됨 (호스트):**
- 그래프(router softmax_scale, expert GeGLU, dense MLP와 합, norm 접기, K==V, per-layer input 없음), 변환기, 양자화기 텐서 순서 → `Gemma4MoeDifferentialTest` FP32에서 HF와 일치. 변형 실험: K==V 끄기 통과, global kv 4 통과, global kv 2(실제 26B)도 통과(KV cache를 16으로 늘린 뒤 — x86 rotary 꼬리 overshoot 때문, §7).
- 변환 바이트 수 = config 산술, 양자화 출력 크기 = 산술(12,336 MiB).
- `moe_layer_host_check`에 GeGLU 케이스·항등식·act 거부 통과.

**확인 안 됨 (기기 미측정):**
1. **HVX GeGLU** `hvx_geglu_det_sf` — gtest `HvxSwigluDet.GegluMatchesScalarBitExact` 추가했지만 **아직 기기에서 안 돌림**(첫 시도는 선언 순서 컴파일 에러 → 906d05e로 수정).
2. **HTP MoE 커널의 Gemma 형상**(inter 704 = 22 n-tile → 16+6 배치 나머지; LFM2는 56 = 3×16+8) — gtest 추가(0dcb278), **미실행**.
3. **ARM 커널**: attention head_dim 512(full)·256(sliding), GQA 8(q 16 / kv 2), Q4_0 ARM(q4_0_4) 레이아웃의 FC·임베딩 lookup·tied lm_head(런타임 repack). 작은 Gemma4(`res/gemma4`, head 512·kv 1)가 이 기기에서 정상 출력을 낸 적이 있는지 **모른다** — 사용자에게 물어볼 것.
4. **QS4CX_WH 파일**: LFM2와 같은 코드 경로지만 Gemma 형상에서 기기로 검증된 적 없음.
5. **실제 가중치의 CPU 경로**(PC x86 Q4_0): 사용자 PC의 x86 빌드는 모든 모델 테스트가 `~Transformer()`(HFTokenizer 소멸자)에서 segfault — conda/.venv libstdc++ 의심, 미해결. 모델 실행 자체는 시도 안 함.

## 6. 용의자와 가르는 수단

| 용의자 | 가르는 수단 | 결과 해석 |
|---|---|---|
| A. HTP MoE 커널 (GeGLU, Gemma 형상) | `run_u8i4_layer_on_device.sh` → `GEGLU_DET_FIELD bad_out`, `MoeLayerMatchesTwoCallReference`의 `gemma4` 형상 `bad_elems` | 0이 아니면 커널. 둘 다 0이면 커널은 (이 입력 분포에서) 빠짐 |
| B. ARM attention/Q4_0/임베딩 | 작은 Gemma4를 같은 기기에서 (`nntr_gemma4_q40_embdq6k.bin`) | 깨지면 ARM 쪽(MoE 무관). 정상이면 ARM 혐의 약화(단, kv 1·embd Q6_K라 GQA 8·Q4_0 임베딩은 안 덮음) |
| C. 전체 CPU 경로(실제 가중치) | PC x86 `q40_x86` 실행 `NNTR_PPL=1` | 정상이면 "변환·양자화·그래프 전부 맞고 기기 커널만 남음". 깨지면 실제 가중치 양자화 쪽(Q4_0 임베딩/lm_head, 변환기 키 처리) |
| D. **한 번의 기기 실행으로 층별 판정** | §6.1의 `NNTR_MOE_DIFF`/`NNTR_MOE_SHADOW` (아직 없음, 설계만) | 층마다 HTP 결과 vs CPU 참조 SNR; SHADOW로 텍스트가 살아나면 HTP MoE가 범인, 안 살아나면 ARM 쪽 |

사용자가 A·B·C 같은 "여러 번 돌려 보기"보다 **D처럼 한 번에 어느 연산인지 보이는 분석**을 원했다. D를 먼저 만들 것.

### 6.1 만들려던 도구: `NNTR_MOE_DIFF=1` / `NNTR_MOE_SHADOW=1`

레포에 같은 패턴이 이미 둘 있다: `NNTR_L2_DIFF`/`NNTR_L2_SHADOW`(`htp_compute_ops.cpp` 695~770행, `l2Diff`, fused SwiGLU 경로용)와 `NNTR_CONV_BLOCK_DIFF`/`_SHADOW`(`conv_block_layer.cpp` 35~58행, `snrDb`). **그대로 따라 MoE 층 콜에 붙인다.**

- 훅 위치: `Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.cpp` `tryMoeLayerOnAccelerator()`의 `call(dst)` 뒤(약 866행; split 경로면 합산 뒤). 그때 손에 있는 것: `input`(M×hidden f32), `output`/`dst`, `expert_assignments[e]` = (row, weight) 목록, `gelu`, `experts_virtual`, 각 expert의 `ExpertFileDesc`(`expertDesc(gu, dn)`: `fd, off_gu, off_dn, K, inter, N_out`; 430행 근처).
- CPU 참조: expert별로 **모델 파일에서 직접** `pread`(아레나·DSP와 무관): QS4CX_WH 텐서 = `whBytes(K,N)` 니블 + N개 f32 스케일 + N개 f32 colsum(파일은 colsum을 f32로 둠; `htp_compute_ops.cpp` 2822·1088행 주석). 니블은 `q + 8` 무부호(792행 주석), 값 = (nib − 8)·scale[n]. WH → row-major 역변환은 `htp_wh_layout.h`의 `whSlot(r,c)`/`whPack`을 뒤집으면 된다(`whUnpack`, 10줄; 타일 (kt,nt)의 512바이트, 원소 (r,c)는 slot s = whSlot(r,c), byte s/2의 하위(짝수 s)/상위(홀수 s) 니블). gate_up은 [K=hidden][N=2·inter] (앞 inter열 = gate), down은 [inter][hidden]. 참조 = x(f32)·W_gate_up → act(gate)·up (gelu면 `geglu_det_one`, 아니면 `swiglu_det_one` — 또는 그냥 f32 수식) → ·W_down → row_weight 곱해 scatter-add. 활성화 u8 재양자화는 참조에 넣지 않는다(순수 f32).
- 출력: 층마다 `[MOE-DIFF] layer=<trace_layer> M=<tokens> experts=<n> snr=<dB> max_abs_ref=<..> max_abs_got=<..>`. 기대: 커널이 맞으면 u8 활성화 양자화 때문에 **30~45 dB**; 틀리면 **0 dB 근처나 음수**(LFM2에서 fused 경로 142 dB / 67~80 dB 사례 참고: 44·52 문서).
- `NNTR_MOE_SHADOW=1`: DIFF를 계산한 뒤 **참조 값을 output에 덮어쓴다**(l2Shadow와 같은 뜻). 텍스트가 살아나면 HTP MoE 값이 범인, 그대로 잡음이면 MoE 밖(ARM attention/임베딩/lm_head).
- 비용: prefill 447토큰 × 30층 × 128 expert의 니블 읽기·역양자화·f32 행렬곱 — ARM에서 수십 초~수 분. decode도 켜지면 토큰당 8 expert × 30층. 진단용이므로 `num_to_generate`를 16 정도로 줄여 돌린다. 가상 expert(virtual)만 지원하면 된다(Gemma는 virtual; 상주 LFM2는 기존 L2_DIFF가 있다) — `ponytail:` 주석으로 한계 명시.
- 함께 찍으면 좋은 것: 층 입력 `input`의 통계(mean/std/max). 첫 MoE 층부터 입력이 이미 이상(예: std가 수백)하면 **attention 쪽**이 먼저 깨진 것이고, 입력은 정상인데 출력 SNR이 낮으면 **MoE 커널**이다. 이 둘을 한 실행에서 가른다.

### 6.2 ARM 쪽을 가르는 추가 수단 (D가 "MoE 밖"이라고 하면)
- `NNTR_HTP_KEEP_ARM_WEIGHTS`(`htp_compute_ops.cpp` 2902행)가 있다 — FC HTP 경로용. Gemma4는 아직 FC를 CPU에서 하므로 무관.
- 층별 활성화 통계 덤프(`forEachLayer`로 각 층 출력의 mean/std/max를 prefill 뒤 한 번 출력)를 **x86 PC(q40_x86)와 기기에서 같이** 찍어 첫 갈라지는 층을 찾는다. x86이 돌면 가장 빠른 길. x86 segfault는 `ldd build_x86/Applications/CausalLM/nntrainer_causallm | grep -E "stdc\+\+|gomp|openblas"`로 conda 경로가 섞였는지부터.
- Q4_0 tied 임베딩/lm_head: 소스 `tie_word_embedding.cpp` 306~360행(런타임 `repack_q4_0` → blocked, 실패 시 per-row 경로), 임베딩 lookup은 `dequantize_row_q4_0`(267·282행). 레포의 작은 gemma4 설정이 embedding을 **Q6_K**로 둔 점이 눈에 띈다. `--embd_dtype Q6_K`로 재양자화(~30분 + push 5분)가 이 가설의 직접 실험.

## 7. 열려 있는 다른 것들

- **긴 프롬프트 크래시**: ≈1000토큰 프롬프트 + 256 생성 때 `Creating shared tensor of size bigger than tensor memory`(decode 끝 무렵). `init_seq_len`=1024 또는 `sliding_window`=1024 경계 의심. 446토큰에서는 안 난다.
- **x86 rotary 꼬리**: `avx2::compute_rotary_emb_value`의 fp16 꼬리 저장이 head_dim 8(half 4)에서 8 lane을 써 16폭 KV cache 마지막 행을 넘친다(valgrind). 실제 head_dim은 16의 배수라 안 탄다. tiny fixture는 `max_seq_len` 16으로 회피.
- **성능**: C=5에서 prefill 14.9 s(예상 cold 3.7 s), decode 3.8 TPS, peak RSS 3.2 GB(예상 2.3 GiB). 정확도 뒤에 `NNTR_HTP_PROFILE=2`로 분해.
- **Phase 4(projection·dense FFN HTP)**: Gemma4에 엔진 키 미연결. 55 §6.4 예산 주의(+0.77 GiB).
- **dense FFN HTP 경로(kind=1)**는 아직 silu 고정(`invokeMoeLayer` glu=0).
- `NetworkGraph::getTensor`의 `unordered_map::at` 예외는 잡혀서 무해(gdb `catch throw`에 잡히는 노이즈).

## 8. 일하는 규칙 (54 §6~§7 그대로)

- ponytail 모드(`CLAUDE.md`, `01_working_style.md`): 필요한가 → 이미 있나(§6.1의 DIFF/SHADOW 패턴처럼) → 최소 코드. 측정 전 가설로 코드를 바꾸지 않는다.
- **기기 측정은 사용자가 한다.** 한국어 복붙 가이드: 단계별 코드 블록 + 기대 결과 + 실패하면. 실제 경로 사용(`$HOME/workspace/Hexagon_SDK/6.4.0.2`, `$HOME/workspace/hxkl-beta2/hexkl_addon`, `~/workspace/android-ndk-r26d`, 기기 `R3CY10WM83Y`). IDL/DSP 바꾸면 stub·skel·앱 셋 다 재빌드 + skel push.
- 커밋: `git commit -s`, `[component] 제목`, 한 커밋 한 주제, 바꾼 줄만 clang-format-14(`git clang-format --force HEAD -- <files>`), 커밋 전 `bash test/htp/host/run_host_checks.sh`와 호스트 `unittest_causallm_models`. 모델 이름을 커밋·주석에 쓰지 않는다. push는 `claude/epic-hopper-occf31`에만.
- 돌려 보지 못한 것은 "기기 미측정"이라고 쓴다. 결과는 55 §10.x에 날짜·조건·표·판정·다음 순서로.
