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

## 9. 복붙 가이드 (2026-10-01) — 재배치한 가중치로 다시 재고, 안 된 부분만 DIFF로 가른다

**먼저 읽을 것: 55 §10.5.** §1~§8이 "기기에서만 도는 코드"를 용의자로 지목했지만, 호스트에서 확정된 원인은 **PC 변환 경로**다. FP32 중간 파일의 MoE 블록 순서가 양자화기가 위치로 읽는 순서와 달라서, 층마다 router norm gamma가 router 행렬의 앞 22행이 되고 expert마다 gate|up이 섞여 들어갔다. 층당 float 수가 양쪽 같아서 아무 검사도 걸리지 않았다. 아래 1~4가 그 수정을, 5~7이 그래도 남는 것을 가른다.

**IDL은 바뀌지 않았다** → stub·skel 재빌드 불필요, skel push 불필요. 앱(`libcausallm_core.so`, `nntrainer_causallm`)만 다시 만든다. (IDL을 건드리는 변경이 생기면 §3의 주의대로 stub·skel·앱 셋 다.)

기기가 두 대 붙어 있으니 맨 앞에 한 번:
```bash
export ANDROID_SERIAL=R3CY10WM83Y
adb devices
```
기대 결과: `R3CY10WM83Y device` 한 줄이 보인다.

### 1단계 — 브랜치와 호스트 검사

```bash
cd ~/workspace/nntrainer
git fetch origin claude/gemma4-accuracy-diff
git checkout claude/gemma4-accuracy-diff
ninja -C build
bash test/htp/host/run_host_checks.sh
./build/Applications/CausalLM/unittest_causallm_models 2>&1 | tail -3
./build/test/unittest/unittest_nntrainer_cpu_backend --gtest_filter='*wh_pack*'
```
기대 결과: 호스트 검사 5개 모두 `... OK`, `unittest_causallm_models`는 `[  PASSED  ] 87 tests.`(+ fixture 없는 14개 SKIPPED, FAILED 0), `wh_pack_unpacks_like_the_load_path` 통과.
실패하면: `git submodule update --init --depth 1` 뒤 `ninja -C build`를 다시. 87이 아니고 FAILED가 있으면 그 테스트 이름을 알려 줄 것 — 재양자화는 하지 말 것.

### 2단계 — 재배치한 FP32를 다시 양자화 (PC, 약 40분)

재배치 파일은 이미 PC에 있다(`res/gemma4_26ba4b/nntr_gemma4_fp32_fixed.safetensors`, 100.93 GB). 양자화기는 이제 입력 헤더를 보고 자기가 읽는 순서와 다르면 **dry run에서** 멈춘다.

```bash
cd ~/workspace/nntrainer
R=Applications/CausalLM/res/gemma4_26ba4b
cp $R/nntr_config.json $R/nntr_config_fixed.json
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("Applications/CausalLM/res/gemma4_26ba4b/nntr_config_fixed.json")
c = json.loads(p.read_text())
c["model_file_name"] = "nntr_gemma4_fp32_fixed.safetensors"
p.write_text(json.dumps(c, indent=2))
PY
./build/Applications/CausalLM/nntr_quantize_stream $R \
  --config $R/nntr_config_fixed.json \
  -o $R/q40_fixed \
  --fc_dtype Q4_0 --moe_dtype QS4CX_WH --embd_dtype Q4_0 --isa ARM
ls -l $R/q40_fixed/
```
기대 결과: `Quantized layer 1/30` … `30/30` 뒤 `Output size: 12336 MiB` 근처, `Streaming quantization complete`. 출력 `.bin`이 12,9xx,xxx,xxx B(= 12,336 MiB)이고 §10.3의 옛 파일과 **크기가 같다**(바뀐 것은 값이 아니라 순서다).
실패하면:
- `Input layout mismatch at tensor <n>: ...` → 재배치가 덜 됐거나 잘못된 파일을 가리켰다. 그 줄을 그대로 알려 줄 것. 옛 파일(`nntr_gemma4_fp32.safetensors`)로 돌리면 `tensor 15`에서 같은 메시지가 나오는 게 정상이다.
- 디스크가 모자라면(`df -h /`에서 20 GB 미만) 옛 `q40/nntr_gemma4_q40_arm.bin`을 먼저 지운다. 기기에 있는 것과 같은 깨진 파일이다.

### 3단계 — 앱 재빌드 (PC, 약 15분)

```bash
cd ~/workspace/nntrainer/Applications/CausalLM
export HEXAGON_SDK_ROOT=$HOME/workspace/Hexagon_SDK/6.4.0.2
export HEXKL_ROOT=$HOME/workspace/hxkl-beta2/hexkl_addon
export ANDROID_NDK=~/workspace/android-ndk-r26d
./build_android.sh --htp
```
기대 결과: `[SUCCESS] Build completed successfully!`와 `libcausallm_core.so`, `nntrainer_causallm` 두 줄의 `[OK]`.
실패하면: `HexKL addon not found` → `HEXKL_ROOT` 확인. `generate_stub.sh` 에러 → `HEXAGON_SDK_ROOT` 확인. 스크립트가 "this does NOT rebuild libnntr_hvx_skel.so"라고 알리는 것은 정상이다(이번엔 skel이 그대로여도 된다).

### 4단계 — 기기에 올리고 ppl을 다시 잰다

기기 `/data/user/0` 여유가 16 GB뿐이라 모델은 **같은 경로에 덮어쓴다**(지금 올라가 있는 파일이 깨진 그 파일이다).

```bash
cd ~/workspace/nntrainer/Applications/CausalLM
./install_android.sh
M=/data/local/tmp/nntrainer/causallm/models/gemma4-26b-a4b-qs4cx-wh
adb push res/gemma4_26ba4b/q40_fixed/nntr_gemma4_q40_arm.bin $M/nntr_gemma4_q40_arm.bin
adb shell "ls -l $M/nntr_gemma4_q40_arm.bin"
```
기대 결과: push가 12.9 GB를 올리고(약 10~20분), 기기의 크기가 PC 파일과 같다.
실패하면: `No space left` → `adb shell rm $M/nntr_gemma4_q40_arm.bin` 뒤 다시 push.

```bash
adb shell "cd /data/local/tmp/nntrainer/causallm && \
  LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. NNTR_NUM_THREADS=8 \
  NNTR_MOE_CACHE_EXPERTS=5 NNTR_PPL=1 \
  ./nntrainer_causallm ./models/gemma4-26b-a4b-qs4cx-wh" 2>&1 | tail -40
```
기대 결과(**이번 과제의 판정 기준**): `[PPL] prompt tokens=446 nll/token=<한 자리>`이고 생성이 영어 문장이다. §4의 13.2649(균등분포 12.48보다 나쁨)에서 내려와야 한다.
- 한 자리로 내려오고 문장이 읽히면: 원인은 §10.5의 변환 순서 하나였다. 5~6단계는 돌릴 필요 없고, 7단계만 돌려 커널 게이트를 닫는다.
- 여전히 12~13이면: 남은 원인이 있다. 5단계로(단, 5단계의 "이 도구가 볼 수 없는 것"을 먼저 읽을 것).
- 중간이면(예: 9~11, 문장 조각): 두 원인이 겹쳐 있다. 5단계로.

실패하면: `AEE_EBADPARM`이 첫 콜에서 나면 skel과 앱의 IDL이 어긋난 것이다(이번 변경은 IDL을 안 건드렸으니, 그렇다면 3단계 앱이 아니라 기기의 옛 skel이 원인 — §3 경로로 skel을 다시 올린다). `Creating shared tensor of size bigger than tensor memory`는 프롬프트가 1024에 걸린 별개 문제(§7)이니 446 토큰 설정을 그대로 쓴다.

### 5단계 — 그래도 잡음이면: 한 실행으로 층별 판정 (`NNTR_MOE_DIFF`)

4단계가 잡음일 때만 돌린다. MoE 층마다 **앞 8토큰**을 CPU에서 f32로 다시 계산해 HTP 결과와의 SNR을 찍고, 그 층 **입력의 mean/std/max**를 같이 찍는다. 생성은 짧게.

```bash
adb shell "cd /data/local/tmp/nntrainer/causallm && \
  LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. NNTR_NUM_THREADS=8 \
  NNTR_MOE_CACHE_EXPERTS=5 NNTR_PPL=1 NNTR_MOE_DIFF=8 \
  ./nntrainer_causallm ./models/gemma4-26b-a4b-qs4cx-wh" 2>&1 | grep -aE "MOE-DIFF|\[PPL\]|FATAL" | head -40
```
기대 결과: 층 0~29마다 한 줄,
`[MOE-DIFF] layer=0 M=447 rows=8 experts=NN snr=XX.XdB max_abs_ref=.. max_abs_got=.. in_mean=.. in_std=.. in_max=..`

읽는 법 — 이 두 숫자가 "MoE 커널"과 "그 앞 attention"을 가른다:

| 입력 통계 | SNR | 판정 |
|---|---|---|
| std가 정상 범위(대략 0.1~10), 층마다 서서히 증가 | 30~45 dB | MoE 커널 정상. 범인은 MoE 밖(attention·임베딩·lm_head) |
| 정상 | 0 dB 근처·음수 | **MoE 커널**. 층 번호와 SNR을 알려 줄 것 |
| **첫 몇 층부터 std가 수백~수천, 또는 inf/nan** | 아무 값 | **attention 쪽이 먼저 깨진다.** 가장 낮은 layer 번호가 출발점 |
| 층이 올라가며 std가 발산 | 점점 낮아짐 | 누적. 처음 꺾이는 층을 본다 |

**이 도구가 볼 수 없는 것(§10.5가 가르쳐 준 것):** 참조는 커널이 읽는 **바로 그 파일 바이트**를 읽는다. 따라서 파일 자체가 틀리면 참조와 커널이 **똑같이** 틀려서 SNR은 높게 나온다. 모든 층이 30~45 dB인데 출력이 여전히 잡음이면 그것은 "다 정상"이 아니라 **"커널은 파일과 일치한다"**는 뜻뿐이고, 남은 용의자는 (a) 파일의 값 자체, (b) MoE 밖 경로다. (a)의 가장 싼 실험은 §10.5의 남은 가정 하나 — 재배치에서 expert의 gate와 up을 바꿔 2단계부터 다시 하는 것(한 줄, PC 40분 + push).

또 하나: 참조 자체에는 호스트 테스트가 없다(이 경로는 기기 전용이라 호스트에서 돌 수 없다. `whUnpack`의 왕복만 검사된다). **모든** 층이 0 dB 근처인데 7단계 gtest가 전부 0이면, 커널이 아니라 참조를 의심할 것.

실패하면: `[MOE-DIFF] layer=.. expert=..: read failed` → 모델 파일 fd를 못 읽었다(push가 덜 됐는지 확인). `has no file fd` → `NNTR_MOE_CACHE_EXPERTS`가 빠져 expert가 상주 모드다(5를 꼭 넣는다). 한 층에 수 초 걸리는 게 정상이다(ARM에서 f32로 다시 계산한다).

### 6단계 — 값이 범인인지 부작용인지 (`NNTR_MOE_SHADOW`)

5단계에서 SNR이 낮게 나왔을 때만. **모든** 행의 참조를 계산해 모델에 그 값을 대신 넘긴다. prefill 한 층에 수 분이니 생성은 짧게 둔다.

```bash
adb shell "cd /data/local/tmp/nntrainer/causallm && \
  LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. NNTR_NUM_THREADS=8 \
  NNTR_MOE_CACHE_EXPERTS=5 NNTR_PPL=1 NNTR_MOE_SHADOW=1 \
  ./nntrainer_causallm ./models/gemma4-26b-a4b-qs4cx-wh" 2>&1 | grep -aE "MOE-DIFF|\[PPL\]|FATAL" | tail -10
```
기대 결과: ppl이 나오기까지 아주 오래 걸린다(수십 분). 판정은 두 갈래뿐이다.
- **텍스트가 살아나고 nll이 내려간다** → HTP MoE가 내는 **값**이 범인. 5단계의 층·SNR이 쫓을 대상.
- **그대로 잡음** → MoE 값은 범인이 아니다. 범인은 MoE 밖(ARM attention head 512·GQA 8, Q4_0 임베딩/lm_head) 또는 MoE 콜의 부작용(버퍼 aliasing 등)이다.

실패하면: 너무 오래 걸려 못 기다리겠으면 중단해도 된다 — 5단계의 표만으로도 다음 수가 정해진다.

### 7단계 — 아직 안 돌린 기기 gtest 두 개

4단계 결과와 무관하게 한 번은 돌려 둔다. 둘 다 커널 쪽 용의자(§6 A)를 닫는 게이트고, 지금까지 **미실행**이다.

```bash
cd ~/workspace/nntrainer
export HEXAGON_SDK_ROOT=$HOME/workspace/Hexagon_SDK/6.4.0.2
export HEXKL_ROOT=$HOME/workspace/hxkl-beta2/hexkl_addon
export HEXKL_SDK_VER=6.4.0.2
export ANDROID_NDK=~/workspace/android-ndk-r26d
export DEFAULT_HEXAGON_TOOLS_ROOT=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.04
bash test/htp/run_u8i4_layer_on_device.sh 2>&1 | tail -30
grep -aE "GEGLU_DET_FIELD|U8I4_FIELD path=moe_layer" /tmp/hvx_softmax_device_run.log /tmp/hvx_mm_u8i4_device_run.log
```
기대 결과:
- `HvxSwigluDet.GegluMatchesScalarBitExact` 통과, `GEGLU_DET_FIELD bad_exp=0 bad_recip=0 bad_out=0 of <N>` — HVX GeGLU가 호스트 스칼라와 비트 일치.
- `MoeLayerMatchesTwoCallReference`가 `lfm2`와 `gemma4`(K 2816, inter 704) 두 형상 모두 통과, `U8I4_FIELD path=moe_layer field=bad_elems value=0`.
- 둘 다 0이면 HTP MoE 커널은 이 입력 분포에서 혐의를 벗는다.

실패하면:
- `DEFAULT_HEXAGON_TOOLS_ROOT ... does not exist` → `ls $HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/`로 버전 디렉터리 이름을 확인해 바꿔 넣는다(§3의 그 함정).
- `bad_out`이나 `bad_elems`가 0이 아니면 **그게 커널 버그다.** 숫자와 형상(`gemma4`인지 `lfm2`인지)을 알려 줄 것.
- 이 스크립트는 skel을 새로 빌드해 `/data/local/tmp/htp_u8i4_layer_test`에만 올린다. 앱이 쓰는 `/data/local/tmp/nntrainer/causallm/libnntr_hvx_skel.so`는 건드리지 않으니 4단계와 섞이지 않는다.

### 보내 줄 것

1~4단계는 각 블록의 마지막 출력 그대로, 5·7단계는 `MOE-DIFF` 줄 전체와 `*_FIELD` 줄. 4단계에서 한 자리 nll과 영어 문장이 나오면 거기서 멈추고 알려 주면 된다 — 나머지는 55 §10.6에 기록하고 닫는다.
