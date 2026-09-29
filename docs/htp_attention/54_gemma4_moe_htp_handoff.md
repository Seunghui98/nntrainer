# 54. 다음 과제 인계: Gemma-4 26B-A4B를 모바일에서 — expert 스트리밍 + MM 연산 HTP

작성 2026-09-29 · 출발점 `claude/eager-keller-f91z9o` @ 이 문서가 들어간 커밋 · 결과 정리는 `53_flash_experts_results_summary.md`

이 문서만 읽고 새 세션이 시작할 수 있게 쓴다. 무엇을 만들지, 무엇을 재사용할지, 무엇이 위험한지, 어떻게 일하고 커밋할지, 사용자에게 무엇을 어떻게 넘길지를 담는다.

## 1. 목표

| # | 목표 | 성공 기준 |
|---|---|---|
| G1 | **google/gemma-4-26B-A4B**를 모바일 RAM 안에서 돌린다. MoE expert는 LFM2에서 만든 flash 스트리밍(cached-slim)으로 | 물리 메모리(RSS + 아레나)가 정한 예산 안, 출력이 CPU 참조와 같은 ppl |
| G2 | FFN(expert, dense MLP)과 projection(q/k/v/o 등)의 **행렬곱을 HTP(NPU)로** | prefill·decode가 CPU 경로보다 빠르고, ppl이 CPU 참조와 같은 수준 |
| G3 | 지금까지의 최적화를 그대로 가져간다 | 53의 "바꾼 것" 표 전부 (offset 등록, 선읽기, 제자리 재바인딩, 배치 swap, 캐시 정책 등) |

**하지 않는 것(초기):** attention 본체(softmax·QK·PV)의 HTP 이관, 멀티모달(vision·audio) 타워, 학습. 필요해지면 그때 문서로 따로 연다.

## 2. 먼저 확인할 것 — 이 문서가 모르는 것

이 문서를 쓰는 시점에 **Gemma-4 26B-A4B의 config를 직접 보지 못했다.** 아래 숫자를 추측으로 채우지 말고, Phase 0에서 HF의 `config.json`(`text_config`)에서 읽어 이 표를 채운 뒤 시작한다.

| 필드 | 왜 필요한가 |
|---|---|
| `num_hidden_layers`, `layer_types` | 층 수, sliding/full attention 배치 |
| `enable_moe_block`, `num_experts`, `top_k_experts`, `expert_intermediate_size` | expert 수·크기 → 아레나 칸 크기, prefill에 층당 필요한 expert 수 |
| MoE 층에 dense MLP가 **같이** 있는지 (`intermediate_size`, `use_double_wide_mlp`) | dense FFN도 HTP로 보낼지, 메모리 예산에 상주로 넣을지 |
| `hidden_size`, `head_dim`, `global_head_dim`, `num_attention_heads`, `num_key_value_heads` | projection 형상, KV cache 크기 |
| `hidden_activation` | **HTP MoE 커널은 SwiGLU 전용이다.** Gemma는 `gelu_pytorch_tanh`(GeGLU)일 가능성이 높다 → 커널 에필로그 추가 필요 (§5 R1) |
| `hidden_size_per_layer_input`, `num_kv_shared_layers`, `final_logit_softcapping`, `sliding_window` | 이 레포의 gemma4 구현이 이미 다루는지 확인 |

참고: 레포의 `Applications/CausalLM/models/gemma4/`는 작은 텍스트 모델용이다(config에 `enable_moe_block: false`, `num_experts: null`). **MoE 블록은 아직 구현돼 있지 않다.**

## 3. 재사용할 것 (이미 기기에서 검증됨)

| 구성 요소 | 위치 | 하는 일 | 검증 |
|---|---|---|---|
| expert LRU | `Applications/CausalLM/models/lfm2_moe/expert_lru.h` | 22층 공유 풀, hold/unhold, makeRoom | `unittest_expert_lru` 9/9 |
| 가상 expert + 스테이징 + 선읽기 | `lfm2_moe/lfm2_moe_layer.cpp` (`stage()`, `take_batch`, 선읽기 루프, `expertPrefetchDepth`) | prefill 선읽기(풀이 허용하는 만큼), decode 미스 처리 | 53 전체 |
| HTP 백엔드: 아레나·swap·리더 | `nntrainer/tensor/htp_backend/htp_compute_ops.cpp` (`takeExpertSlot`, `readWeight`, `registerStagedBatch`, `prefetch_qs4cx_wh_experts_begin/_end`, `prefetchReaderLoop`) | ION 아레나 칸, 파일 → 아레나 pread, offset만으로 배치 swap, 리더 스레드 | 53 |
| DSP 레지스트리 | `hmx/hexkl_mm_u8i4_dma.c/.h` (`register_arena`, `rebind_arena`, `tail_from_arena`) | 아레나 가중치를 handle로, 스케일·colsum을 아레나에서 DMA | `HmxArenaSlotReuse.*` 3개 |
| MoE 층 커널 | `hmx/hexkl_mm_u8i4_moe.c` | 한 콜에 층의 모든 expert: gather → gate_up → **SwiGLU** → down → scatter | LFM2 ppl 62.0916 |
| FC/projection 커널 | `hmx/hexkl_mm_u8i4_dma.c` `hexkl_mm_u8i4_layer_run`, 문서 50·51 | q/k/v/o, conv proj, dense FFN을 HTP로 | 문서 51 §2.21 (전부 켬 prefill 618) |
| skel·IDL | `test/htp/nntr_hvx.idl`, `test/htp/nntr_hvx_mm_u8i4.c`, `test/htp/build.sh`, `nntrainer/tensor/htp_backend/generate_stub.sh` | FastRPC 진입점 | — |
| 양자화기 | `Applications/CausalLM/quantize_stream.cpp` (+ `htp_wh_layout.h`) | 메모리 제한 스트리밍 양자화, QS4CX_WH 파일 생성 | LFM2 모델 파일 |
| 엔진 선택 키 | `lfm2/lfm2_causallm.cpp:340~354`, `transformer.cpp:794~` | `nntr_config.json`의 `attn_proj_engine`, `dense_ffn_engine`, `moe_engine` 등으로 층별 CPU/HTP | 문서 50·51 |
| 캐시 시뮬레이터 | `tools/moe_expert_cache_sim.py` + `NNTR_MOE_TRACE` | 정책·C 비교, belady 상한 | C=8에서 기기와 일치 |
| 호스트 체크 | `test/htp/host/run_host_checks.sh` | DSP 코드를 x86에서 스텁으로 검사 (swap, MoE 층, FC) | 매 커밋 |

**재사용 원칙:** lfm2_moe의 코드를 복사해 gemma4용을 따로 만들지 말고, 공통 부분(LRU, 스테이징, 선읽기)을 모델과 무관한 곳으로 옮겨 두 모델이 같이 쓰게 한다. 옮기는 커밋은 동작 변경 없이 따로 두고, LFM2 ppl 62.0916이 그대로인지 확인한 뒤 다음으로 간다.

## 4. 단계 계획

각 단계는 "기기 측정 1회로 판정 가능한 크기"로 자른다. 단계마다 문서에 결과를 적고 다음으로 간다.

| Phase | 할 일 | 끝났다는 기준 |
|---|---|---|
| **0. 조사·예산** | HF config 읽어 §2 표 채우기. 모델 전체·expert 부분·비 expert 부분 크기, expert 1개 크기, 층당 expert 수 계산. 메모리 예산(사용자와 합의) → 가능한 C 범위 표. §5 위험 항목 각각 "해당/비해당" 판정 | 표가 채워진 문서 커밋, 사용자에게 예산 질문 1개 |
| **1. CPU 참조** | gemma4에 MoE 블록 구현(CPU). ppl 기준값 확보 | 기기(또는 호스트)에서 CPU ppl 숫자 1개와 짧은 출력 |
| **2. 모델 파일** | `quantize_stream.cpp`에 gemma4 MoE 지원, QS4CX_WH 파일 생성 | 파일 크기가 Phase 0 계산과 일치, 로더가 읽음 |
| **3. MoE를 HTP로 (cached-slim)** | 공통 부분 추출 커밋 → gemma4 MoE 층을 가상 expert + LRU + 선읽기로. 활성화 함수가 GeGLU면 커널 에필로그 추가(R1) | ppl이 CPU 참조와 같은 수준, C 두 개(warm·cold·decode) 측정 |
| **4. projection·dense FFN을 HTP로** | 엔진 키로 층별 HTP. 형상별 콜 비용 확인(문서 50 §3.6의 transport 교훈) | prefill·decode 개선, ppl 유지 |
| **5. 튜닝·정리** | C 스윕, 시뮬레이터, 53 형식의 결과 문서 | 결과 문서 |

## 5. 알려진 위험 (Phase 0에서 각각 판정)

| # | 위험 | 왜 | 대응 |
|---|---|---|---|
| R1 | **활성화 함수** | MoE 커널이 SwiGLU 고정(`hvx_dequant_swiglu_acc_tiles_to_f32`, `hvx_swiglu_f32`). GELU HVX 커널은 레포에 없다 | GeGLU 에필로그 추가(tanh 근사 GELU × up). 호스트 체크에 참조 비교, 기기 ppl로 판정 |
| R2 | **층당 expert 수가 32보다 많을 때** | prefill 콜은 층의 "토큰이 배정된 모든 expert"를 한 번에 가져간다. 128개면 한 층만으로 풀이 커진다 | 콜을 나누는 `NNTR_MOE_SPLIT` 경로가 있으나 **LFM2에서 split하면 ppl이 62.09 → 63.03으로 변했고 원인 미확인**(기기 테스트 `MoeLayerSplitMatchesWhole` 미실행). split이 필요하면 이 테스트부터 돌려 원인을 먼저 닫는다 |
| R3 | 형상 제약 | K, inter, N은 32의 배수, VTCM 레이아웃(`hexkl_mm_u8i4_moe_layout`), `MOE_MAX_CHUNKS 16` | Phase 0에서 형상으로 layout 함수를 호스트에서 돌려 확인 |
| R4 | handle 수 | DSP 레지스트리 최대 2048(`HEXKL_MM_U8I4_MAX_WEIGHTS`). 풀 칸 × 2 + 상주 FC 가중치 | 계산해 보고 넘으면 상수 조정(DSP 정적 메모리 ~48 B/칸) |
| R5 | 아레나·주소공간 | 256 MiB 청크, rpcmem 크기 int(2 GB-1), DSP 주소공간·힙 한계(문서 50 §3.2~3.3에서 실제로 막힘) | 예산표에 청크 수 포함, 첫 실행은 작은 C로 |
| R6 | cold prefill = flash 바닥 | 이 기기 flash 3.0 GB/s. prefill 중 읽을 바이트 ÷ 3.0이 바닥(53 §5.6) | Phase 0 예산표에 C별 cold 바닥 열을 넣어 사용자가 C를 고를 때 보이게 |
| R7 | attention 차원 | `global_head_dim 512` 등 큰 head는 HTP attention 커널이 안 다룬다 | attention은 CPU 유지(초기 범위 밖) |

## 6. 일하는 방식

### 6.1 ponytail 모드
- **ponytail은 이 환경에 설치된 skill이 아니라 작업 방식이다.** 정의는 `CLAUDE.md`와 `docs/htp_attention/01_working_style.md`. 사용자 PC의 Claude에 `ponytail` skill이 따로 있다면 그걸 불러도 되고, 없으면 두 문서를 읽고 따르면 같다.
- 핵심: 필요한가 → 이미 레포에 있나 → 표준 라이브러리 → 한 줄로 되나 → 그다음에야 최소 코드. 추상화·"나중을 위한" 뼈대 금지. 삭제가 추가보다 낫다.
- 일부러 모서리를 자른 곳은 `ponytail:` 주석으로 한계와 업그레이드 경로를 적는다.
- **줄이면 안 되는 것:** 신뢰 경계의 입력 검증, 데이터 손실을 막는 에러 처리, 하드웨어·보정 손잡이.

### 6.2 이 프로젝트가 비싸게 배운 습관
1. **가설보다 분해 측정이 먼저.** 첫 추정이 틀린 적이 여러 번 있다(16~32%라 했던 것이 실제 92%, "flash 상한" 계산의 GB/GiB 혼동 등). 단계별 프로파일(`NNTR_HTP_PROFILE=2`)로 쪼개 보고 움직인다.
2. **"보기에 맞다"는 검증이 아니다.** 돌려 보지 못했으면 "기기 미측정"이라고 쓴다.
3. **실측이 산술 바닥에 붙어 있는지로 끝을 판정한다**(53 §6). 붙어 있으면 코드를 더 만지지 않는다.
4. **측정 조건을 통제한다.** 상주 실행은 맨 뒤(page cache를 밀어냄), cold는 evict 뒤 여러 번 재서 느린 쪽, warm은 같은 구성 직후. cold는 ±100 ms 흔들린다.
5. **테스트가 실패하면 테스트부터 의심할 것.** `HmxArenaSlotReuse.Swap`은 재바인딩 이후 늘 실패하던 테스트 버그였다.

### 6.3 역할 분담
- **기기 측정은 사용자가 한다.** 에이전트는 구현, 호스트 검사, 한국어 복붙 가이드, 결과 해석, 문서 기록.
- 가이드 형식(지금까지 잘 된 방식):
  - 단계별 코드 블록(빌드 → 설치 → 유닛 테스트 → 측정), 각 블록 아래 "기대 결과"와 "실패하면"
  - 기기 스크립트는 base64 한 줄 + md5 앞 8자리 + `grep -c '^=== '` 개수로 전달 확인
  - 기기가 여러 대면 `export ANDROID_SERIAL=<serial>` 먼저
  - 자리표시자(`<SDK 경로>`)는 zsh가 리다이렉션으로 읽으니 실제 경로를 넣는다 (`$HOME/workspace/Hexagon_SDK/6.4.0.2`, HexKL `$HOME/workspace/hxkl-beta2/hexkl_addon`, NDK `~/workspace/android-ndk-r26d`)
- **IDL이나 DSP 코드를 바꾸면 stub(`generate_stub.sh`)과 skel(`test/htp/build.sh`) 둘 다 다시 빌드하고 skel을 기기에 push.** IDL 주석만 바꿔도 meson이 stub 재생성을 요구한다. 설치 스크립트는 skel을 올리지 않는다.
- 유닛 테스트는 Android 바이너리라 기기에서 돈다(`test/jni`에서 ndk-build → push → `adb shell`).

### 6.4 문서
- 새 과제 문서는 `docs/htp_attention/55_gemma4_moe_htp_task.md`로 시작해 §1(과제) → §10.x(측정 기록) 형식을 52처럼 따른다. 결과가 나오면 53처럼 요약 문서를 따로 만든다.
- `00_START_HERE.md`에 한 줄 추가. 측정 기록은 날짜, 조건, 표, 판정, 다음 순서로.
- 수치에는 단위를 GiB/MiB와 GB로 섞지 않는다(한 번 틀렸다).

## 7. 커밋·브랜치 규칙

| 항목 | 규칙 |
|---|---|
| 브랜치 | 세션이 지정한 브랜치에서만 작업·push. 출발점은 `claude/eager-keller-f91z9o`(이 인프라가 다 들어 있음). main이나 다른 사람 브랜치에 push 금지 |
| 제목 | `[<component>] <subject>` — 예: `[CausalLM/Gemma4] ...`, `[HTP/MoE] ...`, `[HTP/test] ...`, `[Docs] ...` |
| 서명 | `git commit -s` (DCO, 필수) |
| 트레일러 | 에이전트가 쓴 커밋은 `Co-Authored-By:` 트레일러 (세션이 알려 주는 형식 그대로). 모델 이름을 커밋 본문·코드 주석에 쓰지 않는다 |
| 포맷 | 바꾼 줄만 clang-format-14 (`git clang-format-14 <base>`; 없으면 `git clang-format --force HEAD -- <files>`) |
| 단위 | 한 커밋 = 한 주제. 코드 이동(동작 불변)과 동작 변경을 섞지 않는다. 문서 기록은 `[Docs]` 커밋으로 따로 |
| 본문 | 왜 바꿨는지, 무엇을 쟀는지, 기기 미측정이면 그렇다고 |
| 금지 | `subprojects/` 수정, 테스트를 끄거나 건너뛰어 통과시키기, 빈 커밋 |
| PR | 사용자가 요청할 때만. `.github/PULL_REQUEST_TEMPLATE.md` 형식(커밋별 `<details>`, Self evaluation, Signed-off-by). `CLAUDE.md`는 작업 브랜치 전용이라 PR에 넣지 않는다 |
| 검사 | 커밋 전 `bash test/htp/host/run_host_checks.sh`, 바꾼 C++는 구문 검사(호스트에는 Hexagon SDK가 없으니 IDL에서 생성한 헤더 스텁으로) |

## 8. Phase 0 산출물 형식 (예시 틀)

| 항목 | 값 | 근거 |
|---|---|---|
| MoE 층 수 × 층당 expert | ? × ? | config |
| expert 1개 크기 (int4 WH + 스케일·colsum) | ? MiB | 계산 |
| expert 전체 / 비 expert 전체 | ? GB / ? GB | 계산 |
| 상주 시 물리 메모리 | ? GiB | 계산 |
| C별 아레나·물리 합계·prefill 중 읽을 바이트·cold 바닥 | 표 | 53 §3.2, §5.6 방식 |
| R1~R7 판정 | 해당/비해당 | 근거 한 줄씩 |

이 표와 "메모리 예산을 몇 GiB로 할까요?" 질문 하나로 Phase 0을 끝낸다.
