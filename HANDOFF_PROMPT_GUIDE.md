# 다음 에이전트를 위한 프롬프팅 가이드

## 추천 프롬프트 (복사해서 사용)

---

### 프롬프트 (전체 작업 맡길 때)

```
nntrainer에 PE-Lang-L14-448 encoder (ScreenAI caption v3.0.0-C62) 포팅 작업을 이어받아서 완성해라.

## 필독 (순서대로)
1. /home/leeseunghui/workspace/nntrainer/HANDOFF_PELANG.md — 이전 에이전트의 핸드오프 문서. 완료된 작업, 현재 블로커, 남은 작업이 전부 정리되어 있다.
2. /home/leeseunghui/workspace/nntrainer/PELANG_L14_448_SUPPORT_PLAN.md — 원래 작업 지시서. Phase 1-7 + G1-G11 게이트 + 함정 15개.
3. /home/leeseunghui/workspace/models/pelang_golden/meta.json — 검증용 golden 메타.

## 현재 상태
- 브랜치: feature/pelang-l14-448-encoder (체크아웃 완료)
- Phase 1-4 코드 전부 작성됨, x86 fp32 빌드 성공
- 런타임 블로커 1건: model->compile()에서 "Creating shared tensor of size bigger than tensor memory" 에러
- 그래프 생성(constructModel)은 전부 성공, compile 단계에서 메모리 플래닝 실패
- 0-layer 테스트에서도 같은 에러 → 블록 구조가 아니라 patch embed / weight layer 구조 자체가 원인

## 블로커 해결 (최우선)
HANDOFF_PELANG.md §2의 "의심 원인" 4가지를 순서대로 검증:
1. weight layer에 input 텐서를 인자로 넘기는 패턴 (pe_rope_sin/cos이 [1,1,1024,64]인데 input이 [1,3,448,448])
2. concat layer axis=2 ([1,1,1,1024] + [1,1,1024,1024] → [1,1,1025,1024])
3. pe_rope 3-input layer의 메모리 aliasing
4. layer_scale 2-input + 1-weight의 메모리 aliasing

디버깅 방법:
- 0-layer 그래프에서 pe_rope_sin/cos weight layer를 제거해 보기
- concat을 addition으로 임시 교체해 보기
- nntrainer/graph/network_graph.cpp의 compile 경로에 DBG 추가
- model->compile() 직전에 graph 구조 dump

## 블로커 해결 후 순서
1. pelang_vision_encoder.cpp의 [PE-Lang DBG] 프린트 전부 제거
2. G2: encodePixels(pixel.npy) vs s0/s1/s2/s3 — cos>0.9999, maxdiff<1e-3
3. G3: encoder_hidden.npy vs nntr 출력 — cos>0.9999, rel-L2<1e-3
4. G4: --decoder-init-parity — argmax==2048, cos>0.999
5. G4 통과하면 멈추고 보고 — ARM/양자화/성능(Phase 5-7)은 별도 에이전트 스코프

## 빌드 & 실행
```bash
cd ~/workspace/nntrainer
meson setup --reconfigure build -Denable-transformer=true -Denable-fp16=false -Dthread-backend=omp -Denable-app=true -Denable-test=false
ninja -C build Applications/CausalLM/nntr_causallm

export LD_LIBRARY_PATH=$PWD/build/nntrainer:$PWD/build/Applications/CausalLM/layers:$PWD/build/Applications/CausalLM/models/gpt_oss:$PWD/build/Applications/CausalLM/models/qwen3_moe:$PWD/build/Applications/CausalLM/models/qwen3_slim_moe:$PWD/build/Applications/CausalLM/models/gpt_oss_cached_slim:$PWD/build/Applications/CausalLM/models/qwen3_cached_slim_moe:$LD_LIBRARY_PATH

./build/Applications/CausalLM/nntr_causallm --dump-encoder Applications/CausalLM/res/pelang-encoder --input-pixels ~/workspace/models/pelang_golden/pixel.npy
```

## 규칙
- 계획서와 실제 코드가 다르면 코드를 믿고 차이를 보고할 것
- 계획서 §6 함정 15개를 구현 전에 반드시 읽을 것
- 프로파일링 없이 커널부터 최적화하지 말 것
- 기존 SigLIP2 경로는 지우지 말고 config로 분기할 것 (G10 회귀)
- 디버딩은 1단계 greedy만 구현 (beam=3은 2단계)
- ponytail 모드: 최소 diff, stdlib/기존 코드 우선, 불필요한 추상화 금지
```

---

### 프롬프트 (블로커 해결만 맡길 때 — 더 짧은 버전)

```
nntrainer PE-Lang encoder 포팅의 런타임 블로커를 해결해라.

## 상황
- 브랜치: feature/pelang-l14-448-encoder
- 빌드는 성공, 런타임에 model->compile()에서 실패
- 에러: "Creating shared tensor of size bigger than tensor memory"
- 위치: nntrainer/tensor/tensor_base.cpp:228 (getSharedDataTensor)
- 그래프 생성은 전부 성공 (23블록 + projection), compile 단계에서 메모리 플래닝 실패
- num_hidden_layers=0으로 설정해도 같은 에러 → patch embed 구조 자체가 원인

## 필독
- /home/leeseunghui/workspace/nntrainer/HANDOFF_PELANG.md §2 (블로커 분석)
- Applications/CausalLM/models/pelang/pelang_vision_encoder.cpp (createPatchEmbed, constructModel)
- Applications/CausalLM/models/siglip2/siglip2_vision_encoder.cpp (작동하는 참조 구현)

## 의심 원인
1. weight layer에 input([1,3,448,448])을 넘기는데 weight dim이 [1,1,1024,64]인 것 (pe_rope_sin/cos)
2. concat(axis=2)으로 [1,1,1,1024] + [1,1,1024,1024]를 합치는 것
3. 3-input 커스텀 레이어(pe_rope)의 메모리 aliasing
4. 2-input+1-weight 커스텀 레이어(layer_scale)의 메모리 aliasing

## 해결 조건
- compile() 성공 + incremental_inference 실행 가능
- 기존 SigLIP2 경로 회귀 없음 (G10)
- 원인을 보고할 것

## 빌드
cd ~/workspace/nntrainer && ninja -C build Applications/CausalLM/nntr_causallm
export LD_LIBRARY_PATH=$PWD/build/nntrainer:$PWD/build/Applications/CausalLM/layers:$PWD/build/Applications/CausalLM/models/gpt_oss:$PWD/build/Applications/CausalLM/models/qwen3_moe:$PWD/build/Applications/CausalLM/models/qwen3_slim_moe:$PWD/build/Applications/CausalLM/models/gpt_oss_cached_slim:$PWD/build/Applications/CausalLM/models/qwen3_cached_slim_moe:$LD_LIBRARY_PATH
./build/Applications/CausalLM/nntr_causallm --dump-encoder Applications/CausalLM/res/pelang-encoder --input-pixels ~/workspace/models/pelang_golden/pixel.npy
```

---

## 프롬프팅 팁

### 1. 컨텍스트를 주되 너무 많이 주지 말 것
위 프롬프트는 "필독 3개 + 현재 상태 + 블로커 + 빌드 명령" 구조. 에이전트가 직접 파일을 읽게 하는 것이 코드를 프롬프트에 붙여넣는 것보다 효율적.

### 2. "의심 원인"을 명시하되 정답을 주지 말 것
4가지 의심 원인을 순서대로 주되, 정답을 알려주지 않으면 에이전트가 직접 검증하면서 구조를 이해함. 정답을 주면 그것만 확인하고 넘어감.

### 3. "해결 조건"을 명확히 할 것
- compile 성공
- incremental_inference 실행 가능
- G10 회귀 없음
이 3가지가 "done"의 기준. 모호하면 에이전트가 자기 기준으로 "해결됐다"고 보고함.

### 4. ponytail 모드를 켜라
코드를 더 만들기보다 고치는 작업이므로, "최소 diff" 원칙이 중요. 특히 nntrainer 코어(graph planner 등)를 고치게 되면 회귀 리스크가 크니까 "코어 수정 없이 레이어/모델 쪽에서 해결"이라는 제약을 주는 것이 좋음.

### 5. 디버그 프린트 제거를 잊지 말 것
현재 pelang_vision_encoder.cpp에 `[PE-Lang DBG]` 프린트가 20개 정도 남아 있음. 블로커 해결 후 반드시 제거해야 함. 프롬프트에 명시.

### 6. 에이전트 선택
- 블로커 해결: `deep` 카테고리 (자율적 문제 해결) 또는 `ultrabrain` (로직 헤비)
- G2-G4 검증: `quick` 또는 `unspecified-low` (실행 + 비교만)
- 전체 이어받기: `deep` 카테고리로 블로커 먼저, 해결 후 `quick`으로 G2-G4
```
