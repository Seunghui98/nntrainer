# DDTree 검증 런북 (VERIFY.md)

모든 작업은 Docker 컨테이너 **`my-dev`** 안에서 한다. 호스트에서 `docker exec my-dev ...`.
- repo: `/workspace/nntrainer` (branch `add/DDTree`)
- 검증 자산/venv: `/workspace/qwen3run/`
- 빌드 환경: gcc-10, `-Denable-fp16=false`, `ninja -j4` (RAM 7.8G라 -j4 필수)

## 핵심 개념
DDTree(tree speculative decoding)는 **lossless** → greedy와 같은 토큰이 나와야 정상.
검증 목표 = (1) nntrainer DDTree 코어 == Python `ddtree.py` (값까지), (2) DDTree 디코드 == greedy, (3) nntrainer forward == HF.

## 1) 한눈에 (먼저 실행)
    docker exec my-dev /workspace/qwen3run/venv/bin/python /workspace/qwen3run/demo_verify.py
기대: (A) DDTree e2e == greedy 64/64, (B) nntrainer C++ 트리 == Python 트리 (모든 필드 동일).

## 2) 전체 자동 (① 단위테스트 ②Task9 ③Task10 ④60케이스 ⑤logw ⑥e2e)
    docker exec my-dev bash /workspace/qwen3run/verify_all.sh
기대: 19 PASSED / compactTail PASSED / mask PASSED / `PARITY: 60/60 ALL-FIELDS IDENTICAL` /
worst max|Δlogw|=3.8e-6 / e2e 텍스트 == greedy.

## 3) 항목별
- 60케이스 트리 패리티: `docker exec my-dev bash -c "cd /workspace/qwen3run && ./venv/bin/python run_batch.py"`
- 단일 트리 전필드: `... py_tree.py && ./ddtree_parity 15 151936 31 198 18 18 trace/draft_logits.f32 trace/cpp_tree.json && ... compare_trees.py`
- logw 값: `... logw_compare.py`
- DDTree==greedy: `... e2e_py.py`
- nntrainer vs HF base: `docker exec my-dev /workspace/nntrainer/build-app/Applications/CausalLM/nntr_causallm /workspace/nntrainer/Applications/CausalLM/res/qwen3/qwen3-0.6b` + `... hf_ref.py`
- C++ unit tests: `build/test/unittest/unittest_ddtree`, `build-app/Applications/CausalLM/unittest_{kv_cache_manager,mha_core_mask}`

## 값 직접 열람
    docker exec my-dev cat /workspace/qwen3run/trace/cpp_tree.json   # nntrainer 트리
    docker exec my-dev cat /workspace/qwen3run/trace/py_tree.json    # Python 트리
    docker exec my-dev cat /workspace/qwen3run/e2e/py_e2e.json       # e2e 토큰+accept

## 스크립트 역할 요약
- rebuild_ref.py -> ddtree_ref.py : ddtree.py 함수 6개 verbatim 추출(기준)
- capture_logits.py / capture_many.py : qwen3로 draft logits(1개 / 60케이스) 생성
- py_tree.py / ddtree_parity(.cpp) : Python / nntrainerC++ 트리 덤프
- compare_trees.py / run_batch.py : 트리 구조 1:1 / 60케이스 비교
- ddtree_logw(.cpp) / logw_compare.py : 노드 logw 값 비교
- hf_ref.py : HF greedy 기준 ; e2e_py.py : Python DDTree 디코드 루프
- demo_verify.py : (A)(B) 시연 ; verify_all.sh : 원샷

## 컨테이너가 꺼져 있으면
    docker start my-dev
빌드가 없으면(드묾): 메모리 `ddtree-nntrainer-build-env.md` 참고해서
`CC=gcc-10 CXX=g++-10 meson setup --wipe build -Denable-fp16=false -Denable-test=true -Denable-tflite-backbone=false -Denable-tflite-interpreter=false -Denable-app=false && ninja -j4 -C build`,
앱은 build-app(enable-app, enable-transformer). venv 깨졌으면 /workspace/qwen3run/venv 재생성(torch cpu + transformers>=4.51).

## 아직 안 된 것
nntrainer에서 DDTree 디코드 루프 전체(verify forward까지)를 도는 런타임은 미구현(Stage B:
lm_head 전체위치 출력 + harness 필요). 코어는 위처럼 Python과 값까지 검증됨.
