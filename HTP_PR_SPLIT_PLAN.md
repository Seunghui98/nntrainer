# HTP PR 분할 · 리뷰 순서 계획

`#4256` (Flash Attention with HTP, 18,776 lines / 76 files / 94 commits) 까지
올리기 위한 PR 분할과 머지 순서. `#4226` 은 머지 완료(main `ff2f556`).

---

## 1. 현재 상태 진단

| PR | 표시 diff | 실제 새 내용 | 문제 |
|----|-----------|--------------|------|
| #4226 | 275 / 6 files | — | ✅ 머지됨 |
| #4236 | 1,235 / 12 files | 1,238 / 13 files | ✅ base 정상, 그대로 리뷰 가능 |
| #4245 | 3,551 / 26 files | 2,089 / 20 files | ❌ #4226·#4236 중복 포함, 3개 주제가 한 PR |
| #4244 | 3,360 / 24 files | 788 / 14 files | ❌ base 가 `main` (본문은 `htp/u8i4-dma-cross` 기준이라고 적혀 있음) |
| #4249 | 3,841 / 26 files | 613 / 13 files | ❌ base 가 `main`, #4244 내용 전부 중복 |
| #4256 | 18,776 / 76 files | ~5,000 (코드) | ❌ 문서 7,000줄 + 오래된 base + 계측 커밋 혼재 |

### 근본 원인 두 가지

1. **base 가 전부 `main`(구버전 `f97c2e26`)** 이라 스택이 안 잡혀 있음.
   그래서 #4245/#4244/#4249 가 서로의 커밋을 통째로 다시 보여 준다.
   리뷰어 입장에서는 같은 코드를 3~4번 보게 된다.
2. **#4245 안에 독립적인 주제 3개**가 섞여 있음 (softmax / u8i4 layer path /
   session 정리).

---

## 2. 분할 결과 (push 완료 · 2026-08-11 #4236 `2f22a22` 기준으로 리베이스됨)

`origin = Seunghui98/nntrainer`

```
main (9a3b5b8, #4226 포함)
 │
 ├─ PR-1  #4236  hvx_mm_op                        [READY2MERGE]  7 commits  +1,222
 │   │            u8i4(A8W4) 정확도 파이프라인
 │   │
 │   ├─ PR-2  claude/pr-split-cleanup-plan-42mprx-02-hvx-softmax
 │   │            6 commits  +914   HVX exp / row-wise softmax
 │   │
 │   └─ PR-3  claude/pr-split-cleanup-plan-42mprx-03-u8i4-dma-layer
 │       │        5 commits  +1,175  DMA ring · weight registry · session · layer endpoint
 │       │
 │       └─ PR-4  claude/pr-split-cleanup-plan-42mprx-04-u8i8-layer
 │           │        5 commits  +788   u8i8 mirror  (= #4244 내용만)
 │           │
 │           └─ PR-5  claude/pr-split-cleanup-plan-42mprx-05-quant-opt
 │                    4 commits  +613   async copy-out · vectorized quant · worker pool  (= #4249 내용만)
 │
 └─ PR-6  (PR-2 + PR-3 머지 후) session calling-convention 통일  2 commits  ~40줄
```

### 각 PR 내용

**PR-2 — HVX softmax** (base: #4236)
```
[test]  Add FastRPC entries for the HVX softmax bring-up
[htp]   Add a vector exp for f32 lanes on HVX
[htp]   Add a row-wise f32 softmax on HVX with tail handling
[test]  Cover multi-row, in-place, negative scale and row ranges for softmax
[test]  Unify LANES and put the softmax IDL comments in English
[chore] Fix doxygen tag on multi-line comments in HVX softmax/exp/add
```
IDL 안 한국어 주석은 영문으로 바뀐 상태로 포함되어 있음. u8i4 쪽과 파일 의존성이
없어 PR-3 과 **병렬 리뷰 가능**.

**PR-3 — u8i4 DMA / registry / session / layer endpoint** (base: #4236)
```
[HTP]  Add a dmlink user-DMA ring for weight staging
[HTP]  Bake u8i4 weights once and prefetch across a layer
[HTP]  Hold HMX for the session and add the u8i4 layer endpoint
[test] Cover the u8i4 layer endpoint
[test] Add an end-to-end device runner for the u8i4 layer endpoint
```

**PR-4 — u8i8 mirror** (base: PR-3) — 기존 #4244 를 이 내용으로 갱신하면 됨
**PR-5 — quant/dequant 최적화** (base: PR-4) — 기존 #4249 를 이 내용으로 갱신하면 됨

**PR-6 — session calling convention 통일** (PR-2, PR-3 둘 다 머지된 뒤)
```
[htp] Take the session handle in the softmax entries
[htp] Check for HVX once per session instead of per call
```
PR-2 와 PR-3 를 병렬로 가져간 대가. `test/htp/build.sh` 의 `SRCS` 한 줄과
`nntr_hvx.idl` 엔트리 순서 충돌만 있고(확인 완료), 나머지는 자동 머지된다.
나중에 머지되는 쪽에서 이 충돌만 풀면 된다.

---

## 3. #4256 을 여기서 더 쪼갤 방법

PR-5 까지 머지된 뒤 `#4256` 에 남는 것은 코드 ~5,000줄 + 문서 ~7,000줄.

### 3-1. 먼저 버려야 할 것 (리뷰에 올리면 안 되는 것)

| 대상 | 줄 수 | 이유 |
|------|-------|------|
| `docs/htp_attention/*` 20개 파일 | ~6,100 | 내부 작업 노트. `00_START_HERE`, `01_working_style`, `12_prompt_kit`(agent 프롬프트), `13_htp_pr_plan` 은 upstream 성격이 아님. 한국어 설계 문서(`c2e304e`)도 포함 |
| `CLAUDE.md` | 63 | 에이전트 설정 파일 |
| `tools/htp_fc_report.py`, `tools/htp_attn_report.py` | 1,516 | 측정 리포트 스크립트. 필요하면 계측 PR 로 따로 |
| `2fa199c` ↔ `28f98fc` revert 쌍 | — | "Keep the S band in VTCM" 후 즉시 revert. 두 커밋 다 drop |
| 오래된 base 로 인한 역행 diff | -3,790 | kleidiai, CausalLM repack, `tensor.cpp`, `nntrainer.spec` 등 이미 main 에 있는 작업이 삭제로 잡힘 → **rebase 필수** |

문서 중 살릴 가치가 있는 건 설계 근거가 들어간
`ref_08_attention_hmx_design.md`, `35_hmx_hvx_overlap.md`, `31_dataflow_as_built.md`
정도. 이건 별도 docs PR 로 마지막에.

### 3-2. 남는 코드를 6개로

| # | 주제 | 주요 파일 | 대략 |
|---|------|-----------|------|
| PR-7 | KV block quantizer (host) + skel 진입점 | `hexkl_kv_quant.c/h`, `unittest_hexkl_kv_quant.cpp` | ~750 |
| PR-8 | softmax 헬퍼 헤더 분리 + blocked masked softmax | `hvx_softmax_util.h`, `hvx_softmax_blocked_f32.c/h` | ~420 + 테스트 |
| PR-9 | attention dtype vtable + `S = Q·Kᵀ` | `hexkl_attn_dtype.c/h`, `hexkl_acc_tile.c/h`, `hexkl_mm_opts.h` | ~350 |
| PR-10 | fused `attn_forward` (Q·Kᵀ + masked softmax + P·V) | `hexkl_attn_u8.c/h`, `nntr_hvx_attn.c`, `unittest_hvx_attn.cpp` | ~1,770 |
| PR-11 | VTCM tile dequant / quant pooling / transport (DCVS·QoS·ION) | `hexkl_mm_u8i4_dma.c`, `hvx_dequant_i32.c` 변경분 | ~380 |
| PR-12 | 계측·벤치 인프라 (선택) | `htp_rpc_bench.h`, `unittest_hvx_fc.cpp`, `hexkl_probe.c/h`, `tools/*.py` | ~2,300 |

PR-7 은 host-only 라 **PR-3 이후 아무 때나 병렬**로 넣을 수 있다.
PR-8~PR-10 은 순서대로 스택.
PR-11 은 성능 전용이라 정확도 PR 들과 분리하는 게 좋다.
PR-12 는 리뷰 가치가 낮으니 마지막에 넣거나 아예 upstream 에서 빼는 것을 권장.

### 3-3. #4256 히스토리 정리

94 커밋 중 30개가 `[doc]`/`[docs]`/`[tools]`, 여러 개가 계측 추가 → 제거
(`338e92b` 추가 → `11ab9e2` "stop instrumenting the production path" 제거) 왕복이다.
PR 로 올릴 때는 위 6개 주제 단위로 **squash 해서 새로 쌓는 것**이 맞다.
지금 히스토리는 작업 로그이지 리뷰용 시리즈가 아니다.

---

## 4. 리뷰 · 머지 순서

```
1)  #4236                                  ← 지금 리뷰 중, 그대로 진행
2)  PR-2 (softmax)  ‖  PR-3 (u8i4 layer)   ← 병렬. #4236 머지 후 base 를 main 으로
3)  PR-6 (session 통일)                     ← PR-2, PR-3 둘 다 들어간 뒤. 40줄
4)  PR-4 (u8i8)      = #4244 갱신
5)  PR-5 (quant opt) = #4249 갱신
6)  PR-7 (KV quant)                        ← PR-3 이후 병렬 가능
7)  PR-8  → PR-9 → PR-10 (flash attention 본체)
8)  PR-11 (성능) → PR-12 (계측, 선택) → docs (선별)
```

**지금 당장 해야 하는 것**

- [x] #4244 / #4249 를 재구성 내용으로 force-push (2026-08-11, #4236 `2f22a22` 위로 리베이스 완료)
- [ ] #4245 를 닫고 PR-2 / PR-3 두 개로 다시 연다 (head 가 `dlwlzzero` fork 라 push 불가)
- [ ] #4244 head(`htp/u8i8-dma-cross`)를 PR-4 내용으로 force-push, base 를 PR-3 로 변경
- [ ] #4249 head(`htp/quant-dequant-hvx-opt`)를 PR-5 내용으로 force-push, base 를 PR-4 로 변경
- [ ] #4256 은 draft 유지, 최신 main 으로 rebase 해서 역행 diff 제거
- [ ] `[Wait for#4243]`, `[Wait for#4244]` 제목 접두사는 base 를 제대로 잡으면 필요 없음

---

## 5. 검증

모든 브랜치는 cherry-pick 으로 재구성했고, 원본과 트리가 동일함을 확인:

- `#4245` 의 u8i4 부분 트리 == `#4236` 팁 트리 (htp 경로 diff 0)
- `#4244` 의 u8i4 base 트리 == PR-3 팁 트리 (diff 0)
- `#4249` 의 u8i8 base 트리 == PR-4 팁 트리 (diff 0)

즉 **코드 내용은 한 줄도 바뀌지 않았고**, base 와 커밋 배치만 재구성했다.
디바이스 검증 결과(S25 Ultra V79, 14/14 통과)는 그대로 유효하다.
