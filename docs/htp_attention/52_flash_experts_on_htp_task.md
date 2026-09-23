# 52 — expert 가중치를 flash에서: HTP용 cached-slim (다음 세션 과제, 자체 완결)

이 문서 하나로 다음 세션이 시작할 수 있게 쓴다: 어디까지 와 있는지, 무엇을 만들지,
무엇은 이미 있어서 만들면 안 되는지, 어떻게 일하고 어떻게 재는지. 상태: **설계·분석만,
코드 없음.** 아래의 숫자 중 "측정"이라 적힌 것만 기기 실측이고 나머지는 산술이다.

## 0. 한 줄

지금 HTP 경로는 22층 × 32 expert 전부(3696 MiB)를 ION 아레나에 **상주**시킨다 → peak RSS
5.2 GB. CPU 쪽에는 이미 두 가지 flash 방식(slim: 콜마다 mmap, cached-slim: LRU 상주)이
있다 (PR nntrainer#4264). 같은 두 방식을 **HTP 경로**에 만든다: expert의 WH 바이트를
파일에서 필요할 때 ION 슬롯으로 읽어 등록하고, LRU로 내보낸다. DSP 커널은 **손대지
않는다** — 콜은 이미 expert별 핸들 배열을 받는다.

## 1. 지금 어디까지 왔나 (2026-09-22 기준, 브랜치 `claude/lfm2-moe-ffn-hexkl-2ivn5v` = PR #4327 head)

| 항목 | 상태 | 근거 |
|---|---|---|
| MoE FFN 22층, HTP 한 콜 | 완료, 콜당 14.4 ms prefill / 1.44 ms decode | 49, 51 §2.24 |
| conv 블록 18층 한 콜 | 완료, 4.4 ms/콜, ppl 손실 0 | 51 §2 |
| attention q/k/v(+norm) 한 콜, o_proj FC | 완료 | 51 §2.23 |
| dense FFN 2층 | 켤 수 있음, ppl +4.2% (−10 ms) | 51 §2.17, 2.21 |
| prefill 444 토큰 / decode | **585~616 ms / 24.2 TPS** (전부 켬, ppl 62.09) | 51 §2.24, 2.27 |
| 순수 CPU (같은 프롬프트) | 921~1013 ms / 20.6 TPS, ppl 57.00 | 50 §3.7, 51 §2.17 |
| 남은 HTP 소프트웨어 지렛대 | 없음 (각 ≤5 ms, 기기 편차 ±25 아래) | 51 §2.27 |
| 구조적 잔여 | MoE HMX 활용률 59% (M=444), ARM attention core ~120 ms(다른 담당), 스테이징 25 ms | 51 §2.24 |
| **메모리** | **peak RSS 5.2 GB**: 아레나 15 × 256 MiB = 3840 MiB(expert) + 나머지 | 49 §A4, 51 §2.21 로그 |

정확도 게이트는 perplexity다(`NNTR_PPL=1`, 51 §2.15). 텍스트 비교는 지표가 아니다(51 §2.9).
현재 config의 기준값: **ppl 62.0916**(전부 켬), 57.12(conv만), 57.00(CPU). 산술을 안 바꾸는
변경은 소수점까지 같아야 한다 — 이 과제도 그렇다(가중치 값이 아니라 위치만 바뀐다).

## 2. CPU 쪽에 이미 있는 두 방식 — 읽고 옮길 것, 다시 만들지 말 것

PR nntrainer#4264 (`Jungwon-Lee`, LFM2-MoE 지원)가 넣었고 이 브랜치에 있다.
`Applications/CausalLM/models/lfm2_moe/`:

| 파일 | 모델 architectures 키 | 레이어 타입 | 방식 |
|---|---|---|---|
| `lfm2_moe_layer.cpp` | `Lfm2MoeForCausalLM` | `lfm2_moe` | 전부 상주. **HTP 경로는 이것** (`moe_engine: htp`) |
| `lfm2_moe_layer_fsu.cpp` | `Lfm2SlimMoeForCausalLM` | `lfm2_moe_slim` | **slim**: expert 가중치를 FSU 가상 텐서로 두고, 콜마다 `activate()`(mmap) → dot → `deactivate()`(munmap). 상주 0, I/O 최대 |
| `lfm2_moe_layer_cached.cpp` | `Lfm2CachedSlimMoeForCausalLM` | `lfm2_moe_cached_slim` | **cached-slim**: 위 + 층당 **LRU** `NNTR_MOE_CACHE_EXPERTS`개(기본 32 = 전부) mmap 상주; 미스면 LRU 꼬리를 `deactivate()`하고 새 expert를 `activate()`. 라우팅의 top-k + `EXTRA_TOPK`(5)개로 recency 갱신(다음에 쓸 것을 미리 앞으로) |

메커니즘 (`nntrainer/tensor/tensor.cpp` `Tensor::activate`, `swap_device.cpp`):

- nntr_config `"fsu": true`(+ `fsu_lookahead`)면 로더가 expert 가중치를 **읽지 않고** 파일 오프셋만
  기록한다(`Tensor::setFileOffset`, `requestWeight(..., is_virtual=true)`). 모델 파일 fd를 텐서가 들고
  있다.
- `activate()` = `mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, off&~4095)` + Android면
  `madvise(MADV_WILLNEED)`; `deactivate()` = munmap. 즉 **flash → page cache → 사용자 주소**이고,
  "읽기"는 커널의 page fault다.
- CPU 경로는 mmap된 주소로 바로 GEMM을 돈다. **HTP는 그럴 수 없다**: DSP가 읽는 메모리는
  ION(dma-buf)이어야 하고(`fastrpc_mmap`), 파일 mmap 페이지는 DSP에 매핑되지 않는다. 그래서
  HTP 버전은 "mmap 후 사용"이 아니라 **"ION 슬롯으로 복사(또는 pread) 후 등록"** 이다. 이것이
  CPU 방식과 유일하게 다른 점이고, 설계 전부가 여기서 나온다.

## 3. HTP 경로가 지금 가중치를 어떻게 쥐고 있나 (바꿀 자리)

`nntrainer/tensor/htp_backend/htp_compute_ops.cpp`:

- 로드 시(`transformer.cpp`의 등록 분기, `lfm2_moe` 타입) `register_qs4cx_weight(data, scale, K, N, wh=true)`
  → 파일의 QS4CX_WH 바이트(이미 HMX 타일 배치, `htp_wh_layout.h`)를 아레나 청크에 memcpy → 
  `registerFromArena` → DSP `weight_register_u8i4_arena(K, N, arena, wh_off, w_scale, colsum_w, bias)`
  (4 KB만 건넘, 바이트는 제자리 차용) → `releaseArmSource`(ARM 사본 `MADV_DONTNEED`).
- 아레나: `ensureArena` → `rpcmem_alloc` 256 MiB 청크 → `fastrpc_mmap(FASTRPC_MAP_FD)` →
  `nntr_hvx_arena_attach(fd)`; `placeExisting`이 bump 포인터로 자리 배정(4 KB 정렬). **해제·재사용
  없음** — 이것이 만들 것의 핵심.
- DSP 쪽(`test/htp/nntr_hvx.idl`): `weight_register_u8i4_arena`, `weight_release_u8i4(handle)`,
  `arena_detach`(핸들이 남아 있으면 EBADSTATE). 슬롯 테이블 `HEXKL_MM_U8I4_MAX_WEIGHTS`=2048.
- 콜: `invokeMoeLayer(session, h_gu, h_dn, row_index, row_count, row_weight, ...)` — **핸들 배열은
  콜마다 넘어간다**. LRU가 expert→핸들을 바꿔도 커널은 모른다. 등록 프로파일: convert 0.31~0.57
  ms/weight(colsum 계산 + memcpy), FastRPC register 0.46 ms/weight.
- 메모리 상수: expert 하나 = gate_up 2048×3584 + down 1792×2048 int4 = **5.25 MiB**(WH 바이트) +
  4 KB(scale/colsum/bias). 22층 × 32 = 704 expert = 3696 MiB.

## 4. 설계 — HTP cached-slim

```
파일(QS4CX_WH, 이미 WH 바이트) ──pread──▶ ION 아레나 슬롯 ──register_arena──▶ DSP 핸들
                                          ▲ 층당 C개 슬롯 LRU              │ weight_release on evict
                                          └── 미스: LRU 꼬리 release → pread → register
```

1. **아레나를 슬롯 풀로.** 청크 크기는 그대로(256 MiB = 48 슬롯), 슬롯 = 5.25 MiB + 4 KB 정렬.
   총 슬롯 = 22 × C. C=8 → 924 MiB, C=16 → 1.85 GB, C=32 → 지금(3.7 GB). 아레나 청크 수를 C에서
   계산해 `ensureArena`를 그만큼만.
2. **로드 시**: expert 가중치는 FSU 가상 텐서로(config `fsu: true` — CPU cached-slim과 같은 파일·
   같은 오프셋 기록). 등록 분기는 바이트 대신 **(fd, file_offset, K, N, scale)** 만 기록한다. colsum은
   바이트가 있어야 계산되므로 (a) 로드 시 한 번 전부 읽어 계산해 두거나(704 × 4 KB = 2.8 MB 상주,
   읽기 3.7 GB 한 번 — 기동 +수 초) (b) 미스 때 계산(0.3 ms/weight, 미스 비용에 얹힘). **(a)로 시작**
   — 미스 비용이 결정적 변수라서 거기서 0.6 ms를 빼는 게 맞다. 더 좋은 것은 (c) colsum을 파일에
   같이 굽는 것(양자화기 변경, 문서 45 §8.4의 "한 번 bake" 항목) — 2단계.
3. **콜 직전**(`gemm_qs4cx_fused_moe...`/`invokeMoeLayer` 호출부, 라우팅이 끝나 이번 콜의 expert
   집합을 아는 지점): 층의 LRU를 본다. 히트 → 핸들. 미스 → 꼬리 expert의 두 핸들 `weight_release_u8i4`
   → 슬롯 재사용: `pread(fd, slot_ptr, 5.25 MiB, off)` **직접 ION으로**(mmap→memcpy는 page cache 한 번
   더 거친다; `MAP_PRIVATE` mmap은 CPU 방식의 유산이지 HTP에는 필요 없다) → `weight_register_u8i4_arena`.
   CPU cached-slim의 `EXTRA_TOPK` recency 갱신 규칙을 그대로 옮긴다.
4. **prefill**은 444 토큰 × top-4면 32 expert 전부가 활성이라 **첫 prefill이 곧 전체 로드**다 (C < 32면
   층마다 32 − C번 미스·교체가 매 prefill 콜에). decode는 토큰당 층당 4 expert; 히트율은 라우팅 국소성이
   정한다 — **재기 전엔 모른다**(§6의 첫 측정).
5. **DSP 커널·IDL·skel 변경 없음.** 호스트 `htp_compute_ops.cpp`와 `transformer.cpp`의 등록 분기,
   config 키 하나(`moe_cache_experts` 또는 기존 `NNTR_MOE_CACHE_EXPERTS` 환경변수 재사용 — 후자가
   CPU 쪽과 같은 손잡이라 낫다).

미스 비용 산술 (expert 하나, 5.25 MiB): UFS 순차 읽기 ~1.5~2 GB/s → **2.6~3.5 ms** + register 0.9 ms
(두 weight) + (colsum (b)면 +0.6). decode 토큰당 미스가 m개면 +3.5m ms — 42 ms/토큰에서 m=1이면
−8% TPS, m=4면 −33%. **히트율 표가 첫 결과물이다.** prefetch(다음 층의 라우팅은 이번 층이 끝나야
나오므로, 문서의 EXTRA_TOPK처럼 "이번 콜의 5~9위"를 다음 토큰의 후보로 미리 읽는 것)는 히트율 표를
본 뒤에.

slim(캐시 0, 콜마다 읽기)은 HTP에서는 **만들지 않는다**: 콜당 32 × 3.5 ms = 112 ms가 커널 14 ms 위에
얹힌다. C=0은 cached-slim의 경계값으로 자연히 나온다(측정용).

## 5. 만들 순서 (게이트 있는 단계)

| 단계 | 내용 | 게이트 |
|---|---|---|
| 0 | CPU cached-slim을 그대로 한 번 돌린다: `Lfm2CachedSlimMoeForCausalLM`, `fsu: true`, `NNTR_MOE_CACHE_EXPERTS=8/16/32` — TPS와 peak RSS | CPU 쪽 히트율·비용의 감. 코드 0줄 |
| 1 | 등록 분기: FSU 가상 expert의 (fd, offset)만 기록, colsum (a). `NNTR_MOE_CACHE_EXPERTS=32`면 지금과 같은 상주 상태를 **이 새 경로로** 만들어 ppl 62.09·prefill 동일 확인 | 회귀 0 |
| 2 | LRU + release/pread/register. C=16, 8. peak RSS, prefill, decode TPS, 층별 미스 수(프로파일 행 추가: `M==1` 행 옆에 "miss/call, read ms/call") | RSS가 C에 비례해 내려가고 ppl 동일 |
| 3 | 히트율 표를 보고: prefetch(EXTRA_TOPK) / colsum 굽기 / 슬롯 크기 | 측정이 시키는 것만 |

호스트에서 검증 가능한 것: 등록 분기와 LRU는 `HtpComputeOps` 안이라 HTP 없이는 안 돈다. LRU
자체는 순수 자료구조라 **작은 유닛 테스트**(가짜 register/release 카운터로 evict 순서·용량 검증)를
`test/unittest/`에 두는 것이 "runnable check"다. 나머지는 기기.

## 6. 기기 측정 방법 (그대로 복사해 쓰는 절차)

빌드: DSP skel은 `test/htp/build.sh`(커널·IDL이 바뀔 때만 — 이 과제는 안 바뀐다), 앱은
`Applications/CausalLM/build_android.sh --htp`, 설치 `install_android.sh --model=<dir>`. 상세·환경변수·
SDK 경로 함정은 `mobile_e2e_run_guide.md` §1~5. 기기: Galaxy S25 Ultra(V79).

모델 디렉터리 (기기 `/data/local/tmp/nntrainer/causallm/models/`):

| 디렉터리 | 용도 |
|---|---|
| `lfm2.5-8b-a1b-q40-qs4cx-wh` | **HTP용** (MoE `--moe_dtype QS4CX_WH`, `moe_engine: htp`) — 이 과제의 파일 |
| `lfm2.5-8b-a1b-q40` | 순수 CPU 기준 (`moe_engine` 없음). HTP로 못 돈다 |

nntr_config.json (HTP 디렉터리, 지금 전부 켬 상태):
```json
"moe_engine": "htp",
"conv_block_engine": "htp",
"dense_ffn_engine": "htp",
"attn_proj_engine": "htp"
```
이 과제가 더할 것: `"fsu": true`(+ `architectures`를 cached-slim 변형으로 — 또는 `lfm2_moe` 레이어에
캐시를 붙이면 그대로).

명령 (헤드라인은 **비프로파일 2회 최솟값**, 분해는 PROFILE=2 1회; PPL은 prefill 시간을 부풀리므로
프로파일 실행에만):
```bash
cd /data/local/tmp/nntrainer/causallm
LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. NNTR_NUM_THREADS=8 ./nntrainer_causallm ./models/lfm2.5-8b-a1b-q40-qs4cx-wh
LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. NNTR_NUM_THREADS=8 NNTR_HTP_PROFILE=2 NNTR_PPL=1 ./nntrainer_causallm ./models/lfm2.5-8b-a1b-q40-qs4cx-wh
```
환경변수: `NNTR_HTP_PROFILE=0/2/3`(3 = 콜 안 5회 반복 최솟값, ≤5 ms 차이는 이걸로만 판정),
`NNTR_PPL=1`, `NNTR_HTP_POLL_US`(기본 5000), `NNTR_MOE_CACHE_EXPERTS`, `NNTR_M0_PROFILE=1`(층별 ARM
시간), `NNTR_HTP_KEEP_ARM_WEIGHTS=1`(디버그).

읽는 법: 프로파일 첫 줄 `qos_mode=2`여야 유효. 행: `M>1`(MoE prefill), `M>1 dense/conv/FC`, `M==1`(decode
MoE). 열의 뜻은 51 §2.24 표. `peak memory`/`Max Resident Set Size`가 이 과제의 헤드라인 지표.
**기기 편차**: 세션이 다르면 ±25 ms prefill(51 §2.27) — 비교는 같은 세션에서 A/B로.

기준값(전부 켬, 2026-09-22): prefill 585~616, decode 24.2, ppl 62.0916, peak RSS 5,252 MB, 등록 2.9~4.6 s.

## 7. 일하는 방식 (이 저장소·이 사람의 규칙)

- 스타일: `01_working_style.md`("ponytail 모드") — 사다리를 오르고 첫 칸에서 멈춘다, 문제를 끝까지
  이해한 뒤에. **가설에 코드 쓰기 전에 분해를 재라**(이번 세션의 51 §2.25~2.27이 교재: 셋 중 둘이
  틀렸고, 프로파일이 즉시 가렸다). 깎은 자리는 `ponytail:` 주석.
- 검증: 돌릴 수 없으면 "돌리지 못했다"고 쓴다. 호스트 체크는 `bash test/htp/host/run_host_checks.sh`
  (MoE·conv·FC 커널 비트 동일, 워커 풀), LFM2 유닛 테스트는
  `build/Applications/CausalLM/unittest_causallm_models --gtest_filter='*Lfm2*'`(호스트 빌드:
  `meson setup build -Denable-transformer=true -Denable-blas=false -Denable-tflite-interpreter=false
  -Denable-tflite-backbone=false`, `Applications/CausalLM/json.hpp`에 nlohmann 단일 헤더 필요 — git
  ignore됨).
- 커밋: 저자·커미터 `SeungHui Lee <shsh1004.lee@samsung.com>`, 제목 `[<component>] <subject>`
  (component: HTP / CausalLM / Docs / test …), 본문은 왜·무엇·측정치, 마지막 두 줄 정확히
  ```
  Co-authored-by: Claude <noreply@anthropic.com>
  Signed-off-by: SeungHui Lee <shsh1004.lee@samsung.com>
  ```
  그 외 트레일러(세션 링크, 모델명) **없음**. 명령:
  `git -c user.name="SeungHui Lee" -c user.email="shsh1004.lee@samsung.com" commit -q --author="SeungHui Lee <shsh1004.lee@samsung.com>" -m "..."`.
  clang-format-18(로컬에 14가 없어서)을 **바뀐 줄에만**(`git diff -U0` 범위로 `--lines=`), 새 파일은 통째.
  `subprojects/`는 건드리지 않는다. 푸시는 두 브랜치 동시에:
  `git push -q origin HEAD:claude/htp-lfm2-moe-ffn HEAD:claude/lfm2-moe-ffn-hexkl-2ivn5v`.
- 문서: 측정·결정은 그날 바로 해당 문서의 다음 절(§N.M)에 적는다 — 기대치, 실측, 판정, 되돌린 것과
  이유. `00_START_HERE.md` 표에 한 줄.
- ponytail 스킬(선택): `https://github.com/DietrichGebert/ponytail`. Claude Code에서
  `/plugin marketplace add DietrichGebert/ponytail` 다음 `/plugin install ponytail@ponytail`(두 프롬프트로
  따로). `/ponytail-review`가 diff의 과잉 설계를 잡는다. 없어도 `01_working_style.md`가 같은 규칙이다.

## 8. 다음 세션의 첫 프롬프트 (복사해서 쓰기)

```
nntrainer 저장소, 브랜치 claude/lfm2-moe-ffn-hexkl-2ivn5v (PR nntrainer/nntrainer#4327의 head
claude/htp-lfm2-moe-ffn에도 같이 푸시). 먼저 CLAUDE.md → docs/htp_attention/00_START_HERE.md →
docs/htp_attention/52_flash_experts_on_htp_task.md를 읽어. 52가 이번 과제다: HTP 경로의 MoE expert
가중치를 flash에서 LRU로 가져오는 cached-slim (peak RSS 5.2 GB를 C에 비례해 내리기). 52 §7의
커밋 형식과 §6의 측정 절차를 그대로 따르고, 52 §5의 단계 순서로 가. 단계 0(CPU cached-slim
한 번 돌리기)은 내가 기기에서 돌릴 테니 config와 명령을 먼저 줘. 기기 측정은 전부 내가 한다 —
너는 결과를 받아 문서에 기록하고 다음 단계를 정해. DSP 커널·IDL·skel은 이 과제에서 안 바꾼다.
```

이 뒤에 기기 로그를 붙여 넣으면 된다. 로그를 줄 때는 실행 명령 줄까지 같이(이번 세션에서 두
디렉터리를 헷갈린 일이 한 번 있었다, 51 §2.21 앞).

## 9. 함정 목록 (이번 세션에서 실제로 밟은 것)

- `-q40` 디렉터리는 CPU용이다. HTP 프로파일이 안 찍히면 경로부터 본다.
- 텍스트가 달라진 것은 정확도 지표가 아니다. ppl로 본다.
- 세션 간 prefill ±25 ms는 기기 상태다. 콜당 host 합으로 비교한다.
- "DRAIN이 작으니 DMA는 여유"가 아니다 — HMX 옆의 DMA는 ~12 GB/s다(51 §2.26).
- 프로파일 행 키가 (K, N, M==1, kind)라 같은 모양의 다른 콜이 한 행에 섞일 수 있다(51 §2.21).
- 커널 바꾸면 skel도 다시 빌드·푸시. 안 하면 첫 콜이 AEE_EBADPARM.
- `Applications/CausalLM/json.hpp`가 없으면 호스트 빌드가 CausalLM에서 멈춘다.

## 10. 구현 세션 (2026-09-22) — 코드가 §4를 어디서 고쳤나, 그리고 게이트

브랜치 `claude/eager-keller-f91z9o`(5622c74에서 분기). §1~§9는 그대로 두고, 코드를 끝까지 따라가
보니 §4의 설계가 세 곳에서 바뀌었다. 전부 **기기 미측정** — 호스트에서는 빌드·LFM2 유닛 테스트·
LRU 유닛 테스트·HTP ops 파일의 구문 검사(스텁 헤더)까지만 했다.

### 10.1 §4와 다른 것

1. **colsum은 이미 파일에 있다.** QS4CX_WH의 파일 레이아웃은 `[nibbles][scale N][colsum N]`
   (`quantize_stream.cpp:851`, `QS4CX_WH_Tensor::size()`, `get_or_register_wh`가 `matAscale + N`을 읽음).
   §4.2의 (a)/(b)/(c)는 전부 필요 없다. 미스 = pread 2회(nibbles → ION 슬롯, scale+colsum 44 KB → 힙).
2. **`fsu: true`는 필요 없고 HTP 경로에서는 해롭다.** virtual 여부는 `requestWeight(..., is_virtual)`
   하나로 정해지고(`tensor_pool.cpp:38` VIRTUAL → UNMANAGED, 메모리 0), 파일 오프셋은 fsu와 무관하게
   모든 weight에 기록되며(`neuralnet.cpp:964`), fd는 추론이면 항상 열려 read에 전달되고 virtual tensor가
   저장한다(`tensor.cpp:1422`). `fsu: true`가 실제로 하는 일은 weight_pool을 swap 캐시 풀로 바꿔 **모든**
   non-virtual weight를 층마다 load/unload하는 것(`manager.h:148`, `neuralnet.cpp:528`) — HTP 경로에선
   attention/conv/dense 가중치까지 토큰마다 파일에서 다시 읽게 된다. **config에 넣지 않는다.**
3. **HTP 콜 하나가 한 층의 활성 expert 전부를 동시에 요구한다.** prefill은 층마다 32개 전부가 활성이므로
   층당 C < 32의 LRU로는 한 콜을 만들 수 없다. 콜을 쪼개면(호스트 누적) fp32 합산 순서가 바뀌어 §1의
   "ppl 소수점 동일" 게이트를 잃는다. 그래서 **슬롯 풀은 22층이 공유하는 하나의 LRU**다(총 S = 22×C;
   `expert_lru.h`, `lfm2_moe_layer.cpp`의 `g_expert_lru`). 콜 직전에 그 콜의 expert를 전부 핀한 채 확보하고,
   S ≥ 32이면 한 콜/층이 그대로다. **C의 최솟값은 2**(44슬롯, 231 MiB); C=1은 콜 분할이 필요해 안 만들었다.
   C < 32이면 prefill은 매번 704 expert를 전부 다시 읽는다(층 L의 32개가 앞 층들을 밀어냄) — 이건 C로 안
   줄고 3단계 prefetch(S ≥ 64)로만 겹쳐진다.

### 10.2 만든 것

| 파일 | 무엇 |
|---|---|
| `compute_ops.h` | `register_qs4cx_wh_expert_file(key_gu, key_dn, fd, off_gu, off_dn, K, inter, N_out, at_load)`, `release_qs4cx_wh_expert(key_gu)` |
| `htp_compute_ops.cpp` | expert 쌍 슬롯(gate_up 3.5 MiB + down 1.75 MiB, 4 KB 정렬, 256 MiB 청크당 48개) + free list. 미스: free list pop(없으면 bump) → `pread` ×2 → `registerFromArena` ×2. release: `weight_release_u8i4` ×2 → 슬롯을 free list로. `handle_cache_`는 tensor 주소를 키로. 프로파일: MoE 행에 `expert misses n (x/call), file read, register+release rpc` 열과 총계 한 줄 — 전부 `host=` 밖 |
| `tensor.h` | `getFd()` 한 줄 (virtual tensor가 read 때 저장한 fd) |
| `lfm2_moe_layer.cpp/.h` | `NNTR_MOE_CACHE_EXPERTS` + 가속기 엔진이면 expert를 virtual로 요청. `tryMoeLayerOnAccelerator`: 활성 expert만 `g_expert_lru.acquire`(evict → release, miss → register_from_file), 배열을 활성 expert로 압축(행 0인 expert는 기여 0이라 산술 동일), key + null scale로 콜, 콜 뒤 EXTRA_TOPK(5) recency 갱신. `preloadExperts`: 로드 때 층 순서로 풀을 채움(청크 할당이 전부 로드 때 끝남) |
| `transformer.cpp` | virtual expert는 `register_qs4cx_weight` 대신 `preloadExperts`; 워밍업은 층 0이 전부 상주일 때만 |
| `expert_lru.h` + `unittest_expert_lru.cpp` | 공유 LRU(층별 용량 합, 핀, evict → load 순서, refresh)와 그 유닛 테스트 6개 |
| `unittest_hvx_mm_u8i4.cpp` | `ArenaSlotReuseMatchesHeap`: 슬롯에 A → 콜 → release → 같은 오프셋에 B → 콜, 힙 핸들과 memcmp==0 |

환경변수가 없으면 코드 경로가 하나도 안 바뀐다(virtual 아님, 압축 없음, LRU 없음).

### 10.3 왜 슬롯 재사용에 기기 테스트가 먼저인가

출하 커널은 가중치를 DMA로만 읽지만(`hexkl_mm_u8i4_moe.c:143`; HVX 꼬리는 `MOE_TAIL_MAX_ROWS=0`으로 꺼짐)
디스크립터가 `src_bypass=0`이라 DDR 소스 읽기가 DSP L2를 거친다. 기존 아레나 테스트는 "빈 슬롯에 첫 쓰기"만
증명했다. LRU가 만드는 "DSP가 읽음 → 호스트가 덮어씀 → DSP가 다시 읽음"에서 DSP 매핑이 캐시 가능이면 stale
라인이 나올 수 있다. `ArenaSlotReuseMatchesHeap`가 그 판정이고, 실패하면 커널 없이 되는 우회가 없다
(`weight_register_u8i4_arena`에 L2 invalidate 한 줄 = DSP 변경 = 이 과제 밖).

### 10.4 기기에서 잴 것 (순서대로)

| # | 실행 | 보는 것 | 게이트 |
|---|---|---|---|
| 0 | CPU cached-slim(`-q40`, architectures만 바꿈, fsu 없음) C = 1, 2, 4, 8 (+ 순수 CPU 1회) | prefill, decode TPS, VmRSS peak, Max RSS, ppl(C=32) | CPU 쪽 감 |
| 1 | HTP, `NNTR_MOE_CACHE_EXPERTS=32` 비프로파일 2회 + PROFILE=2 PPL 1회 | ppl, 콜당 host, 등록 시간(convert 열 = 파일 읽기), peak RSS | **ppl 62.0916 동일**, 콜당 동일 |
| 2a | `run_u8i4_layer_on_device.sh` (`ArenaSlotReuseMatchesHeap` 포함) | `arena_slot_reuse pass=1 bad_elems` | **0** |
| 2 | HTP, C = 2 → 4 → 8, 각 cold 1회(drop_caches) + warm 2회 + PROFILE=2 PPL 1회 | peak RSS, prefill, decode TPS, MoE 행의 misses/call·read·rpc, `[M0-PROF] miss=` | RSS ∝ C, ppl 동일 |

예상(§1 585 ms / 24.2 TPS 기준): C<32의 prefill은 704 미스 — warm(page cache) +1.4~2 s, cold(flash)
+2.8~3.5 s. decode: C=2(44 < 작업집합 88) 거의 전부 미스 ≈ 2~4 TPS(기능 확인용), C=4·8은 히트율이 정함 —
그 표가 첫 결과물. 미스 하나 = pread 5.25 MiB + FastRPC 4회(release 2 + register 2, ≈1~1.8 ms).

### 10.5 단계 0 실측: CPU cached-slim — RSS는 C에 비례, 히트율은 C/32, LRU는 무작위와 같다 (2026-09-23)

`-q40`, `Lfm2CachedSlimMoeForCausalLM`, `fsu` 없음(파일에 `"fsu": false`), 444 토큰 prefill / 512 토큰 decode,
NNTR_NUM_THREADS=8, 같은 세션 연속 실행(순서: 기준 → C=1 → 2 → 4 → 8 → C=32+PPL). 기기 R3CY10WM83Y.

| 실행 | prefill ms | decode TPS | Max RSS MB | 비고 |
|---|---|---|---|---|
| 순수 CPU (`Lfm2MoeForCausalLM`) | 2711 | **50.5** | **4866** | 세션 첫 실행 |
| C=1 | 2334 / 1678 | 19.7 / 18.6 | 963 | |
| C=2 | 2489 / 2474 | 16.4 / 15.8 | 1074 | C=1보다 느림 — 아래 |
| C=4 | 2484 / 2470 | 19.1 / 19.0 | 1338 | |
| C=8 | 2546 / 2784 | 24.1 / 22.2 | 1871 | |
| C=32 + `NNTR_PPL=1` | 3889 | 40.8 | 4898 | **ppl 51.5912** (443 토큰) |

읽기 (decode 토큰당 ms = 1000/TPS, 기준 19.8 ms):

- **RSS ∝ C, 기울기 5.76 MB/expert, 바닥 814 MB.** expert 하나 5.25 MiB + 페이지 반올림 그대로. C=32 예측 4869 vs 실측
  4783(peak memory 값 기준). 바닥 814 MB가 CPU 경로의 "expert 빼고 전부"다 — HTP 경로의 같은 수치(§10.4 2단계)와 비교할 기준.
- **prefill은 C와 무관하다** (2.3~2.8 s, 기준 2.7). 층마다 32 expert를 전부 mmap→읽기→munmap 해도 상주와 같다: 3.7 GB
  expert 파일이 **page cache에 통째로 살아 있다**(기기 RAM이 충분). §10.1의 "C<32는 매 prefill이 flash를 다시 읽는다"는
  cold일 때만 맞고, warm에서는 memcpy급이다 — HTP 2단계도 warm/cold를 나눠 재야 하는 이유가 실측으로 확인됐다.
- **미스 하나 = 0.37 ms** (C=1: 토큰당 88 미스 전부 → +31~34 ms). mmap + page cache fault + munmap 5.25 MiB의 값.
- **히트율 = C/32, 국소성 없음.** C=4: +32.8 ms = 88 미스 = 히트 0%. C=8: +21.6~25.3 = 58~68 미스 = **히트 22~34%**;
  32개 중 8개가 상주할 때 무작위 top-4의 기대 히트 25%. C=32(cached, 전부 상주): +4.7 ms(§ 아래). 즉 decode의 expert
  접근은 recency 기준으로 **거의 균일**이고, EXTRA_TOPK 갱신을 포함한 LRU가 무작위 교체 대비 얻는 것이 없다. 이것이
  "히트율 표"의 답이다: **캐시 크기가 곧 히트율이고, 정책은 무관하다.**
- C=2가 C=1보다 느린 것(+41~44 ms, 미스 88개로 설명 안 됨)과 cached C=32(40.8)가 상주(50.5)보다 20% 느린 것은
  이 데이터로 못 가른다 — 실행 순서(열)거나 file-backed mmap vs 익명 힙의 차이. 우리 경로가 아니라 추적 안 함.

**HTP에 대한 함의 (산술, 2단계에서 잰다).** 라우팅이 균일하면 decode 미스/토큰 = 88 × (1 − C/32): C=8 → 66, C=16 → 44.
HTP 미스 = FastRPC 4회(1~1.8 ms) + pread 5.25 MiB(warm 0.5~1 ms) ≈ **1.5~3 ms**, CPU의 0.37 ms의 4~8배. C=8이면 토큰당
+100~200 ms → **4~8 TPS**, C=16이면 +66~130 → 6~12 TPS. 같은 메모리(1.9 GB)에서 CPU cached-slim이 23 TPS다. 미스 비용을
0.4 ms 아래로 내리지 못하면 HTP cached-slim은 decode에서 CPU cached-slim에 진다. 내릴 수 있는 것은 RPC 4회 → IDL 배치
등록/해제(DSP 변경, 이 과제 밖)와 pread의 백그라운드 선읽기(히트율이 균일이라 "무엇을" 선읽을지가 top-k 밖에 없음 —
EXTRA_TOPK 후보 5개 중 다음 토큰에 실제로 쓰일 확률도 균일 가정이면 5/32). prefill은 한 콜/층이 살아 있어 HTP가 이긴다
(704 × warm 미스 ≈ +1.4~2 s 위에 0.585 → 2~2.6 s vs CPU cached-slim 2.5 s — 비슷).

**기록과 어긋나는 값 둘 — 1단계에서 같은 세션 A/B로 가른다.**

1. **ppl 51.59 vs §1의 57.00.** 이 세션은 순수 CPU의 ppl을 안 쟀다(스크립트 누락). cached-slim 레이어는 SwiGLU를
   `acti_func`로, 상주 레이어는 `swiglu_det`로 계산하므로 마지막 비트는 다를 수 있지만 10%는 아니다. 57.00은 §2.17
   (qkv 융합 커밋 1f538c2 이전)에 기록됐고 그 뒤 CPU 경로의 ppl은 다시 안 쟀다 — 기준값 자체가 움직였을 가능성이 크다.
   순수 CPU + `NNTR_PPL=1` 한 번이면 끝난다.
2. **순수 CPU decode 50.5 TPS vs §1의 20.6.** prefill도 2711 vs 921~1013. 같은 프롬프트(443 토큰)인데 두 방향으로 다르다.
   HTP 전부 켬(§1: 24.2 TPS)이 오늘의 CPU 50.5보다 느리다면 이야기가 달라지므로, 1단계에 **HTP 상주(환경변수 없음)
   실행을 같은 세션에** 넣어 오늘의 A를 다시 잡는다.

### 10.6 단계 1 실측: 게이트 통과, 그리고 5.2 GB의 정체 — 아레나는 RSS에 없었다 (2026-09-23)

같은 세션, `-qs4cx-wh`, 전부 켬 config, 실행 순서 A(상주, 환경변수 없음) ×2 → B(`NNTR_MOE_CACHE_EXPERTS=32`) ×2 →
B PROFILE=2+PPL → A PROFILE=2+PPL. 새 바이너리 확인: C=1은 `ExpertLru: one call needs 31 experts resident but the pool
holds 22`로 즉시 종료(첫 층 prefill의 활성 expert가 32가 아니라 31 — 444×4 라우팅에서 하나가 안 뽑힘, 정상).

| | A 상주 | B 파일 경로 C=32 | 판정 |
|---|---|---|---|
| prefill (비프로파일 2회) | 644 / 657 | 688 / 707 | +40~50 (아래) |
| decode TPS | 23.9 / 23.7 | 23.6 / 23.5 | 동일 |
| **ppl** | 62.0916 | **62.0916** | **소수점까지 동일 — 게이트 통과** |
| MoE M>1 host/콜 | 14429 (dsp 13807, transport 621) | 14469 (dsp 13743, transport 726) | dsp 동일, transport +105 |
| M==1 host/콜 | 1443.7 (transport 90.3) | 1437.8 (transport 90.7) | 동일 |
| conv / dense / FC transport | 479 / 537 / 423·570 | 619 / 680 / 562·719 | 전부 +100~140 |
| 등록 | 1520 weights, convert 847 + rpc 727 = 4614 ms | 816, **파일 읽기 3594** + rpc 699 = 6265 ms | 읽기 3.7 GB에 3.6 s = 1.0 GB/s |
| **peak RSS** | **5257 MB** | **894 MB** | 같은 상주인데 −4.4 GB |

- **5.2 GB는 아레나가 아니라 로더의 임시 사본이었다.** 아레나는 ION dma-buf 매핑이라 RSS에 잡히지 않는다. 옛 경로는
  로더가 3.7 GB를 힙으로 읽고 → memcpy → `MADV_DONTNEED`; ru_maxrss는 그 순간의 힙을 기억했다. 새 경로는 pread가
  page cache → ION으로 바로 쓰므로 힙 사본이 없고, 894 MB가 "expert 빼고 전부"(CPU 경로의 814 MB와 같은 급)다.
  §1의 "peak RSS 5.2 GB: 아레나 3840 + 나머지"라는 해석은 틀렸다. **실제 물리 메모리 = RSS + 아레나**(ION은 핀됨,
  회수 불가) + page cache(회수 가능). C=32: 0.89 + 3.84 = 4.7 GB. 앞으로 헤드라인은 `Max RSS`와 프로파일의
  `[HTP] arena chunk … mapped total` 두 줄을 같이 적는다 — RSS는 C와 무관하게 ~0.9 GB로 평평해야 하고, 아레나가
  ⌈22C/48⌉ × 256 MiB로 움직여야 한다.
- **prefill +40~50 중 설명되는 것은 ~7 ms**: 모든 종류의 콜에서 transport가 +100~140 us(23 MoE + 19 conv + 13 FC +
  3 dense ≈ 7 ms). dsp 시간은 동일하므로 DSP 밖이다. 원인 미상(B는 page cache가 3.7 GB 더 차 있는 상태 — 메모리
  압박이 FastRPC 드라이버의 콜당 작업에 닿나?). 나머지 ~35는 세션 드리프트 범위(§2.27 ±25, B가 A보다 뒤에 돌아
  더 뜨거움). **B → A 순서로 한 번 더** 돌려 transport 차가 순서를 따라가는지 보기 전엔 쫓지 않는다.
- **등록 6.3 s (+1.7 s).** 파일 읽기 1.0 GB/s는 memcpy(옛 경로 convert 0.56 ms/weight = 4.4 GB/s)보다 훨씬 느리다 —
  flash에서 온 것(A의 로더가 읽은 뒤 page cache에 남아 있어야 하는데, 3.7 GB 파일 둘 + ION 3.84 + RSS로 12 GB 기기의
  page cache가 못 버텼을 가능성)이거나 uncached 매핑으로의 copy_to_user가 CPU 병목. 2단계의 cold/warm 분리가 이걸
  가른다(warm 미스의 read ms가 memcpy급이면 후자가 아님). 기동 1회 비용이라 우선순위는 낮다.

**단계 0의 정정.** 기기 `config.json`이 단계 0 이전에 이미 `Lfm2CachedSlimMoeForCausalLM`으로 바뀌어 있었다(수동 편집
안내를 먼저 따른 뒤 스크립트가 그 상태를 `.orig`로 백업). 따라서 §10.5의 "순수 CPU 2711 ms / 50.5 TPS / 4866 MB"와
오늘의 0번(3140 / 48.6 / ppl 51.59)은 **둘 다 cached-slim C=32**다. §10.5의 "cached C=32(40.8)가 상주(50.5)보다 20%
느리다"는 항목은 cached 대 cached라 소멸. **순수 CPU(`Lfm2MoeForCausalLM`)는 아직 한 번도 안 쟀고, ppl 51.59 vs 57.00은
cached-slim의 SwiGLU(`acti_func`, 정확한 swish) 대 상주 CPU의 `swiglu_det`(결정적 근사) 차이일 가능성이 남는다.**
2단계 스크립트 0번이 config를 원복하고 순수 CPU + PPL을 한 번 잰다.
