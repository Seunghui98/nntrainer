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

### 10.7 단계 2 실측: 미스 = 1.4 ms, 그 82%가 page cache → ION 복사 — decode는 C=8에서 9.2 TPS (2026-09-23)

같은 세션, 전부 warm(root가 없어 `drop_caches` 거부 — cold는 못 쟀다). 순서: 순수 CPU+PPL → B(C=32) → A → C=2 ×3 + 프로파일
→ C=4 → C=8. 2a(슬롯 재사용 유닛 테스트)는 아직 안 돌렸다 — 아래 "슬롯 재사용" 항목.

| | 물리 메모리 (RSS + 아레나) | prefill ms | decode TPS | ppl | 미스/decode 콜 | 히트율 (무작위 기대) |
|---|---|---|---|---|---|---|
| HTP C=32 | 0.90 + 3.84 = **4.7 GB** | 621 (B 먼저) / 673 (A 뒤) | 24.4 / 23.8 | 62.0916 | 0 | — |
| HTP C=8 | 0.89 + 1.00 = **1.9 GB** | 1566~1575 | **9.2** | 62.0916 | 1.71 | **57%** (25%) |
| HTP C=4 | 0.89 + 0.50 = 1.4 GB | 1537~1710 | 7.6~8.5 | 62.0916 | 2.36 | **41%** (12.5%) |
| HTP C=2 | 0.90 + 0.25 = 1.15 GB | 1597~1604 | 6.1 | 62.0916 | 4.00 | 0% (작업집합 88 > 44슬롯) |
| CPU cached-slim C=8 (§10.5) | 1.87 GB RSS(file-backed) | 2546~2784 | 22~24 | 51.59 | — | — |
| **순수 CPU** (`Lfm2MoeForCausalLM`, 이번에 처음) | peak 5.9 GB | 2460 | **48.2** | **50.6006** | — | — |

**미스 하나의 분해 (프로파일 열, 세 C에서 일관):** 파일 읽기 **1.12~1.38 ms** + register/release rpc **0.24~0.26 ms** =
**1.4~1.6 ms**. decode 토큰당 추가 시간은 전부 이걸로 설명된다: C=8 실측 +67 ms vs 22 × 1.71 × (1.37+0.26) = 62;
C=4 +85 vs 78; C=2 +123 vs 120. prefill도 같다: C=8 +920 vs 503 미스 × 1.64 = 825.

- **읽기가 82%다, RPC가 아니다.** 5.25 MiB / 1.2~1.4 ms = **3.8~4.7 GB/s** — page cache에서 uncached ION 매핑으로의
  단일 스레드 copy_to_user, 옛 경로의 memcpy(4.4 GB/s)와 같은 속도. FastRPC 4회는 0.25 ms(콜당 ≈60 us)뿐이라
  §10.5의 "IDL 배치 등록" 지렛대는 **기각**: 읽기를 0으로 만들어도 C=8 decode 상한은 1000/(42 + 22×1.71×0.26) ≈
  **19 TPS**이고, 읽기가 남는 한 그 아래다. CPU cached-slim이 0.37 ms/미스인 이유는 복사가 없어서다(page cache를
  그대로 매핑) — HTP는 DSP가 dma-buf만 읽을 수 있으니 **복사 한 번이 구조적 비용**이다.
- **히트율은 C/32가 아니었다 — §10.5 정정.** 실제 미스 카운트로 C=4 41%, C=8 57%(무작위면 12.5%, 25%). 국소성이
  있고 공유·핀 LRU가 그걸 잡는다. §10.5의 "LRU = 무작위"는 CPU cached-slim의 시간에서 0.37 ms/미스를 가정해
  역산한 것이었고, 그 레이어는 콜 안에서 자기 expert를 evict할 수 있어(핀 없음) 히트율이 낮게 나온다. 카운트가 맞고
  역산이 틀렸다.
- **RSS는 C와 무관하게 893~898 MB, 아레나가 ⌈22C/48⌉ × 256 MiB로 정확히 움직인다** (C=2 1청크, 4 → 2, 8 → 4). §10.6의
  해석 그대로.
- **ppl 62.0916이 C=2/4/8 전부에서 소수점까지 같다.** 산술 불변이 LRU와 압축(활성 expert만 콜에 넘김) 아래서도 성립.
- **B → A 순서 재확인**: B 621 / 24.4 → A 673 / 23.8. §10.6의 "+40~50"은 순서(열)였다. **B ≡ A**. transport 차도
  쫓지 않는다.
- **슬롯 재사용은 실전에서 안전하다.** C=2 한 실행에 45,687번, 세 실행 합쳐 ~93,000번 슬롯을 재사용했고 ppl이 비트
  단위로 같다. LRU 꼬리 슬롯은 마지막으로 읽힌 뒤 최소 한 층 콜(≥168 MiB DMA)이 지나서야 다시 채워지므로,
  유닛 테스트 2a의 "바로 연속 재사용"은 실제 워크로드보다 가혹한 경우다. 2a는 기록용으로 한 번 돌릴 가치는 있지만
  게이트에서는 뺀다.
- **§10.6의 "등록 파일 읽기 1.0 GB/s"는 틀렸다.** `convert to registry` 열에 아레나 청크 할당(rpcmem_alloc +
  fastrpc_mmap + attach, 청크당 ~180 ms)이 첫 expert의 읽기 시간으로 섞여 든다 — A에서는 같은 시간이 "alloc + other"
  3040 ms에 있었다. 런타임 미스가 보여주듯 읽기는 memcpy급이다. C=2의 rpc 1339 ms(156 weights)가 A의 727보다 큰 것은
  미해결, 기동 1회라 안 쫓는다.

**기준값 정정 — §1은 오늘 기준으로 틀려 있다.** 순수 CPU는 ppl **50.60**(§1: 57.00), decode **48.2 TPS**(§1: 20.6),
prefill 2460 ms(§1: 921~1013). 같은 세션의 HTP 전부 켬은 ppl 62.09, decode 23.8, prefill 621~673. 즉 오늘 기기에서
HTP 경로는 prefill이 CPU의 3.7~4배 빠르고 **decode는 CPU의 절반**이며 ppl은 **+22.7%**(§1이 말한 +8.9%가 아니라).
57.00은 qkv 융합(1f538c2) 이전 값이고 20.6은 어떤 조건이었는지 이 문서들로는 알 수 없다. 이 과제의 결론과는 별개지만
문서 45·51의 "much faster than CPU" 서사가 오늘 수치로는 prefill에만 성립한다는 점은 기록해 둔다. cached-slim(51.59)과
순수 CPU(50.60)의 2%는 SwiGLU 구현(`acti_func` vs `swiglu_det`) 차이로 보이며 우리 경로가 아니다.

**이 과제의 결론(측정 기준).** flash에서 LRU로 가져오는 HTP cached-slim은 **동작하고, ppl이 정확히 같고, 물리 메모리를
4.7 → 1.9 GB(C=8)로 내린다.** 값은 prefill +0.95 s(0.65 → 1.6 s, 704 expert 재복사)와 decode 24 → 9 TPS다. 같은 1.9 GB의
CPU cached-slim(prefill 2.6 s, decode 23 TPS)과 비교하면 prefill은 1.6배 빠르고 decode는 2.5배 느리다. 남은 지렛대는
하나뿐이다: 미스의 82%인 복사를 여러 스레드로 나눠 쓰는 것(§10.8). 그것으로도 decode 상한은 C=8 ≈ 13, 읽기 0이라도 19다.
CPU cached-slim의 23을 넘을 길은 이 구조에 없다. 이 지점에서 "메모리 손잡이로 출하할 것인가"는 측정이 아니라 제품
판단이다.

### 10.8 3단계: 미스의 읽기를 워커 풀로 나눈다 (코드, 기기 미측정)

§10.7이 시키는 유일한 지렛대. `readExpertWeight`가 nibble 3.5 / 1.75 MiB를 `ThreadManager::parallel_for`로 최대 8조각
(NNTR_NUM_THREADS=8이면 448 / 224 KiB씩, 4 KB 정렬이라 두 스레드가 한 페이지를 안 나눔)으로 pread한다. 워커는 throw할 수
없으므로 조각별 errno를 모아 join 뒤 한 번 던진다. scale+colsum 44 KB는 그대로 직렬. 기동 시 preload의 704 expert도
같은 길을 타므로 등록 시간도 따라 내려가야 한다.

기대: 단일 코어의 uncached 쓰기 3.8~4.7 GB/s가 병목이면 4 코어 이상에서 3~4배, 즉 **읽기 1.2~1.4 → 0.3~0.5 ms/미스**.
DDR이 병목이면 그보다 덜. 판정 지표는 프로파일의 `file read … ms/miss`이고, TPS는 그 결과다: 읽기 0.4면 C=8 decode ≈
1000/(42 + 22 × 1.71 × (0.4+0.26)) ≈ **15 TPS**, C=4 ≈ 13. 읽기가 0이어도 19가 상한(§10.7)이라 CPU cached-slim의 23은
이 구조로는 안 넘는다 — 이 측정은 "얼마나 가까이"를 정한다.

측정: `stage3_parallel_read.sh` — C=8, 4 각 warm 2회 + PROFILE=2·PPL 1회, C=32 프로파일 1회(등록 시간·회귀). ppl은
그대로 62.0916이어야 한다(바이트가 같은 자리에 같은 값으로 들어갈 뿐).

### 10.9 3단계 실측: 8스레드로 나눠도 1.12 ms — uncached 매핑 쓰기 경로가 ~4.9 GB/s에서 막힌다 (2026-09-23)

같은 세션, warm, 17ee185(nibble pread를 8조각으로) 적용.

| | 읽기 ms/미스 (2단계 → 3단계) | decode TPS (2단계 → 3단계) | prefill | ppl |
|---|---|---|---|---|
| C=8 | 1.37 → **1.12** (prefill 미스 1.38 → 1.31) | 9.2 → **10.7 / 11.2** | 1130 / 1446 (프로파일 2627) | 62.0916 |
| C=4 | 1.24 → **1.12** (1.37 → 1.16) | 7.6~8.5 → **8.6 / 8.7** | 1538 / 1594 | 62.0916 |
| C=32 | — | 22.7 (프로파일+PPL, 1단계 22.96) | 1820 (1단계 1847) | 62.0916 |

등록: C=32 6265 → 5885 ms, C=8 4522 → 4029. decode 추가 시간은 여전히 미스 열로 설명된다(C=8: +47~52 실측 vs
22 × 1.71 × (1.12 + 0.25) = 51.5; C=4: +73 vs 71).

- **스레드는 답이 아니었다.** 8조각으로 나눠 −18%뿐. 5.25 MiB / 1.12 ms = **4.9 GB/s**, 그리고 이 값은 옛 경로의
  단일 스레드 memcpy(힙 → 아레나, 4.4 GB/s)와 같다. 즉 병목은 코어의 store 속도가 아니라 **비캐시(Normal-NC) ION
  매핑으로의 쓰기 경로 자체**다 — 코어를 늘려도 안 는다. §10.8의 기대(0.3~0.5)는 틀렸고, 코드는 무해하니(−18%) 둔다.
- **남은 선택지는 하나, 목적지를 캐시 가능 메모리로 바꾸는 것.** expert 슬롯 청크만 `RPCMEM_DEFAULT_FLAGS`(cached)로
  잡으면 pread가 memcpy 속도(10 GB/s 이상, ≈0.35 ms)로 끝나고, 대신 DSP가 읽기 전에 CPU 캐시를 DDR로 내려야 한다.
  방법은 둘: (a) dma-buf fd에 `DMA_BUF_IOCTL_SYNC` — 표준 ioctl은 **버퍼 전체**(256 MiB)를 내려 미스당 수 ms라 안 되고,
  Qualcomm/Samsung 커널의 `DMA_BUF_IOCTL_SYNC_PARTIAL`(offset, len)이 있으면 5.25 MiB만 → ≈0.15 ms; (b) 사용자 공간
  `dc cvac` 루프(arm64 EL0 허용) + `dsb` — PoC가 DSP DMA가 보는 지점인지는 기기에서 증명해야 한다. 어느 쪽이든 문서 46
  §34가 일부러 uncached를 고른 결정을 뒤집는 것이라 **`ArenaUncachedWriteAfterMap`의 cached 판(쓰기 → clean → DSP
  checksum) 기기 테스트가 먼저**다. 기대: 미스 ≈ 0.35 + 0.15 + 0.25 = **0.75 ms** → C=8 decode ≈ **14 TPS**, prefill
  ≈ 1.0 s; C=4 ≈ 12. 읽기가 0이어도 19가 상한이다(§10.7).

**이 과제의 최종 표 (2026-09-23, 같은 기기, warm).**

| 구성 | 물리 메모리 | prefill | decode | ppl |
|---|---|---|---|---|
| HTP 상주 (C=32, 파일 경로) | 4.7 GB (RSS 0.9 + 아레나 3.84) | 0.62~0.67 s | 24 TPS | 62.0916 |
| **HTP cached-slim C=8** | **1.9 GB** | **1.1~1.4 s** | **10.7~11.2 TPS** | **62.0916** |
| HTP cached-slim C=4 | 1.4 GB | 1.5~1.6 s | 8.6 TPS | 62.0916 |
| (cached 목적지 실험 시 예상, C=8) | 1.9 GB | ~1.0 s | ~14 TPS | 62.0916 |
| CPU cached-slim C=8 | 1.87 GB (file-backed RSS) | 2.6 s | 22~24 TPS | 51.59 |
| 순수 CPU | 5.9 GB peak | 2.5 s | 48 TPS | 50.60 |

과제가 요구한 것 — "peak RSS 5.2 GB를 C에 비례해 내리기" — 은 됐고, ppl은 모든 C에서 비트 단위로 같다. 그 대가는 decode
24 → 11 TPS(C=8)다. 같은 메모리에서 CPU cached-slim이 decode 2배 빠르고 prefill 2배 느리다는 것이 이 기기에서의
사실이고, HTP 쪽 decode를 14 위로 올릴 길은 이 구조에 없다. 여기서부터는 제품 판단이다: (1) 메모리 손잡이로 그대로 둔다,
(2) cached 목적지 실험을 한다(+3 TPS, 기기 테스트 하나 + 아레나 메모리 타입 변경), (3) 이 경로는 prefill 전용으로 보고
decode를 다른 구성에 맡긴다(가중치 파일 포맷이 달라 별도 과제).

### 10.10 4단계: 정책 시뮬레이터, prefill 선읽기, cached 아레나 (코드, 기기 미측정)

§10.9 뒤의 지렛대 분석에서 DSP를 안 바꾸고 할 수 있는 셋. 전부 환경변수 뒤에 있어 한 빌드로 A/B가 된다. 셋 다 꺼져 있으면
코드 경로는 §10.9와 같다(인터페이스만 `ExpertFileDesc`로 바뀜).

| 스위치 | 무엇 | 기대 (C=8) | 게이트 |
|---|---|---|---|
| `NNTR_MOE_TRACE=<path>` | 층 콜마다 `<layer> <tokens> \| <routed> \| <top-(k+5)/토큰>` 한 줄. `tools/moe_expert_cache_sim.py`가 이걸 ours / lru / lfu / random / **belady(오라클)** 로 재생한다(콜의 expert 핀, 층 공유 풀 = 기기와 같은 제약). 라우팅은 캐시와 무관하므로 상주 실행 한 번이 모든 C·정책에 답한다 | 없음 — 측정 | 시뮬의 `ours`가 §10.7 실측 미스/콜(C=2 4.00, C=4 2.36, C=8 1.71)을 재현해야 시뮬을 믿는다. belady가 57%에서 얼마 안 멀면 정책 작업은 접는다 |
| `NNTR_MOE_PREFETCH=1` | prefill에서 층 L 콜을 보내기 전에 L+1의 비상주 expert만큼 LRU 꼬리(L의 expert·L+1 상주분 제외)를 비우고, 그 슬롯으로의 pread를 스레드 4개로 시작한 뒤 콜 → 콜이 돌아오면 join + register. 겹치는 것은 읽기뿐(register는 같은 FastRPC 세션·DSP 가중치 테이블이라 콜 뒤). 슬롯이 32+32 미만(C=2)이면 아무것도 안 한다 | prefill 미스 503 중 층당 14.4 ms만큼 숨김 → 1.1~1.4 → **≈0.9~1.1 s** | ppl 동일, 프로파일 `expert prefetch: N experts … exposed wait` 줄, M>1 행의 miss/call 감소 |
| `NNTR_HTP_ARENA_CACHED=1` | 아레나 청크를 cached rpcmem으로, 아레나에 쓰는 세 곳(expert pread, WH memcpy, FC slice memcpy) 뒤에 `dc cvac` + `dsb sy`(EL0 허용, PoC까지). 문서 46 §34의 uncached 결정을 뒤집는 것 | 읽기 1.12 → ~0.35+clean, 미스 ≈ 0.75 ms → decode 11 → **≈14 TPS** | **기기 테스트 `HmxArenaSlotReuse.CachedCleaned` 통과가 먼저**, 그 다음 ppl 동일 |

기기 테스트도 고쳤다. 기존 `ArenaSlotReuseMatchesHeap`은 슬롯을 `bake_export`로 채웠는데, 그건 RPC 출력이라 **DSP가** ION 버퍼에
쓰는 것이었다 — 호스트 쓰기를 한 번도 시험하지 않았다(실전 증거는 §10.7의 ~93,000회 재사용). 이제 `HmxArenaSlotReuse`가
bake_export → 힙 벡터 → CPU memcpy로 슬롯을 채우고 세 가지로 돈다: `Uncached`(출하 경로, assert), `CachedCleaned`(스위치의
게이트, assert), `CachedNotCleaned`(보고만: 실패하면 clean이 필요하다는 증명, 통과하면 우연히 write-back됐거나 IO-coherent).

prefetch와 cached를 둘 다 켜면 읽기(≈0.4 ms × 23/층 ≈ 9 ms)가 콜(14.4 ms) 안에 다 들어가 prefill ≈ 0.65 + register 0.13 ≈ **0.8 s**.
decode는 cached만 움직인다. 읽기가 0이어도 decode 상한 19(§10.7)는 그대로다.

### 10.11 4단계 실측: 두 스위치 모두 짐 — 그리고 §10.9의 "쓰기 상한"은 flash 읽기였다 (2026-09-23)

같은 세션, C=8, 순서: 라우팅 trace(상주) → base → prefetch → cached → both → smaps. 기기 테스트(`HmxArenaSlotReuse`)는 돌리지
않고 `cached`로 직행했다 — ppl이 모든 실행에서 62.0916이라 정확성은 실전으로 확인됐지만 결론이 "느림"이라 무의미해졌다.

| C=8 | prefill ms (비프로파일 2회) | decode TPS | 읽기 ms/미스 (decode / prefill) | M==1 dsp us/콜 | ppl |
|---|---|---|---|---|---|
| 상주(trace 실행) | 589 | 23.9 | — | — | — |
| **base** | 3051* / **1030** | **15.1 / 14.9** | **0.39** / 0.65 | 1368.0 | 62.0916 |
| + prefetch | 1349 / 1334 | 14.9 / 14.8 | 0.38 / (prefill 미스 0, 노출 대기 0.64/expert) | 1368.5 | 62.0916 |
| + cached arena | 1067 / 1073 | 14.6 / 14.6 | 0.46 / 0.53 | **1447.6** | 62.0916 |
| + both | 1783 / 1769 | 14.4 / 14.0 | 0.47 / (노출 대기 1.40/expert) | 1479.3 | 62.0916 |

\* 세션 첫 C=8 실행, 이상치로 둔다.

**§10.9 정정 — 병목은 쓰기 경로가 아니라 page cache 적중이었다.** 코드가 같은 base에서 읽기가 1.12 → **0.39 ms/미스**(4.9 →
**14 GB/s**), decode 11 → **15 TPS**로 움직였다. 차이는 앞선 실행: 1단계·2단계 스크립트는 둘 다 `-q40`(CPU 모델, 파일 ~4 GB)을
먼저 돌려 12 GB 기기의 page cache에서 HTP 모델 파일을 밀어냈고, 4단계는 HTP 파일 전체를 읽는 상주 실행으로 시작했다. 4~5 GB/s는
UFS 4.0 순차 읽기 속도다. 따라서 §10.7~10.9의 "읽기 82%, uncached 매핑이 4.9 GB/s에서 막힘"은 **부분적으로 flash에서 읽은 값**이고,
uncached ION은 8스레드로 14 GB/s를 받는다. §10.9의 결론(스레드 무효, cached 목적지가 유일한 길)은 철회한다.

그래서 이 기능에는 **숫자가 두 벌** 있다. 어느 쪽이 실제인지는 page cache가 3.7 GB 파일을 붙잡고 있느냐가 정한다:

| C=8, 물리 1.9 GB (RSS 0.9 + 아레나 1.0) | 미스 | prefill | decode |
|---|---|---|---|
| warm (파일이 page cache에) | 0.39 + 0.25 rpc = **0.64 ms** | **1.0 s** | **15 TPS** |
| flash (메모리 압박으로 page cache에서 밀려남) | 1.1~1.4 + 0.25 = **1.4~1.6 ms** | 1.4~1.6 s | 9~11 TPS |

page cache는 회수 가능한 메모리라 "1.9 GB"에 안 잡히지만, warm 숫자는 그 3.7 GB가 남아 있다는 가정 위에 있다. 메모리를 아끼려는
상황이 곧 page cache가 밀려나는 상황이므로, 제품 판단에는 flash 행이 더 정직하다.

- **cached arena: 기각, 되돌림(8fc9fa4).** 읽기는 오히려 느리고(0.46, dc cvac 포함), **DSP decode 콜이 +80 us(+5.8%)** 느려졌다
  (mm 764 → 813) — CPU-cacheable 버퍼를 DSP DMA가 읽을 때의 일관성 비용으로 보인다. 전제(쓰기 상한)부터 틀렸다.
- **prefetch: 이득 없음, prefill +300 ms.** prefill 미스는 0이 됐지만 노출 대기가 0.64 ms/expert로 동기 읽기(0.65)와 같다 — 읽기가
  콜과 전혀 겹치지 않은 것처럼 보인다. 산술로는 24 expert × 3.1 스레드-ms / 4 스레드 = 18.6 ms가 14.4 ms 콜에 거의 숨어야 한다.
  후보 둘: (a) DSP 콜 중 리더 스레드가 느리다(poll 모드 호출 스레드가 큰 코어를 돌리고 리더가 작은 코어로), (b) CPU 복사와 DSP의
  가중치 DMA가 DDR에서 부딪혀 콜 자체가 느려졌다. 기존 로그의 M>1 행(콜당 host/dsp) 비교가 가른다 — 콜이 느려졌으면 (b)라 구조적,
  아니면 (a)라 스레드 배치 문제. 가를 때까지 기본은 꺼둔다.
- **RSS 분해(C=8, decode 중간)**: Rss 743 MB = 익명 702 MB(비expert 가중치 + KV + 활성) + 파일 30 MB. peak 0.9 GB는 로드 중.
  decode가 FC를 CPU에서 돌리므로(M=1은 HTP 안 탐) 비expert 가중치의 ARM 사본은 뺄 수 없다.

**다음 (측정이 시키는 순서).** ① 시뮬레이터 결과(trace 11,286줄 수집됨): belady가 57%에서 멀면 정책이 가장 싼 지렛대다.
② prefetch 진단(위 M>1 행, 재실행 없음). ③ warm에서 미스의 40%가 rpc(0.25)가 됐으므로 IDL 한 콜 swap(release 2 + register 2 → 1)이
C=8 decode 15 → ~17 TPS — DSP 변경이라 결정이 필요하다. flash 행을 올리는 것은 이 셋이 아니라 히트율(①)뿐이다.

### 10.12 5단계: IDL swap 콜 — 미스 하나 = FastRPC 한 번 (코드, 호스트 체크 통과, 기기 미측정)

§10.11의 warm 미스 0.64 ms 중 0.25가 FastRPC 4회(release 2 + register 2)다. DSP IDL 변경을 승인받아 한 콜로 합쳤다.

- **`weight_swap_u8i4_arena(old_gu, old_dn, K, inter, N_out, arena, off_gu, off_dn, gu_scale, gu_colsum, dn_scale, dn_colsum → h_gu, h_dn)`**
  (IDL 맨 끝에 추가 — 기존 메서드 번호가 안 바뀌어 옛 skel은 이 콜만 에러를 낸다). 새 쌍을 `weight_register_u8i4_arena`로
  등록(bias 0)한 **뒤에** 옛 쌍을 release한다. 옛 쌍은 먼저 검증(둘 다 또는 둘 다 없음, 살아 있음, 아레나 차용, 서로 다름)하고, down
  등록이 실패하면 gate_up을 되돌린다 — 어떤 에러에서도 레지스트리는 그대로다.
- **호스트**: `release_qs4cx_wh_expert`가 DSP를 부르지 않는다. 키만 handle_cache_에서 지우고(즉시 히트 아님) 두 핸들은 슬롯에 실려
  free list로 간다. 그 슬롯에 다음 expert를 pread한 뒤 swap 한 번이 새 쌍 등록 + 옛 쌍 해제. 옛 핸들은 누구도 DSP에 넘길 수 없고
  슬롯 바이트는 콜과 콜 사이에만 덮이므로 안전하다. 로드 시·prefetch 등록도 1회(이전 2회). 프로파일 열 이름은 `swap rpc`.
- **호스트 체크 `swap_host_check.c`**: 실제 `nntr_hvx_mm_u8i4.c`의 진입점을 실제 레지스트리(`hexkl_mm_u8i4_dma.c`) 위에서 돌린다
  (아레나 = 정렬된 호스트 메모리). 제자리 등록·scale/colsum/bias 0 내용·swap 뒤 옛 쌍 소멸·거부 4종·롤백을 확인하고, 변이 둘(롤백
  없음, 등록 전 release)에서 실패한다. qaic가 만들 헤더를 IDL에서 생성(`gen_nntr_hvx_h.py`)해 쓰므로 **skel 정의와 IDL의 시그니처
  일치도 호스트에서 검사된다** — `tools/htp_syntax_check.sh`가 못 하던 것.
- **기기 테스트 `HmxArenaSlotReuse.Swap`**: 옛 쌍이 등록된 채로 CPU가 슬롯을 덮고 swap — 실제 풀의 순서 그대로. 비트 동일 +
  swap된 핸들의 재release 거부.

기대(C=8, warm): rpc 0.25 → ~0.07 ms/미스 → decode 15 → **~17 TPS**, prefill 미스 503 × 0.18 = −90 ms, C=32 등록 −0.35 s(704 ×
1 콜 절약 × 0.48). flash 행(1.1~1.4 ms 읽기)은 거의 안 움직인다(−0.18/1.5 = −12%).

측정(`stage5.sh`): ⓪ 상주 1회로 page cache를 데움 → warm: C=8 base·prefetch, C=4, C=32 등록 → ⓕ CPU 모델을 한 번 돌려 HTP
파일을 page cache에서 밀어낸 뒤 C=8(**flash 행을 코드로 재현**, root 불필요). prefetch 진단용으로 MoE `M>1` 행도 같이 뽑는다.

### 10.13 5단계 실측: swap으로 C=8 decode 15 → 16.8 TPS, prefetch는 호스트 쪽에서 막힌다, cold prefill 3.1 s (2026-09-28)

같은 세션, 상주 1회로 page cache를 데운 뒤 warm C=8 / C=8+prefetch / C=4 / C=32, 이어서 CPU 모델로 page cache를 비운 뒤 C=8.
(첫 실행은 `build_android.sh`를 `--htp` 없이 돌린 앱이라 `htp Context is not registered`로 전부 실패 — 안내 실수, 재빌드 후 재측정.)
ppl은 모든 프로파일 실행에서 **62.0916**.

| | prefill ms | decode TPS | 미스당 읽기 / swap rpc | M>1 MoE host / dsp us/콜 |
|---|---|---|---|---|
| 상주 | 612 | 24.2 | — | — |
| **C=8** | 2082* / **1026** | **16.8 / 16.7** | 0.37 / **0.10** (4단계: 0.39 / 0.25) | 14533 / 13910 |
| C=8 + prefetch | 1183 / 1194 | 16.5 / 16.5 | 0.37 / 0.11, 노출 대기 0.45/expert | **21564** / 14018 |
| C=4 | 1066 / 1071 | 14.3 / 13.9 | 0.41 / 0.11 | 14702 / 14047 |
| C=32 (프로파일+PPL) | 2027 | 21.9 | 등록 6.2 s | 14536 / 13857 |
| **C=8, CPU 모델 직후(cold)** | **3137** (프로파일) | 15.4 | prefill 미스 **4.40** / decode 미스 0.38 | 15564 / 14711 |
| C=8, 그 다음 실행 | 1034 | 15.7 | — | — |

\* 상주 실행 직후 첫 C=8 실행. 4단계(3051)에서도 같은 자리에서 나왔다 — 원인 미상, 두 번째 실행을 쓴다.

- **swap: 예측대로.** FastRPC가 미스당 0.25 → **0.10 ms**(4회 → 1회). 미스 = 0.37 + 0.10 = **0.47 ms**(0.64에서). C=8 decode
  15.0 → **16.8 TPS**(+11%, 예측 ~17). 토큰당 추가 시간이 미스 열로 다 설명된다: C=8 +18.1~18.6 ms vs 22 × 1.71 × 0.47 = 17.8,
  C=4 +28.7~30.4 vs 26.8. 이제 warm 미스의 **79%가 읽기(page cache → ION 복사)**이고 rpc는 21%다.
- **prefetch: 기각, 원인 확정.** 켜면 층 콜의 **DSP 시간은 그대로**(13910 → 14018)인데 **호스트 시간이 콜당 +7.0 ms**(transport 624 →
  7545) 늘고, 리더 스레드도 콜 동안 거의 진행하지 못한다(노출 대기 0.45/expert = 동기 읽기 0.46). DSP의 가중치 DMA와 DDR에서
  부딪힌 것이 아니라(그랬다면 dsp 시간이 늘었어야 한다) **FastRPC의 호스트 경로와 리더 스레드가 호스트 CPU·메모리 쪽에서 서로를
  막는다**. 두 세션 연속 prefill +160~300 ms. 숨길 수 있는 최대치도 prefill 읽기 0.23 s뿐이라 더 파지 않는다 — 코드를 뺀다.
- **cold(flash) 행을 처음 제대로 쟀다.** CPU 모델을 돌려 HTP 파일을 page cache에서 밀어낸 직후 prefill 미스 읽기가 **4.40 ms**
  (5.25 MiB 단위 UFS 읽기 1.25 GB/s) → prefill **3.1 s**. 그런데 prefill이 704 expert를 전부 읽으면서 파일이 다시 page cache에
  올라와 **decode는 곧바로 warm**(미스 0.38 ms, 15.4 TPS)이고 다음 실행 prefill은 1.03 s다. 즉 이 기기에서 cold 비용은 **첫
  prefill의 +2 s 한 번**이다. 2~3단계에서 decode 미스가 1.1~1.4 ms였던 것은 CPU·HTP 모델 파일 두 개(~8 GB)가 page cache를
  다투던 상태였기 때문이고, 파일이 하나일 때는 decode 중 page cache가 유지된다. 지속적인 메모리 압박에서의 decode는 그 사이 어딘가다.
- **C=32 등록 6.2 s**는 이전(5.9~6.3)과 구분되지 않는다(예측 −0.35 s는 청크 할당·FC 등록 등 다른 항목의 편차 아래). 분해 행은
  이번 grep에 없었다.
- 기기 유닛 테스트 `HmxArenaSlotReuse.Swap` 결과는 아직 못 받았다. 실전에서는 이 세션만 swap ~6만 회가 비트 동일 ppl로 돌았다.

**현재 최종 표 (warm, 같은 기기).**

| 구성 | 물리 메모리 | prefill (warm / cold) | decode | ppl |
|---|---|---|---|---|
| HTP 상주 | 4.7 GB | 0.61 s | 24.2 TPS | 62.0916 |
| **HTP cached-slim C=8** | **1.9 GB** | **1.03 s / 3.1 s** | **16.8 TPS** | 62.0916 |
| HTP cached-slim C=4 | 1.4 GB | 1.07 s | 14.1 TPS | 62.0916 |
| CPU cached-slim C=8 (§10.5) | 1.87 GB | 2.6 s | 22~24 TPS | 51.59 |
| 순수 CPU (§10.7) | 5.9 GB peak | 2.46 s | 48.2 TPS | 50.60 |

**남은 지렛대는 히트율 하나다.** 미스 비용은 읽기 0.37 + rpc 0.10에서 더 줄일 구조적 수단이 이 과제 안에 없다(cached arena, 스레드,
prefetch 모두 측정으로 기각). decode를 더 올리는 것은 미스 수 — C=8에서 1.71/콜, 히트 57% — 이고, 수집된 trace로 돌리는
시뮬레이터가 그 여지(belady와의 차이)를 말해 준다.

### 10.14 6단계: C=1 — 풀보다 큰 콜은 나눠 보낸다 (코드, 기기 미측정)

C=1은 공유 풀이 22칸(22층 × 1)인데 prefill 층 콜은 31~32 expert를 동시에 상주시켜야 해서 지금까지 `ExpertLru`가 거부했다(§10.1의
"C 최솟값 2"). 측정 요청에 따라 두 가지를 넣었다.

- **풀보다 큰 콜은 풀 크기 묶음으로.** `tryMoeLayerOnAccelerator`의 "expert 묶음 하나를 상주시키고 커널 배열을 만드는" 부분을
  `stage(group)`으로 빼고, 활성 expert 수가 풀 용량보다 크면 용량 단위로 나눠 콜을 여러 번 보낸다(C=1 prefill: 22 + 9~10). 커널은
  콜마다 출력 전체를 0으로 채우고 자기 expert의 행만 scatter-add하므로, 첫 묶음은 출력에 바로, 나머지는 임시 버퍼에 받아 호스트에서
  더한다. **fp32 덧셈 순서가 묶음 경계에서 바뀌므로 C=1의 ppl은 끝자리가 달라질 수 있다** — 이 경로 말고는 모두 비트 동일. decode
  콜(4 expert)과 C≥2는 이 분기를 안 탄다.
- **아레나를 풀 크기에 맞춤.** 청크가 항상 256 MiB였으므로 C=1(22 × 5.25 = 116 MiB)도 C=2와 같은 메모리를 잡았다. 레이어가 풀
  용량을 `reserve_qs4cx_wh_expert_slots(n)`로 한 번 알려 주면 새 청크를 남은 슬롯 수에 맞춰(64 MiB 단위 올림) 잡는다 → C=1은
  **128 MiB**. C=2(231 → 256), C=4·8(256 청크들)은 그대로다.

기대(warm): 물리 ≈ 0.9 + 0.125 = **1.0 GB**. decode는 작업 집합 88 > 22칸이라 거의 전부 미스(C=2 실측 4.00/콜) → 1000/(41.4 + 22 × 4
× 0.47) ≈ **12 TPS**. prefill은 층 콜 44회(+22 × ~1.7 ms)와 미스 ~700 × 0.47 ≈ +0.37 s → ≈ **1.1~1.2 s**. ppl은 62.09 근처.

### 10.15 6단계 실측: C=1은 1.0 GB에 12 TPS, 그러나 ppl 63.03 — 분할 경로가 비트 동일이 아니다 (2026-09-28)

같은 세션, warm(상주 실행으로 page cache를 데운 뒤), C=1 → C=2 → C=8 순서로 각 2회 + PROFILE=2·PPL 1회.

| 구성 | 아레나 | 물리 (RSS 0.87 GB + 아레나) | prefill warm | decode | decode 미스/콜 | ppl |
|---|---|---|---|---|---|---|
| C=1 | 128 MiB | **1.0 GB** | 1.13 s | 12.1 TPS | 4.00 | **63.0346** |
| C=2 | 256 MiB | 1.13 GB | 1.10 s | 11.0 TPS | 4.00 | 62.0916 |
| C=8 | 960 MiB | 1.81 GB | 1.06 s | 15.2 TPS | 1.71 | 62.0916 |

- **아레나를 풀에 맞춘 것은 동작한다.** C=1 `arena chunk 0: 128 MiB`, C=2·8은 이전과 같다. peak RSS는 셋 다 873 MiB.
- **prefill 분할은 예상대로 나갔다.** C=1의 M>1 행 22층 × 2 = 44콜, rows 19536 = 2 × 9768(C=2의 23콜 중 22콜과 같은 M=444). 콜당
  7.7 ms로 C=2(14.6 ms)의 반쯤이고 합은 같다(338 vs 336 ms). 미스 653, 미스당 0.46 ms.
- **ppl이 끝자리가 아니라 1.5% 움직였다(62.0916 → 63.0346).** 생성도 갈라졌다(C=1은 498 토큰에서 EOS, C=2·8은 512). §10.14의
  "덧셈 순서만 바뀐다"는 이 크기를 설명하지 못한다. 알려진 것:
  - 커널은 분할에 불변이다 — 호스트 하네스(`moe_layer_host_check.c`, 이번에 추가)에서 expert {0,1,2} + {3,4}로 나눠 더한 결과가
    한 번에 부른 결과와 최대값 대비 1.3e-7로 같다. 커널은 콜마다 출력 전체를 0으로 채우고(`hexkl_mm_u8i4_moe.c` memset out_c),
    활성값 양자화는 행 단위다. 단, 하네스는 HMX 산술을 스텁으로 바꾼 것이라 기기 산술은 검증하지 않는다.
  - 레이어 쪽에서 입력과 출력은 다른 버퍼이고, 슬롯 배치(`place`)는 겹치지 않는다(읽어서 확인, 실행 검증 아님).
  - C=1 분할에만 있는 접근 패턴이 하나 있다: **두 번째 묶음이 방금 끝난 첫 콜이 읽은 슬롯을, 그 사이 다른 DSP 콜 없이 곧바로
    다시 채우고 쓴다.** C≥2에서는 한 층 콜이 읽은 슬롯이 다시 채워지기 전에 항상 attention 등 다른 HTP 콜이 끼어 있다. 기기
    `HmxArenaSlotReuse.Swap`(CPU가 슬롯을 다시 채우고 swap하는 경우)의 결과는 아직 없다.
  - 가를 실험: `NNTR_MOE_SPLIT=<n>`(이번에 추가, 측정 스위치)로 C=2에서 분할만 켠다. C=2 풀(44)은 층 하나(32)를 다 담으므로 두
    번째 묶음은 이전 층의 슬롯만 다시 채운다. `C=2 SPLIT=22`(C=1과 같은 묶음)와 `SPLIT=16`이 **62.0916이면 분할 산술은 무죄이고,
    읽은 직후 다시 채운 슬롯을 DSP가 옛 내용으로 읽는 것**이다 — 이 경우 C≥2에도 잠재된 정합성 버그다. **63.03 근처면 분할 자체가
    기기 산술을 바꾼다.**
- **decode: C=1과 C=2는 둘 다 히트 0이다.** 미스가 정확히 4.00/콜. 공유 풀 22C칸이 토큰 하나의 작업 집합 88(22층 × 4)보다 작으면
  22층을 도는 순환 접근에서 LRU는 다음 사용 전에 모든 expert를 내보낸다. 그래서 **전역 LRU에서 C=2는 C=1보다 메모리만 더 쓰고
  얻는 것이 없다.** C=1 decode 토큰당 82 ms ≈ 22 × (DSP 1.35 + 읽기 4 × 0.38 + swap 0.40) = 72 ms + 비-MoE.
- **decode 수치는 실행 순서에 따라 ~10% 치우친다.** 같은 일을 하는 decode 콜의 DSP 시간이 실행 순서대로 1267 → 1317 → 1452 us(mm
  764 → 771 → 832)로 올랐다 — 기기 온도. C=2가 C=1보다 느린 것과 C=8이 §10.13의 16.8보다 낮은 15.2인 것 모두 이것이다.
- C=1 첫 실행 prefill 2.48 s는 §10.13의 한 번짜리 cold와 같은 것이다(직전 상주 실행이 파일 페이지를 밀어냄). 두 번째부터 1.13 s.

**다음.** ppl 문제가 먼저다 — C=1이 "1.0 GB에 12 TPS"라는 결과는 ppl이 62.0916으로 돌아와야 성립한다. 위 `NNTR_MOE_SPLIT` 실험
하나로 두 가설이 갈린다. 작은 C의 히트 0은 정책 문제다(같은 층 안에서 먼저 내보내는 등 층마다 몇 칸을 지키는 정책이면 연속 토큰의
겹침이 히트가 된다). 코드 전에 trace로 시뮬레이터를 돌려 C=1·2에서 그 여지를 먼저 잰다.

### 10.16 분할 실험: ppl을 바꾸는 것은 분할 자체다 — 옛 슬롯 읽기가 아니다 (2026-09-28)

`NNTR_PPL=1`, 같은 세션.

| 실행 | 묶음 | ppl | 생성 토큰 |
|---|---|---|---|
| resident | — | 62.0916 | 512 |
| C=2 | 32 한 콜 | 62.0916 | 512 |
| C=2 `NNTR_MOE_SPLIT=22` | 22 + 9~10 | **63.0346** | 498 |
| C=2 `NNTR_MOE_SPLIT=16` | 16 + 16 | **63.4937** | 501 |
| C=1 (두 번) | 22 + 9~10 | 63.0346, 63.0346 | 498, 498 |

- C=2의 22 묶음은 방금 읽은 슬롯을 다시 채우지 않는데도 C=1과 **끝자리까지 같다**. 두 번 돌린 C=1도 같다. §10.15의 두 번째
  가설(옛 슬롯 읽기)은 기각이다.
- 묶는 방식이 바뀌면 값이 바뀐다(22 → 63.03, 16 → 63.49). fp32 덧셈 순서만의 차이라면 이 크기가 나올 수 없다. 남는 것은
  **한 행의 출력이 같은 콜에 든 다른 expert들에 따라 달라지는 것**, 즉 커널이다. 호스트 하네스(§10.15)는 이것을 못 본다 — DMA
  스텁이 즉시 끝나서 실제 비동기 파이프라인(가중치 DMA, 백그라운드 pack, 에필로그 잡)의 타이밍이 없기 때문이다.
- **어느 쪽이 틀렸는지는 아직 모른다.** HTP ppl 62.09는 CPU 50.60보다 22.7% 높고(§10.7) 그 차이는 설명된 적이 없다. 기존 기기
  테스트 `MoeLayerMatchesTwoCallReference`는 한 콜과 expert별 두 콜 참조를 비트 단위로 비교하지만 규모가 4 expert·M=200이다.
- 추가: 기기 테스트 `HmxMmU8I4Layer.MoeLayerSplitMatchesWhole` — 모델 규모(32 expert, M=444, top-4, 가중치 4쌍 순환). (1) 한
  콜을 expert별 참조와, (2) 22+10·16+16 분할을 한 콜과 비교한다. 한 묶음만 건드린 행은 `x + 0.0f == x`이므로 분할이 한 콜과
  **비트 동일해야 한다**. 호스트에서는 IDL로 생성한 헤더에 대해 문법만 확인했고 기기 빌드·실행은 안 했다.
  - `one_group_rows`가 0이 아니면 커널이 같은 콜의 다른 expert에 따라 행 출력을 바꾼다.
  - `whole_vs_ref`가 0이 아니면 **한 콜(현재 모든 C의 경로) 쪽이 32 expert에서 틀린 것**이고, 62.09와 50.60 사이 차이의 일부일
    수 있다.

### 10.17 C=1~8 한 세션 스윕: prefill·decode·히트율 (2026-09-28)

warm(맨 앞 C=8 한 번으로 page cache 채움), C 사이 60 s 휴식, C마다 계측 없는 실행 1회(prefill·decode) + `NNTR_HTP_PROFILE=2
NNTR_PPL=1` 1회(미스·ppl·메모리). peak memory는 앱이 찍는 `ru_maxrss`(최대 RSS)이고 ION 아레나는
여기 안 잡히므로 물리 = peak + 아레나. prefill 히트는 1 − 미스/704(22층 × 32)로 어림값이다.

| C | peak memory KB (RSS) | 물리 MB (peak + 아레나) | prefill ms | prefill TPS | decode TPS | decode 히트 | prefill 미스 (히트≈) | ppl |
|---|---|---|---|---|---|---|---|---|
| 1 | 894288 | 1001 | 962 | 461.5 | 12.59 | 0.0% | 653 (7%) | 63.0346 (분할) |
| 2 | 894668 | 1129 | 963 | 461.1 | 12.27 | 0.0% | 631 (10%) | 62.0916 |
| 3 | 894880 | 1257 | 894 | 496.6 | 11.99 | 0.0% | 609 (13%) | 62.0916 |
| 4 | 894948 | 1385 | 878 | 505.7 | 14.78 | 41.0% | 587 (17%) | 62.0916 |
| 5 | 894928 | 1513 | 867 | 512.1 | 15.11 | 47.1% | 566 (20%) | 62.0916 |
| 6 | 894980 | 1577 | 969 | 458.2 | 15.15 | 50.2% | 544 (23%) | 62.0916 |
| 7 | 894884 | 1705 | 846 | 524.8 | 15.19 | 51.3% | 524 (26%) | 62.0916 |
| 8 | 894848 | 1833 | 845 | 525.4 | 15.91 | 57.2% | 503 (29%) | 62.0916 |
| 상주 (§10.13) | — | ~4700 | 610 | ~730 | 24.2 | 100% | 0 | 62.0916 |

- **decode는 C=3과 C=4 사이에서 계단이 진다.** 풀 22C가 토큰 하나의 작업 집합 88(22 × 4)보다 작은 C ≤ 3은 순환 접근에서 LRU가
  전부 내보내 히트 0, 12 TPS 근처다. C=4(풀 88)부터 히트 41% → 14.8 TPS이고, 그 뒤로는 C당 +2~6%p 히트, +0.1~0.7 TPS로 완만하다.
  C ≤ 3은 메모리만 더 쓰고 이득이 없다.
- **prefill은 C에 거의 둔감하다(845~969 ms).** 층마다 거의 모든 expert를 쓰므로 미스가 503~653로 모두 많고, 미스 150개 차이
  × 0.45 ms ≈ 70 ms가 C=1과 8의 차이 전부다. C=6의 969 ms는 단일 실행 편차로 본다(히트는 순서대로 늘었다).
- decode 한 콜의 DSP 시간이 실행 순서대로 1264 → 1390 us로 올랐다(60 s 휴식에도 기기 온도). 뒤쪽 C가 decode에서 몇 % 손해를 본
  상태의 수치다.
- **운영점: C=4(1.4 GB, 14.8 TPS) 또는 C=8(1.8 GB, 15.9 TPS).** 1~3은 쓸 이유가 없다.

### 10.18 prefill 선읽기 재검토 1단계: 리더가 층 콜 동안 진행하는가 (코드, 기기 미측정)

**동기.** prefill 미스 처리(C=8, 층당 약 22개 × 0.58 ms = 12.7 ms)는 DSP 층 콜(14 ms) 앞에서 직렬로 돌아, 상주 대비 +0.27 s가
전부 여기서 나온다(§10.17). 층 N 콜 동안 층 N+k의 expert를 읽어 두면 숨길 수 있다. C=8은 176칸이라 층 N~N+k를 동시에 담는
k ≤ 4까지 가능하고, k=4면 층 하나의 읽기(약 10 ms)를 콜 4개(56 ms)에 나눠 할 수 있어 콜 중 진행률 25%면 충분하다.

**걸림돌.** §10.13에서 k=1 선읽기는 콜 중 리더 진행률이 사실상 0이었고(노출 대기 0.45 ms/expert = 동기 읽기) 콜의 호스트
시간이 +7 ms였다. 깊이를 늘려도 진행률이 0이면 숨길 수 없으니, 먼저 원인을 가른다.

| 후보 | 그렇다면 |
|---|---|
| (a) 스케줄링: 리더 스레드와 FastRPC 호출 스레드가 같은 코어를 다툼 | 리더 코어 고정으로 해결 |
| (b) uncached ION 쓰기 경로가 콜 중 막힘 | 힙으로 읽고 ION 복사는 콜 사이로 |
| (c) FastRPC 드라이버가 콜 중 다른 스레드를 막음 | 구조적, 선읽기 포기 |

**추가한 것(측정 스위치, 기본값 아님).** `NNTR_MOE_PREFETCH=1`일 때만 쓰인다.

- `NNTR_MOE_PREFETCH_READERS=<n>`: 리더 스레드 수(기본 4).
- `NNTR_MOE_PREFETCH_CPUS=<a,b,..>`: 리더를 이 코어들에 고정.
- `NNTR_MOE_PREFETCH_HEAP=1`: 리더가 ION 칸 대신 힙 버퍼로 읽고, 콜이 끝난 뒤 힙 → ION 복사 후 등록. 힙 버퍼는 층당 선읽기
  수만큼(최대 32 × 5.25 MiB = 168 MiB) 한 번 잡아 계속 쓴다.
- 프로파일(`NNTR_HTP_PROFILE≥1`) 줄 `expert prefetch progress: X of N read when the call returned (P%), heap->ION copy,
  caller cpus {..}, reader cpus {..}`.

**판정.** 진행률 ≥ 30%이면서 MoE `M>1` 행의 transport 증가 ≤ 1 ms인 변형이 있으면 2단계(깊이 k 파이프라인)로 간다. 없으면
선읽기는 이 기기에서 접고, 층당 swap을 한 RPC로 묶는 배치 swap만 남긴다.

### 10.19 1단계 실측: 리더는 진행한다 — 막은 것은 호출 스레드와 같은 코어였다 (2026-09-28)

C=8, `NNTR_HTP_PROFILE=2 NNTR_PPL=1`, 실행 사이 30 s. 기기 코어: cpu0~5 3.53 GHz, cpu6~7 4.47 GHz. prefill 시간은 프로파일
오버헤드가 붙은 값이라 실행끼리만 비교한다. ppl은 전부 62.0916.

| 실행 | 리더 코어 (실측) | 콜 끝날 때 읽은 비율 | 층 콜 transport (base 대비) | 콜 뒤 남은 대기 | heap→ION 복사 |
|---|---|---|---|---|---|
| base (선읽기 끔) | — | — | 0.52 ms | (동기 미스 읽기 214 + swap 49 ms) | — |
| p0 기본 | {6} | 81% | 7.84 ms (+7.3) | 81 ms | — |
| p1_low `CPUS=0,1,2,3` | {0,1,2,3} | 52% | 0.60 ms (**+0.08**) | 167 ms | — |
| p1_high `CPUS=4,5,6,7` | {4,5,6,7} | 72% | 4.72 ms (+4.2) | 97 ms | — |
| p2_heap | {6} | 77% | 7.99 ms (+7.5) | 118 ms | 136 ms |
| p3_heap_high | {4,5,6,7} | 70% | 4.91 ms (+4.4) | 115 ms | 136 ms |
| p4 `READERS=1` | {6} | 52% | 2.75 ms (+2.2) | 210 ms | — |

- **§10.13의 "리더가 콜 동안 진행하지 못한다"는 틀렸다.** 기본 설정에서도 81%를 읽었다. 콜을 +7 ms 늦춘 것은 **리더가 전부
  호출 스레드와 같은 cpu6에서 돌았기 때문**이다. `ThreadManager`가 메인 스레드를 가장 빠른 코어(cpu6)에 고정하고
  (`thread_manager.cpp` `pinSelfToCore(core_map[0])`), `std::thread`는 만든 스레드의 친화도를 물려받는다. 호출 스레드가
  `caller cpus {6}`로 찍혔고, cpu6에 얹힌 리더 양에 따라 transport가 늘었다: 리더 4개 +7.3, 1개 +2.2, 4~7에 나눠 +4.2, 0~3 +0.08.
- **(b) uncached ION 쓰기는 걸림돌이 아니다.** 힙으로 읽어도 진행률이 같고(77% vs 81%) 복사 136 ms만 늘었다. HEAP은 기각.
- **호출 코어를 피하면 콜은 느려지지 않지만, 한 콜 동안 층 하나를 다 읽지는 못한다.** cpu0~3 리더 4개가 14.6 ms 콜 동안 24개 중
  52%(≈12.5개)를 읽었다. 깊이 1에서는 나머지가 콜 뒤에 노출되어 순이득이 약 −27 ms(대기 167 + 등록 59 + 콜 +8 = 236 vs base 263)로
  잡음 안이다. **깊이 2 이상이면 층 하나의 읽기(24개)를 콜 2개(≈25개분)에 나눌 수 있어 전부 숨는다.** C=8(176칸)은 층 N~N+4를
  함께 담을 수 있어 깊이 4까지 가능하다. C=4(88칸)는 깊이 1이 한계다.
- 판정(§10.18): p1_low가 기준(진행률 ≥ 30%, transport ≤ +1 ms)을 넘었다 → 2단계(깊이 k 파이프라인)로 간다. 리더는 호출 스레드의
  코어를 뺀 코어에 둔다. 예상(warm, 계측 없음): prefill 845 → 약 640 ms(읽기 214 ms를 숨기고 등록 ~55 ms는 콜 사이에 남음),
  배치 swap까지 하면 약 600 ms(상주 579 ms).

### 10.20 2단계: 깊이 k 선읽기 파이프라인 (코드, 기기 미측정)

§10.19에서 리더가 호출 코어를 피하면 콜을 늦추지 않고 콜 하나 동안 층 하나의 절반(≈12개)을 읽었다. 그래서 층 하나를 콜 2~4개에
나눠 읽도록 여러 층을 동시에 띄운다.

- **`NNTR_MOE_PREFETCH=<k>`** (0/없음 = 끔, 이전의 `=1`은 이제 깊이 1). prefill에서 층 L은 자기 배치를 등록·스테이징한 뒤,
  층 L+1 ~ L+k 중 아직 큐에 넣지 않은 층의 비상주 expert를 큐에 넣는다. 층 0이 prefill의 시작이라 이전 prefill이 남긴 배치를
  먼저 비우고 큐 위치를 1로 되돌린다.
- **칸 계산.** 큐에 넣을 때 `makeRoom`으로 이번 콜의 expert와 층 L+1 ~ 대상 층의 상주 expert를 빼고 가장 오래된 것부터 비운 뒤,
  읽는 중인 칸 수를 `ExpertLru::hold()`로 잡아 둔다. 그동안 다른 미스가 그 칸을 가져가지 않는다. 층이 자기 배치를 받을 때
  `unhold()` 후 LRU에 넣는다. 칸이 모자라면 그 층에서 큐잉을 멈추고 다음 층에서 다시 시도한다. C=8(176칸)은 이번 층 32 +
  4개 층 128 = 160이라 k=4까지 들어간다. C=4(88칸)는 k=1이 한계다.
- **백엔드.** `prefetch_qs4cx_wh_experts_begin`은 배치를 큐에 넣고 바로 돌아오고, `_end`는 가장 오래된 배치를 기다려 등록한다.
  리더 스레드(기본 4개, `NNTR_MOE_PREFETCH_READERS`)는 한 번 만들어 계속 쓰고, **호출 스레드의 코어를 뺀 모든 코어**에 고정한다
  (`NNTR_MOE_PREFETCH_CPUS`로 덮어쓰기). 등록(swap RPC)은 여전히 콜 사이에서만 한다. §10.18의 HEAP 경로는 기각되어 뺐다.
- 프로파일 줄 `expert prefetch progress`는 이제 **층이 자기 배치를 요청한 순간 이미 읽혀 있던 비율**이다. 100%에 가까우면 읽기가
  다 숨은 것이다.
- `unittest_expert_lru`에 hold/unhold 테스트 1개 추가. 백엔드는 호스트 스텁으로 컴파일만 확인했다.

예상(C=8, warm, 계측 없음): k=1은 §10.19대로 이득이 거의 없고, k=2부터 진행률 ≈100%로 prefill 845 → 약 640 ms. 남는 것은 콜
사이의 등록 약 55 ms다.

### 10.21 2단계 실측: prefill 892 → 690 ms (k=2), 선읽기 100% (2026-09-28)

C=8, warm, 실행 사이 30 s. 속도는 계측 없는 실행, 분석은 `NNTR_HTP_PROFILE=2 NNTR_PPL=1` 실행이다. ppl은 전부 62.0916.

| k | prefill (계측 없음) | decode | 층이 요청할 때 읽혀 있던 비율 | 노출 대기 | 등록 | 층 콜 host / dsp / transport |
|---|---|---|---|---|---|---|
| 0 | 892 ms | 16.89 TPS | — (동기 미스 503개: 읽기 270 + swap 51 ms) | — | — | 14.35 / 13.84 / 0.51 ms |
| 1 | **693 ms** | 16.91 | (프로파일 없음) | | | |
| 2 | **690 ms** | 16.86 | **100%** (528/528) | **0.0 ms** | 53.5 ms | 15.83 / 14.71 / 1.13 ms |
| 4 | 738 ms | 16.80 | 100% (528/528) | 0.0 ms | 54.4 ms | 16.15 / 14.80 / 1.35 ms |
| 상주 (§10.13) | 579~610 ms | 24.2 | | | | |

- **prefill −23%(892 → 690 ms).** 읽기는 전부 숨었다(동기 미스 0, 노출 대기 0). 리더는 호출 코어 cpu6을 피해 {0..5,7}에서 돌았다.
- **k=1도 k=2와 같다.** §10.19에서 깊이 1이 절반만 읽은 것은 콜이 끝나자마자 join했기 때문이다. 이제는 층 L+1의 배치를 층 L+1이
  실제로 요청할 때까지(사이의 attention·CPU 작업 동안에도) 읽으므로 깊이 1로도 충분했다. 다만 k=1은 프로파일 행이 없어 진행률은 모른다.
- **k=4는 오히려 느리다(738 ms).** 층 0에서 1~4층 128개를 한꺼번에 큐에 넣어 리더가 앞쪽 콜들과 더 오래 겹친다. 그 비용이 아래다.
- **남은 비용 두 가지(k=2 기준, 상주 대비 약 +110 ms).**
  - 등록(swap RPC) 53.5 ms: 콜 사이에서만 할 수 있어 선읽기로 못 숨긴다.
  - **층 콜이 +1.5 ms 느려진다(×23 ≈ +34 ms).** transport +0.6 ms에 DSP 시간도 +0.9 ms인데, 늘어난 것은 DDR을 만지는 단계들이다
    (quant 247 → 326, requant 47 → 179, dequant 216 → 451, scatter 27 → 74 us; mm은 거의 그대로). 리더의 uncached ION 쓰기가 DSP와
    DDR 대역폭을 다투는 것으로 보인다. k=4에서 더 커진다(+1.8 ms).
- decode는 영향 없음(16.8~16.9 TPS). 이번 세션 decode가 §10.17의 15.9보다 높은 것은 기기 온도 차이다(같은 세션 안에서만 비교).

**다음.**
1. **리더 수를 줄여 DDR 경합 줄이기** — 진행률 100%에 노출 대기 0이라 여유가 있다. k=2에서 `NNTR_MOE_PREFETCH_READERS=1,2`를
   재서 콜 지연(+1.5 ms)이 줄고 여전히 100%인지 본다. 코드 변경 없음.
2. **cold(flash)에서 재기** — 미스 하나가 4.4 ms라 층 하나를 읽는 데 콜 여러 개가 필요하다. k=2와 4의 차이는 여기서 드러날 것이다.
3. **배치 swap(IDL 변경)** — prefill 등록 53 ms → 수 ms. decode는 콜당 미스가 1.71개뿐이라 묶어도 토큰당 약 1.5 ms(+2~3%)만 준다.
   §10.19의 "decode 16.8 TPS" 예상은 과했다.

### 10.22 k 전부 프로파일 재측정: k=1·2·4 모두 −22%, k는 warm에서 차이 없음 (2026-09-28)

같은 조건(C=8, warm, 30 s 간격), 이번에는 모든 k에 프로파일 실행을 붙였다(`prefetch_k.sh`, 요약 `s10_summary.awk`).

| k | prefill | decode | 준비율 | 노출 대기 | 등록 | 층 콜 host / dsp / transport (us) |
|---|---|---|---|---|---|---|
| 0 | 866 ms | 15.74 | — (동기 미스 503: 읽기 207 + swap 50 ms) | — | — | 14326 / 13808 / 519 |
| 1 | 673 ms | 16.87 | 100% | 0.0 | 55.8 ms | 15791 / 14659 / 1132 |
| 2 | 672 ms | 16.65 | 100% | 0.0 | 55.8 ms | 16051 / 14702 / 1349 |
| 4 | 665 ms | 16.80 | 100% | 0.0 | 53.5 ms | 15526 / 14703 / 823 |

- **k=1, 2, 4가 prefill 665~673 ms로 같다.** §10.21의 k=4 738 ms는 편차였다. warm에서는 k=1로 충분하다.
- 절감 193 ms ≈ 숨긴 읽기 207 ms − 층 콜 지연(+1.2~1.7 ms × 23 ≈ 30~40 ms) + 편차.
- **층 콜의 DSP 시간은 k와 상관없이 +0.85~0.9 ms**다(quant·requant·dequant·scatter가 늘고 mm은 그대로). 리더가 도는 동안 DSP의 DDR
  단계가 느려진다는 §10.21의 해석과 맞는다. transport는 실행마다 0.8~1.3 ms로 흔들린다.
- 상주(약 580 ms)까지 남은 약 90 ms = 등록 약 55 ms + 층 콜 지연 약 35 ms.
- decode 히트 57.2%로 모두 같고, t_k0의 decode 15.74는 첫 실행이라 낮게 나온 편차로 본다(프로파일 실행은 넷 다 16.4~16.5).

### 10.23 배치 swap: 한 번의 RPC로 여러 expert 등록 (코드, 호스트 체크 통과, 기기 미측정)

§10.22에서 상주 대비 남은 약 90 ms 중 55 ms가 prefill 동안의 등록(expert 528개 × swap RPC 약 0.1 ms)이었다. decode도 콜당 미스
1.71개를 각각 RPC 하나로 등록한다(토큰당 약 3.8 ms).

- **IDL `weight_swap_batch_u8i4_arena`** (맨 뒤에 추가, 기존 번호 유지). expert n개의 old 쌍·아레나 위치·스케일·colsum을 평평한 배열로
  받아 skel이 기존 단일 swap을 차례로 부른다. expert i에서 실패하면 0..i−1은 반영, i부터는 그대로이고 `n_done`과 `err`로 알린다.
  **호출 자체는 길이가 안 맞을 때만 실패한다** — FastRPC는 실패한 호출의 출력을 호스트로 돌려보내지 않을 수 있어, 부분 실패도
  성공으로 돌아와야 호스트가 어디까지 됐는지 안다.
- **호스트.** `registerStagedBatch`가 한 배치를 RPC 한 번으로 등록하고, 반영된 것은 파일링, 나머지는 슬롯을 빈 목록으로(DSP가 옛
  쌍을 그대로 들고 있으므로 다음 swap이 해제) 돌린다.
  - prefill 선읽기: `prefetch_qs4cx_wh_experts_end`가 층 배치(약 24개)를 한 번에 → 층당 RPC 24회 → 1회.
  - 동기 미스(decode, 선읽기 없는 prefill): 새 `ComputeOps::register_qs4cx_wh_expert_files`. 레이어의 `acquire`가 미스를 모두
    비운 뒤 이름을 넘기므로, 미스를 모아 읽고 한 번에 등록한다. 기본 구현은 하나씩(다른 백엔드용).
- `swap_host_check`에 배치 검사 추가: 첫 expert는 반영·두 번째(아레나 밖)는 거부되어 `n_done=1`, 호출은 성공, 길이 불일치는 통째
  거부. 실제 skel 함수와 실제 레지스트리, IDL에서 만든 헤더로 확인한다.
- **skel과 stub을 다시 만들어야 한다.** 옛 skel에서는 이 호출이 에러로 돌아오고 레이어가 멈춘다(메시지가 skel 재빌드를 안내).

예상(C=8, warm): prefill 등록 55 ms → 층당 RPC 1회 + expert당 DSP 등록 비용. expert당 DSP 비용을 모르므로 절감 폭은 기기에서
잰다. decode는 콜당 RPC 1.71 → 1회 이하(미스 있는 콜만)라 토큰당 1~2 ms(+2~3%) 예상.

### 10.24 배치 swap 실측: 같은 세션 상주 627 ms 대비 C=8 662 ms(+5.6%), 리더 2개가 최적 (2026-09-28)

`opt_all.sh`, 같은 세션, 30 s 간격. 속도는 계측 없는 실행, 나머지는 프로파일 실행. ppl은 전부 62.0916.

| 구성 | prefill | decode | 준비율 / 대기 | 등록 | decode swap 합 | 층 콜 dsp / transport (us) | peak RSS |
|---|---|---|---|---|---|---|---|
| 상주 | **627 ms** | 24.04 | — | — | — | 13946 / 550 | 4.7~5.2 GB |
| C=8 k=0 | (1387*) | 16.58 | 동기 미스 503 | swap 36.7 ms | 1661 ms | 14137 / 794 | 0.89 GB |
| C=8 k=1, 리더 4 | 667 ms | 16.76 | 100% / 0 | 38.2 ms | 1683 ms | 14888 / **1865** | 0.89 GB |
| **C=8 k=1, 리더 2** | **662 ms** | 16.51 | 100% / 0.3 ms | 37.7 ms | 1684 ms | **14583 / 691** | 0.89 GB |
| C=8 k=1, 리더 1 | 777 ms | 16.69 | **77% / 101 ms** | 38.3 ms | 1665 ms | 14367 / 618 | 0.89 GB |

\* t_k0는 상주 실행 바로 다음이라 page cache가 밀려난 부분 cold다(§10.13, §10.15와 같은 패턴). warm 값은 §10.22의 866 ms.

- **같은 세션 상주 대비 +35 ms(+5.6%).** 이번 세션 상주는 627 ms(579는 다른 날의 최선값). 메모리는 약 5.2 → 1.83 GB.
- **배치 swap은 등록을 55 → 38 ms(−31%)만 줄였다.** decode swap도 1950 → 1670 ms(−14%). expert당 약 0.105 → 0.072 ms로, 줄어든
  것은 RPC 왕복분 약 0.03 ms뿐이다. **남은 0.07 ms는 DSP 쪽 등록 자체다**: 한 swap이 레지스트리에서 weight 2개 각각에 malloc 3회 +
  memcpy(스케일·colsum·bias), 옛 쌍 각각 free 3회, 빈 슬롯 선형 탐색, 0 bias용 calloc 1회를 한다(`hexkl_weight_u8i4_fill_slot`,
  `nntr_hvx_weight_swap_u8i4_arena`).
- **리더 2개가 최적이다.** 4개와 prefill이 같고(662 vs 667) 층 콜 부작용이 작다(DSP +0.64 ms, transport +0.14 ms vs 4개의 +0.94 /
  +1.3 ms). 1개는 읽기가 못 따라간다(준비율 77%, 대기 101 ms → 777 ms).
- decode는 구성과 무관하게 16.5~16.8 TPS(편차 안). decode swap 합이 약 290 ms(토큰당 0.57 ms) 줄었지만 편차에 묻힌다.

**다음 지렛대: DSP 쪽 제자리 재바인딩.** 옛 쌍과 새 expert는 모양(K, N)이 같으므로, 옛 핸들을 해제하고 새로 등록하는 대신 옛 핸들의
아레나 포인터와 스케일·colsum만 덮어쓰면 malloc/free 12회·calloc·슬롯 탐색이 없어지고 memcpy(약 45 KB)만 남는다. 예상 등록
38 → 약 10 ms, decode swap 토큰당 약 −2 ms(+3%). skel만 바뀌고 IDL은 그대로다.

### 10.25 DSP 제자리 재바인딩, decode 선읽기 상한 지표 (코드, 호스트 체크 통과, 기기 미측정)

§10.24에서 배치 swap 뒤에도 expert당 0.07 ms가 DSP 쪽 등록 자체(weight 2개 × malloc 3 + memcpy, 옛 쌍 free 6, 슬롯 탐색, 0 bias
calloc)였다.

- **`hexkl_weight_u8i4_rebind_arena`** (레지스트리): 살아 있는 아레나 차용 슬롯을 같은 모양의 새 바이트로 제자리에서 바꾼다 — 핸들 번호와
  배열은 그대로, 아레나 포인터 교체, 스케일·colsum memcpy, bias 0. 할당·해제·탐색이 없다.
- **`weight_swap_u8i4_arena`**: 옛 쌍이 새 expert와 모양이 같으면(expert 풀에서는 항상) 두 핸들을 재바인딩하고 **같은 핸들 번호를 돌려준다.**
  아레나 범위·정렬을 두 weight 모두 먼저 검사해서, 거부되면 옛 쌍은 그대로다(all-or-nothing 유지). 모양이 다르거나 옛 쌍이 없으면 기존의
  등록 + 해제 경로. IDL은 그대로이고 skel만 바뀐다. 배치 swap도 이 경로를 탄다.
  - 호스트는 핸들 번호로 역참조하지 않는다(`handle_cache_`는 키 → 핸들이고, 해제 때 옛 키가 지워진다). 핸들 번호로 캐시하는 커널도 없다
    (레지스트리 슬롯에 캐시 필드가 없음을 확인).
  - 아레나 범위 검사는 `arena_weight_at`으로 빼서 등록과 재바인딩이 같은 검사를 쓴다.
- `swap_host_check`: 재바인딩이 같은 번호·같은 배열·새 스케일·0 bias를 돌려주는지, down이 범위 밖이거나 정렬이 틀리면 **gate_up도 안 바뀌는지**,
  모양이 다른 옛 쌍은 등록 + 해제로 가는지. 사전 정렬 검사를 뺀 변형은 이 검사에서 실패한다(확인함).
- **`tools/moe_expert_cache_sim.py`**: decode 선읽기의 상한 지표 추가 — 같은 층의 직전 토큰 top-(4+m)이 이번 토큰의 expert를 얼마나
  덮는지(m=0..5). 기본 비용도 §10.24 값으로 갱신(base 41.6 ms, 미스 0.47 ms).

예상: 등록 38 → 약 10 ms(prefill −25 ms, 같은 세션 상주 대비 +10 ms 안), decode swap 토큰당 약 −1.5 ms.

측정(`opt2.sh`): 같은 세션 상주·C=8 k=1 리더 2(warm), 라우팅 trace 1회(시뮬레이터용), CPU 모델을 먼저 돌려 page cache를 밀어낸 cold
첫 prefill(k=0, k=1 r2, k=4 r2, k=4 r4).

### 10.26 재바인딩·cold·시뮬레이터 실측: warm prefill은 상주와 같아졌고, 남은 차이는 decode다 (2026-09-28)

`opt2.sh`, 같은 세션. ppl 62.0916(측정한 것 전부).

**warm.** 계측 없는 t_k1r2(1173 ms)는 상주 실행 바로 다음이라 page cache가 밀려난 부분 cold다(§10.24의 t_k0와 같은 패턴 — 상주 실행이
3.75 GB 아레나를 잡으며 모델 파일 페이지를 밀어낸다). 같은 조건으로 비교되는 프로파일 실행끼리는 **상주 1697 ms vs C=8 k1r2 1699 ms로 같다.**

| | 상주 | C=8 k1r2 (§10.24 → 이번) |
|---|---|---|
| 층 콜 dsp / transport (us) | 13822 / 692 | 14583 / 691 → **14333 / 588** |
| 선읽기 준비율 / 대기 | — | 100% / 0.3 → 98% / 9.4 ms |
| 등록 (prefill) | — | 37.7 → **30.9 ms** (expert당 0.071 → 0.058 ms) |
| decode swap 합 (512토큰) | — | 1684 → **1342 ms** (토큰당 3.3 → 2.6 ms) |
| decode | 24.05 TPS | 16.5 → 16.8 TPS |

- **제자리 재바인딩은 등록을 18%만 줄였다.** 할당·해제가 주 비용이라는 §10.24의 추정은 틀렸다. 남은 expert당 0.058 ms는 RPC 인자
  (expert당 스케일·colsum 약 45 KB, 층 배치당 약 1 MB)를 FastRPC가 DSP로 옮기고 DSP가 레지스트리로 memcpy하는 데 드는 것으로 본다.
  다음 수는 **스케일·colsum도 아레나에 읽어 두고 레지스트리가 빌려 쓰는 것**이다(파일 레이아웃이 이미 `[WH][N scales][N colsums]`라
  한 번의 pread로 같이 들어온다). 그러면 swap 인자는 오프셋뿐이다.

**cold 첫 prefill.** CPU 모델을 먼저 돌려 page cache를 밀어낸 뒤(`Cached` 약 2.9 GB로 떨어짐) 매번 한 번씩.

| k / 리더 | cold prefill | decode |
|---|---|---|
| 0 | 2318 ms | 17.4 |
| 1 / 2 | 1329 ms | 17.6 |
| 4 / 2 | 1097 ms (준비율 49%, 대기 439 ms) | 17.5 |
| **4 / 4** | **935 ms** | 17.6 |

- **cold 첫 prefill −60%(2318 → 935 ms).** warm에서는 k와 리더 수가 무관했지만(§10.22·10.24), cold에서는 flash 읽기(미스당 약 4 ms)를
  더 많은 콜과 더 많은 리더에 나눠야 해서 k=4·리더 4가 가장 좋다. k=4 리더 2도 준비율 49%로 flash가 못 따라간다. 리더 8은 미측정.
- warm에서 리더 4는 리더 2 대비 콜당 DSP +0.3 ms 정도(§10.24)라, **기본값으로는 k=4·리더 4**가 cold와 warm 모두에 맞다.

**시뮬레이터(이번 trace, 11286 콜).** C=8 `ours` 57.2%가 기기 실측 57.2%와 정확히 같다 — 시뮬레이터가 기기를 재현한다.

| C | ours | lfu | belady(상한) | ours TPS(산술) | belady TPS |
|---|---|---|---|---|---|
| 4 | 41.0% | 38.5% | 59.6% | 15.1 | 17.2 |
| 8 | 57.2% | 59.1% | **78.5%** | 16.9 | 19.8 |
| 12 | 72.7% | 74.4% | 88.5% | 18.9 | 21.6 |
| 16 | 84.6% | 85.4% | 94.1% | 20.8 | 22.7 |

- **정책의 여지는 있다(C=8에서 57 → 최대 78.5%).** belady는 미래를 아는 상한이라 다 가져올 수는 없지만, LFU가 이미 +1.9%p다. 빈도와
  최근성을 섞는 정책(LRFU/ARC류)을 시뮬레이터에서 먼저 재 본다 — 기기 없이 이 trace로 된다.
- **decode 투기적 선읽기는 접는다.** 직전 토큰의 top-9가 이번 토큰의 expert를 64%밖에 못 덮고, 그중 상당수는 이미 캐시 히트다(57%).
  후보 9개를 읽는 DDR 비용에 비해 남는 것이 작다.
- C를 키우는 것은 확실한 선택지다: C=12(+0.46 GB, 약 2.3 GB)면 73% · 산술 18.9 TPS, C=16(약 2.8 GB)이면 85% · 20.8 TPS.

### 10.27 정책 후보를 시뮬레이터에, 선읽기를 기본값으로 (코드, 기기 미측정)

- **`tools/moe_expert_cache_sim.py`에 정책 두 종류 추가.**
  - `lrfu:H` — 사용할 때마다 점수 +1, 점수는 H 토큰마다 절반. 가장 낮은 것을 내보낸다. H가 작으면 LRU, 크면 LFU에 가깝다.
  - `lrfu+:H` — lrfu에, decode 토큰이 5~9위로 매긴 expert마다 +0.5. 라우터가 "다음에 올 만한 것"이라고 본 것을 조금 더 붙잡는다.
  - 기본 실행이 H = 2, 8, 32, 128 토큰을 모두 돌린다. §10.26의 trace(11286 콜)로 PC에서 몇 초. selftest에 두 정책의 동작 검사를 넣었다.
  - 판정: C=8에서 `ours` 57.2%보다 3%p 이상 높은 정책이 있으면 그것을 `ExpertLru`에 넣고, 없으면 정책 작업은 접는다(C를 키우는 쪽으로).
- **선읽기가 기본으로 켜진다.** `NNTR_MOE_PREFETCH`가 없으면 k=4, `=0`이면 끔. 리더 기본 4개는 그대로. warm에서 k는 차이가 없었고
  cold 첫 prefill은 k=4·리더 4에서 2318 → 935 ms였다(§10.26). 칸이 모자라는 C(≤3)에서는 큐잉이 알아서 멈추고, C=1의 분할 경로는 선읽기를
  쓰지 않는다.
- 측정(`defaults.sh`): 환경변수 없는 기본값 vs `NNTR_MOE_PREFETCH=0`을 warm·cold로 비교하고, **상주 실행은 맨 뒤에** 둔다(상주가 모델 파일
  페이지를 밀어내 다음 실행을 cold로 만드는 §10.24·10.26의 오염을 피한다).

### 10.28 기본값 실측: warm 803 → 639 ms, cold 2438 → 928 ms. 리더 4개의 warm 비용이 보인다 (2026-09-28)

`defaults.sh`, 같은 세션, 상주는 맨 뒤(직전에 C=8 예열). ppl 전부 62.0916.

| 구성 | prefill | decode | 층 콜 host / dsp / transport (us) | 등록 | decode swap 합 |
|---|---|---|---|---|---|
| **기본값** (k=4, 리더 4) | **639 ms** | 17.95 | 15845 / 14659 / 1186 | 33.7 ms | 1347 ms |
| 선읽기 끔 | 803 ms | 17.76 | 14243 / 13730 / 513 | (swap 30.7) | 1355 ms |
| cold, 기본값 | **928 ms** | 17.76 | | | |
| cold, 선읽기 끔 | 2438 ms | 17.83 | | | |
| 상주 | 577 ms | 24.61 | 14102 / 13645 / 457 | | |

- 기본값으로 warm −20%, cold −62%. 준비율 100%, 대기 0.
- **상주 대비 +62 ms(+11%)로, §10.26의 k=1·리더 2(+35 ms)보다 크다.** 차이는 층 콜이다: 리더 4개는 DSP +1.0 ms, transport +0.7 ms
  (리더 2개는 +0.5 / −0.1 ms, §10.26). ×23 ≈ 40 ms + 등록 34 ms. warm에서는 리더 2개가 낫고 cold에서는 4개가 낫다(§10.26: 1097 vs 935).
- decode는 선읽기와 무관(17.8~18.0). 이번 세션 미스 읽기가 0.32 ms로 빨라 decode가 조금 높다. 상주 대비 −27%는 여전히 미스 수(히트 57%) 몫이다.

**다음.**
1. **읽기 속도에 맞춰 리더 수·깊이를 바꾸기.** 읽기가 page cache에서 오면(expert당 1 ms 미만) 리더 2개·깊이 1, flash에서 오면 리더 4개·깊이 4.
   백엔드가 최근 읽기 시간을 알고 있으니 그것으로 정한다. 예상: warm은 k1r2 수준(+35 ms), cold는 지금 수준(약 930 ms) 유지.
2. **스케일·colsum을 아레나에서 빌려 쓰기**(§10.26): 등록 34 ms와 decode swap 토큰당 2.6 ms의 대부분.
3. 정책: 시뮬레이터 결과 대기(§10.27).

### 10.29 decode 차이 분해: 전부 미스 복사 비용이고, C=8에서는 어떤 정책으로도 상주 decode에 못 간다 (2026-09-28)

§10.28의 decode 숫자와 시뮬레이터(`--cache 8 12`, 같은 trace)를 맞춰 본 결과.

| | ms/토큰 |
|---|---|
| 상주 (24.61 TPS) | 40.6 |
| 기본값 C=8 (17.95 TPS) | 55.7 |
| 차이 | **15.1** |
| 미스 수 1.71/콜 × 22층 | 37.6개/토큰 |
| swap (1347 ms / 512 토큰) | 2.6 |
| 나머지 12.5 ms / 37.6 | **미스당 읽기 0.33 ms** (5.25 MiB, page cache → ION, 약 16 GB/s) |

- 차이 15.1 ms를 미스 37.6개 × (읽기 0.33 + swap 0.07 ms)가 전부 설명한다. 다른 원인이 남지 않는다.
- 미스당 비용은 지연이 아니라 **DDR 대역폭**이다. 상주는 토큰당 가중치를 약 462 MiB(4 × 22 × 5.25) 읽는다. C=8은 여기에 197 MiB를 더
  복사한다(읽기 + 쓰기면 약 395 MiB). 그래서 CPU 코어를 더 쓰거나 스레드를 더 붙여도 줄지 않고, **복사하는 바이트를 줄여야만** 줄어든다.
- 시뮬레이터의 TPS 산술을 보정했다: `--base-ms 40.6 --miss-ms 0.40`(이전 41.6 / 0.47은 16.9를 예측했고 실측은 17.95). 보정 뒤 ours는 17.96으로 실측과 같다.

| C=8 정책 | 히트 % | TPS (보정 산술) |
|---|---|---|
| ours (지금) | 57.2 | 17.96 |
| lfu | 59.1 | 18.2 |
| lrfu:32 (가장 좋은 실현 가능 정책) | 59.6 | 18.2 |
| belady (미래를 아는 상한) | 78.5 | 20.7 |
| C=12 ours / lrfu:128 / belady | 72.7 / 74.7 / 88.5 | 19.9 / 20.2 / 22.4 |

- 정책을 바꾸면 +2.4%p, +0.26 TPS다. 채택 기준 3%p에 못 미치므로 **ours를 유지**한다.
- 미래를 아는 belady조차 20.7이다. **C=8에서 decode 24.6은 캐시 정책으로 달성할 수 없다.** 상주 decode와 같아지려면 미스가 0이어야 하고, 그 말은 C=64(상주)다.
- decode 선읽기(추측)도 상한이 낮다. 직전 토큰 top-9가 64%를 덮지만 9개 중 약 5개는 헛읽기가 되어, 줄이려던 DDR 복사를 오히려 늘린다.

**남은 레버(바이트를 줄이는 것만):**
1. C를 키운다. C=12는 약 +0.5 GB 아레나에 19.9 TPS, C=16은 시뮬레이터로 따로 확인한다.
2. 히트 expert 계산과 미스 읽기를 겹친다(층 콜을 히트분 → 미스분으로 나누기). 둘 다 DDR을 쓰므로 겹치는 이득은 제한적이다.
   decode 층 콜 DSP 시간과 미스 읽기 시간을 먼저 재고 판단한다.
3. swap 0.07 ms/미스(토큰당 2.6 ms)는 §10.28 다음 2번(스케일·colsum 빌려 쓰기)으로 줄일 수 있다. 최대 +1.1 TPS.

### 10.30 등록을 offset만으로, 리더는 "다음 배치"에만 넓게 (코드, 호스트 체크 통과, 기기 미측정)

§10.28의 warm +62 ms 두 몫을 각각 없앤다. 등록 34 ms(expert마다 스케일·colsum 44 KB를 RPC 인자로 넘기고 DSP가 복사)와,
리더 4개가 아레나에 쓰는 동안 층 콜이 느려지는 약 40 ms(DSP +1.0, transport +0.7 ms/콜; 리더 2개면 +0.4).

**1. 스케일·colsum을 아레나에 두고 DSP가 가져간다.**
- 호스트 `readWeight`: 파일의 꼬리(N f32 스케일, N f32 colsum)를 읽어 colsum만 i32로 바꾼 뒤 **WH 바이트 바로 뒤 아레나 칸에** 쓴다.
  칸 stride는 `round4K(whBytes + 8N)`, expert당 +44 KB(+0.8%). `ArenaEntry`의 벡터는 비워 두고, 그게 swap 호출에 "인자가 아레나에 있다"는 신호다.
- IDL은 그대로다. `weight_swap_u8i4_arena`·`_batch`에 **네 sequence가 모두 비면** 꼬리를 아레나에서 읽는다(옛 skel은 길이 오류로 거부 →
  호스트 오류 메시지가 skel 재빌드를 말한다). 배열을 넘기는 옛 형식도 그대로 받는다(테스트, 상주 경로의 `register_u8i4_arena`).
- 레지스트리 `hexkl_weight_u8i4_register_arena`/`rebind_arena`에 NULL 배열 = 꼬리 모드. 꼬리는 **memcpy가 아니라 DMA**로 힙 배열에 옮긴다:
  아레나는 호스트 uncached DDR이라 스칼라 코어가 1 GB/s 아래로 읽고(문서 46 §10), DMA는 대역폭대로 읽는다. 앞뒤로 `dcinva`(원본: 이전 swap이
  L2에 남긴 줄, 목적지: epilogue가 L1에 남긴 줄). 세 배열은 128 정렬 블록 하나로 합쳤다(`arrays`). epilogue는 전과 같이 힙 배열을 읽는다 —
  아레나를 직접 가리키면 타일마다 uncached 로드가 나서 dequant가 망가진다.
- `arena_weight_at`의 범위 검사가 꼬리 8N을 포함한다.
- 기대: 배치 RPC 인자 1 MB → 500 B. 층당 등록 1.5 → ~0.3 ms(DMA 48회 + 왕복), prefill 34 → **~7 ms**. decode swap은 왕복이 대부분이라 2.6 → ~2 ms/토큰.
- 호스트 체크 `swap_host_check` 7번: 제자리 재바인딩·등록 경로·꼬리 범위 초과 거부·반만 빈 길이 거부·배치. 범위 검사를 빼면, 또는
  재바인딩의 꼬리 복사를 빼면 실패한다(변이 2개 확인). 기기 테스트 `HmxArenaSlotReuse.TailSwap` 추가(같은 칸을 두 번 지나는 stale 캐시 검사).
  **`Swap`·`TailSwap` 둘 다 기기에서 아직 안 돌렸다.** `Swap`은 힙 참조에 bias가 있어 늘 실패했을 것이라 참조를 0 bias로 다시 등록하게 고쳤다.

**2. 리더 폭을 배치 순서로 정한다(임계값 없음).**
- `NNTR_MOE_PREFETCH_AHEAD_READERS`(기본 2): 층이 **다음에 물어볼 배치**(가장 오래된 것)의 expert는 리더 4개가 다 붙고, 그보다 앞선 배치는
  2개까지만 동시에 읽는다. warm이면 다음 배치가 늘 먼저 끝나 있어 리더 2개로 돌고(k1r2의 +0.4 ms/콜), cold면 다음 배치가 늦어 4개가 붙는다(§10.26의 935 ms).
  읽기 속도 임계값으로 warm/cold를 가르는 안은 버렸다: 리더 수에 따라 expert당 시간이 달라져(UFS 4.0에서 2개면 2.6 ms, page cache에서 4개면 ~1.3 ms) 경계가 얇다.
- `=4`가 §10.28의 동작이다(A/B용). 프로파일에 `expert prefetch readers: X ms/expert on a reader, N for the next batch (full width), M further ahead` 행 추가.
- 기대: warm prefill 639 → **약 590 ms**(상주 577 + 등록 ~7 + 콜 +0.4 × 23). cold는 930 유지.

**측정(`tail.sh`).** 순서: 예열 → default(t, p) → ahead4(t, p) → off → cold default → cold ahead4 → 예열 → 상주(t, p). 볼 것: ppl 62.0916(꼬리 프로토콜의
정확성 — 틀리면 dcinva가 L2에 안 닿은 것), prefill, `register rpc` ms, `readers` 행, 층 콜 host/dsp/transport. 그 전에 `HmxArenaSlotReuse.TailSwap`.

**빌드.** skel 재빌드 필수(레지스트리·swap 변경). IDL 서명은 그대로라 stub 재생성은 불필요하지만, 브랜치를 오갔다면 §10.24의 규칙대로 둘 다 다시 만든다.

### 10.31 §10.30 실측: 등록 34 → 7 ms로 warm prefill이 상주 +9 ms. 리더 폭 제한은 기각 (2026-09-29)

기기 테스트 `HmxArenaSlotReuse.{Uncached,Swap,TailSwap}` 전부 통과, 두 pass 모두 `bad_elems=0`. 꼬리를 아레나에서 가져오는 경로가 같은 칸을
두 번째로 지날 때도 stale이 없다. `Swap`의 첫 실패는 테스트 버그였다(7236f06): §10.25 재바인딩 뒤로 옛 handle이 곧 새 handle인데 테스트가 그것을
release해 pass 1 콜이 해제된 handle로 호출됐다. 그 전까지 `Swap`은 한 번도 통과한 적이 없었다.

`tail.sh`, 같은 세션, ppl은 모든 실행에서 62.0916.

| 구성 | prefill (t) | decode (t) | 층 콜 host / dsp / transport (us, p) | 등록 (p) | 리더 ms/expert |
|---|---|---|---|---|---|
| 앞 배치 리더 2 (§10.30 기본) | 630 ms | 17.88 | 14628 / 14180 / 448 | 6.9 ms | 1.42 |
| **리더 4 (`AHEAD=4`)** | **606 ms** | 17.81 | 15589 / 14525 / 1064 | 7.0 ms | 2.11 |
| 선읽기 끔 | 1069 ms | 17.70 | | | |
| cold, 앞 배치 리더 2 | 1097 ms | 17.94 | | | |
| cold, 리더 4 | 988 ms | 18.05 | | | |
| 상주 | 597 ms | 24.05 | 14354 / 13842 / 512 | | |

- **등록: 33.7 → 6.9 ms(−80%).** 예측(~7 ms)대로다. decode swap도 1347 → 697 ms, 토큰당 2.6 → 1.4 ms.
- **warm prefill은 리더 4개로 606 ms, 상주 대비 +9 ms(+1.5%).** §10.28의 +62 ms에서, 이 기기의 실행 간 편차(±20 ms) 안으로 들어왔다.
- **리더 폭 제한은 기각했다.** 콜 하나만 보면 리더 2개가 덜 느리게 한다(+274 vs +1235 us/콜). 그런데 end-to-end로는 warm 630 vs 606, cold 1097 vs 988로
  둘 다 졌다. 깊이 4에서는 배치가 "다음"이 되기 전에 이미 다 읽혀 있다(528개 중 16개만 다음 배치로 읽힘). 그래서 이 장치는 사실상 "항상 리더 2개"였다.
  2개로 읽으면 읽기가 더 오래 이어져 층 콜 밖의 구간(attention·conv 콜)까지 겹치는 것으로 보인다(재지는 않았다). 코드와 knob을 지웠고, 리더 읽기 시간 행만 남겼다.
- decode 차이는 §10.29 그대로 미스 수다: 1000/17.81 − 1000/24.05 = 14.6 ms/토큰 = 읽기 13.1(0.35 ms × 37.6) + swap 1.4.

### 10.32 cold prefill은 flash 상한이다: 이 기기의 UFS는 3.0 GB/s, 2.8 GB ÷ 3.0 = 0.93 s (2026-09-29)

"page cache를 안 늘리고 warm처럼"은 정의상 없다. 바이트가 있을 DRAM은 page cache 아니면 아레나뿐이고, 아레나를 키우는 것이 C다. 그래서 물음은
"cold를 얼마나 낮출 수 있나"이고, 답은 flash 대역폭이 정한다. `dd`로 쟀다(evict 뒤, 같은 구간을 두 번 읽어 cold/warm 구분; 파일 4.3 GB 안의 offset만).

| | 시간 | 대역폭 |
|---|---|---|
| 8스트림 × 256 MB, cold | 0.69 s | **3.0 GB/s** |
| 같은 8스트림, warm | 0.19 s | 11 GB/s |
| 4스트림 × 512 MB, cold | 0.69 s | **3.0 GB/s** |

- 스트림 수와 무관하게 3.0 GB/s. **리더를 늘려도 얻을 것이 없다.**
- C=8 prefill이 flash에서 읽는 양 528 × 5.25 MiB = 2.8 GB → 0.93 s. **실측 cold prefill 0.93 s와 같다.** 선읽기는 읽기를 다 숨기고 있고, 남은 시간은
  전부 flash다. 앱에서 더 줄일 것이 없다.
- 기각한 것들: `O_DIRECT`로 아레나에 직접 DMA(dma-buf 매핑은 `remap_pfn_range`라 페이지 고정이 안 될 가능성이 크고, 되더라도 상한은 같은 3.0), 압축(int4 엔트로피
  ~3.6비트, 10%를 CPU 디코딩이 먹는다), prefill expert 건너뛰기(ppl), 더 일찍 읽기(둘 곳이 없다).
- 남는 것은 바이트, 즉 C다. C=12: 2.3 GB → ~0.78 s. **C=16: 1.85 GB → 0.62 s, 계산 0.6 s와 같아져 cold ≈ warm(~0.65 s)**, 물리 2.75 GB, decode 히트 85%.

**C=8의 제품 수치는 cold 행이 정직하다:** 물리 1.9 GB, prefill 0.93 s, decode 17.9 TPS. warm 0.6 s는 OS가 남는 메모리에 파일 3.7 GB를 붙잡아 둔 경우의 값이고,
메모리를 아끼는 상황에서는 그 전제가 먼저 깨진다.

### 10.33 C=16 첫 실측과 선읽기 시작 지점 (코드, 기기 미측정)

`NNTR_MOE_CACHE_EXPERTS=16` 두 번 실행(기본 프롬프트 444 토큰): **cold prefill 799 ms, warm 592 ms, decode 21.7 TPS**(시뮬레이터 예측 20.8, 히트 85%).
아레나 1.85 GB, 물리 약 2.75 GB.

cold 바닥은 §10.32 산술로 read 1.85 GB ÷ 3.0 = 0.62 s + 층 한두 개 = **약 0.67~0.70 s**인데 0.80이 나왔다. 원인은 선읽기가 늦게 시작해서다.
- 로드 때 풀을 층 순서로 채우니 C=16은 층 0~10이 상주다. 읽을 것이 있는 첫 층은 11이다.
- 선읽기 지평이 "현재 층 + 4"라서 층 11의 배치는 **층 7에서야** 큐에 들어간다. 그때부터 11개 배치 × 56 ms = 616 ms를 flash가 내야 하니 189 + 616 + 27 ≈ 830 ms.
  실측 799와 맞는다.
- 층 1부터 시작할 수 있었다. 층 0이 끝나면 그 32칸이 비고, 그 칸에 층 11의 배치를 읽으면 된다. 막은 것은 지평이었지 칸이 아니었다.

**고침.** 지평 기본값을 "남은 층 전부"로(`NNTR_MOE_PREFETCH` 미설정), 핀은 **뒤에 올 모든 층의 상주 expert**로(전에는 지평 안의 층만). 그래서 makeRoom이
비울 수 있는 것은 이미 끝난 층의 expert뿐이고, 자리가 없으면 큐가 멈췄다가 다음 층에서 이어진다 — 풀 크기가 유일한 한계가 된다.
- C=8: 풀 176에 현재 층 32를 빼면 배치 4.5개라 전과 같은 동작(깊이 4와 동일)이다. 바뀌는 것이 없어야 하고, 실측으로 확인한다.
- C=16 기대: cold 0.80 → **약 0.68 s**. warm은 그대로.
- 측정: `NNTR_MOE_CACHE_EXPERTS=16`으로 cold(evict 뒤) 1회 + warm 1회, `=8`로 warm 1회(회귀 확인).

**실측 (2026-09-29, 380edd5).** C=16 cold **799 → 718 ms**(−81; 바닥 산술 ~680), warm 602, decode 21.6~21.7. C=8 warm 606(같음), **C=8 cold 1021**
(직전 값 929~988; 산술로는 동작이 같아야 하고 cold 실행의 편차가 ±50 ms라 판정 보류 — 한 번 더 재서 950 안쪽이면 잡음, 1000 넘으면 되돌린다).
