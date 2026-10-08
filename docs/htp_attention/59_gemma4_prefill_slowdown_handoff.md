# 59. Gemma-4 전부-NPU prefill — PR 4343 a16 이후 느려진 원인과 다음 최적화 (인수인계, 2026-10-08)

다음 세션이 **"PR 4343 최신 커밋 이후 prefill이 느려진 원인 분석 + 더 줄일 수 있는 곳 찾기"**를
이 문서 하나로 시작할 수 있게 쓴 인수인계서다. 그 전 배경(모델, 구성, 이전 측정 이력)은
`58_gemma4_prefill_handoff.md`, 상세 이력은 `57_gemma4_all_npu_task.md` §9.19–§9.22.
측정 원본은 `59_data/`(아래 §6).

- 브랜치: `claude/zealous-bell-a2pot9` (Seunghui98/nntrainer). 이 문서 시점 코드 HEAD `6a31ba72`.
- 기기: Galaxy S25 Ultra `R3CY10WM83Y`, Hexagon V79. 사용자 workstation에 SDK·NDK·adb가 있다
  (`HEXAGON_SDK_ROOT=$HOME/workspace/Hexagon_SDK/6.4.0.2`). 세션이 그 머신에서 돌면 skel 빌드와 기기
  실행을 직접 할 수 있다(클라우드 세션이면 못 한다 — 사용자에게 명령 블록을 준다).
- 사용자와는 한국어. 기기에서 안 잰 숫자는 "기기 미측정". 가설보다 측정 분해가 먼저.

---

## 0. 한 줄 요약

정확도는 해결됐다(P0: DSP router의 qf32 누산, 7eeea62e, nll 12.33 → 3.5). PR 4343 최신 3커밋(a16/kv8
attention)을 넣으니 nll은 3.536 → **3.508**로 좋아졌지만 cfgB prefill이 **4.43–4.53 s → 4.79 s(+0.3 s)**.
늘어난 몫은 DSP가 아니라 **PR이 host(mha_core)에 넣은 스칼라 루프 두 개**다(측정으로 분해, §2).
남은 몫: flash 바닥 3.4 s까지 ≈ 1.4 s.

---

## 1. 지금 코드 상태 (최신 → 과거)

| 커밋 | 내용 |
|---|---|
| 6a31ba72 | a3b066ab의 `submit_job`을 이 브랜치의 void `hvx_worker_pool_submit`에 맞춤(파일이 PR과 이 함수만 다름) |
| 4b08b64b | PR 5a213bfc: u16 Q·출력 + per-head (scale, zp) 인코딩, IDL 변경(`attn_q2_step`/`attn_q2_prefill`). 충돌 병합: Q/out ION 스테이징(0583c9eb)은 u16으로 유지, Q 보정은 PR의 min/max 채택 |
| 48f04ddb | PR a3b066ab: a16/kv8 row-blocked attention(16-bit Q·P를 u8×i8 array에) |
| f145d8b6 | PR 9005bc9f: 정규화 16-bit P 정수 softmax, 스레드 분할 |
| 7eeea62e | **P0 수정**: `hvx_router_rows_f32` 누산을 IEEE sf로(qf32는 V79에서 4행 중 3행 1e37/NaN, 호스트 에뮬은 통과) |
| d5baf8c2 | CPU-engine `dense_ffn`의 QS4CX pack(정확도 사다리용) |
| b62c3ec9 | `NNTR_ACT_STATS=1`: 첫 prefill에서 노드별 출력 nan/inf/maxabs/rms (libnntrainer, `neuralnet.cpp`) |

설정 파일(사용자 PC `~/workspace/nntrainer/`, git 밖): `cfgA.json`(엔진 5개 htp, fp16 attention, seq 2048),
`cfgB.json`(= cfgA + `attention_kv_dtype: q8` + seq 1088, **속도 기준**), `cfgB_ppl.json`(= cfgB +
`skip_prefill: false` + 생성 16, **정확도 기준**), `acc_0..4.json`(엔진 사다리, fp16 attention).

---

## 2. 측정 (모두 기기 실측, 1024토큰 Ardley prompt)

### 2.1 PR 전후 (cfgB, `NNTR_HTP_PROFILE=1 NNTR_HTP_ATTN_TRACE=1`)

| | e6a2022b (PR 전) | 6a31ba72 (PR 최신) |
|---|---|---|
| prefill | 4,486 (사용자) / 4,428 / 4,531 ms | **4,796 / 4,791 ms** |
| nll (`cfgB_ppl`, q8) | — (fp16 attention 3.536) | **3.508** |
| staging memcpy | 541 ms (7.77 GB) | 470 ms (7.20 GB) — Q·출력이 u16이라 줄었다 |
| attention `accel_call` 30층 합 | ≈ 540 ms (§9.17: 17–18 ms/층) | **1,080 ms** |
| MoE `K=2816 N=2816 M>1` | 1,567 ms (50.5/call) | 1,540 ms (49.7/call) |
| decode | 2.87 TPS | 2.71–2.83 TPS |

### 2.2 attention 한 층 분해 (PR 최신, `59_data/attn_trace_after_pr.txt`에서 평균)

| ms/층 | sliding (25층, hd 256, KV 8) | full (5층, hd 512, KV 2) |
|---|---|---|
| `accel_call_us` (host가 본 attention 노드 몫) | **33.3** | **49.2** |
| ├ q2 보정 (`calibrate_q2_scales`, 첫 prefill만) | **11.5** | **22.1** |
| ├ FastRPC wall (`attn_q2_step`) | 14.0 | 14.3 |
| │ ├ DSP append (K/V int8 양자화 + bake) | 7.3 (quant 6.2, bake 1.1) | 3.0 (quant 2.5, bake 0.5) |
| │ ├ DSP kernel | 5.1 (qk 2.5, softmax 2.2, pv 1.8) | 8.4 (qk 3.6, softmax 2.3, pv 3.5) |
| │ └ 전송 (wall − total) | 1.6 | 2.9 |
| └ host 나머지 (Q f32→u16, 출력 u16→f32 변환 등) | **7.8** | **12.9** |

PR 전(§9.17, sliding): host 보정 0.6 · 전송 2.5 · append 6.7 · kernel 4.65 · Q/out memcpy ≈ 4 → 17–18 ms.

### 2.3 느려진 원인 (측정 + 코드)

1. **Q 범위 보정이 스칼라**: PR이 Q 보정을 abs-max → per-head min/max(u16 비대칭)로 바꿨고, 병합에서
   이 브랜치의 NEON abs-max(2e3eadd1, `layers/abs_max.h`)가 Q 쪽에서 빠졌다.
   `mha_core.cpp` `calibrate_q2_scales`(1005행~)의 `q_lo/q_hi` 루프. 1024×4096 f32(sliding) /
   1024×8192(full)를 스칼라로. 30층 합 ≈ 398 ms. K/V abs-max는 여전히 NEON.
2. **Q·출력 f32↔u16 변환이 스칼라, 단일 스레드**: `try_quantized_attention`(1056행~) 안
   `q2_q_u16` 채우는 루프(1170행 부근)와 `q2_out_u16` → `io.out` 루프(1196행 부근). 30층 합 ≈ 260 ms.
   PR 주석 그대로 "quantized graph에서는 이웃 레이어 형식이 이미 u16"이라는 전제의 임시 변환이다.
3. DSP kernel은 sliding 4.65 → 5.1 ms로 약간 늘었다(16-bit Q·P). 전송은 u16이라 줄었다.

---

## 3. 다음 최적화 후보 (기대치는 산술, 모두 기기 미측정)

| # | 항목 | 지금 | 기대 | 방법 / 파일 |
|---|---|---|---|---|
| A | Q min/max 보정 NEON + 스레드 | 398 ms | −0.37 s | `layers/abs_max.h`에 min/max 변형 추가(같은 `__aarch64__` 가드 + 스칼라 경로, compare+select로 스칼라와 비트 동일 — 2e3eadd1 방식). `ThreadManager::Global().parallel_for`로 행 분할. 검사: `test/htp/host/abs_max_check.cpp` 확장, `aarch64-linux-gnu-g++ -static` + `qemu-aarch64` |
| B | f32↔u16 변환 NEON + 스레드 | 260 ms | −0.2 s | 같은 파일들. round·clamp는 스칼라와 비트 동일하게(`+0.5f` 후 truncate, 0..65535 clamp). 더 나아가 qkv 호출의 post 단계가 u16 Q를 바로 쓰게 하면 변환 자체가 없어진다(skel 변경) |
| C | 보정을 첫 prefill 밖으로 | (A 뒤 남는 몫) | 작음 | 보정은 첫 prefill에서만 돈다. A로 충분하면 생략 |
| D | DSP append 양자화 | 6.2 ms/층(sliding) ≈ 0.17 s | −0.1~0.15 s | `hexkl_kv_q.c` `append_fixed_i8`(430행): K/V 1024행 int8 양자화를 HMX 스레드 혼자 한다 → worker pool 분산 |
| E | MoE 호출 회귀 | 49.7 ms/call (§9.13 41.5) | −0.15 s | §9.22: prefetch 경합 몫은 3 ms뿐. 나머지 5–6 ms는 DSP 단계 분해가 필요한데 `mm_u8i4_moe_layer_norm`(norm 포함 진입점, `test/htp/nntr_hvx_mm_u8i4.c` 1404행)에는 timed 변형이 없어 `NNTR_HTP_PROFILE=2`가 0을 준다. timed 변형(IDL 변경 → skel·stub 둘 다 재빌드) 또는 측정용으로 out_norm을 CPU로 돌려 기존 timed 진입점(1443행)을 타게 해서 gather/requant/mm/glu/scatter를 본다 |
| F | 노드 간 activation 복사 | staging 470 ms | −0.3~0.4 s | 58 §4 #2: 연속 NPU 노드 사이 activation을 ION에 상주 |
| G | epilogue `residual_add` 2호출/층 | ≈ 0.46 s | −0.3 s | 58 §4 #3 |
| H | router | 8.8–9.2 ms/call (264–276 ms) | −0.2 s | 58 §4 #4. **지금 kernel은 sf 누산(7eeea62e)이다. 고칠 때 qf32로 되돌리지 말 것**, 그리고 기기에서 행 단위로 CPU와 비교(§5 함정) |
| I | FC 효율 | qkv 16.5 ms/call | 불확실 | 58 §4 #6 |

A+B만으로 PR 전보다 빨라질 것으로 본다(4.79 − 0.57 ≈ 4.2 s, 기기 미측정).

---

## 4. 명령

### 4.1 빌드·푸시 (사용자 PC)

```bash
cd ~/workspace/nntrainer
export HEXAGON_SDK_ROOT=$HOME/workspace/Hexagon_SDK/6.4.0.2 HEXKL_ROOT=$HOME/workspace/hxkl-beta2/hexkl_addon ANDROID_NDK=$HOME/workspace/android-ndk-r26d
export ANDROID_SERIAL=R3CY10WM83Y
D=/data/local/tmp/nntrainer/causallm; M=$D/models/gemma4-26b-a4b-qs4cx-arm
cool() { while adb shell 'ps -A' | grep -q 'nntrainer_causall[m]'; do sleep 10; done; sleep 120; }
# skel: DSP 소스·IDL이 바뀌었을 때만. SDK env는 SDK 디렉터리 안에서 source해야 한다
(cd $HEXAGON_SDK_ROOT && source ./setup_sdk_env.source >/dev/null && cd ~/workspace/nntrainer/test/htp && ./build.sh 2>&1 | grep -E "error:|undefined symbols")
# 앱: IDL이 바뀌었으면 --cache 금지(stub 재생성). CausalLM 쪽만 바뀌었으면 --cache 가능(수 분)
(cd Applications/CausalLM && ./build_android.sh --htp) 2>&1 | tail -2
adb push test/htp/build/libnntr_hvx_skel.so $D/
adb push Applications/CausalLM/jni/libs/arm64-v8a/nntrainer_causallm $D/
adb push Applications/CausalLM/jni/libs/arm64-v8a/libcausallm_core.so $D/
adb push builddir/android_build_result/lib/arm64-v8a/libnntrainer.so $D/
adb push builddir/android_build_result/lib/arm64-v8a/libccapi-nntrainer.so $D/
```

### 4.2 정확도 게이트 (모든 변경 뒤)

```bash
adb push cfgB_ppl.json $M/nntr_config.json; cool
adb shell "cd $D && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. NNTR_NUM_THREADS=8 NNTR_PPL=1 ./nntrainer_causallm $M" 2>&1 | tee ppl.log | grep -a '\[PPL\] prompt'
sed -n '/<|turn>model/,/=====/p' ppl.log | grep -v '^\[' | head -3     # 기대: nll ≈ 3.51, "…harbour town of Ardley…"
```

### 4.3 속도 + 분해

```bash
adb push cfgB.json $M/nntr_config.json; cool; adb logcat -c
adb shell "cd $D && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. NNTR_NUM_THREADS=8 NNTR_HTP_PROFILE=1 NNTR_HTP_ATTN_TRACE=1 ./nntrainer_causallm $M" 2>&1 | tee speed.log | grep -a 'prefill:\|staging memcpy'
adb logcat -d -s nntrainer | grep -a 'mha_core trace\|attn trace q2' | grep -a 'rows=1024\|n_q=1024' > attn_trace.txt
```

노드별 by-op 표(프로파일 빌드, 마지막 by-op은 int8 이전이라 새로 잴 가치가 있다):

```bash
P=/data/local/tmp/nntrainer/causallm_prof
(cd Applications/CausalLM && ./build_android.sh --htp --profile)   # builddir 삭제, jni/libs 덮어씀
adb shell "mkdir -p $P && rm -f $P/libnntr_hvx_skel.so"           # skel은 $D 것을 쓴다
adb push Applications/CausalLM/jni/libs/arm64-v8a/nntrainer_causallm $P/
adb push Applications/CausalLM/jni/libs/arm64-v8a/libcausallm_core.so $P/
adb push builddir/android_build_result/lib/arm64-v8a/libnntrainer.so $P/
adb push builddir/android_build_result/lib/arm64-v8a/libccapi-nntrainer.so $P/
adb push cfgB.json $M/nntr_config.json; cool
adb shell "cd $P && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=$D NNTR_NUM_THREADS=8 NNTR_HTP_PROFILE=1 ./nntrainer_causallm $M" 2>&1 | tee prof.log | grep -a 'prefill:'
python3 tools/prefill_timeline.py prof.log --config cfgB.json --by-op --by-layer
# 끝나면 일반 빌드로 되돌릴 것: (cd Applications/CausalLM && ./build_android.sh --htp)
```

실행 간 노이즈 ±3–5%(FC 행 ±10%). 100 ms 단위 결론은 냉각(2분) 후 2–3회.

---

## 5. 함정 (이번 세션에서 실제로 밟은 것, 58 §7에 추가)

- **호스트 HVX lane 에뮬 통과 ≠ 기기 정상.** router kernel은 실제 모양(M=1025)에서도 에뮬을 통과했지만
  기기에서 4행 중 3행이 1e37/NaN이었다(qf32 누산, 단일 스레드에서도 재현, 누산 초기값을 바꾸면 살아남는 행이
  바뀜). HVX kernel을 새로 쓰거나 바꾸면 **기기에서 CPU 결과와 행 단위로 비교**한다. qf32 누산을 쓰면 특히.
- **CausalLM 앱은 `-ffast-math`**(`jni/Android.mk`): 앱 코드의 `std::isfinite`/`isnan`이 상수로 접힌다.
  NaN 검사는 지수 비트(`(u & 0x7f800000) == 0x7f800000`)로. libnntrainer(`NNTR_ACT_STATS`)는 해당 없음.
- DSP top-k(`hvx_router_topk_rows_f32`)는 logits 버퍼에 **제자리 softmax**를 덮어쓴다: 돌아온 `router_logits`는
  확률이다. logits를 비교하려면 top_k=0으로 부른다.
- `NNTR_MOE_DIFF`의 `max_abs_ref`는 `std::max`라 NaN을 조용히 버린다(라우팅 weight NaN을 "router 정상"으로 오판했었다).
- `test/htp/build.sh`는 `undefined symbols: ldexpf lround ceil ldexp _Log`로 exit 1을 내지만 skel은 만들어진다.
  원래 skel에도 있던 경고. `error:` 줄만 본다.
- PR 4343 코드는 `hvx_worker_pool_submit`이 int를 돌려주는 API 기준. 이 브랜치는 void. 가져올 때마다 같은 대응.
- 엔진 사다리에서 MoE는 htp 고정(CPU MoE는 기기 연결을 끊는다). `skip_prefill: true`면 `NNTR_PPL`이 안 찍힌다.

---

## 6. 측정 원본 (`59_data/`)

| 파일 | 내용 |
|---|---|
| `profile_before_pr_e6a2022b.txt` | PR 전, cfgB, 사용자 실행: `[HTP-PROFILE]` 전체(행별 host ms, staging, 등록, expert prefetch) |
| `profile_after_pr_run1.txt`, `_run2.txt` | PR 최신(6a31ba72), cfgB, 2회 |
| `attn_trace_after_pr.txt` | PR 최신 run 2의 logcat: 층마다 `mha_core trace`(보정 us, accel_call_us)와 `HTP attn trace q2`(DSP append/kernel 분해), 층 0..29 순 |
| `moe_prefetch_ab.txt` | `NNTR_MOE_PREFETCH` 1/0 번갈아 2회씩: prefill, MoE 행, prefetch 통계 |
| `accuracy.txt` | 엔진 사다리 acc_0..4(fp16 attention)와 cfgB_ppl(a16)의 prompt nll |

---

## 7. 작업 규칙

58 §9 그대로: `CLAUDE.md`, `AGENTS.md`, `01_working_style.md`. 커밋 제목 `[component] subject`, 한 주제 한
커밋, author `SeungHui Lee <shsh1004.lee@samsung.com>`, committer Claude, trailer `Co-authored-by: Claude
<noreply@anthropic.com>` + `Signed-off-by: SeungHui Lee <shsh1004.lee@samsung.com>` + 세션 attribution 줄.
바뀐 줄만 `clang-format-diff-14`. NEON은 `__aarch64__` 가드 + 스칼라 경로. 결과는 57 §9 새 절, push.
**각 최적화는 측정 분해 → 수정 → 정확도 게이트(§4.2) → 속도(§4.3).**
