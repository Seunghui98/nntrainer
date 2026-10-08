# 58. Gemma-4 26B-A4B 전부-NPU prefill — 인수인계 (2026-10-08)

이 문서 하나로 다음 세션이 이어서 작업할 수 있게 쓴 인수인계서다. 자세한 이력은
`57_gemma4_all_npu_task.md` §9(특히 §9.10–§9.17)에 있고, 여기서는 **지금 상태, 열린 문제,
다음에 할 일, 그것을 하는 방법과 함정**을 모은다.

- 브랜치: `claude/zealous-bell-a2pot9` (Seunghui98/nntrainer, PR 4385 위에 세움, PR 4343 일부 cherry-pick)
- 마지막 코드 커밋: `2e3eadd1` (문서 `1077f69f`)
- 사용자: 한국어로 소통. 기기(Galaxy S25 Ultra) 실행은 사용자가 직접 하고 로그를 붙여 준다.
  클라우드 세션에는 Hexagon SDK가 없어 **skel(DSP 코드)을 컴파일할 수 없다**.

---

## 0. 한 줄 요약

Gemma-4 26B-A4B를 embedding만 빼고 전부 NPU(HTP: HMX+HVX)에서 돌린다. 1024토큰 prefill은
**9.38 s(446 tok) → 4.864 s(1024 tok)** 까지 왔다. 그러나 **생성 문장이 깨져 있다**(아래 §3 P0).
속도 작업보다 정확도 원인 찾기가 먼저다. 속도 쪽 남은 몫은 flash 바닥 3.4 s까지 약 1.46 s다.

---

## 1. 대상과 구성

| 항목 | 값 |
|---|---|
| 기기 | Galaxy S25 Ultra, `ANDROID_SERIAL=R3CY10WM83Y`, Hexagon **V79** (PR 4343의 실측은 v81이라 수치를 그대로 비교 못 함) |
| 모델 경로 | `/data/local/tmp/nntrainer/causallm/models/gemma4-26b-a4b-qs4cx-arm` (QS4CX FC bin) |
| 모델 모양 | 30층, hidden 2816, MoE 128 expert top-8 (expert 3,010,560 B), sliding 층 hd 256 / KV 8 head, full 층(L5,11,17,23,29) hd 512 / KV 2 head, Q 16 head, window 1024 |
| config 핵심 키 | `moe_engine`, `attn_proj_engine`, `dense_ffn_engine`, `attention_engine`, `lmhead_engine` 모두 `"htp"`; `fc_layer_dtype: QS4CX`; `model_tensor_type: Q4_0-FP32`; `moe_cache_experts: 16`(30×16=480 slot); `skip_prefill: true`(⚠ §3 P0); prompt = 1024토큰 Ardley 요약 글; `num_to_generate: 32` |
| int8 attention | `"attention_kv_dtype": "q8"` 추가 시 PR 4343 row-blocked 커널. 이때 `init_seq_len`/`max_seq_len`을 **1088**로 줄여야 int8 KV cache(DSP heap, 층당 K/V 마스터+타일 = sliding 16 MiB)가 들어간다 |
| 연산 위치 | 모든 matmul은 HMX. FC/qkv/o/dense/MoE = u8×i4(`mm_u8i4_*`, QS4CX), fp16 attention = fp16×fp16(`attn_f16_prefill`), int8 attention = u8×i8 inline HMX(`hexkl_attn_q2.c`), lm_head = HVX Q4M1 GEMV |
| expert 스트리밍 | arena(ION) 480 slot ≈ 1.45 GB. prefill은 층 N 계산 중 N+2의 128 expert를 flash에서 선읽기(3,360개/prefill). flash 바닥 ≈ 3.4 s(C=16), C=24면 3.1 s. prefill ≈ max(계산, flash 읽기) |

---

## 2. 측정 이력 (모두 사용자 기기 실측)

| 시점 / 커밋 | 조건 | prefill | 메모 |
|---|---|---|---|
| §9.9 | 전부-NPU 첫 실행, 446 tok | 9,378 ms | 출력 `<i></i>` 반복 |
| §9.10 수정 6개 (600ac20d…780dca95) | | | dense 등록을 로드로, K^T 전치 vshuff, lm_head 로드 배치, KV cache ION, router/RoPE ION |
| §9.12 b0bbd264 | 1024 tok, attention·lm_head CPU | 6,452 ms | |
| §9.13 e8f0c7b0 | 1023 tok 전부-NPU | 6,230 ms | 출력 "A model is a set of instructions…" — **요약이 아님(당시 정상으로 오판)** |
| §9.16 4e903b97+798d3032 | router 블로킹 | 5,971 ms | router 648→248 ms. 출력 `A` 반복 |
| 9b6d59a0 | attention fp16 스테이징 우회 | 5,120 ms | attention 노드 91→56.7 ms/층. 출력 `■s a, a/A/AA/A/A/` |
| PR 4343 cherry-pick + `q8` | int8 row-blocked | 4,967 ms | attention 호출 28–36 ms/층 |
| 0583c9eb + 2e3eadd1 | Q/out ION, NEON 보정 | **4,864 ms** | attention 호출 **17–18 ms/층**. 출력 fp16과 **글자까지 같은** 깨진 문장 |

실행 간 노이즈 ±3–5%(FC 행은 ±10%). 100 ms 단위 비교는 냉각(2분) 후 3회 평균이 필요하다.

### 2.1 지금 prefill 4.86 s의 구성 (마지막 실행, `NNTR_HTP_PROFILE=1`)

| 행(HTP-PROFILE) | 의미 | host ms | ms/call |
|---|---|---|---|
| `K=2816 N=2816 M>1` (무표기, 31 calls) | MoE 융합 호출 | 1,576 | 50.8 |
| `K=2816 N=8192 M>1 FC` | qkv 융합 | 538 | 17.9 |
| `K=2816 N=2816 M>1 dense` | dense FFN | 444 | 14.3 |
| `K=2816 N=128 M>1 FC` | router(+top-k) | 262 | 8.7 |
| `K=4096 N=2816 M>1 FC` | o-proj 등 | 218 | 2.9 |
| `K=5632 N=2816 M>1 FC` | | 157 | 5.2 |
| `K=2816 N=10240 M>1 FC` | full 층 qkv | 135 | 27.0 |
| `arm staging memcpy` | 호출 밖 tensor↔ION 복사 | 583 (7.8 GB, 14 GB/s) | |
| attention (logcat trace) | 30층 | ≈ 18×30 = 540 | |

프로파일 빌드 by-op(int8 이전, §9.17 직전): sparse_moe 2,147 · attention 1,329 · qkv 805 · ffn 493 · attention_out 361 · post_ffn_norm(residual_add) 292 · post_attention_norm(residual_add) 171.

### 2.2 int8 attention 한 층(sliding, 1024행) 분해

| 구간 | 지금 | 비고 |
|---|---|---|
| host 스케일 보정 | 0.6 ms | 스칼라일 때 6.7–7.4 (첫 prefill만) |
| FastRPC 전송 | 2.5 ms | `wall_us − total` |
| DSP append | 6.7 ms | quant 5.7 + bake 1.0. K/V 1024행 int8 양자화를 HMX 스레드 혼자 |
| **DSP 커널** | **4.65 ms** | qk 1.75, softmax 1.86, pv 1.15, wait 0.1 |
| Q/out memcpy | ≈ 4 ms | staging memcpy +125 ms/30층 |

---

## 3. 열린 문제 (우선순위 순)

### P0. 생성 문장이 깨진다 — 정확도 원인 미확정

- 증상: fp16 attention과 int8 attention 실행이 **같은 깨진 문장**을 낸다 → attention 경로가 원인이 아니다.
- **유력 가설 (코드로 확인, 기기 미측정)**: config에 `skip_prefill: true`가 있다.
  `CausalLM::run`은 이때 prompt의 N−1 토큰만 prefill하고, 마지막 토큰을 decode 경로(M==1)로 넣어
  첫 생성 토큰을 만든다(`causal_lm.cpp` `SKIP_PREFILL` 분기, Gemma4는 `appendSkipPrefillIfNeeded`로
  lm_head 등에 `skip_prefill` 속성). 즉 **생성된 토큰은 전부 decode(M==1) NPU 경로에서 나온다**.
  prefill이 맞아도 decode 경로(dspq MoE, M==1 QS4CX FC, decode attention, `htpDecodeLmHead`)가
  틀리면 지금 증상이 된다. §9.13의 "fluent하지만 엉뚱한 문장"도 decode가 prefill KV를 제대로
  못 읽는 경우와 맞는다.
- 같은 이유로 **`NNTR_PPL=1`이 `[PPL] no positions scored`만 찍는다**: skip_prefill이면 lm_head가
  prefill을 건너뛰어(`tie_word_embedding.cpp` `incremental_forwarding_lmhead`의 `skip_prefill && is_prefill`
  조기 return) 채점할 행이 없다.
- **MoE를 CPU로 돌리면 기기 연결이 끊긴다**(`waiting for device`, 2회 재현). CPU MoE가 메모리를
  넘는 것으로 보인다(미확인). 그래서 "전부 CPU" 기준선을 이 모델로는 못 잡았다.
- 다음 실험(재빌드 없이 config만):
  1. `skip_prefill: false` + `NNTR_PPL=1` + `num_to_generate: 16`로 엔진 사다리(아래 §6.4 스크립트,
     MoE는 htp 고정). PPL이 찍히면 prefill 정확도를 엔진별로 비교할 수 있다.
  2. 같은 실행에서 **첫 생성 토큰은 prefill 로짓에서 나온다**. 첫 토큰은 말이 되는데 이후가 깨지면
     decode 경로 문제로 확정.
  3. decode 경로라면 decode 엔진을 하나씩 끈다: `attention_engine` cpu(decode attention),
     `lmhead_engine` cpu(`htpDecodeLmHead`), `NNTR_HTP_DSPQ=0`(dspq MoE 끔).

### P1. MoE 호출 회귀 (+290 ms)

`K=2816 N=2816 M>1` 41.5 ms/call(§9.13) → router 블로킹(4e903b97) 이후 네 실행 연속 49–51 ms.
MoE 커널은 안 바뀌었다. 가설(미측정): 열, 또는 router가 빨라져 MoE 호출이 prefetch reader 7스레드의
flash burst와 더 겹침. `NNTR_HTP_PROFILE=2`는 융합 호출(qkv·MoE·dense·router)의 DSP 단계를 주지 않는다
(dsp≈0). → `mm_u8i4_moe_layer_norm`에 timed 변형(또는 stats out)을 붙여 gather·requant·mm·swiglu·scatter를
보는 것이 먼저. 4e903b97을 되돌린 skel로 A/B 한 번이면 router와의 관계가 확정된다.

### P2. 실행 간 노이즈

FC 행이 실행마다 ±10%. 결론은 3회 평균으로.

---

## 4. 최적화 후보 (P0 해결 뒤, 기대치는 모두 산술·기기 미측정)

| # | 항목 | 지금 | 기대 | 방법 / 파일 |
|---|---|---|---|---|
| 1 | MoE 회귀 | 50.8 ms/층 | −0.29 s | P1 참조 |
| 2 | 노드 간 activation 복사 | staging 583 ms | −0.4~0.5 s | 연속 NPU 노드 사이 activation을 ION에 상주(KV cache의 `CausalLM::installKVCacheSharedAllocator` 방식). 호출 출력 ION 버퍼를 다음 호출 입력으로. `htp_compute_ops.cpp`의 `act_pool_`/`out_pool_`/`stagedMemcpy` 경로 |
| 3 | epilogue `residual_add` 2호출/층 | ≈0.46 s | −0.3 s | o-proj·down 호출의 post 단계로 합쳐 round trip 2개 제거(2와 겹침). `residual_add` 레이어, `mm_u8i4_layer_norm` post 단계 |
| 4 | router | 8.7 ms/층 | −0.2 s | `hvx_router_rows_f32.c`: w(1.4 MB)를 VTCM에, 루프를 행 바깥·k 안쪽으로. 지금은 KB=256 k-chunk마다 x(1024×2816 f32=11.5 MB)를 다시 읽어 호출당 ~127 MB, 메모리 bound |
| 5 | int8 attention append | 6.7 ms/층 | −0.12~0.2 s | `hexkl_kv_q.c` `append_fixed_i8`의 행 루프를 worker pool에 분산. 또는 qkv 호출 post 단계에서 K/V를 int8 마스터로 바로 써서 append 제거 |
| 6 | FC 효율 (qkv·dense·o) | qkv 17.9 ms/call ≈ 2.6 TOPS | −0.3~0.5 s(불확실) | `NNTR_HTP_PROFILE=2` FC 분해(quant/mm/acc/dequant)로 병목 확인 후. 매트멀 중 caller는 HMX, worker 5개는 dequant |
| 7 | flash 바닥 | 3.4 s | 3.1 s | 계산이 3.4 s 아래로 가면 `moe_cache_experts` 24 |
| — | decode | 3.84 TPS (260 ms/tok) | 별개 | expert miss 2.23/call → 파일 읽기 141 ms/tok, RPC 85회/tok |

---

## 5. 지금까지 한 일 (커밋 단위 요약, 최신 → 과거)

| 커밋 | 내용 |
|---|---|
| 2e3eadd1 | int8 attention 첫 prefill 스케일 보정을 NEON으로(`layers/abs_max.h`). compare+select로 `std::max`와 비트 동일(vmaxnm은 sNaN에서 다름). 검사 `test/htp/host/abs_max_check.cpp`, qemu-aarch64로 NEON 경로 실행 |
| 0583c9eb | `sdpa_q2_kvcache`: Q/out을 ION 스테이징(`attn_q_pool_`/`attn_out_pool_`) |
| 6c559ff7 | `NNTR_HTP_ATTN_TRACE`: q2 DSP 단계(append/kernel/qk/softmax/pv/wait)와 host 보정 시간 logcat |
| 86bb496b | PR 4343 커널을 이 브랜치의 void `hvx_worker_pool_submit`에 맞춤(skel 컴파일 에러 수정) |
| 9b6d59a0 | Android fp16 빌드에서 가속 attention일 때 Q/K/V/O fp16 스테이징 생략 |
| 2d4ab5a1…8eb32af5 | PR 4343(haehun) 11개 cherry-pick: row-blocked int8 attention, 정수 softmax, convert-unit readout, inline HMX, CPU A8W8 참조, gtest. IDL `session_info` 충돌은 8칸 형태로 병합(`res[7]`=fp16 HMX rate) |
| 798d3032 | mha_core host 몫 트레이스 |
| 4e903b97 | router rows 커널 register blocking(호스트 에뮬 `ROUTER ROWS OK`) |
| 780dca95, 6486e76f, e86e8951, 7ebacefa, 600ac20d | §9.10 감사 수정: router/RoPE ION, lm_head 로드 배치·CPU 사본 생략, K^T vshuff 전치, KV cache ION, dense 등록 로드로 |
| 3f29c72d, e6fcee38, 0456e0ee | 기기 빌드 에러 수정(tanhf 없음, malloc.h/size_t, IDL 예약어 `out`) |
| 그 이전 | RoPE·norm·router top-k·epilogue·lm_head(norm+softcap)를 HTP 호출 안으로(57 §9.1–9.7) |

---

## 6. 명령

### 6.1 클라우드 세션(host)에서 할 수 있는 검증

```bash
cd /home/user/nntrainer
git submodule sync && git submodule update --init --depth 1   # 처음 한 번
meson build -Denable-transformer=true                          # 처음 한 번
ninja -C build
./build/Applications/CausalLM/unittest_causallm_models          # 93 passed / 15 skipped 기대
./build/test/unittest/unittest_cpu_kv_q_attention               # PR 4343, 6 passed
./build/test/unittest/unittest_hvx_softmax_q                    # PR 4343, 7 passed
bash test/htp/host/run_host_checks.sh                           # HVX 에뮬 검사 전부, 몇 분
bash tools/htp_syntax_check.sh                                  # htp_compute_ops.cpp 타입 검사 (FastRPC 시그니처는 아님)
git diff -U0 | clang-format-diff-18 -p1 -i                      # 바뀐 줄만 포맷
```

- skel(DSP) 소스는 여기서 **컴파일되지 않는다**. DSP 코드를 바꾸면 PR 쪽과 우리 쪽 헤더의
  함수 선언을 비교하는 식으로 최대한 정적으로 확인하고, 사용자에게 `test/htp/build.sh` 결과를 받는다.
- NEON 코드는 `aarch64-linux-gnu-g++ -static` + `qemu-aarch64`로 실행 검사(이번 세션에서 apt로 설치함;
  새 컨테이너면 `apt-get install -y qemu-user g++-aarch64-linux-gnu`).

### 6.2 사용자 PC 빌드 (사용자에게 줄 명령)

```bash
cd ~/workspace/nntrainer
git pull origin claude/zealous-bell-a2pot9 && git log -1 --oneline
export HEXAGON_SDK_ROOT=$HOME/workspace/Hexagon_SDK/6.4.0.2 HEXKL_ROOT=$HOME/workspace/hxkl-beta2/hexkl_addon ANDROID_NDK=$HOME/workspace/android-ndk-r26d
export ANDROID_SERIAL=R3CY10WM83Y
D=/data/local/tmp/nntrainer/causallm; P=/data/local/tmp/nntrainer/causallm_prof
M=$D/models/gemma4-26b-a4b-qs4cx-arm
cool() { while adb shell 'ps -A' | grep -q 'nntrainer_causall[m]'; do sleep 10; done; sleep 120; }

./test/htp/build.sh 2>&1 | grep -E "error|UNDEFINED|built:"     # DSP 소스나 IDL이 바뀌었을 때만
(cd Applications/CausalLM && ./build_android.sh --htp)          # 15분, --cache 금지, stub도 재생성
grep -c "attn_q2_step" test/htp/generated/nntr_hvx.h nntrainer/tensor/htp_backend/generated/nntr_hvx.h   # 둘 다 >0

adb push test/htp/build/libnntr_hvx_skel.so $D/
adb push Applications/CausalLM/jni/libs/arm64-v8a/nntrainer_causallm $D/
adb push Applications/CausalLM/jni/libs/arm64-v8a/libcausallm_core.so $D/
adb push builddir/android_build_result/lib/arm64-v8a/libnntrainer.so $D/
adb push builddir/android_build_result/lib/arm64-v8a/libccapi-nntrainer.so $D/
```

### 6.3 기기 실행

```bash
adb push cfgB.json $M/nntr_config.json        # cfgA = 엔진 5개 htp, cfgB = cfgA + q8 + seq 1088
cool; adb logcat -c
adb shell "cd $D && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. NNTR_NUM_THREADS=8 NNTR_HTP_PROFILE=1 NNTR_HTP_ATTN_TRACE=1 \
  ./nntrainer_causallm $M" 2>&1 | tee run.log
grep -a 'prefill:\|generation:\|staging memcpy\|failed' run.log
adb logcat -d -s nntrainer | grep -E 'mha_core|attn trace|failed' | tail -9
sed -n '/<|turn>model/,/=====/p' run.log | grep -v '^\[HTP' | head -6
```

프로파일 빌드(노드별 by-op 표):

```bash
(cd Applications/CausalLM && ./build_android.sh --htp --profile)   # builddir 삭제, jni/libs 덮어씀
adb shell "rm -f $P/libnntr_hvx_skel.so"                            # skel은 $D 것을 씀
# 앱 4개 파일을 $P로 push (위와 같은 경로)
adb shell "cd $P && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=$D NNTR_NUM_THREADS=8 NNTR_HTP_PROFILE=1 ./nntrainer_causallm $M" 2>&1 | tee prof.log
python3 tools/prefill_timeline.py prof.log --config cfgB.json --by-op --by-layer
```

환경 변수: `NNTR_HTP_PROFILE=1|2`(2는 plain FC만 DSP 분해), `NNTR_HTP_ATTN_TRACE=1`(logcat에
`HTP attn trace f16|q2`, `mha_core trace`), `NNTR_PPL=1`(prompt nll, skip_prefill false일 때만),
`NNTR_MOE_PREFETCH=0`, `NNTR_MOE_CACHE_EXPERTS=C`, `NNTR_HTP_DSPQ=0`.

### 6.4 정확도 사다리 (P0용, MoE는 htp 고정 — CPU MoE는 기기 연결을 끊음)

```bash
python3 - <<'EOF'
import json
base = json.load(open('cfgA.json'))
base.pop('attention_kv_dtype', None)
base.update(init_seq_len=2048, max_seq_len=2048, num_to_generate=16, skip_prefill=False)
E = ['attention_engine', 'lmhead_engine', 'dense_ffn_engine', 'attn_proj_engine']
for i in range(len(E) + 1):
    c = dict(base, moe_engine='htp')
    for j, e in enumerate(E): c[e] = 'cpu' if j < i else 'htp'
    json.dump(c, open(f'acc_{i}.json', 'w'), indent=2, ensure_ascii=False)
EOF
for i in 0 1 2 3 4; do
  adb push acc_$i.json $M/nntr_config.json >/dev/null; adb logcat -c
  adb shell "cd $D && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. NNTR_NUM_THREADS=8 NNTR_PPL=1 ./nntrainer_causallm $M" > acc_$i.log 2>&1
  echo "== acc_$i"; grep -a '\[PPL\]' acc_$i.log
  sed -n '/<|turn>model/,/=====/p' acc_$i.log | grep -v '^\[HTP' | head -4
  cool
done 2>&1 | tee acc_summary.txt
```

acc_0은 전부 htp, acc_1은 attention cpu, acc_2는 +lmhead, acc_3은 +dense, acc_4는 +attn_proj(MoE만 htp).

---

## 7. 함정 (이번 작업에서 실제로 밟은 것)

- **IDL이 바뀌면 skel과 앱 stub을 둘 다 새로 만들어야 한다.** `test/htp/build.sh`는
  `test/htp/generated/`(gtest·skel용)를, `build_android.sh --htp`는 `nntrainer/tensor/htp_backend/generated/`
  (앱용)를 만든다. 하나만 옛것이면 메서드 번호가 어긋나 `0x8000040e` 같은 실패나 엉뚱한 함수 호출.
- `build.sh`의 qaic 단계가 실패하면 옛 generated 파일이 남는다. 에러를 꼭 확인.
- 프로파일 빌드는 builddir을 지우고 `jni/libs`를 덮어쓴다. 일반 실행 전 다시 `--htp` 빌드.
- `$P`(프로파일 디렉터리)에 옛 `libnntr_hvx_skel.so`가 있으면 그것이 먼저 로드된다. 지울 것.
- logcat은 `adb logcat -c` 후 `tail`로 볼 것. `head`면 이전 실행 줄이 섞인다(한 번 오판함).
- Hexagon 쪽 libc에 `malloc.h`, `tanhf`가 없다. qaic는 `out`을 예약어로 쓴다.
- PR 4343은 `hvx_worker_pool_submit`이 int를 돌려주는 옛 API 기준이다. 이 브랜치는 void,
  `wait`는 진행 중 작업이 없으면 no-op. 다른 PR 코드를 가져오면 헤더 선언부터 비교할 것.
- `skip_prefill: true`면 PPL이 안 찍히고 첫 토큰이 decode 경로에서 나온다.
- MoE를 CPU로 돌리면 기기 연결이 끊긴다(2회).
- int8 KV cache는 DSP heap(calloc)이다. `max_seq_len 2048`이면 30층 ≈ 440 MiB → 1088로 줄일 것.
  등록 실패 시 logcat `quantized KV cache register failed; this layer stays on the fp16 path`.
- "검사로 확인했다"는 검증이 아니다. 기기에서 안 잰 것은 "기기 미측정"이라고 적는다.
- 가설보다 측정 분해가 먼저다(이 프로젝트에서 첫 추측이 두 번 틀렸다: attention host 시간, PR 커널 미적용).

---

## 8. 코드 지도

| 영역 | 파일 |
|---|---|
| 모델 그래프·엔진 키 | `Applications/CausalLM/models/gemma4/gemma4_causallm.cpp`, `models/transformer.cpp`, `models/causal_lm.cpp` |
| attention 레이어 | `Applications/CausalLM/layers/mha_core.cpp` (`try_quantized_attention`, `try_accelerated_attention`, `calibrate_q2_scales`) |
| lm_head·PPL | `Applications/CausalLM/layers/tie_word_embedding.cpp` |
| HTP 호스트 쪽 | `nntrainer/tensor/htp_backend/htp_compute_ops.cpp` (`sdpa_q2_kvcache`, `sdpa_fp16_kvcache`, 스테이징 풀, 프로파일) |
| DSP 진입점 | `test/htp/nntr_hvx.idl`, `test/htp/nntr_hvx_mm_u8i4.c`(FC·MoE·router), `test/htp/nntr_hvx_attn_q.c`(q2 step), `test/htp/nntr_hvx_attn.c`(fp16) |
| DSP 커널 | `nntrainer/tensor/htp_backend/hmx/hexkl_attn_q2.c`, `hexkl_kv_q.c`, `hexkl_mm_u8i4_dma.c`; `hvx/hvx_router_rows_f32.c`, `hvx_worker_pool.c` |
| 호스트 에뮬 검사 | `test/htp/host/run_host_checks.sh` |
| 도구 | `tools/prefill_timeline.py`, `tools/htp_syntax_check.sh` |

---

## 9. 작업 규칙

- `CLAUDE.md`, `AGENTS.md`, `docs/htp_attention/01_working_style.md`를 먼저 읽는다.
- 커밋: 제목 `[component] subject`, 한 주제 한 커밋. author `SeungHui Lee <shsh1004.lee@samsung.com>`,
  committer Claude(`git -c user.name=Claude -c user.email=noreply@anthropic.com commit --author=...`).
  trailer: `Co-authored-by: Claude <noreply@anthropic.com>`, `Signed-off-by: SeungHui Lee <shsh1004.lee@samsung.com>`,
  그다음 세션 하네스가 주는 attribution 줄.
- 바뀐 줄만 `clang-format-diff-18`. `subprojects/` 수정 금지. cross-platform 유지(NEON은 `__aarch64__` 가드 + 스칼라 경로).
- 작업이 끝나면 커밋하고 `git push -u origin claude/zealous-bell-a2pot9`.
- 결과는 `57_gemma4_all_npu_task.md` §9에 새 절로 적고, 기기에서 안 잰 숫자는 "기기 미측정".
- 사용자에게는 한국어로, 기기 명령은 복붙할 수 있는 한 블록으로 준다.
