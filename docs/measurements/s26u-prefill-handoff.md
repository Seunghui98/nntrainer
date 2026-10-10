# S26 Ultra Gemma-4 2-bit prefill 최적화 — 인수인계 (2026-10-10, 밤 갱신)

다른 Claude 세션이 이 문서만 보고 이어서 작업할 수 있도록 환경, 빌드, 설치, 실행, 측정 규칙,
지금까지의 결과, 남은 작업을 모두 적는다. 숫자는 모두 기기 실측이다 (S26 Ultra, 2분 cool-down).

## 0-1. 저장소와 브랜치

| 이름 | URL | 쓰임 |
|---|---|---|
| origin (내 fork) | `git@github.com:Seunghui98/nntrainer.git` (https://github.com/Seunghui98/nntrainer) | push 대상 |
| upstream | https://github.com/nntrainer/nntrainer | PR들이 열린 곳 |
| dlwlzero (PR 4412 작성자 fork) | https://github.com/dlwlzzero/nntrainer | PR 4412의 head `htp_decode` |

| 브랜치 / PR | 내용 |
|---|---|
| `origin/htp/s26-prefill-opt` | **PR용 깨끗한 브랜치**, 최적화 커밋 8개 (§10). PR base는 `dlwlzzero/nntrainer:htp_decode` 권장 |
| `origin/htp/s26-prefill-handoff` | **이 문서 + B WIP 커밋** (`htp/s26-prefill-opt` 위). 다음 세션은 여기서 시작 |
| PR 4412 https://github.com/nntrainer/nntrainer/pull/4412 | "gemma htp_decode" (dlwlzzero). 이 작업의 base (`b398e078c`). decode 쪽 작업 |
| PR 4415 https://github.com/nntrainer/nntrainer/pull/4415 | 내 이전 S25 prefill 작업 (`Seunghui98:claude/zealous-bell-a2pot9`), PR 4412가 merge해 감 |
| PR 4343 https://github.com/nntrainer/nntrainer/pull/4343 | haehun의 llama.cpp 스타일 attention (`hexkl_attn_q2`, u16 a16 경로, `mha_core` 변환의 원작) |

시작:
```bash
cd ~/workspace/nntrainer
git fetch origin && git checkout htp/s26-prefill-handoff     # 문서 + B WIP
git log --oneline -10
git fetch upstream pull/4412/head:pr4412                     # base 최신 확인용
```
PR 4412가 업데이트되면 `git fetch upstream pull/4412/head` 후 rebase 여부를 사용자에게 물어볼 것.

## 0-2. 작업 방식 (ponytail)

`~/.claude/CLAUDE.md`의 **Ponytail mode**, 저장소 `CLAUDE.md`, `docs/htp_attention/01_working_style.md`를
먼저 읽고 따른다 (별도 Skill로 등록돼 있지 않다; CLAUDE.md 규칙이다). 요지:
- 게으른 시니어: 가장 좋은 코드는 안 쓴 코드. 문제를 끝까지 이해한 뒤 사다리를 오른다 —
  필요한가? → 이미 코드베이스에 있나(재사용) → 표준 라이브러리 → 플랫폼 기능 → 설치된 의존성 → 한 줄 → 최소 코드.
- 버그는 근본 원인에서 고친다 (호출자 전부 grep, 공유 함수에 가드 하나).
- 요청 안 한 추상화·scaffolding 금지. 삭제가 추가보다 낫다. 가장 짧은 diff — 단 이해를 건너뛰지 않는다.
- 일부러 자른 모서리는 `ponytail:` 주석으로 한계와 업그레이드 경로를 적는다.
- 분기·루프·파서 같은 비자명 로직은 실행 가능한 검사 하나를 남긴다 (여기선 host check 케이스).
- **측정 먼저**: 가설을 고치기 전에 구간을 잰다. 기기에서 안 돌린 건 "기기 미측정"이라고 쓴다.
- 이 프로젝트 고유: qf32 누적 금지, attention 변환에 CPU(NEON) 연산 금지(사용자 요구), 비교는 기기 대 기기.

## 0. 한 줄 요약

- 브랜치 `htp/s26-prefill-opt`에 커밋 11개 (§10). base는 PR 4412 head (`b398e078c`).
  마지막 3개(2026-10-10 밤)가 attention 가장자리 작업의 확정분: FC 호출 u16 입력, attention f32 Q·u16
  context를 DSP에서, NEON quant contraction off. **push는 아직 안 됨** (이 워크스테이션에 GitHub 자격증명
  없음: `gh auth login` 또는 SSH 키 뒤 `git push origin htp/s26-prefill-opt htp/s26-prefill-handoff --force-with-lease`).
- `htp/s26-prefill-handoff` = opt + 이 문서 + WIP 커밋 1개(qkv 패스에서 Q를 정적 scale로 u16 출력, 정확도
  미확정, §7).
- 그 다음: §7의 열린 질문(정확도 게이트), 남은 prefill 최적화(§8), decode(§9).

## 1. 작업 환경 (이 워크스테이션)

| 항목 | 경로 / 값 |
|---|---|
| 저장소 | `~/workspace/nntrainer` |
| Hexagon SDK | 공용 `/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1` (이 머신엔 6.4.0.2 없음. 원래 머신은 `~/workspace/Hexagon_SDK/6.4.0.2`) |
| HexKL | `~/workspace/hxkl-beta2/hexkl_addon` (1.0-beta.2; `qpm-cli --download-only hexagon_kl -v 1.0.0-beta2`로 받음, lib `6.6.0.0/armv8_android26`, DSP `hexagon_toolv19_v81`) |
| Android NDK | `~/workspace/android-ndk-r26d` (dl.google.com에서 받아 풂) |
| meson / python | meson 1.3.2, python 3.12 (pip 없음) |
| clang-format | `~/.local/bin/clang-format-diff-14` → `~/.venv/cf`의 pip wheel 14.0.6 (sudo 없음) |
| 셸 | bash. `setup_sdk_env.source`는 0이 아닌 값을 돌려주므로 `&&`가 아니라 `;`로 이어야 한다 |

`tools/htp/env.sh`는 다른 머신 경로라 쓰지 않는다. 새 셸마다 아래를 export 한다.

```bash
export ANDROID_NDK=$HOME/workspace/android-ndk-r26d
export PATH=$ANDROID_NDK:$HOME/.local/bin:$PATH
export HEXAGON_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1
export HEXKL_ROOT=$HOME/workspace/hxkl-beta2/hexkl_addon
```
`ninja -C builddir install`은 이 머신에서 gmock을 `/usr/local`에 넣으려다 실패한다:
`ninja -C builddir && meson install -C builddir --no-rebuild --skip-subprojects`.
**세션 하나만**: `claude --resume`으로 같은 세션이 둘 떠서 같은 트리·기기를 동시에 쓴 적이 있다(측정 오염).
시작 전 `ps -eo pid,etime,cmd | grep -E 'claude|adb shell|run.sh'`.

환경이 없는 머신이면: 위 네 디렉터리가 있는지 `ls`로 먼저 확인하고, 없으면 사용자에게 받아야 한다
(SDK/HexKL은 Qualcomm 배포물이라 받을 수 없음). `builddir`이 없으면 §3.2로 만든다.

## 2. 기기

| 기기 | serial | SoC | DSP | 비고 |
|---|---|---|---|---|
| **S26 Ultra (현재)** | `R5KL309ALRK` | SM8850 | **V81**, HVX 8개, VTCM 8 MB, 2112 MHz | RAM 11.4 GB |
| S25 Ultra (이전) | `R3CY10WM83Y` | SM8750 | V79, HVX 6개 | |

serial을 모르면:
```bash
adb devices -l                     # serial 확인
adb -s <serial> shell 'getprop ro.product.model; getprop ro.soc.model'   # SM-S948N / SM8850 이면 S26U
export ANDROID_SERIAL=<serial>
```
- V81(S26U)이면 skel을 `HEX_ARCH=v81`로, V79(S25)면 기본값(v79)으로 빌드한다. 다른 SoC면 이 문서의 숫자는 다시 재야 한다.
- 기기 작업 디렉터리 `D=/data/local/tmp/nntrainer/causallm`, 바이너리는 `$D/pr4412/`.
- 모델: `$D/models/gemma4-26b-a4b-ternary-fcqs4cx/` (`nntr_gemma4_26b_a4b_qs2cx_fcqs4cx.bin`, 7,540,724,856 B).
  로컬 원본: `Applications/CausalLM/res/gemma4_26ba4b/ternary_fcqs4cx/`. 새 기기면 `adb push`(약 3분).
- `$D/pr4412/nntrainer_causallm.farf`(내용 `0x1f`)가 있으면 DSP FARF 로그가 logcat에 나온다.

## 3. 빌드

### 3.1 매번 (코드 바꾼 뒤)
```bash
cd ~/workspace/nntrainer
# (1) IDL(test/htp/nntr_hvx.idl)을 바꿨으면 ARM stub 재생성 -- 안 하면 ninja가 "stub is older than idl"로 실패
(unset HEXAGON_SDK_ROOT; cd $HOME/workspace/Hexagon_SDK/6.4.0.2 && source ./setup_sdk_env.source >/dev/null \
  && cd ~/workspace/nntrainer && bash nntrainer/tensor/htp_backend/generate_stub.sh)
# (2) DSP skel (S26U = v81). 출력 끝에 "ARCH OK (V81)", error: 줄이 없으면 성공
(unset HEXAGON_SDK_ROOT; cd $HOME/workspace/Hexagon_SDK/6.4.0.2 && source ./setup_sdk_env.source >/dev/null \
  && cd ~/workspace/nntrainer/test/htp && HEX_ARCH=v81 ./build.sh)
#     진단 플래그는 HEX_EXTRA_CFLAGS="-D..." 로 (zsh에서는 export 해서 넘길 것)
# (3) libnntrainer (위 export 필요)
ninja -C builddir install
# (4) 앱 (libcausallm_core.so, nntrainer_causallm)
(cd Applications/CausalLM && ./build_android.sh --htp --cache)
```
산출물:
- skel: `test/htp/build/libnntr_hvx_skel.so`
- `builddir/android_build_result/lib/arm64-v8a/{libnntrainer.so,libccapi-nntrainer.so,libc++_shared.so,libsdkl.so}`
- `Applications/CausalLM/jni/libs/arm64-v8a/{nntrainer_causallm,libcausallm_core.so}`

### 3.2 builddir 새로 만들기 (없을 때만)
```bash
cd ~/workspace/nntrainer
git submodule update --init --depth 1
meson setup builddir -Dplatform=android -Dopenblas-num-threads=1 -Denable-tflite-interpreter=false \
  -Denable-tflite-backbone=false -Denable-fp16=true -Dnntr-num-threads=4 -Dhgemm-experimental-kernel=false \
  -Denable-htp=true -Dhexkl-sdk-root=$HEXKL_ROOT -Dhexkl-lib-subdir=6.6.0.0/armv8_android26 \
  -Dhexagon-sdk-root=$HEXAGON_SDK_ROOT -Darm-arch=armv8.2-a "-Darm-march=-march=armv8.2-a+fp16+dotprod+i8mm"
```
(stub이 없으면 meson이 멈춘다: 먼저 3.1 (1).) `rm -rf builddir`는 권한 거부될 수 있으니 `meson setup --wipe`.

### 3.3 host 검사 (기기 없이, 커밋 전 필수)
```bash
cd ~/workspace/nntrainer/test/htp/host && bash run_host_checks.sh > /tmp/hc.log 2>&1
grep -c "WRONG\|DIFFERENT\|MUTANT SURVIVED\|fatal error" /tmp/hc.log   # 0 이어야 함
grep "ROUTER ROWS\|ROPE ROWS\|MOE KERNEL MATCHES" /tmp/hc.log           # 모두 OK
```
약 10분. skel 소스(`test/htp/nntr_hvx_mm_u8i4.c` 등)는 host에서도 컴파일되므로 HVX intrinsic /
`hexagon_types.h`를 넣으면 안 된다 (HVX 코드는 `nntrainer/tensor/htp_backend/hvx/*.c`에, host는 `hvx_emu`로 돈다).

## 4. 설치
```bash
export ANDROID_SERIAL=R5KL309ALRK; D=/data/local/tmp/nntrainer/causallm
adb shell mkdir -p $D/pr4412
adb push test/htp/build/libnntr_hvx_skel.so $D/pr4412/
adb push builddir/android_build_result/lib/arm64-v8a/{libnntrainer.so,libccapi-nntrainer.so,libc++_shared.so,libsdkl.so} $D/pr4412/
adb push Applications/CausalLM/jni/libs/arm64-v8a/{nntrainer_causallm,libcausallm_core.so} $D/pr4412/
adb shell "cd $D/pr4412 && md5sum *.so"   # 로컬 md5와 비교
```

## 5. 실행

### 5.1 config (기기에 이미 있음; 새 기기면 만든다)
`$D/g1024_i2048`, `$D/g2048_i2048`, `$D/g4096_i2048` — 각 디렉터리에 `nntr_config.json` + greedy
`generation_config.json`(`do_sample:false`) + 모델 폴더의 `config.json / tokenizer*.json` 심링크.
프롬프트는 `docs/measurements/260-prompt{1024,2048,4096}.txt`, `num_to_generate` 128,
`init_seq_len` 2048 (4096 프롬프트는 2048 행씩 chunk 2번), `max_seq_len` = 프롬프트 + 128 이상
(1024: 2048, 2048: 2176, 4096: 4224). 엔진: moe/attn_proj/dense_ffn/attention = htp, lmhead = cpu,
`attention_kv_dtype` q8, `skip_prefill` false, `bad_word_ids` [] (없으면 앱이 바로 죽음).
로컬 사본: `~/workspace/device_pull_R3CY10WM83Y/g*_i*/`.

새로 만들 때 (python, 로컬에서 만들고 push):
```python
import json
base = json.load(open('Applications/CausalLM/res/gemma4_26ba4b/ternary_fcqs4cx/nntr_config.json'))
MD = '/data/local/tmp/nntrainer/causallm/models/gemma4-26b-a4b-ternary-fcqs4cx'
for p, mx in ((1024, 2048), (2048, 2176), (4096, 4224)):
    d = dict(base)
    d.update(skip_prefill=False, model_file_name=MD + '/nntr_gemma4_26b_a4b_qs2cx_fcqs4cx.bin',
             tokenizer_file=MD + '/tokenizer.json', attn_proj_engine='htp', dense_ffn_engine='htp',
             attention_engine='htp', lmhead_engine='cpu', attention_kv_dtype='q8',
             sample_input=open('docs/measurements/260-prompt%d.txt' % p).read(),
             num_to_generate=128, init_seq_len=2048, max_seq_len=mx, bad_word_ids=[])
    json.dump(d, open('/tmp/g%d.json' % p, 'w'), indent=2, ensure_ascii=False)
```
기기: `mkdir $D/g<p>_i2048`, `nntr_config.json` push, `generation_config.json`(do_sample false로 고친 것) push,
나머지 json은 `ln -sf $MD/<f> $D/g<p>_i2048/<f>`.

### 5.2 실행 명령
```bash
export ANDROID_SERIAL=R5KL309ALRK; D=/data/local/tmp/nntrainer/causallm
ENV="NNTR_NUM_THREADS=8 NNTR_HTP_E2E=1 NNTR_HTP_E2E_FREE_KVQ=1 NNTR_HTP_KV_DROP_CPU=1 NNTR_HTP_DROP_HOST_FC=1 NNTR_HTP_ATTN_M1_Q8=2 NNTR_HTP_FC_IN_ARENA=1 NNTR_MOE_CACHE_EXPERTS=32"
adb shell "cd $D/pr4412 && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH='$D/pr4412' $ENV ./nntrainer_causallm $D/g2048_i2048" | grep -a "prefill:\|generation"
```
| env | 뜻 |
|---|---|
| `NNTR_HTP_E2E=1` | one-PD decode (PR 4412) |
| `NNTR_HTP_E2E_FREE_KVQ=1` / `NNTR_HTP_KV_DROP_CPU=1` | prefill KV 사본 정리 (decode 메모리) |
| `NNTR_HTP_DROP_HOST_FC=1` | CPU 쪽 FC weight 사본 해제 |
| `NNTR_HTP_ATTN_M1_Q8=2` | int8 decode attention |
| `NNTR_HTP_FC_IN_ARENA=1` | **필수**: FC weight를 heap 대신 arena에 (없으면 4k가 DSP heap에서 실패). 확인: 로그 `s1_heap_kib` ≈ 130000 |
| `NNTR_MOE_CACHE_EXPERTS=N` | 층당 상주 expert 수 C (기본 config 16). S26U 최대: 1k/2k 40, 4k 28 (4k는 24 권장) |
| `NNTR_HTP_PROFILE=1/2` | 구간·staging 위치별 시간 / FC·MoE DSP 세부 |
| `NNTR_HTP_ATTN_TRACE=1` | attention 층별 DSP 시간 (`adb logcat -d \| grep "attn trace"`) |
| `NNTR_HTP_ATTN_F32=0` | (B 패치에만 있음) attention 변환을 기존 NEON 경로로 |
| `NNTR_HTP_ATTN_F32_CHECK=1/2` | (B 패치) 두 경로 원소 비교 / f32 경로 결정성 |

## 6. 측정 규칙
- 실행 사이 **2분 cool-down** (`sleep 120`), 비교는 **번갈아(A,B,A,B) 2–3회**. 1회 숫자는 ±100 ms 흔들린다.
- 같은 기기에서 두 실행을 동시에 돌리지 않는다 (사용자가 직접 돌릴 때 백그라운드 측정을 멈출 것).
- 정확도: 출력 문장 md5 (`grep -a -o '<channel|>.*' log | head -1 | md5sum`; p4096은 128토큰 안에
  `<channel|>`가 없으므로 `<|turn>model` 뒤 텍스트에서 `[HTP` 줄 빼고 비교). 기준 md5 (현재 커밋):
  p1024 `1fd0625a`, p2048 `bfc92564`. `PROFILE=2`면 p1024 `8a8665bd`(다른 경로라 다름; PROFILE끼리만 비교).
- 기기 실측 아닌 것은 "기기 미측정"이라고 쓴다. qf32 누적은 쓰지 않는다. 비교는 기기 대 기기.
- 기기 재부팅 위험: `init_seq_len` 8192 (p4096에 2×규칙), `NNTR_MOE_PIN=2`. 하지 말 것.

## 7. attention 가장자리 (B) — 확정분과 열린 질문

**확정 (opt 브랜치 커밋 3개, 2026-10-10 밤, 기기 실측)**
- `7d9490881` FC 호출이 u16 activation을 받음: `hexkl_mm_opts.act_u16`, quant 패스의 pre-norm scratch에서
  head별 `(u − zp)·scale`로 역양자화, `mm_u8i4_layer_res_add_u16`. host check 비트 동일.
- `f5a70c4d5` attention: `attn_q2_step_fq`가 f32 Q(qkv 호출의 staging 버퍼, `ionAlias`로 복사 없이)를 받아
  pool에서 u16로(row-major + l2fetch, 층당 1.0 ms), u16 context는 `attn_out_pool_`에 남기고 o-proj
  (`gemm_qs4cx_res_add_fp32`)가 `attn_out16_` 레코드로 찾아 그대로 읽음. 커널·인코딩·수식 동일.
  mha_core는 빌더가 준 `attn_out_u16`(접힌 o-proj가 HTP일 때)에서만 이 경로. `NNTR_HTP_ATTN_F32=0` 기준선,
  `NNTR_HTP_ATTN_F32_CHECK=1` 두 경로 context 원소 비교, `=2` DSP 경로 2회(결정성).
- `66e023786` `quant_u16_f32`(NEON)에 `#pragma clang fp contract(off)`: 두 경로 context 차이 최대 125 → 8 step.

| p1024 30층 합 | NEON 기준선 | 확정분 |
|---|---|---|
| attention 호출 wall | 202 ms | 226 (DSP +40: Q 패스 30) |
| host 변환·복사(Q 20.7, 출력 29.1, Q·출력·o-proj 입력 staging 56) | 106 | 0 |
| mha_core accel 합 | 294 | 268 |
| arm staging memcpy 전체 | 192 | 142 |
| prefill p1024 | 2,514 / 2,254 | 2,201 / 2,188 |
| prefill p2048 | 4,466 / 4,530 | 4,194 / 4,527 |
결정성 0 불일치/146,800,640. 문장 md5 p1024 `1fd0625a`→`2ec61a41`, p2048 `bfc92564`→`aff26409`(둘 다 일관된
요약, near-tie 토큰에서 갈림). decode TPS 변화 없음(p1024 마지막 64: 12.1–14.3).

**WIP (handoff 브랜치 `78c2ad6a1`, PR 제외): qkv 패스에서 Q를 u16로, 정적 scale**
- `hvx_norm_rope_rows_ld_q16_f32` / `mm_u8i4_layer_norm_ld_q16`: norm+RoPE 행이 캐시에 있을 때 u16도 씀. 인코딩은
  head 공통 정적 상한 B = √(2·hd)·max|q gamma·q_scale| (norm 뒤 원소 ≤ √hd·|gamma|, RoPE 쌍은 그 √2배).
  mha_core는 `ComputeOps::attn_q16_enc`의 인코딩을 쓰고 Q 보정을 안 함. `QKVLayer attn_q16`(빌더가 q8+HTP일 때),
  `NNTR_HTP_ATTN_Q16=0`이면 동적 인코딩.
- 기기 실측: 상한/실제 범위 1.6–4.9배(Q ≥ 13비트 유지), 커널 k9 ≤ 2(창 −6..7). attention host 294→217 ms.
  prefill p1024 2,308/2,298 → 2,228/2,368, p2048 4,515/5,066 → 4,325/4,394.
- **정확도 미확정**이라 PR 제외. 문장이 바뀜(p2048 둘째 토큰 " l's a summary" 같은 어색한 run 있음).
  decode teacher-forced NLL(`NNTR_PPL_DECODE`)로는 못 가름: 수치가 같은 두 동적 경로(DSP/NEON 변환)끼리도 교차 nll
  초과가 +0.04~0.07인데 정적은 +0.03~0.09. 자기 경로 nll은 정적이 더 높음(p1024 0.419 vs 0.375, p2048 0.273 vs 0.195).
- **열린 질문: 프롬프트 NLL 게이트.** `NNTR_PPL`은 `TieWordEmbedding` 전용. untied `lm_head`에 같은 채점을
  붙여 봤더니(마지막 행 빼고 모든 prefill 행을 lm_head로) nll/token ≈ 13 (거의 uniform). 마지막 행은 맞다
  (argmax = 첫 생성 토큰 "thought"), 다른 행들은 결정적이고 경로에 따라 달라지지만 다음 토큰을 못 맞춘다.
  `skip_prefill` false, E2E 0/1 무관. 왜 마지막 행만 유효한지(어느 층이 prefill에서 마지막 행만 쓰는지)가
  밝혀져야 정적 scale을 판정할 수 있다. 그 코드는 커밋하지 않았다(채점 루프는 `tie_word_embedding.cpp`의 것을
  `lm_head.cpp` incremental_forwarding 끝에 그대로 옮기면 된다; 디버그 출력 `[PPL] diag …`로 행별 argmax 확인).

**다음 (B 이후 남은 CPU 연산)**: 첫 chunk의 K/V abs-max 보정(`calibrate_q2_scales`, NEON, 층당 0.4–0.8 ms)과
qkv 출력 Q f32의 host 복사(doc 60 §3.8). 정적 scale이 채택되면 Q f32 복사는 바로 뺄 수 있다.

## 8. 남은 prefill 최적화 (S26U p1024 ≈ 2.4 s 기준, 기대 효과는 기기 미측정)
| # | 항목 | 근거 |
|---|---|---|
| 7 | MoE 남은 대기 + HVX 8개 맞춤 (acc_tiles, DN ring, 2-bit expand) | drain_dn 3.5 ms/call, worker 40%만 사용 |
| 4 | attention 주변 (Q·출력 변환은 §7, KV 쓰기 28 ms, 버퍼 재할당 11 ms) | ATTN_TRACE |
| 5 | attention softmax 64 ms | ATTN_TRACE |
| 6′ | staging 복사 제거 (qkv 출력 1.0 GB 56 ms, o-proj residual/out 27 ms, attention out 28 ms) | PROFILE staging at |
| 9 | p4096을 chunk 1개(init 4096)로 | chunk마다 expert 재읽기 |
| 15 | 첫 토큰 KV seed(p4096 2.0 s)를 prefill 뒤로 숨기기 | t1 kv_seed_rpc_ms |
| — | router fp16 HMX (정확도 trade-off, 사용자 결정 필요) | logits 4.07 ms/call, IEEE sf 처리량이 한계 |
해 봤는데 효과 없음: staging memcpy를 4스레드로 (느려짐), router 행 블록 4→6, DCVS MAX corner(2112 MHz 그대로).

## 9. decode (prefill 다음)
p1024 C32 토큰당 ~70 ms. miss 대기가 최대. env 조합 측정 일부: A 기준 13.01 TPS(마지막 64) /
B(`NNTR_MOE_LRFU=32 NNTR_HTP_DENSE_EARLY=1`) 14.18. 남은 후보: C(+`NNTR_HTP_PREDICT=1`),
D(+`NNTR_MOE_MISS_READERS=8`), miss 감시 스레드 prime 코어 고정(notice 127 µs vs S25 1 µs),
작은 op 합치기(norm 3.3 ms), decode ATTN 10 ms, ROUTER 2.9 ms. FC/LM_HEAD/DENSE는 대역폭(~53 GB/s) 한계.

## 10. 커밋된 것 (htp/s26-prefill-opt, PR 4412 위)
| 커밋 | 내용 | 효과 (S26U) |
|---|---|---|
| a76483470 | chunked prefill (`init_seq_len` 단위) | p4096 동작 |
| bd3694efc | ids_history 범위 초과 수정 | p2048 종료 abort 해결 |
| 9f0185ffe | `NNTR_HTP_FC_IN_ARENA` | heap 880→130 MiB, p4096 동작 |
| 775ac4b1f | 양자화 KV cache 로드 시 등록 | p1024 −206 ms |
| c44306fa1 | MoE 2-bit down epilogue foreground | drain 13.5→4.4 ms, p2048 398→426 |
| 904cc7ff3 | router norm 커널 안으로 + VTCM | router 195→156 ms |
| 72f80829c | qkv post norm + RoPE 한 번에 | 148→97 ms |
| 7280511bb | router VTCM 복사를 HVX 파일로 (host 빌드 수정) | router 150.6 ms |
| 7d9490881 | FC 호출 u16 activation (quant 패스 안 역양자화) | §7 |
| f5a70c4d5 | attention f32 Q를 DSP에서 u16로, u16 context를 o-proj에 | p1024 2,514/2,254→2,201/2,188, p2048 4,466/4,530→4,194/4,527 |
| 66e023786 | NEON quant_u16_f32 contraction off | 두 경로 차이 125→8 step |
현재 p1024 ≈ 2.2 s (≈465 TPS), p2048 4.2–4.5 s (≈450–490 TPS), p4096 미측정(이 작업 후). decode p1024 마지막 64 ≈ 12–14 TPS(C32).

## 11. 커밋 규칙

형식 예시 (실제 `c44306fa1`):
```
[HTP] MoE kernel: 2-bit calls run the down epilogue on the foreground lane

<무엇이 문제였는지: 측정 숫자와 함께, 72자 줄바꿈, 영어>
<무엇을 바꿨는지: 함수/경로 이름>
<결과: 기기, 모델, 조건, before -> after, 출력 동일 여부, host check>

Co-authored-by: Claude <noreply@anthropic.com>
Signed-off-by: SeungHui Lee <shsh1004.lee@samsung.com>
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_...
```
명령:
```bash
git diff -U0 | clang-format-diff-14 -p1 -i          # 바뀐 줄만 포맷
git add <파일들>                                     # res/ 삭제분은 절대 add 하지 말 것
git -c user.name=Claude -c user.email=noreply@anthropic.com commit \
  --author="SeungHui Lee <shsh1004.lee@samsung.com>" -F msg.txt
git push origin <branch>
```
- 제목: `[HTP]`(htp_backend, test/htp), `[CausalLM]`(Applications/CausalLM), `[docs]`, `[Test]` 등 component.
  PR 4412 쪽 커밋들도 같은 형식이다 (`git log b398e078c~10..b398e078c`).
- 한 커밋 = 한 변경. 진단용 코드(trace FARF 등)는 커밋하지 않는다.
- WIP는 `htp/s26-prefill-handoff`에만, 확인된 최적화만 `htp/s26-prefill-opt`에 (cherry-pick).

### 11-1. 기존 커밋 규칙 요약
- 제목 `[HTP] ...` / `[CausalLM] ...`, 본문에 측정 근거. author `SeungHui Lee <shsh1004.lee@samsung.com>`,
  `git -c user.name=Claude -c user.email=noreply@anthropic.com commit --author=...`, trailer 순서:
  `Co-authored-by: Claude <noreply@anthropic.com>` / `Signed-off-by: SeungHui Lee <shsh1004.lee@samsung.com>` /
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- `git diff -U0 | clang-format-diff-14 -p1 -i` (바뀐 줄만). `subprojects/` 수정 금지.
- `Applications/CausalLM/res/gemma3|gemma4` 삭제(사용자 로컬 변경)는 커밋하지 않는다.
- 커밋 전 host 검사(§3.3) 통과 + 기기에서 출력 md5 확인. push는 `origin htp/s26-prefill-opt`.
- 한국어로 답한다. 측정 없이 결론 내지 않는다.
