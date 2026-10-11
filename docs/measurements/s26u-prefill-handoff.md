# S26 Ultra Gemma-4 2-bit prefill 최적화 — 인수인계 (2026-10-11 새벽 갱신)

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

- 브랜치 `htp/s26-prefill-opt`에 커밋 14개 (§10). base는 PR 4412 head (`b398e078c`). **push는 아직 안 됨**
  (이 워크스테이션에 GitHub 자격증명 없음: `gh auth login` 또는 SSH 키 뒤
  `git push origin htp/s26-prefill-opt htp/s26-static-q-wip && git push --force-with-lease origin htp/s26-prefill-handoff`).
- `htp/s26-prefill-handoff` = opt + 이 문서. `htp/s26-static-q-wip` = (옛 opt `66e023786` 위) 정적 Q scale WIP, 정확도 미확정(§7).
- 현재(S26U, C32): p1024 ≈ 2.2 s (≈ 460 TPS), p2048 ≈ 4.2–4.4 s (≈ 460–490 TPS), p4096 C24 9.2 s. decode p1024 마지막 64 ≈ 12–14 TPS.
- host·transport는 거의 바닥(staging 22–39 ms/prefill, 층당 RPC 6회). 남은 것은 DSP 연산: §8-1.

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

**WIP (브랜치 `htp/s26-static-q-wip`, PR 제외): qkv 패스에서 Q를 u16로, 정적 scale**
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

## 7-1. 블록 텐서를 DSP에 두기 (2026-10-11, opt 커밋 3개)

- `fb6e71850` pre-norm scratch를 M×K → 레인당 4행 (DSP heap 15–24 MB → 0.6 MB). **이게 없으면 p4096 C24가
  MoE 호출에서 `AEE_ENOMEMORY`** (u16 o-proj 엔트리의 M×K scratch가 heap 한계를 넘김).
- `e1e4ffca2` FC slice를 K ≥ 4096에서 1 MiB로: o-proj가 1024행 1회 호출(행 상한 896 → 1,472, K=8192는 704).
  K=2816까지 넓히면 qkv도 2048행 1회가 되지만 뒤의 MoE 호출이 ENOMEMORY (원인 미확인) → 안 함.
- `1148f40f5` 텐서를 통째로 staging하고 행 chunk는 offset으로 호출; `copyOutOf(to_host=false)`는 `sampledSame`이 보는
  33개 창(2 KB)만 host에 씀 → 다음 호출이 기존 내용 검사로 버퍼를 swap. `out_to_host` 속성(residual_add·dense_ffn·
  lfm2_moe·qkv_layer)을 Gemma-4 빌더가 "모든 소비자가 HTP 호출"일 때만 false로. mha_core는 `ComputeOps::host_view`로
  K/V/Q를 staging 버퍼에서 읽음. CPU fallback은 `sync_to_host` 먼저. decode 한 행·`PROFILE>=2`·마지막 층 출력은 전체 복사.

| | 그 전 opt | 지금 |
|---|---|---|
| staging memcpy p1024 | 143–151 ms (2,940 MB) | 22 ms (301 MB) |
| staging memcpy p2048 | 402 ms (7,444 MB) | 39 ms (408 MB) |
| o-proj 호출 수 p1024 / p2048 | 95 / 130 | 65 / 95 |
| prefill p1024 | 2,322 / 2,308 / 2,264 | 2,230 / 2,279 / 2,172 |
| prefill p2048 | 4,728 / 5,020 / 5,293 | 4,438 / 4,178 / 4,438 |
문장은 기준과 **바이트 동일**(p1024·p2048, `ATTN_F32=0`, `PROFILE=2`), 결정성 0 불일치, host check 통과.

**이 라운드에서 배운 것 (다음 세션 필독)**
- 정확도 게이트(프롬프트 NLL)가 없는 동안은 **비트 동일한 최적화만** opt에 넣는다. 검증 = 같은 env에서 기준 빌드와
  문장 md5 비교. 기기에 두 빌드를 `$D/pr4412_base`·`$D/pr4412_<tag>` 로 나란히 두고 번갈아 돌리면 재설치가 없다.
- 작은 이득(< 100 ms)은 벽시계(±150 ms)로 안 보인다: `NNTR_HTP_PROFILE=1`의 호출 수·host ms·staging MB가 1차 근거.
- host 복사를 건너뛰는 변경은 **행 chunk 호출**에서 깨진다(텐서 ≠ chunk 버퍼). 항목별 스위치로 기기에서 이분해 찾았다.
- `NNTR_MOE_SPLIT`은 기준 빌드에서도 깨진 문장을 낸다(기존 문제): fallback 검증에 못 쓴다.
- 분석해서 뺀 것: MoE 층 host "other" 169 ms는 거의 전부 router RPC(157 ms)였다. dense+router, attention+o-proj
  호출 융합은 RPC 60회 ≈ 20 ms라 보류.

## 7-2. DSP 쪽 라운드 (2026-10-11 오전, WIP 커밋 5개 — 수치 기입 뒤 opt로)

**먼저 읽을 것**: 워크스테이션이 12:08에 재부팅돼 `/tmp` scratchpad(로그·스크립트)가 전부 지워졌다. 스크립트는 이제
`~/workspace/s26u-work/`(build.sh, dev.sh, sum.sh)에 둔다. 재부팅 뒤 adb는 `no permissions`(USB ACL이 `nntrainer` 계정,
이 계정은 plugdev 아님) — 사용자에게 `sudo setfacl -m u:shsh1004-lee:rw /dev/bus/usb/003/*`를 부탁한다.

| 기기 폴더 | 내용 (누적) | p1024 | p2048 | p4096 C24 |
|---|---|---|---|---|
| `pr4412_b15` | 지난 라운드 끝 (opt tip `1148f40f5`) | 2,259 / 2,370 / 2,310 | 4,970 / 4,485 / 4,494 | 9,246 |
| `pr4412_b16` | `2f95e012c` MoE: 행을 한 번만 양자화 + down scatter 행 단위 | 1,969 / 1,962 / 2,023 | 3,969 / 4,024 / 4,109 | 8,563 |
| `pr4412_b17` | + `c792707e1` MoE pre norm을 scan에, 결과 제자리 누적 + `beec11b87` skel이 FC 행 chunk를 돎(qkv·o-proj RPC 1회) + `706564faf` KV seed를 staging으로 | 1,901 / 1,978 / 1,864 | 4,079 / 4,261 / 3,620 | 7,591 / 7,562 |
| `pr4412_b18` | + `2c71810a2` qkv의 per-head norm·RoPE를 dequant epilogue에 (행 단위) | 1,930 (1회) | 3,475 (1회) | 미확정 |

b15↔b16, b16↔b17은 번갈아 3쌍(2분 냉각). 문장은 전부 동일: 답 줄의 md5(`grep -a -o '<channel|>.*' log | head -1 | md5sum | cut -c1-8`)
p1024 `2ec61a41`, p2048 `aff26409`, p4096 `028868b0`, `NNTR_HTP_ATTN_F32=0` p1024 `1fd0625a`. decode 변화 없음(p1024 ≈13, p2048 ≈11 TPS).
**남은 확인**: b17↔b18 교대 측정(재부팅으로 끊김), p4096 반복(b17 한 번 18.9 s, b18 한 번 10.9 s — mem_available 1.6 GiB일 때; 코드 문제인지 메모리 압박인지 미확정),
작업 트리의 `stageClass` 1/4 옥타브(미측정, 미커밋).

**이 라운드에서 알게 된 것**
- MoE 커널의 병목은 HMX가 아니라 **HVX 워커의 메모리 접근**이었다. 계측(워커 종류별 시간): p2048 호출당 워커 226 ms 중 down scatter 128
  (tile 단위라 (행, tile)마다 DDR 미스), pack 63(행을 route된 슬롯 수만큼 f32에서 다시 양자화). 행 단위 scatter + l2fetch → 74, 한 번 양자화 + u8 gather → 27.
  out 접근을 빼면 16까지 내려가므로 남은 scatter 비용은 출력 23 MB를 8번 RMW하는 대역폭이다(구조적).
- `NNTR_HTP_PROFILE=2`는 **실제 경로가 아니다**: timed 엔트리 + host norm. 실제(`moe_layer_norm_add`)는 norm 패스 → 커널 안 act 복사 → …
  → out 복사 → post norm → add 였고, 그 중복 패스를 `c792707e1`이 없앴다. PROFILE=2의 `rest<=`는 대부분 probe 자체 비용(tile당 timer 2회).
- **FastRPC 드라이버는 인자로 넘긴 버퍼를 호출마다 통째로 매핑·sync한다**: 인자 길이가 아니라 버퍼 크기 MiB당 ≈27 µs. 그래서
  행 chunk 호출 2–4회가 그대로 2–4배였다(qkv 4.3 ms/call, o-proj 2.2) → skel이 chunk를 돌게 해 RPC 1회로(`beec11b87`).
  같은 이유로 2의 거듭제곱 class가 손해(23 MB 텐서가 32 MiB 버퍼) → 작업 트리의 1/4 옥타브 class.
- qkv DSP 시간의 46 %가 matmul 뒤의 norm+RoPE 패스(층당 5.9 of 12.7 ms)였다 → `2c71810a2`. PROFILE=1의 qkv host 506 → 339 ms(p2048).
- MoE 노드의 CPU 몫은 작다: top-k 0.2 ms/층, "other" 12.9 ms는 router RPC 10.7.
- attention 노드: kv_write 2.3 ms, calibration 2.4 ms(층마다 첫 호출), RPC 15–18 ms.

**해 봤는데 뺀 것**
- `RPCMEM_TRY_MAP_STATIC`(staging 버퍼 사전 매핑): p2048 3,139 ms(652 TPS)까지 나오지만 **문장이 깨진다**. host에서 읽기 전 `DC CIVAC`를
  넣어도 그대로 — DSP가 쓴 내용이 host에 안 보인다. host가 읽는 버퍼만 non-static으로 두면 p2048은 맞지만 p1024는 깨지고 p4096은
  `layer_norm` 실패(0x80000583류). 캐시 일관성을 문서로 보장받기 전에는 쓰지 말 것. 이득은 p2048 RPC 합 −285 ms.
- uncached staging(`flags=0`): 호출당 3–8 % 수준, 벽시계로 구분 안 됨. futex wake 생략: 효과 없음.
- `NNTR_HTP_KV_SEED_Q=1`: 문장 깨짐(기존). KV seed를 ION으로 보낸 것은 1,055–1,097 → 973 ms: 남은 것은 DSP의 scalar append
  (`hvx_attn_m1_f32.c` `q8_of`, 층당 ≈30 ms).

## 8-1. 지금 시간이 어디 가는지 (2026-10-11, pr4412_b15, 같은 실행에서 노드·RPC 동시 측정)

p2048 4,604 ms(프로파일 켠 실행; 평소 4,180–4,440) 기준. 노드 시간은 `NNTR_LAYER_PROFILE=1`, RPC host 시간은 `NNTR_HTP_PROFILE=1`, 같은 실행.

| 노드 | p2048 ms | 안의 RPC host | 노드 − RPC (CPU 몫) |
|---|---|---|---|
| sparse_moe | 2,313 | MoE 1,828 + router 328 | 157 (top-k·배정표·호출 준비) |
| attention | 724 | ≈450 (wall 15–20 ms/층 × 30) | ≈270 (CPU KV 쓰기 + Q view + transport) |
| qkv | 643 | 484 + 117 | 42 |
| ffn (dense) | 459 | 461 | 0 |
| post_attention_norm (o-proj) | 372 | 295 + 87 | 0 |
| staging memcpy | 44 | — | — |

p1024(2,321 ms): moe 1,234(RPC 976+164 → CPU 94) / attention 339 / qkv 282(267) / ffn 254(256) / o-proj 163(173).

**MoE 호출 하나의 DSP 분해 (PROFILE=2, ms/call)** — p1024 → p2048. M에 비례해 커지는 것은 weight DMA가 아니다.

| 구간 | p1024 | p2048 |
|---|---|---|
| **타이머 밖** (`rest<=`) | **9.2** | **16.5** |
| acc read (HMX 완료 대기 포함; HexKL micro는 int32 누산기 1개) | 5.3 | 8.5 |
| drain_dn (down 2-bit 확장 job 대기) | 3.75 | 7.5 |
| drain (gate_up) | 0.93 | 0.95 |
| mm (HMX issue) | 3.65 | 5.9 |
| stage (act 복사 + out memset + out 복사) | 1.25 | 2.5 |
| scatter / quant / dequant | 0.9 / 0.64 / 0.57 | 2.1 / 1.06 / 1.05 |
| alloc (scratch 재확보) | 0.19 | 1.8 |
| DSP 합계 / host / transport(host−dsp) | 26.75 / 28.1 / 1.3 | 48.3 / 51.6 / **3.2** |
| 64행 블록 수 → 채움률 | 198 → 65 % | 320 → 80 % |

dense 호출(p2048) 10.2 ms: mm 1.6, stage 2.2, acc 2.0, requant 1.5, dequant 1.06, quant 0.76, scatter 0.39; transport 1.8.
attention 호출(p2048 sliding 층, window 1024) DSP 12.3 ms: append(KV quant 1.96 + bake 1.8) 3.8, q16 1.45, kernel 8.5(qk 3.2, **softmax 5.5**, pv 1.4, wait 1.0); wall 15.1 → transport 2.8. full 층(hd 512) 15–20 ms. 0층·5층 노드는 +25/+42 ms(첫 호출 calibration·할당).
t1(첫 토큰) KV seed: p2048 1,055 ms, p4096 2,137 ms — 30층 K/V f32(1 GB)를 non-ION `std::vector`로 보낸다. `NNTR_HTP_KV_SEED_Q=1`은 773 ms지만 **문장이 깨진다**(p2048 빈 출력, p1024 반복) — 쓰지 말 것.

## 8. 남은 prefill 최적화 후보 (2026-10-11 재분석; 기대 효과는 모두 기기 미측정, p2048 ≈ 4,400 ms 기준)

비트 동일 조건을 만족하는 것만 opt에 들어간다(§7-1). 순서 = 기대 효과 × 확실성 ÷ 노력.
**2026-10-11 오전 갱신**: 아래 1·2·10(MoE)·11의 qkv 절반·14는 §7-2에서 처리됨. 3은 §7-2의 이유로 보류. 남은 순서: (a) b18·staging class 기기 확정 → 커밋 메시지에 수치 → opt cherry-pick, (b) KV seed의 DSP append 벡터화(TTFT), (c) attention softmax 5.5 ms/call, (d) o-proj의 post 패스 2.5 ms/층, (e) router.

| # | 항목 | 근거 (p2048) | 기대 | 첫 걸음 |
|---|---|---|---|---|
| 1 | **MoE 타이머 밖 16.5 ms/call** | rest 500 ms(11 %), M에 비례 | 모름(가장 큰 미설명 덩어리) | fg 스레드 타임라인 프로브(블록 루프 밖·pool submit/wait·pack 대기) |
| 2 | **MoE drain_dn 7.5 ms/call** | 225 ms(5 %), M에 비례 → DMA가 아니라 **background lane FIFO에서 2-bit 확장 job이 pack 유닛 뒤에 선다** | −150~−225 | 확장 전용 lane 또는 pack을 fg worker로; 겸해서 "행을 한 번만 양자화 + u8 gather"(pack 바이트 4분의 1, 비트 동일) |
| 3 | **FastRPC 캐시 유지보수 (transport)** | staging pool은 cached ION(`HTP_RPC_FLAGS_DEFAULT`) → 호출마다 in/out 46 MB flush/invalidate: MoE 3.2 + dense 1.8 ms/call = 155 ms; attention wall−dsp 2.8 ms/call = 84 ms; qkv·o-proj·router 미계측 | −250~−400 | pool을 weight arena처럼 `fastrpc_mmap` 1회 매핑·uncached로, host가 읽는 창(sampled 2 KB, K/V view)만 DMA_BUF sync. env 플래그로 한 shape 먼저 |
| 4 | **t1 KV seed 1.06 s (p4096 2.1 s)** — TTFT | f32 K/V 1 GB를 non-ION vector로 전송 | TTFT −0.8~−1.0 s(p2048), −2 s(p4096); prefill TPS 무관 | (a) seed 행을 ION staging으로(비트 동일, 반나절) (b) prefill attention 호출 안에서 DSP가 가진 f32 K/V로 decode 캐시를 바로 채우기(CPU 캐시는 UINT16 저장이라 f32→f16→f32 RNE를 똑같이 — `compute_fp32_to_fp16`과 비트 비교 필요); (b)면 #6도 같이 사라짐 |
| 5 | **attention softmax 5.5 ms/call + append 3.8** | softmax 168 ms(4 %; 25M 점수를 4.6 G/s), KV quant+bake 114 ms | −150~−250 | softmax 커널 pcycle 프로파일(이론 5× 여유); KV int8 타일을 qkv 호출 epilogue에서 바로 내기(같은 양자화기 → 비트 동일) |
| 6 | **CPU KV 캐시 쓰기** | attention 노드 host 몫 ≈ 5 ms/층 = 150 ms(3.5 %); t1 seed와 NEON fallback만 읽음 | −150 | #4(b) 뒤에 `NNTR_HTP_KV_DROP_CPU`에서 쓰기 자체를 생략 |
| 7 | **MoE 노드 CPU 157 ms** | router softmax·top-k·배정표가 CPU | −100 | `buildExpertAssignments` 스레딩 확인; router RPC가 top-k 결과(2048×8)만 돌려주기 — 선택 규칙이 CPU와 비트 동일한지(`m1_ops_det.h`) md5로 확인 |
| 8 | **router 328 ms** | 138 GFLOP/s ≈ HVX f32 추정 피크의 26 %; 이론 ~3 ms vs 10.9 ms/call | −100~−150 | 커널 stall 프로파일(행당 scalar splat 의심). dense 호출에 넣어 숨기기는 dense가 이미 HVX 80 %라 RPC 1회·23 MB 읽기 1회(≈60 ms)만 남음. fp16 HMX router = 정확도 변경, 사용자 결정 |
| 9 | **dense 461 ms** | mm 1.6 of 10.2; 3조각(704×3) pseudo-expert로 돌아 out에 3번 scatter RMW(23 MB씩) + memset | −60~−100 | 첫 조각은 memset 없이 바로 쓰기; stage 복사(#10)와 함께 |
| 10 | **MoE stage 2.5 + alloc 1.8 ms/call** | 78 + 56 ms(3 %); alloc은 p1024 0.19 → p2048 1.8: `n_slots_cap` 상한이 호출마다 넘어 재확보 | −80~−130 | reserve를 M·top_k 최대로 1회; act_f32를 scan이 ION에서 직접 읽고 out을 바로 쓰는지 검토(`moe_dma_copy` 이유 확인 먼저) |
| 11 | **FC(qkv/o-proj) HMX 효율** | qkv 2.5, o-proj 2.1 TMAC/s vs MoE ≈ 4.4; qkv f32 출력 67 MB/층을 DDR에 쓰고 attention이 다시 읽음 | −150~−300 | PROFILE=2에서 `_timed` FC 엔트리 쓰게 해 분해부터; Q u16(enc는 calibration 뒤 고정)·K/V int8을 qkv epilogue에서 바로 내기 |
| 12 | p4096 chunk 1개 | expert 스트리밍·확장 2회, seed 2.1 s | −150~−300 of 9,246 | config `init_seq_len` 4096 + DSP scratch 확인 |
| 13 | 첫 층 calibration·할당 | 0층 +25, 5층 +42 ms | −65 | prefill 사이에 재사용 |
| 14 | RPC 횟수(6/층) | 고정비 ≈ 0.2 ms × 180 = 40 ms | −20 | #3 뒤에만 의미 |
| — | MoE acc read 8.5 ms/call | HexKL micro int32 누산기 1개 → tile마다 issue→read 직렬; down은 read당 MAC 22개 | 라이브러리 한계 | 보류 |
| — | HMX 64행 패딩(p1024 35 %, p2048 20 %) | HVX tail은 패딩된 HMX 블록보다 느림 | 구조적 | 보류 |

해 봤는데 효과 없음(이전): staging memcpy 4스레드, router 행 블록 4→6, DCVS MAX corner. 쓰면 안 됨: `NNTR_HTP_KV_SEED_Q=1`(문장 깨짐), `NNTR_MOE_SPLIT`(기준에서도 깨짐).
전부 들어가면 p2048 ≈ 1.0–1.5 s 감소(≈ 570–700 TPS)가 상한 추정이고, 절반이 현실적이다 — 기기 미측정.

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
| fb6e71850 | pre-norm scratch 레인당 4행 | p4096 C24 ENOMEMORY 해결 |
| e1e4ffca2 | FC slice 1 MiB (K≥4096), o-proj 1회 호출 | o-proj 호출 95→65 (p1024) |
| 1148f40f5 | 블록 텐서를 DSP에 (통째 staging, 표본 copy-out) | staging 143→22 ms (p1024), 402→39 ms (p2048) |
현재 p1024 ≈ 2.2 s (≈460 TPS), p2048 4.2–4.4 s (≈460–490 TPS), p4096 C24 9.2 s. decode p1024 마지막 64 ≈ 12–14 TPS(C32).

## 11. 커밋 규칙

형식 예시 (실제 `c44306fa1`):
```
[HTP] MoE kernel: 2-bit calls run the down epilogue on the foreground lane

<무엇이 문제였는지: 측정 숫자와 함께, 72자 줄바꿈, 영어>
<무엇을 바꿨는지: 함수/경로 이름>
<결과: 기기, 모델, 조건, before -> after, 출력 동일 여부, host check>

Co-authored-by: Claude <noreply@anthropic.com>
Signed-off-by: SeungHui Lee <shsh1004.lee@samsung.com>
Co-Authored-By: <세션 attribution 줄> 
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
