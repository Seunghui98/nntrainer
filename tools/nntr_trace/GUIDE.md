# nntr trace 사용 가이드

nntrainer를 폰에서 돌리는 동안 CPU와 Hexagon HTP(HMX / HVX / DMA)가 각각
언제 무엇을 했는지를 한 타임라인으로 보는 도구입니다. 결과 파일은 Chrome
Trace Event Format(`trace.json`)이라 이 디렉터리의 `viewer.html`,
[ui.perfetto.dev](https://ui.perfetto.dev), `chrome://tracing` 어디서나 열립니다.
웹서버는 필요 없습니다.

이 문서 하나로 "처음 보는 사람이 기기 없이 체험 → 자기 폰에서 측정 → 결과
읽기 → 남에게 공유"까지 할 수 있게 쓰는 것이 목표입니다. 설계 배경은
[`docs/backend_guide/HTP_TRACE_PROFILER.md`](../../docs/backend_guide/HTP_TRACE_PROFILER.md),
뷰어 기능 로드맵은 [`VIEWER_PLAN.md`](VIEWER_PLAN.md)에 있습니다.

---

## 1. 브랜치와 구성 요소

작업은 두 브랜치에 나뉘어 있습니다. 보는 쪽(도구)은 어떤 빌드와도 상관없고,
측정하는 쪽(런타임 레코더)은 HTP 커널이 있는 브랜치 위에 올라가 있기 때문입니다.

| 브랜치 (Seunghui98/nntrainer) | 기반 | 들어 있는 것 | 언제 필요한가 |
|---|---|---|---|
| `claude/gallant-bell-7mnznj` | `nntrainer/nntrainer:main` | 뷰어, 변환기, `summarize.py`, 샘플·테스트, 설계 문서, 이 가이드 | 결과를 볼 때 (항상) |
| `claude/gallant-bell-7mnznj-htp-trace` | `claude/lfm2-moe-ffn-hexkl-2ivn5v` | 런타임 레코더 `HtpTrace` (`NNTR_TRACE`) | 폰에서 호출별 타임라인을 뽑을 때 |

두 브랜치를 한 체크아웃에서 같이 쓰는 방법:

```bash
git fetch origin claude/gallant-bell-7mnznj claude/gallant-bell-7mnznj-htp-trace
git checkout claude/gallant-bell-7mnznj-htp-trace                      # 빌드할 쪽
git worktree add ../nntr-trace-tools origin/claude/gallant-bell-7mnznj # 보는 쪽
# 이후 도구 경로: ../nntr-trace-tools/tools/nntr_trace/
```

### 파일 지도

| 파일 | 하는 일 |
|---|---|
| `viewer.html` | 타임라인 뷰어. 파일 하나, 빌드·서버·외부 스크립트 없음 |
| `bundle.py` | `trace.json`을 뷰어에 구워 넣어 HTML 한 파일로 만듦 (공유용) |
| `summarize.py` | 뷰어와 같은 지표를 CLI로 계산, `--fail-if` 게이트 |
| `htp_profile_to_trace.py` | `NNTR_HTP_PROFILE=2` 요약 출력 → trace (형태별 대표 호출 1개) |
| `stage_us_to_trace.py` | `unittest_hvx_fc` / `unittest_hvx_attn`의 `FC_STAGE` / `ATTN_STAGE` 로그 → trace |
| `qnn_optrace_to_nntr.py` | QNN HTP optrace(`*_chromeTrace_opTrace.json`) → trace (pid 3) |
| `make_sample_trace.py` | 합성 샘플 trace (문서에 기록된 실측값을 스케일) |
| `test/run.sh` | 샘플 생성, 번들, headless Chromium 검사, python 테스트 |
| (HTP 브랜치) `nntrainer/tensor/htp_backend/htp_trace.{h,cpp}` | 런타임 레코더 |

---

## 2. 5분 체험 (기기 없이)

```bash
cd tools/nntr_trace
python3 make_sample_trace.py -o /tmp/sample.json
python3 bundle.py -o /tmp/demo.html --trace "sample=/tmp/sample.json"
# /tmp/demo.html 을 브라우저로 더블클릭
```

`viewer.html`을 직접 열고 **Load**로 아무 `trace.json`이나 골라도 됩니다.
브라우저는 `file://` 페이지가 옆 파일을 마음대로 읽게 두지 않으므로, 파일은
Load로 고르거나 `bundle.py`로 구워 넣어야 합니다.

`test/data/htp_profile_sample.log`는 LFM2.5-8B-A1B를 S25U에서 실제로 돌린
`[HTP-PROFILE]` 출력입니다. 진짜 숫자로 보고 싶으면:

```bash
python3 htp_profile_to_trace.py test/data/htp_profile_sample.log -o /tmp/lfm2.json
python3 bundle.py -o /tmp/lfm2.html --trace "LFM2 8B (S25U)=/tmp/lfm2.json"
```

---

## 3. 폰에서 측정하기

### 3.0 준비물

`docs/htp_attention/mobile_e2e_run_guide.md` §1과 같습니다.

- Hexagon SDK 6.4.0.2 (`HEXAGON_SDK_ROOT`), Android NDK r26d (`ANDROID_NDK`)
- `hexkl_addon`: `--htp` 빌드에는 SDK 번들형(`$HEXAGON_SDK_ROOT/addons/hexkl_addon`),
  DSP skel(`test/htp/build.sh`)에는 독립 베타 드롭. 둘을 바꿔 넣으면 조용히 실패합니다.
- HTP가 켜진 빌드가 이미 한 번 돌아간 폰 (검증 기기: Galaxy S25 Ultra / V79)

### 3.1 경로 A: 재빌드 없이 (형태별 평균)

HTP 빌드라면 이미 들어 있는 기능입니다. 타임라인은 없고, 호출 형태
(K, N, decode/prefill)마다 stage 평균을 대표 호출 하나로 그립니다.

```bash
adb shell "cd /data/local/tmp/nntrainer/causallm && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  NNTR_NUM_THREADS=8 NNTR_HTP_PROFILE=2 ./nntrainer_causallm ./models/<model>" 2>&1 | tee run.log

python3 tools/nntr_trace/htp_profile_to_trace.py run.log -o profile.json
python3 tools/nntr_trace/bundle.py -o report.html --trace "run=profile.json"
```

### 3.2 경로 B: 호출별 타임라인 (`NNTR_TRACE`)

**빌드.** `claude/gallant-bell-7mnznj-htp-trace`를 체크아웃하고:

```bash
cd Applications/CausalLM
./build_android.sh --htp          # builddir를 새로 만들고 -Denable-htp=true를 넘김
./install_android.sh --model=<model_name>
```

- 이 브랜치를 처음 빌드할 때는 `--cache`를 붙이지 마세요. `--cache`는
  nntrainer 재빌드를 건너뛰어 레코더가 바이너리에 들어가지 않습니다.
- 확인 (저장소 루트에서):
  `readelf -d builddir/android_build_result/lib/arm64-v8a/libnntrainer.so | grep -E "libsdkl|libcdsprpc"`
  둘 다 보여야 HTP가 켜진 빌드입니다.
- **DSP skel은 다시 빌드할 필요 없습니다.** 이 브랜치는 ARM 쪽만 바꾸고 IDL은
  그대로입니다. 폰에 있는 `libnntr_hvx_skel.so`가 기반 브랜치와 맞으면 됩니다.
- `builddir/jni/arm64-v8a/libcdsprpc.so`는 링크용 스텁입니다. 폰에 push하면
  안 됩니다 (run guide §5).

**실행.** 트레이스 실행은 생성 토큰을 줄이세요. 모델 디렉터리의
`nntr_config.json`에서 `"num_to_generate": 32` 정도면 충분합니다.
512토큰 전체를 뜨면 LFM2 기준 약 27 MB, 13만 이벤트라 뷰어가 버거워집니다
(토큰당 약 22회 FastRPC, 호출당 약 2.4 KB).

```bash
adb shell "cd /data/local/tmp/nntrainer/causallm && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  NNTR_NUM_THREADS=8 NNTR_TRACE=/data/local/tmp/trace.json \
  ./nntrainer_causallm ./models/<model>"
# 끝에 이 줄이 나와야 성공:
#   [HTP-TRACE] wrote /data/local/tmp/trace.json: N calls, M spans

adb pull /data/local/tmp/trace.json
python3 tools/nntr_trace/bundle.py -o run.html --trace "<model> $(date +%m%d)=trace.json"
```

- `NNTR_TRACE`를 주면 profile level 2가 자동으로 켜집니다. 기존
  `[HTP-PROFILE]` 요약도 그대로 같이 나옵니다.
- 파일은 프로세스가 정상 종료할 때 한 번 써집니다. 중간에 죽거나 Ctrl-C로
  끊으면 파일이 없습니다.
- 같은 실행에서 A의 요약과 B의 trace가 모두 나오므로, 두 결과를 뷰어의
  **Compare**로 나란히 볼 수 있습니다.

### 3.3 다른 입력

```bash
# HTP 디바이스 테스트 로그 (FC_STAGE / ATTN_STAGE 마커)
python3 tools/nntr_trace/stage_us_to_trace.py /tmp/hvx_fc_device_run.log -o fc.json

# QNN HTP optrace. 실제 파일로 필드명을 아직 검증하지 못했습니다(스크립트 docstring 참고)
python3 tools/nntr_trace/qnn_optrace_to_nntr.py qnn_opTrace.json -o qnn.json --ts-unit cycles
```

---

## 4. 뷰어 사용법

### 조작

| 하고 싶은 것 | 방법 |
|---|---|
| 확대 / 축소 | 휠, 또는 `W` / `S` |
| 좌우 이동 | 드래그, 또는 `A` / `D` |
| 전체 보기 | `0`, 또는 빈 곳 더블클릭 |
| 슬라이스 정보 | 클릭 (아래 Selection 탭) |
| 구간 지정 | 눈금자에서 드래그, 또는 Shift+드래그. 선택한 슬라이스 구간은 `m`. 해제는 `Esc` |
| 찾기 | 상단 Find 입력 후 `Enter` / `Shift+Enter`로 이동, `f`로 그 슬라이스에 맞춤 |
| 프로세스·스레드 접기 | 프로세스 헤더 클릭, 스레드 이름 더블클릭 |
| 두 실행 비교 | `Compare:`에서 두 번째 trace 선택 또는 파일, `View:`로 A / B / A+B |
| 색 기준 | `Color:` op / name / engine |
| 링크로 상태 공유 | 줌·선택·탭·필터가 URL `#` 뒤에 저장됨 (같은 HTML 파일에서만 유효) |

### 화면 구성

- **각 프로세스 맨 위 `units busy` 띠**: 그 순간 바쁜 레인 수(HMX, HVX, DMA).
  옅으면 한 레인만 일하는 직렬 구간입니다.
- **맨 위 개요 띠**: 전체 실행의 밀도와 현재 보는 구간.
- **노란 경고 배너**: CPU로 떨어진 op(빗금 표시), 링버퍼 누락, 클럭 위반,
  HTP 비활성. 배너가 있으면 숫자를 믿기 전에 먼저 확인하세요.

### 아래 분석 탭이 답하는 질문

| 탭 | 답하는 질문 |
|---|---|
| Selection | 이 슬라이스가 무엇이고 얼마나 걸렸나, cy/elem, TOPS, 인자 |
| Wall time | 시간이 CPU / FastRPC / HMX만 / HMX∥HVX / HVX만 / DMA / DSP idle 중 어디로 갔나 |
| Engines | 각 레인이 몇 % 바빴나, HVX 풀 꼬리(가장 느린 unit ÷ 평균) |
| Kernels | 어떤 커널이 시간을 가장 많이 쓰나, cy/elem이 QNN 기준치와 비교해 이상한가 |
| Layers | 레이어별 분해 (레이어 span이 있는 trace에서만) |
| Transport | FastRPC 고정비용과 MB당 비용, 그 직선에서 벗어난 호출 |
| Tokens | 토큰마다 시간이 어떻게 늘어나나, TTFT, 토큰당 호출 수 |
| Compare | 두 실행의 버킷·커널·레이어 차이 |

---

## 5. 무엇이 실측이고 무엇이 아닌가

같은 뷰어로 보지만 입력마다 믿을 수 있는 범위가 다릅니다.

| 입력 | 호출 시각·순서 | 각 stage 길이 | 호출 안 stage 위치 | HMX∥HVX 겹침 | cy/elem |
|---|---|---|---|---|---|
| `NNTR_TRACE` (경로 B) | 실측 | 실측 (DSP `stage_us`) | 순서대로 늘어놓음 | MoE·dense·conv 호출의 SwiGLU만 실측, 나머지 0 | 1200 MHz 가정 |
| `NNTR_HTP_PROFILE=2` 요약 (경로 A) | 없음 (형태별 대표 1개) | 수천 호출 평균 | 순서대로 늘어놓음 | 0 | 1200 MHz 가정 |
| `FC_STAGE` / `ATTN_STAGE` 로그 | 없음 | 실측 총합 | 순서대로 늘어놓음 | 0 | 1200 MHz 가정 |
| QNN optrace | 실측 | 실측 | 실측 | 실측 | 실측 사이클 |
| `make_sample_trace.py` | 합성 | 문서 실측값 스케일 | 합성 | 합성 | 합성 |

- "순서대로 늘어놓음"은 DSP가 stage별 **총합**만 돌려주기 때문입니다. 호출 안에서
  각 stage가 정확히 언제 돌았는지는 DSP 쪽 링버퍼(설계 문서 P3)가 생겨야 압니다.
- MoE 레이어 커널의 SWIGLU 칸은 HMX가 도는 동안 HVX 워커가 처리한 시간이라
  (`HtpProfile`의 `swiglu_hidden`), `HVX pool (under HMX)` 레인에 HMX 구간과 겹쳐
  그립니다. 워커 합산값이 HMX 구간보다 길면 구간 길이로 자르고, 원래 값은
  args의 `worker_us`에 남깁니다.
- cy/elem은 `args.cycles`가 없으면 `dur × 1200 MHz`로 추정합니다. 절대값보다
  같은 커널끼리, 같은 trace 안에서 비교하는 용도로 쓰세요.
- `hmx_peak_tops`는 기기에서 재기 전까지 비워 두었습니다. 그래서 "% of HMX
  peak"는 표시되지 않습니다.

---

## 6. `summarize.py`: 숫자를 CLI·CI에서

```bash
python3 tools/nntr_trace/summarize.py trace.json -o metrics.json
python3 tools/nntr_trace/summarize.py trace.json --range phase:1     # 두 번째 phase만
python3 tools/nntr_trace/summarize.py trace.json \
  --fail-if 'compression<1.5' --fail-if 'idle_ratio>0.05'            # 넘으면 exit 1
```

출력 스키마는 `nntr_trace.metrics.v1`이고, 뷰어의 Wall time 탭 **Copy metrics
JSON**과 같은 숫자입니다(테스트가 1e-6 상대오차로 대조합니다).

---

## 7. 다른 사람과 공유하기

| 방법 | 받는 사람이 할 일 | 언제 |
|---|---|---|
| `bundle.py`로 만든 HTML 한 파일 | 더블클릭 | 기본. 오프라인, 설치 없음 |
| `trace.json` 그대로 | ui.perfetto.dev에 끌어다 놓기 | Perfetto에 익숙한 사람 |
| HTML 안 URL `#...` | 같은 HTML을 연 상태에서 링크 열기 | 특정 구간을 짚어 줄 때 |

- 여러 실행을 한 파일에 넣으려면 `--trace`를 여러 번 주면 됩니다. 받는 사람이
  상단 선택상자로 바꿔 보고 **Compare**로 비교할 수 있습니다.
- `report.html`의 크기는 trace 크기와 거의 같습니다. 메일 첨부 한도가 걱정되면
  `num_to_generate`를 줄여서 다시 뜨세요.

---

## 8. 문제 해결

| 증상 | 원인과 조치 |
|---|---|
| `[HTP-TRACE] wrote ...` 줄이 없다 | `NNTR_TRACE`가 전달되지 않았거나, 프로세스가 정상 종료하지 않았거나, 레코더 없는 바이너리(`--cache` 빌드 등) |
| `[HTP-TRACE] cannot open ...` | 경로에 쓸 수 없음. `/data/local/tmp/` 아래로 |
| HTP 레인이 비어 있고 CPU만 있다 | `readelf`로 `libsdkl.so` 확인. 스텁 `libcdsprpc.so`를 push하지 않았는지 확인 |
| `AEE_EBADPARM` 예외 | 폰의 skel이 바이너리와 안 맞음. `test/htp/build.sh`로 skel 재빌드 후 push (run guide §5) |
| wait span에 `note: untimed entry` | timed 엔트리가 없는 호출 경로. DSP 분해 없이 호스트 시간만 있음 |
| FC 호출에서 HMX∥HVX가 0 | 정상. §5 표 참고 |
| 뷰어가 느리다 | 이벤트가 10만 개를 넘음. `num_to_generate`를 줄이거나 Range로 구간을 좁혀서 보기 |
| 노란 배너에 fallback | 그 op은 HTP가 아니라 CPU에서 돌았음. 성능 숫자 해석 전에 원인부터 |

---

## 9. 개발자 메모

- 바꾼 뒤에는 `bash tools/nntr_trace/test/run.sh`. 샘플 생성, 번들, headless
  Chromium 검사(`test/check.js`), python 테스트 5개가 돕니다. Chromium과
  playwright가 필요합니다(`CHROMIUM=...`, `NODE_PATH=...`로 지정 가능).
- 새 FastRPC 엔트리에 `stage_us`가 생기면 `htp_trace.cpp`의 `kLayouts`에 행을
  하나 추가합니다(슬롯 번호 → 레인, 이름, 분류). 슬롯 enum은
  `htp_compute_ops.cpp`의 `HTP_*_T_*`와 같아야 합니다.
- 레코더는 `HtpProfile`의 `addInvoke*` / `addStaging` / `addRegister`에서
  호출됩니다. 새 호출 경로가 이 함수들을 지나면 트레이스에도 자동으로 들어갑니다.
- 다음 단계는 설계 문서의 P1(호스트 레이어·스레드 span, `trace_sink`)과
  P3(DSP 링버퍼로 진짜 레인 겹침)입니다.
