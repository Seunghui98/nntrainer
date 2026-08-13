<!-- SPDX-License-Identifier: Apache-2.0 -->

# 44 -- u8 RoPE 커널 수정 검증: 실제 기기 e2e 가이드

`43_rope_u8_fix_task.md`의 수정 사항(`05a87c4`, `71f25aa`, `63e2b91`,
`55dfb94`, 브랜치 `claude/hvx-rope-calculation-4fd8gr`)을 Galaxy S25 Ultra
같은 실제 기기에서 검증하는 절차입니다. 호스트 빌드/테스트는 이미 이 세션에서
돌려 확인했습니다 (§0). 여기서부터는 Hexagon SDK와 물리 기기가 있어야 하는
부분입니다.

---

## 0. 이미 확인된 것 -- 다시 할 필요 없음

이 세션에서 `meson build -Denable-transformer=true` (blas/tflite는 이 샌드박스
환경 제약으로 꺼서 구성 — 실제 개발 머신에서는 필요 없는 우회입니다)로
`unittest_mha_htp_host_model`을 빌드하고 돌렸습니다:

```
[==========] 13 tests from 2 test suites ran. (3665 ms total)
[  PASSED  ] 13 tests.
```

`RopeU8*` 6개 테스트 전부 포함, 그리고 그 과정에서 **이번 수정과 무관한
기존 버그 2개**를 실제로 찾아 고쳤습니다 (`63e2b91`) -- 이 테스트가 이 브랜치
역사상 한 번도 실행된 적이 없었다는 뜻이고, `00_START_HERE.md` §5가 경고한
바로 그 패턴입니다.

**당신의 개발 머신에서 확인용으로 다시 돌리려면:**

```bash
cd <repo>
git fetch origin claude/hvx-rope-calculation-4fd8gr
git checkout claude/hvx-rope-calculation-4fd8gr
git submodule sync && git submodule update --init --depth 1
meson build -Denable-transformer=true
ninja -C build
cd build && meson test unittest_mha_htp_host_model --print-errorlogs
```

`build`를 쓰세요 (체크인된 `builddir`는 안드로이드 크로스 빌드용).

---

## 1. 환경 (한 번)

```bash
export HEXAGON_SDK_ROOT=~/workspace/Hexagon_SDK/6.4.0.2
export DEFAULT_HEXAGON_TOOLS_ROOT=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.04
export HEXKL_ROOT=~/workspace/hxkl-beta2/hexkl_addon   # beta2! addons/ 밑은 beta1
export HEXKL_SDK_VER=6.4.0.2
export ANDROID_NDK=~/workspace/android-ndk-r26d
```

`setup_sdk_env.source`가 "missed components"로 실패하면 고치지 말고 위
두 변수를 손으로 넣으세요 (`13_htp_pr_plan.md` §3a에 기록된 함정).

기기 확인:

```bash
adb devices   # 디바이스 시리얼이 나와야 함, 예: R3CY10WM83Y
```

---

## 2. DSP skel 빌드

```bash
cd <repo>/test/htp
bash build.sh
```

**확인할 것:**

- `-Werror`에서 **warning 0** — 특히 `hvx_rope_u8.c`의 stride 파라미터
  추가와 `nntr_hvx_rope.c`의 새 시그니처가 타입 불일치 warning 없이
  깨끗해야 합니다.
- 출력 경로: `test/htp/build/libnntr_hvx_skel.so` (v79, hexkl 6.4.0.2)

**qaic가 IDL을 거부하면:** `unexpected "o" / expecting "in", "rout" or
"inrout"`는 스칼라 out에 `out`을 쓴 경우입니다. 이번 IDL 변경
(`rope_cache_init`의 새 시그니처)은 전부 `rout uint32 generation` 하나뿐이라
이미 맞게 되어 있지만, 만약 이 오류가 나면 그 지점을 먼저 보세요.

이 단계가 사실상 **가장 중요한 검증**입니다: `Q6_Wuh_vunpack_Vub`가 실제
SDK 헤더에 그 이름과 시그니처로 존재하는지, `hvx_rope_u8.c`의 나머지
인트린식들이 이 SDK 버전에서 문제없이 컴파일되는지는 이 단계에서만
확인됩니다. 이 문서를 쓴 세션은 Hexagon 컴파일러가 없어서 이 부분을
**한 번도 실행하지 못했습니다.**

---

## 3. ARM 쪽 빌드

```bash
cd <repo>
ANDROID_NDK=$ANDROID_NDK PATH=$ANDROID_NDK:$PATH \
  ./tools/package_android.sh --arm-arch=armv8.2-a -Dwerror=false
# -> builddir/jni/arm64-v8a/libnntrainer.so

ln -sfn "$PWD/subprojects/googletest/googletest" test/jni/googletest   # 아무도 안 만들어 줌

cd test/jni
"$ANDROID_NDK/ndk-build" \
  NDK_PROJECT_PATH=. NDK_APPLICATION_MK=./Application.mk \
  APP_BUILD_SCRIPT=./Android.mk \
  NNTRAINER_ROOT="<repo>" HEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT" \
  unittest_hvx_rope
# -> test/jni/obj/local/arm64-v8a/unittest_hvx_rope   (libs/ 아님!)
```

**`NNTRAINER_ROOT`를 반드시 명시하세요.** 이 개발 머신들의 프로필이 다른
체크아웃을 가리키는 `NNTRAINER_ROOT`를 export 하고 있고, Android.mk의
`ifndef` 기본값이 조용히 거기에 집니다 (`13_htp_pr_plan.md` §3b에서 실제로
물린 함정).

`subprojects/iniparser`가 빈 wrap-git placeholder면:

```bash
git submodule update --init subprojects/iniparser
```

---

## 4. push + 실행

가장 간단한 방법은 기존 스크립트를 그대로 쓰는 것입니다 — `unittest_hvx_rope`가
이미 그 안에 배선되어 있습니다:

```bash
cd <repo>/test/htp
bash run_u8i4_layer_on_device.sh
```

RoPE만 다시 돌리려면:

```bash
DEVICE_TMP=/data/local/tmp/htp_u8i4_layer_test
adb shell "cd $DEVICE_TMP && \
  LD_LIBRARY_PATH=$DEVICE_TMP ADSP_LIBRARY_PATH=$DEVICE_TMP \
  ./unittest_hvx_rope" 2>&1 | tee /tmp/hvx_rope_device_run.log
```

`ADSP_LIBRARY_PATH`가 skel이 있는 디렉터리를 가리켜야 DSP가 로드합니다.
unsigned PD는 `unittest_hvx_rope.cpp`의 `HtpSession::SetUp()`이 이미
`remote_session_control`로 켭니다.

---

## 5. 게이트 -- 무엇이 통과해야 하는가

`unittest_hvx_rope`가 아래 8개 테스트를 전부 `[  PASSED  ]`로 내야 합니다:

| 테스트 | 확인하는 것 |
| :-- | :-- |
| `RejectsBadLengths` | 길이 검증 |
| **`DoesNotWritePastTheOutputBuffer`** | **§2.1 오버런 수정의 회귀 테스트.** 가장 중요 |
| `MatchesReferenceWithinOneLsb` | 복원된 매트릭스 (dim/width-dim/n_rows 전 축), max_lsb <= 1, amax 비율 <= sqrt(2)+0.05 |
| `BroadcastEqualsPerRowSameValue` | stride 0/1 경로 동등성 |
| `LaneOrderAndSaturationAreObservable` | lane 순서, saturation 카운터 |
| `GeneratesCacheAndReusesIt` | 새 `rope_cache_init` 시그니처, `attention_scaling` 변경 시 generation 증가 |
| `ZeroedThetaGivesIdentityRotation` | partial rotary 메커니즘 |
| `RejectsCacheRangeMismatch` | 범위 검증 |

**`DoesNotWritePastTheOutputBuffer`가 실패하면 (드물게 있을 수 있음):**
가드 바이트가 우연히 다른 페이지에 배정되어 통과했을 수도 있다는 뜻이 아니라,
**FastRPC가 정말로 버퍼 밖을 건드렸다는 뜻**입니다. 이 테스트는 문서
`43_rope_u8_fix_task.md` §4가 명시한 대로 "best-effort detector"이지만,
`hvx_rope_u8.c:96-99`(수정 전 코드)의 알려진 결함을 정확히 겨냥하고 있으므로
실패하면 그 수정이 device에서 재현되지 않았다는 뜻입니다 -- 코드를 다시
확인하세요, tolerance를 조정하지 마세요.

**출력에서 확인할 값들** (`ROPE_FIELD ...` 줄):

```
ROPE_FIELD rows=... width=... dim=... max_lsb=... rms_lsb=...
           mismatch_ratio=... saturated=... max_amax_ratio=...
```

- `max_lsb`는 모든 케이스에서 **0 또는 1**이어야 합니다 (2 이상이면 반올림
  경로 문제, `43_rope_u8_fix_task.md` §6.3 진단표 참고)
- `max_amax_ratio`는 모든 케이스에서 **1.4 (~sqrt(2)) 근방 이하**여야 합니다
- `saturated`는 이 매트릭스에서 대부분 0에 가까워야 정상입니다 (일부러 만든
  `LaneOrderAndSaturationAreObservable`의 두 번째 절만 예외)

**threshold assert가 없는 것은 의도입니다** (`43_rope_u8_fix_task.md` RULE
ZERO, thermal throttling 때문에 flaky). 값을 눈으로 확인하세요.

---

## 6. 전체 스위트 회귀 확인 (선택이지만 권장)

RoPE만 바꿨어도 `run_u8i4_layer_on_device.sh`는 u8i4/u8i8 layer, softmax,
attention, FC 스위트를 전부 같이 돌립니다. 이번 변경이
`nntr_hvx_session.h` (세션 구조체)와 `hvx_add_f32.c` (`nntr_hvx_close`)를
건드렸으므로, **다른 엔트리 포인트가 이 세션 구조체 변경으로 깨지지
않았는지** 확인하는 의미가 있습니다:

```bash
grep -E "^\[  (PASSED|FAILED)|U8I[48]_FIELD" /tmp/hvx_mm_u8i4_device_run.log
grep -E "^\[  (PASSED|FAILED)|BLOCKED_FIELD" /tmp/hvx_softmax_device_run.log
grep -E "^\[  (PASSED|FAILED)|ATTN_FIELD" /tmp/hvx_attn_device_run.log
```

전부 `PASSED`가 나와야 합니다. 이 스위트들은 `nntr_hvx_session` 구조체의
다른 필드(`weights_u8i4`, `quant_pool` 등)를 쓰므로, 제 RoPE 필드 재배치가
구조체 레이아웃을 실수로 깬 게 없는지의 가장 직접적인 증거입니다.

---

## 7. 실패 시 진단

| 증상 | 먼저 볼 곳 |
| :-- | :-- |
| skel 빌드에서 `Q6_Wuh_vunpack_Vub` 관련 오류 | 이 SDK 버전의 헤더에 그 이름이 없거나 시그니처가 다름 -- `hvx_quant_u8.c`나 다른 기존 커널이 쓰는 위닝 인트린식으로 교체 |
| `DoesNotWritePastTheOutputBuffer` 실패 | §5 참고. `hvx_rope_u8.c`의 `rotated` 계산과 tail memcpy 조건(`out != row`) 재확인 |
| `GeneratesCacheAndReusesIt`에서 generation이 안 바뀜 | `nntr_hvx_rope.c`의 cache-hit 비교 조건에 `memcmp(thetas)` 등 새 필드가 빠졌는지 확인 |
| `MatchesReferenceWithinOneLsb`에서 특정 `dim`만 실패 | tail(`dim=96`) 관련이면 `half % ROPE_LANES` 처리, `rope_block`의 `n` 계산 확인 |
| 전체 스위트에서 RoPE 아닌 다른 테스트가 깨짐 | `nntr_hvx_session.h` 필드 재배치가 다른 엔트리 포인트의 offset 가정과 충돌하는지 확인 (일반적으로는 이름으로 접근하니 문제 없어야 함) |

---

## 8. 여기까지 끝나면 다음은

이 태스크(43번, S0)의 범위는 커널 수정과 device 게이트까지입니다.
`42_rope_u8_review_and_next.md` §6의 순서표 기준으로 다음 단계는:

- **S1b** -- QNN cos/sin `$Const` 텐서의 실제 encoding 추출 (`table_scale`/
  `table_zp`의 진짜 값. 지금은 파라미터화만 되어 있고 테스트는 placeholder
  값을 씁니다)
- **S2** -- QNN RoPE 입출력을 덤프해서 지금 이 fused 커널과 LSB 차이 측정
- 그 결과에 따라 S3(per-op) 또는 종료

이 문서(44번)의 범위는 여기까지입니다. S1b/S2는 QNN 툴체인(`qnn-net-run`
등)이 필요하고 별도 태스크입니다.
