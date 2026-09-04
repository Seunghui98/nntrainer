# LFM2 MoE FFN on HTP — Mobile E2E Run Guide

이 문서는 이번 세션에서 구현/커밋한 것(브랜치 `claude/lfm2-moe-ffn-hexkl-2ivn5v`,
`nntrainer/nntrainer` PR #4327 기준)을 실제 기기에서 끝까지 돌려보기 위한
실행 가이드입니다. 이 컨테이너에는 adb / Hexagon SDK / Android NDK가 없어서
여기서는 호스트 빌드 + 유닛테스트까지만 검증했고, 아래 전부는 **직접 실행해서
확인해야 하는 부분**입니다 (`CLAUDE.md`: "verified by inspection is not
verification").

---

## 0. 이번 세션에서 실제로 구현된 것 (커밋 7개)

| 커밋 | 내용 |
|---|---|
| `[CausalLM] Add --moe_dtype to nntr_quantize, wire it to Lfm2MoELayer` | `nntr_quantize --moe_dtype` 옵션 + `weight_dtype` 로더측 연결 |
| `[nntrainer] Refuse a Q4_0 safetensors file packed for the wrong ISA` | safetensors 로드시 ISA(x4/x8) 불일치를 throw로 거부 |
| `[HTP] Let the M4_0 accel gates opt in at M == 1` | decode(M=1)에서 HTP 커널을 타도록 게이트 수정 (오타: 커밋 제목의 "M4_0"은 "Q4_0"의 오타입니다) |
| `[CausalLM] Give the LFM2 MoE prefill workspace the input's ComputeOps` | prefill 워크스페이스 텐서에 ComputeOps 컨텍스트 상속 (안 하면 조용히 CPU로 폴백) |
| `[CausalLM] Add engine + a layer-id HTP subset to the LFM2 MoE FFN` | `moe_engine` / `moe_htp_layers` config 키로 레이어 단위 HTP on/off |
| `[Android] Add --htp to CausalLM's build script; fix its NDK detection` | `build_android.sh --htp` 플래그, NDK 하드코딩 버그 수정 |
| `[test] Add LFM2-MoE's missing Q4_0 differential test, fix stale comments` | tiny 모델 Q4_0 골든 테스트 추가 |

호스트에서 확인된 것: 전체 meson 테스트 45/45 통과, tiny fixture로
`--moe_dtype`/`moe_engine`/`moe_htp_layers` 세 가지 다 동작 확인 (단,
이 호스트는 `ENABLE_HEXKL` 없이 빌드되어 있어서 `engine=htp`는 항상
"htp Context is not registered"로 즉시 실패합니다 — 이게 기대한 동작이고,
디바이스에서 `--htp`로 빌드해야 실제로 돕니다).

---

## 1. 사전 준비물

- Hexagon SDK 6.4.0.2 checkout (`qaic`, `<remote.h>` 등)
- Android NDK r26d
- `hexkl_addon` — **주의: 두 가지 배포판이 있고 용도가 다릅니다**

| | 레이아웃 | 용도 |
|---|---|---|
| SDK 번들형: `$HEXAGON_SDK_ROOT/addons/hexkl_addon` | `lib/<arch>/libsdkl.so` (flat) | `--htp` 빌드 (`-Dhexkl-sdk-root=`) |
| 독립 베타 드롭 (예: `hxkl-beta2/hexkl_addon`) | `lib/<ver>/<arch>/libhexkl_micro.a` (버전 포함) | `test/htp/build.sh` / DSP skel |

`-Dhexkl-sdk-root`에 독립 베타 드롭을 넣으면 조용히 실패합니다(버전 세그먼트
불일치). 반드시 SDK 번들형을 사용하세요.

- 연결된 Android 기기, 개발 중 검증된 디바이스는 Galaxy S25 Ultra / V79
  (`R3CY10WM83Y`, `R5KL30G6MLT`)

```bash
export HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.4.0.2
export ANDROID_NDK=/path/to/android-ndk-r26d
```

---

## 2. 베이스라인 커널 확인 (이미 검증된 커널이 여전히 동작하는지)

새 코드를 신뢰하기 전에 먼저 확인:

```bash
HEXKL_ROOT=/path/to/hxkl-beta2/hexkl_addon \  # 독립 베타 드롭
HEXKL_SDK_VER=6.4.0.2 \
bash test/htp/run_u8i4_layer_on_device.sh
```

`unittest_hvx_mm_u8i4`/`_softmax`/`_attn`/`_fc` 전체 38개 테스트가 통과해야
합니다. `speedup_vs_harness`가 50배 이상 찍히는 건 정상입니다 (그 테스트
주석 자체가 "printed, never asserted"라고 명시함 — 매 호출마다 weight를
다시 굽는 harness와 안 굽는 layer_x4 간의 차이일 뿐).

---

## 3. 모델 quantize

`--isa ARM`이 핵심입니다 (안 붙이면 x86 호스트에서는 q4_0x8이 나오고,
HTP 경로는 x4를 기대하므로 **에러 없이 조용히 틀린 결과**가 나옵니다 —
이번 세션에서 이 함정의 절반(safetensors ISA 메타데이터 불일치 감지)은
막아뒀지만, `.bin` 포맷은 애초에 ISA를 기록할 수 없으므로 여전히 파일명
접미사(`_ARM.bin`)를 직접 확인해야 합니다).

### 3-1. 기본 레시피 (FFN도 나머지도 전부 Q4_0)

```bash
build/Applications/CausalLM/nntr_quantize <fp32_model_dir> \
  -o <out_dir> \
  --fc_dtype Q4_0 --embd_dtype Q4_0 --lmhead_dtype Q4_0 \
  --isa ARM
ls <out_dir>/*_ARM.bin   # 반드시 존재해야 함. 다른 접미사면 잘못된 패킹
```

`--moe_dtype`은 기본값이 `--fc_dtype`과 같아서 위 명령에는 안 붙여도
됩니다 (FFN도 이미 Q4_0). 이 옵션은 정확도 A/B 테스트용입니다:

```bash
# FFN만 FP32로 남기고 나머지는 Q4_0 — "HTP 경로가 틀렸나" vs
# "Q4_0 자체가 문제인가"를 가르는 용도
build/Applications/CausalLM/nntr_quantize <fp32_model_dir> \
  -o <out_dir_fp32ffn> \
  --fc_dtype Q4_0 --moe_dtype FP32 --embd_dtype Q4_0 --lmhead_dtype Q4_0 \
  --isa ARM
```

### 3-2. 출력 nntr_config.json에 HTP 엔진 켜기

quantize가 만든 `nntr_config.json`을 열어서 두 키를 추가합니다 (이번
세션에서 새로 추가된 config 키; `nntr_quantize`는 자동으로 안 써줍니다,
런타임 선택이라서):

```jsonc
{
  ... (기존 필드들) ...
  "moe_engine": "htp",
  "moe_htp_layers": "0"        // 처음엔 레이어 1개만. 되면 "0,1"로 확장
}
```

`moe_htp_layers`를 비워두면 **모든** MoE 레이어가 htp로 갑니다 — tiny
fixture처럼 작은 모델(레이어 2개)에서만 쓰세요. 실제 LFM2-8B-A1B(MoE 레이어
22개)는 절대 전체를 켜지 마세요: 핸들 1,408개가 필요한데
`HEXKL_MM_U8I4_MAX_WEIGHTS=512`이고 상주 메모리도 3.6GiB가 필요해서
지금 트리에서는 실패합니다 (Stage 6, 이번 세션 범위 밖).

---

## 4. HTP 활성화 빌드

```bash
HEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT bash \
  nntrainer/tensor/htp_backend/generate_stub.sh

cd Applications/CausalLM
./build_android.sh --htp
```

`--htp`는 자동으로:
1. `HEXAGON_SDK_ROOT` 없으면 이름을 대며 즉시 에러
2. `generate_stub.sh` 재실행 (멱등이라 두 번 해도 무해)
3. `./tools/package_android.sh -Denable-htp=true -Dhexkl-sdk-root=$HEXAGON_SDK_ROOT/addons/hexkl_addon` 호출

빌드 후 확인:

```bash
readelf -d builddir/jni/arm64-v8a/libnntrainer.so | grep NEEDED
#   libcdsprpc.so 와 libsdkl.so 가 리스트에 있어야 함
```

기존 `builddir`가 `-Denable-htp` 옵션 없이 만들어진 거면 `meson configure`로는
못 주워서 지워야 합니다 — `build_android.sh`는 매번 `builddir`를 지우고
다시 만드므로 (`--cache` 안 쓰면) 신경 안 써도 됩니다.

---

## 5. 기기에 설치

```bash
cd Applications/CausalLM
./install_android.sh --model=<model_name>
```

**추가로 수동으로 필요한 것** (`install_android.sh`는 skel을 안 챙깁니다):

```bash
DEVICE_DIR=/data/local/tmp/nntrainer/causallm   # install_android.sh와 동일 경로
adb push test/htp/build/libnntr_hvx_skel.so "$DEVICE_DIR/"   # §2에서 베이스라인 검증 때 만들어진 것
```

**절대 하지 말 것**: `builddir/jni/arm64-v8a/libcdsprpc.so`를 기기에
push하지 마세요. 그건 링크타임 스텁이고, 기기의 진짜 `/vendor/lib64/libcdsprpc.so`를
`LD_LIBRARY_PATH`/`ADSP_LIBRARY_PATH`에서 가리면 `HtpBackend::enabled() == 0`이
되거나 결과가 조용히 깨집니다.

---

## 6. 실행 및 증거 확인

```bash
adb shell "cd $DEVICE_DIR && \
  LD_LIBRARY_PATH=$DEVICE_DIR ADSP_LIBRARY_PATH=$DEVICE_DIR \
  ./nntr_causallm <model_dir> --text 'hello' --max_new_tokens 8"
```

### 증거 단계 1 (지금 코드로 바로 가능) — 전송이 실제로 일어났는가

```bash
adb logcat -d | grep -iE "nntrainer|adsprpc"
```

`libnntr_hvx_skel.so`에 대한 `remote_handle64_open` 이후 깨끗한
`remote_handle64_close`가 보이면 성공. 이건 지금 코드 그대로 확인
가능합니다.

### 증거 단계 2, 3 (추가 훅 필요 — 아직 코드에 없음)

`test/htp/nntr_hvx_mm_u8i4.c`에는 이미 `nntr_hvx_mm_u8i4_layer_timed`
(`stage_us[]` 반환, `FC_T_DSP_TOTAL` 등)가 있지만, `nntrainer/tensor/htp_backend/htp_compute_ops.cpp`의
`HtpComputeOps::gemm_q4_0_accel_fp32`는 지금 `nntr_hvx_mm_u8i4_layer`
(일반, non-timed)만 호출합니다. FFN 모양 호출(K=2048/N=3584 또는
K=1792/N=2048)에서 `FC_T_DSP_TOTAL != 0`을 확인하려면, 디버그 플래그로
`_timed` 버전을 호출하고 `stage_us[]`를 출력하는 작은 훅을
`get_or_register`/`gemm_q4_0_accel_fp32` 부근에 추가해야 합니다
(41_moe_ffn_e2e_and_perf_task.md §4 E3 item 2가 정확히 이 훅을
설명합니다). 마찬가지로 호출 횟수 카운터(증거 3)도 아직 없습니다.
이건 디바이스 접근이 있는 세션에서 실측하며 추가하는 게 맞다고 판단해서
이번 세션에서는 만들지 않았습니다 — 필요하면 다음 세션에 요청하세요.

### 정확도: SNR, max relative error 아님

이 프로젝트 자체 하네스가 `max_rel = 1908×`에서도 정상(passing)으로
찍히는 걸 이미 확인해뒀습니다 (4비트 양자화에서 0에 가까운 출력 원소가
있으면 상대오차가 폭발하는 게 정상이라서). SNR(dB)로 비교하세요, Stage 3의
측정치 22.8dB가 기준선입니다.

---

## 7. 기대할 성능 (미리 알고 있어야 결과를 오해 안 함)

- **decode는 대역폭 바운드**입니다. 토큰당 활성 weight 바이트가 ~485MB라서
  CPU든 HTP든 DDR 트래픽만으로 ~35ms/token이 바닥입니다.
- Tier 1(지금 붙인 것, 단일 weight당 FastRPC 1콜)은 decode에서
  레이어당 8콜 × 22레이어 = 176콜/token × 326µs ≈ **57ms/token 전송만으로
  이미 CPU보다 느립니다.** 이건 버그가 아니라 예상된 결과입니다
  (41_moe_ffn_e2e_and_perf_task.md가 이걸 "plumbing milestone, 결과는
  느려지는 게 정상"이라고 명시). 이번 세션은 그 배관을 놓는 것까지이고,
  call-count를 줄이는 Tier 2/3(batch/grouped 커널)은 다음 단계입니다.
- 첫 실행에서 라우터 불균형으로 prefill 쪽 M이 커지면(특히 긴 프롬프트)
  `AEE_ENOMEMORY`가 날 수 있습니다 (VTCM이 K×N weight 전체를 double-buffer로
  들고 있어서 M≥1024에서 터짐). 발생하면 프롬프트를 줄이거나, 크래시가
  아니라 예상된 한계로 기록해두세요.

---

## 8. 문제가 생기면 확인할 순서

1. `_ARM.bin` (또는 safetensors의 `nntr_q4_0_isa: "arm"`) 맞는지
2. `libcdsprpc.so`를 실수로 push 안 했는지
3. `hexkl_addon`이 SDK 번들형(§1 표)인지
4. `moe_htp_layers`가 실제 존재하는 레이어 id인지 (`num_dense_layers` 이상)
5. logcat에 `remote_handle64_open`이 실제로 찍히는지 — 안 찍히면 `engine=htp`가
   해당 레이어에 실제로 안 걸린 것 (§3-2 config 키 재확인)
