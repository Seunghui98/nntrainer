# HexKL Build and Run — from Host Build to On-Device qwen3-0.6b Execution

Describes the end-to-end flow of building an Android binary with the HTP backend enabled, and running actual qwen3-0.6b (WH-baked FP16) inference on-device.

## 1. Required Environment Variables

Set the following environment variables on the host machine before building.

```bash
# Path to the HexKL Addon library inside the Hexagon SDK (includes libsdkl.so and headers)
export HEXKL_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1/addons/hexkl_addon

# Android NDK path
export ANDROID_NDK=/opt/android-ndk-r26d
export PATH=$ANDROID_NDK:$PATH
```

## 2. Build the nntrainer HTP Library

Build for the Galaxy S25 Ultra (SM-S938N, Snapdragon 8 Elite) target:

```bash
# hexkl-lib-subdir=armv8_android26: meson default; avoids a known SIGILL crash on this device
./tools/package_android.sh \
  --arm-arch=armv8.2-a \
  -Denable-htp=true \
  -Dhexkl-sdk-root=$HEXKL_SDK_ROOT \
  -Dhexkl-lib-subdir=armv8_android26 \
  -Dmmap-read=false \
  -Dwerror=false
```
Output: `builddir/android_build_result/lib/arm64-v8a/libnntrainer.so`, used by the CausalLM app in the next step.

## 3. Build the CausalLM App (`nntrainer_causallm`)

Build the CausalLM app that links the HTP `libnntrainer.so` produced in §2:

```bash
# --cache reuses the HTP libnntrainer.so from §2; without it, this rebuilds
# nntrainer from scratch without the HTP flags
cd Applications/CausalLM
./build_android.sh --cache
```

`build_android.sh` auto-detects HTP from the meson `builddir` (§2): when it was
configured with `-Denable-htp=true`, the script builds `nntr_quantize` with
`ENABLE_HEXKL` so the §5 WH bake works. It resolves the HexKL addon root from
`$HEXKL_ADDON_ROOT`, else the builddir's `hexkl-sdk-root`, else the default SDK
path — no extra flag needed. If the builddir is HTP but no addon root resolves,
the build stops with an error rather than silently producing a non-baking
`nntr_quantize`.

Output: `Applications/CausalLM/jni/libs/arm64-v8a/{nntrainer_causallm, nntr_quantize, libcausallm_core.so}`, deployed to the device in the next step.

## 4. On-Device Deployment (`install_android.sh`)

```bash
# pushes libsdkl.so matching §2's hexkl-lib-subdir (armv8_android26) — avoids the armv9 SIGILL crash
./install_android.sh
```

Output: pushes `nntrainer_causallm`, `nntr_quantize`, `libcausallm_core.so`, `libnntrainer.so`, `libsdkl.so`, and other runtime libs to `/data/local/tmp/nntrainer/causallm/`, and generates `run_causallm.sh`, `run_quantize.sh`, `run_safetensors_info.sh`.

## 5. Prepare the qwen3-0.6b Model (WH-baked FP16)

Bakes FP32 weights into a WH-baked FP16 `.bin` on-device, so prefill uses pinned WH pointers with no `rm_to_wh` recomputation (details: [03 §3.6](03_backend_internals.md); record E2E measurements in [05](05_e2e_performance_results.md)).

### (Optional) Obtain the FP32 `.bin`

Skip this — `Applications/CausalLM/res/qwen3/qwen3-0.6b/` already ships `nntr_qwen3_0.6b_fp32.bin`. Only needed to convert a fresh GGUF file:

```bash
python3 Applications/CausalLM/res/qwen3/qwen3-0.6b/gguf_to_nntrainer.py \
  /path/to/qwen3-0.6b.gguf \
  -o nntr_qwen3_0.6b_fp32.bin --target arm --emit-nntr-config
```

### Push and bake WH on-device

```bash
adb push Applications/CausalLM/res/qwen3/qwen3-0.6b \
  /data/local/tmp/nntrainer/causallm/models/

# source nntr_config.json must declare "FP32-FP32"/"FP32" — a wrong dtype here silently corrupts the weights
adb shell \
  "/data/local/tmp/nntrainer/causallm/run_quantize.sh \
   models/qwen3-0.6b --fc_dtype FP16_WH \
   --output_bin nntr_qwen3_0.6b_fp16_wh.bin"
```

Verify the bake actually ran (a `nntr_quantize` built without `ENABLE_HEXKL`
prints a WARNING and writes no trailer):

```bash
# The bake output must NOT contain this line:
#   [WARNING] --fc_dtype FP16_WH requested but this build was compiled without ENABLE_HEXKL
# Confirm the model carries the WH trailer (last 4 bytes = "WHF1"):
adb shell "tail -c 4 /data/local/tmp/nntrainer/causallm/models/qwen3-0.6b/nntr_qwen3_0.6b_fp16_wh.bin | od -An -c"
#   expected: W  H  F  1
```

On first inference (§6), logcat shows `[HTP] Registered N/N pre-baked WH weights`
and prefill runs in the ~240-296 TPS range (a trailerless model instead re-bakes
every run, collapsing prefill to single-digit TPS).

Output: `nntr_qwen3_0.6b_fp16_wh.bin` (WH-baked) + `nntr_config_quantized.json`, which the app auto-loads in place of `nntr_config.json`.

## 6. Run and Verify Success

```bash
adb shell \
  "/data/local/tmp/nntrainer/causallm/run_causallm.sh \
   /data/local/tmp/nntrainer/causallm/models/qwen3-0.6b \
   'The capital of France is'"
```

## 7. Running Unittests

See [04. Unittest Guide and Results](04_unittest_guide.md) for building, deploying, and running the HTP unittest binaries on-device.
