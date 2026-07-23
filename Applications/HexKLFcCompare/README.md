# HexKL fc_layer mm comparison (u8i8 vs u8i4)

A standalone example that runs **one fully-connected matmul**
`C[M,N] = A[M,K] × W[K,N]` through the HTP quantized paths and reports accuracy
(relErr vs an FP32 reference) and latency for **u8i8** (INT8 weight) and
**u8i4** (INT4 weight). This is *not* the qwen3-0.6b model — it is a single
fc_layer's matmul in isolation, so the quantized GEMM is compared directly with
FP32.

Weight quantization is the production `quantize_qint{4,8}_weight`
(per-output-channel symmetric INT8/INT4 + `zp_corr`); activation is per-tensor
UINT8 (zp=128). The integer accumulator `C_i32 = Σ_k X_u8·W_q` is what the NPU
computes:

- **On a Hexagon device (`ENABLE_HEXKL`)** the real `sdkl_npu_mm_u8i{4,8}_i32`
  kernels run → latency is true NPU latency.
- **On any host** the same integer GEMM runs on the CPU. The integer product is
  exact either way, so the **relErr equals the on-device relErr**; only latency
  differs (host latency is CPU-emulated and labelled).

## Shape

Default = qwen3-0.6b `q_proj` (`M=64` prefill tile, `N=2048`, `K=1024`).

```
--proj q_proj|k_proj|v_proj|o_proj|ffn_gate|ffn_up|ffn_down   pick a preset
--M <m>   rows / tokens (prefill tile e.g. 64; decode = 1)
--N <n>   out features (multiple of 32)
--K <k>   in  features (even, for INT4 packing)
--iters <n> --warmup <n>
```
`--N`/`--K` override `--proj`. Presets (qwen3-0.6b, hidden 1024, intermediate
3072, GQA q=2048/kv=1024): q_proj N=2048 K=1024; k/v/o_proj N=1024 K=1024;
ffn_gate/up N=3072 K=1024; ffn_down N=1024 K=3072.

## Build & run — host (no NPU, real accuracy numbers)

The example is a normal app target; enable apps and it links `libnntrainer`:

```bash
meson setup build -Denable-app=true    # (plus your usual host options)
ninja -C build Applications/HexKLFcCompare/jni/hexkl_fc_compare
./build/Applications/HexKLFcCompare/jni/hexkl_fc_compare            # q_proj
./build/Applications/HexKLFcCompare/jni/hexkl_fc_compare --proj ffn_down
./build/Applications/HexKLFcCompare/jni/hexkl_fc_compare --M 1      # decode
```

Example output:

```
| method | relErr vs FP32 | latency (ms) | engine |
|---|---|---|---|
| u8i8 (INT8 weight) | 0.00547 | ... | CPU-emulated |
| u8i4 (INT4 weight) | 0.07207 | ... | CPU-emulated |
```

## Build & run — Android device (real NPU latency)

Uses a flat ndk-build project (like `test/unittest/jni_htp`). Build the HTP
`libnntrainer.so` first (see
[docs/backend_guide/htp_backend/02_build_and_run.md](../../docs/backend_guide/htp_backend/02_build_and_run.md) §1–2), then:

```bash
export HEXKL_SDK_ROOT=<Hexagon_SDK>/addons/hexkl_addon
export ANDROID_NDK=/opt/android-ndk-r26d ; export PATH=$ANDROID_NDK:$PATH

# 1. HTP library (produces builddir/jni/arm64-v8a/libnntrainer.so)
./tools/package_android.sh --arm-arch=armv8.2-a \
  -Denable-htp=true -Dhexkl-sdk-root=$HEXKL_SDK_ROOT \
  -Dhexkl-lib-subdir=armv8_android26 -Dmmap-read=false -Dwerror=false

# 2. the example binary
ndk-build -C Applications/HexKLFcCompare/jni \
  NDK_PROJECT_PATH=. APP_BUILD_SCRIPT=Android.mk \
  NDK_APPLICATION_MK=Application.mk -j$(nproc)

# 3. push binary + runtime .so deps (libnntrainer/libsdkl/libc++_shared)
adb push Applications/HexKLFcCompare/libs/arm64-v8a/. /data/local/tmp/
adb push Applications/HexKLFcCompare/obj/local/arm64-v8a/hexkl_fc_compare /data/local/tmp/
#   the CDSP skeleton (libhexkl_skel.so, V79) must already be in /data/local/tmp

# 4. run (inject library + skeleton paths)
adb shell "cd /data/local/tmp && \
  LD_LIBRARY_PATH=/data/local/tmp ADSP_LIBRARY_PATH=/data/local/tmp \
  ./hexkl_fc_compare --proj q_proj"
adb shell "cd /data/local/tmp && \
  LD_LIBRARY_PATH=/data/local/tmp ADSP_LIBRARY_PATH=/data/local/tmp \
  ./hexkl_fc_compare --proj ffn_down --M 64"
```

On device the `engine` column reads `NPU/HMX` and `latency (ms)` is the real
per-call kernel time.
