# HexKL fc_layer mm comparison (u8i8 vs u8i4)

A standalone example that runs **one fully-connected matmul**
`C[M,N] = A[M,K] × W[K,N]` through the HTP quantized paths and reports accuracy
(relErr vs an FP32 reference) and latency for **u8i8** (INT8 weight) and
**u8i4** (INT4 weight). This is *not* the qwen3-0.6b model — it is a single
fc_layer's matmul in isolation, so the quantized GEMM is compared directly with
FP32.

Weight quantization (per-output-channel symmetric INT8/INT4 + `zp_corr`) and
activation quantization (per-tensor UINT8, zp=128) are done inline, matching
`quantize_qint{4,8}_weight` / `shgemm_u8i{4,8}_i32`. The integer accumulator
`C_i32 = Σ_k X_u8·W_q` is what the NPU computes:

- **On a Hexagon device (`ENABLE_HEXKL`)** the example brings up the CDSP
  session itself (`sdkl_npu_initialize`) and calls the real
  `sdkl_npu_mm_u8i{4,8}_i32` kernels → latency is true NPU latency.
- **On any host** the same integer GEMM runs on the CPU. The integer product is
  exact either way, so the **relErr equals the on-device relErr**; only latency
  differs (host latency is CPU-emulated and labelled).

The example is **self-contained**: it links only `libsdkl.so` (the sdkl C API)
and uses no `libnntrainer` C++ symbol, so it is unaffected by which symbols the
Android `libnntrainer.so` exports.

## Shape

Default = qwen3-0.6b `q_proj` (`M=64` prefill tile, `N=2048`, `K=1024`).

```
--proj q_proj|k_proj|v_proj|o_proj|ffn_gate|ffn_up|ffn_down   pick a preset
--M <m>   rows / tokens (prefill tile e.g. 64; decode = 1)
--N <n>   out features (multiple of 32)
--K <k>   in  features (even, for INT4 packing)
--iters <n> --warmup <n>      steady-state iterations / warmup calls
```

Iteration counts can also come from env vars (a CLI flag wins if both are set):
`HEXKL_FC_ITERS`, `HEXKL_FC_WARMUP`.

## Timing breakdown (QNN-style phases)

The report mirrors a `qnn-net-run` profile so the two can be compared:

| phase | HexKL source | QNN analog |
| :--- | :--- | :--- |
| One-time init | `sdkl_npu_initialize` (once) | One-time init |
| Cold run | first `sdkl_npu_mm` execute | Cold run |
| Steady-state NetRun (**compute**) | mean `sdkl_npu_mm` (host-observed execute = RPC round-trip + device compute + return) | Steady-state NetRun |
| Data movement | host-side `sdkl_npu_alloc` + H2D copies + RM→WH pack + D2H copy | (device Convert/Writeback, host-side here) |

sdkl does **not** expose the intra-execute device timeline (Convert / FC /
Reshape / Writeback), so — unlike QNN's `QnnProfile` — those sub-phases inside
one `sdkl_npu_mm` call cannot be split. "Steady-state NetRun" is the finest
pure-compute number sdkl gives.
`--N`/`--K` override `--proj`. Presets (qwen3-0.6b, hidden 1024, intermediate
3072, GQA q=2048/kv=1024): q_proj N=2048 K=1024; k/v/o_proj N=1024 K=1024;
ffn_gate/up N=3072 K=1024; ffn_down N=1024 K=3072.

## Build & run — host (no NPU, real accuracy numbers)

The example is a normal app target (self-contained; the host build needs no
HTP/NPU support):

```bash
meson setup build -Denable-app=true    # (plus your usual host options)
ninja -C build Applications/HexKLFcCompare/jni/hexkl_fc_compare
./build/Applications/HexKLFcCompare/jni/hexkl_fc_compare            # q_proj
./build/Applications/HexKLFcCompare/jni/hexkl_fc_compare --proj ffn_down
./build/Applications/HexKLFcCompare/jni/hexkl_fc_compare --M 1      # decode
```

Example output (on device the `movement` columns are non-zero and `compute` is
the pure `sdkl_npu_mm` kernel time; `compute` = kernel only, `movement` = NPU
alloc + host↔NPU copies + WH layout pack):

```
| method | relErr vs FP32 | compute us | movement us | engine |
|---|---|---|---|---|
| u8i8 (INT8 weight) | 0.00547 |   1650.0 |    320.0 | NPU/HMX |
| u8i4 (INT4 weight) | 0.07207 |    246.0 |    210.0 | NPU/HMX |

movement breakdown (us) — alloc / H2D copy / WH pack / D2H copy:
  u8i8: alloc=..  h2d=..  whpack=..  d2h=..
  u8i4: alloc=..  h2d=..  whpack=..  d2h=..
```

## Build & run — Android device (real NPU latency)

Uses a flat ndk-build project (like `test/unittest/jni_htp`). The example links
only `libsdkl.so`, so it does **not** need the HTP `libnntrainer.so` — just the
Hexagon SDK (`HEXKL_SDK_ROOT`) and the NDK:

```bash
export HEXKL_SDK_ROOT=<Hexagon_SDK>/addons/hexkl_addon
export ANDROID_NDK=/opt/android-ndk-r26d ; export PATH=$ANDROID_NDK:$PATH

# 1. build the example (self-contained; only libsdkl.so is linked)
ndk-build -C Applications/HexKLFcCompare/jni \
  NDK_PROJECT_PATH=. APP_BUILD_SCRIPT=Android.mk \
  NDK_APPLICATION_MK=Application.mk -j$(nproc)

# 2. push binary + libsdkl.so + libc++_shared.so.
#    ndk-build with -C <jni> NDK_PROJECT_PATH=. writes libs/ and obj/ UNDER the
#    jni dir, so the paths include jni/. Push libs/ first, then the (unstripped)
#    obj/ binary last so it is not overwritten by the stripped libs/ copy.
adb push Applications/HexKLFcCompare/jni/libs/arm64-v8a/. /data/local/tmp/
adb push Applications/HexKLFcCompare/jni/obj/local/arm64-v8a/hexkl_fc_compare /data/local/tmp/
#   the CDSP skeleton (libhexkl_skel.so, V79) must already be in /data/local/tmp

# 4. run (inject library + skeleton paths). Add /vendor/lib64 (and keep the
#    existing $LD_LIBRARY_PATH): libsdkl.so pulls rpcmem_alloc2 from
#    /vendor/lib64/libcdsprpc.so (FastRPC). Two failure modes to avoid:
#    - LD_LIBRARY_PATH=/data/local/tmp alone drops /vendor/lib64 -> loader
#      fails with "cannot locate symbol rpcmem_alloc2 referenced by libsdkl.so".
#    - Do NOT add /system/lib64: it drags in Android system libs (e.g.
#      libinput.so) built against a different libfmt and fails to link.
adb shell "cd /data/local/tmp && \
  LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64 \
  ADSP_LIBRARY_PATH=/data/local/tmp \
  ./hexkl_fc_compare --proj q_proj"
adb shell "cd /data/local/tmp && \
  LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64 \
  ADSP_LIBRARY_PATH=/data/local/tmp \
  ./hexkl_fc_compare --proj ffn_down --M 64"
```

On device the `engine` column reads `NPU/HMX` and `latency (us)` is the real
per-call kernel time.
