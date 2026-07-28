# HexKLQkvChain

Q/K/V projections run back to back through HexKL, to measure what three
`fc_layer` calls cost against one fused one.

Shapes default to **qwen3-0.6b** (hidden 1024, 16 query heads / 8 KV heads,
head_dim 128):

| projection | M | N (out) | K (in) |
| :--- | :---: | :---: | :---: |
| q_proj | 64 | 2048 | 1024 |
| k_proj | 64 | 1024 | 1024 |
| v_proj | 64 | 1024 | 1024 |

`M` = 64 is a prefill tile; `M` = 1 is decode. Q, K and V all read the same
activation — they are parallel in the dataflow but issue back to back, which is
what makes the per-call overhead worth measuring.

Weights are **random** (uniform `[-0.5, 0.5]`, fixed seed). This is a timing
tool, not a correctness check — for accuracy see `HexKLFcE2E`.

## What it reports

Three things:

1. **`model`** — a real nntrainer graph (Input feeding three FullyConnected
   layers), saved with QINT4_HTP weights, loaded back and run through
   `NeuralNetwork::inference()`. This exercises the layer plumbing, the
   `weight_dtype` property and `RunLayerContext`, not just the kernel.
2. **`direct`** — the same three projections as three `Tensor::dot()` calls.
   Always runs, so there is a measurement even if the graph path fails.
3. **`fused`** — one matmul at `N = 2048 + 1024 + 1024 = 4096`. Same
   arithmetic as Q+K+V in a single call; the difference against the three-call
   total is what fusing QKV would buy.

## Build

Needs `libnntrainer.so` built for Android **with HTP**, which the app links
against. Build that first:

```bash
cd $NNTRAINER_ROOT
export HEXKL_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1/addons/hexkl_addon

./tools/package_android.sh \
  -Denable-htp=true \
  -Dhexkl-sdk-root=$HEXKL_SDK_ROOT
```

That leaves `builddir/jni/arm64-v8a/libnntrainer.so` (and `libccapi-nntrainer.so`,
`libsdkl.so`), which is where this app's `Android.mk` looks.

Then the app:

```bash
cd $NNTRAINER_ROOT/Applications/HexKLQkvChain
ndk-build NDK_PROJECT_PATH=. APP_BUILD_SCRIPT=./jni/Android.mk
```

The binary lands in `Applications/HexKLQkvChain/libs/arm64-v8a/hexkl_qkv_chain`.

> **Push from `libs/`, not `jni/libs/`.** If a stale `jni/libs/` exists it is
> not the ndk-build output and pushing it silently benchmarks an old binary.
> The app prints a `built:` timestamp on every run — check it matches the build
> you just did.

## Run

```bash
DEV=/data/local/tmp/hexkl
adb shell mkdir -p $DEV

# libraries (once per build)
adb push builddir/jni/arm64-v8a/libnntrainer.so       $DEV/
adb push builddir/jni/arm64-v8a/libccapi-nntrainer.so $DEV/
adb push builddir/jni/arm64-v8a/libsdkl.so            $DEV/
adb push $ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so $DEV/

# the app
adb push Applications/HexKLQkvChain/libs/arm64-v8a/hexkl_qkv_chain $DEV/
adb shell chmod +x $DEV/hexkl_qkv_chain

adb shell "cd $DEV && LD_LIBRARY_PATH=$DEV ./hexkl_qkv_chain"
```

### Options

```
--M <n>           tokens (default 64; use 1 for decode)
--K <n>           hidden size (default 1024)
--q-N <n>         q out features (default 2048)
--kv-N <n>        k/v out features (default 1024)
--iters <n>       timed iterations (default 100)
--warmup <n>      untimed iterations first (default 10)
--seed <n>        RNG seed for the random weights (default 1234)
--model-path <p>  where to write the baked model
--no-model        skip the graph path, run `direct` only
--no-fused        skip the single N=4096 comparison
```

Useful variations:

```bash
# decode instead of prefill
adb shell "cd $DEV && LD_LIBRARY_PATH=$DEV ./hexkl_qkv_chain --M 1"

# kernel + dispatch only, no graph
adb shell "cd $DEV && LD_LIBRARY_PATH=$DEV ./hexkl_qkv_chain --no-model"

# scalar host compute, to see what NEON is worth here
adb shell "cd $DEV && LD_LIBRARY_PATH=$DEV NNTR_HTP_NO_SIMD=1 ./hexkl_qkv_chain"
```

## Reading the output

The per-projection table reports **median**, mean and max. Median is the
headline: a few calls on this device land in the milliseconds (scheduler or
RPC, not the kernel), and the mean follows them while the median does not. The
max is printed so a bad tail stays visible rather than hidden.

The phase breakdown (`scan / quant / stage / npu / dequant`) comes from
`hmx::lastMmProfile()`. `scan` and `quant` are host work over `M*K` — the same
activation for all three projections, which is precisely the work a fused QKV
would do once instead of three times.

## If the model path fails

`direct` still runs and its numbers stand. The graph path needs three things
that are easy to miss:

- **`engine=htp` on each FC layer.** Without it the graph hands the layer's
  tensors the CPU compute context, and a QINT4_HTP weight has nowhere to
  dispatch. This is the most common cause.
- **`model_tensor_type=QINT4_HTP-FP32`** on the model that loads the baked
  weights.
- A **saved and reloaded** model. A freshly initialized graph cannot simply be
  given QINT4_HTP weights — the scales and `zp_corr` come from the bake, and a
  randomly initialized quantized tensor has none. Save → load is the real path
  a quantized model takes.

The app prints the exception text at whichever step failed rather than
swallowing it.
