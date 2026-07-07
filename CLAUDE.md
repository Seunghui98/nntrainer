# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

NNTrainer is an on-device AI training and LLM inference framework targeting mobile/embedded devices (Android, Tizen, Yocto, Windows). The `hexkl_integration` branch adds a Qualcomm HTP (Hexagon Tensor Processor) backend via the HexKL SDK (`libsdkl.so`).

Working directory for this branch: `/home/j2z0/Project/nntrainer_hexkl` (dedicated git worktree — do **not** touch `htp_libs_integration` or other worktrees).

---

## Build commands

### First-time setup (required)
```bash
git submodule sync && git submodule update --init --depth 1
```

### x86 / Linux (default, no HTP)
```bash
meson setup build -Denable-transformer=true
ninja -C build
```

### Android cross-build (HTP, arm64 device)
```bash
export HEXKL_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1/addons/hexkl_addon
export ANDROID_NDK=/opt/android-ndk-r26d
export PATH=$ANDROID_NDK:$PATH

./tools/package_android.sh \
  --arm-arch=armv9.2-a \
  -Denable-htp=true \
  -Dhexkl-sdk-root=$HEXKL_SDK_ROOT \
  -Dhexkl-lib-subdir=armv9_android26 \
  -Dmmap-read=false \
  -Dwerror=false
```
Build a single test binary to save time:
```bash
ninja -C build_android test/unittest/unittest_nntrainer_htp_kernels
```

### QNN (Qualcomm AI Engine Direct) backend
```bash
meson setup build -Denable-npu=true -Dqnn-sdk-root=/opt/qcom/aistack/qnn-2.24.0.240626
ninja -C build
```

---

## Running tests

### x86 / Linux
```bash
ninja -C build test                     # all tests
./build/test/unittest/unittest_nntrainer_tensor  # single binary
```

Use `--gtest_filter=<Suite>.<Test>` to run a specific test case.

### On-device (HTP tests) — primary device: S25 Ultra, ADB `R3CY205ZMND`
```bash
# Push binary and dependencies
adb -s R3CY205ZMND push build_android/test/unittest/unittest_nntrainer_htp_kernels /data/local/tmp/
adb -s R3CY205ZMND push $HEXKL_SDK_ROOT/lib/armv9_android26/libsdkl.so /data/local/tmp/
# libhexkl_skel.so must already be in /data/local/tmp (device skeleton)

# Run
adb -s R3CY205ZMND shell \
  "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ADSP_LIBRARY_PATH=/data/local/tmp \
   ./unittest_nntrainer_htp_kernels --gtest_color=no"
```

NPU-dependent tests are `[ SKIPPED ]` automatically when no NPU is available.

---

## Code formatting

Run clang-format **14** on every `.c`/`.cpp`/`.h` you change:
```bash
clang-format-14 -i <changed files>
# or match CI exactly (changed lines only):
git clang-format-14 <base-sha>
```

CI (`cpp_linter`) checks only the lines you changed, and only with version 14.

---

## Commit and PR rules

**Commit message format:** `[<component>] <short description>` + blank line + body.

Required trailers on every commit:
```
Signed-off-by: dlwlzzero <dlwlzzero@gmail.com>
Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```
Use `git commit -s` to add the `Signed-off-by` automatically; append `Co-Authored-By` manually.

**PR description:** one `<details>` block per commit (commit message + Self evaluation + Signed-off-by), followed by a Summary section. Template: `.github/PULL_REQUEST_TEMPLATE.md`.

---

## Architecture: backend dispatch

```
Engine (process-wide singleton)
  └─ Context          keyed by name: "cpu" / "gpu" / "qnn" / "htp"
       └─ ContextData  per-vendor state (compute_ops*, allocator, ...)
            └─ ComputeOps*   virtual interface — one method per kernel
                 └─ kernels  NEON / AVX / OpenCL / sdkl calls
```

- `TensorBase` carries `ct_data_` (`shared_ptr<ContextData>`) — which backend a tensor belongs to is stored on the Pimpl, not the `Tensor` handle.
- At layer compile time (`network_graph.cpp::finalizeContext`), every tensor of a layer is stamped with its backend's `ContextData`.
- `ComputeOps` methods default-throw (not pure virtual) so backends only override what they implement. Accelerator-only ops are paired with `supports_*()` predicates (default `false`).
- Cross-backend mismatches throw `std::invalid_argument`; use `Tensor::to(target_ct)` to migrate.

### HTP backend (this branch)

| File | Role |
|------|------|
| `nntrainer/htp_context.cpp` | `HtpContext::Global()` singleton, registers as `"htp"` |
| `nntrainer/tensor/htp_backend/htp_backend.{h,cpp}` | `HtpBackend` — wraps `sdkl_npu_initialize()` lifecycle |
| `nntrainer/tensor/htp_backend/htp_compute_ops.cpp` | `HtpComputeOps` — overrides `supports_shgemm()` and the actual NPU kernel calls |
| `nntrainer/tensor/htp_backend/htp_mem_allocator.{h,cpp}` | `HtpMemAllocator` — `sdkl_npu_alloc`-backed allocator installed when NPU is up |
| `nntrainer/tensor/htp_backend/hmx_ops/hexkl_mm.{h,cpp}` | Low-level sdkl matmul wrappers; contains 64 MB FIFO `WHCache` for WH-layout weights |
| `test/unittest/unittest_nntrainer_htp_backend.cpp` | nntrainer-API-level HTP tests |
| `test/unittest/unittest_nntrainer_htp_kernels.cpp` | Direct sdkl kernel accuracy + perf tests |

Compiled only when `ENABLE_HEXKL=1` (set by meson when `enable-htp=true`). The HTP backend falls back transparently to CPU when initialization fails — `enabled()` returns false and all `supports_*()` return false.

**Routing logic** (`nntrainer/tensor/float_tensor.cpp::dotFloat32Float16`):

| Condition | Path | Reason |
|-----------|------|--------|
| M == 1 (decode) | CPU `hsgemv` always | Memory-BW bound; CPU NEON FP16 is competitive, avoids HMX precision drift |
| M > 1 && N%32==0 && `supports_shgemm()` (prefill) | NPU `shgemm` | Compute-bound; HMX tile parallelism effective |
| M > 1 && (N%32!=0 or NPU down) | CPU `shgemm` fallback | `hexkl_mm` requires N%32==0; guard here avoids a throw |

The N%32 constraint comes from `sdkl_npu_mm_f32f16_f32` alignment requirements. For M>1 calls, `hexkl_mm` bypasses `WHCache` (weight accessed once per prefill pass); `WHCache` is retained in the code for potential future decode-NPU scenarios.

### LLM inference

`Applications/CausalLM` is the production LLM inference engine. FSU (Flash Storage Utilization) loads MoE experts from flash on-the-fly to keep peak memory low.

---

## Key meson options

| Option | Default | Notes |
|--------|---------|-------|
| `enable-htp` | false | HexKL/HTP NPU backend |
| `hexkl-sdk-root` | `''` | Path to `hexkl_addon` directory (required when `enable-htp=true`) |
| `hexkl-lib-subdir` | `armv9_android26` | Which prebuilt `libsdkl.so` to link (S25 Ultra = `armv9_android26`) |
| `enable-npu` | false | QNN (Qualcomm AI Engine Direct) backend |
| `qnn-sdk-root` | `''` | Path to QNN SDK root |
| `enable-transformer` | false | CausalLM / transformer layers |
| `enable-fp16` | false | FP16 support (required for HTP cross-build) |
| `mmap-read` | true | Set false on Android when mmap of model files is unsupported |
| `enable-fsu` | false | Flash Storage Utilization — stream MoE experts from flash at runtime |
| `enable-kernel-caching` | false | Cache compiled kernels to disk for faster subsequent runs |
| `enable-debug` | false | Enable debug-level logging |
| `arm-arch` | `none` | Target ARM architecture (`armv9.2-a` for S25 Ultra); derives `-march` flags |

---

## Do not edit

`subprojects/` — vendored dependencies (minja, benchmark, …). Changes there do not belong in nntrainer PRs.
