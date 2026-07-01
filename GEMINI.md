# GEMINI.md

This file provides workspace instructions, architectural details, build/test commands, and contribution guidelines for Gemini agents working in this repository.

## 1. Project Overview & Main Technologies

NNTrainer is an on-device AI training and LLM inference framework targeting mobile, embedded, and PC devices (Android, Tizen, Yocto, Windows, Linux).
This specific branch (`hexkl_integration`) focuses on integrating the **Qualcomm HTP (Hexagon Tensor Processor)** backend via the **Qualcomm HexKL CPU Macro API / HexKL SDK (`libsdkl.so`)**.

### Dedicated Workspace
*   **Working Directory:** `/home/j2z0/Project/nntrainer_hexkl` (dedicated git worktree).
*   **Important:** Do **not** touch `htp_libs_integration` or other sibling worktrees.

---

## 2. Architecture & Backend Dispatch

The backend routing and execution follow a process-wide engine singleton structure:

```
Engine (process-wide singleton)
  └─ Context          keyed by name: "cpu" / "gpu" / "qnn" / "htp"
       └─ ContextData  per-vendor state (compute_ops*, allocator, ...)
            └─ ComputeOps*   virtual interface — one method per kernel
                 └─ kernels  NEON / AVX / OpenCL / sdkl calls
```

*   **TensorBackend Binding:** `TensorBase` carries a Pimpl field `ct_data_` (`shared_ptr<ContextData>`). Which backend a tensor belongs to is stored on the Pimpl, not on the `Tensor` handle.
*   **Backend Stamping:** At layer compile/finalize time (`network_graph.cpp::finalizeContext`), every tensor of a layer is stamped with its backend's `ContextData`.
*   **Default Fallbacks:** `ComputeOps` methods default-throw (not pure virtual). Backends override only the kernels they support. Accelerator-specific ops are paired with `supports_*()` predicates (returning `false` by default).
*   **Migration:** Mismatches between backend contexts during operations throw `std::invalid_argument`. Use `Tensor::to(target_ct)` to migrate a tensor.

### HTP Backend Details (Branch-Specific)
*   Compiled only when `ENABLE_HEXKL=1` (triggered by meson option `-Denable-htp=true`).
*   Falls back transparently to CPU when initialization fails: `enabled()` returns false and all `supports_*()` predicates return false.

| Component / File | Description |
|------------------|-------------|
| `nntrainer/htp_context.cpp` | `HtpContext::Global()` singleton, registers context under name `"htp"`. |
| `nntrainer/tensor/htp_backend/htp_backend.{h,cpp}` | `HtpBackend` class — manages `sdkl_npu_initialize()` lifecycle. |
| `nntrainer/tensor/htp_backend/htp_compute_ops.cpp` | `HtpComputeOps` class — overrides `supports_shgemm()` and invokes NPU kernels. |
| `nntrainer/tensor/htp_backend/htp_mem_allocator.{h,cpp}` | `HtpMemAllocator` class — `sdkl_npu_alloc`-backed allocator installed when NPU is up. |
| `nntrainer/tensor/htp_backend/hmx_ops/hexkl_mm.{h,cpp}` | Low-level sdkl matrix multiplication wrappers. Contains a 64 MB FIFO `WHCache` for WH-layout weights. |
| `test/unittest/unittest_nntrainer_htp_backend.cpp` | nntrainer-API-level unit tests for the HTP backend. |
| `test/unittest/unittest_nntrainer_htp_kernels.cpp` | Direct sdkl kernel accuracy and performance tests. |

#### Routing & Fallback Logic (`nntrainer/tensor/float_tensor.cpp::dotFloat32Float16`)
Due to performance profiles and alignment requirements, the dot/matmul routing is as follows:
*   **M == 1 (Decode phase):** Always routes to **CPU `hsgemv`**. This avoids memory-bandwidth bottlenecks, precision drift on HMX, and takes advantage of highly competitive CPU NEON FP16 performance.
*   **M > 1 && N % 32 == 0 && `supports_shgemm()` (Prefill phase):** Routes to **NPU `shgemm`** (HMX tile parallelism is highly effective here).
*   **M > 1 && (N % 32 != 0 || NPU down):** Falls back to **CPU `shgemm`**. The `hexkl_mm` kernel strictly requires `N` to be a multiple of 32; this guard prevents runtime crashes or assertion failures.
*   *Note on WHCache:* For M > 1 calls, `hexkl_mm` bypasses the `WHCache` since weights are accessed only once per prefill pass.

---

## 3. Environment Variables & Prerequisites

The HTP backend cross-build and on-device testing rely on specific environment paths.

```bash
# Hexagon SDK with HexKL Addon (Required for HTP cross-build)
export HEXKL_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1/addons/hexkl_addon

# Android NDK
export ANDROID_NDK=/opt/android-ndk-r26d
export PATH=$ANDROID_NDK:$PATH
```

---

## 4. Building and Running

### First-Time Workspace Setup
Sync and update the repository submodules:
```bash
git submodule sync && git submodule update --init --depth 1
```

### Build Commands

#### A. x86 / Linux (Default Host, No HTP)
Builds the core library with transformer layers enabled for x86:
```bash
meson setup build -Denable-transformer=true
ninja -C build
```

#### B. Android Cross-Build (HTP Enabled)
Cross-compiles for ARMv9-A (such as S25 Ultra, SM8750) targeting the HTP backend:
```bash
./tools/package_android.sh \
  --arm-arch=armv9.2-a \
  -Denable-htp=true \
  -Dhexkl-sdk-root=$HEXKL_SDK_ROOT \
  -Dhexkl-lib-subdir=armv9_android26 \
  -Dmmap-read=false \
  -Dwerror=false
```

To build only a **single test binary** to save compile time:
```bash
ninja -C build_android test/unittest/unittest_nntrainer_htp_kernels
```

#### C. QNN Backend Build (Qualcomm AI Engine Direct)
Builds using the standard QNN direct SDK:
```bash
meson setup build -Denable-npu=true -Dqnn-sdk-root=/opt/qcom/aistack/qnn-2.24.0.240626
ninja -C build
```

### Key Meson Options Reference

| Option | Default | Description |
|--------|---------|-------------|
| `enable-htp` | `false` | Enable HexKL/HTP NPU backend support |
| `hexkl-sdk-root` | `''` | Root directory of the `hexkl_addon` SDK |
| `hexkl-lib-subdir` | `armv8_android26` | Prebuilt `libsdkl.so` subdirectory. S25 Ultra targets `armv9_android26`. |
| `enable-npu` | `false` | Enable Qualcomm QNN backend |
| `qnn-sdk-root` | `''` | Path to Qualcomm QNN SDK root |
| `enable-transformer`| `false` | Enable CausalLM and Transformer layers |
| `enable-fp16` | `false` | Enable FP16 compilation (required for HTP cross-build) |
| `mmap-read` | `true` | Set false on Android if mmap for model files is unsupported or fails |
| `enable-fsu` | `false` | Flash Storage Utilization — streams MoE experts from flash at runtime |
| `enable-kernel-caching` | `false` | Cache compiled kernels to disk to accelerate startup |
| `enable-debug` | `false` | Enable debug-level logging and assertions |
| `arm-arch` | `'none'` | Set target ARM architecture (`armv9.2-a` for modern Snapdragon targets) |

*Do not edit `subprojects/` files—these are vendored dependencies and changes there do not belong in nntrainer PRs.*

---

## 5. Running Tests

### x86 / Linux (Local Host)
Run all unit tests:
```bash
ninja -C build test
```
Run a single unit test binary:
```bash
./build/test/unittest/unittest_nntrainer_tensor
```
Filter for specific test suites or cases using Google Test filters:
```bash
./build/test/unittest/unittest_nntrainer_tensor --gtest_filter=TensorTest.dot_01
```

### On-Device Testing (HTP Backends)
*   **Primary Device:** Samsung Galaxy S25 Ultra (ADB Identifier: `R3CY205ZMND`)
*   **NPU Tests Skip Policy:** NPU-dependent tests automatically print `[ SKIPPED ]` and exit cleanly when no NPU is available.

To push and run HTP unit tests on the target device:
```bash
# 1. Push test binary
adb -s R3CY205ZMND push build_android/test/unittest/unittest_nntrainer_htp_kernels /data/local/tmp/

# 2. Push HexKL library dependency
adb -s R3CY205ZMND push $HEXKL_SDK_ROOT/lib/armv9_android26/libsdkl.so /data/local/tmp/

# Note: The device skeleton (libhexkl_skel.so) must already reside in /data/local/tmp

# 3. Execute with LD_LIBRARY_PATH and ADSP_LIBRARY_PATH configured
adb -s R3CY205ZMND shell \
  "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ADSP_LIBRARY_PATH=/data/local/tmp \
   ./unittest_nntrainer_htp_kernels --gtest_color=no"
```

---

## 6. Coding Style & Formatting

CI enforces strict guidelines, specifically checking only modified lines.

*   **Formatter:** Must use `clang-format-14` on all modified `.c`, `.cpp`, and `.h` files.
*   Formatting single files:
    ```bash
    clang-format-14 -i <file_path>
    ```
*   Formatting changed lines only (matching CI exactly):
    ```bash
    git clang-format-14 <base-sha>
    ```

---

## 7. Commit & PR Guidelines

Every commit and pull request must strictly follow these formats to pass CI checks and maintain team standards.

### Commit Message Format
```
[<component>] <short description>

<optional detailed description body explaining the WHY>

Signed-off-by: dlwlzzero <dlwlzzero@gmail.com>
Co-Authored-By: Gemini 3.5 Flash <noreply@google.com>
```
*   **Signatures:** Every commit **must** contain both the `Signed-off-by` and the `Co-Authored-By` trailers.
*   **Tip:** Run `git commit -s` to automatically generate the `Signed-off-by` line, then append the `Co-Authored-By` trailer.

### PR Description Format
PR descriptions must adhere to the template in `.github/PULL_REQUEST_TEMPLATE.md`. Each PR description requires:
1. One `<details>` block per commit (containing the commit message, a Self evaluation, and the Signed-off-by).
2. A comprehensive summary section.
