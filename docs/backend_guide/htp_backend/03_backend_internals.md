# HTP Backend Internals: Lifecycle, Memory, and Kernels

Covers the `HtpBackend` singleton's lifecycle, the physical-buffer (ION/RPC) memory allocation strategy, and the low-level kernels and WHCache mechanism behind HTP-accelerated ops.

## 1. HtpBackend Architecture and Lifecycle

### 1.1 HtpBackend Singleton
A single process-wide `HtpBackend` singleton avoids repeating the heavy NPU init.
- Header: `nntrainer/tensor/htp_backend/htp_backend.h`
- Impl: `nntrainer/tensor/htp_backend/htp_backend.cpp`

```cpp
class HtpBackend {
public:
  static HtpBackend &global();
  bool enabled() const { return enabled_; }
  int domain() const { return domain_; }
};
```

### 1.2 Init/Teardown Lifecycle
- **Lazy init**: `sdkl_npu_initialize(domain, ...)` runs once, on first access to the HTP context or kernels — not at program startup.
- **Version query**: on success, logs the CDSP firmware version via `sdkl_npu_get_version`.
- **Teardown**: `sdkl_npu_finalize(domain)` runs in the destructor on normal process exit.
  - `HtpBackend` is destroyed *before* the NPU scratch/cache statics, since it's a function-local static. Their destructors used to call `sdkl_npu_free` on an already-finalized NPU, printing `Err=1` three times at shutdown.
  - Fixed by an atomic flag, `g_npu_alive`: `HtpBackend` sets it `false` right before `sdkl_npu_finalize`, and every teardown free goes through `npuFreeIfAlive()`, which skips the free once the flag is down. This is a safe skip, not a leak — `finalize` already reclaimed that memory.

### 1.3 CPU Fallback
Handles init failure — no HTP hardware on a test PC, or a bad on-device CDSP skeleton path (`/data/local/tmp/libhexkl_skel.so`).
- On a nonzero `sdkl_npu_initialize` return, logs a warning and leaves `enabled_` as `false`.
- `HtpComputeOps::supports_shgemm()` tracks that `enabled()` flag, so it returns `false` for the rest of the process — the ops layer (`float_tensor.cpp` etc.) transparently switches to CPU NEON.

## 2. Memory Allocation and Physical Buffer Mapping

The Hexagon Tensor Processor DMAs directly with the hardware via physical buffers (ION/RPC memory), not virtual addresses. This section covers the current allocation strategy and the NPU DMA pool budget.

### 2.1 HtpMemAllocator — Currently Disabled

`HtpMemAllocator` (`htp_mem_allocator.h`/`.cpp`) wraps `sdkl_npu_alloc`/`sdkl_npu_free` as a `MemAllocator`. It exists in the source tree but **isn't registered in the init path** — allocating every model tensor (weights, activations) from the small NPU DMA pool would exhaust it at full transformer scale (see the comment in `htp_context.cpp::initialize()`).

So the default CPU allocator (regular virtual memory) holds model tensors, and NPU DMA buffers are only allocated/freed per kernel call.

### 2.2 NPU DMA Pool Budget

| Method | Result | Note |
| :--- | :--- | :--- |
| Transient probe: 4 MB chunks, all freed before measuring | **4000 MB** | not usable for pin-once sizing |
| Sustained-pin probe: 4/6/6 MB cycling, 84 allocations, never freed | **448 MB** | hit the loop cap (84), allocator never failed |
| Effective pin budget during kernel execution | **~48 MB** | first `sdkl_npu_mm_f32f16_f32` call permanently reserves ~400 MB of internal scratch |

`sdkl_npu_mm_f32f16_f32` permanently reserves ~400 MB of internal scratch on its first call. That leaves an effective pin-once budget of **B_effective ≈ 48 MB**, not the raw 448 MB — raising the cap to 512 MB made kernel scratch allocation fail during testing, which corrupted SDKL's internal state and caused a SIGSEGV. These figures come from one-off probes run against the device; they are not part of the test suite, because a probe that allocates to failure leaves the allocator unusable for anything that runs after it.

### 2.3 Active Strategy: Reusable NpuScratch Staging Buffers

The FP16 GEMM path (`shgemm_f32f16_f32`) stages every call through three
process-persistent, grow-on-demand scratch buffers — `g_scratch_X`,
`g_scratch_A`, `g_scratch_W` (`NpuScratch` instances in `hexkl_mm.cpp`). Each
`get(bytes)` returns the existing buffer if it's large enough, otherwise frees
and re-allocs to the new size. They are **never freed per call** (the `cleanup`
lambda explicitly leaves them alone), only in their destructors at process exit.
Reusing them across the prefill matmuls removes the ~0.35 ms `sdkl_npu_alloc`/`free`
churn per call.

| Buffer | Backed by | Size | Lifetime |
| :--- | :--- | :--- | :--- |
| `X_npu` (FP32 input, A) | `g_scratch_X` | `Mp × K × 4 bytes` (`Mp` = M rounded up to a multiple of 32) | reused across calls; freed at process exit |
| `A_npu` (FP32 output, C) | `g_scratch_A` | `Mp × N × 4 bytes` | reused across calls; freed at process exit |
| `W_npu` (FP16 weight, WH, prefill) | pin cache **or** `g_scratch_W` | `N × K × 2 bytes` | pinned for the process on a PrefillWHCache hit / pre-baked hit; otherwise staged transiently into `g_scratch_W` (reused, not per-call freed) |

Only the weight has a residency choice: a PrefillWHCache/pre-baked hit returns a
pinned WH pointer directly, while a miss stages the WH bytes into the reused
`g_scratch_W` buffer. The QINT8 path (§3.4) is the only kernel that still calls
`sdkl_npu_alloc`/`sdkl_npu_free` per call.

### 2.4 PrefillWHCache — Pin-Once Residency

`g_prefill_wh` (a `PrefillWHCache`, inside `hexkl_mm.cpp`) is a persistent cache that pays the WH-layout conversion cost only once per weight, on the prefill (M>1) path. See [§3.3.1](#331-prefill-m--1--prefillwhcache-pin-once--transient-fallback) for the kernel-dispatch view of the same mechanism.

```cpp
struct PrefillWHCache {
  std::map<const void*, WHEntry> cache;  // key = CPU-side weight pointer
  size_t total_bytes = 0;
  std::mutex mtx;
  ~PrefillWHCache() { /* sdkl_npu_free all entries */ }
} g_prefill_wh;
```

`getOrCreatePrefillWH(B, N, K)`:

| State | Action | Returns |
| :--- | :--- | :--- |
| Cache hit (pointer + N,K match) | return immediately | existing pinned WH pointer |
| Cache miss + `total_bytes + new_bytes ≤ PREFILL_WH_PIN_MAX_BYTES` | `sdkl_npu_alloc` + memcpy + `sdkl_cpu_rm_to_wh_f16_inplace` + pin | new WH pointer |
| Cache miss + over cap or alloc fails | no allocation, error ignored | `nullptr` → caller uses the transient path |

**Constant:** `PREFILL_WH_PIN_MAX_BYTES = 48 MB` (`hexkl_mm.cpp:79`)

For Qwen3-0.6B, weights pinnable within 48 MB: q_proj(4 MB)×3 + gate_up_proj(6 MB)×3 + down_proj(6 MB)×3 = **48 MB (layers 1-3, 9 weights)**. Layers 4-28 use the transient path.

The cache lives for the **process lifetime** (no eviction); everything is unpinned in the destructor at process exit.

## 3. Integrated Kernels and the WHCache Mechanism

The low-level matrix multiplication kernels and buffer management strategy behind HTP-accelerated ops.

### 3.1 ComputeOps Binding and Kernels

`HtpComputeOps` inherits `CpuComputeOps` and overrides two NPU-only kernels.

| Method | SDKL API | Data types | Condition |
| :--- | :--- | :--- | :--- |
| `shgemm` | `sdkl_npu_mm_f32f16_f32` | FP32 input × FP16 weight → FP32 output | `ENABLE_FP16` |
| `shgemm_u8i8` | `sdkl_npu_mm_u8i8_i32` | U8 input × I8 weight (WH) → I32 → FP32 output | always |

Both kernels' `supports_shgemm()`/`supports_shgemm_u8i8()` return `HtpBackend::global().enabled()`, so they fall back to CPU automatically when the NPU isn't initialized.

### 3.2 RM → WH Layout Transform

Qualcomm HMX tiles need the weight matrix in WH (Width-Height) layout to run in parallel. CPU-loaded weights are row-major, so they're converted via `sdkl_cpu_rm_to_wh_f16_inplace` (FP16) or a pre-converted WH buffer before reaching the NPU.

- **FP16 GEMM**: `shgemm_f32f16_f32` does the RM→WH conversion in-place internally.
- **QINT8 GEMM**: `shgemm_u8i8_i32` already receives a WH-converted `B_wh` pointer (converted ahead of time during offline quantization) — no extra conversion.

### 3.3 FP16 GEMM Buffer Management: Per-Path Strategy

`shgemm_f32f16_f32` branches into two paths based on M.

#### 3.3.1 Prefill (M > 1) — PrefillWHCache (Pin-Once) + Transient Fallback

`g_prefill_wh` pays the WH-layout conversion cost only once per weight. (See [§2.4](#24-prefillwhcache--pin-once-residency) for the memory-allocation view.)

| State | Action | `W_npu` lifetime |
| :--- | :--- | :--- |
| Cache **hit** (same pointer + N,K) | return the pinned WH pointer immediately | lives for the process |
| Cache **miss** + `total_bytes + new_bytes ≤ 48 MB` | alloc + memcpy + rm_to_wh + pin | lives for the process |
| Cache **miss** + over cap or alloc fails | not pinned, returns `nullptr` → transient path | staged into the reused `g_scratch_W` buffer (§2.3), not per-call freed |

**Constant:** `PREFILL_WH_PIN_MAX_BYTES = 48 MB` (`hexkl_mm.cpp:79`)

For Qwen3-0.6B, 9 weights (3 FC types × layers 1-3) fit in the 48 MB pin budget; layers 4-28 get `nullptr` and use the transient path.

#### 3.3.2 Decode (M = 1) — NPU by default, same routing as prefill

Decode (M=1) routes through the same `shgemm_f32f16_f32` kernel used by
prefill (Mp is padded to 32 internally) whenever `supports_shgemm() && N % 32
== 0` — the same condition prefill uses, with no separate runtime opt-in.
`float_tensor.cpp::dotFloat32Float16` falls back to CPU NEON `hsgemv` only
when the NPU is down or `N % 32 != 0`.

The decode path probes `lookupPrefillWH` first and, on a hit, stages the
pre-baked WH bytes into scratch with a plain `memcpy` (no rm_to_wh), just like
the prefill fast path; on a registry miss it falls back to the `g_wh_cache`
path (64 MB FIFO, rm_to_wh per miss), which on models larger than the cap
thrashes.

This became the default after on-device A/B benchmarking showed
HTP decode at parity or better than CPU decode, both in throughput and
generated-text accuracy, once the WH-registry reuse fix landed.
Even on a registry hit, decode still pays a per-token full-weight staging
`memcpy` and computes an Mp=32-padded matmul for a single real row; a
dedicated NPU GEMV kernel (which would remove both costs) remains a possible
follow-up. See `05_e2e_performance_results.md` for the measurements that
motivated this default.

### 3.4 QINT8 GEMM Pipeline (`shgemm_u8i8_i32`)

The QINT8 path doesn't use WHCache — every buffer (`X_npu`, `C_npu`, `W_npu`) is allocated and freed per call.

```
Input A (FP32, M×K)
  ↓  per-tensor dynamic quantization
X_u8 (U8, Mp×K)  [act_scale = max_abs(A) / 127, zp = 128]
  ↓  sdkl_npu_mm_u8i8_i32
C_i32 (I32, Mp×N)
  ↓  dequantization
C (FP32, M×N)  [C[m,n] = act_scale × wt_scale[n] × (C_i32[m,n] − zp_corr[n])]
```

- `B_wh`: I8 weight, already stored in WH layout during offline quantization
- `wt_scale` (FP32, size N): per-output-channel weight scale
- `zp_corr` (I32, size N): precomputed zero-point correction vector
- M is padded internally to a multiple of 64 (`Mp = ⌈M/64⌉×64`); only the real M rows are returned

### 3.5 Routing Decision (`float_tensor.cpp`)

| Condition | Path | Reason |
| :--- | :--- | :--- |
| M == 1 (decode) && N%32==0 && `supports_shgemm()` | NPU `shgemm_f32f16_f32` | probes pre-baked WH (memcpy, no rm_to_wh), else g_wh_cache; still pays per-token staging + Mp=32 padding. Default behavior (on-device A/B showed parity-or-better vs CPU) |
| M == 1 (decode) && (N%32≠0 or NPU down) | CPU `hsgemv` fallback | same alignment/availability guard as prefill |
| M > 1 && N%32==0 && `supports_shgemm()` && PrefillWHCache hit | NPU `shgemm_f32f16_f32` (pinned WH) | pin-once path — no rm_to_wh cost |
| M > 1 && N%32==0 && `supports_shgemm()` && PrefillWHCache miss | NPU `shgemm_f32f16_f32` (transient `W_npu`) | over cap or first call — recomputes WH every call when pin space is unavailable |
| M > 1 && (N%32≠0 or NPU down) | CPU `shgemm` fallback | avoids an SDK exception on N-alignment violation |

### 3.6 Offline WH Bake Pipeline (Integrated .bin)

The §3.3.1 pin-once cache only amortizes rm_to_wh cost within the 48 MB cap, and only after the first process call. Doing the WH conversion offline once, before deployment, lets weights beyond that cap also skip rm_to_wh from the very first prefill call.

Pipeline:

0. **Precondition: the source must really be FP32.** `nntr_quantize` trusts the dtype declared in the source `nntr_config.json`. If an FP32 file is mislabeled as FP16, the loader reads 4-byte weights as 2-byte values, corrupting them and producing an empty trailer (`Registered 0/0`) plus garbage output. So `nntr_quantize` now aborts if the source isn't declared `"model_tensor_type": "FP32-FP32"` / `"fc_layer_dtype": "FP32"` (`quantize.cpp`; it used to only warn). See [02 §5](02_build_and_run.md).
1. **Bake** (`nntr_quantize --fc_dtype FP16_WH`) — a tool directive, not a real `DataType`; it triggers `setWHBakeRequested(true)` and otherwise saves as normal FP16. During save, `g_wh_collector` records each FC layer's RM→WH conversion as a `WHTrailerEntry{name, N, K, wh_bytes}`.
   - **Past bug**: FC layers store weights as `[K,N]` row-major (`TransB=false`), but the NPU shgemm path and WH-bake both assumed the opposite `[N,K]` layout — mislabeling every FC/QKV/FFN weight's bytes and making **all NPU output garbage** (CPU stayed correct since it reads layout via `ldb`). Fixed by transposing before WH-bake whenever `TransB==false`.
2. **Trailer write** — the RM FP16 bytes are untouched; a `WHF1`-tagged trailer is appended to the file end (`writeWHTrailer`). Older loaders that don't know the trailer just read the same RM binary, so the format stays backward-compatible.
3. **Load-time registration** (`neuralnet.cpp`) — if `readWHTrailer` finds a trailer, each entry is registered via `registerPrefillWH(ptr, N, K, wh_bytes)`. This runs under `ExecutionMode::INFERENCE` too, which is what the CausalLM app always uses.
4. **Prefill fast path** (`shgemm_f32f16_f32`) — looks up `lookupPrefillWH(ptr, N, K)` first. A hit skips rm_to_wh entirely; a miss falls back to the §3.3.1 pin-once/transient path. Neither case throws.
5. **Decode benefits identically** — decode (§3.3.2) probes the same `lookupPrefillWH` registry as prefill, so a trailer-baked model skips rm_to_wh on decode too.

Measured (with Qwen3-0.6B on S25 Ultra): loading an integrated bin with the trailer logged `[HTP] Registered 196/196 pre-baked WH weights`, confirming all FC weights were registered; most layers hit the fast path from the very first prefill call, dropping prefill time from 3178 ms to 96 ms (record new E2E measurements in `05_e2e_performance_results.md`).

### 3.7 Diagnostic Env Toggles

| Toggle | Effect |
| :--- | :--- |
| `NNTR_HTP_DISABLE_PREBAKED_WH=1` | forces `lookupPrefillWH` to miss, using the known-good transient RM→WH path (escape hatch / bisection) |

This is the only toggle the backend reads. Equivalence between the pre-baked
and transient paths is asserted by
`PrefillWHRegistry_UsedByShgemmMatchesTransient` and
`OfflineWH_ConversionIsDeterministicAndByteIdentical`, so it is a property the
suite proves rather than something to re-check by hand at runtime.
