// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hexkl_mm.cpp
 * @date   18 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  HMX matrix-multiply operations backed by HexKL (Phase 1).
 */

#ifdef ENABLE_HEXKL

#include <htp_backend.h>

// sdkl.h references CDSP_DOMAIN_ID / CDSP1_DOMAIN_ID, which come from
// remote.h; it must be included first regardless of which sections below
// end up compiled.
#include <remote.h>
#include <sdkl.h>

namespace {
// Free NPU memory only while the session is alive. At process teardown the
// HtpBackend singleton finalizes the NPU before these namespace-scope
// statics are destroyed; freeing afterwards yields sdkl_npu_free Err=1.
//
// Defined before the ENABLE_FP16 block so later integer-kernel sections can
// share a single definition.
static void npuFreeIfAlive(void *p) {
  if (p != nullptr && nntrainer::npuAlive())
    sdkl_npu_free(p);
}
} // namespace

#ifdef ENABLE_FP16

#include <hexkl_mm.h>
#include <htp_backend.h>

#include <remote.h>
#include <sdkl.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>

// CPU reference GEMM (nntrainer/tensor/cpu_backend) — used only by the
// NNTR_HTP_VERIFY_PREFILL_MM diagnostic below to diff the NPU result against
// a known-correct implementation on the exact same real inputs.
namespace nntrainer {
void shgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
            const unsigned int M, const unsigned int N, const unsigned int K,
            const float alpha, const float *A, const unsigned int lda,
            const _FP16 *B, const unsigned int ldb, const float beta, float *C,
            const unsigned int ldc);
}

namespace {

/**
 * @brief One cached WH-layout (converted) weight resident in NPU memory,
 *        keyed by its original RM host pointer.
 */
struct WHEntry {
  void *npu;
  unsigned int N, K;
  size_t bytes;
};

// Weight cache with a hard cap on total NPU bytes to avoid DMA exhaustion.
// When the cap is exceeded, the oldest entry (by insertion order) is evicted.
static constexpr size_t WH_CACHE_MAX_BYTES = 64 * 1024 * 1024; // 64 MB
// Byte cap for pinned prefill WH weights. Must stay below the measured NPU
// DMA pool (Task 1 probe) minus per-call transient headroom (X_npu + A_npu +
// one W_transient, a few MB for prefill shapes). 48 MB is safe under the
// ~64 MB worst-case estimate; raise it toward (measured_B - 32 MB) after the
// pool probe confirms a larger budget.
static constexpr size_t PREFILL_WH_PIN_MAX_BYTES = 48ull * 1024 * 1024; // 48 MB

/**
 * @brief FIFO-evicting decode (M==1) WH weight cache with a hard cap on
 *        total resident NPU bytes, to avoid DMA pool exhaustion.
 */
struct WHCache {
  std::map<const void *, WHEntry> cache;
  // Insertion-order tracking for FIFO eviction.
  std::vector<const void *> order;
  size_t total_bytes = 0;
  std::mutex mtx;

  void evict_one() {
    if (order.empty())
      return;
    const void *oldest = order.front();
    order.erase(order.begin());
    auto it = cache.find(oldest);
    if (it != cache.end()) {
      total_bytes -= it->second.bytes;
      npuFreeIfAlive(it->second.npu);
      cache.erase(it);
    }
  }

  ~WHCache() {
    for (auto &e : cache)
      npuFreeIfAlive(e.second.npu);
  }
} g_wh_cache;

// Persistent (pinned) WH weight residency for prefill (M>1). Once a weight is
// converted RM->WH and allocated in NPU memory it is kept for the process
// lifetime — never evicted. This matches prefill's single-pass access pattern:
// an evicting cache smaller than the working set thrashes to ~0% hits on a
// sequential scan. Weights that would exceed PREFILL_WH_PIN_MAX_BYTES are not
// cached; those calls use the transient path instead (correct, just slower).
/**
 * @brief Never-evicting, process-lifetime WH weight residency for prefill
 *        (M>1). Weights that would exceed PREFILL_WH_PIN_MAX_BYTES are not
 *        cached and fall back to the transient conversion path instead.
 */
struct PrefillWHCache {
  std::map<const void *, WHEntry> cache;
  size_t total_bytes = 0;
  std::mutex mtx;

  ~PrefillWHCache() {
    for (auto &e : cache)
      npuFreeIfAlive(e.second.npu);
  }
} g_prefill_wh;

// sdkl's WH-bake and NPU matmul both expect the weight physically laid out
// [N, K] row-major (N rows of K-wide vectors). nntrainer's FC-family layers
// store weights [K, N] row-major (TensorDim(1,1,in_dim.width(),unit)) and
// call dot(..., trans=false, trans_in=false) — i.e. TransB=false. Relabeling
// such a buffer as [N,K] (swapping the dims passed to
// sdkl_cpu_rm_to_wh_f16_inplace without moving any bytes) silently scrambles
// every element's tile position; an actual transpose is required first.
// copyForWHBake writes B into `dst` (already sized N*K) in true [N,K]
// row-major order, ready for sdkl_cpu_rm_to_wh_f16_inplace(N, K, dst).
void copyForWHBake(const _FP16 *B, unsigned int N, unsigned int K, bool transB,
                   _FP16 *dst) {
  if (transB) {
    // B is already [N, K] row-major (matches the sdkl/hexkl_mm convention).
    std::memcpy(dst, B, (size_t)N * K * sizeof(_FP16));
  } else {
    // B is [K, N] row-major: dst[n,k] = B[k,n].
    for (unsigned int k = 0; k < K; ++k)
      for (unsigned int n = 0; n < N; ++n)
        dst[(size_t)n * K + k] = B[(size_t)k * N + n];
  }
}

// Return a pinned WH pointer for RM weight B, converting and pinning on first
// request. Never evicts. Returns nullptr (caller falls back to transient) when
// adding this weight would exceed PREFILL_WH_PIN_MAX_BYTES, or on any alloc /
// conversion failure. Never throws — a throw here would crash inference.
const _Float16 *getOrCreatePrefillWH(const void *B, unsigned int N,
                                     unsigned int K, bool transB) {
  if (B == nullptr)
    return nullptr;

  std::lock_guard<std::mutex> lk(g_prefill_wh.mtx);

  auto it = g_prefill_wh.cache.find(B);
  if (it != g_prefill_wh.cache.end() && it->second.N == N && it->second.K == K)
    return static_cast<const _Float16 *>(it->second.npu); // hit

  // Same pointer, different shape (rare pointer reuse): drop the stale entry.
  if (it != g_prefill_wh.cache.end()) {
    g_prefill_wh.total_bytes -= it->second.bytes;
    npuFreeIfAlive(it->second.npu);
    g_prefill_wh.cache.erase(it);
  }

  const size_t new_bytes = (size_t)N * (size_t)K * sizeof(_FP16);

  // Pin budget reached: do not cache, do not evict. Caller uses transient path.
  if (g_prefill_wh.total_bytes + new_bytes > PREFILL_WH_PIN_MAX_BYTES)
    return nullptr;

  void *new_npu = nullptr;
  int err = sdkl_npu_alloc(new_bytes, &new_npu);
  if (err != 0 || new_npu == nullptr)
    return nullptr; // pool exhausted: transient fallback

  copyForWHBake(static_cast<const _FP16 *>(B), N, K, transB,
                static_cast<_FP16 *>(new_npu));
  int werr =
    sdkl_cpu_rm_to_wh_f16_inplace((size_t)N, (size_t)K, (_Float16 *)new_npu);
  if (werr != 0) {
    npuFreeIfAlive(new_npu);
    return nullptr; // conversion failed: transient fallback recomputes safely
  }

  g_prefill_wh.cache[B] = {new_npu, N, K, new_bytes};
  g_prefill_wh.total_bytes += new_bytes;
  return static_cast<const _Float16 *>(new_npu);
}

// Host-resident pre-baked WH weights, keyed by the RM weight pointer. Populated
// offline-then-loaded (see loader). Holds already-converted WH bytes in normal
// host RAM (NOT NPU DMA) — cheap to keep the whole model resident, sidestepping
// the ~48 MB NPU pin budget. shgemm memcpies these into NPU scratch per call.
/**
 * @brief Host-RAM registry of pre-baked (offline-converted) WH weight
 *        bytes, keyed by the RM host weight pointer. Populated by the
 *        loader; cheap to keep the whole model resident since it lives in
 *        normal host RAM rather than the NPU DMA pool.
 */
struct PrefillWHRegistry {
  /**
   * @brief One registered pre-baked weight: its WH-layout bytes and shape.
   */
  struct Entry {
    std::vector<_FP16> wh; // WH-layout bytes, N*K elements
    unsigned int N, K;
  };
  std::map<const void *, Entry> reg;
  std::mutex mtx;
} g_prefill_wh_reg;

// Reusable NPU scratch buffers for prefill staging. sdkl_npu_alloc/free per
// call is ~0.35 ms; reusing across the 84 prefill matmuls removes that churn.
// Grows on demand; never shrinks. Not thread-safe by itself — guarded by the
// caller's single-threaded prefill loop; a mutex makes reuse safe if that
// changes.
/**
 * @brief Reusable, growable NPU scratch buffer for prefill staging, to
 *        avoid an sdkl_npu_alloc/free round trip on every matmul call.
 */
struct NpuScratch {
  void *p = nullptr;
  size_t cap = 0;
  std::mutex mtx;
  void *get(size_t bytes) {
    std::lock_guard<std::mutex> lk(mtx);
    if (bytes <= cap && p != nullptr)
      return p;
    if (p != nullptr) {
      npuFreeIfAlive(p);
      p = nullptr;
      cap = 0;
    }
    void *np = nullptr;
    if (sdkl_npu_alloc(bytes, &np) != 0 || np == nullptr)
      return nullptr;
    p = np;
    cap = bytes;
    return p;
  }
  ~NpuScratch() {
    if (p != nullptr)
      npuFreeIfAlive(p);
  }
};
static NpuScratch g_scratch_X, g_scratch_A, g_scratch_W;
} // namespace

namespace nntrainer {
namespace hmx {

void shgemm_f32f16_f32(const unsigned int TStorageOrder, bool TransA,
                       bool TransB, const unsigned int M, const unsigned int N,
                       const unsigned int K, const float alpha, const float *A,
                       const unsigned int lda, const _FP16 *B,
                       const unsigned int ldb, const float beta, float *C,
                       const unsigned int ldc) {
  // Constraints (§2, sdkl.h 6.4.0.1):
  //   (1) 2D only — calculateFlattenDot already flattened 4D to M/N/K
  //   (2) sdkl_npu_mm_f32f16_f32: result = X * W^T
  //       where X is FP32 input [M×K], W is FP16 weight [N×K] in WH layout,
  //       and the FP32 output is A [M×N].
  //   (3) B is [N, K] row-major — must convert RM→WH before NPU call.
  //   (4) All buffers must be allocated with sdkl_npu_alloc() for alignment.
  //
  // Naming: the sdkl API uses (A=output, X=input, W=weight).
  //   Our nntrainer parameters use (A=input, B=weight, C=output).
  //   Internally we call: sdkl_npu_mm_f32f16_f32(domain,
  //     n_row=M, n_col=N, n_inner=K,
  //     A_npu [out], X_npu [A input], W_npu [B weight WH])
  //
  // alpha/beta: nntrainer always passes alpha=1.0, beta=0.0 for
  //   dotFloat32Float16. General alpha/beta are supported below.
  //
  // TransA/TransB/TStorageOrder: forwarded from HtpComputeOps::shgemm which
  //   mirrors CpuComputeOps. sdkl_npu_mm_f32f16_f32 computes input × weight^T
  //   natively, so TransB=true is the expected call pattern (weight [N,K]).

  int domain = HtpBackend::global().domain();

  // HMX F16 tile alignment constraints (sdkl 6.4, verified on V79/V81):
  //   N (n_col) must be a multiple of 32 — this is a hard weight-layout
  //   requirement that callers must satisfy.
  //   M (n_row) is padded internally: we round Mp = ceil(M/32)*32, allocate
  //   Mp rows for X and A buffers, and only copy back the real M output rows.
  if (N % 32 != 0)
    throw std::runtime_error(
      "shgemm_f32f16_f32: N must be a multiple of 32 (N=" + std::to_string(N) +
      ")");

  // Round M up to the next multiple of 32 for NPU tile alignment.
  const unsigned int Mp = (M + 31u) & ~31u;

  // --- Allocate NPU-accessible staging buffers ---
  // sdkl_npu_alloc(size_t size, void** buffer_ptr): returns 0 on success.
  // sdkl_npu_free(void* addr): no domain parameter.
  void *X_npu = nullptr; // FP32 input (nntrainer's A), Mp*K rows
  void *A_npu = nullptr; // FP32 output (nntrainer's C), Mp*N rows

  auto cleanup = [&]() {
    // X_npu / A_npu are borrowed from g_scratch_* and must NOT be freed here.
  };

  int err = 0;

  X_npu = g_scratch_X.get((size_t)Mp * K * sizeof(float));
  if (X_npu == nullptr) {
    throw std::runtime_error("shgemm_f32f16_f32: X scratch alloc failed");
  }
  A_npu = g_scratch_A.get((size_t)Mp * N * sizeof(float));
  if (A_npu == nullptr) {
    throw std::runtime_error("shgemm_f32f16_f32: A scratch alloc failed");
  }

  // --- Copy input into NPU memory ---
  // X: zero-fill the full Mp*K buffer, then copy the real M rows.
  std::memset(X_npu, 0, Mp * K * sizeof(float));
  std::memcpy(X_npu, A, M * K * sizeof(float));

  // M>1 (prefill): use the persistent WH residency cache (populated at
  // warmup). Heavy weight staging (alloc + memcpy + RM->WH) is done once and
  // reused; the real prefill call becomes lookup + matmul + copy-back.
  // On cache miss/alloc failure, fall back to a transient conversion so the
  // result is always correct (only slower). WHCache below is also used by
  // decode (M==1), which routes through this same function by default.
  if (M > 1) {
    if (beta != 0.0f) {
      cleanup();
      throw std::runtime_error(
        "shgemm_f32f16_f32: beta != 0 not supported (Phase 1)");
    }
    std::memset(A_npu, 0, Mp * N * sizeof(float));

    const size_t w_bytes = (size_t)N * (size_t)K * sizeof(_FP16);
    const _Float16 *W_wh = nullptr;
    void *W_transient = nullptr;

    // Fast path: pre-baked WH bytes registered on the host (offline bake).
    // Stage them into NPU memory with a plain memcpy — no rm_to_wh.
    const _FP16 *wh_host =
      lookupPrefillWH(reinterpret_cast<const void *>(B), N, K);
    if (wh_host != nullptr) {
      W_transient = g_scratch_W.get(w_bytes);
      if (W_transient == nullptr) {
        cleanup();
        throw std::runtime_error("shgemm_f32f16_f32: W scratch alloc failed");
      }
      std::memcpy(W_transient, wh_host, w_bytes);
      W_wh = static_cast<const _Float16 *>(W_transient);

      if (const char *ve = std::getenv("NNTR_HTP_VERIFY_PREBAKED_WH")) {
        if (ve[0] == '1') {
          std::vector<_FP16> chk((size_t)N * K);
          copyForWHBake(B, N, K, TransB, chk.data());
          int vrc = sdkl_cpu_rm_to_wh_f16_inplace(
            (size_t)N, (size_t)K, reinterpret_cast<_Float16 *>(chk.data()));
          if (vrc != 0) {
            std::fprintf(stderr,
                         "[HTP][verify] rm_to_wh failed rc=%d N=%u K=%u\n", vrc,
                         N, K);
          } else if (std::memcmp(chk.data(), wh_host, w_bytes) != 0) {
            size_t nmis = 0, first = (size_t)-1;
            for (size_t i = 0; i < (size_t)N * K; ++i) {
              if (std::memcmp(&chk[i], &wh_host[i], sizeof(_FP16)) != 0) {
                if (first == (size_t)-1)
                  first = i;
                ++nmis;
              }
            }
            std::fprintf(stderr,
                         "[HTP][verify] MISMATCH B=%p N=%u K=%u nmis=%zu "
                         "first=%zu\n",
                         B, N, K, nmis, first);
          } else {
            std::fprintf(stderr, "[HTP][verify] OK B=%p N=%u K=%u\n", B, N, K);
          }
        }
      }
    } else {
      // No pre-baked WH: fall back to the pin cache, then transient convert.
      W_wh = getOrCreatePrefillWH(B, N, K, TransB);
      if (W_wh == nullptr) {
        W_transient = g_scratch_W.get(w_bytes);
        if (W_transient == nullptr) {
          cleanup();
          throw std::runtime_error("shgemm_f32f16_f32: W scratch alloc failed");
        }
        copyForWHBake(B, N, K, TransB, static_cast<_FP16 *>(W_transient));
        int werr = sdkl_cpu_rm_to_wh_f16_inplace((size_t)N, (size_t)K,
                                                 (_Float16 *)W_transient);
        if (werr != 0) {
          cleanup();
          throw std::runtime_error("rm_to_wh_f16_inplace failed (prefill): " +
                                   std::to_string(werr));
        }
        W_wh = static_cast<const _Float16 *>(W_transient);
      }
    }

    err = sdkl_npu_mm_f32f16_f32(domain, (int)Mp, (int)N, (int)K,
                                 static_cast<float *>(A_npu),
                                 static_cast<const float *>(X_npu), W_wh);
    if (err != 0) {
      cleanup();
      throw std::runtime_error("sdkl_npu_mm_f32f16_f32 failed (prefill): " +
                               std::to_string(err));
    }

    const float *A_npu_f32 = static_cast<const float *>(A_npu);
    if (alpha == 1.0f) {
      std::memcpy(C, A_npu_f32, M * N * sizeof(float));
    } else {
      for (unsigned int i = 0; i < M * N; ++i)
        C[i] = alpha * A_npu_f32[i];
    }

    if (const char *dbg = std::getenv("NNTR_HTP_VERIFY_PREFILL_MM")) {
      if (dbg[0] == '1') {
        static int call_idx = 0;
        std::vector<float> C_ref((size_t)M * N);
        nntrainer::shgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda,
                          B, ldb, beta, C_ref.data(), N);
        float max_ref = 0.f, max_err = 0.f, max_a = 0.f, max_b = 0.f;
        for (size_t i = 0; i < (size_t)M * K; ++i)
          max_a = std::max(max_a, std::fabs(A[i]));
        for (size_t i = 0; i < (size_t)N * K; ++i)
          max_b = std::max(max_b, std::fabs((float)B[i]));
        for (size_t i = 0; i < (size_t)M * N; ++i) {
          max_ref = std::max(max_ref, std::fabs(C_ref[i]));
          max_err = std::max(max_err, std::fabs(C[i] - C_ref[i]));
        }
        float rel = max_err / (max_ref + 1e-6f);
        std::fprintf(
          stderr,
          "[HTP][mm-verify] call=%d TransA=%d TransB=%d M=%u N=%u K=%u "
          "lda=%u ldb=%u ldc=%u prebaked=%d maxA=%.4f maxB=%.4f maxRef=%.4f "
          "maxErr=%.4f relErr=%.6f%s\n",
          call_idx++, (int)TransA, (int)TransB, M, N, K, lda, ldb, ldc,
          (int)(wh_host != nullptr), max_a, max_b, max_ref, max_err, rel,
          rel > 0.05f ? " MISMATCH" : "");
      }
    }

    cleanup();
    return;
  }

  // --- Decode (M==1) weight staging ---
  // Fast path: reuse the pre-baked WH registry (same weight pointer as
  // prefill), staging into transient scratch with a plain memcpy — no
  // rm_to_wh. Mirrors the M>1 prefill fast path (see above). Falls back to the
  // g_wh_cache path on a registry miss so the result is always correct.
  // NOTE: this removes the rm_to_wh thrash but not the per-token full-weight
  // staging memcpy nor the Mp=32 padding; a dedicated GEMV kernel (out of
  // scope) would remove those. See the fix-first design doc.
  const _Float16 *W_wh = nullptr;
  const _FP16 *wh_host =
    lookupPrefillWH(reinterpret_cast<const void *>(B), N, K);
  if (wh_host != nullptr) {
    const size_t w_bytes = (size_t)N * (size_t)K * sizeof(_FP16);
    void *W_transient = g_scratch_W.get(w_bytes);
    if (W_transient == nullptr) {
      cleanup();
      throw std::runtime_error(
        "shgemm_f32f16_f32: W scratch alloc failed (decode)");
    }
    std::memcpy(W_transient, wh_host, w_bytes);
    W_wh = static_cast<const _Float16 *>(W_transient);
  } else {
    // --- Weight cache: convert RM->WH once per unique weight pointer ---
    // FSU is off for engine=htp (transformer FSU guard), so B pointer is
    // stable. Cache is size-limited (WH_CACHE_MAX_BYTES) to avoid NPU DMA
    // exhaustion; FIFO eviction removes the oldest entry when the limit is
    // exceeded.
    void *W_npu_cached = nullptr;
    {
      std::lock_guard<std::mutex> lk(g_wh_cache.mtx);
      auto it = g_wh_cache.cache.find(B);
      if (it == g_wh_cache.cache.end() || it->second.N != N ||
          it->second.K != K) {
        // Cache miss: evict stale entry for same B pointer if present.
        if (it != g_wh_cache.cache.end()) {
          g_wh_cache.total_bytes -= it->second.bytes;
          sdkl_npu_free(it->second.npu);
          auto oit =
            std::find(g_wh_cache.order.begin(), g_wh_cache.order.end(), B);
          if (oit != g_wh_cache.order.end())
            g_wh_cache.order.erase(oit);
          g_wh_cache.cache.erase(it);
        }
        // Evict oldest entries until we are below the cap.
        const size_t new_bytes = N * K * sizeof(_FP16);
        while (g_wh_cache.total_bytes + new_bytes > WH_CACHE_MAX_BYTES &&
               !g_wh_cache.order.empty())
          g_wh_cache.evict_one();

        void *new_npu = nullptr;
        err = sdkl_npu_alloc(new_bytes, &new_npu);
        if (err != 0 || new_npu == nullptr) {
          cleanup();
          throw std::runtime_error(
            "shgemm_f32f16_f32: weight NPU alloc failed (err=" +
            std::to_string(err) + ")");
        }
        copyForWHBake(B, N, K, TransB, static_cast<_FP16 *>(new_npu));
        int werr = sdkl_cpu_rm_to_wh_f16_inplace((size_t)N, (size_t)K,
                                                 (_Float16 *)new_npu);
        if (werr != 0) {
          sdkl_npu_free(new_npu);
          cleanup();
          throw std::runtime_error("rm_to_wh_f16_inplace failed: " +
                                   std::to_string(werr));
        }
        g_wh_cache.cache[B] = {new_npu, N, K, new_bytes};
        g_wh_cache.order.push_back(B);
        g_wh_cache.total_bytes += new_bytes;
        W_npu_cached = new_npu;
      } else {
        W_npu_cached = it->second.npu;
      }
    }
    W_wh = static_cast<const _Float16 *>(W_npu_cached);
  }

  // sdkl_npu_mm_f32f16_f32 is write-only (not accumulate); beta != 0 is
  // unsupported in Phase 1. Phase 2: support accumulation if the SDK adds an
  // accumulate mode.
  if (beta != 0.0f) {
    cleanup();
    throw std::runtime_error("shgemm_f32f16_f32: beta != 0 is not supported in "
                             "Phase 1 (NPU write-only)");
  }
  std::memset(A_npu, 0, Mp * N * sizeof(float));

  // --- NPU matrix multiply ---
  // sdkl_npu_mm_f32f16_f32(int domain,
  //                         int n_row, int n_col, int n_inner,
  //                         float* A   [output FP32],
  //                         const float* X  [input FP32],
  //                         const _Float16* W [weight FP16 WH])
  // Note: no lda/ldb/ldc — contiguous row-major assumed.
  // Use Mp (padded row count) so the NPU sees a 32-aligned M.
  err = sdkl_npu_mm_f32f16_f32(domain, (int)Mp, (int)N, (int)K,
                               static_cast<float *>(A_npu),
                               static_cast<const float *>(X_npu), W_wh);
  if (err != 0) {
    cleanup();
    throw std::runtime_error("sdkl_npu_mm_f32f16_f32 failed: " +
                             std::to_string(err));
  }

  // --- Copy result back and apply alpha ---
  // Only copy M real output rows (not the Mp padded rows).
  const float *A_npu_f32 = static_cast<const float *>(A_npu);
  if (alpha == 1.0f) {
    std::memcpy(C, A_npu_f32, M * N * sizeof(float));
  } else {
    for (unsigned int i = 0; i < M * N; ++i)
      C[i] = alpha * A_npu_f32[i];
  }

  cleanup();
}

void registerPrefillWH(const void *rm_ptr, unsigned int N, unsigned int K,
                       const void *wh_src) {
  if (rm_ptr == nullptr || wh_src == nullptr)
    return;
  const size_t n = (size_t)N * (size_t)K;
  std::lock_guard<std::mutex> lk(g_prefill_wh_reg.mtx);
  auto &e = g_prefill_wh_reg.reg[rm_ptr];
  e.wh.assign(reinterpret_cast<const _FP16 *>(wh_src),
              reinterpret_cast<const _FP16 *>(wh_src) + n);
  e.N = N;
  e.K = K;
}

const _FP16 *lookupPrefillWH(const void *rm_ptr, unsigned int N,
                             unsigned int K) {
  if (rm_ptr == nullptr)
    return nullptr;
  // Escape hatch / bisection aid: force the transient RM->WH path.
  if (const char *e = std::getenv("NNTR_HTP_DISABLE_PREBAKED_WH")) {
    if (e[0] == '1')
      return nullptr;
  }
  std::lock_guard<std::mutex> lk(g_prefill_wh_reg.mtx);
  auto it = g_prefill_wh_reg.reg.find(rm_ptr);
  if (it == g_prefill_wh_reg.reg.end() || it->second.N != N ||
      it->second.K != K)
    return nullptr;
  return it->second.wh.data();
}

size_t prefillWHRegistrySize() {
  std::lock_guard<std::mutex> lk(g_prefill_wh_reg.mtx);
  return g_prefill_wh_reg.reg.size();
}

void prefillWHRegistryClear() {
  std::lock_guard<std::mutex> lk(g_prefill_wh_reg.mtx);
  g_prefill_wh_reg.reg.clear();
}

size_t prefillWHCacheSize() {
  std::lock_guard<std::mutex> lk(g_prefill_wh.mtx);
  return g_prefill_wh.cache.size();
}

void prefillWHCacheClear() {
  std::lock_guard<std::mutex> lk(g_prefill_wh.mtx);
  for (auto &e : g_prefill_wh.cache)
    npuFreeIfAlive(e.second.npu);
  g_prefill_wh.cache.clear();
  g_prefill_wh.total_bytes = 0;
}

} // namespace hmx
} // namespace nntrainer

#endif // ENABLE_FP16

#endif // ENABLE_HEXKL
