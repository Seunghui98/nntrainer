// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_nntrainer_htp_kernels.cpp
 * @date   23 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Direct sdkl kernel accuracy + performance tests (kernel level).
 *
 * Compiled only with -Denable-htp=true. NPU tests are runtime-skipped when
 * HtpBackend::global().enabled() is false. Unlike
 * unittest_nntrainer_htp_backend (which goes through the nntrainer compute
 * path), this file calls the sdkl C API directly to validate each kernel in
 * isolation and measure its performance.
 */

#ifdef ENABLE_HEXKL
#ifdef ENABLE_FP16

#include <fp16.h>
#include <gtest/gtest.h>
#include <htp_backend.h>

#include <sdkl_compat.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

using _FP16 = _Float16;

} // namespace

// Forward declaration: shgemm_f32f16_f32 lives in libnntrainer (linked in).
// libnntrainer.so is compiled with USE__FP16=1, so _FP16 == __fp16 there;
// use __fp16 here to match the mangled symbol (half != DF16_).
namespace nntrainer {
namespace hmx {
void shgemm_f32f16_f32(unsigned int TStorageOrder, bool TransA, bool TransB,
                       unsigned int M, unsigned int N, unsigned int K,
                       float alpha, const float *A, unsigned int lda,
                       const __fp16 *B, unsigned int ldb, float beta, float *C,
                       unsigned int ldc);
size_t prefillWHCacheSize();
void prefillWHCacheClear();
void registerPrefillWH(const void *rm_ptr, unsigned int N, unsigned int K,
                       const void *wh_src);
size_t prefillWHRegistrySize();
void prefillWHRegistryClear();
} // namespace hmx

// Forward declaration: CPU reference shgemm (nntrainer/tensor/cpu_backend).
// Same __fp16-vs-_FP16 mangling note as above applies here.
void shgemm(unsigned int TStorageOrder, bool TransA, bool TransB,
            unsigned int M, unsigned int N, unsigned int K, float alpha,
            const float *A, unsigned int lda, const __fp16 *B, unsigned int ldb,
            float beta, float *C, unsigned int ldc);
} // namespace nntrainer

namespace {

// ---- Random generators -----------------------------------------------------

static std::vector<float> makeRandF32(int n, float lo = -1.0f, float hi = 1.0f,
                                      uint32_t seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto &x : v)
    x = dist(rng);
  return v;
}

static std::vector<_FP16> makeRandF16(int n, float lo = -0.5f, float hi = 0.5f,
                                      uint32_t seed = 7) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<_FP16> v(n);
  for (auto &x : v)
    x = (_FP16)dist(rng);
  return v;
}

// C[M,N] = A[M,K] (f32) * W[N,K]^T (f16), fp32 accumulate, fp32 store.
static void cpuGemmF32F16(int M, int N, int K, const float *A, const _FP16 *W,
                          float *C) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k)
        acc += A[m * K + k] * (float)W[n * K + k];
      C[m * N + n] = acc;
    }
}

// Relative error over fp32 arrays.
static float relErrorF32(const float *npu, const float *cpu, int n) {
  float ref_max = 0.0f, err_max = 0.0f;
  for (int i = 0; i < n; ++i) {
    ref_max = std::max(ref_max, std::abs(cpu[i]));
    err_max = std::max(err_max, std::abs(npu[i] - cpu[i]));
  }
  return err_max / (ref_max + 1e-6f);
}

static std::vector<uint8_t> makeRandU8(int n, int lo = 0, int hi = 15,
                                       uint32_t seed = 11) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(lo, hi);
  std::vector<uint8_t> v(n);
  for (auto &x : v)
    x = (uint8_t)dist(rng);
  return v;
}

static std::vector<int8_t> makeRandI8(int n, int lo = -8, int hi = 7,
                                      uint32_t seed = 13) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(lo, hi);
  std::vector<int8_t> v(n);
  for (auto &x : v)
    x = (int8_t)dist(rng);
  return v;
}

// C[M,N] = X[M,K] (u8) · W[N,K] (i8)^T, exact int32.
static void cpuGemmI32(int M, int N, int K, const uint8_t *X, const int8_t *W,
                       int32_t *C) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      int32_t acc = 0;
      for (int k = 0; k < K; ++k)
        acc += (int32_t)X[m * K + k] * (int32_t)W[n * K + k];
      C[m * N + n] = acc;
    }
}

static bool exactMatchI32(const int32_t *a, const int32_t *b, int n) {
  for (int i = 0; i < n; ++i)
    if (a[i] != b[i])
      return false;
  return true;
}

// ---- NPU buffer RAII -------------------------------------------------------

/**
 * @brief RAII wrapper for an NPU-allocated buffer; frees itself via
 *        sdkl_npu_free when it goes out of scope.
 */
struct NpuBuf {
  void *p = nullptr;
  NpuBuf() = default;
  explicit NpuBuf(size_t bytes) {
    if (sdkl_npu_alloc(bytes, &p) != 0)
      p = nullptr;
  }
  ~NpuBuf() {
    if (p)
      sdkl_npu_free(p);
  }
  NpuBuf(const NpuBuf &) = delete;
  NpuBuf &operator=(const NpuBuf &) = delete;
  bool ok() const { return p != nullptr; }
};

// ---- Fixture ---------------------------------------------------------------

/**
 * @brief Fixture for direct sdkl kernel accuracy/perf tests; captures NPU
 *        availability and the HTP compute domain for use by test bodies.
 */
class HtpKernelTest : public ::testing::Test {
protected:
  void SetUp() override {
    npu_enabled = nntrainer::HtpBackend::global().enabled();
    domain = nntrainer::HtpBackend::global().domain();
  }
  bool npu_enabled = false;
  int domain = 0;
};

// ---- u8i8_i32: ui8 activations, i8 weights (WH), i32 output ----------------

TEST_F(HtpKernelTest, Accuracy_u8i8_i32) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  // HMX INT8 requires M % 64 == 0 and N % 32 == 0 (observed on V79).
  // sdkl_npu_mm_u8i8_i32 writes row-major output directly (per sdkl.h:696).
  constexpr int M = 64, N = 64, K = 128;
  auto X = makeRandU8(M * K);
  auto W = makeRandI8(N * K);
  std::vector<int32_t> C_cpu(M * N, 0);
  cpuGemmI32(M, N, K, X.data(), W.data(), C_cpu.data());

  NpuBuf Xb(M * K * sizeof(uint8_t)), Wb(N * K * sizeof(int8_t)),
    Ab(M * N * sizeof(int32_t));
  ASSERT_TRUE(Xb.ok() && Wb.ok() && Ab.ok());

  std::memcpy(Xb.p, X.data(), M * K * sizeof(uint8_t));
  std::memcpy(Wb.p, W.data(), N * K * sizeof(int8_t));

  ASSERT_EQ(sdkl_cpu_rm_to_wh_i8_inplace((size_t)N, (size_t)K,
                                         static_cast<int8_t *>(Wb.p)),
            0);
  ASSERT_EQ(sdkl_npu_mm_u8i8_i32(domain, M, N, K, static_cast<int32_t *>(Ab.p),
                                 static_cast<const uint8_t *>(Xb.p),
                                 static_cast<const int8_t *>(Wb.p)),
            0);

  // Output is already row-major (sdkl.h:696); compare directly.
  EXPECT_TRUE(
    exactMatchI32(static_cast<const int32_t *>(Ab.p), C_cpu.data(), M * N))
    << "u8i8_i32 output does not match int32 reference exactly";
}

// ---- Performance sweeps ----------------------------------------------------

static void perfSweepU8I8(int domain, const std::vector<Shape> &shapes,
                          const char *tag) {
  for (const auto &s : shapes) {
    const int M = s.M, N = s.N, K = s.K;
    // HMX INT8 requires M % 64 == 0.
    if (M % 64 != 0) {
      printf("SKIP %s u8i8 %dx%dx%d: M not multiple of 64\n", tag, M, N, K);
      continue;
    }
    auto X = makeRandU8(M * K);
    auto W = makeRandI8(N * K);
    std::vector<int32_t> C_cpu(M * N, 0);
    NpuBuf Xb(M * K), Wb(N * K), Ab(M * N * sizeof(int32_t));
    if (!(Xb.ok() && Wb.ok() && Ab.ok())) {
      printf("SKIP %s u8i8 %dx%dx%d: alloc failed\n", tag, M, N, K);
      continue;
    }
    std::memcpy(Xb.p, X.data(), M * K);
    std::memcpy(Wb.p, W.data(), N * K);
    sdkl_cpu_rm_to_wh_i8_inplace((size_t)N, (size_t)K,
                                 static_cast<int8_t *>(Wb.p));
    int last_rc = 0;
    auto kernelOnly = [&]() {
      last_rc = sdkl_npu_mm_u8i8_i32(
        domain, M, N, K, static_cast<int32_t *>(Ab.p),
        static_cast<const uint8_t *>(Xb.p), static_cast<const int8_t *>(Wb.p));
    };
    TimeStats k = timeIt(10, 50, kernelOnly);
    ASSERT_EQ(last_rc, 0) << "u8i8_i32 kernel returned non-zero for shape " << M
                          << "x" << N << "x" << K;
    TimeStats cpu = timeIt(
      1, 3, [&]() { cpuGemmI32(M, N, K, X.data(), W.data(), C_cpu.data()); });
    recordPerf("u8i8_i32", M, N, K, k, k, cpu.mean_ms, true); // quant=true
  }
}

TEST_F(HtpKernelTest, Perf_u8i8_i32) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";
  perfSweepU8I8(domain, generalShapes(), "general");
  perfSweepU8I8(domain, llmShapes(), "llm");
  SUCCEED();
}

// ---- M-padding: arbitrary M (not multiple of 32) must work ------------------

TEST_F(HtpKernelTest, Padding_NonMultipleOf32_f32f16_f32) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";
  for (int M : {1, 33}) {
    const int N = 64, K = 128;
    std::vector<float> A = makeRandF32(M * K);
    std::vector<_FP16> W = makeRandF16(N * K); // weight [N,K] RM
    std::vector<float> C(M * N, 0.f), ref(M * N, 0.f);
    cpuGemmF32F16(M, N, K, A.data(), W.data(), ref.data()); // C = A * W^T
    nntrainer::hmx::shgemm_f32f16_f32(
      0, false, true, M, N, K, 1.0f, A.data(), K,
      reinterpret_cast<const __fp16 *>(W.data()), K, 0.0f, C.data(), N);
    float e = relErrorF32(C.data(), ref.data(), M * N);
    EXPECT_LT(e, 1e-2f) << "M=" << M << " relError=" << e;
  }
}

TEST_F(HtpKernelTest, Accuracy_f32f16_f32_Prefill) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  struct PrefillShape {
    int M, N, K;
    const char *name;
  };
  // Qwen3-0.6B prefill shapes (M=16 = init_seq_len).
  // N values are multiples of 32 (sdk requirement).
  const PrefillShape shapes[] = {
    {16, 2048, 1024, "q_proj"},    {16, 1024, 1024, "kv_proj"},
    {16, 1024, 2048, "o_proj"},    {16, 3072, 1024, "gate_up_proj"},
    {16, 1024, 3072, "down_proj"},
  };
  for (const auto &s : shapes) {
    std::vector<float> A = makeRandF32(s.M * s.K);
    std::vector<_FP16> W = makeRandF16(s.N * s.K);
    std::vector<float> C(s.M * s.N, 0.f), ref(s.M * s.N, 0.f);
    cpuGemmF32F16(s.M, s.N, s.K, A.data(), W.data(), ref.data());
    nntrainer::hmx::shgemm_f32f16_f32(
      0, false, true, s.M, s.N, s.K, 1.0f, A.data(), s.K,
      reinterpret_cast<const __fp16 *>(W.data()), s.K, 0.0f, C.data(), s.N);
    float e = relErrorF32(C.data(), ref.data(), s.M * s.N);
    EXPECT_LT(e, 1e-2f) << s.name << " M=" << s.M << " N=" << s.N
                        << " K=" << s.K << " relError=" << e;
  }
}

// Reproduces the ACTUAL production calling convention used by
// FullyConnectedLayer / SharedFullyConnectedLayer for wq/wk/wv/wo and the FFN
// gate/up/down projections: `input_.dot(weight, hidden_, false, false)`, i.e.
// TransA=false, TransB=false. The FC weight tensor is constructed as
// TensorDim(1, 1, in_dim.width(), unit) — physically stored ROW-MAJOR AS
// [K, N] (K rows of N-wide vectors), NOT [N, K] like
// Accuracy_f32f16_f32_Prefill above (which always passes TransB=true and
// builds W as [N,K]). Unlike that test, this one matches what real Qwen3
// prefill actually sends to hexkl_mm.cpp.
TEST_F(HtpKernelTest, Accuracy_f32f16_f32_Prefill_TransBFalse_KNLayout) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  // The pin cache keys on raw pointer address only (production weight
  // pointers are stable for a model's lifetime). Clear it so a freed
  // std::vector from an earlier test can't alias this test's buffers via
  // heap address reuse and return a stale WH entry baked under a different
  // TransB convention.
  nntrainer::hmx::prefillWHCacheClear();

  struct PrefillShape {
    int M, N, K;
    const char *name;
  };
  // Same Qwen3-0.6B prefill shapes as Accuracy_f32f16_f32_Prefill.
  const PrefillShape shapes[] = {
    {16, 2048, 1024, "q_proj"},    {16, 1024, 1024, "kv_proj"},
    {16, 1024, 2048, "o_proj"},    {16, 3072, 1024, "gate_up_proj"},
    {16, 1024, 3072, "down_proj"},
  };
  for (const auto &s : shapes) {
    std::vector<float> A = makeRandF32(s.M * s.K);
    // Weight stored [K, N] row-major — matches the real FC layer layout
    // (TensorDim height=K, width=N).
    std::vector<_FP16> W_kn = makeRandF16(s.K * s.N);
    std::vector<float> C(s.M * s.N, 0.f), ref(s.M * s.N, 0.f);

    // CPU reference: C[m,n] = sum_k A[m,k] * W_kn[k,n] — no transpose,
    // matching dot(..., trans=false, trans_in=false) semantics exactly.
    for (int m = 0; m < s.M; ++m)
      for (int n = 0; n < s.N; ++n) {
        float acc = 0.f;
        for (int k = 0; k < s.K; ++k)
          acc += A[m * s.K + k] * (float)W_kn[k * s.N + n];
        ref[m * s.N + n] = acc;
      }

    // Production calling convention: TransB=false, ldb = weight's own last
    // axis = N (see TensorBase::calculateFlattenDot, !trans && !trans_in
    // branch: ldb = input_last_axis).
    nntrainer::hmx::shgemm_f32f16_f32(
      0, false, false, s.M, s.N, s.K, 1.0f, A.data(), s.K,
      reinterpret_cast<const __fp16 *>(W_kn.data()), s.N, 0.0f, C.data(), s.N);
    float e = relErrorF32(C.data(), ref.data(), s.M * s.N);
    EXPECT_LT(e, 1e-2f) << s.name << " M=" << s.M << " N=" << s.N
                        << " K=" << s.K << " relError=" << e;
  }
}

TEST_F(HtpKernelTest, PrefillWHResidency_ReusesCacheAcrossCalls) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  nntrainer::hmx::prefillWHCacheClear();
  ASSERT_EQ(nntrainer::hmx::prefillWHCacheSize(), 0u);

  const int M = 16, N = 1024, K = 1024;
  std::vector<float> A = makeRandF32(M * K);
  std::vector<_FP16> W = makeRandF16(N * K);
  std::vector<float> C(M * N, 0.f), ref(M * N, 0.f);
  cpuGemmF32F16(M, N, K, A.data(), W.data(), ref.data());

  auto call = [&]() {
    nntrainer::hmx::shgemm_f32f16_f32(
      0, false, true, M, N, K, 1.0f, A.data(), K,
      reinterpret_cast<const __fp16 *>(W.data()), K, 0.0f, C.data(), N);
  };

  // First call: populates one resident WH entry, result correct.
  call();
  EXPECT_EQ(nntrainer::hmx::prefillWHCacheSize(), 1u);
  EXPECT_LT(relErrorF32(C.data(), ref.data(), M * N), 1e-2f);

  // Second call, SAME weight pointer: cache hit, no new entry, still correct.
  std::fill(C.begin(), C.end(), 0.f);
  call();
  EXPECT_EQ(nntrainer::hmx::prefillWHCacheSize(), 1u);
  EXPECT_LT(relErrorF32(C.data(), ref.data(), M * N), 1e-2f);

  nntrainer::hmx::prefillWHCacheClear();
  EXPECT_EQ(nntrainer::hmx::prefillWHCacheSize(), 0u);
}

// Pin-once: distinct weight pointers each add a resident entry, and entries
// are never evicted when re-touched (the failure mode of the old FIFO cache
// on a sequential scan). Uses three small weights that fit well under the cap.
TEST_F(HtpKernelTest, PrefillWHResidency_PinsMultipleNeverEvicts) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  nntrainer::hmx::prefillWHCacheClear();
  ASSERT_EQ(nntrainer::hmx::prefillWHCacheSize(), 0u);

  const int M = 16, N = 1024, K = 1024;
  // Three distinct weight buffers => three distinct cache keys (pointers).
  std::vector<_FP16> W0 = makeRandF16(N * K, -0.5f, 0.5f, 1);
  std::vector<_FP16> W1 = makeRandF16(N * K, -0.5f, 0.5f, 2);
  std::vector<_FP16> W2 = makeRandF16(N * K, -0.5f, 0.5f, 3);
  std::vector<float> A = makeRandF32(M * K);
  std::vector<float> C(M * N, 0.f);

  auto call = [&](const std::vector<_FP16> &W) {
    nntrainer::hmx::shgemm_f32f16_f32(
      0, false, true, M, N, K, 1.0f, A.data(), K,
      reinterpret_cast<const __fp16 *>(W.data()), K, 0.0f, C.data(), N);
  };

  call(W0);
  call(W1);
  call(W2);
  EXPECT_EQ(nntrainer::hmx::prefillWHCacheSize(), 3u);

  // Re-touch the first weight: must be a hit, size unchanged (no eviction).
  call(W0);
  EXPECT_EQ(nntrainer::hmx::prefillWHCacheSize(), 3u);

  nntrainer::hmx::prefillWHCacheClear();
  EXPECT_EQ(nntrainer::hmx::prefillWHCacheSize(), 0u);
}

// Diagnostic: break the transient prefill path into phases and time each, to
// verify that weight staging (alloc + memcpy + rm_to_wh) — not the matmul or
// copy-back — dominates. Mirrors the M>1 transient branch of shgemm_f32f16_f32
// without touching the library. Not a pass/fail gate: always SUCCEED().
// A registered pre-baked WH weight must be used by shgemm's prefill path and
// produce the same result as the transient rm_to_wh path. We build the WH bytes
// by running rm_to_wh ourselves, register them, and confirm (a) the registry
// reports the entry and (b) shgemm output matches an unregistered reference
// run.
TEST_F(HtpKernelTest, PrefillWHRegistry_UsedByShgemmMatchesTransient) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  nntrainer::hmx::prefillWHRegistryClear();
  ASSERT_EQ(nntrainer::hmx::prefillWHRegistrySize(), 0u);

  const int M = 16, N = 1024, K = 1024;
  std::vector<_FP16> W = makeRandF16(N * K, -0.5f, 0.5f, 7);
  std::vector<float> A = makeRandF32(M * K);
  std::vector<float> C_ref(M * N, 0.f), C_reg(M * N, 0.f);

  auto run = [&](std::vector<float> &C) {
    nntrainer::hmx::shgemm_f32f16_f32(
      0, false, true, M, N, K, 1.0f, A.data(), K,
      reinterpret_cast<const __fp16 *>(W.data()), K, 0.0f, C.data(), N);
  };

  // Reference: no registration -> transient rm_to_wh path.
  run(C_ref);

  // Build WH bytes on host by mirroring the SDK conversion, then register.
  std::vector<_FP16> WH = W; // copy RM
  ASSERT_EQ(sdkl_cpu_rm_to_wh_f16_inplace(
              (size_t)N, (size_t)K, reinterpret_cast<_Float16 *>(WH.data())),
            0);
  nntrainer::hmx::registerPrefillWH(reinterpret_cast<const void *>(W.data()), N,
                                    K, WH.data());
  EXPECT_EQ(nntrainer::hmx::prefillWHRegistrySize(), 1u);

  // Registered run must hit the pre-baked path and match the reference.
  run(C_reg);
  for (int i = 0; i < M * N; ++i)
    ASSERT_NEAR(C_reg[i], C_ref[i], 1e-2f) << "mismatch at " << i;

  nntrainer::hmx::prefillWHRegistryClear();
  EXPECT_EQ(nntrainer::hmx::prefillWHRegistrySize(), 0u);
}

// Scratch reuse must not change results even when call shapes vary between
// calls (a larger shape then a smaller one must both be correct). We compare
// each shgemm result against a fresh CPU reference (nntrainer::shgemm).
TEST_F(HtpKernelTest, ScratchReuse_MixedShapesStayCorrect) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";
  nntrainer::hmx::prefillWHRegistryClear();
  nntrainer::hmx::prefillWHCacheClear();

  struct S {
    int M, N, K;
  };
  const S shapes[] = {{16, 2048, 1024}, {16, 1024, 1024}, {16, 3072, 1024}};
  for (const auto &s : shapes) {
    std::vector<_FP16> W = makeRandF16(s.N * s.K, -0.3f, 0.3f, s.N + s.K);
    std::vector<float> A = makeRandF32(s.M * s.K);
    std::vector<float> C_npu(s.M * s.N, 0.f), C_cpu(s.M * s.N, 0.f);

    nntrainer::hmx::shgemm_f32f16_f32(
      0, false, true, s.M, s.N, s.K, 1.0f, A.data(), s.K,
      reinterpret_cast<const __fp16 *>(W.data()), s.K, 0.0f, C_npu.data(), s.N);
    nntrainer::shgemm(0, false, true, s.M, s.N, s.K, 1.0f, A.data(), s.K,
                      reinterpret_cast<const __fp16 *>(W.data()), s.K, 0.0f,
                      C_cpu.data(), s.N);

    for (int i = 0; i < s.M * s.N; ++i)
      ASSERT_NEAR(C_npu[i], C_cpu[i], 5e-2f)
        << "shape " << s.N << "x" << s.K << " idx " << i;
  }
}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  return rc;
}

#else
#include <gtest/gtest.h>
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif // ENABLE_FP16
#else
// ENABLE_HEXKL not set — provide a no-op main so the binary links on x86.
#include <gtest/gtest.h>
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif // ENABLE_HEXKL
