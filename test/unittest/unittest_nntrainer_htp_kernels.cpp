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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

// ---- u8i4_i32: ui8 activations, i4 weights (WH), i32 output ----------------

TEST_F(HtpKernelTest, Accuracy_u8i4_i32) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  // HMX INT4 shares the u8i8 tiling: M % 64 == 0 and N % 32 == 0.
  // sdkl_npu_mm_u8i4_i32 takes the weight as a packed uint8 in WH layout and
  // writes row-major output (per sdkl.h). Weights are drawn from the valid i4
  // range [-7, 7], stored one value per int8 byte (sign-extended), which is
  // the documented input contract for sdkl_cpu_rm_to_wh_i4.
  constexpr int M = 64, N = 64, K = 128;
  auto X = makeRandU8(M * K);
  auto W = makeRandI8(N * K, -7, 7);
  std::vector<int32_t> C_cpu(M * N, 0);
  cpuGemmI32(M, N, K, X.data(), W.data(), C_cpu.data());

  // Over-allocate the WH output to the unpacked size so any HMX tile padding
  // beyond N*K/2 cannot overflow. rm_to_wh_i4 is NOT in-place: it reads the
  // i4-in-i8 input and writes the packed WH bytes to a separate buffer.
  const size_t w_unpacked = (size_t)N * (size_t)K * sizeof(int8_t);
  NpuBuf Xb(M * K * sizeof(uint8_t)), Win(w_unpacked), Wwh(w_unpacked),
    Ab(M * N * sizeof(int32_t));
  ASSERT_TRUE(Xb.ok() && Win.ok() && Wwh.ok() && Ab.ok());

  std::memcpy(Xb.p, X.data(), M * K * sizeof(uint8_t));
  std::memcpy(Win.p, W.data(), w_unpacked);

  // Device-verified: sdkl_cpu_rm_to_wh_i4 expects (wt_rows=K, wt_cols=N) — the
  // inner dimension first — to produce a WH weight that sdkl_npu_mm_u8i4_i32
  // consumes as C = X * W^T for a logically [N, K] weight. (Passing (N, K),
  // the order used by the in-place i8 variant, yields wrong values.)
  ASSERT_EQ(sdkl_cpu_rm_to_wh_i4(static_cast<uint8_t *>(Wwh.p),
                                 static_cast<int8_t *>(Win.p), (size_t)K,
                                 (size_t)N),
            0);
  ASSERT_EQ(sdkl_npu_mm_u8i4_i32(domain, M, N, K, static_cast<int32_t *>(Ab.p),
                                 static_cast<const uint8_t *>(Xb.p),
                                 static_cast<const uint8_t *>(Wwh.p)),
            0);

  // C_cpu = X * W^T (W as [N, K]); output is row-major (per sdkl.h).
  EXPECT_TRUE(
    exactMatchI32(static_cast<const int32_t *>(Ab.p), C_cpu.data(), M * N))
    << "u8i4_i32 output does not match int32 reference exactly";
}

// ---- Full-pipeline u8i4 accuracy -------------------------------------------
// Mirrors quantize_qint4_weight (per-output-channel i4 weight in [-7,7] +
// zp_corr) and shgemm_u8i4_i32 (per-tensor u8 activation with zp=128, M padded
// to 64, dequant) exactly, at realistic non-square / small-M LLM shapes the
// square integer-only test cannot exercise. Returns the relative error vs the
// FP32 reference so a numeric bug (huge error) is distinguishable from INT4
// quantization loss (small, bounded error). Negative return = setup failure.
static float u8i4FullPipelineRelErr(int M, int N, int K, int domain,
                                    uint32_t seed) {
  const int Mp = ((M + 63) / 64) * 64;
  auto A = makeRandF32(M * K, -1.0f, 1.0f, seed);
  auto W = makeRandF32(N * K, -0.5f, 0.5f, seed + 1);

  // FP32 reference: C[m,n] = sum_k A[m,k] * W[n,k].
  std::vector<float> C_ref((size_t)M * N, 0.0f);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k)
        acc += A[(size_t)m * K + k] * W[(size_t)n * K + k];
      C_ref[(size_t)m * N + n] = acc;
    }

  // Quantize weight per-output-channel to i4 [-7,7]; compute zp_corr.
  std::vector<int8_t> Wi4((size_t)N * K, 0);
  std::vector<float> wt_scale(N, 1.0f);
  std::vector<int32_t> zp_corr(N, 0);
  for (int n = 0; n < N; ++n) {
    float maxabs = 0.0f;
    for (int k = 0; k < K; ++k)
      maxabs = std::max(maxabs, std::abs(W[(size_t)n * K + k]));
    float s = maxabs > 0.0f ? maxabs / 7.0f : 1.0f;
    wt_scale[n] = s;
    int32_t rowsum = 0;
    for (int k = 0; k < K; ++k) {
      long q = std::lround(W[(size_t)n * K + k] / s);
      int8_t c = (int8_t)std::max<long>(-7, std::min<long>(7, q));
      Wi4[(size_t)n * K + k] = c;
      rowsum += c;
    }
    zp_corr[n] = 128 * rowsum;
  }

  // Pack i4 weight to WH layout (rm_to_wh_i4 takes (wt_rows=K, wt_cols=N)).
  const size_t w_unpacked = (size_t)N * K;
  NpuBuf Win(w_unpacked), Wwh(w_unpacked);
  if (!Win.ok() || !Wwh.ok())
    return -1.0f;
  std::memcpy(Win.p, Wi4.data(), w_unpacked);
  if (sdkl_cpu_rm_to_wh_i4(static_cast<uint8_t *>(Wwh.p),
                           static_cast<int8_t *>(Win.p), (size_t)K,
                           (size_t)N) != 0)
    return -2.0f;

  // Quantize activation per-tensor to u8 (zp=128); pad to Mp rows with 128.
  float amax = 0.0f;
  for (int i = 0; i < M * K; ++i)
    amax = std::max(amax, std::abs(A[i]));
  float act_scale = amax > 0.0f ? amax / 127.0f : 1.0f;
  std::vector<uint8_t> Xu8((size_t)Mp * K, 128);
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k) {
      float q = std::round(A[(size_t)m * K + k] / act_scale) + 128.0f;
      q = std::max(0.0f, std::min(255.0f, q));
      Xu8[(size_t)m * K + k] = (uint8_t)q;
    }

  NpuBuf Xb((size_t)Mp * K * sizeof(uint8_t)),
    Cb((size_t)Mp * N * sizeof(int32_t));
  if (!Xb.ok() || !Cb.ok())
    return -3.0f;
  std::memcpy(Xb.p, Xu8.data(), (size_t)Mp * K * sizeof(uint8_t));
  std::memset(Cb.p, 0, (size_t)Mp * N * sizeof(int32_t));
  if (sdkl_npu_mm_u8i4_i32(domain, Mp, N, K, static_cast<int32_t *>(Cb.p),
                           static_cast<const uint8_t *>(Xb.p),
                           static_cast<const uint8_t *>(Wwh.p)) != 0)
    return -4.0f;

  // Dequant: C[m,n] = act_scale * wt_scale[n] * (C_i32[m,n] - zp_corr[n]).
  const int32_t *Ci = static_cast<const int32_t *>(Cb.p);
  std::vector<float> C((size_t)M * N, 0.0f);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n)
      C[(size_t)m * N + n] = act_scale * wt_scale[n] *
                             ((float)Ci[(size_t)m * N + n] - (float)zp_corr[n]);

  return relErrorF32(C.data(), C_ref.data(), M * N);
}

TEST_F(HtpKernelTest, Accuracy_u8i4_full_pipeline) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  struct Case {
    const char *name;
    int M, N, K;
  };
  // Qwen3-0.6B FC shapes (hidden 1024, intermediate 3072, GQA q=2048/kv=1024).
  const std::vector<Case> cases = {
    {"prefill_wq", 5, 2048, 1024},       {"decode_wq", 1, 2048, 1024},
    {"prefill_wk", 5, 1024, 1024},       {"prefill_ffn_up", 5, 3072, 1024},
    {"prefill_ffn_down", 5, 1024, 3072}, {"decode_ffn_down", 1, 1024, 3072},
  };
  for (const auto &c : cases) {
    float e = u8i4FullPipelineRelErr(c.M, c.N, c.K, domain,
                                     100u + (uint32_t)(c.M * 131 + c.N));
    std::printf("[u8i4 full-pipeline] %-18s M=%4d N=%4d K=%4d  relErr=%.4f\n",
                c.name, c.M, c.N, c.K, e);
    std::fflush(stdout);
    ASSERT_GE(e, 0.0f) << c.name << ": kernel/setup failed (rc=" << e << ")";
    // INT4 loss over these K is expected but bounded; a layout/scale/padding
    // bug pushes the error far past quantization noise.
    EXPECT_LT(e, 0.30f) << c.name << ": relative error too high — likely a bug,"
                        << " not just INT4 quantization loss";
  }
}

// ---- u8i8 full pipeline: same as u8i4 but 8-bit weights --------------------

// Mirror of u8i4FullPipelineRelErr for the u8i8 path: per-output-channel INT8
// weight ([-127,127], scale = max_abs/127, zp_corr = 128*sum), WH-packed via
// the in-place i8 converter, per-tensor u8 activation, kernel, dequant. Same
// fc_layer shapes so the u8i4 and u8i8 HexKL kernels can be compared directly
// (this is the HexKL half of the fc_layer mm HexKL-vs-QNN comparison; QNN is
// fed the identical quantized operands on device). Negative return = setup
// failure.
static float u8i8FullPipelineRelErr(int M, int N, int K, int domain,
                                    uint32_t seed) {
  const int Mp = ((M + 63) / 64) * 64; // HMX INT8 requires M % 64 == 0
  auto A = makeRandF32(M * K, -1.0f, 1.0f, seed);
  auto W = makeRandF32(N * K, -0.5f, 0.5f, seed + 1);

  // FP32 reference: C[m,n] = sum_k A[m,k] * W[n,k].
  std::vector<float> C_ref((size_t)M * N, 0.0f);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k)
        acc += A[(size_t)m * K + k] * W[(size_t)n * K + k];
      C_ref[(size_t)m * N + n] = acc;
    }

  // Quantize weight per-output-channel to i8 [-127,127]; compute zp_corr.
  std::vector<int8_t> Wi8((size_t)N * K, 0);
  std::vector<float> wt_scale(N, 1.0f);
  std::vector<int32_t> zp_corr(N, 0);
  for (int n = 0; n < N; ++n) {
    float maxabs = 0.0f;
    for (int k = 0; k < K; ++k)
      maxabs = std::max(maxabs, std::abs(W[(size_t)n * K + k]));
    float s = maxabs > 0.0f ? maxabs / 127.0f : 1.0f;
    wt_scale[n] = s;
    int32_t rowsum = 0;
    for (int k = 0; k < K; ++k) {
      long q = std::lround(W[(size_t)n * K + k] / s);
      int8_t c = (int8_t)std::max<long>(-127, std::min<long>(127, q));
      Wi8[(size_t)n * K + k] = c;
      rowsum += c;
    }
    zp_corr[n] = 128 * rowsum;
  }

  // Pack i8 weight to WH layout in place (rm_to_wh_i8_inplace takes (N, K)).
  const size_t w_bytes = (size_t)N * K;
  NpuBuf Wb(w_bytes);
  if (!Wb.ok())
    return -1.0f;
  std::memcpy(Wb.p, Wi8.data(), w_bytes);
  if (sdkl_cpu_rm_to_wh_i8_inplace((size_t)N, (size_t)K,
                                   static_cast<int8_t *>(Wb.p)) != 0)
    return -2.0f;

  // Quantize activation per-tensor to u8 (zp=128); pad to Mp rows with 128.
  float amax = 0.0f;
  for (int i = 0; i < M * K; ++i)
    amax = std::max(amax, std::abs(A[i]));
  float act_scale = amax > 0.0f ? amax / 127.0f : 1.0f;
  std::vector<uint8_t> Xu8((size_t)Mp * K, 128);
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k) {
      float q = std::round(A[(size_t)m * K + k] / act_scale) + 128.0f;
      q = std::max(0.0f, std::min(255.0f, q));
      Xu8[(size_t)m * K + k] = (uint8_t)q;
    }

  NpuBuf Xb((size_t)Mp * K * sizeof(uint8_t)),
    Cb((size_t)Mp * N * sizeof(int32_t));
  if (!Xb.ok() || !Cb.ok())
    return -3.0f;
  std::memcpy(Xb.p, Xu8.data(), (size_t)Mp * K * sizeof(uint8_t));
  std::memset(Cb.p, 0, (size_t)Mp * N * sizeof(int32_t));
  if (sdkl_npu_mm_u8i8_i32(domain, Mp, N, K, static_cast<int32_t *>(Cb.p),
                           static_cast<const uint8_t *>(Xb.p),
                           static_cast<const int8_t *>(Wb.p)) != 0)
    return -4.0f;

  // Dequant: C[m,n] = act_scale * wt_scale[n] * (C_i32[m,n] - zp_corr[n]).
  const int32_t *Ci = static_cast<const int32_t *>(Cb.p);
  std::vector<float> C((size_t)M * N, 0.0f);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n)
      C[(size_t)m * N + n] = act_scale * wt_scale[n] *
                             ((float)Ci[(size_t)m * N + n] - (float)zp_corr[n]);

  return relErrorF32(C.data(), C_ref.data(), M * N);
}

TEST_F(HtpKernelTest, Accuracy_u8i8_full_pipeline) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  struct Case {
    const char *name;
    int M, N, K;
  };
  const std::vector<Case> cases = {
    {"prefill_wq", 5, 2048, 1024},       {"decode_wq", 1, 2048, 1024},
    {"prefill_wk", 5, 1024, 1024},       {"prefill_ffn_up", 5, 3072, 1024},
    {"prefill_ffn_down", 5, 1024, 3072}, {"decode_ffn_down", 1, 1024, 3072},
  };
  for (const auto &c : cases) {
    float e = u8i8FullPipelineRelErr(c.M, c.N, c.K, domain,
                                     200u + (uint32_t)(c.M * 131 + c.N));
    std::printf("[u8i8 full-pipeline] %-18s M=%4d N=%4d K=%4d  relErr=%.4f\n",
                c.name, c.M, c.N, c.K, e);
    std::fflush(stdout);
    ASSERT_GE(e, 0.0f) << c.name << ": kernel/setup failed (rc=" << e << ")";
    // INT8 is far tighter than INT4; a few % worst case.
    EXPECT_LT(e, 0.05f) << c.name << ": relative error too high for INT8";
  }
}
// ---- Performance sweeps ----------------------------------------------------

// ---- fc_layer mm-level comparison: HexKL u8i4 vs u8i8 ----------------------

/**
 * @brief Mean/min timing statistics over repeated calls, in milliseconds.
 */
struct TimeStats {
  double mean_ms, min_ms;
};

template <typename Fn> static TimeStats timeIt(int warmup, int iters, Fn &&fn) {
  for (int i = 0; i < warmup; ++i)
    fn();
  double sum = 0.0, mn = 0.0;
  for (int i = 0; i < iters; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    fn();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    sum += ms;
    mn = (i == 0) ? ms : std::min(mn, ms);
  }
  return {sum / iters, mn};
}

// Times the u8i4 and u8i8 HexKL kernels on the same fc_layer mm shapes with
// identical (matched) quantization, printing kernel latency and relErr vs the
// FP32 reference side by side. This is the HexKL side of the fc_layer
// HexKL-vs-QNN comparison; the QNN column is filled in on a device with the
// QNN SDK by feeding QNN the same quantized operands (see
// docs/backend_guide/htp_backend/06_fc_mm_hexkl_vs_qnn.md). Kernel-only timing
// hoists all allocation/quantization/WH packing out of the measured loop so
// only the sdkl_npu_mm_* call is timed.
TEST_F(HtpKernelTest, FcMm_Compare_HexKL_u8i4_u8i8) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  struct Case {
    const char *name;
    int M, N, K;
  };
  const std::vector<Case> cases = {
    {"prefill_wq", 64, 2048, 1024},     {"prefill_wk", 64, 1024, 1024},
    {"prefill_ffn_up", 64, 3072, 1024}, {"prefill_ffn_down", 64, 1024, 3072},
    {"decode_wq", 1, 2048, 1024},       {"decode_ffn_down", 1, 1024, 3072},
  };

  std::printf("\n| shape | M | N | K | u8i4 ms | u8i4 relErr | u8i8 ms | "
              "u8i8 relErr |\n");
  std::printf("|---|---|---|---|---|---|---|---|\n");

  for (const auto &c : cases) {
    const int M = c.M, N = c.N, K = c.K;
    const int Mp = ((M + 63) / 64) * 64;

    // Shared operands (identical for both kernels).
    auto A = makeRandF32(M * K, -1.0f, 1.0f, 900u + (uint32_t)(c.N));
    auto W = makeRandF32(N * K, -0.5f, 0.5f, 901u + (uint32_t)(c.N));

    // --- u8i4 kernel-only timing ---
    double u8i4_ms = -1.0;
    float u8i4_err = u8i4FullPipelineRelErr(M, N, K, domain, 900u + c.N);
    {
      std::vector<int8_t> Wi4((size_t)N * K, 0);
      for (int n = 0; n < N; ++n) {
        float mx = 0.0f;
        for (int k = 0; k < K; ++k)
          mx = std::max(mx, std::abs(W[(size_t)n * K + k]));
        float s = mx > 0.0f ? mx / 7.0f : 1.0f;
        for (int k = 0; k < K; ++k) {
          long q = std::lround(W[(size_t)n * K + k] / s);
          Wi4[(size_t)n * K + k] =
            (int8_t)std::max<long>(-7, std::min<long>(7, q));
        }
      }
      NpuBuf Win((size_t)N * K), Wwh((size_t)N * K), Xb((size_t)Mp * K),
        Cb((size_t)Mp * N * sizeof(int32_t));
      bool packed = false;
      if (Win.ok() && Wwh.ok() && Xb.ok() && Cb.ok()) {
        std::memcpy(Win.p, Wi4.data(), (size_t)N * K);
        packed = sdkl_cpu_rm_to_wh_i4(static_cast<uint8_t *>(Wwh.p),
                                      static_cast<int8_t *>(Win.p), (size_t)K,
                                      (size_t)N) == 0;
      }
      if (packed) {
        std::memset(Xb.p, 128, (size_t)Mp * K);
        int rc = 0;
        auto call = [&]() {
          rc =
            sdkl_npu_mm_u8i4_i32(domain, Mp, N, K, static_cast<int32_t *>(Cb.p),
                                 static_cast<const uint8_t *>(Xb.p),
                                 static_cast<const uint8_t *>(Wwh.p));
        };
        TimeStats t = timeIt(5, 30, call);
        if (rc == 0)
          u8i4_ms = t.mean_ms;
      }
    }

    // --- u8i8 kernel-only timing ---
    double u8i8_ms = -1.0;
    float u8i8_err = u8i8FullPipelineRelErr(M, N, K, domain, 902u + c.N);
    {
      std::vector<int8_t> Wi8((size_t)N * K, 0);
      for (int n = 0; n < N; ++n) {
        float mx = 0.0f;
        for (int k = 0; k < K; ++k)
          mx = std::max(mx, std::abs(W[(size_t)n * K + k]));
        float s = mx > 0.0f ? mx / 127.0f : 1.0f;
        for (int k = 0; k < K; ++k) {
          long q = std::lround(W[(size_t)n * K + k] / s);
          Wi8[(size_t)n * K + k] =
            (int8_t)std::max<long>(-127, std::min<long>(127, q));
        }
      }
      NpuBuf Wb((size_t)N * K), Xb((size_t)Mp * K),
        Cb((size_t)Mp * N * sizeof(int32_t));
      if (Wb.ok() && Xb.ok() && Cb.ok()) {
        std::memcpy(Wb.p, Wi8.data(), (size_t)N * K);
        sdkl_cpu_rm_to_wh_i8_inplace((size_t)N, (size_t)K,
                                     static_cast<int8_t *>(Wb.p));
        std::memset(Xb.p, 128, (size_t)Mp * K);
        int rc = 0;
        auto call = [&]() {
          rc =
            sdkl_npu_mm_u8i8_i32(domain, Mp, N, K, static_cast<int32_t *>(Cb.p),
                                 static_cast<const uint8_t *>(Xb.p),
                                 static_cast<const int8_t *>(Wb.p));
        };
        TimeStats t = timeIt(5, 30, call);
        if (rc == 0)
          u8i8_ms = t.mean_ms;
      }
    }

    std::printf("| %s | %d | %d | %d | %.4f | %.4f | %.4f | %.4f |\n", c.name,
                M, N, K, u8i4_ms, u8i4_err, u8i8_ms, u8i8_err);
    std::fflush(stdout);
    EXPECT_GE(u8i4_err, 0.0f) << c.name << ": u8i4 setup failed";
    EXPECT_GE(u8i8_err, 0.0f) << c.name << ": u8i8 setup failed";
  }
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

// The offline bake stores WH bytes produced by sdkl_cpu_rm_to_wh_f16_inplace.
// For a pre-baked weight to be substitutable at runtime, the conversion must be
// deterministic: identical RM input -> identical WH output. Verify byte
// equality across two independent conversions (host buffer and NPU buffer),
// covering the three prefill FC shapes.
TEST_F(HtpKernelTest, OfflineWH_ConversionIsDeterministicAndByteIdentical) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  struct S {
    int N, K;
  };
  const S shapes[] = {{2048, 1024}, {3072, 1024}, {1024, 3072}};
  for (const auto &s : shapes) {
    const size_t n = (size_t)s.N * (size_t)s.K;
    const size_t bytes = n * sizeof(_FP16);
    std::vector<_FP16> rm = makeRandF16(n, -0.5f, 0.5f, s.N ^ s.K);

    // Conversion 1: plain host buffer.
    std::vector<_FP16> wh_host = rm;
    ASSERT_EQ(
      sdkl_cpu_rm_to_wh_f16_inplace(
        (size_t)s.N, (size_t)s.K, reinterpret_cast<_Float16 *>(wh_host.data())),
      0);

    // Conversion 2: NPU-accessible buffer (the path quantizer.cpp uses).
    NpuBuf nb(bytes);
    ASSERT_TRUE(nb.ok());
    std::memcpy(nb.p, rm.data(), bytes);
    ASSERT_EQ(sdkl_cpu_rm_to_wh_f16_inplace((size_t)s.N, (size_t)s.K,
                                            static_cast<_Float16 *>(nb.p)),
              0);

    ASSERT_EQ(std::memcmp(wh_host.data(), nb.p, bytes), 0)
      << "WH bytes differ for " << s.N << "x" << s.K;
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
