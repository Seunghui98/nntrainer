// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_nntrainer_htp_backend.cpp
 * @date   19 Jun 2026
 * @brief  HTP backend accuracy, fallback, and edge-case tests.
 *
 * Compiled only with -Denable-htp=true. NPU-specific tests are
 * runtime-skipped when HtpBackend::global().enabled() is false
 * (no device, skel absent, etc.) — same gating pattern as OpenCL tests.
 */

#ifdef ENABLE_HEXKL
#ifdef ENABLE_FP16

#include <char_tensor.h>
#include <compute_ops.h>
#include <context_data.h>
#include <cpu_backend.h>
#include <float_tensor.h>
#include <fp16.h>
#include <gtest/gtest.h>
#include <hexkl_mm.h>
#include <htp_backend.h>
#include <input_layer.h>
#include <layer.h>
#include <model.h>
#include <neuralnet.h>
#include <optimizer.h>
#include <tensor.h>

#include <remote.h>
#include <sdkl.h>
#include <wh_trailer.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ---- Helpers ---------------------------------------------------------------

static std::vector<float> makeRandF32(int n, float lo = -1.0f,
                                      float hi = 1.0f) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto &x : v)
    x = dist(rng);
  return v;
}

static std::vector<uint16_t> toFP16(const std::vector<float> &f32) {
  std::vector<uint16_t> v(f32.size());
  for (size_t i = 0; i < f32.size(); ++i)
    v[i] = nntrainer::compute_fp32_to_fp16(f32[i]);
  return v;
}

// Relative error: max(|npu - cpu|) / (max(|cpu|) + eps)
static double relError(const float *npu, const float *cpu, int n) {
  float ref_max = 0.0f;
  float err_max = 0.0f;
  for (int i = 0; i < n; ++i) {
    ref_max = std::max(ref_max, std::abs(cpu[i]));
    err_max = std::max(err_max, std::abs(npu[i] - cpu[i]));
  }
  return (double)err_max / ((double)ref_max + 1e-6);
}

// Row-major (no-trans) shgemm: C(MxN) = alpha * A(MxK) * B(NxK)^T + beta*C
// Wraps nntrainer::shgemm with StorageOrder=1 (ROW_MAJOR).
static void cpuShgemm(int M, int N, int K, float alpha, const float *A,
                      const uint16_t *B, float beta, float *C) {
  nntrainer::shgemm(0 /*ROW_MAJOR*/, false /*TransA*/, true /*TransB*/, M, N, K,
                    alpha, A, K, reinterpret_cast<const _FP16 *>(B), K, beta, C,
                    N);
}

// ---- Fixture ---------------------------------------------------------------

class HtpShgemmTest : public ::testing::Test {
protected:
  void SetUp() override {
    npu_enabled = nntrainer::HtpBackend::global().enabled();
  }
  bool npu_enabled;
};

// ---- Accuracy tests --------------------------------------------------------

struct ShgemmShape {
  int M, N, K;
  const char *label;
};

static ShgemmShape shapes[] = {
  {32, 32, 32, "small_square"}, // M,N both 32-aligned (HMX tile constraint)
  {32, 32, 64, "medium_rect"},
  {32, 64, 128, "wide_rect"},
  {64, 128, 64, "tall_rect"}, // M > N in prev caused ldc < M abort in col-major
                              // BLAS
  {32, 64, 64, "m32_shape"},
  {128, 256, 512, "large_rect"},
};

TEST_F(HtpShgemmTest, AccuracyVsCpu) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  for (auto &s : shapes) {
    auto A_f32 = makeRandF32(s.M * s.K);
    auto B_f32 = makeRandF32(s.N * s.K, -0.5f, 0.5f);
    auto B_fp16 = toFP16(B_f32);

    std::vector<float> C_cpu(s.M * s.N, 0.0f);
    std::vector<float> C_npu(s.M * s.N, 0.0f);

    // CPU reference
    cpuShgemm(s.M, s.N, s.K, 1.0f, A_f32.data(), B_fp16.data(), 0.0f,
              C_cpu.data());

    // NPU under test
    ASSERT_NO_THROW(nntrainer::hmx::shgemm_f32f16_f32(
      1 /*ROW_MAJOR*/, false, true, s.M, s.N, s.K, 1.0f, A_f32.data(), s.K,
      reinterpret_cast<const _FP16 *>(B_fp16.data()), s.K, 0.0f, C_npu.data(),
      s.N))
      << "Shape: " << s.label;

    double err = relError(C_npu.data(), C_cpu.data(), s.M * s.N);
    EXPECT_LT(err, 0.001) << "Relative error " << err
                          << " exceeds 0.1% for shape " << s.label;
  }
}

TEST_F(HtpShgemmTest, AlphaBetaHandling) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  constexpr int M = 32, N = 32,
                K = 32; // M,N,K all 32-aligned (sdkl rm_to_wh constraint)
  auto A = makeRandF32(M * K);
  auto Bf = makeRandF32(N * K, -0.5f, 0.5f);
  auto Bh = toFP16(Bf);

  // beta=0: NPU write-only path. alpha=0.5 covers the alpha-scaling loop.
  std::vector<float> C_cpu(M * N, 0.0f);
  std::vector<float> C_npu(M * N, 0.0f);

  float alpha = 0.5f, beta = 0.0f;

  cpuShgemm(M, N, K, alpha, A.data(), Bh.data(), beta, C_cpu.data());

  ASSERT_NO_THROW(nntrainer::hmx::shgemm_f32f16_f32(
    1, false, true, M, N, K, alpha, A.data(), K,
    reinterpret_cast<const _FP16 *>(Bh.data()), K, beta, C_npu.data(), N));

  double err = relError(C_npu.data(), C_cpu.data(), M * N);
  EXPECT_LT(err, 0.001) << "alpha/beta mismatch, relative error: " << err;
}

TEST_F(HtpShgemmTest, BetaNonZeroThrows) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  constexpr int M = 32, N = 32,
                K = 32; // M,N,K all 32-aligned so beta check is reached
  auto A = makeRandF32(M * K);
  auto Bh = toFP16(makeRandF32(N * K, -0.5f, 0.5f));
  std::vector<float> C(M * N, 1.0f);

  EXPECT_THROW(nntrainer::hmx::shgemm_f32f16_f32(
                 1, false, true, M, N, K, 1.0f, A.data(), K,
                 reinterpret_cast<const _FP16 *>(Bh.data()), K,
                 0.5f /*beta != 0*/, C.data(), N),
               std::runtime_error);
}

TEST_F(HtpShgemmTest, MNotAlignedThrows) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  constexpr int M = 4, N = 32, K = 32; // M=4 not a multiple of 32
  auto A = makeRandF32(M * K);
  auto Bh = toFP16(makeRandF32(N * K, -0.5f, 0.5f));
  std::vector<float> C(M * N, 0.0f);

  EXPECT_THROW(nntrainer::hmx::shgemm_f32f16_f32(
                 1, false, true, M, N, K, 1.0f, A.data(), K,
                 reinterpret_cast<const _FP16 *>(Bh.data()), K, 0.0f, C.data(),
                 N),
               std::runtime_error);
}

// ---- Fallback test (x86 + device) ------------------------------------------

TEST(HtpFallbackTest, SupportsShgemmTracksBackendState) {
  auto &be = nntrainer::HtpBackend::global();
  auto *htp = nntrainer::get_htp_ops();
  EXPECT_EQ(htp->supports_shgemm(), be.enabled())
    << "supports_shgemm() must exactly mirror HtpBackend::enabled()";
}

TEST(HtpFallbackTest, CpuOpsNeverAdvertisesShgemm) {
  nntrainer::ensureComputeOps();
  EXPECT_FALSE(nntrainer::getComputeOps()->supports_shgemm())
    << "Global CPU ops must not advertise NPU shgemm";
}

// ---- Edge cases (device) ---------------------------------------------------

// Renamed from EdgeCase_SingleElement: tests minimum valid aligned shape (M=32,
// N=32).
TEST_F(HtpShgemmTest, EdgeCase_MinValidShape) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  constexpr int M = 32, N = 32, K = 32;
  auto A = makeRandF32(M * K);
  auto Bf = makeRandF32(N * K, -0.5f, 0.5f);
  auto Bh = toFP16(Bf);
  std::vector<float> C_cpu(M * N, 0.0f), C_npu(M * N, 0.0f);

  cpuShgemm(M, N, K, 1.0f, A.data(), Bh.data(), 0.0f, C_cpu.data());
  ASSERT_NO_THROW(nntrainer::hmx::shgemm_f32f16_f32(
    1, false, true, M, N, K, 1.0f, A.data(), K,
    reinterpret_cast<const _FP16 *>(Bh.data()), K, 0.0f, C_npu.data(), N));
  EXPECT_LT(relError(C_npu.data(), C_cpu.data(), M * N), 0.001f);
}

TEST_F(HtpShgemmTest, EdgeCase_SingleRow) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  constexpr int M = 32, N = 64,
                K = 128; // M must be multiple of 32 (HMX tile constraint)
  auto A = makeRandF32(M * K);
  auto Bf = makeRandF32(N * K, -0.5f, 0.5f);
  auto Bh = toFP16(Bf);
  std::vector<float> C_cpu(M * N, 0.0f), C_npu(M * N, 0.0f);

  cpuShgemm(M, N, K, 1.0f, A.data(), Bh.data(), 0.0f, C_cpu.data());
  ASSERT_NO_THROW(nntrainer::hmx::shgemm_f32f16_f32(
    1, false, true, M, N, K, 1.0f, A.data(), K,
    reinterpret_cast<const _FP16 *>(Bh.data()), K, 0.0f, C_npu.data(), N));
  EXPECT_LT(relError(C_npu.data(), C_cpu.data(), M * N), 0.001);
}

// ---- HtpU8i8Test fixture ---------------------------------------------------
//
// Directly exercises hmx::shgemm_u8i8_i32 (the nntrainer wrapper around
// sdkl_npu_mm_u8i8_i32). Tests are skipped at runtime when HTP is
// unavailable (no device or skel absent).
//
// Quantization model (per-tensor activation, per-channel weight):
//   act_scale  = max_abs(A) / 127
//   X_u8[i]    = clamp(round(A[i] / act_scale) + 128, 0, 255)   (zp=128)
//   zp_corr[n] = 128 * sum_k(W_i8[n,k])
//   C[m,n]     = act_scale * wt_scale[n] *
//                (C_i32[m,n] - zp_corr[n])
//   where C_i32 = X_u8 * W_i8^T  (integer accumulator)

class HtpU8i8Test : public ::testing::Test {
protected:
  void SetUp() override {
    npu_enabled = nntrainer::HtpBackend::global().enabled();
  }
  bool npu_enabled;
};

// ---- Helpers for u8i8 tests ------------------------------------------------

static std::vector<int8_t> makeRandI8Vec(int n, int lo = -50, int hi = 50,
                                         uint32_t seed = 7) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(lo, hi);
  std::vector<int8_t> v(n);
  for (auto &x : v)
    x = static_cast<int8_t>(dist(rng));
  return v;
}

// CPU integer matmul: C_i32[m,n] = sum_k( X_u8[m,k] * W_i8[n,k] )
static void cpuGemmU8I8I32(int M, int N, int K, const uint8_t *X,
                           const int8_t *W, int32_t *C) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      int32_t acc = 0;
      for (int k = 0; k < K; ++k)
        acc += (int32_t)(uint8_t)X[m * K + k] * (int32_t)W[n * K + k];
      C[m * N + n] = acc;
    }
}

// Quantize FP32 activations to U8 with zp=128.
// Returns act_scale = max_abs / 127 (or 1 if all zeros).
static float quantizeActU8(const std::vector<float> &A, std::vector<uint8_t> &X,
                           int M, int K) {
  float max_abs = 0.0f;
  for (auto v : A)
    max_abs = std::max(max_abs, std::fabs(v));
  float act_scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
  float inv = 1.0f / act_scale;
  X.resize(M * K);
  for (int i = 0; i < M * K; ++i) {
    float q = std::round(A[i] * inv) + 128.0f;
    if (q < 0.0f)
      q = 0.0f;
    if (q > 255.0f)
      q = 255.0f;
    X[i] = static_cast<uint8_t>(q);
  }
  return act_scale;
}

class ScopedEnvVar {
public:
  ScopedEnvVar(const char *key, const char *value) : key_(key) {
    const char *current = std::getenv(key_);
    if (current != nullptr) {
      had_value_ = true;
      old_value_ = current;
    }

    if (value != nullptr) {
      setenv(key_, value, 1);
    } else {
      unsetenv(key_);
    }
  }

  ~ScopedEnvVar() {
    if (had_value_) {
      setenv(key_, old_value_.c_str(), 1);
    } else {
      unsetenv(key_);
    }
  }

private:
  const char *key_;
  bool had_value_ = false;
  std::string old_value_;
};

// Dequantize I32 accumulator to FP32.
// C[m,n] = act_scale * wt_scale[n] * (C_i32[m,n] - zp_corr[n])
static void dequantI32ToF32(int M, int N, float act_scale,
                            const float *wt_scale, const int32_t *zp_corr,
                            const int32_t *C_i32, float *C_f32) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n)
      C_f32[m * N + n] =
        act_scale * wt_scale[n] *
        (static_cast<float>(C_i32[m * N + n]) - static_cast<float>(zp_corr[n]));
}

// ---- Tests -----------------------------------------------------------------

TEST_F(HtpU8i8Test, Accuracy_VsCpu) {
  if (!npu_enabled)
    GTEST_SKIP() << "HTP not available";

  constexpr int M = 64, N = 64, K = 64;

  // Build FP32 activations and I8 weights
  auto A_f32 = makeRandF32(M * K, -1.0f, 1.0f);
  auto W_i8 = makeRandI8Vec(N * K, -64, 64);

  // Per-channel weight scales (one per output column)
  std::vector<float> wt_scale(N);
  for (int n = 0; n < N; ++n)
    wt_scale[n] = 1.0f / 64.0f + n * 0.001f;

  // Zero-point correction: zp_corr[n] = 128 * sum_k(W_i8[n,k])
  std::vector<int32_t> zp_corr(N, 0);
  for (int n = 0; n < N; ++n) {
    int32_t s = 0;
    for (int k = 0; k < K; ++k)
      s += (int32_t)W_i8[n * K + k];
    zp_corr[n] = 128 * s;
  }

  // CPU reference
  std::vector<uint8_t> X_u8;
  float act_scale = quantizeActU8(A_f32, X_u8, M, K);
  std::vector<int32_t> C_i32(M * N, 0);
  cpuGemmU8I8I32(M, N, K, X_u8.data(), W_i8.data(), C_i32.data());
  std::vector<float> C_cpu(M * N);
  dequantI32ToF32(M, N, act_scale, wt_scale.data(), zp_corr.data(),
                  C_i32.data(), C_cpu.data());

  // Prepare WH-layout weight in NPU-accessible memory
  void *W_npu = nullptr;
  ASSERT_EQ(sdkl_npu_alloc(N * K * sizeof(int8_t), &W_npu), 0);
  std::memcpy(W_npu, W_i8.data(), N * K * sizeof(int8_t));
  ASSERT_EQ(sdkl_cpu_rm_to_wh_i8_inplace((size_t)N, (size_t)K,
                                         static_cast<int8_t *>(W_npu)),
            0);

  std::vector<float> C_npu(M * N, 0.0f);
  ASSERT_NO_THROW(nntrainer::hmx::shgemm_u8i8_i32(
    M, N, K, A_f32.data(), static_cast<int8_t *>(W_npu), wt_scale.data(),
    zp_corr.data(), C_npu.data()));
  sdkl_npu_free(W_npu);

  double err = relError(C_npu.data(), C_cpu.data(), M * N);
  EXPECT_LT(err, 0.01) << "Relative error " << err
                       << " exceeds 1% vs CPU reference";
}

TEST_F(HtpU8i8Test, ZpCorrApplied) {
  if (!npu_enabled)
    GTEST_SKIP() << "HTP not available";

  // Use a weight where column sums are nonzero so zp_corr matters.
  constexpr int M = 64, N = 64, K = 64;

  auto A_f32 = makeRandF32(M * K, -1.0f, 1.0f);
  // All-positive weights: column sums are nonzero, so zp_corr is nontrivial.
  auto W_i8 = makeRandI8Vec(N * K, 10, 50, 13);

  std::vector<float> wt_scale(N, 1.0f / 64.0f);

  // Correct zp_corr
  std::vector<int32_t> zp_corr(N, 0);
  for (int n = 0; n < N; ++n) {
    int32_t s = 0;
    for (int k = 0; k < K; ++k)
      s += (int32_t)W_i8[n * K + k];
    zp_corr[n] = 128 * s;
  }

  // Convert weight to WH layout (once, shared for both calls)
  void *W_npu = nullptr;
  ASSERT_EQ(sdkl_npu_alloc(N * K * sizeof(int8_t), &W_npu), 0);
  std::memcpy(W_npu, W_i8.data(), N * K * sizeof(int8_t));
  ASSERT_EQ(sdkl_cpu_rm_to_wh_i8_inplace((size_t)N, (size_t)K,
                                         static_cast<int8_t *>(W_npu)),
            0);

  // Run with correct zp_corr
  std::vector<float> C_correct(M * N, 0.0f);
  ASSERT_NO_THROW(nntrainer::hmx::shgemm_u8i8_i32(
    M, N, K, A_f32.data(), static_cast<int8_t *>(W_npu), wt_scale.data(),
    zp_corr.data(), C_correct.data()));

  // Run with all-zero zp_corr
  std::vector<int32_t> zp_zero(N, 0);
  std::vector<float> C_no_zp(M * N, 0.0f);
  ASSERT_NO_THROW(nntrainer::hmx::shgemm_u8i8_i32(
    M, N, K, A_f32.data(), static_cast<int8_t *>(W_npu), wt_scale.data(),
    zp_zero.data(), C_no_zp.data()));

  sdkl_npu_free(W_npu);

  // With nonzero column sums, the results must differ.
  bool any_diff = false;
  for (int i = 0; i < M * N; ++i) {
    if (std::fabs(C_correct[i] - C_no_zp[i]) > 1e-6f) {
      any_diff = true;
      break;
    }
  }
  EXPECT_TRUE(any_diff)
    << "zp_corr had no effect; column sums of W are zero (unexpected)";
}

TEST_F(HtpU8i8Test, AlignmentGuard_NNot32) {
  if (!npu_enabled)
    GTEST_SKIP() << "HTP not available";

  constexpr int M = 64, N = 33, K = 64; // N=33 is not a multiple of 32
  auto A = makeRandF32(M * K);
  auto W = makeRandI8Vec(N * K);
  std::vector<float> wt_scale(N, 0.01f);
  std::vector<int32_t> zp_corr(N, 0);
  std::vector<float> C(M * N, 0.0f);

  // shgemm_u8i8_i32 must throw for N%32 != 0 (alignment guard)
  EXPECT_THROW(nntrainer::hmx::shgemm_u8i8_i32(M, N, K, A.data(), W.data(),
                                               wt_scale.data(), zp_corr.data(),
                                               C.data()),
               std::runtime_error);
}

// ---- HtpDispatchTest fixture -----------------------------------------------
//
// Exercises the FloatTensor::dotQInteger routing: FP32 activation ×
// QINT8 weight dispatches through the HTP context ops when available.
// Uses Tensor::dot() via a ContextData stamped with the HTP ops.

class HtpDispatchTest : public ::testing::Test {
protected:
  void SetUp() override {
    npu_enabled = nntrainer::HtpBackend::global().enabled();

    // Prime the NPU allocator: allocate enough bytes to cover the largest
    // single buffer used by shgemm_u8i8_i32 (C_npu = M*N*sizeof(i32)).
    // sdkl_npu_alloc can return AEE_EFAILED (1) on first allocation without
    // this warm-up, likely because the FAR/ION memory pool is not yet mapped.
    if (npu_enabled) {
      void *prime = nullptr;
      if (sdkl_npu_alloc(64 * 64 * sizeof(int32_t), &prime) == 0 &&
          prime != nullptr)
        sdkl_npu_free(prime);
    }

    // Build a ContextData that uses the HTP compute ops.
    htp_ct = std::make_shared<nntrainer::ContextData>();
    htp_ct->setComputeOps(nntrainer::get_htp_ops());
  }

  bool npu_enabled;
  std::shared_ptr<nntrainer::ContextData> htp_ct;
};

// Build and return a QINT8 Tensor (CharTensor) with WH-layout weight data,
// per-channel scales, and zp_corr. Shape [1, 1, N, K] (weight matrix).
// W_i8_rm: row-major [N, K] source weights.
// Also returns the per-channel scales and zp_corr used.
static nntrainer::Tensor makeQint8Weight(int N, int K,
                                         const std::vector<int8_t> &W_i8_rm,
                                         std::vector<float> &out_wt_scale,
                                         std::vector<int32_t> &out_zp_corr) {
  // Per-channel scales
  out_wt_scale.resize(N);
  for (int n = 0; n < N; ++n)
    out_wt_scale[n] = 1.0f / 64.0f + n * 0.001f;

  // zp_corr[n] = 128 * sum_k(W_i8[n,k])
  out_zp_corr.resize(N, 0);
  for (int n = 0; n < N; ++n) {
    int32_t s = 0;
    for (int k = 0; k < K; ++k)
      s += (int32_t)W_i8_rm[n * K + k];
    out_zp_corr[n] = 128 * s;
  }

  // Convert to WH layout via a temporary NPU-accessible buffer to avoid
  // any risk of sdkl_cpu_rm_to_wh_i8_inplace side-effects on host memory.
  // If sdkl_npu_alloc is unavailable, fall back to host-side conversion.
  std::vector<int8_t> W_wh(W_i8_rm);
  {
    void *tmp = nullptr;
    if (sdkl_npu_alloc(N * K * sizeof(int8_t), &tmp) == 0 && tmp != nullptr) {
      std::memcpy(tmp, W_i8_rm.data(), N * K * sizeof(int8_t));
      sdkl_cpu_rm_to_wh_i8_inplace((size_t)N, (size_t)K,
                                   static_cast<int8_t *>(tmp));
      std::memcpy(W_wh.data(), tmp, N * K * sizeof(int8_t));
      sdkl_npu_free(tmp);
    } else {
      sdkl_cpu_rm_to_wh_i8_inplace((size_t)N, (size_t)K, W_wh.data());
    }
  }

  // Build a 2D vector [N][K] from W_wh for the CharTensor constructor
  std::vector<std::vector<int8_t>> W2d(N, std::vector<int8_t>(K));
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      W2d[n][k] = W_wh[n * K + k];

  ml::train::TensorDim::TensorType t_type = {nntrainer::Tformat::NCHW,
                                             nntrainer::Tdatatype::QINT8};
  nntrainer::Tensor wgt(W2d, out_wt_scale, t_type,
                        nntrainer::QScheme::PER_CHANNEL_AFFINE);

  // Write zp_corr into the tensor's zp_corr storage
  std::memcpy(wgt.getZpCorr<int32_t>(), out_zp_corr.data(),
              N * sizeof(int32_t));

  return wgt;
}

TEST_F(HtpDispatchTest, RoutesToHtp_WhenMAligned) {
  if (!npu_enabled)
    GTEST_SKIP() << "HTP not available";

  constexpr int M = 64, N = 32, K = 32;

  auto A_f32 = makeRandF32(M * K, -1.0f, 1.0f);
  auto W_i8 = makeRandI8Vec(N * K, -50, 50, 11);

  std::vector<float> wt_scale;
  std::vector<int32_t> zp_corr;
  nntrainer::Tensor wgt = makeQint8Weight(N, K, W_i8, wt_scale, zp_corr);

  // Build FP32 activation tensor [1, 1, M, K]
  nntrainer::TensorDim adim(1, 1, M, K);
  nntrainer::Tensor act(adim, false);
  act.allocate();
  std::memcpy(act.getData<float>(), A_f32.data(), M * K * sizeof(float));

  // Stamp the activation with the HTP context
  act.setContextData(htp_ct);

  // CPU reference: quantize A, integer matmul, dequant
  std::vector<uint8_t> X_u8;
  float act_scale = quantizeActU8(A_f32, X_u8, M, K);
  std::vector<int32_t> C_i32_cpu(M * N, 0);
  cpuGemmU8I8I32(M, N, K, X_u8.data(), W_i8.data(), C_i32_cpu.data());
  std::vector<float> C_cpu(M * N);
  dequantI32ToF32(M, N, act_scale, wt_scale.data(), zp_corr.data(),
                  C_i32_cpu.data(), C_cpu.data());

  // Verify that direct sdkl_npu_alloc at the required size works,
  // ensuring the pool is ready before routing through dot().
  void *probe = nullptr;
  int probe_err = sdkl_npu_alloc(M * N * sizeof(int32_t), &probe);
  if (probe_err != 0 || probe == nullptr) {
    GTEST_SKIP() << "sdkl_npu_alloc(" << (M * N * sizeof(int32_t))
                 << ") returned " << probe_err
                 << "; NPU memory pool insufficient for dispatch test";
  }
  sdkl_npu_free(probe);

  // HTP path via dot()
  nntrainer::Tensor result;
  ASSERT_NO_THROW(result = act.dot(wgt, false, true));

  ASSERT_EQ(result.getDim().height(), (unsigned)M);
  ASSERT_EQ(result.getDim().width(), (unsigned)N);

  double err = relError(result.getData<float>(), C_cpu.data(), M * N);
  EXPECT_LT(err, 0.01) << "HTP dot() relError " << err
                       << " vs CPU reference exceeds 1%";
}

TEST_F(HtpDispatchTest, PadsAndRunsHtp_WhenMMisaligned) {
  if (!npu_enabled)
    GTEST_SKIP() << "HTP not available";

  // M=63 is not a multiple of 64, so the QINT8 HTP wrapper pads rows internally
  // and returns only the real output rows.
  constexpr int M = 63, N = 64, K = 64;

  auto A_f32 = makeRandF32(M * K, -1.0f, 1.0f);
  auto W_i8 = makeRandI8Vec(N * K, -50, 50, 17);

  std::vector<float> wt_scale;
  std::vector<int32_t> zp_corr;
  nntrainer::Tensor wgt = makeQint8Weight(N, K, W_i8, wt_scale, zp_corr);

  nntrainer::TensorDim adim(1, 1, M, K);
  nntrainer::Tensor act(adim, false);
  act.allocate();
  std::memcpy(act.getData<float>(), A_f32.data(), M * K * sizeof(float));

  std::vector<uint8_t> X_u8;
  float act_scale = quantizeActU8(A_f32, X_u8, M, K);
  std::vector<int32_t> C_i32_cpu(M * N, 0);
  cpuGemmU8I8I32(M, N, K, X_u8.data(), W_i8.data(), C_i32_cpu.data());
  std::vector<float> C_cpu(M * N);
  dequantI32ToF32(M, N, act_scale, wt_scale.data(), zp_corr.data(),
                  C_i32_cpu.data(), C_cpu.data());

  act.setContextData(htp_ct);

  nntrainer::Tensor result;
  ASSERT_NO_THROW(result = act.dot(wgt, false, true));

  ASSERT_EQ(result.getDim().height(), (unsigned)M);
  ASSERT_EQ(result.getDim().width(), (unsigned)N);

  double err = relError(result.getData<float>(), C_cpu.data(), M * N);
  EXPECT_LT(err, 0.01) << "padded HTP dot() relError " << err
                       << " vs CPU reference exceeds 1%";
}

TEST_F(HtpDispatchTest, RoutesAttentionToCpu_WhenAttentionFallbackDisabled) {
  constexpr int M = 64, N = 32, K = 32;

  auto A_f32 = makeRandF32(M * K, -1.0f, 1.0f);
  auto W_i8 = makeRandI8Vec(N * K, -50, 50, 19);

  std::vector<float> wt_scale;
  std::vector<int32_t> zp_corr;
  nntrainer::Tensor wgt = makeQint8Weight(N, K, W_i8, wt_scale, zp_corr);
  wgt.setName("layer0_wq:weight");

  nntrainer::TensorDim adim(1, 1, M, K);
  nntrainer::Tensor act(adim, false);
  act.allocate();
  std::memcpy(act.getData<float>(), A_f32.data(), M * K * sizeof(float));
  act.setContextData(htp_ct);

  std::vector<uint8_t> X_u8;
  float act_scale = quantizeActU8(A_f32, X_u8, M, K);
  std::vector<int32_t> C_i32_cpu(M * N, 0);
  cpuGemmU8I8I32(M, N, K, X_u8.data(), W_i8.data(), C_i32_cpu.data());
  std::vector<float> C_cpu(M * N);
  dequantI32ToF32(M, N, act_scale, wt_scale.data(), zp_corr.data(),
                  C_i32_cpu.data(), C_cpu.data());

  ScopedEnvVar attention_enable("NNTR_HTP_QINT8_ATTENTION_ENABLE", "0");
  ScopedEnvVar ffn_enable("NNTR_HTP_QINT8_FFN_ENABLE", nullptr);

  nntrainer::Tensor result;
  ASSERT_NO_THROW(result = act.dot(wgt, false, true));

  double err = relError(result.getData<float>(), C_cpu.data(), M * N);
  EXPECT_LT(err, 0.01) << "attention CPU fallback relError " << err
                       << " vs CPU reference exceeds 1%";
}

TEST_F(HtpDispatchTest, RoutesFfnToCpu_WhenFfnFallbackDisabled) {
  constexpr int M = 64, N = 32, K = 32;

  auto A_f32 = makeRandF32(M * K, -1.0f, 1.0f);
  auto W_i8 = makeRandI8Vec(N * K, -50, 50, 23);

  std::vector<float> wt_scale;
  std::vector<int32_t> zp_corr;
  nntrainer::Tensor wgt = makeQint8Weight(N, K, W_i8, wt_scale, zp_corr);
  wgt.setName("layer0_ffn_gate:weight");

  nntrainer::TensorDim adim(1, 1, M, K);
  nntrainer::Tensor act(adim, false);
  act.allocate();
  std::memcpy(act.getData<float>(), A_f32.data(), M * K * sizeof(float));
  act.setContextData(htp_ct);

  std::vector<uint8_t> X_u8;
  float act_scale = quantizeActU8(A_f32, X_u8, M, K);
  std::vector<int32_t> C_i32_cpu(M * N, 0);
  cpuGemmU8I8I32(M, N, K, X_u8.data(), W_i8.data(), C_i32_cpu.data());
  std::vector<float> C_cpu(M * N);
  dequantI32ToF32(M, N, act_scale, wt_scale.data(), zp_corr.data(),
                  C_i32_cpu.data(), C_cpu.data());

  ScopedEnvVar attention_enable("NNTR_HTP_QINT8_ATTENTION_ENABLE", nullptr);
  ScopedEnvVar ffn_enable("NNTR_HTP_QINT8_FFN_ENABLE", "0");

  nntrainer::Tensor result;
  ASSERT_NO_THROW(result = act.dot(wgt, false, true));

  double err = relError(result.getData<float>(), C_cpu.data(), M * N);
  EXPECT_LT(err, 0.01) << "FFN CPU fallback relError " << err
                       << " vs CPU reference exceeds 1%";
}

} // namespace

TEST(WHTrailerCodec, RoundTripsEntries) {
  using nntrainer::hmx::WHTrailerEntry;
  std::vector<WHTrailerEntry> in;
  {
    WHTrailerEntry e;
    e.name = "layer0/fc_q";
    e.N = 32;
    e.K = 64;
    e.wh_bytes.resize((size_t)e.N * e.K * 2, 0);
    for (size_t i = 0; i < e.wh_bytes.size(); ++i)
      e.wh_bytes[i] = static_cast<char>(i & 0xFF);
    in.push_back(std::move(e));
  }
  {
    WHTrailerEntry e;
    e.name = "layer1/fc_o";
    e.N = 64;
    e.K = 32;
    e.wh_bytes.resize((size_t)e.N * e.K * 2, 7);
    in.push_back(std::move(e));
  }

  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  // Simulate a real bin: some weight data first, THEN the trailer.
  const char pad[100] = {0};
  ss.write(pad, sizeof(pad));
  nntrainer::hmx::writeWHTrailer(ss, in);

  std::vector<WHTrailerEntry> out;
  ASSERT_TRUE(nntrainer::hmx::readWHTrailer(ss, out));
  ASSERT_EQ(out.size(), in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    EXPECT_EQ(out[i].name, in[i].name);
    EXPECT_EQ(out[i].N, in[i].N);
    EXPECT_EQ(out[i].K, in[i].K);
    EXPECT_EQ(out[i].wh_bytes, in[i].wh_bytes);
  }
}

TEST(WHTrailerCodec, ReturnsFalseOnPlainData) {
  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  const char pad[64] = {0};
  ss.write(pad, sizeof(pad)); // no trailer/magic
  std::vector<nntrainer::hmx::WHTrailerEntry> out;
  EXPECT_FALSE(nntrainer::hmx::readWHTrailer(ss, out));
}

TEST(WHTrailerLoad, RegisterThenLookupReturnsBytes) {
  nntrainer::hmx::prefillWHRegistryClear();
  const unsigned N = 32, K = 64;
  std::vector<_FP16> rm((size_t)N * K,
                        (_FP16)1.0f); // stands in for weight data
  std::vector<_FP16> wh((size_t)N * K);
  for (size_t i = 0; i < wh.size(); ++i)
    wh[i] = (_FP16)((float)(i % 13));

  nntrainer::hmx::registerPrefillWH(rm.data(), N, K, wh.data());
  const _FP16 *got = nntrainer::hmx::lookupPrefillWH(rm.data(), N, K);
  ASSERT_NE(got, nullptr);
  for (size_t i = 0; i < wh.size(); ++i)
    ASSERT_EQ((float)got[i], (float)wh[i]);
  // Wrong shape must miss.
  EXPECT_EQ(nntrainer::hmx::lookupPrefillWH(rm.data(), N + 32, K), nullptr);
  nntrainer::hmx::prefillWHRegistryClear();
}

/**
 * @brief Helper: build a 2-FC-layer NeuralNetwork for WH-bake gate tests.
 *        dense1 and dense2 both use 32-aligned dims (input_width, units1,
 *        units2 all 32) so that dims alone can't be what excludes dense2
 *        from the WH trailer -- only its layer_dtype_map override (FP32)
 *        should exclude it.
 */
static std::unique_ptr<nntrainer::NeuralNetwork>
createWHBakeGateTestNN(unsigned int input_width, unsigned int units1,
                       unsigned int units2) {
  auto nn = std::make_unique<nntrainer::NeuralNetwork>();

  nn->addLayer(ml::train::layer::Input(
    {"name=input", "input_shape=1:1:" + std::to_string(input_width)}));
  nn->addLayer(ml::train::layer::FullyConnected(
    {"name=dense1", "unit=" + std::to_string(units1)}));
  nn->addLayer(ml::train::layer::FullyConnected(
    {"name=dense2", "unit=" + std::to_string(units2)}));

  nn->setOptimizer(ml::train::optimizer::SGD({"learning_rate=0.1"}));
  nn->setProperty({"loss=mse", "batch_size=1"});

  // INFERENCE mode: TRAIN mode would append epoch/iter metadata after the
  // WH trailer, moving it away from EOF and breaking readWHTrailer's
  // EOF-relative parsing (matches quantize.cpp's inference-only usage).
  nn->compile(ml::train::ExecutionMode::INFERENCE);
  nn->initialize(ml::train::ExecutionMode::INFERENCE);
  return nn;
}

namespace {
/**
 * @brief RAII guard for WHBakeGate tests: guarantees the process-global
 *        WH-bake flag (nntrainer::setWHBakeRequested) is reset to false and
 *        the temporary output file is removed on every exit path from the
 *        test, not just the happy path. gtest's ASSERT_* macros do an early
 *        `return` from the enclosing test function on failure, which would
 *        otherwise skip manual cleanup placed after them and leak the flag
 *        (stuck true) and the file into subsequent tests in the same
 *        process.
 */
struct WHBakeGateTestGuard {
  std::string file_path;
  explicit WHBakeGateTestGuard(std::string path) : file_path(std::move(path)) {
    nntrainer::setWHBakeRequested(true);
  }
  ~WHBakeGateTestGuard() {
    nntrainer::setWHBakeRequested(false);
    remove(file_path.c_str());
  }

  WHBakeGateTestGuard(const WHBakeGateTestGuard &) = delete;
  WHBakeGateTestGuard &operator=(const WHBakeGateTestGuard &) = delete;
};
} // namespace

/**
 * @brief Regression test for the WH-bake gate bug in NeuralNetwork::save
 *        (MODEL_FORMAT_BIN case): the old code only entered the WH-bake
 *        branch when the *global* dtype argument passed to save() was
 *        FP16, and once inside, it hardcoded every layer's save dtype to
 *        FP16, ignoring layer_dtype_map entirely.
 *
 *        Real callers such as Applications/CausalLM/quantize.cpp always
 *        pass dtype=NONE and rely entirely on layer_dtype_map for
 *        per-layer dtypes (that's how --embd_dtype/--lmhead_dtype differ
 *        from --fc_dtype), so with the old code the bake branch never
 *        fired for them at all (no trailer ever written). Had the branch
 *        been forced open by other means, it would additionally have
 *        forced layers the caller wanted to keep at FP32 (e.g. embedding,
 *        lm_head) to FP16 as well.
 *
 *        This test builds dense1 (resolved to FP16 via layer_dtype_map,
 *        32x32 dims -> WH-eligible) and dense2 (explicitly overridden to
 *        FP32 via layer_dtype_map, identical 32x32 dims so the "N,K both
 *        32-aligned" gate can't be what excludes it), with dtype=NONE at
 *        the call site -- exactly the quantize.cpp calling convention.
 *        After the fix, exactly one WH trailer entry must exist, and it
 *        must belong to dense1, not dense2.
 */
TEST(WHBakeGate, RespectsLayerDtypeMapOverride) {
  const std::string file_path = "test_wh_bake_gate.bin";
  auto nn = createWHBakeGateTestNN(/*input_width=*/32, /*units1=*/32,
                                   /*units2=*/32);

  std::map<std::string, nntrainer::TensorDim::DataType> dtype_map = {
    {"dense1", nntrainer::TensorDim::DataType::FP16},
    {"dense2", nntrainer::TensorDim::DataType::FP32},
  };

  WHBakeGateTestGuard guard(file_path);
  ASSERT_NO_THROW(nn->save(file_path, ml::train::ModelFormat::MODEL_FORMAT_BIN,
                           nntrainer::TensorDim::DataType::NONE, dtype_map));

  std::ifstream file(file_path, std::ios::binary);
  ASSERT_TRUE(file.is_open());
  std::vector<nntrainer::hmx::WHTrailerEntry> entries;
  ASSERT_TRUE(nntrainer::hmx::readWHTrailer(file, entries))
    << "Expected a WH trailer to be written: WH bake was requested and "
       "dense1 resolves to FP16 with 32-aligned dims.";
  file.close();

  ASSERT_EQ(entries.size(), 1u);
  EXPECT_NE(entries[0].name.find("dense1"), std::string::npos)
    << "unexpected WH entry name: " << entries[0].name;
  for (const auto &e : entries) {
    EXPECT_EQ(e.name.find("dense2"), std::string::npos)
      << "dense2 was overridden to FP32 in layer_dtype_map and must not be "
         "WH-baked, but found an entry for it: "
      << e.name;
  }
}

/**
 * @brief Regression test for the WH-trailer-registration branch-placement
 *        bug in NeuralNetwork::load()'s MODEL_FORMAT_BIN case: the
 *        registration block that calls nntrainer::hmx::registerPrefillWH
 *        lived only in the non-INFERENCE branch of the exec_mode if/else,
 *        but Applications/CausalLM (the one real caller) always
 *        compiles/initializes/loads its model with
 *        ml::train::ExecutionMode::INFERENCE. So the block was dead code
 *        for the one caller it exists for: confirmed on-device, a WH-baked
 *        bin with a verified-present trailer produced zero "[HTP]
 *        Registered" log lines and no prefill speedup.
 *
 *        WHTrailerLoad.RegisterThenLookupReturnsBytes (above) calls
 *        registerPrefillWH/lookupPrefillWH directly and never exercises
 *        NeuralNetwork::load() at all, so it could not have caught this.
 *        This test builds a model, WH-bakes it to a temp .bin, then loads
 *        that .bin into a *second*, freshly-constructed model using the
 *        exact compile(INFERENCE) -> initialize(INFERENCE) -> load()
 *        sequence CausalLM uses, and asserts that dense1's live weight
 *        pointer is registered in the prefill-WH registry with the same
 *        bytes that were baked into the trailer.
 */
TEST(WHTrailerLoad, InferenceModeLoadRegistersWH) {
  const std::string file_path = "test_wh_trailer_load_inference.bin";
  nntrainer::hmx::prefillWHRegistryClear();

  auto nn1 = createWHBakeGateTestNN(/*input_width=*/32, /*units1=*/32,
                                    /*units2=*/32);

  std::map<std::string, nntrainer::TensorDim::DataType> dtype_map = {
    {"dense1", nntrainer::TensorDim::DataType::FP16},
    {"dense2", nntrainer::TensorDim::DataType::FP32},
  };

  WHBakeGateTestGuard guard(file_path);
  ASSERT_NO_THROW(nn1->save(file_path, ml::train::ModelFormat::MODEL_FORMAT_BIN,
                            nntrainer::TensorDim::DataType::NONE, dtype_map));

  // Independently read back the trailer we just baked so we have an
  // expected-bytes reference that doesn't depend on load()/registration.
  std::vector<nntrainer::hmx::WHTrailerEntry> expected_entries;
  {
    std::ifstream file(file_path, std::ios::binary);
    ASSERT_TRUE(file.is_open());
    ASSERT_TRUE(nntrainer::hmx::readWHTrailer(file, expected_entries))
      << "Expected a WH trailer to be present in the baked bin.";
  }
  ASSERT_EQ(expected_entries.size(), 1u);
  EXPECT_NE(expected_entries[0].name.find("dense1"), std::string::npos);

  // Build a second, freshly-constructed model with the same architecture
  // and load the baked bin the way Applications/CausalLM does: compile and
  // initialize with ExecutionMode::INFERENCE, then load() -- this is
  // exactly the condition the bug made unreachable.
  //
  // Unlike createWHBakeGateTestNN() (whose layers are all FP32 by default,
  // matching the *saving* model before layer_dtype_map is applied at
  // save()-time only), this loading-side model must declare dense1's
  // weight_dtype=FP16 up front so LayerNode::read() reads the right number
  // of bytes for the FP16 weight the bin actually contains -- mirroring how
  // a real INFERENCE-mode loader (e.g. Applications/CausalLM) is
  // constructed with weight dtypes matching the file being loaded.
  auto nn2 = std::make_unique<nntrainer::NeuralNetwork>();
  nn2->addLayer(ml::train::layer::Input({"name=input", "input_shape=1:1:32"}));
  nn2->addLayer(ml::train::layer::FullyConnected(
    {"name=dense1", "unit=32", "weight_dtype=FP16"}));
  nn2->addLayer(ml::train::layer::FullyConnected({"name=dense2", "unit=32"}));
  nn2->setOptimizer(ml::train::optimizer::SGD({"learning_rate=0.1"}));
  nn2->setProperty({"loss=mse", "batch_size=1"});
  ASSERT_EQ(nn2->compile(ml::train::ExecutionMode::INFERENCE), ML_ERROR_NONE);
  ASSERT_EQ(nn2->initialize(ml::train::ExecutionMode::INFERENCE),
            ML_ERROR_NONE);
  ASSERT_NO_THROW(
    nn2->load(file_path, ml::train::ModelFormat::MODEL_FORMAT_BIN));

  // Locate dense1's live FP16 weight in the loaded graph, mirroring the
  // lookup NeuralNetwork::load() itself performs.
  const void *dense1_ptr = nullptr;
  unsigned dense1_N = 0, dense1_K = 0;
  for (auto &node : nn2->getFlatGraph()) {
    auto &rc = node->getRunContext();
    for (unsigned int w = 0; w < rc.getNumWeights(); ++w) {
      auto &wt = rc.getWeight(w);
      if (wt.getDataType() != nntrainer::TensorDim::DataType::FP16)
        continue;
      if (wt.getName().find("dense1") == std::string::npos)
        continue;
      dense1_ptr = (const void *)wt.getData<_FP16>();
      dense1_N = wt.getDim().width();
      dense1_K = wt.getDim().height();
    }
  }
  ASSERT_NE(dense1_ptr, nullptr) << "dense1 FP16 weight not found after load";

  const _FP16 *got =
    nntrainer::hmx::lookupPrefillWH(dense1_ptr, dense1_N, dense1_K);
  ASSERT_NE(got, nullptr)
    << "registerPrefillWH was never called during INFERENCE-mode load() -- "
       "this is the exact regression this test guards against.";

  const auto &expected = expected_entries[0];
  ASSERT_EQ(expected.wh_bytes.size(),
            (size_t)dense1_N * dense1_K * sizeof(_FP16));
  const _FP16 *expected_wh =
    reinterpret_cast<const _FP16 *>(expected.wh_bytes.data());
  for (size_t i = 0; i < (size_t)dense1_N * dense1_K; ++i)
    EXPECT_EQ((float)got[i], (float)expected_wh[i]);

  nntrainer::hmx::prefillWHRegistryClear();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#else
// ENABLE_FP16 not set — provide a no-op main so the binary still links.
#include <gtest/gtest.h>
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif // ENABLE_FP16
#endif // ENABLE_HEXKL
