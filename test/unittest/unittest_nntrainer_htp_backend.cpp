// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_nntrainer_htp_backend.cpp
 * @date   19 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
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
  // StorageOrder=0 (ROW_MAJOR), TransA=false, TransB=true
  nntrainer::shgemm(0, false, true, M, N, K, alpha, A, K,
                    reinterpret_cast<const _FP16 *>(B), K, beta, C, N);
}

// ---- Fixture ---------------------------------------------------------------

/**
 * @brief Fixture for hmx::shgemm_f32f16_f32 accuracy/edge-case tests; tracks
 *        whether the NPU backend is available so tests can self-skip.
 */
class HtpShgemmTest : public ::testing::Test {
protected:
  void SetUp() override {
    npu_enabled = nntrainer::HtpBackend::global().enabled();
  }
  bool npu_enabled;
};

// ---- Accuracy tests --------------------------------------------------------

/**
 * @brief One (M, N, K) shgemm test shape, with a label for failure messages.
 */
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

    // NPU under test: StorageOrder=1 (ROW_MAJOR)
    ASSERT_NO_THROW(nntrainer::hmx::shgemm_f32f16_f32(
      1, false, true, s.M, s.N, s.K, 1.0f, A_f32.data(), s.K,
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

  // beta != 0 is expected to trigger the throw path below.
  EXPECT_THROW(nntrainer::hmx::shgemm_f32f16_f32(
                 1, false, true, M, N, K, 1.0f, A.data(), K,
                 reinterpret_cast<const _FP16 *>(Bh.data()), K, 0.5f, C.data(),
                 N),
               std::runtime_error);
}

// A non-32-aligned M is NOT a throw condition: shgemm_f32f16_f32 rounds M up to
// a multiple of 32 internally and returns only the real rows (verified at the
// kernel level by Padding_NonMultipleOf32_f32f16_f32). The real hard constraint
// is N % 32 == 0 (the WH tile width); confirm the kernel rejects a bad N.
TEST_F(HtpShgemmTest, NNotAlignedThrows) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  constexpr int M = 32, N = 33, K = 32; // N=33 not a multiple of 32
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

/**
 * @brief Fixture for hmx::shgemm_u8i8_i32 tests; tracks whether the NPU
 *        backend is available so tests can self-skip.
 */
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

/**
 * @brief Fixture exercising the FloatTensor::dotQInteger routing: FP32
 *        activation x QINT8 weight dispatches through the HTP context ops
 *        when available. Uses Tensor::dot() via a ContextData stamped with
 *        the HTP ops.
 */
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

} // namespace




#ifdef ENABLE_HEXKL
TEST(HtpBackendLifecycle, npuAliveTracksEnabled) {
  // On a host with no NPU, both are false and must agree; on a device
  // with an initialized NPU, both are true. They must never disagree
  // while the process is running.
  EXPECT_EQ(nntrainer::HtpBackend::global().enabled(), nntrainer::npuAlive());
}
#endif

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
