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
#include <quantizer.h>
#include <tensor.h>

#include <sdkl_compat.h>
#include <wh_trailer.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
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

/**
 * @brief A zero weight must produce a zero result.
 *
 * @note Degenerate on the CPU, but not on the NPU: the accumulator is staged
 * in a scratch buffer that outlives the call, so a kernel that forgets to
 * clear it returns the previous call's C and still matches a CPU reference
 * everywhere the two happen to agree. Running this after AccuracyVsCpu has
 * filled that scratch with non-zero values is what gives it teeth.
 */
TEST_F(HtpShgemmTest, ZeroWeightGivesZero) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  constexpr int M = 32, N = 32, K = 32;
  auto A = makeRandF32(M * K);
  std::vector<uint16_t> Bh(N * K, 0); // FP16 +0.0

  // Prime the accumulator scratch so a stale-buffer bug cannot pass by
  // reading back zeros it never wrote.
  std::vector<float> C_npu(M * N, 1234.5f);

  ASSERT_NO_THROW(nntrainer::hmx::shgemm_f32f16_f32(
    1, false, true, M, N, K, 1.0f, A.data(), K,
    reinterpret_cast<const _FP16 *>(Bh.data()), K, 0.0f, C_npu.data(), N));

  for (int i = 0; i < M * N; ++i)
    ASSERT_FLOAT_EQ(C_npu[i], 0.0f) << "index " << i << " is not zero";
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

// ---- u8i4 (INT4 weight) ----------------------------------------------------
//
// Same quantization model as u8i8, but the weight is symmetric INT4 ([-7, 7])
// packed two-per-byte in WH layout. These also cover the NPU-resident weight
// cache: the kernel keys resident weights by host pointer, so a stale or
// aliasing entry would silently return another weight's result.

/**
 * @brief Fixture for hmx::shgemm_u8i4_i32 tests; tracks whether the NPU
 *        backend is available so tests can self-skip.
 */
class HtpU8i4Test : public ::testing::Test {
protected:
  void SetUp() override {
    npu_enabled = nntrainer::HtpBackend::global().enabled();
  }
  bool npu_enabled;
};

// Random INT4 weights held one value per byte, sign-extended, in [-7, 7]
// (the range quantize_qint4_weight emits).
static std::vector<int8_t> makeRandI4Vec(int n, uint32_t seed = 21) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(-7, 7);
  std::vector<int8_t> v(n);
  for (auto &x : v)
    x = static_cast<int8_t>(dist(rng));
  return v;
}

// Pack an [N, K] INT4 weight into WH layout in NPU memory, mirroring
// quantize_qint4_weight: sdkl_cpu_rm_to_wh_i4 takes (wt_rows=K, wt_cols=N).
// Returns the NPU buffer holding N*K/2 packed bytes, or nullptr on failure.
// The caller owns the buffer and must sdkl_npu_free it.
static void *packI4WeightToNpu(const std::vector<int8_t> &W_i4, int N, int K) {
  const size_t unpacked = static_cast<size_t>(N) * K;
  void *src = nullptr;
  void *dst = nullptr;
  if (sdkl_npu_alloc(unpacked, &src) != 0 || src == nullptr)
    return nullptr;
  // Over-allocate the destination to the unpacked size so tile padding beyond
  // N*K/2 cannot overflow (same guard quantize_qint4_weight uses).
  if (sdkl_npu_alloc(unpacked, &dst) != 0 || dst == nullptr) {
    sdkl_npu_free(src);
    return nullptr;
  }
  std::memcpy(src, W_i4.data(), unpacked);
  const int rc = sdkl_cpu_rm_to_wh_i4(
    static_cast<uint8_t *>(dst), static_cast<int8_t *>(src),
    static_cast<size_t>(K), static_cast<size_t>(N));
  sdkl_npu_free(src);
  if (rc != 0) {
    sdkl_npu_free(dst);
    return nullptr;
  }
  return dst;
}

// zp_corr[n] = 128 * sum_k W[n,k]
static std::vector<int32_t> makeZpCorr(const std::vector<int8_t> &W, int N,
                                       int K) {
  std::vector<int32_t> zp(N, 0);
  for (int n = 0; n < N; ++n) {
    int32_t s = 0;
    for (int k = 0; k < K; ++k)
      s += static_cast<int32_t>(W[n * K + k]);
    zp[n] = 128 * s;
  }
  return zp;
}

// ---- Exact CPU mirrors of the u8i4 pipeline --------------------------------
//
// The whole u8i4 path is reproducible on the CPU *bit for bit*, not merely to
// within a tolerance, and the tests below assert that rather than a percentage:
//
//   - the integer GEMM sums u8 x i4 products into int32. No rounding, order
//     does not matter, and the largest magnitude at K=4096 is 255*7*4096 =
//     7.3M, far inside int32. So it is exactly reproducible.
//   - the dequantize is elementwise float arithmetic on those exact integers,
//     so mirroring the operand grouping reproduces it exactly.
//
// This matters because comparing the INT4 result against an FP32 reference
// gives ~7% error from quantization alone. A threshold loose enough to admit
// that is loose enough to admit a real bug -- a wrong tile, a shifted nibble,
// an off-by-one zero point. Those all survive 7%; none survive ==.
//
// The two mirrors below are written out here rather than calling into the
// kernel's own helpers on purpose: a reference that shares code with the code
// under test only proves the two agree with each other. These are transcribed
// from the u8i4 contract --
//
//   act_scale  = max|A| / 127
//   X_u8       = clamp(round(A / act_scale) + 128, 0, 255)
//   C          = act_scale * wt_scale[n] * (C_i32 - zp_corr[n])
//
// -- so a change in the kernel that breaks the contract fails here instead of
// being mirrored into the expectation.

/** @brief The u8i4 activation quantization, transcribed from the contract. */
static float quantizeActLikeKernel(const std::vector<float> &A,
                                   std::vector<uint8_t> &X, int M, int Mp,
                                   int K) {
  constexpr float kZeroPoint = 128.0f;
  constexpr float kQuantMax = 127.0f;

  float max_abs = 0.0f;
  for (size_t i = 0, n = static_cast<size_t>(M) * K; i < n; ++i) {
    if (!std::isfinite(A[i]))
      continue;
    const float v = std::fabs(A[i]);
    if (v > max_abs)
      max_abs = v;
  }

  // In double, narrowed once. Computing this as `max_abs / 127.0f` can land a
  // ULP away, which is enough to shift a quantized value by one and break an
  // exact comparison for reasons unrelated to the code under test.
  const double candidate =
    static_cast<double>(max_abs) / static_cast<double>(kQuantMax);
  const float act_scale =
    (std::isfinite(candidate) &&
     candidate >= static_cast<double>(std::numeric_limits<float>::min()))
      ? static_cast<float>(candidate)
      : 1.0f;
  const float inv_act_scale = 1.0f / act_scale;

  X.assign(static_cast<size_t>(Mp) * K, 0);
  for (int m = 0; m < Mp; ++m) {
    const size_t off = static_cast<size_t>(m) * K;
    // Rows at or past M are padding; filling them with the zero point makes
    // them contribute nothing once zp_corr is subtracted.
    if (m >= M) {
      std::memset(X.data() + off, static_cast<int>(kZeroPoint), K);
      continue;
    }
    for (int k = 0; k < K; ++k) {
      const float a = A[off + k];
      if (!std::isfinite(a)) {
        X[off + k] = static_cast<uint8_t>(kZeroPoint);
        continue;
      }
      // std::round is ties-away-from-zero, which is the rule the kernel's
      // FRINTA follows; ties-to-even would disagree on an exact .5.
      float q = std::round(a * inv_act_scale) + kZeroPoint;
      if (q < 0.0f)
        q = 0.0f;
      if (q > 255.0f)
        q = 255.0f;
      X[off + k] = static_cast<uint8_t>(q);
    }
  }
  return act_scale;
}

/**
 * @brief C[m,n] = act_scale * wt_scale[n] * (C_i32[m,n] - zp_corr[n]).
 * @note The left-to-right grouping is load bearing: regrouping the three
 *       multiplies changes the intermediate rounding and costs exactness.
 */
static void dequantizeLikeKernel(const int32_t *C_i32, float *C, int M, int N,
                                 float act_scale, const float *wt_scale,
                                 const int32_t *zp_corr) {
  for (int m = 0; m < M; ++m) {
    const size_t off = static_cast<size_t>(m) * N;
    for (int n = 0; n < N; ++n)
      C[off + n] =
        act_scale * wt_scale[n] *
        (static_cast<float>(C_i32[off + n]) - static_cast<float>(zp_corr[n]));
  }
}

/** @brief Index of the first differing element, or -1. */
template <typename T> static long firstDiff(const T *a, const T *b, size_t n) {
  for (size_t i = 0; i < n; ++i)
    if (a[i] != b[i])
      return static_cast<long>(i);
  return -1;
}

// Level 1: the raw int32 accumulator the NPU produces must equal a plain CPU
// integer GEMM over the identical quantized operands. This is the sharpest
// test in the file -- it removes every float from the comparison, so anything
// it catches is a genuine defect in layout, packing, tiling or padding rather
// than arithmetic drift.
TEST_F(HtpU8i4Test, IntAccumulator_BitExactVsCpu) {
  if (!npu_enabled)
    GTEST_SKIP() << "HTP not available";

  struct Shape {
    const char *name;
    int M, N, K;
  };
  // M=1 is decode; 63 and 65 straddle the 64-row tile, where a padding row
  // leaking into the output would show up and a 64-only test never would.
  const Shape shapes[] = {
    {"tile", 64, 64, 64},       {"decode", 1, 64, 64},
    {"below_tile", 63, 96, 64}, {"above_tile", 65, 96, 64},
    {"min_align", 64, 32, 32},
  };

  for (const auto &s : shapes) {
    SCOPED_TRACE(testing::Message() << "shape=" << s.name << " M=" << s.M
                                    << " N=" << s.N << " K=" << s.K);
    const int Mp = ((s.M + 63) / 64) * 64;

    auto A_f32 = makeRandF32(s.M * s.K, -1.0f, 1.0f);
    auto W_i4 = makeRandI4Vec(s.N * s.K);

    std::vector<uint8_t> X_u8;
    quantizeActLikeKernel(A_f32, X_u8, s.M, Mp, s.K);

    // Reference over the real rows only; the padded rows are the kernel's
    // business and must not reach the caller's output.
    std::vector<int32_t> C_ref(static_cast<size_t>(s.M) * s.N, 0);
    cpuGemmU8I8I32(s.M, s.N, s.K, X_u8.data(), W_i4.data(), C_ref.data());

    void *W_wh = packI4WeightToNpu(W_i4, s.N, s.K);
    ASSERT_NE(W_wh, nullptr) << "INT4 WH packing failed";

    void *X_npu = nullptr;
    void *C_npu = nullptr;
    const size_t x_bytes = static_cast<size_t>(Mp) * s.K;
    const size_t c_bytes = static_cast<size_t>(Mp) * s.N * sizeof(int32_t);
    ASSERT_EQ(sdkl_npu_alloc(x_bytes, &X_npu), 0);
    ASSERT_EQ(sdkl_npu_alloc(c_bytes, &C_npu), 0);
    std::memcpy(X_npu, X_u8.data(), x_bytes);
    std::memset(C_npu, 0, c_bytes);

    const int rc = sdkl_npu_mm_u8i4_i32(
      nntrainer::HtpBackend::global().domain(), Mp, s.N, s.K,
      static_cast<int32_t *>(C_npu), static_cast<const uint8_t *>(X_npu),
      static_cast<const uint8_t *>(W_wh));
    ASSERT_EQ(rc, 0) << "sdkl_npu_mm_u8i4_i32 failed";

    const int32_t *C_got = static_cast<const int32_t *>(C_npu);
    const long d =
      firstDiff(C_ref.data(), C_got, static_cast<size_t>(s.M) * s.N);
    EXPECT_EQ(d, -1) << "int32 accumulator differs at " << d
                     << ": cpu=" << (d >= 0 ? C_ref[d] : 0)
                     << " npu=" << (d >= 0 ? C_got[d] : 0);

    sdkl_npu_free(X_npu);
    sdkl_npu_free(C_npu);
    sdkl_npu_free(W_wh);
  }
}

// Level 2: the whole kernel -- quantize, matmul, dequantize -- against a CPU
// mirror of the same pipeline. Also exact: see the note above.
TEST_F(HtpU8i4Test, Pipeline_BitExactVsCpu) {
  if (!npu_enabled)
    GTEST_SKIP() << "HTP not available";

  struct Shape {
    const char *name;
    int M, N, K;
  };
  const Shape shapes[] = {
    {"tile", 64, 64, 64},       {"decode", 1, 2048, 1024},
    {"below_tile", 63, 96, 64}, {"above_tile", 65, 96, 64},
    {"q_proj", 64, 2048, 1024},
  };

  for (const auto &s : shapes) {
    SCOPED_TRACE(testing::Message() << "shape=" << s.name << " M=" << s.M
                                    << " N=" << s.N << " K=" << s.K);
    const int Mp = ((s.M + 63) / 64) * 64;

    auto A_f32 = makeRandF32(s.M * s.K, -1.0f, 1.0f);
    auto W_i4 = makeRandI4Vec(s.N * s.K);
    std::vector<float> wt_scale(s.N);
    for (int n = 0; n < s.N; ++n)
      wt_scale[n] = 1.0f / 7.0f + n * 0.001f;
    auto zp_corr = makeZpCorr(W_i4, s.N, s.K);

    std::vector<uint8_t> X_u8;
    const float act_scale = quantizeActLikeKernel(A_f32, X_u8, s.M, Mp, s.K);
    std::vector<int32_t> C_i32(static_cast<size_t>(s.M) * s.N, 0);
    cpuGemmU8I8I32(s.M, s.N, s.K, X_u8.data(), W_i4.data(), C_i32.data());
    std::vector<float> C_cpu(static_cast<size_t>(s.M) * s.N);
    dequantizeLikeKernel(C_i32.data(), C_cpu.data(), s.M, s.N, act_scale,
                         wt_scale.data(), zp_corr.data());

    void *W_wh = packI4WeightToNpu(W_i4, s.N, s.K);
    ASSERT_NE(W_wh, nullptr) << "INT4 WH packing failed";

    std::vector<float> C_npu(static_cast<size_t>(s.M) * s.N, 0.0f);
    ASSERT_NO_THROW(nntrainer::hmx::shgemm_u8i4_i32(
      s.M, s.N, s.K, A_f32.data(), static_cast<int8_t *>(W_wh), wt_scale.data(),
      zp_corr.data(), C_npu.data()));
    sdkl_npu_free(W_wh);

    const long d =
      firstDiff(C_cpu.data(), C_npu.data(), static_cast<size_t>(s.M) * s.N);
    EXPECT_EQ(d, -1) << "pipeline differs at " << d
                     << ": cpu=" << (d >= 0 ? C_cpu[d] : 0.0f)
                     << " npu=" << (d >= 0 ? C_npu[d] : 0.0f)
                     << " (relErr overall "
                     << relError(C_npu.data(), C_cpu.data(),
                                 static_cast<size_t>(s.M) * s.N)
                     << ")";
  }
}

// Level 3b: how much accuracy INT4 actually costs against an FP32 reference.
// Unlike the two above this is a quality measurement, not a bug detector --
// the error here is quantization, which is expected and irreducible. The bound
// is loose on purpose; the point is the printed number and catching a
// regression that doubles it, not the pass/fail.
TEST_F(HtpU8i4Test, QuantizationError_VsFp32Reference) {
  if (!npu_enabled)
    GTEST_SKIP() << "HTP not available";

  constexpr int M = 64, N = 512, K = 512;

  auto A_f32 = makeRandF32(M * K, -1.0f, 1.0f);
  auto W_f32 = makeRandF32(N * K, -0.5f, 0.5f);

  // Per-channel symmetric INT4 weight quant, mirroring quantize_qint4_weight.
  std::vector<int8_t> W_i4(static_cast<size_t>(N) * K);
  std::vector<float> wt_scale(N);
  for (int n = 0; n < N; ++n) {
    float max_abs = 0.0f;
    for (int k = 0; k < K; ++k)
      max_abs = std::max(max_abs, std::fabs(W_f32[n * K + k]));
    const float sc = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;
    wt_scale[n] = sc;
    for (int k = 0; k < K; ++k) {
      const long q = std::lround(W_f32[n * K + k] / sc);
      W_i4[n * K + k] =
        static_cast<int8_t>(std::max<long>(-7, std::min<long>(7, q)));
    }
  }
  auto zp_corr = makeZpCorr(W_i4, N, K);

  // FP32 reference: C[m,n] = sum_k A[m,k] * W[n,k]
  std::vector<float> C_ref(static_cast<size_t>(M) * N, 0.0f);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k)
        acc += A_f32[m * K + k] * W_f32[n * K + k];
      C_ref[m * N + n] = acc;
    }

  void *W_wh = packI4WeightToNpu(W_i4, N, K);
  ASSERT_NE(W_wh, nullptr) << "INT4 WH packing failed";
  std::vector<float> C_npu(static_cast<size_t>(M) * N, 0.0f);
  ASSERT_NO_THROW(nntrainer::hmx::shgemm_u8i4_i32(
    M, N, K, A_f32.data(), static_cast<int8_t *>(W_wh), wt_scale.data(),
    zp_corr.data(), C_npu.data()));
  sdkl_npu_free(W_wh);

  const double err =
    relError(C_npu.data(), C_ref.data(), static_cast<size_t>(M) * N);
  std::cout << "[ INFO     ] u8i4 quantization relErr vs FP32 = " << err
            << std::endl;
  EXPECT_LT(err, 0.15) << "INT4 quantization error " << err
                       << " is far above the ~0.07 previously measured; "
                          "suspect the quantizer, not the matmul";
}

// The kernel's shape precondition. u8i8 has a guard test; u8i4 did not.
//
// shgemm_u8i4_i32 throws on two conditions, but only one of them is reachable:
// after N % 32 == 0 passes, N is even, so N*K is even and the "N*K must be
// even for INT4 packing" check can never fire. That second guard is dead code
// -- harmless, but not something a test can cover, and worth knowing before
// someone writes a case for it.
TEST_F(HtpU8i4Test, AlignmentGuard_Rejects_NNotMultipleOf32) {
  if (!npu_enabled)
    GTEST_SKIP() << "HTP not available";

  constexpr int M = 64, N = 48, K = 64; // N = 48 is not a whole HMX tile
  auto A = makeRandF32(M * K, -1.0f, 1.0f);
  auto W_i4 = makeRandI4Vec(N * K);
  std::vector<float> wt_scale(N, 1.0f / 7.0f);
  auto zp_corr = makeZpCorr(W_i4, N, K);
  std::vector<float> C(static_cast<size_t>(M) * N, 0.0f);

  EXPECT_THROW(nntrainer::hmx::shgemm_u8i4_i32(M, N, K, A.data(), W_i4.data(),
                                               wt_scale.data(), zp_corr.data(),
                                               C.data()),
               std::runtime_error);
}

// The resident-weight cache is keyed by host pointer, so two different weights
// must not alias and a repeated call on the first weight must still return the
// first weight's result. Interleaving A, B, A catches both a stale entry and a
// key collision -- failures the accuracy test alone would miss, since it only
// ever uses one weight.
TEST_F(HtpU8i4Test, ResidentWeightCache_NoAliasingAcrossWeights) {
  if (!npu_enabled)
    GTEST_SKIP() << "HTP not available";

  constexpr int M = 64, N = 64, K = 64;

  auto A_f32 = makeRandF32(M * K, -1.0f, 1.0f);
  auto W_a = makeRandI4Vec(N * K, 101);
  auto W_b = makeRandI4Vec(N * K, 202);
  std::vector<float> wt_scale(N, 1.0f / 7.0f);
  auto zp_a = makeZpCorr(W_a, N, K);
  auto zp_b = makeZpCorr(W_b, N, K);

  void *wh_a = packI4WeightToNpu(W_a, N, K);
  void *wh_b = packI4WeightToNpu(W_b, N, K);
  ASSERT_NE(wh_a, nullptr);
  ASSERT_NE(wh_b, nullptr);

  auto run = [&](void *wh, const std::vector<int32_t> &zp) {
    std::vector<float> C(M * N, 0.0f);
    nntrainer::hmx::shgemm_u8i4_i32(M, N, K, A_f32.data(),
                                    static_cast<int8_t *>(wh), wt_scale.data(),
                                    zp.data(), C.data());
    return C;
  };

  const std::vector<float> a1 = run(wh_a, zp_a); // uploads + caches A
  const std::vector<float> b1 = run(wh_b, zp_b); // uploads + caches B
  const std::vector<float> a2 = run(wh_a, zp_a); // must hit A's entry
  const std::vector<float> b2 = run(wh_b, zp_b); // must hit B's entry

  sdkl_npu_free(wh_a);
  sdkl_npu_free(wh_b);

  // Distinct weights must give distinct results, otherwise the aliasing check
  // below would pass trivially.
  EXPECT_GT(relError(a1.data(), b1.data(), M * N), 1e-3)
    << "weights A and B produced the same result; test cannot detect aliasing";

  // Cached repeats must reproduce the first result bit-for-bit: the kernel is
  // deterministic and the only thing that changed is the cache path.
  EXPECT_EQ(a1, a2) << "repeat call on weight A did not reproduce its result "
                       "(stale or aliased cache entry)";
  EXPECT_EQ(b1, b2) << "repeat call on weight B did not reproduce its result "
                       "(stale or aliased cache entry)";
}

// Measures one fc_layer matmul through the production kernel (not the sdkl C
// API directly), at real qwen3-0.6b projection shapes.
//
// The first call uploads the weight; every later call with the same weight
// hits the resident cache. Reporting both separates "what an inference
// actually pays" from the one-off upload, and the steady-state figure is the
// one to compare against a QNN NetRun.
//
// Reports only -- no timing assertion, since numbers move with device thermal
// state. Iteration count via HEXKL_FC_ITERS (default 20).
TEST_F(HtpU8i4Test, FcLayerMm_Perf_ResidentWeight) {
  if (!npu_enabled)
    GTEST_SKIP() << "HTP not available";

  int iters = 20;
  if (const char *e = std::getenv("HEXKL_FC_ITERS")) {
    const int v = std::atoi(e);
    if (v > 0)
      iters = v;
  }

  struct Case {
    const char *name;
    int M, N, K;
  };
  // qwen3-0.6b: hidden 1024, intermediate 3072, GQA q=2048/kv=1024.
  // M=64 is one prefill tile; M=1 is a decode step.
  const std::vector<Case> cases = {
    {"q_proj   prefill", 64, 2048, 1024}, {"o_proj   prefill", 64, 1024, 1024},
    {"ffn_up   prefill", 64, 3072, 1024}, {"ffn_down prefill", 64, 1024, 3072},
    {"q_proj   decode", 1, 2048, 1024},   {"ffn_down decode", 1, 1024, 3072},
  };

  std::printf("\nfc_layer mm through hmx::shgemm_u8i4_i32 (iters=%d)\n", iters);
  std::printf("| shape | M | N | K | first call us | steady us | relErr |\n");
  std::printf("|---|---|---|---|---|---|---|\n");

  for (const auto &c : cases) {
    auto A_f32 = makeRandF32(c.M * c.K, -1.0f, 1.0f);
    auto W_i4 = makeRandI4Vec(c.N * c.K, 77);
    std::vector<float> wt_scale(c.N, 1.0f / 7.0f);
    auto zp_corr = makeZpCorr(W_i4, c.N, c.K);

    void *W_wh = packI4WeightToNpu(W_i4, c.N, c.K);
    ASSERT_NE(W_wh, nullptr) << c.name;

    std::vector<float> C(c.M * c.N, 0.0f);
    auto call = [&]() {
      nntrainer::hmx::shgemm_u8i4_i32(
        c.M, c.N, c.K, A_f32.data(), static_cast<int8_t *>(W_wh),
        wt_scale.data(), zp_corr.data(), C.data());
    };

    // First call: uploads the weight into NPU memory.
    auto t0 = std::chrono::steady_clock::now();
    ASSERT_NO_THROW(call()) << c.name;
    const double first_us = std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - t0)
                              .count();

    // Steady state: weight is resident, scratch is reused.
    double sum_us = 0.0;
    for (int i = 0; i < iters; ++i) {
      auto s = std::chrono::steady_clock::now();
      call();
      sum_us += std::chrono::duration<double, std::micro>(
                  std::chrono::steady_clock::now() - s)
                  .count();
    }
    const double steady_us = sum_us / iters;

    // Confirm the cached path still computes the right answer.
    std::vector<uint8_t> X_u8;
    float act_scale = quantizeActU8(A_f32, X_u8, c.M, c.K);
    std::vector<int32_t> C_i32(c.M * c.N, 0);
    cpuGemmU8I8I32(c.M, c.N, c.K, X_u8.data(), W_i4.data(), C_i32.data());
    std::vector<float> C_cpu(c.M * c.N);
    dequantI32ToF32(c.M, c.N, act_scale, wt_scale.data(), zp_corr.data(),
                    C_i32.data(), C_cpu.data());
    const double err = relError(C.data(), C_cpu.data(), c.M * c.N);

    sdkl_npu_free(W_wh);

    std::printf("| %s | %d | %d | %d | %9.1f | %9.1f | %.5f |\n", c.name, c.M,
                c.N, c.K, first_us, steady_us, err);
    std::fflush(stdout);
    EXPECT_LT(err, 0.01) << c.name << ": cached path lost accuracy";
  }
  SUCCEED();
}

// Shapes vary between prefill and decode, so the scratch pools must grow and
// then be reused by smaller calls without corrupting results.
TEST_F(HtpU8i4Test, ScratchPool_HandlesGrowingAndShrinkingShapes) {
  if (!npu_enabled)
    GTEST_SKIP() << "HTP not available";

  struct Case {
    int M, N, K;
  };
  // Grow (M and N up), then shrink back below the high-water mark.
  const std::vector<Case> cases = {{64, 64, 64}, {128, 128, 64}, {64, 64, 64}};

  for (const auto &c : cases) {
    auto A_f32 = makeRandF32(c.M * c.K, -1.0f, 1.0f);
    auto W_i4 = makeRandI4Vec(c.N * c.K, 55);
    std::vector<float> wt_scale(c.N, 1.0f / 7.0f);
    auto zp_corr = makeZpCorr(W_i4, c.N, c.K);

    std::vector<uint8_t> X_u8;
    float act_scale = quantizeActU8(A_f32, X_u8, c.M, c.K);
    std::vector<int32_t> C_i32(c.M * c.N, 0);
    cpuGemmU8I8I32(c.M, c.N, c.K, X_u8.data(), W_i4.data(), C_i32.data());
    std::vector<float> C_cpu(c.M * c.N);
    dequantI32ToF32(c.M, c.N, act_scale, wt_scale.data(), zp_corr.data(),
                    C_i32.data(), C_cpu.data());

    void *W_wh = packI4WeightToNpu(W_i4, c.N, c.K);
    ASSERT_NE(W_wh, nullptr) << "M=" << c.M << " N=" << c.N << " K=" << c.K;

    std::vector<float> C_npu(c.M * c.N, 0.0f);
    ASSERT_NO_THROW(nntrainer::hmx::shgemm_u8i4_i32(
      c.M, c.N, c.K, A_f32.data(), static_cast<int8_t *>(W_wh), wt_scale.data(),
      zp_corr.data(), C_npu.data()));
    sdkl_npu_free(W_wh);

    EXPECT_LT(relError(C_npu.data(), C_cpu.data(), c.M * c.N), 0.01)
      << "M=" << c.M << " N=" << c.N << " K=" << c.K;
  }
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

/**
 * @brief A ContextData carrying the HTP compute ops.
 *
 * Tensor::dot() only routes to the NPU when the activation carries these, so
 * a test that forgets it silently measures the CPU path instead of the one
 * under test. Built once and shared.
 */
static std::shared_ptr<nntrainer::ContextData> htpContextData() {
  static std::shared_ptr<nntrainer::ContextData> ct = [] {
    auto c = std::make_shared<nntrainer::ContextData>();
    c->setComputeOps(nntrainer::get_htp_ops());
    return c;
  }();
  return ct;
}

// ---- Level 3/4: the fc_layer-facing path ------------------------------------
//
// Everything above calls hmx::shgemm_u8i4_i32 directly. An fc_layer does not:
// it calls Tensor::dot(), which routes on dtype through float_tensor.cpp into
// HtpComputeOps. That routing -- dimension order, the transpose flags, the
// QINT4_HTP dispatch condition, the output shape -- was untested for INT4, so
// a weight indexed [K,N] where the kernel wants [N,K] would have produced a
// plausible-looking wrong answer with nothing to catch it.
//
// The comparison stays bit-exact for the same reason as Level 2: dot() runs
// the same kernel, so anything other than equality is a defect.

/**
 * @brief Build a QINT4_HTP weight tensor exactly the way
 *        quantize_qint4_weight does.
 *
 * Two details are not guessable and both were got wrong first time:
 *
 *   - the tensor must be constructed from a TensorDim, not from the
 *     vector<vector<int8_t>> overload, which accepts only QINT4 and QINT8 and
 *     throws on QINT4_HTP;
 *   - the dims are [K, N], the runtime FC weight layout, not the logical
 *     [N, K] of the weight itself. scale_size() is width(), so emitting [N, K]
 *     would give K scales where the kernel wants N. The WH-packed bytes are
 *     layout-agnostic -- the kernel takes N and K from the matmul dims -- so
 *     only the metadata differs.
 *
 * The payload is N*K/2 WH-packed bytes at the head of an N*K-byte buffer.
 */
static nntrainer::Tensor makeQint4HtpWeight(int N, int K,
                                            const std::vector<int8_t> &W_i4_rm,
                                            std::vector<float> &out_wt_scale,
                                            std::vector<int32_t> &out_zp_corr) {
  out_wt_scale.resize(N);
  for (int n = 0; n < N; ++n)
    out_wt_scale[n] = 1.0f / 7.0f + n * 0.001f;
  out_zp_corr = makeZpCorr(W_i4_rm, N, K);

  const size_t unpacked = static_cast<size_t>(N) * K;
  std::vector<int8_t> packed(unpacked, 0);
  void *src = nullptr;
  void *dst = nullptr;
  if (sdkl_npu_alloc(unpacked, &src) == 0 && src != nullptr &&
      sdkl_npu_alloc(unpacked, &dst) == 0 && dst != nullptr) {
    std::memcpy(src, W_i4_rm.data(), unpacked);
    // (wt_rows=K, wt_cols=N): the i4 packer's argument order is the transpose
    // of the in-place i8 variant's. Getting this backwards scrambles every
    // tile, which is exactly what these tests exist to notice.
    if (sdkl_cpu_rm_to_wh_i4(static_cast<uint8_t *>(dst),
                             static_cast<int8_t *>(src), (size_t)K,
                             (size_t)N) == 0)
      std::memcpy(packed.data(), dst, unpacked);
  }
  if (src)
    sdkl_npu_free(src);
  if (dst)
    sdkl_npu_free(dst);

  nntrainer::TensorDim quant_dim(
    1, 1, K, N, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT4_HTP});
  nntrainer::Tensor wgt(quant_dim, true, nntrainer::Initializer::NONE, "",
                        nntrainer::QScheme::PER_CHANNEL_AFFINE_I4);
  std::memcpy(wgt.getData<int8_t>(), packed.data(), unpacked);
  std::memcpy(wgt.getScale<float>(), out_wt_scale.data(), N * sizeof(float));
  std::memcpy(wgt.getZpCorr<int32_t>(), out_zp_corr.data(),
              N * sizeof(int32_t));
  return wgt;
}

/**
 * @brief Fixture for the QINT4_HTP dot() path, with the HTP context data the
 *        dispatch requires on the activation.
 */
class HtpFcLayerTest : public ::testing::Test {
protected:
  void SetUp() override {
    npu_enabled = nntrainer::HtpBackend::global().enabled();
  }
  bool npu_enabled;
};

// Level 3a: dot() on a QINT4_HTP weight must reproduce the kernel pipeline
// exactly. Shapes include decode (M=1) and both sides of the 64-row tile.
TEST_F(HtpFcLayerTest, Dot_Qint4Htp_BitExactVsCpu) {
  if (!npu_enabled)
    GTEST_SKIP() << "HTP not available";

  struct Shape {
    const char *name;
    int M, N, K;
  };
  const Shape shapes[] = {
    {"tile", 64, 32, 32},
    {"decode", 1, 32, 32},
    {"below_tile", 63, 64, 64},
    {"q_proj_small", 64, 256, 256},
  };

  for (const auto &s : shapes) {
    SCOPED_TRACE(testing::Message() << "shape=" << s.name << " M=" << s.M
                                    << " N=" << s.N << " K=" << s.K);
    const int Mp = ((s.M + 63) / 64) * 64;

    auto A_f32 = makeRandF32(s.M * s.K, -1.0f, 1.0f);
    auto W_i4 = makeRandI4Vec(s.N * s.K);
    std::vector<float> wt_scale;
    std::vector<int32_t> zp_corr;
    nntrainer::Tensor wgt =
      makeQint4HtpWeight(s.N, s.K, W_i4, wt_scale, zp_corr);

    nntrainer::TensorDim adim(1, 1, s.M, s.K);
    nntrainer::Tensor act(adim, false);
    act.allocate();
    std::memcpy(act.getData<float>(), A_f32.data(),
                static_cast<size_t>(s.M) * s.K * sizeof(float));
    act.setContextData(htpContextData());

    // Reference: the same pipeline, on the CPU, exactly.
    std::vector<uint8_t> X_u8;
    const float act_scale = quantizeActLikeKernel(A_f32, X_u8, s.M, Mp, s.K);
    std::vector<int32_t> C_i32(static_cast<size_t>(s.M) * s.N, 0);
    cpuGemmU8I8I32(s.M, s.N, s.K, X_u8.data(), W_i4.data(), C_i32.data());
    std::vector<float> C_cpu(static_cast<size_t>(s.M) * s.N);
    dequantizeLikeKernel(C_i32.data(), C_cpu.data(), s.M, s.N, act_scale,
                         wt_scale.data(), zp_corr.data());

    nntrainer::Tensor result;
    ASSERT_NO_THROW(result = act.dot(wgt, false, true));
    ASSERT_EQ(result.getDim().height(), (unsigned)s.M);
    ASSERT_EQ(result.getDim().width(), (unsigned)s.N);

    const long d = firstDiff(C_cpu.data(), result.getData<float>(),
                             static_cast<size_t>(s.M) * s.N);
    EXPECT_EQ(d, -1) << "dot() differs from the CPU pipeline at " << d;
  }
}

// Level 4: the production bake. quantize_qint4_weight does the per-channel
// quantization, the zp_corr and the WH pack itself; this checks that what it
// emits is what the kernel consumes, end to end from an FP32 weight. A bake
// that packed with the wrong argument order would pass every test above --
// they all build the weight themselves -- and fail only here.
TEST_F(HtpFcLayerTest, BakedWeight_MatchesCpuPipeline) {
  if (!npu_enabled)
    GTEST_SKIP() << "HTP not available";

  constexpr int M = 64, N = 256, K = 256;

  auto A_f32 = makeRandF32(M * K, -1.0f, 1.0f);
  auto W_f32 = makeRandF32(N * K, -0.5f, 0.5f);

  nntrainer::Tensor wt_f32(
    1, 1, N, K, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});
  wt_f32.allocate();
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      wt_f32.setValue(0, 0, n, k, W_f32[n * K + k]);

  nntrainer::Tensor baked;
  ASSERT_NO_THROW(baked = nntrainer::quantize_qint4_weight(wt_f32, false));
  ASSERT_EQ(baked.getDataType(), nntrainer::Tdatatype::QINT4_HTP);

  // Mirror the bake's own quantization to get the logical int4 weights back;
  // the tensor itself now holds WH-packed bytes, which cannot be read as [N,K].
  std::vector<int8_t> W_i4(static_cast<size_t>(N) * K);
  const float *scale = baked.getScale<float>();
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) {
      const long q = std::lround(W_f32[n * K + k] / scale[n]);
      W_i4[n * K + k] =
        static_cast<int8_t>(std::max<long>(-7, std::min<long>(7, q)));
    }

  const int32_t *zp_corr = baked.getZpCorr<int32_t>();
  auto zp_expected = makeZpCorr(W_i4, N, K);
  for (int n = 0; n < N; ++n)
    ASSERT_EQ(zp_corr[n], zp_expected[n]) << "zp_corr mismatch at n=" << n;

  nntrainer::TensorDim adim(1, 1, M, K);
  nntrainer::Tensor act(adim, false);
  act.allocate();
  std::memcpy(act.getData<float>(), A_f32.data(), M * K * sizeof(float));
  act.setContextData(htpContextData());

  std::vector<uint8_t> X_u8;
  const float act_scale = quantizeActLikeKernel(A_f32, X_u8, M, M, K);
  std::vector<int32_t> C_i32(static_cast<size_t>(M) * N, 0);
  cpuGemmU8I8I32(M, N, K, X_u8.data(), W_i4.data(), C_i32.data());
  std::vector<float> C_cpu(static_cast<size_t>(M) * N);
  dequantizeLikeKernel(C_i32.data(), C_cpu.data(), M, N, act_scale, scale,
                       zp_corr);

  nntrainer::Tensor result;
  ASSERT_NO_THROW(result = act.dot(baked, false, true));

  const long d = firstDiff(C_cpu.data(), result.getData<float>(),
                           static_cast<size_t>(M) * N);
  EXPECT_EQ(d, -1) << "baked weight differs from the CPU pipeline at " << d;
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

#if defined(ENABLE_HEXKL) && defined(ENABLE_FP16)
#include <cstdlib>
TEST(HtpPrefillWH, disableToggleForcesMiss) {
  // Register a dummy WH entry and confirm the env toggle forces a miss.
  const unsigned int N = 32, K = 32;
  std::vector<_FP16> wh(N * K);
  for (size_t i = 0; i < wh.size(); ++i)
    wh[i] = static_cast<_FP16>(1.0f);
  const void *key = reinterpret_cast<const void *>(0xABCD);

  nntrainer::hmx::prefillWHRegistryClear();
  nntrainer::hmx::registerPrefillWH(key, N, K, wh.data());

  unsetenv("NNTR_HTP_DISABLE_PREBAKED_WH");
  EXPECT_NE(nntrainer::hmx::lookupPrefillWH(key, N, K), nullptr);

  setenv("NNTR_HTP_DISABLE_PREBAKED_WH", "1", 1);
  EXPECT_EQ(nntrainer::hmx::lookupPrefillWH(key, N, K), nullptr);
  unsetenv("NNTR_HTP_DISABLE_PREBAKED_WH");

  nntrainer::hmx::prefillWHRegistryClear();
}
#endif

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
  // input_width=32, units1=32, units2=32
  auto nn = createWHBakeGateTestNN(32, 32, 32);

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

  // input_width=32, units1=32, units2=32
  auto nn1 = createWHBakeGateTestNN(32, 32, 32);

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
