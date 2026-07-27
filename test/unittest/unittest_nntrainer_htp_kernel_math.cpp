// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_nntrainer_htp_kernel_math.cpp
 * @date   16 Jul 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Host-runnable numeric tests for the HTP u8i4 / u8i8 GEMM pipeline.
 *
 * Unlike unittest_nntrainer_htp_kernels (which calls the sdkl NPU C API and
 * only runs on a Hexagon device), this file validates the CPU-side math that
 * surrounds the NPU integer GEMM: per-channel weight quantization, the u8
 * activation zero-point correction (zp_corr) and the dequantize formula.
 *
 * The NPU op sdkl_npu_mm_u8i{4,8}_i32 computes the exact integer product
 *   C_i32[m,n] = sum_k X_u8[m,k] * W_q[n,k]
 * so it is emulated here by a plain int32 GEMM. Everything else (the scale /
 * zp_corr computed by quantize_qint{4,8}_weight and the dequant applied by
 * shgemm_u8i{4,8}_i32) is the real production math, mirrored and cross-checked
 * against nntrainer::quantize_qint{4,8}_weight. These tests build and pass on
 * any host without the Hexagon SDK, and match the quantization a QNN INT4/INT8
 * matmul must reproduce for an apples-to-apples fc_layer comparison.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <iostream>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <hexkl_quant.h>
#include <quantizer.h>
#include <tensor.h>

namespace {

// ---- Reference pipeline (mirrors hexkl_mm.cpp / quantizer.cpp exactly) -----

/**
 * @brief Per-channel symmetric weight quantization result: one scale and one
 *        zp_corr per output channel plus the [-QMAX, QMAX] integer weights
 *        stored in logical [N, K] order (w_q[n * K + k]).
 */
struct WeightQuant {
  std::vector<int8_t> w_q;      // [N * K], value in [-QMAX, QMAX]
  std::vector<float> scale;     // [N]
  std::vector<int32_t> zp_corr; // [N] = 128 * sum_k w_q[n, k]
};

// Mirrors quantize_qint{4,8}_weight: scale[n] = max_abs(W[n,:]) / q_max,
// w_q[n,k] = clamp(round(W[n,k] / scale[n]), -q_max, q_max),
// zp_corr[n] = 128 * sum_k w_q[n,k].
static WeightQuant refWeightQuant(const float *W, int N, int K, int q_max) {
  WeightQuant out;
  out.w_q.resize(static_cast<size_t>(N) * K);
  out.scale.resize(N);
  out.zp_corr.resize(N);
  for (int n = 0; n < N; ++n) {
    float max_abs = 0.0f;
    for (int k = 0; k < K; ++k)
      max_abs = std::max(max_abs, std::fabs(W[n * K + k]));

    float scale = max_abs > 0.0f ? max_abs / static_cast<float>(q_max) : 1.0f;
    if (!std::isfinite(scale) || scale <= 0.0f)
      scale = 1.0f;
    out.scale[n] = scale;

    int32_t row_sum = 0;
    for (int k = 0; k < K; ++k) {
      const long q = std::lround(W[n * K + k] / scale);
      const int8_t clamped = static_cast<int8_t>(
        std::max<long>(-q_max, std::min<long>(q_max, q)));
      out.w_q[n * K + k] = clamped;
      row_sum += clamped;
    }
    out.zp_corr[n] = 128 * row_sum;
  }
  return out;
}

// Mirrors the activation path of shgemm_u8i{4,8}_i32: per-tensor scale,
// zp = 128, X_u8 = clamp(round(A / act_scale) + 128, 0, 255).
static std::vector<uint8_t> quantizeActU8(const float *A, int M, int K,
                                          float &act_scale) {
  float max_abs = 0.0f;
  for (int i = 0; i < M * K; ++i)
    if (std::isfinite(A[i]))
      max_abs = std::max(max_abs, std::fabs(A[i]));

  const double cand = static_cast<double>(max_abs) / 127.0;
  act_scale = (std::isfinite(cand) &&
               cand >= static_cast<double>(std::numeric_limits<float>::min()))
                ? static_cast<float>(cand)
                : 1.0f;
  const float inv = 1.0f / act_scale;

  std::vector<uint8_t> x(static_cast<size_t>(M) * K);
  for (int i = 0; i < M * K; ++i) {
    float q = std::round(A[i] * inv) + 128.0f;
    q = std::max(0.0f, std::min(255.0f, q));
    x[i] = static_cast<uint8_t>(q);
  }
  return x;
}

// The exact integer product the NPU performs: C[m,n] = sum_k X[m,k] * W[n,k].
static std::vector<int32_t> cpuIntGemm(int M, int N, int K, const uint8_t *X,
                                       const int8_t *W) {
  std::vector<int32_t> c(static_cast<size_t>(M) * N, 0);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      int32_t acc = 0;
      for (int k = 0; k < K; ++k)
        acc += static_cast<int32_t>(X[m * K + k]) *
               static_cast<int32_t>(W[n * K + k]);
      c[m * N + n] = acc;
    }
  return c;
}

// Mirrors the dequant tail of shgemm_u8i{4,8}_i32:
// C[m,n] = act_scale * scale[n] * (C_i32[m,n] - zp_corr[n]).
static std::vector<float> dequant(const int32_t *c_i32, int M, int N,
                                  float act_scale, const float *scale,
                                  const int32_t *zp_corr) {
  std::vector<float> c(static_cast<size_t>(M) * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n)
      c[m * N + n] = act_scale * scale[n] *
                     (static_cast<float>(c_i32[m * N + n]) -
                      static_cast<float>(zp_corr[n]));
  return c;
}

// FP32 reference GEMM with W stored as [N, K]: C[m,n] = sum_k A[m,k] * W[n,k].
static std::vector<float> cpuGemmF32(int M, int N, int K, const float *A,
                                     const float *W) {
  std::vector<float> c(static_cast<size_t>(M) * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k)
        acc += A[m * K + k] * W[n * K + k];
      c[m * N + n] = acc;
    }
  return c;
}

static float relErr(const std::vector<float> &got,
                    const std::vector<float> &ref) {
  float ref_max = 0.0f, err_max = 0.0f;
  for (size_t i = 0; i < ref.size(); ++i) {
    ref_max = std::max(ref_max, std::fabs(ref[i]));
    err_max = std::max(err_max, std::fabs(got[i] - ref[i]));
  }
  return err_max / (ref_max + 1e-6f);
}

static std::vector<float> randVec(int n, float lo, float hi, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto &x : v)
    x = dist(rng);
  return v;
}

// Build a [1,1,N,K] FP32 tensor (logical [N, K]) from a row-major buffer.
static nntrainer::Tensor makeWeightNK(const std::vector<float> &W, int N,
                                      int K) {
  nntrainer::Tensor t(
    1, 1, N, K, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      t.setValue(0, 0, n, k, W[n * K + k]);
  return t;
}

// End-to-end u8i{bits} pipeline relative error against an FP32 reference,
// using the *production* quantize_qint{4,8}_weight for the weights.
static float pipelineRelErr(int M, int N, int K, int bits, uint32_t seed) {
  const int q_max = (bits == 4) ? 7 : 127;
  auto A = randVec(M * K, -1.0f, 1.0f, seed);
  auto W = randVec(N * K, -0.5f, 0.5f, seed + 100);

  nntrainer::Tensor wt = makeWeightNK(W, N, K);
  nntrainer::Tensor q = (bits == 4)
                          ? nntrainer::quantize_qint4_weight(wt, false)
                          : nntrainer::quantize_qint8_weight(wt, false);
  const int8_t *w_q = q.getData<int8_t>();     // [N*K] logical [N,K] on host
  const float *scale = q.getScale<float>();    // [N]
  const int32_t *zp_corr = q.getZpCorr<int32_t>(); // [N]

  float act_scale = 1.0f;
  auto x_u8 = quantizeActU8(A.data(), M, K, act_scale);
  auto c_i32 = cpuIntGemm(M, N, K, x_u8.data(), w_q);
  auto c_deq = dequant(c_i32.data(), M, N, act_scale, scale, zp_corr);
  auto c_ref = cpuGemmF32(M, N, K, A.data(), W.data());
  (void)q_max;
  return relErr(c_deq, c_ref);
}

// Representative fc_layer projection shapes for qwen3-0.6b (hidden = 1024).
// N is a multiple of 32 and K is even, matching the kernel constraints.
struct FcShape {
  const char *name;
  int M; // rows (prefill batch or decode = 1)
  int N; // out features
  int K; // in features
};
static const FcShape kFcShapes[] = {
  {"wqkv_prefill", 5, 2048, 1024}, {"wo_prefill", 5, 1024, 1024},
  {"ffn_up_prefill", 5, 3072, 1024}, {"ffn_down_prefill", 5, 1024, 3072},
  {"wqkv_decode", 1, 2048, 1024}, {"ffn_up_decode", 1, 3072, 1024},
};

} // namespace

// ---- Production quantization cross-check ------------------------------------

TEST(HtpKernelMath, QInt4Weight_ScaleZpCorr_MatchReference) {
  constexpr int N = 64, K = 128;
  auto W = randVec(N * K, -0.7f, 0.7f, 1);
  nntrainer::Tensor wt = makeWeightNK(W, N, K);

  nntrainer::Tensor q = nntrainer::quantize_qint4_weight(wt, false);
  ASSERT_EQ(q.getDataType(), nntrainer::Tdatatype::QINT4_HTP);
  ASSERT_EQ(q.scale_size(), static_cast<size_t>(N)); // one scale per out chan

  WeightQuant ref = refWeightQuant(W.data(), N, K, /*q_max=*/7);
  const float *scale = q.getScale<float>();
  const int32_t *zp_corr = q.getZpCorr<int32_t>();
  const int8_t *w_q = q.getData<int8_t>();

  for (int n = 0; n < N; ++n) {
    EXPECT_FLOAT_EQ(scale[n], ref.scale[n]) << "n=" << n;
    EXPECT_EQ(zp_corr[n], ref.zp_corr[n]) << "n=" << n;
  }
  for (int i = 0; i < N * K; ++i) {
    ASSERT_GE(w_q[i], -7);
    ASSERT_LE(w_q[i], 7);
    EXPECT_EQ(w_q[i], ref.w_q[i]) << "i=" << i;
  }
}

TEST(HtpKernelMath, QInt8Weight_ScaleZpCorr_MatchReference) {
  constexpr int N = 64, K = 128;
  auto W = randVec(N * K, -0.7f, 0.7f, 2);
  nntrainer::Tensor wt = makeWeightNK(W, N, K);

  nntrainer::Tensor q = nntrainer::quantize_qint8_weight(wt, false);
  ASSERT_EQ(q.getDataType(), nntrainer::Tdatatype::QINT8);

  WeightQuant ref = refWeightQuant(W.data(), N, K, /*q_max=*/127);
  const float *scale = q.getScale<float>();
  const int32_t *zp_corr = q.getZpCorr<int32_t>();
  const int8_t *w_q = q.getData<int8_t>();

  for (int n = 0; n < N; ++n) {
    EXPECT_FLOAT_EQ(scale[n], ref.scale[n]) << "n=" << n;
    EXPECT_EQ(zp_corr[n], ref.zp_corr[n]) << "n=" << n;
  }
  for (int i = 0; i < N * K; ++i)
    EXPECT_EQ(w_q[i], ref.w_q[i]) << "i=" << i;
}

// ---- Zero-point correction is algebraically exact ---------------------------

// C_i32 - zp_corr[n] must equal sum_k (X_u8[m,k] - 128) * W_q[n,k] bit-for-bit.
// This is the identity the u8 activation path relies on; a wrong zp_corr would
// silently bias every output.
TEST(HtpKernelMath, U8I4_ZpCorrection_ExactInteger) {
  constexpr int M = 8, N = 32, K = 64;
  auto A = randVec(M * K, -1.0f, 1.0f, 3);
  auto W = randVec(N * K, -0.5f, 0.5f, 4);

  WeightQuant wq = refWeightQuant(W.data(), N, K, 7);
  float act_scale = 1.0f;
  auto x_u8 = quantizeActU8(A.data(), M, K, act_scale);
  auto c_i32 = cpuIntGemm(M, N, K, x_u8.data(), wq.w_q.data());

  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      int32_t signed_dot = 0;
      for (int k = 0; k < K; ++k)
        signed_dot += (static_cast<int32_t>(x_u8[m * K + k]) - 128) *
                      static_cast<int32_t>(wq.w_q[n * K + k]);
      EXPECT_EQ(c_i32[m * N + n] - wq.zp_corr[n], signed_dot)
        << "m=" << m << " n=" << n;
    }
}

// ---- INT4 two-per-byte nibble packing round-trips ---------------------------

TEST(HtpKernelMath, Int4NibblePack_Roundtrip) {
  // Pack a signed int4 [-7,7] pair into one byte (low then high nibble),
  // unpack with sign extension, expect exact recovery.
  auto packNibble = [](int8_t lo, int8_t hi) -> uint8_t {
    return static_cast<uint8_t>((lo & 0x0F) | ((hi & 0x0F) << 4));
  };
  auto unpackLo = [](uint8_t b) -> int8_t {
    int8_t v = static_cast<int8_t>(b & 0x0F);
    return static_cast<int8_t>((v ^ 0x08) - 0x08); // sign-extend 4 -> 8
  };
  auto unpackHi = [](uint8_t b) -> int8_t {
    int8_t v = static_cast<int8_t>((b >> 4) & 0x0F);
    return static_cast<int8_t>((v ^ 0x08) - 0x08);
  };

  for (int lo = -7; lo <= 7; ++lo)
    for (int hi = -7; hi <= 7; ++hi) {
      uint8_t b = packNibble(static_cast<int8_t>(lo), static_cast<int8_t>(hi));
      EXPECT_EQ(unpackLo(b), lo) << "lo=" << lo << " hi=" << hi;
      EXPECT_EQ(unpackHi(b), hi) << "lo=" << lo << " hi=" << hi;
    }
}

// ---- End-to-end accuracy vs FP32 (fc_layer mm shapes) -----------------------

TEST(HtpKernelMath, U8I8_Pipeline_WithinInt8Tolerance) {
  for (const auto &s : kFcShapes) {
    float err = pipelineRelErr(s.M, s.N, s.K, /*bits=*/8, /*seed=*/17);
    // Per-channel INT8 weight + per-tensor u8 activation: a few % worst case.
    EXPECT_LT(err, 0.05f) << s.name << " (M=" << s.M << " N=" << s.N
                          << " K=" << s.K << ") relErr=" << err;
  }
}

TEST(HtpKernelMath, U8I4_Pipeline_WithinInt4Tolerance) {
  for (const auto &s : kFcShapes) {
    float err = pipelineRelErr(s.M, s.N, s.K, /*bits=*/4, /*seed=*/23);
    // Per-channel INT4 is coarse over large K; ~10-15% is expected loss.
    EXPECT_LT(err, 0.20f) << s.name << " (M=" << s.M << " N=" << s.N
                          << " K=" << s.K << ") relErr=" << err;
  }
}

// INT8 must be strictly more accurate than INT4 on the same data (sanity that
// the bit-width actually buys precision, not just a looser threshold).
TEST(HtpKernelMath, Int8MoreAccurateThanInt4) {
  constexpr int M = 5, N = 1024, K = 1024;
  float e8 = pipelineRelErr(M, N, K, 8, 31);
  float e4 = pipelineRelErr(M, N, K, 4, 31);
  EXPECT_LT(e8, e4);
  EXPECT_LT(e8, 0.05f);
}

// ---- NEON vs scalar ---------------------------------------------------------
//
// hexkl_quant.h promises the NEON routines are bit-identical to the scalar
// reference, not merely close, so these compare with ==. On a host without
// NEON the dispatched entry point *is* the scalar one and these pass trivially
// -- the tests earn their keep when this binary is built for arm64 and run on
// the device, which is the only place the NEON code executes.

namespace q = nntrainer::hmx::quant;

/** @brief Deterministic activations, including the awkward values. */
static std::vector<float> makeActs(size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-3.0f, 3.0f);
  std::vector<float> a(n);
  for (size_t i = 0; i < n; ++i)
    a[i] = d(rng);
  if (n > 8) {
    // Ties, sign-symmetric ties, saturation either way, and non-finites: the
    // cases where round-mode, clamp order and the finite mask can diverge.
    a[0] = 0.5f;
    a[1] = -0.5f;
    a[2] = 1.5f;
    a[3] = 1e30f;
    a[4] = -1e30f;
    a[5] = std::numeric_limits<float>::infinity();
    a[6] = -std::numeric_limits<float>::infinity();
    a[7] = std::numeric_limits<float>::quiet_NaN();
  }
  return a;
}

TEST(HtpQuantSimd, MaxAbs_MatchesScalarExactly) {
  // Lengths either side of the 16- and 4-wide blocks, so the tail is covered.
  for (size_t n : {0u, 1u, 3u, 4u, 15u, 16u, 17u, 63u, 64u, 1024u, 65537u}) {
    auto a = makeActs(n, 1234u + static_cast<unsigned>(n));
    EXPECT_EQ(q::activationMaxAbsScalar(a.data(), n),
              q::activationMaxAbs(a.data(), n))
      << "n=" << n;
  }
}

TEST(HtpQuantSimd, MaxAbs_IgnoresNonFinite) {
  std::vector<float> a(64, 1.0f);
  a[10] = std::numeric_limits<float>::infinity();
  a[20] = -std::numeric_limits<float>::infinity();
  a[30] = std::numeric_limits<float>::quiet_NaN();
  a[40] = -2.5f;
  EXPECT_FLOAT_EQ(2.5f, q::activationMaxAbs(a.data(), a.size()));
}

TEST(HtpQuantSimd, QuantizeU8_MatchesScalarExactly) {
  constexpr float kZp = 128.0f;
  for (size_t K : {1u, 4u, 15u, 16u, 17u, 33u, 1024u}) {
    const size_t M = 3, Mp = 64; // exercises the zero-point row padding too
    auto a = makeActs(M * K, 77u + static_cast<unsigned>(K));
    const float inv = 1.0f / (q::activationMaxAbs(a.data(), M * K) / 127.0f);

    std::vector<uint8_t> ref(Mp * K, 0), got(Mp * K, 0);
    q::quantizeActivationU8Scalar(a.data(), ref.data(), M, Mp, K, inv, kZp);
    q::quantizeActivationU8(a.data(), got.data(), M, Mp, K, inv, kZp);
    EXPECT_EQ(ref, got) << "K=" << K;
  }
}

TEST(HtpQuantSimd, QuantizeU8_SaturatesAndPads) {
  constexpr float kZp = 128.0f;
  const size_t M = 1, Mp = 2, K = 8;
  // inv_scale of 1 puts +1000 far past 255 and -1000 far below 0.
  std::vector<float> a = {1000.0f,
                          -1000.0f,
                          0.0f,
                          127.0f,
                          -128.0f,
                          std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity(),
                          1.0f};
  std::vector<uint8_t> got(Mp * K, 7);
  q::quantizeActivationU8(a.data(), got.data(), M, Mp, K, 1.0f, kZp);

  EXPECT_EQ(255, got[0]);
  EXPECT_EQ(0, got[1]);
  EXPECT_EQ(128, got[2]);
  EXPECT_EQ(255, got[3]);
  EXPECT_EQ(0, got[4]);
  EXPECT_EQ(128, got[5]); // NaN -> zero point
  EXPECT_EQ(128, got[6]); // +inf -> zero point
  EXPECT_EQ(129, got[7]);
  for (size_t k = 0; k < K; ++k)
    EXPECT_EQ(128, got[K + k]) << "padding row k=" << k; // rows [M, Mp)
}

TEST(HtpQuantSimd, Dequantize_MatchesScalarExactly) {
  std::mt19937 rng(9182);
  std::uniform_int_distribution<int32_t> ci(-100000, 100000);
  std::uniform_real_distribution<float> sc(0.001f, 0.05f);

  for (size_t N : {1u, 7u, 8u, 9u, 32u, 2048u}) {
    const size_t M = 3;
    std::vector<int32_t> c_i32(M * N), zp(N);
    std::vector<float> ws(N), ref(M * N, 0.0f), got(M * N, 0.0f);
    for (size_t i = 0; i < M * N; ++i)
      c_i32[i] = ci(rng);
    for (size_t n = 0; n < N; ++n) {
      zp[n] = ci(rng);
      ws[n] = sc(rng);
    }
    const float act_scale = 0.0137f;

    q::dequantizeToF32Scalar(c_i32.data(), ref.data(), M, N, act_scale,
                             ws.data(), zp.data());
    q::dequantizeToF32(c_i32.data(), got.data(), M, N, act_scale, ws.data(),
                       zp.data());
    for (size_t i = 0; i < M * N; ++i)
      ASSERT_EQ(ref[i], got[i]) << "N=" << N << " i=" << i;
  }
}

// Reports which path this binary exercises. Without it a green run on x86 is
// indistinguishable from a green run that actually tested the NEON code.
TEST(HtpQuantSimd, ReportsWhichPathIsUnderTest) {
  std::cout << "[ INFO     ] hexkl_quant dispatch: "
            << (q::neonEnabled() ? "NEON (AArch64)"
                                 : "scalar (no NEON on this target)")
            << std::endl;
  SUCCEED();
}

int main(int argc, char **argv) {
  int result = -1;
  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }
  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }
  return result;
}
