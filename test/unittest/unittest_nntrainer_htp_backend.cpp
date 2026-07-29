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

// ---- Tests -----------------------------------------------------------------





// Build and return a QINT8 Tensor (CharTensor) with WH-layout weight data,
// per-channel scales, and zp_corr. Shape [1, 1, N, K] (weight matrix).
// W_i8_rm: row-major [N, K] source weights.
// Also returns the per-channel scales and zp_corr used.



} // namespace




#if defined(ENABLE_HEXKL) && defined(ENABLE_FP16)
#include <cstdlib>
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
