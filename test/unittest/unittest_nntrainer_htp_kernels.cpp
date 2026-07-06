// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_nntrainer_htp_kernels.cpp
 * @date   23 Jun 2026
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

#include <remote.h>
#include <sdkl.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
const __fp16 *lookupPrefillWH(const void *rm_ptr, unsigned int N,
                              unsigned int K);
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

// ---- Error metrics ---------------------------------------------------------

// Relative error over fp16 arrays, computed in fp32 space.
static double relErrorF16(const _FP16 *npu, const _FP16 *cpu, int n) {
  float ref_max = 0.0f, err_max = 0.0f;
  for (int i = 0; i < n; ++i) {
    float c = (float)cpu[i], u = (float)npu[i];
    ref_max = std::max(ref_max, std::abs(c));
    err_max = std::max(err_max, std::abs(u - c));
  }
  return (double)err_max / ((double)ref_max + 1e-6);
}

// ---- CPU references --------------------------------------------------------

// C[M,N] = X[M,K] * W[N,K]^T, fp32 accumulate, fp16 store.
static void cpuGemmF16(int M, int N, int K, const _FP16 *X, const _FP16 *W,
                       _FP16 *C) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k)
        acc += (float)X[m * K + k] * (float)W[n * K + k];
      C[m * N + n] = (_FP16)acc;
    }
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

class HtpKernelTest : public ::testing::Test {
protected:
  void SetUp() override {
    npu_enabled = nntrainer::HtpBackend::global().enabled();
    domain = nntrainer::HtpBackend::global().domain();
  }
  bool npu_enabled = false;
  int domain = 0;
};

// ---- f16f16_f16: RM activations in/out, WH weight --------------------------

TEST_F(HtpKernelTest, Accuracy_f16f16_f16) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  constexpr int M = 32, N = 64, K = 128; // all 32-aligned for M,N
  auto X = makeRandF16(M * K);
  auto W = makeRandF16(N * K);
  std::vector<_FP16> C_cpu(M * N, (_FP16)0.0f);

  cpuGemmF16(M, N, K, X.data(), W.data(), C_cpu.data());

  NpuBuf Xb(M * K * sizeof(_FP16)), Wb(N * K * sizeof(_FP16)),
    Ab(M * N * sizeof(_FP16));
  ASSERT_TRUE(Xb.ok() && Wb.ok() && Ab.ok()) << "sdkl_npu_alloc failed";

  std::memcpy(Xb.p, X.data(), M * K * sizeof(_FP16));
  std::memcpy(Wb.p, W.data(), N * K * sizeof(_FP16));

  ASSERT_EQ(sdkl_cpu_rm_to_wh_f16_inplace((size_t)N, (size_t)K,
                                          static_cast<_FP16 *>(Wb.p)),
            0);
  ASSERT_EQ(sdkl_npu_mm_f16f16_f16(domain, M, N, K, static_cast<_FP16 *>(Ab.p),
                                   static_cast<const _FP16 *>(Xb.p),
                                   static_cast<const _FP16 *>(Wb.p)),
            0);

  double err =
    relErrorF16(static_cast<const _FP16 *>(Ab.p), C_cpu.data(), M * N);
  // FP16-accumulate kernels lose precision as K grows; 1e-2 is the initial
  // tolerance (spec §3). Tighten/loosen after observing device output.
  EXPECT_LT(err, 1e-2) << "f16f16_f16 relative error " << err;
}

// ---- f16: AH activations in/out, WH weight ---------------------------------

TEST_F(HtpKernelTest, Accuracy_f16) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  constexpr int M = 32, N = 64, K = 128;
  auto X = makeRandF16(M * K);
  auto W = makeRandF16(N * K);
  std::vector<_FP16> C_cpu(M * N, (_FP16)0.0f);
  cpuGemmF16(M, N, K, X.data(), W.data(), C_cpu.data());

  NpuBuf Xb(M * K * sizeof(_FP16)), Wb(N * K * sizeof(_FP16)),
    Ab(M * N * sizeof(_FP16));
  ASSERT_TRUE(Xb.ok() && Wb.ok() && Ab.ok());

  std::memcpy(Xb.p, X.data(), M * K * sizeof(_FP16));
  std::memcpy(Wb.p, W.data(), N * K * sizeof(_FP16));

  // X RM -> AH (activation HMX layout), W RM -> WH (weight HMX layout).
  ASSERT_EQ(sdkl_cpu_rm_to_ah_f16_inplace((size_t)M, (size_t)K,
                                          static_cast<_FP16 *>(Xb.p)),
            0);
  ASSERT_EQ(sdkl_cpu_rm_to_wh_f16_inplace((size_t)N, (size_t)K,
                                          static_cast<_FP16 *>(Wb.p)),
            0);

  ASSERT_EQ(sdkl_npu_mm_f16(domain, M, N, K, static_cast<_FP16 *>(Ab.p),
                            static_cast<const _FP16 *>(Xb.p),
                            static_cast<const _FP16 *>(Wb.p)),
            0);

  // Output A is in AH layout -> convert back to RM before comparing.
  ASSERT_EQ(sdkl_cpu_ah_to_rm_f16_inplace((size_t)M, (size_t)N,
                                          static_cast<_FP16 *>(Ab.p)),
            0);

  double err =
    relErrorF16(static_cast<const _FP16 *>(Ab.p), C_cpu.data(), M * N);
  EXPECT_LT(err, 1e-2) << "f16 relative error " << err;
}

// ---- Tensor descriptor helper ----------------------------------------------

static void fillTensorF16(sdkl_tensor_t &t, void *data, uint64_t rows,
                          uint64_t cols, sdkl_tensor_layout_e layout) {
  std::memset(&t, 0, sizeof(t));
  t.ndims = 2;
  t.dims[0] = rows;
  t.dims[1] = cols;
  t.strides[0] = cols; // row-major: row stride = #cols (in elements)
  t.strides[1] = 1;
  t.num_elements = rows * cols;
  t.data_offset = 0;
  t.data = data;
  t.data_dtype = SDKL_DTYPE_FP16;
  t.quantization = SDKL_QUANT_NONE;
  t.layout = layout;
  t.is_continuous = 1;
}

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

// ---- u8i4_i32: ui8 activations, i4 weights (WH tiled), i32 output ----------

TEST_F(HtpKernelTest, Accuracy_u8i4_i32) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  // HMX INT4 requires M % 64 == 0 and N % 32 == 0 (same alignment as INT8).
  // sdkl_npu_mm_u8i4_i32 writes row-major output directly (per sdkl.h:727 and
  // confirmed by official Qualcomm example which does no ah_to_i32_rm
  // conversion).
  constexpr int M = 64, N = 64, K = 128;

  // i4 weights: one value per int8 byte, sign-extended, range [-8, +7].
  auto X = makeRandU8(M * K);
  auto W = makeRandI8(N * K, -8, 7, 17);
  std::vector<int32_t> C_cpu(M * N, 0);
  cpuGemmI32(M, N, K, X.data(), W.data(), C_cpu.data());

  // Tiled buffer: 2 i4 values packed per byte; N and K rounded to 32.
  // Size = (N_aligned * K_aligned) / 2 bytes.
  // Note: sdkl_cpu_rm_to_wh_i4 args are (out, in, wt_rows=K, wt_cols=N).
  const size_t N_aligned = ((size_t)(N + 31) & ~(size_t)31);
  const size_t K_aligned = ((size_t)(K + 31) & ~(size_t)31);
  const size_t tiled_bytes = (N_aligned * K_aligned) / 2;

  NpuBuf Xb(M * K * sizeof(uint8_t)), Wb(tiled_bytes),
    Ab(M * N * sizeof(int32_t));
  ASSERT_TRUE(Xb.ok() && Wb.ok() && Ab.ok());

  std::memcpy(Xb.p, X.data(), M * K * sizeof(uint8_t));

  // Pack RM i4 (int8, sign-extended [-8,+7]) -> WH tiled buffer.
  // wt_rows=K_aligned, wt_cols=N_aligned (per official Qualcomm example).
  ASSERT_EQ(sdkl_cpu_rm_to_wh_i4(static_cast<uint8_t *>(Wb.p), W.data(),
                                 K_aligned, N_aligned),
            0);

  ASSERT_EQ(sdkl_npu_mm_u8i4_i32(domain, (size_t)M, (size_t)N, (size_t)K,
                                 static_cast<int32_t *>(Ab.p),
                                 static_cast<const uint8_t *>(Xb.p),
                                 static_cast<const uint8_t *>(Wb.p)),
            0);

  // Output is already row-major (sdkl.h:727); compare directly.
  EXPECT_TRUE(
    exactMatchI32(static_cast<const int32_t *>(Ab.p), C_cpu.data(), M * N))
    << "u8i4_i32 output does not match int32 reference exactly";
}

// ---- sdkl_mm_tensor: generic tensor GEMM (FP16) ----------------------------
//
// sdkl_mm_tensor_validate requires standard (non-transposed) matmul layout:
//   left[M, K] * right[K, N] = result[M, N]
// i.e. right is stored [K, N] row-major (NOT the transposed [N,K] used by
// the typed kernels sdkl_npu_mm_f16 etc.).
// CPU reference: C[M,N] = X[M,K] * W_kn[K,N] (standard GEMM, no transpose).

TEST_F(HtpKernelTest, Accuracy_mm_tensor_f16) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  // sdkl_mm_tensor computes result[M,N] = left[M,K] @ right[K,N] (standard
  // GEMM). sdkl_mm_tensor_validate enforces left.dims[1] == right.dims[0],
  // i.e. right must be shaped [K,N] in the descriptor.
  //
  // The NPU path (SDKL_PLATFORM_NPU0) requires HMX layouts:
  //   left   -> SDKL_LAYOUT_2D_ROW_MAJOR_ACTIVATION_HMX
  //   right  -> SDKL_LAYOUT_2D_ROW_MAJOR_WEIGHTS_HMX
  //   result -> SDKL_LAYOUT_2D_ROW_MAJOR_ACTIVATION_HMX
  //
  // The WH (weights-HMX) packing expects data logically as [N_out, N_inner]
  // i.e. [N, K] — the transposed form. We generate W as [N,K] (matching the
  // other kernel tests), apply sdkl_cpu_rm_to_wh_f16_inplace(N, K), and then
  // place it in a descriptor with dims [K,N] — which tells the kernel the
  // un-tiled logical shape is [K,N] = right-hand side of the matmul.
  // CPU reference uses X[M,K] @ W_nk[N,K]^T = C[M,N] (via cpuGemmF16).
  constexpr int M = 32, N = 64, K = 128;
  auto X = makeRandF16(M * K);    // row-major [M, K]
  auto W_nk = makeRandF16(N * K); // row-major [N, K] — transposed weights
  std::vector<_FP16> C_cpu(M * N, (_FP16)0.0f);
  // C = X[M,K] @ W_nk[N,K]^T  — same contract as sdkl_npu_mm_f16
  cpuGemmF16(M, N, K, X.data(), W_nk.data(), C_cpu.data());

  NpuBuf Xb(M * K * sizeof(_FP16)), Wb(N * K * sizeof(_FP16)),
    Ab(M * N * sizeof(_FP16));
  ASSERT_TRUE(Xb.ok() && Wb.ok() && Ab.ok());

  // NPU0: convert to HMX layouts before dispatch.
  // X[M,K] RM -> AH;  W_nk[N,K] RM -> WH (WH packing args are n_row=N,
  // n_col=K).
  std::memcpy(Xb.p, X.data(), M * K * sizeof(_FP16));
  std::memcpy(Wb.p, W_nk.data(), N * K * sizeof(_FP16));
  std::memset(Ab.p, 0, M * N * sizeof(_FP16));

  // Validate descriptors with RM layout (shape/metadata check, no NPU call).
  // Use actual NPU buffer pointers; a single-byte dummy would be UB if the
  // validator ever reads through the pointer.
  {
    sdkl_tensor_t left_v, right_v, result_v;
    fillTensorF16(left_v, Xb.p, (uint64_t)M, (uint64_t)K,
                  SDKL_LAYOUT_2D_ROW_MAJOR);
    fillTensorF16(right_v, Wb.p, (uint64_t)K, (uint64_t)N,
                  SDKL_LAYOUT_2D_ROW_MAJOR);
    fillTensorF16(result_v, Ab.p, (uint64_t)M, (uint64_t)N,
                  SDKL_LAYOUT_2D_ROW_MAJOR);
    ASSERT_EQ(sdkl_tensor_validate(&left_v), 0);
    ASSERT_EQ(sdkl_tensor_validate(&right_v), 0);
    ASSERT_EQ(sdkl_tensor_validate(&result_v), 0);
    ASSERT_EQ(sdkl_mm_tensor_validate(&result_v, &left_v, &right_v), 0);
  }

  ASSERT_EQ(sdkl_cpu_rm_to_ah_f16_inplace((size_t)M, (size_t)K,
                                          static_cast<_FP16 *>(Xb.p)),
            0);
  ASSERT_EQ(sdkl_cpu_rm_to_wh_f16_inplace((size_t)N, (size_t)K,
                                          static_cast<_FP16 *>(Wb.p)),
            0);

  // Descriptor: right has logical shape [K,N] (the GEMM right-hand side),
  // but the underlying data is in WH format packed from W_nk[N,K].
  // Strides for HMX layouts are still set row-major (they're ignored by the
  // HMX path; only dims and layout tag matter for dispatch).
  sdkl_tensor_t left, right, result;
  fillTensorF16(left, Xb.p, (uint64_t)M, (uint64_t)K,
                SDKL_LAYOUT_2D_ROW_MAJOR_ACTIVATION_HMX);
  fillTensorF16(right, Wb.p, (uint64_t)K, (uint64_t)N,
                SDKL_LAYOUT_2D_ROW_MAJOR_WEIGHTS_HMX);
  fillTensorF16(result, Ab.p, (uint64_t)M, (uint64_t)N,
                SDKL_LAYOUT_2D_ROW_MAJOR_ACTIVATION_HMX);

  ASSERT_EQ(sdkl_mm_tensor(SDKL_PLATFORM_NPU0, &result, &left, &right), 0)
    << "sdkl_mm_tensor failed with HMX layouts";

  // Result is in AH layout; convert back to RM for comparison.
  ASSERT_EQ(sdkl_cpu_ah_to_rm_f16_inplace((size_t)M, (size_t)N,
                                          static_cast<_FP16 *>(Ab.p)),
            0);

  double err =
    relErrorF16(static_cast<const _FP16 *>(Ab.p), C_cpu.data(), M * N);
  EXPECT_LT(err, 1e-2) << "mm_tensor f16 relative error " << err;
}

// ---- Constraint: misaligned M/N must be rejected by the kernel -------------

TEST_F(HtpKernelTest, Constraint_MisalignedRejected_f16f16) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  // M = 16 (not a multiple of 32). N, K aligned.
  constexpr int M = 16, N = 64, K = 128;
  auto X = makeRandF16(M * K);
  auto W = makeRandF16(N * K);

  NpuBuf Xb(M * K * sizeof(_FP16)), Wb(N * K * sizeof(_FP16)),
    Ab(M * N * sizeof(_FP16));
  ASSERT_TRUE(Xb.ok() && Wb.ok() && Ab.ok());
  std::memcpy(Xb.p, X.data(), M * K * sizeof(_FP16));
  std::memcpy(Wb.p, W.data(), N * K * sizeof(_FP16));
  (void)sdkl_cpu_rm_to_wh_f16_inplace((size_t)N, (size_t)K,
                                      static_cast<_FP16 *>(Wb.p));

  int rc = sdkl_npu_mm_f16f16_f16(domain, M, N, K, static_cast<_FP16 *>(Ab.p),
                                  static_cast<const _FP16 *>(Xb.p),
                                  static_cast<const _FP16 *>(Wb.p));
  EXPECT_NE(rc, 0) << "kernel accepted M=16 (not multiple of 32)";
}

// ---- Perf harness ----------------------------------------------------------

struct TimeStats {
  double mean_ms, min_ms, std_ms;
};

template <typename Fn> static TimeStats timeIt(int warmup, int iters, Fn &&fn) {
  for (int i = 0; i < warmup; ++i)
    fn();
  std::vector<double> samples;
  samples.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    fn();
    auto t1 = std::chrono::steady_clock::now();
    samples.push_back(
      std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  double sum = 0.0, mn = samples[0];
  for (double s : samples) {
    sum += s;
    mn = std::min(mn, s);
  }
  double mean = sum / iters, var = 0.0;
  for (double s : samples)
    var += (s - mean) * (s - mean);
  return {mean, mn, std::sqrt(var / iters)};
}

struct PerfRow {
  std::string op;
  int M, N, K;
  double kernel_ms, full_ms, throughput, cpu_ms, speedup;
  std::string unit; // "GFLOPS" or "TOPS"
};

static std::vector<PerfRow> g_perf;

static void recordPerf(const std::string &op, int M, int N, int K,
                       const TimeStats &kernel, const TimeStats &full,
                       double cpu_ms, bool quant) {
  double flops = 2.0 * (double)M * (double)N * (double)K;
  double thr = quant ? (flops / (kernel.mean_ms * 1e-3)) / 1e12 // TOPS
                     : (flops / (kernel.mean_ms * 1e-3)) / 1e9; // GFLOPS
  double speedup = (kernel.mean_ms > 0.0) ? cpu_ms / kernel.mean_ms : 0.0;
  g_perf.push_back({op, M, N, K, kernel.mean_ms, full.mean_ms, thr, cpu_ms,
                    speedup, quant ? "TOPS" : "GFLOPS"});
}

// Shape sets (all M%32==0, N%32==0).
struct Shape {
  int M, N, K;
};

static const std::vector<Shape> &generalShapes() {
  static const std::vector<Shape> s = {
    {32, 32, 32},     {64, 64, 64},       {128, 128, 128},    {256, 256, 256},
    {512, 512, 512},  {1024, 1024, 1024}, {2048, 2048, 2048}, {32, 2048, 2048},
    {2048, 32, 2048}, {512, 512, 2048},
  };
  return s;
}

// Qwen3-0.6B: hidden 1024, intermediate 3072, GQA q=2048/kv=1024, vocab 151936.
static const std::vector<Shape> &llmShapes() {
  static const std::vector<Shape> s = {
    {32, 2048, 1024},   {128, 2048, 1024}, {512, 2048, 1024}, // q_proj
    {32, 1024, 1024},   {128, 1024, 1024}, {512, 1024, 1024}, // kv_proj
    {32, 1024, 2048},   {128, 1024, 2048}, {512, 1024, 2048}, // o_proj
    {32, 3072, 1024},   {128, 3072, 1024}, {512, 3072, 1024}, // gate/up
    {32, 1024, 3072},   {128, 1024, 3072}, {512, 1024, 3072}, // down
    {32, 151936, 1024}, // lm_head (M=32 only)
  };
  return s;
}

static void writePerfCsv() {
  const char *path = std::getenv("HTP_PERF_OUT");
  std::string out = path ? path : "hexkl-kernel-perf.csv";
  std::ofstream f(out);
  f << "op,M,N,K,kernel_ms,full_ms,throughput,unit,cpu_ms,speedup\n";
  for (const auto &r : g_perf)
    f << r.op << ',' << r.M << ',' << r.N << ',' << r.K << ',' << r.kernel_ms
      << ',' << r.full_ms << ',' << r.throughput << ',' << r.unit << ','
      << r.cpu_ms << ',' << r.speedup << '\n';
}

static void printPerfMarkdown() {
  printf("\n| op | M | N | K | kernel_ms | full_ms | throughput | unit | "
         "cpu_ms | speedup |\n");
  printf("|---|---|---|---|---|---|---|---|---|---|\n");
  for (const auto &r : g_perf)
    printf("| %s | %d | %d | %d | %.4f | %.4f | %.2f | %s | %.4f | %.2fx |\n",
           r.op.c_str(), r.M, r.N, r.K, r.kernel_ms, r.full_ms, r.throughput,
           r.unit.c_str(), r.cpu_ms, r.speedup);
}

// ---- Performance sweeps ----------------------------------------------------

static void perfSweepF16f16(int domain, const std::vector<Shape> &shapes,
                            const char *tag) {
  for (const auto &s : shapes) {
    const int M = s.M, N = s.N, K = s.K;
    auto X = makeRandF16(M * K);
    auto W = makeRandF16(N * K);
    std::vector<_FP16> C_cpu(M * N, (_FP16)0.0f);

    NpuBuf Xb(M * K * sizeof(_FP16)), Wb(N * K * sizeof(_FP16)),
      Ab(M * N * sizeof(_FP16));
    if (!(Xb.ok() && Wb.ok() && Ab.ok())) {
      printf("SKIP %s f16f16 %dx%dx%d: alloc failed\n", tag, M, N, K);
      continue;
    }
    std::memcpy(Xb.p, X.data(), M * K * sizeof(_FP16));

    int last_rc = 0;
    auto kernelOnly = [&]() {
      last_rc = sdkl_npu_mm_f16f16_f16(
        domain, M, N, K, static_cast<_FP16 *>(Ab.p),
        static_cast<const _FP16 *>(Xb.p), static_cast<const _FP16 *>(Wb.p));
    };
    auto withXform = [&]() {
      std::memcpy(Wb.p, W.data(), N * K * sizeof(_FP16));
      sdkl_cpu_rm_to_wh_f16_inplace((size_t)N, (size_t)K,
                                    static_cast<_FP16 *>(Wb.p));
      kernelOnly();
    };
    // Prepare WH weight once for the kernel-only timing.
    std::memcpy(Wb.p, W.data(), N * K * sizeof(_FP16));
    sdkl_cpu_rm_to_wh_f16_inplace((size_t)N, (size_t)K,
                                  static_cast<_FP16 *>(Wb.p));

    TimeStats k = timeIt(10, 50, kernelOnly);
    ASSERT_EQ(last_rc, 0) << "f16f16_f16 kernel returned non-zero for shape "
                          << M << "x" << N << "x" << K;
    TimeStats full = timeIt(5, 20, withXform);
    TimeStats cpu = timeIt(
      1, 3, [&]() { cpuGemmF16(M, N, K, X.data(), W.data(), C_cpu.data()); });
    recordPerf("f16f16_f16", M, N, K, k, full, cpu.mean_ms, /*quant=*/false);
  }
}

TEST_F(HtpKernelTest, Perf_f16f16_f16) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";
  perfSweepF16f16(domain, generalShapes(), "general");
  perfSweepF16f16(domain, llmShapes(), "llm");
  SUCCEED();
}

static void perfSweepF16(int domain, const std::vector<Shape> &shapes,
                         const char *tag) {
  for (const auto &s : shapes) {
    const int M = s.M, N = s.N, K = s.K;
    auto X = makeRandF16(M * K);
    auto W = makeRandF16(N * K);
    std::vector<_FP16> C_cpu(M * N, (_FP16)0.0f);

    NpuBuf Xb(M * K * sizeof(_FP16)), Wb(N * K * sizeof(_FP16)),
      Ab(M * N * sizeof(_FP16));
    if (!(Xb.ok() && Wb.ok() && Ab.ok())) {
      printf("SKIP %s f16 %dx%dx%d: alloc failed\n", tag, M, N, K);
      continue;
    }

    // Prepare AH-layout X and WH-layout W once for kernel-only timing.
    std::memcpy(Xb.p, X.data(), M * K * sizeof(_FP16));
    sdkl_cpu_rm_to_ah_f16_inplace((size_t)M, (size_t)K,
                                  static_cast<_FP16 *>(Xb.p));
    std::memcpy(Wb.p, W.data(), N * K * sizeof(_FP16));
    sdkl_cpu_rm_to_wh_f16_inplace((size_t)N, (size_t)K,
                                  static_cast<_FP16 *>(Wb.p));

    int last_rc = 0;
    auto kernelOnly = [&]() {
      last_rc = sdkl_npu_mm_f16(domain, M, N, K, static_cast<_FP16 *>(Ab.p),
                                static_cast<const _FP16 *>(Xb.p),
                                static_cast<const _FP16 *>(Wb.p));
    };
    auto withXform = [&]() {
      std::memcpy(Xb.p, X.data(), M * K * sizeof(_FP16));
      sdkl_cpu_rm_to_ah_f16_inplace((size_t)M, (size_t)K,
                                    static_cast<_FP16 *>(Xb.p));
      std::memcpy(Wb.p, W.data(), N * K * sizeof(_FP16));
      sdkl_cpu_rm_to_wh_f16_inplace((size_t)N, (size_t)K,
                                    static_cast<_FP16 *>(Wb.p));
      kernelOnly();
      sdkl_cpu_ah_to_rm_f16_inplace((size_t)M, (size_t)N,
                                    static_cast<_FP16 *>(Ab.p));
    };

    TimeStats k = timeIt(10, 50, kernelOnly);
    ASSERT_EQ(last_rc, 0) << "f16 kernel returned non-zero for shape " << M
                          << "x" << N << "x" << K;
    TimeStats full = timeIt(5, 20, withXform);
    TimeStats cpu = timeIt(
      1, 3, [&]() { cpuGemmF16(M, N, K, X.data(), W.data(), C_cpu.data()); });
    recordPerf("f16", M, N, K, k, full, cpu.mean_ms, /*quant=*/false);
  }
}

TEST_F(HtpKernelTest, Perf_f16) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";
  perfSweepF16(domain, generalShapes(), "general");
  perfSweepF16(domain, llmShapes(), "llm");
  SUCCEED();
}

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
    recordPerf("u8i8_i32", M, N, K, k, k, cpu.mean_ms, /*quant=*/true);
  }
}

TEST_F(HtpKernelTest, Perf_u8i8_i32) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";
  perfSweepU8I8(domain, generalShapes(), "general");
  perfSweepU8I8(domain, llmShapes(), "llm");
  SUCCEED();
}

static void perfSweepU8I4(int domain, const std::vector<Shape> &shapes,
                          const char *tag) {
  for (const auto &s : shapes) {
    const int M = s.M, N = s.N, K = s.K;
    // HMX INT4 requires M % 64 == 0.
    if (M % 64 != 0) {
      printf("SKIP %s u8i4 %dx%dx%d: M not multiple of 64\n", tag, M, N, K);
      continue;
    }
    auto X = makeRandU8(M * K);
    auto W = makeRandI8(N * K, -8, 7, 17);
    std::vector<int32_t> C_cpu(M * N, 0);

    const size_t N_aligned = ((size_t)(N + 31) & ~(size_t)31);
    const size_t K_aligned = ((size_t)(K + 31) & ~(size_t)31);
    const size_t tiled_bytes = (N_aligned * K_aligned) / 2;

    NpuBuf Xb(M * K), Wb(tiled_bytes), Ab(M * N * sizeof(int32_t));
    if (!(Xb.ok() && Wb.ok() && Ab.ok())) {
      printf("SKIP %s u8i4 %dx%dx%d: alloc failed\n", tag, M, N, K);
      continue;
    }
    std::memcpy(Xb.p, X.data(), M * K);
    sdkl_cpu_rm_to_wh_i4(static_cast<uint8_t *>(Wb.p), W.data(), K_aligned,
                         N_aligned);
    int last_rc = 0;
    auto kernelOnly = [&]() {
      last_rc = sdkl_npu_mm_u8i4_i32(
        domain, (size_t)M, (size_t)N, (size_t)K, static_cast<int32_t *>(Ab.p),
        static_cast<const uint8_t *>(Xb.p), static_cast<const uint8_t *>(Wb.p));
    };
    TimeStats k = timeIt(10, 50, kernelOnly);
    ASSERT_EQ(last_rc, 0) << "u8i4_i32 kernel returned non-zero for shape " << M
                          << "x" << N << "x" << K;
    TimeStats cpu = timeIt(
      1, 3, [&]() { cpuGemmI32(M, N, K, X.data(), W.data(), C_cpu.data()); });
    recordPerf("u8i4_i32", M, N, K, k, k, cpu.mean_ms, /*quant=*/true);
  }
}

TEST_F(HtpKernelTest, Perf_u8i4_i32) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";
  perfSweepU8I4(domain, generalShapes(), "general");
  perfSweepU8I4(domain, llmShapes(), "llm");
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

TEST_F(HtpKernelTest, Perf_f32f16_f32_Prefill) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  struct PrefillShape {
    int M, N, K;
    const char *name;
  };
  const PrefillShape shapes[] = {
    {16, 2048, 1024, "q_proj"},    {16, 1024, 1024, "kv_proj"},
    {16, 1024, 2048, "o_proj"},    {16, 3072, 1024, "gate_up_proj"},
    {16, 1024, 3072, "down_proj"},
  };
  for (const auto &s : shapes) {
    std::vector<float> A = makeRandF32(s.M * s.K);
    std::vector<_FP16> W = makeRandF16(s.N * s.K);
    std::vector<float> C(s.M * s.N, 0.f);

    auto npu_call = [&]() {
      nntrainer::hmx::shgemm_f32f16_f32(
        0, false, true, s.M, s.N, s.K, 1.0f, A.data(), s.K,
        reinterpret_cast<const __fp16 *>(W.data()), s.K, 0.0f, C.data(), s.N);
    };
    std::vector<float> ref(s.M * s.N, 0.f);
    TimeStats full = timeIt(3, 10, npu_call);
    TimeStats cpu = timeIt(1, 3, [&]() {
      cpuGemmF32F16(s.M, s.N, s.K, A.data(), W.data(), ref.data());
    });
    recordPerf(std::string("f32f16_f32_prefill_") + s.name, s.M, s.N, s.K, full,
               full, cpu.mean_ms, /*quant=*/false);
  }
  SUCCEED();
}

// Diagnostic: break the transient prefill path into phases and time each, to
// verify that weight staging (alloc + memcpy + rm_to_wh) — not the matmul or
// copy-back — dominates. Mirrors the M>1 transient branch of shgemm_f32f16_f32
// without touching the library. Not a pass/fail gate: always SUCCEED().
TEST_F(HtpKernelTest, PhaseTiming_TransientPrefillBreakdown) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  struct Shape {
    int M, N, K;
    const char *name;
  };
  const Shape shapes[] = {
    {16, 2048, 1024, "q_proj"},
    {16, 3072, 1024, "gate_up_proj"},
    {16, 1024, 3072, "down_proj"},
  };

  for (const auto &s : shapes) {
    const unsigned int Mp = (s.M + 31u) & ~31u;
    std::vector<float> A = makeRandF32(s.M * s.K);
    std::vector<_FP16> W = makeRandF16(s.N * s.K);
    const size_t w_bytes = (size_t)s.N * (size_t)s.K * sizeof(_FP16);

    // Persistent staging buffers (not part of the timed phases below).
    NpuBuf Xb((size_t)Mp * s.K * sizeof(float));
    NpuBuf Ab((size_t)Mp * s.N * sizeof(float));
    if (!Xb.ok() || !Ab.ok()) {
      printf("[PhaseTiming] %s: staging alloc failed, skipping\n", s.name);
      continue;
    }
    std::memset(Xb.p, 0, (size_t)Mp * s.K * sizeof(float));
    std::memcpy(Xb.p, A.data(), (size_t)s.M * s.K * sizeof(float));

    // Phase: NPU weight alloc (+ free) per iteration.
    TimeStats t_alloc = timeIt(3, 20, [&]() {
      void *w = nullptr;
      sdkl_npu_alloc(w_bytes, &w);
      if (w)
        sdkl_npu_free(w);
    });

    // Reusable weight buffer for the memcpy / rm2wh / mm phases.
    NpuBuf Wb(w_bytes);
    ASSERT_TRUE(Wb.ok()) << s.name << ": weight alloc failed";

    TimeStats t_memcpy =
      timeIt(3, 20, [&]() { std::memcpy(Wb.p, W.data(), w_bytes); });

    // rm_to_wh is in-place; refill from RM each iteration first (that refill is
    // the same memcpy measured above; here we isolate the conversion cost).
    TimeStats t_rm2wh = timeIt(3, 20, [&]() {
      std::memcpy(Wb.p, W.data(), w_bytes);
      sdkl_cpu_rm_to_wh_f16_inplace((size_t)s.N, (size_t)s.K,
                                    static_cast<_FP16 *>(Wb.p));
    });

    // Convert once, then time the matmul alone.
    std::memcpy(Wb.p, W.data(), w_bytes);
    sdkl_cpu_rm_to_wh_f16_inplace((size_t)s.N, (size_t)s.K,
                                  static_cast<_FP16 *>(Wb.p));
    std::memset(Ab.p, 0, (size_t)Mp * s.N * sizeof(float));
    TimeStats t_mm = timeIt(3, 20, [&]() {
      sdkl_npu_mm_f32f16_f32(
        domain, (int)Mp, s.N, s.K, static_cast<float *>(Ab.p),
        static_cast<const float *>(Xb.p), static_cast<const _Float16 *>(Wb.p));
    });

    std::vector<float> C(s.M * s.N, 0.f);
    TimeStats t_copyback = timeIt(3, 20, [&]() {
      std::memcpy(C.data(), Ab.p, (size_t)s.M * s.N * sizeof(float));
    });

    printf("[PhaseTiming] %-13s alloc=%.3f memcpy=%.3f rm2wh=%.3f mm=%.3f "
           "copyback=%.3f (ms, mean)\n",
           s.name, t_alloc.mean_ms, t_memcpy.mean_ms, t_rm2wh.mean_ms,
           t_mm.mean_ms, t_copyback.mean_ms);
  }
  SUCCEED();
}

// Measures the usable NPU DMA residency budget by allocating fixed-size
// chunks until failure. The SDK has no pool-query API, so this empirical
// value gates the prefill pin cap (PREFILL_WH_PIN_MAX_BYTES). Chunk size
// matches a typical prefill weight so the number reflects the real
// many-small-buffers residency pattern (subject to fragmentation).
TEST_F(HtpKernelTest, PoolProbe_MeasureMaxResidentBytes) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  const size_t chunk = 4ull * 1024 * 1024; // 4 MB, ~one prefill weight
  std::vector<void *> bufs;
  size_t total = 0;
  while (bufs.size() < 4096u) { // 16 GB safety cap; never reached in practice
    void *p = nullptr;
    int err = sdkl_npu_alloc(chunk, &p);
    if (err != 0 || p == nullptr)
      break;
    bufs.push_back(p);
    total += chunk;
  }
  for (void *p : bufs)
    sdkl_npu_free(p);

  const size_t mb = total / (1024u * 1024u);
  printf("[PoolProbe] max resident: %zu MB across %zu chunks of %zu MB\n", mb,
         bufs.size(), chunk / (1024u * 1024u));
  RecordProperty("pool_max_mb", static_cast<int>(mb));
  SUCCEED();
}

// Measures the usable NPU DMA **sustained-pin** budget by allocating
// variably-sized buffers matching real Qwen3-0.6B FC weight sizes and NOT
// freeing them — exactly the pin-once access pattern. This is distinct from
// PoolProbe_MeasureMaxResidentBytes (transient alloc-then-free-all), which
// measured ~4000 MB but is not a valid proxy for how much can be kept pinned
// simultaneously (real ceiling found to be ~48 MB / 9 weights on S25 Ultra).
// Weight sizes: q_proj=4MB (N=2048,K=1024), gate_up_proj=6MB (N=3072,K=1024),
// down_proj=6MB (N=1024,K=3072); cycle repeats 28 times (28 layers × 3 FC).
TEST_F(HtpKernelTest, PoolProbe_MeasureMaxSustainedPinBytes) {
  if (!npu_enabled)
    GTEST_SKIP() << "NPU not available on this device";

  // Sizes in bytes for the three FC weight shapes in Qwen3-0.6B.
  static const size_t kWeightSizes[] = {
    2048ull * 1024 * sizeof(_FP16), // q_proj:       N=2048, K=1024 → 4 MB
    3072ull * 1024 * sizeof(_FP16), // gate_up_proj:  N=3072, K=1024 → 6 MB
    1024ull * 3072 * sizeof(_FP16), // down_proj:     N=1024, K=3072 → 6 MB
  };
  const size_t kNumShapes = sizeof(kWeightSizes) / sizeof(kWeightSizes[0]); // 3
  const size_t kMaxWeights = 84; // 28 layers × 3 FC

  std::vector<void *> bufs;
  size_t total_bytes = 0;

  for (size_t i = 0; i < kMaxWeights; ++i) {
    size_t sz = kWeightSizes[i % kNumShapes];
    void *p = nullptr;
    int err = sdkl_npu_alloc(sz, &p);
    if (err != 0 || p == nullptr)
      break; // pool exhausted — stop, do NOT free any prior allocations
    bufs.push_back(p);
    total_bytes += sz;
  }

  // Cleanup — free only after the full probe is complete.
  for (void *p : bufs)
    sdkl_npu_free(p);

  const size_t mb = total_bytes / (1024u * 1024u);
  printf("[SustainedPinProbe] max pinned: %zu MB across %zu weights "
         "(q/gu/d shapes cycling: 4/6/6 MB)\n",
         mb, bufs.size());
  RecordProperty("sustained_pin_max_mb", static_cast<int>(mb));
  SUCCEED();
}

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
  writePerfCsv();
  printPerfMarkdown();
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
