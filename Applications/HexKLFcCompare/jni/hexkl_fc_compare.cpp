// SPDX-License-Identifier: Apache-2.0
/**
 * @file   hexkl_fc_compare.cpp
 * @date   17 Jul 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Standalone fc_layer mm comparison: HexKL u8i8 vs u8i4 vs FP32.
 *
 * Runs a single fully-connected matmul  C[M,N] = A[M,K] * W[K,N]  through the
 * HTP u8i8 and u8i4 quantized paths and reports accuracy (relErr vs an FP32
 * reference) and latency. This is deliberately NOT the qwen3-0.6b model: it is
 * one fc_layer's matmul in isolation, so the quantized GEMM can be compared
 * apples-to-apples with FP32 (and, on a QNN build, with a matched QNN matmul).
 *
 * The weight quantization is the production quantize_qint{4,8}_weight
 * (per-output-channel symmetric INT8/INT4 + zp_corr); the activation is
 * per-tensor UINT8 (zp=128). The integer accumulator C_i32 = sum_k X_u8 * W_q
 * is what the NPU op computes:
 *   - with ENABLE_HEXKL (on a Hexagon device): the real
 *     sdkl_npu_mm_u8i{4,8}_i32 kernels run, so latency is the true NPU latency.
 *   - without it (any host): the same integer GEMM is evaluated on the CPU.
 *     Because the integer product is exact either way, the relErr printed on
 *     the host equals the on-device relErr; only the latency differs (host
 *     latency is CPU-emulated and marked as such).
 *
 * Default shape = qwen3-0.6b q_proj (M=64 prefill tile, N=2048, K=1024);
 * override with --M/--N/--K or pick a projection with --proj.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <quantizer.h>
#include <tensor.h>

#ifdef ENABLE_HEXKL
#include <hexkl_mm.h> // nntrainer::hmx::shgemm_u8i{4,8}_i32
#endif

namespace {

struct Args {
  int M = 64;   // rows = tokens (prefill tile); decode is M=1
  int N = 2048; // out features (q_proj)
  int K = 1024; // in features  (hidden)
  int iters = 30;
  int warmup = 5;
  std::string proj = "q_proj";
};

// qwen3-0.6b projection presets (hidden 1024, intermediate 3072, GQA
// q=2048/kv=1024). K = in features, N = out features.
static bool applyProj(const std::string &name, int &N, int &K) {
  if (name == "q_proj") { N = 2048; K = 1024; return true; }
  if (name == "k_proj" || name == "v_proj" || name == "o_proj") {
    N = 1024; K = 1024; return true;
  }
  if (name == "ffn_gate" || name == "ffn_up") { N = 3072; K = 1024; return true; }
  if (name == "ffn_down") { N = 1024; K = 3072; return true; }
  return false;
}

static Args parseArgs(int argc, char **argv) {
  Args a;
  bool proj_set = false, n_set = false, k_set = false;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto next = [&](int &dst) {
      if (i + 1 < argc)
        dst = std::atoi(argv[++i]);
    };
    if (s == "--M")
      next(a.M);
    else if (s == "--N") { next(a.N); n_set = true; }
    else if (s == "--K") { next(a.K); k_set = true; }
    else if (s == "--iters")
      next(a.iters);
    else if (s == "--warmup")
      next(a.warmup);
    else if (s == "--proj") {
      if (i + 1 < argc) { a.proj = argv[++i]; proj_set = true; }
    } else if (s == "--help" || s == "-h") {
      std::printf(
        "usage: hexkl_fc_compare [--proj q_proj|k_proj|o_proj|ffn_up|ffn_down]"
        " [--M m] [--N n] [--K k] [--iters n] [--warmup n]\n"
        "  default: qwen3-0.6b q_proj  M=64 N=2048 K=1024\n");
      std::exit(0);
    }
  }
  // --N/--K win over --proj; otherwise expand the preset.
  if (proj_set && !(n_set && k_set)) {
    int pn = a.N, pk = a.K;
    if (applyProj(a.proj, pn, pk)) {
      if (!n_set) a.N = pn;
      if (!k_set) a.K = pk;
    } else {
      std::printf("unknown --proj '%s' (using N=%d K=%d)\n", a.proj.c_str(),
                  a.N, a.K);
    }
  }
  if (!proj_set && (n_set || k_set))
    a.proj = "custom";
  return a;
}

static std::vector<float> randVec(int n, float lo, float hi, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto &x : v)
    x = dist(rng);
  return v;
}

// Build a [1,1,N,K] FP32 tensor (logical [N,K]) from a row-major buffer.
static nntrainer::Tensor makeWeightNK(const std::vector<float> &W, int N,
                                      int K) {
  nntrainer::Tensor t(
    1, 1, N, K, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      t.setValue(0, 0, n, k, W[(size_t)n * K + k]);
  return t;
}

// Per-tensor u8 activation quant (zp=128), mirrors shgemm_u8i{4,8}_i32.
static std::vector<uint8_t> quantizeActU8(const std::vector<float> &A, int M,
                                          int K, float &act_scale) {
  float mx = 0.0f;
  for (int i = 0; i < M * K; ++i)
    if (std::isfinite(A[i]))
      mx = std::max(mx, std::fabs(A[i]));
  act_scale = mx > 0.0f ? mx / 127.0f : 1.0f;
  const float inv = 1.0f / act_scale;
  std::vector<uint8_t> x((size_t)M * K);
  for (int i = 0; i < M * K; ++i) {
    float q = std::round(A[i] * inv) + 128.0f;
    q = std::max(0.0f, std::min(255.0f, q));
    x[i] = (uint8_t)q;
  }
  return x;
}

// FP32 reference: C[m,n] = sum_k A[m,k] * W[n,k]  (W stored [N,K]).
static std::vector<float> gemmF32(int M, int N, int K,
                                  const std::vector<float> &A,
                                  const std::vector<float> &W) {
  std::vector<float> c((size_t)M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k)
        acc += A[(size_t)m * K + k] * W[(size_t)n * K + k];
      c[(size_t)m * N + n] = acc;
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

// One quantized result: dequantized output + how it was produced.
struct QResult {
  std::vector<float> C;
  double ms = 0.0;   // per-call latency (mean)
  bool on_npu = false;
};

// CPU-exact integer GEMM: C_i32[m,n] = sum_k X_u8[m,k] * W_q[n,k].
static void cpuIntGemm(int M, int N, int K, const uint8_t *X, const int8_t *W,
                       int32_t *C) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      int32_t acc = 0;
      for (int k = 0; k < K; ++k)
        acc += (int32_t)X[(size_t)m * K + k] * (int32_t)W[(size_t)n * K + k];
      C[(size_t)m * N + n] = acc;
    }
}

// Run the u8i{bits} path: production weight quant + activation quant + integer
// GEMM (real NPU kernel on device, CPU emulation on host) + dequant.
static QResult runQuant(int M, int N, int K, const std::vector<float> &A,
                        const std::vector<float> &W, int bits, int warmup,
                        int iters) {
  QResult r;
  nntrainer::Tensor wt = makeWeightNK(W, N, K);
  nntrainer::Tensor q = (bits == 4)
                          ? nntrainer::quantize_qint4_weight(wt, false)
                          : nntrainer::quantize_qint8_weight(wt, false);
  const int8_t *w_q = q.getData<int8_t>();      // WH-packed (device) / [N,K] (host)
  const float *scale = q.getScale<float>();     // [N]
  const int32_t *zp_corr = q.getZpCorr<int32_t>(); // [N]

#ifdef ENABLE_HEXKL
  // Real NPU kernel: dequantizes internally, returns FP32 directly. It also
  // needs the WH-packed weight; quantize_qint{4,8}_weight already produced the
  // WH bytes in w_q under ENABLE_HEXKL, so hand them straight to the kernel.
  std::vector<float> C_out((size_t)M * N, 0.0f);
  auto call = [&]() {
    if (bits == 4)
      nntrainer::hmx::shgemm_u8i4_i32((unsigned)M, (unsigned)N, (unsigned)K,
                                      A.data(), w_q, scale, zp_corr,
                                      C_out.data());
    else
      nntrainer::hmx::shgemm_u8i8_i32((unsigned)M, (unsigned)N, (unsigned)K,
                                      A.data(), w_q, scale, zp_corr,
                                      C_out.data());
  };
  for (int i = 0; i < warmup; ++i)
    call();
  double sum = 0.0;
  for (int i = 0; i < iters; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    call();
    auto t1 = std::chrono::steady_clock::now();
    sum += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  r.ms = iters > 0 ? sum / iters : 0.0;
  r.on_npu = true;
  r.C = std::move(C_out);
#else
  // Host: emulate the NPU integer GEMM on the CPU, then dequant with the exact
  // production formula. relErr is identical to on-device; latency is CPU-only.
  (void)warmup;
  float act_scale = 1.0f;
  auto x_u8 = quantizeActU8(A, M, K, act_scale);
  std::vector<int32_t> c_i32((size_t)M * N, 0);
  double sum = 0.0;
  int it = std::max(1, iters);
  for (int i = 0; i < it; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    cpuIntGemm(M, N, K, x_u8.data(), w_q, c_i32.data());
    auto t1 = std::chrono::steady_clock::now();
    sum += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  r.ms = sum / it;
  r.on_npu = false;
  r.C.assign((size_t)M * N, 0.0f);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n)
      r.C[(size_t)m * N + n] =
        act_scale * scale[n] *
        ((float)c_i32[(size_t)m * N + n] - (float)zp_corr[n]);
#endif
  return r;
}

} // namespace

int main(int argc, char **argv) {
  Args a = parseArgs(argc, argv);

  if (a.N % 32 != 0) {
    std::printf("error: N must be a multiple of 32 (got N=%d)\n", a.N);
    return 1;
  }
  if (a.K % 2 != 0) {
    std::printf("error: K must be even for INT4 packing (got K=%d)\n", a.K);
    return 1;
  }

  std::printf("fc_layer mm comparison (HexKL u8i8 / u8i4 vs FP32)\n");
  std::printf("  proj=%s  M=%d  N=%d  K=%d  (%s)\n", a.proj.c_str(), a.M, a.N,
              a.K,
#ifdef ENABLE_HEXKL
              "NPU kernels — real device latency"
#else
              "host: CPU-emulated integer GEMM — relErr real, latency reference"
#endif
  );

  auto A = randVec(a.M * a.K, -1.0f, 1.0f, 42);
  auto W = randVec(a.N * a.K, -0.5f, 0.5f, 1337);

  auto C_ref = gemmF32(a.M, a.N, a.K, A, W);

  QResult r8 = runQuant(a.M, a.N, a.K, A, W, 8, a.warmup, a.iters);
  QResult r4 = runQuant(a.M, a.N, a.K, A, W, 4, a.warmup, a.iters);

  float e8 = relErr(r8.C, C_ref);
  float e4 = relErr(r4.C, C_ref);

  std::printf("\n| method | relErr vs FP32 | latency (ms) | engine |\n");
  std::printf("|---|---|---|---|\n");
  std::printf("| u8i8 (INT8 weight) | %.5f | %.4f | %s |\n", e8, r8.ms,
              r8.on_npu ? "NPU/HMX" : "CPU-emulated");
  std::printf("| u8i4 (INT4 weight) | %.5f | %.4f | %s |\n", e4, r4.ms,
              r4.on_npu ? "NPU/HMX" : "CPU-emulated");

  std::printf("\nsummary: INT8 relErr=%.4f  INT4 relErr=%.4f  (INT4/INT8 = "
              "%.2fx error)\n",
              e8, e4, e8 > 0.0f ? e4 / e8 : 0.0f);
  return 0;
}
