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
 * apples-to-apples with FP32.
 *
 * Self-contained on purpose: the quantization (per-output-channel symmetric
 * INT8/INT4 weight + zp_corr, per-tensor UINT8 activation) is done inline, and
 * the NPU is driven through the sdkl C API directly (sdkl_npu_initialize +
 * sdkl_cpu_rm_to_wh_* + sdkl_npu_mm_u8i{4,8}_i32 + sdkl_npu_finalize). It links
 * only libsdkl.so (no libnntrainer C++ symbols), which the Android shared
 * library does not reliably export.
 *
 *   - with ENABLE_HEXKL (Hexagon device): the real sdkl kernels run, so
 *     latency is the true NPU latency.
 *   - without it (any host): the integer GEMM the NPU performs is emulated on
 *     the CPU. The integer product is exact, so the relErr equals the
 *     on-device relErr; only latency differs (CPU-emulated, and labelled).
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

#ifdef ENABLE_HEXKL
#include <remote.h> // CDSP_DOMAIN_ID
#include <sdkl.h>   // sdkl_npu_* / sdkl_cpu_rm_to_wh_*
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

// Per-output-channel symmetric weight quant (mirrors quantize_qint{4,8}_weight):
// scale[n] = max_abs(W[n,:]) / q_max, w_q[n,k] = clamp(round(W/scale), ±q_max),
// zp_corr[n] = 128 * sum_k w_q[n,k]. w_q is logical [N,K] (w_q[n*K+k]).
struct WQuant {
  std::vector<int8_t> w_q;
  std::vector<float> scale;
  std::vector<int32_t> zp_corr;
};
static WQuant quantWeight(const std::vector<float> &W, int N, int K,
                          int q_max) {
  WQuant o;
  o.w_q.resize((size_t)N * K);
  o.scale.resize(N);
  o.zp_corr.resize(N);
  for (int n = 0; n < N; ++n) {
    float mx = 0.0f;
    for (int k = 0; k < K; ++k)
      mx = std::max(mx, std::fabs(W[(size_t)n * K + k]));
    float s = mx > 0.0f ? mx / (float)q_max : 1.0f;
    if (!std::isfinite(s) || s <= 0.0f)
      s = 1.0f;
    o.scale[n] = s;
    int32_t rowsum = 0;
    for (int k = 0; k < K; ++k) {
      long q = std::lround(W[(size_t)n * K + k] / s);
      int8_t c = (int8_t)std::max<long>(-q_max, std::min<long>(q_max, q));
      o.w_q[(size_t)n * K + k] = c;
      rowsum += c;
    }
    o.zp_corr[n] = 128 * rowsum;
  }
  return o;
}

// Per-tensor u8 activation quant (zp=128), mirrors shgemm_u8i{4,8}_i32.
static std::vector<uint8_t> quantAct(const std::vector<float> &A, int M, int K,
                                     float &act_scale) {
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

// Dequant: C[m,n] = act_scale * scale[n] * (C_i32[m,n] - zp_corr[n]).
static std::vector<float> dequant(const int32_t *c_i32, int M, int N,
                                  float act_scale, const std::vector<float> &sc,
                                  const std::vector<int32_t> &zp) {
  std::vector<float> c((size_t)M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n)
      c[(size_t)m * N + n] = act_scale * sc[n] *
                             ((float)c_i32[(size_t)m * N + n] - (float)zp[n]);
  return c;
}

struct QResult {
  std::vector<float> C;
  double us = 0.0;  // per-call latency (microseconds)
  bool on_npu = false;
};

#ifdef ENABLE_HEXKL
// RAII NPU buffer.
struct NpuBuf {
  void *p = nullptr;
  explicit NpuBuf(size_t bytes) {
    if (sdkl_npu_alloc(bytes, &p) != 0)
      p = nullptr;
  }
  ~NpuBuf() { if (p) sdkl_npu_free(p); }
  NpuBuf(const NpuBuf &) = delete;
  NpuBuf &operator=(const NpuBuf &) = delete;
  bool ok() const { return p != nullptr; }
};

// Drive the real NPU kernel directly through sdkl. bits = 4 or 8.
// Returns false if a kernel/alloc call fails.
static bool runNpu(int M, int N, int K, int bits, const std::vector<uint8_t> &x,
                   const WQuant &wq, int warmup, int iters, int domain,
                   double &us, std::vector<int32_t> &c_i32) {
  const int Mp = ((M + 63) / 64) * 64;

  // WH-pack the weight (out of the timed loop).
  const size_t nk = (size_t)N * K;
  NpuBuf Xb((size_t)Mp * K), Cb((size_t)Mp * N * sizeof(int32_t));
  if (!Xb.ok() || !Cb.ok())
    return false;
  // Activation: real M rows, pad the rest to zp=128.
  std::memset(Xb.p, 128, (size_t)Mp * K);
  std::memcpy(Xb.p, x.data(), (size_t)M * K);

  bool ok = false;
  if (bits == 4) {
    NpuBuf Win(nk), Wwh(nk);
    if (!Win.ok() || !Wwh.ok())
      return false;
    std::memcpy(Win.p, wq.w_q.data(), nk);
    if (sdkl_cpu_rm_to_wh_i4(static_cast<uint8_t *>(Wwh.p),
                             static_cast<int8_t *>(Win.p), (size_t)K,
                             (size_t)N) != 0)
      return false;
    int rc = 0;
    auto call = [&]() {
      rc = sdkl_npu_mm_u8i4_i32(domain, Mp, N, K, static_cast<int32_t *>(Cb.p),
                                static_cast<const uint8_t *>(Xb.p),
                                static_cast<const uint8_t *>(Wwh.p));
    };
    for (int i = 0; i < warmup; ++i) call();
    double sum = 0.0;
    for (int i = 0; i < iters; ++i) {
      auto t0 = std::chrono::steady_clock::now();
      call();
      auto t1 = std::chrono::steady_clock::now();
      sum += std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    us = iters > 0 ? sum / iters : 0.0;
    ok = (rc == 0);
  } else {
    NpuBuf Wb(nk);
    if (!Wb.ok())
      return false;
    std::memcpy(Wb.p, wq.w_q.data(), nk);
    if (sdkl_cpu_rm_to_wh_i8_inplace((size_t)N, (size_t)K,
                                     static_cast<int8_t *>(Wb.p)) != 0)
      return false;
    int rc = 0;
    auto call = [&]() {
      rc = sdkl_npu_mm_u8i8_i32(domain, Mp, N, K, static_cast<int32_t *>(Cb.p),
                                static_cast<const uint8_t *>(Xb.p),
                                static_cast<const int8_t *>(Wb.p));
    };
    for (int i = 0; i < warmup; ++i) call();
    double sum = 0.0;
    for (int i = 0; i < iters; ++i) {
      auto t0 = std::chrono::steady_clock::now();
      call();
      auto t1 = std::chrono::steady_clock::now();
      sum += std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    us = iters > 0 ? sum / iters : 0.0;
    ok = (rc == 0);
  }
  if (!ok)
    return false;
  // Copy back only the real M rows of the int32 accumulator.
  c_i32.assign((size_t)M * N, 0);
  std::memcpy(c_i32.data(), Cb.p, (size_t)M * N * sizeof(int32_t));
  return true;
}
#endif

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

// Full u8i{bits} path: quant (inline) + integer GEMM (NPU on device, CPU
// emulation on host) + dequant.
static QResult runQuant(int M, int N, int K, const std::vector<float> &A,
                        const std::vector<float> &W, int bits, int warmup,
                        int iters, bool npu_ok, int domain) {
  QResult r;
  const int q_max = (bits == 4) ? 7 : 127;
  WQuant wq = quantWeight(W, N, K, q_max);
  float act_scale = 1.0f;
  auto x = quantAct(A, M, K, act_scale);
  std::vector<int32_t> c_i32;

#ifdef ENABLE_HEXKL
  if (npu_ok && runNpu(M, N, K, bits, x, wq, warmup, iters, domain, r.us,
                       c_i32)) {
    r.on_npu = true;
    r.C = dequant(c_i32.data(), M, N, act_scale, wq.scale, wq.zp_corr);
    return r;
  }
#else
  (void)npu_ok;
  (void)domain;
#endif
  // Host / NPU-unavailable: emulate the integer GEMM on the CPU.
  (void)warmup;
  c_i32.assign((size_t)M * N, 0);
  int it = std::max(1, iters);
  double sum = 0.0;
  for (int i = 0; i < it; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    cpuIntGemm(M, N, K, x.data(), wq.w_q.data(), c_i32.data());
    auto t1 = std::chrono::steady_clock::now();
    sum += std::chrono::duration<double, std::micro>(t1 - t0).count();
  }
  r.us = sum / it;
  r.on_npu = false;
  r.C = dequant(c_i32.data(), M, N, act_scale, wq.scale, wq.zp_corr);
  return r;
}

} // namespace

int main(int argc, char **argv) {
  // Unbuffered stdout so partial output survives a crash in the NPU path.
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  Args a = parseArgs(argc, argv);

  if (a.N % 32 != 0) {
    std::printf("error: N must be a multiple of 32 (got N=%d)\n", a.N);
    return 1;
  }
  if (a.K % 2 != 0) {
    std::printf("error: K must be even for INT4 packing (got K=%d)\n", a.K);
    return 1;
  }

  bool npu_ok = false;
  int domain = 0;
#ifdef ENABLE_HEXKL
  // Same NPU bring-up as HtpBackend: initialize the CDSP session directly so
  // the example depends only on libsdkl.so.
  domain = CDSP_DOMAIN_ID;
  int init_err = sdkl_npu_initialize(domain, nullptr, nullptr);
  npu_ok = (init_err == 0);
  if (!npu_ok)
    std::fprintf(stderr, "[warn] sdkl_npu_initialize failed (err=%d); "
                         "falling back to CPU emulation\n",
                 init_err);
#endif

  std::printf("fc_layer mm comparison (HexKL u8i8 / u8i4 vs FP32)\n");
  std::printf("  proj=%s  M=%d  N=%d  K=%d  (%s)\n", a.proj.c_str(), a.M, a.N,
              a.K,
              npu_ok
                ? "NPU kernels — real device latency"
                : "CPU-emulated integer GEMM — relErr real, latency reference");

  auto A = randVec(a.M * a.K, -1.0f, 1.0f, 42);
  auto W = randVec(a.N * a.K, -0.5f, 0.5f, 1337);

  auto C_ref = gemmF32(a.M, a.N, a.K, A, W);

  std::fprintf(stderr, "[run] u8i8 ...\n");
  QResult r8 = runQuant(a.M, a.N, a.K, A, W, 8, a.warmup, a.iters, npu_ok, domain);
  std::fprintf(stderr, "[run] u8i8 done; u8i4 ...\n");
  QResult r4 = runQuant(a.M, a.N, a.K, A, W, 4, a.warmup, a.iters, npu_ok, domain);
  std::fprintf(stderr, "[run] u8i4 done\n");

  float e8 = relErr(r8.C, C_ref);
  float e4 = relErr(r4.C, C_ref);

  std::printf("\n| method | relErr vs FP32 | latency (us) | engine |\n");
  std::printf("|---|---|---|---|\n");
  std::printf("| u8i8 (INT8 weight) | %.5f | %8.1f | %s |\n", e8, r8.us,
              r8.on_npu ? "NPU/HMX" : "CPU-emulated");
  std::printf("| u8i4 (INT4 weight) | %.5f | %8.1f | %s |\n", e4, r4.us,
              r4.on_npu ? "NPU/HMX" : "CPU-emulated");

  std::printf("\nsummary: INT8 relErr=%.4f  INT4 relErr=%.4f  (INT4/INT8 = "
              "%.2fx error)\n",
              e8, e4, e8 > 0.0f ? e4 / e8 : 0.0f);

#ifdef ENABLE_HEXKL
  if (npu_ok)
    sdkl_npu_finalize(domain);
#endif
  return 0;
}
