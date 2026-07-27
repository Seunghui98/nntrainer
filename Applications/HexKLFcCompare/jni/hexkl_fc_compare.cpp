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
#include <stdexcept>
#include <string>
#include <vector>

#ifdef ENABLE_HEXKL
#include <remote.h> // CDSP_DOMAIN_ID
#include <sdkl.h>   // sdkl_npu_* / sdkl_cpu_rm_to_wh_*

// The production kernels, declared here rather than via <hexkl_mm.h> so this
// stays free of the nntrainer include tree; only libnntrainer.so is needed at
// link time. Used by --engine nntr.
namespace nntrainer {
namespace hmx {
void shgemm_u8i8_i32(unsigned int M, unsigned int N, unsigned int K,
                     const float *A, const int8_t *B_wh, const float *wt_scale,
                     const int32_t *zp_corr, float *C);
void shgemm_u8i4_i32(unsigned int M, unsigned int N, unsigned int K,
                     const float *A, const int8_t *B_wh, const float *wt_scale,
                     const int32_t *zp_corr, float *C);
} // namespace hmx
} // namespace nntrainer
#endif

namespace {

struct Args {
  int M = 64;   // rows = tokens (prefill tile); decode is M=1
  int N = 2048; // out features (q_proj)
  int K = 1024; // in features  (hidden)
  int iters = 30;
  int warmup = 5;
  std::string proj = "q_proj";
  bool sweep = false;  // decompose fixed per-call overhead vs compute
  bool msweep = false; // vary M at fixed N,K: is the kernel weight-bound?
  // Which weight width(s) to run: 4, 8, or 0 = both. --sweep defaults to 4
  // (the QNN-comparable config); the normal compare mode defaults to both.
  int bits = -1;
  // "sdkl": drive sdkl_npu_mm_* directly, with the full phase breakdown.
  // "nntr": call nntrainer's hmx::shgemm_u8i{4,8}_i32, which is what an
  //         fc_layer actually goes through -- including its resident weight
  //         cache and scratch pools. Only cold vs steady are visible there,
  //         since quantization, upload and dequantize happen inside one call.
  std::string engine = "sdkl";
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
  bool iters_set = false, warmup_set = false;
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
    else if (s == "--iters") { next(a.iters); iters_set = true; }
    else if (s == "--warmup") { next(a.warmup); warmup_set = true; }
    else if (s == "--sweep")
      a.sweep = true;
    else if (s == "--msweep")
      a.msweep = true;
    else if (s == "--engine") {
      if (i + 1 < argc)
        a.engine = argv[++i];
    } else if (s == "--bits") {
      if (i + 1 < argc) {
        std::string v = argv[++i];
        a.bits = (v == "both") ? 0 : std::atoi(v.c_str());
      }
    }
    else if (s == "--proj") {
      if (i + 1 < argc) { a.proj = argv[++i]; proj_set = true; }
    } else if (s == "--help" || s == "-h") {
      std::printf(
        "usage: hexkl_fc_compare [--proj q_proj|k_proj|o_proj|ffn_up|ffn_down]"
        " [--M m] [--N n] [--K k] [--iters n] [--warmup n]\n"
        "                        [--bits 4|8|both] [--engine sdkl|nntr]"
        " [--sweep] [--msweep]\n"
        "  default: qwen3-0.6b q_proj  M=64 N=2048 K=1024\n"
        "  --engine sdkl: drive sdkl_npu_mm_* directly (full phase breakdown)\n"
        "  --engine nntr: call nntrainer hmx::shgemm_u8i{4,8}_i32, the path a\n"
        "                 real fc_layer takes; reports cold vs steady, where\n"
        "                 the gap is the weight upload its cache removes\n"
        "  --sweep:  shrink the MAC count at a fixed call structure, to split\n"
        "            fixed per-call overhead (RPC+setup) from compute\n"
        "  --msweep: vary M at fixed N,K to test whether it is weight-bound\n"
        "  env: HEXKL_FC_ITERS, HEXKL_FC_WARMUP\n");
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
  // Iteration counts via env var (explicit --iters/--warmup on the CLI win).
  if (!iters_set)
    if (const char *e = std::getenv("HEXKL_FC_ITERS")) {
      int v = std::atoi(e);
      if (v > 0)
        a.iters = v;
    }
  if (!warmup_set)
    if (const char *e = std::getenv("HEXKL_FC_WARMUP")) {
      int v = std::atoi(e);
      if (v >= 0)
        a.warmup = v;
    }
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

// Per-phase timing (microseconds). "movement" = everything that loads/moves
// data (NPU alloc + host->NPU copies + WH layout pack + NPU->host copy);
// "compute" = the sdkl_npu_mm kernel only.
// Timing split by what a real inference actually pays per call.
//
// In production the weight is quantized and WH-packed OFFLINE (nntr_quantize)
// and would be uploaded to the NPU once and kept resident, exactly like QNN
// keeps weights in its context binary. Only the activation upload, the kernel
// and the result download are truly per-call. The current kernel does not have
// a resident-weight cache for u8i8/u8i4, so it re-pays the weight cost on every
// call -- reported separately so the gap is visible.
struct Timing {
  // --- per call (unavoidable) ---
  double h2d_act_us = 0.0; // host -> NPU activation upload
  double kernel_us = 0.0;  // steady-state sdkl_npu_mm mean (execute)
  double d2h_us = 0.0;     // NPU -> host int32 accumulator download
  // --- one-time in production (per call in the current implementation) ---
  double alloc_us = 0.0;      // sdkl_npu_alloc of X / C / weight buffers
  double h2d_weight_us = 0.0; // host -> NPU weight upload
  double whpack_us = 0.0;     // RM -> WH pack (offline at bake time)
  // --- informational ---
  double cold_us = 0.0; // first execute
  // A mean alone hides how unstable a run was. Once host threads are involved
  // the worker wake-up (futex, across big.LITTLE under DVFS) can dominate a
  // single iteration, and the same configuration has been seen to swing 5x
  // run to run. min is the cleanest read of what the path can do; the spread
  // between min and max says how much of a difference is scheduling noise.
  double kernel_min_us = 0.0;
  double kernel_med_us = 0.0;
  double kernel_max_us = 0.0;

  double perCallUs() const { return h2d_act_us + kernel_us + d2h_us; }
  double oneTimeUs() const { return alloc_us + h2d_weight_us + whpack_us; }
  // What the current implementation actually costs on every call.
  double currentPerCallUs() const { return perCallUs() + oneTimeUs(); }
};

struct QResult {
  std::vector<float> C;
  Timing t;
  bool on_npu = false;
};

/** @brief Fold per-iteration samples into mean/min/median/max. */
static void summarize(std::vector<double> &samples, Timing &tm) {
  if (samples.empty())
    return;
  std::sort(samples.begin(), samples.end());
  double sum = 0.0;
  for (double v : samples)
    sum += v;
  tm.kernel_us = sum / samples.size();
  tm.kernel_min_us = samples.front();
  tm.kernel_med_us = samples[samples.size() / 2];
  tm.kernel_max_us = samples.back();
}

static inline double usSince(std::chrono::steady_clock::time_point s) {
  return std::chrono::duration<double, std::micro>(
           std::chrono::steady_clock::now() - s)
    .count();
}

#ifdef ENABLE_HEXKL
// RAII NPU buffer.
struct NpuBuf {
  void *p = nullptr;
  explicit NpuBuf(size_t bytes) {
    if (bytes != 0 && sdkl_npu_alloc(bytes, &p) != 0)
      p = nullptr;
  }
  ~NpuBuf() { if (p) sdkl_npu_free(p); }
  NpuBuf(const NpuBuf &) = delete;
  NpuBuf &operator=(const NpuBuf &) = delete;
  bool ok() const { return p != nullptr; }
};

// Drive the real NPU kernel directly through sdkl, timing each phase so data
// movement is separated from pure compute. bits = 4 or 8. Returns false on a
// kernel/alloc failure.
static bool runNpu(int M, int N, int K, int bits, const std::vector<uint8_t> &x,
                   const WQuant &wq, int warmup, int iters, int domain,
                   Timing &tm, std::vector<int32_t> &c_i32) {
  using clk = std::chrono::steady_clock;
  const int Mp = ((M + 63) / 64) * 64;
  const size_t nk = (size_t)N * K;

  // --- alloc: X, C, and weight buffers (i4 needs an rm source + packed dst;
  // i8 packs in place in a single buffer) ---
  auto ta = clk::now();
  NpuBuf Xb((size_t)Mp * K), Cb((size_t)Mp * N * sizeof(int32_t));
  NpuBuf Wi4src(bits == 4 ? nk : 0); // i4 only: RM int4 source
  NpuBuf Wbuf(nk);                   // WH weight (packed dst for i4, in-place i8)
  if (!Xb.ok() || !Cb.ok() || !Wbuf.ok() || (bits == 4 && !Wi4src.ok()))
    return false;
  tm.alloc_us = usSince(ta);

  // --- H2D activation (per call): pad rows to zp=128, then upload ---
  auto tha = clk::now();
  std::memset(Xb.p, 128, (size_t)Mp * K);
  std::memcpy(Xb.p, x.data(), (size_t)M * K);
  tm.h2d_act_us = usSince(tha);

  // --- H2D weight (one-time in production; per call today) ---
  auto thw = clk::now();
  std::memcpy(bits == 4 ? Wi4src.p : Wbuf.p, wq.w_q.data(), nk);
  tm.h2d_weight_us = usSince(thw);

  // --- WH pack: RM -> WH weight layout ---
  auto tp = clk::now();
  if (bits == 4) {
    if (sdkl_cpu_rm_to_wh_i4(static_cast<uint8_t *>(Wbuf.p),
                             static_cast<int8_t *>(Wi4src.p), (size_t)K,
                             (size_t)N) != 0)
      return false;
  } else {
    if (sdkl_cpu_rm_to_wh_i8_inplace((size_t)N, (size_t)K,
                                     static_cast<int8_t *>(Wbuf.p)) != 0)
      return false;
  }
  tm.whpack_us = usSince(tp);

  // --- compute: the sdkl matmul kernel only ---
  int rc = 0;
  auto call = [&]() {
    rc = (bits == 4)
           ? sdkl_npu_mm_u8i4_i32(domain, Mp, N, K,
                                  static_cast<int32_t *>(Cb.p),
                                  static_cast<const uint8_t *>(Xb.p),
                                  static_cast<const uint8_t *>(Wbuf.p))
           : sdkl_npu_mm_u8i8_i32(domain, Mp, N, K,
                                  static_cast<int32_t *>(Cb.p),
                                  static_cast<const uint8_t *>(Xb.p),
                                  static_cast<const int8_t *>(Wbuf.p));
  };
  // Cold run: the very first execute (mirrors QNN's "Cold run" — carries any
  // first-call device-side setup).
  auto tcold = clk::now();
  call();
  tm.cold_us = usSince(tcold);
  // Warmup (discarded), then steady-state NetRun mean.
  for (int i = 0; i < warmup; ++i)
    call();
  std::vector<double> samples;
  samples.reserve(iters > 0 ? iters : 0);
  for (int i = 0; i < iters; ++i) {
    auto t0 = clk::now();
    call();
    samples.push_back(usSince(t0));
  }
  summarize(samples, tm);
  if (rc != 0)
    return false;

  // --- D2H: copy back the real M rows of the int32 accumulator ---
  auto td = clk::now();
  c_i32.assign((size_t)M * N, 0);
  std::memcpy(c_i32.data(), Cb.p, (size_t)M * N * sizeof(int32_t));
  tm.d2h_us = usSince(td);
  return true;
}
#endif

#ifdef ENABLE_HEXKL
// Drive the production kernel instead of sdkl directly. It takes FP32
// activations and the WH-packed weight and does quantization, buffer
// management and dequantize internally, so the only visible split is the first
// call (which uploads the weight into NPU memory) versus the steady state
// (weight resident, scratch reused). Returns false if the WH pack fails.
//
// The WH buffer must stay alive for the whole loop: the resident cache is
// keyed by that host pointer, so freeing it between calls would defeat it.
static bool runViaNntrainer(int M, int N, int K, int bits,
                            const std::vector<float> &A, const WQuant &wq,
                            int warmup, int iters, Timing &tm,
                            std::vector<float> &C_out) {
  // libnntrainer owns the CDSP session and opens it lazily inside the first
  // kernel call, so nothing here may touch sdkl_npu_alloc before that happens.
  // Prime it with a throwaway matmul fed entirely from host memory: the kernel
  // resolves HtpBackend::global() (which performs the init) before it allocates
  // anything, and it copies the weight in from wherever the caller put it. The
  // weight is not WH-packed, so the result is meaningless -- it is discarded.
  // Doing this first also keeps the backend bring-up out of the cold-run figure.
  {
    const int dM = 64, dN = 32, dK = 32;
    std::vector<int8_t> dwh(static_cast<size_t>(dN) * dK / 2, 0x11);
    std::vector<float> dA(static_cast<size_t>(dM) * dK, 0.5f);
    std::vector<float> dC(static_cast<size_t>(dM) * dN, 0.0f);
    std::vector<float> dscale(dN, 1.0f / 7.0f);
    std::vector<int32_t> dzp(dN, 0);
    try {
      nntrainer::hmx::shgemm_u8i4_i32(dM, dN, dK, dA.data(), dwh.data(),
                                      dscale.data(), dzp.data(), dC.data());
    } catch (const std::exception &e) {
      std::fprintf(stderr, "[warn] backend priming call failed: %s\n", e.what());
      return false;
    }
  }

  // SDKL is up now, so the WH pack below can use NPU-accessible buffers (the
  // layout path is documented to fault on host-only memory).
  const size_t nk = static_cast<size_t>(N) * K;
  NpuBuf Wsrc(nk), Wwh(nk);
  if (!Wsrc.ok() || !Wwh.ok())
    return false;
  std::memcpy(Wsrc.p, wq.w_q.data(), nk);

  const int8_t *B_wh = nullptr;
  if (bits == 4) {
    if (sdkl_cpu_rm_to_wh_i4(static_cast<uint8_t *>(Wwh.p),
                             static_cast<int8_t *>(Wsrc.p), (size_t)K,
                             (size_t)N) != 0)
      return false;
    B_wh = static_cast<const int8_t *>(Wwh.p);
  } else {
    if (sdkl_cpu_rm_to_wh_i8_inplace((size_t)N, (size_t)K,
                                     static_cast<int8_t *>(Wsrc.p)) != 0)
      return false;
    B_wh = static_cast<const int8_t *>(Wsrc.p);
  }

  C_out.assign(static_cast<size_t>(M) * N, 0.0f);
  auto call = [&]() {
    if (bits == 4)
      nntrainer::hmx::shgemm_u8i4_i32((unsigned)M, (unsigned)N, (unsigned)K,
                                      A.data(), B_wh, wq.scale.data(),
                                      wq.zp_corr.data(), C_out.data());
    else
      nntrainer::hmx::shgemm_u8i8_i32((unsigned)M, (unsigned)N, (unsigned)K,
                                      A.data(), B_wh, wq.scale.data(),
                                      wq.zp_corr.data(), C_out.data());
  };

  try {
    auto tc = std::chrono::steady_clock::now();
    call(); // first call on the real weight: uploads it into NPU memory
    tm.cold_us = usSince(tc);

    for (int i = 0; i < warmup; ++i)
      call();
    std::vector<double> samples;
    samples.reserve(iters > 0 ? iters : 0);
    for (int i = 0; i < iters; ++i) {
      auto t0 = std::chrono::steady_clock::now();
      call();
      samples.push_back(usSince(t0));
    }
    summarize(samples, tm);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[warn] nntrainer kernel failed: %s\n", e.what());
    return false;
  }
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
                        int iters, bool npu_ok, int domain,
                        bool via_nntr = false) {
  QResult r;
  const int q_max = (bits == 4) ? 7 : 127;
  WQuant wq = quantWeight(W, N, K, q_max);
  float act_scale = 1.0f;
  auto x = quantAct(A, M, K, act_scale);
  std::vector<int32_t> c_i32;

#ifdef ENABLE_HEXKL
  if (npu_ok && via_nntr &&
      runViaNntrainer(M, N, K, bits, A, wq, warmup, iters, r.t, r.C)) {
    r.on_npu = true;
    return r; // the kernel already returned dequantized FP32
  }
  if (npu_ok && !via_nntr &&
      runNpu(M, N, K, bits, x, wq, warmup, iters, domain, r.t, c_i32)) {
    r.on_npu = true;
    r.C = dequant(c_i32.data(), M, N, act_scale, wq.scale, wq.zp_corr);
    return r;
  }
#else
  (void)npu_ok;
  (void)domain;
  (void)via_nntr;
#endif
  // Host / NPU-unavailable: emulate the integer GEMM on the CPU. No data
  // movement, so only the compute (kernel) phase is timed.
  (void)warmup;
  c_i32.assign((size_t)M * N, 0);
  int it = std::max(1, iters);
  std::vector<double> samples;
  samples.reserve(it);
  for (int i = 0; i < it; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    cpuIntGemm(M, N, K, x.data(), wq.w_q.data(), c_i32.data());
    samples.push_back(usSince(t0));
  }
  summarize(samples, r.t);
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

  if (a.engine != "nntr" && a.engine != "sdkl") {
    std::printf("unknown --engine '%s' (using sdkl)\n", a.engine.c_str());
    a.engine = "sdkl";
  }
  const bool via_nntr = (a.engine == "nntr");

  bool npu_ok = false;
  int domain = 0;
  double init_us = 0.0;
#ifdef ENABLE_HEXKL
  domain = CDSP_DOMAIN_ID;
  if (via_nntr) {
    // libnntrainer's HtpBackend owns the CDSP session and brings it up lazily
    // on the first kernel call. Initializing here too makes that second
    // sdkl_npu_initialize fail (Err=1), which leaves HtpBackend marked
    // disabled -- so its teardown then skips freeing the NPU buffers it holds.
    // Leave the session to it; the throwaway call in runViaNntrainer absorbs
    // the init so it does not land in the cold-run figure.
    npu_ok = true;
  } else {
    // Same bring-up HtpBackend performs. Timed = QNN "One-time init".
    auto tinit = std::chrono::steady_clock::now();
    int init_err = sdkl_npu_initialize(domain, nullptr, nullptr);
    init_us = usSince(tinit);
    npu_ok = (init_err == 0);
    if (!npu_ok)
      std::fprintf(stderr, "[warn] sdkl_npu_initialize failed (err=%d); "
                           "falling back to CPU emulation\n",
                   init_err);
  }
#endif

  std::printf("fc_layer mm comparison (HexKL u8i8 / u8i4 vs FP32)\n");
  std::printf("  proj=%s  M=%d  N=%d  K=%d  iters=%d warmup=%d  engine=%s\n",
              a.proj.c_str(), a.M, a.N, a.K, a.iters, a.warmup,
              via_nntr ? "nntr (hmx::shgemm_*, production path)"
                       : "sdkl (direct sdkl_npu_mm_*)");
  std::printf("  %s\n",
              npu_ok
                ? "NPU kernels — real device latency"
                : "CPU-emulated integer GEMM — relErr real, latency reference");

  // ---- M sweep: is the kernel weight-bandwidth-bound? ----------------------
  // At fixed N,K the weight bytes are constant while the MAC count grows
  // linearly with M. So:
  //   - weight-bandwidth-bound -> time barely grows with M (weight traffic,
  //     which is constant, dominates) and us/MMAC collapses.
  //   - compute-bound          -> time grows ~linearly with M and us/MMAC is
  //     flat.
  // This is the matmul equivalent of the weight reuse a conv gets for free by
  // sweeping a small kernel over many spatial positions, and it says directly
  // whether raising M (more tokens per call) is the lever.
  if (a.msweep) {
    const int Ms[] = {1, 8, 32, 64, 128, 256, 512};
    const int sweep_bits = (a.bits < 0) ? 4 : a.bits;
    const size_t wbytes =
      (size_t)a.N * a.K / (sweep_bits == 4 ? 2 : 1); // packed weight bytes
    std::printf("\nM sweep (N=%d K=%d, %s, iters=%d): weight=%.2f MiB fixed\n",
                a.N, a.K, sweep_bits == 4 ? "u8i4" : "u8i8", a.iters,
                (double)wbytes / (1024.0 * 1024.0));
    std::printf("|    M |   MACs(M) | exec us | us/MMAC | weight GB/s |\n");
    std::printf("|---|---|---|---|---|\n");
    auto Wm = randVec(a.N * a.K, -0.5f, 0.5f, 1337);
    for (int m : Ms) {
      auto Am = randVec(m * a.K, -1.0f, 1.0f, 42);
      QResult q = runQuant(m, a.N, a.K, Am, Wm, sweep_bits, a.warmup, a.iters,
                           npu_ok, domain, via_nntr);
      double macs = (double)m * a.N * a.K / 1e6;
      double gbs = (double)wbytes / (q.t.kernel_us * 1e-6) / 1e9;
      std::printf("| %4d | %9.1f | %7.1f | %7.3f | %11.2f |\n", m, macs,
                  q.t.kernel_us, q.t.kernel_us / macs, gbs);
      std::fflush(stdout);
    }
    std::printf("\nif us/MMAC collapses as M grows -> weight-bandwidth-bound:\n"
                "  raising M (more tokens per call) is the lever, and keeping\n"
                "  the weight resident on-chip is what QNN's conv path buys.\n"
                "if us/MMAC stays flat -> compute-bound: M does not help.\n");
#ifdef ENABLE_HEXKL
    if (npu_ok && !via_nntr)
      sdkl_npu_finalize(domain);
#endif
    return 0;
  }

  // ---- overhead decomposition sweep ---------------------------------------
  // Shrink the MAC count while keeping the call structure identical. The
  // smallest shape's execute time is essentially the fixed per-call cost
  // (FastRPC round trip + kernel setup); subtracting it from a full-size shape
  // yields the actual compute time, which is what QNN's device-timeline "FC
  // (weight load + compute)" number measures.
  if (a.sweep) {
    struct S { int M, N, K; };
    const S shapes[] = {
      {a.M, 32, 32},      {a.M, 32, 1024},   {a.M, 256, 1024},
      {a.M, 2048, 128},   {a.M, 2048, 1024}, {a.M, 3072, 1024},
    };
    // --sweep defaults to u8i4 only (the QNN-comparable config); --bits 8 or
    // --bits both opts into the slower INT8 runs.
    const int sweep_bits = (a.bits < 0) ? 4 : a.bits;
    const bool do8 = (sweep_bits == 8 || sweep_bits == 0);
    const bool do4 = (sweep_bits == 4 || sweep_bits == 0);
    std::printf("\nOverhead sweep (M=%d, iters=%d, bits=%s): execute vs MACs\n",
                a.M, a.iters,
                sweep_bits == 0 ? "both" : (sweep_bits == 4 ? "u8i4" : "u8i8"));
    std::printf("|     N |    K |   MACs(M) |");
    if (do8) std::printf(" u8i8 exec us |");
    if (do4) std::printf(" u8i4 exec us |");
    std::printf("\n|---|---|---|");
    if (do8) std::printf("---|");
    if (do4) std::printf("---|");
    std::printf("\n");
    double base8 = -1.0, base4 = -1.0, full8 = 0.0, full4 = 0.0;
    for (const auto &s : shapes) {
      auto As = randVec(s.M * s.K, -1.0f, 1.0f, 42);
      auto Ws = randVec(s.N * s.K, -0.5f, 0.5f, 1337);
      double macs = (double)s.M * s.N * s.K / 1e6;
      std::printf("| %5d | %4d | %9.1f |", s.N, s.K, macs);
      if (do8) {
        QResult q8 =
          runQuant(s.M, s.N, s.K, As, Ws, 8, a.warmup, a.iters, npu_ok,
                   domain, via_nntr);
        std::printf(" %12.1f |", q8.t.kernel_us);
        if (base8 < 0.0) base8 = q8.t.kernel_us;
        full8 = q8.t.kernel_us;
      }
      if (do4) {
        QResult q4 =
          runQuant(s.M, s.N, s.K, As, Ws, 4, a.warmup, a.iters, npu_ok,
                   domain, via_nntr);
        std::printf(" %12.1f |", q4.t.kernel_us);
        if (base4 < 0.0) base4 = q4.t.kernel_us;
        full4 = q4.t.kernel_us;
      }
      std::printf("\n");
      std::fflush(stdout);
    }
    std::printf("\nfixed per-call overhead (smallest shape):");
    if (do8) std::printf("  u8i8=%.1f us", base8);
    if (do4) std::printf("  u8i4=%.1f us", base4);
    std::printf("\ncompute-only at the largest shape       :");
    if (do8) std::printf("  u8i8=%.1f us", full8 - base8);
    if (do4) std::printf("  u8i4=%.1f us", full4 - base4);
    std::printf("\n  (compare against the QNN device-timeline 'FC (weight load "
                "+ compute)' number)\n");
#ifdef ENABLE_HEXKL
    if (npu_ok && !via_nntr)
      sdkl_npu_finalize(domain);
#endif
    return 0;
  }

  auto A = randVec(a.M * a.K, -1.0f, 1.0f, 42);
  auto W = randVec(a.N * a.K, -0.5f, 0.5f, 1337);

  auto C_ref = gemmF32(a.M, a.N, a.K, A, W);

  // --bits selects which width(s) to run; default (both) keeps the comparison.
  const bool run8 = (a.bits < 0 || a.bits == 0 || a.bits == 8);
  const bool run4 = (a.bits < 0 || a.bits == 0 || a.bits == 4);
  QResult r8, r4;
  if (run8) {
    std::fprintf(stderr, "[run] u8i8 ...\n");
    r8 = runQuant(a.M, a.N, a.K, A, W, 8, a.warmup, a.iters, npu_ok, domain,
                  via_nntr);
    std::fprintf(stderr, "[run] u8i8 done\n");
  }
  if (run4) {
    std::fprintf(stderr, "[run] u8i4 ...\n");
    r4 = runQuant(a.M, a.N, a.K, A, W, 4, a.warmup, a.iters, npu_ok, domain,
                  via_nntr);
    std::fprintf(stderr, "[run] u8i4 done\n");
  }

  float e8 = run8 ? relErr(r8.C, C_ref) : 0.0f;
  float e4 = run4 ? relErr(r4.C, C_ref) : 0.0f;

  // ---- QNN-style phase breakdown (microseconds) ----------------------------
  // Mapping to a qnn-net-run profile:
  //   One-time init            <- sdkl_npu_initialize (once per process)
  //   Cold run                 <- first execute (first sdkl_npu_mm call)
  //   Steady-state NetRun      <- mean sdkl_npu_mm (host-observed execute:
  //                               RPC round-trip + device compute + return)
  //   data movement            <- host-side alloc + H2D + WH pack + D2H
  // sdkl does not expose the intra-execute device timeline (convert / FC /
  // reshape / writeback), so those sub-phases cannot be split here.
  std::printf("\nPhase                                    Time (us)\n");
  std::printf("One-time init (sdkl_npu_initialize)      %10.1f\n", init_us);
  auto dumpMethod = [&](const char *tag, float err, const QResult &r) {
    std::printf("[%s]  relErr=%.5f  engine=%s\n", tag, err,
                r.on_npu ? "NPU/HMX" : "CPU-emulated");
    if (via_nntr) {
      // The production kernel does quantization, buffer management and
      // dequantize inside one call, so the phases below are not observable
      // from here. Cold vs steady is the whole story: the gap is the weight
      // upload that the resident cache removes from every later call.
      std::printf("  Cold run (first call, uploads weight)  %10.1f\n",
                  r.t.cold_us);
      std::printf("  Steady state (weight resident)         %10.1f   <- "
                  "per-call cost (mean)\n",
                  r.t.kernel_us);
      std::printf("    min / median / max                   %10.1f /%8.1f /"
                  "%8.1f\n",
                  r.t.kernel_min_us, r.t.kernel_med_us, r.t.kernel_max_us);
      return;
    }
    std::printf("  Cold run (first execute)               %10.1f\n",
                r.t.cold_us);
    std::printf("  PER CALL (unavoidable)                 %10.1f\n",
                r.t.perCallUs());
    std::printf("    H2D activation upload                %10.1f\n",
                r.t.h2d_act_us);
    std::printf("    Steady NetRun (mean execute)         %10.1f   <- compute\n",
                r.t.kernel_us);
    std::printf("      min / median / max                 %10.1f /%8.1f /"
                "%8.1f\n",
                r.t.kernel_min_us, r.t.kernel_med_us, r.t.kernel_max_us);
    std::printf("    D2H int32 accumulator                %10.1f\n", r.t.d2h_us);
    std::printf("  ONE-TIME in production                 %10.1f\n",
                r.t.oneTimeUs());
    std::printf("    alloc (X/C/W buffers)                %10.1f\n",
                r.t.alloc_us);
    std::printf("    H2D weight upload                    %10.1f\n",
                r.t.h2d_weight_us);
    std::printf("    WH pack (offline at bake time)       %10.1f\n",
                r.t.whpack_us);
    std::printf("  => today's per-call cost               %10.1f  (no resident"
                "-weight cache yet)\n",
                r.t.currentPerCallUs());
  };
  if (run8)
    dumpMethod("u8i8", e8, r8);
  if (run4)
    dumpMethod("u8i4", e4, r4);

  // Compact comparison table. "per-call (ideal)" is what the kernel would cost
  // once the weight is uploaded once and kept resident (what QNN does via its
  // context binary); "today" is what the current implementation pays per call.
  std::printf("\n| method | relErr vs FP32 | compute us | per-call ideal us | "
              "today us | engine |\n");
  std::printf("|---|---|---|---|---|---|\n");
  if (run8)
    std::printf("| u8i8 (INT8 weight) | %.5f | %8.1f | %8.1f | %8.1f | %s |\n",
                e8, r8.t.kernel_us, r8.t.perCallUs(), r8.t.currentPerCallUs(),
                r8.on_npu ? "NPU/HMX" : "CPU-emulated");
  if (run4)
    std::printf("| u8i4 (INT4 weight) | %.5f | %8.1f | %8.1f | %8.1f | %s |\n",
                e4, r4.t.kernel_us, r4.t.perCallUs(), r4.t.currentPerCallUs(),
                r4.on_npu ? "NPU/HMX" : "CPU-emulated");

  if (run8 && run4)
    std::printf("\nsummary: INT8 relErr=%.4f  INT4 relErr=%.4f  (INT4/INT8 = "
                "%.2fx error);  compute u8i4=%.1fus u8i8=%.1fus\n",
                e8, e4, e8 > 0.0f ? e4 / e8 : 0.0f, r4.t.kernel_us,
                r8.t.kernel_us);

#ifdef ENABLE_HEXKL
  if (npu_ok && !via_nntr)
    sdkl_npu_finalize(domain);
#endif
  return 0;
}
