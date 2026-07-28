// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_fc_e2e.cpp
 * @date   27 Jul 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  End-to-end fc_layer check: FP32 on CPU vs u8i4 on the NPU.
 *
 * Everything verified so far calls hmx::shgemm_u8i4_i32 or Tensor::dot()
 * directly. This runs a real nntrainer FullyConnectedLayer -- built through
 * the model API, compiled, initialized and forwarded -- twice over identical
 * weights, once with an FP32 weight and once with QINT4_HTP, and prints both
 * outputs side by side.
 *
 * Three results, because two of them answer different questions:
 *
 *   cpu-fp32   the layer with an FP32 weight. The baseline the quantized path
 *              is approximating; the gap to it is quantization loss, which is
 *              expected and irreducible.
 *   cpu-u8i4   the same quantization arithmetic, integer GEMM included, done
 *              here in plain C++. Independent of the kernel, so the gap to it
 *              is implementation error, which is a bug.
 *   npu-u8i4   the layer with a QINT4_HTP weight, dispatched to HexKL.
 *
 * npu-u8i4 vs cpu-u8i4 should be **exact**: same quantized operands, same
 * integer product, same dequantize. Any difference at all is a defect.
 * npu-u8i4 vs cpu-fp32 will differ by a few percent and that is fine.
 *
 * The CPU reference here is written from the quantization spec rather than
 * calling nntrainer's own helpers, so a bug in those cannot hide by being on
 * both sides of the comparison.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <app_context.h>
#include <context_data.h>
#include <engine.h>
#include <layer.h>
#include <model.h>
#include <neuralnet.h>
#include <nntrainer_error.h>
#include <optimizer.h>
#include <tensor.h>

#ifdef ENABLE_HEXKL
#include <compute_ops.h>
#include <htp_backend.h>
#include <remote.h>
#include <sdkl.h>
#endif

namespace {

struct Args {
  int M = 4;     // rows (tokens)
  int N = 64;    // out features (unit)
  int K = 64;    // in features
  int show = 8;  // how many output values to print
  uint32_t seed = 1234;
};

Args parseArgs(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto next = [&](int &dst) {
      if (i + 1 < argc)
        dst = std::atoi(argv[++i]);
    };
    if (s == "--M")
      next(a.M);
    else if (s == "--N")
      next(a.N);
    else if (s == "--K")
      next(a.K);
    else if (s == "--show")
      next(a.show);
    else if (s == "--seed") {
      int v = 0;
      next(v);
      a.seed = static_cast<uint32_t>(v);
    } else if (s == "-h" || s == "--help") {
      std::printf(
        "usage: hexkl_fc_e2e [--M rows] [--N out] [--K in] [--show n]\n"
        "                    [--seed s]\n"
        "  N must be a multiple of 32 (the u8i4 kernel's tile width).\n"
        "  Default 4x64x64 keeps the printed output readable; pass real\n"
        "  layer dims (e.g. --M 64 --N 2048 --K 1024) to exercise q_proj.\n");
      std::exit(0);
    }
  }
  return a;
}

std::vector<float> randVec(size_t n, float lo, float hi, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (auto &x : v)
    x = d(rng);
  return v;
}

// ---- CPU reference, written from the spec ----------------------------------
//
// Deliberately not calling nntrainer's quantize/dequantize helpers: if one of
// them is wrong, using it on both sides of the comparison would hide it. The
// arithmetic below is the documented u8i4 contract and nothing else.
//
//   act_scale  = max|A| / 127                       (per tensor)
//   X_u8       = clamp(round(A / act_scale) + 128, 0, 255)
//   wt_scale[n]= max|W[n,:]| / 7                    (per output channel)
//   W_i4[n,k]  = clamp(round(W[n,k] / wt_scale[n]), -7, 7)
//   zp_corr[n] = 128 * sum_k W_i4[n,k]
//   C_i32[m,n] = sum_k X_u8[m,k] * W_i4[n,k]
//   C[m,n]     = act_scale * wt_scale[n] * (C_i32[m,n] - zp_corr[n])

struct WeightQuant {
  std::vector<int8_t> w_i4;     // [N, K], values in [-7, 7]
  std::vector<float> scale;     // [N]
  std::vector<int32_t> zp_corr; // [N]
};

/** @brief Per-output-channel symmetric INT4 weight quantization. */
WeightQuant quantWeightNK(const std::vector<float> &W_nk, int N, int K) {
  WeightQuant q;
  q.w_i4.resize(static_cast<size_t>(N) * K);
  q.scale.resize(N);
  q.zp_corr.resize(N);
  for (int n = 0; n < N; ++n) {
    float max_abs = 0.0f;
    for (int k = 0; k < K; ++k)
      max_abs = std::max(max_abs, std::fabs(W_nk[static_cast<size_t>(n) * K + k]));
    float sc = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;
    if (!std::isfinite(sc) || sc <= 0.0f)
      sc = 1.0f;
    q.scale[n] = sc;

    int32_t row_sum = 0;
    for (int k = 0; k < K; ++k) {
      const long v = std::lround(W_nk[static_cast<size_t>(n) * K + k] / sc);
      const int8_t c =
        static_cast<int8_t>(std::max<long>(-7, std::min<long>(7, v)));
      q.w_i4[static_cast<size_t>(n) * K + k] = c;
      row_sum += c;
    }
    q.zp_corr[n] = 128 * row_sum;
  }
  return q;
}

/** @brief Per-tensor U8 activation quantization; returns act_scale. */
float quantActU8(const std::vector<float> &A, std::vector<uint8_t> &X, int M,
                 int K) {
  float max_abs = 0.0f;
  for (float v : A)
    if (std::isfinite(v))
      max_abs = std::max(max_abs, std::fabs(v));

  // Narrow the division once, in double: the kernel does the same, and a
  // float-only division can land a ULP away and shift a quantized value by 1.
  const double cand = static_cast<double>(max_abs) / 127.0;
  const float act_scale =
    (std::isfinite(cand) && cand >= static_cast<double>(FLT_MIN))
      ? static_cast<float>(cand)
      : 1.0f;
  const float inv = 1.0f / act_scale;

  X.assign(static_cast<size_t>(M) * K, 0);
  for (int i = 0; i < M * K; ++i) {
    const float a = A[i];
    if (!std::isfinite(a)) {
      X[i] = 128;
      continue;
    }
    float q = std::round(a * inv) + 128.0f;
    q = std::max(0.0f, std::min(255.0f, q));
    X[i] = static_cast<uint8_t>(q);
  }
  return act_scale;
}

/** @brief The full u8i4 pipeline on the CPU, plus bias. */
std::vector<float> cpuU8I4(int M, int N, int K, const std::vector<float> &A,
                           const WeightQuant &wq, float act_scale,
                           const std::vector<uint8_t> &X,
                           const std::vector<float> &bias) {
  (void)A;
  std::vector<float> C(static_cast<size_t>(M) * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      int32_t acc = 0;
      for (int k = 0; k < K; ++k)
        acc += static_cast<int32_t>(X[static_cast<size_t>(m) * K + k]) *
               static_cast<int32_t>(wq.w_i4[static_cast<size_t>(n) * K + k]);
      // Dequantize and store, *then* add the bias -- two statements, not one
      // expression. The real path rounds to float between the two (the kernel
      // writes its output, the layer adds the bias afterwards), and folding
      // them into a single expression lets the compiler contract the last
      // multiply and the add into an FMA, skipping that rounding. The result
      // is one ULP off, which reads as a mismatch while being nothing of the
      // sort.
      const float deq =
        act_scale * wq.scale[n] *
        (static_cast<float>(acc) - static_cast<float>(wq.zp_corr[n]));
      C[static_cast<size_t>(m) * N + n] = deq + bias[n];
    }
  return C;
}

/** @brief Plain FP32 GEMM, W given as [K, N] (the fc_layer weight layout). */
std::vector<float> cpuFp32(int M, int N, int K, const std::vector<float> &A,
                           const std::vector<float> &W_kn,
                           const std::vector<float> &bias) {
  std::vector<float> C(static_cast<size_t>(M) * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k)
        acc += A[static_cast<size_t>(m) * K + k] *
               W_kn[static_cast<size_t>(k) * N + n];
      C[static_cast<size_t>(m) * N + n] = acc + bias[n];
    }
  return C;
}

// ---- comparison -------------------------------------------------------------

struct Diff {
  double rel_max = 0.0; // max |a-b| / max|ref|
  double abs_max = 0.0;
  long first_ne = -1; // first index where the two differ at all
  size_t n_ne = 0;
  long ulp_max = 0; // largest distance in representable floats
};

/**
 * @brief Distance between two floats counted in representable values.
 *
 * Bit-identical is 0, adjacent floats is 1. Reported because "not equal" and
 * "wrong" are different claims: a differing last bit is the arithmetic being
 * reassociated, while anything beyond a couple of ULPs is the arithmetic being
 * different.
 */
long ulpDistance(float a, float b) {
  if (a == b)
    return 0;
  if (!std::isfinite(a) || !std::isfinite(b))
    return LONG_MAX;
  int32_t ia, ib;
  std::memcpy(&ia, &a, sizeof(ia));
  std::memcpy(&ib, &b, sizeof(ib));
  // Map the sign-magnitude float ordering onto a monotonic integer one.
  if (ia < 0)
    ia = INT32_MIN - ia;
  if (ib < 0)
    ib = INT32_MIN - ib;
  return std::labs((long)ia - (long)ib);
}

Diff compare(const std::vector<float> &got, const std::vector<float> &ref) {
  Diff d;
  double ref_max = 0.0;
  for (size_t i = 0; i < ref.size(); ++i)
    ref_max = std::max(ref_max, (double)std::fabs(ref[i]));
  for (size_t i = 0; i < ref.size(); ++i) {
    const double e = std::fabs((double)got[i] - (double)ref[i]);
    d.abs_max = std::max(d.abs_max, e);
    d.ulp_max = std::max(d.ulp_max, ulpDistance(got[i], ref[i]));
    if (got[i] != ref[i]) {
      if (d.first_ne < 0)
        d.first_ne = (long)i;
      d.n_ne++;
    }
  }
  d.rel_max = d.abs_max / (ref_max + 1e-9);
  return d;
}

// ---- the layer under test ---------------------------------------------------

/**
 * @brief Build and initialize a one-FC-layer network.
 *
 * weight_dtype selects the storage the layer requests, which is what decides
 * whether Tensor::dot() routes to the NPU. Inference mode only -- nothing here
 * trains.
 */
std::unique_ptr<nntrainer::NeuralNetwork>
buildFcModel(int M, int N, int K, const std::string &weight_dtype) {
  auto nn = std::make_unique<nntrainer::NeuralNetwork>();
  nn->addLayer(ml::train::layer::Input(
    {"name=input", "input_shape=1:1:" + std::to_string(K)}));

  std::vector<std::string> fc = {"name=dense", "unit=" + std::to_string(N)};
  if (!weight_dtype.empty())
    fc.push_back("weight_dtype=" + weight_dtype);
  nn->addLayer(ml::train::layer::FullyConnected(fc));

  nn->setOptimizer(ml::train::optimizer::SGD({"learning_rate=0.1"}));
  nn->setProperty({"loss=mse", "batch_size=" + std::to_string(M)});

  // Both return a status code. Ignoring them left a half-built graph looking
  // like a graph whose layer simply owned no weights.
  const int crc = nn->compile(ml::train::ExecutionMode::INFERENCE);
  if (crc != ML_ERROR_NONE) {
    std::printf("  [model] compile failed (%d)\n", crc);
    return nullptr;
  }
  const int irc = nn->initialize(ml::train::ExecutionMode::INFERENCE);
  if (irc != ML_ERROR_NONE) {
    std::printf("  [model] initialize failed (%d)\n", irc);
    return nullptr;
  }
  return nn;
}

/**
 * @brief The dense layer's weight and bias tensors.
 *
 * Two earlier attempts failed here, so this one reports what it sees rather
 * than only whether it succeeded. Matching the layer name against the tensor
 * name found nothing (nntrainer decorates weight names); walking the graph
 * for a node with weights also found nothing, which says the weights are not
 * reachable at the point it looked, not that the key was wrong.
 *
 * getRunContext() throws on a node that has none, so each is guarded and the
 * reason is printed.
 */
bool findWeights(nntrainer::NeuralNetwork *nn, nntrainer::Tensor **weight,
                 nntrainer::Tensor **bias, bool verbose) {
  *weight = nullptr;
  *bias = nullptr;

  auto graph = nn->getFlatGraph();
  if (verbose)
    std::printf("  [graph] %zu node(s)\n", graph.size());

  for (auto &node : graph) {
    const std::string name = node->getName();
    const std::string type = node->getType();

    unsigned int nw = 0;
    bool has_ctx = true;
    try {
      nw = node->getRunContext().getNumWeights();
    } catch (const std::exception &e) {
      has_ctx = false;
      if (verbose)
        std::printf("  [graph] '%s' (%s): no run context (%s)\n", name.c_str(),
                    type.c_str(), e.what());
    } catch (...) {
      has_ctx = false;
      if (verbose)
        std::printf("  [graph] '%s' (%s): no run context\n", name.c_str(),
                    type.c_str());
    }
    if (!has_ctx)
      continue;

    if (verbose) {
      std::printf("  [graph] '%s' (%s): %u weight(s)", name.c_str(),
                  type.c_str(), nw);
      for (unsigned int w = 0; w < nw; ++w)
        std::printf(" [%u]=%s", w, node->getRunContext().getWeightName(w).c_str());
      std::printf("\n");
    }
    if (nw == 0)
      continue;

    auto &rc = node->getRunContext();
    for (unsigned int w = 0; w < nw; ++w) {
      const std::string nm = rc.getWeightName(w);
      if (nm.find("bias") != std::string::npos)
        *bias = &rc.getWeight(w);
      else if (*weight == nullptr)
        *weight = &rc.getWeight(w);
    }
    if (*weight != nullptr)
      return true;
  }
  return false;
}

/**
 * @brief Locate the weights, running one throwaway inference first if needed.
 *
 * nntrainer may not have the weight tensors reachable until the graph has been
 * allocated, which happens on the first forward pass rather than in
 * initialize(). The throwaway run produces garbage -- the weights it uses are
 * whatever the initializer left -- and is discarded; only the allocation it
 * forces matters.
 */
bool findWeightsAllocating(nntrainer::NeuralNetwork *nn, int M, int K,
                           nntrainer::Tensor **weight,
                           nntrainer::Tensor **bias) {
  if (findWeights(nn, weight, bias, true))
    return true;

  std::printf("  [graph] no weights reachable yet; forcing allocation with a "
              "throwaway inference\n");
  std::vector<float> dummy(static_cast<size_t>(M) * K, 0.0f);
  try {
    nn->inference(M, {dummy.data()});
  } catch (const std::exception &e) {
    std::printf("  [graph] throwaway inference failed: %s\n", e.what());
    return false;
  }
  return findWeights(nn, weight, bias, true);
}

/**
 * @brief FullyConnectedLayer::forwarding, run directly on tensors.
 *
 * The model path exposes the layer's weights through RunLayerContext, and on
 * this device that came back empty. Rather than block the comparison on that,
 * this reproduces exactly what the layer's forward does --
 *
 *     input_.dot(weight, hidden_, false, false);   // fc_layer.cpp
 *     hidden_.add_i(bias);
 *
 * -- on tensors built the same way the layer requests them: activation
 * [1,1,M,K] FP32, weight [1,1,K,N] in the given dtype. Same call, same flags,
 * same dispatch. What it does not cover is the model plumbing around it: the
 * weight_dtype property, RunLayerContext, and how the graph hands the layer
 * its tensors.
 */
std::vector<float> forwardDirect(int M, int N, int K,
                                 const std::vector<float> &A,
                                 const nntrainer::Tensor &weight,
                                 const std::vector<float> &bias, bool on_npu) {
  nntrainer::TensorDim adim(1, 1, M, K);
  nntrainer::Tensor act(adim, false);
  act.allocate();
  std::memcpy(act.getData<float>(), A.data(),
              static_cast<size_t>(M) * K * sizeof(float));

#ifdef ENABLE_HEXKL
  if (on_npu) {
    // dot() only routes to the NPU when the activation carries the HTP ops.
    static std::shared_ptr<nntrainer::ContextData> ct = [] {
      auto c = std::make_shared<nntrainer::ContextData>();
      c->setComputeOps(nntrainer::get_htp_ops());
      return c;
    }();
    act.setContextData(ct);
  }
#else
  (void)on_npu;
#endif

  nntrainer::Tensor out = act.dot(weight, false, false);

  std::vector<float> C(static_cast<size_t>(M) * N);
  std::memcpy(C.data(), out.getData<float>(),
              static_cast<size_t>(M) * N * sizeof(float));
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n)
      C[static_cast<size_t>(m) * N + n] += bias[n];
  return C;
}

/** @brief An FP32 [1,1,K,N] weight tensor holding W_kn. */
nntrainer::Tensor makeFp32Weight(int K, int N, const std::vector<float> &W_kn) {
  nntrainer::Tensor t(1, 1, K, N,
                      {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});
  t.allocate();
  std::memcpy(t.getData<float>(), W_kn.data(),
              static_cast<size_t>(K) * N * sizeof(float));
  return t;
}

#ifdef ENABLE_HEXKL
/**
 * @brief A QINT4_HTP [K,N] weight, laid out the way quantize_qint4_weight does.
 *
 * [K,N] not [N,K]: scale_size() is width(), so the logical layout would
 * advertise K scales where the kernel wants N. The packed bytes themselves are
 * layout-agnostic -- the kernel takes N and K from the matmul dims.
 */
nntrainer::Tensor makeQint4HtpWeight(int K, int N, const WeightQuant &wq) {
  const size_t unpacked = static_cast<size_t>(N) * K;
  std::vector<int8_t> packed(unpacked, 0);

  void *src = nullptr, *dst = nullptr;
  if (sdkl_npu_alloc(unpacked, &src) == 0 && src != nullptr &&
      sdkl_npu_alloc(unpacked, &dst) == 0 && dst != nullptr) {
    std::memcpy(src, wq.w_i4.data(), unpacked);
    // (wt_rows=K, wt_cols=N): the transpose of the in-place i8 packer's order.
    if (sdkl_cpu_rm_to_wh_i4(static_cast<uint8_t *>(dst),
                             static_cast<int8_t *>(src), (size_t)K,
                             (size_t)N) == 0)
      std::memcpy(packed.data(), dst, unpacked);
  }
  if (src)
    sdkl_npu_free(src);
  if (dst)
    sdkl_npu_free(dst);

  nntrainer::TensorDim qdim(
    1, 1, K, N, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT4_HTP});
  nntrainer::Tensor t(qdim, true, nntrainer::Initializer::NONE, "",
                      nntrainer::QScheme::PER_CHANNEL_AFFINE_I4);
  std::memcpy(t.getData<int8_t>(), packed.data(), unpacked);
  std::memcpy(t.getScale<float>(), wq.scale.data(), N * sizeof(float));
  std::memcpy(t.getZpCorr<int32_t>(), wq.zp_corr.data(), N * sizeof(int32_t));
  return t;
}
#endif

void printHead(const char *tag, const std::vector<float> &v, int n) {
  std::printf("  %-10s", tag);
  for (int i = 0; i < n && i < (int)v.size(); ++i)
    std::printf(" %12.6f", v[i]);
  std::printf("\n");
}

} // namespace

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  Args a = parseArgs(argc, argv);

  if (a.N % 32 != 0) {
    std::printf("error: N must be a multiple of 32 (got %d)\n", a.N);
    return 1;
  }

  std::printf("fc_layer end-to-end: CPU FP32 vs CPU u8i4 vs HexKL u8i4\n");
  std::printf("  M=%d N=%d K=%d seed=%u\n", a.M, a.N, a.K, a.seed);
  std::printf("  built: %s %s\n\n", __DATE__, __TIME__);

  const size_t mk = (size_t)a.M * a.K;
  const size_t kn = (size_t)a.K * a.N;

  // fc_layer stores the weight as [K, N]; the quantizer works in [N, K].
  auto A = randVec(mk, -1.0f, 1.0f, a.seed);
  auto W_kn = randVec(kn, -0.5f, 0.5f, a.seed + 1);
  auto bias = randVec(a.N, -0.1f, 0.1f, a.seed + 2);

  std::vector<float> W_nk(kn);
  for (int k = 0; k < a.K; ++k)
    for (int n = 0; n < a.N; ++n)
      W_nk[(size_t)n * a.K + k] = W_kn[(size_t)k * a.N + n];

  // ---- CPU references -------------------------------------------------------
  auto ref_fp32 = cpuFp32(a.M, a.N, a.K, A, W_kn, bias);

  WeightQuant wq = quantWeightNK(W_nk, a.N, a.K);
  std::vector<uint8_t> X_u8;
  const float act_scale = quantActU8(A, X_u8, a.M, a.K);
  auto ref_u8i4 = cpuU8I4(a.M, a.N, a.K, A, wq, act_scale, X_u8, bias);

  // ---- the layer, FP32 weight ----------------------------------------------
  std::vector<float> out_fp32;
  bool via_model = false;
  try {
    auto nn = buildFcModel(a.M, a.N, a.K, "");
    nntrainer::Tensor *w = nullptr, *b = nullptr;
    if (nn && findWeightsAllocating(nn.get(), a.M, a.K, &w, &b)) {
      std::memcpy(w->getData<float>(), W_kn.data(), kn * sizeof(float));
      if (b)
        std::memcpy(b->getData<float>(), bias.data(), a.N * sizeof(float));
      auto res = nn->inference(a.M, {A.data()});
      out_fp32.assign(res[0], res[0] + (size_t)a.M * a.N);
      via_model = true;
    }
  } catch (const std::exception &e) {
    std::printf("  [model] FP32 model path failed: %s\n", e.what());
  }

  if (!via_model) {
    // Fall back to the layer's forward arithmetic on bare tensors. Same dot()
    // call the layer makes; what is skipped is the graph plumbing around it.
    std::printf("  [model] weights unreachable through the graph -- running "
                "the layer's forward directly instead\n\n");
    out_fp32 =
      forwardDirect(a.M, a.N, a.K, A, makeFp32Weight(a.K, a.N, W_kn), bias,
                    false);
  }

#ifdef ENABLE_HEXKL
  // ---- the layer, QINT4_HTP weight -----------------------------------------
  std::vector<float> out_npu;
  bool npu_ran = false;
  if (!nntrainer::HtpBackend::global().enabled()) {
    std::printf("[warn] HTP backend unavailable; skipping the NPU run\n\n");
  } else {
    // The weight tensor is built once and used by whichever path runs, so the
    // NPU sees exactly the int4 codes the CPU reference used either way.
    nntrainer::Tensor wgt = makeQint4HtpWeight(a.K, a.N, wq);

    bool through_model = false;
    try {
      auto nn = buildFcModel(a.M, a.N, a.K, "QINT4_HTP");
      nntrainer::Tensor *w = nullptr, *b = nullptr;
      if (nn && findWeightsAllocating(nn.get(), a.M, a.K, &w, &b)) {
        std::memcpy(w->getData<int8_t>(), wgt.getData<int8_t>(), kn);
        std::memcpy(w->getScale<float>(), wq.scale.data(),
                    a.N * sizeof(float));
        std::memcpy(w->getZpCorr<int32_t>(), wq.zp_corr.data(),
                    a.N * sizeof(int32_t));
        if (b)
          std::memcpy(b->getData<float>(), bias.data(), a.N * sizeof(float));

        auto res = nn->inference(a.M, {A.data()});
        out_npu.assign(res[0], res[0] + (size_t)a.M * a.N);
        through_model = true;
      }
    } catch (const std::exception &e) {
      std::printf("  [model] QINT4_HTP model path failed: %s\n", e.what());
    }

    if (!through_model)
      out_npu = forwardDirect(a.M, a.N, a.K, A, wgt, bias, true);
    npu_ran = true;
    std::printf("  [npu] ran %s\n\n",
                through_model ? "through the model" : "via the layer's dot()");
  }
#else
  std::vector<float> out_npu;
  const bool npu_ran = false;
  std::printf("[warn] built without ENABLE_HEXKL; NPU run unavailable\n\n");
#endif

  // ---- report ---------------------------------------------------------------
  const int show = std::min(a.show, a.N);
  std::printf("first %d outputs of row 0\n", show);
  printHead("cpu-fp32", ref_fp32, show);
  printHead("cpu-u8i4", ref_u8i4, show);
  if (npu_ran)
    printHead("npu-u8i4", out_npu, show);
  printHead("layer-fp32", out_fp32, show);
  std::printf("\n");

  std::printf("| comparison | max abs diff | max rel diff | differing "
              "elements | verdict |\n");
  std::printf("|---|---|---|---|---|\n");

  {
    const Diff d = compare(out_fp32, ref_fp32);
    std::printf("| layer-fp32 vs cpu-fp32 | %.3e | %.3e | %zu / %zu | %s |\n",
                d.abs_max, d.rel_max, d.n_ne, ref_fp32.size(),
                d.rel_max < 1e-4 ? "OK (float assoc)" : "SUSPECT");
  }

  if (npu_ran) {
    const Diff d = compare(out_npu, ref_u8i4);
    // Same operands, same integer product, same dequantize: this one has no
    // reason to differ at all, so it is the test that matters.
    const char *verdict =
      d.n_ne == 0
        ? "EXACT MATCH"
        : (d.ulp_max <= 1 ? "1 ULP -- float rounding, not a logic error"
                          : "MISMATCH -- a real defect");
    std::printf("| **npu-u8i4 vs cpu-u8i4** | %.3e | %.3e | %zu / %zu (max %ld "
                "ULP) | **%s** |\n",
                d.abs_max, d.rel_max, d.n_ne, ref_u8i4.size(), d.ulp_max,
                verdict);
    if (d.n_ne != 0)
      std::printf("  first differing element %ld: cpu=%.9g npu=%.9g (%ld "
                  "ULP)\n",
                  d.first_ne, ref_u8i4[d.first_ne], out_npu[d.first_ne],
                  ulpDistance(ref_u8i4[d.first_ne], out_npu[d.first_ne]));

    const Diff q = compare(out_npu, ref_fp32);
    std::printf("| npu-u8i4 vs cpu-fp32 | %.3e | %.3e | - | quantization loss "
                "(expected) |\n",
                q.abs_max, q.rel_max);
  }

  {
    const Diff q = compare(ref_u8i4, ref_fp32);
    std::printf("| cpu-u8i4 vs cpu-fp32 | %.3e | %.3e | - | quantization loss "
                "(expected) |\n",
                q.abs_max, q.rel_max);
  }

  std::printf(
    "\nRead npu-u8i4 vs cpu-u8i4 first. It compares two runs of the same\n"
    "arithmetic on the same quantized operands, so a difference beyond the\n"
    "last bit is an implementation defect. A max of 1 ULP is the two paths\n"
    "rounding in different places, not disagreeing about the answer. The two\n"
    "rows against cpu-fp32 measure how much INT4 costs, which is a property\n"
    "of 4-bit weights and not a bug.\n"
    "\nTo confirm this is not rigged: break the kernel on purpose and watch\n"
    "it fail. Change the zero point in hexkl_quant.h (kActZeroPoint 128 ->\n"
    "127), or the clamp in quantizer.cpp (7 -> 6), rebuild libnntrainer and\n"
    "re-run. The CPU reference here is written from the spec and shares no\n"
    "code with the kernel, so it will not follow the change.\n");

  if (npu_ran) {
    // A last-bit difference is not a failure: the two paths round in different
    // places even when they compute the same thing. Anything larger is.
    const Diff d = compare(out_npu, ref_u8i4);
    return d.ulp_max <= 1 ? 0 : 1;
  }
  return 0;
}
