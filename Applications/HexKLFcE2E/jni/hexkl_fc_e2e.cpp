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
      C[static_cast<size_t>(m) * N + n] =
        act_scale * wq.scale[n] *
          (static_cast<float>(acc) - static_cast<float>(wq.zp_corr[n])) +
        bias[n];
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
};

Diff compare(const std::vector<float> &got, const std::vector<float> &ref) {
  Diff d;
  double ref_max = 0.0;
  for (size_t i = 0; i < ref.size(); ++i)
    ref_max = std::max(ref_max, (double)std::fabs(ref[i]));
  for (size_t i = 0; i < ref.size(); ++i) {
    const double e = std::fabs((double)got[i] - (double)ref[i]);
    d.abs_max = std::max(d.abs_max, e);
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
  nn->compile(ml::train::ExecutionMode::INFERENCE);
  nn->initialize(ml::train::ExecutionMode::INFERENCE);
  return nn;
}

/**
 * @brief The dense layer's weight and bias tensors.
 *
 * Located by walking the graph for a node that owns weights, rather than by
 * matching the layer name against the tensor name: nntrainer decorates weight
 * names, and a substring match on "dense" silently found nothing. fc_layer
 * requests "weight" first and "bias" second, so index order is the reliable
 * key, with the recorded names used only to confirm it.
 *
 * getRunContext() throws on a node that has none, so each is guarded.
 */
bool findWeights(nntrainer::NeuralNetwork *nn, nntrainer::Tensor **weight,
                 nntrainer::Tensor **bias, bool verbose) {
  *weight = nullptr;
  *bias = nullptr;
  for (auto &node : nn->getFlatGraph()) {
    unsigned int nw = 0;
    try {
      nw = node->getRunContext().getNumWeights();
    } catch (...) {
      continue; // no run context: input layers and the like
    }
    if (nw == 0)
      continue;

    auto &rc = node->getRunContext();
    if (verbose) {
      std::printf("  [graph] node '%s' (%s) has %u weight(s):", 
                  node->getName().c_str(), node->getType().c_str(), nw);
      for (unsigned int w = 0; w < nw; ++w)
        std::printf(" %s", rc.getWeightName(w).c_str());
      std::printf("\n");
    }

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
  try {
    auto nn = buildFcModel(a.M, a.N, a.K, "");
    nntrainer::Tensor *w = nullptr, *b = nullptr;
    if (!findWeights(nn.get(), &w, &b, true)) {
      std::printf("error: could not locate the dense weight\n");
      return 1;
    }
    std::memcpy(w->getData<float>(), W_kn.data(), kn * sizeof(float));
    if (b)
      std::memcpy(b->getData<float>(), bias.data(), a.N * sizeof(float));

    auto res = nn->inference(a.M, {A.data()});
    out_fp32.assign(res[0], res[0] + (size_t)a.M * a.N);
  } catch (const std::exception &e) {
    std::printf("error: FP32 fc_layer failed: %s\n", e.what());
    return 1;
  }

#ifdef ENABLE_HEXKL
  // ---- the layer, QINT4_HTP weight -----------------------------------------
  std::vector<float> out_npu;
  bool npu_ran = false;
  if (!nntrainer::HtpBackend::global().enabled()) {
    std::printf("[warn] HTP backend unavailable; skipping the NPU run\n\n");
  } else
    try {
      auto nn = buildFcModel(a.M, a.N, a.K, "QINT4_HTP");
      nntrainer::Tensor *w = nullptr, *b = nullptr;
      if (!findWeights(nn.get(), &w, &b, true)) {
        std::printf("error: could not locate the dense QINT4_HTP weight\n");
        return 1;
      }

      // WH-pack the same int4 weights the CPU reference used, so the two runs
      // differ only in who does the arithmetic. sdkl_cpu_rm_to_wh_i4 takes
      // (wt_rows=K, wt_cols=N) -- the transpose of the in-place i8 variant.
      void *src = nullptr, *dst = nullptr;
      if (sdkl_npu_alloc(kn, &src) != 0 || src == nullptr ||
          sdkl_npu_alloc(kn, &dst) != 0 || dst == nullptr) {
        std::printf("error: sdkl_npu_alloc failed for the WH pack\n");
        return 1;
      }
      std::memcpy(src, wq.w_i4.data(), kn);
      const int rc =
        sdkl_cpu_rm_to_wh_i4(static_cast<uint8_t *>(dst),
                             static_cast<int8_t *>(src), (size_t)a.K,
                             (size_t)a.N);
      if (rc != 0) {
        std::printf("error: sdkl_cpu_rm_to_wh_i4 failed (%d)\n", rc);
        return 1;
      }
      std::memcpy(w->getData<int8_t>(), dst, kn);
      sdkl_npu_free(src);
      sdkl_npu_free(dst);

      std::memcpy(w->getScale<float>(), wq.scale.data(), a.N * sizeof(float));
      std::memcpy(w->getZpCorr<int32_t>(), wq.zp_corr.data(),
                  a.N * sizeof(int32_t));
      if (b)
        std::memcpy(b->getData<float>(), bias.data(), a.N * sizeof(float));

      auto res = nn->inference(a.M, {A.data()});
      out_npu.assign(res[0], res[0] + (size_t)a.M * a.N);
      npu_ran = true;
    } catch (const std::exception &e) {
      std::printf("error: QINT4_HTP fc_layer failed: %s\n", e.what());
      return 1;
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
    std::printf("| **npu-u8i4 vs cpu-u8i4** | %.3e | %.3e | %zu / %zu | **%s** "
                "|\n",
                d.abs_max, d.rel_max, d.n_ne, ref_u8i4.size(),
                d.n_ne == 0 ? "EXACT MATCH" : "MISMATCH -- a real defect");
    if (d.n_ne != 0)
      std::printf("  first differing element %ld: cpu=%.9g npu=%.9g\n",
                  d.first_ne, ref_u8i4[d.first_ne], out_npu[d.first_ne]);

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

  std::printf("\nRead npu-u8i4 vs cpu-u8i4 first. It compares two runs of the\n"
              "same arithmetic on the same quantized operands, so anything\n"
              "other than an exact match is an implementation defect. The two\n"
              "rows against cpu-fp32 measure how much INT4 costs, which is a\n"
              "property of 4-bit weights and not a bug.\n");

  if (npu_ran) {
    const Diff d = compare(out_npu, ref_u8i4);
    return d.n_ne == 0 ? 0 : 1;
  }
  return 0;
}
