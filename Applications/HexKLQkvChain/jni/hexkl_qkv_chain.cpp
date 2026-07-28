// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_qkv_chain.cpp
 * @date   28 Jul 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Q/K/V projections run back to back through HexKL, as a model.
 *
 * Everything measured so far has been one fc_layer in isolation. An attention
 * block issues three of them against the same activation, so the question this
 * answers is what three back-to-back calls cost against one -- which is the
 * whole argument for a fused QKV weight.
 *
 * Shapes are qwen3-0.6b (hidden 1024, 16 q heads / 8 kv heads, head_dim 128),
 * so q projects to 2048 and k/v to 1024 each. Weights are random; nothing here
 * is a correctness check, and the numbers are timings only.
 *
 * Two paths, both reported:
 *
 *   model    a real nntrainer graph -- Input feeding three FullyConnected
 *            layers -- saved with QINT4_HTP weights, loaded back, and run
 *            through NeuralNetwork::inference(). This is the honest end to
 *            end: layer plumbing, weight_dtype, RunLayerContext and all.
 *   direct   the same three projections as three Tensor::dot() calls on
 *            QINT4_HTP weights baked in-process. This is the path the kernel
 *            tests cover, so it works whether or not the graph does.
 *
 * The model path is attempted first and the failure, if any, is printed rather
 * than swallowed -- an earlier attempt at it failed with a message that blamed
 * the tile width when the real cause was the layer not carrying `engine=htp`,
 * and silence there cost a lot of time. `direct` always runs, so there is
 * always a measurement.
 *
 * The fused comparison at the end runs a single N=4096 matmul, which is the
 * same arithmetic as q+k+v (2048+1024+1024) in one call. The difference
 * between that and the three-call total is what fusing QKV would buy, and it
 * is the number this app exists to produce.
 */

#include <algorithm>
#include <chrono>
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
#include <nntrainer_error.h>
#include <optimizer.h>
#include <quantizer.h>
#include <tensor.h>

#ifdef ENABLE_HEXKL
#include <compute_ops.h>
#include <hexkl_mm.h>
#include <htp_backend.h>
#include <sdkl_compat.h>
#endif

namespace {

// ---- Configuration ---------------------------------------------------------

struct Args {
  int M = 64;      // tokens: 64 = prefill tile, 1 = decode
  int K = 1024;    // hidden size
  int q_N = 2048;  // 16 heads * 128 head_dim
  int kv_N = 1024; //  8 kv heads * 128 head_dim
  int iters = 100;
  int warmup = 10;
  uint32_t seed = 1234;
  bool run_model = true;
  bool run_fused = true;
  std::string model_path = "/data/local/tmp/hexkl_qkv.bin";
};

void usage() {
  std::printf(
    "usage: hexkl_qkv_chain [options]\n"
    "  --M <n>        tokens (default 64; use 1 for decode)\n"
    "  --K <n>        hidden size (default 1024)\n"
    "  --q-N <n>      q out features (default 2048)\n"
    "  --kv-N <n>     k/v out features (default 1024)\n"
    "  --iters <n>    timed iterations (default 100)\n"
    "  --warmup <n>   untimed iterations first (default 10)\n"
    "  --seed <n>     RNG seed for the random weights (default 1234)\n"
    "  --model-path <p>  where to write the baked model\n"
    "                    (default /data/local/tmp/hexkl_qkv.bin)\n"
    "  --no-model     skip the graph path, run `direct` only\n"
    "  --no-fused     skip the single N=q_N+2*kv_N comparison\n"
    "\n"
    "  defaults are qwen3-0.6b: hidden 1024, q 2048, k/v 1024 each\n");
}

// ---- Timing ----------------------------------------------------------------

double nowUs() {
  return std::chrono::duration<double, std::micro>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

/**
 * @brief Median rather than mean.
 *
 * A handful of calls on this device land in the milliseconds -- scheduler or
 * RPC, not the kernel. The mean follows those and the median does not, and
 * what is being compared here is the per-call cost, not the tail. The max is
 * printed alongside so a bad tail is still visible instead of hidden.
 */
struct Stat {
  double median = 0.0, mean = 0.0, max = 0.0;
};

Stat summarize(std::vector<double> v) {
  Stat s;
  if (v.empty())
    return s;
  std::sort(v.begin(), v.end());
  s.median = v[v.size() / 2];
  s.max = v.back();
  double sum = 0.0;
  for (double x : v)
    sum += x;
  s.mean = sum / static_cast<double>(v.size());
  return s;
}

// ---- Random weights --------------------------------------------------------

/**
 * @brief Random FP32 weights in [-0.5, 0.5], in fc_layer's [K, N] layout.
 *
 * The range is arbitrary but not meaningless: INT4 quantization is per output
 * channel and symmetric, so only the ratio of each channel's values to that
 * channel's max matters, not the absolute scale. A uniform spread fills the
 * [-7, 7] codebook reasonably evenly, which keeps the timing representative --
 * a degenerate weight would not change the arithmetic cost, but it would make
 * the printed output useless for sanity-checking that anything ran.
 */
std::vector<float> randomWeightKN(int K, int N, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-0.5f, 0.5f);
  std::vector<float> w(static_cast<size_t>(K) * N);
  for (auto &x : w)
    x = d(rng);
  return w;
}

std::vector<float> randomActs(int M, int K, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  std::vector<float> a(static_cast<size_t>(M) * K);
  for (auto &x : a)
    x = d(rng);
  return a;
}

/** @brief An FP32 [1, 1, K, N] weight tensor, the layout fc_layer stores. */
nntrainer::Tensor makeFp32WeightKN(int K, int N, const std::vector<float> &w) {
  nntrainer::Tensor t(1, 1, K, N,
                      {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});
  t.allocate();
  std::memcpy(t.getData<float>(), w.data(),
              static_cast<size_t>(K) * N * sizeof(float));
  return t;
}

#ifdef ENABLE_HEXKL
/**
 * @brief The HTP compute context, as a tensor's context data.
 *
 * dot() picks its backend from the tensor's ContextData and falls back to the
 * process default when there is none, so an activation without this silently
 * runs on the CPU -- or, for QINT4_HTP, throws. In the graph path the
 * equivalent comes from the layer's `engine=htp` property.
 */
std::shared_ptr<nntrainer::ContextData> htpContextData() {
  static std::shared_ptr<nntrainer::ContextData> ct = [] {
    auto c = std::make_shared<nntrainer::ContextData>();
    c->setComputeOps(nntrainer::get_htp_ops());
    return c;
  }();
  return ct;
}
#endif

// ---- One projection --------------------------------------------------------

/**
 * @brief One QINT4_HTP projection, held across iterations.
 *
 * The baked weight is built once and reused, which is deliberate: baking is a
 * load-time cost a model pays once, and including it per iteration would
 * measure the wrong thing. It also lets the kernel's resident-weight cache
 * behave the way it does in a real model, where the same weight pointer comes
 * back every token.
 */
struct Projection {
  const char *name;
  int N;
  nntrainer::Tensor weight; // QINT4_HTP [K, N]
  Stat stat;
  // Phase breakdown from the last timed call, in microseconds.
  double scan = 0, quant = 0, stage = 0, npu = 0, dequant = 0;
};

Projection bakeProjection(const char *name, int K, int N, uint32_t seed) {
  Projection p;
  p.name = name;
  p.N = N;
  auto w_kn = randomWeightKN(K, N, seed);
  nntrainer::Tensor fp32 = makeFp32WeightKN(K, N, w_kn);
  // transpose_input=true: the source is [K, N] and the quantizer wants the
  // logical [N, K] to take per-output-channel scales over. Same call
  // Layer::save makes.
  p.weight = nntrainer::quantize_qint4_weight(fp32, true);
  return p;
}

// ---- direct: three dot() calls --------------------------------------------

/**
 * @brief Time each projection over `iters`, all three against one activation.
 *
 * Q, K and V are parallel in the dataflow -- same input, three weights -- but
 * they issue back to back, so what is timed here is one iteration of all three
 * in sequence, per-call and totalled.
 */
void runDirect(const Args &a, std::vector<Projection> &projs,
               const std::vector<float> &A, Stat &total_stat) {
  nntrainer::TensorDim adim(1, 1, a.M, a.K);
  nntrainer::Tensor act(adim, false);
  act.allocate();
  std::memcpy(act.getData<float>(), A.data(),
              static_cast<size_t>(a.M) * a.K * sizeof(float));
#ifdef ENABLE_HEXKL
  act.setContextData(htpContextData());
#endif

  std::vector<std::vector<double>> per(projs.size());
  std::vector<double> totals;

  for (int i = 0; i < a.warmup + a.iters; ++i) {
    const bool timed = i >= a.warmup;
    double round = 0.0;
    for (size_t p = 0; p < projs.size(); ++p) {
      const double t0 = nowUs();
      // fc_layer's own call: input_.dot(weight, hidden_, false, false).
      // trans_in must be false -- the weight is [K, N], so with trans_in=true
      // the N and K roles swap and any shape with N != K is rejected.
      nntrainer::Tensor out = act.dot(projs[p].weight, false, false);
      const double dt = nowUs() - t0;
      round += dt;
      if (timed) {
        per[p].push_back(dt);
#ifdef ENABLE_HEXKL
        const auto &mp = nntrainer::hmx::lastMmProfile();
        projs[p].scan = mp.scan_us;
        projs[p].quant = mp.quant_us;
        projs[p].stage = mp.stage_us;
        projs[p].npu = mp.npu_us;
        projs[p].dequant = mp.dequant_us;
#endif
      }
    }
    if (timed)
      totals.push_back(round);
  }

  for (size_t p = 0; p < projs.size(); ++p)
    projs[p].stat = summarize(per[p]);
  total_stat = summarize(totals);
}

/** @brief One matmul at the combined width, for the fused comparison. */
Stat runFused(const Args &a, int fused_N, const std::vector<float> &A) {
  Projection f = bakeProjection("qkv_fused", a.K, fused_N, a.seed + 99);

  nntrainer::TensorDim adim(1, 1, a.M, a.K);
  nntrainer::Tensor act(adim, false);
  act.allocate();
  std::memcpy(act.getData<float>(), A.data(),
              static_cast<size_t>(a.M) * a.K * sizeof(float));
#ifdef ENABLE_HEXKL
  act.setContextData(htpContextData());
#endif

  std::vector<double> v;
  for (int i = 0; i < a.warmup + a.iters; ++i) {
    const double t0 = nowUs();
    nntrainer::Tensor out = act.dot(f.weight, false, false);
    const double dt = nowUs() - t0;
    if (i >= a.warmup)
      v.push_back(dt);
  }
  return summarize(v);
}

// ---- model: a real graph ---------------------------------------------------

/**
 * @brief Input feeding three FullyConnected layers, all FP32.
 *
 * Q, K and V each read the same input, so all three name it in input_layers
 * rather than chaining. `engine=htp` is what routes the layer's tensors to the
 * HTP compute context; without it the graph hands them the CPU ops and a
 * QINT4_HTP weight has nowhere to dispatch.
 */
std::unique_ptr<nntrainer::NeuralNetwork> buildFp32Model(const Args &a) {
  auto nn = std::make_unique<nntrainer::NeuralNetwork>();

  nn->addLayer(ml::train::layer::Input(
    {"name=input", "input_shape=1:1:" + std::to_string(a.K)}));
  nn->addLayer(ml::train::layer::FullyConnected(
    {"name=q_proj", "input_layers=input", "unit=" + std::to_string(a.q_N),
     "engine=htp"}));
  nn->addLayer(ml::train::layer::FullyConnected(
    {"name=k_proj", "input_layers=input", "unit=" + std::to_string(a.kv_N),
     "engine=htp"}));
  nn->addLayer(ml::train::layer::FullyConnected(
    {"name=v_proj", "input_layers=input", "unit=" + std::to_string(a.kv_N),
     "engine=htp"}));

  nn->setOptimizer(ml::train::optimizer::SGD({"learning_rate=0.1"}));
  nn->setProperty({"loss=mse", "batch_size=" + std::to_string(a.M)});
  nn->compile();
  nn->initialize();
  return nn;
}

/**
 * @brief Build FP32, save with QINT4_HTP, load back, time inference.
 *
 * A model built from scratch cannot simply be handed QINT4_HTP weights -- the
 * scales and zp_corr come from the bake, and a randomly initialized quantized
 * tensor has none. Saving and reloading is the real path a quantized model
 * takes, and it is the only way to get valid weights into the graph.
 *
 * @return true when the whole flow ran; false with a printed reason otherwise.
 */
bool runModel(const Args &a, const std::vector<float> &A, Stat &stat) {
  std::printf("[model] building FP32 graph (input -> q_proj/k_proj/v_proj, "
              "engine=htp)\n");
  std::unique_ptr<nntrainer::NeuralNetwork> fp32;
  try {
    fp32 = buildFp32Model(a);
  } catch (const std::exception &e) {
    std::printf("[model] FAILED to build: %s\n", e.what());
    return false;
  }

  std::printf("[model] saving with QINT4_HTP weights -> %s\n",
              a.model_path.c_str());
  try {
    fp32->save(a.model_path, ml::train::ModelFormat::MODEL_FORMAT_BIN,
               ml::train::TensorDim::DataType::QINT4_HTP);
  } catch (const std::exception &e) {
    std::printf("[model] FAILED to save: %s\n", e.what());
    return false;
  }

  std::printf("[model] rebuilding with model_tensor_type=QINT4_HTP-FP32 and "
              "loading\n");
  std::unique_ptr<nntrainer::NeuralNetwork> q4;
  try {
    q4 = std::make_unique<nntrainer::NeuralNetwork>();
    q4->addLayer(ml::train::layer::Input(
      {"name=input", "input_shape=1:1:" + std::to_string(a.K)}));
    q4->addLayer(ml::train::layer::FullyConnected(
      {"name=q_proj", "input_layers=input", "unit=" + std::to_string(a.q_N),
       "engine=htp"}));
    q4->addLayer(ml::train::layer::FullyConnected(
      {"name=k_proj", "input_layers=input", "unit=" + std::to_string(a.kv_N),
       "engine=htp"}));
    q4->addLayer(ml::train::layer::FullyConnected(
      {"name=v_proj", "input_layers=input", "unit=" + std::to_string(a.kv_N),
       "engine=htp"}));
    q4->setProperty({"loss=mse", "batch_size=" + std::to_string(a.M),
                     "model_tensor_type=QINT4_HTP-FP32"});
    q4->setOptimizer(ml::train::optimizer::SGD({"learning_rate=0.1"}));
    q4->compile();
    q4->initialize();
    q4->load(a.model_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
  } catch (const std::exception &e) {
    std::printf("[model] FAILED to build/load quantized model: %s\n", e.what());
    return false;
  }

  std::vector<float> in(A);
  std::vector<double> v;
  try {
    for (int i = 0; i < a.warmup + a.iters; ++i) {
      const double t0 = nowUs();
      q4->inference(a.M, {in.data()});
      const double dt = nowUs() - t0;
      if (i >= a.warmup)
        v.push_back(dt);
    }
  } catch (const std::exception &e) {
    std::printf("[model] FAILED during inference: %s\n", e.what());
    return false;
  }

  stat = summarize(v);
  std::printf("[model] OK\n");
  return true;
}

// ---- Reporting -------------------------------------------------------------

void printRow(const char *label, int N, const Stat &s) {
  std::printf("  %-10s %6d  %9.1f %9.1f %9.1f\n", label, N, s.median, s.mean,
              s.max);
}

} // namespace

int main(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto next = [&](int &dst) {
      if (i + 1 < argc)
        dst = std::atoi(argv[++i]);
    };
    if (s == "--M")
      next(a.M);
    else if (s == "--K")
      next(a.K);
    else if (s == "--q-N")
      next(a.q_N);
    else if (s == "--kv-N")
      next(a.kv_N);
    else if (s == "--iters")
      next(a.iters);
    else if (s == "--warmup")
      next(a.warmup);
    else if (s == "--seed") {
      if (i + 1 < argc)
        a.seed = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (s == "--model-path") {
      if (i + 1 < argc)
        a.model_path = argv[++i];
    } else if (s == "--no-model")
      a.run_model = false;
    else if (s == "--no-fused")
      a.run_fused = false;
    else {
      usage();
      return s == "--help" || s == "-h" ? 0 : 2;
    }
  }

  // A build stamp, because reading results off a stale binary has happened
  // here before and looks exactly like a real measurement.
  std::printf("hexkl_qkv_chain  built: %s %s\n", __DATE__, __TIME__);
  std::printf("  qwen3-0.6b QKV   M=%d  K=%d   q_N=%d  k_N=%d  v_N=%d\n", a.M,
              a.K, a.q_N, a.kv_N, a.kv_N);
  std::printf("  iters=%d  warmup=%d  seed=%u  weights=random uniform[-0.5, "
              "0.5]\n",
              a.iters, a.warmup, a.seed);

#ifdef ENABLE_HEXKL
  nntrainer::Engine::Global();
  const bool htp = nntrainer::HtpBackend::global().enabled();
  std::printf("  HTP backend: %s\n", htp ? "enabled" : "NOT AVAILABLE");
  if (!htp) {
    std::printf(
      "\nThe NPU is not up, so nothing below would measure HexKL.\n"
      "Check the skel is on the device and libsdkl.so is loadable.\n");
    return 1;
  }
#else
  std::printf("  built without ENABLE_HEXKL -- CPU only, no HexKL path\n");
  return 1;
#endif

  if (a.q_N % 32 || a.kv_N % 32 || a.K % 2) {
    std::printf("\nq_N and kv_N must be multiples of 32 and K must be even "
                "(HMX tiling).\n");
    return 2;
  }

  const auto A = randomActs(a.M, a.K, a.seed + 7);

  // ---- model path ----
  Stat model_stat;
  bool model_ok = false;
  if (a.run_model) {
    std::printf("\n=== model path (real graph) ===\n");
    model_ok = runModel(a, A, model_stat);
    if (!model_ok)
      std::printf(
        "[model] falling through to `direct`; the numbers below are\n"
        "        the kernel and dispatch only, not graph plumbing.\n");
  } else {
    std::printf("\n=== model path skipped (--no-model) ===\n");
  }

  // ---- direct path ----
  std::printf("\n=== direct path (3 x Tensor::dot, QINT4_HTP) ===\n");
  std::vector<Projection> projs;
  projs.push_back(bakeProjection("q_proj", a.K, a.q_N, a.seed + 1));
  projs.push_back(bakeProjection("k_proj", a.K, a.kv_N, a.seed + 2));
  projs.push_back(bakeProjection("v_proj", a.K, a.kv_N, a.seed + 3));

  Stat total;
  runDirect(a, projs, A, total);

  std::printf("\n  %-10s %6s  %9s %9s %9s\n", "proj", "N", "median", "mean",
              "max");
  std::printf("  %-10s %6s  %9s %9s %9s\n", "----------", "------", "---------",
              "---------", "---------");
  for (const auto &p : projs)
    printRow(p.name, p.N, p.stat);
  printRow("QKV total", a.q_N + 2 * a.kv_N, total);

  std::printf("\n  phase breakdown of the last timed call (us)\n");
  std::printf("  %-10s %8s %8s %8s %8s %8s\n", "proj", "scan", "quant", "stage",
              "npu", "dequant");
  for (const auto &p : projs)
    std::printf("  %-10s %8.1f %8.1f %8.1f %8.1f %8.1f\n", p.name, p.scan,
                p.quant, p.stage, p.npu, p.dequant);

  // ---- fused comparison ----
  if (a.run_fused) {
    const int fused_N = a.q_N + 2 * a.kv_N;
    std::printf("\n=== fused comparison (1 x N=%d) ===\n", fused_N);
    const Stat f = runFused(a, fused_N, A);
    printRow("qkv_fused", fused_N, f);

    const double saving = total.median - f.median;
    std::printf("\n  three calls   %8.1f us\n", total.median);
    std::printf("  one call      %8.1f us\n", f.median);
    std::printf("  difference    %8.1f us  (%.1f%% of the three-call cost)\n",
                saving, total.median > 0 ? 100.0 * saving / total.median : 0.0);
    std::printf(
      "\n  Same arithmetic either way -- 2048+1024+1024 = %d columns\n"
      "  against K=%d. What the difference buys is two fewer\n"
      "  per-call overheads: activation scan and quantize repeated\n"
      "  over the same input, plus two extra RPC round trips.\n",
      fused_N, a.K);
  }

  if (a.run_model && model_ok) {
    std::printf("\n=== model vs direct ===\n");
    std::printf("  model inference   %8.1f us  (median, includes the graph)\n",
                model_stat.median);
    std::printf("  direct 3 x dot    %8.1f us  (median)\n", total.median);
    std::printf("  graph overhead    %8.1f us\n",
                model_stat.median - total.median);
  }

  return 0;
}
