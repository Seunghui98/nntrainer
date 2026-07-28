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
  double min = 0.0, median = 0.0, mean = 0.0, max = 0.0;
};

Stat summarize(std::vector<double> v) {
  Stat s;
  if (v.empty())
    return s;
  std::sort(v.begin(), v.end());
  s.min = v.front();
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
  double cold_us = 0.0;     // very first call: uploads the weight
  Stat stat;                // steady state, weight resident
  // Phase breakdown from the last timed call, in microseconds.
  double scan = 0, quant = 0, stage = 0, npu = 0, dequant = 0;
  bool neon = false;
  bool p_valid = false; // false = the kernel recorded no profile

  double hostComputeUs() const { return scan + quant + dequant; }
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
nntrainer::Tensor makeActivation(const Args &a, const std::vector<float> &A) {
  nntrainer::TensorDim adim(1, 1, a.M, a.K);
  nntrainer::Tensor act(adim, false);
  act.allocate();
  std::memcpy(act.getData<float>(), A.data(),
              static_cast<size_t>(a.M) * a.K * sizeof(float));
#ifdef ENABLE_HEXKL
  act.setContextData(htpContextData());
#endif
  return act;
}

/** @brief One dot() call, timed, recording the kernel's phase profile. */
double callOnce(nntrainer::Tensor &act, Projection &p, bool record) {
#ifdef ENABLE_HEXKL
  if (record)
    nntrainer::hmx::resetMmProfile();
#endif
  const double t0 = nowUs();
  // fc_layer's own call: input_.dot(weight, hidden_, false, false). trans_in
  // must be false -- the weight is [K, N], so with trans_in=true the N and K
  // roles swap and any shape with N != K is rejected.
  nntrainer::Tensor out = act.dot(p.weight, false, false);
  const double dt = nowUs() - t0;
#ifdef ENABLE_HEXKL
  if (record) {
    const auto &mp = nntrainer::hmx::lastMmProfile();
    // total_us stays zero if the kernel never ran its timers, which is how a
    // stale library shows up. Gating on it silently would make that look like
    // a working run with nothing to report.
    p.p_valid = mp.total_us > 0.0;
    p.scan = mp.scan_us;
    p.quant = mp.quant_us;
    p.stage = mp.stage_us;
    p.npu = mp.npu_us;
    p.dequant = mp.dequant_us;
    p.neon = mp.neon;
  }
#else
  (void)record;
#endif
  return dt;
}

void runDirect(const Args &a, std::vector<Projection> &projs,
               const std::vector<float> &A, Projection &total) {
  nntrainer::Tensor act = makeActivation(a, A);

  // Cold first: the very first call on each weight uploads it to NPU memory,
  // and every call after that hits the resident cache. Folding it into warmup
  // would hide the upload cost entirely.
  for (auto &p : projs)
    p.cold_us = callOnce(act, p, false);

  std::vector<std::vector<double>> per(projs.size());
  std::vector<double> totals;

  for (int i = 0; i < a.warmup + a.iters; ++i) {
    const bool timed = i >= a.warmup;
    double round = 0.0;
    for (size_t p = 0; p < projs.size(); ++p) {
      const double dt = callOnce(act, projs[p], timed);
      round += dt;
      if (timed)
        per[p].push_back(dt);
    }
    if (timed)
      totals.push_back(round);
  }

  for (size_t p = 0; p < projs.size(); ++p)
    projs[p].stat = summarize(per[p]);

  // The QKV row is a real measurement of the three in sequence, not a sum of
  // three medians -- summing medians would understate a run where the calls
  // interfere with each other.
  total.name = "QKV total";
  total.N = a.q_N + 2 * a.kv_N;
  total.stat = summarize(totals);
  total.p_valid = true;
  for (const auto &p : projs) {
    total.cold_us += p.cold_us;
    total.scan += p.scan;
    total.quant += p.quant;
    total.stage += p.stage;
    total.npu += p.npu;
    total.dequant += p.dequant;
    total.neon = p.neon;
    total.p_valid = total.p_valid && p.p_valid;
  }
}

/** @brief One matmul at the combined width, for the fused comparison. */
Projection runFused(const Args &a, int fused_N, const std::vector<float> &A) {
  Projection f = bakeProjection("qkv_fused", a.K, fused_N, a.seed + 99);
  nntrainer::Tensor act = makeActivation(a, A);

  f.cold_us = callOnce(act, f, false);

  std::vector<double> v;
  for (int i = 0; i < a.warmup + a.iters; ++i) {
    const bool timed = i >= a.warmup;
    const double dt = callOnce(act, f, timed);
    if (timed)
      v.push_back(dt);
  }
  f.stat = summarize(v);
  return f;
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

/**
 * @brief One projection's phase block, in hexkl_fc_compare's layout.
 *
 * Same shape as that tool's output on purpose: these two get read side by
 * side, and a second format would mean re-learning where to look.
 */
void dumpMethod(const Projection &p, bool show_cold) {
  std::printf("[%s]  N=%d  engine=%s\n", p.name, p.N,
              p.p_valid ? "NPU/HMX" : "unknown");
  if (show_cold)
    std::printf("  Cold run (first call, uploads weight)  %10.1f\n", p.cold_us);
  std::printf("  Steady state (weight resident)         %10.1f   <- "
              "per-call cost (mean)\n",
              p.stat.mean);
  std::printf("    min / median / max                   %10.1f /%8.1f /%8.1f\n",
              p.stat.min, p.stat.median, p.stat.max);
  if (!p.p_valid) {
    std::printf("    [phase split unavailable: kernel profile not recorded]\n");
    return;
  }
  std::printf("    NPU matmul (sdkl_npu_mm)             %10.1f   <- device\n",
              p.npu);
  std::printf("    Host compute (%-6s)                 %10.1f   <- ARM\n",
              p.neon ? "NEON" : "scalar", p.hostComputeUs());
  std::printf("      max-abs scan (M*K)                 %10.1f\n", p.scan);
  std::printf("      quantize F32->U8 (Mp*K)            %10.1f\n", p.quant);
  std::printf("      dequantize I32->F32 (M*N)          %10.1f\n", p.dequant);
  std::printf("    Buffer staging (scratch + resident)  %10.1f\n", p.stage);
}

void tableRow(const Projection &p) {
  std::printf("| %s | %d | %8.1f | %8.1f | %8.1f | %8.1f | %s |\n", p.name, p.N,
              p.stat.mean, p.stat.median, p.npu, p.hostComputeUs(),
              p.p_valid ? "NPU/HMX" : "unknown");
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

  double init_us = 0.0;
#ifdef ENABLE_HEXKL
  nntrainer::Engine::Global();
  // HtpBackend's first touch is what performs the sdkl_npu_initialize
  // bring-up, so timing it here is the same "One-time init" phase
  // hexkl_fc_compare reports.
  const double t_init = nowUs();
  const bool htp = nntrainer::HtpBackend::global().enabled();
  init_us = nowUs() - t_init;
  std::printf("  HTP backend: %s\n", htp ? "enabled" : "NOT AVAILABLE");
  if (!htp) {
    // sdkl_npu_initialize failing with Err=1 almost always means FastRPC could
    // not find the CDSP skeleton, and the reason is almost always a missing
    // ADSP_LIBRARY_PATH rather than a missing file -- so name it first.
    const char *adsp = std::getenv("ADSP_LIBRARY_PATH");
    std::printf("\nThe NPU is not up, so nothing below would measure HexKL.\n");
    if (adsp == nullptr)
      std::printf(
        "\n  ADSP_LIBRARY_PATH is NOT SET. FastRPC needs it to find the CDSP\n"
        "  skeleton (libhexkl_skel.so); without it sdkl_npu_initialize fails\n"
        "  with Err = 1. Re-run as:\n"
        "\n"
        "    cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp \\\n"
        "      ADSP_LIBRARY_PATH=/data/local/tmp ./hexkl_qkv_chain\n");
    else
      std::printf("\n  ADSP_LIBRARY_PATH=%s\n"
                  "  Check libhexkl_skel.so (V79) is in that directory.\n",
                  adsp);
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
  std::vector<Projection> projs;
  projs.push_back(bakeProjection("q_proj", a.K, a.q_N, a.seed + 1));
  projs.push_back(bakeProjection("k_proj", a.K, a.kv_N, a.seed + 2));
  projs.push_back(bakeProjection("v_proj", a.K, a.kv_N, a.seed + 3));

  Projection total;
  runDirect(a, projs, A, total);

  const int fused_N = a.q_N + 2 * a.kv_N;
  Projection fused;
  if (a.run_fused)
    fused = runFused(a, fused_N, A);

  // ---- report, in hexkl_fc_compare's phase layout ----
  std::printf("\nPhase                                    Time (us)\n");
  std::printf("One-time init (sdkl_npu_initialize)      %10.1f\n", init_us);
  for (const auto &p : projs)
    dumpMethod(p, true);
  // No cold row on the total: the three uploads happen once each and summing
  // them would read like a per-call cost, which it is not.
  dumpMethod(total, false);
  if (a.run_fused)
    dumpMethod(fused, true);

  std::printf("\n| method | N | mean us | median us | NPU us | host us | "
              "engine |\n");
  std::printf("|---|---|---|---|---|---|---|\n");
  for (const auto &p : projs)
    tableRow(p);
  tableRow(total);
  if (a.run_fused)
    tableRow(fused);

  if (a.run_fused) {
    const double saving = total.stat.median - fused.stat.median;
    std::printf("\nfused vs three calls (median)\n");
    std::printf("  three calls   %8.1f us\n", total.stat.median);
    std::printf("  one call      %8.1f us\n", fused.stat.median);
    std::printf(
      "  difference    %8.1f us  (%.1f%% of the three-call cost)\n", saving,
      total.stat.median > 0 ? 100.0 * saving / total.stat.median : 0.0);
    if (total.p_valid && fused.p_valid) {
      // Attribute the difference rather than just stating it: the host part is
      // the redundant scan+quantize, the rest is per-call NPU overhead.
      std::printf("    of which host  %8.1f us  (scan + quantize done 3x over "
                  "the same activation)\n",
                  total.hostComputeUs() - fused.hostComputeUs());
      std::printf("    of which NPU   %8.1f us  (two extra per-call "
                  "overheads)\n",
                  total.npu - fused.npu);
    }
    std::printf("\n  Same arithmetic either way -- %d+%d+%d = %d columns "
                "against K=%d.\n",
                a.q_N, a.kv_N, a.kv_N, fused_N, a.K);
  }

  if (a.run_model && model_ok) {
    std::printf("\nmodel vs direct (median)\n");
    std::printf("  model inference   %8.1f us  (includes the graph)\n",
                model_stat.median);
    std::printf("  direct 3 x dot    %8.1f us\n", total.stat.median);
    std::printf("  graph overhead    %8.1f us\n",
                model_stat.median - total.stat.median);
  }

  return 0;
}
