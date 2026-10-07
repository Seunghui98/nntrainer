// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   gemma4_attn_bench.cpp
 * @date   01 Oct 2026
 * @brief  One Gemma-4 attention layer, prefill latency on the CPU / HTP paths
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The nntrainer counterpart of NPUForge's GEMMA4_ATTN_BENCH.md: an isolated
 * attention layer of google/gemma-4-26B-A4B built from config with
 * untrained weights -- q/k/v projections, per-head q/k/v RMSNorm, RoPE,
 * attention over the KV cache, o projection -- prefilled with S random
 * hidden-state rows, timed over repeated runs. Which layer flavour
 * (sliding: head_dim 256, 8 KV heads; global: head_dim 512, 2 KV heads, V
 * reusing K) comes from layer_types[0] of the config.json, and which path
 * runs the attention from nntr_config.json's attention_engine /
 * attention_kv_dtype, exactly as in the CausalLM app.
 *
 *   gemma4_attn_bench <model_dir> [iters] [warmup] [chunk]
 *
 * model_dir holds config.json, generation_config.json, nntr_config.json;
 * init_seq_len is S. chunk (default S) prefills S rows as consecutive
 * chunk-row steps, the way a chunked prefill feeds the layer. Prints one
 * line: median / min / mean ms per full prefill. NNTR_BENCH_CPUS=<hex mask>
 * pins every thread after initialization (see pin_all_threads).
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <sched.h>
#include <string>
#include <vector>

#include "json.hpp"
#include <app_context.h>
#include <engine.h>

#include <gemma4_causallm.h>

using json = nlohmann::json;

namespace {

using causallm::Tensor;

/** @brief Gemma4CausalLM with the graph cut down to one attention block. */
class Gemma4AttnBench : public causallm::Gemma4CausalLM {
public:
  Gemma4AttnBench(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(sanitizeConfig(cfg),
                sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg,
                causallm::ModelType::CAUSALLM),
    Gemma4CausalLM(cfg, generation_cfg, nntr_cfg) {}

  unsigned int seqLen() const { return INIT_SEQ_LEN; }

  /** @brief hidden_states (B, S, H) -> attention block of layer 0. */
  std::pair<Tensor, Tensor> constructModel() override {
    Tensor x({BATCH_SIZE, 1u, INIT_SEQ_LEN, static_cast<unsigned int>(DIM)},
             "input0");
    Tensor y = createAttention(0, static_cast<int>(INIT_SEQ_LEN), NUM_HEADS,
                               HEAD_DIM, x, x, x);
    return {x, y};
  }

  /**
   * @brief Prefills S rows warmup + iters times, each from an empty cache.
   */
  void bench(unsigned int iters, unsigned int warmup, unsigned int chunk) {
    allocateAndBindKVCache();
    const size_t n = static_cast<size_t>(BATCH_SIZE) * INIT_SEQ_LEN * DIM;
    std::vector<float> hidden(n);
    uint32_t s = 0x6E4A0001u;
    for (auto &v : hidden) {
      s = s * 1664525u + 1013904223u;
      v = (static_cast<float>(s >> 8) / 16777216.0f * 2.0f - 1.0f);
    }
    // As the app: the graph lists the cache placeholders as inputs after
    // hidden_states, in name order (cache_k_l0, cache_v_l0).
    std::vector<float *> in{
      hidden.data(),
      reinterpret_cast<float *>(kv_cache.getKeyCache(0).getData()),
      reinterpret_cast<float *>(kv_cache.getValueCache(0).getData())};
    std::vector<float *> label;
    std::vector<double> ms;
    for (unsigned int i = 0; i < warmup + iters; ++i) {
      setKVCachePosition(0);
      const auto t0 = std::chrono::steady_clock::now();
      for (unsigned int from = 0; from < INIT_SEQ_LEN; from += chunk) {
        const unsigned int to = std::min(INIT_SEQ_LEN, from + chunk);
        auto out = model->incremental_inference(BATCH_SIZE, in, label,
                                                INIT_SEQ_LEN, from, to, false);
        for (auto *o : out) {
          delete[] o;
        }
      }
      const auto t1 = std::chrono::steady_clock::now();
      if (i >= warmup) {
        ms.push_back(
          std::chrono::duration<double, std::milli>(t1 - t0).count());
      }
    }
    std::sort(ms.begin(), ms.end());
    double sum = 0.0;
    for (double v : ms) {
      sum += v;
    }
    const double median = ms.size() % 2
                            ? ms[ms.size() / 2]
                            : 0.5 * (ms[ms.size() / 2 - 1] + ms[ms.size() / 2]);
    std::cout << "GEMMA4_ATTN layer_type="
              << (layer_types.empty() ? "?" : layer_types[0])
              << " S=" << INIT_SEQ_LEN << " head_dim=" << getAttentionHeadDim(0)
              << " kv_heads=" << getKVHeadCount(0) << " engine="
              << (ATTENTION_ENGINE.empty() ? "cpu" : ATTENTION_ENGINE)
              << " kv_dtype="
              << (ATTENTION_KV_DTYPE.empty() ? "fp16" : ATTENTION_KV_DTYPE)
              << " fc_dtype=" << FC_LAYER_DTYPE << " chunk=" << chunk
              << " iters=" << ms.size() << " median_ms=" << median
              << " min_ms=" << ms.front() << " mean_ms=" << sum / ms.size()
              << std::endl;
  }
};

/**
 * @brief Pins every thread of this process to the cores in @a mask.
 *
 * Opening the FastRPC session resets the process affinity to all cores,
 * so a taskset on the command line only reaches the CPU-only
 * configuration; applied here, after initialize(), both configurations
 * run the CPU layers on the same cores.
 */
void pin_all_threads(unsigned long mask) {
  cpu_set_t set;
  CPU_ZERO(&set);
  for (int c = 0; c < 64; ++c) {
    if (mask & (1ul << c)) {
      CPU_SET(c, &set);
    }
  }
  DIR *d = opendir("/proc/self/task");
  if (!d) {
    sched_setaffinity(0, sizeof(set), &set);
    return;
  }
  while (dirent *e = readdir(d)) {
    if (e->d_name[0] == '.') {
      continue;
    }
    sched_setaffinity(std::atoi(e->d_name), sizeof(set), &set);
  }
  closedir(d);
}

json load_json(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    throw std::runtime_error("cannot open " + path);
  }
  return json::parse(f);
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <model_dir> [iters] [warmup]\n";
    return 1;
  }
  const std::string dir = argv[1];
  const unsigned int iters = argc > 2 ? std::stoul(argv[2]) : 20u;
  const unsigned int warmup = argc > 3 ? std::stoul(argv[3]) : 3u;
  const unsigned int chunk = argc > 4 ? std::stoul(argv[4]) : 0u;
  try {
    json cfg = load_json(dir + "/config.json");
    json gen = load_json(dir + "/generation_config.json");
    json ncfg = load_json(dir + "/nntr_config.json");
    Gemma4AttnBench bench(cfg, gen, ncfg);
    bench.initialize();
    if (const char *m = std::getenv("NNTR_BENCH_CPUS")) {
      pin_all_threads(std::strtoul(m, nullptr, 16));
    }
    bench.bench(iters, warmup, chunk ? chunk : bench.seqLen());
  } catch (const std::exception &e) {
    std::cerr << "FATAL: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
