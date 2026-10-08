// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   htp_e2e_test.cpp
 * @date   27 Sep 2026
 * @brief  Host E2E driver of the in-process HTP build (#84): the tiny
 *         LFM2-MoE fixture, prefill plus greedy decode, through nntrainer's
 *         own decode loop and the real ARM-side HTP path
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Built only with -Dhtp-inproc=true (Applications/CausalLM/meson.build).
 * Thin on purpose: the code under test is HtpComputeOps / HtpBackend and
 * the model's layers, reached through CausalLMTestAdapter exactly as the
 * tiny-fixture tests reach them; this only picks the engine, prints the
 * E2E lines (hvx_impl's format, LEDGER section 4 lift 4) and writes the
 * per-step logits next to the MoE dumps NNTR_HTP_DUMP produces.
 *
 *   htp_e2e_test --model <quantized dir> --tokenizer <tokenizer.json>
 *                [--prompt 16] [--steps 8] [--moe-engine htp|cpu]
 *                [--dump <dir>] [--max-seq N] [--run] [--repack]
 *                [--load-only]
 *
 * The prompt is deterministic, ids[i] = 1 + (7 i mod 30): inside the
 * 32-token vocabulary, never bos (0) or eos (31). Output:
 *   E2E step k pos=p n=m top1=t logprob=l margin=d   (k = 0 is the prefill;
 *                                                     d = top1 - top2 logit)
 *   E2E gen t0 t1 ...
 * --run goes through the app's own CausalLM::run instead (#134): the same
 * ids as text (the fixture tokenizer is WordLevel + Whitespace, so
 * 1 -> hello, 2 -> world, k -> tok<k> maps back 1:1), num_to_generate =
 * steps - 1 (run emits the prefill token plus that many), no E2E step
 * lines, and E2E gen read back from the model's token history. It is the
 * path NNTR_PPL_DECODE lives on.
 * The model class follows config.json's architectures: Gemma4ForCausalLM
 * is the Gemma 4 MoE fixture ([plan 201 S4]), anything else LFM2-MoE.
 * --repack calls repack_weight after the load, as the app's main does
 * (#225): the load-time FC registrations, the FC WH sidecar and the
 * warm-up calls (which the MoE dumps then hold too).
 * [#260] --load-only stops after the load (and --repack): the backend's
 * load banners, then `E2E load kv_alloc=<0|1> kv_shared=<0|1>`: whether
 * the model installed the attention engine's shared KV allocator, and
 * whether a slab came from it (0 in-process: no rpcmem there).
 * Exit 0, or 1 with `E2E FAIL <reason>` on any exception.
 */

#include <causallm_test_utils.h>
#include <gemma4_moe_causallm.h>
#include <lfm2_moe_causallm.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

struct Options {
  std::string model, tokenizer, engine = "htp", dump;
  unsigned prompt = 16, steps = 8, max_seq = 0;
  bool run = false, repack = false, load_only = false;
};

Options parse(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc)
        throw std::invalid_argument(a + " needs a value");
      return argv[++i];
    };
    if (a == "--model")
      o.model = value();
    else if (a == "--tokenizer")
      o.tokenizer = value();
    else if (a == "--moe-engine")
      o.engine = value();
    else if (a == "--dump")
      o.dump = value();
    else if (a == "--prompt")
      o.prompt = static_cast<unsigned>(std::stoul(value()));
    else if (a == "--steps")
      o.steps = static_cast<unsigned>(std::stoul(value()));
    else if (a == "--max-seq")
      o.max_seq = static_cast<unsigned>(std::stoul(value()));
    else if (a == "--run")
      o.run = true;
    else if (a == "--load-only")
      o.load_only = true;
    else if (a == "--repack")
      o.repack = true;
    else
      throw std::invalid_argument("unknown option " + a);
  }
  if (o.model.empty() || o.tokenizer.empty())
    throw std::invalid_argument("--model and --tokenizer are required");
  if (o.engine != "htp" && o.engine != "cpu")
    throw std::invalid_argument("--moe-engine must be htp or cpu");
  if (o.prompt == 0 || o.steps == 0)
    throw std::invalid_argument("--prompt and --steps must be > 0");
  if (o.max_seq == 0)
    o.max_seq = o.prompt + o.steps;
  if (o.max_seq < o.prompt + o.steps)
    throw std::invalid_argument("--max-seq is below prompt + steps");
  return o;
}

void writeLogits(const std::string &dir, size_t step, const float *p,
                 size_t n) {
  const std::string path = dir + "/logits_" + std::to_string(step) + ".f32";
  FILE *f = std::fopen(path.c_str(), "wb");
  if (f == nullptr || std::fwrite(p, sizeof(float), n, f) != n)
    throw std::runtime_error("cannot write " + path);
  std::fclose(f);
}

template <typename Model>
int runModel(const Options &o, nlohmann::json &cfg, nlohmann::json &gen,
             nlohmann::json &nntr, const std::string &weights);

int run(const Options &o) {
  namespace fs = std::filesystem;
  const fs::path dir = o.model;
  auto cfg = causallm::LoadJsonFile((dir / "config.json").string());
  auto gen = causallm::LoadJsonFile((dir / "generation_config.json").string());
  auto nntr = causallm::LoadJsonFile((dir / "nntr_config.json").string());
  nntr["tokenizer_file"] = o.tokenizer;
  nntr["moe_engine"] = o.engine;
  nntr["max_seq_len"] = o.max_seq;
  nntr["num_to_generate"] = o.run ? o.steps - 1 : o.steps;
  // the prefill buffer; the fixture says 4. run() records the prefill's
  // token only when the prompt is shorter than it, as in the app's config.
  // A config that asks for more keeps it (#222: init_seq_len 1024).
  nntr["init_seq_len"] =
    std::max(nntr.value("init_seq_len", 0u), o.run ? o.prompt + 1 : o.prompt);
  if (nntr["init_seq_len"].get<unsigned>() > o.max_seq)
    throw std::invalid_argument("the config's init_seq_len is above --max-seq");
  const bool gemma = cfg.contains("architectures") &&
                     cfg["architectures"].is_array() &&
                     !cfg["architectures"].empty() &&
                     cfg["architectures"][0] == "Gemma4ForCausalLM";
  // Gemma 4 nests its shape in text_config; the model lifts it the same
  // way (Gemma4Transformer::sanitizeConfig), before the override below
  if (gemma && cfg.contains("text_config"))
    for (auto it = cfg["text_config"].begin(); it != cfg["text_config"].end();
         ++it)
      if (!cfg.contains(it.key()))
        cfg[it.key()] = it.value();
  if (cfg.value("max_position_embeddings", 0u) < o.max_seq)
    cfg["max_position_embeddings"] = o.max_seq;
  const std::string weights =
    (dir / nntr["model_file_name"].get<std::string>()).string();

  if (!o.dump.empty()) {
    fs::create_directories(o.dump);
    // The MoE dumps (htp_compute_ops.cpp, dumpMoeCall) land in the same
    // directory as the logits; the hook reads the variable once, before
    // the first call, which is after the model is built below.
    setenv("NNTR_HTP_DUMP", o.dump.c_str(), 1);
  }

  return gemma
           ? runModel<causallm::Gemma4MoECausalLM>(o, cfg, gen, nntr, weights)
           : runModel<causallm::Lfm2MoeCausalLM>(o, cfg, gen, nntr, weights);
}

template <typename Model>
int runModel(const Options &o, nlohmann::json &cfg, nlohmann::json &gen,
             nlohmann::json &nntr, const std::string &weights) {
  causallm_test::CausalLMTestAdapter<Model> model(cfg, gen, nntr);
  model.initializeModel();
  model.loadWeight(weights);
  if (o.repack)
    model.repack_weight();
  if (o.load_only) {
    const int kv = model.kvCacheShared();
    std::printf("E2E load kv_alloc=%d kv_shared=%d\n", kv & 1, kv >> 1);
    return 0;
  }

  std::vector<unsigned int> ids(o.prompt);
  for (unsigned i = 0; i < o.prompt; ++i)
    ids[i] = 1u + (7u * i) % 30u;
  const size_t vocab = cfg["vocab_size"].get<size_t>();

  if (o.run) {
    std::string text;
    for (unsigned i = 0; i < o.prompt; ++i)
      text += (i ? " " : "") + (ids[i] == 1   ? std::string("hello")
                                : ids[i] == 2 ? std::string("world")
                                              : "tok" + std::to_string(ids[i]));
    model.runPrompt(text);
    std::string line = "E2E gen";
    for (unsigned k = 0; k < o.steps; ++k)
      line += " " + std::to_string(model.tokenAt(o.prompt + k));
    std::printf("%s\n", line.c_str());
    return 0;
  }

  const auto tokens = model.greedyGenerateFromIds(
    ids, o.steps, [&](size_t step, const float *logits) {
      const float *top = std::max_element(logits, logits + vocab);
      double lse = 0.0;
      float second = -INFINITY;
      for (size_t v = 0; v < vocab; ++v) {
        lse += std::exp(static_cast<double>(logits[v] - *top));
        if (logits + v != top && logits[v] > second)
          second = logits[v];
      }
      const double logprob = -std::log(lse);
      std::printf(
        "E2E step %zu pos=%u n=%u top1=%ld logprob=%.6f margin=%.6f\n", step,
        step == 0 ? 0u : o.prompt + static_cast<unsigned>(step) - 1u,
        step == 0 ? o.prompt : 1u, static_cast<long>(top - logits), logprob,
        static_cast<double>(*top - second));
      if (!o.dump.empty())
        writeLogits(o.dump, step, logits, vocab);
    });
  std::string line = "E2E gen";
  for (unsigned int t : tokens)
    line += " " + std::to_string(t);
  std::printf("%s\n", line.c_str());
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    return run(parse(argc, argv));
  } catch (const std::exception &e) {
    std::printf("E2E FAIL %s\n", e.what());
    std::fflush(stdout);
    // _exit: the model's destructors throw on a half-built model, and a
    // throw during unwinding would hide the message above.
    _exit(1);
  }
}
