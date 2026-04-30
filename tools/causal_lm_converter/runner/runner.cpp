// SPDX-License-Identifier: Apache-2.0
/**
 * @file   runner.cpp
 * @brief  Thin CLI wrapper around ``causal_lm::Importer`` for build-and-run
 *         validation of converted causal-LM models.
 *
 * The interesting logic lives in ``causal_lm_importer.{h,cpp}`` and is
 * intended to be reused by other applications (a tokenizer-driven generator,
 * an Android JNI bridge, a benchmark harness, etc.). This binary exists only
 * so that:
 *   - Tests have something concrete to invoke.
 *   - End users have a one-shot "does my converted model load and forward?"
 *     diagnostic without writing any C++ themselves.
 *
 * Pipeline (delegated to causal_lm::import_model):
 *   register_custom_layers -> createModel -> setProperty -> loadFromConfig
 *   -> compile -> initialize -> load(safetensors)
 *
 * Then we feed all-zeros token IDs (vocab-bounds-safe for any model) and
 * print the first few logits so callers can diff non-finite outputs.
 */

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "causal_lm_importer.h"

namespace {

void usage(const char *argv0) {
  std::cerr
    << "Usage: " << argv0
    << " <model.ini> <model.safetensors> <nntr_config.json> [--no-forward]\n"
    << "  Loads a converter-produced causal-LM and (optionally) runs one\n"
    << "  forward pass with all-zero token IDs.\n";
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    usage(argv[0]);
    return 1;
  }
  causal_lm::ImportOptions opts;
  opts.ini_path = argv[1];
  opts.weights_path = argv[2];
  opts.runtime_config_path = argv[3];
  opts.execution_mode = ml::train::ExecutionMode::INFERENCE;

  bool run_forward = true;
  for (int i = 4; i < argc; ++i) {
    if (std::strcmp(argv[i], "--no-forward") == 0) run_forward = false;
  }

  try {
    causal_lm::register_custom_layers();
    auto imported = causal_lm::import_model(opts);
    auto &model = *imported.model;
    const auto &rc = imported.runtime;

    std::cout << "[runner] ini=" << opts.ini_path << "\n"
              << "[runner] weights=" << opts.weights_path << "\n"
              << "[runner] batch=" << rc.batch_size
              << " init_seq=" << rc.init_seq_len
              << " max_seq=" << rc.max_seq_len
              << " tensor_type=" << rc.model_tensor_type << std::endl;
    std::cout << "[runner] weights loaded" << std::endl;

    if (!run_forward) {
      std::cout << "[runner] OK (no forward)" << std::endl;
      return 0;
    }

    // All-zeros input keeps embed_idx within any vocab. Real applications
    // would replace this with tokenizer outputs.
    std::vector<float> input(
      static_cast<size_t>(rc.batch_size) * rc.init_seq_len, 0.0f);
    std::vector<float *> inputs = {input.data()};

    auto outputs = model.incremental_inference(
      rc.batch_size, inputs, /*labels=*/{}, rc.init_seq_len, 0,
      rc.init_seq_len);

    if (outputs.empty() || outputs.front() == nullptr) {
      throw std::runtime_error("forward returned no outputs");
    }

    constexpr int kSample = 8;
    bool all_finite = true;
    for (int i = 0; i < kSample; ++i) {
      if (!std::isfinite(outputs.front()[i])) {
        all_finite = false;
        break;
      }
    }
    std::cout << "[runner] forward OK, first " << kSample << " logits:";
    for (int i = 0; i < kSample; ++i) {
      std::cout << " " << outputs.front()[i];
    }
    std::cout << std::endl;
    if (!all_finite) {
      std::cerr << "[runner] FAIL: non-finite values in output" << std::endl;
      return 2;
    }
    std::cout << "[runner] OK" << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "[runner] ERROR: " << e.what() << std::endl;
    return 1;
  }
}
