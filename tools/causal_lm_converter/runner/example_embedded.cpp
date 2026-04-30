// SPDX-License-Identifier: Apache-2.0
/**
 * @file   example_embedded.cpp
 * @brief  Smallest possible example of consuming the importer from your own
 *         application. Build target: ``causal_lm_example_embedded``.
 *
 * This binary mirrors what the runner does, but stripped of CLI niceties so
 * that the import flow itself is the only thing on screen. Copy it, change
 * three lines, and you have a model loader for any application.
 */

#include <cstdlib>
#include <iostream>
#include <vector>

#include "causal_lm_importer.h"

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <model.ini> <model.safetensors> <nntr_config.json>\n";
    return EXIT_FAILURE;
  }

  // 1. Register the layer factories. Once per process is enough.
  causal_lm::register_custom_layers();

  // 2. Build, compile, initialize, load weights.
  causal_lm::ImportOptions opts;
  opts.ini_path            = argv[1];
  opts.weights_path        = argv[2];
  opts.runtime_config_path = argv[3];
  opts.execution_mode      = ml::train::ExecutionMode::INFERENCE;
  auto imported = causal_lm::import_model(opts);

  // 3. Use the model. The runtime config tells us what shapes to feed.
  std::vector<float> input(
    static_cast<size_t>(imported.runtime.batch_size) *
      imported.runtime.init_seq_len,
    0.0f);
  std::vector<float *> inputs = {input.data()};

  auto outputs = imported.model->incremental_inference(
    imported.runtime.batch_size, inputs, /*labels=*/{},
    imported.runtime.init_seq_len, 0, imported.runtime.init_seq_len);

  std::cout << "loaded ok; first logit = "
            << (outputs.empty() ? 0.0f : outputs.front()[0]) << std::endl;
  return EXIT_SUCCESS;
}
