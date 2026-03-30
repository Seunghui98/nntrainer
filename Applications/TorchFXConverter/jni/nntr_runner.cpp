// SPDX-License-Identifier: Apache-2.0
/**
 * @file   nntr_runner.cpp
 * @date   30 Mar 2026
 * @brief  Standalone NNTrainer model runner for TorchFXConverter output.
 *
 * Loads a NNTrainer model from an INI configuration file, registers all
 * CausalLM custom layers, compiles, initializes, and summarizes the model.
 * No per-model recompilation is needed — any converted INI can be run directly.
 *
 * Usage:
 *   nntr_runner <model.ini>
 *   nntr_runner <model.ini> <weights.bin>
 *
 * Exit codes:
 *   0  — model loaded, compiled, and initialized successfully
 *   1  — error (printed to stderr)
 */

#include <iostream>
#include <stdexcept>
#include <string>

#include <app_context.h>
#include <engine.h>
#include <model.h>

// CausalLM custom layers
#include <embedding_layer.h>
#include <mha_core.h>
#include <reshaped_rms_norm.h>
#include <rms_norm.h>
#include <swiglu.h>
#include <tie_word_embedding.h>

using ModelHandle = std::unique_ptr<ml::train::Model>;

/**
 * @brief Register all CausalLM custom layers into the NNTrainer AppContext.
 *
 * Must be called before loadFromConfig so that INI layer types such as
 * "rms_norm", "mha_core", "swiglu", etc. are recognized.
 */
static void registerCustomLayers() {
  auto &ct_engine = nntrainer::Engine::Global();
  auto *app_context = static_cast<nntrainer::AppContext *>(
    ct_engine.getRegisteredContext("cpu"));

  app_context->registerFactory(
    nntrainer::createLayer<causallm::EmbeddingLayer>);
  app_context->registerFactory(
    nntrainer::createLayer<causallm::MHACoreLayer>);
  app_context->registerFactory(
    nntrainer::createLayer<causallm::RMSNormLayer>);
  app_context->registerFactory(
    nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
  app_context->registerFactory(
    nntrainer::createLayer<causallm::SwiGLULayer>);
  app_context->registerFactory(
    nntrainer::createLayer<causallm::TiedEmbeddingLayer>);
}

/**
 * @brief Load, compile, initialize, and summarize a NNTrainer model from INI.
 *
 * @param ini_path   Path to the NNTrainer INI configuration file.
 * @param bin_path   Path to the weight binary file (empty = skip weight load).
 * @return int       0 on success, 1 on failure.
 */
static int runModel(const std::string &ini_path,
                    const std::string &bin_path) {
  std::cout << "[nntr_runner] Loading model: " << ini_path << std::endl;

  ModelHandle model =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  if (model->loadFromConfig(ini_path) != 0) {
    std::cerr << "[nntr_runner] ERROR: Failed to load config: " << ini_path
              << std::endl;
    return 1;
  }

  if (model->compile(ml::train::ExecutionMode::INFERENCE) != 0) {
    std::cerr << "[nntr_runner] ERROR: Model compilation failed." << std::endl;
    return 1;
  }

  if (model->initialize(ml::train::ExecutionMode::INFERENCE) != 0) {
    std::cerr << "[nntr_runner] ERROR: Model initialization failed."
              << std::endl;
    return 1;
  }

  if (!bin_path.empty()) {
    std::cout << "[nntr_runner] Loading weights: " << bin_path << std::endl;
    try {
      model->load(bin_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
    } catch (const std::exception &e) {
      std::cerr << "[nntr_runner] ERROR: Weight load failed: " << e.what()
                << std::endl;
      return 1;
    }
  }

  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

  std::cout << "[nntr_runner] Model initialized successfully." << std::endl;
  return 0;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: nntr_runner <model.ini> [weights.bin]" << std::endl;
    std::cerr << "  model.ini   : NNTrainer INI configuration (required)"
              << std::endl;
    std::cerr << "  weights.bin : Pre-converted weight binary (optional)"
              << std::endl;
    return 1;
  }

  const std::string ini_path = argv[1];
  const std::string bin_path = (argc >= 3) ? argv[2] : "";

  try {
    registerCustomLayers();
    return runModel(ini_path, bin_path);
  } catch (const std::exception &e) {
    std::cerr << "[nntr_runner] FATAL: " << e.what() << std::endl;
    return 1;
  }
}
