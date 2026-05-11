// SPDX-License-Identifier: Apache-2.0
/**
 * @file   importer.h
 * @brief  Reusable runtime-side importer for causal_lm_converter artifacts.
 *
 * The importer turns the three artifacts produced by ``nntr_causal_lm_converter``
 * (an INI graph, a name-keyed safetensors blob, and a small JSON runtime
 * config) into a fully-initialized ``ml::train::Model`` ready to run forward
 * passes.
 *
 * Typical usage from another application:
 *
 * @code
 *   #include "causal_lm_importer.h"
 *
 *   causal_lm::register_custom_layers();   // call once per process
 *
 *   causal_lm::ImportOptions opts;
 *   opts.ini_path             = "qwen3.ini";
 *   opts.weights_path         = "qwen3.safetensors";
 *   opts.runtime_config_path  = "nntr_config.json";
 *   opts.execution_mode       = ml::train::ExecutionMode::INFERENCE;
 *
 *   causal_lm::ImportedModel imported = causal_lm::import_model(opts);
 *   auto &model     = *imported.model;
 *   auto batch      = imported.runtime.batch_size;
 *   auto init_seq   = imported.runtime.init_seq_len;
 *
 *   std::vector<float> input(batch * init_seq, 0.0f);
 *   std::vector<float *> inputs = {input.data()};
 *   auto outputs = model.incremental_inference(batch, inputs, {}, init_seq, 0, init_seq);
 * @endcode
 *
 * The importer is header-light by design: it adds no global state of its own
 * and keeps the existing ``ml::train::Model`` semantics. Side-effects are
 * limited to (a) layer-factory registration via the AppContext and (b) file
 * IO inside the call.
 */

#pragma once

#include <memory>
#include <string>

#include <model.h>

namespace causal_lm {

/**
 * @brief Runtime hints read from ``nntr_config.json``.
 *
 * Mirrors what the Python ``RuntimeConfig`` writes, so importers only need
 * the fields the C++ side actually consumes. Anything additional in the JSON
 * is silently ignored.
 */
struct RuntimeConfig {
  unsigned int batch_size      = 1;
  unsigned int init_seq_len    = 8;
  unsigned int max_seq_len     = 8;
  /** Vocabulary size; 0 means "not declared in the JSON". Useful so callers
   *  can compute the last-token logit slice without parsing the INI. */
  unsigned int vocab_size      = 0;
  std::string  model_tensor_type = "FP32-FP32";
  std::string  model_file_name   = "model.safetensors";
};

/** @brief Options that drive a single import. */
struct ImportOptions {
  std::string ini_path;
  std::string weights_path;
  std::string runtime_config_path;
  ml::train::ExecutionMode execution_mode = ml::train::ExecutionMode::INFERENCE;
  /** If true, skip ``model->load(weights)`` so callers can stage it later. */
  bool skip_weight_load = false;
};

/** @brief A model + the runtime config it was imported with. */
struct ImportedModel {
  std::unique_ptr<ml::train::Model> model;
  RuntimeConfig runtime;
};

/**
 * @brief Register the causal-LM custom layers with the global AppContext.
 *
 * Safe to call multiple times: duplicate registrations are caught and logged
 * but do not throw. Must be invoked at least once before
 * :func:`import_model` so that ``createLayer("rms_norm" / "swiglu" / ...)``
 * succeeds during ``loadFromConfig``.
 *
 * Currently registers: ``rms_norm``, ``reshaped_rms_norm``, ``swiglu``,
 * ``mha_core``, ``embedding_layer``, ``tie_word_embeddings``.
 */
void register_custom_layers();

/**
 * @brief Parse the runtime JSON without performing any model construction.
 *
 * Useful for callers who want to inspect the config before deciding whether
 * to call :func:`import_model`.
 */
RuntimeConfig parse_runtime_config(const std::string &json_path);

/**
 * @brief Build, compile, initialize, and (optionally) load weights.
 *
 * Pipeline:
 *  1. ``createModel(NEURAL_NET)``
 *  2. ``setProperty(batch_size, model_tensor_type)`` from the runtime config
 *  3. ``loadFromConfig(ini_path)`` to build the symbolic graph
 *  4. ``compile(execution_mode)``
 *  5. ``initialize(execution_mode)``
 *  6. ``load(weights_path, MODEL_FORMAT_SAFETENSORS)`` unless skipped
 *
 * Throws ``std::runtime_error`` with a contextual message on any step's
 * failure; the partial model is dropped before the exception propagates.
 */
ImportedModel import_model(const ImportOptions &opts);

} // namespace causal_lm
