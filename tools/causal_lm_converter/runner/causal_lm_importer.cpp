// SPDX-License-Identifier: Apache-2.0
/**
 * @file   importer.cpp
 * @brief  Implementation of the runtime-side importer; see importer.h.
 */

#include "causal_lm_importer.h"

#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <app_context.h>
#include <engine.h>
#include <layer.h>

#include <embedding_layer.h>
#include <mha_core.h>
#include <reshaped_rms_norm.h>
#include <rms_norm.h>
#include <swiglu.h>
#include <tie_word_embedding.h>

namespace causal_lm {

namespace {

std::string trim(const std::string &s) {
  size_t b = 0;
  while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  size_t e = s.size();
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

std::string read_file(const std::string &path) {
  std::ifstream f(path);
  if (!f) {
    throw std::runtime_error("cannot open " + path);
  }
  std::stringstream buf;
  buf << f.rdbuf();
  return buf.str();
}

/**
 * Tiny single-purpose JSON field extractor.
 *
 * We intentionally avoid pulling jsoncpp/nlohmann into this small library:
 * the runtime config is a flat object of "key": (string|number) pairs and
 * anything more elaborate is the Python emitter's responsibility.
 */
std::string json_field(const std::string &json, const std::string &key) {
  const std::string needle = "\"" + key + "\"";
  size_t pos = json.find(needle);
  if (pos == std::string::npos) return "";
  pos = json.find(':', pos);
  if (pos == std::string::npos) return "";
  ++pos;
  while (pos < json.size() &&
         std::isspace(static_cast<unsigned char>(json[pos]))) {
    ++pos;
  }
  if (pos >= json.size()) return "";
  if (json[pos] == '"') {
    size_t end = json.find('"', pos + 1);
    if (end == std::string::npos) return "";
    return json.substr(pos + 1, end - pos - 1);
  }
  // Numeric / boolean / null: read up to the next delimiter.
  size_t end = pos;
  while (end < json.size() && json[end] != ',' && json[end] != '}' &&
         json[end] != '\n') {
    ++end;
  }
  return trim(json.substr(pos, end - pos));
}

} // namespace

void register_custom_layers() {
  auto &engine = nntrainer::Engine::Global();
  auto *ctx = static_cast<nntrainer::AppContext *>(
    engine.getRegisteredContext("cpu"));

  // Each registration is tolerant of duplicates so that callers can invoke
  // this from multiple modules / tests without coordinating order.
  auto try_reg = [&](auto fn, const char *name) {
    try {
      ctx->registerFactory(fn);
    } catch (const std::invalid_argument &e) {
      std::cerr << "[causal_lm::importer] note: " << name
                << " already registered (" << e.what() << ")" << std::endl;
    }
  };
  try_reg(nntrainer::createLayer<causallm::SwiGLULayer>, "swiglu");
  try_reg(nntrainer::createLayer<causallm::RMSNormLayer>, "rms_norm");
  try_reg(nntrainer::createLayer<causallm::MHACoreLayer>, "mha_core");
  try_reg(nntrainer::createLayer<causallm::TieWordEmbedding>,
          "tie_word_embeddings");
  try_reg(nntrainer::createLayer<causallm::EmbeddingLayer>, "embedding_layer");
  try_reg(nntrainer::createLayer<causallm::ReshapedRMSNormLayer>,
          "reshaped_rms_norm");
}

RuntimeConfig parse_runtime_config(const std::string &json_path) {
  std::string j = read_file(json_path);
  RuntimeConfig rc;
  if (auto v = json_field(j, "batch_size"); !v.empty()) {
    rc.batch_size = static_cast<unsigned int>(std::stoul(v));
  }
  if (auto v = json_field(j, "init_seq_len"); !v.empty()) {
    rc.init_seq_len = static_cast<unsigned int>(std::stoul(v));
  }
  if (auto v = json_field(j, "max_seq_len"); !v.empty()) {
    rc.max_seq_len = static_cast<unsigned int>(std::stoul(v));
  }
  if (auto v = json_field(j, "model_tensor_type"); !v.empty()) {
    rc.model_tensor_type = v;
  }
  if (auto v = json_field(j, "model_file_name"); !v.empty()) {
    rc.model_file_name = v;
  }
  if (auto v = json_field(j, "vocab_size"); !v.empty()) {
    rc.vocab_size = static_cast<unsigned int>(std::stoul(v));
  }
  return rc;
}

ImportedModel import_model(const ImportOptions &opts) {
  if (opts.ini_path.empty()) {
    throw std::runtime_error("ImportOptions.ini_path must be set");
  }
  if (opts.runtime_config_path.empty()) {
    throw std::runtime_error("ImportOptions.runtime_config_path must be set");
  }
  if (!opts.skip_weight_load && opts.weights_path.empty()) {
    throw std::runtime_error(
      "ImportOptions.weights_path is empty; set it or use skip_weight_load");
  }

  ImportedModel imp;
  imp.runtime = parse_runtime_config(opts.runtime_config_path);

  imp.model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  std::vector<std::string> props = {
    "batch_size=" + std::to_string(imp.runtime.batch_size),
    "epochs=1",
    "model_tensor_type=" + imp.runtime.model_tensor_type,
  };
  imp.model->setProperty(props);

  if (imp.model->loadFromConfig(opts.ini_path) != 0) {
    throw std::runtime_error("loadFromConfig failed for " + opts.ini_path);
  }
  if (imp.model->compile(opts.execution_mode) != 0) {
    throw std::runtime_error("compile failed");
  }
  if (imp.model->initialize(opts.execution_mode) != 0) {
    throw std::runtime_error("initialize failed");
  }

  if (!opts.skip_weight_load) {
    imp.model->load(opts.weights_path,
                    ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS);
  }
  return imp;
}

} // namespace causal_lm
