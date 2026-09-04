// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jungwon-Lee <jungone.lee@samsung.com>
 *
 * @file   lfm2_moe_causallm.cpp
 * @date   06 July 2026
 * @brief  This defines the LFM2-8B-A1B Mixture-of-Experts causal LM.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <lfm2_moe_causallm.h>
#include <lfm2_moe_layer.h>

#include <app_context.h>
#include <engine.h>
#include <llm_util.hpp>
#include <model.h>

#include <sstream>

namespace causallm {

namespace {
/**
 * @brief Parse a comma-separated list of layer_ids, e.g. "0,1,3". Blank
 *        entries (from "", or a trailing/doubled comma) are skipped rather
 *        than throwing, so an empty moe_htp_layers value stays "no
 *        restriction" (see MOE_HTP_LAYERS's doc in the header).
 */
std::set<int> parseLayerIdList(const std::string &csv) {
  std::set<int> ids;
  std::stringstream ss(csv);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (!tok.empty())
      ids.insert(std::stoi(tok));
  }
  return ids;
}
} // namespace

void Lfm2MoeCausalLM::setupParameters(json &cfg, json &generation_cfg,
                                      json &nntr_cfg) {
  // Parse the LFM2 backbone parameters (dims, layer_types, conv, ...).
  Lfm2CausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);

  // MoE-specific parameters.
  try {
    NUM_EXPERTS = cfg["num_experts"];
    NUM_EXPERTS_PER_TOK = cfg["num_experts_per_tok"];
    MOE_INTERMEDIATE_SIZE = cfg["moe_intermediate_size"];
  } catch (const std::exception &e) {
    throw std::runtime_error(
      "Lfm2Moe: num_experts, num_experts_per_tok and moe_intermediate_size "
      "must be specified in the config file");
  }
  // Layers [0, num_dense_layers) keep the dense SwiGLU FFN. Optional (default 0).
  NUM_DENSE_LAYERS = cfg.value("num_dense_layers", 0);

  // MoE expert FFN weight dtype. Defaults to FC_LAYER_DTYPE (set by
  // Lfm2CausalLM::setupParameters above) so an unquantized or fc_dtype==
  // moe_dtype model is unaffected; nntr_quantize's --moe_dtype writes
  // moe_layer_dtype when the two are meant to differ (see quantize.cpp's
  // buildLayerDtypeMap docstring).
  MOE_LAYER_DTYPE = nntr_cfg.value("moe_layer_dtype", FC_LAYER_DTYPE);

  // MoE FFN engine. Defaults to "cpu" so an existing run is unaffected
  // unless nntr_config.json opts in; MOE_HTP_LAYERS (empty = every MoE
  // layer) bounds a real checkpoint's device run to what the HTP weight
  // registry can hold at once -- see the two fields' docs in the header.
  MOE_ENGINE = nntr_cfg.value("moe_engine", std::string("cpu"));
  MOE_HTP_LAYERS =
    parseLayerIdList(nntr_cfg.value("moe_htp_layers", std::string("")));
}

Tensor Lfm2MoeCausalLM::createMoeLayer(const int layer_id, Tensor input) {
  const std::string engine =
    (MOE_HTP_LAYERS.empty() || MOE_HTP_LAYERS.count(layer_id)) ? MOE_ENGINE
                                                               : "cpu";
  LayerHandle moe(createLayer(
    "lfm2_moe",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", MOE_INTERMEDIATE_SIZE),
     withKey("num_experts", NUM_EXPERTS),
     withKey("num_experts_per_token", NUM_EXPERTS_PER_TOK),
     withKey("moe_activation", "swish"),
     withKey("weight_dtype", MOE_LAYER_DTYPE), withKey("engine", engine)}));
  return moe(input);
}

Tensor Lfm2MoeCausalLM::createMlp(const int layer_id, int dim, int hidden_dim,
                                  Tensor input) {
  // Dense SwiGLU FFN for the first NUM_DENSE_LAYERS layers.
  if (layer_id < static_cast<int>(NUM_DENSE_LAYERS))
    return Transformer::createMlp(layer_id, dim, hidden_dim, input);

  // MoE FFN for the remaining layers.
  return createMoeLayer(layer_id, input);
}

void Lfm2MoeCausalLM::registerCustomLayers() {

  Lfm2CausalLM::registerCustomLayers();
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(nntrainer::createLayer<causallm::Lfm2MoELayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register Lfm2MoELayer factory, reason: " << e.what()
              << std::endl;
  }
}

} // namespace causallm
