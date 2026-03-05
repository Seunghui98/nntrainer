// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   debertav2.cpp
 * @date   14 January 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note  Please refer to the following code :
 * https://github.com/huggingface/transformers/blob/5c1c72b/src/transformers/models/deberta/modeling_deberta.py
 */

#include "json.hpp"

#include <filesystem>
#include <fstream>
#include <deberta_v2.h>
#include <deberta_attention_layer.h>
#include <shared_fully_connected_layer.h>
#include <llm_util.hpp>
#include <layer_context.h>
#include <app_context.h>
#include <weight_layer.h>
#include <engine.h>
#include <layer_node.h>

#include <sstream>
#include <filesystem>

using json = nlohmann::json;

namespace causallm {

void DebertaV2::constructModel() {
  std::vector<LayerHandle> layers;

  // create model
  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  layers.push_back(createLayer(
    "input", {withKey("name", "input0"),
              withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))}));

  // create embedding layer
  layers.push_back(createLayer(
    "embedding_layer",
    {"name=embedding0",
     "input_layers=input0",
     "in_dim=" + std::to_string(NUM_VOCAB),
     "weight_dtype=" + EMBEDDING_DTYPE, "out_dim=" + std::to_string(DIM),
     "scale=" + std::to_string(EMBEDDING_SCALE)}));
     
  layers.push_back(createLayer(
    "layer_normalization",
    {withKey("name", "embeddings_norm"),
     withKey("input_layers", "embedding0"),
     withKey("axis", "3"),
     withKey("epsilon", "1e-7")}));
     
  int rel_embed_size = (POSITION_BUCKETS > 0) ? POSITION_BUCKETS * 2 : ((MAX_RELATIVE_POSITIONS < 1) ? MAX_POSITION_EMBEDDINGS * 2 : MAX_RELATIVE_POSITIONS * 2);
  layers.push_back(createLayer(
    "weight",
    {withKey("name", "rel_embeddings"),
     withKey("input_layers", "embeddings_norm"),
     withKey("weight_name", "rel_embeddings"),
     withKey("dim",
             "1:1:" + std::to_string(rel_embed_size) + ":" + std::to_string(DIM)),
    withKey("input_shape",
             "1:1:" + std::to_string(rel_embed_size) + ":" + std::to_string(DIM)),
     withKey("weight_initializer", "none")}));

  layers.push_back(createLayer(
    "layer_normalization",
    {withKey("name", "rel_embeddings_norm"),
     withKey("input_layers", "rel_embeddings"),
     withKey("axis", "3"),
     withKey("epsilon", "1e-7")}));

  // create Deberta layers
  std::string last_input = "embeddings_norm";
  
  for (int i = 0; i < NUM_LAYERS; ++i) {
    std::vector<LayerHandle> encoder_layer = createDebertaLayer(i, last_input, "rel_embeddings_norm");
    layers.insert(layers.end(), encoder_layer.begin(), encoder_layer.end());
    last_input = "layer" + std::to_string(i) + "_output";
  }

  for (auto &layer : layers) {
    model->addLayer(layer);
  }
}

std::vector<LayerHandle> DebertaV2::createDebertaLayer(const int layer_id,
                                                       std::string input_name,
                                                       std::string rel_embeddings_name) {
  std::vector<LayerHandle> layers;
  std::string prefix = "layer" + std::to_string(layer_id);

  // 1. Attention Block
  std::vector<LayerHandle> attn_layers = createDebertaV2Attention(layer_id, input_name, rel_embeddings_name);
  layers.insert(layers.end(), attn_layers.begin(), attn_layers.end());

  layers.push_back(createLayer(
    "addition",
    {withKey("name", prefix + "_attention_add"),
     withKey("input_layers", {input_name, prefix + "_attention_out"})}));

  layers.push_back(createLayer(
    "layer_normalization",
    {withKey("name", prefix + "_attention_norm"),
     withKey("epsilon", "1e-7"),
     withKey("axis", "3"),
     withKey("input_layers", prefix + "_attention_add")}));

  // 2. Intermediate Block
  std::vector<std::string> inter_params = {
    withKey("name", prefix + "_intermediate"), withKey("unit", INTERMEDIATE_SIZE),
    withKey("input_layers", prefix + "_attention_norm"), withKey("activation", "gelu")};
  layers.push_back(createLayer("fully_connected", inter_params));

  std::vector<std::string> output_params = {
    withKey("name", prefix + "_output_dense"), withKey("unit", DIM),
    withKey("input_layers", prefix + "_intermediate")};
  layers.push_back(createLayer("fully_connected", output_params));

  layers.push_back(createLayer(
    "addition",
    {withKey("name", prefix + "_output_add"),
     withKey("input_layers", {prefix + "_attention_norm", prefix + "_output_dense"})}));

  layers.push_back(createLayer(
    "layer_normalization",
    {withKey("name", prefix + "_output"),
     withKey("epsilon", "1e-7"),
     withKey("axis", "3"),
     withKey("input_layers", prefix + "_output_add")}));

  return layers;
}

std::vector<LayerHandle> DebertaV2::createDebertaV2Attention(const int layer_id,
                                                             std::string input_name,
                                                             std::string rel_embeddings_name) {
  std::vector<LayerHandle> layers;
  auto Q = "layer" + std::to_string(layer_id) + "_wq";
  auto K = "layer" + std::to_string(layer_id) + "_wk";
  auto V = "layer" + std::to_string(layer_id) + "_wv";
  auto A = "layer" + std::to_string(layer_id) + "_attention";
  auto O = "layer" + std::to_string(layer_id) + "_attention_out";

  // V layer
  std::vector<std::string> v_params = {
    withKey("name", V), withKey("unit", DIM),
    withKey("disable_bias", "false"), withKey("input_layers", input_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", v_params));

  // K layer
  std::vector<std::string> k_params = {
    withKey("name", K), withKey("unit", DIM),
    withKey("disable_bias", "false"), withKey("input_layers", input_name),
    withKey("weight_initializer", "none")};
  layers.push_back(createLayer("shared_fully_connected", k_params));

  // Q layer
  std::vector<std::string> q_params = {
    withKey("name", Q), withKey("unit", DIM),
    withKey("disable_bias", "false"), withKey("input_layers", input_name),
    withKey("weight_initializer", "none")};
  layers.push_back(createLayer("shared_fully_connected", q_params));
  
  std::string attn_input_layers = Q + "," + K + "," + V;

  // p2c uses Q_rel (projected relative embeddings using Query weights)
  // DebertaAttentionLayer expects p2c input 5th (idx 4)
  if (p2c) {
    auto Q_rel = "layer" + std::to_string(layer_id) + "_wq_rel";
    // Add Q as dependency to force Q to be finalized before Q_rel
    std::string q_rel_input_layers = rel_embeddings_name;
    std::vector<std::string> q_rel_params = {
      withKey("name", Q_rel), withKey("unit", DIM),
      withKey("shared_mode", "true"),
      withKey("shared_from", Q),
      withKey("disable_bias", "false"), withKey("input_layers", {rel_embeddings_name, Q}),
      withKey("full_input_range", "true"),
      withKey("weight_initializer", "none")
    };

    layers.push_back(createLayer("shared_fully_connected", q_rel_params));
    attn_input_layers += "," + Q_rel;
  }

  // c2p uses K_rel (projected relative embeddings using Key weights)
  // DebertaAttentionLayer expects c2p input 4th (idx 3)
  if (c2p) {
    auto K_rel = "layer" + std::to_string(layer_id) + "_wk_rel";
    // Add K as dependency to force K to be finalized before K_rel
    std::string k_rel_input_layers = rel_embeddings_name;
    std::vector<std::string> k_rel_params = {
      withKey("name", K_rel), withKey("unit", DIM),
      withKey("shared_mode", "true"),
      withKey("shared_from", K),
      withKey("disable_bias", "false"), withKey("input_layers", {rel_embeddings_name, K}),
      withKey("full_input_range", "true"),
      withKey("weight_initializer", "none")
    };
    layers.push_back(createLayer("shared_fully_connected", k_rel_params));
    attn_input_layers += "," + K_rel;
  }
 
  std::vector<std::string> attn_params = {
    withKey("name", A),
    withKey("num_heads", NUM_HEADS),
    withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
    withKey("max_relative_positions", MAX_RELATIVE_POSITIONS),
    withKey("c2p", c2p?"true":"false"),
    withKey("p2c", p2c?"true":"false"),
    withKey("share_att_key", SHARE_ATT_KEY?"true":"false"),
    withKey("position_buckets", POSITION_BUCKETS),
    withKey("relative_attention", RELATIVE_ATTENTION?"true":"false"),
    withKey("disable_bias", "false"), // Q/K/V bias control
    withKey("input_layers", attn_input_layers)
  };
  layers.push_back(createLayer("deberta_attention", attn_params));

  // Output Projection
  std::vector<std::string> out_params = {
    withKey("name", O), withKey("unit", DIM),
    withKey("disable_bias", "false"), withKey("input_layers", A), withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", out_params));
  
  return layers;
}


void DebertaV2::setupParameters(json &cfg, json &generation_cfg,
                                json &nntr_cfg) {
  Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);
  
  try {
    std::string tokenizer_file = nntr_cfg["tokenizer_file"].get<std::string>();
    std::filesystem::path tokenizer_path(tokenizer_file);
    std::string model_path = tokenizer_path.parent_path().string();
    std::filesystem::path config_path = std::filesystem::path(model_path) / "encoder_config" / "config.json";
    
    if (!std::filesystem::exists(config_path)) {
      config_path = std::filesystem::path(model_path) / "config.json";
    }

    json encoder_cfg = causallm::LoadJsonFile(config_path.string());
    if (encoder_cfg.contains("max_relative_positions")) {
        MAX_RELATIVE_POSITIONS = encoder_cfg["max_relative_positions"].get<int>();
        if (MAX_RELATIVE_POSITIONS == -1) {
            MAX_RELATIVE_POSITIONS = MAX_POSITION_EMBEDDINGS;
        }
    }

    if (encoder_cfg.contains("pos_att_type")) {
      std::vector<std::string> pos_att_type_vec;
      if (encoder_cfg["pos_att_type"].is_array()) {
        pos_att_type_vec = encoder_cfg["pos_att_type"].get<std::vector<std::string>>();
      } else if (encoder_cfg["pos_att_type"].is_string()) {
        std::string pos_att_str = encoder_cfg["pos_att_type"].get<std::string>();

        std::stringstream ss(pos_att_str);
        std::string token;
        
        while (std::getline(ss, token, '|')) {
          token.erase(0, token.find_first_not_of(" \t"));
          token.erase(token.find_last_not_of(" \t") + 1);
          
          if (!token.empty()) {
            pos_att_type_vec.push_back(token);
          }
        }
    }
      
      c2p = std::find(pos_att_type_vec.begin(), pos_att_type_vec.end(), "c2p") != pos_att_type_vec.end();
      p2c = std::find(pos_att_type_vec.begin(), pos_att_type_vec.end(), "p2c") != pos_att_type_vec.end();
    } else {
      c2p = false;
      p2c = false; 
    }

    if (encoder_cfg.contains("share_att_key")) {
      SHARE_ATT_KEY = encoder_cfg["share_att_key"].get<bool>();
    } else {
      SHARE_ATT_KEY = false;
    }

    if (encoder_cfg.contains("relative_attention")) {
      RELATIVE_ATTENTION = encoder_cfg["relative_attention"].get<bool>();
    } else {
      RELATIVE_ATTENTION = true;
    }

    if (encoder_cfg.contains("position_buckets")) {
      POSITION_BUCKETS = encoder_cfg["position_buckets"].get<int>();
    } else {
      POSITION_BUCKETS = -1;
    }

  } catch (const std::exception &e) {
    std::cerr << "\n[!] FATAL ERROR: " << e.what() << "\n";
  }
}

void DebertaV2::registerCustomLayers() {
  Transformer::registerCustomLayers();
  
  const auto &ct_engine = nntrainer::Engine::Global();
  auto app_context = static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(nntrainer::createLayer<causallm::DebertaAttentionLayer>);
    app_context->registerFactory(nntrainer::createLayer<causallm::SharedFullyConnectedLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what() << std::endl;
  }
}

} // namespace causallm
