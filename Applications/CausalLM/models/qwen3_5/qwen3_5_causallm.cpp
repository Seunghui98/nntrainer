// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   qwen3_5_causallm.cpp
 * @date   7 April 2026
 * @brief  Qwen3.5 hybrid model implementation
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <llm_util.hpp>
#include <model.h>
#include <qwen3_5_causallm.h>

#include <app_context.h>
#include <engine.h>
#include <attn_gate.h>
#include <gated_delta_net.h>
#include <reshaped_rms_norm.h>

namespace causallm {

std::vector<LayerHandle> Qwen3_5Transformer::createMlp(const int layer_id,
                                                       int dim,
                                                       int hidden_dim,
                                                       std::string input_name) {
  std::vector<LayerHandle> layers;

  // gate_proj
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
     withKey("unit", hidden_dim),
     withKey("disable_bias", "true"),
     withKey("input_layers", input_name),
     withKey("weight_initializer", "ones")}));

  // up_proj
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
     withKey("unit", hidden_dim),
     withKey("disable_bias", "true"),
     withKey("input_layers", input_name),
     withKey("weight_initializer", "ones")}));

  // swiglu = silu(in0) * in1, input order: gate, up
  layers.push_back(createLayer(
    "swiglu",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_ffn_gate,"
             "layer" + std::to_string(layer_id) + "_ffn_up")}));

  // down_proj
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", dim),
     withKey("disable_bias", "true"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
     withKey("weight_initializer", "ones")}));

  return layers;
}

void Qwen3_5Transformer::setupQwen35Parameters(json &cfg, json &nntr_cfg) {
  unsigned int num_layers = cfg.contains("num_hidden_layers")
                              ? cfg["num_hidden_layers"].get<unsigned int>()
                              : NUM_LAYERS;

  is_self_attn_layer.resize(num_layers, false);

  if (nntr_cfg.contains("self_attn_layers")) {
    auto layers = nntr_cfg["self_attn_layers"].get<std::vector<unsigned int>>();
    for (auto idx : layers) {
      if (idx < num_layers)
        is_self_attn_layer[idx] = true;
    }
  } else {
    unsigned int pattern = nntr_cfg.contains("self_attn_pattern")
                             ? nntr_cfg["self_attn_pattern"].get<unsigned int>()
                             : 4;
    for (unsigned int i = pattern - 1; i < num_layers; i += pattern) {
      is_self_attn_layer[i] = true;
    }
  }

  if (cfg.contains("rope_parameters") &&
      cfg["rope_parameters"].contains("rope_theta")) {
    ROPE_THETA = cfg["rope_parameters"]["rope_theta"].get<unsigned int>();
  }

  LINEAR_NUM_V_HEADS = nntr_cfg.contains("linear_num_v_heads")
                         ? nntr_cfg["linear_num_v_heads"].get<unsigned int>()
                         : 16;
  LINEAR_HEAD_K_DIM = nntr_cfg.contains("linear_head_k_dim")
                        ? nntr_cfg["linear_head_k_dim"].get<unsigned int>()
                        : 128;
  LINEAR_HEAD_V_DIM = nntr_cfg.contains("linear_head_v_dim")
                        ? nntr_cfg["linear_head_v_dim"].get<unsigned int>()
                        : 128;
  LINEAR_CONV_KERNEL = nntr_cfg.contains("linear_conv_kernel")
                         ? nntr_cfg["linear_conv_kernel"].get<unsigned int>()
                         : 4;

  SELF_ATTN_HEAD_DIM =
    cfg.contains("head_dim") ? cfg["head_dim"].get<unsigned int>() : HEAD_DIM;
}

void Qwen3_5Transformer::constructModel() {
  // Use addLayer API (not Tensor API) to match support_qwen3_5_1 branch.
  // The topological sort produces different layer ordering between the two APIs,
  // which causes weight loading misalignment.
  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  std::vector<std::string> model_props = {
    withKey("batch_size", BATCH_SIZE), withKey("epochs", "1"),
    withKey("model_tensor_type", MODEL_TENSOR_TYPE)};
  if (MEMORY_SWAP) {
    model_props.emplace_back(withKey("fsu", "true"));
    model_props.emplace_back(withKey("fsu_lookahead", FSU_LOOKAHEAD));
  }
  model->setProperty(model_props);

  std::vector<LayerHandle> layers;

  // Input layer
  layers.push_back(createLayer(
    "input", {withKey("name", "input0"),
              withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))}));

  // Embedding layer
  const std::string embedding_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";
  layers.push_back(createLayer(
    embedding_type,
    {"name=embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
     "weight_dtype=" + EMBEDDING_DTYPE, "out_dim=" + std::to_string(DIM),
     "scale=" + std::to_string(EMBEDDING_SCALE)}));

  // Build decoder layers
  for (int i = 0; i < NUM_LAYERS; ++i) {
    std::string input_name =
      (i == 0) ? "embedding0"
               : ("layer" + std::to_string(i - 1) + "_decoder_output");

    std::vector<LayerHandle> block;
    if (is_self_attn_layer[i]) {
      block = createTransformerDecoderBlock(i, input_name);
    } else {
      block = createLinearAttentionBlock(i, input_name);
    }
    layers.insert(layers.end(), block.begin(), block.end());
  }

  // Final RMS norm
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "output_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("input_layers",
             "layer" + std::to_string(NUM_LAYERS - 1) + "_decoder_output"),
     withKey("packed", "false")}));

  // Add all layers to model
  for (auto &layer : layers) {
    model->addLayer(layer);
  }
}

std::vector<LayerHandle> Qwen3_5Transformer::createTransformerDecoderBlock(
  const int layer_id, std::string input_name) {

  std::vector<LayerHandle> layers;

  // Attention norm
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("input_layers", input_name),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  auto att_layer =
    createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                    "layer" + std::to_string(layer_id) + "_attention_norm",
                    "layer" + std::to_string(layer_id) + "_attention_norm",
                    "layer" + std::to_string(layer_id) + "_attention_norm");
  layers.insert(layers.end(), att_layer.begin(), att_layer.end());

  // Residual add
  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_add"),
     withKey("input_layers", input_name + ",layer" + std::to_string(layer_id) +
                               "_attention_out")}));

  // FFN norm
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_decoder_add"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  // MLP
  auto ffn_layer = createMlp(layer_id, DIM, INTERMEDIATE_SIZE,
                             "layer" + std::to_string(layer_id) + "_ffn_norm");
  layers.insert(layers.end(), ffn_layer.begin(), ffn_layer.end());

  // Residual add
  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output"),
     withKey("input_layers", "layer" + std::to_string(layer_id) +
                               "_decoder_add,layer" + std::to_string(layer_id) +
                               "_ffn_down")}));

  return layers;
}

std::vector<LayerHandle> Qwen3_5Transformer::createAttention(
  const int layer_id, int seq_len, int n_heads, int head_dim,
  std::string query_name, std::string key_name, std::string value_name) {

  std::vector<LayerHandle> layers;
  auto prefix = "layer" + std::to_string(layer_id);
  auto V = prefix + "_wv";
  auto K = prefix + "_wk";
  auto K_norm = prefix + "_k_norm";
  auto Q = prefix + "_wq";
  auto Q_gate = prefix + "_wq_gate";
  auto Q_norm = prefix + "_q_norm";
  auto A = prefix + "_attention";
  auto AG = prefix + "_attn_gate";
  auto O = prefix + "_attention_out";

  unsigned int kv_dim = SELF_ATTN_HEAD_DIM * n_heads / GQA_SIZE;
  unsigned int q_dim = SELF_ATTN_HEAD_DIM * n_heads;

  // V projection
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", V), withKey("unit", kv_dim),
     withKey("disable_bias", "true"), withKey("input_layers", value_name),
     withKey("weight_initializer", "ones")}));

  // K projection
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", K), withKey("unit", kv_dim),
     withKey("disable_bias", "true"), withKey("input_layers", key_name),
     withKey("weight_initializer", "ones")}));

  // K-reshaped-norm
  layers.push_back(createLayer(
    "reshaped_rms_norm",
    {withKey("name", K_norm), withKey("input_layers", K),
     withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(SELF_ATTN_HEAD_DIM))}));

  // Q projection (query part)
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", Q), withKey("unit", q_dim),
     withKey("disable_bias", "true"), withKey("input_layers", query_name),
     withKey("weight_initializer", "ones")}));

  // Q_gate projection (gate part)
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", Q_gate), withKey("unit", q_dim),
     withKey("disable_bias", "true"), withKey("input_layers", query_name),
     withKey("weight_initializer", "ones")}));

  // Q-reshaped-norm
  layers.push_back(createLayer(
    "reshaped_rms_norm",
    {withKey("name", Q_norm), withKey("input_layers", Q),
     withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(SELF_ATTN_HEAD_DIM))}));

  // Attention core (partial_rotary_factor=0.25 for Qwen3.5)
  layers.push_back(createLayer(
    "mha_core",
    {withKey("name", A), withKey("num_heads", n_heads),
     withKey("num_heads_kv", n_heads / GQA_SIZE),
     withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
     withKey("sliding_window", SLIDING_WINDOW),
     withKey("rope_theta", ROPE_THETA),
     withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
     withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
     withKey("partial_rotary_factor", 0.25f),
     withKey("input_layers", Q_norm + "," + K_norm + "," + V)}));

  // Attention gate: output = attn_output * sigmoid(gate)
  layers.push_back(createLayer("attn_gate",
                               {withKey("name", AG),
                                withKey("input_layers", A + "," + Q_gate)}));

  // O projection
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", O), withKey("unit", DIM),
     withKey("disable_bias", "true"), withKey("input_layers", AG),
     withKey("weight_initializer", "ones")}));

  return layers;
}

std::vector<LayerHandle>
Qwen3_5Transformer::createLinearAttentionBlock(const int layer_id,
                                               std::string input_name) {
  std::vector<LayerHandle> layers;

  // Input layernorm
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("input_layers", input_name),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  // Gated DeltaNet layer (linear attention)
  layers.push_back(createLayer(
    "gated_delta_net",
    {withKey("name", "layer" + std::to_string(layer_id) + "_linear_attn"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("num_v_heads", LINEAR_NUM_V_HEADS),
     withKey("head_k_dim", LINEAR_HEAD_K_DIM),
     withKey("head_v_dim", LINEAR_HEAD_V_DIM),
     withKey("conv_kernel", LINEAR_CONV_KERNEL)}));

  // Residual add
  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_add"),
     withKey("input_layers",
             input_name + ",layer" + std::to_string(layer_id) +
               "_linear_attn")}));

  // FFN norm
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_decoder_add"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  // MLP
  auto ffn_layer = createMlp(layer_id, DIM, INTERMEDIATE_SIZE,
                             "layer" + std::to_string(layer_id) + "_ffn_norm");
  layers.insert(layers.end(), ffn_layer.begin(), ffn_layer.end());

  // Residual add
  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output"),
     withKey("input_layers", "layer" + std::to_string(layer_id) +
                               "_decoder_add,layer" + std::to_string(layer_id) +
                               "_ffn_down")}));

  return layers;
}

void Qwen3_5Transformer::registerCustomLayers() {
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::GatedDeltaNetLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::AttnGateLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

void Qwen3_5CausalLM::constructModel() {
  // Build hybrid model using addLayer API
  Qwen3_5Transformer::constructModel();

  // Add lm_head
  const std::string lmhead_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "fully_connected";
  std::vector<std::string> lmhead_prop = {
    withKey("name", "output_of_causallm"),
    withKey("unit", NUM_VOCAB),
    withKey("disable_bias", "true"),
    withKey("input_layers", "output_norm"),
    withKey("weight_dtype", LMHEAD_DTYPE),
  };
  if (TIE_WORD_EMBEDDINGS)
    lmhead_prop.emplace_back(withKey("shared_from", "embedding0"));
  model->addLayer(createLayer(lmhead_type, lmhead_prop));
}

void Qwen3_5CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  Qwen3_5Transformer::registerCustomLayers();
}

} // namespace causallm
