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

Tensor Qwen3_5Transformer::createMlp(const int layer_id, int dim,
                                      int hidden_dim, Tensor input) {
  using ml::train::createLayer;

  // Weight converter saves: up_proj, gate_proj, down_proj
  // Layer creation order must match converter save order for weight loading.

  // up_proj (created first to match weight converter order)
  LayerHandle ffn_up = createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones")});
  Tensor up = ffn_up(input);

  // gate_proj (created second to match weight converter order)
  LayerHandle ffn_gate = createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones")});
  Tensor gate = ffn_gate(input);

  // swiglu = silu(in0) * in1, so input order is: gate, up
  LayerHandle swiglu = createLayer(
    "swiglu",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_swiglu")});
  Tensor activated = swiglu({gate, up});

  // down_proj
  LayerHandle ffn_down = createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones")});
  Tensor down = ffn_down(activated);

  return down;
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
  using ml::train::createLayer;

  // Input layer
  LayerHandle input_layer = createLayer(
    "input", {withKey("name", "input0"),
              withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))});
  Tensor input = input_layer(Tensor());
  std::vector<Tensor> all_inputs = {input};

  // Embedding layer
  const std::string embedding_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";
  LayerHandle embedding = createLayer(
    embedding_type,
    {"name=embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
     "weight_dtype=" + EMBEDDING_DTYPE, "out_dim=" + std::to_string(DIM),
     "scale=" + std::to_string(EMBEDDING_SCALE)});
  Tensor x = embedding(input);

  // Build decoder layers
  for (int i = 0; i < NUM_LAYERS; ++i) {
    if (is_self_attn_layer[i]) {
      x = Transformer::createTransformerDecoderBlock(i, x);
    } else {
      x = createLinearAttentionBlock(i, x);
    }
  }

  // Final RMS norm
  LayerHandle output_norm = createLayer(
    "rms_norm", {withKey("name", "output_norm"),
                 withKey("epsilon", std::to_string(NORM_EPS)),
                 withKey("packed", "false")});
  x = output_norm(x);

  // Compile
  std::vector<Tensor> outputs = {x};
  model->compile(all_inputs, outputs, ml::train::ExecutionMode::INFERENCE);
}

Tensor Qwen3_5Transformer::createAttention(const int layer_id, int seq_len,
                                            int n_heads, int head_dim,
                                            Tensor query, Tensor key,
                                            Tensor value) {
  using ml::train::createLayer;

  auto prefix = "layer" + std::to_string(layer_id);

  unsigned int kv_dim = SELF_ATTN_HEAD_DIM * n_heads / GQA_SIZE;
  unsigned int q_dim = SELF_ATTN_HEAD_DIM * n_heads;

  // Weight converter saves: q_gate, q, q_norm, k, k_norm, v, o
  // Layer creation order MUST match for correct weight loading.

  // 1) Q_gate projection (gate part of q_proj, de-interleaved)
  LayerHandle q_gate_proj = createLayer(
    "fully_connected",
    {withKey("name", prefix + "_wq_gate"), withKey("unit", q_dim),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones")});
  Tensor q_gate = q_gate_proj(query);

  // 2) Q projection (query part of q_proj, de-interleaved)
  LayerHandle q_proj = createLayer(
    "fully_connected",
    {withKey("name", prefix + "_wq"), withKey("unit", q_dim),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones")});
  Tensor q = q_proj(query);

  // 3) Q-reshaped-norm (only applied to query, not gate)
  LayerHandle q_norm = createLayer(
    "reshaped_rms_norm",
    {withKey("name", prefix + "_q_norm"), withKey("packed", "false"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(SELF_ATTN_HEAD_DIM))});
  Tensor q_normed = q_norm(q);

  // 4) K projection
  LayerHandle k_proj = createLayer(
    "fully_connected",
    {withKey("name", prefix + "_wk"), withKey("unit", kv_dim),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones")});
  Tensor k = k_proj(key);

  // 5) K-reshaped-norm
  LayerHandle k_norm = createLayer(
    "reshaped_rms_norm",
    {withKey("name", prefix + "_k_norm"), withKey("packed", "false"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(SELF_ATTN_HEAD_DIM))});
  Tensor k_normed = k_norm(k);

  // 6) V projection
  LayerHandle v_proj = createLayer(
    "fully_connected",
    {withKey("name", prefix + "_wv"), withKey("unit", kv_dim),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones")});
  Tensor v = v_proj(value);

  // 7) Attention core (with partial_rotary_factor=0.25 for Qwen3.5)
  LayerHandle attn = createLayer(
    "mha_core",
    {withKey("name", prefix + "_attention"), withKey("num_heads", n_heads),
     withKey("num_heads_kv", n_heads / GQA_SIZE),
     withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
     withKey("sliding_window", SLIDING_WINDOW),
     withKey("rope_theta", ROPE_THETA),
     withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
     withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
     withKey("partial_rotary_factor", 0.25f)});
  Tensor a = attn({q_normed, k_normed, v});

  // Attention gate: output = attn_output * sigmoid(gate)
  LayerHandle attn_gate =
    createLayer("attn_gate", {withKey("name", prefix + "_attn_gate")});
  Tensor gated = attn_gate({a, q_gate});

  // O projection
  LayerHandle o_proj = createLayer(
    "fully_connected",
    {withKey("name", prefix + "_attention_out"), withKey("unit", DIM),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones")});
  Tensor o = o_proj(gated);

  return o;
}

Tensor Qwen3_5Transformer::createLinearAttentionBlock(const int layer_id,
                                                      Tensor input) {
  using ml::train::createLayer;

  auto prefix = "layer" + std::to_string(layer_id);

  // Input layernorm
  LayerHandle att_norm = createLayer(
    "rms_norm", {withKey("name", prefix + "_attention_norm"),
                 withKey("epsilon", std::to_string(NORM_EPS)),
                 withKey("packed", "false")});
  Tensor normed = att_norm(input);

  // Gated DeltaNet layer (linear attention)
  LayerHandle gdn = createLayer(
    "gated_delta_net",
    {withKey("name", prefix + "_linear_attn"),
     withKey("num_v_heads", LINEAR_NUM_V_HEADS),
     withKey("head_k_dim", LINEAR_HEAD_K_DIM),
     withKey("head_v_dim", LINEAR_HEAD_V_DIM),
     withKey("conv_kernel", LINEAR_CONV_KERNEL)});
  Tensor gdn_out = gdn(normed);

  // Residual add
  Tensor residual = input.add(gdn_out);

  // FFN norm
  LayerHandle ffn_norm = createLayer(
    "rms_norm", {withKey("name", prefix + "_ffn_norm"),
                 withKey("epsilon", std::to_string(NORM_EPS)),
                 withKey("packed", "false")});
  Tensor ffn_normed = ffn_norm(residual);

  // MLP
  Tensor ffn_out = createMlp(layer_id, DIM, INTERMEDIATE_SIZE, ffn_normed);

  // Residual add
  Tensor decoder_out = residual.add(ffn_out);

  return decoder_out;
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
  // Build hybrid model
  Qwen3_5Transformer::constructModel();

  // Add lm_head
  const std::string lmhead_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "lm_head";
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
