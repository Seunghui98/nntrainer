// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 NNTrainer Contributors
 *
 * @file   qwen35_causallm.cpp
 * @date   07 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 * @brief  Qwen3.5 Causal Language Model implementation (language_model only).
 *
 * Architecture notes:
 *   - Standard Transformer decoder (RMSNorm pre-norm, residual connections)
 *   - Per-head Q and K normalization via ReshapedRMSNormLayer
 *   - Grouped Query Attention (GQA) with RoPE
 *   - SwiGLU FFN  (gate_proj × silu(up_proj) → down_proj)
 *   - Optionally tied word embeddings (tie_word_embeddings in config.json)
 *
 * Weight loading order (must match weight_converter.py output order):
 *   embed_tokens
 *   for each layer i in [0, num_hidden_layers):
 *     input_layernorm
 *     q_proj  q_norm  k_proj  k_norm  v_proj  o_proj
 *     post_attention_layernorm
 *     mlp.up_proj  mlp.gate_proj  mlp.down_proj
 *   model.norm
 *   lm_head  (only if not tie_word_embeddings)
 */

#include <llm_util.hpp>
#include <model.h>
#include <qwen35_causallm.h>

#include <app_context.h>
#include <engine.h>
#include <reshaped_rms_norm.h>

namespace causallm {

// ---------------------------------------------------------------------------
// Qwen35Transformer::createAttention
// ---------------------------------------------------------------------------
Tensor Qwen35Transformer::createAttention(const int layer_id, int seq_len,
                                          int n_heads, int head_dim,
                                          Tensor query, Tensor key,
                                          Tensor value) {
  using ml::train::createLayer;

  auto Q_name      = "layer" + std::to_string(layer_id) + "_wq";
  auto Q_norm_name = "layer" + std::to_string(layer_id) + "_q_norm";
  auto K_name      = "layer" + std::to_string(layer_id) + "_wk";
  auto K_norm_name = "layer" + std::to_string(layer_id) + "_k_norm";
  auto V_name      = "layer" + std::to_string(layer_id) + "_wv";
  auto A_name      = "layer" + std::to_string(layer_id) + "_attention";
  auto O_name      = "layer" + std::to_string(layer_id) + "_attention_out";

  // V projection: [B,1,T,DIM] → [B,1,T, kv_heads * head_dim]
  LayerHandle v_proj = createLayer(
    "fully_connected",
    {withKey("name", V_name),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones")});
  Tensor v = v_proj(value);

  // K projection + per-head K norm
  LayerHandle k_proj = createLayer(
    "fully_connected",
    {withKey("name", K_name),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones")});
  Tensor k = k_proj(key);

  LayerHandle k_norm = createLayer(
    "reshaped_rms_norm",
    {withKey("name", K_norm_name),
     withKey("packed", "false"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(head_dim))});
  Tensor k_normed = k_norm(k);

  // Q projection + per-head Q norm
  LayerHandle q_proj = createLayer(
    "fully_connected",
    {withKey("name", Q_name),
     withKey("unit", head_dim * n_heads),
     withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones")});
  Tensor q = q_proj(query);

  LayerHandle q_norm = createLayer(
    "reshaped_rms_norm",
    {withKey("name", Q_norm_name),
     withKey("packed", "false"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(head_dim))});
  Tensor q_normed = q_norm(q);

  // MHA core: GQA with RoPE and KV-cache
  LayerHandle attn = createLayer(
    "mha_core",
    {withKey("name", A_name),
     withKey("num_heads", n_heads),
     withKey("num_heads_kv", n_heads / GQA_SIZE),
     withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
     withKey("rope_theta", ROPE_THETA),
     withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
     withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE))});
  Tensor a = attn({q_normed, k_normed, v});

  // O projection: [B,1,T, num_heads * head_dim] → [B,1,T,DIM]
  LayerHandle o_proj = createLayer(
    "fully_connected",
    {withKey("name", O_name),
     withKey("unit", DIM),
     withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones")});
  Tensor o = o_proj(a);

  return o;
}

// ---------------------------------------------------------------------------
// Qwen35Transformer::registerCustomLayers
// ---------------------------------------------------------------------------
void Qwen35Transformer::registerCustomLayers() {
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
}

// ---------------------------------------------------------------------------
// Qwen35CausalLM::registerCustomLayers
// ---------------------------------------------------------------------------
void Qwen35CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  Qwen35Transformer::registerCustomLayers();
}

} // namespace causallm
