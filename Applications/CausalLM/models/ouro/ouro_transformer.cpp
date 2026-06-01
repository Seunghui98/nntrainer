// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   ouro_transformer.cpp
 * @date   1 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Ouro Universal Transformer decoder block construction. The block
 *         layout matches HF Ouro: extra RMSNorm after attention and after
 *         FFN, loop-unrolled `total_ut_steps` times with `shared_from` weight
 *         tying so the parameter set is stored only once.
 */

#include <climits>

#include <llm_util.hpp>
#include <ouro_transformer.h>

namespace causallm {

void OuroTransformer::setupOuroParameters(json &cfg) {
  if (cfg.contains("total_ut_steps") && !cfg["total_ut_steps"].is_null()) {
    TOTAL_UT_STEPS = cfg["total_ut_steps"].get<unsigned int>();
  }
  if (TOTAL_UT_STEPS == 0)
    TOTAL_UT_STEPS = 1;
}

std::string OuroTransformer::ouroNodeName(int layer_id, int current_ut,
                                          const std::string &role) const {
  if (current_ut == 0)
    return "layer" + std::to_string(layer_id) + "_" + role;
  return "layer" + std::to_string(layer_id) + "_ut" + std::to_string(current_ut) +
         "_" + role;
}

std::string OuroTransformer::ouroSharedFromName(int layer_id,
                                                const std::string &role) const {
  return "layer" + std::to_string(layer_id) + "_" + role;
}

Tensor OuroTransformer::createTransformerDecoderBlock(const int layer_id,
                                                      Tensor input) {
  return createOuroDecoderBlock(layer_id, /*current_ut=*/0, input);
}

std::pair<Tensor, Tensor>
OuroTransformer::createOuroKVCachePlaceholders(int layer_id, int current_ut,
                                               int n_heads) {
  const int flat_id = current_ut * NUM_LAYERS + layer_id;
  const unsigned int max_timestep = static_cast<unsigned int>(MAX_SEQ_LEN);
  const unsigned int kv_width =
    static_cast<unsigned int>(HEAD_DIM * n_heads / GQA_SIZE);

#ifdef ENABLE_FP16
  ml::train::TensorDim cache_dim(
    {BATCH_SIZE, 1, max_timestep, kv_width},
    {ml::train::TensorDim::Format::NCHW, ml::train::TensorDim::DataType::FP16});

  Tensor cache_k(cache_dim, "cache_k_l" + std::to_string(flat_id));
  Tensor cache_v(cache_dim, "cache_v_l" + std::to_string(flat_id));
  return {cache_k, cache_v};
#else
  const std::string cache_shape = std::to_string(BATCH_SIZE) +
                                  ":1:" + std::to_string(max_timestep) + ":" +
                                  std::to_string(kv_width);

  LayerHandle cache_k_input(createLayer(
    "input",
    {withKey("name", "cache_k_l" + std::to_string(flat_id)),
     withKey("input_shape", cache_shape), withKey("input_dtype", "UINT16")}));
  LayerHandle cache_v_input(createLayer(
    "input",
    {withKey("name", "cache_v_l" + std::to_string(flat_id)),
     withKey("input_shape", cache_shape), withKey("input_dtype", "UINT16")}));

  return {cache_k_input(Tensor()), cache_v_input(Tensor())};
#endif
}

Tensor OuroTransformer::createOuroAttention(int layer_id, int current_ut,
                                            int seq_len, int n_heads,
                                            int head_dim, Tensor query,
                                            Tensor key, Tensor value) {
  auto make_proj = [&](const std::string &role, unsigned int unit) {
    std::vector<std::string> props = {
      withKey("name", ouroNodeName(layer_id, current_ut, role)),
      withKey("unit", unit),
      withKey("disable_bias", "true"),
      withKey("weight_initializer", "ones"),
    };
    if (current_ut > 0) {
      props.push_back(
        withKey("shared_from", ouroSharedFromName(layer_id, role)));
    }
    return LayerHandle(createLayer("fully_connected", props));
  };

  Tensor q = make_proj("wq", head_dim * n_heads)(query);
  Tensor k = make_proj("wk", head_dim * n_heads / GQA_SIZE)(key);
  Tensor v = make_proj("wv", head_dim * n_heads / GQA_SIZE)(value);

  auto [cache_k, cache_v] =
    createOuroKVCachePlaceholders(layer_id, current_ut, n_heads);

  LayerHandle mha(createLayer(
    "mha_core",
    {withKey("name", ouroNodeName(layer_id, current_ut, "attention")),
     withKey("num_heads", n_heads), withKey("num_heads_kv", n_heads / GQA_SIZE),
     withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
     withKey("sliding_window", (layer_id + 1) % SLIDING_WINDOW_PATTERN
                                 ? SLIDING_WINDOW
                                 : UINT_MAX),
     withKey("rope_theta", ROPE_THETA),
     withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
     withKey("is_causal", IS_CAUSAL ? "true" : "false")}));
  Tensor a = mha({q, k, v, cache_k, cache_v});

  LayerHandle wo = make_proj("attention_out", DIM);
  return wo(a);
}

Tensor OuroTransformer::createOuroMlp(int layer_id, int current_ut, int dim,
                                      int hidden_dim, Tensor input) {
  auto make_proj = [&](const std::string &role, unsigned int unit) {
    std::vector<std::string> props = {
      withKey("name", ouroNodeName(layer_id, current_ut, role)),
      withKey("unit", unit),
      withKey("disable_bias", "true"),
      withKey("weight_initializer", "ones"),
    };
    if (current_ut > 0) {
      props.push_back(
        withKey("shared_from", ouroSharedFromName(layer_id, role)));
    }
    return LayerHandle(createLayer("fully_connected", props));
  };

  Tensor up = make_proj("ffn_up", hidden_dim)(input);
  Tensor gate = make_proj("ffn_gate", hidden_dim)(input);

  LayerHandle swiglu(createLayer(
    "swiglu",
    {withKey("name", ouroNodeName(layer_id, current_ut, "ffn_swiglu"))}));
  Tensor act = swiglu({gate, up});

  return make_proj("ffn_down", dim)(act);
}

Tensor OuroTransformer::createOuroDecoderBlock(int layer_id, int current_ut,
                                               Tensor input) {
  auto make_norm = [&](const std::string &role) {
    std::vector<std::string> props = {
      withKey("name", ouroNodeName(layer_id, current_ut, role)),
      withKey("epsilon", std::to_string(NORM_EPS)),
      withKey("packed", "false"),
    };
    if (current_ut > 0) {
      props.push_back(
        withKey("shared_from", ouroSharedFromName(layer_id, role)));
    }
    return LayerHandle(createLayer("rms_norm", props));
  };

  // Attention pre-norm
  LayerHandle attn_norm = make_norm("attention_norm");
  Tensor normed = attn_norm(input);

  // Attention sub-graph
  Tensor att_out = createOuroAttention(layer_id, current_ut, INIT_SEQ_LEN,
                                       NUM_HEADS, HEAD_DIM, normed, normed,
                                       normed);

  // Extra RMSNorm after attention (Ouro-specific)
  LayerHandle attn_norm_2 = make_norm("attention_norm_2");
  att_out = attn_norm_2(att_out);

  // Residual after attention
  LayerHandle decoder_add(createLayer(
    "addition",
    {withKey("name", ouroNodeName(layer_id, current_ut, "decoder_add"))}));
  Tensor residual = decoder_add({input, att_out});

  // FFN pre-norm
  LayerHandle ffn_norm = make_norm("ffn_norm");
  Tensor ffn_normed = ffn_norm(residual);

  // FFN sub-graph
  Tensor ffn_out =
    createOuroMlp(layer_id, current_ut, DIM, INTERMEDIATE_SIZE, ffn_normed);

  // Extra RMSNorm after FFN (Ouro-specific)
  LayerHandle ffn_norm_2 = make_norm("ffn_norm_2");
  ffn_out = ffn_norm_2(ffn_out);

  // Residual after FFN
  LayerHandle decoder_output(createLayer(
    "addition",
    {withKey("name", ouroNodeName(layer_id, current_ut, "decoder_output"))}));
  return decoder_output({residual, ffn_out});
}

Tensor OuroTransformer::applyOuroOutputNorm(int current_ut, Tensor input) {
  const std::string base = "output_norm";
  std::vector<std::string> props = {
    withKey("name", current_ut == 0
                      ? base
                      : base + "_ut" + std::to_string(current_ut)),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("packed", "false"),
  };
  if (current_ut > 0)
    props.push_back(withKey("shared_from", base));

  LayerHandle out_norm(createLayer("rms_norm", props));
  return out_norm(input);
}

} // namespace causallm
