// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   gemma4_assistant_causallm.cpp
 * @date   29 Oct 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Gemma4AssistantForCausalLM, the MTP drafter of Gemma 4, on the CPU
 *         (gemma4_assistant_causallm.h)
 */

#include <app_context.h>
#include <assistant_attention.h>
#include <engine.h>
#include <gemma4_assistant_causallm.h>
#include <layer_node.h>
#include <llm_util.hpp>
#include <model.h>
#include <nntrainer_error.h>

namespace causallm {

Gemma4AssistantCausalLM::Gemma4AssistantCausalLM(json &cfg,
                                                 json &generation_cfg,
                                                 json &nntr_cfg) :
  Transformer(sanitizeConfig(cfg),
              sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg,
              ModelType::CAUSALLM),
  Gemma4CausalLM(cfg, generation_cfg, nntr_cfg) {
  NNTR_THROW_IF(!cfg.contains("backbone_hidden_size"), std::invalid_argument)
    << "[Gemma4Assistant] config.json has no backbone_hidden_size";
  NNTR_THROW_IF(cfg.value("use_ordered_embeddings", false),
                std::invalid_argument)
    << "[Gemma4Assistant] use_ordered_embeddings (the E2B/E4B centroid head) "
       "is not supported; the 26B-A4B and 31B assistants use a dense head";
  BACKBONE_DIM = cfg["backbone_hidden_size"].get<unsigned int>();
  LMHEAD_UNTIE = true; // no embedding0: the head owns the table
}

Tensor Gemma4AssistantCausalLM::createAttention(const int layer_id, int seq_len,
                                                int n_heads, int head_dim,
                                                Tensor query, Tensor key,
                                                Tensor value) {
  const std::string pfx = "layer" + std::to_string(layer_id);
  const bool sliding = isSlidingAttentionLayer(layer_id);
  const int hd = static_cast<int>(getAttentionHeadDim(layer_id));

  LayerHandle wq(createLayer(
    "fully_connected",
    {withKey("name", pfx + "_wq"), withKey("unit", hd * n_heads),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  LayerHandle q_norm(
    createLayer("reshaped_rms_norm",
                {withKey("name", pfx + "_q_norm"), withKey("packed", "false"),
                 withKey("epsilon", std::to_string(NORM_EPS)),
                 withKey("feature_size", std::to_string(hd))}));
  const std::string &rope_type =
    sliding ? SLIDING_ATTENTION_ROPE_TYPE : FULL_ATTENTION_ROPE_TYPE;
  const float partial =
    rope_type != "proportional"
      ? 1.0f
      : (sliding ? SLIDING_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR
                 : FULL_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR);
  LayerHandle attn(createLayer(
    "assistant_attention",
    {withKey("name", pfx + "_attention"), withKey("num_heads", n_heads),
     withKey("num_heads_kv", getKVHeadCount(layer_id)),
     withKey("rope_theta", sliding ? SLIDING_ATTENTION_ROPE_THETA
                                   : FULL_ATTENTION_ROPE_THETA),
     withKey("rope_partial_rotary_factor", std::to_string(partial))}));
  LayerHandle wo(createLayer(
    "fully_connected",
    {withKey("name", pfx + "_attention_out"), withKey("unit", DIM),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  return wo(attn(q_norm(wq(query))));
}

std::pair<Tensor, Tensor> Gemma4AssistantCausalLM::constructModel() {
  Tensor x({1, 1, 1, 2 * BACKBONE_DIM}, "input0");
  LayerHandle pre(createLayer(
    "fully_connected",
    {withKey("name", "pre_projection"), withKey("unit", DIM),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  Tensor h = pre(x);
  for (int i = 0; i < NUM_LAYERS; ++i)
    h = createTransformerDecoderBlock(i, h);
  LayerHandle out_norm(
    createLayer("rms_norm", {withKey("name", "output_norm"),
                             withKey("epsilon", std::to_string(NORM_EPS)),
                             withKey("packed", "false")}));
  h = out_norm(h);

  Tensor logits = buildLmHeadOutput(h, false);
  LayerHandle post(createLayer(
    "fully_connected",
    {withKey("name", "post_projection"), withKey("unit", BACKBONE_DIM),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  LayerHandle cat(createLayer(
    "concat", {withKey("name", "assistant_output"), withKey("axis", 3)}));
  return {x, cat({logits, post(h)})};
}

void Gemma4AssistantCausalLM::registerCustomLayers() {
  Gemma4CausalLM::registerCustomLayers();
  nntrainer::Engine::Global().registerLayerFactory(
    "cpu", nntrainer::createLayer<AssistantAttentionLayer>);
}

void Gemma4AssistantCausalLM::setSharedKV(
  const float *k_sliding, const float *v_sliding, unsigned int n_sliding,
  const float *k_full, const float *v_full, unsigned int n_full,
  unsigned int pos) {
  auto fn = [&](ml::train::Layer &l, nntrainer::RunLayerContext &, void *) {
    auto *node = dynamic_cast<nntrainer::LayerNode *>(&l);
    auto *a = node ? dynamic_cast<AssistantAttentionLayer *>(node->getLayer())
                   : nullptr;
    if (a == nullptr)
      return;
    const std::string &name = node->getName(); // "layer<i>_attention"
    const int id = std::stoi(name.substr(5, name.find('_') - 5));
    if (isSlidingAttentionLayer(id))
      a->setSharedKV(k_sliding, v_sliding, n_sliding, pos);
    else
      a->setSharedKV(k_full, v_full, n_full, pos);
  };
  model->forEachLayer(fn, nullptr);
}

const float *Gemma4AssistantCausalLM::draftStep(const float *in) {
  return model->incremental_inference(1, {const_cast<float *>(in)}, {}, 1, 0,
                                      1)[0];
}

} // namespace causallm
