// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   ouro_causallm.cpp
 * @date   1 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file defines OuroCausalLM's basic actions
 */

#include <app_context.h>
#include <common.h>
#include <layer_context.h>
#include <lm_head.h>
#include <mha_core.h>
#include <nntrainer_error.h>
#include <tensor.h>

#include <llm_util.hpp>
#include <ouro_causallm.h>

namespace causallm {

std::pair<Tensor, Tensor> OuroCausalLM::constructModel() {

  Tensor x =
    Tensor({1, 1, 1, static_cast<unsigned int>(INIT_SEQ_LEN)}, "input0");

  // Ouro: embed_tokens (vocab -> intermediate_size) + embed_projection
  // (intermediate_size -> hidden_size).
  const unsigned int EMBED_PROJ_DIM = INTERMEDIATE_SIZE;

  const std::string embedding_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";

  LayerHandle embedding(createLayer(
    embedding_type,
    {"name=embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
     "weight_dtype=" + EMBEDDING_DTYPE,
     "out_dim=" + std::to_string(EMBED_PROJ_DIM),
     "scale=" + std::to_string(EMBEDDING_SCALE)}));
  Tensor h = embedding(x);

  LayerHandle embed_proj(createLayer(
    "fully_connected",
    {withKey("name", "embed_projection"), withKey("unit", DIM),
     withKey("disable_bias", "true"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  h = embed_proj(h);

  // Universal-Transformer unroll. Weights of UT step k>0 are tied to step 0
  // via shared_from inside createTransformerDecoderBlock / applyOuroOutputNorm,
  // so only one set of parameters is stored. current_ut_ steers the leaf-name
  // / shared_from logic in OuroTransformer overrides.
  for (int ut = 0; ut < static_cast<int>(TOTAL_UT_STEPS); ++ut) {
    current_ut_ = ut;
    for (int i = 0; i < NUM_LAYERS; ++i) {
      h = createTransformerDecoderBlock(i, h);
    }
    h = applyOuroOutputNorm(ut, h);
  }
  current_ut_ = 0;

  const std::string lmhead_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "lm_head";

  std::vector<std::string> lmhead_prop = {
    withKey("name", "output_of_causallm"),
    withKey("unit", NUM_VOCAB),
    withKey("disable_bias", "true"),
    withKey("weight_dtype", LMHEAD_DTYPE),
  };

  if (TIE_WORD_EMBEDDINGS) {
    lmhead_prop.emplace_back(withKey("shared_from", "embedding0"));
  }

  LayerHandle lmhead(createLayer(lmhead_type, lmhead_prop));
  h = lmhead(h);

  return {x, h};
}

void OuroCausalLM::allocateAndBindKVCache() {
  const unsigned int total_slots =
    static_cast<unsigned int>(TOTAL_UT_STEPS) *
    static_cast<unsigned int>(NUM_LAYERS);

  if (!kv_cache.isAllocated()) {
#ifdef ENABLE_FP16
    const auto cache_dtype = ml::train::TensorDim::DataType::FP16;
#else
    const auto cache_dtype = ml::train::TensorDim::DataType::UINT16;
#endif
    const unsigned int max_timestep = static_cast<unsigned int>(MAX_SEQ_LEN);
    kv_cache.allocate(total_slots, BATCH_SIZE, max_timestep,
                      static_cast<unsigned int>(NUM_KEY_VALUE_HEADS),
                      static_cast<unsigned int>(HEAD_DIM), cache_dtype);
    kv_cache_bound = false;
  }

  if (kv_cache_bound)
    return;

  auto find_cache_placeholder = [this](const std::string &base_name) {
    for (const auto &suffix : {":0", ":input0", ":out0", ""}) {
      auto *tensor = model->getTensor(base_name + suffix);
      if (tensor != nullptr)
        return tensor;
    }
    return static_cast<nntrainer::Tensor *>(nullptr);
  };

  for (int ut = 0; ut < static_cast<int>(TOTAL_UT_STEPS); ++ut) {
    for (int i = 0; i < NUM_LAYERS; ++i) {
      const int flat = ut * NUM_LAYERS + i;
      const std::string attn_name =
        (ut == 0)
          ? ("layer" + std::to_string(i) + "_attention")
          : ("layer" + std::to_string(i) + "_ut" + std::to_string(ut) +
             "_attention");

      auto &kc = kv_cache.getKeyCache(flat);
      auto &vc = kv_cache.getValueCache(flat);

      auto *kp = model->getTensor(attn_name + ":input3");
      auto *vp = model->getTensor(attn_name + ":input4");
      if (kp == nullptr)
        kp = find_cache_placeholder("cache_k_l" + std::to_string(flat));
      if (vp == nullptr)
        vp = find_cache_placeholder("cache_v_l" + std::to_string(flat));

      NNTR_THROW_IF(kp == nullptr || vp == nullptr, std::runtime_error)
        << "OuroCausalLM::allocateAndBindKVCache: cache_k_l" << flat
        << " / cache_v_l" << flat << " input placeholder not found (attn="
        << attn_name << ")";
      NNTR_THROW_IF(kp->getDataType() != kc.getDataType() ||
                      vp->getDataType() != vc.getDataType(),
                    std::runtime_error)
        << "OuroCausalLM::allocateAndBindKVCache: dtype mismatch at slot "
        << flat;

      kp->setData(kc.getMemoryData(), kc.getOffset(), false);
      vp->setData(vc.getMemoryData(), vc.getOffset(), false);
    }
  }

  kv_cache_bound = true;
}

} // namespace causallm
