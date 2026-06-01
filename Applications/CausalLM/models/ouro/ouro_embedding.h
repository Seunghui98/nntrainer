// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   ouro_embedding.h
 * @date   1 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   OuroEmbedding builds the Ouro graph for sentence-embedding usage:
 *           1. Token embedding (vocab -> intermediate_size).
 *           2. embed_projection (intermediate_size -> hidden_size).
 *           3. UT loop with shared_from weight tying.
 *           4. Pooling + L2 Normalize.
 *         The KV cache contains TOTAL_UT_STEPS * NUM_LAYERS slots; the
 *         SentenceTransformer base binds them through kvCacheSlotCount()
 *         and attentionNodeName().
 */

#ifndef __OURO_EMBEDDING_H__
#define __OURO_EMBEDDING_H__

#include <ouro_transformer.h>
#include <sentence_transformer.h>

namespace causallm {

/**
 * @brief OuroEmbedding class
 */
class OuroEmbedding : public SentenceTransformer, public OuroTransformer {

public:
  static constexpr const char *architectures = "OuroModel";

  OuroEmbedding(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::EMBEDDING),
    SentenceTransformer(cfg, generation_cfg, nntr_cfg),
    OuroTransformer(cfg, generation_cfg, nntr_cfg) {}

  virtual ~OuroEmbedding() = default;

  /**
   * @brief embed -> embed_projection -> UT loop with per-step output_norm
   *        -> pooling -> normalize.
   */
  std::pair<Tensor, Tensor> constructModel() override;

protected:
  /**
   * @brief One cache slot per (UT step, layer).
   */
  unsigned int kvCacheSlotCount() const override {
    return static_cast<unsigned int>(TOTAL_UT_STEPS) *
           static_cast<unsigned int>(NUM_LAYERS);
  }

  /**
   * @brief Map flat slot -> "layer<i>_attention" for step 0,
   *        "layer<i>_ut<k>_attention" otherwise.
   */
  std::string attentionNodeName(unsigned int slot) const override {
    const int layers = NUM_LAYERS;
    const int ut = static_cast<int>(slot) / layers;
    const int i = static_cast<int>(slot) % layers;
    if (ut == 0)
      return "layer" + std::to_string(i) + "_attention";
    return "layer" + std::to_string(i) + "_ut" + std::to_string(ut) +
           "_attention";
  }
};

} // namespace causallm

#endif /* __OURO_EMBEDDING_H__ */
