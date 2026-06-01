// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   ouro_embedding.h
 * @date   1 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   OuroEmbedding overrides constructModel() to add:
 *   - embed_projection layer (EMBED_PROJ_DIM → DIM)
 *   - UT loop unrolling with shared_from weight sharing
 *   The extra RMSNorm layers are handled by
 *   OuroTransformer::createTransformerDecoderBlock().
 */

#ifndef __OURO_EMBEDDING_H__
#define __OURO_EMBEDDING_H__

#include <ouro_transformer.h>
#include <sentence_transformer.h>

namespace causallm {

/**
 * @brief OuroEmbedding class
 *
 * Overrides constructModel() to handle Ouro-specific architecture:
 * 1. Embedding with EMBED_PROJ_DIM output + embed_projection (EMBED_PROJ_DIM → DIM)
 * 2. UT loop unrolling with shared_from weight sharing
 * 3. Extra RMSNorm layers via OuroTransformer::createTransformerDecoderBlock()
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
   * @brief Override constructModel for Ouro-specific architecture
   *
   * Creates: embedding(EMBED_PROJ_DIM) → embed_projection → UT loop
   * decoder blocks → output_norm → pooling → normalize
   */
  std::pair<Tensor, Tensor> constructModel() override;
};

} // namespace causallm

#endif /* __OURO_EMBEDDING_H__ */
