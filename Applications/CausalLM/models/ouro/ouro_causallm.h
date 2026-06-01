// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   ouro_causallm.h
 * @date   1 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   OuroCausalLM overrides constructModel() to add:
 *   - embed_projection layer (EMBED_PROJ_DIM → DIM)
 *   - UT loop unrolling with shared_from weight sharing
 *   The extra RMSNorm layers are handled by
 *   OuroTransformer::createTransformerDecoderBlock().
 */

#ifndef __OURO_CAUSAL_LM_H__
#define __OURO_CAUSAL_LM_H__

#include <causal_lm.h>
#include <ouro_transformer.h>

namespace causallm {

/**
 * @brief OuroCausalLM class
 *
 * Overrides constructModel() to handle Ouro-specific architecture:
 * 1. Embedding with EMBED_PROJ_DIM output + embed_projection (EMBED_PROJ_DIM → DIM)
 * 2. UT loop unrolling with shared_from weight sharing
 * 3. Extra RMSNorm layers via OuroTransformer::createTransformerDecoderBlock()
 */
class OuroCausalLM : public CausalLM, public OuroTransformer {

public:
  static constexpr const char *architectures = "OuroForCausalLM";

  OuroCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
    CausalLM(cfg, generation_cfg, nntr_cfg),
    OuroTransformer(cfg, generation_cfg, nntr_cfg) {}

  virtual ~OuroCausalLM() = default;

  /**
   * @brief Override constructModel for Ouro-specific architecture
   *
   * Creates: embedding(EMBED_PROJ_DIM) → embed_projection → UT loop
   * decoder blocks → output_norm → lm_head
   */
  std::pair<Tensor, Tensor> constructModel() override;
};

} // namespace causallm

#endif /* __OURO_CAUSAL_LM_H__ */
