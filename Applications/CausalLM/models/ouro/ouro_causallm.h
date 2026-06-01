// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   ouro_causallm.h
 * @date   1 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   OuroCausalLM builds the full Ouro decoder-only graph:
 *           1. Token embedding (vocab -> intermediate_size).
 *           2. embed_projection (intermediate_size -> hidden_size).
 *           3. UT loop: total_ut_steps iterations of the Ouro decoder stack
 *              with extra RMSNorms and weight sharing via shared_from.
 *           4. LM head.
 *         The KV cache is allocated with TOTAL_UT_STEPS * NUM_LAYERS slots
 *         so each (step, layer) pair owns a private cache_k/cache_v.
 */

#ifndef __OURO_CAUSAL_LM_H__
#define __OURO_CAUSAL_LM_H__

#include <causal_lm.h>
#include <ouro_transformer.h>

namespace causallm {

/**
 * @brief OuroCausalLM class
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
   * @brief Build the Ouro graph: embed -> embed_projection -> UT loop ->
   *        per-step output_norm -> lm_head.
   */
  std::pair<Tensor, Tensor> constructModel() override;

protected:
  /**
   * @brief Allocate TOTAL_UT_STEPS * NUM_LAYERS KV cache slots and bind
   *        each one to the corresponding `layer<i>[_ut<k>]_attention` node.
   */
  void allocateAndBindKVCache() override;
};

} // namespace causallm

#endif /* __OURO_CAUSAL_LM_H__ */
