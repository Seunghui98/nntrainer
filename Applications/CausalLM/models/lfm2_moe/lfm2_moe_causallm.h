// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jungwon-Lee <jungone.lee@samsung.com>
 *
 * @file   lfm2_moe_causallm.h
 * @date   06 July 2026
 * @brief  This declares the LFM2-8B-A1B Mixture-of-Experts causal LM.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   Inherits the dense LFM2 model (Lfm2CausalLM) and replaces the dense
 *         SwiGLU FFN with MoE routing on layers >= num_dense_layers.
 */

#ifndef __LFM2_MOE_CAUSALLM_H__
#define __LFM2_MOE_CAUSALLM_H__

#include <set>

#include "lfm2_causallm.h"

namespace causallm {

/**
 * @brief Lfm2MoeCausalLM - LFM2-8B-A1B MoE (Base variant, "lfm2_moe" layer)
 */
class Lfm2MoeCausalLM : public Lfm2CausalLM {

public:
  static constexpr const char *architectures = "Lfm2MoeForCausalLM";

  Lfm2MoeCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
    Lfm2CausalLM(cfg, generation_cfg, nntr_cfg) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~Lfm2MoeCausalLM() = default;

  Tensor createMlp(const int layer_id, int dim, int hidden_dim,
                   Tensor input) override;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  void registerCustomLayers() override;

protected:
  unsigned int NUM_EXPERTS = 0;
  unsigned int NUM_EXPERTS_PER_TOK = 0;
  unsigned int MOE_INTERMEDIATE_SIZE = 0;
  unsigned int NUM_DENSE_LAYERS = 0;
  /** MoE expert FFN weight dtype; defaults to FC_LAYER_DTYPE (see
   * setupParameters) unless nntr_config.json carries its own
   * moe_layer_dtype, which nntr_quantize's --moe_dtype writes. */
  std::string MOE_LAYER_DTYPE;
  /** engine ("cpu"/"htp"/...) for MoE FFN layers selected by
   * MOE_HTP_LAYERS (or every MoE layer, if that set is empty); defaults to
   * "cpu" so an existing model run is unaffected unless nntr_config.json
   * opts in. See setupParameters / createMoeLayer. */
  std::string MOE_ENGINE = "cpu";
  /** layer_ids that get MOE_ENGINE; empty means "every MoE layer" (the
   * whole-model case, e.g. the tiny fixture). Non-empty bounds a real
   * checkpoint's device run to the handles/VTCM the residency wall allows
   * (docs/htp_attention/40_moe_ffn_htp_task.md section2.4) -- layer_ids not
   * in this set always get "cpu", regardless of MOE_ENGINE. Read from
   * nntr_config.json's moe_htp_layers (comma-separated layer_ids). */
  std::set<int> MOE_HTP_LAYERS;

  /**
   * @brief Create the variant-specific MoE layer for a given layer id.
   * @note Overridden by the Slim / CachedSlim variants to emit their layer type.
   */
  virtual Tensor createMoeLayer(const int layer_id, Tensor input);
};

} // namespace causallm

#endif /* __LFM2_MOE_CAUSALLM_H__ */
