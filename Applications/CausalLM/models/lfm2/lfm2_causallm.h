// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   lfm2_causallm.h
 * @date   13 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Auto-generated
 * @bug    No known bugs except for NYI items
 * @note   LiquidAI LFM2/LFM2.5 hybrid model: Double-gated short conv + GQA
 *         attention. Most layers use conv; selected layers use full attention
 *         with QK-norm.
 */

#ifndef __LFM2_CAUSAL_LM_H__
#define __LFM2_CAUSAL_LM_H__

#include <causal_lm.h>

namespace causallm {

/**
 * @brief LFM2Transformer class - hybrid double-gated conv + GQA attention
 *
 * Uses addLayer API to ensure topological sort produces the same layer
 * ordering as the weight converter save order.
 *
 * Conv block structure (Lfm2ShortConv):
 *   in_proj(hidden, 3*hidden) → split B,C,x → B*x → CausalConv1D → C*conv →
 *   out_proj(hidden, hidden)
 *
 * Attention block structure (Lfm2Attention):
 *   q_proj → q_norm → k_proj → k_norm → v_proj → RoPE + GQA → o_proj
 */
class LFM2Transformer : virtual public Transformer {
public:
  static constexpr const char *architectures = "LFM2Transformer";

  LFM2Transformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg) {
    setupLFM2Parameters(cfg, nntr_cfg);
  }

  virtual ~LFM2Transformer() = default;

  void constructModel() override;

  std::vector<LayerHandle>
  createAttentionBlock(const int layer_id, std::string input_name);

  std::vector<LayerHandle>
  createAttention(const int layer_id, int seq_len, int n_heads, int head_dim,
                  std::string query_name, std::string key_name,
                  std::string value_name);

  std::vector<LayerHandle> createMlp(const int layer_id, int dim,
                                     int hidden_dim, std::string input_name);

  std::vector<LayerHandle> createConvBlock(const int layer_id,
                                           std::string input_name);

  void registerCustomLayers() override;

protected:
  void setupLFM2Parameters(json &cfg, json &nntr_cfg);

  std::vector<bool> is_attn_layer;
  unsigned int CONV_L_CACHE;
  unsigned int ACTUAL_INTERMEDIATE_SIZE;
  bool HAS_EMBEDDING_NORM;
};

/**
 * @brief LFM2CausalLM class
 */
class LFM2CausalLM : public CausalLM, public LFM2Transformer {

public:
  static constexpr const char *architectures = "Lfm2ForCausalLM";

  static json &getTextConfig(json &cfg) {
    return cfg.contains("text_config") ? cfg["text_config"] : cfg;
  }

  LFM2CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(getTextConfig(cfg), generation_cfg, nntr_cfg,
                ModelType::CAUSALLM),
    CausalLM(getTextConfig(cfg), generation_cfg, nntr_cfg),
    LFM2Transformer(getTextConfig(cfg), generation_cfg, nntr_cfg) {}

  virtual ~LFM2CausalLM() = default;

  void constructModel() override;

  void registerCustomLayers() override;
};

} // namespace causallm

#endif /* __LFM2_CAUSAL_LM_H__ */
