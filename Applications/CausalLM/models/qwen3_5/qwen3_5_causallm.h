// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   qwen3_5_causallm.h
 * @date   7 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   Qwen3.5 hybrid model: Gated DeltaNet (linear attn) + Transformer
 *         (self attn). Most layers use linear attention; every Nth layer uses
 *         standard self-attention with QK-norm.
 */

#ifndef __QWEN3_5_CAUSAL_LM_H__
#define __QWEN3_5_CAUSAL_LM_H__

#include <causal_lm.h>
#include <qwen3_causallm.h>

namespace causallm {

/**
 * @brief Qwen3_5Transformer class - hybrid linear + self attention
 *
 * Uses addLayer API (not Tensor API) to ensure topological sort produces
 * the same layer ordering as support_qwen3_5_1 branch, which matches
 * the weight_converter save order.
 */
class Qwen3_5Transformer : virtual public Transformer {
public:
  static constexpr const char *architectures = "Qwen3_5Transformer";

  Qwen3_5Transformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg) {
    setupQwen35Parameters(cfg, nntr_cfg);
  }

  virtual ~Qwen3_5Transformer() = default;

  void constructModel() override;

  std::vector<LayerHandle>
  createTransformerDecoderBlock(const int layer_id, std::string input_name);

  std::vector<LayerHandle>
  createAttention(const int layer_id, int seq_len, int n_heads, int head_dim,
                  std::string query_name, std::string key_name,
                  std::string value_name);

  std::vector<LayerHandle> createMlp(const int layer_id, int dim,
                                     int hidden_dim,
                                     std::string input_name);

  std::vector<LayerHandle>
  createLinearAttentionBlock(const int layer_id, std::string input_name);

  void registerCustomLayers() override;

protected:
  void setupQwen35Parameters(json &cfg, json &nntr_cfg);

  std::vector<bool> is_self_attn_layer;
  unsigned int LINEAR_NUM_V_HEADS;
  unsigned int LINEAR_HEAD_K_DIM;
  unsigned int LINEAR_HEAD_V_DIM;
  unsigned int LINEAR_CONV_KERNEL;
  unsigned int SELF_ATTN_HEAD_DIM;
  bool force_untie_embedding = false; // set by CausalLM when dtypes differ
};

/**
 * @brief Qwen3_5CausalLM class
 */
class Qwen3_5CausalLM : public CausalLM, public Qwen3_5Transformer {

public:
  static constexpr const char *architectures = "Qwen3_5ForCausalLM";

  static json &getTextConfig(json &cfg) {
    return cfg.contains("text_config") ? cfg["text_config"] : cfg;
  }

  Qwen3_5CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(getTextConfig(cfg), generation_cfg, nntr_cfg,
                ModelType::CAUSALLM),
    CausalLM(getTextConfig(cfg), generation_cfg, nntr_cfg),
    Qwen3_5Transformer(getTextConfig(cfg), generation_cfg, nntr_cfg) {}

  virtual ~Qwen3_5CausalLM() = default;

  void constructModel() override;

  void registerCustomLayers() override;
};

} // namespace causallm

#endif /* __QWEN3_5_CAUSAL_LM_H__ */
