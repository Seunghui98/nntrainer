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
 *
 *         Layer type pattern for Qwen3.5-2B (24 layers):
 *           linear, linear, linear, self_attn,  (0-3)
 *           linear, linear, linear, self_attn,  (4-7)
 *           ...
 *           linear, linear, linear, self_attn   (20-23)
 */

#ifndef __QWEN3_5_CAUSAL_LM_H__
#define __QWEN3_5_CAUSAL_LM_H__

#include <causal_lm.h>
#include <qwen3_causallm.h>

namespace causallm {

/**
 * @brief Qwen3_5Transformer class - hybrid linear + self attention
 */
class Qwen3_5Transformer : virtual public Transformer {
public:
  static constexpr const char *architectures = "Qwen3_5Transformer";

  Qwen3_5Transformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg) {
    setupQwen35Parameters(cfg, nntr_cfg);
  }

  virtual ~Qwen3_5Transformer() = default;

  /**
   * @brief Override constructModel to build hybrid architecture
   */
  void constructModel() override;

  /**
   * @brief Create self-attention block (reuses Qwen3 attention with QK-norm
   *        + output gate for Qwen3.5)
   */
  Tensor createAttention(const int layer_id, int seq_len, int n_heads,
                          int head_dim, Tensor query, Tensor key,
                          Tensor value) override;

  /**
   * @brief Create MLP (SwiGLU) block
   */
  Tensor createMlp(const int layer_id, int dim, int hidden_dim,
                    Tensor input) override;

  /**
   * @brief Create linear attention (Gated DeltaNet) decoder block
   */
  Tensor createLinearAttentionBlock(const int layer_id, Tensor input);

  void registerCustomLayers() override;

protected:
  void setupQwen35Parameters(json &cfg, json &nntr_cfg);

  /** Which layers use self-attention (true) vs linear attention (false) */
  std::vector<bool> is_self_attn_layer;

  /** Linear attention parameters */
  unsigned int LINEAR_NUM_V_HEADS;
  unsigned int LINEAR_HEAD_K_DIM;
  unsigned int LINEAR_HEAD_V_DIM;
  unsigned int LINEAR_CONV_KERNEL;
  unsigned int SELF_ATTN_HEAD_DIM;
};

/**
 * @brief Qwen3_5CausalLM class
 */
class Qwen3_5CausalLM : public CausalLM, public Qwen3_5Transformer {

public:
  static constexpr const char *architectures = "Qwen3_5ForCausalLM";

  /** Helper to unwrap VLM nested config */
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
