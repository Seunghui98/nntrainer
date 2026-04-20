// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 NNTrainer Contributors
 *
 * @file   qwen35_causallm.h
 * @date   07 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 * @note   Qwen3.5 Causal Language Model (language_model component only).
 *         Architecture: standard transformer decoder with per-head Q/K RMSNorm
 *         and Grouped Query Attention (GQA), identical structure to Qwen3.
 *
 *         HuggingFace reference:
 *           https://huggingface.co/Qwen/Qwen3.5-2B
 *
 *         Supported model types (architectures field in config.json):
 *           "Qwen3_5ForCausalLM"
 *
 *         Layer structure per decoder block:
 *           input → RMSNorm → [V,K,Kn,Q,Qn projections] → MHA → +residual
 *                           → RMSNorm → SwiGLU FFN → +residual
 */

#ifndef __QWEN35_CAUSAL_LM_H__
#define __QWEN35_CAUSAL_LM_H__

#include <causal_lm.h>

namespace causallm {

/**
 * @brief Qwen35Transformer
 *
 * Overrides createAttention() to add per-head Q and K RMSNorm layers
 * (ReshapedRMSNormLayer), matching the Qwen3.5 attention design.
 */
class Qwen35Transformer : virtual public Transformer {
public:
  static constexpr const char *architectures = "Qwen35Transformer";

  Qwen35Transformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg) {}

  virtual ~Qwen35Transformer() = default;

  /**
   * @brief Build attention sub-graph for one decoder block.
   *
   * Graph: input → V_proj
   *              → K_proj → K_norm (reshaped RMS)
   *              → Q_proj → Q_norm (reshaped RMS)
   *              → MHA core (GQA + RoPE + KV-cache)
   *              → O_proj
   */
  Tensor createAttention(const int layer_id, int seq_len, int n_heads,
                         int head_dim, Tensor query, Tensor key,
                         Tensor value) override;

  void registerCustomLayers() override;
};

/**
 * @brief Qwen35CausalLM
 *
 * Full decoder-only language model using Qwen35Transformer blocks.
 * Reads all hyperparameters from config.json at construction time.
 *
 * Expected config.json fields (standard HuggingFace Qwen3.5 format):
 *   hidden_size, num_hidden_layers, num_attention_heads,
 *   num_key_value_heads, head_dim, intermediate_size,
 *   vocab_size, rms_norm_eps, rope_theta,
 *   max_position_embeddings, tie_word_embeddings
 *
 * If the model is a vision-language model, set "lm_prefix" in
 * nntr_config.json to "language_model" so the weight converter
 * strips the vision encoder weights.
 */
class Qwen35CausalLM : public CausalLM, public Qwen35Transformer {
public:
  static constexpr const char *architectures = "Qwen3_5ForCausalLM";

  Qwen35CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
    CausalLM(cfg, generation_cfg, nntr_cfg),
    Qwen35Transformer(cfg, generation_cfg, nntr_cfg) {}

  virtual ~Qwen35CausalLM() = default;

  void registerCustomLayers() override;
};

} // namespace causallm

#endif /* __QWEN35_CAUSAL_LM_H__ */
