// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   ouro_transformer.h
 * @date   1 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This ouro_transformer.h constructs a class for Ouro model which
 * extends Transformer with extra RMSNorm layers in the decoder block.
 * @note   Key differences from base Transformer decoder block:
 *   - Extra RMSNorm after attention: input_layernorm_2
 *   - Extra RMSNorm after FFN: post_attention_layernorm_2
 * @note   The UT loop and embedding projection are handled in
 * OuroCausalLM/OuroEmbedding's constructModel() override.
 */

#ifndef __OURO_TRANSFORMER_H__
#define __OURO_TRANSFORMER_H__

#include <transformer.h>

namespace causallm {

/**
 * @brief OuroTransformer class
 *
 * Ouro model extends the base Transformer with extra RMSNorm layers
 * in the decoder block. Only createTransformerDecoderBlock is overridden
 * to add the extra normalization layers.
 *
 * The UT loop unrolling and embedding projection are handled separately
 * in OuroCausalLM and OuroEmbedding via their constructModel() overrides.
 */
class OuroTransformer : virtual public Transformer {

public:
  static constexpr const char *architectures = "OuroTransformer";

  OuroTransformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg) {}

  virtual ~OuroTransformer() = default;

  /**
   * @brief Create one Ouro decoder block with extra RMSNorm layers
   *
   * Structure:
   *   residual = x
   *   x = input_layernorm(x)
   *   x = attention(x)
   *   x = input_layernorm_2(x)       ← extra
   *   x = residual + x
   *   residual = x
   *   x = post_attention_layernorm(x)
   *   x = mlp(x)
   *   x = post_attention_layernorm_2(x) ← extra
   *   x = residual + x
   */
  Tensor createTransformerDecoderBlock(const int layer_id,
                                       Tensor input) override;
};

} // namespace causallm

#endif /* __OURO_TRANSFORMER_H__ */
