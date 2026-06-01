// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   ouro_transformer.cpp
 * @date   1 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file defines OuroTransformer's basic actions
 */

#include <llm_util.hpp>
#include <ouro_transformer.h>

namespace causallm {

Tensor OuroTransformer::createTransformerDecoderBlock(const int layer_id,
                                                      Tensor input) {

  // Attention pre-norm
  LayerHandle attn_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor normed = attn_norm(input);

  // Attention sub-graph (Q/K/V projections + mha_core + output projection)
  Tensor att_out = createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                                   normed, normed, normed);

  // Extra RMSNorm after attention (Ouro-specific)
  LayerHandle attn_norm_2(createLayer(
    "rms_norm",
    {withKey("name",
             "layer" + std::to_string(layer_id) + "_attention_norm_2"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  att_out = attn_norm_2(att_out);

  // Residual connection after attention
  LayerHandle decoder_add(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_add")}));
  Tensor residual = decoder_add({input, att_out});

  // FFN pre-norm
  LayerHandle ffn_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor ffn_normed = ffn_norm(residual);

  // FFN sub-graph
  Tensor ffn_out = createMlp(layer_id, DIM, INTERMEDIATE_SIZE, ffn_normed);

  // Extra RMSNorm after FFN (Ouro-specific)
  LayerHandle ffn_norm_2(createLayer(
    "rms_norm",
    {withKey("name",
             "layer" + std::to_string(layer_id) + "_ffn_norm_2"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  ffn_out = ffn_norm_2(ffn_out);

  // Residual connection after FFN
  LayerHandle decoder_output(createLayer(
    "addition",
    {withKey("name",
             "layer" + std::to_string(layer_id) + "_decoder_output")}));
  return decoder_output({residual, ffn_out});
}

} // namespace causallm
