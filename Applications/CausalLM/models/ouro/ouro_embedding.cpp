// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   ouro_embedding.cpp
 * @date   1 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file defines OuroEmbedding's basic actions
 */

#include <app_context.h>
#include <common.h>
#include <embedding_normalize_layer.h>
#include <embedding_pooling_layer.h>
#include <layer_context.h>
#include <mha_core.h>
#include <nntrainer_error.h>
#include <tensor.h>

#include <llm_util.hpp>
#include <ouro_embedding.h>

namespace causallm {

std::pair<Tensor, Tensor> OuroEmbedding::constructModel() {

  Tensor x =
    Tensor({1, 1, 1, static_cast<unsigned int>(INIT_SEQ_LEN)}, "input0");

  // Ouro: embed_tokens (vocab -> intermediate_size) + embed_projection
  // (intermediate_size -> hidden_size).
  const unsigned int EMBED_PROJ_DIM = INTERMEDIATE_SIZE;

  const std::string embedding_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";

  LayerHandle embedding(createLayer(
    embedding_type,
    {"name=embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
     "weight_dtype=" + EMBEDDING_DTYPE,
     "out_dim=" + std::to_string(EMBED_PROJ_DIM),
     "scale=" + std::to_string(EMBEDDING_SCALE)}));
  Tensor h = embedding(x);

  LayerHandle embed_proj(createLayer(
    "fully_connected",
    {withKey("name", "embed_projection"), withKey("unit", DIM),
     withKey("disable_bias", "true"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  h = embed_proj(h);

  for (int ut = 0; ut < static_cast<int>(TOTAL_UT_STEPS); ++ut) {
    for (int i = 0; i < NUM_LAYERS; ++i) {
      h = createOuroDecoderBlock(i, ut, h);
    }
    h = applyOuroOutputNorm(ut, h);
  }

  // Pooling: mean over the (actual) sequence positions, matching
  // sentence-transformers 1_Pooling/config.json. Without the explicit
  // pooling_mode_mean_tokens flag the layer falls through to setZero().
  LayerHandle pooling(createLayer(
    "embedding_pooling",
    {withKey("name", "embedding_pooling"),
     withKey("pooling_mode_mean_tokens", "true"),
     withKey("word_embedding_dimension", DIM)}));
  h = pooling(h);

  LayerHandle normalize(createLayer(
    "embedding_normalize", {withKey("name", "embedding_normalize_layer")}));
  h = normalize(h);

  return {x, h};
}

} // namespace causallm
