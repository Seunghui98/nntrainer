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

#include <ouro_embedding.h>
#include <llm_util.hpp>

namespace causallm {

std::pair<Tensor, Tensor> OuroEmbedding::constructModel() {

  // Input
  Tensor x =
    Tensor({1, 1, 1, static_cast<unsigned int>(INIT_SEQ_LEN)}, "input0");

  // Ouro uses embed_tokens (vocab → intermediate_size) + embed_projection
  // (intermediate_size → hidden_size)
  const unsigned int EMBED_PROJ_DIM = INTERMEDIATE_SIZE; // 1536

  // Embedding: vocab → EMBED_PROJ_DIM (intermediate_size)
  const std::string embedding_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";

  LayerHandle embedding(createLayer(
    embedding_type,
    {"name=embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
     "weight_dtype=" + EMBEDDING_DTYPE,
     "out_dim=" + std::to_string(EMBED_PROJ_DIM),
     "scale=" + std::to_string(EMBEDDING_SCALE)}));
  Tensor h = embedding(x);

  // Embed projection: EMBED_PROJ_DIM → DIM (intermediate_size → hidden_size)
  LayerHandle embed_proj(createLayer(
    "fully_connected",
    {withKey("name", "embed_projection"),
     withKey("unit", DIM),
     withKey("disable_bias", "true"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  h = embed_proj(h);

  // Transformer decoder blocks (UT step 0: create new blocks)
  for (int i = 0; i < NUM_LAYERS; ++i) {
    h = createTransformerDecoderBlock(i, h);
  }

  // TODO: UT loop unrolling (total_ut_steps > 1)
  // For subsequent UT steps, create decoder blocks with shared_from
  // pointing to the first step's blocks.

  // Final RMSNorm
  LayerHandle out_norm(createLayer(
    "rms_norm", {withKey("name", "output_norm"),
                 withKey("epsilon", std::to_string(NORM_EPS)),
                 withKey("packed", "false")}));
  h = out_norm(h);

  // Pooling (mean over sequence dimension) - uses custom embedding_pooling layer
  LayerHandle pooling(createLayer(
    "embedding_pooling", {withKey("name", "embedding_pooling")}));
  h = pooling(h);

  // L2 Normalize - uses custom embedding_normalize layer
  LayerHandle normalize(createLayer(
    "embedding_normalize", {withKey("name", "embedding_normalize_layer")}));
  h = normalize(h);

  return {x, h};
}

} // namespace causallm
