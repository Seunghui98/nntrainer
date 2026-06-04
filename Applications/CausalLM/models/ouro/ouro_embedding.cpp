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

#include <algorithm>

#include <app_context.h>
#include <common.h>
#include <layer_context.h>
#include <mha_core.h>
#include <nntrainer_error.h>
#include <tensor.h>

#include <llm_util.hpp>
#include <ouro_embedding.h>

namespace causallm {

std::pair<Tensor, Tensor> OuroEmbedding::constructTransformerModule() {

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

  // current_ut_ steers the leaf-name / shared_from logic in OuroTransformer
  // overrides. Step k>0 leaves get layer<i>_ut<k>_<role> + shared_from=
  // layer<i>_<role>, so only one set of weights is stored in the pool.
  for (int ut = 0; ut < static_cast<int>(TOTAL_UT_STEPS); ++ut) {
    current_ut_ = ut;
    for (int i = 0; i < NUM_LAYERS; ++i) {
      h = createTransformerDecoderBlock(i, h);
    }
    h = applyOuroOutputNorm(ut, h);
  }
  current_ut_ = 0;

  // Pooling + normalize are appended by SentenceTransformer::constructModel
  // based on modules.json (1_Pooling, 2_Normalize).
  return {x, h};
}

std::vector<float *> OuroEmbedding::encode(const WSTR prompt,
                                           const WSTR system_prompt,
                                           const WSTR tail_prompt) {
  if (!is_initialized) {
    throw std::runtime_error("OuroEmbedding model is not initialized. Please "
                             "call initialize() before encode().");
  }

  std::string prompt_ = system_prompt + prompt + tail_prompt;
  auto _input = tokenizer->Encode(prompt_, true);

  unsigned int input_len =
    std::min(static_cast<unsigned int>(_input.size()),
             static_cast<unsigned int>(MAX_SEQ_LEN));

  std::vector<float> input_sample(static_cast<size_t>(BATCH_SIZE) * MAX_SEQ_LEN,
                                  0.0f);
  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    for (unsigned int i = 0; i < input_len; ++i) {
      input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN + i] =
        static_cast<float>(_input[i]);
    }
  }

  // 3-input mha_core mode: the graph has no cache_k_l<i> / cache_v_l<i>
  // placeholder inputs, so the only runtime input is the token-id buffer.
  // This bypasses SentenceTransformer::allocateAndBindKVCache which would
  // try to find non-existent placeholders.
  std::vector<float *> input;
  input.push_back(input_sample.data());

  std::vector<float *> label;
  return model->incremental_inference(BATCH_SIZE, input, label, input_len, 0,
                                      input_len, false);
}

} // namespace causallm
