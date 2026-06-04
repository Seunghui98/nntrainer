// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   ouro_embedding.h
 * @date   1 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   OuroEmbedding plugs the Ouro Universal-Transformer backbone into
 *         SentenceTransformer. The backbone graph (token embedding,
 *         embed_projection, UT loop with shared_from weight tying, per-step
 *         RMSNorm) is built by constructTransformerModule(); pooling and
 *         L2 normalization are appended by SentenceTransformer::constructModel
 *         from modules.json (1_Pooling, 2_Normalize).
 *         The embedding pipeline runs as a single prefill so OuroEmbedding
 *         flips ouro_use_external_kv_cache_ off (3-input mha_core), and
 *         overrides encode() to skip SentenceTransformer's external KV-cache
 *         binding which is not applicable in this mode.
 */

#ifndef __OURO_EMBEDDING_H__
#define __OURO_EMBEDDING_H__

#include <ouro_transformer.h>
#include <sentence_transformer.h>

namespace causallm {

/**
 * @brief OuroEmbedding class
 */
class OuroEmbedding : public SentenceTransformer, public OuroTransformer {

public:
  static constexpr const char *architectures = "OuroModel";

  OuroEmbedding(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::EMBEDDING),
    SentenceTransformer(cfg, generation_cfg, nntr_cfg),
    OuroTransformer(cfg, generation_cfg, nntr_cfg) {
    // Single-prefill embedding pipeline: let mha_core own its scratch KV
    // cache internally (3-input mode) so the graph has no cache_k_l<i> /
    // cache_v_l<i> placeholder inputs to bind.
    ouro_use_external_kv_cache_ = false;
  }

  virtual ~OuroEmbedding() = default;

  /**
   * @brief Encode without SentenceTransformer's KV-cache binding (the
   *        embedding graph runs mha_core in 3-input mode, so there are no
   *        external cache placeholders to plumb).
   */
  std::vector<float *> encode(const WSTR prompt,
                              const WSTR system_prompt = "",
                              const WSTR tail_prompt = "") override;

protected:
  /**
   * @brief Build the Ouro backbone: embed -> embed_projection -> UT loop ->
   *        per-step output_norm. SentenceTransformer::constructModel adds
   *        pooling + normalize on top of this from modules.json.
   */
  std::pair<Tensor, Tensor> constructTransformerModule() override;
};

} // namespace causallm

#endif /* __OURO_EMBEDDING_H__ */
