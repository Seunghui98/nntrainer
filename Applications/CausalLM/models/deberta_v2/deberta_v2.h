// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   debertav2.h
 * @date   14 January 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note  Please refer to the following code :
 * https://github.com/huggingface/transformers/blob/5c1c72b/src/transformers/models/deberta/modeling_deberta.py
 */

#ifndef __DEBERTA_V2_H__
#define __DEBERTA_V2_H__

#include <sentence_transformer.h>
#include <transformer.h>

namespace causallm {

/**
 * @brief DebertaV2 class
 */
class DebertaV2 : public SentenceTransformer {

public:
  static constexpr const char *architectures = "DebertaV2";

  DebertaV2(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::EMBEDDING),
    SentenceTransformer(cfg, generation_cfg, nntr_cfg) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~DebertaV2() = default;

protected:
  /**
   * @brief Construct Model
   */
  void constructModel() override;

  std::vector<LayerHandle> createDebertaLayer(const int layer_id,
                                              std::string input_name,
                                              std::string rel_embeddings_name);

  std::vector<LayerHandle>
  createDebertaV2Attention(const int layer_id, std::string input_name,
                           std::string rel_embeddings_name);

  /**
   * @brief Setup the parameters for the Deberta V2 model
   */
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  /**
   * @brief register CustomLayers
   */
  void registerCustomLayers() override;

  int MAX_RELATIVE_POSITIONS;
  bool C2P;
  bool P2C;
  bool SHARE_ATT_KEY;
  bool RELATIVE_ATTENTION;
  int POSITION_BUCKETS;
};

} // namespace causallm

#endif /* __DEBERTA_V2_H__ */
