// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   gliner2_multi_v1.h
 * @date   30 January 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   Please refer to the following code :
 *  https://github.com/fastino-ai/GLiNER2/blob/72241d6/gliner2/inference/engine.py
 */

#ifndef __GLINER2_MULTI_V1_H__
#define __GLINER2_MULTI_V1_H__

#include "../embedding.h"
#include <deberta_v2.h>
#include <transformer.h>

namespace causallm {

/**
 * @brief GLiner2MultiV1 class
 */
class GLiner2MultiV1 : public DebertaV2 {

public:
  static constexpr const char *architectures = "GLiner2MultiV1";

  GLiner2MultiV1(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::EMBEDDING),
    DebertaV2(cfg, generation_cfg, nntr_cfg) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~GLiner2MultiV1() = default;

protected:
  /**
   * @brief Construct Model
   */
  void constructModel() override;

  std::vector<LayerHandle> createCountLSTMLayer(std::string input_name, std::string name_prefix, int in_dim, int out_dim, int hidden_dim);

  /**
   * @brief Setup the parameters for the GLiner2 Multi V1 model
   */
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  /**
   * @brief create Feed Forward Layer
   */
  std::vector<LayerHandle> createMlp(const std::string &layer_name, int dim,
                                     int hidden_dim,
                                     std::string input_name);
  
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = "", const WSTR tail_prompt = "") override;
  
  /**
   * @brief Encode the prompt and return the embedding
   * @param prompt User prompt
   * @param system_prompt System prompt
   * @param tail_prompt Tail prompt
   * @return Embedding output from the model
   */
  std::vector<float *> encode(const WSTR prompt, const WSTR system_prompt = "",
                              const WSTR tail_prompt = "") override;

  /**
   * @brief Generate span indices
   * @param seq_len Sequence length
   * @param max_width Maximum span width
   * @return Pair of start and end indices vectors
   */
  std::pair<std::vector<float>, std::vector<float>>
  generate_spans(int seq_len, int max_width, int offset = 0);

  /**
   * @brief register CustomLayers
   */
  void registerCustomLayers() override;

private:
  int MAX_RELATIVE_POSITIONS;
  bool c2p;
  bool p2c;
  bool SHARE_ATT_KEY;
  bool RELATIVE_ATTENTION;
  int POSITION_BUCKETS;
  int MAX_WIDTH = 12; // default value
  std::vector<std::string> labels;
};

} // namespace causallm

#endif /* __GLINER2_MULTI_V1_H__ */
