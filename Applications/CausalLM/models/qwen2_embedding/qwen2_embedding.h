#ifndef __QWEN2_EMBEDDING_H__
#define __QWEN2_EMBEDDING_H__

#include "../embedding.h"
#include "../qwen2/qwen2_causallm.h"

namespace causallm {

/**
 * @brief Qwen2Embedding class
 */
class Qwen2Embedding : public Embedding, public Qwen2Transformer {

public:
  Qwen2Embedding(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::EMBEDDING),
    Embedding(cfg, generation_cfg, nntr_cfg),
    Qwen2Transformer(cfg, generation_cfg, nntr_cfg) {}

  virtual ~Qwen2Embedding() {}
};

} // namespace causallm

#endif /* __QWEN2_EMBEDDING_H__ */
