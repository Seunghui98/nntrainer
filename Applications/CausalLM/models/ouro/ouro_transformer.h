// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   ouro_transformer.h
 * @date   1 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   OuroTransformer extends the base Transformer with:
 *           1. Extra RMSNorm after attention (input_layernorm_2) and after
 *              FFN (post_attention_layernorm_2).
 *           2. Universal-Transformer loop unrolling. The whole decoder stack
 *              is iterated `total_ut_steps` times; weights of step k>0 are
 *              tied to step 0 through nntrainer's `shared_from` mechanism.
 *           3. Final RMSNorm applied at the end of each UT step (matches HF
 *              `self.norm(hidden_states)` placed inside the UT loop).
 *         The embedding-projection and the final pooling/lm_head are added
 *         by OuroEmbedding / OuroCausalLM in their constructModel().
 */

#ifndef __OURO_TRANSFORMER_H__
#define __OURO_TRANSFORMER_H__

#include <transformer.h>

namespace causallm {

/**
 * @brief OuroTransformer class
 */
class OuroTransformer : virtual public Transformer {

public:
  static constexpr const char *architectures = "OuroTransformer";

  OuroTransformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg) {
    setupOuroParameters(cfg);
  }

  virtual ~OuroTransformer() = default;

  /**
   * @brief Backward-compatible single-step block creator. Equivalent to
   *        createOuroDecoderBlock(layer_id, /*current_ut=*/0, input).
   */
  Tensor createTransformerDecoderBlock(const int layer_id,
                                       Tensor input) override;

protected:
  /**
   * @brief Build one Ouro decoder block at UT step `current_ut`.
   *        For current_ut == 0 the leaf layers get the canonical names
   *        ("layer<i>_<role>"). For current_ut > 0 they receive
   *        "layer<i>_ut<k>_<role>" + `shared_from=layer<i>_<role>` so the
   *        underlying weight tensors are shared with step 0.
   */
  virtual Tensor createOuroDecoderBlock(int layer_id, int current_ut,
                                        Tensor input);

  /**
   * @brief UT-aware variant of Transformer::createAttention. Replicates the
   *        Q/K/V/O projection layout but routes the cache placeholders to
   *        a unique flat id (current_ut * NUM_LAYERS + layer_id) so each
   *        (step, layer) pair owns a separate KV cache slot.
   */
  virtual Tensor createOuroAttention(int layer_id, int current_ut, int seq_len,
                                     int n_heads, int head_dim, Tensor q,
                                     Tensor k, Tensor v);

  /**
   * @brief UT-aware variant of Transformer::createMlp.
   */
  virtual Tensor createOuroMlp(int layer_id, int current_ut, int dim,
                               int hidden_dim, Tensor input);

  /**
   * @brief UT-aware variant of Transformer::createKVCachePlaceholders. The
   *        placeholder names are "cache_k_l<flat_id>" / "cache_v_l<flat_id>"
   *        where flat_id = current_ut * NUM_LAYERS + layer_id.
   */
  std::pair<Tensor, Tensor>
  createOuroKVCachePlaceholders(int layer_id, int current_ut, int n_heads);

  /**
   * @brief Apply the final RMSNorm. Step 0 keeps the canonical name
   *        "output_norm"; step k>0 uses "output_norm_ut<k>" tied to
   *        "output_norm" via shared_from.
   */
  Tensor applyOuroOutputNorm(int current_ut, Tensor input);

  /**
   * @brief Parse Ouro-specific config fields (total_ut_steps). Called from
   *        the constructor so it is available before constructModel() runs.
   */
  void setupOuroParameters(json &cfg);

  /**
   * @brief Compose a leaf layer name. step 0 -> "layer<i>_<role>",
   *        step k>0 -> "layer<i>_ut<k>_<role>".
   */
  std::string ouroNodeName(int layer_id, int current_ut,
                           const std::string &role) const;

  /**
   * @brief Step-0 name "layer<i>_<role>" used as shared_from target.
   */
  std::string ouroSharedFromName(int layer_id,
                                 const std::string &role) const;

  /**
   * @brief total_ut_steps from config.json (default 1).
   */
  unsigned int TOTAL_UT_STEPS = 1;
};

} // namespace causallm

#endif /* __OURO_TRANSFORMER_H__ */
