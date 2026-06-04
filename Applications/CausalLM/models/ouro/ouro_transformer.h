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
   * @brief Build one Ouro decoder block for the UT step set in
   *        `current_ut_`. Step 0 keeps the canonical "layer<i>_<role>"
   *        names; step k>0 receives "layer<i>_ut<k>_<role>" plus
   *        `shared_from=layer<i>_<role>` so the underlying weight
   *        tensors are shared with step 0.
   */
  Tensor createTransformerDecoderBlock(const int layer_id,
                                       Tensor input) override;

protected:
  /**
   * @brief UT-aware override. Names attention sub-graph nodes per the
   *        current UT step, ties FC weights via `shared_from`, and
   *        routes mha_core to either internal (3-input) or external
   *        (5-input) cache mode depending on ouro_use_external_kv_cache_.
   */
  Tensor createAttention(const int layer_id, int seq_len, int n_heads,
                         int head_dim, Tensor query, Tensor key,
                         Tensor value) override;

  /**
   * @brief UT-aware override. Names FFN sub-graph nodes per the current
   *        UT step and ties FC weights via `shared_from`.
   */
  Tensor createMlp(const int layer_id, int dim, int hidden_dim,
                   Tensor input) override;

  /**
   * @brief Per-(step, layer) external cache placeholders. Only used when
   *        ouro_use_external_kv_cache_ is true (OuroCausalLM).
   *        Placeholder names are "cache_k_l<flat>" / "cache_v_l<flat>"
   *        where flat = current_ut_ * NUM_LAYERS + layer_id.
   */
  std::pair<Tensor, Tensor>
  createOuroKVCachePlaceholders(int layer_id, int n_heads);

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
  std::string ouroNodeName(int layer_id, const std::string &role) const;

  /**
   * @brief Step-0 name "layer<i>_<role>" used as shared_from target.
   */
  std::string ouroSharedFromName(int layer_id,
                                 const std::string &role) const;

  /**
   * @brief total_ut_steps from config.json (default 1).
   */
  unsigned int TOTAL_UT_STEPS = 1;

  /**
   * @brief UT step the next createTransformerDecoderBlock / createAttention
   *        / createMlp call should emit. OuroCausalLM and OuroEmbedding
   *        update this before each layer-loop pass in constructModel.
   */
  int current_ut_ = 0;

  /**
   * @brief When true (OuroCausalLM default) mha_core is wired in 5-input
   *        mode with externally-bound cache_k_l<flat> placeholders owned by
   *        OuroCausalLM's KVCacheManager. When false (set by OuroEmbedding)
   *        mha_core is wired in 3-input mode and allocates its scratch KV
   *        cache internally, so no host cache management is needed for the
   *        single-prefill embedding path.
   */
  bool ouro_use_external_kv_cache_ = true;
};

} // namespace causallm

#endif /* __OURO_TRANSFORMER_H__ */
