// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   gemma4_assistant_causallm.h
 * @date   29 Oct 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Gemma4AssistantForCausalLM, the MTP drafter of Gemma 4, on the CPU
 *
 * One drafting step, as transformers' Gemma4AssistantCandidateGenerator:
 *   input0  [1, 2 * backbone_hidden]: the target's scaled embedding of the
 *           last token, then the target's last hidden state (after its
 *           final norm)
 *   output  [1, vocab + backbone_hidden]: the draft logits (no soft-cap),
 *           then post_projection(hidden), the next step's hidden state
 * The decoder blocks are Gemma4's, except that the attention has no K/V of
 * its own (AssistantAttentionLayer): setSharedKV() hands it the target's
 * last sliding and last full attention layers' rows before each step.
 *
 * The tied head is built untied (output_of_causallm, a fully_connected),
 * since nothing looks a token up in the drafter's table: nntr_quantize
 * then writes it as Q4_0 for the CPU with --lmhead_dtype Q4_0.
 */

#ifndef __GEMMA4_ASSISTANT_CAUSAL_LM_H__
#define __GEMMA4_ASSISTANT_CAUSAL_LM_H__

#include <gemma4_causallm.h>

namespace causallm {

/**
 * @brief Gemma 4 assistant (MTP drafter)
 */
class Gemma4AssistantCausalLM : public Gemma4CausalLM {
public:
  static constexpr const char *architectures = "Gemma4AssistantForCausalLM";

  Gemma4AssistantCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg);

  std::pair<Tensor, Tensor> constructModel() override;
  Tensor createAttention(const int layer_id, int seq_len, int n_heads,
                         int head_dim, Tensor query, Tensor key,
                         Tensor value) override;
  void registerCustomLayers() override;

  /**
   * @brief Give every attention layer the target's K/V for the next step:
   *        sliding layers the last sliding_window + 1 rows of the last
   *        sliding layer (what transformers' bidirectional sliding mask
   *        keeps), full layers every row of the last full layer; rows
   *        [n][kv_width] each, after k_norm, RoPE and v_norm.
   * @param pos the position of the token being drafted from (constant over
   *            one drafting block)
   */
  void setSharedKV(const float *k_sliding, const float *v_sliding,
                   unsigned int n_sliding, const float *k_full,
                   const float *v_full, unsigned int n_full, unsigned int pos);

  /**
   * @brief One drafting step after setSharedKV().
   * @param in  2 * backbone_hidden floats (see the file comment)
   * @return    vocab logits, then backbone_hidden floats of the next hidden
   *            state; owned by the model, valid until the next step
   */
  const float *draftStep(const float *in);

  unsigned int getBackboneDim() const { return BACKBONE_DIM; }

private:
  unsigned int BACKBONE_DIM;
};

} // namespace causallm

#endif /* __GEMMA4_ASSISTANT_CAUSAL_LM_H__ */
