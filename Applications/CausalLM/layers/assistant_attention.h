// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   assistant_attention.h
 * @date   29 Oct 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Query-only attention of the Gemma 4 assistant (MTP drafter) over
 *         K/V rows the target model produced.
 *
 * Gemma4AssistantForCausalLM's layers own no K/V projection: every query
 * reads the target's last sliding or last full attention layer
 * (shared_kv_states). The rows are handed in with setSharedKV(), already
 * after k_norm, RoPE and v_norm, laid out [rows][num_heads_kv * head_dim]
 * like the KV cache. Every query attends to all of them (the assistant's
 * mask is bidirectional; the caller passes only the window rows to a
 * sliding layer), at scaling 1.0, with RoPE at the one position the
 * drafting block holds constant.
 */

#ifndef __ASSISTANT_ATTENTION_H__
#define __ASSISTANT_ATTENTION_H__

#include <common_properties.h>
#include <fused_llm_properties.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <mha_core.h>
#include <node_exporter.h>

namespace causallm {

/**
 * @brief Gemma 4 assistant attention over shared target K/V
 */
class AssistantAttentionLayer final : public nntrainer::Layer {
public:
  AssistantAttentionLayer() :
    props(nntrainer::props::NumHeads(), props::NumHeads_KV(),
          nntrainer::props::RopeTheta(),
          nntrainer::props::RopePartialRotaryFactor()) {}

  void finalize(nntrainer::InitLayerContext &context) override;
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;
  void calcDerivative(nntrainer::RunLayerContext &context) override;
  bool supportBackwarding() const override { return false; }
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override {}
  const std::string getType() const override { return type; }
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @brief Point the layer at the target's K/V rows for the next forward.
   * @param k,v    n_rows rows of num_heads_kv * head_dim floats; must stay
   *               valid until the forward returns
   * @param pos    the position the queries are rotated to
   */
  void setSharedKV(const float *k, const float *v, unsigned int n_rows,
                   unsigned int pos) {
    k_ = k;
    v_ = v;
    n_rows_ = n_rows;
    pos_ = pos;
  }

  inline static const std::string type = "assistant_attention";

private:
  std::tuple<nntrainer::props::NumHeads, props::NumHeads_KV,
             nntrainer::props::RopeTheta,
             nntrainer::props::RopePartialRotaryFactor>
    props;
  const float *k_ = nullptr;
  const float *v_ = nullptr;
  unsigned int n_rows_ = 0;
  unsigned int pos_ = 0;
};

} // namespace causallm

#endif /* __ASSISTANT_ATTENTION_H__ */
