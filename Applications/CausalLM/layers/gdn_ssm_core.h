// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   gdn_ssm_core.h
 * @date   9 April 2026
 * @brief  GDN SSM Core layer: state-space model core for GatedDeltaNet
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @note   Handles the core SSM computation:
 *         1. Split conv_out → Q, K, V
 *         2. L2 normalize Q, K per head
 *         3. Scale Q
 *         4. Compute decay and beta from a, b projections
 *         5. SSM state update: S = decay*S + k⊗(beta*(v - S^T@k))
 *         6. Output: o = S^T @ q
 *         7. Gated RMS norm: rms_norm(o) * silu(z)
 *
 *   Inputs (4):
 *     [0] conv_out: (batch, 1, seq_len, conv_dim)   - Q,K,V concatenated
 *     [1] a:        (batch, 1, seq_len, num_v_heads) - decay projection
 *     [2] b:        (batch, 1, seq_len, num_v_heads) - gate projection
 *     [3] z:        (batch, 1, seq_len, value_dim)   - output gate
 *   Output: (batch, 1, seq_len, value_dim)
 *
 *   Weights: A_log (num_v_heads), dt_bias (num_v_heads), norm (head_v_dim)
 *   State:   recurrent_state (batch, num_v_heads * head_k_dim, head_v_dim)
 */

#ifndef __GDN_SSM_CORE_LAYER_H__
#define __GDN_SSM_CORE_LAYER_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

#include <base_properties.h>

namespace causallm {

namespace props {

// Reuse NumVHeads, HeadKDim, HeadVDim from gated_delta_net.h
// But define here if needed independently
class SSMNumVHeads : public nntrainer::PositiveIntegerProperty {
public:
  SSMNumVHeads(unsigned int value = 16) { set(value); };
  static constexpr const char *key = "ssm_num_v_heads";
  using prop_tag = nntrainer::uint_prop_tag;
};

class SSMHeadKDim : public nntrainer::PositiveIntegerProperty {
public:
  SSMHeadKDim(unsigned int value = 128) { set(value); };
  static constexpr const char *key = "ssm_head_k_dim";
  using prop_tag = nntrainer::uint_prop_tag;
};

class SSMHeadVDim : public nntrainer::PositiveIntegerProperty {
public:
  SSMHeadVDim(unsigned int value = 128) { set(value); };
  static constexpr const char *key = "ssm_head_v_dim";
  using prop_tag = nntrainer::uint_prop_tag;
};

} // namespace props

WIN_EXPORT class GdnSsmCoreLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT GdnSsmCoreLayer();
  WIN_EXPORT ~GdnSsmCoreLayer() = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT bool supportBackwarding() const override { return false; };
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override {};
  WIN_EXPORT const std::string getType() const override {
    return GdnSsmCoreLayer::type;
  };
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;
  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "gdn_ssm_core";

private:
  std::tuple<props::SSMNumVHeads, props::SSMHeadKDim, props::SSMHeadVDim>
    ssm_props;

  unsigned int num_v_heads;
  unsigned int head_k_dim;
  unsigned int head_v_dim;
  unsigned int key_dim;
  unsigned int value_dim;

  enum WeightIdx { W_A_LOG = 0, W_DT_BIAS, W_NORM, NUM_WEIGHTS };
  std::array<unsigned int, NUM_WEIGHTS> wt_idx;
  unsigned int state_idx;   // recurrent_state
  unsigned int ws_qkv_idx;  // workspace: qkv_buf (conv_dim)
  unsigned int ws_kv_idx;   // workspace: kv_mem (head_v_dim)
  unsigned int ws_delta_idx; // workspace: delta (head_v_dim)
};

} // namespace causallm

#endif /* __GDN_SSM_CORE_LAYER_H__ */
