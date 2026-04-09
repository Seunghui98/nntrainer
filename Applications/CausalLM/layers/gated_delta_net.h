// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   gated_delta_net.h
 * @date   7 April 2026
 * @brief  Gated DeltaNet (Linear Attention) layer for Qwen3.5 hybrid model
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This layer implements the Gated DeltaNet linear attention mechanism
 *         used in Qwen3.5's hybrid architecture. It replaces standard
 *         self-attention in most layers with a recurrent state-space model.
 *
 *   Reference: https://github.com/huggingface/transformers/blob/main/
 *              src/transformers/models/qwen3_5/modeling_qwen3_5.py
 *
 *   Computation flow (inference, single token):
 *     input → in_proj_qkv → conv1d → SiLU → split(Q,K,V)
 *           → L2 norm(Q,K) → expand heads
 *           → compute decay g = -exp(A_log) * softplus(a + dt_bias)
 *           → compute beta = sigmoid(b)
 *           → state update: S = exp(g)*S + outer(k, beta*(v - S^T k))
 *           → output: o = S @ q
 *           → gated_rms_norm(o, z) → out_proj → output
 */

#ifndef __GATED_DELTA_NET_LAYER_H__
#define __GATED_DELTA_NET_LAYER_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>
#include <tensor.h>

#include <base_properties.h>
#include <causallm_common_properties.h>

namespace causallm {

namespace props {

class NumVHeads : public nntrainer::PositiveIntegerProperty {
public:
  NumVHeads(unsigned int value = 16) { set(value); };
  static constexpr const char *key = "num_v_heads";
  using prop_tag = nntrainer::uint_prop_tag;
};

class HeadKDim : public nntrainer::PositiveIntegerProperty {
public:
  HeadKDim(unsigned int value = 128) { set(value); };
  static constexpr const char *key = "head_k_dim";
  using prop_tag = nntrainer::uint_prop_tag;
};

class HeadVDim : public nntrainer::PositiveIntegerProperty {
public:
  HeadVDim(unsigned int value = 128) { set(value); };
  static constexpr const char *key = "head_v_dim";
  using prop_tag = nntrainer::uint_prop_tag;
};

class ConvKernel : public nntrainer::PositiveIntegerProperty {
public:
  ConvKernel(unsigned int value = 4) { set(value); };
  static constexpr const char *key = "conv_kernel";
  using prop_tag = nntrainer::uint_prop_tag;
};

} // namespace props

/**
 * @brief Gated DeltaNet Layer
 *
 * This is a monolithic layer that implements the full Gated DeltaNet
 * linear attention block. It contains all projections and the SSM state
 * internally, managed as weights and tensors.
 *
 * Weights (loaded from file):
 *   0: in_proj_qkv  (hidden_size, conv_dim) where conv_dim = 2*key_dim+value_dim
 *   1: conv1d        (conv_dim, 1, conv_kernel)
 *   2: A_log          (num_v_heads)
 *   3: dt_bias        (num_v_heads)
 *   4: in_proj_a     (hidden_size, num_v_heads)
 *   5: in_proj_b     (hidden_size, num_v_heads)
 *   6: in_proj_z     (hidden_size, value_dim)
 *   7: norm           (head_v_dim) - RMS norm weight
 *   8: out_proj      (value_dim, hidden_size)
 *
 * Tensors (internal state):
 *   0: recurrent_state (batch, num_v_heads, head_k_dim, head_v_dim)
 *   1: conv_state      (batch, conv_dim, conv_kernel-1)
 */
WIN_EXPORT class GatedDeltaNetLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT GatedDeltaNetLayer();
  WIN_EXPORT ~GatedDeltaNetLayer() = default;

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
    return GatedDeltaNetLayer::type;
  };
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;
  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "gated_delta_net";

private:
  /** Properties */
  std::tuple<props::NumVHeads, props::HeadKDim, props::HeadVDim,
             props::ConvKernel>
    gdn_props;

  /** Cached dimensions */
  unsigned int num_v_heads;
  unsigned int head_k_dim;
  unsigned int head_v_dim;
  unsigned int conv_kernel;
  unsigned int key_dim;   // num_v_heads * head_k_dim
  unsigned int value_dim; // num_v_heads * head_v_dim
  unsigned int conv_dim;  // 2 * key_dim + value_dim
  unsigned int hidden_size;

  /** Weight indices */
  enum WeightIdx {
    W_IN_PROJ_QKV = 0,
    W_CONV1D,
    W_A_LOG,
    W_DT_BIAS,
    W_IN_PROJ_A,
    W_IN_PROJ_B,
    W_IN_PROJ_Z,
    W_NORM,
    W_OUT_PROJ,
    NUM_WEIGHTS
  };
  std::array<unsigned int, NUM_WEIGHTS> wt_idx;

  /** Tensor indices for internal state */
  enum TensorIdx {
    T_RECURRENT_STATE = 0,
    T_CONV_STATE,
    NUM_TENSORS
  };
  std::array<unsigned int, NUM_TENSORS> tensor_idx;

  /** Helper functions */
  void recurrent_forward(nntrainer::RunLayerContext &context,
                         unsigned int batch);
  void prefill_forward(nntrainer::RunLayerContext &context, unsigned int batch,
                       unsigned int seq_len);
};

} // namespace causallm

#endif /* __GATED_DELTA_NET_LAYER_H__ */
