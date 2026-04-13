// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   lfm2_conv.h
 * @date   13 April 2026
 * @brief  LFM2 double-gated short convolution layer
 * @note   Implements: in_proj(3H) -> split B,C,x -> B*x -> depthwise_conv1d ->
 *         C*conv_out -> out_proj
 */

#ifndef __LFM2_CONV_LAYER_H__
#define __LFM2_CONV_LAYER_H__

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
class LFM2HiddenSize : public nntrainer::PositiveIntegerProperty {
public:
  LFM2HiddenSize(unsigned int value = 1024) { set(value); };
  static constexpr const char *key = "hidden_size";
  using prop_tag = nntrainer::uint_prop_tag;
};
class LFM2ConvKernel : public nntrainer::PositiveIntegerProperty {
public:
  LFM2ConvKernel(unsigned int value = 3) { set(value); };
  static constexpr const char *key = "conv_kernel_size";
  using prop_tag = nntrainer::uint_prop_tag;
};
} // namespace props

/**
 * @brief LFM2 Double-Gated Short Convolution Layer
 *
 * Forward pass:
 *   BCx = in_proj(input)             // (hidden, 3*hidden)
 *   B, C, x = split(BCx, 3)
 *   Bx = B * x                      // input gating
 *   conv_out = depthwise_conv1d(Bx)  // causal conv (circular buffer)
 *   y = C * conv_out                 // output gating
 *   output = out_proj(y)             // (hidden, hidden)
 *
 * Weights:
 *   0: in_proj  (hidden_size, 3*hidden_size)
 *   1: conv     (hidden_size, conv_kernel_size)
 *   2: out_proj (hidden_size, hidden_size)
 *
 * Tensors:
 *   0: conv_state (hidden_size, conv_kernel_size) - circular buffer
 */
WIN_EXPORT class LFM2ConvLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT LFM2ConvLayer();
  WIN_EXPORT ~LFM2ConvLayer() = default;

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
    return LFM2ConvLayer::type;
  };
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "lfm2_conv";

private:
  std::tuple<props::LFM2HiddenSize, props::LFM2ConvKernel> lfm2_props;
  unsigned int hidden_size;
  unsigned int conv_kernel_size;
  unsigned int write_pos; // circular buffer position (per-instance)
};

} // namespace causallm

#endif /* __LFM2_CONV_LAYER_H__ */
