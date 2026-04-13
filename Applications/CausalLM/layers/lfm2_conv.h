// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   lfm2_conv.h
 * @date   13 April 2026
 * @brief  LFM2 gated convolution layer (lightweight: gating + conv only)
 * @note   The heavy in_proj and out_proj are separate FC layers.
 *         This layer only does: split B,C,x -> B*x -> depthwise_conv1d -> C*conv
 *
 *         Input:  (batch, seq, 3 * hidden_size) from in_proj FC
 *         Output: (batch, seq, hidden_size)
 *
 *         Weight[0]: conv kernel (hidden_size, conv_kernel_size)
 *         Tensor[0]: conv_state circular buffer (hidden_size, conv_kernel_size)
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
 * @brief LFM2 Gated Convolution Layer (lightweight)
 *
 * Forward pass (input is already projected by FC in_proj):
 *   input shape: (batch, seq, 3 * hidden_size)
 *   B, C, x = split(input, 3)   // each (batch, seq, hidden_size)
 *   Bx = B * x                   // input gating
 *   conv_out = depthwise_conv1d(Bx)
 *   output = C * conv_out         // output gating
 *   output shape: (batch, seq, hidden_size)
 *
 * Weights:
 *   0: conv kernel (hidden_size, conv_kernel_size)
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
  unsigned int write_pos;
};

} // namespace causallm

#endif /* __LFM2_CONV_LAYER_H__ */
