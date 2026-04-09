// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   causal_conv1d.h
 * @date   9 April 2026
 * @brief  Causal Conv1D layer with sliding window state for GatedDeltaNet
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @note   1D depthwise causal convolution + SiLU activation.
 *         Maintains conv_state across incremental forwarding calls.
 *         Input:  (batch, 1, seq_len, channels)
 *         Output: (batch, 1, seq_len, channels)
 */

#ifndef __CAUSAL_CONV1D_LAYER_H__
#define __CAUSAL_CONV1D_LAYER_H__

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

class ConvChannels : public nntrainer::PositiveIntegerProperty {
public:
  ConvChannels(unsigned int value = 1) { set(value); };
  static constexpr const char *key = "conv_channels";
  using prop_tag = nntrainer::uint_prop_tag;
};

class ConvKernelSize : public nntrainer::PositiveIntegerProperty {
public:
  ConvKernelSize(unsigned int value = 4) { set(value); };
  static constexpr const char *key = "conv_kernel_size";
  using prop_tag = nntrainer::uint_prop_tag;
};

} // namespace props

/**
 * @brief Causal Conv1D Layer
 *
 * Depthwise 1D convolution with causal padding (state-based).
 * For each channel independently:
 *   output[ch] = dot(conv_state[ch] ++ input[ch], kernel[ch]) then SiLU
 *
 * Weights:
 *   0: conv_kernel (1, 1, channels, kernel_size)
 *
 * Tensors (internal state):
 *   0: conv_state (batch, 1, channels, kernel_size - 1)
 */
WIN_EXPORT class CausalConv1dLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT CausalConv1dLayer();
  WIN_EXPORT ~CausalConv1dLayer() = default;

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
    return CausalConv1dLayer::type;
  };
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;
  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "causal_conv1d";

private:
  std::tuple<props::ConvChannels, props::ConvKernelSize> conv_props;
  unsigned int channels;
  unsigned int kernel_size;
  unsigned int wt_idx;
  unsigned int state_idx;
};

} // namespace causallm

#endif /* __CAUSAL_CONV1D_LAYER_H__ */
