// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   causal_conv1d.h
 * @date   9 April 2026
 * @brief  Causal Conv1D layer with circular buffer + channel SIMD vectorization
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
 * Optimizations:
 * - Circular buffer state (no shift-left per token)
 * - Transposed kernel/state layout (kernel_size, channels) for channel SIMD
 * - AVX2/NEON vectorized across channels
 *
 * Weights:
 *   0: conv_kernel - original (channels, kernel_size), transposed on load
 *
 * Tensors:
 *   0: conv_state (kernel_size-1, channels) - transposed layout
 *   1: write_pos (1) - circular buffer position
 *   2: kernel_transposed (kernel_size, channels) - transposed layout for SIMD
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
  unsigned int wt_idx;         // original kernel weight
  unsigned int state_idx;      // circular buffer state
  unsigned int kernel_t_idx;   // transposed kernel tensor
  bool kernel_transposed;      // flag for first-run transpose
  unsigned int write_pos;      // circular buffer write position (per-instance)
};

} // namespace causallm

#endif /* __CAUSAL_CONV1D_LAYER_H__ */
