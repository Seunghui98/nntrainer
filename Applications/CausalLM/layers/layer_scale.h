// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   layer_scale.h
 * @date   18 Aug 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  LayerScale layer for the PE-Lang-L14-448 encoder (EVA ViT family).
 *
 * Fused residual + per-channel LayerScale gamma multiplication:
 *   out = residual + gamma ⊙ input
 *
 * Combines `mul` + `addition` into one node, saving one 1025*1024 buffer
 * round-trip per attention/MLP sub-block (23*2 = 46 trips in the encoder,
 * ~184 MB traffic — plan §3.3b).
 *
 * Two inputs (input, residual), one weight (gamma [C]). All share the same
 * per-row channel count C (= embed_dim, broadcast across the token axis).
 */

#ifndef __LAYER_SCALE_LAYER_H__
#define __LAYER_SCALE_LAYER_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>
#include <utility>

#include <causallm_common_properties.h>
#include <common_properties.h>
#include <connection.h>
#include <tensor.h>
#include <tensor_wrap_specs.h>

namespace causallm {

namespace props {

/**
 * @brief (Optional) per-row channel override for LayerScale gamma. Defaults to
 *        the input's last dim — set when the graph needs to pin it before the
 *        input shape is known (rare).
 */
class LsNumChannels : public nntrainer::Property<unsigned int> {
public:
  LsNumChannels(unsigned int value = 0) { set(value); }
  static constexpr const char *key = "num_channels";
  using prop_tag = nntrainer::uint_prop_tag;
};

} // namespace props

/**
 * @brief Fused residual + per-channel-scale layer.
 *
 *   out = residual + gamma ⊙ input
 *
 * Inputs (order):
 *   [0] input    — the sub-block output to be scaled
 *   [1] residual — the skip-connection tensor
 * Weight:
 *   [0] gamma — [C], named `gamma` (excluded from Q8_0 by dtype map §3.5)
 */
WIN_EXPORT class LayerScaleLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT LayerScaleLayer() : Layer(), wt_idx({0}) {}
  WIN_EXPORT ~LayerScaleLayer() {}

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT bool supportBackwarding() const override { return false; }
  WIN_EXPORT void exportTo(
    nntrainer::Exporter &exporter,
    const ml::train::ExportMethods &method) const override {}
  WIN_EXPORT const std::string getType() const override {
    return LayerScaleLayer::type;
  }
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override {
    auto remain = loadProperties(values, layer_scale_props);
    NNTR_THROW_IF(!remain.empty(), std::invalid_argument)
      << "[layer_scale] Unknown Layer Properties count "
      << std::to_string(values.size());
  }
  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "layer_scale";

private:
  // Weight offsets requested in finalize().
  //   [0] gamma
  std::array<unsigned int, 1> wt_idx;
  std::tuple<props::LsNumChannels, nntrainer::props::SkipPrefill>
    layer_scale_props;
  bool skip_prefill = false;

  // Apply `out = residual + gamma * input` for the input slice [from, to).
  void runScale(nntrainer::RunLayerContext &context, unsigned int from,
                unsigned int to);
};

} // namespace causallm

#endif /* __LAYER_SCALE_LAYER_H__ */
