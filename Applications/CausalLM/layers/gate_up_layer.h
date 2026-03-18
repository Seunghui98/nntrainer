// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   gate_up_layer.h
 * @date   17 March 2026
 * @brief  Fused Gate+Up projection layer for FFN (SwiGLU)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __GATE_UP_LAYER_H__
#define __GATE_UP_LAYER_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <common_properties.h>
#include <layer_impl.h>

namespace causallm {

namespace props {

class UpUnit : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "up_unit";
  using prop_tag = nntrainer::uint_prop_tag;
};

class GateUnit : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "gate_unit";
  using prop_tag = nntrainer::uint_prop_tag;
};

} // namespace props

/**
 * @class   GateUpLayer
 * @brief   Fused gate_proj + up_proj layer for FFN.
 *          Performs multi-weight GEMV with single input quantization.
 */
WIN_EXPORT class GateUpLayer : public nntrainer::LayerImpl {
public:
  WIN_EXPORT GateUpLayer();

  WIN_EXPORT ~GateUpLayer() = default;

  WIN_EXPORT GateUpLayer(GateUpLayer &&rhs) noexcept = default;

  WIN_EXPORT GateUpLayer &operator=(GateUpLayer &&rhs) = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  WIN_EXPORT void calcGradient(nntrainer::RunLayerContext &context) override;

  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;

  WIN_EXPORT const std::string getType() const override {
    return GateUpLayer::type;
  };

  WIN_EXPORT bool supportBackwarding() const override { return true; }

  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "gate_up_layer";

private:
  enum Params { UP, GATE };
  std::tuple<props::UpUnit, props::GateUnit> gate_up_props;
  std::array<unsigned int, 2> weight_idx; /**< indices of the weights */
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __GATE_UP_LAYER_H__ */
