// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   transpose_layer.h
 * @date   02 April 2026
 * @brief  This is configurable transpose layer class
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __TRANSPOSE_LAYER_H__
#define __TRANSPOSE_LAYER_H__

#ifdef __cplusplus

#pragma once

#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <array>
#include <string>
#include <vector>

#include <layer_devel.h>
#include <layer_impl.h>
#include <node_exporter.h>

namespace causallm {

/**
 * @class   TransposeLayer
 * @brief   Configurable transpose layer
 */
WIN_EXPORT class TransposeLayer : public nntrainer::LayerImpl {
public:
  WIN_EXPORT TransposeLayer();
  WIN_EXPORT ~TransposeLayer() = default;

  WIN_EXPORT TransposeLayer(TransposeLayer &&rhs) noexcept = default;
  WIN_EXPORT TransposeLayer &operator=(TransposeLayer &&rhs) = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;

  WIN_EXPORT const std::string getType() const override {
    return TransposeLayer::type;
  }

  WIN_EXPORT bool supportBackwarding() const override { return false; }

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  using Layer::setProperty;
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "custom_transpose";

private:
  std::string perm_str_;
  std::array<size_t, 3> perm_;

  void parsePerm();
};

} // namespace causallm

#endif // __cplusplus
#endif // __TRANSPOSE_LAYER_H__
