// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   custom_slice.h
 * @date   02 April 2026
 * @brief  Custom slice layer for CausalLM
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __CUSTOM_SLICE_LAYER_H__
#define __CUSTOM_SLICE_LAYER_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

namespace causallm {

/**
 * @brief Custom slice layer.
 *
 * @note Uses the same properties as nntrainer's slice layer:
 *       - axis
 *       - start_index
 *       - end_index
 *
 *       incremental_forwarding() is overridden to avoid the generic unary-op
 *       incremental slicing assumption on height axis.
 */
WIN_EXPORT class CustomSliceLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT CustomSliceLayer() = default;
  WIN_EXPORT ~CustomSliceLayer() = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  WIN_EXPORT bool supportBackwarding() const override { return true; }

  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override {
    exporter.saveResult(slice_props, method, this);
  }

  WIN_EXPORT const std::string getType() const override {
    return CustomSliceLayer::type;
  }

  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "custom_slice";

private:
  static constexpr size_t SINGLE_INOUT_IDX = 0;

  unsigned int axis = 0;
  unsigned int start = 0;
  std::tuple<nntrainer::props::Axis,
             nntrainer::props::StartIndex,
             nntrainer::props::EndIndex>
    slice_props;
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __CUSTOM_SLICE_LAYER_H__ */
