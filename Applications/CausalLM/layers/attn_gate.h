// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   attn_gate.h
 * @date   8 April 2026
 * @brief  Attention output gate layer: output = input * sigmoid(gate)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @note   Used in Qwen3.5 self-attention where attn_output_gate=true.
 *         Takes two inputs: [0] attention output, [1] gate values.
 *         Computes: output = input[0] * sigmoid(input[1])
 */

#ifndef __ATTN_GATE_LAYER_H__
#define __ATTN_GATE_LAYER_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

namespace causallm {

WIN_EXPORT class AttnGateLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT AttnGateLayer() : Layer() {}
  WIN_EXPORT ~AttnGateLayer() {}

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
    return AttnGateLayer::type;
  };
  WIN_EXPORT void
  setProperty(const std::vector<std::string> &values) override {};
  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "attn_gate";
};

} // namespace causallm

#endif /* __ATTN_GATE_LAYER_H__ */
