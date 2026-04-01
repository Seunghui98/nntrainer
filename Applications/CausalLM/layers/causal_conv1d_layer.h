// SPDX-License-Identifier: Apache-2.0
/**
* Copyright (C) 2026 Hyeong-Gwon Hong
*
* @file   causal_conv1d_layer.h
* @date   01 April 2026
* @brief  Causal depthwise Conv1D layer for CausalLM
* @see    https://github.com/nntrainer/nntrainer
* @author Hyeong-Gwon Hong
* @bug    No known bugs except for NYI items
*
*/

#ifndef __CAUSAL_LM_CAUSAL_CONV1D_LAYER_H__
#define __CAUSAL_LM_CAUSAL_CONV1D_LAYER_H__

#include <array>
#include <limits>
#include <string>

#include <layer_context.h>
#include <layer_impl.h>
#include <node_exporter.h>
#include <tensor_dim.h>

namespace causallm {

class CausalConv1DLayer : public nntrainer::LayerImpl {
public:
  CausalConv1DLayer();
  ~CausalConv1DLayer() override = default;

  void finalize(nntrainer::InitLayerContext &context) override;
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;

  void calcDerivative(nntrainer::RunLayerContext &context) override;
  void calcGradient(nntrainer::RunLayerContext &context) override;

  const std::string getType() const override;
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  bool supportBackwarding() const override { return false; }

  inline static const std::string type = "causal_conv1d";

private:
  enum CausalConv1DParams {
    weight = 0,
  };

  static constexpr size_t SINGLE_INOUT_IDX = 0;
  static constexpr unsigned int KERNEL_SIZE = 3;

  std::array<unsigned int, 1> weight_idx;

  void validateInputShape(const nntrainer::TensorDim &input_dim) const;
};

} // namespace causallm

#endif // __CAUSAL_LM_CAUSAL_CONV1D_LAYER_H__