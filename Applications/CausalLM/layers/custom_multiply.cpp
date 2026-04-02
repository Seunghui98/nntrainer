// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   custom_multiply.cpp
 * @date   02 April 2026
 * @brief  Custom multiply layer for CausalLM
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <custom_multiply.h>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <util_func.h>

namespace causallm {

void CustomMultiplyLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 2, std::invalid_argument)
    << "CustomMultiplyLayer requires exactly 2 inputs";

  const auto &input_dims = context.getInputDimensions();
  nntrainer::TensorDim out_dim = input_dims[0];
  nntrainer::TensorDim dim1 = input_dims[1];

  // Same broadcasting rule as MultiplyLayer
  for (unsigned int i = 0; i < ml::train::TensorDim::MAXDIM; ++i) {
    if (out_dim[i] != dim1[i]) {
      if (out_dim[i] == 1) {
        out_dim.setTensorDim(i, dim1[i]);
      } else if (dim1[i] != 1) {
        throw std::invalid_argument(
          "CustomMultiplyLayer: incompatible shapes for broadcasting at dim " +
          std::to_string(i) + " (" + std::to_string(out_dim[i]) + " vs " +
          std::to_string(dim1[i]) + ")");
      }
    }
  }

  context.setOutputDimensions({out_dim});
}

void CustomMultiplyLayer::forwarding(nntrainer::RunLayerContext &context,
                                     bool training) {
  nntrainer::Tensor &in0 = context.getInput(INPUT_IDX_0);
  nntrainer::Tensor &in1 = context.getInput(INPUT_IDX_1);
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);

  in0.multiply(in1, out);
}

void CustomMultiplyLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  // Avoid BinaryOperationLayer's generic incremental slicing, which assumes
  // the sequence axis is height. For CausalLM conv path tensors shaped
  // [B, C, 1, T], run full forwarding instead.
  forwarding(context, training);
}

void CustomMultiplyLayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  context.getOutgoingDerivative(INPUT_IDX_0).copy(
    context.getIncomingDerivative(OUT_IDX)
      .multiply(context.getInput(INPUT_IDX_1)));

  context.getOutgoingDerivative(INPUT_IDX_1).copy(
    context.getIncomingDerivative(OUT_IDX)
      .multiply(context.getInput(INPUT_IDX_0)));
}

void CustomMultiplyLayer::setProperty(
  const std::vector<std::string> &values) {
  NNTR_THROW_IF(!values.empty(), std::invalid_argument)
    << "[CustomMultiplyLayer] Unknown Layer Properties count "
    << std::to_string(values.size());
}

#ifdef PLUGGABLE

nntrainer::Layer *create_custom_multiply_layer() {
  auto layer = new CustomMultiplyLayer();
  return layer;
}

void destroy_custom_multiply_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_custom_multiply_layer,
  destroy_custom_multiply_layer};
}

#endif

} // namespace causallm
