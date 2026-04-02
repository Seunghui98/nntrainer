// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   custom_slice.cpp
 * @date   02 April 2026
 * @brief  Custom slice layer for CausalLM
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <custom_slice.h>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <util_func.h>

namespace causallm {

void CustomSliceLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "CustomSliceLayer requires exactly 1 input";

  axis = std::get<nntrainer::props::Axis>(slice_props).get();
  start = std::get<nntrainer::props::StartIndex>(slice_props).get() - 1;
  unsigned int end = std::get<nntrainer::props::EndIndex>(slice_props).get() - 1;

  const nntrainer::TensorDim &in_dim = context.getInputDimensions()[0];
  nntrainer::TensorDim out_dim = in_dim;

  NNTR_THROW_IF(axis >= ml::train::TensorDim::MAXDIM, std::invalid_argument)
    << "CustomSliceLayer: invalid axis " << axis;

  NNTR_THROW_IF(end < start, std::invalid_argument)
    << "CustomSliceLayer: end_index must be greater than start_index";

  NNTR_THROW_IF(end > in_dim.getTensorDim(axis), std::invalid_argument)
    << "CustomSliceLayer: end_index exceeds input dimension size";

  out_dim.setTensorDim(axis, end - start);
  context.setOutputDimensions({out_dim});
}

void CustomSliceLayer::forwarding(nntrainer::RunLayerContext &context,
                                  bool training) {
  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  for (unsigned int b = 0; b < output.batch(); ++b) {
    for (unsigned int c = 0; c < output.channel(); ++c) {
      for (unsigned int h = 0; h < output.height(); ++h) {
        for (unsigned int w = 0; w < output.width(); ++w) {
          unsigned int c_idx = (axis == 1) ? c + start : c;
          unsigned int h_idx = (axis == 2) ? h + start : h;
          unsigned int w_idx = (axis == 3) ? w + start : w;
          output.setValue(b, c, h, w, input.getValue(b, c_idx, h_idx, w_idx));
        }
      }
    }
  }
}

void CustomSliceLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  // Avoid UnaryOperationLayer's generic incremental slicing, which assumes
  // the sequence axis is height. For CausalLM conv path tensors shaped
  // [B, C, 1, T], run full forwarding instead.
  forwarding(context, training);
}

void CustomSliceLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  const nntrainer::Tensor &in_deriv =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out_deriv =
    context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  for (unsigned int b = 0; b < in_deriv.batch(); ++b) {
    for (unsigned int c = 0; c < in_deriv.channel(); ++c) {
      for (unsigned int h = 0; h < in_deriv.height(); ++h) {
        for (unsigned int w = 0; w < in_deriv.width(); ++w) {
          unsigned int c_idx = (axis == 1) ? c + start : c;
          unsigned int h_idx = (axis == 2) ? h + start : h;
          unsigned int w_idx = (axis == 3) ? w + start : w;
          out_deriv.setValue(b, c_idx, h_idx, w_idx,
                             in_deriv.getValue(b, c, h, w));
        }
      }
    }
  }
}

void CustomSliceLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, slice_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[CustomSliceLayer] Unknown Layer Properties count "
    << std::to_string(values.size());
}

#ifdef PLUGGABLE

nntrainer::Layer *create_custom_slice_layer() {
  auto layer = new CustomSliceLayer();
  return layer;
}

void destroy_custom_slice_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_custom_slice_layer,
  destroy_custom_slice_layer};
}

#endif

} // namespace causallm
