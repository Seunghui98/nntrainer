// SPDX-License-Identifier: Apache-2.0
/**
* Copyright (C) 2026 Hyeong-Gwon Hong
*
* @file   causal_conv1d_layer.cpp
* @date   01 April 2026
* @brief  Causal depthwise Conv1D layer for CausalLM
* @see    https://github.com/nntrainer/nntrainer
* @author Hyeong-Gwon Hong
* @bug    No known bugs except for NYI items
*
*/

#include "causal_conv1d_layer.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <cpu_backend.h>
#include <fp16.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>

namespace causallm {

CausalConv1DLayer::CausalConv1DLayer() : LayerImpl() {
  weight_idx.fill(std::numeric_limits<unsigned int>::max());
}

void CausalConv1DLayer::validateInputShape(
  const nntrainer::TensorDim &input_dim) const {
  NNTR_THROW_IF(input_dim.rank() != 4, std::invalid_argument)
    << "[CausalConv1DLayer] input rank must be 4, but got "
    << input_dim.rank();

  NNTR_THROW_IF(input_dim.channel() != 1, std::invalid_argument)
    << "[CausalConv1DLayer] input channel must be 1 for Bx1xHxW layout, but got "
    << input_dim.channel();

  NNTR_THROW_IF(input_dim.height() < 1 || input_dim.width() < 1,
                std::invalid_argument)
    << "[CausalConv1DLayer] invalid input shape: H and W must be positive.";

  NNTR_THROW_IF(input_dim.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "[CausalConv1DLayer] input dtype must be FP32.";
}

void CausalConv1DLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "[CausalConv1DLayer] requires exactly 1 input, but got "
    << context.getNumInputs();

  const nntrainer::TensorDim &input_dim = context.getInputDimensions()[0];
  validateInputShape(input_dim);

  const unsigned int W = input_dim.width();

#ifdef ENABLE_FP16
  nntrainer::TensorDim weight_dim(
    1, 1, KERNEL_SIZE, W, ml::train::TensorDim::DataType::FP16);
#else
  nntrainer::TensorDim weight_dim(
    1, 1, KERNEL_SIZE, W, ml::train::TensorDim::DataType::UINT16);
#endif

  nntrainer::TensorDim output_dim = input_dim;
  output_dim.setDataType(ml::train::TensorDim::DataType::FP32);

  context.setOutputDimensions({output_dim});

  context.requestWeight(weight_dim,
                        nntrainer::Initializer::NONE, nntrainer::WeightRegularizer::NONE,
                        0.0f, 0.0f, "causal_conv1d_weight");
}

void CausalConv1DLayer::forwarding(nntrainer::RunLayerContext &context,
                                   bool training) {
  throw std::runtime_error(
    "[CausalConv1DLayer] forwarding() is not used. "
    "Use incremental_forwarding().");
}

void CausalConv1DLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  NNTR_THROW_IF(training, std::invalid_argument)
    << "[CausalConv1DLayer] training/backward is not supported yet.";

  NNTR_THROW_IF(to == 0 || to <= from, std::invalid_argument)
    << "[CausalConv1DLayer] invalid incremental range: from=" << from
    << ", to=" << to;

  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &conv_weight = context.getWeight(weight_idx[weight]);

  const nntrainer::TensorDim &in_dim = input.getDim();

  const unsigned int B = in_dim.batch();
  const unsigned int H = in_dim.height();
  const unsigned int W = in_dim.width();

  NNTR_THROW_IF(input.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "[CausalConv1DLayer] input must be FP32.";

  NNTR_THROW_IF(output.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "[CausalConv1DLayer] output must be FP32.";

#ifdef ENABLE_FP16
  NNTR_THROW_IF(conv_weight.getDataType() !=
                  ml::train::TensorDim::DataType::FP16,
                std::invalid_argument)
    << "[CausalConv1DLayer] weight must be FP16.";
#else
  NNTR_THROW_IF(conv_weight.getDataType() !=
                  ml::train::TensorDim::DataType::UINT16,
                std::invalid_argument)
    << "[CausalConv1DLayer] weight must be UINT16 when ENABLE_FP16 is off.";
#endif

  NNTR_THROW_IF(!input.isContiguous(), std::invalid_argument)
    << "[CausalConv1DLayer] input tensor must be contiguous.";
  NNTR_THROW_IF(!output.isContiguous(), std::invalid_argument)
    << "[CausalConv1DLayer] output tensor must be contiguous.";
  NNTR_THROW_IF(!conv_weight.isContiguous(), std::invalid_argument)
    << "[CausalConv1DLayer] weight tensor must be contiguous.";
  NNTR_THROW_IF(to > H, std::invalid_argument)
    << "[CausalConv1DLayer] invalid incremental end: to=" << to
    << ", H=" << H;

  const size_t prefix_elems = static_cast<size_t>(B) * to * W;
  std::vector<uint16_t> input_fp16(prefix_elems);

  const float *input_ptr = input.getData<float>();
  for (size_t i = 0; i < prefix_elems; ++i) {
    input_fp16[i] = fp16_ieee_from_fp32_value(input_ptr[i]);
  }

#ifdef ENABLE_FP16
  const uint16_t *weight_ptr =
    reinterpret_cast<const uint16_t *>(conv_weight.getData<__fp16>());
#else
  const uint16_t *weight_ptr = conv_weight.getData<uint16_t>();
#endif

  nntrainer::causal_depthwise_conv1d_k3_fp16(
    input_fp16.data(), weight_ptr, output.getData<float>(), B, to, W);
}

void CausalConv1DLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw std::runtime_error(
    "[CausalConv1DLayer] calcDerivative() is not implemented. "
    "This layer is inference-only for now.");
}

void CausalConv1DLayer::calcGradient(nntrainer::RunLayerContext &context) {
  throw std::runtime_error(
    "[CausalConv1DLayer] calcGradient() is not implemented. "
    "This layer is inference-only for now.");
}

void CausalConv1DLayer::exportTo(
  nntrainer::Exporter &exporter,
  const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
}

} // namespace causallm

#ifdef PLUGGABLE
extern "C" {

nntrainer::Layer *create_causal_conv1d_layer() {
  return new causallm::CausalConv1DLayer();
}

void destroy_causal_conv1d_layer(nntrainer::Layer *layer) {
  delete layer;
}

nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_causal_conv1d_layer,
  destroy_causal_conv1d_layer,
  causallm::CausalConv1DLayer::type
};

}
#endif