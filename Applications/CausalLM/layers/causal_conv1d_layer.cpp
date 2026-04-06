// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Hyeong-Gwon Hong
 *
 * @file   causal_conv1d_layer.cpp
 * @date   01 April 2026
 * @brief  Causal depthwise Conv1D layer with conv-state cache for CausalLM
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeong-Gwon Hong
 * @bug    No known bugs except for NYI items
 */

#include "causal_conv1d_layer.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <nntrainer_error.h>
#include <nntrainer_log.h>

namespace causallm {

CausalConv1DLayer::CausalConv1DLayer() : LayerImpl() {
  weight_idx.fill(std::numeric_limits<unsigned int>::max());
  tensor_idx.fill(std::numeric_limits<unsigned int>::max());
}

void CausalConv1DLayer::validateInputShape(
  const nntrainer::TensorDim &input_dim) const {
  NNTR_THROW_IF(input_dim.channel() != 1, std::invalid_argument)
    << "[CausalConv1DLayer] input channel must be 1 (B×1×T×W layout), got "
    << input_dim.channel();
  NNTR_THROW_IF(input_dim.height() < 1 || input_dim.width() < 1,
                std::invalid_argument)
    << "[CausalConv1DLayer] invalid input shape: H and W must be positive.";
}

void CausalConv1DLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "[CausalConv1DLayer] requires exactly 1 input, got "
    << context.getNumInputs();

  const nntrainer::TensorDim &in_dim = context.getInputDimensions()[0];
  validateInputShape(in_dim);

  const unsigned int B = in_dim.batch();
  const unsigned int W = in_dim.width(); // number of features/channels

  // Weight: [1, 1, KERNEL_SIZE, W] FP32
  //   Row k  (offset k*W) = kernel weights for position k:
  //     k=0 → w0: applied to current token x_t
  //     k=1 → w1: applied to x_{t-1}
  //     k=2 → w2: applied to x_{t-2}
  nntrainer::TensorDim weight_dim(
    1, 1, KERNEL_SIZE, W, ml::train::TensorDim::DataType::FP32);
  weight_idx[weight] =
    context.requestWeight(weight_dim, nntrainer::Initializer::NONE,
                          nntrainer::WeightRegularizer::NONE, 0.0f, 0.0f,
                          "causal_conv1d_weight", false);

  // Conv-state cache: [B, 1, KERNEL_SIZE-1, W] FP32
  //   state[b, 0, 0, f] = x_{t-2}
  //   state[b, 0, 1, f] = x_{t-1}
  nntrainer::TensorDim state_dim(
    B, 1, KERNEL_SIZE - 1, W, ml::train::TensorDim::DataType::FP32);
  tensor_idx[conv_state] =
    context.requestTensor(state_dim, "conv_state",
                          nntrainer::Initializer::ZEROS, false,
                          nntrainer::TensorLifespan::MAX_LIFESPAN);

  // Output has same shape as input
  context.setOutputDimensions({in_dim});
}

void CausalConv1DLayer::forwarding(nntrainer::RunLayerContext &context,
                                   bool training) {
  throw std::runtime_error(
    "[CausalConv1DLayer] forwarding() is not used – call "
    "incremental_forwarding() instead.");
}

void CausalConv1DLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {

  NNTR_THROW_IF(training, std::invalid_argument)
    << "[CausalConv1DLayer] training/backward is not supported.";
  NNTR_THROW_IF(to == 0 || to <= from, std::invalid_argument)
    << "[CausalConv1DLayer] invalid range: from=" << from << ", to=" << to;

  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &w_tensor = context.getWeight(weight_idx[weight]);
  nntrainer::Tensor &state = context.getTensor(tensor_idx[conv_state]);

  const unsigned int B = input.batch();
  const unsigned int H = input.height(); // full sequence length (INIT_SEQ_LEN)
  const unsigned int W = input.width();  // feature dimension

  NNTR_THROW_IF(to > H, std::invalid_argument)
    << "[CausalConv1DLayer] to=" << to << " exceeds H=" << H;

  const float *w_ptr = w_tensor.getData<float>(); // [KERNEL_SIZE, W]
  const float *w0 = w_ptr;           // kernel for current  token  (k=0)
  const float *w1 = w_ptr + W;       // kernel for token at t-1    (k=1)
  const float *w2 = w_ptr + 2 * W;   // kernel for token at t-2    (k=2)

  float *state_data = state.getData<float>(); // [B, 1, KERNEL_SIZE-1, W]

  if (to - from == 1) {
    // ----------------------------------------------------------------
    // Single-token decode path: O(W) per step.
    // Use the persistent state for positions t-1 and t-2.
    // ----------------------------------------------------------------
    const unsigned int t = from; // absolute position to compute

    for (unsigned int b = 0; b < B; ++b) {
      const float *x = input.getData<float>() + b * H * W;
      float *y = output.getData<float>() + b * H * W;
      // state row layout: [s0=x_{t-2}, s1=x_{t-1}] each of width W
      const float *s0 = state_data + b * (KERNEL_SIZE - 1) * W;       // x_{t-2}
      const float *s1 = state_data + b * (KERNEL_SIZE - 1) * W + W;   // x_{t-1}

      for (unsigned int f = 0; f < W; ++f) {
        y[t * W + f] =
          w0[f] * x[t * W + f] + w1[f] * s1[f] + w2[f] * s0[f];
      }
    }

    // Update state: shift left and insert x_t
    //   new state[0] = old state[1]  (x_{t-1} becomes x_{t-2})
    //   new state[1] = x_t            (current becomes the new x_{t-1})
    for (unsigned int b = 0; b < B; ++b) {
      float *s = state_data + b * (KERNEL_SIZE - 1) * W;
      const float *x = input.getData<float>() + b * H * W + from * W;
      // Shift: s[0] <- s[1]
      std::memcpy(s, s + W, W * sizeof(float));
      // Insert current: s[1] <- x[t]
      std::memcpy(s + W, x, W * sizeof(float));
    }

  } else {
    // ----------------------------------------------------------------
    // Prefill path: compute all positions [0, to).
    // ----------------------------------------------------------------
    for (unsigned int b = 0; b < B; ++b) {
      const float *x = input.getData<float>() + b * H * W;
      float *y = output.getData<float>() + b * H * W;

      for (unsigned int t = 0; t < to; ++t) {
        const float *cur = x + t * W;
        const float *p1 = (t >= 1) ? x + (t - 1) * W : nullptr;
        const float *p2 = (t >= 2) ? x + (t - 2) * W : nullptr;

        for (unsigned int f = 0; f < W; ++f) {
          float val = w0[f] * cur[f];
          if (p1) val += w1[f] * p1[f];
          if (p2) val += w2[f] * p2[f];
          y[t * W + f] = val;
        }
      }
    }

    // Save last KERNEL_SIZE-1 positions to state for subsequent decode steps
    for (unsigned int b = 0; b < B; ++b) {
      float *s = state_data + b * (KERNEL_SIZE - 1) * W;
      const float *x = input.getData<float>() + b * H * W;

      // state[0] = x_{to-2}  (or zeros if to < 2)
      if (to >= 2)
        std::memcpy(s, x + (to - 2) * W, W * sizeof(float));
      else
        std::memset(s, 0, W * sizeof(float));

      // state[1] = x_{to-1}
      std::memcpy(s + W, x + (to - 1) * W, W * sizeof(float));
    }
  }
}

void CausalConv1DLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw std::runtime_error(
    "[CausalConv1DLayer] calcDerivative() not implemented (inference only).");
}

void CausalConv1DLayer::calcGradient(nntrainer::RunLayerContext &context) {
  throw std::runtime_error(
    "[CausalConv1DLayer] calcGradient() not implemented (inference only).");
}

void CausalConv1DLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  // No dynamic updates needed
}

void CausalConv1DLayer::exportTo(
  nntrainer::Exporter &exporter,
  const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
}

void CausalConv1DLayer::setProperty(const std::vector<std::string> &values) {
  if (!values.empty()) {
    throw std::invalid_argument(
      "[CausalConv1DLayer] Unknown layer properties: " + values[0]);
  }
}

} // namespace causallm

#ifdef PLUGGABLE
extern "C" {

nntrainer::Layer *create_causal_conv1d_layer() {
  return new causallm::CausalConv1DLayer();
}

void destroy_causal_conv1d_layer(nntrainer::Layer *layer) { delete layer; }

nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_causal_conv1d_layer, destroy_causal_conv1d_layer,
  causallm::CausalConv1DLayer::type};
}
#endif
