// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   lfm2_conv.cpp
 * @date   13 April 2026
 * @brief  LFM2 gated convolution layer (lightweight: gating + conv only)
 * @note   In_proj and out_proj are separate FC layers using optimized BLAS.
 *         This layer only handles: split B,C,x -> B*x -> conv -> C*conv
 *
 *         Weight[0]: conv kernel (hidden_size, conv_kernel_size)
 *         Tensor[0]: conv_state (hidden_size, conv_kernel_size)
 */

#include <cmath>
#include <cstring>
#include <lfm2_conv.h>

namespace causallm {

LFM2ConvLayer::LFM2ConvLayer() :
  Layer(),
  lfm2_props(props::LFM2HiddenSize(), props::LFM2ConvKernel()),
  hidden_size(0), conv_kernel_size(0), write_pos(0) {}

void LFM2ConvLayer::setProperty(const std::vector<std::string> &values) {
  auto remain = loadProperties(values, lfm2_props);
  NNTR_THROW_IF(!remain.empty(), std::invalid_argument)
    << "[lfm2_conv] Unknown properties";
}

void LFM2ConvLayer::finalize(nntrainer::InitLayerContext &context) {
  hidden_size = std::get<props::LFM2HiddenSize>(lfm2_props).get();
  conv_kernel_size = std::get<props::LFM2ConvKernel>(lfm2_props).get();

  auto input_dims = context.getInputDimensions();
  NNTR_THROW_IF(input_dims.size() != 1, std::invalid_argument)
    << "LFM2Conv requires exactly 1 input";

  // Input: (batch, seq, 3 * hidden_size) from FC in_proj
  // Output: (batch, seq, hidden_size)
  auto in_dim = input_dims[0];
  nntrainer::TensorDim out_dim = in_dim;
  out_dim.width(hidden_size);
  context.setOutputDimensions({out_dim});

  // Weight[0]: conv kernel (hidden_size, conv_kernel_size) - depthwise
  context.requestWeight(
    nntrainer::TensorDim({1, 1, hidden_size, conv_kernel_size}),
    nntrainer::Initializer::ONES, nntrainer::WeightRegularizer::NONE, 0.0f,
    0.0f, "conv", false);

  // Tensor[0]: conv_state circular buffer (hidden_size, conv_kernel_size)
  context.requestTensor(
    nntrainer::TensorDim({1, 1, hidden_size, conv_kernel_size}), "conv_state",
    nntrainer::Initializer::ZEROS, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN, false);
}

void LFM2ConvLayer::forwarding(nntrainer::RunLayerContext &context,
                               bool training) {
  auto &input = context.getInput(0);
  auto &output = context.getOutput(0);
  auto &conv_w = context.getWeight(0);
  auto &conv_state = context.getTensor(0);

  unsigned int batch = input.getDim().batch();
  unsigned int seq_len = input.getDim().height();

  const float *in_data = input.getData<float>();
  float *out_data = output.getData<float>();
  const float *conv_data = conv_w.getData<float>();
  float *state_data = conv_state.getData<float>();

  // Input width is 3 * hidden_size (from FC in_proj)
  const unsigned int in_width = 3 * hidden_size;

  for (unsigned int b = 0; b < batch; ++b) {
    for (unsigned int t = 0; t < seq_len; ++t) {
      const float *bcx = in_data + (b * seq_len + t) * in_width;
      float *y = out_data + (b * seq_len + t) * hidden_size;

      // Split: B = [0..H), C = [H..2H), x = [2H..3H)
      const float *B = bcx;
      const float *C = bcx + hidden_size;
      const float *x_proj = bcx + 2 * hidden_size;

      // 1. Bx = B * x (element-wise input gating)
      // 2. Write to circular buffer + depthwise conv
      unsigned int wp = write_pos % conv_kernel_size;

      for (unsigned int ch = 0; ch < hidden_size; ++ch) {
        state_data[ch * conv_kernel_size + wp] = B[ch] * x_proj[ch];
      }

      for (unsigned int ch = 0; ch < hidden_size; ++ch) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < conv_kernel_size; ++k) {
          unsigned int idx = (wp + 1 + k) % conv_kernel_size;
          sum += state_data[ch * conv_kernel_size + idx] *
                 conv_data[ch * conv_kernel_size + k];
        }
        // 3. C * conv_out (output gating)
        y[ch] = C[ch] * sum;
      }

      write_pos++;
    }
  }
}

void LFM2ConvLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                           unsigned int from, unsigned int to,
                                           bool training) {
  forwarding(context, training);
}

void LFM2ConvLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw std::runtime_error("LFM2ConvLayer::calcDerivative not supported");
}

} // namespace causallm
