// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   lfm2_conv.cpp
 * @date   13 April 2026
 * @brief  LFM2 double-gated short convolution layer implementation
 * @note   Forward: in_proj(hidden, 3*hidden) -> split B,C,x -> B*x ->
 *         depthwise_conv1d(kernel=L_cache) -> C*conv_out -> out_proj(hidden,
 *         hidden)
 *
 *         Weights:
 *           0: in_proj  (hidden_size, 3 * hidden_size)
 *           1: conv     (hidden_size, conv_kernel_size)
 *           2: out_proj (hidden_size, hidden_size)
 *
 *         Tensors (state):
 *           0: conv_state circular buffer (hidden_size, conv_kernel_size)
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

  auto in_dim = input_dims[0];

  // Output: same shape as input
  nntrainer::TensorDim out_dim = in_dim;
  out_dim.width(hidden_size);
  context.setOutputDimensions({out_dim});

  // Weight[0]: in_proj (hidden_size, 3 * hidden_size)
  context.requestWeight(
    nntrainer::TensorDim({1, 1, hidden_size, 3 * hidden_size}),
    nntrainer::Initializer::ONES, nntrainer::WeightRegularizer::NONE, 0.0f,
    0.0f, "in_proj", false);

  // Weight[1]: conv kernel (hidden_size, conv_kernel_size)
  context.requestWeight(
    nntrainer::TensorDim({1, 1, hidden_size, conv_kernel_size}),
    nntrainer::Initializer::ONES, nntrainer::WeightRegularizer::NONE, 0.0f,
    0.0f, "conv", false);

  // Weight[2]: out_proj (hidden_size, hidden_size)
  context.requestWeight(
    nntrainer::TensorDim({1, 1, hidden_size, hidden_size}),
    nntrainer::Initializer::ONES, nntrainer::WeightRegularizer::NONE, 0.0f,
    0.0f, "out_proj", false);

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

  auto &in_proj_w = context.getWeight(0);
  auto &conv_w = context.getWeight(1);
  auto &out_proj_w = context.getWeight(2);
  auto &conv_state = context.getTensor(0);

  unsigned int batch = input.getDim().batch();
  unsigned int seq_len = input.getDim().height();

  const float *in_data = input.getData<float>();
  float *out_data = output.getData<float>();
  const float *in_proj_data = in_proj_w.getData<float>();
  const float *conv_data = conv_w.getData<float>();
  const float *out_proj_data = out_proj_w.getData<float>();
  float *state_data = conv_state.getData<float>();

  const unsigned int proj_dim = 3 * hidden_size;

  // Pre-allocate buffers
  std::vector<float> bcx_buf(proj_dim);
  std::vector<float> bx_buf(hidden_size);
  std::vector<float> conv_out_buf(hidden_size);
  std::vector<float> gated_buf(hidden_size);

  for (unsigned int b = 0; b < batch; ++b) {
    for (unsigned int t = 0; t < seq_len; ++t) {
      const float *x = in_data + (b * seq_len + t) * hidden_size;
      float *y = out_data + (b * seq_len + t) * hidden_size;

      // 1. in_proj: x (1,H) @ in_proj_w (H, 3H) = BCx (1, 3H)
      for (unsigned int j = 0; j < proj_dim; ++j) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < hidden_size; ++k) {
          sum += x[k] * in_proj_data[k * proj_dim + j];
        }
        bcx_buf[j] = sum;
      }

      // 2. Split: B = [0..H), C = [H..2H), x_proj = [2H..3H)
      const float *B = bcx_buf.data();
      const float *C = bcx_buf.data() + hidden_size;
      const float *x_proj = bcx_buf.data() + 2 * hidden_size;

      // 3. Bx = B * x_proj (element-wise input gating)
      for (unsigned int j = 0; j < hidden_size; ++j) {
        bx_buf[j] = B[j] * x_proj[j];
      }

      // 4. Write Bx to circular buffer and compute depthwise conv
      unsigned int wp = write_pos % conv_kernel_size;
      for (unsigned int ch = 0; ch < hidden_size; ++ch) {
        state_data[ch * conv_kernel_size + wp] = bx_buf[ch];
      }

      for (unsigned int ch = 0; ch < hidden_size; ++ch) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < conv_kernel_size; ++k) {
          // Read from circular buffer: oldest first
          unsigned int idx = (wp + 1 + k) % conv_kernel_size;
          sum +=
            state_data[ch * conv_kernel_size + idx] *
            conv_data[ch * conv_kernel_size + k];
        }
        conv_out_buf[ch] = sum;
      }

      write_pos++;

      // 5. C * conv_out (output gating)
      for (unsigned int j = 0; j < hidden_size; ++j) {
        gated_buf[j] = C[j] * conv_out_buf[j];
      }

      // 6. out_proj: gated (1,H) @ out_proj_w (H,H) = y (1,H)
      for (unsigned int j = 0; j < hidden_size; ++j) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < hidden_size; ++k) {
          sum += gated_buf[k] * out_proj_data[k * hidden_size + j];
        }
        y[j] = sum;
      }
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
