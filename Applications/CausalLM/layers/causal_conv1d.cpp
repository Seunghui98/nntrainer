// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   causal_conv1d.cpp
 * @date   9 April 2026
 * @brief  Causal Conv1D layer implementation
 */

#include <cmath>
#include <cstring>
#include <causal_conv1d.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace causallm {

static inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static inline float silu(float x) { return x * sigmoid(x); }

// Apply SiLU to a buffer in-place using SIMD
static void silu_inplace(float *data, unsigned int len) {
  unsigned int i = 0;
#ifdef __AVX2__
  for (; i + 7 < len; i += 8) {
    alignas(32) float buf[8];
    __m256 v = _mm256_loadu_ps(&data[i]);
    _mm256_store_ps(buf, v);
    for (int j = 0; j < 8; ++j)
      buf[j] = buf[j] / (1.0f + std::exp(-buf[j]));
    __m256 sig = _mm256_load_ps(buf);
    _mm256_storeu_ps(&data[i], _mm256_mul_ps(v, sig));
  }
#elif defined(__ARM_NEON)
  for (; i + 3 < len; i += 4) {
    alignas(16) float buf[4];
    float32x4_t v = vld1q_f32(&data[i]);
    vst1q_f32(buf, v);
    for (int j = 0; j < 4; ++j)
      buf[j] = 1.0f / (1.0f + std::exp(-buf[j]));
    float32x4_t sig = vld1q_f32(buf);
    vst1q_f32(&data[i], vmulq_f32(v, sig));
  }
#endif
  for (; i < len; ++i)
    data[i] = silu(data[i]);
}

CausalConv1dLayer::CausalConv1dLayer() :
  Layer(),
  conv_props(props::ConvChannels(), props::ConvKernelSize()),
  channels(0),
  kernel_size(0),
  wt_idx(std::numeric_limits<unsigned int>::max()),
  state_idx(std::numeric_limits<unsigned int>::max()) {}

void CausalConv1dLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain = loadProperties(values, conv_props);
  NNTR_THROW_IF(!remain.empty(), std::invalid_argument)
    << "[causal_conv1d] Unknown properties";
}

void CausalConv1dLayer::finalize(nntrainer::InitLayerContext &context) {
  channels = std::get<props::ConvChannels>(conv_props).get();
  kernel_size = std::get<props::ConvKernelSize>(conv_props).get();

  auto input_dims = context.getInputDimensions();
  unsigned int batch_size = input_dims[0].batch();

  // Output same shape as input
  context.setOutputDimensions(input_dims);

  auto wt_type = nntrainer::TensorDim::TensorType(
    context.getFormat(), context.getWeightDataType());
  auto act_type = nntrainer::TensorDim::TensorType(
    context.getFormat(), context.getActivationDataType());

  // Weight: conv kernel (1, 1, channels, kernel_size)
  wt_idx = context.requestWeight(
    nntrainer::TensorDim(1, 1, channels, kernel_size, wt_type),
    nntrainer::Initializer::NONE, nntrainer::WeightRegularizer::NONE, 1.0f,
    0.0f, "conv_kernel", false);

  // State: (batch, 1, channels, kernel_size - 1)
  state_idx = context.requestTensor(
    nntrainer::TensorDim(batch_size, 1, channels, kernel_size - 1, act_type),
    "conv_state", nntrainer::Initializer::ZEROS, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);
}

void CausalConv1dLayer::forwarding(nntrainer::RunLayerContext &context,
                                   bool training) {}

void CausalConv1dLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {

  nntrainer::Tensor &input = context.getInput(0);
  nntrainer::Tensor &output = context.getOutput(0);
  nntrainer::Tensor &w_conv = context.getWeight(wt_idx);
  nntrainer::Tensor &conv_state = context.getTensor(state_idx);

  unsigned int batch_size = input.getDim().batch();
  unsigned int seq_len = to - from;
  unsigned int state_width = kernel_size - 1;

  const float *conv_w = w_conv.getData<float>();

  for (unsigned int b = 0; b < batch_size; ++b) {
    const float *in_ptr =
      input.getData<float>() + b * input.getDim().getFeatureLen();
    float *out_ptr =
      output.getData<float>() + b * output.getDim().getFeatureLen();
    float *st_ptr =
      conv_state.getData<float>() + b * conv_state.getDim().getFeatureLen();

    for (unsigned int t = 0; t < seq_len; ++t) {
      const float *x = in_ptr + t * channels;
      float *y = out_ptr + t * channels;

      // Conv1d per channel: dot product of [state..., new_val] with kernel
      for (unsigned int ch = 0; ch < channels; ++ch) {
        float *ch_state = st_ptr + ch * state_width;
        const float *kernel = conv_w + ch * kernel_size;

        float val = 0.0f;
        for (unsigned int k = 0; k < state_width; ++k)
          val += ch_state[k] * kernel[k];
        val += x[ch] * kernel[state_width];
        y[ch] = val;

        // Shift state left, append new value
        for (unsigned int k = 0; k < state_width - 1; ++k)
          ch_state[k] = ch_state[k + 1];
        if (state_width > 0)
          ch_state[state_width - 1] = x[ch];
      }

      // Batched SiLU over all channels (SIMD optimized)
      silu_inplace(y, channels);
    }
  }
}

void CausalConv1dLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(0, input_dimensions[0]);
  context.updateOutput(0, input_dimensions[0]);
}

void CausalConv1dLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

#ifdef PLUGGABLE
nntrainer::Layer *create_causal_conv1d_layer() {
  return new CausalConv1dLayer();
}
void destroy_causal_conv1d_layer(nntrainer::Layer *layer) { delete layer; }
extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_causal_conv1d_layer, destroy_causal_conv1d_layer};
}
#endif

} // namespace causallm
