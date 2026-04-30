// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   causal_conv1d.cpp
 * @date   9 April 2026
 * @brief  Causal Conv1D - circular buffer + channel SIMD vectorization
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

// SiLU over all channels (SIMD vectorized)
static void silu_inplace(float *data, unsigned int len) {
  unsigned int i = 0;
#ifdef __AVX2__
  for (; i + 7 < len; i += 8) {
    alignas(32) float buf[8];
    __m256 v = _mm256_loadu_ps(&data[i]);
    _mm256_store_ps(buf, v);
    for (int j = 0; j < 8; ++j)
      buf[j] = 1.0f / (1.0f + std::exp(-buf[j]));  // sigmoid
    _mm256_storeu_ps(&data[i], _mm256_mul_ps(v, _mm256_load_ps(buf)));
  }
#elif defined(__ARM_NEON)
  for (; i + 3 < len; i += 4) {
    alignas(16) float buf[4];
    float32x4_t v = vld1q_f32(&data[i]);
    vst1q_f32(buf, v);
    for (int j = 0; j < 4; ++j)
      buf[j] = 1.0f / (1.0f + std::exp(-buf[j]));
    vst1q_f32(&data[i], vmulq_f32(v, vld1q_f32(buf)));
  }
#endif
  for (; i < len; ++i) data[i] = silu(data[i]);
}

CausalConv1dLayer::CausalConv1dLayer() :
  Layer(), conv_props(props::ConvChannels(), props::ConvKernelSize()),
  channels(0), kernel_size(0),
  wt_idx(std::numeric_limits<unsigned int>::max()),
  state_idx(std::numeric_limits<unsigned int>::max()),
  kernel_t_idx(std::numeric_limits<unsigned int>::max()),
  kernel_transposed(false), write_pos(0) {}

void CausalConv1dLayer::setProperty(const std::vector<std::string> &values) {
  auto remain = loadProperties(values, conv_props);
  NNTR_THROW_IF(!remain.empty(), std::invalid_argument)
    << "[causal_conv1d] Unknown properties";
}

void CausalConv1dLayer::finalize(nntrainer::InitLayerContext &context) {
  channels = std::get<props::ConvChannels>(conv_props).get();
  kernel_size = std::get<props::ConvKernelSize>(conv_props).get();

  auto input_dims = context.getInputDimensions();
  unsigned int batch_size = input_dims[0].batch();
  unsigned int state_width = kernel_size - 1;

  context.setOutputDimensions(input_dims);

  auto wt_type = nntrainer::TensorDim::TensorType(
    context.getFormat(), context.getWeightDataType());
  auto act_type = nntrainer::TensorDim::TensorType(
    context.getFormat(), context.getActivationDataType());

  // Weight: original kernel (1, 1, channels, kernel_size) - loaded from file
  wt_idx = context.requestWeight(
    nntrainer::TensorDim(1, 1, channels, kernel_size, wt_type),
    nntrainer::Initializer::NONE, nntrainer::WeightRegularizer::NONE,
    1.0f, 0.0f, "conv_kernel", false);

  // State: circular buffer (batch, 1, state_width, channels) - transposed layout
  state_idx = context.requestTensor(
    nntrainer::TensorDim(batch_size, 1, state_width, channels, act_type),
    "conv_state", nntrainer::Initializer::ZEROS, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);

  // Transposed kernel: (1, 1, kernel_size, channels) for SIMD channel access
  kernel_t_idx = context.requestTensor(
    nntrainer::TensorDim(1, 1, kernel_size, channels, act_type),
    "kernel_transposed", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);

  kernel_transposed = false;
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
  nntrainer::Tensor &kernel_t = context.getTensor(kernel_t_idx);

  unsigned int batch_size = input.getDim().batch();
  unsigned int seq_len = to - from;
  unsigned int state_width = kernel_size - 1;

  // Transpose kernel on first use: (channels, kernel_size) → (kernel_size, channels)
  if (!kernel_transposed) {
    const float *src = w_conv.getData<float>();
    float *dst = kernel_t.getData<float>();
    for (unsigned int ch = 0; ch < channels; ++ch)
      for (unsigned int k = 0; k < kernel_size; ++k)
        dst[k * channels + ch] = src[ch * kernel_size + k];
    kernel_transposed = true;
  }

  const float *kt = kernel_t.getData<float>();
  // Reset write_pos on prefill (from==0 with multiple tokens)
  if (from == 0 && to > 1)
    write_pos = 0;

  for (unsigned int b = 0; b < batch_size; ++b) {
    const float *in_ptr = input.getData<float>() + b * input.getDim().getFeatureLen();
    float *out_ptr = output.getData<float>() + b * output.getDim().getFeatureLen();
    float *st_ptr = conv_state.getData<float>() + b * conv_state.getDim().getFeatureLen();

    for (unsigned int t = 0; t < seq_len; ++t) {
      const float *x = in_ptr + t * channels;
      float *y = out_ptr + t * channels;

      // Compute conv output vectorized across channels
      std::memset(y, 0, channels * sizeof(float));

      // Accumulate state contributions (circular buffer, no shift)
      for (unsigned int k = 0; k < state_width; ++k) {
        unsigned int state_row = (write_pos + k) % state_width;
        const float *s_row = st_ptr + state_row * channels;
        const float *k_row = kt + k * channels;
        unsigned int ch = 0;
#ifdef __AVX2__
        for (; ch + 7 < channels; ch += 8) {
          __m256 o = _mm256_loadu_ps(&y[ch]);
          __m256 sv = _mm256_loadu_ps(&s_row[ch]);
          __m256 kv = _mm256_loadu_ps(&k_row[ch]);
          _mm256_storeu_ps(&y[ch], _mm256_fmadd_ps(sv, kv, o));
        }
#elif defined(__ARM_NEON)
        for (; ch + 3 < channels; ch += 4) {
          float32x4_t o = vld1q_f32(&y[ch]);
          float32x4_t sv = vld1q_f32(&s_row[ch]);
          float32x4_t kv = vld1q_f32(&k_row[ch]);
          vst1q_f32(&y[ch], vfmaq_f32(o, sv, kv));
        }
#endif
        for (; ch < channels; ++ch)
          y[ch] += s_row[ch] * k_row[ch];
      }

      // Add current input * last kernel position
      const float *k_last = kt + state_width * channels;
      unsigned int ch = 0;
#ifdef __AVX2__
      for (; ch + 7 < channels; ch += 8) {
        __m256 o = _mm256_loadu_ps(&y[ch]);
        __m256 xv = _mm256_loadu_ps(&x[ch]);
        __m256 kv = _mm256_loadu_ps(&k_last[ch]);
        _mm256_storeu_ps(&y[ch], _mm256_fmadd_ps(xv, kv, o));
      }
#elif defined(__ARM_NEON)
      for (; ch + 3 < channels; ch += 4) {
        float32x4_t o = vld1q_f32(&y[ch]);
        float32x4_t xv = vld1q_f32(&x[ch]);
        float32x4_t kv = vld1q_f32(&k_last[ch]);
        vst1q_f32(&y[ch], vfmaq_f32(o, xv, kv));
      }
#endif
      for (; ch < channels; ++ch)
        y[ch] += x[ch] * k_last[ch];

      // Update circular buffer (single memcpy, no shift!)
      float *s_write = st_ptr + write_pos * channels;
      std::memcpy(s_write, x, channels * sizeof(float));
      write_pos = (write_pos + 1) % state_width;

      // SiLU activation (SIMD vectorized)
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
nntrainer::Layer *create_causal_conv1d_layer() { return new CausalConv1dLayer(); }
void destroy_causal_conv1d_layer(nntrainer::Layer *layer) { delete layer; }
extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_causal_conv1d_layer, destroy_causal_conv1d_layer};
}
#endif

} // namespace causallm
