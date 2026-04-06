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

// Platform-specific SIMD headers
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace causallm {

// ---------------------------------------------------------------------------
// SIMD kernel: y[f] = w0[f]*x[f] + w1[f]*s1[f] + w2[f]*s0[f]  for W floats.
// Called once per decode step – this is the hot path.
// ---------------------------------------------------------------------------
static void conv_fma_decode(const float * __restrict__ w0,
                            const float * __restrict__ w1,
                            const float * __restrict__ w2,
                            const float * __restrict__ x,
                            const float * __restrict__ s1,
                            const float * __restrict__ s0,
                            float       * __restrict__ y,
                            unsigned int W) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  unsigned int f = 0;
  for (; f + 4 <= W; f += 4) {
    float32x4_t vw0 = vld1q_f32(w0 + f);
    float32x4_t vw1 = vld1q_f32(w1 + f);
    float32x4_t vw2 = vld1q_f32(w2 + f);
    float32x4_t vx  = vld1q_f32(x  + f);
    float32x4_t vs1 = vld1q_f32(s1 + f);
    float32x4_t vs0 = vld1q_f32(s0 + f);

    float32x4_t vy = vmulq_f32(vw0, vx);
    vy = vfmaq_f32(vy, vw1, vs1);
    vy = vfmaq_f32(vy, vw2, vs0);
    vst1q_f32(y + f, vy);
  }
  for (; f < W; ++f)
    y[f] = w0[f] * x[f] + w1[f] * s1[f] + w2[f] * s0[f];

#elif defined(__AVX2__)
  unsigned int f = 0;
  for (; f + 8 <= W; f += 8) {
    __m256 vw0 = _mm256_loadu_ps(w0 + f);
    __m256 vw1 = _mm256_loadu_ps(w1 + f);
    __m256 vw2 = _mm256_loadu_ps(w2 + f);
    __m256 vx  = _mm256_loadu_ps(x  + f);
    __m256 vs1 = _mm256_loadu_ps(s1 + f);
    __m256 vs0 = _mm256_loadu_ps(s0 + f);

    __m256 vy  = _mm256_mul_ps(vw0, vx);
    vy = _mm256_fmadd_ps(vw1, vs1, vy);
    vy = _mm256_fmadd_ps(vw2, vs0, vy);
    _mm256_storeu_ps(y + f, vy);
  }
  for (; f < W; ++f)
    y[f] = w0[f] * x[f] + w1[f] * s1[f] + w2[f] * s0[f];

#else
  for (unsigned int f = 0; f < W; ++f)
    y[f] = w0[f] * x[f] + w1[f] * s1[f] + w2[f] * s0[f];
#endif
}

// ---------------------------------------------------------------------------
// SIMD kernel: prefill causal depthwise conv1d (kernel size 3, no bias).
// Processes to tokens of width W per batch b.
// Uses the same packed-weight layout [w0|w1|w2] (each W floats).
// ---------------------------------------------------------------------------
static void conv_fma_prefill(const float * __restrict__ w0,
                             const float * __restrict__ w1,
                             const float * __restrict__ w2,
                             const float * __restrict__ x,
                             float       * __restrict__ y,
                             unsigned int to,
                             unsigned int W) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  unsigned int c = 0;
  for (; c + 4 <= W; c += 4) {
    float32x4_t vw0 = vld1q_f32(w0 + c);
    float32x4_t vw1 = vld1q_f32(w1 + c);
    float32x4_t vw2 = vld1q_f32(w2 + c);

    float32x4_t prev2 = vdupq_n_f32(0.0f);
    float32x4_t prev1 = vdupq_n_f32(0.0f);

    for (unsigned int t = 0; t < to; ++t) {
      float32x4_t cur = vld1q_f32(x + t * W + c);

      float32x4_t vy = vmulq_f32(vw0, cur);
      vy = vfmaq_f32(vy, vw1, prev1);
      vy = vfmaq_f32(vy, vw2, prev2);
      vst1q_f32(y + t * W + c, vy);

      prev2 = prev1;
      prev1 = cur;
    }
  }
  // scalar tail for remaining features
  for (; c < W; ++c) {
    float prev2 = 0.0f, prev1 = 0.0f;
    for (unsigned int t = 0; t < to; ++t) {
      float cur = x[t * W + c];
      y[t * W + c] = w0[c] * cur + w1[c] * prev1 + w2[c] * prev2;
      prev2 = prev1;
      prev1 = cur;
    }
  }

#elif defined(__AVX2__)
  unsigned int c = 0;
  for (; c + 8 <= W; c += 8) {
    __m256 vw0 = _mm256_loadu_ps(w0 + c);
    __m256 vw1 = _mm256_loadu_ps(w1 + c);
    __m256 vw2 = _mm256_loadu_ps(w2 + c);

    __m256 prev2 = _mm256_setzero_ps();
    __m256 prev1 = _mm256_setzero_ps();

    for (unsigned int t = 0; t < to; ++t) {
      __m256 cur = _mm256_loadu_ps(x + t * W + c);

      __m256 vy  = _mm256_mul_ps(vw0, cur);
      vy = _mm256_fmadd_ps(vw1, prev1, vy);
      vy = _mm256_fmadd_ps(vw2, prev2, vy);
      _mm256_storeu_ps(y + t * W + c, vy);

      prev2 = prev1;
      prev1 = cur;
    }
  }
  // scalar tail
  for (; c < W; ++c) {
    float prev2 = 0.0f, prev1 = 0.0f;
    for (unsigned int t = 0; t < to; ++t) {
      float cur = x[t * W + c];
      y[t * W + c] = w0[c] * cur + w1[c] * prev1 + w2[c] * prev2;
      prev2 = prev1;
      prev1 = cur;
    }
  }

#else
  // Scalar fallback
  for (unsigned int t = 0; t < to; ++t) {
    const float *cur = x + t * W;
    const float *p1  = (t >= 1) ? x + (t - 1) * W : nullptr;
    const float *p2  = (t >= 2) ? x + (t - 2) * W : nullptr;
    float *yt = y + t * W;
    for (unsigned int f = 0; f < W; ++f) {
      float val = w0[f] * cur[f];
      if (p1) val += w1[f] * p1[f];
      if (p2) val += w2[f] * p2[f];
      yt[f] = val;
    }
  }
#endif
}

// ===========================================================================

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
    {1, 1, KERNEL_SIZE, W},
    {context.getFormat(), ml::train::TensorDim::DataType::FP32});
  weight_idx[weight] =
    context.requestWeight(weight_dim, nntrainer::Initializer::NONE,
                          nntrainer::WeightRegularizer::NONE, 0.0f, 0.0f,
                          "causal_conv1d_weight", false);

  // Conv-state cache: [B, 1, KERNEL_SIZE-1, W] FP32
  //   state[b, 0, 0, f] = x_{t-2}
  //   state[b, 0, 1, f] = x_{t-1}
  nntrainer::TensorDim state_dim(
    {B, 1, KERNEL_SIZE - 1, W},
    {context.getFormat(), ml::train::TensorDim::DataType::FP32});
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

  nntrainer::Tensor &input   = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output  = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &w_tensor = context.getWeight(weight_idx[weight]);
  nntrainer::Tensor &state    = context.getTensor(tensor_idx[conv_state]);

  const unsigned int B = input.batch();
  const unsigned int H = input.height(); // full sequence length (INIT_SEQ_LEN)
  const unsigned int W = input.width();  // feature dimension

  // Packed weight layout: [w0 | w1 | w2]  (each W floats)
  const float *w_ptr = w_tensor.getData<float>();
  const float *w0    = w_ptr;
  const float *w1    = w_ptr + W;
  const float *w2    = w_ptr + 2 * W;

  float *state_data = state.getData<float>(); // [B, 1, KERNEL_SIZE-1, W]

  if (to - from == 1) {
    // ----------------------------------------------------------------
    // Decode path (hot): single-token inference.
    // NNTrainer places the current token at offset 0 within each batch
    // slice (not at offset `from`).
    // ----------------------------------------------------------------
    for (unsigned int b = 0; b < B; ++b) {
      const float *x_cur = input.getData<float>()  + b * H * W;
      float       *y_cur = output.getData<float>()  + b * H * W;
      const float *s0    = state_data + b * (KERNEL_SIZE - 1) * W;
      const float *s1    = s0 + W;

      // SIMD FMA: y = w0*x + w1*s1 + w2*s0
      conv_fma_decode(w0, w1, w2, x_cur, s1, s0, y_cur, W);

      // Update state: shift and insert current token
      //   s[0] <- s[1]  (x_{t-2} <- x_{t-1})
      //   s[1] <- x_cur (x_{t-1} <- x_t     )
      float *s = state_data + b * (KERNEL_SIZE - 1) * W;
      std::memcpy(s,     s + W,  W * sizeof(float)); // shift
      std::memcpy(s + W, x_cur,  W * sizeof(float)); // insert
    }

  } else {
    // ----------------------------------------------------------------
    // Prefill path: process all positions [0, to).
    // ----------------------------------------------------------------
    for (unsigned int b = 0; b < B; ++b) {
      const float *x = input.getData<float>()  + b * H * W;
      float       *y = output.getData<float>()  + b * H * W;

      // SIMD prefill kernel (column-major SIMD, row-major fallback)
      conv_fma_prefill(w0, w1, w2, x, y, to, W);

      // Save last KERNEL_SIZE-1 tokens to state for decode steps
      float *s = state_data + b * (KERNEL_SIZE - 1) * W;

      if (to >= 2)
        std::memcpy(s, x + (to - 2) * W, W * sizeof(float)); // x_{to-2}
      else
        std::memset(s, 0, W * sizeof(float));

      std::memcpy(s + W, x + (to - 1) * W, W * sizeof(float));  // x_{to-1}
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
  LayerImpl::setProperty(values);
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
