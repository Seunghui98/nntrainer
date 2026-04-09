// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   gdn_ssm_core.cpp
 * @date   9 April 2026
 * @brief  GDN SSM Core layer implementation with BLAS + SIMD optimization
 */

#include <cmath>
#include <cstring>
#include <vector>

#include <gdn_ssm_core.h>
#include <cblas_interface.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace causallm {

// ============================================================
// SIMD-optimized helper functions
// ============================================================

#ifdef __AVX2__
static inline __m256 sigmoid_avx2(__m256 x) {
  // sigmoid(x) = 1 / (1 + exp(-x))
  // Approximate: use tanh-based identity: sigmoid(x) = 0.5*(1+tanh(x/2))
  // For accuracy, use scalar exp path
  alignas(32) float buf[8];
  _mm256_store_ps(buf, x);
  for (int i = 0; i < 8; ++i)
    buf[i] = 1.0f / (1.0f + std::exp(-buf[i]));
  return _mm256_load_ps(buf);
}

static inline __m256 silu_avx2(__m256 x) {
  return _mm256_mul_ps(x, sigmoid_avx2(x));
}
#endif

#ifdef __ARM_NEON
static inline float32x4_t sigmoid_neon(float32x4_t x) {
  alignas(16) float buf[4];
  vst1q_f32(buf, x);
  for (int i = 0; i < 4; ++i)
    buf[i] = 1.0f / (1.0f + std::exp(-buf[i]));
  return vld1q_f32(buf);
}

static inline float32x4_t silu_neon(float32x4_t x) {
  return vmulq_f32(x, sigmoid_neon(x));
}
#endif

static inline float softplus(float x) {
  if (x > 20.0f)
    return x;
  return std::log(1.0f + std::exp(x));
}

static inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static inline float silu(float x) { return x * sigmoid(x); }

// L2 normalize with SIMD
static void l2_normalize(float *data, unsigned int len, float eps = 1e-6f) {
  float norm_sq = 0.0f;
  unsigned int i = 0;

#ifdef __AVX2__
  __m256 sum_v = _mm256_setzero_ps();
  for (; i + 7 < len; i += 8) {
    __m256 v = _mm256_loadu_ps(&data[i]);
    sum_v = _mm256_fmadd_ps(v, v, sum_v);
  }
  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, sum_v);
  norm_sq = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
#elif defined(__ARM_NEON)
  float32x4_t sum_v = vdupq_n_f32(0);
  for (; i + 3 < len; i += 4) {
    float32x4_t v = vld1q_f32(&data[i]);
    sum_v = vfmaq_f32(sum_v, v, v);
  }
  norm_sq = vaddvq_f32(sum_v);
#endif
  for (; i < len; ++i)
    norm_sq += data[i] * data[i];

  float inv_norm = 1.0f / std::sqrt(norm_sq + eps);
  i = 0;

#ifdef __AVX2__
  __m256 scale_v = _mm256_set1_ps(inv_norm);
  for (; i + 7 < len; i += 8) {
    __m256 v = _mm256_loadu_ps(&data[i]);
    _mm256_storeu_ps(&data[i], _mm256_mul_ps(v, scale_v));
  }
#elif defined(__ARM_NEON)
  float32x4_t scale_v = vdupq_n_f32(inv_norm);
  for (; i + 3 < len; i += 4) {
    float32x4_t v = vld1q_f32(&data[i]);
    vst1q_f32(&data[i], vmulq_f32(v, scale_v));
  }
#endif
  for (; i < len; ++i)
    data[i] *= inv_norm;
}

// SSM state update: S = decay*S + outer(k, delta)
// S is (head_k_dim, head_v_dim), row-major
static void ssm_state_update(float *S, const float *k, const float *delta,
                             float decay, unsigned int k_dim,
                             unsigned int v_dim) {
  for (unsigned int ki = 0; ki < k_dim; ++ki) {
    float k_val = k[ki];
    float *row = S + ki * v_dim;
    unsigned int vi = 0;

#ifdef __AVX2__
    __m256 decay_v = _mm256_set1_ps(decay);
    __m256 k_v = _mm256_set1_ps(k_val);
    for (; vi + 7 < v_dim; vi += 8) {
      __m256 s = _mm256_loadu_ps(&row[vi]);
      __m256 d = _mm256_loadu_ps(&delta[vi]);
      s = _mm256_fmadd_ps(k_v, d, _mm256_mul_ps(s, decay_v));
      _mm256_storeu_ps(&row[vi], s);
    }
#elif defined(__ARM_NEON)
    float32x4_t decay_v = vdupq_n_f32(decay);
    float32x4_t k_v = vdupq_n_f32(k_val);
    for (; vi + 3 < v_dim; vi += 4) {
      float32x4_t s = vld1q_f32(&row[vi]);
      float32x4_t d = vld1q_f32(&delta[vi]);
      s = vfmaq_f32(vmulq_f32(s, decay_v), k_v, d);
      vst1q_f32(&row[vi], s);
    }
#endif
    for (; vi < v_dim; ++vi)
      row[vi] = decay * row[vi] + k_val * delta[vi];
  }
}

// Gated RMS norm + SiLU: out = rms_norm(in, weight) * silu(z)
static void gated_rms_silu(const float *in, const float *z, float *out,
                           const float *weight, unsigned int len,
                           float eps = 1e-6f) {
  // First compute RMS norm
  float sum_sq = 0.0f;
  unsigned int i = 0;

#ifdef __AVX2__
  __m256 sum_v = _mm256_setzero_ps();
  for (; i + 7 < len; i += 8) {
    __m256 v = _mm256_loadu_ps(&in[i]);
    sum_v = _mm256_fmadd_ps(v, v, sum_v);
  }
  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, sum_v);
  sum_sq = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
#elif defined(__ARM_NEON)
  float32x4_t sum_v = vdupq_n_f32(0);
  for (; i + 3 < len; i += 4) {
    float32x4_t v = vld1q_f32(&in[i]);
    sum_v = vfmaq_f32(sum_v, v, v);
  }
  sum_sq = vaddvq_f32(sum_v);
#endif
  for (; i < len; ++i)
    sum_sq += in[i] * in[i];

  float scale = 1.0f / std::sqrt(sum_sq / len + eps);

  // Fused: out = (in * scale * weight) * silu(z)
  i = 0;
#ifdef __AVX2__
  __m256 scale_v = _mm256_set1_ps(scale);
  for (; i + 7 < len; i += 8) {
    __m256 x = _mm256_loadu_ps(&in[i]);
    __m256 w = _mm256_loadu_ps(&weight[i]);
    __m256 zv = _mm256_loadu_ps(&z[i]);
    __m256 normed = _mm256_mul_ps(_mm256_mul_ps(x, scale_v), w);
    _mm256_storeu_ps(&out[i], _mm256_mul_ps(normed, silu_avx2(zv)));
  }
#elif defined(__ARM_NEON)
  float32x4_t scale_v = vdupq_n_f32(scale);
  for (; i + 3 < len; i += 4) {
    float32x4_t x = vld1q_f32(&in[i]);
    float32x4_t w = vld1q_f32(&weight[i]);
    float32x4_t zv = vld1q_f32(&z[i]);
    float32x4_t normed = vmulq_f32(vmulq_f32(x, scale_v), w);
    vst1q_f32(&out[i], vmulq_f32(normed, silu_neon(zv)));
  }
#endif
  for (; i < len; ++i)
    out[i] = in[i] * scale * weight[i] * silu(z[i]);
}

// ============================================================
// Layer implementation
// ============================================================

GdnSsmCoreLayer::GdnSsmCoreLayer() :
  Layer(),
  ssm_props(props::SSMNumVHeads(), props::SSMHeadKDim(), props::SSMHeadVDim()),
  num_v_heads(0),
  head_k_dim(0),
  head_v_dim(0),
  key_dim(0),
  value_dim(0),
  state_idx(std::numeric_limits<unsigned int>::max()) {
  wt_idx.fill(std::numeric_limits<unsigned int>::max());
}

void GdnSsmCoreLayer::setProperty(const std::vector<std::string> &values) {
  auto remain = loadProperties(values, ssm_props);
  NNTR_THROW_IF(!remain.empty(), std::invalid_argument)
    << "[gdn_ssm_core] Unknown properties";
}

void GdnSsmCoreLayer::finalize(nntrainer::InitLayerContext &context) {
  num_v_heads = std::get<props::SSMNumVHeads>(ssm_props).get();
  head_k_dim = std::get<props::SSMHeadKDim>(ssm_props).get();
  head_v_dim = std::get<props::SSMHeadVDim>(ssm_props).get();

  key_dim = num_v_heads * head_k_dim;
  value_dim = num_v_heads * head_v_dim;

  auto input_dims = context.getInputDimensions();
  unsigned int batch_size = input_dims[0].batch();

  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[0] = input_dims[0];
  output_dims[0].width(value_dim);
  context.setOutputDimensions(output_dims);

  auto wt_type = nntrainer::TensorDim::TensorType(
    context.getFormat(), context.getWeightDataType());
  auto act_type = nntrainer::TensorDim::TensorType(
    context.getFormat(), context.getActivationDataType());

  wt_idx[W_A_LOG] = context.requestWeight(
    nntrainer::TensorDim(1, 1, 1, num_v_heads, wt_type),
    nntrainer::Initializer::NONE, nntrainer::WeightRegularizer::NONE, 1.0f,
    0.0f, "A_log", false);

  wt_idx[W_DT_BIAS] = context.requestWeight(
    nntrainer::TensorDim(1, 1, 1, num_v_heads, wt_type),
    nntrainer::Initializer::NONE, nntrainer::WeightRegularizer::NONE, 1.0f,
    0.0f, "dt_bias", false);

  wt_idx[W_NORM] = context.requestWeight(
    nntrainer::TensorDim(1, 1, 1, head_v_dim, wt_type),
    nntrainer::Initializer::NONE, nntrainer::WeightRegularizer::NONE, 1.0f,
    0.0f, "norm", false);

  state_idx = context.requestTensor(
    nntrainer::TensorDim(batch_size, 1, num_v_heads * head_k_dim, head_v_dim,
                         act_type),
    "recurrent_state", nntrainer::Initializer::ZEROS, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);
}

void GdnSsmCoreLayer::forwarding(nntrainer::RunLayerContext &context,
                                 bool training) {}

void GdnSsmCoreLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {

  nntrainer::Tensor &conv_out_tensor = context.getInput(0);
  nntrainer::Tensor &a_tensor = context.getInput(1);
  nntrainer::Tensor &b_tensor = context.getInput(2);
  nntrainer::Tensor &z_tensor = context.getInput(3);
  nntrainer::Tensor &output = context.getOutput(0);

  nntrainer::Tensor &recurrent_state = context.getTensor(state_idx);

  const float *a_log = context.getWeight(wt_idx[W_A_LOG]).getData<float>();
  const float *dt_bias = context.getWeight(wt_idx[W_DT_BIAS]).getData<float>();
  const float *norm_w = context.getWeight(wt_idx[W_NORM]).getData<float>();

  unsigned int batch_size = conv_out_tensor.getDim().batch();
  unsigned int seq_len = to - from;
  unsigned int conv_dim = conv_out_tensor.getDim().width();

  for (unsigned int b = 0; b < batch_size; ++b) {
    const float *conv_ptr =
      conv_out_tensor.getData<float>() +
      b * conv_out_tensor.getDim().getFeatureLen();
    const float *a_ptr =
      a_tensor.getData<float>() + b * a_tensor.getDim().getFeatureLen();
    const float *b_ptr =
      b_tensor.getData<float>() + b * b_tensor.getDim().getFeatureLen();
    const float *z_ptr =
      z_tensor.getData<float>() + b * z_tensor.getDim().getFeatureLen();
    float *out_ptr =
      output.getData<float>() + b * output.getDim().getFeatureLen();
    float *state_ptr =
      recurrent_state.getData<float>() +
      b * recurrent_state.getDim().getFeatureLen();

    for (unsigned int t = 0; t < seq_len; ++t) {
      const float *conv_t = conv_ptr + t * conv_dim;
      const float *a_t = a_ptr + t * num_v_heads;
      const float *b_t = b_ptr + t * num_v_heads;
      const float *z_t = z_ptr + t * value_dim;
      float *y = out_ptr + t * value_dim;

      // Copy conv_out to mutable buffer, split into Q, K, V
      std::vector<float> qkv_buf(conv_dim);
      std::memcpy(qkv_buf.data(), conv_t, conv_dim * sizeof(float));

      float *q_data = qkv_buf.data();
      float *k_data = qkv_buf.data() + key_dim;
      float *v_data = qkv_buf.data() + 2 * key_dim;

      // L2 normalize Q and K per head (SIMD-optimized)
      for (unsigned int h = 0; h < num_v_heads; ++h) {
        l2_normalize(q_data + h * head_k_dim, head_k_dim);
        l2_normalize(k_data + h * head_k_dim, head_k_dim);
      }

      // Scale query
      float q_scale = 1.0f / std::sqrt((float)head_k_dim);
      unsigned int qi = 0;
#ifdef __AVX2__
      __m256 qs = _mm256_set1_ps(q_scale);
      for (; qi + 7 < key_dim; qi += 8)
        _mm256_storeu_ps(&q_data[qi],
                         _mm256_mul_ps(_mm256_loadu_ps(&q_data[qi]), qs));
#elif defined(__ARM_NEON)
      float32x4_t qs = vdupq_n_f32(q_scale);
      for (; qi + 3 < key_dim; qi += 4)
        vst1q_f32(&q_data[qi], vmulq_f32(vld1q_f32(&q_data[qi]), qs));
#endif
      for (; qi < key_dim; ++qi)
        q_data[qi] *= q_scale;

      // Compute beta and decay per head
      std::vector<float> beta_val(num_v_heads);
      std::vector<float> decay_val(num_v_heads);
      for (unsigned int h = 0; h < num_v_heads; ++h) {
        beta_val[h] = sigmoid(b_t[h]);
        float g = -std::exp(a_log[h]) * softplus(a_t[h] + dt_bias[h]);
        decay_val[h] = std::exp(g);
      }

      // SSM state update + output per head (BLAS + SIMD optimized)
      for (unsigned int h = 0; h < num_v_heads; ++h) {
        float *S = state_ptr + h * head_k_dim * head_v_dim;
        const float *q_h = q_data + h * head_k_dim;
        const float *k_h = k_data + h * head_k_dim;
        const float *v_h = v_data + h * head_v_dim;
        float decay = decay_val[h];
        float beta_h = beta_val[h];
        float *o_h = y + h * head_v_dim;

        // kv_mem = S^T @ k using BLAS sgemv
        // S is (head_k_dim, head_v_dim) row-major
        // S^T @ k = (head_v_dim, head_k_dim) @ (head_k_dim,) = (head_v_dim,)
        std::vector<float> kv_mem(head_v_dim, 0.0f);
        nntrainer::__cblas_sgemv(0 /*RowMajor*/, true /*Trans*/, head_k_dim, head_v_dim,
                      1.0f, S, head_v_dim, k_h, 1, 0.0f, kv_mem.data(), 1);

        // delta = beta * (v - kv_mem)
        std::vector<float> delta(head_v_dim);
        unsigned int vi = 0;
#ifdef __AVX2__
        __m256 beta_v = _mm256_set1_ps(beta_h);
        for (; vi + 7 < head_v_dim; vi += 8) {
          __m256 vv = _mm256_loadu_ps(&v_h[vi]);
          __m256 km = _mm256_loadu_ps(&kv_mem[vi]);
          _mm256_storeu_ps(&delta[vi],
                           _mm256_mul_ps(beta_v, _mm256_sub_ps(vv, km)));
        }
#elif defined(__ARM_NEON)
        float32x4_t beta_v = vdupq_n_f32(beta_h);
        for (; vi + 3 < head_v_dim; vi += 4) {
          float32x4_t vv = vld1q_f32(&v_h[vi]);
          float32x4_t km = vld1q_f32(&kv_mem[vi]);
          vst1q_f32(&delta[vi], vmulq_f32(beta_v, vsubq_f32(vv, km)));
        }
#endif
        for (; vi < head_v_dim; ++vi)
          delta[vi] = beta_h * (v_h[vi] - kv_mem[vi]);

        // S = decay*S + outer(k, delta) (SIMD optimized hot loop)
        ssm_state_update(S, k_h, delta.data(), decay, head_k_dim, head_v_dim);

        // o = S^T @ q using BLAS sgemv
        std::memset(o_h, 0, head_v_dim * sizeof(float));
        nntrainer::__cblas_sgemv(0 /*RowMajor*/, true /*Trans*/, head_k_dim, head_v_dim,
                      1.0f, S, head_v_dim, q_h, 1, 0.0f, o_h, 1);
      }

      // Gated RMS norm + SiLU (fused, SIMD optimized)
      for (unsigned int h = 0; h < num_v_heads; ++h) {
        gated_rms_silu(y + h * head_v_dim, z_t + h * head_v_dim,
                       y + h * head_v_dim, norm_w, head_v_dim);
      }
    }
  }
}

void GdnSsmCoreLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(0, input_dimensions[0]);
  auto a_dim = input_dimensions[0];
  a_dim.width(num_v_heads);
  context.updateInput(1, a_dim);
  context.updateInput(2, a_dim);
  auto z_dim = input_dimensions[0];
  z_dim.width(value_dim);
  context.updateInput(3, z_dim);
  context.updateOutput(0, z_dim);
}

void GdnSsmCoreLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

#ifdef PLUGGABLE
nntrainer::Layer *create_gdn_ssm_core_layer() {
  return new GdnSsmCoreLayer();
}
void destroy_gdn_ssm_core_layer(nntrainer::Layer *layer) { delete layer; }
extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_gdn_ssm_core_layer, destroy_gdn_ssm_core_layer};
}
#endif

} // namespace causallm
