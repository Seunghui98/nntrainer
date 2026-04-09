// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   gdn_ssm_core.cpp
 * @date   9 April 2026
 * @brief  GDN SSM Core layer - optimized with SIMD + zero malloc hot path
 */

#include <cmath>
#include <cstring>

#include <gdn_ssm_core.h>
#include <thread_manager.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace causallm {

// ============================================================
// SIMD helpers
// ============================================================

#ifdef __AVX2__
static inline __m256 fast_sigmoid_avx2(__m256 x) {
  alignas(32) float buf[8];
  _mm256_store_ps(buf, x);
  for (int i = 0; i < 8; ++i)
    buf[i] = 1.0f / (1.0f + std::exp(-buf[i]));
  return _mm256_load_ps(buf);
}
static inline __m256 fast_silu_avx2(__m256 x) {
  return _mm256_mul_ps(x, fast_sigmoid_avx2(x));
}
#endif

static inline float softplus(float x) {
  return x > 20.0f ? x : std::log(1.0f + std::exp(x));
}
static inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static inline float silu(float x) { return x * sigmoid(x); }

// L2 normalize in-place
static void l2_normalize(float *data, unsigned int len, float eps = 1e-6f) {
  float norm_sq = 0.0f;
  unsigned int i = 0;
#ifdef __AVX2__
  __m256 sv = _mm256_setzero_ps();
  for (; i + 7 < len; i += 8) {
    __m256 v = _mm256_loadu_ps(&data[i]);
    sv = _mm256_fmadd_ps(v, v, sv);
  }
  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, sv);
  norm_sq = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
#elif defined(__ARM_NEON)
  float32x4_t sv = vdupq_n_f32(0);
  for (; i + 3 < len; i += 4) {
    float32x4_t v = vld1q_f32(&data[i]);
    sv = vfmaq_f32(sv, v, v);
  }
  norm_sq = vaddvq_f32(sv);
#endif
  for (; i < len; ++i) norm_sq += data[i] * data[i];
  float inv = 1.0f / std::sqrt(norm_sq + eps);
  i = 0;
#ifdef __AVX2__
  __m256 sc = _mm256_set1_ps(inv);
  for (; i + 7 < len; i += 8)
    _mm256_storeu_ps(&data[i], _mm256_mul_ps(_mm256_loadu_ps(&data[i]), sc));
#elif defined(__ARM_NEON)
  float32x4_t sc = vdupq_n_f32(inv);
  for (; i + 3 < len; i += 4)
    vst1q_f32(&data[i], vmulq_f32(vld1q_f32(&data[i]), sc));
#endif
  for (; i < len; ++i) data[i] *= inv;
}

// Transposed matvec: out = A^T @ x, A is (rows, cols) row-major
static void matvec_transposed(const float *A, const float *x, float *out,
                              unsigned int rows, unsigned int cols) {
  std::memset(out, 0, cols * sizeof(float));
  for (unsigned int i = 0; i < rows; ++i) {
    const float *row = A + i * cols;
    float xv = x[i];
    unsigned int j = 0;
#ifdef __AVX2__
    __m256 xvv = _mm256_set1_ps(xv);
    for (; j + 7 < cols; j += 8) {
      __m256 o = _mm256_loadu_ps(&out[j]);
      __m256 a = _mm256_loadu_ps(&row[j]);
      _mm256_storeu_ps(&out[j], _mm256_fmadd_ps(a, xvv, o));
    }
#elif defined(__ARM_NEON)
    float32x4_t xvv = vdupq_n_f32(xv);
    for (; j + 3 < cols; j += 4) {
      float32x4_t o = vld1q_f32(&out[j]);
      float32x4_t a = vld1q_f32(&row[j]);
      vst1q_f32(&out[j], vfmaq_f32(o, a, xvv));
    }
#endif
    for (; j < cols; ++j) out[j] += row[j] * xv;
  }
}

// SSM state update: S = decay*S + outer(k, delta)
static void ssm_state_update(float *S, const float *k, const float *delta,
                             float decay, unsigned int k_dim, unsigned int v_dim) {
  for (unsigned int ki = 0; ki < k_dim; ++ki) {
    float kv = k[ki];
    float *row = S + ki * v_dim;
    unsigned int vi = 0;
#ifdef __AVX2__
    __m256 dv = _mm256_set1_ps(decay);
    __m256 kvv = _mm256_set1_ps(kv);
    for (; vi + 7 < v_dim; vi += 8) {
      __m256 s = _mm256_loadu_ps(&row[vi]);
      __m256 d = _mm256_loadu_ps(&delta[vi]);
      _mm256_storeu_ps(&row[vi], _mm256_fmadd_ps(kvv, d, _mm256_mul_ps(s, dv)));
    }
#elif defined(__ARM_NEON)
    float32x4_t dv = vdupq_n_f32(decay);
    float32x4_t kvv = vdupq_n_f32(kv);
    for (; vi + 3 < v_dim; vi += 4) {
      float32x4_t s = vld1q_f32(&row[vi]);
      float32x4_t d = vld1q_f32(&delta[vi]);
      vst1q_f32(&row[vi], vfmaq_f32(vmulq_f32(s, dv), kvv, d));
    }
#endif
    for (; vi < v_dim; ++vi)
      row[vi] = decay * row[vi] + kv * delta[vi];
  }
}

// Fused gated RMS norm + SiLU: out = rms_norm(in, w) * silu(z)
static void gated_rms_silu(const float *in, const float *z, float *out,
                           const float *w, unsigned int len, float eps = 1e-6f) {
  float sum_sq = 0.0f;
  unsigned int i = 0;
#ifdef __AVX2__
  __m256 sv = _mm256_setzero_ps();
  for (; i + 7 < len; i += 8) {
    __m256 v = _mm256_loadu_ps(&in[i]);
    sv = _mm256_fmadd_ps(v, v, sv);
  }
  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, sv);
  sum_sq = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
#elif defined(__ARM_NEON)
  float32x4_t sv = vdupq_n_f32(0);
  for (; i + 3 < len; i += 4) {
    float32x4_t v = vld1q_f32(&in[i]);
    sv = vfmaq_f32(sv, v, v);
  }
  sum_sq = vaddvq_f32(sv);
#endif
  for (; i < len; ++i) sum_sq += in[i] * in[i];
  float scale = 1.0f / std::sqrt(sum_sq / len + eps);
  i = 0;
#ifdef __AVX2__
  __m256 sc = _mm256_set1_ps(scale);
  for (; i + 7 < len; i += 8) {
    __m256 x = _mm256_loadu_ps(&in[i]);
    __m256 wv = _mm256_loadu_ps(&w[i]);
    __m256 zv = _mm256_loadu_ps(&z[i]);
    __m256 normed = _mm256_mul_ps(_mm256_mul_ps(x, sc), wv);
    _mm256_storeu_ps(&out[i], _mm256_mul_ps(normed, fast_silu_avx2(zv)));
  }
#elif defined(__ARM_NEON)
  float32x4_t sc = vdupq_n_f32(scale);
  for (; i + 3 < len; i += 4) {
    float32x4_t x = vld1q_f32(&in[i]);
    float32x4_t wv = vld1q_f32(&w[i]);
    float32x4_t zv = vld1q_f32(&z[i]);
    // silu_neon inline
    alignas(16) float sb[4];
    vst1q_f32(sb, zv);
    for (int j = 0; j < 4; ++j) sb[j] = 1.0f / (1.0f + std::exp(-sb[j]));
    float32x4_t sig = vld1q_f32(sb);
    float32x4_t normed = vmulq_f32(vmulq_f32(x, sc), wv);
    vst1q_f32(&out[i], vmulq_f32(normed, vmulq_f32(zv, sig)));
  }
#endif
  for (; i < len; ++i)
    out[i] = in[i] * scale * w[i] * silu(z[i]);
}

// ============================================================
// Layer implementation
// ============================================================

GdnSsmCoreLayer::GdnSsmCoreLayer() :
  Layer(),
  ssm_props(props::SSMNumVHeads(), props::SSMHeadKDim(), props::SSMHeadVDim()),
  num_v_heads(0), head_k_dim(0), head_v_dim(0), key_dim(0), value_dim(0),
  state_idx(std::numeric_limits<unsigned int>::max()),
  ws_qkv_idx(std::numeric_limits<unsigned int>::max()),
  ws_kv_idx(std::numeric_limits<unsigned int>::max()),
  ws_delta_idx(std::numeric_limits<unsigned int>::max()) {
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
  unsigned int conv_dim = input_dims[0].width();

  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[0] = input_dims[0];
  output_dims[0].width(value_dim);
  context.setOutputDimensions(output_dims);

  auto wt_type = nntrainer::TensorDim::TensorType(
    context.getFormat(), context.getWeightDataType());
  auto act_type = nntrainer::TensorDim::TensorType(
    context.getFormat(), context.getActivationDataType());

  // Weights
  wt_idx[W_A_LOG] = context.requestWeight(
    nntrainer::TensorDim(1, 1, 1, num_v_heads, wt_type),
    nntrainer::Initializer::NONE, nntrainer::WeightRegularizer::NONE,
    1.0f, 0.0f, "A_log", false);
  wt_idx[W_DT_BIAS] = context.requestWeight(
    nntrainer::TensorDim(1, 1, 1, num_v_heads, wt_type),
    nntrainer::Initializer::NONE, nntrainer::WeightRegularizer::NONE,
    1.0f, 0.0f, "dt_bias", false);
  wt_idx[W_NORM] = context.requestWeight(
    nntrainer::TensorDim(1, 1, 1, head_v_dim, wt_type),
    nntrainer::Initializer::NONE, nntrainer::WeightRegularizer::NONE,
    1.0f, 0.0f, "norm", false);

  // Recurrent state
  state_idx = context.requestTensor(
    nntrainer::TensorDim(batch_size, 1, num_v_heads * head_k_dim, head_v_dim,
                         act_type),
    "recurrent_state", nntrainer::Initializer::ZEROS, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);

  // Pre-allocated workspace tensors (ZERO malloc in hot path)
  ws_qkv_idx = context.requestTensor(
    nntrainer::TensorDim(1, 1, 1, conv_dim, act_type),
    "ws_qkv", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  ws_kv_idx = context.requestTensor(
    nntrainer::TensorDim(1, 1, 1, head_v_dim, act_type),
    "ws_kv_mem", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  ws_delta_idx = context.requestTensor(
    nntrainer::TensorDim(1, 1, 1, head_v_dim, act_type),
    "ws_delta", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
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

  // Pre-allocated workspace buffers (NO malloc in hot path)
  float *qkv_buf = context.getTensor(ws_qkv_idx).getData<float>();
  float *kv_mem = context.getTensor(ws_kv_idx).getData<float>();
  float *delta = context.getTensor(ws_delta_idx).getData<float>();

  unsigned int batch_size = conv_out_tensor.getDim().batch();
  unsigned int seq_len = to - from;
  unsigned int conv_dim = conv_out_tensor.getDim().width();

  // Stack-allocated small arrays (no malloc for 16 floats)
  float beta_val[64];  // max 64 heads
  float decay_val[64];

  for (unsigned int b = 0; b < batch_size; ++b) {
    const float *conv_ptr = conv_out_tensor.getData<float>() +
                            b * conv_out_tensor.getDim().getFeatureLen();
    const float *a_ptr = a_tensor.getData<float>() +
                         b * a_tensor.getDim().getFeatureLen();
    const float *b_ptr = b_tensor.getData<float>() +
                         b * b_tensor.getDim().getFeatureLen();
    const float *z_ptr = z_tensor.getData<float>() +
                         b * z_tensor.getDim().getFeatureLen();
    float *out_ptr = output.getData<float>() +
                     b * output.getDim().getFeatureLen();
    float *state_ptr = recurrent_state.getData<float>() +
                       b * recurrent_state.getDim().getFeatureLen();

    for (unsigned int t = 0; t < seq_len; ++t) {
      const float *conv_t = conv_ptr + t * conv_dim;
      const float *a_t = a_ptr + t * num_v_heads;
      const float *b_t = b_ptr + t * num_v_heads;
      const float *z_t = z_ptr + t * value_dim;
      float *y = out_ptr + t * value_dim;

      // Copy conv_out to workspace (pre-allocated, no malloc)
      std::memcpy(qkv_buf, conv_t, conv_dim * sizeof(float));
      float *q_data = qkv_buf;
      float *k_data = qkv_buf + key_dim;
      float *v_data = qkv_buf + 2 * key_dim;

      // L2 normalize Q and K per head (batched)
      for (unsigned int h = 0; h < num_v_heads; ++h) {
        l2_normalize(q_data + h * head_k_dim, head_k_dim);
        l2_normalize(k_data + h * head_k_dim, head_k_dim);
      }

      // Scale query (SIMD)
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
      for (; qi < key_dim; ++qi) q_data[qi] *= q_scale;

      // Beta and decay (stack arrays, no malloc)
      for (unsigned int h = 0; h < num_v_heads; ++h) {
        beta_val[h] = sigmoid(b_t[h]);
        float g = -std::exp(a_log[h]) * softplus(a_t[h] + dt_bias[h]);
        decay_val[h] = std::exp(g);
      }

      // Per-head SSM + gated RMS norm
      // Each head is fully independent: separate S, q, k, v, output regions.
      //
      // For small num_v_heads or small head dimensions, the thread dispatch
      // overhead of ThreadManager::parallel_for exceeds the compute benefit.
      // Use serial path when per-head work is small enough.
      static constexpr size_t PARALLEL_WORK_THRESHOLD = 4096;
      const size_t per_head_work = static_cast<size_t>(head_k_dim) * head_v_dim;
      const bool use_parallel =
        (num_v_heads >= 4) && (per_head_work >= PARALLEL_WORK_THRESHOLD);

      if (use_parallel) {
        auto &tm = nntrainer::ThreadManager::Global();
        tm.parallel_for(
          0, static_cast<size_t>(num_v_heads), [=](size_t h) {
            float *S = state_ptr + h * head_k_dim * head_v_dim;
            const float *q_h = q_data + h * head_k_dim;
            const float *k_h = k_data + h * head_k_dim;
            const float *v_h = v_data + h * head_v_dim;
            float decay = decay_val[h];
            float beta_h = beta_val[h];
            float *o_h = y + h * head_v_dim;

            float local_kv_mem[256];
            float local_delta[256];

            matvec_transposed(S, k_h, local_kv_mem, head_k_dim, head_v_dim);

            unsigned int vi = 0;
#ifdef __AVX2__
            __m256 bv = _mm256_set1_ps(beta_h);
            for (; vi + 7 < head_v_dim; vi += 8) {
              __m256 vv = _mm256_loadu_ps(&v_h[vi]);
              __m256 km = _mm256_loadu_ps(&local_kv_mem[vi]);
              _mm256_storeu_ps(&local_delta[vi],
                               _mm256_mul_ps(bv, _mm256_sub_ps(vv, km)));
            }
#elif defined(__ARM_NEON)
            float32x4_t bv = vdupq_n_f32(beta_h);
            for (; vi + 3 < head_v_dim; vi += 4) {
              float32x4_t vv = vld1q_f32(&v_h[vi]);
              float32x4_t km = vld1q_f32(&local_kv_mem[vi]);
              vst1q_f32(&local_delta[vi],
                         vmulq_f32(bv, vsubq_f32(vv, km)));
            }
#endif
            for (; vi < head_v_dim; ++vi)
              local_delta[vi] = beta_h * (v_h[vi] - local_kv_mem[vi]);

            ssm_state_update(S, k_h, local_delta, decay, head_k_dim, head_v_dim);
            matvec_transposed(S, q_h, o_h, head_k_dim, head_v_dim);
            gated_rms_silu(o_h, z_t + h * head_v_dim, o_h, norm_w, head_v_dim);
          });
      } else {
        // Serial path: avoid thread dispatch overhead for small workloads.
        // Reuses pre-allocated workspace (kv_mem, delta) since no concurrency.
        for (unsigned int h = 0; h < num_v_heads; ++h) {
          float *S = state_ptr + h * head_k_dim * head_v_dim;
          const float *q_h = q_data + h * head_k_dim;
          const float *k_h = k_data + h * head_k_dim;
          const float *v_h = v_data + h * head_v_dim;
          float decay = decay_val[h];
          float beta_h = beta_val[h];
          float *o_h = y + h * head_v_dim;

          matvec_transposed(S, k_h, kv_mem, head_k_dim, head_v_dim);

          unsigned int vi = 0;
#ifdef __AVX2__
          __m256 bv = _mm256_set1_ps(beta_h);
          for (; vi + 7 < head_v_dim; vi += 8) {
            __m256 vv = _mm256_loadu_ps(&v_h[vi]);
            __m256 km = _mm256_loadu_ps(&kv_mem[vi]);
            _mm256_storeu_ps(&delta[vi],
                             _mm256_mul_ps(bv, _mm256_sub_ps(vv, km)));
          }
#elif defined(__ARM_NEON)
          float32x4_t bv = vdupq_n_f32(beta_h);
          for (; vi + 3 < head_v_dim; vi += 4) {
            float32x4_t vv = vld1q_f32(&v_h[vi]);
            float32x4_t km = vld1q_f32(&kv_mem[vi]);
            vst1q_f32(&delta[vi],
                       vmulq_f32(bv, vsubq_f32(vv, km)));
          }
#endif
          for (; vi < head_v_dim; ++vi)
            delta[vi] = beta_h * (v_h[vi] - kv_mem[vi]);

          ssm_state_update(S, k_h, delta, decay, head_k_dim, head_v_dim);
          matvec_transposed(S, q_h, o_h, head_k_dim, head_v_dim);
          gated_rms_silu(o_h, z_t + h * head_v_dim, o_h, norm_w, head_v_dim);
        }
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
nntrainer::Layer *create_gdn_ssm_core_layer() { return new GdnSsmCoreLayer(); }
void destroy_gdn_ssm_core_layer(nntrainer::Layer *layer) { delete layer; }
extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_gdn_ssm_core_layer, destroy_gdn_ssm_core_layer};
}
#endif

} // namespace causallm
