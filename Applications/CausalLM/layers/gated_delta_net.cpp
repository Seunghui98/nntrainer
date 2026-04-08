// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   gated_delta_net.cpp
 * @date   7 April 2026
 * @brief  Gated DeltaNet (Linear Attention) layer implementation
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cmath>
#include <cstring>
#include <vector>

#include <gated_delta_net.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

/** Utility: softplus(x) = log(1 + exp(x)) */
static inline float softplus(float x) {
  if (x > 20.0f)
    return x; // avoid overflow
  return std::log(1.0f + std::exp(x));
}

/** Utility: sigmoid(x) = 1 / (1 + exp(-x)) */
static inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

/** Utility: silu(x) = x * sigmoid(x) */
static inline float silu(float x) { return x * sigmoid(x); }

/** Utility: L2 normalize a vector in-place */
static void l2_normalize(float *data, unsigned int len, float eps = 1e-6f) {
  float norm_sq = 0.0f;
  for (unsigned int i = 0; i < len; ++i)
    norm_sq += data[i] * data[i];
  float inv_norm = 1.0f / std::sqrt(norm_sq + eps);
  for (unsigned int i = 0; i < len; ++i)
    data[i] *= inv_norm;
}

/** Utility: RMS norm on a vector, multiply by weight */
static void rms_norm(const float *in, float *out, const float *weight,
                     unsigned int len, float eps = 1e-6f) {
  float sum_sq = 0.0f;
  for (unsigned int i = 0; i < len; ++i)
    sum_sq += in[i] * in[i];
  float scale = 1.0f / std::sqrt(sum_sq / len + eps);
  for (unsigned int i = 0; i < len; ++i)
    out[i] = in[i] * scale * weight[i];
}

GatedDeltaNetLayer::GatedDeltaNetLayer() :
  Layer(),
  gdn_props(props::NumVHeads(), props::HeadKDim(), props::HeadVDim(),
            props::ConvKernel()),
  num_v_heads(0),
  head_k_dim(0),
  head_v_dim(0),
  conv_kernel(0),
  key_dim(0),
  value_dim(0),
  conv_dim(0),
  hidden_size(0) {
  wt_idx.fill(std::numeric_limits<unsigned int>::max());
  tensor_idx.fill(std::numeric_limits<unsigned int>::max());
}

void GatedDeltaNetLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, gdn_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[gated_delta_net] Unknown Layer Properties count "
    << std::to_string(values.size());
}

void GatedDeltaNetLayer::finalize(nntrainer::InitLayerContext &context) {
  // Read properties
  num_v_heads = std::get<props::NumVHeads>(gdn_props).get();
  head_k_dim = std::get<props::HeadKDim>(gdn_props).get();
  head_v_dim = std::get<props::HeadVDim>(gdn_props).get();
  conv_kernel = std::get<props::ConvKernel>(gdn_props).get();

  key_dim = num_v_heads * head_k_dim;
  value_dim = num_v_heads * head_v_dim;
  conv_dim = 2 * key_dim + value_dim;

  // Input dim: (batch, 1, seq_len, hidden_size)
  auto input_dims = context.getInputDimensions();
  hidden_size = input_dims[0].width();
  unsigned int batch_size = input_dims[0].batch();

  // Output dim: same as input
  context.setOutputDimensions(input_dims);

  auto wt_type = nntrainer::TensorDim::TensorType(
    context.getFormat(), context.getWeightDataType());
  auto act_type = nntrainer::TensorDim::TensorType(
    context.getFormat(), context.getActivationDataType());

  // Request weights (loaded from binary file in this order)

  // 0: in_proj_qkv (hidden_size, conv_dim) - transposed for matmul
  wt_idx[W_IN_PROJ_QKV] = context.requestWeight(
    nntrainer::TensorDim(1, 1, hidden_size, conv_dim, wt_type),
    nntrainer::Initializer::NONE, nntrainer::WeightRegularizer::NONE, 1.0f,
    0.0f, "in_proj_qkv", false);

  // 1: conv1d (conv_dim, conv_kernel) - stored as (conv_dim, 1, conv_kernel)
  //    but we flatten to (conv_dim, conv_kernel) for our use
  wt_idx[W_CONV1D] = context.requestWeight(
    nntrainer::TensorDim(1, 1, conv_dim, conv_kernel, wt_type),
    nntrainer::Initializer::NONE, nntrainer::WeightRegularizer::NONE, 1.0f,
    0.0f, "conv1d", false);

  // 2: A_log (num_v_heads)
  wt_idx[W_A_LOG] = context.requestWeight(
    nntrainer::TensorDim(1, 1, 1, num_v_heads, wt_type),
    nntrainer::Initializer::NONE, nntrainer::WeightRegularizer::NONE, 1.0f,
    0.0f, "A_log", false);

  // 3: dt_bias (num_v_heads)
  wt_idx[W_DT_BIAS] = context.requestWeight(
    nntrainer::TensorDim(1, 1, 1, num_v_heads, wt_type),
    nntrainer::Initializer::NONE, nntrainer::WeightRegularizer::NONE, 1.0f,
    0.0f, "dt_bias", false);

  // 4: in_proj_a (hidden_size, num_v_heads)
  wt_idx[W_IN_PROJ_A] = context.requestWeight(
    nntrainer::TensorDim(1, 1, hidden_size, num_v_heads, wt_type),
    nntrainer::Initializer::NONE, nntrainer::WeightRegularizer::NONE, 1.0f,
    0.0f, "in_proj_a", false);

  // 5: in_proj_b (hidden_size, num_v_heads)
  wt_idx[W_IN_PROJ_B] = context.requestWeight(
    nntrainer::TensorDim(1, 1, hidden_size, num_v_heads, wt_type),
    nntrainer::Initializer::NONE, nntrainer::WeightRegularizer::NONE, 1.0f,
    0.0f, "in_proj_b", false);

  // 6: in_proj_z (hidden_size, value_dim)
  wt_idx[W_IN_PROJ_Z] = context.requestWeight(
    nntrainer::TensorDim(1, 1, hidden_size, value_dim, wt_type),
    nntrainer::Initializer::NONE, nntrainer::WeightRegularizer::NONE, 1.0f,
    0.0f, "in_proj_z", false);

  // 7: norm (head_v_dim)
  wt_idx[W_NORM] = context.requestWeight(
    nntrainer::TensorDim(1, 1, 1, head_v_dim, wt_type),
    nntrainer::Initializer::NONE, nntrainer::WeightRegularizer::NONE, 1.0f,
    0.0f, "norm", false);

  // 8: out_proj (value_dim, hidden_size)
  wt_idx[W_OUT_PROJ] = context.requestWeight(
    nntrainer::TensorDim(1, 1, value_dim, hidden_size, wt_type),
    nntrainer::Initializer::NONE, nntrainer::WeightRegularizer::NONE, 1.0f,
    0.0f, "out_proj", false);

  // Request tensors for internal state

  // 0: recurrent_state (batch, num_v_heads, head_k_dim, head_v_dim)
  //    stored as (batch, 1, num_v_heads * head_k_dim, head_v_dim)
  tensor_idx[T_RECURRENT_STATE] = context.requestTensor(
    nntrainer::TensorDim(batch_size, 1, num_v_heads * head_k_dim, head_v_dim,
                         act_type),
    "recurrent_state", nntrainer::Initializer::ZEROS, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);

  // 1: conv_state (batch, conv_dim, conv_kernel-1)
  tensor_idx[T_CONV_STATE] = context.requestTensor(
    nntrainer::TensorDim(batch_size, 1, conv_dim, conv_kernel - 1, act_type),
    "conv_state", nntrainer::Initializer::ZEROS, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);
}

void GatedDeltaNetLayer::forwarding(nntrainer::RunLayerContext &context,
                                    bool training) {
  // Not used for inference
}

void GatedDeltaNetLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {

  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  unsigned int batch_size = input.getDim().batch();
  unsigned int seq_len = to - from;

  // Get weights
  nntrainer::Tensor &w_qkv = context.getWeight(wt_idx[W_IN_PROJ_QKV]);
  nntrainer::Tensor &w_conv = context.getWeight(wt_idx[W_CONV1D]);
  nntrainer::Tensor &w_a_log = context.getWeight(wt_idx[W_A_LOG]);
  nntrainer::Tensor &w_dt_bias = context.getWeight(wt_idx[W_DT_BIAS]);
  nntrainer::Tensor &w_proj_a = context.getWeight(wt_idx[W_IN_PROJ_A]);
  nntrainer::Tensor &w_proj_b = context.getWeight(wt_idx[W_IN_PROJ_B]);
  nntrainer::Tensor &w_proj_z = context.getWeight(wt_idx[W_IN_PROJ_Z]);
  nntrainer::Tensor &w_norm = context.getWeight(wt_idx[W_NORM]);
  nntrainer::Tensor &w_out = context.getWeight(wt_idx[W_OUT_PROJ]);

  // Get state tensors
  nntrainer::Tensor &recurrent_state =
    context.getTensor(tensor_idx[T_RECURRENT_STATE]);
  nntrainer::Tensor &conv_state =
    context.getTensor(tensor_idx[T_CONV_STATE]);

  const float *a_log = w_a_log.getData<float>();
  const float *dt_bias = w_dt_bias.getData<float>();
  const float *norm_w = w_norm.getData<float>();

  for (unsigned int b = 0; b < batch_size; ++b) {
    // Get input pointer for this batch: (1, 1, seq_len, hidden_size)
    const float *in_ptr =
      input.getData<float>() + b * input.getDim().getFeatureLen();
    float *out_ptr =
      output.getData<float>() + b * output.getDim().getFeatureLen();

    // Get state pointers for this batch
    float *state_ptr =
      recurrent_state.getData<float>() +
      b * recurrent_state.getDim().getFeatureLen();
    float *conv_st_ptr =
      conv_state.getData<float>() + b * conv_state.getDim().getFeatureLen();

    for (unsigned int t = 0; t < seq_len; ++t) {
      const float *x = in_ptr + t * hidden_size; // current token input
      float *y = out_ptr + t * hidden_size;       // current token output

      // --- Step 1: QKV projection ---
      // mixed_qkv = x @ w_qkv  (hidden_size → conv_dim)
      std::vector<float> mixed_qkv(conv_dim, 0.0f);
      const float *w_qkv_data = w_qkv.getData<float>();
      for (unsigned int i = 0; i < conv_dim; ++i) {
        float sum = 0.0f;
        for (unsigned int j = 0; j < hidden_size; ++j) {
          sum += x[j] * w_qkv_data[j * conv_dim + i];
        }
        mixed_qkv[i] = sum;
      }

      // --- Step 2: Causal conv1d update ---
      // Shift conv state left and append new values
      const float *conv_w = w_conv.getData<float>();
      std::vector<float> conv_out(conv_dim);
      for (unsigned int ch = 0; ch < conv_dim; ++ch) {
        // conv_state for this channel: (conv_kernel - 1) values
        float *ch_state = conv_st_ptr + ch * (conv_kernel - 1);

        // Compute conv output: dot product of [state..., new_val] with kernel
        float val = 0.0f;
        const float *kernel = conv_w + ch * conv_kernel;
        for (unsigned int k = 0; k < conv_kernel - 1; ++k) {
          val += ch_state[k] * kernel[k];
        }
        val += mixed_qkv[ch] * kernel[conv_kernel - 1];

        conv_out[ch] = val;

        // Update state: shift left, append new value
        for (unsigned int k = 0; k < conv_kernel - 2; ++k) {
          ch_state[k] = ch_state[k + 1];
        }
        ch_state[conv_kernel - 2] = mixed_qkv[ch];
      }

      // --- Step 3: SiLU activation ---
      for (unsigned int i = 0; i < conv_dim; ++i) {
        conv_out[i] = silu(conv_out[i]);
      }

      // --- Step 4: Split into Q, K, V ---
      // Q: [0, key_dim), K: [key_dim, 2*key_dim), V: [2*key_dim, conv_dim)
      float *q_data = conv_out.data();
      float *k_data = conv_out.data() + key_dim;
      float *v_data = conv_out.data() + 2 * key_dim;

      // --- Step 5: L2 normalize Q and K per head ---
      for (unsigned int h = 0; h < num_v_heads; ++h) {
        l2_normalize(q_data + h * head_k_dim, head_k_dim);
        l2_normalize(k_data + h * head_k_dim, head_k_dim);
      }

      // --- Step 6: Scale query ---
      float q_scale = 1.0f / std::sqrt((float)head_k_dim);
      for (unsigned int i = 0; i < key_dim; ++i) {
        q_data[i] *= q_scale;
      }

      // --- Step 7: Compute projections a, b, z ---
      // a = x @ w_proj_a → (num_v_heads)
      // b = x @ w_proj_b → (num_v_heads)
      // z = x @ w_proj_z → (value_dim)
      std::vector<float> a_val(num_v_heads, 0.0f);
      std::vector<float> b_val(num_v_heads, 0.0f);
      std::vector<float> z_val(value_dim, 0.0f);

      const float *w_a_data = w_proj_a.getData<float>();
      const float *w_b_data = w_proj_b.getData<float>();
      const float *w_z_data = w_proj_z.getData<float>();

      for (unsigned int i = 0; i < num_v_heads; ++i) {
        float sum_a = 0.0f, sum_b = 0.0f;
        for (unsigned int j = 0; j < hidden_size; ++j) {
          sum_a += x[j] * w_a_data[j * num_v_heads + i];
          sum_b += x[j] * w_b_data[j * num_v_heads + i];
        }
        a_val[i] = sum_a;
        b_val[i] = sum_b;
      }
      for (unsigned int i = 0; i < value_dim; ++i) {
        float sum = 0.0f;
        for (unsigned int j = 0; j < hidden_size; ++j) {
          sum += x[j] * w_z_data[j * value_dim + i];
        }
        z_val[i] = sum;
      }

      // --- Step 8: Compute beta and decay ---
      // beta = sigmoid(b)
      // g = -exp(A_log) * softplus(a + dt_bias)
      std::vector<float> beta(num_v_heads);
      std::vector<float> g_exp(num_v_heads); // exp(g)

      for (unsigned int h = 0; h < num_v_heads; ++h) {
        beta[h] = sigmoid(b_val[h]);
        float g = -std::exp(a_log[h]) * softplus(a_val[h] + dt_bias[h]);
        g_exp[h] = std::exp(g); // decay factor in (0, 1]
      }

      // --- Step 9: Recurrent state update ---
      // For each head:
      //   S = exp(g) * S + outer(k, beta * (v - S^T @ k))
      //   o = S^T @ q  (or equivalently q^T @ S)
      std::vector<float> attn_out(value_dim, 0.0f);

      for (unsigned int h = 0; h < num_v_heads; ++h) {
        // State pointer: (head_k_dim, head_v_dim) for this head
        float *S = state_ptr + h * head_k_dim * head_v_dim;

        // Pointers to Q, K, V for this head
        const float *q_h = q_data + h * head_k_dim;
        const float *k_h = k_data + h * head_k_dim;
        const float *v_h = v_data + h * head_v_dim;

        float decay = g_exp[h];
        float beta_h = beta[h];

        // Compute S^T @ k → kv_mem (head_v_dim)
        // kv_mem[d] = sum_k S[k][d] * k_h[k]
        std::vector<float> kv_mem(head_v_dim, 0.0f);
        for (unsigned int ki = 0; ki < head_k_dim; ++ki) {
          float k_val = k_h[ki];
          for (unsigned int vi = 0; vi < head_v_dim; ++vi) {
            kv_mem[vi] += S[ki * head_v_dim + vi] * k_val;
          }
        }

        // Compute delta = beta * (v - kv_mem)
        std::vector<float> delta(head_v_dim);
        for (unsigned int vi = 0; vi < head_v_dim; ++vi) {
          delta[vi] = beta_h * (v_h[vi] - kv_mem[vi]);
        }

        // Update state: S = decay * S + outer(k, delta)
        for (unsigned int ki = 0; ki < head_k_dim; ++ki) {
          float k_val = k_h[ki];
          for (unsigned int vi = 0; vi < head_v_dim; ++vi) {
            S[ki * head_v_dim + vi] =
              decay * S[ki * head_v_dim + vi] + k_val * delta[vi];
          }
        }

        // Compute output: o = S^T @ q → (head_v_dim)
        // o[d] = sum_k S[k][d] * q_h[k]
        float *o_h = attn_out.data() + h * head_v_dim;
        for (unsigned int ki = 0; ki < head_k_dim; ++ki) {
          float q_val = q_h[ki];
          for (unsigned int vi = 0; vi < head_v_dim; ++vi) {
            o_h[vi] += S[ki * head_v_dim + vi] * q_val;
          }
        }
      }

      // --- Step 10: Gated RMS Norm ---
      // For each head: out = rms_norm(attn_out) * silu(z)
      std::vector<float> normed_out(value_dim);
      for (unsigned int h = 0; h < num_v_heads; ++h) {
        float *o_h = attn_out.data() + h * head_v_dim;
        float *n_h = normed_out.data() + h * head_v_dim;
        const float *z_h = z_val.data() + h * head_v_dim;

        // RMS norm
        rms_norm(o_h, n_h, norm_w, head_v_dim);

        // Gate with silu(z)
        for (unsigned int vi = 0; vi < head_v_dim; ++vi) {
          n_h[vi] *= silu(z_h[vi]);
        }
      }

      // --- Step 11: Output projection ---
      // y = normed_out @ w_out  (value_dim → hidden_size)
      const float *w_out_data = w_out.getData<float>();
      for (unsigned int i = 0; i < hidden_size; ++i) {
        float sum = 0.0f;
        for (unsigned int j = 0; j < value_dim; ++j) {
          sum += normed_out[j] * w_out_data[j * hidden_size + i];
        }
        y[i] = sum;
      }

      // DEBUG: print output stats for first layer, first few tokens
      static int debug_count = 0;
      if (debug_count < 3) {
        float out_norm = 0.0f, in_norm = 0.0f, attn_norm_v = 0.0f;
        for (unsigned int i = 0; i < hidden_size; ++i) {
          out_norm += y[i] * y[i];
          in_norm += x[i] * x[i];
        }
        for (unsigned int i = 0; i < value_dim; ++i)
          attn_norm_v += attn_out[i] * attn_out[i];
        std::cerr << "[DEBUG GDN] "
                  << "in_norm=" << std::sqrt(in_norm)
                  << " attn_out_norm=" << std::sqrt(attn_norm_v)
                  << " final_out_norm=" << std::sqrt(out_norm)
                  << " decay[0]=" << g_exp[0]
                  << " beta[0]=" << beta[0]
                  << std::endl;
        debug_count++;
      }
    }
  }
}

void GatedDeltaNetLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

void GatedDeltaNetLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

#ifdef PLUGGABLE

nntrainer::Layer *create_gated_delta_net_layer() {
  auto layer = new GatedDeltaNetLayer();
  return layer;
}

void destroy_gated_delta_net_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_gated_delta_net_layer, destroy_gated_delta_net_layer};
}

#endif

} // namespace causallm
