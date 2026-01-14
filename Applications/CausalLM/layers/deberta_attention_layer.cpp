// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   deberta_attention_layer.cpp
 * @date   14 January 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Deberta Attention Layer Class
 */

#include <deberta_attention_layer.h>
#include <common_properties.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include <cpu_backend.h>

namespace causallm {

static constexpr size_t INPUT_IDX_Q = 0;
static constexpr size_t INPUT_IDX_K = 1;
static constexpr size_t INPUT_IDX_V = 2;
static constexpr size_t OUTPUT_IDX = 0;

DebertaAttentionLayer::DebertaAttentionLayer() :
  LayerImpl(),
  deberta_props(nntrainer::props::NumHeads(),
                props::MaxPositionEmbeddings(),
                props::MaxRelativePositions(),
                props::C2P(), props::P2C(),
                props::ShareAttKey(),
                props::RelativeAttention(),
                props::PositionBuckets(),
                props::InputLen(),
                nntrainer::props::DisableBias()) {}

DebertaAttentionLayer::~DebertaAttentionLayer() {}

void DebertaAttentionLayer::finalize(nntrainer::InitLayerContext &context) {
  bool share_att_key = std::get<props::ShareAttKey>(deberta_props).get();
  if (!share_att_key) {
    throw nntrainer::exception::not_supported(
      "DebertaAttentionLayer: share_att_key=false is not yet supported.");
  }

  unsigned int expected_inputs = 3;
  if (std::get<props::C2P>(deberta_props).get()) expected_inputs++;
  if (std::get<props::P2C>(deberta_props).get()) expected_inputs++;

  NNTR_THROW_IF(context.getNumInputs() != expected_inputs, std::invalid_argument)
    << "DebertaAttentionLayer expects " << expected_inputs << " inputs (Q, K, V"
    << (std::get<props::C2P>(deberta_props).get() ? ", K_rel" : "")
    << (std::get<props::P2C>(deberta_props).get() ? ", Q_rel" : "") << ")"
    << " but received " << context.getNumInputs();

  const nntrainer::TensorDim &q_dim = context.getInputDimensions()[INPUT_IDX_Q];

  unsigned int num_heads = std::get<nntrainer::props::NumHeads>(deberta_props).get();
  if (num_heads == 0) {
    ml_logw("NumHeads property is not set or 0. Assuming it will be set later or inferred.");
  }

  weight_idx.resize(4);

  // Output same as Q: (B, C, S, dim) in nntrainer terms
  context.setOutputDimensions({q_dim});
}

void DebertaAttentionLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, deberta_props);
  LayerImpl::setProperty(remain_props);
}

void DebertaAttentionLayer::forwarding(nntrainer::RunLayerContext &context,
                                       bool training) {
  nntrainer::Tensor &q = context.getInput(INPUT_IDX_Q);
  nntrainer::Tensor &output = context.getOutput(OUTPUT_IDX);
  output.copyData(q);
}

/**
 * Helper: copy per-head slice from packed width layout.
 * src: (1,1,S,dim) packed as [head0(D), head1(D), ...] for each token
 * dst: (1,1,S,head_dim) contiguous
 */
static inline void slice_head_fp32(const nntrainer::Tensor &src_packed,
                                  nntrainer::Tensor &dst_head,
                                  unsigned int S,
                                  unsigned int dim,
                                  unsigned int head_dim,
                                  unsigned int head_idx) {
  const float *src = src_packed.getData();
  float *dst = dst_head.getData();
  const unsigned int base = head_idx * head_dim;

  for (unsigned int i = 0; i < S; ++i) {
    std::memcpy(dst + i * head_dim,
                src + i * dim + base,
                sizeof(float) * head_dim);
  }
}

/**
 * Helper: write per-head output back into packed width layout.
 * dst_packed: (1,1,S,dim)
 * src_head:   (1,1,S,head_dim)
 */
static inline void write_head_fp32(nntrainer::Tensor &dst_packed,
                                  const nntrainer::Tensor &src_head,
                                  unsigned int S,
                                  unsigned int dim,
                                  unsigned int head_dim,
                                  unsigned int head_idx) {
  float *dst = dst_packed.getData();
  const float *src = src_head.getData();
  const unsigned int base = head_idx * head_dim;

  for (unsigned int i = 0; i < S; ++i) {
    std::memcpy(dst + i * dim + base,
                src + i * head_dim,
                sizeof(float) * head_dim);
  }
}

void DebertaAttentionLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                                  unsigned int from,
                                                  unsigned int to,
                                                  bool training) {
  nntrainer::Tensor &query = context.getInput(INPUT_IDX_Q);
  nntrainer::Tensor &key = context.getInput(INPUT_IDX_K);
  nntrainer::Tensor &value = context.getInput(INPUT_IDX_V);
  nntrainer::Tensor &output = context.getOutput(OUTPUT_IDX);

  unsigned int batch = query.batch();
  unsigned int dim = query.width();
  unsigned int q_len = query.height();

  if ((to - from) != q_len && q_len > 1) {
    ml_logw("Warning: DebertaAttention incremental forwarding received Q with height %d but requested step %d to %d. "
            "Assuming Q is the step input.", q_len, from, to);
  }

  auto get_step_dim = [to, from](const ml::train::TensorDim &dim_) {
    auto step_dim = dim_;
    step_dim.batch(1);
    step_dim.height(to - from);
    return step_dim;
  };

  ml::train::TensorDim query_dim = query.getDim();
  ml::train::TensorDim key_dim = key.getDim();
  ml::train::TensorDim value_dim = value.getDim();
  ml::train::TensorDim output_dim = output.getDim();

  nntrainer::TensorDim query_step_dim = get_step_dim(query_dim);
  nntrainer::TensorDim key_step_dim = get_step_dim(key_dim);
  nntrainer::TensorDim value_step_dim = get_step_dim(value_dim);
  nntrainer::TensorDim out_step_dim = get_step_dim(output_dim);

  unsigned int num_heads = std::get<nntrainer::props::NumHeads>(deberta_props).get();
  NNTR_THROW_IF(num_heads == 0, std::invalid_argument) << "NumHeads must be > 0";

  bool relative_attention = std::get<props::RelativeAttention>(deberta_props).get();
  bool c2p = std::get<props::C2P>(deberta_props).get();
  bool p2c = std::get<props::P2C>(deberta_props).get();

  unsigned int max_relative_positions = std::get<props::MaxRelativePositions>(deberta_props).get();
  if (max_relative_positions < 1) {
    max_relative_positions = std::get<props::MaxPositionEmbeddings>(deberta_props).get();
  }

  NNTR_THROW_IF(dim % num_heads != 0, std::invalid_argument)
    << "dim must be divisible by num_heads. dim=" << dim << " heads=" << num_heads;

  unsigned int head_dim = dim / num_heads;

  int scale_factor_int = 1;
  if (c2p) scale_factor_int += 1;
  if (p2c) scale_factor_int += 1;
  float scale = std::sqrt(static_cast<float>(head_dim * scale_factor_int));
  float scale_factor = 1.0f / scale;

  // NOTE: this code assumes FP32 in this layer for debug parity.
  // If your tensors are FP16/UINT16, you must add dtype branches (like mha_core does).
  NNTR_THROW_IF(query.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "This debug/reference implementation expects FP32 tensors in DebertaAttentionLayer. "
    << "Please add dtype branches if needed.";

  for (unsigned int b = 0; b < batch; ++b) {
    nntrainer::Tensor q_step =
      query.getSharedDataTensor(query_step_dim, b * query_dim.getFeatureLen(), true);
    nntrainer::Tensor k_step =
      key.getSharedDataTensor(key_step_dim, b * key_dim.getFeatureLen(), true);
    nntrainer::Tensor v_step =
      value.getSharedDataTensor(value_step_dim, b * value_dim.getFeatureLen(), true);
    nntrainer::Tensor out_step =
      output.getSharedDataTensor(out_step_dim, b * output_dim.getFeatureLen(), true);

    const unsigned int S_q = q_step.height();
    const unsigned int S_k = k_step.height();

    // Allocate per-head contiguous tensors (mha_core spirit: make layout explicit)
    std::vector<nntrainer::Tensor> qh(num_heads), kh(num_heads), vh(num_heads);
    std::vector<nntrainer::Tensor> score_h(num_heads), ctx_h(num_heads);

    for (unsigned int h = 0; h < num_heads; ++h) {
      nntrainer::TensorDim qh_dim({1, 1, S_q, head_dim}, q_step.getTensorType());
      nntrainer::TensorDim kh_dim({1, 1, S_k, head_dim}, k_step.getTensorType());

      qh[h] = nntrainer::Tensor(qh_dim, true);
      kh[h] = nntrainer::Tensor(kh_dim, true);
      vh[h] = nntrainer::Tensor(kh_dim, true);

      slice_head_fp32(q_step, qh[h], S_q, dim, head_dim, h);
      slice_head_fp32(k_step, kh[h], S_k, dim, head_dim, h);
      slice_head_fp32(v_step, vh[h], S_k, dim, head_dim, h);
    }

    // 0) c2c score per head: (1,1,S_q,head_dim) x (1,1,S_k,head_dim)^T -> (1,1,S_q,S_k)
    for (unsigned int h = 0; h < num_heads; ++h) {
      score_h[h] = qh[h].dot(kh[h], false, true);
      score_h[h].multiply_i(scale_factor);
    }

    if (relative_attention) {
      if (c2p) {
        int rel_input_offset = 4;
        nntrainer::Tensor k_rel = context.getInput(rel_input_offset);

        const unsigned int rel_len = k_rel.height(); 
        // Treat k_rel as packed in width dim=dim (must match)
        NNTR_THROW_IF(k_rel.width() != dim, std::invalid_argument)
          << "K_rel width must equal dim (packed). got " << k_rel.width() << " vs " << dim;

        // Per-head slice of r_k: (1,1,rel_len,head_dim)
        std::vector<nntrainer::Tensor> rkh(num_heads);
        for (unsigned int h = 0; h < num_heads; ++h) {
          nntrainer::TensorDim r_dim({1, 1, rel_len, head_dim}, k_rel.getTensorType());
          rkh[h] = nntrainer::Tensor(r_dim, true);
          slice_head_fp32(k_rel, rkh[h], rel_len, dim, head_dim, h);
        }

        // raw_c2p[h]: (1,1,S_q,rel_len)
        std::vector<nntrainer::Tensor> raw_c2p(num_heads);
        for (unsigned int h = 0; h < num_heads; ++h) {
          raw_c2p[h] = qh[h].dot(rkh[h], false, true);
        }

        // gather bucket and add
        for (unsigned int h = 0; h < num_heads; ++h) {
          float *sc = score_h[h].getData();      // [S_q, S_k]
          float *rc = raw_c2p[h].getData();      // [S_q, rel_len]
          const unsigned int stride_sc_i = S_k;
          const unsigned int stride_rc_i = rel_len;

          for (unsigned int i = 0; i < S_q; ++i) {
            for (unsigned int j = 0; j < S_k; ++j) {
              int q_idx = static_cast<int>(i) + static_cast<int>(from);
              int k_idx = static_cast<int>(j);
              int diff = q_idx - k_idx;
              int bucket_idx = make_log_bucket_position(diff, rel_len, max_relative_positions);
              if (bucket_idx >= 0 && bucket_idx < static_cast<int>(rel_len)) {
                sc[i * stride_sc_i + j] += rc[i * stride_rc_i + bucket_idx] * scale_factor;
              }
            }
          }
        }
      }

      if (p2c) {
        int rel_input_offset = 3;
        nntrainer::Tensor q_rel = context.getInput(rel_input_offset);

        const unsigned int rel_len = q_rel.height();
        NNTR_THROW_IF(q_rel.width() != dim, std::invalid_argument)
          << "Q_rel width must equal dim (packed). got " << q_rel.width() << " vs " << dim;

        std::vector<nntrainer::Tensor> rqh(num_heads);
        for (unsigned int h = 0; h < num_heads; ++h) {
          nntrainer::TensorDim r_dim({1, 1, rel_len, head_dim}, q_rel.getTensorType());
          rqh[h] = nntrainer::Tensor(r_dim, true);
          slice_head_fp32(q_rel, rqh[h], rel_len, dim, head_dim, h);
        }

        // raw_p2c: (1,1,rel_len,S_k) = (1,1,rel_len,head_dim) x (1,1,S_k,head_dim)^T
        // then gather by bucket_idx for each (i,j)
        std::vector<nntrainer::Tensor> raw_p2c(num_heads);
        for (unsigned int h = 0; h < num_heads; ++h) {
          raw_p2c[h] = rqh[h].dot(kh[h], false, true); // [rel_len, S_k]
        }

        for (unsigned int h = 0; h < num_heads; ++h) {
          float *sc = score_h[h].getData();    // [S_q,S_k]
          float *rp = raw_p2c[h].getData();    // [rel_len,S_k]
          const unsigned int stride_sc_i = S_k;
          const unsigned int stride_rp_bucket = S_k;

          for (unsigned int i = 0; i < S_q; ++i) {
            for (unsigned int j = 0; j < S_k; ++j) {
              int q_idx = static_cast<int>(i) + static_cast<int>(from);
              int k_idx = static_cast<int>(j);
              int diff = q_idx - k_idx;
              int bucket_idx = make_log_bucket_position(diff, rel_len, max_relative_positions);
              if (bucket_idx >= 0 && bucket_idx < static_cast<int>(rel_len)) {
                sc[i * stride_sc_i + j] += rp[bucket_idx * stride_rp_bucket + j] * scale_factor;
              }
            }
          }
        }
      }
    } // relative_attention

    // softmax per head
    for (unsigned int h = 0; h < num_heads; ++h) {
      softmax_last_dim(score_h[h]);
    }

    // context per head: (1,1,S_q,S_k) x (1,1,S_k,head_dim) -> (1,1,S_q,head_dim)
    for (unsigned int h = 0; h < num_heads; ++h) {
      ctx_h[h] = score_h[h].dot(vh[h], false, false);
    }

    // merge heads back to packed output (1,1,S_q,dim)
    nntrainer::Tensor merged(out_step.getDim(), true);
    // initialize to 0 just in case
    std::fill(merged.getData(), merged.getData() + merged.size(), 0.0f);

    for (unsigned int h = 0; h < num_heads; ++h) {
      write_head_fp32(merged, ctx_h[h], S_q, dim, head_dim, h);
    }

    out_step.copyData(merged);
  }
}

void DebertaAttentionLayer::softmax_last_dim(nntrainer::Tensor &tensor) {
  float *data = tensor.getData();
  unsigned int input_len = std::get<props::InputLen>(deberta_props).get();
  unsigned int batch = tensor.batch();
  unsigned int channel = tensor.channel();
  unsigned int height = tensor.height();
  unsigned int width = tensor.width();
  unsigned int stride_b = channel * height; 
  unsigned int total_rows = batch * stride_b;

  float minus_inf = -std::numeric_limits<float>::infinity();

  for (unsigned int i = 0; i < total_rows; ++i) {
    float *row_ptr = data + (i * width);
    // for (unsigned int j = input_len-1; j < width; ++j) {
    //   row_ptr[j] = minus_inf;
    // }
    nntrainer::softmax(width, row_ptr, row_ptr);
  }
}

void DebertaAttentionLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for DebertaAttention layer is not supported");
}

void DebertaAttentionLayer::calcGradient(nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcGradient for DebertaAttention layer is not supported");
}

void DebertaAttentionLayer::exportTo(nntrainer::Exporter &exporter,
                                     const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
}

int DebertaAttentionLayer::make_log_bucket_position(int relative_pos,
                                                    int bucket_size,
                                                    int max_position) {
  int sign = (relative_pos >= 0) ? 1 : -1;
  int mid = bucket_size / 2;
  int abs_pos = std::abs(relative_pos);

  if (abs_pos < mid) {
    return relative_pos + mid;
  }

  int max_p = max_position;
  if (max_p == -1) max_p = std::abs(relative_pos);

  double log_val = std::log(static_cast<double>(abs_pos) / mid);
  double log_denom = std::log(static_cast<double>(max_p - 1) / mid);
  int log_pos = static_cast<int>(std::ceil(log_val / log_denom * (mid - 1))) + mid;

  if (log_pos >= bucket_size) log_pos = bucket_size - 1;

  return (sign * log_pos) + mid;
}

} // namespace causallm