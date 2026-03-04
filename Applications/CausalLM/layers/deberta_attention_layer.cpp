// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   deberta_attention_layer.cpp
 * @date   14 January 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Please refer to the following code :
 * https://github.com/huggingface/transformers/blob/5c1c72b/src/transformers/models/deberta/modeling_deberta.py
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

  auto position_buckets = std::get<props::PositionBuckets>(deberta_props).get();

  // NOTE: this code assumes FP32 in this layer for debug parity.
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

    // Allocate per-head contiguous tensors
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

    // 0) c2c score per head
    for (unsigned int h = 0; h < num_heads; ++h) {
      score_h[h] = qh[h].dot(kh[h], false, true);
      score_h[h].multiply_i(scale_factor);
    }

    if (relative_attention) {
      // =========================
      // c2p
      // =========================
      if (c2p) {
        int rel_input_offset = 4;
        nntrainer::Tensor k_rel = context.getInput(rel_input_offset);

        const unsigned int rel_len = k_rel.height();
        NNTR_THROW_IF(k_rel.width() != dim, std::invalid_argument)
          << "K_rel width must equal dim (packed). got " << k_rel.width() << " vs " << dim;

        unsigned int att_span = (position_buckets > 0) ? position_buckets : max_relative_positions;

        // Per-head slice of r_k
        std::vector<nntrainer::Tensor> rkh(num_heads);
        for (unsigned int h = 0; h < num_heads; ++h) {
          nntrainer::TensorDim r_dim({1, 1, rel_len, head_dim}, k_rel.getTensorType());
          rkh[h] = nntrainer::Tensor(r_dim, true);
          slice_head_fp32(k_rel, rkh[h], rel_len, dim, head_dim, h);
        }

        // raw_c2p[h] = qh[h] x rkh[h]^T  => [S_q, rel_len]
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

          const int att_span_i = (position_buckets > 0) ? (int)position_buckets : (int)max_relative_positions;


          for (unsigned int i = 0; i < S_q; ++i) {
            for (unsigned int j = 0; j < S_k; ++j) {
              int diff = (int)i - (int)j;

              // signed bucket_pos (HF make_log_bucket_position output)
              int bucket_pos = make_log_bucket_position(diff, att_span_i, (int)max_relative_positions);

              // HF: c2p_pos = clamp(bucket_pos + att_span, 0, 2*att_span-1)
              int idx = bucket_pos + att_span_i;
              idx = std::max(0, std::min(idx, (int)rel_len - 1));

              sc[i * stride_sc_i + j] += rc[i * stride_rc_i + (unsigned int)idx] * scale_factor;
            }
          }
        }
      }

      // =========================
      // p2c 
      // =========================
      if (p2c) {
        int rel_input_offset = 3;
        nntrainer::Tensor q_rel = context.getInput(rel_input_offset);

        const unsigned int rel_len = q_rel.height();
        NNTR_THROW_IF(q_rel.width() != dim, std::invalid_argument)
          << "Q_rel width must equal dim (packed). got " << q_rel.width() << " vs " << dim;

        unsigned int att_span = (position_buckets > 0) ? position_buckets : max_relative_positions;

        // rqh[h] = [rel_len, head_dim]
        std::vector<nntrainer::Tensor> rqh(num_heads);
        for (unsigned int h = 0; h < num_heads; ++h) {
          nntrainer::TensorDim r_dim({1, 1, rel_len, head_dim}, q_rel.getTensorType());
          rqh[h] = nntrainer::Tensor(r_dim, true);
          slice_head_fp32(q_rel, rqh[h], rel_len, dim, head_dim, h);
        }

        // raw_p2c[h] = key @ rqh^T  => [S_k, rel_len] 
        std::vector<nntrainer::Tensor> raw_p2c(num_heads);
        for (unsigned int h = 0; h < num_heads; ++h) {
          raw_p2c[h] = kh[h].dot(rqh[h], false, true); // [S_k, rel_len]
        }

        for (unsigned int h = 0; h < num_heads; ++h) {
          float *sc = score_h[h].getData();       // [S_q, S_k]
          float *rp = raw_p2c[h].getData();       // [S_k, rel_len]
          const unsigned int stride_sc = S_k;
          const unsigned int stride_rp = rel_len;

          for (unsigned int q = 0; q < S_q; ++q) {
            for (unsigned int k = 0; k < S_k; ++k) {
              int diff_kq = (int)k - (int)q;  

              int r_pos = make_log_bucket_position(diff_kq,
                                                  (int)position_buckets,
                                                  (int)max_relative_positions);

              int idx = (-r_pos) + (int)att_span;
              if (idx < 0) idx = 0;
              if (idx >= (int)rel_len) idx = (int)rel_len - 1;

              float raw = rp[k * stride_rp + idx];
              float add = raw * scale_factor;

              sc[q * stride_sc + k] += add;
            }
          }
        }
      }
    } // relative_attention

    // softmax per head (no masking)
    // Apply softmax only on valid keys [0, valid_k) for each query row.
    {
      const unsigned int valid_k = std::min<unsigned int>(to, S_k);

      for (unsigned int h = 0; h < num_heads; ++h) {
        float *sc = score_h[h].getData(); // shape: [S_q, S_k]

        for (unsigned int q = 0; q < S_q; ++q) {
          const size_t row_start = static_cast<size_t>(q) * S_k;
          const size_t row_end   = row_start + valid_k;

          // softmax only within [row_start, row_end)
          nntrainer::softmax_row_inplace(sc, row_start, row_end, /*num_head=*/1);
        }
      }
    }

    // context per head
    for (unsigned int h = 0; h < num_heads; ++h) {
      ctx_h[h] = score_h[h].dot(vh[h], false, false);
    }
    
    // merge heads back to packed output
    nntrainer::Tensor merged(out_step.getDim(), true);

    std::fill(merged.getData(), merged.getData() + merged.size(), 0.0f);

    for (unsigned int h = 0; h < num_heads; ++h) {
      write_head_fp32(merged, ctx_h[h], S_q, dim, head_dim, h);
    }

    out_step.copyData(merged);
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
                                                    int bucket_size,   // = position_buckets (ex. 256)
                                                    int max_position) {
  const int mid = bucket_size / 2;                 // 128 if bucket_size=256
  const int sign = (relative_pos >= 0) ? 1 : -1;

  // HF: abs_pos = where(|rp|<mid, mid-1, |rp|)
  const int abs_rp = std::abs(relative_pos);
  const int abs_pos = (abs_rp < mid) ? (mid - 1) : abs_rp;

  // HF: log_pos = ceil(log(abs_pos/mid)/log((max_position-1)/mid) * (mid-1)) + mid
  const double num = std::log((double)abs_pos / (double)mid);
  const double den = std::log((double)(max_position - 1) / (double)mid);
  int log_pos = (int)std::ceil(num / den * (mid - 1)) + mid;

  // bucket_pos = where(abs_pos <= mid, relative_pos, log_pos * sign)
  const int bucket_pos = (abs_pos <= mid) ? relative_pos : (log_pos * sign);
  return bucket_pos; 
}

} // namespace causallm
