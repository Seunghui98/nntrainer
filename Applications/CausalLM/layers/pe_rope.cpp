// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   pe_rope.cpp
 * @date   18 Aug 2026
 * @brief  Implementation of pe_rope — static-table interleaved RoPE layer.
 */

#include "pe_rope.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <thread_manager.h>

namespace causallm {

static constexpr size_t IDX_QK = 0;
static constexpr size_t IDX_SIN = 1;
static constexpr size_t IDX_COS = 2;

void PeRopeLayer::finalize(nntrainer::InitLayerContext &context) {
  const auto &dims = context.getInputDimensions();
  // input[0] is the q/k tensor; output mirrors its shape.
  context.setOutputDimensions({dims[0]});

  if (!std::get<nntrainer::props::SkipPrefill>(pe_rope_props).empty())
    skip_prefill = std::get<nntrainer::props::SkipPrefill>(pe_rope_props).get();
}

void PeRopeLayer::runRotate(nntrainer::RunLayerContext &context,
                            unsigned int from, unsigned int to) {
  nntrainer::Tensor &qk = context.getInput(IDX_QK);
  nntrainer::Tensor &sin = context.getInput(IDX_SIN);
  nntrainer::Tensor &cos = context.getInput(IDX_COS);
  nntrainer::Tensor &out = context.getOutput(IDX_QK);

  // num_prefix_tokens: rows [0, num_prefix_tokens) bypass rotation (CLS).
  // num_heads / head_dim: define the layout of C = num_heads * head_dim.
  // Defaults are the PE-Lang-L14-448 values.
  unsigned int num_prefix =
    std::get<props::NumPrefixTokens>(pe_rope_props).empty()
      ? 1u
      : std::get<props::NumPrefixTokens>(pe_rope_props).get();
  unsigned int num_heads =
    std::get<props::NumHeadsRope>(pe_rope_props).empty()
      ? 16u
      : std::get<props::NumHeadsRope>(pe_rope_props).get();
  unsigned int head_dim =
    std::get<props::HeadDimRope>(pe_rope_props).empty()
      ? 64u
      : std::get<props::HeadDimRope>(pe_rope_props).get();

  const auto &dim = qk.getDim();
  const unsigned int batch = dim.batch();
  const unsigned int C = dim.width();
  // Sanity: the q/k channel count must equal num_heads * head_dim since the
  // converter's rope table is built on the timm [heads, seq, head_dim] layout
  // collapsed head-major into the channel axis.
  // (ponytail: assert guards the invariant; a mismatch is a silent-correctness
  // bug, never continue.)
  if (C != num_heads * head_dim) {
    throw std::invalid_argument(
      "[pe_rope] input channel " + std::to_string(C) +
      " != num_heads*head_dim (" + std::to_string(num_heads) + "*" +
      std::to_string(head_dim) + "=" + std::to_string(num_heads * head_dim) +
      "). Check pe_rope layer properties.");
  }

  const float *qk_data = qk.getData<float>();
  const float *sin_data = sin.getData<float>();
  const float *cos_data = cos.getData<float>();
  float *out_data = out.getData<float>();

  const unsigned int seq = dim.height();
  const unsigned int row_stride = C;
  const unsigned int half_head = head_dim / 2; // pairs per head

  // Clamp `to` to the available rows (the encoder prefill always covers
  // [0, SEQ); incremental mode may pass a slice).
  const unsigned int end = std::min<unsigned int>(to, seq);
  const unsigned int num_rows = (end > from) ? (end - from) : 0;

  // Cost is <0.05% of encoder MACs (file header), but this streams 2*C*4
  // bytes per row through memory 46 times per encoder pass (q and k, 23
  // layers), so it is bandwidth- not compute-bound. Row-parallel via
  // ThreadManager (same pattern as mha_core.cpp and layer_scale.cpp) removes
  // that bottleneck; the inner rotation loop stays scalar (ponytail: no
  // speculative NEON without a measured hotspot — trap §6.13).
  auto &tm = nntrainer::ThreadManager::Global();
  tm.parallel_for(
    0, static_cast<size_t>(batch) * num_rows, [&](size_t idx) {
      const unsigned int b = static_cast<unsigned int>(idx / num_rows);
      const unsigned int r = from + static_cast<unsigned int>(idx % num_rows);
      const float *qk_row =
        qk_data + b * dim.getFeatureLen() + r * row_stride;
      float *out_row = out_data + b * dim.getFeatureLen() + r * row_stride;
      if (r < num_prefix) {
        // CLS row: identity (trap §6.6 — RoPE does NOT touch the prefix token).
        for (unsigned int c = 0; c < C; ++c)
          out_row[c] = qk_row[c];
        return;
      }
      const unsigned int patch_idx = r - num_prefix;
      // sin/cos table is [1, 1, PATCH_N, head_dim] -> indexing patch_idx * head_dim.
      const float *sin_row = sin_data + patch_idx * head_dim;
      const float *cos_row = cos_data + patch_idx * head_dim;
      for (unsigned int h = 0; h < num_heads; ++h) {
        const unsigned int head_off = h * head_dim;
        for (unsigned int i = 0; i < half_head; ++i) {
          const unsigned int even = head_off + 2 * i;
          const unsigned int odd = even + 1;
          const float qe = qk_row[even];
          const float qo = qk_row[odd];
          const float s = sin_row[2 * i];
          const float c = cos_row[2 * i];
          // Interleaved RoPE (trap §6.7): half=False path (NOT rotate_half).
          out_row[even] = qe * c - qo * s;
          out_row[odd] = qo * c + qe * s;
        }
      }
    });
}

void PeRopeLayer::forwarding(nntrainer::RunLayerContext &context,
                             bool training) {
  nntrainer::Tensor &qk = context.getInput(IDX_QK);
  runRotate(context, 0, qk.getDim().height());
}

void PeRopeLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) {
  bool is_prefill = !from || (to - from) > 1;
  if (skip_prefill && is_prefill)
    return;
  runRotate(context, from, to);
}

void PeRopeLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(IDX_QK, input_dimensions[0]);
  context.updateInput(IDX_SIN, input_dimensions[1]);
  context.updateInput(IDX_COS, input_dimensions[2]);
  context.updateOutput(IDX_QK, input_dimensions[0]);
}

void PeRopeLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

#ifdef PLUGGABLE
nntrainer::Layer *create_pe_rope_layer() { return new PeRopeLayer(); }
void destroy_pe_rope_layer(nntrainer::Layer *layer) { delete layer; }
extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_pe_rope_layer, destroy_pe_rope_layer};
}
#endif

} // namespace causallm
