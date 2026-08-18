// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   layer_scale.cpp
 * @date   18 Aug 2026
 * @brief  Implementation of LayerScale layer (PE-Lang-L14-448 encoder).
 */

#include "layer_scale.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <thread_manager.h>

namespace causallm {

static constexpr size_t IDX_INPUT = 0;
static constexpr size_t IDX_RESIDUAL = 1;

void LayerScaleLayer::finalize(nntrainer::InitLayerContext &context) {
  const auto &dim = context.getInputDimensions();
  // Both inputs share shape [B, 1, SEQ, C]; output is the same shape.
  context.setOutputDimensions({dim[0]});

  if (!std::get<nntrainer::props::SkipPrefill>(layer_scale_props).empty())
    skip_prefill =
      std::get<nntrainer::props::SkipPrefill>(layer_scale_props).get();

  // gamma weight is a 1-D FP32 tensor of length C (the last input dim).
  // Pinning dtype FP32 keeps LayerScale gamma out of the Q8_0 quant dtype map
  // (plan §3.5 / trap §6.12) regardless of the model's global FC dtype.
  const unsigned int num_channels = dim[0].width();
  nntrainer::TensorDim gamma_dim(
    1, 1, 1, num_channels,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     nntrainer::TensorDim::DataType::FP32));
  wt_idx[0] = context.requestWeight(
    gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", false);
}

void LayerScaleLayer::runScale(nntrainer::RunLayerContext &context,
                               unsigned int from, unsigned int to) {
  nntrainer::Tensor &in = context.getInput(IDX_INPUT);
  nntrainer::Tensor &res = context.getInput(IDX_RESIDUAL);
  nntrainer::Tensor &out = context.getOutput(IDX_INPUT);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[0]);

  const float *in_data = in.getData<float>();
  const float *res_data = res.getData<float>();
  float *out_data = out.getData<float>();
  const float *g = gamma.getData<float>();

  const auto &dim = in.getDim();
  const unsigned int batch = dim.batch();
  const unsigned int C = dim.width();
  const unsigned int row_stride = C; // SEQ is height; one row = C floats

  // The MAC count is negligible (1025*1024*46 = ~48M total, <0.1% of encoder
  // compute — see the file header), but each row round-trips 3*C*4 bytes
  // through memory and this runs 46 times per encoder pass (23 layers * 2
  // sub-blocks), so it is bandwidth- rather than compute-bound. Row-parallel
  // via ThreadManager (already used the same way by mha_core.cpp) removes
  // that bottleneck without needing a NEON kernel; the per-row inner loop
  // itself is left as the simple scalar form (ponytail: no speculative SIMD
  // without a measured hotspot — trap §6.13).
  const unsigned int num_rows = (to > from) ? (to - from) : 0;
  auto &tm = nntrainer::ThreadManager::Global();
  tm.parallel_for(
    0, static_cast<size_t>(batch) * num_rows, [&](size_t idx) {
      const unsigned int b = static_cast<unsigned int>(idx / num_rows);
      const unsigned int r = from + static_cast<unsigned int>(idx % num_rows);
      const float *in_row = in_data + b * dim.getFeatureLen() + r * row_stride;
      const float *res_row =
        res_data + b * dim.getFeatureLen() + r * row_stride;
      float *out_row = out_data + b * dim.getFeatureLen() + r * row_stride;
      for (unsigned int c = 0; c < C; ++c)
        out_row[c] = res_row[c] + g[c] * in_row[c];
    });
}

void LayerScaleLayer::forwarding(nntrainer::RunLayerContext &context,
                                 bool training) {
  nntrainer::Tensor &in = context.getInput(IDX_INPUT);
  runScale(context, 0, in.getDim().height());
}

void LayerScaleLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  bool is_prefill = !from || (to - from) > 1;
  if (skip_prefill && is_prefill)
    return;
  runScale(context, from, to);
}

void LayerScaleLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(IDX_INPUT, input_dimensions[0]);
  context.updateInput(IDX_RESIDUAL, input_dimensions[1]);
  context.updateOutput(IDX_INPUT, input_dimensions[0]);
}

void LayerScaleLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

#ifdef PLUGGABLE
nntrainer::Layer *create_layer_scale_layer() { return new LayerScaleLayer(); }
void destroy_layer_scale_layer(nntrainer::Layer *layer) { delete layer; }
extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_layer_scale_layer,
                                                   destroy_layer_scale_layer};
}
#endif

} // namespace causallm
