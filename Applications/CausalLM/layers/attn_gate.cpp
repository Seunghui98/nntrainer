// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   attn_gate.cpp
 * @date   8 April 2026
 * @brief  Attention output gate: output = input * sigmoid(gate)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 */

#include <cmath>
#include "attn_gate.h"

namespace causallm {

static constexpr size_t OUT_IDX = 0;
static constexpr size_t INPUT_IDX = 0;
static constexpr size_t GATE_IDX = 1;

void AttnGateLayer::finalize(nntrainer::InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void AttnGateLayer::forwarding(nntrainer::RunLayerContext &context,
                               bool training) {}

void AttnGateLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                           unsigned int from, unsigned int to,
                                           bool training) {
  nntrainer::Tensor &in = context.getInput(INPUT_IDX);
  nntrainer::Tensor &gate = context.getInput(GATE_IDX);
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);

  unsigned int iter = to - from;

  for (unsigned int b = 0; b < in.batch(); ++b) {
    for (unsigned int h = 0; h < iter; ++h) {
      float *in_ptr = in.getData<float>() + in.getIndex(b, 0, h, 0);
      float *gate_ptr = gate.getData<float>() + gate.getIndex(b, 0, h, 0);
      float *out_ptr = out.getData<float>() + out.getIndex(b, 0, h, 0);

      for (unsigned int w = 0; w < in.width(); ++w) {
        float sig = 1.0f / (1.0f + std::exp(-gate_ptr[w]));
        out_ptr[w] = in_ptr[w] * sig;
      }
    }
  }
}

void AttnGateLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(INPUT_IDX, input_dimensions[0]);
  context.updateInput(GATE_IDX, input_dimensions[0]);
  context.updateOutput(OUT_IDX, input_dimensions[0]);
}

void AttnGateLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

#ifdef PLUGGABLE

nntrainer::Layer *create_attn_gate_layer() {
  return new AttnGateLayer();
}

void destroy_attn_gate_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_attn_gate_layer,
                                                   destroy_attn_gate_layer};
}

#endif

} // namespace causallm
