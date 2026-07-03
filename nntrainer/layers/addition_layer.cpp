// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   addition_layer.cpp
 * @date   30 July 2020
 * @see    https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Addition Layer Class for Neural Network
 *
 */

#include <algorithm>
#include <cmath>
#include <vector>

#include <addition_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#include <layer_context.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void AdditionLayer::finalize(InitLayerContext &context) {
  if (!std::get<props::SkipPrefill>(add_props).empty())
    skip_prefill = std::get<props::SkipPrefill>(add_props).get();
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void AdditionLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  const bool int8_out =
    (hidden_.getDataType() == nntrainer::Tdatatype::Q8_0_TW);
  bool any_int8_in = false;
  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
    if (context.getInput(idx).getDataType() == nntrainer::Tdatatype::Q8_0_TW) {
      any_int8_in = true;
      break;
    }
  }

  if (!any_int8_in && !int8_out) {
    /** @todo check possibility for in-place of addition layer */
    for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
      const Tensor &input_ = context.getInput(idx);
      if (!idx) {
        hidden_.copy(input_);
      } else {
        hidden_.add_i(input_);
      }
    }
    return;
  }

  // W4A8 static Q8_0: at least one input (or the output) is a persistent
  // int8 (Q8_0_TW) edge. Dequantize every input to FP32 using ITS OWN
  // registered per-tensor scale (Conv2DLayer::supportInt8ActInput() and
  // NetworkGraph::propagateActivationDataTypes() guarantee any promoted
  // input edge here has a registered scale -- see
  // AdditionLayer::supportInt8ActInput()), sum in FP32, then requantize
  // once into the output using this layer's own registered output scale.
  NNTR_THROW_IF(context.getNumInputs() != 2, std::invalid_argument)
    << "[AdditionLayer] int8 activation path only supports exactly 2 "
       "inputs, got "
    << context.getNumInputs();

  const auto &in_scales =
    std::get<std::array<props::InputActivationScale, 2>>(add_props);
  const size_t n = hidden_.size();
  std::vector<float> acc(n, 0.0f);

  for (unsigned int idx = 0; idx < 2; ++idx) {
    const Tensor &input_ = context.getInput(idx);
    if (input_.getDataType() == nntrainer::Tdatatype::Q8_0_TW) {
      NNTR_THROW_IF(in_scales[idx].empty() || in_scales[idx].get() <= 0.0f,
                    std::invalid_argument)
        << "[AdditionLayer] int8 input " << idx
        << " has no registered InputActivationScale";
      const float scale = in_scales[idx].get();
      const int8_t *src = input_.getData<int8_t>();
      for (size_t i = 0; i < n; ++i)
        acc[i] += static_cast<float>(src[i]) * scale;
    }
#ifdef ENABLE_FP16
    else if (input_.getDataType() == nntrainer::Tdatatype::FP16) {
      const _FP16 *src = input_.getData<_FP16>();
      for (size_t i = 0; i < n; ++i)
        acc[i] += static_cast<float>(src[i]);
    }
#endif
    else {
      const float *src = input_.getData<float>();
      for (size_t i = 0; i < n; ++i)
        acc[i] += src[i];
    }
  }

  if (int8_out) {
    const auto &aos = std::get<props::ActivationScale>(add_props);
    NNTR_THROW_IF(aos.empty() || aos.get() <= 0.0f, std::invalid_argument)
      << "[AdditionLayer] int8 output edge with no registered "
         "ActivationScale";
    const float inv_scale = 1.0f / aos.get();
    int8_t *dst = hidden_.getData<int8_t>();
    for (size_t i = 0; i < n; ++i)
      dst[i] = static_cast<int8_t>(
        std::clamp(std::roundf(acc[i] * inv_scale), -127.0f, 127.0f));
  }
#ifdef ENABLE_FP16
  else if (hidden_.getDataType() == nntrainer::Tdatatype::FP16) {
    _FP16 *dst = hidden_.getData<_FP16>();
    for (size_t i = 0; i < n; ++i)
      dst[i] = static_cast<_FP16>(acc[i]);
  }
#endif
  else {
    float *dst = hidden_.getData<float>();
    for (size_t i = 0; i < n; ++i)
      dst[i] = acc[i];
  }
}

void AdditionLayer::incremental_forwarding(RunLayerContext &context,
                                           unsigned int from, unsigned int to,
                                           bool training) {
  bool is_prefill = !from;
  if (skip_prefill && is_prefill)
    return;

  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  TensorDim hidden_dim = hidden_.getDim();
  TensorDim hidden_step_dim = hidden_dim;

  hidden_step_dim.batch(1);
  hidden_step_dim.height(to - from);

  for (unsigned int b = 0; b < hidden_.batch(); ++b) {
    Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

    /** @todo check possibility for in-place of addition layer */
    for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
      const Tensor &input_ = context.getInput(idx);
      TensorDim input_dim = input_.getDim();

      TensorDim input_step_dim = input_dim;
      input_step_dim.batch(1);
      input_step_dim.height(to - from);

      Tensor input_step = input_.getSharedDataTensor(
        input_step_dim, b * input_dim.getFeatureLen(), true);
      if (!idx) {
        hidden_step.copy(input_step);
      } else {
        hidden_step.add_i(input_step);
      }
    }
  }
}

void AdditionLayer::calcDerivative(RunLayerContext &context) {

  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
    /**
     * TODO: replace this with tensor assignment during optimization.
     * Tensor assignment needs to make sure that the previous connected layers
     * are not inplace
     */
    context.getOutgoingDerivative(idx).copy(
      context.getIncomingDerivative(SINGLE_INOUT_IDX));
  }
}

void AdditionLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, add_props);
  if (!remain_props.empty()) {
    std::string msg = "[AdditionLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

void AdditionLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  for (size_t i = 0; i < context.getNumInputs(); ++i) {
    context.updateInput(i, input_dimensions[0]);
  }
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

} /* namespace nntrainer */
