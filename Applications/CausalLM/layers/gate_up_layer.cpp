// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file	gate_up_layer.cpp
 * @date	17 March 2026
 * @brief	Fused Gate+Up projection layer for FFN (SwiGLU)
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <gate_up_layer.h>

#include <engine.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

GateUpLayer::GateUpLayer() :
  LayerImpl(), gate_up_props(props::UpUnit(), props::GateUnit()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

void GateUpLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "GateUpLayer takes only one input";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  const auto &up_unit = std::get<props::UpUnit>(gate_up_props).get();
  const auto &gate_unit = std::get<props::GateUnit>(gate_up_props).get();

  std::vector<nntrainer::TensorDim> output_dims(2);

  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  bool is_nchw = (context.getFormat() == nntrainer::Tformat::NCHW);
  auto const &in_dim = context.getInputDimensions()[0];

  /** Up out */
  output_dims[Params::UP] = in_dim;
  is_nchw ? output_dims[Params::UP].width(up_unit)
          : output_dims[Params::UP].channel(up_unit);
  output_dims[Params::UP].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  /** Gate out */
  output_dims[Params::GATE] = in_dim;
  is_nchw ? output_dims[Params::GATE].width(gate_unit)
          : output_dims[Params::GATE].channel(gate_unit);
  output_dims[Params::GATE].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  /** Up weight */
  nntrainer::TensorDim weight_dim(
    1, is_nchw ? 1 : up_unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? up_unit : in_dim.channel(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);
  weight_idx[Params::UP] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "up_weight", true);

  /** Gate weight */
  weight_dim.width(gate_unit);
  weight_idx[Params::GATE] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "gate_weight", true);
}

void GateUpLayer::exportTo(nntrainer::Exporter &exporter,
                           const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(gate_up_props, method, this);
}

void GateUpLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, gate_up_props);
  LayerImpl::setProperty(remain_props);
}

void GateUpLayer::forwarding(nntrainer::RunLayerContext &context,
                             bool training) {
  return;
}

void GateUpLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) {
  nntrainer::Tensor &UpWeight = context.getWeight(weight_idx[Params::UP]);
  nntrainer::Tensor &GateWeight = context.getWeight(weight_idx[Params::GATE]);
  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &UpHidden_ = context.getOutput(Params::UP);
  nntrainer::Tensor &GateHidden_ = context.getOutput(Params::GATE);

  nntrainer::TensorDim input_dim = input_.getDim();
  nntrainer::TensorDim input_step_dim = input_dim;
  input_step_dim.batch(1);
  input_step_dim.height(to - from);

  nntrainer::Tensor input_step =
    input_.getSharedDataTensor(input_step_dim, 0, true);

  nntrainer::TensorDim UpHidden_step_dim = UpHidden_.getDim();
  UpHidden_step_dim.batch(1);
  UpHidden_step_dim.height(to - from);
  nntrainer::Tensor UpHidden_step =
    UpHidden_.getSharedDataTensor(UpHidden_step_dim, 0, true);

  nntrainer::TensorDim GateHidden_step_dim = GateHidden_.getDim();
  GateHidden_step_dim.batch(1);
  GateHidden_step_dim.height(to - from);
  nntrainer::Tensor GateHidden_step =
    GateHidden_.getSharedDataTensor(GateHidden_step_dim, 0, true);

  std::vector<nntrainer::Tensor *> Weights({&UpWeight, &GateWeight});
  std::vector<nntrainer::Tensor *> Outputs({&UpHidden_step, &GateHidden_step});

  input_step.dot(Weights, Outputs);
}

void GateUpLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  return;
}

void GateUpLayer::calcGradient(nntrainer::RunLayerContext &context) { return; }

void GateUpLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  ml::train::TensorDim input_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  ml::train::TensorDim UpOutput_dim = context.getOutput(Params::UP).getDim();
  ml::train::TensorDim GateOutput_dim =
    context.getOutput(Params::GATE).getDim();

  input_dim.height(input_dimensions[0].height());
  UpOutput_dim.height(input_dimensions[0].height());
  GateOutput_dim.height(input_dimensions[0].height());

  context.updateInput(SINGLE_INOUT_IDX, input_dim);
  context.updateOutput(Params::UP, UpOutput_dim);
  context.updateOutput(Params::GATE, GateOutput_dim);
}
} // namespace causallm
