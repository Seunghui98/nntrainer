// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   transpose_layer.cpp
 * @date   02 April 2026
 * @brief  This is configurable transpose layer
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tensor.h>
#include <tensor_dim.h>
#include <transpose_layer.h>

#include <sstream>
#include <stdexcept>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

TransposeLayer::TransposeLayer() :
  LayerImpl(), perm_str_("2:0:1"), perm_({2, 0, 1}) {}

void TransposeLayer::parsePerm() {
  NNTR_THROW_IF(perm_str_.empty(), std::invalid_argument)
    << "perm property is empty";

  std::stringstream ss(perm_str_);
  std::string token;
  size_t idx = 0;
  bool used[3] = {false, false, false};

  while (std::getline(ss, token, ':')) {
    NNTR_THROW_IF(idx >= 3, std::invalid_argument)
      << "transpose perm must contain exactly 3 axes";

    int axis = std::stoi(token);

    NNTR_THROW_IF(axis < 0 || axis > 2, std::invalid_argument)
      << "transpose perm axis must be in range [0, 2]";

    NNTR_THROW_IF(used[axis], std::invalid_argument)
      << "transpose perm contains duplicated axis";

    used[axis] = true;
    perm_[idx++] = static_cast<size_t>(axis);
  }

  NNTR_THROW_IF(idx != 3, std::invalid_argument)
    << "transpose perm must contain exactly 3 axes";
}

void TransposeLayer::setProperty(const std::vector<std::string> &values) {
  std::vector<std::string> remain_props;

  for (const auto &value : values) {
    auto pos = value.find('=');
    if (pos == std::string::npos) {
      remain_props.push_back(value);
      continue;
    }

    std::string key = value.substr(0, pos);
    std::string val = value.substr(pos + 1);

    if (key == "perm") {
      perm_str_ = val;
    } else {
      remain_props.push_back(value);
    }
  }

  parsePerm();
  LayerImpl::setProperty(remain_props);
}

void TransposeLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "transpose layer takes only one input";

  parsePerm();

  const auto &in_dim = context.getInputDimensions()[0];
  NNTR_THROW_IF(in_dim.getDataLen() == 0, std::invalid_argument)
    << "input dimension is not set";

  nntrainer::TensorDim out_dim = in_dim;

  size_t chw[3] = {
    in_dim.channel(),
    in_dim.height(),
    in_dim.width(),
  };

  out_dim.channel(chw[perm_[0]]);
  out_dim.height(chw[perm_[1]]);
  out_dim.width(chw[perm_[2]]);
  out_dim.setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions({out_dim});
}

void TransposeLayer::forwarding(nntrainer::RunLayerContext &context,
                                bool training) {
  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output_ = context.getOutput(SINGLE_INOUT_IDX);

  input_.transpose(perm_str_, output_);
}

void TransposeLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                            unsigned int from, unsigned int to,
                                            bool training) {
  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output_ = context.getOutput(SINGLE_INOUT_IDX);

  input_.transpose(perm_str_, output_);
}

void TransposeLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for transpose layer is not supported");
}

void TransposeLayer::exportTo(nntrainer::Exporter &exporter,
                              const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
}

void TransposeLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  NNTR_THROW_IF(input_dimensions.empty(), std::invalid_argument)
    << "input_dimensions is empty";

  parsePerm();

  nntrainer::TensorDim in_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  nntrainer::TensorDim out_dim = context.getOutput(SINGLE_INOUT_IDX).getDim();

  in_dim.batch(input_dimensions[0].batch());
  in_dim.channel(input_dimensions[0].channel());
  in_dim.height(input_dimensions[0].height());
  in_dim.width(input_dimensions[0].width());

  size_t chw[3] = {
    in_dim.channel(),
    in_dim.height(),
    in_dim.width(),
  };

  out_dim.batch(in_dim.batch());
  out_dim.channel(chw[perm_[0]]);
  out_dim.height(chw[perm_[1]]);
  out_dim.width(chw[perm_[2]]);

  context.updateInput(SINGLE_INOUT_IDX, in_dim);
  context.updateOutput(SINGLE_INOUT_IDX, out_dim);
}

#ifdef PLUGGABLE

nntrainer::Layer *create_transpose_layer() {
  auto layer = new TransposeLayer();
  std::cout << "transpose layer created\n";
  return layer;
}

void destroy_transpose_layer(nntrainer::Layer *layer) {
  std::cout << "transpose layer deleted\n";
  delete layer;
}

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_transpose_layer,
                                                   destroy_transpose_layer};
}

#endif

} // namespace causallm
