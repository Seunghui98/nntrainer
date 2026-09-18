// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   rms_norm.cpp
 * @date   19 July 2023
 * @brief  Implementation of custom RMS normalization function
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <act_simd.h> // nntr_quantize_affine_i8 (shared with the W8A8 conv)
#include <cmath>
#include <cpu_backend.h>
#include <iostream>
#include <sstream>
#include <vector>

#include "rms_norm.h"

namespace quick_ai {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void RMSNormLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();

  // out_quant=scale:offset -> UINT8 output for an NPU (qnn_graph) consumer.
  // Same string format as qnn_graph's input_quant_param value part.
  if (!std::get<props::OutQuant>(rms_props).empty()) {
    const std::string spec = std::get<props::OutQuant>(rms_props).get();
    std::vector<std::string> tok;
    std::string t;
    std::istringstream iss(spec);
    while (std::getline(iss, t, ':'))
      tok.push_back(t);
    NNTR_THROW_IF(tok.size() != 2, std::invalid_argument)
      << "[rms_norm] out_quant must be scale:offset, got " << spec;
    oq_scale = std::stof(tok[0]);
    oq_offset = std::stoi(tok[1]);
    NNTR_THROW_IF(oq_scale <= 0.f, std::invalid_argument)
      << "[rms_norm] out_quant scale must be > 0";
    out_quant = true;
    for (auto &d : dim)
      d.setDataType(nntrainer::TensorDim::DataType::UINT8);
  }
  context.setOutputDimensions(dim);

  if (!std::get<nntrainer::props::SkipPrefill>(rms_props).empty())
    skip_prefill = std::get<nntrainer::props::SkipPrefill>(rms_props).get();

  // gamma is unquantized and stored as FP32 in the bin. Request it as FP32
  // regardless of the activation dtype; declaring it FP16 reinterprets the
  // on-disk FP32 bytes as FP16 and corrupts gamma (≈FP16-max garbage). The
  // FP16 forward path casts gamma down to FP16 at the multiply site.
  nntrainer::TensorDim gamma_dim(
    1, 1, 1, dim[0].width(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     nntrainer::TensorDim::DataType::FP32));
  wt_idx[RMSParams::gamma] = context.requestWeight(
    gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", true);
}

void RMSNormLayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {}

void RMSNormLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                          unsigned int from, unsigned int to,
                                          bool training) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);

  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out_dim;

  bool is_prefill = !from || (to - from) > 1;
  if (skip_prefill && is_prefill)
    return;

  in_step_dim.batch(1);
  in_step_dim.height(to - from);
  out_step_dim.batch(1);
  out_step_dim.height(to - from);

  unsigned int b_size = in_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor in_step =
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
    nntrainer::Tensor out_step =
      out.getSharedDataTensor(out_step_dim, b * out_dim.getFeatureLen(), true);

    if (out_quant) {
      forwardQuantized(in_step, out_step, gamma, epsilon);
      continue;
    }

    if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      const auto &dim = in_step.getDim();
#ifdef ENABLE_FP16
      nntrainer::rms_norm_wrt_width_fp32_intrinsic(
        in_step.getData<float>(), out_step.getData<float>(), dim.height(),
        dim.width(), epsilon);

      // DO NOT USE rms_norm_wrt_width_fp16_intrinsic. It causes overflow!

      // nntrainer::rms_norm_wrt_width_fp16_intrinsic(
      //   in_step.getData<float>(), out_step.getData<float>(), dim.height(),
      //   dim.width(), epsilon);
#else

      nntrainer::rms_norm_wrt_width_fp32_intrinsic(
        in_step.getData<float>(), out_step.getData<float>(), dim.height(),
        dim.width(), epsilon);
#endif
#ifdef ENABLE_FP16
    } else if (in_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
      const auto &dim = in_step.getDim();
      // FP16 activation: this kernel accumulates the sum-of-squares in FP32
      // (so a wide residual row cannot overflow FP16) and reads/writes FP16.
      nntrainer::rms_norm_wrt_width_fp16_intrinsic(
        in_step.getData<_FP16>(), out_step.getData<_FP16>(), dim.height(),
        dim.width(), epsilon);
#endif
    } else {
      throw std::invalid_argument(
        "Error: not yet implemented for this data type");
    }
    // gamma (unquantized) may be stored at a different dtype than the FP16
    // activation; cast it to match before the elementwise multiply.
    if (gamma.getDataType() != out_step.getDataType()) {
      nntrainer::Tensor gamma_cast = gamma.clone(out_step.getDataType());
      out_step.multiply_i(gamma_cast);
    } else {
      out_step.multiply_i(gamma);
    }
#ifdef DEBUG
    std::cout << context.getName() << " \n input:" << in_step
              << "output:" << out_step << "gamma:" << gamma << std::endl;
#endif
  }
}

void RMSNormLayer::forwardQuantized(const nntrainer::Tensor &in_step,
                                    nntrainer::Tensor &out_step,
                                    const nntrainer::Tensor &gamma,
                                    float epsilon) {
  const auto &dim = in_step.getDim();
  const size_t rows = dim.height();
  const size_t W = dim.width();
  const float *g = gamma.getData<float>();
  uint8_t *out = out_step.getData<uint8_t>();

  // QNN: real = (q + offset) * scale  ->  q = round(real / scale) - offset.
  // nntr_quantize_affine_i8 computes round((x + off) * inv) - 128 as int8,
  // so with off = -offset * scale the uint8 code is that int8 value + 128,
  // i.e. the same byte with the sign bit flipped.
  const float inv = 1.0f / oq_scale;
  const float off = -(float)oq_offset * oq_scale;

  // One FP32 row: sum of squares in FP32 (an FP16 row can overflow), then
  // scale + gamma + quantize in a single streaming pass over the row.
  static thread_local std::vector<float> rowf;
  static thread_local std::vector<int8_t> rowq;
  if (rowf.size() < W) {
    rowf.resize(W);
    rowq.resize(W);
  }

  auto quant_row = [&](const float *x, uint8_t *dst) {
    float ss = 0.f;
    for (size_t k = 0; k < W; ++k)
      ss += x[k] * x[k];
    const float r = 1.0f / std::sqrt(ss / (float)W + epsilon);
    for (size_t k = 0; k < W; ++k)
      rowf[k] = x[k] * r * g[k];
    nntrainer::nntr_quantize_affine_i8(rowf.data(), rowq.data(), W, inv, off);
    for (size_t k = 0; k < W; ++k)
      dst[k] = (uint8_t)(rowq[k] ^ (int8_t)0x80);
  };

  if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
    const float *x = in_step.getData<float>();
    for (size_t h = 0; h < rows; ++h)
      quant_row(x + h * W, out + h * W);
#ifdef ENABLE_FP16
  } else if (in_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
    const _FP16 *x = in_step.getData<_FP16>();
    static thread_local std::vector<float> xin;
    if (xin.size() < W)
      xin.resize(W);
    for (size_t h = 0; h < rows; ++h) {
      for (size_t k = 0; k < W; ++k)
        xin[k] = (float)x[h * W + k];
      quant_row(xin.data(), out + h * W);
    }
#endif
  } else {
    throw std::invalid_argument(
      "[rms_norm] out_quant supports FP32/FP16 input only");
  }
}

void RMSNormLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  nntrainer::TensorDim od = input_dimensions[0];
  if (out_quant)
    od.setDataType(nntrainer::TensorDim::DataType::UINT8);
  context.updateOutput(SINGLE_INOUT_IDX, od);
}

void RMSNormLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

#ifdef PLUGGABLE

nntrainer::Layer *create_rms_norm_layer() {
  auto layer = new RMSNormLayer();
  return layer;
}

void destroy_rms_norm_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_rms_norm_layer,
                                                   destroy_rms_norm_layer};
}

#endif

} // namespace quick_ai
