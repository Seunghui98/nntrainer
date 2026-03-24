// SPDX-License-Identifier: Apache-2.0
/**
* Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
*
* @file   depthwise_conv1d_layer.cpp
* @date   22 March 2026
* @see    https://github.com/nnstreamer/nntrainer
* @author Jijoong Moon <jijoong.moon@samsung.com>
* @bug    No known bugs except for NYI items
* @brief  This is Depthwise Convolution 1D Layer Class for Neural Network
*
*/
#include <algorithm>
#include <cstring>
#include <limits>
#include <string>

#include <depthwise_conv1d_layer.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum DepthwiseConv1DParams { weight, bias };

DepthwiseConv1DLayer::DepthwiseConv1DLayer() :
  LayerImpl(),
  padding({0, 0}),
  conv_props(props::KernelSize(), props::Stride(), props::Padding1D(),
             props::Dilation()) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void DepthwiseConv1DLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "DepthwiseConv1D layer takes only one input";

  const TensorDim &in_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];

  /** Layout requirement: B x 1 x H x W
   *  H = sequence length, W = channel
   */
  NNTR_THROW_IF(in_dim.channel() != 1, std::invalid_argument)
    << "DepthwiseConv1D layer requires input channel dimension to be 1 "
       "(expected input layout: B x 1 x H x W)";

  unsigned int seq_len = in_dim.height();
  unsigned int channels = in_dim.width();

  NNTR_THROW_IF(seq_len == 0 || channels == 0, std::invalid_argument)
    << "DepthwiseConv1D layer requires non-zero height(sequence) and "
       "width(channel)";

  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer = std::get<props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);

  unsigned int kernel_size = std::get<props::KernelSize>(conv_props).get();
  unsigned int stride = std::get<props::Stride>(conv_props).get();
  unsigned int dilation = std::get<props::Dilation>(conv_props).get();

  auto in_t_type = in_dim.getTensorType();
  in_t_type.data_type = context.getWeightDataType();

  /** props::Padding1D::compute() is width-based.
   *  Since this layer uses height as temporal axis, create a fake dim where
   *  width := seq_len so we can reuse the existing padding property logic.
   */
  TensorDim fake_dim = in_dim;
  fake_dim.height(1);
  fake_dim.width(seq_len);

  padding = std::get<props::Padding1D>(conv_props)
              .compute(fake_dim, kernel_size, stride, dilation);

  /** Weight shape: (1, 1, K, W)
   *  K = kernel size along H axis
   *  W = logical channels
   */
  TensorDim weight_dim(1, 1, kernel_size, channels, in_t_type);

  /** Bias shape: (1, 1, 1, W) */
  TensorDim bias_dim(1, 1, 1, channels, in_t_type);

  wt_idx[DepthwiseConv1DParams::weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "filter", true, 0);

  if (disable_bias.empty() || disable_bias.get() == false) {
    wt_idx[DepthwiseConv1DParams::bias] =
      context.requestWeight(bias_dim, bias_initializer,
                            WeightRegularizer::NONE, 1.0f, bias_decay, "bias",
                            true, 0);
  }

  unsigned int eff_in_height = seq_len + padding[0] + padding[1];
  unsigned int eff_k_height = (kernel_size - 1) * dilation + 1;

  NNTR_THROW_IF(eff_in_height < eff_k_height, std::invalid_argument)
    << "Failed to initialize: input height + padding is smaller than "
       "effective kernel size";

  unsigned int out_height = (eff_in_height - eff_k_height) / stride + 1;

  TensorDim out_dim;
  out_dim.batch(in_dim.batch());
  out_dim.channel(1);
  out_dim.height(out_height);
  out_dim.width(channels);
  out_dim.setTensorType(in_dim.getTensorType());

  context.setOutputDimensions({out_dim});
}

void DepthwiseConv1DLayer::forwarding(RunLayerContext &context, bool training) {
  unsigned int kernel_size = std::get<props::KernelSize>(conv_props).get();
  unsigned int stride = std::get<props::Stride>(conv_props).get();
  unsigned int dilation = std::get<props::Dilation>(conv_props).get();

  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &filter_kernel =
    context.getWeight(wt_idx[DepthwiseConv1DParams::weight]);

  const TensorDim &in_dim = input_.getDim();
  const TensorDim &out_dim = hidden_.getDim();

  unsigned int batch = in_dim.batch();
  int in_height = static_cast<int>(in_dim.height());
  unsigned int channels = in_dim.width();
  unsigned int out_height = out_dim.height();
  unsigned int pad_left = padding[0];

  hidden_.setZero();

  auto compute_forward = [&]<typename T>(T) {
    for (unsigned int b = 0; b < batch; ++b) {
      for (unsigned int ow = 0; ow < out_height; ++ow) {
        int base_h =
          static_cast<int>(ow * stride) - static_cast<int>(pad_left);

        for (unsigned int c = 0; c < channels; ++c) {
          T sum = static_cast<T>(0);

          for (unsigned int k = 0; k < kernel_size; ++k) {
            int ih = base_h + static_cast<int>(k * dilation);
            if (ih >= 0 && ih < in_height) {
              sum += input_.getValue<T>(b, 0, ih, c) *
                     filter_kernel.getValue<T>(0, 0, k, c);
            }
          }

          hidden_.setValue(b, 0, ow, c, sum);
        }
      }
    }
  };

  if (input_.getDataType() == nntrainer::Tdatatype::FP32) {
    compute_forward(float{});
  }
#ifdef ENABLE_FP16
  else if (input_.getDataType() == nntrainer::Tdatatype::FP16) {
    compute_forward(_FP16{});
  }
#endif
  else {
    throw std::runtime_error("Not supported datatype");
  }

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &bias_kernel =
      context.getWeight(wt_idx[DepthwiseConv1DParams::bias]);

    int status = hidden_.add_i(bias_kernel);
    if (status != ML_ERROR_NONE) {
      throw std::invalid_argument("[DepthwiseConv1D] adding bias failed");
    }
  }
}

void DepthwiseConv1DLayer::calcDerivative(RunLayerContext &context) {
  unsigned int kernel_size = std::get<props::KernelSize>(conv_props).get();
  unsigned int stride = std::get<props::Stride>(conv_props).get();
  unsigned int dilation = std::get<props::Dilation>(conv_props).get();

  const Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  Tensor &filter_kernel =
    context.getWeight(wt_idx[DepthwiseConv1DParams::weight]);

  const TensorDim &deriv_dim = derivative.getDim();
  unsigned int batch = deriv_dim.batch();
  unsigned int out_height = deriv_dim.height();
  unsigned int channels = deriv_dim.width();
  int in_height = static_cast<int>(input_derivative.getDim().height());
  unsigned int pad_left = padding[0];

  input_derivative.setZero();

  auto compute_deriv = [&]<typename T>(T) {
    for (unsigned int b = 0; b < batch; ++b) {
      for (unsigned int oh = 0; oh < out_height; ++oh) {
        int base_h =
          static_cast<int>(oh * stride) - static_cast<int>(pad_left);

        for (unsigned int c = 0; c < channels; ++c) {
          T grad_out = derivative.getValue<T>(b, 0, oh, c);

          for (unsigned int k = 0; k < kernel_size; ++k) {
            int ih = base_h + static_cast<int>(k * dilation);
            if (ih >= 0 && ih < in_height) {
              T *addr = input_derivative.getAddress<T>(b, 0, ih, c);
              *addr += grad_out * filter_kernel.getValue<T>(0, 0, k, c);
            }
          }
        }
      }
    }
  };

  if (derivative.getDataType() == nntrainer::Tdatatype::FP32) {
    compute_deriv(float{});
  }
#ifdef ENABLE_FP16
  else if (derivative.getDataType() == nntrainer::Tdatatype::FP16) {
    compute_deriv(_FP16{});
  }
#endif
  else {
    throw std::runtime_error("Not supported datatype");
  }
}

void DepthwiseConv1DLayer::calcGradient(RunLayerContext &context) {
  unsigned int kernel_size = std::get<props::KernelSize>(conv_props).get();
  unsigned int stride = std::get<props::Stride>(conv_props).get();
  unsigned int dilation = std::get<props::Dilation>(conv_props).get();

  const Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &delK = context.getWeightGrad(wt_idx[DepthwiseConv1DParams::weight]);

  const TensorDim &deriv_dim = derivative.getDim();
  unsigned int batch = deriv_dim.batch();
  unsigned int out_height = deriv_dim.height();
  unsigned int channels = deriv_dim.width();
  int in_height = static_cast<int>(input_.getDim().height());
  unsigned int pad_left = padding[0];

  delK.setZero();

  auto compute_grad = [&]<typename T>(T) {
    for (unsigned int b = 0; b < batch; ++b) {
      for (unsigned int oh = 0; oh < out_height; ++oh) {
        int base_h =
          static_cast<int>(oh * stride) - static_cast<int>(pad_left);

        for (unsigned int c = 0; c < channels; ++c) {
          T grad_out = derivative.getValue<T>(b, 0, oh, c);

          for (unsigned int k = 0; k < kernel_size; ++k) {
            int ih = base_h + static_cast<int>(k * dilation);
            if (ih >= 0 && ih < in_height) {
              T *addr = delK.getAddress<T>(0, 0, k, c);
              *addr += grad_out * input_.getValue<T>(b, 0, ih, c);
            }
          }
        }
      }
    }
  };

  if (derivative.getDataType() == nntrainer::Tdatatype::FP32) {
    compute_grad(float{});
  }
#ifdef ENABLE_FP16
  else if (derivative.getDataType() == nntrainer::Tdatatype::FP16) {
    compute_grad(_FP16{});
  }
#endif
  else {
    throw std::runtime_error("Not supported datatype");
  }

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &delBias =
      context.getWeightGrad(wt_idx[DepthwiseConv1DParams::bias]);
    delBias.setZero();

    /** derivative shape: (B, 1, out_H, W)
     *  delBias shape:    (1, 1, 1, W)
     *  Reduce over batch and sequence axes.
     */
    derivative.sum({0, 1, 2}, delBias);
  }
}

void DepthwiseConv1DLayer::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(conv_props, method, this);
}

void DepthwiseConv1DLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, conv_props);
  LayerImpl::setProperty(remain_props);
}

} // namespace nntrainer