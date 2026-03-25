// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <limits>
#include <string>

#include <conv_transpose1d_layer.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum ConvTranspose1DParams { weight, bias };

ConvTranspose1DLayer::ConvTranspose1DLayer() :
  LayerImpl(), conv_props(props::KernelSize(), props::Stride()) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void ConvTranspose1DLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "ConvTranspose1D layer takes only one input";

  const TensorDim &in_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];

  // Input layout: (B, 1, H, W)
  NNTR_THROW_IF(in_dim.channel() != 1, std::invalid_argument)
    << "ConvTranspose1D layer requires input layout B x 1 x H x W";

  unsigned int in_height = in_dim.height();
  unsigned int channels = in_dim.width();

  NNTR_THROW_IF(in_height == 0 || channels == 0, std::invalid_argument)
    << "ConvTranspose1D layer requires non-zero height and width";

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

  NNTR_THROW_IF(kernel_size == 0 || stride == 0, std::invalid_argument)
    << "kernel_size and stride must be positive";

  auto in_t_type = in_dim.getTensorType();
  in_t_type.data_type = context.getWeightDataType();

  // Per-channel independent kernel
  TensorDim weight_dim(1, 1, kernel_size, channels, in_t_type);
  TensorDim bias_dim(1, 1, 1, channels, in_t_type);

  wt_idx[ConvTranspose1DParams::weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "filter", true, 0);

  if (disable_bias.empty() || disable_bias.get() == false) {
    wt_idx[ConvTranspose1DParams::bias] =
      context.requestWeight(bias_dim, bias_initializer,
                            WeightRegularizer::NONE, 1.0f, bias_decay, "bias",
                            true, 0);
  }

  unsigned int out_height = (in_height - 1) * stride + kernel_size;

  TensorDim out_dim;
  out_dim.batch(in_dim.batch());
  out_dim.channel(1);
  out_dim.height(out_height);
  out_dim.width(channels);
  out_dim.setTensorType(in_dim.getTensorType());

  context.setOutputDimensions({out_dim});
}

void ConvTranspose1DLayer::forwarding(RunLayerContext &context, bool training) {
  unsigned int kernel_size = std::get<props::KernelSize>(conv_props).get();
  unsigned int stride = std::get<props::Stride>(conv_props).get();

  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &filter_kernel = context.getWeight(wt_idx[ConvTranspose1DParams::weight]);

  const TensorDim &in_dim = input_.getDim();
  const TensorDim &out_dim = hidden_.getDim();

  unsigned int batch = in_dim.batch();
  unsigned int in_height = in_dim.height();
  unsigned int channels = in_dim.width();
  unsigned int out_height = out_dim.height();

  hidden_.setZero();

  auto compute_forward = [&]<typename T>(T) {
    for (unsigned int b = 0; b < batch; ++b) {
      for (unsigned int c = 0; c < channels; ++c) {
        for (unsigned int i = 0; i < in_height; ++i) {
          T in_val = input_.getValue<T>(b, 0, i, c);
          unsigned int base = i * stride;

          for (unsigned int k = 0; k < kernel_size; ++k) {
            unsigned int oh = base + k;
            if (oh < out_height) {
              T *addr = hidden_.getAddress<T>(b, 0, oh, c);
              *addr += in_val * filter_kernel.getValue<T>(0, 0, k, c);
            }
          }
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
    Tensor &bias_kernel = context.getWeight(wt_idx[ConvTranspose1DParams::bias]);
    int status = hidden_.add_i(bias_kernel);
    if (status != ML_ERROR_NONE) {
      throw std::invalid_argument("[ConvTranspose1D] adding bias failed");
    }
  }
}

void ConvTranspose1DLayer::calcDerivative(RunLayerContext &context) {
  unsigned int kernel_size = std::get<props::KernelSize>(conv_props).get();
  unsigned int stride = std::get<props::Stride>(conv_props).get();

  const Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  Tensor &filter_kernel = context.getWeight(wt_idx[ConvTranspose1DParams::weight]);

  const TensorDim &in_dim = input_derivative.getDim();
  const TensorDim &deriv_dim = derivative.getDim();

  unsigned int batch = in_dim.batch();
  unsigned int in_height = in_dim.height();
  unsigned int channels = in_dim.width();
  unsigned int out_height = deriv_dim.height();

  input_derivative.setZero();

  auto compute_deriv = [&]<typename T>(T) {
    for (unsigned int b = 0; b < batch; ++b) {
      for (unsigned int c = 0; c < channels; ++c) {
        for (unsigned int i = 0; i < in_height; ++i) {
          T sum = static_cast<T>(0);
          unsigned int base = i * stride;

          for (unsigned int k = 0; k < kernel_size; ++k) {
            unsigned int oh = base + k;
            if (oh < out_height) {
              sum += derivative.getValue<T>(b, 0, oh, c) *
                     filter_kernel.getValue<T>(0, 0, k, c);
            }
          }

          input_derivative.setValue(b, 0, i, c, sum);
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

void ConvTranspose1DLayer::calcGradient(RunLayerContext &context) {
  unsigned int kernel_size = std::get<props::KernelSize>(conv_props).get();
  unsigned int stride = std::get<props::Stride>(conv_props).get();

  const Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &delK = context.getWeightGrad(wt_idx[ConvTranspose1DParams::weight]);

  const TensorDim &in_dim = input_.getDim();
  const TensorDim &deriv_dim = derivative.getDim();

  unsigned int batch = in_dim.batch();
  unsigned int in_height = in_dim.height();
  unsigned int channels = in_dim.width();
  unsigned int out_height = deriv_dim.height();

  delK.setZero();

  auto compute_grad = [&]<typename T>(T) {
    for (unsigned int b = 0; b < batch; ++b) {
      for (unsigned int c = 0; c < channels; ++c) {
        for (unsigned int i = 0; i < in_height; ++i) {
          T in_val = input_.getValue<T>(b, 0, i, c);
          unsigned int base = i * stride;

          for (unsigned int k = 0; k < kernel_size; ++k) {
            unsigned int oh = base + k;
            if (oh < out_height) {
              T *addr = delK.getAddress<T>(0, 0, k, c);
              *addr += derivative.getValue<T>(b, 0, oh, c) * in_val;
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
    Tensor &delBias = context.getWeightGrad(wt_idx[ConvTranspose1DParams::bias]);
    delBias.setZero();
    derivative.sum({0, 1, 2}, delBias);
  }
}

void ConvTranspose1DLayer::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(conv_props, method, this);
}

void ConvTranspose1DLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, conv_props);
  LayerImpl::setProperty(remain_props);
}

} // namespace nntrainer