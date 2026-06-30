// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   slice_layer.cpp
 * @date   02 April 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is slice layer class (operation layer)
 */

#include "common_properties.h"
#include "tensor_base.h"
#include <cpu_backend.h>
#include <cstring>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <slice_layer.h>
#include <stdexcept>
#include <type_traits>
#include <util_func.h>

#include <layer_context.h>

namespace nntrainer {

void SliceLayer::finalize(InitLayerContext &context) {
  axis = std::get<props::Axis>(slice_props).get();
  start = std::get<props::StartIndex>(slice_props).get() - 1;
  unsigned int end = std::get<props::EndIndex>(slice_props).get() - 1;

  const TensorDim &in_dim = context.getInputDimensions()[0];
  TensorDim outputDim = context.getInputDimensions()[0];

  for (unsigned int i = 0; i < 4; ++i) {
    if (i == axis) {
      outputDim.setTensorDim(i, end - start);
    } else {
      outputDim.setTensorDim(i, in_dim[i]);
    }
  }

  context.setOutputDimensions({outputDim});
}

/**
 * @brief dtype-correct element copy for the slice. getValue()/setValue()
 * default to T=float, so on an FP16 tensor a float read reinterprets two 2-byte
 * halves as one 4-byte float (and indexes at 2x stride) -> garbage. Dispatch on
 * the actual tensor dtype so the copy stays element-correct for any precision.
 */
template <typename T>
static void sliceForwardT(const Tensor &input, Tensor &output,
                          unsigned int axis, unsigned int start) {
  // Fast path: for contiguous NCHW tensors a slice is just a set of contiguous
  // runs, so memcpy the largest run that the sliced axis allows instead of
  // doing width*height*channel strided getValue() calls (each of which
  // recomputes a 4D index). The YOLOv11 channel splits (axis==1) collapse to a
  // single memcpy per batch. Element-correct for any dtype (raw byte copy).
  if (input.getContiguous() && output.getContiguous()) {
    const T *src = input.getData<T>();
    T *dst = output.getData<T>();
    const size_t Ci = input.channel(), Hi = input.height(), Wi = input.width();
    const size_t N = output.batch(), Co = output.channel(),
                 Ho = output.height(), Wo = output.width();
    if constexpr (std::is_same_v<T, float>) {
      getComputeOps()->slice_contiguous_fp32(src, dst, axis, start, N, Ci, Hi,
                                             Wi, Co, Ho, Wo);
    }
#ifdef ENABLE_FP16
    else if constexpr (std::is_same_v<T, _FP16>) {
      getComputeOps()->slice_contiguous_fp16(src, dst, axis, start, N, Ci, Hi,
                                             Wi, Co, Ho, Wo);
    }
#endif
    return;
  }

  // Fallback: strided per-element copy (non-contiguous views / exotic layouts).
  for (unsigned int b = 0; b < output.batch(); ++b) {
    for (unsigned int c = 0; c < output.channel(); ++c) {
      for (unsigned int h = 0; h < output.height(); ++h) {
        for (unsigned int w = 0; w < output.width(); ++w) {
          unsigned int c_idx = (axis == 1) ? c + start : c;
          unsigned int h_idx = (axis == 2) ? h + start : h;
          unsigned int w_idx = (axis == 3) ? w + start : w;
          output.getValue<T>(b, c, h, w) =
            input.getValue<T>(b, c_idx, h_idx, w_idx);
        }
      }
    }
  }
}

void SliceLayer::forwarding_operation(const Tensor &input, Tensor &output) {
#ifdef ENABLE_FP16
  if (output.getDataType() == ml::train::TensorDim::DataType::FP16) {
    sliceForwardT<_FP16>(input, output, axis, start);
    return;
  }
#endif
  sliceForwardT<float>(input, output, axis, start);
}

template <typename T>
static void sliceDerivT(const Tensor &inDeriv, Tensor &outDeriv,
                        unsigned int axis, unsigned int start) {
  for (unsigned int b = 0; b < inDeriv.batch(); ++b) {
    for (unsigned int c = 0; c < inDeriv.channel(); ++c) {
      for (unsigned int h = 0; h < inDeriv.height(); ++h) {
        for (unsigned int w = 0; w < inDeriv.width(); ++w) {
          unsigned int c_idx = (axis == 1) ? c + start : c;
          unsigned int h_idx = (axis == 2) ? h + start : h;
          unsigned int w_idx = (axis == 3) ? w + start : w;
          outDeriv.getValue<T>(b, c_idx, h_idx, w_idx) =
            inDeriv.getValue<T>(b, c, h, w);
        }
      }
    }
  }
}

void SliceLayer::calcDerivative(RunLayerContext &context) {
  const Tensor &inDeriv = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &outDeriv = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

#ifdef ENABLE_FP16
  if (outDeriv.getDataType() == ml::train::TensorDim::DataType::FP16) {
    sliceDerivT<_FP16>(inDeriv, outDeriv, axis, start);
    return;
  }
#endif
  sliceDerivT<float>(inDeriv, outDeriv, axis, start);
}

void SliceLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, slice_props);
  if (!remain_props.empty()) {
    std::string msg = "[SliceLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

} /* namespace nntrainer */
