// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   gather_layer.cpp
 * @date   02 February 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Custom Gather Layer implementation for CausalLM applications
 */

#include "gather_layer.h"
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <stdexcept>
#include <util_func.h>
#include <layer_context.h>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace causallm {

GatherLayer::GatherLayer() :
  LayerImpl(),
  gather_props(nntrainer::props::Axis()) {}

GatherLayer::~GatherLayer() {}

static unsigned int infer_num_indices_dim(const nntrainer::TensorDim &indexDim) {
  // Expect: [B, 1, 1, K] OR [B, 1, K, 1] OR [B, K, 1, 1]
  unsigned int dims[3] = {
    (unsigned int)indexDim[1],
    (unsigned int)indexDim[2],
    (unsigned int)indexDim[3]
  };

  unsigned int non_one_cnt = 0;
  unsigned int k = 0;
  for (int i = 0; i < 3; ++i) {
    if (dims[i] != 1) { non_one_cnt++; k = dims[i]; }
  }

  if (non_one_cnt == 0) {
    // all ones => no indices
    return 0;
  }
  if (non_one_cnt > 1) {
    throw std::invalid_argument(
      "Index tensor must have exactly one non-1 dim among C/H/W");
  }
  return k;
}

// void GatherLayer::finalize(nntrainer::InitLayerContext &context) {
//   axis = std::get<nntrainer::props::Axis>(gather_props).get();

//   nntrainer::TensorDim inputDim = context.getInputDimensions()[0];
//   nntrainer::TensorDim indexDim = context.getInputDimensions()[1];

//   if (axis < 1 || axis > 3) {
//     throw std::invalid_argument(
//       "The axis property of GatherLayer should be between 1 and 3.");
//   }

//   if (inputDim[0] != indexDim[0]) {
//     throw std::invalid_argument(
//       "The batch size of the input and index should be same.");
//   }

//   nntrainer::TensorDim outputDim = inputDim;
//   // outputDim.setTensorDim(axis, indexDim[axis]);
//   outputDim.setTensorDim(axis, indexDim[axis]);
//   context.setOutputDimensions({outputDim});
// }

void GatherLayer::finalize(nntrainer::InitLayerContext &context) {
  axis = std::get<nntrainer::props::Axis>(gather_props).get();

  nntrainer::TensorDim inputDim = context.getInputDimensions()[0];
  nntrainer::TensorDim indexDim = context.getInputDimensions()[1];

  if (axis < 1 || axis > 3) {
    throw std::invalid_argument("Gather axis must be 1..3");
  }
  if (inputDim[0] != indexDim[0]) {
    throw std::invalid_argument("Batch size of input and index must match");
  }

  const unsigned int num_indices = infer_num_indices_dim(indexDim);
  if (num_indices == 0) {
    throw std::invalid_argument("Index tensor has no valid num_indices dim");
  }

  nntrainer::TensorDim outputDim = inputDim;
  // Gather reduces axis-dim to num_indices
  outputDim.setTensorDim(axis, num_indices);

  context.setOutputDimensions({outputDim});
}

// void GatherLayer::forwarding(nntrainer::RunLayerContext &context,
//                              bool training) {
//   nntrainer::Tensor &output = context.getOutput(0);
//   const nntrainer::Tensor &input = context.getInput(0);
//   const nntrainer::Tensor &index = context.getInput(1);

//   unsigned int batch = input.batch();
//   unsigned int channel = input.channel();
//   unsigned int height = input.height();
//   unsigned int width = input.width();

//   // indices dimension: [batch, 1, 1, num_indices] or [batch, 1, num_indices, 1]...
//   unsigned int index_size = index.size() / batch;

//   const float *input_data = input.getData();
//   const float *index_data = index.getData();
//   float *output_data = output.getData();

//   // Axis 1: Channel
//   if (axis == 1) {
//     for (unsigned int b = 0; b < batch; ++b) {
//       for (unsigned int i = 0; i < index_size; ++i) {
//         unsigned int idx =
//           static_cast<unsigned int>(index_data[b * index_size + i]);
//         if (idx >= channel)
//           throw std::out_of_range("Gather index out of range for axis 1");
//         // Copy [height * width] block
//         const float *src =
//           input_data + b * channel * height * width + idx * height * width;
//         float *dst =
//           output_data + b * index_size * height * width + i * height * width;
//         std::copy(src, src + height * width, dst);
//       }
//     }
//   }
//   // Axis 2: Height
//   else if (axis == 2) {
//     for (unsigned int b = 0; b < batch; ++b) {
//       for (unsigned int c = 0; c < channel; ++c) {
//         for (unsigned int i = 0; i < index_size; ++i) {
//           unsigned int idx =
//             static_cast<unsigned int>(index_data[b * index_size + i]);
//           if (idx >= height)
//             throw std::out_of_range("Gather index out of range for axis 2");
//           const float *src = input_data + b * channel * height * width +
//                              c * height * width + idx * width;
//           float *dst = output_data + b * channel * index_size * width +
//                        c * index_size * width + i * width;
//           // Copy row of width
//           std::copy(src, src + width, dst);
//         }
//       }
//     }
//   }
//   // Axis 3: Width
//   else if (axis == 3) {
//     for (unsigned int b = 0; b < batch; ++b) {
//       for (unsigned int c = 0; c < channel; ++c) {
//         for (unsigned int h = 0; h < height; ++h) {
//           for (unsigned int i = 0; i < index_size; ++i) {
//             unsigned int idx =
//               static_cast<unsigned int>(index_data[b * index_size + i]);
//             if (idx >= width)
//               throw std::out_of_range("Gather index out of range for axis 3");
//             float val = input_data[b * channel * height * width +
//                                    c * height * width + h * width + idx];
//             output_data[b * channel * height * index_size +
//                         c * height * index_size + h * index_size + i] = val;
//           }
//         }
//       }
//     }
//   }
// }

void GatherLayer::forwarding(nntrainer::RunLayerContext &context,
                             bool training) {
  nntrainer::Tensor &output = context.getOutput(0);
  const nntrainer::Tensor &input = context.getInput(0);
  const nntrainer::Tensor &index = context.getInput(1);

  const unsigned int batch = input.batch();
  const unsigned int channel = input.channel();
  const unsigned int height = input.height();
  const unsigned int width = input.width();

  // Index 텐서의 총 원소 개수 (배치당)
  // 예: [B, 1, 1, K] -> K
  const unsigned int index_count = index.size() / batch;

  const float *input_data = input.getData();
  const float *index_data = index.getData();
  float *output_data = output.getData();

  // ----------------------------------------------------------------
  // Axis 2: Gather along Height (Sequence Length) 
  // Input: [B, C, H, W] -> Output: [B, C, K, W]
  // GLiNER에서 가장 핵심적으로 사용하는 부분 (H=512, W=768 -> K=NumSpans, W=768)
  // ----------------------------------------------------------------
  if (axis == 2) {
    // Stride 계산 (한 step 건너뛸 때 이동해야 하는 float 개수)
    const unsigned int input_batch_stride = channel * height * width;
    const unsigned int output_batch_stride = channel * index_count * width;
    
    const unsigned int input_channel_stride = height * width;
    const unsigned int output_channel_stride = index_count * width;

    for (unsigned int b = 0; b < batch; ++b) {
      for (unsigned int c = 0; c < channel; ++c) {
        
        // 현재 Batch, Channel의 시작 포인터
        const float *in_bc_ptr = input_data + (b * input_batch_stride) + (c * input_channel_stride);
        float *out_bc_ptr = output_data + (b * output_batch_stride) + (c * output_channel_stride);
        
        // 해당 배치의 인덱스 시작 포인터
        const float *idx_ptr = index_data + (b * index_count);

        for (unsigned int k = 0; k < index_count; ++k) {
          // [중요] float 인덱스를 안전하게 int로 변환 (1.999 -> 1이 되는 문제 방지)
          unsigned int idx = static_cast<unsigned int>(std::round(idx_ptr[k]));

          if (idx >= height) {
            throw std::out_of_range("GatherLayer: Index out of bounds (axis=2)");
          }

          // idx번째 Row 전체(Width만큼)를 복사
          const float *src = in_bc_ptr + (idx * width);
          float *dst = out_bc_ptr + (k * width);

          std::copy(src, src + width, dst);
        }
      }
    }
  }
  // ----------------------------------------------------------------
  // Axis 3: Gather along Width (Feature Dim)
  // Input: [B, C, H, W] -> Output: [B, C, H, K]
  // ----------------------------------------------------------------
  else if (axis == 3) {
    const unsigned int input_batch_stride = channel * height * width;
    const unsigned int output_batch_stride = channel * height * index_count;

    for (unsigned int b = 0; b < batch; ++b) {
      // 해당 배치의 인덱스 시작 포인터 (Axis 3은 보통 모든 row에 대해 동일한 인덱스를 적용할 수도 있지만,
      // 여기서는 Batch당 인덱스가 다르다고 가정하고 평탄화하여 처리)
      const float *idx_ptr = index_data + (b * index_count);

      for (unsigned int c = 0; c < channel; ++c) {
        for (unsigned int h = 0; h < height; ++h) {
          
          // 현재 처리 중인 Row의 시작 포인터
          const float *in_row_ptr = input_data + (b * input_batch_stride) + 
                                    (c * height * width) + (h * width);
          float *out_row_ptr = output_data + (b * output_batch_stride) + 
                               (c * height * index_count) + (h * index_count);

          for (unsigned int k = 0; k < index_count; ++k) {
            unsigned int idx = static_cast<unsigned int>(std::round(idx_ptr[k]));

            if (idx >= width) {
               throw std::out_of_range("GatherLayer: Index out of bounds (axis=3)");
            }

            // 스칼라 값 1개 복사
            out_row_ptr[k] = in_row_ptr[idx];
          }
        }
      }
    }
  }
  // ----------------------------------------------------------------
  // Axis 1: Gather along Channel
  // Input: [B, C, H, W] -> Output: [B, K, H, W]
  // ----------------------------------------------------------------
  else if (axis == 1) {
    // 구현 구조상 Axis 2와 유사하지만 Loop 순서가 다름
    const unsigned int hw = height * width;
    const unsigned int input_batch_stride = channel * hw;
    const unsigned int output_batch_stride = index_count * hw;

    for (unsigned int b = 0; b < batch; ++b) {
      const float *idx_ptr = index_data + (b * index_count);
      
      for (unsigned int k = 0; k < index_count; ++k) {
        unsigned int idx = static_cast<unsigned int>(std::round(idx_ptr[k]));

        if (idx >= channel) {
            throw std::out_of_range("GatherLayer: Index out of bounds (axis=1)");
        }

        // Channel 블록 전체(H*W) 복사
        const float *src = input_data + (b * input_batch_stride) + (idx * hw);
        float *dst = output_data + (b * output_batch_stride) + (k * hw);

        std::copy(src, src + hw, dst);
      }
    }
  }
}


// void GatherLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
//                                          unsigned int from, unsigned int to,
//                                          bool training) {
//   // For incremental forwarding, we can reuse the forwarding logic
//   // The from/to parameters are typically used for sequence processing,
//   // but for gather operation, we process the entire input
//   forwarding(context, training);
// }

void GatherLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from,
                                         unsigned int to,
                                         bool training) {
  forwarding(context, training);                             
  // nntrainer::Tensor &output = context.getOutput(0);
  // const nntrainer::Tensor &input = context.getInput(0);
  // const nntrainer::Tensor &index = context.getInput(1);

  // const unsigned int batch = input.batch();
  // const unsigned int channel = input.channel();
  // const unsigned int height = input.height();
  // const unsigned int width = input.width();

  // // index is assumed like [B, 1, 1, K] or [B, 1, K, 1] or [B, K, 1, 1]
  // const unsigned int index_size = index.size() / batch;

  // if (index_size == 0) {
  //   return;
  // }

  // // Clamp from/to to valid range
  // const unsigned int begin = std::min(from, index_size);
  // const unsigned int end = std::min(to, index_size);
  // if (begin >= end) {
  //   return;
  // }

  // const float *input_data = input.getData<float>();
  // const float *index_data = index.getData<float>();
  // float *output_data = output.getData<float>();

  // // Axis 1: Channel gather -> output: [B, index_size, H, W]
  // if (axis == 1) {
  //   for (unsigned int b = 0; b < batch; ++b) {
  //     for (unsigned int i = begin; i < end; ++i) {
  //       const unsigned int idx =
  //         static_cast<unsigned int>(index_data[b * index_size + i]);
  //       if (idx >= channel) {
  //         throw std::out_of_range("Gather index out of range for axis 1");
  //       }

  //       const float *src =
  //         input_data + b * channel * height * width + idx * height * width;

  //       float *dst =
  //         output_data + b * index_size * height * width + i * height * width;

  //       std::copy(src, src + height * width, dst);
  //     }
  //   }
  // }
  // // Axis 2: Height gather -> output: [B, C, index_size, W]
  // else if (axis == 2) {
  //   for (unsigned int b = 0; b < batch; ++b) {
  //     for (unsigned int c = 0; c < channel; ++c) {
  //       for (unsigned int i = begin; i < end; ++i) {
  //         const unsigned int idx =
  //           static_cast<unsigned int>(index_data[b * index_size + i]);
  //         if (idx >= height) {
  //           throw std::out_of_range("Gather index out of range for axis 2");
  //         }

  //         const float *src =
  //           input_data + b * channel * height * width +
  //           c * height * width + idx * width;

  //         float *dst =
  //           output_data + b * channel * index_size * width +
  //           c * index_size * width + i * width;

  //         std::copy(src, src + width, dst);
  //       }
  //     }
  //   }
  // }
  // // Axis 3: Width gather -> output: [B, C, H, index_size]
  // else if (axis == 3) {
  //   for (unsigned int b = 0; b < batch; ++b) {
  //     for (unsigned int c = 0; c < channel; ++c) {
  //       for (unsigned int h = 0; h < height; ++h) {
  //         for (unsigned int i = begin; i < end; ++i) {
  //           const unsigned int idx =
  //             static_cast<unsigned int>(index_data[b * index_size + i]);
  //           if (idx >= width) {
  //             throw std::out_of_range("Gather index out of range for axis 3");
  //           }

  //           const float val =
  //             input_data[b * channel * height * width +
  //                        c * height * width + h * width + idx];

  //           output_data[b * channel * height * index_size +
  //                       c * height * index_size + h * index_size + i] = val;
  //         }
  //       }
  //     }
  //   }
  // } else {
  //   throw std::invalid_argument("Gather axis must be 1..3");
  // }
}

void GatherLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  // Backwarding not supported
  throw nntrainer::exception::not_supported(
    "calcDerivative for GatherLayer is not supported");
}

void GatherLayer::calcGradient(nntrainer::RunLayerContext &context) {
  // Backwarding not supported
  throw nntrainer::exception::not_supported(
    "calcGradient for GatherLayer is not supported");
}

void GatherLayer::exportTo(nntrainer::Exporter &exporter,
                           const ml::train::ExportMethods &method) const {
  // Nothing to export for this layer
}

void GatherLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, gather_props);
  LayerImpl::setProperty(remain_props);
}

} // namespace causallm
