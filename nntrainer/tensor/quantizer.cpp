// SPDX-License-Identifier: Apache-2.0
/**
 * @file	quantizer.cpp
 * @date	10 December 2024
 * @brief	This defines quantizers for different types of quantization schemes
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <cpu_backend.h>
#include <math.h>
#include <mem_allocator.h>
#include <quantizer.h>
#include <tensor.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#ifdef ENABLE_HEXKL
#include <htp_backend.h>
#include <remote.h>
#include <sdkl.h>
#endif

namespace nntrainer {

/**
 * @brief Helper function for clipping
 *
 * @tparam T data type
 * @param val value to clip
 * @param lower lower bound
 * @param upper upper bound
 * @return T cliped data
 */
template <typename T> T clip(const T &val, const T &lower, const T &upper) {
  return std::max(lower, std::min(val, upper));
}

void Quantizer::calculateMinMaxValue(Tdatatype qtype) {
  unsigned int N;

  if (qtype == Tdatatype::QINT16 || qtype == Tdatatype::UINT16) {
    N = 16;
  } else if (qtype == Tdatatype::QINT8 || qtype == Tdatatype::UINT8) {
    N = 8;
  } else if (qtype == Tdatatype::QINT4 || qtype == Tdatatype::UINT4) {
    N = 4;
  } else {
    throw std::invalid_argument("[Quantizer] Unsupported data type error.");
  }

  // define minimum and maximum valude representable by the type
  quant_max = (qtype == Tdatatype::UINT16 || qtype == Tdatatype::UINT8 ||
               qtype == Tdatatype::UINT4)
                ? static_cast<long>(std::pow(2, N) - 1)
                : static_cast<long>(std::pow(2, N - 1) - 1);
  quant_min = (qtype == Tdatatype::UINT16 || qtype == Tdatatype::UINT8 ||
               qtype == Tdatatype::UINT4)
                ? 0
                : static_cast<long>(-std::pow(2, N - 1));
}

/**
 * @brief PerTensorAffineQuantizer class
 */
std::unique_ptr<Quantizer> PerTensorAffineQuantizer::create() {
  return std::make_unique<PerTensorAffineQuantizer>();
}

void PerTensorAffineQuantizer::calculateQParams(const Tensor &input,
                                                Tdatatype qtype) {
  float max_val = input.max_abs();
  scale = max_val / ((quant_max - quant_min) / 2.0f);
  scale = std::max(scale, std::numeric_limits<float>::epsilon());

  if (qtype == Tdatatype::UINT4) {
    zero_point =
      (unsigned int)(std::round(scale * input.minValue()) + std::pow(2, 3));
  } else if (qtype == Tdatatype::UINT8) {
    zero_point =
      (unsigned int)(std::round(scale * input.minValue()) + std::pow(2, 7));
  } else if (qtype == Tdatatype::UINT16) {
    zero_point =
      (unsigned int)(std::round(scale * input.minValue()) + std::pow(2, 15));
  } else {
    zero_point = 0;
  }
}

Tensor PerTensorAffineQuantizer::quantize(const Tensor &input,
                                          Tdatatype qtype) {
  // 1. Calculate quantization parameters
  calculateMinMaxValue(qtype);
  calculateQParams(input, qtype);

  // 2. Create output tensor with same dimension but different data type
  TensorDim dim = input.getDim();
  dim.setDataType(qtype);
  Tensor output(dim);

  // 3. perform quantization
  quantize(input, output, &scale, &zero_point);

  return output;
}

Tensor &PerTensorAffineQuantizer::quantize(const Tensor &input, Tensor &output,
                                           float *scales,
                                           unsigned int *zero_points) {
  // Currently only full precision floating point is supported. FP16 is NYI
  NNTR_THROW_IF(input.getDataType() != Tdatatype::FP32, std::invalid_argument)
    << "[Quantizer::quantize] Tensor data type is not floating point.";

  // Check if output tensor is valid
  NNTR_THROW_IF(output.empty(), std::invalid_argument)
    << "[Quantizer::quantize] Cannot quantize to an empty tensor.";

  NNTR_THROW_IF(output.getDataType() == Tdatatype::FP32, std::invalid_argument)
    << "[Quantizer::quantize] Cannot quantize to full precision floating "
       "point.";

  NNTR_THROW_IF(scales == nullptr || std::fpclassify(*scales) == FP_ZERO,
                std::invalid_argument)
    << "[Quantizer::quantize] Output scale factor is invalid.";

  NNTR_THROW_IF(input.size() != output.size(), std::invalid_argument)
    << "[Quantizer::quantize] Tensor size does not match.";

  if (output.getDataType() == Tdatatype::UINT4 ||
      output.getDataType() == Tdatatype::UINT8 ||
      output.getDataType() == Tdatatype::UINT16) {
    NNTR_THROW_IF(zero_points == nullptr, std::invalid_argument)
      << "[Quantizer::quantize] Output zero point is invalid.";
  }

  calculateMinMaxValue(output.getDataType());

  long int val;

  /// @todo this is a naive impl. need optimization
  for (unsigned int b = 0; b < output.batch(); ++b) {
    for (unsigned int c = 0; c < output.channel(); ++c) {
      for (unsigned int h = 0; h < output.height(); ++h) {
        for (unsigned int w = 0; w < output.width(); ++w) {
          val = std::lround(input.getValue(b, c, h, w) / *scales);

          if (output.getDataType() == Tdatatype::UINT4 ||
              output.getDataType() == Tdatatype::UINT8 ||
              output.getDataType() == Tdatatype::UINT16) {
            val += *zero_points;
          }

          output.setValue(b, c, h, w,
                          clip<float>(val, static_cast<float>(quant_min),
                                      static_cast<float>(quant_max)));
        }
      }
    }
  }
  *output.getScale<float>() = *scales;

  if (output.getDataType() == Tdatatype::UINT4 ||
      output.getDataType() == Tdatatype::UINT8 ||
      output.getDataType() == Tdatatype::UINT16) {
    *output.getZeroPoint() = *zero_points;
  }

  return output;
}

Tensor PerTensorAffineQuantizer::dequantize(const Tensor &input,
                                            Tdatatype dtype) {
  Tensor output = input.clone(dtype);
  if (output.getDataType() == Tdatatype::UINT4 ||
      input.getDataType() == Tdatatype::UINT8 ||
      input.getDataType() == Tdatatype::UINT16) {
    output.subtract_i(static_cast<float>(*input.getZeroPoint()));
  }

  output.multiply_i(*input.getScale<float>());

  return output;
}

QScheme PerTensorAffineQuantizer::qscheme() const {
  return QScheme::PER_TENSOR_AFFINE;
}

/**
 * @brief PerChannelAffineQuantizer class
 */
std::unique_ptr<Quantizer> PerChannelAffineQuantizer::create() {
  return std::make_unique<PerChannelAffineQuantizer>();
}

Tensor PerChannelAffineQuantizer::quantize(const Tensor &input,
                                           Tdatatype qtype) {
  /// @todo NYI
  return input;
}

Tensor &PerChannelAffineQuantizer::quantize(const Tensor &input, Tensor &output,
                                            float *scales,
                                            unsigned int *zero_points) {
  /// @todo NYI
  return output;
}

Tensor PerChannelAffineQuantizer::dequantize(const Tensor &input,
                                             Tdatatype dtype) {
  /// @todo NYI
  return input;
}

QScheme PerChannelAffineQuantizer::qscheme() const {
  return QScheme::PER_CHANNEL_AFFINE;
}

/**
 * @brief BinaryCodeBasedQuantizer class
 */
std::unique_ptr<Quantizer> BinaryCodeBasedQuantizer::create() {
  return std::make_unique<BinaryCodeBasedQuantizer>();
}

Tensor BinaryCodeBasedQuantizer::quantize(const Tensor &input,
                                          Tdatatype qtype) {
  /// @todo NYI
  return input;
}

Tensor &BinaryCodeBasedQuantizer::quantize(const Tensor &input, Tensor &output,
                                           float *scales,
                                           unsigned int *zero_points) {
  /// @todo NYI
  return output;
}

Tensor BinaryCodeBasedQuantizer::dequantize(const Tensor &input,
                                            Tdatatype dtype) {
  /// @todo NYI
  return input;
}

QScheme BinaryCodeBasedQuantizer::qscheme() const {
  return QScheme::BINARY_CODE_BASED;
}

/**
 * @brief GgmlQuantizer class
 */
std::unique_ptr<Quantizer> GgmlQuantizer::create() {
  return std::make_unique<GgmlQuantizer>(scheme_);
}

Tensor GgmlQuantizer::quantize(const Tensor &input, Tdatatype qtype) {
  NNTR_THROW_IF(input.getDataType() != Tdatatype::FP32, std::invalid_argument)
    << "[GgmlQuantizer::quantize] Input tensor must be FP32.";

  TensorDim dim = input.getDim();
  unsigned int K = dim.height();
  unsigned int N = dim.width();

  Tdatatype out_dtype;
  switch (scheme_) {
  case QScheme::Q4_Kx8:
    out_dtype = Tdatatype::Q4_K;
    break;
  case QScheme::Q6_K:
    out_dtype = Tdatatype::Q6_K;
    break;
  case QScheme::Q4_0:
    out_dtype = Tdatatype::Q4_0;
    break;
  default:
    throw std::invalid_argument(
      "[GgmlQuantizer::quantize] Unsupported QScheme.");
  }

  // For GGML quantization, we need to transpose the weight first (row-major
  // to column-major style), then quantize per row of the transposed matrix.
  // The quantization functions expect data as (nrow, n_per_row).
  Tensor W_t = input.transpose("0:2:1");
  const float *src = W_t.getData<float>();

  // Create output quantized tensor
  Tensor output({dim.batch(), dim.channel(), K, N, Tformat::NCHW, out_dtype},
                true, nntrainer::Initializer::NONE, "", scheme_);

  // Quantize into a temporary buffer first
  size_t out_size = output.size();
  std::vector<char> tmp(out_size);

  switch (scheme_) {
  case QScheme::Q4_Kx8:
    quantize_q4_K(src, tmp.data(), N, K, nullptr);
    break;
  case QScheme::Q6_K:
    quantize_q6_K(src, tmp.data(), N, K, nullptr);
    break;
  case QScheme::Q4_0:
    quantize_q4_0(src, tmp.data(), N, K, nullptr);
    break;
  default:
    break;
  }

  // For Q4_Kx8 and Q4_0, repack into the optimized layout
  if (scheme_ == QScheme::Q4_Kx8) {
    repack_q4_K(output.getData<uint8_t>(), tmp.data(), out_size, N, K);
  } else if (scheme_ == QScheme::Q4_0) {
    repack_q4_0(output.getData<uint8_t>(), tmp.data(), out_size, N, K);
  } else {
    // Q6_K: copy directly (no repacking needed)
    memcpy(output.getData<uint8_t>(), tmp.data(), out_size);
  }

  return output;
}

Tensor &GgmlQuantizer::quantize(const Tensor &input, Tensor &output,
                                float *scales, unsigned int *zero_points) {
  NNTR_THROW_IF(input.getDataType() != Tdatatype::FP32, std::invalid_argument)
    << "[GgmlQuantizer::quantize] Input tensor must be FP32.";

  NNTR_THROW_IF(output.empty(), std::invalid_argument)
    << "[GgmlQuantizer::quantize] Output tensor is empty.";

  NNTR_THROW_IF(output.q_scheme() != scheme_, std::invalid_argument)
    << "[GgmlQuantizer::quantize] Output tensor's quantization scheme does not "
       "match.";

  TensorDim dim = input.getDim();
  unsigned int K = dim.height();
  unsigned int N = dim.width();

  Tensor W_t = input.transpose("0:2:1");
  const float *src = W_t.getData<float>();

  size_t out_size = output.size();
  std::vector<char> tmp(out_size);

  switch (scheme_) {
  case QScheme::Q4_Kx8:
    quantize_q4_K(src, tmp.data(), N, K, nullptr);
    break;
  case QScheme::Q6_K:
    quantize_q6_K(src, tmp.data(), N, K, nullptr);
    break;
  case QScheme::Q4_0:
    quantize_q4_0(src, tmp.data(), N, K, nullptr);
    break;
  default:
    throw std::invalid_argument(
      "[GgmlQuantizer::quantize] Unsupported QScheme.");
  }

  if (scheme_ == QScheme::Q4_Kx8) {
    repack_q4_K(output.getData<uint8_t>(), tmp.data(), out_size, N, K);
  } else if (scheme_ == QScheme::Q4_0) {
    repack_q4_0(output.getData<uint8_t>(), tmp.data(), out_size, N, K);
  } else {
    memcpy(output.getData<uint8_t>(), tmp.data(), out_size);
  }

  return output;
}

Tensor GgmlQuantizer::dequantize(const Tensor &input, Tdatatype dtype) {
  NNTR_THROW_IF(dtype != Tdatatype::FP32, std::invalid_argument)
    << "[GgmlQuantizer::dequantize] Output dtype must be FP32.";

  TensorDim dim = input.getDim();
  unsigned int K = dim.height();
  unsigned int N = dim.width();
  unsigned int total_elems = K * N;

  Tensor output(dim.batch(), dim.channel(), K, N,
                {Tformat::NCHW, Tdatatype::FP32});
  size_t data_size = input.size();
  std::vector<char> tmp(input.size());

  const void *src = input.getData<uint8_t>();

  switch (scheme_) {
  case QScheme::Q4_Kx8:
    ///@todo unpack should be supported to fully support dequtize
    // dequantize_row_q4_K(src, output.getData(), total_elems);
    throw std::invalid_argument(
      "[GgmlQuantizer::dequantize] Q4_Kx8 is not supported yet.");
    break;
  case QScheme::Q6_K:
    dequantize_row_q6_K(src, output.getData(), total_elems);
    break;
  case QScheme::Q4_0:
    unpack_q4_0(src, tmp.data(), data_size, N, K);
    dequantize_row_q4_0(tmp.data(), output.getData(), total_elems);
    break;
  default:
    throw std::invalid_argument(
      "[GgmlQuantizer::dequantize] Unsupported QScheme.");
  }

  return output;
}

QScheme GgmlQuantizer::qscheme() const { return scheme_; }

Tensor quantize_qint8_weight(const Tensor &input, bool transpose_input) {
  NNTR_THROW_IF(input.getDataType() != Tdatatype::FP32, std::invalid_argument)
    << "[quantize_qint8_weight] Input tensor must be FP32.";

  Tensor logical_weight =
    transpose_input ? input.transpose("0:2:1") : input.clone();
  const unsigned int N = logical_weight.height();
  const unsigned int K = logical_weight.width();

  NNTR_THROW_IF(N == 0 || K == 0, std::invalid_argument)
    << "[quantize_qint8_weight] Weight dimensions must be non-zero.";
  NNTR_THROW_IF(N % 32 != 0, std::runtime_error)
    << "[quantize_qint8_weight] QINT8 quantization requires N divisible by 32, "
    << "but got N=" << N;

  TensorDim quant_dim = logical_weight.getDim();
  quant_dim.setTensorType({logical_weight.getFormat(), Tdatatype::QINT8});
  Tensor output(quant_dim, true, Initializer::NONE, "",
                QScheme::PER_CHANNEL_AFFINE);

  const float *src = logical_weight.getData<float>();
  int8_t *dst = output.getData<int8_t>();
  float *scales = output.getScale<float>();
  int32_t *zp_corr = output.getZpCorr<int32_t>();

  for (unsigned int n = 0; n < N; ++n) {
    float max_abs = 0.0f;
    for (unsigned int k = 0; k < K; ++k) {
      const float value = src[n * K + k];
      NNTR_THROW_IF(!std::isfinite(value), std::runtime_error)
        << "[quantize_qint8_weight] Encountered non-finite weight value.";
      max_abs = std::max(max_abs, std::fabs(value));
    }

    float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    if (!std::isfinite(scale) || scale <= 0.0f)
      scale = 1.0f;

    scales[n] = scale;

    int32_t row_sum = 0;
    for (unsigned int k = 0; k < K; ++k) {
      const float value = src[n * K + k] / scale;
      const long q = std::lround(value);
      const int8_t clamped =
        static_cast<int8_t>(std::max<long>(-127, std::min<long>(127, q)));
      dst[n * K + k] = clamped;
      row_sum += static_cast<int32_t>(clamped);
    }
    zp_corr[n] = 128 * row_sum;
  }

#ifdef ENABLE_HEXKL
  const size_t packed_bytes =
    static_cast<size_t>(N) * static_cast<size_t>(K) * sizeof(int8_t);
  void *tmp = nullptr;
  int rc = -1;

  if (nntrainer::HtpBackend::global().enabled()) {
    rc = sdkl_npu_alloc(packed_bytes, &tmp);
  }

  if (rc == 0 && tmp != nullptr) {
    // Route the in-place SDKL layout conversion through an NPU-accessible
    // buffer. Android device tests showed host-only buffers can fault here.
    std::memcpy(tmp, dst, packed_bytes);
    rc = sdkl_cpu_rm_to_wh_i8_inplace(static_cast<size_t>(N),
                                      static_cast<size_t>(K),
                                      static_cast<int8_t *>(tmp));
    if (rc == 0)
      std::memcpy(dst, tmp, packed_bytes);

    const int free_rc = sdkl_npu_free(tmp);
    NNTR_THROW_IF(free_rc != 0, std::runtime_error)
      << "[quantize_qint8_weight] Failed to free temporary SDKL buffer.";
  } else {
    nntrainer::MemAllocator host_alloc;
    host_alloc.alloc(&tmp, packed_bytes, 64);
    std::memcpy(tmp, dst, packed_bytes);
    rc = sdkl_cpu_rm_to_wh_i8_inplace(static_cast<size_t>(N),
                                      static_cast<size_t>(K),
                                      static_cast<int8_t *>(tmp));
    if (rc == 0)
      std::memcpy(dst, tmp, packed_bytes);
    host_alloc.free(tmp);
  }

  NNTR_THROW_IF(rc != 0, std::runtime_error)
    << "[quantize_qint8_weight] Failed to convert QINT8 weight to WH layout.";
#endif

  return output;
}

Tensor quantize_qint4_weight(const Tensor &input, bool transpose_input) {
  NNTR_THROW_IF(input.getDataType() != Tdatatype::FP32, std::invalid_argument)
    << "[quantize_qint4_weight] Input tensor must be FP32.";

  Tensor logical_weight =
    transpose_input ? input.transpose("0:2:1") : input.clone();
  const unsigned int N = logical_weight.height();
  const unsigned int K = logical_weight.width();

  NNTR_THROW_IF(N == 0 || K == 0, std::invalid_argument)
    << "[quantize_qint4_weight] Weight dimensions must be non-zero.";
  NNTR_THROW_IF(N % 32 != 0, std::runtime_error)
    << "[quantize_qint4_weight] QINT4 quantization requires N divisible by 32, "
    << "but got N=" << N;
  // int4 values are packed two-per-byte for the NPU, so K must be even.
  NNTR_THROW_IF(K % 2 != 0, std::runtime_error)
    << "[quantize_qint4_weight] QINT4 quantization requires K to be even, "
    << "but got K=" << K;

  // QINT4_HTP is CharTensor-backed; PER_CHANNEL_AFFINE_I4 selects the u8i4
  // kernel and the dtype lets load/dispatch tell u8i4 apart from u8i8. Emit the
  // output tensor in the runtime FC weight layout [K, N] (in, out) so that
  // scale_size() (= width() = N output channels) agrees between bake and load;
  // the WH-packed bytes are layout-agnostic (the kernel takes N, K from the
  // matmul dims), so only the dim metadata differs from the logical [N, K].
  TensorDim quant_dim = logical_weight.getDim(); // [N, K]
  quant_dim.setTensorType({logical_weight.getFormat(), Tdatatype::QINT4_HTP});
  quant_dim.height(K); // -> [K, N]
  quant_dim.width(N);
  Tensor output(quant_dim, true, Initializer::NONE, "",
                QScheme::PER_CHANNEL_AFFINE_I4);

  const float *src = logical_weight.getData<float>();
  int8_t *dst = output.getData<int8_t>();
  float *scales = output.getScale<float>();
  int32_t *zp_corr = output.getZpCorr<int32_t>();

  // Symmetric INT4: quantize to [-7, 7]. Activation zero-point stays 128
  // (matching the u8 activation path), so zp_corr[n] = 128 * sum_k(W_i4[n,k]).
  for (unsigned int n = 0; n < N; ++n) {
    float max_abs = 0.0f;
    for (unsigned int k = 0; k < K; ++k) {
      const float value = src[n * K + k];
      NNTR_THROW_IF(!std::isfinite(value), std::runtime_error)
        << "[quantize_qint4_weight] Encountered non-finite weight value.";
      max_abs = std::max(max_abs, std::fabs(value));
    }

    float scale = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;
    if (!std::isfinite(scale) || scale <= 0.0f)
      scale = 1.0f;

    scales[n] = scale;

    int32_t row_sum = 0;
    for (unsigned int k = 0; k < K; ++k) {
      const float value = src[n * K + k] / scale;
      const long q = std::lround(value);
      const int8_t clamped =
        static_cast<int8_t>(std::max<long>(-7, std::min<long>(7, q)));
      dst[n * K + k] = clamped;
      row_sum += static_cast<int32_t>(clamped);
    }
    zp_corr[n] = 128 * row_sum;
  }

#ifdef ENABLE_HEXKL
  // sdkl_cpu_rm_to_wh_i4 (sdkl.h) converts a row-major i4 weight — one i4
  // value per int8 byte, sign-extended, values in [-8, 7] — into packed WH
  // layout (uint8, two nibbles per byte, HMX-tiled). It is NOT in-place:
  //   int sdkl_cpu_rm_to_wh_i4(uint8_t *full_wt_tiled, int8_t *wt_old,
  //                            size_t wt_rows, size_t wt_cols);
  // `dst` already satisfies the input contract (int4 in [-7, 7]). The packed
  // result is copied back into `dst`; shgemm_u8i4_i32 reads its first N*K/2
  // bytes.
  const size_t unpacked_bytes =
    static_cast<size_t>(N) * static_cast<size_t>(K) * sizeof(int8_t);
  const size_t packed_bytes = unpacked_bytes / 2;
  void *wh_out = nullptr;
  bool used_npu = false;

  // Prefer an NPU-accessible output buffer; device tests showed host-only
  // buffers can fault in the SDKL layout path. Over-allocate to the unpacked
  // size so tile padding beyond N*K/2 cannot overflow.
  if (nntrainer::HtpBackend::global().enabled() &&
      sdkl_npu_alloc(unpacked_bytes, &wh_out) == 0 && wh_out != nullptr) {
    used_npu = true;
  }

  nntrainer::MemAllocator host_alloc;
  if (!used_npu)
    host_alloc.alloc(&wh_out, unpacked_bytes, 64);

  // Device-verified: sdkl_cpu_rm_to_wh_i4 takes (wt_rows=K, wt_cols=N) — the
  // inner dimension first — to produce a WH weight that sdkl_npu_mm_u8i4_i32
  // consumes as X * W^T for this [N, K] weight. (The in-place i8 variant uses
  // (N, K); the i4 packer's convention is the transpose of that.)
  int rc = sdkl_cpu_rm_to_wh_i4(static_cast<uint8_t *>(wh_out), dst,
                                static_cast<size_t>(K), static_cast<size_t>(N));
  if (rc == 0)
    std::memcpy(dst, wh_out, packed_bytes);

  if (used_npu) {
    const int free_rc = sdkl_npu_free(wh_out);
    NNTR_THROW_IF(free_rc != 0, std::runtime_error)
      << "[quantize_qint4_weight] Failed to free temporary SDKL buffer.";
  } else {
    host_alloc.free(wh_out);
  }

  NNTR_THROW_IF(rc != 0, std::runtime_error)
    << "[quantize_qint4_weight] Failed to convert QINT4 weight to WH layout.";
#endif

  return output;
}

} // namespace nntrainer
