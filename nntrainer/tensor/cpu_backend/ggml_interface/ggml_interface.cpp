// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   ggml_interface.cpp
 * @date   13 August 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Function interface to use ggml lib from cpu_backend
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <ggml_interface.h>
#include <nntr_ggml_impl.h>
#include <nntr_ggml_impl_utils.h>
#include <thread_manager.h>
#ifdef Q4_0
#undef Q4_0
#endif
#ifdef Q8_0
#undef Q8_0
#endif
#include <q8_0_tensor.h>
#include <string>
#include <thread>
#include <vector>

namespace nntrainer {

void __ggml_init() { nntr_ggml_init(); }

size_t __ggml_quantize_q4_0(const float *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights) {
  return nntr_quantize_q4_0(src, dst, nrow, n_per_row, quant_weights);
}

size_t __ggml_quantize_q4_K(const float *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights) {
  return nntr_quantize_q4_K(src, dst, nrow, n_per_row, quant_weights);
}

size_t __ggml_quantize_q6_K(const float *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights) {
  return nntr_quantize_q6_K(src, dst, nrow, n_per_row, quant_weights);
}

size_t __ggml_quantize_q8_0(const float *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights) {
  return nntr_quantize_q8_0(src, dst, nrow, n_per_row, quant_weights);
}

void __ggml_quantize_row_q6_K(const float *src, void *dst, int64_t k) {
  __ggml_quantize_q6_K(src, dst, 1, k, nullptr);
}

template <>
void __ggml_quantize_row_q8_K(const float *src, void *dst, int64_t k) {
  nntr_quantize_row_q8_K(src, dst, k);
}

void __ggml_dequantize_row_q4_0(const void *x_raw, float *y, int64_t k) {
  nntr_dequantize_row_q4_0(x_raw, y, k);
}

void __ggml_dequantize_row_q4_K(const void *x_raw, float *y, int64_t k) {
  nntr_dequantize_row_q4_K(x_raw, y, k);
}

void __ggml_dequantize_row_q8_0(const void *x_raw, float *y, int64_t k) {
  nntr_dequantize_row_q8_0(x_raw, y, k);
}

void __ggml_dequantize_row_q6_K(const void *x, float *y, int64_t k) {
  nntr_dequantize_row_q6_K(x, y, k);
}

template <>
void __ggml_dequantize_row_q8_K(const void *x, float *y, int64_t k) {
  nntr_dequantize_row_q8_K(x, y, k);
}

float __ggml_vec_dot_q6_K_q8_K(const unsigned int K,
                               const void *__restrict v_q6_K,
                               const void *__restrict v_q8_K) {
  float result;
  int bs = 1, bx = 1, by = 1,
      nrc = 1; // unused variables in ggml_vec_dot_q6_K_q8_K
  nntr_vec_dot_q6_K_q8_K(K, &result, bs, v_q6_K, bx, v_q8_K, by, nrc);
  return result;
}

float __ggml_vec_dot_q6_K_f32(const unsigned int K, const void *v_q6_K,
                              const float *f) {
  // Quantization of activations
  int blocks_per_row = (K + QK_K - 1) / QK_K;
  int q8_K_activation_size = sizeof(block_q8_K) * blocks_per_row;
  std::vector<char> v_q8_activation = std::vector<char>(q8_K_activation_size);
  __ggml_quantize_row_q8_K(f, v_q8_activation.data(), K);

  return __ggml_vec_dot_q6_K_q8_K(K, v_q6_K, v_q8_activation.data());
}

float __ggml_vec_dot_q6_K(const unsigned int K, const void *__restrict v_q6_K,
                          const float *__restrict activation) {
  float result;
  int bs = 1, bx = 1, by = 1,
      nrc = 1; // unused variables in ggml_vec_dot_q6_K_q8_K

  int blocks_per_row = (K + QK_K - 1) / QK_K;
  int q8_K_activation_size = sizeof(block_q8_K) * blocks_per_row;
  std::vector<char> v_q8_activation = std::vector<char>(q8_K_activation_size);
  __ggml_quantize_row_q8_K(activation, v_q8_activation.data(), K);

  nntr_vec_dot_q6_K_q8_K(K, &result, bs, v_q6_K, bx, v_q8_activation.data(), by,
                         nrc);
  return result;
}

void __ggml_repack_q4_0_to_q4_0_4(void *dst, void *src, size_t data_size,
                                  const unsigned int M, const unsigned int N) {
  nntr_repack_q4_0_to_q4_0_4_bl(dst, 8, src, data_size, M, N);
}

void __ggml_repack_q4_0_to_q4_0_8(void *dst, void *src, size_t data_size,
                                  const unsigned int M, const unsigned int N) {
  nntr_repack_q4_0_to_q4_0_8_bl(dst, 8, src, data_size, M, N);
}

void __ggml_repack_q4_K_to_q4_K_8(void *dst, void *src, size_t data_size,
                                  const unsigned int M, const unsigned int N) {
  nntr_repack_q4_K_to_q4_K_8_bl(dst, 8, src, data_size, M, N);
}

/**
 * @brief Dispatch Q8_0 x Q8_0 GEMM through the common ggml interface
 */
void __ggml_q8_0_q8_0_GEMM(const unsigned int M, const unsigned int N,
                           const unsigned int K, const float *A,
                           const unsigned int lda, const void *B,
                           const unsigned int ldb, float *C,
                           const unsigned int ldc) {
  (void)lda;
  (void)ldb;

  // Online-quantise A row-by-row to Q8_0 in a scratch buffer the SIMD
  // micro-kernel reads back. nntr_quantize_row_q8_0 produces the exact
  // block_q8_0 layout (fp16 scale + 32 int8 quants per 32-element block).
  const unsigned int nb_per_row = K / QK8_0;
  const size_t qa_row_bytes = sizeof(block_q8_0) * nb_per_row;

  std::vector<char> QA(static_cast<size_t>(M) * qa_row_bytes);
  for (unsigned int m = 0; m < M; ++m) {
    nntr_quantize_row_q8_0(A + static_cast<size_t>(m) * K,
                           QA.data() + static_cast<size_t>(m) * qa_row_bytes,
                           static_cast<int64_t>(K));
  }

  // One unified inner kernel for now; a GEMV specialisation + Q8_0x8
  // interleaved weight layout to match the Q4_0 8x8 micro-tile is a
  // follow-up.
  nntr_gemm_q8_0_q8_0(static_cast<int>(K), C, ldc, B, QA.data(),
                      static_cast<int>(M), static_cast<int>(N));
}

/**
 * @brief FP32-activation Q8_0-weight indirect conv GEMM (W8A32).
 *
 * The W8A16 (FP16-activation) path stores activations as FP16 between layers,
 * which accumulates rounding error across the deep backbone and collapses pose
 * confidence. This FP32 variant keeps activations in FP32 end-to-end (matching
 * an ONNX-Runtime int8 model's accuracy) while still doing int8 SMMLA compute:
 * the FP32 input is gathered on the fly (no im2col materialization) and
 * quantized per row to plain block_q8_0, the pre-repacked q8_0x4 weight is
 * de-interleaved back to plain block_q8_0, and both feed nntr_gemm_q8_0_q8_0
 * (the same int8 core as the FP32-output GEMM), writing FP32 output. It reuses
 * the identical weight file as the FP16 path (no re-quantize). Portable across
 * ISAs (all primitives have neon/avx/sve/fallback impls).
 */
void __ggml_q8_0_q8_0_indirect_GEMM_fp32(const unsigned int M,
                                         const unsigned int N,
                                         const unsigned int K, const float *in,
                                         const ConvGatherParams &geom,
                                         const void *B, const unsigned int ldb,
                                         float *C, const unsigned int ldc) {
  auto &tm = ThreadManager::Global();
  (void)ldb;
  (void)ldc;
  const unsigned int nb = K / QK8_0;

  // 1) De-interleave weight q8_0x4 -> plain block_q8_0 [N][nb] (inverse of the
  //    repack_q8_0 done at weight-export time). N % 4 == 0 is guaranteed by the
  //    Q8_0 conv eligibility guard (out_ch % 32 == 0).
  std::vector<block_q8_0> Wp((size_t)N * nb);
  {
    const block_q8_0x4 *bx4 = (const block_q8_0x4 *)B;
    const unsigned int NB4 = N / 4;
    block_q8_0 *Wp_ptr = Wp.data();
    tm.parallel_for(0, NB4, [=](size_t sc) {
      for (unsigned int j = 0; j < nb; ++j) {
        const block_q8_0x4 &sb = bx4[(size_t)sc * nb + j];
        for (unsigned int r = 0; r < 4; ++r) {
          block_q8_0 &p = Wp_ptr[(size_t)((unsigned)sc * 4 + r) * nb + j];
          p.d = sb.d[r];
          for (unsigned int sub = 0; sub < 4; ++sub)
            std::memcpy(&p.qs[8 * sub], &sb.qs[32 * sub + 8 * r], 8);
        }
      }
    });
  }

  // 2) Gather FP32 activation rows -> plain block_q8_0 [M][nb].
  std::vector<block_q8_0> Ap((size_t)M * nb);
  {
    block_q8_0 *Ap_ptr = Ap.data();
    const unsigned int QCHUNK = 64;
    const size_t qloops = (M + QCHUNK - 1) / QCHUNK;
    tm.parallel_for(0, qloops, [=](size_t q) {
      std::vector<float> tile((size_t)K);
      const unsigned int r0 = (unsigned int)q * QCHUNK;
      const unsigned int r1 = std::min(r0 + QCHUNK, M);
      for (unsigned int r = r0; r < r1; ++r) {
        gather_conv_act_rows_fp32(tile.data(), in, geom, (int)r, 1);
        nntr_quantize_row_q8_0(tile.data(), &Ap_ptr[(size_t)r * nb],
                               (int64_t)K);
      }
    });
  }

  // 3) Plain Q8_0 x Q8_0 GEMM -> FP32 C. Inline scalar int8 dot (correct by
  //    construction): the shared nntr_gemm_q8_0_q8_0 FP32 primitive is
  //    under-tested (returns 0 on the x86 AVX build), so we don't rely on it
  //    here. Parallelized per output row.
  const block_q8_0 *Wp_ptr = Wp.data();
  const block_q8_0 *Ap_ptr = Ap.data();
  tm.parallel_for(0, M, [=](size_t m) {
    const block_q8_0 *a = Ap_ptr + (size_t)m * nb;
    float *crow = C + (size_t)m * N;
    for (unsigned int j = 0; j < N; ++j) {
      const block_q8_0 *b = Wp_ptr + (size_t)j * nb;
      float acc = 0.0f;
      for (unsigned int bi = 0; bi < nb; ++bi) {
        int32_t isum = 0;
        for (int t = 0; t < QK8_0; ++t)
          isum += (int32_t)a[bi].qs[t] * (int32_t)b[bi].qs[t];
        acc += nntr_compute_fp16_to_fp32(a[bi].d) *
               nntr_compute_fp16_to_fp32(b[bi].d) * (float)isum;
      }
      crow[j] = acc;
    }
  });
}

} // namespace nntrainer
