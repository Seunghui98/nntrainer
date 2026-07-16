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
#include <cstdlib>
#include <cstring>
#include <ggml_interface.h>
#include <mutex>
#include <nntr_ggml_impl.h>
#include <nntr_ggml_impl_utils.h>
#include <thread_manager.h>
#include <unordered_map>
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
 * @brief De-interleave + gather + plain-SMMLA W8A32 fallback (NNTR_Q8A32_PLAIN).
 *
 * The original W8A32 pipeline: de-interleaves the pre-repacked q8_0x4 weight
 * back to plain block_q8_0 (cached per weight pointer), quantizes each gathered
 * FP32 activation row to plain block_q8_0, and feeds nntr_gemm_q8_0_q8_0_f32.
 * Kept as a numerical reference / fallback under NNTR_Q8A32_PLAIN. The default
 * path below is ~2x faster (interleaved 4-row quantize + 4x4 SMMLA, no
 * de-interleave), so this stays behind the env only.
 */
static void __q8a32_indirect_plain(const unsigned int M, const unsigned int N,
                                   const unsigned int K, const float *in,
                                   const ConvGatherParams &geom, const void *B,
                                   float *C) {
  auto &tm = ThreadManager::Global();
  const unsigned int nb = K / QK8_0;

  // 1) De-interleave weight q8_0x4 -> plain block_q8_0 [N][nb], cached per
  //    weight-buffer pointer so it runs once instead of every forward.
  const block_q8_0 *Wp_ptr;
  {
    static std::mutex wcache_mtx;
    static std::unordered_map<const void *, std::vector<block_q8_0>> wcache;
    std::lock_guard<std::mutex> lk(wcache_mtx);
    auto it = wcache.find(B);
    if (it == wcache.end()) {
      std::vector<block_q8_0> Wp((size_t)N * nb);
      const block_q8_0x4 *bx4 = (const block_q8_0x4 *)B;
      const unsigned int NB4 = N / 4;
      for (unsigned int sc = 0; sc < NB4; ++sc)
        for (unsigned int j = 0; j < nb; ++j) {
          const block_q8_0x4 &sb = bx4[(size_t)sc * nb + j];
          for (unsigned int r = 0; r < 4; ++r) {
            block_q8_0 &p = Wp[(size_t)(sc * 4 + r) * nb + j];
            p.d = sb.d[r];
            for (unsigned int sub = 0; sub < 4; ++sub)
              std::memcpy(&p.qs[8 * sub], &sb.qs[32 * sub + 8 * r], 8);
          }
        }
      it = wcache.emplace(B, std::move(Wp)).first;
    }
    Wp_ptr = it->second.data();
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

  // 3) Plain Q8_0 x Q8_0 GEMM -> FP32 C, tiled 16x16.
  const unsigned int row_chunk = 16, col_chunk = 16;
  const size_t row_loop = (M + row_chunk - 1) / row_chunk;
  const size_t col_loop = (N + col_chunk - 1) / col_chunk;
  const block_q8_0 *Ap_ptr = Ap.data();
  tm.parallel_for(0, row_loop * col_loop, [=](size_t i) {
    unsigned int r = (unsigned int)(i / col_loop);
    unsigned int c = (unsigned int)(i % col_loop);
    unsigned int r0 = r * row_chunk, r1 = std::min(r0 + row_chunk, M);
    unsigned int c0 = c * col_chunk, c1 = std::min(c0 + col_chunk, N);
    nntr_gemm_q8_0_q8_0_f32((int)K, C + (size_t)r0 * N + c0, N,
                            (const void *)&Wp_ptr[(size_t)c0 * nb],
                            (const void *)&Ap_ptr[(size_t)r0 * nb],
                            (int)(r1 - r0), (int)(c1 - c0));
  });
}

/**
 * @brief FP32-activation Q8_0-weight indirect conv GEMM (W8A32).
 *
 * The W8A16 (FP16-activation) path stores activations as FP16 between layers,
 * which accumulates rounding error across the deep backbone and collapses pose
 * confidence. This FP32 variant keeps activations in FP32 end-to-end (matching
 * an ONNX-Runtime int8 model's accuracy) while still doing int8 SMMLA compute.
 *
 * The default path mirrors the fast W8A16 pipeline exactly, only in FP32:
 * the FP32 input is gathered 4 rows at a time (no im2col materialization) and
 * quantized straight into the interleaved block_q8_0x4 layout via
 * nntr_quantize_mat_q8_0_4x8, so the pre-repacked q8_0x4 weight can be consumed
 * *directly* (no per-forward de-interleave) by the register-blocked 4x4 SMMLA
 * kernel nntr_gemm_q8_0_q8_0_4x4_f32, writing FP32 output. This removes the two
 * costs that made the earlier plain path 10-28x slower than W8A16: the 1-row
 * scalar quantize and the weight de-interleave. It reuses the identical weight
 * file as the FP16 path (no re-quantize) and is portable across ISAs (all
 * primitives have neon/avx/sve/fallback impls). Set NNTR_Q8A32_PLAIN to fall
 * back to the reference de-interleave pipeline.
 */
void __ggml_q8_0_q8_0_indirect_GEMM_fp32(const unsigned int M,
                                         const unsigned int N,
                                         const unsigned int K, const float *in,
                                         const ConvGatherParams &geom,
                                         const void *B, const unsigned int ldb,
                                         float *C, const unsigned int ldc) {
  (void)ldb;
  (void)ldc;

  static const bool use_plain = (std::getenv("NNTR_Q8A32_PLAIN") != nullptr);
  if (use_plain) {
    __q8a32_indirect_plain(M, N, K, in, geom, B, C);
    return;
  }

  auto &tm = ThreadManager::Global();
  const unsigned int nb = K / QK8_0; // blocks per row (K multiple of 32)
  const unsigned int qa_4_rows_size = sizeof(block_q8_0x4) * nb;
  const unsigned int Mfull = (M / 4) * 4; // 4-row-divisible part
  const unsigned int rem = M % 4;
  const unsigned int M4 = M / 4;
  const unsigned int M4c = (M + 3) / 4; // groups incl. padded tail

  // 1) Fused gather + Q8_0 quantize to interleaved q8_0x4, 4 rows at a time.
  //    Parallelized over 64-row chunks (16 tiles) to amortize dispatch, with a
  //    reusable per-thread tile buffer.
  std::vector<char> QA((size_t)M4c * qa_4_rows_size);
  char *QA_ptr = QA.data();

  const unsigned int QCHUNK = 64; // multiple of 4
  if (Mfull > 0) {
    const size_t qloops = (Mfull + QCHUNK - 1) / QCHUNK;
    tm.parallel_for(0, qloops, [=](size_t q) {
      const unsigned int r0 = static_cast<unsigned int>(q) * QCHUNK;
      const unsigned int r1 = std::min(r0 + QCHUNK, Mfull);
      std::vector<float> tilebuf((size_t)4 * K);
      float *tile = tilebuf.data();
      for (unsigned int r = r0; r < r1; r += 4) {
        gather_conv_act_rows_fp32(tile, in, geom, (int)r, 4);
        nntr_quantize_mat_q8_0_4x8(
          tile, QA_ptr + (size_t)(r / 4) * qa_4_rows_size, K);
      }
    });
  }

  // Handle M-tail (rem 1..3) gather and quantization (zero-padded to 4 rows).
  if (rem > 0) {
    std::vector<float> tilebuf((size_t)4 * K, 0.0f);
    gather_conv_act_rows_fp32(tilebuf.data(), in, geom, (int)Mfull, (int)rem);
    nntr_quantize_mat_q8_0_4x8(tilebuf.data(),
                               QA_ptr + (size_t)M4 * qa_4_rows_size, K);
  }

  // 2) Tiled 4x4 SMMLA GEMM over the 4-row-divisible part, direct to C.
  //    Weight B is consumed in its native q8_0x4 layout (stride = nb plain
  //    blocks per 4-column super-block).
  const size_t B_step = (size_t)nb * sizeof(block_q8_0);
  const unsigned int row_chunk_size = 16; // multiple of 4
  const unsigned int col_chunk_size = 16; // multiple of 4

  if (Mfull > 0) {
    const size_t row_loop = (Mfull + row_chunk_size - 1) / row_chunk_size;
    const size_t col_loop = (N + col_chunk_size - 1) / col_chunk_size;
    tm.parallel_for(0, row_loop * col_loop, [=](size_t i) {
      unsigned int r = static_cast<unsigned int>(i / col_loop);
      unsigned int c = static_cast<unsigned int>(i % col_loop);
      unsigned int r_start = r * row_chunk_size;
      unsigned int r_end = std::min(row_chunk_size * (r + 1), Mfull);
      unsigned int c_start = c * col_chunk_size;
      unsigned int c_end = std::min(col_chunk_size * (c + 1), N);

      nntr_gemm_q8_0_q8_0_4x4_f32(
        (int)K, C + (size_t)r_start * N + c_start, N,
        (const void *)((const char *)B + (size_t)c_start * B_step),
        (const void *)(QA_ptr + (size_t)(r_start / 4) * qa_4_rows_size),
        (int)(r_end - r_start), (int)(c_end - c_start));
    });
  }

  // 3) M-tail (rem 1..3): run the padded last group into a 4xN FP32 scratch,
  //    then copy only the valid rem rows into C.
  if (rem > 0) {
    std::vector<float> scratch((size_t)4 * N);
    float *scratch_ptr = scratch.data();
    const char *tail_a = QA_ptr + (size_t)M4 * qa_4_rows_size;
    const unsigned int col_loop = (N + col_chunk_size - 1) / col_chunk_size;
    tm.parallel_for(0, col_loop, [=](size_t c) {
      unsigned int c_start = static_cast<unsigned int>(c) * col_chunk_size;
      unsigned int c_end = std::min(col_chunk_size * ((unsigned int)c + 1), N);
      nntr_gemm_q8_0_q8_0_4x4_f32(
        (int)K, scratch_ptr + c_start, N,
        (const void *)((const char *)B + (size_t)c_start * B_step),
        (const void *)tail_a, 4, (int)(c_end - c_start));
    });
    for (unsigned int rr = 0; rr < rem; ++rr)
      std::memcpy(C + (size_t)(Mfull + rr) * N, scratch_ptr + (size_t)rr * N,
                  (size_t)N * sizeof(float));
  }
}

} // namespace nntrainer
