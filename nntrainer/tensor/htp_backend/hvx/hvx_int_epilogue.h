// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_int_epilogue.h
 * @date   28 Sep 2026
 * @brief  The MoE gate_up epilogue with no f32 in it: int32 accumulator ->
 *         fixed-point dequant -> integer SwiGLU -> per-row u8, scales baked
 *         into 16-bit multipliers and shifts (doc 53 section 9)
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * WHAT THIS IS
 *
 * hvx_dequant_i32.c turns an HMX int32 accumulator into f32, applies the
 * f32 SwiGLU (hvx_swiglu_det.h) and hvx_quant_u8.c scans and requantizes
 * the f32 result to u8 for the down matmul. This file is the same
 * pipeline with every per-element operation in integer arithmetic, the
 * way arXiv 2511.11248 (T-MAN) bakes quantization scales into its tables:
 * per-column and per-row scales become a 15-bit mantissa and a shift, so
 * the element work is multiply-high, shift, add, min, max. The result is a
 * DIFFERENT u8 grid from the f32 path (different rounding points, a
 * polynomial sigmoid), gated by perplexity (doc 53 section 3), not by bit
 * identity with the ARM path.
 *
 * Every operation here is one of a small set with an exact definition
 * that an HVX instruction implements verbatim, so this portable C is the
 * bit-exact reference for a vectorised version and runs as-is on the DSP
 * meanwhile:
 *
 *   mulhi_q15(a, b)  = sat32((a * b * 2 + 2^15) >> 16), b a signed 16-bit
 *                      value           -- Q6_Vw_vmpyo_VwVh_s1_rnd_sat
 *   asr_rnd(x, k)    = (x + 2^(k-1)) >> k, k in [1, 30]  -- vadd + vasr
 *   shifts by a per-lane amount are clamped to [0, 30] first -- vmin +
 *                      vasr/vasl (the hardware masks the amount)
 *   nbits(v)         = 32 - clz(v)
 *
 * NUMBER FORMATS
 *
 * A value is (mantissa, exponent) with value = mantissa * 2^exponent for
 * the baked constants, and Q(F) fixed point (value = q * 2^-F) inside the
 * epilogue, where F is chosen per (row, staged batch) from the data: the
 * largest |acc - zp*colsum| over the batch's tiles and the batch's largest
 * column scale bound the result, and F puts that bound at 2^30. That
 * choice, not a static one, is what keeps 20+ significant bits through the
 * chain; a bound from the static maximum of the accumulator is loose by
 * 2^7-2^11 here and would leave the product g*u with 9 bits. The stored
 * SwiGLU result therefore carries one exponent per (row, batch); the
 * requantization brings a row's batches to a common exponent first.
 *
 * WHAT STAYS FLOAT
 *
 * The bake (once per registered weight, frexpf on w_scale and bias) and
 * the per-row parameter the down matmul's f32 dequant needs (one ldexpf
 * per row). Nothing per element.
 */

#ifndef __NNTRAINER_HVX_INT_EPILOGUE_H__
#define __NNTRAINER_HVX_INT_EPILOGUE_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief One weight's columns as fixed-point constants, baked once from
 *        its f32 w_scale and bias. 3 bytes a column, 8 with a bias.
 */
typedef struct {
  int16_t *wm;  /**< N: mantissa in [2^14, 2^15), 0 for w_scale <= 0 */
  int8_t *we;   /**< N: w_scale = wm * 2^we */
  int32_t *bm;  /**< N, or NULL when the bias is all zero: |bm| in
                     [2^29, 2^30), bias = bm * 2^be */
  int8_t *be;   /**< N or NULL */
  int8_t *t_wb; /**< N/32: every w_scale of the n-tile is < 2^t_wb */
  int8_t *t_bb; /**< N/32 or NULL: every |bias| of the n-tile is < 2^t_bb */
  uint32_t N;
} hvx_int_wq;

/** @brief Allocates and fills *out. N is a multiple of 32.
 *  @return AEE_SUCCESS, AEE_EBADPARM, or AEE_ENOMEMORY */
int hvx_int_wq_bake(hvx_int_wq **out, const float *w_scale, const float *bias,
                    uint32_t N);
/** @brief Frees what hvx_int_wq_bake allocated. Safe on NULL. */
void hvx_int_wq_free(hvx_int_wq *q);

/**
 * @brief The gate_up epilogue over one staged batch of tile PAIRS, as a
 *        pool job split by ROW (each row's exponents come from its own
 *        scan over the batch, so a thread owns whole rows).
 *
 * Staged slot j is gate n-tile g0+j, slot n_pairs+j the up tile opposite
 * it (inter columns further), the layout hvx_dq_swiglu_job takes. Tiles
 * hold RAW accumulators; the zero-point correction happens here.
 *
 * Output: dst[r][c] = round(silu(g)*u * 2^F) for the batch's columns c,
 * with F = h_e[r * e_stride + batch]. Rows at or past m_count are not
 * touched.
 */
typedef struct {
  const uint8_t *tiles_base;
  uint32_t tile_stride;
  uint32_t n_pairs;
  uint32_t g0;
  uint32_t row_stride; /**< int32 elements between tile rows */
  uint32_t m_count;
  const float *act_scale; /**< the block's rows */
  const int32_t *act_zp;
  const int32_t *colsum_w; /**< the full gate_up table, 2 * inter */
  const hvx_int_wq *wq;    /**< the gate_up weight's constants */
  uint32_t inter;
  int32_t *dst; /**< [rows x inter] mantissas, row stride dst_stride */
  uint32_t dst_stride;
  int16_t *h_e; /**< [rows x e_stride] exponents F per (row, batch) */
  uint32_t e_stride;
  uint32_t batch; /**< this run's index within the block's batches */
} hvx_int_gu_job;

/**
 * @brief hvx_worker_pool_func over an hvx_int_gu_job, in two
 *        implementations with identical output: the portable C (_c, the
 *        reference, runs anywhere) and HVX intrinsics (_hvx, DSP builds
 *        only). Callers use hvx_int_gu_worker, which is the HVX one on
 *        the DSP unless HEXKL_INT_EPILOGUE_SCALAR forces the reference
 *        (an A/B, or bisecting a mismatch the self-check reports).
 */
void hvx_int_gu_worker_c(uint32_t n_threads, uint32_t i, void *job);
#if defined(__hexagon__)
void hvx_int_gu_worker_hvx(uint32_t n_threads, uint32_t i, void *job);
#endif
#if defined(__hexagon__) && !defined(HEXKL_INT_EPILOGUE_SCALAR)
#define hvx_int_gu_worker hvx_int_gu_worker_hvx
#define hvx_int_rq_rows hvx_int_rq_rows_hvx
#else
#define hvx_int_gu_worker hvx_int_gu_worker_c
#define hvx_int_rq_rows hvx_int_rq_rows_c
#endif

/**
 * @brief Requantizes rows [m0, m1) of the SwiGLU result to u8 AH tiles,
 *        with the per-row f32 scale and integer zero point the down
 *        matmul's dequant consumes (hvx_quant_u8.h's convention: x =
 *        scale * (u - zp), zp = round(-min / scale), the range widened to
 *        include 0).
 *
 * h holds the batches' mantissas side by side (batch b at columns
 * [b * batch_cols, ...), the last one shorter), h_e their exponents. Rows
 * at or past m_valid are padding: their bytes are zeroed and their
 * parameters set to scale 1, zp 0, as the f32 path leaves them.
 *
 * @param out_ah the destination block's first AH tile (64-row blocks,
 *               2048-byte k-tiles, 32 bytes a row -- hvx_quant_u8.h)
 */
void hvx_int_rq_rows_c(const int32_t *h, uint32_t h_stride, const int16_t *h_e,
                       uint32_t e_stride, uint32_t n_batches,
                       uint32_t batch_cols, uint32_t inter, uint32_t m_valid,
                       uint32_t m0, uint32_t m1, float *scale, int32_t *zp,
                       uint8_t *out_ah);
#if defined(__hexagon__)
/** The HVX version. Rows are done four at a time (one 128-byte store per
 *  k-tile); a group of fewer than four rows at the end of [m0, m1) goes
 *  through the reference. m0 is a multiple of 4. */
void hvx_int_rq_rows_hvx(const int32_t *h, uint32_t h_stride,
                         const int16_t *h_e, uint32_t e_stride,
                         uint32_t n_batches, uint32_t batch_cols,
                         uint32_t inter, uint32_t m_valid, uint32_t m0,
                         uint32_t m1, float *scale, int32_t *zp,
                         uint8_t *out_ah);
#endif

/**
 * @brief sigmoid(x) in Q15 for x = g * 2^-F, |x| clamped to 16 -- the
 *        integer SwiGLU's nonlinearity, exposed for the host check.
 *        exp2 and 1/(1+e) are Q30 polynomials with 15-bit arguments;
 *        maximum error against the real function is about 5e-5.
 */
int32_t hvx_int_sigmoid_q15(int32_t g_q, int F);

/** @brief mulhi_q15 as defined in the file comment, for the host check. */
int32_t hvx_int_mulhi_q15(int32_t a, int32_t b16);

#ifdef __cplusplus
}
#endif

#endif /* __NNTRAINER_HVX_INT_EPILOGUE_H__ */
