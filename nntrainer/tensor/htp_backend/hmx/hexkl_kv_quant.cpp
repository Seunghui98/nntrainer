// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_kv_quant.cpp
 * @date   06 Aug 2026
 * @brief  Per-KV-block symmetric quantizer, parametric over i4 and i8. HOST.
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs.
 *
 * See hexkl_kv_quant.h for the design contract. Width is runtime; ONE code
 * path. fp16 decode reuses nntrainer::compute_fp16_to_fp32 (host-safe, already
 * in tree -- no reason to reinvent it).
 */

#include "hexkl_kv_quant.h"

#include <cmath>
#include <cstring>

#include "fp16.h" /* nntrainer::compute_fp16_to_fp32 */

/* Symmetric per-N quant params for a given width. denom = positive max of the
 * symmetric range (7 for i4, 127 for i8); the negative bound is -(denom+1) for
 * i4 and -denom for i8 (MHA_HTP_U8_TASKS.md §1.5). */
typedef struct {
  int lo;      /* clamp lower bound (inclusive) */
  int hi;      /* clamp upper bound (inclusive) */
  float denom; /* scale denominator == positive max */
} kvq_params;

static inline kvq_params kvq_params_for(hexkl_w_width w) {
  if (w == HEXKL_W_I4) {
    kvq_params p = {-8, 7, 7.0f};
    return p;
  }
  kvq_params p = {-127, 127, 127.0f};
  return p;
}

/* round to nearest, ties away from zero. std::lround does exactly that and
 * returns long; clamp into int range explicitly. */
static inline int kvq_round_to_int(float x) {
  long r = lroundf(x); /* half away from zero */
  if (r < -2147483647L) {
    return -2147483647;
  }
  if (r > 2147483647L) {
    return 2147483647;
  }
  return (int)r;
}

static inline int kvq_clamp(int q, int lo, int hi) {
  if (q < lo) {
    return lo;
  }
  if (q > hi) {
    return hi;
  }
  return q;
}

void hexkl_kvq_pack_kt_block(const uint16_t *k_rows_f16, uint32_t n_rows_valid,
                             uint32_t T, uint32_t head_dim, uint32_t nch,
                             uint32_t head, hexkl_w_width w, int8_t *out_rm,
                             float *out_scale, int32_t *out_colsum) {
  kvq_params p = kvq_params_for(w);

  /* out_rm is [head_dim][T] row-major: row d, col r at d*T + r.
   * Per-N scale: N = kv position r. amax over the head_dim values of that
   * position (one per d). */
  for (uint32_t r = 0; r < T; ++r) {
    out_scale[r] = 1.0f;
    out_colsum[r] = 0;
  }
  /* Zero every element first; the tail overwrite is then free. */
  memset(out_rm, 0, (size_t)head_dim * T);

  if (n_rows_valid > T) {
    n_rows_valid = T;
  }

  for (uint32_t r = 0; r < n_rows_valid; ++r) {
    /* head h's fp16 row r: head_dim contiguous values at (r*nch +
     * head)*head_dim. */
    const uint16_t *row_f16 = k_rows_f16 + ((size_t)r * nch + head) * head_dim;
    float amax = 0.0f;
    for (uint32_t d = 0; d < head_dim; ++d) {
      float v = nntrainer::compute_fp16_to_fp32(row_f16[d]);
      float a = fabsf(v);
      if (a > amax) {
        amax = a;
      }
    }
    float scale = (amax > 0.0f) ? (amax / p.denom) : 1.0f;
    out_scale[r] = scale;
    int32_t colsum = 0;
    float inv_scale = 1.0f / scale;
    for (uint32_t d = 0; d < head_dim; ++d) {
      float v = nntrainer::compute_fp16_to_fp32(row_f16[d]);
      int q = kvq_clamp(kvq_round_to_int(v * inv_scale), p.lo, p.hi);
      out_rm[(size_t)d * T + r] = (int8_t)q;
      colsum += q;
    }
    out_colsum[r] = colsum;
  }
  /* Tail r in [n_rows_valid, T): out_rm already zeroed, scale 1.0f, colsum 0.
   */
}

void hexkl_kvq_pack_v_block(const uint16_t *v_rows_f16, uint32_t n_rows_valid,
                            uint32_t T, uint32_t head_dim, uint32_t nch,
                            uint32_t head, hexkl_w_width w, int8_t *out_rm,
                            float *out_scale, int32_t *out_colsum) {
  kvq_params p = kvq_params_for(w);

  /* out_rm is [T][head_dim] row-major: row t, col d at t*head_dim + d.
   * Per-N scale: N = head_dim column d. amax over the n_rows_valid values of
   * that column (one per t in [0, n_rows_valid)). */
  for (uint32_t d = 0; d < head_dim; ++d) {
    out_scale[d] = 1.0f;
    out_colsum[d] = 0;
  }
  memset(out_rm, 0, (size_t)T * head_dim);

  if (n_rows_valid > T) {
    n_rows_valid = T;
  }

  /* Decode fp16 inline twice (amax + quant); host code, simplicity over
   * avoiding re-decode. */
  for (uint32_t d = 0; d < head_dim; ++d) {
    float amax = 0.0f;
    for (uint32_t t = 0; t < n_rows_valid; ++t) {
      const uint16_t *row_f16 =
        v_rows_f16 + ((size_t)t * nch + head) * head_dim;
      float v = nntrainer::compute_fp16_to_fp32(row_f16[d]);
      float a = fabsf(v);
      if (a > amax) {
        amax = a;
      }
    }
    float scale = (amax > 0.0f) ? (amax / p.denom) : 1.0f;
    out_scale[d] = scale;
    int32_t colsum = 0;
    float inv_scale = 1.0f / scale;
    for (uint32_t t = 0; t < n_rows_valid; ++t) {
      const uint16_t *row_f16 =
        v_rows_f16 + ((size_t)t * nch + head) * head_dim;
      float v = nntrainer::compute_fp16_to_fp32(row_f16[d]);
      int q = kvq_clamp(kvq_round_to_int(v * inv_scale), p.lo, p.hi);
      out_rm[(size_t)t * head_dim + d] = (int8_t)q;
      colsum += q;
    }
    out_colsum[d] = colsum;
  }
  /* Tail rows in [n_rows_valid, T): out_rm already zeroed, scale 1.0f, colsum
   * 0. */
}
