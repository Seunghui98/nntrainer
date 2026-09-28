// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_int_epilogue_impl.h
 * @date   28 Sep 2026
 * @brief  The scalar pieces of the integer epilogue that the portable C
 *         and the HVX implementation share verbatim (doc 53 section 9.5)
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Internal to hvx_int_epilogue.c and hvx_int_epilogue_hvx.c. Everything
 * here is per-row or per-batch scalar work -- formats, bounds, the
 * requantization's multiplier -- and one definition serves both files so
 * the two cannot drift: the vector file differs from the portable one only
 * in the per-element loops.
 */

#ifndef __NNTRAINER_HVX_INT_EPILOGUE_IMPL_H__
#define __NNTRAINER_HVX_INT_EPILOGUE_IMPL_H__

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "hvx_int_epilogue.h"

/** The sigmoid's Q30 polynomials and constants (hvx_int_epilogue.c). */
extern const int32_t hvx_int_exp2_q30[6];
extern const int32_t hvx_int_rcp_q30[8];
#define IQ_LOG2E_HALF_Q15 23637
#define IQ_X_MAX_Q24 (1 << 28)

static inline int32_t iq_sat32(int64_t v) {
  if (v > INT32_MAX) {
    return INT32_MAX;
  }
  if (v < INT32_MIN) {
    return INT32_MIN;
  }
  return (int32_t)v;
}

/** Q6_Vw_vmpyo_VwVh_s1_rnd_sat: round(a * b16 / 2^15), saturating. */
static inline int32_t iq_mulhi_q15(int32_t a, int32_t b16) {
  const int64_t p = (int64_t)a * (int64_t)b16 * 2 + 0x8000;
  return iq_sat32(p >> 16);
}

/** vadd 2^(k-1), vasr k; k clamped to [0, 30] (0 leaves x alone). */
static inline int32_t iq_asr_rnd(int32_t x, int k) {
  if (k <= 0) {
    return x;
  }
  if (k > 30) {
    k = 30;
  }
  return (int32_t)(((int64_t)x + ((int64_t)1 << (k - 1))) >> k);
}

/** vasr by k, no rounding, k clamped to [0, 30]. */
static inline int32_t iq_asr(int32_t x, int k) {
  if (k <= 0) {
    return x;
  }
  if (k > 30) {
    k = 30;
  }
  return x >> k;
}

/** 32 - clz(v): v < 2^nbits(v). */
static inline int iq_nbits(uint32_t v) {
  int n = 0;
  while (v != 0u) {
    ++n;
    v >>= 1;
  }
  return n;
}

/** x << k as the hardware does it (vasl): defined for negative x too,
 *  which C's << is not. */
static inline int32_t iq_shl(int32_t x, int k) {
  return (int32_t)((uint32_t)x << k);
}

static inline int32_t iq_min(int32_t a, int32_t b) { return a < b ? a : b; }
static inline int32_t iq_max(int32_t a, int32_t b) { return a > b ? a : b; }

/** The IEEE-754 fields of a finite, normal, nonzero float: v = (-1)^s *
 *  (2^23 + mant) * 2^(exp - 150). Denormals and zero return 0 (treated
 *  as zero: no scale or bias here is that small), NaN and inf return 0
 *  too. Integer only: the bake runs on the DSP's scalar core once per
 *  weight, and frexpf + lrintf there cost ~0.5 ms an expert -- 15 ms of
 *  a prefill call, since every layer bakes on its first call. */
static inline int iq_float_fields(float v, uint32_t *mant24, int *exp8,
                                  int *neg) {
  uint32_t b;
  memcpy(&b, &v, 4);
  const int e = (int)((b >> 23) & 0xFFu);
  if (e == 0 || e == 0xFF) {
    return 0;
  }
  *mant24 = (b & 0x7FFFFFu) | 0x800000u;
  *exp8 = e;
  *neg = (b >> 31) != 0u;
  return 1;
}

/** v > 0 -> m in [2^14, 2^15), e with v ~= m * 2^e; else 0, 0.
 *  m = round(1.mant * 2^14), ties to even, exactly what lrintf(frexpf(v)
 *  * 32768) gave: the same bits, so the grid the device measured is
 *  unchanged. */
static inline void iq_split15(float v, int16_t *m, int8_t *e) {
  uint32_t mant24;
  int exp8, neg;
  if (!iq_float_fields(v, &mant24, &exp8, &neg) || neg) {
    *m = 0;
    *e = 0;
    return;
  }
  /* 1.mant * 2^14 = mant24 / 2^9: round the low 9 bits, ties to even */
  const uint32_t low = mant24 & 0x1FFu;
  uint32_t mm = mant24 >> 9; /* [2^14, 2^15) */
  if (low > 0x100u || (low == 0x100u && (mm & 1u))) {
    mm += 1u;
  }
  int ex = exp8 - 126; /* frexpf's exponent: v = f * 2^ex, f in [0.5, 1) */
  if (mm >= 32768u) {
    mm = 16384u;
    ex += 1;
  }
  *m = (int16_t)mm;
  *e = (int8_t)(ex - 15);
}

/** signed v -> |m| in [2^29, 2^30), e with v ~= m * 2^e; 0 -> 0, 0.
 *  m = 1.mant * 2^29 exactly (24 significant bits fit), as lrintf(frexpf
 *  (v) * 2^30) gave. */
static inline void iq_split30(float v, int32_t *m, int8_t *e) {
  uint32_t mant24;
  int exp8, neg;
  if (!iq_float_fields(v, &mant24, &exp8, &neg)) {
    *m = 0;
    *e = 0;
    return;
  }
  const int32_t mm = (int32_t)(mant24 << 6); /* [2^29, 2^30) */
  *m = neg ? -mm : mm;
  *e = (int8_t)((exp8 - 126) - 30);
}

/** One row's dequant format for one staged batch (header, NUMBER FORMATS). */
typedef struct {
  int nb; /**< nbits(max |A|) over the batch */
  int sa; /**< A is shifted left by this to fill 30 bits */
  int B;  /**< every |result| < 2^B */
  int F;  /**< q = value * 2^F, |q| < 2^30 */
  int es; /**< the row's activation scale exponent */
  int32_t ms;
} iq_row_fmt;

static inline void iq_row_fmt_set(iq_row_fmt *o, uint32_t amax, int32_t ms,
                                  int es, int t_wb, int t_bb) {
  o->nb = iq_nbits(amax);
  o->sa = 30 - o->nb;
  o->ms = ms;
  o->es = es;
  int B = o->nb + 15 + es + t_wb;
  if (t_bb > B) {
    B = t_bb;
  }
  o->B = B + 1;
  o->F = 30 - o->B;
}

/** The batch's tile bounds: the largest over its n-tiles [t0, t0 + n). */
static inline void iq_batch_bounds(const hvx_int_wq *q, uint32_t t0, uint32_t n,
                                   int *wb, int *bb) {
  int w = -128, b = -128;
  for (uint32_t t = t0; t < t0 + n; ++t) {
    w = q->t_wb[t] > w ? q->t_wb[t] : w;
    if (q->t_bb) {
      b = q->t_bb[t] > b ? q->t_bb[t] : b;
    }
  }
  *wb = w;
  *bb = b;
}

/** A row's requantization parameters from its scanned range. */
typedef struct {
  int zero;  /**< range empty: bytes 0, scale 1, zp 0 */
  int ls;    /**< v and R are shifted left by this */
  int e;     /**< y = round(mulhi(v << ls, M) / 2^e) */
  int32_t M; /**< 255 * 2^(15+e) / (R << ls), in [16320, 32640] */
  int32_t z; /**< the zero point, the same formula applied to -min */
  float scale;
} iq_rq_params;

static inline void iq_rq_params_set(int32_t mn, int32_t mx, int Fmin,
                                    iq_rq_params *p) {
  const int64_t R64 = (int64_t)mx - (int64_t)mn; /* < 2^31 */
  if (R64 == 0) {
    p->zero = 1;
    p->ls = 0;
    p->e = 0;
    p->M = 0;
    p->z = 0;
    p->scale = 1.0f;
    return;
  }
  const int32_t R = (int32_t)R64;
  int ls = 30 - iq_nbits((uint32_t)R);
  if (ls < 0) {
    ls = 0;
  }
  const int32_t Rn = iq_shl(R, ls);
  const int e = iq_nbits((uint32_t)Rn) - 9; /* 21 or 22 */
  const int64_t num = (int64_t)255 << (15 + e);
  p->zero = 0;
  p->ls = ls;
  p->e = e;
  p->M = (int32_t)((num + (Rn / 2)) / Rn);
  p->z = iq_asr_rnd(iq_mulhi_q15(iq_shl(-mn, ls), p->M), e);
  p->scale = ldexpf((float)R / 255.0f, -Fmin);
}

#endif /* __NNTRAINER_HVX_INT_EPILOGUE_IMPL_H__ */
