// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_int_epilogue.c
 * @date   28 Sep 2026
 * @brief  Integer-only MoE gate_up epilogue (see the header)
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Portable C, no intrinsics: every helper below names the HVX instruction
 * it stands for, and a vectorised version must keep these exact
 * semantics so the host check stays its reference. Per-element work only;
 * per-row scalar work (bounds, the requant multiplier) is 64 items a
 * block and stays scalar in the vectorised version too.
 *
 * ponytail: this runs SCALAR on the DSP as it stands -- roughly 100
 * integer operations an element, about 20x the f32 HVX epilogue's worker
 * time -- so the skel built with it measures perplexity (doc 53 section
 * 9's gate), not speed. The upgrade path is one-to-one: each helper is an
 * HVX instruction on 32 lanes, the row loop becomes the lane loop, and
 * the per-lane shift amounts (we, be, k) are vectors loaded per tile.
 */

#include "hvx_int_epilogue.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>

#include "hvx_int_epilogue_impl.h"

/* ---- the primitive set: hvx_int_epilogue_impl.h; exposed for the checks */

int32_t hvx_int_mulhi_q15(int32_t a, int32_t b16) {
  return iq_mulhi_q15(a, b16);
}
#define MULHI(a, b) iq_mulhi_q15((a), (b))

/* ---- the bake -------------------------------------------------------- */

int hvx_int_wq_bake(hvx_int_wq **out, const float *w_scale, const float *bias,
                    uint32_t N) {
  if (!out || !w_scale || N == 0u || (N % 32u) != 0u) {
    return AEE_EBADPARM;
  }
  const uint32_t nt = N / 32u;
  int has_bias = 0;
  if (bias) {
    for (uint32_t c = 0; c < N; ++c) {
      if (bias[c] != 0.0f) {
        has_bias = 1;
        break;
      }
    }
  }
  hvx_int_wq *q = (hvx_int_wq *)calloc(1u, sizeof *q);
  if (!q) {
    return AEE_ENOMEMORY;
  }
  q->N = N;
  /* wm, we and be are read by the HVX version as whole 128-byte vectors
     of which the first 64 or 32 bytes are the tile's, so each carries
     one vector of slack past its last column. */
  q->wm = (int16_t *)calloc(N + 64u, sizeof(int16_t));
  q->we = (int8_t *)calloc(N + 128u, 1u);
  q->t_wb = (int8_t *)malloc(nt);
  if (has_bias) {
    q->bm = (int32_t *)malloc(sizeof(int32_t) * N);
    q->be = (int8_t *)calloc(N + 128u, 1u);
    q->t_bb = (int8_t *)malloc(nt);
  }
  if (!q->wm || !q->we || !q->t_wb ||
      (has_bias && (!q->bm || !q->be || !q->t_bb))) {
    hvx_int_wq_free(q);
    return AEE_ENOMEMORY;
  }
  for (uint32_t t = 0; t < nt; ++t) {
    int wb = -128, bb = -128;
    for (uint32_t c = t * 32u; c < (t + 1u) * 32u; ++c) {
      iq_split15(w_scale[c], &q->wm[c], &q->we[c]);
      if (q->wm[c] != 0) {
        wb = (15 + q->we[c] > wb) ? 15 + q->we[c] : wb;
      }
      if (has_bias) {
        iq_split30(bias[c], &q->bm[c], &q->be[c]);
        if (q->bm[c] != 0) {
          bb = (30 + q->be[c] > bb) ? 30 + q->be[c] : bb;
        }
      }
    }
    q->t_wb[t] = (int8_t)wb;
    if (has_bias) {
      q->t_bb[t] = (int8_t)bb;
    }
  }
  *out = q;
  return AEE_SUCCESS;
}

void hvx_int_wq_free(hvx_int_wq *q) {
  if (!q) {
    return;
  }
  free(q->wm);
  free(q->we);
  free(q->bm);
  free(q->be);
  free(q->t_wb);
  free(q->t_bb);
  free(q);
}

/* ---- the sigmoid ----------------------------------------------------- */

/** 2^-f on [0, 1), f in Q15, result Q30. Degree 5, near-minimax. */
const int32_t hvx_int_exp2_q30[6] = {1073741765, -744256846, 257890762,
                                     -59377499,  9890100,    -1017427};
/** 1/(1+e) on [0, 1], e in Q15, result Q30. Degree 7, near-minimax. */
const int32_t hvx_int_rcp_q30[8] = {1073740448,  -1073555713, 1069536611,
                                    -1036489911, 903790356,   -615589948,
                                    269881404,   -54443308};
/* IQ_LOG2E_HALF_Q15 (log2(e)/2, since the multiplier is 16-bit and
   log2(e) is not) and IQ_X_MAX_Q24 (|x| clamped to 16: sigmoid is within
   1.2e-7 of 0 or 1 beyond it, and 2^28 leaves two bits of headroom) are
   in hvx_int_epilogue_impl.h, shared with the vector version. */

int32_t hvx_int_sigmoid_q15(int32_t g_q, int F) {
  /* x = g * 2^-F -> Q24, saturating. F < 24 is a left shift: the value
     is compared against the clamp BEFORE shifting, so nothing wraps. */
  int32_t x;
  const int k = F - 24;
  if (k >= 0) {
    x = iq_asr_rnd(g_q, k);
  } else {
    const int ls = (-k > 30) ? 30 : -k;
    const int32_t lim = IQ_X_MAX_Q24 >> ls; /* |g_q| >= lim saturates */
    if (g_q >= lim) {
      x = IQ_X_MAX_Q24;
    } else if (g_q <= -lim) {
      x = -IQ_X_MAX_Q24;
    } else {
      x = iq_shl(g_q, ls);
    }
  }
  x = iq_max(iq_min(x, IQ_X_MAX_Q24), -IQ_X_MAX_Q24);
  const int32_t a = x < 0 ? -x : x; /* [0, 2^28] */
  /* t = a * log2(e): Q23 (the Q15 constant carries log2(e)/2). */
  const int32_t t = MULHI(a, IQ_LOG2E_HALF_Q15);
  int n = t >> 23; /* 0..23 */
  if (n > 30) {
    n = 30;
  }
  const int32_t f = t & ((1 << 23) - 1);
  int32_t f15 = (f + 128) >> 8; /* Q23 -> Q15, rounded */
  if (f15 > 32767) {
    f15 = 32767;
  }
  int32_t p = hvx_int_exp2_q30[5];
  for (int i = 4; i >= 0; --i) {
    p = MULHI(p, f15) + hvx_int_exp2_q30[i];
  }
  /* e^-|x| = 2^-n * 2^-f, Q30 -> Q15 for the next polynomial's argument */
  const int32_t e = iq_asr(p, n);
  int32_t e15 = (e + (1 << 14)) >> 15;
  if (e15 > 32767) {
    e15 = 32767;
  }
  int32_t r = hvx_int_rcp_q30[7];
  for (int i = 6; i >= 0; --i) {
    r = MULHI(r, e15) + hvx_int_rcp_q30[i];
  }
  int32_t s = (r + (1 << 14)) >> 15; /* sigmoid(|x|) in Q15, [2^14, 2^15] */
  if (s > 32767) {
    s = 32767;
  }
  return (x >= 0) ? s : 32767 - s;
}

/* ---- the gate_up epilogue -------------------------------------------- */

/** (A * s * w + b) * 2^F, the dequant of hvx_dequant_i32.h in fixed
 *  point. Two multiply-highs and three shifts; the shifts are right
 *  shifts by construction (the bound puts the largest column at 2^30). */
static inline int32_t iq_lin(int32_t A, const iq_row_fmt *fm, int16_t wm,
                             int8_t we, const int32_t *bm, const int8_t *be) {
  int32_t g = 0;
  if (wm != 0) {
    const int32_t A1 = iq_shl(A, fm->sa);
    const int32_t p1 = MULHI(A1, wm);
    const int32_t p2 = MULHI(p1, fm->ms);
    g = iq_asr_rnd(p2, fm->B - fm->nb - fm->es - we - 30);
  }
  if (bm && *bm != 0) {
    g += iq_asr_rnd(*bm, fm->B - 30 - *be);
  }
  return g;
}

/** Largest |acc - zp * colsum| of row r over the batch's tiles starting
 *  at staged slot s0, for n tiles at columns col0 + 32 j. */
static uint32_t iq_row_amax(const hvx_int_gu_job *c, uint32_t r, uint32_t s0,
                            uint32_t col0, int32_t zp) {
  uint32_t amax = 0u;
  for (uint32_t j = 0; j < c->n_pairs; ++j) {
    const int32_t *row =
      (const int32_t *)(c->tiles_base + (size_t)(s0 + j) * c->tile_stride) +
      (size_t)r * c->row_stride;
    const int32_t *cs = c->colsum_w + col0 + j * 32u;
    for (uint32_t l = 0; l < 32u; ++l) {
      const int32_t A = row[l] - zp * cs[l];
      const uint32_t a = (uint32_t)(A < 0 ? -A : A);
      amax = a > amax ? a : amax;
    }
  }
  return amax;
}

void hvx_int_gu_worker_c(uint32_t n_threads, uint32_t i, void *vjob) {
  const hvx_int_gu_job *c = (const hvx_int_gu_job *)vjob;
  const uint32_t lo = (uint32_t)((uint64_t)c->m_count * i / n_threads);
  const uint32_t hi = (uint32_t)((uint64_t)c->m_count * (i + 1) / n_threads);
  const uint32_t cg0 = c->g0 * 32u;    /* first gate column */
  const uint32_t cu0 = c->inter + cg0; /* first up column */
  const uint32_t tg0 = c->g0;          /* first gate n-tile */
  const uint32_t tu0 = c->inter / 32u + c->g0;
  int wb_g, bb_g, wb_u, bb_u;
  iq_batch_bounds(c->wq, tg0, c->n_pairs, &wb_g, &bb_g);
  iq_batch_bounds(c->wq, tu0, c->n_pairs, &wb_u, &bb_u);

  for (uint32_t r = lo; r < hi; ++r) {
    int16_t ms;
    int8_t es;
    iq_split15(c->act_scale[r], &ms, &es);
    const int32_t zp = c->act_zp[r];
    iq_row_fmt fg, fu;
    iq_row_fmt_set(&fg, iq_row_amax(c, r, 0u, cg0, zp), ms, es, wb_g, bb_g);
    iq_row_fmt_set(&fu, iq_row_amax(c, r, c->n_pairs, cu0, zp), ms, es, wb_u,
                   bb_u);
    /* h = (g * sigmoid(g)) * u: the first product keeps g's format, the
       second is a full 32 x 32 through the 16-bit multiplier -- u's high
       15 bits and its low 15 bits separately -- so u keeps its precision
       and h lands in Q(Fg + Fu - 30) with |h| < 2^30. */
    hvx_int_hmeta *meta = c->h_meta + (size_t)r * c->meta_stride + c->batch;
    meta->F = fg.F + fu.F - 30;
    int32_t mn = 0, mx = 0;
    int32_t *dst = c->dst + (size_t)r * c->dst_stride + cg0;
    for (uint32_t j = 0; j < c->n_pairs; ++j) {
      const int32_t *grow =
        (const int32_t *)(c->tiles_base + (size_t)j * c->tile_stride) +
        (size_t)r * c->row_stride;
      const int32_t *urow =
        (const int32_t *)(c->tiles_base +
                          (size_t)(c->n_pairs + j) * c->tile_stride) +
        (size_t)r * c->row_stride;
      const uint32_t cg = cg0 + j * 32u, cu = cu0 + j * 32u;
      for (uint32_t l = 0; l < 32u; ++l) {
        const int32_t Ag = grow[l] - zp * c->colsum_w[cg + l];
        const int32_t Au = urow[l] - zp * c->colsum_w[cu + l];
        const int32_t g = iq_lin(Ag, &fg, c->wq->wm[cg + l], c->wq->we[cg + l],
                                 c->wq->bm ? &c->wq->bm[cg + l] : NULL,
                                 c->wq->be ? &c->wq->be[cg + l] : NULL);
        const int32_t u = iq_lin(Au, &fu, c->wq->wm[cu + l], c->wq->we[cu + l],
                                 c->wq->bm ? &c->wq->bm[cu + l] : NULL,
                                 c->wq->be ? &c->wq->be[cu + l] : NULL);
        const int32_t sg = hvx_int_sigmoid_q15(g, fg.F);
        const int32_t gs = MULHI(g, sg);
        const int32_t u_hi = u >> 15;    /* floor */
        const int32_t u_lo = u & 0x7FFF; /* [0, 2^15) */
        const int32_t h = MULHI(gs, u_hi) + (MULHI(gs, u_lo) >> 15);
        dst[j * 32u + l] = h;
        mn = iq_min(mn, h);
        mx = iq_max(mx, h);
      }
    }
    meta->mn = mn;
    meta->mx = mx;
  }
}

/* ---- the requantization ---------------------------------------------- */

void hvx_int_rq_rows_c(const int32_t *h, uint32_t h_stride,
                       const hvx_int_hmeta *h_meta, uint32_t meta_stride,
                       uint32_t n_batches, uint32_t batch_cols, uint32_t inter,
                       uint32_t m_valid, uint32_t m0, uint32_t m1, float *scale,
                       int32_t *zp, uint8_t *out_ah) {
  const uint32_t n_ktiles = inter / 32u;
  for (uint32_t m = m0; m < m1; ++m) {
    uint8_t *blk = out_ah + (size_t)(m / 64u) * n_ktiles * 2048u;
    uint8_t *row0 = blk + (m % 64u) * 32u;
    if (m >= m_valid) {
      for (uint32_t kt = 0; kt < n_ktiles; ++kt) {
        memset(row0 + (size_t)kt * 2048u, 0, 32u);
      }
      scale[m] = 1.0f;
      zp[m] = 0;
      continue;
    }
    const int32_t *hr = h + (size_t)m * h_stride;
    int sh[16]; /* MOE_MAX_CHUNKS batches at most */
    int Fmin;
    int32_t mn, mx;
    iq_row_range(h_meta + (size_t)m * meta_stride, n_batches, sh, &Fmin, &mn,
                 &mx);
    iq_rq_params P;
    iq_rq_params_set(mn, mx, Fmin, &P);
    if (P.zero) {
      for (uint32_t kt = 0; kt < n_ktiles; ++kt) {
        memset(row0 + (size_t)kt * 2048u, 0, 32u);
      }
      scale[m] = 1.0f;
      zp[m] = 0;
      continue;
    }
    const int ls = P.ls, e = P.e;
    const int32_t M = P.M, z = P.z;
    for (uint32_t b = 0; b < n_batches; ++b) {
      const uint32_t c0 = b * batch_cols;
      const uint32_t c1 = (c0 + batch_cols < inter) ? c0 + batch_cols : inter;
      for (uint32_t cc = c0; cc < c1; ++cc) {
        const int32_t v = iq_asr_rnd(hr[cc], sh[b]);
        const int32_t y = iq_asr_rnd(MULHI(iq_shl(v, ls), M), e);
        int32_t q = y + z;
        q = q < 0 ? 0 : (q > 255 ? 255 : q);
        row0[(size_t)(cc / 32u) * 2048u + (cc % 32u)] = (uint8_t)q;
      }
    }
    scale[m] = P.scale;
    zp[m] = z;
  }
}

/* ---- the stage-by-stage probe the device self-check compares --------- */

void hvx_int_dbg_gu_c(const hvx_int_gu_job *c, uint32_t r, uint32_t j,
                      int32_t *out, int32_t *meta) {
  const uint32_t cg0 = c->g0 * 32u, cu0 = c->inter + cg0;
  int wb_g, bb_g, wb_u, bb_u;
  iq_batch_bounds(c->wq, c->g0, c->n_pairs, &wb_g, &bb_g);
  iq_batch_bounds(c->wq, c->inter / 32u + c->g0, c->n_pairs, &wb_u, &bb_u);
  int16_t ms;
  int8_t es;
  iq_split15(c->act_scale[r], &ms, &es);
  const int32_t zp = c->act_zp[r];
  const uint32_t amax_g = iq_row_amax(c, r, 0u, cg0, zp);
  const uint32_t amax_u = iq_row_amax(c, r, c->n_pairs, cu0, zp);
  iq_row_fmt fg, fu;
  iq_row_fmt_set(&fg, amax_g, ms, es, wb_g, bb_g);
  iq_row_fmt_set(&fu, amax_u, ms, es, wb_u, bb_u);
  const uint32_t cg = cg0 + j * 32u, cu = cu0 + j * 32u;
  const int32_t *grow =
    (const int32_t *)(c->tiles_base + (size_t)j * c->tile_stride) +
    (size_t)r * c->row_stride;
  const int32_t *urow =
    (const int32_t *)(c->tiles_base +
                      (size_t)(c->n_pairs + j) * c->tile_stride) +
    (size_t)r * c->row_stride;
  for (uint32_t l = 0; l < 32u; ++l) {
    const int32_t Ag = grow[l] - zp * c->colsum_w[cg + l];
    const int32_t Au = urow[l] - zp * c->colsum_w[cu + l];
    const int32_t g = iq_lin(Ag, &fg, c->wq->wm[cg + l], c->wq->we[cg + l],
                             c->wq->bm ? &c->wq->bm[cg + l] : NULL,
                             c->wq->be ? &c->wq->be[cg + l] : NULL);
    const int32_t u = iq_lin(Au, &fu, c->wq->wm[cu + l], c->wq->we[cu + l],
                             c->wq->bm ? &c->wq->bm[cu + l] : NULL,
                             c->wq->be ? &c->wq->be[cu + l] : NULL);
    const int32_t sg = hvx_int_sigmoid_q15(g, fg.F);
    const int32_t gs = MULHI(g, sg);
    const int32_t u_hi = u >> 15, u_lo = u & 0x7FFF;
    out[0 * 32 + l] = Ag;
    out[1 * 32 + l] = Au;
    out[2 * 32 + l] = c->wq->we[cg + l];
    out[3 * 32 + l] = (int32_t)((uint32_t)(uint16_t)c->wq->wm[cg + l] << 16);
    out[4 * 32 + l] = g;
    out[5 * 32 + l] = u;
    out[6 * 32 + l] = sg;
    out[7 * 32 + l] = MULHI(gs, u_hi) + (MULHI(gs, u_lo) >> 15);
  }
  meta[0] = (int32_t)amax_g;
  meta[1] = (int32_t)amax_u;
  meta[2] = fg.F;
  meta[3] = fu.F;
  meta[4] = fg.sa;
  meta[5] = fg.B;
  meta[6] = ms;
  meta[7] = es;
}
