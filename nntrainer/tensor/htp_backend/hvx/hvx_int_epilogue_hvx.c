// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_int_epilogue_hvx.c
 * @date   28 Sep 2026
 * @brief  The integer MoE epilogue on HVX, bit-identical to
 *         hvx_int_epilogue.c (doc 53 section 9.5)
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * hvx_int_epilogue.c is the specification: every helper there names the
 * instruction that implements it, and this file is that mapping applied
 * -- the row loops are the same, the 32-column loop becomes one vector.
 * Per-row scalar work (formats, bounds, the requant multiplier) is the
 * same code, from hvx_int_epilogue_impl.h. The two must agree bit for
 * bit, and nntr_hvx_int_epilogue.c's self-check (unittest_hvx_int_epilogue
 * on the device) is where that is established; a mismatch there is a bug
 * in this file or a misread of an instruction, never a tolerance.
 *
 * Instruction notes, where the correspondence needs a word:
 * - mulhi_q15 is Q6_Vw_vmpyo_VwVh_s1_rnd_sat with the 16-bit multiplier in
 *   the HIGH halfword of each lane (vhi16 puts it there).
 * - asr_rnd with a per-lane amount k adds (1 << k) >> 1, which is 0 for
 *   k = 0 and 2^(k-1) otherwise, then vasr by k. Amounts are clamped to
 *   [0, 30] first, as the reference does, because the hardware masks the
 *   amount and the add must not overflow.
 * - zp * colsum is Q6_Vw_vmpyie_VwVuh against a splat of zp: word times
 *   the lane's low unsigned halfword, exact for zp <= 255 and |colsum| <
 *   2^16. (vmpyi(Vu.w, Rt.h) is not that: it pairs even lanes with the
 *   scalar's low halfword and odd lanes with its high one.)
 * - The lane reduce for max |A| and for the requant's min/max is five
 *   rotate-and-combine steps, then lane 0.
 */

#include <string.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_int_epilogue.h"
#include "hvx_int_epilogue_impl.h"

#define VLEN 128u
#define LANES 32u

static inline HVX_Vector vsplat(int32_t v) { return Q6_V_vsplat_R(v); }
static inline HVX_Vector vload(const void *p) {
  return *(const HVX_UVector *)p;
}
/** a 16-bit value in each lane -> the high halfword, for vmulhi. */
static inline HVX_Vector vhi16(HVX_Vector b16) {
  return Q6_Vw_vasl_VwR(b16, 16);
}
/** round(a * b16 / 2^15), b16 in the high halfwords. */
static inline HVX_Vector vmulhi(HVX_Vector a, HVX_Vector b_hi) {
  return Q6_Vw_vmpyo_VwVh_s1_rnd_sat(a, b_hi);
}
/** iq_asr_rnd by one scalar amount. */
static inline HVX_Vector vasr_rnd_R(HVX_Vector x, int k) {
  if (k <= 0) {
    return x;
  }
  if (k > 30) {
    k = 30;
  }
  return Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(x, vsplat(1 << (k - 1))), k);
}
/** k clamped to [0, 30] per lane. */
static inline HVX_Vector vclamp_shift(HVX_Vector k) {
  return Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(k, Q6_V_vzero()), vsplat(30));
}
/** iq_asr_rnd by a per-lane amount already clamped. */
static inline HVX_Vector vasr_rnd_V(HVX_Vector x, HVX_Vector k) {
  const HVX_Vector rnd =
    Q6_Vw_vasr_VwR(Q6_Vw_vasl_VwVw(vsplat(1), k), 1); /* 2^(k-1), 0 at 0 */
  return Q6_Vw_vasr_VwVw(Q6_Vw_vadd_VwVw(x, rnd), k);
}
/** 32 int16 at p -> 32 words (reads a whole vector: 64 bytes of slack). */
static inline HVX_Vector vload_i16x32(const int16_t *p) {
  return Q6_V_lo_W(Q6_Ww_vunpack_Vh(vload(p)));
}
/** 32 int8 at p -> 32 words, sign-extended (96 bytes of slack). */
static inline HVX_Vector vload_i8x32(const int8_t *p) {
  return Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(vload(p)))));
}
/** Lane 0 of v, through an UNALIGNED store to a plain stack array -- the
 *  idiom hvx_quant_u8.c uses. An aligned vector store to a stack buffer
 *  is not safe here: the hardware masks the address to 128 bytes, so if
 *  the frame is not aligned the store lands elsewhere and the read is
 *  stale, silently. */
static inline uint32_t vlane0_u32(HVX_Vector v) {
  uint32_t buf[LANES];
  *(HVX_UVector *)buf = v;
  return buf[0];
}
static inline int32_t vlane0_s32(HVX_Vector v) {
  int32_t buf[LANES];
  *(HVX_UVector *)buf = v;
  return buf[0];
}

/* ---- the sigmoid, lane-wise hvx_int_sigmoid_q15 ---------------------- */

static inline HVX_Vector vsigmoid_q15(HVX_Vector g, int F) {
  HVX_Vector x;
  const int k = F - 24;
  if (k >= 0) {
    x = vasr_rnd_R(g, k);
  } else {
    const int ls = (-k > 30) ? 30 : -k;
    const int32_t lim = IQ_X_MAX_Q24 >> ls;
    const HVX_Vector shifted = Q6_Vw_vasl_VwR(g, ls);
    /* g >= lim and g <= -lim, the first taking precedence as in the
       reference's if / else if */
    const HVX_VectorPred ge = Q6_Q_vcmp_gt_VwVw(g, vsplat(lim - 1));
    const HVX_VectorPred le = Q6_Q_vcmp_gt_VwVw(vsplat(-lim + 1), g);
    x = Q6_V_vmux_QVV(ge, vsplat(IQ_X_MAX_Q24),
                      Q6_V_vmux_QVV(le, vsplat(-IQ_X_MAX_Q24), shifted));
  }
  x = Q6_Vw_vmax_VwVw(Q6_Vw_vmin_VwVw(x, vsplat(IQ_X_MAX_Q24)),
                      vsplat(-IQ_X_MAX_Q24));
  const HVX_Vector a = Q6_Vw_vabs_Vw(x);
  const HVX_Vector t = vmulhi(a, vsplat(IQ_LOG2E_HALF_Q15 << 16));
  const HVX_Vector n = Q6_Vw_vmin_VwVw(Q6_Vw_vasr_VwR(t, 23), vsplat(30));
  const HVX_Vector f = Q6_V_vand_VV(t, vsplat((1 << 23) - 1));
  const HVX_Vector f15 = Q6_Vw_vmin_VwVw(
    Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(f, vsplat(128)), 8), vsplat(32767));
  const HVX_Vector f15h = vhi16(f15);
  HVX_Vector p = vsplat(hvx_int_exp2_q30[5]);
  for (int i = 4; i >= 0; --i) {
    p = Q6_Vw_vadd_VwVw(vmulhi(p, f15h), vsplat(hvx_int_exp2_q30[i]));
  }
  const HVX_Vector e = Q6_Vw_vasr_VwVw(p, n);
  const HVX_Vector e15 = Q6_Vw_vmin_VwVw(
    Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(e, vsplat(1 << 14)), 15), vsplat(32767));
  const HVX_Vector e15h = vhi16(e15);
  HVX_Vector r = vsplat(hvx_int_rcp_q30[7]);
  for (int i = 6; i >= 0; --i) {
    r = Q6_Vw_vadd_VwVw(vmulhi(r, e15h), vsplat(hvx_int_rcp_q30[i]));
  }
  const HVX_Vector s = Q6_Vw_vmin_VwVw(
    Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(r, vsplat(1 << 14)), 15), vsplat(32767));
  const HVX_VectorPred neg = Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), x);
  return Q6_V_vmux_QVV(neg, Q6_Vw_vsub_VwVw(vsplat(32767), s), s);
}

/* ---- the gate_up epilogue -------------------------------------------- */

/** iq_lin on a vector: A * s * w + b in Q(F). */
static inline HVX_Vector vlin(HVX_Vector A, const iq_row_fmt *fm,
                              HVX_Vector wm_hi, HVX_Vector we,
                              const HVX_Vector *bm, const HVX_Vector *be) {
  const HVX_Vector A1 = Q6_Vw_vasl_VwR(A, fm->sa);
  const HVX_Vector p1 = vmulhi(A1, wm_hi);
  const HVX_Vector p2 = vmulhi(p1, vsplat(fm->ms << 16));
  const HVX_Vector k =
    vclamp_shift(Q6_Vw_vsub_VwVw(vsplat(fm->B - fm->nb - fm->es - 30), we));
  HVX_Vector g = vasr_rnd_V(p2, k);
  if (bm) {
    const HVX_Vector kb =
      vclamp_shift(Q6_Vw_vsub_VwVw(vsplat(fm->B - 30), *be));
    g = Q6_Vw_vadd_VwVw(g, vasr_rnd_V(*bm, kb));
  }
  return g;
}

/** The row's tile of raw accumulators minus zp * colsum. */
static inline HVX_Vector vA(const hvx_int_gu_job *c, uint32_t r, uint32_t slot,
                            uint32_t col, int32_t zp) {
  const int32_t *row =
    (const int32_t *)(c->tiles_base + (size_t)slot * c->tile_stride) +
    (size_t)r * c->row_stride;
  /* zp * colsum as word x unsigned-halfword (the low halfword of each
     lane of a splat), NOT vmpyi(Vu.w, Rt.h): that one takes the scalar's
     low halfword for even lanes and its high halfword for odd lanes, so
     odd lanes were multiplied by 0 -- the self-check's "A_gate lane 1". */
  return Q6_Vw_vsub_VwVw(
    vload(row), Q6_Vw_vmpyie_VwVuh(vload(c->colsum_w + col), vsplat(zp)));
}

/** HVX has no unsigned word max; |A| < 2^24 (K = 2048 terms of at most
 *  255 x 8, and the zp * colsum correction of the same size), so the
 *  signed one is exact here, as the reference's unsigned max is. */
static uint32_t vrow_amax(const hvx_int_gu_job *c, uint32_t r, uint32_t s0,
                          uint32_t col0, int32_t zp) {
  HVX_Vector m = Q6_V_vzero();
  for (uint32_t j = 0; j < c->n_pairs; ++j) {
    m = Q6_Vw_vmax_VwVw(m, Q6_Vw_vabs_Vw(vA(c, r, s0 + j, col0 + j * 32u, zp)));
  }
  for (uint32_t rot = VLEN / 2u; rot >= 4u; rot >>= 1) {
    m = Q6_Vw_vmax_VwVw(m, Q6_V_vror_VR(m, (int)rot));
  }
  return vlane0_u32(m);
}

void hvx_int_gu_worker_hvx(uint32_t n_threads, uint32_t i, void *vjob) {
  const hvx_int_gu_job *c = (const hvx_int_gu_job *)vjob;
  const uint32_t lo = (uint32_t)((uint64_t)c->m_count * i / n_threads);
  const uint32_t hi = (uint32_t)((uint64_t)c->m_count * (i + 1) / n_threads);
  const uint32_t cg0 = c->g0 * 32u;
  const uint32_t cu0 = c->inter + cg0;
  int wb_g, bb_g, wb_u, bb_u;
  iq_batch_bounds(c->wq, c->g0, c->n_pairs, &wb_g, &bb_g);
  iq_batch_bounds(c->wq, c->inter / 32u + c->g0, c->n_pairs, &wb_u, &bb_u);
  const int has_bias = c->wq->bm != NULL;

  for (uint32_t r = lo; r < hi; ++r) {
    int16_t ms;
    int8_t es;
    iq_split15(c->act_scale[r], &ms, &es);
    const int32_t zp = c->act_zp[r];
    iq_row_fmt fg, fu;
    iq_row_fmt_set(&fg, vrow_amax(c, r, 0u, cg0, zp), ms, es, wb_g, bb_g);
    iq_row_fmt_set(&fu, vrow_amax(c, r, c->n_pairs, cu0, zp), ms, es, wb_u,
                   bb_u);
    c->h_e[(size_t)r * c->e_stride + c->batch] = (int16_t)(fg.F + fu.F - 30);
    int32_t *dst = c->dst + (size_t)r * c->dst_stride + cg0;
    for (uint32_t j = 0; j < c->n_pairs; ++j) {
      const uint32_t cg = cg0 + j * 32u, cu = cu0 + j * 32u;
      const HVX_Vector Ag = vA(c, r, j, cg, zp);
      const HVX_Vector Au = vA(c, r, c->n_pairs + j, cu, zp);
      const HVX_Vector wmg = vhi16(vload_i16x32(c->wq->wm + cg));
      const HVX_Vector wmu = vhi16(vload_i16x32(c->wq->wm + cu));
      const HVX_Vector weg = vload_i8x32(c->wq->we + cg);
      const HVX_Vector weu = vload_i8x32(c->wq->we + cu);
      HVX_Vector g, u;
      if (has_bias) {
        const HVX_Vector bmg = vload(c->wq->bm + cg);
        const HVX_Vector bmu = vload(c->wq->bm + cu);
        const HVX_Vector beg = vload_i8x32(c->wq->be + cg);
        const HVX_Vector beu = vload_i8x32(c->wq->be + cu);
        g = vlin(Ag, &fg, wmg, weg, &bmg, &beg);
        u = vlin(Au, &fu, wmu, weu, &bmu, &beu);
      } else {
        g = vlin(Ag, &fg, wmg, weg, NULL, NULL);
        u = vlin(Au, &fu, wmu, weu, NULL, NULL);
      }
      const HVX_Vector sg = vsigmoid_q15(g, fg.F);
      const HVX_Vector gs = vmulhi(g, vhi16(sg));
      const HVX_Vector u_hi = Q6_Vw_vasr_VwR(u, 15);
      const HVX_Vector u_lo = Q6_V_vand_VV(u, vsplat(0x7FFF));
      const HVX_Vector h = Q6_Vw_vadd_VwVw(
        vmulhi(gs, vhi16(u_hi)), Q6_Vw_vasr_VwR(vmulhi(gs, vhi16(u_lo)), 15));
      *(HVX_UVector *)(dst + j * 32u) = h;
    }
  }
}

/* ---- the requantization ---------------------------------------------- */

typedef struct {
  int valid;
  int Fmin;
  iq_rq_params P;
  int sh[16]; /**< per batch: h_e - Fmin (MOE_MAX_CHUNKS batches at most) */
} rq_row;

static void rq_row_scan(const int32_t *hr, const int16_t *er,
                        uint32_t n_batches, uint32_t batch_cols, uint32_t inter,
                        rq_row *o) {
  int Fmin = er[0];
  for (uint32_t b = 1; b < n_batches; ++b) {
    Fmin = er[b] < Fmin ? er[b] : Fmin;
  }
  o->Fmin = Fmin;
  HVX_Vector vmn = Q6_V_vzero(), vmx = Q6_V_vzero(); /* the range holds 0 */
  for (uint32_t b = 0; b < n_batches; ++b) {
    const int sh = er[b] - Fmin;
    o->sh[b] = sh;
    const uint32_t c0 = b * batch_cols;
    const uint32_t c1 = (c0 + batch_cols < inter) ? c0 + batch_cols : inter;
    for (uint32_t cc = c0; cc < c1; cc += 32u) {
      const HVX_Vector v = vasr_rnd_R(vload(hr + cc), sh);
      vmn = Q6_Vw_vmin_VwVw(vmn, v);
      vmx = Q6_Vw_vmax_VwVw(vmx, v);
    }
  }
  for (uint32_t rot = VLEN / 2u; rot >= 4u; rot >>= 1) {
    vmn = Q6_Vw_vmin_VwVw(vmn, Q6_V_vror_VR(vmn, (int)rot));
    vmx = Q6_Vw_vmax_VwVw(vmx, Q6_V_vror_VR(vmx, (int)rot));
  }
  iq_rq_params_set(vlane0_s32(vmn), vlane0_s32(vmx), Fmin, &o->P);
}

/** One row's u8 values for k-tile kt, as 32 words in [0, 255]. */
static inline HVX_Vector rq_row_tile(const int32_t *hr, const rq_row *o,
                                     uint32_t kt, uint32_t batch_cols) {
  if (!o->valid || o->P.zero) {
    return Q6_V_vzero();
  }
  const uint32_t b = (kt * 32u) / batch_cols;
  const HVX_Vector v = vasr_rnd_R(vload(hr + kt * 32u), o->sh[b]);
  const HVX_Vector y = vasr_rnd_R(
    vmulhi(Q6_Vw_vasl_VwR(v, o->P.ls), vsplat(o->P.M << 16)), o->P.e);
  const HVX_Vector q = Q6_Vw_vadd_VwVw(y, vsplat(o->P.z));
  return Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(q, Q6_V_vzero()), vsplat(255));
}

void hvx_int_rq_rows_hvx(const int32_t *h, uint32_t h_stride,
                         const int16_t *h_e, uint32_t e_stride,
                         uint32_t n_batches, uint32_t batch_cols,
                         uint32_t inter, uint32_t m_valid, uint32_t m0,
                         uint32_t m1, float *scale, int32_t *zp,
                         uint8_t *out_ah) {
  const uint32_t n_ktiles = inter / 32u;
  uint32_t m = m0;
  for (; m + 4u <= m1; m += 4u) {
    rq_row rows[4];
    for (uint32_t q = 0; q < 4u; ++q) {
      const uint32_t mm = m + q;
      rows[q].valid = mm < m_valid;
      if (rows[q].valid) {
        rq_row_scan(h + (size_t)mm * h_stride, h_e + (size_t)mm * e_stride,
                    n_batches, batch_cols, inter, &rows[q]);
        scale[mm] = rows[q].P.scale;
        zp[mm] = rows[q].P.z;
      } else {
        scale[mm] = 1.0f;
        zp[mm] = 0;
      }
    }
    uint8_t *blk = out_ah + (size_t)(m / 64u) * n_ktiles * 2048u;
    uint8_t *dst0 = blk + (m % 64u) * 32u; /* rows m..m+3: one 128 B store */
    for (uint32_t kt = 0; kt < n_ktiles; ++kt) {
      const HVX_Vector q0 =
        rq_row_tile(h + (size_t)(m + 0) * h_stride, &rows[0], kt, batch_cols);
      const HVX_Vector q1 =
        rq_row_tile(h + (size_t)(m + 1) * h_stride, &rows[1], kt, batch_cols);
      const HVX_Vector q2 =
        rq_row_tile(h + (size_t)(m + 2) * h_stride, &rows[2], kt, batch_cols);
      const HVX_Vector q3 =
        rq_row_tile(h + (size_t)(m + 3) * h_stride, &rows[3], kt, batch_cols);
      /* vpack(Vu, Vv):sat: Vv's elements in the low half, Vu's in the high
         -- the same idiom hvx_quant_u8.c's quant_pack_group4 documents. */
      const HVX_Vector vh01 = Q6_Vh_vpack_VwVw_sat(q1, q0);
      const HVX_Vector vh23 = Q6_Vh_vpack_VwVw_sat(q3, q2);
      *(HVX_UVector *)(dst0 + (size_t)kt * 2048u) =
        Q6_Vub_vpack_VhVh_sat(vh23, vh01);
    }
  }
  if (m < m1) {
    /* a partial group: the reference handles rows one at a time */
    hvx_int_rq_rows_c(h, h_stride, h_e, e_stride, n_batches, batch_cols, inter,
                      m_valid, m, m1, scale, zp, out_ah);
  }
}

/* ---- the stage-by-stage probe the device self-check compares --------- */

void hvx_int_dbg_gu_hvx(const hvx_int_gu_job *c, uint32_t r, uint32_t j,
                        int32_t *out, int32_t *meta) {
  const uint32_t cg0 = c->g0 * 32u, cu0 = c->inter + cg0;
  int wb_g, bb_g, wb_u, bb_u;
  iq_batch_bounds(c->wq, c->g0, c->n_pairs, &wb_g, &bb_g);
  iq_batch_bounds(c->wq, c->inter / 32u + c->g0, c->n_pairs, &wb_u, &bb_u);
  int16_t ms;
  int8_t es;
  iq_split15(c->act_scale[r], &ms, &es);
  const int32_t zp = c->act_zp[r];
  const uint32_t amax_g = vrow_amax(c, r, 0u, cg0, zp);
  const uint32_t amax_u = vrow_amax(c, r, c->n_pairs, cu0, zp);
  iq_row_fmt fg, fu;
  iq_row_fmt_set(&fg, amax_g, ms, es, wb_g, bb_g);
  iq_row_fmt_set(&fu, amax_u, ms, es, wb_u, bb_u);
  const uint32_t cg = cg0 + j * 32u, cu = cu0 + j * 32u;
  const HVX_Vector Ag = vA(c, r, j, cg, zp);
  const HVX_Vector Au = vA(c, r, c->n_pairs + j, cu, zp);
  const HVX_Vector wmg = vhi16(vload_i16x32(c->wq->wm + cg));
  const HVX_Vector wmu = vhi16(vload_i16x32(c->wq->wm + cu));
  const HVX_Vector weg = vload_i8x32(c->wq->we + cg);
  const HVX_Vector weu = vload_i8x32(c->wq->we + cu);
  HVX_Vector g, u;
  if (c->wq->bm) {
    const HVX_Vector bmg = vload(c->wq->bm + cg), bmu = vload(c->wq->bm + cu);
    const HVX_Vector beg = vload_i8x32(c->wq->be + cg);
    const HVX_Vector beu = vload_i8x32(c->wq->be + cu);
    g = vlin(Ag, &fg, wmg, weg, &bmg, &beg);
    u = vlin(Au, &fu, wmu, weu, &bmu, &beu);
  } else {
    g = vlin(Ag, &fg, wmg, weg, NULL, NULL);
    u = vlin(Au, &fu, wmu, weu, NULL, NULL);
  }
  const HVX_Vector sg = vsigmoid_q15(g, fg.F);
  const HVX_Vector gs = vmulhi(g, vhi16(sg));
  const HVX_Vector u_hi = Q6_Vw_vasr_VwR(u, 15);
  const HVX_Vector u_lo = Q6_V_vand_VV(u, vsplat(0x7FFF));
  const HVX_Vector h = Q6_Vw_vadd_VwVw(
    vmulhi(gs, vhi16(u_hi)), Q6_Vw_vasr_VwR(vmulhi(gs, vhi16(u_lo)), 15));
  HVX_UVector *o = (HVX_UVector *)out;
  o[0] = Ag;
  o[1] = Au;
  o[2] = weg; /* the unpacked per-lane exponents, as loaded */
  o[3] = wmg; /* wm in the high halfword */
  o[4] = g;
  o[5] = u;
  o[6] = sg;
  o[7] = h;
  meta[0] = (int32_t)amax_g;
  meta[1] = (int32_t)amax_u;
  meta[2] = fg.F;
  meta[3] = fu.F;
  meta[4] = fg.sa;
  meta[5] = fg.B;
  meta[6] = ms;
  meta[7] = es;
}
