// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_rope_rows_f32.c
 * @date   6 October 2026
 * @brief  RoPE over M rows of heads of f32 on HVX, in place
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "hvx_rope_rows_f32.h"
#include "hvx_convert.h"
#include "hvx_rmsnorm_rows_f32.h"

#include <stddef.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

/** @brief f32 lanes per HVX vector at 128B. */
#define LANES 32u

typedef struct {
  float *x;
  const float *cs, *gamma;
  uint32_t M, n, ld, hd, chunk;
  float eps;
  uint16_t *y16; /**< NULL, or the row's u16 copy (ld16 a row) */
  uint32_t ld16;
  float q_inv, q_zp05; /**< 1 / scale, zp + 0.5 */
} rows_ctx;

/* quant_u16_f32 over one row: x * inv + (zp + 0.5), clamped, truncated
   (Vw_equals_Vsf, the numeric round-to-zero convert), 64 lanes a store. */
static void quant16_row(const float *x, uint16_t *y, uint32_t n, float inv,
                        float zp05) {
  const HVX_Vector vinv = hvx_splat_sf(inv), vzp = hvx_splat_sf(zp05);
  const HVX_Vector vlo = Q6_V_vzero(), vhi = hvx_splat_sf(65535.0f);
  for (uint32_t k = 0; k < n; k += 2u * LANES) {
    HVX_Vector a = Q6_Vsf_vadd_VsfVsf(
      Q6_Vsf_vmpy_VsfVsf(*(const HVX_UVector *)(x + k), vinv), vzp);
    HVX_Vector b = Q6_Vsf_vadd_VsfVsf(
      Q6_Vsf_vmpy_VsfVsf(*(const HVX_UVector *)(x + k + LANES), vinv), vzp);
    a = Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(a, vlo), vhi);
    b = Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(b, vlo), vhi);
    *(HVX_UVector *)(y + k) =
      Q6_Vuh_vpack_VwVw_sat(Q6_Vw_equals_Vsf(b), Q6_Vw_equals_Vsf(a));
  }
}

/* One row's heads, in place; cs_row = cos[hd] then sin[hd]. */
static void rope_row(float *x, uint32_t n, uint32_t hd, const float *cs_row) {
  const uint32_t half = hd / 2u;
  const uint32_t nvec = half / LANES;
  const HVX_UVector *vc = (const HVX_UVector *)cs_row;
  const HVX_UVector *vs = (const HVX_UVector *)(cs_row + hd);
  for (uint32_t h = 0; h < n / hd; ++h) {
    float *head = x + (size_t)h * hd;
    HVX_UVector *va = (HVX_UVector *)head;
    HVX_UVector *vb = (HVX_UVector *)(head + half);
    for (uint32_t k = 0; k < nvec; ++k) {
      const HVX_Vector a = va[k], b = vb[k];
      /* a*cos - b*sin and a*sin + b*cos, each product rounded once as
         the CPU's (no fused multiply-add there either) */
      va[k] = Q6_Vsf_vsub_VsfVsf(Q6_Vsf_vmpy_VsfVsf(a, vc[k]),
                                 Q6_Vsf_vmpy_VsfVsf(b, vs[k]));
      vb[k] = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(a, vs[k]),
                                 Q6_Vsf_vmpy_VsfVsf(b, vc[k]));
    }
  }
}

/* chunk != 0: the row's RMSNorm first (hvx_rmsnorm_rows_ld_f32's, chunk by
   chunk), then its RoPE while the row is still in cache -- one pass over
   the qkv output instead of two (doc: 63 + 55 ms of a 1024-row prefill on
   the S26 Ultra, more than the matmul's 87). */
static void rows_worker(uint32_t n_threads, uint32_t i, void *v) {
  const rows_ctx *c = (const rows_ctx *)v;
  const uint32_t lo = (uint32_t)(((uint64_t)c->M * i) / n_threads);
  const uint32_t hi = (uint32_t)(((uint64_t)c->M * (i + 1u)) / n_threads);
  for (uint32_t r = lo; r < hi; ++r) {
    float *row = c->x + (size_t)r * c->ld;
    if (c->chunk)
      hvx_rmsnorm_row_chunks_f32(row, row, c->gamma, c->n, c->chunk, c->eps);
    rope_row(row, c->n, c->hd, c->cs + (size_t)r * 2u * c->hd);
    if (c->y16)
      quant16_row(row, c->y16 + (size_t)r * c->ld16, c->n, c->q_inv, c->q_zp05);
  }
}

int hvx_rope_rows_f32(float *x, uint32_t M, uint32_t n, uint32_t hd,
                      const float *cs, hvx_worker_pool *pool) {
  return hvx_rope_rows_ld_f32(x, M, n, n, hd, cs, pool);
}

int hvx_rope_rows_ld_f32(float *x, uint32_t M, uint32_t n, uint32_t ld,
                         uint32_t hd, const float *cs, hvx_worker_pool *pool) {
  if (!x || !cs || M == 0u || hd == 0u || (hd / 2u) % LANES != 0u || n == 0u ||
      n % hd != 0u || ld < n) {
    return -1;
  }
  rows_ctx c = {x, cs, NULL, M, n, ld, hd, 0u, 0.0f, NULL, 0u, 0.0f, 0.0f};
  hvx_worker_pool_run(pool, rows_worker, &c, M);
  return 0;
}

int hvx_norm_rope_rows_ld_f32(float *x, uint32_t M, uint32_t n, uint32_t ld,
                              uint32_t chunk, const float *gamma, float eps,
                              uint32_t hd, const float *cs,
                              hvx_worker_pool *pool) {
  if (!x || !cs || M == 0u || hd == 0u || (hd / 2u) % LANES != 0u || n == 0u ||
      n % hd != 0u || ld < n || chunk == 0u || chunk % LANES != 0u ||
      n % chunk != 0u) {
    return -1;
  }
  rows_ctx c = {x, cs, gamma, M, n, ld, hd, chunk, eps, NULL, 0u, 0.0f, 0.0f};
  hvx_worker_pool_run(pool, rows_worker, &c, M);
  return 0;
}

int hvx_norm_rope_rows_ld_q16_f32(float *x, uint32_t M, uint32_t n, uint32_t ld,
                                  uint32_t chunk, const float *gamma, float eps,
                                  uint32_t hd, const float *cs, uint16_t *y16,
                                  uint32_t ld16, float q_inv, float q_zp,
                                  hvx_worker_pool *pool) {
  if (!x || !cs || !y16 || M == 0u || hd == 0u || (hd / 2u) % LANES != 0u ||
      n == 0u || n % hd != 0u || n % (2u * LANES) != 0u || ld < n || ld16 < n ||
      chunk == 0u || chunk % LANES != 0u || n % chunk != 0u ||
      !(q_inv > 0.0f)) {
    return -1;
  }
  rows_ctx c = {x,     cs,  gamma, M,    n,     ld,         hd,
                chunk, eps, y16,   ld16, q_inv, q_zp + 0.5f};
  hvx_worker_pool_run(pool, rows_worker, &c, M);
  return 0;
}
