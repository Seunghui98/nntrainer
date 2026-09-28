// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   nntr_hvx_int_epilogue.c
 * @date   28 Sep 2026
 * @brief  Device self-check: the HVX integer epilogue against its portable
 *         C reference, bit for bit, on data generated here
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Both implementations run on the same LFM2-shaped block (64 rows x
 * 2*1792 columns, four staged batches of 16, 16, 16 and 8 pairs, 61 valid
 * rows so three are padding) and every output is compared: the SwiGLU
 * mantissas and exponents of the gate_up jobs, then the u8 bytes, scales
 * and zero points of the requantization. The inputs come from an integer
 * generator seeded by the caller, with outlier rows and columns, and the
 * odd seeds give the weight a nonzero bias so that path is exercised too.
 * Nothing crosses FastRPC but the seed and eight counters, so the test
 * needs no host model -- the reference IS the host model, compiled for
 * the DSP.
 */

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <remote.h>
#include <stdlib.h>
#include <string.h>

#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

#include "hvx_int_epilogue.h"

#define SC_ROWS 64u
#define SC_VALID 61u
#define SC_INTER 1792u
#define SC_N (2u * SC_INTER)
#define SC_NT (SC_INTER / 32u)
#define SC_HALF 16u
#define SC_ESTRIDE 16u

static uint32_t sc_lcg(uint32_t *s) {
  *s = *s * 1664525u + 1013904223u;
  return *s >> 8;
}

int nntr_hvx_int_epilogue_selfcheck(remote_handle64 handle, uint32 seed,
                                    uint32 *counts, int countsLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || !counts || countsLen < 8) {
    return AEE_EBADPARM;
  }
  memset(counts, 0, sizeof(uint32) * (size_t)countsLen);
  uint32_t st = seed * 2654435761u + 12345u;
  int rc = AEE_ENOMEMORY;

  int32_t *acc = (int32_t *)malloc(sizeof(int32_t) * SC_N * SC_ROWS);
  int32_t *colsum = (int32_t *)malloc(sizeof(int32_t) * SC_N);
  float *w_scale = (float *)malloc(sizeof(float) * SC_N);
  float *bias = (float *)malloc(sizeof(float) * SC_N);
  float *act_scale = (float *)malloc(sizeof(float) * SC_ROWS);
  int32_t *act_zp = (int32_t *)malloc(sizeof(int32_t) * SC_ROWS);
  int32_t *stage =
    (int32_t *)malloc(sizeof(int32_t) * 2u * SC_HALF * SC_ROWS * 32u);
  int32_t *h_c = (int32_t *)malloc(sizeof(int32_t) * SC_ROWS * SC_INTER);
  int32_t *h_v = (int32_t *)malloc(sizeof(int32_t) * SC_ROWS * SC_INTER);
  int16_t *e_c = (int16_t *)malloc(sizeof(int16_t) * SC_ROWS * SC_ESTRIDE);
  int16_t *e_v = (int16_t *)malloc(sizeof(int16_t) * SC_ROWS * SC_ESTRIDE);
  uint8_t *ah_c = (uint8_t *)malloc((size_t)SC_ROWS * SC_INTER);
  uint8_t *ah_v = (uint8_t *)malloc((size_t)SC_ROWS * SC_INTER);
  float *sc_c = (float *)malloc(sizeof(float) * SC_ROWS);
  float *sc_v = (float *)malloc(sizeof(float) * SC_ROWS);
  int32_t *zp_c = (int32_t *)malloc(sizeof(int32_t) * SC_ROWS);
  int32_t *zp_v = (int32_t *)malloc(sizeof(int32_t) * SC_ROWS);
  hvx_int_wq *wq = NULL;
  if (!acc || !colsum || !w_scale || !bias || !act_scale || !act_zp || !stage ||
      !h_c || !h_v || !e_c || !e_v || !ah_c || !ah_v || !sc_c || !sc_v ||
      !zp_c || !zp_v) {
    goto out;
  }

  /* the data: per-channel scales with 3% outliers, a bias on odd seeds,
     accumulators of +-2^22 with outlier rows (x8) and columns (x4) */
  for (uint32_t c = 0; c < SC_N; ++c) {
    w_scale[c] = 0.001f + (float)(sc_lcg(&st) % 10000u) * 1e-6f;
    if (sc_lcg(&st) % 100u < 3u) {
      w_scale[c] *= 8.0f;
    }
    bias[c] = (seed & 1u)
                ? (float)((int32_t)(sc_lcg(&st) % 2001u) - 1000) * 1e-4f
                : 0.0f;
    colsum[c] = (int32_t)(sc_lcg(&st) % 20001u) - 10000;
  }
  for (uint32_t r = 0; r < SC_ROWS; ++r) {
    act_scale[r] = 0.01f + (float)(sc_lcg(&st) % 1000u) * 1e-4f;
    act_zp[r] = 80 + (int32_t)(sc_lcg(&st) % 100u);
  }
  for (uint32_t t = 0; t < SC_N / 32u; ++t) {
    for (uint32_t r = 0; r < SC_ROWS; ++r) {
      for (uint32_t l = 0; l < 32u; ++l) {
        const uint32_t c = t * 32u + l;
        int32_t a = (int32_t)(sc_lcg(&st) % (1u << 23)) - (1 << 22);
        if (r % 17u == 3u) {
          a <<= 3;
        }
        if (c % 97u == 5u) {
          a <<= 2;
        }
        acc[(size_t)t * SC_ROWS * 32u + (size_t)r * 32u + l] =
          a + act_zp[r] * colsum[c];
      }
    }
  }
  rc = hvx_int_wq_bake(&wq, w_scale, bias, SC_N);
  if (rc != AEE_SUCCESS) {
    goto out;
  }

  /* gate_up, batch by batch, both implementations on the same staging */
  memset(h_c, 0, sizeof(int32_t) * SC_ROWS * SC_INTER);
  memset(h_v, 0, sizeof(int32_t) * SC_ROWS * SC_INTER);
  memset(e_c, 0, sizeof(int16_t) * SC_ROWS * SC_ESTRIDE);
  memset(e_v, 0, sizeof(int16_t) * SC_ROWS * SC_ESTRIDE);
  uint32_t n_batches = 0;
  for (uint32_t g0 = 0, b = 0; g0 < SC_NT; g0 += SC_HALF, ++b) {
    const uint32_t np = (SC_NT - g0 < SC_HALF) ? SC_NT - g0 : SC_HALF;
    for (uint32_t j = 0; j < np; ++j) {
      memcpy(stage + (size_t)j * SC_ROWS * 32u,
             acc + (size_t)(g0 + j) * SC_ROWS * 32u,
             sizeof(int32_t) * SC_ROWS * 32u);
      memcpy(stage + (size_t)(np + j) * SC_ROWS * 32u,
             acc + (size_t)(SC_NT + g0 + j) * SC_ROWS * 32u,
             sizeof(int32_t) * SC_ROWS * 32u);
    }
    hvx_int_gu_job jb;
    memset(&jb, 0, sizeof jb);
    jb.tiles_base = (const uint8_t *)stage;
    jb.tile_stride = SC_ROWS * 32u * 4u;
    jb.n_pairs = np;
    jb.g0 = g0;
    jb.row_stride = 32u;
    jb.m_count = SC_VALID;
    jb.act_scale = act_scale;
    jb.act_zp = act_zp;
    jb.colsum_w = colsum;
    jb.wq = wq;
    jb.inter = SC_INTER;
    jb.dst_stride = SC_INTER;
    jb.e_stride = SC_ESTRIDE;
    jb.batch = b;
    jb.dst = h_c;
    jb.h_e = e_c;
    hvx_int_gu_worker_c(1u, 0u, &jb);
    jb.dst = h_v;
    jb.h_e = e_v;
    /* three slices, as the pool would run it */
    for (uint32_t t = 0; t < 3u; ++t) {
      hvx_int_gu_worker_hvx(3u, t, &jb);
    }
    n_batches = b + 1u;
  }
  uint32_t mism_h = 0, mism_e = 0, checksum = 0;
  for (uint32_t r = 0; r < SC_VALID; ++r) {
    for (uint32_t c = 0; c < SC_INTER; ++c) {
      const size_t k = (size_t)r * SC_INTER + c;
      mism_h += (h_c[k] != h_v[k]);
      checksum = checksum * 31u + (uint32_t)h_v[k];
    }
    for (uint32_t b = 0; b < n_batches; ++b) {
      mism_e += (e_c[r * SC_ESTRIDE + b] != e_v[r * SC_ESTRIDE + b]);
    }
  }

  /* requant, both, from the REFERENCE mantissas so the comparison is of
     the requant alone */
  memset(ah_c, 0xA5, (size_t)SC_ROWS * SC_INTER);
  memset(ah_v, 0x5A, (size_t)SC_ROWS * SC_INTER);
  hvx_int_rq_rows_c(h_c, SC_INTER, e_c, SC_ESTRIDE, n_batches, SC_HALF * 32u,
                    SC_INTER, SC_VALID, 0u, SC_ROWS, sc_c, zp_c, ah_c);
  for (uint32_t r0 = 0; r0 < SC_ROWS; r0 += 16u) {
    hvx_int_rq_rows_hvx(h_c, SC_INTER, e_c, SC_ESTRIDE, n_batches,
                        SC_HALF * 32u, SC_INTER, SC_VALID, r0, r0 + 16u, sc_v,
                        zp_v, ah_v);
  }
  uint32_t mism_b = 0, mism_s = 0, mism_z = 0;
  for (size_t k = 0; k < (size_t)SC_ROWS * SC_INTER; ++k) {
    mism_b += (ah_c[k] != ah_v[k]);
  }
  for (uint32_t r = 0; r < SC_ROWS; ++r) {
    uint32_t bc, bv;
    memcpy(&bc, &sc_c[r], 4);
    memcpy(&bv, &sc_v[r], 4);
    mism_s += (bc != bv);
    mism_z += (zp_c[r] != zp_v[r]);
  }
  counts[0] = mism_h;
  counts[1] = mism_e;
  counts[2] = mism_b;
  counts[3] = mism_s;
  counts[4] = mism_z;
  counts[5] = SC_VALID * SC_INTER;
  counts[6] = checksum;
  counts[7] = n_batches;
  if (mism_h | mism_e | mism_b | mism_s | mism_z) {
    FARF(ERROR,
         "int_epilogue_selfcheck seed=%u: h %u e %u bytes %u scale %u zp %u",
         (unsigned)seed, (unsigned)mism_h, (unsigned)mism_e, (unsigned)mism_b,
         (unsigned)mism_s, (unsigned)mism_z);
  }
  rc = AEE_SUCCESS;
out:
  hvx_int_wq_free(wq);
  free(acc);
  free(colsum);
  free(w_scale);
  free(bias);
  free(act_scale);
  free(act_zp);
  free(stage);
  free(h_c);
  free(h_v);
  free(e_c);
  free(e_v);
  free(ah_c);
  free(ah_v);
  free(sc_c);
  free(sc_v);
  free(zp_c);
  free(zp_v);
  return rc;
}
