// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   nntr_hvx_rope.c
 * @brief  FastRPC entry point for the HVX uint8 RoPE kernel.
 */

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <math.h>
#include <remote.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "hvx_rope_u8.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

#define ROPE_QNN_TABLE_SCALE (1.0f / 127.0f)
#define ROPE_QNN_TABLE_ZP 128

static uint8_t rope_quant_qnn(float value) {
  int32_t q =
    (int32_t)nearbyintf(value / ROPE_QNN_TABLE_SCALE) + ROPE_QNN_TABLE_ZP;
  return (uint8_t)(q < 0 ? 0 : q > 255 ? 255 : q);
}

static int16_t rope_table_to_q15(uint8_t q) {
  int32_t v = (int32_t)nearbyintf(
    ((float)((int32_t)q - ROPE_QNN_TABLE_ZP) * ROPE_QNN_TABLE_SCALE) *
    32768.0f);
  return (int16_t)(v < -32768 ? -32768 : v > 32767 ? 32767 : v);
}

static void rope_cache_free(nntr_hvx_session *s) {
  free(s->rope_cos_u8);
  free(s->rope_sin_u8);
  free(s->rope_cos_q15);
  free(s->rope_sin_q15);
  s->rope_cos_u8 = NULL;
  s->rope_sin_u8 = NULL;
  s->rope_cos_q15 = NULL;
  s->rope_sin_q15 = NULL;
  s->rope_cache_positions = 0u;
  s->rope_cache_dim = 0u;
  s->rope_cache_theta = 0.0f;
}

int nntr_hvx_rope_cache_clear(remote_handle64 handle) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  rope_cache_free(s);
  return AEE_SUCCESS;
}

int nntr_hvx_rope_cache_init(remote_handle64 handle, uint32 n_positions,
                             uint32 dim, float theta, uint32 *generation) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || !generation || n_positions == 0u || n_positions > 4096u ||
      dim == 0u || (dim & 1u) != 0u || theta <= 0.0f || dim > UINT32_MAX / 2u ||
      n_positions > UINT32_MAX / (dim / 2u)) {
    return AEE_EBADPARM;
  }
  if (s->rope_cos_u8 && s->rope_cache_positions == n_positions &&
      s->rope_cache_dim == dim && s->rope_cache_theta == theta) {
    *generation = s->rope_cache_generation;
    return AEE_SUCCESS;
  }

  const uint32 half = dim / 2u;
  const size_t count = (size_t)n_positions * half;
  uint8_t *cos_u8 = (uint8_t *)malloc(count);
  uint8_t *sin_u8 = (uint8_t *)malloc(count);
  int16_t *cos_q15 = (int16_t *)malloc(count * sizeof(int16_t));
  int16_t *sin_q15 = (int16_t *)malloc(count * sizeof(int16_t));
  if (!cos_u8 || !sin_u8 || !cos_q15 || !sin_q15) {
    free(cos_u8);
    free(sin_u8);
    free(cos_q15);
    free(sin_q15);
    return AEE_ENOMEMORY;
  }

  for (uint32 pos = 0u; pos < n_positions; ++pos) {
    for (uint32 k = 0u; k < half; ++k) {
      const float inv_freq = powf(theta, -2.0f * (float)k / (float)dim);
      const float angle = (float)pos * inv_freq;
      const size_t i = (size_t)pos * half + k;
      cos_u8[i] = rope_quant_qnn(cosf(angle));
      sin_u8[i] = rope_quant_qnn(sinf(angle));
      cos_q15[i] = rope_table_to_q15(cos_u8[i]);
      sin_q15[i] = rope_table_to_q15(sin_u8[i]);
    }
  }
  rope_cache_free(s);
  s->rope_cos_u8 = cos_u8;
  s->rope_sin_u8 = sin_u8;
  s->rope_cos_q15 = cos_q15;
  s->rope_sin_q15 = sin_q15;
  s->rope_cache_positions = n_positions;
  s->rope_cache_dim = dim;
  s->rope_cache_theta = theta;
  ++s->rope_cache_generation;
  if (s->rope_cache_generation == 0u) {
    s->rope_cache_generation = 1u;
  }
  *generation = s->rope_cache_generation;
  return AEE_SUCCESS;
}

int nntr_hvx_rope_u8(remote_handle64 handle, uint32 n_rows, uint32 width,
                     uint32 dim, const uint8 *x, int xLen, const float *s_in,
                     int s_inLen, const int32 *zp_in, int zp_inLen,
                     const int16 *cos_q15, int cos_q15Len, const int16 *sin_q15,
                     int sin_q15Len, float s_out, int32 zp_out, uint8 *y,
                     int yLen, uint32 *n_saturated) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  if (n_rows == 0u || n_rows > 4096u || width == 0u || dim == 0u ||
      (dim & 1u) != 0u || dim > width || s_out <= 0.0f || n_saturated == NULL ||
      width > UINT32_MAX / n_rows) {
    FARF(ERROR, "rope_u8: bad shape or output scale");
    return AEE_EBADPARM;
  }
  const uint32 expected_x = n_rows * width;
  const uint32 half = dim / 2u;
  if (half > UINT32_MAX / n_rows) {
    FARF(ERROR, "rope_u8: table length overflow");
    return AEE_EBADPARM;
  }
  const uint32 expected_table = n_rows * half;
  if ((uint32)xLen != expected_x || (uint32)yLen != expected_x ||
      ((uint32)s_inLen != 1u && (uint32)s_inLen != n_rows) ||
      ((uint32)zp_inLen != 1u && (uint32)zp_inLen != n_rows) ||
      (uint32)cos_q15Len != expected_table ||
      (uint32)sin_q15Len != expected_table) {
    FARF(ERROR, "rope_u8: bad buffer lengths");
    return AEE_EBADPARM;
  }

  /* The kernel always consumes per-row arrays; expand length-one broadcasts. */
  float row_scale[n_rows];
  int32 row_zp[n_rows];
  for (uint32 m = 0; m < n_rows; ++m) {
    row_scale[m] = s_in[s_inLen == 1 ? 0 : m];
    row_zp[m] = zp_in[zp_inLen == 1 ? 0 : m];
  }
  *n_saturated = 0u;
  if (x != y) {
    memcpy(y, x, (size_t)expected_x);
  }
  hvx_rope_u8_rows(x, y, 0u, n_rows, width, dim, row_scale, row_zp, cos_q15,
                   sin_q15, s_out, zp_out, n_saturated);
  return AEE_SUCCESS;
}

int nntr_hvx_rope_u8_cached(remote_handle64 handle, uint32 n_rows, uint32 width,
                            uint32 dim, uint32 position_start, const uint8 *x,
                            int xLen, const float *s_in, int s_inLen,
                            const int32 *zp_in, int zp_inLen, float s_out,
                            int32 zp_out, uint8 *y, int yLen,
                            uint32 *n_saturated) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || !s->rope_cos_q15 || n_rows == 0u || width == 0u || dim == 0u ||
      (dim & 1u) != 0u || dim > width || s_out <= 0.0f ||
      position_start > s->rope_cache_positions ||
      n_rows > s->rope_cache_positions - position_start) {
    return AEE_EBADPARM;
  }
  if (width > UINT32_MAX / n_rows || s->rope_cache_dim != dim || !n_saturated) {
    return AEE_EBADPARM;
  }
  const uint32 expected = n_rows * width;
  const uint32 half = dim / 2u;
  if ((uint32)xLen != expected || (uint32)yLen != expected ||
      ((uint32)s_inLen != 1u && (uint32)s_inLen != n_rows) ||
      ((uint32)zp_inLen != 1u && (uint32)zp_inLen != n_rows)) {
    return AEE_EBADPARM;
  }
  float row_scale[n_rows];
  int32 row_zp[n_rows];
  for (uint32 m = 0u; m < n_rows; ++m) {
    row_scale[m] = s_in[s_inLen == 1 ? 0 : m];
    row_zp[m] = zp_in[zp_inLen == 1 ? 0 : m];
  }
  *n_saturated = 0u;
  if (x != y) {
    memcpy(y, x, (size_t)expected);
  }
  hvx_rope_u8_rows(x, y, 0u, n_rows, width, dim, row_scale, row_zp,
                   s->rope_cos_q15 + (size_t)position_start * half,
                   s->rope_sin_q15 + (size_t)position_start * half, s_out,
                   zp_out, n_saturated);
  return AEE_SUCCESS;
}
