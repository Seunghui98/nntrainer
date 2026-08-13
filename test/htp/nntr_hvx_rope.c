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

static int16_t rope_real_to_q15(float value) {
  int32_t v = (int32_t)nearbyintf(value * 32768.0f);
  return (int16_t)(v < -32768 ? -32768 : v > 32767 ? 32767 : v);
}

static void rope_cache_free(nntr_hvx_session *s) {
  free(s->rope_cos_q15);
  free(s->rope_sin_q15);
  free(s->rope_thetas);
  s->rope_cos_q15 = NULL;
  s->rope_sin_q15 = NULL;
  s->rope_thetas = NULL;
  s->rope_cache_positions = 0u;
  s->rope_cache_half = 0u;
  s->rope_cache_attention_scaling = 0.0f;
  s->rope_cache_table_scale = 0.0f;
  s->rope_cache_table_zp = 0;
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
                             const float *thetas, int thetasLen,
                             float attention_scaling, float table_scale,
                             int32 table_zp, uint32 *generation) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || !generation || n_positions == 0u || n_positions > 4096u ||
      thetasLen <= 0 || (uint32)thetasLen > UINT32_MAX / n_positions) {
    FARF(ERROR, "rope_cache_init: bad shape");
    return AEE_EBADPARM;
  }
  const uint32 half = (uint32)thetasLen;

  if (s->rope_cos_q15 && s->rope_cache_positions == n_positions &&
      s->rope_cache_half == half &&
      s->rope_cache_attention_scaling == attention_scaling &&
      s->rope_cache_table_scale == table_scale &&
      s->rope_cache_table_zp == table_zp &&
      memcmp(s->rope_thetas, thetas, (size_t)half * sizeof(float)) == 0) {
    *generation = s->rope_cache_generation;
    return AEE_SUCCESS;
  }

  const size_t count = (size_t)n_positions * half;
  int16_t *cos_q15 = (int16_t *)malloc(count * sizeof(int16_t));
  int16_t *sin_q15 = (int16_t *)malloc(count * sizeof(int16_t));
  float *thetas_copy = (float *)malloc((size_t)half * sizeof(float));
  if (!cos_q15 || !sin_q15 || !thetas_copy) {
    free(cos_q15);
    free(sin_q15);
    free(thetas_copy);
    return AEE_ENOMEMORY;
  }
  memcpy(thetas_copy, thetas, (size_t)half * sizeof(float));

  /* thetas already carries every rope scaling type (default/yarn/
     proportional) and the partial rotary factor -- see the IDL comment on
     rope_cache_init. This loop only evaluates cos/sin, the one thing that
     genuinely differs per position and cannot be precomputed on ARM
     without shipping the whole table. */
  for (uint32 pos = 0u; pos < n_positions; ++pos) {
    for (uint32 k = 0u; k < half; ++k) {
      const float angle = (float)pos * thetas[k];
      const size_t i = (size_t)pos * half + k;
      cos_q15[i] = rope_real_to_q15(cosf(angle) * attention_scaling);
      sin_q15[i] = rope_real_to_q15(sinf(angle) * attention_scaling);
    }
  }
  rope_cache_free(s);
  s->rope_cos_q15 = cos_q15;
  s->rope_sin_q15 = sin_q15;
  s->rope_thetas = thetas_copy;
  s->rope_cache_positions = n_positions;
  s->rope_cache_half = half;
  s->rope_cache_attention_scaling = attention_scaling;
  s->rope_cache_table_scale = table_scale;
  s->rope_cache_table_zp = table_zp;
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

  *n_saturated = 0u;
  hvx_rope_u8_rows(x, y, 0u, n_rows, width, dim, s_in,
                   (uint32)s_inLen == 1u ? 0u : 1u, zp_in,
                   (uint32)zp_inLen == 1u ? 0u : 1u, cos_q15, sin_q15, s_out,
                   zp_out, n_saturated);
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
  const uint32 half = dim / 2u;
  if (width > UINT32_MAX / n_rows || s->rope_cache_half != half ||
      !n_saturated) {
    return AEE_EBADPARM;
  }
  const uint32 expected = n_rows * width;
  if ((uint32)xLen != expected || (uint32)yLen != expected ||
      ((uint32)s_inLen != 1u && (uint32)s_inLen != n_rows) ||
      ((uint32)zp_inLen != 1u && (uint32)zp_inLen != n_rows)) {
    return AEE_EBADPARM;
  }
  *n_saturated = 0u;
  hvx_rope_u8_rows(x, y, 0u, n_rows, width, dim, s_in,
                   (uint32)s_inLen == 1u ? 0u : 1u, zp_in,
                   (uint32)zp_inLen == 1u ? 0u : 1u,
                   s->rope_cos_q15 + (size_t)position_start * half,
                   s->rope_sin_q15 + (size_t)position_start * half, s_out,
                   zp_out, n_saturated);
  return AEE_SUCCESS;
}
