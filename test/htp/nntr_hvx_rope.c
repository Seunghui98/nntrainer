// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   nntr_hvx_rope.c
 * @brief  FastRPC entry point for the HVX uint8 RoPE kernel.
 */

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <remote.h>
#include <stdint.h>
#include <string.h>

#include "hvx_rope_u8.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

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
