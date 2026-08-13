// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_rope_u8.h
 * @brief  Row-major uint8-in/uint8-out NeoX RoPE kernel.
 */

#ifndef __NNTRAINER_HVX_ROPE_U8_H__
#define __NNTRAINER_HVX_ROPE_U8_H__

#include <stdint.h>

/**
 * @brief Applies NeoX half-rotation and requantizes each row to uint8.
 *
 * cos_q15 and sin_q15 contain [n_rows][dim / 2] Q15 values, indexed by row
 * m (not m - m_first). s_in/zp_in are read as s_in[m * s_in_stride] and
 * zp_in[m * zp_in_stride] -- pass stride 0 for a single value broadcast to
 * every row, or 1 for one entry per row. This is a stride rather than a
 * caller-expanded per-row array so the caller never has to allocate
 * n_rows-sized scratch for a broadcast input (doc 43 §2.2). y may alias x.
 *
 * Channels past the last whole `dim` segment (when width % dim != 0) are
 * copied unrotated, but only when y != x -- with y == x they are already
 * in place. width % dim == 0 in every real caller (width = n_heads *
 * head_dim); this exists for the general contract, not for partial
 * rotary, which the caller handles by zeroing the corresponding thetas
 * before quantizing them into cos_q15/sin_q15 (40_rope_u8_task.md §1.4).
 *
 * ponytail: the u8 widen and the u8 pack around the vector core are scalar
 * loops -- roughly 600 scalar ops per 64 outputs against the 14-instruction
 * vector core, two orders of magnitude off ref_16 §3.3's 0.07 cy/elem
 * elementwise baseline. Accuracy is fixed first on purpose. Upgrade path:
 * Q6_Wuh_vunpack_Vub in, Q6_Vh_vpack_VwVw_sat -> Q6_Vub_vpack_VhVh_sat out
 * (the chain hvx_quant_u8.c:211-213 already uses), and a vector compare
 * against [0,255] for the saturation count instead of the scalar one.
 */
void hvx_rope_u8_rows(const uint8_t *x, uint8_t *y, uint32_t m_first,
                      uint32_t m_last, uint32_t width, uint32_t dim,
                      const float *s_in, uint32_t s_in_stride,
                      const int32_t *zp_in, uint32_t zp_in_stride,
                      const int16_t *cos_q15, const int16_t *sin_q15,
                      float s_out, int32_t zp_out, uint32_t *n_saturated);

#endif /* __NNTRAINER_HVX_ROPE_U8_H__ */
