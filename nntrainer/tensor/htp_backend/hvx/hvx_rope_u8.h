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
 * cos_q15 and sin_q15 contain [n_rows][dim / 2] Q15 values. The input
 * quantization is per row; s_out and zp_out are common to the output.
 * y may alias x.
 */
void hvx_rope_u8_rows(const uint8_t *x, uint8_t *y, uint32_t m_first,
                      uint32_t m_last, uint32_t width, uint32_t dim,
                      const float *s_in, const int32_t *zp_in,
                      const int16_t *cos_q15, const int16_t *sin_q15,
                      float s_out, int32_t zp_out, uint32_t *n_saturated);

#endif /* __NNTRAINER_HVX_ROPE_U8_H__ */
