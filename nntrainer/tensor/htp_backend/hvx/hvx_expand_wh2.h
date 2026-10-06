// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_expand_wh2.h
 * @date   06 Oct 2026
 * @brief  WH2 (2-bit codes) -> WH (int4 nibbles), in VTCM, before the HMX
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The format is nntrainer::wh2Bytes's (htp_wh_layout.h): source byte j
 * becomes destination bytes 2j and 2j+1, code c becoming the nibble of the
 * int4 value c - 2. Byte-wise, so any run of whole bytes expands on its own
 * and the MoE kernel can split a weight into units however it likes.
 */

#ifndef __NNTRAINER_HVX_EXPAND_WH2_H__
#define __NNTRAINER_HVX_EXPAND_WH2_H__

#include <stdint.h>

/**
 * @brief Expands @a n_src WH2 bytes at @a src into 2 * @a n_src WH bytes at
 *        @a dst.
 * @param src  128-byte aligned
 * @param dst  128-byte aligned, not overlapping src
 * @param n_src a multiple of 128 (a WH2 tile is 256 bytes)
 */
void hvx_expand_wh2(uint8_t *dst, const uint8_t *src, uint32_t n_src);

#endif /* __NNTRAINER_HVX_EXPAND_WH2_H__ */
