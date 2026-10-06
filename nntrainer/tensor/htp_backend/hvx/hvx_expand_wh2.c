// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_expand_wh2.c
 * @date   06 Oct 2026
 * @brief  WH2 (2-bit codes) -> WH (int4 nibbles), one vector at a time
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Per source byte x (four codes): flip bit 1 of each code (x ^ 0xAA), which
 * turns code c into the low two bits of the int4 value c - 2; spread the
 * low two codes into the two nibbles of the even output byte and the high
 * two into the odd one; then copy each nibble's bit 1 (the sign of the
 * 2-bit value) into its bits 2 and 3. Every shift is a halfword shift whose
 * bits crossing a byte boundary are masked off, since HVX has no byte
 * shift. The two halves are byte-interleaved with vshuff.
 *
 * Device-checked against the scalar form (test/htp/host/hvx_scalar_stubs.c
 * and nntrainer::wh2FromWh) by unittest_hvx_mm_u8i4's ExpandWh2MatchesScalar.
 */

#include "hvx_expand_wh2.h"

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

/** @brief Two codes in bits 0-1 and 2-3 of each byte -> two sign-extended
 *         nibbles of that byte. */
static inline HVX_Vector spread2(HVX_Vector t, HVX_Vector m03, HVX_Vector m0c,
                                 HVX_Vector m22) {
  const HVX_Vector e =
    Q6_V_vor_VV(Q6_V_vand_VV(t, m03), Q6_Vh_vasl_VhR(Q6_V_vand_VV(t, m0c), 2));
  const HVX_Vector s = Q6_V_vand_VV(e, m22);
  return Q6_V_vor_VV(e,
                     Q6_V_vor_VV(Q6_Vh_vasl_VhR(s, 1), Q6_Vh_vasl_VhR(s, 2)));
}

void hvx_expand_wh2(uint8_t *dst, const uint8_t *src, uint32_t n_src) {
  const HVX_Vector flip = Q6_V_vsplat_R(0xAAAAAAAA);
  const HVX_Vector m0f = Q6_V_vsplat_R(0x0F0F0F0F);
  const HVX_Vector m03 = Q6_V_vsplat_R(0x03030303);
  const HVX_Vector m0c = Q6_V_vsplat_R(0x0C0C0C0C);
  const HVX_Vector m22 = Q6_V_vsplat_R(0x22222222);
  const HVX_Vector *in = (const HVX_Vector *)src;
  HVX_Vector *out = (HVX_Vector *)dst;
  for (uint32_t i = 0; i < n_src / 128u; ++i) {
    const HVX_Vector x = Q6_V_vxor_VV(in[i], flip);
    const HVX_Vector even = spread2(Q6_V_vand_VV(x, m0f), m03, m0c, m22);
    const HVX_Vector odd =
      spread2(Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(x, 4), m0f), m03, m0c, m22);
    /* Rt = -1: a full byte interleave, the second operand's byte first. */
    const HVX_VectorPair p = Q6_W_vshuff_VVR(odd, even, -1);
    out[2u * i] = Q6_V_lo_W(p);
    out[2u * i + 1u] = Q6_V_hi_W(p);
  }
}
