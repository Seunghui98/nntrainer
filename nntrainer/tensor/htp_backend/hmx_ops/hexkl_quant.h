// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_quant.h
 * @date   27 Jul 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Host-side quantize/dequantize around the HexKL integer matmuls.
 *
 * The activation scan, the FP32->U8 quantization and the I32->FP32 dequantize
 * are element-wise over M*K and M*N. On a Galaxy S25 Ultra they are ~197 us of
 * the 545 us an fc_layer call takes (q_proj, M=64, u8i4) -- comparable to the
 * NPU matmul itself, and until now plain scalar loops.
 *
 * Each routine has a scalar reference and an AArch64 NEON version that must
 * agree with it **exactly**, not approximately:
 *
 *   - the max-abs reduction reorders the max, which is exact for non-NaN;
 *   - quantize uses FRINTA (`vrndaq_f32`), which is round-half-away-from-zero,
 *     the same rule as std::round -- FRINTN would round halves to even and
 *     disagree on exact .5;
 *   - dequantize keeps the scalar's left-to-right grouping,
 *     `(act_scale * wt_scale[n]) * (c - zp)`, so the rounding matches.
 *
 * That is what makes `_scalar` vs dispatched testable by bit equality rather
 * than by tolerance -- see unittest_nntrainer_htp_kernel_math.cpp.
 *
 * Header-only and free of any SDK dependency on purpose, so the tests can
 * include it on a host with no Hexagon SDK.
 */

#ifndef __HEXKL_QUANT_H__
#define __HEXKL_QUANT_H__
#ifdef __cplusplus

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define HEXKL_QUANT_NEON 1
#endif

namespace nntrainer {
namespace hmx {
namespace quant {

/** @brief Zero point of the U8 activation quantization. */
constexpr float kActZeroPoint = 128.0f;
/** @brief Largest magnitude an activation maps to either side of the zp. */
constexpr float kActQuantMax = 127.0f;

/**
 * @brief Per-tensor activation scale, max_abs / 127, guarded.
 *
 * Lives here rather than in the kernel so a test can reproduce a kernel run
 * bit-for-bit. The division is done in double and narrowed once: computing it
 * as `max_abs / 127.0f` instead can land a ULP away, which is enough to shift
 * a quantized value by one and break an exact comparison for reasons that have
 * nothing to do with the code under test.
 *
 * Falls back to 1 when the result is not finite or would denormalize, so the
 * reciprocal the quantizer takes stays usable.
 */
inline float activationScale(float max_abs) {
  const double candidate =
    static_cast<double>(max_abs) / static_cast<double>(kActQuantMax);
  return (std::isfinite(candidate) &&
          candidate >= static_cast<double>(std::numeric_limits<float>::min()))
           ? static_cast<float>(candidate)
           : 1.0f;
}

/** @brief Largest |A[i]| over n, skipping non-finite values. Scalar reference.
 */
inline float activationMaxAbsScalar(const float *A, size_t n) {
  float max_abs = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    if (!std::isfinite(A[i]))
      continue;
    const float v = std::fabs(A[i]);
    if (v > max_abs)
      max_abs = v;
  }
  return max_abs;
}

/**
 * @brief Quantize FP32 activations to U8. Scalar reference.
 * @param Mp Row count padded to the SDKL 64-row tile; rows at or past M are
 *           filled with the zero point so the padding contributes nothing.
 */
inline void quantizeActivationU8Scalar(const float *A, uint8_t *X, size_t M,
                                       size_t Mp, size_t K, float inv_act_scale,
                                       float zero_point) {
  for (size_t m = 0; m < Mp; ++m) {
    const size_t off = m * K;
    if (m >= M) {
      std::memset(X + off, static_cast<int>(zero_point), K);
      continue;
    }
    for (size_t k = 0; k < K; ++k) {
      const float a = A[off + k];
      if (!std::isfinite(a)) {
        X[off + k] = static_cast<uint8_t>(zero_point);
        continue;
      }
      float q = std::round(a * inv_act_scale) + zero_point;
      if (q < 0.0f)
        q = 0.0f;
      if (q > 255.0f)
        q = 255.0f;
      X[off + k] = static_cast<uint8_t>(q);
    }
  }
}

/** @brief C[m,n] = act_scale * wt_scale[n] * (C_i32[m,n] - zp_corr[n]). */
inline void dequantizeToF32Scalar(const int32_t *C_i32, float *C, size_t M,
                                  size_t N, float act_scale,
                                  const float *wt_scale,
                                  const int32_t *zp_corr) {
  for (size_t m = 0; m < M; ++m) {
    const size_t off = m * N;
    for (size_t n = 0; n < N; ++n)
      C[off + n] = act_scale * wt_scale[n] *
                   (static_cast<float>(C_i32[off + n]) -
                    static_cast<float>(zp_corr[n]));
  }
}

#if defined(HEXKL_QUANT_NEON)

/**
 * @brief NEON max-abs. Four accumulators so the max chain does not serialize.
 *
 * Non-finite values are dropped by comparing |v| < +inf: infinities fail it,
 * and NaN fails every ordered compare, so both fall out without a branch.
 */
inline float activationMaxAbsNeon(const float *A, size_t n) {
  const float32x4_t inf = vdupq_n_f32(std::numeric_limits<float>::infinity());
  const float32x4_t zero = vdupq_n_f32(0.0f);
  float32x4_t m0 = zero, m1 = zero, m2 = zero, m3 = zero;

  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    float32x4_t v0 = vabsq_f32(vld1q_f32(A + i));
    float32x4_t v1 = vabsq_f32(vld1q_f32(A + i + 4));
    float32x4_t v2 = vabsq_f32(vld1q_f32(A + i + 8));
    float32x4_t v3 = vabsq_f32(vld1q_f32(A + i + 12));
    m0 = vmaxq_f32(m0, vbslq_f32(vcltq_f32(v0, inf), v0, zero));
    m1 = vmaxq_f32(m1, vbslq_f32(vcltq_f32(v1, inf), v1, zero));
    m2 = vmaxq_f32(m2, vbslq_f32(vcltq_f32(v2, inf), v2, zero));
    m3 = vmaxq_f32(m3, vbslq_f32(vcltq_f32(v3, inf), v3, zero));
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t v = vabsq_f32(vld1q_f32(A + i));
    m0 = vmaxq_f32(m0, vbslq_f32(vcltq_f32(v, inf), v, zero));
  }

  float max_abs =
    vmaxvq_f32(vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2, m3)));
  for (; i < n; ++i) {
    if (!std::isfinite(A[i]))
      continue;
    const float v = std::fabs(A[i]);
    if (v > max_abs)
      max_abs = v;
  }
  return max_abs;
}

/**
 * @brief NEON FP32 -> U8 quantize, 16 lanes per iteration.
 *
 * vrndaq_f32 is FRINTA (ties away from zero), matching std::round. The
 * saturating narrows are not what clamps the result -- the explicit
 * max/min against [0, 255] does that first, so a large input saturates the
 * same way the scalar version's two compares do.
 */
inline void quantizeActivationU8Neon(const float *A, uint8_t *X, size_t M,
                                     size_t Mp, size_t K, float inv_act_scale,
                                     float zero_point) {
  const float32x4_t inf = vdupq_n_f32(std::numeric_limits<float>::infinity());
  const float32x4_t vzp = vdupq_n_f32(zero_point);
  const float32x4_t vlo = vdupq_n_f32(0.0f);
  const float32x4_t vhi = vdupq_n_f32(255.0f);
  const float32x4_t vinv = vdupq_n_f32(inv_act_scale);

  for (size_t m = 0; m < Mp; ++m) {
    const size_t off = m * K;
    if (m >= M) {
      std::memset(X + off, static_cast<int>(zero_point), K);
      continue;
    }

    size_t k = 0;
    for (; k + 16 <= K; k += 16) {
      uint8x16_t out;
      uint16x8_t half[2];
      for (int h = 0; h < 2; ++h) {
        uint32x4_t q32[2];
        for (int q = 0; q < 2; ++q) {
          const float32x4_t a = vld1q_f32(A + off + k + h * 8 + q * 4);
          // Non-finite -> zero point, matching the scalar branch.
          const uint32x4_t fin = vcltq_f32(vabsq_f32(a), inf);
          float32x4_t v = vrndaq_f32(vmulq_f32(a, vinv));
          v = vaddq_f32(v, vzp);
          v = vminq_f32(vmaxq_f32(v, vlo), vhi);
          v = vbslq_f32(fin, v, vzp);
          q32[q] = vcvtq_u32_f32(v);
        }
        half[h] = vcombine_u16(vmovn_u32(q32[0]), vmovn_u32(q32[1]));
      }
      out = vcombine_u8(vmovn_u16(half[0]), vmovn_u16(half[1]));
      vst1q_u8(X + off + k, out);
    }

    for (; k < K; ++k) {
      const float a = A[off + k];
      if (!std::isfinite(a)) {
        X[off + k] = static_cast<uint8_t>(zero_point);
        continue;
      }
      float q = std::round(a * inv_act_scale) + zero_point;
      if (q < 0.0f)
        q = 0.0f;
      if (q > 255.0f)
        q = 255.0f;
      X[off + k] = static_cast<uint8_t>(q);
    }
  }
}

/**
 * @brief NEON I32 -> FP32 dequantize.
 *
 * Grouped as (act_scale * wt_scale[n]) * (c - zp) to match the scalar's
 * left-to-right evaluation exactly; regrouping would change the rounding.
 */
inline void dequantizeToF32Neon(const int32_t *C_i32, float *C, size_t M,
                                size_t N, float act_scale,
                                const float *wt_scale, const int32_t *zp_corr) {
  for (size_t m = 0; m < M; ++m) {
    const size_t off = m * N;
    size_t n = 0;
    for (; n + 8 <= N; n += 8) {
      for (int h = 0; h < 2; ++h) {
        const size_t o = off + n + h * 4;
        const float32x4_t c = vcvtq_f32_s32(vld1q_s32(C_i32 + o));
        const float32x4_t z = vcvtq_f32_s32(vld1q_s32(zp_corr + n + h * 4));
        const float32x4_t s =
          vmulq_n_f32(vld1q_f32(wt_scale + n + h * 4), act_scale);
        vst1q_f32(C + o, vmulq_f32(s, vsubq_f32(c, z)));
      }
    }
    for (; n < N; ++n)
      C[off + n] = act_scale * wt_scale[n] *
                   (static_cast<float>(C_i32[off + n]) -
                    static_cast<float>(zp_corr[n]));
  }
}

#endif // HEXKL_QUANT_NEON

// ---- Dispatch --------------------------------------------------------------
// NEON where the target has it, the scalar reference everywhere else. Both are
// kept compiled so the tests can assert they agree.

/**
 * @brief Runtime override: NNTR_HTP_NO_SIMD=1 forces the scalar path.
 *
 * Exists so the NEON gain can be measured as an A/B on one binary, in one
 * session, on one device. Comparing against a number from an earlier build is
 * how a mismatched executable/library pair got mistaken for a 264 us result;
 * a switch removes every variable except the one under test.
 *
 * Read once into a function-local static -- thread-safe since C++11, and a
 * plain load after the first call, so the hot path pays nothing measurable.
 */
inline bool simdDisabledByEnv() {
  static const bool off = [] {
    const char *e = std::getenv("NNTR_HTP_NO_SIMD");
    return e != nullptr && e[0] == '1' && e[1] == '\0';
  }();
  return off;
}

/** @brief Largest |A[i]| over n, skipping non-finite values. */
inline float activationMaxAbs(const float *A, size_t n) {
#if defined(HEXKL_QUANT_NEON)
  if (!simdDisabledByEnv())
    return activationMaxAbsNeon(A, n);
#endif
  return activationMaxAbsScalar(A, n);
}

/** @brief Quantize FP32 activations to U8, padding rows [M, Mp) with the zp. */
inline void quantizeActivationU8(const float *A, uint8_t *X, size_t M,
                                 size_t Mp, size_t K, float inv_act_scale,
                                 float zero_point) {
#if defined(HEXKL_QUANT_NEON)
  if (!simdDisabledByEnv()) {
    quantizeActivationU8Neon(A, X, M, Mp, K, inv_act_scale, zero_point);
    return;
  }
#endif
  quantizeActivationU8Scalar(A, X, M, Mp, K, inv_act_scale, zero_point);
}

/** @brief C[m,n] = act_scale * wt_scale[n] * (C_i32[m,n] - zp_corr[n]). */
inline void dequantizeToF32(const int32_t *C_i32, float *C, size_t M, size_t N,
                            float act_scale, const float *wt_scale,
                            const int32_t *zp_corr) {
#if defined(HEXKL_QUANT_NEON)
  if (!simdDisabledByEnv()) {
    dequantizeToF32Neon(C_i32, C, M, N, act_scale, wt_scale, zp_corr);
    return;
  }
#endif
  dequantizeToF32Scalar(C_i32, C, M, N, act_scale, wt_scale, zp_corr);
}

/**
 * @brief True when the dispatched routines above really take the NEON path.
 *
 * Reflects the runtime override too, so a profile that says "NEON" was not
 * merely compiled with it available.
 */
inline bool neonEnabled() {
#if defined(HEXKL_QUANT_NEON)
  return !simdDisabledByEnv();
#else
  return false;
#endif
}

} // namespace quant
} // namespace hmx
} // namespace nntrainer

#endif // __cplusplus
#endif // __HEXKL_QUANT_H__
