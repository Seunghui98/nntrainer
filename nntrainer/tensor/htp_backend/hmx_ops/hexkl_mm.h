// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hexkl_mm.h
 * @date   18 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  HMX matrix-multiply operations backed by HexKL (sdkl_npu_mm_*).
 *
 * These are the actual HexKL-calling routines. HtpComputeOps only
 * dispatches into them. Compiled only when ENABLE_HEXKL is defined.
 *
 * Phase 0: declarations + stub. The real sdkl_npu_mm_f32f16_f32 call,
 * layout (RM->WH) handling and sdkl_npu_alloc staging land in Phase 1.
 */

#ifndef __HEXKL_MM_H__
#define __HEXKL_MM_H__
#ifdef __cplusplus
#ifdef ENABLE_HEXKL
#ifdef ENABLE_FP16

#include <tensor_dim.h> // _FP16

namespace nntrainer {
namespace hmx {

/**
 * @brief F32 activations x F16 weights -> F32 GEMM on the NPU/HMX.
 *
 * Mirrors nntrainer::shgemm's parameter list so HtpComputeOps::shgemm
 * can forward 1:1. Internally maps onto sdkl_npu_mm_f32f16_f32 (computes
 * A = X * W^T) with the required RM->WH weight transform.
 *
 * @note Phase 0 stub — not yet implemented (throws). Phase 1 implements.
 */
void shgemm_f32f16_f32(const unsigned int TStorageOrder, bool TransA,
                       bool TransB, const unsigned int M, const unsigned int N,
                       const unsigned int K, const float alpha, const float *A,
                       const unsigned int lda, const _FP16 *B,
                       const unsigned int ldb, const float beta, float *C,
                       const unsigned int ldc);

/**
 * @brief Number of resident WH prefill-weight entries currently cached.
 * @note Introspection/test hook for the persistent prefill WH residency cache.
 */
size_t prefillWHCacheSize();

/**
 * @brief Free and drop all resident WH prefill-weight entries.
 * @note Test-isolation hook; also called implicitly at process exit.
 */
void prefillWHCacheClear();

/**
 * @brief Register pre-baked (already WH-layout) FP16 weight bytes for prefill.
 * @param rm_ptr  Host pointer to the RM weight tensor data (the lookup key —
 *                the same pointer shgemm_f32f16_f32 receives as B).
 * @param N,K     Weight dims (N=output cols, K=inner). Bytes =
 * N*K*sizeof(_FP16).
 * @param wh_src  Host pointer to N*K WH-layout FP16 values; copied internally.
 * @note Overwrites any existing entry for rm_ptr. Host-side only (no NPU
 * alloc).
 */
void registerPrefillWH(const void *rm_ptr, unsigned int N, unsigned int K,
                       const void *wh_src);

/**
 * @brief Look up pre-baked WH bytes for an RM weight pointer.
 * @return Host WH pointer if registered with matching N,K; else nullptr.
 * @note Never throws.
 */
const _FP16 *lookupPrefillWH(const void *rm_ptr, unsigned int N,
                             unsigned int K);

/** @brief Number of registered pre-baked WH weights (test hook). */
size_t prefillWHRegistrySize();

/** @brief Free and drop all registered pre-baked WH weights (test hook). */
void prefillWHRegistryClear();

} // namespace hmx
} // namespace nntrainer

#endif // ENABLE_FP16

// U8 activations x I8 weights (WH-layout) -> I32 accumulator -> FP32 output.
// N must be 32-aligned. M is padded internally to the SDKL 64-row tile.
namespace nntrainer {
namespace hmx {

void shgemm_u8i8_i32(unsigned int M, unsigned int N, unsigned int K,
                     const float *A, const int8_t *B_wh, const float *wt_scale,
                     const int32_t *zp_corr, float *C);

// U8 activations × I4 weights (WH-layout, packed two-per-byte) → I32
// accumulator → FP32 output. B_wh holds N*K/2 packed bytes in an int8 buffer.
// N must be 32-aligned. M is padded internally to the SDKL 64-row tile.
void shgemm_u8i4_i32(unsigned int M, unsigned int N, unsigned int K,
                     const float *A, const int8_t *B_wh, const float *wt_scale,
                     const int32_t *zp_corr, float *C);

} // namespace hmx
} // namespace nntrainer

#endif // ENABLE_HEXKL
#endif // __cplusplus
#endif // __HEXKL_MM_H__
