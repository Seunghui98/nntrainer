// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_micro_mha_bench.c
 * @brief  On-DSP fp16 HMX matmul bench for the two MHA-core GEMMs (Q.K^T and
 *         P.V), per doc13 Sec5 step 1. Runs standalone on the Hexagon DSP, no
 *         FastRPC, no nntrainer -- mirrors hexkl_micro_fc_bench.c's structure
 *         but at fp16's 32-row tile geometry (u8i4 is 64-row) and against a
 *         different shape family (MHA decode/prefill GEMMs, not FC).
 *
 * CURRENT SCOPE (all of the below run, and are measured, on every plain
 * invocation with no env overrides -- see run_micro_mha_bench.sh for the
 * consolidated report):
 *   - qkT_dec_1024 (decode Q.K^T, spec #1): macro full row-major path, micro
 *     naive scalar staging, micro+DMA per-tile (old geometry, kept as an
 *     explicit comparison column), AND the optimized micro+layer-wide-DMA
 *     path (layer-wide contiguous DMA + VTCM repack + pre-baked WH +
 *     n_tiles=ceil(gqa/32)) -- the last one is the headline number.
 *   - pv_dec_1024, pv_dec_512 (decode P.V, spec #4/#5): optimized path only,
 *     plus the fp16-accumulator accuracy report (spec 5a) and the
 *     incremental-vs-full weight bake comparison (spec 5b).
 *   - qk_pre_128, pv_pre_128 (prefill, spec #6/#7): optimized path, with the
 *     mandatory per-chunk incremental bake included in the reported total
 *     (prefill has no true steady state -- every chunk's KV rows are newly
 *     appended and must be baked; see prebake_qk_weight_transposed /
 *     bench_incremental_bake_qk and their P.V counterparts).
 *   - Spec shapes #2 (qk_dec_1024) and #3 (qkT_dec_512) are NOT implemented
 *     -- see the comment on MHA_SHAPES[] below for why raising
 *     MHA_SHAPES_RUN does not get you there.
 *   - Kernel 4 (cross-head prefetch via the dmlink ring) was never built as
 *     its own measurement; the ring it would have used is already the one
 *     every optimized path above runs on.
 * bench_dma_bw_ceiling and bench_output_narrow_vs_full are diagnostic
 * sweeps (M2-f/M2-j), not part of any shape's headline number -- see their
 * own comments and the runner script's footer for how to skip them.
 *
 * Shape mapping (Applications/CausalLM/layers/mha_core.cpp, read but NOT
 * modified): for a fixed kv_head n, the K cache is `[kv, hd]` with per-row
 * stride `nch*hd*sizeof(fp16)` bytes (the GQA stride -- nch heads share one
 * cache row group). Q for decode is `[gqa, hd]` contiguous. S^T = K.Q^T puts
 * the GQA stride on the ACTIVATION (K cache) side and a small `hd x gqa`
 * transpose of Q on the WEIGHT side. nch and gqa are fixed compile-time
 * constants below (8 and 8) representative of the real config; the point of
 * this bench is the DMA/layout mechanics, not a specific model's numbers.
 */

#include "AEEStdErr.h"
#include "qurt_memory.h"
#include "remote.h"
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hmx_hexagon_protos.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hexkl_macro.h"
#include "hexkl_micro.h"
#include "hexkl_test.h"

/*
 *   default            hexagon_sim_timer.h  (simulator)
 *   -DUSE_HAP_PERF     HAP_perf.h           (on-target, FastRPC present)
 */
#if defined(USE_HAP_PERF)
#include "HAP_perf.h"
#define READ_PCYCLES() HAP_perf_get_time_us()
#else
#include "hexagon_sim_timer.h"
#define READ_PCYCLES() hexagon_sim_read_pcycles()
#endif

/* fp16 tile geometry (do NOT reuse the u8i4 WT_TILE_BYTES/ACC_TILE_BYTES --
 * fp16 is 32-row tiles, 2048-byte weight AND accumulator tiles). */
#define WT_TILE_BYTES_F16  (2048U)
#define ACC_TILE_BYTES_F16 (2048U)

/* Representative MHA config for the synthetic activation/weight sources
 * (Sec2 of the spec). Does not need to match a specific model -- only the
 * stride/transpose mechanics matter for this bench. */
#define MHA_NCH (8U) /* num_cache_head: KV heads sharing a cache row group */
#define MHA_GQA (8U) /* gqa_size: query heads per kv head (valid N columns) */

static uint32_t align_up(uint32_t v, uint32_t a) {
  return (v + a - 1u) / a * a;
}

/*!
  @brief max abs err, max rel err, mean rel err of a fp16 result against an
  fp32 reference. Always call this OUTSIDE any timed window -- a correctness
  check inside a timed loop faked a 146us "win" in a previous session.
*/
typedef struct {
  double max_abs;
  double max_rel;
  double mean_rel;
  long max_rel_idx;
} err_stats;

static err_stats err_stats_f16(uint32_t count, const float *ref, const _Float16 *got) {
  err_stats e = {0, 0, 0, -1};
  double sum_rel = 0;
  for (uint32_t i = 0; i < count; i++) {
    double g = (double)(float)got[i];
    double r = (double)ref[i];
    double diff = fabs(g - r);
    double denom = fabs(r) > 1e-6 ? fabs(r) : 1e-6;
    double rel = diff / denom;
    if (diff > e.max_abs)
      e.max_abs = diff;
    if (rel > e.max_rel) {
      e.max_rel = rel;
      e.max_rel_idx = (long)i;
    }
    sum_rel += rel;
  }
  e.mean_rel = count ? sum_rel / count : 0.0;
  return e;
}

static err_stats err_stats_f32(uint32_t count, const float *ref, const float *got) {
  err_stats e = {0, 0, 0, -1};
  double sum_rel = 0;
  for (uint32_t i = 0; i < count; i++) {
    double g = (double)got[i];
    double r = (double)ref[i];
    double diff = fabs(g - r);
    double denom = fabs(r) > 1e-6 ? fabs(r) : 1e-6;
    double rel = diff / denom;
    if (diff > e.max_abs)
      e.max_abs = diff;
    if (rel > e.max_rel) {
      e.max_rel = rel;
      e.max_rel_idx = (long)i;
    }
    sum_rel += rel;
  }
  e.mean_rel = count ? sum_rel / count : 0.0;
  return e;
}

/* =====================================================================
 * shapes (spec Sec3), for reference -- this array only feeds shape #1
 * through run_macro_shape_f16/run_micro_shape_f16's generic loop.
 *
 * DO NOT raise MHA_SHAPES_RUN to "unlock" #2/#3: those functions hardcode
 * the S^T=K.Q^T orientation (K as the GQA-strided activation, Q as the
 * small transposed weight) and ignore the shape's own N (n_tiles is derived
 * from gqa internally, per M2-j). Feeding them shape #2's (M=32, N=1024,
 * K=128) would silently run a meaningless tiny-kv variant of shape #1, NOT
 * the S=Q.K^T orientation #2 actually names -- that orientation needs a
 * dedicated kernel with K as the (transposed) weight, the same shape of
 * work run_qk_prefill_shape_f16 already does for #6, just at decode's M=32
 * scale. Not built this session; a real gap, not a flag away.
 *
 * #4-#7 (P.V decode, Q.K^T/P.V prefill) ARE measured, every run, by default
 * -- via the dedicated run_pv_shape_f16 / run_qk_prefill_shape_f16 /
 * run_pv_prefill_shape_f16 functions called unconditionally from main(),
 * NOT through this array or MHA_SHAPES_RUN. Their entries below are for
 * cataloging the full spec shape list only; the actual M/N/K used are the
 * literals at each call site in main().
 * ===================================================================== */
typedef struct {
  const char *name;
  uint32_t M, N, K;
} mha_shape_t;

static const mha_shape_t MHA_SHAPES[] = {
  {"qkT_dec_1024", 1024, 64, 128},  /* #1 S^T=K.Q^T decode, N=gqa padded to 64 -- RUN (this array) */
  {"qk_dec_1024  ", 32, 1024, 128}, /* #2 S=Q.K^T decode, needs Kt -- NOT IMPLEMENTED (see note above) */
  {"qkT_dec_512  ", 512, 64, 128},  /* #3 same as #1 at kv=512 -- NOT IMPLEMENTED (see note above) */
  {"pv_dec_1024  ", 32, 128, 1024}, /* #4 P.V decode -- run via run_pv_shape_f16(), not this array */
  {"pv_dec_512   ", 32, 128, 512},  /* #5 P.V decode kv=512 -- run via run_pv_shape_f16(), not this array */
  {"qk_pre_128   ", 128, 1024, 128},/* #6 Q.K^T prefill 128-token chunk -- run via run_qk_prefill_shape_f16() */
  {"pv_pre_128   ", 128, 128, 1024},/* #7 P.V prefill 128-token chunk -- run via run_pv_prefill_shape_f16() */
};
#define MHA_SHAPE_COUNT (sizeof(MHA_SHAPES) / sizeof(MHA_SHAPES[0]))
#if !defined(MHA_SHAPES_RUN)
#define MHA_SHAPES_RUN 1u /* only shape #1 is wired to this array's generic loop; see note above */
#endif

/* =====================================================================
 * KERNEL 1: hexkl_macro_mm_f16 full row-major path (the SDKL bar).
 * ===================================================================== */
static void run_macro_shape_f16(const char *name, uint32_t M, uint32_t N, uint32_t K) {
  if (M % 32 || N % 64 || K % 64) {
    printf("[mha_macro][%s] M=%u N=%u K=%u not tile-aligned (M%%32/N%%64/K%%64) "
           "-- padding path not implemented, skipping\n",
           name, (unsigned)M, (unsigned)N, (unsigned)K);
    return;
  }
  const size_t xn = (size_t)M * K, wn = (size_t)K * N, an = (size_t)M * N;

  _Float16 *X_rm = (_Float16 *)malloc(xn * sizeof(_Float16));
  /* W_kn: [K,N] row-major -- the spec's pinned ground truth layout, shared
   * with the micro kernels' reference (hexkl_test_matmul_cm_f16f16_f32). */
  _Float16 *W_kn = (_Float16 *)malloc(wn * sizeof(_Float16));
  /* W_nk: [N,K] row-major -- the MACRO API's own weight convention. The
   * vendor's macro fp16 example (examples/hexkl_macro_mm_f16) inits its W as
   * hexkl_test_init_pattern_f16(N_COL, N_INNER, W, ...) i.e. [n_col,n_inner]
   * = [N,K], and checks against hexkl_test_matmul_rm_f16f16_f16 whose W param
   * is documented `W[n_col][n_inner]` -- the OPPOSITE of the micro path's
   * [K,N]. hexkl_macro_f16_rm_to_f16_wh_inplace(N_COL, N_INNER, W) bakes that
   * [N,K] buffer in place. This is a genuine discrepancy vs the "ground
   * truth" pinned in the task spec for the micro API; it holds only for the
   * macro convenience layer -- flagging it rather than silently building two
   * different GEMM instances. W_nk is the transpose of W_kn so kernel 1 still
   * computes the exact same GEMM instance as kernels 2/3. */
  _Float16 *W_nk = (_Float16 *)malloc(wn * sizeof(_Float16));
  _Float16 *X_ah = (_Float16 *)malloc(xn * sizeof(_Float16));
  _Float16 *A_ah = (_Float16 *)malloc(an * sizeof(_Float16));
  float *ref = (float *)malloc(an * sizeof(float));
  if (!X_rm || !W_kn || !W_nk || !X_ah || !A_ah || !ref) {
    printf("[mha_macro][%s] oom, skipping\n", name);
    goto done;
  }

  hexkl_test_init_pattern_f16(M, K, X_rm, 5, 0.067f);
  hexkl_test_init_pattern_f16(K, N, W_kn, 3, 0.04906f);
  hexkl_test_init_zero_f32((uint32_t)an, ref);
  hexkl_test_matmul_cm_f16f16_f32(M, N, K, N, ref, X_rm, W_kn); /* ground truth, [K,N] */
  for (uint32_t k = 0; k < K; k++)
    for (uint32_t nn = 0; nn < N; nn++)
      W_nk[nn * K + k] = W_kn[k * N + nn]; /* transpose to the macro API's [N,K] */

  {
    int rc = hexkl_macro_f16_rm_to_f16_wh_inplace(N, K, W_nk); /* one-time weight bake, [N,K] */
    if (rc != AEE_SUCCESS) {
      printf("[mha_macro][%s] f16_rm_to_f16_wh_inplace failed 0x%x\n", name, rc);
      goto done;
    }
  }

  /* Cache coherency (header's "Cache Coherency Requirements" warning on
   * hexkl_macro_mm_f16, and the vendor macro fp16 example's exact sequence):
   * HMX reads/writes these DDR-backed buffers directly, bypassing normal CPU
   * cache coherency -- unlike the micro API's VTCM tiles, which are never
   * stale this way. Omitting this is a real, silent-wrong-answer bug: it
   * does not show up on the vendor's small (32x128x64) example, where the
   * buffers likely round-trip through cache incidentally, but reliably
   * corrupts results at this shape's larger (1024x64x128) working set. */
  const size_t xn_b = xn * sizeof(_Float16), wn_b = wn * sizeof(_Float16),
               an_b = an * sizeof(_Float16);

  /* warmup */
  memcpy(X_ah, X_rm, xn * sizeof(_Float16));
  hexkl_macro_f16_rm_to_f16_ah_inplace(M, K, X_ah);
  qurt_mem_cache_clean((qurt_addr_t)X_ah, xn_b, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
  qurt_mem_cache_clean((qurt_addr_t)W_nk, wn_b, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
  qurt_mem_cache_clean((qurt_addr_t)A_ah, an_b, QURT_MEM_CACHE_FLUSH_INVALIDATE, QURT_MEM_DCACHE);
  hexkl_macro_mm_f16((int)M, (int)N, (int)K, A_ah, X_ah, W_nk);
  hexkl_macro_f16_ah_to_f16_rm_inplace(M, N, A_ah);

  const int IT = 20;
  uint64_t t_ah = 0, t_mm = 0, t_rm = 0, ss;
  int rc = AEE_SUCCESS;
  for (int it = 0; it < IT; it++) {
    /* rm_to_f16_ah_inplace mutates its buffer in place (matches the vendor
     * macro fp16 example's exact call sequence) -- refresh from the pristine
     * row-major copy each iteration. */
    memcpy(X_ah, X_rm, xn * sizeof(_Float16));
    ss = READ_PCYCLES();
    rc = hexkl_macro_f16_rm_to_f16_ah_inplace(M, K, X_ah);
    t_ah += READ_PCYCLES() - ss;
    if (rc != AEE_SUCCESS)
      break;
    ss = READ_PCYCLES();
    qurt_mem_cache_clean((qurt_addr_t)X_ah, xn_b, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    qurt_mem_cache_clean((qurt_addr_t)W_nk, wn_b, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    qurt_mem_cache_clean((qurt_addr_t)A_ah, an_b, QURT_MEM_CACHE_FLUSH_INVALIDATE, QURT_MEM_DCACHE);
    rc = hexkl_macro_mm_f16((int)M, (int)N, (int)K, A_ah, X_ah, W_nk);
    t_mm += READ_PCYCLES() - ss; /* includes the mandatory cache-clean calls */
    if (rc != AEE_SUCCESS)
      break;
    ss = READ_PCYCLES();
    rc = hexkl_macro_f16_ah_to_f16_rm_inplace(M, N, A_ah);
    t_rm += READ_PCYCLES() - ss;
    if (rc != AEE_SUCCESS)
      break;
  }
  if (rc != AEE_SUCCESS) {
    printf("[mha_macro][%s] FAIL rc=0x%x\n", name, rc);
    goto done;
  }

  /* correctness OUTSIDE the timed loop -- A_ah now holds row-major fp16 */
  {
    err_stats e = err_stats_f16((uint32_t)an, ref, A_ah);
    double a = (double)t_ah / IT, mm = (double)t_mm / IT, r = (double)t_rm / IT;
    printf("\nMHA_SHAPE name=%s M=%u N=%u K=%u\n", name, (unsigned)M, (unsigned)N, (unsigned)K);
    printf("  === hexkl_macro (SDKL kernel) fp16, on-DSP (M=%u N=%u K=%u) ===\n",
           (unsigned)M, (unsigned)N, (unsigned)K);
    printf("    mm_f16 (core kernel, AH in/out)  %9.1f\n", mm);
    printf("    + rm_to_ah (activation in)       %9.1f\n", a);
    printf("    + ah_to_rm (output out)          %9.1f\n", r);
    printf("    full row-major path (=SDKL sem)  %9.1f   <- compare vs micro+DMA\n", a + mm + r);
    printf("    correctness: maxAbsErr=%.6g maxRelErr=%.6g meanRelErr=%.6g (idx=%ld)\n",
           e.max_abs, e.max_rel, e.mean_rel, e.max_rel_idx);
    printf("MHA_MACRO_US %.1f\n", a + mm + r);
    printf("MHA_MACRO_MAXRELERR %.6g\n", e.max_rel);
    printf("MHA_MACRO_CORRECT %s\n", e.max_rel < 5e-2 ? "OK" : "WARN");
    /* consolidated table row field -- see run_micro_mha_bench.sh's awk block.
     * This is a SINGLE-HEAD (per-call) number, unlike the per-layer (x8
     * kv_heads) fields the optimized kernels below report -- units are in
     * the field name itself so the report can't silently mix them. */
    printf("MHA_FIELD shape=%s field=macro_us_per_head value=%.1f\n", name, a + mm + r);
  }

done:
  free(X_rm);
  free(W_kn);
  free(W_nk);
  free(X_ah);
  free(A_ah);
  free(ref);
}

/* =====================================================================
 * KERNEL 2: micro API, scalar staging (the vendor example's structure,
 * verbatim: copy_submatrix_to_f16 + rm_to_ah_f16 per activation tile,
 * rm_to_wh_f16 per weight tile, re-derived every mm -- nothing resident).
 * ===================================================================== */
static int micro_naive_f16(uint8_t *vtcm_base, uint32_t hmx_config_offset,
                            uint32_t m, uint32_t n, uint32_t k, float *out_f32,
                            const _Float16 *actX, const _Float16 *wtW,
                            uint64_t *t_act, uint64_t *t_wt, uint64_t *t_mm,
                            uint64_t *t_out) {
  const uint32_t row_tiles_in_A = k / HEXKL_HMX_F16_BLOCK_N_INNER; /* K tiles, 32 wide */
  const uint32_t weight_scratch_off = HEXKL_HMX_ACTIVATION_ALIGNMENT * row_tiles_in_A;
  int rc;
  uint64_t s;

  for (uint32_t row = 0; row < m; row += HEXKL_HMX_F16_BLOCK_N_ROW) {
    s = READ_PCYCLES();
    for (uint32_t i = 0; i < row_tiles_in_A; i++) {
      rc = hexkl_micro_hmx_copy_submatrix_to_f16(
        vtcm_base, HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + i), actX,
        row / HEXKL_HMX_F16_BLOCK_N_ROW, i, m, k);
      if (rc != AEE_SUCCESS)
        return rc;
      rc = hexkl_micro_hmx_rm_to_ah_f16(
        vtcm_base, HEXKL_HMX_ACTIVATION_ALIGNMENT * i,
        HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + i));
      if (rc != AEE_SUCCESS)
        return rc;
    }
    *t_act += READ_PCYCLES() - s;

    for (uint32_t col = 0; col < n; col += HEXKL_HMX_F16_BLOCK_N_COL) {
      hexkl_micro_hmx_acc_clear_f16();
      for (uint32_t i = 0; i < row_tiles_in_A; i++) {
        s = READ_PCYCLES();
        rc = hexkl_micro_hmx_rm_to_wh_f16(vtcm_base, weight_scratch_off, wtW, i,
                                          col / HEXKL_HMX_F16_BLOCK_N_COL, n);
        *t_wt += READ_PCYCLES() - s;
        if (rc != AEE_SUCCESS)
          return rc;

        s = READ_PCYCLES();
        rc = hexkl_micro_hmx_mm_f16(vtcm_base, HEXKL_HMX_ACTIVATION_ALIGNMENT * i,
                                    weight_scratch_off);
        *t_mm += READ_PCYCLES() - s;
        if (rc != AEE_SUCCESS)
          return rc;
      }

      s = READ_PCYCLES();
      rc = hexkl_micro_hmx_acc_read_f16(vtcm_base, hmx_config_offset,
                                        HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + 1));
      if (rc != AEE_SUCCESS)
        return rc;
      rc = hexkl_micro_hmx_ah_to_rm_f16(vtcm_base, HEXKL_HMX_ACTIVATION_ALIGNMENT * row_tiles_in_A,
                                        HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + 1));
      if (rc != AEE_SUCCESS)
        return rc;
      rc = hexkl_micro_hmx_copy_f16_to_f32_submatrix(
        vtcm_base, HEXKL_HMX_ACTIVATION_ALIGNMENT * row_tiles_in_A, out_f32,
        row / HEXKL_HMX_F16_BLOCK_N_ROW, col / HEXKL_HMX_F16_BLOCK_N_COL, m, n);
      *t_out += READ_PCYCLES() - s;
      if (rc != AEE_SUCCESS)
        return rc;
    }
  }
  return AEE_SUCCESS;
}

/* ===================================================================
 * user-DMA (Hexagon V68+), lifted verbatim from hexkl_micro_fc_bench.c's
 * dma-queue helpers (llama.cpp ggml-hexagon lineage). Replaces the scalar
 * copy_submatrix_to_f16/output write-back for kernel 3.
 * =================================================================== */
typedef struct __attribute__((aligned(128))) hexdma_desc2d_s {
  void *next;
  uint32_t dst_stride : 24;
  uint32_t desc_size : 2;
  uint32_t dst_comp : 1;
  uint32_t src_comp : 1;
  uint32_t dst_bypass : 1;
  uint32_t src_bypass : 1;
  uint32_t order : 1;
  uint32_t done : 1;
  void *src;
  void *dst;
  uint32_t desc_type : 8;
  uint32_t reserved0 : 24;
  uint32_t row_size : 24;
  uint32_t nrows_lo : 8;
  uint32_t nrows_hi : 8;
  uint32_t src_stride : 24;
  uint32_t offset : 24;
  uint32_t reserved1 : 8;
} hexdma_desc2d;

static inline void hexdma_start(void *p) {
  asm volatile(" release(%0):at" ::"r"(p));
  asm volatile(" dmstart(%0)" ::"r"(p));
}
static inline unsigned hexdma_wait(void) {
  unsigned r = 0;
  asm volatile(" %0 = dmwait" : "=r"(r)::"memory");
  return r;
}

static inline void hexdma_2d_ddr_to_vtcm(hexdma_desc2d *d, void *dst, const void *src,
                                         uint32_t dst_stride, uint32_t src_stride,
                                         uint32_t row_size, uint32_t nrows) {
  d->next = 0;
  d->desc_size = 1;
  d->desc_type = 9;
  d->src_comp = 0;
  d->dst_comp = 0;
  d->src_bypass = 0; /* DDR through cache */
  d->dst_bypass = 1; /* VTCM */
  d->order = 0;
  d->done = 0;
  d->reserved0 = 0;
  d->reserved1 = 0;
  d->offset = 0;
  d->src = (void *)src;
  d->dst = dst;
  d->src_stride = src_stride;
  d->dst_stride = dst_stride;
  d->row_size = row_size;
  d->nrows_lo = nrows & 0xff;
  d->nrows_hi = (nrows >> 8) & 0xff;
  Q6_dccleaninva_A((void *)d);
  hexdma_start(d);
  hexdma_wait();
}

static inline void hexdma_2d_vtcm_to_ddr(hexdma_desc2d *d, void *dst, const void *src,
                                         uint32_t dst_stride, uint32_t src_stride,
                                         uint32_t row_size, uint32_t nrows) {
  d->next = 0;
  d->desc_size = 1;
  d->desc_type = 9;
  d->src_comp = 0;
  d->dst_comp = 0;
  d->src_bypass = 1; /* VTCM */
  d->dst_bypass = 0; /* DDR through cache */
  d->order = 0;
  d->done = 0;
  d->reserved0 = 0;
  d->reserved1 = 0;
  d->offset = 0;
  d->src = (void *)src;
  d->dst = dst;
  d->src_stride = src_stride;
  d->dst_stride = dst_stride;
  d->row_size = row_size;
  d->nrows_lo = nrows & 0xff;
  d->nrows_hi = (nrows >> 8) & 0xff;
  Q6_dccleaninva_A((void *)d);
  hexdma_start(d);
  hexdma_wait();
}

static inline void hexdma_2d_vtcm_to_ddr_async(hexdma_desc2d *d, void *dst, const void *src,
                                               uint32_t dst_stride, uint32_t src_stride,
                                               uint32_t row_size, uint32_t nrows) {
  d->next = 0;
  d->desc_size = 1;
  d->desc_type = 9;
  d->src_comp = 0;
  d->dst_comp = 0;
  d->src_bypass = 1;
  d->dst_bypass = 0;
  d->order = 0;
  d->done = 0;
  d->reserved0 = 0;
  d->reserved1 = 0;
  d->offset = 0;
  d->src = (void *)src;
  d->dst = dst;
  d->src_stride = src_stride;
  d->dst_stride = dst_stride;
  d->row_size = row_size;
  d->nrows_lo = nrows & 0xff;
  d->nrows_hi = (nrows >> 8) & 0xff;
  Q6_dccleaninva_A((void *)d);
  hexdma_start(d); /* no dmwait -- runs in background */
}
static inline void hexdma_poll_done(hexdma_desc2d *d) {
  while (!d->done) {
    unsigned r = 0;
    asm volatile(" %0 = dmpoll" : "=r"(r)::"memory");
    (void)r;
  }
}

/* ===================================================================
 * M2-a: minimal dmlink ring (ported verbatim from hexkl_micro_fc_bench.c's
 * cross-matmul harness, itself ported from llama.cpp ggml-hexagon
 * dma-queue). dmlink self-starts/resumes the single DMA engine, so many
 * transfers chain onto ONE engine and drain in one wait -- no per-descriptor
 * dmstart/dmwait round trip. This is the fix for kernel 3's activation DMA:
 * milestone 1 issued 128 separate BLOCKING 2KB transfers (32 bands x 4
 * k-tiles), each paying a full dmstart/dmwait round trip (~0.58us of pure
 * descriptor overhead per transfer, not transfer time) for ~3.5 GB/s -- a
 * 3-4x shortfall against the weight DMA's 11.7 GB/s on the same engine.
 * =================================================================== */
static inline void dmlink_(void *cur, void *next) {
  asm volatile(" release(%0):at" ::"r"(next));
  asm volatile(" dmlink(%0, %1)" ::"r"(cur), "r"(next));
}
#define RINGN 256u /* power of two; > pushes per band so no wrap */
static hexdma_desc2d g_ring[RINGN] __attribute__((aligned(128)));
static uint32_t r_push, r_pop;
static hexdma_desc2d *r_tail;
static int r_started;
/*!
  @brief M2-e: reset only the ring's bookkeeping (indices/tail/started), NOT
  the 256-descriptor array. ring_push2d already writes every field of the
  descriptor it touches (next, desc_size, desc_type, both comp/bypass bits,
  order, done, reserved0/1, offset, src, dst, both strides, row_size,
  nrows_lo/hi) before it is ever linked or started, so a stale array is never
  read uninitialized. Milestone 1's `memset(g_ring, 0, sizeof(g_ring))` here
  -- 256 descriptors x 128B = 32KB, called once per row-band (32x/matmul) =
  1MB of memset per call -- was exactly the ~35us gap between the phase-sum
  and the wall-clock total (it sat outside every timed phase, so it was
  invisible in the breakdown but not in the wall time).
*/
static void ring_init(void) {
  r_push = 0;
  r_pop = 0;
  r_tail = &g_ring[RINGN - 1];
  r_started = 0;
}
static void ring_wait_idx(uint32_t i) {
  long guard = 0;
  while (!g_ring[i].done && guard++ < 50000000L) {
    unsigned r = 0;
    asm volatile(" %0 = dmpoll" : "=r"(r)::"memory");
    (void)r;
  }
}
static void ring_drain(void) {
  while (r_pop != r_push) {
    ring_wait_idx(r_pop);
    r_pop = (r_pop + 1) & (RINGN - 1);
  }
  r_started = 0; /* engine idle after drain -> next push must dmstart */
}
static void ring_push2d(void *dst, const void *src, uint32_t ds, uint32_t ss, uint32_t rs,
                        uint32_t nr, int src_vtcm, int dst_vtcm) {
  if (((r_push + 1) & (RINGN - 1)) == r_pop) { /* full: reclaim oldest */
    ring_wait_idx(r_pop);
    r_pop = (r_pop + 1) & (RINGN - 1);
  }
  hexdma_desc2d *d = &g_ring[r_push];
  d->next = 0;
  d->desc_size = 1;
  d->desc_type = 9;
  d->src_comp = 0;
  d->dst_comp = 0;
  d->src_bypass = src_vtcm ? 1 : 0;
  d->dst_bypass = dst_vtcm ? 1 : 0;
  d->order = 0;
  d->done = 0;
  d->reserved0 = 0;
  d->reserved1 = 0;
  d->offset = 0;
  d->src = (void *)src;
  d->dst = dst;
  d->src_stride = ss;
  d->dst_stride = ds;
  d->row_size = rs;
  d->nrows_lo = nr & 0xff;
  d->nrows_hi = (nr >> 8) & 0xff;
  Q6_dccleaninva_A((void *)d);
  if (!r_started) {
    hexdma_start(d);
    r_started = 1;
  } else {
    dmlink_(r_tail, d);
  }
  r_tail = d;
  r_push = (r_push + 1) & (RINGN - 1);
}

/*!
  @brief VTCM partitioning for kernel 3 (resident weight, ring-prefetched
  double-buffered activation, ring-written output -- everything past the
  one-time weight DMA goes through the SAME ring so there is never more than
  one dmstart call site; mixing ring-chained transfers with an independent
  manual dmstart is exactly the "second dmstart while busy" crash the FC
  bench's pitfalls warn about). Parameters are always named (m, k, n) and
  called in that order -- the FC bench's plan_vtcm(M,N,K) vs (M,K,N)
  arg-swap bug is exactly what this convention avoids.
*/
typedef struct {
  uint32_t k_tiles, n_tiles, row_bands;
  uint32_t act_ah_off;      /* single-buffer AH landing, k_tiles tiles (converted just before use) */
  uint32_t act_flat_off[2]; /* double-buffered ring DMA landing, k_tiles tiles each */
  uint32_t wt_off, wt_bytes; /* resident WH, k_tiles*n_tiles tiles */
  uint32_t acc_ah_off;       /* scratch: acc_read_f16 landing (AH) */
  uint32_t out_flat_base;    /* n_tiles slots (one per output tile -- no reuse before ring_drain) */
  uint32_t config_off;
  bool fits;
} vtcm_plan_f16;

static vtcm_plan_f16 plan_vtcm_f16(uint32_t m, uint32_t k, uint32_t n, uint32_t vtcm_size) {
  vtcm_plan_f16 p;
  p.k_tiles = k / HEXKL_HMX_F16_BLOCK_N_INNER;
  p.n_tiles = n / HEXKL_HMX_F16_BLOCK_N_COL;
  p.row_bands = m / HEXKL_HMX_F16_BLOCK_N_ROW;
  p.config_off = vtcm_size - hexkl_micro_hmx_config_size();

  p.act_ah_off = 0;
  const uint32_t act_bytes = p.k_tiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;
  p.act_flat_off[0] = p.act_ah_off + act_bytes;
  p.act_flat_off[1] = p.act_flat_off[0] + act_bytes;

  p.wt_off = p.act_flat_off[1] + act_bytes;
  p.wt_bytes = p.k_tiles * p.n_tiles * WT_TILE_BYTES_F16;

  p.acc_ah_off = align_up(p.wt_off + p.wt_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  p.out_flat_base = p.acc_ah_off + HEXKL_HMX_ACTIVATION_ALIGNMENT;

  const uint32_t need = p.out_flat_base + p.n_tiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;
  p.fits = need <= p.config_off;
  return p;
}

/*! @brief Bake every weight tile once into its own resident VTCM slot. */
static int bake_weight_resident_f16(uint8_t *vtcm_base, const vtcm_plan_f16 *p, uint32_t n,
                                    const _Float16 *wtW_rm) {
  for (uint32_t kt = 0; kt < p->k_tiles; kt++) {
    for (uint32_t nt = 0; nt < p->n_tiles; nt++) {
      const uint32_t off = p->wt_off + (kt * p->n_tiles + nt) * WT_TILE_BYTES_F16;
      int rc = hexkl_micro_hmx_rm_to_wh_f16(vtcm_base, off, wtW_rm, kt, nt, n);
      if (rc != AEE_SUCCESS)
        return rc;
    }
  }
  return AEE_SUCCESS;
}

/*!
  @brief Push one row band's k_tiles activation descriptors (GQA-strided K
  cache -> flat VTCM slot) onto the ring. Does NOT drain -- the caller decides
  when to wait. Geometry is unchanged from milestone 1 (row_size=64B,
  src_stride=nch*256B, dst_stride=64B, nrows=32, 2048B-aligned destinations)
  -- only the issue/wait discipline changed, per the M2-a instruction.
*/
static void push_band_activation(uint8_t *vtcm_base, const vtcm_plan_f16 *p,
                                 const _Float16 *kcache_ddr, uint32_t nch, uint32_t k,
                                 uint32_t band, int slot) {
  const uint32_t src_stride = nch * k * (uint32_t)sizeof(_Float16);
  const uint8_t *band_base =
    (const uint8_t *)kcache_ddr + (size_t)band * HEXKL_HMX_F16_BLOCK_N_ROW * src_stride;
  for (uint32_t kt = 0; kt < p->k_tiles; kt++) {
    const uint8_t *asrc = band_base + (size_t)kt * HEXKL_HMX_F16_BLOCK_N_INNER * sizeof(_Float16);
    ring_push2d(vtcm_base + p->act_flat_off[slot] + HEXKL_HMX_ACTIVATION_ALIGNMENT * kt, asrc,
               HEXKL_HMX_F16_BLOCK_N_INNER * (uint32_t)sizeof(_Float16) /* dst_stride */, src_stride,
               HEXKL_HMX_F16_BLOCK_N_INNER * (uint32_t)sizeof(_Float16) /* row_size */,
               HEXKL_HMX_F16_BLOCK_N_ROW /* nrows */, /*src_vtcm=*/0, /*dst_vtcm=*/1);
  }
}

/*! @brief rm_to_ah_f16 the k_tiles flat tiles in the given slot into the single act_ah scratch. */
static int convert_band_to_ah(uint8_t *vtcm_base, const vtcm_plan_f16 *p, int slot) {
  for (uint32_t kt = 0; kt < p->k_tiles; kt++) {
    int rc = hexkl_micro_hmx_rm_to_ah_f16(vtcm_base, p->act_ah_off + HEXKL_HMX_ACTIVATION_ALIGNMENT * kt,
                                          p->act_flat_off[slot] + HEXKL_HMX_ACTIVATION_ALIGNMENT * kt);
    if (rc != AEE_SUCCESS)
      return rc;
  }
  return AEE_SUCCESS;
}

/*!
  @brief M2-a isolated measurement: JUST the ring-based activation DMA (no
  compute, no output) -- chains every row band's k_tiles descriptors onto one
  ring and drains ONCE at the end. This is the clean, directly-comparable
  number for "did fixing the issue/wait discipline help": milestone 1's 128
  BLOCKING transfers (32 bands x 4 k-tiles, one dmstart/dmwait pair each) vs
  this ring's single dmstart + N dmlink + one dmwait for the same bytes.
  Destinations cycle through the 2 flat slots (safe: dmlink chains execute
  strictly in order on one engine, so band N+2 reusing band N's slot always
  waits for band N to finish -- nothing reads the data in this isolated test
  anyway, only the achieved DMA bandwidth is being measured.)
*/
static uint64_t bench_activation_ring_only(uint8_t *vtcm_base, const vtcm_plan_f16 *p,
                                           const _Float16 *kcache_ddr, uint32_t nch, uint32_t k) {
  uint64_t s = READ_PCYCLES();
  ring_init();
  for (uint32_t band = 0; band < p->row_bands; band++)
    push_band_activation(vtcm_base, p, kcache_ddr, nch, k, band, (int)(band & 1));
  ring_drain();
  return READ_PCYCLES() - s;
}

/*!
  @brief Kernel 3: resident weight (pre-baked WH staged from DDR every call,
  mirroring a real multi-layer deployment where WH can't all stay resident),
  activation prefetched one row-band ahead via the dmlink ring (M2-a fix --
  milestone 1's per-tile blocking DMA replaced; see push_band_activation),
  output ALSO written back through the SAME ring (never a second, independent
  dmstart while ring descriptors may still be chained/in-flight). Pattern:
  push band b+1's activation while band b's mm/output run, one ring_drain()
  per band (waits for both band b+1's activation AND band b's output tiles).
*/
static int matmul_micro_dma_f16(uint8_t *vtcm_base, uint32_t vtcm_size, const vtcm_plan_f16 *p,
                                uint32_t m, uint32_t n, uint32_t k, _Float16 *out,
                                const _Float16 *kcache_ddr, /* [kv, nch, hd], GQA-strided */
                                uint32_t nch, const uint8_t *wt_wh_ddr, uint64_t *t_weight,
                                uint64_t *t_act_dma, uint64_t *t_act_layout, uint64_t *t_mm,
                                uint64_t *t_accread, uint64_t *t_ahtorm, uint64_t *t_outdma,
                                uint64_t *t_ringdrain) {
  static hexdma_desc2d g_desc __attribute__((aligned(128)));
  uint64_t s;
  int rc;

  /* weight DDR -> VTCM: one blocking transfer via the OLD standalone hexdma_*
   * helper, done ONCE before the ring is touched at all -- its internal
   * dmwait leaves the engine idle, so ring_init() right after is safe. The
   * honest per-matmul cost in a multi-layer/multi-head deployment where WH
   * can't all stay resident (mirrors matmul_resident_dma in the FC bench). */
  s = READ_PCYCLES();
  {
    uint32_t rs = 262144u;
    while (p->wt_bytes % rs)
      rs >>= 1;
    hexdma_2d_ddr_to_vtcm(&g_desc, vtcm_base + p->wt_off, wt_wh_ddr, rs, rs, rs,
                          p->wt_bytes / rs);
  }
  *t_weight += READ_PCYCLES() - s;

  /* prime: push + drain band 0's activation synchronously (must have data
   * before the first use), then convert to AH. */
  s = READ_PCYCLES();
  ring_init();
  push_band_activation(vtcm_base, p, kcache_ddr, nch, k, 0, 0);
  ring_drain();
  *t_act_dma += READ_PCYCLES() - s;

  s = READ_PCYCLES();
  rc = convert_band_to_ah(vtcm_base, p, 0);
  if (rc != AEE_SUCCESS)
    return rc;
  *t_act_layout += READ_PCYCLES() - s;

  for (uint32_t band = 0; band < p->row_bands; band++) {
    const int cur_slot = (int)(band & 1);
    const int nxt_slot = (int)((band + 1) & 1);

    /* M2-e: no per-band ring_init() -- the ring holds well under RINGN=256
     * descriptors across a whole matmul call (e.g. ~192 for this shape: 4
     * prime + 31*4 activation + 32*2 output), and ring_drain() below already
     * resets r_started=0 (the only per-"session" state ring_push2d reads
     * before its first push) each time it fully drains. r_push/r_pop keep
     * incrementing mod RINGN across bands, which is what the bitmask
     * indexing is for; ring_push2d's own overflow-reclaim path (wait for the
     * oldest slot, then advance r_pop) is the safety net if a future, larger
     * shape ever did exceed one ring's capacity mid-band. */
    s = READ_PCYCLES();
    if (band + 1 < p->row_bands)
      push_band_activation(vtcm_base, p, kcache_ddr, nch, k, band + 1, nxt_slot);
    *t_act_dma += READ_PCYCLES() - s; /* just the (non-blocking) push */

    for (uint32_t nt = 0; nt < p->n_tiles; nt++) {
      s = READ_PCYCLES();
      hexkl_micro_hmx_acc_clear_f16();
      for (uint32_t kt = 0; kt < p->k_tiles; kt++) {
        rc = hexkl_micro_hmx_mm_f16(vtcm_base, p->act_ah_off + HEXKL_HMX_ACTIVATION_ALIGNMENT * kt,
                                    p->wt_off + (kt * p->n_tiles + nt) * WT_TILE_BYTES_F16);
        if (rc != AEE_SUCCESS)
          return rc;
      }
      *t_mm += READ_PCYCLES() - s;

      s = READ_PCYCLES();
      rc = hexkl_micro_hmx_acc_read_f16(vtcm_base, p->config_off, p->acc_ah_off);
      if (rc != AEE_SUCCESS)
        return rc;
      *t_accread += READ_PCYCLES() - s;

      const uint32_t out_slot_off = p->out_flat_base + nt * HEXKL_HMX_ACTIVATION_ALIGNMENT;
      s = READ_PCYCLES();
      rc = hexkl_micro_hmx_ah_to_rm_f16(vtcm_base, out_slot_off, p->acc_ah_off);
      if (rc != AEE_SUCCESS)
        return rc;
      *t_ahtorm += READ_PCYCLES() - s;

      s = READ_PCYCLES();
      _Float16 *dst = out + (size_t)band * HEXKL_HMX_F16_BLOCK_N_ROW * n +
                      (size_t)nt * HEXKL_HMX_F16_BLOCK_N_COL;
      /* output write-back through the SAME ring as the activation prefetch
       * above -- one output tile per slot (n_tiles slots total), so no
       * output tile is overwritten before ring_drain() below confirms it
       * has actually left VTCM. */
      ring_push2d(dst, vtcm_base + out_slot_off, n * (uint32_t)sizeof(_Float16),
                 HEXKL_HMX_F16_BLOCK_N_COL * (uint32_t)sizeof(_Float16),
                 HEXKL_HMX_F16_BLOCK_N_COL * (uint32_t)sizeof(_Float16), HEXKL_HMX_F16_BLOCK_N_ROW,
                 /*src_vtcm=*/1, /*dst_vtcm=*/0);
      *t_outdma += READ_PCYCLES() - s;
    }

    s = READ_PCYCLES();
    ring_drain(); /* waits for band+1's activation AND this band's output tiles */
    *t_ringdrain += READ_PCYCLES() - s;

    if (band + 1 < p->row_bands) {
      s = READ_PCYCLES();
      rc = convert_band_to_ah(vtcm_base, p, nxt_slot);
      if (rc != AEE_SUCCESS)
        return rc;
      *t_act_layout += READ_PCYCLES() - s;
    }
    (void)cur_slot;
  }
  return AEE_SUCCESS;
}

/* =====================================================================
 * M2-i: stop slicing per kv_head -- DMA the layer's K cache contiguously.
 *
 * The full K cache for one layer is [kv_len][nch][head_dim], fully
 * contiguous (2 MiB at kv_len=1024, nch=8, head_dim=128). One descriptor
 * (row_size=256KiB, nrows=8 -- both under the <=256KB single-row limit and
 * at the top of the M2-f curve, ~67 GB/s) brings the WHOLE layer DDR->VTCM
 * in one shot, replacing 8 separate per-head strided pulls (8 x 22.8us).
 * Then, per kv_head, the 32x32 tiles are repacked VTCM->VTCM into the
 * 64B-row / 2048B-aligned layout rm_to_ah_f16 needs -- still 64B rows, but
 * VTCM latency is orders of magnitude below DDR, so the same small-row
 * penalty measured in M2-f (DDR) may or may not carry over. Measured, not
 * assumed, below (two variants).
 *
 * Key addressing trick that makes variant (b) trivial: once the WHOLE
 * layer is VTCM-resident, the physical row width IS the valid data width
 * (nch*head_dim, no gap) -- unlike the single-head DDR case, where the
 * valid head_dim-wide row sits inside a wider nch*head_dim physical stride
 * with no way to express that gap to copy_submatrix_to_f16 (which has no
 * separate stride parameter). So input_cols=nch*head_dim, tile_col =
 * head_n*k_tiles+kt addresses head_n's kt-th column-tile directly, no
 * extraction step needed at all for variant (b).
 * ===================================================================== */

/*! @brief Variant (a): VTCM->VTCM 2D DMA per tile, same geometry as push_band_activation,
    source now the VTCM-resident layer slab instead of DDR. */
static void push_band_repack_vtcm(uint8_t *vtcm_base, const vtcm_plan_f16 *p,
                                  uint32_t layer_kv_off, uint32_t nch, uint32_t k,
                                  uint32_t head_n, uint32_t band, int slot) {
  const uint32_t src_stride = nch * k * (uint32_t)sizeof(_Float16);
  const uint32_t band_base_off = layer_kv_off +
                                 (uint32_t)((size_t)band * HEXKL_HMX_F16_BLOCK_N_ROW * src_stride) +
                                 head_n * k * (uint32_t)sizeof(_Float16);
  for (uint32_t kt = 0; kt < p->k_tiles; kt++) {
    const uint32_t src_off = band_base_off + kt * HEXKL_HMX_F16_BLOCK_N_INNER * (uint32_t)sizeof(_Float16);
    ring_push2d(vtcm_base + p->act_flat_off[slot] + HEXKL_HMX_ACTIVATION_ALIGNMENT * kt,
               vtcm_base + src_off, HEXKL_HMX_F16_BLOCK_N_INNER * (uint32_t)sizeof(_Float16),
               src_stride, HEXKL_HMX_F16_BLOCK_N_INNER * (uint32_t)sizeof(_Float16),
               HEXKL_HMX_F16_BLOCK_N_ROW, /*src_vtcm=*/1, /*dst_vtcm=*/1);
  }
}

/*! @brief Variant (b): scalar hexkl_micro_hmx_copy_submatrix_to_f16, reading
    the VTCM-resident layer slab directly (input_cols=nch*k is the TRUE
    physical row width now, so no gap/stride problem -- see block comment
    above). Writes into the SAME flat landing slot push_band_activation used,
    so convert_band_to_ah works unchanged afterward. */
static int repack_band_scalar_from_vtcm(uint8_t *vtcm_base, const vtcm_plan_f16 *p,
                                        uint32_t layer_kv_off, uint32_t nch, uint32_t k,
                                        uint32_t m, uint32_t head_n, uint32_t band, int slot) {
  const _Float16 *layer_slab = (const _Float16 *)(vtcm_base + layer_kv_off);
  const uint32_t input_cols = nch * k;
  for (uint32_t kt = 0; kt < p->k_tiles; kt++) {
    int rc = hexkl_micro_hmx_copy_submatrix_to_f16(
      vtcm_base, p->act_flat_off[slot] + HEXKL_HMX_ACTIVATION_ALIGNMENT * kt, layer_slab, band,
      head_n * p->k_tiles + kt, m, input_cols);
    if (rc != AEE_SUCCESS)
      return rc;
  }
  return AEE_SUCCESS;
}

/*! @brief Isolated measurement: variant (a) alone, one kv_head, all row bands, one ring/drain. */
static uint64_t bench_repack_ring_only(uint8_t *vtcm_base, const vtcm_plan_f16 *p,
                                       uint32_t layer_kv_off, uint32_t nch, uint32_t k,
                                       uint32_t head_n) {
  uint64_t s = READ_PCYCLES();
  ring_init();
  for (uint32_t band = 0; band < p->row_bands; band++)
    push_band_repack_vtcm(vtcm_base, p, layer_kv_off, nch, k, head_n, band, (int)(band & 1));
  ring_drain();
  return READ_PCYCLES() - s;
}

/*! @brief Isolated measurement: variant (b) alone, one kv_head, all row bands, scalar (synchronous, no DMA at all). */
static uint64_t bench_repack_scalar_only(uint8_t *vtcm_base, const vtcm_plan_f16 *p,
                                         uint32_t layer_kv_off, uint32_t nch, uint32_t k,
                                         uint32_t m, uint32_t head_n) {
  uint64_t s = READ_PCYCLES();
  for (uint32_t band = 0; band < p->row_bands; band++)
    repack_band_scalar_from_vtcm(vtcm_base, p, layer_kv_off, nch, k, m, head_n, band, 0);
  return READ_PCYCLES() - s;
}

/*!
  @brief Full per-head pipeline reading activation from the VTCM-resident
  layer slab via variant (a) (ring-based VTCM->VTCM repack), otherwise
  identical to matmul_micro_dma_f16 (resident weight, ring-based output,
  one ring_drain per band). Used to build the real, verified "per-layer
  total for all 8 kv_heads" number -- not an analytical composition of
  separately-measured phases.
*/
static int matmul_layer_dma_f16(uint8_t *vtcm_base, uint32_t vtcm_size, const vtcm_plan_f16 *p,
                                uint32_t m, uint32_t n, uint32_t k, _Float16 *out,
                                uint32_t layer_kv_off, uint32_t nch, uint32_t head_n,
                                const uint8_t *wt_wh_ddr, uint64_t *t_weight, uint64_t *t_act_dma,
                                uint64_t *t_act_layout, uint64_t *t_mm, uint64_t *t_accread,
                                uint64_t *t_ahtorm, uint64_t *t_outdma, uint64_t *t_ringdrain) {
  static hexdma_desc2d g_desc __attribute__((aligned(128)));
  uint64_t s;
  int rc;

  s = READ_PCYCLES();
  {
    uint32_t rs = 262144u;
    while (p->wt_bytes % rs)
      rs >>= 1;
    hexdma_2d_ddr_to_vtcm(&g_desc, vtcm_base + p->wt_off, wt_wh_ddr, rs, rs, rs,
                          p->wt_bytes / rs);
  }
  *t_weight += READ_PCYCLES() - s;

  s = READ_PCYCLES();
  ring_init();
  push_band_repack_vtcm(vtcm_base, p, layer_kv_off, nch, k, head_n, 0, 0);
  ring_drain();
  *t_act_dma += READ_PCYCLES() - s;

  s = READ_PCYCLES();
  rc = convert_band_to_ah(vtcm_base, p, 0);
  if (rc != AEE_SUCCESS)
    return rc;
  *t_act_layout += READ_PCYCLES() - s;

  for (uint32_t band = 0; band < p->row_bands; band++) {
    const int nxt_slot = (int)((band + 1) & 1);

    s = READ_PCYCLES();
    if (band + 1 < p->row_bands)
      push_band_repack_vtcm(vtcm_base, p, layer_kv_off, nch, k, head_n, band + 1, nxt_slot);
    *t_act_dma += READ_PCYCLES() - s;

    for (uint32_t nt = 0; nt < p->n_tiles; nt++) {
      s = READ_PCYCLES();
      hexkl_micro_hmx_acc_clear_f16();
      for (uint32_t kt = 0; kt < p->k_tiles; kt++) {
        rc = hexkl_micro_hmx_mm_f16(vtcm_base, p->act_ah_off + HEXKL_HMX_ACTIVATION_ALIGNMENT * kt,
                                    p->wt_off + (kt * p->n_tiles + nt) * WT_TILE_BYTES_F16);
        if (rc != AEE_SUCCESS)
          return rc;
      }
      *t_mm += READ_PCYCLES() - s;

      s = READ_PCYCLES();
      rc = hexkl_micro_hmx_acc_read_f16(vtcm_base, p->config_off, p->acc_ah_off);
      if (rc != AEE_SUCCESS)
        return rc;
      *t_accread += READ_PCYCLES() - s;

      const uint32_t out_slot_off = p->out_flat_base + nt * HEXKL_HMX_ACTIVATION_ALIGNMENT;
      s = READ_PCYCLES();
      rc = hexkl_micro_hmx_ah_to_rm_f16(vtcm_base, out_slot_off, p->acc_ah_off);
      if (rc != AEE_SUCCESS)
        return rc;
      *t_ahtorm += READ_PCYCLES() - s;

      s = READ_PCYCLES();
      _Float16 *dst = out + (size_t)band * HEXKL_HMX_F16_BLOCK_N_ROW * n +
                      (size_t)nt * HEXKL_HMX_F16_BLOCK_N_COL;
      ring_push2d(dst, vtcm_base + out_slot_off, n * (uint32_t)sizeof(_Float16),
                 HEXKL_HMX_F16_BLOCK_N_COL * (uint32_t)sizeof(_Float16),
                 HEXKL_HMX_F16_BLOCK_N_COL * (uint32_t)sizeof(_Float16), HEXKL_HMX_F16_BLOCK_N_ROW,
                 /*src_vtcm=*/1, /*dst_vtcm=*/0);
      *t_outdma += READ_PCYCLES() - s;
    }

    s = READ_PCYCLES();
    ring_drain();
    *t_ringdrain += READ_PCYCLES() - s;

    if (band + 1 < p->row_bands) {
      s = READ_PCYCLES();
      rc = convert_band_to_ah(vtcm_base, p, nxt_slot);
      if (rc != AEE_SUCCESS)
        return rc;
      *t_act_layout += READ_PCYCLES() - s;
    }
  }
  return AEE_SUCCESS;
}

/* =====================================================================
 * per-shape driver for the micro block: kernel 2 (naive) then kernel 3
 * (micro+DMA), against a shared reference. Runs standalone -- caller has
 * already done hexkl_micro_hw_init + hmx_lock + setup_acc_read_f16.
 * ===================================================================== */
static void run_micro_shape_f16(uint8_t *vtcm_base, uint32_t vtcm_size, uint32_t hmx_config_offset,
                                const char *name, uint32_t M, uint32_t N_spec, uint32_t K) {
  /* M2-j: N_spec (e.g. 64 for qkT_dec_1024) is the spec's padding, derived
   * from HEXKL_HMX_F16_BLOCK_N_COL_ALIGNMENT=64 -- that constant is a
   * MACRO-kernel requirement (run_macro_shape_f16 still needs it and is
   * untouched). The micro path's tile is 32x32 and we own the tile loop, so
   * the real constraint is just the 32-wide micro tile: align gqa up to 32,
   * not 64. For gqa=8 that is N=32 (n_tiles=1) instead of N=64 (n_tiles=2)
   * -- half the mm/acc_read/ah_to_rm/output-tile count. Confirmed on-device
   * below that mm_f16/acc_read_f16/ah_to_rm_f16 all behave correctly with a
   * single 32-wide tile (no revert needed). */
  const uint32_t N = align_up(MHA_GQA, HEXKL_HMX_F16_BLOCK_N_COL);
  (void)N_spec;
  if (M % 32 || N % HEXKL_HMX_F16_BLOCK_N_COL || K % 64) {
    printf("[mha_micro][%s] M=%u N=%u(micro, from gqa=%u; spec N=%u) K=%u not "
           "tile-aligned, skipping (padding path not implemented)\n",
           name, (unsigned)M, (unsigned)N, (unsigned)MHA_GQA, (unsigned)N_spec, (unsigned)K);
    return;
  }

  /* ---- build a tight [M,K] activation reference copy + [K,N] weight
   * (Q^T, from a small scalar transpose of a natural [gqa,K] Q) ---- */
  _Float16 *X_ref_flat = (_Float16 *)malloc((size_t)M * K * sizeof(_Float16));
  _Float16 *Q_nat = (_Float16 *)malloc((size_t)MHA_GQA * K * sizeof(_Float16));
  _Float16 *W_rm = (_Float16 *)calloc((size_t)K * N, sizeof(_Float16)); /* zero-pad past gqa cols */
  float *ref = (float *)malloc((size_t)M * N * sizeof(float));
  /* GQA-strided K-cache DDR source: [kv, nch, hd], only head n=0 filled. */
  _Float16 *kcache_ddr = (_Float16 *)malloc((size_t)M * MHA_NCH * K * sizeof(_Float16));
  if (!X_ref_flat || !Q_nat || !W_rm || !ref || !kcache_ddr) {
    printf("[mha_micro][%s] oom, skipping\n", name);
    goto done0;
  }

  hexkl_test_init_pattern_f16(M, K, X_ref_flat, 5, 0.067f);
  /* M2-i: fill ALL nch heads (not just n=0) so the layer-wide DMA test has a
   * real, full [kv_len][nch][head_dim] layer to move. Every head gets the
   * SAME per-row pattern (a benchmark simplification for the DMA/repack
   * mechanics test, stated plainly here and in the report -- a real layer
   * has distinct per-head K data, which does not change the byte-mover
   * mechanics being measured). Each head can then be checked against the
   * SAME `ref` computed below. */
  for (uint32_t row = 0; row < M; row++)
    for (uint32_t h = 0; h < MHA_NCH; h++)
      for (uint32_t d = 0; d < K; d++)
        kcache_ddr[(row * MHA_NCH + h) * K + d] = X_ref_flat[row * K + d];

  hexkl_test_init_pattern_f16(MHA_GQA, K, Q_nat, 3, 0.04906f);
  uint64_t t_transpose;
  {
    uint64_t s = READ_PCYCLES();
    for (uint32_t k = 0; k < K; k++)
      for (uint32_t g = 0; g < MHA_GQA && g < N; g++)
        W_rm[k * N + g] = Q_nat[g * K + k];
    t_transpose = READ_PCYCLES() - s;
  }

  hexkl_test_init_zero_f32(M * N, ref);
  hexkl_test_matmul_cm_f16f16_f32(M, N, K, N, ref, X_ref_flat, W_rm);

  printf("\nMHA_SHAPE name=%s M=%u N=%u K=%u (micro block)\n", name, (unsigned)M, (unsigned)N,
         (unsigned)K);
  printf("  weight transpose (Q^T, host scalar, one-time) %9.1f us  (%u x %u, ~%u B)\n",
         (double)t_transpose, (unsigned)K, (unsigned)MHA_GQA, (unsigned)(K * MHA_GQA * 2));

  /* ---- kernel 2: micro, naive scalar staging ---- */
  {
    float *out_f32 = (float *)malloc((size_t)M * N * sizeof(float));
    if (!out_f32) {
      printf("[mha_micro][%s] oom (kernel2), skipping\n", name);
      goto done_k2;
    }
    const int IT = 20;
    uint64_t t_act = 0, t_wt = 0, t_mm = 0, t_out = 0;
    int rc = AEE_SUCCESS;
    for (int it = 0; it < IT; it++) {
      memset(out_f32, 0, (size_t)M * N * sizeof(float));
      rc = micro_naive_f16(vtcm_base, hmx_config_offset, M, N, K, out_f32, X_ref_flat, W_rm,
                           &t_act, &t_wt, &t_mm, &t_out);
      if (rc != AEE_SUCCESS)
        break;
    }
    if (rc != AEE_SUCCESS) {
      printf("[mha_micro][%s] kernel2 FAIL rc=0x%x\n", name, rc);
    } else {
      err_stats e = err_stats_f32(M * N, ref, out_f32);
      double a = (double)t_act / IT, w = (double)t_wt / IT, mm = (double)t_mm / IT,
             o = (double)t_out / IT;
      double tot = a + w + mm + o;
      printf("  === micro-API kernel, naive scalar staging (M=%u N=%u K=%u, iters=%d) ===\n",
             (unsigned)M, (unsigned)N, (unsigned)K, IT);
      printf("    Steady state                       %9.1f   <- per-call\n", tot);
      printf("      activation staging (scalar)      %9.1f\n", a);
      printf("      weight re-bake (rm_to_wh, scalar) %9.1f\n", w);
      printf("      DSP matmul (HMX mm_f16)           %9.1f\n", mm);
      printf("      output (acc_read+ah_to_rm+copy)   %9.1f\n", o);
      printf("    correctness: maxAbsErr=%.6g maxRelErr=%.6g meanRelErr=%.6g (idx=%ld)\n",
             e.max_abs, e.max_rel, e.mean_rel, e.max_rel_idx);
      printf("MHA_MICRO_US %.1f\n", tot);
      printf("MHA_MICRO_MAXRELERR %.6g\n", e.max_rel);
      printf("MHA_MICRO_CORRECT %s\n", e.max_rel < 5e-2 ? "OK" : "WARN");
      /* per-head (single-call) number, same convention as macro_us_per_head. */
      printf("MHA_FIELD shape=%s field=micro_us_per_head value=%.1f\n", name, tot);
    }
    free(out_f32);
  }
done_k2:;

  /* ---- kernel 3: micro + user-DMA (resident weight, pre-baked WH in DDR,
   * GQA-strided activation DMA, double-buffered async output DMA) ---- */
  {
    vtcm_plan_f16 p = plan_vtcm_f16(M, K, N, vtcm_size);
    printf("  [kernel3 vtcm plan] k_tiles=%u n_tiles=%u row_bands=%u act=%u wt=%u "
           "need<=%u %s\n",
           (unsigned)p.k_tiles, (unsigned)p.n_tiles, (unsigned)p.row_bands,
           (unsigned)(p.act_ah_off), (unsigned)p.wt_bytes, (unsigned)p.config_off,
           p.fits ? "" : "<-- DOES NOT FIT, skipping");
    if (!p.fits)
      goto done_k3;

    _Float16 *out_res = (_Float16 *)malloc((size_t)M * N * sizeof(_Float16));
    uint8_t *wt_wh_ddr = (uint8_t *)malloc(p.wt_bytes);
    if (!out_res || !wt_wh_ddr) {
      printf("[mha_micro][%s] oom (kernel3), skipping\n", name);
      free(out_res);
      free(wt_wh_ddr);
      goto done_k3;
    }

    /* one-time: bake WH resident in VTCM, then DMA it out to DDR so the
     * per-call cost below is a pure DMA (mirrors the doc08 Sec6 design:
     * WH baked incrementally at KV-append time, kept in host RAM). */
    uint64_t t_wh_bake;
    {
      static hexdma_desc2d setup_desc __attribute__((aligned(128)));
      uint64_t s = READ_PCYCLES();
      int rc = bake_weight_resident_f16(vtcm_base, &p, N, W_rm);
      if (rc != AEE_SUCCESS) {
        printf("[mha_micro][%s] bake_weight_resident_f16 failed 0x%x\n", name, rc);
        free(out_res);
        free(wt_wh_ddr);
        goto done_k3;
      }
      {
        uint32_t rs = 262144u;
        while (p.wt_bytes % rs)
          rs >>= 1;
        hexdma_2d_vtcm_to_ddr(&setup_desc, wt_wh_ddr, vtcm_base + p.wt_off, rs, rs, rs,
                              p.wt_bytes / rs);
      }
      t_wh_bake = READ_PCYCLES() - s;
    }

    /* M2-a isolated measurement: JUST the ring-based activation DMA (no
     * compute, no output), averaged over the same iteration count -- the
     * clean, directly-comparable number against milestone 1's 128 blocking
     * transfers (74.8us / ~3.5 GB/s for this shape's 256KB) and the weight
     * DMA's 11.7 GB/s on the same engine. */
    {
      const int R_ITERS = 20;
      uint64_t t_ring_only = 0;
      for (int it = 0; it < R_ITERS; it++)
        t_ring_only += bench_activation_ring_only(vtcm_base, &p, kcache_ddr, MHA_NCH, K);
      const size_t act_bytes = (size_t)p.row_bands * p.k_tiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;
      double us = (double)t_ring_only / R_ITERS;
      double act_gbps = us > 0.0 ? ((double)act_bytes / (us * 1.0e-6)) / 1.0e9 : 0.0;
      printf("  === M2-a: activation DMA, ring-only (isolated, no compute/output) ===\n");
      printf("    ring push+drain (%u bands x %u k-tiles, %zu bytes) %9.1f us   GB/s=%.2f\n",
             (unsigned)p.row_bands, (unsigned)p.k_tiles, act_bytes, us, act_gbps);
      printf("MHA_ACT_RING_US %.1f\n", us);
      printf("MHA_ACT_RING_GBPS %.2f\n", act_gbps);
    }

    const int D_ITERS = 20;
    uint64_t t_weight = 0, t_act_dma = 0, t_act_layout = 0, t_mm = 0, t_accread = 0, t_ahtorm = 0,
             t_outdma = 0, t_ringdrain = 0, t_wall = 0, ws;
    int drc = AEE_SUCCESS;
    for (int it = 0; it < D_ITERS; it++) {
      memset(out_res, 0, (size_t)M * N * sizeof(_Float16));
      ws = READ_PCYCLES();
      drc = matmul_micro_dma_f16(vtcm_base, vtcm_size, &p, M, N, K, out_res, kcache_ddr, MHA_NCH,
                                 wt_wh_ddr, &t_weight, &t_act_dma, &t_act_layout, &t_mm,
                                 &t_accread, &t_ahtorm, &t_outdma, &t_ringdrain);
      t_wall += READ_PCYCLES() - ws;
      if (drc != AEE_SUCCESS) {
        printf("[mha_micro][%s] kernel3 FAIL rc=0x%x\n", name, drc);
        break;
      }
    }
    if (drc == AEE_SUCCESS) {
      err_stats e = err_stats_f16(M * N, ref, out_res); /* OUTSIDE the timed loop */
      double wgt = (double)t_weight / D_ITERS, ad = (double)t_act_dma / D_ITERS,
             al = (double)t_act_layout / D_ITERS, mm = (double)t_mm / D_ITERS,
             ar = (double)t_accread / D_ITERS, ah = (double)t_ahtorm / D_ITERS,
             od = (double)t_outdma / D_ITERS, rd = (double)t_ringdrain / D_ITERS,
             tot = (double)t_wall / D_ITERS;
      /* GB/s = bytes / seconds / 1e9; wgt is in us, so seconds = wgt*1e-6 */
      double gbps = wgt > 0.0 ? ((double)p.wt_bytes / (wgt * 1.0e-6)) / 1.0e9 : 0.0;
      printf("  === micro-API + user-DMA (PIPELINED, ring-based act+out), (M=%u N=%u K=%u) ===\n",
             (unsigned)M, (unsigned)N, (unsigned)K);
      printf("    WH bake (rm_to_wh_f16 resident + DMA-out, ONE-TIME) %9.1f us\n",
             (double)t_wh_bake);
      printf("    Steady state (weight re-staged per call)   %9.1f   <- per-call (wall)\n", tot);
      printf("      [phase sum w/o overlap = %.1f]\n", wgt + ad + al + mm + ar + ah + od + rd);
      printf("      weight DDR->VTCM (USER-DMA)     %9.1f   bytes=%u  GB/s=%.2f\n", wgt,
             (unsigned)p.wt_bytes, gbps);
      printf("      activation DMA (ring push, async) %9.1f\n", ad);
      printf("      activation layout (rm_to_ah_f16) %9.1f\n", al);
      printf("      DSP matmul (HMX mm_f16)          %9.1f\n", mm);
      printf("      acc_read (HMX acc->VTCM AH)      %9.1f\n", ar);
      printf("      ah_to_rm (AH->flat, scalar)      %9.1f\n", ah);
      printf("      output write-back (ring push)    %9.1f\n", od);
      printf("      ring_drain (wait: next-band act + this-band output) %9.1f\n", rd);
      printf("    correctness: maxAbsErr=%.6g maxRelErr=%.6g meanRelErr=%.6g (idx=%ld)\n",
             e.max_abs, e.max_rel, e.mean_rel, e.max_rel_idx);
      printf("MHA_DMA_US %.1f\n", tot);
      printf("MHA_GBPS %.2f\n", gbps);
      printf("MHA_DMA_MAXRELERR %.6g\n", e.max_rel);
      printf("MHA_DMA_CORRECT %s\n", e.max_rel < 5e-2 ? "OK" : "WARN");
      /* per-head number, OLD per-tile DDR geometry (pre-M2-i) -- kept as an
       * explicitly-labeled comparison column, NOT the headline (see the
       * layer-wide-DMA block right below for that). */
      printf("MHA_FIELD shape=%s field=dma_pertile_us_per_head value=%.1f\n", name, tot);
    }
    /* ---- M2-i: layer-wide K-cache DMA + VTCM-resident repack (two variants),
     * then the real, verified per-layer total for all MHA_NCH kv_heads ---- */
    {
      const size_t layer_bytes = (size_t)M * MHA_NCH * K * sizeof(_Float16);
      const uint32_t reserve = (uint32_t)layer_bytes + 4096u;
      const uint32_t config_off_ = vtcm_size - hexkl_micro_hmx_config_size();
      if (reserve > config_off_) {
        printf("  [M2-i] layer (%zu bytes) does not fit VTCM, skipping\n", layer_bytes);
        goto done_m2i;
      }
      const uint32_t layer_kv_off =
        align_up(config_off_ - reserve, HEXKL_HMX_ACTIVATION_ALIGNMENT);

      /* one descriptor, the WHOLE layer DDR->VTCM: row_size=256KiB (top of
       * the M2-f curve, ~67 GB/s, and within the <=256KB single-row limit),
       * nrows = layer_bytes/256KiB (8 for this shape). */
      static hexdma_desc2d layer_desc __attribute__((aligned(128)));
      const uint32_t rs = 262144u;
      double layer_us, layer_gbps;
      {
        uint64_t t = 0;
        const int LIT = 20;
        for (int it = 0; it < LIT; it++) {
          uint64_t s = READ_PCYCLES();
          hexdma_2d_ddr_to_vtcm(&layer_desc, vtcm_base + layer_kv_off, kcache_ddr, rs, rs, rs,
                                (uint32_t)(layer_bytes / rs));
          t += READ_PCYCLES() - s;
        }
        layer_us = (double)t / LIT;
        layer_gbps = layer_us > 0.0 ? ((double)layer_bytes / (layer_us * 1.0e-6)) / 1.0e9 : 0.0;
      }
      printf("\n  === M2-i: layer-wide K-cache DMA (one descriptor, %zu bytes, row=%u nrows=%u) ===\n",
             layer_bytes, (unsigned)rs, (unsigned)(layer_bytes / rs));
      printf("    layer DDR->VTCM (USER-DMA)      %9.1f us   GB/s=%.2f\n", layer_us, layer_gbps);
      printf("MHA_LAYER_DMA_US %.1f\n", layer_us);
      printf("MHA_LAYER_DMA_GBPS %.2f\n", layer_gbps);

      /* isolated per-head repack measurement, both variants, one kv_head */
      {
        const int RIT = 20;
        uint64_t t_a = 0, t_b = 0;
        for (int it = 0; it < RIT; it++)
          t_a += bench_repack_ring_only(vtcm_base, &p, layer_kv_off, MHA_NCH, K, /*head_n=*/0);
        for (int it = 0; it < RIT; it++)
          t_b += bench_repack_scalar_only(vtcm_base, &p, layer_kv_off, MHA_NCH, K, M, /*head_n=*/0);
        const size_t repack_bytes = (size_t)p.row_bands * p.k_tiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;
        double us_a = (double)t_a / RIT, us_b = (double)t_b / RIT;
        double gbps_a = us_a > 0.0 ? ((double)repack_bytes / (us_a * 1.0e-6)) / 1.0e9 : 0.0;
        double gbps_b = us_b > 0.0 ? ((double)repack_bytes / (us_b * 1.0e-6)) / 1.0e9 : 0.0;
        printf("  === M2-i: per-head repack, one kv_head, %zu bytes, two variants ===\n",
               repack_bytes);
        printf("    (a) VTCM->VTCM ring DMA per tile      %9.1f us   GB/s=%.2f\n", us_a, gbps_a);
        printf("    (b) scalar copy_submatrix_to_f16      %9.1f us   GB/s=%.2f\n", us_b, gbps_b);
        printf("MHA_REPACK_A_US %.1f\n", us_a);
        printf("MHA_REPACK_A_GBPS %.2f\n", gbps_a);
        printf("MHA_REPACK_B_US %.1f\n", us_b);
        printf("MHA_REPACK_B_GBPS %.2f\n", gbps_b);
      }

      /* full per-layer pipeline, all MHA_NCH kv_heads, variant (a) integrated
       * (the ring-based repack -- variant (b) is scalar/synchronous and was
       * only measured in isolation above, per the task's scoping). Timed
       * over LAYER_ITERS repeats of the whole 8-head sweep; correctness is
       * checked in a SEPARATE, untimed final pass (never inside a timed
       * window). */
      {
        const int LAYER_ITERS = 5;
        uint64_t t_layer_sum = 0;
        /* M2-j: phase-breakdown accumulation across every head x iteration,
         * so a before/after comparison of the N=64->N_eff padding fix is
         * measured, not guessed. */
        uint64_t pt_weight = 0, pt_act_dma = 0, pt_act_layout = 0, pt_mm = 0, pt_accread = 0,
                 pt_ahtorm = 0, pt_outdma = 0, pt_ringdrain = 0;
        const uint32_t n_calls = (uint32_t)LAYER_ITERS * MHA_NCH;
        for (int it = 0; it < LAYER_ITERS; it++) {
          uint64_t t_this_iter = 0;
          for (uint32_t head_n = 0; head_n < MHA_NCH; head_n++) {
            uint64_t t_weight = 0, t_act_dma = 0, t_act_layout = 0, t_mm = 0, t_accread = 0,
                     t_ahtorm = 0, t_outdma = 0, t_ringdrain = 0;
            uint64_t hs = READ_PCYCLES();
            int hrc = matmul_layer_dma_f16(vtcm_base, vtcm_size, &p, M, N, K, out_res, layer_kv_off,
                                           MHA_NCH, head_n, wt_wh_ddr, &t_weight, &t_act_dma,
                                           &t_act_layout, &t_mm, &t_accread, &t_ahtorm, &t_outdma,
                                           &t_ringdrain);
            t_this_iter += READ_PCYCLES() - hs;
            pt_weight += t_weight;
            pt_act_dma += t_act_dma;
            pt_act_layout += t_act_layout;
            pt_mm += t_mm;
            pt_accread += t_accread;
            pt_ahtorm += t_ahtorm;
            pt_outdma += t_outdma;
            pt_ringdrain += t_ringdrain;
            if (hrc != AEE_SUCCESS)
              printf("  [M2-i] iter=%d head=%u FAIL rc=0x%x\n", it, (unsigned)head_n, hrc);
          }
          t_layer_sum += t_this_iter;
        }
        double avg_heads_us = (double)t_layer_sum / LAYER_ITERS;
        double per_layer_us = avg_heads_us + layer_us;
        printf("    [per-head phase breakdown, averaged over %u calls]\n", (unsigned)n_calls);
        printf("      weight DDR->VTCM       %9.3f us\n", (double)pt_weight / n_calls);
        printf("      repack (variant a, ring VTCM->VTCM) %9.3f us\n", (double)pt_act_dma / n_calls);
        printf("      activation layout (rm_to_ah_f16)    %9.3f us\n", (double)pt_act_layout / n_calls);
        printf("      DSP matmul (HMX mm_f16)             %9.3f us\n", (double)pt_mm / n_calls);
        printf("      acc_read                            %9.3f us\n", (double)pt_accread / n_calls);
        printf("      ah_to_rm                             %9.3f us\n", (double)pt_ahtorm / n_calls);
        printf("      output write-back (ring push)        %9.3f us\n", (double)pt_outdma / n_calls);
        printf("      ring_drain (wait)                    %9.3f us\n", (double)pt_ringdrain / n_calls);

        /* correctness: a SEPARATE, untimed final pass over every head */
        double max_rel_seen = 0.0;
        int all_ok = 1;
        for (uint32_t head_n = 0; head_n < MHA_NCH; head_n++) {
          uint64_t tw = 0, ta = 0, tal = 0, tm = 0, tar = 0, tah = 0, to = 0, trd = 0;
          memset(out_res, 0, (size_t)M * N * sizeof(_Float16));
          int hrc = matmul_layer_dma_f16(vtcm_base, vtcm_size, &p, M, N, K, out_res, layer_kv_off,
                                         MHA_NCH, head_n, wt_wh_ddr, &tw, &ta, &tal, &tm, &tar, &tah,
                                         &to, &trd);
          if (hrc != AEE_SUCCESS) {
            all_ok = 0;
            continue;
          }
          err_stats eh = err_stats_f16(M * N, ref, out_res);
          if (eh.max_rel > max_rel_seen)
            max_rel_seen = eh.max_rel;
          if (eh.max_rel >= 5e-2)
            all_ok = 0;
        }

        printf("  === M2-i: FULL per-layer pipeline, all %u kv_heads, variant (a) integrated ===\n",
               (unsigned)MHA_NCH);
        printf("    layer DMA (one-time)                  %9.1f us\n", layer_us);
        printf("    sum of %u per-head pipelines (avg of %d) %9.1f us\n", (unsigned)MHA_NCH,
               LAYER_ITERS, avg_heads_us);
        printf("    PER-LAYER TOTAL (all %u kv_heads)       %9.1f us   <- compare vs CPU 553-590us\n",
               (unsigned)MHA_NCH, per_layer_us);
        printf("    correctness: max relErr across heads = %.6g, all_ok=%s\n", max_rel_seen,
               all_ok ? "yes" : "NO");
        printf("MHA_LAYER_TOTAL_US %.1f\n", per_layer_us);
        printf("MHA_LAYER_TOTAL_CORRECT %s\n", all_ok ? "OK" : "WARN");
        /* THE headline for this shape: layer-wide DMA + VTCM repack +
         * pre-baked WH + n_tiles=ceil(gqa/32), all MHA_NCH kv_heads. Decode
         * has no incremental-bake requirement (steady state assumes an
         * already-populated, pre-baked cache from prior turns -- valid at
         * decode, NOT at prefill; see the prefill shapes for why), so
         * total_us_8heads == dma_layer_us_8heads here. */
        printf("MHA_FIELD shape=%s field=dma_layer_us_8heads value=%.1f\n", name, per_layer_us);
        printf("MHA_FIELD shape=%s field=total_us_8heads value=%.1f\n", name, per_layer_us);
        printf("MHA_FIELD shape=%s field=gbps value=%.2f\n", name, layer_gbps);
        printf("MHA_FIELD shape=%s field=maxrelerr value=%.6g\n", name, max_rel_seen);
        printf("MHA_FIELD shape=%s field=correct value=%s\n", name, all_ok ? "OK" : "WARN");
      }
    }
  done_m2i:;

    free(out_res);
    free(wt_wh_ddr);
  }
done_k3:;

done0:
  free(X_ref_flat);
  free(Q_nat);
  free(W_rm);
  free(ref);
  free(kcache_ddr);
}

/* =====================================================================
 * M2-f: is 11.5 GB/s a per-engine ceiling or a descriptor-shape artifact?
 * Standalone -- no correctness coupling, just timed DMA transfers of a
 * throwaway DDR buffer into a VTCM scratch region. Two axes:
 *   (A) ONE blocking descriptor (single dmstart+dmwait) moving the full
 *       256 KiB, at several (row_size, nrows) splits -- does a single
 *       descriptor's internal shape matter?
 *   (B) the SAME 256 KiB chained via the ring into C equal descriptors
 *       (one dmstart, C-1 dmlinks, one drain) -- does chained descriptor
 *       COUNT change achieved GB/s?
 * =================================================================== */
static void bench_dma_bw_ceiling(uint8_t *vtcm_base, uint32_t vtcm_size) {
  const size_t total = 256u * 1024u; /* matches shape #1's activation total */
  const uint32_t vtcm_off = 0;       /* scratch dest, well clear of config region */
  if ((uint32_t)total + 4096u > vtcm_size) {
    printf("[M2-f] vtcm too small, skipping\n");
    return;
  }
  uint8_t *ddr_src = (uint8_t *)malloc(total);
  if (!ddr_src) {
    printf("[M2-f] oom, skipping\n");
    return;
  }
  for (size_t i = 0; i < total; i++)
    ddr_src[i] = (uint8_t)i;

  printf("\n=== M2-f: DMA bandwidth ceiling (standalone, %zu bytes, no correctness) ===\n",
         total);

  /* (A) single blocking descriptor, several (row_size, nrows) splits of the
   * same 256 KiB. row_size kept <= 256 KiB per the known single-row-DMA HW
   * limit (a single row above ~512 KiB silently corrupts). */
  {
    static hexdma_desc2d d __attribute__((aligned(128)));
    struct {
      const char *name;
      uint32_t row_size, nrows;
    } cases[] = {
      {"1desc row=256KiB nrows=1", 262144u, 1u},   {"1desc row=64KiB  nrows=4", 65536u, 4u},
      {"1desc row=16KiB  nrows=16", 16384u, 16u},  {"1desc row=2KiB   nrows=128", 2048u, 128u},
      {"1desc row=64B    nrows=4096 (our per-tile row width)", 64u, 4096u},
    };
    const int IT = 20;
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
      uint64_t t = 0;
      for (int it = 0; it < IT; it++) {
        uint64_t s = READ_PCYCLES();
        hexdma_2d_ddr_to_vtcm(&d, vtcm_base + vtcm_off, ddr_src, cases[c].row_size,
                              cases[c].row_size, cases[c].row_size, cases[c].nrows);
        t += READ_PCYCLES() - s;
      }
      double us = (double)t / IT;
      double gbps = us > 0.0 ? ((double)total / (us * 1.0e-6)) / 1.0e9 : 0.0;
      printf("  [A] %-46s %8.1f us   %6.2f GB/s\n", cases[c].name, us, gbps);
      printf("MHA_BW_A_GBPS %.2f\n", gbps);
    }
  }

  /* (B) same bytes, chained via the ring into C equal descriptors -- ONE
   * dmstart, C-1 dmlinks, ONE drain. row_size fixed at 2048B (our own
   * geometry) so the only variable is descriptor COUNT. */
  {
    const uint32_t chunk_counts[] = {1, 2, 4, 8, 16, 32, 64, 128};
    const int IT = 20;
    for (size_t c = 0; c < sizeof(chunk_counts) / sizeof(chunk_counts[0]); c++) {
      const uint32_t C = chunk_counts[c];
      if (total % C)
        continue;
      const uint32_t chunk_bytes = (uint32_t)(total / C);
      uint64_t t = 0;
      for (int it = 0; it < IT; it++) {
        uint64_t s = READ_PCYCLES();
        ring_init();
        for (uint32_t i = 0; i < C; i++)
          ring_push2d(vtcm_base + vtcm_off + (size_t)i * chunk_bytes, ddr_src + (size_t)i * chunk_bytes,
                     chunk_bytes, chunk_bytes, chunk_bytes, 1, /*src_vtcm=*/0, /*dst_vtcm=*/1);
        ring_drain();
        t += READ_PCYCLES() - s;
      }
      double us = (double)t / IT;
      double gbps = us > 0.0 ? ((double)total / (us * 1.0e-6)) / 1.0e9 : 0.0;
      printf("  [B] ring, %3u desc x %6u B          %8.1f us   %6.2f GB/s\n", (unsigned)C,
             (unsigned)chunk_bytes, us, gbps);
      printf("MHA_BW_B_GBPS %.2f\n", gbps);
    }
  }

  free(ddr_src);
}

/*!
  @brief M2-j: does narrowing the output DMA to the valid gqa columns help or
  hurt? "measure, don't assume" -- the M2-f sweep says small rows are slow,
  and narrowing row_size from 32*2=64B down to gqa*2=16B makes the row
  SMALLER, not larger. Standalone, no correctness coupling: 32 output tiles
  (matching row_bands for shape #1), each variant via its own ring
  push+drain, same VTCM source tile reused (data content irrelevant --
  pure DMA-mechanics timing).
*/
static void bench_output_narrow_vs_full(uint8_t *vtcm_base, uint32_t vtcm_size, uint32_t gqa) {
  const uint32_t ROW_BANDS = 32; /* matches shape #1's M=1024 -> 32 bands */
  const uint32_t vtcm_src_off = 0; /* reuse whatever's resident -- content irrelevant for a DMA-only timing test */
  if (vtcm_size < 4096) {
    printf("[M2-j] vtcm too small, skipping output narrow-vs-full\n");
    return;
  }
  uint8_t *ddr_full = (uint8_t *)malloc((size_t)ROW_BANDS * HEXKL_HMX_F16_BLOCK_N_ROW *
                                       HEXKL_HMX_F16_BLOCK_N_COL * sizeof(_Float16));
  uint8_t *ddr_narrow =
    (uint8_t *)malloc((size_t)ROW_BANDS * HEXKL_HMX_F16_BLOCK_N_ROW * gqa * sizeof(_Float16));
  if (!ddr_full || !ddr_narrow) {
    printf("[M2-j] oom, skipping output narrow-vs-full\n");
    free(ddr_full);
    free(ddr_narrow);
    return;
  }

  printf("\n=== M2-j: output write-back, narrow (gqa=%u cols) vs full (32 cols) ===\n",
         (unsigned)gqa);
  const int IT = 20;

  /* full: row_size=64B (all 32 cols), matches the current default path */
  {
    uint64_t t = 0;
    for (int it = 0; it < IT; it++) {
      uint64_t s = READ_PCYCLES();
      ring_init();
      for (uint32_t band = 0; band < ROW_BANDS; band++) {
        uint8_t *dst = ddr_full + (size_t)band * HEXKL_HMX_F16_BLOCK_N_ROW *
                                    HEXKL_HMX_F16_BLOCK_N_COL * sizeof(_Float16);
        ring_push2d(dst, vtcm_base + vtcm_src_off,
                   HEXKL_HMX_F16_BLOCK_N_COL * (uint32_t)sizeof(_Float16),
                   HEXKL_HMX_F16_BLOCK_N_COL * (uint32_t)sizeof(_Float16),
                   HEXKL_HMX_F16_BLOCK_N_COL * (uint32_t)sizeof(_Float16), HEXKL_HMX_F16_BLOCK_N_ROW,
                   /*src_vtcm=*/1, /*dst_vtcm=*/0);
      }
      ring_drain();
      t += READ_PCYCLES() - s;
    }
    double us = (double)t / IT;
    printf("  [full 32-wide]   %2u tiles x 64B rows        %8.2f us\n", (unsigned)ROW_BANDS, us);
    printf("MHA_OUT_FULL_US %.2f\n", us);
  }

  /* narrow: row_size=gqa*2 bytes -- only the valid columns, src_stride still
   * 64B so the DMA itself discards the padding columns (no separate compact
   * step). */
  {
    uint64_t t = 0;
    for (int it = 0; it < IT; it++) {
      uint64_t s = READ_PCYCLES();
      ring_init();
      for (uint32_t band = 0; band < ROW_BANDS; band++) {
        uint8_t *dst = ddr_narrow + (size_t)band * HEXKL_HMX_F16_BLOCK_N_ROW * gqa * sizeof(_Float16);
        ring_push2d(dst, vtcm_base + vtcm_src_off, gqa * (uint32_t)sizeof(_Float16),
                   HEXKL_HMX_F16_BLOCK_N_COL * (uint32_t)sizeof(_Float16), gqa * (uint32_t)sizeof(_Float16),
                   HEXKL_HMX_F16_BLOCK_N_ROW, /*src_vtcm=*/1, /*dst_vtcm=*/0);
      }
      ring_drain();
      t += READ_PCYCLES() - s;
    }
    double us = (double)t / IT;
    printf("  [narrow gqa-wide] %2u tiles x %uB rows        %8.2f us\n", (unsigned)ROW_BANDS,
           (unsigned)(gqa * sizeof(_Float16)), us);
    printf("MHA_OUT_NARROW_US %.2f\n", us);
  }

  free(ddr_full);
  free(ddr_narrow);
}

/* =====================================================================
 * M2-k: P.V decode (spec shapes #4/#5), with the mandatory fp16-accumulator
 * accuracy report. P.V swaps roles vs Q.K^T: V carries the GQA stride (same
 * [kv_len][nch][head_dim] layer layout as the K cache -- the layer-wide DMA
 * + VTCM->VTCM repack from M2-i applies directly), while P (softmax output,
 * already per-head, no GQA stride) is the small operand needing a
 * transpose. Concretely for the micro API's LHS(activation)/RHS(weight)
 * roles: P is "activation" (M=gqa padded to 32, K=kv_len), V is "weight"
 * (K=kv_len rows, N=head_dim=128 cols) -- opposite of Q.K^T, where K-cache
 * was the activation. So V needs the repack-then-bake treatment (variant
 * (a) only -- variant (b) is already a proven 400x-slower dead end from
 * M2-i, not worth re-testing here), and P just needs one small,
 * once-per-head scalar stage (it is NOT GQA-strided, so no DMA/repack is
 * needed for it at all).
 * ===================================================================== */

/*! @brief True fp16-ACCUMULATING scalar reference (rounds the running sum to
    fp16 after every term, not just once at the end) -- isolates "fp16
    accumulation is inherently lossy at this depth" from "HMX is wrong". The
    only pre-existing scalar references in hexkl_test.h (_cm_ and _rm_) both
    accumulate in float/double and round once at the very end; this matters
    when K reaches 512-1024 terms. */
static void matmul_fp16acc_ref(uint32_t M, uint32_t N, uint32_t K, float *out,
                               const _Float16 *X, const _Float16 *W) {
  for (uint32_t m = 0; m < M; m++) {
    for (uint32_t n = 0; n < N; n++) {
      _Float16 acc = (_Float16)0.0f;
      for (uint32_t k = 0; k < K; k++) {
        float prod = (float)X[m * K + k] * (float)W[k * N + n];
        acc = (_Float16)((float)acc + prod);
      }
      out[m * N + n] = (float)acc;
    }
  }
}

/*! @brief Softmax-like activation: P_t is [M_pad, K] row-major; rows
    [0,gqa) are a real convex combination (non-negative, sum to 1 across K --
    the actual numerical regime for attention weights), rows [gqa, M_pad) are
    left zero (caller must calloc). Deterministic (no <random> dependency on
    the DSP side), but not a ramp -- every row has a different distribution
    over K via the g-dependent term. */
static void gen_softmax_like_pv(uint32_t M_pad, uint32_t gqa, uint32_t K, _Float16 *P_t) {
  for (uint32_t g = 0; g < gqa && g < M_pad; g++) {
    float sum = 0.0f;
    for (uint32_t k = 0; k < K; k++)
      sum += (float)(((k * 7 + g * 13 + 3) % 97) + 1);
    for (uint32_t k = 0; k < K; k++) {
      float raw = (float)(((k * 7 + g * 13 + 3) % 97) + 1);
      P_t[g * K + k] = (_Float16)(raw / sum);
    }
  }
}

/*! @brief Weight-side (V) bake -- REVISED after measuring the first attempt.
    The first attempt (VTCM->VTCM ring repack into a flat scratch, then
    rm_to_wh_f16 reading that VTCM scratch) cost ~3378us for 128 tiles
    (~26.4us/tile) -- statistically identical to M2-i's variant (b) disaster
    (copy_submatrix_to_f16 from VTCM, ~26.4us/tile). That is not a
    coincidence: it shows the micro API's "extraction-style" scalar
    functions that take row_tile/col_tile/input_rows/cols addressing
    (copy_submatrix_to_f16, and evidently rm_to_wh_f16 too) are ~10x+ slower
    reading VTCM than DDR, regardless of which one it is -- unlike the
    "pure format" functions (rm_to_ah_f16/ah_to_rm_f16, fixed 32x32, no
    larger-matrix addressing), which are fast either way (that is exactly
    why the M2-i activation-side repack worked: it uses ONLY ring_push2d +
    rm_to_ah_f16, never an extraction-style function on a VTCM source).
    There is no flat/format-only WH-bake in this header, so the fix is: one
    bulk VTCM->DDR extraction of this head's whole [K,N] slice (GQA-strided
    source, contiguous dest -- same shape as the ORIGINAL per-head K-cache
    pull from milestone 1), then bake ALL tiles from that DDR buffer with
    row_tile/col_tile addressing, matching kernel 2's PROVEN-fast usage
    (~0.83us/call there) instead of the ~26.4us/call VTCM-source disaster. */
static int bake_weight_from_layer_pv(uint8_t *vtcm_base, const vtcm_plan_f16 *p,
                                     uint32_t layer_kv_off, uint32_t nch, uint32_t n, uint32_t k,
                                     uint32_t head_n, uint8_t *v_head_ddr) {
  static hexdma_desc2d d __attribute__((aligned(128)));
  const uint32_t src_stride = nch * n * (uint32_t)sizeof(_Float16);
  const uint8_t *vtcm_src = vtcm_base + layer_kv_off + head_n * n * (uint32_t)sizeof(_Float16);
  /* one bulk 2D DMA: row_size=n*2 (256B for head_dim=128), src_stride=GQA
   * stride, dst contiguous, nrows=k (up to 1024) -- well under the <=256KB
   * single-row limit; total bytes (256KB at k=1024) match the original
   * per-head K-cache pull's magnitude. */
  hexdma_2d_vtcm_to_ddr(&d, v_head_ddr, vtcm_src, n * (uint32_t)sizeof(_Float16), src_stride,
                       n * (uint32_t)sizeof(_Float16), k);

  for (uint32_t kt = 0; kt < p->k_tiles; kt++) {
    for (uint32_t nt = 0; nt < p->n_tiles; nt++) {
      int rc = hexkl_micro_hmx_rm_to_wh_f16(vtcm_base,
                                            p->wt_off + (kt * p->n_tiles + nt) * WT_TILE_BYTES_F16,
                                            (const _Float16 *)v_head_ddr, kt, nt, n);
      if (rc != AEE_SUCCESS)
        return rc;
    }
  }
  return AEE_SUCCESS;
}

/*! @brief M2-l: activation-side (P) staging, DMA-based -- P is supplied
    ALREADY TRANSPOSED as a contiguous [gqa_pad, kv] fp16 buffer (the layout
    change doc08's softmax owner would make; free on the CPU side, since the
    softmax writer chooses its own write pattern). Unlike the first attempt
    (scalar copy_submatrix_to_f16, ~2.83us/tile -- not the VTCM disaster, but
    still the extraction-style path when a plain DMA suffices), this is a
    per-k-tile 2D DMA (P is NOT GQA-strided, so no per-head offset needed)
    landing flat tiles, then ONLY the pure-format rm_to_ah_f16 -- the same
    recipe that made the M2-a/M2-i activation paths fast. */
static void push_pv_activation(uint8_t *vtcm_base, const vtcm_plan_f16 *p, const _Float16 *P_host,
                               uint32_t k) {
  const uint32_t src_stride = k * (uint32_t)sizeof(_Float16); /* P_host row width = k (contiguous) */
  for (uint32_t kt = 0; kt < p->k_tiles; kt++) {
    const uint8_t *asrc = (const uint8_t *)P_host + kt * HEXKL_HMX_F16_BLOCK_N_INNER * sizeof(_Float16);
    ring_push2d(vtcm_base + p->act_flat_off[0] + HEXKL_HMX_ACTIVATION_ALIGNMENT * kt, asrc,
               HEXKL_HMX_F16_BLOCK_N_INNER * (uint32_t)sizeof(_Float16) /* dst_stride */, src_stride,
               HEXKL_HMX_F16_BLOCK_N_INNER * (uint32_t)sizeof(_Float16) /* row_size */,
               HEXKL_HMX_F16_BLOCK_N_ROW /* nrows = all 32 (padded) gqa rows */, /*src_vtcm=*/0,
               /*dst_vtcm=*/1);
  }
}
static int stage_pv_activation_dma(uint8_t *vtcm_base, const vtcm_plan_f16 *p, const _Float16 *P_host,
                                   uint32_t k) {
  ring_init();
  push_pv_activation(vtcm_base, p, P_host, k);
  ring_drain();
  return convert_band_to_ah(vtcm_base, p, 0); /* pure-format only, slot 0 */
}

/*! @brief M2-l: rough (not optimized) DSP-side transpose of a [kv,gqa]
    16-32KB slab into [gqa,kv], for the "what does it cost if we CANNOT get
    the pre-transposed layout from the softmax owner" question. Plain scalar
    nested loop, no attempt at vectorizing -- a price tag, not a kernel. */
static uint64_t bench_pv_rough_transpose(uint32_t kv, uint32_t gqa) {
  _Float16 *P_nat = (_Float16 *)malloc((size_t)kv * gqa * sizeof(_Float16));           /* [kv,gqa] */
  volatile _Float16 *P_tr = (volatile _Float16 *)malloc((size_t)gqa * kv * sizeof(_Float16)); /* [gqa,kv] */
  if (!P_nat || !P_tr) {
    free(P_nat);
    free((void *)P_tr);
    return 0;
  }
  for (size_t i = 0; i < (size_t)kv * gqa; i++)
    P_nat[i] = (_Float16)((float)(i % 13) * 0.01f);
  /* P_tr is volatile so the compiler cannot prove these stores are dead and
   * elide the whole loop (the first version of this benchmark measured
   * ~0us -- exactly that: an unread destination buffer optimized away
   * entirely under -O3). */
  uint64_t s = READ_PCYCLES();
  for (uint32_t row = 0; row < kv; row++)
    for (uint32_t g = 0; g < gqa; g++)
      P_tr[g * kv + row] = P_nat[row * gqa + g];
  uint64_t t = READ_PCYCLES() - s;
  free(P_nat);
  free((void *)P_tr);
  return t;
}

/*! @brief M2-l: one-time per-head setup -- extract+bake V into VTCM resident
    WH (bake_weight_from_layer_pv, already fixed to use a DDR-source bake),
    then DMA the resident WH OUT to a per-head DDR buffer. This is the
    "pre-baked WH slab in DDR" spec 5b asked for -- run ONCE outside the
    timed loop, not per token. */
static int prebake_pv_weight(uint8_t *vtcm_base, const vtcm_plan_f16 *p, uint32_t layer_kv_off,
                             uint32_t nch, uint32_t n, uint32_t k, uint32_t head_n,
                             uint8_t *v_head_ddr, uint8_t *wt_wh_ddr_out) {
  int rc = bake_weight_from_layer_pv(vtcm_base, p, layer_kv_off, nch, n, k, head_n, v_head_ddr);
  if (rc != AEE_SUCCESS)
    return rc;
  static hexdma_desc2d d __attribute__((aligned(128)));
  uint32_t rs = 262144u;
  while (p->wt_bytes % rs)
    rs >>= 1;
  hexdma_2d_vtcm_to_ddr(&d, wt_wh_ddr_out, vtcm_base + p->wt_off, rs, rs, rs, p->wt_bytes / rs);
  return AEE_SUCCESS;
}

/*! @brief M2-l: steady-state weight restage -- the pre-baked WH slab already
    sits in host RAM (wt_wh_ddr); the per-token cost is a pure DMA back into
    VTCM resident, no re-bake at all. */
static void restage_pv_weight(uint8_t *vtcm_base, const vtcm_plan_f16 *p, const uint8_t *wt_wh_ddr) {
  static hexdma_desc2d d __attribute__((aligned(128)));
  uint32_t rs = 262144u;
  while (p->wt_bytes % rs)
    rs >>= 1;
  hexdma_2d_ddr_to_vtcm(&d, vtcm_base + p->wt_off, wt_wh_ddr, rs, rs, rs, p->wt_bytes / rs);
}

/*! @brief M2-l: incremental bake -- ONE k-tile's n_tiles worth (the tiles a
    single new KV-append dirties), from an ALREADY-EXTRACTED per-head DDR V
    slice. Standalone measurement, not integrated into the steady-state loop
    (append cadence is conceptually separate from the per-token matmul). */
static uint64_t bench_incremental_bake_pv(uint8_t *vtcm_base, const vtcm_plan_f16 *p,
                                          const uint8_t *v_head_ddr, uint32_t n) {
  const uint32_t kt = p->k_tiles - 1; /* "current partial" tile -- arbitrary for timing */
  uint64_t s = READ_PCYCLES();
  for (uint32_t nt = 0; nt < p->n_tiles; nt++) {
    hexkl_micro_hmx_rm_to_wh_f16(vtcm_base, p->wt_off + (kt * p->n_tiles + nt) * WT_TILE_BYTES_F16,
                                 (const _Float16 *)v_head_ddr, kt, nt, n);
  }
  return READ_PCYCLES() - s;
}

/*! @brief Full per-head P.V pipeline, INTENDED design: P staged via DMA
    (pre-transposed input), V weight restaged from an already-pre-baked DDR
    WH slab (no re-bake), single-band mm loop, output via ring. */
static int matmul_pv_f16(uint8_t *vtcm_base, const vtcm_plan_f16 *p, uint32_t m, uint32_t n,
                         uint32_t k, _Float16 *out, const _Float16 *P_host, const uint8_t *wt_wh_ddr,
                         uint64_t *t_act, uint64_t *t_weight, uint64_t *t_mm, uint64_t *t_accread,
                         uint64_t *t_ahtorm, uint64_t *t_outdma) {
  uint64_t s;
  int rc;
  (void)m;

  s = READ_PCYCLES();
  rc = stage_pv_activation_dma(vtcm_base, p, P_host, k);
  if (rc != AEE_SUCCESS)
    return rc;
  *t_act += READ_PCYCLES() - s;

  s = READ_PCYCLES();
  restage_pv_weight(vtcm_base, p, wt_wh_ddr);
  *t_weight += READ_PCYCLES() - s;

  ring_init();
  for (uint32_t nt = 0; nt < p->n_tiles; nt++) {
    s = READ_PCYCLES();
    hexkl_micro_hmx_acc_clear_f16();
    for (uint32_t kt = 0; kt < p->k_tiles; kt++) {
      rc = hexkl_micro_hmx_mm_f16(vtcm_base, p->act_ah_off + HEXKL_HMX_ACTIVATION_ALIGNMENT * kt,
                                  p->wt_off + (kt * p->n_tiles + nt) * WT_TILE_BYTES_F16);
      if (rc != AEE_SUCCESS)
        return rc;
    }
    *t_mm += READ_PCYCLES() - s;

    s = READ_PCYCLES();
    rc = hexkl_micro_hmx_acc_read_f16(vtcm_base, p->config_off, p->acc_ah_off);
    if (rc != AEE_SUCCESS)
      return rc;
    *t_accread += READ_PCYCLES() - s;

    const uint32_t out_slot_off = p->out_flat_base + nt * HEXKL_HMX_ACTIVATION_ALIGNMENT;
    s = READ_PCYCLES();
    rc = hexkl_micro_hmx_ah_to_rm_f16(vtcm_base, out_slot_off, p->acc_ah_off);
    if (rc != AEE_SUCCESS)
      return rc;
    *t_ahtorm += READ_PCYCLES() - s;

    s = READ_PCYCLES();
    _Float16 *dst = out + (size_t)nt * HEXKL_HMX_F16_BLOCK_N_COL;
    ring_push2d(dst, vtcm_base + out_slot_off, n * (uint32_t)sizeof(_Float16),
               HEXKL_HMX_F16_BLOCK_N_COL * (uint32_t)sizeof(_Float16),
               HEXKL_HMX_F16_BLOCK_N_COL * (uint32_t)sizeof(_Float16), HEXKL_HMX_F16_BLOCK_N_ROW,
               /*src_vtcm=*/1, /*dst_vtcm=*/0);
    *t_outdma += READ_PCYCLES() - s;
  }
  s = READ_PCYCLES();
  ring_drain();
  *t_outdma += READ_PCYCLES() - s;
  return AEE_SUCCESS;
}

/*! @brief M2-k driver: builds data for one P.V shape (M=gqa padded to 32,
    N=head_dim=128, K=kv_len), runs the layer-wide V-cache DMA once, times
    the per-head pipeline across all MHA_NCH heads, and reports the mandatory
    accuracy comparison (fp32-accumulating ref AND fp16-accumulating scalar
    ref) in a separate, untimed pass. */
static void run_pv_shape_f16(uint8_t *vtcm_base, uint32_t vtcm_size, uint32_t hmx_config_offset,
                             const char *name, uint32_t K) {
  (void)hmx_config_offset;
  const uint32_t M = align_up(MHA_GQA, HEXKL_HMX_F16_BLOCK_N_ROW); /* 32 */
  const uint32_t N = 128; /* head_dim -- a real HW dimension, not padding waste (already 4*32) */
  if (M % 32 || N % 32 || K % 32) {
    printf("[pv][%s] M=%u N=%u K=%u not tile-aligned, skipping\n", name, (unsigned)M, (unsigned)N,
           (unsigned)K);
    return;
  }

  vtcm_plan_f16 p = plan_vtcm_f16(M, K, N, vtcm_size);
  const uint32_t need_local = p.out_flat_base + p.n_tiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;
  if (need_local > p.config_off) {
    printf("[pv][%s] local VTCM plan does not fit, skipping\n", name);
    return;
  }

  const size_t layer_bytes = (size_t)K * MHA_NCH * N * sizeof(_Float16); /* V layer: [kv][nch][hd] */
  const uint32_t reserve = (uint32_t)layer_bytes + 4096u;
  if (reserve > p.config_off || (uint32_t)(p.config_off - reserve) < need_local) {
    printf("[pv][%s] layer (%zu bytes) does not fit VTCM, skipping\n", name, layer_bytes);
    return;
  }
  const uint32_t layer_kv_off = align_up(p.config_off - reserve, HEXKL_HMX_ACTIVATION_ALIGNMENT);

  _Float16 *V_kn = (_Float16 *)malloc((size_t)K * N * sizeof(_Float16)); /* [K,N] reference weight */
  _Float16 *V_layer_ddr = (_Float16 *)malloc(layer_bytes);               /* [K][nch][N], all heads identical */
  _Float16 *P_t = (_Float16 *)calloc((size_t)M * K, sizeof(_Float16));   /* [gqa_pad,K], ALREADY TRANSPOSED */
  float *ref_fp32 = (float *)malloc((size_t)M * N * sizeof(float));
  float *ref_fp16acc = (float *)malloc((size_t)M * N * sizeof(float));
  _Float16 *out_res = (_Float16 *)malloc((size_t)M * N * sizeof(_Float16));
  uint8_t *v_head_ddr = (uint8_t *)malloc((size_t)K * N * sizeof(_Float16)); /* extraction scratch, reused per head */
  /* M2-l: one pre-baked WH slab per head, in host RAM, populated ONCE
   * outside the timed loop -- the "pre-baked WH slab in DDR" spec 5b asked
   * for. The steady-state loop below only DMAs from these, never re-bakes. */
  uint8_t *wt_wh_ddr[MHA_NCH] = {0};
  int wt_wh_ok = 1;
  for (uint32_t h = 0; h < MHA_NCH; h++) {
    wt_wh_ddr[h] = (uint8_t *)malloc(p.wt_bytes);
    if (!wt_wh_ddr[h])
      wt_wh_ok = 0;
  }
  if (!V_kn || !V_layer_ddr || !P_t || !ref_fp32 || !ref_fp16acc || !out_res || !v_head_ddr ||
      !wt_wh_ok) {
    printf("[pv][%s] oom, skipping\n", name);
    goto done;
  }

  hexkl_test_init_pattern_f16(K, N, V_kn, 7, 0.03f);
  for (uint32_t row = 0; row < K; row++)
    for (uint32_t h = 0; h < MHA_NCH; h++)
      for (uint32_t d = 0; d < N; d++)
        V_layer_ddr[(row * MHA_NCH + h) * N + d] = V_kn[row * N + d];

  /* P is supplied ALREADY TRANSPOSED, i.e. gen_softmax_like_pv already
   * produces the [gqa_pad, K] layout directly -- exactly what the softmax
   * owner would write if asked to change its write pattern (free on the CPU
   * side, per the coordinator's note). No on-DSP transpose is in the timed
   * path at all; see bench_pv_rough_transpose below for "what if we can't
   * get this layout" pricing. */
  gen_softmax_like_pv(M, MHA_GQA, K, P_t);

  hexkl_test_init_zero_f32(M * N, ref_fp32);
  hexkl_test_matmul_cm_f16f16_f32(M, N, K, N, ref_fp32, P_t, V_kn); /* fp32-accumulating ground truth */
  matmul_fp16acc_ref(M, N, K, ref_fp16acc, P_t, V_kn);              /* fp16-accumulating scalar twin */

  printf("\nMHA_SHAPE name=%s M=%u N=%u K=%u (P.V decode, M2-l intended design)\n", name, (unsigned)M,
         (unsigned)N, (unsigned)K);

  /* layer-wide V-cache DMA, one descriptor */
  double layer_us, layer_gbps;
  {
    static hexdma_desc2d layer_desc __attribute__((aligned(128)));
    uint32_t rs = 262144u;
    while (layer_bytes % rs)
      rs >>= 1;
    uint64_t t = 0;
    const int LIT = 20;
    for (int it = 0; it < LIT; it++) {
      uint64_t s = READ_PCYCLES();
      hexdma_2d_ddr_to_vtcm(&layer_desc, vtcm_base + layer_kv_off, V_layer_ddr, rs, rs, rs,
                            (uint32_t)(layer_bytes / rs));
      t += READ_PCYCLES() - s;
    }
    layer_us = (double)t / LIT;
    layer_gbps = layer_us > 0.0 ? ((double)layer_bytes / (layer_us * 1.0e-6)) / 1.0e9 : 0.0;
    printf("  layer-wide V-cache DMA (one descriptor, %zu bytes) %9.1f us   GB/s=%.2f\n",
           layer_bytes, layer_us, layer_gbps);
  }

  /* ONE-TIME setup (NOT in the steady-state timed loop): pre-bake every
   * head's WH slab into host RAM. Timed anyway so the "full bake" cost is
   * quantified and reported, per spec 5b / the coordinator's ask. */
  double full_bake_us_per_head;
  {
    uint64_t t = 0;
    for (uint32_t head_n = 0; head_n < MHA_NCH; head_n++) {
      uint64_t s = READ_PCYCLES();
      int rc = prebake_pv_weight(vtcm_base, &p, layer_kv_off, MHA_NCH, N, K, head_n, v_head_ddr,
                                 wt_wh_ddr[head_n]);
      t += READ_PCYCLES() - s;
      if (rc != AEE_SUCCESS)
        printf("  [pv][%s] prebake head=%u FAIL rc=0x%x\n", name, (unsigned)head_n, rc);
    }
    full_bake_us_per_head = (double)t / MHA_NCH;
  }

  /* incremental bake: one k-tile's worth (n_tiles), from the LAST head's
   * already-extracted v_head_ddr (left populated by prebake above) --
   * standalone, not part of the steady-state loop. */
  double incr_bake_us;
  {
    const int IIT = 20;
    uint64_t t = 0;
    for (int it = 0; it < IIT; it++)
      t += bench_incremental_bake_pv(vtcm_base, &p, v_head_ddr, N);
    incr_bake_us = (double)t / IIT;
  }

  /* rough (unoptimized) cost of a DSP-side [kv,gqa]->[gqa,kv] transpose, for
   * "what if we can't get the pre-transposed layout from the softmax
   * owner" -- a price tag, not something integrated into the timed path. */
  double rough_transpose_us;
  {
    const int TIT = 20;
    uint64_t t = 0;
    for (int it = 0; it < TIT; it++)
      t += bench_pv_rough_transpose(K, MHA_GQA);
    rough_transpose_us = (double)t / TIT;
  }

  /* timed per-head sweep: STEADY STATE ONLY -- P staged via DMA, V weight
   * DMA-restaged from the already-pre-baked wt_wh_ddr[head], no bake at all. */
  const int LAYER_ITERS = 5;
  uint64_t t_layer_sum = 0;
  uint64_t pt_act = 0, pt_weight = 0, pt_mm = 0, pt_accread = 0, pt_ahtorm = 0, pt_outdma = 0;
  const uint32_t n_calls = (uint32_t)LAYER_ITERS * MHA_NCH;
  for (int it = 0; it < LAYER_ITERS; it++) {
    uint64_t t_this_iter = 0;
    for (uint32_t head_n = 0; head_n < MHA_NCH; head_n++) {
      uint64_t t_act = 0, t_weight = 0, t_mm = 0, t_accread = 0, t_ahtorm = 0, t_outdma = 0;
      uint64_t hs = READ_PCYCLES();
      int hrc = matmul_pv_f16(vtcm_base, &p, M, N, K, out_res, P_t, wt_wh_ddr[head_n], &t_act,
                              &t_weight, &t_mm, &t_accread, &t_ahtorm, &t_outdma);
      t_this_iter += READ_PCYCLES() - hs;
      pt_act += t_act;
      pt_weight += t_weight;
      pt_mm += t_mm;
      pt_accread += t_accread;
      pt_ahtorm += t_ahtorm;
      pt_outdma += t_outdma;
      if (hrc != AEE_SUCCESS)
        printf("  [pv][%s] iter=%d head=%u FAIL rc=0x%x\n", name, it, (unsigned)head_n, hrc);
    }
    t_layer_sum += t_this_iter;
  }
  double avg_heads_us = (double)t_layer_sum / LAYER_ITERS;
  double per_layer_us = avg_heads_us + layer_us; /* pure steady-state, bake excluded (one-time cost) */
  double per_layer_us_with_incr = per_layer_us + incr_bake_us * MHA_NCH; /* + one incremental bake/head/token */

  /* correctness: SEPARATE, untimed final pass, against BOTH references */
  err_stats worst_vs_fp32 = {0, 0, 0, -1}, worst_vs_fp16acc = {0, 0, 0, -1};
  int all_ok = 1;
  for (uint32_t head_n = 0; head_n < MHA_NCH; head_n++) {
    uint64_t ta = 0, tw = 0, tm = 0, tar = 0, tah = 0, to = 0;
    memset(out_res, 0, (size_t)M * N * sizeof(_Float16));
    int hrc = matmul_pv_f16(vtcm_base, &p, M, N, K, out_res, P_t, wt_wh_ddr[head_n], &ta, &tw, &tm,
                            &tar, &tah, &to);
    if (hrc != AEE_SUCCESS) {
      all_ok = 0;
      continue;
    }
    err_stats e32 = err_stats_f16(MHA_GQA * N, ref_fp32, out_res);       /* only the valid gqa rows matter */
    err_stats e16 = err_stats_f16(MHA_GQA * N, ref_fp16acc, out_res);
    if (e32.max_rel > worst_vs_fp32.max_rel)
      worst_vs_fp32 = e32;
    if (e16.max_rel > worst_vs_fp16acc.max_rel)
      worst_vs_fp16acc = e16;
    if (e32.max_rel >= 5e-2)
      all_ok = 0;
  }

  printf("    [per-head STEADY-STATE phase breakdown, averaged over %u calls]\n", (unsigned)n_calls);
  printf("      P activation stage (DMA + rm_to_ah_f16) %9.3f us\n", (double)pt_act / n_calls);
  printf("      V weight restage (DMA from pre-baked WH) %9.3f us\n", (double)pt_weight / n_calls);
  printf("      DSP matmul (HMX mm_f16)                  %9.3f us\n", (double)pt_mm / n_calls);
  printf("      acc_read                                 %9.3f us\n", (double)pt_accread / n_calls);
  printf("      ah_to_rm                                 %9.3f us\n", (double)pt_ahtorm / n_calls);
  printf("      output write-back (ring push+drain)      %9.3f us\n", (double)pt_outdma / n_calls);
  printf("  layer DMA (one-time)                       %9.1f us\n", layer_us);
  printf("  V weight FULL bake (one-time, per head, NOT steady-state) %9.1f us/head (%.1f us total, "
         "%u heads)\n",
         full_bake_us_per_head, full_bake_us_per_head * MHA_NCH, (unsigned)MHA_NCH);
  printf("  V weight INCREMENTAL bake (%u tiles, one KV-append)       %9.2f us\n",
         (unsigned)p.n_tiles, incr_bake_us);
  printf("  P rough DSP-side transpose ([kv,gqa]->[gqa,kv], unoptimized) %9.1f us "
         "(price of NOT getting the pre-transposed layout)\n",
         rough_transpose_us);
  printf("  sum of %u per-head STEADY-STATE pipelines (avg of %d)   %9.1f us\n", (unsigned)MHA_NCH,
         LAYER_ITERS, avg_heads_us);
  printf("  PER-LAYER TOTAL, steady-state only (all %u kv_heads)     %9.1f us\n", (unsigned)MHA_NCH,
         per_layer_us);
  printf("  PER-LAYER TOTAL, + 1 incremental bake/head/token         %9.1f us   <- compare vs CPU "
         "648-678us (kv=1024 seq)\n",
         per_layer_us_with_incr);
  printf("  accuracy vs fp32-accumulating ref (ground truth):\n");
  printf("    maxAbsErr=%.6g maxRelErr=%.6g meanRelErr=%.6g\n", worst_vs_fp32.max_abs,
         worst_vs_fp32.max_rel, worst_vs_fp32.mean_rel);
  printf("  accuracy vs fp16-accumulating SCALAR ref (isolates fp16-accum error from HMX):\n");
  printf("    maxAbsErr=%.6g maxRelErr=%.6g meanRelErr=%.6g\n", worst_vs_fp16acc.max_abs,
         worst_vs_fp16acc.max_rel, worst_vs_fp16acc.mean_rel);
  if (worst_vs_fp32.max_rel >= 1e-2) {
    /* per instructions: report only, do not implement the K-chunking
     * mitigation. Estimate its cost from the acc_read time actually
     * measured above. */
    const int chunk_depth = 256;
    const int n_chunks = (int)((K + chunk_depth - 1) / chunk_depth);
    const double extra_accreads = (double)(n_chunks - 1) * p.n_tiles * MHA_NCH;
    const double accread_us_per_call = (double)pt_accread / n_calls / p.n_tiles;
    printf("    max relErr %.6g >= 1e-2 -- NOT implementing K-chunking mitigation, per instructions.\n",
           worst_vs_fp32.max_rel);
    printf("    estimated cost of chunking K every %d: %d chunks, %.0f extra acc_read calls/layer,"
           " ~%.1f us/layer (at %.3f us/acc_read measured above)\n",
           chunk_depth, n_chunks, extra_accreads, extra_accreads * accread_us_per_call,
           accread_us_per_call);
  }
  printf("MHA_PV_LAYER_TOTAL_US_%s %.1f\n", name, per_layer_us_with_incr);
  printf("MHA_PV_LAYER_TOTAL_STEADYONLY_US_%s %.1f\n", name, per_layer_us);
  printf("MHA_PV_FULLBAKE_US_%s %.1f\n", name, full_bake_us_per_head);
  printf("MHA_PV_INCRBAKE_US_%s %.2f\n", name, incr_bake_us);
  printf("MHA_PV_ROUGHTRANSPOSE_US_%s %.1f\n", name, rough_transpose_us);
  printf("MHA_PV_MAXRELERR_FP32REF_%s %.6g\n", name, worst_vs_fp32.max_rel);
  printf("MHA_PV_MAXRELERR_FP16ACCREF_%s %.6g\n", name, worst_vs_fp16acc.max_rel);
  printf("MHA_PV_CORRECT_%s %s\n", name, all_ok ? "OK" : "WARN");
  /* consolidated table row fields. No macro/micro-naive baseline exists for
   * P.V (M2-k went straight to the intended design after M2-l's fix), so
   * those fields simply won't be emitted for this shape -- the report
   * script shows "-" rather than a fabricated number. gbps here is the
   * layer-wide V-cache DMA, same convention as the Q.K^T shape. */
  printf("MHA_FIELD shape=%s field=dma_layer_us_8heads value=%.1f\n", name, per_layer_us);
  printf("MHA_FIELD shape=%s field=incr_bake_us_8heads value=%.1f\n", name, incr_bake_us * MHA_NCH);
  printf("MHA_FIELD shape=%s field=total_us_8heads value=%.1f\n", name, per_layer_us_with_incr);
  printf("MHA_FIELD shape=%s field=gbps value=%.2f\n", name, layer_gbps);
  printf("MHA_FIELD shape=%s field=maxrelerr value=%.6g\n", name, worst_vs_fp32.max_rel);
  printf("MHA_FIELD shape=%s field=correct value=%s\n", name, all_ok ? "OK" : "WARN");

done:
  free(V_kn);
  free(V_layer_ddr);
  free(P_t);
  free(ref_fp32);
  free(ref_fp16acc);
  free(out_res);
  free(v_head_ddr);
  for (uint32_t h = 0; h < MHA_NCH; h++)
    free(wt_wh_ddr[h]);
}

/* =====================================================================
 * M2-h: prefill shapes #6 (qk_pre_128, M=128 N=1024 K=128) and #7
 * (pv_pre_128, M=128 N=128 K=1024) -- everything learned applied: layer-wide
 * contiguous DMA, VTCM->VTCM/DDR repack with pure-format ops only, pre-baked
 * WH restaged (not re-baked) in the steady state, DDR-not-VTCM for any
 * extraction-style call. Arithmetic intensity is ~128x decode's (one K/V
 * read now amortises over 128 query rows via row_bands=4 instead of 1),
 * so this is the regime where mm_f16 might actually become a meaningful
 * fraction of the per-head cost -- reported explicitly below, not assumed.
 * ===================================================================== */

/*! @brief Row-banded, PLAIN (non-GQA-strided) activation staging -- both
    Q (shape #6) and P (shape #7) are already per-head, contiguous [row_bands
    x 32, K] host buffers (no nch interleaving to account for), unlike the
    K/V cache. Same DMA-then-pure-format recipe as every other activation
    path in this file. */
static void push_band_activation_plain(uint8_t *vtcm_base, const vtcm_plan_f16 *p,
                                       const _Float16 *act_host, uint32_t k, uint32_t band,
                                       int slot) {
  const uint32_t src_stride = k * (uint32_t)sizeof(_Float16);
  const uint8_t *band_base =
    (const uint8_t *)act_host + (size_t)band * HEXKL_HMX_F16_BLOCK_N_ROW * src_stride;
  for (uint32_t kt = 0; kt < p->k_tiles; kt++) {
    const uint8_t *asrc = band_base + kt * HEXKL_HMX_F16_BLOCK_N_INNER * sizeof(_Float16);
    ring_push2d(vtcm_base + p->act_flat_off[slot] + HEXKL_HMX_ACTIVATION_ALIGNMENT * kt, asrc,
               HEXKL_HMX_F16_BLOCK_N_INNER * (uint32_t)sizeof(_Float16), src_stride,
               HEXKL_HMX_F16_BLOCK_N_INNER * (uint32_t)sizeof(_Float16), HEXKL_HMX_F16_BLOCK_N_ROW,
               /*src_vtcm=*/0, /*dst_vtcm=*/1);
  }
}

/*! @brief Generic prefill kernel, shared by shapes #6 and #7: weight
    restaged ONCE from a pre-baked DDR WH slab (no re-bake, same as M2-l's
    P.V fix), activation pipelined across row_bands (prefetch band b+1 while
    band b's mm/output run -- the SAME pattern as matmul_layer_dma_f16, just
    with the plain, non-GQA-strided activation source). */
static int matmul_prefill_f16(uint8_t *vtcm_base, const vtcm_plan_f16 *p, uint32_t n, uint32_t k,
                              _Float16 *out, const _Float16 *act_host, const uint8_t *wt_wh_ddr,
                              uint64_t *t_weight, uint64_t *t_act_dma, uint64_t *t_act_layout,
                              uint64_t *t_mm, uint64_t *t_accread, uint64_t *t_ahtorm,
                              uint64_t *t_outdma, uint64_t *t_ringdrain) {
  uint64_t s;
  int rc;

  s = READ_PCYCLES();
  restage_pv_weight(vtcm_base, p, wt_wh_ddr);
  *t_weight += READ_PCYCLES() - s;

  s = READ_PCYCLES();
  ring_init();
  push_band_activation_plain(vtcm_base, p, act_host, k, 0, 0);
  ring_drain();
  *t_act_dma += READ_PCYCLES() - s;

  s = READ_PCYCLES();
  rc = convert_band_to_ah(vtcm_base, p, 0);
  if (rc != AEE_SUCCESS)
    return rc;
  *t_act_layout += READ_PCYCLES() - s;

  for (uint32_t band = 0; band < p->row_bands; band++) {
    const int nxt_slot = (int)((band + 1) & 1);

    s = READ_PCYCLES();
    if (band + 1 < p->row_bands)
      push_band_activation_plain(vtcm_base, p, act_host, k, band + 1, nxt_slot);
    *t_act_dma += READ_PCYCLES() - s;

    for (uint32_t nt = 0; nt < p->n_tiles; nt++) {
      s = READ_PCYCLES();
      hexkl_micro_hmx_acc_clear_f16();
      for (uint32_t kt = 0; kt < p->k_tiles; kt++) {
        rc = hexkl_micro_hmx_mm_f16(vtcm_base, p->act_ah_off + HEXKL_HMX_ACTIVATION_ALIGNMENT * kt,
                                    p->wt_off + (kt * p->n_tiles + nt) * WT_TILE_BYTES_F16);
        if (rc != AEE_SUCCESS)
          return rc;
      }
      *t_mm += READ_PCYCLES() - s;

      s = READ_PCYCLES();
      rc = hexkl_micro_hmx_acc_read_f16(vtcm_base, p->config_off, p->acc_ah_off);
      if (rc != AEE_SUCCESS)
        return rc;
      *t_accread += READ_PCYCLES() - s;

      const uint32_t out_slot_off = p->out_flat_base + nt * HEXKL_HMX_ACTIVATION_ALIGNMENT;
      s = READ_PCYCLES();
      rc = hexkl_micro_hmx_ah_to_rm_f16(vtcm_base, out_slot_off, p->acc_ah_off);
      if (rc != AEE_SUCCESS)
        return rc;
      *t_ahtorm += READ_PCYCLES() - s;

      s = READ_PCYCLES();
      _Float16 *dst = out + (size_t)band * HEXKL_HMX_F16_BLOCK_N_ROW * n +
                      (size_t)nt * HEXKL_HMX_F16_BLOCK_N_COL;
      ring_push2d(dst, vtcm_base + out_slot_off, n * (uint32_t)sizeof(_Float16),
                 HEXKL_HMX_F16_BLOCK_N_COL * (uint32_t)sizeof(_Float16),
                 HEXKL_HMX_F16_BLOCK_N_COL * (uint32_t)sizeof(_Float16), HEXKL_HMX_F16_BLOCK_N_ROW,
                 /*src_vtcm=*/1, /*dst_vtcm=*/0);
      *t_outdma += READ_PCYCLES() - s;
    }

    s = READ_PCYCLES();
    ring_drain();
    *t_ringdrain += READ_PCYCLES() - s;

    if (band + 1 < p->row_bands) {
      s = READ_PCYCLES();
      rc = convert_band_to_ah(vtcm_base, p, nxt_slot);
      if (rc != AEE_SUCCESS)
        return rc;
      *t_act_layout += READ_PCYCLES() - s;
    }
  }
  return AEE_SUCCESS;
}

/*! @brief Shape #6's weight prep: K, unlike V, genuinely needs a transpose
    (Kt = [head_dim,kv_len]) to serve as the RHS of S=Q.K^T -- there is no
    "pre-transposed by convention" escape hatch here the way P had one for
    P.V, because K really is stored [kv,hd] and mm_f16 needs [hd,kv]. Q
    (32KB) is far smaller than K (256KB), so Q is transposed... except Q is
    already the ACTIVATION (LHS) in this orientation and needs no transpose
    at all -- it is K, playing the WEIGHT (RHS) role, that must be
    transposed. Extract+transpose+bake, ONE-TIME, then DMA the resulting WH
    out to DDR (pre-baked, restaged every call thereafter, never re-baked). */
static int prebake_qk_weight_transposed(uint8_t *vtcm_base, const vtcm_plan_f16 *p,
                                        uint32_t layer_kv_off, uint32_t nch, uint32_t head_dim,
                                        uint32_t kv_len, uint32_t head_n, uint8_t *k_head_ddr,
                                        uint8_t *kt_ddr, uint8_t *wt_wh_ddr_out,
                                        uint64_t *t_transpose) {
  static hexdma_desc2d d __attribute__((aligned(128)));
  const uint32_t src_stride = nch * head_dim * (uint32_t)sizeof(_Float16);
  const uint8_t *vtcm_src = vtcm_base + layer_kv_off + head_n * head_dim * (uint32_t)sizeof(_Float16);
  /* 1. bulk VTCM->DDR extraction of this head's [kv,hd] slice (identical
   * shape to bake_weight_from_layer_pv's first step). */
  hexdma_2d_vtcm_to_ddr(&d, k_head_ddr, vtcm_src, head_dim * (uint32_t)sizeof(_Float16), src_stride,
                       head_dim * (uint32_t)sizeof(_Float16), kv_len);

  /* 2. TRANSPOSE [kv,hd] -> [hd,kv] -- a real transpose, timed separately. */
  uint64_t s = READ_PCYCLES();
  const _Float16 *k_nat = (const _Float16 *)k_head_ddr;
  _Float16 *kt = (_Float16 *)kt_ddr;
  for (uint32_t row = 0; row < kv_len; row++)
    for (uint32_t d = 0; d < head_dim; d++)
      kt[d * kv_len + row] = k_nat[row * head_dim + d];
  *t_transpose += READ_PCYCLES() - s;

  /* 3. bake ALL tiles from kt_ddr [hd,kv] = [K,N] (DDR source, the
   * established fast path). */
  for (uint32_t kt_i = 0; kt_i < p->k_tiles; kt_i++) {
    for (uint32_t nt = 0; nt < p->n_tiles; nt++) {
      int rc = hexkl_micro_hmx_rm_to_wh_f16(
        vtcm_base, p->wt_off + (kt_i * p->n_tiles + nt) * WT_TILE_BYTES_F16,
        (const _Float16 *)kt_ddr, kt_i, nt, kv_len);
      if (rc != AEE_SUCCESS)
        return rc;
    }
  }

  /* 4. DMA resident WH -> DDR (pre-baked, done). */
  uint32_t rs = 262144u;
  while (p->wt_bytes % rs)
    rs >>= 1;
  hexdma_2d_vtcm_to_ddr(&d, wt_wh_ddr_out, vtcm_base + p->wt_off, rs, rs, rs, p->wt_bytes / rs);
  return AEE_SUCCESS;
}

/*! @brief M2-m: incremental bake for shape #6 -- only the NEWLY-appended
    new_kv_rows kv positions (128 of 1024 = 1/8 of the n_tiles, since kv_len
    is K's N-dimension in this orientation), reusing the already-extracted
    full [kv,hd] k_head_ddr (no fresh VTCM extraction -- same methodology as
    decode's bench_incremental_bake_pv). Transposes only the new slice, bakes
    only the new n_tiles. Excluding this from the prefill total (as the
    first version of this bench did) is structurally wrong, not just
    conservative: prefill has no true steady state -- every chunk's KV rows
    are newly appended and MUST be baked. */
static uint64_t bench_incremental_bake_qk(const vtcm_plan_f16 *p, uint8_t *vtcm_base,
                                          const uint8_t *k_head_ddr_full, uint32_t head_dim,
                                          uint32_t kv_len, uint32_t new_kv_rows,
                                          uint8_t *kt_new_scratch) {
  const uint32_t row_offset = kv_len - new_kv_rows;
  const _Float16 *k_nat_new = (const _Float16 *)k_head_ddr_full + (size_t)row_offset * head_dim;
  _Float16 *kt = (_Float16 *)kt_new_scratch;
  const uint32_t new_n_tiles = new_kv_rows / HEXKL_HMX_F16_BLOCK_N_COL;
  const uint32_t nt_base = p->n_tiles - new_n_tiles;

  uint64_t s = READ_PCYCLES();
  for (uint32_t row = 0; row < new_kv_rows; row++)
    for (uint32_t d = 0; d < head_dim; d++)
      kt[d * new_kv_rows + row] = k_nat_new[row * head_dim + d];
  for (uint32_t kt_i = 0; kt_i < p->k_tiles; kt_i++) {
    for (uint32_t local_nt = 0; local_nt < new_n_tiles; local_nt++) {
      const uint32_t global_nt = nt_base + local_nt;
      hexkl_micro_hmx_rm_to_wh_f16(vtcm_base,
                                   p->wt_off + (kt_i * p->n_tiles + global_nt) * WT_TILE_BYTES_F16, kt,
                                   kt_i, local_nt, new_kv_rows);
    }
  }
  return READ_PCYCLES() - s;
}

/*! @brief M2-m: incremental bake for shape #7 -- only the newly-appended
    new_kv_rows kv positions (128 of 1024 = 1/8 of the K_TILES here, since
    V's contraction dim IS kv_len, unlike K's case where kv_len is the
    N-dimension). No transpose needed (V never needed one); reuses the
    already-extracted full [kv,hd] v_head_ddr. */
static uint64_t bench_incremental_bake_pv_prefill(const vtcm_plan_f16 *p, uint8_t *vtcm_base,
                                                  const uint8_t *v_head_ddr_full, uint32_t n,
                                                  uint32_t new_kv_rows) {
  const uint32_t new_k_tiles = new_kv_rows / HEXKL_HMX_F16_BLOCK_N_INNER;
  const uint32_t kt_base = p->k_tiles - new_k_tiles;
  uint64_t s = READ_PCYCLES();
  for (uint32_t local_kt = 0; local_kt < new_k_tiles; local_kt++) {
    const uint32_t global_kt = kt_base + local_kt;
    for (uint32_t nt = 0; nt < p->n_tiles; nt++) {
      hexkl_micro_hmx_rm_to_wh_f16(vtcm_base,
                                   p->wt_off + (global_kt * p->n_tiles + nt) * WT_TILE_BYTES_F16,
                                   (const _Float16 *)v_head_ddr_full, global_kt, nt, n);
    }
  }
  return READ_PCYCLES() - s;
}

/*! @brief Shape #6 driver: qk_pre_128 (M=128 N=1024 K=128), S=Q.K^T. */
static void run_qk_prefill_shape_f16(uint8_t *vtcm_base, uint32_t vtcm_size, const char *name,
                                     uint32_t M, uint32_t N, uint32_t K) {
  if (M % 32 || N % 32 || K % 32) {
    printf("[qk_prefill][%s] M=%u N=%u K=%u not tile-aligned, skipping\n", name, (unsigned)M,
           (unsigned)N, (unsigned)K);
    return;
  }
  vtcm_plan_f16 p = plan_vtcm_f16(M, K, N, vtcm_size);
  const uint32_t need_local = p.out_flat_base + p.n_tiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;
  if (need_local > p.config_off) {
    printf("[qk_prefill][%s] local VTCM plan does not fit, skipping\n", name);
    return;
  }
  const size_t layer_bytes = (size_t)N * MHA_NCH * K * sizeof(_Float16); /* K-cache layer: kv_len=N here */
  const uint32_t reserve = (uint32_t)layer_bytes + 4096u;
  if (reserve > p.config_off || (uint32_t)(p.config_off - reserve) < need_local) {
    printf("[qk_prefill][%s] layer (%zu bytes) does not fit VTCM, skipping\n", name, layer_bytes);
    return;
  }
  const uint32_t layer_kv_off = align_up(p.config_off - reserve, HEXKL_HMX_ACTIVATION_ALIGNMENT);

  _Float16 *Q_host = (_Float16 *)malloc((size_t)M * K * sizeof(_Float16));  /* [M,K], plain */
  _Float16 *K_nat = (_Float16 *)malloc((size_t)N * K * sizeof(_Float16));   /* [kv=N, hd=K] natural */
  _Float16 *Kt_kn = (_Float16 *)malloc((size_t)K * N * sizeof(_Float16));   /* [K,N] = transpose, for ref only */
  _Float16 *K_layer_ddr = (_Float16 *)malloc(layer_bytes);                  /* [kv][nch][hd], all heads identical */
  float *ref_fp32 = (float *)malloc((size_t)M * N * sizeof(float));
  _Float16 *out_res = (_Float16 *)malloc((size_t)M * N * sizeof(_Float16));
  uint8_t *k_head_ddr = (uint8_t *)malloc((size_t)N * K * sizeof(_Float16));
  uint8_t *kt_ddr = (uint8_t *)malloc((size_t)K * N * sizeof(_Float16));
  uint8_t *wt_wh_ddr[MHA_NCH] = {0};
  int wt_wh_ok = 1;
  for (uint32_t h = 0; h < MHA_NCH; h++) {
    wt_wh_ddr[h] = (uint8_t *)malloc(p.wt_bytes);
    if (!wt_wh_ddr[h])
      wt_wh_ok = 0;
  }
  if (!Q_host || !K_nat || !Kt_kn || !K_layer_ddr || !ref_fp32 || !out_res || !k_head_ddr || !kt_ddr ||
      !wt_wh_ok) {
    printf("[qk_prefill][%s] oom, skipping\n", name);
    goto done;
  }

  hexkl_test_init_pattern_f16(M, K, Q_host, 5, 0.067f);
  hexkl_test_init_pattern_f16(N, K, K_nat, 3, 0.04906f); /* [kv,hd] */
  for (uint32_t row = 0; row < N; row++)
    for (uint32_t d = 0; d < K; d++)
      Kt_kn[d * N + row] = K_nat[row * K + d]; /* host-side ref transpose, untimed */
  for (uint32_t row = 0; row < N; row++)
    for (uint32_t h = 0; h < MHA_NCH; h++)
      for (uint32_t d = 0; d < K; d++)
        K_layer_ddr[(row * MHA_NCH + h) * K + d] = K_nat[row * K + d];

  hexkl_test_init_zero_f32(M * N, ref_fp32);
  hexkl_test_matmul_cm_f16f16_f32(M, N, K, N, ref_fp32, Q_host, Kt_kn);

  printf("\nMHA_SHAPE name=%s M=%u N=%u K=%u (Q.K^T prefill, M2-h)\n", name, (unsigned)M, (unsigned)N,
         (unsigned)K);

  double layer_us, layer_gbps;
  {
    static hexdma_desc2d layer_desc __attribute__((aligned(128)));
    uint32_t rs = 262144u;
    while (layer_bytes % rs)
      rs >>= 1;
    uint64_t t = 0;
    const int LIT = 20;
    for (int it = 0; it < LIT; it++) {
      uint64_t s = READ_PCYCLES();
      hexdma_2d_ddr_to_vtcm(&layer_desc, vtcm_base + layer_kv_off, K_layer_ddr, rs, rs, rs,
                            (uint32_t)(layer_bytes / rs));
      t += READ_PCYCLES() - s;
    }
    layer_us = (double)t / LIT;
    layer_gbps = layer_us > 0.0 ? ((double)layer_bytes / (layer_us * 1.0e-6)) / 1.0e9 : 0.0;
    printf("  layer-wide K-cache DMA (one descriptor, %zu bytes) %9.1f us   GB/s=%.2f\n", layer_bytes,
           layer_us, layer_gbps);
  }

  /* one-time: per-head extract+transpose+bake+DMA-out */
  double full_bake_us_per_head, transpose_us_per_head;
  {
    uint64_t t_total = 0, t_transpose = 0;
    for (uint32_t head_n = 0; head_n < MHA_NCH; head_n++) {
      uint64_t s = READ_PCYCLES();
      int rc = prebake_qk_weight_transposed(vtcm_base, &p, layer_kv_off, MHA_NCH, K, N, head_n,
                                            k_head_ddr, kt_ddr, wt_wh_ddr[head_n], &t_transpose);
      t_total += READ_PCYCLES() - s;
      if (rc != AEE_SUCCESS)
        printf("  [qk_prefill][%s] prebake head=%u FAIL rc=0x%x\n", name, (unsigned)head_n, rc);
    }
    full_bake_us_per_head = (double)t_total / MHA_NCH;
    transpose_us_per_head = (double)t_transpose / MHA_NCH;
  }

  /* M2-m: incremental bake -- the honest per-chunk cost. Prefill has no true
   * steady state: this chunk's M=128 new tokens' own K rows are exactly what
   * gets freshly appended to the cache (new_kv_rows = M, not a separate
   * constant), so this MUST be paid every chunk, unlike decode where an
   * already-populated cache can be assumed already-baked. Reuses the last
   * head's k_head_ddr (already extracted above) and kt_ddr as scratch (sized
   * K*N, far larger than the head_dim*M slice needed here). */
  double incr_bake_us_per_head;
  {
    const int IIT = 20;
    uint64_t t = 0;
    for (int it = 0; it < IIT; it++)
      t += bench_incremental_bake_qk(&p, vtcm_base, k_head_ddr, K, N, M, kt_ddr);
    incr_bake_us_per_head = (double)t / IIT;
  }

  /* timed per-head sweep: steady state only (weight restaged, not re-baked) */
  const int LAYER_ITERS = 5;
  uint64_t t_layer_sum = 0;
  uint64_t pt_weight = 0, pt_act = 0, pt_actlayout = 0, pt_mm = 0, pt_accread = 0, pt_ahtorm = 0,
           pt_outdma = 0, pt_ringdrain = 0;
  const uint32_t n_calls = (uint32_t)LAYER_ITERS * MHA_NCH;
  for (int it = 0; it < LAYER_ITERS; it++) {
    uint64_t t_this_iter = 0;
    for (uint32_t head_n = 0; head_n < MHA_NCH; head_n++) {
      uint64_t tw = 0, ta = 0, tal = 0, tm = 0, tar = 0, tah = 0, to = 0, trd = 0;
      uint64_t hs = READ_PCYCLES();
      int hrc = matmul_prefill_f16(vtcm_base, &p, N, K, out_res, Q_host, wt_wh_ddr[head_n], &tw, &ta,
                                   &tal, &tm, &tar, &tah, &to, &trd);
      t_this_iter += READ_PCYCLES() - hs;
      pt_weight += tw;
      pt_act += ta;
      pt_actlayout += tal;
      pt_mm += tm;
      pt_accread += tar;
      pt_ahtorm += tah;
      pt_outdma += to;
      pt_ringdrain += trd;
      if (hrc != AEE_SUCCESS)
        printf("  [qk_prefill][%s] iter=%d head=%u FAIL rc=0x%x\n", name, it, (unsigned)head_n, hrc);
    }
    t_layer_sum += t_this_iter;
  }
  double avg_heads_us = (double)t_layer_sum / LAYER_ITERS;
  double per_layer_us = avg_heads_us + layer_us;
  double per_layer_us_with_incr = per_layer_us + incr_bake_us_per_head * MHA_NCH;

  /* correctness: SEPARATE, untimed final pass */
  err_stats worst = {0, 0, 0, -1};
  int all_ok = 1;
  for (uint32_t head_n = 0; head_n < MHA_NCH; head_n++) {
    uint64_t tw = 0, ta = 0, tal = 0, tm = 0, tar = 0, tah = 0, to = 0, trd = 0;
    memset(out_res, 0, (size_t)M * N * sizeof(_Float16));
    int hrc = matmul_prefill_f16(vtcm_base, &p, N, K, out_res, Q_host, wt_wh_ddr[head_n], &tw, &ta,
                                 &tal, &tm, &tar, &tah, &to, &trd);
    if (hrc != AEE_SUCCESS) {
      all_ok = 0;
      continue;
    }
    err_stats e = err_stats_f16(M * N, ref_fp32, out_res);
    if (e.max_rel > worst.max_rel)
      worst = e;
    if (e.max_rel >= 5e-2)
      all_ok = 0;
  }

  printf("    [per-head STEADY-STATE phase breakdown, averaged over %u calls]\n", (unsigned)n_calls);
  printf("      V/K weight restage (DMA from pre-baked WH) %9.3f us\n", (double)pt_weight / n_calls);
  printf("      Q activation DMA                           %9.3f us\n", (double)pt_act / n_calls);
  printf("      activation layout (rm_to_ah_f16)            %9.3f us\n", (double)pt_actlayout / n_calls);
  printf("      DSP matmul (HMX mm_f16)                     %9.3f us\n", (double)pt_mm / n_calls);
  printf("      acc_read                                    %9.3f us\n", (double)pt_accread / n_calls);
  printf("      ah_to_rm                                     %9.3f us\n", (double)pt_ahtorm / n_calls);
  printf("      output write-back (ring push)                %9.3f us\n", (double)pt_outdma / n_calls);
  printf("      ring_drain (wait)                            %9.3f us\n", (double)pt_ringdrain / n_calls);
  printf("  layer DMA (one-time)                       %9.1f us\n", layer_us);
  printf("  K weight FULL bake incl. transpose (one-time/head) %9.1f us/head (%.1f us total)\n",
         full_bake_us_per_head, full_bake_us_per_head * MHA_NCH);
  printf("    of which: K transpose alone                     %9.1f us/head\n", transpose_us_per_head);
  printf("  K weight INCREMENTAL bake (%u new kv rows = %u new n_tiles of %u, %.4fx of full) %9.1f "
         "us/head\n",
         (unsigned)M, (unsigned)(M / HEXKL_HMX_F16_BLOCK_N_COL), (unsigned)p.n_tiles,
         incr_bake_us_per_head / full_bake_us_per_head, incr_bake_us_per_head);
  printf("  sum of %u per-head STEADY-STATE pipelines (avg of %d) %9.1f us\n", (unsigned)MHA_NCH,
         LAYER_ITERS, avg_heads_us);
  printf("  PER-LAYER TOTAL, steady-state only (all %u kv_heads)  %9.1f us\n", (unsigned)MHA_NCH,
         per_layer_us);
  printf("  PER-LAYER TOTAL, + incremental bake/head/chunk (all %u kv_heads) %9.1f us   <- compare "
         "vs CPU\n",
         (unsigned)MHA_NCH, per_layer_us_with_incr);
  printf("  correctness vs fp32 ref: maxAbsErr=%.6g maxRelErr=%.6g meanRelErr=%.6g\n", worst.max_abs,
         worst.max_rel, worst.mean_rel);
  printf("MHA_PREFILL_QK_TOTAL_US %.1f\n", per_layer_us_with_incr);
  printf("MHA_PREFILL_QK_STEADYONLY_US %.1f\n", per_layer_us);
  printf("MHA_PREFILL_QK_INCRBAKE_US %.1f\n", incr_bake_us_per_head * MHA_NCH);
  printf("MHA_PREFILL_QK_MAXRELERR %.6g\n", worst.max_rel);
  printf("MHA_PREFILL_QK_CORRECT %s\n", all_ok ? "OK" : "WARN");
  /* consolidated table row fields. No macro/micro-naive baseline was ever
   * built for prefill shapes (they came directly from the intended-design
   * kernel), so those fields are simply absent for this shape -- reported
   * as "-", not fabricated. total_us_8heads here INCLUDES the incremental
   * bake, unlike the decode Q.K^T shape -- prefill has no true steady state
   * (every chunk's KV rows are newly appended and MUST be baked), see the
   * M2-m note in this function's history for why that is not optional here. */
  printf("MHA_FIELD shape=%s field=dma_layer_us_8heads value=%.1f\n", name, per_layer_us);
  printf("MHA_FIELD shape=%s field=incr_bake_us_8heads value=%.1f\n", name,
         incr_bake_us_per_head * MHA_NCH);
  printf("MHA_FIELD shape=%s field=total_us_8heads value=%.1f\n", name, per_layer_us_with_incr);
  printf("MHA_FIELD shape=%s field=gbps value=%.2f\n", name, layer_gbps);
  printf("MHA_FIELD shape=%s field=maxrelerr value=%.6g\n", name, worst.max_rel);
  printf("MHA_FIELD shape=%s field=correct value=%s\n", name, all_ok ? "OK" : "WARN");

done:
  free(Q_host);
  free(K_nat);
  free(Kt_kn);
  free(K_layer_ddr);
  free(ref_fp32);
  free(out_res);
  free(k_head_ddr);
  free(kt_ddr);
  for (uint32_t h = 0; h < MHA_NCH; h++)
    free(wt_wh_ddr[h]);
}

/*! @brief Shape #7 driver: pv_pre_128 (M=128 N=128 K=1024), P.V. Reuses the
    EXACT same V weight prep as decode P.V (bake_weight_from_layer_pv /
    prebake_pv_weight -- V needs no transpose, unlike K), just with M=128
    (4 row bands) instead of 32 (1 row band) via matmul_prefill_f16. */
static void run_pv_prefill_shape_f16(uint8_t *vtcm_base, uint32_t vtcm_size, const char *name,
                                     uint32_t M, uint32_t N, uint32_t K) {
  if (M % 32 || N % 32 || K % 32) {
    printf("[pv_prefill][%s] M=%u N=%u K=%u not tile-aligned, skipping\n", name, (unsigned)M,
           (unsigned)N, (unsigned)K);
    return;
  }
  vtcm_plan_f16 p = plan_vtcm_f16(M, K, N, vtcm_size);
  const uint32_t need_local = p.out_flat_base + p.n_tiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;
  if (need_local > p.config_off) {
    printf("[pv_prefill][%s] local VTCM plan does not fit, skipping\n", name);
    return;
  }
  const size_t layer_bytes = (size_t)K * MHA_NCH * N * sizeof(_Float16); /* V layer: [kv=K][nch][hd=N] */
  const uint32_t reserve = (uint32_t)layer_bytes + 4096u;
  if (reserve > p.config_off || (uint32_t)(p.config_off - reserve) < need_local) {
    printf("[pv_prefill][%s] layer (%zu bytes) does not fit VTCM, skipping\n", name, layer_bytes);
    return;
  }
  const uint32_t layer_kv_off = align_up(p.config_off - reserve, HEXKL_HMX_ACTIVATION_ALIGNMENT);

  _Float16 *V_kn = (_Float16 *)malloc((size_t)K * N * sizeof(_Float16));
  _Float16 *V_layer_ddr = (_Float16 *)malloc(layer_bytes);
  _Float16 *P_t = (_Float16 *)malloc((size_t)M * K * sizeof(_Float16)); /* [M=128,K], ALL rows valid */
  float *ref_fp32 = (float *)malloc((size_t)M * N * sizeof(float));
  _Float16 *out_res = (_Float16 *)malloc((size_t)M * N * sizeof(_Float16));
  uint8_t *v_head_ddr = (uint8_t *)malloc((size_t)K * N * sizeof(_Float16));
  uint8_t *wt_wh_ddr[MHA_NCH] = {0};
  int wt_wh_ok = 1;
  for (uint32_t h = 0; h < MHA_NCH; h++) {
    wt_wh_ddr[h] = (uint8_t *)malloc(p.wt_bytes);
    if (!wt_wh_ddr[h])
      wt_wh_ok = 0;
  }
  if (!V_kn || !V_layer_ddr || !P_t || !ref_fp32 || !out_res || !v_head_ddr || !wt_wh_ok) {
    printf("[pv_prefill][%s] oom, skipping\n", name);
    goto done;
  }

  hexkl_test_init_pattern_f16(K, N, V_kn, 7, 0.03f);
  for (uint32_t row = 0; row < K; row++)
    for (uint32_t h = 0; h < MHA_NCH; h++)
      for (uint32_t d = 0; d < N; d++)
        V_layer_ddr[(row * MHA_NCH + h) * N + d] = V_kn[row * N + d];

  gen_softmax_like_pv(M, M, K, P_t); /* ALL M=128 rows valid (no gqa-fold padding at prefill) */

  hexkl_test_init_zero_f32(M * N, ref_fp32);
  hexkl_test_matmul_cm_f16f16_f32(M, N, K, N, ref_fp32, P_t, V_kn);

  printf("\nMHA_SHAPE name=%s M=%u N=%u K=%u (P.V prefill, M2-h)\n", name, (unsigned)M, (unsigned)N,
         (unsigned)K);

  double layer_us, layer_gbps;
  {
    static hexdma_desc2d layer_desc __attribute__((aligned(128)));
    uint32_t rs = 262144u;
    while (layer_bytes % rs)
      rs >>= 1;
    uint64_t t = 0;
    const int LIT = 20;
    for (int it = 0; it < LIT; it++) {
      uint64_t s = READ_PCYCLES();
      hexdma_2d_ddr_to_vtcm(&layer_desc, vtcm_base + layer_kv_off, V_layer_ddr, rs, rs, rs,
                            (uint32_t)(layer_bytes / rs));
      t += READ_PCYCLES() - s;
    }
    layer_us = (double)t / LIT;
    layer_gbps = layer_us > 0.0 ? ((double)layer_bytes / (layer_us * 1.0e-6)) / 1.0e9 : 0.0;
    printf("  layer-wide V-cache DMA (one descriptor, %zu bytes) %9.1f us   GB/s=%.2f\n", layer_bytes,
           layer_us, layer_gbps);
  }

  double full_bake_us_per_head;
  {
    uint64_t t = 0;
    for (uint32_t head_n = 0; head_n < MHA_NCH; head_n++) {
      uint64_t s = READ_PCYCLES();
      int rc = prebake_pv_weight(vtcm_base, &p, layer_kv_off, MHA_NCH, N, K, head_n, v_head_ddr,
                                 wt_wh_ddr[head_n]);
      t += READ_PCYCLES() - s;
      if (rc != AEE_SUCCESS)
        printf("  [pv_prefill][%s] prebake head=%u FAIL rc=0x%x\n", name, (unsigned)head_n, rc);
    }
    full_bake_us_per_head = (double)t / MHA_NCH;
  }

  /* M2-m: incremental bake -- this chunk's M new tokens' own V rows are what
   * gets freshly appended (new_kv_rows = M), so this is a mandatory
   * per-chunk cost, not an amortized-away one-time cost. Reuses the last
   * head's already-extracted v_head_ddr (full [K,N] buffer). */
  double incr_bake_us_per_head;
  {
    const int IIT = 20;
    uint64_t t = 0;
    for (int it = 0; it < IIT; it++)
      t += bench_incremental_bake_pv_prefill(&p, vtcm_base, v_head_ddr, N, M);
    incr_bake_us_per_head = (double)t / IIT;
  }

  const int LAYER_ITERS = 5;
  uint64_t t_layer_sum = 0;
  uint64_t pt_weight = 0, pt_act = 0, pt_actlayout = 0, pt_mm = 0, pt_accread = 0, pt_ahtorm = 0,
           pt_outdma = 0, pt_ringdrain = 0;
  const uint32_t n_calls = (uint32_t)LAYER_ITERS * MHA_NCH;
  for (int it = 0; it < LAYER_ITERS; it++) {
    uint64_t t_this_iter = 0;
    for (uint32_t head_n = 0; head_n < MHA_NCH; head_n++) {
      uint64_t tw = 0, ta = 0, tal = 0, tm = 0, tar = 0, tah = 0, to = 0, trd = 0;
      uint64_t hs = READ_PCYCLES();
      int hrc = matmul_prefill_f16(vtcm_base, &p, N, K, out_res, P_t, wt_wh_ddr[head_n], &tw, &ta,
                                   &tal, &tm, &tar, &tah, &to, &trd);
      t_this_iter += READ_PCYCLES() - hs;
      pt_weight += tw;
      pt_act += ta;
      pt_actlayout += tal;
      pt_mm += tm;
      pt_accread += tar;
      pt_ahtorm += tah;
      pt_outdma += to;
      pt_ringdrain += trd;
      if (hrc != AEE_SUCCESS)
        printf("  [pv_prefill][%s] iter=%d head=%u FAIL rc=0x%x\n", name, it, (unsigned)head_n, hrc);
    }
    t_layer_sum += t_this_iter;
  }
  double avg_heads_us = (double)t_layer_sum / LAYER_ITERS;
  double per_layer_us = avg_heads_us + layer_us;
  double per_layer_us_with_incr = per_layer_us + incr_bake_us_per_head * MHA_NCH;

  err_stats worst = {0, 0, 0, -1};
  int all_ok = 1;
  for (uint32_t head_n = 0; head_n < MHA_NCH; head_n++) {
    uint64_t tw = 0, ta = 0, tal = 0, tm = 0, tar = 0, tah = 0, to = 0, trd = 0;
    memset(out_res, 0, (size_t)M * N * sizeof(_Float16));
    int hrc = matmul_prefill_f16(vtcm_base, &p, N, K, out_res, P_t, wt_wh_ddr[head_n], &tw, &ta, &tal,
                                 &tm, &tar, &tah, &to, &trd);
    if (hrc != AEE_SUCCESS) {
      all_ok = 0;
      continue;
    }
    err_stats e = err_stats_f16(M * N, ref_fp32, out_res);
    if (e.max_rel > worst.max_rel)
      worst = e;
    if (e.max_rel >= 5e-2)
      all_ok = 0;
  }

  printf("    [per-head STEADY-STATE phase breakdown, averaged over %u calls]\n", (unsigned)n_calls);
  printf("      V weight restage (DMA from pre-baked WH)   %9.3f us\n", (double)pt_weight / n_calls);
  printf("      P activation DMA                           %9.3f us\n", (double)pt_act / n_calls);
  printf("      activation layout (rm_to_ah_f16)            %9.3f us\n", (double)pt_actlayout / n_calls);
  printf("      DSP matmul (HMX mm_f16)                     %9.3f us\n", (double)pt_mm / n_calls);
  printf("      acc_read                                    %9.3f us\n", (double)pt_accread / n_calls);
  printf("      ah_to_rm                                     %9.3f us\n", (double)pt_ahtorm / n_calls);
  printf("      output write-back (ring push)                %9.3f us\n", (double)pt_outdma / n_calls);
  printf("      ring_drain (wait)                            %9.3f us\n", (double)pt_ringdrain / n_calls);
  printf("  layer DMA (one-time)                       %9.1f us\n", layer_us);
  printf("  V weight FULL bake (one-time/head, NOT steady-state)  %9.1f us/head (%.1f us total)\n",
         full_bake_us_per_head, full_bake_us_per_head * MHA_NCH);
  printf("  V weight INCREMENTAL bake (%u new kv rows = %u new k_tiles of %u, %.4fx of full) %9.1f "
         "us/head\n",
         (unsigned)M, (unsigned)(M / HEXKL_HMX_F16_BLOCK_N_INNER), (unsigned)p.k_tiles,
         incr_bake_us_per_head / full_bake_us_per_head, incr_bake_us_per_head);
  printf("  sum of %u per-head STEADY-STATE pipelines (avg of %d) %9.1f us\n", (unsigned)MHA_NCH,
         LAYER_ITERS, avg_heads_us);
  printf("  PER-LAYER TOTAL, steady-state only (all %u kv_heads)  %9.1f us\n", (unsigned)MHA_NCH,
         per_layer_us);
  printf("  PER-LAYER TOTAL, + incremental bake/head/chunk (all %u kv_heads) %9.1f us   <- compare "
         "vs CPU\n",
         (unsigned)MHA_NCH, per_layer_us_with_incr);
  printf("  correctness vs fp32 ref: maxAbsErr=%.6g maxRelErr=%.6g meanRelErr=%.6g\n", worst.max_abs,
         worst.max_rel, worst.mean_rel);
  printf("MHA_PREFILL_PV_TOTAL_US %.1f\n", per_layer_us_with_incr);
  printf("MHA_PREFILL_PV_STEADYONLY_US %.1f\n", per_layer_us);
  printf("MHA_PREFILL_PV_INCRBAKE_US %.1f\n", incr_bake_us_per_head * MHA_NCH);
  printf("MHA_PREFILL_PV_MAXRELERR %.6g\n", worst.max_rel);
  printf("MHA_PREFILL_PV_CORRECT %s\n", all_ok ? "OK" : "WARN");
  /* consolidated table row fields -- see run_qk_prefill_shape_f16 for the
   * same convention (total INCLUDES the incremental bake here, unlike
   * decode). */
  printf("MHA_FIELD shape=%s field=dma_layer_us_8heads value=%.1f\n", name, per_layer_us);
  printf("MHA_FIELD shape=%s field=incr_bake_us_8heads value=%.1f\n", name,
         incr_bake_us_per_head * MHA_NCH);
  printf("MHA_FIELD shape=%s field=total_us_8heads value=%.1f\n", name, per_layer_us_with_incr);
  printf("MHA_FIELD shape=%s field=gbps value=%.2f\n", name, layer_gbps);
  printf("MHA_FIELD shape=%s field=maxrelerr value=%.6g\n", name, worst.max_rel);
  printf("MHA_FIELD shape=%s field=correct value=%s\n", name, all_ok ? "OK" : "WARN");

done:
  free(V_kn);
  free(V_layer_ddr);
  free(P_t);
  free(ref_fp32);
  free(out_res);
  free(v_head_ddr);
  for (uint32_t h = 0; h < MHA_NCH; h++)
    free(wt_wh_ddr[h]);
}

int main(void) {
  int res, res2;

  printf("[MHA BENCH] start -- decode Q.K^T/P.V (qkT_dec_1024, pv_dec_1024, pv_dec_512) "
         "+ prefill Q.K^T/P.V (qk_pre_128, pv_pre_128), optimized path, all measured by "
         "default. See run_micro_mha_bench.sh for the consolidated report.\n");

  /* macro (SDKL) comparison FIRST -- it acquires then releases the HMX/VTCM
   * resource; both macro and micro holding it at once times out (0x80000401). */
  if (hexkl_macro_initialize() == AEE_SUCCESS) {
    hexkl_macro_lock_hmx();
    const size_t nm = (MHA_SHAPES_RUN) < MHA_SHAPE_COUNT ? (MHA_SHAPES_RUN) : MHA_SHAPE_COUNT;
    for (size_t i = 0; i < nm; i++)
      run_macro_shape_f16(MHA_SHAPES[i].name, MHA_SHAPES[i].M, MHA_SHAPES[i].N, MHA_SHAPES[i].K);
    hexkl_macro_unlock_hmx();
    hexkl_macro_finalize();
  } else {
    printf("  [macro] initialize failed -- skipping SDKL fp16 comparison\n");
  }

  uint8_t *vtcm_base = NULL;
  uint32_t vtcm_size = 0;
  uint32_t hmx_fp16_rate = 0;
  res = hexkl_micro_hw_init(&vtcm_base, &vtcm_size, &hmx_fp16_rate);
  if (res != AEE_SUCCESS) {
    printf("[MHA BENCH][ERROR] hw init failed (%d)\n", res);
    return res;
  }
  printf("[MHA BENCH] VTCM base=%p size=%u bytes (%.2f MiB)  HMX fp16 rate=%u ops/cycle\n",
         (void *)vtcm_base, (unsigned)vtcm_size, (double)vtcm_size / (1024.0 * 1024.0),
         (unsigned)hmx_fp16_rate);
  if (hmx_fp16_rate == 0) {
    printf("[MHA BENCH] HMX FP16 not supported on this hardware -- stopping\n");
    return AEE_SUCCESS;
  }

  res = hexkl_micro_hmx_lock();
  if (res != AEE_SUCCESS) {
    printf("[MHA BENCH][ERROR] HMX lock failed (%d)\n", res);
    return res;
  }

  const uint32_t hmx_config_offset = vtcm_size - hexkl_micro_hmx_config_size();
  res = hexkl_micro_hmx_setup_acc_read_f16(vtcm_base, hmx_config_offset);
  if (res != AEE_SUCCESS) {
    printf("[MHA BENCH][ERROR] setup_acc_read_f16 failed (%d)\n", res);
    return res;
  }

  {
    const size_t n = (MHA_SHAPES_RUN) < MHA_SHAPE_COUNT ? (MHA_SHAPES_RUN) : MHA_SHAPE_COUNT;
    for (size_t i = 0; i < n; i++)
      run_micro_shape_f16(vtcm_base, vtcm_size, hmx_config_offset, MHA_SHAPES[i].name,
                          MHA_SHAPES[i].M, MHA_SHAPES[i].N, MHA_SHAPES[i].K);
    if (n < MHA_SHAPE_COUNT)
      printf("[MHA BENCH] %u of %u MHA_SHAPES[] entries run through the generic macro/micro "
             "loop (only #1 has a matching kernel; #2/#3 are not implemented, #4-#7 run via "
             "their own dedicated functions below, not this array -- see the file header)\n",
             (unsigned)n, (unsigned)MHA_SHAPE_COUNT);
  }

  /* Diagnostic sweeps (M2-f's DMA bandwidth-ceiling curve, M2-j's output
   * narrow-vs-full comparison) -- NOT part of any shape's headline number,
   * gated behind an explicit opt-in so a plain run's output isn't cluttered
   * with numbers that don't feed the report. Build with
   * -DMHA_RUN_DIAGNOSTICS=1 (run_micro_mha_bench.sh: MHA_RUN_DIAGNOSTICS=1
   * ./run_micro_mha_bench.sh) to include them. */
#if defined(MHA_RUN_DIAGNOSTICS) && MHA_RUN_DIAGNOSTICS
  bench_dma_bw_ceiling(vtcm_base, vtcm_size);
  bench_output_narrow_vs_full(vtcm_base, vtcm_size, MHA_GQA);
#else
  printf("[MHA BENCH] diagnostic sweeps (DMA bandwidth ceiling, output narrow-vs-full) "
         "SKIPPED -- build with -DMHA_RUN_DIAGNOSTICS=1 to include them (see M2-f/M2-j "
         "in the file history for what they measure). Not part of any shape's headline "
         "number.\n");
#endif

  /* M2-k: P.V decode, spec shapes #4 (pv_dec_1024) and #5 (pv_dec_512) */
  run_pv_shape_f16(vtcm_base, vtcm_size, hmx_config_offset, "pv_dec_1024", 1024);
  run_pv_shape_f16(vtcm_base, vtcm_size, hmx_config_offset, "pv_dec_512", 512);

  /* M2-h: prefill shapes #6 (qk_pre_128) and #7 (pv_pre_128) */
  run_qk_prefill_shape_f16(vtcm_base, vtcm_size, "qk_pre_128", 128, 1024, 128);
  run_pv_prefill_shape_f16(vtcm_base, vtcm_size, "pv_pre_128", 128, 128, 1024);

  res2 = hexkl_micro_hmx_unlock();
  if (res2 != AEE_SUCCESS)
    printf("[MHA BENCH][ERROR] HMX unlock failed (%d)\n", res2);

  printf("[MHA BENCH] done\n");
  return res2;
}
