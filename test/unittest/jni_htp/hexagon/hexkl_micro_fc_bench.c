// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_vtcm_probe.c
 * @brief  u8i4 vs u8i8 micro+DMA FC matmul: does the cross-matmul
 *         weight-prefetch skel that beat SDKL for u8i4 (doc 13) still win
 *         once the weight tile doubles to 1024 B?
 *
 * Runs on the Hexagon DSP, standalone -- no FastRPC, no nntrainer. It reports
 * the VTCM size, then for each (shape, dtype) pair runs several kernels over
 * identical data and times them:
 *
 *   baseline        the SDK sample's structure: every weight tile is
 *                   re-derived from DDR inside the innermost loop, always
 *                   into one scratch slot, so nothing is ever reused.
 *   resident        every weight tile is converted once into its own VTCM
 *                   slot up front; the matmul then only reads VTCM.
 *   resident+DMA    resident, but output write-back uses user-DMA
 *                   (pipelined against the next tile's compute).
 *   cross-matmul    resident+DMA plus cross-CALL weight prefetch: matmul
 *                   i+1's weight is staged while matmul i computes. This is
 *                   the number that beat SDKL 1.7-2x for u8i4 (doc 13); this
 *                   file's job is to get the identical number for u8i8 too.
 *
 * u8i4 and u8i8 share tile geometry exactly (64x32 activation x 32x32
 * weight -> 64x32 int32 accumulator); the only difference is the weight tile
 * size (512 B packed-nibble vs 1024 B int8) and three symbol names. So this
 * file threads a `dtype_ops` descriptor through one code path instead of
 * forking -- see docs/backend_guide/htp_backend/13_fc_micro_dma_and_mha_plan.md
 * and the task spec this session followed.
 *
 * "resident" is timed twice. The first call pays the conversion; the second
 * reuses what is already in VTCM and pays nothing, which is the case that
 * matters -- an fc_layer is called once per token with the same weight.
 *
 * Every kernel is checked against a plain C reference (which reads the
 * unpacked row-major weight directly -- neither dtype's HexKL entry point
 * wants a pre-packed input, so there is nothing to unpack on the reference
 * side either), so a faster wrong answer fails instead of looking like a
 * win.
 *
 * Why this exists (u8i4 half): on device an fc_layer call costs ~545 us, of
 * which ~298 us is sdkl_npu_mm_u8i4_i32, against 69.9 us for QNN's comparable
 * FC phase. The sample's per-tile conversion is the suspected cause, but
 * sdkl ships compiled so that is an inference. This measured it before any
 * skel work was committed -- doc 13 has the verified result. The u8i8 half
 * exists to test a specific prediction: same tile count, double the weight
 * bytes, so the pipeline should sit right at the compute-bound/DMA-bound
 * crossover (see doc 13's successor doc for the measured verdict).
 */

#include "AEEStdErr.h"
#include "remote.h"
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hmx_hexagon_protos.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hexkl_micro.h"
#include "hexkl_macro.h"

/*
 * Cycle counter. The addon example builds standalone for the Hexagon
 * simulator, where the FastRPC-oriented HAP_perf may not link, so the
 * simulator timer is the default. Only the ratio between the two kernels is
 * read, so any monotonic cycle count will do -- if neither builds, substitute
 * whatever counter the toolchain offers.
 *
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

// Bytes per weight tile: 32x32 values. u8i4 packs two nibbles/byte (512 B);
// u8i8 is one int8/value (1024 B). Both tiles are HEXKL_HMX_WEIGHTS_ALIGNMENT
// (128) aligned automatically since both divide it evenly -- see plan_vtcm.
#define WT_TILE_BYTES_U8I4 (512U)
#define WT_TILE_BYTES_U8I8 (1024U)
// Bytes per int32 accumulator tile: 64x32 values. Same for both dtypes.
#define ACC_TILE_BYTES                                                       \
  (HEXKL_HMX_INT8_BLOCK_N_ROW * HEXKL_HMX_INT8_BLOCK_N_COL * 4U)

static uint32_t align_up(uint32_t v, uint32_t a) {
  return (v + a - 1u) / a * a;
}

/* ===================================================================
 * dtype parametrisation. u8i4 and u8i8 share every micro-API tile op's
 * *shape*; the delta is the tile byte count and three symbol names. The
 * macro-level WH bake is the one place the two genuinely diverge in
 * calling convention (see macro_bake_wh below) so it is not in this table.
 * =================================================================== */
typedef enum { DT_U8I4 = 0, DT_U8I8 = 1, DT_COUNT = 2 } bench_dtype;

typedef int (*micro_wh_bake_fn)(uint8_t *vtcm_base, uint32_t weight_offset,
                                const int8_t *wt_old, uint32_t row_tile,
                                uint32_t col_tile, uint32_t wt_cols);
typedef int (*micro_mm_fn)(uint8_t *vtcm_base, uint32_t activation_offset,
                           uint32_t weight_offset);
typedef int (*macro_mm_fn)(int n_row, int n_col, int n_inner,
                           int32_t *restrict A, const uint8_t *restrict X,
                           const int8_t *restrict W);

typedef struct {
  bench_dtype id;
  const char *name;         // "u8i4" / "u8i8" -- used verbatim in FC_FIELD lines
  const char *micro_mm_name; // "hmx_mm_u8i4" / "hmx_mm_u8i8" -- for prints only
  uint32_t wt_tile_bytes;    // 512 / 1024
  int8_t wt_lo, wt_hi;       // synthetic weight value range: [-8,7] / [-128,127]
  micro_wh_bake_fn micro_bake; // hexkl_micro_hmx_rm_to_wh_i4 / _i8
  micro_mm_fn micro_mm;        // hexkl_micro_hmx_mm_u8i4 / _u8i8
  macro_mm_fn macro_mm;        // hexkl_macro_mm_u8i4 / _u8i8
} dtype_ops;

static const dtype_ops DTYPE_OPS[DT_COUNT] = {
  {DT_U8I4, "u8i4", "hmx_mm_u8i4", WT_TILE_BYTES_U8I4, -8, 7,
   hexkl_micro_hmx_rm_to_wh_i4, hexkl_micro_hmx_mm_u8i4, hexkl_macro_mm_u8i4},
  {DT_U8I8, "u8i8", "hmx_mm_u8i8", WT_TILE_BYTES_U8I8, -128, 127,
   hexkl_micro_hmx_rm_to_wh_i8, hexkl_micro_hmx_mm_u8i8, hexkl_macro_mm_u8i8},
};

/*!
  @brief Bake row-major weight into WH layout, dtype-dispatched.

  Trap: the two macro-level converters are NOT parallel. `hexkl_macro_i4_rm_
  to_i4_wh(dst, src, n_col, n_inner)` is non-in-place (packs into a
  half-size dst, src untouched). `hexkl_macro_i8_rm_to_i8_wh_inplace(n_row,
  n_col, W)` is in-place (dims first, overwrites W) -- so for i8 the row-major
  copy going into the scalar reference MUST be taken before this call, and
  `W_wh_out` must be a *separate* buffer from whatever holds that reference
  copy (the caller passes a copy already; this function bakes it in place).

  @param dt        dtype descriptor (selects the branch)
  @param W_wh_out  destination: i4 -> n_col*n_inner/2 packed bytes (untouched
                   input); i8 -> n_col*n_inner bytes, MUST already contain a
                   copy of the row-major weight (overwritten in place)
  @param W_rm_in   source row-major weight, n_col*n_inner bytes, not modified
                   (i4 branch only -- i8 ignores this and bakes W_wh_out)
  @param n_col     N (weight rows / output columns)
  @param n_inner   K (reduction dim)
*/
static int macro_bake_wh(const dtype_ops *dt, uint8_t *W_wh_out,
                         const int8_t *W_rm_in, uint32_t n_col,
                         uint32_t n_inner) {
  if (dt->id == DT_U8I4)
    return hexkl_macro_i4_rm_to_i4_wh(W_wh_out, W_rm_in, n_col, n_inner);
  // i8: in-place, dims-first. W_wh_out must already hold the row-major copy.
  return hexkl_macro_i8_rm_to_i8_wh_inplace(n_col, n_inner,
                                            (int8_t *)W_wh_out);
}

/*!
  @brief Plain C reference, dtype-agnostic. Weight is [n_inner, W_cols]
  row-major `int8_t`, already at full (unpacked) precision for BOTH dtypes:
  neither `hexkl_micro_hmx_rm_to_wh_i4` nor its i8 counterpart wants a
  pre-packed input -- i4 packs into nibbles *internally*, so the row-major
  int8 buffer the bench builds is exactly what both bakes (and this
  reference) consume. There is nothing to unpack here.
*/
static void matmul_ref_generic(uint32_t A_rows, uint32_t n_inner,
                               uint32_t W_cols, int32_t *out,
                               const uint8_t *act, const int8_t *wt_rm) {
  for (uint32_t row = 0; row < A_rows; row++) {
    for (uint32_t col = 0; col < W_cols; col++) {
      int32_t acc = 0;
      for (uint32_t it = 0; it < n_inner; it++) {
        acc += (int32_t)act[row * n_inner + it] *
               (int32_t)wt_rm[it * W_cols + col];
      }
      out[row * W_cols + col] = acc;
    }
  }
}

/*! @brief Compare two int32 buffers; returns the first differing index or -1.
 */
static long diff_index_i32(size_t n, const int32_t *a, const int32_t *b) {
  for (size_t i = 0; i < n; i++)
    if (a[i] != b[i])
      return (long)i;
  return -1;
}

/*! @brief Trim trailing padding spaces off a shape_t.name for use as an
    FC_FIELD token (the SHAPES[] table pads names for column alignment in
    the human-readable prints; embedding that padding in a marker line's
    shape=<name> token is harmless to awk's whitespace splitting, but ugly
    in the raw log, so strip it once at the point of use). */
static const char *shape_id(const char *name) {
  static char buf[32];
  size_t n = strlen(name);
  while (n > 0 && name[n - 1] == ' ')
    n--;
  size_t cp = n < sizeof(buf) - 1 ? n : sizeof(buf) - 1;
  memcpy(buf, name, cp);
  buf[cp] = 0;
  return buf;
}

/**
 * @brief VTCM partitioning for one shape + dtype.
 *
 * Weight tiles pack back to back: a tile is wt_tile_bytes and the required
 * weight alignment is 128, so no padding is needed between them (512 and
 * 1024 both divide 128... rather, both ARE multiples of 128). Activation
 * tiles and the accumulator read must both land on
 * HEXKL_HMX_ACTIVATION_ALIGNMENT.
 */
typedef struct {
  uint32_t k_tiles;    // n_inner / 32
  uint32_t n_tiles;    // W_cols / 32
  uint32_t row_bands;  // A_rows / 64
  uint32_t wt_tile_bytes; // 512 (u8i4) or 1024 (u8i8)
  uint32_t act_offset; // activations for one row band
  uint32_t act_bytes;
  uint32_t wt_offset; // every weight tile, resident
  uint32_t wt_bytes;
  uint32_t result_offset;
  uint32_t config_offset;
  bool fits; // false when VTCM cannot hold a resident weight
} vtcm_plan;

static vtcm_plan plan_vtcm(uint32_t A_rows, uint32_t n_inner, uint32_t W_cols,
                          uint32_t vtcm_size, uint32_t wt_tile_bytes) {
  vtcm_plan p;
  p.k_tiles = n_inner / HEXKL_HMX_INT8_BLOCK_N_INNER;
  p.n_tiles = W_cols / HEXKL_HMX_INT8_BLOCK_N_COL;
  p.row_bands = A_rows / HEXKL_HMX_INT8_BLOCK_N_ROW;
  p.wt_tile_bytes = wt_tile_bytes;

  p.config_offset = vtcm_size - hexkl_micro_hmx_config_size();

  p.act_offset = 0;
  p.act_bytes = p.k_tiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;

  p.wt_offset = p.act_bytes; // 2048-aligned, hence 128-aligned
  p.wt_bytes = p.k_tiles * p.n_tiles * p.wt_tile_bytes;

  p.result_offset =
    align_up(p.wt_offset + p.wt_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);

  p.fits = (p.result_offset + ACC_TILE_BYTES) <= p.config_offset;
  return p;
}

/*!
  @brief Baseline: the SDK sample's structure, weight re-derived per tile.

  The conversion sits in the innermost loop and always targets the same scratch
  offset, so each of the (row_bands * n_tiles * k_tiles) iterations re-reads the
  tile from DDR and re-converts it.
*/
static int matmul_baseline(uint8_t *vtcm_base, const vtcm_plan *p,
                           uint32_t A_rows, uint32_t n_inner, uint32_t W_cols,
                           int32_t *out, const uint8_t *act, int8_t *wt_rm,
                           const dtype_ops *dt) {
  // One scratch tile, placed where the resident array would start.
  const uint32_t wt_scratch = p->wt_offset;
  const uint32_t result_offset =
    align_up(wt_scratch + p->wt_tile_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);

  int rc = hexkl_micro_hmx_setup_acc_read_int32(vtcm_base, p->config_offset);
  if (rc != AEE_SUCCESS)
    return rc;

  for (uint32_t band = 0; band < p->row_bands; band++) {
    for (uint32_t kt = 0; kt < p->k_tiles; kt++) {
      rc = hexkl_micro_hmx_copy_submatrix_to_8b_activation(
        vtcm_base, p->act_offset + HEXKL_HMX_ACTIVATION_ALIGNMENT * kt,
        (uint8_t *)act, band, kt, A_rows, n_inner);
      if (rc != AEE_SUCCESS)
        return rc;
    }

    for (uint32_t nt = 0; nt < p->n_tiles; nt++) {
      hexkl_micro_hmx_acc_clear_int32();

      for (uint32_t kt = 0; kt < p->k_tiles; kt++) {
        rc = dt->micro_bake(vtcm_base, wt_scratch, wt_rm, kt, nt, W_cols);
        if (rc != AEE_SUCCESS)
          return rc;
        rc = dt->micro_mm(
          vtcm_base, p->act_offset + HEXKL_HMX_ACTIVATION_ALIGNMENT * kt,
          wt_scratch);
        if (rc != AEE_SUCCESS)
          return rc;
      }

      rc = hexkl_micro_hmx_acc_read_int32(vtcm_base, p->config_offset,
                                          result_offset);
      if (rc != AEE_SUCCESS)
        return rc;
      rc = hexkl_micro_hmx_copy_32b_to_submatrix(vtcm_base, result_offset, out,
                                                 band, nt, A_rows, W_cols);
      if (rc != AEE_SUCCESS)
        return rc;
    }
  }
  return AEE_SUCCESS;
}

/*!
  @brief Convert every weight tile once into its own VTCM slot.

  Call this before matmul_resident. Skipping it on a later call with the same
  weight is the point of the exercise: the tiles are still there.
*/
static int load_weight_resident(uint8_t *vtcm_base, const vtcm_plan *p,
                                uint32_t W_cols, int8_t *wt_rm,
                                const dtype_ops *dt) {
  for (uint32_t kt = 0; kt < p->k_tiles; kt++) {
    for (uint32_t nt = 0; nt < p->n_tiles; nt++) {
      const uint32_t off =
        p->wt_offset + (kt * p->n_tiles + nt) * p->wt_tile_bytes;
      int rc = dt->micro_bake(vtcm_base, off, wt_rm, kt, nt, W_cols);
      if (rc != AEE_SUCCESS)
        return rc;
    }
  }
  return AEE_SUCCESS;
}

/*!
  @brief Matmul against weights already resident in VTCM.

  Identical to the baseline except that the inner loop indexes into the
  resident tiles instead of converting: no DDR read, no layout work.
*/
static int matmul_resident(uint8_t *vtcm_base, const vtcm_plan *p,
                           uint32_t A_rows, uint32_t n_inner, uint32_t W_cols,
                           int32_t *out, const uint8_t *act,
                           const dtype_ops *dt) {
  int rc = hexkl_micro_hmx_setup_acc_read_int32(vtcm_base, p->config_offset);
  if (rc != AEE_SUCCESS)
    return rc;

  for (uint32_t band = 0; band < p->row_bands; band++) {
    for (uint32_t kt = 0; kt < p->k_tiles; kt++) {
      rc = hexkl_micro_hmx_copy_submatrix_to_8b_activation(
        vtcm_base, p->act_offset + HEXKL_HMX_ACTIVATION_ALIGNMENT * kt,
        (uint8_t *)act, band, kt, A_rows, n_inner);
      if (rc != AEE_SUCCESS)
        return rc;
    }

    for (uint32_t nt = 0; nt < p->n_tiles; nt++) {
      hexkl_micro_hmx_acc_clear_int32();

      for (uint32_t kt = 0; kt < p->k_tiles; kt++) {
        rc = dt->micro_mm(
          vtcm_base, p->act_offset + HEXKL_HMX_ACTIVATION_ALIGNMENT * kt,
          p->wt_offset + (kt * p->n_tiles + nt) * p->wt_tile_bytes);
        if (rc != AEE_SUCCESS)
          return rc;
      }

      rc = hexkl_micro_hmx_acc_read_int32(vtcm_base, p->config_offset,
                                          p->result_offset);
      if (rc != AEE_SUCCESS)
        return rc;
      rc = hexkl_micro_hmx_copy_32b_to_submatrix(vtcm_base, p->result_offset,
                                                 out, band, nt, A_rows, W_cols);
      if (rc != AEE_SUCCESS)
        return rc;
    }
  }
  return AEE_SUCCESS;
}


/*!
  @brief Resident matmul, timed into three sub-phases (fc_compare format):
    t_act = activation DDR->VTCM staging (scalar copy helper)
    t_hmx = HMX matmul tiles (the multiply itself)
    t_out = accumulator read + output VTCM->DDR write-back (scalar copy)
  Accumulates into the caller's counters so it can be averaged over iters.
*/
static int matmul_resident_timed(uint8_t *vtcm_base, const vtcm_plan *p,
                                 uint32_t A_rows, uint32_t n_inner,
                                 uint32_t W_cols, int32_t *out,
                                 const uint8_t *act, uint64_t *t_act,
                                 uint64_t *t_hmx, uint64_t *t_accread,
                                 uint64_t *t_copyout, const dtype_ops *dt) {
  uint64_t s;
  int rc = hexkl_micro_hmx_setup_acc_read_int32(vtcm_base, p->config_offset);
  if (rc != AEE_SUCCESS)
    return rc;
  for (uint32_t band = 0; band < p->row_bands; band++) {
    s = READ_PCYCLES();
    for (uint32_t kt = 0; kt < p->k_tiles; kt++) {
      rc = hexkl_micro_hmx_copy_submatrix_to_8b_activation(
        vtcm_base, p->act_offset + HEXKL_HMX_ACTIVATION_ALIGNMENT * kt,
        (uint8_t *)act, band, kt, A_rows, n_inner);
      if (rc != AEE_SUCCESS)
        return rc;
    }
    *t_act += READ_PCYCLES() - s;

    for (uint32_t nt = 0; nt < p->n_tiles; nt++) {
      s = READ_PCYCLES();
      hexkl_micro_hmx_acc_clear_int32();
      for (uint32_t kt = 0; kt < p->k_tiles; kt++) {
        rc = dt->micro_mm(
          vtcm_base, p->act_offset + HEXKL_HMX_ACTIVATION_ALIGNMENT * kt,
          p->wt_offset + (kt * p->n_tiles + nt) * p->wt_tile_bytes);
        if (rc != AEE_SUCCESS)
          return rc;
      }
      *t_hmx += READ_PCYCLES() - s;

      s = READ_PCYCLES();
      rc = hexkl_micro_hmx_acc_read_int32(vtcm_base, p->config_offset,
                                          p->result_offset);
      if (rc != AEE_SUCCESS)
        return rc;
      *t_accread += READ_PCYCLES() - s;
      s = READ_PCYCLES();
      rc = hexkl_micro_hmx_copy_32b_to_submatrix(vtcm_base, p->result_offset,
                                                 out, band, nt, A_rows, W_cols);
      if (rc != AEE_SUCCESS)
        return rc;
      *t_copyout += READ_PCYCLES() - s;
    }
  }
  return AEE_SUCCESS;
}


/* ===================================================================
 * user-DMA (Hexagon V68+), lifted/minimized from llama.cpp ggml-hexagon
 * dma-queue.{c,h}. Replaces the scalar copy_32b output write-back.
 * =================================================================== */
typedef struct __attribute__((aligned(128))) hexdma_desc2d_s {
    void *   next;
    uint32_t dst_stride:24;
    uint32_t desc_size:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t order:1;
    uint32_t done:1;
    void *   src;
    void *   dst;
    uint32_t desc_type:8;
    uint32_t reserved0:24;
    uint32_t row_size:24;
    uint32_t nrows_lo:8;
    uint32_t nrows_hi:8;
    uint32_t src_stride:24;
    uint32_t offset:24;
    uint32_t reserved1:8;
} hexdma_desc2d;

static inline void hexdma_start(void *p){ asm volatile(" release(%0):at"::"r"(p)); asm volatile(" dmstart(%0)"::"r"(p)); }
static inline unsigned hexdma_wait(void){ unsigned r=0; asm volatile(" %0 = dmwait":"=r"(r)::"memory"); return r; }

/* one blocking 2D transfer. src in VTCM (bypass L2), dst in DDR (through L2). */
static inline void hexdma_2d_vtcm_to_ddr(hexdma_desc2d *d, void *dst, const void *src,
        uint32_t dst_stride, uint32_t src_stride, uint32_t row_size, uint32_t nrows) {
    d->next = 0; d->desc_size = 1; d->desc_type = 9;
    d->src_comp = 0; d->dst_comp = 0;
    d->src_bypass = 1; /* VTCM */ d->dst_bypass = 0; /* DDR through cache */
    d->order = 0; d->done = 0; d->reserved0 = 0; d->reserved1 = 0; d->offset = 0;
    d->src = (void*)src; d->dst = dst;
    d->src_stride = src_stride; d->dst_stride = dst_stride;
    d->row_size = row_size; d->nrows_lo = nrows & 0xff; d->nrows_hi = (nrows >> 8) & 0xff;
    Q6_dccleaninva_A((void *)d);          /* push descriptor to memory for the DMA engine */
    hexdma_start(d);
    hexdma_wait();
}

/* one blocking 2D transfer, DDR -> VTCM (activation staging). */
static inline void hexdma_2d_ddr_to_vtcm(hexdma_desc2d *d, void *dst, const void *src,
        uint32_t dst_stride, uint32_t src_stride, uint32_t row_size, uint32_t nrows) {
    d->next = 0; d->desc_size = 1; d->desc_type = 9;
    d->src_comp = 0; d->dst_comp = 0;
    d->src_bypass = 0; /* DDR through cache */ d->dst_bypass = 1; /* VTCM */
    d->order = 0; d->done = 0; d->reserved0 = 0; d->reserved1 = 0; d->offset = 0;
    d->src = (void*)src; d->dst = dst;
    d->src_stride = src_stride; d->dst_stride = dst_stride;
    d->row_size = row_size; d->nrows_lo = nrows & 0xff; d->nrows_hi = (nrows >> 8) & 0xff;
    Q6_dccleaninva_A((void *)d);
    hexdma_start(d);
    hexdma_wait();
}

/* async VTCM->DDR: fill + start, DO NOT wait (for pipelining). */
static inline void hexdma_2d_vtcm_to_ddr_async(hexdma_desc2d *d, void *dst, const void *src,
        uint32_t dst_stride, uint32_t src_stride, uint32_t row_size, uint32_t nrows) {
    d->next = 0; d->desc_size = 1; d->desc_type = 9;
    d->src_comp = 0; d->dst_comp = 0;
    d->src_bypass = 1; d->dst_bypass = 0;
    d->order = 0; d->done = 0; d->reserved0 = 0; d->reserved1 = 0; d->offset = 0;
    d->src = (void*)src; d->dst = dst;
    d->src_stride = src_stride; d->dst_stride = dst_stride;
    d->row_size = row_size; d->nrows_lo = nrows & 0xff; d->nrows_hi = (nrows >> 8) & 0xff;
    Q6_dccleaninva_A((void *)d);
    hexdma_start(d);          /* no dmwait -- runs in background */
}
static inline void hexdma_poll_done(hexdma_desc2d *d) {
    while (!d->done) { unsigned r=0; asm volatile(" %0 = dmpoll":"=r"(r)::"memory"); (void)r; }
}

/*!
  @brief Resident matmul, but the per-tile output write-back is user-DMA
  (VTCM->DDR) instead of the scalar hexkl_micro_hmx_copy_32b_to_submatrix.
  acc_read (HMX acc -> VTCM) is kept (not DMA-able). Times act/hmx/accread/dma.
*/
static int matmul_resident_dma(uint8_t *vtcm_base, uint32_t vtcm_size,
                               const vtcm_plan *p, uint32_t A_rows,
                               uint32_t n_inner, uint32_t W_cols, int32_t *out,
                               const uint8_t *act, const uint8_t *wt_wh_ddr,
                               uint64_t *t_act, uint64_t *t_hmx,
                               uint64_t *t_accread, uint64_t *t_dma,
                               uint64_t *t_weight, const dtype_ops *dt) {
  uint64_t s;
  /* one descriptor in DDR (DMA faults fetching it from VTCM). Only ever ONE
     dmstart in flight -- we poll-done before each new dmstart (issuing a second
     dmstart while the engine is busy crashes). The output DMA still overlaps
     the next tile's HMX+acc_read because its wait is DEFERRED to just before
     the next dmstart. Two result slots so acc_read of tile n+1 does not clobber
     the slot tile n's DMA is still reading. */
  static hexdma_desc2d g_desc __attribute__((aligned(128)));
  const uint32_t rs[2] = { vtcm_size - 32768u,
                           vtcm_size - 32768u + ACC_TILE_BYTES };
  int pend = 0;
  int rc = hexkl_micro_hmx_setup_acc_read_int32(vtcm_base, p->config_offset);
  if (rc != AEE_SUCCESS) return rc;

  /* multi-layer reality: weights for a whole model can't all live resident in
     VTCM (u8i8 doubles the bytes vs u8i4 -- see plan_vtcm), so each matmul
     stages ITS weight DDR -> VTCM (blocking here; prefetch to hide it needs
     the dmlink ring). The bench used to pre-load this once and exclude it --
     including it is the honest per-matmul cost. */
  s = READ_PCYCLES();
  { uint32_t rs = 262144u; while (p->wt_bytes % rs) rs >>= 1;   /* largest <512KB row that divides */
    hexdma_2d_ddr_to_vtcm(&g_desc, vtcm_base + p->wt_offset, wt_wh_ddr,
                          rs, rs, rs, p->wt_bytes / rs); }  /* 256KB rows: fewest rows under the >512KB bug */
  *t_weight += READ_PCYCLES() - s;

  for (uint32_t band = 0; band < p->row_bands; band++) {
    if (pend) { hexdma_poll_done(&g_desc); pend = 0; } /* free g_desc for staging */
    /* activation staging: DDR -> VTCM via DMA (blocking) */
    s = READ_PCYCLES();
    for (uint32_t kt = 0; kt < p->k_tiles; kt++) {
      const uint8_t *asrc = act + (size_t)band * 64 * n_inner + (size_t)kt * 32;
      hexdma_2d_ddr_to_vtcm(&g_desc,
                            vtcm_base + p->act_offset + HEXKL_HMX_ACTIVATION_ALIGNMENT * kt,
                            asrc, 32, n_inner, 32, 64);
    }
    *t_act += READ_PCYCLES() - s;

    for (uint32_t nt = 0; nt < p->n_tiles; nt++) {
      const int slot = nt & 1;

      /* HMX -- runs while the PREVIOUS tile's output DMA drains in background */
      s = READ_PCYCLES();
      hexkl_micro_hmx_acc_clear_int32();
      for (uint32_t kt = 0; kt < p->k_tiles; kt++) {
        rc = dt->micro_mm(
          vtcm_base, p->act_offset + HEXKL_HMX_ACTIVATION_ALIGNMENT * kt,
          p->wt_offset + (kt * p->n_tiles + nt) * p->wt_tile_bytes);
        if (rc != AEE_SUCCESS) return rc;
      }
      *t_hmx += READ_PCYCLES() - s;

      /* acc_read into THIS slot -- prev DMA (other slot) still in flight, safe */
      s = READ_PCYCLES();
      rc = hexkl_micro_hmx_acc_read_int32(vtcm_base, p->config_offset, rs[slot]);
      if (rc != AEE_SUCCESS) return rc;
      *t_accread += READ_PCYCLES() - s;

      /* now the deferred wait for the prev DMA (it overlapped HMX+acc_read),
         then start this tile's output DMA async. Single dmstart at a time. */
      s = READ_PCYCLES();
      if (pend) { hexdma_poll_done(&g_desc); pend = 0; }
      int32_t *dst = out + (size_t)band * 64 * W_cols + (size_t)nt * 32;
      hexdma_2d_vtcm_to_ddr_async(&g_desc, dst, vtcm_base + rs[slot],
                                  W_cols * 4, 32 * 4, 32 * 4, 64);
      pend = 1;
      *t_dma += READ_PCYCLES() - s;
    }
  }
  if (pend) hexdma_poll_done(&g_desc); /* drain last output DMA */
  return AEE_SUCCESS;
}

/*!
  @brief §5 optional fix 1: bandwidth-ONLY comparison of the pathological
  32-byte-row activation DMA that matmul_resident_dma uses (k_tiles separate
  calls, row_size=32, nrows=64 each) against a single big-row transfer moving
  the SAME byte count (row_size=n_inner, nrows=M) into a flat scratch buffer.

  This does NOT replace the hot path: the big-row destination is flat
  row-major, not the tile-shuffled activation layout the mm ops need, so
  wiring it in would need an extra on-chip reshuffle step this bench does not
  implement. It answers exactly the question the spec asked -- "what would a
  bigger row buy" -- as a bandwidth number, not a verified-correct kernel.
  Dtype-independent (activation is always uint8, dtype only affects the
  weight), so the caller runs this once per shape, not once per dtype.
*/
static void bench_act_dma_rowsize(uint8_t *vtcm_base, uint32_t vtcm_size,
                                  uint32_t M, uint32_t n_inner,
                                  const uint8_t *act, const char *shname) {
  static hexdma_desc2d d __attribute__((aligned(128)));
  const size_t bytes = (size_t)M * n_inner;
  const uint32_t k_tiles = n_inner / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const uint32_t cfgsz = hexkl_micro_hmx_config_size();
  const uint32_t need = align_up((uint32_t)bytes, 2048u) + 4096u;
  if (need + cfgsz > vtcm_size) {
    printf("  [opt5.1] %s: skip, no scratch room (need=%u avail=%u)\n",
           shname, (unsigned)need, (unsigned)(vtcm_size - cfgsz));
    return;
  }
  const uint32_t scratch = vtcm_size - cfgsz - need;
  const int ITERS = 20;

  /* current path: k_tiles separate 32B-row transfers (mirrors matmul_resident_dma) */
  uint64_t t_small = 0;
  for (int it = 0; it < ITERS; it++) {
    uint64_t s = READ_PCYCLES();
    for (uint32_t kt = 0; kt < k_tiles; kt++)
      hexdma_2d_ddr_to_vtcm(&d, vtcm_base + scratch + kt * 2048u,
                            act + (size_t)kt * 32, 32, n_inner, 32, 64);
    t_small += READ_PCYCLES() - s;
  }

  /* one-shot: row_size = n_inner (a whole DRAM row), nrows = M, into a flat
     contiguous scratch buffer. */
  uint64_t t_big = 0;
  for (int it = 0; it < ITERS; it++) {
    uint64_t s = READ_PCYCLES();
    hexdma_2d_ddr_to_vtcm(&d, vtcm_base + scratch, act, n_inner, n_inner,
                          n_inner, M);
    t_big += READ_PCYCLES() - s;
  }

  double us_small = (double)t_small / ITERS, us_big = (double)t_big / ITERS;
  double gbps_small = us_small > 0 ? (double)bytes / (us_small * 1000.0) : 0.0;
  double gbps_big = us_big > 0 ? (double)bytes / (us_big * 1000.0) : 0.0;
  printf("  [opt5.1] act DMA row-size, %s (%u bytes moved): "
         "32B x%u rows/tile = %.1f us (%.2f GB/s)  vs  %uB x%u one-shot = "
         "%.1f us (%.2f GB/s)\n",
         shname, (unsigned)bytes, (unsigned)k_tiles, us_small, gbps_small,
         (unsigned)n_inner, (unsigned)M, us_big, gbps_big);
  printf("FC_FIELD dtype=act shape=%s field=opt_act_smallrow_us value=%.1f\n",
         shname, us_small);
  printf("FC_FIELD dtype=act shape=%s field=opt_act_smallrow_gbps value=%.2f\n",
         shname, gbps_small);
  printf("FC_FIELD dtype=act shape=%s field=opt_act_bigrow_us value=%.1f\n",
         shname, us_big);
  printf("FC_FIELD dtype=act shape=%s field=opt_act_bigrow_gbps value=%.2f\n",
         shname, gbps_big);
}

/*! @brief One shape to measure. All of M%64, N%32, K%32 must be zero. */
typedef struct {
  const char *name;
  uint32_t M, N, K;
  bool verify; // run the scalar reference and compare
} shape_t;

/*
 * q_proj is the shape the host-side numbers were taken at; the smaller ones
 * come first so a memory or VTCM limit shows up before the largest allocation,
 * and so that a run stopped early still leaves usable output.
 *
 * verify is off for the large shapes on purpose. The reference is scalar C --
 * q_proj alone is 64*2048*1024 = 134M multiply-accumulates, which on a
 * cycle-accurate simulator costs far more than everything being measured. The
 * kernels do not branch on the shape, so proving them on the small shapes
 * proves them; -DPROBE_VERIFY_ALL forces the rest on if you disagree.
 *
 * -DPROBE_SHAPES=N runs only the first N.
 */
static const shape_t SHAPES[] = {
  {"sample     ", 64, 128, 128, true},
  {"small      ", 64, 256, 256, true},
  {"medium     ", 64, 1024, 1024, false},
  {"q_proj     ", 64, 2048, 1024, true},
  {"ffn_up     ", 64, 3072, 1024, false},
};

#define SHAPE_COUNT (sizeof(SHAPES) / sizeof(SHAPES[0]))

#if !defined(PROBE_SHAPES)
#define PROBE_SHAPES SHAPE_COUNT
#endif

static void run_pipelined_layer(uint8_t *, uint32_t, uint32_t, uint32_t,
                                uint32_t, const uint8_t *, int32_t *,
                                const uint8_t *, const int32_t *,
                                const dtype_ops *, bool, double *, bool *);

static int run_shape(const dtype_ops *dt, uint8_t *vtcm_base,
                     uint32_t vtcm_size, const shape_t *s) {
  const uint32_t M = s->M, N = s->N, K = s->K;
  const size_t act_n = (size_t)M * K;
  const size_t out_n = (size_t)M * N;
  const size_t wt_n = (size_t)K * N;
  const char *shname = shape_id(s->name);

  vtcm_plan p = plan_vtcm(M, K, N, vtcm_size, dt->wt_tile_bytes);

  const uint32_t need = p.result_offset + ACC_TILE_BYTES;
  // Every uint32_t is cast: on Hexagon uint32_t is unsigned long, so a bare %u
  // against one is a -Wformat error, and the example builds with -Werror.
  printf("[%s dtype=%s] M=%u N=%u K=%u | tiles k=%u n=%u | wt_tile=%uB | "
         "VTCM need=%u (act=%u wt=%u) avail=%u%s\n",
         s->name, dt->name, (unsigned)M, (unsigned)N, (unsigned)K,
         (unsigned)p.k_tiles, (unsigned)p.n_tiles, (unsigned)p.wt_tile_bytes,
         (unsigned)need, (unsigned)p.act_bytes, (unsigned)p.wt_bytes,
         (unsigned)p.config_offset, p.fits ? "" : "  <-- DOES NOT FIT, skipping");
  printf("FC_FIELD dtype=%s shape=%s field=vtcm_fits value=%d\n", dt->name,
         shname, p.fits ? 1 : 0);
  if (!p.fits)
    return AEE_SUCCESS;

#if defined(PROBE_VERIFY_ALL)
  const bool verify = true;
#else
  const bool verify = s->verify;
#endif

  uint8_t *act = (uint8_t *)malloc(act_n);
  int8_t *wt_rm = (int8_t *)malloc(wt_n); // one full-precision value per byte, both dtypes
  int32_t *out_base = (int32_t *)malloc(out_n * sizeof(int32_t));
  int32_t *out_res = (int32_t *)malloc(out_n * sizeof(int32_t));
  int32_t *out_ref = verify ? (int32_t *)malloc(out_n * sizeof(int32_t)) : NULL;

  if (!act || !wt_rm || !out_base || !out_res || (verify && !out_ref)) {
    printf("  [SKIP] out of memory on the DSP heap for this shape\n");
    free(act);
    free(wt_rm);
    free(out_ref);
    free(out_base);
    free(out_res);
    return AEE_SUCCESS;
  }

  for (size_t i = 0; i < act_n; i++)
    act[i] = (uint8_t)((i * 7 + 3) & 0xFF);
  if (dt->id == DT_U8I4) {
    // Unchanged from the verified u8i4 bench: sign-extended nibble range.
    for (size_t i = 0; i < wt_n; i++) {
      int8_t v = (int8_t)((i * 5 + 1) & 0x0F);
      if (v & 0x08)
        v |= (int8_t)0xF0; // sign-extend into [-8, 7]
      wt_rm[i] = v;
    }
  } else {
    // Full int8 range (trap 5: no nibble packing anywhere in this path).
    for (size_t i = 0; i < wt_n; i++)
      wt_rm[i] = (int8_t)((i * 131 + 17) & 0xFF);
  }
  if (verify)
    matmul_ref_generic(M, K, N, out_ref, act, wt_rm);

  int rc;
  uint64_t c0, c_base, c_load, c_first, c_second;

  memset(out_base, 0, out_n * sizeof(int32_t));
  c0 = READ_PCYCLES();
  rc = matmul_baseline(vtcm_base, &p, M, K, N, out_base, act, wt_rm, dt);
  c_base = READ_PCYCLES() - c0;
  if (rc != AEE_SUCCESS) {
    printf("  [FAIL] baseline rc=%d\n", rc);
    goto done;
  }

  // First resident call: conversion up front, then the matmul.
  memset(out_res, 0, out_n * sizeof(int32_t));
  c0 = READ_PCYCLES();
  rc = load_weight_resident(vtcm_base, &p, N, wt_rm, dt);
  c_load = READ_PCYCLES() - c0;
  if (rc != AEE_SUCCESS) {
    printf("  [FAIL] weight load rc=%d\n", rc);
    goto done;
  }
  c0 = READ_PCYCLES();
  rc = matmul_resident(vtcm_base, &p, M, K, N, out_res, act, dt);
  c_first = READ_PCYCLES() - c0;
  if (rc != AEE_SUCCESS) {
    printf("  [FAIL] resident matmul rc=%d\n", rc);
    goto done;
  }

  // Second call on the same weight: the tiles are still in VTCM, so this is
  // what every call after the first costs.
  memset(out_res, 0, out_n * sizeof(int32_t));
  c0 = READ_PCYCLES();
  rc = matmul_resident(vtcm_base, &p, M, K, N, out_res, act, dt);
  c_second = READ_PCYCLES() - c0;
  if (rc != AEE_SUCCESS) {
    printf("  [FAIL] resident matmul (repeat) rc=%d\n", rc);
    goto done;
  }

  {
    if (verify) {
      const long d_base = diff_index_i32(out_n, out_ref, out_base);
      const long d_res = diff_index_i32(out_n, out_ref, out_res);
      if (d_base >= 0)
        printf("  [FAIL] baseline differs from reference at %ld (%ld vs %ld)\n",
               d_base, (long)out_ref[d_base], (long)out_base[d_base]);
      if (d_res >= 0)
        printf("  [FAIL] resident differs from reference at %ld (%ld vs %ld)\n",
               d_res, (long)out_ref[d_res], (long)out_res[d_res]);
      if (d_base < 0 && d_res < 0)
        printf("  [OK] both match the reference exactly\n");
    } else {
      // Not a correctness statement -- only that the two kernels agree. The
      // reference ran on the small shapes; see the comment on SHAPES.
      const long d_pair = diff_index_i32(out_n, out_base, out_res);
      if (d_pair >= 0)
        printf("  [FAIL] baseline and resident disagree at %ld (%ld vs %ld)\n",
               d_pair, (long)out_base[d_pair], (long)out_res[d_pair]);
      else
        printf("  [OK] baseline and resident agree (reference skipped)\n");
    }

    printf("  pcycles  baseline=%llu  load=%llu  resident_1st=%llu  "
           "resident_2nd=%llu\n",
           (unsigned long long)c_base, (unsigned long long)c_load,
           (unsigned long long)c_first, (unsigned long long)c_second);
    if (c_second > 0)
      printf("  baseline / resident_2nd = %.2fx\n",
             (double)c_base / (double)c_second);
  }


  // --- micro-API kernel in fc_compare format (steady state, resident) ---
  {
    const int T_ITERS = 20;
    uint64_t t_act = 0, t_hmx = 0, t_accread = 0, t_copyout = 0;
    int trc = AEE_SUCCESS;
    for (int it = 0; it < T_ITERS; it++) {
      memset(out_res, 0, out_n * sizeof(int32_t));
      trc = matmul_resident_timed(vtcm_base, &p, M, K, N, out_res, act, &t_act,
                                  &t_hmx, &t_accread, &t_copyout, dt);
      if (trc != AEE_SUCCESS) {
        printf("  [FAIL] timed resident rc=%d\n", trc);
        break;
      }
    }
    if (trc == AEE_SUCCESS) {
      double us_act = (double)t_act / T_ITERS;
      double us_hmx = (double)t_hmx / T_ITERS;
      double us_accread = (double)t_accread / T_ITERS;
      double us_copyout = (double)t_copyout / T_ITERS;
      double us_out = us_accread + us_copyout;
      double us_tot = us_act + us_hmx + us_out;
      printf("\n  === micro-API kernel, fc_compare format "
             "(dtype=%s M=%u N=%u K=%u, iters=%d) ===\n",
             dt->name, (unsigned)M, (unsigned)N, (unsigned)K, T_ITERS);
      printf("  Phase                                    Time (us)\n");
      printf("  [micro] engine=NPU/HMX (micro API, on-DSP, no FastRPC RTT)\n");
      printf("    Steady state (weight resident)      %9.1f   <- per-call\n",
             us_tot);
      printf("      DSP matmul (HMX %-13s)      %9.1f\n", dt->micro_mm_name, us_hmx);
      printf("      activation staging (scalar copy) %9.1f\n", us_act);
      printf("      output write-back TOTAL           %9.1f\n", us_out);
      printf("        acc_read (HMX acc->VTCM, NO DMA) %9.1f\n", us_accread);
      printf("        copy_32b (VTCM->DDR, DMA-able)   %9.1f\n", us_copyout);
      printf("  | method | compute us | per-call us | staging pct | engine |\n");
      printf("  |---|---|---|---|---|\n");
      printf("  | %s micro (resident) | %.1f | %.1f | %.0f%% | micro/HMX |\n",
             dt->name, us_tot, us_tot, 100.0 * (us_act + us_out) / us_tot);
      printf("FC_FIELD dtype=%s shape=%s field=resident_us value=%.1f\n",
             dt->name, shname, us_tot);
    }
  }


  // --- micro-API + user-DMA output write-back (steady state, resident) ---
  {
    const int D_ITERS = 20;
    uint64_t d_act = 0, d_hmx = 0, d_accread = 0, d_dma = 0, d_weight = 0,
             d_wall = 0, ws;
    /* one-time: copy the WH weight VTCM -> DDR to get a per-matmul source
       (mirrors a real model where WH weights live in DDR, staged per matmul) */
    static hexdma_desc2d setup_desc __attribute__((aligned(128)));
    uint8_t *wt_wh_ddr = (uint8_t *)malloc(p.wt_bytes);
    int drc = AEE_SUCCESS;
    if (!wt_wh_ddr) { printf("  [FAIL] wt_wh_ddr oom\n"); goto dma_done; }
    hexdma_2d_vtcm_to_ddr(&setup_desc, wt_wh_ddr, vtcm_base + p.wt_offset,
                          512, 512, 512, p.wt_bytes / 512);  /* nrows-based: single-row >512KB DMA silently corrupts */
    for (int it = 0; it < D_ITERS; it++) {
      memset(out_res, 0, out_n * sizeof(int32_t));
      ws = READ_PCYCLES();
      drc = matmul_resident_dma(vtcm_base, vtcm_size, &p, M, K, N, out_res, act,
                                wt_wh_ddr, &d_act, &d_hmx, &d_accread, &d_dma,
                                &d_weight, dt);
      d_wall += READ_PCYCLES() - ws;
      if (drc != AEE_SUCCESS) { printf("  [FAIL] dma resident rc=%d\n", drc); break; }
    }
    if (drc == AEE_SUCCESS) {
      /* correctness: compare DMA output against the scalar resident output */
      const long dmerr = verify ? diff_index_i32(out_n, out_ref, out_res)
                                : diff_index_i32(out_n, out_base, out_res);
      double a=(double)d_act/D_ITERS, h=(double)d_hmx/D_ITERS,
             ar=(double)d_accread/D_ITERS, dm=(double)d_dma/D_ITERS;
      double tot=(double)d_wall/D_ITERS;   /* wall time: phases overlap under pipelining */
      double wt=(double)d_weight/D_ITERS;
      double wt_gbps = wt > 0 ? (double)p.wt_bytes / (wt * 1000.0) : 0.0;
      printf("\n  === micro-API + user-DMA (PIPELINED), fc_compare format "
             "(dtype=%s M=%u N=%u K=%u) ===\n",
             dt->name, (unsigned)M,(unsigned)N,(unsigned)K);
      printf("    Steady state (weight resident)      %9.1f   <- per-call (wall)\n", tot);
      printf("      [phase sum w/o overlap = %.1f]\n", a+h+ar+dm+wt);
      printf("      weight DDR->VTCM (USER-DMA)      %9.1f   <- per-matmul (multi-layer), %.2f GB/s\n", wt, wt_gbps);
      printf("      DSP matmul (HMX)                 %9.1f\n", h);
      printf("      acc_read (HMX acc->VTCM)         %9.1f\n", ar);
      printf("      activation staging (USER-DMA)    %9.1f\n", a);
      printf("      output write-back (USER-DMA)     %9.1f\n", dm);
      printf("      correctness: %s\n", dmerr < 0 ? "OK (matches reference)"
                                                   : "WRONG (layout de-shuffle needed)");
      printf("  | %s micro+DMA | %.1f us | vs SDKL kernel on-DSP (see hexkl_macro block) | %s |\n",
             dt->name, tot, dmerr < 0 ? "correct" : "timing-only");
      printf("FC_FIELD dtype=%s shape=%s field=single_us value=%.1f\n", dt->name, shname, tot);
      printf("FC_FIELD dtype=%s shape=%s field=single_correct value=%s\n", dt->name, shname, dmerr < 0 ? "OK" : "WRONG");
      printf("FC_FIELD dtype=%s shape=%s field=weight_dma_gbps value=%.2f\n", dt->name, shname, wt_gbps);
    }
    /* cross-matmul weight-prefetch demo (uses same wt_wh_ddr / act / ref) */
    if (drc == AEE_SUCCESS) {
      double cross_us = 0.0; bool cross_ok = false;
      run_pipelined_layer(vtcm_base, vtcm_size, M, N, K, wt_wh_ddr, out_res,
                          act, verify ? out_ref : out_base, dt,
                          /*lite=*/false, &cross_us, &cross_ok);
      printf("FC_FIELD dtype=%s shape=%s field=cross_us value=%.1f\n", dt->name, shname, cross_us);
      printf("FC_FIELD dtype=%s shape=%s field=cross_correct value=%s\n", dt->name, shname, cross_ok ? "OK" : "WRONG");

      /* §5 optional fix 2: same cross-matmul pipeline, but ring_init() no
         longer memsets the 32KB ring per matmul (ring_push2d already writes
         every field it uses before the DMA engine can see the descriptor).
         Kept as a SEPARATE row so it cannot contaminate the core u8i4-vs-u8i8
         comparison above -- see spec §5. */
      double ringlite_us = 0.0; bool ringlite_ok = false;
      run_pipelined_layer(vtcm_base, vtcm_size, M, N, K, wt_wh_ddr, out_res,
                          act, verify ? out_ref : out_base, dt,
                          /*lite=*/true, &ringlite_us, &ringlite_ok);
      printf("FC_FIELD dtype=%s shape=%s field=opt_ring_lite_us value=%.1f\n", dt->name, shname, ringlite_us);
      printf("FC_FIELD dtype=%s shape=%s field=opt_ring_lite_correct value=%s\n", dt->name, shname, ringlite_ok ? "OK" : "WRONG");
    }
    free(wt_wh_ddr);
  dma_done:;
  }

  /* §5 optional fix 1: activation DMA row-size bandwidth comparison.
     Dtype-independent -- run once per shape (on the first dtype pass only,
     so it is not printed/duplicated twice). */
  if (dt->id == DT_U8I4)
    bench_act_dma_rowsize(vtcm_base, vtcm_size, M, K, act, shname);

done:
  free(act);
  free(wt_rm);
  free(out_ref);
  free(out_base);
  free(out_res);
  return AEE_SUCCESS;
}


/* ===================================================================
 * SDKL kernel comparison: hexkl_macro_mm_u8i4 / hexkl_macro_mm_u8i8 (the
 * DSP-side kernels that sdkl_npu_mm_u8i4 / sdkl_npu_mm_u8i8 wrap), measured
 * on-DSP with the same HAP_perf timer so the comparison to micro+DMA carries
 * NO FastRPC difference. Times the core mm plus the rm<->ah conversions that
 * the row-major sdkl_npu_mm_*_i32 (the 270us fc_compare number, u8i4) does
 * per call.
 * =================================================================== */
static void run_macro_shape(const dtype_ops *dt, const shape_t *s) {
  const uint32_t M = s->M, N = s->N, K = s->K;
  const char *shname = shape_id(s->name);
  uint32_t Mp = (M + 63) / 64 * 64, Np = (N + 31) / 32 * 32, Kp = (K + 31) / 32 * 32;
  size_t xn = (size_t)Mp * Kp, wn = (size_t)Np * Kp, an = (size_t)Mp * Np;
  const size_t wh_bytes = (dt->id == DT_U8I4) ? (wn / 2) : wn; // trap 1: i4 packs (half size), i8 doesn't
  uint8_t *act  = (uint8_t *)malloc(xn);
  int8_t  *W    = (int8_t  *)malloc(wn);            /* [N,K] row-major, NEVER modified */
  uint8_t *X_ah = (uint8_t *)malloc(xn);
  uint8_t *W_wh = (uint8_t *)malloc(wh_bytes);      /* i4: packed WH; i8: row-major copy baked in place */
  int32_t *A_ah = (int32_t *)malloc(an * sizeof(int32_t));
  int32_t *A_rm = (int32_t *)malloc(an * sizeof(int32_t));
  int32_t *ref  = (int32_t *)malloc(an * sizeof(int32_t));
  if (!act || !W || !X_ah || !W_wh || !A_ah || !A_rm || !ref) {
    printf("  [macro dtype=%s] %s SKIP (oom)\n", dt->name, shname); goto done;
  }
  for (size_t i = 0; i < xn; i++) act[i] = (uint8_t)((i * 7 + 3) & 0xFF);
  if (dt->id == DT_U8I4) {
    for (size_t i = 0; i < wn; i++) W[i] = (int8_t)((int)(i % 15) - 7);
  } else {
    for (size_t i = 0; i < wn; i++) W[i] = (int8_t)((i * 131 + 17) & 0xFF);
  }
  /* scalar reference: A[m,n] = sum_k act[m,k]*W[n,k] -- taken from the
     untouched row-major W, BEFORE any bake (trap 2: the i8 bake is in-place
     and destroys this). */
  for (uint32_t m = 0; m < M; m++)
    for (uint32_t n = 0; n < N; n++) {
      int32_t acc = 0;
      for (uint32_t k = 0; k < K; k++)
        acc += (int32_t)act[(size_t)m * K + k] * (int32_t)W[(size_t)n * K + k];
      ref[(size_t)m * N + n] = acc;
    }

  if (dt->id == DT_U8I8)
    memcpy(W_wh, W, wn); /* trap 2: i8 bake is in-place, W_wh gets its own copy */
  int rc = macro_bake_wh(dt, W_wh, W, N, K);   /* one-time weight WH bake */
  if (rc != AEE_SUCCESS) { printf("  [macro dtype=%s] %s WH bake failed 0x%x\n", dt->name, shname, rc); goto done; }

  /* warmup */
  hexkl_macro_u8_rm_to_u8_ah(X_ah, act, M, K);
  dt->macro_mm((int)M, (int)N, (int)K, A_ah, X_ah, (const int8_t *)W_wh);
  hexkl_macro_i32_ah_to_i32_rm(M, N, A_ah, A_rm);

  const int IT = 20;
  uint64_t t_ah = 0, t_mm = 0, t_rm = 0, ss;
  for (int it = 0; it < IT; it++) {
    ss = READ_PCYCLES(); hexkl_macro_u8_rm_to_u8_ah(X_ah, act, M, K);                          t_ah += READ_PCYCLES() - ss;
    ss = READ_PCYCLES(); dt->macro_mm((int)M,(int)N,(int)K, A_ah, X_ah, (const int8_t*)W_wh);   t_mm += READ_PCYCLES() - ss;
    ss = READ_PCYCLES(); hexkl_macro_i32_ah_to_i32_rm(M, N, A_ah, A_rm);                        t_rm += READ_PCYCLES() - ss;
  }
  long d = diff_index_i32(an, ref, A_rm);
  double a = (double)t_ah/IT, mm = (double)t_mm/IT, r = (double)t_rm/IT;
  printf("\n  === hexkl_macro (SDKL kernel) on-DSP, fc_compare format "
         "(dtype=%s M=%u N=%u K=%u) ===\n",
         dt->name, (unsigned)M,(unsigned)N,(unsigned)K);
  printf("    mm_%s (core kernel, AH in/out) %9.1f\n", dt->name, mm);
  printf("    + rm_to_ah (activation in)       %9.1f\n", a);
  printf("    + ah_to_rm (output out)          %9.1f\n", r);
  printf("    full row-major path (=SDKL sem)  %9.1f   <- compare vs micro+DMA\n", a+mm+r);
  printf("    correctness: %s\n", d < 0 ? "OK (matches reference)" : "WRONG");
  printf("  | hexkl_macro (SDKL) %s | core %.1f | full %.1f us | %s |\n",
         dt->name, mm, a+mm+r, d < 0 ? "correct" : "timing-only");
  printf("FC_FIELD dtype=%s shape=%s field=sdkl_us value=%.1f\n", dt->name, shname, a+mm+r);
  printf("FC_FIELD dtype=%s shape=%s field=sdkl_core_us value=%.1f\n", dt->name, shname, mm);
  printf("FC_FIELD dtype=%s shape=%s field=sdkl_correct value=%s\n", dt->name, shname, d < 0 ? "OK" : "WRONG");
done:
  free(act); free(W); free(X_ah); free(W_wh); free(A_ah); free(A_rm); free(ref);
}


/* ===================================================================
 * Minimal dmlink ring (ported from llama.cpp dma-queue) + cross-matmul
 * weight-prefetch harness. dmlink self-starts/resumes the single DMA
 * engine, so many transfers (output tiles + weight chunks) chain onto
 * ONE engine and drain during compute -- no concurrent dmstart.
 * =================================================================== */
static inline void dmlink_(void *cur, void *next) {
  asm volatile(" release(%0):at" : : "r"(next));
  asm volatile(" dmlink(%0, %1)" : : "r"(cur), "r"(next));
}
#define RINGN 256u                        /* power of two; > pushes per matmul so no wrap */
static hexdma_desc2d g_ring[RINGN] __attribute__((aligned(128)));
static uint32_t r_push, r_pop;
static hexdma_desc2d *r_tail;
static int r_started;
/*! @brief Cold/full reset: memsets the whole ring. Only needed once (static
    storage is already zero-initialized by the runtime, so even this is
    belt-and-suspenders) -- used for the one-time setup calls outside the
    timed per-matmul loop, never inside it. */
static void ring_init_full(void) {
  memset(g_ring, 0, sizeof(g_ring));
  r_push = 0; r_pop = 0; r_tail = &g_ring[RINGN - 1]; r_started = 0;
}
/*! @brief §5 optional fix 2: the per-matmul reset ring_init() used to do --
    just the scalar bookkeeping. Safe because ring_push2d() writes every
    field of the slot it touches (including `done = 0`) before that slot is
    ever handed to the DMA engine, so memset-ing the whole 32KB ring first is
    dead weight (spec §5.2: ~1.6 us of 56.3 us here, ~3%). */
static void ring_init_lite(void) {
  r_push = 0; r_pop = 0; r_tail = &g_ring[RINGN - 1]; r_started = 0;
}
static void ring_wait_idx(uint32_t i) {
  long guard = 0;
  while (!g_ring[i].done && guard++ < 50000000L) {
    unsigned r = 0; asm volatile(" %0 = dmpoll" : "=r"(r) : : "memory"); (void)r;
  }
}
static void ring_drain(void) {
  while (r_pop != r_push) { ring_wait_idx(r_pop); r_pop = (r_pop + 1) & (RINGN - 1); }
  r_started = 0;   /* engine idle after drain -> next push must dmstart */
}
static void ring_push2d(void *dst, const void *src, uint32_t ds, uint32_t ss,
                        uint32_t rs, uint32_t nr, int src_vtcm, int dst_vtcm) {
  if (((r_push + 1) & (RINGN - 1)) == r_pop) {   /* full: reclaim oldest */
    ring_wait_idx(r_pop); r_pop = (r_pop + 1) & (RINGN - 1);
  }
  hexdma_desc2d *d = &g_ring[r_push];
  d->next = 0; d->desc_size = 1; d->desc_type = 9; d->src_comp = 0; d->dst_comp = 0;
  d->src_bypass = src_vtcm ? 1 : 0; d->dst_bypass = dst_vtcm ? 1 : 0;
  d->order = 0; d->done = 0; d->reserved0 = 0; d->reserved1 = 0; d->offset = 0;
  d->src = (void *)src; d->dst = dst; d->src_stride = ss; d->dst_stride = ds;
  d->row_size = rs; d->nrows_lo = nr & 0xff; d->nrows_hi = (nr >> 8) & 0xff;
  Q6_dccleaninva_A((void *)d);
  if (!r_started) { hexdma_start(d); r_started = 1; } else { dmlink_(r_tail, d); }
  r_tail = d;
  r_push = (r_push + 1) & (RINGN - 1);
}

/*!
  @brief K matmuls back-to-back with cross-matmul weight prefetch.
  Weight double-buffered in VTCM; while matmul i computes (HMX+acc_read, no DMA
  engine), the ring streams matmul i+1's weight (chunked DMA) AND matmul i's
  output tiles -- all on one engine, hidden behind compute. Reports steady per
  matmul (excludes the first, which pays an un-hidden weight load).

  @param dt    dtype descriptor -- selects micro_mm and the weight tile size
               (VTCM budget doubles for u8i8: wt_bytes is 2x, and this
               function double-buffers it -- trap 3. res_base's fit check
               below is exactly the guard that must trigger rather than
               silently corrupt if it does not fit.)
  @param lite  false = original per-matmul ring_init_full() (the number this
               file's core u8i4-vs-u8i8 comparison uses -- unchanged from
               before this session's parametrisation). true = §5 optional
               fix 2 (ring_init_lite()), reported as a SEPARATE row by the
               caller so it cannot contaminate the core comparison.
  @param out_per_us   [out] steady-state per-matmul microseconds
  @param out_correct  [out] true if the result matches `ref`
*/
static long g_after_clear = -12345;
static void run_pipelined_layer(uint8_t *vtcm_base, uint32_t vtcm_size,
                                uint32_t M, uint32_t N, uint32_t K,
                                const uint8_t *wt_wh_ddr, int32_t *out,
                                const uint8_t *act, const int32_t *ref,
                                const dtype_ops *dt, bool lite,
                                double *out_per_us, bool *out_correct) {
  *out_per_us = 0.0;
  *out_correct = false;
  vtcm_plan p = plan_vtcm(M, K, N, vtcm_size, dt->wt_tile_bytes);
  const uint32_t kt_n = p.k_tiles, nt_n = p.n_tiles, wb = p.wt_bytes;
  /* VTCM: act | wbuf0 | wbuf1 | result slots | config */
  const uint32_t act_off = 0;
  const uint32_t wbuf[2] = { p.act_bytes, p.act_bytes + wb };
  const uint32_t RSLOTS = nt_n;   /* one slot per output tile: no intra-matmul reuse */
  const uint32_t res_base = align_up(wbuf[1] + wb, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  if (res_base + RSLOTS * ACC_TILE_BYTES > p.config_offset) {
    printf("  [pipe dtype=%s] VTCM too small (double-buffered wt_bytes=%u x2), skip\n",
           dt->name, (unsigned)wb);
    return;
  }
  const int KMM = 6;                             /* matmuls in the "layer" */

  ring_init_full();
  int rc = hexkl_micro_hmx_setup_acc_read_int32(vtcm_base, p.config_offset);
  if (rc) { printf("  [pipe dtype=%s] setup rc=%d\n", dt->name, rc); return; }
  /* stage activation once (shared), blocking */
  for (uint32_t kt = 0; kt < kt_n; kt++)
    hexdma_2d_ddr_to_vtcm(&g_ring[0],
      vtcm_base + act_off + HEXKL_HMX_ACTIVATION_ALIGNMENT * kt,
      act + (size_t)kt * 32, 32, K, 32, 64);
  ring_init_full();
  /* load matmul 0's weight into wbuf[0] (nrows-based; single-row >512KB corrupts) */
  ring_push2d(vtcm_base + wbuf[0], wt_wh_ddr, 512, 512, 512, wb / 512, 0, 1);
  ring_drain();

  uint64_t t0 = 0, twall = 0, ws;
  for (int i = 0; i < KMM; i++) {
    const uint32_t wcur = wbuf[i & 1];
    const uint32_t wnxt = wbuf[(i + 1) & 1];
    if (lite) ring_init_lite(); else ring_init_full(); /* §5.2: fresh ring each matmul (prev drained) -> no wrap */
    if (i + 1 < KMM) {
      /* B-fix-1: prefetch the ENTIRE next weight as ONE efficient async DMA
         (not nt_n tiny chunks). Issued once, up front, so it runs on
         the DMA engine in the background for the whole matmul's compute.
         Rows stay under 512KB (single-row DMA HW bug above that size). */
      uint32_t rs = 262144u;
      while (wb % rs) rs >>= 1;
      ring_push2d(vtcm_base + wnxt, wt_wh_ddr, rs, rs, rs, wb / rs, 0, 1);
    }
    ws = READ_PCYCLES();
    for (uint32_t nt = 0; nt < nt_n; nt++) {
      hexkl_micro_hmx_acc_clear_int32();
      if (i == 0 && nt == 0) {
        uint32_t sc = res_base + nt_n * ACC_TILE_BYTES;   /* scratch beyond result slots */
        hexkl_micro_hmx_acc_read_int32(vtcm_base, p.config_offset, sc);
        g_after_clear = (long)((const int32_t *)(vtcm_base + sc))[0];
      }
      for (uint32_t kt = 0; kt < kt_n; kt++) {
        int mrc = dt->micro_mm(vtcm_base,
          act_off + HEXKL_HMX_ACTIVATION_ALIGNMENT * kt,
          wcur + (kt * nt_n + nt) * p.wt_tile_bytes);
        if (mrc != AEE_SUCCESS) { printf("  [pipe dtype=%s] mm rc=0x%x i=%d nt=%u kt=%u\n", dt->name, mrc, i, (unsigned)nt, (unsigned)kt); return; }
      }
      uint32_t roff = res_base + nt * ACC_TILE_BYTES;
      int arc = hexkl_micro_hmx_acc_read_int32(vtcm_base, p.config_offset, roff);
      if (arc != AEE_SUCCESS) { printf("  [pipe dtype=%s] acc_read rc=0x%x i=%d nt=%u\n", dt->name, arc, i, (unsigned)nt); return; }
      /* output tile nt -> DDR via ring (overlaps next tiles' compute) */
      ring_push2d(out + (size_t)nt * 32, vtcm_base + roff, N * 4, 32 * 4,
                  32 * 4, 64, 1, 0);
    }
    ring_drain();                 /* next weight fully in wnxt before matmul i+1 */
    uint64_t dt_cyc = READ_PCYCLES() - ws;
    if (i >= 2) { t0 += dt_cyc; twall++; }   /* steady state only */
  }
  double per = twall ? (double)t0 / twall : 0.0;
  long d = diff_index_i32((size_t)M * N, ref, out);
  size_t ndiff = 0;
  for (size_t k = 0; k < (size_t)M * N; k++) if (ref[k] != out[k]) ndiff++;
  (void)KMM;
  printf("\n  >>> CROSS-MATMUL (weight prefetch, dtype=%s%s): %8.1f us/matmul   correctness=%s\n",
         dt->name, lite ? " ring_lite" : "", per, d < 0 ? "OK" : "WRONG");
  if (d >= 0)
    printf("      WRONG first@ idx=%ld got=%ld want=%ld ndiff=%zu\n",
           d, (long)out[d], (long)ref[d], ndiff);
  (void)twall;
  *out_per_us = per;
  *out_correct = (d < 0);
}

int main(void) {
  uint8_t *vtcm_base = NULL;
  uint32_t vtcm_size = 0;
  int res, res2;

  printf("[VTCM PROBE] start\n");

  /* SDKL-kernel (hexkl_macro) comparison FIRST -- it acquires then releases the
     HMX/VTCM resource via initialize/finalize, so the micro path can acquire it
     afterwards. (Both holding it at once times out: 0x80000401.)
     u8i4 runs before u8i8 in every loop below, same order as the original
     (pre-parametrisation) file -- so the u8i4 numbers this session reports
     are the acceptance-gate comparison against doc 13, taken under the same
     conditions (first shapes run, same session) as before. */
  if (hexkl_macro_initialize() == AEE_SUCCESS) {
    hexkl_macro_lock_hmx();
    const size_t nm = (PROBE_SHAPES) < SHAPE_COUNT ? (PROBE_SHAPES) : SHAPE_COUNT;
    for (int d = 0; d < DT_COUNT; d++)
      for (size_t i = 0; i < nm; i++)
        run_macro_shape(&DTYPE_OPS[d], &SHAPES[i]);
    hexkl_macro_unlock_hmx();
    hexkl_macro_finalize();
  } else {
    printf("  [macro] initialize failed -- skipping SDKL comparison\n");
  }

  res = hexkl_micro_hw_init(&vtcm_base, &vtcm_size, &(uint32_t){0});
  if (res != AEE_SUCCESS) {
    printf("[VTCM PROBE][ERROR] hw init failed (%d)\n", res);
    return res;
  }
  printf("[VTCM PROBE] VTCM base = %p  size = %u bytes (%.2f MiB)\n",
         (void *)vtcm_base, (unsigned)vtcm_size,
         (double)vtcm_size / (1024.0 * 1024.0));
  printf("[VTCM PROBE] hmx config = %u bytes | tile: act=%u wt_i4=%u wt_i8=%u acc=%u\n",
         (unsigned)hexkl_micro_hmx_config_size(),
         (unsigned)HEXKL_HMX_ACTIVATION_ALIGNMENT, (unsigned)WT_TILE_BYTES_U8I4,
         (unsigned)WT_TILE_BYTES_U8I8, (unsigned)ACC_TILE_BYTES);
  printf("FC_FIELD dtype=all shape=all field=vtcm_size_bytes value=%u\n",
         (unsigned)vtcm_size);

  // Held across every shape: locking per matmul is itself overhead worth
  // avoiding, and this mirrors wrapping a whole forward pass.
  res = hexkl_micro_hmx_lock();
  if (res != AEE_SUCCESS) {
    printf("[VTCM PROBE][ERROR] HMX lock failed (%d)\n", res);
    return res;
  }

  {
    const size_t n =
      (PROBE_SHAPES) < SHAPE_COUNT ? (PROBE_SHAPES) : SHAPE_COUNT;
    for (int d = 0; d < DT_COUNT; d++) {
      for (size_t i = 0; i < n; i++)
        run_shape(&DTYPE_OPS[d], vtcm_base, vtcm_size, &SHAPES[i]);
      if (n < SHAPE_COUNT)
        printf("[VTCM PROBE] dtype=%s: %u of %u shapes run (PROBE_SHAPES)\n",
               DTYPE_OPS[d].name, (unsigned)n, (unsigned)SHAPE_COUNT);
    }
  }

  res2 = hexkl_micro_hmx_unlock();
  if (res2 != AEE_SUCCESS)
    printf("[VTCM PROBE][ERROR] HMX unlock failed (%d)\n", res2);


  printf("[VTCM PROBE] done\n");
  return res2;
}
