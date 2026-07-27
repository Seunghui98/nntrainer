// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_vtcm_probe.c
 * @brief  Does keeping the WH weight resident in VTCM speed up u8i4 matmul?
 *
 * Runs on the Hexagon DSP, standalone -- no FastRPC, no nntrainer. It reports
 * the VTCM size, then for each shape runs two kernels over identical data and
 * times both:
 *
 *   baseline  the SDK sample's structure: every weight tile is re-derived from
 *             DDR inside the innermost loop, always into one 512-byte scratch
 *             slot, so nothing is ever reused.
 *   resident  every weight tile is converted once into its own VTCM slot up
 *             front; the matmul then only reads VTCM.
 *
 * "resident" is timed twice. The first call pays the conversion; the second
 * reuses what is already in VTCM and pays nothing, which is the case that
 * matters -- an fc_layer is called once per token with the same weight.
 *
 * Both kernels are checked against a plain C reference, so a faster wrong
 * answer fails instead of looking like a win.
 *
 * Why this exists: on device an fc_layer call costs ~545 us, of which ~298 us
 * is sdkl_npu_mm_u8i4_i32, against 69.9 us for QNN's comparable FC phase. The
 * sample's per-tile conversion is the suspected cause, but sdkl ships compiled
 * so that is an inference. This measures it before any skel work is committed.
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

#include "HAP_perf.h"
#include "hexkl_micro.h"

// Bytes per packed int4 weight tile: 32x32 values, two per byte.
#define WT_TILE_BYTES (512U)
// Bytes per int32 accumulator tile: 64x32 values.
#define ACC_TILE_BYTES (HEXKL_HMX_INT8_BLOCK_N_ROW * HEXKL_HMX_INT8_BLOCK_N_COL * 4U)

static uint32_t align_up(uint32_t v, uint32_t a) { return (v + a - 1u) / a * a; }

/*!
  @brief Sign-extend one 4-bit value out of a packed byte.
  @param packed Byte holding two 4-bit values
  @param index 0 for the low nibble, 1 for the high nibble
*/
static inline int8_t unpack_i4(uint8_t packed, int index) {
  int8_t val = (int8_t)(index == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F));
  if (val & 0x08)
    val |= (int8_t)0xF0;
  return val;
}

/*! @brief Pack two 4-bit values into one byte (low nibble first). */
static inline uint8_t pack_i4(int8_t lo, int8_t hi) {
  return (uint8_t)((lo & 0x0F) | ((hi & 0x0F) << 4));
}

/*!
  @brief Plain C reference. Weight is [n_inner, W_cols] row-major, packed two
         values per byte -- the layout hexkl_micro_hmx_rm_to_wh_i4 expects.
*/
static void matmul_u8i4_ref(uint32_t A_rows, uint32_t n_inner, uint32_t W_cols,
                            int32_t *out, const uint8_t *act,
                            const uint8_t *wt_packed) {
  for (uint32_t row = 0; row < A_rows; row++) {
    for (uint32_t col = 0; col < W_cols; col++) {
      int32_t acc = 0;
      for (uint32_t it = 0; it < n_inner; it++) {
        const uint32_t flat = it * W_cols + col;
        const int8_t w = unpack_i4(wt_packed[flat / 2], (int)(flat % 2));
        acc += (int32_t)act[row * n_inner + it] * (int32_t)w;
      }
      out[row * W_cols + col] = acc;
    }
  }
}

/*! @brief Compare two int32 buffers; returns the first differing index or -1. */
static long diff_index_i32(size_t n, const int32_t *a, const int32_t *b) {
  for (size_t i = 0; i < n; i++)
    if (a[i] != b[i])
      return (long)i;
  return -1;
}

/**
 * @brief VTCM partitioning for one shape.
 *
 * Weight tiles pack back to back: a tile is 512 bytes and the required weight
 * alignment is 128, so no padding is needed between them. Activation tiles and
 * the accumulator read must both land on HEXKL_HMX_ACTIVATION_ALIGNMENT.
 */
typedef struct {
  uint32_t k_tiles;      // n_inner / 32
  uint32_t n_tiles;      // W_cols / 32
  uint32_t row_bands;    // A_rows / 64
  uint32_t act_offset;   // activations for one row band
  uint32_t act_bytes;
  uint32_t wt_offset;    // every weight tile, resident
  uint32_t wt_bytes;
  uint32_t result_offset;
  uint32_t config_offset;
  bool fits;             // false when VTCM cannot hold a resident weight
} vtcm_plan;

static vtcm_plan plan_vtcm(uint32_t A_rows, uint32_t n_inner, uint32_t W_cols,
                           uint32_t vtcm_size) {
  vtcm_plan p;
  p.k_tiles = n_inner / HEXKL_HMX_INT8_BLOCK_N_INNER;
  p.n_tiles = W_cols / HEXKL_HMX_INT8_BLOCK_N_COL;
  p.row_bands = A_rows / HEXKL_HMX_INT8_BLOCK_N_ROW;

  p.config_offset = vtcm_size - hexkl_micro_hmx_config_size();

  p.act_offset = 0;
  p.act_bytes = p.k_tiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;

  p.wt_offset = p.act_bytes; // 2048-aligned, hence 128-aligned
  p.wt_bytes = p.k_tiles * p.n_tiles * WT_TILE_BYTES;

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
                           int32_t *out, const uint8_t *act, int8_t *wt_rm) {
  // One scratch tile, placed where the resident array would start.
  const uint32_t wt_scratch = p->wt_offset;
  const uint32_t result_offset =
    align_up(wt_scratch + WT_TILE_BYTES, HEXKL_HMX_ACTIVATION_ALIGNMENT);

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
        rc = hexkl_micro_hmx_rm_to_wh_i4(vtcm_base, wt_scratch, wt_rm, kt, nt,
                                         W_cols);
        if (rc != AEE_SUCCESS)
          return rc;
        rc = hexkl_micro_hmx_mm_u8i4(
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
                                uint32_t W_cols, int8_t *wt_rm) {
  for (uint32_t kt = 0; kt < p->k_tiles; kt++) {
    for (uint32_t nt = 0; nt < p->n_tiles; nt++) {
      const uint32_t off = p->wt_offset + (kt * p->n_tiles + nt) * WT_TILE_BYTES;
      int rc = hexkl_micro_hmx_rm_to_wh_i4(vtcm_base, off, wt_rm, kt, nt, W_cols);
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
                           int32_t *out, const uint8_t *act) {
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
        rc = hexkl_micro_hmx_mm_u8i4(
          vtcm_base, p->act_offset + HEXKL_HMX_ACTIVATION_ALIGNMENT * kt,
          p->wt_offset + (kt * p->n_tiles + nt) * WT_TILE_BYTES);
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

/*! @brief One shape to measure. All of M%64, N%32, K%32 must be zero. */
typedef struct {
  const char *name;
  uint32_t M, N, K;
} shape_t;

// q_proj is the shape the host-side numbers were taken at; the smaller ones
// come first so a memory or VTCM limit shows up before the largest allocation.
static const shape_t SHAPES[] = {
  {"sample     ", 64, 128, 128},
  {"small      ", 64, 256, 256},
  {"medium     ", 64, 1024, 1024},
  {"q_proj     ", 64, 2048, 1024},
  {"ffn_up     ", 64, 3072, 1024},
};

static int run_shape(uint8_t *vtcm_base, uint32_t vtcm_size, const shape_t *s) {
  const uint32_t M = s->M, N = s->N, K = s->K;
  const size_t act_n = (size_t)M * K;
  const size_t out_n = (size_t)M * N;
  const size_t wt_n = (size_t)K * N;

  vtcm_plan p = plan_vtcm(M, K, N, vtcm_size);

  const uint32_t need = p.result_offset + ACC_TILE_BYTES;
  printf("[%s] M=%u N=%u K=%u | tiles k=%u n=%u | VTCM need=%u (act=%u wt=%u) "
         "avail=%u%s\n",
         s->name, M, N, K, p.k_tiles, p.n_tiles, need, p.act_bytes, p.wt_bytes,
         p.config_offset, p.fits ? "" : "  <-- DOES NOT FIT, skipping");
  if (!p.fits)
    return AEE_SUCCESS;

  uint8_t *act = (uint8_t *)malloc(act_n);
  int8_t *wt_rm = (int8_t *)malloc(wt_n);          // one int4 per byte
  uint8_t *wt_packed = (uint8_t *)malloc(wt_n / 2); // reference input
  int32_t *out_ref = (int32_t *)malloc(out_n * sizeof(int32_t));
  int32_t *out_base = (int32_t *)malloc(out_n * sizeof(int32_t));
  int32_t *out_res = (int32_t *)malloc(out_n * sizeof(int32_t));

  if (!act || !wt_rm || !wt_packed || !out_ref || !out_base || !out_res) {
    printf("  [SKIP] out of memory on the DSP heap for this shape\n");
    free(act); free(wt_rm); free(wt_packed);
    free(out_ref); free(out_base); free(out_res);
    return AEE_SUCCESS;
  }

  for (size_t i = 0; i < act_n; i++)
    act[i] = (uint8_t)((i * 7 + 3) & 0xFF);
  for (size_t i = 0; i < wt_n; i++) {
    int8_t v = (int8_t)((i * 5 + 1) & 0x0F);
    if (v & 0x08)
      v |= (int8_t)0xF0; // sign-extend into [-8, 7]
    wt_rm[i] = v;
  }
  for (size_t i = 0; i < wt_n / 2; i++)
    wt_packed[i] = pack_i4(wt_rm[2 * i], wt_rm[2 * i + 1]);

  matmul_u8i4_ref(M, K, N, out_ref, act, wt_packed);

  int rc;
  uint64_t c0, c_base, c_load, c_first, c_second;

  memset(out_base, 0, out_n * sizeof(int32_t));
  c0 = HAP_perf_get_pcycles();
  rc = matmul_baseline(vtcm_base, &p, M, K, N, out_base, act, wt_rm);
  c_base = HAP_perf_get_pcycles() - c0;
  if (rc != AEE_SUCCESS) {
    printf("  [FAIL] baseline rc=%d\n", rc);
    goto done;
  }

  // First resident call: conversion up front, then the matmul.
  memset(out_res, 0, out_n * sizeof(int32_t));
  c0 = HAP_perf_get_pcycles();
  rc = load_weight_resident(vtcm_base, &p, N, wt_rm);
  c_load = HAP_perf_get_pcycles() - c0;
  if (rc != AEE_SUCCESS) {
    printf("  [FAIL] weight load rc=%d\n", rc);
    goto done;
  }
  c0 = HAP_perf_get_pcycles();
  rc = matmul_resident(vtcm_base, &p, M, K, N, out_res, act);
  c_first = HAP_perf_get_pcycles() - c0;
  if (rc != AEE_SUCCESS) {
    printf("  [FAIL] resident matmul rc=%d\n", rc);
    goto done;
  }

  // Second call on the same weight: the tiles are still in VTCM, so this is
  // what every call after the first costs.
  memset(out_res, 0, out_n * sizeof(int32_t));
  c0 = HAP_perf_get_pcycles();
  rc = matmul_resident(vtcm_base, &p, M, K, N, out_res, act);
  c_second = HAP_perf_get_pcycles() - c0;
  if (rc != AEE_SUCCESS) {
    printf("  [FAIL] resident matmul (repeat) rc=%d\n", rc);
    goto done;
  }

  {
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

    printf("  pcycles  baseline=%llu  load=%llu  resident_1st=%llu  "
           "resident_2nd=%llu\n",
           (unsigned long long)c_base, (unsigned long long)c_load,
           (unsigned long long)c_first, (unsigned long long)c_second);
    if (c_second > 0)
      printf("  baseline / resident_2nd = %.2fx\n",
             (double)c_base / (double)c_second);
  }

done:
  free(act); free(wt_rm); free(wt_packed);
  free(out_ref); free(out_base); free(out_res);
  return AEE_SUCCESS;
}

int main(void) {
  uint8_t *vtcm_base = NULL;
  uint32_t vtcm_size = 0;
  int res, res2;

  printf("[VTCM PROBE] start\n");

  res = hexkl_micro_hw_init(&vtcm_base, &vtcm_size);
  if (res != AEE_SUCCESS) {
    printf("[VTCM PROBE][ERROR] hw init failed (%d)\n", res);
    return res;
  }
  printf("[VTCM PROBE] VTCM base = %p  size = %u bytes (%.2f MiB)\n",
         (void *)vtcm_base, (unsigned)vtcm_size,
         (double)vtcm_size / (1024.0 * 1024.0));
  printf("[VTCM PROBE] hmx config = %u bytes | tile: act=%u wt=%u acc=%u\n",
         (unsigned)hexkl_micro_hmx_config_size(),
         (unsigned)HEXKL_HMX_ACTIVATION_ALIGNMENT, (unsigned)WT_TILE_BYTES,
         (unsigned)ACC_TILE_BYTES);

  // Held across every shape: locking per matmul is itself overhead worth
  // avoiding, and this mirrors wrapping a whole forward pass.
  res = hexkl_micro_hmx_lock();
  if (res != AEE_SUCCESS) {
    printf("[VTCM PROBE][ERROR] HMX lock failed (%d)\n", res);
    return res;
  }

  for (size_t i = 0; i < sizeof(SHAPES) / sizeof(SHAPES[0]); i++)
    run_shape(vtcm_base, vtcm_size, &SHAPES[i]);

  res2 = hexkl_micro_hmx_unlock();
  if (res2 != AEE_SUCCESS)
    printf("[VTCM PROBE][ERROR] HMX unlock failed (%d)\n", res2);

  printf("[VTCM PROBE] done\n");
  return res2;
}
