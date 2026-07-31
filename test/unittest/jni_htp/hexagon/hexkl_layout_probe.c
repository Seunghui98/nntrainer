// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_layout_probe.c
 * @brief  What are RM, AH and WH, concretely?
 *
 * Runs on the Hexagon DSP, standalone -- no FastRPC, no nntrainer. Nothing
 * here is timed; the question is only where each element ends up.
 *
 * The method: fill a 32x32 fp16 tile with 0,1,2,...,1023, so every element
 * carries its own row-major index. fp16 represents integers exactly up to
 * 2048, so a 32x32 tile round-trips without loss. Convert the tile, then read
 * the destination back raw: whatever value sits in slot i came from RM index
 * that value. That is the permutation, read off directly rather than inferred.
 *
 * Five things it answers:
 *
 *   1. is "flat" really plain row-major?          (copy_submatrix_to_f16)
 *   2. what permutation is RM -> AH?              (rm_to_ah_f16)
 *   3. is AH -> RM its exact inverse?             (ah_to_rm_f16)
 *   4. what permutation is RM -> WH?              (rm_to_wh_f16)
 *   5. is the accumulator's AH the same AH?       (acc_read_f16)
 *
 * (5) is the one that decides the attention design. If the tile that
 * acc_read_f16 leaves in VTCM is byte-identical to what rm_to_ah_f16 would
 * produce, then the QK^T result can be fed straight into the P*V matmul as an
 * activation with no conversion, and the score matrix never leaves VTCM.
 *
 * Plus two shape probes, because a square tile cannot distinguish row from
 * column: on a 64x64 matrix, which sub-block do tile_row / tile_col actually
 * select? That settles whether the weight is read as [K][N] or [N][K], which
 * decides whether K or V is the operand that needs transposing.
 *
 * Build alongside the addon examples (same flags as
 * examples/hexkl_micro_hmx_mm_f16). Optional:
 *   -DPROBE_PRINT_GRID   dump the full 32x32 index grid per layout
 */

#include "AEEStdErr.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hexkl_micro.h"

#define TILE_R    (32U)
#define TILE_C    (32U)
#define TILE_N    (TILE_R * TILE_C)          // 1024 elements
#define TILE_BYTES (TILE_N * sizeof(_Float16)) // 2048

// Big matrix for the shape probes: 2x2 tiles, so tile (0,1) and (1,0) differ.
#define BIG_R (64U)
#define BIG_C (64U)

/* ------------------------------------------------------------------ */
/* VTCM slots. One fp16 tile each, on the activation alignment.        */
/* ------------------------------------------------------------------ */
typedef struct {
  uint32_t flat;    // RM staging, filled by copy_submatrix_to_f16
  uint32_t ah;      // rm_to_ah_f16 output
  uint32_t back;    // ah_to_rm_f16 output (round trip)
  uint32_t wh;      // rm_to_wh_f16 output
  uint32_t acc;     // acc_read_f16 output
  uint32_t scratch; // spare
  uint32_t config;  // HMX config, at the end of VTCM
} slots_t;

static slots_t plan_slots(uint32_t vtcm_size) {
  const uint32_t A = HEXKL_HMX_ACTIVATION_ALIGNMENT;
  slots_t s;
  s.flat = A * 0;
  s.ah = A * 1;
  s.back = A * 2;
  s.wh = A * 3;
  s.acc = A * 4;
  s.scratch = A * 5;
  s.config = vtcm_size - hexkl_micro_hmx_config_size();
  return s;
}

/* ------------------------------------------------------------------ */
/* Reading a tile back out of VTCM as plain fp16                       */
/* ------------------------------------------------------------------ */

/*!
  @brief Read one tile slot as 1024 integers.

  Every value written was a whole number in [0, 1023], so the fp16 round trip
  is exact and a cast is safe. Out-of-range means the slot held something that
  was never one of our indices -- reported rather than silently truncated.
*/
static int read_tile_indices(const uint8_t *vtcm_base, uint32_t offset,
                             int *out /* [TILE_N] */) {
  const _Float16 *p = (const _Float16 *)(const void *)(vtcm_base + offset);
  int bad = 0;
  for (uint32_t i = 0; i < TILE_N; i++) {
    const float v = (float)p[i];
    const int iv = (int)v;
    if ((float)iv != v || iv < 0 || iv >= (int)TILE_N) {
      out[i] = -1;
      bad++;
    } else {
      out[i] = iv;
    }
  }
  return bad;
}

/*!
  @brief Invert a permutation: where[v] = slot holding value v.
  @return number of values that never appeared (0 means a true permutation)
*/
static int invert_perm(const int *slot_to_val, int *where /* [TILE_N] */) {
  for (uint32_t v = 0; v < TILE_N; v++)
    where[v] = -1;
  for (uint32_t i = 0; i < TILE_N; i++) {
    const int v = slot_to_val[i];
    if (v >= 0 && v < (int)TILE_N)
      where[v] = (int)i;
  }
  int missing = 0;
  for (uint32_t v = 0; v < TILE_N; v++)
    if (where[v] < 0)
      missing++;
  return missing;
}

/* ------------------------------------------------------------------ */
/* Candidate layout formulas                                           */
/* ------------------------------------------------------------------ */

typedef uint32_t (*index_fn)(uint32_t r, uint32_t c);

static uint32_t f_row_major(uint32_t r, uint32_t c) { return r * TILE_C + c; }
static uint32_t f_col_major(uint32_t r, uint32_t c) { return c * TILE_R + r; }

/* Row-pair interleave, halfword granularity. Element (r,c) at byte
   (r/2)*128 + c*4 + (r%2)*2, i.e. fp16 index (r/2)*64 + c*2 + (r%2).
   This is the arrangement llama.cpp's HMX kernels use on the same unit. */
static uint32_t f_rowpair(uint32_t r, uint32_t c) {
  return (r / 2u) * 64u + c * 2u + (r % 2u);
}

/* Same idea transposed: column pairs instead of row pairs. */
static uint32_t f_colpair(uint32_t r, uint32_t c) {
  return (c / 2u) * 64u + r * 2u + (c % 2u);
}

typedef struct {
  const char *name;
  index_fn fn;
} candidate_t;

static const candidate_t CANDIDATES[] = {
  {"row-major  (r*32 + c)", f_row_major},
  {"col-major  (c*32 + r)", f_col_major},
  {"row-pair   ((r/2)*64 + c*2 + r%2)", f_rowpair},
  {"col-pair   ((c/2)*64 + r*2 + c%2)", f_colpair},
};
#define N_CANDIDATES (sizeof(CANDIDATES) / sizeof(CANDIDATES[0]))

/*!
  @brief Report which candidate formula the observed permutation matches.

  where[v] is the slot holding the element whose row-major index is v, so for
  v = r*32+c the formula must predict where[v].
*/
static void classify(const char *label, const int *where) {
  for (size_t k = 0; k < N_CANDIDATES; k++) {
    bool ok = true;
    for (uint32_t r = 0; r < TILE_R && ok; r++)
      for (uint32_t c = 0; c < TILE_C && ok; c++)
        if (where[r * TILE_C + c] != (int)CANDIDATES[k].fn(r, c))
          ok = false;
    if (ok) {
      printf("  [%s] MATCHES  %s\n", label, CANDIDATES[k].name);
      return;
    }
  }
  printf("  [%s] matches none of the candidates -- see the map below\n", label);
}

/*!
  @brief Print enough of the mapping to recognise the pattern by eye.

  One line per row r: the slot index each column of that row landed in.
  The first four rows are usually enough to read the stride off.
*/
static void print_map(const char *label, const int *where, uint32_t n_rows) {
  printf("  [%s] element (r,c) -> slot index\n", label);
  for (uint32_t r = 0; r < n_rows && r < TILE_R; r++) {
    printf("    r=%2u:", (unsigned)r);
    for (uint32_t c = 0; c < 8 && c < TILE_C; c++)
      printf(" %4d", where[r * TILE_C + c]);
    printf("  ...  c=31: %4d\n", where[r * TILE_C + (TILE_C - 1)]);
  }
}

#if defined(PROBE_PRINT_GRID)
static void print_grid(const char *label, const int *slot_to_val) {
  printf("  [%s] slot -> source RM index, 32 per line\n", label);
  for (uint32_t i = 0; i < TILE_N; i += 32) {
    printf("    %4u:", (unsigned)i);
    for (uint32_t j = 0; j < 32; j++)
      printf(" %4d", slot_to_val[i + j]);
    printf("\n");
  }
}
#endif

/* ------------------------------------------------------------------ */
/* Probes                                                              */
/* ------------------------------------------------------------------ */

/*!
  @brief 1-3: is flat row-major, what is RM->AH, and does AH->RM invert it?
*/
static int probe_ah(uint8_t *vtcm_base, const slots_t *s, const _Float16 *tile,
                    int *ah_slot_to_val /* [TILE_N], for probe 5 */) {
  int slot_to_val[TILE_N];
  int where[TILE_N];
  int rc, bad, missing;

  printf("\n=== 1. flat: is copy_submatrix_to_f16 plain row-major? ===\n");
  rc = hexkl_micro_hmx_copy_submatrix_to_f16(vtcm_base, s->flat, tile,
                                             /*tile_row=*/0, /*tile_col=*/0,
                                             TILE_R, TILE_C);
  if (rc != AEE_SUCCESS) {
    printf("  [FAIL] copy_submatrix_to_f16 rc=%d\n", rc);
    return rc;
  }
  bad = read_tile_indices(vtcm_base, s->flat, slot_to_val);
  if (bad)
    printf("  [WARN] %d slots held a value outside [0,1023]\n", bad);
  {
    bool identity = true;
    for (uint32_t i = 0; i < TILE_N; i++)
      if (slot_to_val[i] != (int)i) {
        identity = false;
        break;
      }
    printf("  flat is %s\n",
           identity ? "plain row-major (slot i holds RM index i)"
                    : "NOT plain row-major -- it is already permuted");
    if (!identity) {
      invert_perm(slot_to_val, where);
      classify("flat", where);
      print_map("flat", where, 4);
    }
  }

  printf("\n=== 2. RM -> AH ===\n");
  rc = hexkl_micro_hmx_rm_to_ah_f16(vtcm_base, /*activation_out=*/s->ah,
                                    /*flat_in=*/s->flat);
  if (rc != AEE_SUCCESS) {
    printf("  [FAIL] rm_to_ah_f16 rc=%d\n", rc);
    return rc;
  }
  bad = read_tile_indices(vtcm_base, s->ah, slot_to_val);
  memcpy(ah_slot_to_val, slot_to_val, sizeof(slot_to_val));
  if (bad)
    printf("  [WARN] %d slots outside [0,1023]\n", bad);
  missing = invert_perm(slot_to_val, where);
  if (missing)
    printf("  [WARN] %d source indices never appeared -- AH is not a pure "
           "permutation of one tile\n",
           missing);
  classify("AH", where);
  print_map("AH", where, 4);
#if defined(PROBE_PRINT_GRID)
  print_grid("AH", slot_to_val);
#endif

  printf("\n=== 3. AH -> RM: exact inverse? ===\n");
  rc = hexkl_micro_hmx_ah_to_rm_f16(vtcm_base, /*flat_out=*/s->back,
                                    /*activation_in=*/s->ah);
  if (rc != AEE_SUCCESS) {
    printf("  [FAIL] ah_to_rm_f16 rc=%d\n", rc);
    return rc;
  }
  {
    int back[TILE_N];
    read_tile_indices(vtcm_base, s->back, back);
    long first_diff = -1;
    for (uint32_t i = 0; i < TILE_N; i++)
      if (back[i] != (int)i) {
        first_diff = (long)i;
        break;
      }
    if (first_diff < 0)
      printf("  round trip is exact -- ah_to_rm inverts rm_to_ah\n");
    else
      printf("  [FAIL] differs at slot %ld (got %d, expected %ld)\n",
             first_diff, back[first_diff], first_diff);
  }
  return AEE_SUCCESS;
}

/*!
  @brief 4: what is RM -> WH?

  rm_to_wh_f16 reads the source from DDR, not from VTCM, so it takes the host
  pointer and tile coordinates rather than a flat offset.
*/
static int probe_wh(uint8_t *vtcm_base, const slots_t *s, const _Float16 *tile) {
  int slot_to_val[TILE_N];
  int where[TILE_N];

  printf("\n=== 4. RM -> WH ===\n");
  int rc = hexkl_micro_hmx_rm_to_wh_f16(vtcm_base, /*weight_offset=*/s->wh,
                                        tile, /*row_tile=*/0, /*col_tile=*/0,
                                        /*wt_cols=*/TILE_C);
  if (rc != AEE_SUCCESS) {
    printf("  [FAIL] rm_to_wh_f16 rc=%d\n", rc);
    return rc;
  }
  const int bad = read_tile_indices(vtcm_base, s->wh, slot_to_val);
  if (bad)
    printf("  [WARN] %d slots outside [0,1023]\n", bad);
  const int missing = invert_perm(slot_to_val, where);
  if (missing)
    printf("  [WARN] %d source indices never appeared\n", missing);
  classify("WH", where);
  print_map("WH", where, 4);
#if defined(PROBE_PRINT_GRID)
  print_grid("WH", slot_to_val);
#endif
  return AEE_SUCCESS;
}

/*!
  @brief 5: is the accumulator's AH the same AH that rm_to_ah produces?

  Multiplies the index tile by an identity weight and reads the accumulator
  back raw, then compares against the AH tile from probe 2. A byte-for-byte
  match means a matmul result can be reused as an activation with no
  conversion -- which is what lets an attention score matrix stay in VTCM
  between Q*K^T and P*V.
*/
static int probe_acc(uint8_t *vtcm_base, const slots_t *s, const _Float16 *eye,
                     const int *ah_slot_to_val) {
  printf("\n=== 5. accumulator layout vs rm_to_ah layout ===\n");

  int rc = hexkl_micro_hmx_setup_acc_read_f16(vtcm_base, s->config);
  if (rc != AEE_SUCCESS) {
    printf("  [FAIL] setup_acc_read_f16 rc=%d\n", rc);
    return rc;
  }
  rc = hexkl_micro_hmx_rm_to_wh_f16(vtcm_base, /*weight_offset=*/s->scratch,
                                    eye, 0, 0, TILE_C);
  if (rc != AEE_SUCCESS) {
    printf("  [FAIL] rm_to_wh_f16 (identity) rc=%d\n", rc);
    return rc;
  }

  hexkl_micro_hmx_acc_clear_f16();
  rc = hexkl_micro_hmx_mm_f16(vtcm_base, /*activation_offset=*/s->ah,
                              /*weight_offset=*/s->scratch);
  if (rc != AEE_SUCCESS) {
    printf("  [FAIL] mm_f16 rc=%d\n", rc);
    return rc;
  }
  rc = hexkl_micro_hmx_acc_read_f16(vtcm_base, s->config, /*out_offset=*/s->acc);
  if (rc != AEE_SUCCESS) {
    printf("  [FAIL] acc_read_f16 rc=%d\n", rc);
    return rc;
  }

  int acc[TILE_N];
  const int bad = read_tile_indices(vtcm_base, s->acc, acc);
  if (bad)
    printf("  [WARN] %d accumulator slots outside [0,1023] -- identity may not "
           "be the identity under this convention\n",
           bad);

  long diff = -1;
  for (uint32_t i = 0; i < TILE_N; i++)
    if (acc[i] != ah_slot_to_val[i]) {
      diff = (long)i;
      break;
    }
  if (diff < 0) {
    printf("  accumulator output is byte-identical to the rm_to_ah tile.\n");
    printf("  => a matmul result can feed the next matmul as an activation "
           "with NO conversion.\n");
  } else {
    printf("  differs from the rm_to_ah tile, first at slot %ld (acc=%d, "
           "ah=%d)\n",
           diff, acc[diff], ah_slot_to_val[diff]);
    printf("  => either the layouts differ, or W=I is not the identity under "
           "this operand convention.\n");
    int where[TILE_N];
    invert_perm(acc, where);
    classify("ACC", where);
    print_map("ACC", where, 4);
  }
  return AEE_SUCCESS;
}

/*!
  @brief Shape probe: which sub-block do tile_row / tile_col select?

  A square tile cannot tell rows from columns. On a 64x64 matrix carrying
  value r*64+c, asking for tile (0,1) and then (1,0) shows which index moves
  along which axis, and therefore whether the matrix is read as [rows][cols]
  with the passed stride being the column count.
*/
static void probe_shape(uint8_t *vtcm_base, const slots_t *s,
                        const _Float16 *big) {
  printf("\n=== 6. tile_row / tile_col on a 64x64 matrix ===\n");
  printf("  value at (r,c) is r*64+c, so the first element of a sub-block\n"
         "  names the block: 32 => rows 0-31 cols 32-63, 2048 => rows 32-63 "
         "cols 0-31\n");

  const _Float16 *p;

  int rc = hexkl_micro_hmx_copy_submatrix_to_f16(vtcm_base, s->flat, big,
                                                 /*tile_row=*/0, /*tile_col=*/1,
                                                 BIG_R, BIG_C);
  if (rc == AEE_SUCCESS) {
    p = (const _Float16 *)(const void *)(vtcm_base + s->flat);
    printf("  copy_submatrix_to_f16(tile_row=0, tile_col=1) first element = "
           "%d\n",
           (int)(float)p[0]);
  } else {
    printf("  [FAIL] copy_submatrix_to_f16(0,1) rc=%d\n", rc);
  }

  rc = hexkl_micro_hmx_copy_submatrix_to_f16(vtcm_base, s->flat, big,
                                             /*tile_row=*/1, /*tile_col=*/0,
                                             BIG_R, BIG_C);
  if (rc == AEE_SUCCESS) {
    p = (const _Float16 *)(const void *)(vtcm_base + s->flat);
    printf("  copy_submatrix_to_f16(tile_row=1, tile_col=0) first element = "
           "%d\n",
           (int)(float)p[0]);
  } else {
    printf("  [FAIL] copy_submatrix_to_f16(1,0) rc=%d\n", rc);
  }

  // The weight path takes wt_cols only -- no row count -- so this is where
  // [K][N] versus [N][K] is decided.
  rc = hexkl_micro_hmx_rm_to_wh_f16(vtcm_base, s->wh, big, /*row_tile=*/0,
                                    /*col_tile=*/1, /*wt_cols=*/BIG_C);
  if (rc == AEE_SUCCESS) {
    int slot_to_val[TILE_N];
    read_tile_indices(vtcm_base, s->wh, slot_to_val); // may report out of range
    p = (const _Float16 *)(const void *)(vtcm_base + s->wh);
    int lo = 1 << 30, hi = -1;
    for (uint32_t i = 0; i < TILE_N; i++) {
      const int v = (int)(float)p[i];
      if (v < lo)
        lo = v;
      if (v > hi)
        hi = v;
    }
    printf("  rm_to_wh_f16(row_tile=0, col_tile=1, wt_cols=64) values span "
           "[%d, %d]\n",
           lo, hi);
    printf("    rows 0-31 x cols 32-63 would span [32, 2047]\n");
    printf("    rows 32-63 x cols 0-31 would span [2048, 4063]\n");
  } else {
    printf("  [FAIL] rm_to_wh_f16(0,1) rc=%d\n", rc);
  }
}

/* ------------------------------------------------------------------ */

int main(void) {
  uint8_t *vtcm_base = NULL;
  uint32_t vtcm_size = 0;
  uint32_t hmx_fp16_rate = 0;
  int res, res2;

  printf("[LAYOUT PROBE] start\n");

  res = hexkl_micro_hw_init(&vtcm_base, &vtcm_size, &hmx_fp16_rate);
  if (res != AEE_SUCCESS) {
    printf("[LAYOUT PROBE][ERROR] hw init failed (%d)\n", res);
    return res;
  }
  printf("[LAYOUT PROBE] VTCM base=%p size=%u bytes (%.2f MiB)\n",
         (void *)vtcm_base, (unsigned)vtcm_size,
         (double)vtcm_size / (1024.0 * 1024.0));
  printf("[LAYOUT PROBE] HMX fp16 rate = %lu ops/cycle%s\n",
         (unsigned long)hmx_fp16_rate,
         hmx_fp16_rate == 0 ? "   <-- fp16 HMX NOT SUPPORTED HERE" : "");
  // The weight alignment macro is spelled differently across addon revisions;
  // only the activation one is load-bearing here, so the weight figure is
  // printed when it happens to be defined and skipped otherwise.
#if defined(HEXKL_HMX_WEIGHTS_ALIGNMENT)
  printf("[LAYOUT PROBE] activation align=%u  weight align=%u  config=%u "
         "bytes\n",
         (unsigned)HEXKL_HMX_ACTIVATION_ALIGNMENT,
         (unsigned)HEXKL_HMX_WEIGHTS_ALIGNMENT,
         (unsigned)hexkl_micro_hmx_config_size());
#else
  printf("[LAYOUT PROBE] activation align=%u  config=%u bytes\n",
         (unsigned)HEXKL_HMX_ACTIVATION_ALIGNMENT,
         (unsigned)hexkl_micro_hmx_config_size());
#endif

  const slots_t s = plan_slots(vtcm_size);
  if (s.config < HEXKL_HMX_ACTIVATION_ALIGNMENT * 6u) {
    printf("[LAYOUT PROBE][ERROR] VTCM too small for six tiles plus config\n");
    return AEE_ENOMEMORY;
  }

  _Float16 *tile = (_Float16 *)malloc(TILE_N * sizeof(_Float16));
  _Float16 *eye = (_Float16 *)malloc(TILE_N * sizeof(_Float16));
  _Float16 *big = (_Float16 *)malloc((size_t)BIG_R * BIG_C * sizeof(_Float16));
  if (!tile || !eye || !big) {
    printf("[LAYOUT PROBE][ERROR] out of DSP heap\n");
    free(tile); free(eye); free(big);
    return AEE_ENOMEMORY;
  }

  // Each element carries its own row-major index.
  for (uint32_t r = 0; r < TILE_R; r++)
    for (uint32_t c = 0; c < TILE_C; c++)
      tile[r * TILE_C + c] = (_Float16)(float)(r * TILE_C + c);

  for (uint32_t r = 0; r < TILE_R; r++)
    for (uint32_t c = 0; c < TILE_C; c++)
      eye[r * TILE_C + c] = (_Float16)(r == c ? 1.0f : 0.0f);

  // 64x64 spans 0..4095, past the 2048 limit for exact fp16 integers, so the
  // shape probe reads only the low corner values and the reported span is
  // indicative, not exact, above 2048.
  for (uint32_t r = 0; r < BIG_R; r++)
    for (uint32_t c = 0; c < BIG_C; c++)
      big[r * BIG_C + c] = (_Float16)(float)(r * BIG_C + c);

  res = hexkl_micro_hmx_lock();
  if (res != AEE_SUCCESS) {
    printf("[LAYOUT PROBE][ERROR] HMX lock failed (%d)\n", res);
    free(tile); free(eye); free(big);
    return res;
  }

  {
    int ah_slot_to_val[TILE_N];
    memset(ah_slot_to_val, 0, sizeof(ah_slot_to_val));

    if (probe_ah(vtcm_base, &s, tile, ah_slot_to_val) == AEE_SUCCESS) {
      probe_wh(vtcm_base, &s, tile);
      if (hmx_fp16_rate != 0)
        probe_acc(vtcm_base, &s, eye, ah_slot_to_val);
      else
        printf("\n=== 5. skipped: fp16 HMX not supported on this hardware ===\n");
    }
    probe_shape(vtcm_base, &s, big);
  }

  res2 = hexkl_micro_hmx_unlock();
  if (res2 != AEE_SUCCESS)
    printf("[LAYOUT PROBE][ERROR] HMX unlock failed (%d)\n", res2);

  free(tile); free(eye); free(big);
  printf("\n[LAYOUT PROBE] done\n");
  return res2;
}
