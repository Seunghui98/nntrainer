// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_gemv_probe.c
 * @brief  Does sdkl_npu_mm_u8i4_i32 really accept an unaligned n_row?
 *
 * Host-side (ARM), standalone -- no nntrainer. 1.0.0-beta2 documents this
 * kernel as accepting "arbitrary (unaligned) dimensions", handling "X layout
 * conversion and output padding internally". beta1 says no such thing, and
 * hexkl_mm.cpp pads M to 64 on the strength of an uncited comment that was
 * quite possibly copied from the u8i8 wrapper. So the claim is documentation,
 * not a measurement, and lm_head's accumulator (37 MiB at Mp=64 versus 594 KB
 * at M=1) rides on it.
 *
 * Two questions, answered in one run per case:
 *
 *   1. Is the result correct when n_row is passed unpadded?
 *      X and W are all ones, so every element of A must equal n_inner. A
 *      non-zero return code is one failure mode; silently wrong numbers are
 *      the worse one, and only a value check catches it.
 *
 *   2. Does the kernel write past row n_row-1?
 *      "Handles output padding internally" reads two ways: the kernel keeps
 *      its own padded buffer and copies n_row rows out, or it expects the
 *      caller's buffer to be padded. Under the second reading, shrinking A to
 *      n_row rows is a heap overrun.
 *
 * The probe never bets on the answer: A is always allocated at the padded
 * size and filled with a poison pattern, then only n_row is passed. Rows past
 * n_row are read back afterwards -- poison still there means the buffer can be
 * shrunk, poison gone means it cannot.
 *
 * N is varied too, aligned and not, because the same sentence covers n_col and
 * hexkl_mm.cpp throws on N % 32 != 0.
 *
 *   usage: hexkl_gemv_probe [small|lmhead|all]     (default: small)
 *
 *     small   a table of little shapes; seconds to run
 *     lmhead  the real thing, N=151936 K=1024 -- allocates ~150 MB and needs
 *             the pin budget to be there, so run hexkl_pin_probe first
 *
 * Run it alone. A failure mode here is a SIGSEGV inside SDKL, so do not fold
 * it into another binary.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// sdkl.h references CDSP_DOMAIN_ID, which comes from remote.h. beta2's header
// includes remote.h itself, but beta1's does not and this order works for both.
#include <remote.h>
#include <sdkl.h>

#define MB (1024ull * 1024ull)

// Any value the kernel cannot legitimately produce here. Results are sums of
// n_inner ones, so they are small positive integers.
#define POISON (0x5EEDBEEF)

// What hexkl_mm.cpp currently rounds n_row up to.
#define TILE_ROWS (64u)

/* ------------------------------------------------------------------ */

static double now_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

typedef struct {
  unsigned M, N, K;
  const char *note;
  // Which claim the case is evidence for. n_row and n_col turned out to be
  // governed by different rules, and a summary that pools them says nothing
  // useful about either.
  enum { CASE_CONTROL, CASE_NROW, CASE_NCOL } tests;
} probe_case;

typedef struct {
  void *X;  // uint8 activations [Mp, K], all ones
  void *W;  // int4 weights, WH layout, all ones
  void *A;  // int32 result [Mp, N], poison-filled
  size_t a_elems;
} bufs;

static void bufs_free(bufs *b) {
  if (b->X) sdkl_npu_free(b->X);
  if (b->W) sdkl_npu_free(b->W);
  if (b->A) sdkl_npu_free(b->A);
  memset(b, 0, sizeof(*b));
}

/*!
  @brief Allocate and fill the buffers for one case.

  X and W are all ones so A[m][n] must come back as K -- a known answer that
  does not depend on getting the WH permutation right, which is the point: this
  probe is about dimensions, not layout.

  A is sized for Mp rows even though only M are asked for. That is deliberate;
  see the file comment.
*/
static int bufs_init(bufs *b, unsigned M, unsigned N, unsigned K) {
  memset(b, 0, sizeof(*b));

  const unsigned Mp = ((M + TILE_ROWS - 1u) / TILE_ROWS) * TILE_ROWS;
  const size_t x_bytes = (size_t)Mp * K * sizeof(uint8_t);
  b->a_elems = (size_t)Mp * N;
  const size_t a_bytes = b->a_elems * sizeof(int32_t);

  // sdkl_cpu_i4_rm_to_i4_wh wants one i4 per int8 byte on the way in and packs
  // two per byte on the way out; over-allocating to the unpacked size keeps
  // any tile padding past N*K/2 in bounds. Same policy as quantize_qint4_weight.
  const size_t w_unpacked = (size_t)N * K * sizeof(int8_t);

  if (sdkl_npu_alloc(x_bytes, &b->X) || sdkl_npu_alloc(w_unpacked, &b->W) ||
      sdkl_npu_alloc(a_bytes, &b->A)) {
    printf("    [alloc] failed (X=%.1f MB, W=%.1f MB, A=%.1f MB)\n",
           (double)x_bytes / (double)MB, (double)w_unpacked / (double)MB,
           (double)a_bytes / (double)MB);
    bufs_free(b);
    return -1;
  }

  memset(b->X, 1, x_bytes); // every activation byte = 1

  int8_t *w_rm = (int8_t *)b->W;
  for (size_t i = 0; i < (size_t)N * K; i++)
    w_rm[i] = 1;

  // The packer is not in-place, so convert into scratch and copy back --
  // exactly what quantize_qint4_weight does.
  void *wh = NULL;
  if (sdkl_npu_alloc(w_unpacked, &wh) != 0 || wh == NULL) {
    printf("    [alloc] WH scratch failed\n");
    bufs_free(b);
    return -1;
  }
  // Device-verified argument order (see quantizer.cpp): the i4 packer takes
  // (wt_rows = K, wt_cols = N) -- the inner dimension first.
  const int rc = sdkl_cpu_i4_rm_to_i4_wh((uint8_t *)wh, w_rm, (size_t)K,
                                         (size_t)N);
  if (rc == 0)
    memcpy(b->W, wh, (size_t)N * K / 2);
  sdkl_npu_free(wh);

  if (rc != 0) {
    printf("    [pack] sdkl_cpu_i4_rm_to_i4_wh failed rc=%d\n", rc);
    bufs_free(b);
    return -1;
  }

  int32_t *A = (int32_t *)b->A;
  for (size_t i = 0; i < b->a_elems; i++)
    A[i] = POISON;

  return 0;
}

/*!
  @brief Run one case and report both answers.
  @return 0 if the result is correct, non-zero otherwise
*/
static int run_case(int domain, const probe_case *c) {
  const unsigned Mp = ((c->M + TILE_ROWS - 1u) / TILE_ROWS) * TILE_ROWS;

  printf("  --- M=%u N=%u K=%u   (Mp would be %u%s%s)\n", c->M, c->N, c->K, Mp,
         c->note ? ", " : "", c->note ? c->note : "");

  bufs b;
  if (bufs_init(&b, c->M, c->N, c->K) != 0)
    return -1;

  const double t0 = now_us();
  const int rc = sdkl_npu_mm_u8i4_i32(domain, (size_t)c->M, (size_t)c->N,
                                      (size_t)c->K, (int32_t *)b.A,
                                      (const uint8_t *)b.X,
                                      (const uint8_t *)b.W);
  const double us = now_us() - t0;

  if (rc != 0) {
    printf("      n_row=%u REJECTED, rc=%d -> the padding is required\n", c->M,
           rc);
    bufs_free(&b);
    return rc;
  }

  // --- (1) are the M rows we asked for correct? ---
  const int32_t *A = (const int32_t *)b.A;
  size_t bad = 0;
  int32_t first_bad = 0;
  size_t first_bad_at = 0;
  for (size_t i = 0; i < (size_t)c->M * c->N; i++) {
    if (A[i] != (int32_t)c->K) {
      if (bad == 0) {
        first_bad = A[i];
        first_bad_at = i;
      }
      bad++;
    }
  }

  if (bad)
    printf("      WRONG: %zu/%zu elements differ; A[%zu]=%d, expected %u\n",
           bad, (size_t)c->M * c->N, first_bad_at, first_bad, c->K);
  else
    printf("      result correct (all %zu elements = %u), %.0f us\n",
           (size_t)c->M * c->N, c->K, us);

  // --- (2) did it touch anything past row M-1? ---
  if (Mp > c->M) {
    size_t touched = 0;
    for (size_t i = (size_t)c->M * c->N; i < b.a_elems; i++)
      if (A[i] != POISON)
        touched++;

    const size_t beyond = b.a_elems - (size_t)c->M * c->N;
    if (touched == 0)
      printf("      rows %u..%u untouched (%zu poisoned elements intact)\n"
             "        -> A can be allocated at M rows: %.1f KB instead of "
             "%.1f KB\n",
             c->M, Mp - 1, beyond,
             (double)((size_t)c->M * c->N * sizeof(int32_t)) / 1024.0,
             (double)(b.a_elems * sizeof(int32_t)) / 1024.0);
    else
      printf("      rows %u..%u WERE written (%zu/%zu elements changed)\n"
             "        -> A must stay allocated at %u rows even when passing "
             "n_row=%u\n",
             c->M, Mp - 1, touched, beyond, Mp, c->M);
  }

  bufs_free(&b);
  return bad ? -2 : 0;
}

/* ------------------------------------------------------------------ */

/*!
  @brief Small shapes: unaligned M, unaligned N, and an aligned control.
*/
static int mode_small(int domain) {
  printf("\n=== mode: small ===\n");

  static const probe_case CASES[] = {
    {64, 32, 32, "aligned control -- must pass", CASE_CONTROL},
    {1, 32, 32, "the lm_head row count", CASE_NROW},
    {1, 96, 32, "N aligned, larger", CASE_NROW},
    {1, 100, 32, "N NOT a multiple of 32", CASE_NCOL},
    {7, 64, 32, "M unaligned and not 1", CASE_NROW},
    {33, 64, 64, "M straddles the tile", CASE_NROW},
  };
  const size_t n = sizeof(CASES) / sizeof(CASES[0]);

  size_t nrow_ok = 0, nrow_total = 0;
  size_t ncol_ok = 0, ncol_total = 0;
  int control_ok = 0;

  for (size_t i = 0; i < n; i++) {
    const int ok = run_case(domain, &CASES[i]) == 0;
    switch (CASES[i].tests) {
    case CASE_CONTROL: control_ok = ok; break;
    case CASE_NROW:    nrow_total++; nrow_ok += ok; break;
    case CASE_NCOL:    ncol_total++; ncol_ok += ok; break;
    }
  }

  printf("\n  --- verdict ---\n");

  if (!control_ok) {
    printf("  the aligned control failed. Something is wrong with the setup,\n"
           "  not with the kernel -- nothing below means anything.\n");
    return 1;
  }

  printf("  unaligned n_row: %zu/%zu correct -> %s\n", nrow_ok, nrow_total,
         nrow_ok == nrow_total
           ? "SUPPORTED. hexkl_mm.cpp can pass M and size C to M rows."
           : "NOT supported. Keep the Mp padding.");

  printf("  unaligned n_col: %zu/%zu correct -> %s\n", ncol_ok, ncol_total,
         ncol_ok == ncol_total
           ? "supported too; the N %% 32 throws are over-restrictive."
           : "NOT supported. The N %% 32 throws are correct, keep them.");

  printf("\n  beta2's \"accepts arbitrary (unaligned) dimensions\" therefore "
         "covers\n  %s.\n",
         (nrow_ok == nrow_total && ncol_ok != ncol_total)
           ? "n_row but not n_col"
           : (nrow_ok == nrow_total ? "both" : "neither"));

  // Only n_row governs the lm_head accumulator, so that is the exit status.
  return nrow_ok == nrow_total ? 0 : 1;
}

/*!
  @brief The real lm_head shape, plus the M=64 comparison it replaces.

  Heavy: the weight alone is 74 MB and this allocates the unpacked staging
  alongside it. Run hexkl_pin_probe first.
*/
static int mode_lmhead(int domain) {
  printf("\n=== mode: lmhead (N=151936, K=1024) ===\n");
  printf("  this allocates roughly 150 MB before the matmul; if it fails to "
         "allocate,\n  that is the pin budget answering, not this test.\n");

  static const probe_case CASES[] = {
    {1, 151936, 1024, "what lm_head actually needs", CASE_NROW},
    {64, 151936, 1024, "what the wrapper asks for today", CASE_CONTROL},
  };

  int rc = 0;
  for (size_t i = 0; i < 2; i++)
    if (run_case(domain, &CASES[i]) != 0)
      rc = 1;

  printf("\n  compare the two timings: if they are close, the kernel pads "
         "internally\n  anyway and what M=1 buys is host memory, not kernel "
         "time -- which is\n  still the 37 MiB accumulator, the memset and the "
         "pool thrash.\n");
  return rc;
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
  const char *mode = (argc > 1) ? argv[1] : "small";
  const int domain = CDSP_DOMAIN_ID;

  printf("[GEMV PROBE] mode=%s  domain=%d\n", mode, domain);

  sdkl_npu_hw_info_t hw;
  memset(&hw, 0, sizeof(hw));
  printf("[GEMV PROBE] get_hw_info rc=%d\n",
         sdkl_npu_get_hw_info(domain, &hw, 1));

  const int rc = sdkl_npu_initialize(domain, NULL, NULL);
  if (rc != 0) {
    printf("[GEMV PROBE][ERROR] sdkl_npu_initialize failed rc=%d\n", rc);
    return rc;
  }

  char version[SDKL_VERSION_STR_LEN];
  memset(version, 0, sizeof(version));
  if (sdkl_npu_get_version(domain, version) == 0)
    printf("[GEMV PROBE] CDSP version = %s\n", version);
  printf("[GEMV PROBE] the unaligned-dimension claim is a beta2 doc string; "
         "check the\n             version above is in fact beta2 before "
         "believing a PASS.\n");

  int res = 0;
  if (strcmp(mode, "lmhead") == 0) {
    res = mode_lmhead(domain);
  } else if (strcmp(mode, "all") == 0) {
    res = mode_small(domain);
    res |= mode_lmhead(domain);
  } else {
    res = mode_small(domain);
  }

  sdkl_npu_finalize(domain);
  printf("\n[GEMV PROBE] done\n");
  return res;
}
