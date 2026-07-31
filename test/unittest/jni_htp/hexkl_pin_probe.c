// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_pin_probe.c
 * @brief  Can one ~79 MB weight stay pinned in NPU memory while matmuls run?
 *
 * Host-side (ARM), standalone -- no nntrainer. This is the gate for putting
 * lm_head on the NPU as u8i4: the weight is 151936 x 1024 packed int4 plus
 * per-channel scales, about 79 MB, and it only pays off if it can be allocated
 * once and reused. Staging it per token would move more bytes than the Q6_K
 * CPU path it replaces.
 *
 * The earlier 48 MB figure came from a cycling probe -- many small allocations,
 * never freed -- which is a different question from "one large block". And the
 * first sdkl_npu_mm_* call permanently reserves a large internal scratch, so
 * the order matters. Hence three modes:
 *
 *   after   run a matmul first, then try to allocate. This is the realistic
 *           order: the model does FC matmuls before it reaches lm_head.
 *   before  allocate first, then try to matmul. Shows whether holding the
 *           block starves the kernel's own scratch.
 *   sweep   same as `after`, but step the size up to find where it breaks.
 *
 * Allocating is not the test. Every mode re-runs the matmul while the block is
 * held and checks the result, because a previous round of this work found that
 * over-allocating made the kernel's scratch allocation fail and corrupted
 * SDKL's internal state rather than returning an error.
 *
 *   usage: hexkl_pin_probe [after|before|sweep]   (default: sweep)
 *
 * Run it alone. A failure mode here is a SIGSEGV inside SDKL, so do not fold
 * it into another binary.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// sdkl.h references CDSP_DOMAIN_ID, which comes from remote.h; it must be
// included first. Same order every other sdkl.h consumer uses.
#include <remote.h>
#include <sdkl.h>

#define MB (1024ull * 1024ull)

// The shape that has to keep working while the block is held. Small on
// purpose: this measures whether the matmul runs at all, not how fast.
#define CHK_M (32)
#define CHK_N (32)
#define CHK_K (32)

/* lm_head for qwen3-0.6b: vocab 151936, hidden 1024, int4 packed two per byte,
   plus one fp32 scale and one int32 zero-point correction per output channel. */
static size_t lmhead_u8i4_bytes(void) {
  const size_t N = 151936, K = 1024;
  return N * K / 2 + N * sizeof(float) + N * sizeof(int32_t);
}

/* ------------------------------------------------------------------ */
/* A known-answer matmul, used to prove the kernel still works         */
/* ------------------------------------------------------------------ */

typedef struct {
  void *X; // fp32 activations  [M, K]
  void *W; // fp16 weights      [N, K], WH layout
  void *A; // fp32 result       [M, N]
  int ok;
} check_bufs;

static void check_free(check_bufs *b) {
  if (b->X) sdkl_npu_free(b->X);
  if (b->W) sdkl_npu_free(b->W);
  if (b->A) sdkl_npu_free(b->A);
  b->X = b->W = b->A = NULL;
  b->ok = 0;
}

/*!
  @brief Allocate and fill the check buffers.

  X is all ones and W is all ones, so every result element must equal K. Any
  other value means the kernel ran but produced garbage, which is the failure
  this probe most wants to catch.
*/
static int check_init(check_bufs *b) {
  memset(b, 0, sizeof(*b));

  if (sdkl_npu_alloc((size_t)CHK_M * CHK_K * sizeof(float), &b->X) ||
      sdkl_npu_alloc((size_t)CHK_N * CHK_K * sizeof(_Float16), &b->W) ||
      sdkl_npu_alloc((size_t)CHK_M * CHK_N * sizeof(float), &b->A)) {
    printf("    [check] could not allocate the check buffers\n");
    check_free(b);
    return -1;
  }

  float *X = (float *)b->X;
  for (int i = 0; i < CHK_M * CHK_K; i++)
    X[i] = 1.0f;

  _Float16 *W = (_Float16 *)b->W;
  for (int i = 0; i < CHK_N * CHK_K; i++)
    W[i] = (_Float16)1.0f;

  // beta2 renamed the layout helpers to <dtype>_rm_to_<dtype>_<layout>.
  // beta1 called this sdkl_cpu_rm_to_wh_f16_inplace.
  const int rc = sdkl_cpu_f16_rm_to_f16_wh_inplace(CHK_N, CHK_K, W);
  if (rc != 0) {
    printf("    [check] rm_to_wh failed rc=%d\n", rc);
    check_free(b);
    return -1;
  }

  b->ok = 1;
  return 0;
}

/*!
  @brief Run the check matmul and verify every element equals K.
  @return 0 on success, -1 if the call failed, -2 if the numbers are wrong
*/
static int check_run(int domain, check_bufs *b, const char *tag) {
  if (!b->ok)
    return -1;

  memset(b->A, 0, (size_t)CHK_M * CHK_N * sizeof(float));

  const int rc = sdkl_npu_mm_f32f16_f32(domain, CHK_M, CHK_N, CHK_K,
                                        (float *)b->A, (const float *)b->X,
                                        (const _Float16 *)b->W);
  if (rc != 0) {
    printf("    [check:%s] sdkl_npu_mm_f32f16_f32 FAILED rc=%d\n", tag, rc);
    return -1;
  }

  const float *A = (const float *)b->A;
  for (int i = 0; i < CHK_M * CHK_N; i++) {
    if (fabsf(A[i] - (float)CHK_K) > 0.5f) {
      printf("    [check:%s] WRONG RESULT at %d: got %.3f, expected %d\n", tag,
             i, (double)A[i], CHK_K);
      return -2;
    }
  }
  printf("    [check:%s] matmul ok\n", tag);
  return 0;
}

/* ------------------------------------------------------------------ */

/*!
  @brief Allocate `bytes`, touch both ends, and report.

  Touching matters: sdkl_npu_alloc going through fastrpc_mmap can succeed
  while the mapping is not fully backed, and a probe that only checks the
  return code would call that a pin.
*/
static void *try_pin(size_t bytes, const char *label) {
  void *p = NULL;
  const int rc = sdkl_npu_alloc(bytes, &p);
  if (rc != 0 || p == NULL) {
    printf("  [%s] alloc %.1f MB FAILED (rc=%d)\n", label,
           (double)bytes / (double)MB, rc);
    return NULL;
  }
  volatile uint8_t *q = (volatile uint8_t *)p;
  q[0] = 0xA5;
  q[bytes / 2] = 0x5A;
  q[bytes - 1] = 0xC3;
  if (q[0] != 0xA5 || q[bytes / 2] != 0x5A || q[bytes - 1] != 0xC3) {
    printf("  [%s] alloc %.1f MB returned memory that does not hold writes\n",
           label, (double)bytes / (double)MB);
    sdkl_npu_free(p);
    return NULL;
  }
  printf("  [%s] alloc %.1f MB ok at %p\n", label, (double)bytes / (double)MB,
         p);
  return p;
}

/*!
  @brief Realistic order: matmul first, then hold the weight-sized block.
*/
static int mode_after(int domain, size_t target) {
  printf("\n=== mode: after (matmul first, then allocate) ===\n");

  check_bufs b;
  if (check_init(&b) != 0)
    return -1;
  if (check_run(domain, &b, "pre") != 0) {
    check_free(&b);
    return -1;
  }

  void *pin = try_pin(target, "pin");
  if (!pin) {
    printf("  => %.1f MB cannot be pinned after the kernel reserved its "
           "scratch.\n",
           (double)target / (double)MB);
    check_free(&b);
    return 1;
  }

  int bad = 0;
  for (int i = 0; i < 3; i++)
    if (check_run(domain, &b, "held") != 0)
      bad++;

  if (bad == 0)
    printf("  => %.1f MB stays pinned and the kernel keeps working. PASS\n",
           (double)target / (double)MB);
  else
    printf("  => the block allocated, but the kernel broke while it was held "
           "(%d/3 failed). NOT usable.\n",
           bad);

  sdkl_npu_free(pin);
  check_free(&b);
  return bad == 0 ? 0 : 1;
}

/*!
  @brief Reverse order: hold the block first, then see if the kernel can still
         reserve its own scratch.
*/
static int mode_before(int domain, size_t target) {
  printf("\n=== mode: before (allocate first, then matmul) ===\n");

  void *pin = try_pin(target, "pin");
  if (!pin) {
    printf("  => cannot even allocate %.1f MB on a fresh session.\n",
           (double)target / (double)MB);
    return 1;
  }

  check_bufs b;
  if (check_init(&b) != 0) {
    printf("  => holding %.1f MB left no room for the check buffers.\n",
           (double)target / (double)MB);
    sdkl_npu_free(pin);
    return 1;
  }

  const int rc = check_run(domain, &b, "first");
  if (rc == 0)
    printf("  => the kernel reserved its scratch with %.1f MB already held. "
           "PASS\n",
           (double)target / (double)MB);
  else
    printf("  => the kernel could not run with %.1f MB held.\n",
           (double)target / (double)MB);

  check_free(&b);
  sdkl_npu_free(pin);
  return rc == 0 ? 0 : 1;
}

/*!
  @brief Step the size up to find where a single held block stops working.

  Each size is tested on its own -- allocated, exercised, freed -- so the
  answer is "how large can one resident weight be", not "how much can be
  accumulated".
*/
static int mode_sweep(int domain, size_t target) {
  printf("\n=== mode: sweep (largest single block that keeps the kernel "
         "working) ===\n");

  check_bufs b;
  if (check_init(&b) != 0)
    return -1;
  if (check_run(domain, &b, "pre") != 0) {
    check_free(&b);
    return -1;
  }

  static const size_t SIZES_MB[] = {16, 32, 48, 64, 79, 96, 128, 192, 256};
  const size_t n = sizeof(SIZES_MB) / sizeof(SIZES_MB[0]);

  size_t best = 0;
  for (size_t i = 0; i < n; i++) {
    const size_t bytes = SIZES_MB[i] * MB;
    printf("  --- %zu MB%s\n", SIZES_MB[i],
           bytes >= target && best * MB < target ? "   <-- lm_head needs this"
                                                 : "");
    void *p = try_pin(bytes, "sweep");
    if (!p)
      break;
    const int rc = check_run(domain, &b, "held");
    sdkl_npu_free(p);
    if (rc != 0) {
      printf("  the kernel stopped working at %zu MB; the last good size is "
             "%zu MB\n",
             SIZES_MB[i], best);
      break;
    }
    best = SIZES_MB[i];
  }

  printf("\n  largest single block that allocated AND left the kernel "
         "working: %zu MB\n",
         best);
  printf("  lm_head u8i4 needs %.1f MB -> %s\n", (double)target / (double)MB,
         (best * MB >= target) ? "FITS, the resident plan is viable"
                               : "DOES NOT FIT, see the fallback");

  check_free(&b);
  return 0;
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
  const char *mode = (argc > 1) ? argv[1] : "sweep";
  const int domain = CDSP_DOMAIN_ID;
  const size_t target = lmhead_u8i4_bytes();

  printf("[PIN PROBE] mode=%s  domain=%d\n", mode, domain);
  printf("[PIN PROBE] lm_head u8i4 target = %zu bytes (%.1f MB)\n", target,
         (double)target / (double)MB);

  // Documented as callable before initialize; print_info=1 makes SDKL dump it.
  sdkl_npu_hw_info_t hw;
  memset(&hw, 0, sizeof(hw));
  const int hw_rc = sdkl_npu_get_hw_info(domain, &hw, 1);
  printf("[PIN PROBE] get_hw_info rc=%d\n", hw_rc);

  // beta2 takes a config and an info struct. NULL keeps the beta1 behaviour;
  // once the header shows what the config controls, set it here and re-run to
  // see whether the budget moves.
  const int rc = sdkl_npu_initialize(domain, NULL, NULL);
  if (rc != 0) {
    printf("[PIN PROBE][ERROR] sdkl_npu_initialize failed rc=%d\n", rc);
    return rc;
  }

  char version[SDKL_VERSION_STR_LEN];
  memset(version, 0, sizeof(version));
  if (sdkl_npu_get_version(domain, version) == 0)
    printf("[PIN PROBE] CDSP version = %s\n", version);

  int res;
  if (strcmp(mode, "after") == 0)
    res = mode_after(domain, target);
  else if (strcmp(mode, "before") == 0)
    res = mode_before(domain, target);
  else
    res = mode_sweep(domain, target);

  sdkl_npu_finalize(domain);
  printf("\n[PIN PROBE] done\n");
  return res;
}
