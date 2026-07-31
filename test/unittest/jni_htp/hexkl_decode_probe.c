// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_decode_probe.c
 * @brief  How much of a decode token goes into staging weights, not matmul?
 *
 * Host-side (ARM), standalone -- no nntrainer. The decode path in
 * hexkl_mm.cpp:487 memcpy's the whole WH weight from host into an NPU scratch
 * buffer on every call:
 *
 *     // --- Decode (M==1) weight staging ---
 *     std::memcpy(W_transient, wh_host, w_bytes);
 *
 * For qwen3-0.6b at fp16 that is 30 MiB per decoder layer and 840 MiB per
 * generated token across 28 layers. Measured generation is 9.8 TPS -- about
 * 102 ms per token -- and 840 MiB at a plausible memcpy rate lands in the same
 * range, which is the hypothesis this probe exists to confirm or kill.
 *
 * It reproduces both sides with the real shapes at decode's M=1 (rounded up to
 * the fp16 kernel's 32-row tile, exactly as shgemm_f32f16_f32 does):
 *
 *   staged    a host-resident WH weight, memcpy'd into NPU scratch before each
 *             matmul. What the code does today.
 *   resident  the same weight allocated in NPU memory once. What
 *             hexkl_pin_probe showed there is room for (>= 1 GiB measured).
 *
 * The difference between the two totals is what making FC weights resident
 * would buy per token. Both are timed separately from the matmuls, so the
 * answer is not just "faster" but "how much of the token was never compute".
 *
 *   usage: hexkl_decode_probe [layer|full] [iters]     (default: layer 20)
 *
 *     layer  one decoder layer's seven weights (30 MiB), looped; the per-token
 *            figure is 28x that. Cheap, and the per-byte costs it measures do
 *            not depend on how many layers exist.
 *     full   all 28 layers with their own buffers -- 840 MiB host and, if it
 *            allocates, 840 MiB NPU. The real thing: `layer` reads the same
 *            30 MiB over and over, which is friendlier to cache than a real
 *            token's walk through 840 MiB, so it can flatter the staged case.
 *
 * Run it alone.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <remote.h>
#include <sdkl.h>

#define MB (1024ull * 1024ull)
#define N_LAYERS (28U)

/* sdkl_npu_mm_f32f16_f32 wants n_row on a 32-row tile. Only
   sdkl_npu_mm_u8i4_i32 documents (and, measured, delivers) arbitrary
   dimensions -- passing M=1 here returns 0x8000040d and leaves SDKL unable to
   free its buffers afterwards. shgemm_f32f16_f32 rounds up for the same
   reason (hexkl_mm.cpp:311), so padding here is not an artifact of the probe:
   it is what the decode path really does. */
#define F16_TILE_ROWS (32U)
#define DECODE_MP (F16_TILE_ROWS) /* M=1 decode, rounded up to one tile */

static double now_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

/* qwen3-0.6b decoder weights, in sdkl terms: A[M,N] = X[M,K] * W[N,K]^T.
   hidden 1024, 16 q heads and 8 kv heads at head_dim 128, intermediate 3072. */
typedef struct {
  const char *name;
  unsigned N; /* output columns */
  unsigned K; /* inner */
} wshape;

static const wshape LAYER[] = {
  {"q_proj", 2048, 1024},   {"k_proj", 1024, 1024},
  {"v_proj", 1024, 1024},   {"o_proj", 1024, 2048},
  {"ffn_gate", 3072, 1024}, {"ffn_up", 3072, 1024},
  {"ffn_down", 1024, 3072},
};
#define N_WEIGHTS (sizeof(LAYER) / sizeof(LAYER[0]))

static size_t wbytes(const wshape *w) {
  return (size_t)w->N * w->K * sizeof(_Float16);
}

/* ------------------------------------------------------------------ */

typedef struct {
  _Float16 *host_wh; /* plain host memory: what the decode path copies from */
  void *npu_wh;      /* NPU copy, for the resident measurement (may be NULL) */
  size_t bytes;
  const wshape *shape;
} weight;

/*!
  @brief Build one weight: all ones, converted to WH, in host memory.

  All ones means A[0][n] must come back as K, which is checked once so a
  timing run cannot be quietly measuring a broken call. The conversion runs in
  an NPU buffer because that is what hexkl_mm.cpp does and the layout helpers
  document an alignment expectation; the result is then copied to host memory,
  which is where the decode path actually keeps it.
*/
static int weight_init(weight *w, const wshape *s, int want_resident) {
  memset(w, 0, sizeof(*w));
  w->shape = s;
  w->bytes = wbytes(s);

  void *tmp = NULL;
  if (sdkl_npu_alloc(w->bytes, &tmp) != 0 || tmp == NULL) {
    printf("    [%s] WH staging alloc failed (%.1f MB)\n", s->name,
           (double)w->bytes / (double)MB);
    return -1;
  }

  _Float16 *t = (_Float16 *)tmp;
  for (size_t i = 0; i < (size_t)s->N * s->K; i++)
    t[i] = (_Float16)1.0f;

  const int rc = sdkl_cpu_f16_rm_to_f16_wh_inplace(s->N, s->K, t);
  if (rc != 0) {
    printf("    [%s] rm_to_wh failed rc=%d\n", s->name, rc);
    sdkl_npu_free(tmp);
    return -1;
  }

  w->host_wh = (_Float16 *)malloc(w->bytes);
  if (w->host_wh == NULL) {
    printf("    [%s] host alloc failed (%.1f MB)\n", s->name,
           (double)w->bytes / (double)MB);
    sdkl_npu_free(tmp);
    return -1;
  }
  memcpy(w->host_wh, t, w->bytes);

  if (want_resident) {
    if (sdkl_npu_alloc(w->bytes, &w->npu_wh) == 0 && w->npu_wh != NULL)
      memcpy(w->npu_wh, t, w->bytes);
    else
      w->npu_wh = NULL; /* caller reports; the staged path still works */
  }

  sdkl_npu_free(tmp);
  return 0;
}

static void weight_free(weight *w) {
  if (w->host_wh) free(w->host_wh);
  if (w->npu_wh) sdkl_npu_free(w->npu_wh);
  memset(w, 0, sizeof(*w));
}

/* ------------------------------------------------------------------ */

typedef struct {
  double stage_us; /* host -> NPU memcpy */
  double mm_us;    /* sdkl_npu_mm_f32f16_f32 */
  size_t bytes;    /* staged */
} totals;

/*!
  @brief One decode-shaped matmul, optionally preceded by the staging memcpy.

  Decode is one token, so the real M is 1 -- but the fp16 kernel takes a
  32-row tile, so DECODE_MP rows go in and only row 0 is read back. That is
  not the probe rounding for convenience; it is what shgemm_f32f16_f32 does.
*/
static int run_one(int domain, const weight *w, void *scratch, float *X,
                   float *A, int staged, totals *t, int check) {
  const wshape *s = w->shape;
  const void *W = NULL;

  if (staged) {
    const double t0 = now_us();
    memcpy(scratch, w->host_wh, w->bytes);
    t->stage_us += now_us() - t0;
    t->bytes += w->bytes;
    W = scratch;
  } else {
    W = w->npu_wh;
  }

  const double t1 = now_us();
  const int rc = sdkl_npu_mm_f32f16_f32(domain, (int)DECODE_MP, (int)s->N,
                                        (int)s->K, A, X, (const _Float16 *)W);
  t->mm_us += now_us() - t1;

  if (rc != 0) {
    printf("    [%s] sdkl_npu_mm_f32f16_f32 failed rc=%d (0x%x)\n", s->name, rc,
           (unsigned)rc);
    printf("    once this fails, SDKL cannot free its buffers either -- the "
           "munmap errors\n    that follow are fallout, not separate "
           "problems.\n");
    return -1;
  }

  if (check) {
    /* Row 0 is the real decode row; rows 1..DECODE_MP-1 are tile padding. */
    for (unsigned n = 0; n < s->N; n++) {
      if (fabsf(A[n] - (float)s->K) > 0.5f) {
        printf("    [%s] WRONG at %u: got %.1f, expected %u\n", s->name, n,
               (double)A[n], s->K);
        return -2;
      }
    }
  }
  return 0;
}

static void report(const char *label, const totals *t, unsigned iters,
                   unsigned scale) {
  const double per_token_stage = t->stage_us / iters * scale;
  const double per_token_mm = t->mm_us / iters * scale;
  const double per_token_bytes = (double)t->bytes / iters * scale;

  printf("  %-9s  stage %8.2f ms   matmul %8.2f ms   total %8.2f ms\n", label,
         per_token_stage / 1000.0, per_token_mm / 1000.0,
         (per_token_stage + per_token_mm) / 1000.0);
  if (per_token_bytes > 0.0)
    printf("             staged %.1f MB per token at %.1f GB/s\n",
           per_token_bytes / (double)MB,
           per_token_bytes / per_token_stage / 1000.0);
}

/* ------------------------------------------------------------------ */

/*!
  @brief Per-shape NPU buffers, sized exactly.

  Not an optimisation -- a correctness requirement. hexkl_mm.cpp's NpuScratch
  keys its pool on the exact byte size because handing sdkl a buffer left over
  from a larger shape "produced wrong results on device", and an earlier
  version of this probe reproduced that: one X and one A sized for the largest
  shape gave q_proj a correct first 128 elements and zeros after. The real
  decode path allocates per size, so this does too. There are only seven
  shapes, and their sizes collapse to three, so this stays small.
*/
typedef struct {
  void *scratch; /* staging destination, wbytes(shape) */
  void *X;       /* fp32 activation, DECODE_MP * K */
  void *A;       /* fp32 result,     DECODE_MP * N */
} shapebufs;

static void shapebufs_free(shapebufs *b, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (b[i].scratch) sdkl_npu_free(b[i].scratch);
    if (b[i].X) sdkl_npu_free(b[i].X);
    if (b[i].A) sdkl_npu_free(b[i].A);
  }
}

/*!
  @brief Time the staged and resident paths over `n` weights.
  @param scale multiply per-iteration figures by this to get a per-token number
*/
static int measure(int domain, weight *ws, size_t n, unsigned iters,
                   unsigned scale) {
  shapebufs bufs[N_WEIGHTS];
  memset(bufs, 0, sizeof(bufs));

  for (size_t i = 0; i < N_WEIGHTS; i++) {
    const wshape *sh = &LAYER[i];
    const size_t x_bytes = (size_t)DECODE_MP * sh->K * sizeof(float);
    const size_t a_bytes = (size_t)DECODE_MP * sh->N * sizeof(float);
    if (sdkl_npu_alloc(wbytes(sh), &bufs[i].scratch) != 0 ||
        sdkl_npu_alloc(x_bytes, &bufs[i].X) != 0 ||
        sdkl_npu_alloc(a_bytes, &bufs[i].A) != 0) {
      printf("  buffer alloc failed for %s\n", sh->name);
      shapebufs_free(bufs, N_WEIGHTS);
      return -1;
    }
    /* Mirror the decode path: zero the padding rows, real data in row 0. */
    memset(bufs[i].X, 0, x_bytes);
    float *X = (float *)bufs[i].X;
    for (unsigned k = 0; k < sh->K; k++)
      X[k] = 1.0f;
    memset(bufs[i].A, 0, a_bytes);
  }

  /* Correctness once, before any timing. Bail on the first failure: a failed
     call leaves SDKL in a state where even the frees below report errors, and
     continuing only buries the one message that matters. */
  totals warm = {0, 0, 0};
  for (size_t i = 0; i < n; i++) {
    shapebufs *b = &bufs[i % N_WEIGHTS];
    if (run_one(domain, &ws[i], b->scratch, (float *)b->X, (float *)b->A, 1,
                &warm, 1) != 0) {
      shapebufs_free(bufs, N_WEIGHTS);
      return -1;
    }
  }
  printf("  known-answer check passed on all %zu weights\n\n", n);

  totals staged = {0, 0, 0};
  for (unsigned it = 0; it < iters; it++)
    for (size_t i = 0; i < n; i++) {
      shapebufs *b = &bufs[i % N_WEIGHTS];
      run_one(domain, &ws[i], b->scratch, (float *)b->X, (float *)b->A, 1,
              &staged, 0);
    }
  report("staged", &staged, iters, scale);

  size_t resident_n = 0;
  for (size_t i = 0; i < n; i++)
    resident_n += ws[i].npu_wh != NULL;

  if (resident_n == n) {
    totals res = {0, 0, 0};
    for (unsigned it = 0; it < iters; it++)
      for (size_t i = 0; i < n; i++) {
        shapebufs *b = &bufs[i % N_WEIGHTS];
        run_one(domain, &ws[i], b->scratch, (float *)b->X, (float *)b->A, 0,
                &res, 0);
      }
    report("resident", &res, iters, scale);

    const double saved =
      (staged.stage_us + staged.mm_us - res.stage_us - res.mm_us) / iters *
      scale / 1000.0;
    printf("\n  making these weights resident saves %.2f ms per token\n",
           saved);
    printf("  measured generation is about 102 ms per token (9.8 TPS), so "
           "that is %.0f%% of it\n",
           saved / 102.0 * 100.0);
  } else {
    printf("\n  only %zu/%zu weights could be held in NPU memory; skipping the "
           "resident\n  comparison. That is itself the answer for this "
           "configuration.\n",
           resident_n, n);
  }

  shapebufs_free(bufs, N_WEIGHTS);
  return 0;
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
  const char *mode = (argc > 1) ? argv[1] : "layer";
  const unsigned iters = (argc > 2) ? (unsigned)atoi(argv[2]) : 20u;
  const int domain = CDSP_DOMAIN_ID;
  const int full = (strcmp(mode, "full") == 0);

  size_t per_layer_bytes = 0;
  for (size_t i = 0; i < N_WEIGHTS; i++)
    per_layer_bytes += wbytes(&LAYER[i]);

  printf("[DECODE PROBE] mode=%s iters=%u domain=%d\n", mode, iters, domain);
  printf("[DECODE PROBE] qwen3-0.6b fp16: %.1f MB per layer, %.1f MB per token "
         "across %u layers\n",
         (double)per_layer_bytes / (double)MB,
         (double)per_layer_bytes * N_LAYERS / (double)MB, N_LAYERS);

  sdkl_npu_hw_info_t hw;
  memset(&hw, 0, sizeof(hw));
  sdkl_npu_get_hw_info(domain, &hw, 1);

  const int rc = sdkl_npu_initialize(domain, NULL, NULL);
  if (rc != 0) {
    printf("[DECODE PROBE][ERROR] sdkl_npu_initialize failed rc=%d\n", rc);
    return rc;
  }

  char version[SDKL_VERSION_STR_LEN];
  memset(version, 0, sizeof(version));
  if (sdkl_npu_get_version(domain, version) == 0)
    printf("[DECODE PROBE] CDSP version = %s\n", version);

  const size_t n = full ? N_WEIGHTS * N_LAYERS : N_WEIGHTS;
  const unsigned scale = full ? 1u : N_LAYERS;

  printf("\n=== building %zu weights (%.1f MB host%s) ===\n", n,
         (double)per_layer_bytes * (full ? N_LAYERS : 1) / (double)MB,
         full ? ", and as much again in NPU memory" : "");

  weight *ws = (weight *)calloc(n, sizeof(weight));
  size_t built = 0;
  for (; built < n; built++)
    if (weight_init(&ws[built], &LAYER[built % N_WEIGHTS], 1) != 0)
      break;

  int res = 1;
  if (built != n) {
    printf("  only %zu/%zu weights could be built; not enough memory for this "
           "mode\n",
           built, n);
  } else {
    if (!full)
      printf("\n  NOTE: `layer` re-reads the same %.1f MB every iteration, "
             "which is kinder to\n  cache than a real token walking %.1f MB. "
             "Run `full` for the honest number.\n",
             (double)per_layer_bytes / (double)MB,
             (double)per_layer_bytes * N_LAYERS / (double)MB);
    printf("\n=== per generated token (M=1, %u layers) ===\n", N_LAYERS);
    res = measure(domain, ws, n, iters, scale);
  }

  for (size_t i = 0; i < built; i++)
    weight_free(&ws[i]);
  free(ws);

  sdkl_npu_finalize(domain);
  printf("\n[DECODE PROBE] done\n");
  return res;
}
