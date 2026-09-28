/* Host check for hvx_int_epilogue.c: accuracy of the integer gate_up
   epilogue against the f32 path's grid and against exact arithmetic, on
   synthetic data shaped like the LFM2 MoE (K=2048 accumulators, per-row
   u8 activations, per-channel int4 scales with outlier columns), plus the
   structural cases: batch exponents, thread split, padding rows.

   What "passes" means: the integer path reconstructs silu(g)*u with the
   same signal-to-noise as the f32 path does -- both are u8 grids, so the
   floor is the u8 quantization noise itself -- and the two grids disagree
   by at most one u8 step on a small fraction of elements. Those numbers
   are printed; the thresholds are generous because the real gate is
   perplexity on device (doc 53 section 3). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hvx_int_epilogue.h"

static uint32_t st = 2463534242u;
static uint32_t rnd(void) {
  st ^= st << 13;
  st ^= st >> 17;
  st ^= st << 5;
  return st;
}
static double unif(void) { return (rnd() >> 8) / 16777216.0; }
static double gauss(void) {
  double u = unif() + 1e-12, v = unif();
  return sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v);
}

#define M 64
#define INTER 1792
#define N (2 * INTER)
#define ACC_STRIDE 32

/* The f32 path's per-row quantizer (hvx_quant_u8.c's formula, as the
   scalar stub quant_row writes it). */
static void quant_row_f32(const float *x, uint32_t k, uint8_t *q, float *scale,
                          int32_t *zp) {
  float lo = 0.f, hi = 0.f;
  for (uint32_t j = 0; j < k; ++j) {
    if (x[j] < lo)
      lo = x[j];
    if (x[j] > hi)
      hi = x[j];
  }
  float s = (hi - lo) / 255.f;
  if (s <= 0.f)
    s = 1.f;
  *scale = s;
  long z = lrintf(-lo / s);
  z = z < 0 ? 0 : (z > 255 ? 255 : z);
  *zp = (int32_t)z;
  for (uint32_t j = 0; j < k; ++j) {
    long v = lrintf(x[j] / s) + *zp;
    v = v < 0 ? 0 : (v > 255 ? 255 : v);
    q[j] = (uint8_t)v;
  }
}

int main(void) {
  int fail = 0;

  /* ---- 1. the sigmoid alone --------------------------------------- */
  {
    double worst = 0.0;
    for (int F = 8; F <= 40; F += 8) {
      for (int i = -20000; i <= 20000; ++i) {
        const double x = i * 20.0 / 20000.0; /* [-20, 20] */
        const int32_t gq = (int32_t)llrint(ldexp(x, F));
        if (fabs(ldexp(x, F)) >= 1073741824.0)
          continue;
        const double s = hvx_int_sigmoid_q15(gq, F) / 32768.0;
        /* against the x the Q(F) input actually holds, not the one it
           was rounded from -- at F = 8 that rounding alone is 5e-4 */
        const double r = 1.0 / (1.0 + exp(-ldexp((double)gq, -F)));
        const double d = fabs(s - r);
        if (d > worst)
          worst = d;
      }
    }
    printf("sigmoid Q15       : max abs err %.2e (want < 1e-4)\n", worst);
    if (worst >= 1e-4)
      fail = 1;
  }

  /* ---- 2. a realistic 64-row block, whole rows in one batch and in
          batches of 16 pairs -------------------------------------------- */
  int32_t *acc = (int32_t *)malloc(sizeof(int32_t) * (size_t)N * M);
  float *w_scale = (float *)malloc(sizeof(float) * N);
  float *bias = (float *)malloc(sizeof(float) * N);
  int32_t *colsum = (int32_t *)malloc(sizeof(int32_t) * N);
  float act_scale[M];
  int32_t act_zp[M];
  /* per-channel scales: lognormal around 0.008, 3% outlier columns x8 */
  for (uint32_t c = 0; c < N; ++c) {
    w_scale[c] = 0.008f * (float)exp(0.6 * gauss());
    if (unif() < 0.03)
      w_scale[c] *= 8.0f;
    bias[c] = 0.0f;
    colsum[c] = (int32_t)lrint(gauss() * 3000.0);
  }
  for (uint32_t r = 0; r < M; ++r) {
    act_scale[r] = 0.03f * (float)exp(0.4 * gauss());
    act_zp[r] = 100 + (int32_t)(rnd() % 60u);
  }
  /* accumulators: raw sums u8 x i4 over K=2048 -- gaussian with sigma
     2^13, plus the zp*colsum offset the epilogue removes, plus a few
     outlier rows (x16) and outlier columns (x10) */
  for (uint32_t r = 0; r < M; ++r) {
    const double rs = (r % 17u == 3u) ? 16.0 : 1.0;
    for (uint32_t c = 0; c < N; ++c) {
      const double cs = (c % 97u == 5u) ? 10.0 : 1.0;
      double a = gauss() * 8192.0 * rs * cs;
      acc[(size_t)c / 32u * (M * 32) + (size_t)r * 32 + (c % 32u)] =
        (int32_t)lrint(a) + act_zp[r] * colsum[c];
    }
  }
  /* tiles: slot t = n-tile t, 64 rows x 32, so tile_stride = 64*32*4 and
     the gate/up pairing is slot j <-> slot inter_ntiles + j: exactly the
     staged layout when one batch holds every pair. */
  hvx_int_wq *wq = NULL;
  if (hvx_int_wq_bake(&wq, w_scale, bias, N) != 0) {
    printf("bake failed\n");
    return 1;
  }

  /* exact reference and the f32 path's grid */
  double *h_exact = (double *)malloc(sizeof(double) * M * INTER);
  float *h_f32 = (float *)malloc(sizeof(float) * M * INTER);
  for (uint32_t r = 0; r < M; ++r) {
    for (uint32_t c = 0; c < INTER; ++c) {
      const uint32_t cu = INTER + c;
      const int32_t ag =
        acc[(size_t)c / 32u * (M * 32) + (size_t)r * 32 + (c % 32u)] -
        act_zp[r] * colsum[c];
      const int32_t au =
        acc[(size_t)cu / 32u * (M * 32) + (size_t)r * 32 + (cu % 32u)] -
        act_zp[r] * colsum[cu];
      const double g = (double)ag * act_scale[r] * w_scale[c] + bias[c];
      const double u = (double)au * act_scale[r] * w_scale[cu] + bias[cu];
      h_exact[(size_t)r * INTER + c] = g / (1.0 + exp(-g)) * u;
      const float gf = (float)ag * act_scale[r] * w_scale[c] + bias[c];
      const float uf = (float)au * act_scale[r] * w_scale[cu] + bias[cu];
      h_f32[(size_t)r * INTER + c] = gf / (1.f + expf(-gf)) * uf;
    }
  }

  /* integer path, ONE batch of all 56 pairs */
  int32_t *hq = (int32_t *)malloc(sizeof(int32_t) * M * INTER);
  int16_t he[M * 8];
  uint8_t *ah = (uint8_t *)malloc((size_t)M * INTER);
  float sc[M];
  int32_t zq[M];
  hvx_int_gu_job job;
  memset(&job, 0, sizeof job);
  job.tiles_base = (const uint8_t *)acc;
  job.tile_stride = M * 32 * 4;
  job.n_pairs = INTER / 32;
  job.g0 = 0;
  job.row_stride = ACC_STRIDE;
  job.m_count = M;
  job.act_scale = act_scale;
  job.act_zp = act_zp;
  job.colsum_w = colsum;
  job.wq = wq;
  job.inter = INTER;
  job.dst = hq;
  job.dst_stride = INTER;
  job.h_e = he;
  job.e_stride = 8;
  job.batch = 0;
  /* three threads, as the pool would split it */
  for (uint32_t t = 0; t < 3; ++t)
    hvx_int_gu_worker(3, t, &job);
  hvx_int_rq_rows(hq, INTER, he, 8, 1, INTER, INTER, M, 0, M, sc, zq, ah);

  /* the same block through batches of 16 pairs (16,16,16,8) */
  int32_t *hq2 = (int32_t *)malloc(sizeof(int32_t) * M * INTER);
  int16_t he2[M * 8];
  uint8_t *ah2 = (uint8_t *)malloc((size_t)M * INTER);
  float sc2[M];
  int32_t zq2[M];
  {
    /* stage the batch: slots [0,np) gate tiles g0.., [np,2np) up tiles */
    int32_t *stage = (int32_t *)malloc(sizeof(int32_t) * 32 * M * 32);
    for (uint32_t g0 = 0, b = 0; g0 < INTER / 32; g0 += 16, ++b) {
      const uint32_t np = (INTER / 32 - g0 < 16) ? INTER / 32 - g0 : 16;
      for (uint32_t j = 0; j < np; ++j) {
        memcpy(stage + (size_t)j * M * 32, acc + (size_t)(g0 + j) * M * 32,
               sizeof(int32_t) * M * 32);
        memcpy(stage + (size_t)(np + j) * M * 32,
               acc + (size_t)(INTER / 32 + g0 + j) * M * 32,
               sizeof(int32_t) * M * 32);
      }
      hvx_int_gu_job jb = job;
      jb.tiles_base = (const uint8_t *)stage;
      jb.n_pairs = np;
      jb.g0 = g0;
      jb.dst = hq2;
      jb.h_e = he2;
      jb.batch = b;
      hvx_int_gu_worker(1, 0, &jb);
    }
    free(stage);
    hvx_int_rq_rows(hq2, INTER, he2, 8, 4, 512, INTER, M, 0, M, sc2, zq2, ah2);
  }

  /* metrics */
  double e_int = 0, e_f32 = 0, e_int2 = 0, sig = 0;
  uint32_t n1 = 0, n2 = 0, nb1 = 0, nb2 = 0, tot = 0;
  for (uint32_t r = 0; r < M; ++r) {
    float sf;
    int32_t zf;
    uint8_t *qf = (uint8_t *)malloc(INTER);
    quant_row_f32(h_f32 + (size_t)r * INTER, INTER, qf, &sf, &zf);
    for (uint32_t c = 0; c < INTER; ++c) {
      const double ex = h_exact[(size_t)r * INTER + c];
      const uint8_t qi = ah[(size_t)(c / 32u) * 2048u + r * 32u + (c % 32u)];
      const uint8_t qi2 = ah2[(size_t)(c / 32u) * 2048u + r * 32u + (c % 32u)];
      const double xi = (double)sc[r] * ((int)qi - zq[r]);
      const double xi2 = (double)sc2[r] * ((int)qi2 - zq2[r]);
      const double xf = (double)sf * ((int)qf[c] - zf);
      e_int += (xi - ex) * (xi - ex);
      e_int2 += (xi2 - ex) * (xi2 - ex);
      e_f32 += (xf - ex) * (xf - ex);
      sig += ex * ex;
      /* disagreement with the f32 grid, in that grid's steps */
      const double d = fabs(xi - xf) / sf;
      if (d > 0.5)
        ++n1;
      if (d > 1.5)
        ++n2;
      const double d2 = fabs(xi2 - xi) / sc[r];
      if (d2 > 0.5)
        ++nb1;
      if (d2 > 1.5)
        ++nb2;
      ++tot;
    }
    free(qf);
  }
  const double snr_int = 10 * log10(sig / e_int);
  const double snr_int2 = 10 * log10(sig / e_int2);
  const double snr_f32 = 10 * log10(sig / e_f32);
  printf("SNR vs exact      : f32 path %.2f dB   int path %.2f dB   int "
         "path in 4 batches %.2f dB\n",
         snr_f32, snr_int, snr_int2);
  printf("int vs f32 grid   : >0.5 step %u (%.2f%%), >1.5 step %u of %u\n", n1,
         100.0 * n1 / tot, n2, tot);
  printf("1 batch vs 4      : >0.5 step %u (%.2f%%), >1.5 step %u\n", nb1,
         100.0 * nb1 / tot, nb2);
  if (snr_int < snr_f32 - 0.5 || snr_int2 < snr_f32 - 0.5 || n2 != 0 ||
      nb2 != 0)
    fail = 1;

  /* ---- 3. padding rows and an all-zero row -------------------------- */
  {
    for (uint32_t c = 0; c < N; ++c) {
      acc[(size_t)c / 32u * (M * 32) + (size_t)5 * 32 + (c % 32u)] =
        act_zp[5] * colsum[c]; /* A == 0 for row 5 */
    }
    job.m_count = 7;
    hvx_int_gu_worker(1, 0, &job);
    memset(ah, 0xA5, (size_t)M * INTER);
    hvx_int_rq_rows(hq, INTER, he, 8, 1, INTER, INTER, 7, 4, 8, sc, zq, ah);
    int ok = (sc[5] == 1.0f && zq[5] == 0 && sc[7] == 1.0f && zq[7] == 0);
    for (uint32_t c = 0; c < INTER && ok; ++c) {
      if (ah[(size_t)(c / 32u) * 2048u + 5 * 32u + (c % 32u)] != 0 ||
          ah[(size_t)(c / 32u) * 2048u + 7 * 32u + (c % 32u)] != 0)
        ok = 0;
    }
    printf("zero row / padding: %s\n", ok ? "scale 1, zp 0, bytes 0" : "WRONG");
    if (!ok)
      fail = 1;
  }

  printf(fail ? "\nFAIL\n" : "\nINT EPILOGUE OK\n");
  hvx_int_wq_free(wq);
  return fail;
}
