/* Host harness for hexkl_mm_u8i4_moe_layer_run: scalar stand-ins for every
   primitive it calls, then the kernel against a straightforward reference.
   The point is the loop structure -- which rows each expert gets, which
   weights it uses, whether the buffer reuse clobbers anything, whether the
   scatter lands on the right output row with the right routing weight --
   not HMX's arithmetic, which the stubs define self-consistently for both
   sides. */
#include "hexkl_acc_tile.h"
#include "hexkl_dma_ring.h"
#include "hexkl_mm_u8i4_moe.h"
#include "hexkl_probe.h"
#include "hvx_dequant_i32.h"
#include "hvx_gather_ah_u8.h"
#include "hvx_gemm_u8i4_wh.h"
#include "hvx_scale_add_f32.h"
#include <AEEStdErr.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hvx_scalar_stubs.h"

/* gelu_tanh in the form the kernel computes it: x * sigmoid(t),
   t = x (C0 + C1 x^2). The reference below uses it too, because the
   kernel's output is REQUANTIZED to u8 before the down matmul: the textbook
   0.5 x (1 + tanh y) differs from this by a few f32 ulps, which flips one
   u8 level on a couple of elements and reads as 9e-5 after down (the doc 44
   L2 lesson). The identity itself is checked separately in main, before
   any quantization, where ulps are what they are. */
static float gelu_sig(float g) {
  const float t = g * (1.5957691216f + 0.0713548163f * g * g);
  return g / (1.f + expf(-t));
}

/* The straightforward reference: per routed row, quantize, gate_up,
   act(gate)*up, requantize, down, weighted accumulate. */
static void ref_layer(int gelu, uint32_t M, uint32_t K, uint32_t inter,
                      uint32_t N_out, uint32_t NE, const W *wg, const W *wd,
                      const uint32_t *rc_, const uint32_t *ridx,
                      const float *rw, const float *act, float *want) {
  uint8_t *aq = (uint8_t *)malloc(K);
  uint8_t *mq = (uint8_t *)malloc(inter);
  float *gu = (float *)malloc(sizeof(float) * 2 * inter);
  float *dn = (float *)malloc(sizeof(float) * N_out);
  float *mid = (float *)malloc(sizeof(float) * inter);
  memset(want, 0, sizeof(float) * M * N_out);
  uint32_t base = 0;
  for (uint32_t e = 0; e < NE; ++e) {
    for (uint32_t i = 0; i < rc_[e]; ++i) {
      uint32_t row = ridx[base + i];
      float as;
      int32_t az;
      quant_row(act + (size_t)row * K, K, aq, &as, &az);
      ref_mm(&wg[e], aq, as, az, gu);
      for (uint32_t j = 0; j < inter; ++j) {
        const float g = gu[j];
        const float a = gelu ? gelu_sig(g) : g / (1.f + expf(-g));
        mid[j] = a * gu[inter + j];
      }
      float ms;
      int32_t mz;
      quant_row(mid, inter, mq, &ms, &mz);
      ref_mm(&wd[e], mq, ms, mz, dn);
      for (uint32_t c = 0; c < N_out; ++c) {
        /* Two operations through a volatile, matching what the kernel and
           the ARM path both do -- see hvx_scale_add_rows_f32's stub. */
        volatile float p = dn[c] * rw[base + i];
        want[(size_t)row * N_out + c] = want[(size_t)row * N_out + c] + p;
      }
    }
    base += rc_[e];
  }
  free(aq);
  free(mq);
  free(gu);
  free(dn);
  free(mid);
}

int main(void) {
  const uint32_t M = 37, K = 64, inter = 32, N_out = 64, NE = 5;
  static uint8_t vtcm[8u << 20];

  hexkl_moe_layout L;
  int rc = hexkl_mm_u8i4_moe_layout(K, inter, N_out, sizeof vtcm, &L);
  printf(
    "layout rc=%d total=%u (act %u gu %u dn %u gate %u mid %u stage %u res "
    "%u)\n",
    rc, L.total, L.act_off, L.w_gu_off, L.w_dn_off, L.gate_off, L.mid_off,
    L.result_off, L.res_f32_off);
  if (rc)
    return 1;

  W wg[8], wd[8];
  for (uint32_t e = 0; e < NE; ++e) {
    make_weight(e, K, 2 * inter, &wg[e]);
    make_weight(NE + e, inter, N_out, &wd[e]);
  }
  uint32_t hg[8], hd[8];
  for (uint32_t e = 0; e < NE; ++e) {
    hg[e] = e;
    hd[e] = NE + e;
  }

  float *act = (float *)malloc(sizeof(float) * M * K);
  for (uint32_t i = 0; i < M * K; ++i)
    act[i] = rndf();

  /* Routing: expert 0 gets many rows (multi-block, a 6-row tail for the
     HVX path), expert 2 gets none, expert 3 a tail of exactly the
     threshold, expert 4 a second block too big for it (stays on the HMX);
     rows repeat across experts the way top-k does. */
  uint32_t rc_[8] = {70, 12, 0, 64 + HVX_GEMM_U8I4_MAX_ROWS,
                     64 + HVX_GEMM_U8I4_MAX_ROWS + 10};
  uint32_t n_rows = 0;
  for (uint32_t e = 0; e < NE; ++e)
    n_rows += rc_[e];
  uint32_t *ridx = (uint32_t *)malloc(sizeof(uint32_t) * n_rows);
  float *rw = (float *)malloc(sizeof(float) * n_rows);
  for (uint32_t i = 0; i < n_rows; ++i) {
    ridx[i] = rnd() % M;
    rw[i] = 0.1f + 0.9f * ((float)(rnd() % 100u) / 100.f);
  }

  float *got = (float *)malloc(sizeof(float) * M * N_out);
  /* One scratch across every call below, the way the session holds it: the
     later, smaller calls must work out of the block the first one grew. */
  hexkl_moe_scratch scratch = {NULL, NULL, 0};
  rc = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K,
                                   inter, N_out, NE, hg, hd, ridx, rc_, rw, act,
                                   got, HVX_GLU_SILU, NULL, &scratch);
  printf("run rc=%d  n_rows=%u\n", rc, n_rows);
  if (rc)
    return 1;

  /* reference */
  float *want = (float *)calloc(M * N_out, sizeof(float));
  ref_layer(0, M, K, inter, N_out, NE, wg, wd, rc_, ridx, rw, act, want);

  double worst = 0.0;
  uint32_t bad = 0;
  for (uint32_t i = 0; i < M * N_out; ++i) {
    double d = fabs((double)got[i] - (double)want[i]);
    double s = fabs((double)want[i]) + 1e-6;
    if (d / s > 1e-5) {
      ++bad;
    }
    if (d / s > worst)
      worst = d / s;
  }
  printf("mismatches=%u of %u   worst_rel=%g\n", bad, M * N_out, worst);
  printf(bad == 0 ? "MOE KERNEL MATCHES REFERENCE\n" : "MOE KERNEL DIFFERS\n");
  int fail = (bad != 0);

  /* Every active expert's gate_up and down must have been pushed. The DMA
     stub here completes immediately, so a missing push cannot show up as a
     wrong result the way a premature one does -- this counter is the only
     thing that catches it, and it is also what the device profile divides
     by to get GB/s, so a wrong count would quietly misreport the bandwidth
     the whole plan is gated on. */
  {
    uint32_t active = 0;
    for (uint32_t e = 0; e < NE; ++e) {
      if (rc_[e] != 0u)
        ++active;
    }
    const uint32_t gu_kb = ((K / 32u) * ((2u * inter) / 32u) * 512u) >> 10;
    const uint32_t dn_kb = ((inter / 32u) * (N_out / 32u) * 512u) >> 10;
    const uint64_t want = (uint64_t)active * (gu_kb + dn_kb);
    const uint64_t got_kb = hexkl_probe_us[HEXKL_PROBE_DMA_KB];
    printf("weight DMA        : %llu KB over %u experts (want %llu)\n",
           (unsigned long long)got_kb, active, (unsigned long long)want);
    if (got_kb != want)
      fail = 1;
    /* HMX blocks only: 70 -> 1 + a 6-row tail on the HVX, 12 -> 1,
       80 -> 1 + a 16-row tail, 90 -> 2 (its 26-row second block is over
       the tail threshold). 6 would mean a tail went to the HMX after all,
       4 that a full block was skipped. */
    printf("HMX blocks        : %llu (want 5: two tails on the HVX)\n",
           (unsigned long long)hexkl_probe_us[HEXKL_PROBE_BLOCKS]);
    if (hexkl_probe_us[HEXKL_PROBE_BLOCKS] != 5u)
      fail = 1;
  }

  /* (After the probe counts above: the two calls below would add to them.) */
  /* The identity the GeGLU epilogue rests on, x sigmoid(2y) == 0.5 x (1 +
     tanh y), on a sweep of gate values in f32 and before any quantization:
     a wrong constant shows up here as 1e-3, not as a u8 flip. */
  {
    double worst_id = 0.0;
    for (int i = -2000; i <= 2000; ++i) {
      const float g = (float)i * 0.005f; /* [-10, 10] */
      /* In double: f32's 1 + tanh(y) cancels to a few ulps for y < -5. */
      const double gd = g;
      const double ref =
        0.5 * gd *
        (1.0 + tanh(0.7978845608028654 * (gd + 0.044715 * gd * gd * gd)));
      const double d = fabs((double)gelu_sig(g) - ref) / (fabs(ref) + 1e-6);
      if (d > worst_id)
        worst_id = d;
    }
    printf("gelu identity     : worst_rel=%g (want < 1e-5)\n", worst_id);
    fail |= (worst_id > 1e-5);
  }

  /* The same layer with the GeGLU epilogue (doc 55: Gemma-4's experts). */
  {
    float *got_g = (float *)malloc(sizeof(float) * M * N_out);
    float *want_g = (float *)calloc(M * N_out, sizeof(float));
    int r = hexkl_mm_u8i4_moe_layer_run(
      &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K, inter, N_out, NE, hg, hd,
      ridx, rc_, rw, act, got_g, HVX_GLU_GELU_TANH, NULL, &scratch);
    ref_layer(1, M, K, inter, N_out, NE, wg, wd, rc_, ridx, rw, act, want_g);
    uint32_t bad_g = 0;
    double worst_g = 0.0;
    for (uint32_t i = 0; i < M * N_out; ++i) {
      double d = fabs((double)got_g[i] - (double)want_g[i]);
      double sc = fabs((double)want_g[i]) + 1e-6;
      if (d / sc > 1e-5)
        ++bad_g;
      if (d / sc > worst_g)
        worst_g = d / sc;
    }
    printf("gelu epilogue     : rc=%d mismatches=%u worst_rel=%g\n", r, bad_g,
           worst_g);
    fail |= (r != 0 || bad_g != 0);
    r = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M,
                                    K, inter, N_out, NE, hg, hd, ridx, rc_, rw,
                                    act, got_g, 7u, NULL, &scratch);
    printf("act=7             : rc=%d (want %d)\n", r, AEE_EBADPARM);
    fail |= (r != AEE_EBADPARM);
    free(got_g);
    free(want_g);
  }

  /* Split (doc 52 section 10.14): experts {0,1,2} then {3,4}, summed on
     the host, against the whole call. Only the fp32 addition order may
     differ, so the error is taken against the output's largest magnitude:
     per element, a sum that cancels to near zero reads 3.5e-5 here. */
  {
    float *a = (float *)malloc(sizeof(float) * M * N_out);
    float *b = (float *)malloc(sizeof(float) * M * N_out);
    uint32_t n0 = rc_[0] + rc_[1] + rc_[2];
    int r = hexkl_mm_u8i4_moe_layer_run(
      &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K, inter, N_out, 3, hg, hd,
      ridx, rc_, rw, act, a, HVX_GLU_SILU, NULL, &scratch);
    r |= hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M,
                                     K, inter, N_out, 2, hg + 3, hd + 3,
                                     ridx + n0, rc_ + 3, rw + n0, act, b,
                                     HVX_GLU_SILU, NULL, &scratch);
    double w = 0.0, big = 0.0;
    for (uint32_t i = 0; i < M * N_out; ++i) {
      double d = fabs((double)(a[i] + b[i]) - (double)got[i]);
      if (d > w)
        w = d;
      if (fabs((double)got[i]) > big)
        big = fabs((double)got[i]);
    }
    w /= big;
    printf("split 3+2 vs whole: rc=%d worst_rel=%g\n", r, w);
    fail |= (r != 0 || w > 1e-5);
    free(a);
    free(b);
  }

  /* --- edge cases the routing can actually produce --------------------- */
  {
    uint32_t z[8] = {0, 0, 0, 0, 0};
    memset(got, 0xA5, sizeof(float) * M * N_out);
    int r = hexkl_mm_u8i4_moe_layer_run(
      &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K, inter, N_out, NE, hg, hd,
      ridx, z, rw, act, got, HVX_GLU_SILU, NULL, &scratch);
    int ok = (r == 0);
    for (uint32_t i = 0; i < M * N_out; ++i) {
      if (got[i] != 0.f) {
        ok = 0;
        break;
      }
    }
    printf("all-empty routing : %s\n", ok ? "zeroed, rc=0" : "WRONG");
    fail |= !ok;
  }
  {
    uint32_t c64[8] = {64, 0, 0, 0, 0};
    int r = hexkl_mm_u8i4_moe_layer_run(
      &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K, inter, N_out, NE, hg, hd,
      ridx, c64, rw, act, got, HVX_GLU_SILU, NULL, &scratch);
    printf("exactly 64 rows   : rc=%d\n", r);
    fail |= (r != 0);
  }
  {
    uint32_t c1[8] = {1, 0, 0, 0, 0};
    uint32_t bad_row[1] = {M};
    int r = hexkl_mm_u8i4_moe_layer_run(
      &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K, inter, N_out, NE, hg, hd,
      bad_row, c1, rw, act, got, HVX_GLU_SILU, NULL, &scratch);
    printf("row_index >= M    : rc=%d (want %d)\n", r, AEE_EBADPARM);
    fail |= (r != AEE_EBADPARM);
  }
  {
    hexkl_moe_layout R;
    int r = hexkl_mm_u8i4_moe_layout(2048, 1792, 2048, 8300u * 1024u, &R);
    printf("LFM2 shapes       : rc=%d total=%.2f MB (doc 47 section 11 says "
           "6.92)\n",
           r, R.total / 1048576.0);
    fail |= (r != 0);
    r = hexkl_mm_u8i4_moe_layout(2048, 1792, 2048, 4u << 20, &R);
    printf("4 MB arena        : rc=%d (want %d)\n", r, AEE_ENOMEMORY);
    fail |= (r != AEE_ENOMEMORY);
  }
  printf(fail ? "\nFAIL\n" : "\nALL CHECKS PASS\n");
  hexkl_moe_scratch_free(&scratch);
  return fail;
}
