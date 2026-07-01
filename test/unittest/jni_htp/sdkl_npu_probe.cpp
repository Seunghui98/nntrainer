// Minimal sdkl_npu_mm_f32f16_f32 probe — three controlled cases:
//   1. W = identity  → result must equal X (within FP tolerance)
//   2. W = zero      → result must be zero
//   3. Deterministic random W → NPU vs naive CPU reference
// Build via ndk-build, push + run on device.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <remote.h>
#include <sdkl.h>

static const int M = 32, N = 32, K = 32;

static float relErr(const float *a, const float *b, int n) {
  float num = 0.f, den = 0.f;
  for (int i = 0; i < n; ++i) {
    num = fmaxf(num, fabsf(a[i] - b[i]));
    den = fmaxf(den, fabsf(b[i]));
  }
  return num / (den + 1e-6f);
}

static float maxAbs(const float *a, int n) {
  float mx = 0.f;
  for (int i = 0; i < n; ++i) mx = fmaxf(mx, fabsf(a[i]));
  return mx;
}

// CPU reference: C[M,N] = X[M,K] * W[N,K]^T, float32 accumulate
static void cpuGemm(const float *X, const float *W, float *C) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float s = 0.f;
      for (int k = 0; k < K; ++k) s += X[i*K+k] * W[j*K+k];
      C[i*N+j] = s;
    }
}

int main() {
  int domain = CDSP_DOMAIN_ID;  // = 3
  int err = sdkl_npu_initialize(domain, nullptr, nullptr);
  if (err != 0) {
    fprintf(stderr, "FAIL: sdkl_npu_initialize err=%d\n", err);
    return 1;
  }
  char ver[128] = {};
  sdkl_npu_get_version(domain, ver);
  fprintf(stderr, "NPU ready: domain=%d version=%s\n", domain, ver);

  void *Xb = nullptr, *Wb = nullptr, *Ab = nullptr;
  if (sdkl_npu_alloc(M*K*sizeof(float),   &Xb) ||
      sdkl_npu_alloc(N*K*sizeof(_Float16), &Wb) ||
      sdkl_npu_alloc(M*N*sizeof(float),   &Ab)) {
    fprintf(stderr, "FAIL: sdkl_npu_alloc\n");
    return 1;
  }

  int failures = 0;

  // ----------------------------------------------------------------
  // Case 1: W = identity  →  X * I^T = X
  // ----------------------------------------------------------------
  {
    std::vector<float> X(M*K), W_f32(N*K, 0.f), C_npu(M*N), C_cpu(M*N);
    for (int i = 0; i < M*K; ++i) X[i] = ((i % 17) + 1) * 0.1f;
    for (int n = 0; n < N && n < K; ++n) W_f32[n*K+n] = 1.f;  // diag=1

    std::vector<_Float16> W_f16(N*K);
    for (int i = 0; i < N*K; ++i) W_f16[i] = (_Float16)W_f32[i];

    memcpy(Xb, X.data(),    M*K*sizeof(float));
    memcpy(Wb, W_f16.data(), N*K*sizeof(_Float16));
    err = sdkl_cpu_rm_to_wh_f16_inplace((size_t)N, (size_t)K,
                                        static_cast<_Float16*>(Wb));
    if (err) { fprintf(stderr, "FAIL case1: rm_to_wh err=%d\n", err); ++failures; goto case2; }

    memset(Ab, 0, M*N*sizeof(float));
    err = sdkl_npu_mm_f32f16_f32(domain, M, N, K,
                                  static_cast<float*>(Ab),
                                  static_cast<const float*>(Xb),
                                  static_cast<const _Float16*>(Wb));
    if (err) { fprintf(stderr, "FAIL case1: npu_mm err=%d\n", err); ++failures; goto case2; }

    memcpy(C_npu.data(), Ab, M*N*sizeof(float));
    cpuGemm(X.data(), W_f32.data(), C_cpu.data());

    {
      float re = relErr(C_npu.data(), C_cpu.data(), M*N);
      if (re < 0.01f) fprintf(stderr, "PASS case1 W=I:    rel_err=%.5f\n", re);
      else {
        fprintf(stderr, "FAIL case1 W=I:    rel_err=%.5f\n", re);
        for (int i = 0; i < 8; ++i)
          fprintf(stderr, "  [%d] npu=%.5f cpu=%.5f\n", i, C_npu[i], C_cpu[i]);
        ++failures;
      }
    }
  }

case2:
  // ----------------------------------------------------------------
  // Case 2: W = zero  →  all outputs must be ~0
  // ----------------------------------------------------------------
  {
    std::vector<float> X(M*K);
    for (int i = 0; i < M*K; ++i) X[i] = (float)((i%7)+1);
    std::vector<_Float16> W_f16(N*K, (_Float16)0.f);
    std::vector<float> C_npu(M*N);

    memcpy(Xb, X.data(),    M*K*sizeof(float));
    memcpy(Wb, W_f16.data(), N*K*sizeof(_Float16));
    sdkl_cpu_rm_to_wh_f16_inplace((size_t)N, (size_t)K, static_cast<_Float16*>(Wb));
    memset(Ab, 0, M*N*sizeof(float));
    err = sdkl_npu_mm_f32f16_f32(domain, M, N, K,
                                  static_cast<float*>(Ab),
                                  static_cast<const float*>(Xb),
                                  static_cast<const _Float16*>(Wb));
    if (err) { fprintf(stderr, "FAIL case2: npu_mm err=%d\n", err); ++failures; goto case3; }
    memcpy(C_npu.data(), Ab, M*N*sizeof(float));

    {
      float mx = maxAbs(C_npu.data(), M*N);
      if (mx < 0.001f) fprintf(stderr, "PASS case2 W=0:    max_abs=%.5f\n", mx);
      else {
        fprintf(stderr, "FAIL case2 W=0:    max_abs=%.5f (expected ~0)\n", mx);
        for (int i = 0; i < 8; ++i)
          fprintf(stderr, "  [%d] npu=%.5f\n", i, C_npu[i]);
        ++failures;
      }
    }
  }

case3:
  // ----------------------------------------------------------------
  // Case 3: Deterministic random W, NPU vs CPU reference
  // ----------------------------------------------------------------
  {
    std::vector<float> X(M*K), W_f32(N*K), C_npu(M*N), C_cpu(M*N);
    for (int i = 0; i < M*K; ++i) X[i]     = ((i*7+3) % 100) / 100.f - 0.5f;
    for (int i = 0; i < N*K; ++i) W_f32[i] = ((i*13+5) % 100) / 200.f - 0.25f;
    std::vector<_Float16> W_f16(N*K);
    for (int i = 0; i < N*K; ++i) W_f16[i] = (_Float16)W_f32[i];

    memcpy(Xb, X.data(),    M*K*sizeof(float));
    memcpy(Wb, W_f16.data(), N*K*sizeof(_Float16));
    sdkl_cpu_rm_to_wh_f16_inplace((size_t)N, (size_t)K, static_cast<_Float16*>(Wb));
    memset(Ab, 0, M*N*sizeof(float));
    err = sdkl_npu_mm_f32f16_f32(domain, M, N, K,
                                  static_cast<float*>(Ab),
                                  static_cast<const float*>(Xb),
                                  static_cast<const _Float16*>(Wb));
    if (err) { fprintf(stderr, "FAIL case3: npu_mm err=%d\n", err); ++failures; goto done; }
    memcpy(C_npu.data(), Ab, M*N*sizeof(float));
    cpuGemm(X.data(), W_f32.data(), C_cpu.data());

    {
      float re = relErr(C_npu.data(), C_cpu.data(), M*N);
      if (re < 0.01f) fprintf(stderr, "PASS case3 rand:   rel_err=%.5f\n", re);
      else {
        fprintf(stderr, "FAIL case3 rand:   rel_err=%.5f\n", re);
        for (int i = 0; i < 8; ++i)
          fprintf(stderr, "  [%d] npu=%.5f cpu=%.5f\n", i, C_npu[i], C_cpu[i]);
        ++failures;
      }
    }
  }

done:
  sdkl_npu_free(Xb);
  sdkl_npu_free(Wb);
  sdkl_npu_free(Ab);
  sdkl_npu_finalize(domain);
  fprintf(stderr, failures ? "=== FAIL: %d case(s) failed ===\n"
                           : "=== PASS: all cases OK ===\n", failures);
  return failures > 0 ? 1 : 0;
}
