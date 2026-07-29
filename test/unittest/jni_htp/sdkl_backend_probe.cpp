// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   sdkl_backend_probe.cpp
 * @brief  Minimal probe to test sdkl HTP backend functionality
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs
 */

// Minimal probe: call nntrainer::hmx::shgemm_f32f16_f32 via libnntrainer.so
// (no gtest, no HtpShgemmTest fixture) and compare against a CPU reference.
// If this passes but unittest_nntrainer_htp_backend fails, the bug is in the
// test harness, not the underlying NPU computation.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cpu_backend.h> // nntrainer::shgemm
#include <fp16.h>        // nntrainer::compute_fp32_to_fp16
#include <hexkl_mm.h>    // nntrainer::hmx::shgemm_f32f16_f32

static const int M = 32, N = 32, K = 32;

static float relErr(const float *npu, const float *cpu, int n) {
  float num = 0.f, den = 0.f;
  for (int i = 0; i < n; ++i) {
    num = fmaxf(num, fabsf(npu[i] - cpu[i]));
    den = fmaxf(den, fabsf(cpu[i]));
  }
  return num / (den + 1e-6f);
}

// CPU reference: C[M,N] = X[M,K] * W[N,K]^T  (TransB=true, row-major)
static void cpuGemm(const float *X, const float *W_f32, float *C) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float s = 0.f;
      for (int k = 0; k < K; ++k)
        s += X[i * K + k] * W_f32[j * K + k];
      C[i * N + j] = s;
    }
}

int main() {
  int failures = 0;

  // ---- Case 1: W = identity ------------------------------------------------
  {
    std::vector<float> X(M * K), W_f32(N * K, 0.f), C_npu(M * N), C_cpu(M * N);
    std::vector<uint16_t> W_fp16(N * K, 0);

    for (int i = 0; i < M * K; ++i)
      X[i] = ((i % 17) + 1) * 0.1f;
    for (int n = 0; n < N && n < K; ++n) {
      W_f32[n * K + n] = 1.f;
      W_fp16[n * K + n] = nntrainer::compute_fp32_to_fp16(1.f);
    }
    cpuGemm(X.data(), W_f32.data(), C_cpu.data());

    nntrainer::hmx::shgemm_f32f16_f32(
      1 /** ROW_MAJOR */, false /** TransA */, true /** TransB */, M, N, K, 1.0f,
      X.data(), K, reinterpret_cast<const _FP16 *>(W_fp16.data()), K, 0.0f,
      C_npu.data(), N);

    float re = relErr(C_npu.data(), C_cpu.data(), M * N);
    if (re < 0.01f) {
      fprintf(stderr, "PASS case1 W=I:    rel_err=%.5f\n", re);
    } else {
      fprintf(stderr, "FAIL case1 W=I:    rel_err=%.5f\n", re);
      for (int i = 0; i < 8; ++i)
        fprintf(stderr, "  [%d] npu=%.5f cpu=%.5f\n", i, C_npu[i], C_cpu[i]);
      ++failures;
    }
  }

  // ---- Case 2: W = zero ----------------------------------------------------
  {
    std::vector<float> X(M * K), C_npu(M * N, 0.f);
    std::vector<uint16_t> W_fp16(N * K, 0);

    for (int i = 0; i < M * K; ++i)
      X[i] = (float)((i % 7) + 1);

    nntrainer::hmx::shgemm_f32f16_f32(
      1, false, true, M, N, K, 1.0f, X.data(), K,
      reinterpret_cast<const _FP16 *>(W_fp16.data()), K, 0.0f, C_npu.data(), N);

    float mx = 0.f;
    for (int i = 0; i < M * N; ++i)
      mx = fmaxf(mx, fabsf(C_npu[i]));
    if (mx < 0.001f) {
      fprintf(stderr, "PASS case2 W=0:    max_abs=%.5f\n", mx);
    } else {
      fprintf(stderr, "FAIL case2 W=0:    max_abs=%.5f\n", mx);
      ++failures;
    }
  }

  // ---- Case 3: deterministic random ----------------------------------------
  {
    std::vector<float> X(M * K), W_f32(N * K), C_npu(M * N), C_cpu(M * N);
    std::vector<uint16_t> W_fp16(N * K);

    for (int i = 0; i < M * K; ++i)
      X[i] = ((i * 7 + 3) % 100) / 100.f - 0.5f;
    for (int i = 0; i < N * K; ++i)
      W_f32[i] = ((i * 13 + 5) % 100) / 200.f - 0.25f;
    for (int i = 0; i < N * K; ++i)
      W_fp16[i] = nntrainer::compute_fp32_to_fp16(W_f32[i]);
    cpuGemm(X.data(), W_f32.data(), C_cpu.data());

    nntrainer::hmx::shgemm_f32f16_f32(
      1, false, true, M, N, K, 1.0f, X.data(), K,
      reinterpret_cast<const _FP16 *>(W_fp16.data()), K, 0.0f, C_npu.data(), N);

    float re = relErr(C_npu.data(), C_cpu.data(), M * N);
    if (re < 0.01f) {
      fprintf(stderr, "PASS case3 rand:   rel_err=%.5f\n", re);
    } else {
      fprintf(stderr, "FAIL case3 rand:   rel_err=%.5f\n", re);
      for (int i = 0; i < 8; ++i)
        fprintf(stderr, "  [%d] npu=%.5f cpu=%.5f\n", i, C_npu[i], C_cpu[i]);
      ++failures;
    }
  }

  // ---- Case 4: nntrainer::shgemm vs shgemm_f32f16_f32 (same inputs) --------
  // Tests whether nntrainer::shgemm (CPU HGEMM) agrees with the NPU kernel.
  // This mirrors AccuracyVsCpu in unittest_nntrainer_htp_backend.cpp:
  // if they diverge here, the CPU reference is wrong, not the NPU.
  {
    std::vector<float> X(M * K), C_npu(M * N, 0.f), C_shgemm(M * N, 0.f);
    std::vector<uint16_t> W_fp16(N * K);

    // Random inputs (same deterministic pattern as case3 for reproducibility)
    for (int i = 0; i < M * K; ++i)
      X[i] = ((i * 7 + 3) % 100) / 100.f - 0.5f;
    std::vector<float> W_f32_2(N * K);
    for (int i = 0; i < N * K; ++i)
      W_f32_2[i] = ((i * 13 + 5) % 100) / 200.f - 0.25f;
    for (int i = 0; i < N * K; ++i)
      W_fp16[i] = nntrainer::compute_fp32_to_fp16(W_f32_2[i]);

    // NPU path (via libnntrainer.so)
    nntrainer::hmx::shgemm_f32f16_f32(
      1, false, true, M, N, K, 1.0f, X.data(), K,
      reinterpret_cast<const _FP16 *>(W_fp16.data()), K, 0.0f, C_npu.data(), N);

    // CPU HGEMM path (nntrainer::shgemm) — exactly as cpuShgemm in the backend
    // test
    nntrainer::shgemm(1 /** ROW_MAJOR */, false /** TransA */, true /** TransB */, M, N,
                      K, 1.0f, X.data(), K,
                      reinterpret_cast<const _FP16 *>(W_fp16.data()), K, 0.0f,
                      C_shgemm.data(), N);

    float re = relErr(C_npu.data(), C_shgemm.data(), M * N);
    if (re < 0.01f) {
      fprintf(stderr, "PASS case4 npu==shgemm: rel_err=%.5f\n", re);
    } else {
      fprintf(
        stderr,
        "FAIL case4 npu!=shgemm: rel_err=%.5f  ← CPU ref or NPU mismatch\n",
        re);
      for (int i = 0; i < 8; ++i)
        fprintf(stderr, "  [%d] npu=%.5f shgemm=%.5f\n", i, C_npu[i],
                C_shgemm[i]);
      ++failures;
    }
  }

  fprintf(stderr,
          failures ? "=== FAIL: %d case(s) failed ===\n"
                   : "=== PASS: all cases OK ===\n",
          failures);
  return failures > 0 ? 1 : 0;
}
