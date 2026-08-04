// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   mha_cpu_bench.cpp
 * @brief  Standalone ARM CPU baseline for the two MHA-core GEMMs, matching
 *         the DSP numbers in hexkl_micro_mha_bench.c shape-for-shape. Does
 *         NOT modify Applications/CausalLM/layers/mha_core.cpp -- links
 *         directly against an already-built libnntrainer.so and calls the
 *         exact same nntrainer::compute_kcaches / compute_fp16vcache_transposed
 *         symbols that file's fp16 path calls, for both branches:
 *
 *           decode  (sequence_len==1 / (to-from)==1): one call per kv_head
 *           prefill (sequence_len!=1 / (to-from)!=1): one call per query row
 *
 * kv_len=1024 (and 512 for the extra decode P.V point), nch=8, gqa=8,
 * head_dim=128, tile_size=4 -- matching every shape in the DSP bench.
 *
 * _FP16 in the prebuilt is __fp16 (built with -DUSE__FP16=1) -- this must
 * match exactly, since __fp16 vs _Float16 can mangle differently and link
 * against the wrong (or no) symbol.
 *
 * Build/run: see run_mha_cpu_bench.sh (NDK cross-compile, adb push, adb
 * shell run -- no gtest, no meson, no full nntrainer rebuild required).
 */
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

namespace nntrainer {
extern void compute_kcaches(const __fp16 *in, const __fp16 *kcache,
                            __fp16 *output, int num_rows, int num_cache_head,
                            int head_dim, int gqa_size, int tile_size,
                            size_t local_window_size, int head_start,
                            int head_end);
extern void compute_fp16vcache_transposed(int row_num, const __fp16 *in,
                                          const __fp16 *vcache, __fp16 *output,
                                          int num_cache_head, int gqa_size,
                                          int head_dim,
                                          size_t local_window_size,
                                          int head_start, int head_end);
}

static double now_us(std::chrono::high_resolution_clock::time_point t0,
                     std::chrono::high_resolution_clock::time_point t1) {
  return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
           .count() /
         1000.0;
}

static const int NUM_CACHE_HEAD = 8;
static const int GQA_SIZE = 8;
static const int HEAD_DIM = 128;
static const int TILE_SIZE = 4;
static const size_t LOCAL_WINDOW = ((size_t)-1) >> 1; // effectively no window
static const int NTHREADS = 8;

/* =====================================================================
 * DECODE: compute_kcaches (Q.K^T), one call per kv_head, sequence_len==1.
 * ===================================================================== */
static void bench_kcaches_decode(int kv_len) {
  const size_t in_size = (size_t)NUM_CACHE_HEAD * GQA_SIZE * HEAD_DIM;
  const size_t kcache_size = (size_t)kv_len * NUM_CACHE_HEAD * HEAD_DIM;
  const size_t out_size = (size_t)kv_len * NUM_CACHE_HEAD * GQA_SIZE;

  std::vector<__fp16> in(in_size), kcache(kcache_size), out(out_size);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (auto &v : in)
    v = (__fp16)dist(gen);
  for (auto &v : kcache)
    v = (__fp16)dist(gen);

  printf("[mha_cpu_bench] DECODE compute_kcaches kv_len=%d nch=%d gqa=%d hd=%d\n",
         kv_len, NUM_CACHE_HEAD, GQA_SIZE, HEAD_DIM);

  const int ITERS = 200;
  // (1) ONE kv_head, single thread
  {
    nntrainer::compute_kcaches(in.data(), kcache.data(), out.data(), kv_len,
                               NUM_CACHE_HEAD, HEAD_DIM, GQA_SIZE, TILE_SIZE,
                               LOCAL_WINDOW, 0, 1); // warmup
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; it++)
      nntrainer::compute_kcaches(in.data(), kcache.data(), out.data(), kv_len,
                                 NUM_CACHE_HEAD, HEAD_DIM, GQA_SIZE, TILE_SIZE,
                                 LOCAL_WINDOW, 0, 1);
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = now_us(t0, t1) / ITERS;
    printf("  (1) ONE kv_head, single thread:      %8.2f us/call\n", us);
    printf("KCACHE_DEC_1HEAD_US_kv%d %.2f\n", kv_len, us);
  }
  // (2) ALL kv_heads, single thread sequential
  {
    nntrainer::compute_kcaches(in.data(), kcache.data(), out.data(), kv_len,
                               NUM_CACHE_HEAD, HEAD_DIM, GQA_SIZE, TILE_SIZE,
                               LOCAL_WINDOW, 0, NUM_CACHE_HEAD);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; it++)
      nntrainer::compute_kcaches(in.data(), kcache.data(), out.data(), kv_len,
                                 NUM_CACHE_HEAD, HEAD_DIM, GQA_SIZE, TILE_SIZE,
                                 LOCAL_WINDOW, 0, NUM_CACHE_HEAD);
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = now_us(t0, t1) / ITERS;
    printf("  (2) ALL %d kv_heads, single thread:   %8.2f us/call\n", NUM_CACHE_HEAD, us);
    printf("KCACHE_DEC_8HEAD_SEQ_US_kv%d %.2f\n", kv_len, us);
  }
  // (3) ALL kv_heads, 8 threads pooled (spawned once, ITERS internal repeats
  // each -- thread create/destroy amortized, not paid every timed call; see
  // the M2-n note at the bottom of this file for why this matters).
  {
    std::vector<std::thread> threads;
    threads.reserve(NUM_CACHE_HEAD);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int h = 0; h < NUM_CACHE_HEAD; h++) {
      threads.emplace_back([&, h]() {
        for (int it = 0; it < ITERS; it++)
          nntrainer::compute_kcaches(in.data(), kcache.data(), out.data(),
                                     kv_len, NUM_CACHE_HEAD, HEAD_DIM,
                                     GQA_SIZE, TILE_SIZE, LOCAL_WINDOW, h,
                                     h + 1);
      });
    }
    for (auto &th : threads)
      th.join();
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = now_us(t0, t1) / ITERS;
    printf("  (3) ALL %d kv_heads, %d threads (pooled): %8.2f us/call\n",
           NUM_CACHE_HEAD, NUM_CACHE_HEAD, us);
    printf("KCACHE_DEC_8HEAD_PAR_US_kv%d %.2f\n", kv_len, us);
  }
}

/* =====================================================================
 * DECODE: compute_fp16vcache_transposed (P.V), one call per kv_head,
 * (to-from)==1.
 * ===================================================================== */
static void bench_vcache_decode(int kv_len) {
  const int num_cache_head = NUM_CACHE_HEAD, gqa_size = GQA_SIZE, head_dim = HEAD_DIM;
  const int row_num = kv_len - 1;
  /* IMPORTANT: compute_fp16vcache_transposed's internal loop is
   *   for (row = start_row; row <= row_num; ++row) {
   *     attn_row = row - start_row;
   *     a_val = in[attn_row*(nch*gqa) + n*gqa + h];
   *     ...
   *   }
   * i.e. it reads WINDOW rows of `in` per call (window = min(row_num+1,
   * local_window_size) = kv_len here), not one. Sizing `in` at just
   * nch*gqa (one row) is a real out-of-bounds read, not a simplification --
   * an earlier version of this file had exactly that bug and it produced
   * numbers that did not reproduce run-to-run for reasons that looked like
   * ordinary DVFS noise but were actually undefined behavior. */
  const size_t in_size = (size_t)kv_len * num_cache_head * gqa_size;
  const size_t vcache_size = (size_t)kv_len * num_cache_head * head_dim;
  const size_t out_size = (size_t)num_cache_head * gqa_size * head_dim;

  std::vector<__fp16> in(in_size), vcache(vcache_size), out(out_size);
  std::mt19937 gen(44);
  // softmax-like: non-negative, each (head,g) column sums to ~1 over kv_len
  for (int n = 0; n < num_cache_head; n++) {
    for (int g = 0; g < gqa_size; g++) {
      std::vector<float> raw(kv_len);
      std::uniform_real_distribution<float> dist(0.f, 1.f);
      float sum = 0.f;
      for (auto &v : raw) {
        v = dist(gen);
        sum += v;
      }
      for (int row = 0; row < kv_len; row++)
        in[(size_t)row * num_cache_head * gqa_size + n * gqa_size + g] =
          (__fp16)(raw[row] / sum);
    }
  }
  std::uniform_real_distribution<float> vdist(-1.f, 1.f);
  for (auto &v : vcache)
    v = (__fp16)vdist(gen);

  printf("[mha_cpu_bench] DECODE compute_fp16vcache_transposed kv_len=%d nch=%d gqa=%d hd=%d\n",
         kv_len, num_cache_head, gqa_size, head_dim);

  const int ITERS = 200;
  {
    nntrainer::compute_fp16vcache_transposed(row_num, in.data(), vcache.data(),
                                             out.data(), num_cache_head, gqa_size,
                                             head_dim, LOCAL_WINDOW, 0, 1);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; it++)
      nntrainer::compute_fp16vcache_transposed(row_num, in.data(), vcache.data(),
                                               out.data(), num_cache_head, gqa_size,
                                               head_dim, LOCAL_WINDOW, 0, 1);
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = now_us(t0, t1) / ITERS;
    printf("  (1) ONE kv_head, single thread:      %8.2f us/call\n", us);
    printf("VCACHE_DEC_1HEAD_US_kv%d %.2f\n", kv_len, us);
  }
  {
    nntrainer::compute_fp16vcache_transposed(row_num, in.data(), vcache.data(),
                                             out.data(), num_cache_head, gqa_size,
                                             head_dim, LOCAL_WINDOW, 0, num_cache_head);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; it++)
      nntrainer::compute_fp16vcache_transposed(row_num, in.data(), vcache.data(),
                                               out.data(), num_cache_head, gqa_size,
                                               head_dim, LOCAL_WINDOW, 0, num_cache_head);
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = now_us(t0, t1) / ITERS;
    printf("  (2) ALL %d kv_heads, single thread:   %8.2f us/call\n", num_cache_head, us);
    printf("VCACHE_DEC_8HEAD_SEQ_US_kv%d %.2f\n", kv_len, us);
  }
  {
    std::vector<std::thread> threads;
    threads.reserve(num_cache_head);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int h = 0; h < num_cache_head; h++) {
      threads.emplace_back([&, h]() {
        for (int it = 0; it < ITERS; it++)
          nntrainer::compute_fp16vcache_transposed(row_num, in.data(), vcache.data(),
                                                   out.data(), num_cache_head,
                                                   gqa_size, head_dim, LOCAL_WINDOW,
                                                   h, h + 1);
      });
    }
    for (auto &th : threads)
      th.join();
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = now_us(t0, t1) / ITERS;
    printf("  (3) ALL %d kv_heads, %d threads (pooled): %8.2f us/call\n",
           num_cache_head, num_cache_head, us);
    printf("VCACHE_DEC_8HEAD_PAR_US_kv%d %.2f\n", kv_len, us);
  }
}

/* =====================================================================
 * PREFILL: compute_kcaches (Q.K^T), one call PER QUERY ROW (128-token
 * chunk), sequence_len != 1. row_to_compute fixed to kv_len-1 = 1023 for
 * every row (the "128-token chunk against a 1024-entry cache" worst-case-
 * per-row number) -- a defensible, simple overestimate of the causal
 * average.
 * ===================================================================== */
static void bench_qk_prefill() {
  const int SEQ_LEN = 128, KV_LEN = 1024;
  const int num_head = NUM_CACHE_HEAD * GQA_SIZE;
  const size_t in_size = (size_t)SEQ_LEN * num_head * HEAD_DIM;
  const size_t kcache_size = (size_t)KV_LEN * NUM_CACHE_HEAD * HEAD_DIM;
  const size_t out_size = (size_t)SEQ_LEN * NUM_CACHE_HEAD * GQA_SIZE;

  std::vector<__fp16> in(in_size), kcache(kcache_size), out(out_size);
  std::mt19937 gen(45);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (auto &v : in)
    v = (__fp16)dist(gen);
  for (auto &v : kcache)
    v = (__fp16)dist(gen);

  auto run_seq_once = [&]() {
    for (int i = 0; i < SEQ_LEN; i++) {
      const __fp16 *input_addr = in.data() + (size_t)num_head * HEAD_DIM * i;
      nntrainer::compute_kcaches(input_addr, kcache.data(), out.data(), KV_LEN,
                                 NUM_CACHE_HEAD, HEAD_DIM, GQA_SIZE, TILE_SIZE,
                                 LOCAL_WINDOW, 0, NUM_CACHE_HEAD);
    }
  };
  run_seq_once();
  const int ITERS = 20;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int it = 0; it < ITERS; it++)
    run_seq_once();
  auto t1 = std::chrono::high_resolution_clock::now();
  double us_seq = now_us(t0, t1) / ITERS;
  printf("[mha_cpu_bench] PREFILL qk_pre_128 (compute_kcaches, seq_len=%d, "
         "kv_len=%d): single-thread %8.2f us/chunk\n",
         SEQ_LEN, KV_LEN, us_seq);
  printf("PREFILL_QK_CPU_SEQ_US %.2f\n", us_seq);

  // M2-n: threads spawned ONCE, each running all ITERS internal repeats of
  // its row slice, joined once -- thread create/destroy amortized across 20
  // iterations instead of paid every one (the original per-iteration-spawn
  // version made prefill -- which is compute-bound, unlike decode -- look
  // like it only scaled ~2.2x across 8 threads; that number was measuring
  // spawn/join overhead, not a real CPU ceiling).
  //
  // IMPORTANT, read before trusting this number as settled: with the fix
  // applied, 8-thread scaling on THIS shape measured only ~1.6-2.1x across
  // 5 runs (sometimes *worse* than the flawed per-iteration-spawn number),
  // not the ~8x a purely compute-bound kernel should show. The likely cause
  // is memory-bandwidth contention across 8 threads concurrently reading
  // the K-cache, or thermal throttling under sustained multi-core load --
  // but this is NOT pinned down (would need perf counters, out of scope
  // here). Do not read the number below as a settled CPU ceiling; it is the
  // best available measurement, reported deliberately rather than assumed.
  const int rows_per_thread = SEQ_LEN / NTHREADS;
  {
    std::vector<std::thread> threads;
    threads.reserve(NTHREADS);
    auto t0p = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < NTHREADS; t++) {
      threads.emplace_back([&, t]() {
        int lo = t * rows_per_thread;
        int hi = (t == NTHREADS - 1) ? SEQ_LEN : lo + rows_per_thread;
        for (int it = 0; it < ITERS; it++) {
          for (int i = lo; i < hi; i++) {
            const __fp16 *input_addr = in.data() + (size_t)num_head * HEAD_DIM * i;
            nntrainer::compute_kcaches(input_addr, kcache.data(), out.data(),
                                       KV_LEN, NUM_CACHE_HEAD, HEAD_DIM,
                                       GQA_SIZE, TILE_SIZE, LOCAL_WINDOW, 0,
                                       NUM_CACHE_HEAD);
          }
        }
      });
    }
    for (auto &th : threads)
      th.join();
    auto t1p = std::chrono::high_resolution_clock::now();
    double us_par = now_us(t0p, t1p) / ITERS;
    printf("[mha_cpu_bench] PREFILL qk_pre_128: %d-thread (pooled, spawn amortized, "
           "scaling NOT fully explained -- see comment above) %8.2f us/chunk\n",
           NTHREADS, us_par);
    printf("PREFILL_QK_CPU_PAR_US %.2f\n", us_par);
  }
}

/* =====================================================================
 * PREFILL: compute_fp16vcache_transposed (P.V), one call per query row.
 * ===================================================================== */
static void bench_pv_prefill() {
  const int SEQ_LEN = 128, KV_LEN = 1024;
  /* Same bug class as bench_vcache_decode: every one of the 128 rows uses
   * row_num=KV_LEN-1 (this file's "against a 1024-entry cache" simplification
   * -- see the file-level comment), so EVERY call's window is the full
   * KV_LEN, and `in` must have KV_LEN*nch*gqa elements available from
   * whatever pointer is passed in, not SEQ_LEN*nch*gqa total shared across
   * all 128 calls. Since every row shares the same (worst-case) window
   * size in this simplification, all 128 calls reuse the SAME full-window
   * buffer (no per-row offset) rather than provisioning 128 separate
   * windows (which would need SEQ_LEN*KV_LEN*nch*gqa elements, ~16.8MB --
   * correct but wasteful for no benchmarking benefit here). Only the
   * OUTPUT differs per row. */
  const size_t in_size = (size_t)KV_LEN * NUM_CACHE_HEAD * GQA_SIZE;
  const size_t vcache_size = (size_t)KV_LEN * NUM_CACHE_HEAD * HEAD_DIM;
  const size_t out_size = (size_t)SEQ_LEN * NUM_CACHE_HEAD * GQA_SIZE * HEAD_DIM;
  const int row_num = KV_LEN - 1;

  std::vector<__fp16> in(in_size), vcache(vcache_size), out(out_size);
  std::mt19937 gen(46);
  std::uniform_real_distribution<float> dist(0.f, 0.1f);
  for (auto &v : in)
    v = (__fp16)dist(gen);
  std::uniform_real_distribution<float> vdist(-1.f, 1.f);
  for (auto &v : vcache)
    v = (__fp16)vdist(gen);

  auto run_seq_once = [&]() {
    for (int i = 0; i < SEQ_LEN; i++) {
      __fp16 *o = out.data() + (size_t)i * (NUM_CACHE_HEAD * GQA_SIZE * HEAD_DIM);
      nntrainer::compute_fp16vcache_transposed(row_num, in.data(), vcache.data(), o,
                                               NUM_CACHE_HEAD, GQA_SIZE, HEAD_DIM,
                                               LOCAL_WINDOW, 0, NUM_CACHE_HEAD);
    }
  };
  run_seq_once();
  const int ITERS = 20;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int it = 0; it < ITERS; it++)
    run_seq_once();
  auto t1 = std::chrono::high_resolution_clock::now();
  double us_seq = now_us(t0, t1) / ITERS;
  printf("[mha_cpu_bench] PREFILL pv_pre_128 (compute_fp16vcache_transposed, "
         "seq_len=%d, kv_len=%d): single-thread %8.2f us/chunk\n",
         SEQ_LEN, KV_LEN, us_seq);
  printf("PREFILL_PV_CPU_SEQ_US %.2f\n", us_seq);

  const int rows_per_thread = SEQ_LEN / NTHREADS;
  {
    std::vector<std::thread> threads;
    threads.reserve(NTHREADS);
    auto t0p = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < NTHREADS; t++) {
      threads.emplace_back([&, t]() {
        int lo = t * rows_per_thread;
        int hi = (t == NTHREADS - 1) ? SEQ_LEN : lo + rows_per_thread;
        for (int it = 0; it < ITERS; it++) {
          for (int i = lo; i < hi; i++) {
            __fp16 *o = out.data() + (size_t)i * (NUM_CACHE_HEAD * GQA_SIZE * HEAD_DIM);
            nntrainer::compute_fp16vcache_transposed(row_num, in.data(), vcache.data(),
                                                     o, NUM_CACHE_HEAD, GQA_SIZE,
                                                     HEAD_DIM, LOCAL_WINDOW, 0,
                                                     NUM_CACHE_HEAD);
          }
        }
      });
    }
    for (auto &th : threads)
      th.join();
    auto t1p = std::chrono::high_resolution_clock::now();
    double us_par = now_us(t0p, t1p) / ITERS;
    printf("[mha_cpu_bench] PREFILL pv_pre_128: %d-thread (pooled, spawn amortized) "
           "%8.2f us/chunk\n",
           NTHREADS, us_par);
    printf("PREFILL_PV_CPU_PAR_US %.2f\n", us_par);
  }
}

int main() {
  bench_kcaches_decode(1024);
  bench_vcache_decode(1024);
  bench_vcache_decode(512);
  bench_qk_prefill();
  bench_pv_prefill();
  return 0;
}
