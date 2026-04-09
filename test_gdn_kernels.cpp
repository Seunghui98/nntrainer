// SPDX-License-Identifier: Apache-2.0
// Unit tests for GDN kernel optimizations
// Tests scalar reference vs SIMD-optimized implementations

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>

#ifdef __AVX2__
#include <immintrin.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// ============================================================
// Scalar reference implementations (ground truth)
// ============================================================

namespace ref {

static inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static inline float silu(float x) { return x * sigmoid(x); }
static inline float softplus(float x) {
  return x > 20.0f ? x : std::log(1.0f + std::exp(x));
}

static void l2_normalize(float *data, unsigned int len, float eps = 1e-6f) {
  float norm_sq = 0.0f;
  for (unsigned int i = 0; i < len; ++i) norm_sq += data[i] * data[i];
  float inv = 1.0f / std::sqrt(norm_sq + eps);
  for (unsigned int i = 0; i < len; ++i) data[i] *= inv;
}

static void matvec_transposed(const float *A, const float *x, float *out,
                              unsigned int rows, unsigned int cols) {
  std::memset(out, 0, cols * sizeof(float));
  for (unsigned int i = 0; i < rows; ++i)
    for (unsigned int j = 0; j < cols; ++j)
      out[j] += A[i * cols + j] * x[i];
}

static void ssm_state_update(float *S, const float *k, const float *delta,
                             float decay, unsigned int k_dim, unsigned int v_dim) {
  for (unsigned int ki = 0; ki < k_dim; ++ki)
    for (unsigned int vi = 0; vi < v_dim; ++vi)
      S[ki * v_dim + vi] = decay * S[ki * v_dim + vi] + k[ki] * delta[vi];
}

static void gated_rms_silu(const float *in, const float *z, float *out,
                           const float *w, unsigned int len, float eps = 1e-6f) {
  float sum_sq = 0.0f;
  for (unsigned int i = 0; i < len; ++i) sum_sq += in[i] * in[i];
  float scale = 1.0f / std::sqrt(sum_sq / len + eps);
  for (unsigned int i = 0; i < len; ++i)
    out[i] = in[i] * scale * w[i] * silu(z[i]);
}

static void silu_inplace(float *data, unsigned int len) {
  for (unsigned int i = 0; i < len; ++i) data[i] = silu(data[i]);
}

static void attn_gate(const float *in, const float *gate, float *out, unsigned int len) {
  for (unsigned int i = 0; i < len; ++i)
    out[i] = in[i] * sigmoid(gate[i]);
}

// Conv1d with circular buffer (reference)
static void conv1d_circular(const float *x_ch, float *state, unsigned int *write_pos,
                            const float *kernel, unsigned int kernel_size,
                            unsigned int channels, float *out) {
  unsigned int state_width = kernel_size - 1;
  for (unsigned int ch = 0; ch < channels; ++ch) {
    float val = 0.0f;
    unsigned int wp = write_pos[ch];
    for (unsigned int k = 0; k < state_width; ++k) {
      unsigned int idx = (wp + k) % state_width;
      val += state[ch * state_width + idx] * kernel[ch * kernel_size + k];
    }
    val += x_ch[ch] * kernel[ch * kernel_size + state_width];
    out[ch] = val;
    state[ch * state_width + wp] = x_ch[ch];
    write_pos[ch] = (wp + 1) % state_width;
  }
}

} // namespace ref

// ============================================================
// Optimized implementations (to be tested)
// ============================================================

namespace opt {

// --- Fast sigmoid approximation ---
static inline float fast_sigmoid(float x) {
  // Piecewise linear + rational approximation, max error < 5e-4
  if (x > 6.0f) return 1.0f;
  if (x < -6.0f) return 0.0f;
  return 1.0f / (1.0f + std::exp(-x));  // exact for now, can swap to poly
}
static inline float fast_silu(float x) { return x * fast_sigmoid(x); }

#ifdef __AVX2__
static inline __m256 fast_sigmoid_avx2(__m256 x) {
  alignas(32) float buf[8];
  _mm256_store_ps(buf, x);
  for (int i = 0; i < 8; ++i)
    buf[i] = 1.0f / (1.0f + std::exp(-buf[i]));
  return _mm256_load_ps(buf);
}
static inline __m256 fast_silu_avx2(__m256 x) {
  return _mm256_mul_ps(x, fast_sigmoid_avx2(x));
}
#endif

// --- L2 normalize ---
static void l2_normalize(float *data, unsigned int len, float eps = 1e-6f) {
  float norm_sq = 0.0f;
  unsigned int i = 0;
#ifdef __AVX2__
  __m256 sum_v = _mm256_setzero_ps();
  for (; i + 7 < len; i += 8) {
    __m256 v = _mm256_loadu_ps(&data[i]);
    sum_v = _mm256_fmadd_ps(v, v, sum_v);
  }
  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, sum_v);
  norm_sq = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
#endif
  for (; i < len; ++i) norm_sq += data[i] * data[i];
  float inv = 1.0f / std::sqrt(norm_sq + eps);
  i = 0;
#ifdef __AVX2__
  __m256 sc = _mm256_set1_ps(inv);
  for (; i + 7 < len; i += 8)
    _mm256_storeu_ps(&data[i], _mm256_mul_ps(_mm256_loadu_ps(&data[i]), sc));
#endif
  for (; i < len; ++i) data[i] *= inv;
}

// --- Batched L2 normalize across multiple heads ---
static void l2_normalize_batched(float *data, unsigned int num_heads,
                                 unsigned int head_dim, float eps = 1e-6f) {
  for (unsigned int h = 0; h < num_heads; ++h)
    l2_normalize(data + h * head_dim, head_dim, eps);
}

// --- matvec_transposed ---
static void matvec_transposed(const float *A, const float *x, float *out,
                              unsigned int rows, unsigned int cols) {
  std::memset(out, 0, cols * sizeof(float));
  for (unsigned int i = 0; i < rows; ++i) {
    const float *row = A + i * cols;
    float x_val = x[i];
    unsigned int j = 0;
#ifdef __AVX2__
    __m256 xv = _mm256_set1_ps(x_val);
    for (; j + 7 < cols; j += 8) {
      __m256 o = _mm256_loadu_ps(&out[j]);
      __m256 a = _mm256_loadu_ps(&row[j]);
      _mm256_storeu_ps(&out[j], _mm256_fmadd_ps(a, xv, o));
    }
#endif
    for (; j < cols; ++j) out[j] += row[j] * x_val;
  }
}

// --- SSM state update ---
static void ssm_state_update(float *S, const float *k, const float *delta,
                             float decay, unsigned int k_dim, unsigned int v_dim) {
  for (unsigned int ki = 0; ki < k_dim; ++ki) {
    float k_val = k[ki];
    float *row = S + ki * v_dim;
    unsigned int vi = 0;
#ifdef __AVX2__
    __m256 dv = _mm256_set1_ps(decay);
    __m256 kv = _mm256_set1_ps(k_val);
    for (; vi + 7 < v_dim; vi += 8) {
      __m256 s = _mm256_loadu_ps(&row[vi]);
      __m256 d = _mm256_loadu_ps(&delta[vi]);
      _mm256_storeu_ps(&row[vi], _mm256_fmadd_ps(kv, d, _mm256_mul_ps(s, dv)));
    }
#endif
    for (; vi < v_dim; ++vi)
      row[vi] = decay * row[vi] + k_val * delta[vi];
  }
}

// --- Gated RMS SiLU (fused) ---
static void gated_rms_silu(const float *in, const float *z, float *out,
                           const float *w, unsigned int len, float eps = 1e-6f) {
  float sum_sq = 0.0f;
  unsigned int i = 0;
#ifdef __AVX2__
  __m256 sv = _mm256_setzero_ps();
  for (; i + 7 < len; i += 8) {
    __m256 v = _mm256_loadu_ps(&in[i]);
    sv = _mm256_fmadd_ps(v, v, sv);
  }
  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, sv);
  sum_sq = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
#endif
  for (; i < len; ++i) sum_sq += in[i] * in[i];
  float scale = 1.0f / std::sqrt(sum_sq / len + eps);
  i = 0;
#ifdef __AVX2__
  __m256 sc = _mm256_set1_ps(scale);
  for (; i + 7 < len; i += 8) {
    __m256 x = _mm256_loadu_ps(&in[i]);
    __m256 wv = _mm256_loadu_ps(&w[i]);
    __m256 zv = _mm256_loadu_ps(&z[i]);
    __m256 normed = _mm256_mul_ps(_mm256_mul_ps(x, sc), wv);
    _mm256_storeu_ps(&out[i], _mm256_mul_ps(normed, fast_silu_avx2(zv)));
  }
#endif
  for (; i < len; ++i)
    out[i] = in[i] * scale * w[i] * fast_silu(z[i]);
}

// --- Batched gated_rms_silu across heads ---
static void gated_rms_silu_batched(float *data, const float *z,
                                   const float *norm_w,
                                   unsigned int num_heads,
                                   unsigned int head_v_dim, float eps = 1e-6f) {
  for (unsigned int h = 0; h < num_heads; ++h)
    gated_rms_silu(data + h * head_v_dim, z + h * head_v_dim,
                   data + h * head_v_dim, norm_w, head_v_dim, eps);
}

// --- SiLU inplace (fixed) ---
static void silu_inplace(float *data, unsigned int len) {
  unsigned int i = 0;
#ifdef __AVX2__
  for (; i + 7 < len; i += 8) {
    __m256 v = _mm256_loadu_ps(&data[i]);
    _mm256_storeu_ps(&data[i], _mm256_mul_ps(v, fast_sigmoid_avx2(v)));
  }
#endif
  for (; i < len; ++i) data[i] = fast_silu(data[i]);
}

// --- Attn gate ---
static void attn_gate(const float *in, const float *gate, float *out, unsigned int len) {
  unsigned int i = 0;
#ifdef __AVX2__
  for (; i + 7 < len; i += 8) {
    __m256 iv = _mm256_loadu_ps(&in[i]);
    __m256 gv = _mm256_loadu_ps(&gate[i]);
    _mm256_storeu_ps(&out[i], _mm256_mul_ps(iv, fast_sigmoid_avx2(gv)));
  }
#endif
  for (; i < len; ++i) out[i] = in[i] * fast_sigmoid(gate[i]);
}

// --- Conv1d with circular buffer + channel vectorization ---
// State layout: (kernel_size-1, channels) transposed for SIMD channel access
static void conv1d_circular_vectorized(
    const float *x_ch, float *state, unsigned int write_pos,
    const float *kernel, unsigned int kernel_size,
    unsigned int channels, float *out) {
  // kernel layout: (kernel_size, channels) - transposed for vectorization
  // state layout: (kernel_size-1, channels) - transposed
  unsigned int state_width = kernel_size - 1;

  // Compute conv output: sum over kernel positions
  std::memset(out, 0, channels * sizeof(float));

  // Accumulate state contributions
  for (unsigned int k = 0; k < state_width; ++k) {
    unsigned int state_row = (write_pos + k) % state_width;
    const float *s_row = state + state_row * channels;
    const float *k_row = kernel + k * channels;
    unsigned int ch = 0;
#ifdef __AVX2__
    for (; ch + 7 < channels; ch += 8) {
      __m256 o = _mm256_loadu_ps(&out[ch]);
      __m256 sv = _mm256_loadu_ps(&s_row[ch]);
      __m256 kv = _mm256_loadu_ps(&k_row[ch]);
      _mm256_storeu_ps(&out[ch], _mm256_fmadd_ps(sv, kv, o));
    }
#endif
    for (; ch < channels; ++ch)
      out[ch] += s_row[ch] * k_row[ch];
  }

  // Add current input contribution (last kernel position)
  const float *k_last = kernel + state_width * channels;
  unsigned int ch = 0;
#ifdef __AVX2__
  for (; ch + 7 < channels; ch += 8) {
    __m256 o = _mm256_loadu_ps(&out[ch]);
    __m256 xv = _mm256_loadu_ps(&x_ch[ch]);
    __m256 kv = _mm256_loadu_ps(&k_last[ch]);
    _mm256_storeu_ps(&out[ch], _mm256_fmadd_ps(xv, kv, o));
  }
#endif
  for (; ch < channels; ++ch)
    out[ch] += x_ch[ch] * k_last[ch];

  // Update state: write new values at write_pos (no shift needed!)
  float *s_write = state + write_pos * channels;
  std::memcpy(s_write, x_ch, channels * sizeof(float));
}

} // namespace opt

// ============================================================
// Test harness
// ============================================================

static float max_abs_diff(const float *a, const float *b, unsigned int len) {
  float m = 0;
  for (unsigned int i = 0; i < len; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}
static void fill_random(float *d, unsigned int len, float lo=-1, float hi=1) {
  for (unsigned int i = 0; i < len; ++i) d[i] = lo + (float)rand()/RAND_MAX*(hi-lo);
}

static int g_pass = 0, g_fail = 0;
static void check(const char *name, float max_diff, float tol) {
  if (max_diff < tol) { printf("  PASS: %-40s (diff=%.2e)\n", name, max_diff); g_pass++; }
  else { printf("  FAIL: %-40s (diff=%.2e > tol=%.2e)\n", name, max_diff, tol); g_fail++; }
}

int main() {
  srand(42);
  const float tol = 1e-4f;
  float md;

  printf("=== GDN Kernel Optimization Unit Tests ===\n\n");

  // --- l2_normalize ---
  printf("[l2_normalize]\n");
  for (unsigned len : {16u, 128u, 130u, 256u}) {
    std::vector<float> a(len), b(len);
    fill_random(a.data(), len);
    std::copy(a.begin(), a.end(), b.begin());
    opt::l2_normalize(a.data(), len);
    ref::l2_normalize(b.data(), len);
    md = max_abs_diff(a.data(), b.data(), len);
    char nm[64]; snprintf(nm, 64, "len=%u", len); check(nm, md, tol);
  }

  // --- matvec_transposed ---
  printf("[matvec_transposed]\n");
  for (auto [r,c] : std::vector<std::pair<unsigned,unsigned>>{{8,8},{128,128},{16,130},{128,256}}) {
    std::vector<float> A(r*c), x(r), oa(c), ob(c);
    fill_random(A.data(), r*c); fill_random(x.data(), r);
    opt::matvec_transposed(A.data(), x.data(), oa.data(), r, c);
    ref::matvec_transposed(A.data(), x.data(), ob.data(), r, c);
    md = max_abs_diff(oa.data(), ob.data(), c);
    char nm[64]; snprintf(nm, 64, "%ux%u", r, c); check(nm, md, tol);
  }

  // --- ssm_state_update ---
  printf("[ssm_state_update]\n");
  for (auto [kd,vd] : std::vector<std::pair<unsigned,unsigned>>{{8,8},{128,128},{16,130}}) {
    std::vector<float> sa(kd*vd), sb(kd*vd), k(kd), d(vd);
    fill_random(sa.data(), kd*vd); std::copy(sa.begin(), sa.end(), sb.begin());
    fill_random(k.data(), kd); fill_random(d.data(), vd);
    opt::ssm_state_update(sa.data(), k.data(), d.data(), 0.95f, kd, vd);
    ref::ssm_state_update(sb.data(), k.data(), d.data(), 0.95f, kd, vd);
    md = max_abs_diff(sa.data(), sb.data(), kd*vd);
    char nm[64]; snprintf(nm, 64, "k=%u,v=%u", kd, vd); check(nm, md, tol);
  }

  // --- gated_rms_silu ---
  printf("[gated_rms_silu]\n");
  for (unsigned len : {8u, 128u, 130u, 256u}) {
    std::vector<float> in(len), z(len), w(len), oa(len), ob(len);
    fill_random(in.data(), len, -2, 2); fill_random(z.data(), len, -2, 2);
    fill_random(w.data(), len, 0.5, 1.5);
    opt::gated_rms_silu(in.data(), z.data(), oa.data(), w.data(), len);
    ref::gated_rms_silu(in.data(), z.data(), ob.data(), w.data(), len);
    md = max_abs_diff(oa.data(), ob.data(), len);
    char nm[64]; snprintf(nm, 64, "len=%u", len); check(nm, md, tol);
  }

  // --- gated_rms_silu in-place ---
  printf("[gated_rms_silu_inplace]\n");
  for (unsigned len : {128u, 256u}) {
    std::vector<float> a(len), b(len), z(len), w(len);
    fill_random(a.data(), len, -2, 2); std::copy(a.begin(), a.end(), b.begin());
    fill_random(z.data(), len, -2, 2); fill_random(w.data(), len, 0.5, 1.5);
    opt::gated_rms_silu(a.data(), z.data(), a.data(), w.data(), len);
    ref::gated_rms_silu(b.data(), z.data(), b.data(), w.data(), len);
    md = max_abs_diff(a.data(), b.data(), len);
    char nm[64]; snprintf(nm, 64, "inplace len=%u", len); check(nm, md, tol);
  }

  // --- silu_inplace ---
  printf("[silu_inplace]\n");
  for (unsigned len : {16u, 128u, 6144u}) {
    std::vector<float> a(len), b(len);
    fill_random(a.data(), len, -3, 3); std::copy(a.begin(), a.end(), b.begin());
    opt::silu_inplace(a.data(), len);
    ref::silu_inplace(b.data(), len);
    md = max_abs_diff(a.data(), b.data(), len);
    char nm[64]; snprintf(nm, 64, "len=%u", len); check(nm, md, tol);
  }

  // --- attn_gate ---
  printf("[attn_gate]\n");
  for (unsigned len : {16u, 2048u, 2050u}) {
    std::vector<float> in(len), g(len), oa(len), ob(len);
    fill_random(in.data(), len, -2, 2); fill_random(g.data(), len, -3, 3);
    opt::attn_gate(in.data(), g.data(), oa.data(), len);
    ref::attn_gate(in.data(), g.data(), ob.data(), len);
    md = max_abs_diff(oa.data(), ob.data(), len);
    char nm[64]; snprintf(nm, 64, "len=%u", len); check(nm, md, tol);
  }

  // --- conv1d_circular_vectorized ---
  printf("[conv1d_circular_vectorized]\n");
  {
    unsigned int channels = 6144, ks = 4;
    unsigned int sw = ks - 1;
    // Reference: (channels, kernel_size) layout
    std::vector<float> kernel_ref(channels * ks);
    fill_random(kernel_ref.data(), channels * ks, -0.5, 0.5);
    // Optimized: (kernel_size, channels) transposed layout
    std::vector<float> kernel_opt(ks * channels);
    for (unsigned ch = 0; ch < channels; ++ch)
      for (unsigned k = 0; k < ks; ++k)
        kernel_opt[k * channels + ch] = kernel_ref[ch * ks + k];

    // State: ref=(channels, sw), opt=(sw, channels)
    std::vector<float> state_ref(channels * sw, 0.0f);
    std::vector<float> state_opt(sw * channels, 0.0f);
    std::vector<unsigned int> write_pos_ref(channels, 0);
    unsigned int write_pos_opt = 0;

    // Run 5 tokens
    for (int t = 0; t < 5; ++t) {
      std::vector<float> x(channels), out_ref(channels), out_opt(channels);
      fill_random(x.data(), channels, -1, 1);

      ref::conv1d_circular(x.data(), state_ref.data(), write_pos_ref.data(),
                           kernel_ref.data(), ks, channels, out_ref.data());
      opt::conv1d_circular_vectorized(x.data(), state_opt.data(), write_pos_opt,
                                      kernel_opt.data(), ks, channels, out_opt.data());
      write_pos_opt = (write_pos_opt + 1) % sw;

      md = max_abs_diff(out_ref.data(), out_opt.data(), channels);
      char nm[64]; snprintf(nm, 64, "token=%d channels=%u", t, channels); check(nm, md, tol);
    }
  }

  // --- Batched L2 normalize ---
  printf("[l2_normalize_batched]\n");
  {
    unsigned int nh = 16, hd = 128, total = nh * hd;
    std::vector<float> a(total), b(total);
    fill_random(a.data(), total); std::copy(a.begin(), a.end(), b.begin());
    opt::l2_normalize_batched(a.data(), nh, hd);
    for (unsigned h = 0; h < nh; ++h) ref::l2_normalize(b.data() + h*hd, hd);
    md = max_abs_diff(a.data(), b.data(), total);
    check("16heads x 128dim", md, tol);
  }

  printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
  return g_fail > 0 ? 1 : 0;
}
