// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   rtmcc_head.cpp
 * @date   15 July 2026
 * @brief  RTMCC SimCC pose head layer implementation (NCHW/NHWC-agnostic).
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "rtmcc_head.h"

namespace quick_ai {

static constexpr size_t SINGLE_INOUT_IDX = 0;

#ifdef __ARM_NEON
#include <arm_neon.h>

namespace {
// Vectorized expf (Cephes polynomial, ~1e-6 rel error) — identical to the
// conv layer's nntr_vexpq_f32. Used to vectorize the GAU SiLU epilogue, which
// otherwise runs a scalar (double-promoted) std::exp per element over
// K*(2E+S) ~= 100k values and dominates the head's non-GEMM cost.
static inline float32x4_t rtmcc_vexpq_f32(float32x4_t x) {
  const float32x4_t hi = vdupq_n_f32(88.3762626647950f);
  const float32x4_t lo = vdupq_n_f32(-88.3762626647949f);
  x = vminq_f32(vmaxq_f32(x, lo), hi);
  const float32x4_t log2e = vdupq_n_f32(1.44269504088896341f);
  const float32x4_t c0 = vdupq_n_f32(0.693359375f);
  const float32x4_t c1 = vdupq_n_f32(-2.12194440e-4f);
  const float32x4_t one = vdupq_n_f32(1.0f);
  float32x4_t fx = vmlaq_f32(vdupq_n_f32(0.5f), x, log2e);
  int32x4_t emm0 = vcvtq_s32_f32(fx);
  float32x4_t tmp = vcvtq_f32_s32(emm0);
  uint32x4_t mask = vcgtq_f32(tmp, fx);
  fx = vsubq_f32(
    tmp, vreinterpretq_f32_u32(vandq_u32(mask, vreinterpretq_u32_f32(one))));
  emm0 = vcvtq_s32_f32(fx);
  x = vmlsq_f32(x, fx, c0);
  x = vmlsq_f32(x, fx, c1);
  const float32x4_t p0 = vdupq_n_f32(1.9875691500E-4f);
  const float32x4_t p1 = vdupq_n_f32(1.3981999507E-3f);
  const float32x4_t p2 = vdupq_n_f32(8.3334519073E-3f);
  const float32x4_t p3 = vdupq_n_f32(4.1665795894E-2f);
  const float32x4_t p4 = vdupq_n_f32(1.6666665459E-1f);
  const float32x4_t p5 = vdupq_n_f32(5.0000001201E-1f);
  float32x4_t z = vmulq_f32(x, x);
  float32x4_t y = vmlaq_f32(p1, p0, x);
  y = vmlaq_f32(p2, y, x);
  y = vmlaq_f32(p3, y, x);
  y = vmlaq_f32(p4, y, x);
  y = vmlaq_f32(p5, y, x);
  y = vmlaq_f32(x, y, z);
  y = vaddq_f32(y, one);
  int32x4_t pow2n = vshlq_n_s32(vaddq_s32(emm0, vdupq_n_s32(0x7f)), 23);
  return vmulq_f32(y, vreinterpretq_f32_s32(pow2n));
}

// SiLU(x) = x / (1 + exp(-x)), in place. NEON path with two Newton-Raphson
// reciprocal refinements (matches the conv layer's exact-NEON SiLU precision).
static inline void rtmcc_silu_inplace(float *p, unsigned int n) {
  const float32x4_t one = vdupq_n_f32(1.0f);
  unsigned int i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t v = vld1q_f32(p + i);
    float32x4_t d = vaddq_f32(one, rtmcc_vexpq_f32(vnegq_f32(v)));
    float32x4_t r = vrecpeq_f32(d);
    r = vmulq_f32(r, vrecpsq_f32(d, r));
    r = vmulq_f32(r, vrecpsq_f32(d, r));
    vst1q_f32(p + i, vmulq_f32(v, r));
  }
  for (; i < n; ++i)
    p[i] = p[i] / (1.0f + std::exp(-p[i]));
}
} // namespace
#endif

void RTMCCHeadLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, head_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[rtmcc_head] Unknown Layer Properties count "
    << std::to_string(values.size());
}

void RTMCCHeadLayer::finalize(nntrainer::InitLayerContext &context) {
  const auto &in_dim = context.getInputDimensions()[0];

  // Conv output [B, K, H, W]: K keypoints (tokens), H*W spatial (feature).
  num_token = in_dim.channel();
  flatten = in_dim.height() * in_dim.width();

  e_dims = std::get<props::RtmccExpansion>(head_props).get();
  s_dims = std::get<props::RtmccKeyDim>(head_props).get();
  hidden = std::get<props::RtmccHidden>(head_props).get();
  simcc = std::get<props::RtmccSimcc>(head_props).get();
  is_nchw = (context.getFormat() == nntrainer::Tformat::NCHW);

  // Output: [B, 1, 2*K, SIMCC] (cls_x rows over cls_y rows). C=1 makes the
  // storage identical in NCHW and NHWC, so main.cpp decodes [2*K, SIMCC].
  // Always FP32 so the decoder reads float* regardless of the activation dtype
  // (the FP16-activation W8A16 path still yields an FP32 pose output).
  auto out_type = nntrainer::TensorDim::TensorType(
    context.getFormat(), nntrainer::TensorDim::DataType::FP32);
  nntrainer::TensorDim out_dim(in_dim.batch(), 1, 2 * num_token, simcc,
                               out_type);
  context.setOutputDimensions({out_dim});

  // Head weights + all internal token tensors are kept NCHW-typed regardless
  // of the model format: the input gather below is format-aware, but Tensor::dot
  // extracts its M/N/K from the (logical) last two dims assuming NCHW, so a
  // NHWC-typed operand would mis-shape the GEMM.
  auto wtype = nntrainer::TensorDim::TensorType(
    nntrainer::Tformat::NCHW, nntrainer::TensorDim::DataType::FP32);
  auto req = [&](const nntrainer::TensorDim &d, const std::string &name) {
    return context.requestWeight(d, nntrainer::props::InitializerInfo::Enum::NONE,
                                 nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f,
                                 name, true);
  };

  wt_idx[mlp_ln_g] = req({1, 1, 1, 1, wtype}, "mlp_ln_g");
  wt_idx[mlp_w] = req({1, 1, flatten, hidden, wtype}, "mlp_w");
  wt_idx[gau_uv] = req({1, 1, hidden, 2 * e_dims + s_dims, wtype}, "gau_uv");
  wt_idx[gau_o] = req({1, 1, e_dims, hidden, wtype}, "gau_o");
  wt_idx[gau_gamma] = req({1, 1, 2, s_dims, wtype}, "gau_gamma");
  wt_idx[gau_beta] = req({1, 1, 2, s_dims, wtype}, "gau_beta");
  wt_idx[gau_ln_g] = req({1, 1, 1, 1, wtype}, "gau_ln_g");
  wt_idx[gau_res_scale] = req({1, 1, 1, hidden, wtype}, "gau_res_scale");
  wt_idx[cls_x] = req({1, 1, hidden, simcc, wtype}, "cls_x");
  wt_idx[cls_y] = req({1, 1, hidden, simcc, wtype}, "cls_y");
}

void RTMCCHeadLayer::forwarding(nntrainer::RunLayerContext &context,
                                bool training) {
  incremental_forwarding(context, 0, num_token, training);
}

void RTMCCHeadLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                            unsigned int, unsigned int, bool) {
  using nntrainer::Tensor;
  using DataType = ml::train::TensorDim::DataType;

  Tensor &in_raw = context.getInput(SINGLE_INOUT_IDX);
  Tensor &out = context.getOutput(SINGLE_INOUT_IDX);

  // W8A8: the feature map may arrive as a per-tensor-scale QINT8 activation
  // (the producing conv is int8-resident). The head is an FP32 island, so
  // dequantize once up front and run the FP32 path unchanged.
  Tensor in = in_raw;
  std::vector<float> w8a8_deq;
  if (in_raw.getDataType() == DataType::QINT8) {
    const int8_t *q = in_raw.getData<int8_t>();
    const float s = in_raw.getScale<float>()[0];
    w8a8_deq.resize(in_raw.size());
    for (size_t i = 0; i < w8a8_deq.size(); ++i)
      w8a8_deq[i] = s * (float)q[i];
    nntrainer::TensorDim d = in_raw.getDim();
    d.setDataType(DataType::FP32);
    in = nntrainer::Tensor::Map<float>(w8a8_deq.data(),
                                       w8a8_deq.size() * sizeof(float), d);
  }

  const bool in_fp16 = (in.getDataType() == DataType::FP16);
#ifndef ENABLE_FP16
  NNTR_THROW_IF(in_fp16, std::invalid_argument)
    << "[rtmcc_head] FP16 activations require an ENABLE_FP16 build";
#endif
  NNTR_THROW_IF(!in_fp16 && in.getDataType() != DataType::FP32,
                std::invalid_argument)
    << "[rtmcc_head] only FP32/FP16 activations are supported";

  const unsigned int K = num_token;
  const unsigned int F = flatten;
  const unsigned int D = hidden;
  const unsigned int E = e_dims;
  const unsigned int S = s_dims;
  const unsigned int UV = 2 * E + S;
  const float eps = std::get<nntrainer::props::Epsilon>(head_props).get();
  const float mlp_scale = 1.0f / std::sqrt(static_cast<float>(F));
  const float gau_scale = 1.0f / std::sqrt(static_cast<float>(D));
  const float sqrt_s = std::sqrt(static_cast<float>(S));

  auto silu = [](float v) { return v / (1.0f + std::exp(-v)); };
  (void)silu; // ARM builds vectorize SiLU below; keep for the scalar fallback

  Tensor &w_mlp_g = context.getWeight(wt_idx[mlp_ln_g]);
  Tensor &w_mlp = context.getWeight(wt_idx[mlp_w]);
  Tensor &w_uv = context.getWeight(wt_idx[gau_uv]);
  Tensor &w_o = context.getWeight(wt_idx[gau_o]);
  Tensor &t_gamma = context.getWeight(wt_idx[gau_gamma]);
  Tensor &t_beta = context.getWeight(wt_idx[gau_beta]);
  Tensor &w_gau_g = context.getWeight(wt_idx[gau_ln_g]);
  Tensor &t_rs = context.getWeight(wt_idx[gau_res_scale]);
  Tensor &w_cx = context.getWeight(wt_idx[cls_x]);
  Tensor &w_cy = context.getWeight(wt_idx[cls_y]);

  const float mlp_g = w_mlp_g.getData<float>()[0];
  const float gau_g = w_gau_g.getData<float>()[0];
  const float *gamma_p = t_gamma.getData<float>();
  const float *beta_p = t_beta.getData<float>();
  const float *rs_p = t_rs.getData<float>();

  const nntrainer::TensorDim in_dim = in.getDim();
  const unsigned int batch = in_dim.batch();
  const float *in_all = in_fp16 ? nullptr : in.getData<float>();
#ifdef ENABLE_FP16
  const _FP16 *in_all16 = in_fp16 ? in.getData<_FP16>() : nullptr;
#endif
  float *out_all = out.getData<float>();
  const size_t in_feat = static_cast<size_t>(K) * F;
  const size_t out_feat = static_cast<size_t>(2 * K) * simcc;

  // read conv-output element (keypoint k, spatial s) as float, format-aware.
  auto read_in = [&](size_t base, unsigned int k, unsigned int s) -> float {
    size_t idx = is_nchw ? (static_cast<size_t>(k) * F + s)
                         : (static_cast<size_t>(s) * K + k);
#ifdef ENABLE_FP16
    if (in_fp16)
      return static_cast<float>(in_all16[base + idx]);
#endif
    return in_all[base + idx];
  };

  // Internal token tensors are NCHW so Tensor::dot shapes the GEMM correctly.
  const auto nchw = nntrainer::TensorDim::TensorType(
    nntrainer::Tformat::NCHW, nntrainer::TensorDim::DataType::FP32);
  nntrainer::TensorDim kf(1, 1, K, F, nchw);
  nntrainer::TensorDim kd(1, 1, K, D, nchw);

  for (unsigned int b = 0; b < batch; ++b) {
    const size_t base = static_cast<size_t>(b) * in_feat;
    float *outb = out_all + b * out_feat;

    // 1. Gather tokens X[K, F] from the conv output (format-aware) and apply
    //    ScaleNorm (RMS over F, scalar gain mlp_g).
    Tensor X(kf, true);
    float *xp = X.getData<float>();
    for (unsigned int k = 0; k < K; ++k) {
      float ss = 0.0f;
      for (unsigned int s = 0; s < F; ++s) {
        float v = read_in(base, k, s);
        xp[k * F + s] = v;
        ss += v * v;
      }
      float norm = std::sqrt(ss) * mlp_scale;
      if (norm < eps)
        norm = eps;
      float inv = mlp_g / norm;
      for (unsigned int s = 0; s < F; ++s)
        xp[k * F + s] *= inv;
    }


    // 2. mlp: H = X @ Wmlp  -> [K, D]
    Tensor H = X.dot(w_mlp);

    // 3. GAU on H -> G [K, D].
    // 3a. ScaleNorm over D (scalar gau_g).
    Tensor H_ln(kd, true);
    const float *hp = H.getData<float>();
    float *hlp = H_ln.getData<float>();
    for (unsigned int k = 0; k < K; ++k) {
      float ss = 0.0f;
      for (unsigned int d = 0; d < D; ++d)
        ss += hp[k * D + d] * hp[k * D + d];
      float norm = std::sqrt(ss) * gau_scale;
      if (norm < eps)
        norm = eps;
      float inv = gau_g / norm;
      for (unsigned int d = 0; d < D; ++d)
        hlp[k * D + d] = hp[k * D + d] * inv;
    }


    // 3b. uv = SiLU(H_ln @ Wuv) ; split u, v, base ; build q, k.
    Tensor uv = H_ln.dot(w_uv);
    float *uvp = uv.getData<float>();
#ifdef __ARM_NEON
    rtmcc_silu_inplace(uvp, K * UV);
#else
    for (unsigned int i = 0; i < K * UV; ++i)
      uvp[i] = silu(uvp[i]);
#endif

    nntrainer::TensorDim ke(1, 1, K, E, nchw);
    nntrainer::TensorDim ks(1, 1, K, S, nchw);
    Tensor u(ke, true), v(ke, true), q(ks, true), kk(ks, true);
    float *up = u.getData<float>();
    float *vp = v.getData<float>();
    float *qp = q.getData<float>();
    float *kp = kk.getData<float>();
    for (unsigned int k = 0; k < K; ++k) {
      const float *row = uvp + k * UV;
      std::copy(row, row + E, up + k * E);
      std::copy(row + E, row + 2 * E, vp + k * E);
      const float *base = row + 2 * E;
      for (unsigned int s = 0; s < S; ++s) {
        qp[k * S + s] = base[s] * gamma_p[s] + beta_p[s];
        kp[k * S + s] = base[s] * gamma_p[S + s] + beta_p[S + s];
      }
    }

    // 3c. kernel = relu(q k^T / sqrt_s)^2 ; attn = kernel @ v ; gated = u*attn.
    Tensor qk = q.dot(kk, false, true); // [K, K]
    float *qkp = qk.getData<float>();
    for (unsigned int i = 0; i < K * K; ++i) {
      float r = qkp[i] / sqrt_s;
      r = r > 0.0f ? r : 0.0f;
      qkp[i] = r * r;
    }
    Tensor attn = qk.dot(v); // [K, E]
    float *ap = attn.getData<float>();
    for (unsigned int i = 0; i < K * E; ++i)
      ap[i] *= up[i];

    // 3d. out_gau = gated @ Wo ; G = res_scale * H + out_gau.
    Tensor out_gau = attn.dot(w_o); // [K, D]
    const float *ogp = out_gau.getData<float>();
    Tensor G(kd, true);
    float *gp = G.getData<float>();
    for (unsigned int k = 0; k < K; ++k)
      for (unsigned int d = 0; d < D; ++d)
        gp[k * D + d] = rs_p[d] * hp[k * D + d] + ogp[k * D + d];

    // 4. cls_x, cls_y ; write [cls_x rows ; cls_y rows] into the output.
    Tensor px = G.dot(w_cx); // [K, simcc]
    Tensor py = G.dot(w_cy); // [K, simcc]
    const float *pxp = px.getData<float>();
    const float *pyp = py.getData<float>();
    std::copy(pxp, pxp + K * simcc, outb);
    std::copy(pyp, pyp + K * simcc, outb + static_cast<size_t>(K) * simcc);
  }
}

void RTMCCHeadLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw std::runtime_error("[rtmcc_head] training is not supported");
}

#ifdef PLUGGABLE

nntrainer::Layer *create_rtmcc_head_layer() { return new RTMCCHeadLayer(); }

void destroy_rtmcc_head_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_rtmcc_head_layer,
                                                   destroy_rtmcc_head_layer};
}

#endif

} // namespace quick_ai
