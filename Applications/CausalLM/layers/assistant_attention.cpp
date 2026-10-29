// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   assistant_attention.cpp
 * @date   29 Oct 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Query-only attention of the Gemma 4 assistant over shared target
 *         K/V rows (assistant_attention.h)
 */

#include <algorithm>
#include <cmath>
#include <vector>

#include <assistant_attention.h>
#include <nntrainer_error.h>

namespace causallm {

void AssistantAttentionLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain = loadProperties(values, props);
  NNTR_THROW_IF(!remain.empty(), std::invalid_argument)
    << "[assistant_attention] unknown property " << remain[0];
}

void AssistantAttentionLayer::finalize(nntrainer::InitLayerContext &context) {
  const auto &in = context.getInputDimensions()[0];
  const unsigned int n_q = std::get<nntrainer::props::NumHeads>(props).get();
  const unsigned int n_kv = std::get<props::NumHeads_KV>(props).get();
  NNTR_THROW_IF(in.width() % n_q != 0 || n_q % n_kv != 0 ||
                  (in.width() / n_q) % 2 != 0,
                std::invalid_argument)
    << "[assistant_attention] width " << in.width() << " does not split into "
    << n_q << " even heads over " << n_kv << " kv heads";
  context.setOutputDimensions({in});
}

/**
 * @brief Attend @a rows query rows of @a q_in to the shared K/V into @a out.
 *        RoPE as Gemma4TextRotaryEmbedding: the angle of pair (i, i + hd/2)
 *        is pos * theta^(-2i/hd) for i < partial * hd / 2 and 0 above.
 */
static void attend(const float *q_in, float *out, unsigned int rows,
                   unsigned int n_q, unsigned int n_kv, unsigned int hd,
                   float theta, float partial, const float *k, const float *v,
                   unsigned int n, unsigned int pos) {
  const unsigned int half = hd / 2, kv_w = n_kv * hd, group = n_q / n_kv;
  const unsigned int angles = static_cast<unsigned int>(partial * hd / 2);
  std::vector<float> c(half, 1.0f), s(half, 0.0f), q(hd), p(n);
  for (unsigned int i = 0; i < angles; ++i) {
    const float inv = 1.0f / std::pow(theta, 2.0f * i / hd);
    const float a = static_cast<float>(pos) * inv;
    c[i] = std::cos(a);
    s[i] = std::sin(a);
  }
  for (unsigned int r = 0; r < rows; ++r) {
    for (unsigned int h = 0; h < n_q; ++h) {
      const float *x = q_in + (size_t)r * n_q * hd + (size_t)h * hd;
      const unsigned int off = (h / group) * hd;
      for (unsigned int i = 0; i < half; ++i) {
        q[i] = x[i] * c[i] - x[i + half] * s[i];
        q[i + half] = x[i + half] * c[i] + x[i] * s[i];
      }
      float m = -INFINITY, l = 0.0f;
      for (unsigned int j = 0; j < n; ++j) {
        const float *kj = k + (size_t)j * kv_w + off;
        float d = 0.0f;
        for (unsigned int i = 0; i < hd; ++i)
          d += q[i] * kj[i];
        p[j] = d;
        m = std::max(m, d);
      }
      for (unsigned int j = 0; j < n; ++j) {
        p[j] = std::exp(p[j] - m);
        l += p[j];
      }
      float *y = out + (size_t)r * n_q * hd + (size_t)h * hd;
      std::fill(y, y + hd, 0.0f);
      for (unsigned int j = 0; j < n; ++j) {
        const float *vj = v + (size_t)j * kv_w + off;
        const float w = p[j] / l;
        for (unsigned int i = 0; i < hd; ++i)
          y[i] += w * vj[i];
      }
    }
  }
}

void AssistantAttentionLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::Tensor &in = context.getInput(0);
  nntrainer::Tensor &out = context.getOutput(0);
  NNTR_THROW_IF(in.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "[assistant_attention] FP32 activations only";
  NNTR_THROW_IF(k_ == nullptr || v_ == nullptr || n_rows_ == 0,
                std::invalid_argument)
    << "[assistant_attention] setSharedKV() was not called";
  const unsigned int n_q = std::get<nntrainer::props::NumHeads>(props).get();
  const unsigned int n_kv = std::get<props::NumHeads_KV>(props).get();
  const unsigned int hd = in.width() / n_q;
  const unsigned int rows = to - from;
  for (unsigned int b = 0; b < in.batch(); ++b)
    attend(
      in.getData<float>() + (size_t)b * in.getDim().getFeatureLen(),
      out.getData<float>() + (size_t)b * out.getDim().getFeatureLen(), rows,
      n_q, n_kv, hd,
      static_cast<float>(std::get<nntrainer::props::RopeTheta>(props).get()),
      std::get<nntrainer::props::RopePartialRotaryFactor>(props).get(), k_, v_,
      n_rows_, pos_);
}

void AssistantAttentionLayer::forwarding(nntrainer::RunLayerContext &context,
                                         bool training) {
  incremental_forwarding(context, 0, context.getInput(0).height(), training);
}

void AssistantAttentionLayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  throw std::runtime_error("[assistant_attention] training is not supported");
}

} // namespace causallm
