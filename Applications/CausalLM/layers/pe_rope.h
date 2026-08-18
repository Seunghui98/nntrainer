// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   pe_rope.h
 * @date   18 Aug 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Static-table interleaved RoPE layer for the PE-Lang-L14-448 encoder.
 *
 * Inputs (in order):
 *   [0] q_or_k    [B, 1, SEQ, C]   with C = num_heads * head_dim (= 1024)
 *   [1] rope_sin  [1, 1, PATCH_N, head_dim]  head-broadcast static table
 *   [2] rope_cos  [1, 1, PATCH_N, head_dim]
 *
 * Output: same shape as input[0]. The leading `num_prefix_tokens` rows (CLS)
 * are passed through unchanged (EVA convention — plan §1.3, trap §6.6). Patch
 * rows [num_prefix_tokens, SEQ) are rotated by the **interleaved** RoPE rule
 *   out[r, h*hd + 2i]   = x[...] * cos - x[h*hd + 2i+1] * sin
 *   out[r, h*hd + 2i+1] = x[...] * cos + x[h*hd + 2i]   * sin
 * where cos/sin index by (r - num_prefix_tokens, 2i) in the [1024, 64] table.
 *
 * Notes:
 *   - half=False (interleaved), NOT rotate_half — trap §6.7. Don't reuse
 *     mha_core's internal use_rope path; that one is rotate_half-style.
 *   - The rope table is shared across all 23 blocks (rope_mixed=False) —
 *     trap §6.15. Two `weight` layers (pe_rope_sin, pe_rope_cos) are created
 *     ONCE at graph top; every block's pe_rope instance consumes the same
 *     constant tensors, so no per-block duplication.
 *   - This is a 0-weight layer: sin/cos flow as inputs, not via context weights.
 *
 * ponytail: NEON vld2q_f32 / vfmaq fastpath is deferred to Phase 5 (§4.4 —
 * RoPE is <0.05% of encoder cost). The x86 fp32 parity path needs the simple
 * loop; speculatively hand-rolling NEON here would invite the trap §6.13
 * (premature kernel work without profiling).
 */

#ifndef __PE_ROPE_LAYER_H__
#define __PE_ROPE_LAYER_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>
#include <utility>

#include <causallm_common_properties.h>
#include <common_properties.h>
#include <connection.h>
#include <tensor.h>
#include <tensor_wrap_specs.h>

namespace causallm {

namespace props {

/**
 * @brief Number of leading prefix tokens that bypass rotation (CLS for EVA).
 */
class NumPrefixTokens : public nntrainer::PositiveIntegerProperty {
public:
  NumPrefixTokens(unsigned int value = 1) { set(value); }
  static constexpr const char *key = "num_prefix_tokens";
  using prop_tag = nntrainer::uint_prop_tag;
};

/**
 * @brief Number of attention heads (last-dim C = num_heads * head_dim).
 */
class NumHeadsRope : public nntrainer::PositiveIntegerProperty {
public:
  NumHeadsRope(unsigned int value = 16) { set(value); }
  static constexpr const char *key = "num_heads";
  using prop_tag = nntrainer::uint_prop_tag;
};

/**
 * @brief Head dimension (per-head channel width — also rope table width).
 */
class HeadDimRope : public nntrainer::PositiveIntegerProperty {
public:
  HeadDimRope(unsigned int value = 64) { set(value); }
  static constexpr const char *key = "head_dim";
  using prop_tag = nntrainer::uint_prop_tag;
};

} // namespace props

WIN_EXPORT class PeRopeLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT PeRopeLayer() : Layer() {}
  WIN_EXPORT ~PeRopeLayer() {}

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT bool supportBackwarding() const override { return false; }
  WIN_EXPORT void exportTo(
    nntrainer::Exporter &exporter,
    const ml::train::ExportMethods &method) const override {}
  WIN_EXPORT const std::string getType() const override {
    return PeRopeLayer::type;
  }
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override {
    auto remain = loadProperties(values, pe_rope_props);
    NNTR_THROW_IF(!remain.empty(), std::invalid_argument)
      << "[pe_rope] Unknown Layer Properties count "
      << std::to_string(values.size());
  }
  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "pe_rope";

private:
  std::tuple<props::NumPrefixTokens, props::NumHeadsRope, props::HeadDimRope,
             nntrainer::props::SkipPrefill>
    pe_rope_props;
  bool skip_prefill = false;

  // Apply rotation to q/k rows [from, to). Reads input0, sin, cos; writes out.
  void runRotate(nntrainer::RunLayerContext &context, unsigned int from,
                 unsigned int to);
};

} // namespace causallm

#endif /* __PE_ROPE_LAYER_H__ */
