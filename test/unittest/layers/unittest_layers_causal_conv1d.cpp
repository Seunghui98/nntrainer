// SPDX-License-Identifier: Apache-2.0
/**
* Copyright (C) 2026 Hyeong-Gwon Hong
*
* @file   unittest_layers_causal_conv1d.cpp
* @date   01 April 2026
* @brief  Causal Conv1D Layer Golden Test
* @see    https://github.com/nntrainer/nntrainer
* @author Hyeong-Gwon Hong
* @bug    No known bugs except for NYI items
*/

#include <layers_common_tests.h>
#include "causal_conv1d_layer.h"
#include <tuple>

#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace {

auto create_causal_conv1d_layer =
  [](const std::vector<nntrainer::props::Property> &properties)
  -> std::unique_ptr<nntrainer::Layer> {
  (void)properties;
  auto layer = std::make_unique<causallm::CausalConv1DLayer>();
  return layer;
};

auto semantic_causal_conv1d = LayerSemanticsParamType(
  create_causal_conv1d_layer,
  causallm::CausalConv1DLayer::type,
  {},
  0,
  false,
  1);

GTEST_PARAMETER_TEST(CausalConv1D, LayerSemantics,
                     ::testing::Values(semantic_causal_conv1d));

static constexpr int causal_conv1d_golden_option =
  LayerGoldenTestParamOptions::USE_INC_FORWARD |
  LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV;

auto causal_conv1d_sb_h1_w8 = LayerGoldenTestParamType(
  create_causal_conv1d_layer,
  {},
  "1:1:1:8",
  "causal_conv1d_sb_h1_w8.nnlayergolden",
  causal_conv1d_golden_option,
  "nchw",
  "fp16",
  "fp32");

auto causal_conv1d_sb_h2_w8 = LayerGoldenTestParamType(
  create_causal_conv1d_layer,
  {},
  "1:1:2:8",
  "causal_conv1d_sb_h2_w8.nnlayergolden",
  causal_conv1d_golden_option,
  "nchw",
  "fp16",
  "fp32");

auto causal_conv1d_sb_h4_w8 = LayerGoldenTestParamType(
  create_causal_conv1d_layer,
  {},
  "1:1:4:8",
  "causal_conv1d_sb_h4_w8.nnlayergolden",
  causal_conv1d_golden_option,
  "nchw",
  "fp16",
  "fp32");

auto causal_conv1d_mb_h4_w8 = LayerGoldenTestParamType(
  create_causal_conv1d_layer,
  {},
  "3:1:4:8",
  "causal_conv1d_mb_h4_w8.nnlayergolden",
  causal_conv1d_golden_option,
  "nchw",
  "fp16",
  "fp32");

auto causal_conv1d_sb_h16_w32 = LayerGoldenTestParamType(
  create_causal_conv1d_layer,
  {},
  "1:1:16:32",
  "causal_conv1d_sb_h16_w32.nnlayergolden",
  causal_conv1d_golden_option,
  "nchw",
  "fp16",
  "fp32");

GTEST_PARAMETER_TEST(
  CausalConv1D,
  LayerGoldenTest,
  ::testing::Values(
    causal_conv1d_sb_h1_w8,
    causal_conv1d_sb_h2_w8,
    causal_conv1d_sb_h4_w8,
    causal_conv1d_mb_h4_w8,
    causal_conv1d_sb_h16_w32));

} // namespace