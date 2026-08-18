// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   unittest_layer_scale.cpp
 * @date   18 Aug 2026
 * @brief  CausalLM layer_scale (PE-Lang fused residual + gamma-scale) tests.
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <layer_context.h>
#include <layer_scale.h>
#include <tensor.h>
#include <var_grad.h>
#include <weight.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

namespace {

std::vector<nntrainer::Weight *>
makeWeightView(std::vector<nntrainer::Weight> &weights) {
  std::vector<nntrainer::Weight *> view;
  view.reserve(weights.size());
  for (auto &weight : weights)
    view.push_back(&weight);
  return view;
}

std::vector<nntrainer::Var_Grad *>
makeVarGradView(std::vector<nntrainer::Var_Grad> &vars) {
  std::vector<nntrainer::Var_Grad *> view;
  view.reserve(vars.size());
  for (auto &var : vars)
    view.push_back(&var);
  return view;
}

nntrainer::RunLayerContext
makeRunContext(std::vector<nntrainer::Weight> &weights,
               std::vector<nntrainer::Var_Grad> &inputs,
               std::vector<nntrainer::Var_Grad> &outputs,
               std::vector<nntrainer::Var_Grad> &tensors) {
  return nntrainer::RunLayerContext(
    "layer_scale", true, 0.0f, false, 1.0f, nullptr, false,
    makeWeightView(weights), makeVarGradView(inputs), makeVarGradView(outputs),
    makeVarGradView(tensors));
}

nntrainer::TensorDim makeDim(unsigned int batch, unsigned int height,
                             unsigned int width) {
  return nntrainer::TensorDim(
    {batch, 1, height, width, nntrainer::Tformat::NCHW,
     nntrainer::TensorDim::DataType::FP32});
}

nntrainer::Weight makeGammaWeight(const std::vector<float> &values) {
  nntrainer::TensorDim dim(
    {1, 1, 1, static_cast<unsigned int>(values.size()),
     nntrainer::Tformat::NCHW, nntrainer::TensorDim::DataType::FP32});
  nntrainer::Tensor gamma_tensor(dim, true);
  std::copy(values.begin(), values.end(), gamma_tensor.getData<float>());
  return nntrainer::Weight(gamma_tensor, nntrainer::Tensor(),
                           nntrainer::Tensor(), "gamma");
}

} // namespace

TEST(LayerScaleTest, forwardingComputesResidualPlusGammaTimesInput) {
  // out = residual + gamma * input, C=3.
  causallm::LayerScaleLayer layer;

  nntrainer::InitLayerContext init_context(
    {makeDim(1, 1, 3), makeDim(1, 1, 3)}, {true}, false, "layer_scale", "",
    0.0f, {"NCHW", "FP32", "FP32"});
  ASSERT_NO_THROW(layer.finalize(init_context));
  ASSERT_EQ(init_context.getOutSpecs().size(), 1u);
  EXPECT_EQ(init_context.getOutSpecs()[0].variable_spec.dim, makeDim(1, 1, 3));
  ASSERT_EQ(init_context.getNumWeights(), 1u);

  std::vector<nntrainer::Weight> weights;
  weights.push_back(makeGammaWeight({2.f, 3.f, 4.f}));

  std::vector<nntrainer::Var_Grad> inputs;
  std::vector<nntrainer::Var_Grad> outputs;
  std::vector<nntrainer::Var_Grad> tensors;

  inputs.emplace_back(makeDim(1, 1, 3), nntrainer::Initializer::NONE, true,
                      true, "input");
  inputs.emplace_back(makeDim(1, 1, 3), nntrainer::Initializer::NONE, true,
                      true, "residual");
  outputs.emplace_back(makeDim(1, 1, 3), nntrainer::Initializer::NONE, true,
                       true, "output");

  float *in_data = inputs[0].getVariableRef().getData<float>();
  in_data[0] = 1.f;
  in_data[1] = 1.f;
  in_data[2] = 1.f;
  float *res_data = inputs[1].getVariableRef().getData<float>();
  res_data[0] = 10.f;
  res_data[1] = 20.f;
  res_data[2] = 30.f;

  auto run_context = makeRunContext(weights, inputs, outputs, tensors);
  layer.forwarding(run_context, false);

  const float *out = run_context.getOutput(0).getData<float>();
  EXPECT_FLOAT_EQ(out[0], 12.f); // 10 + 2*1
  EXPECT_FLOAT_EQ(out[1], 23.f); // 20 + 3*1
  EXPECT_FLOAT_EQ(out[2], 34.f); // 30 + 4*1
}

TEST(LayerScaleTest, incrementalForwardingOnlyTouchesRequestedRows) {
  // SEQ=2 rows, C=2; recompute only row 1 and check row 0 is untouched.
  causallm::LayerScaleLayer layer;

  nntrainer::InitLayerContext init_context(
    {makeDim(1, 2, 2), makeDim(1, 2, 2)}, {true}, false, "layer_scale", "",
    0.0f, {"NCHW", "FP32", "FP32"});
  ASSERT_NO_THROW(layer.finalize(init_context));

  std::vector<nntrainer::Weight> weights;
  weights.push_back(makeGammaWeight({1.f, 2.f}));

  std::vector<nntrainer::Var_Grad> inputs;
  std::vector<nntrainer::Var_Grad> outputs;
  std::vector<nntrainer::Var_Grad> tensors;

  inputs.emplace_back(makeDim(1, 2, 2), nntrainer::Initializer::NONE, true,
                      true, "input");
  inputs.emplace_back(makeDim(1, 2, 2), nntrainer::Initializer::NONE, true,
                      true, "residual");
  outputs.emplace_back(makeDim(1, 2, 2), nntrainer::Initializer::NONE, true,
                       true, "output");

  float *in_data = inputs[0].getVariableRef().getData<float>();
  in_data[0] = 100.f;
  in_data[1] = 100.f; // row 0 -- must not affect the output
  in_data[2] = 5.f;
  in_data[3] = 6.f; // row 1
  float *res_data = inputs[1].getVariableRef().getData<float>();
  res_data[0] = 0.f;
  res_data[1] = 0.f;
  res_data[2] = 1.f;
  res_data[3] = 2.f;

  float *out_data = outputs[0].getVariableRef().getData<float>();
  out_data[0] = -1.f;
  out_data[1] = -2.f;
  out_data[2] = -3.f;
  out_data[3] = -4.f;

  auto run_context = makeRunContext(weights, inputs, outputs, tensors);
  layer.incremental_forwarding(run_context, 1, 2, false);

  const float *out = run_context.getOutput(0).getData<float>();
  EXPECT_FLOAT_EQ(out[0], -1.f); // row 0 untouched
  EXPECT_FLOAT_EQ(out[1], -2.f);
  EXPECT_FLOAT_EQ(out[2], 6.f); // 1 + 1*5
  EXPECT_FLOAT_EQ(out[3], 14.f); // 2 + 2*6
}

TEST(LayerScaleMultiBatch, forwardingIsIndependentPerBatchRow) {
  // Regression guard for the row-parallel ThreadManager rewrite: batch=2,
  // seq=2 exercises the combined (batch*row) index -> (b, r) unflattening in
  // runScale, not just a single row.
  causallm::LayerScaleLayer layer;

  nntrainer::InitLayerContext init_context(
    {makeDim(2, 2, 2), makeDim(2, 2, 2)}, {true}, false, "layer_scale", "",
    0.0f, {"NCHW", "FP32", "FP32"});
  ASSERT_NO_THROW(layer.finalize(init_context));

  std::vector<nntrainer::Weight> weights;
  weights.push_back(makeGammaWeight({1.f, 1.f}));

  std::vector<nntrainer::Var_Grad> inputs;
  std::vector<nntrainer::Var_Grad> outputs;
  std::vector<nntrainer::Var_Grad> tensors;

  inputs.emplace_back(makeDim(2, 2, 2), nntrainer::Initializer::NONE, true,
                      true, "input");
  inputs.emplace_back(makeDim(2, 2, 2), nntrainer::Initializer::NONE, true,
                      true, "residual");
  outputs.emplace_back(makeDim(2, 2, 2), nntrainer::Initializer::NONE, true,
                       true, "output");

  // input is all zeros (gamma * 0 = 0), so out == residual; residual carries
  // distinct per-batch, per-row values so a cross-batch index bug shows up.
  float *res_data = inputs[1].getVariableRef().getData<float>();
  res_data[0] = 1.f;
  res_data[1] = 2.f; // batch 0, row 0
  res_data[2] = 3.f;
  res_data[3] = 4.f; // batch 0, row 1
  res_data[4] = -1.f;
  res_data[5] = -2.f; // batch 1, row 0
  res_data[6] = -3.f;
  res_data[7] = -4.f; // batch 1, row 1

  auto run_context = makeRunContext(weights, inputs, outputs, tensors);
  layer.forwarding(run_context, false);

  const float *out = run_context.getOutput(0).getData<float>();
  for (int i = 0; i < 8; ++i)
    EXPECT_FLOAT_EQ(out[i], (inputs[1].getVariableRef().getData<float>())[i]);
}
