// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   unittest_pe_rope.cpp
 * @date   18 Aug 2026
 * @brief  CausalLM pe_rope (PE-Lang static-table interleaved RoPE) tests.
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <layer_context.h>
#include <pe_rope.h>
#include <var_grad.h>
#include <weight.h>

#include <gtest/gtest.h>

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
    "pe_rope", true, 0.0f, false, 1.0f, nullptr, false,
    makeWeightView(weights), makeVarGradView(inputs), makeVarGradView(outputs),
    makeVarGradView(tensors));
}

nntrainer::TensorDim makeDim(unsigned int batch, unsigned int height,
                             unsigned int width) {
  return nntrainer::TensorDim(
    {batch, 1, height, width, nntrainer::Tformat::NCHW,
     nntrainer::TensorDim::DataType::FP32});
}

/**
 * @brief Fixture wiring one PeRopeLayer up with hand-computable q/k, sin, cos
 *        tensors: 1 head, head_dim=4 (half_head=2), num_prefix_tokens=1
 *        (CLS row 0). Patch row 1 uses sin/cos duplicated per interleaved
 *        pair -- sin=[0,0,1,1], cos=[1,1,0,0] -- so pair0 is a 0-degree
 *        rotation (identity) and pair1 is a 90-degree rotation
 *        ((qe,qo) -> (-qo,qe)), matching runRotate's
 *        out[even]=qe*c-qo*s, out[odd]=qo*c+qe*s.
 */
class PeRopeTest : public ::testing::Test {
protected:
  void SetUp() override {
    layer.setProperty(
      {"num_prefix_tokens=1", "num_heads=1", "head_dim=4"});

    nntrainer::InitLayerContext init_context(
      {makeDim(1, 2, 4), makeDim(1, 1, 4), makeDim(1, 1, 4)}, {true}, false,
      "pe_rope", "", 0.0f, {"NCHW", "FP32", "FP32"});
    ASSERT_NO_THROW(layer.finalize(init_context));
    ASSERT_EQ(init_context.getOutSpecs().size(), 1u);
    EXPECT_EQ(init_context.getOutSpecs()[0].variable_spec.dim, makeDim(1, 2, 4));

    inputs.emplace_back(makeDim(1, 2, 4), nntrainer::Initializer::NONE, true,
                        true, "qk");
    inputs.emplace_back(makeDim(1, 1, 4), nntrainer::Initializer::NONE, true,
                        true, "sin");
    inputs.emplace_back(makeDim(1, 1, 4), nntrainer::Initializer::NONE, true,
                        true, "cos");
    outputs.emplace_back(makeDim(1, 2, 4), nntrainer::Initializer::NONE, true,
                         true, "out");

    // row 0 = CLS (must pass through untouched), row 1 = the one patch row.
    float *qk = inputs[0].getVariableRef().getData<float>();
    qk[0] = 10.f;
    qk[1] = 20.f;
    qk[2] = 30.f;
    qk[3] = 40.f;
    qk[4] = 1.f;
    qk[5] = 2.f;
    qk[6] = 3.f;
    qk[7] = 4.f;

    float *sin = inputs[1].getVariableRef().getData<float>();
    sin[0] = 0.f;
    sin[1] = 0.f;
    sin[2] = 1.f;
    sin[3] = 1.f;

    float *cos = inputs[2].getVariableRef().getData<float>();
    cos[0] = 1.f;
    cos[1] = 1.f;
    cos[2] = 0.f;
    cos[3] = 0.f;
  }

  causallm::PeRopeLayer layer;
  std::vector<nntrainer::Weight> weights;
  std::vector<nntrainer::Var_Grad> inputs;
  std::vector<nntrainer::Var_Grad> outputs;
  std::vector<nntrainer::Var_Grad> tensors;
};

} // namespace

TEST_F(PeRopeTest, forwardingSkipsClsAndRotatesPatchRowInterleaved) {
  auto run_context = makeRunContext(weights, inputs, outputs, tensors);
  layer.forwarding(run_context, false);

  const float *out = run_context.getOutput(0).getData<float>();
  // Row 0 (CLS): identity, regardless of sin/cos.
  EXPECT_FLOAT_EQ(out[0], 10.f);
  EXPECT_FLOAT_EQ(out[1], 20.f);
  EXPECT_FLOAT_EQ(out[2], 30.f);
  EXPECT_FLOAT_EQ(out[3], 40.f);
  // Row 1 pair0 (0deg): (1,2) -> (1,2).
  EXPECT_FLOAT_EQ(out[4], 1.f);
  EXPECT_FLOAT_EQ(out[5], 2.f);
  // Row 1 pair1 (90deg): (3,4) -> (-4,3).
  EXPECT_FLOAT_EQ(out[6], -4.f);
  EXPECT_FLOAT_EQ(out[7], 3.f);
}

TEST_F(PeRopeTest, incrementalForwardingOnlyTouchesRequestedRows) {
  float *out_data = outputs[0].getVariableRef().getData<float>();
  for (int i = 0; i < 8; ++i)
    out_data[i] = -999.f;

  auto run_context = makeRunContext(weights, inputs, outputs, tensors);
  // Only recompute row 1 (the patch row); row 0 must stay untouched.
  layer.incremental_forwarding(run_context, 1, 2, false);

  const float *out = run_context.getOutput(0).getData<float>();
  EXPECT_FLOAT_EQ(out[0], -999.f);
  EXPECT_FLOAT_EQ(out[1], -999.f);
  EXPECT_FLOAT_EQ(out[2], -999.f);
  EXPECT_FLOAT_EQ(out[3], -999.f);
  EXPECT_FLOAT_EQ(out[4], 1.f);
  EXPECT_FLOAT_EQ(out[5], 2.f);
  EXPECT_FLOAT_EQ(out[6], -4.f);
  EXPECT_FLOAT_EQ(out[7], 3.f);
}

TEST(PeRopeMultiBatch, forwardingIsIndependentPerBatchRow) {
  // Regression guard for the row-parallel ThreadManager rewrite: batch=2,
  // seq=3 (1 CLS + 2 patch rows) exercises the combined (batch*row) index ->
  // (b, r) unflattening in runRotate, not just a single row.
  causallm::PeRopeLayer layer;
  layer.setProperty({"num_prefix_tokens=1", "num_heads=1", "head_dim=2"});

  nntrainer::InitLayerContext init_context(
    {makeDim(2, 3, 2), makeDim(1, 2, 2), makeDim(1, 2, 2)}, {true}, false,
    "pe_rope", "", 0.0f, {"NCHW", "FP32", "FP32"});
  ASSERT_NO_THROW(layer.finalize(init_context));

  std::vector<nntrainer::Weight> weights;
  std::vector<nntrainer::Var_Grad> inputs;
  std::vector<nntrainer::Var_Grad> outputs;
  std::vector<nntrainer::Var_Grad> tensors;

  inputs.emplace_back(makeDim(2, 3, 2), nntrainer::Initializer::NONE, true,
                      true, "qk");
  inputs.emplace_back(makeDim(1, 2, 2), nntrainer::Initializer::NONE, true,
                      true, "sin");
  inputs.emplace_back(makeDim(1, 2, 2), nntrainer::Initializer::NONE, true,
                      true, "cos");
  outputs.emplace_back(makeDim(2, 3, 2), nntrainer::Initializer::NONE, true,
                       true, "out");

  // 90-degree rotation table for both patch rows: sin=1, cos=0 everywhere.
  float *sin = inputs[1].getVariableRef().getData<float>();
  float *cos = inputs[2].getVariableRef().getData<float>();
  for (int i = 0; i < 4; ++i) {
    sin[i] = 1.f;
    cos[i] = 0.f;
  }

  // Distinct, easily-traceable values per batch so a cross-batch index bug
  // (writing batch 0's result into batch 1's slot or vice versa) shows up.
  float *qk = inputs[0].getVariableRef().getData<float>();
  // batch 0: CLS=(100,200); patch0=(1,2); patch1=(3,4)
  qk[0] = 100.f;
  qk[1] = 200.f;
  qk[2] = 1.f;
  qk[3] = 2.f;
  qk[4] = 3.f;
  qk[5] = 4.f;
  // batch 1: CLS=(-100,-200); patch0=(5,6); patch1=(7,8)
  qk[6] = -100.f;
  qk[7] = -200.f;
  qk[8] = 5.f;
  qk[9] = 6.f;
  qk[10] = 7.f;
  qk[11] = 8.f;

  auto run_context = makeRunContext(weights, inputs, outputs, tensors);
  layer.forwarding(run_context, false);

  const float *out = run_context.getOutput(0).getData<float>();
  // batch 0
  EXPECT_FLOAT_EQ(out[0], 100.f);
  EXPECT_FLOAT_EQ(out[1], 200.f);
  EXPECT_FLOAT_EQ(out[2], -2.f); // (1,2) -> (-2,1)
  EXPECT_FLOAT_EQ(out[3], 1.f);
  EXPECT_FLOAT_EQ(out[4], -4.f); // (3,4) -> (-4,3)
  EXPECT_FLOAT_EQ(out[5], 3.f);
  // batch 1
  EXPECT_FLOAT_EQ(out[6], -100.f);
  EXPECT_FLOAT_EQ(out[7], -200.f);
  EXPECT_FLOAT_EQ(out[8], -6.f); // (5,6) -> (-6,5)
  EXPECT_FLOAT_EQ(out[9], 5.f);
  EXPECT_FLOAT_EQ(out[10], -8.f); // (7,8) -> (-8,7)
  EXPECT_FLOAT_EQ(out[11], 7.f);
}
