// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 * @file   unittest_causallm_gemma4_moe_reference.cpp
 * @date   01 October 2026
 * @brief  Tiny Gemma-4 MoE model against the HF reference fixture
 *         (causallm_reference/gemma4_moe_tiny, made by
 *         generators/generate_gemma4_moe_reference.py): the router, the
 *         experts beside the dense MLP, K == V on the full-attention layer
 *         and the weight file order, all at once (doc 55).
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#include <gemma4_causallm.h>

#include <memory>

namespace {

class ReferenceGemma4Moe final
  : public causallm_test::CausalLMTestAdapter<causallm::Gemma4CausalLM> {
public:
  ReferenceGemma4Moe(causallm::json &cfg, causallm::json &generation_cfg,
                     causallm::json &nntr_cfg) :
    causallm::Transformer(causallm::Gemma4Transformer::sanitizeConfig(cfg),
                          causallm::Gemma4Transformer::sanitizeGenerationConfig(
                            generation_cfg, cfg),
                          nntr_cfg, causallm::ModelType::CAUSALLM),
    causallm_test::CausalLMTestAdapter<causallm::Gemma4CausalLM>(
      cfg, generation_cfg, nntr_cfg) {}
};

causallm_test::DifferentialModel gemma4MoeModel() {
  return {
    "gemma4_moe_tiny",
    [](causallm::json &cfg, causallm::json &gen_cfg, causallm::json &nntr_cfg) {
      return std::make_unique<ReferenceGemma4Moe>(cfg, gen_cfg, nntr_cfg);
    },
  };
}

TEST(Gemma4MoeDifferentialTest, FP32MatchesHFReference) {
  causallm_test::runFp32DifferentialChecks(gemma4MoeModel());
}

} // namespace
