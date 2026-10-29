// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causallm_gemma4_assistant_reference.cpp
 * @date   29 Oct 2026
 * @brief  Gemma 4 assistant (MTP drafter) against transformers: two chained
 *         drafting steps over shared target K/V, in FP32 and after
 *         nntr_quantize's CPU recipe (QS4CX FC, Q4_0 head).
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Fixture: generators/generate_gemma4_assistant_reference.py. The test skips
 * when it is absent; the quantized case also when NNTR_QUANTIZE_BIN is unset.
 */

#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#include <gemma4_assistant_causallm.h>

#include <cstdlib>
#include <filesystem>

namespace {

namespace fs = std::filesystem;
using causallm::json;

/**
 * @brief Run the fixture's two drafting steps on the pack in @a dir
 */
void runSteps(const fs::path &dir, float atol, float cosine_min) {
  const fs::path fixture =
    causallm_test::findFixtureDir("gemma4_assistant_tiny");
  if (!fs::exists(fixture / "reference.json") ||
      !fs::exists(fixture / "nntr_gemma4_assistant_fp32.bin"))
    GTEST_SKIP() << "run generate_gemma4_assistant_reference.py";
  json ref = causallm::LoadJsonFile((fixture / "reference.json").string());
  json cfg = causallm::LoadJsonFile((dir / "config.json").string());
  json gen = causallm::LoadJsonFile((dir / "generation_config.json").string());
  json nntr = causallm::LoadJsonFile((dir / "nntr_config.json").string());

  causallm::Gemma4AssistantCausalLM model(cfg, gen, nntr);
  model.initialize();
  model.load_weight(
    (dir / nntr["model_file_name"].get<std::string>()).string());

  const auto ks = ref["k_sliding"].get<std::vector<float>>();
  const auto vs = ref["v_sliding"].get<std::vector<float>>();
  const auto kf = ref["k_full"].get<std::vector<float>>();
  const auto vf = ref["v_full"].get<std::vector<float>>();
  model.setSharedKV(ks.data(), vs.data(), ref["sliding_rows"].get<unsigned>(),
                    kf.data(), vf.data(), ref["full_rows"].get<unsigned>(),
                    ref["pos"].get<unsigned>());
  for (const auto &step : ref["steps"]) {
    const auto in = step["input"].get<std::vector<float>>();
    const auto logits = step["logits"].get<std::vector<float>>();
    const auto hidden = step["hidden"].get<std::vector<float>>();
    const float *out = model.draftStep(in.data());
    causallm_test::expectLogitsNear(
      std::vector<float>(out, out + logits.size()), logits, atol);
    causallm_test::expectEmbeddingNear(
      std::vector<float>(out + logits.size(),
                         out + logits.size() + hidden.size()),
      hidden, atol, cosine_min);
  }
}

} // namespace

/**
 * @brief The FP32 pack reproduces transformers' drafting steps
 */
TEST(Gemma4AssistantTest, FP32MatchesHFReference) {
  runSteps(causallm_test::findFixtureDir("gemma4_assistant_tiny"), 1e-4f,
           0.9999f);
}

/**
 * @brief nntr_quantize's CPU drafter recipe stays close to the reference
 */
TEST(Gemma4AssistantTest, QS4CXFcQ40HeadCloseToHFReference) {
  const char *bin = std::getenv("NNTR_QUANTIZE_BIN");
  if (bin == nullptr)
    GTEST_SKIP() << "NNTR_QUANTIZE_BIN not set";
  const fs::path src = causallm_test::findFixtureDir("gemma4_assistant_tiny");
  const fs::path out = fs::temp_directory_path() / "gemma4_assistant_qs4cx";
  fs::remove_all(out);
  const std::string cmd = std::string(bin) + " " + src.string() + " -o " +
                          out.string() +
                          " --fc_dtype QS4CX --lmhead_dtype Q4_0 2>&1";
  ASSERT_EQ(std::system(cmd.c_str()), 0) << cmd;
  // int4 noise on a 32-wide random model: the drafted ids must hold, the
  // values only roughly
  runSteps(out, 1.0f, 0.98f);
  fs::remove_all(out);
}
