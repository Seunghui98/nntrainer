// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 * @file   unittest_causallm_gemma4_moe.cpp
 * @date   29 September 2026
 * @brief  Tiny Gemma-4 MoE (enable_moe_block) CausalLM unit tests: the MoE
 *         block beside the dense MLP, the softmax_scale router, no per-layer
 *         input, and K == V on the full-attention layer (doc 55).
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#include <gemma4_causallm.h>
#include <layer.h>
#include <layer_context.h>

#include <iostream>
#include <map>

namespace {

constexpr int tiny_num_layers = 2;

/**
 * @brief Tiny Gemma4 MoE CausalLM adapter (same constructor shape as the
 *        dense Gemma4 tiny test: text_config is flattened first)
 */
class TinyGemma4MoeCausalLM final
  : public causallm_test::CausalLMTestAdapter<causallm::Gemma4CausalLM> {
public:
  TinyGemma4MoeCausalLM(causallm::json &cfg, causallm::json &generation_cfg,
                        causallm::json &nntr_cfg) :
    causallm::Transformer(sanitizeConfig(cfg),
                          sanitizeGenerationConfig(generation_cfg, cfg),
                          nntr_cfg, causallm::ModelType::CAUSALLM),
    causallm_test::CausalLMTestAdapter<causallm::Gemma4CausalLM>(
      cfg, generation_cfg, nntr_cfg) {}
};

/**
 * @brief Deterministic weights. Everything outside the MoE layer is the
 * dense Gemma4 tiny setup (FC zero, norms one, embedding rows 1 and 4 set),
 * so the residual stream carries the embedding through. Inside the MoE
 * layer every weight is NON-zero except the [E] per-expert scale: under
 * router_type softmax_scale that scale multiplies every routed weight, so
 * the experts' output is exactly zero and the logits equal the dense case.
 * A layer that read that vector as LFM2's selection bias, or ignored it,
 * would let the experts' non-zero output through and fail the logits check.
 */
void setupGemma4MoeDeterministicWeights(TinyGemma4MoeCausalLM &model) {
  model.forEachLayer(
    [](ml::train::Layer &layer, nntrainer::RunLayerContext &context, void *) {
      if (layer.getName() == "output_of_causallm")
        return;

      for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
        auto &weight = context.getWeight(i);
        if (weight.getDataType() != ml::train::TensorDim::DataType::FP32)
          continue;

        if (layer.getType() == "lfm2_moe") {
          const bool is_scale =
            weight.getName().find("expert_bias") != std::string::npos;
          weight.setValue(is_scale ? 0.0f : 0.5f);
          continue;
        }

        weight.setValue(0.0f);
        if (layer.getType() == "rms_norm" ||
            layer.getType() == "reshaped_rms_norm") {
          weight.setValue(1.0f);
        } else if (layer.getName() == "embedding0") {
          weight.setValue(0, 0, 1, 0, 1.0f);
          weight.setValue(0, 0, 4, 0, 2.0f);
        } else if (layer.getName().find("_layer_scalar") != std::string::npos) {
          weight.setValue(1.0f);
        }
      }
    });
}

/** @brief gemma-4-26B-A4B's text_config shape, made tiny */
causallm::json makeTinyGemma4MoeConfig() {
  return {
    {"architectures", {"Gemma4ForConditionalGeneration"}},
    {"bos_token_id", 0},
    {"eos_token_id", {31}},
    {"text_config",
     {
       {"attention_k_eq_v", true},
       {"enable_moe_block", true},
       {"global_head_dim", 8},
       {"head_dim", 8},
       {"hidden_activation", "gelu_pytorch_tanh"},
       {"hidden_size", 64},
       {"hidden_size_per_layer_input", 0},
       {"intermediate_size", 64},
       {"layer_types", {"sliding_attention", "full_attention"}},
       {"max_position_embeddings", 8},
       {"moe_intermediate_size", 64},
       {"num_attention_heads", 8},
       {"num_experts", 4},
       {"num_global_key_value_heads", 2},
       {"num_hidden_layers", tiny_num_layers},
       {"num_key_value_heads", 4},
       {"num_kv_shared_layers", 0},
       {"rms_norm_eps", 1e-6},
       {"rope_theta", 1000000},
       {"sliding_window", 4},
       {"tie_word_embeddings", true},
       {"top_k_experts", 2},
       {"use_bidirectional_attention", "vision"},
       {"vocab_size", 32},
       {"vocab_size_per_layer_input", 32},
     }},
  };
}

/** @brief FP32 only: the router gate is [64, 4] and cannot be Q4_0 */
std::map<std::string, ml::train::TensorDim::DataType>
makeGemma4MoeLayerDtypeMap(const causallm_test::TinyCausalLMDataType &) {
  return {};
}

/** @brief The dense tiny Gemma4 logits: logit[1]=8, logit[4]=16 */
std::vector<float> makeExpectedGemma4MoeLogits() {
  std::vector<float> logits(32, 0.0f);
  logits[1] = 8.0f;
  logits[4] = 16.0f;
  return logits;
}

causallm_test::TinyCausalLMCase
makeGemma4MoeCase(const causallm_test::TinyCausalLMDataType &data_type) {
  return {
    "Gemma4Moe_" + data_type.name,
    data_type,
    {"hello tok4", makeExpectedGemma4MoeLogits(), 1e-4f},
    makeTinyGemma4MoeConfig,
    makeGemma4MoeLayerDtypeMap,
    [](causallm::json &cfg, causallm::json &generation_cfg,
       causallm::json &nntr_cfg) {
      return std::make_unique<TinyGemma4MoeCausalLM>(cfg, generation_cfg,
                                                     nntr_cfg);
    },
    [](causallm_test::TinyCausalLMRunner &runner) {
      setupGemma4MoeDeterministicWeights(
        static_cast<TinyGemma4MoeCausalLM &>(runner));
    },
  };
}

class Gemma4MoeTinyModelTest
  : public ::testing::TestWithParam<causallm_test::TinyCausalLMCase> {
protected:
  causallm_test::TinyCausalLMFiles makeFiles() const {
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string suite_name = "Gemma4MoeTinyModelTest";
    std::string test_name = "Unknown";

    if (info != nullptr) {
      suite_name = info->test_suite_name();
      test_name = info->name();
    }

    return causallm_test::makeTinyCausalLMFiles(suite_name, test_name,
                                                GetParam().name);
  }
};

/**
 * @brief The weight file order. The .bin is written in the compiled graph's
 * order, and res/gemma4/weight_converter.py and quantize_stream.cpp's
 * writeGemma4Moe reproduce this list by hand; a graph change that reorders
 * a weight-bearing layer has to change all three, and this is what says so.
 */
TEST_P(Gemma4MoeTinyModelTest, WeightBearingLayerOrderMatchesConverter) {
  const auto files = makeFiles();
  auto config =
    causallm_test::makeTinyCausalLMConfig(GetParam(), files.tokenizer_path);
  auto model =
    GetParam().create_model(config.model, config.generation, config.nntrainer);
  model->initializeModel();

  std::vector<std::string> order;
  static_cast<TinyGemma4MoeCausalLM &>(*model).forEachLayer(
    [&](ml::train::Layer &layer, nntrainer::RunLayerContext &context, void *) {
      if (context.getNumWeights() != 0)
        order.push_back(layer.getName());
    });

  std::vector<std::string> expected = {"embedding0"};
  for (int i = 0; i < tiny_num_layers; ++i) {
    const std::string p = "layer" + std::to_string(i);
    // _qkv holds attention_norm, q, q_norm, k, k_norm (and v on a sliding
    // layer; layer 1 is full_attention: K == V); _ffn holds pre_ffn_norm,
    // gate, up, down, post_ffn_norm_1; _sparse_moe holds pre_ffn_norm_2,
    // router_norm, the router and the experts, post_ffn_norm_2 -- each in
    // the file's order, so the tensor order is unchanged by the fusion.
    std::vector<std::string> block = {p + "_qkv"};
    for (const char *n : {"_attention_out", "_post_attention_norm", "_ffn",
                          "_sparse_moe", "_post_ffn_norm", "_layer_scalar"})
      block.push_back(p + n);
    expected.insert(expected.end(), block.begin(), block.end());
  }
  expected.push_back("output_norm");
  // tied lm head: the binary save re-emits the shared embedding after it
  expected.push_back("output_of_causallm");

  if (order != expected)
    for (const auto &n : order)
      std::cout << "  weight-bearing: " << n << '\n';
  EXPECT_EQ(order, expected);
}

TEST_P(Gemma4MoeTinyModelTest, GreedyGenerationSelectsArgmaxLogit) {
  const auto files = makeFiles();
  auto config =
    causallm_test::makeTinyCausalLMConfig(GetParam(), files.tokenizer_path);
  auto model =
    GetParam().create_model(config.model, config.generation, config.nntrainer);

  causallm_test::expectGreedyGenerationSelectsArgmax(*model);
}

TEST_P(Gemma4MoeTinyModelTest, WeightRoundTripProducesSameLogits) {
  const auto files = makeFiles();
  causallm_test::expectWeightRoundTripProducesSameLogits(GetParam(), files);
}

TEST_P(Gemma4MoeTinyModelTest, PromptProducesExpectedLogits) {
  const auto files = makeFiles();
  causallm_test::expectPromptProducesExpectedLogits(GetParam(), files);
}

INSTANTIATE_TEST_SUITE_P(
  Gemma4Moe, Gemma4MoeTinyModelTest,
  ::testing::Values(makeGemma4MoeCase(causallm_test::makeTinyFp32DataType())),
  [](const ::testing::TestParamInfo<causallm_test::TinyCausalLMCase> &info) {
    return info.param.name;
  });

} // namespace
