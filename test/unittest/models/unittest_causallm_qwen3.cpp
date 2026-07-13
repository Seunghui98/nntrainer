// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causallm_qwen3.cpp
 * @date   15 May 2026
 * @brief  Tiny Qwen3 model unit tests
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#include <layer.h>
#include <layer_context.h>
#include <qwen3_causallm.h>
#include <qwen3_embedding.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

/**
 * @brief Helpers for tiny Qwen3 model tests
 */
namespace {

/**
 * @brief Tiny Qwen3 CausalLM adapter (inference shared by CausalLMTestAdapter)
 */
class TinyQwen3CausalLM final : public causallm::Qwen3CausalLM,
                                public causallm_test::TinyCausalLMRunner {
public:
  struct QuantizedWeightInfo {
    ml::train::TensorDim::DataType dtype;
    nntrainer::QScheme qscheme;
    size_t scale_size;
    std::vector<float> scales;
    std::vector<int32_t> zp_corr;
  };

  /**
   * @brief Construct a tiny Qwen3 CausalLM test adapter
   */
  TinyQwen3CausalLM(causallm::json &cfg, causallm::json &generation_cfg,
                    causallm::json &nntr_cfg) :
    causallm::Transformer(cfg, generation_cfg, nntr_cfg,
                          causallm::ModelType::CAUSALLM),
    causallm::Qwen3CausalLM(cfg, generation_cfg, nntr_cfg) {}

  void initializeModel() override { initialize(); }

  void saveWeight(const std::string &path) override { save_weight(path); }

  void saveWeightWithDtype(
    const std::string &path,
    const std::map<std::string, ml::train::TensorDim::DataType>
      &layer_dtype_map) override {
    save_weight(path, ml::train::TensorDim::DataType::NONE, layer_dtype_map);
  }

  void loadWeight(const std::string &path) override { load_weight(path); }

  void setDeterministicWeights() {
    auto set_weights = [](ml::train::Layer &layer,
                          nntrainer::RunLayerContext &context, void *) {
      if (layer.getName() == "output_of_causallm")
        return;

      for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
        auto &weight = context.getWeight(i);
        if (weight.getDataType() != ml::train::TensorDim::DataType::FP32)
          continue;

        weight.setValue(0.0f);
        if (layer.getType() == "rms_norm" ||
            layer.getType() == "reshaped_rms_norm") {
          weight.setValue(1.0f);
        } else if (layer.getName() == "embedding0") {
          weight.setValue(0.0f);
          weight.setValue(0, 0, 1, 0, 1.0f);
          weight.setValue(0, 0, 4, 0, 2.0f);
        }
      }
    };

    model->forEachLayer(set_weights, nullptr);
  }

  void setQuantizedRoundTripWeights() {
    auto set_weights = [](ml::train::Layer &layer,
                          nntrainer::RunLayerContext &context, void *) {
      for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
        auto &weight = context.getWeight(i);
        if (weight.getDataType() != ml::train::TensorDim::DataType::FP32)
          continue;

        if (layer.getType() == "rms_norm" ||
            layer.getType() == "reshaped_rms_norm") {
          weight.setValue(1.0f);
          continue;
        }

        float *data = weight.getData<float>();
        for (unsigned int idx = 0; idx < weight.size(); ++idx) {
          const int pattern = static_cast<int>(idx % 13) - 6;
          data[idx] = static_cast<float>(pattern) / 8.0f;
        }
      }
    };

    model->forEachLayer(set_weights, nullptr);
  }

  void runPrompt(const std::string &prompt) override {
    run(prompt, false, "", "", false);
  }

  std::vector<float> prefillLogits(const std::string &prompt) override {
    allocateAndBindKVCache();

    auto encoded = tokenizer->Encode(prompt);
    if (encoded.empty())
      throw std::invalid_argument("tiny Qwen3 prompt encoded to no tokens");

    const unsigned int num_allow_str = MAX_SEQ_LEN - NUM_TO_GENERATE;
    const unsigned int init_len = static_cast<unsigned int>(
      std::min<size_t>(encoded.size(), num_allow_str));
    std::vector<float> input_sample(
      static_cast<size_t>(BATCH_SIZE) * MAX_SEQ_LEN, 0.0f);

    for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
      for (unsigned int i = 0; i < init_len; ++i) {
        const auto token_id = static_cast<unsigned int>(encoded[i]);
        input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN + i] =
          static_cast<float>(token_id);
        ids_history[static_cast<size_t>(b) * MAX_SEQ_LEN + i] = token_id;
      }
    }

    std::vector<std::pair<std::string, float *>> cache_inputs;
    cache_inputs.reserve(static_cast<size_t>(NUM_LAYERS) * 2);
    for (int i = 0; i < NUM_LAYERS; ++i) {
      cache_inputs.emplace_back(
        "cache_k_l" + std::to_string(i),
        reinterpret_cast<float *>(kv_cache.getKeyCache(i).getData()));
      cache_inputs.emplace_back(
        "cache_v_l" + std::to_string(i),
        reinterpret_cast<float *>(kv_cache.getValueCache(i).getData()));
    }

    std::sort(
      cache_inputs.begin(), cache_inputs.end(),
      [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

    std::vector<float *> input;
    input.reserve(1 + cache_inputs.size());
    input.push_back(input_sample.data());
    for (const auto &cache_input : cache_inputs)
      input.push_back(cache_input.second);

    std::vector<float *> label;
    setKVCachePosition(0);
    auto output = model->incremental_inference(BATCH_SIZE, input, label,
                                               init_len, 0, init_len, false);
    std::vector<float> logits(output[0], output[0] + NUM_VOCAB);
    for (auto &out : output)
      delete[] out;

    return logits;
  }

  std::string getOutputText(int batch_idx = 0) const override {
    return getOutput(batch_idx);
  }

  bool hasRun() const override { return causallm::CausalLM::hasRun(); }

  unsigned int tokenAt(size_t idx) const override { return ids_history[idx]; }

  std::vector<unsigned int>
  generateFromLogits(float *logits, bool do_sample, float repetition_penalty,
                     unsigned int *input_ids,
                     unsigned int num_input_ids) override {
    return generate(logits, do_sample, repetition_penalty, input_ids,
                    num_input_ids);
  }

  QuantizedWeightInfo inspectWeight(const std::string &layer_name) {
    struct InspectState {
      const std::string *target_name;
      QuantizedWeightInfo info{ml::train::TensorDim::DataType::NONE,
                               nntrainer::QScheme::PER_TENSOR_AFFINE,
                               0,
                               {},
                               {}};
      bool found = false;
    } state{&layer_name};

    auto inspect = [](ml::train::Layer &layer, nntrainer::RunLayerContext &ctx,
                      void *user_data) {
      auto *state = static_cast<InspectState *>(user_data);
      if (state->found || layer.getName() != *state->target_name ||
          ctx.getNumWeights() == 0)
        return;

      auto &weight = ctx.getWeight(0);
      state->info.dtype = weight.getDataType();
      state->info.qscheme = weight.q_scheme();
      state->info.scale_size = weight.scale_size();
      if (weight.getDataType() == ml::train::TensorDim::DataType::QINT8) {
        const size_t count = weight.getDim().width();
        state->info.scales.assign(weight.getScale<float>(),
                                  weight.getScale<float>() + count);
        state->info.zp_corr.assign(weight.getZpCorr<int32_t>(),
                                   weight.getZpCorr<int32_t>() + count);
      }
      state->found = true;
    };

    model->forEachLayer(inspect, &state);
    if (!state.found)
      throw std::invalid_argument("unknown layer: " + layer_name);

    return state.info;
  }
};

/**
 * @brief Populate deterministic tiny Qwen3 weights for golden token tests
 */
void setupQwen3DeterministicWeights(TinyQwen3CausalLM &model) {
  model.setDeterministicWeights();
}

/**
 * @brief Files generated for one tiny Qwen3 embedding test invocation
 */
struct TinyQwen3EmbeddingFiles {
  std::filesystem::path dir;            /**< Temporary test directory */
  std::filesystem::path tokenizer_path; /**< Tiny tokenizer.json path */
  std::filesystem::path modules_path;   /**< Tiny modules.json path */
  std::filesystem::path weight_path;    /**< Tiny model weight path */
};

/**
 * @brief Tiny Qwen3 Embedding adapter for model-level tests
 */
class TinyQwen3Embedding final : public causallm::Qwen3Embedding {
public:
  /**
   * @brief Construct a tiny Qwen3 Embedding test adapter
   */
  TinyQwen3Embedding(causallm::json &cfg, causallm::json &generation_cfg,
                     causallm::json &nntr_cfg) :
    causallm::Transformer(cfg, generation_cfg, nntr_cfg,
                          causallm::ModelType::EMBEDDING),
    causallm::Qwen3Embedding(cfg, generation_cfg, nntr_cfg) {}

  /**
   * @brief Initialize the tiny Qwen3 embedding model
   */
  void initializeModel() { initialize(); }

  /**
   * @brief Save tiny Qwen3 embedding weights with dtype conversion
   */
  void saveWeightWithDtype(
    const std::string &path,
    const std::map<std::string, ml::train::TensorDim::DataType>
      &layer_dtype_map) {
    save_weight(path, ml::train::TensorDim::DataType::NONE, layer_dtype_map);
  }

  /**
   * @brief Load tiny Qwen3 embedding model weights
   */
  void loadWeight(const std::string &path) { load_weight(path); }

  /**
   * @brief Set deterministic tiny Qwen3 embedding weights
   */
  void setDeterministicWeights() {
    auto set_weights = [](ml::train::Layer &layer,
                          nntrainer::RunLayerContext &context, void *) {
      for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
        auto &weight = context.getWeight(i);
        if (weight.getDataType() != ml::train::TensorDim::DataType::FP32)
          continue;

        weight.setValue(0.0f);
        if (layer.getType() == "rms_norm" ||
            layer.getType() == "reshaped_rms_norm") {
          weight.setValue(1.0f);
        } else if (layer.getName() == "embedding0") {
          weight.setValue(0, 0, 1, 0, 1.0f);
          weight.setValue(0, 0, 4, 0, 2.0f);
        }
      }
    };

    model->forEachLayer(set_weights, nullptr);
  }

  /**
   * @brief Encode one prompt and copy the embedding output
   */
  std::vector<float> encodePrompt(const std::string &prompt) {
    auto output = encode(prompt);
    std::vector<float> embedding(output[0], output[0] + BATCH_SIZE * DIM);
    for (auto &out : output)
      delete[] out;
    return embedding;
  }
};

/**
 * @brief Write a string into a file
 */
void writeTextFile(const std::filesystem::path &path,
                   const std::string &content) {
  std::ofstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("failed to open " + path.string());

  file << content;
  if (!file.good())
    throw std::runtime_error("failed to write " + path.string());
}

/**
 * @brief Make tiny SentenceTransformer module config files
 */
std::filesystem::path
writeTinyQwen3EmbeddingModules(const std::filesystem::path &dir) {
  auto modules_path = dir / "modules.json";
  auto pooling_dir = dir / "1_Pooling";
  std::filesystem::create_directories(pooling_dir);

  writeTextFile(modules_path, R"([
  {
    "idx": 0,
    "name": "0",
    "path": "",
    "type": "sentence_transformers.models.Transformer"
  },
  {
    "idx": 1,
    "name": "1",
    "path": "1_Pooling",
    "type": "sentence_transformers.models.Pooling"
  }
])");

  writeTextFile(pooling_dir / "config.json", R"({
  "word_embedding_dimension": 64,
  "pooling_mode_cls_token": false,
  "pooling_mode_mean_tokens": false,
  "pooling_mode_max_tokens": false,
  "pooling_mode_mean_sqrt_len_tokens": false,
  "pooling_mode_weightedmean_tokens": false,
  "pooling_mode_lasttoken": true,
  "include_prompt": true
})");

  return modules_path;
}

/**
 * @brief Make the tiny Qwen3 model config
 */
causallm::json makeTinyQwen3Config() {
  return {
    {"architectures", {"Qwen3ForCausalLM"}},
    {"bos_token_id", 0},
    {"eos_token_id", {31}},
    {"head_dim", 8},
    {"hidden_size", 64},
    {"intermediate_size", 64},
    {"is_causal", true},
    {"max_position_embeddings", 8},
    {"num_attention_heads", 8},
    {"num_hidden_layers", 1},
    {"num_key_value_heads", 4},
    {"rms_norm_eps", 1e-5},
    {"rope_theta", 10000},
    {"tie_word_embeddings", true},
    {"vocab_size", 32},
  };
}

/**
 * @brief Make the tiny Qwen3 embedding model config
 */
causallm::json makeTinyQwen3EmbeddingConfig() {
  auto cfg = makeTinyQwen3Config();
  cfg["vocab_size"] = 33;
  return cfg;
}

/**
 * @brief Make tiny Qwen3 embedding test files
 */
TinyQwen3EmbeddingFiles makeQwen3EmbeddingFiles() {
  const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
  std::string suite_name = "Qwen3EmbeddingTinyModelTest";
  std::string test_name = "Unknown";

  if (info != nullptr) {
    suite_name = info->test_suite_name();
    test_name = info->name();
  }

  auto files = causallm_test::makeTinyCausalLMFiles(suite_name, test_name,
                                                    "Qwen3Embedding_Q40_FP32");

  return {
    files.dir,
    files.tokenizer_path,
    writeTinyQwen3EmbeddingModules(files.dir),
    files.dir / "qwen3_embedding_tiny.bin",
  };
}

/**
 * @brief Make the tiny Qwen3 embedding nntrainer config
 */
causallm::json makeTinyQwen3EmbeddingNntrainerConfig(
  const TinyQwen3EmbeddingFiles &files,
  const causallm_test::TinyCausalLMDataType &data_type) {
  auto cfg =
    causallm_test::makeTinyNntrainerConfig(files.tokenizer_path, data_type);
  cfg["model_type"] = "Embedding";
  cfg["module_config_path"] = files.modules_path.string();
  return cfg;
}

/**
 * @brief Make the tiny Qwen3 layer dtype map
 */
std::map<std::string, ml::train::TensorDim::DataType>
makeQwen3LayerDtypeMap(const causallm_test::TinyCausalLMDataType &data_type) {
  std::map<std::string, ml::train::TensorDim::DataType> dtype_map;

  if (data_type.embedding_dtype != "FP32")
    dtype_map["embedding0"] =
      causallm_test::toTensorDataType(data_type.embedding_dtype);

  if (data_type.fc_layer_dtype != "FP32") {
    const auto dtype =
      causallm_test::toTensorDataType(data_type.fc_layer_dtype);
    dtype_map["layer0_wq"] = dtype;
    dtype_map["layer0_wk"] = dtype;
    dtype_map["layer0_wv"] = dtype;
    dtype_map["layer0_attention_out"] = dtype;
    dtype_map["layer0_ffn_up"] = dtype;
    dtype_map["layer0_ffn_gate"] = dtype;
    dtype_map["layer0_ffn_down"] = dtype;
  }

  if (data_type.lmhead_dtype != "FP32")
    dtype_map["output_of_causallm"] =
      causallm_test::toTensorDataType(data_type.lmhead_dtype);

  return dtype_map;
}

/**
 * @brief Make the tiny Qwen3 embedding Q4_0 layer dtype map
 */
std::map<std::string, ml::train::TensorDim::DataType>
makeQwen3EmbeddingQ40LayerDtypeMap() {
  std::map<std::string, ml::train::TensorDim::DataType> dtype_map;
  const auto dtype = ml::train::TensorDim::DataType::Q4_0;

  dtype_map["embedding0"] = dtype;
  dtype_map["layer0_wq"] = dtype;
  dtype_map["layer0_wk"] = dtype;
  dtype_map["layer0_wv"] = dtype;
  dtype_map["layer0_attention_out"] = dtype;
  dtype_map["layer0_ffn_up"] = dtype;
  dtype_map["layer0_ffn_gate"] = dtype;
  dtype_map["layer0_ffn_down"] = dtype;

  return dtype_map;
}

/**
 * @brief Create a loaded tiny Qwen3 embedding model
 */
std::unique_ptr<TinyQwen3Embedding>
makeLoadedQwen3Embedding(const TinyQwen3EmbeddingFiles &files) {
  const auto fp32_data_type = causallm_test::makeTinyFp32DataType();
  const auto q40_data_type = causallm_test::makeTinyQ40Fp32DataType();
  auto source_model_cfg = makeTinyQwen3EmbeddingConfig();
  auto source_generation_cfg = causallm_test::makeTinyGenerationConfig();
  auto source_nntr_cfg =
    makeTinyQwen3EmbeddingNntrainerConfig(files, fp32_data_type);

  TinyQwen3Embedding source(source_model_cfg, source_generation_cfg,
                            source_nntr_cfg);
  source.initializeModel();
  source.setDeterministicWeights();
  source.saveWeightWithDtype(files.weight_path.string(),
                             makeQwen3EmbeddingQ40LayerDtypeMap());

  auto loaded_model_cfg = makeTinyQwen3EmbeddingConfig();
  auto loaded_generation_cfg = causallm_test::makeTinyGenerationConfig();
  auto loaded_nntr_cfg =
    makeTinyQwen3EmbeddingNntrainerConfig(files, q40_data_type);
  auto loaded = std::make_unique<TinyQwen3Embedding>(
    loaded_model_cfg, loaded_generation_cfg, loaded_nntr_cfg);
  loaded->initializeModel();
  loaded->loadWeight(files.weight_path.string());

  return loaded;
}

/**
 * @brief Make the expected tiny Qwen3 prefill logits
 */
std::vector<float> makeExpectedQwen3Logits() {
  std::vector<float> logits(32, 0.0f);
  logits[1] = 7.99936008f;
  logits[4] = 15.99872017f;
  return logits;
}

/**
 * @brief Make a Qwen3 tiny CausalLM test case
 */
causallm_test::TinyCausalLMCase
makeQwen3Case(const causallm_test::TinyCausalLMDataType &data_type) {
  return {
    "Qwen3_" + data_type.name,
    data_type,
    {"hello tok4", makeExpectedQwen3Logits(),
     data_type.name == "FP32"       ? 1e-4f
     : data_type.name == "Q40_FP16" ? 2e-3f
                                    : 1e-3f},
    makeTinyQwen3Config,
    makeQwen3LayerDtypeMap,
    [](causallm::json &cfg, causallm::json &generation_cfg,
       causallm::json &nntr_cfg) {
      return std::make_unique<TinyQwen3CausalLM>(cfg, generation_cfg, nntr_cfg);
    },
    [](causallm_test::TinyCausalLMRunner &runner) {
      setupQwen3DeterministicWeights(static_cast<TinyQwen3CausalLM &>(runner));
    },
  };
}

/**
 * @brief Parameterized fixture for tiny CausalLM model cases
 */
class CausalLMTinyModelTest
  : public ::testing::TestWithParam<causallm_test::TinyCausalLMCase> {
protected:
  /**
   * @brief Make test files for the current parameterized case
   */
  causallm_test::TinyCausalLMFiles makeFiles() const {
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string suite_name = "CausalLMTinyModelTest";
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
 * @brief Test that greedy generation chooses the argmax logit
 */
TEST_P(CausalLMTinyModelTest, GreedyGenerationSelectsArgmaxLogit) {
  const auto files = makeFiles();
  auto config =
    causallm_test::makeTinyCausalLMConfig(GetParam(), files.tokenizer_path);
  auto model =
    GetParam().create_model(config.model, config.generation, config.nntrainer);

  causallm_test::expectGreedyGenerationSelectsArgmax(*model);
}

/**
 * @brief Test that a save/load round-trip preserves logits
 */
TEST_P(CausalLMTinyModelTest, WeightRoundTripProducesSameLogits) {
  const auto files = makeFiles();
  causallm_test::expectWeightRoundTripProducesSameLogits(GetParam(), files);
}

/**
 * @brief Test that a prompt produces the expected golden logits
 */
TEST_P(CausalLMTinyModelTest, PromptProducesExpectedLogits) {
  const auto files = makeFiles();
  causallm_test::expectPromptProducesExpectedLogits(GetParam(), files);
}

INSTANTIATE_TEST_SUITE_P(
  Qwen3, CausalLMTinyModelTest,
  ::testing::Values(makeQwen3Case(causallm_test::makeTinyFp32DataType()),
                    makeQwen3Case(causallm_test::makeTinyQ40Fp32DataType())),
  [](const ::testing::TestParamInfo<causallm_test::TinyCausalLMCase> &info) {
    return info.param.name;
  });

TEST(Qwen3Qint8WeightRoundTripTest, SaveLoadRestoresQuantMetadata) {
  const auto files = causallm_test::makeTinyCausalLMFiles(
    "Qwen3Qint8WeightRoundTripTest", "SaveLoadRestoresQuantMetadata",
    "Qwen3_QINT8_FP32");
  const auto fp32_data_type = causallm_test::makeTinyFp32DataType();
  const auto qint8_data_type = causallm_test::makeTinyQint8Fp32DataType();

  auto source_model_cfg = makeTinyQwen3Config();
  auto source_generation_cfg = causallm_test::makeTinyGenerationConfig();
  auto source_nntr_cfg = causallm_test::makeTinyNntrainerConfig(
    files.tokenizer_path, fp32_data_type);
  TinyQwen3CausalLM source(source_model_cfg, source_generation_cfg,
                           source_nntr_cfg);
  source.initializeModel();
  source.setQuantizedRoundTripWeights();

  ASSERT_NO_THROW(source.saveWeightWithDtype(
    files.weight_path.string(), makeQwen3LayerDtypeMap(qint8_data_type)));

  auto loaded_model_cfg = makeTinyQwen3Config();
  auto loaded_generation_cfg = causallm_test::makeTinyGenerationConfig();
  auto loaded_nntr_cfg = causallm_test::makeTinyNntrainerConfig(
    files.tokenizer_path, qint8_data_type);
  TinyQwen3CausalLM loaded(loaded_model_cfg, loaded_generation_cfg,
                           loaded_nntr_cfg);
  loaded.initializeModel();
  ASSERT_NO_THROW(loaded.loadWeight(files.weight_path.string()));

  const auto embedding = loaded.inspectWeight("embedding0");
  EXPECT_EQ(embedding.dtype, ml::train::TensorDim::DataType::QINT8);
  EXPECT_EQ(embedding.qscheme, nntrainer::QScheme::PER_CHANNEL_AFFINE);
  EXPECT_EQ(embedding.scale_size, embedding.scales.size());
  ASSERT_FALSE(embedding.scales.empty());
  ASSERT_FALSE(embedding.zp_corr.empty());
  EXPECT_TRUE(
    std::any_of(embedding.scales.begin(), embedding.scales.end(),
                [](float v) { return std::isfinite(v) && v > 0.0f; }));
  EXPECT_NE(std::accumulate(embedding.zp_corr.begin(), embedding.zp_corr.end(),
                            int64_t{0}),
            int64_t{0});

  const auto wq = loaded.inspectWeight("layer0_wq");
  EXPECT_EQ(wq.dtype, ml::train::TensorDim::DataType::QINT8);
  EXPECT_EQ(wq.qscheme, nntrainer::QScheme::PER_CHANNEL_AFFINE);
  EXPECT_EQ(wq.scale_size, wq.scales.size());
  ASSERT_FALSE(wq.scales.empty());
  ASSERT_FALSE(wq.zp_corr.empty());
  EXPECT_TRUE(std::any_of(wq.scales.begin(), wq.scales.end(), [](float v) {
    return std::isfinite(v) && v > 0.0f;
  }));
  EXPECT_NE(std::accumulate(wq.zp_corr.begin(), wq.zp_corr.end(), int64_t{0}),
            int64_t{0});
}

#ifdef ENABLE_FP16
INSTANTIATE_TEST_SUITE_P(
  Qwen3Fp16, CausalLMTinyModelTest,
  ::testing::Values(makeQwen3Case(causallm_test::makeTinyQ40Fp16DataType())),
  [](const ::testing::TestParamInfo<causallm_test::TinyCausalLMCase> &info) {
    return info.param.name;
  });
#endif

/**
 * @brief Test that a tiny Qwen3 embedding model can save/load and encode
 */
TEST(Qwen3EmbeddingTinyModelTest,
     WeightRoundTripEncodesPromptWithQ40VocabRemainder) {
  const auto files = makeQwen3EmbeddingFiles();
  auto model = makeLoadedQwen3Embedding(files);

  std::vector<float> embedding;
  ASSERT_NO_THROW(embedding = model->encodePrompt("hello tok4"));
  ASSERT_EQ(embedding.size(), 64u);

  bool has_non_zero_value = false;
  for (float value : embedding) {
    EXPECT_TRUE(std::isfinite(value));
    has_non_zero_value = has_non_zero_value || std::abs(value) > 1e-5f;
  }
  EXPECT_TRUE(has_non_zero_value);
}

} // namespace
