/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	main.cpp
 * @date	23 July 2025
 * @brief	This is a main file for CausalLM application
 * @see		https://github.com/nnstreamer/
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "json.hpp"
#include <app_context.h>
#include <engine.h>
#include <factory.h>
#include <llm_util.hpp>
#include <mha_core.h>

#include "causal_lm.h"
#include "chat_template.h"
#include "deberta_v2.h"
#include "embedding_gemma.h"
#include "gemma3_causallm.h"
#include "gemma4_causallm.h"
#if !defined(_WIN32)
#include "gptoss_cached_slim_causallm.h"
#endif
#include "gptoss_causallm.h"
#if !defined(_WIN32) && !defined(__ANDROID__)
#include "multilingual_tinybert_16mb.h"
#endif
#include "qwen2_causallm.h"
#include "qwen2_embedding.h"
#if !defined(_WIN32)
#include "qwen3_cached_slim_moe_causallm.h"
#endif
#include "qwen3_causallm.h"
#include "qwen3_embedding.h"
#include "qwen3_moe_causallm.h"
#include "qwen3_slim_moe_causallm.h"
#include "siglip2/siglip2_vision_encoder.h"
#include "timm_vit/timm_vit_transformer.h"
#if !defined(_WIN32)
#include "bert_decoder/bert_decoder.h"
#include "screenai_caption/screenai_caption.h"
#endif
#include <models/gemma3/function.h>
#if !defined(_WIN32)
#include <sys/resource.h>
#endif

#include <atomic>
#include <chrono>
#include <filesystem>
#include <thread>

using json = nlohmann::json;

std::atomic<size_t> peak_rss_kb{0};
std::atomic<bool> tracking_enabled{true};

/**
 * @brief Focused TDD unit check for mha_core static read-only cross-attention.
 *
 * Builds a single mha_core layer in external-cache (5-input) mode with
 * cross_attention=true, binds a pre-filled cross-cache of enc_len=196 rows,
 * feeds a query of height 1, and asserts the output equals a hand-computed
 * softmax(QKᵀ/sqrt(head_dim))·V over all 196 rows (fp32, tol 1e-4).
 *
 * Returns 0 on PASS, 1 on FAIL.
 */
static int runMhaCrossAttnUnit() {
  using ml::train::createLayer;
  using namespace causallm;

  constexpr unsigned int ENC_LEN = 196;
  constexpr unsigned int NUM_HEADS = 1;
  constexpr unsigned int HEAD_DIM = 4;
  constexpr unsigned int DIM = NUM_HEADS * HEAD_DIM; // 4

  // Register the mha_core factory (same as Transformer::registerCustomLayers).
  try {
    const auto &ct_engine = nntrainer::Engine::Global();
    auto *app_context = static_cast<nntrainer::AppContext *>(
      ct_engine.getRegisteredContext("cpu"));
    app_context->registerFactory(
      nntrainer::createLayer<causallm::MHACoreLayer>);
  } catch (const std::exception &e) {
    // Already registered is fine.
  }

  auto shape = [](unsigned int h, unsigned int w) {
    return "1:1:" + std::to_string(h) + ":" + std::to_string(w);
  };

  // ---- build the graph: 5 fp32 input layers -> mha_core(cross_attention) ----
  auto model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  model->setProperty(
    {withKey("batch_size", "1"), withKey("model_tensor_type", "FP32-FP32")});

  ml::train::LayerHandle in_q(createLayer(
    "input", {withKey("name", "q"), withKey("input_shape", shape(1, DIM))}));
  ml::train::LayerHandle in_k(createLayer(
    "input", {withKey("name", "k"), withKey("input_shape", shape(1, DIM))}));
  ml::train::LayerHandle in_v(createLayer(
    "input", {withKey("name", "v"), withKey("input_shape", shape(1, DIM))}));
  ml::train::LayerHandle in_ck(
    createLayer("input", {withKey("name", "ck"),
                          withKey("input_shape", shape(ENC_LEN, DIM))}));
  ml::train::LayerHandle in_cv(
    createLayer("input", {withKey("name", "cv"),
                          withKey("input_shape", shape(ENC_LEN, DIM))}));

  ml::train::LayerHandle mha(createLayer(
    "mha_core",
    {withKey("name", "xattn"), withKey("num_heads", std::to_string(NUM_HEADS)),
     withKey("num_heads_kv", std::to_string(NUM_HEADS)),
     withKey("max_timestep", std::to_string(ENC_LEN)),
     withKey("use_rope", "false"), withKey("is_causal", "false"),
     withKey("cross_attention", "true"),
     withKey("input_layers", "q,k,v,ck,cv")}));

  std::vector<ml::train::LayerHandle> layers = {in_q,  in_k,  in_v,
                                                in_ck, in_cv, mha};
  for (auto &l : layers)
    model->addLayer(l);

  if (model->compile(ml::train::ExecutionMode::INFERENCE) != 0) {
    std::cerr << "[mha-cross-unit] FAIL: compile failed\n";
    return 1;
  }
  if (model->initialize(ml::train::ExecutionMode::INFERENCE) != 0) {
    std::cerr << "[mha-cross-unit] FAIL: initialize failed\n";
    return 1;
  }

  // ---- deterministic inputs ----
  std::vector<float> q(DIM), kdummy(DIM, 0.0f), vdummy(DIM, 0.0f);
  std::vector<float> ck(ENC_LEN * DIM), cv(ENC_LEN * DIM);

  for (unsigned int d = 0; d < DIM; ++d)
    q[d] = 0.1f * (float)(d + 1);

  // Pre-fill the cross-cache with a smooth deterministic pattern.
  for (unsigned int r = 0; r < ENC_LEN; ++r) {
    for (unsigned int d = 0; d < DIM; ++d) {
      ck[r * DIM + d] = std::sin(0.01f * (float)(r + 1) * (float)(d + 1));
      cv[r * DIM + d] = std::cos(0.02f * (float)(r + 1) + 0.1f * (float)d);
    }
  }

  // ---- hand-computed reference: softmax(QKᵀ/sqrt(d))·V over all ENC_LEN ----
  std::vector<float> scores(ENC_LEN);
  const float inv_sqrt = 1.0f / std::sqrt((float)HEAD_DIM);
  float maxs = -std::numeric_limits<float>::infinity();
  for (unsigned int r = 0; r < ENC_LEN; ++r) {
    float dot = 0.0f;
    for (unsigned int d = 0; d < DIM; ++d)
      dot += q[d] * ck[r * DIM + d];
    scores[r] = dot * inv_sqrt;
    maxs = std::max(maxs, scores[r]);
  }
  float denom = 0.0f;
  for (unsigned int r = 0; r < ENC_LEN; ++r) {
    scores[r] = std::exp(scores[r] - maxs);
    denom += scores[r];
  }
  std::vector<float> ref(DIM, 0.0f);
  for (unsigned int r = 0; r < ENC_LEN; ++r) {
    const float w = scores[r] / denom;
    for (unsigned int d = 0; d < DIM; ++d)
      ref[d] += w * cv[r * DIM + d];
  }

  // ---- run the layer (query height 1, from=0,to=1) ----
  std::vector<float *> inputs = {q.data(), kdummy.data(), vdummy.data(),
                                 ck.data(), cv.data()};
  std::vector<float *> labels;
  auto out = model->incremental_inference(1, inputs, labels, 1, 0, 1, false);
  if (out.empty() || out[0] == nullptr) {
    std::cerr << "[mha-cross-unit] FAIL: no output\n";
    return 1;
  }

  // ---- compare (output height must be 1 => DIM values) ----
  float max_abs = 0.0f;
  std::cout << "[mha-cross-unit] idx  layer_out      reference\n";
  for (unsigned int d = 0; d < DIM; ++d) {
    float diff = std::fabs(out[0][d] - ref[d]);
    max_abs = std::max(max_abs, diff);
    std::cout << "  [" << d << "] " << std::setprecision(8) << out[0][d]
              << "   " << ref[d] << "   |diff|=" << diff << "\n";
  }
  delete[] out[0];

  const float TOL = 1e-4f;
  std::cout << "[mha-cross-unit] max |diff| = " << max_abs << " (tol " << TOL
            << ")\n";
  if (max_abs <= TOL) {
    std::cout << "[mha-cross-unit] PASS\n";
    return 0;
  }
  std::cerr << "[mha-cross-unit] FAIL: max diff exceeds tolerance\n";
  return 1;
}

/**
 * @brief Print the maximum resident set size for the current process.
 */
void printMemoryUsage() {
#if defined(_WIN32)
  std::cout << "Max Resident Set Size: unavailable on Windows" << std::endl;
#else
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  std::cout << "Max Resident Set Size: " << usage.ru_maxrss << " KB"
            << std::endl;
#endif
}

/**
 * @brief Read the current process resident set size on Linux.
 */
size_t read_vm_rss_kb() {
#if defined(_WIN32)
  return 0;
#else
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.find("VmRSS:") == 0) {
      size_t kb = 0;
      sscanf(line.c_str(), "VmRSS: %zu kB", &kb);
      return kb;
    }
  }
  return 0;
#endif
}

/**
 * @brief Read private resident memory from smaps_rollup on Linux.
 */
size_t read_private_rss_kb() {
#if defined(_WIN32)
  return 0;
#else
  std::ifstream smaps("/proc/self/smaps_rollup");
  std::string line;
  size_t total = 0;
  while (std::getline(smaps, line)) {
    if (line.find("Private_Clean:") == 0 || line.find("Private_Dirty:") == 0) {
      size_t kb;
      sscanf(line.c_str(), "%*s %zu", &kb);
      total += kb;
    }
  }
  return total;
#endif
}

/**
 * @brief Start a background sampler for peak private RSS.
 */
void start_peak_tracker() {
  std::thread([] {
    while (tracking_enabled.load()) {
      size_t current = read_private_rss_kb();
      size_t prev = peak_rss_kb.load();
      if (current > prev) {
        peak_rss_kb.store(current);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }).detach();
}

/**
 * @brief Stop the memory sampler and print the observed peak.
 */
void stop_and_print_peak() {
  tracking_enabled.store(false);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  std::cout << "Peak memory usage (VmRSS): " << peak_rss_kb.load() << " KB"
            << std::endl;
}

/**
 * @brief Resolve config architecture names to registered model factory names.
 */
std::string resolve_architecture(std::string model_type,
                                 const std::string &architecture) {
  std::transform(model_type.begin(), model_type.end(), model_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (model_type == "embedding") {
    if (architecture == "Qwen3ForCausalLM") {
      return "Qwen3Embedding";
    } else if (architecture == "Gemma3ForCausalLM" ||
               architecture == "Gemma3TextModel") {
      return "EmbeddingGemma";
    } else if (architecture == "Qwen2Model") {
      return "Qwen2Embedding";
    } else if (architecture == "BertForMaskedLM") {
      return "MultilingualTinyBert";
    } else if (architecture == "TimmViT" ||
               architecture == "vit_base_patch16_siglip_224") {
      return "TimmViT";
    } else if (architecture == "deberta-v2" ||
               architecture == "DebertaV2Model" ||
               architecture == "DebertaV2ForMaskedLM") {
      return "DebertaV2";
    } else {
      throw std::invalid_argument(
        "Unsupported architecture for embedding model: " + architecture);
    }
  }

  if (architecture == "TimmViT" ||
      architecture == "vit_base_patch16_siglip_224") {
    return "TimmViT";
  }

  if (architecture == "Gemma4ForConditionalGeneration") {
    return "Gemma4ForCausalLM";
  }

  if (architecture == "VisionEncoderDecoderModel") {
    return "ScreenAICaption";
  }

  return architecture;
}

/**
 * @brief Entry point for loading, initializing, and running a CausalLM model.
 */
int main(int argc, char *argv[]) {

  auto start_time = std::chrono::high_resolution_clock::now();

  /** Register all runnable causallm models to factory */
  causallm::Factory::Instance().registerModel(
    "LlamaForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::CausalLM>(cfg, generation_cfg,
                                                  nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen2ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen2CausalLM>(cfg, generation_cfg,
                                                       nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen2Embedding", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen2Embedding>(cfg, generation_cfg,
                                                        nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen3ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3CausalLM>(cfg, generation_cfg,
                                                       nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen3MoeForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3MoECausalLM>(cfg, generation_cfg,
                                                          nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen3SlimMoeForCausalLM",
    [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3SlimMoECausalLM>(
        cfg, generation_cfg, nntr_cfg);
    });
#if !defined(_WIN32)
  causallm::Factory::Instance().registerModel(
    "Qwen3CachedSlimMoeForCausalLM",
    [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3CachedSlimMoECausalLM>(
        cfg, generation_cfg, nntr_cfg);
    });
#endif
  causallm::Factory::Instance().registerModel(
    "Qwen3Embedding", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3Embedding>(cfg, generation_cfg,
                                                        nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "GptOssForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::GptOssForCausalLM>(cfg, generation_cfg,
                                                           nntr_cfg);
    });
#if !defined(_WIN32)
  causallm::Factory::Instance().registerModel(
    "GptOssCachedSlimCausalLM",
    [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::GptOssCachedSlimCausalLM>(
        cfg, generation_cfg, nntr_cfg);
    });
#endif
  causallm::Factory::Instance().registerModel(
    "Gemma3ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Gemma3CausalLM>(cfg, generation_cfg,
                                                        nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Gemma4ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Gemma4CausalLM>(cfg, generation_cfg,
                                                        nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "EmbeddingGemma", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::EmbeddingGemma>(cfg, generation_cfg,
                                                        nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "DebertaV2", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::DebertaV2>(cfg, generation_cfg,
                                                   nntr_cfg);
    });
#if !defined(_WIN32) && !defined(__ANDROID__)
  causallm::Factory::Instance().registerModel(
    "MultilingualTinyBert", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::MultilingualTinyBert>(
        cfg, generation_cfg, nntr_cfg);
    });
#endif
  causallm::Factory::Instance().registerModel(
    "TimmViT", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::TimmViTTransformer>(cfg, generation_cfg,
                                                            nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Siglip2VisionEncoder", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Siglip2VisionEncoder>(
        cfg, generation_cfg, nntr_cfg);
    });
#if !defined(_WIN32)
  causallm::Factory::Instance().registerModel(
    "ScreenAICaption", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::ScreenAICaption>(cfg, generation_cfg,
                                                         nntr_cfg);
    });
#endif

  // Validate arguments
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model_path> [input_prompt]\n"
              << "  <model_path>   : Path to model directory\n"
              << "  [input_prompt] : Optional input text (uses sample_input or "
                 "chat_input if omitted)\n"
              << "  --dump-encoder <model_path> : Run SigLIP2 encoder and dump "
                 "output to nntr_encoder_hidden.npy\n";
    return EXIT_FAILURE;
  }

  // --mha-cross-attn-unit: focused TDD unit check for mha_core static
  // read-only cross-attention (Task 5). Exits 0 on PASS, 1 on FAIL.
  if (argc >= 2 && std::string(argv[1]) == "--mha-cross-attn-unit") {
    return runMhaCrossAttnUnit();
  }

  // --decoder-init-parity: load real decoder weights and golden encoder_hidden,
  // run one decode step with token_id=101 (CLS) at position=0, save logits to
  // nntr_decoder_init_logits.npy for parity comparison with PyTorch golden.
  // Usage: ./nntr_causallm --decoder-init-parity <model_dir>
  if (argc >= 3 && std::string(argv[1]) == "--decoder-init-parity") {
    const std::string model_dir = argv[2];
    const std::string weight_file =
      model_dir + "/nntr_caption_decoder_fp32.bin";
    const std::string enc_npy = model_dir + "/golden.json.encoder_hidden.npy";

    try {
      // ----- Load encoder_hidden [1,196,256] from npy -----
      std::vector<float> enc_data;
      {
        std::ifstream npy_in(enc_npy, std::ios::binary);
        if (!npy_in)
          throw std::runtime_error("Cannot open: " + enc_npy);
        char magic[8];
        npy_in.read(magic, 8);
        if (magic[0] != '\x93' || std::string(magic + 1, 5) != "NUMPY")
          throw std::runtime_error("Not a .npy: " + enc_npy);
        uint16_t hlen = 0;
        npy_in.read(reinterpret_cast<char *>(&hlen), 2);
        npy_in.seekg(hlen, std::ios::cur);
        constexpr int ENC_COUNT = 1 * 196 * 256; // 50176
        enc_data.resize(ENC_COUNT);
        npy_in.read(reinterpret_cast<char *>(enc_data.data()),
                    static_cast<std::streamsize>(ENC_COUNT * sizeof(float)));
        if (!npy_in)
          throw std::runtime_error("Short read from: " + enc_npy);
        std::cout << "[parity] encoder_hidden loaded, first 4: " << enc_data[0]
                  << " " << enc_data[1] << " " << enc_data[2] << " "
                  << enc_data[3] << "\n";
      }

      // ----- Build and initialize BertDecoder -----
      std::cout << "[parity] constructing BertDecoder...\n";
      auto dec = std::make_unique<causallm::BertDecoder>();
      dec->initialize();
      std::cout << "[parity] loading weights from: " << weight_file << "\n";
      dec->load_weight(weight_file);
      std::cout << "[parity] decoder ready — running decode step\n";

      // ----- Run one decode step and save logits -----
      const std::string out_npy = "nntr_decoder_init_logits.npy";
      int argmax = dec->decodeStep(enc_data.data(), 101, out_npy);
      std::cout << "[parity] done. argmax=" << argmax << "\n";
      return EXIT_SUCCESS;
    } catch (const std::exception &e) {
      std::cerr << "[parity] FAILED: " << e.what() << "\n";
      return EXIT_FAILURE;
    }
  }

  // --caption-parity / --caption-run: drive the ScreenAICaption orchestrator.
  //   --caption-parity <model_dir> : Stage A (PyTorch-preprocessed pixels from
  //       golden.json.pixel.npy, bypassing C++ resize) + Stage B (run on
  //       sample.png through the C++ resize). Emits token_ids to
  //       nntr_tokens_stageA.json / nntr_tokens_stageB.json for verify_parity.
  //   --caption-run <model_dir> [image] : Stage B only (image-path captioning).
  if (argc >= 3 && (std::string(argv[1]) == "--caption-parity" ||
                    std::string(argv[1]) == "--caption-run")) {
    const bool parity_mode = std::string(argv[1]) == "--caption-parity";
    const std::string cap_dir = argv[2];

    auto load_npy_floats = [](const std::string &path,
                              int expected_count) -> std::vector<float> {
      std::ifstream npy_in(path, std::ios::binary);
      if (!npy_in)
        throw std::runtime_error("Cannot open npy: " + path);
      char magic[8];
      npy_in.read(magic, 8);
      if (magic[0] != '\x93' || std::string(magic + 1, 5) != "NUMPY")
        throw std::runtime_error("Not a .npy file: " + path);
      uint16_t hlen = 0;
      npy_in.read(reinterpret_cast<char *>(&hlen), 2);
      npy_in.seekg(hlen, std::ios::cur);
      std::vector<float> data(static_cast<size_t>(expected_count));
      npy_in.read(reinterpret_cast<char *>(data.data()),
                  static_cast<std::streamsize>(expected_count * sizeof(float)));
      if (!npy_in)
        throw std::runtime_error("Short read from npy: " + path);
      return data;
    };

    auto dump_tokens_json = [](const std::string &path,
                               const std::vector<int> &ids) {
      std::ofstream out(path);
      out << "{\"token_ids\": [";
      for (size_t i = 0; i < ids.size(); ++i)
        out << (i ? ", " : "") << ids[i];
      out << "]}";
      out.close();
      std::cout << "[caption] wrote " << path << " (" << ids.size()
                << " tokens)\n";
    };

    try {
      json cap_cfg = causallm::LoadJsonFile(cap_dir + "/config.json");
      json cap_gen = json::object();
      if (std::filesystem::exists(cap_dir + "/generation_config.json"))
        cap_gen = causallm::LoadJsonFile(cap_dir + "/generation_config.json");
      json cap_nntr = causallm::LoadJsonFile(cap_dir + "/nntr_config.json");

      auto cap =
        std::make_unique<causallm::ScreenAICaption>(cap_cfg, cap_gen, cap_nntr);
      cap->setModelDir(cap_dir);
      std::cout << "[caption] initialize...\n";
      cap->initialize();
      std::cout << "[caption] load_weight...\n";
      cap->load_weight("");

      if (parity_mode) {
        // ----- Stage A (MODEL gate): PyTorch-preprocessed pixels, bypass the
        //       C++ resize so the encoder input is bit-identical to golden.
        // NOTE: the compiled encoder/decoder graph is memory-planned for a
        // single full run; Stage B (image path) must be driven in a separate
        // process via --caption-run to avoid exhausting the planned tensor
        // pool.
        const std::string pixel_npy = cap_dir + "/golden.json.pixel.npy";
        constexpr int PIXEL_COUNT = 1 * 3 * 224 * 224;
        std::cout << "[caption] Stage A: encodePixels from " << pixel_npy
                  << "\n";
        std::vector<float> pixels = load_npy_floats(pixel_npy, PIXEL_COUNT);
        cap->runFromPixels(pixels.data(), static_cast<size_t>(PIXEL_COUNT),
                           true);
        dump_tokens_json("nntr_tokens_stageA.json", cap->token_ids());
      } else {
        // ----- Stage B: image-path captioning through the C++ resize -----
        std::string cap_image =
          (argc >= 4) ? std::string(argv[3]) : (cap_dir + "/sample.png");
        std::cout << "[caption] Stage B: run on image " << cap_image << "\n";
        cap->run(cap_image, false, "", "", true);
        dump_tokens_json("nntr_tokens_stageB.json", cap->token_ids());
      }

      return EXIT_SUCCESS;
    } catch (const std::exception &e) {
      std::cerr << "[caption] FAILED: " << e.what() << "\n";
      return EXIT_FAILURE;
    }
  }

  // --smoke-caption-decoder: initialize BertDecoder, allocate KV cache,
  // run one incremental step with dummy inputs, and report success/failure.
  if (argc >= 2 && std::string(argv[1]) == "--smoke-caption-decoder") {
    try {
      std::cout << "[smoke] Constructing BertDecoder...\n";
      auto dec = std::make_unique<causallm::BertDecoder>();

      std::cout << "[smoke] initialize()...\n";
      dec->initialize();
      std::cout << "[smoke] initialize() OK\n";

      bool ok = dec->smokeTest();
      return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception &e) {
      std::cerr << "[smoke] FAILED: " << e.what() << "\n";
      return EXIT_FAILURE;
    }
  }

  // --dump-encoder: run Siglip2VisionEncoder and dump [1,196,256] to npy
  // Optional --input-pixels <path.npy> bypasses C++ image resize: loads a
  // [1,3,224,224] float32 npy produced by verify_parity.py --dump so the
  // encoder math can be compared against the golden with identical pixels.
  if (argc >= 3 && std::string(argv[1]) == "--dump-encoder") {
    const std::string dump_model_path = argv[2];

    // Parse optional --input-pixels <npy_path> (must come before image arg)
    std::string pixel_npy_path;
    std::string dump_image;
    {
      int arg_idx = 3;
      while (arg_idx < argc) {
        if (std::string(argv[arg_idx]) == "--input-pixels" &&
            arg_idx + 1 < argc) {
          pixel_npy_path = argv[arg_idx + 1];
          arg_idx += 2;
        } else {
          dump_image = argv[arg_idx];
          ++arg_idx;
        }
      }
      if (dump_image.empty())
        dump_image = dump_model_path + "/sample.png";
    }

    try {
      json dump_cfg = causallm::LoadJsonFile(dump_model_path + "/config.json");
      json dump_gen_cfg = json::object();
      json dump_nntr_cfg =
        causallm::LoadJsonFile(dump_model_path + "/nntr_config.json");

      // Patch nntr_cfg to satisfy Transformer base requirements
      if (!dump_nntr_cfg.contains("model_type"))
        dump_nntr_cfg["model_type"] = "Model";
      if (!dump_nntr_cfg.contains("skip_tokenizer"))
        dump_nntr_cfg["skip_tokenizer"] = true;

      const std::string enc_weight =
        dump_model_path + "/" +
        dump_nntr_cfg["encoder_model_file_name"].get<std::string>();

      auto enc = std::make_unique<causallm::Siglip2VisionEncoder>(
        dump_cfg, dump_gen_cfg, dump_nntr_cfg);
      std::cout << "[dump-encoder] initialize...\n";
      enc->initialize();
      std::cout << "[dump-encoder] load_weight from: " << enc_weight << "\n";
      enc->load_weight(enc_weight);

      std::vector<float> out;
      if (!pixel_npy_path.empty()) {
        // --input-pixels path: load [1,3,224,224] float32 npy, bypass C++
        // resize
        std::cout << "[dump-encoder] loading pixels from npy: "
                  << pixel_npy_path << "\n";
        std::ifstream npy_in(pixel_npy_path, std::ios::binary);
        if (!npy_in)
          throw std::runtime_error("Cannot open pixel npy: " + pixel_npy_path);
        // Read magic (6 bytes) + ver (2 bytes)
        char magic[8];
        npy_in.read(magic, 8);
        if (magic[0] != '\x93' || std::string(magic + 1, 5) != "NUMPY")
          throw std::runtime_error("Not a valid .npy file: " + pixel_npy_path);
        // Read header length (2 bytes little-endian)
        uint16_t hlen = 0;
        npy_in.read(reinterpret_cast<char *>(&hlen), 2);
        // Skip header
        npy_in.seekg(hlen, std::ios::cur);
        // Read floats: 1*3*224*224 = 150528
        constexpr int PIXEL_COUNT = 1 * 3 * 224 * 224;
        std::vector<float> pixel_data(PIXEL_COUNT);
        npy_in.read(reinterpret_cast<char *>(pixel_data.data()),
                    static_cast<std::streamsize>(PIXEL_COUNT * sizeof(float)));
        if (!npy_in)
          throw std::runtime_error("Failed to read pixel data from: " +
                                   pixel_npy_path);
        std::cout << "[dump-encoder] pixels loaded (" << PIXEL_COUNT
                  << " floats). "
                  << "First 3: " << pixel_data[0] << " " << pixel_data[1] << " "
                  << pixel_data[2] << "\n";

        // Run encoder with pre-loaded pixels (bypasses C++ image preprocessing)
        out = enc->encodePixels(pixel_data.data(),
                                static_cast<size_t>(PIXEL_COUNT));
      } else {
        std::cout << "[dump-encoder] encode: " << dump_image << "\n";
        out = enc->encode(dump_image);
      }
      std::cout << "[dump-encoder] encode done, first 10:";
      for (int i = 0; i < 10 && i < (int)out.size(); ++i)
        std::cout << " " << out[i];
      std::cout << "\n";

      // Write numpy .npy file (float32, shape [1,196,256])
      const std::string npy_path = "nntr_encoder_hidden.npy";
      {
        std::ofstream npy(npy_path, std::ios::binary);
        // NPY magic + version
        const char magic[] = "\x93NUMPY\x01\x00";
        npy.write(magic, 8);
        // Header: shape (1, 196, 256), float32, C order.
        // NPY spec: hlen (the 2-byte field) is the length of the header dict
        // string INCLUDING the terminating '\n'. Total header block is
        // magic(8) + hlen_field(2) + header_bytes, and must be a multiple
        // of 64.
        std::string header =
          "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 196, 256), }";
        // Append '\n' first, then pad with spaces before it so that
        // (8 + 2 + header.size()) is a multiple of 64.
        header += '\n';
        while ((8 + 2 + header.size()) % 64 != 0) {
          // Insert a space before the trailing '\n'
          header.insert(header.size() - 1, 1, ' ');
        }
        uint16_t hlen = static_cast<uint16_t>(header.size());
        npy.write(reinterpret_cast<const char *>(&hlen), 2);
        npy.write(header.c_str(), static_cast<std::streamsize>(header.size()));
        npy.write(reinterpret_cast<const char *>(out.data()),
                  static_cast<std::streamsize>(out.size() * sizeof(float)));
      }
      std::cout << "Dumped encoder output to " << npy_path << " (" << out.size()
                << " floats)\n";
      return EXIT_SUCCESS;
    } catch (const std::exception &e) {
      std::cerr << "\n[!] FATAL ERROR (--dump-encoder): " << e.what() << "\n";
      return EXIT_FAILURE;
    }
  }

  const std::string model_path = argv[1];
  std::string input_text;
  std::string system_head_prompt = "";
  std::string system_tail_prompt = "";

  std::cout << model_path << std::endl;

  try {
    // Load configuration files
    json cfg = causallm::LoadJsonFile(model_path + "/config.json");
    json generation_cfg = json::object();
    std::string generation_config_path = model_path + "/generation_config.json";
    if (std::filesystem::exists(generation_config_path)) {
      generation_cfg = causallm::LoadJsonFile(generation_config_path);
    }
    json nntr_cfg = causallm::LoadJsonFile(model_path + "/nntr_config.json");

    if (nntr_cfg.contains("system_prompt")) {
      system_head_prompt =
        nntr_cfg["system_prompt"]["head_prompt"].get<std::string>();
      system_tail_prompt =
        nntr_cfg["system_prompt"]["tail_prompt"].get<std::string>();
    }

    // Construct weight file path
    const std::string weight_file =
      model_path + "/" + nntr_cfg["model_file_name"].get<std::string>();

    std::cout << weight_file << std::endl;

    // Initialize and run model
    std::string architecture;
    if (cfg.contains("architectures") && cfg["architectures"].is_array() &&
        !cfg["architectures"].empty()) {
      architecture = cfg["architectures"].get<std::vector<std::string>>()[0];
    } else if (cfg.contains("architecture") &&
               cfg["architecture"].is_string()) {
      architecture = cfg["architecture"].get<std::string>();
    } else if (cfg.contains("model_type") && cfg["model_type"].is_string()) {
      architecture = cfg["model_type"].get<std::string>();
    } else {
      throw std::invalid_argument(
        "config.json must contain 'architectures', 'architecture', or "
        "'model_type'.");
    }

    if (nntr_cfg.contains("model_type")) {
      std::string model_type = nntr_cfg["model_type"].get<std::string>();
      architecture = resolve_architecture(model_type, architecture);
    }

    // Load chat template from tokenizer_config.json or jinja (if available)
    std::optional<causallm::ChatTemplate> chat_template;
    if (causallm::ChatTemplate::Exists(model_path)) {
      chat_template.emplace(causallm::ChatTemplate::Load(model_path));
    }

    // Determine input text
    if (argc >= 3) {
      input_text = argv[2];
    } else {
      if (nntr_cfg.contains("chat_input")) {
        if (chat_template.has_value()) {
          input_text = chat_template->apply(nntr_cfg["chat_input"]);
          system_head_prompt.clear();
          system_tail_prompt.clear();
        } else {
          std::cerr << "[Warning] 'chat_input' is set but support for model "
                       "architecture '"
                    << architecture
                    << "' is not implemented. Falling back to 'sample_input'."
                    << std::endl;
          input_text = nntr_cfg["sample_input"].get<std::string>();
        }
      } else {
        input_text = nntr_cfg["sample_input"].get<std::string>();
      }
    }

    // Inject model_path so image-only orchestrators (e.g. ScreenAICaption) can
    // resolve their weight / tokenizer files relative to the model directory.
    if (!nntr_cfg.contains("model_dir"))
      nntr_cfg["model_dir"] = model_path;

    auto model = causallm::Factory::Instance().create(architecture, cfg,
                                                      generation_cfg, nntr_cfg);
    if (!model) {
      std::cerr << "Unknown architecture: " << architecture << std::endl;
      std::cerr << "Registered architectures:";
      causallm::Factory::Instance().printRegistered(std::cerr);
      std::cerr << std::endl;
      return EXIT_FAILURE;
    }
    model->initialize();
    model->load_weight(weight_file);

    bool do_sample = generation_cfg.value("do_sample", false);

#ifdef PROFILE
    start_peak_tracker();
#endif
#if defined(_WIN32)
    model->run(input_text.c_str(), do_sample, system_head_prompt.c_str(),
               system_tail_prompt.c_str());
#else
    model->run(input_text, do_sample, system_head_prompt, system_tail_prompt);
#endif
#ifdef PROFILE
    stop_and_print_peak();
#endif
    auto finish_time = std::chrono::high_resolution_clock::now();
    auto e2e_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      finish_time - start_time);
    std::cout << "[e2e time]: " << e2e_duration.count() << " ms \n";
    printMemoryUsage();

  } catch (const std::exception &e) {
    std::cerr << "\n[!] FATAL ERROR: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
