// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   quantize.cpp
 * @date   04 March 2026
 * @brief  Quantization utility for CausalLM models (including Qwen3.5)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 *
 * @usage
 *   nntr_quantize <model_path> [options]
 *
 *   Options:
 *     --output, -o <path> Output directory (default: <model_path>)
 *     --fc_dtype <type>   Target dtype for FC layers (default: Q4_0)
 *     --embd_dtype <type> Target dtype for embedding (default: FP32)
 *     --lmhead_dtype <type> Target dtype for LM head (default: FP32)
 *     --output_bin <name> Output bin filename (auto-generated if omitted)
 *     --config <path>     Use a target nntr_config.json for dtype settings
 *
 *   Supported data types: FP32, FP16, Q4_0, Q6_K, Q4_K
 *
 *   Example:
 *     nntr_quantize /path/to/qwen3_5-2b --fc_dtype Q4_0
 */

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "json.hpp"
#include <app_context.h>
#include <factory.h>
#include <tensor_dim.h>

#include "causal_lm.h"
#include "embedding_gemma.h"
#include "gemma3_causallm.h"
#include "gptoss_cached_slim_causallm.h"
#include "gptoss_causallm.h"
#include "qwen2_causallm.h"
#include "qwen2_embedding.h"
#include "qwen3_cached_slim_moe_causallm.h"
#include "qwen3_causallm.h"
#include "qwen3_5_causallm.h"
#include "qwen3_embedding.h"
#include "qwen3_moe_causallm.h"
#include "qwen3_slim_moe_causallm.h"
#include "multilingual_tinybert_16mb.h"

using json = nlohmann::json;
using DataType = ml::train::TensorDim::DataType;

namespace {

const std::map<std::string, DataType> dtype_str_map = {
  {"FP32", DataType::FP32}, {"FP16", DataType::FP16}, {"Q4_0", DataType::Q4_0},
  {"Q6_K", DataType::Q6_K}, {"Q4_K", DataType::Q4_K}, {"NONE", DataType::NONE},
};

DataType strToDataType(const std::string &s) {
  std::string upper = s;
  std::transform(upper.begin(), upper.end(), upper.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  auto it = dtype_str_map.find(upper);
  if (it == dtype_str_map.end())
    throw std::invalid_argument("Unsupported data type: " + s);
  return it->second;
}

std::string dataTypeToStr(DataType dt) {
  for (const auto &[key, val] : dtype_str_map)
    if (val == dt)
      return key;
  return "NONE";
}

std::string buildModelTensorType(const std::string &fc_dtype) {
  return fc_dtype + "-FP32";
}

std::string generateOutputBinName(const std::string &original_bin,
                                  const std::string &fc_dtype,
                                  const std::string &embd_dtype) {
  std::string base = original_bin;
  auto dot_pos = base.rfind(".bin");
  if (dot_pos != std::string::npos)
    base = base.substr(0, dot_pos);

  std::vector<std::string> dtype_suffixes = {"_fp32", "_fp16", "_q40", "_q4_0",
                                             "_q6k",  "_q6_k", "_q4k", "_q4_k"};
  for (const auto &suffix : dtype_suffixes) {
    auto pos = base.rfind(suffix);
    if (pos != std::string::npos && pos + suffix.size() == base.size()) {
      base = base.substr(0, pos);
      break;
    }
  }

  std::string fc_lower = fc_dtype;
  std::transform(fc_lower.begin(), fc_lower.end(), fc_lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  std::string fc_clean = fc_lower;
  fc_clean.erase(std::remove(fc_clean.begin(), fc_clean.end(), '_'),
                 fc_clean.end());

  return base + "_" + fc_clean + ".bin";
}

std::string resolve_architecture(std::string model_type,
                                 const std::string &architecture) {
  std::transform(model_type.begin(), model_type.end(), model_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (model_type == "embedding") {
    if (architecture == "Qwen3ForCausalLM")
      return "Qwen3Embedding";
    else if (architecture == "Gemma3ForCausalLM" ||
             architecture == "Gemma3TextModel")
      return "EmbeddingGemma";
    else if (architecture == "Qwen2Model")
      return "Qwen2Embedding";
  }
  return architecture;
}

void registerAllModels() {
  auto &factory = causallm::Factory::Instance();

  factory.registerModel("LlamaForCausalLM",
    [](json cfg, json gen, json nntr) {
      return std::make_unique<causallm::CausalLM>(cfg, gen, nntr); });
  factory.registerModel("Qwen2ForCausalLM",
    [](json cfg, json gen, json nntr) {
      return std::make_unique<causallm::Qwen2CausalLM>(cfg, gen, nntr); });
  factory.registerModel("Qwen2Embedding",
    [](json cfg, json gen, json nntr) {
      return std::make_unique<causallm::Qwen2Embedding>(cfg, gen, nntr); });
  factory.registerModel("Qwen3ForCausalLM",
    [](json cfg, json gen, json nntr) {
      return std::make_unique<causallm::Qwen3CausalLM>(cfg, gen, nntr); });
  factory.registerModel("Qwen3_5ForConditionalGeneration",
    [](json cfg, json gen, json nntr) {
      return std::make_unique<causallm::Qwen3_5CausalLM>(cfg, gen, nntr); });
  factory.registerModel("Qwen3MoeForCausalLM",
    [](json cfg, json gen, json nntr) {
      return std::make_unique<causallm::Qwen3MoECausalLM>(cfg, gen, nntr); });
  factory.registerModel("Qwen3SlimMoeForCausalLM",
    [](json cfg, json gen, json nntr) {
      return std::make_unique<causallm::Qwen3SlimMoECausalLM>(cfg, gen, nntr); });
  factory.registerModel("Qwen3CachedSlimMoeForCausalLM",
    [](json cfg, json gen, json nntr) {
      return std::make_unique<causallm::Qwen3CachedSlimMoECausalLM>(cfg, gen, nntr); });
  factory.registerModel("Qwen3Embedding",
    [](json cfg, json gen, json nntr) {
      return std::make_unique<causallm::Qwen3Embedding>(cfg, gen, nntr); });
  factory.registerModel("GptOssForCausalLM",
    [](json cfg, json gen, json nntr) {
      return std::make_unique<causallm::GptOssForCausalLM>(cfg, gen, nntr); });
  factory.registerModel("GptOssCachedSlimCausalLM",
    [](json cfg, json gen, json nntr) {
      return std::make_unique<causallm::GptOssCachedSlimCausalLM>(cfg, gen, nntr); });
  factory.registerModel("Gemma3ForCausalLM",
    [](json cfg, json gen, json nntr) {
      return std::make_unique<causallm::Gemma3CausalLM>(cfg, gen, nntr); });
  factory.registerModel("EmbeddingGemma",
    [](json cfg, json gen, json nntr) {
      return std::make_unique<causallm::EmbeddingGemma>(cfg, gen, nntr); });
  factory.registerModel("BertForMaskedLM",
    [](json cfg, json gen, json nntr) {
      return std::make_unique<causallm::BertModel>(cfg, gen, nntr); });
}

/**
 * @brief Build layer dtype map including Qwen3.5 GDN FC layer names
 */
std::map<std::string, DataType>
buildLayerDtypeMap(int num_layers, DataType fc_dtype, DataType embd_dtype,
                   DataType lmhead_dtype, bool tie_word_embeddings,
                   const std::vector<bool> &is_self_attn_layer) {

  std::map<std::string, DataType> dtype_map;

  // tie_word_embeddings: embedding and lm_head share weights.
  // Quantized shared weights are not supported, so keep both as FP32.
  if (tie_word_embeddings) {
    // Skip embedding0 and output_of_causallm regardless of user dtype request
  } else {
    if (embd_dtype != DataType::FP32 && embd_dtype != DataType::NONE)
      dtype_map["embedding0"] = embd_dtype;
  }

  for (int i = 0; i < num_layers; ++i) {
    std::string prefix = "layer" + std::to_string(i);

    if (fc_dtype != DataType::FP32 && fc_dtype != DataType::NONE) {
      bool is_self_attn = (i < (int)is_self_attn_layer.size())
                            ? is_self_attn_layer[i]
                            : true; // default: self-attention

      if (is_self_attn) {
        // Self-attention FC layers (Qwen3-style with QK-norm + output gate)
        dtype_map[prefix + "_wv"] = fc_dtype;
        dtype_map[prefix + "_wk"] = fc_dtype;
        dtype_map[prefix + "_wq"] = fc_dtype;
        dtype_map[prefix + "_wq_gate"] = fc_dtype;
        dtype_map[prefix + "_attention_out"] = fc_dtype;
      } else {
        // Linear attention (GDN) decomposed FC layers
        dtype_map[prefix + "_gdn_in_proj_qkv"] = fc_dtype;
        // in_proj_a/b: width=num_v_heads(16), not divisible by 32 → skip Q4_0
        dtype_map[prefix + "_gdn_in_proj_z"] = fc_dtype;
        dtype_map[prefix + "_gdn_out_proj"] = fc_dtype;
      }

      // MLP FC layers (shared by both self-attn and linear-attn blocks)
      dtype_map[prefix + "_ffn_gate"] = fc_dtype;
      dtype_map[prefix + "_ffn_up"] = fc_dtype;
      dtype_map[prefix + "_ffn_down"] = fc_dtype;
    }
  }

  if (!tie_word_embeddings &&
      lmhead_dtype != DataType::FP32 && lmhead_dtype != DataType::NONE)
    dtype_map["output_of_causallm"] = lmhead_dtype;

  return dtype_map;
}

void printUsage(const char *prog) {
  std::cout
    << "Usage: " << prog << " <model_path> [options]\n\n"
    << "Quantize a CausalLM model from FP32 to a target data type.\n\n"
    << "Options:\n"
    << "  --output, -o <path>   Output directory (default: <model_path>)\n"
    << "  --fc_dtype <type>     Target dtype for FC layers (default: Q4_0)\n"
    << "  --embd_dtype <type>   Target dtype for embedding (default: FP32)\n"
    << "  --lmhead_dtype <type> Target dtype for LM head (default: FP32)\n"
    << "  --output_bin <name>   Output .bin filename\n"
    << "  --config <path>       Use a target nntr_config.json\n"
    << "  --help, -h            Show this help\n\n"
    << "Supported: FP32, FP16, Q4_0, Q6_K, Q4_K\n\n"
    << "Examples:\n"
    << "  " << prog << " /path/to/qwen3_5-2b --fc_dtype Q4_0\n"
    << "  " << prog << " /path/to/qwen3_5-2b --fc_dtype Q4_0 --embd_dtype Q6_K\n";
}

} // anonymous namespace

int main(int argc, char *argv[]) {
  if (argc < 2) { printUsage(argv[0]); return EXIT_FAILURE; }
  if (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
    printUsage(argv[0]); return EXIT_SUCCESS;
  }

  std::string model_path = argv[1];
  std::string output_dir = "", fc_dtype_str = "Q4_0", embd_dtype_str = "FP32";
  std::string lmhead_dtype_str = "", output_bin_name = "", target_config_path = "";

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--output" || arg == "-o") && i+1 < argc) output_dir = argv[++i];
    else if (arg == "--fc_dtype" && i+1 < argc) fc_dtype_str = argv[++i];
    else if (arg == "--embd_dtype" && i+1 < argc) embd_dtype_str = argv[++i];
    else if (arg == "--lmhead_dtype" && i+1 < argc) lmhead_dtype_str = argv[++i];
    else if (arg == "--output_bin" && i+1 < argc) output_bin_name = argv[++i];
    else if (arg == "--config" && i+1 < argc) target_config_path = argv[++i];
    else { std::cerr << "Unknown: " << arg << "\n"; return EXIT_FAILURE; }
  }

  try {
    std::cout << "==========================================================\n"
              << "  NNTrainer CausalLM Quantization Utility\n"
              << "==========================================================\n"
              << "[1/5] Loading configurations from: " << model_path << "\n";

    json cfg = causallm::LoadJsonFile(model_path + "/config.json");
    json generation_cfg = causallm::LoadJsonFile(model_path + "/generation_config.json");
    json nntr_cfg = causallm::LoadJsonFile(model_path + "/nntr_config.json");

    if (!target_config_path.empty()) {
      json target_cfg = causallm::LoadJsonFile(target_config_path);
      if (target_cfg.contains("fc_layer_dtype")) fc_dtype_str = target_cfg["fc_layer_dtype"];
      if (target_cfg.contains("embedding_dtype")) embd_dtype_str = target_cfg["embedding_dtype"];
      if (target_cfg.contains("lmhead_dtype")) lmhead_dtype_str = target_cfg["lmhead_dtype"];
      if (target_cfg.contains("model_file_name") && output_bin_name.empty())
        output_bin_name = target_cfg["model_file_name"];
    }

    if (lmhead_dtype_str.empty()) lmhead_dtype_str = embd_dtype_str;
    DataType fc_dtype = strToDataType(fc_dtype_str);
    DataType embd_dtype = strToDataType(embd_dtype_str);
    DataType lmhead_dtype = strToDataType(lmhead_dtype_str);

    if (output_dir.empty()) output_dir = model_path;
    std::filesystem::create_directories(output_dir);

    std::string original_bin = nntr_cfg.contains("safe_tensor_file_name")
      ? nntr_cfg["safe_tensor_file_name"].get<std::string>()
      : nntr_cfg["model_file_name"].get<std::string>();
    if (output_bin_name.empty())
      output_bin_name = generateOutputBinName(original_bin, dataTypeToStr(fc_dtype), dataTypeToStr(embd_dtype));

    std::string src_weight_path = model_path + "/" + original_bin;
    std::string dst_weight_path = output_dir + "/" + output_bin_name;

    // Handle VLM nested config
    json text_cfg = cfg.contains("text_config") ? cfg["text_config"] : cfg;
    int num_layers = text_cfg["num_hidden_layers"].get<int>();
    bool tie_word_embeddings = text_cfg.contains("tie_word_embeddings")
      ? text_cfg["tie_word_embeddings"].get<bool>() : false;

    // Determine self-attention vs linear attention layers (Qwen3.5 hybrid)
    std::vector<bool> is_self_attn_layer(num_layers, true); // default: all self-attn
    if (nntr_cfg.contains("self_attn_layers")) {
      std::fill(is_self_attn_layer.begin(), is_self_attn_layer.end(), false);
      for (auto idx : nntr_cfg["self_attn_layers"].get<std::vector<unsigned int>>())
        if ((int)idx < num_layers) is_self_attn_layer[idx] = true;
    } else if (nntr_cfg.contains("self_attn_pattern")) {
      unsigned int pattern = nntr_cfg["self_attn_pattern"].get<unsigned int>();
      std::fill(is_self_attn_layer.begin(), is_self_attn_layer.end(), false);
      for (unsigned int i = pattern - 1; i < (unsigned int)num_layers; i += pattern)
        is_self_attn_layer[i] = true;
    }

    std::cout << "  Architecture: " << text_cfg.value("architectures", json::array({"unknown"}))[0] << "\n"
              << "  Num layers:   " << num_layers << "\n"
              << "  Source:       " << src_weight_path << "\n"
              << "  Target:       " << dst_weight_path << "\n"
              << "  FC dtype:     " << dataTypeToStr(fc_dtype) << "\n"
              << "  Embed dtype:  " << dataTypeToStr(embd_dtype) << "\n"
              << "  LMHead dtype: " << dataTypeToStr(lmhead_dtype) << "\n\n";

    std::cout << "[2/5] Creating and initializing model...\n";
    registerAllModels();

    std::string architecture = cfg.contains("architectures")
      ? cfg["architectures"].get<std::vector<std::string>>()[0]
      : text_cfg["architectures"].get<std::vector<std::string>>()[0];
    if (nntr_cfg.contains("model_type"))
      architecture = resolve_architecture(nntr_cfg["model_type"], architecture);

    auto model = causallm::Factory::Instance().create(architecture, cfg, generation_cfg, nntr_cfg);
    model->initialize();
    std::cout << "  Model initialized.\n";

    std::cout << "[3/5] Loading FP32 weights from: " << src_weight_path << "\n";
    model->load_weight(src_weight_path);
    std::cout << "  Weights loaded.\n";

    std::cout << "[4/5] Quantizing and saving to: " << dst_weight_path << "\n";
    if (tie_word_embeddings) {
      std::cout << "  NOTE: tie_word_embeddings=true, embedding/lm_head kept FP32\n";
    }
    auto layer_dtype_map = buildLayerDtypeMap(num_layers, fc_dtype, embd_dtype,
                                              lmhead_dtype, tie_word_embeddings,
                                              is_self_attn_layer);

    std::cout << "  Layers to quantize: " << layer_dtype_map.size() << "\n";
    model->save_weight(dst_weight_path, DataType::NONE, layer_dtype_map);

    auto src_size = std::filesystem::file_size(src_weight_path);
    auto dst_size = std::filesystem::file_size(dst_weight_path);
    std::cout << "  Source:  " << (src_size / (1024*1024)) << " MB\n"
              << "  Output:  " << (dst_size / (1024*1024)) << " MB\n"
              << "  Ratio:   " << std::fixed << std::setprecision(1)
              << (100.0 * dst_size / src_size) << "%\n";

    std::cout << "[5/5] Generating nntr_config.json...\n";
    json new_cfg = nntr_cfg;
    new_cfg["model_file_name"] = output_bin_name;
    new_cfg["fc_layer_dtype"] = dataTypeToStr(fc_dtype);
    new_cfg["embedding_dtype"] = dataTypeToStr(embd_dtype);
    new_cfg["lmhead_dtype"] = dataTypeToStr(lmhead_dtype);
    new_cfg["model_tensor_type"] = buildModelTensorType(dataTypeToStr(fc_dtype));

    std::string cfg_path = (output_dir == model_path)
      ? output_dir + "/nntr_config_quantized.json"
      : output_dir + "/nntr_config.json";
    std::ofstream(cfg_path) << new_cfg.dump(4) << std::endl;
    std::cout << "  Config: " << cfg_path << "\n";

    std::cout << "\n==========================================================\n"
              << "  Quantization complete!\n"
              << "==========================================================\n"
              << "\nTo run: nntr_causallm " << output_dir << "\n";

  } catch (const std::exception &e) {
    std::cerr << "\n[!] FATAL ERROR: " << e.what() << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
