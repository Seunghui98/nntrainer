// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   gliner2_multi_v1.cpp
 * @date   14 January 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include "json.hpp"

#include <filesystem>
#include <fstream>
#include <gliner2_multi_v1.h>
#include <layer_context.h>
#include <app_context.h>
#include <engine.h>
#include <layer_node.h>
#include "../../layers/gather_layer.h"

#include <llm_util.hpp>

#include <sstream>
#include <filesystem>

using json = nlohmann::json;

namespace causallm {

void GLiner2MultiV1::constructModel() {
  
  DebertaV2::constructModel();

  std::vector<LayerHandle> layers;

  std::string encoder_last_layer = "layer" + std::to_string(NUM_LAYERS - 1) + "_output";
  
  // Calculate Max Spans
  unsigned int num_spans = MAX_SEQ_LEN * MAX_WIDTH;
  std::string span_input_shape = "1:1:" + std::to_string(num_spans);

  // Define inputs for Span Representation
  layers.push_back(createLayer(
    "input", {withKey("name", "span_start_idxs"),
              withKey("input_shape", span_input_shape)}));
  layers.push_back(createLayer(
    "input", {withKey("name", "span_end_idxs"),
              withKey("input_shape", span_input_shape)}));

  // SpanRepLayer - Parallel Project Start/End
  // Both take encoder_last_layer as input
  int HIDDEN_SIZE = DIM;
  auto proj_start_layers = createMlp("span_rep_project_start", HIDDEN_SIZE, INTERMEDIATE_SIZE, encoder_last_layer);
  layers.insert(layers.end(), proj_start_layers.begin(), proj_start_layers.end());
  // [필수] Python forward 함수 내부의 숨겨진 ReLU를 구현
  layers.push_back(createLayer(
    "activation",
    {withKey("name", "span_rep_project_start_relu"),
     withKey("activation", "relu"), 
     withKey("input_layers", "span_rep_project_start_ffn_fc2")})); 


  // std::string proj_start_out = "span_rep_project_start_ffn_fc2";
  std::string proj_start_out = "span_rep_project_start_relu";

  auto proj_end_layers = createMlp("span_rep_project_end", HIDDEN_SIZE, INTERMEDIATE_SIZE, encoder_last_layer);
  layers.insert(layers.end(), proj_end_layers.begin(), proj_end_layers.end());
    layers.push_back(createLayer(
    "activation",
    {withKey("name", "span_rep_project_end_relu"),
     withKey("activation", "relu"), 
     withKey("input_layers", "span_rep_project_end_ffn_fc2")})); 
  std::string proj_end_out = "span_rep_project_end_relu";

  // Gather (Start Indices)
  layers.push_back(createLayer(
    "causallm_gather",
    {withKey("name", "span_rep_gather_start"),
     withKey("input_layers", proj_start_out + ",span_start_idxs"),
     withKey("axis", "2")})); // Access along Sequence Length (Height)

  // Gather (End Indices)
  layers.push_back(createLayer(
    "causallm_gather",
    {withKey("name", "span_rep_gather_end"),
     withKey("input_layers", proj_end_out + ",span_end_idxs"),
     withKey("axis", "2")})); // Access along Sequence Length (Height)
  
  // Concatenate (Start + End)
  layers.push_back(createLayer(
    "concat",
    {withKey("name", "span_rep_concat"),
     withKey("input_layers", "span_rep_gather_start,span_rep_gather_end"),
     withKey("axis", "3")})); // Feature axis (Width in nntrainer convention depending on usage, or Channel?)
     // Wait, if proj output is [1, 1, seq, dim], then axis 3 is dim (Width).
     // Concatting [1, 1, num_spans, dim] + [1, 1, num_spans, dim] -> [1, 1, num_spans, 2*dim]
     // So axis 3 (Width) is correct for concatenating features.

  // SpanRepLayer - Project Out
  auto proj_out_layers = createMlp("span_rep_project_out", HIDDEN_SIZE, INTERMEDIATE_SIZE, "span_rep_concat");
  layers.insert(layers.end(), proj_out_layers.begin(), proj_out_layers.end());
  std::string span_rep_out = "span_rep_project_out_ffn_fc2";

  // Count Predictor
  auto count_pred_layers = createMlp("count_pred", 20, HIDDEN_SIZE*2, encoder_last_layer);
  layers.insert(layers.end(), count_pred_layers.begin(), count_pred_layers.end());
  std::string cout_pred_out = "count_pred_ffn_fc2";

  // Classifier
  auto classifier_layers = createMlp("classifier", 1, HIDDEN_SIZE*2, span_rep_out);
  layers.insert(layers.end(), classifier_layers.begin(), classifier_layers.end());
  std::string classifier_out = "classifier_ffn_fc2";

  for (auto &layer : layers) {
    model->addLayer(layer);
  }
}

void GLiner2MultiV1::setupParameters(json &cfg, json &generation_cfg,
                                json &nntr_cfg) {
  try {
    std::string tokenizer_file = nntr_cfg["tokenizer_file"].get<std::string>();
    std::filesystem::path tokenizer_path(tokenizer_file);
    std::string model_path = tokenizer_path.parent_path().string();
    std::filesystem::path encoder_config_path = std::filesystem::path(model_path) / "encoder_config" / "config.json";
    
    json encoder_cfg = causallm::LoadJsonFile(encoder_config_path.string());
    Transformer::setupParameters(encoder_cfg, generation_cfg, nntr_cfg);
    if (encoder_cfg.contains("max_relative_positions")) {
        MAX_RELATIVE_POSITIONS = encoder_cfg["max_relative_positions"].get<int>();
        if (MAX_RELATIVE_POSITIONS == -1) {
            MAX_RELATIVE_POSITIONS = MAX_POSITION_EMBEDDINGS;
        }
    }

    if (encoder_cfg.contains("pos_att_type")) {
      std::vector<std::string> pos_att_type_vec;
      if (encoder_cfg["pos_att_type"].is_array()) {
        pos_att_type_vec = encoder_cfg["pos_att_type"].get<std::vector<std::string>>();
      } else if (encoder_cfg["pos_att_type"].is_string()) {
        std::string pos_att_str = encoder_cfg["pos_att_type"].get<std::string>();

        std::stringstream ss(pos_att_str);
        std::string token;
        
        while (std::getline(ss, token, '|')) {
          token.erase(0, token.find_first_not_of(" \t"));
          token.erase(token.find_last_not_of(" \t") + 1);
          
          if (!token.empty()) {
            pos_att_type_vec.push_back(token);
          }
        }
    }
      c2p = std::find(pos_att_type_vec.begin(), pos_att_type_vec.end(), "c2p") != pos_att_type_vec.end();
      p2c = std::find(pos_att_type_vec.begin(), pos_att_type_vec.end(), "p2c") != pos_att_type_vec.end();
    } else {
      c2p = false;
      p2c = false; 
    }

    if (encoder_cfg.contains("share_att_key")) {
      SHARE_ATT_KEY = encoder_cfg["share_att_key"].get<bool>();
    } else {
      SHARE_ATT_KEY = false;
    }

    if (encoder_cfg.contains("relative_attention")) {
      RELATIVE_ATTENTION = encoder_cfg["relative_attention"].get<bool>();
    } else {
      RELATIVE_ATTENTION = true;
    }

    if (encoder_cfg.contains("position_buckets")) {
      POSITION_BUCKETS = encoder_cfg["position_buckets"].get<int>();
    } else {
      POSITION_BUCKETS = -1;
    }

    // Load root config.json for GLiNER params (e.g. max_width)
    std::filesystem::path root_config_path = std::filesystem::path(model_path) / "config.json";
    if (std::filesystem::exists(root_config_path)) {
      json root_cfg = causallm::LoadJsonFile(root_config_path.string());
      if (root_cfg.contains("max_width")) {
        MAX_WIDTH = root_cfg["max_width"].get<int>();
      }
    } else if (cfg.contains("max_width")) {
      // Fallback to cfg if root config file not found
      MAX_WIDTH = cfg["max_width"].get<int>();
    }

  } catch (const std::exception &e) {
    std::cerr << "\n[!] FATAL ERROR: " << e.what() << "\n";
  }
}

std::vector<LayerHandle>
GLiner2MultiV1::createMlp(const std::string &layer_name, int dim, int hidden_dim,
                          std::string input_name) {

  std::vector<LayerHandle> layers;

  // FC1
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", layer_name + "_ffn_fc1"),
     withKey("unit", hidden_dim),
     withKey("input_layers", input_name),
     withKey("disable_bias", "false"), 
     withKey("weight_initializer", "ones")}));

  // Activation (ReLU)
  layers.push_back(createLayer(
    "activation",
    {withKey("name", layer_name + "_ffn_act"),
     withKey("activation", "relu"),
     withKey("input_layers",
             layer_name + "_ffn_fc1")}));

  // FC2
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", layer_name + "_ffn_fc2"),
     withKey("unit", dim),
     withKey("disable_bias", "false"), 
     withKey("input_layers",
             layer_name + "_ffn_act"),
     withKey("weight_initializer", "ones")}));

  return layers;
}

std::pair<std::vector<float>, std::vector<float>>
GLiner2MultiV1::generate_spans(int seq_len, int max_width) {
  std::vector<float> start_idxs;
  std::vector<float> end_idxs;

  for (int i = 0; i < seq_len; ++i) {
    for (int k = 0; k < max_width; ++k) {
      if (i + k < seq_len) {
        start_idxs.push_back(static_cast<float>(i));
        end_idxs.push_back(static_cast<float>(i + k));
      }
    }
  }
  return {start_idxs, end_idxs};
}

std::vector<float *> GLiner2MultiV1::encode(const WSTR prompt,
                                            const WSTR system_prompt,
                                            const WSTR tail_prompt) {
  if (!is_initialized) {
    throw std::runtime_error("Model is not initialized. Please call "
                             "initialize() before encode().");
  }

#if defined(_WIN32)
  std::wstring prompt_ = system_prompt + prompt + tail_prompt;
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  auto _input = tokenizer->Encode(converter.to_bytes(prompt_), true);
#else
  std::string prompt_ = system_prompt + prompt + tail_prompt;
  auto _input = tokenizer->Encode(prompt_, true);
#endif

  std::vector<int64_t> init_input;
  unsigned int input_len =
    std::min((unsigned int)_input.size() - 1, (unsigned int)MAX_SEQ_LEN);

  // feed only available length
  for (unsigned int i = 0; i < input_len; ++i)
    init_input.push_back(_input[i]);

  // 1. Prepare Token Input
  float *input_sample =
    (float *)malloc(sizeof(float) * BATCH_SIZE * MAX_SEQ_LEN);

  std::fill(input_sample, input_sample + BATCH_SIZE * MAX_SEQ_LEN, 0.0f);

  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    for (unsigned int i = 0; i < input_len; ++i) {
      input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN + i] =
        static_cast<float>(init_input[i]);
    }
  }

  // 2. Generate Spans
  auto [start_idxs_vec, end_idxs_vec] = generate_spans(input_len, MAX_WIDTH);
  size_t num_spans = start_idxs_vec.size();
  size_t required = (size_t)MAX_SEQ_LEN * (size_t)MAX_WIDTH;

  // 3. Prepare Span Inputs
  // Note: Input shape for spans should likely match the number of spans.
  // Assuming the model can handle dynamic span count or we have a max span count.
  // For 'incremental_inference', fixed size inputs are usually expected unless dynamic shape/batch is handled.
  // Here we allocate enough space. If model input shape is 1:1:1, it might be auto-adjusted or we need to ensure match.
  // Let's assume we pass flattened span indices.
    
  float *start_input = (float *)malloc(sizeof(float) * required);
  float *end_input = (float *)malloc(sizeof(float) * required);
  
  if (!start_input || !end_input) {
    free(input_sample);
    if(start_input) free(start_input);
    if(end_input) free(end_input);
    throw std::runtime_error("Failed to allocate memory for span inputs");
  }

  std::fill(start_input, start_input + required, 0.0f);
  std::fill(end_input, end_input + required, 0.0f);

  std::copy(start_idxs_vec.begin(), start_idxs_vec.end(), start_input);
  std::copy(end_idxs_vec.begin(), end_idxs_vec.end(), end_input);

  std::vector<float *> input;
  input.push_back(end_input);
  input.push_back(start_input);
  input.push_back(input_sample);

  std::vector<float *> label; // Empty label for inference

  // Run inference
  // Note: 'input_len' used in incremental_inference usually affects the primary input (tokens).
  // Span inputs are auxiliary. This might differ based on generic implementation.
  std::vector<float *> output = model->incremental_inference(
    BATCH_SIZE, input, label, input_len, 0, input_len, true);

  free(input_sample);
  free(start_input);
  free(end_input);

  return output;
}

void GLiner2MultiV1::registerCustomLayers() {
  DebertaV2::registerCustomLayers();
  
  const auto &ct_engine = nntrainer::Engine::Global();
  auto app_context = static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(nntrainer::createLayer<causallm::GatherLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what() << std::endl;
  }
}

} // namespace causallm
