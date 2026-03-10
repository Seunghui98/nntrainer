// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   gliner2_multi_v1.cpp
 * @date   14 January 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @brief  GLiNER2 Multi V1 inference wrapper.
 *
 * This file builds the task-specific GLiNER head on top of DeBERTa-v2 encoder and
 * runs incremental inference with auxiliary inputs:
 *   - word_start_idxs: token indices for word pooling
 *   - span_start_idxs/span_end_idxs: flattened span indices (word-level spans)
 *   - label_idxs_input/label_slot_idxs_input: schema label anchors
 *
 * Design note :
 *  - encode() performs: input preprocessing + incremental_inference() call.
 *  - run() performs: encode() call + post-processing/decode using intermediate layer outputs.
 * Please refer to the following code :
 *  https://github.com/fastino-ai/GLiNER2/blob/72241d6/gliner2/inference/engine.py
 */

#include "json.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <codecvt>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <app_context.h>
#include <engine.h>
#include <gliner2_multi_v1.h>
#include <layer_context.h>
#include <layer_node.h>

#include "../../layers/gather_layer.h"
#include <custom_gru_layer.h>
#include <llm_util.hpp>

using json = nlohmann::json;

namespace causallm {

namespace {

// Maximum number of labels supported by the head (must match model definition).
constexpr int kMaxLabels = 20;

// Probability threshold for candidate creation (sigmoid(logit) > threshold).
constexpr float kProbThreshold = 0.5f;

/**
 * @brief Candidate span for entity extraction.
 *
 * start/end are word indices (not token indices). 'label' is one of schema labels.
 * score is sigmoid probability. text is surface form constructed from original words.
 */
struct EntityCandidate {
  int start = 0;
  int end = 0;
  std::string label;
  float score = 0.0f;
  std::string text;
};

/**
 * @brief Parsed label anchors from schema header.
 *
 * label_idxs: token indices of [E] markers in the header (size = kMaxLabels).
 * label_slot_idxs: [0..found_cnt-1] used to gather from count_embed outputs.
 * valid_labels: actual label strings (size = found_cnt).
 */
struct LabelParseResult {
  std::vector<float> label_idxs;
  std::vector<float> label_slot_idxs;
  std::vector<std::string> valid_labels;
  int found_cnt = 0;
};

std::string ltrim_copy(std::string s) {
  s.erase(s.begin(),
          std::find_if(s.begin(), s.end(),
                       [](unsigned char ch) { return !std::isspace(ch); }));
  return s;
}

std::string to_lower_copy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

/**
 * @brief Split by ASCII whitespace.
 *
 * This is used to create "word-level" boundaries for span generation.
 * Words are derived from a normalized text string (punctuation-separated).
 */
std::vector<std::string> split_ws(const std::string &s) {
  std::vector<std::string> out;
  std::istringstream iss(s);
  std::string w;
  while (iss >> w) out.push_back(w);
  return out;
}

/**
 * @brief Normalize entity surface text to match Python behavior.
 *
 * - collapse repeated spaces
 * - remove spaces around apostrophes: "d ' Or" -> "d'Or"
 * - trim
 */
std::string normalize_entity_text(std::string s) {
  s = std::regex_replace(s, std::regex(" +"), " ");
  s = std::regex_replace(s, std::regex(" ?' ?"), "'");
  s = std::regex_replace(s, std::regex("^ +| +$"), "");
  return s;
}

/**
 * @brief Greedy decode candidates for a single label.
 *
 * This matches typical GLiNER decoding:
 *  - sort by score desc
 *  - keep non-overlapping spans
 *  - return spans sorted by start asc
 */
std::vector<EntityCandidate> greedy_decode_one_label(std::vector<EntityCandidate> cands,
                                                           int max_idx = 2048) {
  if (cands.empty()) return {};

  std::sort(cands.begin(), cands.end(),
            [](const EntityCandidate &a, const EntityCandidate &b) { return a.score > b.score; });

  std::vector<bool> occupied(max_idx, false);
  std::vector<EntityCandidate> results;
  results.reserve(cands.size());

  for (const auto &cand : cands) {
    if (cand.start < 0 || cand.end < cand.start || cand.end >= max_idx) continue;

    bool overlap = false;
    for (int i = cand.start; i <= cand.end; ++i) {
      if (occupied[i]) {
        overlap = true;
        break;
      }
    }
    if (overlap) continue;

    results.push_back(cand);
    for (int i = cand.start; i <= cand.end; ++i) occupied[i] = true;
  }

  std::sort(results.begin(), results.end(),
            [](const EntityCandidate &a, const EntityCandidate &b) { return a.start < b.start; });

  return results;
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

/**
 * @brief Normalize raw text for tokenizer parity.
 *
 * - Separates punctuation with spaces (C++ regex [[:punct:]])
 * - Compresses multiple spaces
 * - Ensures prefix space for SentencePiece-style behavior (add_prefix_space=True)
 * - Adds spacing before trailing '.' if needed (legacy behavior)
 */
std::string normalize_punct_spacing(std::string s) {
  s = std::regex_replace(s, std::regex("([[:punct:]])"), " $1 ");
  s = std::regex_replace(s, std::regex(" +"), " ");

  if (!s.empty() && s.front() != ' ') s = " " + s;

  if (!s.empty() && s.back() == '.') {
    if (s.size() > 1 && s[s.size() - 2] != ' ') {
      s.insert(s.size() - 1, " ");
    }
  }
  return s;
}

/**
 * @brief Extract substring after special separator token and trim leading whitespace.
 */
std::string extract_text_after_sep(const std::string &full, const std::string &sep_token) {
  const size_t pos = full.find(sep_token);
  if (pos == std::string::npos) return "";
  return ltrim_copy(full.substr(pos + sep_token.size()));
}

/**
 * @brief Join words [s..e] into a surface string.
 */
std::string join_words(const std::vector<std::string> &words, int s, int e) {
  if (s < 0 || e < s) return {};
  std::string out;
  for (int i = s; i <= e; ++i) {
    if (i < 0 || static_cast<size_t>(i) >= words.size()) break;
    if (!out.empty()) out += " ";
    out += words[static_cast<size_t>(i)];
  }
  return out;
}

/**
 * @brief Build word_start_idxs input (length = MAX_SEQ_LEN).
 *
 * word_start_idxs is a list of token positions where each word begins in the
 * LOWERCASED text segment. The first value is text_offset, and subsequent values
 * are computed by tokenizing the incremental prefix:
 *
 *   idx[i] = text_offset + len(Encode(words_lower[0..i], add_special_tokens=false))
 *
 * We copy (num_words) start indices into the first part of the input tensor.
 * Remaining entries are padded with zeros.
 */
std::vector<float> build_word_start_idxs(const std::vector<std::string> &words_lower,
                                                int text_offset,
                                                tokenizers::Tokenizer *tokenizer,
                                                int max_seq_len) {
  std::vector<float> word_start_input(static_cast<size_t>(max_seq_len), 0.0f);
  if (!tokenizer || words_lower.empty()) return word_start_input;

  std::vector<float> word_start_idxs_vec;
  word_start_idxs_vec.reserve(words_lower.size() + 1);

  word_start_idxs_vec.push_back(static_cast<float>(text_offset));

  std::string accumulated;
  for (size_t i = 0; i < words_lower.size(); ++i) {
    if (i > 0) accumulated += " ";
    accumulated += words_lower[i];

    // Note: Encode(..., add_special_tokens=false) is required for consistent offsets.
    auto prefix_tokens = tokenizer->Encode(accumulated, false);
    word_start_idxs_vec.push_back(static_cast<float>(text_offset + static_cast<int>(prefix_tokens.size())));
  }

  // Model expects only word start indices (excluding the trailing "end" marker).
  if (word_start_idxs_vec.size() >= 2) {
    const size_t copy_n = std::min(word_start_idxs_vec.size() - 1, static_cast<size_t>(max_seq_len));
    std::copy(word_start_idxs_vec.begin(),
              word_start_idxs_vec.begin() + static_cast<long>(copy_n),
              word_start_input.begin());
  }
  return word_start_input;
}

/**
 * @brief Parse labels from schema header based on "[E]" marker tokens.
 *
 * We scan input tokens in the header range [0, text_offset) and find tokens
 * whose decoded string contains "[E]". Each occurrence is matched in order to
 * the configured label list.
 */
LabelParseResult parse_labels_from_header(const std::vector<int> &input_tokens_lower,
                                                 int text_offset,
                                                 const std::vector<std::string> &labels,
                                                 tokenizers::Tokenizer *tokenizer) {
  LabelParseResult res;
  res.label_idxs.assign(kMaxLabels, 0.0f);
  res.label_slot_idxs.assign(kMaxLabels, 0.0f);

  if (!tokenizer || labels.empty() || input_tokens_lower.empty() || text_offset <= 0) {
    res.found_cnt = 0;
    return res;
  }

  const int max_scan = std::min(text_offset, static_cast<int>(input_tokens_lower.size()));
  const int num_labels = static_cast<int>(labels.size());

  for (int i = 0; i < max_scan; ++i) {
    std::string decoded = tokenizer->Decode({input_tokens_lower[i]});
    if (decoded.find("[E]") != std::string::npos) {
      if (res.found_cnt >= num_labels || res.found_cnt >= kMaxLabels) break;
      res.label_idxs[res.found_cnt] = static_cast<float>(i);
      res.valid_labels.push_back(labels[res.found_cnt]);
      res.found_cnt++;
    }
  }

  for (int i = 0; i < res.found_cnt && i < kMaxLabels; ++i) {
    res.label_slot_idxs[i] = static_cast<float>(i);
  }
  return res;
}

/**
 * @brief Get raw output buffer pointer for a named layer output(0).
 *
 * nntrainer's Model::getLayer takes const char* and returns a Layer handle.
 * We cast it to LayerNode to access tensors.
 */
// float *get_layer_output_ptr(ml::train::Model *model, const std::string &layer_name) {
//   if (!model) return nullptr;

//   try {
//     std::shared_ptr<ml::train::Layer> layer_ptr;
//     model->getLayer(layer_name.c_str(), &layer_ptr);
//     auto node = std::dynamic_pointer_cast<nntrainer::LayerNode>(layer_ptr);
//     if (!node) return nullptr;
//     return node->getOutput(0).getData();
//   } catch (const std::exception &e) {
//     std::cerr << "error while getting layout output! details: "
//               << e.what() << std::endl;
//     return nullptr;
//   }
// }

/**
 * @brief Print results as a Python-like dictionary:
 *
 * {'entities': {'LabelA': ['x','y'], 'LabelB': [...]}}
 */
void print_entities_python_like(
   const std::unordered_map<std::string, std::vector<EntityCandidate>> &final_by_label) {
  std::cout << "{'entities': {";

  bool first_label = true;
  for (const auto &kv : final_by_label) {
    const auto &label = kv.first;
    const auto &ents = kv.second;

    // Remove duplicates within the same label by surface string.
    std::unordered_set<std::string> seen;
    std::vector<std::string> uniq;
    uniq.reserve(ents.size());
    for (const auto &e : ents) {
      if (seen.insert(e.text).second) uniq.push_back(e.text);
    }

    if (!first_label) std::cout << ", ";
    first_label = false;

    std::cout << "'" << label << "': [";
    for (size_t i = 0; i < uniq.size(); ++i) {
      if (i) std::cout << ", ";
      std::cout << "'" << uniq[i] << "'";
    }
    std::cout << "]";
  }

  std::cout << "}}\n";
}

} // namespace

void GLiner2MultiV1::constructModel() {
  DebertaV2::constructModel();

  std::vector<LayerHandle> layers;

  const std::string encoder_last_layer =
    "layer" + std::to_string(NUM_LAYERS - 1) + "_output";

  // Preserve last encoder output for downstream heads that need the original states.
  layers.push_back(createLayer("activation",
                               {withKey("name", "preserved_encoder_output"),
                                withKey("activation", "none"),
                                withKey("input_layers", encoder_last_layer)}));

  // Span tensors are provided as flattened vectors of size MAX_SEQ_LEN * MAX_WIDTH.
  const unsigned int num_spans = MAX_SEQ_LEN * MAX_WIDTH;
  const std::string span_input_shape = "1:1:" + std::to_string(num_spans);

  layers.push_back(createLayer("input",
                               {withKey("name", "span_start_idxs"),
                                withKey("input_shape", span_input_shape)}));
  layers.push_back(createLayer("input",
                               {withKey("name", "span_end_idxs"),
                                withKey("input_shape", span_input_shape)}));
  layers.push_back(createLayer("input",
                               {withKey("name", "label_idxs_input"),
                                withKey("input_shape", "1:1:20")}));
  layers.push_back(createLayer("input",
                               {withKey("name", "label_slot_idxs_input"),
                                withKey("input_shape", "1:1:20")}));
  layers.push_back(createLayer("input",
                               {withKey("name", "word_start_idxs"),
                                withKey("input_shape", "1:1:512")}));

  const int hidden_size = DIM;

  // Word pooling: gather encoder states at word boundary token indices.
  layers.push_back(createLayer("causallm_gather",
                               {withKey("name", "word_pooling"),
                                withKey("input_layers", encoder_last_layer + ",word_start_idxs"),
                                withKey("axis", "2")}));

  // Span representation projections (start/end) computed from pooled word states.
  auto proj_start_layers =
    createMlp("span_rep_project_start", hidden_size, INTERMEDIATE_SIZE, "word_pooling");
  auto proj_end_layers =
    createMlp("span_rep_project_end", hidden_size, INTERMEDIATE_SIZE, "word_pooling");

  layers.insert(layers.end(), proj_start_layers.begin(), proj_start_layers.end());
  layers.push_back(createLayer("activation",
                               {withKey("name", "span_rep_project_start_relu"),
                                withKey("activation", "relu"),
                                withKey("input_layers", "span_rep_project_start_ffn_fc2")}));
  const std::string proj_start_out = "span_rep_project_start_relu";

  layers.insert(layers.end(), proj_end_layers.begin(), proj_end_layers.end());
  layers.push_back(createLayer("activation",
                               {withKey("name", "span_rep_project_end_relu"),
                                withKey("activation", "relu"),
                                withKey("input_layers", "span_rep_project_end_ffn_fc2")}));
  const std::string proj_end_out = "span_rep_project_end_relu";

  // Gather projected start/end vectors for each span index list.
  layers.push_back(createLayer("causallm_gather",
                               {withKey("name", "span_rep_gather_start"),
                                withKey("input_layers", proj_start_out + ",span_start_idxs"),
                                withKey("axis", "2")}));
  layers.push_back(createLayer("causallm_gather",
                               {withKey("name", "span_rep_gather_end"),
                                withKey("input_layers", proj_end_out + ",span_end_idxs"),
                                withKey("axis", "2")}));

  // Concatenate start/end span vectors along feature axis.
  layers.push_back(createLayer("concat",
                               {withKey("name", "span_rep_concat"),
                                withKey("input_layers", "span_rep_gather_start,span_rep_gather_end"),
                                withKey("axis", "3")}));

  // Final projection to DIM for span vectors.
  auto proj_out_layers =
    createMlp("span_rep_project_out", hidden_size, INTERMEDIATE_SIZE, "span_rep_concat");
  layers.insert(layers.end(), proj_out_layers.begin(), proj_out_layers.end());

  // Count predictor head (predicts 20-slot schema count embeddings).
  auto count_pred_layers =
    createMlp("count_pred", 20, hidden_size * 2, encoder_last_layer);
  layers.insert(layers.end(), count_pred_layers.begin(), count_pred_layers.end());
  const std::string count_pred_out = "count_pred_ffn_fc2";

  // Count embedding: lookup 'rel_embeddings' with predicted slots.
  layers.push_back(createLayer("weight",
                               {withKey("name", "count_embed_pos_embedding"),
                                withKey("input_layers", count_pred_out),
                                withKey("weight_name", "rel_embeddings"),
                                withKey("dim", "1:1:20:" + std::to_string(DIM)),
                                withKey("input_shape", "1:1:20:" + std::to_string(DIM)),
                                withKey("weight_initializer", "none")}));

  // Gather schema embeddings from encoder output at label token indices.
  layers.push_back(createLayer("causallm_gather",
                               {withKey("name", "gather_schema_emb"),
                                withKey("input_layers", "preserved_encoder_output,label_idxs_input"),
                                withKey("axis", "2")}));

  // GRU over (count_embed_pos_embedding, gather_schema_emb) to mix slot + schema.
  const std::string gru_layer_name = "count_embed_gru";
  layers.push_back(createLayer("custom_gru",
                               {withKey("name", gru_layer_name),
                                withKey("unit", DIM),
                                withKey("disable_bias", "false"),
                                withKey("return_sequences", "true"),
                                withKey("reset_after", "true"),
                                withKey("input_layers", "count_embed_pos_embedding,gather_schema_emb")}));

  // Gather only active label slots.
  layers.push_back(createLayer("causallm_gather",
                               {withKey("name", "gather_gru_labels"),
                                withKey("input_layers", gru_layer_name + ",label_slot_idxs_input"),
                                withKey("axis", "2")}));
  layers.push_back(createLayer("causallm_gather",
                               {withKey("name", "gather_schema_labels"),
                                withKey("input_layers", "gather_schema_emb,label_slot_idxs_input"),
                                withKey("axis", "2")}));

  // Concatenate GRU outputs + schema embeddings.
  layers.push_back(createLayer("concat",
                               {withKey("name", "count_embed_concat"),
                                withKey("input_layers", "gather_gru_labels,gather_schema_labels"),
                                withKey("axis", "3")}));

  // Project concatenated label vectors back to DIM.
  auto count_embed_mlp_layers =
    createMlp("count_embed_mlp", hidden_size, hidden_size * 4, "count_embed_concat");
  layers.insert(layers.end(), count_embed_mlp_layers.begin(), count_embed_mlp_layers.end());

  // (Optional) classifier head kept for parity, not required for entity extraction.
  auto classifier_layers =
    createMlp("classifier", 1, hidden_size * 2, encoder_last_layer);
  layers.insert(layers.end(), classifier_layers.begin(), classifier_layers.end());

  for (auto &layer : layers) model->addLayer(layer);
}

void GLiner2MultiV1::setupParameters(json &cfg, json &generation_cfg, json &nntr_cfg) {
  if (!nntr_cfg.contains("task") || nntr_cfg["task"] != "EntityExtraction") {
    throw nntrainer::exception::not_supported(
    "Only EntityExtraction are currently supported in Gliner2MultiV1");
  }

  try {
    const std::string tokenizer_file = nntr_cfg["tokenizer_file"].get<std::string>();
    const std::filesystem::path tokenizer_path(tokenizer_file);
    const std::string model_path = tokenizer_path.parent_path().string();

    const std::filesystem::path encoder_config_path =
      std::filesystem::path(model_path) / "encoder_config" / "config.json";

    json encoder_cfg = causallm::LoadJsonFile(encoder_config_path.string());
    Transformer::setupParameters(encoder_cfg, generation_cfg, nntr_cfg);

    if (encoder_cfg.contains("max_relative_positions")) {
      MAX_RELATIVE_POSITIONS = encoder_cfg["max_relative_positions"].get<int>();
      if (MAX_RELATIVE_POSITIONS == -1) MAX_RELATIVE_POSITIONS = MAX_POSITION_EMBEDDINGS;
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
          if (!token.empty()) pos_att_type_vec.push_back(token);
        }
      }
      c2p = std::find(pos_att_type_vec.begin(), pos_att_type_vec.end(), "c2p") != pos_att_type_vec.end();
      p2c = std::find(pos_att_type_vec.begin(), pos_att_type_vec.end(), "p2c") != pos_att_type_vec.end();
    } else {
      c2p = false;
      p2c = false;
    }

    SHARE_ATT_KEY = encoder_cfg.contains("share_att_key") ? encoder_cfg["share_att_key"].get<bool>() : false;
    RELATIVE_ATTENTION = encoder_cfg.contains("relative_attention") ? encoder_cfg["relative_attention"].get<bool>() : true;
    POSITION_BUCKETS = encoder_cfg.contains("position_buckets") ? encoder_cfg["position_buckets"].get<int>() : -1;

    // Load GLiNER root config (max_width) if available.
    const std::filesystem::path root_config_path = std::filesystem::path(model_path) / "config.json";
    if (std::filesystem::exists(root_config_path)) {
      json root_cfg = causallm::LoadJsonFile(root_config_path.string());
      if (root_cfg.contains("max_width")) MAX_WIDTH = root_cfg["max_width"].get<int>();
    } else if (cfg.contains("max_width")) {
      MAX_WIDTH = cfg["max_width"].get<int>();
    }

    labels = nntr_cfg["labels"].get<std::vector<std::string>>();
  } catch (const std::exception &e) {
    std::cerr << "\n[!] FATAL ERROR: " << e.what() << "\n";
  }
}

std::vector<LayerHandle> GLiner2MultiV1::createMlp(const std::string &layer_name,
                                                   int dim,
                                                   int hidden_dim,
                                                   std::string input_name) {
  std::vector<LayerHandle> layers;

  layers.push_back(createLayer("fully_connected",
                               {withKey("name", layer_name + "_ffn_fc1"),
                                withKey("unit", hidden_dim),
                                withKey("input_layers", input_name),
                                withKey("disable_bias", "false"),
                                withKey("packed", "false"),
                                withKey("weight_initializer", "ones")}));

  layers.push_back(createLayer("activation",
                               {withKey("name", layer_name + "_ffn_act"),
                                withKey("activation", "relu"),
                                withKey("input_layers", layer_name + "_ffn_fc1")}));

  layers.push_back(createLayer("fully_connected",
                               {withKey("name", layer_name + "_ffn_fc2"),
                                withKey("unit", dim),
                                withKey("disable_bias", "false"),
                                withKey("packed", "false"),
                                withKey("input_layers", layer_name + "_ffn_act"),
                                withKey("weight_initializer", "ones")}));

  return layers;
}

std::pair<std::vector<float>, std::vector<float>>
GLiner2MultiV1::generate_spans(int seq_len, int max_width, int offset) {
  std::vector<float> start_idxs;
  std::vector<float> end_idxs;

  start_idxs.reserve(static_cast<size_t>(seq_len) * static_cast<size_t>(max_width));
  end_idxs.reserve(static_cast<size_t>(seq_len) * static_cast<size_t>(max_width));

  for (int i = 0; i < seq_len; ++i) {
    for (int k = 0; k < max_width; ++k) {
      if (i + k < seq_len) {
        start_idxs.push_back(static_cast<float>(i + offset));
        end_idxs.push_back(static_cast<float>(i + k + offset));
      } else {
        // Padding spans: keep valid shape, but will be filtered by text_len checks.
        start_idxs.push_back(static_cast<float>(offset));
        end_idxs.push_back(static_cast<float>(offset));
      }
    }
  }
  return {start_idxs, end_idxs};
}

std::vector<float *> GLiner2MultiV1::encode(const WSTR prompt,
                                            const WSTR system_prompt,
                                            const WSTR tail_prompt) {
  /**
   * encode() runs the model forward (incremental inference) with all auxiliary inputs
   * required by the GLiNER head. Post-processing / decoding is intentionally not done
   * here so that run() can control output formatting and task-level behavior.
   *
   * Important: This function must call incremental_inference() (project requirement).
   */
  if (!is_initialized) {
    throw std::runtime_error("Model is not initialized. Please call initialize() before encode().");
  }

  // ---------------------------------------------------------------------------
  // 1) Build schema + text (same as run()) so model graph inputs match.
  // ---------------------------------------------------------------------------
  const std::string P_TOKEN = "[P]";
  const std::string E_TOKEN = "[E]";
  const std::string SEP_TOKEN = "[SEP_TEXT]";
  const std::string CONTEXT_WORD = "entities";

  std::string schema_str = "( " + P_TOKEN + " " + CONTEXT_WORD + " (";
  for (const auto &label : labels) schema_str += " " + E_TOKEN + " " + label;
  schema_str += " ) )";

#if defined(_WIN32)
  std::wstring prompt_ = system_prompt + prompt + tail_prompt;
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  std::string text_str = converter.to_bytes(prompt_);
#else
  (void)system_prompt;
  (void)tail_prompt;
  std::string text_str = prompt;
#endif

  text_str = normalize_punct_spacing(text_str);

  const std::string full_input = schema_str + " " + SEP_TOKEN + " " + text_str;
  const std::string full_lower_input = schema_str + " " + SEP_TOKEN + " " + to_lower_copy(text_str);

  // text_offset: header tokens + one [CLS] token.
  const std::string header_str = schema_str + " " + SEP_TOKEN;
  const auto header_tokens = tokenizer->Encode(header_str, false);
  int text_offset = static_cast<int>(header_tokens.size()) + 1;

  const auto full_lower_tokens = tokenizer->Encode(full_lower_input, true);
  int text_len = static_cast<int>(full_lower_tokens.size()) - text_offset;
  if (text_len < 0) text_len = 0;
  if (text_len > static_cast<int>(MAX_SEQ_LEN) - text_offset)
    text_len = static_cast<int>(MAX_SEQ_LEN) - text_offset;

  auto input_origin = tokenizer->Encode(full_input, true);
  auto input_lower = tokenizer->Encode(full_lower_input, true);

  // Handle tokenizer that prepends [CLS].
  const int cls_id = tokenizer->TokenToId("[CLS]");
  if (!input_lower.empty() && input_lower.front() == cls_id) {
    input_lower.erase(input_lower.begin());
    if (text_offset > 0) {
      text_offset--;
      text_len--;
    }
  }
  if (!input_origin.empty() && input_origin.front() == cls_id) input_origin.erase(input_origin.begin());

  // Ensure [EOS] exists if applicable.
  const int eos_id = tokenizer->TokenToId("[EOS]");
  if (!input_lower.empty()) {
    const int last = input_lower.back();
    if (eos_id != -1 && last != eos_id) {
      input_lower.push_back(eos_id);
      text_len++;
    }
  }
  if (!input_origin.empty()) {
    const int last = input_origin.back();
    if (eos_id != -1 && last != eos_id) input_origin.push_back(eos_id);
  }

  // init_seq_len in incremental_inference is token length.
  const unsigned int input_len =
    std::min(static_cast<unsigned int>(input_lower.size() > 0 ? input_lower.size() - 1 : 0),
             static_cast<unsigned int>(MAX_SEQ_LEN));

  // Token-id input tensor: [B, MAX_SEQ_LEN] (float IDs for nntrainer input).
  std::vector<float> input_sample(static_cast<size_t>(BATCH_SIZE) * MAX_SEQ_LEN, 0.0f);
  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    for (unsigned int i = 0; i < input_len; ++i) {
      input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN + i] = static_cast<float>(input_lower[i]);
    }
  }

  // Word boundary lists (lowercase) for word_start_idxs.
  const std::string text_lower = extract_text_after_sep(full_lower_input, SEP_TOKEN);
  const auto words_lower = split_ws(text_lower);
  const int num_words = static_cast<int>(words_lower.size());

  std::vector<float> word_start_input =
    build_word_start_idxs(words_lower, text_offset, tokenizer.get(), static_cast<int>(MAX_SEQ_LEN));

  // Word-level spans (num_words * MAX_WIDTH).
  auto spans = generate_spans(num_words, MAX_WIDTH, 0);
  const auto &start_idxs_vec = spans.first;
  const auto &end_idxs_vec = spans.second;
  const size_t num_spans = start_idxs_vec.size();

  // Flattened span inputs are fixed buffers sized MAX_SEQ_LEN * MAX_WIDTH.
  const size_t required = static_cast<size_t>(MAX_SEQ_LEN) * static_cast<size_t>(MAX_WIDTH);
  std::vector<float> start_input(required, 0.0f);
  std::vector<float> end_input(required, 0.0f);
  std::copy(start_idxs_vec.begin(), start_idxs_vec.end(), start_input.begin());
  std::copy(end_idxs_vec.begin(), end_idxs_vec.end(), end_input.begin());

  // Parse schema label anchors from header tokens.
  std::vector<int> input_lower_i32;
  input_lower_i32.reserve(input_lower.size());
  for (int t : input_lower) input_lower_i32.push_back(t);

  const auto label_info =
    parse_labels_from_header(input_lower_i32, text_offset, labels, tokenizer.get());

  std::vector<float> label_idxs(label_info.label_idxs.begin(), label_info.label_idxs.end());
  std::vector<float> label_slot_idxs(label_info.label_slot_idxs.begin(), label_info.label_slot_idxs.end());

  // Model input order must match graph definition.
  std::vector<float *> inputs;
  inputs.reserve(6);
  inputs.push_back(word_start_input.data());  // word_start_idxs
  inputs.push_back(label_slot_idxs.data());   // label_slot_idxs_input
  inputs.push_back(label_idxs.data());        // label_idxs_input
  inputs.push_back(end_input.data());         // span_end_idxs
  inputs.push_back(start_input.data());       // span_start_idxs
  inputs.push_back(input_sample.data());      // token ids

  /**
   * custom_to_map:
   * key: a logical custom layer group name (e.g., "span_rep", "count_embed")
   * value: the custom 'to' length to apply for that group during incremental inference.
   *
   * This allows overriding the global 'to' for specific subgraphs whose effective
   * sequence length differs from token sequence length (e.g., span count).
   */
  std::unordered_map<std::string, unsigned int> custom_to_map = {
    {"span_rep", static_cast<unsigned int>(num_spans)},
    {"count_embed", 20},
  };

  std::vector<float *> label_dummy;

  auto start_prefill = std::chrono::high_resolution_clock::now();

  std::vector<float *> outputs =
    model->incremental_inference(BATCH_SIZE, inputs, label_dummy, input_len, 0, input_len, true, &custom_to_map);

  auto finish_prefill = std::chrono::high_resolution_clock::now();
  auto prefill_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    finish_prefill - start_prefill);

  std::cout << "prefill: " << input_len << " tokens, "
            << prefill_duration.count() << " ms, "
            << ((double)input_len / prefill_duration.count() * 1000)
            << " TPS\n";
  return outputs;
}

void GLiner2MultiV1::run(const WSTR prompt, bool do_sample, const WSTR system_prompt,
                         const WSTR tail_prompt) {
  try {
    // 1) Run inference via encode() (encode() MUST call incremental_inference()).
    std::vector<float*> model_outputs = encode(prompt, system_prompt, tail_prompt);

    // 2) Recompute lightweight metadata needed for decoding (no header changes required).
    const std::string P_TOKEN = "[P]";
    const std::string E_TOKEN = "[E]";
    const std::string SEP_TOKEN = "[SEP_TEXT]";
    const std::string CONTEXT_WORD = "entities";

    std::string schema_str = "( " + P_TOKEN + " " + CONTEXT_WORD + " (";
    for (const auto &label : labels) schema_str += " " + E_TOKEN + " " + label;
    schema_str += " ) )";

#if defined(_WIN32)
    std::wstring prompt_ = system_prompt + prompt + tail_prompt;
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::string text_str = converter.to_bytes(prompt_);
#else
    (void)system_prompt;
    (void)tail_prompt;
    std::string text_str = prompt;
#endif

    text_str = normalize_punct_spacing(text_str);

    const std::string full_input = schema_str + " " + SEP_TOKEN + " " + text_str;
    const std::string full_lower_input = schema_str + " " + SEP_TOKEN + " " + to_lower_copy(text_str);

    // text_offset: header tokens + [CLS].
    const std::string header_str = schema_str + " " + SEP_TOKEN;
    const auto header_tokens = tokenizer->Encode(header_str, false);
    int text_offset = static_cast<int>(header_tokens.size()) + 1;

    const auto full_lower_tokens = tokenizer->Encode(full_lower_input, true);
    int text_len = static_cast<int>(full_lower_tokens.size()) - text_offset;
    if (text_len < 0) text_len = 0;
    if (text_len > static_cast<int>(MAX_SEQ_LEN) - text_offset)
      text_len = static_cast<int>(MAX_SEQ_LEN) - text_offset;

    auto input_lower = tokenizer->Encode(full_lower_input, true);
    const int cls_id = tokenizer->TokenToId("[CLS]");
    if (!input_lower.empty() && input_lower.front() == cls_id) {
      input_lower.erase(input_lower.begin());
      if (text_offset > 0) {
        text_offset--;
        text_len--;
      }
    }

    // Word lists for surface form and offset computation:
    // - words_orig: preserve casing for final output
    // - words_lower: used for span generation length
    const std::string text_orig = extract_text_after_sep(full_input, SEP_TOKEN);
    const std::string text_lower = extract_text_after_sep(full_lower_input, SEP_TOKEN);
    const auto words_orig = split_ws(text_orig);
    const auto words_lower = split_ws(text_lower);
    const int num_words = static_cast<int>(words_lower.size());

    // Spans must match those used by encode() (same num_words/MAX_WIDTH).
    auto spans = generate_spans(num_words, MAX_WIDTH, 0);
    const auto &start_idxs_vec = spans.first;
    const auto &end_idxs_vec = spans.second;
    const size_t num_spans = start_idxs_vec.size();

    // Parse labels again (same logic as encode()) so valid label list matches.
    std::vector<int> input_lower_i32;
    input_lower_i32.reserve(input_lower.size());
    for (int t : input_lower) input_lower_i32.push_back(t);

    const auto label_info =
      parse_labels_from_header(input_lower_i32, text_offset, labels, tokenizer.get());
    const size_t num_labels = label_info.valid_labels.size();

    // 3) Fetch intermediate outputs produced by encode()->incremental_inference().
    float *raw_span_vecs = model_outputs[0];
    float *raw_label_vecs = model_outputs[1];

    if (num_spans == 0 || num_labels == 0 || !raw_span_vecs || !raw_label_vecs) {
      if (num_labels == 0) std::cout << "[WARN] No labels found in prompt (Check [E] token)." << std::endl;
      return;
    }

    // 4) Compute logits = span_tensor * label_tensor^T.
    nntrainer::TensorDim span_dim(1, 1, num_spans, DIM);
    nntrainer::Tensor span_tensor(span_dim);
    std::memcpy(span_tensor.getData(), raw_span_vecs, num_spans * DIM * sizeof(float));

    nntrainer::TensorDim label_dim(1, 1, num_labels, DIM);
    nntrainer::Tensor label_tensor(label_dim);
    std::memcpy(label_tensor.getData(), raw_label_vecs, num_labels * DIM * sizeof(float));

    nntrainer::TensorDim score_dim(1, 1, num_spans, num_labels);
    nntrainer::Tensor score_tensor(score_dim);
    span_tensor.dot(label_tensor, score_tensor, false, true);

    float *logits = score_tensor.getData();

    // 5) Build candidates by thresholding sigmoid(logit).
    std::vector<EntityCandidate> candidates;
    candidates.reserve(num_spans); // rough

    for (size_t i = 0; i < num_spans; ++i) {
      const int start_pos = static_cast<int>(i / static_cast<size_t>(MAX_WIDTH));
      const int span_width = static_cast<int>(i % static_cast<size_t>(MAX_WIDTH));

      // Filter invalid spans beyond effective text length.
      if (start_pos + span_width >= text_len) continue;

      const int s_word_idx = static_cast<int>(start_idxs_vec[i]);
      const int e_word_idx = static_cast<int>(end_idxs_vec[i]);
      if (s_word_idx < 0 || e_word_idx < s_word_idx) continue;
      if (s_word_idx >= num_words || e_word_idx >= num_words) continue;

      for (size_t j = 0; j < num_labels; ++j) {
        const float prob = sigmoid(logits[i * num_labels + j]);
        if (prob <= kProbThreshold) continue;

        std::string surface =
          normalize_entity_text(join_words(words_orig, s_word_idx, e_word_idx));
        candidates.push_back(EntityCandidate{s_word_idx, e_word_idx, label_info.valid_labels[j], prob, surface});
      }
    }

    // 6) Greedy decode per label and print Python-like dictionary.
    std::unordered_map<std::string, std::vector<EntityCandidate>> by_label;
    for (auto &c : candidates) by_label[c.label].push_back(c);

    std::unordered_map<std::string, std::vector<EntityCandidate>> final_by_label;
    for (auto &kv : by_label) final_by_label[kv.first] = greedy_decode_one_label(kv.second);

    print_entities_python_like(final_by_label);

  } catch (const std::exception &e) {
    std::cerr << "Error during embedding run: " << e.what() << std::endl;
  }
}

void GLiner2MultiV1::registerCustomLayers() {
  DebertaV2::registerCustomLayers();

  const auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(nntrainer::createLayer<causallm::GatherLayer>);
    app_context->registerFactory(nntrainer::createLayer<causallm::CustomGRULayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what() << std::endl;
  }
}

} // namespace causallm
