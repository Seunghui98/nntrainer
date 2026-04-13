// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   lfm2_causallm.cpp
 * @date   13 April 2026
 * @brief  LFM2/LFM2.5 hybrid model implementation
 * @note   LiquidAI LFM2 architecture: double-gated short conv + GQA attention
 * @see    https://github.com/nntrainer/nntrainer
 */

#include <llm_util.hpp>
#include <model.h>
#include <lfm2_causallm.h>

#include <app_context.h>
#include <engine.h>
#include <lfm2_conv.h>
#include <reshaped_rms_norm.h>

namespace causallm {

// ============================================================================
// Parameter setup
// ============================================================================

void LFM2Transformer::setupLFM2Parameters(json &cfg, json &nntr_cfg) {
  unsigned int num_layers = cfg.contains("num_hidden_layers")
                              ? cfg["num_hidden_layers"].get<unsigned int>()
                              : NUM_LAYERS;

  is_attn_layer.resize(num_layers, false);

  // Parse layer_types from config.json (e.g. ["conv","conv","full_attention",...])
  if (cfg.contains("layer_types")) {
    auto layer_types = cfg["layer_types"].get<std::vector<std::string>>();
    for (unsigned int i = 0; i < layer_types.size() && i < num_layers; ++i) {
      is_attn_layer[i] = (layer_types[i] == "full_attention");
    }
  } else if (nntr_cfg.contains("attn_layers")) {
    // Alternative: explicit list of attention layer indices
    auto layers = nntr_cfg["attn_layers"].get<std::vector<unsigned int>>();
    for (auto idx : layers) {
      if (idx < num_layers)
        is_attn_layer[idx] = true;
    }
  }

  // RoPE theta
  if (cfg.contains("rope_parameters") &&
      cfg["rope_parameters"].contains("rope_theta")) {
    ROPE_THETA = cfg["rope_parameters"]["rope_theta"].get<unsigned int>();
  }

  // Conv parameters
  CONV_L_CACHE = cfg.contains("conv_L_cache")
                   ? cfg["conv_L_cache"].get<unsigned int>()
                   : 3;

  // MLP intermediate size with auto-adjustment (matches HF Lfm2MLP)
  unsigned int raw_intermediate = cfg.contains("intermediate_size")
                                    ? cfg["intermediate_size"].get<unsigned int>()
                                    : INTERMEDIATE_SIZE;

  bool auto_adjust = cfg.contains("block_auto_adjust_ff_dim")
                       ? cfg["block_auto_adjust_ff_dim"].get<bool>()
                       : true;
  float ff_multiplier = cfg.contains("block_ffn_dim_multiplier")
                          ? cfg["block_ffn_dim_multiplier"].get<float>()
                          : 1.0f;
  unsigned int multiple_of = cfg.contains("block_multiple_of")
                               ? cfg["block_multiple_of"].get<unsigned int>()
                               : 256;

  if (auto_adjust) {
    raw_intermediate = (unsigned int)(2 * raw_intermediate / 3);
  }
  raw_intermediate = (unsigned int)(ff_multiplier * raw_intermediate);
  ACTUAL_INTERMEDIATE_SIZE =
    multiple_of * ((raw_intermediate + multiple_of - 1) / multiple_of);

  // Check for embedding_norm (LFM2 has it, most other models don't)
  HAS_EMBEDDING_NORM = true; // LFM2 always has embedding_norm
}

// ============================================================================
// MLP (SwiGLU): w2(silu(w1(x)) * w3(x))
// HF names: w1=gate_proj, w3=up_proj, w2=down_proj
// ============================================================================

std::vector<LayerHandle>
LFM2Transformer::createMlp(const int layer_id, int dim, int hidden_dim,
                            std::string input_name) {
  std::vector<LayerHandle> layers;
  auto prefix = "layer" + std::to_string(layer_id);

  // w1 (gate_proj)
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", prefix + "_ffn_gate"),
     withKey("unit", hidden_dim),
     withKey("disable_bias", "true"),
     withKey("input_layers", input_name),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  // w3 (up_proj)
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", prefix + "_ffn_up"),
     withKey("unit", hidden_dim),
     withKey("disable_bias", "true"),
     withKey("input_layers", input_name),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  // swiglu = silu(w1) * w3
  layers.push_back(createLayer(
    "swiglu",
    {withKey("name", prefix + "_ffn_swiglu"),
     withKey("input_layers", prefix + "_ffn_gate," + prefix + "_ffn_up")}));

  // w2 (down_proj)
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", prefix + "_ffn_down"),
     withKey("unit", dim),
     withKey("disable_bias", "true"),
     withKey("input_layers", prefix + "_ffn_swiglu"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  return layers;
}

// ============================================================================
// Attention block (GQA + QK-norm + RoPE)
// ============================================================================

std::vector<LayerHandle>
LFM2Transformer::createAttention(const int layer_id, int seq_len, int n_heads,
                                 int head_dim, std::string query_name,
                                 std::string key_name,
                                 std::string value_name) {
  std::vector<LayerHandle> layers;
  auto prefix = "layer" + std::to_string(layer_id);
  auto V = prefix + "_wv";
  auto K = prefix + "_wk";
  auto K_norm = prefix + "_k_norm";
  auto Q = prefix + "_wq";
  auto Q_norm = prefix + "_q_norm";
  auto A = prefix + "_attention";
  auto O = prefix + "_attention_out";

  unsigned int kv_dim = head_dim * n_heads / GQA_SIZE;
  unsigned int q_dim = head_dim * n_heads;

  // V projection
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", V), withKey("unit", kv_dim),
     withKey("disable_bias", "true"), withKey("input_layers", value_name),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  // K projection
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", K), withKey("unit", kv_dim),
     withKey("disable_bias", "true"), withKey("input_layers", key_name),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  // K RMSNorm (per-head)
  layers.push_back(createLayer(
    "reshaped_rms_norm",
    {withKey("name", K_norm), withKey("input_layers", K),
     withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(head_dim))}));

  // Q projection
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", Q), withKey("unit", q_dim),
     withKey("disable_bias", "true"), withKey("input_layers", query_name),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  // Q RMSNorm (per-head)
  layers.push_back(createLayer(
    "reshaped_rms_norm",
    {withKey("name", Q_norm), withKey("input_layers", Q),
     withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(head_dim))}));

  // MHA core with full RoPE (no partial_rotary_factor, no attn_gate)
  layers.push_back(createLayer(
    "mha_core",
    {withKey("name", A), withKey("num_heads", n_heads),
     withKey("num_heads_kv", n_heads / GQA_SIZE),
     withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
     withKey("sliding_window", SLIDING_WINDOW),
     withKey("rope_theta", ROPE_THETA),
     withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
     withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
     withKey("input_layers", Q_norm + "," + K_norm + "," + V)}));

  // O projection (no attn_gate - direct from attention output)
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", O), withKey("unit", DIM),
     withKey("disable_bias", "true"), withKey("input_layers", A),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  return layers;
}

// ============================================================================
// Self-attention decoder block
// ============================================================================

std::vector<LayerHandle>
LFM2Transformer::createAttentionBlock(const int layer_id,
                                      std::string input_name) {
  std::vector<LayerHandle> layers;
  auto prefix = "layer" + std::to_string(layer_id);
  auto norm_name = prefix + "_operator_norm";

  // Operator norm (attention norm)
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", norm_name),
     withKey("input_layers", input_name),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  // Attention
  auto att_layer =
    createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                    norm_name, norm_name, norm_name);
  layers.insert(layers.end(), att_layer.begin(), att_layer.end());

  // Residual add
  layers.push_back(createLayer(
    "addition",
    {withKey("name", prefix + "_decoder_add"),
     withKey("input_layers",
             input_name + "," + prefix + "_attention_out")}));

  // FFN norm
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", prefix + "_ffn_norm"),
     withKey("input_layers", prefix + "_decoder_add"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  // MLP
  auto ffn_layer = createMlp(layer_id, DIM, ACTUAL_INTERMEDIATE_SIZE,
                             prefix + "_ffn_norm");
  layers.insert(layers.end(), ffn_layer.begin(), ffn_layer.end());

  // Residual add
  layers.push_back(createLayer(
    "addition",
    {withKey("name", prefix + "_decoder_output"),
     withKey("input_layers",
             prefix + "_decoder_add," + prefix + "_ffn_down")}));

  return layers;
}

// ============================================================================
// Conv block (double-gated short convolution)
// ============================================================================

std::vector<LayerHandle>
LFM2Transformer::createConvBlock(const int layer_id, std::string input_name) {
  std::vector<LayerHandle> layers;
  auto prefix = "layer" + std::to_string(layer_id);
  auto norm_name = prefix + "_operator_norm";

  // Operator norm
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", norm_name),
     withKey("input_layers", input_name),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  // LFM2 conv: in_proj(3H) → split B,C,x → B*x → conv1d → C*conv → out_proj
  layers.push_back(createLayer(
    "lfm2_conv",
    {withKey("name", prefix + "_conv"),
     withKey("input_layers", norm_name),
     withKey("hidden_size", DIM),
     withKey("conv_kernel_size", CONV_L_CACHE),
     withKey("packed", "false")}));

  // Residual add
  layers.push_back(createLayer(
    "addition",
    {withKey("name", prefix + "_decoder_add"),
     withKey("input_layers", input_name + "," + prefix + "_conv")}));

  // FFN norm
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", prefix + "_ffn_norm"),
     withKey("input_layers", prefix + "_decoder_add"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  // MLP
  auto ffn_layer = createMlp(layer_id, DIM, ACTUAL_INTERMEDIATE_SIZE,
                             prefix + "_ffn_norm");
  layers.insert(layers.end(), ffn_layer.begin(), ffn_layer.end());

  // Residual add
  layers.push_back(createLayer(
    "addition",
    {withKey("name", prefix + "_decoder_output"),
     withKey("input_layers",
             prefix + "_decoder_add," + prefix + "_ffn_down")}));

  return layers;
}

// ============================================================================
// Full model construction
// ============================================================================

void LFM2Transformer::constructModel() {
  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  std::vector<LayerHandle> layers;

  // Input layer
  layers.push_back(createLayer(
    "input", {withKey("name", "input0"),
              withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))}));

  // Embedding layer
  const std::string embedding_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";
  layers.push_back(createLayer(
    embedding_type,
    {"name=embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
     "weight_dtype=" + EMBEDDING_DTYPE, "out_dim=" + std::to_string(DIM),
     "scale=" + std::to_string(EMBEDDING_SCALE)}));

  // Embedding norm (LFM2 specific - applies RMSNorm after embedding)
  std::string prev_output = "embedding0";
  if (HAS_EMBEDDING_NORM) {
    layers.push_back(createLayer(
      "rms_norm",
      {withKey("name", "embedding_norm"),
       withKey("input_layers", "embedding0"),
       withKey("epsilon", std::to_string(NORM_EPS)),
       withKey("packed", "false")}));
    prev_output = "embedding_norm";
  }

  // Build decoder layers
  for (int i = 0; i < NUM_LAYERS; ++i) {
    std::string input_name =
      (i == 0) ? prev_output
               : ("layer" + std::to_string(i - 1) + "_decoder_output");

    std::vector<LayerHandle> block;
    if (is_attn_layer[i]) {
      block = createAttentionBlock(i, input_name);
    } else {
      block = createConvBlock(i, input_name);
    }
    layers.insert(layers.end(), block.begin(), block.end());
  }

  // Final RMS norm (no separate final norm in LFM2 - the last layer's output
  // goes to lm_head, but HF has model.norm which is actually embedding_norm)
  // Actually looking again, LFM2Model doesn't have a final_norm separate from
  // embedding_norm. The model just outputs the last decoder layer directly.
  // But nntrainer CausalLM base expects an "output_norm" layer before lm_head.
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "output_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("input_layers",
             "layer" + std::to_string(NUM_LAYERS - 1) + "_decoder_output"),
     withKey("packed", "false")}));

  // Add all layers to model
  for (auto &layer : layers) {
    model->addLayer(layer);
  }
}

// ============================================================================
// Custom layer registration
// ============================================================================

void LFM2Transformer::registerCustomLayers() {
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  // Register ReshapedRMSNormLayer (for QK-norm in attention)
  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory: " << e.what() << std::endl;
  }

  // Register LFM2ConvLayer (double-gated short convolution)
  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::LFM2ConvLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory: " << e.what() << std::endl;
  }
}

// ============================================================================
// CausalLM wrapper
// ============================================================================

void LFM2CausalLM::constructModel() {
  LFM2Transformer::constructModel();

  // Add lm_head
  const std::string lmhead_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "lm_head";
  std::vector<std::string> lmhead_prop = {
    withKey("name", "output_of_causallm"),
    withKey("unit", NUM_VOCAB),
    withKey("disable_bias", "true"),
    withKey("input_layers", "output_norm"),
    withKey("weight_dtype", LMHEAD_DTYPE),
  };
  if (TIE_WORD_EMBEDDINGS)
    lmhead_prop.emplace_back(withKey("shared_from", "embedding0"));
  model->addLayer(createLayer(lmhead_type, lmhead_prop));

  // Set model properties
  std::vector<std::string> model_props = {
    withKey("batch_size", BATCH_SIZE), withKey("epochs", "1"),
    withKey("model_tensor_type", MODEL_TENSOR_TYPE)};
  if (MEMORY_SWAP) {
    model_props.emplace_back(withKey("fsu", "true"));
    model_props.emplace_back(withKey("fsu_lookahead", FSU_LOOKAHEAD));
  }
  model->setProperty(model_props);

  // Compile and initialize
  if (model->compile(ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("LFM2 model compilation failed.");
  }
  if (model->initialize(ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("LFM2 model initialization failed.");
  }
}

void LFM2CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  LFM2Transformer::registerCustomLayers();
}

} // namespace causallm
