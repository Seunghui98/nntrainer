// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gemma4_moe_causallm.cpp
 * @brief  Gemma4 MoE causal language model implementation.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs
 */

#include <app_context.h>
#include <engine.h>
#include <gemma4_moe_causallm.h>
#include <gemma4_moe_layer.h>
#include <layer_context.h>
#include <llm_util.hpp>
#include <model.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>

#ifdef ENABLE_HEXKL
#include <compute_ops.h>
#include <htp_graph_desc.h>
#endif

namespace causallm {

void Gemma4MoECausalLM::setupParameters(json &cfg, json &generation_cfg,
                                        json &nntr_cfg) {
  Gemma4CausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);

  NNTR_THROW_IF(!cfg.contains("num_experts") || cfg["num_experts"].is_null() ||
                  !cfg.contains("top_k_experts") ||
                  cfg["top_k_experts"].is_null() ||
                  !cfg.contains("moe_intermediate_size") ||
                  cfg["moe_intermediate_size"].is_null(),
                std::invalid_argument)
    << "[Gemma4MoE] num_experts, top_k_experts, and moe_intermediate_size "
       "must be provided";

  num_experts = cfg["num_experts"].get<unsigned int>();
  top_k_experts = cfg["top_k_experts"].get<unsigned int>();
  moe_intermediate_size = cfg["moe_intermediate_size"].get<unsigned int>();
  moe_cache_size =
    nntr_cfg.contains("moe_cache_size") && !nntr_cfg["moe_cache_size"].is_null()
      ? nntr_cfg["moe_cache_size"].get<unsigned int>()
      : 0;

  NNTR_THROW_IF(num_experts == 0 || top_k_experts == 0 ||
                  top_k_experts > num_experts || moe_intermediate_size == 0,
                std::invalid_argument)
    << "[Gemma4MoE] invalid expert configuration";
  NNTR_THROW_IF(NUM_KV_SHARED_LAYERS > 0, std::invalid_argument)
    << "[Gemma4MoE] shared KV layers are not supported";

  // [plan 201 S4] moe_engine=htp: the experts and the dense FFN are GeGLU
  // on the HTP (the session's MoE flag, sent with the first MoE call)
  moe_engine = nntr_cfg.value("moe_engine", std::string("cpu"));
#ifdef ENABLE_HEXKL
  if (moe_engine == "htp")
    nntrainer::get_htp_ops()->set_moe_geglu(true);
  // [plan 201 S4] NNTR_HTP_E2E=1: the whole decode token on the HTP, one
  // call a token; the list is built after load (load_weight)
  const char *e2e_env = std::getenv("NNTR_HTP_E2E");
  htp_e2e =
    e2e_env != nullptr && std::atoi(e2e_env) != 0 && moe_engine == "htp";
  NNTR_THROW_IF(htp_e2e && (HIDDEN_SIZE_PER_LAYER_INPUT > 0 ||
                            ATTN_LOGIT_SOFTCAPPING > 0.0f),
                std::invalid_argument)
    << "[Gemma4MoE] NNTR_HTP_E2E=1: the decode list has no per-layer input "
       "(PLE) and no attention logit soft-cap";
#endif
}

void Gemma4MoECausalLM::load_weight(const std::string &weight_path) {
  Gemma4CausalLM::load_weight(weight_path);
#ifdef ENABLE_HEXKL
  if (!htp_e2e)
    return;
  // The weights by layer name (the names the graph builders give).
  std::map<std::string, std::vector<nntrainer::Tensor *>> w;
  model->forEachLayer(
    [&w](ml::train::Layer &l, nntrainer::RunLayerContext &rc, void *) {
      for (auto *t : rc.getWeights())
        w[l.getName()].push_back(&t->getVariableRef());
    },
    nullptr);
  auto weight = [&w](const std::string &layer,
                     size_t i) -> nntrainer::Tensor & {
    auto it = w.find(layer);
    if (it == w.end() || it->second.size() <= i)
      throw std::runtime_error("[Gemma4MoE] NNTR_HTP_E2E: layer " + layer +
                               " has no weight " + std::to_string(i));
    return *it->second[i];
  };
  auto f32 = [&weight](const std::string &layer, size_t i,
                       size_t n) -> const float * {
    nntrainer::Tensor &t = weight(layer, i);
    if (t.getDataType() != ml::train::TensorDim::DataType::FP32 ||
        t.size() != n)
      throw std::runtime_error("[Gemma4MoE] NNTR_HTP_E2E: " + layer +
                               " weight " + std::to_string(i) + " is not " +
                               std::to_string(n) + " FP32 values");
    return t.getData<float>();
  };

  // The list, now that layer_scalar (a checkpoint weight) is in memory.
  const uint32_t n_layers = static_cast<uint32_t>(NUM_LAYERS);
  std::vector<uint8_t> is_full(n_layers);
  std::vector<float> scalar(n_layers);
  for (uint32_t l = 0; l < n_layers; ++l) {
    is_full[l] = !isSlidingAttentionLayer(static_cast<int>(l));
    scalar[l] = f32("layer" + std::to_string(l) + "_layer_scalar", 0, 1)[0];
    // the list's ADD stores 0.0f as "no multiplier" (#221)
    if (scalar[l] == 0.0f)
      throw std::runtime_error("[Gemma4MoE] NNTR_HTP_E2E: layer " +
                               std::to_string(l) +
                               "'s layer_scalar is 0, which the decode list "
                               "cannot carry");
  }
  htp_graph_gemma_shape shape{};
  shape.n_layers = n_layers;
  shape.hidden = static_cast<uint32_t>(DIM);
  shape.inter_dense = static_cast<uint32_t>(INTERMEDIATE_SIZE);
  shape.inter_moe = moe_intermediate_size;
  shape.n_experts = num_experts;
  shape.top_k = top_k_experts;
  shape.n_heads = static_cast<uint32_t>(NUM_HEADS);
  shape.n_kv = static_cast<uint32_t>(NUM_KEY_VALUE_HEADS);
  shape.head_dim = static_cast<uint32_t>(HEAD_DIM);
  shape.n_kv_full = ATTENTION_K_EQ_V
                      ? NUM_GLOBAL_KEY_VALUE_HEADS
                      : static_cast<uint32_t>(NUM_KEY_VALUE_HEADS);
  shape.head_dim_full = GLOBAL_HEAD_DIM;
  shape.k_eq_v = ATTENTION_K_EQ_V ? 1u : 0u;
  shape.window = SLIDING_WINDOW;
  shape.vocab = NUM_VOCAB;
  shape.max_seq = MAX_SEQ_LEN;
  shape.eps = NORM_EPS;
  // ponytail: the CPU's logit_softcapping layer after the head caps the
  // logits the HTP hands back, so the list's LM_HEAD caps nothing (the
  // DSP's argmax on raw logits picks the same id: tanh is monotonic). The
  // upgrade: skip that layer at a resident row and cap on the DSP.
  shape.softcap = 0.0f;
  std::vector<uint32_t> words(htp_graph_words_for(n_layers, HTP_GRAPH_MAX_OPS));
  const uint32_t n_words = htp_graph_gemma_build(
    words.data(), static_cast<uint32_t>(words.size()), &shape, is_full.data(),
    scalar.data(), HTP_GRAPH_KINDS_ALL);
  if (n_words == 0u)
    throw std::runtime_error("[Gemma4MoE] NNTR_HTP_E2E: the decode op list "
                             "does not fit HTP_GRAPH_MAX_OPS");
  words.resize(n_words);
  auto *ops = nntrainer::get_htp_ops();
  if (!ops->set_decode_graph_desc(words))
    throw std::runtime_error(
      "[Gemma4MoE] NNTR_HTP_E2E: the HTP backend has no per-token entry");

  // The f32 parameters by name. The RMSNORMs of a layer in the builder's
  // order (htp_graph_gemma_build): input, post-attention, the MoE branch's
  // pre / post, the dense branch's pre / post, post-FFN; the tail's final.
  static const char *const kNorms[7] = {
    "_attention_norm",  "_post_attention_norm", "_pre_ffn_norm_2",
    "_post_ffn_norm_2", "_pre_ffn_norm",        "_post_ffn_norm_1",
    "_post_ffn_norm"};
  const uint32_t n_ops = words[3], H = shape.hidden, E = num_experts;
  auto hand = [ops](uint32_t op, uint32_t which, const float *d, size_t n) {
    if (!ops->set_decode_graph_param(op, which, d, static_cast<unsigned>(n)))
      throw std::runtime_error(
        "[Gemma4MoE] NNTR_HTP_E2E: the backend took no parameter");
  };
  htp_params.clear();
  uint32_t layer = HTP_GRAPH_NO_OP, norm = 0;
  for (uint32_t i = 0; i < n_ops; ++i) {
    const htp_graph_op *op = htp_graph_op_cat(words.data(), i);
    if (op->layer != layer) {
      layer = op->layer;
      norm = 0;
    }
    const std::string p = "layer" + std::to_string(layer);
    if (op->kind == HTP_OP_RMSNORM) {
      if (layer < n_layers && norm == 7u)
        throw std::runtime_error("[Gemma4MoE] NNTR_HTP_E2E: layer " + p +
                                 " has more RMSNORM ops than names");
      const std::string name =
        layer == n_layers ? std::string("output_norm") : p + kNorms[norm++];
      hand(i, HTP_GRAPH_PARAM_GAMMA, f32(name, 0, H), H);
    } else if (op->kind == HTP_OP_QK_NORM) {
      const uint32_t hd = op->head_dim;
      std::vector<float> g(2u * hd);
      std::copy_n(f32(p + "_q_norm", 0, hd), hd, g.begin());
      std::copy_n(f32(p + "_k_norm", 0, hd), hd, g.begin() + hd);
      htp_params.push_back(std::move(g));
      hand(i, HTP_GRAPH_PARAM_GAMMA, htp_params.back().data(), 2u * hd);
    } else if (op->kind == HTP_OP_ROUTER_TOPK) {
      // lfm2_moe (softmax): router [H][E], router_scale [H], per-expert
      // scale [E]; ROUTER_BIAS is g | per-expert scale with g the layer's
      // own router_scale / sqrt(H) (Lfm2MoELayer::route)
      const std::string m = p + "_sparse_moe";
      hand(i, HTP_GRAPH_PARAM_ROUTER_W, f32(m, 0, size_t(H) * E),
           size_t(H) * E);
      const float *rs = f32(m, 1, H), *pes = f32(m, 2, E);
      const float hs = 1.0f / std::sqrt(static_cast<float>(H));
      std::vector<float> b(H + E);
      for (uint32_t f = 0; f < H; ++f)
        b[f] = rs[f] * hs;
      std::copy_n(pes, E, b.begin() + H);
      htp_params.push_back(std::move(b));
      hand(i, HTP_GRAPH_PARAM_ROUTER_BIAS, htp_params.back().data(), H + E);
    }
  }

  // The Q4_0 weights of the FC / DENSE_FFN / LM_HEAD ops, in list order:
  // q | k (| v), o; up, gate, down; the tied table.
  auto q4 = [&weight, ops](const std::string &layer, bool tied) {
    nntrainer::Tensor &t = weight(layer, 0);
    if (t.getDataType() != ml::train::TensorDim::DataType::Q4_0)
      throw std::runtime_error("[Gemma4MoE] NNTR_HTP_E2E: " + layer +
                               " is not Q4_0 (the resident FC kinds' type)");
    const unsigned K = tied ? t.width() : t.height();
    const unsigned N = tied ? t.height() : t.width();
    if (!ops->add_decode_graph_q4_0(t.getData<char>(), K, N, tied))
      throw std::runtime_error(
        "[Gemma4MoE] NNTR_HTP_E2E: the backend took no Q4_0 weight");
  };
  layer = HTP_GRAPH_NO_OP;
  uint32_t fc = 0;
  for (uint32_t i = 0; i < n_ops; ++i) {
    const htp_graph_op *op = htp_graph_op_cat(words.data(), i);
    if (op->layer != layer) {
      layer = op->layer;
      fc = 0;
    }
    const std::string p = "layer" + std::to_string(layer);
    if (op->kind == HTP_OP_FC && fc++ == 0) {
      q4(p + "_wq", false);
      q4(p + "_wk", false);
      if (w.count(p + "_wv"))
        q4(p + "_wv", false);
    } else if (op->kind == HTP_OP_FC) {
      q4(p + "_attention_out", false);
    } else if (op->kind == HTP_OP_DENSE_FFN) {
      q4(p + "_ffn_up", false);
      q4(p + "_ffn_gate", false);
      q4(p + "_ffn_down", false);
    } else if (op->kind == HTP_OP_LM_HEAD) {
      q4(TIE_WORD_EMBEDDINGS ? "embedding0" : "output_of_causallm",
         TIE_WORD_EMBEDDINGS);
    }
  }
  std::fprintf(stderr,
               "[HTP] gemma: list n_ops=%u layers=%u params=%zu by name\n",
               n_ops, n_layers, htp_params.size());
#endif
}

void Gemma4MoECausalLM::repack_weight() {
  Gemma4CausalLM::repack_weight();
#ifdef ENABLE_HEXKL
  // [plan 201 S4] the E2E FC arena, after the experts' registration above
  if (htp_e2e)
    nntrainer::get_htp_ops()->finish_decode_graph_q4_0();
#endif
}

Tensor Gemma4MoECausalLM::createFeedForwardBlock(const int layer_id,
                                                 Tensor post_attention,
                                                 bool is_kv_shared_layer) {
  std::vector<std::string> pre_ffn_norm_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_pre_ffn_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  appendSkipPrefillIfNeeded(pre_ffn_norm_props, is_kv_shared_layer);
  LayerHandle pre_ffn_norm(createLayer("rms_norm", pre_ffn_norm_props));
  Tensor dense_input = pre_ffn_norm(post_attention);
  Tensor dense_output =
    createMlp(layer_id, DIM, INTERMEDIATE_SIZE, dense_input);

  LayerHandle post_dense_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_post_ffn_norm_1"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor post_dense = post_dense_norm(dense_output);

  LayerHandle pre_sparse_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_pre_ffn_norm_2"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor sparse_input = pre_sparse_norm(post_attention);
  LayerHandle sparse_moe(createLayer(
    "gemma4_moe",
    {withKey("name", "layer" + std::to_string(layer_id) + "_sparse_moe"),
     withKey("unit", std::to_string(moe_intermediate_size)),
     withKey("num_experts", std::to_string(num_experts)),
     withKey("num_experts_per_token", std::to_string(top_k_experts)),
     withKey("moe_cache_size", std::to_string(moe_cache_size)),
     withKey("moe_activation", "tanh_gelu"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  Tensor sparse_output = sparse_moe({sparse_input, post_attention});

  LayerHandle post_sparse_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_post_ffn_norm_2"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor post_sparse = post_sparse_norm(sparse_output);
  LayerHandle combine_ffn(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_combine_ffn")}));
  Tensor combined_ffn = combine_ffn({post_dense, post_sparse});

  LayerHandle post_combined_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_post_ffn_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  return post_combined_norm(combined_ffn);
}

void Gemma4MoECausalLM::registerCustomLayers() {
  Gemma4CausalLM::registerCustomLayers();
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));
  try {
    app_context->registerFactory(nntrainer::createLayer<Gemma4MoELayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

} // namespace causallm
