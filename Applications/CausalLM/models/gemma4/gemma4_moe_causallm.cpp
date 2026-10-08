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
#ifdef ENABLE_HEXKL
  // [plan 201 S4, #260] moe_engine=htp: #4415's lfm2_moe layer runs the
  // experts; their GeGLU goes into the session's MoE word too, which the
  // one-PD token's MOE and DENSE_FFN ops read (#4415's prefill call ORs it
  // in per call anyway)
  if (MOE_ENGINE == "htp")
    nntrainer::get_htp_ops()->set_moe_geglu(true);
  // [plan 201 S4] NNTR_HTP_E2E=1: the whole decode token on the HTP, one
  // call a token; the list is built after load (load_weight)
  const char *e2e_env = std::getenv("NNTR_HTP_E2E");
  htp_e2e =
    e2e_env != nullptr && std::atoi(e2e_env) != 0 && MOE_ENGINE == "htp";
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
  // The weights by name, "<layer>:<weight>" (#4415's fused layers hold
  // several each: qkv's gammas and projections, dense_ffn's gate / up /
  // down and its two gammas, residual_add's gamma and the block's scalar),
  // never by index. Keyed under the owning layer too: a shared_from layer
  // (the tied head) names its own weights after the layer it shares with.
  std::map<std::string, nntrainer::Tensor *> w;
  model->forEachLayer(
    [&w](ml::train::Layer &l, nntrainer::RunLayerContext &rc, void *) {
      for (auto *t : rc.getWeights()) {
        const std::string &n = t->getName();
        w[n] = &t->getVariableRef();
        const size_t colon = n.rfind(':');
        if (colon != std::string::npos)
          w.emplace(l.getName() + n.substr(colon), &t->getVariableRef());
      }
    },
    nullptr);
  auto weight = [&w](const std::string &name) -> nntrainer::Tensor & {
    auto it = w.find(name);
    if (it == w.end())
      throw std::runtime_error("[Gemma4MoE] NNTR_HTP_E2E: no weight " + name);
    return *it->second;
  };
  auto f32 = [&weight](const std::string &name, size_t n) -> const float * {
    nntrainer::Tensor &t = weight(name);
    if (t.getDataType() != ml::train::TensorDim::DataType::FP32 ||
        t.size() != n)
      throw std::runtime_error("[Gemma4MoE] NNTR_HTP_E2E: " + name +
                               " is not " + std::to_string(n) + " FP32 values");
    return t.getData<float>();
  };

  // The list, now that layer_scalar (a checkpoint weight) is in memory.
  const uint32_t n_layers = static_cast<uint32_t>(NUM_LAYERS);
  std::vector<uint8_t> is_full(n_layers);
  std::vector<float> scalar(n_layers);
  for (uint32_t l = 0; l < n_layers; ++l) {
    is_full[l] = !isSlidingAttentionLayer(static_cast<int>(l));
    scalar[l] = f32(
      "layer" + std::to_string(l) + "_post_ffn_norm:scalar_multiplier", 1)[0];
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
  shape.inter_moe = MOE_INTERMEDIATE_SIZE;
  shape.n_experts = NUM_EXPERTS;
  shape.top_k = NUM_EXPERTS_PER_TOK;
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
  // a tied head folds the final norm and the softcap (#4415's
  // FOLD_OUTPUT_NORM), so the DSP caps; an untied one keeps the CPU's
  // logit_softcapping layer after the head, which caps the logits
  shape.softcap = TIE_WORD_EMBEDDINGS ? FINAL_LOGIT_SOFTCAPPING : 0.0f;
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
  // #4415's graph holds all of them in fused layers: the input norm in
  // qkv, the MoE branch's two in the MoE layer (in_norm / out_norm), the
  // dense branch's two in dense_ffn, post-attention and post-FFN in the
  // residual_adds.
  static const char *const kNorms[7] = {
    "_qkv:in_norm_gamma",        "_post_attention_norm:gamma",
    "_sparse_moe:in_norm_gamma", "_sparse_moe:out_norm_gamma",
    "_ffn:in_norm_gamma",        "_ffn:out_norm_gamma",
    "_post_ffn_norm:gamma"};
  const uint32_t n_ops = words[3], H = shape.hidden, E = NUM_EXPERTS;
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
      const std::string name = layer < n_layers ? p + kNorms[norm++]
                               : TIE_WORD_EMBEDDINGS
                                 ? std::string("output_of_causallm:gamma")
                                 : std::string("output_norm:gamma");
      hand(i, HTP_GRAPH_PARAM_GAMMA, f32(name, H), H);
    } else if (op->kind == HTP_OP_QK_NORM) {
      const uint32_t hd = op->head_dim;
      std::vector<float> g(2u * hd);
      std::copy_n(f32(p + "_qkv:q_norm_gamma", hd), hd, g.begin());
      std::copy_n(f32(p + "_qkv:k_norm_gamma", hd), hd, g.begin() + hd);
      htp_params.push_back(std::move(g));
      hand(i, HTP_GRAPH_PARAM_GAMMA, htp_params.back().data(), 2u * hd);
    } else if (op->kind == HTP_OP_ROUTER_TOPK) {
      // #4415's lfm2_moe (router_type softmax_scale): router [H][E], its
      // norm's gamma [H] with router.scale * H^-0.5 already folded in the
      // file, the per-expert scale [E]; ROUTER_BIAS is gamma | scale, as
      // the file has them
      const std::string m = p + "_sparse_moe:";
      hand(i, HTP_GRAPH_PARAM_ROUTER_W, f32(m + "gate", size_t(H) * E),
           size_t(H) * E);
      std::vector<float> b(H + E);
      std::copy_n(f32(m + "router_norm_gamma", H), H, b.begin());
      std::copy_n(f32(m + "expert_bias", E), E, b.begin() + H);
      htp_params.push_back(std::move(b));
      hand(i, HTP_GRAPH_PARAM_ROUTER_BIAS, htp_params.back().data(), H + E);
    }
  }

  // The Q4_0 weights of the FC / DENSE_FFN / LM_HEAD ops, in list order:
  // q | k (| v), o; up, gate, down; the tied table.
  // [#260] an FC / dense weight may be QS4CX (fc_layer_dtype QS4CX): it
  // binds the WH handles the prefill registers from the same bytes
  auto q4 = [&weight, ops](const std::string &name, bool tied) {
    nntrainer::Tensor &t = weight(name);
    const auto dt = t.getDataType();
    const bool qs4cx = !tied && dt == ml::train::TensorDim::DataType::QS4CX;
    if (dt != ml::train::TensorDim::DataType::Q4_0 && !qs4cx)
      throw std::runtime_error("[Gemma4MoE] NNTR_HTP_E2E: " + name +
                               " is not Q4_0 or (not tied) QS4CX, the "
                               "resident FC kinds' types");
    const unsigned K = tied ? t.width() : t.height();
    const unsigned N = tied ? t.height() : t.width();
    if (qs4cx ? !ops->add_decode_graph_qs4cx(t.getData<char>(),
                                             t.getScale<float>(), K, N)
              : !ops->add_decode_graph_q4_0(t.getData<char>(), K, N, tied))
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
      q4(p + "_qkv:qweight", false);
      q4(p + "_qkv:kweight", false);
      if (w.count(p + "_qkv:vweight")) // absent under k = v
        q4(p + "_qkv:vweight", false);
    } else if (op->kind == HTP_OP_FC) {
      q4(p + "_attention_out:weight", false);
    } else if (op->kind == HTP_OP_DENSE_FFN) {
      // the list's order up, gate, down; the file's is gate-first
      q4(p + "_ffn:up", false);
      q4(p + "_ffn:gate", false);
      q4(p + "_ffn:down", false);
    } else if (op->kind == HTP_OP_LM_HEAD) {
      q4(TIE_WORD_EMBEDDINGS ? "embedding0:Embedding"
                             : "output_of_causallm:weight",
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

} // namespace causallm
