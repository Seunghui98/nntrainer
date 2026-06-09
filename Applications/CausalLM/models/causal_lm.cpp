// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 * Copyright (C) 2025 Seungback Hong <sb92.hong@samsung.com>
 * Copyright (C) 2025 Hyeonseok Lee <hs89.lee@samsung.com>
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   causal_lm.cpp
 * @date   10 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file defines CausalLM's basic actions
 * @note   This causal_lm.h constructs a class for Transformer-based Causal
 * Language Model (CausalLM). It aims to support AutoModelForCausalLM with
 * nntrainer. It supports the following models:
 *          - Llama
 */

#include <algorithm>
#include <app_context.h>
#include <cmath>
#include <cstdlib>
#include <engine.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

#include <common.h>
#include <ddtree.h>
#include <ddtree_sliding.h>
#include <ddtree_types.h>
#include <layer_context.h>
#include <lm_head.h>
#include <mha_core.h>
#include <nntrainer_error.h>
#include <tensor.h>
#include <tie_word_embedding.h>

#include <causal_lm.h>
#include <llm_util.hpp>

namespace causallm {

CausalLM::CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
  Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM) {
  setupParameters(cfg, generation_cfg, nntr_cfg);
}

void CausalLM::setupParameters(json &cfg, json &generation_cfg,
                               json &nntr_cfg) {
  // Initialize output list
  for (unsigned int i = 0; i < BATCH_SIZE; ++i)
    output_list.push_back("");

  // allocate memory for the internal buffer
  ids_history = (unsigned int *)malloc(static_cast<size_t>(BATCH_SIZE) *
                                       MAX_SEQ_LEN * sizeof(unsigned int));

  BAD_WORD_IDS = nntr_cfg["bad_word_ids"].get<std::vector<unsigned int>>();
  NUM_BADWORDS = BAD_WORD_IDS.size();

  LMHEAD_DTYPE = nntr_cfg.contains("lmhead_dtype")
                   ? nntr_cfg["lmhead_dtype"]
                   : nntr_cfg["embedding_dtype"];

  SKIP_PREFILL = nntr_cfg.contains("skip_prefill")
                   ? nntr_cfg["skip_prefill"].get<bool>()
                   : false;

  DDTREE_ENABLE =
    nntr_cfg.contains("ddtree") ? nntr_cfg["ddtree"].get<bool>() : false;

  USE_KVCACHE = false;
  PRE_COMPUTED_CACHE_PATH = "";
  SYS_PROMP_LEN = 0;

  if (nntr_cfg.contains("system_prompt") &&
      nntr_cfg["system_prompt"].contains("kvcache")) {
    USE_KVCACHE = true;
    PRE_COMPUTED_CACHE_PATH =
      nntr_cfg["system_prompt"]["kvcache"]["pre_computed_cache_path"];
    if (nntr_cfg["system_prompt"]["kvcache"].contains("sys_prompt_token_size"))
      SYS_PROMP_LEN =
        nntr_cfg["system_prompt"]["kvcache"]["sys_prompt_token_size"]
          .get<unsigned int>();
  }

  if (generation_cfg["eos_token_id"].is_array()) {
    EOS_TOKEN_ID =
      generation_cfg["eos_token_id"].empty()
        ? cfg["eos_token_id"].get<std::vector<unsigned int>>()
        : generation_cfg["eos_token_id"].get<std::vector<unsigned int>>();
  } else {
    EOS_TOKEN_ID.clear();
    EOS_TOKEN_ID.push_back(generation_cfg["eos_token_id"].get<unsigned int>());
  }
  BOS_TOKEN_ID = generation_cfg["bos_token_id"].empty()
                   ? cfg["bos_token_id"].get<unsigned int>()
                   : generation_cfg["bos_token_id"].get<unsigned int>();
  TOP_K = generation_cfg.contains("top_k")
            ? generation_cfg["top_k"].get<unsigned int>()
            : 20;
  TOP_P = generation_cfg.contains("top_p")
            ? generation_cfg["top_p"].get<float>()
            : 0.95;
  TEMPERATURE = generation_cfg.contains("temperature")
                  ? generation_cfg["temperature"].get<float>()
                  : 0.7;
  global_token_len = 0;
}

void CausalLM::allocateAndBindKVCache() {
  if (!kv_cache.isAllocated()) {
    // dtype matches mha_core's cache placeholders so external cache storage
    // is interpreted consistently across platforms.
#ifdef ENABLE_FP16
    const auto cache_dtype = ml::train::TensorDim::DataType::FP16;
#else
    const auto cache_dtype = ml::train::TensorDim::DataType::UINT16;
#endif

    const unsigned int max_timestep = static_cast<unsigned int>(MAX_SEQ_LEN);

    kv_cache.allocate(static_cast<unsigned int>(NUM_LAYERS), BATCH_SIZE,
                      max_timestep,
                      static_cast<unsigned int>(NUM_KEY_VALUE_HEADS),
                      static_cast<unsigned int>(HEAD_DIM), cache_dtype);
    kv_cache_bound = false;
  }

  if (kv_cache_bound)
    return;

  // Bind each (layer, K|V) buffer into the corresponding input layer
  // declared by Transformer::createKVCachePlaceholders(). The names here
  // must match what createKVCachePlaceholders() registers with the model.
  // We look up each placeholder by name and point it at our cache slab;
  // this is the same wiring Model::setExternalTensors used to do, just
  // without going through that API.
  for (int i = 0; i < NUM_LAYERS; ++i) {
    auto &kc = kv_cache.getKeyCache(i);
    auto &vc = kv_cache.getValueCache(i);

    auto find_cache_placeholder = [this](const std::string &base_name) {
      for (const auto &suffix : {":0", ":input0", ":out0", ""}) {
        auto *tensor = model->getTensor(base_name + suffix);
        if (tensor != nullptr)
          return tensor;
      }
      return static_cast<nntrainer::Tensor *>(nullptr);
    };

    auto *kp =
      model->getTensor("layer" + std::to_string(i) + "_attention:input3");
    auto *vp =
      model->getTensor("layer" + std::to_string(i) + "_attention:input4");
    if (kp == nullptr)
      kp = find_cache_placeholder("cache_k_l" + std::to_string(i));
    if (vp == nullptr)
      vp = find_cache_placeholder("cache_v_l" + std::to_string(i));
    NNTR_THROW_IF(kp == nullptr || vp == nullptr, std::runtime_error)
      << "allocateAndBindKVCache: cache_k_l" << i << " / cache_v_l" << i
      << " input placeholder not found in compiled graph";
    NNTR_THROW_IF(kp->getDataType() != kc.getDataType() ||
                    vp->getDataType() != vc.getDataType(),
                  std::runtime_error)
      << "allocateAndBindKVCache: cache placeholder dtype mismatch for layer "
      << i;

    kp->setData(kc.getMemoryData(), kc.getOffset(), false);
    vp->setData(vc.getMemoryData(), vc.getOffset(), false);
  }

  kv_cache_bound = true;
}

void CausalLM::setKVCachePosition(unsigned int pos) {
  kv_cache.setPosition(pos);
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [pos](ml::train::Layer &l, nntrainer::RunLayerContext &, void *) {
      if (l.getType() == causallm::MHACoreLayer::type)
        l.setProperty({"cache_index=" + std::to_string(pos)});
    };
  model->forEachLayer(fn, nullptr);
}

void CausalLM::advanceKVCachePosition(unsigned int step_size) {
  // mha_core advances its own cache_index inside forwarding(), so the host
  // only has to keep KVCacheManager's tracked position in sync.
  kv_cache.advance(step_size);
}

std::pair<Tensor, Tensor> CausalLM::constructModel() {

  // base transformer (input, output_norm)
  auto [x, h] = Transformer::constructModel();

  const std::string lmhead_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "lm_head";

  std::vector<std::string> lmhead_prop = {
    withKey("name", "output_of_causallm"),
    withKey("unit", NUM_VOCAB),
    withKey("disable_bias", "true"),
    withKey("weight_dtype", LMHEAD_DTYPE),
  };

  if (TIE_WORD_EMBEDDINGS)
    lmhead_prop.emplace_back(withKey("shared_from", "embedding0"));

  LayerHandle lmhead(createLayer(lmhead_type, lmhead_prop));
  Tensor y = lmhead(h);

  return {x, y};
}

void CausalLM::registerOutputs(
  std::unique_ptr<tokenizers::Tokenizer> &tokenizer,
  std::vector<unsigned int> ids, unsigned int pos,
  const std::vector<bool> &eos_list, bool log_output) {

  static const std::vector<char> puncts{',', '!', ':', ';', '?'};
  for (size_t b = 0; b < ids.size(); ++b) {
    if (!eos_list[b]) {
      pending_ids_.push_back(static_cast<int>(ids[b]));
      ids_history[b * MAX_SEQ_LEN + pos] = ids[b];
      std::string decoded_str = tokenizer->Decode(pending_ids_);

      if (decoded_str.empty()) {
        continue;
      }

      if (std::find(puncts.begin(), puncts.end(), decoded_str.back()) !=
          puncts.end()) {
        // last symbol is a punctuation, hold on
      } else if (decoded_str.size() >= 3 &&
                 decoded_str.compare(decoded_str.size() - 3, 3, "") == 0) {
        // ends with an incomplete token, hold on
      } else {
        if (log_output) {
          std::cout << decoded_str;
          std::cout.flush();
        }
        output_list[b].append(decoded_str);
        pending_ids_.clear();
      }
    }
  }
}

void CausalLM::save_kvcache(std::string path, int to_) {
  if (!kv_cache.isAllocated()) {
    throw std::runtime_error(
      "save_kvcache called before allocateAndBindKVCache()");
  }
  kv_cache.save(path, static_cast<unsigned int>(to_));
}

void CausalLM::load_kvcache(std::string path, int to_) {
  if (!kv_cache.isAllocated()) {
    allocateAndBindKVCache();
  }
  kv_cache.load(path, static_cast<unsigned int>(to_));
  // mha_core layers each track their own cache_index; sync them all to the
  // newly-loaded position so the next forwarding() writes at the right slot.
  setKVCachePosition(static_cast<unsigned int>(to_));
}

std::vector<unsigned int> CausalLM::generate(float *logits, bool do_sample,
                                             float repetition_penalty,
                                             unsigned int *input_ids,
                                             unsigned int NUM_INPUT_IDS) {

  std::vector<unsigned int> outputs;
  for (unsigned int iteration = 0; iteration < BATCH_SIZE; ++iteration) {

    // apply repetition penalty
    if (repetition_penalty != 1 && input_ids != nullptr && NUM_INPUT_IDS != 0) {
      applyRepetitionPenalty(logits, input_ids, NUM_INPUT_IDS,
                             repetition_penalty);
    }

    // apply bad words penalty
    if (BAD_WORD_IDS.size() != 0 && NUM_BADWORDS != 0) {
      applyBadWordsPenalty(logits, BAD_WORD_IDS.data(), NUM_BADWORDS);
    }

    // return argmax if do_sample is false
    if (do_sample == false) {
      unsigned int argmax_idx =
        std::distance(logits, std::max_element(logits, logits + NUM_VOCAB));
      outputs.push_back(argmax_idx);
    } else {
      // apply temperature & top-k & top-p to logits
      float max_logits = applyTKP(logits, NUM_VOCAB, TEMPERATURE, TOP_K, TOP_P);
      // transform logits to softmax
      float sum_exp_logits = 0;
      for (unsigned int i = 0; i < NUM_VOCAB; i++) {
        float exp_x = exp(logits[i] - max_logits);
        sum_exp_logits += exp_x;
        logits[i] = exp_x;
      }

      for (unsigned int i = 0; i < NUM_VOCAB; ++i) {
        logits[i] /= sum_exp_logits;
      }

      // sample from final logits
      std::discrete_distribution<int> dist(logits, logits + NUM_VOCAB);
      unsigned int sampled_idx = dist(rng);

      // add sampled word
      outputs.push_back(sampled_idx);
    }

    // set batch offset
    logits = logits + NUM_VOCAB;
    input_ids = input_ids + MAX_SEQ_LEN;
  }

  return outputs;
};

void CausalLM::registerCustomLayers() {
  Transformer::registerCustomLayers();
  const auto &ct_engine = nntrainer::Engine::Global();
  const auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));
  try {
    app_context->registerFactory(nntrainer::createLayer<causallm::LmHeadLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

void CausalLM::runDDTree(const WSTR &prompt, bool log_output) {
  // DDTree self-draft tree speculative decode:
  // prefill -> [self-draft horizon rollout -> buildTree/compile -> masked
  // verify forward -> argmax posterior -> followVerified ->
  // KVCacheManager::compactTail -> commit accepted path] until num_to_generate
  // tokens or EOS. Greedy (argmax); lossless vs the autoregressive greedy path
  // modulo genuine near-ties.
  const int HORIZON = 15;
  const int BUDGET = 31;
  // Honor the res-dir config's num_to_generate so a longer generation (more
  // DDTree blocks) can be requested without recompiling; fall back to 64.
  const unsigned int MAX_NEW =
    NUM_TO_GENERATE > 0 ? (unsigned int)NUM_TO_GENERATE : 64u;
  allocateAndBindKVCache();

  auto start_total = std::chrono::high_resolution_clock::now();

  // reset decode output state for this run
  output_list.assign(BATCH_SIZE, "");
  pending_ids_.clear();
  std::vector<bool> eos_list(BATCH_SIZE, false);

  // echo the (already templated) input prompt, like the autoregressive run().
  if (log_output)
    std::cout << prompt << std::endl;

  auto _input = tokenizer->Encode(prompt);
  const unsigned int init_len = static_cast<unsigned int>(_input.size());
  float *input_sample =
    (float *)malloc(sizeof(float) * BATCH_SIZE * MAX_SEQ_LEN);

  auto build_inputs = [&]() {
    std::vector<std::pair<std::string, float *>> ci;
    for (int i = 0; i < NUM_LAYERS; ++i) {
      ci.emplace_back(
        "cache_k_l" + std::to_string(i),
        reinterpret_cast<float *>(kv_cache.getKeyCache(i).getData()));
      ci.emplace_back(
        "cache_v_l" + std::to_string(i),
        reinterpret_cast<float *>(kv_cache.getValueCache(i).getData()));
    }
    std::sort(ci.begin(), ci.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    std::vector<float *> in;
    in.push_back(input_sample);
    for (const auto &c : ci)
      in.push_back(c.second);
    return in;
  };
  std::vector<float *> label;
  std::vector<float *> input = build_inputs();

  std::vector<int> tokens(_input.begin(), _input.end());

  // ---- prefill ----
  auto start_prefill = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < init_len; ++i)
    input_sample[i] = static_cast<float>(_input[i]);

  unsigned int pos; // tokens[pos] == current block root
  setKVCachePosition(0);
  if (SKIP_PREFILL && init_len > 1) {
    // Prefill only the first init_len-1 tokens; the last prompt token becomes
    // the root of the first DDTree block (matches the greedy SKIP_PREFILL path
    // required by gemma4's KV-shared layers). Do NOT sample a prefill argmax.
    auto o = model->incremental_inference(BATCH_SIZE, input, label,
                                          init_len - 1, 0, init_len - 1, false);
    for (auto p : o)
      delete[] p;
    pos =
      init_len - 1; // tokens[pos] == last prompt token (already in `tokens`)
  } else {
    auto o = model->incremental_inference(BATCH_SIZE, input, label, init_len, 0,
                                          init_len);
    tokens.push_back(
      (int)std::distance(o[0], std::max_element(o[0], o[0] + NUM_VOCAB)));
    for (auto p : o)
      delete[] p;
    pos = init_len; // tokens[pos] == current block root
  }

  auto finish_prefill = std::chrono::high_resolution_clock::now();
  auto prefill_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    finish_prefill - start_prefill);

  nntrainer::ddtree::DDTreeConfig cfg;
  cfg.budget = BUDGET;
  cfg.depthLimit = HORIZON;
#ifdef ENABLE_FP16
  // fp16 attention: use the fp16 lowest (finite) so the additive verify mask
  // never overflows to -inf/NaN in fp16 score arithmetic. Any sufficiently
  // negative value fully masks (softmax -> 0), so this stays lossless vs
  // greedy.
  cfg.maskFillValue = -65504.0f;
#else
  cfg.maskFillValue = std::numeric_limits<float>::lowest();
#endif

  unsigned int generated = 0;
  unsigned int blocks = 0;
  bool stop = false;
  std::vector<int> accept_lengths;

  auto is_eos = [&](int t) {
    return std::find(EOS_TOKEN_ID.begin(), EOS_TOKEN_ID.end(),
                     (unsigned int)t) != EOS_TOKEN_ID.end();
  };

  auto start_generation = std::chrono::high_resolution_clock::now();

  while (generated < MAX_NEW && !stop &&
         pos + 1 + (unsigned)HORIZON + 64 < MAX_SEQ_LEN) {
    const unsigned int root = (unsigned int)tokens[pos];

    // ---- self-draft: greedy HORIZON rollout from cache [0,pos) ----
    setKVCachePosition(pos);
    std::vector<float> draft((size_t)HORIZON * NUM_VOCAB);
    unsigned int cur = root;
    for (int d = 0; d < HORIZON; ++d) {
      input_sample[0] = static_cast<float>(cur);
      auto o = model->incremental_inference(BATCH_SIZE, input, label, 1,
                                            pos + d, pos + d + 1);
      std::copy(o[0], o[0] + NUM_VOCAB, &draft[(size_t)d * NUM_VOCAB]);
      cur = (unsigned int)std::distance(
        o[0], std::max_element(o[0], o[0] + NUM_VOCAB));
      for (auto p : o)
        delete[] p;
    }

    // ---- build + compile the tree ----
    nntrainer::ddtree::DDTreeStructure tree =
      nntrainer::ddtree::buildTree(draft.data(), HORIZON, (int)NUM_VOCAB, cfg);
    const int cl = tree.currentLength;
    const int past = (int)pos;
    const int stride = past + cl;
    std::vector<int32_t> vii(cl), vpi(cl);
    std::vector<float> cmask((size_t)cl * stride);
    nntrainer::ddtree::compile((int32_t)root, past, past, tree, cfg, vii.data(),
                               vpi.data(), cmask.data(), stride);

    // ---- sliding-window variant for gemma-style models (no-op for Qwen3) ----
    // makeSlidingMasks is the single source of truth: when sliding_window<=0 it
    // returns sliding==full (pointing back into cmask) rather than a distinct
    // buffer, so we honor sliding_masks.hasSliding / .sliding instead of
    // assuming smask always holds the sliding variant.
    std::vector<float> smask;
    const bool has_sliding = ddtreeHasSlidingLayers();
    const int sliding_window = ddtreeSlidingWindow();
    nntrainer::ddtree::SlidingMasks sliding_masks;
    if (has_sliding) {
      smask.resize((size_t)cl * stride);
      sliding_masks = nntrainer::ddtree::makeSlidingMasks(
        cmask.data(), vpi.data(), cl, stride, sliding_window,
        true, // hasSlidingLayers
        cfg, smask.data());
    }

    // ---- masked verify forward -> node logits ----
    for (int i = 0; i < cl; ++i)
      input_sample[i] = static_cast<float>(vii[i]);
    nntrainer::Tensor maskT(1, 1, (unsigned int)cl, (unsigned int)stride);
    std::copy(cmask.begin(), cmask.end(), maskT.getData<float>());
    nntrainer::Tensor posT(1, 1, 1, (unsigned int)cl);
    for (int i = 0; i < cl; ++i)
      posT.getData<float>()[i] = static_cast<float>(vpi[i]);
    std::vector<float> node_logits((size_t)cl * NUM_VOCAB);

    nntrainer::Tensor slidingT;
    if (sliding_masks.hasSliding) {
      // sliding_masks.sliding may alias cmask (window<=0) or point at smask
      // (finite window); copy from whichever the core returned.
      slidingT =
        nntrainer::Tensor(1, 1, (unsigned int)cl, (unsigned int)stride);
      std::copy(sliding_masks.sliding,
                sliding_masks.sliding + (size_t)cl * stride,
                slidingT.getData<float>());
    }
    causallm::MHACoreLayer::setGlobalDDTreeVerify(
      &maskT, sliding_masks.hasSliding ? &slidingT : nullptr, &posT);
    causallm::TieWordEmbedding::setGlobalVerifyDump(node_logits.data());
    setKVCachePosition(pos);
    {
      auto v = model->incremental_inference(
        BATCH_SIZE, input, label, (unsigned)cl, pos, pos + (unsigned)cl);
      for (auto p : v)
        delete[] p;
    }
    causallm::MHACoreLayer::setGlobalDDTreeVerify(nullptr, nullptr, nullptr);
    causallm::TieWordEmbedding::setGlobalVerifyDump(nullptr);

    // ---- posterior (argmax per node) + accepted path ----
    std::vector<int32_t> posterior(cl);
    for (int i = 0; i < cl; ++i) {
      float *r = &node_logits[(size_t)i * NUM_VOCAB];
      posterior[i] =
        (int32_t)std::distance(r, std::max_element(r, r + NUM_VOCAB));
    }

    nntrainer::ddtree::Accepted acc =
      nntrainer::ddtree::followVerified(tree.childMaps, posterior.data());
    const int alen = (int)acc.indices.size();

    // ---- compact KV tail to the accepted path ----
    kv_cache.setPosition(pos + (unsigned)cl);
    kv_cache.compactTail(pos, acc.indices); // sets position = pos + alen

    // ---- commit tokens: accepted[1..] then bonus next token, truncating at
    // the first EOS so the emitted sequence matches autoregressive greedy ----
    int committed = 0;
    for (int k = 1; k < alen; ++k) {
      const int tok = vii[acc.indices[k]];
      tokens.push_back(tok);
      ++committed;
      if (is_eos(tok)) {
        stop = true;
        break;
      }
    }
    if (!stop) {
      tokens.push_back(acc.nextToken);
      ++committed;
      if (is_eos(acc.nextToken))
        stop = true;
    }

    pos += (unsigned int)alen;
    generated += (unsigned int)committed;
    ++blocks;
    accept_lengths.push_back(committed);
  }

  auto finish_generation = std::chrono::high_resolution_clock::now();

  // Emit the generated tokens (tokens[init_len:]) through the normal output
  // path so they accumulate in output_list and stream to stdout, exactly like
  // the autoregressive generation loop.
  for (unsigned int i = init_len; i < tokens.size(); ++i)
    registerOutputs(tokenizer, {static_cast<unsigned int>(tokens[i])}, i,
                    eos_list, log_output);

  free(input_sample);

  auto decode_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    finish_generation - start_generation);
  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    finish_generation - start_total);
  size_t peak_memory = getPeakMemoryKb();
  const unsigned int decode_tokens = (unsigned int)(tokens.size() - init_len);
  // Self-draft cost: HORIZON single-token draft forwards + 1 verify forward
  // (which processes the whole tree at once) per block.
  const unsigned long draft_forwards =
    (unsigned long)blocks * (unsigned long)HORIZON;
  const unsigned long verify_forwards = (unsigned long)blocks;
  const unsigned long total_forwards = draft_forwards + verify_forwards;
  const double mean_block = blocks ? (double)decode_tokens / blocks : 0.0;
  int max_block = 0;
  for (int a : accept_lengths)
    max_block = std::max(max_block, a);
  const double prefill_tps =
    prefill_duration.count()
      ? (double)init_len / prefill_duration.count() * 1000
      : 0.0;
  const double decode_tps =
    decode_duration.count()
      ? (double)decode_tokens / decode_duration.count() * 1000
      : 0.0;
  const double fwd_per_token =
    decode_tokens ? (double)total_forwards / decode_tokens : 0.0;

  performance_metrics.prefill_tokens = init_len;
  performance_metrics.prefill_duration_ms = prefill_duration.count();
  performance_metrics.generation_tokens = decode_tokens;
  performance_metrics.generation_duration_ms = decode_duration.count();
  performance_metrics.total_duration_ms = total_duration.count();
  performance_metrics.peak_memory_kb = peak_memory;

  if (log_output) {
    std::cout << "\n\n";
    std::cout << "===============[ DDTree 16-tok / 32-tree ]================\n";
    std::cout << "prefill: " << init_len << " tokens, "
              << prefill_duration.count() << " ms, " << prefill_tps << " TPS\n";
    std::cout << "decode: " << decode_tokens << " tokens, "
              << decode_duration.count() << " ms, " << decode_tps << " TPS\n";
    std::cout << "blocks: " << blocks << ", mean tokens/block: " << mean_block
              << ", max: " << max_block << "\n";
    std::cout << "forwards: " << draft_forwards << " draft + "
              << verify_forwards << " verify = " << total_forwards << " ("
              << fwd_per_token << " fwd/token)\n";
    std::cout << "total: " << total_duration.count() << " ms\n";
    std::cout << "peak memory: " << peak_memory << " KB\n";
    std::cout << "==========================================================\n";
    // machine-parseable summary for benchmark scripts.
    std::cout << "[DDTREE_METRICS] decode_tokens=" << decode_tokens
              << " decode_ms=" << decode_duration.count()
              << " decode_tps=" << decode_tps << " blocks=" << blocks
              << " mean_tokens_per_block=" << mean_block
              << " fwd_per_token=" << fwd_per_token
              << " prefill_ms=" << prefill_duration.count()
              << " peak_kb=" << peak_memory << "\n";
  }
}

void CausalLM::run(const WSTR prompt, bool do_sample, const WSTR system_prompt,
                   const WSTR tail_prompt, bool log_output) {

  auto start_total = std::chrono::high_resolution_clock::now();
  if (!is_initialized) {
    throw std::runtime_error("CausalLM model is not initialized. Please call "
                             "initialize() before run().");
  }

  // DDTree tree speculative decoding: enabled by the "ddtree" config flag.
  if (DDTREE_ENABLE) {
    runDDTree(prompt, log_output);
    return;
  }

  // Allocate the host-owned KV cache and bind it to mha_core's external cache
  // input slots. Idempotent: only the first call does work; subsequent runs
  // reuse the same buffers and continue from the computed absolute token
  // position below.
  allocateAndBindKVCache();

  has_run_ = false;

  output_list.clear();
  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    output_list.push_back("");
  }

  if (MAX_SEQ_LEN < INIT_SEQ_LEN) {
    throw std::invalid_argument(
      "MAX_SEQ_LEN must be greater than or equal to INIT_SEQ_LEN");
  }

  /**
   * Variables for Log
   */
  unsigned int generation_cnt = 0;
  int64_t total_generation_duration = 0;

  /**
   * INPUT PREPARATION
   */
  std::vector<float *> input;
  std::vector<float *> label;

  /**
   * SAVE_KVCACHE ?
   *  if USE_KVCACHE && system_prompt is given && but the
   * PRE_COMPUTED_CACHE_PATH does not exist
   */
  SAVE_KVCACHE = (USE_KVCACHE && system_prompt != "" &&
                  !std::filesystem::exists(PRE_COMPUTED_CACHE_PATH));

  // print input text
  if (log_output)
    std::cout << system_prompt << prompt << tail_prompt << std::endl;

  // actual prompt to be used in computation
  std::string prompt_;

  if (USE_KVCACHE) {
    prompt_ = SAVE_KVCACHE ? system_prompt : (prompt + tail_prompt);
  } else {
    prompt_ = system_prompt + prompt + tail_prompt;
  }

  if (USE_KVCACHE && !SAVE_KVCACHE && SYS_PROMP_LEN == 0)
    SYS_PROMP_LEN = tokenizer->Encode(system_prompt).size();

  auto _input = tokenizer->Encode(prompt_);
  ///@note insert bos token at the beginning of the input
  // _input.insert(_input.begin(), BOS_TOKEN_ID);

  // | <------------------- MAX_SEQ_LEN -------------------> |
  //                       ||             ||
  // |<-- System prompt -->||<-- input -->||<-- generate -->|

  std::vector<int64_t> init_input;
  unsigned int _len = _input.size();
  unsigned int num_allow_str = MAX_SEQ_LEN - NUM_TO_GENERATE;
  unsigned int text_len = _len;

  if (_len > num_allow_str)
    text_len = num_allow_str;

  // feed only available length
  // if _input is allowed, it feeds all of the _input
  // otherwise, feeds only a part of _input
  for (unsigned int i = 0; i < text_len; ++i)
    init_input.push_back(_input[i]);

  ///@todo currently, the whole sequence may not be fed into the model
  /// This should be handled later.
  _input.clear();

  unsigned int init_len = init_input.size();
  float *input_sample =
    (float *)malloc(sizeof(float) * BATCH_SIZE * MAX_SEQ_LEN);
  std::vector<bool> eos_list(BATCH_SIZE, false);

  unsigned int input_len = init_len;

  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    for (unsigned int i = 0; i < input_len; ++i) {
      input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN + i] =
        static_cast<float>(init_input[i]);
      ids_history[static_cast<size_t>(b) * MAX_SEQ_LEN + i] = init_input[i];
    }
  }

  /**
   * PREFILL
   */
  std::vector<int64_t> token_ids;
  input.push_back(input_sample);
  auto build_inference_inputs = [&]() {
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

    std::vector<float *> inference_inputs;
    inference_inputs.reserve(1 + cache_inputs.size());
    inference_inputs.push_back(input_sample);
    for (const auto &cache_input : cache_inputs)
      inference_inputs.push_back(cache_input.second);
    return inference_inputs;
  };
  input = build_inference_inputs();

  ///@note contains possible bug
  // std::vector<ml::train::TensorDim> input_dims;
  // ml::train::TensorDim input_dim(1, 1, input_len, DIM);
  // input_dims.push_back(input_dim);
  // model->resetInputDimension(input_dims);

  auto start_prefill = std::chrono::high_resolution_clock::now();

  std::vector<float *> output;

  if (SAVE_KVCACHE) {
    //@note This is for the save the kv cache. precomputed kv cache should be
    // always located at the begining of the prompt.
    // Therefore, it start from 0. and system prompt should be saved in the
    // init_input, so that we can compute system prompt size properly
    //
    // The structure of this precomputed K,V Cache is :
    //
    //  //<-- System Prompt -->/<-- Input Tokens -->/<-- Tail prompt --> //
    //  //< Precomputed cache >/<--given as input-->/<--- from json ---->//
    //

    if (log_output)
      std::cout << "\n==============[KV CACHE SAVE MODE]================\n";
    allocateAndBindKVCache();
    setKVCachePosition(0);
    output = model->incremental_inference(BATCH_SIZE, input, label, input_len,
                                          0, input_len, false);

    SYS_PROMP_LEN = input_len;
    save_kvcache(PRE_COMPUTED_CACHE_PATH, SYS_PROMP_LEN);

    if (log_output) {

      std::cout << "kv caches are saved in " << PRE_COMPUTED_CACHE_PATH
                << std::endl
                << "and the size of prompt is " << SYS_PROMP_LEN << ".\n"
                << "You may need this prompt length to set the "
                   "\"sys_prompt_token_size\""
                << "\n==================================================\n"
                << std::endl;
    }
    return;
  }

  if (USE_KVCACHE) {
    load_kvcache(PRE_COMPUTED_CACHE_PATH, SYS_PROMP_LEN);
  } else {
    SYS_PROMP_LEN = 0;
  }
  allocateAndBindKVCache();
  const unsigned int prefill_from = SYS_PROMP_LEN + global_token_len;
  std::vector<unsigned int> id_list;

  if (SKIP_PREFILL && init_len > 1) {
    // Prefill only N-1 tokens; the last input token will be used as the first
    // token in the generation phase (assigned directly, not sampled).
    unsigned int skipped_token =
      static_cast<unsigned int>(init_input[init_len - 1]);

    const unsigned int prefill_to = prefill_from + input_len - 1;
    setKVCachePosition(prefill_from);
    output = model->incremental_inference(
      BATCH_SIZE, input, label, init_len - 1, prefill_from, prefill_to, false);

    for (unsigned int b = 0; b < BATCH_SIZE; ++b)
      id_list.push_back(skipped_token);

    // Adjust lengths so the generation loop processes the skipped token
    // at the correct KV cache position.
    input_len -= 1;
    init_len -= 1;
  } else {
    const unsigned int prefill_to = prefill_from + input_len;
    setKVCachePosition(prefill_from);
    output = model->incremental_inference(BATCH_SIZE, input, label, init_len,
                                          prefill_from, prefill_to, false);

    // post process of model output
    id_list = generate(output[0], do_sample, 1, ids_history, init_len);

    if (init_len < INIT_SEQ_LEN)
      registerOutputs(tokenizer, id_list, init_len, eos_list, log_output);
  }
  // output should be deallocated after use
  for (auto &out : output) {
    delete[] out;
  }

  auto finish_prefill = std::chrono::high_resolution_clock::now();
  auto prefill_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    finish_prefill - start_prefill);

  /**
   * TOKEN GENERATION
   */

  input_len += SYS_PROMP_LEN;

  // Update generated token by prefill as an input
  for (unsigned int b = 0; b < BATCH_SIZE; ++b)
    input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN] =
      static_cast<float>(id_list[b]);

  auto start_generation = std::chrono::high_resolution_clock::now();

  // Optional authoritative greedy token-id capture (env NNTR_GREEDY_IDS):
  // record each step's batch-0 argmax id so a DDTree run can be compared
  // token-for-token (not just via reconstructed stdout text). Dormant unless
  // the env var is set.
  std::vector<unsigned int> greedy_ids_dump;
  const bool dump_greedy_ids = std::getenv("NNTR_GREEDY_IDS") != nullptr;

  for (unsigned int token_generation_idx = input_len + 1;
       token_generation_idx < input_len + 1 + NUM_TO_GENERATE;
       ++token_generation_idx) {

    allocateAndBindKVCache();
    auto output_interval =
      model->incremental_inference(BATCH_SIZE, input, label, input_len,
                                   token_generation_idx - 1 + global_token_len,
                                   token_generation_idx + global_token_len);
    std::vector<unsigned int> ids_list(generate(output_interval[0], do_sample));

    if (dump_greedy_ids)
      greedy_ids_dump.push_back(ids_list[0]);

    // Feed the newly generated token back as the next input token.
    // token_generation_idx always starts at input_len + 1, so we are
    // always in the auto-regressive generation phase here.
    for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
      input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN] =
        static_cast<float>(ids_list[b]);
    }
    registerOutputs(tokenizer, ids_list, token_generation_idx, eos_list,
                    log_output);
    ++generation_cnt;

    // output should be deallocated after use
    for (auto out : output_interval) {
      delete[] out;
    }

    // check FINISH
    for (unsigned int j = 0; j < BATCH_SIZE; ++j) {
      if (!eos_list[j] && (std::find(EOS_TOKEN_ID.begin(), EOS_TOKEN_ID.end(),
                                     ids_list[j]) != EOS_TOKEN_ID.end())) {
        eos_list[j] = true;
      }
    }

    bool is_finish = true;
    for (unsigned int j = 0; j < BATCH_SIZE; ++j) {
      if (!eos_list[j]) {
        is_finish = false;
        break;
      }
    }

    if (is_finish) {
      break;
    }
  }

  // Always release the input buffer after the generation loop, whether
  // the loop exited early (EOS found) or ran to the maximum token limit.
  free(input_sample);

  if (dump_greedy_ids) {
    std::ofstream gf(std::getenv("NNTR_GREEDY_IDS"));
    gf << "{\"gen_ids\":[";
    for (size_t i = 0; i < greedy_ids_dump.size(); ++i)
      gf << (i ? "," : "") << greedy_ids_dump[i];
    gf << "]}\n";
    gf.close();
    std::cout << "[GREEDY] dumped " << greedy_ids_dump.size()
              << " greedy token ids -> " << std::getenv("NNTR_GREEDY_IDS")
              << "\n";
  }

  global_token_len += (generation_cnt + init_len);

  auto finish_generation = std::chrono::high_resolution_clock::now();
  auto generation_duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(finish_generation -
                                                          start_generation);

  auto finish_total = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    finish_total - start_total);
  size_t peak_memory = getPeakMemoryKb();

  if (log_output) {

    std::cout << "\n\n";
    std::cout << "=================[ LLM with NNTrainer ]===================\n";
    std::cout << "prefill: " << init_len << " tokens, "
              << prefill_duration.count() << " ms, "
              << ((double)init_len / prefill_duration.count() * 1000)
              << " TPS\n";
    std::cout << "generation: " << generation_cnt << " tokens, "
              << generation_duration.count() << " ms, "
              << ((double)generation_cnt / generation_duration.count() * 1000)
              << " TPS\n";
    std::cout << "total: " << total_duration.count() << " ms\n";
    std::cout << "peak memory: " << peak_memory << " KB\n";
    std::cout << "==========================================================\n";
  }

  performance_metrics.prefill_tokens = init_len;
  performance_metrics.prefill_duration_ms = prefill_duration.count();
  performance_metrics.generation_tokens = generation_cnt;
  performance_metrics.generation_duration_ms = generation_duration.count();
  performance_metrics.total_duration_ms = total_duration.count();
  performance_metrics.peak_memory_kb = peak_memory;

  has_run_ = true;
}

std::string CausalLM::getOutput(int batch_idx) const {
  if (batch_idx < 0 || batch_idx >= static_cast<int>(output_list.size())) {
    return "";
  }
  return output_list[batch_idx];
}

} // namespace causallm
