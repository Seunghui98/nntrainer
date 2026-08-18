// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   pelang_vision_encoder.cpp
 * @date   18 Aug 2026
 * @brief  PE-Lang-L14-448 (EVA ViT-L/14/448) vision encoder implementation.
 *
 * Forward (verified against timm `vit_pe_lang_large_patch14_448` — plan §1.2):
 *   x = Conv2d(3->1024, k=14, s=14, bias=False)(pixels)        [1,1024,32,32]
 *   x = reshape+permute -> [1,1,1024,1024]
 *   x = x + pe_pos_embed                                      [1,1,1024,1024]
 *   x = concat([pe_cls_row, x], axis=2)                       [1,1,1025,1024]
 *       (CLS fuse: cls_row = class_emb + pos_emb[0] precomputed in converter)
 *   x = LayerNorm(eps=1e-5)(x)                                [1,1,1025,1024]
 *   for i in 0..22:
 *     h  = LN1(x, eps=1e-5)
 *     q  = Wq(h), k = Wk(h), v = Wv(h)              (split from fused in_proj)
 *     q,k = pe_rope(q, sin, cos), pe_rope(k, sin, cos)         (interleaved)
 *     a  = mha_core(q,k,v) [non-causal, no built-in RoPE]
 *     o  = Wo(a)
 *     x  = layer_scale(o, x, gamma=ls1)           (out = x + ls1 * o)
 *     h2 = LN2(x, eps=1e-5)
 *     h2 = fc1(h2,4096) -> erf GELU -> fc2(h2,1024)
 *     x  = layer_scale(h2, x, gamma=ls2)
 *   hidden = Linear(1024->512)(x)                              [1,1025,512]
 *
 * Weight loading order MUST match collect_encoder in
 * res/pelang-encoder/weight_converter.py (423 tensors total).
 *
 * ponytail: NEON GEMM/layer optimization is deferred to Phase 5 (§4) — no
 * profiling exists yet, so we use the standard nntrainer FC/conv/LN layers.
 */

#include "pelang_vision_encoder.h"

#include <app_context.h>
#include <engine.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <factory.h>
#include <iomanip>
#include <llm_util.hpp>
#include <nntrainer-api-common.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "layers/layer_scale.h"
#include "layers/pe_cls_pos.h"
#include "layers/pe_rope.h"

namespace causallm {

/* -------------------------------------------------------------------------
 * Image preprocessing
 *
 * PE-Lang uses Pillow BICUBIC (a=-0.5, support=2.0) resize, NOT the BILINEAR
 * path SigLIP2 reproduces (plan §1.5). The bit-exact BICUBIC implementation
 * (PIL `Resample.c` fixed-point pipeline) is part of G7 (Phase 6+), gated
 * behind the on-device path. Phase 1-4 parity (G2/G3/G4) uses ONLY
 * `encodePixels()` with PyTorch-produced pixel.npy, so C++ resize never runs.
 *
 * `encode(path)` is therefore a deferred stub at this stage — calling it
 * throws a clear "not yet implemented" rather than silently producing wrong
 * numbers via the SigLIP2 BILINEAR arm. (ponytail: don't widen scope before
 * the easier parity win is captured.)
 * --------------------------------------------------------------------- */

/* -------------------------------------------------------------------------
 * Parameter setup
 * --------------------------------------------------------------------- */

void PELangVisionEncoder::setupParameters(json &cfg, json &generation_cfg,
                                          json &nntr_cfg) {
  (void)generation_cfg;

  BATCH_SIZE = nntr_cfg.value("batch_size", 1);
  MODEL_TENSOR_TYPE = nntr_cfg.value("model_tensor_type", "FP32-FP32");
  EMBEDDING_DTYPE = nntr_cfg.value("embedding_dtype", "FP32");
  FC_LAYER_DTYPE = nntr_cfg.value("fc_layer_dtype", "FP32");
  // patch_embed_dtype is IGNORED for PE-Lang (FP32 forced — trap §6.1).
  // Reading it is intentional so the value is visible in diagnostics, but
  // runtime uses PATCH_EMBED_DTYPE = "FP32" below.
  (void)nntr_cfg.value("patch_embed_dtype", "FP32");

  json enc_cfg = cfg.contains("encoder") ? cfg["encoder"] : cfg;

  DIM =
    enc_cfg.value("hidden_size", nntr_cfg.value("encoder_hidden_size", 1024));
  INTERMEDIATE_SIZE = enc_cfg.value(
    "intermediate_size", nntr_cfg.value("encoder_intermediate_size", 4096));
  NUM_LAYERS = enc_cfg.value("num_hidden_layers",
                             nntr_cfg.value("encoder_num_layers", 23));
  NUM_HEADS = enc_cfg.value("num_attention_heads",
                            nntr_cfg.value("encoder_num_heads", 16));
  HEAD_DIM = DIM / NUM_HEADS; // 64
  NUM_KEY_VALUE_HEADS = NUM_HEADS;
  NORM_EPS = enc_cfg.value("layer_norm_eps",
                           nntr_cfg.value("encoder_layer_norm_eps", 1e-5f));
  GQA_SIZE = 1;

  IS_CAUSAL = false;
  ROPE_THETA = 0;
  TIE_WORD_EMBEDDINGS = false;
  SLIDING_WINDOW = UINT_MAX;

  IMG_SIZE = enc_cfg.value("image_size", nntr_cfg.value("img_size", 448));
  PATCH_SIZE = enc_cfg.value("patch_size", nntr_cfg.value("patch_size", 14));
  NUM_PATCHES = nntr_cfg.value("num_patches", 1024u);
  IMG_CHANNELS = 3;
  INCLUDE_CLS_TOKEN = nntr_cfg.value("include_cls_token", true);
  NUM_TOKENS = INCLUDE_CLS_TOKEN ? NUM_PATCHES + 1 : NUM_PATCHES;

  INIT_SEQ_LEN = nntr_cfg.value("init_seq_len", NUM_TOKENS);
  MAX_SEQ_LEN = nntr_cfg.value("max_seq_len", NUM_TOKENS);
  NUM_TO_GENERATE = nntr_cfg.value("num_to_generate", 0);
  MEMORY_SWAP = nntr_cfg.contains("fsu") ? nntr_cfg["fsu"].get<bool>() : false;
  FSU_LOOKAHEAD = nntr_cfg.contains("fsu_lookahead")
                    ? nntr_cfg["fsu_lookahead"].get<unsigned int>()
                    : 1;
  MAX_POSITION_EMBEDDINGS = NUM_TOKENS;
}

/* -------------------------------------------------------------------------
 * Graph construction helpers
 *
 * createLayer order MUST match collect_encoder tensor order in the converter
 * (423 tensors): pe_patch_conv -> pe_pos_embed -> pe_cls_row -> pe_rope_sin ->
 * pe_rope_cos -> pe_ln_pre -> 23x[block] -> enc_to_dec_proj.
 * --------------------------------------------------------------------- */

/**
 * @brief Build patch embed, CLS/pos/rope constants, ln_pre.
 *
 * Weight creation order (matches converter 1..7):
 *   1. pe_patch_conv:filter
 *   2. pe_pos_embed:pe_pos_embed
 *   3. pe_cls_row:pe_cls_row
 *   4. pe_rope_sin:pe_rope_sin
 *   5. pe_rope_cos:pe_rope_cos
 *   6/7. pe_ln_pre:gamma, beta
 */
Tensor PELangVisionEncoder::createPatchEmbed(Tensor input) {
  const int embed_dim = DIM;

  LayerHandle conv(createLayer(
    "conv2d", {withKey("name", "pe_patch_conv"),
               withKey("kernel_size", {std::to_string(PATCH_SIZE),
                                       std::to_string(PATCH_SIZE)}),
               withKey("filters", std::to_string(embed_dim)),
               withKey("stride", {std::to_string(PATCH_SIZE),
                                  std::to_string(PATCH_SIZE)}),
               withKey("padding", "valid"), withKey("disable_bias", "true"),
               withKey("weight_dtype", PATCH_EMBED_DTYPE)}));
  Tensor h = conv(input);

  LayerHandle flatten(createLayer(
    "reshape", {withKey("name", "pe_patch_flatten"),
                withKey("target_shape", "1:" + std::to_string(embed_dim) + ":" +
                                          std::to_string(NUM_PATCHES))}));
  h = flatten(h);
  LayerHandle transpose(
    createLayer("permute", {withKey("name", "pe_patch_transpose"),
                            withKey("direction", {1, 3, 2})}));
  h = transpose(h);

  LayerHandle pos_layer(
    createLayer("weight", {withKey("name", "pe_pos_embed"),
                           withKey("dim", "1:1:" + std::to_string(NUM_PATCHES) +
                                            ":" + std::to_string(embed_dim)),
                           withKey("tensor_dtype", "FP32"),
                           withKey("weight_name", "pe_pos_embed")}));
  Tensor pos = pos_layer(input);

  LayerHandle cls_layer(
    createLayer("weight", {withKey("name", "pe_cls_row"),
                           withKey("dim", "1:1:1:" + std::to_string(embed_dim)),
                           withKey("tensor_dtype", "FP32"),
                           withKey("weight_name", "pe_cls_row")}));
  Tensor cls = cls_layer(input);
  LayerHandle cls_pos(
    createLayer("pe_cls_pos", {withKey("name", "pe_cls_pos")}));
  h = cls_pos({h, cls, pos});

  LayerHandle rope_sin_layer(
    createLayer("weight", {withKey("name", "pe_rope_sin"),
                           withKey("dim", "1:1:" + std::to_string(NUM_PATCHES) +
                                            ":" + std::to_string(HEAD_DIM)),
                           withKey("tensor_dtype", "FP32"),
                           withKey("weight_name", "pe_rope_sin")}));
  Tensor rope_sin_const = rope_sin_layer(input);
  LayerHandle rope_cos_layer(
    createLayer("weight", {withKey("name", "pe_rope_cos"),
                           withKey("dim", "1:1:" + std::to_string(NUM_PATCHES) +
                                            ":" + std::to_string(HEAD_DIM)),
                           withKey("tensor_dtype", "FP32"),
                           withKey("weight_name", "pe_rope_cos")}));
  Tensor rope_cos_const = rope_cos_layer(input);
  LayerHandle ln_pre(createLayer(
    "layer_normalization", {withKey("name", "pe_ln_pre"), withKey("axis", "3"),
                            withKey("epsilon", std::to_string(NORM_EPS)),
                            withKey("packed", "false")}));
  Tensor h_normed = ln_pre(h);
  rope_stash_sin_ = rope_sin_const;
  rope_stash_cos_ = rope_cos_const;

  return h_normed;
}

/**
 * @brief Create pre-norm attention sub-block with RoPE + LayerScale.
 *
 * Weight order per layer (matches collect_encoder 8..17):
 *   enc_layer{i}_ln1:gamma, beta
 *   enc_layer{i}_wq:weight, bias
 *   enc_layer{i}_wk:weight, bias
 *   enc_layer{i}_wv:weight, bias
 *   enc_layer{i}_out:weight, bias
 *   enc_layer{i}_ls1:gamma         (after pe_rope+attn, before-ln2)
 *
 * NOTE: the pe_rope layer instances (named enc_layer{i}_pe_rope_q/k) do NOT
 *       request weights — they take sin/cos as INPUTS from the shared
 *       pe_rope_sin/pe_rope_cos constant layers. mha_core has no persistent
 *       weights either. So the per-layer weight count stays 18 (matches the
 *       assert in the converter).
 */
Tensor PELangVisionEncoder::createAttention(const int layer_id, Tensor input,
                                            Tensor rope_sin, Tensor rope_cos) {
  const std::string pfx = "enc_layer" + std::to_string(layer_id);

  // Pre-LN1 (eps=1e-5, axis=3, packed=false)
  LayerHandle norm(createLayer(
    "layer_normalization", {withKey("name", pfx + "_ln1"), withKey("axis", "3"),
                            withKey("epsilon", std::to_string(NORM_EPS)),
                            withKey("packed", "false")}));
  Tensor normed = norm(input);

  // Separate q/k/v FCs (fused in_proj_weight split at conversion time, trap
  // §6.8)
  LayerHandle q_proj(
    createLayer("fully_connected", {withKey("name", pfx + "_wq"),
                                    withKey("unit", std::to_string(DIM)),
                                    withKey("disable_bias", "false")}));
  LayerHandle k_proj(
    createLayer("fully_connected", {withKey("name", pfx + "_wk"),
                                    withKey("unit", std::to_string(DIM)),
                                    withKey("disable_bias", "false")}));
  LayerHandle v_proj(
    createLayer("fully_connected", {withKey("name", pfx + "_wv"),
                                    withKey("unit", std::to_string(DIM)),
                                    withKey("disable_bias", "false")}));
  Tensor q = q_proj(normed);
  Tensor k = k_proj(normed);
  Tensor v = v_proj(normed);

  // RoPE: interleaved, CLS pass-through. (Trap §6.6/6.7: NOT mha_core's
  // built-in use_rope path; use a dedicated pe_rope layer per q/k.)
  LayerHandle rope_q(
    createLayer("pe_rope", {withKey("name", pfx + "_pe_rope_q"),
                            withKey("num_prefix_tokens", "1"),
                            withKey("num_heads", std::to_string(NUM_HEADS)),
                            withKey("head_dim", std::to_string(HEAD_DIM))}));
  Tensor q_rot = rope_q({q, rope_sin, rope_cos});

  LayerHandle rope_k(
    createLayer("pe_rope", {withKey("name", pfx + "_pe_rope_k"),
                            withKey("num_prefix_tokens", "1"),
                            withKey("num_heads", std::to_string(NUM_HEADS)),
                            withKey("head_dim", std::to_string(HEAD_DIM))}));
  Tensor k_rot = rope_k({k, rope_sin, rope_cos});

  // MHA core: non-causal, NO built-in RoPE (we already rotated above).
  LayerHandle attention(createLayer(
    "mha_core", {withKey("name", pfx + "_attn"),
                 withKey("num_heads", std::to_string(NUM_HEADS)),
                 withKey("num_heads_kv", std::to_string(NUM_HEADS)),
                 withKey("max_timestep", std::to_string(NUM_TOKENS + 1)),
                 withKey("is_causal", "false"), withKey("use_rope", "false"),
                 withKey("rope_theta", std::to_string(ROPE_THETA))}));
  Tensor context = attention({q_rot, k_rot, v});

  // Output projection
  LayerHandle out_proj(
    createLayer("fully_connected", {withKey("name", pfx + "_out"),
                                    withKey("unit", std::to_string(DIM)),
                                    withKey("disable_bias", "false")}));
  Tensor attn_out = out_proj(context);

  // LayerScale(ls1) + residual:  out = input + ls1 * attn_out
  LayerHandle ls1(createLayer("layer_scale", {withKey("name", pfx + "_ls1")}));
  return ls1({attn_out, input});
}

/**
 * @brief Create pre-norm MLP sub-block with erf GELU + LayerScale.
 *
 * Weight order per layer (matches collect_encoder 18..23):
 *   enc_layer{i}_ln2:gamma, beta
 *   enc_layer{i}_fc1:weight, bias
 *   enc_layer{i}_fc2:weight, bias
 *   enc_layer{i}_ls2:gamma
 */
Tensor PELangVisionEncoder::createMlp(const int layer_id, Tensor input) {
  const std::string pfx = "enc_layer" + std::to_string(layer_id);

  // Pre-LN2 (eps=1e-5)
  LayerHandle norm(createLayer(
    "layer_normalization", {withKey("name", pfx + "_ln2"), withKey("axis", "3"),
                            withKey("epsilon", std::to_string(NORM_EPS)),
                            withKey("packed", "false")}));
  Tensor h = norm(input);

  // FC1 -> erf GELU (trap §6.2: NOT tanh_gelu) -> FC2
  LayerHandle fc1(createLayer(
    "fully_connected", {withKey("name", pfx + "_fc1"),
                        withKey("unit", std::to_string(INTERMEDIATE_SIZE)),
                        withKey("disable_bias", "false")}));
  h = fc1(h);

  LayerHandle gelu(createLayer("activation", {withKey("name", pfx + "_gelu"),
                                              withKey("activation", "gelu")}));
  h = gelu(h);

  LayerHandle fc2(
    createLayer("fully_connected", {withKey("name", pfx + "_fc2"),
                                    withKey("unit", std::to_string(DIM)),
                                    withKey("disable_bias", "false")}));
  Tensor mlp_out = fc2(h);

  // LayerScale(ls2) + residual:  out = input + ls2 * mlp_out
  LayerHandle ls2(createLayer("layer_scale", {withKey("name", pfx + "_ls2")}));
  return ls2({mlp_out, input});
}

/**
 * @brief One transformer block: attention -> mlp.
 */
Tensor PELangVisionEncoder::createTransformerDecoderBlock(const int layer_id,
                                                          Tensor input) {
  Tensor h = createAttention(layer_id, input, rope_stash_sin_, rope_stash_cos_);
  return createMlp(layer_id, h);
}

/**
 * @brief Construct the full PE-Lang encoder inference graph.
 *
 * Full weight order matches collect_encoder in weight_converter.py:
 *   pe_patch_conv:filter -> pe_pos_embed -> pe_cls_row -> pe_rope_sin ->
 *   pe_rope_cos -> pe_ln_pre -> 23x[ln1,q/k/v,out,ls1,ln2,fc1,fc2,ls2] ->
 *   enc_to_dec_proj
 *
 * NO post_ln — norm is Identity in timm (trap §6.3). Output == encoder
 * features [1, 1025, 1024] -> projection -> [1, 1025, 512].
 */
std::pair<Tensor, Tensor> PELangVisionEncoder::constructModel() {
  Tensor input({BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE}, "input0");
  Tensor h = createPatchEmbed(input);

  for (int i = 0; i < NUM_LAYERS; i++) {
    h = createTransformerDecoderBlock(i, h);
  }

  // Identity post-LN (no layer_normalization here).
  // Encoder-to-decoder projection: [1, 1025, 1024] -> [1, 1025, 512]
  LayerHandle enc_proj(createLayer(
    "fully_connected", {withKey("name", "enc_to_dec_proj"),
                        withKey("unit", std::to_string(ENC_TO_DEC_DIM)),
                        withKey("disable_bias", "false")}));
  h = enc_proj(h);

  return {input, h};
}

/* -------------------------------------------------------------------------
 * Custom layer registration
 * --------------------------------------------------------------------- */

void PELangVisionEncoder::registerCustomLayers() {
  Transformer::registerCustomLayers();

  static std::once_flag pe_layers_registered;
  std::call_once(pe_layers_registered, []() {
    const auto &ct_engine = nntrainer::Engine::Global();
    const auto app_context = static_cast<nntrainer::AppContext *>(
      ct_engine.getRegisteredContext("cpu"));
    app_context->registerFactory(nntrainer::createLayer<causallm::PeRopeLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::PeClsPosLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::LayerScaleLayer>);
  });
}

/* -------------------------------------------------------------------------
 * Inference
 * --------------------------------------------------------------------- */

std::vector<float> PELangVisionEncoder::encode(const std::string &image_path) {
  if (!is_initialized)
    throw std::runtime_error(
      "PELangVisionEncoder is not initialized. Call initialize() first.");
  (void)image_path;
  // ponytail: BICUBIC arm is §7/G7 work — deferred. Phase-1..4 parity
  // verification uses encodePixels() exclusively, so encode(path) is a stub.
  // Phase 6 (Quick.AI integration) replaces this with the real resizer.
  throw std::runtime_error(
    "PELangVisionEncoder::encode(path) not implemented yet — BICUBIC "
    "preprocessing is G7 work. Use encodePixels() with a pre-prepared "
    "pixel tensor for parity testing.");
}

std::vector<float> PELangVisionEncoder::encodePixels(const float *pixel_data,
                                                     size_t pixel_count) {
  if (!is_initialized)
    throw std::runtime_error(
      "PELangVisionEncoder is not initialized. Call initialize() first.");
  const size_t expected =
    static_cast<size_t>(IMG_CHANNELS) * IMG_SIZE * IMG_SIZE;
  if (pixel_count != expected)
    throw std::runtime_error("encodePixels: expected " +
                             std::to_string(expected) + " floats, got " +
                             std::to_string(pixel_count));

  std::vector<float> pixel_buf(pixel_data, pixel_data + pixel_count);
  std::vector<float *> inputs{pixel_buf.data()};
  std::vector<float *> labels;
  std::vector<float *> output = model->incremental_inference(
    BATCH_SIZE, inputs, labels, NUM_TOKENS, 0, NUM_TOKENS, false);

  const int out_size =
    static_cast<int>(NUM_TOKENS) * static_cast<int>(ENC_TO_DEC_DIM);
  enc_output_buf_.assign(output[0], output[0] + out_size);
  for (auto *p : output)
    delete[] p;
  return enc_output_buf_;
}

void PELangVisionEncoder::run(const WSTR prompt, bool do_sample,
                              const WSTR system_prompt, const WSTR tail_prompt,
                              bool log_output) {
  (void)do_sample;
  (void)system_prompt;
  (void)tail_prompt;
  (void)log_output;
  if (!is_initialized)
    throw std::runtime_error(
      "PELangVisionEncoder is not initialized. Call initialize() first.");

  std::string image_path_str(prompt);
  std::vector<float> result = encode(image_path_str);

  std::cout << std::setprecision(9)
            << "Encoder output [1,1025,512], first 10 values:";
  const int print_count = 10;
  for (int i = 0; i < print_count; ++i)
    std::cout << " [" << i << "]=" << result[i];
  std::cout << std::endl;
}

} // namespace causallm
