// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   pelang_vision_encoder.h
 * @date   18 Aug 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  PE-Lang-L14-448 (timm `vit_pe_lang_large_patch14_448`, EVA family)
 *         vision encoder with enc_to_dec projection.
 *         Produces [1, 1025, 512] projected encoder hidden states.
 *
 * Architecture deltas vs Siglip2VisionEncoder (plan §1.2):
 *   - image / patch : 448 / 14  (vs 224 / 16)
 *   - tokens        : 1025 (1 CLS + 1024 patch)
 *   - depth / dim / heads : 23 / 1024 / 16  (head_dim 64)
 *   - FFN : 4096, **erf GELU**  (NOT tanh_gelu — trap §6.2)
 *   - patch conv bias : **off** (disable_bias=true)
 *   - pre-norm : yes (ln_pre, eps=1e-5 — trap §6.4)
 *   - post-norm : **no**  (Identity — trap §6.3)
 *   - LayerScale : gamma_1 / gamma_2 per-layer [1024]
 *   - position : learned abs + **2D axial RoPE** (interleaved, trap §6.7)
 *     implemented as static table by `pe_rope` custom layer
 *   - Q/K/V : fused `in_proj` [3072,1024] split in converter to 3 separate FCs
 *     (q/k/v order in [0:1024]/[1024:2048]/[2048:3072], trap §6.8)
 *   - LN eps : 1e-5 (vs 1e-6)
 *   - proj output : 1024 -> 512  (vs 768 -> 256)
 *
 * RoPE tables (sin/cos [1024,64]) are static constant `weight` layers created
 * ONCE at graph top (pe_rope_sin / pe_rope_cos) and consumed by every block's
 * two pe_rope invocations (q and k) — trap §6.15.
 *
 * Weight file order MUST match collect_encoder in
 * res/pelang-encoder/weight_converter.py (423 tensors):
 *   pe_patch_conv:filter, pe_pos_embed:pe_pos_embed, pe_cls_row:pe_cls_row,
 *   pe_rope_sin:pe_rope_sin, pe_rope_cos:pe_rope_cos, pe_ln_pre:{gamma,beta},
 *   23 x [enc_layer{i}_ln1{g,b}, _wq{w,b}, _wk{w,b}, _wv{w,b}, _out{w,b},
 *         _ls1:gamma, _ln2{g,b}, _fc1{w,b}, _fc2{w,b}, _ls2:gamma],
 *   enc_to_dec_proj{w,b}
 */

#ifndef __PELANG_VISION_ENCODER_H__
#define __PELANG_VISION_ENCODER_H__

#include <transformer.h>
#include <vector>

namespace causallm {

/**
 * @brief PELangVisionEncoder class — EVA-family ViT-L/14/448 + projection.
 */
class PELangVisionEncoder : virtual public Transformer {

public:
  static constexpr const char *architectures = "PELangVisionEncoder";
  static constexpr unsigned int ENC_TO_DEC_DIM = 512;
  static constexpr unsigned int ENC_NUM_TOKENS = 1025; // 1 CLS + 1024 patches

  /** @brief Construct from JSON configs (model, generation, nntrainer). */
  PELangVisionEncoder(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::MODEL) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~PELangVisionEncoder() = default;

  /**
   * @brief Run encoder on an image file. Returns [1, ENC_NUM_TOKENS,
   *        ENC_TO_DEC_DIM] floats owned by this object (valid until next call).
   */
  std::vector<float> encode(const std::string &image_path);

  /**
   * @brief Run encoder on pre-loaded CHW float pixels (bypass C++ resize).
   *        Used for parity testing against a PyTorch-produced golden pixel
   *        tensor (plan §5 G2/G3).
   */
  std::vector<float> encodePixels(const float *pixel_data, size_t pixel_count);

protected:
  /** @brief Create patch / cls / rope constant layers + ln_pre. */
  Tensor createPatchEmbed(Tensor input);

  /** @brief Create one pre-norm self-attention block with RoPE + LayerScale. */
  Tensor createAttention(const int layer_id, Tensor input, Tensor rope_sin,
                         Tensor rope_cos);

  /** @brief Create one pre-norm erf-GELU FFN block. */
  Tensor createMlp(const int layer_id, Tensor input);

protected:
  std::pair<Tensor, Tensor> constructModel() override;
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;
  Tensor createTransformerDecoderBlock(const int layer_id,
                                       Tensor input) override;
  void registerCustomLayers() override;
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = WSTR(), const WSTR tail_prompt = WSTR(),
           bool log_output = true) override;

private:
  unsigned int IMG_SIZE = 448;
  unsigned int PATCH_SIZE = 14;
  unsigned int NUM_PATCHES = 1024; ///< 32x32 patch grid
  unsigned int IMG_CHANNELS = 3;
  unsigned int NUM_TOKENS = ENC_NUM_TOKENS; ///< NUM_PATCHES + 1 (CLS)
  bool INCLUDE_CLS_TOKEN = true;

  /**
   * @brief Static RoPE constant tensors (outputs of the pe_rope_sin /
   *        pe_rope_cos `weight` layers created once inside createPatchEmbed).
   *        Reused as additional inputs to every block's pe_rope(q)/pe_rope(k)
   *        layer instance (trap §6.15 — shared across all 23 blocks, NOT
   *        duplicated per layer).
   */
  Tensor rope_stash_sin_;
  Tensor rope_stash_cos_;

  /**
   * @brief patch_embed conv weight dtype. **Always FP32** for PE-Lang —
   *        CRS = 3*14*14 = 588 NOT divisible by 32, so Q8_0 packing is
   *        impossible (trap §6.1). Compared to SigLIP2 where the conv *can*
   *        be Q8_0 (CRS=768=32*24), PE-Lang must force FP32 here regardless
   *        of `nntr_cfg["patch_embed_dtype"]`. The quantizer Phase 5 must
   *        ALSO skip writing Q8_0 for this conv.
   */
  std::string PATCH_EMBED_DTYPE = "FP32";

  std::vector<float> enc_output_buf_;
};

} // namespace causallm

#endif /* __PELANG_VISION_ENCODER_H__ */
