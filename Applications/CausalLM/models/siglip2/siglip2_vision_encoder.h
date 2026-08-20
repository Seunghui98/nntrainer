// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   siglip2_vision_encoder.h
 * @date   18 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  SigLIP2 ViT-B/16 vision encoder with enc_to_dec projection.
 *         Produces [1, num_patches, enc_to_dec_dim] projected encoder hidden
 *         states (num_patches = (image_size / patch_size)^2, e.g. 196 for
 *         224px, 576 for 384px; enc_to_dec_dim is config-driven via
 *         nntr_config.json's "decoder_hidden_size", e.g. 256 for the 224px
 *         checkpoint, 512 for the 384px one — it must equal the paired
 *         BertDecoder's hidden size since the projection feeds it directly).
 */

#ifndef __SIGLIP2_VISION_ENCODER_H__
#define __SIGLIP2_VISION_ENCODER_H__

#include <transformer.h>
#include <vector>

namespace causallm {

/**
 * @brief Siglip2VisionEncoder class
 *
 * Implements the SigLIP2 ViT-B/16 vision encoder. Architecture:
 *   - Conv2D patch embed (with bias)
 *   - Learned position embedding
 *   - 12x transformer blocks (pre-LN, separate Q/K/V, tanh_gelu MLP)
 *   - Post layer norm
 *   - Linear projection enc_to_dec_proj: [768 -> enc_to_dec_dim]
 *
 * Output: [1, num_patches, enc_to_dec_dim] projected encoder hidden states.
 */
class Siglip2VisionEncoder : virtual public Transformer {

public:
  static constexpr const char *architectures = "Siglip2VisionEncoder";
  /** Fallback enc_to_dec_proj output dim when nntr_config.json has no
   *  "decoder_hidden_size" (matches the 224px checkpoint this encoder was
   *  first verified against). Actual value is resolved in setupParameters()
   *  and exposed via getEncToDecDim() — use that, not this constant. */
  static constexpr unsigned int ENC_TO_DEC_DIM_DEFAULT = 256;

  /**
   * @brief Construct a Siglip2VisionEncoder object.
   */
  Siglip2VisionEncoder(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::MODEL) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  /**
   * @brief Destroy the Siglip2VisionEncoder object.
   */
  virtual ~Siglip2VisionEncoder() = default;

  /**
   * @brief Run the encoder on an image and return projected hidden states.
   * @param image_path Path to input image file.
   * @return Projected encoder hidden states [1, num_patches, 256] as raw float
   *         buffer.
   *         The buffer is owned by this object and valid until the next call.
   */
  std::vector<float> encode(const std::string &image_path);

  /**
   * @brief Run the encoder on pre-loaded CHW float pixels (bypass C++ resize).
   *
   * Accepts a pre-normalized [1,3,IMG_SIZE,IMG_SIZE] float32 buffer (e.g.
   * from PyTorch/PIL preprocessing) so encoder math can be compared against
   * a golden produced with identical input pixels.
   *
   * @param pixel_data Pointer to 1*3*IMG_SIZE*IMG_SIZE floats in CHW order.
   * @param pixel_count Total number of floats (must equal 3*IMG_SIZE*IMG_SIZE).
   * @return Projected encoder hidden states [1, num_patches, 256].
   *
   * @note Internal/debug-only tooling — not part of the production interface.
   *       Production callers (Task 5+) use only initialize(), load_weight(),
   *       and encode(). This method exists solely for parity verification
   *       (verify_parity.py --dump) and may be removed or restricted in
   *       future releases.
   */
  std::vector<float> encodePixels(const float *pixel_data, size_t pixel_count);

  /**
   * @brief Get the resolved enc_to_dec_proj output dim (== the paired
   *        decoder's hidden size). Valid after construction (resolved in
   *        setupParameters() from nntr_config.json's "decoder_hidden_size").
   */
  unsigned int getEncToDecDim() const { return enc_to_dec_dim_; }

protected:
  /**
   * @brief Create patch embedding layers.
   */
  Tensor createPatchEmbed(Tensor input);

  /**
   * @brief Create a pre-normalized self-attention block.
   */
  Tensor createAttention(const int layer_id, Tensor input);

  /**
   * @brief Create a pre-normalized feed-forward block with tanh_gelu.
   */
  Tensor createMlp(const int layer_id, Tensor input);

protected:
  /**
   * @brief Construct the SigLIP2 encoder graph and return input/output tensors.
   */
  std::pair<Tensor, Tensor> constructModel() override;

  /**
   * @brief Set model parameters from HuggingFace and nntrainer configs.
   */
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  /**
   * @brief Create a transformer encoder block with residual connections.
   */
  Tensor createTransformerDecoderBlock(const int layer_id,
                                       Tensor input) override;

  /**
   * @brief Register custom layers required by the base transformer.
   */
  void registerCustomLayers() override;

  /**
   * @brief Run the encoder on an image (override for encoder-specific
   * behavior).
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = WSTR(), const WSTR tail_prompt = WSTR(),
           bool log_output = true) override;

private:
  unsigned int IMG_SIZE = 224;    /**< Image height/width */
  unsigned int PATCH_SIZE = 16;   /**< Patch height/width */
  unsigned int NUM_PATCHES = 196; /**< Number of patches (resolved in setupParameters) */
  unsigned int IMG_CHANNELS = 3;  /**< Image channels (RGB) */
  /** enc_to_dec_proj output dim; resolved in setupParameters() from
   *  nntr_config.json's "decoder_hidden_size" (defaults to
   *  ENC_TO_DEC_DIM_DEFAULT). Must equal the paired BertDecoder's hidden
   *  size — baked into the graph at build time, so it must be correct
   *  before initialize() is called (setupParameters() runs in the
   *  constructor, before any initialize() call, so no extra setter is
   *  needed here unlike BertDecoder's setEncoderLength()/setDecoderDims()). */
  unsigned int enc_to_dec_dim_ = ENC_TO_DEC_DIM_DEFAULT;
  /** Use PIL BICUBIC (vs. BILINEAR) resize; must match the checkpoint's
   *  preprocessor_config.json "resample" (nntr_config.json "resample"). */
  bool use_bicubic_resample_ = false;

  /** Owned buffer for the last encode() call output. */
  std::vector<float> enc_output_buf_;
};

} // namespace causallm

#endif /* __SIGLIP2_VISION_ENCODER_H__ */
