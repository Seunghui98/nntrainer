// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   deberta_attention_layer.h
 * @date   14 January 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Please refer to the following code :
 * https://github.com/huggingface/transformers/blob/5c1c72b/src/transformers/models/deberta/modeling_deberta.py
 */

#ifndef __DEBERTA_ATTENTION_LAYER_H__
#define __DEBERTA_ATTENTION_LAYER_H__

#include <layer_impl.h>
#include <util_func.h>
#include <common_properties.h>

namespace causallm {

namespace props {

/**
 * @brief MaxRelativePositions property
 */
class MaxRelativePositions : public nntrainer::Property<unsigned int> {
public:
  static constexpr const char *key = "max_relative_positions";
  using prop_tag = nntrainer::uint_prop_tag;
  MaxRelativePositions(unsigned int value = 0) { set(value); }
};

/**
 * @brief C2P property
 */
class C2P : public nntrainer::Property<bool> {
public:
  static constexpr const char *key = "c2p";
  using prop_tag = nntrainer::bool_prop_tag;
  C2P(bool value = false) { set(value); }
};

/**
 * @brief P2C property
 */
class P2C : public nntrainer::Property<bool> {
public:
  static constexpr const char *key = "p2c";
  using prop_tag = nntrainer::bool_prop_tag;
  P2C(bool value = false) { set(value); }
};

/**
 * @brief MaxPositionEmbeddings property
 */
class MaxPositionEmbeddings : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "max_position_embeddings";
  using prop_tag = nntrainer::uint_prop_tag;
  MaxPositionEmbeddings(unsigned int value = 512) { set(value); }
};

/**
 * @brief ShareAttKey property
 */
class ShareAttKey : public nntrainer::Property<bool> {
public:
  static constexpr const char *key = "share_att_key";
  using prop_tag = nntrainer::bool_prop_tag;
  ShareAttKey(bool value = false) { set(value); }
};

/**
 * @brief RelativeAttention property
 */
class RelativeAttention : public nntrainer::Property<bool> {
public:
  static constexpr const char *key = "relative_attention";
  using prop_tag = nntrainer::bool_prop_tag;
  RelativeAttention(bool value = true) { set(value); }
};

/**
 * @brief PositionBuckets property
 */
class PositionBuckets : public nntrainer::Property<int> {
public:
  static constexpr const char *key = "position_buckets";
  using prop_tag = nntrainer::int_prop_tag;
  PositionBuckets(int value = -1) { set(value); }
};

/**
 * @brief InputLen property
 */
class InputLen : public nntrainer::Property<unsigned int> {
public:
  static constexpr const char *key = "input_len";
  using prop_tag = nntrainer::uint_prop_tag;
  InputLen(unsigned int value = 0) { set(value); }
};

} // namespace props

/**
 * @class DebertaAttentionLayer
 * @brief Deberta Attention Layer
 */
class DebertaAttentionLayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief Construct a new Deberta Attention Layer object
   */
  DebertaAttentionLayer();

  /**
   * @brief Destroy the Deberta Attention Layer object
   */
  ~DebertaAttentionLayer();

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, const ExportMethods &method)
   */
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return DebertaAttentionLayer::type; };

  static constexpr const char *type = "deberta_attention";
  
  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return false; }

private:
  void softmax_last_dim(nntrainer::Tensor &tensor, unsigned int input_len);

  int make_log_bucket_position(int relative_pos, int bucket_size,
                               int max_position);

  void one_batch_incremental_forwarding(
    const unsigned int batch, const unsigned int from, const unsigned int to,
    nntrainer::Tensor &query_step, nntrainer::Tensor &key_step,
    nntrainer::Tensor &value_step, nntrainer::Tensor &rel_embeddings,
    nntrainer::Tensor &output_step, nntrainer::Tensor &mask_step, nntrainer::RunLayerContext &context);

private:
  std::tuple<nntrainer::props::NumHeads, props::MaxPositionEmbeddings,
             props::MaxRelativePositions, props::C2P, props::P2C,
             props::ShareAttKey, props::RelativeAttention,
             props::PositionBuckets, props::InputLen, nntrainer::props::DisableBias>
    deberta_props;

  std::vector<unsigned int> weight_idx;
};

} // namespace causallm

#endif /* __DEBERTA_ATTENTION_LAYER_H__ */
