// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moone <jijoong.moon@samsung.com>
 *
 * @file   qkv_layer.h
 * @date   14 May 2020
 * @brief  This is Fully Connected Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __QKV_LAYER_H__
#define __QKV_LAYER_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <causallm_common_properties.h>
#include <common_properties.h>
#include <layer_impl.h>

namespace causallm {

namespace props {

class QUnit : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "q_unit";
  using prop_tag = nntrainer::uint_prop_tag;
};

class KUnit : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "k_unit";
  using prop_tag = nntrainer::uint_prop_tag;
};

class VUnit : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "v_unit";
  using prop_tag = nntrainer::uint_prop_tag;
};

} // namespace props

/**
 * @class   QKVLayer
 * @brief   The q, k and v projections of one attention block as ONE layer,
 *          with the per-head RMS norm of q and k folded in when
 *          feature_size is set.
 *
 * One layer so the three matmuls share one activation: an accelerator
 * takes them as one call (Tensor::dot's vector overload, doc 51 section
 * 2.21 -- three calls each quantized and shipped the same 3.6 MB
 * activation), and on the CPU it is the same three GEMMs. With
 * feature_size (the head dim) the layer holds, in the model file's
 * order, q, q_norm's gamma, k, k_norm's gamma, v, and applies the same
 * rms_norm_wrt_width + gamma multiply ReshapedRMSNormLayer would; the
 * outputs are q_normed, k_normed, v. Without it: q, k, v as they are.
 */
WIN_EXPORT class QKVLayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief     Constructor of Fully Connected Layer
   */
  WIN_EXPORT QKVLayer();

  /**
   * @brief     Destructor of Fully Connected Layer
   */
  WIN_EXPORT ~QKVLayer() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] FullyConnected &&
   */
  WIN_EXPORT QKVLayer(QKVLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs QKVLayer to be moved.
   */
  WIN_EXPORT QKVLayer &operator=(QKVLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  /**
￼   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
￼   * int from, unsigned int to, bool training)
￼   */
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   * @note
   * [note for LoRA] implicit calcDerivative is implicitly applied.
   * The weight is already updated with the LoRA's (W = W + W_lora)
   */
  WIN_EXPORT void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  WIN_EXPORT const std::string getType() const override {
    return QKVLayer::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  WIN_EXPORT bool supportBackwarding() const override { return true; }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "qkv_layer";

private:
  /** feature_size: the head dim q and k are normed over; unset = no norm */
  std::tuple<props::QUnit, props::KUnit, props::VUnit, props::FeatureSize,
             nntrainer::props::Epsilon>
    qkv_props;
  std::array<unsigned int, 5> weight_idx; /**< q, [q_gamma,] k, [k_gamma,] v */
  std::array<unsigned int, 2> tensor_idx; /**< q and k before the norm */
  unsigned int feature_size = 0;
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __QKV_LAYER_H__ */
