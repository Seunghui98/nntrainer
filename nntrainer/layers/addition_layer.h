// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   addition_layer.h
 * @date   30 July 2020
 * @see    https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Addition Layer Class for Neural Network
 *
 */

#ifndef __ADDITION_LAYER_H__
#define __ADDITION_LAYER_H__
#ifdef __cplusplus

#include <array>

#include <common_properties.h>
#include <layer_devel.h>

namespace nntrainer {

/**
 * @class   Addition Layer
 * @brief   Addition Layer
 */
class AdditionLayer : public Layer {
public:
  /**
   * @brief     Constructor of Addition Layer
   */
  AdditionLayer() :
    Layer(),
    add_props(props::Print(), props::SkipPrefill(), props::ActivationScale(),
              std::array<props::InputActivationScale, 2>()) {}

  /**
   * @brief     Destructor of Addition Layer
   */
  ~AdditionLayer(){};

  /**
   *  @brief  Move constructor of AdditionLayer.
   *  @param[in] AdditionLayer &&
   */
  AdditionLayer(AdditionLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs AdditionLayer to be moved.
   */
  AdditionLayer &operator=(AdditionLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  void incremental_forwarding(RunLayerContext &context, unsigned int from,
                              unsigned int to, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override {}

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return AdditionLayer::type; };

  void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  /**
   * @copydoc Layer::supportInt8ActInput()
   * @note W4A8: Addition always has exactly 2 inputs in this model
   * (residual connections). Each input is dequantized using ITS OWN
   * registered scale (they need not match -- e.g. m10/add_pe combines
   * m10/attn and m10/pe/dw, which happen to share a scale via the
   * calibration tool's union-find grouping, but this code does not assume
   * that), summed in float, then requantized with this layer's own output
   * scale. Requires BOTH per-input scales to be registered.
   */
  bool supportInt8ActInput() const override {
    const auto &in_scales =
      std::get<std::array<props::InputActivationScale, 2>>(add_props);
    return !in_scales[0].empty() && in_scales[0].get() > 0.0f &&
           !in_scales[1].empty() && in_scales[1].get() > 0.0f;
  }

  /**
   * @copydoc Layer::supportInt8ActOutput()
   */
  bool supportInt8ActOutput() const override {
    const auto &aos = std::get<props::ActivationScale>(add_props);
    return !aos.empty() && aos.get() > 0.0f;
  }

  /**
   * @copydoc Layer::hasActivationScale()
   */
  bool hasActivationScale() const override { return supportInt8ActOutput(); }

  std::tuple<props::Print, props::SkipPrefill, props::ActivationScale,
             std::array<props::InputActivationScale, 2>>
    add_props; /**< fc layer properties : unit - number of output neurons */
  bool skip_prefill = false;

  static constexpr const char *type = "addition";
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __ADDITION_LAYER_H__ */
