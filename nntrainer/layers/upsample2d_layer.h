// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 heka1024 <heka1024@gmail.com>
 *
 * @file   upsample2d_layer.h
 * @date   15 June 2024
 * @brief  This is Upsample2d Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author heka1024 <heka1024@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __UPSAMPLE2D_LAYER_H__
#define __UPSAMPLE2D_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_impl.h>

#include <node_exporter.h>

namespace nntrainer {

constexpr const unsigned int UPSAMPLE2D_DIM = 2;

/**
 * @class   Upsample2dLayer
 * @brief   Upsamle 2d layer
 */
class Upsample2dLayer : public Layer {
public:
  /**
   * @brief Construct a new Upsample layer object
   *
   */
  Upsample2dLayer();

  /**
   * @brief Destroy the Upsample layer object
   *
   */
  ~Upsample2dLayer() {}

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override{};

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return Upsample2dLayer::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::supportInt8ActInput()
   * @note W4A8: only nearest-neighbor upsample is int8-safe without dequant
   * -- it copies an existing raw value verbatim, no interpolation math (see
   * the Q8_0_TW dispatch in forwarding()). Bilinear computes a real weighted
   * average of multiple values and would need dequant/requant -- not
   * implemented, so it stays FP16/FP32-only (false here).
   */
  bool supportInt8ActInput() const override {
    const auto &mode = std::get<props::UpsampleMode>(upsample2d_props);
    return !mode.empty() &&
           mode.get() == props::UpsampleModeInfo::Interpolation::nearest;
  }

  /**
   * @copydoc Layer::supportInt8ActOutput()
   */
  bool supportInt8ActOutput() const override { return supportInt8ActInput(); }

  /**
   * @copydoc Layer::isInt8PassThrough()
   * @note nearest upsample only ever copies through an existing input value
   * verbatim -- it never combines values under a different effective scale.
   */
  bool isInt8PassThrough() const override { return supportInt8ActInput(); }

  static constexpr const char *type = "upsample2d";

private:
  std::tuple<props::UpsampleMode, std::array<props::KernelSize, UPSAMPLE2D_DIM>>
    upsample2d_props; /* mode, size of kernel */
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __UPSAMPLE2D_LAYER_H__ */
