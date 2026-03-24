// SPDX-License-Identifier: Apache-2.0
/**
* Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
*
* @file   depthwise_conv1d_layer.h
* @date   22 March 2026
* @see    https://github.com/nnstreamer/nntrainer
* @author Jijoong Moon <jijoong.moon@samsung.com>
* @bug    No known bugs except for NYI items
* @brief  This is Depthwise Convolution 1D Layer Class for Neural Network
*
*/

#ifndef __DEPTHWISE_CONV1D_LAYER_H_
#define __DEPTHWISE_CONV1D_LAYER_H_
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_impl.h>

namespace nntrainer {

/**
* @class   DepthwiseConv1DLayer
* @brief   Depthwise Convolution 1D Layer
*
* Tensor layout convention in this layer:
*
* Input  shape: (B, 1, H, W)
*   - H : sequence length
*   - W : channel
*
* Weight shape: (1, 1, K, W)
*   - K : kernel size along sequence axis(H)
*   - W : per-channel independent filter coefficients
*
* Output shape: (B, 1, out_H, W)
*/
class DepthwiseConv1DLayer : public LayerImpl {
public:
  DepthwiseConv1DLayer();
  ~DepthwiseConv1DLayer() = default;

  DepthwiseConv1DLayer(DepthwiseConv1DLayer &&rhs) noexcept = default;
  DepthwiseConv1DLayer &operator=(DepthwiseConv1DLayer &&rhs) = default;

  void finalize(InitLayerContext &context) override;
  void forwarding(RunLayerContext &context, bool training) override;
  void calcDerivative(RunLayerContext &context) override;
  void calcGradient(RunLayerContext &context) override;

  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  const std::string getType() const override {
    return DepthwiseConv1DLayer::type;
  };

  bool supportBackwarding() const override { return true; }

  using Layer::setProperty;
  void setProperty(const std::vector<std::string> &values) override;

  static constexpr const char *type = "depthwiseconv1d";

private:
  std::array<unsigned int, 2> padding;
  std::tuple<props::KernelSize, props::Stride, props::Padding1D,
             props::Dilation>
    conv_props;

  std::array<unsigned int, 2> wt_idx;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __DEPTHWISE_CONV1D_LAYER_H_ */