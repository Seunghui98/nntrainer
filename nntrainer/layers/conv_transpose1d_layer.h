// SPDX-License-Identifier: Apache-2.0
#ifndef __CONV_TRANSPOSE1D_LAYER_H__
#define __CONV_TRANSPOSE1D_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_impl.h>

namespace nntrainer {

class ConvTranspose1DLayer : public LayerImpl {
public:
  ConvTranspose1DLayer();
  ~ConvTranspose1DLayer() = default;

  ConvTranspose1DLayer(ConvTranspose1DLayer &&rhs) noexcept = default;
  ConvTranspose1DLayer &operator=(ConvTranspose1DLayer &&rhs) = default;

  void finalize(InitLayerContext &context) override;
  void forwarding(RunLayerContext &context, bool training) override;
  void calcDerivative(RunLayerContext &context) override;
  void calcGradient(RunLayerContext &context) override;

  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  const std::string getType() const override {
    return ConvTranspose1DLayer::type;
  }

  bool supportBackwarding() const override { return true; }

  using Layer::setProperty;
  void setProperty(const std::vector<std::string> &values) override;

  static constexpr const char *type = "convtranspose1d";

private:
  std::tuple<props::KernelSize, props::Stride> conv_props;
  std::array<unsigned int, 2> wt_idx;
};

} // namespace nntrainer

#endif
#endif