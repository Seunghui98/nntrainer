// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   custom_gru_layer.h
 * @date   14 January 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Custom GRU Layer compatible with PyTorch weight order (R -> Z -> N)
 */

#ifndef __CUSTOM_GRU_H__
#define __CUSTOM_GRU_H__
#ifdef __cplusplus

#include <acti_func.h>
#include <common_properties.h>
#include <layer_impl.h>

namespace causallm {

/**
 * @class   CustomGRULayer
 * @brief   Custom GRU Layer that follows PyTorch weight order (R, Z, N)
 */
class CustomGRULayer : public nntrainer::LayerImpl {
public:
  CustomGRULayer();
  ~CustomGRULayer() = default;
  CustomGRULayer(CustomGRULayer &&rhs) noexcept = default;
  CustomGRULayer &operator=(CustomGRULayer &&rhs) = default;

  void finalize(nntrainer::InitLayerContext &context) override;
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;
  void calcDerivative(nntrainer::RunLayerContext &context) override;
  void calcGradient(nntrainer::RunLayerContext &context) override;
  void exportTo(nntrainer::Exporter &exporter, const ml::train::ExportMethods &method) const override;
  
  const std::string getType() const override { return CustomGRULayer::type; };
  bool supportBackwarding() const override { return true; }
  void setProperty(const std::vector<std::string> &values) override;
  void setBatch(nntrainer::RunLayerContext &context, unsigned int batch) override;

  static constexpr const char *type = "custom_gru";

private:
  static constexpr unsigned int NUM_GATE = 3;

  std::tuple<nntrainer::props::Unit, nntrainer::props::HiddenStateActivation,
             nntrainer::props::RecurrentActivation, nntrainer::props::ReturnSequences,
             nntrainer::props::DropOutRate, nntrainer::props::IntegrateBias, nntrainer::props::ResetAfter>
    gru_props;
  std::array<unsigned int, 9> wt_idx;

  nntrainer::ActiFunc acti_func;
  nntrainer::ActiFunc recurrent_acti_func;
  float epsilon;
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __CUSTOM_GRU_H__ */
