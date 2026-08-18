// SPDX-License-Identifier: Apache-2.0
#ifndef __PE_CLS_POS_LAYER_H__
#define __PE_CLS_POS_LAYER_H__

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

namespace causallm {
class PeClsPosLayer final : public nntrainer::Layer {
public:
  void finalize(nntrainer::InitLayerContext &context) override;
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;
  void calcDerivative(nntrainer::RunLayerContext &context) override;
  bool supportBackwarding() const override { return false; }
  void exportTo(nntrainer::Exporter &,
                const ml::train::ExportMethods &) const override {}
  const std::string getType() const override { return type; }
  void setProperty(const std::vector<std::string> &values) override;
  void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;
  inline static const std::string type = "pe_cls_pos";
};
} // namespace causallm
#endif
