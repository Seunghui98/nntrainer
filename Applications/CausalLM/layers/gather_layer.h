// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   gather_layer.h
 * @date   02 February 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Custom Gather Layer for CausalLM applications
 */

#ifndef __CAUSALLM_GATHER_LAYER_H__
#define __CAUSALLM_GATHER_LAYER_H__

#include <layer_impl.h>
#include <util_func.h>
#include <common_properties.h>

namespace causallm {

/**
 * @class GatherLayer
 * @brief Custom Gather Layer that gathers elements along specified axis
 */
class GatherLayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief Construct a new Gather Layer object
   */
  GatherLayer();

  /**
   * @brief Destroy the Gather Layer object
   */
  ~GatherLayer();

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
  const std::string getType() const override { return GatherLayer::type; };

  static constexpr const char *type = "causallm_gather";

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return false; }

private:
  std::tuple<nntrainer::props::Axis> gather_props;
  unsigned int axis;
};

} // namespace causallm

#endif /* __CAUSALLM_GATHER_LAYER_H__ */
