// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   slice_layer.h
 * @date   07 April 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is slice layer class (operation layer)
 */

#ifndef __SLICE_LAYER_H__
#define __SLICE_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>
#include <operation_layer.h>

namespace nntrainer {

/**
 * @class Slice Layer
 * @brief Slice Layer
 */
class SliceLayer : public UnaryOperationLayer {
public:
  /**
   * @brief Constructor of Slice Layer
   */
  SliceLayer() :
    UnaryOperationLayer(),
    slice_props(props::Print(), props::StartIndex(), props::EndIndex(),
                props::Axis()),
    support_backwarding(true) {}

  /**
   * @brief Destructor of Slice Layer
   */
  ~SliceLayer(){};

  /**
   *  @brief  Move constructor of Slice Layer.
   *  @param[in] SliceLayer &&
   */
  SliceLayer(SliceLayer &&rhs) noexcept = default;

  /**
   * @brief Move assignment operator.
   * @parma[in] rhs SliceLayer to be moved.
   */
  SliceLayer &operator=(SliceLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) final;

  /**
   * @brief forwarding operation for slice
   *
   * @param input tensor to be sliced from
   * @param hidden tensor to store the result value
   */
  void forwarding_operation(const Tensor &input, Tensor &hidden) final;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) final;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const final { return support_backwarding; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const final {}

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) final;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const final { return SliceLayer::type; };

  /**
   * @copydoc Layer::supportInt8ActInput()
   * @note W4A8: forwarding_operation() has a dtype-correct Q8_0_TW dispatch
   * (sliceForwardT<int8_t>), so an int8 input is copied byte-identically.
   */
  bool supportInt8ActInput() const final { return true; }

  /**
   * @copydoc Layer::supportInt8ActOutput()
   */
  bool supportInt8ActOutput() const final { return true; }

  /**
   * @copydoc Layer::isInt8PassThrough()
   * @note Slice only extracts a contiguous sub-range -- it does not
   * recompute values, so whatever dtype arrives is exactly what its own
   * consumers receive. NetworkGraph::propagateActivationDataTypes() must
   * recurse through it to check the REAL downstream consumers.
   */
  bool isInt8PassThrough() const final { return true; }

  std::tuple<props::Print, props::StartIndex, props::EndIndex, props::Axis>
    slice_props;

  inline static const std::string type = "slice";
  bool support_backwarding;
  unsigned int axis;
  unsigned int start;
  // TensorDim starts;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __SLICE_LAYER_H__ */
