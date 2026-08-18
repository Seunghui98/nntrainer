// SPDX-License-Identifier: Apache-2.0
#include "pe_cls_pos.h"
#include <algorithm>
#include <stdexcept>

namespace causallm {
void PeClsPosLayer::finalize(nntrainer::InitLayerContext &context) {
  const auto &d = context.getInputDimensions();
  if (d.size() != 3 || d[0].batch() != d[1].batch() ||
      d[0].batch() != d[2].batch() || d[0].width() != d[1].width() ||
      d[0].width() != d[2].width() || d[0].height() != d[2].height())
    throw std::invalid_argument("[pe_cls_pos] expected patches, cls, and pos");
  auto out = d[0];
  out.height(d[0].height() + d[1].height());
  context.setOutputDimensions({out});
}
void PeClsPosLayer::forwarding(nntrainer::RunLayerContext &context, bool) {
  const auto &patches = context.getInput(0);
  const auto &cls = context.getInput(1);
  const auto &pos = context.getInput(2);
  auto &out = context.getOutput(0);
  if (patches.getDataType() != nntrainer::TensorDim::DataType::FP32)
    throw std::invalid_argument("[pe_cls_pos] only supports FP32");
  const size_t patch_size = patches.getDim().getFeatureLen();
  const size_t cls_size = cls.getDim().getFeatureLen();
  for (unsigned int b = 0; b < patches.getDim().batch(); ++b) {
    const float *p = patches.getData<float>() + b * patch_size;
    const float *c = cls.getData<float>() + b * cls_size;
    const float *position = pos.getData<float>() + b * patch_size;
    float *o = out.getData<float>() + b * out.getDim().getFeatureLen();
    std::copy(c, c + cls_size, o);
    for (size_t i = 0; i < patch_size; ++i)
      o[cls_size + i] = p[i] + position[i];
  }
}
void PeClsPosLayer::incremental_forwarding(nntrainer::RunLayerContext &c,
                                           unsigned int, unsigned int,
                                           bool training) {
  forwarding(c, training);
}
void PeClsPosLayer::calcDerivative(nntrainer::RunLayerContext &) {
  throw std::runtime_error("[pe_cls_pos] training is not supported");
}
void PeClsPosLayer::setProperty(const std::vector<std::string> &values) {
  if (!values.empty())
    throw std::invalid_argument("[pe_cls_pos] does not accept properties");
}
void PeClsPosLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context, std::vector<nntrainer::TensorDim> dims) {
  for (size_t i = 0; i < dims.size(); ++i)
    context.updateInput(i, dims[i]);
  auto out = dims[0];
  out.height(dims[0].height() + dims[1].height());
  context.updateOutput(0, out);
}
} // namespace causallm
