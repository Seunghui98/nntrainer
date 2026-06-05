// SPDX-License-Identifier: Apache-2.0
/**
 * @file ddtree_sliding.cpp
 * @brief Sliding-window mask stage. Mirrors ddtree.py 219-255.
 */
#include <ddtree_sliding.h>

#include <cstring>

namespace nntrainer {
namespace ddtree {

SlidingMasks makeSlidingMasks(float *attentionMask,
                              const int32_t *verifyPositionIds,
                              int currentLength, int kvLength, int slidingWindow,
                              bool hasSlidingLayers, const DDTreeConfig &cfg,
                              float *slidingBuffer) {
  SlidingMasks out;
  out.full = attentionMask;

  // No sliding layers -> single mask for all layers (ddtree.py 227-229).
  if (!hasSlidingLayers) {
    out.hasSliding = false;
    out.sliding = nullptr;
    return out;
  }

  // window <= 0 -> full == sliding (ddtree.py 231-236).
  out.hasSliding = true;
  if (slidingWindow <= 0) {
    out.sliding = attentionMask;
    return out;
  }

  const int past = kvLength - currentLength;
  // key_positions: [:past] = arange, [past:] = verify_position_ids (238-240).
  // query_positions = verify_position_ids.
  // sliding_visible iff (key <= query) & (key > query - window) (242-245).
  std::memcpy(slidingBuffer, attentionMask,
              sizeof(float) * static_cast<size_t>(currentLength) * kvLength);
  for (int i = 0; i < currentLength; ++i) {
    const int q = verifyPositionIds[i];
    for (int j = 0; j < kvLength; ++j) {
      const int k = (j < past) ? j : verifyPositionIds[j - past];
      const bool visible = (k <= q) && (k > q - slidingWindow);
      if (!visible)
        slidingBuffer[static_cast<size_t>(i) * kvLength + j] = cfg.maskFillValue;
    }
  }
  out.sliding = slidingBuffer;
  return out;
}

} // namespace ddtree
} // namespace nntrainer
