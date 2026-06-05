// SPDX-License-Identifier: Apache-2.0
/**
 * @file ddtree_sliding.h
 * @brief Sliding-window additive-mask stage
 *        (== prepare_ddtree_attention_mask_for_target).
 */
#ifndef __NNTRAINER_DDTREE_SLIDING_H__
#define __NNTRAINER_DDTREE_SLIDING_H__

#include <cstdint>

#include <ddtree_types.h>

namespace nntrainer {
namespace ddtree {

/**
 * @brief Build full/sliding additive masks for a tree verify pass.
 *        Operates on contiguous row-major [currentLength, kvLength] buffers.
 * @param attentionMask     full mask, [currentLength, kvLength] (from compile)
 * @param verifyPositionIds [currentLength] absolute query positions
 * @param currentLength     number of tree nodes (rows)
 * @param kvLength          pastLength + currentLength (mask columns)
 * @param slidingWindow     model sliding_window (<=0 => full==sliding)
 * @param hasSlidingLayers  true iff model layer_types contains "sliding_attention"
 * @param cfg               uses cfg.maskFillValue
 * @param slidingBuffer     [currentLength, kvLength] scratch for sliding variant
 */
SlidingMasks makeSlidingMasks(float *attentionMask,
                              const int32_t *verifyPositionIds,
                              int currentLength, int kvLength, int slidingWindow,
                              bool hasSlidingLayers, const DDTreeConfig &cfg,
                              float *slidingBuffer);

} // namespace ddtree
} // namespace nntrainer

#endif // __NNTRAINER_DDTREE_SLIDING_H__
