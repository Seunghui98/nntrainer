// SPDX-License-Identifier: Apache-2.0
/**
 * @file ddtree_sampling.h
 * @brief Convenience greedy sampling (temperature 0). The caller owns real
 *        sampling; this mirrors sample(logits, temperature=0) == argmax.
 */
#ifndef __NNTRAINER_DDTREE_SAMPLING_H__
#define __NNTRAINER_DDTREE_SAMPLING_H__

#include <cstdint>

namespace nntrainer {
namespace ddtree {

/** argmax over one row of `vocab` logits; ties resolve to the lowest index. */
int32_t argmaxRow(const float *logits, int vocab);

/** Per-row argmax for `rows` rows of `vocab` logits into out[rows]. */
void sampleGreedy(const float *logits, int rows, int vocab, int32_t *out);

} // namespace ddtree
} // namespace nntrainer

#endif // __NNTRAINER_DDTREE_SAMPLING_H__
