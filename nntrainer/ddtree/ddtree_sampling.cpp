// SPDX-License-Identifier: Apache-2.0
/**
 * @file ddtree_sampling.cpp
 * @brief Greedy sampling convenience helpers.
 */
#include <ddtree_sampling.h>

#include <cstddef>

namespace nntrainer {
namespace ddtree {

int32_t argmaxRow(const float *logits, int vocab) {
  int32_t best = 0;
  float bestVal = logits[0];
  for (int i = 1; i < vocab; ++i) {
    if (logits[i] > bestVal) {
      bestVal = logits[i];
      best = i;
    }
  }
  return best;
}

void sampleGreedy(const float *logits, int rows, int vocab, int32_t *out) {
  for (int r = 0; r < rows; ++r)
    out[r] = argmaxRow(logits + static_cast<size_t>(r) * vocab, vocab);
}

} // namespace ddtree
} // namespace nntrainer
