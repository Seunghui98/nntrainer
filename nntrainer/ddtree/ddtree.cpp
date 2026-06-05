// SPDX-License-Identifier: Apache-2.0
/**
 * @file ddtree.cpp
 * @brief DDTree core implementation. Mirrors ddtree.py bit-for-bit.
 */
#include <ddtree.h>

#include <algorithm>
#include <cmath>
#include <queue>

namespace nntrainer {
namespace ddtree {

DDTreeStructure buildTree(const float *draftLogits, int depthLimit, int vocab,
                          const DDTreeConfig &cfg) {
  DDTreeStructure t;

  // Empty case (ddtree.py 98-108): budget<=0 or depth_limit==0 -> root only.
  if (cfg.budget <= 0 || depthLimit == 0) {
    t.nodeCount = 0;
    t.currentLength = 1;
    t.parents = {-1};
    t.childMaps.resize(1);
    t.visibility = {1};
    return t;
  }

  // Heap expansion added in Task 3.
  (void)draftLogits;
  (void)vocab;
  return t;
}

} // namespace ddtree
} // namespace nntrainer
