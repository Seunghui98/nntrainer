// SPDX-License-Identifier: Apache-2.0
/**
 * @file unittest_ddtree.cpp
 * @brief Unit tests for the DDTree speculative-decoding core.
 */
#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include <ddtree.h>
#include <ddtree_types.h>

TEST(DDTreeScaffold, ConfigDefaults) {
  nntrainer::ddtree::DDTreeConfig cfg;
  EXPECT_EQ(cfg.budget, 31);
  EXPECT_EQ(cfg.depthLimit, 0);

  nntrainer::ddtree::DDTreeStructure root;
  EXPECT_EQ(root.currentLength, 1);
  EXPECT_EQ(root.nodeCount, 0);
}

using nntrainer::ddtree::buildTree;
using nntrainer::ddtree::DDTreeConfig;
using nntrainer::ddtree::DDTreeStructure;

TEST(DDTreeBuild, EmptyBudgetReturnsRootOnly) {
  std::vector<float> logits(8, 0.0f); // depthLimit=2, vocab=4, unused here
  DDTreeConfig cfg;
  cfg.budget = 0;
  DDTreeStructure t = buildTree(logits.data(), /*depthLimit=*/2, /*vocab=*/4, cfg);
  EXPECT_EQ(t.nodeCount, 0);
  EXPECT_EQ(t.currentLength, 1);
  ASSERT_EQ(t.parents.size(), 1u);
  EXPECT_EQ(t.parents[0], -1);
  ASSERT_EQ(t.childMaps.size(), 1u);
  EXPECT_TRUE(t.childMaps[0].empty());
  ASSERT_EQ(t.visibility.size(), 1u);
  EXPECT_EQ(t.visibility[0], 1u);
}

TEST(DDTreeBuild, ZeroDepthReturnsRootOnly) {
  std::vector<float> logits(1, 0.0f);
  DDTreeConfig cfg;
  cfg.budget = 8;
  DDTreeStructure t = buildTree(logits.data(), /*depthLimit=*/0, /*vocab=*/4, cfg);
  EXPECT_EQ(t.currentLength, 1);
  EXPECT_EQ(t.visibility[0], 1u);
}

// 2 depths, vocab 3. Row-major [depth, vocab].
// Depth 0 logits: token0 highest, token1 next, token2 lowest.
// Depth 1 logits: token2 highest, token0 next, token1 lowest.
static std::vector<float> kSmallLogits() {
  return {
    2.0f, 1.0f, 0.0f,  // depth 0
    0.0f, -1.0f, 1.0f, // depth 1
  };
}

TEST(DDTreeBuild, SmallTreeStructure) {
  DDTreeConfig cfg;
  cfg.budget = 3;
  auto logits = kSmallLogits();
  DDTreeStructure t = buildTree(logits.data(), /*depthLimit=*/2, /*vocab=*/3, cfg);

  // budget 3 -> 3 nodes, currentLength 4.
  EXPECT_EQ(t.nodeCount, 3);
  EXPECT_EQ(t.currentLength, 4);

  // First popped node: depth 1, rank 0 -> token_ids[0][0] = token 0.
  EXPECT_EQ(t.nodeDepths[0], 1);
  EXPECT_EQ(t.nodeTokenIds[0], 0);
  EXPECT_EQ(t.parents[1], 0); // child of root placeholder (index 0)

  // parents/visibility well-formed: root visible to all; each node sees itself.
  ASSERT_EQ(t.parents.size(), 4u);
  EXPECT_EQ(t.parents[0], -1);
  ASSERT_EQ(t.visibility.size(), 16u);
  int L = t.currentLength;
  EXPECT_EQ(t.visibility[0 * L + 0], 1u);
  for (int i = 1; i < L; ++i) {
    EXPECT_EQ(t.visibility[i * L + 0], 1u); // root always visible
    EXPECT_EQ(t.visibility[i * L + i], 1u); // self visible
  }
  // A node never sees a strictly-later node (upper triangle = 0).
  for (int i = 0; i < L; ++i)
    for (int j = i + 1; j < L; ++j)
      EXPECT_EQ(t.visibility[i * L + j], 0u);
}

TEST(DDTreeBuild, BudgetExceedsVocabTopkClamped) {
  DDTreeConfig cfg;
  cfg.budget = 100; // > vocab
  auto logits = kSmallLogits();
  // Must not read past topk = min(budget, vocab) = 3 columns; no crash.
  DDTreeStructure t = buildTree(logits.data(), /*depthLimit=*/2, /*vocab=*/3, cfg);
  EXPECT_GE(t.nodeCount, 1);
  EXPECT_LE(t.nodeCount, 100);
}

TEST(DDTreeCompile, IdsPositionsAndMask) {
  using nntrainer::ddtree::compile;
  using nntrainer::ddtree::CompiledTree;

  DDTreeConfig cfg;
  cfg.budget = 3;
  cfg.maskFillValue = -1e30f;
  auto logits = kSmallLogits();
  DDTreeStructure t = buildTree(logits.data(), 2, 3, cfg);

  const int past = 2;
  const int start = 5;
  const int32_t rootToken = 99;
  const int L = t.currentLength;
  std::vector<int32_t> ids(L), pos(L);
  const int stride = past + L;
  std::vector<float> mask(static_cast<size_t>(L) * stride, 7.0f); // sentinel

  CompiledTree c = compile(rootToken, start, past, t, cfg, ids.data(),
                           pos.data(), mask.data(), stride);
  EXPECT_EQ(c.pastLength, past);
  EXPECT_EQ(c.currentLength, L);

  // ids[0] = root; ids[i] = node token.
  EXPECT_EQ(ids[0], rootToken);
  for (int i = 1; i < L; ++i)
    EXPECT_EQ(ids[i], t.nodeTokenIds[i - 1]);

  // pos[0] = start; pos[i] = start + depth.
  EXPECT_EQ(pos[0], start);
  for (int i = 1; i < L; ++i)
    EXPECT_EQ(pos[i], start + t.nodeDepths[i - 1]);

  // Prefix block [0, past) fully visible (== 0) for every row.
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < past; ++j)
      EXPECT_EQ(mask[i * stride + j], 0.0f);

  // Tree block [past, past+L) == visibility ? 0 : fill.
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < L; ++j) {
      float expected = t.visibility[i * L + j] ? 0.0f : cfg.maskFillValue;
      EXPECT_EQ(mask[i * stride + past + j], expected);
    }
}

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;
  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Failed to init gtest\n";
  }
  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Failed to run test.\n";
  }
  return result;
}
