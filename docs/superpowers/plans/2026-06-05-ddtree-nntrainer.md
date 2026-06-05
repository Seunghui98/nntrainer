# DDTree (Tree Speculative Decoding) for nntrainer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the DDTree tree-based speculative-decoding algorithm from `/home/shsh1004/littlesd_inference/ddtree.py` into nntrainer as a stateless, reusable C++ core plus the minimal runtime hooks (additive attention mask in `mha_core`, `KVCacheManager::compactTail`) needed to run and verify token-for-token parity with Python gemma4 on CPU.

**Architecture:** A pure, cache-agnostic algorithm core at `nntrainer/nntrainer/ddtree/` (build tree → compile verify ids/pos/mask → sliding-window mask → follow accepted path → keep-index + raw-pointer compaction helper). The model runtime owns the KV cache and forwards; nntrainer's `mha_core` gains an additive-mask path and `KVCacheManager` gains tail compaction. Verification is two-layer: a fast trace-replay parity test (model-free) and a live gemma4-on-CPU harness.

**Tech Stack:** C++17 (stdlib only in the core), Meson/Ninja build, GoogleTest, nntrainer `Tensor`, gemma4 CausalLM. Golden vectors dumped from the Python reference.

**Source of truth:** `ddtree.py`. Every algorithmic decision must match it bit-for-bit. The 7 parity details are in spec §6 (`docs/superpowers/specs/2026-06-05-ddtree-nntrainer-design.md`).

---

## File Structure

**New — pure core (`nntrainer/nntrainer/ddtree/`):**
- `meson.build` — declares `ddtree_sources` / `ddtree_headers`, appends to `nntrainer_sources` / `nntrainer_headers`.
- `ddtree_types.h` — `DDTreeConfig`, `DDTreeStructure`, `CompiledTree`, `SlidingMasks`, `Accepted`.
- `ddtree.h` / `ddtree.cpp` — `buildTree`, `compile`, `followVerified`.
- `ddtree_sliding.h` / `ddtree_sliding.cpp` — `makeSlidingMasks`.
- `ddtree_compact.h` / `ddtree_compact.cpp` — `compactTail` raw-pointer helper.
- `ddtree_sampling.h` / `ddtree_sampling.cpp` — `argmax` / greedy helper.
- `README.md` — architecture, structure, usage guide (Task 16).

**Modified — build registration:**
- `nntrainer/nntrainer/meson.build` — add `'ddtree'` to `nntrainer_elements`.
- `test/unittest/meson.build` — register `unittest_ddtree`.

**Modified — runtime:**
- `Applications/CausalLM/layers/mha_core.h` / `.cpp` — additive-mask attention path.
- `Applications/CausalLM/kv_cache_manager.h` / `.cpp` — `compactTail` method.

**New — tests & tooling:**
- `test/unittest/unittest_ddtree.cpp` — all core gtest suites.
- `test/unittest/ddtree_golden/` — golden JSON vectors dumped from Python.
- `test/unittest/ddtree_golden/gen_golden.py` — standalone Python dumper.
- `Applications/CausalLM/tools/ddtree_cpu_harness.cpp` — live gemma4-on-CPU DDTree loop.

**Note on signature refinements vs spec §5:** the spec's API surface is illustrative. Two concrete refinements (documented in code comments and the README):
1. `makeSlidingMasks` operates on contiguous row-major `[currentLength, kvLength]` buffers and takes `hasSlidingLayers` + `kvLength` explicitly (the model config decides `hasSlidingLayers`, which the pure core cannot know).
2. `compile`'s `attnMaskRowStride` lets the caller back the mask with a wider pre-allocated buffer; the sliding stage assumes contiguous rows (`stride == kvLength`) for simplicity. The harness allocates the mask contiguously so both agree.

---

## Parity Reference Card (keep open while implementing)

From `ddtree.py` `build_ddtree_tree` (lines 92–174) and spec §6:

- **Heap entry** (line 126): tuple `(-logw, ranks_tuple, parent_index, depth, rank, logw)`, popped by `heapq` (min-heap, lexicographic). Effective ordering: `-logw → ranks → parent_index → depth → rank`. `ranks_tuple` is variable-length; Python tuple compare = `std::vector` lexicographic (shorter prefix sorts first).
- **Numerics** (lines 114–117, 149, 154): `top_log_probs` is **fp32**; `logw` accumulates in **double** (`float(...)` promotion). `topk = min(budget, vocab)` (line 110).
- **Seed** (lines 125–126): `first_logw = top_log_probs[0,0]`; heap = `[(-first_logw, (0,), 0, 1, 0, first_logw)]`. Root parent_index is `0` (the placeholder root node index), depth starts at `1`.
- **Pop body** (lines 136–155): `token_id = top_token_ids[depth-1][rank]`; `current_index = node_count+1`; record token/depth/parent; `child_maps[parent_index][token_id] = current_index`. Sibling pushed iff `rank+1 < topk`; child pushed iff `depth < depth_limit`.
- **Empty case** (lines 98–108): `budget<=0` or `depth_limit==0` → root only: `visibility=[[1]]`, `parents=[-1]`, `child_maps=[{}]`, `node_count=0`, `current_length=1`.
- **Visibility** (lines 160–167): `vis[0,0]=1`; for `index` in `[1,current_length)`: `vis[index,:index] = vis[parent,:index]`, `vis[index,index]=1`.
- **compile** (lines 193–216): `pos[0]=start`, `pos[i]=start+depth[i-1]`; mask zeroed (prefix block all-0/visible), tree block `[:, past:past+cur]` filled `finfo.min` then `0` where `visibility==1`.
- **sliding** (lines 219–255): no `sliding_attention` in `layer_types` → return full unchanged; `window<=0` → full==sliding; else `key_pos[:past]=arange`, `key_pos[past:]=verify_pos`, visible iff `(key<=query) & (key>query-window)`.
- **follow** (lines 282–293): `accepted=[0]`, walk `child_maps` by `posterior[current_index]` until token not a child.
- **keep-index** (lines 595–600): `keepIndices == accepted_indices` (includes root `0`); assert physical tail length == `currentLength`; final length = `pastLen + keepCount`.

---

# Phase A — Pure core module

### Task 1: Module scaffolding, meson wiring, and `ddtree_types.h`

**Files:**
- Create: `nntrainer/nntrainer/ddtree/ddtree_types.h`
- Create: `nntrainer/nntrainer/ddtree/meson.build`
- Modify: `nntrainer/nntrainer/meson.build` (add `'ddtree'` to `nntrainer_elements`, around line 44–54)
- Create: `test/unittest/unittest_ddtree.cpp`
- Modify: `test/unittest/meson.build` (register `unittest_ddtree`, around line 57)

- [ ] **Step 1: Create `ddtree_types.h`**

```cpp
// SPDX-License-Identifier: Apache-2.0
/**
 * @file   ddtree_types.h
 * @brief  Plain-old-data structures for the DDTree speculative-decoding core.
 *         Pure host logic; no nntrainer/Tensor dependency. Mirrors ddtree.py.
 */
#ifndef __NNTRAINER_DDTREE_TYPES_H__
#define __NNTRAINER_DDTREE_TYPES_H__

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace nntrainer {
namespace ddtree {

/** Runtime configuration. budget = max tree nodes minus 1 ("32 tree" = budget 31). */
struct DDTreeConfig {
  int budget = 31;          /**< heap expansion cap; "32 tree" == budget 31 */
  int depthLimit = 0;       /**< draft horizon = block_size - 1 */
  float maskFillValue = 0;  /**< additive-mask -inf == finfo(dtype).min (fp32 or fp16 min) */
};

/** Output of buildTree (== build_ddtree_tree). currentLength = 1 + nodeCount. */
struct DDTreeStructure {
  std::vector<int32_t> nodeTokenIds;  /**< [nodeCount] */
  std::vector<int32_t> nodeDepths;    /**< [nodeCount], 1-based depth */
  std::vector<int32_t> parents;       /**< [currentLength], parents[0] = -1 */
  std::vector<std::unordered_map<int32_t, int32_t>> childMaps; /**< [currentLength] */
  std::vector<uint8_t> visibility;    /**< [currentLength*currentLength] row-major bitmap */
  int nodeCount = 0;
  int currentLength = 1;
};

/** Slice lengths returned by compile (== compile_ddtree_tree). */
struct CompiledTree {
  int pastLength = 0;
  int currentLength = 0;
};

/** Result of makeSlidingMasks. When hasSliding is false, use `full` for all layers. */
struct SlidingMasks {
  float *full = nullptr;     /**< full-attention layers */
  float *sliding = nullptr;  /**< sliding-attention layers (== full when window<=0) */
  bool hasSliding = false;   /**< true => select full vs sliding per layer type */
};

/** Result of followVerified (== follow_verified_tree). */
struct Accepted {
  std::vector<int32_t> indices;  /**< accepted node indices, indices[0]==0 (root) */
  int32_t nextToken = 0;         /**< bonus token after the accepted path */
};

} // namespace ddtree
} // namespace nntrainer

#endif // __NNTRAINER_DDTREE_TYPES_H__
```

- [ ] **Step 2: Create `nntrainer/nntrainer/ddtree/meson.build`**

```meson
ddtree_sources = [
  'ddtree.cpp',
  'ddtree_sliding.cpp',
  'ddtree_compact.cpp',
  'ddtree_sampling.cpp',
]

ddtree_headers = [
  'ddtree_types.h',
  'ddtree.h',
  'ddtree_sliding.h',
  'ddtree_compact.h',
  'ddtree_sampling.h',
]

foreach s : ddtree_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : ddtree_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
```

- [ ] **Step 3: Register the module.** In `nntrainer/nntrainer/meson.build`, add `'ddtree'` to the `nntrainer_elements` list (the list currently ends with `'graph'`). Result:

```meson
nntrainer_elements = [
  'compiler',
  'schema',
  'dataset',
  'layers',
  'models',
  'optimizers',
  'tensor',
  'utils',
  'graph',
  'ddtree'
]
```

- [ ] **Step 4: Write the failing scaffolding test.** Create `test/unittest/unittest_ddtree.cpp`:

```cpp
// SPDX-License-Identifier: Apache-2.0
/**
 * @file unittest_ddtree.cpp
 * @brief Unit tests for the DDTree speculative-decoding core.
 */
#include <gtest/gtest.h>

#include <ddtree_types.h>

TEST(DDTreeScaffold, ConfigDefaults) {
  nntrainer::ddtree::DDTreeConfig cfg;
  EXPECT_EQ(cfg.budget, 31);
  EXPECT_EQ(cfg.depthLimit, 0);

  nntrainer::ddtree::DDTreeStructure root;
  EXPECT_EQ(root.currentLength, 1);
  EXPECT_EQ(root.nodeCount, 0);
}
```

- [ ] **Step 5: Register the test target.** In `test/unittest/meson.build`, add to the `test_target` list (near the other `['unittest_nntrainer_*', []]` entries):

```meson
  ['unittest_ddtree', []],
```

- [ ] **Step 6: Configure the build (one-time).** Run from repo root:

```bash
cd /home/shsh1004/nntrainer
meson setup build -Denable-fp16=true -Denable-test=true
```
Expected: configuration succeeds, `build/` created. If `build/` already exists, use `meson setup --reconfigure build -Denable-fp16=true -Denable-test=true`.

- [ ] **Step 7: Compile and run the scaffolding test.**

```bash
ninja -C build unittest_ddtree && meson test -C build unittest_ddtree -v
```
Expected: PASS (`DDTreeScaffold.ConfigDefaults`). This proves meson wiring + header install path resolve.

- [ ] **Step 8: Commit.**

```bash
git add nntrainer/nntrainer/ddtree/ddtree_types.h nntrainer/nntrainer/ddtree/meson.build \
        nntrainer/nntrainer/meson.build test/unittest/unittest_ddtree.cpp test/unittest/meson.build
git commit -m "feat(ddtree): scaffold pure core module + meson wiring + scaffold test"
```

---

### Task 2: `buildTree` — empty / root-only case

**Files:**
- Create: `nntrainer/nntrainer/ddtree/ddtree.h`
- Create: `nntrainer/nntrainer/ddtree/ddtree.cpp`
- Test: `test/unittest/unittest_ddtree.cpp`

- [ ] **Step 1: Write the failing test.** Append to `unittest_ddtree.cpp`:

```cpp
#include <ddtree.h>

using nntrainer::ddtree::buildTree;
using nntrainer::ddtree::DDTreeConfig;
using nntrainer::ddtree::DDTreeStructure;

TEST(DDTreeBuild, EmptyBudgetReturnsRootOnly) {
  std::vector<float> logits(8, 0.0f); // depthLimit=2, vocab=4 worth, unused
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
```

- [ ] **Step 2: Create `ddtree.h`** (full declarations used through Phase A):

```cpp
// SPDX-License-Identifier: Apache-2.0
/**
 * @file ddtree.h
 * @brief DDTree core: tree build, verify-buffer compile, accepted-path follow.
 *        Pure host logic mirroring ddtree.py (build/compile/follow).
 */
#ifndef __NNTRAINER_DDTREE_H__
#define __NNTRAINER_DDTREE_H__

#include <cstdint>

#include <ddtree_types.h>

namespace nntrainer {
namespace ddtree {

/**
 * @brief Build the best-first candidate token tree (== build_ddtree_tree).
 * @param draftLogits row-major [depthLimit, vocab] draft logits (fp32)
 * @param depthLimit  draft horizon (block_size - 1)
 * @param vocab       vocabulary size (row width of draftLogits)
 * @param cfg         config; uses cfg.budget
 */
DDTreeStructure buildTree(const float *draftLogits, int depthLimit, int vocab,
                          const DDTreeConfig &cfg);

/**
 * @brief Flatten a tree into verify ids / position ids / additive mask
 *        (== compile_ddtree_tree). Writes into caller buffers.
 * @param rootTokenId      token at tree root (block_output_ids[:, 0])
 * @param start            absolute position of the root
 * @param pastLength       length of the already-cached prefix
 * @param tree             output of buildTree
 * @param cfg              config; uses cfg.maskFillValue
 * @param verifyInputIds   [currentLength] out
 * @param verifyPositionIds[currentLength] out
 * @param attentionMask    [currentLength, pastLength+currentLength] out (row stride below)
 * @param attnMaskRowStride elements between consecutive mask rows (>= pastLength+currentLength)
 */
CompiledTree compile(int32_t rootTokenId, int start, int pastLength,
                     const DDTreeStructure &tree, const DDTreeConfig &cfg,
                     int32_t *verifyInputIds, int32_t *verifyPositionIds,
                     float *attentionMask, int attnMaskRowStride);

/**
 * @brief Walk the longest accepted root->leaf path (== follow_verified_tree).
 * @param childMaps tree.childMaps
 * @param posterior [currentLength] per-node sampled token id
 */
Accepted followVerified(
  const std::vector<std::unordered_map<int32_t, int32_t>> &childMaps,
  const int32_t *posterior);

} // namespace ddtree
} // namespace nntrainer

#endif // __NNTRAINER_DDTREE_H__
```

- [ ] **Step 3: Create `ddtree.cpp` with the empty-case `buildTree` only** (heap loop added in Task 3):

```cpp
// SPDX-License-Identifier: Apache-2.0
/**
 * @file ddtree.cpp
 * @brief DDTree core implementation. Mirrors ddtree.py bit-for-bit.
 */
#include <ddtree.h>

#include <algorithm>
#include <cmath>
#include <queue>
#include <tuple>

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
```

- [ ] **Step 4: Compile and run.**

```bash
ninja -C build unittest_ddtree && meson test -C build unittest_ddtree -v
```
Expected: `DDTreeBuild.EmptyBudgetReturnsRootOnly` and `DDTreeBuild.ZeroDepthReturnsRootOnly` PASS.

- [ ] **Step 5: Commit.**

```bash
git add nntrainer/nntrainer/ddtree/ddtree.h nntrainer/nntrainer/ddtree/ddtree.cpp test/unittest/unittest_ddtree.cpp
git commit -m "feat(ddtree): buildTree empty/root-only case"
```

---

### Task 3: `buildTree` — heap expansion, tie-break, visibility (PARITY-CRITICAL)

This is the highest-risk task (spec §6.1/§6.2/§6.3). Implement the heap key comparator to match Python `heapq` tuple ordering exactly, accumulate `logw` in `double`, keep `top_log_probs` in `float`.

**Files:**
- Modify: `nntrainer/nntrainer/ddtree/ddtree.cpp`
- Test: `test/unittest/unittest_ddtree.cpp`

- [ ] **Step 1: Write the failing test** using a small hand-computable fixture. Append to `unittest_ddtree.cpp`:

```cpp
// 2 depths, vocab 3. Row-major [depth, vocab].
// Depth 0 logits: token0 highest, token1 next, token2 lowest.
// Depth 1 logits: token2 highest, token0 next, token1 lowest.
static std::vector<float> kSmallLogits() {
  return {
    2.0f, 1.0f, 0.0f,   // depth 0
    0.0f, -1.0f, 1.0f,  // depth 1
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
    EXPECT_EQ(t.visibility[i * L + 0], 1u);  // root always visible
    EXPECT_EQ(t.visibility[i * L + i], 1u);  // self visible
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
```

- [ ] **Step 2: Run to confirm failure.**

```bash
ninja -C build unittest_ddtree && meson test -C build unittest_ddtree -v
```
Expected: `DDTreeBuild.SmallTreeStructure` FAILS (nodeCount 0).

- [ ] **Step 3: Implement the heap loop.** Replace the "Heap expansion added in Task 3" placeholder block in `ddtree.cpp` (keep the empty-case guard above it) with:

```cpp
  const int topk = std::min(cfg.budget, vocab);

  // Per-row top-k: top_log_probs (fp32) and top_token_ids, sorted by
  // (logit desc, token index asc) to match torch.topk on CPU.
  // topLogProbs[d][r] = top_logit - logsumexp(row d). (ddtree.py 114-117)
  std::vector<std::vector<float>> topLogProbs(depthLimit);
  std::vector<std::vector<int32_t>> topTokenIds(depthLimit);
  for (int d = 0; d < depthLimit; ++d) {
    const float *row = draftLogits + static_cast<size_t>(d) * vocab;

    std::vector<int32_t> idx(vocab);
    for (int i = 0; i < vocab; ++i)
      idx[i] = i;
    std::partial_sort(
      idx.begin(), idx.begin() + topk, idx.end(),
      [row](int32_t a, int32_t b) {
        if (row[a] != row[b])
          return row[a] > row[b]; // logit desc
        return a < b;             // tie-break: lower token index first
      });

    // logsumexp in double for accuracy; result stored as fp32 (parity §6.2).
    float maxLogit = row[0];
    for (int i = 1; i < vocab; ++i)
      maxLogit = std::max(maxLogit, row[i]);
    double sumExp = 0.0;
    for (int i = 0; i < vocab; ++i)
      sumExp += std::exp(static_cast<double>(row[i]) - maxLogit);
    const float logZ = static_cast<float>(maxLogit + std::log(sumExp));

    topLogProbs[d].resize(topk);
    topTokenIds[d].resize(topk);
    for (int r = 0; r < topk; ++r) {
      topTokenIds[d][r] = idx[r];
      topLogProbs[d][r] = row[idx[r]] - logZ; // fp32 subtraction
    }
  }

  // Heap entry mirrors the Python tuple (-logw, ranks, parent, depth, rank, logw).
  // Comparison is the Python tuple order; ranks is variable-length lexicographic.
  struct Entry {
    double negLogw;
    std::vector<int32_t> ranks;
    int parentIndex;
    int depth;
    int rank;
    double logw;
  };
  // pythonLess(a,b) == (a < b) under Python tuple comparison.
  auto pythonLess = [](const Entry &a, const Entry &b) {
    if (a.negLogw != b.negLogw)
      return a.negLogw < b.negLogw;
    if (a.ranks != b.ranks)
      return a.ranks < b.ranks; // std::vector lexicographic (prefix sorts first)
    if (a.parentIndex != b.parentIndex)
      return a.parentIndex < b.parentIndex;
    if (a.depth != b.depth)
      return a.depth < b.depth;
    if (a.rank != b.rank)
      return a.rank < b.rank;
    return a.logw < b.logw;
  };
  // priority_queue top == Python-smallest -> comp(x,y) = pythonLess(y,x).
  auto comp = [&pythonLess](const Entry &x, const Entry &y) {
    return pythonLess(y, x);
  };
  std::priority_queue<Entry, std::vector<Entry>, decltype(comp)> heap(comp);

  const double firstLogw = static_cast<double>(topLogProbs[0][0]);
  heap.push(Entry{-firstLogw, {0}, /*parent*/ 0, /*depth*/ 1, /*rank*/ 0, firstLogw});

  t.parents.assign(cfg.budget + 1, 0);
  t.parents[0] = -1;
  t.childMaps.clear();
  t.childMaps.emplace_back(); // root placeholder child map
  t.nodeTokenIds.clear();
  t.nodeDepths.clear();

  int nodeCount = 0;
  while (!heap.empty() && nodeCount < cfg.budget) {
    Entry e = heap.top();
    heap.pop();

    const int32_t tokenId = topTokenIds[e.depth - 1][e.rank];
    const int currentIndex = nodeCount + 1;
    t.nodeTokenIds.push_back(tokenId);
    t.nodeDepths.push_back(e.depth);
    t.parents[currentIndex] = e.parentIndex;
    t.childMaps.emplace_back();
    t.childMaps[e.parentIndex][tokenId] = currentIndex;
    ++nodeCount;

    if (e.rank + 1 < topk) {
      std::vector<int32_t> siblingRanks = e.ranks;
      siblingRanks.back() = e.rank + 1; // ranks[:-1] + (rank+1,)
      const double siblingLogw = e.logw -
        static_cast<double>(topLogProbs[e.depth - 1][e.rank]) +
        static_cast<double>(topLogProbs[e.depth - 1][e.rank + 1]);
      heap.push(Entry{-siblingLogw, std::move(siblingRanks), e.parentIndex,
                      e.depth, e.rank + 1, siblingLogw});
    }

    if (e.depth < depthLimit) {
      std::vector<int32_t> childRanks = e.ranks;
      childRanks.push_back(0); // ranks + (0,)
      const double childLogw =
        e.logw + static_cast<double>(topLogProbs[e.depth][0]);
      heap.push(Entry{-childLogw, std::move(childRanks), currentIndex,
                      e.depth + 1, 0, childLogw});
    }
  }

  // Visibility bitmap (ddtree.py 160-167).
  const int currentLength = 1 + nodeCount;
  t.nodeCount = nodeCount;
  t.currentLength = currentLength;
  t.parents.resize(currentLength);
  t.visibility.assign(static_cast<size_t>(currentLength) * currentLength, 0);
  t.visibility[0] = 1; // vis[0,0]
  for (int index = 1; index < currentLength; ++index) {
    const int parent = t.parents[index];
    for (int j = 0; j < index; ++j)
      t.visibility[static_cast<size_t>(index) * currentLength + j] =
        t.visibility[static_cast<size_t>(parent) * currentLength + j];
    t.visibility[static_cast<size_t>(index) * currentLength + index] = 1;
  }
  return t;
```

Remove the now-dead `(void)draftLogits; (void)vocab; return t;` lines.

- [ ] **Step 4: Compile and run.**

```bash
ninja -C build unittest_ddtree && meson test -C build unittest_ddtree -v
```
Expected: all `DDTreeBuild.*` PASS.

- [ ] **Step 5: Commit.**

```bash
git add nntrainer/nntrainer/ddtree/ddtree.cpp test/unittest/unittest_ddtree.cpp
git commit -m "feat(ddtree): buildTree heap expansion + tie-break + visibility"
```

> Golden-vector parity for `buildTree` (against Python on the same fixed logits) is added in Task 11–12. This task locks structure; Task 12 locks exact token ids/depths.

---

### Task 4: `compile` — verify ids / position ids / additive mask

**Files:**
- Modify: `nntrainer/nntrainer/ddtree/ddtree.cpp`
- Test: `test/unittest/unittest_ddtree.cpp`

- [ ] **Step 1: Write the failing test.** Append:

```cpp
#include <ddtree.h>
using nntrainer::ddtree::compile;
using nntrainer::ddtree::CompiledTree;

TEST(DDTreeCompile, IdsPositionsAndMask) {
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
```

- [ ] **Step 2: Run to confirm failure** (link error: `compile` undefined).

```bash
ninja -C build unittest_ddtree
```
Expected: undefined reference to `compile`.

- [ ] **Step 3: Implement `compile`.** Append to `ddtree.cpp` (inside the namespace, after `buildTree`):

```cpp
CompiledTree compile(int32_t rootTokenId, int start, int pastLength,
                     const DDTreeStructure &tree, const DDTreeConfig &cfg,
                     int32_t *verifyInputIds, int32_t *verifyPositionIds,
                     float *attentionMask, int attnMaskRowStride) {
  const int cur = tree.currentLength;

  // verify_input_ids: [root, node tokens...] (ddtree.py 197-200).
  verifyInputIds[0] = rootTokenId;
  for (int i = 1; i < cur; ++i)
    verifyInputIds[i] = tree.nodeTokenIds[i - 1];

  // verify_position_ids: [start, start+depth...] (ddtree.py 202-206).
  verifyPositionIds[0] = start;
  for (int i = 1; i < cur; ++i)
    verifyPositionIds[i] = start + tree.nodeDepths[i - 1];

  // Mask (ddtree.py 195, 211-213): prefix block 0 (visible); tree block
  // filled with maskFillValue then opened (0) where visibility == 1.
  const int kv = pastLength + cur;
  for (int i = 0; i < cur; ++i) {
    float *r = attentionMask + static_cast<size_t>(i) * attnMaskRowStride;
    for (int j = 0; j < pastLength; ++j)
      r[j] = 0.0f;
    for (int j = 0; j < cur; ++j)
      r[pastLength + j] =
        tree.visibility[static_cast<size_t>(i) * cur + j] ? 0.0f
                                                          : cfg.maskFillValue;
  }
  (void)kv;

  return CompiledTree{pastLength, cur};
}
```

- [ ] **Step 4: Compile and run.**

```bash
ninja -C build unittest_ddtree && meson test -C build unittest_ddtree -v
```
Expected: `DDTreeCompile.IdsPositionsAndMask` PASS.

- [ ] **Step 5: Commit.**

```bash
git add nntrainer/nntrainer/ddtree/ddtree.cpp test/unittest/unittest_ddtree.cpp
git commit -m "feat(ddtree): compile verify ids/positions/additive-mask"
```

---

### Task 5: `makeSlidingMasks` — sliding-window mask stage

**Files:**
- Create: `nntrainer/nntrainer/ddtree/ddtree_sliding.h`
- Create: `nntrainer/nntrainer/ddtree/ddtree_sliding.cpp`
- Test: `test/unittest/unittest_ddtree.cpp`

- [ ] **Step 1: Write the failing test.** Append:

```cpp
#include <ddtree_sliding.h>
using nntrainer::ddtree::makeSlidingMasks;
using nntrainer::ddtree::SlidingMasks;

TEST(DDTreeSliding, NoSlidingLayersPassthrough) {
  std::vector<float> full(4, 0.0f);
  std::vector<int32_t> pos = {0, 1};
  std::vector<float> slide(4, -1.0f);
  DDTreeConfig cfg;
  SlidingMasks m = makeSlidingMasks(full.data(), pos.data(),
                                    /*currentLength=*/2, /*kvLength=*/2,
                                    /*slidingWindow=*/8,
                                    /*hasSlidingLayers=*/false, cfg, slide.data());
  EXPECT_FALSE(m.hasSliding);
  EXPECT_EQ(m.full, full.data());
}

TEST(DDTreeSliding, WindowZeroFullEqualsSliding) {
  std::vector<float> full(4, 0.0f);
  std::vector<int32_t> pos = {0, 1};
  std::vector<float> slide(4, -1.0f);
  DDTreeConfig cfg;
  SlidingMasks m = makeSlidingMasks(full.data(), pos.data(), 2, 2,
                                    /*slidingWindow=*/0,
                                    /*hasSlidingLayers=*/true, cfg, slide.data());
  EXPECT_TRUE(m.hasSliding);
  EXPECT_EQ(m.sliding, m.full);
}

TEST(DDTreeSliding, PositiveWindowVisibility) {
  // past=1, cur=2, kv=3. positions: query = [5,6]; key[:past]=arange=[0], key[past:]=[5,6].
  const int past = 1, cur = 2, kv = 3, window = 2;
  std::vector<int32_t> pos = {5, 6};
  std::vector<float> full(static_cast<size_t>(cur) * kv, 0.0f); // all visible
  std::vector<float> slide(static_cast<size_t>(cur) * kv, 123.0f);
  DDTreeConfig cfg;
  cfg.maskFillValue = -1e30f;
  SlidingMasks m = makeSlidingMasks(full.data(), pos.data(), cur, kv, window,
                                    /*hasSlidingLayers=*/true, cfg, slide.data());
  ASSERT_TRUE(m.hasSliding);
  ASSERT_NE(m.sliding, m.full);

  // key positions: [0, 5, 6]. visible iff key<=query && key>query-window.
  auto vis = [&](int qi, int kj) {
    int q = pos[qi];
    int k = (kj < past) ? kj : pos[kj - past];
    return (k <= q) && (k > q - window);
  };
  for (int i = 0; i < cur; ++i)
    for (int j = 0; j < kv; ++j) {
      float expected = vis(i, j) ? 0.0f : cfg.maskFillValue;
      EXPECT_EQ(m.sliding[i * kv + j], expected) << "i=" << i << " j=" << j;
    }
}
```

- [ ] **Step 2: Create `ddtree_sliding.h`.**

```cpp
// SPDX-License-Identifier: Apache-2.0
/**
 * @file ddtree_sliding.h
 * @brief Sliding-window additive-mask stage (== prepare_ddtree_attention_mask_for_target).
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
 * @param attentionMask    full mask, [currentLength, kvLength] (from compile)
 * @param verifyPositionIds[currentLength] absolute query positions
 * @param currentLength    number of tree nodes (rows)
 * @param kvLength         pastLength + currentLength (mask columns)
 * @param slidingWindow    model sliding_window (<=0 => full==sliding)
 * @param hasSlidingLayers true iff model layer_types contains "sliding_attention"
 * @param cfg              uses cfg.maskFillValue
 * @param slidingBuffer    [currentLength, kvLength] scratch for the sliding variant
 */
SlidingMasks makeSlidingMasks(float *attentionMask,
                              const int32_t *verifyPositionIds,
                              int currentLength, int kvLength, int slidingWindow,
                              bool hasSlidingLayers, const DDTreeConfig &cfg,
                              float *slidingBuffer);

} // namespace ddtree
} // namespace nntrainer

#endif // __NNTRAINER_DDTREE_SLIDING_H__
```

- [ ] **Step 3: Create `ddtree_sliding.cpp`.**

```cpp
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
  // key_positions: [:past] = arange, [past:] = verify_position_ids (ddtree.py 238-240).
  // query_positions = verify_position_ids.
  // sliding_visible iff (key <= query) & (key > query - window) (ddtree.py 242-245).
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
```

- [ ] **Step 4: Compile and run.**

```bash
ninja -C build unittest_ddtree && meson test -C build unittest_ddtree -v
```
Expected: all `DDTreeSliding.*` PASS.

- [ ] **Step 5: Commit.**

```bash
git add nntrainer/nntrainer/ddtree/ddtree_sliding.h nntrainer/nntrainer/ddtree/ddtree_sliding.cpp test/unittest/unittest_ddtree.cpp
git commit -m "feat(ddtree): sliding-window additive-mask stage"
```

---

### Task 6: `followVerified` — accepted-path walk

**Files:**
- Modify: `nntrainer/nntrainer/ddtree/ddtree.cpp`
- Test: `test/unittest/unittest_ddtree.cpp`

- [ ] **Step 1: Write the failing test.** Append:

```cpp
using nntrainer::ddtree::Accepted;
using nntrainer::ddtree::followVerified;

// Build a fixed child_maps by hand: root(0) -> {10:1}; node1 -> {20:2}; node2 -> {}.
static std::vector<std::unordered_map<int32_t, int32_t>> kChain() {
  std::vector<std::unordered_map<int32_t, int32_t>> cm(3);
  cm[0][10] = 1;
  cm[1][20] = 2;
  return cm;
}

TEST(DDTreeFollow, FullAccept) {
  auto cm = kChain();
  std::vector<int32_t> posterior = {10, 20, 77}; // root->1->2, then bonus 77
  Accepted a = followVerified(cm, posterior.data());
  EXPECT_EQ(a.indices, (std::vector<int32_t>{0, 1, 2}));
  EXPECT_EQ(a.nextToken, 77);
}

TEST(DDTreeFollow, PartialAccept) {
  auto cm = kChain();
  std::vector<int32_t> posterior = {10, 999, 77}; // root->1, then 999 not a child
  Accepted a = followVerified(cm, posterior.data());
  EXPECT_EQ(a.indices, (std::vector<int32_t>{0, 1}));
  EXPECT_EQ(a.nextToken, 999);
}

TEST(DDTreeFollow, ImmediateReject) {
  auto cm = kChain();
  std::vector<int32_t> posterior = {5, 20, 77}; // root token 5 not a child of root
  Accepted a = followVerified(cm, posterior.data());
  EXPECT_EQ(a.indices, (std::vector<int32_t>{0}));
  EXPECT_EQ(a.nextToken, 5);
}
```

- [ ] **Step 2: Run to confirm failure** (undefined `followVerified`).

```bash
ninja -C build unittest_ddtree
```
Expected: undefined reference.

- [ ] **Step 3: Implement `followVerified`.** Append to `ddtree.cpp`:

```cpp
Accepted followVerified(
  const std::vector<std::unordered_map<int32_t, int32_t>> &childMaps,
  const int32_t *posterior) {
  // ddtree.py 282-293.
  Accepted a;
  a.indices.push_back(0);
  int currentIndex = 0;
  int32_t nextToken = posterior[currentIndex];
  while (true) {
    auto it = childMaps[currentIndex].find(nextToken);
    if (it == childMaps[currentIndex].end())
      break;
    currentIndex = it->second;
    a.indices.push_back(currentIndex);
    nextToken = posterior[currentIndex];
  }
  a.nextToken = nextToken;
  return a;
}
```

- [ ] **Step 4: Compile and run.**

```bash
ninja -C build unittest_ddtree && meson test -C build unittest_ddtree -v
```
Expected: all `DDTreeFollow.*` PASS.

- [ ] **Step 5: Commit.**

```bash
git add nntrainer/nntrainer/ddtree/ddtree.cpp test/unittest/unittest_ddtree.cpp
git commit -m "feat(ddtree): followVerified accepted-path walk"
```

---

### Task 7: `compactTail` — raw-pointer KV tail reorder helper

**Files:**
- Create: `nntrainer/nntrainer/ddtree/ddtree_compact.h`
- Create: `nntrainer/nntrainer/ddtree/ddtree_compact.cpp`
- Test: `test/unittest/unittest_ddtree.cpp`

- [ ] **Step 1: Write the failing test.** Append:

```cpp
#include <ddtree_compact.h>
using nntrainer::ddtree::compactTail;

TEST(DDTreeCompact, SubsetReorderRowMajor) {
  // 1 row of "prefix" then a tail of 4 rows, each row = 2 elems (float).
  // Layout: rows 0..4, rowElems=2, seqStride=2 (contiguous).
  const int rowElems = 2, seqStride = 2, past = 1, tailLen = 4;
  std::vector<float> buf = {
    -1, -1,        // row 0 (prefix, untouched)
    10, 11,        // tail idx 0
    20, 21,        // tail idx 1
    30, 31,        // tail idx 2
    40, 41,        // tail idx 3
  };
  std::vector<int32_t> keep = {0, 2, 3}; // keep tail rows 0,2,3
  compactTail(buf.data(), sizeof(float), seqStride, rowElems, past, tailLen,
              keep.data(), (int)keep.size());
  // After: tail rows [past..past+3) = old rows 0,2,3.
  EXPECT_EQ(buf[2], 10); EXPECT_EQ(buf[3], 11); // dst0 = src0
  EXPECT_EQ(buf[4], 30); EXPECT_EQ(buf[5], 31); // dst1 = src2
  EXPECT_EQ(buf[6], 40); EXPECT_EQ(buf[7], 41); // dst2 = src3
  EXPECT_EQ(buf[0], -1); // prefix untouched
}

TEST(DDTreeCompact, IdentityNoOp) {
  std::vector<float> buf = {0, 1, 2, 3};
  std::vector<int32_t> keep = {0, 1};
  compactTail(buf.data(), sizeof(float), 2, 2, 0, 2, keep.data(), 2);
  EXPECT_EQ(buf, (std::vector<float>{0, 1, 2, 3}));
}

TEST(DDTreeCompact, EmptyKeepNoOp) {
  std::vector<float> buf = {9, 9};
  std::vector<int32_t> keep;
  compactTail(buf.data(), sizeof(float), 2, 2, 0, 1, keep.data(), 0);
  EXPECT_EQ(buf, (std::vector<float>{9, 9}));
}

TEST(DDTreeCompact, WorksWithSeqStrideGreaterThanRow) {
  // seqStride 3 but rowElems 2 (padded rows). 3 tail rows.
  const int rowElems = 2, seqStride = 3, past = 0, tailLen = 3;
  std::vector<float> buf = {
    10, 11, 99,   // tail 0 (col 2 = pad)
    20, 21, 99,   // tail 1
    30, 31, 99,   // tail 2
  };
  std::vector<int32_t> keep = {2, 0};
  compactTail(buf.data(), sizeof(float), seqStride, rowElems, past, tailLen,
              keep.data(), 2);
  EXPECT_EQ(buf[0], 30); EXPECT_EQ(buf[1], 31); // dst0 = src2
  EXPECT_EQ(buf[3], 10); EXPECT_EQ(buf[4], 11); // dst1 = src0
}
```

- [ ] **Step 2: Create `ddtree_compact.h`.**

```cpp
// SPDX-License-Identifier: Apache-2.0
/**
 * @file ddtree_compact.h
 * @brief Reusable, model-agnostic KV tail reorder (== _compact_appended_window).
 */
#ifndef __NNTRAINER_DDTREE_COMPACT_H__
#define __NNTRAINER_DDTREE_COMPACT_H__

#include <cstdint>

namespace nntrainer {
namespace ddtree {

/**
 * @brief Reorder cache rows [pastLen, pastLen+tailLen) by keepIndices into
 *        [pastLen, pastLen+keepCount). Gathers into a temporary first, so any
 *        permutation is safe. No-op when keepCount == 0.
 * @param cacheBase     base pointer of the cache buffer (row 0)
 * @param elemSizeBytes element size (2 for fp16, 4 for fp32)
 * @param seqStrideElems elements between consecutive rows (sequence positions)
 * @param rowElems       valid elements per row to copy
 * @param pastLen        index of the first tail row
 * @param tailLen        number of tail rows (must equal the verified window)
 * @param keepIndices    [keepCount] tail-relative indices to keep, in order
 * @param keepCount      number of kept rows (<= tailLen)
 */
void compactTail(void *cacheBase, int elemSizeBytes, int seqStrideElems,
                 int rowElems, int pastLen, int tailLen,
                 const int32_t *keepIndices, int keepCount);

} // namespace ddtree
} // namespace nntrainer

#endif // __NNTRAINER_DDTREE_COMPACT_H__
```

- [ ] **Step 3: Create `ddtree_compact.cpp`.**

```cpp
// SPDX-License-Identifier: Apache-2.0
/**
 * @file ddtree_compact.cpp
 * @brief KV tail reorder helper. Mirrors _compact_appended_window semantics.
 */
#include <ddtree_compact.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace nntrainer {
namespace ddtree {

void compactTail(void *cacheBase, int elemSizeBytes, int seqStrideElems,
                 int rowElems, int pastLen, int tailLen,
                 const int32_t *keepIndices, int keepCount) {
  if (keepCount <= 0 || tailLen <= 0)
    return;

  auto *base = static_cast<uint8_t *>(cacheBase);
  const size_t rowBytes = static_cast<size_t>(rowElems) * elemSizeBytes;
  const size_t strideBytes = static_cast<size_t>(seqStrideElems) * elemSizeBytes;

  // Gather kept rows into a packed temporary (index_select semantics).
  std::vector<uint8_t> tmp(static_cast<size_t>(keepCount) * rowBytes);
  for (int d = 0; d < keepCount; ++d) {
    const int src = keepIndices[d];
    const uint8_t *srcRow = base + (static_cast<size_t>(pastLen) + src) * strideBytes;
    std::memcpy(tmp.data() + static_cast<size_t>(d) * rowBytes, srcRow, rowBytes);
  }
  // Write back into [pastLen, pastLen+keepCount).
  for (int d = 0; d < keepCount; ++d) {
    uint8_t *dstRow = base + (static_cast<size_t>(pastLen) + d) * strideBytes;
    std::memcpy(dstRow, tmp.data() + static_cast<size_t>(d) * rowBytes, rowBytes);
  }
}

} // namespace ddtree
} // namespace nntrainer
```

- [ ] **Step 4: Compile and run.**

```bash
ninja -C build unittest_ddtree && meson test -C build unittest_ddtree -v
```
Expected: all `DDTreeCompact.*` PASS.

- [ ] **Step 5: Commit.**

```bash
git add nntrainer/nntrainer/ddtree/ddtree_compact.h nntrainer/nntrainer/ddtree/ddtree_compact.cpp test/unittest/unittest_ddtree.cpp
git commit -m "feat(ddtree): raw-pointer compactTail KV tail reorder helper"
```

---

### Task 8: `argmax` / greedy sampling helper

**Files:**
- Create: `nntrainer/nntrainer/ddtree/ddtree_sampling.h`
- Create: `nntrainer/nntrainer/ddtree/ddtree_sampling.cpp`
- Test: `test/unittest/unittest_ddtree.cpp`

- [ ] **Step 1: Write the failing test.** Append:

```cpp
#include <ddtree_sampling.h>
using nntrainer::ddtree::argmaxRow;
using nntrainer::ddtree::sampleGreedy;

TEST(DDTreeSampling, ArgmaxAndGreedyRows) {
  std::vector<float> logits = {
    0.1f, 0.9f, 0.2f,   // row 0 -> token 1
    5.0f, 1.0f, 2.0f,   // row 1 -> token 0
  };
  EXPECT_EQ(argmaxRow(logits.data(), 3), 1);
  std::vector<int32_t> out(2);
  sampleGreedy(logits.data(), 2, 3, out.data());
  EXPECT_EQ(out[0], 1);
  EXPECT_EQ(out[1], 0);
}

TEST(DDTreeSampling, ArgmaxTieLowestIndex) {
  std::vector<float> logits = {2.0f, 2.0f, 1.0f}; // tie -> lowest index 0
  EXPECT_EQ(argmaxRow(logits.data(), 3), 0);
}
```

- [ ] **Step 2: Create `ddtree_sampling.h`.**

```cpp
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
```

- [ ] **Step 3: Create `ddtree_sampling.cpp`.**

```cpp
// SPDX-License-Identifier: Apache-2.0
/**
 * @file ddtree_sampling.cpp
 */
#include <ddtree_sampling.h>

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
```

- [ ] **Step 4: Compile and run.**

```bash
ninja -C build unittest_ddtree && meson test -C build unittest_ddtree -v
```
Expected: all `DDTreeSampling.*` PASS.

- [ ] **Step 5: Commit.**

```bash
git add nntrainer/nntrainer/ddtree/ddtree_sampling.h nntrainer/nntrainer/ddtree/ddtree_sampling.cpp test/unittest/unittest_ddtree.cpp
git commit -m "feat(ddtree): greedy sampling convenience helper"
```

---

# Phase B — Runtime integration

### Task 9: `KVCacheManager::compactTail`

> **Read first:** `Applications/CausalLM/kv_cache_manager.h` and `.cpp` in full. Confirm the real names/signatures of: the per-layer cache accessors (exploration reported `getKeyCache(layer)` / `getValueCache(layer)` and members `layer_caches_`, `kv_widths_`, `cache_pos_`, `setPosition`), the dtype member (`dtype_`), and how to obtain a raw data pointer for a `nntrainer::Tensor` (`getData<float>()` / `getData<_FP16>()`). Adjust the code below to the actual API; the structure (loop layers → call core `compactTail` on each key/value tensor → `setPosition`) is fixed.

**Files:**
- Modify: `Applications/CausalLM/kv_cache_manager.h`
- Modify: `Applications/CausalLM/kv_cache_manager.cpp`
- Test: `Applications/CausalLM/test/unittest/...` — add to the existing `unittest_kv_cache_manager` target (registered in `Applications/CausalLM/meson.build`).

- [ ] **Step 1: Write the failing test.** Locate the existing `unittest_kv_cache_manager.cpp` (exploration: `test/unittest/layers/unittest_kv_cache_manager.cpp` under CausalLM test wiring). Append a test that: allocates a small cache, writes a known tail, calls `compactTail`, and reads back the compacted rows. Use the file's existing constants (`NUM_LAYERS`, `BATCH_SIZE`, `MAX_SEQ_LEN`, `NUM_HEADS_KV`, `HEAD_DIM`, `KV_WIDTH`).

```cpp
TEST(KVCacheManager, CompactTailReordersAppendedWindow) {
  causallm::KVCacheManager kv;
  std::vector<unsigned int> widths(NUM_LAYERS, KV_WIDTH);
  kv.allocate(NUM_LAYERS, BATCH_SIZE, MAX_SEQ_LEN, widths,
              ml::train::TensorDim::DataType::FP32);

  const unsigned int past = 3;
  const unsigned int tail = 4;
  // Write recognisable values into tail rows [past, past+tail) of layer 0, batch 0.
  nntrainer::Tensor &k0 = kv.getKeyCache(0);
  float *base = k0.getData<float>();
  const unsigned int width = KV_WIDTH;
  for (unsigned int r = 0; r < tail; ++r)
    for (unsigned int c = 0; c < width; ++c)
      base[(past + r) * width + c] = static_cast<float>((r + 1) * 100 + c);

  kv.setPosition(past + tail);
  std::vector<int32_t> keep = {0, 2, 3}; // drop tail row 1
  kv.compactTail(past, keep);

  EXPECT_EQ(kv.getPosition(), past + keep.size());
  // Row at past+1 should now hold old tail row 2.
  EXPECT_EQ(base[(past + 1) * width + 0], 300.0f);
  EXPECT_EQ(base[(past + 2) * width + 0], 400.0f);
}
```

- [ ] **Step 2: Declare the method.** In `kv_cache_manager.h`, in the public section near `advance` / `setPosition`:

```cpp
  /**
   * @brief Reorder the appended KV tail [pastLen, pastLen+tailLen) to the
   *        accepted DDTree path, then set position = pastLen + keepIndices.size().
   *        tailLen is inferred as getPosition() - pastLen and must equal the
   *        verified tree window. Operates on every layer's key & value cache.
   * @param pastLen     start of the appended window
   * @param keepIndices tail-relative accepted indices (includes root index 0)
   */
  void compactTail(unsigned int pastLen, const std::vector<int32_t> &keepIndices);
```

Ensure `<vector>` and `<cstdint>` are included.

- [ ] **Step 3: Implement the method.** In `kv_cache_manager.cpp`, add `#include <ddtree_compact.h>` and:

```cpp
void KVCacheManager::compactTail(unsigned int pastLen,
                                 const std::vector<int32_t> &keepIndices) {
  const unsigned int tailLen = cache_pos_ - pastLen;
  const int keepCount = static_cast<int>(keepIndices.size());

  const int elemSize =
    (dtype_ == ml::train::TensorDim::DataType::FP16) ? 2 : 4;

  for (auto &lc : layer_caches_) {
    for (nntrainer::Tensor *cache : {&lc.key_cache, &lc.value_cache}) {
      ml::train::TensorDim dim = cache->getDim();
      const int width = static_cast<int>(dim.width());        // kv_width
      const int batchStride = static_cast<int>(dim.getFeatureLen()); // per-batch elems
      void *base = (dtype_ == ml::train::TensorDim::DataType::FP16)
                     ? static_cast<void *>(cache->getData<_FP16>())
                     : static_cast<void *>(cache->getData<float>());
      for (unsigned int b = 0; b < dim.batch(); ++b) {
        void *batchBase = static_cast<uint8_t *>(base) +
                          static_cast<size_t>(b) * batchStride * elemSize;
        nntrainer::ddtree::compactTail(batchBase, elemSize, /*seqStride*/ width,
                                       /*rowElems*/ width, (int)pastLen,
                                       (int)tailLen, keepIndices.data(),
                                       keepCount);
      }
    }
  }
  setPosition(pastLen + static_cast<unsigned int>(keepCount));
}
```

> If `_FP16` is unavailable without the fp16 build flag, guard the fp16 branch behind `#ifdef ENABLE_FP16` and the test uses FP32. Confirm `dim.width()` / `dim.getFeatureLen()` are the right accessors against the real `TensorDim` API while reading the file.

- [ ] **Step 4: Register/confirm the cache manager test builds, then compile and run.**

```bash
ninja -C build unittest_kv_cache_manager && meson test -C build unittest_kv_cache_manager -v
```
Expected: `KVCacheManager.CompactTailReordersAppendedWindow` PASS.

- [ ] **Step 5: Commit.**

```bash
git add Applications/CausalLM/kv_cache_manager.h Applications/CausalLM/kv_cache_manager.cpp \
        Applications/CausalLM/test/unittest/layers/unittest_kv_cache_manager.cpp
git commit -m "feat(causallm): KVCacheManager::compactTail for DDTree path compaction"
```

---

### Task 10: `mha_core` additive-mask attention path

> **Read first:** `Applications/CausalLM/layers/mha_core.h` and `.cpp` in full. Key anchors from exploration: input enum `INOUT_INDEX { QUERY=0, KEY=1, VALUE=2, MASK=3 }` (`mha_core.h:379-389`); commented `// attention_mask,` (`mha_core.h:398`); input-count check (`mha_core.cpp:137-141`); `incremental_forwarding` (`mha_core.cpp:406-564`); `one_batch_incremental_forwarding` builds the score tensor `out_` (`mha_core.cpp:767-771`) and calls `softmax_triangle(out_, ...)` (line 778). The `out_` score tensor layout is `(1,1, attn_index_height, num_heads_Q)` — for non-causal it is `step_size * cache_to` rows. The mask must be added **into `out_` before softmax** and a **full (non-triangular) softmax** used.

**Design:** Gate entirely on mask presence. When `context.getNumInputs()` indicates a MASK input is bound and non-empty, take a new branch: add the additive mask to `out_`, then call a full-softmax over each query row's full key range. Otherwise keep the existing `softmax_triangle` path untouched (no regression — spec R4). The mask tensor for a tree verify pass has shape `(1,1,currentLength, pastLength+currentLength)` matching `out_`'s logical `(query, key)`.

**Files:**
- Modify: `Applications/CausalLM/layers/mha_core.h`
- Modify: `Applications/CausalLM/layers/mha_core.cpp`
- Test: add `MHACoreMaskedAttention` to a CausalLM unittest target (or a new `unittest_mha_core_mask.cpp` registered in `Applications/CausalLM/meson.build`).

- [ ] **Step 1: Write the failing numeric test.** Create `Applications/CausalLM/test/unittest/layers/unittest_mha_core_mask.cpp` (and register it as an executable in `Applications/CausalLM/meson.build` mirroring `unittest_kv_cache_manager`). The test builds a tiny single-head attention, supplies Q/K/V and a known additive tree mask, and compares the output to a hand-computed numpy-style reference computed inline:

```cpp
// Reference: out = softmax(QK^T / sqrt(d) + mask) @ V, full (non-triangular).
// Single head, d=2, 1 query, 2 keys; mask hides key 1 (-inf).
TEST(MHACoreMask, MaskedSoftmaxMatchesReference) {
  // Construct an MHACoreLayer configured for 1 head, head_dim=2, non-causal,
  // with a MASK input bound. (Use the layer's finalize/forward harness used by
  // the existing mha_core tests; if none exists, drive softmax helper directly.)
  // ... see Step 3 for the helper under test.
  FAIL() << "to be replaced by concrete harness in Step 3";
}
```

> Because driving the full layer in a unit test is heavy, prefer testing the **new masked-softmax routine** directly (Step 3 extracts it as a small testable function). Replace the `FAIL()` body once the helper signature exists.

- [ ] **Step 2: Add mask extraction + a `use_mask` flag.** In `mha_core.cpp` `incremental_forwarding`, after the Q/K/V extraction (around lines 453-459), add:

```cpp
  // Optional additive attention mask (DDTree tree-DAG visibility). Present when
  // a non-empty MASK input is bound. Gates the additive-mask softmax path.
  nntrainer::Tensor *attn_mask = nullptr;
  if (context.getNumInputs() > INOUT_INDEX::MASK) {
    nntrainer::Tensor &m = context.getInput(INOUT_INDEX::MASK);
    if (!m.empty() && m.size() != 0)
      attn_mask = &m;
  }
```

Thread `attn_mask` into `one_batch_incremental_forwarding` (add a trailing `nntrainer::Tensor *attn_mask` parameter, default `nullptr`, in both the declaration in `mha_core.h` and the definition).

- [ ] **Step 3: Add the masked full-softmax branch.** In `one_batch_incremental_forwarding`, where it currently calls `softmax_triangle(out_, step_size, num_heads_Q, cache_from);` (line ~778), branch:

```cpp
  if (attn_mask != nullptr) {
    add_mask_and_softmax_full(out_, *attn_mask, batch, step_size, cache_to,
                              num_heads_Q);
  } else {
    softmax_triangle(out_, step_size, num_heads_Q, cache_from);
  }
```

Implement `add_mask_and_softmax_full` as a new private method. It must: for each query row `i` in `[0, step_size)`, over key columns `j` in `[0, cache_to)`, add `mask[i, j]` to the score and apply a numerically-stable softmax across the `cache_to` keys, per head. The score tensor `out_` for the non-causal case is laid out with `step_size * cache_to` rows of `num_heads_Q` width, where logical `(query i, key j)` maps to flat row `i * cache_to + j` (matches the non-causal `start_row = i * to` indexing at `mha_core.cpp:1234-1236`). Declaration in `mha_core.h` (near `softmax_triangle`):

```cpp
  WIN_EXPORT void add_mask_and_softmax_full(nntrainer::Tensor &qk_out,
                                            nntrainer::Tensor &mask,
                                            unsigned int batch,
                                            size_t step_size, unsigned int to,
                                            size_t num_head);
```

Definition in `mha_core.cpp` (fp32 path; mirror an fp16 path guarded by `ENABLE_FP16` if the score tensor can be fp16, following the existing `softmax_triangle` fp16 structure):

```cpp
void MHACoreLayer::add_mask_and_softmax_full(nntrainer::Tensor &qk_out,
                                             nntrainer::Tensor &mask,
                                             unsigned int batch,
                                             size_t step_size, unsigned int to,
                                             size_t num_head) {
  float *qk = qk_out.getData<float>();
  const float *mrow = mask.getData<float>(); // [step_size, kv_len], kv_len >= to
  const unsigned int kv_len = mask.width();

  auto &tm = nntrainer::ThreadManager::Global();
  tm.parallel_for(0, step_size, [=](size_t i) {
    for (size_t h = 0; h < num_head; ++h) {
      // add mask then stable softmax over keys [0, to) for (query i, head h).
      float maxv = -std::numeric_limits<float>::infinity();
      for (unsigned int j = 0; j < to; ++j) {
        float v = qk[(i * to + j) * num_head + h] +
                  mrow[i * kv_len + j];
        qk[(i * to + j) * num_head + h] = v;
        maxv = std::max(maxv, v);
      }
      float sum = 0.0f;
      for (unsigned int j = 0; j < to; ++j) {
        float e = std::exp(qk[(i * to + j) * num_head + h] - maxv);
        qk[(i * to + j) * num_head + h] = e;
        sum += e;
      }
      const float inv = 1.0f / sum;
      for (unsigned int j = 0; j < to; ++j)
        qk[(i * to + j) * num_head + h] *= inv;
    }
  });
  (void)batch;
}
```

> Verify the exact `out_` indexing (`(i * to + j) * num_head + h`) against the real `compute_kcaches` writer and the non-causal branch of `softmax_triangle` while reading the file; adjust if the head/key axes differ. The masked path must reproduce the same normalization the reference test asserts.

- [ ] **Step 4: Point the Step 1 test at `add_mask_and_softmax_full`.** Replace the `FAIL()` test with one that allocates a small `qk_out` tensor (shape `(1,1, step_size*to, num_head)`), a mask tensor, fills known QK scores, calls the method, and asserts the softmax-with-mask result equals an inline reference (mask `-1e30` on a key ⇒ that key's weight ≈ 0; remaining keys form a proper softmax summing to 1).

- [ ] **Step 5: Compile and run.**

```bash
ninja -C build unittest_mha_core_mask && meson test -C build unittest_mha_core_mask -v
```
Expected: `MHACoreMask.MaskedSoftmaxMatchesReference` PASS. Also run the existing mha_core / causallm model tests to confirm **no regression** on the causal path:

```bash
meson test -C build --suite unittests
```
Expected: previously-passing CausalLM attention tests still pass.

- [ ] **Step 6: Commit.**

```bash
git add Applications/CausalLM/layers/mha_core.h Applications/CausalLM/layers/mha_core.cpp \
        Applications/CausalLM/test/unittest/layers/unittest_mha_core_mask.cpp Applications/CausalLM/meson.build
git commit -m "feat(causallm): mha_core additive-mask (non-causal) attention path"
```

---

# Phase C — Golden vectors & end-to-end parity

### Task 11: Python golden-vector dumper

**Files:**
- Create: `test/unittest/ddtree_golden/gen_golden.py`
- Create (generated): `test/unittest/ddtree_golden/*.json`

- [ ] **Step 1: Write a self-contained dumper.** `gen_golden.py` must NOT import the heavy `ddtree` module (it pulls `model`, `dflash`). Instead, copy the four pure functions (`build_ddtree_tree`, `compile_ddtree_tree`, `prepare_ddtree_attention_mask_for_target`-equivalent, `follow_verified_tree`) **verbatim** from `/home/shsh1004/littlesd_inference/ddtree.py`, with trivial stubs:

```python
#!/usr/bin/env python3
"""Dump DDTree golden vectors from the canonical Python algorithm.
Copies the pure functions verbatim from ddtree.py with timing stubs."""
import json, heapq
import numpy as np
import torch

# --- stubs replacing dflash timing helpers (no behavioral effect) ---
def cuda_time(): return 0.0
def empty_stage_times(order): return {k: 0.0 for k in order}
DDTREE_TREE_BUILD_STAGE_ORDER = ("tree_build_copy", "tree_build_heap", "tree_build_visibility")

# --- PASTE build_ddtree_tree / compile_ddtree_tree / follow_verified_tree
#     verbatim from ddtree.py (lines 92-216, 282-293) below this line ---
# (do not edit the algorithm bodies)

def dump_build(name, logits, budget):
    t = torch.tensor(logits, dtype=torch.float32)
    node_token_ids, node_depths, parents, child_maps, visibility, _ = build_ddtree_tree(t, budget)
    return {
        "name": name,
        "logits": logits,
        "budget": budget,
        "node_token_ids": [int(x) for x in node_token_ids.tolist()],
        "node_depths": [int(x) for x in node_depths.tolist()],
        "parents": [int(x) for x in parents],
        "visibility": visibility.to(torch.int32).tolist(),
    }

if __name__ == "__main__":
    cases = []
    cases.append(dump_build("small", [[2.0,1.0,0.0],[0.0,-1.0,1.0]], 3))
    cases.append(dump_build("tie", [[1.0,1.0,0.0],[0.5,0.5,0.5]], 4))  # equal logw tie-break
    cases.append(dump_build("budget_gt_vocab", [[2.0,1.0,0.0],[0.0,-1.0,1.0]], 100))
    cases.append(dump_build("full_fill", [[3.0,2.0,1.0,0.0],[3.0,2.0,1.0,0.0],[3.0,2.0,1.0,0.0]], 8))
    import os
    out = os.path.join(os.path.dirname(__file__), "build_golden.json")
    with open(out, "w") as f:
        json.dump(cases, f, indent=2)
    print("wrote", out)
```

- [ ] **Step 2: Run the dumper.**

```bash
cd /home/shsh1004/nntrainer/test/unittest/ddtree_golden
python3 gen_golden.py
```
Expected: `wrote .../build_golden.json`. Inspect the file: each case has `node_token_ids`, `node_depths`, `parents`, `visibility`.

- [ ] **Step 3: Commit the dumper and golden JSON.**

```bash
cd /home/shsh1004/nntrainer
git add test/unittest/ddtree_golden/gen_golden.py test/unittest/ddtree_golden/build_golden.json
git commit -m "test(ddtree): Python golden-vector dumper + buildTree golden vectors"
```

---

### Task 12: `DDTreeBuild` golden parity test (exact token/depth/visibility)

**Files:**
- Modify: `test/unittest/unittest_ddtree.cpp`
- Modify: `test/unittest/meson.build` (pass golden dir as a compile define, or load by relative path)

- [ ] **Step 1: Make the golden directory discoverable.** In `test/unittest/meson.build`, for the `unittest_ddtree` target add a compile arg with the golden path. Simplest: define a macro. Add (near the `executable(...)` for tests, or specifically for ddtree) `cpp_args: ['-DDDTREE_GOLDEN_DIR="' + meson.current_source_dir() + '/ddtree_golden"']`. If the shared `foreach target` loop can't take per-target args, split `unittest_ddtree` into its own `executable()` block above the loop with this `cpp_args` and its own `test()` registration.

- [ ] **Step 2: Write the golden parity test.** Append to `unittest_ddtree.cpp` (use a minimal hand JSON parser or pull in the repo's existing JSON lib if one is already linked in tests; if unsure, parse the small fixed file with a tiny tokenizer). Concretely, use `nlohmann::json` if available in the tree (CausalLM uses `json`); otherwise read values via a small helper. Test logic:

```cpp
#include <fstream>
#include <string>
// Loads build_golden.json and re-runs buildTree per case, asserting equality.
TEST(DDTreeBuild, MatchesPythonGolden) {
  const std::string path = std::string(DDTREE_GOLDEN_DIR) + "/build_golden.json";
  std::ifstream in(path);
  ASSERT_TRUE(in.good()) << "missing " << path;
  // Parse cases: for each, read logits [depth][vocab], budget, and expected
  // node_token_ids / node_depths / parents / visibility.
  // (Use the JSON library already linked into the unittest target.)
  for (const auto &c : parseGoldenCases(in)) {
    DDTreeConfig cfg;
    cfg.budget = c.budget;
    DDTreeStructure t = buildTree(c.logits.data(), c.depth, c.vocab, cfg);
    EXPECT_EQ(t.nodeTokenIds, c.expNodeTokenIds) << c.name;
    EXPECT_EQ(t.nodeDepths, c.expNodeDepths) << c.name;
    EXPECT_EQ(t.parents, c.expParents) << c.name;
    // visibility: compare flattened bitmap to expected LxL int matrix.
    ASSERT_EQ((int)t.visibility.size(), c.expVis.size()) << c.name;
    for (size_t k = 0; k < t.visibility.size(); ++k)
      EXPECT_EQ((int)t.visibility[k], c.expVis[k]) << c.name << " vis@" << k;
  }
}
```

Implement `parseGoldenCases` + a `GoldenCase` struct in the test file using whichever JSON facility the unittest target already links (confirm by checking `test/unittest/meson.build` deps). If no JSON lib is linked into core unittests, add the header-only `nlohmann/json` include already vendored under `Applications/CausalLM` or parse manually.

- [ ] **Step 3: Compile and run.**

```bash
ninja -C build unittest_ddtree && meson test -C build unittest_ddtree -v
```
Expected: `DDTreeBuild.MatchesPythonGolden` PASS for all 4 cases (small, tie, budget_gt_vocab, full_fill). **If the `tie` case fails, the heap comparator (Task 3) or topk tie-break is wrong — fix before proceeding (spec R3).**

- [ ] **Step 4: Commit.**

```bash
git add test/unittest/unittest_ddtree.cpp test/unittest/meson.build
git commit -m "test(ddtree): buildTree exact parity vs Python golden vectors"
```

---

### Task 13: Trace-replay parity test (gemma4 algorithm parity, model-free)

This satisfies acceptance §11.3 / spec §8.1: replay captured gemma4 `draft_logits` + `posterior` through the core and assert the identical final token sequence.

**Files:**
- Modify: `test/unittest/ddtree_golden/gen_golden.py` (add a trace-capture path) OR create `test/unittest/ddtree_golden/gen_trace.py`
- Create (generated): `test/unittest/ddtree_golden/gemma4_trace.json`
- Modify: `test/unittest/unittest_ddtree.cpp` (add `DDTreeParityReplay`)

- [ ] **Step 1: Capture a real gemma4 trace from Python.** Using `littlesd_inference`, run `ddtree_generate` with `save_tree_traces=True` plus added instrumentation to also dump, per round: `draft_logits` (the `[draft_horizon, vocab]` slice fed to `build_ddtree_tree`), the sampled `posterior` (`[current_length]` ints), `start`, `past_length`, plus the final `output_ids`. Config: gemma4 + littlebit draft, `tree_budget=31`, `max_new_tokens=16`, `temperature=0`. Write `gemma4_trace.json` with: `prompt_len`, `final_output_ids` (the full token sequence), and `rounds: [{start, past_length, draft_logits, posterior}]`.

> This requires the Python environment + checkpoints. If the trace cannot be produced in this session, document the exact command in the README CPU guide and mark this test `DISABLED_` until the trace file exists; the live harness (Task 14) then becomes the primary parity evidence. Prefer producing the trace.

- [ ] **Step 2: Write the replay test.** Append to `unittest_ddtree.cpp`:

```cpp
TEST(DDTreeParityReplay, ReproducesGemma4TokenSequence) {
  const std::string path = std::string(DDTREE_GOLDEN_DIR) + "/gemma4_trace.json";
  std::ifstream in(path);
  if (!in.good()) { GTEST_SKIP() << "no gemma4_trace.json"; }
  Trace tr = parseTrace(in);

  DDTreeConfig cfg;
  cfg.budget = 31;
  cfg.maskFillValue = -3.4028235e38f; // finfo(float32).min

  std::vector<int32_t> produced(tr.promptTokens.begin(), tr.promptTokens.end());
  // The first generated token is the prefill bonus (captured as round[0].rootToken
  // / posterior chain start); replay rounds exactly as ddtree_generate does.
  for (const auto &round : tr.rounds) {
    DDTreeStructure t = buildTree(round.draftLogits.data(), tr.draftHorizon,
                                  tr.vocab, cfg);
    Accepted a = followVerified(t.childMaps, round.posterior.data());
    // accepted tokens = verifyInputIds[accepted_indices]; next = a.nextToken.
    // Reconstruct verifyInputIds via compile to mirror Python exactly.
    std::vector<int32_t> ids(t.currentLength), pos(t.currentLength);
    std::vector<float> mask((size_t)t.currentLength *
                            (round.pastLength + t.currentLength));
    compile(round.rootToken, round.start, round.pastLength, t, cfg, ids.data(),
            pos.data(), mask.data(), round.pastLength + t.currentLength);
    for (int idx : a.indices)
      produced.push_back(ids[idx]);
    produced.push_back(a.nextToken);
    // (truncate/advance bookkeeping mirrors ddtree_generate: start += accepted)
  }
  // Compare to Python's final_output_ids (trim to same length / stop handling).
  ASSERT_GE(produced.size(), tr.finalOutputIds.size());
  for (size_t i = 0; i < tr.finalOutputIds.size(); ++i)
    EXPECT_EQ(produced[i], tr.finalOutputIds[i]) << "token " << i;
}
```

> Match the exact `output_ids` write/advance logic of `ddtree_generate` (lines 603-607): `output_ids[start : start+len(accepted)] = accepted_tokens; output_ids[start+len(accepted)] = next_token; start += len(accepted)`. Reconstruct `produced` with that same indexing rather than naive append if rounds overlap.

- [ ] **Step 3: Run.**

```bash
ninja -C build unittest_ddtree && meson test -C build unittest_ddtree -v
```
Expected: `DDTreeParityReplay.ReproducesGemma4TokenSequence` PASS (identical 16-token sequence), or SKIP if no trace.

- [ ] **Step 4: Commit.**

```bash
git add test/unittest/ddtree_golden/gen_trace.py test/unittest/ddtree_golden/gemma4_trace.json test/unittest/unittest_ddtree.cpp
git commit -m "test(ddtree): gemma4 trace-replay token-parity test"
```

---

### Task 14: CPU live-forward gemma4 harness

> **Read first:** `Applications/CausalLM/main.cpp` (model registration/load/run flow), `Applications/CausalLM/models/causal_lm.{h,cpp}` (`incremental_inference`, `setKVCachePosition`, `advanceKVCachePosition`, how logits/hidden are read, how `attention_mask`/`position_ids` are threaded into layers), and `gemma4/gemma4_causallm.{h,cpp}` (`layer_types`, `sliding_window`, `getKVCacheWidth`). Confirm the real `incremental_inference` signature and how to pass a per-pass attention mask + position ids — exploration reported these but did not pin exact argument lists.

**Files:**
- Create: `Applications/CausalLM/tools/ddtree_cpu_harness.cpp`
- Modify: `Applications/CausalLM/meson.build` (add `nntr_ddtree_harness` executable)

- [ ] **Step 1: Add the executable to meson.** In `Applications/CausalLM/meson.build`, mirroring the `nntr_causallm` executable block:

```meson
e_ddtree = executable('nntr_ddtree_harness',
  'tools/ddtree_cpu_harness.cpp',
  dependencies: [nntrainer_dep, causallm_layer_dependencies, causallm_dep],
  include_directories: causallm_inc,
  install: false,
)
```

- [ ] **Step 2: Implement the harness.** `ddtree_cpu_harness.cpp` loads a gemma4 target + the littlebit draft, then runs the DDTree loop using the core. Structure (fill model-forward calls from the real `causal_lm.cpp` API while implementing):

```cpp
// SPDX-License-Identifier: Apache-2.0
// DDTree CPU live-forward harness: gemma4 target + littlebit draft, temp 0.
#include <ddtree.h>
#include <ddtree_sliding.h>
#include <ddtree_sampling.h>
// + CausalLM / Gemma4 includes, factory, json loaders (see main.cpp).

int main(int argc, char **argv) {
  // 1. Parse args: --target <dir> --draft <dir> --prompt <str>
  //    --tree-budget 31 --max-new-tokens 16 (temperature fixed 0).
  // 2. Register + load both models via causallm::Factory (main.cpp pattern):
  //    model->initialize(); model->load_weight(...).
  // 3. Read gemma4 config: depthLimit = block_size-1, vocab, sliding_window,
  //    hasSlidingLayers = layer_types contains "sliding_attention".
  // 4. Prefill target on the prompt; sample first token (argmax).
  // 5. Decode loop (mirror ddtree_generate):
  //    a. draft forward over block -> draft_logits [depthLimit, vocab].
  //    b. DDTreeStructure t = buildTree(draft_logits, depthLimit, vocab, cfg).
  //    c. compile(rootToken, start, pastLen=start, t, cfg, ids, pos, mask, stride).
  //    d. makeSlidingMasks(mask, pos, t.currentLength, pastLen+t.currentLength,
  //         sliding_window, hasSlidingLayers, cfg, slidingBuf).
  //    e. target verify forward with ids/pos/mask (full+sliding per layer type)
  //       -> verify logits per node.
  //    f. posterior[i] = argmaxRow(verifyLogits[i], vocab).
  //    g. Accepted a = followVerified(t.childMaps, posterior).
  //    h. kv_cache.compactTail(start, a.indices); write accepted tokens +
  //       next_token into output; start += a.indices.size().
  //    i. stop on EOS or when max_new_tokens reached.
  // 6. Print the generated token ids (space-separated) to stdout.
}
```

> The verify forward must feed the additive mask into `mha_core` (Task 10): bind the mask tensor to each attention layer's MASK input, choosing `m.full` vs `m.sliding` by the layer's type, and pass `verify_position_ids`. Use the same per-layer binding mechanism `causal_lm.cpp` already uses for KV cache tensors. Confirm and reuse it.

- [ ] **Step 3: Build the harness.**

```bash
ninja -C build nntr_ddtree_harness
```
Expected: links cleanly.

- [ ] **Step 4: Run end-to-end on CPU and compare to Python.** With the gemma4 + draft checkpoints available locally:

```bash
./build/Applications/CausalLM/nntr_ddtree_harness \
  --target <gemma4_dir> --draft <draft_dir> \
  --prompt "<same prompt as Python>" --tree-budget 31 --max-new-tokens 16 \
  > /tmp/ddtree_cpu_tokens.txt
```
Compare `/tmp/ddtree_cpu_tokens.txt` to the Python `ddtree_generate` gemma4 output token ids (same prompt / budget 31 / 16 tokens / temp 0). Expected: **identical token sequence**.

> If checkpoints are not present in this session, document the exact run command + expected comparison in the README CPU guide and verify the harness builds + runs as far as model load; full token-parity evidence is then produced when checkpoints are available. The trace-replay test (Task 13) remains the model-free parity guarantee.

- [ ] **Step 5: Commit.**

```bash
git add Applications/CausalLM/tools/ddtree_cpu_harness.cpp Applications/CausalLM/meson.build
git commit -m "feat(causallm): DDTree CPU live-forward gemma4 harness"
```

---

# Phase D — Documentation

### Task 15: Full build + test sweep (verification gate)

- [ ] **Step 1: Clean build.**

```bash
cd /home/shsh1004/nntrainer
meson setup --reconfigure build -Denable-fp16=true -Denable-test=true
ninja -C build
```
Expected: compiles clean (acceptance §11.1). Capture output.

- [ ] **Step 2: Run the full unit-test suite.**

```bash
meson test -C build --suite unittests
```
Expected: green, including `unittest_ddtree` (all `DDTree*`, `DDTreeParityReplay`), `unittest_kv_cache_manager` (`CompactTail*`), `unittest_mha_core_mask`. Capture the summary. (acceptance §11.2)

- [ ] **Step 3: Record evidence.** Save the `meson test` summary output into the README CPU guide section (Task 16) as the verification record. No commit yet (combined with Task 16).

---

### Task 16: README + CPU usage guide

**Files:**
- Create: `nntrainer/nntrainer/ddtree/README.md`

- [ ] **Step 1: Write the README** covering, with real content (no placeholders):
  - **Architecture** — responsibility boundary table (core = stateless build/compile/sliding/follow/compact-index + raw `compactTail` helper; caller owns forwards/sampling/KV storage), mapping each core function to its `ddtree.py` source function.
  - **Module structure** — the file list from this plan with one-line responsibilities.
  - **Parity notes** — the 7 parity details (heap tie-break, fp32/double, topk/empty, compile mask, sliding arithmetic, keep-index, caller-side sampling) and that golden vectors lock them.
  - **Pure-core usage example** — a compilable snippet: `buildTree` → `compile` → (`makeSlidingMasks`) → caller verify forward → `followVerified` → `compactTail`.
  - **CPU harness usage** — exact build (`meson setup build -Denable-fp16=true -Denable-test=true && ninja -C build nntr_ddtree_harness`) and run command (Task 14 Step 4), plus how to regenerate golden vectors (`gen_golden.py`) and the gemma4 trace (`gen_trace.py`).
  - **Verification record** — paste the `meson test` summary from Task 15.
  - **Quick.AI / QNN integration sketch** — illustrative: include `<ddtree.h>` + `<ddtree_compact.h>`, call the same core, apply the mask to the QNN attention graph, and call `compactTail` on QNN's KV (out of scope here; interface only).

- [ ] **Step 2: Commit README + verification evidence.**

```bash
git add nntrainer/nntrainer/ddtree/README.md
git commit -m "docs(ddtree): README architecture/usage + CPU guide + verification record"
```

---

## Self-Review (spec coverage)

- §3 item 1 (core module: build/compile/sliding/follow/keep-index/compactTail/sampling) → Tasks 1–8. ✓
- §3 item 2 (mha_core additive mask) → Task 10. ✓
- §3 item 3 (`KVCacheManager::compactTail`) → Task 9. ✓
- §3 item 4 (unit tests + CPU harness) → Tasks 2–13 (unit), 14 (harness). ✓
- §3 item 5 (README) → Task 16. ✓
- §5 API surface (all signatures) → Tasks 2–8 (with the two documented refinements). ✓
- §6 parity 1–2 (heap tie-break, fp32/double) → Task 3 + golden Task 12 (incl. `tie` case). ✓
- §6 parity 3 (topk/empty) → Tasks 2, 3, 12 (`budget_gt_vocab`). ✓
- §6 parity 4 (compile mask) → Task 4. ✓
- §6 parity 5 (sliding arithmetic) → Task 5. ✓
- §6 parity 6 (keep-index incl. root, tail-length assert) → Tasks 7, 9. ✓
- §6 parity 7 (caller-side sampling) → Task 8 (convenience only) + harness. ✓
- §8.1 trace-replay → Task 13. ✓
- §8.2 live CPU harness → Task 14. ✓
- §9 unit-test suites (Build/Compile/Sliding/Follow/Compact/ParityReplay/MHACoreMask/KVCacheCompactTail) → Tasks 2–13. ✓
- §11 acceptance (build / meson test / Python parity / CPU runtime) → Tasks 12, 13, 14, 15. ✓

**Known external dependencies flagged in-plan:** Tasks 9, 10, 14 require reading the real `kv_cache_manager` / `mha_core` / `causal_lm` APIs (exploration hedged on exact signatures); each task opens with a "Read first" note and pins the structure that must hold. Tasks 13–14 require Python checkpoints to produce live parity evidence; both define a documented fallback (SKIP/document) so the build+unit gate (Tasks 12, 15) remains achievable without them.
