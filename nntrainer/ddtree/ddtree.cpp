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
    std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                      [row](int32_t a, int32_t b) {
                        if (row[a] != row[b])
                          return row[a] > row[b]; // logit desc
                        return a < b;             // tie: lower token index first
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
      const double siblingLogw =
        e.logw - static_cast<double>(topLogProbs[e.depth - 1][e.rank]) +
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
}

} // namespace ddtree
} // namespace nntrainer
