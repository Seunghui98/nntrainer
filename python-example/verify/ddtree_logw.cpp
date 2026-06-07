// Dump per-node cumulative logw (the tree's log-prob "value") computed with the
// EXACT top-k/logsumexp recipe used by ddtree.cpp::buildTree, plus topLogProbs.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <ddtree.h>
#include <ddtree_types.h>

using namespace nntrainer::ddtree;

int main(int argc, char **argv) {
  const int depth = atoi(argv[1]);
  const int vocab = atoi(argv[2]);
  const int budget = atoi(argv[3]);
  const std::string fin = argv[4], fout = argv[5];

  std::vector<float> logits((size_t)depth * vocab);
  std::ifstream f(fin, std::ios::binary);
  f.read(reinterpret_cast<char *>(logits.data()),
         (std::streamsize)(logits.size() * sizeof(float)));
  f.close();

  DDTreeConfig cfg;
  cfg.budget = budget;
  cfg.depthLimit = depth;
  DDTreeStructure tree = buildTree(logits.data(), depth, vocab, cfg);

  const int topk = std::min(budget, vocab);
  // VERBATIM recipe from ddtree.cpp::buildTree (logsumexp in double -> fp32).
  std::vector<std::vector<float>> tlp(depth);
  std::vector<std::vector<int32_t>> tti(depth);
  for (int d = 0; d < depth; ++d) {
    const float *row = logits.data() + (size_t)d * vocab;
    std::vector<int32_t> idx(vocab);
    for (int i = 0; i < vocab; ++i)
      idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                      [row](int32_t a, int32_t b) {
                        if (row[a] != row[b])
                          return row[a] > row[b];
                        return a < b;
                      });
    float maxLogit = row[0];
    for (int i = 1; i < vocab; ++i)
      maxLogit = std::max(maxLogit, row[i]);
    double sumExp = 0.0;
    for (int i = 0; i < vocab; ++i)
      sumExp += std::exp((double)row[i] - maxLogit);
    const float logZ = (float)(maxLogit + std::log(sumExp));
    tlp[d].resize(topk);
    tti[d].resize(topk);
    for (int r = 0; r < topk; ++r) {
      tti[d][r] = idx[r];
      tlp[d][r] = row[idx[r]] - logZ;
    }
  }

  const int cl = tree.currentLength;
  std::vector<double> nodeLogw(cl, 0.0); // root = 0
  for (int idx = 1; idx < cl; ++idx) {
    const int k = idx - 1; // node array index
    const int dep = tree.nodeDepths[k];
    const int tok = tree.nodeTokenIds[k];
    int rank = -1;
    for (int r = 0; r < topk; ++r)
      if (tti[dep - 1][r] == tok) { rank = r; break; }
    const int parent = tree.parents[idx];
    nodeLogw[idx] = nodeLogw[parent] + (double)tlp[dep - 1][rank];
  }

  FILE *o = fopen(fout.c_str(), "w");
  fprintf(o, "{\"current_length\": %d, \"node_logw\": [", cl);
  for (int i = 0; i < cl; ++i)
    fprintf(o, "%s%.9g", i ? "," : "", nodeLogw[i]);
  fprintf(o, "]}\n");
  fclose(o);
  printf("cpp logw dumped: cl=%d\n", cl);
  return 0;
}
