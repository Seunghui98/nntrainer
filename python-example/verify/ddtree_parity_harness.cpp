// Step 4: C++ DDTree tree-build parity harness. Reads the SAME draft logits the
// Python side used, runs nntrainer::ddtree::buildTree/compile/followVerified, and
// dumps cpp_tree.json in the same schema as py_tree.json for 1:1 comparison.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include <ddtree.h>
#include <ddtree_types.h>

using namespace nntrainer::ddtree;

int main(int argc, char **argv) {
  if (argc < 9) {
    fprintf(stderr,
            "usage: %s depth vocab budget root start past in.f32 out.json\n",
            argv[0]);
    return 2;
  }
  const int depth = atoi(argv[1]);
  const int vocab = atoi(argv[2]);
  const int budget = atoi(argv[3]);
  const int root = atoi(argv[4]);
  const int start = atoi(argv[5]);
  const int past = atoi(argv[6]);
  const std::string fin = argv[7], fout = argv[8];

  std::vector<float> logits((size_t)depth * vocab);
  std::ifstream f(fin, std::ios::binary);
  f.read(reinterpret_cast<char *>(logits.data()),
         (std::streamsize)(logits.size() * sizeof(float)));
  if (!f) {
    fprintf(stderr, "failed to read %s\n", fin.c_str());
    return 1;
  }
  f.close();

  DDTreeConfig cfg;
  cfg.budget = budget;
  cfg.depthLimit = depth;
  cfg.maskFillValue = std::numeric_limits<float>::lowest();

  DDTreeStructure tree = buildTree(logits.data(), depth, vocab, cfg);
  const int cl = tree.currentLength;

  std::vector<int32_t> vii(cl), vpi(cl);
  const int stride = past + cl;
  std::vector<float> mask((size_t)cl * stride);
  compile(root, start, past, tree, cfg, vii.data(), vpi.data(), mask.data(),
          stride);

  // posterior: leftmost-child token (min key) per node, else -1 (matches py)
  std::vector<int32_t> posterior(cl);
  for (int i = 0; i < cl; ++i) {
    const auto &cm = tree.childMaps[i];
    if (cm.empty()) {
      posterior[i] = -1;
    } else {
      int32_t mn = INT32_MAX;
      for (const auto &kv : cm)
        if (kv.first < mn)
          mn = kv.first;
      posterior[i] = mn;
    }
  }
  Accepted acc = followVerified(tree.childMaps, posterior.data());

  FILE *o = fopen(fout.c_str(), "w");
  fprintf(o, "{\n");
  fprintf(o, "\"node_count\": %d,\n", tree.nodeCount);
  fprintf(o, "\"current_length\": %d,\n", cl);
  auto arr = [&](const char *name, const int32_t *p, int n) {
    fprintf(o, "\"%s\": [", name);
    for (int i = 0; i < n; ++i)
      fprintf(o, "%s%d", i ? "," : "", p[i]);
    fprintf(o, "],\n");
  };
  arr("node_token_ids", tree.nodeTokenIds.data(), tree.nodeCount);
  arr("node_depths", tree.nodeDepths.data(), tree.nodeCount);
  arr("parents", tree.parents.data(), cl);
  arr("verify_input_ids", vii.data(), cl);
  arr("verify_position_ids", vpi.data(), cl);
  fprintf(o, "\"visibility\": [");
  for (int i = 0; i < cl; ++i) {
    fprintf(o, "%s[", i ? "," : "");
    for (int j = 0; j < cl; ++j)
      fprintf(o, "%s%d", j ? "," : "", (int)tree.visibility[(size_t)i * cl + j]);
    fprintf(o, "]");
  }
  fprintf(o, "],\n");
  fprintf(o, "\"mask_visible\": [");
  for (int i = 0; i < cl; ++i) {
    fprintf(o, "%s[", i ? "," : "");
    for (int j = 0; j < past + cl; ++j)
      fprintf(o, "%s%d", j ? "," : "",
              (mask[(size_t)i * stride + j] == 0.0f) ? 1 : 0);
    fprintf(o, "]");
  }
  fprintf(o, "],\n");
  arr("posterior", posterior.data(), cl);
  fprintf(o, "\"accepted_indices\": [");
  for (size_t i = 0; i < acc.indices.size(); ++i)
    fprintf(o, "%s%d", i ? "," : "", acc.indices[i]);
  fprintf(o, "],\n");
  fprintf(o, "\"next_token\": %d\n}\n", acc.nextToken);
  fclose(o);

  printf("cpp node_count=%d current_length=%d accepted_len=%zu next_token=%d\n",
         tree.nodeCount, cl, acc.indices.size(), acc.nextToken);
  return 0;
}
