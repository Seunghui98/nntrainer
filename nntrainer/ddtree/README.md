# DDTree — Tree-based Speculative Decoding Core for nntrainer

A stateless, cache-agnostic C++ core that ports the **DDTree** tree-based
speculative-decoding algorithm from the Python reference
(`/home/shsh1004/littlesd_inference/ddtree.py`) into nntrainer. The core builds a
best-first candidate token tree from draft logits, compiles it into verify
buffers + an additive attention mask, follows the longest accepted root→leaf
path after target verification, and provides a reusable KV-tail compaction
helper.

- **Design spec:** `docs/superpowers/specs/2026-06-05-ddtree-nntrainer-design.md`
- **Implementation plan:** `docs/superpowers/plans/2026-06-05-ddtree-nntrainer.md`
- **Ground truth:** `ddtree.py` — every algorithmic decision matches it bit-for-bit.
- **Paper:** arxiv 2604.12989

---

## Architecture

The core is **pure host logic** and **never holds a KV cache**. It only decides
*what to keep* and provides a *how-to-reorder* helper. The KV cache and all model
forwards live in the runtime (nntrainer CausalLM for CPU; QNN for on-device).

| Stage | Owner | Core function | ddtree.py |
|-------|-------|---------------|-----------|
| draft forward → draft logits | caller | — | `model(...)` |
| build candidate tree | **core** | `buildTree` | `build_ddtree_tree` |
| compile verify ids/pos/mask | **core** | `compile` | `compile_ddtree_tree` |
| sliding-window mask variant | **core** | `makeSlidingMasks` | `prepare_ddtree_attention_mask_for_target` |
| target verify forward (appends tree to KV) | caller + mha_core mask path | — | `target(...)` |
| sample posterior from verify logits | caller (`argmaxRow` helper) | `argmaxRow` | `sample(...)` |
| follow accepted path | **core** | `followVerified` | `follow_verified_tree` |
| keep-indices | **core** | `followVerified().indices` | `accepted_indices` |
| physically reorder KV tail | runtime, via core helper | `compactTail` | `_compact_appended_window` |
| KV cache storage | runtime (never the core) | — | `DynamicCache` |

---

## Module structure

```
nntrainer/nntrainer/ddtree/
├── meson.build            # appends ddtree_sources/headers to libnntrainer
├── ddtree_types.h         # DDTreeConfig, DDTreeStructure, CompiledTree, SlidingMasks, Accepted
├── ddtree.h / .cpp        # buildTree, compile, followVerified
├── ddtree_sliding.h/.cpp  # makeSlidingMasks (sliding-window additive mask)
├── ddtree_compact.h/.cpp  # compactTail (raw-pointer KV tail reorder)
└── ddtree_sampling.h/.cpp # argmaxRow / sampleGreedy (temperature-0 convenience)
```

Headers install to `include/nntrainer` automatically (leaf-module convention), so
downstream consumers (e.g. Quick.AI/QNN) get them by linking `libnntrainer`.
Dependencies: **C++ standard library only** — no Tensor/model dependency.

---

## Parity requirements (locked by tests)

These must hold bit-for-bit vs `ddtree.py` (spec §6):

1. **Heap tie-break** — best-first heap ordered by the Python tuple
   `(-logw, ranks, parent_index, depth, rank, logw)`; `ranks` is variable-length
   and compared lexicographically (shorter prefix sorts first, like
   `std::vector::operator<`). Effective tie-break: `ranks → parent → depth → rank`.
2. **Numeric precision** — `top_log_probs` stored **fp32**; cumulative `logw`
   accumulated in **double**. (Mismatch flips tie-break order — see the
   `budget_gt_vocab` golden case.)
3. **topk & edge cases** — `topk = min(budget, vocab)`; `budget<=0` or
   `depthLimit==0` returns a single root (`visibility=[[1]]`, `parents=[-1]`).
4. **compile mask** — prefix block `[0:past]` fully visible (0); tree block
   filled with `maskFillValue` then opened (0) where `visibility==1`;
   `pos[0]=start`, `pos[i]=start+depth[i]`.
5. **sliding window** — `key_pos[:past]=arange`, `key_pos[past:]=verify_pos`;
   visible iff `(key<=query) & (key>query-window)`; no sliding layers → mask
   unchanged; `window<=0` → full==sliding.
6. **keep-index** — `keepIndices == accepted_indices` (includes root index 0);
   tail length must equal the verified window; final length = `past + keepCount`.
7. **sampling** — caller-side; `ddtree_sampling.h` is a convenience only.

Golden vectors dumped from the reference (`test/unittest/ddtree_golden/`) lock
items 1–3; per-feature unit tests lock 4–7. (torch/numpy are unavailable on this
machine, so `gen_golden.py` is a faithful stdlib transcription of
`build_ddtree_tree` using Python's native tuple-heap — identical tie-break
semantics. End-to-end token parity against real gemma4 lives in the trace-replay
test.)

---

## Pure-core usage

```cpp
#include <ddtree.h>
#include <ddtree_sliding.h>
#include <ddtree_sampling.h>
#include <ddtree_compact.h>
using namespace nntrainer::ddtree;

DDTreeConfig cfg;
cfg.budget = 31;                 // "32 tree" == 32 nodes
cfg.depthLimit = blockSize - 1;  // draft horizon
cfg.maskFillValue = -3.4028235e38f; // finfo(float32).min (use fp16 min for fp16)

// 1) build the candidate tree from draft logits [depthLimit, vocab].
DDTreeStructure tree = buildTree(draftLogits, cfg.depthLimit, vocab, cfg);

// 2) compile verify buffers + additive mask. Caller owns the buffers.
const int cur = tree.currentLength;
std::vector<int32_t> ids(cur), pos(cur);
const int stride = past + cur;
std::vector<float> mask((size_t)cur * stride);
CompiledTree c = compile(rootTokenId, start, past, tree, cfg,
                         ids.data(), pos.data(), mask.data(), stride);

// 3) (sliding models) derive full vs sliding masks.
std::vector<float> slidingBuf((size_t)cur * stride);
SlidingMasks m = makeSlidingMasks(mask.data(), pos.data(), cur, past + cur,
                                  slidingWindow, hasSlidingLayers, cfg,
                                  slidingBuf.data());

// 4) caller runs the target verify forward with ids/pos/mask -> verifyLogits.
//    Then sample a posterior token per node (temperature 0 == argmax):
std::vector<int32_t> posterior(cur);
sampleGreedy(verifyLogits, cur, vocab, posterior.data());

// 5) follow the accepted path.
Accepted a = followVerified(tree.childMaps, posterior.data());
// accepted tokens = ids[a.indices...]; bonus token = a.nextToken.

// 6) compact the KV tail to the accepted path (runtime owns the cache).
//    See KVCacheManager::compactTail, which calls the core helper per layer.
```

---

## CPU harness usage (gemma4 end-to-end)

> **Status:** the CPU live-forward harness and the two runtime hooks
> (`mha_core` additive-mask path, `KVCacheManager::compactTail`) require building
> `Applications/CausalLM`, which needs system dev packages not present on this
> machine — see **Build** below.

```bash
# build the harness (after installing app deps + configuring build-app)
ninja -C build-app nntr_ddtree_harness

# run gemma4 target + littlebit draft, budget 31, 16 tokens, temperature 0
./build-app/Applications/CausalLM/nntr_ddtree_harness \
  --target <gemma4_dir> --draft <draft_dir> \
  --prompt "<prompt>" --tree-budget 31 --max-new-tokens 16
```
Compare the printed token ids against the Python `ddtree_generate` gemma4 run at
the same config — they must be identical.

### Regenerating golden vectors / traces
```bash
python3 test/unittest/ddtree_golden/gen_golden.py     # build_golden.{txt,json}
# trace capture (needs working torch + gemma4 checkpoints): see plan Task 13
```

---

## Build & test

The pure core builds into `libnntrainer` and is tested with `unittest_ddtree`.

```bash
# Pure core + unit tests (this machine's working configuration):
meson setup build \
  -Denable-fp16=true -Denable-test=true \
  -Denable-tflite-backbone=false -Denable-tflite-interpreter=false \
  -Denable-app=false
meson test -C build unittest_ddtree -v
```

> **Why the extra flags:** the default build pulls TensorFlow-Lite, and
> `-Denable-app` force-builds `Applications/ReinforcementLearning/DeepQ`
> (needs `libjsoncpp-dev`) and `YOLOv2/3` (need `libopencv-dev`) — none present
> here, and there is no per-app flag to skip them.

### Runtime hooks + harness (Tasks 9, 10, 14)
These touch `Applications/CausalLM` and need a separate app build:
```bash
sudo apt install -y libjsoncpp-dev libopencv-dev
meson setup build-app \
  -Denable-fp16=true -Denable-test=true -Denable-transformer=true \
  -Denable-tflite-backbone=false -Denable-tflite-interpreter=false
```

---

## Verification record

Pure-core unit suite (`unittest_ddtree`), 19 tests across 7 suites — see the
testlog for the latest run:
- `DDTreeScaffold`, `DDTreeBuild` (incl. `MatchesPythonGolden`), `DDTreeCompile`,
  `DDTreeSliding`, `DDTreeFollow`, `DDTreeCompact`, `DDTreeSampling`.

```
meson test -C build unittest_ddtree
1/1 unittest_ddtree   OK
Ok: 1   Fail: 0
```

Runtime-hook tests (`KVCacheCompactTail`, `MHACoreMaskedAttention`) and the
gemma4 trace-replay / live-forward parity are pending the app build + Python
checkpoints (see Build above).

---

## Quick.AI / QNN integration (interface only — not implemented here)

A downstream QNN pipeline reuses the same core unchanged:
```cpp
#include <ddtree.h>
#include <ddtree_compact.h>
// 1) buildTree / compile / makeSlidingMasks exactly as above.
// 2) inject the additive mask into the QNN attention graph.
// 3) after verify, followVerified -> accepted indices.
// 4) call ddtree::compactTail(...) on QNN's own dense KV tail (same helper),
//    or implement an equivalent reorder. The core stays cache-agnostic.
```
The mask format (`[currentLength, pastLength+currentLength]` additive, 0 visible /
`maskFillValue` hidden) and the `compactTail` signature are the integration
contract.
