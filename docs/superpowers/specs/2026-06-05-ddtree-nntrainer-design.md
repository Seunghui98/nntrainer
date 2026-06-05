# DDTree (Tree-based Speculative Decoding) for nntrainer — Design Spec

- **Date:** 2026-06-05
- **Branch:** `add/DDTree`
- **Reference implementation (source of truth):** `/home/shsh1004/littlesd_inference/ddtree.py`
- **Paper:** https://arxiv.org/abs/2604.12989
- **Target codebase:** `/home/shsh1004/nntrainer`
- **Downstream consumer (consider only, do NOT implement now):** `/home/shsh1004/Quick.AI` (QNN on-device SD = Speculative Decoding)

---

## 1. Goal

Port the DDTree tree-based speculative-decoding algorithm from the Python reference into nntrainer
as a reusable, self-contained C++ core, plus the minimal nntrainer runtime capabilities required to
**actually run and verify it on CPU**. The result must be shaped so that Quick.AI can later wire it into
its QNN SD pipeline by `#include`-ing a header and linking `libnntrainer` — without Quick.AI-side code
being written in this effort.

**The Python implementation is the canonical reference.** Every algorithmic decision is defined by
`ddtree.py`; when in doubt, match the Python behavior bit-for-bit.

---

## 2. Background

DDTree is a **tree-based speculative decoding** method. Per decode round:

1. A **draft model** produces draft logits over `depth_limit = block_size - 1` positions.
2. `build_ddtree_tree` builds a best-first candidate **token tree** (≤ `1 + budget` nodes) from those logits.
3. `compile_ddtree_tree` flattens the tree into `verify_input_ids`, `verify_position_ids`, and an
   **additive attention mask** encoding tree visibility (each node attends to prefix + its ancestors).
4. The **target model** verifies all tree nodes in **one forward pass**; `follow_verified_tree` walks the
   longest accepted root→leaf path.
5. `compact_dynamic_cache` shrinks the appended KV-cache window down to the accepted path.

"**32 tree**" = a candidate tree of **32 total nodes**. Since Python uses `max_tree_nodes = 1 + tree_budget`,
this corresponds to **`tree_budget = 31`** (matches Python benchmark default `--tree-budget "31,63"`).
`budget` is a runtime parameter; the code supports any value. **(Open item — confirm in §13.)**

The core algorithm is **host-side control logic** (heap expansion, small integer arrays, O(L²) bitmaps with
L ≤ 33). It is not a heavy tensor kernel and is backend-agnostic.

---

## 3. Scope

### In scope (implemented in nntrainer)

| # | Item | Nature |
|---|------|--------|
| 1 | **DDTree core module** `nntrainer/nntrainer/ddtree/` — `buildTree`, `compile` (+ sliding-window stage), `followVerified`, keep-index, reusable raw-pointer `compactTail` helper, optional sampling helper | Pure algorithm |
| 2 | **`mha_core` extension** — consume an arbitrary **additive attention mask** (the `MASK` input that is currently commented out) instead of the built-in causal triangle | nntrainer runtime |
| 3 | **`KVCacheManager::compactTail(pastLen, keepIndices)`** — reorder appended KV tail by kept indices + `setPosition(pastLen + keepCount)` | nntrainer runtime (CPU) |
| 4 | **Unit tests** (per-feature) + **CPU integration harness** | Verification |
| 5 | **README** with architecture, structure, and usage-guide examples | Documentation |

### Out of scope (Quick.AI / QNN — interface considered, code not written)

- QNN draft/target forward driving and the `quick_dot_ai_api.cpp` generate loop.
- Injecting the mask into the QNN attention graph.
- Applying compaction to QNN's own KV cache (Quick.AI calls the same core `compactTail` helper, or implements its own).

---

## 4. Architecture

### 4.1 Responsibility boundary

| Stage | Owner | Python ref |
|-------|-------|-----------|
| draft model forward → draft logits | **caller** (Quick.AI / CPU harness) | `model(...)` |
| build candidate tree | **core** | `build_ddtree_tree` |
| compile verify ids / pos / mask | **core** | `compile_ddtree_tree` |
| sliding-window mask variant | **core** (optional stage, **included**) | `prepare_ddtree_attention_mask_for_target` |
| target verify forward (appends whole tree to KV) | **caller** + (mha_core mask path) | `target(...)` |
| sample posterior from verify logits | **caller** | `sample(...)` |
| follow accepted path | **core** | `follow_verified_tree` |
| compute keep-indices | **core** | (= `accepted_indices`) |
| physically reorder KV tail | **runtime owner** (nntrainer `KVCacheManager` for CPU; QNN for device) using core helper | `compact_dynamic_cache` / `_compact_appended_window` |
| KV cache storage / ownership | **runtime owner** (never the core — core is stateless) | `DynamicCache` |

**Key principle:** the DDTree core is **stateless and cache-agnostic** — it never holds a KV cache. It only
emits *what to keep* and provides a reusable *how-to-reorder* helper. The cache lives in the model runtime.

### 4.2 Module layout

```
nntrainer/nntrainer/ddtree/
├── meson.build           # ddtree_sources / ddtree_headers
├── ddtree_types.h        # DDTreeConfig, DDTreeStructure, CompiledTree, Accepted
├── ddtree.h / .cpp       # buildTree, compile, followVerified
├── ddtree_sliding.h/.cpp # sliding-window mask stage
├── ddtree_compact.h/.cpp # keep-index + raw-pointer compactTail helper
└── ddtree_sampling.h/.cpp# argmax / temperature helper (convenience; caller owns loop sampling)
```

Register `'ddtree'` in `nntrainer/nntrainer/meson.build`'s `nntrainer_elements`. Per the leaf-module
convention, `ddtree_sources` append to `nntrainer_sources` (compiled into `libnntrainer`) and
`ddtree_headers` append to `nntrainer_headers` (installed via `install_headers` to `include/nntrainer`),
so Quick.AI gets the headers automatically.

Dependencies: standard library only (no model/Tensor dependency in the pure core; a thin
`nntrainer::Tensor` overload may adapt via `getData<float>()`).

---

## 5. Core API surface

All pure host logic. Buffers are caller-provided; the core never allocates model state.

```cpp
namespace nntrainer::ddtree {

struct DDTreeConfig {
  int   budget       = 31;     // "32 tree" = 32 nodes = budget 31 (runtime param)
  int   depthLimit;            // = block_size - 1 (draft horizon)
  float maskFillValue;         // additive-mask "-inf" (= finfo(target.dtype).min); fp32 or fp16
};

// build_ddtree_tree
struct DDTreeStructure {
  std::vector<int32_t> nodeTokenIds;   // [nodeCount]
  std::vector<int32_t> nodeDepths;     // [nodeCount]
  std::vector<int32_t> parents;        // [1+nodeCount], parents[0] = -1 (root placeholder)
  std::vector<std::unordered_map<int32_t,int32_t>> childMaps; // [1+nodeCount], follow() lookup
  std::vector<uint8_t> visibility;     // (L*L) flattened bitmap, L = 1+nodeCount
  int nodeCount, currentLength;        // currentLength = 1 + nodeCount
};
DDTreeStructure buildTree(const float* draftLogits, int depthLimit,
                          int vocab, const DDTreeConfig& cfg);

// compile_ddtree_tree — writes into caller buffers, returns slice lengths
struct CompiledTree { int pastLength, currentLength; };
CompiledTree compile(int32_t rootTokenId, int start, int pastLength,
                     const DDTreeStructure& tree, const DDTreeConfig& cfg,
                     int32_t* verifyInputIds,     // [currentLength]
                     int32_t* verifyPositionIds,  // [currentLength]
                     float*   attentionMask,      // [currentLength, pastLength+currentLength]
                     int      attnMaskRowStride);

// prepare_ddtree_attention_mask_for_target (sliding-window stage)
struct SlidingMasks { float* full; float* sliding; bool hasSliding; };
SlidingMasks makeSlidingMasks(float* attentionMask, const int32_t* verifyPositionIds,
                              const CompiledTree& c, int slidingWindow,
                              const DDTreeConfig& cfg, float* slidingBuffer);

// follow_verified_tree
struct Accepted { std::vector<int32_t> indices; int32_t nextToken; };
Accepted followVerified(const std::vector<std::unordered_map<int32_t,int32_t>>& childMaps,
                        const int32_t* posterior /* [currentLength] */);

// _compact_appended_window — reusable, model-agnostic (raw pointers)
//   reorders [pastLen, pastLen+tailLen) by keepIndices into [pastLen, pastLen+keepCount)
void compactTail(void* cacheBase, int elemSizeBytes, int seqStrideElems, int rowElems,
                 int pastLen, int tailLen, const int32_t* keepIndices, int keepCount);

} // namespace nntrainer::ddtree
```

---

## 6. Parity requirements (must hold bit-for-bit vs Python)

These were found by cross-checking the design against `ddtree.py` and **must** be encoded in code + tests:

1. **Heap tie-break ordering** — Python key `(-logw, ranks_tuple, parent_index, depth, rank, logw)` compared
   lexicographically (lines 126/150/155). Effective tie-break: `ranks → parent_index → depth → rank`. The C++
   comparator must replicate this exactly. *(Highest-risk item.)*
2. **Numeric precision** — `top_log_probs` stored **fp32**; cumulative `logw` accumulated in **double**
   (Python promotes float32→double via `float(...)`, lines 117/149/154). Mismatch breaks tie-break #1.
3. **topk & edge cases** — `topk = min(budget, vocab)` (line 110). Empty case (`budget<=0` or `depthLimit==0`)
   returns single root: `visibility=[[1]]`, `parents=[-1]`, no nodes (lines 98–108).
4. **compile mask structure** — `pos[0]=start`, `pos[i]=start+depth[i]`; mask buffer zeroed first so the
   **prefix block `[0:pastLen]` is fully visible (0)**, tree block `[pastLen:pastLen+cur]` filled with
   `maskFillValue` then opened where `visibility==1` (lines 195–215).
5. **Sliding-window arithmetic** — `key_positions[:past]=arange` (assumes dense absolute prefix positions),
   `[past:]=verify_position_ids`; visible iff `(key<=query) & (key>query-window)`; fallbacks: no sliding layer →
   return mask unchanged; `window<=0` → `full==sliding` (lines 219–255).
6. **Compaction keep-index** — `keepIndices == accepted_indices` directly (includes root index 0); assert tail
   length == `currentLength`; final logical length = `pastLen + keepCount` (lines 296–421, 595–600).
7. **Sampling ownership** — `sample()` is caller-side (line 581); the core consumes a per-node `posterior`.
   `ddtree_sampling.h` is a convenience only.

---

## 7. nntrainer runtime changes

### 7.1 `mha_core` additive-mask path (Applications/CausalLM/layers/mha_core.{h,cpp})

- Today `incremental_forwarding` applies `softmax_triangle` (built-in causal triangle); the `MASK=3` /
  `attention_mask` input is declared but commented out ("intended for later use").
- Add a code path: **when a mask tensor is provided**, apply `score + additive_mask` then a full (non-triangular)
  softmax, instead of `softmax_triangle`. Supports the arbitrary tree-DAG visibility mask.
- Sliding-window models: apply `full` vs `sliding` mask per layer type (ties into §6.5).

### 7.2 `KVCacheManager::compactTail` (Applications/CausalLM/kv_cache_manager.{h,cpp})

- New method `compactTail(unsigned int pastLen, const std::vector<int32_t>& keepIndices)`:
  for each layer, apply core `compactTail` helper to `key_cache`/`value_cache` tail, then
  `setPosition(pastLen + keepIndices.size())`.
- `KVCacheManager` currently has only `advance` (forward write-position); compaction (shrink-to-accepted) is new
  and **required for correctness** — without it, rejected branches pollute the cache and corrupt later decoding.

---

## 8. Verification strategy

Two independent layers. **gemma4 is available in nntrainer CausalLM** (`Applications/CausalLM/models/gemma4/`,
`Gemma4CausalLM : public CausalLM, public Gemma4Transformer`) after fast-forwarding `add/DDTree` to
`upstream/main` (upstream `nntrainer/nntrainer`). It uses `mha_core` + `KVCacheManager` (so §7 changes apply) and
has **sliding-attention layers** (`layer_types: ["sliding_attention", ...]`), so the §6.5 sliding-window mask
stage is **required, not optional**, for gemma4 parity.

> **gemma4 sliding-cache density (parity item, Python 305–312):** `_compact_appended_window` requires a **dense**
> target KV cache so it can compact by absolute DDTree slots; HF windowed sliding caches cannot. nntrainer's
> `KVCacheManager` stores a dense `(batch,1,max_seq_len,kv_width)` cache, so this is naturally satisfied — but
> `compactTail` must operate on the dense cache by absolute slot (do not introduce HF-style windowed tails).

### 8.1 Trace-replay parity test — exact gemma4 parity (fast / isolated)

To get **exact gemma4-level token parity without porting gemma4 forward into nntrainer**:

1. Instrument the Python `ddtree_generate` (gemma4, the littlebit draft checkpoint) to dump, per round:
   `draft_logits`, the sampled `posterior`, plus the final tree structure, accepted indices, and output token
   sequence — to JSON/`.npy`.
2. The C++ test **replays** the captured `draft_logits` through `buildTree`/`compile` and the captured
   `posterior` through `followVerified` + compaction, asserting **identical** trees, masks, accepted paths, and
   the **identical final token sequence**.
3. This isolates the DDTree algorithm from the model forward → satisfies "runtime identical to Python, model =
   gemma4" rigorously for the algorithm, at the configured **32 tree / 16 tokens**.

### 8.2 CPU live-forward harness — true gemma4 end-to-end (primary)

- A driver loads **gemma4** (`Gemma4CausalLM` + `gemma4_littlebit_checkpoint_sample` draft) and runs the full
  DDTree loop with **real forwards on CPU**, comparing the output token sequence to the Python `ddtree_generate`
  gemma4 run at **32 tree (budget 31) / 16 tokens / temperature 0**.
- Exercises the **live integration**: `mha_core` additive-mask path + sliding-window mask stage +
  `KVCacheManager::compactTail` against real gemma4 forwards.
- Because gemma4 is now in nntrainer, this gives **token-for-token gemma4 parity on CPU** directly, not only via
  trace-replay. §8.1 remains as a fast, model-load-free algorithm parity check.

---

## 9. Unit test plan (per feature)

`test/unittest/unittest_ddtree.cpp` (gtest), registered in `test/unittest/meson.build` `test_target`.
Golden vectors under `test/unittest/ddtree_golden/` (dumped from Python on small fixed inputs).

| Suite | Cases | Verifies |
|-------|-------|----------|
| `DDTreeBuild` | small fixed logits; tie-break edge (equal `logw`); `budget>vocab`; empty (`budget<=0`, `depth==0`); full budget fill | §6.1, §6.2, §6.3; tree/visibility/parents == golden |
| `DDTreeCompile` | verify ids/pos values; prefix block all-visible; tree block == visibility; mask fill value | §6.4 |
| `DDTreeSliding` | no-sliding passthrough; `window<=0`; positive-window visibility; full vs sliding | §6.5 |
| `DDTreeFollow` | full accept; partial accept; immediate reject; deep path | `follow_verified_tree` |
| `DDTreeCompact` | identity (no-op); subset reorder; empty keep; length-mismatch assertion | §6.6, raw-pointer helper |
| `DDTreeParityReplay` | replay captured gemma4 trace → identical final token sequence | §8.1, 32 tree / 16 tokens |
| `MHACoreMaskedAttention` | small Q/K/V + known tree mask vs numpy reference | §7.1 |
| `KVCacheCompactTail` | append tail → compact → read view matches expected | §7.2 |

---

## 10. Deliverables

1. **Spec** — this document (committed on `add/DDTree`).
2. **Core module** `nntrainer/nntrainer/ddtree/` (§4.2, §5) + meson wiring.
3. **Runtime changes** — `mha_core` mask path (§7.1), `KVCacheManager::compactTail` (§7.2).
4. **Tests** — per-feature unit tests + golden vectors + CPU harness (§8, §9).
5. **README** `nntrainer/nntrainer/ddtree/README.md` — architecture, module structure, and usage-guide examples
   (pure-core usage, CPU harness usage, and an illustrative Quick.AI/QNN integration snippet).
6. **CPU usage guide** — documented steps to build + run the CPU harness and the trace-replay parity test.

---

## 11. Acceptance criteria

A change is complete only when **all** hold (evidence required, no assertions without command output):

1. **Build succeeds** — `meson setup builddir && meson compile -C builddir` clean (x86/CPU).
2. **Unit tests pass** — `meson test` green, including all suites in §9.
3. **Python parity** — the trace-replay parity test (§8.1) produces the **identical token sequence** to the
   Python `ddtree_generate` run, **verification model = gemma4**, at **32 tree (budget 31) / 16 generated tokens**.
4. **CPU runtime** — the CPU live-forward harness (§8.2) runs end-to-end without cache corruption.

---

## 12. Run / verification configuration

- **Tree:** 32 nodes ⇒ `tree_budget = 31` (confirm §13).
- **Generation length:** `max_new_tokens = 16`.
- **Verification model:** gemma4 (`gemma-4-E2B-it`) + `gemma4_littlebit_checkpoint_sample` draft.
- **Temperature:** 0.0 (greedy) for deterministic parity (Python default).
- **Reference command (Python):** derived from `littlesd_inference/run_benchmark_gemma4.sh` with
  `--block-size` from draft model, `--tree-budget 31`, `--max-new-tokens 16`, `--temperature 0`.

---

## 13. Risks & open decisions

| # | Item | Resolution needed |
|---|------|-------------------|
| R1 | **"32 tree" = budget 31 vs 32** | Confirm interpretation (spec assumes **budget 31 = 32 nodes**, per Python `1+budget` and default `31,63`). |
| R2 | ~~gemma4 not in nntrainer~~ **RESOLVED** | `add/DDTree` fast-forwarded to `upstream/main`; `Gemma4CausalLM` now present and uses `mha_core`+`KVCacheManager`+sliding attention. True gemma4-on-CPU parity (§8.2) is achievable. |
| R3 | **Heap tie-break + fp32/double parity** (§6.1, §6.2) | Highest implementation risk; locked by golden-vector tests before anything depends on tree output. |
| R4 | **mha_core mask path correctness** | Tree mask is non-causal; must not regress existing causal path (gate on MASK presence). |
| R5 | **Quick.AI/QNN integration contract** | Documented only; verify mask format + KV `compactTail` signature align when Quick.AI integration happens later. |

---

## 14. References

- Python reference: `/home/shsh1004/littlesd_inference/ddtree.py` (functions: `build_ddtree_tree`,
  `compile_ddtree_tree`, `prepare_ddtree_attention_mask_for_target`, `follow_verified_tree`,
  `compact_dynamic_cache`, `_compact_appended_window`, `ddtree_generate`).
- nntrainer leaf-module convention: `nntrainer/nntrainer/meson.build`, `nntrainer/nntrainer/tensor/meson.build`.
- nntrainer attention: `Applications/CausalLM/layers/mha_core.{h,cpp}` (`MASK=3`, `softmax_triangle`).
- nntrainer KV cache: `Applications/CausalLM/kv_cache_manager.{h,cpp}`.
- Tensor host access: `nntrainer/tensor/tensor.h` (`getData<T>()`).
