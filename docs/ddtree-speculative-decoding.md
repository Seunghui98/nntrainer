# DDTree Speculative Decoding 🌲

This document describes how to integrate nntrainer's **DDTree** (tree-based
speculative decoding) into Quick.AI's QNN/NPU runtime to accelerate token
generation for supported QNN models (for example, Gemma 4).

DDTree itself lives in nntrainer as a small, dependency-free C++ core. Quick.AI
links that core, supplies the NPU forward passes (draft and verify), and adapts
attention masks to the QNN tensor format. The tree-construction and
accepted-path logic are reused as-is; Quick.AI does not reimplement them.

| | |
|---|---|
| **nntrainer reference branch** | `add/DDTree` (latest verified: `53a09b59`) |
| **Core headers** | `nntrainer/ddtree/*.h` (installed to `include/nntrainer/`) |
| **Quick.AI target** | the SD-enabled QNN model (`src/models/qnn/<model>/<model>_sd_qnn.*`) |
| **Status** | Reference design. Quick.AI-side draft/verify wiring not yet implemented. |

---

## Contents

1. [Overview](#1-overview)
2. [Terminology](#2-terminology)
3. [Architecture and responsibilities](#3-architecture-and-responsibilities)
4. [Execution flow](#4-execution-flow)
5. [API reference](#5-api-reference)
6. [Data structures](#6-data-structures)
7. [Tree construction algorithm](#7-tree-construction-algorithm)
8. [Attention masks](#8-attention-masks)
9. [Mask format conversion](#9-mask-format-conversion)
10. [Integration with the SD QNN model](#10-integration-with-the-sd-qnn-model)
11. [Prerequisites and constraints](#11-prerequisites-and-constraints)
12. [Appendix A — Reference call skeleton](#appendix-a--reference-call-skeleton)

---

## 1. Overview

A standard autoregressive decode produces one token per forward pass. Because an
NPU forward is expensive, speculative decoding amortizes that cost:

- **Draft** — cheaply propose candidates for the next several positions.
- **Verify** — run a single target forward over all candidates and accept the
  longest prefix the target agrees with.

DDTree proposes candidates as a **tree** rather than a single linear chain. One
verify forward therefore evaluates multiple alternative continuations at once,
which raises the number of tokens accepted per step.

Two configuration values control the tree:

- **Draft horizon** (`depthLimit`) `= block_size - 1` — the maximum tree depth.
  A `block_size` of 16 yields a horizon of 15 ("16 tokens").
- **Budget** (`budget`) — the number of nodes beyond the root. `budget = 31`
  produces a 32-node tree ("32 tree").

Both are runtime fields of `DDTreeConfig`; changing them requires no nntrainer
rebuild. Depth and node count are independent: depth bounds how far the tree
extends, while the budget bounds how many nodes are spent across that depth.

---

## 2. Terminology

| Term | Meaning |
|---|---|
| **Target model** | The full model whose output is authoritative. Its verify pass determines acceptance. |
| **Draft model** | A lightweight model that cheaply proposes candidate distributions (EAGLE/littlesd style; see §11). |
| **Node** | A candidate token in the tree. The root is the last committed token. |
| **Depth** | A node's distance from the root (1-based for non-root nodes); equals its sequence position offset. |
| **`logw`** | Cumulative log-probability of the path from the root to a node. Higher is more probable. |
| **`visibility`** | Per-node bitmap describing which other nodes a node may attend to (its ancestors and itself). |
| **`currentLength`** | Total node count including the root (`1 + nodeCount`). |
| **`past` / `pastLength`** | Length of the already-cached KV prefix. |
| **Accepted path** | The longest root-to-leaf path the target agrees with during verify. |

---

## 3. Architecture and responsibilities

| Component | Owner | Responsibility |
|---|---|---|
| Tree core (CPU) | nntrainer `ddtree::` | `buildTree`, `compile`, `makeSlidingVisibility` (0/1) / `makeSlidingMasks` (fp32), `followVerified`, `compactTail` — invoked, not reimplemented |
| Forward passes (NPU) | Quick.AI QNN graphs | Draft forward and verify forward |
| Mask conversion (CPU) | Quick.AI | Translate the 0/1 `visibility` (full) and `makeSlidingVisibility` (sliding) bitmaps to the QNN `uint16` format |

Design principles:

1. **Tree construction and acceptance logic stay in nntrainer.** Quick.AI calls
   the functions and consumes their outputs.
2. **Mask value format is Quick.AI's concern.** Only the QNN graph defines the
   meaning of its `uint16` mask values, so nntrainer stays format-neutral and
   Quick.AI performs the conversion.
3. **Correctness is independent of draft quality.** The target verify pass
   guarantees lossless output; draft quality affects only the acceptance rate
   (and therefore throughput).

---

## 4. Execution flow

The example below assumes the prefix `"I love"` has been processed, the last
committed token (root) is `"my"`, `depthLimit = 15`, and `budget = 31`.

```text
DDTree decode cycle

STEP 0  Target prefill -> capture target_hidden                     [NPU]  new
        target_hidden is an input feature for the draft model,
        not a KV cache entry.

STEP 1  Draft forward (littlesd draft graph, single non-causal pass) [NPU]  new (primary work)
        draft_model(target_hidden, block_embeddings)
          -> draft_hidden[horizon] -> target head
          -> draft_logits[horizon, vocab]
        One pass yields all horizon distributions; each row spans the
        full vocabulary, so top-k candidates are available at no extra cost.

STEP 2  ddtree::buildTree(draft_logits, depthLimit, vocab, cfg)      [CPU]  call
          -> DDTreeStructure (visibility, childMaps, nodeTokenIds, ...)

STEP 3  ddtree::compile(...)                                         [CPU]  call
          -> verifyInputIds[<=32], verifyPositionIds[<=32]
          (the fp32 additive mask compile() writes is unused by Quick.AI)

STEP 4  Mask conversion (visibility / sliding -> uint16)             [CPU]  new
          Fill attention_mask and sliding_attention_mask;
          index position_ids by verifyPositionIds (siblings share a position).

STEP 5  Verify forward (verification graph, single pass)             [NPU]  reuse
          -> per-node logits -> posterior[i] = argmax

STEP 6  ddtree::followVerified(childMaps, posterior)                 [CPU]  call
          -> Accepted{ indices (accepted path), nextToken (bonus) }

STEP 7  ddtree::compactTail(KV, accepted indices)                    [CPU]  call
          -> drop rejected nodes from the NPU KV cache, keep accepted

STEP 8  Emit accepted tokens; root = nextToken; repeat
```

Legend: **call** — invoke the nntrainer function directly; **reuse** — existing
QNN graph; **new** — implemented in Quick.AI.

---

## 5. API reference

Namespace `nntrainer::ddtree`. Include the corresponding headers
(`ddtree.h`, `ddtree_sliding.h`, `ddtree_compact.h`, `ddtree_sampling.h`).

### 5.1 `buildTree` — construct the candidate tree (STEP 2)

```cpp
DDTreeStructure buildTree(const float *draftLogits, int depthLimit, int vocab,
                          const DDTreeConfig &cfg);
```

- **When** — immediately after draft logits are available.
- **Input** — `draftLogits`, a row-major `[depthLimit, vocab]` fp32 buffer;
  `cfg.budget`.
- **Output** — a `DDTreeStructure` (visibility, childMaps, nodeTokenIds,
  parents, ...). Node count is `1 + min(budget, vocab)`.

### 5.2 `compile` — flatten the tree into verify buffers (STEP 3)

```cpp
CompiledTree compile(int32_t rootTokenId, int start, int pastLength,
                     const DDTreeStructure &tree, const DDTreeConfig &cfg,
                     int32_t *verifyInputIds, int32_t *verifyPositionIds,
                     float *attentionMask, int attnMaskRowStride);
```

- **When** — immediately after `buildTree`.
- **Output** (written into caller-owned buffers):
  - `verifyInputIds[currentLength]` — `[root, node tokens...]`; used directly (int32).
  - `verifyPositionIds[currentLength]` — `[start, start + depth...]`; used
    directly (siblings share a position).
  - `attentionMask[currentLength, attnMaskRowStride]` — fp32 additive mask for
    nntrainer's own CPU path. **Quick.AI discards this** and derives its mask
    from `tree.visibility` instead (see §9).

### 5.3 `makeSlidingMasks` — sliding-window variant (STEP 4)

```cpp
SlidingMasks makeSlidingMasks(float *attentionMask, const int32_t *verifyPositionIds,
                              int currentLength, int kvLength, int slidingWindow,
                              bool hasSlidingLayers, const DDTreeConfig &cfg,
                              float *slidingBuffer);
```

- **When** — for models that mix full and sliding attention layers (e.g. Gemma 4).
- Behavior:
  - `hasSlidingLayers == false` → `out.sliding == nullptr` (a single mask
    suffices; e.g. full-attention-only models).
  - `slidingWindow <= 0` → `out.sliding == out.full`.
  - otherwise → `slidingBuffer` receives a copy of the full mask with
    out-of-window keys masked; `out.sliding` points to it.
- Visibility rule: `visible = (k <= q) && (k > q - window)`, where `k` and `q`
  are key and query positions.
- This output is an **fp32 additive** mask whose hidden value is
  `cfg.maskFillValue`. A `uint16`/gating consumer must therefore set
  `cfg.maskFillValue` to a non-zero sentinel before calling, then threshold the
  result. For QNN, prefer `makeSlidingVisibility` (§5.3.1), which avoids both.

### 5.3.1 `makeSlidingVisibility` — sliding as a 0/1 bitmap (recommended for QNN)

```cpp
void makeSlidingVisibility(const uint8_t *treeVisibility,
                           const int32_t *verifyPositionIds, int currentLength,
                           int kvLength, int slidingWindow, uint8_t *outVisible);
```

- **When** — instead of `makeSlidingMasks` when the consumer builds its own
  integer/gating mask (e.g. QNN `uint16`).
- **What** — emits the sliding visibility directly as a `[currentLength, kvLength]`
  **0/1** bitmap (`out[i][j] = treeVisible(i,j) AND windowVisible(i,j)`), so the
  full mask (`tree.visibility`) and the sliding mask are produced in the same 0/1
  form. No fp32 additive round-trip and **no `cfg.maskFillValue` dependence**.
- Takes `tree.visibility` (from `buildTree`) and `verifyPositionIds` (from
  `compile`) as inputs; equivalent to thresholding `makeSlidingMasks`'s sliding
  output (validated by `unittest_ddtree`, `DDTreeSliding.VisibilityMatchesThresholdedMask`).

### 5.4 `followVerified` — select the accepted path (STEP 6)

```cpp
Accepted followVerified(
  const std::vector<std::unordered_map<int32_t, int32_t>> &childMaps,
  const int32_t *posterior);
```

- **When** — after the verify forward produces a per-node argmax (`posterior`).
- **Input** — `tree.childMaps`; `posterior[currentLength]` (the token the target
  selected at each node).
- **Output** — `Accepted{ indices, nextToken }`, where `indices[0] == 0` (root)
  and `nextToken` is the bonus token following the accepted path. The cycle
  commits `indices.size() - 1` accepted tokens plus the bonus token.

### 5.5 `compactTail` — compact the KV cache (STEP 7)

```cpp
void compactTail(void *cacheBase, int elemSizeBytes, int seqStrideElems,
                 int rowElems, int pastLen, int tailLen,
                 const int32_t *keepIndices, int keepCount);
```

- **When** — after acceptance, to drop rejected nodes. Call once per layer.
- A model-agnostic raw-pointer routine; applies directly to the QNN KV buffers
  (`uint16_t*`).
- Parameters:
  - `cacheBase` — start of the layer's KV buffer.
  - `elemSizeBytes` — 2 for fp16/uint16, 4 for fp32.
  - `seqStrideElems` — element stride between consecutive sequence rows.
  - `rowElems` — elements per row.
  - `pastLen` — length before the tree was appended.
  - `tailLen` — appended tree length.
  - `keepIndices` / `keepCount` — accepted node indices to retain.
- Effect: gathers the kept rows into `[pastLen, pastLen + keepCount)`.

### 5.6 `argmaxRow` / `sampleGreedy` — sampling helpers

```cpp
int32_t argmaxRow(const float *logits, int vocab);            // ties resolve to lowest index
void    sampleGreedy(const float *logits, int rows, int vocab, int32_t *out);
```

- **When** — to derive `posterior` from verify logits under greedy decoding.
  Quick.AI may substitute its own sampler.

---

## 6. Data structures

Defined in `ddtree_types.h` (plain data; no `nntrainer::Tensor` dependency).

```cpp
struct DDTreeConfig {
  int   budget        = 31;  // node count beyond the root ("32 tree" == 31)
  int   depthLimit    = 0;   // draft horizon = block_size - 1 ("16 tokens" -> 15)
  float maskFillValue = 0;   // additive "-inf" (fp32/fp16 min); unused by Quick.AI
};

struct DDTreeStructure {
  std::vector<int32_t> nodeTokenIds;  // [nodeCount]
  std::vector<int32_t> nodeDepths;    // [nodeCount], 1-based
  std::vector<int32_t> parents;       // [currentLength], parents[0] == -1
  std::vector<std::unordered_map<int32_t, int32_t>> childMaps; // [currentLength]
  std::vector<uint8_t> visibility;    // [currentLength * currentLength], row-major 0/1
  int nodeCount    = 0;
  int currentLength = 1;              // == 1 + nodeCount
};

struct CompiledTree { int pastLength; int currentLength; };

struct SlidingMasks {                 // result of makeSlidingMasks
  float *full;     // for full-attention layers
  float *sliding;  // for sliding layers (nullptr if none; equals full if window <= 0)
  bool  hasSliding;
};

struct Accepted {
  std::vector<int32_t> indices;  // accepted node indices; indices[0] == 0 (root)
  int32_t nextToken;             // bonus token after the accepted path
};
```

The field Quick.AI relies on for masking is `DDTreeStructure.visibility`.

---

## 7. Tree construction algorithm

`buildTree` performs a best-first expansion: the candidate with the highest
cumulative log-probability (`logw`) is added to the tree first.

### Mechanics

- `logw` is the cumulative log-probability of a path; higher means more probable.
- Candidates are stored in a min-heap keyed by `-logw`, so the most probable
  candidate is popped first.
- Each popped node pushes two new candidates:
  - **Child** — the top-1 token at the next depth:
    `child_logw = logw + topLogProb[depth][0]`.
  - **Sibling** — the next-ranked token at the same depth:
    `sibling_logw = logw - topLogProb[d][r] + topLogProb[d][r+1]`.
- Expansion stops once `budget` nodes have been added.

### Why log-probabilities are summed

Path probability is a product of per-step probabilities. Applying a logarithm
(`log(a * b) = log a + log b`) turns the product into a sum. Working in log space
avoids floating-point underflow over long paths and keeps comparisons cheap;
because `log` is monotonic, the highest-probability path is also the
highest-`logw` path.

### Worked example (`depthLimit = 3`, `budget = 5`)

Top-k log-probabilities per position:

```text
pos 1: A=-0.1  B=-1.0  C=-2.5
pos 2: D=-0.2  E=-1.5  F=-3.0
pos 3: G=-0.3  H=-1.8  I=-2.0
```

| Iter | Pop (`logw`) | Node added | Push child | Push sibling | Heap (top) |
|---|---|---|---|---|---|
| 1 | A (-0.1) | A (parent root) | D (-0.3) | B (-1.0) | D, B |
| 2 | D (-0.3) | D (parent A) | G (-0.6) | E (-1.6) | G, B, E |
| 3 | G (-0.6) | G (parent D) | — (max depth) | H (-2.1) | B, E, H |
| 4 | B (-1.0) | B (parent root) | D' (-1.2) | C (-2.5) | D', E, H, C |
| 5 | D' (-1.2) | D' (parent B) | — | — | budget reached |

Resulting tree:

```text
root +- A - D - G      (probable branch extended deeply)
     +- B - D'         (less probable branch kept shallow)
```

The budget concentrates on the most probable paths; E, H, and C are never
expanded. Ties are broken deterministically by a `ranks` tuple, matching the
Python reference byte-for-byte (validated by golden tests).

---

## 8. Attention masks

### 8.1 `visibility` (from `buildTree`)

A `currentLength x currentLength` matrix of 0/1 values; `visibility[i][j] == 1`
means node `i` may attend to node `j`. It is constructed by a single rule:

```text
visibility[i] = visibility[parent[i]]   // inherit the parent's row
visibility[i][i] = 1                     // mark self
```

This yields `visible(i) = {ancestors of i} U {i}`. For the tree
`root, A, D, G, B, D'` with `parents = [-1, 0, 1, 2, 0, 4]`:

```text
        root  A  D  G  B  D'
root :   1    0  0  0  0  0
A    :   1    1  0  0  0  0
D    :   1    1  1  0  0  0
G    :   1    1  1  1  0  0     // G's path only
B    :   1    0  0  0  1  0
D'   :   1    0  0  0  1  1     // D''s path only
```

### 8.2 `sliding` (from `makeSlidingMasks`)

For sliding-attention layers, the full mask is further restricted so that a
query attends only to keys within its window: `visible = (k <= q) && (k > q - window)`.
Example with `window = 4`, query `G` at position `q = 8`, prefix positions 0–4:

```text
        prefix(0..4)   root(5) A(6) D(7) G(8) B D'
full :  visible x5      v       v    v    v    -  -
slide:  - x5            v       v    v    v    -  -
        (prefix dropped: outside the window)
```

| | `visibility` | `sliding` |
|---|---|---|
| Basis | tree parent/child (ancestors) | `visibility` plus a position window |
| Used by | full-attention layers | sliding-attention layers |

---

## 9. Mask format conversion

nntrainer produces an fp32 **additive** mask (`0` = visible, `-inf` = hidden) for
its own CPU attention path. The QNN graph expects a `uint16` mask
(`65535` = visible, `0` = hidden) laid out as a 2-D
`[currentLength, columns]` tensor. Quick.AI performs the conversion, since only
the QNN graph defines its mask convention.

### Full-attention mask — derived from `tree.visibility`

```cpp
// Prefix columns are fully visible; tree columns follow visibility.
for (int i = 0; i < cur; ++i) {
  for (int j = 0; j < past; ++j)
    verification_attention_mask[i * cols + j] = 65535;                 // past KV visible
  for (int j = 0; j < cur; ++j)
    verification_attention_mask[i * cols + past + j] =
        tree.visibility[i * cur + j] ? 65535 : 0;                      // tree visibility
}
```

### Sliding mask — derived from `makeSlidingVisibility` (recommended)

Uses the 0/1 helper, so the sliding mask is built exactly like the full mask —
no fp32 additive round-trip and no `cfg.maskFillValue` dependence:

```cpp
std::vector<uint8_t> svis((size_t)cur * (past + cur));
ddtree::makeSlidingVisibility(tree.visibility.data(), verifyPos, cur, past + cur,
                              slidingWindow, /*out*/ svis.data());
for (int idx = 0; idx < cur * (past + cur); ++idx)
  sliding_attention_mask[idx] = svis[idx] ? 65535 : 0;
```

> Alternative (fp32 path): `makeSlidingMasks` returns an fp32 additive mask;
> thresholding it requires a **non-zero** `cfg.maskFillValue` (the default `0`
> makes hidden and visible indistinguishable) and a contiguous
> `[currentLength, kvLength]` mask (`attnMaskRowStride == past + cur`). Prefer
> `makeSlidingVisibility` to avoid both pitfalls.

### Position IDs

Index the RoPE cos/sin caches per node using `verifyPositionIds`; sibling nodes
share the same position.

No additional mask logic is required in nntrainer: `buildTree` (visibility) and
`makeSlidingMasks` (sliding) already compute the structure. A parameterized
`fillMask` helper could be added to nntrainer to centralize the layout, but it is
optional.

---

## 10. Integration with the SD QNN model

The current implementation (the `*_sd_qnn` model under `src/models/qnn/`, WIP)
loads four graphs (prefill, verification, draft_prefill, generation) and
maintains dual KV caches (`kv_cache_`, `draft_kv_cache_`). The speculative loop
is not yet written: `run()` still generates one token per pass, and the
verification/draft handles are bound but unused.

### Step-to-asset mapping

| Step | Existing asset | Work |
|---|---|---|
| 0 — `target_hidden` | target / prefill graph | new: emit hidden states |
| 1 — draft | current plain draft | new: replace with littlesd draft graph (primary work) |
| 2 — `buildTree` | — | call nntrainer |
| 3 — `compile` | — | call nntrainer |
| 4 — mask conversion | `fill_attention_mask_*` (uint16, 2-D) | new: tree-mask variant |
| 5 — verify | `verification_graph` (already accepts a 2-D `[block_size, cols]` mask) | reuse with tree mask values |
| 6 — `followVerified` | — | call nntrainer |
| 7 — `compactTail` | `kv_cache_` raw `uint16_t*` | call nntrainer (per layer) |

### Build wiring

- Update the nntrainer submodule to `add/DDTree` (latest).
- Add `nntrainer/ddtree` to the include paths in `meson.build` and rebuild.
- Include `<ddtree.h>`, `<ddtree_types.h>`, `<ddtree_sliding.h>`,
  `<ddtree_compact.h>`.

### Constraint

The verification graph width (`block_size`) must be at least the tree node count
(`budget + 1`). A 32-node tree therefore requires a 32-wide verify graph.

---

## 11. Prerequisites and constraints

- **A littlesd draft model is required.** The Python reference draft consumes
  `target_hidden` and emits all horizon distributions in a single non-causal
  forward (EAGLE/littlesd style). A draft model trained for the target, exported
  to a QNN graph, is required; a generic standalone draft model does not match
  this contract.
- **The draft is approximate; the target verify guarantees correctness.** Deep
  non-greedy branches reuse per-depth distributions, so they are approximate.
  The target verify recomputes true per-node logits, keeping output lossless;
  draft quality affects only the acceptance rate.
- **Mask value semantics belong to the QNN graph.** How `uint16` values are
  applied (additive or gating) is owned by Quick.AI. nntrainer remains
  format-neutral (0/1 visibility, fp32 additive).
- **No CPU oracle for QNN-only models.** nntrainer does not contain every QNN
  target; for a model that exists only as a Quick.AI QNN graph, no CPU reference
  can be produced there. (Gemma 4 does have runtime tree parity tests on
  `add/DDTree`: commits `7aa3d940`, `53a09b59`.) Verification for a QNN-only
  model must be performed within Quick.AI.
- **"16 / 32" are configuration values.** Set via `DDTreeConfig{budget, depthLimit}`
  at call time; no nntrainer rebuild is needed.

---

## Appendix A — Reference call skeleton

```cpp
#include <ddtree.h>
#include <ddtree_sliding.h>
#include <ddtree_compact.h>
using namespace nntrainer::ddtree;

DDTreeConfig cfg;
cfg.budget     = 31;
cfg.depthLimit = block_size - 1;

while (generating) {
  // STEP 1: draft (NPU, littlesd) -> draft_logits[depthLimit, vocab]
  run_draft_forward(target_hidden, block_emb, /*out*/ draft_logits);

  // STEP 2-3: tree (CPU)
  DDTreeStructure tree = buildTree(draft_logits, cfg.depthLimit, vocab, cfg);
  int cur = tree.currentLength, past = kv_len;
  std::vector<int32_t> vids(cur), vpos(cur);
  std::vector<float>   fmask((size_t)cur * (past + cur));
  compile(root_token, /*start*/ kv_len, past, tree, cfg,
          vids.data(), vpos.data(), fmask.data(), past + cur);

  // STEP 4: mask conversion (CPU, Quick.AI) — full + sliding both from 0/1 visibility
  fill_tree_mask_uint16(tree.visibility, past, cur, /*out*/ verification_attention_mask);
  if (has_sliding) {
    std::vector<uint8_t> svis((size_t)cur * (past + cur));
    makeSlidingVisibility(tree.visibility.data(), vpos.data(), cur, past + cur,
                          sliding_window, /*out*/ svis.data());
    for (int idx = 0; idx < cur * (past + cur); ++idx)
      sliding_attention_mask[idx] = svis[idx] ? 65535 : 0;
  }
  fill_positions(vpos, /*out*/ verification_position_ids);

  // STEP 5: verify (NPU) -> node_logits -> posterior
  run_verify_forward(vids, verification_attention_mask, verification_position_ids,
                     /*out*/ node_logits);
  std::vector<int32_t> posterior(cur);
  sampleGreedy(node_logits, cur, vocab, posterior.data());

  // STEP 6: accept (CPU)
  Accepted acc = followVerified(tree.childMaps, posterior.data());

  // STEP 7: compact KV (CPU call, NPU KV target) - per layer
  for (auto &layer : kv_layers)
    compactTail(layer.base, /*elem*/ 2, layer.stride, layer.row,
                past, cur - 1, acc.indices.data() + 1, (int)acc.indices.size() - 1);

  // STEP 8: emit / advance
  emit_tokens(acc);
  kv_len += (int)acc.indices.size();
  root_token = acc.nextToken;
}
```

---

*Reference: nntrainer `add/DDTree` (`53a09b59`). Authoritative function
signatures are in `nntrainer/ddtree/*.h`.*
