# DDTree — debug output & sliding-window wiring (handoff for a new session)

**Branch:** `add/DDTree` · **Scope:** how to read the per-block dump, the
tree-dump/renderer added this session, and the one open runtime task
(sliding-window mask wiring for gemma-style models).

> Background/handoff for the core implementation is in `DDTREE-RESUME.md`
> (status, build, verify). This file is specifically about the **debug output**
> and the **sliding-window runtime gap**.

---

## 1. The per-block dump — what every field means

`NNTR_DDTREE_DUMP=<path> ./nntr_causallm <model>` runs `CausalLM::runDDTreeDump`
(`Applications/CausalLM/models/causal_lm.cpp`, dump-only, greedy self-draft) and
writes:

- `<path>.tokens.json` — `gen_ids` (generated tokens), `accept_lengths` (per-block).
- `<path>.blocks.json` — one JSON object per decode block (the focus here).

One block = `root → self-draft HORIZON(15) → buildTree(budget 31 ⇒ 32 nodes) →
masked verify forward → posterior argmax → followVerified → commit`.

**The three arrays have DIFFERENT index/value meanings** (this is the usual
confusion):

| field | index (axis) | value | meaning |
|---|---|---|---|
| `draft_am` | draft **step** 0..14 | **token id** | greedy token-by-token rollout (15 tokens) |
| `node_depths` | **tree node** 1..31 | **depth** | full tree shape (root excluded; root=depth 0) |
| `acc` | accepted-path **rank** 0..alen-1 | **tree node #** | accepted root→leaf path, by node index |
| `parents`* | tree node 0..cl-1 | **parent node #** | exact edges (`parents[0]=-1`) |
| `node_tokens`* | tree node 0..cl-1 | **token id** | every node's token (index 0 = root) |
| `posterior`* | tree node 0..cl-1 | **token id** | target argmax at that node |

`*` = added this session (see §2). Scalars: `pos` (root seq position), `root`
(root token id, == previous block's `next`), `alen` (new tokens committed =
accepted depth + 1), `next` (bonus token = posterior at the accepted leaf).

### How to read it
- Node `i` (i≥1) depth = `node_depths[i-1]`; node 0 is the root (depth 0).
- A child `c` of node `p` is **accepted** iff `p` is accepted AND
  `posterior[p] == node_tokens[c]` (that is exactly `followVerified`). A branch is
  rejected when the parent's posterior points at a different token.
- Accepted tokens = `node_tokens[acc[1..]]`; before the new fields they could only
  be inferred (they equal the `draft_am` prefix, because draft==target==greedy).
- **Accept length per block == 1 + (depth of the deepest path the budget-31 tree
  grew).** Confident/peaky draft ⇒ deep+narrow tree ⇒ high `alen`; uncertain draft
  ⇒ shallow+bushy tree ⇒ low `alen`.

Worked qwen3-0.6b example (lossless, draft==target greedy): 7 blocks,
`alen = 16,14,11,9,6,5,7` → 68 tokens, mean accept length ≈ 9.7. Each block's
accepted path depth equaled that block's max tree depth (tree depth is the
bottleneck, driven by draft confidence). `next` of block k == `root` of block k+1.

---

## 2. What was added this session (committed on `add/DDTree`)

Commit `[CausalLM/ddtree] Dump full tree (parents/tokens/posterior) + ASCII renderer`:

1. **`runDDTreeDump` block record** now also emits the full 32-node tree
   (`Applications/CausalLM/models/causal_lm.cpp`, the `std::ostringstream r`
   block): `parents`, `node_tokens` (= `verify_input_ids`), `posterior`
   (all length == `current_length`, index 0 == root). Previously only
   `node_depths` was dumped, so non-accepted branches' tokens / structure /
   reject-reason were invisible.

2. **`python-example/verify/render_blocks.py`** — renders `blocks.json` as ASCII:
   - NEW dump (parents/node_tokens/posterior present): exact tree with the
     accepted path marked (`▸`) and rejected branches explained.
   - OLD dump (depths only): depth-layer fallback (node count per depth, spine marked).
   - `--model <hf_dir>` decodes token ids to text (transformers optional);
     `--block N` renders one block.
   - Usage: `python python-example/verify/render_blocks.py <path>.blocks.json [--model <hf_dir>] [--block N]`
   - Note: existing/old `blocks.json` only triggers the fallback view; rebuild +
     re-run to get the exact tree.

---

## 3. Reference parity check done this session

`ddtree.py` mask/sliding logic vs nntrainer core (both bit-for-bit, see README §parity):

- **Tree attention mask**: `ddtree.py::compile_ddtree_tree` (195–216) ↔
  `nntrainer/ddtree/ddtree.cpp::compile()`. **Wired & live** in the runtime:
  `compile()` → `MHACoreLayer::setGlobalDDTreeVerify(&maskT,&posT)` →
  `mha_core.cpp:296 attn_mask_ = s_verify_mask_` → `mha_core.cpp:828
  add_mask_and_softmax_full(...)`. (verify node logits == HF 32/32.)
- **Sliding-window mask**: `ddtree.py::prepare_ddtree_attention_mask_for_target`
  (219–255) ↔ `nntrainer/ddtree/ddtree_sliding.cpp::makeSlidingMasks()`.
  Implemented + unit-tested (`test/unittest/unittest_ddtree.cpp:155–193`) but
  **NOT wired into the runtime** — `makeSlidingMasks` is called only from tests.

---

## 4. OPEN TASK — sliding-window runtime wiring (for gemma-style models)

qwen3-0.6b is full-attention, so `prepare_ddtree_attention_mask_for_target` is a
no-op for it (early return at `ddtree.py:228` when no `sliding_attention` layer) —
the qwen3 dump is correct without sliding. Sliding matters only for models that
interleave sliding + full layers (gemma3/gemma4). To run DDTree there, the
**runtime** still needs:

1. **Model-type gate** — port `ddtree.py::should_rebuild_ddtree_target_cache`
   (257–262, detects gemma4 / `sliding_attention` in `layer_types`) and the
   `sliding_window` config read into the CausalLM runtime.
2. **Build both masks** — after `compile()` in `runDDTreeDump`, call
   `makeSlidingMasks(full, verifyPos, cl, past+cl, slidingWindow, hasSliding, cfg,
   slidingBuf)` to get `{full, sliding}`.
3. **Per-layer mask selection in MHA** — `mha_core` currently takes one global
   verify mask (`setGlobalDDTreeVerify`). It must instead pick full vs sliding
   per layer based on the layer's attention type (mirror HF's per-`layer_types`
   selection). Likely: extend the global setter to carry both masks + a per-layer
   flag, or set the right mask before each layer's forward.
4. **Verify** against HF on a gemma sliding model (needs gemma weights + config;
   NOT available in the web container — must run in the `my-dev` Docker on the
   host per `DDTREE-RESUME.md`).

Files: `Applications/CausalLM/models/causal_lm.cpp` (runDDTreeDump),
`Applications/CausalLM/layers/mha_core.{h,cpp}` (mask injection),
`nntrainer/ddtree/ddtree_sliding.{h,cpp}` (ready core),
`python-example/ddtree.py` (ground truth).

---

## 5. Build / run / verify

See `DDTREE-RESUME.md` and `python-example/verify/README.md`. Key facts: gcc-10,
`-Denable-fp16=false`, `ninja -j4`; app build with `-Denable-app=true
-Denable-transformer=true`; model weights + venv live under `/workspace/qwen3run/`
inside Docker container `my-dev` (NOT in the web/cloud container).
