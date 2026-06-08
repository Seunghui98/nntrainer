# DDTree on gemma4-E2B — Completion Record

**Branch:** `add/DDTree`  ·  **Status:** DONE — gemma4 DDTree runs, decode == greedy (lossless
modulo genuine quantization near-ties), and the 32-node tree is element-by-element identical to the
Python `ddtree.py` reference across all blocks.

This document records how DDTree (tree speculative decoding) was made to run on **gemma4-E2B**
(Gemma 3n E2B, Q4_0) in nntrainer, what was changed, and how it is verified.

---

## 1. Result

- **gemma4-E2B (Q4_0) decode**: `runDDTreeDump` (env `NNTR_DDTREE_DUMP`) produces a full
  self-draft DDTree decode. Output is **token-for-token equal to plain greedy** except at genuine
  top-2 near-ties (see §5), where Q4_0 compute noise can flip the argmax — proven numerical, not a bug.
- **Tree-algorithm parity**: feeding the runtime's own per-block draft logits back through the Python
  reference (`python-example/verify/ddtree_ref.py`) reproduces the runtime's 32-node tree
  **element-by-element** (parents, tokens, depths, positions, the full NxN visibility matrix,
  posterior, accepted path, bonus token) — for **every block, at every generation length tested
  (up to 512 tokens / 42 blocks)**.
- **Qwen3-0.6B**: unchanged / lossless (`accept_lengths [16,14,11,9,6,5,7]`), no regression.

---

## 2. The two gemma4 gaps (vs the already-working Qwen3 path)

DDTree already worked on Qwen3 (full-attention). gemma4 needed:

1. **Per-layer sliding/full attention mask** — gemma4 interleaves 28 `sliding_attention` + 7
   `full_attention` layers (`sliding_window=512`). The tree-verify additive mask must be built in
   two variants and selected per layer.
2. **gemma4-specific verify-forward support** — three runtime gaps, each fixed:
   - **step>1 layers**: the DDTree verify forwards the whole tree (`to-from > 1`) in one
     `incremental_forwarding`, but gemma4's `embedding_layer` / `operation_layer` (add/mul/scalar) /
     `multiout_layer` asserted `step == 1`. Extended to process the full `(to-from)` window
     (matching the proven `SharedFullyConnectedLayer` pattern). Qwen3 doesn't use these layers.
   - **SKIP_PREFILL in `runDDTreeDump`**: gemma4 sets `skip_prefill=true` (KV-shared layers).
     The greedy path prefills N-1 tokens and uses the last prompt token as the first block root;
     `runDDTreeDump` prefilled all N and sampled a prefill argmax (→ `<pad>` root on gemma4).
     Now honors SKIP_PREFILL.
   - **skip_prefill misclassifying the verify forward**: the `skip_prefill` optimization treated any
     `step_size > 1` forward as prompt prefill and returned early *before* computing attention — so
     the 20 KV-shared layers skipped attention during the masked tree-verify, corrupting the
     posterior. The verify forward is uniquely identified by `attn_mask_ != nullptr`; excluded that
     case from `is_prefill`.

### Files changed (runtime)
| File | Change |
|---|---|
| `Applications/CausalLM/layers/mha_core.{h,cpp}` | `setGlobalDDTreeVerify(full,sliding,pos)` + per-layer `selectVerifyMask`; verify forward excluded from `skip_prefill` |
| `Applications/CausalLM/models/causal_lm.{h,cpp}` | `runDDTreeDump` builds full+sliding masks, honors SKIP_PREFILL; `ddtreeHasSlidingLayers()`/`ddtreeSlidingWindow()` virtuals; env-gated verification dumps (§4) |
| `Applications/CausalLM/models/gemma4/gemma4_causallm.h` | gemma4 overrides for the sliding virtuals |
| `nntrainer/layers/embedding.cpp`, `nntrainer/layers/multiout_layer.cpp`, `nntrainer/layers/operation_layer.h` | step>1 (multi-token) `incremental_forwarding` |
| `test/unittest/layers/unittest_mha_core_mask.cpp` | per-layer full-vs-sliding mask selection unit test |

> **NOT runtime bugs (debunked):** the gemma4 "garbage output" first observed was a **wrong chat
> template** in `nntr_config.json` `sample_input` (gemma4-E2B uses `<|turn>`/`<turn|>`, not
> `<start_of_turn>`), reproduced on pristine `main` independent of any DDTree change. The
> speculative GGUF-converter / `EmbeddingLayer` Q4_0-dequant / PLE-dtype edits explored during
> debugging were unnecessary and reverted.

---

## 3. Weights (gemma4-E2B)

The runtime loads a quantized weight from `Applications/CausalLM/res/gemma4/gemma4-e2b/`. The
working weight is `nntr_gemma4_q40_embdq6k.bin` (Q4_0 FC, Q6_K embedding). A converter is provided:

- `res/gemma4/gemma4-e2b/gguf_to_nntrainer.py` — converts an **unsloth `gemma-4-E2B-it-Q4_0.gguf`**
  (native `gemma4` GGUF arch) to nntrainer weights, mapping every GGUF tensor to the gemma4 graph
  weight name. `--fp32` dequantizes everything to plain F32 (used for the §5 numerical-noise check;
  ~18.5 GB, vs 2.7 GB for Q4_0). Tensor manifest + quant policy in `res/gemma4/gemma4-e2b/README.md`.

`nntr_config.json` selects the weight + dtypes and the gemma4 `<|turn>` chat template. Memory: Q4_0
peak ~6.5 GB; fp32 peak ~22-25 GB (the per-layer-input embedding table `[262144 x 8960]` alone is
9.4 GB at fp32 — "E2B" = effective-2B compute, ~4.6B stored params).

---

## 4. Verification dumps (env-gated, dormant in production)

`runDDTreeDump` and the greedy path emit verification artifacts only when these env vars are set:

| Env var | Emits | Used by |
|---|---|---|
| `NNTR_DDTREE_DUMP=<p>` | `<p>.blocks.json` (per-block tree: parents/node_tokens/node_depths/posterior/acc/next/draft_am) + `<p>.tokens.json` | all harnesses |
| `NNTR_DDTREE_LOGITS=<p>` | `<p>.<i>.bin` (raw `[HORIZON,vocab]` fp32 draft logits) + `.meta.json`, per block | tree parity |
| `NNTR_DDTREE_NODELOGITS=<p>` | masked-verify node logits for the divergence block | verify-vs-sequential diagnostic |
| `NNTR_GREEDY_IDS=<p>` | authoritative greedy token-id sequence | decode==greedy comparison |
| `NNTR_DDTREE_MAXBLK=<n>` | cap blocks (debug) | — |

These are the data source for the parity harnesses; they are inert unless the env var is present.

---

## 5. Verification & the near-tie analysis

1. **decode == greedy** (`NNTR_GREEDY_IDS` vs `NNTR_DDTREE_DUMP` gen_ids): identical until the first
   genuine near-tie.
2. **tree parity** (`compare_gemma4_tree_all.py`): all blocks, all fields, all `N*N` visibility
   cells identical, at 192 / 256 / 512 tokens (14 / 18 / 42 blocks).
3. **near-tie root cause** (`compare_verify_vs_sequential.py`): at a divergence (e.g. `' light'`
   2214 vs `' **light'` 5213), the two tokens sit within **~0.18 logits in BOTH** the masked-verify
   and the sequential forward — a genuine tie. The verify-vs-sequential `max_abs_diff` is **~1.4** in
   Q4_0 (same noise floor as the lossless blocks → no compaction drift) and collapses to **~0.003**
   in fp32 (~500x). In fp32 the tie resolves to the greedy choice (`accept_length` recovers 3→16).
   **Verdict: the only decode divergence is Q4_0 compute noise on a genuine top-2 tie, not a bug.**

The comparisons are genuine, not hardcoded: the Python side recomputes the tree from the runtime's
raw logits (`ddtree_ref.build_ddtree_tree`); the `--tamper` self-test confirms that corrupting an
nntrainer value yields DIFF and that perturbing the input logits changes the Python tree.

---

## 6. Build & run

```bash
# App build (gcc-13, fp16=false). DeepQ subdir needs jsoncpp+libcurl which CausalLM does not —
# comment it out locally in Applications/meson.build (do NOT commit) to avoid installing those.
meson setup build-app -Denable-app=true -Denable-transformer=true -Denable-fp16=false \
  -Denable-test=true -Denable-tflite-backbone=false -Denable-tflite-interpreter=false
ninja -j4 -C build-app Applications/CausalLM/nntr_causallm

RES=Applications/CausalLM/res/gemma4/gemma4-e2b
# greedy
./build-app/Applications/CausalLM/nntr_causallm $RES
# DDTree decode + parity dumps
NNTR_DDTREE_DUMP=/tmp/g4 NNTR_DDTREE_LOGITS=/tmp/g4.draft ./build-app/Applications/CausalLM/nntr_causallm $RES

# verify (Python venv with torch+numpy+transformers)
python python-example/verify/compare_gemma4_tree_all.py /tmp/g4.draft /tmp/g4.blocks.json
python python-example/verify/visualize_ddtree_parity.py /tmp/g4.blocks.json /tmp/g4.draft \
  --block 2 --model $RES --tamper
```

See `python-example/verify/README.md` for the full script reference and
`docs/superpowers/specs/2026-06-08-ddtree-gemma4-design.md` for the design.
