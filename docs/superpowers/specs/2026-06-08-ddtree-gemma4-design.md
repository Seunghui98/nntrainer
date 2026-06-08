# DDTree on gemma4-E2B (sliding-window runtime + quantized safetensors weights) — Design Spec

- **Date:** 2026-06-08
- **Branch:** `add/DDTree`
- **Repo (work + execution):** `/home/shsh1004/nntrainer` (this host, x86_64 CPU — **NOT** the `my-dev` Docker)
- **Reference implementation (source of truth):** `python-example/ddtree.py` (== `/home/shsh1004/littlesd_inference/ddtree.py`)
- **Builds on:** `2026-06-05-ddtree-nntrainer-design.md` (core spec) + `DDTREE-RESUME.md` / `DDTREE-DEBUG-OUTPUT.md` (runtime handoff)
- **Predecessor result:** DDTree runtime proven on **Qwen3-0.6B**, fp32, CPU: nntrainer full DDTree decode == greedy **69/69 (lossless)**, all stages bit-parity vs Python.

---

## 1. Goal

Run the already-implemented nntrainer DDTree (tree speculative decoding) runtime on **gemma4-E2B**
on x86 CPU, and supply the weights it needs. Two gaps separate gemma4 from the proven Qwen3 path:

1. **Sliding-window attention** — gemma4 interleaves sliding + full attention layers. The DDTree
   tree-verify mask must therefore be built in **two variants** (full / sliding) and selected
   **per layer**. The pure core (`makeSlidingMasks`) is done + unit-tested but **not wired into the
   runtime**. Qwen3-0.6B is pure full-attention, so this never mattered before.
2. **Weights** — there is no `res/gemma4/` directory, no gemma4 weight converter. The runtime loads
   from `Applications/CausalLM/res/<model>/` and requires `config.json` + a `*.safetensors` weight
   file + `nntr_config.json` to be physically present.

When both are filled, the same `CausalLM::runDDTreeDump` runtime that produced Qwen3 parity runs on
gemma4.

### Non-goals
- No new core algorithm (buildTree/compile/follow/compactTail/makeSlidingMasks already done + golden).
- No real external DFlash draft model (gemma4 run uses the proven **self-draft greedy** path, like Qwen3).
- No ARM build (x86_64 only here).
- No multimodal (vision/audio) gemma4 path — text decode only.

---

## 2. Background: gemma4-E2B shape (from `Applications/CausalLM/models/gemma4/config.json`)

`text_config`:
- **35 layers**: `layer_types` = 28 × `sliding_attention` + 7 × `full_attention` (every 5th layer is full).
- **`sliding_window` = 512**.
- `hidden_size` 1536, `head_dim` 256, `global_head_dim` 512, `num_attention_heads` 8,
  `num_key_value_heads` 1 (MQA), `hidden_size_per_layer_input` 256, `intermediate_size` 6144.
- `final_logit_softcapping` 30.0, `vocab_size` 262144, `hidden_activation` gelu_pytorch_tanh.

The model implementation `Applications/CausalLM/models/gemma4/gemma4_causallm.{cpp,h}` already exists and
uses `mha_core` + `KVCacheManager` (so the §7 runtime hooks from the core spec apply). Its weight
registration order is the **authoritative tensor manifest** for the converter (§5).

---

## 3. Architecture overview

Three workstreams, executed A → B → C, with **B's GGUF-source confirmation as a blocking prerequisite**:

| WS | Title | Nature | Output |
|----|-------|--------|--------|
| A | Sliding-window runtime wiring | nntrainer runtime (mha_core + runDDTreeDump) | per-layer full/sliding mask selection live |
| B | gemma4-E2B weights | converters + res dir | `res/gemma4/gemma4-e2b/` runnable (fp32 + quantized, both safetensors) |
| C | Verification | parity harness + decode | gemma4 decode == greedy, stage parity vs Python |

Responsibility boundary is unchanged from the core spec: the DDTree core stays stateless/cache-agnostic;
the runtime owns the KV cache and the per-layer mask choice.

---

## 4. Workstream A — Sliding-window runtime wiring

**Reference:** `ddtree.py::prepare_ddtree_attention_mask_for_target` (219–255) and
`should_rebuild_ddtree_target_cache` (257–262). **Open-task notes:** `DDTREE-DEBUG-OUTPUT.md` §4.

### 4.1 Current state (anchors)
- `CausalLM::runDDTreeDump` (`causal_lm.cpp:341`): builds tree → `compile()` (`:436`) →
  `MHACoreLayer::setGlobalDDTreeVerify(&maskT,&posT)` (`:449`) → verify forward → argmax →
  `followVerified` → `kv_cache.compactTail(pos, acc.indices)` (`:475`) → reset setter (`:459`).
- `MHACoreLayer::setGlobalDDTreeVerify(mask,pos)` is a **static** setter (`mha_core.h:253`); the layer
  reads `s_verify_mask_` into `attn_mask_` (`mha_core.cpp:295–297`) and applies it in
  `add_mask_and_softmax_full` (`mha_core.cpp:828`, impl `:1262`). Today there is **one** global mask.
- `mha_core.h:75` already defines a `"sliding_window"` property key for the layer.

### 4.2 Changes (chosen approach: **Option A — extended global setter + per-layer selection**)

1. **Config plumbing.** Read `layer_types` + `sliding_window` from gemma4 config into the runtime
   (CausalLM model config). Derive `hasSlidingLayers` (any `sliding_attention` present) and the
   per-layer boolean `isSliding[layerIdx]`. Port `should_rebuild_ddtree_target_cache` as the gate
   (`ddtree.py:257–262`). For Qwen3 (`hasSlidingLayers == false`) the path is a no-op → unchanged.
2. **Build both masks.** In `runDDTreeDump`, after `compile()` produces the full additive mask, call
   `ddtree::makeSlidingMasks(full, verifyPos, currentLength, past+currentLength, slidingWindow,
   hasSliding, cfg, slidingBuf)` → `{full, sliding}` (contiguous `[currentLength, kvLength]` buffers).
3. **Per-layer mask selection in MHA.** Extend the global setter to carry **both** masks plus a way
   for each layer to choose:
   - `setGlobalDDTreeVerify(fullMask, slidingMask, pos)` (overload / extended signature; keep the
     single-mask form working so the Qwen3 path is untouched, or pass `slidingMask == fullMask`).
   - Each `MHACoreLayer` already knows whether it is a sliding layer (via the `"sliding_window"`
     property / a per-layer `isSliding` flag set at construction from `layer_types`). In
     `mha_core.cpp:295`, pick `attn_mask_ = isSliding ? s_verify_sliding_ : s_verify_full_`.
   - This mirrors HF's per-`layer_types` selection and keeps the existing global-setter pattern.
4. **Lifetime.** Both mask tensors live for the verify forward and are cleared together at `:459`.

### 4.3 Why not the alternatives
- **Set mask per layer from the runtime loop:** the forward is graph-driven; the runtime cannot
  cleanly interleave a per-layer setter call between layer forwards. Rejected.
- **Single combined mask:** full and sliding are genuinely different masks; cannot be unified. Rejected.

### 4.4 Acceptance (A)
- Qwen3-0.6B DDTree decode still == greedy 69/69 (no regression on the full-attention path).
- gemma4 verify node logits match HF reference (per-node), with sliding layers masked correctly.

---

## 5. Workstream B — gemma4-E2B weights (complete runnable `res/` dir)

**Runtime requirement (user constraint):** to run on x86 CPU the directory
`Applications/CausalLM/res/gemma4/gemma4-e2b/` must contain `config.json`, a `*.safetensors`
weight file, and `nntr_config.json` (plus greedy `generation_config.json` and tokenizer). Both the
fp32 and the quantized weights are delivered **as nntrainer safetensors** (the runtime loads
safetensors; `nntr_config.json` `model_file_name` + `model_tensor_type` select which).

### 5.0 Feasibility (confirmed)
`nntrainer/utils/safetensors_util.cpp` maps quantized dtypes to safetensors labels:
`QINT4→"I4"`, `QINT8→"I8"`, `QINT16→"I16"`, `UINT4→"U4"`, `UINT8/16/32`. So quantized weights can be
stored in nntrainer safetensors. **Open verification (R2):** confirm the safetensors **loader**
reconstructs the exact packed layout nntrainer's quantized FC/embedding kernels expect (e.g. Q4_0
block-scale + `q4_0x8` x86 repack), i.e. that an `"I4"` safetensors tensor == the runtime's expected
quantized layout. Confirm against `gemma4_causallm.cpp` weight loading + the safetensors load path
before finalizing the quantized converter's byte layout.

### 5.1 B1 — fp32 path first (parity baseline; chosen Option 1 ordering)
- `Applications/CausalLM/res/gemma4/gemma4-e2b/weight_converter.py`: HF safetensors → nntrainer
  **fp32 safetensors**. Adapt from `res/gemma3/weight_converter.py` (closest existing gemma), handling
  gemma4 specifics: per-layer-input hidden (256), dual head_dim (256 / global 512), MQA (kv_heads=1),
  `final_logit_softcapping` 30, gemma embedding scale (×√hidden_size), dual norms, tied embeddings.
- Tensor names/order are taken from **`gemma4_causallm.cpp`** (authoritative manifest), cross-checked
  against the HF gemma4 `state_dict`.
- Produce `nntr_config.json` (`model_tensor_type: "FP32-FP32"`, `model_file_name:
  nntr_gemma4_e2b_fp32.safetensors`), greedy `generation_config.json`, and tokenizer.
- **Purpose:** establish the sliding-wiring + decode==greedy parity baseline in fp32 (bit-exact to
  Python greedy), exactly as Qwen3 was validated.
- `torch`/`transformers` required → a Python venv setup step (this host has Python 3.12, no torch).

### 5.2 B2 — quantized path (HF GGUF → nntrainer safetensors)
- `Applications/CausalLM/res/gemma4/gemma4-e2b/gguf_to_nntrainer.py`: read a HuggingFace **quantized
  GGUF** of gemma4-E2B and write nntrainer **safetensors** with quantized dtype labels (I4/I8/…),
  matching nntrainer's expected packed layout. Adapt the GGUF reader + quant repack logic from
  `res/qwen3/qwen3-0.6b/gguf_to_nntrainer.py` (numpy-only, no torch — fits this host), but **emit
  safetensors instead of the legacy `.bin`**, and map gemma4 GGUF tensor names → the
  `gemma4_causallm.cpp` manifest.
- Produce a matching quantized `nntr_config.json` (`model_tensor_type` reflecting the quant scheme,
  `model_file_name: nntr_gemma4_e2b_<quant>.safetensors`).
- **Purpose:** the actual quantized x86 DDTree run.

### 5.3 Acceptance (B)
- `res/gemma4/gemma4-e2b/` loads in the CausalLM runtime (plain greedy generate succeeds) for both
  the fp32 and the quantized config — no missing-tensor / shape errors.

---

## 6. Workstream C — Verification (same shape as Qwen3)

> **Verification integrity (non-negotiable).** Tests must **never** be made to pass artificially
> (no loosened tolerances to hide a mismatch, no skipping the comparison, no asserting on
> self-produced values). The bar is: run the **Python `ddtree.py` (32-tree, budget 31)** and the
> **nntrainer runtime** on the *same* input, capture both, and compare the **actual values**
> element-by-element — tree structure, masks (full + sliding), per-node verify logits, posterior,
> accepted path, and the final token sequence must be **identical** (fp32 path: bit-exact; quantized
> path: exact where the algorithm is deterministic, with quantization error isolated and reported, not
> masked). A passing test means the two outputs genuinely match. **Build success and unit-test passing
> are the baseline, not the goal** — the goal is demonstrated Python↔nntrainer equality.

1. **Stage parity vs Python.** Run `python-example/ddtree.py` with gemma4 (ground truth) and compare,
   per decode block: tree (parents/tokens/depths), full+sliding masks, verify node logits, accepted
   path, and final token sequence. Reuse `python-example/verify/` harnesses + `render_blocks.py`.
2. **Runtime decode parity.** `NNTR_DDTREE_DUMP=<path> nntr_causallm <gemma4>` → DDTree decode ==
   greedy (target: lossless, as Qwen3's 69/69), in fp32 first (B1), then characterize the quantized
   run (B2; quantization error means quantized ≠ bit-exact-greedy — report accept-length + token
   agreement instead).
3. **No-regression.** Re-run the Qwen3-0.6B verify suite to confirm A didn't break the full path.

---

## 7. Build / run prerequisites (this host)

- **System deps (no passwordless sudo → user installs):** `libjsoncpp-dev`, `libopencv-dev`
  (required by the app build; `-Denable-transformer=true` force-builds DeepQ/YOLO which need them).
- **Compiler:** gcc/g++ **13.3** (this host; gcc-10 absent). 13 supports `_Float16`; keep the proven
  `-Denable-fp16=false` (→ KV cache stored as **UINT16 (2 bytes)**; `compactTail` elemSize already
  handles this, commit 8fa141f8).
- **App build:** `meson setup build-app -Denable-app=true -Denable-transformer=true
  -Denable-fp16=false -Denable-test=true -Denable-tflite-backbone=false
  -Denable-tflite-interpreter=false` then `ninja -j4 -C build-app`. (`enable-transformer` default is
  `false` in `meson_options.txt:70` — must be set true for the CausalLM runtime.)
- **Weights access:** gemma4-E2B is gated → HF token + license acceptance; a confirmed quantized
  **GGUF source** repo (B2) and the HF safetensors model (B1).
- **Python:** venv with `torch` (cpu) + `transformers` for B1/C; B2 converter is numpy-only.

---

## 8. Risks & open items

| # | Risk | Mitigation |
|---|------|-----------|
| R1 | gemma4-E2B GGUF source existence/access (Gemma 3n E2B family); GGUF tensor set may not map cleanly to the nntr gemma4 graph (per-layer-input, AltUp/Laurel/MatFormer quirks). | **Blocking pre-step in B2:** confirm GGUF repo, dump its tensor names, build a name→manifest mapping table before writing the converter. |
| R2 | Quantized safetensors load layout: does an `"I4"` safetensors tensor reconstruct the exact Q4_0 packed/repacked layout the runtime kernels need? | Verify the safetensors load path + `gemma4_causallm.cpp` weight loading before fixing B2's byte layout; fall back to fp32-only quantization-deferred if unsupported. |
| R3 | Per-layer mask selection regresses the existing causal / Qwen3 full-attention path. | Keep single-mask setter form working; gate sliding strictly on `hasSlidingLayers`; re-run Qwen3 69/69 suite. |
| R4 | gemma4 numeric quirks (final_logit_softcapping, embedding scale, dual head_dim) cause logit mismatch vs HF. | Cross-check converter + runtime against HF per-node verify logits before trusting decode parity. |
| R5 | App build force-builds DeepQ/YOLO (no per-app skip flag). | Install `libjsoncpp-dev`+`libopencv-dev`; keep pure-core `build/` separate from `build-app`. |

---

## 9. Deliverables

1. This spec (committed on `add/DDTree`).
2. **WS-A:** `mha_core` per-layer full/sliding mask selection + `runDDTreeDump` two-mask build +
   config gate (`layer_types`/`sliding_window`).
3. **WS-B:** `res/gemma4/gemma4-e2b/` with `weight_converter.py` (fp32 safetensors),
   `gguf_to_nntrainer.py` (quantized → safetensors), `nntr_config.json` (fp32 + quantized),
   `generation_config.json`, tokenizer.
4. **WS-C:** gemma4 stage-parity + decode-parity verification artifacts; Qwen3 no-regression re-run.
5. Updated handoff docs (`DDTREE-RESUME.md` / `DDTREE-DEBUG-OUTPUT.md` §4 closed).

---

## 10. Acceptance criteria (all must hold, evidence required)

**Baseline (necessary, not sufficient):**
1. **Build:** `build-app` configures + compiles clean on this host (gcc-13, the §7 flags).
2. **Unit tests:** `unittest_ddtree` and the runtime unittests pass (genuinely — no forced passes).
3. **Weights:** `res/gemma4/gemma4-e2b/` loads in the runtime (fp32 and quantized configs) — plain
   greedy generate runs without missing-tensor/shape errors.

**The real bar — Python↔nntrainer exact equality (per §6 verification integrity):**
4. **Stage equality vs Python (32-tree).** For the same input, `ddtree.py` (budget 31) and the
   nntrainer runtime produce **identical** tree structure, full+sliding masks, per-node verify logits,
   posterior, and accepted path — compared by actual value, element-by-element (fp32: bit-exact).
5. **Decode equality.** gemma4 fp32 DDTree decode == Python greedy, **token-for-token (lossless)**.
   Quantized run: report token agreement + accept length with quantization error isolated, not hidden.
6. **No regression:** Qwen3-0.6B DDTree decode still == greedy 69/69.

No criterion may be satisfied by weakening the comparison. Evidence = captured Python output +
captured nntrainer output + the diff showing they match.
