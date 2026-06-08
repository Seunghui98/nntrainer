# DDTree on gemma4-E2B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the existing nntrainer DDTree runtime on gemma4-E2B (x86 CPU) by wiring per-layer sliding-window masks and supplying gemma4 weights, proven by exact Python↔nntrainer equality.

**Architecture:** Two gaps separate gemma4 from the proven Qwen3 path. (A) gemma4 interleaves sliding/full attention, so the tree-verify additive mask must be built in two variants and selected per layer — each `MHACoreLayer` already carries a `sliding_window` property (512 for sliding, `UINT_MAX` for full), so selection is local. (B) gemma4 has no weights in `res/`; we convert HF safetensors→nntr fp32 safetensors (parity baseline) and HF quantized GGUF→nntr quantized safetensors (the actual run). Verification compares `ddtree.py` (32-tree, budget 31) against the nntrainer runtime element-by-element.

**Tech Stack:** C++17 (nntrainer `Tensor`, mha_core, KVCacheManager, ddtree core), Meson/Ninja (gcc-13, `-Denable-transformer=true -Denable-fp16=false`), GoogleTest, Python (numpy for GGUF; torch+transformers for HF/parity), nntrainer safetensors format.

**Source of truth:** `python-example/ddtree.py`. Spec: `docs/superpowers/specs/2026-06-08-ddtree-gemma4-design.md`.

**Verification integrity (applies to EVERY task):** Never make a test pass artificially — no loosened tolerances to hide a mismatch, no skipped comparisons, no asserting on self-produced values. A passing parity test means captured Python output and captured nntrainer output genuinely match by value. Build + unittest passing is the baseline, not the goal.

---

## File Structure

**Modified — runtime (Workstream A):**
- `Applications/CausalLM/layers/mha_core.h` — extend `setGlobalDDTreeVerify` to carry full+sliding masks; add static `s_verify_sliding_`.
- `Applications/CausalLM/layers/mha_core.cpp` — in `forwarding`, pick full vs sliding mask by the layer's own `sliding_window` property.
- `Applications/CausalLM/models/causal_lm.{h,cpp}` — in `runDDTreeDump`, read `sliding_window`/`layer_types` (via the model), build the sliding mask with `makeSlidingMasks`, pass both masks.

**New — tests (Workstream A & C):**
- `Applications/CausalLM/unittest_mha_core_mask.cpp` (exists) — add a sliding-vs-full selection case.
- `python-example/verify/gemma4_parity.py` — drive `ddtree.py` on gemma4, dump per-block ground truth.
- `python-example/verify/compare_gemma4.py` — element-by-element diff of Python vs nntrainer dumps.

**New — weights (Workstream B):**
- `Applications/CausalLM/res/gemma4/gemma4-e2b/weight_converter.py` — HF safetensors → nntr fp32 safetensors.
- `Applications/CausalLM/res/gemma4/gemma4-e2b/gguf_to_nntrainer.py` — HF quantized GGUF → nntr quantized safetensors.
- `Applications/CausalLM/res/gemma4/gemma4-e2b/nntr_config.json` (fp32) + `nntr_config_q4.json` (quantized).
- `Applications/CausalLM/res/gemma4/gemma4-e2b/generation_config.json` (greedy), `config.json`, tokenizer (copied from HF).
- `Applications/CausalLM/res/gemma4/gemma4-e2b/README.md` — converter usage + tensor map.

**Reference (read, do not modify):**
- `Applications/CausalLM/res/qwen3/qwen3-0.6b/{weight_converter.py,gguf_to_nntrainer.py}` — converter base.
- `Applications/CausalLM/models/gemma4/gemma4_causallm.{cpp,h}` — authoritative weight manifest + `sliding_window` setup (`:550,575`), `isSlidingAttentionLayer` (`:35-36`).
- `nntrainer/ddtree/ddtree_sliding.{h,cpp}` — `makeSlidingMasks` (done, unit-tested).
- `nntrainer/utils/safetensors_util.cpp` — quantized dtype labels (I4/I8/U4…).

---

# Phase 0 — Environment, build, and weights acquisition

### Task 0.1: App build (baseline, no DDTree yet) — NO apt install needed

**Files:** `Applications/meson.build` (local-only edit, do NOT commit).

> **Discovered 2026-06-08:** CausalLM does NOT depend on opencv/jsoncpp/curl. The only configure
> blocker is `Applications/ReinforcementLearning/DeepQ/jni/meson.build:9` (`dependency('jsoncpp')` +
> libcurl). opencv is non-fatal (YOLO does not `dependency()` it). GTest/iniparser resolve via
> `subprojects/`. So instead of installing system deps, skip the DeepQ subdir locally.

- [ ] **Step 1: Skip the DeepQ subdir locally** (do NOT commit this edit). In `Applications/meson.build`
  comment out the DeepQ line:

```meson
  # subdir('ReinforcementLearning/DeepQ/jni')   # needs jsoncpp+libcurl; CausalLM does not
```

- [ ] **Step 2: Configure the app build** (keep the pure-core `build/` as-is)

```bash
cd /home/shsh1004/nntrainer
git submodule update --init Applications/CausalLM/third_party/minja
meson setup build-app -Denable-app=true -Denable-transformer=true \
  -Denable-fp16=false -Denable-test=true \
  -Denable-tflite-backbone=false -Denable-tflite-interpreter=false
```
Expected: configuration succeeds (uses gcc-13). If it exists: add `--reconfigure`.

- [ ] **Step 3: Build the CausalLM runtime + mask unittests**

```bash
ninja -j4 -C build-app Applications/CausalLM/nntr_causallm \
  Applications/CausalLM/unittest_kv_cache_manager \
  Applications/CausalLM/unittest_mha_core_mask
```
Expected: all three targets build, exit 0.

- [ ] **Step 4: Sanity-run the existing mask + kv unittests**

```bash
./build-app/Applications/CausalLM/unittest_mha_core_mask
./build-app/Applications/CausalLM/unittest_kv_cache_manager
```
Expected: all PASS (baseline before any change).

### Task 0.2: Qwen3-0.6B no-regression baseline capture

**Files:** none (capture artifact).

- [ ] **Step 1: Ensure Qwen3 res dir + weights exist** (per `python-example/verify/setup.sh` steps 3-4). If `Applications/CausalLM/res/qwen3/qwen3-0.6b/nntr_qwen3_0.6b_fp32.safetensors` is missing, run that converter.

- [ ] **Step 2: Capture the proven Qwen3 DDTree decode**

```bash
cd /home/shsh1004/nntrainer
NNTR_DDTREE_DUMP=/tmp/qwen3_baseline ./build-app/Applications/CausalLM/nntr_causallm \
  Applications/CausalLM/res/qwen3/qwen3-0.6b/nntr_config.json
```
Expected: writes `/tmp/qwen3_baseline.tokens.json` + `.blocks.json`. Record the `gen_ids`/accept lengths — this is the 69/69 regression oracle for Task A.5.

### Task 0.3: gemma4-E2B weights download (gated)

**Files:** none (downloads to a scratch dir, e.g. `/home/shsh1004/gemma4run/hf`).

- [ ] **Step 1: HF login (interactive — user runs in the session)**

```bash
! huggingface-cli login   # paste a token with gemma4-E2B license accepted
```

- [ ] **Step 2: Create a venv with torch (cpu) + transformers** (B1/C need it)

```bash
python3 -m venv /home/shsh1004/gemma4run/venv
/home/shsh1004/gemma4run/venv/bin/pip install -q --upgrade pip
/home/shsh1004/gemma4run/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
/home/shsh1004/gemma4run/venv/bin/pip install -q 'transformers>=4.51' numpy accelerate safetensors sentencepiece huggingface_hub gguf
```

- [ ] **Step 3: Download HF safetensors (B1 source)**

```bash
/home/shsh1004/gemma4run/venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='unsloth/gemma-4-E2B-it (base; for tokenizer.json/config) + unsloth/gemma-4-E2B-it-GGUF (Q4_0 gguf)', local_dir='/home/shsh1004/gemma4run/hf/gemma4-e2b')"
```
**Blocking decision:** confirm `unsloth/gemma-4-E2B-it (base; for tokenizer.json/config) + unsloth/gemma-4-E2B-it-GGUF (Q4_0 gguf)` (the exact HF model id matching `models/gemma4/config.json`). Record it in the res README.

- [ ] **Step 4: Confirm a quantized GGUF source (B2) — risk R1**

```bash
/home/shsh1004/gemma4run/venv/bin/python -c "from huggingface_hub import list_repo_files; print('\n'.join(list_repo_files('unsloth/gemma-4-E2B-it-GGUF')))"
```
Record the chosen `.gguf` filename + repo. If no compatible GGUF exists, STOP and report — B2 cannot proceed; B1 (fp32) still can.

---

# Phase A — Sliding-window runtime wiring

### Task A.1: Extend `setGlobalDDTreeVerify` to carry both masks

**Files:**
- Modify: `Applications/CausalLM/layers/mha_core.h:253-257, 408-409`
- Modify: `Applications/CausalLM/layers/mha_core.cpp:41-42`

- [ ] **Step 1: Add the static sliding-mask pointer + extended setter.** In `mha_core.h`, replace the existing setter (lines 253-257):

```cpp
  /** DDTree: set/clear the global tree additive-mask(s) + RoPE positions used
   *  by every mha forward until cleared. `sliding` may be null/equal to `full`
   *  for full-attention-only models (e.g. Qwen3): then every layer uses `full`. */
  static void setGlobalDDTreeVerify(nntrainer::Tensor *full,
                                    nntrainer::Tensor *sliding,
                                    nntrainer::Tensor *pos) {
    s_verify_mask_ = full;
    s_verify_sliding_ = sliding;
    s_verify_pos_ = pos;
  }
```

- [ ] **Step 2: Declare the new static member.** In `mha_core.h` after line 408 (`static nntrainer::Tensor *s_verify_mask_;`) add:

```cpp
  static nntrainer::Tensor *s_verify_sliding_;
```

- [ ] **Step 3: Define the static member.** In `mha_core.cpp` after line 42 (`...*s_verify_pos_ = nullptr;`) add:

```cpp
nntrainer::Tensor *MHACoreLayer::s_verify_sliding_ = nullptr;
```

- [ ] **Step 4: Build to confirm it compiles (callers updated next task).**

```bash
ninja -j4 -C build-app Applications/CausalLM/nntr_causallm 2>&1 | tail -5
```
Expected: FAILS at the old 2-arg call site in `causal_lm.cpp` (fixed in Task A.3). This proves the signature changed.

### Task A.2: Per-layer mask selection in `forwarding`

**Files:**
- Modify: `Applications/CausalLM/layers/mha_core.cpp:295-301`

- [ ] **Step 1: Pick full vs sliding by the layer's own window.** Replace lines 295-301 in `mha_core.cpp`:

```cpp
  if (s_verify_mask_ != nullptr) {
    // gemma-style models: sliding layers (finite local_window_size) use the
    // sliding mask; full-attention layers (UINT_MAX) use the full mask. When
    // s_verify_sliding_ is null (full-attention-only models), always use full.
    const bool layer_is_sliding =
      (s_verify_sliding_ != nullptr) && (local_window_size != UINT_MAX);
    attn_mask_ = layer_is_sliding ? s_verify_sliding_ : s_verify_mask_;
    tree_pos_ = s_verify_pos_;
  } else {
    attn_mask_ = nullptr;
    tree_pos_ = nullptr;
  }
```
**Note:** confirm the member name holding the per-layer window — it is set from `props::SlidingWindow` (mha_core.h:366). If the member is named differently than `local_window_size`, use that member (grep `local_window_size` in mha_core.cpp:49/166 confirms it is the window size; verify it stores `UINT_MAX` for full layers — if it is clamped, instead read `std::get<props::SlidingWindow>(...).get()`).

- [ ] **Step 2: Build (still fails at caller — expected).**

```bash
ninja -j4 -C build-app Applications/CausalLM/nntr_causallm 2>&1 | tail -5
```
Expected: still only the `causal_lm.cpp` 2-arg call-site error.

### Task A.3: Build both masks in `runDDTreeDump`

**Files:**
- Modify: `Applications/CausalLM/models/causal_lm.cpp:436-459`
- Reference: `nntrainer/ddtree/ddtree_sliding.h` (`makeSlidingMasks` signature).

- [ ] **Step 1: After `compile(...)` (causal_lm.cpp ~:436), build the sliding mask.** Insert immediately after the `compile(...)` call and before the `maskT` tensor construction:

```cpp
    // ---- sliding-window variant for gemma-style models (no-op for Qwen3) ----
    // hasSlidingLayers + slidingWindow come from the model config (Task A.4).
    std::vector<float> smask;            // sliding mask buffer (== cmask when no sliding)
    const bool has_sliding = ddtreeHasSlidingLayers();   // Task A.4
    const int sliding_window = ddtreeSlidingWindow();    // Task A.4
    if (has_sliding) {
      smask.resize((size_t)cl * stride);
      nntrainer::ddtree::makeSlidingMasks(
        cmask.data(), vpi.data(), cl, stride, sliding_window,
        /*hasSlidingLayers=*/true, cfg, smask.data());
    }
```

- [ ] **Step 2: Build a sliding mask tensor and pass both to the setter.** Replace the single `setGlobalDDTreeVerify(&maskT, &posT)` call (~:449) with:

```cpp
    nntrainer::Tensor slidingT;
    if (has_sliding) {
      slidingT = nntrainer::Tensor(1, 1, (unsigned int)cl, (unsigned int)stride);
      std::copy(smask.begin(), smask.end(), slidingT.getData<float>());
    }
    causallm::MHACoreLayer::setGlobalDDTreeVerify(
      &maskT, has_sliding ? &slidingT : nullptr, &posT);
```

- [ ] **Step 3: Update the clear call.** Replace `setGlobalDDTreeVerify(nullptr, nullptr)` (~:459) with:

```cpp
    causallm::MHACoreLayer::setGlobalDDTreeVerify(nullptr, nullptr, nullptr);
```

- [ ] **Step 4: Add `#include <ddtree_sliding.h>`** near the other ddtree includes at the top of `causal_lm.cpp` (grep `ddtree.h` to find them).

### Task A.4: Config plumbing — `hasSlidingLayers` / `slidingWindow`

**Files:**
- Modify: `Applications/CausalLM/models/causal_lm.h` (declare two protected helpers)
- Modify: `Applications/CausalLM/models/causal_lm.cpp` (define them, default = no sliding)
- Modify: `Applications/CausalLM/models/gemma4/gemma4_causallm.{h,cpp}` (override for gemma4)

- [ ] **Step 1: Declare base helpers in `causal_lm.h`** (protected section, near other virtuals):

```cpp
  /** DDTree sliding-window support. Base = full-attention only (Qwen3). */
  virtual bool ddtreeHasSlidingLayers() const { return false; }
  virtual int  ddtreeSlidingWindow() const { return 0; }
```

- [ ] **Step 2: gemma4 overrides** in `gemma4_causallm.h` (the `Gemma4CausalLM` / transformer class that owns `layer_types` + `SLIDING_WINDOW`):

```cpp
  bool ddtreeHasSlidingLayers() const override {
    for (const auto &t : layer_types)
      if (t == "sliding_attention") return true;
    return false;
  }
  int ddtreeSlidingWindow() const override {
    return static_cast<int>(SLIDING_WINDOW);
  }
```
**Note:** confirm `layer_types` and `SLIDING_WINDOW` are reachable from the class that defines `runDDTreeDump`'s `this`. `runDDTreeDump` lives on `CausalLM`; the gemma4 object is the concrete `model`. If `runDDTreeDump` cannot virtual-dispatch onto the gemma4 subclass (it is a `CausalLM` method), read the values from the parsed config JSON the model already holds instead (grep how `SLIDING_WINDOW`/`layer_types` are stored — `gemma4_causallm.cpp:114,550`), exposing them via the base virtuals above.

- [ ] **Step 3: Build the runtime.**

```bash
ninja -j4 -C build-app Applications/CausalLM/nntr_causallm 2>&1 | tail -5
```
Expected: builds clean (exit 0) — all call sites updated.

- [ ] **Step 4: Commit Phase A runtime.**

```bash
git add Applications/CausalLM/layers/mha_core.h Applications/CausalLM/layers/mha_core.cpp \
        Applications/CausalLM/models/causal_lm.h Applications/CausalLM/models/causal_lm.cpp \
        Applications/CausalLM/models/gemma4/gemma4_causallm.h Applications/CausalLM/models/gemma4/gemma4_causallm.cpp
git commit -m "feat(causallm/ddtree): per-layer full/sliding verify mask selection for gemma4"
```

### Task A.5: Qwen3 no-regression (full-attention path unchanged)

**Files:** none (verification).

- [ ] **Step 1: Re-run the Qwen3 DDTree decode.**

```bash
NNTR_DDTREE_DUMP=/tmp/qwen3_afterA ./build-app/Applications/CausalLM/nntr_causallm \
  Applications/CausalLM/res/qwen3/qwen3-0.6b/nntr_config.json
```

- [ ] **Step 2: Diff against the Task 0.2 baseline (must be identical).**

```bash
diff /tmp/qwen3_baseline.tokens.json /tmp/qwen3_afterA.tokens.json && echo "QWEN3 UNCHANGED"
```
Expected: no diff, prints `QWEN3 UNCHANGED`. (`has_sliding == false` for Qwen3 → sliding path is a genuine no-op.) If it differs, Phase A regressed the full path — STOP and fix.

### Task A.6: Sliding-vs-full selection unit test

**Files:**
- Modify: `Applications/CausalLM/unittest_mha_core_mask.cpp` (add a TEST before its `main()`).

- [ ] **Step 1: Write a failing test** that drives one full layer and one sliding layer (`local_window_size` finite) through `forwarding` with distinct full/sliding masks and asserts each layer consumes the correct mask. Use the existing test's MHACoreLayer construction helpers; set `sliding_window` via the layer property. Assert the softmax output differs exactly as the two masks dictate (compute the expected masked-softmax by hand for a 2×2 score). Insert BEFORE `main()`.

- [ ] **Step 2: Run to confirm it fails** (before A.1-A.4 it would not compile; after, it must reflect real selection):

```bash
ninja -j4 -C build-app Applications/CausalLM/unittest_mha_core_mask && \
  ./build-app/Applications/CausalLM/unittest_mha_core_mask --gtest_filter='*Sliding*'
```
Expected: the new test runs and PASSES only if selection is correct (do not weaken the assertion to force it).

- [ ] **Step 3: Commit.**

```bash
git add Applications/CausalLM/unittest_mha_core_mask.cpp
git commit -m "test(causallm/ddtree): mha per-layer full vs sliding mask selection"
```

---

# Phase B — gemma4-E2B weights

### Task B.1: Derive the gemma4 tensor manifest (discovery, not a guess)

**Files:**
- Create: `Applications/CausalLM/res/gemma4/gemma4-e2b/README.md` (start with the tensor map).

- [ ] **Step 1: Dump the nntrainer-side expected weight names/order/shapes** from the model graph. Read `models/gemma4/gemma4_causallm.cpp` and list every `createLayer(... name=...)` + its weight (embedding0, per_layer_input_embedding, per_layer_input_projection, per_layer_model_proj_scale, per_layer_projection_norm, attention Q/K/V/O, q_norm/k_norm, mlp gate/up/down, input/post norms, final norm, lm_head/tie). Record the exact nntr tensor names + shapes in the README as column 1.

- [ ] **Step 2: Dump the HF state_dict keys + shapes** (column 2):

```bash
/home/shsh1004/gemma4run/venv/bin/python - <<'PY'
from safetensors import safe_open
import glob, os
d='/home/shsh1004/gemma4run/hf/gemma4-e2b'
for f in sorted(glob.glob(os.path.join(d,'*.safetensors'))):
    with safe_open(f,'pt') as t:
        for k in t.keys(): print(k, list(t.get_slice(k).get_shape()))
PY
```

- [ ] **Step 3: Dump the GGUF tensor names + shapes** (column 3, for B.3):

```bash
/home/shsh1004/gemma4run/venv/bin/python - <<'PY'
from gguf import GGUFReader
r=GGUFReader('/home/shsh1004/gemma4run/gguf/gemma-4-E2B-it-Q4_0.gguf')
for t in r.tensors: print(t.name, list(t.shape), t.tensor_type)
PY
```

- [ ] **Step 4: Write the 3-column mapping table** (nntr name ↔ HF key ↔ GGUF name) + per-tensor transform notes (transpose?, dtype, gemma4 quirks) into the README. **Commit** the README — every later converter task references this table by row.

```bash
git add Applications/CausalLM/res/gemma4/gemma4-e2b/README.md
git commit -m "docs(gemma4): DDTree weight tensor mapping (nntr<->HF<->GGUF)"
```

### Task B.2: fp32 safetensors converter (`weight_converter.py`)

**Files:**
- Create: `Applications/CausalLM/res/gemma4/gemma4-e2b/weight_converter.py` (base: `res/gemma3/weight_converter.py`).
- Create: `Applications/CausalLM/res/gemma4/gemma4-e2b/nntr_config.json`.

- [ ] **Step 1: Copy gemma3's converter as the base and adapt the manifest** to the Task B.1 table. Concrete gemma4-specific transforms to encode (each documented inline):
  - Embedding scale: gemma multiplies token embeddings by `sqrt(hidden_size)` (1536) — match how the nntr graph expects it (check whether the graph applies the scale or the weight must be pre-scaled; `gemma4_causallm.cpp` scalar_multiply layers indicate runtime scaling — do NOT double-scale).
  - Attention: split `head_dim` (256) vs `global_head_dim` (512) per layer using `isSliding(layer_id)`; MQA `num_key_value_heads=1`.
  - `final_logit_softcapping` 30.0: a runtime op, not a weight — converter does nothing, just confirm the graph applies it.
  - Per-layer-input embedding/projection/proj_scale/projection_norm tensors (gemma4-specific) — map from the HF keys found in B.1.
  - Norm weights → F32 (nntr keeps norms in F32); RMSNorm "+1" offset convention: match gemma3 converter's handling exactly.
  - Tied embeddings: write `lm_head` only if not tied (config `tie_word_embeddings`).

- [ ] **Step 2: Run the converter.**

```bash
RES=/home/shsh1004/nntrainer/Applications/CausalLM/res/gemma4/gemma4-e2b
/home/shsh1004/gemma4run/venv/bin/python "$RES/weight_converter.py" \
  --model_path /home/shsh1004/gemma4run/hf/gemma4-e2b \
  --output_name "$RES/nntr_gemma4_e2b_fp32" --safetensors
```
Expected: writes `nntr_gemma4_e2b_fp32.safetensors`.

- [ ] **Step 3: Write `nntr_config.json`** (model the qwen3 one): `model_tensor_type: "FP32-FP32"`, `model_file_name: "nntr_gemma4_e2b_fp32.safetensors"`, `fc_layer_dtype: "FP32"`, `embedding_dtype: "FP32"`, correct `tokenizer_file` path, greedy `sample_input`. Copy `config.json`, `generation_config.json` (greedy: `do_sample:false, temperature:0`), and tokenizer files from the HF dir into the res dir.

- [ ] **Step 4: Load test — plain greedy generate (no DDTree).**

```bash
./build-app/Applications/CausalLM/nntr_causallm "$RES/nntr_config.json"
```
Expected: model loads with no missing-tensor/shape error and emits coherent text. If a tensor is missing/mis-shaped, fix the B.1 mapping — do not silently skip tensors.

- [ ] **Step 5: Commit.**

```bash
git add Applications/CausalLM/res/gemma4/gemma4-e2b/weight_converter.py \
        Applications/CausalLM/res/gemma4/gemma4-e2b/nntr_config.json \
        Applications/CausalLM/res/gemma4/gemma4-e2b/config.json \
        Applications/CausalLM/res/gemma4/gemma4-e2b/generation_config.json
git commit -m "feat(gemma4): fp32 safetensors weight_converter + runnable nntr_config"
```

### Task B.3: quantized GGUF → nntr safetensors converter

**Files:**
- Create: `Applications/CausalLM/res/gemma4/gemma4-e2b/gguf_to_nntrainer.py` (base: qwen3's GGUF reader + quant repack).
- Create: `Applications/CausalLM/res/gemma4/gemma4-e2b/nntr_config_q4.json`.

- [ ] **Step 1: Verify the safetensors quantized-load layout — risk R2.** Before writing bytes, confirm what an `"I4"`/`"I8"` safetensors tensor must contain for the runtime kernels. Read the safetensors load path (grep where `dtypeToString`'s inverse parses `"I4"`) and `gemma4_causallm.cpp` FC weight loading. Record: does the loader expect raw Q4_0 blocks (scale+nibbles) and the `q4_0x8` x86 repack, or a plain int4 array? Write the finding into the README. If quantized safetensors load is NOT supported by the runtime, STOP B.3 and report (fp32 path B.2 still delivers a runnable gemma4).

- [ ] **Step 2: Adapt qwen3's `gguf_to_nntrainer.py`** to: (a) the gemma4 GGUF→nntr name map (B.1 col 3), (b) emit **safetensors** (use `nntrainer::safetensors` header layout: `{"__metadata__":{"format":"nntrainer"}, name:{dtype,shape,data_offsets}}`) instead of the legacy `.bin`, with quantized dtype labels per Step 1, (c) gemma4 quirks (per-layer-input tensors, dual head_dim, MQA, embedding Q6_K).

- [ ] **Step 3: Run the converter.**

```bash
RES=/home/shsh1004/nntrainer/Applications/CausalLM/res/gemma4/gemma4-e2b
/home/shsh1004/gemma4run/venv/bin/python "$RES/gguf_to_nntrainer.py" \
  /home/shsh1004/gemma4run/gguf/gemma-4-E2B-it-Q4_0.gguf \
  -o "$RES/nntr_gemma4_e2b_q4.safetensors" --target x86 --emit-nntr-config
```
Expected: writes `nntr_gemma4_e2b_q4.safetensors` + `nntr_config_q4.json`.

- [ ] **Step 4: Load test — plain greedy generate on the quantized weights.**

```bash
./build-app/Applications/CausalLM/nntr_causallm "$RES/nntr_config_q4.json"
```
Expected: loads + emits coherent text (quantized).

- [ ] **Step 5: Commit.**

```bash
git add Applications/CausalLM/res/gemma4/gemma4-e2b/gguf_to_nntrainer.py \
        Applications/CausalLM/res/gemma4/gemma4-e2b/nntr_config_q4.json \
        Applications/CausalLM/res/gemma4/gemma4-e2b/README.md
git commit -m "feat(gemma4): quantized GGUF->nntr safetensors converter + quant nntr_config"
```

---

# Phase C — Python↔nntrainer exact equality verification

### Task C.1: Python ground-truth dump (gemma4, 32-tree)

**Files:**
- Create: `python-example/verify/gemma4_parity.py`.

- [ ] **Step 1: Write a script** that loads gemma4 via the same HF model used by `ddtree.py`, runs `ddtree_generate` (budget 31 → 32-tree, `max_new_tokens` to match the run, temperature 0/greedy, self-draft to mirror the nntr runtime), and dumps per block: `draft_logits` (or self-draft token rollout), `parents`, `node_tokens` (verify_input_ids), `node_depths`, full mask, sliding mask, per-node verify logits, `posterior`, accepted indices, `next`, and the final `gen_ids`. Write JSON matching the nntr `blocks.json` field names (`DDTREE-DEBUG-OUTPUT.md` §1).

- [ ] **Step 2: Run it.**

```bash
/home/shsh1004/gemma4run/venv/bin/python python-example/verify/gemma4_parity.py \
  --model /home/shsh1004/gemma4run/hf/gemma4-e2b --budget 31 --out /tmp/gemma4_py
```
Expected: writes `/tmp/gemma4_py.blocks.json` + `.tokens.json`.

- [ ] **Step 3: Commit the script.**

```bash
git add python-example/verify/gemma4_parity.py
git commit -m "test(gemma4): python ddtree ground-truth dumper (32-tree)"
```

### Task C.2: nntrainer runtime dump (gemma4, fp32)

**Files:** none (run + capture).

- [ ] **Step 1: Run the nntr DDTree dump on the fp32 gemma4 weights.**

```bash
RES=/home/shsh1004/nntrainer/Applications/CausalLM/res/gemma4/gemma4-e2b
NNTR_DDTREE_DUMP=/tmp/gemma4_nntr ./build-app/Applications/CausalLM/nntr_causallm "$RES/nntr_config.json"
```
Expected: writes `/tmp/gemma4_nntr.blocks.json` + `.tokens.json` with the sliding-aware tree dump.

### Task C.3: Element-by-element comparison (the real bar)

**Files:**
- Create: `python-example/verify/compare_gemma4.py`.

- [ ] **Step 1: Write the comparator** that loads both `blocks.json` and asserts, per block and per element (NO tolerance loosening to hide mismatch; fp32 verify logits compared with a documented, tight fp tolerance only for float rounding, exact for ints):
  - identical `parents`, `node_tokens`, `node_depths` (tree structure exact),
  - identical full mask and sliding mask values,
  - per-node verify logits match (tight fp tolerance; report max abs diff),
  - identical `posterior`, accepted indices, `next`,
  - identical final `gen_ids` (token-for-token).
  It must print the first mismatching block/field/index and exit non-zero on any mismatch.

- [ ] **Step 2: Run the comparison (must pass genuinely).**

```bash
/home/shsh1004/gemma4run/venv/bin/python python-example/verify/compare_gemma4.py \
  /tmp/gemma4_py.blocks.json /tmp/gemma4_nntr.blocks.json
```
Expected: prints `GEMMA4 PARITY OK` and exits 0 — meaning Python (32-tree) and nntrainer produced identical trees/masks/logits/posterior/path/tokens. If it fails, debug the runtime (sliding mask wiring / numeric quirks per R4) until truly equal — do not weaken the comparator.

- [ ] **Step 3: Quantized characterization.** Run C.2 with `nntr_config_q4.json` → `/tmp/gemma4_nntr_q4`, then compare token agreement + accept-length vs Python greedy (quantization error expected; isolate and report it, do not claim bit-parity).

- [ ] **Step 4: Commit.**

```bash
git add python-example/verify/compare_gemma4.py
git commit -m "test(gemma4): element-by-element python<->nntrainer DDTree parity comparator"
```

### Task C.4: Update handoff docs + close the sliding open task

**Files:**
- Modify: `docs/superpowers/plans/DDTREE-DEBUG-OUTPUT.md` (§4 → DONE), `DDTREE-RESUME.md` (status), `docs/superpowers/plans/2026-06-05-ddtree-nntrainer-PROGRESS.md`.

- [ ] **Step 1: Record results** — gemma4 fp32 parity result (block count, accept lengths, max logit diff), quantized characterization, and mark the sliding-window runtime wiring task DONE with the commit hashes.

- [ ] **Step 2: Commit.**

```bash
git add docs/superpowers/plans/DDTREE-DEBUG-OUTPUT.md docs/superpowers/plans/DDTREE-RESUME.md \
        docs/superpowers/plans/2026-06-05-ddtree-nntrainer-PROGRESS.md
git commit -m "docs(ddtree): gemma4 sliding wiring done + python<->nntrainer parity recorded"
```

---

## Acceptance (maps to spec §10)

- **Baseline:** `build-app` builds clean (Task 0.1); `unittest_ddtree` + mask/kv unittests pass (0.1, A.6); both gemma4 configs load + generate (B.2.4, B.3.3).
- **Real bar:** gemma4 fp32 — Python(32-tree) ↔ nntrainer **identical** tree/masks/logits/posterior/path/tokens (C.3.2); quantized characterized (C.3.3); Qwen3 still 69/69 (A.5).
- **No criterion satisfied by weakening a comparison.** Evidence = captured Python dump + captured nntr dump + the passing diff.
