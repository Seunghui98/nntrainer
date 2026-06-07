# DDTree (tree speculative decoding) — verification suite

Scripts that verify the nntrainer DDTree implementation against the Python
ground truth (`python-example/ddtree.py`) on **Qwen3-0.6B**, end to end.

> All work runs **inside the Docker container `my-dev`** (ubuntu:20.04, root),
> repo at `/workspace/nntrainer` (branch `add/DDTree`), with a Python venv +
> downloaded model + converted weights under **`/workspace/qwen3run/`**.
> If that working dir is missing (fresh machine), run `setup.sh` first.

## What is proven
1. **Tree core** (`ddtree::buildTree/compile/followVerified`) == `ddtree.py`,
   value-for-value: 60/60 synthetic+real cases, node log-prob (logw) fp32-equal.
2. **base qwen3 forward** (nntrainer) == HuggingFace greedy: tokens 64/64.
3. **DDTree verify forward** (Task10 additive mask + per-node RoPE positions) ==
   HF node logits: argmax 32/32.
4. **Full DDTree decode** in nntrainer (`CausalLM::runDDTreeDump`) == greedy:
   tokens 69/69 (LOSSLESS), accept_lengths == Python e2e.

Per block: **16-token window (root + 15 draft horizon) -> 32-node tree**
(budget=31). See `print_tree.py` for an ASCII render.

## Environment / build (key facts)
- Toolchain: **gcc-10** (GCC 9 lacks `_Float16` and `_mm256_loadu2_m128i`).
- **fp16 disabled** (`-Denable-fp16=false`); GCC<12 has no x86 `_Float16`.
  The KV cache is therefore stored as **UINT16** (2-byte fp16 proxy).
- **7.8 GiB RAM -> build with `ninja -j4`** (default OOM-kills cc1plus).
- Pure-core build (unittest_ddtree):
  `CC=gcc-10 CXX=g++-10 meson setup --wipe build -Denable-fp16=false -Denable-test=true -Denable-tflite-backbone=false -Denable-tflite-interpreter=false -Denable-app=false && ninja -j4 -C build`
- App build (CausalLM runtime): `build-app` with `-Denable-app=true -Denable-transformer=true -Denable-fp16=false` (+ minja submodule, libjsoncpp-dev, libopencv-dev).

## Run the verification

### One-shot
    docker exec my-dev bash /workspace/nntrainer/python-example/verify/verify_all.sh

### Headline demo (DDTree==greedy & nntr-tree==py-tree)
    docker exec my-dev /workspace/qwen3run/venv/bin/python <dir>/demo_verify.py

### nntrainer full DDTree decode (runtime) + per-block tree dump
    docker exec my-dev bash -c "NNTR_DDTREE_DUMP=/tmp/x \
      /workspace/nntrainer/build-app/Applications/CausalLM/nntr_causallm \
      /workspace/nntrainer/Applications/CausalLM/res/qwen3/qwen3-0.6b"
    docker exec my-dev cat /tmp/x.blocks.json     # per-block: draft_am, node_depths(32-node tree), acc
    # NNTR_DDTREE_DUMP unset => normal decode (all DDTree code is dormant)

### Python reference (real ddtree.py functions)
    .../venv/bin/python <dir>/e2e_py.py      # full DDTree loop -> e2e/py_e2e.json
    .../venv/bin/python <dir>/print_tree.py  # ASCII 32-node tree for block 0

## Script index
| script | role |
|---|---|
| `rebuild_ref.py` -> `ddtree_ref.py` | extract the 6 ddtree.py functions verbatim (the reference) |
| `capture_logits.py` / `capture_many.py` | capture qwen3 draft logits (1 / 60 cases) |
| `py_tree.py` / `ddtree_parity_harness.cpp` | Python / nntrainerC++ tree dump (same logits) |
| `compare_trees.py` / `run_batch.py` | tree structure parity (single / 60 cases) |
| `ddtree_logw.cpp` / `logw_compare.py` | per-node logw VALUE parity |
| `hf_ref.py` | HF greedy reference tokens |
| `compare_draft.py` | Phase A: runtime draft logits vs Python |
| `compare_verify.py` | Phase C: runtime verify node logits vs HF |
| `compare_cache.py` | KV cache vs HF (debug; found the UINT16 elemSize bug) |
| `e2e_py.py` | Python full DDTree decode loop (self-draft) |
| `demo_verify.py` | headline A/B demonstration |
| `print_tree.py` | ASCII 32-node tree render |
| `verify_all.sh` | one-shot runner |

The C++ harnesses (`ddtree_parity`, `ddtree_logw`) compile WITHOUT libnntrainer
(ddtree*.cpp are pure host), e.g.:
    g++-10 -std=c++17 -O2 -I /workspace/nntrainer/nntrainer/ddtree \
      ddtree_parity_harness.cpp \
      /workspace/nntrainer/nntrainer/ddtree/ddtree{,_compact,_sampling,_sliding}.cpp \
      -o /workspace/qwen3run/ddtree_parity

## nntrainer runtime entry point
`Applications/CausalLM/models/causal_lm.cpp :: CausalLM::runDDTreeDump` — dormant
unless env `NNTR_DDTREE_DUMP` is set; triggered from `CausalLM::run`. Uses
process-global setters `MHACoreLayer::setGlobalDDTreeVerify(mask,pos)` and
`TieWordEmbedding::setGlobalVerifyDump(buf)` (cross-.so `dynamic_cast` is
unreliable — use `getType()` + these statics).
