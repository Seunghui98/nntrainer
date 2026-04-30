# causal_lm_converter

A minimal, extensible converter that turns HuggingFace causal language models
into NNTrainer-native artifacts and runs them from C++ via NNTrainer's stock
`loadFromConfig` + safetensors loader. **No generated `.cpp/.h` files.**
**No per-model meson entries.** Adding a new architecture means writing one
small Python module — the C++ runner stays unchanged.

## What this is (and what it isn't)

This tool was built to address three concrete pain points with the existing
`tools/TorchFXConverter` workflow:

1. **Generated C++ rots fast.** Each new model needs a custom `.cpp/.h` that
   developers end up hand-patching at runtime when shapes or properties drift.
2. **Build system coupling.** Every converted model required a new entry in
   `meson.build` and a `meson reconfigure`/rebuild cycle.
3. **Runtime debugging is awkward.** Mismatches between the generated graph
   and the layer factory are caught only at `compile()`/`initialize()` time,
   often inside an opaque binary.

This tool replaces those with a **two-file artifact** (`<model>.ini` +
`<model>.safetensors`, plus a tiny `nntr_config.json`) and a **single generic
C++ runner**. The INI file is the source of truth for the graph; both humans
and `loadFromConfig` consume it directly. Weights are looked up by
`"<layer_name>:<role>"` keys in safetensors so a partial weight overlay (e.g.
swapping in fine-tuned LoRA shards) is straightforward.

The tool intentionally does **not**:
- Re-implement training loops (that lives in NNTrainer + the CausalLM app).
- Replace `tools/TorchFXConverter` for non-CausalLM use cases.
- Provide tokenization, sampling, or KV-cache management.

## Architecture

```
┌──────────────────┐    Python      ┌──────────────┐                ┌────────────────┐
│  HuggingFace     │  ───────────►  │ Architecture │  ───────────►  │  <model>.ini   │
│  config.json     │                │  Builder     │                │ (graph topo +  │
│  state_dict      │                │  (qwen3, …)  │                │  layer props)  │
└──────────────────┘                └──────────────┘                └────────────────┘
                                            │                              │
                                            │                              │ loadFromConfig()
                                            ▼                              ▼
                                    ┌──────────────┐                ┌────────────────┐
                                    │   weights    │                │  causal_lm_    │
                                    │   binder     │  ───────────►  │  runner (C++)  │
                                    └──────────────┘   safetensors  └────────────────┘
                                                       (name-based)        │
                                                                           │ inference
                                                                           ▼
                                                                     finite logits
```

### Layered responsibility

| Layer | File(s) | Responsibility |
|---|---|---|
| Architecture registry | `nntr_causal_lm_converter/architectures/` | Plug-in point. Each module registers a builder for one HF `model_type`. |
| INI emitter | `nntr_causal_lm_converter/ini.py` | Stable, ordered serialization of `[Section] key = value` blocks. |
| Weight binder | `nntr_causal_lm_converter/weights.py` | HF tensor name → NNTrainer `"layer:role"` mapping, transpose & dtype cast, safetensors write. |
| Runtime config | `nntr_causal_lm_converter/runtime_config.py` | Tiny JSON consumed by the C++ runner (batch / seq lens / dtypes). |
| CLI | `nntr_causal_lm_converter/cli.py` | Glue + `python -m nntr_causal_lm_converter`. |
| Runner | `runner/runner.cpp` | Generic INI + safetensors loader; runs one forward pass for validation. |
| Build | `runner/meson.build` | Single executable target; reuses CausalLM layer sources directly. |

### Data flow

```
1. Python side
   HF config dict ──► ArchitectureBuilder.build_sections()      ──► <model>.ini
   HF state_dict  ──► ArchitectureBuilder.build_weight_bindings()
                  ──► weights.materialize() (transpose + cast)  ──► <model>.safetensors
   <init/max seq, dtype>                                        ──► nntr_config.json

2. C++ side (causal_lm_runner)
   register_custom_layers()                  // mha_core, rms_norm, swiglu,
                                             // reshaped_rms_norm, embedding_layer,
                                             // tie_word_embeddings
   model = createModel(NEURAL_NET)
   model->setProperty(batch / tensor_type)
   model->loadFromConfig(<model>.ini)
   model->compile(INFERENCE)
   model->initialize(INFERENCE)
   model->load(<model>.safetensors, MODEL_FORMAT_SAFETENSORS)
   model->incremental_inference(...)
```

The runner contains **no** model-specific code. Switching models means
switching the INI / safetensors paths.

### Naming conventions

The Python converter and the C++ CausalLM application both use the same layer
names (`embedding0`, `layer{i}_attention_norm`, `layer{i}_wq`, …, `output_norm`,
`lm_head`). NNTrainer addresses safetensors entries by
`"<layer_name>:<role>"`:

| Layer type | Role suffix(es) |
|---|---|
| `embedding_layer` / `tie_word_embeddings` | `:Embedding` |
| `fully_connected` | `:weight`, `:bias` |
| `rms_norm` / `reshaped_rms_norm` | `:gamma` |
| `mha_core` | (no learned weights unless `use_sink`) |

FC weights are transposed during conversion (HF stores `[out, in]`,
NNTrainer expects `[in, out]`).

## Quick start

### Prerequisites

* `meson >= 0.55`, `ninja`, a C++17 compiler.
* For the converter: Python 3.8+, plus `numpy`, `safetensors`, and (when
  loading real models) `torch`, `transformers`.
* NNTrainer must be configured with `-Denable-app=true -Denable-transformer=true`
  so the CausalLM custom layers are available.

### Build the runner

```bash
# from the nntrainer repo root
meson setup build -Denable-app=true -Denable-transformer=true
ninja -C build tools/causal_lm_converter/runner/causal_lm_runner
```

The output binary lands at
`build/tools/causal_lm_converter/runner/causal_lm_runner`.

### Convert a model

```bash
cd tools/causal_lm_converter

# Dump the graph only (INI + runtime config, no weights). Fast; useful when
# debugging the graph layout against the C++ side.
python -m nntr_causal_lm_converter \
  --model /path/to/Qwen3-0.6B \
  --output ./out \
  --init-seq-len 64 --max-seq-len 128

# Convert weights too. Requires torch + transformers.
python -m nntr_causal_lm_converter \
  --model /path/to/Qwen3-0.6B \
  --output ./out \
  --weights --dtype float32
```

Outputs:

```
out/
├── Qwen3-0.6B.ini             # graph topology + layer properties
├── Qwen3-0.6B.safetensors     # weights, name-keyed
└── nntr_config.json           # batch / seq / dtype runtime hints
```

### Run end-to-end

```bash
build/tools/causal_lm_converter/runner/causal_lm_runner \
    out/Qwen3-0.6B.ini \
    out/Qwen3-0.6B.safetensors \
    out/nntr_config.json
```

Expected output ends with:

```
[runner] forward OK, first 8 logits: ...
[runner] OK
```

Use `--no-forward` to validate the load path without a forward pass:

```bash
.../causal_lm_runner out/<model>.ini out/<model>.safetensors out/nntr_config.json --no-forward
```

## Embedding the importer in your own application

The runner is just a 100-line CLI on top of a small reusable C++ library
(`runner/causal_lm_importer.{h,cpp}`, target `causal_lm_importer`). Other
applications — a tokenizer-driven generator, a benchmark harness, an
Android JNI bridge — can link against it and skip the CLI entirely.

### Public API

```cpp
// runner/causal_lm_importer.h
namespace causal_lm {

struct RuntimeConfig {
  unsigned int batch_size, init_seq_len, max_seq_len;
  std::string  model_tensor_type;   // e.g. "FP32-FP32"
  std::string  model_file_name;     // hint; usually informational
};

struct ImportOptions {
  std::string ini_path;
  std::string weights_path;
  std::string runtime_config_path;
  ml::train::ExecutionMode execution_mode = ml::train::ExecutionMode::INFERENCE;
  bool skip_weight_load = false;    // for staged weight loading
};

struct ImportedModel {
  std::unique_ptr<ml::train::Model> model;
  RuntimeConfig                     runtime;
};

void register_custom_layers();                   // call once per process
RuntimeConfig parse_runtime_config(const std::string &json_path);
ImportedModel import_model(const ImportOptions &opts);

} // namespace causal_lm
```

### Minimal embedded example

The full file is `runner/example_embedded.cpp`; build target
`causal_lm_example_embedded`. Excerpt:

```cpp
#include "causal_lm_importer.h"

causal_lm::register_custom_layers();              // 1. layer factories

causal_lm::ImportOptions opts;
opts.ini_path            = "qwen3.ini";
opts.weights_path        = "qwen3.safetensors";
opts.runtime_config_path = "nntr_config.json";
auto imported = causal_lm::import_model(opts);     // 2. build + load

std::vector<float> input(imported.runtime.batch_size *
                         imported.runtime.init_seq_len, 0.0f);
std::vector<float *> inputs = {input.data()};
auto outputs = imported.model->incremental_inference(
    imported.runtime.batch_size, inputs, {},
    imported.runtime.init_seq_len, 0, imported.runtime.init_seq_len);  // 3. forward
```

### Linking from your own meson.build

```meson
# in your application's meson.build, after adding nntrainer as a dependency:
my_app = executable(
  'my_app',
  ['main.cpp'],
  dependencies: [
    causal_lm_importer_dep,         # exposed by tools/causal_lm_converter/runner
    nntrainer_dep, nntrainer_ccapi_dep, openmp_dep,
  ],
)
```

The importer library is a static `libcausal_lm_importer.a` so it folds into
your binary without runtime path concerns; if you build inside the same
project, simply depend on `causal_lm_importer_dep`.

### Staged weight loading

Pass `opts.skip_weight_load = true` if you want to defer the safetensors
load (for example, to overlay LoRA adapters before the main weights):

```cpp
opts.skip_weight_load = true;
auto imp = causal_lm::import_model(opts);
imp.model->load(base_weights_path,
                ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS);
imp.model->load(adapter_weights_path,
                ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS);
```

NNTrainer's safetensors loader resolves each weight by its layer-keyed name
(see below), so a partial overlay only touches the matching layers.

## Safetensors usage in detail

### Naming convention

NNTrainer addresses tensors inside a safetensors file by
`"<layer_name>:<role>"`, where the role is decided by the C++ layer
implementation. The Python converter and the C++ runtime agree on the
following table:

| Layer type (INI `Type`) | Role suffix(es) | HF source (Qwen3 example) |
|---|---|---|
| `embedding_layer` / `tie_word_embeddings` | `:Embedding` | `model.embed_tokens.weight` |
| `fully_connected` | `:weight` (mandatory), `:bias` (optional) | `q_proj.weight`, `o_proj.weight`, `mlp.up_proj.weight`, … |
| `rms_norm` | `:gamma` | `input_layernorm.weight`, `post_attention_layernorm.weight`, `model.norm.weight` |
| `reshaped_rms_norm` | `:gamma` | `q_norm.weight`, `k_norm.weight` |
| `mha_core` | (no learned weights unless `use_sink`) | — |

So the safetensors entries for a Qwen3 transformer block look like:

```
layer0_attention_norm:gamma
layer0_wq:weight       layer0_wk:weight       layer0_wv:weight
layer0_q_norm:gamma    layer0_k_norm:gamma
layer0_attention_out:weight
layer0_ffn_norm:gamma
layer0_ffn_up:weight   layer0_ffn_gate:weight   layer0_ffn_down:weight
```

### How the runtime matches weights

`model->load(path, MODEL_FORMAT_SAFETENSORS)` (in `nntrainer/models/neuralnet.cpp`):

1. Parses the safetensors JSON header into a `name -> (offset, length)` map.
2. Walks the compiled graph in topological order and for each weight asks
   the map for the weight's full name.
3. **Match found** → uses the offset directly (so weight order is irrelevant).
4. **Match missing** → logs a warning and falls back to sequential offset
   assignment (the legacy `.bin` behavior).

This means:
* Adding extra tensors to the safetensors blob is harmless — the runtime
  simply skips them.
* Renaming a weight in the file (without updating the INI section name) is
  detected as a warning during load and a likely error during forward.
* You can ship multiple safetensors files (base weights + adapters) and
  load them sequentially; the second `load()` overrides any layer whose name
  is present in the second file.

### FC weight transposition

HuggingFace stores `nn.Linear.weight` as `[out_features, in_features]`;
NNTrainer's `fully_connected` expects `[in_features, out_features]`. The
Python converter handles this via `WeightBinding(transpose=True)` for every
FC-style projection (q/k/v/o, mlp up/gate/down, untied lm_head). RMSNorm
gammas, embedding tables, and bias vectors are shape-compatible and not
transposed.

### Inspecting an artifact

```bash
# List every tensor with dtype, shape, and byte size:
python -m nntr_causal_lm_converter.inspect safetensors out/Qwen3-0.6B.safetensors

# Filter to one block:
python -m nntr_causal_lm_converter.inspect safetensors \
    out/Qwen3-0.6B.safetensors --filter layer0_

# TSV for grep / awk / spreadsheet pipelines:
python -m nntr_causal_lm_converter.inspect safetensors \
    out/Qwen3-0.6B.safetensors --tsv > tensors.tsv
```

The same command also inspects INI files:

```bash
python -m nntr_causal_lm_converter.inspect ini out/Qwen3-0.6B.ini
python -m nntr_causal_lm_converter.inspect ini out/Qwen3-0.6B.ini \
    --filter attention_norm
```

The inspector reads files standalone (no model construction, no torch),
so it is safe to run on weights too large to load into memory.

### Common safetensors troubleshooting

| Symptom | Likely cause |
|---|---|
| `[runner] ERROR: input word index is greater than in_dim` | The embedding layer's `in_dim` is smaller than the input token IDs — typically a vocab mismatch between the INI and your tokenizer. Check the embedding section's `in_dim`. |
| `[runner]` finishes with NaN logits | A required weight was missing from the safetensors and silently fell back to sequential assignment, ending up mis-aligned. Run the inspector and diff against the expected layer names. |
| `loadFromConfig failed` | A layer `Type` in the INI is not registered. Make sure `causal_lm::register_custom_layers()` ran *before* `import_model`. |
| Shape mismatch during compile | INI and safetensors disagree on a dimension (often `hidden_size` vs `intermediate_size`). The inspector's `--filter` lets you cross-check the relevant tensors. |

## Tests

```bash
cd tools/causal_lm_converter
python -m pytest tests/ -v
```

Test groups:

| File | Scope | Requires |
|---|---|---|
| `test_ini_emit.py` | INI emitter + Qwen3 graph layout invariants | numpy only |
| `test_weight_mapping.py` | HF → NNTrainer name mapping, transpose/dtype | numpy only |
| `test_runtime_config.py` | `nntr_config.json` schema | std lib only |
| `test_inspect.py` | Safetensors / INI inspector CLI | numpy + safetensors |
| `test_build_and_run.py` | End-to-end: synth model → convert → load → forward via both `causal_lm_runner` and the embedded example | torch + transformers + built binaries |

The build-and-run test auto-discovers the runner under `build/` (override with
`NNTRAINER_BUILD_DIR=...`). It is automatically **skipped** if the runner is
not built, so unit tests can run in isolation.

## Extending: adding a new architecture

1. Create `architectures/<your_arch>.py`.
2. Subclass `ArchitectureBuilder` and implement:
   * `build_sections() -> list[IniSection]` — graph topology.
   * `build_weight_bindings() -> list[WeightBinding]` — HF → NNTrainer weight
     names with transpose flags.
3. Call `register("<hf_model_type>", YourBuilder)` at module import.
4. Add the import in `architectures/__init__.py` so it self-registers.
5. Mirror `tests/test_ini_emit.py` and `tests/test_weight_mapping.py` for
   your architecture.

If your architecture needs a new C++ custom layer, add the source file to
`runner/meson.build` and register the factory in `runner/runner.cpp`. All
existing models will keep working unchanged.

## Supported architectures

| HF `model_type` | Status | Notes |
|---|---|---|
| `qwen3` | ✅ | Reference implementation, includes per-head Q/K reshaped RMSNorm and tied/untied LM head. Verified against synthetic and real Qwen3-0.6B. |

## Design notes

* **Why not generated C++?** A `.cpp` generator amplifies architectural drift:
  every variant produces a divergent file, every divergence accumulates
  per-model branches in the emitter, and developers fix runtime mismatches by
  hand-editing the generated source. INI is text the runtime already parses,
  so the artifact and the loader speak the same language — and you can
  hand-edit the artifact if needed.
* **Why not a brand-new container format?** Inventing a binary format before
  observing real-world failure modes locks in suboptimal choices and
  complicates debugging. INI lets us iterate, and a `.nnt`-style container can
  be packaged from INI later (one extra tool, no replumbing).
* **Why a single runner?** Per-model executables couple the build system to
  the model catalog and double the artifact count. A single runner that takes
  `<ini> <weights> <config>` as arguments scales from 1 to N models for free.
* **Why Python builders over `torch.fx` tracing?** Tracing-based mapping
  (`tools/TorchFXConverter`) needs heuristics to recover semantic structure
  from a flat op graph; those heuristics break on every novel variant. A
  config-driven builder is ~200 lines per architecture and is robust to
  surface-level model changes.

## Layout

```
tools/causal_lm_converter/
├── README.md                       # this file
├── meson.build                     # adds the runner subdir
├── nntr_causal_lm_converter/       # Python package
│   ├── __init__.py
│   ├── __main__.py                 # python -m entrypoint
│   ├── cli.py                      # CLI glue + convert() function
│   ├── ini.py                      # ordered INI section emitter
│   ├── weights.py                  # HF -> NNTrainer name binder + writer
│   ├── runtime_config.py           # nntr_config.json builder
│   └── architectures/              # plug-in registry
│       ├── __init__.py             # registry; importing a module registers it
│       ├── base.py                 # ArchitectureBuilder ABC
│       └── qwen3.py                # Qwen3 reference implementation
├── runner/
│   ├── meson.build                 # single executable, reuses CausalLM layer .cpp
│   └── runner.cpp                  # generic INI + safetensors runner
└── tests/
    ├── conftest.py
    ├── test_ini_emit.py
    ├── test_weight_mapping.py
    ├── test_runtime_config.py
    └── test_build_and_run.py
```

## License

SPDX-License-Identifier: Apache-2.0 (matches the rest of the repository).
