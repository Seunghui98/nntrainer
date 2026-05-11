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

The converter is exposed as both a CLI (`python -m nntr_causal_lm_converter`)
and a library (`from nntr_causal_lm_converter import convert`). Both produce
the same three artifacts.

#### CLI

```bash
cd tools/causal_lm_converter

# Dump the graph only (INI + runtime config, no weights). Fast; useful when
# debugging the graph layout against the C++ side without paying the weight
# materialization cost.
python -m nntr_causal_lm_converter \
  --model /path/to/Qwen3-0.6B \
  --output ./out \
  --model-name qwen3-0.6b \
  --init-seq-len 64 --max-seq-len 128

# Full conversion: graph + weights + runtime config. Requires torch +
# transformers (used only to load the HF state_dict; nothing about the
# emitted artifact depends on torch at runtime).
python -m nntr_causal_lm_converter \
  --model /path/to/Qwen3-0.6B \
  --output ./out \
  --model-name qwen3-0.6b \
  --weights --dtype float32 \
  --init-seq-len 128 --max-seq-len 1024
```

CLI flags:

| Flag | Default | Purpose |
|---|---|---|
| `--model` | required | Path to a local HF model directory (must contain `config.json`; `model.safetensors` or `pytorch_model.bin` if `--weights`). HuggingFace Hub IDs work too if the model is cached locally. |
| `--output` | required | Directory to write artifacts into (created if missing). |
| `--model-name` | basename of `--model` | Stem used for `<name>.ini` / `<name>.safetensors`. |
| `--init-seq-len` | 8 | Sequence length compiled into the graph (`input_shape = 1:1:N`). Pick the longest prompt you'll feed to `model->incremental_inference`. |
| `--max-seq-len` | 8 | Total tokens (init + generation). Drives `mha_core`'s `max_timestep` and the runtime config. |
| `--weights` | off | Also load the HF `state_dict` and write `<name>.safetensors`. |
| `--dtype` | `float32` | Weight dtype written to the safetensors blob. `float16` halves the file size; the runner picks the matching tensor type from `nntr_config.json`. |

#### Programmatic API

```python
import torch, json
from transformers import AutoModelForCausalLM
from nntr_causal_lm_converter import convert

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",
                                             torch_dtype=torch.float32).eval()
hf_cfg = json.load(open("/path/to/Qwen3-0.6B/config.json"))

paths = convert(
    hf_config=hf_cfg,
    output_dir="./out",
    model_name="qwen3-0.6b",
    init_seq_len=128,
    max_seq_len=1024,
    state_dict=model.state_dict(),
    dtype="float32",
)
print(paths)
# {'ini': './out/qwen3-0.6b.ini',
#  'safetensors': './out/qwen3-0.6b.safetensors',
#  'runtime_config': './out/nntr_config.json'}
```

The library entry point is useful when you want to:
* drive conversion from a notebook or test harness,
* feed a synthesized `state_dict` (e.g. quantized weights, randomly
  initialized stand-ins for performance benchmarking),
* fan out conversions for multiple seq-len configurations from one
  `state_dict` load.

#### What happens during conversion

The converter is a four-stage pipeline; each stage maps to one Python module.

```
HF config + state_dict
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: dispatch                                              │
│    architectures.get_builder(hf_config["model_type"])           │
│    → Qwen3Builder, ...                                          │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: emit graph                                            │
│    builder.build_sections() -> list[IniSection]                 │
│    → ordered (Model, input0, embedding0, layer{0..N-1}_*,       │
│       output_norm, lm_head)                                     │
│    ini.render_ini(...)  -> "<model>.ini"                        │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: bind + materialize weights                            │
│    builder.build_weight_bindings() -> list[WeightBinding]       │
│    → e.g. ("model.layers.0.self_attn.q_proj.weight",            │
│            "layer0_wq:weight", transpose=True)                  │
│    weights.materialize(bindings, state_dict, dtype)             │
│    → transpose FC weights, dtype-cast, sanity-check coverage    │
│    weights.write_safetensors(...) -> "<model>.safetensors"      │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 4: runtime config                                        │
│    RuntimeConfig(...).to_json() -> "nntr_config.json"           │
└─────────────────────────────────────────────────────────────────┘
```

Concretely, on a tiny Qwen3 (4 layers, hidden=128, vocab=512) the run
looks like:

```
$ python -m nntr_causal_lm_converter \
      --model ./hf_model --output ./out --model-name qwen3 \
      --weights --init-seq-len 32 --max-seq-len 64
  ini              -> ./out/qwen3.ini
  safetensors      -> ./out/qwen3.safetensors
  runtime_config   -> ./out/nntr_config.json

$ ls -la ./out/
-rw-r--r-- 1 user user      327 nntr_config.json
-rw-r--r-- 1 user user    7,077 qwen3.ini
-rw-r--r-- 1 user user 2,499,536 qwen3.safetensors
```

##### Layer-name conventions used by Stage 2

These are not configurable: they match the existing
`Applications/CausalLM/models/qwen3` C++ implementation so weight files are
interchangeable.

| Section name | INI `Type` | Notes |
|---|---|---|
| `Model` | `NeuralNetwork` | mandatory; runtime overrides batch/dtype via `setProperty`. |
| `input0` | `input` | shape `1:1:<init_seq_len>`. |
| `embedding0` | `embedding_layer` or `tie_word_embeddings` | switches based on `tie_word_embeddings`. |
| `layer{i}_attention_norm` | `rms_norm` | pre-attn norm. |
| `layer{i}_w{q,k,v}` | `fully_connected` | bias-less; `unit = head_dim * n_heads` (Q) or `head_dim * n_kv_heads` (K/V). |
| `layer{i}_{q,k}_norm` | `reshaped_rms_norm` | Qwen3 per-head Q/K norm. |
| `layer{i}_attention` | `mha_core` | inputs in `q,k,v` order; carries RoPE / GQA / max_timestep. |
| `layer{i}_attention_out` | `fully_connected` | O projection; bias-less. |
| `layer{i}_residual_attn` | `addition` | block input + attention out. |
| `layer{i}_ffn_norm` | `rms_norm` | pre-MLP norm. |
| `layer{i}_ffn_{up,gate}` | `fully_connected` | bias-less. |
| `layer{i}_ffn_swiglu` | `swiglu` | takes `(up, gate)`. |
| `layer{i}_ffn_down` | `fully_connected` | bias-less. |
| `layer{i}_residual_ffn` | `addition` | post-attn residual + ffn down. |
| `output_norm` | `rms_norm` | final norm. |
| `lm_head` | `fully_connected` or `tie_word_embeddings` | tied variant carries `unit = vocab` and `shared_from = embedding0`. |

##### Weight transformations applied by Stage 3

| HF tensor | NNTrainer key | Transform |
|---|---|---|
| `model.embed_tokens.weight` | `embedding0:Embedding` | none (`[vocab, hidden]` matches) |
| `model.layers.{i}.input_layernorm.weight` | `layer{i}_attention_norm:gamma` | none |
| `model.layers.{i}.post_attention_layernorm.weight` | `layer{i}_ffn_norm:gamma` | none |
| `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` | `layer{i}_w{q,k,v}:weight`, `layer{i}_attention_out:weight` | **transpose** `[out, in] → [in, out]` |
| `model.layers.{i}.self_attn.{q,k}_norm.weight` | `layer{i}_{q,k}_norm:gamma` | none |
| `model.layers.{i}.mlp.{up,gate,down}_proj.weight` | `layer{i}_ffn_{up,gate,down}:weight` | **transpose** |
| `model.norm.weight` | `output_norm:gamma` | none |
| `lm_head.weight` (untied only) | `lm_head:weight` | **transpose** |

Tied embeddings: when `tie_word_embeddings` is true, the converter omits a
binding for `lm_head` — the runtime resolves the `shared_from = embedding0`
property and reuses the embedding weight buffer.

#### Verifying conversion output

Two quick checks before running the model:

```bash
# 1. Inspect the graph (no model construction, no torch).
python -m nntr_causal_lm_converter.inspect ini out/qwen3-0.6b.ini --filter layer0_

# 2. Inspect weight coverage. The number of tensors should equal:
#      1 (embed) + N_layers * 11 (per-block tensors) + 1 (output_norm)
#                + (1 if untied else 0)
python -m nntr_causal_lm_converter.inspect safetensors out/qwen3-0.6b.safetensors
```

The smoke-load can be done without a forward pass:

```bash
build/tools/causal_lm_converter/runner/causal_lm_runner \
    out/qwen3-0.6b.ini out/qwen3-0.6b.safetensors out/nntr_config.json \
    --no-forward
# [runner] OK (no forward)
```

If `--no-forward` succeeds but the full run fails, the failure is in the
forward path (vocab / shape / NaN) rather than the load path — narrowing
debugging significantly.

#### Common conversion errors

| Error | Cause | Fix |
|---|---|---|
| `KeyError: 5 HF tensors missing from state_dict` | Loading from HF Hub without `--weights` and then asking the runner for safetensors, or HF model uses an unexpected key (rare for Qwen3). | Re-run with `--weights`; check the inspector output for the missing tensors. |
| `No builder registered for model_type='...'` | Architecture not yet supported. | Add an `ArchitectureBuilder` under `architectures/` (see "Extending" below). |
| `hf_config missing required key 'num_hidden_layers'` | Local `config.json` is from a non-causal-LM checkpoint or has been edited. | Confirm the model is a CausalLM and `model_type` is supported. |
| Empty / 8-byte safetensors | Incomplete write (out-of-disk during conversion). | Re-run conversion; the file is only flushed when complete. |

### Run end-to-end

`causal_lm_runner` is a small CLI for build-and-run validation. It does
**not** include a tokenizer — that lives next to your model
(`tokenizer.json`) and is consumed by either the bundled Python helper or
your own application.

#### Smoke test (no tokenizer, no real input)

```bash
build/tools/causal_lm_converter/runner/causal_lm_runner \
    out/Qwen3-0.6B.ini \
    out/Qwen3-0.6B.safetensors \
    out/nntr_config.json
```

The runner feeds **all-zero token IDs** (always within vocab) and prints the
first 8 logits of position 0 so you can verify the graph is wired correctly:

```
[runner] no --input-tokens; feeding all-zeros.
[runner] forward OK, first 8 logits (position 0): 0.36 1.58 0.53 ...
[runner] OK
```

`--no-forward` validates the load path without running forward:

```bash
.../causal_lm_runner out/<model>.ini out/<model>.safetensors out/nntr_config.json --no-forward
# [runner] OK (no forward)
```

#### Real input + next-token decoding

Two things happen for "real" inference:

1. **Tokenize the prompt** — turn text into integer token IDs using the HF
   tokenizer that ships with the model. The bundled
   `nntr_causal_lm_converter.tokenize` CLI does exactly this:

   ```bash
   python -m nntr_causal_lm_converter.tokenize encode \
       --model ~/models/Qwen3-0.6B \
       --prompt "Hello, world." \
       --pad-to 128 > /tmp/prompt.ids
   cat /tmp/prompt.ids
   # 9707,11,1879,13,0,0,...,0
   ```

   `--pad-to N` right-pads with 0 so the length matches the INI's
   `init_seq_len` (the runner refuses oversized inputs, warns on shorter).

2. **Feed those IDs to the runner and ask for top-K next tokens**:

   ```bash
   ./causal_lm_runner \
       out/qwen3-0.6b.ini \
       out/qwen3-0.6b.safetensors \
       out/nntr_config.json \
       --input-tokens-file /tmp/prompt.ids \
       --top-k 5
   ```

   Output:

   ```
   [runner] fed 128 tokens (first 16: 9707 11 1879 13 0 0 0 0 0 0 0 0 0 0 0 0)
   [runner] forward OK, first 8 logits (position 0): ...
   [runner] top-5 next-token logits (the row generation uses):
     rank 1: token_id=72548  logit=3.05
     rank 2: token_id=15299  logit=2.85
     rank 3: token_id=135729 logit=2.64
     rank 4: token_id=13246  logit=2.53
     rank 5: token_id=16734  logit=2.51
   ```

   `argmax(top-K)` = the predicted next token. Decode it back to text with
   the tokenize CLI:

   ```bash
   python -m nntr_causal_lm_converter.tokenize decode \
       --model ~/models/Qwen3-0.6B --ids 72548
   ```

#### Runner CLI flags

| Flag | Meaning |
|---|---|
| `--no-forward` | Skip the forward pass after loading. Smoke-test the load path only. |
| `--input-tokens "1,2,3,..."` | Comma-separated token IDs to feed into the embedding layer. Truncated to `init_seq_len` (warned), zero-padded if shorter. |
| `--input-tokens-file PATH` | Same as above but read from a file (whitespace- or comma-separated). |
| `--top-k N` | After forward, print the top-N token IDs + logits for the **last** input position — this is the row generation argmaxes / samples to pick the next token. Requires `vocab_size` in `nntr_config.json` (the converter writes it automatically). |
| `--print-shape` | Print the output buffer shape (`batch=1 vocab=151936 …`) for debugging. |

#### What the runner does NOT do

By design:
- **Tokenization** — see the `tokenize` Python helper or use HF tokenizer in
  your own app.
- **Generation loop** — the runner runs *one* forward pass. Looping (with KV
  cache, sampling, repetition penalty, EOS detection) belongs in your
  application; the importer makes that trivial (see next section).
- **Sampling / softmax** — output is raw logits. `argmax` is one line of C++;
  temperature / top-p / top-k sampling is application policy.

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
