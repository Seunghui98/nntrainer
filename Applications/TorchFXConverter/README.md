# TorchFXConverter

Converts HuggingFace PyTorch models to NNTrainer C++ code using `torch.fx` graph tracing.

## Overview

The converter traces a PyTorch model's forward pass, maps the resulting FX graph nodes to NNTrainer layer definitions, detects high-level patterns (attention, FFN, transformer blocks), and emits C++ source, INI config, or JSON metadata.

Supported architectures:
- **Decoder-only** (CausalLM): Qwen3, LLaMA, Granite, LFM2
- **Encoder-only**: XLM-RoBERTa (multilingual-e5)
- **Encoder-decoder**: T5, Gemma2 (experimental)
- **Embedding models**: Qwen3, Gemma2, KaLM (base models without LM head)
- **Custom models**: GLiNER2 (NER extractor with span markers)

## Quick Start — Qwen3-0.5B end-to-end example

`nntr_runner` works like `nntr_causallm`: point it at a directory produced by
`converter.py` and it auto-discovers INI, JSON, C++ header, and weight binary,
then loads the model generically (via `loadFromConfig`) and runs an inference
pass — no per-model C++ compilation required.

### 1. Build NNTrainer and nntr_runner (once)

```bash
cd /path/to/nntrainer
meson setup builddir
ninja -C builddir Applications/TorchFXConverter/jni/nntr_runner
```

### 2. Convert Qwen3-0.5B to a single output directory

```bash
cd Applications/TorchFXConverter

python converter.py \
  --model  Qwen/Qwen3-0.5B \
  --output ./out/qwen3_05b/ \
  --format all \
  --weights \
  --dtype  float16
```

Converter output:

```
out/qwen3_05b/
├── qwen3_0.5b.ini       # NNTrainer INI config  ← used by nntr_runner
├── qwen3_0.5b.json      # layer metadata + weight map
├── qwen3_0.5b.h         # auto-generated C++ class header
├── qwen3_0.5b.cpp       # auto-generated C++ class source
└── qwen3_0.5b.bin       # converted weight binary (float16)
```

### 3. Run with `nntr_runner` — just like `nntr_causallm`

```bash
export LD_LIBRARY_PATH=\
../../builddir/nntrainer:\
../../builddir/Applications/CausalLM/layers

RUNNER=../../builddir/Applications/TorchFXConverter/jni/nntr_runner

# Primary form: pass the directory — nntr_runner auto-discovers all files
$RUNNER ./out/qwen3_05b/
```

Expected output:

```
[nntr_runner] C++ class: Qwen3CausalLM
[nntr_runner] Model metadata:
  model_type: qwen3
  arch_type: decoder_only
  hidden_size: 1024
  num_layers: 28
  num_heads: 16
  num_kv_heads: 8
  head_dim: 64
  intermediate_size: 3072
  vocab_size: 151936
  tie_word_emb: true
  total_layers: 314
[nntr_runner] Loading config: ./out/qwen3_05b/qwen3_0.5b.ini
[nntr_runner] Loading weights: ./out/qwen3_05b/qwen3_0.5b.bin
+-----------------------------------------------+
| Model                                         |
+-----------------------------------------------+
...
[nntr_runner] Running dummy inference (zero input)...
[nntr_runner] Inference OK — 1 output tensor(s).
[nntr_runner] Model initialized successfully.
```

You can also pass files explicitly with named flags or by extension:

```bash
# Named flags — any subset works
$RUNNER --ini ./out/qwen3_05b/qwen3_0.5b.ini \
        --json ./out/qwen3_05b/qwen3_0.5b.json \
        --header ./out/qwen3_05b/qwen3_0.5b.h \
        --weights ./out/qwen3_05b/qwen3_0.5b.bin

# Auto-detect by extension (same as directory form but explicit paths)
$RUNNER out/qwen3_05b/qwen3_0.5b.ini \
        out/qwen3_05b/qwen3_0.5b.json \
        out/qwen3_05b/qwen3_0.5b.h \
        out/qwen3_05b/qwen3_0.5b.bin
```

### 4. One-liner via Python importer

```python
from importer import NNTrainerImporter

imp = NNTrainerImporter(build_dir="../../builddir", dtype="float16")

# Convert + run in one call
result = imp.convert_and_run(
    "Qwen/Qwen3-0.5B",
    output_dir="./out/qwen3_05b/",
    export_weights=True,
)
assert result.success, result.error
print(result.summary)

# Or run a pre-converted directory directly (like nntr_causallm)
result = imp.run_dir("./out/qwen3_05b/")
assert result.success
```

```bash
# CLI: convert + run
python importer.py --model Qwen/Qwen3-0.5B --output ./out/qwen3_05b/ --weights

# CLI: run an existing directory (skip conversion)
python importer.py --dir ./out/qwen3_05b/
```

---

## Quick Start

### Option A — `nntr_runner` (recommended)

Works like `nntr_causallm`: convert to an output directory, then pass that
directory to `nntr_runner`. It auto-discovers INI, JSON, header, and weights,
loads the model via `loadFromConfig`, and runs a dummy inference pass to
verify the full pipeline — no per-model C++ compilation required.

```bash
# Step 1: build nntr_runner once
cd /path/to/nntrainer
meson setup builddir
ninja -C builddir Applications/TorchFXConverter/jni/nntr_runner

# Step 2: convert to a directory (all formats)
cd Applications/TorchFXConverter
python converter.py --model Qwen/Qwen3-0.5B --output ./out/qwen3_05b/ \
  --format all --weights --dtype float16

# Step 3: run — just like nntr_causallm
LD_LIBRARY_PATH=../../builddir/nntrainer:../../builddir/Applications/CausalLM/layers \
  ../../builddir/Applications/TorchFXConverter/jni/nntr_runner ./out/qwen3_05b/
```

```python
# Python importer
from importer import NNTrainerImporter
imp = NNTrainerImporter(build_dir="../../builddir", dtype="float16")
result = imp.convert_and_run("Qwen/Qwen3-0.5B", "./out/qwen3_05b/", export_weights=True)
assert result.success
```

```bash
# importer CLI
python importer.py --model Qwen/Qwen3-0.5B --output ./out/qwen3_05b/ --weights
python importer.py --dir ./out/qwen3_05b/   # run pre-converted directory
```

### Option B — C++ class mode

Generates a full C++ model class (requires per-model meson build):

```bash
python converter.py --model Qwen/Qwen3-0.5B --output ./out/ --format cpp
cp out/qwen3_0.5b.cpp out/qwen3_0.5b.h jni/
meson setup --reconfigure ../../builddir
ninja -C ../../builddir Applications/TorchFXConverter/jni/converter_qwen3_gen_test
```

### converter.py CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | HuggingFace model ID or local path (required) | — |
| `--output` | Output directory (required) | — |
| `--format` | Output format: `cpp`, `ini`, `json`, `all` | `all` |
| `--weights` | Convert and save model weights | off |
| `--dtype` | Weight dtype: `float32`, `float16` | `float32` |
| `--batch-size` | Batch size for INI config | `1` |
| `--seq-len` | Sequence length for tracing | `8` |
| `--model-name` | Override output file basename | derived from `--model` |
| `--quiet` | Suppress progress output | off |

### nntr_runner usage

```
# Primary form — pass the converter output directory (like nntr_causallm)
nntr_runner <output_dir/>

# Named flags (any subset)
nntr_runner --ini model.ini [--json model.json] [--header model.h] [--weights model.bin]

# Auto-detect by extension (pass any files in any order)
nntr_runner model.ini model.json model.h model.bin

# Backward-compatible positional
nntr_runner model.ini [model.bin]
```

| Flag | Extension | Role |
|------|-----------|------|
| `--ini` | `.ini` | NNTrainer INI config — used for model loading (**required**) |
| `--json` | `.json` | Converter metadata — printed (model_type, hidden_size, …) |
| `--header` | `.h` / `.hpp` | Auto-generated C++ header — class name printed |
| `--weights` | `.bin` / `.safetensors` | Weight binary — loaded after initialization |

`nntr_runner` registers the following custom layer types automatically:
`rms_norm`, `reshaped_rms_norm`, `mha_core`, `swiglu`, `embedding_layer`,
`tie_word_embeddings`.

### Weight converter

The weight converter (`weight_converter.py`) converts HuggingFace state dicts
to NNTrainer's binary format.  Two output formats are supported:

| Format | Extension | Description |
|--------|-----------|-------------|
| Binary | `.bin` | Raw tensor bytes in layer-creation order (legacy) |
| Safetensors | `.safetensors` | Self-describing format with JSON header; order-independent |

```python
from weight_converter import WeightConverter

# After converting a model with converter.py:
from decomposer import AdaptiveConverter
from transformers import AutoConfig, AutoModelForCausalLM
import torch

config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
model  = AutoModelForCausalLM.from_config(config)
conv   = AdaptiveConverter(model, seq_len=8)
layers, structure = conv.convert()

wc = WeightConverter(layers)
wc.convert(model.state_dict(), "qwen3.bin", dtype="float32")
wc.convert(model.state_dict(), "qwen3.safetensors",
           output_format="safetensors", dtype="float16")
wc.summary()   # print weight mapping table
```

Standalone weight conversion script:

```python
# Generate a self-contained conversion script
script = wc.generate_script()
with open("convert_weights.py", "w") as f:
    f.write(script)
# Run it later (no NNTrainer dependency needed):
# python convert_weights.py Qwen/Qwen3-0.6B qwen3.bin float16
```

## Tests

### Prerequisites

1. **NNTrainer must be built** with meson before running build-and-run tests:
   ```bash
   # From the nntrainer repo root
   meson setup builddir
   ninja -C builddir
   ```

2. **Python dependencies**: `torch`, `transformers`, `pytest`
   ```bash
   pip install torch transformers pytest
   ```

### Running Tests

All test commands should be run from the `Applications/TorchFXConverter/` directory.

#### Full test suite (all unit tests)

```bash
python -m pytest tests/ -v
```

#### Unit tests only (no NNTrainer build required)

These tests validate the Python converter pipeline (tracing, mapping, pattern detection, code emission) without building or running C++:

```bash
# Tracer tests
python -m pytest tests/test_tracer_simple.py tests/test_tracer_qwen3.py -v

# Node mapper and decomposer tests
python -m pytest tests/test_node_mapper.py tests/test_decomposer.py -v

# Pattern detection tests
python -m pytest tests/test_pattern_detector.py tests/test_pattern_modules.py -v

# Emitter tests (C++, INI, JSON output generation)
python -m pytest tests/test_emitters.py tests/test_emitter_cpp_modules.py tests/test_emitter_ini_modules.py -v

# End-to-end conversion (no build)
python -m pytest tests/test_e2e.py tests/test_multi_arch.py -v
```

#### nntr_runner integration tests (recommended, requires NNTrainer build)

These tests convert each model to INI format and verify it with `nntr_runner` —
no per-model C++ compilation step required:

```bash
# All nntr_runner tests
python -m pytest tests/test_build_and_run.py::TestNNTrainerRunner -v

# Individual model tests
python -m pytest tests/test_build_and_run.py::TestNNTrainerRunner::test_qwen3_tiny -v
python -m pytest tests/test_build_and_run.py::TestNNTrainerRunner::test_llama_tiny -v
python -m pytest tests/test_build_and_run.py::TestNNTrainerRunner::test_multilingual_e5 -v
```

#### C++ build-and-run integration tests (requires NNTrainer build)

These tests run the full pipeline: convert model → generate C++ → build with meson/ninja → run the executable with NNTrainer:

```bash
# All build-and-run tests
python -m pytest tests/test_build_and_run.py::TestConverterBuildAndRun -v

# Individual model tests
python -m pytest tests/test_build_and_run.py::TestConverterBuildAndRun::test_qwen3_tiny_build_and_run -v
python -m pytest tests/test_build_and_run.py::TestConverterBuildAndRun::test_gliner2 -v
python -m pytest tests/test_build_and_run.py::TestConverterBuildAndRun::test_multilingual_e5 -v
```

#### Test models in build-and-run suite

| Test | Architecture | Model Type |
|------|-------------|------------|
| `test_qwen3_tiny_build_and_run` | Decoder-only | Qwen3 CausalLM |
| `test_llama_tiny_build_and_run` | Decoder-only | LLaMA CausalLM |
| `test_qwen3_tied_embeddings` | Decoder-only | Qwen3 (tied embeddings) |
| `test_qwen3_06b` | Decoder-only | Qwen3-0.6B config |
| `test_qwen3_17b` | Decoder-only | Qwen3-1.7B config |
| `test_granite_40` | Decoder-only | GraniteMoeHybrid (dense) |
| `test_lfm_700m` | Decoder-only | LFM2 |
| `test_qwen3_embedding` | Embedding | Qwen3 (no LM head) |
| `test_embedding_gemma` | Embedding | Gemma2 |
| `test_kalm_embedding` | Embedding | KaLM (Qwen2-based) |
| `test_multilingual_e5` | Encoder-only | XLM-RoBERTa |
| `test_gliner2` | Custom | GLiNER2 NER extractor |
| `test_t5gemma2` | Encoder-decoder | T5-Gemma2 (skipped) |
| `test_prebuilt_qwen3_executable` | — | Pre-built binary check |

### How the build-and-run tests work

Each test follows this pipeline:

1. **Create model** — Generates a small synthetic model with random weights (tiny hidden dims, 1-2 layers) matching the target architecture's config
2. **Convert** — Runs `converter.py` to produce `.cpp` and `.h` files
3. **Copy to jni/** — Places generated files in `jni/` where `meson.build` can find them
4. **Meson reconfigure** — Runs `meson setup --reconfigure` to pick up new source files
5. **Ninja build** — Builds the test executable linking against `libnntrainer.so` and custom layer libraries
6. **Run** — Executes the binary, which calls `initialize()` (compile + init the NNTrainer graph) and `summarize()` to verify the model works end-to-end

Generated files are cleaned up after each test. The `jni/meson.build` conditionally includes source files only if they exist via `fs.exists()`.

## Project Structure

```
TorchFXConverter/
├── converter.py              # CLI entry point
├── importer.py               # Python importer: convert + run with nntr_runner
├── custom_models.py          # Custom model loaders (GLiNER2, etc.)
├── tracer.py                 # torch.fx callback tracer
├── decomposer.py             # Multi-pass conversion pipeline
├── node_mapper.py            # FX node → NNTrainer layer mapping
├── module_mapper.py          # nn.Module mapping (Linear, Conv, etc.)
├── function_mapper.py        # torch.* function mapping
├── method_mapper.py          # Tensor.* method mapping
├── op_registry.py            # Operation lookup tables
├── mapper_helpers.py         # Name sanitization, scoping utilities
├── nntrainer_layers.py       # NNTrainer layer type constants
├── pattern_detector.py       # High-level pattern detection
├── weight_converter.py       # Weight format conversion (.bin / .safetensors)
├── emitter_cpp/              # C++ code generation
│   ├── __init__.py           # emit_cpp(), emit_cpp_header(), emit_cpp_source()
│   ├── header.py             # Class header generation
│   ├── source_construct.py   # constructModel() for flat & transformer models
│   ├── source_attention.py   # Attention block generation
│   ├── source_ffn.py         # FFN/MLP block generation
│   ├── source_block.py       # Transformer block assembly
│   ├── source_custom.py      # Custom layer registration
│   └── helpers.py            # C++ generation utilities
├── emitter_ini/              # INI config generation
├── emitter_json.py           # JSON metadata generation
├── patterns/                 # Pattern detection modules
├── jni/                      # NNTrainer runners and test harness
│   ├── nntr_runner.cpp       # Standalone runner: loads any INI, no recompile
│   ├── main.cpp              # Per-model test driver template (C++ class mode)
│   ├── meson.build           # Build config (nntr_runner + per-model tests)
│   └── test_model.*          # Pre-built Qwen3 reference model
├── tests/                    # Test suite
│   ├── test_build_and_run.py # Integration tests:
│   │                         #   TestNNTrainerRunner    (INI + nntr_runner)
│   │                         #   TestConverterBuildAndRun (C++ compile + run)
│   ├── test_tracer_*.py      # Tracer unit tests
│   ├── test_node_mapper.py   # Mapper unit tests
│   ├── test_decomposer.py    # Decomposer unit tests
│   ├── test_emitters.py      # Emitter unit tests
│   └── ...
└── DESIGN.md                 # Architecture documentation
```
