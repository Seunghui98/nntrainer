# ☄️ CausalLM Inference with NNTrainer

- This application provides examples to run causal llm models using nntrainer.
- This example only provides *inference* mode, not *training* mode yet.

## Supported models

- Llama
- Qwen3 (1.7b/4b/7b/14b)
- Qwen3MoE (30b-A3b)
- Gpt-Oss-20b 
- You can try your own model with custom layers! 
- Feel free to contribute! 😊

## How to run

- download and copy the model files from hugingface to `res/{model}` directory.
- The folder should contain
    - config.json
    - generation_config.json
    - tokenizer.json
    - tokenizer_config.json
    - vocab.json
    - nntr_config.json
    - nntrainer weight binfile (matches with the name in nntr_config.json)
    - which are usuallyl included in HF model deployment.
- compile the Application
- If you test CausalLM on your PC, build with `-Denable-transformer=true`
- run the model with the following command

```
$ cd build/Applications/CausalLM
$ ./nntr_causallm {your model config folder}
```

e.g.,

```
$ ./nntr_causallm /tmp/nntrainer/Applications/CausalLM/res/qwen3-4b/
```

### Recommended Configuration 

- PC test
```
$ meson build -Denable-fp16=true -Dthread-backend=omp -Denable-transformer=true -Domp-num-threads=4
$ export OMP_THREAD_LIMIT=16 && export OMP_WAIT_POLICY=active && export OMP_PROC_BIND=true && export OMP_PLACES=cores && export OMP_NUM_THREADS=4
```

- Android test
```
$ ./tools/package_android.sh -Domp-num-threads=4 -Dthread-backend=omp
```

## Supported Models

- Qwen3 (0.6B, 1.7B, 4B, 8B, 14B, 32B) [[link](https://huggingface.co/Qwen/Qwen3-4B)]
- Qwen3-MoE (30B-A3B) [[link](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)]
- GPT-OSS (MoE: 20B, 120B) [[link](https://huggingface.co/openai/gpt-oss-20b)]

For more details, please refer to the [Model Documentation](models/README.md).

---

## nntr_runner — Generic Model Runner for Converter Output

`nntr_runner` is a lightweight standalone binary that loads any model produced
by `tools/TorchFXConverter/converter.py` and runs one inference pass.
Unlike `nntr_causallm`, it does **not** require a per-model C++ class — it
uses NNTrainer's `loadFromConfig()` API to load the generated INI directly.

**Supported architectures:**
- Encoder-only (BERT, XLM-RoBERTa, DistilBERT, …)
- Decoder-only (Qwen3, LLaMA, Granite, …)
- Embedding models (no LM head)

### Build

```bash
# From repo root
meson setup builddir -Denable-transformer=true
ninja -C builddir Applications/CausalLM/nntr_runner
```

### End-to-end example: BERT encoder model

```bash
# Step 1 — Convert the model (from tools/TorchFXConverter/)
python converter.py \
    --model zl369/multilingual-tinyBERT-16MB \
    --output ./out/bert/ \
    --format all \
    --weights \
    --seq-len 128

# Step 2 — Run (directory form — auto-discovers INI, JSON, and weights)
./builddir/Applications/CausalLM/nntr_runner ./out/bert/

# Step 3 — Run with real token IDs
./builddir/Applications/CausalLM/nntr_runner ./out/bert/ \
    --input "101 31178 117 48029 10271 10124 102"
```

### End-to-end example: Qwen3 decoder model

```bash
# Step 1 — Convert
python tools/TorchFXConverter/converter.py \
    --model Qwen/Qwen3-0.5B \
    --output ./out/qwen3/ \
    --format all \
    --weights \
    --dtype float16 \
    --seq-len 512

# Step 2 — Run
./builddir/Applications/CausalLM/nntr_runner ./out/qwen3/

# Step 3 — Run with prompt token IDs
./builddir/Applications/CausalLM/nntr_runner ./out/qwen3/ \
    --input "1 9906 29892 3186 29991"
```

### CLI reference

```
nntr_runner <output_dir/> [options]
nntr_runner --ini FILE [--json FILE] [--weights FILE] [options]
```

| Option | Description |
|--------|-------------|
| `<output_dir/>` | Converter output directory (auto-discovers `.ini`, `.json`, `.bin`) |
| `--ini FILE` | NNTrainer INI config |
| `--json FILE` | Converter JSON metadata (used to detect encoder vs decoder) |
| `--weights FILE` | Weight binary `.bin` or `.safetensors` |
| `--input "1 2 3"` | Space-separated token IDs for inference input |
| `--input-file FILE` | File with token IDs (one per line or space-separated) |

Without `--input` / `--input-file`, the runner uses sequential dummy token
IDs `[1, 2, 3, …]`.  Input shorter than the model's sequence length is
zero-padded automatically.

For full documentation see
[tools/TorchFXConverter/README.md](../../tools/TorchFXConverter/README.md#nntr_runner).