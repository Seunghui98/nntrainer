# Qwen3-0.6B (Q4_0 FC + Q6_K embedding)

Resources and converter for running Qwen3-0.6B inside `Applications/CausalLM`
from a HuggingFace GGUF file, without the usual FP32 round-trip.

## Files

- `gguf_to_nntrainer.py` — converts a GGUF to an nntrainer `.bin`:
  - Q6_K `token_embd.weight` is copied byte-for-byte.
  - Each Q4_0 linear weight is repacked into nntrainer's interleaved
    `q4_0x8` (x86, default) or `q4_0x4` (ARM) layout with the `0x88` nibble
    XOR the C backends apply at save time.
  - Q4_1 / Q8_0 tensors (common for `ffn_down.weight` in HF Q4_0 GGUFs) are
    transparently dequantised and re-quantised to Q4_0, even in `--strict`
    mode.
  - RMSNorm scales are emitted as FP32.
  - When the GGUF has `tie_word_embeddings=true` (the default for 0.6B),
    `output.weight` is omitted so the shared embedding weight is only written
    once, matching nntrainer's save path.
- `nntr_config.json` — matching config (`Q4_0-FP32`, Q6_K embedding).

## Usage

```bash
# Grab a Qwen3-0.6B GGUF with Q4_0 FC + Q6_K embedding
# (e.g. Qwen/Qwen3-0.6B-GGUF, file Qwen3-0.6B-Q4_0.gguf).
python3 gguf_to_nntrainer.py \
    /path/to/Qwen3-0.6B-Q4_0.gguf \
    -o nntr_qwen3_0.6b_q40_embdq6k.bin \
    --target x86          # or 'arm' for Android builds
    --strict              # fail unless embedding=Q6_K and FC=Q4_0/Q4_1/Q8_0
```

Optional flags:
- `--fc-dtype {q4_0,q6_k,fp32}` — override target dtype for all FC weights
  (also updates `fc_layer_dtype` in the emitted `nntr_config.json`).
- `--ffn-down-dtype {q4_0,q6_k}` — Q6_K just for `ffn_down.weight`. Note
  current nntrainer has a single `fc_layer_dtype` so mixed Q4_0 + Q6_K
  will not load cleanly unless you extend the runtime.
- `--emit-nntr-config` — (re-)generate `nntr_config.json` next to the .bin.

Then copy `config.json`, `generation_config.json`, and `tokenizer.json`
from the HuggingFace `Qwen/Qwen3-0.6B` repo into this directory (alongside
`nntr_config.json`) and run `nntr_causallm` against the directory.

## Layout the script produces

The binary is a concatenation of raw tensor bytes in the order the
`Qwen3CausalLM` graph traverses them:

```
embedding0                      (Q6_K)
for each layer i:
  layer{i}_attention_norm       (FP32)
  layer{i}_wq                   (Q4_0x8 / Q4_0x4)
  layer{i}_q_norm               (FP32)
  layer{i}_wk                   (Q4_0x8 / Q4_0x4)
  layer{i}_k_norm               (FP32)
  layer{i}_wv                   (Q4_0x8 / Q4_0x4)
  layer{i}_attention_out        (Q4_0x8 / Q4_0x4)
  layer{i}_ffn_norm             (FP32)
  layer{i}_ffn_up               (Q4_0x8 / Q4_0x4)
  layer{i}_ffn_gate             (Q4_0x8 / Q4_0x4)
  layer{i}_ffn_down             (Q4_0x8 / Q4_0x4)
output_norm                     (FP32)
# output_of_causallm is shared with embedding when tie_word_embeddings=true
```

`N` (rows = `out_features`) must be divisible by the interleave group size
(8 for x86, 4 for ARM) — this is true for every linear in Qwen3-0.6B
(hidden=1024, q=2048, kv=1024, ffn=3072).
