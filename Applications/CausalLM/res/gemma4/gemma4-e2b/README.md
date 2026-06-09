# gemma4-E2B (Gemma 3n E2B) — DDTree weights

Source: **unsloth/gemma-4-E2B-it-GGUF**, file `gemma-4-E2B-it-Q4_0.gguf` (GGUF arch = native `gemma4`).
`tokenizer.json` / `config.json` / `tokenizer_config.json` / `generation_config.json` from base **unsloth/gemma-4-E2B-it**.

Converted to nntrainer **safetensors** via `gguf_to_nntrainer.py`
(native-gemma4 Q4_0 GGUF -> nntr quantized safetensors). Run on x86:

```
python gguf_to_nntrainer.py \
  /path/to/gemma-4-E2B-it-Q4_0.gguf \
  -o nntr_gemma4_e2b_q4.safetensors --target x86
./build-app/Applications/CausalLM/nntr_causallm \
  Applications/CausalLM/res/gemma4/gemma4-e2b
```

## Quant / layout policy

| nntr weight group | dtype | layout |
|---|---|---|
| FC weights (wq/wk/wv/attention_out, ffn_gate/up/down, per-layer FCs) | **Q4_0** | repacked **q4_0x8** (x86); nntr does NOT repack at load |
| `embedding0` (TieWordEmbedding, tied lm_head) | **Q6_K** | plain row-major (210 B/block); dequantized per-row at load |
| `per_layer_input_embedding` (regular `embedding_layer`, `EMBEDDING_DTYPE`) | **Q6_K** | plain row-major (210 B/block); dequantized per-row at load (same as `embedding0`) |
| norms / 1-D / scalar (`:gamma`, `:scalar_multiplier`) | **F32** | dense (graph uses `packed=false` -> weight dtype follows FP32 activation) |

`nntr_config.json`: `fc_layer_dtype=Q4_0`, `embedding_dtype=Q6_K`, `lmhead_dtype=Q6_K`,
`model_tensor_type="Q4_0-FP32"`.

## How nntr matches safetensors tensors

`neuralnet.cpp` `MODEL_FORMAT_SAFETENSORS` load matches tensors **by NAME**
(`weight->getName()` vs the header keys); it reads ONLY `data_offsets` from the
header — the `dtype`/`shape` labels are informational and ignored on load. Each
weight's resolved dtype comes from the graph, and the raw bytes are read directly
into the weight's memory. So the emitted byte count per tensor MUST equal that
weight's `getMemoryBytes()` for its resolved dtype. Names follow nntr's
`"<layer_name>:<suffix>"` convention: `fully_connected` -> `:weight`,
embedding / tie-embedding -> `:Embedding`, `rms_norm`/`reshaped_rms_norm` ->
`:gamma`, `scalar_multiply` -> `:scalar_multiplier`.

Safetensors header (matches `safetensors_util.cpp` `buildHeader`):
`{"__metadata__":{"format":"nntrainer"}, "<name>":{"dtype":..,"shape":[..],"data_offsets":[s,e]}, ...}`,
prefixed by an 8-byte little-endian header length and space-padded to an 8-byte
boundary, followed by the raw tensor data.

## Model topology (from config / GGUF)

- 35 decoder blocks; hidden 1536; vocab 262144; n_heads 8; head_count_kv 1 (MQA).
- **sliding** layers: head_dim 256 (`key_length_swa`), q=[2048,1536], k/v=[256,1536], attn_out=[1536,2048], window 512.
- **full-attention** layers (idx 4,9,14,19,24,29,34): head_dim 512 (`key_length`), q=[4096,1536], k/v=[512,1536], attn_out=[1536,4096].
- **KV-shared** layers: last `num_kv_shared_layers=20` (idx 15..34). These reuse a
  previous same-type layer's K/V -> the nntr graph's `createSharedAttention` emits
  **only** `wq` / `q_norm` / `attention_out` for them (NO wk/wv/k_norm). The converter
  therefore skips `attn_k`/`attn_v`/`attn_k_norm` for layers 15..34.
- **double-wide MLP** (`use_double_wide_mlp=true`): KV-shared layers use intermediate
  12288 (gate/up=[12288,1536], down=[1536,12288]); layers 0..14 use 6144.
- per-layer input (Gemma 3n PLE): `per_layer_input_embedding` [262144, 35*256=8960],
  `per_layer_input_projection` [8960,1536], plus per-block PLE gate/proj/norm/scalar.

## GGUF tensor -> nntr weight mapping (resolved)

Top-level:

| GGUF tensor | shape | ggml | nntr weight name | dtype |
|---|---|---|---|---|
| token_embd.weight | [262144,1536] | Q4_K | `embedding0:Embedding` | Q6_K |
| per_layer_token_embd.weight | [262144,8960] | Q5_K | `per_layer_input_embedding:Embedding` | Q6_K |
| per_layer_model_proj.weight | [8960,1536] | BF16 | `per_layer_input_projection:weight` | Q4_0 |
| per_layer_proj_norm.weight | [256] | F32 | `per_layer_projection_norm:gamma` | F32 |
| output_norm.weight | [1536] | F32 | `output_norm:gamma` | F32 |
| rope_freqs.weight | [256] | F32 | (RoPE freqs; computed in graph, not a stored weight) |

Per block N (`blk.N.` -> `layerN_`):

| GGUF tensor | ggml | nntr weight name | dtype | note |
|---|---|---|---|---|
| attn_norm.weight | F32 | `layerN_attention_norm:gamma` | F32 | input layernorm |
| attn_q.weight | Q4_0 | `layerN_wq:weight` | Q4_0 | always |
| attn_q_norm.weight | F32 | `layerN_q_norm:gamma` | F32 | always (feature = head_dim) |
| attn_k.weight | Q4_0 | `layerN_wk:weight` | Q4_0 | **non-shared layers only** (0..14) |
| attn_k_norm.weight | F32 | `layerN_k_norm:gamma` | F32 | **non-shared layers only** |
| attn_v.weight | Q4_0 | `layerN_wv:weight` | Q4_0 | **non-shared layers only** |
| (no attn_v_norm in GGUF) | — | `layerN_v_norm` | — | nntr `v_norm` has `use_gamma=false` -> **no weight requested**; nothing to supply |
| attn_output.weight | Q4_0 | `layerN_attention_out:weight` | Q4_0 | |
| post_attention_norm.weight | F32 | `layerN_post_attention_norm:gamma` | F32 | |
| ffn_norm.weight | F32 | `layerN_pre_ffn_norm:gamma` | F32 | pre-FFN norm |
| ffn_gate.weight | Q4_0 | `layerN_ffn_gate:weight` | Q4_0 | |
| ffn_up.weight | Q4_0 | `layerN_ffn_up:weight` | Q4_0 | |
| ffn_down.weight | Q4_1/Q4_0 | `layerN_ffn_down:weight` | Q4_0 | Q4_1 dequant->requant to Q4_0 (layers 0..14); Q4_0 elsewhere |
| post_ffw_norm.weight | F32 | `layerN_post_ffn_norm:gamma` | F32 | |
| inp_gate.weight | F32 | `layerN_per_layer_input_gate:weight` | Q4_0 | PLE gate FC |
| proj.weight | F32 | `layerN_per_layer_input_proj:weight` | Q4_0 | PLE projection FC |
| post_norm.weight | F32 | `layerN_post_per_layer_input_norm:gamma` | F32 | |
| layer_output_scale.weight | F32 | `layerN_layer_scalar:scalar_multiplier` | F32 | scalar_multiply (use_weight) |

### Resolution of the open mapping questions

- **v_norm**: the nntr `createAttention` builds `v_norm` as a `reshaped_rms_norm`
  with `use_gamma=false`, so it requests **no** weight. The GGUF has no `attn_v_norm`
  and none is needed — the converter emits nothing for v_norm (consistent for both
  sliding and full layers). For KV-shared layers there is no k_norm/v_norm at all
  (K/V are shared from the source layer).
- **5 GGUF norms -> nntr per-block norms**: `attn_norm`->`attention_norm`,
  `post_attention_norm`->`post_attention_norm`, `ffn_norm`->`pre_ffn_norm`,
  `post_ffw_norm`->`post_ffn_norm`, `post_norm`->`post_per_layer_input_norm`.
- **proj / per_layer_model_proj**: GGUF `per_layer_model_proj` [8960,1536] ->
  nntr `per_layer_input_projection` (top-level FC, unit=8960). GGUF per-block
  `proj` [1536,256] -> nntr per-block `layerN_per_layer_input_proj` (FC unit=1536).
  `per_layer_model_proj_scale` / `per_layer_input_scale` are `scalar_multiply`
  layers with no weight (constant multiplier baked in the graph).
- **inp_gate -> per_layer_input gate**: GGUF per-block `inp_gate` [256,1536] ->
  nntr `layerN_per_layer_input_gate` (FC unit=256).
- **layer_output_scale -> layer_scalar**: GGUF per-block `layer_output_scale` [1]
  -> nntr `layerN_layer_scalar:scalar_multiplier` (single FP32 element).

Embedding scale (`sqrt(DIM)`), per-layer-input scales and final logit softcapping
are applied by the nntr graph itself; they are NOT baked into the weights.
