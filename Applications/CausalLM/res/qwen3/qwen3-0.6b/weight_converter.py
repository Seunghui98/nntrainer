## @file weight_converter.py
## @brief weight conversion script for qwen3-0.6b model
## @author Eunju Yang <ej.yang@samsung.com>

import argparse
import struct
import json
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# Qwen3-0.6B uses tie_word_embeddings=True:
#   - embedding0 layer type: tie_word_embeddings (mode::embedding)
#     weight key: "embedding0:Embedding"
#   - output_of_causallm layer type: tie_word_embeddings (mode::lm_head)
#     read() is a no-op — shares weight with embedding0
# So lm_head weight must NOT be written to the weight file.


def save_qwen3_for_nntrainer(params, n_layers, dtype, file, tie_word_embeddings=True):
    """Convert and save weights as nntrainer binary format for qwen3-0.6b model."""

    def save_weight(weight):
        np.array(weight, dtype=dtype).tofile(file)

    def save_projection(layer_name, proj_name):
        lora_key = f"{layer_name}{proj_name}.lora_A.default.weight"
        if lora_key in params:
            save_weight(params[f"{layer_name}{proj_name}.base_layer.weight"].permute(1, 0))
            save_weight(params[f"{layer_name}{proj_name}.lora_A.default.weight"].permute(1, 0))
            save_weight(params[f"{layer_name}{proj_name}.lora_B.default.weight"].permute(1, 0))
        else:
            save_weight(params[f"{layer_name}{proj_name}.weight"].permute(1, 0))

    def save_attention(layer_name):
        save_weight(params[f"{layer_name}input_layernorm.weight"])
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            save_projection(layer_name, f"self_attn.{proj}")
            proj_norm_name = f"{layer_name}self_attn.{proj[0]}_norm.weight"
            if proj_norm_name in params:
                save_weight(params[proj_norm_name])

    def save_feed_forward(layer_name):
        save_weight(params[f"{layer_name}post_attention_layernorm.weight"])
        for proj in ["up_proj", "gate_proj", "down_proj"]:
            save_projection(layer_name, f"mlp.{proj}")

    # Embedding: nntrainer key = "embedding0:Embedding"
    save_weight(params["model.embed_tokens.weight"])

    for layer_idx in range(n_layers):
        layer_prefix = f"model.layers.{layer_idx}."
        save_attention(layer_prefix)
        save_feed_forward(layer_prefix)

    save_weight(params["model.norm.weight"])

    # lm_head is NOT written for tied models:
    # TieWordEmbedding.read() is a no-op in lm_head mode — it shares embedding0.
    if not tie_word_embeddings:
        save_weight(params["lm_head.weight"].permute(1, 0))


def collect_qwen3_for_nntrainer(params, n_layers, dtype, tie_word_embeddings=True):
    """Collect weights as an ordered list of (nntrainer_name, ndarray) tuples."""
    weights = []

    def add(name, tensor, transpose=False):
        t = tensor.permute(1, 0) if transpose else tensor
        weights.append((name, t.detach().numpy().astype(dtype)))

    def add_projection(nntr_name, layer_name, proj_name, transpose=True):
        lora_key = f"{layer_name}{proj_name}.lora_A.default.weight"
        if lora_key in params:
            add(nntr_name, params[f"{layer_name}{proj_name}.base_layer.weight"], transpose)
        else:
            add(nntr_name, params[f"{layer_name}{proj_name}.weight"], transpose)

    # Embedding layer registers weight as "Embedding" (capital E)
    add("embedding0:Embedding", params["model.embed_tokens.weight"])

    for i in range(n_layers):
        p = f"model.layers.{i}."
        pfx = f"layer{i}"

        add(f"{pfx}_attention_norm:gamma",    params[f"{p}input_layernorm.weight"])
        add_projection(f"{pfx}_wq:weight",    p, "self_attn.q_proj")
        add(f"{pfx}_q_norm:gamma",            params[f"{p}self_attn.q_norm.weight"])
        add_projection(f"{pfx}_wk:weight",    p, "self_attn.k_proj")
        add(f"{pfx}_k_norm:gamma",            params[f"{p}self_attn.k_norm.weight"])
        add_projection(f"{pfx}_wv:weight",    p, "self_attn.v_proj")
        add_projection(f"{pfx}_attention_out:weight", p, "self_attn.o_proj")
        add(f"{pfx}_ffn_norm:gamma",          params[f"{p}post_attention_layernorm.weight"])
        add_projection(f"{pfx}_ffn_up:weight",   p, "mlp.up_proj")
        add_projection(f"{pfx}_ffn_gate:weight", p, "mlp.gate_proj")
        add_projection(f"{pfx}_ffn_down:weight", p, "mlp.down_proj")

    add("output_norm:gamma", params["model.norm.weight"])

    # No lm_head entry for tied models: TieWordEmbedding lm_head read() is no-op
    if not tie_word_embeddings:
        add("output_of_causallm:weight", params["lm_head.weight"], transpose=True)

    return weights


def save_safetensors(weights, output_path):
    """Write weights to a .safetensors file (F32, name-keyed)."""
    metadata = {"format": "pt"}

    offset = 0
    tensor_meta = {}
    raw_buffers = []
    for name, arr in weights:
        nbytes = arr.nbytes
        tensor_meta[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        raw_buffers.append(arr.tobytes())
        offset += nbytes

    header = {"__metadata__": metadata}
    header.update(tensor_meta)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad

    with open(output_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for buf in raw_buffers:
            f.write(buf)

    print(f"Saved safetensors: {output_path} ({offset / 1e9:.2f} GB tensor data)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./Qwen3-0.6B")
    parser.add_argument("--output_name", type=str, default="./nntr_qwen3_0.6b_fp32.bin")
    parser.add_argument("--data_type", type=str, default="float32")
    parser.add_argument("--safetensors", action="store_true",
                        help="Save in safetensors format instead of binary")
    args = parser.parse_args()

    data_dtype = args.data_type
    model_path = args.model_path
    output_name = args.output_name
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="float", trust_remote_code=True)
    model.eval()

    is_tied = getattr(config, 'tie_word_embeddings', True)
    print(f"tie_word_embeddings: {is_tied}")

    if args.safetensors:
        output_name = output_name.replace(".bin", ".safetensors")
        if not output_name.endswith(".safetensors"):
            output_name += ".safetensors"
        weights = collect_qwen3_for_nntrainer(model.state_dict(), config.num_hidden_layers,
                                              data_dtype, tie_word_embeddings=is_tied)
        save_safetensors(weights, output_name)
    else:
        with open(output_name, "wb") as f_model:
            save_qwen3_for_nntrainer(model.state_dict(), config.num_hidden_layers,
                                     data_dtype, f_model, tie_word_embeddings=is_tied)
        print(f"Saved binary: {output_name}")
