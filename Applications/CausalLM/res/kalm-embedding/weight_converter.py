## @file weight_converter.py
## @brief weight conversion script for kalm-embedding model (Qwen2-based)
## @author Seunghui Lee <shsh1004.lee@samsung.com>

import argparse
import struct
import json
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Newer transformers versions removed rope_theta as a direct attribute on
# Qwen2Config; patch __getattr__ so the KaLM custom modeling.py still works.
try:
    from transformers import Qwen2Config
    _orig_getattr = getattr(Qwen2Config, "__getattr__", None)

    def _qwen2_getattr_patch(self, name):
        if name == "rope_theta":
            # Fall back to rope_scaling dict or Qwen2 default (1 000 000)
            rs = self.__dict__.get("rope_scaling") or {}
            return rs.get("rope_theta", 1_000_000.0)
        if _orig_getattr:
            return _orig_getattr(self, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    Qwen2Config.__getattr__ = _qwen2_getattr_patch
except ImportError:
    pass

# KaLM-embedding uses Qwen2 architecture (Qwen2Model):
#   - Q/K/V projections have biases (o_proj does NOT)
#   - No q_norm / k_norm (Qwen3-specific)
#   - embedding0 uses EmbeddingLayer → weight key = "embedding0:Embedding"
#   - No lm_head (embedding model); only output_norm at the end
#   - Pooling / Normalize modules have no learnable weights


def save_kalm_embedding_for_nntrainer(params, n_layers, dtype, file):
    """Convert and save weights as nntrainer binary format for kalm-embedding."""

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
            if proj != "o_proj":
                save_weight(params[f"{layer_name}self_attn.{proj}.bias"])

    def save_feed_forward(layer_name):
        save_weight(params[f"{layer_name}post_attention_layernorm.weight"])
        for proj in ["up_proj", "gate_proj", "down_proj"]:
            save_projection(layer_name, f"mlp.{proj}")

    # Embedding: nntrainer key = "embedding0:Embedding"
    save_weight(params["0.auto_model.embed_tokens.weight"])

    for layer_idx in range(n_layers):
        layer_prefix = f"0.auto_model.layers.{layer_idx}."
        save_attention(layer_prefix)
        save_feed_forward(layer_prefix)

    save_weight(params["0.auto_model.norm.weight"])


def collect_kalm_embedding_for_nntrainer(params, n_layers, dtype):
    """Collect weights as an ordered list of (nntrainer_name, ndarray) tuples."""
    weights = []

    def add(name, tensor, transpose=False):
        # .contiguous() forces a memory-contiguous tensor before numpy()
        # conversion. Without it, .numpy() on a permuted view can produce a
        # non-contiguous array, and .astype()/.tobytes() then walks strides
        # differently depending on numpy/PyTorch version — leading to bytes
        # that do not match what the binary save_weight() writes via
        # np.array(t, dtype=...).tofile().
        t = tensor.permute(1, 0).contiguous() if transpose else tensor.contiguous()
        weights.append((name, t.detach().numpy().astype(dtype)))

    def add_projection(nntr_name, layer_name, proj_name, transpose=True):
        lora_key = f"{layer_name}{proj_name}.lora_A.default.weight"
        if lora_key in params:
            add(nntr_name, params[f"{layer_name}{proj_name}.base_layer.weight"], transpose)
        else:
            add(nntr_name, params[f"{layer_name}{proj_name}.weight"], transpose)

    # Embedding: EmbeddingLayer registers weight as "Embedding" (capital E)
    add("embedding0:Embedding", params["0.auto_model.embed_tokens.weight"])

    for i in range(n_layers):
        p = f"0.auto_model.layers.{i}."
        pfx = f"layer{i}"

        add(f"{pfx}_attention_norm:gamma", params[f"{p}input_layernorm.weight"])

        # Q/K/V have both weight and bias; O has no bias
        for proj, nntr in [("q_proj", f"{pfx}_wq"), ("k_proj", f"{pfx}_wk"),
                            ("v_proj", f"{pfx}_wv")]:
            add_projection(f"{nntr}:weight", p, f"self_attn.{proj}")
            add(f"{nntr}:bias", params[f"{p}self_attn.{proj}.bias"])

        add_projection(f"{pfx}_attention_out:weight", p, "self_attn.o_proj")

        add(f"{pfx}_ffn_norm:gamma", params[f"{p}post_attention_layernorm.weight"])
        add_projection(f"{pfx}_ffn_up:weight",   p, "mlp.up_proj")
        add_projection(f"{pfx}_ffn_gate:weight", p, "mlp.gate_proj")
        add_projection(f"{pfx}_ffn_down:weight", p, "mlp.down_proj")

    add("output_norm:gamma", params["0.auto_model.norm.weight"])

    # No lm_head — this is an embedding model
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
    parser.add_argument("--model_path", type=str,
                        default="KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5")
    parser.add_argument("--output_name", type=str,
                        default="./nntr_kalm_embedding_fp32.bin")
    parser.add_argument("--data_type", type=str, default="float32")
    parser.add_argument("--safetensors", action="store_true",
                        help="Save in safetensors format instead of binary")
    args = parser.parse_args()

    data_dtype = args.data_type
    model_path = args.model_path
    output_name = args.output_name

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = SentenceTransformer(model_path, trust_remote_code=True,
                                model_kwargs={"torch_dtype": data_dtype})
    model.eval()

    if args.safetensors:
        output_name = output_name.replace(".bin", ".safetensors")
        if not output_name.endswith(".safetensors"):
            output_name += ".safetensors"
        weights = collect_kalm_embedding_for_nntrainer(
            model.state_dict(), config.num_hidden_layers, data_dtype)
        save_safetensors(weights, output_name)
    else:
        with open(output_name, "wb") as f_model:
            save_kalm_embedding_for_nntrainer(
                model.state_dict(), config.num_hidden_layers, data_dtype, f_model)
        print(f"Saved binary: {output_name}")
