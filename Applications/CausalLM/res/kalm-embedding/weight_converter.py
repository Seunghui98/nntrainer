# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>

# @file weight_converter.py
# @brief weight conversion script for kalm-embedding model
# @author Seunghui Lee <shsh1004.lee@samsung.com>

import argparse
import json
import struct

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
from sentence_transformers import SentenceTransformer


SAFETENSORS_DTYPE_MAP = {
    "float32": "F32",
}


def tensor_to_numpy(tensor, dtype, transpose=False):
    if transpose:
        tensor = tensor.permute(1, 0)
    return np.ascontiguousarray(tensor.detach().cpu().numpy().astype(dtype))


def get_safetensors_output_name(output_name):
    if output_name.endswith(".bin"):
        return output_name[:-4] + ".safetensors"
    if output_name.endswith(".safetensors"):
        return output_name
    return output_name + ".safetensors"


def save_kalm_embedding_for_nntrainer(params, n_layers, dtype, file):
    """Convert and save kalm-embedding weights as nntrainer binary format."""

    def save_weight(weight, transpose=False):
        tensor_to_numpy(weight, dtype, transpose=transpose).tofile(file)

    def save_projection(layer_name, proj_name):
        lora_a = f"{layer_name}{proj_name}.lora_A.default.weight"
        if lora_a in params:
            save_weight(params[f"{layer_name}{proj_name}.base_layer.weight"], transpose=True)
            save_weight(params[lora_a], transpose=True)
            save_weight(params[f"{layer_name}{proj_name}.lora_B.default.weight"], transpose=True)
        else:
            save_weight(params[f"{layer_name}{proj_name}.weight"], transpose=True)

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

    save_weight(params["0.auto_model.embed_tokens.weight"])

    for layer_idx in range(n_layers):
        layer_prefix = f"0.auto_model.layers.{layer_idx}."
        save_attention(layer_prefix)
        save_feed_forward(layer_prefix)

    save_weight(params["0.auto_model.norm.weight"])


def collect_kalm_embedding_for_nntrainer(params, n_layers, dtype):
    """Collect weights as ordered (nntrainer_name, ndarray) pairs for safetensors."""
    weights = []

    def add(name, tensor, transpose=False):
        arr = tensor_to_numpy(tensor, dtype, transpose=transpose)
        weights.append((name, arr))

    def add_projection(nntr_name, layer_name, proj_name):
        lora_a = f"{layer_name}{proj_name}.lora_A.default.weight"
        if lora_a in params:
            add(nntr_name, params[f"{layer_name}{proj_name}.base_layer.weight"], transpose=True)
            return
        add(nntr_name, params[f"{layer_name}{proj_name}.weight"], transpose=True)

    add("embedding0:Embedding", params["0.auto_model.embed_tokens.weight"])

    for layer_idx in range(n_layers):
        hf_prefix = f"0.auto_model.layers.{layer_idx}."
        nntr_prefix = f"layer{layer_idx}"

        add(f"{nntr_prefix}_attention_norm:gamma", params[f"{hf_prefix}input_layernorm.weight"])

        add_projection(f"{nntr_prefix}_wq:weight", hf_prefix, "self_attn.q_proj")
        add(f"{nntr_prefix}_wq:bias", params[f"{hf_prefix}self_attn.q_proj.bias"])
        add_projection(f"{nntr_prefix}_wk:weight", hf_prefix, "self_attn.k_proj")
        add(f"{nntr_prefix}_wk:bias", params[f"{hf_prefix}self_attn.k_proj.bias"])
        add_projection(f"{nntr_prefix}_wv:weight", hf_prefix, "self_attn.v_proj")
        add(f"{nntr_prefix}_wv:bias", params[f"{hf_prefix}self_attn.v_proj.bias"])
        add_projection(f"{nntr_prefix}_attention_out:weight", hf_prefix, "self_attn.o_proj")

        add(f"{nntr_prefix}_ffn_norm:gamma", params[f"{hf_prefix}post_attention_layernorm.weight"])
        add_projection(f"{nntr_prefix}_ffn_up:weight", hf_prefix, "mlp.up_proj")
        add_projection(f"{nntr_prefix}_ffn_gate:weight", hf_prefix, "mlp.gate_proj")
        add_projection(f"{nntr_prefix}_ffn_down:weight", hf_prefix, "mlp.down_proj")

    add("output_norm:gamma", params["0.auto_model.norm.weight"])

    return weights


def save_safetensors(weights, output_path, dtype):
    if dtype not in SAFETENSORS_DTYPE_MAP:
        raise ValueError(f"Unsupported safetensors dtype: {dtype}")

    safetensors_dtype = SAFETENSORS_DTYPE_MAP[dtype]
    metadata = {"format": "pt"}

    offset = 0
    tensor_meta = {}
    raw_buffers = []

    for name, arr in weights:
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        nbytes = arr.nbytes
        tensor_meta[name] = {
            "dtype": safetensors_dtype,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        raw_buffers.append(arr.tobytes(order="C"))
        offset += nbytes

    header = {"__metadata__": metadata}
    header.update(tensor_meta)

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad

    with open(output_path, "wb") as output_file:
        output_file.write(struct.pack("<Q", len(header_bytes)))
        output_file.write(header_bytes)
        for buffer in raw_buffers:
            output_file.write(buffer)

    print(f"Saved safetensors: {output_path}")
    print(f"Tensor data size: {offset / 1e9:.2f} GB")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
    )
    parser.add_argument(
        "--output_name", type=str, default="./nntr_kalm_embedding_fp32.bin"
    )
    parser.add_argument("--data_type", type=str, default="float32", choices=["float32"])
    parser.add_argument(
        "--safetensors",
        action="store_true",
        help="Save weights in safetensors format instead of binary format",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    AutoTokenizer.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)

    model = SentenceTransformer(
        args.model_path, trust_remote_code=True, model_kwargs={"torch_dtype": args.data_type}
    )
    model.eval()

    params = model.state_dict()

    if args.safetensors:
        safetensors_path = get_safetensors_output_name(args.output_name)
        weights = collect_kalm_embedding_for_nntrainer(
            params, config.num_hidden_layers, args.data_type
        )
        save_safetensors(weights, safetensors_path, args.data_type)
        return

    with open(args.output_name, "wb") as f_model:
        save_kalm_embedding_for_nntrainer(
            params, config.num_hidden_layers, args.data_type, f_model
        )
    print(f"Saved binary: {args.output_name}")


if __name__ == "__main__":
    main()
