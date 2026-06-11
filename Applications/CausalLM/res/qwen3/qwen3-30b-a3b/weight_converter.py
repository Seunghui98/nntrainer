## SPDX-License-Identifier: Apache-2.0
## Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>

## @file weight_converter.py
## @brief weight conversion script for qwen3_moe model
## @author Eunju Yang <ej.yang@samsung.com>

import argparse
import json
import struct

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


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


def save_qwen3_moe_for_nntrainer(params, config, dtype, file):
    """Convert and save weights as nntrainer binary format for Qwen3 MoE."""
    n_layers = config.num_hidden_layers
    n_experts = config.num_experts

    def save_weight(weight_name, is_transpose=False):
        print(weight_name, params[weight_name].shape)
        arr = tensor_to_numpy(params[weight_name], dtype, transpose=is_transpose)
        arr.tofile(file)

    def save_projection(layer_name, proj_name):
        lora_a = f"{layer_name}{proj_name}.lora_A.default.weight"
        if lora_a in params:
            save_weight(f"{layer_name}{proj_name}.base_layer.weight", True)
            save_weight(f"{layer_name}{proj_name}.lora_A.default.weight", True)
            save_weight(f"{layer_name}{proj_name}.lora_B.default.weight", True)
        else:
            save_weight(f"{layer_name}{proj_name}.weight", True)

    def save_attention(layer_name):
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            save_projection(layer_name, f"self_attn.{proj}")
            proj_norm_name = f"{layer_name}self_attn.{proj[0]}_norm.weight"
            if proj_norm_name in params:
                save_weight(proj_norm_name)

    def save_feed_forward(layer_name):
        save_weight(f"{layer_name}mlp.gate.weight", True)
        for expert_id in range(n_experts):
            for proj in ["up_proj", "gate_proj", "down_proj"]:
                save_projection(layer_name, f"mlp.experts.{expert_id}.{proj}")

    save_weight("model.embed_tokens.weight")

    for layer_idx in range(n_layers):
        layer_prefix = f"model.layers.{layer_idx}."
        save_weight(f"{layer_prefix}input_layernorm.weight")
        save_attention(layer_prefix)
        save_weight(f"{layer_prefix}post_attention_layernorm.weight")
        save_feed_forward(layer_prefix)

    save_weight("model.norm.weight")
    save_weight("lm_head.weight", True)


def collect_qwen3_moe_for_nntrainer(params, config, dtype):
    """Collect weights as ordered (nntrainer_name, ndarray) pairs for safetensors."""
    n_layers = config.num_hidden_layers
    n_experts = config.num_experts
    tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
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

    add("embedding0:Embedding", params["model.embed_tokens.weight"])

    for layer_idx in range(n_layers):
        hf_prefix = f"model.layers.{layer_idx}."
        nntr_prefix = f"layer{layer_idx}"

        add(f"{nntr_prefix}_attention_norm:gamma", params[f"{hf_prefix}input_layernorm.weight"])

        add_projection(f"{nntr_prefix}_wq:weight", hf_prefix, "self_attn.q_proj")
        q_norm_key = f"{hf_prefix}self_attn.q_norm.weight"
        if q_norm_key in params:
            add(f"{nntr_prefix}_q_norm:gamma", params[q_norm_key])

        add_projection(f"{nntr_prefix}_wk:weight", hf_prefix, "self_attn.k_proj")
        k_norm_key = f"{hf_prefix}self_attn.k_norm.weight"
        if k_norm_key in params:
            add(f"{nntr_prefix}_k_norm:gamma", params[k_norm_key])

        add_projection(f"{nntr_prefix}_wv:weight", hf_prefix, "self_attn.v_proj")
        add_projection(f"{nntr_prefix}_attention_out:weight", hf_prefix, "self_attn.o_proj")

        add(
            f"{nntr_prefix}_ffn_norm:gamma",
            params[f"{hf_prefix}post_attention_layernorm.weight"],
        )

        # MoE layer is named "layer{N}_ffn_down" in nntrainer; weights inside are
        # prefixed with that layer name and suffixed with internal weight names.
        moe_layer = f"{nntr_prefix}_ffn_down"
        add(f"{moe_layer}:gate", params[f"{hf_prefix}mlp.gate.weight"], transpose=True)
        for expert_id in range(n_experts):
            add_projection(
                f"{moe_layer}:expert_up_{expert_id}",
                hf_prefix,
                f"mlp.experts.{expert_id}.up_proj",
            )
            add_projection(
                f"{moe_layer}:expert_gate_{expert_id}",
                hf_prefix,
                f"mlp.experts.{expert_id}.gate_proj",
            )
            add_projection(
                f"{moe_layer}:expert_down_{expert_id}",
                hf_prefix,
                f"mlp.experts.{expert_id}.down_proj",
            )

    add("output_norm:gamma", params["model.norm.weight"])
    if not tie_word_embeddings:
        add("output_of_causallm:weight", params["lm_head.weight"], transpose=True)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./Qwen3-30b-a3b")
    parser.add_argument(
        "--output_name", type=str, default="./nntr_qwen3_30b_a3b_fp32.bin"
    )
    parser.add_argument("--data_type", type=str, default="float32", choices=["float32"])
    parser.add_argument(
        "--safetensors",
        action="store_true",
        help="Save weights in safetensors format instead of binary format",
    )
    args = parser.parse_args()

    AutoTokenizer.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype="float", trust_remote_code=True
    )
    model.eval()

    params = model.state_dict()

    if args.safetensors:
        safetensors_path = get_safetensors_output_name(args.output_name)
        weights = collect_qwen3_moe_for_nntrainer(params, config, args.data_type)
        save_safetensors(weights, safetensors_path, args.data_type)
    else:
        with open(args.output_name, "wb") as f_model:
            save_qwen3_moe_for_nntrainer(params, config, args.data_type, f_model)
        print(f"Saved binary: {args.output_name}")
