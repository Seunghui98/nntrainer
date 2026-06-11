"""
SPDX-License-Identifier: Apache-2.0
Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>

@file weight_converter.py
@brief gpt-oss-20b weight conversion script
@author Eunju Yang <ej.yang@samsung.com>
"""

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
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().float()
    return np.ascontiguousarray(np.array(tensor, dtype=dtype))


def get_safetensors_output_name(output_name):
    if output_name.endswith(".bin"):
        return output_name[:-4] + ".safetensors"
    if output_name.endswith(".safetensors"):
        return output_name
    return output_name + ".safetensors"


def save_gpt_oss_for_nntrainer(params, config, dtype, file):
    """Convert and save weights as nntrainer binary format for GPT-OSS."""
    n_layers = config.num_hidden_layers
    n_experts = config.num_local_experts

    def save_weight(weight_name, is_transpose=False):
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
        for proj in ["q_proj", "k_proj", "v_proj"]:
            save_projection(layer_name, f"self_attn.{proj}")
            save_weight(f"{layer_name}self_attn.{proj}.bias")

        save_weight(f"{layer_name}self_attn.sinks")

        save_projection(layer_name, "self_attn.o_proj")
        save_weight(f"{layer_name}self_attn.o_proj.bias")

    def save_feed_forward(layer_name):
        save_weight(f"{layer_name}mlp.router.weight", True)
        save_weight(f"{layer_name}mlp.router.bias")

        gate_up = params[f"{layer_name}mlp.experts.gate_up_proj"]
        gate_up_bias = params[f"{layer_name}mlp.experts.gate_up_proj_bias"]
        down = params[f"{layer_name}mlp.experts.down_proj"]
        down_bias = params[f"{layer_name}mlp.experts.down_proj_bias"]

        for expert_idx in range(n_experts):
            up_w = gate_up[..., 1::2][expert_idx]
            up_b = gate_up_bias[..., 1::2][expert_idx]
            tensor_to_numpy(up_w, dtype).tofile(file)
            tensor_to_numpy(up_b, dtype).tofile(file)

            gate_w = gate_up[..., ::2][expert_idx]
            gate_b = gate_up_bias[..., ::2][expert_idx]
            tensor_to_numpy(gate_w, dtype).tofile(file)
            tensor_to_numpy(gate_b, dtype).tofile(file)

            down_w = down[expert_idx]
            down_b = down_bias[expert_idx]
            tensor_to_numpy(down_w, dtype).tofile(file)
            tensor_to_numpy(down_b, dtype).tofile(file)

    save_weight("model.embed_tokens.weight")

    for layer_idx in range(n_layers):
        layer_prefix = f"model.layers.{layer_idx}."
        save_weight(f"{layer_prefix}input_layernorm.weight")
        save_attention(layer_prefix)
        save_weight(f"{layer_prefix}post_attention_layernorm.weight")
        save_feed_forward(layer_prefix)

    save_weight("model.norm.weight")
    save_weight("lm_head.weight", True)


def collect_gpt_oss_for_nntrainer(params, config, dtype):
    """Collect weights as ordered (nntrainer_name, ndarray) pairs for safetensors."""
    n_layers = config.num_hidden_layers
    n_experts = config.num_local_experts
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
        add(f"{nntr_prefix}_wq:bias", params[f"{hf_prefix}self_attn.q_proj.bias"])
        add_projection(f"{nntr_prefix}_wk:weight", hf_prefix, "self_attn.k_proj")
        add(f"{nntr_prefix}_wk:bias", params[f"{hf_prefix}self_attn.k_proj.bias"])
        add_projection(f"{nntr_prefix}_wv:weight", hf_prefix, "self_attn.v_proj")
        add(f"{nntr_prefix}_wv:bias", params[f"{hf_prefix}self_attn.v_proj.bias"])

        # attention sink lives on the mha_core layer in nntrainer.
        add(f"{nntr_prefix}_attention:sink", params[f"{hf_prefix}self_attn.sinks"])

        add_projection(f"{nntr_prefix}_attention_out:weight", hf_prefix, "self_attn.o_proj")
        add(f"{nntr_prefix}_attention_out:bias", params[f"{hf_prefix}self_attn.o_proj.bias"])

        add(
            f"{nntr_prefix}_ffn_norm:gamma",
            params[f"{hf_prefix}post_attention_layernorm.weight"],
        )

        # MoE layer is named "layer{N}_ffn_down" in nntrainer.
        moe = f"{nntr_prefix}_ffn_down"
        add(f"{moe}:gate", params[f"{hf_prefix}mlp.router.weight"], transpose=True)
        add(f"{moe}:gate_bias", params[f"{hf_prefix}mlp.router.bias"])

        gate_up = params[f"{hf_prefix}mlp.experts.gate_up_proj"]
        gate_up_bias = params[f"{hf_prefix}mlp.experts.gate_up_proj_bias"]
        down = params[f"{hf_prefix}mlp.experts.down_proj"]
        down_bias = params[f"{hf_prefix}mlp.experts.down_proj_bias"]

        for i in range(n_experts):
            add(f"{moe}:expert_up_{i}", gate_up[..., 1::2][i])
            add(f"{moe}:expert_up_bias_{i}", gate_up_bias[..., 1::2][i])
            add(f"{moe}:expert_gate_{i}", gate_up[..., ::2][i])
            add(f"{moe}:expert_gate_bias_{i}", gate_up_bias[..., ::2][i])
            add(f"{moe}:expert_down_{i}", down[i])
            add(f"{moe}:expert_down_bias_{i}", down_bias[i])

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
    parser.add_argument("--model_path", type=str, default=".")
    parser.add_argument("--output_name", type=str, default="./nntr_gpt_oss_20b.bin")
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
        args.model_path, torch_dtype=args.data_type, trust_remote_code=True
    )
    model.eval()

    params = model.state_dict()

    if args.safetensors:
        safetensors_path = get_safetensors_output_name(args.output_name)
        weights = collect_gpt_oss_for_nntrainer(params, config, args.data_type)
        save_safetensors(weights, safetensors_path, args.data_type)
    else:
        with open(args.output_name, "wb") as f_model:
            save_gpt_oss_for_nntrainer(params, config, args.data_type, f_model)
        print(f"Saved binary: {args.output_name}")
