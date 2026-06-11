## SPDX-License-Identifier: Apache-2.0
## Copyright (C) 2025 Seungbaek Hong <sb92.hong@samsung.com>
##
## @file weight_converter.py
## @brief weight conversion script for gemma3 model
## @author SeungBaek Hong <sb92.hong@samsung.com>

import argparse
import json
import struct

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


SAFETENSORS_DTYPE_MAP = {
    "float32": "F32",
    "float16": "F16",
}


def tensor_to_numpy(tensor, dtype, transpose=False, is_rms=False):
    """Convert torch tensor to numpy array. RMSNorm gamma has +1.0 added."""
    if is_rms:
        tensor = tensor + 1.0
    if transpose:
        tensor = tensor.permute(1, 0)
    return np.ascontiguousarray(tensor.detach().cpu().numpy().astype(dtype))


def get_safetensors_output_name(output_name):
    if output_name.endswith(".bin"):
        return output_name[:-4] + ".safetensors"
    if output_name.endswith(".safetensors"):
        return output_name
    return output_name + ".safetensors"


def save_gemma3_for_nntrainer(params, config, dtype, file, save_lm_head=True):
    """Convert and save weights as nntrainer binary format for Gemma3."""
    n_layers = config.num_hidden_layers

    def save_weight(weight, is_rms=False, transpose=False):
        arr = tensor_to_numpy(weight, dtype, transpose=transpose, is_rms=is_rms)
        arr.tofile(file)

    def save_projection(layer_name, proj_name):
        lora_a = f"{layer_name}{proj_name}.lora_A.default.weight"
        if lora_a in params:
            save_weight(params[f"{layer_name}{proj_name}.base_layer.weight"], transpose=True)
            save_weight(params[lora_a], transpose=True)
            save_weight(params[f"{layer_name}{proj_name}.lora_B.default.weight"], transpose=True)
        else:
            save_weight(params[f"{layer_name}{proj_name}.weight"], transpose=True)

    def save_attention(layer_name):
        save_weight(params[f"{layer_name}input_layernorm.weight"], is_rms=True)
        save_projection(layer_name, "self_attn.q_proj")
        if f"{layer_name}self_attn.q_norm.weight" in params:
            save_weight(params[f"{layer_name}self_attn.q_norm.weight"], is_rms=True)
        save_projection(layer_name, "self_attn.k_proj")
        if f"{layer_name}self_attn.k_norm.weight" in params:
            save_weight(params[f"{layer_name}self_attn.k_norm.weight"], is_rms=True)
        save_projection(layer_name, "self_attn.v_proj")
        save_projection(layer_name, "self_attn.o_proj")

    def save_feed_forward(layer_name):
        save_weight(params[f"{layer_name}post_attention_layernorm.weight"], is_rms=True)
        save_weight(params[f"{layer_name}pre_feedforward_layernorm.weight"], is_rms=True)
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            save_projection(layer_name, f"mlp.{proj}")
        save_weight(params[f"{layer_name}post_feedforward_layernorm.weight"], is_rms=True)

    save_weight(params["model.embed_tokens.weight"])

    for layer_idx in range(n_layers):
        layer_prefix = f"model.layers.{layer_idx}."
        save_attention(layer_prefix)
        save_feed_forward(layer_prefix)

    save_weight(params["model.norm.weight"], is_rms=True)
    if save_lm_head and "lm_head.weight" in params:
        save_weight(params["lm_head.weight"], transpose=True)


def collect_gemma3_for_nntrainer(params, config, dtype, save_lm_head=True):
    """Collect weights as ordered (nntrainer_name, ndarray) pairs for safetensors."""
    n_layers = config.num_hidden_layers
    weights = []

    def add(name, tensor, is_rms=False, transpose=False):
        arr = tensor_to_numpy(tensor, dtype, transpose=transpose, is_rms=is_rms)
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

        # Note: gemma3 cpp uses names without underscore: "layer{N}pre_ffn_norm" / "layer{N}post_ffn_norm".
        add(
            f"{nntr_prefix}_attention_norm:gamma",
            params[f"{hf_prefix}input_layernorm.weight"],
            is_rms=True,
        )

        add_projection(f"{nntr_prefix}_wq:weight", hf_prefix, "self_attn.q_proj")
        if f"{hf_prefix}self_attn.q_norm.weight" in params:
            add(
                f"{nntr_prefix}_q_norm:gamma",
                params[f"{hf_prefix}self_attn.q_norm.weight"],
                is_rms=True,
            )

        add_projection(f"{nntr_prefix}_wk:weight", hf_prefix, "self_attn.k_proj")
        if f"{hf_prefix}self_attn.k_norm.weight" in params:
            add(
                f"{nntr_prefix}_k_norm:gamma",
                params[f"{hf_prefix}self_attn.k_norm.weight"],
                is_rms=True,
            )

        add_projection(f"{nntr_prefix}_wv:weight", hf_prefix, "self_attn.v_proj")
        add_projection(
            f"{nntr_prefix}_attention_out:weight", hf_prefix, "self_attn.o_proj"
        )

        add(
            f"{nntr_prefix}_post_attention_norm:gamma",
            params[f"{hf_prefix}post_attention_layernorm.weight"],
            is_rms=True,
        )
        add(
            f"{nntr_prefix}pre_ffn_norm:gamma",
            params[f"{hf_prefix}pre_feedforward_layernorm.weight"],
            is_rms=True,
        )

        add_projection(f"{nntr_prefix}_ffn_gate:weight", hf_prefix, "mlp.gate_proj")
        add_projection(f"{nntr_prefix}_ffn_up:weight", hf_prefix, "mlp.up_proj")
        add_projection(f"{nntr_prefix}_ffn_down:weight", hf_prefix, "mlp.down_proj")

        add(
            f"{nntr_prefix}post_ffn_norm:gamma",
            params[f"{hf_prefix}post_feedforward_layernorm.weight"],
            is_rms=True,
        )

    add("output_norm:gamma", params["model.norm.weight"], is_rms=True)
    if save_lm_head and "lm_head.weight" in params:
        add("output_of_causallm:weight", params["lm_head.weight"], transpose=True)

    return weights


def save_embedding_gemma_for_nntrainer(model, dtype, file):
    params = model.state_dict()
    gemma_params = {
        key.removeprefix("0.auto_model.").removeprefix("0."): value
        for key, value in params.items()
        if key.startswith("0.")
    }

    save_gemma3_for_nntrainer(
        gemma_params, model[0].auto_model.config, dtype, file, save_lm_head=False
    )

    def save_weight(weight, transpose=False):
        arr = tensor_to_numpy(weight, dtype, transpose=transpose)
        arr.tofile(file)

    for module_name, module in model._modules.items():
        component = module.__class__.__name__
        if component in ["Transformer", "Pooling", "Normalize"]:
            continue
        if component != "Dense":
            raise NotImplementedError(
                f"Unsupported SentenceTransformer module: {module_name} ({component})"
            )

        save_weight(params[f"{module_name}.linear.weight"], transpose=True)
        bias_key = f"{module_name}.linear.bias"
        if bias_key in params:
            save_weight(params[bias_key])


def collect_embedding_gemma_for_nntrainer(model, dtype):
    params = model.state_dict()
    gemma_params = {
        key.removeprefix("0.auto_model.").removeprefix("0."): value
        for key, value in params.items()
        if key.startswith("0.")
    }

    weights = collect_gemma3_for_nntrainer(
        gemma_params, model[0].auto_model.config, dtype, save_lm_head=False
    )

    dense_idx = 0
    for module_name, module in model._modules.items():
        component = module.__class__.__name__
        if component in ["Transformer", "Pooling", "Normalize"]:
            continue
        if component != "Dense":
            raise NotImplementedError(
                f"Unsupported SentenceTransformer module: {module_name} ({component})"
            )

        dense_weight = params[f"{module_name}.linear.weight"]
        weights.append(
            (
                f"dense{dense_idx}:weight",
                tensor_to_numpy(dense_weight, dtype, transpose=True),
            )
        )
        bias_key = f"{module_name}.linear.bias"
        if bias_key in params:
            weights.append(
                (f"dense{dense_idx}:bias", tensor_to_numpy(params[bias_key], dtype))
            )
        dense_idx += 1

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
    parser.add_argument("--model_type", choices=["causal", "embedding"], default="causal")
    parser.add_argument("--model_path", type=str, default="./270m")
    parser.add_argument("--output_name", type=str, default="./nntr_gemma3_270m_fp32.bin")
    parser.add_argument("--data_type", choices=["float32", "float16"], default="float32")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--safetensors",
        action="store_true",
        help="Save weights in safetensors format instead of binary format",
    )
    args = parser.parse_args()

    data_dtype = args.data_type
    model_path = args.model_path
    output_name = args.output_name

    if args.model_type == "causal":
        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=args.trust_remote_code
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32, trust_remote_code=args.trust_remote_code
        )
        model.eval()
        print(model)

        save_lm_head = not getattr(config, "tie_word_embeddings", False)

        if args.safetensors:
            safetensors_path = get_safetensors_output_name(output_name)
            weights = collect_gemma3_for_nntrainer(
                model.state_dict(), config, data_dtype, save_lm_head=save_lm_head
            )
            save_safetensors(weights, safetensors_path, data_dtype)
        else:
            with open(output_name, "wb") as f_model:
                save_gemma3_for_nntrainer(
                    model.state_dict(),
                    config,
                    data_dtype,
                    f_model,
                    save_lm_head=save_lm_head,
                )
            print(f"Saved binary: {output_name}")
    else:
        from sentence_transformers import SentenceTransformer

        torch_dtype = torch.float16 if data_dtype == "float16" else torch.float32
        model = SentenceTransformer(
            model_path,
            trust_remote_code=args.trust_remote_code,
            model_kwargs={"torch_dtype": torch_dtype},
        )
        model.eval()

        if args.safetensors:
            safetensors_path = get_safetensors_output_name(output_name)
            weights = collect_embedding_gemma_for_nntrainer(model, data_dtype)
            save_safetensors(weights, safetensors_path, data_dtype)
        else:
            with open(output_name, "wb") as f_model:
                save_embedding_gemma_for_nntrainer(model, data_dtype, f_model)
            print(f"Saved binary: {output_name}")
