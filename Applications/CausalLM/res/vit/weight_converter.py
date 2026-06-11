## SPDX-License-Identifier: Apache-2.0
## Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
##
## @file weight_converter.py
## @brief weight conversion script for vit model (timm version)
## @author SeungBaek Hong <sb92.hong@samsung.com>

import argparse
import json
import struct

import numpy as np
import safetensors.torch
import torch


SAFETENSORS_DTYPE_MAP = {
    np.dtype("float32"): "F32",
    np.dtype("float16"): "F16",
}


def load_state_dict(model_path):
    """Load a timm checkpoint from safetensors or PyTorch bin format."""
    if model_path.endswith(".safetensors"):
        state_dict = safetensors.torch.load_file(model_path)
    else:
        state_dict = torch.load(model_path, map_location="cpu")

    if isinstance(state_dict, dict):
        for key in ("state_dict", "model"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break

    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def _to_numpy(weight, dtype, transpose=False):
    if isinstance(weight, np.ndarray):
        array = weight
    else:
        array = weight.detach().cpu().numpy()
    if transpose and array.ndim >= 2:
        array = array.T
    return np.ascontiguousarray(array.astype(dtype))


def save_weight(weight, dtype, file, transpose=False):
    """Save weight tensor to nntrainer binary format."""
    _to_numpy(weight, dtype, transpose=transpose).tofile(file)


def convert_timm_vit_to_nntrainer(model_path, output_path, dtype=np.float32):
    """Convert timm ViT weights to nntrainer binary format."""

    print(f"Loading model from: {model_path}")
    state_dict = load_state_dict(model_path)

    print(f"Converting weights to: {output_path}")

    with open(output_path, "wb") as f:
        print("  Processing patch embedding...")
        save_weight(state_dict["patch_embed.proj.weight"], dtype, f)
        if "patch_embed.proj.bias" in state_dict:
            save_weight(state_dict["patch_embed.proj.bias"], dtype, f)

        print("  Processing position embedding...")
        pos_embed_reshaped = state_dict["pos_embed"].unsqueeze(1)
        save_weight(pos_embed_reshaped, dtype, f)

        num_layers = 12
        print(f"  Processing {num_layers} transformer blocks...")
        for i in range(num_layers):
            layer_prefix = f"blocks.{i}."

            save_weight(state_dict[layer_prefix + "norm1.weight"], dtype, f)
            save_weight(state_dict[layer_prefix + "norm1.bias"], dtype, f)

            qkv_weight = state_dict[layer_prefix + "attn.qkv.weight"]
            qkv_bias = state_dict[layer_prefix + "attn.qkv.bias"]

            dim = 768
            Q_weight = qkv_weight[:dim, :]
            K_weight = qkv_weight[dim : 2 * dim, :]
            V_weight = qkv_weight[2 * dim :, :]

            Q_bias = qkv_bias[:dim]
            K_bias = qkv_bias[dim : 2 * dim]
            V_bias = qkv_bias[2 * dim :]

            save_weight(Q_weight, dtype, f, transpose=True)
            save_weight(Q_bias, dtype, f)
            save_weight(K_weight, dtype, f, transpose=True)
            save_weight(K_bias, dtype, f)
            save_weight(V_weight, dtype, f, transpose=True)
            save_weight(V_bias, dtype, f)

            save_weight(state_dict[layer_prefix + "attn.proj.weight"], dtype, f, transpose=True)
            save_weight(state_dict[layer_prefix + "attn.proj.bias"], dtype, f)

            save_weight(state_dict[layer_prefix + "norm2.weight"], dtype, f)
            save_weight(state_dict[layer_prefix + "norm2.bias"], dtype, f)

            save_weight(state_dict[layer_prefix + "mlp.fc1.weight"], dtype, f, transpose=True)
            save_weight(state_dict[layer_prefix + "mlp.fc1.bias"], dtype, f)
            save_weight(state_dict[layer_prefix + "mlp.fc2.weight"], dtype, f, transpose=True)
            save_weight(state_dict[layer_prefix + "mlp.fc2.bias"], dtype, f)

        print("  Processing final normalization...")
        save_weight(state_dict["norm.weight"], dtype, f)
        save_weight(state_dict["norm.bias"], dtype, f)

        if "attn_pool.latent" in state_dict:
            print("  Processing attention pool...")
            save_weight(state_dict["attn_pool.latent"], dtype, f)
            save_weight(state_dict["attn_pool.q.weight"], dtype, f, transpose=True)
            save_weight(state_dict["attn_pool.q.bias"], dtype, f)
            save_weight(state_dict["attn_pool.kv.weight"], dtype, f, transpose=True)
            save_weight(state_dict["attn_pool.kv.bias"], dtype, f)
            save_weight(state_dict["attn_pool.proj.weight"], dtype, f, transpose=True)
            save_weight(state_dict["attn_pool.proj.bias"], dtype, f)
            save_weight(state_dict["attn_pool.norm.weight"], dtype, f)
            save_weight(state_dict["attn_pool.norm.bias"], dtype, f)
            save_weight(state_dict["attn_pool.mlp.fc1.weight"], dtype, f, transpose=True)
            save_weight(state_dict["attn_pool.mlp.fc1.bias"], dtype, f)
            save_weight(state_dict["attn_pool.mlp.fc2.weight"], dtype, f, transpose=True)
            save_weight(state_dict["attn_pool.mlp.fc2.bias"], dtype, f)

        print("  Skipping head (num_classes=0)")

    print(f"\nConversion complete!")
    print(f"  Total weights saved to: {output_path}")


def collect_timm_vit_for_nntrainer(state_dict, dtype, num_layers=12, dim=768):
    """Collect weights as ordered (nntrainer_name, ndarray) pairs for safetensors."""
    weights = []

    def add(name, tensor, transpose=False):
        weights.append((name, _to_numpy(tensor, dtype, transpose=transpose)))

    # Patch embedding (Conv2D layer "patch_embed/conv")
    add("patch_embed/conv:weight", state_dict["patch_embed.proj.weight"])
    if "patch_embed.proj.bias" in state_dict:
        add("patch_embed/conv:bias", state_dict["patch_embed.proj.bias"])

    # Position embedding tensor (created via the "weight" layer "pos_embed/weights").
    add("pos_embed/weights:weight", state_dict["pos_embed"].unsqueeze(1))

    for i in range(num_layers):
        layer_prefix = f"blocks.{i}."
        nntr_prefix = f"layer{i}_"

        add(f"{nntr_prefix}attention_norm:gamma", state_dict[layer_prefix + "norm1.weight"])
        add(f"{nntr_prefix}attention_norm:beta", state_dict[layer_prefix + "norm1.bias"])

        qkv_weight = state_dict[layer_prefix + "attn.qkv.weight"]
        qkv_bias = state_dict[layer_prefix + "attn.qkv.bias"]

        Q_weight = qkv_weight[:dim, :]
        K_weight = qkv_weight[dim : 2 * dim, :]
        V_weight = qkv_weight[2 * dim :, :]
        Q_bias = qkv_bias[:dim]
        K_bias = qkv_bias[dim : 2 * dim]
        V_bias = qkv_bias[2 * dim :]

        add(f"{nntr_prefix}qkv_q:weight", Q_weight, transpose=True)
        add(f"{nntr_prefix}qkv_q:bias", Q_bias)
        add(f"{nntr_prefix}qkv_k:weight", K_weight, transpose=True)
        add(f"{nntr_prefix}qkv_k:bias", K_bias)
        add(f"{nntr_prefix}qkv_v:weight", V_weight, transpose=True)
        add(f"{nntr_prefix}qkv_v:bias", V_bias)

        add(f"{nntr_prefix}attention_out:weight", state_dict[layer_prefix + "attn.proj.weight"], transpose=True)
        add(f"{nntr_prefix}attention_out:bias", state_dict[layer_prefix + "attn.proj.bias"])

        add(f"{nntr_prefix}ffn_norm:gamma", state_dict[layer_prefix + "norm2.weight"])
        add(f"{nntr_prefix}ffn_norm:beta", state_dict[layer_prefix + "norm2.bias"])

        add(f"{nntr_prefix}ffn_up:weight", state_dict[layer_prefix + "mlp.fc1.weight"], transpose=True)
        add(f"{nntr_prefix}ffn_up:bias", state_dict[layer_prefix + "mlp.fc1.bias"])
        add(f"{nntr_prefix}ffn_down:weight", state_dict[layer_prefix + "mlp.fc2.weight"], transpose=True)
        add(f"{nntr_prefix}ffn_down:bias", state_dict[layer_prefix + "mlp.fc2.bias"])

    add("output_norm:gamma", state_dict["norm.weight"])
    add("output_norm:beta", state_dict["norm.bias"])

    return weights


def save_safetensors(weights, output_path, np_dtype):
    if np_dtype not in SAFETENSORS_DTYPE_MAP:
        raise ValueError(f"Unsupported safetensors dtype: {np_dtype}")

    safetensors_dtype = SAFETENSORS_DTYPE_MAP[np_dtype]
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


def get_safetensors_output_name(output_name):
    if output_name.endswith(".bin"):
        return output_name[:-4] + ".safetensors"
    if output_name.endswith(".safetensors"):
        return output_name
    return output_name + ".safetensors"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="model.safetensors",
                        help="Input model path (.safetensors or .bin)")
    parser.add_argument("--output", type=str, default="nntr_vit_base_fp32.bin",
                        help="Output nntrainer weight file")
    parser.add_argument("--dtype", type=str, default="float32",
                        help="Data type (float32 or float16)")
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument(
        "--safetensors",
        action="store_true",
        help="Save weights in safetensors format instead of binary format",
    )
    args = parser.parse_args()

    dtype_map = {"float32": np.float32, "float16": np.float16}
    dtype = dtype_map.get(args.dtype, np.float32)

    if args.safetensors:
        state_dict = load_state_dict(args.input)
        output_path = get_safetensors_output_name(args.output)
        weights = collect_timm_vit_for_nntrainer(
            state_dict, dtype, num_layers=args.num_layers, dim=args.dim
        )
        save_safetensors(weights, output_path, np.dtype(dtype))
    else:
        convert_timm_vit_to_nntrainer(args.input, args.output, dtype)
