# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

# @file weight_converter.py
# @brief Weight conversion script for multilingual TinyBERT sentence embeddings.
# @author Samsung Electronics Co., Ltd.

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModel


SAFETENSORS_DTYPE_MAP = {
    np.dtype("float32"): "F32",
    np.dtype("float16"): "F16",
}


def _to_numpy(weight, dtype, transpose=False):
    if transpose:
        weight = weight.transpose(0, 1)
    return np.ascontiguousarray(weight.detach().cpu().numpy().astype(dtype))


def _save_weight(file, weight, dtype, transpose=False):
    _to_numpy(weight, dtype, transpose=transpose).tofile(file)


def _save_linear(file, params, prefix, dtype):
    """Save FC weight (transposed) and bias for nntrainer."""
    _save_weight(file, params[f"{prefix}.weight"], dtype, transpose=True)
    _save_weight(file, params[f"{prefix}.bias"], dtype)


def save_tinybert_for_nntrainer(params, config, dtype, file):
    """Save TinyBERT encoder weights in nntrainer layer order (binary)."""
    # Token / position / token_type embeddings + embedding LayerNorm.
    _save_weight(file, params["embeddings.word_embeddings.weight"], dtype)
    _save_weight(file, params["embeddings.position_embeddings.weight"], dtype)
    _save_weight(file, params["embeddings.token_type_embeddings.weight"], dtype)
    _save_weight(file, params["embeddings.LayerNorm.weight"], dtype)
    _save_weight(file, params["embeddings.LayerNorm.bias"], dtype)

    for layer_idx in range(config.num_hidden_layers):
        layer_prefix = f"encoder.layer.{layer_idx}"
        attn_prefix = f"{layer_prefix}.attention"
        self_prefix = f"{attn_prefix}.self"

        _save_linear(file, params, f"{self_prefix}.query", dtype)
        _save_linear(file, params, f"{self_prefix}.key", dtype)
        _save_linear(file, params, f"{self_prefix}.value", dtype)
        _save_linear(file, params, f"{attn_prefix}.output.dense", dtype)
        _save_weight(file, params[f"{attn_prefix}.output.LayerNorm.weight"], dtype)
        _save_weight(file, params[f"{attn_prefix}.output.LayerNorm.bias"], dtype)

        _save_linear(file, params, f"{layer_prefix}.intermediate.dense", dtype)
        _save_linear(file, params, f"{layer_prefix}.output.dense", dtype)
        _save_weight(file, params[f"{layer_prefix}.output.LayerNorm.weight"], dtype)
        _save_weight(file, params[f"{layer_prefix}.output.LayerNorm.bias"], dtype)


def collect_tinybert_for_nntrainer(params, config, dtype):
    """Collect weights as ordered (nntrainer_name, ndarray) pairs for safetensors."""
    weights = []

    def add(name, tensor, transpose=False):
        weights.append((name, _to_numpy(tensor, dtype, transpose=transpose)))

    def add_linear(nntr_name, hf_prefix):
        add(f"{nntr_name}:weight", params[f"{hf_prefix}.weight"], transpose=True)
        add(f"{nntr_name}:bias", params[f"{hf_prefix}.bias"])

    add("embedding0:Embedding", params["embeddings.word_embeddings.weight"])
    add("position_embedding:Embedding", params["embeddings.position_embeddings.weight"])
    add("token_type_embedding:Embedding", params["embeddings.token_type_embeddings.weight"])
    add("embedding_norm:gamma", params["embeddings.LayerNorm.weight"])
    add("embedding_norm:beta", params["embeddings.LayerNorm.bias"])

    for layer_idx in range(config.num_hidden_layers):
        layer_prefix = f"encoder.layer.{layer_idx}"
        attn_prefix = f"{layer_prefix}.attention"
        self_prefix = f"{attn_prefix}.self"
        nntr_prefix = f"layer{layer_idx}"

        add_linear(f"{nntr_prefix}_wq", f"{self_prefix}.query")
        add_linear(f"{nntr_prefix}_wk", f"{self_prefix}.key")
        add_linear(f"{nntr_prefix}_wv", f"{self_prefix}.value")
        add_linear(f"{nntr_prefix}_attention_out", f"{attn_prefix}.output.dense")
        add(
            f"{nntr_prefix}_attention_norm:gamma",
            params[f"{attn_prefix}.output.LayerNorm.weight"],
        )
        add(
            f"{nntr_prefix}_attention_norm:beta",
            params[f"{attn_prefix}.output.LayerNorm.bias"],
        )
        add_linear(f"{nntr_prefix}_ffn_fc1", f"{layer_prefix}.intermediate.dense")
        add_linear(f"{nntr_prefix}_ffn_down", f"{layer_prefix}.output.dense")
        add(
            f"{nntr_prefix}_ffn_norm:gamma",
            params[f"{layer_prefix}.output.LayerNorm.weight"],
        )
        add(
            f"{nntr_prefix}_ffn_norm:beta",
            params[f"{layer_prefix}.output.LayerNorm.bias"],
        )

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


def _strip_bert_prefix(params):
    """Strip optional SentenceTransformer / encoder-only prefixes from keys."""
    if "embeddings.word_embeddings.weight" in params:
        return params

    prefixes = (
        "0.auto_model.",
        "auto_model.",
        "bert.",
    )
    for prefix in prefixes:
        key = prefix + "embeddings.word_embeddings.weight"
        if key in params:
            return {
                name[len(prefix):]: value
                for name, value in params.items()
                if name.startswith(prefix)
            }
    raise KeyError("Cannot find BERT encoder weights in state_dict")


def _load_model_config_and_params(model_path):
    try:
        model = AutoModel.from_pretrained(model_path)
        params = model.state_dict()
        if "bert.embeddings.word_embeddings.weight" in params:
            params = _strip_bert_prefix(params)
        return model.config, _strip_bert_prefix(params)
    except Exception:
        from sentence_transformers import SentenceTransformer

        st_model = SentenceTransformer(model_path, trust_remote_code=True)
        auto_model = getattr(st_model[0], "auto_model", None)
        if auto_model is None:
            raise AttributeError("first SentenceTransformer module has no auto_model")
        return auto_model.config, _strip_bert_prefix(st_model.state_dict())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="huawei-noah/TinyBERT_General_4L_312D",
        help="Hugging Face model directory or hub id",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="./tinybert_fp32.bin",
        help="Output nntrainer weight path",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="float32",
        choices=["float32", "float16"],
    )
    parser.add_argument(
        "--safetensors",
        action="store_true",
        help="Save weights in safetensors format instead of binary format",
    )
    args = parser.parse_args()

    dtype = np.float16 if args.data_type == "float16" else np.float32
    config, params = _load_model_config_and_params(args.model_path)

    if args.safetensors:
        output_path = Path(get_safetensors_output_name(args.output_name))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            weights = collect_tinybert_for_nntrainer(params, config, dtype)
        save_safetensors(weights, str(output_path), np.dtype(dtype))
        return

    output_path = Path(args.output_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), output_path.open("wb") as output_file:
        save_tinybert_for_nntrainer(params, config, dtype, output_file)
    print(f"Saved binary: {output_path}")


if __name__ == "__main__":
    main()
