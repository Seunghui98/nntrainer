# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>

## @file weight_converter.py
## @brief weight conversion script for Ouro Universal-Transformer model.
##
## Ouro is a Universal-Transformer style decoder. The HuggingFace state_dict
## contains exactly one set of decoder weights (model.layers.0..N-1); the loop
## over `total_ut_steps` is structural, not parametric. nntrainer's
## OuroTransformer ties every UT step k>0 back to step 0 through `shared_from`,
## so this converter only needs to emit the step-0 names. The shared_from
## tensors are resolved from the weight pool at graph init time.

import argparse
import json
import os
import struct
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM


SAFETENSORS_DTYPE_MAP = {
    "float32": "F32",
}


def tensor_to_numpy(tensor, dtype, transpose=False):
    """Convert a torch tensor to a contiguous numpy array, optionally transposed."""
    if transpose:
        tensor = tensor.permute(1, 0)
    return np.ascontiguousarray(tensor.detach().cpu().numpy().astype(dtype))


def get_safetensors_output_name(output_name):
    if output_name.endswith(".bin"):
        return output_name[:-4] + ".safetensors"
    if output_name.endswith(".safetensors"):
        return output_name
    return output_name + ".safetensors"


def _maybe_get(params: Dict[str, torch.Tensor], *candidates: str):
    for key in candidates:
        if key in params:
            return params[key]
    return None


def collect_ouro_for_nntrainer(
    params: Dict[str, torch.Tensor],
    n_layers: int,
    dtype: str,
    tie_word_embeddings: bool,
) -> List[Tuple[str, np.ndarray]]:
    """Collect weights as ordered (nntrainer_name, ndarray) pairs.

    Only step-0 layer names are emitted (layer{i}_<role>). UT step k>0 names
    (layer{i}_ut{k}_<role>) are resolved through nntrainer's shared_from at
    graph init, so we do not duplicate the tensors here.
    """
    weights: List[Tuple[str, np.ndarray]] = []

    # AutoModelForCausalLM wraps OuroModel under self.model, so keys are
    # "model.<...>". AutoModel returns OuroModel directly, so keys are just
    # "<...>". Detect once and reuse.
    if "model.embed_tokens.weight" in params:
        base = "model."
    elif "embed_tokens.weight" in params:
        base = ""
    else:
        sample_keys = list(params.keys())[:5]
        raise KeyError(
            "Could not locate embed_tokens.weight in the state_dict. "
            f"First keys: {sample_keys}"
        )

    def must_get(key: str) -> torch.Tensor:
        full = base + key
        if full not in params:
            raise KeyError(f"Missing key: {full}")
        return params[full]

    def add(name: str, tensor: torch.Tensor, transpose: bool = False) -> None:
        weights.append((name, tensor_to_numpy(tensor, dtype, transpose=transpose)))

    # 1. Embedding: HF [vocab, embed_dim=intermediate_size] -> nntrainer same layout
    add("embedding0:Embedding", must_get("embed_tokens.weight"))

    # 2. embed_projection: HF Linear(embed_dim, hidden_size) -> transpose for FC
    add("embed_projection:weight", must_get("embed_projection.weight"),
        transpose=True)

    # 3. Per-layer decoder weights (step 0 names only)
    for layer_idx in range(n_layers):
        prefix_hf = f"layers.{layer_idx}."
        prefix_nn = f"layer{layer_idx}"

        # Attention pre-norm + post-attention norm (Ouro extra)
        add(
            f"{prefix_nn}_attention_norm:gamma",
            must_get(f"{prefix_hf}input_layernorm.weight"),
        )

        # Q/K/V/O projections
        add(
            f"{prefix_nn}_wq:weight",
            must_get(f"{prefix_hf}self_attn.q_proj.weight"),
            transpose=True,
        )
        add(
            f"{prefix_nn}_wk:weight",
            must_get(f"{prefix_hf}self_attn.k_proj.weight"),
            transpose=True,
        )
        add(
            f"{prefix_nn}_wv:weight",
            must_get(f"{prefix_hf}self_attn.v_proj.weight"),
            transpose=True,
        )
        add(
            f"{prefix_nn}_attention_out:weight",
            must_get(f"{prefix_hf}self_attn.o_proj.weight"),
            transpose=True,
        )
        add(
            f"{prefix_nn}_attention_norm_2:gamma",
            must_get(f"{prefix_hf}input_layernorm_2.weight"),
        )

        # FFN pre-norm + post-FFN norm (Ouro extra)
        add(
            f"{prefix_nn}_ffn_norm:gamma",
            must_get(f"{prefix_hf}post_attention_layernorm.weight"),
        )

        # SwiGLU MLP: gate / up / down
        add(
            f"{prefix_nn}_ffn_up:weight",
            must_get(f"{prefix_hf}mlp.up_proj.weight"),
            transpose=True,
        )
        add(
            f"{prefix_nn}_ffn_gate:weight",
            must_get(f"{prefix_hf}mlp.gate_proj.weight"),
            transpose=True,
        )
        add(
            f"{prefix_nn}_ffn_down:weight",
            must_get(f"{prefix_hf}mlp.down_proj.weight"),
            transpose=True,
        )
        add(
            f"{prefix_nn}_ffn_norm_2:gamma",
            must_get(f"{prefix_hf}post_attention_layernorm_2.weight"),
        )

    # 4. Final RMSNorm (applied at the end of each UT step; step 0 name)
    add("output_norm:gamma", must_get("norm.weight"))

    # 5. LM head (only when not tied). lm_head lives at the top level even when
    #    AutoModelForCausalLM is used.
    if not tie_word_embeddings:
        lm_head = _maybe_get(params, "lm_head.weight")
        if lm_head is not None:
            add("output_of_causallm:weight", lm_head, transpose=True)

    # Note: early_exit_gate is intentionally skipped — the C++ Ouro graph does
    # not use it for embedding/causalLM inference.

    return weights


def save_safetensors(weights, output_path, dtype):
    if dtype not in SAFETENSORS_DTYPE_MAP:
        raise ValueError(f"Unsupported safetensors dtype: {dtype}")

    safetensors_dtype = SAFETENSORS_DTYPE_MAP[dtype]
    metadata = {"format": "pt"}

    offset = 0
    tensor_meta: Dict[str, Any] = {}
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

    with open(output_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for buf in raw_buffers:
            f.write(buf)

    print(f"Saved safetensors: {output_path}")
    print(f"Tensor data size : {offset / 1e9:.2f} GB")


def save_binary(weights, output_path):
    with open(output_path, "wb") as f:
        for _, arr in weights:
            arr.tofile(f)
    print(f"Saved binary: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Local path to the Ouro HuggingFace model (config.json + weights).",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="./nntr_ouro_fp32.safetensors",
        help="Output weight file path. Suffix selects format (.safetensors or .bin).",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="float32",
        choices=["float32"],
        help="Output data type.",
    )
    parser.add_argument(
        "--causallm",
        action="store_true",
        help="Load with AutoModelForCausalLM (default loads with AutoModel for "
             "the OuroModel embedding architecture).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    loader = AutoModelForCausalLM if args.causallm else AutoModel
    model = loader.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    tie_word_embeddings = bool(getattr(config, "tie_word_embeddings", False))
    n_layers = int(getattr(config, "num_hidden_layers"))
    total_ut_steps = int(getattr(config, "total_ut_steps", 1))
    print(f"num_hidden_layers   : {n_layers}")
    print(f"total_ut_steps      : {total_ut_steps} (resolved via shared_from)")
    print(f"tie_word_embeddings : {tie_word_embeddings}")

    params = model.state_dict()
    weights = collect_ouro_for_nntrainer(
        params, n_layers, args.data_type, tie_word_embeddings
    )

    output_name = args.output_name
    if output_name.endswith(".bin"):
        save_binary(weights, output_name)
    else:
        save_safetensors(
            weights, get_safetensors_output_name(output_name), args.data_type
        )


if __name__ == "__main__":
    main()
