#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gemma4AssistantForCausalLM (the MTP drafter) -> nntrainer FP32 pack.

The pack is what Gemma4AssistantCausalLM loads and what nntr_quantize
quantizes for the CPU drafter (int4 per-channel FC, Q4_0 head):

  python3 assistant_weight_converter.py --source <HF dir> --output <dir>
  nntr_quantize <dir> --fc_dtype QS4CX --lmhead_dtype Q4_0

The output directory must not exist: nothing is overwritten.
"""

import argparse
import json
from pathlib import Path
import struct

import numpy as np

MODEL_FILE = "nntr_gemma4_assistant_fp32.bin"


def layer_weights(i):
    """HF names of layer i in Gemma4AssistantCausalLM's execution order, and
    whether the tensor is an FC weight (HF [out, in], nntrainer [in, out])."""
    p = f"model.layers.{i}."
    return [(p + "input_layernorm.weight", False),
            (p + "self_attn.q_proj.weight", True),
            (p + "self_attn.q_norm.weight", False),
            (p + "self_attn.o_proj.weight", True),
            (p + "post_attention_layernorm.weight", False),
            (p + "pre_feedforward_layernorm.weight", False),
            (p + "mlp.gate_proj.weight", True),
            (p + "mlp.up_proj.weight", True),
            (p + "mlp.down_proj.weight", True),
            (p + "post_feedforward_layernorm.weight", False),
            (p + "layer_scalar", False)]


def weight_order(n_layers):
    """Every tensor of the pack, in the order nntrainer reads them."""
    order = [("pre_projection.weight", True)]
    for i in range(n_layers):
        order += layer_weights(i)
    # the tied head: the table is the head, written [hidden, vocab]
    return order + [("model.norm.weight", False),
                    ("model.embed_tokens.weight", True),
                    ("post_projection.weight", True)]


def write_weights(tensors, n_layers, stream):
    """tensors: HF name -> float32 ndarray. Returns the bytes written."""
    total = 0
    if "model.embed_tokens.weight" not in tensors and "lm_head.weight" in tensors:
        tensors["model.embed_tokens.weight"] = tensors["lm_head.weight"]
    for name, fc in weight_order(n_layers):
        if name not in tensors:
            raise KeyError(f"checkpoint has no {name}")
        w = np.ascontiguousarray(tensors[name].T if fc else tensors[name],
                                 dtype=np.float32)
        stream.write(w.tobytes())
        total += w.nbytes
    return total


def read_safetensors(root):
    """Every *.safetensors file of root as float32, without torch."""
    out = {}
    for path in sorted(root.glob("*.safetensors")):
        with path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(size))
            body = stream.read()
        for name, e in header.items():
            if name == "__metadata__":
                continue
            raw = body[e["data_offsets"][0]:e["data_offsets"][1]]
            if e["dtype"] == "BF16":
                a = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
            elif e["dtype"] == "F32":
                a = np.frombuffer(raw, np.float32)
            else:
                raise ValueError(f"{name}: unsupported dtype {e['dtype']}")
            out[name] = a.reshape(e["shape"])
    return out


def model_config(config):
    """config.json for nntrainer. transformers >= 5 writes the full layers'
    head_dim and kv heads as per_layer_config; Gemma4Transformer reads
    global_head_dim and num_global_key_value_heads."""
    text = config["text_config"]
    for entry in (text.pop("per_layer_config", None) or {}).values():
        text["global_head_dim"] = entry.get("head_dim", text.get("global_head_dim"))
        if "num_key_value_heads" in entry:
            text["num_global_key_value_heads"] = entry["num_key_value_heads"]
    return config


def nntr_config(model_file):
    return {"model_type": "CausalLM", "model_tensor_type": "FP32-FP32",
            "model_file_name": model_file, "fc_layer_dtype": "FP32",
            "embedding_dtype": "FP32", "lmhead_dtype": "FP32",
            "batch_size": 1, "init_seq_len": 1, "max_seq_len": 1,
            "num_to_generate": 1, "bad_word_ids": []}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="HF assistant dir")
    parser.add_argument("--output", type=Path, required=True, help="new dir (must not exist)")
    args = parser.parse_args()

    config = json.loads((args.source / "config.json").read_text())
    if config.get("use_ordered_embeddings", False):
        raise SystemExit("use_ordered_embeddings (E2B/E4B centroid head) is not supported")
    n_layers = config["text_config"]["num_hidden_layers"]
    args.output.mkdir(parents=True)
    with (args.output / MODEL_FILE).open("wb") as stream:
        size = write_weights(read_safetensors(args.source), n_layers, stream)
    (args.output / "config.json").write_text(
        json.dumps(model_config(config), indent=2) + "\n")
    gen = args.source / "generation_config.json"
    (args.output / "generation_config.json").write_text(
        gen.read_text() if gen.exists() else "{}\n")
    (args.output / "nntr_config.json").write_text(
        json.dumps(nntr_config(MODEL_FILE), indent=2) + "\n")
    print(f"{args.output / MODEL_FILE}: {size} bytes")


if __name__ == "__main__":
    main()
