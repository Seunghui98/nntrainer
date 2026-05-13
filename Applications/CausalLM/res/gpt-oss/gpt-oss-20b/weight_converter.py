"""
SPDX-License-Identifier: Apache-2.0
Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>

@file weight_converter.py
@brief gpt-oss-20b weight converter — supports both nntrainer binary and
       safetensors output from a single model load.
@author Eunju Yang <ej.yang@samsung.com>

Tested with transformers 4.53.2.

gpt-oss specifics
=================
  - MoE FFN: the C++ `gpt_oss_moe` layer is named `layer{i}_ffn_down` and
    registers these weights (per expert e, with bias on each):
        layer{i}_ffn_down:gate           (router, transposed)
        layer{i}_ffn_down:gate_bias
        layer{i}_ffn_down:expert_up_{e},   :expert_up_bias_{e}
        layer{i}_ffn_down:expert_gate_{e}, :expert_gate_bias_{e}
        layer{i}_ffn_down:expert_down_{e}, :expert_down_bias_{e}
    The expert matrices are stored NOT-transposed (PyTorch already keeps
    them in [in, out] orientation after the gate_up_proj split).
  - Attention sink: mha_core registers a per-head `sink` scalar under
    `layer{i}_attention:sink` (`use_sink="true"` in createAttention).
  - Q/K/V/O all carry bias (`disable_bias="false"`).
  - The checkpoint stores experts in MXFP4. We rely on transformers'
    AutoModelForCausalLM.from_pretrained to dequantize the weights into
    regular float tensors; reading safetensors directly would hand us
    raw 4-bit blocks that we would have to dequantize ourselves.
  - gate_up_proj is packed: indices ::2 are the gate columns, 1::2 are
    the up columns. We split before writing.
"""

import argparse
import json
import struct

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _split_gate_up(params, layer_prefix):
    """Return (gate, gate_bias, up, up_bias) tensors per expert.

    gate_up_proj packs gate and up side-by-side along the last axis:
      packed[..., ::2] -> gate columns
      packed[..., 1::2] -> up columns
    Result tensors have shape ``[n_experts, hidden, intermediate]`` for
    the weight and ``[n_experts, intermediate]`` for the bias.
    """
    packed_w = params[f"{layer_prefix}mlp.experts.gate_up_proj"]
    packed_b = params[f"{layer_prefix}mlp.experts.gate_up_proj_bias"]
    return (
        packed_w[..., ::2],
        packed_b[..., ::2],
        packed_w[..., 1::2],
        packed_b[..., 1::2],
    )


# ----------------------------------------------------------------------------
# Binary (sequential) writer
# ----------------------------------------------------------------------------

def save_gpt_oss_for_nntrainer(params, config, dtype, file):
    n_layers = config.num_hidden_layers
    n_experts = config.num_local_experts

    def save_weight(weight_name, is_transpose=False):
        w = params[weight_name].float()
        if is_transpose:
            w = w.permute(1, 0)
        np.array(w.contiguous(), dtype=dtype).tofile(file)

    def save_projection(layer_name, proj_name):
        lora_key = f"{layer_name}{proj_name}.lora_A.default.weight"
        if lora_key in params:
            save_weight(f"{layer_name}{proj_name}.base_layer.weight", True)
            save_weight(f"{layer_name}{proj_name}.lora_A.default.weight", True)
            save_weight(f"{layer_name}{proj_name}.lora_B.default.weight", True)
        else:
            save_weight(f"{layer_name}{proj_name}.weight", True)

    def save_attention(layer_name):
        # Q, K, V: weight then bias (each registered as a single FC layer)
        for proj in ["q_proj", "k_proj", "v_proj"]:
            save_projection(layer_name, f"self_attn.{proj}")
            save_weight(f"{layer_name}self_attn.{proj}.bias")
        # Attention sink (mha_core "sink" weight)
        save_weight(f"{layer_name}self_attn.sinks")
        # Output projection
        save_projection(layer_name, "self_attn.o_proj")
        save_weight(f"{layer_name}self_attn.o_proj.bias")

    def save_feed_forward(layer_name):
        # Router (gate + gate_bias). PyTorch stores router.weight as
        # [n_experts, hidden]; the C++ gate dim is [hidden, n_experts],
        # so transpose.
        save_weight(f"{layer_name}mlp.router.weight", True)
        save_weight(f"{layer_name}mlp.router.bias")

        gate_w, gate_b, up_w, up_b = _split_gate_up(params, layer_name)
        down_w = params[f"{layer_name}mlp.experts.down_proj"]
        down_b = params[f"{layer_name}mlp.experts.down_proj_bias"]

        for e in range(n_experts):
            # The C++ MoE layer registers per-expert weights in this exact
            # order: up, up_bias, gate, gate_bias, down, down_bias.
            np.array(up_w[e].float().contiguous(), dtype=dtype).tofile(file)
            np.array(up_b[e].float().contiguous(), dtype=dtype).tofile(file)
            np.array(gate_w[e].float().contiguous(), dtype=dtype).tofile(file)
            np.array(gate_b[e].float().contiguous(), dtype=dtype).tofile(file)
            np.array(down_w[e].float().contiguous(), dtype=dtype).tofile(file)
            np.array(down_b[e].float().contiguous(), dtype=dtype).tofile(file)

    save_weight("model.embed_tokens.weight")

    for layer_idx in range(n_layers):
        layer_prefix = f"model.layers.{layer_idx}."
        save_weight(f"{layer_prefix}input_layernorm.weight")
        save_attention(layer_prefix)
        save_weight(f"{layer_prefix}post_attention_layernorm.weight")
        save_feed_forward(layer_prefix)

    save_weight("model.norm.weight")
    save_weight("lm_head.weight", True)


# ----------------------------------------------------------------------------
# Safetensors (name-keyed) collector
# ----------------------------------------------------------------------------

def collect_gpt_oss_for_nntrainer(params, config, dtype):
    """Return ``[(nntr_name, ndarray), ...]`` for the safetensors writer."""
    n_layers = config.num_hidden_layers
    n_experts = config.num_local_experts
    weights = []

    def add(name, tensor, transpose=False):
        t = tensor.float()
        if transpose:
            t = t.permute(1, 0)
        t = t.contiguous()
        weights.append((name, t.detach().numpy().astype(dtype)))

    def add_projection(nntr_name, layer_name, proj_name):
        lora_key = f"{layer_name}{proj_name}.lora_A.default.weight"
        if lora_key in params:
            add(nntr_name,
                params[f"{layer_name}{proj_name}.base_layer.weight"],
                transpose=True)
        else:
            add(nntr_name,
                params[f"{layer_name}{proj_name}.weight"],
                transpose=True)

    add("embedding0:Embedding", params["model.embed_tokens.weight"])

    for i in range(n_layers):
        lp = f"model.layers.{i}."
        pfx = f"layer{i}"

        add(f"{pfx}_attention_norm:gamma",
            params[f"{lp}input_layernorm.weight"])

        # Q/K/V with bias
        for proj, nntr in [("q_proj", "wq"), ("k_proj", "wk"), ("v_proj", "wv")]:
            add_projection(f"{pfx}_{nntr}:weight", lp, f"self_attn.{proj}")
            add(f"{pfx}_{nntr}:bias", params[f"{lp}self_attn.{proj}.bias"])

        # Attention sink (mha_core "sink" — same layer name as the core)
        add(f"{pfx}_attention:sink", params[f"{lp}self_attn.sinks"])

        # Output projection
        add_projection(f"{pfx}_attention_out:weight", lp, "self_attn.o_proj")
        add(f"{pfx}_attention_out:bias",
            params[f"{lp}self_attn.o_proj.bias"])

        # Post-attention norm
        add(f"{pfx}_ffn_norm:gamma",
            params[f"{lp}post_attention_layernorm.weight"])

        # MoE: layer name is `layer{i}_ffn_down`. Router first, then per-expert.
        moe = f"{pfx}_ffn_down"
        add(f"{moe}:gate", params[f"{lp}mlp.router.weight"], transpose=True)
        add(f"{moe}:gate_bias", params[f"{lp}mlp.router.bias"])

        gate_w, gate_b, up_w, up_b = _split_gate_up(params, lp)
        down_w = params[f"{lp}mlp.experts.down_proj"]
        down_b = params[f"{lp}mlp.experts.down_proj_bias"]
        for e in range(n_experts):
            add(f"{moe}:expert_up_{e}",       up_w[e])
            add(f"{moe}:expert_up_bias_{e}",  up_b[e])
            add(f"{moe}:expert_gate_{e}",     gate_w[e])
            add(f"{moe}:expert_gate_bias_{e}", gate_b[e])
            add(f"{moe}:expert_down_{e}",     down_w[e])
            add(f"{moe}:expert_down_bias_{e}", down_b[e])

    add("output_norm:gamma", params["model.norm.weight"])
    # gpt-oss-20b is NOT tied — separate lm_head.
    add("output_of_causallm:weight", params["lm_head.weight"], transpose=True)

    return weights


# ----------------------------------------------------------------------------
# Safetensors file format writer
# ----------------------------------------------------------------------------

def save_safetensors(weights, output_path):
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


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=".",
                        help="HuggingFace model id (e.g. 'openai/gpt-oss-20b') "
                             "or a local directory containing config.json + weights.")
    parser.add_argument("--output_name", type=str,
                        default="./nntr_gpt_oss_20b.bin")
    parser.add_argument("--data_type", type=str, default="float32")
    parser.add_argument("--safetensors", action="store_true",
                        help="Also save in safetensors format alongside binary")
    args = parser.parse_args()

    data_dtype = args.data_type
    model_path = args.model_path
    output_name = args.output_name

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    # AutoModelForCausalLM handles MXFP4 dequantization of the expert
    # tensors. Loading the safetensors checkpoint directly would yield
    # raw 4-bit blocks that the converter would have to dequantize.
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=data_dtype, trust_remote_code=True)
    model.eval()

    state = model.state_dict()

    # Sanity check
    emb_row0 = state["model.embed_tokens.weight"][0, :4].tolist()
    print(f"layers={config.num_hidden_layers}, experts={config.num_local_experts}, "
          f"hidden={config.hidden_size}, vocab={config.vocab_size}")
    print(f"embed_tokens[0,:4] = {emb_row0}")

    # Always write binary.
    bin_name = output_name
    if bin_name.endswith(".safetensors"):
        bin_name = bin_name.replace(".safetensors", ".bin")
    if not bin_name.endswith(".bin"):
        bin_name += ".bin"
    with open(bin_name, "wb") as f_model:
        save_gpt_oss_for_nntrainer(state, config, data_dtype, f_model)
    print(f"Saved binary: {bin_name}")

    # Optionally also write safetensors from the SAME state_dict snapshot.
    if args.safetensors:
        st_name = bin_name.replace(".bin", ".safetensors")
        weights = collect_gpt_oss_for_nntrainer(state, config, data_dtype)
        save_safetensors(weights, st_name)
