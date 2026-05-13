## @file weight_converter.py
## @brief weight conversion script for qwen3_moe (Qwen3-30B-A3B)
##
## Generates both nntrainer binary and safetensors files for the dense
## Qwen3 MoE architecture (Qwen3-30B-A3B). The smaller Qwen3 dense models
## (qwen3-0.6b, qwen3-4b) live one level up and use their own converters.
##
## Architecture notes
## ==================
## Qwen3-MoE keeps the same Qwen3 attention block as the dense variants:
##   * Q/K/V/O projections have NO bias (Qwen3 disable_bias="true")
##   * Q/K each carry their own RMSNorm (`q_norm`, `k_norm`)
##   * No attention sink
## The MLP is replaced by a Mixture-of-Experts layer named in the C++
## graph as ``layer{i}_ffn_down``. That single layer registers:
##   layer{i}_ffn_down:gate                  (router weight, transposed)
##   layer{i}_ffn_down:expert_up_{e}         (per-expert up)
##   layer{i}_ffn_down:expert_gate_{e}       (per-expert gate)
##   layer{i}_ffn_down:expert_down_{e}       (per-expert down)
## No biases on either the router or the experts.
##
## The PyTorch checkpoint stores per-expert weights as SEPARATE keys
## (``mlp.experts.{e}.up_proj.weight`` etc.), unlike gpt-oss which packs
## gate/up into one tensor. So no split step is needed here.

import argparse
import json
import os
import struct

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer


# ----------------------------------------------------------------------------
# Checkpoint loading
# ----------------------------------------------------------------------------

def load_checkpoint_state(model_path):
    """Load all checkpoint shards directly via safetensors.

    Returns a flat ``{key: torch.Tensor}`` dict. ``model_path`` may be a
    HuggingFace model id or a local directory.
    """
    if os.path.isdir(model_path):
        local_dir = model_path
    else:
        local_dir = snapshot_download(model_path)

    state = {}
    for f in sorted(os.listdir(local_dir)):
        full = os.path.join(local_dir, f)
        if f.endswith(".safetensors"):
            state.update(load_file(full))
        elif f.endswith(".bin"):
            state.update(torch.load(full, weights_only=True, map_location="cpu"))
    if not state:
        raise RuntimeError(f"No checkpoint files found in {local_dir}")
    return state


# ----------------------------------------------------------------------------
# Binary (sequential) writer
# ----------------------------------------------------------------------------

def save_qwen3_moe_for_nntrainer(params, config, dtype, file):
    n_layers = config.num_hidden_layers
    # transformers exposes both `num_experts` (older) and
    # `num_local_experts` (newer). Pick whichever is present.
    n_experts = getattr(config, "num_experts",
                        getattr(config, "num_local_experts", 0))
    if n_experts == 0:
        raise RuntimeError("Could not find num_experts in config")

    def save_weight(weight_name, is_transpose=False):
        w = params[weight_name].to(torch.float32)
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
        # Q/K/V/O: weight only (Qwen3 disable_bias=true)
        # Q and K have an extra RMSNorm registered right after the FC.
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            save_projection(layer_name, f"self_attn.{proj}")
            proj_norm_name = f"{layer_name}self_attn.{proj[0]}_norm.weight"
            if proj_norm_name in params:
                save_weight(proj_norm_name)

    def save_feed_forward(layer_name):
        # Router weight (transposed from PyTorch's [n_experts, hidden]
        # to nntrainer's [hidden, n_experts]). No router bias.
        save_weight(f"{layer_name}mlp.gate.weight", True)

        # Per-expert: up -> gate -> down (matches C++ requestWeight order).
        for e in range(n_experts):
            for proj in ["up_proj", "gate_proj", "down_proj"]:
                save_projection(layer_name, f"mlp.experts.{e}.{proj}")

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

def collect_qwen3_moe_for_nntrainer(params, config, dtype):
    n_layers = config.num_hidden_layers
    n_experts = getattr(config, "num_experts",
                        getattr(config, "num_local_experts", 0))
    weights = []

    def add(name, tensor, transpose=False):
        t = tensor.to(torch.float32)
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

        # Q/K/V/O (no bias). Q and K have their own norm right after.
        add_projection(f"{pfx}_wq:weight", lp, "self_attn.q_proj")
        if f"{lp}self_attn.q_norm.weight" in params:
            add(f"{pfx}_q_norm:gamma",
                params[f"{lp}self_attn.q_norm.weight"])
        add_projection(f"{pfx}_wk:weight", lp, "self_attn.k_proj")
        if f"{lp}self_attn.k_norm.weight" in params:
            add(f"{pfx}_k_norm:gamma",
                params[f"{lp}self_attn.k_norm.weight"])
        add_projection(f"{pfx}_wv:weight", lp, "self_attn.v_proj")
        add_projection(f"{pfx}_attention_out:weight", lp, "self_attn.o_proj")

        add(f"{pfx}_ffn_norm:gamma",
            params[f"{lp}post_attention_layernorm.weight"])

        # MoE layer name in C++ is `layer{i}_ffn_down`. Router + per-expert.
        moe = f"{pfx}_ffn_down"
        add(f"{moe}:gate", params[f"{lp}mlp.gate.weight"], transpose=True)
        for e in range(n_experts):
            add_projection(f"{moe}:expert_up_{e}",   lp, f"mlp.experts.{e}.up_proj")
            add_projection(f"{moe}:expert_gate_{e}", lp, f"mlp.experts.{e}.gate_proj")
            add_projection(f"{moe}:expert_down_{e}", lp, f"mlp.experts.{e}.down_proj")

    add("output_norm:gamma", params["model.norm.weight"])
    # Qwen3-30B-A3B is NOT tied — separate lm_head.
    add("output_of_causallm:weight",
        params["lm_head.weight"], transpose=True)

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
    parser.add_argument("--model_path", type=str, default="./Qwen3-30B-A3B",
                        help="HuggingFace model id (e.g. 'Qwen/Qwen3-30B-A3B') "
                             "or a local directory.")
    parser.add_argument("--output_name", type=str,
                        default="./nntr_qwen3_30b_a3b_fp32.bin")
    parser.add_argument("--data_type", type=str, default="float32")
    parser.add_argument("--safetensors", action="store_true",
                        help="Also save in safetensors format alongside binary")
    args = parser.parse_args()

    data_dtype = args.data_type
    model_path = args.model_path
    output_name = args.output_name

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    state = load_checkpoint_state(model_path)

    n_experts = getattr(config, "num_experts",
                        getattr(config, "num_local_experts", 0))
    print(f"hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"heads={config.num_attention_heads}, vocab={config.vocab_size}, "
          f"experts={n_experts}")
    print("embed_tokens[0,:4] =",
          state["model.embed_tokens.weight"][0, :4].tolist())

    # Always write binary.
    bin_name = output_name
    if bin_name.endswith(".safetensors"):
        bin_name = bin_name.replace(".safetensors", ".bin")
    if not bin_name.endswith(".bin"):
        bin_name += ".bin"
    with open(bin_name, "wb") as f_model:
        save_qwen3_moe_for_nntrainer(state, config, data_dtype, f_model)
    print(f"Saved binary: {bin_name}")

    if args.safetensors:
        st_name = bin_name.replace(".bin", ".safetensors")
        weights = collect_qwen3_moe_for_nntrainer(state, config, data_dtype)
        save_safetensors(weights, st_name)
