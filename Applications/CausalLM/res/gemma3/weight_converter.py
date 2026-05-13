## SPDX-License-Identifier: Apache-2.0
## Copyright (C) 2025 Seungbaek Hong <sb92.hong@samsung.com>
##
## @file weight_converter.py
## @brief weight conversion script for Gemma3 / FunctionGemma models
## @author SeungBaek Hong <sb92.hong@samsung.com>
##
## Generates both nntrainer binary and safetensors files for any model
## whose architecture matches the C++ Gemma3Transformer / Gemma3CausalLM
## graph (gemma-3-270m, FunctionGemma, and other Gemma3 variants).
##
## Gemma3-specific quirks the converter has to honor:
##   - All RMS-norm weights are stored on disk as ``gamma - 1``. The model
##     reads them and adds 1 at run time, so the converter has to do the
##     same (+1.0) when writing.
##   - The C++ graph names the pre-/post-FFN RMS norms WITHOUT an
##     underscore between layer index and the rest of the name
##     (``layer0pre_ffn_norm`` / ``layer0post_ffn_norm``). The safetensors
##     keys have to match exactly.
##   - ``tie_word_embeddings`` defaults to true (set by C++
##     Gemma3Transformer::sanitizeConfig). When true the lm_head shares
##     storage with embedding0 and we must NOT write a separate
##     ``output_of_causallm:weight`` entry.

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
    """Return ``state`` and ``tie_word_embeddings`` from the checkpoint.

    ``model_path`` may be either a HuggingFace model id or a local
    directory that already contains the safetensors / pytorch_model.bin
    plus a config.json. We bypass AutoModel.from_pretrained and read the
    checkpoint files directly so we get the trained tensors with no risk
    of silent random-init fallbacks (the failure mode we hit on KaLM
    Embedding).
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

    # Gemma3 ships the encoder weights under the "model." prefix. Provide
    # them under that exact name so the converter functions below can use
    # the standard keys.
    cfg_path = os.path.join(local_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    # sanitizeConfig in C++ defaults this to True.
    tie = cfg.get("tie_word_embeddings", True)
    return state, tie


# ----------------------------------------------------------------------------
# Binary (sequential) writer — order must match the C++ graph traversal
# ----------------------------------------------------------------------------

def save_gemma3_for_nntrainer(params, config, dtype, file, tie_word_embeddings):
    n_layers = config.num_hidden_layers

    def save_weight(weight, is_rms=False):
        if is_rms:
            weight = weight + 1.0
        np.array(weight, dtype=dtype).tofile(file)

    def save_projection(layer_name, proj_name):
        lora_key = f"{layer_name}{proj_name}.lora_A.default.weight"
        if lora_key in params:
            save_weight(params[f"{layer_name}{proj_name}.base_layer.weight"].permute(1, 0))
            save_weight(params[f"{layer_name}{proj_name}.lora_A.default.weight"].permute(1, 0))
            save_weight(params[f"{layer_name}{proj_name}.lora_B.default.weight"].permute(1, 0))
        else:
            save_weight(params[f"{layer_name}{proj_name}.weight"].permute(1, 0))

    def save_attention(layer_name):
        # attention_norm -> Q -> K -> V -> q_norm -> k_norm -> O
        save_weight(params[f"{layer_name}input_layernorm.weight"], is_rms=True)
        save_projection(layer_name, "self_attn.q_proj")
        save_projection(layer_name, "self_attn.k_proj")
        save_projection(layer_name, "self_attn.v_proj")
        if f"{layer_name}self_attn.q_norm.weight" in params:
            save_weight(params[f"{layer_name}self_attn.q_norm.weight"], is_rms=True)
        if f"{layer_name}self_attn.k_norm.weight" in params:
            save_weight(params[f"{layer_name}self_attn.k_norm.weight"], is_rms=True)
        save_projection(layer_name, "self_attn.o_proj")

    def save_feed_forward(layer_name):
        # post_attention_norm -> pre_ffn_norm -> gate -> up -> down -> post_ffn_norm
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
    # lm_head is only written for untied models. For tied models the
    # C++ TieWordEmbedding lm_head read() is a no-op (the tensor is
    # shared with embedding0).
    if not tie_word_embeddings:
        save_weight(params["lm_head.weight"].permute(1, 0))


# ----------------------------------------------------------------------------
# Safetensors (name-keyed) collector — names must match C++ layer names
# ----------------------------------------------------------------------------

def collect_gemma3_for_nntrainer(params, config, dtype, tie_word_embeddings):
    n_layers = config.num_hidden_layers
    weights = []

    def add(name, tensor, transpose=False, is_rms=False):
        if is_rms:
            tensor = tensor + 1.0
        # .contiguous() before .numpy() so the safetensors byte layout
        # matches what np.array(t, dtype=...).tofile() writes.
        t = tensor.permute(1, 0).contiguous() if transpose else tensor.contiguous()
        weights.append((name, t.detach().numpy().astype(dtype)))

    def add_projection(nntr_name, layer_name, proj_name):
        lora_key = f"{layer_name}{proj_name}.lora_A.default.weight"
        if lora_key in params:
            # Match the binary file's base-only layout (LoRA weights are
            # written contiguously after the base in the binary path; for
            # safetensors we keep only the base, callers can merge LoRA
            # via add_projection extension if needed later).
            add(nntr_name, params[f"{layer_name}{proj_name}.base_layer.weight"], transpose=True)
        else:
            add(nntr_name, params[f"{layer_name}{proj_name}.weight"], transpose=True)

    # Embedding
    add("embedding0:Embedding", params["model.embed_tokens.weight"])

    for i in range(n_layers):
        lp = f"model.layers.{i}."
        pfx = f"layer{i}"

        add(f"{pfx}_attention_norm:gamma",
            params[f"{lp}input_layernorm.weight"], is_rms=True)
        add_projection(f"{pfx}_wq:weight", lp, "self_attn.q_proj")
        add_projection(f"{pfx}_wk:weight", lp, "self_attn.k_proj")
        add_projection(f"{pfx}_wv:weight", lp, "self_attn.v_proj")
        if f"{lp}self_attn.q_norm.weight" in params:
            add(f"{pfx}_q_norm:gamma",
                params[f"{lp}self_attn.q_norm.weight"], is_rms=True)
        if f"{lp}self_attn.k_norm.weight" in params:
            add(f"{pfx}_k_norm:gamma",
                params[f"{lp}self_attn.k_norm.weight"], is_rms=True)
        add_projection(f"{pfx}_attention_out:weight", lp, "self_attn.o_proj")

        add(f"{pfx}_post_attention_norm:gamma",
            params[f"{lp}post_attention_layernorm.weight"], is_rms=True)
        # Note: the C++ graph names the pre-/post-FFN norms WITHOUT an
        # underscore between layer index and the rest. Keep this exact.
        add(f"{pfx}pre_ffn_norm:gamma",
            params[f"{lp}pre_feedforward_layernorm.weight"], is_rms=True)
        add_projection(f"{pfx}_ffn_gate:weight", lp, "mlp.gate_proj")
        add_projection(f"{pfx}_ffn_up:weight",   lp, "mlp.up_proj")
        add_projection(f"{pfx}_ffn_down:weight", lp, "mlp.down_proj")
        add(f"{pfx}post_ffn_norm:gamma",
            params[f"{lp}post_feedforward_layernorm.weight"], is_rms=True)

    add("output_norm:gamma", params["model.norm.weight"], is_rms=True)

    if not tie_word_embeddings:
        add("output_of_causallm:weight", params["lm_head.weight"], transpose=True)

    return weights


# ----------------------------------------------------------------------------
# Safetensors file format writer
# ----------------------------------------------------------------------------

def save_safetensors(weights, output_path):
    """Write weights to a .safetensors file (F32, name-keyed)."""
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

    print(f"Saved safetensors: {output_path} ({offset / 1e6:.2f} MB tensor data)")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./270m",
                        help="HuggingFace model id (e.g. 'google/gemma-3-270m') "
                             "or a local directory containing config.json + weights.")
    parser.add_argument("--output_name", type=str,
                        default="./nntr_gemma3_270m_fp32.bin")
    parser.add_argument("--data_type", type=str, default="float32")
    parser.add_argument("--safetensors", action="store_true",
                        help="Also save in safetensors format alongside binary")
    args = parser.parse_args()

    data_dtype = args.data_type
    model_path = args.model_path
    output_name = args.output_name

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    state, tie = load_checkpoint_state(model_path)

    # tie_word_embeddings reflects the value in config.json after
    # sanitizeConfig (default true if absent).
    print(f"tie_word_embeddings = {tie}")
    print(f"hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"heads={config.num_attention_heads}, vocab={config.vocab_size}")
    print("embed_tokens[0,:4] =",
          state["model.embed_tokens.weight"][0, :4].tolist())
    print("model.norm.weight[:4] (raw) =",
          state["model.norm.weight"][:4].tolist())

    # Always write binary.
    bin_name = output_name
    if bin_name.endswith(".safetensors"):
        bin_name = bin_name.replace(".safetensors", ".bin")
    if not bin_name.endswith(".bin"):
        bin_name += ".bin"
    with open(bin_name, "wb") as f_model:
        save_gemma3_for_nntrainer(state, config, data_dtype, f_model, tie)
    print(f"Saved binary: {bin_name}")

    # Optionally also write safetensors from the SAME checkpoint state.
    if args.safetensors:
        st_name = bin_name.replace(".bin", ".safetensors")
        weights = collect_gemma3_for_nntrainer(state, config, data_dtype, tie)
        save_safetensors(weights, st_name)
