## @file weight_converter.py
## @brief weight conversion script for multilingual TinyBERT
## @author Seunghui Lee <shsh1004.lee@samsung.com>
##
## TinyBERT (zl369/multilingual-tinyBERT-16MB) is a BertForMaskedLM model.
## Unlike the Qwen-family converters elsewhere in this directory, BERT uses:
##   - 3 input embeddings (word, position, token_type) summed together
##   - LayerNorm (gamma + beta) instead of RMSNorm
##   - Separate intermediate/output FC stacks (no SwiGLU gating)
##   - Q/K/V projections with bias
##   - Optional pooler (CLS-token projection) for sentence-level outputs
##
## The HuggingFace checkpoint is loaded directly via safetensors to avoid
## the silent key-mismatch problems we hit with SentenceTransformer for the
## kalm-embedding model.

import argparse
import os
import struct
import json
import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer


# ----------------------------------------------------------------------------
# Checkpoint loading
# ----------------------------------------------------------------------------

def load_checkpoint_state(model_path):
    """Load the model checkpoint directly from the HuggingFace snapshot.

    Returns ``(state, prefix)`` where ``state`` is a flat dictionary and
    ``prefix`` is either ``"bert."`` or ``""`` — auto-detected so the
    converter works for ``BertForMaskedLM`` checkpoints (which have a
    ``bert.`` prefix on the encoder weights) and bare ``BertModel``
    checkpoints (which do not).
    """
    cache_dir = snapshot_download(model_path)
    state = {}
    for f in sorted(os.listdir(cache_dir)):
        path = os.path.join(cache_dir, f)
        if f.endswith(".safetensors"):
            state.update(load_file(path))
        elif f.endswith(".bin"):
            state.update(torch.load(path, weights_only=True, map_location="cpu"))
    if not state:
        raise RuntimeError(f"No checkpoint files found in {cache_dir}")

    if "bert.embeddings.word_embeddings.weight" in state:
        prefix = "bert."
    elif "embeddings.word_embeddings.weight" in state:
        prefix = ""
    else:
        sample = sorted(list(state.keys()))[:10]
        raise RuntimeError(
            "Could not locate BERT embeddings in the checkpoint. "
            f"First 10 keys: {sample}")
    return state, prefix


# ----------------------------------------------------------------------------
# Binary (sequential) writer
# ----------------------------------------------------------------------------

def save_tinybert_for_nntrainer(params, n_layers, dtype, file, prefix="bert."):
    """Write weights as a sequential nntrainer binary file.

    Layer order must match the order the C++ TinyBERT graph registers
    weights so that the binary read pointer lands on the right tensor.
    """

    def save_weight(weight):
        np.array(weight, dtype=dtype).tofile(file)

    def fc_weight(name):
        # nntrainer FC stores [in, out]; PyTorch stores [out, in].
        save_weight(params[f"{name}.weight"].permute(1, 0))

    def fc_bias(name):
        save_weight(params[f"{name}.bias"])

    def layernorm(name):
        save_weight(params[f"{name}.weight"])  # gamma
        save_weight(params[f"{name}.bias"])    # beta

    # Embeddings: word + position + token_type, then LayerNorm
    save_weight(params[f"{prefix}embeddings.word_embeddings.weight"])
    save_weight(params[f"{prefix}embeddings.position_embeddings.weight"])
    save_weight(params[f"{prefix}embeddings.token_type_embeddings.weight"])
    layernorm(f"{prefix}embeddings.LayerNorm")

    for i in range(n_layers):
        p = f"{prefix}encoder.layer.{i}"
        # Self-attention Q/K/V (with bias)
        fc_weight(f"{p}.attention.self.query")
        fc_bias(f"{p}.attention.self.query")
        fc_weight(f"{p}.attention.self.key")
        fc_bias(f"{p}.attention.self.key")
        fc_weight(f"{p}.attention.self.value")
        fc_bias(f"{p}.attention.self.value")
        # Attention output projection + post-attention LayerNorm
        fc_weight(f"{p}.attention.output.dense")
        fc_bias(f"{p}.attention.output.dense")
        layernorm(f"{p}.attention.output.LayerNorm")
        # FFN: intermediate (up) → output (down) + post-FFN LayerNorm
        fc_weight(f"{p}.intermediate.dense")
        fc_bias(f"{p}.intermediate.dense")
        fc_weight(f"{p}.output.dense")
        fc_bias(f"{p}.output.dense")
        layernorm(f"{p}.output.LayerNorm")

    # Pooler (CLS-token projection). Present in BertModel checkpoints; only
    # used by sentence-embedding models. Skipped silently if absent.
    if f"{prefix}pooler.dense.weight" in params:
        fc_weight(f"{prefix}pooler.dense")
        fc_bias(f"{prefix}pooler.dense")


# ----------------------------------------------------------------------------
# Safetensors (name-keyed) collector
# ----------------------------------------------------------------------------

def collect_tinybert_for_nntrainer(params, n_layers, dtype, prefix="bert."):
    """Return ``[(nntr_name, ndarray), ...]`` for the safetensors writer.

    Naming follows the same scheme used by the Qwen converters here:
      *  embedding         → ``<name>:Embedding`` (capital E from EmbeddingLayer)
      *  LayerNorm         → ``<name>:gamma``, ``<name>:beta``
      *  FC weight / bias  → ``<name>:weight``, ``<name>:bias``
    """
    weights = []

    def add(name, tensor, transpose=False):
        # .contiguous() before .numpy() so the byte layout exactly matches what
        # np.array(t, dtype=...).tofile() writes in the binary path.
        t = tensor.permute(1, 0).contiguous() if transpose else tensor.contiguous()
        weights.append((name, t.detach().numpy().astype(dtype)))

    def add_fc(nntr_name, ckpt_name):
        add(f"{nntr_name}:weight", params[f"{ckpt_name}.weight"], transpose=True)
        add(f"{nntr_name}:bias", params[f"{ckpt_name}.bias"])

    def add_layernorm(nntr_name, ckpt_name):
        add(f"{nntr_name}:gamma", params[f"{ckpt_name}.weight"])
        add(f"{nntr_name}:beta", params[f"{ckpt_name}.bias"])

    # Embeddings
    add("embedding0:Embedding",
        params[f"{prefix}embeddings.word_embeddings.weight"])
    add("position_embedding:Embedding",
        params[f"{prefix}embeddings.position_embeddings.weight"])
    add("token_type_embedding:Embedding",
        params[f"{prefix}embeddings.token_type_embeddings.weight"])
    add_layernorm("embedding_norm", f"{prefix}embeddings.LayerNorm")

    for i in range(n_layers):
        p = f"{prefix}encoder.layer.{i}"
        pfx = f"layer{i}"
        add_fc(f"{pfx}_wq", f"{p}.attention.self.query")
        add_fc(f"{pfx}_wk", f"{p}.attention.self.key")
        add_fc(f"{pfx}_wv", f"{p}.attention.self.value")
        add_fc(f"{pfx}_attention_out", f"{p}.attention.output.dense")
        add_layernorm(f"{pfx}_attention_norm",
                      f"{p}.attention.output.LayerNorm")
        # nntrainer BERT model registers the FFN dense layers as ffn_fc1
        # (intermediate / up) and ffn_fc2 (output / down).
        add_fc(f"{pfx}_ffn_fc1", f"{p}.intermediate.dense")
        add_fc(f"{pfx}_ffn_fc2", f"{p}.output.dense")
        add_layernorm(f"{pfx}_ffn_norm", f"{p}.output.LayerNorm")

    # Pooler — sentence-embedding only; skipped silently if absent.
    if f"{prefix}pooler.dense.weight" in params:
        add_fc("pooler", f"{prefix}pooler.dense")

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
    parser.add_argument("--model_path", type=str,
                        default="zl369/multilingual-tinyBERT-16MB")
    parser.add_argument("--output_name", type=str,
                        default="./tinybert_fp32.bin")
    parser.add_argument("--data_type", type=str, default="float32")
    parser.add_argument("--safetensors", action="store_true",
                        help="Also save in safetensors format alongside binary")
    args = parser.parse_args()

    data_dtype = args.data_type
    model_path = args.model_path
    output_name = args.output_name

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    state, prefix = load_checkpoint_state(model_path)

    # Sanity check so the user can confirm we read the trained weights
    # (not a random-init fallback).
    print(f"detected key prefix = '{prefix}'")
    print("word_embeddings[0,:4] =",
          state[f"{prefix}embeddings.word_embeddings.weight"][0, :4].tolist())
    print("layer0.q.bias[:4]     =",
          state[f"{prefix}encoder.layer.0.attention.self.query.bias"][:4].tolist())
    print(f"vocab={config.vocab_size}, hidden={config.hidden_size}, "
          f"layers={config.num_hidden_layers}, heads={config.num_attention_heads}, "
          f"intermediate={config.intermediate_size}, "
          f"max_pos={config.max_position_embeddings}, "
          f"type_vocab={config.type_vocab_size}")

    # Always write binary.
    bin_name = output_name
    if bin_name.endswith(".safetensors"):
        bin_name = bin_name.replace(".safetensors", ".bin")
    if not bin_name.endswith(".bin"):
        bin_name += ".bin"
    with open(bin_name, "wb") as f_model:
        save_tinybert_for_nntrainer(
            state, config.num_hidden_layers, data_dtype, f_model, prefix=prefix)
    print(f"Saved binary: {bin_name}")

    # Optionally write safetensors from the SAME state.
    if args.safetensors:
        st_name = bin_name.replace(".bin", ".safetensors")
        weights = collect_tinybert_for_nntrainer(
            state, config.num_hidden_layers, data_dtype, prefix=prefix)
        save_safetensors(weights, st_name)
