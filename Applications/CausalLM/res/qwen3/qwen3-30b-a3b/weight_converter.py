## @file weight_converter.py
## @brief weight conversion script for qwen3_moe (Qwen3-30B-A3B)
##
## Memory-efficient streaming version. Never materialises the full
## ~60GB BF16 model in RAM — uses safetensors safe_open to fetch one
## tensor at a time, cast to FP32, write to disk, and free.
##
## Architecture notes
## ==================
## Qwen3-MoE keeps the same Qwen3 attention block as the dense variants:
##   * Q/K/V/O projections have NO bias (Qwen3 disable_bias="true")
##   * Q and K each carry their own RMSNorm (`q_norm`, `k_norm`)
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
##
## Memory strategy
## ===============
## A naive ``model.state_dict()`` load holds ~60GB BF16 in RAM, and the
## previous safetensors collector then materialised every weight again
## as a numpy array (~120GB FP32) — well over 180GB peak. This rewrite:
##   1. Opens every safetensors shard with safe_open (no data loaded).
##   2. Pre-computes per-tensor byte sizes from the on-disk shapes.
##   3. Writes the safetensors header upfront with correct offsets.
##   4. Iterates the operation list ONCE, fetching one tensor per step,
##      casting to FP32, writing to .bin (and .safetensors if requested),
##      and freeing immediately.
## Peak RAM stays around the size of the largest single expert
## (hidden * intermediate * 4B ≈ a few hundred MB).

import argparse
import json
import os
import struct

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer


# ----------------------------------------------------------------------------
# Lazy checkpoint accessor
# ----------------------------------------------------------------------------

class LazyState:
    """Lazy tensor accessor backed by safetensors safe_open handles.

    Keeps all shard files open and routes ``get_tensor(key)`` /
    ``get_shape(key)`` to the right handle. Nothing is loaded into RAM
    until ``get_tensor`` is called.
    """

    def __init__(self, model_path):
        if os.path.isdir(model_path):
            local_dir = model_path
        else:
            local_dir = snapshot_download(model_path)

        shard_paths = sorted(
            os.path.join(local_dir, f) for f in os.listdir(local_dir)
            if f.endswith(".safetensors"))
        if not shard_paths:
            raise RuntimeError(
                f"No .safetensors files found in {local_dir}. "
                "This streaming converter requires safetensors shards; "
                "convert .bin checkpoints to safetensors first.")

        self.handles = [safe_open(p, framework="pt", device="cpu")
                        for p in shard_paths]
        self.key_to_handle = {}
        for h in self.handles:
            for k in h.keys():
                self.key_to_handle[k] = h

    def __contains__(self, key):
        return key in self.key_to_handle

    def get_tensor(self, key):
        h = self.key_to_handle.get(key)
        if h is None:
            raise KeyError(key)
        return h.get_tensor(key)

    def get_shape(self, key):
        h = self.key_to_handle.get(key)
        if h is None:
            raise KeyError(key)
        # get_slice returns a slice object whose .get_shape() reports
        # the on-disk shape without copying data.
        return list(h.get_slice(key).get_shape())


# ----------------------------------------------------------------------------
# Operation list — describes each output tensor symbolically
# ----------------------------------------------------------------------------
# Each op: (name_nntr, source_kind, *source_args).
# Currently only "key" kind is needed (no packed gate/up like gpt-oss).
#   "key" args: (source_key, transpose)
# ----------------------------------------------------------------------------

def make_ops(config, state):
    """Return the ordered list of operations for Qwen3-MoE.

    The order matches the C++ Qwen3MoE graph traversal so the binary
    file can be read positionally; safetensors uses the names directly.
    """
    n_layers = config.num_hidden_layers
    n_experts = getattr(config, "num_experts",
                        getattr(config, "num_local_experts", 0))
    if n_experts == 0:
        raise RuntimeError("Could not find num_experts in config")

    ops = []
    ops.append(("embedding0:Embedding", "key", "model.embed_tokens.weight", False))

    for i in range(n_layers):
        lp = f"model.layers.{i}."
        pfx = f"layer{i}"

        ops.append((f"{pfx}_attention_norm:gamma", "key",
                    f"{lp}input_layernorm.weight", False))

        # Q/K/V/O (no bias). q_norm/k_norm appear right after Q and K respectively.
        ops.append((f"{pfx}_wq:weight", "key",
                    f"{lp}self_attn.q_proj.weight", True))
        if f"{lp}self_attn.q_norm.weight" in state:
            ops.append((f"{pfx}_q_norm:gamma", "key",
                        f"{lp}self_attn.q_norm.weight", False))
        ops.append((f"{pfx}_wk:weight", "key",
                    f"{lp}self_attn.k_proj.weight", True))
        if f"{lp}self_attn.k_norm.weight" in state:
            ops.append((f"{pfx}_k_norm:gamma", "key",
                        f"{lp}self_attn.k_norm.weight", False))
        ops.append((f"{pfx}_wv:weight", "key",
                    f"{lp}self_attn.v_proj.weight", True))
        ops.append((f"{pfx}_attention_out:weight", "key",
                    f"{lp}self_attn.o_proj.weight", True))

        ops.append((f"{pfx}_ffn_norm:gamma", "key",
                    f"{lp}post_attention_layernorm.weight", False))

        # MoE: layer named `layer{i}_ffn_down`. Router first, then per-expert.
        moe = f"{pfx}_ffn_down"
        ops.append((f"{moe}:gate", "key", f"{lp}mlp.gate.weight", True))
        for e in range(n_experts):
            ops.append((f"{moe}:expert_up_{e}", "key",
                        f"{lp}mlp.experts.{e}.up_proj.weight", True))
            ops.append((f"{moe}:expert_gate_{e}", "key",
                        f"{lp}mlp.experts.{e}.gate_proj.weight", True))
            ops.append((f"{moe}:expert_down_{e}", "key",
                        f"{lp}mlp.experts.{e}.down_proj.weight", True))

    ops.append(("output_norm:gamma", "key", "model.norm.weight", False))
    # Qwen3-30B-A3B is NOT tied — separate lm_head.
    ops.append(("output_of_causallm:weight", "key", "lm_head.weight", True))

    return ops


def _compute_shape(state, op):
    """Return the FP32 output shape of an op without loading the tensor."""
    kind = op[1]
    if kind == "key":
        key, transpose = op[2], op[3]
        shape = state.get_shape(key)
        if transpose:
            shape = shape[::-1]
        return shape
    raise ValueError(f"Unknown op kind: {kind}")


def _materialise(state, op):
    """Fetch a tensor for an op, cast to FP32, return a contiguous numpy."""
    kind = op[1]
    if kind == "key":
        key, transpose = op[2], op[3]
        t = state.get_tensor(key)
        if transpose:
            t = t.permute(1, 0)
        return t.to(torch.float32).contiguous().detach().numpy()
    raise ValueError(f"Unknown op kind: {kind}")


# ----------------------------------------------------------------------------
# Streaming writer — one materialisation per tensor, no list of arrays
# ----------------------------------------------------------------------------

def write_streaming(state, config, dtype, bin_path, st_path=None):
    """Write the binary file and (optionally) the safetensors file in one
    streaming pass over the operation list. Each tensor is held in
    memory only as long as it takes to write its bytes to disk.
    """
    ops = make_ops(config, state)

    # Per-tensor sizes/shapes — cheap since get_shape() doesn't copy data.
    sizes = []
    shapes = []
    for op in ops:
        shape = _compute_shape(state, op)
        shapes.append(shape)
        nbytes = 4  # FP32
        for s in shape:
            nbytes *= s
        sizes.append(nbytes)

    # Pre-build the safetensors header (offsets known once sizes are).
    st_file = None
    if st_path is not None:
        metadata = {"format": "pt"}
        offset = 0
        tensor_meta = {}
        for op, nbytes, shape in zip(ops, sizes, shapes):
            name = op[0]
            tensor_meta[name] = {
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [offset, offset + nbytes],
            }
            offset += nbytes
        header = {"__metadata__": metadata}
        header.update(tensor_meta)
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        pad = (8 - len(header_bytes) % 8) % 8
        header_bytes += b" " * pad

        st_file = open(st_path, "wb")
        st_file.write(struct.pack("<Q", len(header_bytes)))
        st_file.write(header_bytes)
        total_data_gb = offset / 1e9
    else:
        total_data_gb = sum(sizes) / 1e9

    print(f"Total tensors: {len(ops)}, data section: {total_data_gb:.2f} GB")

    bin_file = open(bin_path, "wb")
    try:
        for idx, op in enumerate(ops):
            arr = _materialise(state, op)
            if arr.dtype != np.dtype(dtype):
                arr = arr.astype(dtype, copy=False)
            buf = arr.tobytes()
            if buf and len(buf) != sizes[idx]:
                raise RuntimeError(
                    f"Size mismatch for {op[0]}: expected {sizes[idx]}, "
                    f"got {len(buf)}")
            bin_file.write(buf)
            if st_file is not None:
                st_file.write(buf)
            del arr
            del buf
            if (idx + 1) % 100 == 0 or idx == len(ops) - 1:
                print(f"  {idx + 1}/{len(ops)}: {op[0]}")
    finally:
        bin_file.close()
        if st_file is not None:
            st_file.close()

    print(f"Saved binary: {bin_path}")
    if st_path is not None:
        print(f"Saved safetensors: {st_path}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./Qwen3-30B-A3B",
                        help="HuggingFace model id (e.g. 'Qwen/Qwen3-30B-A3B') "
                             "or a local directory containing the safetensors "
                             "shards and config.json.")
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

    # Open all shards lazily — keys are indexed but tensors are not loaded.
    state = LazyState(model_path)

    n_experts = getattr(config, "num_experts",
                        getattr(config, "num_local_experts", 0))
    print(f"hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"heads={config.num_attention_heads}, vocab={config.vocab_size}, "
          f"experts={n_experts}")
    emb_row0 = state.get_tensor("model.embed_tokens.weight")[0, :4]
    print("embed_tokens[0,:4] =", emb_row0.to(torch.float32).tolist())
    del emb_row0

    bin_name = output_name
    if bin_name.endswith(".safetensors"):
        bin_name = bin_name.replace(".safetensors", ".bin")
    if not bin_name.endswith(".bin"):
        bin_name += ".bin"
    st_name = bin_name.replace(".bin", ".safetensors") if args.safetensors else None

    write_streaming(state, config, data_dtype, bin_name, st_name)
