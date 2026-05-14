"""
SPDX-License-Identifier: Apache-2.0
Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>

@file weight_converter.py
@brief gpt-oss-20b weight converter — memory-efficient streaming version.
@author Eunju Yang <ej.yang@samsung.com>

Tested with transformers 4.53.2.

Memory strategy
===============
The previous version held the entire FP32 model in RAM (~80GB for 20B
params) and then materialised every weight again as a numpy array
inside ``collect_*_for_nntrainer`` (another ~80GB) before writing the
safetensors file. That peaked around 160GB and OOM-killed the converter.

This version:
  1. Loads the model in BF16 (~40GB) with ``low_cpu_mem_usage=True``.
     BF16 -> FP32 is a lossless zero-extension on the mantissa, so the
     final FP32 bytes on disk are identical to loading in FP32.
  2. Builds a single OPERATION LIST describing each output tensor as a
     symbolic recipe (key + optional transpose / expert-index / packed
     gate-up split). No tensors are materialised at this stage.
  3. Pre-computes per-tensor byte sizes from the state_dict's tensor
     shapes alone (no data copy) so the safetensors header can be
     written upfront with correct offsets.
  4. Iterates the operation list ONCE, materialising one FP32 tensor
     at a time, streaming the bytes to both the .bin file and the
     .safetensors data section, then immediately freeing the temporary.

Peak extra memory per step ~ 2 * sizeof(largest_tensor) ~ few hundred
MB. The model itself in BF16 stays at ~40GB.

gpt-oss specifics
=================
  - MoE FFN: the C++ `gpt_oss_moe` layer is named `layer{i}_ffn_down`
    and registers (per expert e, all with bias):
        layer{i}_ffn_down:gate / gate_bias
        layer{i}_ffn_down:expert_up_{e},   :expert_up_bias_{e}
        layer{i}_ffn_down:expert_gate_{e}, :expert_gate_bias_{e}
        layer{i}_ffn_down:expert_down_{e}, :expert_down_bias_{e}
  - Attention sink: mha_core registers a per-head `sink` weight under
    `layer{i}_attention:sink` (`use_sink="true"` in createAttention).
  - Q/K/V/O all carry bias (`disable_bias="false"`).
  - Experts are stored in MXFP4. AutoModelForCausalLM dequantises them
    on load so the converter sees regular float tensors.
  - gate_up_proj is packed: indices ::2 are gate columns, 1::2 are up
    columns. We split per-expert at materialise-time.
"""

import argparse
import json
import struct

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


# ----------------------------------------------------------------------------
# Operation list — describes each output tensor symbolically
# ----------------------------------------------------------------------------
#
# Each op is a tuple ``(name_nntr, source_kind, *args)`` where source_kind is:
#   "key"    -> state_dict[args[0]], optionally transposed (args[1])
#   "expert" -> state_dict[args[0]][args[1]]   (drops the expert dim)
#   "split"  -> state_dict[args[0]][..., ::2 or 1::2][args[1]]
#               args[2] = parity (0=gate, 1=up)
# ----------------------------------------------------------------------------

def make_ops(config):
    """Return the ordered list of operations for gpt-oss-20b.

    The order matches the C++ graph traversal so the binary file can be
    read positionally; safetensors uses the names from each op.
    """
    n_layers = config.num_hidden_layers
    n_experts = config.num_local_experts
    ops = []

    ops.append(("embedding0:Embedding", "key", "model.embed_tokens.weight", False))

    for i in range(n_layers):
        lp = f"model.layers.{i}."
        pfx = f"layer{i}"

        ops.append((f"{pfx}_attention_norm:gamma",
                    "key", f"{lp}input_layernorm.weight", False))

        # Q / K / V (each: weight transposed, then bias)
        for proj, nntr in [("q_proj", "wq"), ("k_proj", "wk"), ("v_proj", "wv")]:
            ops.append((f"{pfx}_{nntr}:weight",
                        "key", f"{lp}self_attn.{proj}.weight", True))
            ops.append((f"{pfx}_{nntr}:bias",
                        "key", f"{lp}self_attn.{proj}.bias", False))

        # Attention sink (mha_core `sink`, registered under the core layer name)
        ops.append((f"{pfx}_attention:sink",
                    "key", f"{lp}self_attn.sinks", False))

        # Output projection
        ops.append((f"{pfx}_attention_out:weight",
                    "key", f"{lp}self_attn.o_proj.weight", True))
        ops.append((f"{pfx}_attention_out:bias",
                    "key", f"{lp}self_attn.o_proj.bias", False))

        # Post-attention norm
        ops.append((f"{pfx}_ffn_norm:gamma",
                    "key", f"{lp}post_attention_layernorm.weight", False))

        # MoE: layer named `layer{i}_ffn_down`. Router first, then per-expert.
        moe = f"{pfx}_ffn_down"
        ops.append((f"{moe}:gate", "key", f"{lp}mlp.router.weight", True))
        ops.append((f"{moe}:gate_bias", "key", f"{lp}mlp.router.bias", False))

        for e in range(n_experts):
            # Per-expert order matches C++: up, up_bias, gate, gate_bias, down, down_bias
            ops.append((f"{moe}:expert_up_{e}",
                        "split", f"{lp}mlp.experts.gate_up_proj", e, 1))
            ops.append((f"{moe}:expert_up_bias_{e}",
                        "split", f"{lp}mlp.experts.gate_up_proj_bias", e, 1))
            ops.append((f"{moe}:expert_gate_{e}",
                        "split", f"{lp}mlp.experts.gate_up_proj", e, 0))
            ops.append((f"{moe}:expert_gate_bias_{e}",
                        "split", f"{lp}mlp.experts.gate_up_proj_bias", e, 0))
            ops.append((f"{moe}:expert_down_{e}",
                        "expert", f"{lp}mlp.experts.down_proj", e))
            ops.append((f"{moe}:expert_down_bias_{e}",
                        "expert", f"{lp}mlp.experts.down_proj_bias", e))

    ops.append(("output_norm:gamma", "key", "model.norm.weight", False))
    # gpt-oss-20b is NOT tied — separate lm_head.
    ops.append(("output_of_causallm:weight", "key", "lm_head.weight", True))

    return ops


def _compute_shape(params, op):
    """Compute the output FP32 shape of an op without materialising data."""
    kind = op[1]
    if kind == "key":
        key, transpose = op[2], op[3]
        shape = list(params[key].shape)
        if transpose:
            shape = shape[::-1]
        return shape
    if kind == "expert":
        key, _e = op[2], op[3]
        # Drop the leading expert dim
        return list(params[key].shape[1:])
    if kind == "split":
        key, _e, _parity = op[2], op[3], op[4]
        # Drop expert dim, halve the last dim (::2 selects every other column)
        shape = list(params[key].shape[1:])
        shape[-1] = shape[-1] // 2
        return shape
    raise ValueError(f"Unknown op kind: {kind}")


def _materialise(params, op):
    """Produce a single FP32 contiguous numpy array for an op."""
    kind = op[1]
    if kind == "key":
        key, transpose = op[2], op[3]
        t = params[key]
        if transpose:
            t = t.permute(1, 0)
    elif kind == "expert":
        key, e = op[2], op[3]
        t = params[key][e]
    elif kind == "split":
        key, e, parity = op[2], op[3], op[4]
        full = params[key]
        # parity 0 -> gate (::2), 1 -> up (1::2)
        t = full[..., ::2][e] if parity == 0 else full[..., 1::2][e]
    else:
        raise ValueError(f"Unknown op kind: {kind}")
    return t.to(torch.float32).contiguous().detach().numpy()


# ----------------------------------------------------------------------------
# Streaming writer — one materialisation per tensor, no list of arrays
# ----------------------------------------------------------------------------

def write_streaming(params, config, dtype, bin_path, st_path=None):
    """Write the binary file and (optionally) the safetensors file in one
    streaming pass over the operation list. Each tensor is held in
    memory for only as long as it takes to write its bytes to disk.
    """
    ops = make_ops(config)

    # Per-tensor byte sizes — cheap since .shape doesn't copy data.
    sizes = []
    shapes = []
    for op in ops:
        shape = _compute_shape(params, op)
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
        for (name_op, *_), nbytes, shape in zip(ops, sizes, shapes):
            name = name_op  # first element is the nntrainer name
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
            arr = _materialise(params, op)
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
            # Help the GC reclaim immediately.
            del arr
            del buf
            if (idx + 1) % 50 == 0 or idx == len(ops) - 1:
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

    # Memory-efficient load:
    #   - torch_dtype=torch.bfloat16: keep the model at BF16 in RAM (~40GB
    #     for gpt-oss-20b instead of ~80GB at FP32). BF16 -> FP32 is a
    #     lossless zero-extension on the mantissa, so the FP32 bytes we
    #     write to disk are bit-identical to loading the model in FP32
    #     directly.
    #   - low_cpu_mem_usage=True: stream-loads the checkpoint shard by
    #     shard instead of buffering it all before allocating, which
    #     keeps the peak load-time RSS close to the model size.
    # AutoModelForCausalLM is still required (not raw safetensors) so
    # the MXFP4 expert weights get dequantised at load time.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    state = model.state_dict()

    emb_row0 = state["model.embed_tokens.weight"][0, :4].to(torch.float32).tolist()
    print(f"layers={config.num_hidden_layers}, experts={config.num_local_experts}, "
          f"hidden={config.hidden_size}, vocab={config.vocab_size}")
    print(f"embed_tokens[0,:4] = {emb_row0}")

    bin_name = output_name
    if bin_name.endswith(".safetensors"):
        bin_name = bin_name.replace(".safetensors", ".bin")
    if not bin_name.endswith(".bin"):
        bin_name += ".bin"
    st_name = bin_name.replace(".bin", ".safetensors") if args.safetensors else None

    write_streaming(state, config, data_dtype, bin_name, st_name)
