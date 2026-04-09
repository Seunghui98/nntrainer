## @file weight_converter.py
## @brief weight conversion script for Qwen3.5 hybrid model (Gated DeltaNet + Transformer)
## @author Seunghui Lee <shsh1004.lee@samsung.com>

import argparse
import os
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def get_text_config(config):
    """Extract text model config from potentially nested VLM config"""
    if hasattr(config, 'text_config'):
        return config.text_config
    return config


def save_qwen3_5_for_nntrainer(params, config, dtype, file):
    """Convert and save Qwen3.5 hybrid model weights for NNTrainer"""

    text_cfg = get_text_config(config)
    n_layers = text_cfg.num_hidden_layers
    tie_word_embeddings = getattr(config, 'tie_word_embeddings', False)

    # Determine layer types
    layer_types = []
    if hasattr(text_cfg, 'layer_types') and text_cfg.layer_types:
        layer_types = text_cfg.layer_types
    else:
        # Infer from state dict
        for i in range(n_layers):
            has_self_attn = f'model.layers.{i}.self_attn.q_proj.weight' in params
            layer_types.append('full_attention' if has_self_attn else 'linear_attention')

    # Self-attention head_dim (for de-interleaving q_proj query/gate)
    head_dim_sa = getattr(text_cfg, 'head_dim', 256)

    print(f"Layer types: {layer_types}")
    print(f"Self-attention head_dim: {head_dim_sa}")

    total_floats = [0]  # mutable counter

    def save_weight(weight, is_rms=False):
        """Save weight tensor. For RMS norm weights, add 1.0 (offset format)."""
        if is_rms:
            weight = weight + 1.0
        arr = weight.detach().float().cpu().numpy().astype(dtype)
        arr.tofile(file)
        total_floats[0] += arr.size

    def save_projection(key, transpose=True):
        """Save a weight tensor, optionally transposed"""
        w = params[key]
        if transpose:
            w = w.permute(1, 0)
        save_weight(w)
        print(f"  {key}: {params[key].shape}")

    def save_linear_attn(layer_prefix):
        """Save Gated DeltaNet (linear attention) layer weights.

        Weight order must match GatedDeltaNetLayer::finalize() weight request order:
          0: in_proj_qkv  (conv_dim, hidden_size) -> transposed to (hidden_size, conv_dim)
          1: conv1d        (conv_dim, 1, conv_kernel) -> reshaped to (conv_dim, conv_kernel)
          2: A_log          (num_v_heads)
          3: dt_bias        (num_v_heads)
          4: in_proj_a     (num_v_heads, hidden_size) -> transposed
          5: in_proj_b     (num_v_heads, hidden_size) -> transposed
          6: in_proj_z     (value_dim, hidden_size) -> transposed
          7: norm           (head_v_dim)
          8: out_proj      (hidden_size, value_dim) -> transposed
        """
        prefix = f"{layer_prefix}linear_attn."

        # in_proj_qkv: (conv_dim, hidden_size) -> (hidden_size, conv_dim)
        save_projection(f"{prefix}in_proj_qkv.weight", transpose=True)

        # conv1d: (conv_dim, 1, conv_kernel) -> flatten to (conv_dim, conv_kernel)
        conv_w = params[f"{prefix}conv1d.weight"]
        conv_w = conv_w.squeeze(1)  # (conv_dim, conv_kernel)
        save_weight(conv_w)
        print(f"  {prefix}conv1d.weight: {params[f'{prefix}conv1d.weight'].shape} -> {conv_w.shape}")

        # A_log: (num_v_heads)
        save_weight(params[f"{prefix}A_log"])
        print(f"  {prefix}A_log: {params[f'{prefix}A_log'].shape}")

        # dt_bias: (num_v_heads)
        save_weight(params[f"{prefix}dt_bias"])
        print(f"  {prefix}dt_bias: {params[f'{prefix}dt_bias'].shape}")

        # in_proj_a: (num_v_heads, hidden_size) -> (hidden_size, num_v_heads)
        save_projection(f"{prefix}in_proj_a.weight", transpose=True)

        # in_proj_b: (num_v_heads, hidden_size) -> (hidden_size, num_v_heads)
        save_projection(f"{prefix}in_proj_b.weight", transpose=True)

        # in_proj_z: (value_dim, hidden_size) -> (hidden_size, value_dim)
        save_projection(f"{prefix}in_proj_z.weight", transpose=True)

        # norm: (head_v_dim) - Qwen3_5RMSNormGated uses weight directly (NO +1.0)
        # Unlike Qwen3_5RMSNorm which uses (1+weight), the gated norm is
        # initialized to torch.ones and uses self.weight directly.
        save_weight(params[f"{prefix}norm.weight"])
        print(f"  {prefix}norm.weight (no +1.0): {params[f'{prefix}norm.weight'].shape}")

        # out_proj: (hidden_size, value_dim) -> (value_dim, hidden_size)
        save_projection(f"{prefix}out_proj.weight", transpose=True)

    def save_self_attn(layer_prefix):
        """Save self-attention weights in NNTrainer execution order.

        HF q_proj layout: (num_heads * head_dim * 2, hidden_size)
        When viewed as (num_heads, head_dim*2, hidden_size), each head has
        [query(head_dim), gate(head_dim)] interleaved. We must de-interleave
        into separate query and gate weight matrices.
        """
        prefix = f"{layer_prefix}self_attn."

        q_weight = params[f"{prefix}q_proj.weight"]  # (4096, 2048) for 2B
        total_out = q_weight.shape[0]
        num_heads_sa = total_out // (head_dim_sa * 2)  # 4096 / (256*2) = 8
        # De-interleave: extract query and gate rows per head
        q_query_rows = []
        q_gate_rows = []
        for h in range(num_heads_sa):
            start = h * head_dim_sa * 2
            q_query_rows.append(q_weight[start : start + head_dim_sa, :])
            q_gate_rows.append(q_weight[start + head_dim_sa : start + head_dim_sa * 2, :])
        q_query = torch.cat(q_query_rows, dim=0)  # (2048, 2048)
        q_gate = torch.cat(q_gate_rows, dim=0)    # (2048, 2048)

        # NNTrainer execution order for self-attn block:

        # 6) v_proj
        save_projection(f"{prefix}v_proj.weight", transpose=True)

        # 4) k_proj
        save_projection(f"{prefix}k_proj.weight", transpose=True)

        # 5) k_norm (NO +1)
        norm_key = f"{prefix}k_norm.weight"
        if norm_key in params:
            save_weight(params[norm_key], is_rms=True)
            print(f"  {norm_key} (no +1): {params[norm_key].shape}")

        # 2) q_query
        save_weight(q_query.permute(1, 0))
        print(f"  {prefix}q_proj.weight(query, de-interleaved): {q_query.shape}")

        # 1) q_gate
        save_weight(q_gate.permute(1, 0))
        print(f"  {prefix}q_proj.weight(gate, de-interleaved): {q_gate.shape}")

        # 3) q_norm (NO +1)
        norm_key = f"{prefix}q_norm.weight"
        if norm_key in params:
            save_weight(params[norm_key], is_rms=True)
            print(f"  {norm_key} (no +1): {params[norm_key].shape}")

        # 7) o_proj
        save_projection(f"{prefix}o_proj.weight", transpose=True)

    def save_feed_forward(layer_prefix):
        """Save MLP weights: up_proj, gate_proj, down_proj
        NOTE: Must match createMlp() layer creation order (gate first, then up)
        """
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            save_projection(f"{layer_prefix}mlp.{proj}.weight", transpose=True)
    
    # === Save weights in NNTrainer layer creation order ===

    # 1. Embedding
    save_weight(params["model.embed_tokens.weight"])
    print(f"model.embed_tokens.weight: {params['model.embed_tokens.weight'].shape}")

    # 2. Decoder layers
    for layer_idx in range(n_layers):
        layer_prefix = f"model.layers.{layer_idx}."
        is_self_attn = (layer_types[layer_idx] == 'full_attention')

        print(f"\n--- Layer {layer_idx} ({'self_attn' if is_self_attn else 'linear_attn'}) ---")

        # Input layernorm (RMS norm: add 1.0 for offset format)
        save_weight(params[f"{layer_prefix}input_layernorm.weight"], is_rms=True)
        print(f"  {layer_prefix}input_layernorm.weight (+1.0)")

        # Attention (linear or self)
        if is_self_attn:
            save_self_attn(layer_prefix)
        else:
            save_linear_attn(layer_prefix)

        # Post-attention layernorm (RMS norm: add 1.0)
        save_weight(params[f"{layer_prefix}post_attention_layernorm.weight"], is_rms=True)
        print(f"  {layer_prefix}post_attention_layernorm.weight")

        # MLP
        save_feed_forward(layer_prefix)

    # 3. Final norm (RMS norm: add 1.0)
    save_weight(params["model.norm.weight"], is_rms=True)
    print(f"\nmodel.norm.weight (+1.0): {params['model.norm.weight'].shape}")

    # 4. LM head (only if not tied)
    if not tie_word_embeddings:
        save_projection("lm_head.weight", transpose=True)
    else:
        print("tie_word_embeddings=true: skipping lm_head (shared with embedding)")

    print(f"\nTotal floats saved: {total_floats[0]:,}")
    print(f"Total bytes: {total_floats[0] * 4:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--output_name", type=str, default="./nntr_qwen3_5_2b_fp32.bin")
    parser.add_argument("--data_type", type=str, default="float32")
    args = parser.parse_args()

    data_dtype = args.data_type
    model_path = args.model_path
    output_name = args.output_name

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)

    # Load full VLM, then extract language model only
    full_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, trust_remote_code=True)
    full_model.eval()

    # Extract the text/language model part (skip vision encoder)
    if hasattr(full_model, 'model') and hasattr(full_model.model, 'layers'):
        # Direct access: model.layers exists -> use full_model directly
        model = full_model
    elif hasattr(full_model, 'language_model'):
        model = full_model.language_model
    elif hasattr(full_model, 'text_model'):
        model = full_model.text_model
    else:
        model = full_model
    print(f"Using model class: {type(model).__name__}")
    model.eval()

    text_cfg = get_text_config(config)
    print(f"Model: {model_path}")
    print(f"model_type: {getattr(config, 'model_type', 'N/A')}")
    print(f"tie_word_embeddings: {getattr(config, 'tie_word_embeddings', 'N/A')}")
    print(f"num_hidden_layers: {text_cfg.num_hidden_layers}")
    print(f"hidden_size: {text_cfg.hidden_size}")
    print(f"vocab_size: {text_cfg.vocab_size}")

    # Print all available state dict keys for verification
    sd = model.state_dict()
    text_keys = [k for k in sd.keys() if 'layers.0.' in k or 'embed' in k or k == 'model.norm.weight']
    print(f"\nSample state dict keys:")
    for k in sorted(text_keys)[:15]:
        print(f"  {k}: {sd[k].shape}")

    # Check if keys use a prefix (e.g., "model.text_model." vs "model.")
    sample_key = "model.embed_tokens.weight"
    if sample_key not in sd:
        # Try common VLM prefixes
        for prefix in ["model.text_model.", "text_model.", "language_model.model."]:
            alt_key = f"{prefix}embed_tokens.weight"
            if alt_key in sd:
                print(f"\nWARNING: state dict uses prefix '{prefix}'. Adjusting keys...")
                # Rename all keys
                new_sd = {}
                for k, v in sd.items():
                    if k.startswith(prefix):
                        new_sd["model." + k[len(prefix):]] = v
                    else:
                        new_sd[k] = v
                sd = new_sd
                break
        else:
            print(f"\nERROR: Cannot find '{sample_key}' in state dict!")
            print(f"Available keys with 'embed': {[k for k in sd.keys() if 'embed' in k]}")
            exit(1)

    with open(output_name, "wb") as f_model:
        save_qwen3_5_for_nntrainer(sd, config, data_dtype, f_model)

    file_size = os.path.getsize(output_name)
    print(f"\nSaved to {output_name}")
    print(f"File size: {file_size:,} bytes ({file_size / 1024**3:.2f} GB)")
    print(f"Expected floats: {file_size // 4:,}")
