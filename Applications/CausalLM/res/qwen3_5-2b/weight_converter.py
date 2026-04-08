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

    print(f"Layer types: {layer_types}")

    total_floats = [0]  # mutable counter

    def save_weight(weight):
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

        # norm: (head_v_dim)
        save_weight(params[f"{prefix}norm.weight"])
        print(f"  {prefix}norm.weight: {params[f'{prefix}norm.weight'].shape}")

        # out_proj: (hidden_size, value_dim) -> (value_dim, hidden_size)
        save_projection(f"{prefix}out_proj.weight", transpose=True)

    def save_self_attn(layer_prefix):
        """Save self-attention layer weights.

        Weight order matches Qwen3_5Transformer::createAttention():
          V, K, K_norm, Q_query, Q_gate, Q_norm, mha_core(no wt), attn_gate(no wt), O

        Note: q_proj is (num_heads*head_dim*2, hidden_size) because of attn_output_gate.
              First half = query, second half = gate. We split and save separately.
        """
        prefix = f"{layer_prefix}self_attn."

        # V projection
        save_projection(f"{prefix}v_proj.weight", transpose=True)

        # K projection
        save_projection(f"{prefix}k_proj.weight", transpose=True)

        # K norm
        norm_key = f"{prefix}k_norm.weight"
        if norm_key in params:
            save_weight(params[norm_key])
            print(f"  {norm_key}: {params[norm_key].shape}")

        # Q projection - split into query and gate halves
        q_weight = params[f"{prefix}q_proj.weight"]  # (num_heads*head_dim*2, hidden_size)
        q_half = q_weight.shape[0] // 2
        q_query = q_weight[:q_half, :]   # first half: query
        q_gate = q_weight[q_half:, :]    # second half: gate
        print(f"  {prefix}q_proj.weight: {q_weight.shape} -> split into query{q_query.shape} + gate{q_gate.shape}")
        save_weight(q_query.permute(1, 0))  # Q_query FC weight
        save_weight(q_gate.permute(1, 0))   # Q_gate FC weight

        # Q norm
        norm_key = f"{prefix}q_norm.weight"
        if norm_key in params:
            save_weight(params[norm_key])
            print(f"  {norm_key}: {params[norm_key].shape}")

        # O projection
        save_projection(f"{prefix}o_proj.weight", transpose=True)

    def save_feed_forward(layer_prefix):
        """Save MLP weights: up_proj, gate_proj, down_proj"""
        for proj in ["up_proj", "gate_proj", "down_proj"]:
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

        # Input layernorm
        save_weight(params[f"{layer_prefix}input_layernorm.weight"])
        print(f"  {layer_prefix}input_layernorm.weight")

        # Attention (linear or self)
        if is_self_attn:
            save_self_attn(layer_prefix)
        else:
            save_linear_attn(layer_prefix)

        # Post-attention layernorm
        save_weight(params[f"{layer_prefix}post_attention_layernorm.weight"])
        print(f"  {layer_prefix}post_attention_layernorm.weight")

        # MLP
        save_feed_forward(layer_prefix)

    # 3. Final norm
    save_weight(params["model.norm.weight"])
    print(f"\nmodel.norm.weight: {params['model.norm.weight'].shape}")

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
