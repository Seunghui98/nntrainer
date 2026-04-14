## @file weight_converter.py
## @brief weight conversion script for LFM2/LFM2.5 hybrid model
## (double-gated short conv + GQA attention)

import argparse
import os
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def save_lfm2_for_nntrainer(params, config, dtype, file):
    """Convert and save LFM2/LFM2.5 model weights for NNTrainer

    Weight save order must match the addLayer() order in
    LFM2Transformer::constructModel().
    """
    n_layers = config.num_hidden_layers
    tie_embedding = getattr(config, 'tie_embedding', True)
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = hidden_size // num_heads

    # Determine layer types
    layer_types = config.layer_types

    print(f"Layer types: {layer_types}")
    print(f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")

    total_floats = [0]

    def save_weight(weight, is_rms=False):
        """Save weight tensor.
        LFM2 RMSNorm: NO +1.0 offset needed.
        nntrainer RMSNorm does: output = gamma * x / rms(x)
        LFM2 HF RMSNorm: output = weight * x / rms(x)
        Both use weight directly, so save as-is.
        """
        arr = weight.detach().float().cpu().numpy().astype(dtype)
        arr.tofile(file)
        total_floats[0] += arr.size

    def save_projection(key, transpose=True):
        w = params[key]
        if transpose:
            w = w.permute(1, 0)
        save_weight(w)
        print(f"  {key}: {params[key].shape}")

    def save_conv_block(layer_prefix):
        """Save decomposed LFM2 conv block weights.

        Layer order in constructModel (createConvBlock):
          1. RMSNorm (operator_norm)
          2. FC in_proj (hidden → 3*hidden)     [fully_connected weight]
          3. LFM2Conv gated_conv:
             - weight[0]: conv kernel (hidden, kernel_size)
          4. FC out_proj (hidden → hidden)       [fully_connected weight]
        """
        prefix = f"{layer_prefix}conv."

        # FC: in_proj (hidden_size, 3*hidden_size)
        save_projection(f"{prefix}in_proj.weight", transpose=True)

        # LFM2Conv: conv1d kernel (hidden_size, kernel_size)
        # HF stores as (hidden_size, 1, kernel_size) → squeeze to (hidden_size, kernel_size)
        conv_w = params[f"{prefix}conv.weight"]
        conv_w = conv_w.squeeze(1)
        save_weight(conv_w)
        print(f"  {prefix}conv.weight: {params[f'{prefix}conv.weight'].shape} -> {conv_w.shape}")

        # FC: out_proj (hidden_size, hidden_size)
        save_projection(f"{prefix}out_proj.weight", transpose=True)

    def save_attention_block(layer_prefix):
        """Save LFM2 attention block weights.

        Layer order in constructModel (createAttention):
          V -> K -> K_norm -> Q -> Q_norm -> MHA -> O
        """
        prefix = f"{layer_prefix}self_attn."

        # V projection
        save_projection(f"{prefix}v_proj.weight", transpose=True)

        # K projection
        save_projection(f"{prefix}k_proj.weight", transpose=True)

        # K norm (RMSNorm, add +1.0)
        save_weight(params[f"{prefix}k_layernorm.weight"], is_rms=True)
        print(f"  {prefix}k_layernorm.weight (+1.0)")

        # Q projection
        save_projection(f"{prefix}q_proj.weight", transpose=True)

        # Q norm (RMSNorm, add +1.0)
        save_weight(params[f"{prefix}q_layernorm.weight"], is_rms=True)
        print(f"  {prefix}q_layernorm.weight (+1.0)")

        # O projection
        save_projection(f"{prefix}out_proj.weight", transpose=True)

    def save_feed_forward(layer_prefix):
        """Save MLP weights: w1 (gate), w3 (up), w2 (down)
        Must match createMlp() order: gate first, then up, then down.
        """
        save_projection(f"{layer_prefix}feed_forward.w1.weight", transpose=True)
        save_projection(f"{layer_prefix}feed_forward.w3.weight", transpose=True)
        save_projection(f"{layer_prefix}feed_forward.w2.weight", transpose=True)

    # === Save weights in NNTrainer layer creation order ===

    # 1. Embedding
    save_weight(params["model.embed_tokens.weight"])
    print(f"model.embed_tokens.weight: {params['model.embed_tokens.weight'].shape}")

    # 2. Embedding norm (LFM2 specific)
    save_weight(params["model.embedding_norm.weight"], is_rms=True)
    print(f"model.embedding_norm.weight (+1.0): {params['model.embedding_norm.weight'].shape}")

    # 3. Decoder layers
    for layer_idx in range(n_layers):
        layer_prefix = f"model.layers.{layer_idx}."
        is_attn = (layer_types[layer_idx] == 'full_attention')

        print(f"\n--- Layer {layer_idx} ({'attention' if is_attn else 'conv'}) ---")

        # Operator norm (RMSNorm, add 1.0)
        save_weight(params[f"{layer_prefix}operator_norm.weight"], is_rms=True)
        print(f"  {layer_prefix}operator_norm.weight (+1.0)")

        # Attention or Conv
        if is_attn:
            save_attention_block(layer_prefix)
        else:
            save_conv_block(layer_prefix)

        # FFN norm (RMSNorm, add 1.0)
        save_weight(params[f"{layer_prefix}ffn_norm.weight"], is_rms=True)
        print(f"  {layer_prefix}ffn_norm.weight (+1.0)")

        # MLP
        save_feed_forward(layer_prefix)

    # 4. Output norm
    # LFM2 does NOT have a separate final norm (no model.norm.weight).
    # NNTrainer model creates an output_norm RMSNorm layer before lm_head.
    # Save identity weights: all zeros → becomes all 1.0 after +1.0 offset
    # → RMSNorm with gamma=1.0 is identity.
    if "model.norm.weight" in params:
        save_weight(params["model.norm.weight"], is_rms=True)
        print(f"\nmodel.norm.weight (+1.0): {params['model.norm.weight'].shape}")
    else:
        # Save identity norm: all 1.0 (gamma=1.0 is identity for RMSNorm)
        identity_norm = torch.ones(hidden_size)
        save_weight(identity_norm)
        print(f"\noutput_norm: synthetic identity (all 1.0), shape=({hidden_size},)")

    # 5. LM head (only if not tied)
    if not tie_embedding:
        save_projection("lm_head.weight", transpose=True)
        print(f"lm_head.weight: {params['lm_head.weight'].shape}")
    else:
        print("tie_embedding=true: skipping lm_head (shared with embedding)")

    print(f"\nTotal floats saved: {total_floats[0]:,}")
    print(f"Total bytes: {total_floats[0] * 4:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--output_name", type=str, default="./nntr_lfm2_5_350m_fp32.bin")
    parser.add_argument("--data_type", type=str, default="float32")
    args = parser.parse_args()

    data_dtype = args.data_type
    model_path = args.model_path
    output_name = args.output_name

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, trust_remote_code=True)
    model.eval()

    print(f"Model: {model_path}")
    print(f"model_type: {getattr(config, 'model_type', 'N/A')}")
    print(f"tie_embedding: {getattr(config, 'tie_embedding', 'N/A')}")
    print(f"num_hidden_layers: {config.num_hidden_layers}")
    print(f"hidden_size: {config.hidden_size}")
    print(f"vocab_size: {config.vocab_size}")
    print(f"intermediate_size: {config.intermediate_size}")

    sd = model.state_dict()

    # Print sample keys
    text_keys = [k for k in sd.keys() if 'layers.0.' in k or 'embed' in k or 'norm' in k]
    print(f"\nSample state dict keys:")
    for k in sorted(text_keys)[:20]:
        print(f"  {k}: {sd[k].shape}")

    with open(output_name, "wb") as f_model:
        save_lfm2_for_nntrainer(sd, config, data_dtype, f_model)

    file_size = os.path.getsize(output_name)
    print(f"\nSaved to {output_name}")
    print(f"File size: {file_size:,} bytes ({file_size / 1024**3:.2f} GB)")
