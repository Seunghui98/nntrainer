## @file weight_converter.py
## @brief weight conversion script for qwen3.5 model
## @note  Qwen3.5 uses the same architecture as Qwen3 (Qwen3ForCausalLM)
##        but typically uses tie_word_embeddings=true for smaller models.

import argparse
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

def get_text_config(config):
    """Extract text model config from potentially nested VLM config (e.g. Qwen3.5)"""
    if hasattr(config, 'text_config'):
        return config.text_config
    return config

def save_qwen3_for_nntrainer(params, config, dtype, file):
    """Convert and save weights as nntrainer format for multi-head attention model"""

    # Qwen3.5 is a VLM with nested config: use text_config for the LM part
    text_cfg = get_text_config(config)
    n_layers = text_cfg.num_hidden_layers
    tie_word_embeddings = getattr(config, 'tie_word_embeddings', False)

    def save_weight(weight):
        np.array(weight.float(), dtype=dtype).tofile(file)

    def save_projection(layer_name, proj_name):
        """Helper function to handle base/lora weight saving"""
        lora_key = f"{layer_name}{proj_name}.lora_A.default.weight"
        if lora_key in params:
            save_weight(params[f"{layer_name}{proj_name}.base_layer.weight"].permute(1, 0))
            save_weight(params[f"{layer_name}{proj_name}.lora_A.default.weight"].permute(1, 0))
            save_weight(params[f"{layer_name}{proj_name}.lora_B.default.weight"].permute(1, 0))
        else:
            save_weight(params[f"{layer_name}{proj_name}.weight"].permute(1, 0))

    def save_attention(layer_name):
        """Save attention layer weights"""
        save_weight(params[f"{layer_name}input_layernorm.weight"])

        # Save V/K/Q/O projections to match NNTrainer layer creation order
        for proj in ["v_proj", "k_proj", "q_proj", "o_proj"]:
            save_projection(layer_name, f"self_attn.{proj}")
            # Qwen3: save norm weight after corresponding projection
            proj_norm_name = f"{layer_name}self_attn.{proj[0]}_norm.weight"
            if proj_norm_name in params:
                print(proj_norm_name)
                save_weight(params[proj_norm_name])

    def save_feed_forward(layer_name):
        """Save feed forward layer weights"""
        save_weight(params[f"{layer_name}post_attention_layernorm.weight"])

        # Save MLP projections using helper
        for proj in ["up_proj", "gate_proj", "down_proj"]:
            save_projection(layer_name, f"mlp.{proj}")

    # Save embedding layer
    save_weight(params["model.embed_tokens.weight"])

    # Process all layers
    for layer_idx in range(n_layers):
        layer_prefix = f"model.layers.{layer_idx}."
        save_attention(layer_prefix)
        save_feed_forward(layer_prefix)

    # Save final layers
    save_weight(params["model.norm.weight"])

    # Only save lm_head weights if not using tie_word_embeddings
    if not tie_word_embeddings:
        save_weight(params["lm_head.weight"].permute(1, 0))
    else:
        print("tie_word_embeddings=true: skipping lm_head (shared with embedding)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-2B")
    parser.add_argument("--output_name", type=str, default="./nntr_qwen3_5_2b_fp32.bin")
    parser.add_argument("--data_type", type=str, default="float32")
    args = parser.parse_args()

    data_dtype = args.data_type
    model_path = args.model_path
    output_name = args.output_name

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="float", trust_remote_code=True)
    model.eval()

    text_cfg = get_text_config(config)
    print(f"Model: {model_path}")
    print(f"model_type: {getattr(config, 'model_type', 'N/A')}")
    print(f"tie_word_embeddings: {getattr(config, 'tie_word_embeddings', 'N/A')}")
    print(f"num_hidden_layers: {text_cfg.num_hidden_layers}")
    print(f"hidden_size: {text_cfg.hidden_size}")
    print(f"vocab_size: {text_cfg.vocab_size}")

    # Print state dict keys for debugging
    sd = model.state_dict()
    print(f"\n=== State dict keys (layer 0) ===")
    for k in sorted(sd.keys()):
        if 'layers.0.' in k or 'embed' in k or 'norm' in k or 'lm_head' in k:
            print(f"  {k}: {sd[k].shape}")

    with open(output_name, "wb") as f_model:
        save_qwen3_for_nntrainer(sd, config, data_dtype, f_model)

    print(f"Saved to {output_name}")
