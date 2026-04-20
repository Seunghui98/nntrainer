"""
weight_converter.py – Qwen3.5-2B → NNTrainer binary format

Usage:
    python weight_converter.py \
        --model_path ./Qwen3.5-2B \
        --output     ./nntr_qwen35_2b_fp32.bin \
        --dtype      float32

For vision-language variants that store the language model under a sub-key
(e.g. Qwen3.5-VL), pass --lm_prefix "language_model":
    python weight_converter.py \
        --model_path ./Qwen3.5-2B-Instruct \
        --output     ./nntr_qwen35_2b_fp32.bin \
        --lm_prefix  language_model

Weight saving order matches NNTrainer layer construction order
(Transformer::constructModel + Qwen35Transformer::createAttention +
Transformer::createMlp):

  embed_tokens
  for i in range(num_layers):
    input_layernorm             # attention pre-norm
    v_proj (transposed)
    k_proj (transposed)
    k_norm
    q_proj (transposed)
    q_norm
    o_proj (transposed)
    post_attention_layernorm    # FFN pre-norm
    mlp.up_proj   (transposed)
    mlp.gate_proj (transposed)
    mlp.down_proj (transposed)
  model.norm
  lm_head (transposed)          # only if NOT tie_word_embeddings
"""

import argparse
import sys
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM


def save_weight(tensor: torch.Tensor, dtype: str, f) -> None:
    np_dtype = np.float32 if dtype in ("float32", "fp32") else np.float16
    arr = tensor.detach().cpu().to(torch.float32 if np_dtype == np.float32
                                   else torch.float16).numpy()
    f.write(arr.astype(np_dtype).tobytes())


def convert(model_path: str, output_path: str, dtype: str,
            lm_prefix: str = "") -> None:
    """
    Convert Qwen3.5-2B HuggingFace weights to NNTrainer binary.

    Args:
        model_path:  Path to the local HuggingFace model directory.
        output_path: Output .bin file path.
        dtype:       "float32" or "float16".
        lm_prefix:   If non-empty, all weight keys are expected under
                     "model.<lm_prefix>.model.*" (for VL models).
                     Set to "language_model" for vision-language variants.
    """
    print(f"Loading model from: {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model  = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, trust_remote_code=True)
    model.eval()

    params = model.state_dict()

    # Build the key prefix based on whether this is a VL model
    # Standard:  model.embed_tokens / model.layers.X / model.norm / lm_head
    # VL variant: model.language_model.model.embed_tokens / ... / lm_head
    if lm_prefix:
        base = f"model.{lm_prefix}.model"
        lmhead_key = f"model.{lm_prefix}.lm_head.weight"
    else:
        base = "model"
        lmhead_key = "lm_head.weight"

    # Detect actual key format (some models omit the "model." prefix)
    if f"{base}.embed_tokens.weight" not in params:
        alt = base.replace("model.", "", 1)
        if f"{alt}.embed_tokens.weight" in params:
            base = alt

    n_layers = config.num_hidden_layers
    tie      = getattr(config, "tie_word_embeddings", False)

    print(f"  layers={n_layers}, tie_word_embeddings={tie}")
    print(f"  key prefix: {base}")

    with open(output_path, "wb") as f:
        # 1. Embedding
        emb_key = f"{base}.embed_tokens.weight"
        print(f"  saving {emb_key}")
        save_weight(params[emb_key], dtype, f)

        # 2. Decoder layers
        for i in range(n_layers):
            lp = f"{base}.layers.{i}."

            # Attention pre-norm
            save_weight(params[f"{lp}input_layernorm.weight"], dtype, f)

            # V projection (transposed: [out,in] → [in,out])
            save_weight(params[f"{lp}self_attn.v_proj.weight"].T, dtype, f)

            # K projection + K norm
            save_weight(params[f"{lp}self_attn.k_proj.weight"].T, dtype, f)
            save_weight(params[f"{lp}self_attn.k_norm.weight"], dtype, f)

            # Q projection + Q norm
            save_weight(params[f"{lp}self_attn.q_proj.weight"].T, dtype, f)
            save_weight(params[f"{lp}self_attn.q_norm.weight"], dtype, f)

            # O projection
            save_weight(params[f"{lp}self_attn.o_proj.weight"].T, dtype, f)

            # FFN pre-norm
            save_weight(params[f"{lp}post_attention_layernorm.weight"], dtype, f)

            # FFN: up, gate, down (SwiGLU)
            save_weight(params[f"{lp}mlp.up_proj.weight"].T,   dtype, f)
            save_weight(params[f"{lp}mlp.gate_proj.weight"].T, dtype, f)
            save_weight(params[f"{lp}mlp.down_proj.weight"].T, dtype, f)

            if (i + 1) % 4 == 0:
                print(f"  layer {i+1}/{n_layers} done")

        # 3. Final norm
        save_weight(params[f"{base}.norm.weight"], dtype, f)

        # 4. LM head (only if weights are NOT tied)
        if not tie:
            if lmhead_key in params:
                print(f"  saving lm_head: {lmhead_key}")
                save_weight(params[lmhead_key].T, dtype, f)
            else:
                print(f"  WARNING: lm_head key '{lmhead_key}' not found; "
                      "skipping (tie_word_embeddings may be True in the model)")

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Qwen3.5-2B weights to NNTrainer binary format")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to local HuggingFace model directory")
    parser.add_argument("--output", type=str,
                        default="nntr_qwen35_2b_fp32.bin",
                        help="Output binary file name")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16"],
                        help="Output data type (default: float32)")
    parser.add_argument("--lm_prefix", type=str, default="",
                        help="Sub-model prefix for VL models "
                             "(e.g. 'language_model'); leave empty for "
                             "language-only Qwen3.5 models")
    args = parser.parse_args()
    convert(args.model_path, args.output, args.dtype, args.lm_prefix)
