# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

## @package generate_gemma4_moe_reference
## @brief Generate Hugging Face reference fixtures for a tiny Gemma4 MoE model.
## @author Jungwon-Lee <jungone.lee@samsung.com>

"""Generate Hugging Face reference fixtures for a tiny Gemma4 MoE model.

The NNTrainer binary is written through the production Gemma4 MoE converter so
the fixture also verifies PLE-disabled, K-equals-V, router, norm, and fused
expert ordering.
"""

import argparse
import importlib.util
import json
import pathlib
import types

import numpy as np
import torch
import transformers
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

from generate_gemma4_reference import INPUT_IDS, N_GEN, TINY_TOKENIZER


THIS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[4]
DEFAULT_OUT = THIS_DIR.parent / "gemma4_moe_tiny"
CONVERTER_PATH = (
    REPO_ROOT / "Applications/CausalLM/res/gemma4/gemma4_moe_weight_converter.py"
)

TINY_MOE_TEXT_CONFIG = dict(
    hidden_size=64,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=8,
    global_head_dim=8,
    num_global_key_value_heads=4,
    hidden_size_per_layer_input=0,
    vocab_size_per_layer_input=32,
    vocab_size=32,
    max_position_embeddings=8,
    rms_norm_eps=1e-6,
    rope_theta=1000000,
    sliding_window=4,
    layer_types=["sliding_attention", "full_attention"],
    tie_word_embeddings=True,
    hidden_activation="gelu_pytorch_tanh",
    attention_dropout=0.0,
    pad_token_id=0,
    num_kv_shared_layers=0,
    use_double_wide_mlp=False,
    attention_k_eq_v=True,
    enable_moe_block=True,
    num_experts=4,
    top_k_experts=2,
    moe_intermediate_size=32,
)


def load_production_converter():
    """Load the production converter directly from the repository."""
    spec = importlib.util.spec_from_file_location(
        "gemma4_moe_weight_converter", CONVERTER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(seed, cfg=None, random_layer_scalar=False, random_norms=False,
                router_scale=1.0):
    """Build a deterministic tiny Gemma4 MoE text model.

    The init leaves every norm weight, the router's input scale and its
    per-expert scale at one and the router near uniform, which a test of
    who-gets-which-weight cannot see through. random_layer_scalar puts the
    layer_scalar buffers at seeded values in [0.5, 1.5); random_norms puts
    every one of those vectors at 1 + 0.25 N(0, 1); router_scale multiplies
    the router projection so the top-k is not decided by rounding.
    """
    torch.manual_seed(seed)
    config = Gemma4TextConfig(**(cfg or TINY_MOE_TEXT_CONFIG))
    model = Gemma4TextModel(config)
    model.eval()
    with torch.no_grad():
        if random_layer_scalar:
            for layer in model.layers:
                layer.layer_scalar.fill_(0.5 + float(torch.rand(1)))
        if random_norms:
            for name, param in model.named_parameters():
                if param.dim() == 1 and (
                    "norm" in name or name.endswith("router.scale")
                    or name.endswith("per_expert_scale")
                ):
                    param.copy_(1.0 + 0.25 * torch.randn_like(param))
        if router_scale != 1.0:
            for layer in model.layers:
                layer.router.proj.weight.mul_(router_scale)
    return model


def convert_weights(model, output_path):
    """Write weights using the same ordered walker as production conversion."""
    converter = load_production_converter()
    wrapper_config = types.SimpleNamespace(text_config=model.config)
    with open(output_path, "wb") as output_file:
        converter.save_gemma4_moe_bin(
            model.state_dict(), wrapper_config, "float32", output_file, True
        )


def softcap(logits, cap):
    """Gemma4ForCausalLM's final logit soft-cap (none when cap is None)."""
    if cap is None:
        return logits
    return torch.tanh(logits / cap) * cap


def run_forward(model, input_ids, cap=None):
    """Return tied-embedding logits for the final prompt token."""
    ids = torch.tensor([input_ids], dtype=torch.long)
    with torch.no_grad():
        hidden = model(ids, use_cache=False).last_hidden_state[0, -1, :]
        logits = softcap(hidden @ model.embed_tokens.weight.T, cap)
    return logits.float().tolist()


def run_greedy_with_margin(model, input_ids, count, cap=None):
    """Generate greedy tokens and return the minimum top-two logit margin."""
    ids = list(input_ids)
    generated = []
    minimum_margin = float("inf")
    with torch.no_grad():
        for _ in range(count):
            inputs = torch.tensor([ids], dtype=torch.long)
            hidden = model(inputs, use_cache=False).last_hidden_state[0, -1, :]
            logits = softcap(hidden @ model.embed_tokens.weight.T, cap)
            top2 = torch.topk(logits.float(), k=2).values
            minimum_margin = min(
                minimum_margin, float((top2[0] - top2[1]).item())
            )
            token = int(logits.argmax().item())
            generated.append(token)
            ids.append(token)
    return generated, minimum_margin


def find_stable_seed(count, generated_tokens, cfg=None, cap=None, **knobs):
    """Select the seed with the largest minimum greedy logit margin."""
    candidates = []
    for seed in range(count):
        model = build_model(seed, cfg, **knobs)
        _, margin = run_greedy_with_margin(
            model, INPUT_IDS, generated_tokens, cap
        )
        candidates.append((margin, seed))
    candidates.sort(reverse=True)
    print(f"[search] best seeds={candidates[:10]}")
    return candidates[0][1]


def write_configs(output_dir, binary_name, tokenizer_path, cfg=None, cap=None):
    """Write the three configs consumed by the CausalLM differential tests."""
    c = cfg or TINY_MOE_TEXT_CONFIG
    config_json = {
        "architectures": ["Gemma4ForCausalLM"],
        "bos_token_id": 0,
        "eos_token_id": [31],
        "num_hidden_layers": c["num_hidden_layers"],
        "text_config": {
            "attention_k_eq_v": c["attention_k_eq_v"],
            "enable_moe_block": True,
            "global_head_dim": c["global_head_dim"],
            "head_dim": c["head_dim"],
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": c["hidden_size"],
            "hidden_size_per_layer_input": 0,
            "intermediate_size": c["intermediate_size"],
            "layer_types": c["layer_types"],
            "max_position_embeddings": c["max_position_embeddings"],
            "moe_intermediate_size": c["moe_intermediate_size"],
            "num_attention_heads": c["num_attention_heads"],
            "num_experts": c["num_experts"],
            "num_global_key_value_heads": c["num_global_key_value_heads"],
            "num_hidden_layers": c["num_hidden_layers"],
            "num_key_value_heads": c["num_key_value_heads"],
            "num_kv_shared_layers": 0,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000,
                },
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": 1000000,
                    "partial_rotary_factor": 0.25,
                },
            },
            "rope_theta": 1000000,
            "sliding_window": c["sliding_window"],
            "tie_word_embeddings": True,
            "top_k_experts": c["top_k_experts"],
            "use_double_wide_mlp": False,
            "vocab_size": 32,
            "vocab_size_per_layer_input": 32,
        },
    }
    if cap is not None:
        config_json["text_config"]["final_logit_softcapping"] = cap
    generation_json = {
        "bos_token_id": 0,
        "eos_token_id": 31,
        "do_sample": False,
        "top_k": 1,
        "top_p": 1.0,
        "temperature": 1.0,
    }
    nntrainer_json = {
        "bad_word_ids": [],
        "batch_size": 1,
        "embedding_dtype": "FP32",
        "fc_layer_dtype": "FP32",
        "init_seq_len": 4,
        "lmhead_dtype": "FP32",
        "max_seq_len": c["max_position_embeddings"],
        "model_file_name": binary_name,
        "model_tensor_type": "FP32-FP32",
        "model_type": "CausalLM",
        "num_to_generate": 1,
        "tokenizer_file": pathlib.Path(tokenizer_path).name,
    }
    for filename, payload in (
        ("config.json", config_json),
        ("generation_config.json", generation_json),
        ("nntr_config.json", nntrainer_json),
    ):
        with open(output_dir / filename, "w") as output_file:
            json.dump(payload, output_file, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Generate tiny Gemma4 MoE Hugging Face fixtures"
    )
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    # Seed 17 was selected from [0, 100) for a 0.102 minimum greedy margin.
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--n", type=int, default=N_GEN)
    parser.add_argument(
        "--search-seeds",
        type=int,
        default=0,
        help="Search seeds [0, N) and use the one with the widest greedy margin",
    )
    parser.add_argument("--transformers-commit", default="unknown")
    # [plan 201 S4] the HTP decode fixture's shape (head_dim >= 64: the
    # NPU's attention kinds take multiples of 64); unset = the hd8 fixture
    for flag, key in (
        ("--hidden", "hidden_size"),
        ("--inter", "intermediate_size"),
        ("--heads", "num_attention_heads"),
        ("--kv-heads", "num_key_value_heads"),
        ("--head-dim", "head_dim"),
        ("--global-head-dim", "global_head_dim"),
        ("--global-kv-heads", "num_global_key_value_heads"),
        ("--max-pos", "max_position_embeddings"),
        ("--sliding-window", "sliding_window"),
        ("--experts", "num_experts"),
        ("--top-k", "top_k_experts"),
        ("--moe-inter", "moe_intermediate_size"),
    ):
        parser.add_argument(flag, dest=key, type=int, default=None)
    parser.add_argument(
        "--layer-types",
        default=None,
        help="comma-separated sliding_attention / full_attention",
    )
    parser.add_argument(
        "--final-softcap",
        type=float,
        default=None,
        help="final_logit_softcapping, applied to the reference logits too",
    )
    parser.add_argument(
        "--random-layer-scalar",
        action="store_true",
        help="seeded layer_scalar values in [0.5, 1.5) instead of ones",
    )
    parser.add_argument(
        "--random-norms",
        action="store_true",
        help="norm weights, router scale and per-expert scale at 1 + 0.25 N",
    )
    parser.add_argument(
        "--router-scale",
        type=float,
        default=1.0,
        help="multiply the router projection (a decisive top-k)",
    )
    args = parser.parse_args()
    knobs = dict(
        random_layer_scalar=args.random_layer_scalar,
        random_norms=args.random_norms,
        router_scale=args.router_scale,
    )

    cfg = dict(TINY_MOE_TEXT_CONFIG)
    for key in list(cfg):
        if getattr(args, key, None) is not None:
            cfg[key] = getattr(args, key)
    if args.layer_types:
        cfg["layer_types"] = args.layer_types.split(",")
        cfg["num_hidden_layers"] = len(cfg["layer_types"])
    cap = args.final_softcap

    if args.search_seeds > 0:
        args.seed = find_stable_seed(args.search_seeds, args.n, cfg, cap,
                                     **knobs)

    output_dir = args.out.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(args.seed, cfg, **knobs)
    print(f"[generate] parameters={sum(p.numel() for p in model.parameters()):,}")

    binary_name = "nntr_gemma4_moe_tiny_fp32.bin"
    convert_weights(model, output_dir / binary_name)

    tokenizer_path = output_dir / "tokenizer.json"
    with open(tokenizer_path, "w") as output_file:
        json.dump(TINY_TOKENIZER, output_file, indent=2)
    write_configs(output_dir, binary_name, tokenizer_path, cfg, cap)

    logits = run_forward(model, INPUT_IDS, cap)
    tokens, greedy_margin = run_greedy_with_margin(
        model, INPUT_IDS, args.n, cap
    )
    for filename, payload in (
        ("input_ids.json", INPUT_IDS),
        ("reference_logits.json", logits),
        ("reference_tokens.json", tokens),
    ):
        with open(output_dir / filename, "w") as output_file:
            json.dump(payload, output_file)

    sorted_logits = np.sort(np.asarray(logits, dtype=np.float32))
    top2_margin = float(sorted_logits[-1] - sorted_logits[-2])
    meta = {
        "seed": args.seed,
        "n_gen": args.n,
        "input_ids": INPUT_IDS,
        "logits_atol_fp32": 1e-2,
        "logits_atol_q40": 5.0,
        "prefix_match_min": 2,
        "top2_logit_margin": top2_margin,
        "minimum_greedy_margin": greedy_margin,
        "transformers_version": transformers.__version__,
        "transformers_commit": args.transformers_commit,
        "torch_version": torch.__version__,
    }
    with open(output_dir / "meta.json", "w") as output_file:
        json.dump(meta, output_file, indent=2)

    print(
        f"[generate] seed={args.seed}, argmax={int(np.argmax(logits))}, "
        f"margin={top2_margin}, greedy_margin={greedy_margin}"
    )
    print(f"[generate] tokens={tokens}")
    print(f"[generate] output={output_dir}")


if __name__ == "__main__":
    main()
