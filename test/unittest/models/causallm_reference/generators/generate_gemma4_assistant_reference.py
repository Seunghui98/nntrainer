# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

## @file generate_gemma4_assistant_reference.py
## @brief Golden fixture for the Gemma 4 assistant (MTP drafter) test.
##
## A tiny Gemma4AssistantForCausalLM with random weights, random shared K/V
## of one sliding and one full target layer, and two chained drafting steps
## run by transformers; the pack is written by the production converter
## (res/gemma4/assistant_weight_converter.py).
##
## Usage: python3 generate_gemma4_assistant_reference.py [--out <dir>] [--seed <int>]

import argparse
import json
import pathlib
import sys

import torch
from transformers import Gemma4AssistantConfig, Gemma4AssistantForCausalLM
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

THIS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[4]
sys.path.insert(0, str(REPO_ROOT / "Applications" / "CausalLM" / "res" / "gemma4"))
import assistant_weight_converter as conv  # noqa: E402

BACKBONE = 24
WINDOW = 4
KV_ROWS = 7  # > WINDOW + 1, so the sliding layer sees a cut window
TEXT = dict(hidden_size=32, intermediate_size=48, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2,
            num_global_key_value_heads=1, head_dim=8, global_head_dim=16,
            attention_k_eq_v=True, vocab_size=32, max_position_embeddings=64,
            sliding_window=WINDOW, rms_norm_eps=1e-6,
            layer_types=["sliding_attention", "full_attention"],
            rope_parameters={
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {"rope_type": "proportional",
                                   "rope_theta": 1000000.0,
                                   "partial_rotary_factor": 0.25}},
            hidden_size_per_layer_input=0, vocab_size_per_layer_input=0,
            num_kv_shared_layers=2, enable_moe_block=False,
            use_double_wide_mlp=False, tie_word_embeddings=True,
            hidden_activation="gelu_pytorch_tanh", pad_token_id=0)


def rows(t):
    """[1, n_kv, L, hd] -> [L][n_kv * hd], the KV cache layout."""
    return t[0].permute(1, 0, 2).reshape(t.shape[2], -1).flatten().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path,
                        default=THIS_DIR.parent / "gemma4_assistant_tiny")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    cfg = Gemma4AssistantConfig(text_config=Gemma4TextConfig(**TEXT),
                                backbone_hidden_size=BACKBONE)
    cfg.architectures = ["Gemma4AssistantForCausalLM"]
    model = Gemma4AssistantForCausalLM(cfg).eval()
    for _, p in model.named_parameters():
        p.data.normal_(0.0, 0.3)

    n = KV_ROWS
    ks, vs = torch.randn(1, 2, n, 8), torch.randn(1, 2, n, 8)
    kf, vf = torch.randn(1, 1, n, 16), torch.randn(1, 1, n, 16)
    shared = {"sliding_attention": (ks, vs), "full_attention": (kf, vf)}
    steps, hidden = [], torch.randn(1, 1, BACKBONE)
    for _ in range(2):
        x = torch.cat([torch.randn(1, 1, BACKBONE), hidden], dim=-1)
        with torch.no_grad():
            o = model(inputs_embeds=x, position_ids=torch.tensor([[n]]),
                      attention_mask=torch.ones(1, n + 1, dtype=torch.long),
                      shared_kv_states=shared)
        hidden = o.last_hidden_state
        steps.append({"input": x.flatten().tolist(),
                      "logits": o.logits.flatten().tolist(),
                      "hidden": hidden.flatten().tolist()})

    sd = {k: v.float().numpy() for k, v in model.state_dict().items()}
    with (out / conv.MODEL_FILE).open("wb") as stream:
        conv.write_weights(sd, TEXT["num_hidden_layers"], stream)
    (out / "config.json").write_text(
        json.dumps(conv.model_config(cfg.to_dict()), indent=2))
    (out / "generation_config.json").write_text("{}\n")
    (out / "nntr_config.json").write_text(
        json.dumps(conv.nntr_config(conv.MODEL_FILE), indent=2))
    # transformers' bidirectional sliding mask keeps the last window + 1 rows
    sl = WINDOW + 1
    (out / "reference.json").write_text(json.dumps({
        "pos": n, "full_rows": n, "sliding_rows": sl,
        "k_sliding": rows(ks[:, :, n - sl:]), "v_sliding": rows(vs[:, :, n - sl:]),
        "k_full": rows(kf), "v_full": rows(vf), "steps": steps}))
    print(f"[generate] {out}: argmax per step",
          [max(range(len(s["logits"])), key=s["logits"].__getitem__) for s in steps])


if __name__ == "__main__":
    main()
