## @file pytorch_layer0_probe.py
## @brief Reference probe — runs Qwen3-0.6B on the same prompt nntr_causallm
##        uses and dumps layer 0 Q/K/V (pre-rope, post-rope) and attention
##        output for token positions 0, 1, and last.
##
## Requirements
## ============
## transformers >= 4.51 (Qwen3 architecture support was added in 4.51).
## If you see "KeyError: 'qwen3'" or "model type `qwen3` … is out of date",
## upgrade:
##     pip install -U transformers tokenizers
##     # or, in a conda env: pip install -U 'transformers>=4.51'
##
## Compare against the nntrainer side `[mha_core probe]` lines:
##   PyTorch q_proj output[0, t, 0:4]   ↔ nntrainer query_step / Q[t,h=0]
##   PyTorch k_proj output[0, t, 0:4]   ↔ nntrainer key_step / K[t,h=0]
##   PyTorch v_proj output[0, t, 0:4]   ↔ nntrainer value_step
##   PyTorch attn_output[0, t, 0:4]     ↔ nntrainer attn_out[t]
##
## If the PRE-rope values match but POST-rope diverges, RoPE is broken in
## nntrainer. If PRE-rope already differs, FC weights / earlier layers are
## the suspect.
##
## Usage:
##   python pytorch_layer0_probe.py [--model Qwen/Qwen3-0.6B]
##                                  [--prompt "..."]

import argparse
import sys
from typing import Dict

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def _check_transformers_version():
    """Qwen3 native support landed in transformers 4.51. Older releases will
    fail with KeyError: 'qwen3' inside AutoConfig.from_pretrained. Surface a
    helpful upgrade hint instead of the cryptic stack trace."""
    try:
        major, minor, *_ = transformers.__version__.split(".")
        major_i, minor_i = int(major), int(minor)
    except Exception:
        return
    if (major_i, minor_i) < (4, 51):
        print(
            f"\n[fatal] transformers=={transformers.__version__} is too old "
            "to understand Qwen3 (need >= 4.51).\n"
            "        Run:  pip install -U 'transformers>=4.51' tokenizers\n",
            file=sys.stderr,
        )
        sys.exit(2)


def make_hook(name: str, store: Dict[str, torch.Tensor]):
    def _hook(_module, _inp, output):
        # Output of an nn.Linear is just the projected tensor.
        if isinstance(output, tuple):
            store[name] = output[0].detach().cpu()
        else:
            store[name] = output.detach().cpu()
    return _hook


def fmt(t: torch.Tensor, n: int = 4) -> str:
    return " ".join(f"{x:.6g}" for x in t.flatten()[:n].tolist())


def main():
    _check_transformers_version()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--prompt",
        default=("<|im_start|>user\nGive me a short introduction to large "
                 "language model.<|im_end|>\n<|im_start|>assistant\n"),
        help="Same prompt the C++ nntr_causallm runs.")
    args = parser.parse_args()

    print(f"transformers=={transformers.__version__}, torch=={torch.__version__}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, trust_remote_code=True)
    model.eval()

    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    print(f"hidden={cfg.hidden_size} n_heads_q={cfg.num_attention_heads} "
          f"n_heads_kv={cfg.num_key_value_heads} head_dim={head_dim} "
          f"rope_theta={getattr(cfg, 'rope_theta', None)}")

    layer0 = model.model.layers[0].self_attn
    store: Dict[str, torch.Tensor] = {}
    handles = [
        layer0.q_proj.register_forward_hook(make_hook("q_proj_out", store)),
        layer0.k_proj.register_forward_hook(make_hook("k_proj_out", store)),
        layer0.v_proj.register_forward_hook(make_hook("v_proj_out", store)),
        layer0.o_proj.register_forward_hook(make_hook("o_proj_in",  store)),
    ]

    # q_norm / k_norm exist on Qwen3, not Qwen2.
    if hasattr(layer0, "q_norm"):
        handles.append(
            layer0.q_norm.register_forward_hook(make_hook("q_norm_out", store)))
    if hasattr(layer0, "k_norm"):
        handles.append(
            layer0.k_norm.register_forward_hook(make_hook("k_norm_out", store)))

    inputs = tok(args.prompt, return_tensors="pt")
    n_tok = inputs.input_ids.shape[-1]
    print(f"\nprompt token count = {n_tok}")

    with torch.no_grad():
        out = model(**inputs, output_attentions=False, use_cache=False)

    for h in handles:
        h.remove()

    # Probe positions: 0, 1, and the last token of the prompt.
    positions = sorted({0, 1, n_tok - 1})

    def dump(label: str, t: torch.Tensor):
        # Most outputs are [batch=1, seq, hidden]. q_norm / k_norm on Qwen3
        # are reshaped to per-head before the norm, so flatten head dim back.
        if t.dim() == 4:
            # [b, n_heads, seq, head_dim] → [b, seq, n_heads * head_dim]
            t = t.permute(0, 2, 1, 3).contiguous().view(t.shape[0],
                                                       t.shape[2], -1)
        print(f"\n[{label}] shape={list(t.shape)}")
        for p in positions:
            if p < t.shape[1]:
                print(f"    [t={p}] first4 = {fmt(t[0, p])}")

    print("\n========================================")
    print(" Layer 0 PRE-norm projection outputs")
    print(" (these are what nntrainer's wq / wk / wv FC produces)")
    print("========================================")
    dump("q_proj_out (PRE-rope, PRE-norm)", store["q_proj_out"])
    dump("k_proj_out (PRE-rope, PRE-norm)", store["k_proj_out"])
    dump("v_proj_out (no rope, no norm)",   store["v_proj_out"])

    if "q_norm_out" in store:
        print("\n========================================")
        print(" Layer 0 q_norm / k_norm outputs")
        print(" (this is what nntrainer's q_norm / k_norm produces, and the")
        print("  PRE-rope tensor that the C++ probes call 'query_step' /")
        print("  'key_step')")
        print("========================================")
        dump("q_norm_out (PRE-rope, POST-norm)", store["q_norm_out"])
    if "k_norm_out" in store:
        dump("k_norm_out (PRE-rope, POST-norm)", store["k_norm_out"])

    print("\n========================================")
    print(" Layer 0 self_attn output (input to o_proj)")
    print(" (POST-attention POST-softmax mixed-V — compare with nntrainer's")
    print("  attn_out[t=0/1/last] probe lines)")
    print("========================================")
    dump("o_proj_in (attention output)", store["o_proj_in"])


if __name__ == "__main__":
    main()
