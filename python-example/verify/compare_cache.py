# Compare nntrainer's layer-0 KV cache (after block1) vs HF's, per position,
# to locate the corrupted cache row.
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL = "/workspace/qwen3run/hf/Qwen3-0.6B"
cfg = json.load(open(MODEL + "/config.json"))
H = cfg["num_key_value_heads"]
hd = cfg["head_dim"]
W = H * hd
print("num_kv_heads", H, "head_dim", hd, "kv_width", W)

prompt = "<|im_start|>user\nGive me a short introduction to large language model.<|im_end|>\n<|im_start|>assistant\n"
tok = AutoTokenizer.from_pretrained(MODEL)
ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
P = ids.shape[1]
gen = json.load(open("/workspace/qwen3run/trace/nntr_e2e.tokens.json"))["gen_ids"]

kc = np.fromfile("/workspace/qwen3run/trace/nntr_e2e.kcache0.f32", dtype=np.float32)
L = kc.size // W
kc = kc.reshape(L, W)
ncommit = L - P
print("cache positions L=", L, "prompt P=", P, "committed in cache=", ncommit,
      "gen_ids len=", len(gen))

full = torch.cat([ids, torch.tensor([gen[:ncommit]])], dim=1)  # [1, L]
pkv = DynamicCache()
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
with torch.no_grad():
    m(full, use_cache=True, past_key_values=pkv)
# layer 0 key: [1, H, L, hd]
hf_k = pkv.layers[0].keys[0].float().numpy()        # [H, L, hd]
hf_k_pos = np.transpose(hf_k, (1, 0, 2)).reshape(L, W)  # [L, H*hd]

diffs = np.max(np.abs(kc - hf_k_pos), axis=1)
print("per-position max|ΔK| (first bad highlighted):")
for p in range(L):
    flag = "  <-- BIG" if diffs[p] > 0.5 else ""
    if p < P + 2 or diffs[p] > 0.5 or p >= L - 4:
        print(f"  pos {p:2d}: max|dK|={diffs[p]:.4f}{flag}")
bad = [p for p in range(L) if diffs[p] > 0.5]
print("positions with big K diff:", bad)
