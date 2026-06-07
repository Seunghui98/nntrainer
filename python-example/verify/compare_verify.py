# Phase C: compare nntrainer verify-forward node logits vs HF target forward on
# the SAME tree (same verify_input_ids / positions / additive mask). Isolates the
# Task10 mask + per-node RoPE-position integration in the real nntrainer model.
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import ddtree_ref as R

MODEL = "/workspace/qwen3run/hf/Qwen3-0.6B"
NCFG = "/workspace/nntrainer/Applications/CausalLM/res/qwen3/qwen3-0.6b/nntr_config.json"
DP = "/workspace/qwen3run/trace/nntr_v.f32"
HORIZON = 15

prompt = json.load(open(NCFG))["sample_input"]
j = json.load(open(DP + ".json"))
cl = j["current_length"]; start = j["start"]; vocab = j["vocab"]
nntr_nodes = np.fromfile(DP + ".nodes.f32", dtype=np.float32).reshape(cl, vocab)
draft = torch.from_numpy(np.fromfile(DP, dtype=np.float32).reshape(HORIZON, vocab))

# Rebuild the SAME tree (Phase B proved nntr tree == ddtree.py on nntr draft).
nti, nd, parents, child_maps, vis, _ = R.build_ddtree_tree(draft, 31)
root = j["root"]; past = start; mn = 32
vii, vpi, amask, ps, c_len = R.compile_ddtree_tree(
    torch.tensor(root), start, nti, nd, vis, past, torch.float32,
    torch.device("cpu"), torch.empty((1, mn), dtype=torch.long),
    torch.empty((1, mn), dtype=torch.long),
    torch.zeros((1, 1, mn, past + mn), dtype=torch.float32),
    torch.empty((mn, mn), dtype=torch.bool), 0, 0)
assert vii[0].tolist() == j["verify_input_ids"], "tree verify_ids mismatch nntr vs py"
assert vpi[0].tolist() == j["verify_position_ids"], "tree positions mismatch"
print("tree (verify_ids/positions) nntr == py: True")

# HF: prefill prompt -> cache[0,start); then verify forward with tree mask+positions.
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
assert ids.shape[1] == start, (ids.shape[1], start)
pkv = DynamicCache()
with torch.no_grad():
    model(ids, use_cache=True, past_key_values=pkv)         # cache holds [0,start)
    vout = model(vii, position_ids=vpi, attention_mask=amask,
                 past_key_values=pkv, use_cache=True)
py_nodes = vout.logits[0].float().numpy()                    # [cl, vocab]

maxabs = float(np.max(np.abs(py_nodes - nntr_nodes)))
am_py = py_nodes.argmax(1); am_nntr = nntr_nodes.argmax(1)
am_match = int((am_py == am_nntr).sum())
print(f"verify node_logits shape nntr={nntr_nodes.shape} py={py_nodes.shape}")
print(f"max |Δ node_logits|        = {maxabs:.4e}")
print(f"per-node argmax(posterior) match = {am_match}/{cl}")
print(f"nntr posterior(json)[:12] = {j['posterior'][:12]}")
print(f"py   posterior     [:12]  = {am_py[:12].tolist()}")
print(f"=> nntrainer verify forward matches HF on the tree (decision-level): {am_match==cl}")
