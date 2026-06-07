# Compare nntrainer runDDTreeDump draft_logits vs Python self-draft (same
# semantics: prefill -> root=argmax -> self-draft 15 steps from prompt+root).
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/workspace/qwen3run/hf/Qwen3-0.6B"
NCFG = "/workspace/nntrainer/Applications/CausalLM/res/qwen3/qwen3-0.6b/nntr_config.json"
NNTR_BIN = "/workspace/qwen3run/trace/nntr_draft.f32"
HORIZON = 15

prompt = json.load(open(NCFG))["sample_input"]
meta = {}
for tok_ in open(NNTR_BIN + ".meta").read().split():
    k, v = tok_.split("=")
    meta[k] = int(v)
vocab = meta["vocab"]
nntr = np.fromfile(NNTR_BIN, dtype=np.float32).reshape(HORIZON, vocab)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
start = ids.shape[1]
with torch.no_grad():
    root = int(model(ids).logits[0, -1].argmax())
    rows = []
    cur = torch.cat([ids, torch.tensor([[root]])], dim=1)
    for d in range(HORIZON):
        lg = model(cur).logits[0, -1, :].float()
        rows.append(lg.numpy().astype(np.float32))
        cur = torch.cat([cur, torch.tensor([[int(lg.argmax())]])], dim=1)
py = np.stack(rows, 0)

print(f"meta(nntr): root={meta['root']} start={meta['start']} vocab={vocab}")
print(f"py        : root={root} start={start}")
print(f"root match: {root==meta['root']}   start match: {start==meta['start']}")
print(f"draft_logits shape nntr={nntr.shape} py={py.shape}")
maxabs = float(np.max(np.abs(nntr - py)))
# per-row argmax agreement (the token the draft would pick at each position)
am_nntr = nntr.argmax(1); am_py = py.argmax(1)
am_match = int((am_nntr == am_py).sum())
print(f"max |Δ draft_logits| = {maxabs:.4e}")
print(f"per-position argmax match = {am_match}/{HORIZON}")
print(f"nntr argmax row tokens: {am_nntr.tolist()}")
print(f"py   argmax row tokens: {am_py.tolist()}")
