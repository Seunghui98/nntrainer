# Step 2: capture realistic draft logits [draft_horizon, vocab] fp32 from
# qwen3-0.6b (greedy teacher-forcing 15 steps from the chat prompt). Save as
# .npy and a raw row-major fp32 .bin for the C++ harness, plus meta.json.
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/workspace/qwen3run/hf/Qwen3-0.6B"
OUTDIR = "/workspace/qwen3run/trace"
import os
os.makedirs(OUTDIR, exist_ok=True)

BLOCK_SIZE = 16
DRAFT_HORIZON = BLOCK_SIZE - 1   # 15
TREE_BUDGET = 31                 # "32 tree" == budget 31

prompt = "<|im_start|>user\nGive me a short introduction to large language model.<|im_end|>\n<|im_start|>assistant\n"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
model.eval()

ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
rows = []
with torch.no_grad():
    cur = ids
    for step in range(DRAFT_HORIZON):
        out = model(cur)
        logits = out.logits[0, -1, :].float()   # [vocab]
        rows.append(logits.numpy().astype(np.float32))
        nxt = int(torch.argmax(logits).item())
        cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)

draft = np.ascontiguousarray(np.stack(rows, axis=0).astype(np.float32))  # [15, vocab]
vocab = draft.shape[1]
np.save(f"{OUTDIR}/draft_logits.npy", draft)
draft.tofile(f"{OUTDIR}/draft_logits.f32")   # row-major fp32
meta = {"depth_limit": DRAFT_HORIZON, "vocab": int(vocab),
        "budget": TREE_BUDGET, "block_size": BLOCK_SIZE,
        "root_token_id": int(ids[0, -1].item()), "start": int(ids.shape[1]),
        "past_length": int(ids.shape[1])}
json.dump(meta, open(f"{OUTDIR}/meta.json", "w"), indent=2)
print("saved draft_logits", draft.shape, "vocab", vocab)
print("meta", meta)
