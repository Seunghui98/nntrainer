# Capture MANY draft-logit cases from qwen3-0.6b: several prompts x several
# 15-step windows. Each window -> one [15, vocab] fp32 logits file. Cases then
# pair each logits window with several tree budgets. Writes cases.json.
import json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/workspace/qwen3run/hf/Qwen3-0.6B"
OUT = "/workspace/qwen3run/cases"
os.makedirs(OUT, exist_ok=True)

DRAFT_HORIZON = 15
WINDOWS_PER_PROMPT = 3
BUDGETS = [7, 15, 31, 63]   # tree sizes 8, 16, 32, 64

prompts = [
    "<|im_start|>user\nGive me a short introduction to large language model.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nExplain quantum entanglement in two sentences.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWrite a haiku about the sea.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nList three benefits of unit testing.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\n1 + 1 = ? Answer with reasoning.<|im_end|>\n<|im_start|>assistant\n",
]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
model.eval()

cases = []
case_id = 0
for pi, prompt in enumerate(prompts):
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    n_steps = DRAFT_HORIZON * WINDOWS_PER_PROMPT
    rows = []
    with torch.no_grad():
        cur = ids
        for _ in range(n_steps):
            logits = model(cur).logits[0, -1, :].float()
            rows.append(logits.numpy().astype(np.float32))
            cur = torch.cat([cur, torch.tensor([[int(torch.argmax(logits))]])], dim=1)
    allrows = np.stack(rows, 0).astype(np.float32)   # [n_steps, vocab]
    vocab = allrows.shape[1]
    for w in range(WINDOWS_PER_PROMPT):
        win = np.ascontiguousarray(allrows[w*DRAFT_HORIZON:(w+1)*DRAFT_HORIZON])
        lf = f"{OUT}/logits_p{pi}_w{w}.f32"
        win.tofile(lf)
        np.save(lf.replace(".f32", ".npy"), win)
        root = int(np.argmax(win[0]))
        start = 18 + w * 16 + pi      # varied
        past = start
        for b in BUDGETS:
            cases.append({"id": case_id, "logits_npy": lf.replace(".f32", ".npy"),
                          "logits_f32": lf, "depth_limit": DRAFT_HORIZON,
                          "vocab": int(vocab), "budget": b, "root_token_id": root,
                          "start": start, "past_length": past,
                          "prompt_idx": pi, "window": w})
            case_id += 1

json.dump(cases, open(f"{OUT}/cases.json", "w"), indent=2)
print(f"wrote {len(cases)} cases ({len(prompts)} prompts x {WINDOWS_PER_PROMPT} windows x {len(BUDGETS)} budgets), vocab={vocab}")
