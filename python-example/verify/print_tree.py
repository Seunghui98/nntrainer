# Render block-0's 32-node DDTree (parent/child + token + depth) as ASCII,
# using the REAL ddtree.py build function (ddtree_ref) on qwen3 self-draft.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import ddtree_ref as R

MODEL = "/workspace/qwen3run/hf/Qwen3-0.6B"
HORIZON = 15
BUDGET = 31
prompt = "<|im_start|>user\nGive me a short introduction to large language model.<|im_end|>\n<|im_start|>assistant\n"

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids

with torch.no_grad():
    root = int(m(ids).logits[0, -1].argmax())     # block-0 root
    rows, cur = [], torch.cat([ids, torch.tensor([[root]])], 1)
    for _ in range(HORIZON):
        lg = m(cur).logits[0, -1, :].float()
        rows.append(lg.numpy()); cur = torch.cat([cur, torch.tensor([[int(lg.argmax())]])], 1)
draft = torch.tensor(rows)

nti, nd, parents, child_maps, vis, _ = R.build_ddtree_tree(draft, BUDGET)
nti = nti.tolist(); nd = nd.tolist()
cl = 1 + len(nti)
print(f"tree nodes (currentLength) = {cl}  (budget {BUDGET} + root)")
print(f"draft horizon (block_size-1) = {HORIZON}")

# token per tree-index: 0=root, k>=1 -> nti[k-1]
def tokstr(i):
    t = root if i == 0 else nti[i-1]
    return f"{t}({tok.decode([t]).strip()!r})"

# children: child_maps[parent] = {token: child_index}
children = {i: sorted(child_maps[i].values()) for i in range(cl)}
def draw(i, prefix=""):
    d = 0 if i == 0 else nd[i-1]
    print(f"{prefix}#{i} d{d} {tokstr(i)}")
    ch = children.get(i, [])
    for j, c in enumerate(ch):
        draw(c, prefix + "   ")

print("\n=== block-0 32-node tree (root -> children) ===")
draw(0)
print("\n(branches = multiple children at a node = tree, not a chain)")
