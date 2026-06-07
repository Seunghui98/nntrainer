# Stage A: Python reference end-to-end DDTree decode loop with SELF-DRAFT.
# Reuses the REAL ddtree.py functions (ddtree_ref) for tree build/compile/follow
# and KV-tail compaction. Draft = qwen3 itself (greedy horizon rollout) so both
# the C++ harness and this reference can replicate it exactly.
# Greedy + self-draft => output MUST equal plain greedy decoding (built-in check).
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import ddtree_ref as R

MODEL = "/workspace/qwen3run/hf/Qwen3-0.6B"
OUT = "/workspace/qwen3run/e2e"
import os; os.makedirs(OUT, exist_ok=True)

BLOCK = 16
HORIZON = BLOCK - 1      # 15
BUDGET = 31              # 32-node tree
MAX_NEW = 64
EOS = {151645, 151643}
MASK_TOKEN = 151643
PROMPT = "<|im_start|>user\nGive me a short introduction to large language model.<|im_end|>\n<|im_start|>assistant\n"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
model.eval()
dtype = torch.float32
dev = torch.device("cpu")

ids = tok(PROMPT, return_tensors="pt", add_special_tokens=False).input_ids
num_input = ids.shape[1]
max_length = num_input + MAX_NEW
max_nodes = 1 + BUDGET

output_ids = torch.full((1, max_length + max_nodes), MASK_TOKEN, dtype=torch.long)

@torch.no_grad()
def self_draft(context_ids):
    """qwen3 greedy rollout for HORIZON steps -> draft_logits [HORIZON, vocab]."""
    rows = []
    cur = context_ids.clone()
    for _ in range(HORIZON):
        lg = model(cur, use_cache=False).logits[0, -1, :].float()
        rows.append(lg)
        cur = torch.cat([cur, lg.argmax().view(1, 1)], dim=1)
    return torch.stack(rows, 0)

# ---- prefill: cache holds [0 .. num_input-1] ----
pkv = DynamicCache()
with torch.no_grad():
    out = model(ids, use_cache=True, past_key_values=pkv)
output_ids[:, :num_input] = ids
output_ids[:, num_input] = out.logits[:, -1, :].argmax(-1)

start = num_input
blocks = []
while start < max_length:
    root = output_ids[:, start:start + 1]                 # [1,1]
    draft_logits = self_draft(output_ids[:, :start + 1])  # [15, vocab]
    nti, nd, parents, child_maps, vis, _ = R.build_ddtree_tree(draft_logits, BUDGET)

    vib = torch.empty((1, max_nodes), dtype=torch.long)
    vpb = torch.empty((1, max_nodes), dtype=torch.long)
    amb = torch.zeros((1, 1, max_nodes, start + max_nodes), dtype=dtype)
    tvb = torch.empty((max_nodes, max_nodes), dtype=torch.bool)
    vii, vpi, amask, p_start, c_len = R.compile_ddtree_tree(
        root[0, 0], start, nti, nd, vis, start, dtype, dev, vib, vpb, amb, tvb, 0, 0)

    tmask = R.prepare_ddtree_attention_mask_for_target(model, amask, vpi, 0, 0)
    with torch.no_grad():
        vout = model(vii, position_ids=vpi, attention_mask=tmask,
                     past_key_values=pkv, use_cache=True)
    posterior = vout.logits.argmax(-1)                    # [1, c_len] greedy
    accepted, next_token = R.follow_verified_tree(child_maps, posterior)
    acc_t = torch.tensor(accepted, dtype=torch.long)
    accepted_tokens = vii.index_select(1, acc_t)

    R.compact_dynamic_cache(pkv, start, accepted, expected_current_length=c_len)

    output_ids[:, start:start + len(accepted)] = accepted_tokens
    output_ids[:, start + len(accepted)] = next_token

    blocks.append({
        "pos": int(start), "root": int(root[0, 0]),
        "current_length": int(c_len), "node_count": int(nti.numel()),
        "alen": len(accepted), "next": int(next_token),
        "draft_am": [int(x) for x in draft_logits.argmax(-1).tolist()],
        "acc": [int(x) for x in accepted],
        "node_depths": [int(x) for x in nd.tolist()],
    })
    start += len(accepted)
    newly = accepted_tokens[0, 1:].tolist() + [int(next_token)]
    if any(t in EOS for t in newly):
        break

gen = output_ids[0, num_input:start + 1].tolist()
gen = [t for t in gen if t != MASK_TOKEN]
# trim at first EOS
for i, t in enumerate(gen):
    if t in EOS:
        gen = gen[:i + 1]; break

json.dump({"gen_ids": gen, "blocks": blocks,
           "accept_lengths": [b["accept_len"] for b in blocks]},
          open(f"{OUT}/py_e2e.json", "w"))
print("gen tokens:", len(gen))
print("accept lengths per block:", [b["accept_len"] for b in blocks])
print("GEN_TEXT:", json.dumps(tok.decode(gen, skip_special_tokens=False)))
