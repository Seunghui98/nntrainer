# Step 3: run the REAL ddtree.py tree functions (via ddtree_ref) on the captured
# draft logits and dump tree/compile/follow results to JSON for C++ comparison.
import json
import numpy as np
import torch
import ddtree_ref as R

T = "/workspace/qwen3run/trace"
meta = json.load(open(f"{T}/meta.json"))
budget = meta["budget"]; start = meta["start"]; past = meta["past_length"]
root = meta["root_token_id"]

draft = torch.from_numpy(np.load(f"{T}/draft_logits.npy"))  # [15, vocab] fp32

# --- build ---
node_token_ids, node_depths, parents, child_maps, visibility, _ = R.build_ddtree_tree(draft, budget)
node_count = int(node_token_ids.numel())
current_length = 1 + node_count

# --- compile (allocate buffers like ddtree_generate does) ---
max_nodes = 1 + budget
dtype = torch.float32
vib = torch.empty((1, max_nodes), dtype=torch.long)
vpb = torch.empty((1, max_nodes), dtype=torch.long)
amb = torch.zeros((1, 1, max_nodes, past + max_nodes), dtype=dtype)
tvb = torch.empty((max_nodes, max_nodes), dtype=torch.bool)
vii, vpi, amask, p_start, c_len = R.compile_ddtree_tree(
    torch.tensor(root), start, node_token_ids, node_depths, visibility,
    past, dtype, torch.device("cpu"), vib, vpb, amb, tvb, 0, 0)

# deterministic posterior: leftmost-child token of each node (else -1) -> walks a
# concrete accepted path; identical array is fed to the C++ side.
posterior = []
for i in range(current_length):
    cm = child_maps[i]
    posterior.append(min(cm.keys()) if cm else -1)
posterior_t = torch.tensor([posterior], dtype=torch.long)
accepted, next_token = R.follow_verified_tree(child_maps, posterior_t)

mask_np = amask[0, 0, :c_len, :past + c_len].numpy()
fillmin = float(torch.finfo(dtype).min)
out = {
    "node_count": node_count,
    "current_length": current_length,
    "node_token_ids": node_token_ids.tolist(),
    "node_depths": node_depths.tolist(),
    "parents": [int(x) for x in parents],
    "visibility": visibility.to(torch.int).tolist(),
    "verify_input_ids": vii[0].tolist(),
    "verify_position_ids": vpi[0].tolist(),
    "past_length": int(p_start),
    "mask_visible": (mask_np == 0.0).astype(int).tolist(),
    "fill_min": fillmin,
    "posterior": posterior,
    "accepted_indices": [int(x) for x in accepted],
    "next_token": int(next_token),
}
json.dump(out, open(f"{T}/py_tree.json", "w"))
print("node_count =", node_count, " current_length =", current_length)
print("=> 32-node tree?" , current_length == 32, "(budget", budget, ")")
print("accepted path len =", len(accepted), "next_token =", next_token)
