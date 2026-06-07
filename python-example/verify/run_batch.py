# Batch tree-build parity: for EVERY case, run real ddtree.py funcs vs the C++
# harness on identical logits and compare all fields. Prints a table + aggregate.
import json, subprocess, sys
import numpy as np
import torch
import ddtree_ref as R

ROOT = "/workspace/qwen3run"
CASES = json.load(open(f"{ROOT}/cases/cases.json"))
HARNESS = f"{ROOT}/ddtree_parity"

FIELDS = ["node_count", "current_length", "node_token_ids", "node_depths",
          "parents", "verify_input_ids", "verify_position_ids",
          "visibility", "mask_visible", "posterior",
          "accepted_indices", "next_token"]

def py_case(c):
    draft = torch.from_numpy(np.load(c["logits_npy"]))
    budget = c["budget"]; start = c["start"]; past = c["past_length"]; root = c["root_token_id"]
    nti, nd, parents, child_maps, vis, _ = R.build_ddtree_tree(draft, budget)
    node_count = int(nti.numel()); cl = 1 + node_count
    max_nodes = 1 + budget; dtype = torch.float32
    vib = torch.empty((1, max_nodes), dtype=torch.long)
    vpb = torch.empty((1, max_nodes), dtype=torch.long)
    amb = torch.zeros((1, 1, max_nodes, past + max_nodes), dtype=dtype)
    tvb = torch.empty((max_nodes, max_nodes), dtype=torch.bool)
    vii, vpi, amask, p_start, c_len = R.compile_ddtree_tree(
        torch.tensor(root), start, nti, nd, vis, past, dtype,
        torch.device("cpu"), vib, vpb, amb, tvb, 0, 0)
    posterior = [ (min(child_maps[i]) if child_maps[i] else -1) for i in range(cl) ]
    acc, nxt = R.follow_verified_tree(child_maps, torch.tensor([posterior], dtype=torch.long))
    mask_np = amask[0, 0, :c_len, :past + c_len].numpy()
    return {
        "node_count": node_count, "current_length": cl,
        "node_token_ids": nti.tolist(), "node_depths": nd.tolist(),
        "parents": [int(x) for x in parents],
        "verify_input_ids": vii[0].tolist(), "verify_position_ids": vpi[0].tolist(),
        "visibility": vis.to(torch.int).tolist(),
        "mask_visible": (mask_np == 0.0).astype(int).tolist(),
        "posterior": posterior, "accepted_indices": [int(x) for x in acc],
        "next_token": int(nxt),
    }

def cpp_case(c):
    out = f"/tmp/cpp_case_{c['id']}.json"
    subprocess.run([HARNESS, str(c["depth_limit"]), str(c["vocab"]), str(c["budget"]),
                    str(c["root_token_id"]), str(c["start"]), str(c["past_length"]),
                    c["logits_f32"], out], check=True)
    return json.load(open(out))

n_ok = 0; diffs = []
print(f"{'id':>3} {'prompt':>6} {'win':>3} {'budget':>6} {'cur_len':>7}  result")
for c in CASES:
    a = py_case(c); b = cpp_case(c)
    bad = [k for k in FIELDS if a.get(k) != b.get(k)]
    ok = not bad
    n_ok += ok
    if not ok:
        diffs.append((c["id"], bad))
    print(f"{c['id']:>3} {c['prompt_idx']:>6} {c['window']:>3} {c['budget']:>6} "
          f"{a['current_length']:>7}  {'OK' if ok else 'DIFF '+','.join(bad)}")

print()
print(f"PARITY: {n_ok}/{len(CASES)} cases ALL-FIELDS IDENTICAL (C++ == ddtree.py)")
# show that tree size == budget+1 across cases
sizes = sorted(set((c['budget'], py_case(c)['current_length']) for c in CASES[:0]))  # skip recompute
if diffs:
    print("DIFFS:", diffs)
sys.exit(0 if n_ok == len(CASES) else 1)
