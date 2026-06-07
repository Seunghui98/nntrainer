# Compare per-node cumulative logw (tree's log-prob "value") C++ vs Python over
# ALL cases. Python uses torch.topk + torch.logsumexp (fp32), exactly like
# ddtree.py build_ddtree_tree; C++ uses ddtree.cpp's recipe (ddtree_logw).
import json, subprocess
import numpy as np
import torch
import ddtree_ref as R

ROOT = "/workspace/qwen3run"
CASES = json.load(open(f"{ROOT}/cases/cases.json"))
LOGW = f"{ROOT}/ddtree_logw"

def py_node_logw(c):
    draft = torch.from_numpy(np.load(c["logits_npy"])).float()
    budget = c["budget"]
    topk = min(budget, draft.shape[-1])
    top_logits, top_ids = torch.topk(draft, k=topk, dim=-1)
    log_z = torch.logsumexp(draft, dim=-1, keepdim=True)
    tlp = (top_logits - log_z).to(torch.float32).numpy()
    tti = top_ids.numpy()
    nti, nd, parents, child_maps, vis, _ = R.build_ddtree_tree(draft, budget)
    nti = nti.tolist(); nd = nd.tolist()
    cl = 1 + len(nti)
    logw = [0.0] * cl
    for idx in range(1, cl):
        k = idx - 1
        dep = nd[k]; tok = nti[k]
        row = tti[dep - 1]
        rank = int(np.where(row == tok)[0][0])
        logw[idx] = logw[parents[idx]] + float(tlp[dep - 1][rank])
    return logw

worst = 0.0
n_exact = 0
worst_case = None
print(f"{'id':>3} {'budget':>6} {'cl':>4}  max|Δlogw|")
for c in CASES:
    pw = py_node_logw(c)
    out = f"/tmp/logw_{c['id']}.json"
    subprocess.run([LOGW, str(c["depth_limit"]), str(c["vocab"]), str(c["budget"]),
                    c["logits_f32"], out], check=True, stdout=subprocess.DEVNULL)
    cw = json.load(open(out))["node_logw"]
    d = max((abs(a - b) for a, b in zip(pw, cw)), default=0.0)
    if d == 0.0:
        n_exact += 1
    if d > worst:
        worst, worst_case = d, c["id"]
    if c["id"] % 4 == 2:  # print the budget=31 rows
        print(f"{c['id']:>3} {c['budget']:>6} {len(pw):>4}  {d:.3e}")

print()
print(f"node-logw parity over {len(CASES)} cases:")
print(f"  exact-equal cases: {n_exact}/{len(CASES)}")
print(f"  worst max|Δlogw| = {worst:.3e} (case {worst_case})")
print(f"  fp32 epsilon ~ 1.2e-7; values are log-probs O(1..40)")
