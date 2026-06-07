# Side-by-side demonstration: (A) DDTree decode is lossless (== greedy), and
# (B) nntrainer C++ ddtree core == Python ddtree.py (actual values).
import json

print("="*70)
print("(A) Is DDTree decoding WRONG/different? -> No. DDTree is LOSSLESS.")
print("="*70)
hf = None
for ln in open("/tmp/hfref.log"):
    if ln.startswith("GEN_TOKEN_IDS:"):
        hf = json.loads(ln.split(":",1)[1])
e2e = json.load(open("/workspace/qwen3run/e2e/py_e2e.json"))["gen_ids"]
n = min(len(hf), len(e2e))
match = sum(1 for i in range(n) if hf[i] == e2e[i])
print(f"  plain greedy tokens (HF)      : {len(hf)}")
print(f"  Python DDTree e2e tokens      : {len(e2e)}  (just generated more, not different)")
print(f"  overlap match (first {n})      : {match}/{n}")
print(f"  first 12 ids  greedy : {hf[:12]}")
print(f"  first 12 ids  DDTree : {e2e[:12]}")
print(f"  => DDTree reproduces greedy EXACTLY on the overlap: {match==n}")

print()
print("="*70)
print("(B) Did nntrainer implement ddtree correctly vs Python? actual values")
print("="*70)
cpp = json.load(open("/workspace/qwen3run/trace/cpp_tree.json"))
py  = json.load(open("/workspace/qwen3run/trace/py_tree.json"))
print(f"  current_length  C++={cpp['current_length']}  Python={py['current_length']}")
print(f"  node_count      C++={cpp['node_count']}  Python={py['node_count']}")
print("  idx | node_token_id (C++ / py) | depth (C++ / py) | parent (C++/py)")
for i in range(12):
    print(f"   {i:2d} |   {cpp['node_token_ids'][i]:6d} / {py['node_token_ids'][i]:6d}   |"
          f"   {cpp['node_depths'][i]:2d} / {py['node_depths'][i]:2d}     |"
          f"  {cpp['parents'][i+1]:2d} / {py['parents'][i+1]:2d}")
fields = ["node_token_ids","node_depths","parents","verify_input_ids",
          "verify_position_ids","visibility","mask_visible","accepted_indices","next_token"]
allok = all(cpp[f]==py[f] for f in fields)
print(f"  ALL FIELDS identical (C++ == Python): {allok}")
print(f"  accepted path  C++={cpp['accepted_indices']}")
print(f"                 py ={py['accepted_indices']}   next_token C++={cpp['next_token']} py={py['next_token']}")
