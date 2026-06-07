# Step 5: field-by-field exact comparison of py_tree.json vs cpp_tree.json.
import json, sys
T = "/workspace/qwen3run/trace"
a = json.load(open(f"{T}/py_tree.json"))
b = json.load(open(f"{T}/cpp_tree.json"))

fields = ["node_count", "current_length", "node_token_ids", "node_depths",
          "parents", "verify_input_ids", "verify_position_ids",
          "visibility", "mask_visible", "posterior",
          "accepted_indices", "next_token"]

all_ok = True
for k in fields:
    av, bv = a.get(k), b.get(k)
    ok = (av == bv)
    all_ok &= ok
    if isinstance(av, list):
        # summarize large grids
        def shape(x):
            if isinstance(x, list) and x and isinstance(x[0], list):
                return f"{len(x)}x{len(x[0])}"
            return f"len={len(x)}"
        extra = ""
        if not ok:
            # find first diff
            extra = " FIRST-DIFF"
        print(f"[{'OK' if ok else 'DIFF'}] {k:20s} py:{shape(av):>10s} cpp:{shape(bv):>10s}{extra}")
    else:
        print(f"[{'OK' if ok else 'DIFF'}] {k:20s} py={av} cpp={bv}")

print()
print("ALL FIELDS IDENTICAL:", all_ok)
print(f"tree size: current_length={a['current_length']} (node_count={a['node_count']}) "
      f"=> {'32-node tree CONFIRMED' if a['current_length']==32 else 'NOT 32'}")
sys.exit(0 if all_ok else 1)
