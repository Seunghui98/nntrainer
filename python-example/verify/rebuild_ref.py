# Rebuild ddtree_ref.py fully: __future__ annotations first (defers type-hint
# evaluation), imports, stubs, then 6 verbatim functions from ddtree.py.
import io
DD = "/workspace/nntrainer/python-example/ddtree.py"
REF = "/workspace/qwen3run/ddtree_ref.py"
src = io.open(DD, encoding="utf-8").read().splitlines(keepends=True)

def extract(name):
    start = next(i for i, ln in enumerate(src) if ln.startswith(f"def {name}("))
    end = len(src)
    for j in range(start + 1, len(src)):
        if src[j].startswith("def ") or src[j].startswith("class ") or src[j].startswith("@"):
            end = j; break
    return "".join(src[start:end]).rstrip() + "\n"

header = '''from __future__ import annotations
# AUTO-EXTRACTED verbatim from python-example/ddtree.py (tree + mask + compact
# functions) with trivial stubs. __future__ annotations defers type-hint eval.
import heapq
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, DynamicCache

DDTREE_TREE_BUILD_STAGE_ORDER = ("tree_build_copy", "tree_build_heap", "tree_build_visibility")
_CPP_COMPACT_ENABLED = False

def cuda_time():
    return 0.0

def empty_stage_times(order):
    return {k: 0.0 for k in order}

def load_cpp_compact_module():
    return None

def get_model_text_config(target):
    cfg = target.config
    return getattr(cfg, "text_config", cfg)


'''

fns = ["build_ddtree_tree", "compile_ddtree_tree", "follow_verified_tree",
       "prepare_ddtree_attention_mask_for_target", "_compact_appended_window",
       "compact_dynamic_cache"]
body = "\n\n".join(extract(f) for f in fns) + "\n"
io.open(REF, "w", encoding="utf-8").write(header + body)

import sys; sys.path.insert(0, "/workspace/qwen3run")
import importlib, ddtree_ref; importlib.reload(ddtree_ref)
for f in fns:
    assert hasattr(ddtree_ref, f), f
print("ddtree_ref rebuilt OK with all 6 functions")
