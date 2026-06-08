# Render the per-block DDTree dump (NNTR_DDTREE_DUMP -> *.blocks.json) as ASCII.
#
# Two formats are handled automatically:
#   * NEW dump (runDDTreeDump with parents/node_tokens/posterior): draws the exact
#     32-node tree (real parent->child edges), marks the accepted root->leaf path,
#     and shows why a branch was rejected (parent posterior != child token).
#   * OLD dump (node_depths only): the per-node tokens/parents are absent, so it
#     falls back to a depth-layer view (how many nodes sit at each depth, with the
#     accepted spine marked) -- the most that depths alone allow.
#
# Token ids are decoded to text when a tokenizer is given (--model <hf_dir>);
# otherwise raw ids are shown. transformers is optional (only for decoding).
#
# Usage:
#   python render_blocks.py <path>.blocks.json [--model <hf_dir>] [--block N]
import argparse
import json
import sys


def make_decoder(model_dir):
  """Return id->text fn. Falls back to str(id) when no tokenizer is available."""
  if not model_dir:
    return lambda t: str(t)
  try:
    from transformers import AutoTokenizer
  except Exception as e:  # transformers not installed -> ids only
    print(f"[warn] transformers unavailable ({e}); showing raw token ids",
          file=sys.stderr)
    return lambda t: str(t)
  tok = AutoTokenizer.from_pretrained(model_dir)
  return lambda t: f"{t}({tok.decode([t]).strip()!r})"


def depth_of(node_depths, i):
  """Depth of tree node i. node 0 is the root (depth 0); node_depths is for 1..N."""
  return 0 if i == 0 else node_depths[i - 1]


def render_exact(b, dec):
  """Draw the exact tree from parents/node_tokens/posterior (NEW dump format)."""
  parents = b["parents"]
  toks = b["node_tokens"]
  post = b["posterior"]
  nd = b["node_depths"]
  cl = len(toks)
  accepted = set(b["acc"])

  # parent -> sorted children (tree node order is best-first, so sort by index).
  children = {i: [] for i in range(cl)}
  for i in range(1, cl):
    children[parents[i]].append(i)

  def line(i):
    d = depth_of(nd, i)
    mark = "▸ " if i in accepted else "  "  # ▸ on the accepted path
    s = f"{mark}#{i:<2d} d{d:<2d} {dec(toks[i])}"
    # Posterior = what the model actually wanted at this node. For an accepted
    # node it tells you which child gets taken next (or, at the leaf, the bonus).
    s += f"   [post={dec(post[i])}]"
    if i in accepted and i != 0:
      pass
    elif i != 0:
      # rejected node: it lost because its parent wanted a different token.
      p = parents[i]
      if p in accepted and post[p] != toks[i]:
        s += "  (rejected: parent wanted post above)"
    return s

  def walk(i, prefix="", last=True):
    branch = "" if i == 0 else ("└─ " if last else "├─ ")
    print(prefix + branch + line(i))
    kids = children[i]
    child_prefix = prefix + ("" if i == 0 else ("   " if last else "│  "))
    for k, c in enumerate(kids):
      walk(c, child_prefix, k == len(kids) - 1)

  walk(0)


def render_layers(b, dec):
  """Depth-layer fallback when only node_depths is present (OLD dump format)."""
  nd = b["node_depths"]
  acc = b["acc"]
  # accepted depth for each accepted node (root is depth 0).
  acc_depths = {depth_of(nd, i) for i in acc}
  maxd = max(nd) if nd else 0
  # count nodes per depth (root is the single depth-0 node).
  by_depth = {0: 1}
  for d in nd:
    by_depth[d] = by_depth.get(d, 0) + 1
  print("  (old dump: depths only -> tree shape, not per-node tokens)")
  for d in range(0, maxd + 1):
    n = by_depth.get(d, 0)
    spine = "▸" if d in acc_depths else " "  # ▸ marks the accepted spine
    dots = "▸" + ("●" * (n - 1)) if d in acc_depths else "●" * n
    print(f"  {spine} d{d:<2d} {dots}  ({n} node{'s' if n != 1 else ''})")


def render_block(b, dec, idx):
  alen = b["alen"]
  cl = len(b.get("node_tokens", [])) or (1 + len(b["node_depths"]))
  print(f"\n=== block {idx}  pos={b['pos']}  root={dec(b['root'])}  "
        f"nodes={cl}  accepted={alen} (depth {alen - 1})  "
        f"bonus next={dec(b['next'])} ===")
  if "parents" in b and "node_tokens" in b:
    render_exact(b, dec)
  else:
    render_layers(b, dec)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("blocks_json", help="path to *.blocks.json")
  ap.add_argument("--model", default=None, help="HF tokenizer dir for decoding")
  ap.add_argument("--block", type=int, default=None, help="render only block N")
  args = ap.parse_args()

  blocks = json.load(open(args.blocks_json))
  dec = make_decoder(args.model)

  total_alen = sum(b["alen"] for b in blocks)
  print(f"{len(blocks)} blocks, {total_alen} tokens committed "
        f"(mean accept length {total_alen / max(1, len(blocks)):.2f})")
  for i, b in enumerate(blocks):
    if args.block is not None and i != args.block:
      continue
    render_block(b, dec, i)


if __name__ == "__main__":
  main()
