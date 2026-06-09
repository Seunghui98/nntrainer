# DDTree → gemma4 Runtime PR — Handoff (new session)

> **Audience:** a fresh Claude session with **zero context**. Read this top-to-bottom
> before doing anything. Also read the auto-memory index at
> `~/.claude/projects/-home-shsh1004-nntrainer/memory/MEMORY.md` and the two
> linked memories (`ddtree-project-state`, `sd-quickai-gauss4-integration`).

---

## 1. Goal

Create a **new branch + PR** that bundles everything needed to **run DDTree
(tree speculative decoding) on gemma4-E2B in nntrainer's CPU CausalLM runtime** —
i.e. the *application/runtime layer*, **separate from the already-submitted
pure-core PR**.

- **Already upstreamed (do NOT touch):** the pure core lives on branch
  `ddtree-core` → PR **nntrainer/nntrainer#3972** (`nntrainer/ddtree/*`, unit
  tests, golden vectors, packaging). This is "goal A".
- **This task ("goal B"):** the runtime + gemma4 enablement on top of the core.

**This PR depends on the core.** Strategy: **branch FROM `ddtree-core`** (not
`upstream/main`), so the core headers are present. The PR is *stacked on #3972* —
say so in the PR description; once #3972 merges, rebase onto `main`.

---

## 2. Repo facts

- Working dir: `/home/shsh1004/nntrainer`
- Remotes: `origin` = `https://github.com/Seunghui98/nntrainer` (user fork, push here),
  `upstream` = `https://github.com/nntrainer/nntrainer`.
- All the runtime work is **already committed on branch `add/DDTree`** but
  **interleaved with debug/doc commits** — do NOT cherry-pick blindly. Use the
  **rebuild approach** (§5).
- Push needs a one-off GitHub PAT — **ask the user for it each time**; use it only
  in a one-off URL, mask it in output (`sed "s/$TOKEN/<TOKEN>/g"`), never persist
  to git config. Remind the user to revoke it after.

---

## 3. What goes IN this PR (exact files, grouped into logical commits)

All paths are versions **as they exist on `add/DDTree`**. Suggested commit split
(`[module] Imperative subject`, nntrainer convention). Refine if needed.

**Commit 1 — `[layer] Support multi-token incremental forwarding (step > 1)`**
Core layers asserted `to-from==1`; tree verify pushes N tokens in one forward.
- `nntrainer/layers/embedding.cpp`
- `nntrainer/layers/multiout_layer.cpp`
- `nntrainer/layers/operation_layer.h`

**Commit 2 — `[CausalLM] Add DDTree verify attention path to MHACore`**
Additive non-causal mask path, external-cache mask input slot, per-token RoPE
positions, full-vs-sliding mask selection (`selectVerifyMask`,
`setGlobalDDTreeVerify`).
- `Applications/CausalLM/layers/mha_core.cpp`
- `Applications/CausalLM/layers/mha_core.h`

**Commit 3 — `[CausalLM] Add KV-cache tail compaction for the DDTree path`**
`KVCacheManager::compactTail` calling `ddtree::compactTail` per layer
(UINT16/elemSize-aware).
- `Applications/CausalLM/kv_cache_manager.cpp`
- `Applications/CausalLM/kv_cache_manager.h`

**Commit 4 — `[CausalLM] Emit all-position verify logits in LMHead/TieWordEmbedding`**
- `Applications/CausalLM/layers/lm_head.cpp` / `.h`
- `Applications/CausalLM/layers/tie_word_embedding.cpp` / `.h`

**Commit 5 — `[CausalLM] Drive the DDTree speculative-decode loop`**
`runDDTreeDump` full speculative loop, SKIP_PREFILL handling, env-gated dumps
(`NNTR_DDTREE_DUMP/LOGITS/NODELOGITS`, `NNTR_GREEDY_IDS`).
- `Applications/CausalLM/models/causal_lm.cpp` / `.h`
- `Applications/CausalLM/meson.build`

**Commit 6 — `[CausalLM] Enable DDTree for gemma4-E2B`**
gemma4 sliding/full overrides + runnable resources + GGUF→safetensors converter.
- `Applications/CausalLM/models/gemma4/gemma4_causallm.h`
- `Applications/CausalLM/res/gemma4/gemma4-e2b/gguf_to_nntrainer.py`
- `Applications/CausalLM/res/gemma4/gemma4-e2b/nntr_config.json`
- `Applications/CausalLM/res/gemma4/gemma4-e2b/README.md`
- `Applications/CausalLM/res/qwen3/qwen3-0.6b/nntr_config.json`
- `Applications/CausalLM/res/qwen3/qwen3-0.6b/generation_config.json`

**Commit 7 — `[test] Add MHACore mask and KV-cache compaction unit tests`**
- `test/unittest/layers/unittest_mha_core_mask.cpp`
- `test/unittest/layers/unittest_kv_cache_manager.cpp`
- the corresponding entries in `test/unittest/meson.build` (the **mha/kv** lines;
  the `unittest_ddtree` line already belongs to the core PR — keep only the two
  new lines here).

> Decide with the user: are the **env-gated runtime dumps** (commit 5) wanted in
> upstream, or should the verify path be wired more cleanly (a real API, not
> `NNTR_*` env dumps)? The dumps were debugging scaffolding. Cleaning them into a
> proper interface may be expected by reviewers.

---

## 4. What to EXCLUDE (do NOT commit to this PR)

- **`Applications/meson.build`** — has a LOCAL DeepQ-subdir workaround (needs
  jsoncpp+libcurl, not installable here). It is intentionally uncommitted on
  `add/DDTree`. Never stage it.
- **`docs/superpowers/**`** — internal brainstorm/plan/spec/handoff (incl. this
  file). Not for upstream.
- **`python-example/**`** — the parity/verification suite + original Python
  reference. Out of scope here; if ever upstreamed, a separate trimmed PR.
- **`docs/ddtree-speculative-decoding.md`** — the Quick.AI/QNN guide (belongs to
  the QNN track, not the CPU-runtime PR).
- Anything already in `ddtree-core` (the `nntrainer/ddtree/*` core, golden, etc.).

### Host-local scrub (MUST check before committing)
- `res/gemma4/gemma4-e2b/nntr_config.json` and `res/qwen3/qwen3-0.6b/nntr_config.json`:
  scrub absolute host paths (weights, tokenizer, `sample_input`). gemma4-E2B's
  chat template uses `<|turn>`/`<turn|>` (NOT `<start_of_turn>`) — keep that, but
  no `/home/...` paths.
- `gguf_to_nntrainer.py`: remove any `/home/shsh1004/...` defaults; make paths
  args/relative.
- Grep the staged diff: `git diff --cached | grep -nE "/home/|shsh1004|littlesd|qwen3run"`.

---

## 5. Branch construction (rebuild, like the core PR was done)

The `add/DDTree` history is interleaved, so **rebuild a clean branch** rather than
cherry-pick:

```bash
# 0) stash the local DeepQ workaround so it can't leak
git stash push -m deepq Applications/meson.build

# 1) start from the core branch (this PR is stacked on #3972)
git checkout -B ddtree-gemma4 ddtree-core

# 2) soft-reset to the core tip so the runtime files appear as a clean slate,
#    then bring each commit's files in from add/DDTree and re-commit in groups.
#    (See §3 for the file→commit mapping. Use:)
git checkout add/DDTree -- <files for commit N>
git commit --author="SeungHui Lee <shsh1004.lee@samsung.com>" -m "[module] ..."
```

Per commit:
- **Author/identity:** `SeungHui Lee <shsh1004.lee@samsung.com>`, add a
  `Signed-off-by: SeungHui Lee <shsh1004.lee@samsung.com>` trailer.
- **NO `Co-Authored-By: Claude`** anywhere (verify: `git log ... --format='%b' | grep -ci co-authored` → 0).

> NB: `test/unittest/meson.build` and `nntrainer/meson.build` already carry core
> changes on `ddtree-core`. When you bring the runtime versions from `add/DDTree`,
> you only want the **runtime-specific additions** (mha/kv test entries, CausalLM
> wiring) layered on top — diff carefully so you don't revert/duplicate the core's
> lines. Inspect `git diff ddtree-core:test/unittest/meson.build add/DDTree:test/unittest/meson.build`.

---

## 6. Conventions & CI gotchas (learned the hard way on the core PR)

**New files need a doxygen file header** (the doxygen-tag CI enforces it):
```
// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @brief  <one line>
 * @file   <filename>
 * @date   <DD Mon 2026>
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
```
**doxygen-tag checker rules** (`.github/workflows/static.check.scripts/doxygen-tag.sh`):
- Every C/C++ file: `@file @brief @author @bug` at top.
- Every **struct/class** needs `@brief` (checked via ctags `--c-kinds=sc`).
- **No inline `/*arg=*/` comments** — the checker demands every `/*` open a `/**`
  block. Use trailing `//` or drop the name.
- Python files need `@package` OR `@brief` somewhere in the file.
- (Per-*function* `@brief` is effectively disabled by a bug in the script, so
  `.cpp` function defs are not checked — but struct/class IS.)
- Reproduce locally: `report_path=/dev/null bash .github/workflows/static.check.scripts/doxygen-tag.sh <filelist> 1`
  (local `ctags` may be emacs etags and can't run the struct check; install
  `universal-ctags` if you need it, or eyeball struct `@brief`s).

**clang-format v14** (CI uses clang-format-14): binary already present at
`/home/shsh1004/qwen3run/venv/bin/clang-format`. Run `-i` on every changed C++
file; verify 0 replacements via `--output-replacements-xml | grep -c '<replacement '`.

**Cross-arch / Windows-ARM64 test job**: if you add any **bit-exact** test (like a
golden), make sure float-derived decisions have a **large margin** (≫ fp32 ULP
~1.2e-7) and no *coincidental* exact ties — otherwise x86 vs ARM64 libm rounding
flips results and the ARM64 job fails. (This bit the core PR; fixed by widening
`budget_gt_vocab` logits.)

**Android/packaging**: the core auto-flows into `jni/Android.mk.in` via meson
`nntrainer_sources`. Applications/CausalLM is its own build; check whether the new
runtime needs `Applications/.../Android.mk` / meson entries if Android coverage is
expected. Confirm with the user whether this PR targets Android too.

---

## 7. Build & verify (CPU runtime needs the app build)

```bash
# transformer/CausalLM runtime build (this host, NOT docker): gcc-13
# locally comment out the DeepQ subdir in Applications/meson.build (do NOT commit)
meson setup build-app -Denable-test=true -Denable-transformer=true -Denable-fp16=false \
  -Denable-tflite-backbone=false -Denable-tflite-interpreter=false
ninja -C build-app
meson test -C build-app unittest_mha_core_mask unittest_kv_cache_manager -v
```
- End-to-end gemma4 DDTree run weight: `res/gemma4/gemma4-e2b/nntr_gemma4_q40_embdq6k.bin`
  (Q4_0). Decode must equal greedy modulo genuine Q4_0 near-ties (proven numeric,
  not a bug — see `ddtree-project-state` memory).
- The 32-node tree is element-by-element identical to Python `ddtree_ref` across
  all blocks (verification scripts in `python-example/verify/`, kept OUT of the PR).

---

## 8. PR description

Follow nntrainer/nntrainer PR #3928's style (the core PR's description is a good
local template too). Include: motivation, what's added per layer, **"stacked on
#3972"** dependency note, build/test instructions, and a parity/verification
summary. End WITHOUT any "Co-Authored-By" / Claude attribution.

---

## 9. First moves for the new session

1. Read this doc + the two memories.
2. `git fetch upstream && git fetch origin` and confirm `ddtree-core` (== #3972)
   and `add/DDTree` tips.
3. Ask the user the open decisions: (a) keep env-gated dumps or build a clean API?
   (b) does this PR target Android packaging too? (c) one PR or split into the 7
   commits as separate stacked PRs?
4. Rebuild `ddtree-gemma4` per §5, scrub host-local (§4), apply headers/format (§6),
   build+test (§7), then push and open the PR.
