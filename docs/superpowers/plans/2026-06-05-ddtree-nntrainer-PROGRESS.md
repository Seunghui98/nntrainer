# DDTree nntrainer — Implementation Progress / Handoff

**Last updated:** 2026-06-05 (session 2)
**Branch:** `add/DDTree` (repo `/home/shsh1004/nntrainer`)
**Plan:** `docs/superpowers/plans/2026-06-05-ddtree-nntrainer.md` (16 tasks)
**Spec:** `docs/superpowers/specs/2026-06-05-ddtree-nntrainer-design.md`
**Python ground truth:** `/home/shsh1004/littlesd_inference/ddtree.py`

---

## Status board

| Task | Status | Notes |
|------|--------|-------|
| 1. Scaffold module + meson + types | ✅ DONE | committed |
| 2. buildTree empty/root-only case | ✅ DONE | committed |
| 3. buildTree heap+tie-break+visibility | ✅ DONE | parity-critical comparator; committed |
| 4. compile ids/positions/mask | ✅ DONE | committed |
| 5. makeSlidingMasks | ✅ DONE | committed |
| 6. followVerified | ✅ DONE | committed |
| 7. compactTail raw helper | ✅ DONE | committed |
| 8. sampling helper (argmax/greedy) | ✅ DONE | committed |
| 11. Python golden dumper | ✅ DONE | pure-stdlib transcription; committed |
| 12. buildTree golden parity test | ✅ DONE | **all 4 golden cases pass incl. tie-break**; committed |
| 16. README + CPU guide | ✅ DONE | committed |
| 9. KVCacheManager::compactTail | ✅ DONE | committed 9cab91fa; CompactTail test PASS |
| 10. mha_core additive-mask path | ✅ DONE | committed 3a076328; MaskedSoftmax test PASS; qwen3 no-regression |
| 13. trace-replay parity test | ⬜ pending (needs torch + checkpoints) | not started |
| 14. CPU gemma4 harness | ⬜ pending (needs app build + checkpoints) | not started |
| 15. full build+test sweep | 🟡 PARTIAL | pure-core sweep green (320/320 build, 19/19 ddtree tests); app+runtime tests pending |

**Pure core (Phase A) + golden parity + README are COMPLETE and committed.**
**19/19 `unittest_ddtree` tests pass; full pure-core build is clean (exit 0).**

### This session's commits (on `add/DDTree`, after spec `27552da2`)
```
55127cf6 scaffold pure core module + meson wiring + scaffold test
a4c17498 buildTree empty/root-only case
0a12e633 step-by-step implementation plan
88342f98 progress/handoff doc (session 1)
696245a7 buildTree heap expansion + tie-break + visibility
d2dea216 compile verify ids/positions/additive-mask
b181309a sliding-window additive-mask stage
4c923c87 followVerified accepted-path walk
d0b4ec5a raw-pointer compactTail KV tail reorder helper
20f89eea greedy sampling convenience helper
63418484 pure-stdlib golden-vector dumper + buildTree golden vectors
e0837fd3 buildTree exact parity vs Python golden vectors
be8b7c0a README architecture/usage/parity/CPU guide + verification record
(+ this PROGRESS update)
```

---

## What's left (Tasks 9, 10, 13, 14) and the exact unblock steps

### Unblock 1 — system dev packages (DECIDED: install)
Tasks 9, 10, 14 touch `Applications/CausalLM/`. Building `Applications/` needs
`-Denable-app=true` (default) + `-Denable-transformer=true`, which **also
force-builds** `ReinforcementLearning/DeepQ` (needs `libjsoncpp-dev`) and
`YOLOv2/3` (needs `libopencv-dev`). There is **no per-app flag** to skip them.
This machine has **no passwordless sudo**, so the user installs:
```bash
sudo apt install -y libjsoncpp-dev libopencv-dev
```
Then configure a SEPARATE app build (keep the pure-core `build/` as-is):
```bash
meson setup build-app \
  -Denable-fp16=true -Denable-test=true -Denable-transformer=true \
  -Denable-tflite-backbone=false -Denable-tflite-interpreter=false
```
Build/run a target with e.g. `meson test -C build-app unittest_kv_cache_manager -v`.

### Unblock 2 — Python torch + gemma4 checkpoints (for Tasks 13 & 14 parity)
The reference venv `/home/shsh1004/littlesd_inference/venv` is **broken**: its
`python`/`python3`/`python3.10` symlinks point to a missing `/usr/bin/python` /
`python3.10` (system is python3.12, no numpy/torch). To run `ddtree.py` /
`ddtree_generate` (needed to capture the gemma4 trace for Task 13 and to compare
tokens for Task 14), the user must provide a working torch env (recreate the venv
on py3.12 and `pip install -r requirements.txt`, or fix the interpreter) **and**
the gemma4 + littlebit-draft checkpoints. Until then:
- Task 13 test should `GTEST_SKIP()` when `gemma4_trace.json` is absent.
- Task 14 harness can be built and run as far as model load; full token parity is
  produced once checkpoints exist.

---

## CRITICAL environment findings (carry forward)

1. **Pure-core build/test command** (NOT the plan's — extra flags required):
   ```bash
   meson setup build -Denable-fp16=true -Denable-test=true \
     -Denable-tflite-backbone=false -Denable-tflite-interpreter=false -Denable-app=false
   meson test -C build unittest_ddtree -v
   ```
   `build/` already exists configured this way. Use `meson test` (auto-rebuilds);
   `ninja -C build unittest_ddtree` does NOT work — ninja needs the full path
   `test/unittest/unittest_ddtree`.
2. **Test convention:** each unittest defines its own `int main()` at the END of
   the file. ⚠️ Insert new `TEST(...)` blocks BEFORE that `main()`.
3. **Module real path:** `/home/shsh1004/nntrainer/nntrainer/ddtree/` (single
   `nntrainer/` subdir). `#include <ddtree.h>` resolves via `nntrainer_dep`.
4. **Test target wiring:** `unittest_ddtree` is a standalone `executable()` block
   in `test/unittest/meson.build` (not the shared `test_target` loop) carrying
   `-DDDTREE_GOLDEN_DIR="<src>/ddtree_golden"`.
5. **Golden fp32 parity:** `gen_golden.py` MUST round `top_log_probs` to fp32
   (via `struct.pack('f')`) — `ddtree.py` stores them fp32 (line 117) and the C++
   core uses fp32; without this, tied `logw` values pop in a different heap order
   (`budget_gt_vocab` case caught this).

---

## How to resume (next concrete step)

After `libjsoncpp-dev`/`libopencv-dev` are installed and `build-app` configured:
1. **Task 9** — `KVCacheManager::compactTail`. READ `Applications/CausalLM/
   kv_cache_manager.{h,cpp}` first to confirm the real API (exploration reported
   `getKeyCache(i)`/`getValueCache(i)`, members `layer_caches_`, `kv_widths_`,
   `cache_pos_`, `setPosition`; dtype member; `Tensor::getData<T>()`). The core
   helper `nntrainer::ddtree::compactTail` (already done & tested) does the
   per-layer reorder. Plan Task 9 has the full method + test.
2. **Task 10** — `mha_core` additive-mask path. READ
   `Applications/CausalLM/layers/mha_core.{h,cpp}` (anchors: `INOUT_INDEX::MASK=3`
   h:379-389; commented `// attention_mask,` h:398; `incremental_forwarding`
   cpp:406-564; score tensor `out_` cpp:767-771; `softmax_triangle` call line 778).
   Plan Task 10 has the full branch + numeric test.
3. **Task 13/14** — once torch + checkpoints exist (Unblock 2). Plan Tasks 13/14.
4. **Task 15** — final full sweep across `build` (pure core) and `build-app`
   (runtime tests).

The pure-core API (all done) is in `nntrainer/ddtree/README.md` with a usage
example; the runtime hooks call into it.

---

## Verification snapshot (current)
```
ninja -C build                       # 320/320, exit 0 (pure-core full build)
meson test -C build unittest_ddtree  # Ok: 1  Fail: 0
./build/test/unittest/unittest_ddtree
[==========] Running 19 tests from 7 test suites.
[  PASSED  ] 19 tests.
  DDTreeScaffold(1), DDTreeBuild(6 incl. MatchesPythonGolden),
  DDTreeCompile(1), DDTreeSliding(3), DDTreeFollow(3),
  DDTreeCompact(4), DDTreeSampling(2)
```


---

## Claims to verify (added session 3 — assistant answers that need empirical proof)

These were stated from code reading / reasoning and MUST be verified, not assumed:

1. **DDTree pure core is always compiled into libnntrainer (any arch, core & app builds).**
   - DONE for x86 pure-core (19/19 tests). TODO: confirm ddtree symbols in
     `build-app` libs (`nm -C .../libnntrainer.a | grep -i ddtree`).
   - ARM is UNTESTED in the my-dev container (x86_64 ubuntu:20.04) — treat the
     ARM half of the claim as a caveat, not verified.
2. **The additive-mask attention path is the internal-cache, 4-input case**
   (Q,K,V,MASK with MASK=INOUT_INDEX::MASK=3, is_causal=false), and MASK=3 does
   not collide with external-cache slots (`use_external_cache = numInputs>=5`).
   - Derived from `mha_core.cpp` (early `use_external_cache` return at ~l411 →
     `forwarding()`; mask path runs only when numInputs==4). VERIFY against how
     `causal_lm.cpp` / the DDTree verify harness actually binds MASK.
3. **gemma4 runs on CPU at runtime; fp16 is not required (fp32 works).**
   - UNVERIFIED end-to-end. fp16=false is forced here (GCC<12 lacks _Float16).
     Verify by building+running the Task 14 harness to model-load, and to full
     generation only when gemma4 checkpoints are provided.

### Env facts already verified empirically (not just claimed)
- GCC 9.4 lacks `_Float16` AND `_mm256_loadu2_m128i`; gcc-10 has the intrinsic
  (test-compiled) but still lacks `_Float16`. → build with gcc-10, fp16=false.
- 7.8 GiB RAM / 16 cores → default ninja OOMs; build with `ninja -j4`.
- Pure-core: 19/19 unittest_ddtree PASS. Task 9: KVCacheManager.CompactTail PASS
  (kv suite 18/18). build-app (enable-app, transformer, fp16=false) links clean.


---

## Runtime verification: qwen3-0.6b on CPU (session 3) — VERIFIED

Empirical proof for claim #3 (CPU runtime works; fp16 NOT required), using
Qwen3-0.6B since gemma4 checkpoints are unavailable.

**Pipeline:** HF `Qwen/Qwen3-0.6B` -> `res/qwen3/qwen3-0.6b/weight_converter.py
--safetensors` -> nntrainer F32 safetensors (2.38 GB) -> `nntr_causallm
res/qwen3/qwen3-0.6b` (built in build-app, gcc-10, fp16=false).

Model dir `Applications/CausalLM/res/qwen3/qwen3-0.6b/` now holds: config.json,
generation_config.json (do_sample=false for greedy), tokenizer.json/.txt/vocab,
nntr_config.json (model_file_name=nntr_qwen3_0.6b_fp32.safetensors), and the
converted safetensors.

**Result: token-for-token greedy parity vs HuggingFace transformers 4.57.6.**
- Same prompt (chat-formatted sample_input), greedy (do_sample=false), 64 new tokens.
- nntr first 64 generated tokens == HF 64 tokens, exact identical prefix 64/64,
  first divergence: none.
- Output: `<think>\nOkay, the user wants a short introduction to a large language
  model. ... Then explain` (identical on both sides).
- Perf (CPU fp32): prefill 18 tok @ 7.8 TPS, gen 64 tok @ 2.7 TPS, peak ~3.47 GB.

=> nntrainer CausalLM CPU runtime is numerically correct in fp32 (no fp16/_Float16
needed). This exercises the base model path (internal cache); DDTree tree-spec
decode itself is still Task 13/14. But the runtime foundation is proven.

---

## DDTree tree-build value parity (session 3) — VERIFIED at real scale (16/32)

Verifies the user's question: 'how do we confirm a 32-node tree is built from a
16-token block, and that the values match the Python ground truth?'

**Setup:** real ddtree.py functions (build/compile/follow extracted verbatim into
ddtree_ref.py) vs nntrainer C++ buildTree/compile/followVerified, fed the SAME
draft logits captured from qwen3-0.6b: shape [depth_limit=15, vocab=151936] fp32
(block_size=16, tree_budget=31), root=198, start=18, past=18.

C++ harness compiled WITHOUT libnntrainer (ddtree*.cpp are pure host):
  g++-10 -std=c++17 -I nntrainer/ddtree harness.cpp ddtree{,_compact,_sampling,_sliding}.cpp

**Result: ALL FIELDS IDENTICAL (C++ == Python).**
- node_count=31, current_length=32  => 32-node tree CONFIRMED from 16-token horizon.
- node_token_ids[31], node_depths[31], parents[32], verify_input_ids[32],
  verify_position_ids[32], visibility[32x32], additive-mask visibility[32x50],
  posterior[32], accepted_indices[14], next_token=-1 — every element matches.

Artifacts in /workspace/qwen3run: ddtree_ref.py, capture_logits.py, py_tree.py,
ddtree_parity_harness.cpp, compare_trees.py, trace/{draft_logits.npy,.f32,meta.json,
py_tree.json,cpp_tree.json}.

### Full end-to-end decode (Task: ddtree_generate parity) — BLOCKED on draft model
ddtree_generate needs a DFlashDraftModel (consumes target hidden states; the
'littlebit-draft' checkpoint) plus littlesd model.py/dflash.py — NOT in repo. So
the multi-block accept/commit loop cannot be reproduced as-is. The tree algorithm
itself is now proven byte-identical to Python; the missing piece for full E2E is
the draft model + its modules.

### Tree-build parity extended to MANY cases — 60/60 IDENTICAL
Not a single case: 5 prompts x 3 draft-logit windows x 4 budgets {7,15,31,63}
(tree sizes 8/16/32/64), all from real qwen3-0.6b logits. For every case the C++
buildTree/compile/followVerified output matched ddtree.py on ALL fields
(node ids/depths/parents, verify ids/positions, visibility, additive-mask
visibility, posterior, accepted path, next_token). current_length == budget+1 in
every case (tree-size control verified). Driver: run_batch.py over cases/cases.json.

### Tree node log-prob (logw) VALUE parity — fp32-level (not bit-exact, expected)
Beyond structure, compared each node's cumulative logw (the tree's log-prob value)
C++ vs Python over all 60 cases (ddtree_logw.cpp vs logw_compare.py):
- max |Δlogw| = 3.8e-6 across all cases (most ~1e-9..2e-6); 0/60 bit-exact.
- Cause (intended): torch computes logsumexp in fp32; ddtree.cpp accumulates
  logsumexp in double then stores fp32 ('for accuracy', parity §6.2). Difference
  is at/below fp32 epsilon for log-probs of magnitude up to ~40.
- This does NOT change heap ordering => tree STRUCTURE is byte-identical (60/60).
Conclusion: input draft logits are byte-identical both sides; derived log-prob
values agree to fp32 precision; resulting tree is identical.

---

## Stage A: Python end-to-end DDTree reference loop (self-draft) — DONE & self-verified
e2e_py.py runs the FULL DDTree decode loop on qwen3-0.6b using the REAL ddtree.py
functions (build/compile/follow + prepare_mask + compact_dynamic_cache) with
self-draft (qwen3 greedy horizon rollout) replacing the unavailable DFlash draft.

Loop: prefill -> per block: self-draft logits[15,vocab] -> build_ddtree_tree(31)
-> compile (verify ids/pos + additive tree mask) -> target verify forward with the
tree mask + DynamicCache -> argmax posterior -> follow_verified_tree -> commit +
compact_dynamic_cache (KV tail compaction) -> repeat.

Built-in correctness check: self-draft + greedy DDTree MUST equal plain greedy
decoding. Result: gen[:64] == HF greedy 64/64 tokens (first divergence none).
Accept lengths/block: [16,14,11,9,6,5,7] (tree speculation accepting multi-token).
This is the REFERENCE for the C++ harness (Stage B / Task 14).

Artifacts: /workspace/qwen3run/{ddtree_ref.py (6 verbatim fns), e2e_py.py, e2e/py_e2e.json}.

---

## Stage B (C++ DDTree e2e) progress — core made verify-capable; harness remains
Feasibility finding: real qwen3 uses the EXTERNAL-cache mha path (forwarding();
KV passed as cache_k_l{i}/cache_v_l{i} inputs), and RoPE position = from+row
(contiguous). DDTree verify needs a tree additive-mask AND per-node depth
positions on that path. Implemented as regression-safe core additions:
- d79aafc2: mha_core forwarding() extracts an optional additive MASK at input
  slot 5 (6-input external mode). Base (5 inputs) unaffected.
- cccb599d: mha_core per-token RoPE positions via optional input slot 6
  (tree_pos_); apply_rotary_emb uses positions[row] instead of from+row. Base
  (no slot 6) unaffected. Both verified: base qwen3 output byte-identical.
Now nntrainer can, in principle, run a tree verify forward (Task10 mask + these
positions + Task9 compactTail).

REMAINING (large, separable): a DDTree harness needs (a) a flag-gated qwen3 graph
variant exposing shared mask+pos input layers wired to every mha (7-input), (b)
the incremental_inference input-vector wiring for those, (c) a full loop: prefill
(causal mask+contiguous pos) -> per-block self-draft (throwaway forward + cache
position rollback) -> buildTree/compile -> verify forward (tree mask+depth pos) ->
argmax posterior -> followVerified -> KVCacheManager::compactTail -> repeat, then
(d) compare tokens to e2e/py_e2e.json. This is the bulk of Task 14 runtime work.

---

## Stage B COMPLETE: nntrainer C++ DDTree decode == greedy (69/69) — runtime proven
Implemented the full DDTree speculative-decode runtime in CausalLM::runDDTreeDump
(dormant; env NNTR_DDTREE_DUMP), exercising Task9 compactTail + Task10 mask +
per-token RoPE positions + lm-head all-position dump, all via process-global
setters (cross-.so dynamic_cast is unreliable):
prefill -> per block [self-draft 15-step greedy (KV rollback) -> ddtree::buildTree
/compile -> masked verify forward (node logits) -> argmax posterior ->
followVerified -> KVCacheManager::compactTail] -> repeat.

Stage-by-stage runtime parity vs Python/HF:
- 3-1 draft logits: argmax 15/15 (raw fp32 ~2e-2 cache-vs-nocache).
- 3-2 tree from runtime draft: all fields == ddtree.py.
- verify node logits (tree mask + depth positions): argmax 32/32 == HF.
- full decode: nntr tokens == greedy 69/69, accept_lengths [16,14,11,9,6,5,7]
  == Python e2e. LOSSLESS confirmed.

ROOT-CAUSE bug fixed: KVCacheManager::compactTail used elemSize=(FP16?2:4) but
the fp16=false build stores the KV cache as UINT16 (2 bytes, fp16 proxy), so the
real cache was compacted with a 4-byte stride and corrupted after the first
non-identity accept (block 2). Fix: elemSize=(FP32?4:2). Task9 unittest used an
FP32 cache so it missed this; only the live UINT16 model exposed it.

=> nntrainer runs DDTree end-to-end and is provably correct (lossless == greedy).
