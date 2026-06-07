# DDTree on nntrainer — RESUME (handoff for a new session)

**Branch:** `add/DDTree`  ·  **Last updated:** session 3 (DDTree runtime DONE & proven)

## TL;DR status
DDTree (tree speculative decoding) is implemented in nntrainer AND proven correct
end-to-end on Qwen3-0.6B (CPU, fp32): nntrainer's full DDTree decode == greedy
**69/69 tokens (lossless)**, and every stage matches the Python ground truth
(`python-example/ddtree.py`). All code is committed on `add/DDTree`.

## Where things are
- Repo: `/workspace/nntrainer` inside Docker container **`my-dev`** (ubuntu:20.04, root).
- Verification suite + guide: `python-example/verify/` (README.md + setup.sh + scripts).
- Working dir (venv, downloaded model, converted weights, built harnesses):
  `/workspace/qwen3run/` — **persists on this host's docker volume**; on a fresh
  machine run `python-example/verify/setup.sh` to recreate it.
- Progress log: `docs/superpowers/plans/2026-06-05-ddtree-nntrainer-PROGRESS.md`.
- Plan: `docs/superpowers/plans/2026-06-05-ddtree-nntrainer.md`.

## What is DONE (committed)
- Task 1-8,11,12,16: pure-core DDTree module (buildTree/compile/follow/compactTail/
  sliding/sampling) + golden parity + README. `unittest_ddtree` 19/19.
- Task 9: `KVCacheManager::compactTail` (+ elemSize UINT16 fix, commit 8fa141f8).
- Task 10: `mha_core` additive-mask non-causal attention (internal + external paths).
- mha per-token RoPE positions; lm_head/TieWordEmbedding all-position verify dump.
- `CausalLM::runDDTreeDump`: full self-draft DDTree decode runtime (dormant; env
  `NNTR_DDTREE_DUMP`). nntr decode == greedy 69/69.
- Verified: tree 60/60 + logw fp32; base 64/64 vs HF; verify node logits 32/32;
  full decode 69/69. Per block: 16-token window -> 32-node tree (budget 31).

## Build (gcc-10, fp16=false, ninja -j4) — see verify/README.md
Pure core: `CC=gcc-10 CXX=g++-10 meson setup --wipe build -Denable-fp16=false -Denable-test=true -Denable-tflite-backbone=false -Denable-tflite-interpreter=false -Denable-app=false && ninja -j4 -C build`
App: `build-app` with `-Denable-app=true -Denable-transformer=true -Denable-fp16=false`.

## How to re-verify (fast)
    docker start my-dev
    docker exec my-dev bash /workspace/nntrainer/python-example/verify/verify_all.sh
    docker exec my-dev /workspace/qwen3run/venv/bin/python /workspace/nntrainer/python-example/verify/demo_verify.py

## Gotchas learned (don't re-discover)
1. GCC 9 too old (no `_Float16`, no `_mm256_loadu2_m128i`) -> use gcc-10, fp16=false.
2. 7.8 GiB RAM -> `ninja -j4` (default OOMs, once crashed Docker Desktop).
3. fp16=false stores KV cache as **UINT16 (2 bytes)** -> compactTail elemSize=(FP32?4:2).
4. cross-.so `dynamic_cast` to layer types returns null -> use `getType()` + static
   setters (MHACoreLayer::setGlobalDDTreeVerify, TieWordEmbedding::setGlobalVerifyDump).
5. masked verify needs `is_causal` temporarily false (compute_kcaches/vcache layout).
6. lm_head/TieWordEmbedding emit last-token logits only -> verify uses a separate
   all-position dump buffer.

## Possible next work (not done)
- Clean up env-gated debug dumps in `runDDTreeDump` (block/cache diagnostics) if a
  production path is wanted; or promote `runDDTreeDump` to a real `runDDTree` decode
  mode (currently dump-only, greedy, self-draft).
- Sliding-window models (gemma) path: `makeSlidingMasks` exists but untested
  (qwen3-0.6b is full-attention).
- Real DFlash draft model (not self-draft) for true speculative speedup + Task 13/14
  trace parity (needs the littlebit-draft checkpoint, absent here).
- ARM build (only x86_64 tested here).
