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
| 9. KVCacheManager::compactTail | ⏭️ **NEXT** (needs app build) | not started |
| 10. mha_core additive-mask path | ⬜ pending (needs app build) | not started |
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
