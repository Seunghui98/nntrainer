#!/bin/bash
# DDTree verification runbook — run inside the my-dev container:
#   docker exec my-dev bash /workspace/qwen3run/verify_all.sh
set +e
NN=/workspace/nntrainer
QR=/workspace/qwen3run
PY=$QR/venv/bin/python

echo "############################################################"
echo "# 1) DDTree pure-core C++ unit tests  (expect: 19 PASSED)"
echo "############################################################"
"$NN"/build/test/unittest/unittest_ddtree 2>&1 | tail -3

echo
echo "############################################################"
echo "# 2) Task 9  KVCacheManager::compactTail  (expect: PASSED)"
echo "############################################################"
"$NN"/build-app/Applications/CausalLM/unittest_kv_cache_manager --gtest_filter='*CompactTail*' 2>&1 | tail -4

echo
echo "############################################################"
echo "# 3) Task 10 mha_core additive-mask softmax  (expect: PASSED)"
echo "############################################################"
"$NN"/build-app/Applications/CausalLM/unittest_mha_core_mask 2>&1 | tail -4

echo
echo "############################################################"
echo "# 4) Tree-build parity: 60 cases  C++(nntrainer) vs ddtree.py"
echo "#    (expect: 60/60 ALL-FIELDS IDENTICAL)"
echo "############################################################"
cd "$QR" && "$PY" run_batch.py 2>&1 | grep -vi warning | tail -6

echo
echo "############################################################"
echo "# 5) Tree node log-prob (logw) VALUE parity  (fp32 level)"
echo "############################################################"
cd "$QR" && "$PY" logw_compare.py 2>&1 | grep -viE 'warning|^ id|e-0|e-1' | tail -5

echo
echo "############################################################"
echo "# 6) Python self-draft DDTree e2e  ==  plain greedy  (slow)"
echo "############################################################"
cd "$QR" && "$PY" e2e_py.py 2>&1 | grep -E 'accept lengths|GEN_TEXT'
echo "  (HF greedy 64-token reference:)"
grep 'GEN_TEXT' /tmp/hfref.log 2>/dev/null | head -1

echo
echo "==== DONE. Raw value dumps you can inspect: ===="
echo "  $QR/trace/py_tree.json   $QR/trace/cpp_tree.json   (single 32-node tree, all fields)"
echo "  $QR/cases/cases.json                               (60 case configs)"
echo "  $QR/e2e/py_e2e.json                                (python e2e tokens + per-block accept)"
