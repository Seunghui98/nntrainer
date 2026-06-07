#!/bin/bash
# Recreate the DDTree verification working dir (/workspace/qwen3run) from scratch.
# Run inside the my-dev container:  bash python-example/verify/setup.sh
set -e
QR=/workspace/qwen3run
NN=/workspace/nntrainer
mkdir -p "$QR"
cd "$QR"

echo "== 1. system deps (gcc-10 + app libs + python3.9) =="
DEBIAN_FRONTEND=noninteractive apt-get update -y >/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y gcc-10 g++-10 \
  libjsoncpp-dev libopencv-dev python3.9 python3.9-venv python3.9-distutils >/dev/null

echo "== 2. python venv (torch cpu + transformers>=4.51) =="
[ -d venv ] || python3.9 -m venv venv
./venv/bin/pip -q install --upgrade pip
./venv/bin/pip -q install torch --index-url https://download.pytorch.org/whl/cpu
./venv/bin/pip -q install 'transformers>=4.51' numpy accelerate safetensors sentencepiece huggingface_hub

echo "== 3. download Qwen3-0.6B =="
./venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-0.6B', local_dir='$QR/hf/Qwen3-0.6B')"

echo "== 4. convert weights -> nntrainer safetensors (into res dir) =="
RES=$NN/Applications/CausalLM/res/qwen3/qwen3-0.6b
./venv/bin/python "$RES/weight_converter.py" --model_path "$QR/hf/Qwen3-0.6B" \
  --output_name "$RES/nntr_qwen3_0.6b_fp32" --safetensors
# copy HF json + tokenizer into the model dir if missing
for f in config.json generation_config.json tokenizer.json tokenizer_config.json merges.txt vocab.json; do
  [ -f "$RES/$f" ] || cp "$QR/hf/Qwen3-0.6B/$f" "$RES/"
done
# greedy generation_config + nntr_config must already exist in res dir (committed)

echo "== 5. regenerate ddtree_ref.py from python-example/ddtree.py =="
cp "$NN/python-example/verify"/*.py "$QR/" 2>/dev/null || true
cp "$NN/python-example/verify"/*.cpp "$QR/" 2>/dev/null || true
cp "$NN/python-example/verify/verify_all.sh" "$QR/" 2>/dev/null || true
./venv/bin/python "$QR/rebuild_ref.py"

echo "== 6. build C++ parity harnesses =="
cd "$NN"
g++-10 -std=c++17 -O2 -I nntrainer/ddtree "$QR/ddtree_parity_harness.cpp" \
  nntrainer/ddtree/ddtree.cpp nntrainer/ddtree/ddtree_compact.cpp \
  nntrainer/ddtree/ddtree_sampling.cpp nntrainer/ddtree/ddtree_sliding.cpp -o "$QR/ddtree_parity"
g++-10 -std=c++17 -O2 -I nntrainer/ddtree "$QR/ddtree_logw.cpp" \
  nntrainer/ddtree/ddtree.cpp nntrainer/ddtree/ddtree_compact.cpp \
  nntrainer/ddtree/ddtree_sampling.cpp nntrainer/ddtree/ddtree_sliding.cpp -o "$QR/ddtree_logw"

echo "== 7. build nntrainer (pure-core + app) =="
CC=gcc-10 CXX=g++-10 meson setup --wipe build -Denable-fp16=false -Denable-test=true \
  -Denable-tflite-backbone=false -Denable-tflite-interpreter=false -Denable-app=false
ninja -j4 -C build
git submodule update --init Applications/CausalLM/third_party/minja
CC=gcc-10 CXX=g++-10 meson setup build-app -Denable-fp16=false -Denable-test=true \
  -Denable-transformer=true -Denable-tflite-backbone=false -Denable-tflite-interpreter=false
ninja -j4 -C build-app Applications/CausalLM/nntr_causallm \
  Applications/CausalLM/unittest_kv_cache_manager Applications/CausalLM/unittest_mha_core_mask

echo "== DONE. Now run: bash $NN/python-example/verify/verify_all.sh =="
