# PE-Lang L14-448 encoder — next-agent handoff

## Current status

Branch/worktree: `feature/pelang-l14-448-encoder` (worktree is intentionally dirty; preserve unrelated files).

The x86 FP32 encoder and decoder path is numerically validated:

| Gate | Result |
|---|---|
| G2 stage parity | positional cosine `1.0000000000`; block-0 cosine `0.9999999999` |
| G3 encoder parity | cosine `0.9999999950`, rel-L2 `1.00e-4` |
| G4 decoder parity | argmax `2048`; cosine `0.9999999998` |

The original compile blocker was fixed with the PE-Lang-only `pe_cls_pos` layer. The first-block mismatch was caused by `mha_core` storing K/V cache as `UINT16` in an FP32 build; the `#else` branch now uses FP32 cache. The `#ifdef ENABLE_FP16` branch remains FP16 for ARM FP16 builds.

Do not revert the cache conditional:

```cpp
#ifdef ENABLE_FP16
  // FP16 cache (ARM FP16 build)
#else
  // FP32 cache (x86 FP32 build)
#endif
```

Temporary `[PE-Lang DBG]` output and tensor dumps were removed from active sources. Existing `.orig` files and unrelated dirty files belong to the workspace; do not clean them broadly.

Latest verification: the complete x86 Meson/Ninja build completed successfully (105 targets). Android NDK is not currently installed at `/home/leeseunghui/workspace/android-ndk-r26`, so the Android portion of G11 is environment-blocked. The repository does not contain the SigLIP2 v2.3 checkpoint, so G10 requires the external `screen_ai_v23_q80` model directory/device setup.

## Important files

- `Applications/CausalLM/models/pelang/pelang_vision_encoder.cpp`
- `Applications/CausalLM/layers/pe_cls_pos.{h,cpp}`
- `Applications/CausalLM/layers/pe_rope.{h,cpp}`
- `Applications/CausalLM/layers/layer_scale.{h,cpp}`
- `Applications/CausalLM/layers/mha_core.cpp`
- `Applications/CausalLM/res/pelang-encoder/weight_converter.py`
- `Applications/CausalLM/res/pelang-encoder/nntr_config.json`
- `Applications/CausalLM/main.cpp`
- `HANDOFF_PELANG.md`
- `PELANG_L14_448_SUPPORT_PLAN.md`

## Remaining scope

1. G5: ARM FP32 encoder parity (`cos > 0.999`, `rel-L2 < 5e-2`).
2. G6: ARM Q8_0 caption parity; greedy token difference <= 1.
3. G7: C++ BICUBIC preprocessing bit-exact with PIL.
4. G8: three-run e2e performance versus ONNX.
5. G9: peak RSS versus ONNX.
6. G10: existing SigLIP2 v2.3 regression.
7. G11: x86 Meson and Android NDK/device build and run.
8. Phase 5 quantization validation, Phase 6 Quick.AI integration, Phase 7 ONNX comparison.
9. Optional later: `pe_rope`/`layer_scale` unit tests and beam=3 decoding.

## Build and x86 verification

```bash
cd /home/leeseunghui/workspace/nntrainer
ninja -C build Applications/CausalLM/nntr_causallm
export LD_LIBRARY_PATH=$PWD/build/nntrainer:$PWD/build/Applications/CausalLM/layers:$PWD/build/Applications/CausalLM/models/gpt_oss:$PWD/build/Applications/CausalLM/models/qwen3_moe:$PWD/build/Applications/CausalLM/models/qwen3_slim_moe:$PWD/build/Applications/CausalLM/models/gpt_oss_cached_slim:$PWD/build/Applications/CausalLM/models/qwen3_cached_slim_moe:$LD_LIBRARY_PATH
./build/Applications/CausalLM/nntr_causallm --dump-encoder Applications/CausalLM/res/pelang-encoder --input-pixels /home/leeseunghui/workspace/models/pelang_golden/pixel.npy
./build/Applications/CausalLM/nntr_causallm --decoder-init-parity Applications/CausalLM/res/pelang-encoder
```

Golden files are in `/home/leeseunghui/workspace/models/pelang_golden/`.

## Rules for the next agent

- Read `HANDOFF_PELANG.md`, `PELANG_L14_448_SUPPORT_PLAN.md`, and this file first.
- Preserve the SigLIP2 path and run G10 before claiming completion.
- Do not optimize kernels before profiling.
- Keep the change minimal and cross-platform.
- Run `clang-format-14` on changed C/C++ files.
- Do not commit unless explicitly requested; if committing, use `git commit -s`.
