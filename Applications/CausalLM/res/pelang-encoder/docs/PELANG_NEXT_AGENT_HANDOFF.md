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

## Since this snapshot (docs + quantize.cpp + mha_core.cpp follow-up)

Handoff docs were reviewed against the actual tree and found stale in three
places (see `docs/HANDOFF_PELANG.md`'s top banner for the annotated diff);
`docs/HANDOFF_PROMPT_GUIDE.md` was deleted (its blocker is resolved) and the
five PE-Lang docs were moved from the repo root into this `docs/` directory
so they ship with the resource dir instead of littering the repo root.

Two code issues found by re-deriving Wave 2 (quantize.cpp) from scratch and
by re-reading `mha_core.cpp`'s KV-cache change, both fixed and verified:

1. **`quantize.cpp` never actually ran for PE-Lang.** `is_encoder` only
   matched `"Siglip2VisionEncoder"`, `original_bin` only read
   `nntr_cfg["model_file_name"]` (PE-Lang's merged nntr_config.json only has
   `encoder_model_file_name`/`decoder_model_file_name`), and
   `buildEncoderLayerDtypeMap` unconditionally quantized `patch_embed_conv`
   for Q8_0 — which for PE-Lang's `pe_patch_conv` would violate trap §6.1
   (CRS = 3·14·14 = 588, not a multiple of 32) had the name even matched.
   Fixed: `is_encoder` now includes `PELangVisionEncoder`; `original_bin`
   falls back to the role-specific `encoder_model_file_name` /
   `decoder_model_file_name` key (mirrors `main.cpp`'s existing lookup
   order); `buildEncoderLayerDtypeMap` takes an explicit `patch_conv_name`
   + `allow_patch_conv_quant` (computed from `num_channels * patch_size^2 %
   32 == 0`) instead of a hardcoded name/architecture check. Verified
   end-to-end with a throwaway tiny synthetic PE-Lang encoder (CRS=27,
   deliberately not 32-aligned): `nntr_quantize --fc_dtype Q8_0` correctly
   quantized the 6 FC weights + `enc_to_dec_proj` and left `pe_patch_conv` /
   `_ls1` / `_ls2` / `pe_rope_sin` / `pe_rope_cos` / `pe_cls_row` /
   `pe_pos_embed` / all LayerNorms at FP32, and the output
   `nntr_config.json` kept `patch_embed_dtype: "FP32"`. Real-weight Q8_0
   numeric parity (G6) is still unverified — no PE-Lang checkpoint in this
   environment — but the wiring that was silently broken end-to-end no
   longer is.
2. **Latent `mha_core.cpp` `gemm_attention()` FP16-cache assumption.** This
   function is entirely dead code today (`#if defined(NNTR_ENABLE_GEMM_ATTENTION)`,
   never defined), but it unconditionally read the K/V cache as raw
   `uint16_t` FP16 bits — which this same PE-Lang commit made incorrect for
   non-`ENABLE_FP16` builds (x86 FP32), since `finalize()` now allocates
   that cache as FP32 there. Plan §4.2 explicitly proposes reviving this
   path for N=1025 prefill. Added a `NNTR_THROW_IF` guard at the top of
   `gemm_attention()` that fails loudly if the K/V cache isn't FP16, so a
   future §4.2 revival can't silently reinterpret FP32 bytes as garbage.
   No behavior change today (the function still never compiles in).

Four more follow-ups landed after that, everything verifiable without the
real checkpoint (built + tested in-tree; nothing here required golden data):

3. **Phase 3c unit tests** (previously missing) — `unittest_pe_rope.cpp` and
   `unittest_layer_scale.cpp`, covering CLS passthrough, the interleaved
   rotation formula, `incremental_forwarding` row-slicing, and a
   batch>1 case. All pass.
4. **`pe_rope`/`layer_scale` were single-threaded scalar loops.** The plan
   justified that by MAC count (<0.1%/<0.05% of encoder compute), but each
   streams 2-3x its row width through memory per row across 46 calls per
   encoder pass (23 layers x 2 sub-blocks) — bandwidth, not MACs, is what
   matters there. Threaded both over `(batch, row)` via
   `ThreadManager::parallel_for` (the same pattern `mha_core.cpp` already
   uses); the per-row math itself is untouched (still scalar, still no
   speculative NEON per trap §6.13). The new unit tests' batch>1 cases
   specifically exercise the `(batch*row)` index unflattening this
   introduced, and pass.
5. **`res/pelang-encoder/verify_encoder.py`** — the plan's Phase 5 res/
   deliverable list asked for this alongside `weight_converter.py`; it
   didn't exist. Comparison-only (no torch/timm dependency, unlike
   `res/siglip2-encoder/verify_encoder.py` which regenerates golden itself):
   feed it two `.npy` files and a `--gate {g2,g3,g4,g5,g6}` and it applies
   that gate's cosine/rel-L2/max-abs-diff/argmax thresholds from the plan.
   Verified against synthetic pass/fail cases.
6. **G7 (BICUBIC preprocessing) implemented**, not just planned.
   `encode(path)` was a stub that threw; it now calls a real bit-exact port
   of PIL's `Image.BICUBIC`, structured exactly like
   `siglip2_vision_encoder.cpp`'s existing bit-exact `Image.BILINEAR` port
   (same fixed-point `Resample.c` pipeline), with the bicubic (a=-0.5)
   kernel and support=2.0 swapped in for BILINEAR's triangle/support=1.0.
   Pulled out into its own header, `pelang_bicubic_resize.h`, so it's
   independently testable. **Actually diffed against real PIL** during
   development (not just structurally mirrored) across 5 shapes —
   downscale+noise, heavy upscale, exact-size/no-resize passthrough,
   asymmetric per-axis scale, and a degenerate 1x1 source — all
   **0/N pixels different**. One of those cases is frozen as a checked-in,
   PIL-independent regression test
   (`unittest_pelang_bicubic_resize.cpp`). G7's still not formally closed
   (that needs the full pipeline run against the real checkpoint + a
   documented device/CI run), but the actual hard part — does the resize
   kernel match — is now verified, not just asserted.

## Important files

- `Applications/CausalLM/models/pelang/pelang_vision_encoder.cpp`
- `Applications/CausalLM/models/pelang/pelang_bicubic_resize.h` (G7 BICUBIC
  resize; header-only so `unittest_pelang_bicubic_resize.cpp` can exercise it
  directly)
- `Applications/CausalLM/layers/pe_cls_pos.{h,cpp}`
- `Applications/CausalLM/layers/pe_rope.{h,cpp}` (now row-threaded — see above)
- `Applications/CausalLM/layers/layer_scale.{h,cpp}` (now row-threaded)
- `Applications/CausalLM/layers/mha_core.cpp`
- `Applications/CausalLM/quantize.cpp` (`buildEncoderLayerDtypeMap`, the
  `is_encoder`/`original_bin` resolution above the Step-4 dtype-map switch)
- `Applications/CausalLM/res/pelang-encoder/weight_converter.py`
- `Applications/CausalLM/res/pelang-encoder/verify_encoder.py` (new — gate
  comparison tool, see above)
- `Applications/CausalLM/res/pelang-encoder/nntr_config.json`
- `Applications/CausalLM/main.cpp`
- `Applications/CausalLM/res/pelang-encoder/docs/HANDOFF_PELANG.md`
- `Applications/CausalLM/res/pelang-encoder/docs/PELANG_L14_448_SUPPORT_PLAN.md`

## Remaining scope

Everything left needs either the real v3.0.0-C62 checkpoint, an Android
NDK + ARM device, or the external Quick.AI/screenai_onnx_compare repos —
none of which exist in a plain nntrainer checkout. Nothing below is
implementable further from source alone; the next agent needs that
environment first.

1. G5: ARM FP32 encoder parity (`cos > 0.999`, `rel-L2 < 5e-2`).
2. G6: ARM Q8_0 caption parity; greedy token difference <= 1. (Wiring for
   this is fixed — see above — but needs the real checkpoint + a device to
   actually run. There is also still no greedy-decode caption loop in
   nntrainer itself, only the 1-step `--decoder-init-parity` check; decide
   whether that loop belongs in Quick.AI or gets a temporary `main.cpp`
   driver for G6 testing.)
3. G7: run the now-implemented BICUBIC path against the real checkpoint's
   golden pixel/image pair and confirm the full preprocess-through-encoder
   chain (not just the resize kernel, which is already verified — see
   above).
4. G8: three-run e2e performance versus ONNX.
5. G9: peak RSS versus ONNX.
6. G10: existing SigLIP2 v2.3 regression (needs the external
   `screen_ai_v23_q80` model directory).
7. G11: x86 Meson (already green here) and Android NDK/device build and run.
8. Phase 6 Quick.AI integration, Phase 7 ONNX comparison. (Phase 5's
   quantize.cpp wiring is done; add a `res/pelang-encoder`-generated Q8_0
   `nntr_config.json` and run G6 once weights exist.)
9. Optional later: beam=3 decoding; a `--dump-taps` flag on `--dump-encoder`
   (or an equivalent script) so G2's per-stage s0/s1/s2/s3 comparison can be
   re-run from this tree instead of only from the golden-dump script's own
   copy -- deliberately not built without golden data to validate it
   against.

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

- Read `HANDOFF_PELANG.md` (now `docs/HANDOFF_PELANG.md` — check its stale-banner before trusting a section), `PELANG_L14_448_SUPPORT_PLAN.md`, and this file first.
- Preserve the SigLIP2 path and run G10 before claiming completion.
- Do not optimize kernels before profiling.
- Keep the change minimal and cross-platform.
- Run `clang-format-14` on changed C/C++ files.
- Do not commit unless explicitly requested; if committing, use `git commit -s`.
