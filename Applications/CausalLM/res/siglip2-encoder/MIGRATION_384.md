# SigLIP2 224 → 384 migration — verification guide

Branch `claude/siglip2-encoder-migration-zx5qp2` (base: PR #4061 /
`support/screen_ai`) moved the vision encoder from the 224px checkpoint
verified in PR #4007 to `siglip2-base-patch16-384` (576 patches instead of
196). The code changes are build-verified and the resize kernel is checked
bit-exact against PIL, but nothing here has been run against the real 384px
checkpoint yet — no checkpoint or device access was available while writing
this. This doc is the remaining checklist, plus a prompt for handing that
part off to an agent.

## Status

| Item | State |
| :-- | :-- |
| `NUM_PATCHES` derived from `image_size`/`patch_size` (no more hardcoded 196) | done, build-verified |
| PIL BICUBIC resample support, selectable via `nntr_config.json` | done, verified vs PIL (0/442368 px diff) |
| `BertDecoder` cross-attention length config-driven (`setEncoderLength()`) | done, build-verified |
| `res/siglip2-encoder`, `res/bert-decoder` configs updated to 384px/576 | done |
| Encoder parity vs the real 384px checkpoint | **not run** — needs checkpoint |
| Decoder parity vs the real 384px checkpoint | **not run** — needs checkpoint + new golden |
| `resample` filter (bilinear vs bicubic) confirmed against the checkpoint's own `preprocessor_config.json` | **not confirmed** — currently guessed as `"bicubic"` |

## 0. Prerequisites

Desktop x86 build. These are the packages this branch needed on a bare
Ubuntu 24.04 box; skip whatever you already have:

```bash
pip3 install meson
sudo apt-get install -y libopenblas-dev flatbuffers-compiler \
  libjsoncpp-dev libcurl4-openssl-dev nlohmann-json3-dev ninja-build
```

`Applications/CausalLM/json.hpp` is not tracked in git and is fetched at
build time. The normal path is:

```bash
bash jni/prepare_encoder.sh <builddir> 0.2
```

If that download is blocked, `nlohmann-json3-dev`'s header is a drop-in
substitute (same upstream single header):

```bash
cp /usr/include/nlohmann/json.hpp Applications/CausalLM/json.hpp
```

## 1. Build

```bash
meson setup builddir-desktop -Denable-transformer=true \
  -Denable-tflite-backbone=false -Denable-tflite-interpreter=false
ninja -C builddir-desktop \
  Applications/CausalLM/nntr_causallm \
  Applications/CausalLM/nntr_quantize
```

(`enable-tflite-*` is only disabled here because it's unrelated to this
model; keep your normal flags if your build already sets them.)

## 2. Check the resample filter before converting anything

Do this before weight conversion — a resample mismatch produces a small,
consistent error across every downstream number instead of an obvious
failure.

```bash
python3 -c "import json; print(json.load(open('$CKPT/preprocessor_config.json'))['resample'])"
# 2 -> bilinear, 3 -> bicubic (PIL / HF numbering)
```

Compare against `res/siglip2-encoder/nntr_config.json`'s `"resample"` key
(currently set to `"bicubic"`, based on HF's `SiglipImageProcessor` default —
not yet confirmed against an actual checkpoint) and fix it if it disagrees.

## 3. Convert weights

```bash
W=/tmp/siglip2-384-check && mkdir -p "$W"

# encoder — weight_converter.py reuses all 199 tensors unchanged; shapes are
# read dynamically from the safetensors file, no script changes needed.
python3 Applications/CausalLM/res/siglip2-encoder/weight_converter.py \
  --model_path "$CKPT/model.safetensors" \
  --encoder_output "$W/nntr_siglip2_encoder_fp32.bin"

# decoder (BERT-small dims are unrelated to the vision resolution)
python3 Applications/CausalLM/res/bert-decoder/weight_converter.py \
  --model_path "$CKPT/model.safetensors" \
  --decoder_output "$W/nntr_bert_decoder_fp32.bin"

cp Applications/CausalLM/res/siglip2-encoder/{config.json,nntr_config.json,generation_config.json,sample.png} "$W/"
cp Applications/CausalLM/res/bert-decoder/{config.json,nntr_config.json,generation_config.json,tokenizer.json} "$W/"
```

## 4. Encoder parity

`verify_encoder.py` builds the PyTorch golden through `AutoImageProcessor`
(whatever resolution/filter/mean/std the checkpoint actually declares), then
compares it against nntrainer's `--dump-encoder` output.

```bash
cd "$W"
python3 <nntrainer_repo>/Applications/CausalLM/res/siglip2-encoder/verify_encoder.py \
  --ckpt "$CKPT" --image sample.png --out golden

<nntrainer_repo>/builddir-desktop/Applications/CausalLM/nntr_causallm \
  --dump-encoder "$W" sample.png

python3 <nntrainer_repo>/Applications/CausalLM/res/siglip2-encoder/verify_encoder.py \
  --ckpt "$CKPT" --image sample.png --out golden \
  --nntr-npy nntr_encoder_hidden.npy
```

Check `golden encoder_hidden shape` prints `(1, 576, 256)` first — if it
doesn't, either `config.json`'s `image_size` is wrong or the checkpoint
itself isn't 384px.

## 5. Decoder parity

Reuse the golden from step 4 as the decoder's golden and confirm
cross-attention actually consumes the new 576-length encoder output.

```bash
<nntrainer_repo>/builddir-desktop/Applications/CausalLM/nntr_causallm \
  --decoder-init-parity "$W"
```

Compare `nntr_decoder_init_logits.npy`'s argmax and cosine similarity
against the PyTorch decoder run on the same `golden.encoder_hidden.npy`,
following the same pattern as `compare_bert_vs_python.sh` in PR #4061's
description.

## 6. (Optional) On-device

```bash
cd Applications/CausalLM
./build_android.sh --cache && ./install_android.sh
adb push "$W"/* /data/local/tmp/nntrainer/causallm/models/siglip2-384/
adb shell /data/local/tmp/nntrainer/causallm/run_causallm.sh \
  --dump-encoder /data/local/tmp/nntrainer/causallm/models/siglip2-384
```

## Pass criteria

| Stage | Metric | Threshold |
| :-- | :-- | :-- |
| Resize | pixel diff (uint8) vs PIL | 0 / 442368 |
| Encoder (x86 fp32) | cosine | ≥ 0.9999 |
| Encoder (x86 fp32) | rel-L2 | ≤ 1e-2 |
| Encoder (ARM fp16) | cosine | ≥ 0.999 |
| Decoder | argmax token | matches PyTorch |
| Decoder | logits cosine | ≥ 0.9999 |

The x86 encoder thresholds are the numbers PR #4007 actually measured at
224px (cosine 0.999998). If 384px can't clear the same bar, suspect the
resample filter first.

## Delegating the rest to an agent

Everything past step 1 needs a real 384px checkpoint (and step 6 needs an
adb device), which this session didn't have. That makes it a reasonable
hand-off. Prompt template — fill in the checkpoint path before sending, the
rest is self-contained:

```text
Repo nntrainer (Seunghui98/nntrainer or nntrainer/nntrainer), branch
claude/siglip2-encoder-migration-zx5qp2 (base: PR #4061 / support/screen_ai).
It moves the SigLIP2 vision encoder from 224px to 384px
(google/siglip2-base-patch16-384-class, actually a VisionEncoderDecoder
checkpoint retrained at that resolution). Follow
Applications/CausalLM/res/siglip2-encoder/MIGRATION_384.md in the repo for
the exact steps and pass criteria.

Already done (4 commits, build-verified only — never run against a real
checkpoint):
1. NUM_PATCHES derived from (image_size/patch_size)^2 instead of a
   hardcoded 196.
2. PIL BICUBIC resample support, selectable via nntr_config.json's
   "resample" key.
3. BertDecoder's cross-attention length made runtime-configurable via
   setEncoderLength() instead of a hardcoded BD_ENC_LEN.
4. res/siglip2-encoder and res/bert-decoder configs updated to 384px/576
   patches; verify_encoder.py switched to AutoImageProcessor.

Your job:
1. Get a 384px-encoder VisionEncoderDecoder checkpoint (ask the user for
   the path — do not guess or hardcode one).
2. Confirm the checkpoint's preprocessor_config.json "resample" value
   against res/siglip2-encoder/nntr_config.json's "resample" (currently
   "bicubic", unconfirmed) and fix it if they disagree.
3. Convert encoder + decoder weights with the two weight_converter.py
   scripts.
4. Run verify_encoder.py + nntr_causallm --dump-encoder; target cosine >=
   0.9999, rel-L2 <= 1e-2 (x86 fp32, same bar PR #4007 hit at 224px).
5. Regenerate golden.encoder_hidden.npy at [1,576,256] and run
   nntr_causallm --decoder-init-parity; target: argmax match, logits
   cosine >= 0.9999.
6. If something fails, check the resample filter first, then whether the
   checkpoint's actual hidden_size/num_hidden_layers/etc. match config.json's
   defaults (768/12/12/3072).
7. Report the numbers (PASS/FAIL per metric). If res/ configs needed
   fixing, commit that to the same branch, matching the existing commit
   style: "[CausalLM] ..." subject, Signed-off-by: SeungHui Lee, Co-authored-by:
   Claude trailers.

If you don't have checkpoint or device access, say exactly where you're
blocked and what you need — do not proceed on assumptions.
```

Two things worth keeping in that prompt specifically: **"do not proceed on
assumptions"** — the resample mismatch is the kind of failure that's easy to
paper over with a slightly-off number instead of catching; and **the commit
style line** — this branch already has a fixed message format the agent
won't otherwise know to follow.
