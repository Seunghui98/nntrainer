# SigLIP2 224 → 384 migration — verification guide

Branch `claude/siglip2-encoder-migration-zx5qp2` (base: PR #4061 /
`support/screen_ai`) moved the vision encoder from the 224px checkpoint
verified in PR #4007 to `siglip2-base-patch16-384` (576 patches instead of
196). The code changes are build-verified and the resize kernel is checked
bit-exact against PIL, but nothing here has been run against the real 384px
checkpoint yet — no checkpoint or device access was available while writing
this. This doc is the remaining checklist, plus a prompt for handing that
part off to an agent.

**2026-08-20 update:** a real checkpoint was located locally
(`/home/leeseunghui/workspace/v4.0.0-S1/`) and inspected. That closed the
resample question (see Status) but also surfaced three *new* gaps this doc
didn't previously know about — the decoder's hidden dims, the
encoder→decoder connector's weight source, and a dead config field. See
"## Findings" below before touching anything; the "Delegating the rest to
an agent" section at the bottom has been rewritten to hand off the full,
current picture. No code has been changed yet — this pass was analysis
only, no build/convert/verify commands were actually run.

## Status

| Item | State |
| :-- | :-- |
| `NUM_PATCHES` derived from `image_size`/`patch_size` (no more hardcoded 196) | done, build-verified |
| PIL BICUBIC resample support, selectable via `nntr_config.json` | done, verified vs PIL (0/442368 px diff) |
| `BertDecoder` cross-attention length config-driven (`setEncoderLength()`) | done, build-verified |
| `res/siglip2-encoder`, `res/bert-decoder` configs updated to 384px/576 | done |
| `resample` filter (bilinear vs bicubic) confirmed against the real checkpoint | **confirmed: `"bicubic"` is correct** — see Findings §1 |
| Decoder hidden dims (`BD_DIM` etc.) match the real checkpoint | **confirmed WRONG** — repo has 256, checkpoint is 512; see Findings §2 |
| `ENC_TO_DEC_DIM` / decoder dims are config-driven (not hardcoded `constexpr`) | **not done** — see Findings §3 |
| `weight_converter.py` (encoder) loads the `enc_to_dec_proj` connector correctly | **not done** — see Findings §4 |
| Encoder parity vs the real 384px checkpoint | **not run** — checkpoint now available, code needs the Findings §2-4 fixes first |
| Decoder parity vs the real 384px checkpoint | **not run** — same; also needs a regenerated golden |

## Findings (2026-08-20 session)

A real checkpoint for this exact model exists locally at
`/home/leeseunghui/workspace/v4.0.0-S1/` (README: "ScreenAI caption v4.0.0
(S1)", `model_type: siglip2_b16_384_bert_small`). It is laid out as three
separate pieces, not one merged `model.safetensors` the way step 3's
commands assume:

```text
v4.0.0-S1/
├── siglip2-base-patch16-384/model.safetensors   # vision encoder only, "vision_model.*" prefix, 208 tensors
├── best/decoder/model.safetensors               # BertLMHeadModel decoder, "bert.*" prefix, 114 tensors
├── best/encoder_to_decoder.pt                   # raw torch state_dict {weight, bias} — the connector
└── README.md                                    # documents actual preprocessing ("Canonical Inference")
```

### §1. Resample: `"bicubic"` is confirmed correct

`v4.0.0-S1/code/evaluation/vqa100/infer_screen_model.py:78` does
`.resize((image_size, image_size), Image.Resampling.BICUBIC)` — this is the
actual eval/inference code for this checkpoint. The checkpoint's own
`siglip2-base-patch16-384/preprocessor_config.json` says `"resample": 2`
(bilinear), but that file is a byproduct of extracting just the vision
tower from `google/siglip2-base-patch16-384` via `AutoImageProcessor` — it
was never consulted by the actual training/eval pipeline, which does its
own manual resize (`model_config.json`'s `"preprocess":
"direct_resize_384"`, not `SiglipImageProcessor`). **Do not "fix" this to
bilinear** — trust the README/eval code over the stale HF artifact.

### §2. Decoder hidden dims: repo assumes 256, checkpoint is 512

`Applications/CausalLM/models/bert_decoder/bert_decoder.h` hardcodes (as
`static constexpr`, set unconditionally in the parameterless `BertDecoder()`
constructor):

| Constant | repo value | `v4.0.0-S1/best/decoder/config.json` value |
| :-- | :-- | :-- |
| `BD_DIM` (hidden_size) | 256 | **512** |
| `BD_INTERMEDIATE_SIZE` | 1024 | **2048** |
| `BD_NUM_HEADS` | 4 | **8** |
| `BD_HEAD_DIM` | 64 | 64 (unchanged — hidden/heads ratio preserved) |
| `BD_NUM_LAYERS` | 4 | 4 (unchanged) |

Confirmed directly from the decoder's own safetensors:
`bert.encoder.layer.0.crossattention.self.key.weight` is `[512, 512]`, not
`[256, 256]`. `res/bert-decoder/config.json`'s `"decoder"` block and
`res/bert-decoder/golden.encoder_hidden.npy` (currently `(1,196,256)`, the
old 224px golden) both need updating to match.

### §3. The connector exists but is hardcoded, and its weight source is wrong

The encoder→decoder projection is **not missing** — it already exists as
the last layer of `Siglip2VisionEncoder`
(`siglip2_vision_encoder.h:32,40`: `ENC_TO_DEC_DIM = 256`, a `static
constexpr`, referenced statically from `main.cpp:321`). It just has two
problems:

1. It's hardcoded to 256 (must be 512, matching `BD_DIM` above — the
   connector output dim and the decoder hidden dim are the same tensor by
   construction).
2. Unlike `IMG_SIZE`/`NUM_PATCHES` (instance fields resolved in
   `setupParameters(cfg, generation_cfg, nntr_cfg)`, already config-driven
   on this branch), `ENC_TO_DEC_DIM` was left as a compile-time constant
   instead of following that same pattern. This branch already made
   `image_size`/`num_patches`/decoder cross-attention length config-driven
   (see the earlier commits) — `ENC_TO_DEC_DIM` and the `BD_*` decoder dims
   in §2 are the pieces that got missed, and should be fixed the same way
   (a setter called before `initialize()`, mirroring the existing
   `setEncoderLength()` contract, since these are baked into the graph at
   build time).

Separately, `res/bert-decoder/nntr_config.json` already has a
`"decoder_hidden_size": 256` field — but `grep` across
`bert_decoder.{h,cpp}` and `main.cpp` confirms **it is never read
anywhere**. It's dead JSON. Whatever fix lands for §2/§3 should make this
field (or an equivalent) the actual, single source of truth instead of
adding a second hardcoded constant next to an unread config key.

### §4. `weight_converter.py`'s connector loading assumes the wrong file layout

`res/siglip2-encoder/weight_converter.py:167-168`:

```python
add("enc_to_dec_proj:weight", sd["enc_to_dec_proj.weight"], transpose=True)
add("enc_to_dec_proj:bias",   sd["enc_to_dec_proj.bias"])
```

This assumes `enc_to_dec_proj.{weight,bias}` live inside the same
`--model_path` safetensors as the encoder (`sd`, loaded from
`--model_path`). Confirmed by inspecting the real checkpoint: `sd` only has
`vision_model.*` keys (208 total) — no `enc_to_dec_proj` key exists there
at all. In the real checkpoint the connector is a **separate file**,
`best/encoder_to_decoder.pt`, a raw `torch.save`'d state dict with bare
`weight` ([512, 768]) / `bias` ([512]) keys (no prefix), loaded with
`torch.load`, not `safetensors.safe_open`. `weight_converter.py` needs a new
argument (e.g. `--connector_path`) to load this separately.

### §5. Housekeeping found along the way (not blocking, but should be fixed)

- `res/siglip2-encoder/nntr_siglip2_encoder_fp32.bin` and
  `res/bert-decoder/nntr_bert_decoder_fp32.bin` are currently **symlinks**
  pointing at a stale 224px build in a different workspace
  (`/home/leeseunghui/workspace/Quick.AI/nntrainer/...`). Replace with the
  freshly converted 384px files once conversion is redone; don't leave the
  symlinks in place.
- `main.cpp:360` (`--decoder-init-parity`) hardcodes `enc_len * 256` for the
  golden-encoder-hidden read size — this `256` needs to come from the same
  config-driven decoder hidden size as §2/§3, or it'll silently read the
  wrong number of floats once `BD_DIM` becomes 512.

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

Unlike the previous version of this doc, a real checkpoint now exists
locally (see Findings above) — the blocker is no longer "no checkpoint",
it's "the decoder's assumed dims and the connector's weight source are
wrong, and three constants that should be config-driven per this branch's
own established pattern aren't." Prompt template — self-contained, no
further checkpoint hunting needed:

```text
Repo nntrainer, branch claude/siglip2-encoder-migration-zx5qp2 (base: PR
#4061 / support/screen_ai). Follow
Applications/CausalLM/res/siglip2-encoder/MIGRATION_384.md in full —
especially the "## Findings (2026-08-20 session)" section — before making
any change. That section documents, with exact file/line references and
tensor-shape evidence already gathered, everything below; don't re-derive
it from scratch.

Checkpoint (already located, no need to ask the user): 
/home/leeseunghui/workspace/v4.0.0-S1/
- encoder: siglip2-base-patch16-384/model.safetensors (vision_model.* only)
- decoder: best/decoder/model.safetensors (bert.* , hidden=512, not 256)
- connector: best/encoder_to_decoder.pt (raw torch state_dict, keys
  "weight"/"bias", Linear(768,512), applied once before all decoder layers)

Already done on this branch (prior commits, build-verified only, never run
against a real checkpoint): NUM_PATCHES config-driven, BICUBIC resample
support, BertDecoder cross-attention length config-driven
(setEncoderLength()), res/ configs bumped to 384px/576. The resample
question is now closed — "bicubic" is confirmed correct (Findings §1) — do
NOT change it to bilinear despite the checkpoint's own
preprocessor_config.json saying resample:2; that field is a stale, unused
HF artifact (see Findings §1 for why).

Your job:
1. Make ENC_TO_DEC_DIM (siglip2_vision_encoder.h) and BD_DIM /
   BD_NUM_HEADS / BD_HEAD_DIM / BD_INTERMEDIATE_SIZE (bert_decoder.h)
   config-driven instead of hardcoded `static constexpr`, mirroring the
   existing setEncoderLength() setter pattern (called before initialize(),
   since these are baked into the graph at build time). Wire them to
   res/bert-decoder's existing (currently dead) "decoder_hidden_size"
   nntr_config.json field and a new equivalent for the encoder connector
   dim, and fix main.cpp:360's hardcoded "* 256". See Findings §2/§3 for
   the exact current values (256/1024/4/64 today) vs the real checkpoint's
   (512/2048/8/64).
2. Fix res/siglip2-encoder/weight_converter.py: it currently reads
   enc_to_dec_proj.weight/bias out of the SAME --model_path safetensors as
   the encoder (weight_converter.py:167-168) — the real checkpoint doesn't
   have that key there at all. Add a --connector_path argument to load it
   from a separate encoder_to_decoder.pt-style file instead (see Findings
   §4).
3. Update res/bert-decoder/config.json ("decoder" block) and
   res/bert-decoder/nntr_config.json to the real dims (512/2048/8), and
   res/siglip2-encoder/nntr_config.json with the new connector-dim key
   (512).
4. Build (builddir-desktop may already exist from a prior session — try
   `ninja -C builddir-desktop` first before re-running `meson setup`),
   convert both encoder and decoder weights from the v4.0.0-S1 paths above,
   and replace the stale symlinks in res/siglip2-encoder/ and
   res/bert-decoder/ (currently pointing at a different workspace's old
   224px build — Findings §5) with the freshly converted files.
5. Run verify_encoder.py + nntr_causallm --dump-encoder; target cosine >=
   0.9999, rel-L2 <= 1e-2 (x86 fp32, same bar PR #4007 hit at 224px).
   Careful: verify_encoder.py's AutoImageProcessor may pick up this
   checkpoint's stale bilinear preprocessor_config.json instead of the
   confirmed-correct bicubic (Findings §1) — check what it actually does
   and override if needed; don't trust a plausible-looking wrong number.
6. Regenerate golden.encoder_hidden.npy at the new [1,576,768] pre-connector
   / whatever shape the decoder actually consumes, and run nntr_causallm
   --decoder-init-parity against a real PyTorch forward through
   best/decoder + best/encoder_to_decoder.pt; target: argmax match, logits
   cosine >= 0.9999.
7. Report the numbers (PASS/FAIL per metric) and what you had to deviate
   from this prompt on, if anything. Commit to this branch, matching the
   existing commit style: "[CausalLM] ..." subject, detailed body
   explaining why (not just what), Signed-off-by: SeungHui Lee
   <shsh1004.lee@samsung.com>, Co-authored-by: Claude <noreply@anthropic.com>
   trailers.

If something is still genuinely blocked (e.g. decoder parity's PyTorch
reference can't be reproduced), say exactly where and what you need — do
not proceed on assumptions.
```

Two things worth keeping in that prompt specifically: **"do not proceed on
assumptions"** — a silently-wrong resample or dimension produces a
plausible-looking but incorrect number instead of an obvious failure; and
**the commit style line** — this branch already has a fixed message format
the agent won't otherwise know to follow.
