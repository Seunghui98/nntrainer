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
"## Findings" below before touching anything.

**2026-08-20 second update:** the Findings §2-4 fixes (config-driven
`ENC_TO_DEC_DIM`/`BD_*` dims, `weight_converter.py` connector loading) were
implemented and build-verified in *this* environment — a Claude Code
remote-execution container, not the machine the checkpoint lives on. That
container has neither `/home/leeseunghui/workspace/v4.0.0-S1/` nor
huggingface.co reachable (outbound proxy returns 403 for both), so **the
actual weight conversion and encoder/decoder parity runs against the real
checkpoint (steps 3-5 below) were not executed** — see "## Handoff: what
still needs a machine with the checkpoint" at the bottom for the exact
commands to run there, and two more gaps this pass found (§6, §7) beyond
the original three. What *was* possible here: pypi.org is reachable (unlike
huggingface.co), so `torch`/`transformers`/`safetensors` install fine, and
both `weight_converter.py` scripts plus both verify scripts were
functionally run end-to-end against a synthetic checkpoint built locally
with the real layer counts and both known key-prefix layouts (see Findings
§7's "Update" note) — so the code paths themselves are exercised and
working, just not with the real checkpoint's actual weights.

## Status

| Item | State |
| :-- | :-- |
| `NUM_PATCHES` derived from `image_size`/`patch_size` (no more hardcoded 196) | done, build-verified |
| PIL BICUBIC resample support, selectable via `nntr_config.json` | done, verified vs PIL (0/442368 px diff) |
| `BertDecoder` cross-attention length config-driven (`setEncoderLength()`) | done, build-verified |
| `res/siglip2-encoder`, `res/bert-decoder` configs updated to 384px/576 | done |
| `resample` filter (bilinear vs bicubic) confirmed against the real checkpoint | **confirmed: `"bicubic"` is correct** — see Findings §1 |
| Decoder hidden dims (`BD_DIM` etc.) match the real checkpoint | confirmed wrong at 256; config now says 512/2048/8 — see Findings §2 |
| `ENC_TO_DEC_DIM` / decoder dims are config-driven (not hardcoded `constexpr`) | **done, build-verified** — see Findings §3 |
| `weight_converter.py` (encoder) loads the `enc_to_dec_proj` connector correctly | **done** (`--connector_path`), but prefix auto-detect also needed — see Findings §4/§6 |
| `weight_converter.py` (decoder) key prefixes match the real checkpoint | **done** (auto-detect) — see Findings §6 |
| `verify_encoder.py` / new `verify_decoder.py` support the real split checkpoint | done, functionally verified end-to-end against a synthetic checkpoint matching the real layer counts/layouts (torch/transformers ARE installable here via pip) — see Findings §7 |
| Encoder parity vs the real 384px checkpoint | **PASS (x86 fp32)** — cosine 0.99999071, rel-L2 4.31e-3, `--input-pixels`-bypassed run — see Findings §8 |
| Decoder parity vs the real 384px checkpoint | **PASS** — logits cosine 1.00000000, argmax exact match — see Findings §9 |

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

**§2-§5 status:** all fixed and build-verified in this pass (see the code
diff on this branch). `ENC_TO_DEC_DIM` is now an instance field
`enc_to_dec_dim_` resolved from `nntr_config.json`'s `"decoder_hidden_size"`
(read in `Siglip2VisionEncoder::setupParameters()`, so it needs no separate
setter — `setupParameters()` already runs in the constructor, before any
`initialize()` call). `BD_DIM`/`BD_NUM_HEADS`/`BD_HEAD_DIM`/
`BD_INTERMEDIATE_SIZE` are now instance fields with a new
`BertDecoder::setDecoderDims(hidden, num_heads, intermediate)` setter,
following the exact `setEncoderLength()` contract (call before
`initialize()`); `main.cpp`'s `--decoder-init-parity` and `quantize.cpp`'s
`BertDecoder` factory both call it now, reading `decoder_hidden_size` /
`decoder_num_heads` / `decoder_intermediate_size` from `nntr_config.json`
(`res/bert-decoder/nntr_config.json` and `res/siglip2-encoder/
nntr_config.json` both updated to 512/8/2048; the previously-dead
`decoder_hidden_size` key is now the actual source of truth). `main.cpp:360`
no longer hardcodes `256`. The stale symlinks from §5 do not exist in this
git repository (they were local files on the checkpoint machine, outside
version control) — nothing to replace here; the Handoff section below
covers creating the real files there instead.

### §6. `weight_converter.py` key-prefix mismatch (found fixing §4, not previously known)

Fixing §4's connector loading surfaced a second, blocking problem in the
*same* function: `res/siglip2-encoder/weight_converter.py`'s
`collect_encoder()` hardcodes `vp = "encoder.vision_model."`, but the real
checkpoint's `siglip2-base-patch16-384/model.safetensors` (a standalone
vision-tower-only file, per the Prerequisites in this doc's own header) only
has `vision_model.*` keys — no `encoder.` prefix. Every tensor lookup before
even reaching the connector would `KeyError`. Same problem in
`res/bert-decoder/weight_converter.py`: `bp = "decoder.bert."` and
`cp = "decoder.cls."` hardcoded, but the real
`best/decoder/model.safetensors` uses bare `bert.*` / `cls.*` (114-tensor
count matches either way, so the count assert wouldn't have caught this).

Fixed both converters with a `_detect_prefix()`/`_detect_vision_prefix()`
helper that checks which prefix variant is actually present in the loaded
state dict and uses that, so the same script works on both the old combined
checkpoint layout and the real split one.

### §7. `verify_encoder.py` can't load the real checkpoint's structure at all

`verify_encoder.py` assumed `transformers.VisionEncoderDecoderModel.
from_pretrained(ckpt)` — a single HF repo dir with both the vision tower and
the decoder under one `config.json`. The real checkpoint is three separate
pieces (own `config.json`s for the vision tower and the decoder, connector
in a raw `.pt`), so that call would fail outright before doing anything
useful. Rewrote it to support both: `--ckpt` keeps the old combined-checkpoint
path; `--vision_path` + `--connector_path` load the vision tower via
`SiglipVisionModel.from_pretrained()` and the connector via `torch.load()`,
applying it as a plain `linear(enc, weight, bias)`. Also added `--resample`
(default `"bicubic"`), which now **forces** the resize filter on the
`AutoImageProcessor` instead of trusting its `preprocessor_config.json`
verbatim — that field is confirmed stale for this checkpoint (§1); silently
trusting it would reproduce a wrong-but-plausible number. The script prints
which resample it actually used, so a mismatch is visible instead of silent.

Wrote a new `res/bert-decoder/verify_decoder.py` (didn't exist before — the
`compare_bert_vs_python.sh` this doc's step 5 used to point to only exists
in PR #4061's description text, not in this repository) that loads
`BertLMHeadModel.from_pretrained(decoder_path)`, feeds it the golden
(post-connector) encoder hidden state plus token 101 at position 0, and
compares argmax + logits cosine against `nntr_decoder_init_logits.npy`.

**Update:** `pip install torch transformers safetensors pillow torchvision`
DOES work in this environment (pypi.org is reachable through the proxy even
though huggingface.co and download.pytorch.org are not) — only the real
checkpoint itself is unreachable. So both `weight_converter.py` scripts and
both verify scripts were functionally exercised end-to-end against a
*synthetic* checkpoint built locally with `transformers` (`SiglipVisionModel`
+ `BertLMHeadModel`, 12/4 layers to match the real layer counts so the
converters' `assert len(...) == 199/114` pass, a separate raw
`torch.save`'d connector `.pt`, and both the split "vision_model."-prefix
layout and the old combined "encoder.vision_model."-prefix layout with an
embedded connector for the fallback path). All four scripts ran cleanly:
`_detect_vision_prefix`/`_detect_prefix` picked the right prefix in every
layout tested, `--connector_path` loaded correctly, `verify_encoder.py`
produced a `(1, num_patches, decoder_hidden_size)`-shaped golden with the
resample line confirming `Resampling.BICUBIC` was actually used (not
whatever the fake `preprocessor_config.json` said), and `verify_decoder.py`
loaded `BertLMHeadModel`, ran a real forward pass, and its shape-check +
cosine + argmax comparison logic all produced the expected PASS on a
self-comparison. This confirms the code paths and `transformers` API usage
are correct in general — it does **not** confirm anything about the real
checkpoint's numbers (that needs the actual weights; see Handoff). The one
environment gap hit along the way: this transformers version (5.15.1)
requires `torchvision` for `AutoImageProcessor` even with `use_fast=False`;
install it alongside the others if you hit the same `ImportError`.

### §8. Encoder parity PASSED against the real checkpoint (x86 fp32) — and a real `.safetensors`-loader bug found along the way

**2026-08-20, fourth pass, run on the actual machine with the checkpoint**
(not this analysis session's container): encoder parity was run end to end
against the real `screenai-caption-v40` checkpoint (same architecture as
`v4.0.0-S1`, real `model.safetensors` + `encoder_to_decoder.pt`) and
**passed**: cosine `0.99999071`, rel-L2 `4.31e-3` — both clear of the
0.9999 / 1e-2 thresholds, close to PR #4007's 224px number (`0.999998`).
Run with `--input-pixels golden.pixel.npy` (bypassing nntrainer's own image
resize, feeding it the exact PyTorch-preprocessed pixels) to isolate
encoder math from preprocessing during debugging; re-run without
`--input-pixels` to also confirm the C++ resize path end to end.

Getting there surfaced one real bug and two process pitfalls, all now
fixed or documented:

- **`nntrainer`'s `.safetensors` output format is broken for this custom
  encoder graph.** The exact same conversion (`weight_converter.py
  --connector_path ... --safetensors`) that produces a working `.bin` file
  produces a `.safetensors` file that loads without error, with correct
  file-level tensor data (independently verified byte-exact against the
  source `encoder_to_decoder.pt` via a standalone Python check — the
  connector's weight/bias tensors matched exactly), but gives **completely
  uncorrelated output at runtime** (cosine ≈ -0.045, i.e. noise) — with the
  identical config and weights, switching only the output format
  (`--encoder_output foo.bin` vs `foo.safetensors` + `--safetensors`) took
  cosine from -0.045 to 0.99999071. This means `nntr_causallm`'s
  `.safetensors` loader for this encoder graph does not correctly map
  tensors by name (or has some other bug) — **use `.bin` output for this
  encoder until that loader bug is separately investigated and fixed; do
  not trust `.safetensors` conversions for `Siglip2VisionEncoder` in the
  meantime.** This wasn't caught earlier because Findings §1-§7 (and the
  synthetic-checkpoint verification in §7's Update) never happened to
  exercise the `.safetensors` output path against a real checkpoint's
  actual runtime — only `.bin`.
- **The HF-side `config.json` needed for `verify_encoder.py --vision_path`
  and this repo's own `res/*/config.json` are NOT interchangeable, even
  though they're both named `config.json` and both describe the same
  model.** nntrainer's `config.json` nests architecture params under an
  `"encoder"` key (`cfg.contains("encoder") ? cfg["encoder"] : cfg`); HF's
  `SiglipVisionConfig`/`SiglipVisionModel.from_pretrained()` expects them
  flat at the top level and silently falls back to library defaults
  (`image_size=224`) for anything it can't find nested — which surfaces
  later as a confusing `Reinit due to size mismatch: ckpt torch.Size([576,
  768]) vs model torch.Size([196, 768])` from `transformers`' loader,
  not an obvious "wrong config" error. Separately, the real checkpoint's
  own top-level `config.json` (from the original `siglip2-base-patch16-384`
  HF repo) is *also* not directly usable — it's the full `SiglipConfig`
  (text+vision), with `image_size` nested under `"vision_config"` and most
  vision-tower fields relying on `SiglipVisionConfig` defaults rather than
  stating them explicitly. The fix that worked: flatten nntrainer's own
  `"encoder"`-nested config.json into one file with everything at the top
  level — this satisfies HF's loader (which just needs the fields present
  anywhere at the top) *and* still satisfies nntrainer's C++ side (whose
  fallback is exactly "no `encoder` key → read fields from the top level
  directly"), so one flat `config.json` works for both consumers instead of
  needing two.
- **Local scratch copies of `weight_converter.py`/`verify_encoder.py` left
  in ad hoc checkpoint resource directories (e.g. a `res/<checkpoint-name>/
  encoder/` set up by hand, separate from `res/siglip2-encoder/`) drift out
  of sync with fixes landed on this branch and are easy to invoke by
  accident** (`python3 weight_converter.py` from inside that directory
  silently picks up the stale local copy instead of the fixed one one
  level up in `res/siglip2-encoder/`) — this cost several debugging rounds
  in this session (`enc_to_dec_sd` `NameError`, missing `--connector_path`
  argument, both from a stale local copy, not the real bug). When setting
  up a new checkpoint's resource directory, keep only the checkpoint's own
  data files there (weights, its own `config.json`/`nntr_config.json`,
  `encoder_to_decoder.pt`, etc.) and always invoke the converter/verify
  scripts by their path under `res/siglip2-encoder/` /
  `res/bert-decoder/` directly, rather than copying the scripts alongside
  the data.

### §9. Decoder parity PASSED against the real checkpoint — full pipeline now confirmed end to end

**2026-08-20, fifth pass, same machine as §8:** decoder parity was run
against the real `best/decoder` checkpoint (`BertLMHeadModel`,
`hidden_size=512`/`num_attention_heads=8`/`intermediate_size=2048`/
`num_hidden_layers=4`, matching Findings §2 exactly) using the encoder's
just-verified `golden.encoder_hidden.npy` (`[1,576,512]`, from §8) as the
cross-attention input, token 101 at position 0. Result: **logits cosine
`1.00000000`, argmax exact match** — better than the encoder's already
excellent 0.99999071 (expected: a single decode step through 4 layers has
far less float accumulation-order drift than a 12-layer ViT forward pass).

This closes the loop this doc's Handoff section was written for: **both
the encoder and decoder halves of the 384px SigLIP2→BertDecoder pipeline
are now confirmed correct against the real `screenai-caption-v40`
checkpoint** (same architecture as `v4.0.0-S1`), using the exact
config-driven dimension plumbing from Findings §2/§3 (`decoder_hidden_size`
/ `decoder_num_heads` / `decoder_intermediate_size` in `nntr_config.json`,
`setDecoderDims()`), the fixed `weight_converter.py` scripts from
Findings §4/§6 (`--connector_path`, prefix auto-detection), and `.bin`
output per the Findings §8 `.safetensors`-loader-bug workaround (the
decoder conversion in this pass used `.bin` throughout — its
`.safetensors` path was not separately tested, so the §8 bug's scope is
still specifically confirmed only for `Siglip2VisionEncoder`, not ruled
out for `BertDecoder`).

`verify_decoder.py`'s `transformers` API usage (`BertLMHeadModel.
from_pretrained()`, the forward signature, `.logits` shape) is now
confirmed correct against a real checkpoint too — no fixes were needed
from the synthetic-checkpoint version verified in Findings §7.

Not yet done: on-device (Android/ARM) verification (`## 6. (Optional)
On-device` below) — still untested in any form, no `ndk-build`/`adb`
access has been available in any pass of this work so far.

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

This is now a sanity check, not an open question — Findings §1 already
confirmed `"bicubic"` is correct and it must NOT be changed to bilinear even
though the checkpoint's own file disagrees:

```bash
python3 -c "import json; print(json.load(open('$CKPT/siglip2-base-patch16-384/preprocessor_config.json'))['resample'])"
# prints 2 (bilinear) — this is the STALE value; ignore it, see Findings §1.
```

`res/siglip2-encoder/nntr_config.json`'s `"resample"` key is already
`"bicubic"` — leave it. `verify_encoder.py --resample` also now defaults to
`bicubic` and forces it on the `AutoImageProcessor` regardless of what
`preprocessor_config.json` says (Findings §7), so this step is now a
double-check rather than something you need to act on.

## 3. Convert weights

The real checkpoint is three separate files, not one merged
`model.safetensors` — `CKPT` below is `v4.0.0-S1/`:

> **Do NOT copy both `res/siglip2-encoder/` and `res/bert-decoder/`'s
> `config.json`/`nntr_config.json`/`generation_config.json` into the SAME
> directory.** Both resource directories use those exact filenames, and
> `--dump-encoder`/`--decoder-init-parity` each just do
> `LoadJsonFile(dir + "/config.json")` — if both get `cp`'d into one shared
> `$W`, the second `cp` silently overwrites the first's `config.json` and
> `nntr_config.json`. This actually happened in this session's first
> real-checkpoint run: `--dump-encoder` loaded the **decoder's**
> `nntr_config.json` (whose `model_file_name` points at
> `nntr_bert_decoder_fp32.bin`, ~132MB) into the **encoder's** graph (which,
> with no `image_size` in the wrong config.json, silently fell back to
> `IMG_SIZE=224`/`NUM_PATCHES=196` and needs ~345MB) — nntrainer's loader has
> no bounds check against the actual file size (`nntrainer/models/
> neuralnet.cpp`'s `load()` computes each weight's read offset purely from
> the graph's expected sizes, then `mmap`s the file and reads through it by
> pointer arithmetic), so reading past the 132MB mapping mid-graph
> segfaulted instead of throwing a catchable error. Use two separate output
> directories, `$W/encoder` and `$W/decoder`, to avoid this entirely — never
> point `--dump-encoder`/`--decoder-init-parity` at a directory that has
> both resource sets copied into it.

> Do NOT pass `--safetensors` to the encoder conversion below — Findings §8
> found nntrainer's `.safetensors` loader is broken for `Siglip2VisionEncoder`
> (loads without error but produces uncorrelated output at runtime). Plain
> `.bin` output (the default, no flag needed) is what's confirmed working.

```bash
W=/tmp/siglip2-384-check && mkdir -p "$W/encoder" "$W/decoder"

# encoder — vp prefix ("vision_model." vs "encoder.vision_model.") is now
# auto-detected (Findings §6); --connector_path loads the connector from its
# own file (Findings §4). Do NOT add --safetensors here (Findings §8).
python3 Applications/CausalLM/res/siglip2-encoder/weight_converter.py \
  --model_path "$CKPT/siglip2-base-patch16-384/model.safetensors" \
  --connector_path "$CKPT/best/encoder_to_decoder.pt" \
  --encoder_output "$W/encoder/nntr_siglip2_encoder_fp32.bin"

cp Applications/CausalLM/res/siglip2-encoder/{config.json,nntr_config.json,generation_config.json,sample.png} "$W/encoder/"

# decoder — bp/cp prefixes ("bert."/"cls." vs "decoder.bert."/"decoder.cls.")
# are also now auto-detected (Findings §6).
python3 Applications/CausalLM/res/bert-decoder/weight_converter.py \
  --model_path "$CKPT/best/decoder/model.safetensors" \
  --decoder_output "$W/decoder/nntr_bert_decoder_fp32.bin"

cp Applications/CausalLM/res/bert-decoder/{config.json,nntr_config.json,generation_config.json,tokenizer.json} "$W/decoder/"
```

Then replace `res/siglip2-encoder/nntr_siglip2_encoder_fp32.bin` and
`res/bert-decoder/nntr_bert_decoder_fp32.bin` in the repo with the freshly
converted files from `$W/encoder`/`$W/decoder` (Findings §5 — do NOT leave
any stale symlink to another workspace's 224px build in place, if one
exists on your machine).

## 4. Encoder parity

`verify_encoder.py` now supports the real split layout directly via
`--vision_path`/`--connector_path` (Findings §7); `--ckpt` still works for a
combined checkpoint if you have one.

```bash
cd "$W/encoder"
python3 <nntrainer_repo>/Applications/CausalLM/res/siglip2-encoder/verify_encoder.py \
  --vision_path "$CKPT/siglip2-base-patch16-384" \
  --connector_path "$CKPT/best/encoder_to_decoder.pt" \
  --image sample.png --out golden --resample bicubic

<nntrainer_repo>/builddir-desktop/Applications/CausalLM/nntr_causallm \
  --dump-encoder "$W/encoder" sample.png

python3 <nntrainer_repo>/Applications/CausalLM/res/siglip2-encoder/verify_encoder.py \
  --vision_path "$CKPT/siglip2-base-patch16-384" \
  --connector_path "$CKPT/best/encoder_to_decoder.pt" \
  --image sample.png --out golden --resample bicubic \
  --nntr-npy nntr_encoder_hidden.npy
```

Check `golden encoder_hidden shape` prints `(1, 576, 512)` first (576
patches, `decoder_hidden_size`=512 — NOT 256, and NOT 768; the projection
is included) — if it doesn't, check `nntr_config.json`'s `image_size` /
`decoder_hidden_size` against the checkpoint, or confirm the checkpoint
itself is really 384px. Also confirm the printed
`using resample=... (forced)` line actually says bicubic — the script
forces it, but double-check the log instead of assuming.

## 5. Decoder parity

Copy the golden from step 4 into `$W/decoder/golden.encoder_hidden.npy`
(shape `(1, 576, 512)` — see step 4) and confirm cross-attention actually
consumes the new 576-length, 512-dim encoder output.

```bash
cp "$W/encoder/golden.encoder_hidden.npy" "$W/decoder/golden.encoder_hidden.npy"

cd "$W/decoder"
<nntrainer_repo>/builddir-desktop/Applications/CausalLM/nntr_causallm \
  --decoder-init-parity "$W/decoder"

python3 <nntrainer_repo>/Applications/CausalLM/res/bert-decoder/verify_decoder.py \
  --decoder_path "$CKPT/best/decoder" \
  --golden "$W/decoder/golden.encoder_hidden.npy" \
  --nntr-npy "$W/decoder/nntr_decoder_init_logits.npy"
```

`verify_decoder.py` is new (Findings §7) — its `transformers` API usage
(constructor args, forward signature, `.logits` shape) was functionally
verified against a synthetic `BertLMHeadModel` checkpoint, but not against
the real one. If `BertLMHeadModel.from_pretrained()` or its forward
signature don't match what `best/decoder/config.json` actually declares,
expect to need small fixes; report exactly what broke rather than working
around it silently.

## 6. (Optional) On-device

Same rule as step 3 — push encoder and decoder resources to **separate**
device directories, not merged into one (see the warning under "## 3.
Convert weights").

```bash
cd Applications/CausalLM
./build_android.sh --cache && ./install_android.sh
adb push "$W/encoder"/* /data/local/tmp/nntrainer/causallm/models/siglip2-384-encoder/
adb push "$W/decoder"/* /data/local/tmp/nntrainer/causallm/models/siglip2-384-decoder/
adb shell /data/local/tmp/nntrainer/causallm/run_causallm.sh \
  --dump-encoder /data/local/tmp/nntrainer/causallm/models/siglip2-384-encoder
adb shell /data/local/tmp/nntrainer/causallm/run_causallm.sh \
  --decoder-init-parity /data/local/tmp/nntrainer/causallm/models/siglip2-384-decoder
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

## Handoff: what's left (on-device only)

**2026-08-20, fifth pass — done:** everything this section originally asked
for is now complete. Both encoder parity (Findings §8: cosine 0.99999071,
rel-L2 4.31e-3) and decoder parity (Findings §9: logits cosine 1.00000000,
argmax exact match) passed against the real checkpoint
(`screenai-caption-v40`, same architecture as `v4.0.0-S1`), run on the
machine that actually has it — not the Claude Code remote-execution
container earlier passes of this doc were written from (that container has
neither the checkpoint nor huggingface.co access; `pypi.org` is reachable
there, which is how Findings §7's synthetic-checkpoint dry run was possible
before real hardware access was available).

**What's left is on-device (Android/ARM) only** — "## 6. (Optional)
On-device" below has never been attempted in any pass. If that's needed:

1. Follow "## 6. (Optional) On-device" as written, remembering the
   Findings §8 `.safetensors` bug: convert with plain `.bin` output (no
   `--safetensors` flag) for both encoder and decoder — untested on
   `.safetensors` either way, but `.bin` is the confirmed-working path.
2. Watch for `ENABLE_FP16`-specific behavior: `Android.mk` builds with
   `-DENABLE_FP16=1`, which switches `BertDecoder`'s KV-cache storage from
   `UINT16` to native `FP16` (see the `#ifdef ENABLE_FP16` branches in
   `bert_decoder.cpp`) — a code path never exercised on the x86 desktop
   build used for Findings §8/§9, since that build doesn't define
   `ENABLE_FP16`.
3. Compare against the ARM fp16 threshold in "## Pass criteria" below
   (cosine ≥ 0.999, looser than x86 fp32's 0.9999) — some numeric drift
   from the fp16 KV-cache path is expected there.
4. Report actual PASS/FAIL numbers, the same way Findings §8/§9 did — not
   a plausible-sounding guess.
5. Commit with the existing style on this branch: `[CausalLM] ...` subject,
   a body that explains why (not just what) and cites which Findings
   section it resolves, trailers:
   ```
   Signed-off-by: SeungHui Lee <shsh1004.lee@samsung.com>
   Co-authored-by: Claude <noreply@anthropic.com>
   ```

Separately, worth a follow-up at some point but not blocking: the Findings
§8 `.safetensors` loader bug in `Siglip2VisionEncoder` (loads without error,
produces uncorrelated runtime output) should get its own root-cause fix
rather than being permanently worked around with `.bin`-only conversions.
