# LFM2 MoE FFN on HTP — the quantized file, the wiring, and the E2E run

**Task doc. Self-contained for a fresh session.** Read §0 and §1 before touching
code; §1 is a read-only audit done on 2026-09-04 at `5ae9830` and it removes
three things a session would otherwise spend a day re-deriving.

Prerequisite reading, in this order:

1. `01_working_style.md` — how to work here. The ladder in §3 is the rung-2
   check for every edit below.
2. `40_moe_ffn_htp_task.md` — the three tiers (§3), the shapes and the residency
   wall (§2.4), the decode bandwidth framing (§2.5), the device recipe (§6.2).
3. `41_moe_ffn_e2e_and_perf_task.md` — the four blockers (§2), the FastRPC
   call-count arithmetic (§3), the E0-E4 / P0-P3 stages (§4, §5).

This doc does not repeat 40 or 41; it links them. Where it and 41 disagree it
says so at the point of disagreement — there is exactly one, §2.1.

---

## 0. The goal, and the two decisions already taken

**Goal:** run LFM2-8B-A1B's MoE FFN through the HexKL `u8i4` HMX kernel, leave
attention / conv / router / lm_head on the ARM CPU, and get an end-to-end
Android run with a measured performance breakdown.

Two decisions were taken by the owner on 2026-09-04. They are recorded here so
the next session does not re-open them:

**D1 — one quantized file, `--isa ARM` Q4_0, for every layer.** No qs4cx (u8i4)
weight format on disk. `41 §B5` already argued this; §1.4 below adds the
concrete reason it is not merely a preference: the `QS4CX` on-disk format that
already exists in this tree is KleidiAI's, not HexKL's, so "store the FFN in the
kernel's format" is a *new* format, not a reuse. What separates an HTP layer
from a CPU layer is the layer's runtime `engine` property, never the file.

**D2 — the device belongs to the owner, not to the implementing session.**
Stages Q and W below are host-verifiable and are the implementing session's
scope. Stages E and P need `adb` + Hexagon SDK + NDK; the implementing session
writes the recipe and the expected output, the owner runs it. `CLAUDE.md`'s
second habit applies without exception: *"verified by inspection" is not
verification*. Say which gates were not run.

---

## 1. Read-only audit, 2026-09-04 at `5ae9830`

Every line reference below was opened and read. Facts N1-N9 are new relative to
40/41 or correct them.

### 1.1 The "rest of the model as CPU ARM Q4_0" half needs no new code

**N1.** `nntr_quantize` already takes `--fc_dtype`, `--embd_dtype`,
`--lmhead_dtype`, `--isa`, `--output_format` and `--config`
(`Applications/CausalLM/quantize.cpp:647-660`). The recipe the goal asks for —
attention/conv/lm_head/embedding all Q4_0 in the ARM repack — is one existing
command today:

```bash
nntr_quantize <fp32_dir> -o <out_dir> \
  --fc_dtype Q4_0 --embd_dtype Q4_0 --lmhead_dtype Q4_0 --isa ARM
```

**N9.** Both ends of that are safe for this checkpoint: the embedding Q4_0 save
requires `width % 32 == 0` (`Applications/CausalLM/layers/embedding_layer.cpp:768-771`)
and LFM2-8B-A1B's `hidden_size` is 2048; the LM head is tied
(`TieWordEmbedding`), and `--lmhead_dtype` defaults to `--embd_dtype` at both
the quantizer (`quantize.cpp:705-706`) and the loader
(`Applications/CausalLM/models/causal_lm.cpp:89-91`), so passing both the same
value is consistent rather than redundant.

### 1.2 The MoE FFN already receives `fc_dtype` — and cannot receive anything else without a second edit

**N2.** The MoE layer's own node name is `layer{i}_ffn_down`
(`Applications/CausalLM/models/lfm2_moe/lfm2_moe_causallm.cpp:44`), and
`buildLayerDtypeMap` already maps `_ffn_down` to `fc_dtype`
(`quantize.cpp:536`). So at save time the experts are already quantized with the
FC dtype. The name is shared with the *dense* MLP's down projection for layers
below `num_dense_layers`, so a name-only option cannot separate the two;
`num_dense_layers` (read at `lfm2_moe_causallm.cpp:38`, present in
`config.json`) is what separates them.

**N3.** At load time `createMoeLayer` passes no `weight_dtype`
(`lfm2_moe_causallm.cpp:44-47`), so the expert tensors take the model-level
`model_tensor_type` through `context.getWeightDataType()`
(`lfm2_moe_layer.cpp:119` and `:126`). Every other model in this tree passes
`withKey("weight_dtype", FC_LAYER_DTYPE)` explicitly (e.g.
`models/gemma3/gemma3_causallm.cpp:137`).

⇒ **Any FFN-specific dtype must change both halves or neither.** A save-side map
entry without the matching `weight_dtype` produces a file whose FFN bytes and
the loader's tensor dtype disagree, and nothing in the load path checks that.

### 1.3 What HexKL's registry wants, and why the existing `QS4CX` is not it

**N4.** `htp_qs4cx_from_q4_0x4` (`nntrainer/tensor/htp_q4_0_convert.h`) produces
three arrays: int4 values in **int8 containers**, K×N row-major; one `w_scale`
per output channel; one **`colsum_w`** per output channel — HexKL's dequant
correction for unsigned activations.

The `DataType::QS4CX` save path that already exists writes something else:
packed nibbles (`N * ceil(K/2)` bytes) followed by `N` floats of scale, and **no
colsum** (`nntrainer/layers/layer_devel.h:414-436`). That is KleidiAI's
`qsi4cxp` input, not HexKL's. `Lfm2MoELayer::save` does not implement it at all
— its QS4CX branch throws (`lfm2_moe_layer.cpp:555`).

⇒ Storing the FFN "in the kernel's format" is rung 7 (new format, new save
branch, new load-side dtype plumbing, a new `HtpComputeOps` branch, and no CPU
fallback for those layers), not rung 2. That is D1's justification, and it is
why `41 §B5`'s "do not split-quantize" stands.

### 1.4 The one thing about the file that *can* silently produce garbage

`41 §B5` established that `--isa` picks the repack variant at save time and that
feeding a `_DEFAULT`/`_X86` (q4_0**x8**) file to `htp_qs4cx_from_q4_0x4` yields
plausible-looking garbage, not an error.

**N10 (new).** The safetensors writer already records the answer —
`metadata["nntr_q4_0_isa"] = "arm" | "x86"` at
`nntrainer/models/neuralnet.cpp:907` — and `safetensors::parseMetadata`
(`nntrainer/utils/safetensors_util.h:99`) already reads it back. But
`parseMetadata`'s only callers are `inspect()` (`safetensors_util.cpp:366`) and
`test/unittest/unittest_safetensors_quantize.cpp`. **Nothing on the load path
checks it.** The information exists, is written, is parseable, and is thrown
away.

That is a trust boundary, and `01_working_style.md` is explicit that input
validation at a trust boundary is not the kind of thing "lazy" removes. Stage Q2
closes it.

### 1.5 `engine="htp"` on a custom layer works — do not go looking for this bug

**N5.** `Lfm2MoELayer` is registered only into the **"cpu"** app context
(`lfm2_moe_causallm.cpp:61-73`), which reads like it would break
`engine="htp"` at layer creation. It does not:
`HtpContext::createLayerObject` delegates straight to
`AppContext::Global().createLayerObject` (`nntrainer/htp_context.cpp:25-35`),
for both the string and the int key. Neither 40 nor 41 says so. There is no work
here.

### 1.6 Confirmations of 41's blockers

**N6.** Both `M > 1` gates are where 41 §B3 says: the batch path at
`nntrainer/tensor/float_tensor.cpp:798`, the single-weight accel path at
`:1000`.

**N7.** `Applications/CausalLM/build_android.sh:7` hardcodes
`ANDROID_NDK=~/Desktop/workspace/android-ndk-r26d`, and `:122` invokes
`./tools/package_android.sh` with no HTP options — 41 §B2, confirmed. Separately
the repo-root `build_android.sh:28` contains a bare
`cd /home/jwon/Desktop/workspace/release/Quick.AI/nntrainer`. That line makes the
script fail on any other machine and **must not reach a PR**; it is a one-line
delete (the script already computes its own root two lines later).

**N8.** The stale "router gate width (=4) is not divisible by 32" comment that
41 §1 item 2 asks to fix lives in **two** files, not one:
`test/unittest/models/unittest_causallm_lfm2_moe.cpp:216-217` and the header of
`test/unittest/models/unittest_causallm_lfm2_moe_reference.cpp:15-17`. Both are
wrong for the same reason — `Lfm2MoELayer::save` special-cases `gate_idx` and
`expert_bias_idx` to `DataType::NONE` (`lfm2_moe_layer.cpp:511-517`).

---

## 2. Stage Q — the quantized file

### 2.1 Q1 — `--moe_dtype`, both halves

This is the "argument option" half of the goal. Under D1 its value for the
device recipe is `Q4_0`, i.e. the same as `--fc_dtype`, so **it changes nothing
about the file the device run uses**. Say that in the commit message rather than
implying a behaviour change. It earns its place for one reason: it is the switch
that lets the FFN be held at `FP32` while everything else is `Q4_0`, which is
the A/B that isolates "the HTP path is wrong" from "Q4_0 is wrong" during E3's
accuracy gate. Build it for that, and no larger.

Both halves, per N3:

1. `quantize.cpp`: `--moe_dtype <type>`, default = `fc_dtype`. Applied in
   `buildLayerDtypeMap` to `layer{i}_ffn_down` for `i >= num_dense_layers` only
   (N2) — so `buildLayerDtypeMap` needs `num_dense_layers`, read from `cfg` at
   the call site (`quantize.cpp:826`) with `cfg.value("num_dense_layers", 0)`,
   the same default `lfm2_moe_causallm.cpp:38` uses. Mirror the chosen value
   into `new_nntr_cfg["moe_layer_dtype"]` next to the existing three
   (`quantize.cpp:857-861`), and add the option to `printUsage` and to the
   file-header usage block.
2. `lfm2_moe_causallm.cpp`: read `moe_layer_dtype` from `nntr_cfg` in
   `setupParameters` (defaulting to `FC_LAYER_DTYPE`, which
   `Lfm2CausalLM::setupParameters` has already set) and pass
   `withKey("weight_dtype", MOE_LAYER_DTYPE)` in `createMoeLayer`.

**Gate (host):** `nntr_quantize --moe_dtype FP32 --fc_dtype Q4_0` on the tiny
fixture produces a file that loads and generates; `--moe_dtype Q4_0` produces a
file byte-identical to one produced without the flag. The second half is the one
that catches a wrong `num_dense_layers` boundary.

### 2.2 Q2 — refuse a wrong-ISA file instead of computing garbage

Per N10 the ISA is already written and already parseable; only the check is
missing. Smallest thing that closes it:

- Where the safetensors loader parses the header, compare `nntr_q4_0_isa`
  against the layout the running backend expects and **throw** on a mismatch,
  naming both values. Do not warn — 41 §B5 is explicit that the failure mode is
  silent garbage, and a warning in a device logcat is not a gate.
- `.bin` cannot carry the metadata, so it cannot be checked. Do not invent a
  sidecar for it: instead make Stage E's recipe use `--output_format
  safetensors` (41 §B5 recommends it for exactly this) and leave `.bin`'s
  filename suffix as the only signal it has, as today.

**Gate (host):** a new case in `test/unittest/unittest_safetensors_quantize.cpp`
— save with `--isa X86`, load on a build whose expected layout is ARM, expect
the throw with both ISA names in the message. Note for `check_count`
(`AGENTS.md`): the test count changes.

**Ponytail note.** Q2 is not scope creep bolted onto Q1: it is the validation
`01_working_style.md` names as never-simplify-away, on the exact input D1 makes
load-bearing. If it cannot be done in a small diff at the loader, say so and
leave a `ponytail:` comment naming the hole rather than half-doing it.

---

## 3. Stage W — wiring the FFN to the kernel

This is 41 §4 E1 (B1, B3, B4) plus B2. Nothing here is new design; the
disagreements between 40 and 41 are already resolved in 41's favour.

1. **B3, the `M > 1` gates** (N6). Add a predicate to `ComputeOps` — the shape
   40 §Stage 4.1 suggests, `accelerates_at_m1()` or equivalent, default `false`,
   `true` in `HtpComputeOps` — and use it at `float_tensor.cpp:798` and `:1000`.
   **Do not remove the gate for CPU/GPU** (40 §5.5). Do not change any existing
   virtual's signature (40 §5.4).
2. **B4, the prefill workspace context.** `Tensor::inheritContextTo`
   (`nntrainer/tensor/tensor.h:2056-2060`) exists for this. Four calls after the
   workspace is built, sourced from `input`, in both
   `Lfm2MoELayer::forwarding` (`lfm2_moe_layer.cpp:246-266`) and
   `incremental_forwarding` (`:444-464`). Do **not** convert the workspace to
   `requestTensor` — its dims depend on `max_assigned_tokens`, which is router
   output and unknown at finalize.
3. **B1, the engine property.** `createMoeLayer` gains
   `withKey("engine", MOE_ENGINE)`, read from `nntr_cfg` and **defaulting to
   `"cpu"`** so no existing run changes behaviour. Add alongside it the layer-id
   subset 41 §3.5 needs for E4 — a config key naming which MoE layer ids get
   `"htp"` — because the full 22-layer model does not fit (40 §2.4: 1,408
   handles against `HEXKL_MM_U8I4_MAX_WEIGHTS = 512`, 3.6 GiB of WH bytes). One
   layer is 64 handles and ~168 MiB. Ship the subset with W, not later: without
   it E4 has no runnable configuration at all.
4. **B2, the app build.** `Applications/CausalLM/build_android.sh`: an opt-in
   flag (`--htp`) that adds `-Denable-htp=true -Dhexkl-sdk-root=…
   -Dhexkl-lib-subdir=armv8_android26` to the `package_android.sh` call at
   `:122`, and stops hardcoding the NDK at `:7` (honour an existing
   `ANDROID_NDK`, fall back to a candidate list, error out by name if neither
   exists — the repo-root `build_android.sh` already does exactly this at
   `:31-47`; copy it, do not reinvent it). Delete the root script's
   `cd /home/jwon/...` (N7).
5. **E0, the host gate everything else rests on.** Add
   `TEST(Lfm2MoeDifferentialTest, Q40MatchesHFReference)` calling
   `causallm_test::runQ40DifferentialChecks(lfm2MoeModel())` to
   `test/unittest/models/unittest_causallm_lfm2_moe_reference.cpp`, and fix the
   stale comment in **both** places (N8). 41 §1 records this as already
   confirmed to pass; if it does not, stop — something changed since 2026-09-04.

**Gate (host):** E0's Q4_0 differential test and the existing FP32 one both pass
with the default (CPU) engine, and `--moe_dtype`'s two cases from Q1 still pass.
This proves the working path is intact. It **cannot** prove the HTP path works:
`ENABLE_HEXKL` does not compile on the host, so `HtpComputeOps` does not exist
there. Say that in the commit message.

```bash
meson build -Denable-transformer=true && ninja -C build
cd build && meson configure -Denable-test=true && ninja
cd build && NNTR_QUANTIZE_BIN=$PWD/Applications/CausalLM/nntr_quantize \
  ./Applications/CausalLM/unittest_causallm_models --gtest_filter='*Lfm2Moe*'
```

---

## 4. Stage E — the device run (owner's machine, per D2)

Follow 41 §4 E2-E4 unchanged. In short, with the exact recipe and both
`hexkl_addon` traps in 40 §6.2:

| | what | gate |
|---|---|---|
| E2 | baseline kernels, unchanged tree | all 38 tests across `unittest_hvx_mm_u8i4`/`_softmax`/`_attn`/`_fc` pass |
| E3 | tiny fixture E2E on HMX | the three evidence tiers below, plus SNR |
| E4 | one, then two, MoE layers of the real 8B | same evidence at real shapes, no `AEE_ENOMEMORY`, `stage_us` recorded |

The three evidence tiers for E3/E4, weakest to strongest — get at least the
first two (41 §4 E3):

1. `adb logcat -d | grep -iE "nntrainer|adsprpc"` shows `remote_handle64_open`
   on `libnntr_hvx_skel.so` and a clean close.
2. `nntr_hvx_mm_u8i4_layer_timed` behind a debug flag: non-zero
   `FC_T_DSP_TOTAL` on an FFN-shaped call (K=2048/N=3584 or K=1792/N=2048) is
   the proof HMX ran; `FC_T_ACC_COPY == 0` confirms the in-place dequant path.
3. A call counter in `HtpComputeOps` (count + accumulated M/K/N), dumped at
   exit, checked against 41 §3.1's arithmetic.

**Accuracy: SNR in dB, never max relative error** — this project's own harness
passes at `max_rel = 1908×`. Compare against Stage 3's measured **22.8 dB**
(40 §Stage 3).

**Two failure modes to expect rather than debug from scratch:**

- Wrong repack variant → plausible garbage, not an error (41 §B5). Q2 turns this
  into a throw for safetensors; for `.bin`, check the `_ARM` suffix by eye
  before every run.
- Router imbalance at prefill → per-expert `M` is data-dependent and can reach
  the whole prompt; VTCM holds the whole K×N weight double-buffered, so
  `M ≥ 1024` returns `AEE_ENOMEMORY` with no fallback (41 §3.4). Either chunk
  the token loop or handle the error — do not discover it as a crash on a long
  prompt.

**State the expected result before running it:** Tier 1 is one FastRPC call per
`dot()`, so decode is 176 calls/token at ~326 µs ≈ **57 ms/token of transport
alone**, against a ~35 ms/token DDR floor (40 §2.5). **E4 is expected to be
slower than the CPU path.** That is the plumbing milestone landing, not a bug,
and 41 §8.1 forbids optimizing during goal 1. Record the breakdown and stop.

---

## 5. Stage P — performance

Entry condition: E4's `stage_us` breakdown exists. Order and rationale are 41
§5; do not reorder them without a measurement that says to.

- **P0** instrumentation first, including a per-call-site counter (which
  projection, which expert, prefill vs decode, M). 41 §5.0's table lists what is
  already measured *and what is already rejected* — read it before proposing
  anything, so no session re-derives a rejected idea.
- **P1** call-count reduction, the dominant term: Tier 2 (batch gate/up at
  decode, 176 → 110 calls) then Tier 3 (grouped down, → 44). Tier 2 is reuse
  (40 §Stage 4); Tier 3 is new DSP code and needs a device (40 §5.2).
- **P2/P3** the u8 boundary, then HMX/HVX overlap — both prefill levers; P3 does
  nothing for decode.

Frame every number the way 41 §6 says: decode is bandwidth-bound and the honest
framing is CPU-core offload and power, not tokens/sec; the 2.2× kernel win is a
prefill number.

---

## 6. Rules

Inherits 40 §5 and 41 §8 in full. In particular: do not modify the
device-verified `hmx/*`, `hvx/*`, `nntr_hvx.idl` or
`unittest_hvx_mm_u8i{4,8}.cpp`; do not touch `lfm2_moe_layer_fsu.cpp` or
`lfm2_moe_layer_cached.cpp`; never push the Hexagon SDK's `libcdsprpc.so` to the
device. Additionally:

1. **Work the ladder** (`01_working_style.md` §3, `CLAUDE.md`). Rung 1 first:
   §2.1 already applies it to `--moe_dtype` and lands on "build it, but only for
   the A/B it enables". Mark every deliberate corner cut with a `ponytail:`
   comment naming the ceiling and the upgrade path.
2. **Commits.** `git commit -s`, subject `[<component>] <description>`
   (`[HTP]` backend, `[CausalLM]` layer/app/quantizer, `[test]` test-only), body
   in this branch's established style — what changed and *why*, with `file:line`
   for anything a reviewer would otherwise have to hunt for, and any trap named
   so the next session does not re-find it. Trailers, in this order:

   ```
   Co-authored-by: Claude <noreply@anthropic.com>
   Signed-off-by: SeungHui Lee <shsh1004.lee@samsung.com>
   ```

   One commit per topic: Q1, Q2, each numbered item of §3, and E0 are separate
   commits. `clang-format-14` on changed lines only
   (`git clang-format-14 <base-sha>`).
3. **Every non-trivial change leaves one runnable check behind.** Q1 and Q2 name
   theirs above; §3's items are covered by E0.
4. **Do not report a device gate as passed unless it was run** (D2, `CLAUDE.md`).
