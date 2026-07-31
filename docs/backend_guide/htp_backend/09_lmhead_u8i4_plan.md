# lm_head on the NPU as u8i4 — Implementation Plan

Milestone: run **qwen3-0.6b end to end on the HTP backend**. The FC layers are
done; `lm_head` is the remaining matmul on the CPU.

Two decisions are already made and are not revisited here:

- **`tie_word_embeddings` is off.** lm_head has to run on the NPU, and the tied
  path shares the embedding tensor, which is read by an entirely different
  (gather) code path. An untied, separately quantised lm_head weight is the
  only shape the u8i4 kernel can consume.
- **The weight dtype is `QINT4_HTP` (u8i4).** Reasoning in §2.

---

## 1. Where lm_head is today

| | |
| :-- | :-- |
| layer | `Applications/CausalLM/layers/lm_head.cpp` (`LmHeadLayer`) |
| built by | `Applications/CausalLM/models/causal_lm.cpp:232-244` |
| shape | `weight_dim` NCHW `(1, 1, in_dim.width(), unit)` = **`[K=1024, N=151936]`** — identical in form to an FC layer |
| the matmul | `lm_head.cpp:154` — `input_step.dot(weight, hidden_step, false, false)` |
| M | `input_step_dim.height(1)` at `lm_head.cpp:143` — **always 1**. lm_head is a GEMV, prefill included (`SkipPrefill` skips it entirely). |
| engine | `causal_lm.cpp:240` — `withKey("engine", "cpu")`, hardcoded |

The tied variant (`tie_word_embedding.cpp`) is the one actually used by
qwen3-0.6b today, running Q6_K on the CPU via `dotQnK`. It is being retired for
this path, not modified.

---

## 2. Why u8i4 and not something else

lm_head is **bandwidth-bound**, not compute-bound: one activation vector of
1024 floats against a 155.6 M-parameter weight, about 0.5 MAC per byte read.
The only lever that matters is bytes moved per token.

| dtype | bytes for the lm_head weight | where it runs |
| :-- | --: | :-- |
| FP32 | 622 MB | — |
| FP16 | 311 MB | — |
| Q6_K (today) | **121.7 MiB** | CPU |
| QS4CX-FP16 | **~74 MiB** | CPU |
| QINT4_HTP (u8i4) | **74.2 MiB** | NPU |

u8i4 moves ~39 % fewer bytes than Q6_K *and* moves them off the CPU. FP16 on
HMX would be 4× the traffic of Q6_K — it is the wrong lever here even though it
is the obvious NPU dtype. That is the whole argument.

The cost is precision: Q6_K is ~6.56 bits with 16-element sub-blocks, while
`quantize_qint4_weight` is symmetric int4 with **one scale per output channel**
(1024 values per scale). That is a large step down and is the main risk — see §8.

### Measure the CPU int4 baseline first

Note the third row. `QS4CX-FP16` moves the *same* bytes as u8i4 and needs
nothing from this plan — no branch merge, no NPU residency, no CPU fallback,
no `engine` change. Just a re-quantise.

Being bandwidth-bound cuts both ways: if the weight is 74 MiB in DDR either
way, the engine reading it matters much less than the byte count, and u8i4's
remaining advantages over CPU int4 are power, freeing the CPU, and whatever
extra memory bandwidth the DSP can sustain. Those are real but they are not
the 39 % the Q6_K comparison suggests.

**So measure `QS4CX-FP16` before doing any of §4–§7.** It is a day of work, it
sets the bar u8i4 has to clear, and if it clears the milestone on its own the
rest of this plan does not need to happen. §10 records it as a fallback; it is
better understood as the baseline.

---

## 3. The gate: can 74 MiB stay resident?

`hexkl_mm.cpp` uploads a WH weight into NPU memory once and keeps it
(`residentIntWeight`, `hexkl_mm.cpp:676`). If the weight cannot stay resident,
every token pays `sdkl_npu_alloc` + a 74 MiB `memcpy` — which moves *more*
bytes than the Q6_K CPU path it is replacing, and the whole idea is off.

Three numbers, all in NPU memory:

| | MiB |
| :-- | --: |
| lm_head packed int4 weight (151936 × 1024 / 2) | 74.2 |
| lm_head int32 accumulator (see §4) | 37.1 |
| 28 decoder-layer u8i4 weights, if `fc_layer_dtype=QINT4_HTP` too | 210.0 |
| **total for a fully resident model** | **321.3** |

Against a current cap of `INT_WEIGHT_MAX_BYTES = 48 MiB` (`hexkl_mm.cpp:667`).
The 48 MiB figure came from an old cycling probe — many small allocations,
never freed — which is a different question from "one large block", so it is
not evidence about this case either way.

**`test/unittest/jni_htp/hexkl_pin_probe.c` answers it.** Four modes:

```bash
export HEXKL_LIB_SUBDIR=6.4.0.2/armv8_android26   # beta2 lib layout
ndk-build -C test/unittest/jni_htp NDK_PROJECT_PATH=. \
  APP_BUILD_SCRIPT=Android.mk NDK_APPLICATION_MK=Application.mk \
  hexkl_pin_probe -j$(nproc)

adb push libs/arm64-v8a/hexkl_pin_probe /data/local/tmp/
adb push $HEXKL_SDK_ROOT/lib/6.4.0.2/armv8_android26/libsdkl.so /data/local/tmp/
adb shell "cd /data/local/tmp && chmod +x hexkl_pin_probe && \
  LD_LIBRARY_PATH=/data/local/tmp ./hexkl_pin_probe sweep"
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp \
  ./hexkl_pin_probe total"
```

- `sweep` — largest **single** block. Decides lm_head alone.
- `total` — largest **cumulative** hold. Decides whether the whole model can be
  resident, which is the question that actually governs the design.
- `after` / `before` — ordering, because the first `sdkl_npu_mm_*` call
  permanently reserves a large internal scratch.

Every mode re-runs a known-answer matmul (`X = W = 1`, every element must equal
`K`) **while the block is held**. Allocating is not the test: an earlier round
of this work found over-allocation corrupting SDKL's internal state instead of
returning an error.

### Reading the result

| outcome | what to do |
| :-- | :-- |
| ≥ 285 MiB cumulative | everything resident. Proceed, raise the cap to cover it. |
| ≥ 75 MiB cumulative | lm_head weight resident; decoder weights stream. Proceed. |
| ≥ 74 MiB single, less cumulative | proceed, but §4's fix (1) becomes mandatory, not optional. |
| < 74 MiB | **stop.** u8i4 lm_head is not viable; fall back to CPU int4 (`QS4CX-FP16`), which halves Q6_K's bytes without needing the NPU. |

The accumulator row of the table above assumes the current `Mp = 64` wrapper.
Once §4's fix (1) lands it drops from 37.1 MiB to 594 KB, so the milestones
become 74.2 MiB for lm_head alone and 285 MiB for a fully resident model —
`hexkl_pin_probe total` prints both figures, but read them against the fixed
numbers.

`sdkl_npu_init_config_t` was on this list as a possible way to shrink SDKL's
internal scratch reservation. It is **an empty struct** in beta2 ("reserved for
future use"), so there is nothing to tune — passing `NULL` is the only option
and the budget is whatever the probe reports.

---

## 4. Two problems in `shgemm_u8i4_i32` that lm_head exposes

Both come from `M` being padded to the 64-row tile while lm_head's real `M` is 1.

**(a) A 37 MiB accumulator for one row of output.**
`hexkl_mm.cpp` computes `c_bytes = Mp * N * sizeof(int32_t)` with `Mp = 64`, so
`N = 151936` gives 37.1 MiB — of which `dequantizeToF32` reads one row.
Worse, the call does `std::memset(C_npu, 0, c_bytes)` first: **37 MiB zeroed per
token**, on the order of a millisecond of pure waste.

**(b) The scratch pool thrashes.**
`NpuScratch::get` (`hexkl_mm.cpp:735`) allocates first, then evicts others to
get under `SCRATCH_MAX_BYTES = 32 MiB`. A 37 MiB entry never fits, so:

```
token N:   FC layers insert their small accumulators (N=1024/2048/3072, ~1.6 MiB)
           lm_head inserts 37 MiB  -> evicts all of them
token N+1: first FC layer misses  -> allocates -> evicts lm_head's 37 MiB
           lm_head misses         -> allocates 37 MiB again
```

So a 37 MiB `sdkl_npu_alloc`/`free` pair happens **once per generated token**,
plus re-allocating every FC accumulator. This is a fastrpc mmap round trip, not
a pool hit.

### beta2 removed the padding requirement. The wrapper is stale, not wrong.

`hexkl_mm.cpp:695` asserts "SDKL requires M%64==0" with no citation. The
1.0.0-beta2 header contradicts it, in the doc block for this exact function:

> `sdkl_npu_mm_u8i4_i32` — This kernel is optimized for int32 matmul on the
> Hexagon NPU. **It accepts arbitrary (unaligned) dimensions and handles X
> layout conversion and output padding internally.**

Neither `M % 64` nor `N % 32` is a precondition. The kernel pads for itself.

**That sentence does not exist in beta1.** beta1's doc block for the same
function says only the usual "assumes A/X row-major, W in WH" plus a note that
buffers should come from `sdkl_npu_alloc()` — which is *address* alignment, not
dimension alignment. So `Mp = 64` was very likely correct against beta1, which
is what the "verified on V79/V81" comment at `hexkl_mm.cpp:299` is recording.
The wrapper is not wrong-headed; it is one SDK revision behind.

### Measured on device — `n_row` yes, `n_col` no

`hexkl_gemv_probe small`, Galaxy S25 Ultra, `1_0_56_beta.2_HEXAGON_V79`:

| M | N | K | result | rows past `M` |
| --: | --: | --: | :-- | :-- |
| 64 | 32 | 32 | correct (control) | — |
| **1** | 32 | 32 | **correct** | **untouched** |
| **1** | 96 | 32 | **correct** | **untouched** |
| 1 | **100** | 32 | **wrong — 4 of 100** | untouched |
| **7** | 64 | 32 | **correct** | **untouched** |
| **33** | 64 | 64 | **correct** | **untouched** |

**Unaligned `n_row` works, and the kernel does not write past row `M-1`.** Both
halves of the question came back the good way: `A` can be passed as `M` rows and
sized to `M` rows. 37.1 MiB → 594 KB, the `memset` and the pool thrash go with
it, and traffic per token drops from 111.3 MiB to 74.8 — the same as any GEMV.

**Unaligned `n_col` does not work**, and the failure says exactly why:
`100 = 3 × 32 + 4`, the first 96 columns are correct, and only the trailing
4 are garbage (`A[96] = 20`, expected 32). A partial tile is not handled. So
beta2's "arbitrary (unaligned) dimensions" covers `n_row` and overstates
`n_col`, and the `N % 32` throws at `hexkl_mm.cpp:700` and in
`quantize_qint4_weight` are **correct — keep them**.

The rest of this section is why the doc string was not taken at face value; it
is kept because the reasoning applies the next time a beta doc string decides
something.

**The claim alone was not a measurement.** It is one Doxygen sentence in a beta
SDK, and three things kept it from being settled:

- Beta documentation runs ahead of implementation routinely. The same header
  ships `sdkl_npu_init_config_t` as an empty "reserved for future use" struct
  in both revisions.
- "Handles output padding internally" reads two ways: the kernel keeps its own
  padded buffer and copies `n_row` rows out, or it expects the *caller's*
  buffer to be padded. Under the second reading, shrinking `A` to `M` rows is a
  heap overrun.
- The version markers in `hexkl_mm.cpp` ("sdkl.h 6.4.0.1", "sdkl 6.4") are
  **Hexagon SDK** versions, not HexKL ones, so they do not pin which HexKL
  revision was tested. And since the u8i4 wrapper was written after the u8i8
  one, its `M % 64` line was plausibly copied rather than verified — meaning
  unaligned `M` may never have been tried on this kernel at all.

`test/unittest/jni_htp/hexkl_gemv_probe.c` settles it without betting on the
answer: `A` is always allocated at the padded size and poison-filled, only `M`
is passed, and the rows past `M` are read back afterwards. That gives both
answers — is the result correct, and does the kernel write past row `M-1` — in
one run, with no overrun risk if the second reading turns out to be the right
one. It also varies `N` on and off a 32 boundary, since the same sentence
covers `n_col`.

```bash
export HEXKL_LIB_SUBDIR=6.4.0.2/armv8_android26
ndk-build -C test/unittest/jni_htp NDK_PROJECT_PATH=. \
  APP_BUILD_SCRIPT=Android.mk NDK_APPLICATION_MK=Application.mk \
  hexkl_gemv_probe -j$(nproc)
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp \
  ./hexkl_gemv_probe small"     # then `lmhead` for the real shape
```

Two consequences:

- **This fix requires the beta2 migration.** Every SDKL layout call in the tree
  still uses beta1 spellings; the rename table is in
  [08](08_attention_hmx_design.md) §4. Signatures are unchanged, so it is
  mechanical, but it has to happen first.
- **It does not transfer to u8i8.** `sdkl_npu_mm_u8i8_i32` carries no such
  sentence in either revision and was not probed, so `shgemm_u8i8_i32` keeps
  its `Mp = 64` until someone measures it.

So `shgemm_u8i4_i32`, once on beta2, passes `n_row = M` and sizes its buffers
to `M`:

| | now (`Mp = 64`) | measured (`M = 1`) |
| :-- | --: | --: |
| X buffer | 64 KiB | 1 KiB |
| **C accumulator** | **37.1 MiB** | **594 KiB** |
| `memset` | 37.1 MiB | 594 KiB |
| pool thrash (b) | every token | gone |
| **traffic per token** | **111.3 MiB** | **74.8 MiB** |

`A` is row-major `[n_row, n_col]` and the padding is internal, so the 64-row
accumulator stays in VTCM / the HMX accumulator and only the real rows reach
DDR — which is what the probe's untouched poison rows confirm. That puts HMX at
the same 74.8 MiB as an HVX GEMV or the CPU int4 baseline: **no 1.5× penalty,
and no reason to ask for an HVX GEMV entry point.**

None of this touches the FC layers: `Mp = 32` and `N ≤ 3072` put their
accumulators at 393 KB either way. `N = 151936` created the entire problem.

Fixes, in order:

1. **Pass `M` instead of `Mp`** and size X and C to `M`. Measured good above;
   this alone removes (a) and (b) completely.
2. **Drop the `memset`.** At `M = 1` it is 594 KB rather than 37 MiB, so it
   stops mattering much, but the kernel overwrites `A` and it is redundant.
3. **Leave the `N % 32` throws alone.** `hexkl_mm.cpp:700` and
   `quantize_qint4_weight` are correct: the probe's `N = 100` case returned
   garbage in exactly the trailing partial tile.
4. **Chunking `N`** is no longer needed — it was the fallback for (1) failing.

### Why QNN shows no padding

A QNN deployment of the same layer has no padding at all. It is worth being
precise about why, because it is the same reason `sdkl_npu_mm_u8i4_i32` does
not either.

HMX is a tile engine in both cases; the question is only **where the padding
sits**. QNN is a graph compiler — it knows at `finalize()` that the op is
`[1, 1024] × [1024, 151936]`, declares a `1 × 151936` output, and keeps the
tile padding inside the generated kernel. `sdkl_npu_mm_u8i4_i32` does the same
thing behind a library call ("handles ... output padding internally"). **Our
wrapper is the only place that hoisted the padding out to the caller**, and it
did so on an assumption the header does not support.

A secondary point that still holds: a GEMV has no activation reuse, which is the
premise of a matrix engine, so a compiler is free to map `M = 1` onto HVX
instead. Whether QNN does is unverified and no longer decision-relevant.

What follows:

- Neither engine is compute-limited here. lm_head is 155.6 MMAC — 10.0 GMAC even
  if all 64 padded rows were computed — and both HMX and HVX clear that in well
  under a millisecond while the 74 MiB read takes 2.5–3.7 ms. **The wasted MACs
  cost no time**, and once the padded rows stop reaching DDR the wasted writes
  cost nothing either.
- So there is no engine question left for lm_head. Fix (1) above and HMX lands
  at the same traffic as any GEMV would.

---

## 5. Model preparation

### 5.1 Untie the converter

`Applications/CausalLM/res/qwen3/qwen3-0.6b/weight_converter.py:105-108`:

```python
if not tie_word_embeddings:
    save_weight(params["lm_head.weight"], transpose=True)
```

`tie_word_embeddings` comes from the HF config (`get_tie_word_embeddings`,
line 30), and qwen3-0.6b sets it **true** — so today nothing is written. Add an
override so the flag can be forced off, and materialise the weight:

- add `--untie_lm_head` (or `--tie_word_embeddings {auto,true,false}`)
- when forcing untied, use `params["lm_head.weight"]` if the state dict has it
  (transformers usually keeps the tied parameter registered), and otherwise
  fall back to `params["model.embed_tokens.weight"]`. They are the same tensor;
  the point is not to crash when only one name is present.
- the same override belongs in `collect_qwen3_for_nntrainer` (line 209) for the
  safetensors path.

`transpose=True` is what produces the `[K=1024, N=151936]` that
`lm_head.cpp` expects.

### 5.2 Quantise

`Applications/CausalLM/quantize.cpp` already accepts `QINT4_HTP`
(`dtype_str_map`), maps `output_of_causallm -> lmhead_dtype`
(`buildLayerDtypeMap:558`), and forces `compute_engine: "htp"` into the emitted
config when any u8i4/qint8 dtype is used (`usesQint8:186`).

```bash
./quantize --model_path <untied_fp32_model> \
           --fc_layer_dtype QINT4_HTP \
           --lmhead_dtype   QINT4_HTP \
           --output_dir     <out>
```

**Constraint: this must run where `libsdkl` exists.** `quantize_qint4_weight`
(`quantizer.cpp:507`) does not just quantise — it calls
`sdkl_cpu_rm_to_wh_i4` to bake the WH layout, under `#ifdef ENABLE_HEXKL`.
Check whether the addon ships an x86 host library:

```bash
ls $HEXKL_SDK_ROOT/lib/*/
```

If it is ARM/Android only, quantisation runs on the device, against a ~622 MB
FP32 lm_head. Budget for that.

Shape preconditions, both satisfied: `N = 151936 % 32 == 0` (4748 tiles),
`K = 1024` even.

### 5.3 Config

The emitted `nntr_config.json` must carry:

```json
"tie_word_embeddings": false,
"lmhead_dtype": "QINT4_HTP",
"compute_engine": "htp"
```

Model size grows by the new lm_head tensor: **+74.2 MiB** packed, plus 1.2 MiB
of scales and zero-point corrections.

---

## 6. Wiring the layer

One line, and it is the whole blocker:

```diff
--- a/Applications/CausalLM/models/causal_lm.cpp
@@ -238,7 +238,7 @@
     withKey("weight_dtype", LMHEAD_DTYPE),
-    withKey("engine", "cpu"),
+    withKey("engine", COMPUTE_ENGINE),
```

`engine=cpu` was pinned deliberately at the time; with a u8i4 weight it is
exactly wrong — `FloatTensor::dotQInteger` throws when the tensor got the CPU
`ComputeOps` (`float_tensor.cpp:1250`), and the error message says so.

The same edit applies to the other model families that build an lm_head with a
hardcoded engine (`gemma4_causallm.cpp:822`, `lfm2_causallm.cpp:359`) — check
each rather than assuming.

Nothing in `lm_head.cpp` itself needs to change. `input_step.dot(weight, ...)`
already routes on the weight's dtype:
`dot` → `dotQInteger` → `QINT4_HTP` → `shgemm_u8i4` → `hmx::shgemm_u8i4_i32`.

---

## 7. The CPU fallback is missing

`float_tensor.cpp:1241` is a TODO, and the u8i4 branch **throws** when the HTP
context is absent:

```cpp
// TODO: add a CPU dequant fallback for the u8i4 path (mirroring
// qint8CpuFallback) so decode / NPU-down cases do not throw.
```

The u8i8 path has `qint8CpuFallback`; u8i4 has nothing. Until this exists, a
model quantised to `QINT4_HTP` is unrunnable on any device where the NPU fails
to come up — which contradicts the "robust fallback" goal in
[01_introduction.md](01_introduction.md).

It is harder than the u8i8 version because the bytes are **WH-tiled and packed
two-per-byte**, so the fallback has to invert the WH permutation to read them.
Options:

- keep an un-baked RM copy alongside the WH bytes (costs 74 MiB of host RAM),
- or write the WH → RM inverse on the host. The DSP-side layout probe
  (`hexagon/hexkl_layout_probe.c`, probe 4) reads the RM → WH permutation off
  the hardware, which is the input this needs.

**This is a prerequisite for enabling the path by default, not a follow-up.**

---

## 8. Validation

### Accuracy — do this before any performance number

Per-channel int4 over `K = 1024` is much coarser than Q6_K's 16-element blocks,
and lm_head feeds `argmax` / top-k / top-p directly
(`causal_lm.cpp:326-346`), so quantisation error lands straight on token
selection.

1. Logit-level: same prompt, same prefix, compare the FP32 reference against
   Q6_K and against u8i4 — max abs error, and **top-1 agreement rate**, which is
   the number that matters.
2. Generation-level: fixed seed, greedy decode, compare token sequences.
3. If it degrades: the kernel takes one scale per output channel and no group
   size, so per-group int4 is not available. The realistic mitigations are to
   keep lm_head at u8i8 (`shgemm_u8i8_i32`, 148 MiB — worse than Q6_K on bytes
   but exact-ish), or to leave lm_head on the CPU and accept the milestone with
   FC-only NPU coverage.

### Performance

Report the `MmProfile` breakdown (`scan / stage / quant / npu / dequant`) for
the lm_head call specifically, and check that `stage_us` is near zero after the
first token — a non-trivial `stage_us` every token means the weight is **not**
resident and §3's gate was misread.

Two baselines, not one:

- **Q6_K `dotQnK`** — what ships today. u8i4 must beat it or there is no point.
- **`QS4CX-FP16` CPU int4** — the same 74 MiB on the CPU. This is the honest
  comparison, because it isolates "moved to the NPU" from "moved fewer bytes".
  If u8i4 does not beat it by enough to justify §4 and §7, it does not justify
  §4 and §7.

---

## 9. Order of work

| # | step | blocked by |
| --: | :-- | :-- |
| 0 | quantise to `QS4CX-FP16` and measure it — the baseline u8i4 must clear (§2) | — |
| 1 | run `hexkl_pin_probe sweep` **and** `total` | — |
| 2 | merge `origin/claude/u8i4-split/8-qkv-chain` (kernel, quantizer, dtype, `ComputeOps` seam, and the `HexKLFcCompare` / `HexKLFcE2E` / `HexKLQkvChain` tools) | 1 says FITS, 0 leaves room to win |
| 3 | migrate the tree to beta2 layout-function names ([08](08_attention_hmx_design.md) §4) | 2 |
| 3a | run `hexkl_gemv_probe small` — does the kernel accept an unaligned `n_row`? (§4) | 3 |
| 3b | if 3a passes: pass `M` instead of `Mp` in `shgemm_u8i4_i32`, size X/C to `M` | 3a |
| 4 | untie the converter, produce the FP32 untied model (§5.1) | — (can run in parallel) |
| 5 | quantise to `QINT4_HTP` (§5.2) | 2, 4 |
| 6 | `engine=cpu` → `COMPUTE_ENGINE` (§6) | 2 |
| 7 | u8i4 CPU fallback (§7) | 2 |
| 8 | accuracy validation (§8) | 5, 6 |
| 9 | performance, then enable by default | 3b, 7, 8 |

Steps 0, 1 and 4 need nothing merged and can start now. **Step 0 is the one that
can make steps 2–9 unnecessary**, so it goes first even though it is not part
of the u8i4 path.

Two items that used to sit here are resolved and are gone: reading
`sdkl_npu_init_config_t` (it is an empty struct — §3) and the `M = 1`
experiments (the header answers them — §4).

---

## 10. The CPU int4 path — baseline, and fallback if the gate fails

`QS4CX-FP16` — CPU int4, already supported by `quantize.cpp`'s `dtype_str_map`.
It halves Q6_K's bytes (about 74 MiB) without needing NPU residency at all, so
it captures the entire bandwidth win and none of the residency risk. lm_head
stays on the CPU, `engine=cpu` stays, and the qwen3-0.6b milestone is met with
the FC layers on the NPU and lm_head on an int4 CPU kernel.

It appears twice in this document on purpose. As a **fallback** it is what to do
when §3's probe says 74 MiB cannot stay resident. As a **baseline** (§2, §8, and
step 0 of §9) it is what u8i4 has to beat, and the reason to run it first: it
costs one re-quantise, and the two paths move the same number of bytes, so the
entire u8i4 argument rests on what the NPU adds *beyond* the byte count.

Either way it is a smaller result, not a failed one.
