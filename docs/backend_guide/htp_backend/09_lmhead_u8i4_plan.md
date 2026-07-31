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
| ≥ 321 MiB cumulative | everything resident. Proceed, raise the cap to cover it. |
| ≥ 112 MiB cumulative | lm_head weight + accumulator resident; decoder weights stream. Proceed. |
| ≥ 74 MiB single, < 112 cumulative | proceed, but §4 becomes mandatory, not optional. |
| < 74 MiB | **stop.** u8i4 lm_head is not viable; fall back to CPU int4 (`QS4CX-FP16`), which halves Q6_K's bytes without needing the NPU. |

Also worth one read before running: `sed -n '355,460p' $HEXKL_SDK_ROOT/include/sdkl.h`
— `sdkl_npu_init_config_t` is new in beta2 and may let the ~400 MB internal
scratch reservation shrink, which would move every number above.

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

### The 64-row tile is a real SDKL constraint, so (a) is not a bug

The padding looked at first like caller-side caution that could simply be
removed. The evidence in this tree says otherwise:

- `hexkl_mm.cpp:276` cites **`§2, sdkl.h 6.4.0.1`** — the constraints are
  documented in the header, not inferred.
- `hexkl_mm.cpp:299` says **"verified on V79/V81"** — someone measured it.
- fp16 rounds to **32** (`(M + 31u) & ~31u`) while u8i8/u8i4 round to **64**.
  A dtype-dependent figure is a hardware tile spec — int8 processes twice the
  rows per tile pass — not a safety margin someone picked.

Confirm the exact u8i4 wording with
`grep -n -B3 -A8 "u8i4\|n_row\|align" $HEXKL_SDK_ROOT/include/sdkl.h`, but plan
on it being real.

**That settles the second experiment too.** If `M = 64` is what the kernel is
told, it computes 64 rows and writes 64 rows — rows 1..63 come out as computed
zeros from the zero-padded activation, but they are still written. So the
37 MiB of write traffic is **structural**, not a defect, and no host-side change
removes it.

Which reverses the "engine does not matter" reading:

| | read | write | total per token |
| :-- | --: | --: | --: |
| HMX u8i4, best case (memset gone) | 74.2 | 37.1 | **111.3 MiB** |
| HVX GEMV | 74.2 | 0.6 | **74.8 MiB** |
| QS4CX CPU int4 | 74.2 | 0.6 | **74.8 MiB** |

The wasted MACs really are free. The wasted *output writes* are not:
**for lm_head specifically, HMX moves about 1.5× the bytes of a GEMV.** It also
means HMX u8i4 moves more bytes than the CPU int4 baseline it is meant to beat,
so the DSP has to win that 1.5× back on bandwidth and power alone — possible,
but not a given. This raises the value of step 0 in §9.

None of this touches the FC layers: `Mp = 32` and `N ≤ 3072` put their
accumulators at 393 KB, where a 32× write amplification is irrelevant. The
problem is created entirely by `N = 151936`.

Fixes, in order of preference:

1. **Drop the `memset`.** Now *more* certain, not less: if the kernel writes all
   64 rows, pre-zeroing them is provably redundant. 37 MiB per token, for free.
   Confirm results are unchanged and remove it.
2. **Chunk `N`.** Split 151936 into e.g. 8 × 18992 (still `% 32 == 0`) so the
   accumulator is 4.6 MiB instead of 37.1. This bounds the *footprint* — it fits
   the pool, ends the thrash in (b), and gives 32 MiB back to the residency
   budget — but the total write traffic is unchanged.
3. **Ask the HexKL owner for a u8i4 GEMV entry point.** The 1.5× above cannot be
   recovered from the host, which moves this from a nice-to-have to the only
   remaining lever. `hexkl_mm.cpp:473` already names a dedicated GEMV kernel as
   what would remove the staging memcpy and the padding together.
4. **Raise `SCRATCH_MAX_BYTES`** — only after (1)–(2), and only if the probe
   says there is room.

### Why QNN shows no padding, and what that does and does not mean

A QNN deployment of the same layer reportedly has no padding at all. The
difference is not that QNN found a way around the hardware — HMX is a tile
engine either way — it is **where the padding sits**:

- QNN is a **graph compiler**. It knows at `finalize()` that the op is
  `[1, 1024] × [1024, 151936]`, declares the output tensor as `1 × 151936`
  (594 KB), and emits a loop nest specialised to `M = 1`. Any tile padding
  lives inside the generated kernel and costs the caller nothing.
- HexKL is a **precompiled library with a fixed signature**. `M % 64 == 0` is
  an API precondition, so the host wrapper pads (`hexkl_mm.cpp:705`) — and the
  caller pays for it in memory and in a `memset`.

There is very likely a second reason: **a GEMV has no activation reuse**, which
is the entire premise of a matrix engine, so a compiler is free to map `M = 1`
onto HVX instead of HMX. `sdkl_npu_mm_*` is HMX by construction and has no such
choice. (Inferred from the structure, not from reading QNN — checking whether
the op shows up as HVX or HMX in a QNN profile would confirm it.)

What follows for us:

- Neither engine is compute-limited here. lm_head is 155.6 MMAC, 10.0 GMAC even
  with the padding; both HMX and HVX clear that in well under a millisecond
  while the 74 MiB read takes 2.5–3.7 ms. **The wasted MACs cost no time.**
- What QNN actually avoids is the 37 MiB of padded output writes, and per the
  section above that is not recoverable from the host. So the gap is real, and
  an HVX u8i4 GEMV entry point is the fix — item (3) in the list above.

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
| 2 | read `sdkl_npu_init_config_t` in beta2's `sdkl.h` | — |
| 3 | confirm the 64-row tile in `sdkl.h`, drop the `memset`, and raise the GEMV request (§4) | — |
| 4 | merge `origin/claude/u8i4-split/8-qkv-chain` (kernel, quantizer, dtype, `ComputeOps` seam, and the `HexKLFcCompare` / `HexKLFcE2E` / `HexKLQkvChain` tools) | 1 says FITS, 0 leaves room to win |
| 5 | fix the accumulator size / memset / pool thrash (§4) | 3, 4 |
| 6 | untie the converter, produce the FP32 untied model (§5.1) | — (can run in parallel) |
| 7 | quantise to `QINT4_HTP` (§5.2) | 4, 6 |
| 8 | `engine=cpu` → `COMPUTE_ENGINE` (§6) | 4 |
| 9 | u8i4 CPU fallback (§7) | 4 |
| 10 | accuracy validation (§8) | 7, 8 |
| 11 | performance, then enable by default | 5, 9, 10 |

Steps 0–3 and 6 need nothing merged and can start now. **Step 0 is the one that
can make steps 4–11 unnecessary**, so it goes first even though it is not part
of the u8i4 path.

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
