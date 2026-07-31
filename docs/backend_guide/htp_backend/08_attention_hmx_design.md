# Attention on HMX — Design Record

Status: **analysis complete, implementation deferred.** The attention path is
entangled with HVX work owned by someone else (the softmax kernels), so this
document records what was decided and why, so the work can be picked up when
that lands. The active milestone is [lm_head on u8i4](09_lmhead_u8i4_plan.md).

---

## 1. What is being accelerated

`Applications/CausalLM/layers/mha_core.cpp` is the CPU-optimised attention
core. The decode path is `one_batch_incremental_forwarding`
(`mha_core.cpp:693`), three stages:

```
compute_kcaches(...)              // Q · Kᵀ   -> scores
softmax_triangle(...)             // softmax over the causal triangle
compute_fp16vcache_transposed(...)// P · V    -> attention output
```

`out_` is a `(1, 1, packed_triangular_len, num_heads_Q)` tensor: the scores are
**triangular-packed and head-minor**. That layout is what makes the CPU kernels
cheap and is also the single biggest obstacle to handing a stage to a GEMM, which
wants a dense `[M, K] × [K, N]`.

### The FP16 fact that changes everything

`mha_core.cpp:346` is guarded by `#if ENABLE_FP16 && defined(__ANDROID__)`, and
`package_android.sh:76` hardcodes `-Denable-fp16=true`. **On device, Q, K, V and
the scores are all `_FP16` already.** Any plan written against the FP32 reading
of this file is wrong. In particular there is no conversion to save — the FP16
operands are what the HMX fp16 kernel wants.

The KV cache is separately hardcoded to FP16/UINT16 at `mha_core.cpp:233-245`;
it is not driven by `fc_layer_dtype`. So the attention "weight" operand is
always fp16 regardless of how the FC layers are quantised.

---

## 2. Which stage is actually the easy one

The first instinct — that `P · V` is harder than `Q · Kᵀ` — is backwards.

| stage | M | N | K | N % 32 |
| :-- | :-- | :-- | :-- | :-- |
| `Q · Kᵀ` | q_len | **kv_len** | head_dim | varies with the cache position — needs padding |
| `P · V`  | q_len | **head_dim = 128** | kv_len | always aligned |

SDKL requires `N % 32 == 0`. `P · V` satisfies it for free because `N` is
`head_dim`. `Q · Kᵀ` has `N = kv_len`, which changes every token, so it needs
padding on every call. **`P · V` is the stage to do first.**

### The GQA stride problem

The cache is laid out `[kv_position][kv_head][head_dim]`. A single kv_head is
therefore *strided*, not contiguous (see `neon_impl.cpp:2431`). SDKL wants a
contiguous `[N, K]`. So the cache does **not** already match the kernel's
expectation whenever `num_heads_KV > 1` — a gather/pack step is unavoidable, and
its cost has to be counted against the GEMM's gain.

---

## 3. RM, AH and WH

Three layouts, not a chain. This was the most-repeated point of confusion, so it
is stated plainly:

```
            rm_to_ah                        rm_to_wh
   RM  ─────────────────►  AH        RM ─────────────────►  WH
   (row-major, "flat")   (activation)      (weights, HMX-tiled)
        ▲                    │
        └──── ah_to_rm ──────┘
```

- **RM** — plain row-major. What the rest of nntrainer holds.
- **AH** — *Activation*-HMX. The tiling the HMX unit wants for the **left**
  operand.
- **WH** — *Weights*-HMX. The tiling for the **right** operand. 32×32 tiles.

RM → AH and RM → WH are two **parallel** conversions from the same source.
AH is not an intermediate on the way to WH, and **there is no `ah_to_wh`
function** in any addon revision. A result that comes out of the accumulator as
AH cannot be re-used as a weight without going back through RM.

Whether an AH accumulator output can be fed straight back in as an AH
*activation* is exactly what probe (5) in
`test/unittest/jni_htp/hexagon/hexkl_layout_probe.c` measures — see that
directory's README. If it can, a score matrix could stay in VTCM between
`Q · Kᵀ` and `P · V`; if it cannot, every stage boundary costs a round trip.

---

## 4. HexKL 1.0.0-beta2

Obtained through `qpm-cli` (the public Software Center channel only carries
beta1; beta2 needs a signed Product Kit License Agreement). Relevant changes:

| change | why it matters here |
| :-- | :-- |
| **`sdkl_npu_mm_u8i8` and `sdkl_npu_mm_u8i4`** (AH-native) | **new in beta2** — beta1 shipped only the `_i32` row-major forms. A fused attention chain needs these; under beta1 it was not expressible at all. |
| `sdkl_npu_mm_u8i4_i32` now "accepts arbitrary (unaligned) dimensions and handles X layout conversion and output padding internally" | **New sentence, but not new behaviour.** beta1's own sample for this kernel already passes `n_row` unpadded while rounding `n_col` and `n_inner` up to 32, so beta2 documented what was already true. `hexkl_mm.cpp`'s `Mp = 64` looks copied from the u8i8 wrapper rather than verified. See [09](09_lmhead_u8i4_plan.md) §4. |
| `sdkl_npu_mm_f16f16_f16` | fp16 in, fp16 out. Removes *all* dtype conversion from the attention path, which is fp16 end to end on device. |
| `sdkl_npu_get_hw_info` / `sdkl_npu_hw_info_t` | new. Reports `vtcm_size`, `num_hvx_units`, `hmx_fp16_rate`. `hexkl_pin_probe.c` calls it, so that probe is beta2-only. |
| New AH converters: `sdkl_cpu_u8_rm_to_u8_ah(_inplace)`, `sdkl_cpu_f32_rm_to_f16_ah`, `sdkl_cpu_f16_ah_to_f32_rm`, non-inplace `sdkl_cpu_f16_rm_to_f16_ah` | new. Needed to feed the AH-native kernels above. |
| Header now `#include "remote.h"` itself | the include-order workaround at `hexkl_mm.cpp:17` becomes unnecessary. |
| Layout helpers renamed to `<dtype>_rm_to_<dtype>_<layout>` | beta1 spellings do not compile — see the migration table below. |
| Micro API shipped (`hexkl_micro.h`, `libhexkl_micro.a`) | makes the hand-written-DSP-kernel option real rather than theoretical. |
| `hexkl_micro_hw_init` takes **three** args | `(vtcm_base, vtcm_size, hmx_fp16_rate)`. beta1 probes pass two. |

`sdkl_npu_init_config_t` is **not** new and is not a lever: it is an empty
struct in both revisions, "reserved for future use".

### Migrating this tree from beta1 to beta2

Every SDKL layout call in the tree still uses the beta1 spelling. The
signatures are unchanged, so this is a mechanical rename:

| beta1 (what is in the tree) | beta2 | callers |
| :-- | :-- | :-- |
| `sdkl_cpu_rm_to_wh_f16_inplace` | `sdkl_cpu_f16_rm_to_f16_wh_inplace` | `hexkl_mm.cpp`, `layer_devel.h`, `sdkl_npu_probe.cpp`, `unittest_nntrainer_htp_kernels.cpp` |
| `sdkl_cpu_rm_to_wh_i8_inplace` | `sdkl_cpu_i8_rm_to_i8_wh_inplace` | `quantizer.cpp`, both htp unittests |
| `sdkl_cpu_rm_to_wh_i4` | `sdkl_cpu_i4_rm_to_i4_wh` | `quantizer.cpp` (u8i4 branch) |
| `sdkl_cpu_rm_to_ah_f16_inplace` | `sdkl_cpu_f16_rm_to_f16_ah_inplace` | — |
| `sdkl_cpu_ah_to_rm_f16_inplace` | `sdkl_cpu_f16_ah_to_f16_rm_inplace` | — |
| `sdkl_cpu_ui8i8_ah_to_i32_rm` | `sdkl_cpu_i32_ah_to_i32_rm` (+ `_inplace`) | — |
| `sdkl_cpu_ui8i4_ah_to_i32_rm` | **removed** | — |
| `sdkl_cpu_rm_to_wh_i8` | `sdkl_cpu_u8_rm_to_u8_ah` | — |

Two of those rows are more than renames:

- **`sdkl_cpu_rm_to_wh_i8` was misnamed in beta1.** Its parameters are
  `(n_inner, n_row, X_i8_cpu, Xq)` — an *activation*, not a weight — and beta2
  renames it to `sdkl_cpu_u8_rm_to_u8_ah`, i.e. it always produced AH, not WH.
  Nothing in this tree calls it (only the `_inplace` weight variant), so there
  is no latent bug, but do not reach for it expecting a WH converter.
- **The u8i4-specific AH→RM converter is gone.** beta1 had
  `sdkl_cpu_ui8i4_ah_to_i32_rm`; beta2 keeps only `sdkl_cpu_i32_ah_to_i32_rm`,
  documented as the u8i8 variant. If the AH-native `sdkl_npu_mm_u8i4` is used,
  whether that one reads its output correctly is an open question.

### The naming is about layout, not output dtype

`sdkl_npu_mm_u8i4` and `sdkl_npu_mm_u8i4_i32` both produce int32. The suffix
marks the **layout contract**, and the same split runs through the fp16
kernels:

| kernel | X and A layout | who converts |
| :-- | :-- | :-- |
| `sdkl_npu_mm_u8i4` | **AH** | the caller |
| `sdkl_npu_mm_u8i4_i32` | **row-major** | the kernel, internally |
| `sdkl_npu_mm_f16` | **AH** | the caller |
| `sdkl_npu_mm_f16f16_f16` | **row-major** | the kernel, internally |

The unsuffixed forms are the HMX-native ones — "the ideal kernel … assuming the
caller handles layout and type preparation". The suffixed forms are the
convenience wrappers.

This is the seam that matters for attention: a fused `Q·Kᵀ → softmax → P·V`
that keeps the score matrix on the DSP needs the **AH-native** kernels, since
the suffixed ones round-trip through row-major on every call. It is also why
probe (5) in `hexkl_layout_probe.c` — is the accumulator's AH the same AH that
`rm_to_ah` produces — decides whether that fusion is reachable at all.

`sdkl_npu_mm_u8i4_i32` is also the only kernel taking `size_t` dimensions
rather than `int` — true in beta1 as well, so it has been the odd one out for a
while. What is new in beta2 is the unaligned-dimension guarantee. See
[09](09_lmhead_u8i4_plan.md) §4 for what that changes.

---

## 5. Two implementation options

### Option A — macro API (`sdkl_npu_mm_*`)

Treat HexKL as a black-box GEMM. Host prepares operands, calls the kernel.

- Cheap to build; reuses everything in `hexkl_mm.cpp`.
- Every stage boundary is a host round trip: pack → convert → call → read back.
- The score matrix cannot stay on the DSP between `Q · Kᵀ` and `P · V`.

### Option B — micro API (`hexkl_micro_*`)

Write the attention kernel as a DSP-side program: load tiles into VTCM, issue
HMX ops, keep the intermediate resident.

- The only way to fuse the three stages and keep the scores in VTCM.
- Requires a Hexagon toolchain path nntrainer does not have today, and the
  softmax between the two matmuls is HVX work owned by someone else.

**Decision: A first, B only if the measured stage-boundary cost justifies it.**

---

## 6. The WH KV-cache question

A WH-baked KV cache resident in NPU memory is not possible: at the sequence
lengths of interest the cache is ~224 MB against an NPU weight budget measured
in tens of MB (see [09](09_lmhead_u8i4_plan.md) §3 — the same budget question).

The workable shape is a **WH cache in ordinary host RAM**, memcpy'd into the NPU
buffer per call. The point is that `rm_to_wh` is the expensive part (~2.7 ms for
the shapes measured) and the memcpy is not (~22 µs); keeping the *converted*
bytes around means only the cheap half is paid per token. Only the newly
appended KV entry needs converting each step.

---

## 7. What transfers from llama.cpp's `ggml-hexagon/htp`

Reviewed for adoptable infrastructure. The conclusion is narrower than it first
looks: llama.cpp writes its **own** DSP kernels, so its DSP-side machinery
(work-queue, dma-queue, hmx-queue, VTCM layout builders) has nothing to attach
to while nntrainer calls a black-box vendor GEMM.

What does transfer, all host-side:

- **GQA row-folding** — fold the `gqa_size` Q heads that share a kv_head into
  the `M` dimension of one GEMM instead of issuing `gqa_size` separate calls.
  This is the single largest structural win available and needs no DSP code.
- **Profitability gates** — llama.cpp checks shape before dispatching to the
  DSP at all. nntrainer needs the same: at short `kv_len` the pack cost exceeds
  the GEMM gain.
- **Tile planning ahead of dispatch** rather than inside the call.
- **`dspqueue` batching** — amortising FastRPC round trips across calls.
- **Per-stage instrumentation** — llama.cpp's counters are the model for the
  `MmProfile` breakdown already present in `hexkl_mm.cpp`.

---

## 8. Blocked on the HVX owner

The softmax is not ours. What is needed from that side, and nothing more:

1. A softmax entry point that consumes the **triangular-packed, head-minor**
   `out_` layout, or agreement to change that layout.
2. fp16 in / fp16 out (the on-device dtype).
3. Whether the online/streaming form is available, which decides whether the
   flash-attention shape is reachable.
4. Where the sliding-window and sink masks are applied — before or inside.
5. Whether it can run on a VTCM-resident buffer (required for Option B).

Everything else previously bundled into a single checklist — WH bake timing,
cache residency, layout conversion — belongs to the HMX/layout owner and is
tracked in this document instead.

---

## 9. Implementation order, when unblocked

1. **Seam.** `mha_core.cpp` has no `ComputeOps` seam. Add one — a *single*
   fused `attn_forward`, not three ops. Three ops would force the intermediate
   through host memory at every boundary, which is the cost the fusion exists
   to avoid.
2. **`P · V` on HMX** behind that seam, CPU path unchanged and selected by a
   shape gate.
3. **GQA row-folding** in the pack step.
4. **`Q · Kᵀ`** with `N` padded to 32.
5. Revisit Option B only with stage-boundary numbers in hand.

---

## 10. Probes in this tree

| probe | side | question |
| :-- | :-- | :-- |
| `test/unittest/jni_htp/hexkl_pin_probe.c` | host (ARM) | how much NPU memory can stay resident |
| `test/unittest/jni_htp/hexagon/hexkl_layout_probe.c` | DSP | what RM/AH/WH actually permute to, read off the hardware |

### The fp16 gate is cleared

`hmx_fp16_rate` was the precondition on everything above: zero there means the
device has no HMX fp16 and no `mm_f16`-based plan is worth measuring.
`sdkl_npu_get_hw_info` reports it, and `hexkl_gemv_probe` prints it on the way
past. Galaxy S25 Ultra, `1_0_56_beta.2_HEXAGON_V79`:

```
[HW_INFO] Hexagon Architecture Version: 35961     (0x8C79 -> ISA 0x79 = V79)
[HW_INFO] HMX FP16 Rate (ops/cycle): 8
[HW_INFO] Number of HVX Units: 6
[HW_INFO] VTCM total size (bytes): 18874368       (18 MiB)
```

**8 ops/cycle, so HMX fp16 exists.** The direction is open.

**VTCM is 18 MiB**, considerably more than the single-digit MB assumed while
sizing Option B. It is still nowhere near the 74 MiB an lm_head weight needs, so
nothing changes for §6 — but for attention it is roomy: a full set of Q, K and
score tiles for one head fits with margin, which is what a fused
`Q·Kᵀ → softmax → P·V` needs to keep the intermediate resident.

The DSP-side layout probe is still the one that answers §3 — what RM, AH and WH
actually permute to, and whether the accumulator's AH matches `rm_to_ah`'s.
