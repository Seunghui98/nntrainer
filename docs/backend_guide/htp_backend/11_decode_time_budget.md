# Where a Decode Token Goes

Measured on a Galaxy S25 Ultra, HexKL `1_0_56_beta.2_HEXAGON_V79`, qwen3-0.6b
fp16 (WH-baked). Generation runs at **9.8 TPS — about 102 ms per token**
([05](05_e2e_performance_results.md)).

**39.3 ms of that is a `memcpy`.**

---

## 1. The measurement

`hexkl_mm.cpp:487`, the decode (`M == 1`) path:

```cpp
// --- Decode (M==1) weight staging ---
std::memcpy(W_transient, wh_host, w_bytes);
```

Every FC matmul copies its entire WH weight from host memory into an NPU
buffer before running. Seven weights per decoder layer, 28 layers:

| | |
| :-- | --: |
| per layer (q, k, v, o, gate, up, down) | 30 MiB |
| **per generated token** | **840 MiB** |

`hexkl_decode_probe stagecost full` times exactly that copy at the real shapes,
with a separate host buffer per weight so a token walks the same 840 MiB a real
one does rather than re-reading a cache-resident 30 MiB:

```
staging 840.0 MB per token: 39.27 ms at 22.4 GB/s
```

**38 % of a token, before any arithmetic happens.**

For scale, `lm_head` on the NPU as u8i4 — the subject of
[09](09_lmhead_u8i4_plan.md), and the thing this work started on — measures
6.88 ms, about 7 %.

---

## 2. What else is not compute

Not measured directly, but worth the same scrutiny: decode issues **196
`sdkl_npu_mm_*` calls per token** (7 weights × 28 layers). `hexkl_gemv_probe`
timed trivially small shapes — `M=7 N=64 K=32` — at **91 µs**, which at that
size is essentially FastRPC round-trip overhead rather than arithmetic.

196 × ~90 µs ≈ **18 ms, another 17 %**.

Treat that as an estimate until it is measured at FC shapes, but it points the
same way as the staging figure: **more than half of a decode token is not
compute.** llama.cpp's `dspqueue` batching (see
[08](08_attention_hmx_design.md) §7) is the pattern that addresses it.

---

## 3. Removing the 39 ms

`hexkl_pin_probe total` held **≥ 1024 MiB** of NPU memory with the kernel still
producing correct results, and never found a ceiling — it stopped at its own
64-block cap. So residency is not the constraint.

### The constraint was not the cap — it was a bypass

`hexkl_mm.cpp` had an NPU-resident decode cache (`g_wh_cache`) all along. The
decode path never reached it for a WH-baked model:

```cpp
const _FP16 *wh_host = lookupPrefillWH(B, N, K);
if (wh_host != nullptr) {
    void *W_transient = g_scratch_W.get(w_bytes);
    std::memcpy(W_transient, wh_host, w_bytes);   // every call
} else {
    // g_wh_cache — the resident path, reached only on a registry *miss*
}
```

qwen3-0.6b ships WH-baked, so `lookupPrefillWH` always hits and the resident
cache is dead code. The registry was added to avoid re-running `rm_to_wh` — and
it does — but it took a permanent full-weight `memcpy` in exchange.

**Fixed**: both sources now fill the same cache. The registry is a cheap way to
*populate* it (a copy, no `rm_to_wh`) rather than a reason to skip it. The first
call for a weight costs what the old path cost; every call after copies nothing.

The cap moves with it, from a 64 MiB constant to `whCacheMaxBytes()`:

```cpp
NNTR_HTP_WH_CACHE_MB=<mib>   // default 960, covers qwen3-0.6b fp16 (840 MiB)
                             // set 64 to restore the old behaviour
```

A weight that cannot be made resident — cap full, allocation failed, conversion
failed — falls back to the old transient staging rather than failing the call,
so overshooting the cap is safe.

The u8i4 path has the same shape of problem in `INT_WEIGHT_MAX_BYTES = 48 MiB`
(u8i4 branch), which needs the same treatment when that branch lands.

| approach | NPU held | staging per token | needs |
| :-- | --: | --: | :-- |
| today (fp16, 64 MiB cap) | 64 MiB | **39.3 ms** | — |
| **(a) fp16 resident** | 840 MiB | **0** | raise `WH_CACHE_MAX_BYTES` to ~900 MiB. **No model change.** |
| (b) u8i4 FC, not resident | 48 MiB | 9.8 ms | `fc_layer_dtype=QINT4_HTP` (220 MB / 22.4 GB/s) |
| **(c) u8i4 FC resident** | 210 MiB | **0** | (b) plus the cap |

Both (a) and (c) fit inside the measured budget with room to spare.

**(a) is a one-constant change** and needs no re-quantised model, which makes it
the cheapest thing on this page to try. Its cost is memory: the host keeps its
WH copy *and* the NPU holds one, so 840 MiB + 840 MiB ≈ **1.7 GB live at once**.
(c) is the same idea at 210 + 210 MiB and is the better end state, but it needs
the u8i4 FC path and a re-quantised model first.

An intermediate worth considering if 1.7 GB is too much: raise the cap far
enough to hold the *hot* weights rather than all of them. The cache already
evicts FIFO; the pin probe says the ceiling is high enough that the policy, not
the hardware, is what needs choosing.

---

## 4. What this does to the plan

[09](09_lmhead_u8i4_plan.md) is a plan for 6.88 ms. This page is 39.3 ms, on
the same hardware, gated on a constant rather than on a model conversion, an
accuracy risk, or a missing CPU fallback.

That does not make the lm_head work wrong — it is needed to run qwen3-0.6b
end-to-end on the NPU at all, which is the milestone. But **it should not be
first**, and its performance case was never the strong part of it: at 11.3 GB/s
effective ([09](09_lmhead_u8i4_plan.md) §8) the NPU may not beat a
multithreaded CPU int4 kernel over the same bytes.

Suggested order, given what is now measured:

| # | step | why |
| --: | :-- | :-- |
| 1 | migrate to beta2 ([08](08_attention_hmx_design.md) §4) | prerequisite for everything below |
| 2 | raise `WH_CACHE_MAX_BYTES`, measure TPS | 39.3 ms, one constant |
| 3 | measure per-call overhead at FC shapes | sizes the ~18 ms estimate above |
| 4 | u8i4 FC weights + cap | same win at a quarter of the memory |
| 5 | lm_head u8i4 ([09](09_lmhead_u8i4_plan.md)) | 6.88 ms, and the milestone |

Step 2 is worth doing even if it is later reverted: it converts the 39.3 ms
from a microbenchmark into an end-to-end TPS delta, which is the only number
that settles how much of it was really on the critical path.

---

## 5. Caveats

- 39.3 ms is a standalone `memcpy` benchmark. In the running model it competes
  with everything else for bandwidth, so treat it as a **floor**, not a ceiling.
- The pin probe allocated ≥ 1 GiB and kept a small matmul correct. It did not
  run a live model, under thermal load, alongside the host-side allocations the
  model itself makes. Holding 840 MiB in practice is a different test.
- The ~18 ms call-overhead figure is extrapolated from small-shape timings, not
  measured at FC shapes.
