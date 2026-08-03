# Handoff — HTP decode/lm_head optimisation

State of the work as of branch `claude/hexkl-mha-hmx-optimization-6ycsx0`.
Read [11](11_decode_time_budget.md) first, then [09](09_lmhead_u8i4_plan.md).

---

## 1. One-paragraph summary

The work started as "put `lm_head` on the NPU as u8i4" ([09](09_lmhead_u8i4_plan.md)).
Every gate that plan was blocked on has now been measured open on device — the
NPU holds ≥ 1 GiB, the u8i4 kernel accepts an unaligned `n_row`, and the device
has HMX fp16. But the same measurements turned up something five times larger:
**39.3 ms of a 102 ms decode token is a weight-staging `memcpy`**
([11](11_decode_time_budget.md)), caused by a bypass in `hexkl_mm.cpp` rather
than by any hardware limit. That has been fixed in code and **not yet run on a
device**. Measuring it is the next task.

---

## 2. Branches

| branch | what it is |
| :-- | :-- |
| `claude/hexkl-mha-hmx-optimization-6ycsx0` | **this work.** Based on `claude/htp-split/7-docs`. |
| `claude/htp-split/0…7-*` | the fp16 HTP backend this branch builds on |
| `claude/u8i4-split/8-qkv-chain` | **a parallel HTP implementation**, not a descendant of `htp-split/7-docs` |

> **Do not merge `u8i4-split/8-qkv-chain` casually.** The two branches added the
> HTP backend independently — their merge base is a kleidiai commit
> (`308bdd4`) that predates any HTP file — so `hexkl_mm.h`,
> `htp_compute_ops.cpp`, `quantizer.cpp/h`, both meson files and
> `test/unittest/jni_htp/Android.mk` all conflict add/add. That merge is a real
> task for someone who can build and run the result. Everything in
> [09](09_lmhead_u8i4_plan.md) from the u8i4 kernel onward depends on it.

Other sessions may be moving these branches. Fetch before assuming.

---

## 3. What is measured, and must not be re-litigated

All on a Galaxy S25 Ultra, HexKL `1_0_56_beta.2_HEXAGON_V79`.

| finding | value |
| :-- | :-- |
| NPU residency budget | `sweep` 256 MB, `total` **≥ 1024 MB** — neither found a ceiling |
| `sdkl_npu_mm_u8i4_i32` `n_row` | **no alignment needed.** M = 1, 7, 33 all correct |
| …and rows past `M-1` | **never written** (poison intact) |
| `sdkl_npu_mm_f32f16_f32` `n_row` | **alignment required.** M = 1 → `0x8000040d`, and SDKL cannot free afterwards |
| lm_head u8i4, weight resident | M=1 **6.88 ms**, M=64 **27.3 ms** → 11.3 GB/s effective |
| decode weight staging | **39.27 ms / token at 22.4 GB/s** for 840 MiB |
| HMX fp16 | `hmx_fp16_rate = 8` — present |
| VTCM | 18 MiB, 6 HVX units |

Two conclusions worth stating because they were each reached the wrong way
first:

- **Engine choice (HMX vs HVX) is not the axis for a GEMV.** Once `M` is passed
  unpadded, HMX moves exactly the bytes any GEMV would. What is left is kernel
  quality: 11.3 GB/s against a 22.4 GB/s memcpy on the same DRAM. That is worth
  raising with Qualcomm, and is not worth porting a kernel stack for.
- **`hexkl_mm.cpp`'s `M % 64` for u8i4 has no source in either SDK revision.**
  beta1's own vendor sample passes `n_row` unpadded while rounding `n_col` and
  `n_inner` to 32. It looks copied from the u8i8 wrapper.

---

## 4. The one production change, and it is untested

`hexkl_mm.cpp`, commit `e1f5bb4`. Everything else on this branch is probes and
documentation.

The decode path checked the pre-baked WH registry first and, on a hit, copied
the whole weight into transient scratch and returned — so `g_wh_cache`, the
NPU-resident cache, was only reachable on a registry *miss*, and qwen3-0.6b
ships WH-baked. Both sources now fill the same cache.

- `WH_CACHE_MAX_BYTES` → `whCacheMaxBytes()`, default 960 MiB
- `NNTR_HTP_WH_CACHE_MB` overrides it; **64 restores the old behaviour**
- failure to make a weight resident falls back to transient staging rather than
  throwing

**It has never been compiled.** There is no NDK, HexKL SDK or device in the
session it was written in. Expect to fix build errors.

---

## 5. Next task

Build for Android, then A/B on device with the same binary:

```bash
NNTR_HTP_WH_CACHE_MB=64    # old behaviour
NNTR_HTP_WH_CACHE_MB=960   # resident
```

Report generation TPS for both. Baseline is **9.8 TPS** at 32/32 tokens
([05](05_e2e_performance_results.md)). 39.3 ms is a standalone `memcpy`
benchmark; only the TPS delta says how much of it was on the critical path.

Watch memory: at fp16 the host WH copies and the NPU copies are live together,
about **1.7 GB**. If that is too much, `NNTR_HTP_WH_CACHE_MB` is a partial cap
and the cache evicts FIFO.

Then, in order: [11](11_decode_time_budget.md) §4.

---

## 6. Environment (dev machine)

```bash
export HEXKL_SDK_ROOT=~/workspace/hxkl-beta2/hexkl_addon      # 1.0.0-beta2
export HEXAGON_SDK_ROOT=~/workspace/Hexagon_SDK/6.4.0.2
export HEXKL_LIB_SUBDIR=6.4.0.2/armv8_android26
```

Device dir `/data/local/tmp/hexkl_beta2`, kept separate from the beta1 setup at
`/data/local/tmp`. Skeleton is
`$HEXKL_SDK_ROOT/lib/6.4.0.2/hexagon_toolv19_v79/libhexkl_skel.so`.

**The tree still uses beta1 SDKL symbol names.** `libnntrainer.so` therefore
links against beta1; only the standalone probes are built against beta2. The
rename table is in [08](08_attention_hmx_design.md) §4 and the migration is
step 3 in [11](11_decode_time_budget.md) §4.

Full build/deploy/read instructions for the probes: [10](10_probe_runbook.md).
It also records the traps — the `libsdkl.so` prebuilt collision, pushing
executables from `obj/` not `libs/`, `ADSP_LIBRARY_PATH`.

---

## 7. Open, in rough priority order

| | |
| :-- | :-- |
| `hexkl_mm.cpp` change | never compiled, never run |
| per-call overhead at FC shapes | ~18 ms/token is extrapolated from a 91 µs small-shape timing, not measured |
| QKV and gate/up fusion | q/k/v share an input, gate/up share an input. 196 calls/token → 112. Check whether `HexKLQkvChain` on the u8i4 branch already did this |
| `hexkl_gemv_probe` N=100 | the weight is now built on 32-padded extents per the vendor sample; **needs a re-run** before any claim about `n_col` |
| `hexkl_decode_probe` matmul path | raw `sdkl_npu_mm_f32f16_f32` outside nntrainer returns a partial result — each 1/16th of `n_col` starts correct and the correct prefix shrinks, independent of `n_row`, `n_inner` and allocation churn. Unexplained. **Irrelevant to `stagecost`**, which is why it was left. If the matmul side is wanted, call `nntrainer::hmx::shgemm_f32f16_f32` rather than reimplementing it |
| `hexkl_layout_probe` | never built. Needs the Hexagon toolchain, which is now installed. Answers the RM/AH/WH questions in [08](08_attention_hmx_design.md) §3 |
| u8i8 `n_row` alignment | unmeasured, so `shgemm_u8i8_i32` keeps `Mp = 64` |
| QS4CX-FP16 CPU baseline | never run. [09](09_lmhead_u8i4_plan.md) step 0 — decides whether NPU u8i4 lm_head is worth it at all |
| `sdkl_armv8` prebuilt | has never actually been linked by anything; `sdkl_npu_probe` and `unittest_nntrainer_htp_kernels` silently get the builddir copy. See the comment in `test/unittest/jni_htp/Android.mk` |

---

## 8. Attention / MHA

[08](08_attention_hmx_design.md) is analysis only, deliberately. It is blocked
on HVX softmax work owned by someone else. The hardware gate (`hmx_fp16_rate`)
is now known open, and beta2 added the AH-native kernels a fused chain would
need — beta1 had only the row-major `_i32` forms, so it was not expressible at
all before.
