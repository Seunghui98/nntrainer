# START HERE — MHA-core on Hexagon HTP

**This branch (`htp/attention-handoff`) is a working reference, not a PR.** It is
never merged upstream. It exists so that a fresh session — a person or an agent —
can pick up the attention-on-HTP work without re-deriving a month of measurements.

Read this file first, then §"What to read next" tells you which of the others
apply to your task.

---

## 1. What this work is

`Applications/CausalLM/layers/mha_core.cpp` runs multi-head attention on the ARM
CPU. The goal is to run its two matmuls — `Q·Kᵀ` and `P·V` — plus softmax on the
Hexagon DSP (HMX for the matmuls, HVX for softmax), in **one FastRPC call per
attention layer**, and then to make that call cheaper with flash-style KV
streaming and, later, block sparsity.

Three things are already true and measured on device (Galaxy S25 Ultra, V79,
`R3CY10WM83Y`). Do not re-derive them:

| | |
| :-- | :-- |
| the DSP beats the CPU on both attention matmuls | 2.4–3.4× at decode, **43–65× at prefill** (`ref_14` §3) |
| the win is **data movement**, not HMX | HMX `mm_f16` is 1.7% of a decode call; DMA row size and DDR-vs-VTCM are the whole game (`ref_14` §5) |
| a u8i4 / u8i8 integer matmul path already ships in this tree | weight registry, WH bake, DMA ring with cross-block prefetch, HVX quant/dequant, QuRT worker pool — all device-verified (PR #4243/#4244/#4249) |

## 2. The two decisions that shape everything

**(a) One FastRPC call per layer, so the three stages must be fused.**
FastRPC costs ~404 µs fixed per call. An unfused seam (matmul → CPU softmax →
matmul) pays that three times per layer. This **inverts** `ref_08` §8 and
`ref_14` §1, which concluded fusion was unnecessary — those measurements were
taken inside a standalone DSP `main()` with no FastRPC in the timed region at
all (`ref_14` §7 says so). Softmax therefore runs on the DSP, on HVX.

**(b) The matmuls use the integer path (u8 activation × i4/i8 weight), not fp16.**
This is a later decision than `10_mha_htp_plan.md` was written under; where the
two disagree, **`11_u8_task_split.md` wins**. The reason is (a) above plus
reuse: the fp16 attention path exists only as a bench on another branch, while
the integer path is in this tree and verified. Halving (i8) or quartering (i4)
the KV bytes also cuts the dominant cost directly.

Everything else in `10_mha_htp_plan.md` — the transpose-free formulation, the
VTCM budget, the sparsity level, the FastRPC arithmetic, the verification
strategy — carries over unchanged to the integer path.

## 3. Where the work stands

| piece | state |
| :-- | :-- |
| u8i4 layer endpoint, DMA ring, cross-matmul prefetch | shipped, device-verified — PR #4243 |
| u8i8 mirror | shipped, device-verified — PR #4244 |
| quant/dequant path optimisation (async output DMA, vectorised quant, QuRT worker pool) | shipped, device-verified — PR #4249 |
| HVX f32 softmax + vector `exp` | **PR #4245, another contributor's**, contains #4243's commits plus the softmax. Not on this branch |
| KV block quantizer (task 1 of the split) | drafted on `htp/u8i8-dma-cross` at `f901e6f6d`, **two known defects — see §5** |
| everything else (tasks 2–11) | not started |

**Task 0 of `11_u8_task_split.md` is still open**: rebasing the u8i8 work onto
`pr4245` so the HVX softmax and the u8 matmul share one skel. It conflicts in
`test/htp/build.sh` (both sides add sources to `$SRCS`) — a small, real merge,
but it needs a device run afterwards, so it was not folded into this branch.

## 4. What to read next

| you are doing | read |
| :-- | :-- |
| **the flash-attention task** | `30_flash_attention_task.md` — self-contained, start there |
| **the LFM2-8B-A1B MoE FFN task** | `40_moe_ffn_htp_task.md` — self-contained, start there |
| **MoE FFN E2E bring-up / making it fast** | `41_moe_ffn_e2e_and_perf_task.md` — the follow-on to 40's Stages 1-3; read 40 first |
| **the FC projections on the HTP (conv in_proj, attention q/k/v/o)** | `50_fc_proj_on_htp.md` — the config switch, why the per-call round trip decides which projections win (N ≥ ~1,300 at K=2048), the memory budget, and the device recipe with what to report. Measured: in_proj −13, all-on +22; only block-level calls win |
| **block-level calls: the dense FFN and the conv block as one call each** | `51_block_calls_on_htp.md` — dense FFN through the MoE kernel (measured 10.4 ms/call, as predicted), and the conv block kernel `hexkl_conv_block.c` (weights resident, rows streamed twice; measured 4.9 ms/call as predicted, prefill 848 → 716 and decode 20.6 → 23.7 TPS with the poll window at 5000 us, perplexity 57.1 against the CPU path's 57.0, so the conv block is free; the dense FFN costs 4% and stays off). The `conv_block_engine` switch and the D measurement config |
| **NEXT: Gemma-4 26B-A4B on mobile (handoff)** | `54_gemma4_moe_htp_handoff.md` — goals, what to reuse from the LFM2 work, phases, risks (SwiGLU-only MoE kernel, >32 experts per layer and the unresolved split ppl gap), working style, commit rules, how to guide the user |
| **NEXT: Gemma-4 26B-A4B runs on the device but outputs noise -- find the operation (handoff)** | `56_gemma4_accuracy_handoff.md` -- state at 0dcb278, what the HF-reference test already clears (the CPU graph), the device-only suspects (HTP MoE kernel with the GeGLU epilogue at Gemma's shape, ARM attention/Q4_0 paths, the WH file), the per-layer NNTR_MOE_DIFF/SHADOW tool to build next, and the open crash, perf and memory items |
| **Gemma-4 26B-A4B noise: ROOT CAUSE FOUND on the host** | `55_gemma4_moe_htp_task.md` §10.5 -- the FP32 intermediate's MoE block order disagrees with what nntr_quantize_stream reads positionally (router norm gamma came from the router matrix's first rows, each expert's gate|up halves were consecutive gate rows), and the per-layer byte totals are identical so every size check passed. Not the device. Fixed by a header-vs-walk guard in the quantizer plus a re-ordered FP32 file; `56_gemma4_accuracy_handoff.md` §9 is the re-measure guide, and NNTR_MOE_DIFF/SHADOW is the instrument for whatever is left |
| **Gemma-4 26B-A4B Phase 0 (config, memory budget, risks)** | `55_gemma4_moe_htp_task.md` — config read (30 layers x 128 experts, top-8, GeGLU, dense MLP beside the experts), expert slot 2.871 MiB, budget per C (C >= 5 avoids the split; DSP address space caps C at 34-44; cold prefill floor 2.9-3.7 s), R1-R7 judged; all arithmetic, not measured on device |
| **expert streaming results summary (for sharing)** | `53_flash_experts_results_summary.md` — final numbers for C=8 / C=16 / resident, memory, profile tables, the reasoning for picking C, rejected ideas, how to reproduce |
| **expert weights from flash on the HTP path (cached-slim), MEASURED** | `52_flash_experts_on_htp_task.md` — §1~9 the task as set; §10 what was built on `claude/eager-keller-f91z9o` and measured: expert weights left virtual and pread from the model file into an LRU of ION slots shared by the 22 layers (one layer call needs all 32 experts, so C >= 2 per layer), ppl 62.0916 unchanged at every C, RSS flat at 0.9 GB (the old 5.2 GB was the loader's transient copy, the arena was never in RSS) with the arena at 22C x 5.25 MiB; the cost is the miss: C=8 is 1.9 GB physical, prefill 0.65 -> 1.6 s, decode 24 -> 16.8 TPS and prefill 0.61 -> 1.03 s warm (3.1 s cold, once), a miss 0.47 ms after a DSP swap call cut its FastRPC from four calls to one (section 10.13), hit rate 57%; a cached arena, extra read threads and prefill read-ahead all measured no better, so the hit rate is the lever left. C=1 (section 10.15): prefill calls split to fit the 22-slot pool, 1.0 GB physical, 12 TPS, but ppl 63.03, not 62.09 -- the split itself (C=2 with NNTR_MOE_SPLIT=22 gives the same 63.0346, =16 gives 63.49), so a row's output depends on the other experts in the call; device test MoeLayerSplitMatchesWhole (section 10.16) tells which of whole/split is wrong; C=1..8 swept in one session (section 10.17): decode hit 0% for C <= 3 (pool below the 88-expert token working set), 41% at C=4 (14.8 TPS, 1.4 GB) up to 57% at C=8 (15.9 TPS, 1.8 GB); prefill 0.85-0.97 s at every C. Prefill read-ahead re-examined (sections 10.18-10.19): the old +7 ms per call was the readers inheriting the caller's core (ThreadManager pins the main thread to cpu6); off that core the call is unslowed and 4 readers read half a layer per call, and the depth-k pipeline (NNTR_MOE_PREFETCH=<k>, section 10.21) cut C=8 prefill 892 -> 690 ms at k=2 with every read hidden; what is left is the between-call register (54 ms) and +1.5 ms per layer call from the readers' DDR traffic. With the batched swap (section 10.24) and 2 readers, C=8 prefill is 662 ms against 627 ms resident in the same session (+5.6%) at 1.83 GB; the swap's remaining cost is argument transfer, not allocation (in-place rebind, section 10.26, took only 18% off). Section 10.26: warm C=8 prefill now matches resident under the profiler, cold first prefill 2318 -> 935 ms with read-ahead k=4 and 4 readers, and the cache simulator reproduces the device's 57.2% hit rate with a Belady ceiling of 78.5% at C=8 -- decode is the gap left, and it is a hit-rate problem. Section 10.28 (defaults measured): warm 639 ms against 577 resident, cold 928; section 10.29: the decode gap (17.95 vs 24.61 TPS) is exactly the miss copy bytes, 37.6 misses x 0.40 ms a token, and no policy reaches resident at C=8 (Belady 20.7). Section 10.30 (code, unmeasured): swap arguments cut to offsets -- scales and column sums ride in the arena slot after the WH bytes and the DSP fetches them by DMA -- and the readers widen only on the batch the layer asks for next. Section 10.31 (measured): register 34 -> 7 ms and warm C=8 prefill 606 ms against 597 resident (+1.5%, within run-to-run noise); the reader-width cap lost end to end (warm 630, cold 1097 vs 988) and was removed. Section 10.32: the device's flash reads 3.0 GB/s at any stream count, and C=8's cold prefill (2.8 GB of misses) is exactly that -- 0.93 s is the floor; only a bigger C (16: cold ~0.65 s, 2.75 GB) lowers it, and the honest C=8 figure is the cold one (1.9 GB, ~1.0 s fully cold, 17.9 TPS). Section 10.33: read-ahead horizon = whatever the pool fits; C=16 is 2.75 GB, cold 718 ms, warm 0.60 s, decode 21.7 TPS. Also re-baselined today's CPU path (ppl 50.6, decode 48 TPS) against the HTP path (62.09, 24 TPS) — §10.7 |
| **"much faster than CPU" — the whole-model plan, CURRENT** | `45_whole_model_on_htp_plan.md` — why the MoE FFN alone caps at 1.2× whole-model prefill (packaging is 3.4× the matmul), what keeping the residual stream on the DSP between layers changes (2–2.5× prefill, and decode stops being 0×), the per-block kernel inventory (matmul and attention exist; five small HVX ops, MoE batching and the orchestration do not), the phases with gates, and the one gate that can kill it (DSP address space for 4.3 GB of weights) — which is why it goes first |
| making the MoE FFN fast — measured, superseded as a plan | `44_moe_ffn_bottleneck_map.md` — the per-stage breakdown as one design: matmul is 14% of the wall, FastRPC transport 34%; the L2 accuracy failure's mechanism (a spec-compliant SwiGLU approximation difference flipping one u8 level, device-confirmed) and its fix (a bit-identical NEON/HVX SwiGLU); the ranked lever list A1 → P3 → P1 and why that order. **Start here.** Detail and the raw measurement log stay in `43_moe_ffn_measured_next_levers.md` — the E2E path runs and is measurably NOT faster than CPU; per-stage device numbers, the ranked levers, and the one-lever-at-a-time protocol. **Supersedes 41 §5's ordering.** Start here for perf work |
| **the quantized file, the FFN wiring, and the E2E/perf run** | `42_moe_ffn_quantize_and_wiring_task.md` — the current task; read 40 and 41 first |
| implementing any task from the split | `11_u8_task_split.md` §1 (design) then your task in §3 |
| prompting an agent to do one | `12_prompt_kit.md` |
| the architecture, VTCM budget, sparsity, verification strategy | `10_mha_htp_plan.md` |
| why a HexKL micro function is slow, or a DMA descriptor shape | `ref_14` §5 — the measured rules |
| the RM / AH / WH layouts | `ref_08` §3 |
| branch layout, device recipe, environment gotchas | `13_htp_pr_plan.md` |
| how to work here | `01_working_style.md` |

`ref_*.md` are copies of `docs/backend_guide/htp_backend/*` from branch
`claude/hexkl-mha-hmx-optimization-6ycsx0`, brought here so this branch is
self-contained. They are historical records: where they disagree with
`10_`/`11_`, the newer document says so explicitly and wins.

## 5. Known defects in the drafted Task 1 (`f901e6f6d`, other branch)

The quantization arithmetic is correct — including the one thing most likely to
be wrong, `V`'s per-column scale computed over that block's rows only. The host
test is genuine (144 combinations, bound derived rather than hardcoded, `colsum`
independently recomputed, tail poisoning). It passes.

Two defects, both about placement rather than arithmetic:

1. **It is `hexkl_kv_quant.cpp`, and it calls `nntrainer::compute_fp16_to_fp32`
   from `fp16.h`.** That function is C++-namespaced and defined in
   libnntrainer. The consumer of this file is the DSP skel, built by
   `test/htp/build.sh` with `hexagon-clang` as **C**, with no C++ runtime and no
   libnntrainer in the link. As written the file can never enter the skel — it
   compiles only because the gtest is currently its only consumer. Fix: rename
   to `.c`, replace the fp16 decode with a self-contained `static inline` bit
   conversion, `<cmath>`/`<cstring>` → `<math.h>`/`<string.h>`, and add it to
   `test/htp/build.sh`'s `$SRCS`. The `extern "C"` guard in the header is
   already correct.
2. **It was committed onto `htp/u8i8-dma-cross`, which is PR #4244's branch.**
   New work needs its own branch created *before* the first commit.

## 6. The one process rule that has repeatedly paid off here

**Measure the breakdown before acting on a hypothesis.** The FastRPC
investigation's first hypothesis (marshalling dominates) measured 16–32%; the
real cost was a scalar accumulator copy-out at 92% of DSP-internal time, and it
was found by adding per-stage timing rather than by guessing a second time.
Every stage of this plan asks for a per-stage breakdown for that reason.
