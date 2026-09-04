# LFM2-8B-A1B MoE FFN on HTP — E2E bring-up, then performance

**Task doc. Self-contained for a fresh session.** Read §0, §1, §2, §3 before
touching code — §2 and §3 are the two things that will otherwise cost you a day
each.

Prerequisite reading, in this order:
1. `01_working_style.md` — how to work here.
2. `40_moe_ffn_htp_task.md` — the seam this builds on: the three tiers (§3), the
   shapes and the residency wall (§2.4), the decode bandwidth framing (§2.5),
   the device recipe (§6.2). **This doc does not repeat those; it links them.**
3. `34_fc_measured.md` §2-§5 — every FC number quoted here.

Where this doc and `40_moe_ffn_htp_task.md` disagree, **this one wins** and says
so explicitly at the point of disagreement (there is exactly one, §3.3).

---

## 0. What this doc is, and what it is not

`40_moe_ffn_htp_task.md` built the seam. Its Stages 1-3 are done and committed
(`3941d787c`, `d7f552ff4`, `5df5f013b`, `cd3536d14`). Stage 3 was verified on
device by **calling `get_htp_ops()->gemm_q4_0_accel_fp32()` directly from a
standalone binary** — see `cd3536d14`'s message. That is a kernel-level
verification and it is sound (SNR 22.8 dB at M=64/K=1024/N=1024, against the
harness's own 23.5 dB for direct qs4cx).

**What has never run: the model path.** No tensor in any model has ever
dispatched to `HtpComputeOps`. §2 shows there are four independent reasons why,
and that the count of MoE-FFN matmuls reaching HMX today is **zero, at every M**.

This doc covers two goals, in order:

| goal | meaning | this doc |
|---|---|---|
| **1. E2E** | LFM2 MoE FFN actually executes on HMX through the HEXKL kernel, in a real model run, provably | §4, Stages E0-E4 |
| **2. Performance** | if it is slow, make it fast — tiling / pipelining / VTCM / grouping | §5, Stages P0-P3 |

**Goal 1 is a plumbing milestone, not a performance milestone.** §3's arithmetic
says Tier 1 alone will be *slower* than the CPU path, by a lot. That is the
expected outcome, not a bug. Do not treat E4's regression as a failure and do not
start optimizing inside goal 1.

---

## 1. Verified starting state

Run on this machine (x86 host, `build/`, 2026-09-04). **These are pasted
outputs, not claims:**

```
$ cd build && ./test/unittest/unittest_nntrainer_cpu_backend --gtest_filter='*qs4cx*'
[ RUN      ] nntrainer_cpu_backend_standalone.htp_qs4cx_from_q4_0x4_accuracy
htp_qs4cx_from_q4_0x4: max_abs_err=0.18941 mean_abs_err=0.045134
[       OK ] nntrainer_cpu_backend_standalone.htp_qs4cx_from_q4_0x4_accuracy (55 ms)
[  PASSED  ] 1 test.

$ ./Applications/CausalLM/unittest_causallm_models --gtest_filter='*Lfm2Moe*'
[       OK ] Lfm2MoeDifferentialTest.FP32MatchesHFReference (385 ms)
[       OK ] LFM2Moe/Lfm2MoeTinyModelTest.GreedyGenerationSelectsArgmaxLogit/LFM2Moe_FP32
[       OK ] LFM2Moe/Lfm2MoeTinyModelTest.WeightRoundTripProducesSameLogits/LFM2Moe_FP32
[       OK ] LFM2Moe/Lfm2MoeTinyModelTest.PromptProducesExpectedLogits/LFM2Moe_FP32
[  PASSED  ] 4 tests.
```

Note **every tiny-model case is `LFM2Moe_FP32`**. There is no Q4_0 coverage for
LFM2-MoE anywhere today, which matters because the HTP path only exists on the
Q4_0 branch of `FloatTensor::dotQnK`.

**Two things were established experimentally for this doc** (edits reverted; the
repo is clean):

1. **The Q4_0 differential gate works and passes.** Adding
   `runQ40DifferentialChecks(lfm2MoeModel())` to
   `test/unittest/models/unittest_causallm_lfm2_moe_reference.cpp` passes as-is:
   `[ OK ] Lfm2MoeDifferentialTest.Q40MatchesHFReference (181 ms)`. It produced a
   real 73,568-byte Q4_0 file from the 439,648-byte FP32 fixture (5.98×, i.e.
   the experts really were quantized — they are the bulk of a MoE checkpoint).
   ⇒ **E0 below is a two-line addition, already known to pass.**
2. **The comment at `unittest_causallm_lfm2_moe.cpp:216-217` is stale.** It says
   Q4_0 is omitted because the router gate's width (=4) is not divisible by 32.
   `Lfm2MoELayer::save` (`lfm2_moe_layer.cpp:511-514`) already special-cases
   `gate_idx` and `expert_bias_idx` to `DataType::NONE`, so that reason no longer
   holds. Fix the comment when you add E0.
   *(Separately: extending the synthetic `INSTANTIATE_TEST_SUITE_P` at
   `:218` to `makeTinyQ40Fp32DataType()` fails — `PromptProducesExpectedLogits`
   expects FP32 logits at `atol=0.001`. That is the wrong gate; use the
   differential one, which carries `logits_atol_q40 = 5.0`.)*

---

## 2. Why the MoE FFN reaches HMX in zero cases today

Four independent blockers. B3 and B4 are complementary — B3 kills every `M == 1`
call and B4 kills every `M > 1` call, so between them they cover every call.

### B1 — no layer in the CausalLM app ever sets `engine`

`grep -rn 'withKey("engine"' Applications/CausalLM/` → **nothing.**

`Lfm2MoeCausalLM::createMoeLayer` (`lfm2_moe_causallm.cpp:41-49`) passes `name`,
`unit`, `num_experts`, `num_experts_per_token`, `moe_activation` and nothing
else. So `getComputeEngine(props)` (`layer_node.cpp:141-158`) returns the default
`LayerComputeEngine::CPU`, `network_graph.cpp:936` resolves the `"cpu"` context,
and the `ct_data` attached to every tensor of that layer carries **CPU** ops.

`"htp"` *is* a valid engine string — `ComputeEngineTypeInfo::EnumStr[] = {"cpu",
"gpu", "qnn", "htp"}` (`base_properties.h:816`) and `engine.cpp:71-72` registers
the context under that name. The mechanism exists and is unused.

### B2 — `ENABLE_HEXKL` is not in the device app build

`Applications/CausalLM/build_android.sh:122` calls `./tools/package_android.sh`
with **no** `-Denable-htp=true -Dhexkl-sdk-root=...`. Without those,
`registerContext("htp", ...)` is compiled out (`engine.cpp:69-73` is
`#if defined(ENABLE_HEXKL) && ENABLE_HEXKL == 1`), so `engine="htp"` cannot
resolve. Failure is loud (unregistered context), not silent.

Also: `build_android.sh:8` hardcodes `ANDROID_NDK=~/Desktop/workspace/android-ndk-r26d`,
which does not exist on this machine (`~/workspace/android-ndk-r26d` does).

### B3 — the `M > 1` gate blocks every decode call, on the Tier-1 path too

`float_tensor.cpp:1000`:

```cpp
if (o->supports_gemm_q4_0_accel_fp32() && M > 1) {
  o->gemm_q4_0_accel_fp32((void *)mdata, data, rdata, M, N, K);
} else {
  o->gemm_q4_0_fp32(M, N, K, data, K, (void *)mdata, N, rdata, N);
}
```

`40_moe_ffn_htp_task.md` §2.2 flagged the `M > 1` gate on the **batch** path
(`gemm_q4_0_batch_fp32`). **The single-weight accel path has the same gate**, and
40 does not mention it. Consequences:

- decode (`incremental_forwarding`, `total_tokens == 1` ⇒ `M == 1`): CPU.
- any prefill expert that was assigned exactly one token (`num_tokens == 1`,
  `lfm2_moe_layer.cpp:338`): CPU.

Per `34_fc_measured.md` §5, **the reason the gate exists for CPU/GPU does not
apply to HTP and the "GEMV instead of HMX at M=1" alternative is already
REJECTED**: the M=1 padding tax measured ~40 µs, not the ~778 µs the old cost
model predicted, and DSP-only cost is 113 µs at M=1 vs 156 µs at M=64 — 64× the
rows for 1.38× the cost, because the accumulator is 64 rows wide either way.
M=1 is a *fine* shape for this kernel. The gate is simply wrong for HTP.

### B4 — the prefill workspace tensors carry no `ContextData`, so `M > 1` also goes to CPU

This is the subtle one, and it fails **silently**.

Dispatch keys off the **left-hand (activation) tensor**, not the weight:
`FloatTensor::dotQnK` does `auto *o = getOps();` at `float_tensor.cpp:988`, and
`TensorBase::getOps()` (`tensor_base.cpp:17-23`) returns
`ct_data_->getComputeOps()` **or falls back to the global CPU table when
`ct_data_` is null.**

`configureRunContext` (`layer_node.cpp:943-972`) attaches `ct_data` to weights,
inputs, outputs and `requestTensor`-ed tensors. But `Lfm2MoELayer::forwarding`
builds the prefill workspace as **four locally-constructed `nntrainer::Tensor`s**
whenever `max_assigned_tokens > 1` (`lfm2_moe_layer.cpp:242-266`). Those have no
`ct_data`, and `getSharedDataTensor` inherits whatever the parent has:

| call site | LHS tensor | source | `ct_data`? | dispatches to |
|---|---|---|---|---|
| `:358` `token_input.dot(gate_up_proj, …)`, `num_tokens > 1` | `token_input` | `workspace.token_input->getSharedDataTensor` (`:342`) — local temp | **no** | CPU |
| `:358`, `num_tokens == 1` | `token_input` | `input.getSharedDataTensor` (`:339`) — `context.getInput()` | yes | HTP, but B3 blocks it (M=1) |
| `:376` `acti_out.dot(down_proj, expert_output)` (prefill) | `acti_out` | `workspace.activation_output->getSharedDataTensor` (`:356`) — local temp | **no** | CPU |

And `Tensor::checkContextCompatibility` (`tensor.h:2077-2087`) is **permissive
when either side lacks `ct_data`** — by design, for backward compatibility. So a
context-less activation meeting an HTP-tagged weight produces no error, no
warning: just CPU.

At **decode** the workspace points at `context.getTensor(decode_*_idx)`
(`:246-251`, requested at `:152-163`), which *do* carry `ct_data`. So decode needs
only B3 fixed; prefill needs B4.

**Fix for B4 is rung 2 — the helper already exists for exactly this.**
`Tensor::inheritContextTo` (`tensor.h:2056-2060`), whose docstring says it is
"used by binary/unary ops so that subsequent calls on the result dispatch to the
same vendor backend as the receiver". Four calls after the workspace is built, in
both `forwarding` and `incremental_forwarding`. Do **not** convert the workspace
to `requestTensor` — its dims depend on `max_assigned_tokens`, which is
router-output-dependent at runtime and unknown at finalize time.

### B5 — the on-disk repack variant must be x4, and nothing checks it

**This is verified, not theoretical.** `htp_qs4cx_from_q4_0x4` requires
Q4_0**x4**. The repack is chosen at **quantize/save** time, not load time, by
`nntr_quantize --isa`:

| `--isa` | on x86 host | on ARM host |
|---|---|---|
| `ARM` | q4_0**x4** ✅ | q4_0**x4** ✅ |
| `X86` | q4_0x8 ❌ | q4_0x8 ❌ |
| `DEFAULT` / omitted | **q4_0x8** ❌ | q4_0x4 ✅ |

(`x86_compute_backend.cpp` / `arm_compute_backend.cpp` `repack_q4_0`; the caller
is `lfm2_moe_layer.cpp:550` via `layer_devel.h:410`.)

Observed on this machine: `runQuantize` in `causallm_test_utils.cpp:500-501`
passes **no `--isa`**, and produced
`nntr_lfm2_moe_tiny_q40_embdfp32_DEFAULT.bin` — i.e. **x8**. The `Q4_0` dtype
does not record which variant the bytes are in, and nothing validates it, so
feeding an x8 file to the HTP path yields **garbage, not an error**.

⇒ The file for any device HTP run must be quantized `--isa ARM`. The filename
encodes it (`quantize.cpp:237-239` appends `_<isa>.bin`) — **check the suffix
before every device run.**

**One quantize run covers both consumers — do not split-quantize.** There is no
separate "HTP weight format" to produce: device CPU GEMM and
`htp_qs4cx_from_q4_0x4` both want the q4_0**x4** packing, so a single
`nntr_quantize … --isa ARM` file serves `engine=cpu` and `engine=htp` layers
alike. Which tensor goes where is decided at **runtime by the layer's
`engine` property**, never by the file. The qs4cx/WH conversion happens lazily
per weight pointer in `HtpComputeOps::get_or_register`
(`htp_compute_ops.cpp:69-98`), and the WH bake cannot be precomputed on the
host (needs the DSP's VTCM arena — see §7's chain). A request for "per-part
quantization" (FFN for HTP, the rest for CPU) is answered by the one-run
recipe in §9; the only file-level choice that matters is `--isa ARM`. For
hand-offs, prefer `--output_format safetensors`: a `.bin` cannot record which
ISA its Q4_0 bytes are in (this section), while safetensors writes
`nntr_q4_0_isa` metadata, so a wrong-architecture file is detectable before
load.

⇒ Corollary that shapes the plan: **the same file cannot be validated on host
and on device.** An x4 file run on the x86 host CPU path is wrong (x86 GEMM
expects x8). Host gates use `DEFAULT`; device gates use `ARM`.

*(If a colleague hands over a pre-quantized checkpoint: this is the one thing to
verify about it. See §7.)*

---

## 3. The number that sets expectations: FastRPC transport

`34_fc_measured.md` §2: **transport is 326 µs per FastRPC call** (M=64, i4,
one matmul; 330 µs at i8), against 156 µs of DSP work in the same call.
`x6`-grouped calls amortize it to 68 µs/mm.

Tier 1 (`40_moe_ffn_htp_task.md` §3) is **one FastRPC call per `dot()`**. Count
the calls the MoE FFN makes:

### 3.1 Decode, per token

`topk = 4` experts × 2 projections (`gate_up`, `down`) = **8 calls per MoE
layer**, × **22 MoE layers** = **176 FastRPC calls per token**.

| | transport | note |
|---|---|---|
| Tier 1 only | 176 × 326 µs ≈ **57 ms/token** | pure transport, before any compute |
| + Tier 2 (gate/up batched, 1 call for 4 experts) | 110 calls ≈ **36 ms/token** | `down` still per-expert |
| + Tier 3 (grouped `down`) | 44 calls ≈ **14 ms/token** | |
| one fused call per MoE layer | 22 calls ≈ **7 ms/token** | needs SwiGLU on DSP; does not exist |

Against `40_moe_ffn_htp_task.md` §2.5's **~35 ms/token DDR floor** for the same
weights on either processor.

**⇒ Tier 1 alone roughly doubles decode time. It is a regression, and the
arithmetic above says so before you run it.**

### 3.2 Prefill, seq 512

Average tokens per expert = `512 × 4 / 32 = 64`, so M ≈ 64 — the exact shape
`34_fc_measured.md` measured. All 32 experts are active per layer, so **64 calls
per layer**, × 22 = **1,408 calls**: ≈ **459 ms of transport** against ≈ 220 ms
of DSP compute (156 µs × 1,408). Transport is ~2× the compute it carries.

### 3.3 Where this corrects `40_moe_ffn_htp_task.md` §2.3

40 §2.3 argues that `attn_forward`'s "fuse everything into one FastRPC call"
reasoning "does not apply here because there is no chain of three matmul-shaped
ops with the same activation". **That is right about the shape and wrong about
the conclusion.** The reason to fuse is not shape symmetry with attention — it is
that Tier 1 issues 176 (decode) / 1,408 (prefill) transports per token/prefill,
and at 326 µs each that dominates everything else in this task. The SwiGLU
between the two projections is what makes fusion require new DSP code, so it
stays out of default scope — but **grouping is mandatory for any win, not an
optional Stage 5** as 40 §Stage 5 frames it.

Restated priority: **grouping (call-count reduction) > everything else in §5.**

### 3.4 VTCM does not limit grouping, but it does limit prefill M

From `hexkl_mm_u8i4_dma.c:183-194`, the VTCM layout is
`activation | weight double-buffer (sized for the widest handle) | one 8 KB
result tile`, and **each of the two weight buffers holds an entire K×N weight**.
There is no K/N blocking, and `vtcm_size` is used only as a pass/fail bound
(`:195-200`, returning `AEE_ENOMEMORY` with no fallback to a smaller tiling).

For `gate_up` (K=2048, N=3584): `k_tiles=64`, `n_tiles=112`,
`wh_bytes = 64 × 112 × 512 = 3.5 MiB`; double-buffered = **7.0 MiB** of the
~8.3 MB usable arena (`35_hmx_hvx_overlap.md` §5).

| M | `n_rblocks` | activation | total | fits ~8.3 MB? |
|---|---|---|---|---|
| 1 (decode) | 1 | 128 KiB | ≈ 7.13 MiB | yes |
| 64 (prefill avg) | 1 | 128 KiB | ≈ 7.13 MiB | yes |
| 512 | 8 | 1 MiB | ≈ 8.01 MiB | **marginal** |
| 1024 | 16 | 2 MiB | ≈ 9.0 MiB | **no — `AEE_ENOMEMORY`** |

*(Uses `HEXKL_HMX_ACTIVATION_ALIGNMENT = 2048`, consistent with a 64×32 u8 AH
tile. **Verified**: defined as `2048U` in `hexkl_addon/include/hexkl_micro.h:97`,
and the VTCM layout formula at `hexkl_mm_u8i4_dma.c:184-199` matches the table
above — the M=512 marginal / M≥1024 `AEE_ENOMEMORY` rows are confirmed numbers,
not projections.)*

Two consequences:
- **Grouping is VTCM-free.** `wb_max` is sized for the *widest* handle and there
  are still only two buffers regardless of `n_handles`. Tier 2/3 do not raise
  VTCM use.
- **Router imbalance is a real failure mode.** Per-expert M is data-dependent; one
  expert can be assigned far more than the 64-token average, and at
  `init_seq_len=512` the worst case is all 512. The layer must either cap M per
  call (chunk the token loop) or handle `AEE_ENOMEMORY` — **not** discover it as a
  crash on a long prompt.

### 3.5 Registration cost and the residency wall, in wall-clock

`34_fc_measured.md` §2: the one-time WH bake is **12.2 ms per weight**.
`HtpComputeOps::get_or_register` (`htp_compute_ops.cpp:69-98`) registers lazily
per pointer and **never releases**.

| scope | handles | bake time | WH bytes resident |
|---|---|---|---|
| tiny fixture (2 MoE layers × 4 experts × 2) | 16 | ~0.2 s | ~32 KiB |
| **1 MoE layer** of the 8B (32 experts × 2) | 64 | **~0.8 s** | **~168 MiB** |
| 2 MoE layers | 128 | ~1.6 s | ~336 MiB |
| full 8B (22 × 32 × 2) | **1,408** | **~17 s** | **~3.6 GiB** |

`HEXKL_MM_U8I4_MAX_WEIGHTS = 512` (`hexkl_mm_u8i4_dma.h:29`). So the full model
fails on both counts, exactly as `40_moe_ffn_htp_task.md` §2.4 says.

**But per-layer engine selection makes a bounded real-model run possible today**,
without Stage 6: `createMoeLayer` already receives `layer_id`, so gating
`engine="htp"` on a layer-id subset gives a genuine 8B run at real shapes with
64-128 handles. That is E4.

---

## 4. Goal 1 — E2E bring-up

Stages are gated. **Do not start a stage whose predecessor's gate is not green,
and paste complete output at each gate, not a summary.**

### E0 — host: Q4_0 coverage for LFM2-MoE (no device, no SDK)

Add the Q4_0 differential test to
`test/unittest/models/unittest_causallm_lfm2_moe_reference.cpp`:

```cpp
TEST(Lfm2MoeDifferentialTest, Q40MatchesHFReference) {
  causallm_test::runQ40DifferentialChecks(lfm2MoeModel());
}
```

and fix the stale comment at `unittest_causallm_lfm2_moe.cpp:216-217` (§1, item 2).

**Gate:** passes. Already confirmed to pass on this machine (§1). If it does not,
stop — something changed since 2026-09-04 and the rest of this plan rests on it.

```bash
cd build && NNTR_QUANTIZE_BIN=$PWD/Applications/CausalLM/nntr_quantize \
  ./Applications/CausalLM/unittest_causallm_models --gtest_filter='*Lfm2Moe*'
```

### E1 — the dispatch fixes (B3, B4) — host-verifiable as a regression only

Three edits, all small:

1. **B3:** at `float_tensor.cpp:1000`, let HTP through at `M == 1`. The smaller
   diff is the one 40 §Stage 4.1 already describes — a predicate rather than
   deleting the gate. **Do not remove the gate for CPU/GPU** (40 §5.5: their M=1
   behaviour is intentional).
2. **B4:** four `inheritContextTo` calls after the prefill workspace is built, in
   `Lfm2MoELayer::forwarding` (`lfm2_moe_layer.cpp:242-267`) and
   `incremental_forwarding` (`:440-465`). Source the context from `input`.
3. **B1:** pass `engine` through `Lfm2MoeCausalLM::createMoeLayer`
   (`lfm2_moe_causallm.cpp:41-49`). Read it from `nntr_cfg` so it is
   config-driven, not hardcoded, and **default to CPU** so no existing run
   changes behaviour.

**Gate (host):** `E0`'s Q4_0 test and the FP32 differential test both still pass
with the default (CPU) engine. This proves you did not break the working path.
**It does not and cannot prove the HTP path works** — `ENABLE_HEXKL` is not
compiled on the host, so `HtpComputeOps` does not exist here. Say that plainly in
the commit message rather than implying E1 is verified.

*Optional cross-check, only if OpenCL is already available in your build:*
`ClComputeOps` implements `supports_gemm_q4_0_accel_fp32()` too
(`cl_compute_ops.cpp:43`), so `engine="gpu"` exercises the **same** dispatch
path. If the MoE FFN reaches the GPU kernel, B1/B3/B4 are fixed
backend-agnostically. Do not add an OpenCL build just for this.

### E2 — device: baseline, unchanged

Before trusting anything built on top of the kernels, confirm the kernels still
pass. Full recipe and both `hexkl_addon` traps: `40_moe_ffn_htp_task.md` §6.2.

**Gate:** all 38 tests across `unittest_hvx_mm_u8i4`/`_softmax`/`_attn`/`_fc`
pass. (`speedup_vs_harness` printing 50×+ is expected — 40 §6.2 explains why.)

### E3 — device: tiny model E2E on HMX — **the actual goal-1 gate**

Build the CausalLM app **with** HTP (B2): `-Denable-htp=true
-Dhexkl-sdk-root=$HEXAGON_SDK_ROOT/addons/hexkl_addon
-Dhexkl-lib-subdir=armv8_android26`, and fix the NDK path in
`build_android.sh:8`. Quantize the tiny fixture **`--isa ARM`** (B5) and verify
the `_ARM.bin` suffix. Run with `engine="htp"` on the MoE layers.

**The gate must be an observable, not an inspection.** Three tiers of evidence,
weakest to strongest — get at least the first two:

1. **Transport happened:** `adb logcat -d | grep -iE "nntrainer|adsprpc"` shows
   `remote_handle64_open` on `libnntr_hvx_skel.so` and a clean
   `remote_handle64_close` (40 §6.2).
2. **The DSP did the matmul:** call `nntr_hvx_mm_u8i4_layer_timed`
   (`test/htp/nntr_hvx.idl:64`, impl `test/htp/nntr_hvx_mm_u8i4.c:205-241`)
   instead of `..._layer` behind a debug flag, and print `stage_us[]`. The stage
   enum is at `nntr_hvx_mm_u8i4.c:146-153`: `FC_T_DSP_TOTAL`, `FC_T_QUANT`,
   `FC_T_DEQUANT`, `FC_T_ACC_READ`, `FC_T_ACC_COPY`, `FC_T_DRAIN`,
   `FC_T_ACC_STRIDE`. **Non-zero `FC_T_DSP_TOTAL` on a MoE-FFN-shaped call
   (K=2048/N=3584 or K=1792/N=2048) is the proof that HMX ran.** Also confirm
   `FC_T_ACC_COPY == 0` — that is the in-place dequant path being live.
3. **A call counter** in `HtpComputeOps` (count + accumulated M/K/N), dumped at
   exit, cross-checked against §3.1's expected count: for a 1-token decode over
   2 tiny MoE layers at topk=2, expect `2 × 2 × 2 = 8` calls.

**Accuracy gate:** SNR of the HTP-engine logits against the **CPU-engine logits
from the same `_ARM.bin` file**, and separately against the FP32 reference. Use
**SNR in dB, never max relative error** — 40 §6.2 and `cd3536d14` both explain
why (this project's own harness passes at `max_rel = 1908×`). Compare against
Stage 3's measured **22.8 dB**; a materially worse number means the model path
introduced something the direct-call test did not.

### E4 — device: bounded 8B run at real shapes

Only after E3. Enable `engine="htp"` on **one** MoE layer of LFM2-8B-A1B
(§3.5: 64 handles, ~168 MiB, ~0.8 s of bake). Then two, if that holds.

**Gates:**
- the same three evidence tiers as E3, at the real shapes;
- `AEE_ENOMEMORY` does **not** appear — if it does, you found either the
  residency wall early or §3.4's prefill-M problem; record which, with the shape;
- **record the per-call `stage_us` breakdown.** This is the input to all of §5.
  Do not skip it: `01_working_style.md` and `CLAUDE.md` both name "measure the
  breakdown before acting on a hypothesis" as the habit this project paid for
  (the 16-32% guess vs the real 92%).

**Do not attempt the full 22-layer 8B run.** §3.5 says it needs 1,408 handles
against a 512 table and 3.6 GiB against the DSP heap. That is
`40_moe_ffn_htp_task.md` Stage 6 and it is out of scope here.

---

## 5. Goal 2 — performance

**Entry condition: E4's `stage_us` breakdown exists.** Everything below is
ordered by the measured facts in `34_fc_measured.md` / `35_hmx_hvx_overlap.md` /
`37_t2_pipelining_design.md`. Read §5.0 before proposing anything.

### 5.0 Already measured — do not re-derive, and do not re-open

| item | status | source |
|---|---|---|
| in-place tile dequant (killed a 52.8 µs/readout copy) | **built, verified**; `acc_copy = 0` | 34 §4A |
| weight residency + one-time WH bake | **built**; 12.2 ms once per weight | 34 §4B |
| cross-matmul DMA prefetch, 2 VTCM weight buffers | **built, verified**: 75 → 55 µs/mm, drain 35 → 12.3 µs | 34 §4C |
| vectorized quant + HVX worker pool | **built**: 22 µs for 64 rows | 34 §4D |
| grouped FastRPC (x3/x6) | **built**: transport 326 → 68 µs/mm | 34 §4E |
| poll-QoS + ION buffers | **built**: 44.6 GB/s at 52 MB | 34 §4F |
| GEMV instead of HMX at M=1 | **REJECTED** — padding tax ~40 µs, not 778; decode is transport-bound | 34 §5.1 |
| per-tile (not chunked) HVX jobs | **REJECTED** — 0.55 µs/tile vs multi-µs fork/join | 35 §5, 37 §3 |
| `s_band` in VTCM | **REJECTED on device** — prefill softmax +30% from VTCM bank contention | 35 §6 |
| ggml-style dedicated HMX thread | **REJECTED for now** — HMX lock thread-affinity unknown | 35 §4a |
| the old 2-parameter cost model (0.1 µs/mm) | **INVALIDATED for FC** — real is 0.019 µs/mm | 34 §5.1 |

Two structural facts you cannot design around:
- **One accumulator** ⇒ any pipeline is 2-stage, not deeper; HMX cannot start
  tile *i+1* until `acc_read(i)` drains (35 §4b). A second VTCM result buffer is
  the prerequisite for tile pipelining, and `hexkl_acc_layout_get` currently
  ramp-probes at **one** `result_off` (`hexkl_mm_u8i4_dma.c:207-208`) — with two
  buffers it must probe both **or assert they agree, failing loudly**.
- **Weight buffers hold the whole K×N weight** (§3.4). "VTCM tiling" in the sense
  of adaptive tile sizes **does not exist in this file** and adding it buys
  occupancy, not throughput (there is nothing to prefetch intra-matmul; the
  weight is fully resident before the first micro-mm).

### P0 — instrumentation, before any optimization

`37_t2_pipelining_design.md` §4 is unambiguous and applies here: **once stages
overlap, `quant + dequant + acc_read + …` exceeds `dsp_total`, and the
"remainder = micro-mm" arithmetic in `tools/htp_fc_report.py` breaks silently —
it prints plausible numbers instead of failing.** Land per-lane busy totals
(`hmx_busy_us`, `hvx_busy_us`) and a `pipelined` flag first.

Add to that, MoE-specific: **a per-call-site count** (which projection, which
expert, prefill vs decode, M). §3's arithmetic is only as good as the actual call
count, and the router makes that data-dependent.

**Gate:** with overlap off, per-lane sums equal `dsp_total`; call counts match
§3.1/§3.2's predictions for a known prompt.

### P1 — call-count reduction (the dominant term, §3)

In ladder order, cheapest first:

1. **Tier 2 — batch gate/up at decode.** Already specified in
   `40_moe_ffn_htp_task.md` §Stage 4 (all reuse: `HtpComputeOps::gemm_q4_0_batch_fp32`
   mirroring `ClComputeOps`'s, `n_handles = matAdata.size()`, one
   `mm_u8i4_layer` call, split `out_cat` back per handle in call order). VTCM-free
   (§3.4). Decode: 176 → 110 calls/token.
2. **Tier 3 — grouped `down` at decode, and both projections at prefill.** N
   independent (activation, weight) pairs sharing K, one call. `40` §Stage 5
   flags this as optional; **§3 says it is not** — it is the difference between
   14 ms and 36 ms of decode transport, and at prefill it is the *only* grouping
   that applies (every expert has its own token subset, so the shared-activation
   API cannot express prefill at all — 40 §2.3). This is new DSP code, not a
   parameter reshuffle. **Do not attempt it without a device** (40 §5.2).
3. **Fused MoE-layer call** (both projections + SwiGLU on DSP, one transport per
   layer): 22 calls/token, ≈7 ms. Highest value, largest new-code cost. Scope it
   as its own task doc if P1.2's measurement says transport still dominates.

**Gate for each:** bitwise-or-SNR unchanged vs the pre-change run, call count
dropped by the predicted factor, and **measured** wall-clock improvement. If the
call count drops and wall does not, stop and re-measure — the model was wrong.

### P2 — the u8 boundary (the dominant *compute* term at prefill)

`34_fc_measured.md` §5.3: at M=1024, **quant + dequant = 1,191 of 2,229 µs =
53%**, against 28% for the multiply. Root cause is that every op boundary
round-trips through f32 in DDR (35 §2.2). The fix is a **u8-in / u8-out layer
entry**, and `hexkl_mm_opts` **already takes per-row `act_scale`/`act_zp`**
(`hexkl_mm_u8i4_dma.c:150-152`, `:234-240`) — the caller-supplied path exists and
skips the min/max scan today. It also cuts the FastRPC payload **4×**.

34 §5.3 records this as "the condition FIRED at 53%" and "the next FC lever,
ahead of anything in the kernel". It is **designed, not built**.

For MoE specifically the f32 boundary is between the two projections, around a
CPU-side SwiGLU (`lfm2_moe_layer.cpp:361-373`) — so P2 and P1.3 (fusion) touch
the same seam. Decide between them with P0's numbers, don't build both.

### P3 — HMX/HVX overlap (tile pipelining), last

`35_hmx_hvx_overlap.md` §3 is explicit about ordering: **u8 boundaries first,
then overlap.** They delete HVX work rather than hiding it, and doing overlap
first means tuning chunk sizes around work that is about to be removed. Overlap
alone caps at **1.52×** on the attention block ("you cannot overlap your way out
of being 66% one-lane"); after the u8 change it reaches **1.85×**.

If it comes to this: the design is already written — `37_t2_pipelining_design.md`
§3 (three code changes: `hvx_worker_pool_submit/wait`, second result buffer with
dual-offset layout probe, **16-tile chunks** — HMX ~15.6 µs vs HVX ~8.7 µs per
chunk, 256 KB double-buffered staging). **Build the tile layer only**; the band
layer sits behind a measured "HMX idle > 25%" gate. Discount the expected gain to
**1.4-1.6× of the overlap term**, not its ceiling, for VTCM bank contention
already measured on this device (35 §6).

**Decode note:** overlap's ceiling at M=1 is **1.18× on a 113 µs kernel against
326 µs of transport** (35 §7). **P3 is a prefill lever. It does nothing for
decode** — decode is P1's problem, only.

---

## 6. Framing for the PR / the writeup

State these up front, before any speedup number, or the numbers will be read
wrong:

1. **Decode is bandwidth-bound.** ~485 MB of active weight bytes per token ⇒
   ~35 ms/token from DDR traffic alone, on either processor
   (`40_moe_ffn_htp_task.md` §2.5). The FC kernel's 2.2× DSP-only win is a
   **prefill** number. The honest decode framing is **CPU-core offload and power**
   — attention and conv keep running on ARM while HTP does the FFN — not
   tokens/sec.
2. **Quote `dsp_total`, not wall, when the question is the kernel** — our wall
   moves f32 over FastRPC (4× QNN's bytes) at 64× the rows (34 §3).
3. **The shape check passed and is worth saying:** `gate_up` (2048×3584) and
   `down` (1792×2048) both satisfy HexKL's 32-divisibility with no padding
   (40 §2.4). A 1792-wide dimension satisfying that is not automatic.
4. **Say what was not verified.** If a stage was not run on device, say so
   plainly. "Verified by inspection" is not verification — `CLAUDE.md`'s second
   named habit.

---

## 7. If a pre-quantized checkpoint arrives from someone else

The delivered file is a normal CPU-target Q4_0 checkpoint; **no NPU-specific
export exists or is needed.** The chain is:

```
Q4_0 on disk  →  (already x4- or x8-repacked at quantize time, per --isa)
              →  qs4cx: int4 row-major + w_scale + colsum   [host, htp_qs4cx_from_q4_0x4]
              →  WH tile bytes                              [DSP only, hexkl_weight_u8i4_register]
```

The WH bake **cannot** be precomputed on the host and shipped: it needs the VTCM
arena as scratch and the HMX lock (`hexkl_mm_u8i4_dma.h:52-66`). It is
necessarily a runtime DSP call, already lazy-and-cached per weight pointer
(`htp_compute_ops.cpp:69-98`).

Three things to check about any delivered file, in order:

1. **`--isa ARM`** — the filename must end `_ARM.bin`. Anything else (`_DEFAULT`
   from an x86 host, `_X86`) is x8-packed and will be silently misread as x4
   (B5). This is the one that produces plausible-looking garbage.
2. **It came from this tree's `nntr_quantize`**, not a GGUF/llama.cpp Q4_0 —
   different block layout.
3. **Expert weights were actually quantized.** Sanity check the size ratio: FP32
   → Q4_0 should be ~6× smaller overall for a MoE checkpoint (measured on the
   tiny fixture: 439,648 → 73,568 B = 5.98×). A much smaller ratio means the
   experts were skipped, and the HTP path would then never fire — `dotQnK` is
   only reached for Q4_0 weights.

---

## 8. Rules

Inherits all of `40_moe_ffn_htp_task.md` §5. Additionally:

1. **Do not optimize during goal 1.** §3 predicts E4 is a regression; that is
   the expected result. Record the breakdown and stop.
2. **Do not modify anything already device-verified**:
   `nntrainer/tensor/htp_backend/hmx/*`, `hvx/*`, `test/htp/nntr_hvx.idl`,
   `test/unittest/unittest_hvx_mm_u8i{4,8}.cpp` — except where P1.2/P3 explicitly
   require it, and then only with a device to verify against. If a stage seems to
   need a change there, stop and report which line and why.
3. **Do not touch** `lfm2_moe_layer_fsu.cpp` or `lfm2_moe_layer_cached.cpp`
   (40 §1) — they carry their own copies of the save path (`:542`, `:541`), so
   resist the urge to "fix them too".
4. **Never push the Hexagon SDK's `libcdsprpc.so` to the device.** It is a
   link-time stub that shadows `/vendor/lib64/libcdsprpc.so` and silently makes
   `HtpBackend::enabled() == 0` (40 §6.2).
5. `AGENTS.md`: `git commit -s`, `Co-authored-by:` trailer, `[<component>]`
   subject (`[HTP]` backend, `[CausalLM]` layer/app, `[test]` test-only),
   `clang-format-14` on changed lines only.
6. Every non-trivial change leaves one runnable check behind. E0 is the cheap one
   and it already exists.

---

## 9. Build and run

Host gates: `40_moe_ffn_htp_task.md` §6.1, plus E0's command in §4.

Device build, the `hexkl_addon` trap, the skel push list, and the logcat check:
`40_moe_ffn_htp_task.md` §6.2 — **read it, the two `hexkl_addon` deliveries are
for different consumers and pointing `-Dhexkl-sdk-root` at the wrong one fails
opaquely.**

Device app build (new for this task, B2):

```bash
# fix build_android.sh:8 first -- ANDROID_NDK is ~/workspace/android-ndk-r26d here,
# not ~/Desktop/workspace/android-ndk-r26d
HEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT bash nntrainer/tensor/htp_backend/generate_stub.sh

PATH="$ANDROID_NDK:$PATH" ./tools/package_android.sh --arm-arch=armv8.2-a \
  -Denable-htp=true \
  -Dhexkl-sdk-root=$HEXAGON_SDK_ROOT/addons/hexkl_addon \
  -Dhexkl-lib-subdir=armv8_android26

readelf -d builddir/jni/arm64-v8a/libnntrainer.so | grep NEEDED
#   must list libcdsprpc.so and libsdkl.so
```

Quantize for device (B5 — note `--isa ARM`):

```bash
build/Applications/CausalLM/nntr_quantize <fp32_model_dir> -o <out_dir> \
  --fc_dtype Q4_0 --isa ARM
ls <out_dir>/*_ARM.bin    # must exist; any other suffix is the wrong packing
```
