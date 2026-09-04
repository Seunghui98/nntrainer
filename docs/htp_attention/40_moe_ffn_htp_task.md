# LFM2-8B-A1B MoE FFN on HTP — what's reusable, what's new, staged task

Read `00_START_HERE.md` and `01_working_style.md` first. This branch
(`claude/htp-lfm2-moe-ffn`) is `htp/attention-handoff`'s tree (PR #4256, the
u8i4/u8i8 FC+attention kernels) with LFM2 MoE (PR #4264, `lfm2-moe`) cherry-picked
on top, cleanly — `git log --format=%s 67415241f..HEAD` matches
`git log --format=%s upstream/main..pr4264` commit-for-commit. Written
2026-09-03, no device access this session; everything below is either verified
by reading the tree or explicitly marked as needing one.

---

## 0. What this document is, and the mistake it corrects

The goal: run LFM2-8B-A1B's MoE FFN (22 of 24 layers) through the HTP kernels
PR #4256 already ships, instead of the ARM Q4_0 CPU path.

An earlier pass at this analysis assumed the HTP FC dispatch needed "migrating
off the SDKL macro API" as a prerequisite, because of the documented
[[htp-macro-micro-oneway-door]]-shaped hazard (macro→micro is fine, micro→macro
fails permanently, `0x80000401`, no recovery). **That is already answered in
this tree, do not re-derive it**: `10_mha_htp_plan.md` §6 establishes there is
no SDKL-macro FC left to migrate — `git grep sdkl_npu_mm` over `nntrainer/`
is empty. The only three macro-API calls left are in
`nntrainer/tensor/htp_backend/htp_backend.cpp:38-64`
(`sdkl_npu_initialize`/`sdkl_npu_get_version`/`sdkl_npu_finalize` — a version
print), and `10_mha_htp_plan.md` §6 already recommends deleting them. That is
Stage 1 below, not a separate project.

The real gap is narrower and more interesting than "wire up the FC kernel":
**the ComputeOps seam for multi-weight batched GEMM already exists and already
has a working implementation for a different backend** (OpenCL). Nobody has
pointed it at HTP, and — separately — nothing in this codebase calls it yet, not
even the OpenCL backend's own callers. See §3.

## 1. Scope

In scope: `Lfm2MoELayer`'s FFN (`Applications/CausalLM/models/lfm2_moe/`), and
the `HtpComputeOps` needed to accelerate it. Attention, the router `dot()`, RoPE,
conv layers: untouched, out of scope.

Out of scope, explicitly:

- `lfm2_moe_layer_fsu.cpp` (storage-streaming variant) — its design and DSP
  weight residency are in direct tension (§5.3); mixing them is how this
  becomes unshippable. Land residency against the plain layer first.
- `lfm2_moe_layer_cached.cpp` — same reasoning; check after the plain layer
  works.
- Any change to `nntrainer/tensor/htp_backend/hmx/*` or `hvx/*` files that PR
  #4256 already device-verified (`hexkl_mm_u8i4_dma.c`,
  `hexkl_mm_u8i8_dma.c`, the acc/dequant/quant kernels). They are correct;
  reuse them.
- Generalizing the fix in §3.3/§4 Stage 4 to the OpenCL backend or to other MoE
  models (`qwen3_moe`, `gpt_oss`, `qwen3_cached_slim_moe` all share the same
  per-expert-loop shape). Worth doing later, name it in the PR body, don't do
  it here.

## 2. Established facts — do not re-derive

### 2.1 HTP FC dispatch: the kernel is verified, the ComputeOps wiring does not exist yet

- `nntrainer/tensor/htp_backend/htp_compute_ops.cpp:29` — `get_htp_ops()`
  returns `get_cpu_ops()`, unconditionally. No `HtpComputeOps` class exists.
- `hexkl_mm_u8i4_layer_run` / `hexkl_mm_u8i8_layer_run`
  (`nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i{4,8}_dma.{c,h}`): one shared
  activation, a list of registered weight handles, cross-matmul DMA prefetch,
  in-place accumulator dequant, `hexkl_mm_opts.accumulate` for `+=`. Device-verified,
  measured (`34_fc_measured.md`): DSP-only 156 µs vs QNN 347 µs at K=1024,N=2048,M=64
  (2.2×); decode-shaped x6-grouped calls land at 98 µs/mm wall.
- These live under `nntrainer/tensor/htp_backend/` but are **not** in
  `nntrainer/tensor/htp_backend/meson.build`'s `htp_backend_sources` — they are
  compiled today only into the DSP skel (`test/htp/build.sh`) and the host
  gtests. Wiring them into `libnntrainer` (a `HtpComputeOps` translation unit
  that calls through FastRPC to the already-existing skel, the way
  `test/unittest/unittest_hvx_mm_u8i4.cpp` does) is part of Stage 2.

### 2.2 The ComputeOps seam this needs already exists, with a working implementation for a different backend

`nntrainer/tensor/cpu_backend/compute_ops.h:205-226` declares, as virtuals with
`supports_*()` predicates (the documented pattern for "accelerator-only ops,
CPU-fallback-by-predicate" — see the file's own docstring at the top):

```cpp
virtual bool supports_gemm_q4_0_batch_fp32() const { return false; }
virtual void gemm_q4_0_batch_fp32(std::vector<void *> matAdata, float *matBdata,
                                  std::vector<float *> matCdata, unsigned int M,
                                  std::vector<unsigned int> N, unsigned int K);
```

`matAdata` = one weight pointer per expert, `matBdata` = **one shared
activation**, `matCdata` = one output pointer per expert. This is *exactly*
`hexkl_mm_u8i4_layer_run`'s contract (one activation, N handles, N output
blocks) with pointers instead of handles.

`nntrainer/tensor/cl_operations/cl_compute_ops.cpp` already implements this for
OpenCL — `ClComputeOps::gemm_q4_0_batch_fp32` at line 36, six lines, forwards to
`nntrainer::gemm_q4_0_async_cl`. **This is the template to copy for
`HtpComputeOps`**, not a design to invent (rung 2 of `01_working_style.md`'s
ladder, twice over: reuse the ComputeOps shape *and* the HexKL kernel).

The call site is `FloatTensor::dot(std::vector<Tensor*>, std::vector<Tensor*>,
...)` at `nntrainer/tensor/float_tensor.cpp:771-830`. Read it before writing
`HtpComputeOps` — two details in it are load-bearing and easy to miss:

```cpp
if (o->supports_gemm_q4_0_batch_fp32() && M > 1) {
  o->gemm_q4_0_batch_fp32(mdatas, data, rdatas, M, Ns, K);
} else {
  for (unsigned int i = 0; i < input.size(); ++i)
    o->gemm_q4_0_fp32(M, Ns[i], K, data, K, mdatas[i], Ns[i], rdatas[i], Ns[i]);
}
```

1. **`M > 1` gates the batch path off at decode (M=1).** For the CPU and GPU
   backends this doesn't matter — a single-row GEMM is already fast without
   batching. For HTP it is backwards: decode (M=1, several experts sharing one
   token) is precisely the FastRPC-amortization shape `34_fc_measured.md`
   documents (x4/x6 grouped calls, 516→98 µs/mm). This condition must change
   for HTP to see the win at decode — see Stage 4.
2. **Nothing in this tree currently calls the vector `dot()` overload at all.**
   `grep -rn "supports_gemm_q4_0_batch_fp32\|gemm_q4_0_batch_fp32"` finds only
   the declaration, the CPU no-op default, `ClComputeOps`'s implementation, and
   this dispatch site. No layer anywhere constructs the `vector<Tensor*>` and
   calls it. Wiring `Lfm2MoELayer` to call it is new work, not a bug fix to an
   existing broken path (Stage 4).

### 2.3 What batches and what does not — the one design fact that determines the whole shape of this task

`Lfm2MoELayer::compute_expert_forward_no_critical`
(`Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.cpp`) does two `dot()`
calls per active expert, in a loop over experts:

```cpp
token_input.dot(gate_up_proj, gate_up_out);      // one call per expert, gate/up
...
acti_out.dot(down_proj, expert_output);          // one call per expert, down
```

**Gate/up projection**: at decode (`incremental_forwarding`, one token), every
selected expert's `token_input` is the exact same shared-data view — same
tensor, same offset, because `total_tokens == 1`. This is the shared-activation
shape the batch API wants, unmodified. At prefill each expert already gets its
own multi-token batch (`M = assignments.size()`), so the *existing* single-call
`token_input.dot(gate_up_proj, ...)` per expert already is the natural unit —
nothing to batch across experts there, since each expert's `M` and rows differ.

**Down projection**: `acti_out` is `swiglu(gate_up_out)` — **different per
expert**, even at decode, because each expert's `gate_up_proj` differs.
`gemm_q4_0_batch_fp32`'s contract is *one* activation against N weights; it
cannot express N different activations against N different weights in one
call. **There is no existing ComputeOps virtual for this** (`grep -n "virtual
bool supports_.*batch" compute_ops.h` finds only the shared-activation
variants). A "grouped GEMM" primitive — N independent (activation, weight)
pairs sharing only `K` and one FastRPC/kernel launch — does not exist anywhere
in this tree, on any backend. This is the one piece of this task that is
genuinely new design, not reuse. Treat it accordingly: prove out gate/up first
(Stage 4, all reuse), then decide whether down's fusion is worth its own DSP
entry point (Stage 5, flagged, not started by default).

`swiglu` itself (`nntrainer::swiglu`, called directly from
`compute_expert_forward_no_critical`, not through `ComputeOps`) runs on the
CPU regardless of which backend supplies the two GEMMs around it — that split
is fine and is not part of this task; `attn_forward`'s "fuse everything into
one FastRPC call" reasoning does not apply here because there is no chain of
three matmul-shaped ops with the same activation the way Q·Kᵀ→softmax→P·V has.

### 2.4 LFM2-8B-A1B shapes, and the residency wall

From the real checkpoint's config (`hidden_size=2048`,
`moe_intermediate_size=1792`, `num_experts=32`, `num_experts_per_tok=4`,
`num_hidden_layers=24`, `num_dense_layers=2` → **22 MoE layers**):

| weight | K | N | K, N ÷ 32 |
|---|---|---|---|
| `gate_up_proj` (fused) | 2048 | 3584 | 64, 112 — exact |
| `down_proj` | 1792 | 2048 | 56, 64 — exact |

Both satisfy `hexkl_weight_u8i4_register`'s shape check
(`hexkl_mm_u8i4_dma.c:59-61`, `K % HEXKL_HMX_INT8_BLOCK_N_INNER == 0 && N %
HEXKL_HMX_INT8_BLOCK_N_COL == 0`) with no padding. That is not automatic for a
1792-wide dimension; it is worth stating in the PR that it was checked.

**Residency, the real number:** `hexkl_weight_u8i4_register` bakes into VTCM
scratch, then `memcpy`s the WH bytes into a `malloc`'d DSP-heap block
(`hexkl_mm_u8i4_dma.c:100-115`) that lives until `_release`. At i4:
`gate_up` = 3.5 MiB/expert, `down` = 1.75 MiB/expert → **3.6 GiB total** for
all 22×32 experts, against `HEXKL_MM_U8I4_MAX_WEIGHTS = 512`
(`hexkl_mm_u8i4_dma.h:29`) when **1,408** handles are needed. Neither the DSP
heap budget nor the table size survives contact with the full model as the
registry is written today. This is Stage 6, gated behind Stages 1-5 actually
working — see §5.3 for why it can wait.

**The tiny reference config does not hit this wall.** LFM2 MoE's existing gate
(`test/unittest/models/causallm_reference/lfm2_moe_tiny/config.json`):
`hidden_size=64, moe_intermediate_size=32, num_experts=4,
num_experts_per_tok=2, num_hidden_layers=3, num_dense_layers=1` → 2 MoE layers.
Total i4 WH bytes for every expert in the tiny model: **~32 KiB**, 8 handles.
This means **Stages 1-5 are fully device-verifiable against the existing
`unittest_causallm_lfm2_moe_reference` golden test without solving residency at
all** — see §5.

### 2.5 Decode is bandwidth-bound; frame expectations around that, not around the FC speedup table

Per-token active weight bytes at i4: `970M params × 0.5 byte ≈ 485 MB`
(gate_up+down for 4 of 32 experts × 22 layers). At the measured effective DMA
rate implied by `34_fc_measured.md` (~1 MiB weight in 73 µs DSP-side ⇒ ~14
GB/s), that is **~35 ms/token from weight traffic alone** — in the same range
as the ARM Q4_0 CPU path, because both read the same DDR bytes. The FC kernel's
2.2× DSP-only speedup (§2.1) is real but is not the number to quote for decode
throughput; it is the number for **prefill**, where the same weight byte is
amortized over `M` rows instead of one. State this in the PR up front. The
correct decode framing is CPU-core offload (attention/conv keep running on ARM
while HTP does the FFN) and power, not tokens/sec.

## 3. The seam, precisely — three tiers, only one is new design

| tier | ComputeOps virtual | shape | HexKL call | status |
|---|---|---|---|---|
| 1. single weight | `gemm_q4_0_accel_fp32` (`compute_ops.h`, already declared) | 1 act × 1 weight | `hexkl_mm_u8i4_layer_run` with `n_handles=1` | reuse — mirrors `ClComputeOps::gemm_q4_0_accel_fp32` |
| 2. shared-activation batch | `gemm_q4_0_batch_fp32` | 1 act × N weights | `hexkl_mm_u8i4_layer_run` with `n_handles=N` | reuse — mirrors `ClComputeOps::gemm_q4_0_batch_fp32`; needs the `M>1` gate fixed (§2.2) and a new call site in `Lfm2MoELayer` (§2.3) |
| 3. grouped / multi-activation | none exists | N acts × N weights | none exists | **new** — down_proj at decode only; Stage 5, optional |

Tier 1 alone already benefits every FC call in every model that runs under
`engine=htp` — attention Q/K/V/O, dense MLP, and (via the existing per-expert
loop, unchanged) both MoE projections at prefill and gate/up-only at decode
before Stage 4 lands. It is the highest-leverage, lowest-risk stage and has no
MoE-specific code in it at all.

## 4. Staged task

Each stage has its own verification gate; do not start a stage whose gate the
previous stage didn't clear.

### Stage 1 — delete the dead macro-API calls

`10_mha_htp_plan.md` §6 recommendation 1, already specified: delete the three
`sdkl_npu_*` calls in `htp_backend.cpp` (init, get_version, finalize) and
`sdkl_compat.h`/`libsdkl.so` from the link. `npuAlive()`/`enabled_` key off the
FastRPC session opening (Stage 2) instead of the macro session.

**Gate:** `HtpBackend` builds and links with zero macro-API references
(`git grep sdkl_npu nntrainer/` empty after this stage). No behavior change
observable yet — `get_htp_ops()` still returns `get_cpu_ops()`.

### Stage 2 — FastRPC session wiring (done)

Real session in `HtpBackend`, not a `HtpComputeOps` subclass yet:
`htp_compute_ops.cpp`'s own top-of-file comment already explains why an
empty subclass is the wrong move before the first kernel exists ("with
nothing overridden, a subclass would only duplicate get_cpu_ops()") --
that arrives in Stage 3, bundled with its first real override, not before.

What actually landed:

- `nntrainer/tensor/htp_backend/generate_stub.sh` -- runs `qaic` against
  `test/htp/nntr_hvx.idl` (the same IDL the DSP skel is built from) and
  writes `nntr_hvx.h` / `nntr_hvx_stub.c` into `generated/` (gitignored).
  **Not** a meson `custom_target`: that path was tried first and reverted --
  `jni/meson.build`'s Android build calls `files()` on every entry of
  `nntrainer_sources` to hand them to `ndk-build`, and `files()` requires
  the path to exist at configure time, which a custom_target's lazily-built
  output does not. A manual pre-generation step (mirroring `test/htp/build.sh`'s
  own existing manual-step convention for the same proprietary `qaic` tool)
  sidesteps that entirely: the file exists on disk before meson ever runs.
- `nntrainer/tensor/htp_backend/meson.build` errors out by name if
  `generated/nntr_hvx_stub.c` is missing, then adds it to `nntrainer_sources`
  the same way `htp_backend.cpp` already is -- which is what makes it reach
  **both** builds through one line: the host meson build directly, and the
  Android build via `jni/meson.build`'s `MESON_NNTRAINER_SRCS` (templated into
  `jni/Android.mk.in`), since both read the same `nntrainer_sources` list.
- `HtpBackend`'s constructor: `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, ...)`
  (non-fatal if it fails -- a signed production skel doesn't need it) then
  `nntr_hvx_open(nntr_hvx_URI "&_dom=cdsp", &h)`, mirroring
  `test/unittest/unittest_hvx_mm_u8i4.cpp`'s `HmxMmU8I4::SetUp`. Destructor
  calls `nntr_hvx_close`. `handle()` exposes the session as a plain `uint64_t`
  (not `remote_handle64`) so this header stays free of the Hexagon SDK include
  path -- Stage 3's `HtpComputeOps` is the first caller that needs the real type.

**A trap worth naming so the next session doesn't re-fall into it:** there are
two different `hexkl_addon` deliveries on a dev machine set up for this work,
and the top-level `-Denable-htp=true` build option wants the *other* one from
what `test/htp/build.sh` wants.

| | layout | what wants it |
| :-- | :-- | :-- |
| SDK-bundled (`Hexagon_SDK/<ver>/addons/hexkl_addon`) | `lib/<arch>/libsdkl.so` (flat) | top-level `meson.build`'s `enable-htp` option (`-Dhexkl-sdk-root=`) |
| standalone beta drop (e.g. `hxkl-beta2/hexkl_addon`) | `lib/<ver>/<arch>/libhexkl_micro.a` (versioned) | `test/htp/build.sh`'s `HEXKL_ROOT` (DSP skel, micro API) |

Pointing `-Dhexkl-sdk-root` at the standalone drop fails opaquely (`hexkl-lib-subdir`'s
default has no version component to match the versioned layout). This is a
pre-existing mismatch in `meson.build`/`meson_options.txt`, not something this
stage introduced, and it was not fixed here -- the macro-API `libsdkl.so` link
this option controls is unused dead weight regardless (Stage 1), so "point at
the right directory" was the smaller, correct-scope fix over "repair the
option's path derivation for a library nothing calls."

**Gate (host, no device):** `htp_backend.cpp`, `htp_backend.h`, `htp_context.cpp`
syntax-check clean with `-DENABLE_HEXKL=1` against real Hexagon SDK 6.4.0.2
headers, and the default `enable-htp=false` build is untouched.

**Gate (device, cleared):** a full Android cross-build via
`./tools/package_android.sh --arm-arch=armv8.2-a -Denable-htp=true
-Dhexkl-sdk-root=<Hexagon_SDK>/addons/hexkl_addon -Dhexkl-lib-subdir=armv8_android26`
against a **fresh** `builddir` (an existing one predating these meson options
cannot pick them up via `meson configure`; delete or move it aside first) links
`libnntrainer.so` clean through `ndk-build`. Run `generate_stub.sh` first with
`HEXAGON_SDK_ROOT` pointed at the SDK root (not either hexkl_addon).

### Stage 3 — Tier 1: single-weight accel (done)

LFM2 MoE saves weights as Q4_0, and — this was not obvious going in, see
below — on any Android/ARM64 build (the only kind `ENABLE_HEXKL` compiles for)
`repack_q4_0`'s ARM and DEFAULT branches both bake the loaded bytes into the
**Q4_0x4** interleave, not plain per-block `block_q4_0`. HexKL's registry wants
qs4cx (one scale + one colsum per output channel, not one scale per 32-wide
block). `nntrainer::htp_q4_0_convert.h`'s `htp_qs4cx_from_q4_0x4` requantizes
one into the other, going back through f32 (a real requantization, not a bit
reshuffle — accuracy cost is measured, see below).

**Two things worth naming so the next session doesn't re-derive them:**

1. **There is no working "unpack repacked Q4_0" function to build this on —
   look for `Q4_0Utils::dequantizeQ4_0x4` instead, which does the whole
   decode in one call.** The obvious-looking path is `unpack_q4_0` (paired
   inverse of `repack_q4_0`, same header) → `nntr_dequantize_row_q4_0`. That
   path is a trap: `unpack_q4_0`'s **fallback**-backend implementation
   (`fallback_internal.cpp`) is `throw std::runtime_error("NYI ...")` — real
   only in the arm/x86 backends, and even those differ in which repack width
   each backend's `unpack_q4_0` inverts. `Q4_0Utils::dequantizeQ4_0x4`
   (`nntrainer/tensor/q4_0_utils.cpp`) is portable (no ISA guard) and
   decodes Q4_0x4 straight to f32 in one call — no unpack step exists to look
   for because none is needed.
2. **`htp_q4_0_convert.{h,cpp}` live in `nntrainer/tensor/`, not
   `htp_backend/`, and are not gated behind `ENABLE_HEXKL`.** The function has
   no Hexagon SDK dependency — it is plain code against `Q4_0Utils`, same as
   `q4_0_utils.cpp` next to it. Keeping it ungated is what let its accuracy
   test run in the default host build; gating it behind `ENABLE_HEXKL` (the
   instinctive choice, since only `HtpComputeOps` calls it) would have made
   Stage 3's whole accuracy question undebuggable without a device.

`HtpComputeOps::gemm_q4_0_accel_fp32` (`htp_compute_ops.cpp`) is the first
real override, shaped exactly like `ClComputeOps`'s equivalent: lazy weight
registration (`std::unordered_map<const void*, uint32_t>` keyed on the Q4_0
tensor's data pointer, registered on first sight, never released this stage —
that's Stage 6's problem) via `nntr_hvx_weight_register_u8i4`, then
`nntr_hvx_mm_u8i4_layer` with one handle.

**Gate (host-only, no device): cleared.**
`nntrainer_cpu_backend_standalone.htp_qs4cx_from_q4_0x4_accuracy`
(`test/unittest/unittest_nntrainer_cpu_backend.cpp`) — quantize_q4_0 →
repack_q4_0(ISA::ARM) → htp_qs4cx_from_q4_0x4 → reconstruct → compare
against the original fp32 weight, plus a colsum cross-check. Measured:
max_abs_err=0.189, mean_abs_err=0.045 on a random [-1,1] 768×512 weight.

**Gate (device): cleared, on the connected Galaxy device (R5KL30G6MLT).**
A standalone binary linking the real `-Denable-htp=true` Android build and
calling `get_htp_ops()->gemm_q4_0_accel_fp32()` directly (§6 has the exact
recipe). First measurement at a small shape (M=4,K=256,N=128) printed
max_rel_err=939%, which looks alarming until read against this project's own
established metric for this exact kernel: `unittest_hvx_mm_u8i4.cpp`'s
already-device-verified accuracy harness reports max_rel up to **1908×** at
M=64/K=1024/N=1024 as *passing*, because a handful of near-zero output
elements make max-relative-error the wrong metric for 4-bit quantization —
SNR is what that harness actually gates on. Re-measured with SNR at that
harness's own shape: **22.8 dB**, against the harness's own **23.5 dB** for
direct qs4cx quantization (no Q4_0 hop) at the same shape. The ~0.7 dB gap is
this path's one extra hop, and it is small.

**Not yet done, and the natural next increment before Stage 4:** the
`lfm2_moe_tiny` reference-logit gate this section originally specified
(`unittest_causallm_lfm2_moe_reference.cpp` reproducing under `engine=htp`
end to end, through the real `Tensor::dot()` → `dotQnK` → `ComputeOps` path
rather than a direct kernel call). The kernel-level verification above is
strictly stronger evidence about the arithmetic; the model-level test is
still worth running because it is the first check that the *dispatch path*
(engine selection, tensor context, the real weight pointers a loaded model
produces) works, not just the kernel called directly.

### Stage 4 — Tier 2: shared-activation batch, gate/up at decode

1. Change the `M > 1` gate at `float_tensor.cpp:797` so HTP's batch path fires
   at `M == 1` too — the cleanest fix is querying a new
   `ComputeOps::batches_at_m1()`-style predicate (or just letting HTP's
   `supports_gemm_q4_0_batch_fp32()` mean "always, including M=1" and gating
   the CPU/GPU fallback on `M > 1` explicitly inside their own predicate) —
   pick whichever is the smaller diff once Stage 3 code exists; do not
   over-design this ahead of seeing the actual call site.
2. `HtpComputeOps::gemm_q4_0_batch_fp32` — same handle cache as Tier 1,
   `n_handles = matAdata.size()`, one `mm_u8i4_layer` call, split the
   contiguous `out_cat` back into `matCdata[i]` per handle in call order (the
   kernel's documented output order, `hexkl_mm_u8i4_dma.h`'s `layer_run`
   docstring).
3. `Lfm2MoELayer::incremental_forwarding`'s expert loop (decode path only):
   when `total_tokens == 1`, collect the selected experts'
   `gate_up_proj` weights into a `vector<Tensor*>`, call the batched `dot()`
   once instead of looping. Prefill path (`forwarding`, and decode's own
   down-projection) unchanged in this stage.

**Gate:** same reference-logit test as Stage 3, plus a decode-path-specific
check — force `incremental_forwarding` with `to - from == 1` and confirm the
batched call path is actually taken (not silently falling through to the loop)
and produces the same logits as before this stage, on both the tiny model and
CPU-only (`engine=cpu`, to isolate "did the call site change break anything"
from "did the DSP path work").

### Stage 5 — Tier 3: down_proj grouped matmul (flagged, not default-scope)

No existing primitive. Two honest options, in ladder order:

1. **Ship without it first.** Route decode's down_proj through Tier 1
   (Stage 3), one `gemm_q4_0_accel_fp32` call per expert — correct, reuses
   everything, pays FastRPC's ~326 µs fixed cost `topk` times per layer instead
   of once. Mark this a `ponytail:` comment naming the ceiling (paying
   FastRPC per-expert on down_proj) and the upgrade path (this section).
   Measure whether it matters before building more: `topk=4` decode is 4×326
   µs ≈ 1.3 ms/layer of transport, against the ~35 ms/token bandwidth floor
   (§2.5) — likely small in relative terms, but measure, don't assume.
2. **If measurement says it matters:** a new DSP entry
   (`mm_u8i4_grouped_layer` or similar) taking N independent
   `(activation, handle)` pairs sharing `K`, one FastRPC call, N separate
   quantize passes instead of one. This is new HMX/DMA-ring code, not a
   parameter reshuffle of `hexkl_mm_u8i4_layer_run` — do not attempt it without
   device access to verify it; write it as its own follow-up task doc if it
   comes to that, with the same rigor as `30_flash_attention_task.md`.

**Default position: do (1), measure, stop. Do not build (2) speculatively —
that is rung 1 of the ladder** ("does this need to exist at all"), and nobody
has measured yet that it does.

### Stage 6 — residency (only after Stages 1-5 pass on the tiny model)

Not started by default; scoped here so it isn't invented ad hoc later.

1. `hexkl_weight_u8i4_register_ref`-style variant: reference caller-supplied
   DSP-visible bytes (an ION/rpcmem allocation, `htp_rpc_bench.h`'s `RpcMemApi`
   pattern promoted out of test-only code) instead of `malloc`+copy.
2. The WH bake must happen once at model load into that ION buffer, and it must
   **replace** the CPU-side Q4_0 tensor's storage, not sit alongside it — an
   extra copy re-adds the 3.6 GiB problem this stage exists to remove.
3. `HEXKL_MM_U8I4_MAX_WEIGHTS = 512` → needs to hold 1,408 for the full
   checkpoint; make it caller-sized or a growable table, whichever is the
   smaller diff against `hexkl_mm_u8i4_dma.h`'s current fixed-array struct.
4. Explicitly out of scope here (§1): `lfm2_moe_layer_fsu.cpp`'s storage
   streaming. Land this against the plain layer only.

## 5. Rules

1. Do not modify anything already device-verified:
   `nntrainer/tensor/htp_backend/hmx/*`, `hvx/*`, `test/htp/nntr_hvx.idl`,
   `test/unittest/unittest_hvx_mm_u8i{4,8}.cpp`. If a stage seems to need a
   change there, stop and report which line and why — most likely you have
   reached for rung-3-or-lower code that rung 2 already provides (§3).
2. Do not implement Stage 5 option (2) or Stage 6 without a device to verify
   against. If you believe a stage cannot be completed without one, stop and
   say so — do not simulate a device.
3. Do not touch `lfm2_moe_layer_fsu.cpp` or `lfm2_moe_layer_cached.cpp` (§1).
4. Do not change `compute_ops.h`'s existing virtuals' signatures — add new ones
   if Stage 5 needs them.
5. Do not "fix" the `M > 1` gate by removing batching for other backends —
   CPU/GPU's current behavior at M=1 is intentional (§2.2); the fix is HTP
   opting in at M=1, not everyone losing the gate.
6. Follow `AGENTS.md`: `git commit -s`, `Co-authored-by:` trailer,
   `[<component>] <description>` subject (`[HTP]` for backend files, `[CausalLM]`
   for layer files, `[test]` for test-only changes), `clang-format-14` on
   changed lines.
7. Read `01_working_style.md` and work that way — in particular, §3's table is
   the rung-2 check for this entire task; re-read it before writing a new
   ComputeOps virtual or a new HexKL entry point.

## 6. Build and run

### 6.1 Host-only gates

```bash
meson build -Denable-transformer=true
ninja -C build
meson configure build -Denable-test=true   # first time only
ninja -C build
cd build && meson test unittest_nntrainer_cpu_backend --print-errorlogs
```

`unittest_nntrainer_cpu_backend`'s `htp_qs4cx_from_q4_0x4_accuracy` case is
Stage 3's converter gate and needs no device or SDK — it is not gated behind
`ENABLE_HEXKL` (§Stage 3).

### 6.2 Device gates — the exact recipe this session used, paths included

**Environment on the dev machine this was verified on** (adjust paths for a
different machine, but keep the *distinction*, not just the values — see the
hexkl_addon trap below):

```bash
export HEXAGON_SDK_ROOT=~/workspace/Hexagon_SDK/6.4.0.2
export ANDROID_NDK=~/workspace/android-ndk-r26d
```

**Baseline first — confirm the already-verified kernels still work before
trusting anything built on top of them:**

```bash
HEXKL_ROOT=~/workspace/hxkl-beta2/hexkl_addon \  # the STANDALONE beta drop
HEXKL_SDK_VER=6.4.0.2 \
bash test/htp/run_u8i4_layer_on_device.sh
```

All 38 tests across `unittest_hvx_mm_u8i4`/`_softmax`/`_attn`/`_fc` must
pass. `speedup_vs_harness` printing 50×+ is expected, not a red flag — that
test's own comment says "printed, never asserted"; the harness path rebakes
the weight every call, layer_x4 doesn't, so the ratio is dominated by the
~12 ms bake, not by anything this task changes.

**Then the real `-Denable-htp=true` build — a trap to know about first:**
there are two different `hexkl_addon` deliveries on a dev machine set up for
this work, and they are for different consumers:

| | layout | who wants it |
| :-- | :-- | :-- |
| SDK-bundled: `$HEXAGON_SDK_ROOT/addons/hexkl_addon` | `lib/<arch>/libsdkl.so` (flat) | `-Denable-htp=true -Dhexkl-sdk-root=` |
| standalone beta drop (`hxkl-beta2/hexkl_addon` above) | `lib/<ver>/<arch>/libhexkl_micro.a` (versioned) | `HEXKL_ROOT` for `test/htp/build.sh` / `run_u8i4_layer_on_device.sh` |

Pointing `-Dhexkl-sdk-root` at the standalone drop fails opaquely (missing
version segment in the default `hexkl-lib-subdir`). Use the SDK-bundled one:

```bash
# Generate the FastRPC client stub once (Stage 2) -- meson errors by name
# if this hasn't been run:
HEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT bash nntrainer/tensor/htp_backend/generate_stub.sh

# A builddir that predates -Denable-htp/-Dhexkl-* in meson_options.txt cannot
# pick them up via `meson configure` -- move an existing one aside first.
mv builddir builddir.bak   # only if builddir already exists

PATH="$ANDROID_NDK:$PATH" ./tools/package_android.sh --arm-arch=armv8.2-a \
  -Denable-htp=true \
  -Dhexkl-sdk-root=$HEXAGON_SDK_ROOT/addons/hexkl_addon \
  -Dhexkl-lib-subdir=armv8_android26
```

Confirm the link is real before trusting anything downstream:

```bash
file builddir/jni/arm64-v8a/libnntrainer.so       # ELF ... ARM aarch64
readelf -d builddir/jni/arm64-v8a/libnntrainer.so | grep NEEDED
#   must list libcdsprpc.so and libsdkl.so
```

**On-device: push the skel, the library, and run.** The one thing that will
silently produce `HtpBackend::enabled() == 0` (or worse, garbage results) is
pushing the Hexagon SDK's `libcdsprpc.so` to the device — it is a link-time
stub, not a runtime implementation, and it shadows the device's real one at
`/vendor/lib64/libcdsprpc.so` if it's anywhere on `LD_LIBRARY_PATH`. **Do not
push it.**

```bash
DEVICE_TMP=/data/local/tmp/htp_verify
adb shell "mkdir -p $DEVICE_TMP"
adb push builddir/jni/arm64-v8a/libnntrainer.so "$DEVICE_TMP/"
adb push builddir/jni/arm64-v8a/libsdkl.so "$DEVICE_TMP/"        # unused dead weight (Stage 1), still linked
adb push builddir/jni/arm64-v8a/libc++_shared.so "$DEVICE_TMP/"
adb push test/htp/build/libnntr_hvx_skel.so "$DEVICE_TMP/"        # from the baseline run above
# do NOT: adb push builddir/jni/arm64-v8a/libcdsprpc.so ...

adb shell "cd $DEVICE_TMP && LD_LIBRARY_PATH=$DEVICE_TMP ADSP_LIBRARY_PATH=$DEVICE_TMP ./your_test_binary"
```

`adb logcat -d | grep -iE "nntrainer|adsprpc"` is the fastest way to see
what actually happened inside the FastRPC session — `remote_handle64_open`
opening `libnntr_hvx_skel.so` on a domain, followed later by a clean
`remote_handle64_close`, is what success looks like at the transport level,
independent of whether the *numbers* it returns are also right.

**For accuracy, use SNR, not max relative error.** A handful of near-zero
output elements make max-relative-error explode under 4-bit quantization
regardless of correctness — this project's own `unittest_hvx_mm_u8i4.cpp`
prints `max_rel` up to 1908× on a passing, correct run. Compare SNR (dB)
against the shape-matched number in `34_fc_measured.md` or the harness's own
printed value instead.

Paste complete output at each gate, not a summary — same rule as
`30_flash_attention_task.md` §5.
