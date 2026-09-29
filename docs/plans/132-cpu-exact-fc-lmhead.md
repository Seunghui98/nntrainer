# 132 — PR 2: CPU-exact FC, router, dense FFN and lm_head on the HTP

Issue: dlwlzzero/nntrainer#132 (p1), re-scoped on 2026-09-29. PR 1 (ADD +
ROUTER_TOPK, 95 → 73 → 51) is merged and measured; see `132-add-router.md`.
This file is read against `htp_moe` @ `40e797ed`.

The user's direction of 2026-09-29 has two parts:

* Decode must end up running end to end on the NPU.
* The bit-preserving rule stays (LEDGER rule 45, contract §12): the text must
  be identical to the switch-off run, and there is no PPL route.

So every op this PR makes resident has to produce the **Android CPU's bits**:

* the M=1 Q4_0 FCs (conv `in_proj` / `out_proj`, attention q/k/v/o, the dense
  FFN of layers 0–1),
* the router FC with its top-k,
* the final norm,
* the tied lm_head with the greedy pick.

The method is #164's (`164-cpu-order-norm-scale.md`):

1. Read the order from the shipped `libnntrainer.so`.
2. Write it as a `_det` spec.
3. Make the kernel equal the spec.
4. Gate on ARM, on the host and on the DSP.
5. Check every op against the CPU on the same input with a shadow.

**Two findings decide the shape of this plan (§0.3, §0.4).**

* The FC set and lm_head do not fit in the DSP's 32-bit address space next to
  the MoE arena.
* The exact float chain costs more DSP time than the CPU takes for the same
  work.

Both need a user decision, so the plan has two parts:

* **Part A** is decidable now: the specs, the kernels as test entries, the
  shadows and a microbench sitting.
* **Part B**, residency and 1 call/token, starts only after decision **D**
  (§3.4).

## 0. What the shipped CPU computes (done on the workstation)

**Inputs.** The disassembly is `llvm-objdump` (NDK r30) of the 2026-09-29
set:

* `/local/mnt/workspace/htp_moe/norm/set/libnntrainer.so`, md5
  `c23e6aa2177b1e8c8920f44e8cd6be08`, `-march=armv8.2-a+fp16+dotprod+i8mm`,
  `-ffast-math`, thread backend `omp`;
* `libcausallm_core.so` of the same set.

These are the functions the decode path reaches. On the NPU model
`HtpComputeOps::accelerates_q4_0_at_m1()` is false
(`htp_compute_ops.cpp:999`), so every M=1 Q4_0 dot goes to the CPU table
(`float_tensor.cpp:810-816`, `:1027-1035`).

### 0.1 Q4_0 FC at M = 1 (every FC and lm_head)

Call path:

* `nntrainer::gemm_q4_0<float>` → `__ggml_q4_0_4x8_q8_0_GEMM<float>` at
  `0x33d60c`, source `ggml_interface_omp.cpp:36-57`.
* The batched q/k/v form is `:158-183`.
* The lm_head uses the same call on its blocked twin
  (`tie_word_embedding.cpp:469-476`). The rowwise fallback `:477-496` runs
  only if the twin failed to allocate, and G2 detects that case.

1. **Activation quantizer** `nntr_quantize_row_q8_0` @ `0x330960`
   (`nntr_ggml_impl_quant.cpp:622-662`), one call per FC input, per block of
   32:
   * `amax = fmaxv(|x|)`;
   * `d = fdiv(amax, 127.0f)`. This is a true division; `-ffast-math` did
     **not** turn it into a multiply;
   * `id = fdiv(1.0f, d)`, or 0 when `d == 0`;
   * `q = fcvtns(fmul(x, id))`, round to nearest, ties to even, keeping the
     low byte (`tbl`);
   * the block's `d` is stored as `fcvt h, s` (RN f32 → f16, subnormal f16
     kept).
2. **Threading** splits N into chunks of 16 columns. One call computes each
   column whole, so the thread count does not change a bit.
3. **Per column** `nntr_gemv_q4_0_4x8_q8_0` @ `0x337f2c`
   (`nntr_ggml_impl_neon.cpp:42-78`):
   * `isum_b` = the exact int32 Σ (q_w − 8)·q_a over the block (`sdot` on
     nibbles pre-shifted by 4, then `scvtf #4`, which is exact);
   * `s_b = fmul(f16→f32(d_a), f16→f32(d_w))`. This product is **exact**
     (11 × 11 bits);
   * **one fused chain from +0 in block order**:
     `acc = fmla(acc, isum_b, s_b)` for b = 0 … K/32 − 1. That is 64 steps
     at K = 2048 and 224 at K = 7168.

   So each step is RN(acc + isum·s) with an exact 37-bit product and one
   rounding. The output is never −0.

### 0.2 Router, dense FFN SwiGLU, greedy pick

**Router logits.** The path is `input.dot(gate)` → `__cblas_sgemv` →
statically linked OpenBLAS `cblas_sgemv`, reaching `sgemv_n` @ `0x460b10`
through the table at `0x48f720`.

* Row-major [2048 × 32] with the transpose flag gives column-major
  `sgemv_n` with m = 32 and n = 2048.
* For each output i: y = +0 (`sscal` by β = 0), then `y = fmadd(x_j, a_ji, y)`
  for j = 0 … 2047. This is one fused chain.
* m·n = 65536 is above 9215, so the threaded `sgemv_thread_n` would split m
  into ≥ 4-row slices, which does not change the per-row chain. With
  `openblas-num-threads=1` it is not taken.

**Router scores and weights** (`buildExpertAssignments` @ `0x4e5990` in
`libcausallm_core.so`):

* `sig = fdiv(1, fadd(expf(−l), 1))`. `expf` is the **device's bionic
  libm** (a PLT call). It is not a NEON approximation.
* The select score is `fadd(sig, bias)`. Top-4 uses the total-order
  comparator (`lfm2_moe_layer.cpp:327-331`).
* `wsum = (sig[e1] + sig[e3]) + (sig[e0] + sig[e2])`, with e0..e3 in sorted
  order. Two accumulators; the loop was reassociated by `-ffast-math`.
* `inv = fdiv(1, wsum + 1e-6f)`.
* `w = fmul(sig, inv)`. `ROUTED_SCALING_FACTOR` is 1.0 and was folded away.

PR 1's spec (`m1_ops_det.h:70-84`, `:274`) is not this order:

* its GEMV uses 4 partial sums;
* it uses `exp_det` / `recip_det`.

So the merged ROUTER_TOPK is **not** bit-identical to the CPU. This matches
the "gap < 1e-4" statistic of PR 1.

**Dense FFN.** On the NPU config `createMlp` builds up FC, gate FC, `swiglu`
and down FC (`transformer.cpp:774-822`). `neon::swiglu` @ `0x36ca78`
(`neon_impl.cpp:926-1014`) compiles `exp_ps` (cephes) with:

* exactly **one** fused op per element: `fx = fmla(0.5, x, log2e)`;
* all other steps as separate `fmul` / `fadd` / `fsub`: 11 fmul, 9 fadd and
  2 fsub per element;
* the clamp at ±88.376;
* the floor through `fcvtzs` / `scvtf` / `fcmgt`;
* then `fdiv(y, e + 1)` and `fmul(·, z)`.

**Greedy pick** (`causal_lm.cpp:329-331`): `std::max_element`, where the first
maximum wins. `repetition_penalty` is 1 on the decode call and the bad-word
list is empty.

**Final norm.** It is `rms_norm` (`lfm2_causallm.cpp:430-434`), whose CPU
order #164 owns.

### 0.3 The exact FC chain on the HVX: works, but costs more than the CPU

**Prototype.** The scratch file is `fc132/fcproto.c`, in the planner's
scratchpad, not committed. One *vector-step* is 32 columns × one 32-block.

* Integer part: nibble unpack, 8 × `vrmpyacc(Vub, Rt.b)`, minus 8·Σq_a.
* The product is split exactly: T = isum·m_w as an int32 (26 bits) =
  T_hi·2¹³ + T_lo. Then P1 = f32(T_hi·m_a)·2^(e+13) and
  P2 = f32(T_lo·m_a)·2^e, both exact.
* acc = RN(acc + P1 + P2) by Boldo–Melquiond:
  * Fast2Sum(P1, P2);
  * TwoSum(acc, uh);
  * round-to-odd of tl + ul, emulated with a TwoSum and a one-ulp step;
  * RN(th + v).

**Results** on `hexagon-sim -mv79`:

| check | result |
|---|---|
| bit equality with the host's `fmaf` chain (random weights, signed f16 scales, 2 × 256 columns × 64 blocks) | **512/512 columns** in both builds: `Q6_Vsf_*` intrinsics, and native `vN.sf = vadd/vsub/vmpy(.sf,.sf)` via inline asm |
| the `Q6_Vsf_*` intrinsics lower to `qf32` op + `.sf=.qf32` convert pairs (seen in the `-S` output) | 85 packets per step |
| native IEEE sf instructions (they assemble for v79 with `-mhvx-ieee-fp`) | 76 packets per step |
| ISS pcycles per step, single thread | **164** (intrinsics), **124** (native) |

**Cost for the set.** 453 M FC weights + 262 M lm_head = 715 M weights =
**698 k vector-steps per token**.

| bound | how | ms / token |
|---|---|---|
| optimistic | ISS 124 pcycles/step, 6 HVX lanes scaling perfectly, 1.74 GHz | **8.3** |
| silicon-calibrated | ≈ 1 packet / pcycle aggregate (plan 146: 6.3 pcycles / packet / lane at 6 lanes), 76 packets | **30** |
| bytes alone | 402 MB at the bypass DMA's 57 GB/s in-app (rule 43) | 7.05 (hidden under compute) |
| **the CPU today** | FC + lm_head at 51–63 GB/s (rule 46) | **7.4** |

**So the exact FC path is compute-bound on the DSP and slower than the CPU,
by 1.1× to 4×.** A hand-scheduled loop could remove perhaps a third of the
packets, which does not change the sign. The scalar alternative (HVX
computes isum and s, the scalar core runs `sffma` chains) is no better:

* ISS: 12.4 pcycles per step single-threaded with 4 chains, latency-bound;
* at best about 0.5 packet per step, and it shares issue with HVX.

Both are ISS numbers, which contract §12 keeps out of the gates. The
microbench of step A5 measures the real one.

### 0.4 The address space does not hold the set

`htp_compute_ops.cpp:3348` and doc 46 §41 give the budget:

* 3840 MiB are mapped (15 × 256 MiB, the `fastrpc_mmap` ceiling);
* 3696 MiB of that hold the MoE `QS4CX_WH` weights;
* the loaded app gets about 100 MiB of DSP heap before `AEE_ENOMEMORY`;
* the 2026-09-29 `prof_RQ.log` shows all 15 chunks mapped.

The free room is therefore:

* 144 MiB of arena;
* about 100 MiB of heap, less the 48 MiB ATTN_M1 KV cache.

In total that is **≈ 196 MiB**. The FC weights (243 MiB) plus lm_head
(140 MiB) need **383 MiB**. The cycle-20 comment ("≈ 360 MB of arena, no
residency wall") left the MoE arena out of the count.

**Consequence.** "All FCs + lm_head resident" is impossible in one PD
without a user decision (§3.4). Every option needs a new device reading.
None of this touches Part A: the shadow registers one weight slice at a
time.

## 1. Goal and gate

**Goal.** Every M=1 decode FC, the router and the lm_head + argmax have a
spec that equals the shipped CPU, and a DSP kernel that equals the spec. The
issue's end state is:

* decode resident end to end at `calls/token=1.00`;
* MoE dumps `bit_identical=1` against A;
* nll lines equal;
* text ≡ A.

That end state is Part B, behind D.

| # | check | where | pass |
|---|---|---|---|
| G0 | spec == shipped CPU | new ARM gtests in `unittest_nntrainer_cpu_backend` (`Q4GemvCpuOrder.*`, `Q8QuantCpuOrder.*`, `SwigluCpuOrder.*`, `SgemvNCpuOrder.*`, `ExpfBionic.*`), run on the device | `bad=0`: 2 000 random rows per FC shape (2048→6144 / 2048 / 2560, 7168→2048, 2048→7168) with activations over 2⁻²⁰ … 2²⁰ and all-zero blocks; the replayed shadow inputs (A4); `expf` port == libm over **all 2³² inputs** |
| G1 | kernel == spec on silicon | `unittest_hvx_softmax --gtest_filter='HvxM1Ops.*'` + the new `HvxFcQ4.*` | `bad=0` for the FC kernel (all five shapes, random and near-tie mutants), the quantizer, `router_cpu_det`, `swiglu_cpu_det`, `argmax` |
| G2 | **op shadow: DSP == CPU on real activations** | `dev/fc-shadow` (A4), prompt 512, G = 8, forced on A's tokens | every record bit-equal: 24 × {in_proj or qkv, out_proj or o} + 2 × {up, gate, down} + lm_head (8 × 16 k-row slices) per step; ADD (48 per step) and ROUTER (22 per step: logits, top-4 ids, weights) |
| G3 | exact-GEMV rate on silicon | `HvxFcQ4.Rate` (A5): 6 lanes, VTCM feed with `src_bypass=1`, the 402 MB set's shapes | reported, not gated: µs per call per shape, GB/s, ms/token projected. It feeds D |
| G4 | address space | the `[HTP] arena` lines + a heap probe of the loaded app (`mem_probe_dsp_heap`) at the end of A's prefill | reported: free MiB in the arena and in the heap. It feeds D |
| standing | prefill; text | A-sitting speed cells | prefill ≥ −5 % of A (the shadow run S only); text of S ≡ A (the shadow must be inert) |

Part B gates (after D):

* `calls/token=1.00`;
* dumps `bit_identical=1`;
* every `[PPL] decode` nll equal to A to 17 digits;
* text ≡ A 8/8;
* prefill ≥ −5 %.

The decode tok/s against A is D's own condition (§3.4).

## 2. Where it lives

**New specs** (header-only C99, `nntrainer/tensor/`, beside `m1_ops_det.h`),
one f32 rounding per step, `fmaf` where the CPU fuses:

* `q4_gemv_cpu_det.h`:
  * `q8_0_quant_cpu_det`: integer `div127_rn`, #164's `recip_rn`, an
    `f32→f16` RN helper, RN-to-int;
  * `q4_gemv_cpu_det`: §0.1's chain.
* In `m1_ops_det.h`:
  * `m1_router_cpu_det` replaces `m1_router_topk_det` (`:274`). It has the
    fused 2048-step chain, `expf_bionic_det` (a C port of bionic's expf,
    pinned by G0), `recip_rn`, the wsum order and the unchanged tie rule;
  * `swiglu_cpu_det`: §0.2's op list, with `div_rn` built from `recip_rn`'s
    integer method;
  * `argmax_first`.

**DSP.**

* New `hvx/hvx_q4_gemv_f32.{c,h}`: §0.3's kernel on native IEEE sf
  instructions, one 32-column group per pool unit.
* The weight layout is `Q4M1`, reordered from the ARM's `q4_0x4` at load and
  bit-preserving: per 32 columns × 32-block, 512 B of nibbles plus 64 B of
  f16 d. That is 18 B per 32 weights, the same bytes.
* `hvx_m1_ops_f32.c:172`: `hvx_router_topk_f32` moves to the CPU order. The
  2048-step chain goes on the scalar `sffma` (32 independent chains), so
  there is no HVX FMA emulation.
* `hvx_swiglu_f32.c` gains a `swiglu_cpu` variant. The MoE's own
  `swiglu_det` stays: A is its reference.

**IDL** `test/htp/nntr_hvx.idl`. Test entries go after
`router_topk_det_f32` (`:615`), additive, followed by `generate_stub.sh` and
a skel rebuild:

* `weight_register_q4m1(K, N, bytes) → h`;
* `weight_release_q4m1(h)`;
* `fc_q4m1_f32(h, x, y)`;
* `q8_quant_f32`;
* `swiglu_cpu_f32`;
* `argmax_f32`.

`router_topk_det_f32` keeps its name and follows the new spec.

**ARM.**

* `HtpComputeOps` (`htp_compute_ops.cpp`): a `q4m1` registration, heap or
  arena room, with `registerRm`'s room logic (`:3348`), plus test wrappers.
* `accelerates_q4_0_at_m1` stays false. Part A changes no dispatch.
* `set_decode_graph_desc` (`:1314`) is untouched until Part B.

**Consumers checked.**

* **Quantizer / loader:** `nntr_quantize_stream`'s format tag and the loader
  check do **not** change. The reorder happens at registration and the file
  keeps `q40-qs4cx-wh`.
* **Profile:** `NNTR_HTP_PROFILE`'s graph line `pcyc/op` (`:796-812`) prints
  the new kinds by name in Part B. The stage tables and
  `tools/htp_fc_report.py` do not change in Part A (test entries only). In
  Part B they gain a `q4m1` FC row.
* **Graph:**
  * `htp_graph_desc.h` kinds `:57-67` and the FC rule `:403`;
  * `hexkl_graph.c:235` table (FC / DENSE_FFN / LM_HEAD NULL);
  * `hexkl_graph_forward` `:448`.
  All three are Part B only.
* **Shadow branch** (`dev/fc-shadow`, never merged): it follows
  `dev/norm-shadow` @ `5ead9b7f` (`htp_decode_hook.h` record writer).

## 3. Design

### 3.1 FC: exact chain on the HVX, native sf instructions

§0.3's kernel.

* The integer dot is order-free. The split product is exact.
* The only rounding is the Boldo–Melquiond final add, which proves RN(a + b + c).
* The native `vadd/vsub/vmpy(.sf)` instructions are preferred over the
  intrinsics:
  * 10 % fewer packets;
  * TwoSum's exactness then rests on one IEEE instruction per add, not on a
    `qf32` add plus a convert.

  The `Q6_Vsf_*` build is kept as the host-emulated twin, because `hvx_emu`
  implements the intrinsics.
* **Near-tie mutants** are in the host check and G1:
  * sums that land exactly on a midpoint;
  * P2 of either sign at a midpoint;
  * cancellation to zero;
  * d_w negative;
  * d_a = 0.
* The scalar parts are exact integer helpers, run once per 32-block on the
  calling thread: the quantizer's `div127_rn` / `recip_rn`, the f16 RN
  store, and RN-to-int.

**Rejected alternatives:**

* **(ii) of the issue body** (requant to `QS4CX_WH`, reuse the WH GEMV). It
  moves the weights, so it fails the bit-preserving rule by construction.
* **Scalar `sffma` tail** (HVX makes isum and s, the scalar core chains).
  It needs ≥ 0.5 packet per step on the shared issue and is latency-bound in
  the ISS (§0.3). It is kept only as the router's method, where there are
  32 chains × 2048 steps.

### 3.2 Router, SwiGLU, argmax

* **Router:** HVX computes nothing new. 32 scalar `sffma` chains run in
  j order, then 32 × `expf_bionic_det`, `recip_rn`, and the sort and weights
  of §0.2.
* **SwiGLU:** HVX on native sf instructions. The one fused step `fx` uses
  the same exact FMA emulation (Dekker split, then Boldo–Melquiond), and the
  division uses `div_rn`. There are 14 336 elements per token.
* **Argmax:** an HVX max plus the index, with the lowest index on ties.

The DSP then returns one token id, and returns the logits only under
`NNTR_PPL_DECODE` / the shadow.

### 3.3 Verification per op (the shadow)

`dev/fc-shadow` sits on the PR branch. It is measurement only and never
merged.

* `NNTR_FC_SHADOW=<file>` runs `fc_q4m1_f32` on the **same input** at each
  CPU FC call and appends a record: tag 3, op id, pos, K, N, the input, the
  CPU output and the DSP output.
* The weight is registered once as `q4m1` on the heap. The lm_head goes in
  16 k-row slices of 18 MB, registered, run and released per slice.
* Tag 4 is ADD: the CPU `add_i` against the `hvx_scale_add` test path.
* Tag 5 is the router: the CPU logits, ids and weights against
  `router_topk_det_f32`.

Nothing is resident. The CPU path's outputs stay in use, so the text and nll
of S must equal A's (the inertness check).

### 3.4 Decision D (user), after the Part A sitting

G3 and G4 give the two numbers.

* **D-speed.** With the exact FC at X ms/token against the CPU's 7.4, is a
  resident decode that is slower than A acceptable as the "NPU end to end"
  product? The plan's expectation is X ≥ 7.4 (§0.3).
* **D-space.** Where do the 383 MiB go? Candidates:
  * (a) a second cDSP session / PD for the FC set and lm_head, with its own
    4 GiB space. That is ≥ 2 queues per layer, not 1 call/token. It is
    unverified that this app can reserve a second session;
  * (b) lm_head only (140 MiB into the 144 MiB of arena, if staging leaves
    room), with the FCs on the CPU. That is 49 calls/token with dspqueue;
  * (c) no residency: keep the CPU FCs and stop at the specs and shadows
    (Part A is still useful: it makes ROUTER_TOPK exact).

  Recommendation, if G3 confirms the estimate: (c) for the FCs, with the
  router fix landed, because both walls say the CPU is the faster exact FC
  engine on this phone.

## 4. Steps

**Part A** (one PR, host-gated through rung 3, then one sitting)

* **A1. Specs + host checks.** Write the five specs of §2. In
  `m1_ops_host_check.c`, add independent references:
  * written from §0's disassembly with `fmaf` / `/` / `expf` as the host
    has them. The host `expf` is glibc, so the host checks only the port's
    self-consistency, and G0 pins it to bionic;
  * the midpoint mutants of §3.1;
  * one mutant per rejected order: non-fused chain, 4 partial sums,
    `amax * (1/127)`, the wsum order `((s0+s1)+s2)+s3`.

  Gate: rung 1. `ALL CHECKS PASS` plus `Q4 GEMV CPU-ORDER OK mutants=N/N`.
* **A2. Kernels on `hvx_emu`** (intrinsic build). Gate: `Q4 GEMV
  BIT-IDENTICAL` over the five shapes. Router, swiglu and argmax likewise.
  `graph_host_check` stays green. PR 1's ROUTER_TOPK references follow the
  new spec, and the E2E fixtures' fwd SNR lines are recorded.
* **A3. IDL test entries + native-sf build.** Gate: rung 2,
  `test/htp/build.sh` printing `UNDEFINED SYMBOLS OK`, and the skel md5.
* **A4. Shadow branch** `dev/fc-shadow` (§3.3). Gate: on the host
  (`-Dhtp-inproc`), an hd64 fixture run writes tag 3/4/5 records with every
  record equal, and S's tokens equal off's 8/8.
* **A5. Device gtests:**
  * G0 in `unittest_nntrainer_cpu_backend`, which calls the exported
    `gemm_q4_0`, `nntr_quantize_row_q8_0`, `swiglu`, `cblas_sgemv` and libm
    `expf`;
  * G1 in `HvxFcQ4.*`;
  * G3's `HvxFcQ4.Rate`.

  Gate: rung 3, both binaries built, md5s recorded. Run `llvm-objdump` on
  the new `libnntrainer.so` and check that `nntr_gemv_q4_0_4x8_q8_0` still
  shows the single `fmla` chain, the quantizer the two `fdiv`s, and
  `cblas_sgemv` still dispatches to `sgemv_n`. This is G0's premise.
* **A6. Device sitting (unavoidable; the orchestrator runs the handoff,
  `docs/measurements/132-pr2-exact-fc.md`), about 60 min.** One binary set:
  the shadow branch, which is the PR diff plus the inert measurement commit.

  | variant | what | runs |
  |---|---|---|
  | A | unchanged reference, switch off | full E2E, prompt 512, G = 64 / 512 / 1024 × 2, mirrored `A S S A`; 8 prompts at G = 256 with `NNTR_PPL_DECODE` |
  | S | A + `NNTR_FC_SHADOW` | same cells (inertness: text and nll ≡ A); prompt 512, G = 8 forced on A's tokens for G2 |

  Also in the sitting:
  * gtests G0, G1 and G3 (`Rate`) with a zone0 log;
  * G4's heap probe on A after prefill.

  Stop rules:
  * G1 fails on normal rows: the skel is stale (`0x8000040e`) or the native
    sf instruction is not RN on silicon. Stop, and fall back to the
    intrinsic build in the next sitting;
  * G0 fails: the CPU is not what §0 read. Re-read that set.
* **A7. Fold.** G0–G2 ✓ closes Part A. The tables of G3 and G4 go to the
  user as decision D.

**Part B** (after D, a separate PR; sketched, not planned). Its content
depends on D:

1. Register the FC set in the chosen space.
2. Put the FC / DENSE_FFN / LM_HEAD kernels in `hexkl_graph.c:235`, and turn
   the FC op record (`htp_graph_desc.h:403`) into per-weight handles:
   q/k/v stay 3 handles and one input quantization.
3. The last op returns the token id.
4. Gates: `calls/token=1.00` on the host fixtures and on LFM2.5, dumps
   `bit_identical=1`, nll equal, text ≡ A.
5. Sitting: A / F (everything) / F − LM_HEAD.

## 5. Risks (host vs device)

* **Native sf on silicon.** The prototype's exactness rests on each sf
  add/sub being IEEE RN. The ISS agrees, and rule 37 says silicon agrees on
  normal-range rows, but only for the `qf32`-lowered intrinsics.
  * G1's near-tie mutants test it directly.
  * The intrinsic build is the fallback: 12 % slower, and already exact on
    the ISS.
* **The cost estimate is ISS + a calibration from another kernel**
  (plan 146).
  * G3 measures it with 6 lanes and the DMA feed in one sitting, and reports
    pcycles next to µs.
  * DVFS and thermal drift show in the zone0 log and the mirrored A S S A
    order.
* **Address space.**
  * G4 reads the loaded app's real room.
  * The shadow's slices stay ≤ 18 MB, so Part A cannot hit `AEE_ENOMEMORY`
    (it registers and releases per slice).
* **Stale skel.** The IDL grows, so an old skel fails loudly (`0x8000040e`).
  The md5 is in every log.
* **bionic `expf` may change with a phone update.** G0's exhaustive
  comparison is re-run in every sitting that uses the router.
* **Blocked-twin fallback.** If the lm_head's blocked twin ever fails to
  allocate, the CPU switches to the rowwise kernel, which has a different
  order. G2's lm_head records catch that on the day it happens.
* **ISS vs silicon on tiny values** (rule 37).
  * Q8 blocks with d_a = 0 and f16-subnormal d are in G1's mutants.
  * Real activations may still produce f16-subnormal d_a (amax < 7.8e-3),
    and G2 shows whether they do.

## 6. Docs to update

* **`docs/htp_moe/BENCHMARK.md`**:
  * a #132 PR 2 side table with the G0 / G1 lines;
  * G2 record counts (FC / ADD / ROUTER, equal / total);
  * G3's µs per call per shape and ms/token projected;
  * G4's free MiB;
  * A / S rows (G 64 / 512 / 1024, prefill, text ≡, nll ≡).
* **`docs/htp_moe/LEDGER.md`**:
  * a new rule, if G3 confirms it: *the CPU's Q4_0 GEMV is one fused chain
    per column over 32-blocks; the HTP reproduces it only with an emulated
    FMA at N packets per 32 × 32, X ms/token for the set, slower than the
    CPU*;
  * a new rule, from G4: *the DSP address space holds MoE + ≈ 196 MiB; the
    Q4_0 FCs + lm_head (383 MiB) do not fit in one PD*;
  * ⑨ and ㉓: PR 2's status and decision D;
  * §2: the PR 1 router is not CPU-exact (§0.2), corrected by this PR.

**What remains for NPU end to end after this issue**, separate items:

* ATTN_M1 bit-identical speed: 1737 µs/layer warm at pos 1023 against the
  ≈ 145 µs a 50 tok/s budget allows (#146 re-scope, LEDGER ㉗, rule 45);
* #164 (RMSNORM / QK_NORM, including the final norm) must land for the text
  to be identical;
* decision D above.
