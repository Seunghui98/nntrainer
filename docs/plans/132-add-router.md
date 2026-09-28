# 132 — PR 1: ADD and ROUTER_TOPK resident in the per-token entry

Issue: dlwlzzero/nntrainer#132 (p1; LEDGER ㉓ / ⑨). **This plan covers PR 1
only**, the ADD and ROUTER_TOPK slots. PR 2 (the FC, DENSE_FFN and LM_HEAD slots,
51 → 1) is on hold behind #141's dspqueue microbench (user decision, 2026-09-28)
and is not planned here. Read against `htp_moe` @ `049902af` (cycle 20). Builds on
plan 130 (the hook pattern: `htp_decode_hook.h`, `decode_op_fp32`,
`graph_set_param`, the stretch tables). Accuracy evidence:
`docs/measurements/136-forward-text.md` on `origin/htp/136-forward-text`. Every
resident kind is last-bit exact on silicon, and the quantized pipeline still
flips text, so the accuracy gate is #134's decode PPL (`NNTR_PPL_DECODE`, plan
134).

**Correction to the issue's ladder (verified on the builder).** Adding ADD takes
the count to **73**, not 71, and adding ROUTER_TOPK takes it to **51**, not 49.
The cycle-20 table drops the two DENSE_FFN ops of layers 0–1 in those two rows
but keeps them in the "+ FC → 3" row. I built the lists with
`htp_graph_lfm2_build` (`htp_graph_desc.h:497`) and counted the resident
stretches, which is what `hexkl_graph_forward` (`hexkl_graph.c:378`) makes of a
list:

| mask | LFM2.5 (24 layers) | hd8 (`CAC`, head_dim 8) | hd64 (`CAC`) | lfm25 fixture (`CCACAC`) |
|---|---|---|---|---|
| `MOE` | 22 | — | — | 4 |
| six kinds (B) / hd8's `MOE,RMSNORM,CONV1D_GATE` | **95** | 11 | 12 | 23 |
| + ADD | **73** | 9 | 10 | 19 |
| + ADD + ROUTER_TOPK (D) | **51** | 7 | 8 | 15 |
| + FC (PR 2) | 3 | — | — | — |
| everything | 1 | — | — | — |

Per MoE layer the only non-resident ops left in D are the two FCs. Per dense
layer they are the two FCs plus DENSE_FFN. The tail keeps LM_HEAD. So
22 × 2 + 2 × 3 + 1 = 51.

## 1. Goal and gate

**Goal (issue, PR 1 share).** Make ADD and ROUTER_TOPK resident so that on
LFM2.5 the calls per token fall from 95 to 73 (ADD) and then to 51 (ADD +
ROUTER_TOPK). The router's routing must reach the MOE op inside the same
stretch with no ARM round trip. The PR is host-gated only and ends in
`state:review`.

**Gates of the PR (all must print; rungs 0–3 of `hexagon-gates`):**

1. **Kernel versus spec.** `test/htp/host/m1_ops_host_check.c` prints
   `ROUTER TOPK BIT-IDENTICAL`. It compares `hvx_router_topk_f32` (HVX source on
   `hvx_emu`) with `m1_router_topk_det` (`nntrainer/tensor/m1_ops_det.h`) using
   `memcmp` on logits, selection order and routing weights. Shapes: K 2048 /
   E 32 / top 4, K 128 / E 4 / top 2 and K 64 / E 4 / top 2. The rows are random,
   plus **exact-tie** rows (two identical W columns and equal bias, so the
   lowest index must win) and **1-ulp near-tie** rows. The same file prints
   `ROUTER SPEC vs CPU rows=100000 set_flips=<f> max_gap_at_flip=<g>`. There the
   spec is compared with a verbatim copy of `buildExpertAssignments`
   (`lfm2_moe_layer.cpp:271-313`, `std::exp`, a true divide, the new comparator
   of §3.3) on LFM2.5-shaped random rows. The gate: every flip sits at a CPU
   4th/5th score gap below 1e-4, and exact ties select identically. This check
   pins the known hazard. It proves that the spec and the CPU can disagree only
   at a near-tie, never through a logic or tie-rule difference.
2. **`graph_host_check`** (`run_host_checks.sh`) prints `GRAPH CHECKS PASS` with
   the following additions:
   * The LFM2.5 list validates with `ALL_KINDS | ADD | ROUTER_TOPK`, and
     `hexkl_graph_resident_kinds()` returns exactly that set.
   * There are three new mutations, each with its own code: "ADD resident, a
     RMSNORM not" → `AEE_ENOTALLOWED`; "ROUTER_TOPK resident, its MOE not" →
     `AEE_ENOTALLOWED`; "ADD out_slot 1" → `AEE_EINVALIDFORMAT`.
   * Two new bit-identical stretches on hd64 with the MoE stand-in. The first is
     `[ADD RMSNORM]` of layer 0, after an op-0 RMSNORM stretch has seeded slot 0.
     It must equal `m1_rmsnorm_det(x + a)`, and slot 0 must equal `x + a` byte for
     byte. The second is `[ADD RMSNORM ROUTER_TOPK MOE ADD RMSNORM]` from layer
     1's first ADD to layer 2's operator norm. It must equal the scalar
     composition `rmsnorm_det(h2)` with `h2 = h + standin(n, router_topk_det(n))`
     and `h = s0 + a`.
   * `resume_at` is the next FC. `op_pcycles` is non-zero for exactly the ops
     that ran.
   * One forward negative: a ROUTER_TOPK op with no weights bound →
     `AEE_EBADSTATE`.
   * The check prints `GRAPH STRETCH BIT-IDENTICAL: … ADD+RMSNORM
     ADD+RMSNORM+ROUTER_TOPK+MOE+ADD+RMSNORM`.
3. **`run_inproc_e2e.sh` → `INPROC E2E PASS`.** Every present line must stay
   unchanged: the switch-off golden stays `bit_identical=1`, B-style runs stay at
   11 and 12, and no golden is regenerated. The new lines, with logits files
   only (§3.5):
   * `E2E fwd tiny kinds=MOE,RMSNORM,CONV1D_GATE,ADD calls/token=9` and `E2E eval
     add==noadd-tiny … bit_identical=1` against the existing `fwd-tiny` run. ADD
     is IEEE `a + b` on both sides (§3.1), so the logits may not move by a bit.
   * `E2E fwd hd64 kinds=<six>,ADD calls/token=10` with `bit_identical=1` against
     the six-kind hd64 run.
   * `E2E fwd tiny kinds=…,ADD,ROUTER_TOPK calls/token=7` and `E2E fwd hd64
     kinds=<six>,ADD,ROUTER_TOPK calls/token=8`. Each gets `min_snr_db ≥ 30`
     against its switch-off run (present floor, logits) and `E2E tokens fwd==off
     8/8 expected_mismatch=0` (plan 130 §3.5's policy).
   * `E2E fwd tiny ADD-without-RMSNORM refused: AEE_ENOTALLOWED`, through the
     model's real throw.
   * If #136's `lfm2_moe_tiny_lfm25` fixture (`b4a999bf`) has landed on
     `htp_moe` before this PR: `E2E fwd lfm25 kinds=<six>,ADD,ROUTER_TOPK
     calls/token=15` with the same SNR and token lines. If it has not landed, the
     PR says so and the line follows it.
4. **Standing rung 1–3 set.** `ninja -C build`; `*Lfm2Moe*` 6/6 (this proves the
   comparator change of §3.3 moved no reference logit); `*qs4cx*`;
   `run_host_checks.sh` (`ALL CHECKS PASS`, `M1 OPS BIT-IDENTICAL`, `ATTN M1
   BIT-IDENTICAL`); `tools/htp_syntax_check.sh`; `clang-format-14` on changed
   lines; `generate_stub.sh`; `test/htp/build.sh` (`UNDEFINED SYMBOLS OK`);
   `build_android.sh --htp`; `readelf -d`; the `strings … NNTR_HTP_FORWARD_KINDS`
   count; `ndk-build unittest_hvx_softmax unittest_hvx_attn`. Skel, app and gtest
   md5s go in the PR.
5. **Standing E2E gates**, read in the orchestrator's combined sitting (§4
   step 7), not in this PR:
   * prefill of every variant ≥ −5 % of A;
   * accuracy by the 2026-09-28 rule: text column, `NNTR_PPL_DECODE` against A,
     user approval. "Text identical to the CPU run" is recorded but cannot hold
     for any resident kind (#136).

## 2. Where it lives

`path:line` at `049902af`.

**Shared header and DSP:**

| file | change |
|---|---|
| `nntrainer/tensor/htp_backend/htp_graph_desc.h` | `HTP_GRAPH_PARAM_ROUTER_W`, `HTP_GRAPH_PARAM_ROUTER_BIAS` appended before `_N` (`:139-145`; lengths `K × n_experts` and `n_experts`). The validator (`:296-446`) gains three rules. (1) A resident ADD needs every RMSNORM resident, else `NOTALLOWED` (slot 0 is the DSP's residual only if op 0 seeds it and no CPU norm re-seeds it, §3.1). (2) A resident ROUTER_TOPK needs the next op (its MOE) resident, else `NOTALLOWED` (the routing has no other consumer). (3) Every ADD has `out_slot == 0` and `in_slot != 0`, else `INVALIDFORMAT` (the kernel's implicit second operand, `:155`). The doc comment (`:280-295`) names them. **No wire change**: `HTP_GRAPH_VERSION` stays 2, and the router's shape (`K`, `N = n_experts`, `top_k`) is already in the record (`:373-378`) |
| `nntrainer/tensor/htp_backend/hmx/hexkl_graph.h` | `hexkl_graph` (`:62-75`) gains `uint32_t route_idx[32], route_cnt[32]; float route_w[32];` (one routing record, rewritten by every router op). `hexkl_graph_routing`'s comment (`:50-53`) now says that a resident ROUTER_TOPK fills it |
| `nntrainer/tensor/htp_backend/hmx/hexkl_graph.c` | Two table slots (`:198-199`). `graph_op_add`: `hvx_scale_add_rows_f32(out /* slot 0 */, in, 1.0f, N)` (`hvx_scale_add_f32.h:36`, reused as is). `graph_op_router_topk`: `hvx_router_topk_f32(in, W, bias, K, E, top_k, out /* logits */, sel, w)`, which then fills `route_*` in expert order and points `call->routing` at them (§3.2). `graph_param_len` (`:303-316`) takes the two new `which`. `set_param` (`:318-357`) copies ROUTER_W into a zero-padded `[K][32]` `memalign(128)` buffer. The address-space note (`:13-21`) gains + 5.5 MiB (22 × 2048 × 32 × 4 B). `graph_op_moe` is **unchanged**: it already consumes `call->routing` |
| `nntrainer/tensor/htp_backend/hvx/hvx_m1_ops_f32.{c,h}` | `hvx_router_topk_f32` (§3.2) in the #82 small-ops file. That file is already in `test/htp/build.sh` `SRCS`, the inproc meson list and the graph check's compile line, so no build list changes |
| `nntrainer/tensor/m1_ops_det.h` | `m1_router_topk_det` (the spec, §3.2). It includes `swiglu_det.h` for `swiglu_det_exp` / `swiglu_det_recip` (`:113`, `:170`), as `attn_m1_det.h:71` does |
| `test/htp/nntr_hvx.idl:603` | One test method after `graph_set_param`: `router_topk_det_f32(in uint32 top_k, in sequence<float> x, in sequence<float> w, in sequence<float> bias, rout sequence<float> logits, rout sequence<uint32> sel, rout sequence<float> weight)`. It is additive. `graph_set_param`'s signature is unchanged; its comment lists the two new `which` |
| `test/htp/nntr_hvx_small_ops.c` | `nntr_hvx_router_topk_det_f32`. A bad shape returns `AEE_EINVALIDFORMAT`, as `:34-96` do |
| `test/htp/host/m1_ops_host_check.c`, `graph_host_check.c` | §1 gates 1–2. The shapes (`:116-126`) and the mutation table (`:250-270`) gain the new cases. The stretch half (`check_stretches`) gains the two stretches |
| `test/unittest/unittest_hvx_softmax.cpp` | `HvxM1Ops.RouterTopkMatchesDetBitExact` (beside `:416`), with the shapes and tie rows of gate 1 and `bad=` per output |

**ARM side:**

| file | change |
|---|---|
| `nntrainer/tensor/htp_backend/htp_compute_ops.cpp` | `kArmKinds` (`:1283-1286`) gains ADD and ROUTER_TOPK. `set_decode_graph_desc` (`:1288`) gains an ARM rule beside the QK_NORM one (`:1307-1315`): a resident ROUTER_TOPK needs ADD resident, because the MoE layer's hook is then always mid-stretch (§3.4); otherwise `invalid_argument`. `decode_op_fp32` (`:1415-1540`): the cases ADD (no parameter) and ROUTER_TOPK (bind `ROUTER_W` from `param`, `ROUTER_BIAS` from `state`, once) are added, and the per-kind invoke code becomes **one role dispatch** (§3.4). `gemm_qs4cx_moe_layer_fp32`'s M==1 path (`:1228-1236`): if the MOE op starts a longer stretch, it keeps the activation and the routing pending; if the op is mid-stretch, it throws. `invokeForward` (`:2341`): the level-3 repeat (`:2372`), `dumpMoeCall` (`:2420`) and `addInvokeMoeLayer` (`:2435`) apply only to the **sole-MOE** stretch `[op, op+1)`. An ADD in a stretch mutates slot 0, so a repeat would add twice |
| `Applications/CausalLM/layers/htp_decode_hook.h` | `htpDecodeAdd(pos, addend, out, W)` and `htpDecodeRouter(pos, x, K, gate_w, E, bias)`. The non-HEXKL branch gains `HTP_OP_ADD 6`, `HTP_OP_ROUTER_TOPK 7` (`:47-50`) |
| `Applications/CausalLM/layers/residual_add.{h,cpp}` (new) | A two-input layer `residual_add`: `out = in0; out += in1`, exactly `AdditionLayer::incremental_forwarding` (`nntrainer/layers/addition_layer.cpp:46-82`). At a decode row (FP32, `to - from == 1`) it calls `htpDecodeAdd(from, in1, out, W)` first and skips on 1. Modeled on `custom_multiply.{h,cpp}`. Listed like `qkv_layer` in `layers/meson.build`, `Applications/CausalLM/meson.build:41`, and both `jni/Android.mk` blocks (`:103`, `:227`) |
| `Applications/CausalLM/models/transformer.{h,cpp}` | A protected `std::string RESIDUAL_ADD_TYPE = "addition"`. `createTransformerDecoderBlock`'s two `createLayer("addition", …)` calls (`:654-657`, `:668-671`) read it. Every other model is byte for byte unchanged |
| `Applications/CausalLM/models/lfm2/lfm2_causallm.cpp` | Under `htpForwardSwitch()` (`:66-69`), `RESIDUAL_ADD_TYPE = "residual_add"`. The `conv_block` form's two `Tensor::add` calls (`:157`, `:165`) become `createLayer(RESIDUAL_ADD_TYPE, …)({input, block_out})`, `({residual_b, ffn_out_b})`. The operand order is residual first, addend second. `registerCustomLayers` (`:237-260`) registers `ResidualAddLayer` |
| `Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.cpp` | In `incremental_forwarding` (`:718`), at `total_tokens == 1`, the call `htpDecodeRouter(from, input, hidden, gate_weights, hidden × num_experts, expert_bias)` goes before the router `dot` (`:774`). On 1 the layer returns with its output untouched (§3.4). `buildExpertAssignments`' comparator (`:294-298`) becomes the total order `a.first > b.first || (a.first == b.first && a.second < b.second)` (§3.3) |
| `Applications/CausalLM/models/lfm2_moe/lfm2_moe_causallm.cpp:88-94` | The default mask stays the six kinds. D is `NNTR_HTP_FORWARD_KINDS=…,ADD,ROUTER_TOPK`, so #130/#136 recipes keep their meaning. The comment says so |
| `test/htp/host/run_inproc_e2e.sh` | The runs and lines of §1 gate 3. The ADD runs compare logits only (§3.5) |

**Consumers of a changed contract:**

* **IDL and both stubs.** There is one additive test method. `generate_stub.sh`
  and `test/htp/build.sh` run together (rule 3), and the inproc build
  regenerates. A stale skel refuses a resident ADD / ROUTER_TOPK at
  `graph_init` with `AEE_ECLASSNOTSUPPORT` (named by `graphErr`), and the new
  gtest with `0x8000040E`.
* **`HtpComputeOps`**, as above.
* **Quantizer format tag (`nntr_quantize_stream`) and the loader check:
  unchanged.** The router weights are already FP32 in the file
  (`lfm2_moe_layer.cpp:193`, "Always kept FP32") and travel as a
  `graph_set_param` f32 copy. No weight bytes or layout change.
* **`NNTR_HTP_PROFILE`.** The `graph:` per-kind pcycle table is kind-indexed
  (`addGraphOp`, `:234-238`), so `ADD=` and `ROUTER_TOPK=` appear with no
  change. The M==1 MoE row keeps only sole-MOE stretches (C stays comparable to
  A). `tools/htp_fc_report.py` is unaffected.

## 3. Design

### 3.1 ADD: the residual lives in DSP slot 0 across calls

*Chosen.* The builder already routes ADD `slot 2 + slot 0 → slot 0` and every
RMSNORM `slot 0 → 1` (`htp_graph_desc.h:531-575`). The slots are the graph's
heap (`hexkl_graph.c:269-275`) and persist between `forward` calls. With
RMSNORM resident, op 0 (layer 0's operator norm) is always a stretch start with
`in_slot 0`, so each token's first call copies the embedding row into slot 0
(`hexkl_graph.c:430-432`). From then on only ADD writes slot 0, and no
RMSNORM stretch start re-seeds it, because every other RMSNORM follows an ADD.
So slot 0 **is** the residual stream on the DSP. The CPU's residual tensors go
stale and nothing reads them.

The validator rule "ADD ⇒ every RMSNORM resident" is what makes this true. It
also means an ADD is never the last op of a stretch: every ADD is followed by
a RMSNORM.

The stretch crosses the layer boundary as `[MOE ADD RMSNORM(l+1 operator
norm)]` in the 73 set and as `[ADD RMSNORM ROUTER_TOPK MOE ADD RMSNORM(l+1)]` in
D. The next layer's `rms_norm` hook is the stretch's last op, so it runs the
call and writes the normed row that layer l+1's CPU FC reads. The tail ends the
same way, at the final norm, before LM_HEAD.

The ADD kernel is `hvx_scale_add_rows_f32(slot0, in, 1.0f, N)`. `x · 1.0f` is
exact and the add is one IEEE rounding, so it equals the CPU's `copy` +
`add_i` bit for bit. That is why gate 3 can demand `bit_identical=1` for "+
ADD" against the run without it.

*Rejected: hooking the core `AdditionLayer`* (`nntrainer/layers/addition_layer.cpp`).
The core layer is shared by every model, would need an HTP kind constant, and
is outside the supervision scope. A CausalLM `residual_add` layer used only
under `NNTR_HTP_FORWARD` leaves the switch-off model byte for byte today's.
The switch-on prefill runs the same `copy` + `add_i`, so the prefill gate reads
it.

### 3.2 ROUTER_TOPK: an FP32 GEMV, a `_det` sigmoid and a deterministic top-k, feeding MOE in the same call

*The spec* (`m1_router_topk_det`, normative; HVX implements it operation for
operation):

1. `logit[e]`. Four partial sums `a_j` over `k ≡ j (mod 4)`, sequential in `k`,
   with `a_j = a_j + x[k]·W[k][e]` (multiply and add each rounded, no FMA). Then
   `logit = (a0 + a1) + (a2 + a3)`. On HVX with `E ≤ 32` a row `W[k][0..32)` is
   one vector, so four vector accumulators give exactly this order per lane.
2. `sig[e] = recip_det(1 + exp_det(0 − logit[e]))`: `swiglu_det_exp` /
   `swiglu_det_recip` and their HVX forms (`hvx_swiglu_det.h:110`, `:174`).
   There is no libm on the skel.
3. `score[e] = sig[e] + bias[e]`.
4. Selection, `top_k` rounds: the largest unchosen score wins, and the
   **lowest index wins a tie**. This is scalar on the DSP (32 × 4 compares).
5. `wsum = 0`, then `wsum += sig[sel[r]]` in selection order (the CPU's order,
   `:301-303`). `inv = recip_det(wsum + 1e-6f)`, and `weight[r] = (sig[sel[r]]
   · inv) · 1.0f`. These are LFM2's `NORM_TOPK_PROB`, scale and epsilon
   (`:127-128`). ponytail: the spec hard-codes these constants; another router
   needs a record word.
6. Output: logits to the op's out slot (slot 2; nothing reads it). Routing to
   `g->route_*` **in ascending expert order** with `row_index = 0` and
   `row_count ∈ {0, 1}`, which is `tryMoeLayerOnAccelerator`'s grouping
   (`:520-537`). Then `call->routing` points at it, and `graph_op_moe`
   consumes it with its existing checks.

The weights are bound once through `graph_set_param(op, ROUTER_W, W[K·E])`
and `(op, ROUTER_BIAS, bias[E])` from the MoE layer's first hook. `W` is
already row-major `[K][E]` (gate dim `(1,1,hidden,E)`, `:193-201`). The DSP
pads it to `[K][32]`. This is 256 KiB per layer and 5.5 MiB of DSP heap in
total, with no arena or DMA involved.

The weight read is 256 KiB per layer of DDR through direct HVX loads. The
kernel issues an `l2fetch` of the next 16 KiB block per block, as a hint only;
the bits do not depend on it. Doc 45 §3.2's "DMA hidden behind compute" is met
in that form; a VTCM feed is not worth it at 5.6 MiB/token.

*Rejected: the CPU's exact formula on the DSP* (`expf` and a true divide
statically linked). The logits' summation order of the CPU `dot` (BLAS / NEON)
cannot be reproduced anyway, and statically linking libm risks the #97
undefined-symbol class. Under the ⑨ decision ((c) not taken) the CPU keeps its
router and the DSP runs the `_det` spec. Decode PPL judges the difference.

### 3.3 The tie rule, and how the host pins the near-tie hazard

`std::partial_sort` with `a.first > b.first` (`:294-298`) leaves the order of
exact ties to the library (libc++ on Android, libstdc++ on the host). To make
"the tie-break matches the CPU exactly" true by construction, the CPU
comparator becomes the total order (score descending, then index ascending).
It changes the CPU path only at an exact tie. `*Lfm2Moe*` 6/6 and the
switch-off goldens prove that no fixture moved. Beyond ties, the spec and the
CPU differ in the last bits (summation order, `exp_det` against `std::exp`,
`recip_det` against a divide). They can therefore pick a different expert only
where the 4th and 5th scores are within that noise.

Gate 1's `ROUTER SPEC vs CPU` statistic makes that bound a checked fact rather
than an argument. The graph check and the device gtest pin DSP ≡ spec. What
remains is the numeric class #136 already measured, and the decode PPL column
reads it.

### 3.4 ARM: one role dispatch for every hook

Each hook (`decode_op_fp32`), after binding its op's parameters (and, as
today, re-seeding the conv state / asking for the KV seed), does the
following with `s = stretch_start_[op]` and `e = stretch_end_[op]`:

* **sole or last** (`op + 1 == e`): run `invokeForward(s, e, pos, act, out)`,
  where `act` is the hook's `in` if `s == op`, else the pending row. The pending
  row must belong to `s`, else throw. The pending routing is included if `s` is
  a MOE op. Then clear the pending state.
* **first** (`s == op`, `e > op + 1`): keep `in` as the pending row for `s` and
  return 1.
* **mid**: require the pending row to belong to `s` (else throw: the first hook
  never came), then return 1.

This subsumes plan 130's per-kind branches. With the D mask the roles are:
ADD first or mid; RMSNORM sole (op 0), mid (the FFN norm) or last (operator
and final norms); ROUTER_TOPK always mid, because ROUTER ⇒ ADD (ARM rule),
ADD ⇒ RMSNORM, and ROUTER ⇒ MOE (validator). The MoE layer's single hook
therefore only binds and returns 1, and the layer skips the router `dot`, the
top-k and the expert call.

In the 73 set the MoE path (`gemm_qs4cx_moe_layer_fp32`) is the **first** of
`[MOE ADD RMSNORM]`. It keeps `act` and the three routing vectors pending and
returns without writing `out`; the next layer's norm hook runs the call. It
takes the forward path only when `row_bound_` (the per-row flag op 0's hook
sets) or the stretch is sole. This way a row that binds its MoE handles as it
goes (a one-token prompt) stays on the CPU end to end.

*Rejected: a model-level decode loop* (plan 130 §3.2's deferred end state). At
51 calls every stretch is still bounded by CPU FCs, so the executor has to run
between stretches anyway. The loop becomes the right shape in PR 2, when a
stretch spans a whole layer.

### 3.5 The E2E comparisons

With ADD resident no decode MoE stretch is sole, so the switch-on runs dump no
MoE `_in` / `_out` files. `htp_dump_eval.py` then fails on "missing". The
harness copies the reference run's `logits_*.f32` into a fresh directory and
compares against that. A reference without a manifest is logits-only by
design (`htp_dump_eval.py:129-132`), so no Python change is needed. The
prefill call is untouched by the switch and stays covered by the switch-off
golden.

## 4. Steps

Rungs from `.claude/skills/hexagon-gates`. `run_host_checks.sh` is green at
`049902af` in this worktree (planning session: `ALL CHECKS PASS`, `GRAPH CHECKS
PASS`, `M1 OPS BIT-IDENTICAL`, `ATTN M1 BIT-IDENTICAL`). `run_inproc_e2e.sh`
was not re-run here, so the implementer runs it first.

1. **Spec and kernel.** `m1_router_topk_det`, `hvx_router_topk_f32`,
   `m1_ops_host_check.c` (gate 1, including `ROUTER SPEC vs CPU`). Gate: rung 1
   `run_host_checks.sh` → `ROUTER TOPK BIT-IDENTICAL`, `ALL CHECKS PASS`;
   `clang-format-14`.
2. **Header and graph.** The `htp_graph_desc.h` rules and params,
   `hexkl_graph.{c,h}` (two slots, the routing record, `set_param`), and
   `graph_host_check.c` (mutations and the two stretches). Gate: rung 1 → `GRAPH
   CHECKS PASS` with the new lines.
3. **IDL and skel test entry.** `router_topk_det_f32`, `nntr_hvx_small_ops.c`,
   `HvxM1Ops.RouterTopkMatchesDetBitExact`. Gate: `generate_stub.sh`; rung 2
   `test/htp/build.sh` → `UNDEFINED SYMBOLS OK`, skel md5.
4. **ARM.** `htp_compute_ops.cpp` (§3.4), `htp_decode_hook.h`, `residual_add`,
   the `RESIDUAL_ADD_TYPE` member, the LFM2 wiring, the MoE layer hook and the
   comparator. Gate: rung 1 `ninja -C build`, `*Lfm2Moe*` 6/6, `*qs4cx*`,
   `tools/htp_syntax_check.sh`.
5. **Harness.** The runs and lines of §1 gate 3. Gate: rung 1
   `run_inproc_e2e.sh` → `INPROC E2E PASS`, with every old line unchanged. The
   observed `min_snr_db` and the calls/token numbers go into the PR.
6. **Rung 3.** `build_android.sh --htp`, `readelf -d`, the `strings` count,
   `ndk-build unittest_hvx_softmax unittest_hvx_attn`, md5s. PR into
   `htp_moe`, `state:review`. **The PR ends here.**
7. **Device (unavoidable for any speed or accuracy verdict). No handoff of its
   own.** The orchestrator runs one combined sitting after this PR and #134.
   The design: one binary set, full E2E, prompt 512, G 64 / 512 / 1024,
   `NNTR_NUM_THREADS=8`, A first, ≤ 4 variants:
   * **A**: switch off (the reference);
   * **C**: `NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=MOE` (22 calls);
   * **B**: the six kinds (95 calls);
   * **D**: `…KINDS=MOE,RMSNORM,QK_NORM,ROPE,CONV1D_GATE,ATTN_M1,ADD,ROUTER_TOPK`
     (**51 calls**; the exit line must read `calls/token=51.00`).

   Accuracy for each variant is `NNTR_PPL_DECODE` against A's continuation
   (plan 134), the text column and user approval. Ride-alongs on that skel:
   `unittest_hvx_softmax --gtest_filter='HvxM1Ops.*'`, where the new router
   test must read `bad=0` and `0x8000040e` means a stale skel. If the
   orchestrator allows a fifth non-tok/s log, add D at `NNTR_HTP_PROFILE=2`,
   G=64, for the `ROUTER_TOPK=` and `ADD=` pcycles.

   **Expected D against B:** 44 fewer calls × ≈ 88–92 µs (rule 35) ≈
   −3.9 to −4.0 ms/token of transport. The DSP router adds back ≈ 22 × 15–30 µs
   (256 KiB each by direct HVX read) ≈ +0.3–0.7 ms, and the CPU router and
   staging copies it replaces return ≈ 0.2–0.4 ms. Net ≈ **−3.5 ms/token**. On
   #136's B (26.46 tok/s at G=64, ≈ 37.8 ms) that makes D ≈ 34 ms ≈ 29 tok/s,
   about +10 % over B. **D is still expected below A** (≈ 27.3 ms): 51 > 22
   calls, and ㉗'s ATTN_M1 term (3.2–7.1 ms/token) is in both B and D. So D
   does not meet the issue's "decode ≥ A". That gate belongs to the issue's end
   state (PR 2 or #141), and this cell measures only the call-count term.

## 5. Risks (host-vs-device gaps, and how the sitting shows them)

* **Router weight read rate.** 5.6 MiB/token of f32 from DDR by single-thread
  HVX loads. The host cannot see it. D-prof's `ROUTER_TOPK=` pcycles show it,
  or failing that D − B against the −3.5 ms expectation. If it exceeds ≈ 50 µs
  per op the `l2fetch` lead is the knob, and a pool split of K would be a spec
  change.
* **Near-tie router flips.** The spec and the CPU differ at the last bit (§3.3).
  A flip swaps an expert, which is a large local change. The per-token
  probability is small and bounded by gate 1's statistic. It shows only in the
  PPL column (D against B), never as a logic fault, because the silicon gtest
  pins DSP ≡ spec.
* **Slot 0 continuity across calls** (§3.1). Any stretch that skipped op 0
  would leave slot 0 stale. The validator rule plus the host's
  `bit_identical=1` for "+ ADD" cover the wiring. On silicon, D's text and PPL
  against B read it. A variant with ADD alone would be text-identical to B;
  it is not in the ≤ 4 set, which is noted rather than measured.
* **Subnormals.** ADD's bit identity assumes the same flush behaviour for the
  HVX IEEE add and ARM's (fast-math FTZ). Residual values are O(1), so this is
  out of scope as in `swiglu_det.h`'s note.
* **Stale skel / stub.** A resident ADD or ROUTER_TOPK on a #130 skel fails at
  `graph_init` with `AEE_ECLASSNOTSUPPORT`; the new gtest gives `0x8000040E`.
  The first D token is the check.
* **Address space.** DSP heap + 5.5 MiB (router weights) on top of 48 MiB KV
  and 1.8 MiB params, ≈ 55 of ≈ 182 MiB (rule 8). An `AEE_ENOMEMORY` from
  `graph_set_param` throws and voids D.
* **DVFS / thermal drift.** D against B is read only inside one sitting,
  mirrored order (rule 9). #136's log shows 28 → 59 °C across cells, so the
  thermal line is part of the table.
* **Emulation premise.** The HVX router on `hvx_emu` against silicon is covered
  by `HvxM1Ops.RouterTopk*` riding along.

## 6. Docs to update

* **BENCHMARK.md**: nothing in this PR. The combined sitting adds, under its
  serial, a row "#132 PR 1 D (six kinds + ADD + ROUTER_TOPK, 51 calls/token)"
  beside A / C / B, with the PPL column and text approval, plus the
  `HvxM1Ops.RouterTopk` gtest line.
* **LEDGER.md**:
  * ㉓: correct the ladder to 95 → **73** → **51** → 3 → 1 (the DENSE_FFN ×2 of
    layers 0–1) and record "PR 1 wired, host-gated: ADD bit-identical to the
    CPU add, ROUTER_TOPK `_det` with a lowest-index tie rule shared with the
    CPU comparator; PR 2 held behind #141".
  * ⑨: the D mask and its host counts (hd8 7, hd64 8, lfm25 15).
  * §3a: the PR's skel, app and gtest md5s.
