# Plan 150: the CPU side of decode between the MoE calls, bit for bit

Issue #150 (p1, part of #76). This is lever (3) of the 2026-09-28 direction
change (contract `0001` §12). **Only bit-preserving changes are allowed**:
partitioning, placement, participant count and idle policy may change, and
no accumulation order may change. Base: `htp_moe` @ `08de92b1`, where
dspqueue is the default MoE transport (#141: 14 µs/call, ARM spin up to
`pollUs()` = 5 ms per call). Line numbers below were verified on
`08de92b1`.

## 1. Goal and gate

**Goal.** Cut the ARM time outside the 22 M==1 MoE calls. The budget
arithmetic used for LEDGER budget rows is TPS token time minus
22 × level-2 M==1 `host` (dsp 700.0 + transport 14.1 µs). Applied to
#141 Q it gives ≈ **9.7 ms/token at G=64 and 10.8 at G=512**. This plan
measures that time per op kind first (step 1). It then changes only how
the work is split and scheduled (steps 2–3).

**Byte floor, corrected from the config.** The issue's "≈ 6.5 ms floor"
(lm_head 147 MB + FC / attention / conv ≈ 99 MB at 38 GB/s) under-counts.
It takes doc 48 §1's 170 MB at 0.5 B/weight, which leaves out the Q4_0
scales and the dense FFN of layers 0–1. The shapes in
`q40-qs4cx-wh/config.json` at Q4_0's 18 B / 32 weights give:

| op | MB/token |
|---|---|
| conv in_proj [2048 × 6144] × 18 | 127.4 |
| conv out_proj [2048 × 2048] × 18 | 42.5 |
| attention q/k/v/o × 6 | 35.4 |
| dense FFN layers 0–1 [2048 × 7168] × 3 × 2 | 49.5 |
| router FP32 × 22 | 5.8 |
| lm_head [128000 × 2048] | 147.5 |
| **total** | **408 MB** |

What that total costs at three rates:

| rate | ms/token | source |
|---|---|---|
| 38 GB/s | 10.7 | the rate the issue's floor uses |
| **50 GB/s** | **8.2** | the rate the CPU control run sustains end to end: 953 MB in 19.07 ms, #94 s2 |
| 67.9 GB/s | 6.0 | the CPU's solo DDR rate, #77 ④ |

So ⑰ is worth ≈ 1.5–2.5 ms to the CPU run's own rate, and ≤ 4.7 ms only
if the solo peak were reachable at these sizes. Step 1's GB/s column
settles which.

**Step 1 gate (instrument).** Both must hold:
* **T8 is not inflated.** T8's decode tok/s (mirrored mean, G=512) is
  within 1 % of A0's in the same sitting. Otherwise the table is read as
  shares only.
* **T8 is inert.** `htp_dump_eval.py` gives `bit_identical=1` for T8 vs
  A0. The null A0' vs A0 must also be `=1`.

**Lever gate (step 3), all of these in one sitting, B against A:**

| # | check | pass |
|---|---|---|
| G1 | `NNTR_HTP_DUMP`, prompt 512, G=64: `htp_dump_eval.py` B vs A1, and the null A2 vs A1 | `bit_identical=1` on every file (prefill and decode). A null that is not `=1` voids G1 |
| G2 | text, the 8 prompts of `docs/measurements/prompts/`, G=64 | 8/8 byte-identical to A |
| G3 | text in every tok/s cell | identical to A of the same G and run |
| G4 | decode tok/s (all), prompt 512, G 64 / 512 / 1024, mirrored ×2 | at every G, B's mean > A's mean by more than A's own r1/r2 spread |
| G5 | prefill tok/s, same cells | B ≥ 0.95 × A at each G (standing gate) |
| G6 | `[PPL] decode step=… nll=%.17g` lines (`NNTR_PPL_DECODE=<fresh path>`, self mode) in the dump runs | byte-identical, B vs A1. This covers the final norm and lm_head, which the MoE dumps do not see |
| G7 | mechanism: the `NNTR_OP_TIME` table of B vs T8 | the op kinds the lever targets moved. MoE `call` and the level-2 M==1 `dsp` stay within ±3 % of A |

Contract §1.1 keeps text-vs-CPU as information only (rule 39, ⑱). A
bit-identical change has no PPL or `NNTR_L2_DIFF` column (decisions
table, 2026-09-28).

**Host checks:** rungs 0, 1 and 3 of `hexagon-gates`. There is no rung 2,
because no DSP source changes. Step 2 adds one new host check (§4).

## 2. Where it lives

**Decode path on the ARM (read, not all changed):**
* **Thread pool.** `nntrainer/utils/thread_manager.h`:
  * `defaultComputeThreads` at `:98–111` (`NNTR_NUM_THREADS`, compile
    default 4 from `package_android.sh:76`);
  * `parallelize` at `:262–316`: static equal ranges over all threads,
    and the main thread works too;
  * `SPIN_COUNT = 1000000` at `:339`.
* **Worker loop.** `thread_manager.cpp`:
  * `initialize` at `:71–126` pins the calling (main) thread to
    `core_map[0]` and worker i to `core_map[i]`;
  * `wait_for_new_command` at `:145–174` spins 1 M `yield`s, then
    futex-waits;
  * `thread_parallelize` at `:253–277`: own range first, then steals from
    the tail of the others.
* **Core order.** `thread_manager_util.cpp:141–175`
  (`getCoresByPerformance`) sorts cores by `cpuinfo_max_freq`, and
  `pinSelfToCore` is at `:191`. The Android build uses
  `thread-backend=omp`, but `ggml_interface_omp.cpp` calls `ThreadManager`,
  not OpenMP (no `#pragma omp` on this path).
* **Q4_0 FC at M=1.** The path is:
  * `FloatTensor::dotQnK` (`float_tensor.cpp:987`) → `gemm_q4_0_fp32`,
    because `HtpComputeOps::accelerates_q4_0_at_m1()` is false
    (`htp_compute_ops.cpp:999`);
  * → `arm_compute_backend.cpp:383` → `__ggml_q4_0_4x8_q8_0_GEMM`
    (`ggml_interface_omp.cpp:28–56`).

  That function quantizes the activation once (`nntr_quantize_row_q8_0`,
  main thread), then runs `parallel_for` over ⌈N/16⌉ chunks of 16 output
  rows.
* **The kernel.** `nntr_gemv_q4_0_4x8_q8_0`
  (`nntr_ggml_impl_neon.cpp:31–82`) computes each group of 4 output rows
  over the **whole K** in one `vfmaq_f32` chain, blocks in order.
  **Partitioning is already over N, never K.**
* **q/k/v.** `qkv_layer.cpp:252` → `float_tensor.cpp:811–819` runs three
  separate GEMVs at M=1, each quantizing the same activation.
* **lm_head.** `tie_word_embedding.cpp:466–476` runs the blocked twin
  through the same `gemm_q4_0` (M=1, N=128000, 8000 chunks). Sampling
  (`causal_lm.cpp:303–345`, argmax) runs on the main thread.
* **Attention at M=1.** `mha_core.cpp:682–710` (kcache) and `:1514–1539`
  (vcache) run `parallel_for` over the 8 KV heads. Softmax is on the main
  thread.
* **MoE wait.** `lfm2_moe_layer.cpp:721–925`: the router (`:788`, FP32
  dot), top-k (`:796`) and `tryMoeLayerOnAccelerator` (`:831`). Inside,
  `dspqCall` spins in `read_noblock` for up to `st->arm_spin_us`
  (`htp_compute_ops.cpp:2745`, `:2838–2850`) on the main thread. Meanwhile
  the 7 workers spin in `wait_for_new_command`.

**Instrumentation that exists, and why none of it is the tool:**
* **`--profile` build** (`network_graph.cpp:417–425`, per-TYPE totals).
  Rule 1: never the TPS binary. LEDGER ⑰ measured it adding ≈ 20 ms/token.
  It builds a string per node (`PROFILE_MEM_ANNOTATE`, `neuralnet.cpp:520`)
  and runs a 10 ms `smaps_rollup` sampler thread (`main.cpp:157`, started
  only `#ifdef PROFILE`, `:460–461`).
* **`[M0-PROF]`** (`lfm2_moe_layer.cpp:909`). It prints only for
  `total_tokens > 1`, so decode is silent.
* **`NNTR_HTP_PROFILE`** host columns. They cover HTP calls only, and rule
  15 applies (level 3 inflates `ffn` ~5×).

So a TPS-binary-safe per-node timer is missing.

**Files that change:**
* Step 1:
  * `nntrainer/models/neuralnet.cpp:519–539`: the `forwarding_op`
    lambda. The commented-out per-layer timer at `:525–536` marks the
    spot. This is nntrainer core, outside the supervision scope; this
    plan names it (contract §6).
  * `Applications/CausalLM/models/causal_lm.cpp:704–749`: the decode
    loop.
  * `Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.cpp:721–925`.
  * **New** `tools/htp/op_time_report.py`.
* Step 2, depending on the lever: `nntrainer/utils/thread_manager.{h,cpp}`
  (core, named here), `ggml_interface_omp.cpp:28–56`,
  `tie_word_embedding.cpp:466–476`, `causal_lm.cpp:303–345`, and
  `test/unittest/unittest_nntrainer_cpu_backend.cpp` next to the existing
  `gemm_q4_0` lm_head test (`:451–500`).

**Consumers checked, no change:**
* `test/htp/nntr_hvx.idl` and its stub (`generate_stub.sh`): no DSP
  entry changes.
* `HtpComputeOps`: unchanged in steps 1–3, unless L7 is chosen, which
  touches `:2838–2850` only.
* No weight format changes: the `nntr_quantize_stream` format tag and the
  loader check stay as they are.
* The `NNTR_HTP_PROFILE` stage table (`HTP_MOE_N_STAGES`) is unchanged.
* `tools/htp_fc_report.py` reads only `FC_STAGE` / `FC_FIELD` lines.
* `tools/htp/htp_dump_eval.py` is unchanged.

## 3. Design

### 3.1 Step 1 instrument: `NNTR_OP_TIME=1`

The instrument is off by default, controlled by one env var, and read
once into a function-local `static const bool`. When off, each node pays
one predicted branch. It never touches a tensor.

**1. Per node** (`neuralnet.cpp`). Only for decode calls (`to - from == 1`):
* wrap `node->incremental_forwarding` in two `steady_clock::now()`;
* accumulate `calls`, `sum_ns`, `min_ns`, `max_ns` per execution order
  `f`;
* on a node's first record, also store `wbytes` = Σ
  `getRunContext().getWeight(i).bytes()`;
* at exit, a static reporter prints one line per node:
  `[OP-TIME] node f=<f> name=<name> type=<type> calls=<n> sum_us=<> min_us=<> max_us=<> wbytes=<>`.

  Precedent for printing from a static destructor: `~HtpProfile`,
  `htp_compute_ops.cpp:464`.

min and max are part of the design, not decoration. Doc 46 §45.1 found
over-splitting only through "min stays, avg explodes", and a whole-run sum
cannot show that.

**2. Per token** (`causal_lm.cpp`). Time three things: the
`incremental_inference` call, `generate()` and `registerOutputs()`
(detokenize and stdout). After the summary block, print:
`[OP-TIME] step tokens=<n> infer_us=<> sample_us=<> register_us=<> token_us=<>`.

**3. MoE split** (`lfm2_moe_layer.cpp`). At decode, when the env var is
set:
* set `g_m0_on`;
* add the existing M0 slots `setup + router + topk` and `ffn` into decode
  totals. `ffn` = `tryMoeLayerOnAccelerator` = staging + queue write + ARM
  spin + response, i.e. the MoE wait;
* at exit, print `[OP-TIME] moe calls=<n> cpu_us=<> call_us=<>`.

  No per-call line: 22 × G lines to stdout would distort the timed run.

**4. The report** (`tools/htp/op_time_report.py`, ≈ 80 lines, doxygen
header). It groups nodes by `type` and name suffix into the issue's
kinds:
* FC conv in_proj / out_proj;
* FC attn qkv (the `qkv_layer`, including its q/k head norm) / o;
* FC dense FFN (`_ffn_up|_gate|_down`, or `dense_ffn`);
* attention (`mha_core`);
* conv1d + gate (`causal_conv1d`, `custom_multiply`, `split`);
* RMSNorm;
* residual add;
* MoE CPU part / MoE wait;
* lm_head;
* embedding;
* sampling;
* register;
* other;
* unattributed = token − Σ nodes − sample − register.

Columns: ms/token, share, MB/token, GB/s, avg/min. For `lfm2_moe` the
weight bytes are overridden to the 262 KB router, since the experts are
not read on the ARM. A second log gives a diff column.

### 3.2 Candidate levers (step 2 picks from these by the table)

Every lever leaves each output element to one call of the same kernel on
the same bytes, in the same K order. What changes is only which thread
runs which index, when, and how it waits.

| # | lever | why every accumulation order stays | expected ms/token | triggered by (step 1 table) |
|---|---|---|---|---|
| L1 | **Participant count per `parallel_for` at decode.** An optional `max_threads` in `ThreadManager::parallelize`: ranges go to the first P tids and the others check in empty. The FC GEMV passes P by weight size (e.g. out_proj / o / k / v smaller, lm_head all 8), and prefill keeps 8 | `parallel_for` indices are pure functions writing disjoint outputs. P changes the owner, not the index set or its body | 0.3–0.8. The global 6 threads gave +1.8 % ≈ 0.5 (issue comment); per-op P can do better only where T6/T4 help small FCs while lm_head wants 8 | per-kind ms at T6/T4 < T8 for some FC kinds; avg/min ≥ 1.3 on small FCs |
| L2 | **Placement.** Main plus worker order over the prime / performance clusters, and one core left free for OS / FastRPC threads (today 8 threads pin all 8 cores) | same as L1 | ≤ 0.3 | Step 1's task snapshot shows non-voluntary switches on pinned workers, or a pinning failure |
| L3 | **Idle policy around the MoE call.** Two options. (a) Workers keep spinning, and the spin is bounded by time instead of 1 M `yield`s, so a ≈ 0.7 ms call never drops them into futex. (b) Workers are parked at the call and woken at the next `parallel_for`. The main thread's dspq spin is untouched in both | no arithmetic touched | (a) up to 22 × wake latency ≈ 0.4–1.3 if workers are sleeping now. (b) is a thermal / DVFS trade, sign unknown | (a) the first FC after each MoE call has avg ≫ min while the others do not, and voluntary switches per token ≈ 22 × 7. (b) cluster frequency drops across the G=512 decode |
| L4 | **One barrier and one Q8 quantization for q/k/v** (and dense up+gate): one `parallel_for` over the (weight, 16-row chunk) pairs | same kernel per 4-row group. `nntr_quantize_row_q8_0` is a deterministic function of the same input, so one call gives the bytes three calls gave | ≤ 0.1 (8 barriers + 12 quantizations fewer per token) | only if the barrier cost reads large (unattributed / small-op avg ≫ min) |
| L5 | **lm_head chunk argmax.** Each 16-row chunk writes its logits (kept: PPL reads them) and its local (max, first index) after the bad-word penalty. The main thread reduces the chunk maxima in index order | argmax is comparisons only. A first-index tie-break in index order returns what `std::max_element` returns | ≤ 0.1 | `sample_us` ≥ 0.1 ms |
| L6 | **Warm the next layer's FC weights during the MoE wait.** The CPU idles ≈ 15.7 ms/token; a conv block's 9.4 MB fits one cluster's L2, and each worker touches the rows of its own static range | a prefetch changes no value | up to ≈ 2–3 **if** DDR serves both readers above one reader's rate. It is negative if it slows the DSP's MoE stream by the same bytes (④ two-reader DSP side inconclusive, rule 12) | FC kinds at ≤ 50 GB/s and step 3's G7 `dsp` column. **Not in this issue's first PR:** it needs its own two-reader cell, and gets a new issue if the table points there |
| L7 | **The dspq ARM spin's core.** It spins on the main thread, pinned to `core_map[0]` (a prime core). Options: keep it, or yield inside the spin (`sched_yield` every N polls) | no arithmetic touched | ≈ 0 unless the prime cluster throttles | the prime-cluster frequency trace |

The following are not levers under this issue's rule: a K-split of any
GEMV, KleidiAI or another Q4_0 kernel, the `bstp` / `mixed` thread
backends (their GEMM paths differ), and any change to
`nntr_quantize_row_q8_0`.

**Rejected alternative: a second, decode-only `ThreadManager` with its
own thread count.** Two pools of pinned, spinning threads would put 16
threads on 8 cores, and the two spins would fight each other. L1 gets the
same freedom from the one pool, without a second set of threads.

### 3.3 Contract §2 and doc 45 §3

* **Contract §2.** No DSP code changes. The three walls, the arena and
  the 32-bit address budget are untouched. `QS4CX_WH` still has no CPU
  path: the MoE stays on the HTP, and `moe_htp_layers` stays empty.
* **Doc 45 §3.** Activation handles and DMA are not touched. The only
  quantizer on this path, the CPU Q8_0 activation quantizer, keeps being
  called on the same bytes; L4 calls it once instead of three times on
  identical input. The bit-identity gates and the text gates are G1, G2
  and G6.

## 4. Steps

1. **Breakdown.**

   **1a. Instrument (implementer).** Build the three hooks and the report of
   §3.1. Kernel/app commit separate from the docs commit.
   * Gate: rung 0 and rung 1 (`ninja -C build`, `*Lfm2Moe*` 6 PASSED,
     `run_host_checks.sh` `ALL CHECKS PASS`, `run_inproc_e2e.sh`
     `INPROC E2E PASS`).
   * Plus one host smoke: one `htp_e2e_test --run` invocation of
     `run_inproc_e2e.sh` with `NNTR_OP_TIME=1` →
     `op_time_report.py` prints a table whose rows plus unattributed
     equal `token_us` within 2 %. The same run without the variable
     prints no `[OP-TIME]` line.
   * Then rung 3 (app, `--cache`). There is no rung 2: reuse skel
     `37468a7f…` from `/local/mnt/workspace/htp_moe/141b/set/`.

   **1b. Device breakdown (orchestrator; device measurement unavoidable
   here).** Handoff `docs/measurements/150-cpu-decode.md`, ≈ 25 min.

   Variants (4):

   | variant | binaries | env |
   |---|---|---|
   | **A0** | the #141b set (md5s in its `md5.txt`) | `NNTR_HTP_DSPQ=1`. The default flip `c73384d2` changed only this default, so this is `08de92b1`'s path |
   | **T8** | the new set | `NNTR_OP_TIME=1` |
   | **T6** | the new set | `NNTR_OP_TIME=1 NNTR_NUM_THREADS=6` |
   | **T4** | the new set | `NNTR_OP_TIME=1 NNTR_NUM_THREADS=4` |

   Cells:
   * (a) Sanity at G=8.
   * (b) Prompt 512, G=512, mirrored A0 T8 T6 T4 | T4 T6 T8 A0. Every T
     log carries the table. G 64 / 1024 are not needed for a
     measurement-only step: the budget rows are G=512, and step 3 runs
     all three.
   * (c) During one extra T8 G=512 run, after 20 s and again after 25 s:
     * `/proc/<pid>/task/*/status` (Name, Cpus_allowed_list,
       voluntary / nonvoluntary ctxt switches) and field 39 of `stat`;
     * per-policy `scaling_cur_freq` and `cpuinfo_max_freq`, sampled
       every 0.2 s for 5 s;
     * `topology/cluster_cpus_list`;
     * zone0.

     This gives switches per token per worker, the core map as
     pinned, the clocks, and whether any `pinning … failed` line
     appeared.
   * (d) Dumps, prompt 512, G=64: A0, T8, A0' plus `NNTR_PPL_DECODE`
     self-mode files (step-1 gate).

   Thermal checkpoints after each block. Optional, no rebuild:
   `simpleperf stat --per-thread -e task-clock,context-switches,cpu-migrations`
   on one T8 run.

   **Rule 1:** no cell uses a `--profile` build. **Rule 15:** no cell
   combines `NNTR_OP_TIME` with `NNTR_HTP_PROFILE` (level 2 swaps in the
   timed MoE entry, level 3 inflates `ffn` ~5×).

   Output:
   * the per-kind table for T8 / T6 / T4 (ms/token, GB/s, avg/min);
   * the task / frequency snapshot;
   * the step-1 gate.
2. **Choose and build (planner amends §3.2 with the choice, then the
   implementer).** Apply the "triggered by" column. At most two levers.
   Each goes behind one env switch, default off (e.g.
   `NNTR_DECODE_PARTICIPANTS=`, `NNTR_TM_SPIN_US=`).
   * **If every FC kind already reads ≥ 50 GB/s, stop.** Close ⑰ with
     the table, and file L6 as its own issue if the MoE-wait overlap is
     the only room left.
   * **New host check** (non-trivial logic leaves one check). In
     `unittest_nntrainer_cpu_backend`, run `gemm_q4_0` at M=1 for N ∈
     {512, 2048, 6144, 128000} with participants P = 1…8 (and, for L3,
     the spin policy on / off). Require `memcmp == 0` of every output
     against P = 8 today. L4 and L5 add a byte-compare against today's
     three-call / `max_element` result.
   * Gate: rungs 0, 1 and 3.
3. **Lever sitting (device measurement unavoidable).** Handoff with ≤ 4
   variants:

   | variant | what |
   |---|---|
   | **A** | the unchanged `htp_moe` head set, nothing set |
   | **B** | lever on |
   | **B'** | optional: the lever's second setting |
   | **TB** | B + `NNTR_OP_TIME=1`, one run per G, not a tok/s cell |

   Cells:
   * prompt 512, G 64 / 512 / 1024, mirrored ×2 (G4, G5, G3);
   * dumps A1 / B / A2 at G=64 with PPL self files (G1, G6);
   * the 8-prompt text set at G=64 (G2);
   * one level-2 profile each for A and B at G=64 (M==1 `dsp` ±3 %, G7).
4. **Default.** Making the lever the default is a user decision in rule
   33's form: the win holds at all three G in one sitting, text = A, and
   prefill passes.

## 5. Risks

* **The host shows nothing about speed.** The workstation has no prime /
  performance split and no DSP. Host checks prove only bit-identity and
  that the timer is inert. Every ms comes from the device table.
* **DVFS and thermal drift.** Decode moves ±5 % between sittings (rule 9),
  and T6 / T4 cells run at different heat. Mitigations: mirrored order,
  thermal checkpoints per block, and verdicts only against A's r1/r2
  spread. Step 1b's frequency trace shows whether the clocks moved during
  decode.
* **The instrument inflates the thing it measures.** It costs ≈ 250 nodes
  × 2 clock reads ≈ 15 µs/token by estimation. The T8-vs-A0 tok/s gate
  and the per-token `unattributed` row make any inflation visible.
* **DMA rate / DDR contention.** L3(b) and L6 change what the CPU does
  while the DSP streams weights. G7's level-2 `dsp` and `mm` columns show
  a MoE-side loss that a decode gain could otherwise hide.
* **Stale skel.** No DSP source changes, so both sets carry skel
  `37468a7f…`. The device `md5sum` line is checked against it; a
  mismatch voids the sitting (rule 14's B case).
* **Affinity.** Samsung's cpusets for a shell process may refuse
  `sched_setaffinity`. Step 1b(c) records `Cpus_allowed_list`, and the
  logs are grepped for `pinning thread on cpu`.
* **Address-space budget.** Not affected: ARM only. L6's future warm-up
  touches existing mappings only.

## 6. Docs to update (supervisor)

* **BENCHMARK.md:**
  * Results rows for step 1b (A0 / T8 / T6 / T4, G=512, tagged with
    serial and sha);
  * a "#150 per-op CPU breakdown at M=1" side table with the report's
    columns;
  * later the step-3 rows (A / B / B' at G 64 / 512 / 1024, with text
    and dump columns).
* **LEDGER.md:**
  * ⑰ resized with the measured table; its "≈ 2–3 ms" and the
    budget-row "≈ 6.5 ms byte floor" replaced by the 408 MB figure of
    §1 (6.0 / 8.2 / 10.7 ms at 67.9 / 50 / 38 GB/s);
  * a new rule only if the device contradicts the code reading (e.g.
    workers sleep during the MoE call, or pinning fails);
  * L6 as an open item if step 2 stops there.
* **Contract §1's "730 MB" row** has the same under-count (≈ 892 MB with
  the 408 MB rest). That correction is the user's call.
