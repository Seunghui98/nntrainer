# 162 — G=1024 decode 47.4 → 50 tok/s, bit-preserving: CPU attention first, the prefetch overlap only if the bypass leaves it room

Issue dlwlzzero/nntrainer#162 (p1, LEDGER ㉙, rules 43–46). Contract
`docs/plans/0001-htp-moe-decode-agent-system.md` (§12: bit-preserving levers
only). Read against `htp_moe` @ `c95c0feb` (cycle 22; the ledger fold is PR #163 on
`htp/ledger-2026-09-29` @ `4d6a45a4`). User direction 2026-09-29: decode must
eventually run end to end on the NPU, and the bit-preserving rule stays (#164
covers the HTP norms).

## 0. Summary for the orchestrator

* **Order: step 0 (one device sitting, sizing only) → lever (a) CPU attention
  → lever (b) prefetch overlap, only if step 0 passes rule P′ → one verdict
  sitting.** The HTP-side lever (make the bit-identical ATTN_M1 fast) cannot
  reach 1.1 ms/token at G=1024. It is the end-to-end blocker, but it is out of
  #162's reach (§3.3). The supervisor should re-scope #146 for it.
* **Neither CPU lever survives an NPU end-to-end design as code.** (b)
  survives as a principle and as a number. On the NPU the same idea becomes
  "the next op's weight DMA runs while the current op computes" (doc 45 §3).
  Step 0's re-read of the prefetch under the bypass is exactly the DRAM-share
  number that E2E design needs. (a) does not survive at all. The HTP ATTN_M1
  replaces `mha_core`, and (a) only raises the CPU bar the resident attention
  must meet. (a) still goes first: it is the larger and more certain term,
  it is local, and it is bit-provable per lane. (b)'s sign under the bypass is
  unknown (rule 44 (2)). §3.4 gives the full argument.
* **Step 0 needs a small test-only build.** The issue says "probe only, no
  build", but that cannot hold. The committed `PrefetchOverlap` hard-codes
  the L2 word (`GemvOpts(192, true, true)` = `0x303e1`,
  `test/unittest/unittest_hvx_dma_probe.cpp:958-960`), so today it cannot
  read the bypass. Step 0a adds that bit, plus a device microbench that sizes
  (a). No lever code, no DSP source change.

## 1. Goal and gate

**Goal.** The BENCHMARK Goals row "decode tok/s, NPU, gen 1024" reaches
**≥ 50.00** (≤ 20.00 ms/token). Today it is 47.36, i.e. 21.11 ms/token, so the
change must remove **≥ 1.1 ms/token at G=1024**. Plan for ≥ 1.3 ms, because a
sitting's A drifts by ±0.5 tok/s.

| # | check | where | pass |
|---|---|---|---|
| G1 speed | decode tok/s at G=1024, mean of two mirrored runs, prompt 512 | verdict sitting (§4 step 4), variant B or C vs that sitting's A | **≥ 50.00**; G 64 / 512 not below A by more than A's own r1/r2 spread |
| G2 A validity | A logs | same sitting | `[HTP] moe m1 gemv: on (applied=0x703e1) lead=192KB rows1=1 feed=vtcm dma_bypass=1 source=default` once; `[HTP] dspq: on …` once; `dspq: close calls=22·G served=22·G bad=0`. Anything else voids the sitting (rules 36, 43 (3)) |
| G3 bits | `tools/htp/htp_dump_eval.py` on `NNTR_HTP_DUMP` (G=64, prompt 512) | same sitting | A1 vs A2 `bit_identical=1` (null), A1 vs B / C `bit_identical=1` (2862 files) |
| G4 nll | `NNTR_PPL_DECODE` lines at G=512 | same sitting | `[PPL] decode step=` lines of B, C and A2 byte-identical to A's |
| G5 text | 8-prompt set at G=64 and every tok/s cell | same sitting | `strip` output identical to A's (`cmp`). No PPL route and no approval route (rule 45) |
| G6 prefill | mean prefill per G | same sitting | ≥ −5 % of A. If the phone warms, use the adjacent-cell tie-breaker (the #117 / #158 reading) |
| G7 host | `bash test/htp/host/run_host_checks.sh`, `bash test/htp/host/run_inproc_e2e.sh` | workstation | `ALL CHECKS PASS`, `WORKER POOL LANES OK`, `INPROC E2E PASS`, with each knob unset and set (§4) |
| G8 device kernels | `unittest_nntrainer_cpu_backend_fp16 --gtest_filter='AttnM1F16Det.*:MhaM1*'` | device | every `ATTN_M1_F16 … bad=0`, every `MHA_M1_PHASE … bad=0` |

Standing gates: prefill ≥ −5 % of A (G6). The contract's "text identical to the
CPU run" is met through G5: A's text is the reference, and a bit-identical B
cannot differ from it.

## 2. Where it lives (verified @ `c95c0feb`)

**Step 0a (test only):**

| file | change | refs |
|---|---|---|
| `test/unittest/unittest_hvx_dma_probe.cpp` | `PrefetchOverlap`: OR in `HTP_MOE_FLAG_DMA_BYPASS` when `htp_moe_opts_dma_bypass(getenv("NNTR_MOE_DMA_BYPASS"))` says so. That helper makes unset mean bypass, the app's default. Print `bypass=` in `PREFETCH_CONFIG` | `:958-960` (opts word and echo assert), `:978-981` (config line); helper `nntrainer/tensor/htp_backend/htp_moe_opts.h:58-62`; include path `htp_backend/…` already used at `:58` |
| `test/unittest/unittest_nntrainer_cpu_backend_fp16.cpp` | new `MhaM1Phases.Decode` (§3.1): times and bit-compares the decode sequence at 32 / 8 / 64 | beside `AttnM1F16Det.*` `:953-1115` (shape constants `:953`) |

**Lever (a), CPU attention (decode branch only):**

| file | change | refs |
|---|---|---|
| `Applications/CausalLM/layers/mha_core.cpp` | decode (`to − from == 1`) row of `softmax_triangle`: head-group split (a1), both dtypes. Decode `compute_kcaches` / `compute_fp16vcache_transposed` call the (a2) kernels under the knob | softmax row==1 fp16 `:1334-1341` (f32 `:1287-1295`); kcache fp16 decode `:743-760`; vcache fp16 decode `:1569-1585`; call order `:870-877`; per-call fp16 copies `:530-548`, `out_` alloc `:862` |
| `nntrainer/tensor/cpu_backend/arm/neon_impl_fp16.cpp` (+ `arm_compute_backend_fp16.cpp`, `cpu_backend.h`) | (a2) only, and only if step 0 sizes it: **new** functions (`compute_kcaches_m1_ilp`, `compute_fp16vcache_m1_reg`). The existing ones stay untouched because prefill uses them | `compute_kcaches(__fp16…)` `:2008`, `compute_fp16vcache_transposed` `:1858`, `softmax_row_inplace_no_sink` `:1490` |
| knob | `NNTR_MHA_M1_FAST` (unset = off in the measured build; the flip commit makes unset = on, `=0` = off) | read once, like `NNTR_OP_TIME` (`nntrainer/models/neuralnet.cpp:130-136`) |

**Lever (b), prefetch overlap (only after P′ passes):**

| file | change | refs |
|---|---|---|
| `nntrainer/tensor/cpu_backend/compute_ops.h` | one core virtual: `set_moe_window_work(std::function<void()>)`, a one-shot job run inside the next MoE call's wait | beside `gemm_qs4cx_moe_layer_fp32` `:295-305` |
| `nntrainer/tensor/htp_backend/htp_compute_ops.cpp` | `dspqCall`: after `api.write` succeeds and before the read spin, run the job (it calls `ThreadManager::parallel_for`, and the main thread participates) | write `:2834`, spin `:2843-2852`. FastRPC path (`NNTR_HTP_DSPQ=0`): not hooked |
| `Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.cpp` | before `ops->gemm_qs4cx_moe_layer_fp32` (`:573`), post a touch of the first S MiB of the next layer's first FC weight (conv `in_proj` or `qkv`), with a volatile sink | the next-layer weight pointer is wired at model finalize in `Applications/CausalLM/models/lfm2_moe/` |
| knobs | `NNTR_MOE_PREFETCH_MIB` (unset/0 = off), `NNTR_MOE_PREFETCH_T` (default from step 0) | — |

**Consumers of a changed contract.**
* IDL `test/htp/nntr_hvx.idl` / `generate_stub.sh`: **unchanged**. No DSP
  source changes, and the skel is rebuilt only to match the tree.
* `HtpComputeOps`: only (b)'s window hook.
* Quantizer format tag (`nntr_quantize_stream`) and loader check: unchanged,
  since no weight format changes.
* `NNTR_HTP_PROFILE` stage tables and `tools/htp_fc_report.py`: unchanged.
* `tools/htp/op_time_report.py`: unchanged. The attention row stays the
  `mha_core` node, and (b)'s touch time lands inside "MoE wait", which is
  where it belongs.
* The `NNTR_HTP_FORWARD` resident path does not call `mha_core`'s decode
  kernels, so it is unaffected.

## 3. Design

### 3.1 Where the 1.97 ms goes (what step 0 must confirm)

Decode `mha_core` at LFM2.5's shape (32 q heads, 8 kv heads, head_dim 64,
fp16 on Android) runs three phases per layer.

* `compute_kcaches` runs across the 8 kv heads on the pool (`:743-760`). Each
  score is one dependent chain: 8 `fmla .8h`, the `faddp` tree, then a scalar
  add. That makes it latency-bound, not byte-bound.
* `softmax_row_inplace` runs **single-threaded over all 32 heads**
  (`:1341`). Three passes over `[L][32]` fp16 (max, `exp_f16x8` + sum,
  `fdiv .8h`).
* `compute_fp16vcache_transposed` runs across the 8 kv heads. Its 32
  accumulators live in a `std::vector` (`neon_impl_fp16.cpp:1881`), so every
  `fmla` round-trips through memory.

A rough cycle count (≈ 80 µs kcache, ≈ 130 µs softmax, ≈ 60–80 µs vcache per
layer at L=1024, 2.0–2.7 GHz) adds up to 1.6–2.0 ms/token. That matches
#90's 1.97 and #150's slope (≈ 2 µs per position per token). These are
estimates, not measurements. Step 0's microbench replaces them.

### 3.2 Lever (a): keep every lane's operation sequence, change only who runs it and when

* **(a1) Softmax head-group split.** The fp16 softmax is lane-wise over heads.
  One `float16x8_t` holds 8 heads, and max, exp, sum and divide never cross
  lanes (`:1490-1555`; the x86 f32 twin is the same, `avx2_impl.cpp:1335`).
  Mechanism: gather each group of 8 heads into a thread-local `[L][8]`
  buffer, call the **unchanged exported** `softmax_row_inplace(buf, 0, L, 8)`
  on 4 pool threads, and scatter back. Each lane then sees exactly the same
  instruction sequence as before, so the result is bit-identical by
  construction.
  - Scope: only when `row == 1`, `num_head % 8 == 0`, no sink, no softcap
    and `from < local_window_size`. Otherwise the current code runs.
  - Both dtypes, so the host's f32 path and the lfm25 host fixture
    (32 / 8 heads) exercise the split logic.
* **(a2) Register-resident kernels (NEON fp16 only, decode only, new
  functions).**
  - kcache: interleave 4 positions, 4 independent accumulators. Each
    position's `fmla` chain, `faddp` tree, `0 + t` and `/ 8` stay in the
    same order.
  - vcache: fixed `gqa = 4`, `head_dim = 64`, with the 32 accumulators held
    in registers over two passes of 16. Each accumulator's `fmla` sequence
    over positions stays ascending.
  - Bit identity is per accumulator, so it is proven on the device by the
    step 0 gtest, by `AttnM1F16Det` against `attn_m1_det.h`, and by
    `llvm-objdump` of the shipped `libnntrainer.so` (#152 step 6: the fused
    and unfused instructions must match the spec).
  - Built only if step 0 shows it pays.

**Rejected alternative: fusing kcache → softmax → vcache into one
`parallel_for` of 4 units** (2 kv heads = one 8-lane softmax group each).
It saves two pool dispatches per layer. But its units write interleaved
16-byte slices of every 64-byte `[L][32]` row, which is false sharing through
all three phases. It also turns the ordering into a new protocol for a gain
that is only the dispatch cost. Revisit it only if step 0 shows dispatch
(microbench `total − Σ phases`) above 20 µs per layer.

### 3.3 The HTP-side lever, and why it is not #162's

The bit-identical HTP attention exists: #152 (`attn_m1_det.h`), and the
norm-shadow sitting's control T (`MOE,ROPE,ATTN_M1`) equalled A on every
logit. Replacing CPU attention with it saves 1.1 ms only if each layer's
ATTN_M1, plus its extra call, costs ≤ (1.97 − 1.1) / 6 ≈ **145 µs** at
position ≈ 1024.

Measured: **1736.6 µs** exact at pos 1023 (rule 45, 12× too slow). Even the
non-exact f32 kernel reads 387 µs (#148), and the exact kernel's per-op
`rne16` / TwoSum emulation multiplies the HVX op count on top of that.

So this lever cannot deliver #162. **It is the E2E blocker:** 6 × 1.74 ms =
10.4 ms/token at G=1024 on a resident path. The supervisor should file or
re-scope #146 as "exact ATTN_M1 ≤ CPU parity (≈ 330 µs at pos 1023, lower
once (a) lands)", with the fp16 cache (exact by construction, since every
value is on the fp16 grid). This plan does not build it.

### 3.4 Order, against the NPU end-to-end direction

| | survives E2E? | size at G=1024 | risk | prefill |
|---|---|---|---|---|
| (a) CPU attention | **no.** Deleted when ATTN_M1 goes resident. It also raises the bar ATTN_M1 must meet, and it lifts the CPU `q40` control by the same ms (same `mha_core`), so it does not widen NPU vs CPU | estimated 0.6–1.2 ms (unverified, step 0) | low: local, per-lane provable, no DRAM contention with the DSP | untouched: decode branch only |
| (b) prefetch overlap | **as a principle and a number.** The CPU code dies. The DSP-side "DMA the next op's weights under the current op" and the DRAM share left beside the ring are E2E design inputs, and step 0 measures that share | +0.7–1.4 on the L2 path. Under the bypass ≈ 13 GB/s is left (rule 44 (2)), so the sign is unknown | medium: a cross-layer hook through `ComputeOps`, touch_late stalls, thermal | the hook sits in the MoE window only, but the prefill gate applies (issue) |

**Recommendation:** step 0 → (a) → (b) if P′ holds.

* (a) goes first because it is the larger, more certain and cheaper term.
  Its correctness argument is local.
* (b)'s survivable part (the measurement) is in step 0 whatever the order.
* If (a) alone reaches ≥ 50 in the verdict sitting, (b) is not built. Its
  step-0 row then goes to the ledger as the E2E DRAM-share input.

## 4. Steps

### Step 0a: test-only commit on `htp/162-g1024` (implementer, ≈ 1 h + builds)

1. Probe knob (§2). `PREFETCH_CONFIG … opts=0x703e1 bypass=1 …` when unset,
   and `opts=0x303e1 bypass=0` with `NNTR_MOE_DMA_BYPASS=0`. The existing
   `ASSERT_EQ(applied, feed)` then doubles as the stale-skel check: a
   pre-#159 skel echoes `0x303e1`.
2. `MhaM1Phases.Decode` (one `@brief`, ≈ 200 lines of test code):
   * Inputs: random q / K / V at `L ∈ {513, 1024, 1536}`, 32 / 8 / 64,
     fp16. Run through `ThreadManager::Global()` exactly as `mha_core`'s
     decode does (`NNTR_NUM_THREADS=8`).
   * `impl=as_is`: the three exported functions as at `:743-760`, `:1341`,
     `:1569-1585`.
   * `impl=split4`: the softmax as §3.2 (a1).
   * `impl=kv_ilp`: test-local prototypes of the two (a2) kernels.
   * `impl=all`: split4 and kv_ilp together.
   * Per impl and L, 50 iterations, report medians: `MHA_M1_PHASE L=<L>
     impl=<…> threads=8 kcache_us=… softmax_us=… vcache_us=… total_us=…
     bad=<n> iters=50`, where `bad` counts fp16 outputs (32 × 64) and
     softmax rows that differ bitwise from `as_is`. `EXPECT_EQ(bad, 0)`.
3. Gates: rung 0 (clang-format-14). Rung 1: `ninja -C build` and
   `run_host_checks.sh` `ALL CHECKS PASS` (nothing host-side changed; this is
   a sanity check). Rung 2: `./test/htp/build.sh` → `UNDEFINED SYMBOLS OK`,
   IDL unchanged. Rung 3: `build_android.sh --htp` plus the `test/jni`
   ndk-build of `unittest_hvx_dma_probe unittest_nntrainer_cpu_backend_fp16`.
4. Stage `/local/mnt/workspace/htp_moe/162/set/` with `md5.txt`:
   `nntrainer_causallm`, `libcausallm_core.so`, `libnntrainer.so`,
   `libccapi-nntrainer.so`, `libc++_shared.so`, `libsdkl.so`,
   `libnntr_hvx_skel.so`, `unittest_hvx_dma_probe`,
   `unittest_nntrainer_cpu_backend_fp16`, `prompt512.txt`.
   Sanity: `strings libnntrainer.so | grep -c 'dspq: on'` = 1;
   `strings unittest_hvx_dma_probe | grep -c 'bypass='` ≥ 1;
   `strings unittest_nntrainer_cpu_backend_fp16 | grep -c MHA_M1_PHASE` ≥ 1;
   `strings libcausallm_core.so | grep -c NNTR_HTP_FORWARD_KINDS` ≥ 1;
   `find … -name 'libcdsprpc*' | wc -l` = 0.
   Copy §4 step 0b into `docs/measurements/162-step0.md` with the md5 table
   filled, commit, push.

### Step 0b: device sitting, sizing only (orchestrator; **device measurement unavoidable**; ≈ 15 min)

No variant, no verdict. A is the committed default, used as a ruler.

```
cd <worktree of htp/162-g1024> && git fetch -q && source tools/htp/env.sh
W=/local/mnt/workspace/htp_moe/162; L=$W/logs0; mkdir -p $L
S=R3CY10WM83Y
D=/data/local/tmp/nntrainer/causallm/s162; M=/data/local/tmp/nntrainer/causallm/models/q40-qs4cx-wh
therm() { adb -s $S shell dumpsys battery | grep -E 'level|temperature'; adb -s $S shell cat /sys/class/thermal/thermal_zone0/temp; }
probe() { # probe <log> <gtest filter> [env]
  adb -s $S shell "cd $D && md5sum libnntr_hvx_skel.so unittest_hvx_dma_probe && \
    $3 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_dma_probe --gtest_filter='$2'" 2>&1 \
    | tee $L/$1.log | grep -E 'md5|^PREFETCH_(CONFIG|COLD|OVERLAP)|^DMA_SETTINGS_MEAN|INVALID|PASSED|FAILED'; }
run() { # run <G> <log>
  adb -s $S shell "cd $D && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $1/' $M/nntr_config.json && \
    grep num_to_generate $M/nntr_config.json && md5sum libnntr_hvx_skel.so nntrainer_causallm libnntrainer.so && \
    NNTR_NUM_THREADS=8 NNTR_OP_TIME=1 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm $M \"\$(cat $D/prompt512.txt)\"" \
    2>&1 | tee $L/$2.log | grep -E '^(prefill|generation|peak memory)|dspq:|moe m1 gemv|\[OP-TIME\] (step|moe)'; }
```

0. `adb devices` (record every serial). Run `therm | tee -a $L/therm.log`
   (checkpoint 0). Screen off, charger in, cool start (zone0 ≤ 35 °C).
1. Install and config.
   ```
   adb -s $S shell "rm -rf $D && mkdir -p $D" && adb -s $S push $W/set/. $D/ > /dev/null
   adb -s $S shell "chmod 755 $D/nntrainer_causallm $D/unittest_hvx_dma_probe $D/unittest_nntrainer_cpu_backend_fp16"
   adb -s $S shell "cd $D && md5sum \$(ls -p | grep -v / | sort)" | tee $L/md5_device.log
   diff <(sort -k2 $W/md5.txt | tr -d '\r') <(sort -k2 $L/md5_device.log | tr -d '\r') && echo MD5 OK
   ```
   Then apply the #158 config block unconditionally (`158-dma-bypass.md`
   step 1: greedy, `bad_word_ids [124900]`, `moe_engine: htp`). Expected
   echo: `do_sample": false`, `init_seq_len": 512`, `moe_engine": "htp"`,
   `moe_htp_layers": ""`.
2. Probe, cold, bypass (the decision cells). ≈ 1 min.
   ```
   probe cold_bp '*DmaSettings*:*PrefetchOverlap*'
   grep -c '^DMA_SETTINGS_MEAN' $L/cold_bp.log      # 6
   grep -c '^PREFETCH_OVERLAP' $L/cold_bp.log       # 12
   grep -c INVALID $L/cold_bp.log                   # 0
   ```
   Expected: `PREFETCH_CONFIG readers=8 workers=7 ring_mib=200 sets=16
   opts=0x703e1 bypass=1 chunk_bytes=268435456`. `DMA_SETTINGS_MEAN
   src_bypass=0 queues=1 gbs≈37` (the anchor) and `src_bypass=1 queues=1
   gbs≈69` (#158). `[  PASSED  ] 2 tests`.
   **Stop conditions:** an echo failure `the skel does not know #117's feed
   bit` (stale skel), or `err=0x8000040e`.
3. Probe, cold-ish, L2 (ties this sitting to #90's cells).
   ```
   probe l2 '*PrefetchOverlap*' NNTR_MOE_DMA_BYPASS=0     # PREFETCH_CONFIG … opts=0x303e1 bypass=0
   therm | tee -a $L/therm.log                            # checkpoint 1
   ```
4. CPU attention microbench. ≈ 1 min.
   ```
   adb -s $S shell "cd $D && NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ./unittest_nntrainer_cpu_backend_fp16 \
     --gtest_filter='AttnM1F16Det.*:MhaM1Phases.*'" 2>&1 | tee $L/mha.log | grep -E '^MHA_M1_PHASE|bad=[1-9]|PASSED|FAILED'
   grep -c '^MHA_M1_PHASE' $L/mha.log        # 12 (3 L x 4 impl)
   grep -c 'bad=[1-9]' $L/mha.log            # 0
   ```
5. A, full E2E, G=1024 twice, with `NNTR_OP_TIME=1`. ≈ 2 min.
   ```
   run 1024 A_G1024_r1; run 1024 A_G1024_r2
   grep -h 'moe m1 gemv' $L/A_G1024_r?.log     # applied=0x703e1 lead=192KB rows1=1 feed=vtcm dma_bypass=1 source=default, once each
   grep -c 'dspq: on' $L/A_G1024_r?.log        # 1 each
   grep -h 'dspq: close' $L/A_G1024_r?.log     # calls=22528 served=22528 bad=0
   therm | tee -a $L/therm.log                 # checkpoint 2
   ```
   A log with `dma_bypass=0`, or without `dspq: on`, is void (rules 36, 43).
6. Probe, warm, bypass: `probe warm_bp '*PrefetchOverlap*'`, then
   checkpoint 3.
7. Reports on the workstation:
   `python3 tools/htp/op_time_report.py $L/A_G1024_r1.log` (and r2), then
   `grep -h '^PREFETCH_OVERLAP size_mib=\(4\|10\) ' $L/{cold_bp,l2,warm_bp}.log`.

**Tables to fill** (in `162-step0.md`):

| cell | fill |
|---|---|
| unit, KST window, checkpoints 0–3 (battery °C·10 / zone0 m°C), MD5 OK | |
| DmaSettings anchor / bypass `gbs` (queues=1) | / |
| A G=1024 r1 / r2: prefill, decode, last-64 tok/s | |
| OP-TIME G=1024 r1 / r2: token ms, `attention (mha_core)` ms/token and avg/min, MoE wait ms, `[OP-TIME] moe` call µs | |

| S MiB | T | cold_bp: dmm_pct / staged / net ms | l2: dmm_pct / staged / net ms | warm_bp: dmm_pct / staged / net ms | touch_late (bp) |
|---|---|---|---|---|---|
| 4 | 1 / 2 / 7 | | | | |
| 10 | 1 / 2 / 7 | | | | |

| L | impl | kcache / softmax / vcache / total µs | bad |
|---|---|---|---|
| 513, 1024, 1536 | as_is, split4, kv_ilp, all | | |

**Decision rules (the supervisor applies them to the filled tables):**

* **Coverage:** C = 6 × `total_us(as_is, L=1024)` / OP-TIME attention
  ms/token at G=1024.
  - C ≥ 0.7: the microbench explains the row, so project
    Δa = 6 × (`total_us(as_is)` − `total_us(best)`) at L=1024.
  - C < 0.7: the rest (RoPE, the four fp16 copies and `out_` alloc,
    `:530-548`, `:862`) is itself the next target. Record it before
    building.
* **(a1) / (a2):** build a sub-lever only if its own projected share is
  ≥ 0.15 ms/token, and only with `bad=0` on every line.
* **P′ (b) on the bypass:** build (b) iff for S = 4 or 10 some T has
  `net_ms_per_token` ≥ 0.4 in **both** `cold_bp` and `warm_bp`,
  `dmm_pct` ≤ 5, and `touch_late` ≤ 3/64. Otherwise (b) closes as "no
  room beside the bypass ring" (LEDGER). The `l2` row must reproduce #90's
  sign (+0.7…+1.4 at S=4); if it does not, the probe or the unit drifted
  and P′ is not read.
* **Sum:** if Δa + Δb (the P′ cell's `net`) < 1.3 ms, stop and report to
  the supervisor before building. Then 50 at G=1024 is not reachable with
  these two levers. The remaining bit-preserving candidates are the qkv row
  (0.57 ms at 37 GB/s vs 51–63 elsewhere, rule 46) and the unattributed
  0.28 ms.

### Step 1: lever (a) (implementer, host)

1. (a1) in `softmax_triangle`, both dtypes, under `NNTR_MHA_M1_FAST`.
   (a2) as new NEON functions, only if sized. Mark each with `ponytail:`:
   "CPU decode attention; deleted when ATTN_M1 is resident and fast
   (E2E)".
2. Move the step-0 prototypes into the library and point `MhaM1Phases` at
   the exported functions. Add an `AttnM1F16Det` case that runs the new
   kernels against `attn_m1_det.h` at the eight lengths.
3. Gates:
   * Rung 0.
   * Rung 1: `ALL CHECKS PASS` and `WORKER POOL LANES OK`, plus
     `run_inproc_e2e.sh` `INPROC E2E PASS` unset and with
     `NNTR_MHA_M1_FAST=1` exported. The lfm25 fixture (32 / 8 heads) runs
     (a1) in f32. Add one line: `E2E eval golden-lfm25-mhafast …
     bit_identical=1` against the unset run.
   * Rung 3: app + fp16 gtest.
   * `llvm-objdump -d` of the staged `libnntrainer.so`: the new kernels
     show `fmla .8h`, `faddp .8h`, scalar `fadd h` and no contraction the
     spec does not model (#152 step 6).
   * The host cannot run the fp16 NEON path. For (a2), G8 on the device is
     the first real gate.

### Step 2: lever (b) (implementer, host; only if P′ passed)

1. The window hook (§2), then the touch of the next layer's first FC weight
   (S, T from the P′ cell).
2. Gates: rung 1 with `NNTR_MOE_PREFETCH_MIB` unset and `=<S>`. Add
   `E2E eval dspq-lfm25-prefetch … bit_identical=1` (the dspq stand-in runs
   the hook). Then rung 3.

### Step 3: verdict sitting (orchestrator; **device measurement unavoidable**; ≈ 45 min)

Modelled on `158-dma-bypass.md`, with one set switched by env (rule 21).

| variant | env |
|---|---|
| **A** | nothing (committed default, G2 banners) |
| **B** | `NNTR_MHA_M1_FAST=1` |
| **C** | B + `NNTR_MOE_PREFETCH_MIB=<S> NNTR_MOE_PREFETCH_T=<T>` (only if step 2 ran) |

Sequence:
* Sanity at G=8.
* tok/s at prompt 512, G 64 / 512 / 1024, mirrored per G (A B C | C B A),
  `NNTR_OP_TIME=1` in every cell.
* G8 gtests on the set.
* Dumps A1 / B / C / A2 at G=64, then `NNTR_PPL_DECODE` nll at G=512 (A,
  B, C, A2).
* The 8-prompt text set at G=64.
* Ride-along, not a variant: the CPU `q40` control at G=1024, one run each
  with `NNTR_MHA_M1_FAST` unset and `=1`. This shows how much (a) lifts
  the CPU (§3.4).

Fill G1–G8 per variant.

### Step 4: flip (implementer)

Make the passing knob(s) the unset default (`=0` keeps the old path). Check
the banner on the device (one G=8 run, unset vs `=0`, texts identical). A
has no banner for (a), so add one stderr line at first use
(`[MHA] m1 fast: on (split4=… ilp=…) source=default`) so the next sitting's
A can be checked.

## 5. Risks

* **fp16 path invisible on the host.** x86 runs `mha_core`'s f32 branch.
  Only (a1)'s index logic is host-gated. For (a2), the device gtest (G8) and
  the E2E dumps are the gate, and the objdump check catches a compiler
  fusing differently.
* **DMA rate / bypass contention (b).**
  - Rule 44: ≈ 13 GB/s is left beside a 57 GB/s ring.
  - A touch that slows the ring shows as `dmm_pct` in the probe, and in the
    app as `[OP-TIME] moe call_us` C vs A.
  - A touch that outlasts the call shows as `touch_late`.
  - `net` is read cold **and** warm because #90's S=10 flipped sign between
    them.
* **DVFS.** Decode runs the CPU at 2.0–3.1 GHz (rule 46). Splitting the
  softmax across 4 threads changes the load the governor sees. The
  microbench (steady, no DSP) and the in-app attention row can disagree, and
  the coverage ratio C in step 0 and the B-vs-A attention row in step 3 make
  that visible.
* **Thermal drift between sittings.** Step 0 and step 3 are different
  sittings, so nothing is compared across them. Each has its own A, the
  cells are mirrored, and zone0 is logged at every checkpoint. Step 0's
  numbers are projections only.
* **Stale skel.** The probe's echo assert fails on a pre-#159 skel. The app
  prints `dma_bypass=0` there, which voids A (G2), and the md5 is compared
  on the device.
* **Pool races.** (a1) adds one `parallel_for` per attention layer on the
  ARM `ThreadManager`. This is not the DSP pool of #136. A job-boundary race
  would show as `bad` in the 50-iteration gtest and as `bit_identical=0` in
  G3.
* **Address-space budget:** not touched. (a) is CPU-only. (b) reads
  CPU-mapped weights, and no DSP arena or VTCM change is involved.
* **Goal accounting:** (a) lifts the CPU control too (§3.4). A G=1024 NPU
  ≥ 50 with a CPU also ≥ 50 meets #162's gate but not the contract's
  "above the CPU". The step 3 ride-along records this, and the supervisor
  decides.

## 6. Docs to update (supervisor, from the filled measurements)

* **BENCHMARK.md:**
  - Step 0 as a side table (probe cells under the bypass; microbench;
    A G=1024 re-read with OP-TIME).
  - Step 3's A / B / C rows at G 64 / 512 / 1024.
  - The Goals "now" if G1–G8 pass.
  - A note on the CPU-control ride-along.
* **LEDGER.md:**
  - ㉙ status.
  - A new rule for the prefetch overlap under the bypass (resolves rule 44
    (2) for #90's cell).
  - Rule 46 (3) refined with the attention phase split.
  - §3: the E2E attention item (exact ATTN_M1 ≤ CPU parity; #146
    re-scope; §3.3 arithmetic).
  - §2 row for #162 after step 3.
