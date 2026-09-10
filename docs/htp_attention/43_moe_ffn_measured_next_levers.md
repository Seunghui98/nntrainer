<!-- SPDX-License-Identifier: Apache-2.0 -->

# 43 — MoE FFN on HTP: what the device actually says, and the ranked levers

**Status: the HTP MoE FFN path runs correctly and is measurably NOT faster than
CPU.** This doc records the per-stage device measurements that show *why*, the
arithmetic for each remaining lever, and the protocol for working on them one at
a time. It supersedes `41_moe_ffn_e2e_and_perf_task.md` §5's *ordering* (which
was written from a cost model that the device has now contradicted twice); §5's
list of already-rejected ideas still stands.

**As of 2026-09-09: L1 (parallel WH bake) is in and verified. L2 (fused
gate_up→SwiGLU→down) is DISABLED (`kFusedSwigluEnabled = false` in
`lfm2_moe_layer.cpp`) — TWO structurally different implementations both
produce wrong model output on this model's real weights, despite both passing
thorough unit tests (including a shape sweep matching the real model's own
per-expert M values) and despite the ARM-side call arguments being confirmed
correct via device instrumentation. Root cause of the shared failure has now
been LOCALIZED (not yet device-confirmed): `hvx_swiglu_f32.c`'s
`hvx_recip_qf32` produces NaN for any `gate <= ~-87.98`, a range its own
caller's `exp_top = 88.0f` clamp does not exclude — see §7's last row for the
arithmetic and §5's L2 section for the fix. Do not re-enable L2 without
device-verifying the fix; §7's differential-test recommendation still stands
as the acceptance gate.**

Device: Galaxy S25 Ultra `R3CY10WM83Y`, V79 CDSP, LFM2.5-8B-A1B
(`--fc_dtype Q4_0 --moe_dtype QS4CX --isa ARM`), 444-token prompt / 512
generated, `moe_engine=htp`, `moe_htp_layers="2"` (1 of 22 MoE layers).

---

## 1. The headline: 83% of a call is not the multiply

Measured, `NNTR_HTP_PROFILE=2`, prefill (`M>1`) shapes, one HTP layer:

| stage | gate_up (K2048 N3584) | down (K1792 N2048) | attackable? |
|---|---|---|---|
| **host (wall, per call)** | **1036.9 µs** | **701.5 µs** | |
| FastRPC transport | 556.8 (54%) | 423.3 (60%) | **yes** — payload is f32 both ways |
| HMX multiply (host − stages) | ~174 (17%) | ~87 (12%) | no — 0.019 µs/mm, matches `34` §5.1 |
| dequant (HVX) | 94.1 (9%) | 54.7 (8%) | **yes** — single-threaded today |
| drain (weight DMA wait) | 95.8 (9%) | 43.4 (6%) | partly — 35 GB/s, fully exposed |
| quant (HVX) | 76.3 (7%) | 64.4 (9%) | **yes** — only via fusion |
| acc_read | 39.9 (4%) | 23.3 (3%) | no — vendor code |

**The multiply the accelerator exists for is 17% of the call.** Everything else
is moving bytes and changing their type. That is the whole problem statement,
and it is the number to attack — not the kernel.

Per layer, prefill: 32 gate_up × 1036.9 + 32 down × 701.5 = **55.6 ms**
(matches the profile's own `layer calls total`).

### 1.1 Against CPU, honestly

CPU-only prefill for the **whole 24-layer model** measured 1415 / 2262 / 2109 ms
across three runs in the same session (313.8 / 196.3 / 210.5 TPS). So even if
the MoE FFN were 100% of CPU prefill, CPU spends **≤ 64 ms per MoE layer**.
HTP spends **55.6 ms**.

⇒ **There is no measured per-layer win to amortize anything against.** HTP is at
parity at best, today, before registration is counted at all. Every "HTP is
2.2× faster" number in `34_fc_measured.md` is a *DSP-only* number for a *single
narrow FC*; at MoE widths with FastRPC in the loop it does not survive.

**The one number nobody has measured yet: CPU's actual per-layer MoE FFN prefill
cost.** The ≤64 ms above is a bound, not a measurement. Get it (§5, item M0)
before trusting any speedup claim.

## 2. Registration is the gate on the whole approach

| | measured | note |
|---|---|---|
| convert to registry (ARM) | **9.91 ms/weight** | was 25.40; tiled transpose landed, 2.6× on device |
| register FastRPC (DSP WH bake) | **34–48 ms/weight** | noisy across runs; never optimised |
| total, 64 weights (1 layer) | **3692 ms** | lands entirely inside the reported prefill |

Extrapolated to the full model (22 layers × 32 experts × 2 = **1408 weights**):

- **~81 s of startup** at 57 ms/weight
- **~3.9 GB of DSP heap** for WH bytes (`K*N/2` per weight)
- `HEXKL_MM_U8I4_MAX_WEIGHTS = 512` ⇒ **only 8 layers can be registered at all**

So: prefill for a 1-layer test is ~98% registration (3692 of the ~4000-4800 ms
delta), and the full model cannot be enabled regardless. **Registration is not
a polish item, it is the precondition.**

The DSP bake is `hexkl_weight_u8i4_register`'s loop over `k_tiles × n_tiles`
calling `hexkl_micro_hmx_rm_to_wh_i4` — for gate_up that is 64 × 112 = **7168
tiles at ~6.6 µs/tile**. Each tile reads a distinct source region and writes a
distinct VTCM offset: **embarrassingly parallel, and the 5-worker HVX pool that
the session already creates is not wired into it.**

## 3. Verified this session (do not redo)

| finding | evidence |
|---|---|
| HVX worker pool is already maximal | `qurt_hvx_get_units()=0x600` → `n_hvx=6`, `pool_workers=5` (logged from device). Top-tier V79 config; `n_hvx-1` is correct |
| poll-mode QoS is on and works | `qos_mode=2`; transport per call 423–557 µs vs `34` §2's 326 µs reference at 4× the payload |
| decode must stay on CPU | HTP-on vs HTP-off decode: **25.81 vs 25.73 TPS** (identical). Full-model HTP decode ceiling computed at 17.5 TPS vs CPU 25.9. `accelerates_q4_0_at_m1()` returns false for this reason |
| cross-matmul weight prefetch does nothing at MoE widths | drain/weight-bytes = **35 GB/s with 4 handles and with 1** — nothing hides behind anything. `34` §4C's 75→55 µs was a narrower weight |
| moving quant to ARM is the wrong direction | ARM-side quantize measured 300–420 µs/call (200 µs after a magic-number round), against ~100–150 µs of transport it saved. HVX already writes quantized AH tiles **straight into VTCM** for HMX to read — no DRAM detour ever existed for that step. Reverted; `htp_act_quant.*` + `mm_u8i4_layer_u8in` remain in-tree, unused |
| `htp_qs4cx_from_*`'s transposed store was cache-hostile | tiled 64×64 transpose: 2.2–3.1× on host, 25.40 → 9.91 ms/weight on device, bit-identical output |

## 4. The device's measurement behaviour — read before believing any number

1. **Run-to-run drift is larger than most effects you will chase.** Three
   consecutive identical HTP runs measured decode 25.1 → 16.2 → 14.5 TPS and
   prefill 92.7 → 56.9 → 42.5 TPS. Temperatures stayed 31–56 °C, far below any
   throttle point, and a CPU-only control degraded the same way — so it is not
   thermal and not the code.
2. **Therefore every claim needs a CPU-only control run in the same window.**
   `moe_engine=cpu` → measure → `moe_engine=htp` → measure → `moe_engine=cpu`
   again. If the two controls disagree by more than the effect, the run is void.
3. **`NNTR_HTP_PROFILE`'s `host` column wraps only the FastRPC call.** Anything
   the ARM side does around it (quantize, memcpy, allocation) is invisible.
   This exact blind spot made an ARM-side quantize look free when it cost more
   than it saved. Time ARM-side work separately, or you will fool yourself.
4. **Registration lands inside the reported prefill**, once per process. Subtract
   `registration total` before comparing prefill against anything.
5. **The profile block only prints if HTP was actually called.** No
   `[HTP-PROFILE]` under `NNTR_HTP_PROFILE=2` means zero dispatch — check
   `moe_engine`/`moe_htp_layers` in the config **on the device**, not on the host.

---

## 5. The ranked levers, with the arithmetic

Ordered by measured µs attacked ÷ effort. Each is a standalone task; §6 is the
protocol.

### M0 — measure CPU's per-layer MoE FFN prefill cost (do this first, no code)
Everything below is compared against a **bound** (≤64 ms/layer), not a number.
Add a per-layer timer around `Lfm2MoELayer::forwarding`, or diff prefill with
`num_dense_layers` raised so one MoE layer becomes dense. Cheap, and it decides
whether any of the rest is worth building.
**Gate:** a µs figure for one CPU MoE layer's FFN at prefill, bracketed.

### L1 — parallelise the DSP WH bake across the existing HVX pool
**Attacks:** 34–48 ms/weight → target ~12 ms/weight (5 workers).
**Why it is safe:** each `(kt, nt)` tile reads a disjoint source and writes a
disjoint VTCM offset; `hvx_worker_pool_run` already exists, is already sized to
5, and is already handed to every layer call. `hexkl_weight_u8i4_register` takes
no pool today — thread it through from `nntr_hvx_weight_register_u8i4`.
**Payoff:** 1-layer registration 3692 → ~1000 ms. Full model 81 → ~23 s.
**Files:** `hexkl_mm_u8i4_dma.c` (`hexkl_weight_u8i4_register`),
`hexkl_mm_u8i4_dma.h`, `test/htp/nntr_hvx_mm_u8i4.c`. Skel rebuild required.
**Gate:** registration ms/weight drops; WH bytes bit-identical to the serial
bake (compare a registered weight's matmul output against the current build).

### L2 — fuse gate_up → SwiGLU → down into one DSP call
**Attacks:** one whole transport round trip + one quant + one dequant per expert.
**Arithmetic per expert (prefill, M≈55):**

| | today | fused |
|---|---|---|
| calls | 2 | 1 |
| wire bytes | 450 KB + 787 KB + 394 KB + 450 KB = **2081 KB** | 450 KB in + 450 KB out = **900 KB** |
| transport | 556.8 + 423.3 = 980 µs | 150 fixed + 900 KB/3.0 GB/s ≈ **450 µs** |
| quant | 76.3 × 2 | 76.3 × 1 |
| dequant | 94.1 + 54.7 | 54.7 × 1 |
| dsp total | 758 µs | ~625 µs (incl. ~50 µs HVX SwiGLU) |
| **host** | **1738 µs** | **~1075 µs → 1.6×** |

Per layer: 55.6 → **~34.4 ms**.
**VTCM check (passes):** gate_up WH 3.67 MB + down WH 1.84 MB + act 128 KB +
u8 intermediate ~128 KB + result tile 8 KB ≈ **5.8 MB of ~8.3 MB** — fits
**single-buffered**. Double-buffering both weights (11 MB) does not fit, but
that costs nothing: the prefetch is already measured to hide nothing (§3).
**New code:** SwiGLU on HVX (`hvx_exp_f32.h`'s `hvx_exp_sf` gives the sigmoid),
requantize the intermediate in VTCM, a two-handle entry point, and collapsing
`Lfm2MoELayer::compute_expert_forward_no_critical`'s two `dot()` calls into one
op. This is the largest item here and the only one that changes the verdict.
**Gate:** SNR in dB of the fused layer's output against the unfused path (never
max-relative-error — `40` §6.2), plus the bracketed wall-clock.

**2026-09-09 root-cause candidate (host-verified arithmetic, not yet
device-confirmed):** `hvx_swiglu_f32.c`'s `hvx_recip_qf32(a)` computes its
Newton-Raphson seed as `0x7EF311C2u - bits(a)` (unsigned subtraction on the
raw f32 bit pattern). That is only valid while `bits(a) <= 0x7EF311C2`, i.e.
`a <= ~1.615e38`. The caller (`hvx_swiglu_row_f32`) computes `a = 1 +
exp(min(-gate, 88.0f))`; bisecting on host (magic seed and 3 NR steps taken
verbatim from the shipped code) puts the actual safe boundary at
`-gate <= ~87.978`, i.e. **`gate <= ~-87.98` always makes `bits(a) >
0x7EF311C2`, the subtraction wraps around modulo 2^32, and the wrapped bit
pattern reinterpreted as f32 is NaN** (verified for gate ∈
{-88, -90, -100, -120, -200}, all producing `-nan`; gate = -87.4 still
correct). The `exp_top = 88.0f` clamp — added specifically so `-gate` never
leaves `hvx_exp_sf`'s documented domain — does NOT exclude this range: it
clamps everything below -88 to exactly -88, which is itself 0.02 past the
recip breakpoint, so **every gate at or below the clamp lands in the NaN
zone**, and un-clamped values from -87.98 to -88 do too even without hitting
the clamp. A NaN in one SwiGLU output lane poisons that row's
`hvx_quant_rows_u8_params` min/max scan (this function scans the WHOLE row to
pick one scale/zp per row), which corrupts that row's requantized scale/zp to
NaN, which propagates through the down matmul to the entire output row for
that token — consistent with the observed failure mode (coherent-looking
prefill followed by early, wrong output) and with BOTH L2 implementations
failing identically (both call the same `hvx_swiglu_inplace_f32`). It is also
consistent with every unit test passing: `fill_deterministic`'s synthetic
activations are bounded and do not plausibly produce a raw SwiGLU gate
pre-activation below -88, while a real K=2048 int4 matmul accumulation over
real model weights plausibly does, at least in some row/column/expert/layer
combination across 32 experts × the layers exercised.
**Fix candidates, cheapest first:** (a) lower `exp_top` to a value that keeps
`a` inside `hvx_recip_qf32`'s valid seed range with margin (something like
`87.0f`, re-derive the exact bound rather than eyeballing it) — one constant,
smallest diff; (b) replace the raw-bit-subtraction seed with a form that
saturates instead of wrapping for large `a` (e.g. clamp `bits(a)` to
`0x7EF311C2` before the subtraction, so the seed floors at the smallest
representable positive reciprocal instead of going negative); (c) skip the
reciprocal path entirely when `t` is already at the clamp and hard-code
`sig = 0` (sigmoid of a very negative gate rounds to 0 in f32 anyway, which is
exactly what the reference `expf` path already produces via
overflow-to-inf → 1/(1+inf) == 0). **Whichever fix is chosen, re-run
`GateUpSwigluMatchesHostIntermediate`/`FusedSwigluMatchesTwoCallReference`
with a test row that deliberately includes a gate value ≤ -88 (neither
existing test does — that is *why* both passed) before touching the real
model again**, then re-enable `kFusedSwigluEnabled` and repeat the full
model run + bracketed CPU control from §6.3.

### L3 — send the layer's activation once, not once per expert
**Attacks:** the 4× activation duplication. `topk=4` means every token's row is
sent to 4 different experts, so a layer ships 1776 × 2048 × 4 = **14.5 MB** of
activation for a 3.6 MB input. With a token-index list per expert and a DSP-side
gather, that is 3.6 MB. If the DSP also does the routing-weight multiply and the
scatter-add, the output side collapses the same way (14.5 → 3.6 MB).
**Payoff, stacked on L2:** per-layer wire 66.6 MB → ~7.2 MB, and with all 32
experts in one call the fixed transport collapses to a single 150 µs.
Per layer ≈ **22.5 ms** (2.5× on today's 55.6).
**Cost:** this is "one DSP call per MoE layer" — the endgame shape. Needs the
handle-array call to also take per-expert row-index lists.
**Do not start before L2 lands and measures.**

### L4 — parallelise dequant
**Attacks:** 94.1 µs (gate_up) + 54.7 (down) per call, 9%/8%.
`hvx_dequant_i32_to_f32` and `hvx_dequant_acc_tile_to_f32` take **no pool
argument** while quant does. Not free: the in-place tile dequant runs inside the
HMX loop on the single result tile, so parallelising it needs the second result
buffer from `37_t2_pipelining_design.md` §3, and per-tile fork/join is already
rejected (`35` §5). **L2 deletes one of the two dequants outright — do L2 first
and re-measure before touching this.**

### L5 — zero-copy FastRPC payload (`dmahandle`)
**Attacks:** the bytes half of transport (74% of it, by the fit
`transport ≈ 150 µs + payload/3.0 GB/s`). The IDL is `in sequence<float>` /
`rout sequence<float>` everywhere — marshalled copies; **no `dmahandle` exists
anywhere in the tree**. `34` §4F's 44.6 GB/s was measured at 52 MB payloads; at
1.25 MB we get 2.25 GB/s.
**Why it ranks last:** L2 and L3 remove ~9× of the bytes first, after which this
is worth much less. Microbenchmark it before building it.

### Not worth building (measured)
- **Prefill multi-activation grouping alone** — the fixed cost is 150 µs, not
  the 326 µs the earlier plan assumed, so collapsing 64 calls → 2 saves
  4.65 ms of 55.6 ms (8%) for a new DSP entry point. Only take it as part of L3.
- **ARM/NEON activation quantize** — §3, measured backwards.
- **M=1 GEMV instead of HMX** — decode is out of scope; see §3.
- **DMA/prefetch tuning** — already at 35 GB/s, handle-count-independent.

---

## 6. Protocol for one-lever-at-a-time work

### 6.1 Build and deploy (all four steps, in order)

```bash
# 1. ARM library
cd ~/workspace/nntrainer/builddir
ANDROID_NDK=~/workspace/android-ndk-r26d PATH="$ANDROID_NDK:$PATH" ninja

# 2. TRAP: ninja installs to builddir/jni/, but the app links against
#    builddir/android_build_result/. Copy or you will test the old library.
cp jni/arm64-v8a/libnntrainer.so android_build_result/lib/arm64-v8a/

# 3. DSP skel (only if you touched hmx/*, hvx/*, test/htp/*, or the IDL)
cd ~/workspace/nntrainer
HEXAGON_SDK_ROOT=~/workspace/Hexagon_SDK/6.4.0.2 \
  bash nntrainer/tensor/htp_backend/generate_stub.sh    # if the IDL changed
cd test/htp
HEXAGON_SDK_ROOT=~/workspace/Hexagon_SDK/6.4.0.2 \
DEFAULT_HEXAGON_TOOLS_ROOT=~/workspace/Hexagon_SDK/6.4.0.2/tools/HEXAGON_Tools/19.0.04 \
HEXKL_ROOT=~/workspace/Hexagon_SDK/6.4.0.2/addons/hexkl_addon \
HEXKL_SDK_VER="" bash build.sh
# TRAPS: setup_sdk_env.source derives paths from $PWD (cd into the SDK first or
# it sets HEXAGON_SDK_ROOT to your repo); this SDK's lib path has no version
# subdirectory, hence HEXKL_SDK_VER="".

# 4. App relink + push
cd ~/workspace/nntrainer/Applications/CausalLM/jni
rm -f obj/local/arm64-v8a/{libnntrainer.so,libcausallm_core.so,nntrainer_causallm}
ANDROID_NDK=~/workspace/android-ndk-r26d PATH="$ANDROID_NDK:$PATH" \
NNTRAINER_ROOT=~/workspace/nntrainer \
  ndk-build NDK_PROJECT_PATH=. NDK_LIBS_OUT=./libs NDK_OUT=./obj \
    APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk \
    causallm_core nntrainer_causallm -j$(nproc)
# TRAPS: NNTRAINER_ROOT is preset wrong in the shell profile; ndk-build will
# not relink against a changed prebuilt unless you delete the outputs first.
D=/data/local/tmp/nntrainer/causallm
adb push obj/local/arm64-v8a/libnntrainer.so $D/
adb push obj/local/arm64-v8a/nntrainer_causallm $D/
adb push ~/workspace/nntrainer/test/htp/build/libnntr_hvx_skel.so $D/   # if rebuilt
adb shell "chmod 755 $D/nntrainer_causallm $D/libnntr_hvx_skel.so"
adb shell "md5sum $D/libnntrainer.so"   # must match your local build
```

### 6.2 Measure (bracketed, or it does not count)

```bash
M=$D/models/lfm2.5-8b-a1b-q40-qs4cx
RUN="adb shell 'cd $D && LD_LIBRARY_PATH=$D ADSP_LIBRARY_PATH=$D \
     NNTR_HTP_PROFILE=2 ./nntrainer_causallm $M'"
# edit $M/nntr_config.json's moe_engine on the HOST, then adb push it, then run:
#   cpu -> htp -> cpu
```

Record for every run: prefill ms/TPS, decode ms/TPS, `registration total`,
`layer calls total`, and the per-shape `dsp`/`transport`/`[quant dequant acc
drain]` line. Report `prefill_ms − registration_ms` as the comparable number.

### 6.3 Accept / reject

Keep the change only if **all** hold:
1. Correctness: the generated text is still coherent **and** the accuracy gate
   for that lever passed (bit-identical where claimed, SNR in dB otherwise).
2. The stage it targeted moved in the profile, by roughly the predicted amount.
3. `prefill_ms − registration_ms` improved, with both CPU controls agreeing
   within less than the effect.

If (2) holds but (3) does not, **something outside the timed region ate it** —
find it before keeping the change (§4.3). If the controls disagree by more than
the effect, let the device idle and re-run; do not average across a drifting
sequence.

Append the result to §7 of this doc, pass or fail. A measured failure is the
more valuable record — three of §3's rows are failures.

## 7. Results log

| date | lever | result | numbers |
|---|---|---|---|
| 2026-09-08 | tiled transpose in `htp_qs4cx_from_*` | **kept** | convert 25.40 → 9.91 ms/weight, bit-identical |
| 2026-09-08 | decline M==1 (decode) on HTP | **kept** | decode 25.81 (HTP) vs 25.73 (CPU) — HTP no longer dispatched at decode |
| 2026-09-08 | poll-QoS + ION act/out buffers | **kept** | `qos_mode=2`; transport 423–557 µs at 4× `34` §2's payload |
| 2026-09-08 | QS4CX batch dispatch (`float_tensor.cpp` vector `dot`) | **kept, now dormant** | grouped decode reached HTP (N=14336 bucket appeared); unused since decode left HTP |
| 2026-09-08 | ARM-side u8 activation quantize (`u8in`) | **reverted** | dsp −19…−26%, wall unchanged: ARM quantize 300–420 µs ate it |
| 2026-09-08 | L1: parallel DSP WH bake across the HVX pool | **kept** | register FastRPC 34–48 → 19.1–19.4 ms/weight (registration 3692 → 1591–1636 ms / 64 weights, ~2.3×, full-model 81 → ~35 s); bit-identical: serial vs parallel bake, FNV-1a of the f32 layer output at M64/K2048/N3584 = `0x198748e597cf4105` both, unittest_hvx_mm_u8i4 15/15; bracketed prefill−registration 1774–1837 ms sits inside the CPU control band (1544–1771). Under the ~12 ms/weight ideal by Amdahl: the bake itself scaled ~5–6×, the per-weight serial residue (malloc + 1.8–3.7 MB WH memcpy + RPC fixed cost ≈ 12 ms) is now the register line's floor — that residue, not the pool, is the next registration lever if one is ever needed |
| 2026-09-08 | L2: fused gate_up→SwiGLU→down single DSP call (QS4CX `gemm_qs4cx_fused_swiglu_fp32`, DSP-internal 64-row blocking, M==1 excluded) | **kept, under prediction** | host 1478.2 (908.7+569.5) → **1297.1 µs/call (−12%)**, layer calls 48.6 → 41.5 ms; transport 770.3 → 535.2 µs (one round trip, but not the ~325 single-call floor — the fused call ships act+out together); swiglu absorbed on-DSP at 56 µs/call; dsp 761.9 µs/call. Unfused rows gone (calls=32/rows=1776 all fused). SNR 139.365 dB (≥40), unittest 16/16, coherent text (206 tok — early EOS vs CPU's 512: different sample, same gate). prefill−registration 1860 → 1588 ms, controls 1510/1378 (spread 132 < effect 272). One ENOMEMORY iteration: VTCM layout sized for full M (m_pad=128 ⇒ ~8.75 MB > 8.3 MB, AEE_ENOMEMORY=0x80000402 on expert 6) → layout fixed at one 64-row block, block loop is DSP-internal (no extra RPC). Still ~10% behind the CPU control (~1444 ms) — consistent with M0: L2 alone is parity territory, L2+L3 is the flip |
| 2026-09-08 | M0: CPU MoE FFN per-layer prefill cost (`NNTR_M0_PROFILE` per-layer timer in `Lfm2MoELayer`) | **measured** | **~38 ms/layer bracketed**: 38.2 / 38.8 ms mean over all 22 layers in two CPU controls (1.6% apart), per-layer band 33–49 ms (one control showed a 7-layer 48 ms slow mode — §4.1 drift, gone in the other), total 841/853 ms ≈ **53% of CPU prefill**. Same instrument, htp run: the HTP layer's wall is 68 ms (registration subtracted) vs the profile's own 47.3 ms layer total ⇒ ~21 ms of ARM-side work the host column never sees (§4.3) — so HTP today is **1.5–1.8× slower per layer than CPU**, and L2's fused 34.4 ms DSP target lands at CPU parity, not a win: L2 alone does not flip the verdict, L2+L3 (~22.5 ms/layer) is what could |
| 2026-09-09 | **CORRECTION to the L2 row above** | **L2 reverted — the "kept" verdict was wrong** | The row's own "early EOS vs CPU's 512: different sample, same gate" was the bug, not a benign sampling difference. Full model run, same process, back-to-back: `moe_engine=cpu` → correct coherent 3-sentence summary, 512 tokens. `moe_engine=htp` with L2 live → **"Could you please provide the text you would like summarized?"**, 206 tokens — the model losing its own prompt. `FusedSwigluMatchesTwoCallReference`'s synthetic-weight SNR (77–139 dB, M∈{55,64,100,128}) does not reproduce this: whatever is wrong needs this model's real registered gate_up/down weights or a real router activation, not `fill_deterministic`'s pattern, to show up. Disabled via `Lfm2MoELayer::compute_expert_forward_no_critical`'s `constexpr bool kFusedSwigluEnabled = false` (code kept, not deleted). **Do not re-enable without a differential test against this model's actual weight bytes, staged per pipeline step (quant1 → gate_up mm → dequant-split → SwiGLU → quant2 → down mm), not just final-output SNR** — that is what would localize the divergence the unit test's synthetic data does not hit. |
| 2026-09-09 | **CORRECTION to L1: the "reproducible-but-WRONG weight image" claim was a self-inflicted test bug, not a real one** | **L1 restored** | Chasing L2's bug, an FNV-1a hash compare (real 5-worker pool bake vs a "serial reference") showed a mismatch and was read as HMX-lock corruption from running `hexkl_micro_hmx_rm_to_wh_i4` off the lock-holding thread; L1 was reverted to a manual serial loop on that basis. Root cause was actually in the test: the "serial reference" forced `pool=NULL` into `hvx_worker_pool_run(pool, func, ctx, n_units)`, whose NULL branch is `func(n_units, 0, ctx)` — **not** "run everything on one thread", but "run AS IF n_units threads existed and this call is worker 0's 1/n_units slice" (`lo=n_tiles*0/n_tiles=0, hi=n_tiles*1/n_tiles=1` for n_units=n_tiles — bakes tile 0 only, leaves the rest as stale VTCM scratch). That mis-baked image was what got compared against, not a true serial bake. Re-verified: manual-loop serial and the real N-worker pool (N≈5–6, `n_units` unaffected by the NULL bug since the pool is non-NULL) produce the **same** FNV-1a (`0x198748e597cf4105`, M=64/K=2048/N=3584), `unittest_hvx_mm_u8i4` 16/16. L1 restored (`hexkl_weight_u8i4_register` takes `pool` again, worker-split bake). Device re-measured post-restore: register FastRPC 46.86 → **18.90 ms/weight**, registration total (1 layer, 64 weights) 3819.6 → **1674.7 ms**. Full model run with L2 still disabled: coherent 512-token output, `swiglu=0.0` (confirms L2 dormant), decode unaffected. **Lesson for whoever reads this next: a NULL pool argument to `hvx_worker_pool_run` is only safe when `n_units <= 1` at that call site — check the call, not just the function's one-line doc summary, before using NULL as a "serial" reference for anything.** |
| 2026-09-09 | **L2, attempt 2: split-call reimplementation** — `hexkl_mm_u8i4_gate_up_swiglu_run` (one weight: gate_up matmul → SwiGLU → requantize to u8 AH, new, smaller-surface DSP function) feeding the already-verified `mm_u8i4_layer_u8in` path for down, instead of the one-call six-region `hexkl_mm_u8i4_fused_run` | **reverted again — same real-model failure, root cause still not found** | Built specifically to have a smaller, more inspectable VTCM layout (one weight, no dual-weight double-buffer) than attempt 1, and tested far more rigorously: a NEW stage-1-only test (`GateUpSwigluMatchesHostIntermediate`) compares the requantized intermediate against a host-computed (never-quantized) reference directly, not just the final output — 37.5–37.7 dB, stable across M ∈ {11,55,64,100,128,138,200}, calibrated against the project's own single-hop floor (23.5 dB) rather than copying attempt 1's 40 dB gate blind. End-to-end (`GateUpSwigluPlusU8InMatchesTwoCallReference`): 138.9–140.8 dB, **stable across the same M range — including 128/138/200, where attempt 1's own test still drops to 77–79 dB**, so this version does not have attempt 1's M-anomaly. Wired into `gemm_qs4cx_fused_swiglu_fp32` (same ARM entry point, different internal implementation) and re-enabled. **Same failure on the real model regardless**: `moe_engine=htp` still produces "Could you please provide the text you would like summarized?" (206 tok) where CPU gives the correct 512-token summary. Added `NNTR_L2_DEBUG` prints of the actual dispatch arguments during a real run (K=2048, N=[3584,2048], per-expert M observed 11–138 exactly matching what was unit-tested, all weight/activation/output pointers distinct and consistent) — **the ARM-side call construction is confirmed correct**, ruling out a shape or wiring bug in `Lfm2MoELayer`'s caller code. Two structurally unrelated DSP kernels, both passing thorough shape-swept unit tests at the model's own real M values, both reproducing an identical failure on the real model: **the common factor is not either kernel's arithmetic correctness (verified) or the ARM-side arguments (verified) — it is something neither test controls for**, most likely a systematic (not merely random) bias in `hvx_swiglu_f32.c`'s HVX exp/reciprocal approximation that a per-call SNR-vs-synthetic-data metric cannot catch, potentially compounding across 32 experts and everything downstream. Reverted to `kFusedSwigluEnabled = false` (both kernel implementations kept in the tree, dormant); temporary `NNTR_L2_DEBUG` instrumentation removed after use. **Do not attempt a third implementation before running a differential test against this model's own registered weight bytes and a captured real activation** — every attempt so far has used `fill_deterministic`'s synthetic pattern for both, which is the one variable common to every passing unit test and every failing real run. |
| 2026-09-09 | **L2 root cause candidate, host-verified, not yet device-confirmed**: `hvx_swiglu_f32.c`'s `hvx_recip_qf32` NaNs on large `a` | **found, not yet fixed** | Read-only re-analysis of the two failed attempts (no new kernel written). `hvx_recip_qf32`'s Newton-Raphson seed is `0x7EF311C2u - bits(a)`, unsigned subtraction on the raw bit pattern, valid only while `bits(a) <= 0x7EF311C2` (`a <= ~1.615e38`). The caller computes `a = 1 + exp(min(-gate, 88.0f))`; a host bisection using the shipped magic seed and 3 NR steps verbatim puts the real safe boundary at `-gate <= ~87.978` — **0.02 below the `exp_top = 88.0f` clamp that was specifically added to keep `-gate` inside `hvx_exp_sf`'s domain**. Confirmed on host: gate ∈ {-88, -90, -100, -120, -200} all produce `-nan` through the exact seed+NR sequence; gate = -87.4 is still correct (matches `1/(1+expf(-gate))` to 6 digits). Every gate at or below the clamp — and everything the clamp maps to exactly -88 — lands in the NaN zone; there is no gate value that reaches the clamp and stays safe. A single NaN SwiGLU lane poisons `hvx_quant_rows_u8_params`'s whole-row min/max scan, corrupting that row's requantization scale/zp, which the down matmul then spreads across the entire output row for that token — matches the observed failure (coherent-looking generation truncated by garbage) and explains why BOTH L2 kernels fail identically (both call `hvx_swiglu_inplace_f32`) while every unit test passes (`fill_deterministic`'s bounded synthetic activations don't plausibly produce a raw SwiGLU pre-activation ≤ -88; a real K=2048 int4 matmul accumulation over the model's real weights plausibly does, in at least one row/expert/layer out of 32 experts). **Not yet reproduced on-device or confirmed as THE cause** — this is host arithmetic on the algorithm as shipped, not a captured real activation dump (§7's own prior row's ask). Fix candidates and the required new test case (a gate ≤ -88 row, which neither existing SwiGLU test covers) are in §5's L2 section. **Next action for whoever picks this up: apply one of §5's L2 fix candidates, add the missing extreme-value test case first, re-run the existing SNR gates, then re-enable `kFusedSwigluEnabled` and repeat the full bracketed-CPU-control model run before declaring L2 fixed.** |
| 2026-09-09 | **L2 fix candidate (a) applied: `exp_top` 88.0f → 85.0f**, plus the first test that actually enters the failing range | **written, NOT device-verified — the fail-before/pass-after run is the next session's** | Boundary re-derived independently rather than taken from the row above, and it holds: `bitcast<float>(0x7EF311C2) = 1.61547303e38`, analytic `log(bitcast − 1) = 87.977861348`, bisection over the shipped seed + 3 NR steps transcribed verbatim = `87.977863312` (agree to 2e-6 — so the seed-sign criterion *is* the boundary, not just a bound on it); host repro gives `-nan` at gate ∈ {−87.98, −88, −90, −120, −200} and correct output at −87.4. **New, and the reason 85.0f rather than the ~87.0f the row above suggested: the NaN edge is not where accuracy ends.** The reciprocal's documented 1e-6 relative error holds only to `t = 87.543697`; at `t = 87.9` it is ~10% off and at `87.97` ~5× off — all finite, all silently wrong, all reachable under an 87.0 clamp. At `t = 85`: rel. err `1.55e-8` (65× inside spec), `+36,510,599` ULPs of headroom to the NaN edge — enough that a 100×-worse-than-documented `hvx_exp_sf` still could not cross it. The clamp discards nothing: the largest silu term it can suppress is `85·exp(−85) = 1.03e-35` per unit of `up`, 3.8e32× below one u8 step of an O(1) row, and the host reference is *exactly* 0 there anyway (`expf` → inf, `1/(1+inf) == 0`). `hvx_recip_qf32`'s doc comment said "must be nonzero and finite", which is what let 88.0f look safe; it now states the real domain. **Coverage hole that let both prior attempts pass every gate:** all three existing SwiGLU tests build the gate from `fill_deterministic × fill_deterministic`, whose \|gate\| never nears the clamp — so no test ever entered the failing range. New `SwigluSurvivesExtremeNegativeGate` drives three gate columns to −200/−300/−400 via a bias override (bias is the one term of the dequantized output a test sets directly), spread across separate 32-lane chunks, and `ASSERT_LE(g, -88.0f)` on the device's own `gu_out` first so it cannot pass by never testing anything; it then checks `out_scale` finiteness (where a NaN lane actually lands — `hvx_quant_rows_u8_params` scans the whole row for min/max) and the fused output's finiteness, so a failure names the bug instead of reporting "SNR was nan". Both kernels in one test, since both call `hvx_swiglu_inplace_f32`. `I = 1792 = 56×32` exactly ⇒ no scalar tail, so every column takes the vector path (the scalar remainder uses plain `expf` and was always correct — another reason this hid). **Nothing here has been run on device. Do not mark L2 fixed on this row.** |
| 2026-09-09 | **L2 attempt 3 (exp_top 88.0f → 85.0f) measured on device: the NaN hypothesis is DEAD, and the profile's two blind spots are now instrumented** | **L2 disabled again — but the search is bisected, not restarted** | The fix itself is correct and stays: `hvx_recip_qf32`'s seed really does diverge for `bits(a) > 0x7EF311C2`, host-verified two independent ways, and the clamp now sits at 85 with 36.5M ULPs of headroom. It is simply **not what breaks this model**. `NNTR_L2_CHECK` (new) scans the DSP's own per-row requant scales — where a NaN SwiGLU lane lands, since `hvx_quant_rows_u8_params` scans the whole row for min/max — and the down matmul's f32 output: **zero non-finite values on a full run whose text is still wrong** (identical 206-token "Could you please provide the text you would like summarized?"). Finite and wrong, not NaN and wrong. Three implementations have now failed the same way, so the fault is not in any one kernel's arithmetic. **The variable never controlled for is the DATA**: every gate this path has passed used `fill_deterministic` weights *and* activations, and the model's own registered weight bytes with a real captured activation have never been put through both paths side by side. `NNTR_L2_DIFF` (new, `HtpComputeOps::l2Diff`) runs the unit test's exact reference — `mm_u8i4_layer` on gate_up, host `expf` SwiGLU, `mm_u8i4_layer` on down — against the split-call path on the real weights and the real activation, per expert call, and prints SNR + max_abs_err. High SNR ⇒ the kernels agree and the fault is ARM-side plumbing; low SNR ⇒ they genuinely disagree on real data and the synthetic distribution was hiding it. **Run this before a fourth implementation.** Perf, measured this session with the new `mm<=` column: split-call **41.5–42.6 ms/layer** vs the two-dot HTP path's 55.6 (§1) — L2 is a 25% win, so performance is not why it is off. The `mm<=` residue puts the actual HMX matmul at **21.0% of host** (gate_up ≤185.3 µs, down ≤90.7 µs of 884.3/413.9): 37% is FastRPC transport, 33% format conversion, 13% weight DMA. **Correction to this session's own staging hypothesis:** the ARM staging memcpy, suspected of hiding a large cost outside every `host=` window, measures **1.4 ms per layer** (27.8 MB at 21.1 GB/s) — negligible, and §4.3's ~21 ms/layer of invisible ARM work is still unaccounted. Device drift note: across six runs `layer calls total` held 41.5–42.6 ms (±1%) while registration swung 1530→2157 ms and decode — which contains **no HTP at all**, the profile shows no `M==1` rows — fell 38.7→28.7 TPS; single-run perf comparisons on this phone remain worthless. |
| 2026-09-09 | **L2 root cause CLOSED, quantitatively confirmed: an intrinsic u8-quantization boundary flip, not a code bug** | **root cause found, no code defect to fix** | `NNTR_L2_DIFF`'s three-round diagnostic on this model's own registered weights and a real captured activation, five suspect calls out of ~32 in one forward (SNR 67.6-79.8 dB, well above the 30/40 dB unit-test gates yet still the ones this compounding traces to): (1) row-span baseline killed the outlier-row-quantization theory outright — clean (142+ dB) calls' `call_max_span` (0.17-2.91) fully overlaps and mostly exceeds the flagged calls' (0.30-0.54); (2) `hvx_quant_rows_u8_params`/`hvx_quant_pack_u8_ah` confirmed the SAME function on both paths by reading the DSP source, ruling out two quantizer implementations disagreeing; (3) per-element bin-flip test held to the device's own scale/zp came back 0/1792 on the two WORST calls (M=38 68.44 dB, M=41 67.64 dB), ruling out "boundary sensitivity under a shared scale" as the whole story; (4) reimplementing the quantizer's exact formula on `mid` alone (host's exact SwiGLU) to get an independently-derived scale showed `scale_diff=0.0000%` against the device's own scale on every flagged row — the per-row scale-mismatch theory is also dead. What actually explains it, confirmed on the same data: **`total_flips=1/1792` on ALL FIVE flagged calls**, with SNR varying 67.6-79.8 dB (≈16x in squared error) purely by WHICH of the 1792 activation columns flipped — the down matmul sums over K=1792, so one flipped element's impact scales with that element's own down-weight column magnitude, not with flip count. Mechanism, complete: `hvx_exp_sf`/`hvx_recip_qf32` are within their documented ~1e-6 relative-error spec (unrelated to the exp_top NaN fix, which stays fixed and stays correct); that tiny, spec-compliant difference from the host's exact `expf` occasionally (5/32 ≈ 16% of calls this run) lands an element on the wrong side of its row's u8 quantization boundary — one level, one element, every time it happens. This is not a coding defect reachable by inspection or a kernel rewrite: ANY two independently-computed float pipelines feeding the same row-wise affine u8 quantizer will disagree by exactly one level on some element, for any exp/recip approximation not bit-identical to the host's `expf`. Fixing it for real means widening the SwiGLU-to-down intermediate's quantization past u8 (e.g. int16, doubling that hop's wire/VTCM cost) so a spec-compliant approximation error crosses a bin boundary far less often — a real kernel-design tradeoff, not a bug patch, and not attempted this session pending a steer on whether the ~25% split-call speedup (§ this doc, `mm<=` measured) is worth that cost. `kFusedSwigluEnabled` stays `false`; the NaN clamp fix and `SwigluSurvivesExtremeNegativeGate` stand on their own merits regardless of this finding. |
| 2026-09-09 | **L2 shadow test on device: it IS the values, not a side effect** | **root cause fully closed** | `NNTR_L2_SHADOW=1` runs the fused path in full (every buffer touched, every lock taken, exactly as a normal run) but hands the model the reference's floats instead of the fused kernel's own. Result: a correct, coherent 3-sentence summary of the real prompt -- same shape of output the CPU-reference path (`kFusedSwigluEnabled=false`, which itself is **also all-HTP**: `dotQs4cx` checks `supports_gemm_qs4cx_accel_fp32()` before anything else, so gate_up and down both go through `mm_u8i4_layer` regardless of the flag) produces. This settles the fork the previous row left open: the fused kernel's own SwiGLU values are the fault, not memory aliasing or workspace corruption in the split-call plumbing. Combined with the prior row's mechanism (an ~16%-of-calls, one-level u8 quantization boundary flip from `hvx_exp_sf`/`hvx_recip_qf32`'s spec-compliant ~1e-6 divergence from host `expf`), the chain is now closed end to end on real device data: device SwiGLU value differs from host by a spec-compliant amount -> occasionally crosses a u8 bin boundary the reference does not -> that single flipped K-index changes the down matmul's whole output row by an amount set by that K-index's own down-weight magnitude -> compounds across 22 layers under greedy decoding into a different token stream. **This is not fixable by finding a coding defect** -- HMX's activation port is hardwired 8-bit (per the IDL: "HMX's activation port is always 8-bit"), so there is no wider-than-u8 intermediate to fall back to on this hardware, and no on-device polynomial exp/reciprocal will ever be bit-identical to host `expf`. The only real lever is tightening `hvx_exp_sf`/`hvx_recip_qf32` enough to push the boundary-flip rate from ~16% toward negligible; whether that is worth pursuing against the measured ~25% split-call speedup (§ this doc) is a product call, not an engineering one -- not attempted this session. `kFusedSwigluEnabled` stays `false`. |
| 2026-09-10 | **D1 third confirmation, and a transport anomaly that throttling does not explain** | **D1 holds; the ARM-side number is not usable until a cold re-run** | Full model run, `NNTR_HTP_PROFILE=2`, fused live on the device binary (`gate_up swiglu 56.0`, `down quant 0.0`) while the tree has `kFusedSwigluEnabled = false` — so the wrong text (the same 206-token "Could you please provide the text you would like summarized?") is the already-closed L2 failure, not a new regression, and A1 is wired to nothing yet. **D1**: gate_up `drain` 111.9 → **64.0** µs/call, third run in a row in the 64–68 band; DSP total 612.2 → **560.4** (−8.5%) with every other DSP stage inside noise (quant 98.5→108.3, swiglu 56.6→56.0, dequant 100.5→93.4, acc 59.3→54.8, mm≤ 185.3→184.0). **The anomaly**: gate_up `transport` 272.1 → **2686.1** µs/call (9.9×) while `down`'s went 189.2 → 421.9 (2.2×) — and both calls move the same 564 KB per round trip (450 KB f32 + 114 KB u8, only the direction differs). Thermal throttling cannot produce that split: it is uniform on the ARM clock, and `decode` was **13.7 TPS here vs 13.6 in the previous run whose gate_up transport was 543** — same temperature indicator, 5× the transport. Leading hypothesis, **not acted on**: the large f32 buffer is *inbound* on gate_up (ARM just wrote it ⇒ dirty-line writeback before the FastRPC map) and *outbound* on down (invalidate only), and writeback is both costlier and far more bandwidth-sensitive. Per §11.5's counting rule this is exactly the shape of argument that was wrong three times in Phase A, so the next step is one cold re-run (transport back near 500 ⇒ temperature; still ~2700 ⇒ something real), not a code change. Registration 2304.4 ms / 64 weights (convert 739.0, register 1246.9) — P4 unchanged and still the precondition for running all 22 layers. |
| 2026-09-10 | **A1 단계 A: det SwiGLU를 ARM 경로에 넣고 fused는 끈 채 실행** — 그리고 처음으로 식은 기기에서 잰 값 | **통과. 그리고 D1의 값은 그동안 과소평가되어 있었다** | 텍스트가 정상 3문장 요약(473 tok). `swiglu 0.0`이 두 줄 다, `down`에 `quant 69.1` — 두 번 dot 경로에 ARM det SwiGLU가 돌았다는 확인. **모델은 det SwiGLU를 견딘다** — A1의 마지막 미지수였고, `exp_ps`와 ≤1 ULP 다른 것이 22 레이어 greedy decoding에서 토큰을 바꾸지 않는다. **기기가 식었다**: decode **37.3 TPS**, prefill **137.5 TPS** (직전 세 실행은 13.6/13.7/17.6). 그래서 이 실행의 transport와 벽시계는 §13.3 규칙 4의 온도 게이트를 처음으로 통과한다. **D1 정정**: drain gate_up **111.9 → 32.4** µs/call, down **48.4 → 3.8** — 레이어당 5.13 → **1.16 ms**, 즉 **−3.97 ms**이지 44 §13.1이 적은 −1.6이 아니다. 그동안 잰 58.9/64.0/67.9는 전부 스로틀에 오염된 값이었다. 두 번 dot 경로 전체가 **55.6 → 43.1 ms/layer (−22%)**. 등록도 2304–2738 → **1521.8 ms** (convert 12.62 → 5.51 ms/weight, alloc+other 602 → 20.8) — 전부 온도. staging은 두 번 dot이라 64.2 MB(fused 27.8의 2.3배), 26.4 GB/s. **44 §13.1의 표는 서로 다른 온도의 숫자를 섞고 있으므로 fused 기준값(42.9, drain 111.9)을 같은 온도에서 다시 재야 한다 — 단계 B가 그것을 준다.** |
