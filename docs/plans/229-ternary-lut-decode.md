# 229 — Gemma 4 ternary experts: what a LUT dequantization gains at decode, against ternary → 4-bit + the existing HVX kernels

Issue: dlwlzzero/nntrainer#229 (p1, review first, no kernel). Base
`htp_decode` @ `8d164e534`. Contract `docs/plans/0001`; plan 201 §3.2 /
§3.4 own the Gemma byte and pool arithmetic this plan re-runs; LEDGER
rules 43 / 44 / 59 the rates. Reference survey:
`docs/htp_moe/t-man-lut-reference.md` (T-MAN fork @ `817bd29`, read
read-only in a scratch clone for this plan). **Added 2026-10-06 (user):
upstream PR nntrainer/nntrainer#4410 "[htp] lfm2 moe ffn 2bit gemv"
(Jungwon-Lee, head `0d603f29a`, fetched as `refs/pr/4410`, open, not
merged, merge-base `9deea6e2e` = 491 commits behind `htp_decode`)** — it
already holds options B and C as code, measured on an S25; its doc 54
(`docs/htp_attention/54_w2a8_experts_task.md` on that branch) is the
measured history quoted below.

Evidence tags: **[M]** measured on silicon (doc / rule cited; `[M-LFM]`
= on LFM2.5-8B-A1B, `[M-4410]` = doc 54's S25 unit `R3CY205ZMND`),
**[C]** code read at `path:line`, **[G]** arithmetic from the Gemma config
(plan 201 §2.4) or from [M] inputs, **[E]** estimate / guess, marked where
it enters.

**Headline.** The T-MAN kernel does not need porting. PR #4410's
`QS2CX_WH` (2-bit codes in whPack nibble order + a 4-entry int4 palette
per tensor + per-column scale / colsum) holds ternary {−1, 0, +1} × a
per-column scale **exactly** (one palette entry unused), and its native
GEMV `hvx_gemm_u8i2_wh_col*` feeds `vrmpyacc` from a register LUT with no
int4 materialised in VTCM and the **same int32 accumulator** as the 4-bit
path — bit-identical by construction, +25.9 % decode on LFM where only the
`mm` (DMA) column moved [M-4410 §5.18]. For Gemma 26B-A4B the experts at
2 bits take the byte floor from 34 to 45 tok/s; the 50 bar needs the
attention and dense FCs at 2 bits as well (floor 61), which is only
reachable once those FCs run on WH tiles (#225). What stays open is the
checkpoint's scale layout: a per-group / per-block ternary scale cannot be
held exactly by any per-column format here (needs-user).

## 1. Goal and gate

Acceptance (issue): a plan with numbers that decides whether S6 adds B or
C — per-token bytes and time for A / B / C per op kind on the 26B-A4B
shape against the 50 tok/s bar, the VTCM budget, the pool slot count, the
accuracy gate per option, and the lightest measurement that settles the
one number the host cannot give. Measurable form:

| gate | where | pass |
|---|---|---|
| bytes / time table | §3.2 of this plan | every number tagged [M] / [G] / [E]; the floor per option in tok/s against 50 |
| the format is exact for ternary | host: `whPack2 → expand == whPack` (`expand_i2i4_host_check`), the Gemma writer's palette {−1, 0, +1} | bytes identical to a `QS4CX_WH` write of the same ternary tensor; `moe_layer_host_check` 2-bit cell `bitwise mismatches=0`; `run_inproc_e2e.sh` Gemma lines `bit_identical=1` against the 4-bit run |
| speed (S4, device) | BENCHMARK Gemma block, `decode tok/s, NPU, gen 64 / 512 / 1024`, prompt 512, cool start, same sitting as A | B ≥ A + the §3.2 prediction's lower bound (experts only: ≈ +5 tok/s); the `mm` column is the only stage that moves (the §5.18 signature) |
| standing | prefill ≥ −5 % of the sitting's A (prefill is M > 1 and never takes the GEMV: unchanged by construction, read anyway); text identical to the CPU run for A / B (bit-identical options); `QS4CX_WH` / `QS2CX_WH` have no CPU fallback (`FloatTensor::dot` throws, doc 54 §5.4) | — |

For option C-TMAN (the T-MAN LUT GEMV proper, not #4410's register-LUT)
the text gate is D2's: decode PPL ≤ 1.02 × A pooled, no new loop, user
approval — it is not bit-identical (§3.5).

## 2. Where it lives

Nothing changes in this issue. What S6 (or the issue this plan spawns)
would touch, with the contract consumers:

| piece | `htp_decode` today [C] | #4410 brings [C, `git show refs/pr/4410:…`] |
|---|---|---|
| expert file format | `QS4CX_WH`: `whPack` nibble permutation, `nntrainer/tensor/htp_wh_layout.h:60-90`; writer `Applications/CausalLM/quantize_stream.cpp:376-397` (dtype tag), Gemma gate \| up writer from #231 | `QS2CX_WH` `[K·N/4 codes][4 palette][N f32 scale][N f32 colsum]`, `whPack2` in `qs4cx_tensor.h`, palette DP `htp_wh_palette.h`, `--moe_dtype QS2CX_WH` (commits `1a6928769`, `fe643e45c`) — **the quantizer's format tag and the loader check move** |
| M=1 GEMV | `hvx/hvx_gemm_u8i4_wh.c:65-84` `gemm_row1`: per 128 B of nibbles 1 load, `vasl`, 2 `vand`, 2 `vrmpyacc`; accumulator order documented in `hvx_gemm_u8i4_wh.h:52-60` | `hvx_gemm_u8i2_wh_col{,_nopf}`, `_cols2_nopf` (two columns share the activation splats), `_prefetch` for 256-byte tiles; per 128 B of codes 1 load, `vlsr`, 2 `vand`, 2 `vlut32`, 2 `vasl`, 4 `vand`, 4 `vrmpyacc` — the two `vlut32` results are exactly the two vectors the i4 loop loaded, so the int32 sum is the same (commit `dfca0acbf`) |
| expand (option B proper) | — | `hvx/hvx_expand_i2i4.{c,h}`: 2 `vlut32` + (no `vshuff` after `c2cdfab0a`) per 256 B out, table in a register; IDL entry `expand_i2i4` + device gtest `HvxExpandI2I4.MatchesScalarBitExact` (`08f44155d`, `5aa3b4357`) — **IDL `test/htp/nntr_hvx.idl` + `generate_stub.sh` + skel move** |
| MoE kernel routing | `hmx/hexkl_mm_u8i4_moe.c:538-623` M=1 feed: two gate_up slabs, 4 DMA queues (#177), schedule table | `moe_all_2bit` / `native_i2`: half-size slabs (`feed_gu_bytes = gu_bytes / 2`), `moe_m1_push(…, all_2bit)`, per-expert `g_lut` / `d_lut`, `HEXKL_PROBE_EXPAND` stage — **`NNTR_HTP_PROFILE` stage tables and `tools/htp_fc_report.py` gain the `expand` column** |
| weight registry | `hexkl_weight_u8i4_table` slots; `HtpComputeOps` `takeExpertSlot` (`htp_compute_ops.cpp:4920-4962`), pool miss `readWeight` `:5172`, swap offsets `:4964-4970` | `slots[h].bits` (2 or 4), `wh_bytes` halves — **`HtpComputeOps` slot size, arena budget and the miss read move with `bits`** |
| host checks | `test/htp/host/{moe_layer_host_check.c,gemv_native_check.c}`, `run_host_checks.sh` | `expand_i2i4_host_check.cc`, the 2-bit M1 GEMV cell (`dma_kb=10752 vs 21504 (want half)`), `standin/hvx_scalar.c` `vlut32` twin |
| FC / lm_head kinds | `test/htp/nntr_hvx_fc_q4.c` on `Q4M1` (Q4_0 blocks, `q4_gemv_cpu_det.h:330-335`), VTCM-fed; #225 (on `htp_first_version`, not here) moves the FC set to `QS4CX_WH` tiles and `hexkl_mm_u8i4_fc_m1_run` | nothing: #4410 has no 2-bit FC, attention or lm_head path |

What #4410 lacks that `htp_decode` has (merge-base `9deea6e2e`): the one-PD
token driver (#211), the FSU pool inside the token (#201 S1/S2), the
4-queue feed (#177 / #185), the Gemma S4 kernels (GeGLU epilogue
`HEXKL_MOE_FLAG_GEGLU`, `hexkl_mm_u8i4_moe.h:377`), the hd64 fixtures.
Its "resident MoE decode stream" (`0d603f29a`) parallels our token driver
and is not taken.

## 3. Design

### 3.1 The three options, restated on what exists

| | where the ternary → int4 step runs | DDR bytes, experts | kernel | bits |
|---|---|---|---|---|
| **A** | converter / load: 4-bit `QS4CX_WH` in the file and the arena | 4 bits/w | today's `gemm_row1` | bit-identical to the CPU run (3 levels fit int4 exactly) |
| **B** | DSP, per expert into VTCM: `QS2CX_WH` in DDR, `hvx_expand_i2i4` after the slab lands, then `gemm_row1` on the int4 slab | 2 bits/w | #4410 `be916a261` | bit-identical; ≈ 86 µs/call of expansion still exposed on LFM with two slabs [M-4410 §5.18.5] |
| **C-reg** | none: `QS2CX_WH` read packed, the 128-byte code-pair table in one HVX register, `vlut32` → `vrmpyacc` | 2 bits/w | #4410 `dfca0acbf` (`hvx_gemm_u8i2_wh_*`), host-exact, **not yet measured on a device** | bit-identical (same int32 sums) |
| **C-TMAN** | none: T-MAN `hvx_lut_ctor` (activation → int16 LUT of ±x sums per g = 4) + `hvx_tbl` BitNet path (`vlut16` per bit-plane, int16 → int32 → qf32) + `hvx_bit_serial` | 2 bits/w (2 bit-planes) | port of `hvx_funcs.h` ≈ l.538–700, weights laid out by a port of `hvx_preprocess_weights` | **not** bit-identical: activation quantized to int16 per tensor, qf32 epilogue |

**Chosen: C-reg (#4410's native u8i2 GEMV) ported onto the one-PD pool
path, with B's expand kernel kept only as the host spec and the device
gate for `vlut32`.** Reasons, each from a measurement: (1) the M=1 GEMV is
DDR-bound, not issue-bound — 344 k `vrmpy` ≈ 25 µs of pool time per LFM
call against 676 µs of `mm` [M-4410 §5.18.1], and the 4-bit feed's
engine is busy 640–682 of 712 µs [M-LFM, BENCHMARK cycle 15] — so a LUT
that saves instructions buys nothing, and the only thing 2 bits can buy
is the halved DMA column; C-reg takes all of it, B leaves ≈ 86 µs/call
exposed unless the slab count doubles (§3.4). (2) C-reg's int32 sums are
the 4-bit path's, so every gate stays bit-identity (rule 45's deciding
failure is repetition after a non-identical kernel; D2 allows a loss, the
plan does not spend it here). (3) The code exists with host proofs
(`gemv_native_check`, `moe_layer_host_check`); the port cost is the merge
into the pool path, not a kernel.

**Rejected: C-TMAN.** Its gain over `vrmpy` is instruction count (per
1024 weights ≈ 14 vector ops with four `vlut16` on the permute unit vs
C-reg's ≈ 32), which matters only where compute is the wall — plan 194
L10's reading, now re-confirmed at 2 bits by the 25 µs / 676 µs ratio
above [G from M-4410]. It costs a second activation format (int16 LUT,
per-tensor scale: `static_assert(ActGroupSize == -1)` in the BitNet
`hvx_tbl`), a qf32 epilogue, a second weight layout (bit-planes,
`vlut16` even / odd interleave), has no unit tests upstream, and breaks
bit-identity. It is kept as the fallback only if S3's ISS reading shows
C-reg compute-bound at the Gemma shapes (§4), which the numbers say it is
not.

### 3.2 Bytes a token and the floor, 26B-A4B, per option and per kind

Shape [G, plan 201 §2.4]: hidden 2816, 30 layers, 128 experts top-8,
`moe_intermediate` 704, dense `intermediate` 2112, 25 sliding layers (q/k/v/o
34.6 M weights) + 5 global (49.0 M, K = V), vocab 262 144 tied. Expert =
gate \| up [2816 × 1408] + down [704 × 2816] = 5.95 M weights.

Weights read per token [G]: experts 8 × 30 × 5.95 M = **1.427 B**;
attention **1.11 B**; dense FFN 30 × 17.8 M = **535 M**; lm_head **738 M**.

Bytes per token by storage (QS4CX_WH 0.5 B/w + ≈ 34 KB of scale / colsum
per expert; Q4_0 0.5625 B/w; QS2CX_WH 0.25 B/w):

| kind | A: Q4_0 FCs (plan 201) | A′: #225, FCs on WH 4-bit | 2-bit |
|---|---:|---:|---:|
| experts | 714 MB | 714 | **357** |
| attention | 624 | 555 | 278 |
| dense FFN | 301 | 267 | 134 |
| lm_head | 415 | 369 | 185 |

Rates: DRAM ceiling **70 GB/s** for any reader mix (rule 44 [M-LFM]); the
weight DMA alone **57** in the app (rule 43 [M-LFM]); the one-PD path's
effective rate on both kinds **≈ 44 GB/s** (rule 59 [M-LFM]: FC set
402 MB in 9.2 ms, MoE round 462 MB in 10.5 ms); the remainder outside the
byte streams on LFM one-PD **2.7 ms** (22.4 − 9.2 − 10.5), scaled to
Gemma's 30 layers and head_dim 256 / 512 attention as **≈ 4 ms [E]**.

| option (which kinds at 2 bits) | MB / token | floor at 70 GB/s | at 44 GB/s + 4 ms [E] |
|---|---:|---:|---:|
| A (plan 201's row) | 2 054 | 29.3 ms → **34 tok/s** | 50.7 ms → 20 |
| A′ (#225: FC set on WH tiles, all 4-bit) | 1 905 | 27.2 → 37 | 47.3 → 21 |
| **B / C-reg, experts only** (what #4410 covers) | 1 548 | 22.1 → **45** | 39.2 → 26 |
| + attention + dense FFN at 2 bits (needs #225's WH FCs + the u8i2 kernel on the FC path) | 1 138 | 16.3 → **61** | 29.9 → 33 |
| + lm_head at 2 bits (only if the checkpoint's tied head is ternary) | 954 | 13.6 → 73 | 25.7 → 39 |

Per kind, does the floor move: **experts yes** (−357 MB, −8.1 ms at
44 GB/s: the 30 MoE rounds go ≈ 16.2 → 8.1 ms, scaling §5.18's 679 → 424 µs
per 4-expert call to 8 experts × 30 layers [G from M-4410]); **attention
and dense FFN yes but only after #225**, since the Q4M1 kernel reads Q4_0
blocks and has no 2-bit form, and the ternary checkpoint's attention /
dense weights are presumably ternary too (the user's format answer says);
**lm_head** moves the floor by 184 MB / 4.2 ms only if its weights are
ternary — with a tied embedding that is the least likely of the three
(needs-user). The scale of the bar: the experts alone leave the floor at
45, under 50; the FC set at 2 bits is what puts the floor above the bar.
The 44 GB/s column says the measured path sits at ≈ 60 % of the DRAM
ceiling; the gap (DMA 57 vs 70, the FC feed, misses) is #201 S6's
business, not this plan's.

**Which stored packing changes which number.** DDR under B / C is 2-bit
whatever the file holds, because the converter (or the loader's slot
fill) emits `QS2CX_WH`:

| checkpoint packing | file / page cache | DDR arena & per-token bytes | converter work |
|---|---|---|---|
| 2-bit (4 per byte) | 0.25 B/w | as the table | re-order into whPack2 code order, one palette {−1, 0, +1, 0} |
| 1.58-bit (5 per byte, base-3) | 0.2 B/w: experts file 2.9 → 2.3 GB | **unchanged** (2-bit in the arena; a 5-per-byte DMA format would need a base-3 unpack kernel ≈ 5 `vlut32`-sized passes per vector, ≈ 6 × B's expand [E] — not worth 20 % of the expert bytes, ≈ 1.6 ms/token at 44 GB/s) | base-3 digit extraction on the ARM at convert time; the miss read of 1.46 MiB is a `pread` of the 2-bit file region |
| int8 (one byte per ternary value) | 1 B/w: 11.4 GB of experts, page cache cannot hold it beside the arena (plan 201 §3.3) | unchanged | pack 4 per byte |
| scale per column or per tensor | — | exact in `QS2CX_WH` (`w_scale[n]`, per-tensor = broadcast) | — |
| **scale per group / block along K** | — | **not exact**: the HMX / GEMV epilogue applies one scale per column over all of K (`htp_wh_palette.h` header comment); options: (i) fold the group scale into a per-column scale with error (then SNR / PPL gates, not bit-identity), (ii) a per-group drain of the accumulator (`acc_read` cost, rejected in doc 54), (iii) C-TMAN, whose LUT bakes per-block fp16 scales (`GroupSize` 64 / 128 paths) — **needs-user** |

### 3.3 VTCM budget (C-reg and C-TMAN against the 8 MiB S1 holds)

S1 holds all 8 MiB (`hexkl_micro_hw_init`); the M=1 feed uses the whole
arena below the HMX config block as two gate_up-sized slabs with the downs
packed into them (`hexkl_mm_u8i4_moe.c:569-590`). The FC set's VTCM feed
(6 lanes × 2 groups) uses the same arena in turn (one PD, rule 59).

| | 4-bit Gemma | 2-bit Gemma |
|---|---:|---:|
| gate_up slab | 1.98 MB | 0.99 MB |
| down matrix | 0.99 MB | 0.50 MB |
| 2 slabs (today's feed condition) | 3.96 MB | 1.98 MB |
| 4 slabs (doc 54 §5.18.5's lever: DMA two experts ahead) | 7.9 MB — does not fit beside the config block | **3.96 MB, fits** |
| C-reg register LUT | — | 128 B per weight, a vector register (no VTCM) |
| C-TMAN activation LUT (`K / 4 × 16 × int16`) | — | 22.5 KB (K = 2816), 5.6 KB (K = 704); the `c` accumulator `N × bits × f32` 11 KB / 22.5 KB; T-MAN's `TILE_K = 256` weight tile per thread, trivial |

So the LUT tables are not the budget; the slabs are, and at 2 bits they
halve. C-reg at 2 slabs keeps the ≈ 86 µs/call exposure doc 54 measured
only if the expand stage existed — with the native kernel there is no
expand, and the feed condition `2 × feed_gu_bytes ≤ arena` holds with
room for four slabs, which is the one scheduling change worth carrying
(§4 S2).

### 3.4 Pool arithmetic (plan 201 §3.4 re-run)

| | 4-bit | 2-bit |
|---|---:|---:|
| expert slot (tiles + scale / colsum + palette, page-rounded) | 2.87 MiB | **1.46 MiB** |
| all experts (3 840 slots) | 10.6 GiB | 5.5 GiB — still not resident |
| FC set + lm_head on WH (A′) | 1 136 MiB | 745 MiB if the FCs are 2-bit |
| one PD room = 3 840 − FC set − heap 150–200 [E] | ≈ 2 500–2 550 MiB → 870–890 slots → **C ≈ 29 of 128 (23 %)** | FCs 4-bit: ≈ 1 710–1 750 slots → **C ≈ 57 (45 %)**; FCs 2-bit: ≈ 2 900–2 950 MiB → ≈ 1 990–2 020 slots → **C ≈ 66 (52 %)** |
| miss read per expert (rule 61: 0.5–1.6 ms at 5.25 MiB warm) | ≈ 0.3–0.9 ms [E, proportional] | ≈ 0.15–0.45 ms [E] |

Hit rate at these C is unknown for Gemma (plan 201 §5); the pool simulator
on S5's routing trace answers it. What is certain: the handle limit
(4 096 since S1) covers 2 020 slots, and the mapping stays under 14
windows.

### 3.5 Accuracy per option

| option | arithmetic | gate |
|---|---|---|
| A | int4 codes = the ternary values; per-column scale = the checkpoint's | bit-identical to the CPU run of the same ternary weights (the CPU `QS4CX` / Q4_0 path must read the same codes; S5's converter owns that) |
| B, C-reg | palette {−1, 0, +1, 0} → the same int4 bytes `gemm_row1` would read; `vlut32` result identical (host twin + device gtest `HvxExpandI2I4.MatchesScalarBitExact`, passed on an S25 [M-4410 §5.8]); int32 sums identical | `gemv_native_check` 0 mismatches; `moe_layer_host_check` 2-bit cell; in-process E2E `bit_identical=1` vs the 4-bit run; device MoE dumps `bit_identical=1`, text ≡ A 8 / 8 |
| C-TMAN | x → int16 LUT entries with one per-tensor scale (`max |Σ4 x| / 32767`), int16 → int32 → qf32, `(c · ls + lb) · s`, bit-plane sum in qf32 | not bit-identical: an SNR line vs f32 ≥ the 4-bit path's own (the u8 activation path reads ≈ 16–17 dB on real expert tensors, doc 54 §5.3 [M-4410 host]); then D2: decode PPL ≤ 1.02 × A, no new loop, user approval |
| per-group-scale checkpoint folded to per-column (if the user's answer is "per group") | a weight error, uniform across K | SNR vs the f32 checkpoint; PPL / loop / approval — the bit-identity gate is unavailable by construction |

Doc 54's own accuracy row (SNR 7.8 dB, "ppl not yet measured", §5.18.6)
is **not** this plan's problem: that was a 4-level palette fitted to a
4-bit model's codes; ternary weights already have three levels, so the
palette is lossless.

### 3.6 Contract §2 and doc 45 §3

Walls 1 and 2 (M=1 GEMV, DMA feed) are reused unchanged in shape; wall 3
(transport) untouched. Arena budget: the slot halves, the FC set is as
#225 leaves it, the PD heap grows by nothing (the register LUT is 128 B
per expert in the slot struct). No CPU fallback for `QS2CX_WH` (the throw
stays). DMA hidden behind compute: at 2 bits the compute : DMA ratio is
≈ 25 : 340 µs per LFM call — hidden with margin. `_det` before every
quantizer: the GEMV output feeds the same epilogue; nothing before a
quantizer changes. Bit-identical + text gates: §3.5.

## 4. Steps

Each ends in a rung of `.claude/skills/hexagon-gates`. S0 is the only
blocker; S1–S3 are host work and can start on LFM now.

* **S0. Pin the format (needs-user).** Answer needed: bits stored (2 /
  1.58 / int8), scale scope (per tensor / per column / per group of K),
  whether attention, dense FFN and the tied head are ternary too. Table
  §3.2 says what each answer changes; a per-group scale re-opens §3.5's
  last row and C-TMAN. Gate: the issue comment with the answer; S5's
  converter spec written from it.
* **S1. Port #4410's 2-bit stack onto `htp_decode`** (own PR, LFM, host
  only). Cherry-pick in order `1a6928769` (palette), `fe643e45c`
  (`QS2CX_WH`), `08f44155d` + `5aa3b4357` + `c2cdfab0a` (expand kernel,
  the `vlut32` index fix, the shuffle-free code order), `be916a261` (feed
  at 2 bits), `dfca0acbf` (native u8i2 GEMV); skip `f3874b0d8` (half-DMA
  measurement build), `0d603f29a` (resident stream) and the doc commits.
  Expected conflicts: `hexkl_mm_u8i4_moe.c` M=1 section (4 queues, #185
  schedule table, the pool's handle table from #201 S1), `takeExpertSlot`
  / `readWeight` (slot bytes by `bits`), the IDL (`expand_i2i4` entry —
  keep it, it is the device gate for `vlut32`). Gate: rung 1 (`ALL CHECKS
  PASS` incl. `expand_i2i4_host_check` and the 2-bit M1 GEMV cell,
  `*Lfm2Moe*` 6 / 6 plus a `QS2CX_WH` tiny-fixture line, `INPROC E2E PASS`
  with a 2-bit lfm25 fixture `bit_identical=1` against the 4-bit run through
  the pool at C = 1 / 2 / unset); rung 2 (IDL changed: skel md5, stub
  regenerated, `UNDEFINED SYMBOLS OK`, both arches).
* **S2. Gemma at 2 bits on the host.** The #231 gate \| up `QS4CX_WH`
  writer gains `bits = 2` (ternary codes → palette {−1, 0, +1, 0},
  colsum over the codes, `w_scale` = the checkpoint's column scale);
  `hvx_gemm_u8i2_wh_cols2_nopf` under the GeGLU epilogue
  (`HEXKL_MOE_FLAG_GEGLU`); the slot size and miss read by `bits`; the
  feed at four slabs when `4 × feed_gu_bytes` fits (DMA two experts
  ahead; the host scoreboard `moe_layer_host_check` proves the new
  schedule). Gate: rung 1 (`gemma64 e3 … bit_identical=1` 2-bit vs 4-bit,
  `calls/token=1.00`, tokens 8 / 8), rung 2.
* **S3. The ISS reading (cheap, host, `hexagon-sim -mv79 --timing` as
  plans 132 / 170 used it; `source tools/htp/env.sh` puts it on PATH).**
  A standalone harness in the scratchpad (not committed unless it finds
  something) calling, on VTCM-resident operands at the two Gemma shapes
  (K = 2816, N = 1408 gate \| up; K = 704, N = 2816 down), one thread:
  `gemm_row1` (4-bit), `gemm_i2_row1` / `_cols2` (C-reg), and — only to
  close the issue's question with a number — T-MAN's BitNet `hvx_tbl`
  (`hvx_funcs.h` l.538–700 compiled as-is; weights from a port of
  `hvx_preprocess_weights`, activation LUT from `hvx_lut_ctor`). Report
  pcycles per expert per lane; gate: **C-reg ≤ 0.5 × the 2-bit DMA window
  per expert** (1.49 MB / 57 GB/s = 26 µs ≈ 47 k pcycles at 1.8 GHz, i.e.
  ≤ 23 k pcycles / expert / lane at 6 lanes [G]) — pass means compute
  stays hidden and C-TMAN is closed for good; fail means C-TMAN's
  instruction count matters and gets its own issue. The ISS prices HVX
  compute right and memory-side scalar work wrong (rule 58); this harness
  has no scalar memory side, so the reading stands.
* **S4. Device sitting — unavoidable, after S5's files** (Gemma on the
  attached S26, or, as a bridge before the files, LFM on the farm S25 to
  re-read §5.18 on the one-PD pool path). Handoff
  `docs/measurements/229-2bit.md`, prompt 512, G 64 / 512 / 1024, cool
  start per G, S1 ceiling after every run; variants ≤ 4:

  | variant | what | reads |
  |---|---|---|
  | **A** | 4-bit model, one PD, pool at its largest C (unchanged reference) | tok/s, per-kind lines, text |
  | **B2** | 2-bit model, C-reg, same C as A | tok/s; `mm` is the only stage that moves (the §5.18 signature); misses a token unchanged |
  | **B2-C** | 2-bit, pool at the largest C that loads (§3.4's 57 / 66) | misses a token, ms a miss (halved?), tok/s |
  | **B2-S4** | B2-C with four slabs | `mm` vs B2-C; the two-experts-ahead DMA |

  Gate: §1's speed row; dumps `bit_identical=1`, text ≡ A 8 / 8; RSS and
  arena recorded (the 2-bit arena is the other half of the gain: −45.8 %
  peak RSS on LFM [M-4410]).

## 5. Risks

* **The scale scope** (S0). Per-group scales make every bit-identity gate
  in this plan unavailable; the plan then becomes a PPL plan (§3.5) and
  C-TMAN returns as a candidate. The handoff table shows it as the SNR
  line next to the 4-bit path's.
* **#4410 is open upstream and 491 commits behind**; its `vlut32` reading
  was gated on one S25 unit (v79). The v81 (S26) behaviour of
  `V6_vlutvvb` is unmeasured — the device gtest `HvxExpandI2I4` runs first
  on any new unit, before a tok/s cell.
* **The native u8i2 GEMV has no device number**: §5.18's +25.9 % is the
  expand path (B); C-reg's claim is host-exact only. The sitting's B2
  cell is the first reading; if `mm` does not fall to ≈ half, the stage
  lines (`expand`, `drain`, `mm`) say which assumption failed.
* **DMA rate / DVFS / thermal**: same-sitting A / B, cool start per G,
  rule 52's band; the gain is a byte count, so it should show at every G
  alike — a G-dependent gain is a thermal artefact.
* **Stale skel / stub**: the IDL changes (S1); md5s on both ends, one tree
  per sitting (rule 3).
* **Address space**: a larger pool is still under 14 windows; the ceiling
  cell after every run (rules 50, 54, 59c).
* **Hit rate** at C = 57–66 is a guess until the S5 trace; the sitting
  reads misses a token at both C.
* **Host vs device**: the host twin proves bits and the feed schedule,
  never ms; the ISS proves compute cycles, never the DMA.

## 6. Docs to update

* **`docs/htp_moe/BENCHMARK.md`**: the Gemma block gains §3.2's table
  (computed, per option, marked until S4 measures it) and, after S4, the
  A / B2 / B2-C / B2-S4 rows with RSS + arena; the LFM bridge sitting, if
  run, as one row beside #201's one-PD cells.
* **`docs/htp_moe/LEDGER.md`**: §Upstream gains PR #4410 (`0d603f29a`,
  open, what is taken and what is not); a rule candidate from S4 ("at
  M = 1 the GEMV is DDR-bound, so 2-bit weights buy the halved DMA
  column and nothing else — +25.9 % on LFM, doc 54 §5.18; the register
  LUT keeps the int32 sums"); plan 194 L10's "revisit only if a 2–3-bit
  format is adopted" closed by this plan; open item: the per-group-scale
  consequence until S0 answers.
* **Plan 201 §3.2 / §3.4**: the Gemma columns re-pointed at this plan's
  2-bit rows; S6's lever list gains "FC set at 2 bits after #225".
* **Contract `0001` §12**: the user's S0 answer as a dated row.

## 7. S7 — the 2 GB budget (a later stage; nothing in S1–S5 or in plan 234's port order depends on it)

**User, 2026-10-06:** peak memory (process RSS including the ION arenas)
under 2 GB, the rest streamed from flash — to be considered **last**, after
the ternary 26B decodes end to end on the device. Issue #234 asked for the
arithmetic to be on record now; this section is that record and nothing
more. §3.4's pool numbers (C ≈ 57–66) assumed a 3.8 GB arena and are
superseded here for the 2 GB case only. Tags as §0; **[M-4408]** = doc 55
§10.14 on `refs/pr/4408` (S25, 4-bit experts, hybrid path, one prompt).

### 7.1 Resident bytes per kind, 26B-A4B (MiB)

| kind | where | 4-bit | 2-bit | source |
|---|---|---:|---:|---|
| expert slot (page-rounded) | ION arena | 2.871 | **1.453** (gate \| up 1 003 520 B + down 520 192 B) | [G] doc 55 §3.2 formula, `expertStride` |
| FC set on WH (attention 1.110 B + dense 0.535 B weights) | ION arena (P5 sidecar) | ≈ 790 | ≈ 395 | [G] 0.5 / 0.25 B/w + 8 B per column |
| lm_head on WH (738 M, tied) | ION arena | 352 | 176 (only if the tied head is ternary) | [G] |
| router 30 × [2816][128] f32 | DSP heap | 41 | 41 | [G]; `hexkl_graph.c:589-605` pattern |
| KV cache, FP16, sliding 25 layers (K + V, 8 heads × 256) | DSP heap (attention on the NPU) | 200 at **window 1024** / 800 at max_seq 4096 | same | [G]; today the sliding cache is sized by `max_seq_len` (plan 201 status note: ≈ 800 MiB) — sizing it by the window is the lever |
| KV cache, global 5 layers, K = V (2 heads × 512) | DSP heap | 40 at 4096 | same | [G] |
| DSP heap base (graph, scratch, token mailbox) | DSP heap | ≈ 100 (LFM one PD `heap_used` 92 MiB, rule 63 [M-LFM]; S1 ≈ 70 + S2 ≈ 30 two-PD, plan 201 §2.2) + growth for 30 layers ≈ 150 [E] | same | |
| ARM RSS, model-independent (libs, runtime, graph, tokenizer, activations) | ARM anon | ≈ 240 [E] = LFM's 766 [M-LFM, 216-fadvise step 0] − 141 embedding Q4_0 − 384 FC Q4_0 originals | same | the 766 was never decomposed on the device; this split is arithmetic |
| ARM RSS, model-dependent: embedding table Q4_0 (262 144 × 2816) | ARM | 396 | 176 if ternary | [G]; **mmap-able**: one row a token → page cache, not RSS |
| ARM RSS, model-dependent: FC Q4_0 originals (CPU copies of NPU-resident kinds) | ARM | 888 | — | [G]; LFM keeps them (384 of the 766); with every kind resident they are dead weight |
| page cache (the expert file region) | OS, **not RSS** | 10.8 GiB | 5.5 GiB | [G]; `NNTR_MOE_FADVISE` (#216, merged, env-only) and #219's tier change *where* the complement lives — the tier (anon RSS) **counts**, the page cache does not |

### 7.2 The largest pool under 2 048 MiB

Fixed part, with the two ARM copies gone (embedding mmap'd, no FC
originals — both are required: with them the FC arena alone overflows):
ARM 240 + heap 150 + router 41 + KV 240 (window 1024 + global 4096) =
**671 MiB**.

| configuration | fixed + FC set + lm_head | pool room | slots (1.453) | **C of 128** | prefill read-ahead (30 C ≥ 128 + slack) |
|---|---:|---:|---:|---:|---|
| experts 2-bit, FCs 4-bit, lm_head 4-bit | 671 + 790 + 352 = 1 813 | 235 | 161 | **5** | at the R2 floor; no slack (doc 55 §10.11: C = 5 lost the read-ahead, prefill 40.9 vs 89.3 TPS) |
| experts + FCs 2-bit, lm_head 4-bit | 671 + 395 + 352 = 1 418 | 630 | 433 | **14** | yes |
| everything 2-bit | 671 + 395 + 176 = 1 242 | 806 | 554 | **18** | yes |
| (KV at max_seq 4096 instead of the window) | + 600 | − 413 slots | | C − 13 | the KV window sizing is worth 13 experts a layer |

### 7.3 Misses a token and the decode floor

Hit rate: the only Gemma points are [M-4408] — misses over 512 generated
tokens 81 348 / 62 198 / 41 955 at C = 8 / 16 / 24 → **159 / 121 / 82
misses a token** of 240 routed uses (hit 34 / 49 / 66 %), hybrid LRU, 4-bit
weights, one prompt; LFM's curve (57 / 73 / 85 % at C = 8 / 12 / 16 of 32,
doc 53 §5) has the same shape at the same C / E. Interpolated [E]: C = 5
≈ 170, C = 14 ≈ 130, C = 18 ≈ 110 misses a token.

Cost of one miss at 1.453 MiB [E, scaled from measured rates]: cold UFS
3.0 GB/s (doc 53 §5.5 [M-up]) → **0.5 ms**; the uncached-ION store cap
4.9 GB/s (`readWeight`'s comment, plan 219 §0.1) → 0.3 ms — this is the
warm floor on this path; doc 55 §10.14's warm 0.76 ms per 2.87 MiB (3.9
GB/s) → 0.38 ms; LFM's best warm 0.31–0.37 ms per 5.25 MiB (plan 201
§3.3) → 0.1 ms is a cached-staging rate the ION slot does not reach.

| configuration | DDR term a token (§3.2, 44 GB/s + 4 ms [E]; at 70 GB/s in brackets) | miss bytes a token | miss term cold / warm (0.5 / 0.3 ms) | **floor cold / warm** |
|---|---:|---:|---:|---:|
| C = 5, FCs 4-bit | 39.2 (22.1) ms | 170 × 1.453 = 247 MiB | 85 / 51 ms | 124 ms → **8 tok/s** / 90 → 11 |
| C = 14, FCs 2-bit | 29.9 (16.3) | 189 MiB | 65 / 39 | 95 → **11** / 69 → 14 |
| C = 18, all 2-bit | 25.7 (13.6) | 160 MiB | 55 / 33 | 81 → **12** / 59 → 17 |
| for scale: §3.4's C = 66 (3.8 GB arena), hit ≈ 85–90 % [E] | 25.7 | 35–52 MiB | 12–18 / 7–11 | 38–44 → 23–26 / 33–37 → 27–30 |

Under 2 GB the floor is **miss-bound, ≈ 8–17 tok/s**: the flash term is
1.3–2× the DDR term, and the serial miss read is the wall. The page cache
— not RSS, but the whole difference between the cold and the warm column —
needs ≈ 5.5 GiB for the 2-bit expert region; on a 12 GB phone with ≈ 3.7 GB
of Android and 2 GB of ours that is at the edge, so "warm" is a measurement,
not a plan (rule 61's `pgpgin` / refault columns on every cell).

### 7.4 What flash streaming buys, and which levers matter when the pool is small

It buys the run itself: 2-bit experts alone are 5.5 GiB resident, so under
2 GB there is no all-resident option. The cost is the factor 2–3 above
between the 2 GB floor and the 3.8 GB one. Levers, largest first [E]:

1. **No ARM copy of any NPU-resident weight; the embedding mmap'd** —
   1 284 MiB = 884 slots = C + 29; without it nothing fits (§7.2).
2. **FCs (and the head) at 2 bits** — C 5 → 14 → 18 (§7.2); the same
   code path as §3.2's floor 45 → 61.
3. **KV sized by the sliding window** — C + 13 at max_seq 4096.
4. **Keep the misses warm** — 0.5 → 0.3 ms a miss, −20 to −34 ms a
   token; it is page-cache policy, outside RSS: `NNTR_MOE_FADVISE`'s
   WILLNEED-on-evict (rule 62: 2–10 ms a call on the S25, prefill
   −7..−18 % — re-read on Gemma, not adopted).
5. **Read-ahead from the previous token's routing (P-D)** and **misses
   under the hits' compute (P-A)**: they hide, they do not shrink —
   P-A hides ≤ the hit experts' DMA of the layer (≈ 8 ms of 55–85), P-D
   the previous token's whole compute but at a 64 % routing-prediction
   ceiling on LFM (doc 53 §7 [M-up]); with 110–170 misses a token both
   are worth ≈ 10–20 ms, less than lever 4 and only after it.
6. **Four slabs** (§3.3): a DDR-term lever (≈ −1 to −2 ms); irrelevant to
   the misses.
7. **#219's ARM tier does not fit**: the complement is 3 840 − 554 =
   3 286 experts = 4.7 GiB, and any capped tier slot costs the same 1.453
   MiB as a pool slot while serving a hit at 0.3 ms instead of 0 — under
   an RSS cap a tier slot is strictly worse than a pool slot. Off under
   2 GB.

Open, for S7 proper: the real `RSS 766` decomposition on the device (one
`smaps` read of an LFM run settles the 240), the sliding-cache sizing in
the attention op, whether the loader can skip the CPU originals when every
kind is resident, and S5's own routing trace for the hit curve.
