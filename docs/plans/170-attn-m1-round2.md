# 170, round 2 — the fast bit-identical ATTN_M1 meets its speed gate

Issue: dlwlzzero/nntrainer#170 (p1, contract §12: decode NPU end-to-end,
bit-preserving). Read against `htp_moe` @ `90d88e2b` (PR #176 merged
2026-09-29). Round 1: plan `docs/plans/170-attn-m1-fast-exact.md`, handoff
`docs/measurements/170-attn-m1-hf.md` (S1 constants, S2 results). The
kernel is bit-identical on silicon (G3 / G4 / G5 pass, text approved) and
misses G6 by ≈ 2×: in-model `pcyc/op ATTN_M1` **394,501** at G = 64 and
**700,449** at G = 1024 against T64 = 210 k / T1024 = 350 k.

## 0. Where the 2× is (S2 phase words, `s2/logs/g3_attn.log`, pos 1023 cold, 6 lanes)

| word | lane-summed | wall share of the 716 k call | what it is |
|---|---|---|---|
| append | 95.7 k (caller) | 96 k | 512 scattered halfword stores into the last Kt tile (`append_head`, 64 lines per kv head, cold), 2048 scalar q roundings, 8 scalar v rows |
| scores (P1) | 702 k | 117 k | 128 units × 5.5 k; warm 4.2 k; the probe's L2-resident shape 2.4 k per unit. Cold − warm = the 8 KiB tile fetched with one miss stream per lane |
| softmax (P2 + serial + divides) | 526 k | ≈ 88 k | per P2 unit ≈ 3.5 k: exp16 ≈ 0.7 k, 256 scalar halfword stores into 64 cold ET lines ≈ 2.8 k; the caller's max + sum ≈ 25 k; the divides ≈ 50 k lane-summed |
| pv (P3) | 1,542 k | **385 k** (busiest lane = 2 units) | 8 units × 193 k = **189 pcyc per V row** (warm 122; the probe's pv4 27.6). One 128-byte row per position inside the dependent chain: one DDR miss per row, nothing in flight behind it |
| busy_max / mean lane | 590 k / 462 k | 1.28× | P3's 8 units on 6 lanes |

Sum: 96 + 117 + 88 + 385 + dispatch 30 = 716 k = `dsp_us` 340 × 2.106 GHz.
In the model the call runs at 1.13× (G = 1024) / 1.2× (G = 64) of this line.

Two silicon facts from S1 decide the design: a single lane streams **11.4
GB/s** with demand loads and **32.9 GB/s** with a 32 KiB `l2fetch` lead
(the 6-lane aggregate is 38 GB/s either way, which is why S1 read "no
gain" — that was the saturated case, not the latency-starved one P3 is);
and the hardware queues **three `l2fetch` per thread** and stalls on a
fourth (`hexkl_mm_u8i4_moe.h:250-252`, #113).

## 1. Goal and gate

**Goal (issue, unchanged).** ATTN_M1 per layer at or below the CPU's
per-layer cost (≈ 0.15–0.3 ms at L 1024–1536), bit-identical to the Android
fp16 CPU attention (`attn_m1_det.h`, arithmetic unchanged).

| # | check | where | pass |
|---|---|---|---|
| G6 | speed, in-model | level-2 profile `pcyc/op ATTN_M1=` of the Q2 variant | ≤ **210,000** at G = 64 and ≤ **350,000** at G = 1024 (round 1's gates, kept: §3.5's model lands at 121–138 k / 194–223 k with the fetch overlapped and at 209 k / 343 k with none, so the gates are reachable without a change and a miss means a term of §3 did not land — the phase words say which) |
| G6a | per-term, cold gtest | `HvxAttnM1.PerLayerCost` cold pos 1023 `ATTN_M1_PHASE` | `append` ≤ 10 k, `scores` ≤ 550 k, `softmax` ≤ 160 k, `pv` ≤ 400 k (lane-summed), `busy_max` ≤ 200 k, `pool` ≤ 220 k, `dsp_us` ≤ 110 (from 96 k / 702 k / 526 k / 1,542 k / 590 k / 619 k / 340). Read, not gated: a term over its line names the step to redo |
| G2 | host bit identity | `run_host_checks.sh` | `ATTN M1 HF PRIM OK` (with the new vector-rounding row), `ATTN M1 BIT-IDENTICAL` at L = 1 / 63 / 64 / 65 / 512 / 513 / 1024 / 1536 × pools 0 / 3 / 7, `ATTN M1 PHASES OK`, `ALL CHECKS PASS`; `run_inproc_e2e.sh` `INPROC E2E PASS` with every resident line byte-equal to `90d88e2b`'s (`fwd-hd64 min_snr_db` 37.17 unchanged) |
| G3 | silicon bit identity | `HvxAttnM1.*`, `AttnM1F16Det.*`, `HvxM1Ops.Rope64*` | `out bad=0 bad_stats=0` at the 8 L, `append_chain bad=0`, F16Det `bad=0` at 513 / 1024 / 1536, rope `bad=0` |
| G4 | CPU-vs-HTP shadow | `dev/attn-shadow-170` rebased on the PR, `tools/htp/attn_shadow_check.py` | RQ2 `tag3_heads=1536/1536 … logits_equal_steps=8/8` at G = 8 |
| G5 | E2E text / nll | 8 prompts, G = 256, `NNTR_PPL_DECODE` forced on A | Q2 and RQ2 text ≡ A byte for byte on all 8; nll lines = A's; Q2 tokens ≡ Q1 tokens |
| standing | every E2E cell | | prefill ≥ −5 % of A (mirrored band); text identical to the switch-off run |

The issue closes on G6 with G2–G5 held (G1 was closed in S1 and no new
arithmetic primitive is added: every new vector step is a store, a
permute, a prefetch or an already-proven rounding).

## 2. Where it lives (refs @ `90d88e2b`)

**Changes.**

* `nntrainer/tensor/htp_backend/hvx/hvx_attn_m1_f32.c` — the kernel.
  * `append_head` `:180-190` and the forward's append loop `:465-474`:
    the caller rounds q / k / v as vectors (§3.3); the Kt column merge
    moves into P1's last-tile unit; `hvx_attn_m1_kv_append` `:192-215`
    (the prefill seed, one call per prompt) keeps its scalar column writes.
  * `p1_unit` `:244-280`: the column merge, the next-tile `l2fetch`, q
    splats read as vectors.
  * `p2_unit` `:284-305`: the scalar scatter `:300-303` becomes a 4-head
    transpose + masked vector stores (§3.2).
  * `pv_group` `:310-366`: a windowed `l2fetch` lead on the V rows ahead
    of the chain `:334-348` (§3.1); the output widened as vectors
    `:350-360`.
  * `pv_groups` / `p3_unit` `:369-392` → chain ranges per lane (§3.4).
  * `hvx_attn_m1_forward_prof` `:445-569`: the ET `l2fetch` at call
    start; the phase words keep their meaning (`attn_m1_det.h:116-145`,
    9 words; APPEND is still the caller's part, the merge counts in
    SCORES).
* `hvx/hvx_attn_m1_f32.h`: ctx `:76-91` gains the q-splat scratch and the
  k-row scratch; the budget note `:29-37` (+ 257 KiB heap, no VTCM); the
  threads note `:39-44` (P3 by chain range).
* `hvx/hvx_attn_m1_hf.h`: one vector rounding `hvx_hf_round_rows` (f32
  → fp16 bits: `hvx_rne16_sf` (`hvx_convert.h:98`, proven) + `hvx_hf_narrow`
  `:130-135`); no new arithmetic primitive.
* `test/htp/host/hvx_emu/hvx_hexagon_protos.h` (`vsetq2` `:241`, `vmux`
  `:251`, `vror` `:262`, `vextract` `:272` exist): gains `Q6_l2fetch_AP`
  (no-op), `Q6_vmem_QRIV` (byte-masked store), `Q6_W_vshuff_VVR`.
* `test/htp/host/attn_m1_host_check.c`: an `ATTN M1 HF PRIM` row for the
  vector rounding against `hvx_hf_bits_rne` over the exhaustive f32 range
  the scalar row already walks; `check_phase_words` `:861-880` unchanged
  (LANES is still P1's, `:873-875`).
* `test/htp/nntr_hvx_attn_m1_probe.c` + `test/htp/nntr_attn_m1_probe.h`
  `:38-44`: five cost ops appended after `FETCH_L2F` (21): `PV4_COLD`,
  `PV4_COLD_L2F`, `SCORES1_COLD`, `SCORES1_COLD_L2F`, `SCORES1_SPLAT`
  (§4 step 1). Op codes are `uint32` values: **the IDL does not change**,
  no stub regeneration, no `generate_stub.sh`.
* `test/unittest/unittest_hvx_attn.cpp`: `HvxAttnM1Probe.Cost` `:1420-1466`
  gains the five rows; `PerLayerCost` `:1103-1170` unchanged.

**Consumers checked, not moving.** `test/htp/nntr_hvx.idl` (the
`attn_m1_*` signatures and the probe's); `test/htp/build.sh` `SRCS`
`:77-88`; `HtpComputeOps` (`htp_compute_ops.cpp:1536-1552` the ATTN_M1
case, `:1726-1746` the `cache=24576 KiB` banner); `hexkl_graph.c:175-183`
`graph_op_attn_m1`; `nntr_quantize_stream`'s format tag and the loader (no
weight format touched); the `NNTR_HTP_PROFILE` stage tables and
`tools/htp_fc_report.py` (the `pcyc/op` line keeps its format);
`attn_m1_det.h` (arithmetic and phase-word list unchanged);
`mha_core.cpp`'s hook and seed.

## 3. Design

Every change below moves bytes or reorders *independent* work; no fp16
operation, operand or order of the spec changes. The spec's sequential
steps (the per-head sum over p, each PV chain over p) stay sequential.

### 3.1 PV: the V rows arrive before the chain asks (the 385 k term)

`pv_group` issues, before its chain, an `l2fetch` box of the first two
16 KiB blocks of its V rows (128 positions each: width 128, stride 128,
height 128 — the `Rtt` form of `hvx_gemm_u8i4_wh.c:33-44`), and at every
128th position the box two blocks ahead. That is S1's `FETCH_L2F` shape
(32 KiB lead, 32.9 GB/s on one lane) and never more than three boxes
queued per thread. The chain then reads L2 hits after one initial
latency. The rows per position, the four chains per row and the p-ascending
order are unchanged. Six lanes × 11 GB/s of demand exceed the 38 GB/s DDR,
so P3 becomes bandwidth-bound: 1 MiB / 38 GB/s ≈ 58 k pcyc at L = 1024
against 385 k today.

Rejected: **a DMA of the V (and Kt) slab into VTCM**. Same bytes on the
same DDR; the only extra is rule 43's bypass rate (57–69 GB/s vs 38) —
worth ≈ 45 L pcyc, but it needs a VTCM carve-out beside the MoE feed's
cross-op prefetch (#117 / #177 keep the next op's expert half-slabs there
between ops), a descriptor path outside `hexkl_dma_ring` (reset per `mm`
call), and the host stub cannot see the overlap. Kept as the round-3
lever if round 2 lands fetch-bound within 10 % of a gate.

### 3.2 Softmax: ET by masked vector stores, the sum unchanged

P2's unit holds the tile's four e vectors (heads g = 0..3, positions in
lanes). Two `vshuff` stages (halfword, then word) interleave them into
`(e0[i], e1[i], e2[i], e3[i])` 8-byte tuples, 16 tuples per vector; for
each position i a `vror` puts tuple i at lane `hq0` and one
`Q6_vmem_QRIV` with a 4-lane predicate writes it into ET row i. 64 masked
stores + ≈ 70 permutes per unit replace 256 scalar stores. The lanes past
`n_q` stay 0 forever (never in any mask), every lane `hq` of every row
p < L is written by exactly one unit, and the caller's sequential sum
`:524-536` and the max `:497-514` are untouched. The ET region (128 B × L)
is `l2fetch`ed by the caller at call start so the write-allocates hit L2.

The divides stay in P3 (they need l) and become balanced through §3.4.
The output rows are widened as vectors (`hvx_hf_widen`, two unaligned
`HVX_UVector` stores per head) instead of 64 scalar conversions.

Rejected: **the sum inside per-kv-head units** (no ET, no serial step):
eight units each running L dependent 4-lane adds is ≈ 2 × L × 4 on the
busiest lane, no better than the caller's one L-add chain over all 32
heads, and it forces 8 units on 6 lanes for the whole call.

### 3.3 Append: vectors, and the column merge where the tile is anyway

The caller rounds q (64 sf vectors → 32 hf), k and v (8 heads × 2 sf → 1
hf) with `hvx_rne16_sf` + `hvx_hf_narrow` — the same bits as
`hvx_hf_bits_rne` (a new PRIM row proves it over the exhaustive range) —
stores the v rows and the k rows (1 KiB scratch), and writes the q splat
vectors (`n_q × 64` vectors, 256 KiB: the shape S1's `scores1` measured
at 1.96 pcyc/FMA; the kernel today splats from scalar loads, which the
`SCORES1_SPLAT` probe cell prices). P1's unit for the last tile of kv
head h merges the k column first: 64 × (splat, `vmux` on the lane
`pos % 64`, aligned store) on the rows it is about to read, then runs the
four heads. The bulk seed `hvx_attn_m1_kv_append` keeps writing columns
directly, so a tile is always complete in memory after any entry returns.

### 3.4 Lane balance: P3 by chain range

P3's unit i of n takes q-head chains `[i·n_q/n, (i+1)·n_q/n)` and walks
them by kv head with the existing literal-`ng` `pv_group` (4, then the
remainder). At n = 6 and n_q = 32 the busiest lane runs 6 chains (mean
5.33, 1.125×) instead of 8 (1.5×); at n = 1 / 4 / 8 (host pools 0 / 3 / 7)
the groups are whole kv heads. Each chain's arithmetic is its own, so the
output is byte-equal at any n — the host check's three pools prove it.
P1's 128 units at L = 1024 (72 at 513) are already within 1.05×.

Rejected: **16 pv2 units** (1.125× too, but every V row is loaded twice)
and **32 pv1 units** (four times).

### 3.5 Cost model (silicon constants: S1 rates, S2 phase words; 2.106 GHz, 6 lanes; L positions)

DDR at 38 GB/s = 18 B/pcyc, so Kt (1 KiB per position) and V (1 KiB) are
57 L each: **114 L is the fetch floor** of any design that reads the
cache from DDR (L2 cannot hold six layers' KV across the MoE call).

| term | today (wall, L = 1024) | round 2 | why |
|---|---|---|---|
| append | 96 k | ≈ 4 k | ≈ 100 vector ops on the caller; the merge ≈ 2 k lane-summed in P1 |
| P1 scores | 117 k = 114 L | 65–90 L (fetch hidden behind 63 L probe … 87 L today's warm compute) | one-tile lead per lane; bandwidth-bound at 57 L |
| P2 exp + ET | ≈ 60 k | 17 L | exp 14 L (S2's estimate) + 3 L permutes; ET lines prefetched |
| serial max + sum | ≈ 25 k | 6 L + 2 k | unchanged |
| P3 divides + PV | 385 k | 46 L compute (32 L × 6.9 / 6 × 1.125 + 5 L divides) under 57 L fetch → ≈ 60 L | windowed lead; chain ranges |
| dispatch, out | ≈ 30 k | ≈ 16 k | 3 pool runs at ≈ 5 k `start_max`; vector out |
| **total** | **716 k** | **172–197 k** (L = 1024), **101–115 k** (L = 544.5) | |

In-model (× 1.13 / × 1.2 from S2): **194–223 k at G = 1024 vs 350 k;
121–138 k at G = 64 vs 210 k**. With no fetch overlap at all (P1 144 L,
P3 103 L): 343 k / 209 k — still at the gates. Per layer at T1024 the
kernel is ≈ 0.17 ms, under the CPU's 0.28–0.31 ms E2E line; at 1536 the
model reads 260–300 k (0.13–0.14 ms). The gates therefore stay at
210 k / 350 k; if S3 lands the model's midpoint, round 3 (DMA bypass)
is the only term left and is not opened by this plan.

Contract walls: no arena change, +257 KiB heap of ≈ 182 MiB, no VTCM, no
CPU fallback touched. Doc 45 §3: DMA/fetch hidden behind compute (§3.1,
§3.3), the `_det` spec unchanged, gates bit identity + text.

## 4. Steps

Rungs from `.claude/skills/hexagon-gates`. Every kernel step ends with
rung 1's `ATTN M1 BIT-IDENTICAL` at the 8 L × 3 pools and `ATTN M1 PHASES
OK`; a DSP-source step adds rung 2 (`UNDEFINED SYMBOLS OK`, skel md5).

1. **P3 + probe cells (host).** §3.1's lead in `pv_group`, §3.4's chain
   ranges, the vector output; `Q6_l2fetch_AP` / `Q6_vmem_QRIV` /
   `Q6_W_vshuff_VVR` in `hvx_emu`; the five probe ops (`PV4_COLD`: the
   pv4 chain over a cold 128 KiB slab per lane, lead 0 / 32 KiB;
   `SCORES1_COLD`: the tile loop over a cold slab, lead 0 / one tile;
   `SCORES1_SPLAT`: today's scalar-load splat shape, L2-resident) and
   their `HvxAttnM1Probe.Cost` rows, each at lanes 1 / 2 / 4 / 6 with a
   4 MiB evictor between reps as `FETCH` does.
   **Gate:** rung 0, 1, 2. The `-S` of `pv_group` shows the chain loop's
   packet count unchanged (4 FMAs per 11 packets, no spill) and the
   `l2fetch` outside it.
2. **Append (host).** §3.3: the vector rounding + PRIM row, the k-row and
   q-splat scratch, the P1 column merge, the budget note.
   **Gate:** rung 1 (`ATTN M1 HF PRIM OK` with the new row), rung 2; the
   `-S` of `p1_unit`'s FMA loop shows vector loads for q, no `vmem(r29)`.
3. **P2 + P1 lead (host).** §3.2's transpose + masked stores, the ET
   `l2fetch`, P1's next-tile `l2fetch`.
   **Gate:** rung 1, rung 2.
4. **Host E2E + consumers.** Comments in `hvx_attn_m1_f32.h`; nothing on
   the ARM side changes. `run_inproc_e2e.sh` every resident line
   byte-equal to the base tree's (G2).
   **Gate:** rung 1 complete, `tools/htp_syntax_check.sh`.
5. **Rung 3 + shadow set.** App + `unittest_hvx_attn` +
   `unittest_nntrainer_cpu_backend_fp16` + `unittest_hvx_softmax`, md5s,
   `readelf` NEEDED, `FORWARD_KINDS` count; `dev/attn-shadow-170` rebased
   on the PR (the inert commit `ca2dd5f8`'s content) and pushed.
   **Gate:** rung 3 on both sets.
6. **DEVICE S3 (unavoidable), ≈ 60 min, one sitting, one skel**, a
   `170-s2-run.sh`-style script; handoff `docs/measurements/170-attn-m1-round2.md`.
   * **Variants (4):** **A** = the `90d88e2b` set, switch off, first;
     **Q1** = the same set, `NNTR_HTP_FORWARD_KINDS=MOE,QK_NORM,ROPE,ATTN_M1`
     (round 1's kernel: the same-sitting reference for every phase word
     and `pcyc/op`); **Q2** = the new set, the same mask; **RQ2** = the
     new set, `MOE,RMSNORM,QK_NORM,ROPE,ATTN_M1` (the shadow). Banners
     `calls/token=28.00` / `77.00`, `cache=24576 KiB` in all Q cells.
   * **Microbench half (new set unless said):** (a)
     `HvxAttnM1Probe.Cost` — the five new rows × lanes; (b)
     `HvxAttnM1.*` (G3) with `PerLayerCost` warm / cold at pos 511 / 1023 /
     1535 — the round-2 phase line (G6a) — and the same test on the Q1 set
     for the reference line; (c) `AttnM1F16Det.*`, `HvxM1Ops.Rope64*`.
   * **E2E half:** G4 shadow (RQ2, G = 8); speed A / Q1 / Q2 at G = 64 /
     512 / 1024 × 2, mirrored `A Q1 Q2 | Q2 Q1 A`, prompt 512,
     `NNTR_NUM_THREADS=8`; G6: Q1-prof and Q2-prof (`NNTR_HTP_PROFILE=2`)
     at G = 64 and 1024; G5: 8 prompts at G = 256 for A / Q2 / RQ2 with
     `NNTR_PPL_DECODE` forced on A.
   * Thermal (zone0) and `mhz` at every checkpoint; stop on `0x8000040e`
     or a device md5 mismatch.
   * **Read:** G6 against 210 k / 350 k; G6a per term; the probe rows
     give each lead's hot-vs-cold ratio (rule 31's corollary: a lead
     that reads slower than lead 0 on its cold cell is turned off in the
     fold — a second, gtest-only sitting, since the default is compile-time).
7. **Fold.** PR to `state:review` with S3 filled; BENCHMARK / LEDGER
   rows (§6). If G6 holds, the default of the resident attention stays as
   #176 left it (off): the switch is #132's / the end-to-end plan's, not
   this issue's.

## 5. Risks (host vs device)

* **DDR rate and the lead.** The whole gain rests on S1's one-lane 32.9
  GB/s under `l2fetch` reproducing inside the kernel. Rule 31 says a lead
  can also harm; each lead has its own hot / cold probe cell (step 1) and
  the kernel's phase words separate P1 / P2 / P3, so a wrong lead is
  visible per term and switchable by its compile-time default.
* **L2 size (not measured here).** The windowed 32 KiB lead is sized for
  rule 31's ≈ 384 KiB-per-6-lanes budget and needs no capacity; the S2
  "warm" line (V at 122 pcyc/row even at pos 511, 1.4 MiB footprint)
  suggests the cDSP L2 is ≈ 1 MiB, which is why no design here parks a
  whole slab in L2.
* **Three `l2fetch` per thread.** Each lane keeps ≤ 2 boxes in flight
  (the caller's ET box + P1's tile, or P3's two blocks); a fourth would
  stall the thread silently — the phase words would show it as P3 or P1
  growing, not as an error.
* **Masked stores from six threads into one line.** Silicon merges byte
  enables in L2; the host emulation writes bytes under a mutex-free loop
  and cannot see a merge fault. G3's `bad=0` at the 8 L (pools of 6 lanes
  on the device) is the check; today's scalar halfword scatter already
  relies on the same property.
* **DVFS / thermal drift.** S2 ran the E2E cells at 56–65 °C; the gates
  are pcycles (frequency-free) but the fetch-bound terms scale with the bus
  clock, so Q1 (round 1) is measured in the same sitting and every cell
  carries `mhz` and zone0.
* **Stale skel.** No IDL change, but the skel changes: md5s in the
  handoff, `0x8000040e` stops the script; an old gtest against the new
  skel returns `AEE_EINVALIDFORMAT` on the new op codes, not a hang.
* **Address space.** +256 KiB q splats + 1 KiB k rows on the heap (≈ 182
  MiB free); no VTCM; the banner's `cache=` is unchanged.
* **Compiler folding.** The vector rounding uses `hvx_rne16_sf`'s sf ops,
  which the skel's flags lower to qf32 + convert (round 1 §0); the PRIM
  row and G3 catch a fold, as they did for exp16.

## 6. Docs to update

* **BENCHMARK.md.** The #170 rows are not folded yet: S2's A / Q0 / Q1 ×
  G table, the in-model `ATTN_M1` 2,465 k → 395 k / 4,751 k → 700 k, the
  per-L phase line; then S3's A / Q1 / Q2 × G, `ATTN_M1` at G 64 / 1024
  against 210 k / 350 k, the per-term line, text / nll / shadow. The ⑨
  budget row: attention ms/token (6 layers × `pcyc/op` / 2.1 GHz).
* **LEDGER.md.** ㉗: the round-2 per-L cost. Rule candidates, once S3
  confirms: *a dependent HVX chain reading one 128 B row per step from DDR
  runs at one miss per step (≈ 190 pcyc/row, 8 GB/s aggregate); a 32 KiB
  `l2fetch` window ahead of it turns it bandwidth-bound*; *a scalar
  halfword scatter into cold lines costs ≈ 45 pcyc per store*; the
  `SCORES1_SPLAT` reading (scalar-load splats vs vector loads in an FMA
  loop). Close plan 170's G6 item; note the round-3 lever (DMA bypass
  into VTCM, ≈ 45 L) as not opened.
