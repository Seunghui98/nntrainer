# 170, round 3 — the fast bit-identical ATTN_M1: softmax split and fixed, P1 decided on the device

Issue: dlwlzzero/nntrainer#170 (p1, contract §12: decode NPU end-to-end,
bit-preserving). Read against `htp/170-round2` @ `d0cc947e` (PR #182, open
into `htp_moe`; this plan stacks on it). Round 2: plan
`docs/plans/170-attn-m1-round2.md` (merged, PR #179), handoff
`docs/measurements/170-attn-m1-round2.md` (S3 filled). Round 2 is
bit-identical on silicon (G3 / G4 / G5, text approved 2026-09-30) and misses
G6 narrowly: in-model `pcyc/op ATTN_M1` **250,289** at G = 64 (gate 210 k,
1.19×) and **373,642** at G = 1024 (gate 350 k, 1.07×). PV and append landed;
scores and softmax did not.

## 0. Where round 2 left the call (S3, pos 1023 cold, 6 lanes, lane-summed pcycles) and what the ISS says the softmax word is

| term | S3 cold | S3 warm | round 1 cold | reading |
|---|---|---|---|---|
| append (caller) | 32.9 k | 33.0 k | 95.2 k | 2048 splat-vector stores (256 KiB) + 48 vector roundings |
| scores (P1) | 668 k | 636 k | 698 k | the next-tile lead, the k merge and the 32 KiB of splat loads per unit cost more than they saved (warm +9.5 %); the probe splits on the splat source (vectors win at 6 lanes, scalar at 1) |
| softmax (P2 + caller max / sum + P3 divides) | 518 k | 526 k | 528 k | unchanged by the masked ET stores: the word lumps five pieces and none was ever timed |
| pv (P3) | 386 k | 362 k | 1,545 k | 11.8 pcyc per FMA lane-summed = the probe's cold + 32 KiB-lead cell (11.76): **at the DDR line** |
| busy_max / pool / dsp_us | 278 k / 312 k / 165 | 272 k / 307 k / 162 | 595 k / 625 k / 342 | imbalance 1.09×; in-model = 1.20× pool at G 1024, 1.33× (pool + append) at G 64 |

**Softmax split, measured before designing** (this plan's step 0, done while
planning; `hexagon-sim -mv79 --timing`, pool NULL, one thread, pos 1023;
exploration under contract §12, not a gate; the harness is a copy of the
round-2 kernel with seven pcycle brackets, no source in the tree touched):

| piece | ISS pcycles | share of the word | per 64-lane vector |
|---|---|---|---|
| P2 exp16 | **270–272 k** | **65 %** | 531 (512 vectors) |
| P3 divides | 88–90 k | 21 % | 172 (512 vectors) |
| P2 ET transpose + masked stores | 40 k | 9.5 % | 313 per unit |
| caller sum | 8.6 k | 2 % | 8.4 per dependent add |
| caller max | 5.8 k | 1.4 % | |
| softmax word | 422 k | | S3 silicon 526 k warm: the ISS is 0.8× the lane-summed silicon word for a compute-bound piece |

Other brackets: the k merge 3.7 k, the q splat stores 13.6 k of append's
15.8 k, scores 267–279 k (8.2 pcyc per FMA: the qf32 accumulator chain is
latency-bound on one thread; the 6-lane probe hides it), pv 186 k.

So the S2 inference that the ET scatter was the cost (≈ 2.8 k per unit) was
wrong, as S3 suspected: **exp16 is 65 % of softmax and the largest compute
piece of the call after P1**. And it is *issue-bound*, not latency-bound:
`hvx_hf_exp16_sf` is ≈ 69 vector ops per 32 lanes (each spec rounding = one
qf32 op + one `Vsf_equals_Vqf32`), 145 per 64-lane vector, 3.7 pcycles per
op. Interleaving the four heads' chains (8 independent sf chains, same ops
per lane) read 272 k → 239 k (×4) / 231 k (×8): −12 / −15 %, hash-identical.
Interleaving the divides over two tiles: 88 k → 71 k. Neither is the lever.

**The lever is the spec's own table.** `attn_m1_det_exp_table` /
`attn_m1_det_exp_index` (`nntrainer/tensor/attn_m1_det.h:278-299`) are
"a checked equivalent" of exp16, and `attn_m1_host_check.c:297-315`
already proves `tab[exp_index(d)] == exp16(d)` for every fp16 d ≤ 0 (−0 …
−inf, 31,745 values, the `kernel table == exp16: bad=0` row). The index is a
3-op vector function of the fp16 bits (§3.1). ISS with the table and a
batched scalar gather: **exp 272 k → 115 k (−58 %), the softmax word 422 k →
239 k (−43 %), the call 887 k → 714 k (−19 %)**, output and probability rows
hash-identical to the round-2 kernel at pos 511 / 1023 / 1535.

## 1. Goal and gate

**Goal (issue, unchanged).** ATTN_M1 per layer at or below the CPU's
per-layer cost (≈ 0.15–0.3 ms at L 1024–1536), bit-identical to the Android
fp16 CPU attention (`attn_m1_det.h`): every fp16 operation and its order
unchanged; round 3 changes only how exp16 is *looked up* (a checked table),
how q reaches the P1 lanes (a permute instead of 2048 stored splats), and
what is measured.

| # | check | where | pass |
|---|---|---|---|
| G6 | speed, in-model | level-2 profile `pcyc/op ATTN_M1=` of Q3 | ≤ **210,000** at G = 64 and ≤ **350,000** at G = 1024 (kept from round 1; §3.6 projects 300–310 k at G 1024 and 190–210 k at G 64, the G 64 cell decided by the P1 read of §4 step 6) |
| G6a | per-term, cold gtest, pos 1023 | `HvxAttnM1.PerLayerCost` `ATTN_M1_PHASE` (14 words after step 1) | lane-summed: `append` ≤ 8 k (from 33 k), `exp` ≤ 150 k (from ≈ 340 k), `div` ≤ 120 k, `et` ≤ 55 k, `max` + `sum` ≤ 20 k, `softmax` ≤ 300 k (from 518 k), `scores` ≤ 600 k (from 668 k; the best of the four skels), `pv` ≤ 400 k, `busy_max` ≤ 245 k, `pool` ≤ 275 k (from 312 k), `dsp_us` ≤ 145. Pos 511 cold: `pool` ≤ 148 k (from 165 k). Read, not gated: a term over its line names the step to redo |
| G6b | the split itself | the R2w skel's phase line (round-2 arithmetic + the new words) | `exp` / `softmax` between 0.55 and 0.75 (the ISS says 0.65); `softmax` ≥ `exp` + `et` + `max` + `sum` + `div`; R2w's `scores` + `softmax` + `pv` within 15 % of S3's Q2 line (else the sitting is drifted: flag, continue) |
| G2 | host bit identity | `run_host_checks.sh` | `ATTN M1 HF PRIM OK` with two new rows (the vector exp index at every fp16 d ≤ 0; the vlut16 splat at every d for random rows), `ATTN M1 BIT-IDENTICAL` at L = 1 / 63 / 64 / 65 / 512 / 513 / 1024 / 1536 × pools 0 / 3 / 7 × shapes (8, 4) / (1, 2) / (2, 3) / (1, 8), `ATTN M1 PHASES OK` with the word identity, `ALL CHECKS PASS`; `run_inproc_e2e.sh` `INPROC E2E PASS` with every `E2E …` line byte-equal to `d0cc947e`'s (`fwd-hd64 min_snr_db` 37.17) |
| G3 | silicon bit identity | `HvxAttnM1.*`, `AttnM1F16Det.*`, `HvxM1Ops.Rope64*` on the winner skel | `bad=0 bad_stats=0` at the 8 L, `append_chain bad=0`, F16Det `bad=0` at 513 / 1024 / 1536, rope `bad=0` |
| G4 | CPU-vs-HTP shadow | `dev/attn-shadow-170-r3`, `tools/htp/attn_shadow_check.py` | Q3 and RQ3 `tag3_heads=1536/1536 … logits_equal_steps=8/8` at G = 8, nll = An |
| G5 | E2E text / nll | 8 prompts, G = 256, `NNTR_PPL_DECODE` forced on A | Q3 and RQ3 text ≡ A byte for byte on all 8; nll lines = A's; Q3 tokens ≡ Q2 tokens |
| standing | every E2E cell | | prefill ≥ −5 % of A (mirrored band); text identical to the switch-off run |

The issue closes on G6 with G2–G5 held. If S4 holds G6 at G = 1024 and
reads G = 64 in (210 k, 235 k] with every G6a term at its line, the fold
records that and opens round 4 (the VTCM levers of §3.5) instead of a
round 3b; the default stays off either way (the switch is the end-to-end
plan's).

## 2. Where it lives (refs @ `d0cc947e`)

**Changes.**

* `nntrainer/tensor/attn_m1_det.h:116-145` — the phase words: five new
  indices after `CALL_QT` (`EXP` 9, `ET` 10, `MAX` 11, `SUM` 12, `DIV` 13),
  `ATTN_M1_PROF_WORDS` 9 → 14; the comment block gains their meaning.
  `SOFTMAX` keeps its meaning (P2 + the two caller steps + the divides) so
  the S2 / S3 lines stay comparable. Nothing arithmetic in this header moves.
* `nntrainer/tensor/htp_backend/hvx/hvx_attn_m1_f32.c`:
  * `p2_unit` `:341-387`: exp16 by the table (§3.1) — the index vectors of
    the unit's heads, one gather loop, the reloads, then the transpose as
    today; the `EXP` / `ET` brackets (`if (ps)`, as `pv_group` `:416-420`
    does; a NULL `prof` takes no timestamp).
  * `pv_group` `:403-420`: the divide bracket also adds to `DIV`.
  * `hvx_attn_m1_forward_prof` `:609-652`: `MAX` and `SUM` apart (today
    `serial` holds both); `:573-584` the q rows stored once per head with
    `hvx_hf_round_row_lut` (§3.2), no splat loop; `:663-677` the fold of the
    new words.
  * `p1_unit` `:297-326`: the q operand from `vlut16` (§3.2) under
    `ATTN_M1_Q_LUT` (default 1; 0 = round 2's splat vectors, kept for the
    S4 A/B and removed in the fold).
  * `hvx_attn_m1_create` `:97-154` / `hvx_attn_m1_free`: the exp table
    (37 KiB, built from `attn_m1_det_exp_table` + `hvx_hf_bits_rne`) and
    `qs` shrunk to `n_q` rows (4 KiB).
* `hvx/hvx_attn_m1_f32.h:29-39` the budget note (−252 KiB + 37 KiB heap);
  `:79-95` ctx gains `exp_tab`, `qs` is re-described.
* `hvx/hvx_attn_m1_hf.h`: `hvx_hf_exp16_idx` (the 3-op index, §3.1) and
  `hvx_hf_round_row_lut` (`vshuffe` for `vpacke` in `hvx_hf_round_row`
  `:278-281`); `hvx_hf_exp16` stays (the probe's SEMANTICS op 7 and the
  host row keep using it).
* `test/htp/host/hvx_emu/hvx_hexagon_protos.h` (519 lines; has `vpacke`,
  `vunpack`, `vshuff`, `vmem_QRIV`, `l2fetch`): gains `Q6_Wh_vlut16_VbVhR`
  (the 128B-mode mapping of §3.2, measured on the ISS), `Q6_Vb_vsplat_R`,
  `Q6_Vb_vadd_VbVb`, `Q6_Vuh_vsub_VuhVuh_sat`, `Q6_Vuh_vmin_VuhVuh`,
  `Q6_Vh_vshuffe_VhVh` — each diffed against `hexagon-sim -mv79` on iota
  inputs before use (round 2's pattern for `vshuff`).
* `test/htp/host/attn_m1_host_check.c`: two `ATTN M1 HF PRIM` rows (§3.1,
  §3.2); `check_phase_words` `:902-943`: `taken[]` gains the five words and
  the identity `SOFTMAX ≥ EXP + ET + MAX + SUM + DIV` (the host stub's
  counter is monotonic, so the sums hold by construction).
* `test/unittest/unittest_hvx_attn.cpp`: `m1_print_phase` `:1057-1085`
  prints `exp= et= max= sum= div=` after `softmax=`; `PerLayerCost`
  `:1103-1170` unchanged; the `2u * kM1Nq + ATTN_M1_PROF_WORDS` stats
  buffer `:1031` follows the macro.
* `test/htp/nntr_hvx_attn_m1.c:114-135`: the stats-length check follows the
  macro (a gtest built against the old header gets `AEE_EINVALIDFORMAT`,
  not a hang; both binaries come from one tree, md5s in the handoff).
* `docs/measurements/170-attn-m1-round3.md` + `170-s4-run.sh` (step 5).

**Consumers checked, not moving.** `test/htp/nntr_hvx.idl` (`:589-591`
names the word count symbolically; the `attn_m1_*` signatures and the
probe's are unchanged: **no IDL change, no `generate_stub.sh`**);
`test/htp/build.sh` `SRCS` `:77-92` (no new source; the four S4 skels are
`HEX_EXTRA_CFLAGS` builds, rung 2's variant rule); `HtpComputeOps`
(`htp_compute_ops.cpp` passes no phase words: the ARM app is byte-equal to
round 2's but the shadow commit); `hexkl_graph.c` `graph_op_attn_m1`;
`nntr_quantize_stream`'s format tag and the loader (no weight format
touched); the `NNTR_HTP_PROFILE` stage tables and `tools/htp_fc_report.py`
(the `pcyc/op` line keeps its format; the phase words are the gtest's
channel only); `attn_m1_det.h`'s arithmetic, `attn_m1_det_exp_table` /
`_index` (used, not changed); `mha_core.cpp`'s hook and seed;
`test/htp/nntr_hvx_attn_m1_probe.c` (no new probe op: the "probe" of this
round is the split phase line of the R2w skel, §4 step 6).

## 3. Design

No fp16 operation, operand or order of the spec changes. exp16 becomes a
lookup of the value the spec's checked table holds for that fp16 input; the
q splats become a permute of the same rounded row; the rest is
instrumentation. Every step keeps round 2's rules: no `l2fetch`, timer or
sf op inside any chain loop; the sequential per-position sum and the
p-ascending PV chains untouched.

### 3.1 exp16 by the checked table (the 65 % piece)

For an fp16 d ≤ 0 the spec's index is `((f32bits(d) & 0x7FFFFFFF) >> 13) −
0x1C3FF`, clamped to `[0, EXP_LAST + 1]`. For a normal fp16 magnitude
`hbits = e << 10 | m` the f32 bits shifted are `hbits + 0x1C000`, so the
index is **`hbits − 1023`**; a subnormal (hbits < 0x400) gives a negative
value, i.e. 0 (exp16 = 1); anything at or past 17.5 gives the last entry
(exp16 = 0). As vectors: `mag = d & 0x7FFF`, `idx = vsub_sat_uh(mag,
1023)`, `idx = vmin_uh(idx, EXP_N − 1)` — three integer ops per 64 lanes.
A new PRIM row walks every fp16 d ≤ 0 (the same 31,745 values `cmp_exp`
walks) and compares the vector index with `attn_m1_det_exp_index`; with the
existing `kernel table == exp16` row that makes `tab[idx(d)]` the spec's
exp16 bit for bit at every input the kernel can see (P2 feeds it
`rne16(s − m)` on fp16 lanes; −0 indexes 0 → 1.0, as `exp16(−0)`).

The table: `EXP_N` = 18,531 fp16 values (37 KiB) in the ctx, built at
create from `attn_m1_det_exp_table` → `hvx_hf_bits_rne` (≈ 1 ms once per
session), read from L2 / L1 by scalar loads. The gather: the unit's four
index vectors are stored first, one loop gathers `gqa × 64` halfwords into
a 512-byte stack row, four vector reloads follow — so the HVX-store →
scalar-load hazard is paid twice per unit, not eight times (ISS: 325 →
225 pcycles per vector; the per-head form read 166 k, the batched 115 k).
Then the tile's e vectors go into S and the ET transpose as today.

Rejected: **`vgather` from VTCM** (one instruction per vector; would take
exp16 to ≈ 25 k). The session's VTCM (`s->vtcm_base` / `config_off`,
`hvx_add_f32.c:71-90`) is planned by the MoE feed, which keeps the next
op's expert half-slabs there between ops (#117 / #177); a 40 KiB carve-out
touches that plan and the host stub cannot emulate the residency. Kept as
round 4's lever with the DMA bypass (§3.5). Rejected: **interleaving the
chains** (−12 / −15 % on the ISS, §0) and **`-mhvx-ieee-fp`** (one op per
rounding instead of two, but a different instruction sequence than S1
proved; a new G1 sitting).

### 3.2 P1's q operand: a `vlut16` splat from the rounded row

`Q6_Wh_vlut16_VbVhR(idx, row, rt)` returns, in every output lane, one
halfword of `row` selected by the byte index — a splat with no memory and
no scalar → vector transfer. Measured on the v79 ISS (128-byte mode): with
byte index `i` and `rt`, the halfword read is `2 · (i & 15) + 32 · (rt & 1)
+ (rt >> 1)`, and the match rule is `i[5:4] == rt` (else 0). So a row
stored **zipped** — halfword `2 · (d & 31) + (d >> 5)` holds q[d] — is read
as q[d] by `(i = d, rt = d >> 4)`, and `Q6_Vh_vshuffe_VhVh(hi32, lo32)` on
the two word-vectors `hvx_hf_round_row` already has *is* that zip (ISS:
`[0, 32, 1, 33, …]`), replacing `vpacke` at zero cost. P1's loop keeps its
two passes of four accumulators; per FMA the q operand is
`lo(vlut16(idx, row, dd >> 4))` with `idx` a byte vector advanced by
`vadd_b` (+1 within the four chains, +5 across the step of 8), `dd >> 4` a
scalar. The caller stores 32 rows (4 KiB) instead of 2048 splats
(256 KiB): append 15.8 k → 5.9 k on the ISS; the P1 loop 267 → 287 k on one
thread (+7 %, inside the ±5 % layout noise the div runs show in `scores`),
hash-identical. Its silicon value — 32 KiB less L2 traffic per unit, 4× the
tile's bytes — is what S4's R2w-vs-R3 `scores` read decides; the round-2
splat vectors stay behind `ATTN_M1_Q_LUT=0` for that read only.

A new PRIM row proves the splat: for random fp16 rows, every d, the vlut16
lane equals `Q6_Vh_vsplat_R(row[d])` on all 64 lanes (host emulation, whose
`vlut16` is diffed against the ISS first).

Rejected: **round 1's scalar-load splats** (the probe: 4.19 vs 2.49 wall
pcycles per FMA at 6 lanes — the scalar → vector transfer saturates) and
**two heads per Kt load** (S1: does not scale to 6 lanes; 16 live pair
temporaries spill).

### 3.3 The P1 next-tile lead and the ET lead: decided on the device, not here

S3's probe says the next-tile lead buys 11.8 → 9.96 at 6 lanes on its cold
cell and is pure cost warm; the ET lead (128 KiB at L 1024, issued by the
caller just before P1) competes with P1's tile fetches at P1's start (rule
31's interference). Neither has an in-kernel read. Round 3 does not guess:
the S4 gtest half runs the round-3 kernel as **four skels from one source**
— `r2w` (`ATTN_M1_Q_LUT=0` + exp16 as round 2: the split reference), `r3`
(defaults), `r3n` (`ATTN_M1_P1_LEAD=0`), `r3e` (`ATTN_M1_ET_LEAD=0`) — and
the winner (min cold `pool` at pos 1023, pos 511 the tie-break, ≤ 3 % apart
→ the fewer leads) is the E2E skel and the fold's default. The k merge
(3.7 k on the ISS, 0.5 % of scores) stays.

### 3.4 What stays as it is, and why

* **PV** at 11.8 lane-summed pcycles per FMA = the probe's cold + lead cell:
  bandwidth-bound on the V stream; only a faster path to the bytes (§3.5)
  moves it.
* **The divides** (88 k ISS, 172 per vector): the two-tile interleave read
  −20 % = −3.5 k wall at 6 lanes, under 2 % of the call; not taken (the
  diff is not worth a new loop shape). An f32 divide would be exact here
  (24 ≥ 2 · 11 + 2, the innocuous-double-rounding bound) but HVX has no
  vector divide; the reciprocal + midpoint test is the divide.
* **The caller's max and sum** (14 k ISS ≈ 8 µs): the sum is the spec's
  sequential order on all 32 heads at once and cannot be split; the max
  could move into P1 (order-free) for ≈ 5 k — not worth the unit change.
* **ET transpose + masked stores** (40 k ISS): correct and 9.5 % of the word.

### 3.5 Round 4's levers, named and not opened

DMA of the V (and Kt) slab into VTCM at rule 43's bypass rate (57–69 GB/s
vs 38: ≈ 45 L pcycles), `vgather` of the exp table from VTCM (≈ −90 k
lane-summed), both needing a VTCM carve-out beside the MoE feed. Opened only
if round 3 lands within 10 % of a gate with every §1 term at its line.

### 3.6 Cost model (S3 silicon words; ISS ratios for the compute pieces, 1.25 silicon / ISS)

Pos 1023 cold, lane-summed:

| term | S3 | round 3 | basis |
|---|---|---|---|
| append | 33 k | ≈ 6 k | ISS 15.8 → 5.9 k: 48 roundings, 40 row stores |
| scores | 668 k | 560–668 k | the lut / lead A/B; floor 406 k (the 6-lane probe warm, 12.4 per FMA) |
| softmax | 518 k | ≈ 295 k | ISS 422 → 239 k (exp 115, div 67–88, et 36, max + sum 14) |
| pv | 386 k | 386 k | at the DDR line |
| lane-sum → busy_max (×1.09 / 6) | 1,572 k → 278 k | 1,240–1,350 k → 226–245 k | |
| pool (+ 34 k dispatch + serial) | 312 k | **260–279 k** | |
| in-model G 1024 (×1.08 of pool + append) | 374 k | **288–308 k** vs 350 k (12–18 % margin) | |
| pos 511 cold pool → in-model G 64 (×1.33 of pool + append) | 165 k → 250 k | 136–150 k → **189–207 k** vs 210 k (1–10 %) | the low end needs the P1 read; the ×1.33 also carries 256 KiB of cold scratch per call that the 4 KiB `qs` removes — not modelled |

Contract walls: no arena change, heap −215 KiB net, no VTCM, no CPU
fallback touched, no IDL change. Doc 45 §3: fetch hidden behind compute
kept as round 2 left it, the `_det` spec unchanged, gates bit identity +
text.

## 4. Steps

Rungs from `.claude/skills/hexagon-gates`. Every kernel step ends with rung
1's `ATTN M1 BIT-IDENTICAL` at the 8 L × 3 pools × 4 shapes and `ATTN M1
PHASES OK`; a DSP-source step adds rung 2 (`UNDEFINED SYMBOLS OK`, `-S`).
The ISS harness of §0 is exploration only (contract §12): it may be re-run
per step to confirm the hash and the piece it targets, and is quoted as
"ISS" in the handoff, never as a device number.

0. **Done (planning):** the softmax split and the three candidate reads on
   the ISS (§0), the `vlut16` mapping and the `vshuffe` zip (§3.2), the
   table's index (§3.1).
1. **The words (host).** `attn_m1_det.h` 9 → 14 words; brackets in
   `p2_unit`, `pv_group`, the caller; `check_phase_words` + the identity;
   `m1_print_phase`. **Gate:** rung 0, rung 1, rung 2 (`-S`: no
   `HAP_perf` read inside the P1 FMA loop or any PV chain loop).
2. **exp16 by the table (host).** `exp_tab` in the ctx, `hvx_hf_exp16_idx`,
   the batched gather in `p2_unit`, `Q6_Vuh_vsub_VuhVuh_sat` /
   `Q6_Vuh_vmin_VuhVuh` in `hvx_emu` (ISS-diffed), the index PRIM row.
   **Gate:** rung 1 (`ATTN M1 HF PRIM OK` with the row; `kernel table ==
   exp16: bad=0` still printed), rung 2. ISS (read): `exp` ≤ 130 k at pos
   1023, hash = round 2's.
3. **`vlut16` splats (host).** `hvx_hf_round_row_lut`, the 4 KiB `qs`,
   `p1_unit` under `ATTN_M1_Q_LUT`, the four `hvx_emu` ops (ISS-diffed,
   the mapping of §3.2 written into the emulation's comment), the splat
   PRIM row, the budget note. **Gate:** rung 1 at both `ATTN_M1_Q_LUT`
   values (the host check compiles the source once; run it twice with
   `-D`), rung 2 (`-S`: the P1 loop ≤ 14 packets per 4 FMAs, no `vmem` of
   q, no `vsplat` from a scalar load, the `vlut16` in the loop).
4. **Host E2E + headers.** Comments in `hvx_attn_m1_f32.h` / `.c`;
   nothing on the ARM side changes. **Gate:** rung 1 complete,
   `tools/htp_syntax_check.sh`, `run_inproc_e2e.sh` every `E2E …` line
   byte-equal to `d0cc947e`'s (G2).
5. **Rung 3 + four skels + shadow set + S4 script.** `test/htp/build.sh`
   four times with `HEX_EXTRA_CFLAGS` (`r2w`: `-DATTN_M1_Q_LUT=0
   -DATTN_M1_EXP_TAB=0`; `r3`; `r3n`; `r3e`), each copied to
   `libnntr_hvx_skel.<v>.so` with its md5; app + `unittest_hvx_attn` +
   `unittest_nntrainer_cpu_backend_fp16` + `unittest_hvx_softmax`;
   `dev/attn-shadow-170-r3` = the branch + the inert shadow commit
   (`0c1e2202`'s content), pushed; `170-s4-run.sh` from `170-s3-run.sh`
   (§4 step 6's cells); handoff `docs/measurements/170-attn-m1-round3.md`
   with the ISS table of §0 marked as such. **Gate:** rung 3 on the new set
   (NEEDED `libsdkl` + `libcdsprpc`, `FORWARD_KINDS` 2, md5s), the round-2
   set reused from S3's `new/` dir (its md5s are in S3's `md5.txt`).
6. **DEVICE S4 (unavoidable), ≈ 75 min, one sitting**, `run_s4.sh`;
   `state:needs-measurement`.
   * **Sets:** `new` (round 3: app, 3 gtests, the four skels; the active
     skel swapped by `cp` on the device, md5 re-checked before every gtest)
     and `q2` (S3's `new/` = round 2's app + skel, the same-sitting
     reference).
   * **Variants (4, E2E):** **A** = `new`, switch off (= S3's An: the
     shadow's logits reference; no ARM code differs from round 2's but the
     inert commit), first; **Q2** = `q2`,
     `NNTR_HTP_FORWARD_KINDS=MOE,QK_NORM,ROPE,ATTN_M1` (round 2, the
     same-sitting reference for `pcyc/op` and tok/s); **Q3** = `new` +
     the winner skel, the same mask; **RQ3** = `new` + winner,
     `MOE,RMSNORM,QK_NORM,ROPE,ATTN_M1` (the shadow). Banners
     `calls/token=28.00` / `77.00`, `cache=24576 KiB` in every Q cell (rule
     36: else void).
   * **Gtest half (the probe, ≈ 25 min):** (a) `q2` `PerLayerCost` (round
     2's 9-word line, this sitting); (b) `new` + `r2w` `PerLayerCost`: the
     14-word line of round 2's arithmetic — **G6b**, the split; (c) `r3`,
     `r3n`, `r3e` `PerLayerCost`; the winner W by §3.3; `scores`(r2w) vs
     `scores`(r3) = the lut read, `scores`(r3) vs (r3n) = the P1 lead,
     `softmax` / `et`(r3) vs (r3e) = the ET lead; (d) W: `HvxAttnM1.*`
     (G3, and G6a from its `PerLayerCost`), `AttnM1F16Det.*`,
     `HvxM1Ops.Rope64*`. **Stop before the E2E half** on any `bad ≠ 0`
     in (d).
   * **E2E half (≈ 50 min), W installed:** G4 shadow (A, Q3, RQ3 at G = 8,
     forced on A's tokens); speed A / Q2 / Q3 at G = 64 / 512 / 1024 × 2,
     mirrored `A Q2 Q3 | Q3 Q2 A`, prompt 512, `NNTR_NUM_THREADS=8`; G6:
     Q2-prof and Q3-prof (`NNTR_HTP_PROFILE=2`) at G = 64 and 1024; G5: 8
     prompts at G = 256 for A (self + one forced null check), Q3, RQ3.
   * Thermal (zone0 ≤ 35 °C to start, logged at every checkpoint) and
     `mhz` in every phase line; stop on `0x8000040e` or a device md5
     mismatch (rule 3 / 36).
7. **Fold.** The winner's leads become the compile-time defaults;
   `ATTN_M1_Q_LUT=0` and the `r2w` path are removed if the lut read is a
   win or neutral (kept behind the flag only if it lost, with the reading
   in the header). PR to `state:review` with S4 filled; BENCHMARK / LEDGER
   rows (§6).

## 5. Risks (host vs device)

* **The gather on silicon.** The ISS prices the HVX-store → scalar-load
  hazard and the scalar loads from an L2-hot 37 KiB table at 225 pcycles
  per vector; silicon's scalar pipeline is per hardware thread and six
  threads gather at once. The `exp` word reads it directly (G6a); if
  `exp` > 150 k the per-head form vs the batched form is a one-line switch
  and the `vgather` lever is §3.5.
* **The lut read cannot be predicted from the ISS** (one thread, no L2
  contention): it is the reason `r2w` and `ATTN_M1_Q_LUT=0` exist. A loss
  costs one skel swap, not a sitting.
* **The G 64 gate.** §3.6's low end (189 k) needs the P1 read to land ≈ 10
  %; the high end (207 k) is at the gate. The 1.33× in-model factor at G 64
  carries 256 KiB of cold scratch that shrinks to 4 KiB — visible only in
  the profile cell, so Q2-prof is read beside Q3-prof in the same sitting.
* **Five more timer reads per P2 unit and chain.** `HAP_perf_get_pcycles`
  is a register read (≈ 10 pcycles); ≈ 300 per call, < 0.2 %. The `if
  (ps)` form keeps the production path (prof NULL) free of them; G3's
  byte equality with and without the words is the check, as #146's was.
* **DVFS / thermal drift between sittings.** All gates are pcycles, but
  the bandwidth-bound terms (P1's tiles, P3's V) scale with the bus clock;
  Q2 and the `q2` `PerLayerCost` line are measured in the same sitting, and
  G6b's ±15 % window on `r2w` flags a drifted sitting before the E2E half.
* **Stale skel / mixed builds.** No IDL change, but the word count changes:
  a gtest against a foreign skel returns `AEE_EINVALIDFORMAT` (kM1StaleProf
  names it), not a hang; four skels from one tree, md5 per swap.
* **Address space.** Heap: −252 KiB (`qs`) + 37 KiB (`exp_tab`) of ≈ 182
  MiB; the 512-byte gather row is stack; no VTCM; `cache=` unchanged.
* **Compiler folding.** No new f32 rounding point (the index is integer
  ops; the table holds the spec's values); the PRIM rows and G3 catch a
  fold as they did for exp16.

## 6. Docs to update

* **BENCHMARK.md.** The #170 rows are still unfolded (S2, S3): add S3's
  A / Q1 / Q2 × G table, `ATTN_M1` 411 k → 250 k / 669 k → 374 k, the
  per-term line; then S4's A / Q2 / Q3 × G, `ATTN_M1` at G 64 / 1024
  against 210 k / 350 k, the 14-word line with the split, text / nll /
  shadow. The ⑨ budget row: attention ms/token = 6 × `pcyc/op` / 2.1 GHz.
* **LEDGER.md.** ㉗: the round-3 per-L cost. Rule candidates, once S4
  confirms: *exp16 in the CPU's order on HVX is issue-bound (145 vector ops
  per 64 lanes, 2 per rounding point); interleaving independent chains buys
  ≤ 15 %; the spec's checked table with a batched scalar gather halves it,
  `vgather` from VTCM is the next step*; *`vlut16` in 128-byte mode reads
  halfword `2(i & 15) + 32(rt & 1) + (rt >> 1)` with `i[5:4] == rt`: a row
  stored zipped (`vshuffe`) is splatted by `(i = d, rt = d >> 4)`*; *the
  v79 ISS with `--timing` prices a compute-bound HVX phase at ≈ 0.8× the
  6-lane silicon lane-sum and sees no L2 contention — use it to rank
  pieces, never to project a fetch-bound one*; *a lumped phase word is
  read only after it is split (S2's ET-scatter inference cost one round)*.
  Close plan 170's G6 item or open round 4 (§3.5).
