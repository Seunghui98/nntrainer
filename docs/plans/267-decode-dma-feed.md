# Plan 267: the DDR→VTCM feed of the decode kernels at C = 16

Issue #267 (p1). Host-only reading of the one-PD Gemma-4 26B decode token's
weight feeds from the E sitting of 2026-10-09 (`docs/measurements/260-step2-r2-3897963d8.md`
on PR #265's branch, S25 `R3CY205ZMND`, real file `nntr_gemma4_qs4cx_fc_arm.bin`,
C 16, prompts 512 / 1024; OP_TIME logs `E_p512_g512_optime.log`,
`E_p1024_g512_optime.log`) against #234 P4 (`234-p4-fcwh-decode.md`, same unit,
dummy 2-bit experts, FC WH sidecar on the arena, 0.71 miss / token). Tags: [M-E]
/ [M-P4] measured there, [G] arithmetic from measured rates, [E] estimate.
Tree `htp_decode` @ `3897963d8`. Companion: #266 (the ARM's miss read path);
this plan does not touch how the ARM reads a miss.

## 1. Goal and gate

Issue ask, made measurable: per decode kind, the feed it reads from, its
bytes / token and achieved GB/s (§2), the floor at the DMA bounds (§3), and
the feed levers ranked by ms / day with before → after ms / token (§4),
C = 16, bit-preserving, no weight-file dtype change (a lossless format
repack allowed).

Gates, every lever cell against **A = the sitting's E** (same build family,
same prompts, cool start, C 16):

* **ids identical** to A: p512 G512 = the 169 ids ending in `<eos>`
  (`E_p512_g512.log`), p1024 G512 = its 512 ids; a lever that moves an id
  fails (every lever here changes no byte of any weight or activation, so a
  mismatch is a bug, not a near-tie flip).
* **nll equal**: `NNTR_PPL=1` prompt nll 4.57345 (p512) / 3.61786 (p1024)
  to the digit (the prefill path; L1 touches its DMA flag).
* **prefill ≥ 0.95 × A** of the same sitting (L1 reaches the prefill
  kernels' weight DMA); prefill MoE dumps `bit_identical=1`.
* **memory**: the one-PD address space stays under 3 840 MiB — E at C 16
  holds 1 408 (pool) + 448 (FC arena + head attach) + 400 + 40 (KV) +
  1 395 (DSP heap) = 3 691 MiB; C 24 already failed `0x80000402` at the
  second `attn_m1` set [M-E]. Every lever below adds **0 bytes** of mapped
  or heap memory (§4, "memory" column); `heap_used_kib`, `mapped_mib`,
  peak RSS and the `S1Ceiling` line are recorded per cell.
* **misses / token unchanged** (94.96 at p512, 120.8 at p1024 [M-E]): the
  pool is #266's; a feed lever that changes the miss count is wrong.
* The per-kind readings that decide the ranking come from S0's wall-µs
  line (§4 L0), not from the pcycle line at the sitting's diluted clock.

## 2. What each kind reads from, bytes / token, achieved GB/s

Shapes [G, plan 229 §3.2]: K 2 816, 30 layers; 25 sliding attention layers
(q 16 × 256, k = v 8 × 256, o) + 5 global (q 16 × 512, k = v 2 × 512
shared, o); dense FFN 2 112; 128 experts top-8, `moe_intermediate` 704;
vocab 262 144 tied. Per expert: gate_up `[2816 × 1408]` 1 982 464 B WH +
down `[704 × 2816]` 991 232 B + 8 B × 4 224 columns = 3.008 MB.

| kind | feed today (verified) | MB / token [G] | ms at the sitting's 1 180 MHz [M-E] | ms at 2.09 GHz = pcyc / 2.09 G (wall lower bound) | GB/s at 1 180 / at 2.09 GHz | P4 ms → GB/s [M-P4] (2.09 GHz, no miss traffic) |
|---|---|---:|---:|---:|---:|---:|
| FC (q\|k\|v, o) | WH tiles on the **DSP heap** (`register_locked` → `hexkl_weight_u8i4_register`, `hexkl_mm_u8i4_dma.c:166,180`: `malloc`, `borrowed = 0`), `hexkl_mm_u8i4_fc_m1_run` (`hexkl_mm_u8i4_moe.c:2792`): 8 lanes, each a VTCM double buffer of `FC_M1_BLOCKS` = 4 blocks on its own DMA queue (`:2720-2730`), **`src_bypass = 0`** because `moe_weight_src_bypass` (`:143-146`) needs `borrowed` | 557.8 | 34.48 | 19.47 | 16.2 / 28.6 | 10.65 → **52.4** (sidecar on the arena, bypass 1) |
| DENSE_FFN (up, gate, down) | same handles' kind, the MoE kernel's M = 1 pair path with the chunks as experts of weight 1 (`hexkl_graph.c:486-512`), the two-slab VTCM ring (`:1527-1538`), **`src_bypass = 0`** (heap) | 269.3 | 17.58 | 9.93 | 15.3 / 27.1 | 5.33 → 50.5 |
| LM_HEAD (tied, Q4_0) | Q4M1 slices (16 × 16 384 rows, `htp_compute_ops.cpp:1895`) **attached** from the ARM's Q4_0 buffer (`attach_mib=396`, borrowed, bypass 1); `nntr_hvx_fc_q4m1_graph` (`nntr_hvx_fc_q4.c:370-391`): VTCM feed, 6 lanes, each a double-buffered 32-column group on its own queue (rule 49: 45.5 GB/s) | 415.2 | 14.76 | 8.33 | 28.1 / 49.8 | 8.18 → 50.8 |
| MOE (240 experts) | pool slots on the S1 arena (borrowed, **bypass 1**, `dma_bypass=1` in the banner), the one-queue ring (`dma_q=1`), two gate_up slabs + packed downs (`:1091-1107`, `:1895-2010`); with a miss the present experts after the first miss and every arrived expert run **one call each** (`hexkl_graph.c:113-170`) | 721.9 | 188.3 **incl. the 208.5 ms wait** (the op's pcycles span `tk_miss_wait`, `hexkl_token.c:146-170`) | not separable [M-E]; ≈ 20–30 [E] (§4 L0) | — | 10.4 net → 35 (2-bit dummy, 1.52 MB experts; the 4-bit ring has **no reading on this unit**) |
| ATTN_M1 (KV fp16) | DSP heap caches, HVX direct read behind `l2fetch` leads (`hvx_attn_m1_f32.c:272-300`: P1 next Kt tile, PV two 16 KiB blocks ahead; tuned #170 round 3) | 134 (p512, mean ctx 597) / 288 (p1024, mean ctx 1 280) at 225 KB / position | 21.45 / 41.18 | 12.11 / 22.3 | 6.2 / 11.1 (p512) | 13.74 at ctx 703 → 11.5; slope 17 µs / position (plan 261 §2.3) |
| ROUTER_TOPK | f32 heap weights, `l2fetch` since PR #263 | 43.3 | 5.62 | 3.17 | 7.7 / 13.7 | 4.04 → 10.7 (before #263) |
| RMSNORM + QK_NORM + ROPE + ADD | — | — | 6.3 | 3.55 | — | 2.86 |

**Reading.** (a) FC and DENSE_FFN need **1.85 ×** the pcycles P4 needed for
the same bytes and the same kernels (FC 41.2 M vs 22.3 M), while LM_HEAD
needs the same (17.4 M vs 17.1 M). The three are DMA-bound kernels on the
same VTCM feed pattern; what differs is the **source-side L2 bypass**: P4's
FC WH sidecar was an arena chunk (`borrowed`, bypass on), E's QS4CX images
are the heap copies #4415's `register_qs4cx_weight` makes (`mapped_mib=448`
holds only the head attach + chunks; `s1_heap_kib=875 619` at load holds
the 789 MiB WH set), and the bypass bit is refused for heap slots by
design (`hexkl_mm_u8i4_moe.h:378-396`). Rule 43 measured exactly this gap:
37.3 GB/s through the L2 vs 69.3 around it on the probe, 57 in the app;
E's FC at 2.09 GHz reads 28.6 GB/s ≈ 0.9 × this unit's L2-path bound of
31.2 (rules 32 / 34). **This is the feed lever of this plan (§4 L1).**
(b) The pcycle line cannot say what the compute clock was: at 1 180 MHz
mean over a token that sleeps 72 % of its wall, the non-MoE kinds fit the
80.6 ms non-wait budget at any clock ≥ 1.47 GHz; "at 2.09 GHz" above is
the wall's lower bound. S0's wall-µs line removes the question. (c) The
MoE op's 222 M pcycles include the sleeping wait loop at whatever idle
clock the core ran; the split the issue asks for is L0.

### 2.1 The one-line split for MOE (L0)

`hexkl_graph_forward` brackets every op with `HAP_perf_get_pcycles`
(`hexkl_graph.c:1151-1153`). Add the QTimer beside it (`g->op_us[i]`), sum
per kind into `hexkl_token_stats.kind_us[]` (`hexkl_token.h:74-82`, the
same loop as `tk_pcycles` `hexkl_token.c:88-97`), and time the wait's
pcycles in `tk_miss_wait` (`:146-170`, `st->miss_pcyc`). The response
struct gains `kind_us[11]` and `miss_pcyc` (`htp_dspq_wire.h:83-98`; both
sides compile the same header, no IDL change). The ARM prints one new line
after the existing per-kind line (which stays byte-for-byte for
`docs/measurements/201-s*-run.sh`'s grep):

```
[HTP] graph per-kind us/token: RMSNORM=… FC=… … MOE=<wall> (net of miss wait <wall − miss_us>) … | compute_mhz=<(wall_pcyc − miss_pcyc) / (wall_us − miss_us)>
```

That line gives every kind in wall ms at whatever clock it ran, the MoE's
own DMA + compute, and the clock during compute — the three numbers §4's
ranking is conditional on.

## 3. Byte floors per kind

Bounds on this unit: 31.2 GB/s single-queue through the L2 (rules 32 / 34,
`R3CY205ZMND`); 52 GB/s the WH FC path with bypass on the arena [M-P4,
this unit]; 57 GB/s the bypass ring in the app (rule 43, `R3CY10WM83Y`,
LFM 5.5 MB experts); 70 GB/s the DRAM ceiling for any reader mix (rule 44).

| kind | MB / token | at 31.2 | at 52 | at 57 | at 70 |
|---|---:|---:|---:|---:|---:|
| FC | 557.8 | 17.9 ms | 10.7 | 9.8 | 8.0 |
| DENSE_FFN | 269.3 | 8.6 | 5.2 | 4.7 | 3.8 |
| LM_HEAD | 415.2 | 13.3 | 8.0 | 7.3 | 5.9 |
| MOE | 721.9 | 23.1 | 13.9 | 12.7 | 10.3 |
| ATTN_M1 KV, p1024 | 288 | 9.2 | 5.5 | 5.1 | 4.1 — not the kernel's wall (softmax-bound, 11–13 GB/s) |
| ROUTER | 43.3 | 1.4 | 0.8 | 0.8 | 0.6 |
| **weights, sum** | **1 964** | **63.0** | **37.8** | **34.5** | **28.1** |

With attention 12 (p512) / 22 (p1024), router 3.2 and norms 3.6 at
2.09 GHz on top: the token's floor on this file is ≈ 53–63 ms at 52–57 GB/s
→ **16–19 tok/s with the miss wait at zero**, 8–9 with #266's ≤ 60 ms
wait. 40 tok/s is not in this plan's reach by arithmetic — it is plan 229's
bytes (2-bit, parked) plus plan 261's kernels.

## 4. Levers, ranked by ms / day

Per-kind ms are at 2.09 GHz (the sitting's number in brackets at 1 180 MHz).

| # | lever | ms before → after | gain | days | ms / day | gate | memory |
|---|---|---:|---:|---:|---:|---|---|
| **L1** | **FC + DENSE_FFN DMA with `src_bypass = 1` on the baked heap images** | 29.4 → 16.0 [G: P4's 52.4 / 50.5 on this unit] (52.1 → 28.3) | **−13.4** | 1 | **13.4** | ids = A, nll =, prefill ≥ 0.95 × A + prefill MoE dumps `bit_identical=1` (the flag reaches `moe_push_weight_chunk` `:152-182` too), `moe_layer_host_check` unchanged | 0 |
| **L0** | **OP_TIME split** (§2.1) + the spin cell | 0 | 0 (the instrument) | 0.5 | — | the new line's `kind_us` sum = `wall_us − miss_us` ± 1 %; `token_host_check` runs the stats path | +48 B in the dspq response |
| L2 | **DSP spins through the miss wait** (`NNTR_HTP_E2E_SPIN_US=12000`, exists: `htp_compute_ops.cpp:4463-4475`, `hexkl_token.c:158-166`) — a sitting cell, no code | non-MoE kinds 80.6 (at the diluted clock) → 56.5 if the compute clock is < 2.09 GHz because the core sleeps 29 × per token | 0…−24 [E, bound] | 0.25 | ≤ 96 or 0 | ids = A; read only beside L0's `compute_mhz` | 0 |
| **L3** | **miss-path batching**: the present experts after the first miss as one call, the arrived experts as one call, each writing its `0 + w × res` row (new flag `HEXKL_MOE_FLAG_ROWS_OUT`, bit 25; the scatter `:2015-2024` writes `out + i × N_out` instead of accumulating), the rows then added in expert order as today | ≈ 180 single-expert calls / token → ≈ 60 calls; MoE net ≈ 24 [E] → ≈ 17–19 | −4.5…−8 [E] | 2 | 2.2–4 | ids = A (`graph_host_check` gains a miss case: batched == one-at-a-time bit for bit; `moe_layer_host_check` scoreboard on the flag; MoE dumps `bit_identical=1`) | 0 (`g->moe_rows` is already `top_k × N_out`, `hexkl_graph.c:705`) |
| L4 | MoE 4-slab ring at 4 bits (plan 261 ④ re-ranked) | ≈ 0…−2 [E] | ≤ 2 | 2 | ≤ 1 | as L3 | 0 (VTCM: 4 × 1 982 464 = 7 929 856 B of the 8 388 608 − HMX config; fits iff the config block ≤ 448 KiB — the stand-in's 2 KiB says yes, the device's `nntr_hvx_open` FARF reads the real one; plan 229 §3.3's "does not fit" did not do this subtraction) |

**Order: L0 → L1 (one sitting: A, B = L0, C = L0 + spin, D = L0 + L1) →
L3 (second sitting) → L4 only on S0's trigger.** L1 + L3 together:
DSP 80.6 → ≈ 60 ms at full clock [G/E] = the 16 tok/s of §3 once #266
takes the wait down; the two plans are independent in code (L3 is the
DSP half of #266's ask 2, "compute the resident experts first, missing
experts last", and is owned here).

### 4.1 L1 — the bypass on the heap images

`hexkl_weight_u8i4_register` bakes the WH tiles through VTCM into a
`malloc`'d block (`hexkl_mm_u8i4_dma.c:200-230`, `:166`); the DSP wrote
it, so the L2 may hold dirty lines and the bypass rule refuses it. The
image is written exactly once, at registration, and never again
(prefill and decode only read it). So: after the bake, one
`qurt_mem_cache_clean(wh_bytes, FLUSH, DCACHE)` over the image (and its
`arrays`), then `h->clean = 1`; `moe_weight_src_bypass` returns
`src_bypass && (h->borrowed || h->clean)`. Three places, ≈ 15 lines, the
same `tk_clean` the token driver already uses (`hexkl_token.c:49-58`).
The flush of the 789 MiB set costs ≈ 0.1–0.2 s at load, once. Every
reader of these handles — the token's FC / DENSE_FFN feed, #4415's prefill
`moe_push_weight_chunk` and the HMX layer path — then reads around the L2
with bytes unchanged; the prefill gate reads whether their prefill moves
(expected ≥ 0, rule 43's M > 1 `dsp` −2.7 %).

Rejected: registering the QS4CX images on an ION arena (`borrowed`) —
it changes #4415's `register_qs4cx_weight` / `transformer.cpp:570` path
(rule 1 of plan 260 §7: their prefill unchanged), adds ARM-side
allocation and mapping, and buys the same bit. Also rejected: a format
repack to the WH sidecar file (`fc_wh_file_name`) — it exists for LFM
(`set_fc_wh_file`, `htp_compute_ops.cpp:5119-5165`), is a lossless 4-bit
repack of QS4CX per column, lands on the FC arena (borrowed) and so gets
the bypass for free — but it costs +789 MiB of mapped arena beside the
789 MiB heap copy their prefill keeps (two homes for one set; the C 16
PD has ≈ 150 MiB of headroom), unless their registration is redirected,
which is the rejection above. The flush is the one-line form of the same
lever.

### 4.2 L3 — batching the miss path

Today (`graph_moe_miss`, `hexkl_graph.c:113-170`): the experts before the
first miss run as one call into `out`; every present expert after it and
every arrived expert runs alone into its own row of `g->moe_rows`; the rows
are added at scale 1 in expert order. At 3.2 misses per layer the first
miss sits at position ≈ 2 of 8 on average, so ≈ 6 experts per layer × 30 =
≈ 180 single-expert calls per token [E from the sitting's 95 misses /
token]. A single-expert call exposes what the ring otherwise hides: GU(0)
1.98 MB lands behind only the scan (≈ 8 µs), the down 0.99 MB lands behind
only the requant, and the call pays its own 3–4 pool runs — ≈ 25–45 µs
over the matrix's 52–85 µs of transfer at 57–35 GB/s. Batched, the present
set streams in the ring with the feed one matrix ahead (`:1895-2010`) and
the arrived set likewise after the wait. Bits: expert i's row is
`0 + w_i × res_i` in both forms (one rounding of the product, the add
with 0 exact), and the final adds keep today's order; the all-resident
layer is untouched. The flag rides `env->moe_flags` for that call only
(not `moe_set_opts`, so the `HEXKL_MOE_FLAGS_KNOWN` echo is unchanged);
`MOE_M1_MAX_EXPERTS` 16 ≥ `HEXKL_GRAPH_MISS_MAX` 16 holds the set.

Rejected: letting the kernel accumulate the present set straight into
`out` — the kernel adds in ascending expert order inside one call, so
`(w0 r0 + w2 r2) + w5 r5` before `w3 r3` is not the resident order; the
rows are what keep the bits.

### 4.3 L4 — why the 4-slab ring is last

At 4 bits a gate_up is 1.98 MB (35–57 µs of DMA) against ≈ 5–10 µs of
GEMV per matrix at M = 1 (plan 229 §3.1: 25 µs of compute per 340 µs of
DMA on LFM). The one-queue ring posts GU(i + 2) at A(i)'s join while
GU(i + 1) is still in flight, and D(j + 4) at C(j)'s join with D(j + 1..3)
queued: the engine is never starved on a resident layer, so a deeper ring
moves ≈ nothing there; its only gain is at call boundaries, which L3
already cuts 3×. Plan 261 ④'s −2 ms was priced on the 2-bit dummy's 35
GB/s, whose cause (the 2-bit LUT GEMV's compute per byte is 2× the
4-bit's; this unit's engine; the per-matrix overhead at 0.99 MB) was never
separated. **Trigger**: S0's MoE net rate on the miss-free layers (the
route log tells which layers had no miss; ≈ 4 % of them at 3.2 misses /
layer) reads < 45 GB/s with L1 landed. Else closed, and the ledger says so.

### 4.4 Already covered by plan 261 — not re-planned, re-ranked

* **Router (lever 1, PR #263 landed)**: 5.6 ms at the diluted clock, 3.2
  at 2.09 GHz against its 0.8–1.0 floor; the gate (≤ 1.5) is re-read on
  S0's wall line. Stays plan 261's.
* **ATTN_M1 (3a / 3b)**: no feed lever. The kernel reads 134–288 MB in
  12–22 ms = 11–13 GB/s behind leads that #170 round 3 tuned (P1 649 k →
  387 k pcyc); a DMA KV feed would hide bytes the kernel is not waiting
  for. At p1024 the 41 ms (22 at 2.09 GHz) is the per-position softmax
  (17 µs / position): plan 261 3a (hf exp over 64 positions) then 3b
  (int8 masters, halves the KV heap — the C lever rule 71 names) own it.
* **Softcap, hook-less layers (levers 2, 5)**: ARM side, unchanged.
* **MoE 4-slab (lever 4)**: § 4.3, re-ranked last with the trigger.

### 4.5 Void on this unit (read, closed)

* **LM_HEAD → WH 4-bit repack** (plan 229 §8.1 2a): the tied head is
  Q4_0 (`q4m1 weights=206 handles=16`: 205 QS4CX FCs + 1 Q4_0 head in
  16 slices); Q4_0 → WH per-column is a re-quantization (rule 73), not a
  format repack: **not bit-preserving, rejected.** The head already
  reads at 50.8 GB/s [M-P4] on the attached arena with the bypass.
* **LM_HEAD lanes 6 → 8**: 45.5 (rule 49) vs 50.8 measured; ≤ 0.3 ms.
* **FC M = 1 blocks / lanes**: with the bypass the kernel reads 52 GB/s =
  0.9 × the in-app bypass rate; the first-block exposure is 1/32 of the
  op. Nothing to win beside L1.
* **`NNTR_MOE_DMA_QUEUES` > 1 (#177)**: rule 43 — on the S25 one queue is
  the bypass ceiling, more queues add nothing; v81-only (#197).
* **Poll granularity** `HEXKL_TOKEN_POLL_US` 20 µs × 29 rounds ≤ 0.6 ms;
  L2's spin covers it.

## 5. Where it lives (verified `path:line`)

* L0: `nntrainer/tensor/htp_backend/hmx/hexkl_graph.c:1151-1153`,
  `hmx/hexkl_token.c:88-97`, `:146-170`, `hmx/hexkl_token.h:74-82`,
  `htp_dspq_wire.h:83-98`, `test/htp/nntr_hvx_token.c:138-161`,
  `htp_compute_ops.cpp:4770-4800`. Host: `test/htp/host/token_host_check.c`.
* L1: `hmx/hexkl_mm_u8i4_dma.c:200-230` (register, bake, `:166` malloc,
  `:180` borrowed), `hmx/hexkl_mm_u8i4_moe.c:139-146`
  (`moe_weight_src_bypass`), `hmx/hexkl_mm_u8i4_moe.h:378-396` (the rule's
  comment, amended), the `hexkl_weight_u8i4` struct's new `clean` bit.
  Readers unchanged: `:152-182`, `:1110-1127`, `:2720-2730`.
* L3: `hmx/hexkl_graph.c:87-170`, `hmx/hexkl_mm_u8i4_moe.c:2015-2024`,
  `hmx/hexkl_mm_u8i4_moe.h:420-432` (bit 25). Host:
  `test/htp/host/graph_host_check.c`, `moe_layer_host_check.c`,
  `moe_opts_host_check.c`.
* L4 (if triggered): `hmx/hexkl_mm_u8i4_moe.c:1091-1107`, `:1527-1538`,
  `:1895-2010`; host `moe_layer_host_check.c` scoreboard, `dma_replay` /
  `dma_trace` checks.
* **Not touched**: `test/htp/nntr_hvx.idl` and `generate_stub.sh` (no
  method changes; the dspq response is a shared header), `HtpComputeOps`
  contracts (one print line added), `nntr_quantize_stream` and the loader
  (no format tag), `NNTR_HTP_PROFILE` stage tables (`hexkl_probe_on = 0`
  in the token, `test/htp/nntr_hvx_graph.c:281`), `tools/htp_fc_report.py`.

## 6. Steps (each ends in a `.claude/skills/hexagon-gates` rung)

* **S0 — L0 (0.5 d) + the first sitting (0.5 d, device unavoidable).**
  Rung 1 (`run_host_checks.sh` `ALL CHECKS PASS`, `token_host_check` with
  the `kind_us` sum check, `run_inproc_e2e.sh` lines unchanged), rung 2
  v79 + v81 (md5 in the table), rung 3. Variants, ≤ 4, cool start, p512 /
  p1024 × G 64 / 512 / 1024 for B and D, G 512 only for C: **A** = the
  sitting's E (reference, unchanged binary set), **B** = A + L0 (ids = A;
  prints the wall line), **C** = B + `NNTR_HTP_E2E_SPIN_US=12000`, **D** =
  B + L1. Read: per-kind wall ms, `compute_mhz`, MOE net of wait, FC and
  DENSE_FFN wall in D against P4's 10.65 / 5.33, misses / token, nll,
  prefill. Decisions: C's `compute_mhz` < 2.0 GHz ⇒ L2 is a lever (its
  ms is C − B on the non-MoE kinds); D's FC ≤ 12 ms ⇒ L1 lands as the
  default (bit-identical, rule 33's three-G condition); MOE net rate on
  miss-free layers < 45 GB/s ⇒ L4 opens.
* **S1 — L1 (1 d, before S0's sitting so D rides it).** Rung 1
  (`moe_layer_host_check` unchanged, `run_inproc_e2e.sh` `bit_identical=1`
  lines: the in-process stand-in has no L2, so the host proves bytes
  only; the device cell D proves the flush), rung 2, rung 3.
* **S2 — L3 (2 d).** Spec: the miss case in `graph_host_check.c` (routed
  set with misses at positions {0}, {2, 5}, {7}, {all}: batched output ==
  one-at-a-time output bit for bit, and == the all-resident output when the
  arrived handles equal the resident ones); `moe_layer_host_check` on the
  rows-out flag (scoreboard unchanged, rows written not accumulated), one
  negative (the flag with `M > 1` refused). Rung 1–3. **Second sitting**:
  A = D of S0 (or the sitting's E if L1 did not land), E′ = A + L3, p512 /
  p1024 × G 64 / 512 / 1024; gate ids = A, misses / token = A, MOE net
  −4 ms or better at G 512, MoE dumps `bit_identical=1`.
* **S3 — L4** only on S0's trigger: schedule + host proof (`dma_replay` /
  `dma_trace` on the 4-slab schedule), rides S2's sitting as a fourth
  variant if ready, else its own.

## 7. Risks

* **DMA rate per unit** (rules 30 / 32 / 34): the sitting unit's
  single-queue bound is 31.2 GB/s through the L2; its bypass rate was never
  probed (57 is `R3CY10WM83Y`'s). L1's gain is read as D's FC / DENSE wall
  against P4's 10.65 / 5.33 on the same unit, not against 57.
* **DVFS**: the sitting's 1 180 MHz is a mean over 72 % sleep; every
  per-kind ms in §2 is clock-conditional until S0's wall line. The handoff
  table carries `mhz`, `compute_mhz` and the wall line per cell; C
  isolates the clock from the DDR contention of the ARM's miss reads
  (which persists in C and is #266's).
* **The flush's correctness is device-only**: the host stand-in has no
  L2; a stale line would show as an id mismatch in D (every image is
  read-only after registration, so the risk is a missed flush of
  `arrays`, covered by flushing both).
* **Thermal drift between sittings**: A re-run first and last in each
  sitting (the sitting doc's −6 % decode over 7 h); every verdict is
  same-sitting.
* **Stale skel** (rule 3): L0 / L1 / L3 change DSP sources; the dspq
  response struct changes size in L0 — both sides from one build, md5s
  of skel and app in the table; a stale pair fails the response size
  assert, not silently.
* **Address space**: 0 bytes added by every lever; the C 16 PD stays at
  3 691 MiB. L4's VTCM fit is read from the device's config size before
  any code.
* **#266 lands first or in parallel**: L3's gain scales with the layers
  that miss (≈ 96 % at C 16); a #266 that overlaps the ARM's read with the
  present experts needs exactly L3's "present set as one call". The
  ordering of the two plans' sittings is the supervisor's; the gates do
  not interact (misses / token is read in both).

## 8. Docs to update

BENCHMARK.md Gemma goal row: the S0 cells (B / C / D) and S2's E′ beside
the sitting's E, with the wall line's per-kind ms replacing the pcycle ms
from now on. LEDGER: a rule — "a DMA-fed WH kernel on a DSP-baked heap
image reads through the L2 (`borrowed = 0` refuses the bypass): 28.6 vs
52 GB/s for the same bytes" (after D); ㉘ / plan 229 §8.1's rows 2a (head
WH: rejected, Q4_0 head) and 4 (4-slab: at 4 bits fits, gain ≈ 0 on a
saturated one-queue ring, trigger named); plan 261 §2's table gains L1 /
L3 rows and the ④ re-rank; contract §1 Speed: 16–19 tok/s is this file's
feed ceiling at zero wait. `docs/measurements/267-s0-<commit>.md` for the
sitting with the four variants and the md5 lines.
