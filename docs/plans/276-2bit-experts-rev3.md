# 276 — The 2-bit (ternary) expert file on the revision-3 base: intake, decode (u8i2 pool), prefill (#4415 HMX path), memory

Issue #276 (p0, hexagon, needs-user). Base `htp_decode` @ `19dbf3a78` (= upstream #4415
@ `2db6cad59` prefill + our one-PD decode + #271 miss readers + #272 L0 split / DMA bypass).
Inputs: contract §1 Model row and §12 (2026-10-07..09), plan 229 (S0 format, S1 u8i2 stack,
§7.1 slot bytes), plan 260 §7 (one slot set), plans 266 / 267 and their measurement docs
(`266-miss-readers.md`, `266-predict.md`, `267-dma-bypass.md`), upstream doc 57 §9.27 /
§9.35, `tools/htp/repack_int2_to_qs2cx.py` (#250), `tools/moe_expert_cache_sim.py` on the
r266 route logs (`route_p512_g512.txt`, `route_p1024_g512.txt`, re-run for this plan at
C 16 / 24 / 32 / 40 / 48 / 64). Tags: **[M]** measured on silicon (doc cited), **[C]** code
read at `path:line` on `19dbf3a78`, **[G]** arithmetic from [M] inputs, **[E]** estimate,
**[S]** the cache simulator on the device's own route log.

**Headline.** The 2-bit expert stack of #229 S1 is live on the revision-3 base for the pool
(LFM lines pass), but the Gemma MoE layer — #4415's `lfm2_moe`, which both the prefill LRU
and our decode hand-over go through — refuses a `QS2CX_WH` tensor (`expertDesc`,
`lfm2_moe_layer.cpp:983-997`) and never sets `w_bits`. That is the only code gap between
the file and a run; the HMX prefill path already expands 2-bit chunks in VTCM on the
background lane (`hexkl_mm_u8i4_moe.c:159-281`, `:2207`), so option (i) of ask 3 is code
that exists, not code to write. What the file buys, by arithmetic on the measured numbers:
decode at C = 16 **≈ 5.1 → ≈ 8.0 tok/s** at p512 G512 (miss wait 116 → ≈ 56 ms, MoE net
24 → ≈ 11 ms), **≈ 4.1 → ≈ 6.8** at p1024; **C = 32 at the same 1 408 MiB arena** takes
that to **≈ 10 / 9** (hit rate 60 → 81 % / 50 → 75 % [S]); prefill at 1024 moves from the
flash floor (3.37 s) onto the compute floor (≈ 2.8–2.9 s on upstream's unit), at 2048 /
4096 it is compute-bound already and the file buys ≈ 0 there (and costs ≈ +0.1 s of HVX
expansion). The 40 tok/s target stays out of reach of this file alone — the non-weight
wall (ATTN 12–22 + FC 12 + DENSE 6 + head 8 + router 3 + norms 4 ≈ 46–56 ms) caps the
token at ≈ 16–18 tok/s with zero misses.

## 0. Intake (needs-user): which file arrives, what the loader needs per case

The user's file is produced elsewhere; its format is open. Every case ends at the same
on-device image — `QS2CX_WH` experts (`[K·N/4 codes in whPack2 order][4 palette
bytes][N f32 scale][N f32 colsum]`, `qs4cx_tensor.h:316-408` [C]) in #4415's tensor
layout (`writeGemma4Moe`, `quantize_stream.cpp:1301-1410` [C]: per layer
`_attention_norm, _wq, _q_norm, _wk, _k_norm, [_wv], _attention_out, _post_attention_norm,
_pre_ffn_norm, _ffn_gate, _ffn_up, _ffn_down, _post_ffn_norm_1, _pre_ffn_norm_2,
_router_norm, _router, _per_expert_scale, 128 × (gate_up, down), _post_ffn_norm_2,
_post_ffn_norm, _layer_scalar`; then `output_norm`), FCs `QS4CX`, embedding Q4_0 — the
real 4-bit file's layout with the experts swapped. One copy on flash, one slot set (plan
260 §7, kept).

| case | what arrives | tool | loader / config | risk |
|---|---|---|---|---|
| **(a)** the external "int2" `.bin` (#250's input: no palette, `sl/4` code order, otherwise the writer's byte order) | `tools/htp/repack_int2_to_qs2cx.py in.bin config.json out.bin` → `QS2CX_WH`, every other tensor byte for byte; writes `nntr_config.json` with `moe_layer_dtype QS2CX_WH` | **one change**: `FC_BYTES` gains `"QS4CX": k·n/2 + 4·n` (`repack_int2_to_qs2cx.py:52` has only Q4_0 / FP32 [C]; the real file's FCs are `QS4CX`, embedding Q4_0 — `nntr["fc_layer_dtype"]` / `["embedding_dtype"]` pick them). The per-layer copy count in `segments()` (`:60-85`) already equals the #4415 layout's bytes (12H + 4HE + 4E before the experts, 8H + 4 after — the names moved, the sizes did not); `repack_int2_check.cc` re-walks both files byte for byte | the int2 layout assumption itself (`:28-36`): rows moved inside a column pass the colsum check; only the model-level gate (§1) catches it |
| **(b)** `QS2CX_WH` written directly (`nntr_quantize_stream --fc_dtype QS4CX --moe_dtype QS2CX_WH` from the f32 / safetensors source, or the external writer emitting whPack2 + palette) | none | `moe_layer_dtype QS2CX_WH` in `nntr_config.json`; the rest of the config as the 4-bit sitting's (C, engines, `init_seq_len`) | palette order: the loader takes the palette as written (`e.pal`, `htp_compute_ops.cpp:7236-7238` [C]), so a producer palette other than `fe ff 00 01` is legal — the sanity check reads it, it does not assume it |
| **(c1)** 4-bit `QS4CX_WH` whose nibbles are ternary (values {−1, 0, +1} → nibbles {7, 8, 9}) | a 60-line sibling of #250: `whUnpack → whPack2` with palette `ff 00 01 00` (or `fe ff 00 01` when a −2 appears), scales / colsums copied — lossless by construction (`expand_i2i4_host_check` proves `whPack2 + expand == whPack`) | as (b) | none beyond (a)'s verify path; a 4-bit nibble outside {6..9} means the file is not ternary → stop, needs-user |
| **(c2)** anything else: per-group / per-block scales, 5-per-byte base-3 packing, int8 ternary | plan 229 §3.2's table: per-group scale is **not exact** in any per-column format (bit-identity gate unavailable — a PPL plan); base-3 / int8 need a converter pass to 2-bit (allowed: format repack) | — | **stop and ask**; nothing below starts |

**Sanity checks, every case, before a device minute (S0):**
1. size: `size(2-bit file) == size(nntr_gemma4_qs4cx_fc_arm.bin) − 3 840 × 1 486 840` when
   everything but the experts is byte-identical (4-bit expert 1 993 728 + 1 013 760 =
   3 007 488 B; 2-bit 1 002 500 + 518 148 = 1 520 648 B [G from `whBytes2`, `expertStride`
   `htp_compute_ops.cpp:6658-6672`]); ≈ 7.12 GB with `QS4CX` FCs (the contract's
   7 226 142 840 B is the dummy with Q4_0 FCs); md5 recorded in the handoff and #276;
2. per expert tensor: the 4 palette bytes read back; the 2-bit code histogram over the
   codes — ternary gives three used codes (the −2 slot rare or empty: its count is reported,
   not gated); the colsum field equals the decoded column sum (the #250 checker does this
   for (a); (b) / (c1) get the same walk through `--verify-only`);
3. `nntr_config.json`: `moe_layer_dtype QS2CX_WH`, `fc_layer_dtype QS4CX`,
   `moe_cache_experts` per §3.4, `init_seq_len` ≥ the prompt;
4. a host load of the first layer's experts through `QS2CX_WH_Tensor` (the `--verify-only`
   path) — the loader's own reader, so a wrong `sl/4` assumption fails here, not on the phone.

**Questions for the user (needs-user, not blocking S1–S3):**
- Q1. Which case — (a) int2 `.bin` by #250's producer, (b) `QS2CX_WH` direct, or (c)? If (a):
  same producer script and settings as the dummy (contract §1 Model row says yes)?
- Q2. FCs and embedding in the 2-bit file: the real 4-bit file's `QS4CX` FCs + Q4_0
  embedding byte for byte (so check 1 applies), or re-quantized? (If re-quantized, the
  text / PPL reference moves — §1.)
- Q3. The accuracy reference on the same file: the hybrid A (MoE on the NPU, their LRU —
  S5-0's precedent, contract 2026-10-07 row (2)), since `QS2CX_WH` has no CPU kernel
  (`float_tensor.cpp:775` [C])? A CPU-exact reference needs a `QS4CX` (non-WH) twin of
  the same ternary codes from the producer — wanted, or not?
- Q4. C at 2 bits (§3.4): 32 at the same 1 408 MiB arena (recommended, same footprint as
  today), or keep 16 and bank 704 MiB? The plan reads both in one sitting; the default
  for the rows of record is the user's.

## 1. Goal and gate

Acceptance (issue, made measurable): the 2-bit expert file runs prefill (#4415's path)
and decode (our one-PD) on the S25 `R3CY205ZMND`, read beside the 4-bit rows:

| gate | where | pass |
|---|---|---|
| **file sanity** | §0 checks 1–4, `docs/measurements/276-intake.md` | all four; md5 in #276 |
| **host bit identity** (the 2-bit kernels against the 4-bit spec, same codes) | `run_host_checks.sh`: `expand_i2i4_host_check`, `gemv_native_check` 2-bit cell, `moe_layer_host_check` 2-bit cells (M = 1 native, M > 1 HMX with chunk expansion); `run_inproc_e2e.sh`: the `2bit lfm25` lines unchanged **plus new `2bit gemma64` lines** (§4 S1): prefill `off` 2-bit == 4-bit palette twin `bit_identical=1`, `e3` 2-bit == twin, `e3 pool C=2/3` 2-bit == `e3` 2-bit with `misses>0` | every line `bit_identical=1`, `ALL CHECKS PASS`, `*Lfm2Moe*` 7/7, `*qs4cx*` 3/3 |
| **decode text + PPL** (different file from the 4-bit rows, so no token equality across files) | E (one PD) vs A (hybrid on the same 2-bit file) per prompt: `NNTR_PPL=1` prompt nll, decode PPL ≤ 1.02 × A (`NNTR_PPL_DECODE`), text judged to the first `<turn\|>` (rule of 2026-10-08), loop onset (LEDGER rule 45: a new loop where A has none = fail), user approval; E prefill MoE dumps == A `bit_identical=1` (same kernel, same bytes) | PPL ≤ 1.02 × A, no new loop, approval y; the 2-bit file's nll **recorded beside** the 4-bit file's 4.573 / 3.618 as information (a different quantization, not a gate) |
| **decode speed** | BENCHMARK Gemma goal row: E p512 / p1024 at G 64 / 512 / 1024, miss wait, misses/token, MoE net (the #272 wall line), beside the 4-bit E rows (`266-miss-readers.md` B, `266-predict.md` S2) | the §3 before → after columns; the signature: `misses/token` unchanged at the same C, miss wait ≈ halved, MOE net ≈ halved, every other kind within noise |
| **prefill speed** | prompts 1024 / 2048 / 4096 (contract 2026-10-09), A and E, beside the revision-3 4-bit sitting's cells; `prefetch exposed wait` from the layer's profile | **≥ −5 % of the same sitting's 4-bit A at every prompt** (standing gate; the expansion cost is the only term that can move it down, §3.2); at 1024 the exposed flash wait → ≈ 0 |
| **memory** | `mapped_mib`, `s1_arena_mib`, `heap_used_kib`, peak RSS, S1 ceiling per cell | one-PD sum < 3 840 MiB at the chosen C with the 4096 prompt loaded (§3.4); arena at C = 16 ≈ 704, at C = 32 ≈ 1 408 |
| **standing** | prefill ≥ −5 % of A (above); text identical to the CPU run — **unavailable on this file** (no CPU kernel), replaced by the A reference per Q3 | — |

## 2. Where it lives (`htp_decode` @ `19dbf3a78`, verified)

**The one gap — their Gemma MoE layer, `Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.cpp`:**
`expertDesc` `:983-997` throws unless both tensors are `QS4CX_WH` and builds
`ExpertFileDesc` without `w_bits` (default 4, `compute_ops.h:616`); it feeds the prefill LRU
(`preloadExperts` `:1000-1048`, the miss path `:1268-1284`, the read-ahead `:1393-1418`)
**and** the decode hand-over (`set_decode_moe_experts` `:1213-1218`), so both sides are
4-bit-only today. `weights_wh` `:1160-1163` is a bool (`== QS4CX_WH`) passed as the
kernel's `w_bits` (`:1313`; `htp_compute_ops.cpp:1881` reads 1 as 4) — at `QS2CX_WH` it
reads false and the layer takes the CPU gate at `:1182`. Ours already has the three-way
form (`lfm2_moe_pool_layer.cpp:557-572`, `:1040-1071` [C]): `dt == QS2CX_WH ? 2u : 4u`
into the desc and `w_bits` into the call. The change is ≈ 10 lines mirrored into their
layer (plan 260 §7's rule 1 — "their prefill unchanged" — is kept: nothing in the prefill
schedule moves, only the dtype admission).

**Already live for 2 bits, no change** (what #229 S1 left and the two re-merges kept):
- backend slot / miss / register: `takeExpertSlot` sizes by `w_bits`
  (`htp_compute_ops.cpp:6620-6621`, `expertStride` `:6658`, `codeBytes` `:6665`,
  `paletteBytes` `:6671`); `readWeight` reads the palette after the codes `:7087-7092`;
  `registerFromArena` picks `weight_register_u2i4_arena` on a non-empty palette
  `:7625-7636`; the swap batch `weight_swap_batch_u2i4_arena` `:6848`, `:6948-6978`;
  `poolAnswer` carries `bits` and the palettes in the answer `:3075-3076`
  (`htp_dspq_wire.h:142`); the #271 miss readers read whole weights of whatever length
  (`readExpert(use_pool=false)` `:6678-6697`);
- DSP M = 1 (the token): `moe_all_2bit` `hexkl_mm_u8i4_moe.c:694`, `native_i2` `:1866`,
  half-size slabs `:1562-1563`, `hvx_gemm_u8i2_wh_col*` `:933-955`, `:1032-1040`, the
  per-expert LUT `:1881-1883`;
- DSP M > 1 (#4415's prefill kernel, theirs + ours additive per PR #275): 2-bit chunks land
  packed in the top half of the int4 slot `:159-179`, expanded on the background lane
  (`moe_expand_submit` `:216-253`, `exp_bg` `:2207-2209`, waited at `:2289-2296`) or in
  front of the batch (`hexkl_moe_expand_chunk` `:263-281`). At the Gemma shape
  `acc_tiles` = min(fits, min(44, 32)) = 32 (`:1413-1431`) → `gu_chunks` = ⌈22/16⌉ = 2,
  `dn_chunks` = ⌈88/32⌉ = 3, both ≤ 8 → `exp_bg` on [G from C]; the host
  `moe_layer_host_check` M > 1 2-bit cell is the proof;
- IDL `test/htp/nntr_hvx.idl`: `expand_i2i4` `:1285`, `weight_register_u2i4_arena`,
  `weight_swap_batch_u2i4_arena` ("our three 2-bit entries, last") — **no IDL change, no
  stub regeneration, the skel changes only if a DSP source does (none planned)**;
- quantizer: `--moe_dtype QS2CX_WH` `quantize_stream.cpp:382-404`, the writer `:1043-1101`,
  `writeGemma4Moe` honours `quant.moe_dtype` `:1364-1376`; loader: `transformer.cpp:594-598`
  maps the dtype to `w_bits` (non-virtual path), `neuralnet.cpp:1110` admits the dtype;
  `QS2CX_WH_Tensor` `qs4cx_tensor.h:334`; config: `moe_layer_dtype` →
  `MOE_LAYER_DTYPE` (`lfm2_moe_causallm.cpp:54`; Gemma's `createMoe` takes the same key
  through #4415's `MOE_LAYER_DTYPE`, plan 260 §7.1);
- `NNTR_HTP_PROFILE` stage tables / `tools/htp_fc_report.py`: the `expand` stage
  (`HEXKL_PROBE_EXPAND`) exists since S1; the #272 wall line per kind is unchanged.

**Tools:** `tools/htp/repack_int2_to_qs2cx.py` + `repack_int2_check.cc` (case (a), one dict
entry); a new `tools/htp/repack_wh4_to_qs2cx.py` only for case (c1);
`tools/moe_expert_cache_sim.py` (unchanged; its numbers are §3.4's).

**Fixture:** `run_inproc_e2e.sh:416-432` quantizes the gemma hd64 fixture as
`--fc_dtype QS4CX --moe_dtype QS4CX_WH`; the 2-bit twin is the same `$Q` call with
`--moe_dtype QS2CX_WH` and the palette twin with `--moe_palette_g max` (the lfm25 pattern
`:433-443`); `two_bit()` `:666-695` is reused as is. The generator
(`generate_gemma4_moe_hd64_reference.py`) is untouched: ternary is a property of the
real checkpoint, the fixture only has to be *representable* (its random weights get a
4-level palette per tensor, which is what the lfm25 line already proves).

## 3. Design

### 3.1 Levers, ranked by ms per day

Baseline = the current "now" cells on the 4-bit file, `htp_decode` @ `4c3953bb1`-class
builds (#271 + #272) [M]: p512 G512 **≈ 5.06 tok/s** (S2 cell 4.98 with the 3.2 ms guess
on; wait 115.7, DSP non-wait ≈ 70, ARM + transport ≈ 12), p1024 G512 **4.13** (S1 B: wait
142.2, non-wait 92 before #272's −11 ms → ≈ 81). Expert bytes: 4-bit 2.93 MiB slot /
2.87 MiB file, 2-bit 1.453 MiB / 1.45 MiB. In-app miss rate 2.41–2.43 GiB/s [M, S1 B];
MoE net 24.0–24.5 ms = 722 MB at ≈ 30 GB/s [M, #272 L0]; P4's 2-bit dummy read its MoE
net at 10.4 ms = 365 MB at 35 GB/s on this unit [M-P4].

| # | lever | before → after (p512 G512 / p1024 G512 unless stated) | gain | days | **ms / day** | gate |
|---|---|---|---|---|---|---|
| **L1** | **the file itself on the pool at C = 16** (their layer admits `QS2CX_WH`; everything below it exists): miss bytes 2.93 → 1.45 MiB, slab DMA halved | wait 115.7 → **≈ 56–60** / 142.2 → **≈ 71–76** [G at 2.41 GiB/s; −6 % for the smaller requests per the probe's 1 MiB cells]; MoE net 24.1 → **≈ 11–12** [G/M-P4]; token ≈ 198 → **≈ 125 ms = 8.0 tok/s** / 242 → **≈ 147 = 6.8** | **−73 / −95 ms** | 1 (layer + fixture + lines) + 0.5 (intake) | **≈ 50–60** | §1 host lines; device: `misses/token` == 94.96 / 120.8 at C 16, wait and MoE net ≈ halved, text / PPL vs A |
| **L2** | **C = 32 at the same arena** (960 slots × 1.453 = 1 395 MiB ≈ today's 1 408; `moe_cache_experts 32`) | misses/token 94.96 → **46.8** / 120.8 → **60.0** [S]; wait ≈ 56 → **≈ 28** / 71 → **≈ 35**; token ≈ 125 → **≈ 97 ms = 10.3 tok/s** / 147 → **≈ 111 = 9.0**; prefill reads 3 360 → 2 880 experts (−0.24 s of flash at 2 bits, hidden where compute-bound) | **−28 / −36 ms** | 0 (a config key) + the sitting | **≈ 30** (sitting-bound) | loads with the 4096 prompt (§3.4); `misses/token` == the sim's 46.8 / 60.0 ± 1 %; handles 1 920 + ≈ 450 < 4 096 |
| L3 | **prefill at 1024 on the 2-bit flash floor** (no code: the HMX chunk expansion is live) | flash per prefill 3 360 × 3.007 MB = 10.1 GB → 5.11 GB: floor 3.37 → **1.70 s**; prefill at 1024 = max(flash, compute) + ≈ 0.25 [M, doc 57 §9.35]: on upstream's unit 3.40–3.47 → **≈ 2.9–3.0 s** (compute ≈ 2.8 + expansion ≤ 0.1); on ours the revision-3 sitting's number − its exposed wait (`260 r2` read 5.36 s at 1024 on `41bd65e04`, before §9.28–9.35; the rev-3 cell is the A of this plan) | **≈ −0.5 s at 1024 [E on their compute]; ≈ 0 at 2048 / 4096** (compute-bound already: ≈ 5.6+ / 11+ s vs 3.37 of flash; the 2-bit file only removes the +0.1 s it also adds) | 0 | — | prefill ≥ −5 % of the 4-bit A at each prompt; `prefetch exposed wait` ≈ 0 at 1024 |
| L4 | **#267 L3 miss-path batching** (present experts as one call, arrived as one) — unchanged ask, re-ranked: fewer single-expert calls matter more when each is half the bytes | MoE net ≈ 11 → ≈ 8 [E, scaled from plan 267's −4.5…−8 at 4 bits] | −3…−4 | 2 | ≈ 2 | plan 267 §6 S2 |
| L5 | the 4-slab M = 1 ring at 2 bits (plan 229 §3.3: 3.96 MB fits) | ≈ 0…−2 [E] (plan 267 §4.3: the one-queue ring is not starved on resident layers) | ≤ 2 | 2 | ≤ 1 | plan 267 L4's trigger: MoE net rate on miss-free layers < 45 GB/s |

**Order: S0 intake → S1 (L1 host) → S2 sitting (L1 + L2 + L3 in one grid) → L4 / L5 by
their own plans.** L1 and L2 together: **p512 G512 ≈ 5 → ≈ 10 tok/s, p1024 ≈ 4 → ≈ 9** [G];
the remaining ≈ 70–80 ms of a token is the non-weight wall plus FC / DENSE / head bytes
— plan 229 §8 (2-bit FCs, S2, parked) and plan 261's kernels, not this file.

### 3.2 Prefill: where the 2-bit experts enter #4415's HMX path (ask 3)

| option | where the int4 tiles come from | cost per prefill [G/E] | risk to #4415's prefill | verdict |
|---|---|---|---|---|
| **(i) DSP, at chunk-landing time, in VTCM** — `hexkl_moe_push_weight_chunk` lands the packed half in the slot's top half, `moe_expand_submit` expands it on the background lane one batch ahead (`:159-179`, `:216-253`, `:2207`, `:2289`) [C] | per MoE call every routed expert's chunks are expanded once: 128 experts × 2.97 MB = 380 MB of VTCM writes per call, ≈ 5 vector ops per 256 B → ≈ 7.4 M ops ≈ 3.5 ms of one HVX thread per call, 30 calls → **≈ 0.1 s of worker time per prefill** (+3.8 % of their 94 ms/call worker sum [M, §9.35]); hidden behind HMX when the lane has slack, exposed at worst in full | none to the schedule: the same `exp_bg` slots and waits run on the LFM lines today; the only new thing is the Gemma shape (2 + 3 chunks, §2) | **chosen** — it exists, it keeps the slot 2-bit (so the pool, the miss and the decode GEMV all read half the bytes), and its cost is bounded by the −5 % gate (0.1 of 2.9–3.5 s = 3 %) |
| (ii) ARM reader threads expand into the slot (4-bit image in the arena; DSP kernel unchanged) | ARM NEON table expansion ≈ 1 GB/s a thread [E]: 3 360 × 2.97 MB = 10 GB → 10 s of thread time / 4 readers ≈ 2.5 s beside 1.7 s of flash — on the critical path at 1024; plus the uncached-ION write of 2× the bytes | none to the kernel; but **the slot stays 4-bit**: the pool is 1 408 MiB at C = 16 (no C = 32), the decode miss writes 2.93 MiB per expert again, and the token reads the 4-bit GEMV (MoE net stays 24 ms) — asks 2 and 4 are lost | rejected |
| (iii) a 2-bit HMX path | does not exist; the HMX reads u8 × i4 tiles | — | a new kernel inside their prefill | rejected |

Which bound wins per prompt length (expert reads per prefill are per-layer-complete, so
flash is a per-prefill constant; compute scales with tokens; [M] = doc 57 §9.35 on
`R3CY10WM83Y`, cfgB; our unit's rev-3 cells replace the compute column when read):

| prompt | flash floor 4-bit / 2-bit (C 16) | compute [M on their unit / E] | 4-bit prefill | 2-bit prefill | what the file buys |
|---|---|---|---|---|---|
| 1024 | 3.37 / 1.70 s | ≈ 2.8 s [M] | 3.40–3.47 [M] (flash-bound, exposed wait 0.6) | ≈ 2.9–3.0 (compute + ≤ 0.1 expansion) | **≈ −0.5 s (−14 %)** |
| 2048 | 3.37 / 1.70 | ≈ 5.6–6.5 [E: MoE / FC linear, attention superlinear] | compute-bound | compute + ≤ 0.1 | ≈ 0 (−0.1 to +0.1) |
| 4096 | 3.37 / 1.70 | ≈ 11–14 [E] | compute-bound | compute + ≤ 0.1 | ≈ 0; page cache: 5.8 GB of experts beside 1.3 GB of FCs may stay resident across turns on the 11.1 GB unit (the 4-bit 11.5 GB cannot) — read as `pgpgin` on the second prefill |

So the prefill asks are answered: at 1024 the flash floor wins today and the 2-bit file
removes it; at 2048 / 4096 compute wins on both files and the only prefill lever is
#4415's own kernel work (their §9.35 list: HVX epilogue, staging), not bytes.

### 3.3 Decode: the one-PD token on 2-bit experts (ask 2)

Per token at C = 16 [G from M]: 240 routed experts × 1.52 MB = 365 MB of slab DMA (vs
722); 95 / 121 misses × 1.45 MiB = 138 / 176 MiB from flash (vs 278 / 354). The GEMV is
DDR-bound (plan 229 §3.1: ≈ 25 µs of compute per 340 µs of DMA per LFM call), so the
whole DMA column halves and nothing else moves — the §5.18 signature on LFM (+25.9 %)
was exactly this. Bits: the u8i2 kernels produce the 4-bit path's int32 sums (host
`gemv_native_check` 0 mismatches, device `HmxMmU8I4Layer.MoeLayerU8I2GemvMatchesI4Twin`
12/12 on v79, LEDGER ㊳); on the pool, `2bit pool C=1/2 lfm25 bit_identical=1`
`misses=56/13` passes on the rev-3 base (`266-predict.md` host gates). What the re-merges
broke: (1) the Gemma layer admission (§2) — both prefill and the hand-over; (2) the
`2bit gemma64` inproc lines, dropped in plan 260 §7.3 ("Gemma's file is 4-bit") — restored
in S1 on the #4415 fixture layout; (3) nothing in the backend or the DSP (PR #275 kept the
expansion waits beside their down phase; `weight_swap_batch_u2i4_arena` last in the IDL).

### 3.4 Memory and the C decision (ask 4)

One-PD sum on this unit at C = 16, 4-bit (plan 267 §1 [M]): pool 1 408 + FC arena / head
attach 448 + KV 440 + DSP heap 1 395 = **3 691 MiB** of the 3 840 ceiling; C = 24 at 4 bits
failed `0x80000402` at the second `attn_m1` set (+704 MiB of arena) [M, 260 r2].

| config | pool MiB (slots × 1.453, 64 MiB grain) | one-PD sum | margin to 3 840 | hit % p512 / p1024 [S, LRU] | misses/token | wait ms [G] | token ms / tok/s p512 · p1024 [G] |
|---|---|---|---|---|---|---|---|
| C 16, 2-bit | 704 | 2 987 | 853 | 60.4 / 49.6 | 94.9 / 120.8 | 56 / 71 | 125 → **8.0** · 147 → **6.8** |
| C 24 | 1 046 → 1 088 | 3 371 | 469 | 72.5 / 65.2 | 66.0 / 83.7 | 39 / 49 | 108 → 9.3 · 125 → 8.0 |
| **C 32** | 1 395 → 1 408 | **3 678** (= today's 4-bit C 16 −13) | 162 | **80.6 / 75.0** | **46.8 / 60.0** | **28 / 35** | 97 → **10.3** · 111 → **9.0** |
| C 40 | 1 744 → 1 792 | 4 062 | **−222: does not fit** | 86.0 / 83.0 | 33.6 / 40.8 | 20 / 24 | — |
| for scale, C 48 / 64 (a 2 GB-era question, plan 229 §7) | 2 112 / 2 816 | — | — | 89.7 / 88.5 · 93.8 / 95.5 | 24.6 / 27.6 · 14.7 / 10.8 | | |

(`moe_expert_cache_sim.py --policies lru` on the device route logs; the sim's C = 16
reproduces the app's misses to 0.2 %, `266-miss-readers.md`.) Belady at C = 16 is 79.1 /
74.6 % — **C = 32 under plain LRU reaches what the best possible policy reaches at C = 16**,
which is why it ranks above any policy lever (266 S2 closed lever 3 at these numbers).

**Decision: C = 32, the same arena bytes as today, is the row of record if it loads with
the 4096 prompt; C = 24 is the fallback; C = 16 is the "bank the memory" cell read once
for the ledger.** The 4096 prompt's DSP heap is unmeasured on our tree: the r2 sitting read
+29 MiB of heap from `init_seq_len` 1024 → 2048 (1 395 553 → 1 424 737 KiB), so ≈ +90 MiB
at 4096 [E] against C = 32's 162 MiB margin — tight, which is why the sitting loads the
4096 cell first (§4 S2). The contract's deferred 2 GB stage is not touched: C = 16 at 2
bits (704 MiB) is the number that stage starts from, recorded.

### 3.5 Contract §2 and doc 45 §3

Three walls untouched (the GEMV, the ring, the transport are reused at half the bytes); the
arena budget read per cell with the C ladder; no CPU fallback for `QS2CX_WH` (the throw
stays; the reference is the hybrid A, Q3). Activation handles, DMA behind compute (the
compute : DMA ratio at 2 bits is ≈ 25 : 170 µs per call — hidden with margin), `_det`
before every quantizer (nothing before a quantizer changes), bit-identical gates on the
host where a 4-bit twin exists, text + PPL where it does not (one file on the device).
Rejected alternative to the whole plan: converting the ternary file to 4-bit `QS4CX_WH`
(the user's rule of 2026-10-08: format repacks only, no dtype change; and it would keep
every byte count above).

## 4. Steps (each ends in a rung of `.claude/skills/hexagon-gates`)

* **S0 — intake (0.5 d, host; starts when the file lands; the user's Q1–Q2 pick the row
  of §0).** Case (a): `FC_BYTES["QS4CX"]` in `repack_int2_to_qs2cx.py`, run it, keep
  `--verify-only`'s output; (b): the four checks as a 40-line `tools/htp/qs2cx_file_check.py`
  (size arithmetic, palette, code histogram, colsum walk — the #250 checker's loop over
  `QS2CX_WH_Tensor` offsets); (c1): the sibling repack tool. Gate rung 0 (a tool), the §0
  checks in `docs/measurements/276-intake.md`, md5 posted on #276.
* **S1 — their layer admits `QS2CX_WH`; the gemma64 2-bit lines (1 d, host).**
  `lfm2_moe_layer.cpp:983-997` `expertDesc` → `dt ∈ {QS4CX_WH, QS2CX_WH}`, `w_bits = dt ==
  QS2CX_WH ? 2 : 4` into the desc; `:1160-1163` `weights_wh` → the three-way `w_bits` as
  `lfm2_moe_pool_layer.cpp:1040-1045` (`:1182`'s gate reads `w_bits == 0`, `:1189`'s
  `want` picks the dtype; `:1313` passes `w_bits`). `run_inproc_e2e.sh`: `g64htp2`
  (`--moe_dtype QS2CX_WH --fc_dtype QS4CX --embd_dtype Q4_0`) and `g64htpp` (the palette
  twin, `--moe_palette_g max`); lines `E2E gemma64 off 2bit == palette-twin
  bit_identical=1` (prefill through their LRU + the HMX expansion — the prefill proof),
  `E2E 2bit gemma64 e3 == palette-twin bit_identical=1 calls/token=1.00`, `E2E 2bit pool
  C=2 / C=3 gemma64 == 2bit e3 bit_identical=1 misses=<n>` (the decode proof), `E2E tokens
  2bit==twin-gemma64 8/8`. Gate rung 1 (every existing line unchanged incl. the lfm25
  2-bit lines and `*Lfm2Moe*` 7/7; `moe_layer_host_check` M > 1 2-bit cell at the Gemma
  chunk counts 2 / 3 `bitwise mismatches=0`); **rung 2 not needed** (no DSP source, no
  IDL: the device keeps the revision-3 skel; `git diff -- test/htp nntrainer/tensor/htp_backend`
  empty is the check); rung 3 (`build_android.sh --htp --cache`), md5s.
* **S2 — the device sitting (unavoidable; S25 `R3CY205ZMND`, agent adb, `sitting_lock.sh`,
  cool start per block, A first and last; ≈ 1 d).** Config = the revision-3 4-bit sitting's
  with `moe_layer_dtype QS2CX_WH` and the 2-bit file; prompts 1024 / 2048 / 4096
  (`260-prompt{1024,2048,4096}.txt`) for prefill, p512 / p1024 G512 for the decode cells
  beside the #266 / #267 rows (G 64 / 1024 at the C of record only). Variants ≤ 4:

  | variant | what | reads |
  |---|---|---|
  | **A** | 4-bit file, revision-3 build, C 16, hybrid and E — the revision-3 sitting's own cells, re-run once here for drift (A first / last) | prefill 1024 / 2048 / 4096, E p512 / p1024 G512, text, nll 4.573 / 3.618 |
  | **T16** | 2-bit file, C 16, hybrid (A') and E | §1 gates; wait / MoE net halved at the same `misses/token`; prefill vs A per prompt; nll vs A' on the same file; arena 704 |
  | **T32** | 2-bit, C 32, E (and the hybrid at 4096 once: their LRU at 32 reads 2 880 experts) — **loaded first with the 4096 prompt**; on `0x80000402` / arena failure → C 24 and the T24 column replaces this one | `misses/token` 46.8 / 60.0, wait, tok/s, `mapped` ≈ 1 408, S1 ceiling, heap at 4096 |
  | T16-PPL / T32-PPL | `NNTR_PPL=1` + `NNTR_PPL_DECODE` cells per prompt at the C of record, E vs A' | PPL ≤ 1.02 × A', loop onset, text to the first `<turn\|>` |

  Order: A 1024 G64 → T32 4096 G64 (the load question) → T32 / T16 p512 / p1024 G512 →
  prefill 1024 / 2048 / 4096 for A, T16, T32 → PPL cells → A again. Handoff
  `docs/measurements/276-2bit-rev3.md` with md5s, the four §0 checks, every column of §1.
* **S3 — docs + PR (0.5 d):** §6; the S1 PR into `htp_decode` carries the layer change, the
  lines and the tool entry; the sitting doc rides it or its own docs PR.
* L4 / L5 stay plans 267 / 229's steps, re-read on the 2-bit rows.

## 5. Risks

* **Flash rate at smaller requests**: the in-app 2.41 GiB/s was read on 1.99 / 1.01 MiB
  whole-weight jobs; at 1.0 / 0.5 MiB the probe's 1 MiB cells read −6 % and the per-round
  ARM overhead (`arm_ms/round` 4.4 ms) is unchanged in bytes, so the wait may land at
  60–65 rather than 56 ms. The handoff's `implied GiB/s` and `arm_ms/round` columns show
  which; the `misses/token` column separates it from a routing change.
* **Thermal / DVFS between sittings**: the 4-bit rows of record come from three sittings
  (260 r2, 266, 267) on different days; A is re-run in this sitting first and last, every
  verdict is same-sitting, and the 2-bit gain is a byte count — a G-dependent or
  prompt-dependent gain is a thermal artefact (plan 229 §5).
* **Stale skel**: none expected (S1 changes no DSP source); the handoff records the skel
  md5 == the revision-3 sitting's. A device that answers `AEE_EBADITEM` on the first 2-bit
  register means an older skel without the u2i4 entries (rule 67).
* **Address space at C = 32 with the 4096 prompt**: 162 MiB of margin against an
  unmeasured heap term — the load order in S2 makes it the first thing read; the ladder is
  C 32 → 24 → 16, and the row of record is whichever loads at every prompt.
* **Host vs device**: the host proves bits (the two gemma64 2-bit lines) and the chunk
  schedule (the scoreboard), never the expansion's ms on the lane or the page-cache state
  — the prefill −5 % gate and `prefetch exposed wait` at 1024 are the device's.
* **The file**: (a)'s layout assumption (rows moved within a column pass the colsum check)
  is caught only by the text / PPL gate; a per-group-scale or base-3 file (c2) stops the
  plan at S0 — the §0 table says what each answer changes.
* **Accuracy on a different quantization**: the 2-bit file's nll is not the 4-bit file's;
  a loop that the 4-bit A did not have at the same prompt is read under rule 45 against
  A' (the hybrid on the same file), not against the 4-bit rows.

## 6. Docs to update

* `docs/htp_moe/BENCHMARK.md`: Gemma goal row — the T16 / T32 cells (prefill 1024 / 2048 /
  4096; decode p512 / p1024 at G 64 / 512 / 1024) beside the revision-3 4-bit cells, with
  `misses/token`, miss wait, MoE net (wall line), arena / mapped / heap / peak RSS, nll
  and the §0 md5; Method paragraph: "the 2-bit file is a different quantization — speed
  rows beside the 4-bit rows, accuracy against the same file's hybrid A".
* `docs/htp_moe/LEDGER.md`: §2 verdict row for #276; rule candidates — "at M = 1 the
  2-bit experts buy the halved DMA and miss columns and nothing else (the §5.18
  signature on Gemma)"; "C = 32 at 2 bits = C = 16's arena: LRU 80.6 / 75.0 % — Belady's
  C = 16 ceiling"; "prefill: flash-bound at 1024 on the 4-bit file, compute-bound from 2048
  on either file"; ㊳ (v81 unverified) unchanged; open item: 2-bit FCs (plan 229 §8, S2
  parked) as the next byte lever, lm_head / embedding ternary question.
* Plan 229 §4 S2 / §8: pointed at this plan's S1 (the Gemma layer admission replaces the
  "Gemma at 2 bits on the host" bullet); plan 267 L4 / L3 re-ranked on the 2-bit rows;
  plan 266 §0.2's floor table gains the 1.45 MiB rows.
* Contract `0001` §1 Model row: the file's case and md5 (§0, dated); §12: the C decision
  (Q4) and the accuracy-reference decision (Q3).
