# 201 — `htp_decode`: the decode NPU end-to-end path with FSU in the tree, re-targeted at the Gemma MoE model

Issue: dlwlzzero/nntrainer#201 (p0). Base `htp_decode` @ `0aacbce24`
(= `htp_first_version` @ `d4a898430` + one docs commit). Wherever the
contract says `htp_moe`, read `htp_decode` (contract §12, 2026-09-30).

**Status: draft, waiting for the user (three blocking questions, §0).** No
source change belongs to this issue until the plan is accepted (issue text).

Evidence classes used below, on every claim that matters:
**[C]** code read on this tree, `path:line`; **[H]** run on the workstation
host in this planning session; **[M-LFM]** measured on silicon on
LFM2.5-8B-A1B, *before* FSU and #4343 were merged (doc cited); **[M-up]**
measured by the upstream author on another unit / tree (docs 52 / 53);
**[R]** a reference read at its source; **[U]** recalled, not verified
here; **[G]** arithmetic on unverified inputs or a guess.

## 0. Decisions received, and what still blocks

Received from the user through the orchestrator (2026-09-30, after the
issue was filed):

| # | decision |
|---|---|
| D1 | Speed: as fast as possible; the baseline bar is **50 tok/s** decode |
| D2 | Accuracy: **not bit-preserving**. Slight wording differences against the reference are fine; **sentence-repetition loops or off-context answers fail**. Pointers: the earlier norm optimization failed the bar — if that arithmetic is floating point, consider integer arithmetic; check how QNN computes it |
| D3 | Device: **S25 (v79) only**. S26 / v81 is out of this plan |
| D4 | Model: the reference changes from LFM2.5-8B-A1B to **"Gemma moe (30b)"**, weight data types unchanged |

**Blocking (implementation cannot start past S0 without these):**

| # | question | what was found | proposed default |
|---|---|---|---|
| **B1** | **Which model is "Gemma moe (30b)"?** | The only Gemma MoE this tree and its upstream know is **`google/gemma-4-26B-A4B`**: doc `docs/htp_attention/54_gemma4_moe_htp_handoff.md:11` names it, upstream PR nntrainer/nntrainer#4296 (OPEN, changes requested, last push 2026-09-30) implements it for the CPU ("quantized Gemma4-26B-A4B"). **No "30b" exists in the tree, the PR or on disk** [C][R]. The in-tree `Applications/CausalLM/models/gemma4/` is the dense small model (`config.json`: `enable_moe_block: false`, hidden 1536, 35 layers) [C]. **No Gemma weights or config of the MoE model are on this workstation**: `/local/mnt/workspace/models/` holds `lfm2.5-8b-a1b` and `qwen3-0.6b` only; the HF cache holds LFM only [H] | `google/gemma-4-26B-A4B`; the user confirms and places the HF files (download is a user action) |
| **B2** | **50 tok/s against the bytes this model reads a token.** | "A4B" = ≈ 4 B active parameters a token. At these dtypes (int4 `QS4CX_WH` 0.5 B/weight, `Q4_0` 0.5625) that is **≈ 2.0–2.2 GB of weights a token** [G, from the name and an unverified config, §3.2]; the phone's DRAM tops out at ≈ 70 GB/s for any reader mix (LEDGER rule 44) [M-LFM] → **≥ 29–31 ms a token, ≤ 32–35 tok/s with zero compute and every weight in DRAM**. LFM read 886 MB (rule 42) — that is why 50 was reachable there | The bar is restated per model after S1's table (config in hand): the floor `bytes / 70 GB/s` is the ceiling, the target a fraction of it. 50 stays the bar only if the user names how bytes a token fall (fewer bits, a smaller lm_head read, …) — that would be contract §3.3's separate track |
| **B3** | **Memory budget (RAM the app may take).** | The experts alone are ≈ 10.6 GiB in `QS4CX_WH` [G]; one cDSP PD maps 3840 MiB (rule 8), two ≈ 7.4 GiB (rule 50); the phone has 12 GB. **FSU is not optional for this model**, and decode speed is set by the pool size C (misses come from flash at 3.0 GB/s, doc 53 §5.5 [M-up]) | One PD's arena (≤ ≈ 3.5 GiB of experts) first; the user picks the number from S1's table (doc 54 §8 already ends Phase 0 with this question) |

**Still open, with a safe default (do not block S0–S3):**

| issue Q | narrowed to | default until the user says otherwise |
|---|---|---|
| Q3 FSU on / off, control | With Gemma, **on by necessity** (B3). The control is no longer "FSU off" but: CPU run with the same pool size (PR #4296's `moe_cache_size`) and, once it exists, the hybrid (MoE on the HTP through the FSU pool, the rest on the CPU) | FSU on at the agreed C; CPU cached run = the text / PPL reference; the LFM resident path stays the regression fixture only |
| Q4 prefill gate | LFM's "≥ 497 tok/s" is an LFM number. The *rule* (no lever costs more than 5 % of the same sitting's A prefill) can carry over once a Gemma HTP prefill exists; with FSU the prefill is flash-bound when cold (doc 53 §5.6) | Record warm and cold prefill in every sitting, gate at −5 % of that sitting's A from the first sitting that has an HTP prefill; no absolute number |
| Q6 port #194's L0 / L2–L4 | L0 (ARM wake, walk) and L4 are transport levers and model-independent; L2 / L3 are LFM kernels (router, norms) | Not ported now; L0's reading transfers as a fact (§3.3), its code is re-derived when the Gemma token entry exists |
| Q7 mapping loss 3840 → 3584 | With an FSU pool below ≈ 3.5 GiB the 15th 256 MiB window is never needed, so the defect cannot stop a load [C, §2.3]; it stays unexplained | Accepted with the ceiling cell after every run; not a blocker |
| Q8 what "reference" means | §3.5: structure only for all three; no QNN-built graph, no LUT kernel | — |
| Q5 device | answered (D3) | — |

Q1 and Q2 are answered by D1 / D2 (Q1 re-opened as B2).

## 1. Goal and gate

**Acceptance (issue, re-scoped by D1–D4).** Deliverable 1 of the issue is
this file: the structure review (§2), what is and is not measured (§3.1,
§3.3), the ranked levers (§3.4). Deliverables 2 and 3 move to the Gemma
model: the first **measured per-stage breakdown on the S25** is S4's
handoff; the lever table of §3.4 is rewritten from it.

Made measurable, per step:

| gate | where | pass |
|---|---|---|
| model facts | S1's table in `docs/htp_moe/BENCHMARK.md` (new "Gemma MoE" block): layers, hidden, experts, top-k, expert bytes, bytes a token, pool-size table, flash floor | every cell from `config.json` / the quantized file, none recalled |
| speed | `decode tok/s` line, prompt 512, G 64 / 512 / 1024, cool start, against the same sitting's A | reported against **the byte floor of S1's table** (B2); ≥ 50 only if B2 is answered that way |
| accuracy (D2) | plan 194 §1's readings, re-used as the measurable form of D2: decode PPL forced on A's continuation (`NNTR_PPL_DECODE`, 8 prompts, G = 256) pooled ≤ 1.02 × A and ≤ 1.05 per prompt; **no new loop** (`tools/htp/loop_check.py --prompt`) on a prompt where A has none; **text approval by the user** for "off-context" (the pasted texts; an unapproved text is not a pass) | all three; a differing text alone is not a fail |
| kernel | every new DSP kernel ships a scalar spec in `test/htp/host/` and a host check holding the kernel to it bit for bit, plus an SNR line against the f32 reference (plan 194 §3.3's reading of doc 45 §3.3 for a non-bit-preserving track) | `ALL CHECKS PASS` |
| standing | prefill: Q4's default; `QS4CX_WH` has no CPU fallback (unchanged); S1 ceiling cell after every run; LFM host gates (`*Lfm2Moe*` 6/6, `INPROC E2E PASS`) stay green — they are the only E2E regression fixture | — |

Known weak spots of the accuracy readings, carried from
`docs/measurements/194-sitting-1.md` [M-LFM]: on LFM the baseline A itself
loops on 5 of 8 prompts at G = 256 and the detector missed a sixth by 0.06
(p04, Korean, 100 words); mc-40 sat at chance for A. Whether Gemma's A
loops is S4's first accuracy reading — the gate is only as good as that.

## 2. Where it lives (structure review, as the code is)

### 2.1 One decode token on the E2E path (`NNTR_HTP_E2E=1`, LFM) [C]

```
ARM  causal_lm.cpp:780-835      loop: incremental_inference -> 228-node walk, every layer's hook
     rms_norm.cpp:85            op 0's hook keeps the row (htp_compute_ops.cpp:2025 runStretchOp)
     conv_block_layer.cpp:232, qkv_layer.cpp:255   FC GEMVs skipped (decode_row_resident :2093)
     lfm2_moe_layer.cpp:1199    router hook returns 1 -> the layer skips router, top-k, experts, LRU
     tie_word_embedding.cpp:523 lm_head hook = last op -> invokeForward :3204 -> tokenForward :3894
       :3951 write S1 packet, :3956 write S2 packet (row, 8 KiB); :3964 blocking read S2, :3967 S1
       causal_lm.cpp:361        the id comes back (take_decode_token_id :4062), no logits
S2   nntr_hvx_dspq.c:136 dspq_token -> nntr_hvx_token.c:125 -> hexkl_token.c:195 hexkl_token_main
       per MoE layer: hexkl_graph_forward (hexkl_graph.c:663) on its stretch, tk_post ping (:104),
       tk_take pong (:122: spin 0, then qurt_timer_sleep 20 us, hexkl_token.h:76)
S1   hexkl_token.c:271 hexkl_token_serve: 22 x {tk_take ping, [ROUTER_TOPK, MOE], tk_post pong}
```

One dspqueue packet per session per token, 44 mailbox hops, the two
sessions strictly alternate (each parks its pool, `hexkl_token.c:245,311`).
Per-op pcycles are bracketed in `hexkl_graph_forward` (`:718-720`), summed
per kind per side and printed at close (`htp_compute_ops.cpp:3731-3776`);
the L0 lines split the ARM side (`:3787-3811`).

### 2.2 Where the weights live (LFM) [C]

| what | where | how it is read |
|---|---|---|
| 1408 expert weights, 3696 MiB `QS4CX_WH` | S1: ION arena, uncached, 256 MiB chunks (`htp_compute_ops.cpp:5313`, `:5493`), 15 chunks at the 3840 MiB ceiling | M=1 HVX GEMV fed from VTCM by DMA, `src_bypass=1` (rule 43) |
| router weights 22 × [2048][32] f32 | S1 heap (`hexkl_graph.c:589-605`) | scalar `sffma` chains |
| 67 Q4_0 weights → 74 `Q4M1` handles, 383.6 MiB (448 mapped) | S2: its own ION arena (`registerQ4m1` `:1722`, `placeOn(e.h2, e.dom2, …)` `:1735`, `e2ePlaceFc` `:4085`), placed at load after S1's arena (rule 54; `lfm2_moe_causallm.cpp:223-231`) | per-lane DMA into a 2 MiB L2 scratch (`nntr_hvx_fc_q4.c:95,342-351`): **S2 has no VTCM** (S1's `hexkl_micro_hw_init` holds the 8 MiB; `hvx_add_f32.c:55-68,137-146`) |
| KV cache, gammas, conv state, RoPE table | S2 heap (≈ 30 MiB) | — |
| the CPU's own copies of the FC set | ARM heap, still loaded | unused at decode (skipped) |

### 2.3 What FSU (#4383) is in this tree, and what it does to the E2E path

* **Off by default.** `NNTR_MOE_CACHE_EXPERTS` unset → `experts_virtual`
  false (`lfm2_moe_layer.cpp:231-242,302-307`); the resident path registers
  exactly as before (`get_or_register_wh`, `:5275`). What the merge changed
  for a resident run: the DSP weight slot's three N-sized arrays are one
  allocation (`hexkl_mm_u8i4_dma.c:161-180`), the IDL grew two swap
  entries (skel **and** ARM stub must be rebuilt), reader threads exist
  but start only on a prefetch (`:5117`). None of it is on the token's hot
  path. [C] **Not measured**: no device number of A or E exists on this
  tree with a handoff file (the G = 64 reading in the issue is hearsay).
* **On (C > 0): the ARM owns the pool.** Virtual tensors carry file
  offsets (`transformer.cpp:513-550` `preloadExperts`); each layer call
  makes its routed experts resident through one LRU shared by all layers
  (`tryMoeLayerOnAccelerator`, `lfm2_moe_layer.cpp:790-815`), a miss is
  `pread` into an arena slot (`readWeight` `:5172`) and one swap RPC
  (`:5044-5113`); the DSP rebinds the retired handle **in place, keeping
  its number** (`nntr_hvx_mm_u8i4.c:546-572`,
  `hexkl_mm_u8i4_dma.c:297-321`). Arena = `22 × C` slots of 5.29 MiB
  (`takeExpertSlot` `:4920-4962`).
* **FSU and the E2E path cannot run together today.** (1) At decode the
  router hook returns before the LRU is consulted
  (`lfm2_moe_layer.cpp:1199-1206`), and the router runs on S1 — the ARM
  never learns which experts a token needs. (2) The graph's MoE op holds a
  static `h_gu[32]` / `h_dn[32]` copied once at the layer's first call
  (`bindMoeOp` `:2131-2160`) and read every token (`hexkl_graph.c:104-107`);
  after an eviction the same handle number names another expert's bytes.
  (3) There is **no guard**: the run is refused only by accident, because
  the virtual path compacts the expert array and `bindMoeOp`'s size check
  trips. **[H]** `htp_e2e_test` on the lfm25 fixture (4 experts, top-2,
  prompt 512): `NNTR_HTP_E2E=1` alone runs (`calls/token=1.00`,
  `hops/token=8.00`); with `NNTR_MOE_CACHE_EXPERTS=1 | 2 | 4` it dies with
  `set_decode_graph_desc: MoE op 54 expects experts=4 …, the layer has 3`
  (or 2). On the real model a 512-token prefill activates all 32 experts
  of every layer, the check passes, and decode would read stale handles —
  the "plausible wrong text" class (rule 6). [C, not run on the 8B]
* **Address space.** FSU off: S1 3696 MiB in 15 chunks of a 3840 MiB
  ceiling, S2 448 MiB of its own 4 GiB (rules 8, 50, 54); the per-boot loss
  of one 256 MiB window (nine `LEAK` stops in #194 sitting 1b) stops a load
  because the 15th window is needed. FSU on: the arena is the pool, e.g.
  C = 16 → 1920 MiB (doc 53 §3.2 [M-up]); below 14 windows the loss cannot
  stop a load. The cause of the loss is still unknown.
* **VTCM, feed schedule**: unchanged by FSU (the swap touches the weight
  table only; the M=1 feed reads whatever bytes the handle points at).

### 2.4 What in the E2E path is LFM-specific [C]

| piece | LFM assumption | Gemma-4 MoE (PR #4296's graph; real sizes: S1) |
|---|---|---|
| graph builder | `htp_graph_lfm2_build` (`htp_graph_desc.h:594`), `HTP_GRAPH_MAX_OPS 256` (LFM is 228) | own builder; ≈ 25 ops a layer × the layer count exceeds 256 |
| experts | `HTP_GRAPH_MAX_EXPERTS 32` (`:51`): the op record's handle arrays, the router's `[K][32]` padding | more than 32 a layer (doc 54 R2) |
| handles | `HEXKL_MM_U8I4_MAX_WEIGHTS 2048` → ≤ 1024 resident experts a PD; `NNTR_HVX_Q4M1_SLOTS 80` | pool and FC set both larger |
| MoE kernel | fused `[gate | up]`, **SwiGLU** epilogue (HMX prefill and the M=1 GEMV) | separate gate / up in PR #4296, **GELU-tanh(gate) × up**: no kernel (doc 54 R1) |
| router | sigmoid + expert bias, top-k (`hvx_m1_ops_f32.c:238-300`) | RMS-norm (no gamma) × per-feature scale × 1/√H → dot → softmax → top-k → renormalise × per-expert scale, fed by the un-normed stream (second input): no kernel |
| norm | `hvx_rmsnorm_f32` returns without computing unless the width is a power of two (`hvx_m1_ops_f32.c:64-67`); 49 + 6 norms a token | hidden not a power of two; ≈ 10 norms a layer (attention, q / k / **v**, post-attention, five in the FFN block, the router's own) |
| attention | `ROPE` / `ATTN_M1` admit head_dim 64 only (`htp_graph_desc.h:497`), full causal | head_dim 256 / global 512, sliding window, proportional RoPE with a partial rotary factor, K = V in full layers: no kernel (doc 54 R7) |
| conv kinds | `CONV1D_GATE`, conv in / out proj | none (all layers attention) |
| dense FFN | `DENSE_FFN` = SwiGLU, layers 0–1 only | GeGLU, **in every layer beside the MoE block**, two branches added then normed (`HTP_GRAPH_N_SLOTS 3` is too few) |
| head | vocab 128 000, 8 slices; argmax | vocab 262 144; `final_logit_softcapping` (argmax-invariant, needed for logits / PPL); embedding × √H |
| "every kind resident" | the FC kinds are resident only when every kind is (`:1541-1551`) | false until attention has a kernel → **no one-call-per-token entry for Gemma at first**; it starts as the hybrid |
| token driver × FSU | §2.3: mutually exclusive | FSU is mandatory → a miss protocol (or the router on the ARM) is a precondition |
| host fixtures, gtests | `lfm2_moe_tiny{,_hd64,_lfm25}`, `*Lfm2Moe*`, `run_inproc_e2e.sh` | PR #4296 brings `gemma4_moe_tiny` for the CPU only |
| weights at load | `lfm2_moe_causallm.cpp:154-231` | Gemma4MoECausalLM has no HTP hand-over |

**Consumers that move with a changed contract** (S0 touches none of them;
S5 touches all): IDL `test/htp/nntr_hvx.idl` + `generate_stub.sh` + the
skel (`test/htp/build.sh`) — the generated ARM stub in this checkout was
older than the IDL and had to be regenerated before the host twin would
configure [H]; `HtpComputeOps` (`htp_compute_ops.cpp`: pool, swap, graph
binding); the quantizer (`nntr_quantize_stream`: a Gemma MoE `QS4CX_WH`
writer — same tile layout, so the format tag changes only if gate / up
stay unfused) and the loader check; `NNTR_HTP_PROFILE` stage tables (a
GeGLU stage name) and `tools/htp_fc_report.py`; `tools/moe_expert_cache_sim.py`
+ a routing trace for the Gemma layer (`NNTR_MOE_TRACE` exists only in
`lfm2_moe_layer.cpp:220`).

## 3. Design

### 3.1 The chosen approach

Measure the new model on the path that exists before building any kernel
for it: CPU reference (PR #4296) on the S25 with its own per-stage timers
and a routing trace, then the MoE on the HTP through the FSU pool (doc 54's
phases), and only then rank. The LFM E2E path is **parked as it stands**
(opt-in, bit-identical at mask 0, its host gates kept green); the one
change it gets is the missing FSU guard (S0). Reason: with Gemma the
per-token time is dominated by two terms no LFM lever touches — flash
misses and DRAM bytes (§3.2) — and the E2E entry cannot exist for Gemma
until attention has a kernel (§2.4).

**Rejected alternative:** finish the LFM two-session path first (port L0,
build L5 / L6, §3.3) and carry the design over. Its best projection was
48–49 tok/s on LFM against the hybrid's 52–56 (`194-sitting-1.md`, "Plan
§3.4 re-stated"), it needs the device for every remaining lever, and none
of its kernels (conv, SwiGLU, sigmoid router, head_dim 64 attention) is in
the Gemma graph; D4 makes that work a detour.

### 3.2 The Gemma budget, stated with what is not known

**The config of the MoE model has not been read** (B1: not on disk, not in
the tree). The numbers below are **[U]**: recalled, and consistent with one
line of PR #4296 (`gemma4_moe_layer.cpp`'s "3.3 MB of expert weights" =
3 × 2816 × 704 weights at Q4_0's 18 B / 32) and with the name (26 B total,
4 B active). S1 replaces every cell from `config.json`.

| quantity | formula | value if H = 2816, L = 30, E = 128, k = 8, I_moe = 704, I_dense = 2112, V = 262 144 [U] |
|---|---|---|
| one expert, `QS4CX_WH` | 3·H·I_moe / 2 B (+ 8·N tail, page-rounded slot) | 2.84 MiB (slot 2.87 MiB) |
| all experts | L·E × that | **≈ 10.6 GiB** — 2.9 × one PD's 3840 MiB, ≈ the phone's RAM |
| MoE bytes a token | L·k × one expert | **≈ 714 MB** (LFM: 484) |
| dense FFN a token, Q4_0 | L·3·H·I_dense × 0.5625 | ≈ 301 MB |
| lm_head a token, Q4_0 | V·H × 0.5625 | ≈ 415 MB (the in-tree Gemma config ships it as Q6_K) |
| attention q / k / v / o a token | from the head sizes | unknown; ≈ 0.7 GB if "4 B active" holds [G] |
| **bytes a token** | sum | **≈ 2.0–2.2 GB** [G] → ≥ 29–31 ms at rule 44's 70 GB/s → **≤ 32–35 tok/s** |
| pool in one PD | 3840 MiB / slot; ≤ 1024 experts by the handle table | C ≤ 34 of 128 a layer (≈ 2.9 GiB) as the constants stand |
| a decode miss | one expert from flash at 3.0 GB/s cold [M-up]; warm page cache ≈ 15 GB/s [M-up, scaled] | ≈ 1.0 ms cold, ≈ 0.2 ms warm |
| misses a token | L·k·(1 − hit rate) | 240 uses a token; at LFM's pooled-LRU hit rates (57 % at C = ¼ of the experts, 85 % at ½; doc 53 §5.7) → tens of misses = **tens of ms a token** [G] |

Read plainly: for this model **the flash term, then the DRAM bytes, set the
token time**; the ARM wake (≈ 3 ms), the hops (≈ 0.7 ms) and the FC feed
rate that the LFM work was about are second-order until the first two are
measured. The hit rate is the number nobody has: it needs the model's own
routing trace (S4).

### 3.3 What of the LFM-era measurements transfers

| transfers (platform facts, S25 / v79) | source |
|---|---|
| a PD maps 3840 MiB, heap and mappings share 4 GiB; a second PD has its own 4 GiB, maps 3584 beside the first, gets no VTCM; S1's arena before S2 opens | rules 8, 50, 54 [M-LFM] |
| DRAM ≈ 70 GB/s for any reader mix; weight DMA 57 GB/s in the app with `src_bypass`; CPU Q4_0 GEMV 51–63 GB/s | rules 44, 43, 46 [M-LFM] |
| an ARM blocking read of a DSP answer costs ≈ 2.9–3.4 ms a wake; a bounded ARM spin took it to 0.07 ms; the 228-node walk is ≈ 1.5 ms | `194-sitting-1.md` L0 split and e2 block [M-LFM; e2 is one block at G = 64, informational] |
| a hop waiting with 20 µs sleeps costs ≈ 0.33 ms; a spinning PD steals a hardware thread from the computing one | 1b, rule 56 [M-LFM] |
| flash 3.0 GB/s, miss read and swap-RPC costs, LRU ≈ the best realisable policy | docs 52 / 53 [M-up] |
| a single-session map window for weights is not viable (≈ 1.4 ms a map / unmap pair) | rule 57 [M-LFM] |
| tiny float differences are amplified by the dynamic per-block quantizers; PPL alone does not see loops | rules 39, 45 [M-LFM] |
| v79 traps: asm `.sf` forms, ISS vs silicon, stale skel / stub, thermal drift | rules 3, 52, 53, 58 |

| does not transfer | why |
|---|---|
| every tok/s and ms / token row, the per-kind lines, plan 194 §3.4's ladder | other shapes, other bytes, FSU off |
| A ≈ 52–56, E0 ≈ 31–32, E1 ≈ 33–34 tok/s | LFM; and on this tree unmeasured |
| the accuracy baselines (A's loops, mc-40 score, PPL 1.21) | model property |
| the kernels named in §2.4 | not in the Gemma graph |

### 3.4 Levers, ranked — hypotheses until S4 / S6 read them

| rank | lever | expected size | evidence | blocked by |
|---|---|---|---|---|
| 1 | **Pool size C and where the pool lives** (one PD; a second PD as a second pool — the QNN "sharding" shape, §3.5; RAM budget) | tens of ms a token (the miss count) | [G] §3.2; mechanism [M-up] | B3, the routing trace |
| 2 | **Miss path**: overlap the miss read with the hit experts' compute inside a layer (PR #4296 prefetches one expert ahead; doc 53 §8 lists it as open); batched swap is already in | up to the miss read time of each layer with ≥ 1 hit | [C] PR #4296 `prefetch()`; [M-up] miss = 0.31–0.37 ms read + 1.4 ms / token RPC on LFM | S5 |
| 3 | **Bytes a token on the DSP at the DMA rate** (MoE M=1 GEMV with a GeGLU epilogue; then the dense FFN and projections) | MoE ≈ 714 MB: 12.5 ms at 57 GB/s against the CPU's ≈ 14 at 51 [G] — a small win by itself; the point is prefill and freeing the CPU for the rest | rules 43, 46 [M-LFM] | S5 |
| 4 | **Norm: order-free integer sum of squares** (user pointer; §3.6) | ≈ 300 norm evaluations a token × 16 k pcycles (LFM silicon, scalar chains) ≈ 2–3 ms if resident as it is; a 32-lane integer kernel is ≈ 10× less [G] | rule 47 (cost), rule 45 / #152 (failure), #194 L3 (−0.06 ms on LFM, text unresolved) | a resident norm path for Gemma; host-gateable first |
| 5 | **ARM out of the token** (one call per token, bounded spin on the answer) | −3 to −5 ms a token on LFM | 1b L0 [M-LFM] | attention kernels; the miss protocol (§2.3) |
| 6 | **VTCM-fed FC in a second PD / prefetch across the alternation** | −1 to −2.5 ms on LFM | plan 194 L6, unmeasured [G] | 5 |
| — | guard: refuse `NNTR_MOE_CACHE_EXPERTS` with the per-token entry | correctness, no ms | [C][H] §2.3 | nothing (S0) |
| rejected | hop deadline spin (L4) | 0 net on LFM (MoE +0.55 ms under the spin) | 1b e2 [M-LFM], rule 56 | — |
| rejected | cache policy other than the LRU | +2.4 %p hit on LFM | doc 53 §5.7 [M-up] | re-read on Gemma's trace before closing |

### 3.5 The three references

| reference | what it is (as read) | applies here | already done | ruled out |
|---|---|---|---|---|
| **upstream #4296** (Gemma 4 on the CPU) [R: `gh pr diff`] | `Gemma4MoECausalLM` (dense + sparse FFN a layer), `gemma4_moe` layer (router, separate gate / up / down, LRU of virtual experts, one async prefetch ahead), converter, quantizer key `moe_cache_size`, tiny fixture, timers `NNTR_LAYER_PROFILE` / `NNTR_MOE_PROFILE` / `NNTR_ATTN_PROFILE`, a batched Q4_0 GEMM for M > 1 | **It is the CPU reference of D4** — S2 merges it. Its one-ahead prefetch is lever 2's model; its timers are S4's instrument | the pooled LRU, batched swap, read-ahead (FSU) are richer than its cache; `NNTR_OP_TIME` overlaps `NNTR_LAYER_PROFILE` (one of the two survives the merge in `neuralnet.cpp`) | its batched GEMM is prefill-only; its per-layer cache is weaker than the shared pool (doc 53) |
| **QNN** [R: repo notes `ref_16`, `36`, `33` — a Qwen3-0.6B attention-block optrace, QAIRT 2.47; ExecuTorch's Qualcomm LLM README read on GitHub; **no QNN / QAIRT SDK is installed here** [H]] | one graph execute per token ("KV cache mode"), KV as in / out tensors updated in place, static (calibrated) activation scales, integer data path (`combine_scales`, Q31 multipliers — no float requant), RMSNorm as three integer kernels, **a 4 GB per-context limit answered by sharding the model into several contexts**, weight sharing between prefill and decode graphs | the 4 GB-per-context / sharding shape = our PD limit and a second PD as lever 1; static integer scales = §3.6's second variant | one call per token (LFM E2E), weights pre-shuffled to the HMX layout, DMA double buffering, band tiling instead of spill / fill (doc 36 §2) | a QNN-built graph as the implementation: no dynamic expert streaming in a context binary, SDK absent, and it would replace the whole backend — outside "weight data types unchanged" |
| **T-MAN** [R: doc `53_int_requant_task.md` §4 (kernel source + README read; the paper's PDF was not readable there), README re-read on GitHub now; nothing recalled from the paper] | table-lookup *weight* dequantization for QNN custom ops, int16 static activations, decode on HVX by bit-serial LUT GEMV; gains are for ≤ 2-bit weights and against a baseline that decodes on the CPU | the confirmation that DMA → TCM is the ceiling it also runs at (rule 43) | compute hidden under the DMA at M=1 | a LUT GEMV: no gain at 4 bits where the feed, not compute, bounds the kernel (plan 194 L10); revisit only with a 2–3-bit format |

### 3.6 The norm, the earlier failure, and "integer, QNN-style" as a candidate

* **What the norm computes in today** [C]: f32 floating point on both
  sides. CPU: NEON, four 4-lane FMA accumulators, a horizontal sum, scalar
  `1 / sqrt(mean + eps)`, then `x · scale` (`neon_impl.cpp:1888-1935`). HTP:
  `hvx_rmsnorm_f32` — the same 16 chains as scalar `sffma`, the spec's
  pairwise reduce and integer round-to-nearest sqrt / reciprocal
  (`m1_norm_scale_det`), then `(x · r) · gamma` as IEEE `sf`
  (`hvx_m1_ops_f32.c:48-84`). Bit-identical to the phone's CPU (rule 47:
  392 / 392 rows), at 16 077 pcycles an op on silicon because the chains
  are scalar.
* **The earlier failures** [M-LFM]: (1) #152, 2026-09-29
  (`152-resident-accuracy.md:474-511`, rule 45): with attention already
  bit-identical, the *old* HTP norm (row scale summed in another order,
  2–4 ulp, 136–142 dB against the CPU) left `N_rms` at 2.64 dB at the MoE
  input — a routing flip; decode PPL stayed within +0.9 % on all 8 prompts
  and the user rejected all 8 texts for repetition (a new loop on p04).
  (2) #194 sitting 1b's E2 (`194-sitting-1.md:112-156`): the vector norms
  (L3, 133–146 dB) together with the vector router (L2) left the text at
  the first token and produced repeated meta-commentary; not gated, and L2
  / L3 were never separated. **Why**: not the norm's own error (nearer f64
  than the CPU's, rule 39) but the amplifier behind it — the dynamic
  per-block activation quantizers and the top-k near-ties flip on a 1-ulp
  difference.
* **Candidate N1 — order-free sum of squares, f32 stream kept.** Scale the
  row to fixed point by a per-row power of two (exact), accumulate x² in
  integers (any lane count, any tree, same bits), take r from the integer
  sum with the integer sqrt the spec already has, multiply in f32. The HVX
  kernel becomes 32 lanes wide and the CPU can run the identical spec
  (doc 45 §3.3's `_det` rule), so the norm stops being a source of
  CPU-vs-NPU difference at all. It is **not** bit-identical to today's f32
  norm — D2 allows that; the PPL / loop / approval gate reads it.
  Host-gateable: kernel ≡ spec bit for bit, spec vs f64 SNR.
* **Candidate N2 — QNN-style integer stream.** int16 activations with
  static calibrated scales through the norm and into the FC input (what
  `ref_16` shows as `rmsnorm_*_compute_mean_squared / normvals / normalize`
  and what T-MAN's op takes). It removes the amplifier itself, but needs
  calibration, a new activation format and its own accuracy study; the
  upstream static-u8 recipes all lost on PPL (doc 53_int §4, citing doc 51
  §2.12–2.14). Not planned; listed so the choice is visible.
* Neither is decided. N1 is filed when a resident Gemma norm exists; it is
  ranked 4th because nothing in §3.2 says the norm is where the time is.

### 3.7 Contract §2 and doc 45 §3

Three walls: the M=1 GEMV and the DMA feed are reused for Gemma's experts
(walls 1, 2); wall 3 (transport) returns as the miss round trip. Arena
budget: the pool replaces "all experts resident"; one PD ≤ 3840 MiB, heap
capped (rule 50). `QS4CX_WH` keeps no CPU fallback. DMA hidden behind
compute: the expert feed as today; a miss read is *not* hidden today —
that is lever 2. `_det` before every quantizer: kept as "a scalar spec per
kernel, held bit for bit on the host", with D2's gate in place of text
identity.

## 4. Steps

Each ends in a rung of `.claude/skills/hexagon-gates`. S0 needs no answer;
S1 needs B1; S4 onward need B3.

* **S0. The FSU guard on the per-token entry** (LFM path, ARM side only).
  `set_decode_graph_desc` / `bindMoeOp` throw when a bound layer's experts
  are virtual, naming both switches; `run_inproc_e2e.sh` gains one line
  (`E2E e3 + expert cache refused: …`). No IDL, no skel change. Gate:
  rung 1 (`ALL CHECKS PASS`, `*Lfm2Moe*` 6/6, `INPROC E2E PASS` with the
  new line), rung 3 once before the PR. **No device sitting.**
* **S1. Model facts (no code).** After B1: the user places the HF files
  under `/local/mnt/workspace/models/<name>/hf`; §3.2's table is refilled
  from `text_config`; doc 54 §5's R1–R7 are each judged; the pool-size
  table (C → arena MiB, RAM, cold-prefill floor, handle count) goes to
  BENCHMARK. Ends with B2 and B3 restated on real numbers. Gate: the table
  committed; every cell cites a config key or a file size.
* **S2. CPU reference in the tree.** Merge upstream PR #4296 into
  `htp_decode` (the user's merge, as with every upstream PR; it is open
  with changes requested). Expected conflicts [C]: `compute_ops.h` (the
  `gemm_q4_0_batch` declarations moved; the HTP backend overrides them),
  `neuralnet.cpp` (`NNTR_LAYER_PROFILE` beside `NNTR_OP_TIME`),
  `mha_core.cpp` (#4343 rewrote the same function). Gate: rung 1 plus the
  PR's own `*Gemma4*` gtests; LFM gates unchanged; rung 2 and 3 build.
* **S3. Model files.** PR #4296's converter, then `nntr_quantize_stream`:
  the all-Q4_0 CPU model with `moe_cache_size`; md5s and sizes against
  S1's table. (`QS4CX_WH` for Gemma's experts is S5.) Gate: rung 1 (the
  tiny Gemma fixture's Q4_0 differential), sizes equal the table.
* **S4. Device sitting G1 — unavoidable; the first per-stage breakdown.**
  Handoff `docs/measurements/201-gemma-cpu-reference.md`, full-model E2E,
  prompt 512, G 64 / 512 / 1024, reboot, cool start, ≤ 4 variants:

  | variant | what | reads |
  |---|---|---|
  | **A** | CPU Q4_0, pool at the budget's C (B3), warm | tok/s ×2 per G; `NNTR_OP_TIME=1` and `NNTR_MOE_PROFILE` once per G (the per-stage split: router, per-expert GEMM, activation, scatter, miss read, dense FFN, attention, norms, lm_head); RSS; texts; 8 prompts at G = 256 self (`NNTR_PPL_DECODE` writes the continuation) + `loop_check.py` — **A's own loops** |
  | **Ac** | A, cold (page cache evicted) | the flash term, prefill cold |
  | **B** | pool at C / 2 | the miss sensitivity; PPL / text ≡ A expected (same arithmetic) |
  | **T** | A with a routing trace (one line a layer call; added in S3 if the Gemma layer has none) | replayed offline by `tools/moe_expert_cache_sim.py` for every C → hit rate and misses a token |

  Stop rules: OOM kill (the budget is wrong → back to B3); a token above
  2 s (wrong file / packing). Fold: §3.2 and §3.4 are rewritten from the
  readings; if the flash term is the largest, lever 1 / 2 issues are filed
  before any kernel.
* **S5. The MoE on the HTP for Gemma, hybrid** (doc 54 Phase 3; its own
  issue and plan). The shared pool / staging / read-ahead moved out of
  `lfm2_moe` in a behaviour-neutral commit; `QS4CX_WH` writer for Gemma's
  experts (fused `[gate | up]`); the GeGLU epilogue with its spec; the
  expert-count limits of §2.4; the router stays on the ARM (so the LRU
  does). Gate: rungs 1–3; then sitting G2 (**A** = S4's CPU run, **H** =
  hybrid at the same C, **Hc** cold, `NNTR_HTP_PROFILE=2` for the layer
  call's stage row).
* **S6. Rank again, file the levers.** §3.4 re-read on G2's numbers; one
  issue per lever that reads above 1 ms a token, each with its own plan
  (N1 among them once a Gemma norm is resident).

## 5. Risks

* **Model identity and size (B1, B2).** Everything in §3.2 is recalled;
  a different checkpoint changes every row. S1 exists to remove this
  before any build.
* **RAM / OOM.** Arena (ION, outside RSS) + RSS + page cache on a 12 GB
  phone; the honest number is cold (doc 53 §3.1). S4's Ac and RSS cells
  show it; the stop rule names OOM.
* **Flash rate and page-cache state** move a decode number by more than
  any lever here; every handoff states warm / cold and runs both.
* **Address space / the mapping loss.** Pool sizes are chosen below 14
  windows; the ceiling cell after every run stays (rule 50, 54).
* **DMA rate, DVFS, thermal drift.** Same-sitting A/B only, cool start per
  G block, thermal checkpoints (rules 13, 52).
* **Stale skel / stub.** The ARM stub in this checkout was stale against
  the IDL; every sitting's app, skel and stub come from one tree with
  md5s (rule 3).
* **Accuracy gate power.** If Gemma's A loops under greedy decoding as
  LFM's did, "no new loop" is weak; S4 reads A's loops first and the
  detector's rule is fixed before S5's sitting.
* **Upstream churn.** PR #4296 is under review; a merge now may need a
  re-merge. S2 pins the sha.
* **Host vs device.** No Gemma HTP kernel has a host fixture yet; nothing
  in this plan about Gemma on the DSP is measured.

## 6. Docs to update

* **`docs/htp_moe/BENCHMARK.md`**: a "Gemma MoE" block — S1's fact and
  pool tables, S4's A / Ac / B rows (tok/s at G 64 / 512 / 1024, prefill
  warm / cold, RSS, per-stage split, hit rate per C from the trace,
  PPL, loops), artifact rows (model files, md5s). The LFM "now" rows stay
  as the record of that model; a line says the goal column is re-based by
  B2.
* **`docs/htp_moe/LEDGER.md`**: rules — *the per-token entry and the
  expert pool are mutually exclusive as built* (§2.3, with the host line);
  *for a 4 B-active model the byte floor, not 50 tok/s, is the ceiling*
  (after S1); *Gemma's routing hit rate per pool size* (after S4). Open
  items: the LFM E2E levers (#194's L0–L7) marked parked by D4; a Gemma
  section for levers 1–6; §Upstream gains PR #4296's sha and review state.
* **Contract `0001` §12**: D1–D4 as a dated row; §1's model, "now" and
  goal rows re-based when B1 / B2 are answered; §11 (weights) gains the
  Gemma directories.
