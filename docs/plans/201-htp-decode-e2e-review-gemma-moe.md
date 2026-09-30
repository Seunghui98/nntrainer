# 201 — `htp_decode`: the expert pool (FSU) inside the NPU end-to-end decode, sized for the Gemma MoE model

Issue: dlwlzzero/nntrainer#201 (p0). Base `htp_decode` @ `0aacbce24`
(= `htp_first_version` @ `d4a898430` + one docs commit). Wherever the
contract says `htp_moe`, read `htp_decode` (contract §12, 2026-09-30).

**Revision 2 (2026-09-30), after the user read revision 1 (`dc8a13309`).**
The spine changed: the goal is **FSU × E2E** — the expert pool working
*inside* the per-token entry, and what the pool frees used to improve the
two-session structure. Revision 1's "park the LFM E2E path" and "refuse
FSU with the per-token entry" are withdrawn.

Evidence classes, on every claim that matters:
**[C]** code read on this tree, `path:line`; **[H]** run on the workstation
host in this planning session; **[M-LFM]** measured on silicon on
LFM2.5-8B-A1B *before* FSU and #4343 were merged (doc cited); **[M-up]**
measured by the upstream author on another unit / tree (docs 52 / 53);
**[R]** a reference read at its source; **[G]** arithmetic on measured or
read inputs, or a stated guess.

## 0. Decisions, and what is left for the user

| # | decision (user, 2026-09-30) |
|---|---|
| D1 | Speed: as fast as possible; **50 tok/s** decode is the baseline bar, reported against the byte floor of §3.2 |
| D2 | Accuracy: **not bit-preserving**. Slight wording differences are fine; **repetition loops or off-context answers fail**. Pointers: the earlier norm optimization failed — consider integer arithmetic; check how QNN computes it (§3.7) |
| D3 | Device: **S25 (v79) only** |
| D4 | Model: **`google/gemma-4-26B-A4B`** ("Gemma moe (30b)"), weight data types unchanged. Its files come later; until then the work runs on LFM2.5-8B-A1B with every limit sized for Gemma's shape |
| D5 | **The main goal is FSU × E2E**: apply the expert pool to the two-session end-to-end path and improve that structure with it |
| D6 | The pool size a DSP session holds is found **by measurement** (a sweep), and the miss path must use the page cache the prefill leaves behind |

**Left for the user** (each has a default; none blocks S0–S2):

| question | default |
|---|---|
| Upstream PR nntrainer/nntrainer#4296 (Gemma 4 MoE on the CPU) is still open upstream with changes requested. Merge it into `htp_decode` when the Gemma steps start (S5)? | Yes, at S5, pinned to a sha; its tiny fixture is used from S4 on a work branch without the merge |
| Gemma's lm_head / embedding type. The in-tree Gemma config ships them as `Q6_K`; "weight data types unchanged" reads as `Q4_0`, as on LFM's NPU model | `Q4_0` |
| Prefill gate. "Never below −5 % of the same sitting's A prefill" is kept as a rule; LFM's absolute 497 tok/s does not apply to a run that streams experts from flash | The rule, read warm and cold, no absolute number |

## 1. Goal and gate

**Acceptance (issue + D1–D6).** (1) The structure review of the decode E2E
path with FSU in the tree (§2). (2) A measured per-stage breakdown on the
device with the pool on (S0, S2's handoffs). (3) Ranked levers (§3.6).
Then the build: decode runs end to end on the NPU (`NNTR_HTP_E2E=1`, one
packet per session per token) **with the expert pool on**, and the pool
size, the session count and the miss path are chosen from the sweep.

| gate | where | pass |
|---|---|---|
| FSU × E2E works | host: `run_inproc_e2e.sh` new lines (pool smaller than the expert set, logits against the all-resident E run); device: `calls/token=1.00`, `timeouts=0/0 stale=0/0`, the miss counters | host `bit_identical=1` while the expert sum order is kept (§3.3); device text / PPL against the same sitting's A |
| speed | `decode tok/s`, prompt 512, G 64 / 512 / 1024, cool start, against the same sitting's A | faster than the sitting's two-session resident E0; then as fast as the floor of §3.2 allows, 50 as the bar |
| accuracy (D2) | decode PPL forced on A's continuation (`NNTR_PPL_DECODE`, 8 prompts, G = 256) pooled ≤ 1.02 × A, ≤ 1.05 per prompt; **no new loop** (`tools/htp/loop_check.py --prompt`) where A has none; **the user approves the texts** ("off-context" is a reading, not a number) | all three; a differing text alone is not a fail |
| kernel | every new DSP kernel: a scalar spec in `test/htp/host/` and a host check holding the kernel to it bit for bit, plus an SNR line against f32 | `ALL CHECKS PASS` |
| standing | prefill per §0; `QS4CX_WH` has no CPU fallback; the S1 ceiling cell after every run; host state today: `ALL CHECKS PASS`, `INPROC E2E PASS` on this tree [H] | — |

Weak spot carried from `docs/measurements/194-sitting-1.md` [M-LFM]: LFM's
own A loops on 5 of 8 prompts at G = 256 and the detector missed a sixth by
0.06; "no new loop" is read with A's texts beside it.

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

One dspqueue packet per session per token, 44 mailbox hops, the sessions
strictly alternate. Per-op pcycles are summed per kind per side and printed
at close (`htp_compute_ops.cpp:3731-3776`), the L0 lines split the ARM side
(`:3787-3811`). **Not instrumented inside the token**: the MoE round's
stages (DMA first-ready, drain, compute) and the FC's (quantize, DMA wait,
GEMV) — the stage probes are switched on only by the `_timed` FastRPC
entries (`nntr_hvx_mm_u8i4.c:794`), never by the token driver.

### 2.2 Where the weights live (LFM, FSU off) [C]

| what | where | how it is read |
|---|---|---|
| 1408 expert weights, 3696 MiB `QS4CX_WH` | S1: ION arena, uncached, 256 MiB chunks (`htp_compute_ops.cpp:5313`, `:5493`), 15 chunks at the 3840 MiB ceiling | M=1 HVX GEMV fed from VTCM by DMA, `src_bypass=1` (rule 43) |
| router weights 22 × [2048][32] f32 | S1 heap (`hexkl_graph.c:589-605`) | scalar `sffma` chains |
| 67 Q4_0 weights → 74 `Q4M1` handles, 383.6 MiB (448 mapped) | S2: its own ION arena (`registerQ4m1` `:1722`, `placeOn(e.h2, …)` `:1735`, `e2ePlaceFc` `:4085`), after S1's arena (rule 54) | per-lane DMA into a 2 MiB L2 scratch (`nntr_hvx_fc_q4.c:95,342-351`): **S2 has no VTCM** (S1's `hexkl_micro_hw_init` holds all 8 MiB; `hvx_add_f32.c:55-68,137-146`) |
| KV cache, gammas, conv state, RoPE table | S2 heap (≈ 30 MiB); S1's heap ≈ 70 MiB | — |

**Why there are two sessions at all**: 3696 + 384 MiB does not fit the
3840 MiB one PD maps (rules 8, 49, 57). That premise is what the pool
removes (§3.4).

### 2.3 FSU (#4383) in this tree, and what FSU × E2E needs built

* **Off by default**; `NNTR_MOE_CACHE_EXPERTS=C` turns the experts virtual
  (`lfm2_moe_layer.cpp:231-242,302-307`). With it on, **the ARM owns the
  pool**: each layer call makes its routed experts resident through one LRU
  shared by all layers (`tryMoeLayerOnAccelerator`, `:790-815`); a miss is
  a `pread` from the model file into an arena slot (`readWeight`,
  `htp_compute_ops.cpp:5172`) and one swap RPC (`:5044-5113`); the DSP
  rebinds the retired handle **in place, keeping its number**
  (`nntr_hvx_mm_u8i4.c:546-572`). Scales and column sums sit in the slot
  after the nibbles, so a swap carries offsets only (`:4964-4970`).
  Arena = `layers × C` slots (LFM 5.29 MiB each, `takeExpertSlot`
  `:4920-4962`). The hybrid path (router on the ARM) runs with the pool
  today; on the host the lfm25 fixture generates the same tokens at C = 1,
  2 and unset [H].
* **What stops the pool inside the per-token entry — the build list:**
  1. *The ARM never learns the routed experts.* The router runs on S1 and
     the layer's hook returns before the LRU is consulted
     (`lfm2_moe_layer.cpp:1199-1206`).
  2. *The graph's MoE op holds a static handle table*, `h_gu[32]` /
     `h_dn[32]`, copied once at the layer's first call (`bindMoeOp`
     `:2131-2160`) and read every token (`hexkl_graph.c:104-107`); after an
     eviction the same handle number names another expert's bytes.
  3. *There is no miss path from the DSP*: S1 answers the ARM once, at the
     end of the token (`hexkl_token.c:271-321`).
  4. *No guard.* **[H]** `NNTR_HTP_E2E=1` with `NNTR_MOE_CACHE_EXPERTS=1 |
     2 | 4` on the lfm25 fixture dies only by accident (`MoE op 54 expects
     experts=4 …, the layer has 3`: the virtual path compacts the expert
     array). On the 8B a 512-token prefill activates all 32 experts, the
     check passes, and decode would read stale handles [C, not run].
* **Address space.** FSU off: S1 sits at 3696 of 3840 MiB, so the per-boot
  loss of one 256 MiB window (nine `LEAK` stops in #194 sitting 1b, cause
  unknown) stops a load. With a pool under 14 windows it cannot.
* **VTCM, feed schedule**: untouched by FSU [C].
* Not measured on this tree: any device number with a handoff file (A, E or
  the pool). The G = 64 reading in the issue is hearsay.

### 2.4 Limits sized for LFM that Gemma's shape breaks [C] / [R config]

Gemma's `text_config` (fetched from
`huggingface.co/google/gemma-4-26B-A4B/resolve/main/config.json`, HTTP 200,
2026-09-30 [R]): hidden 2816, 30 layers = 5 × (5 sliding + 1 full
attention), 128 experts, top-8, `moe_intermediate_size` 704,
`intermediate_size` 2112, 16 heads, head_dim 256, 8 KV heads, global
head_dim 512 with 2 global KV heads, `attention_k_eq_v`, sliding window
1024, vocab 262 144 tied, `final_logit_softcapping` 30, `gelu_pytorch_tanh`,
no per-layer input embedding, no shared-KV layers.

| piece | LFM assumption | Gemma needs | host-gateable without the model? |
|---|---|---|---|
| expert count | `HTP_GRAPH_MAX_EXPERTS 32` (`htp_graph_desc.h:51`): op record arrays, router `[K][32]` | 128 a layer | yes (**built in S1**: the pool table replaces the op's arrays) |
| handles | `HEXKL_MM_U8I4_MAX_WEIGHTS 2048` → 1024 experts a PD | up to ≈ 1300 slots a PD (§3.4) | yes (S1) |
| op count | `HTP_GRAPH_MAX_OPS 256` (LFM 228) | ≈ 25 ops × 30 layers | yes (S1) |
| rounds, row | `HEXKL_TOKEN_MAX_ROUNDS 255`, mailbox row 8192 floats | 30 rounds, 2816 floats | fits |
| MoE kernel | fused `[gate | up]`, **SwiGLU** | **GELU-tanh(gate) × up** (doc 54 R1); K 2816, inter 704 are multiples of 32 | yes: spec + kernel on PR #4296's tiny fixture (S4) |
| router | sigmoid + bias top-k (`hvx_m1_ops_f32.c:238-300`) | RMS-norm (no gamma) × scale / √H → dot → softmax → top-8 → renormalise × per-expert scale, fed by the un-normed stream | yes (S4) |
| norm | `hvx_rmsnorm_f32` **returns without computing** unless the width is a power of two (`:64-67`); 55 norms a token | width 2816; ≈ 10 norms a layer incl. q / k / **v** | yes (S4; §3.7's N1 fits here) |
| dense FFN | SwiGLU, layers 0–1 | GeGLU in every layer beside the MoE, two branches added then normed (`HTP_GRAPH_N_SLOTS 3` too few) | yes (S4) |
| attention | `ROPE` / `ATTN_M1` admit head_dim 64 (`htp_graph_desc.h:497`), full causal | head_dim 256 / 512, sliding window, proportional RoPE (partial 0.25), K = V in full layers | spec yes (S4); speed needs silicon (S5 / S6) |
| head | vocab 128 000, 8 slices of 16 384 rows; `NNTR_HVX_Q4M1_SLOTS 80` | 262 144 rows (16 slices), ≈ 230 Q4_0 handles; logit soft-cap (argmax-invariant), embedding × √H | yes (S4) |
| conv kinds | `CONV1D_GATE`, conv projections | none | — |
| "every kind resident" | the FC kinds are resident only when every kind is (`htp_compute_ops.cpp:1541-1551`) | holds for Gemma only when attention has a kernel (S4) | — |
| graph builder, load hand-over | `htp_graph_lfm2_build` (`:594`), `lfm2_moe_causallm.cpp:154-231` | a Gemma builder and hand-over | yes (S4) |
| fixtures | `lfm2_moe_tiny{,_hd64,_lfm25}`, `*Lfm2Moe*` | PR #4296's `gemma4_moe_tiny` (CPU) | — |

**Consumers that move with a changed contract** (S1): the IDL
`test/htp/nntr_hvx.idl` + `generate_stub.sh` + the skel — the generated ARM
stub in this checkout was older than the IDL and had to be regenerated
before the host twin would configure [H]; `HtpComputeOps` (pool ownership,
the miss server, the graph binding); `htp_dspq_wire.h` (the token response
grows the routed sets and miss counters); `hexkl_token.h`'s mailbox layout;
`NNTR_HTP_PROFILE` lines (misses, miss wait) and `tools/htp_fc_report.py`;
`tools/moe_expert_cache_sim.py`. The quantizer's format tag and the loader
check do not move until S5 (Gemma's `QS4CX_WH` writer: same tile layout).

## 3. Design

### 3.1 The spine

1. Measure the pool on the path that already runs it (the hybrid), on this
   tree, now: miss cost, hit rate per pool size, warm against cold (S0).
2. Put the pool inside the token driver (S1), with every table sized for
   Gemma (128 experts, 30 layers, > 256 ops).
3. With the pool smaller than "all experts", place the FC set and the
   lm_head beside it and run **one PD**; read one PD against two in the
   same sitting (S2). That is the structural gain FSU offers the E2E path.
4. Gemma's kernels — attention included — and its end-to-end graph,
   host-gated on PR #4296's tiny fixture (S4); the model end to end on the
   device when its files arrive (S5, no hybrid stage); then its levers
   (S6).

**Rejected alternative** (revision 1): run Gemma as a hybrid first and
leave the E2E path parked. It measures the new model early but builds
nothing of D5, and the mechanism it would postpone (pool in the token) is
the one every later Gemma step depends on.

### 3.2 Bytes a token and the floor they set

Every token reads each active weight once, so *bytes a token ÷ memory
bandwidth* is a floor on the token time no kernel can beat. The bandwidth
is measured: this phone's DRAM gives ≈ 70 GB/s to any mix of readers (#90's
two-reader probe, LEDGER rule 44 [M-LFM]); the weight DMA reaches ≈ 57 of
it in the app (rule 43).

| | LFM2.5-8B-A1B (rule 42) | Gemma-4-26B-A4B [G from the config] |
|---|---|---|
| MoE, `QS4CX_WH` | 484 MB | 30 × 8 × 2.97 MB = **714 MB** |
| attention q / k / v / o, Q4_0 | in the 255 MB below | 25 × 34.6 M + 5 × 49.0 M = 1.11 B weights = **624 MB** |
| dense FFN / other FCs, Q4_0 | 255 MB | 30 × 17.8 M weights = **301 MB** |
| lm_head, Q4_0 | 147 MB | 738 M weights = **415 MB** |
| **bytes a token** | **886 MB** | **2.05 GB** (3.8 B active weights) |
| floor at 70 GB/s | 12.7 ms → 79 tok/s | **29.3 ms → 34 tok/s** |

So for Gemma the byte floor sits below the 50 tok/s bar; results are
reported as a fraction of the floor, and 50 stays the bar. What would lower
the floor (options only, none planned): not reading the whole lm_head every
token (it is 20 % of the bytes), fewer bits per weight, fewer active
experts.

### 3.3 The miss protocol

**What has to exist**: the DSP resolves (layer, expert) → slot itself; a
miss found by S1's router mid-token is served before that layer's experts
run; the pool's owner learns what was routed.

| | alternative | cost of one layer with misses | verdict |
|---|---|---|---|
| **P-A** | **ARM pool server over the mailbox page.** The DSP owns an `id → slot` table (`layers × experts` u16, "absent" = a miss). On a miss S1 posts `{layer, routed set, missing ids}` on a third mailbox slot; an ARM thread that polls that word *only while a token is in flight* picks victims (the existing `ExpertLru`, never a member of the posted routed set), `pread`s each missing expert into its slot (uncached ION: the copy is the write to DDR), posts `{id → slot}`; S1 updates its table, rebinds in place and runs the layer. Hits touch nothing. The routed sets of the token ride back in S1's response and refresh the LRU | one mailbox round (the hop measured 0.33 ms with sleeping waits, 2.4 µs spinning, rule 50 / 1b) + the read: 0.31–0.37 ms per 5.25 MiB expert from a warm page cache, ≈ 1.75 ms cold at 3.0 GB/s [M-up]; several misses read in parallel by the existing reader threads | **chosen** |
| P-B | ARM-brokered over dspqueue + the swap RPC (today's `register_qs4cx_wh_expert_files`, called mid-token) | the same read + a blocking ARM wake, measured ≈ 2.9–3.4 ms per wake on this path (1b L0 [M-LFM]) + the RPC (1.4 ms / token at 37.6 misses on LFM [M-up]) | rejected: the wake alone exceeds the read |
| P-C | DSP-side file read | the unsigned PD has no file mapping of its own (`HAP_mmap` answers `0x80000402`, rule 57); its file I/O is a reverse RPC to the ARM, i.e. P-B with an extra copy [G] | rejected |
| P-D | read-ahead from the previous token's routing, on top of P-A | upstream measured the ceiling at 64 % with the wasted reads costing DDR (doc 53 §7) [M-up, LFM] | not built; re-read on the S0 trace with the simulator before closing |

Details of P-A that the design fixes now:

* **The spinning side is an ARM core, not a DSP thread**, so rule 56 (a PD
  that spins steals a hardware thread from the computing PD) does not
  apply; S1 itself waits for the answer with the hop's sleeping wait. An
  ARM bounded spin on this path took the wake from 2.9 to 0.07 ms (1b e2,
  one block [M-LFM]).
* **Hit experts run while the misses are read.** At M = 1 the experts of a
  layer run one after another; the hit ones go first, each output kept in
  its own row, and the rows are added in expert order at the end — the sum
  order, hence the bits, stay those of the all-resident run. This is what
  upstream lists as open (doc 53 §8) and PR #4296 does one expert ahead on
  the CPU.
* **The static handle arrays go.** `graph_op_moe` takes its handles from
  the pool table; the op record keeps only the shape. That is also what
  lifts the 32-expert limit for Gemma.
* **Prefill is unchanged**: the ARM-driven read-ahead and batched swap of
  #4383 fill the same pool; the table is synchronised once when the first
  decode token starts.
* **Page cache (D6).** After a prefill the experts it read are in the OS
  page cache, so a decode miss is a copy, not a flash read — as long as
  the kernel keeps those pages. The bytes are then held twice (page cache +
  ION arena); the page cache is reclaimable, the arena is not. For LFM the
  expert file section is 3.9 GB and fits beside the arena in 12 GB; for
  Gemma it is 11.4 GB and cannot, so its misses trend cold. Knobs, each a
  sweep cell rather than a guess: `posix_fadvise(WILLNEED)` on the
  runner-up experts the router reports (keeps them warm without a slot),
  `FADV_DONTNEED` on experts now in the arena (frees RAM, makes a re-miss
  cold). `pread` stays: upstream chose it over `mmap` + copy for the
  uncached destination (`htp_compute_ops.cpp:2343-2346`), and `O_DIRECT`
  into a dma-buf was judged not pinnable (doc 53 §7).

### 3.4 One PD or two — the arithmetic

One PD maps 3840 MiB in 256 MiB windows and its heap shares the same
4 GiB (≈ 182 MiB left at 3840 mapped, rule 8); a second PD has its own
4 GiB but maps 3584 beside the first and gets **no VTCM** (rules 50, 54).

| | LFM2.5-8B-A1B | Gemma-4-26B-A4B [G] |
|---|---|---|
| expert slot | 5.29 MiB | 2.87 MiB (1.89 + 0.95 MiB of tiles + tails, page-rounded) |
| all experts | 704 slots = 3696 MiB | 3840 slots = 10.6 GiB |
| FC set + lm_head (Q4_0 → `Q4M1`) | 384 MiB | 883 + 396 = **1279 MiB** |
| heaps today (S1 + S2) | ≈ 100 MiB | more (KV cache for 30 layers at head_dim 256 / 512); 150–200 MiB assumed |
| **one PD**: pool room = 3840 − FC set − heap growth | ≈ 3330–3390 MiB → **≈ 630–640 slots → C = 28 of 32 (88 %)** | ≈ 2350–2450 MiB → **≈ 820–850 slots → C ≈ 27 of 128 (21 %)** |
| **two PDs**, S1 = pool only | all 704 resident (today) or any C | ≈ 3740 MiB → **≈ 1300 slots → C ≈ 43 of 128 (34 %)** |
| two PDs, S2's spare room as more pool | — | S2: 3584 − 1279 − heap ≈ 2100 MiB → + ≈ 730 slots → **C ≈ 67 (52 %)**, but those experts run in a PD with no VTCM and no HMX (decode M=1 only) |

What one PD removes for the E2E path (LFM readings, 1b [M-LFM]): the 44
hops (hop_us 0.66 ms a token + part of the 0.5 ms per-stretch overhead),
the second session's open and mapping order, and **S2's lack of VTCM** —
in one PD the MoE feed and the FC feed use the 8 MiB in turn, so the FC set
is VTCM-fed (the kernel already picks VTCM when it fits,
`nntr_hvx_fc_q4.c:375-381`); the native FC read 41 GB/s L2-fed in the graph
against 68–70 isolated on VTCM. What it costs: for LFM 12 % of the experts
leave DRAM, so some tokens miss; for Gemma the pool shrinks from 34 % to
21 % of the experts. **Whether one PD wins is S2's reading for LFM and the
same sweep for Gemma**; the hit rate per pool size is the unknown in both
(LFM measured 57 / 73 / 85 % at C = 8 / 12 / 16 [M-up]; nothing above 16,
nothing for Gemma).

The one-PD path is mostly existing code: every kind resident in one session
is `NNTR_HTP_FORWARD=1` with all kinds (host line `calls/token=1.00`); it
never ran on the device only because the FC set did not fit beside the
resident experts (113 MiB allocatable, rule 49).

### 3.5 What of the LFM-era measurements transfers

Transfers (platform, S25 / v79): the PD limits (rules 8, 50, 54, 57); DRAM
70 GB/s, DMA 57 GB/s in the app, CPU Q4_0 GEMV 51–63 GB/s (rules 44, 43,
46); the ARM wake 2.9–3.4 ms and the walk ≈ 1.5 ms, hop 0.33 ms (1b); flash
3.0 GB/s, miss read and swap costs (docs 52 / 53); the amplification of
tiny float differences by the dynamic quantizers and the blindness of PPL
to loops (rules 39, 45); the v79 traps (rules 3, 52, 53, 58).

Does not transfer to Gemma: every tok/s and ms / token row, the per-kind
lines, plan 194 §3.4's ladder, the accuracy baselines, the kernels of §2.4.
LFM rows stay valid for LFM — but **on this tree they are unmeasured**.

### 3.6 Levers, ranked — hypotheses until S0 / S2 read them

| rank | lever | expected size | evidence | step |
|---|---|---|---|---|
| 1 | **Pool inside the token (P-A) → one PD**: hops gone, FC VTCM-fed | LFM: −0.7 ms hops, −2 to −3 ms FC feed, + misses at C = 28 (unknown) | §3.4; 1b [M-LFM]; [G] | S1, S2 |
| 2 | **Pool size** (and, for Gemma, S2's room as more pool) | Gemma: 240 expert uses a token; every 10 % of hit rate ≈ 24 misses ≈ 5 ms warm / 24 ms cold [G] | [M-up] miss costs; hit rate unknown | S0 trace, S2 sweep |
| 3 | **Miss read under the hits' compute; page-cache knobs** | up to the read time of each miss layer | §3.3; doc 53 §8 open | S1, S2 |
| 4 | **ARM out of the token**: bounded spin on the answer, the walk skipped (#194's L0) | −3.9 ms a token on LFM | 1b e2 [M-LFM, one block, G = 64] | S3 |
| 5 | **Non-exact kernels now allowed by D2**: native FC (in the tree, mask 0x2), vector router and norms | −1.6 to −1.8 ms (FC, measured); −0.66 ms (router + norms, text unresolved) | 1b [M-LFM] | S3 |
| 6 | **Norm: order-free integer sum of squares** (§3.7) | small on LFM (55 norms); ≈ 300 norm evaluations a token on Gemma [G] | rules 45, 47 | S4 |
| 7 | MoE round at its isolated rate; prefetch across ops | ≈ −1 ms on LFM, cause unknown | plan 194 L5 [G] | after S2's stage timers |
| rejected | hop deadline spin (L4) | 0 net (MoE +0.55 ms under the spin) | 1b e2, rule 56 | — |
| rejected | another cache policy | +2.4 %p hit on LFM | doc 53 §5.7; re-read on Gemma's trace | — |

### 3.7 The norm, the earlier failure, and "integer, QNN-style"

* **Today** [C]: f32 floating point on both sides. CPU: NEON, four 4-lane
  FMA accumulators, a horizontal sum, scalar `1 / sqrt(mean + eps)`
  (`neon_impl.cpp:1888-1935`). HTP: `hvx_rmsnorm_f32` — the same 16 chains
  as scalar `sffma`, the spec's integer round-to-nearest sqrt /
  reciprocal, then `(x · r) · gamma` (`hvx_m1_ops_f32.c:48-84`);
  bit-identical to the phone's CPU (rule 47), 16 077 pcycles an op on
  silicon because the chains are scalar.
* **The failures** [M-LFM]: (1) #152 (`152-resident-accuracy.md:474-511`,
  rule 45): the old HTP norm summed in another order (2–4 ulp, 136–142 dB)
  → `N_rms` 2.64 dB at the MoE input, a routing flip; decode PPL within
  +0.9 % on all 8 prompts, **the user rejected all 8 for repetition**.
  (2) #194 sitting 1b's E2 (`194-sitting-1.md:112-156`): vector norms +
  vector router left the text at the first token with repeated
  meta-commentary; never separated. **Why**: not the norm's own error but
  the amplifier behind it — the dynamic per-block quantizers and top-k
  near-ties flip on a 1-ulp difference (rule 39).
* **N1 — order-free sum of squares, f32 stream kept.** Scale the row to
  fixed point by a per-row power of two (exact), accumulate x² in integers
  (any lane count, any tree, same bits), take r from the integer sum with
  the spec's integer sqrt, multiply in f32. 32 lanes on HVX, the identical
  spec on the CPU, so the norm stops being a CPU-vs-NPU difference. Not
  bit-identical to today's f32 norm — D2 allows it; PPL / loops / approval
  read it. Host-gateable. It also lifts the power-of-two restriction.
* **N2 — QNN-style integer stream.** int16 activations with static
  calibrated scales through the norm into the FC input — what the repo's
  QNN optrace shows as three integer RMSNorm kernels with all scales folded
  to Q31 multipliers (`ref_16_qnn_optrace_analysis.md:63-76,132-142`). It
  removes the amplifier itself but needs calibration and a new activation
  format; upstream's static-u8 recipes lost on PPL (doc `53_int_requant`
  §4). Listed, not planned.

### 3.8 The three references

| reference | applies here | already done | ruled out |
|---|---|---|---|
| **upstream #4296** [R: `gh pr diff`]: `Gemma4MoECausalLM`, the `gemma4_moe` layer (router, gate / up / down, a per-layer LRU of virtual experts with one async prefetch ahead), converter, `moe_cache_size`, tiny fixture, timers | the CPU reference of D4 (S5); its tiny fixture host-gates S4; its one-ahead prefetch is §3.3's "hits run while misses are read" | the shared pool, batched swap and read-ahead of #4383 are richer than its cache; `NNTR_OP_TIME` overlaps its `NNTR_LAYER_PROFILE` | its batched GEMM is prefill-only |
| **QNN** [R: repo notes `ref_16`, `36`, `33`; ExecuTorch's Qualcomm LLM README on GitHub; **no QNN / QAIRT SDK on this workstation** [H]]: one execute per token, KV updated in place, static integer scales, **a 4 GB per-context limit answered by sharding the model across contexts** | sharding = §3.4's second PD as more pool; integer scales = N2 | one call per token, weights in the HMX layout, DMA double buffering, band tiling (doc 36 §2) | a QNN-built graph: no dynamic expert streaming in a context binary, and it would replace the backend |
| **T-MAN** [R: doc `53_int_requant_task.md` §4 (kernel source and README; the paper's PDF was not readable there), README re-read now; nothing recalled from the paper] | DMA → TCM is the ceiling it also runs at (rule 43) | compute hidden under the DMA at M = 1 | a LUT GEMV: no gain at 4 bits (plan 194 L10) |

### 3.9 Contract §2 and doc 45 §3

Walls 1 and 2 (M=1 GEMV, DMA feed) are reused; wall 3 (transport) returns
as the miss round, which P-A keeps off FastRPC. Arena budget: the pool
replaces "all experts resident"; a PD's heap never grows to the end of its
space (rule 50). `QS4CX_WH` keeps no CPU fallback: a miss that cannot be
served fails the token (`AEE_EEXPIRED`), never a silent skip. DMA hidden
behind compute: the feed as today; the miss read is hidden behind the hit
experts. `_det`: a scalar spec per kernel, held bit for bit on the host.

## 4. Steps

Each ends in a rung of `.claude/skills/hexagon-gates`.

* **S0. Device sitting, no code change — the pool on the path that already
  runs it.** Handoff `docs/measurements/201-pool-baseline.md`, this tree as
  it is (app + skel + regenerated stub from one commit, md5s), LFM, prompt
  512, G 64 / 512 / 1024, reboot, cool start per G, S1 ceiling after every
  run. Variants:

  | variant | what | reads |
  |---|---|---|
  | **A** | hybrid, nothing set (unchanged reference) | tok/s ×2 per G; the row of record on this tree; one run with `NNTR_MOE_TRACE` (the routing trace for `tools/moe_expert_cache_sim.py`: hit rate for every C) |
  | **E0** | `NNTR_HTP_E2E=1`, two sessions, all resident | tok/s; the per-kind and L0 lines — the E2E base on this tree |
  | **F28** | A + `NNTR_MOE_CACHE_EXPERTS=28` (the one-PD pool size) | tok/s warm; `NNTR_HTP_PROFILE=2` once: misses a token, read ms a miss, swap RPC; RSS + arena; text / nll against A |
  | **F16** | A + `…=16`, warm and **cold** (page cache evicted before the decode) | the miss price at a high miss rate, warm against cold — D6's first reading |

  Fold: misses a token and ms a miss per C; the simulator calibrated on
  F16 / F28; §3.4's LFM column and §3.6 re-read. Stop rules: `0x8000040e`
  (stale skel / stub), `LEAK`, `AEE_EEXPIRED`.
* **S1. The pool inside the token driver (P-A), sized for Gemma.** DSP: the
  pool table and `graph_op_moe` reading it; the miss slot of the mailbox in
  `hexkl_token_serve`; hit-first expert order with the ordered sum; the
  routed sets and miss counters in the token response; an opt-in
  `HTP_DSPQ_TOKEN_TIMED` flag that switches the existing stage probes on
  for a token (§2.1's gap). ARM: the pool server thread on `ExpertLru` and
  the reader threads; the table sync at the first decode token; the FC set
  placed on S1's arena when `NNTR_HTP_E2E_PDS=1`. Limits: experts a layer
  128, handles 4096, ops 1024. Host: `token_host_check.c` gains the server
  as a third thread (10 000 tokens at a pool of half the experts ≡ the
  all-resident run, timeouts 0); `run_inproc_e2e.sh`: `E2E e3 pool C=1 | 2
  ≡ e3 … bit_identical=1`, one PD ≡ two PDs, the refusal of today replaced
  by these lines. Gate: rung 1 (`ALL CHECKS PASS`, `*Lfm2Moe*` 6/6,
  `INPROC E2E PASS`), rung 2 (IDL / wire changed: skel md5, stub
  regenerated), rung 3.
* **S2. Device sitting — the pool-size sweep and one PD against two**
  (unavoidable). `docs/measurements/201-fsu-e2e.md`, LFM, method as S0.
  Variants: **A**; **E0** (two PDs, all resident — the anchor); **P2**
  (two PDs, pool, C swept 16 / 24 / 28 at G = 64, the best at all G);
  **P1** (one PD, pool C = 28 and the largest C that loads, FC VTCM-fed).
  Each pool cell warm and cold, with and without the two `fadvise` knobs at
  G = 64; one timed run per variant for the stage lines; 8 prompts at
  G = 256 for PPL / loops; texts pasted. Reads: P1 against P2 against E0
  (the structural verdict), misses and miss wait a token, the FC rate on
  VTCM, the MoE round. Fold: §3.4 and §3.6 rewritten; the winner becomes
  the structure the Gemma steps build on.
* **S3. The model-independent levers on the winner** (own issues when S2
  says they still read above 1 ms): L0 re-derived on the new token entry,
  the native FC default under D2's gate, the first-token one-time moved to
  the end of the prefill.
* **S4. Gemma's kernels and its end-to-end graph, host-gated on PR
  #4296's tiny fixture (no device, no 26B files).** Every decode kind
  Gemma needs gets a resident kernel, so the token entry is one call per
  token from the first device run (user, 2026-09-30: Gemma runs end to
  end, no hybrid stage). In the order the structure needs them: the GeGLU
  epilogue of the MoE M=1 path and its spec; the Gemma router and its
  spec; the norm at any width (N1 as the candidate) incl. the v norm; the
  GeGLU dense FFN and the two-branch add; **attention for head_dim 256 /
  512 (sliding window, proportional RoPE with the partial factor, K = V in
  full layers) and its spec**; the 262 144-row head with the soft-cap;
  then the Gemma graph builder and load hand-over on S1 / S2's token
  entry. Gate: rung 1 (each `bit-exact n/n` against its spec, SNR against
  the CPU layer on the fixture, and an in-process E2E line on the tiny
  Gemma fixture with the pool on: calls/token = 1.00, text / SNR against
  the CPU run), rung 2.
* **S5. Gemma on the device, end to end** (when the files exist; PR
  #4296 merged per §0): converter, `nntr_quantize_stream` incl. the
  `QS4CX_WH` writer for its experts. Sitting: **A** = CPU run with
  `moe_cache_size` (the text / PPL reference), **E** = `NNTR_HTP_E2E=1`
  through the pool, the C sweep warm / cold, the routing trace, one timed
  run for the per-kind lines — the first Gemma breakdown; §3.2 / §3.4's
  Gemma columns become measured. A hybrid run is a diagnostic only (to
  split a wrong text between the MoE and the rest), never a deliverable.
* **S6. Gemma's levers from S5's breakdown**: the attention kernels'
  speed on silicon first (S4 gates their arithmetic, not their time), then
  one issue per lever that reads above 1 ms a token.

## 5. Risks

* **Hit rate above C = 16 and for Gemma is unknown**; the whole one-PD
  argument rests on it. S0's trace answers LFM before any code; Gemma's
  waits for S5.
* **A miss mid-token is a new failure mode**: a lost post or a slow read
  must end as `AEE_EEXPIRED` on the token, never a hang or a stale slot;
  the host check drives it with injected delays. The slot a miss
  overwrites must not be one the DSP is reading — the routed set in the
  request is the guard.
* **RAM**: arena + RSS + page cache on 12 GB; the honest number is cold.
  Every pool cell runs warm and cold and records RSS.
* **Address space / the mapping loss**: pools are sized under 14 windows;
  the ceiling cell stays after every run (rules 50, 54).
* **One PD's heap**: the KV cache, graph and scratch join S1's heap; it
  must stay far from the end of the space (rule 50) — the load banner
  prints heap and mapped MiB.
* **DMA rate, DVFS, thermal drift**: same-sitting A/B, cool start per G,
  thermal checkpoints (rules 13, 52).
* **Stale skel / stub**: the stub here was stale; one tree per sitting,
  md5s on both ends (rule 3).
* **Accuracy gate power**: A's own loops; read with the texts.
* **Upstream churn**: PR #4296 is under review; S4 pins a sha.
* **Host vs device**: the host twin is one address space with no cache
  maintenance and no transport time — it proves the protocol and the bits,
  no ms.

## 6. Docs to update

* **`docs/htp_moe/BENCHMARK.md`**: a "FSU × E2E" block — S0's A / E0 /
  F28 / F16 rows (tok/s, misses a token, ms a miss warm / cold, RSS +
  arena, hit rate per C from the trace), S2's P1 / P2 sweep and the
  one-PD verdict, stage lines; a "Gemma MoE" block with §3.2 / §3.4's
  tables marked computed until S5 measures them; artifact rows.
* **`docs/htp_moe/LEDGER.md`**: rules — *the per-token entry and the ARM
  pool were mutually exclusive as merged* (§2.3, the host line); *miss
  price and hit rate per pool size on this tree* (S0); *one PD against two
  with a pool* (S2); *the byte floor per model*. Open items: a FSU × E2E
  section with §3.6's levers; #194's levers re-pointed at S3; §Upstream
  gains PR #4296's sha and review state.
* **Contract `0001` §12**: D1–D6 as a dated row; §1's model and goal rows
  gain the Gemma column and the floor.
