# 260 — Port upstream #4415's all-NPU Gemma prefill onto `htp_decode`, measure the 26B prefill + decode on the S25

Issue #260 (p0, hexagon). Base `htp_decode` @ `7ee43d81e`; `refs/pr/4415` @ `2fd298ecd`;
common base `d4a898430`. Read with #76's 2026-10-08 classification, doc 57 (their
§7.3 keys, §9.10 audit, §9.13 numbers), contract §1 Model row (㊸ option A, rule 73).

## 1. Goal and gate

Acceptance (issue): the Gemma-4 26B dummy runs end to end on the NPU — #4415's all-NPU
prefill (fused qkv+norms+RoPE, fused dense GeGLU, MoE, hd-512 `attn_f16`, lm_head+softcap,
per-layer FastRPC) + our one-PD decode — and both tok/s are read on the S25
`R3CY205ZMND`, prompts 447 and 1023, G 64 / 512 / 1024. Measurable:

* BENCHMARK.md Gemma row gets, per sitting, a 3 × 2 × 3 grid (§4 step 6): prefill
  tok/s for {A hybrid-as-today, N all-NPU} × {447, 1023}; decode tok/s for {A, N
  (hybrid decode), E (one-PD E2E)} × G; per-kind DSP lines, `calls/token` (E = 1.00),
  peak RSS + arena mapped + DSP heap, the S1 ceiling, cool start per G block.
* Bit identity: E's tokens == A's under plan 130 §3.5 (near-tie flips allowed, MoE dumps
  `bit_identical=1`); N's prefill MoE dumps == A's (the MoE call is untouched).
  The FC path is **not** bit-identical to A by construction (Q4_0 → QS4CX requant at
  load, their §2: +0.54 nll; rule 73) — every N / E number carries the label
  "FC requant (dummy)". No text / PPL column (dummy weights, contract §1 Accuracy).
* Host: rung 1 of `hexagon-gates` unchanged on every `htp_decode` line
  (`run_inproc_e2e.sh` incl. the gemma64 lines, `run_host_checks.sh`, the 7 `*Lfm2Moe*`
  gtests) **plus** #4415's five checks (`RMSNORM ROWS OK`, `ROUTER ROWS OK`,
  `ROPE ROWS OK`, `SOFTCAP OK`, `tile_f16_host_check`).
* Standing gates: prefill of A ≥ −5 % of the same sitting's A today (S5-0 / P4: 138–144
  tok/s at 447); A's decode within 2 % of P4's A16 (19.24 / 18.37 / 17.40).

## 2. Where it lives

Dry-run merge (`git merge --no-commit refs/pr/4415` on a scratch worktree, 2026-10-08):
29 files / ≈ 100 conflict hunks; the IDL and the fused layers auto-merge. Classified:

| class | files | rule |
|---|---|---|
| (i) ours, theirs must not touch | `quantize_stream.cpp` (sidecar, palette, QS2CX_WH; 9 hunks), `htp_wh_layout.h` (`FcWhEntry`), `lfm2_moe_layer.cpp/.h` (18 hunks: pool, `routeSoftmax`, 2-bit `moeReadExpertWeight`, diff/shadow), `hexkl_mm_u8i4_moe.{c,h}` (+999 ours: DMA_Q bits 19–20 #177, M=1 feed), `nntr_config.json` (26B), `gemma4_moe_tiny/*` fixture + `unittest_causallm_gemma4.cpp`, `moe_layer_host_check.c`, `standin/hvx_scalar.c`, `hvx_dequant_i32.{c,h}`, `swiglu_det.h`, `hvx_swiglu_det.h`, `causallm_common_properties.h` (`MoERouter`) | ours wins (`git checkout --ours`), then re-add the one additive line each needs (below) |
| (ii) theirs additive, clean | `qkv_layer.cpp/.h` (+265: in_norm, RoPE, q scale, v-from-k), `residual_add.*` (+106), `tie_word_embedding.*` (+110), `dense_ffn_layer.*` (+105, `gate_first`, `:up/:gate/:down`), `hvx_{rmsnorm,router,rope}_rows_f32.*`, `hvx_softcap_f32.*`, `hvx_tile_f16.h` (vshuff Kᵀ), `hexkl_attn_f16_plan.h` (hd 512), `nntr_hvx_mm_u8i4.c` (+220 fused entries), `nntr_hvx_session.h`, `test/htp/host/{rmsnorm,rope,router}_rows_host_check.c`, `softcap_host_check.c`, `tile_f16_host_check.c`, `tools/prefill_timeline.py`, `causal_lm.cpp` (`installKVCacheSharedAllocator`), `transformer.cpp` (dense registration by name, `tie_pending`, QS4CX FC registration), `nntr_hvx.idl` (union, 101 methods) | take as merged |
| (iii) real conflicts | `gemma4_causallm.cpp` (6 hunks: their `createMoe`/`createMlp`/`residual_add` epilogue/`FOLD_OUTPUT_NORM`/engine keys vs our `createFeedForwardBlock` + per-layer-input guard), `htp_compute_ops.cpp` (7: `invokeFc`/`get_or_register_fc(matAscale)` vs our `fcRowStep`/`prefillRows` chunking #236/#225 + `fcwhFind` sidecar; `gemm_qs4cx_moe_layer_fp32` signature `w_bits` vs `weights_wh, gelu, pre/post_gamma, eps`), `compute_ops.{cpp,h}` (new virtuals + the MoE signature), `mha_core.cpp` (4: RoPE table — ours builds any head_dim for the hook, theirs hd 64 only; `rope_rows` vs `max_timestep` #253/#256), `gemma4/meson.build`, `weight_converter.py`, `unittest_hvx_softmax.cpp` | resolved per §3 |

Consumers that move with the contract: IDL `test/htp/nntr_hvx.idl` → `generate_stub.sh`
+ `test/htp/build.sh` (their 4 new `.c` in `SRCS`) → skel md5 changes (rule 3 trap);
`HtpComputeOps` (`htp_compute_ops.cpp:1271,1705,1784,5181,5677,5702,5992` hunks);
`compute_ops.h:225–307` (their virtuals) and `:400` (MoE signature); quantizer: no format
tag change (`QS4CX` FC writing is their `7e2ecdfa7`, one `if`); loader check: our
`fc_wh_format` stays; `NNTR_HTP_PROFILE` stage tables: their fused calls report `dsp≈0`
(their §9.13) — `tools/htp_fc_report.py` / `op_time_report.py` gain no rows for them in
this sitting (recorded as a gap, §6). `Gemma4MoECausalLM::load_weight`
(`gemma4_moe_causallm.cpp:90–268`) — the E2E op list by weight name — is the one file
that must be rewritten for their graph (§3).

## 3. Design

**Chosen: one merge of `refs/pr/4415` into `htp/260-4415-prefill` (off `htp_decode`),
resolved once with the rules below; then two small commits of ours on top.**
Rejected: cherry-picking their ≈ 45 prefill-side commits. Their first three code commits
(`f0761ff90` router type, `e0e617243` MoE block, `2f6011e9a` GeGLU `act`) conflict with
plan 201 S4 by design, and every later pick re-touches `lfm2_moe_layer.cpp` /
`htp_compute_ops.cpp`, so the same 100 hunks would be resolved several times over,
without their history. The merge resolves each once; `git log` keeps their commits for
the ⑧ reconcile later.

Resolution rules (the user's, made concrete):

1. **Decode token path, pool, 2-bit experts, #250 repack, sidecar, #253 — ours.**
   `lfm2_moe_layer.cpp` is taken whole from ours (`moe_router=softmax`, CPU
   `routeSoftmax` at prefill — their DSP router measured 21.6 ms/call, 648 ms at 1023,
   *slower* than the CPU's 191 ms at 512 (their §3 / §9.13); the register-blocked
   `4e903b971` is unmeasured). `router_logits_f32` lands in the skel unused — a lever
   cell for a later sitting, not this one. Their `in_norm`/`out_norm`-in-the-MoE-call
   (`post_gamma`) is not taken: ours keeps `_pre_ffn_norm_2` / `_post_ffn_norm_2` as
   `rms_norm` layers (their §9.10 #7 says in_norm stays on the CPU anyway).
2. **Prefill-side fused ops — theirs.** `gemma4_causallm.cpp` takes their block
   (`qkv_layer` with `in_norm` + RoPE + q scale, `residual_add` epilogues with gamma +
   `scalar_multiplier` weight, `createMlp` dense FFN with `in_norm`/`out_norm` +
   `gate_first`, `tie_word_embedding` with `in_norm` + `softcap` under
   `FOLD_OUTPUT_NORM`, `attention_engine` / `attn_proj_engine` / `dense_ffn_engine` /
   `lmhead_engine` keys); our per-layer-input guard hunk (`:163–178`) is kept as the
   stricter of the two. `createMoe` becomes `virtual`; `Gemma4MoECausalLM` overrides it
   with our chain (`rms_norm` → `lfm2_moe` → `rms_norm`) and drops its
   `createFeedForwardBlock` override. Their `ENABLE_MOE_BLOCK` branch in
   `Gemma4CausalLM::setupParameters` is kept for the CPU `gemma4_moe` fixture path only
   if the gtests need it; `Gemma4MoECausalLM` stays the 26B's class (`main.cpp:303`).
3. **GeGLU flag — one bit, ours.** `HEXKL_MOE_FLAG_GEGLU 0x200000` (bit 21) stays; their
   `0x80000` would alias our DMA_Q field (bits 19–20, #177). Add
   `#define HEXKL_MOE_FLAG_GELU_TANH HEXKL_MOE_FLAG_GEGLU` so their kernel paths compile
   unchanged, and `HVX_GLU_GELU_TANH` maps to it in `hvx_dequant_i32.h`. The IDL `act`
   arg on `mm_u8i4_moe_layer*` / `swiglu_det_f32` is kept (union); the skel ORs
   `act ? HEXKL_MOE_FLAG_GEGLU : 0` into the session's `moe_set_opts` word per call
   (their `nntr_hvx_mm_u8i4.c:1344` form); our `invokeMoeLayer` passes
   `act = moeGeglu()`. Both call sites updated, `moe_opts_host_check` + `geglu_host_check`
   gate it.
4. **IDL — union; skel md5 changes.** 101 methods; our token driver / KV cache B /
   `moe_set_opts` and their `mm_u8i4_layer_norm`, `mm_u8i4_moe_layer_norm`,
   `rmsnorm_add_f32`, `router_logits_f32`, `lm_head_q4m1_f32` coexist. Rung 2 on v79
   and v81.
5. **FC weights, one image set.** `get_or_register_fc` (theirs, `:1705/5677`) takes our
   lookup order: `fcwhFind` sidecar image → offline `QS4CX` (`htp_qs4cx_from_packed`) →
   Q4_0 requant (`htp_qs4cx_from_q4_0x4`). With the dummy (Q4_0 FCs, no f32 source)
   both of the last two are a requant of the same class (rule 73); the sidecar is the one
   we already have on the device (P4: 788.8 MiB, md5 `aeab72fa…`), so the all-NPU
   prefill's fused calls and the E2E token's FC / DENSE_FFN ops read the **same arena
   images** and the FC set is placed once — the "both can coexist" answer, and the
   address-space answer (E16 mapped 2048 MiB + a second 823 MiB FC copy would not fit
   beside the lm_head's 396 and the KV). Verify: `registerFcWh` slices at
   `fcSliceCols(K)` must be head-aligned (their `fbad83d6e` requires `cols % rope_hd == 0`
   for q / k); if not, slice the sidecar handles at a head multiple (one constant).
   Fallback if the sidecar cannot feed the fused call in a day: their load-time requant
   as-is, `fc_wh_file_name` kept for the token, and C lowered until it loads (rule 71) —
   still a valid N / E cell, labelled "FC twice".
6. **lm_head once.** Their `lm_head_q4_0_prepare` (Q4M1 slices in the arena, last
   placement) and our E2E `LM_HEAD` (Q4M1, `attach_mib=396`) are the same bytes in the
   same format: `placeLmheadOnAccelerator` must find the E2E placement when `e2e_` is on
   (one lookup), else `lmhead_engine: cpu` for N / E (one row per prefill, ≈ 18 ms at
   their §9.12; +396 MiB RSS).
7. **E2E op list on the new graph** (`gemma4_moe_causallm.cpp:90–268`): the name map
   changes, the list (`htp_graph_gemma_build`) does not. `_attention_norm` →
   `layer{l}_qkv` IN_GAMMA; `_q_norm`/`_k_norm` → `qkv` weights by index; `_wq/_wk/_wv` →
   `qkv` WQ/WK/WV; `_post_attention_norm` → `residual_add` gamma;
   `_pre_ffn_norm`/`_post_ffn_norm_1` → `layer{l}_ffn` in/out gammas and `:up/:gate/:down`
   by name (gate-first order — never by index); `_post_ffn_norm` + `_layer_scalar` →
   `layer{l}_post_ffn_norm` gamma + `scalar_multiplier`; `output_norm` →
   `output_of_causallm` norm_gamma; `shape.softcap = FINAL_LOGIT_SOFTCAPPING` (the DSP
   caps: `hexkl_graph.c:538–550`; the ponytail at `:150` closes). Hooks:
   `tie_word_embedding`'s decode-row hook condition `!in_norm && softcap == 0` is
   relaxed when the E2E graph is set (the DSP does both); `residual_add`'s "plain row"
   hook likewise. `mha_core`'s RoPE-table hunk is ours (any head_dim for the hook); with
   RoPE in `qkv_layer` at decode (hd 256 / 512, their `:181`) the CPU row is rotated
   before `mha_core`, so at an E2E row the `qkv_layer` hook must return 1 before its
   `ropeRows` (the DSP's ROPE op rotates) — one early return. `htpDecodeKvSeed`
   (`mha_core.cpp:743`) reads the CPU-side `kv_cache`, which their rpcmem allocator keeps
   host-readable: unchanged, verified by the gemma64 `attn_caches=2` line.
8. **Config for the sitting** (`/local/mnt/workspace/models/gemma4_26b/nntr_config.json`,
   #253's original `config.json`): ours + `"attention_engine": "htp"`,
   `"attn_proj_engine": "htp"`, `"dense_ffn_engine": "htp"`, `"lmhead_engine": "htp"`
   (`moe_engine htp`, `moe_layer_dtype QS2CX_WH`, `fc_layer_dtype Q4_0`,
   `model_tensor_type Q4_0-FP32`, `fc_wh_file_name` / `fc_wh_format` as P4's
   `gemma4_26b_fcwh`). No `fc_layer_dtype QS4CX` (no weight conversion; the dummy has no
   f32 source). `init_seq_len 1024` for the 1023 prompt. A = the same file without the
   four keys (today's hybrid). C = 16 throughout (their §2: C ≥ 32 fails on their
   address space; ours rule 71).

Contract §2 respected: three walls untouched (the MoE call, DMA ring, pool are ours),
no CPU fallback for `QS4CX_WH`/`QS2CX_WH`, arena budget read every cell. Doc 45 §3:
their new DSP kernels ship with host checks (5), `_det` stays before every quantizer
(their `norm_rows_in` feeds the same u8 quantizer).

## 4. Steps

1. **Merge** on `htp/260-4415-prefill`: `git merge --no-commit refs/pr/4415`; apply §3
   rules 1–4 (ours-wins files via `checkout --ours`, then the additive lines: their
   `7e2ecdfa7` `if` in `quantize_stream.cpp`, `act` in `moe_layer_host_check.c`,
   `HVX_GLU_*` alias). Gate: rung 0 + `ninja -C build` + the 7 `*Lfm2Moe*` gtests + the
   Gemma4 gtests (ours; theirs' `unittest_causallm_gemma4_moe*.cpp` dropped if they need
   their fixture) + `run_host_checks.sh` `ALL CHECKS PASS` with the five new OK lines.
   ≈ 1 day.
2. **Backend union** (§3 rules 5–6): `get_or_register_fc` lookup order, `lm_head` placement
   lookup, MoE signature `w_bits` + their trailing params. Gate: rung 1 full incl.
   `run_inproc_e2e.sh` every existing line unchanged (`INPROC E2E PASS`). ≈ 0.5 day.
3. **Gemma graph + E2E name map** (§3 rules 2, 7). Gate: the gemma64 lines —
   `E2E gemma64 tokens off==cpu 8/8`, `E2E fwd gemma64 e3 calls/token=1.00 attn_caches=2`,
   `E2E tokens gemma64 e3==off 8/8 expected_mismatch=0`, `E2E e3 pool C=2 gemma64
   bit_identical=1` — the hd64 fixture runs their fused layers at prefill on the host
   stand-in, so this is the port's real gate. ≈ 1–1.5 days. **If this step overruns by a
   day: ship steps 1–2 + 4–5 and measure A and N only (the issue's "partial port"
   deliverable); E becomes a follow-up issue with the failing op named.**
4. **Rung 2** both arches: `test/htp/build.sh` (v79) and `HEX_ARCH=v81`, `UNDEFINED
   SYMBOLS OK`, `ARCH OK`; new skel md5 recorded. Rung 3: `build_android.sh --htp`
   (no `--cache` after an IDL change: their §9.2 trap = our rule 3), `md5.txt`,
   `unittest_hvx_attn_f16 --gtest_filter='*PrefillWideHeads*:*PrefillGemma4LayerShapes*'`
   staged for the device. ≈ 0.5 day.
5. **Device sitting (unavoidable)**, `R3CY205ZMND`, `.sitting.lock`, reboot, cool start
   per G block (zone0 ≤ 35 °C), md5 == staged, `NNTR_NUM_THREADS=8`, no `--profile`
   build for tok/s. Variants (≤ 4):
   * **A** — unchanged reference: today's hybrid (MoE on the HTP via the pool C 16, FC /
     attention / norms / head on the CPU), config without the four keys.
   * **N** — all-NPU prefill, hybrid decode (their M=1 CPU rows + our pool): the four keys.
   * **E** — all-NPU prefill + one-PD decode: N + `NNTR_HTP_E2E=1`.
   * **R** (only if time): N with `router` on the DSP — not in this plan's code; skip.
   Grid: {A, N, E} × prompt {447 = `sample_input`, 1023 = their Ardley prompt from doc 57
   §9.13 / `g4-npu-1024` config, pasted into `sample_input`} × G {64, 512, 1024} = 18
   runs ≈ 2 h incl. loads. Order: A447 ×3G, N447, E447, then 1023 the same, A447 G64
   again at the end (drift). Per run: prefill tok/s, decode median + last-64, per-kind
   DSP lines (`NNTR_HTP_PROFILE=1` on a second G64 run only), `calls/token`, peak RSS,
   `mapped` MiB, `heap_used_kib`, S1 ceiling (`unittest_hvx_two_sessions`), zone0.
   Gates at the device: E tokens == A (plan 130 §3.5), N / E prefill MoE dumps == A,
   A within −5 % / 2 % of P4's A16. **Expected vs theirs**: their 164.2 TPS at 1023 /
   2.96 TPS decode were on `R3CY10WM83Y` with QS4CX-file FCs and the per-layer LRU
   (102 misses × 2 ms). Prefill: N should land near theirs (same kernels; our CPU router
   ≈ 0.4 s vs their DSP router 0.65 s at 1023 → slightly above; the 447 cell has no
   counterpart). Decode: A ≈ 18–19 and E ≈ 15–16 at G 512 (P4's numbers, prefill-
   independent) — 5–6× their 2.96, because the one-PD pool serves misses inside the
   token and FC / DENSE_FFN read the arena sidecar, not 85 FastRPC calls. ≈ 0.5 day.
6. **Docs + PR** (§6). ≈ 0.5 day. Total ≈ 4 days; 2.5 to the partial deliverable (A + N).

## 5. Risks

* **Address space / DSP heap** (rule 71): their prefill attention tiles on the DSP heap
  per call (their §9.10: resident KV impossible for Gemma), beside our two `attn_m1`
  caches (400 + 40 MiB) and the pool; their lm_head placement (396) and FC images.
  The handoff's `mapped` / `heap_used_kib` / S1 ceiling columns per cell show where C 16
  sits; the fallback ladder is §3 rules 5–6, then C 8 (S5-0: flat within 3 %).
* **KV cache in rpcmem** (their `7ebacefa0`): their `installKVCacheSharedAllocator` must
  run for `Gemma4MoECausalLM` too (it overrides `load_weight`, not
  `allocateAndBindKVCache`); if the cache stays on the heap, every attention call copies
  (their §9.10 #4, +0.2–0.6 s) — visible as N's prefill well under 150 TPS at 1023 with
  `attention` dominating `prefill_timeline.py --by-op`.
* **Their §9.10 list, re-checked on our tree**: #1 dense registration by name (their
  `transformer.cpp` change, taken); #2 scalar Kᵀ (their vshuff, taken; `ProbeLayoutsMatchHexkl`
  on the device); #3 lm_head placement (rule 6); #4 KV heap (above); #5 / #6 router / RoPE
  rows through ION (`780dca95a`, taken); #7 in_norm on the CPU (kept, ours). Their
  "broken output" of §9.9 was never attributed in code — the dummy cannot show it; the
  gemma64 fixture (step 3) and the MoE dumps are our only accuracy reading.
* **Thermal / DVFS between sittings**: the grid is one sitting, A first and last; the
  1023 cells are new (no prior A at 1023 on this unit) — read N / E against the same
  sitting's A1023 only. DMA rate differs per unit (rule 34: `R3CY205ZMND` 31.2 GB/s) —
  no cross-unit comparison to their `R3CY10WM83Y` numbers beyond the order of magnitude.
* **Stale skel**: the IDL grows by 5 methods; `AEE_EBADPARM` on the first fused call
  = the device's skel predates the merge (rule 3); md5 line in the handoff.
* **Profile gap**: their fused calls carry no DSP stage timers (`dsp≈0`); per-kind lines
  exist for the token (E) and the MoE / plain FC calls only — recorded, not fixed here.
* **Host-vs-device**: the hd64 fixture exercises hd 64 / global 128; hd 256 / 512 fused
  RoPE and `attn_f16` multi-block are device-only (their gtests) — a wrong-head-alignment
  slice (rule 5) shows as `AEE_EBADPARM` / garbage at the device, not on the host.

## 6. Docs to update

* `docs/measurements/260-4415-prefill-port.md`: artifacts (md5, both skels), the 18-run
  grid, per-kind tables, RSS / arena / heap, the FC-requant label on every N / E cell.
* `BENCHMARK.md`: Gemma row — first prefill cells with the all-NPU path (447 / 1023,
  A vs N), E decode re-read; Method paragraph (cycle n: #4415 merged, FC requant label,
  no "now" move — lever cells until the real `QS4CX` file); Log rows per variant;
  Artifacts rows (skel v79 / v81, app set, sidecar reused).
* `LEDGER.md`: cycle entry; rule: "their DSP router is slower than the CPU's at 1023
  (21.6 ms/call) — keep `routeSoftmax` until the register-blocked kernel is measured";
  §2 verdict row for #260; open items: router-on-DSP lever (R), fused-call DSP timers,
  ⑧ reconcile list (hd-512 `attn_f16` fp16 I/O, Q / out in ION — their §9.14); §3a note
  on the `act` IDL arg and the GeGLU bit alias.
* Contract §1 References: `refs/pr/4415` as the prefill reference (already); add the
  merge commit once landed. `docs/htp_moe/guide` (03-performance) prefill section after
  the sitting.

## 7. Revision 2 (2026-10-08 evening)

Supersedes §3–§5 where they differ. Inputs: the implementer's audit on #260 (the new
26B file `nntr_gemma4_qs4cx_fc_arm.bin`, 12.83 GB, is in **#4415's tensor layout**;
`htp/260-4415-prefill` @ `bd544ead6` = 8 commits on the merge of `refs/pr/4415` @
`86bb496b6`, kept as evidence — host checks pass, gtests 122/122, both skels build,
inproc E2E fails untriaged) and the user's rules of 2026-10-08: (1) #4415's prefill
**unchanged** — their MoE layer, DSP router, norms-in-call, per-layer LRU, GeGLU
arithmetic, FC registration, hooks; (2) our one-PD decode **on top**, optimization on
the decode side only; (3) no data-type conversion of the weight files. §3's "ours wins
on `lfm2_moe_layer.cpp`" and the §3 rules 1, 3 (GeGLU bit), 5 (sidecar first) are
withdrawn; rules 4, 6, 7 survive in the form below.

### 7.1 (A) The base and what of ours comes in

**Base = `refs/pr/4415` @ `86bb496b6`**, branch `htp/260-r2-4415-base` off it; `git merge
htp_decode` with the tie-break reversed against §3: **theirs (= the base, `--ours` on that
branch) wins every prefill-path hunk**, our side enters only as decode-side members and
LFM-only files. Verified on both trees (`git diff --stat refs/pr/4415 htp_decode`):

| file | resolution |
|---|---|
| `models/lfm2_moe/lfm2_moe_layer.{cpp,h}` (theirs: `RouterType`/`InNorm`/`RouterNorm`/`OutNorm`, `router_logits_fp32`, `g_expert_lru` + `expert_lru.h`, read-ahead) | **theirs, byte for byte, as `lfm2_moe` — the Gemma MoE layer.** Ours (`MoERouter`, `routeSoftmax`, `w_bits` / 2-bit `moeReadExpertWeight`, `set_decode_moe_experts` hand-over at `:1080–1097`, diff / shadow) comes in **renamed**: type `lfm2_moe_pool`, class `Lfm2MoePoolLayer`, file `lfm2_moe_pool_layer.cpp`; `lfm2_moe_causallm.cpp:140` creates it; `transformer.cpp:479,488,495,553,569`'s `getType() == "lfm2_moe"` become one `isMoeLayer()` helper that accepts both (load-path only). The 7 `*Lfm2Moe*` gtests and every LFM inproc line run on it unchanged |
| `htp_backend/hmx/hexkl_mm_u8i4_moe.{c,h}` | theirs (layer kernel, `HEXKL_MOE_FLAG_GELU_TANH 0x80000`, IDL `act`) **plus** our ours-only functions, additive: `hexkl_mm_u8i4_fc_m1_run` (+`HEXKL_FC_M1_MAX_PARTS`, P4's token FC), `hexkl_moe_expand_chunk` (2-bit, LFM), `hexkl_dma_lane_push2d`, `hexkl_dma_ring_drain`. Our `HEXKL_MOE_FLAG_GEGLU 0x200000` and the DMA_Q field (#177, bits 19–20) go: the token driver ORs their bit 19 into the resident MOE op; DMA_Q returns at bits ≥ 22 only if the M=1 feed reads it (`moe_opts_host_check` decides), never for the prefill kernel |
| GeGLU: `swiglu_det.h`, `hvx_swiglu_det.h` | **theirs for both prefill and the decode token** (`t = g (C0 + C1 g²)`, `GEGLU_DET_C0/C1`): one inline serves the layer kernel, the dense GeGLU and the token's MOE op, so a second order would be a second kernel. `test/htp/host/geglu_host_check.c` (ours-only) stays and is re-gated against their `geglu_det_one` (its SPEC part is f64, its KERNEL part the scalar twin — both pass on their order; the `run_host_checks.sh` silu-mutant must still fail). The implementer's `78fa63243` (probe to our order) is **not** carried; `unittest_hvx_softmax`'s GeGLU reference = theirs |
| `htp_compute_ops.cpp` | theirs for `invokeFc`, `get_or_register_fc/qs4cx/wh`, `gemm_q4_0_batch_norm_fp32`, `router_logits_fp32`, `invokeMoeLayer`, `lm_head_q4_0_prepare`, attention — i.e. no `prefillRows()` / `NNTR_HTP_PREFILL_ROWS` (#225 / #236 `fcRowStep`: their one call per prefill stands). Ours, additive: the one-PD token driver (`PoolServer`, `poolServe/Answer/Harvest/Sync/Refresh`, `set_decode_moe_experts`, `e2eStart` one-PD form replacing their `h2`/`graph2` two-session state — "one PD only", 2026-10-01), `add_decode_graph_q4_0/qs4cx` + `bindQ4m1` (the implementer's `bd544ead6` is the right shape: a QS4CX weight binds the WH handles their `get_or_register_qs4cx` → `htp_qs4cx_from_packed` makes, never Q4M1), `set_decode_graph_desc/param`, `set_fc_wh_file` + `fcwhFind`/`registerFcWh` (LFM's sidecar; a 3-line lookup ahead of their order, a no-op without `fc_wh_file_name` — Gemma sets none), the 2-bit arena entries. Our `set_moe_geglu` is dropped (their per-call `act`) |
| `compute_ops.h` | union: their virtuals (`accelerates_qs4cx_at_m1`, `supports_router_logits_fp32`, `lm_head_q4_0_*`, `gemm_q4_0_batch_norm_fp32`, kv_q) + ours decode-side (`set_decode_moe_experts`, `set_decode_graph_param`, `has_decode_graph_q4_0`, `set_fc_wh_file`, `add_decode_graph_qs4cx`) |
| DSP decode sources (`hexkl_graph.c` +568, `hexkl_token.c`, `hvx_attn_m1_f32.c`, `hvx_m1_ops_f32.c`, `htp_graph_desc.h`, `nntr_hvx_token/dspq/mailbox.c`, `hvx_expand_i2i4.*`) and their host checks (`graph/token/m1_ops/attn_m1_host_check.c`, `run_inproc_e2e.sh`) | ours (nothing of theirs' prefill lives there). Name clash `hvx_softcap_f32`: **ours** (the E2E LM_HEAD's, worker-pool form) is renamed `hvx_softcap_m1_f32` this time, theirs keeps its name (reverse of `188c491ff`); the libc-import allowance of `188c491ff` and the in-process skel list of `f709168b4` are re-applied as is |
| `nntr_hvx.idl` | theirs (108 methods) + our 3 ours-only (`expand_i2i4`, `weight_register_u2i4_arena`, `weight_swap_batch_u2i4_arena`); skel md5 changes (rule 3) |
| `quantize_stream.cpp` | theirs for `Gemma4MoePlan` / `writeGemma4Moe` (`_pre_ffn_norm_2`, `_router_norm` scale-folded, `_router`, `_per_expert_scale`, experts, `_post_ffn_norm_2` — `quantize_stream.cpp:1079`) and the QS4CX FC writer; ours additive for the sidecar, palette, `QS2CX_WH`, the order guard |
| `gemma4_causallm.{cpp,h}` | theirs whole (`createMoe` at `:555` with `in_norm`/`router_norm`/`out_norm`/`cache_experts`, `ENABLE_MOE_BLOCK`, the engine keys, `FOLD_OUTPUT_NORM`). Not virtual, not throwing |
| `gemma4_moe_causallm.{cpp,h}` (ours) | shrinks to a decode-side subclass of their `Gemma4CausalLM`: `setupParameters` (NNTR_HTP_E2E, their `MOE_ENGINE`/`MOE_LAYER_DTYPE`), `load_weight` (the E2E list, §7.2), `repack_weight` (`finish_decode_graph_q4_0`). `createFeedForwardBlock` / `createMoe` overrides deleted; `main.cpp:303` keeps the class for the 26B |
| `mha_core.cpp` | theirs for `forwarding` (incl. `9b6d59a09`), the RoPE table (`rope_rows`, their `faf832c0f`), int8 KV opt-in (`kv_cache_quant` unset → fp16 master). Ours only inside `htpDecodeAttention` (`:661`, decode row): the any-head-dim `build_rope_table` for the ROPE op (hd 256 / 512) built with `use_rope` off (`b1f984049`'s 4 lines) |
| `causallm_common_properties.h`, `nntr_config.json` (26B), fixtures, `unittest_causallm_gemma4*.cpp` | theirs + our `MoERouter` (for `lfm2_moe_pool`); the 26B config = theirs' keys (§7.3); fixtures §7.3 |

Rejected: one MoE class (theirs + our hand-over + `w_bits`). It puts `w_bits` into their
`gemm_qs4cx_moe_layer_fp32` signature and the 2-bit read into their prefill layer (rule 1),
and re-gates the 7 `*Lfm2Moe*` gtests and the `2bit lfm25` lines on their code. Two classes
cost one rename and one helper.

### 7.2 (B) The decode graft

* **Hand-over LRU → pool = what ours already does, grafted into their layer** (~20 lines,
  decode side): at the first accelerated call of each virtual layer,
  `ops->set_decode_moe_experts(all, [](need, load, evict){ g_expert_lru.acquire(need, load,
  evict); })` (ours `:1080–1097`). The pool's slots **are** the LRU's registered slots
  (`experts_` / `handle_cache_` → `poolSync` tables; a miss inside the token: S1 asks, the ARM
  runs `pool_fn_` = `ExpertLru::acquire`, `releaseExpert` frees the slot, S1 reads into it,
  `poolRefresh` touches the routed keys). One slot set, 1.38 GB at C = 16 (30 × 16 × 2.9 MiB
  4-bit), not two: `mapped` MiB per cell is the check. Eviction ownership stays with
  `ExpertLru` on both sides; the next prefill / turn runs their per-layer `acquire` +
  read-ahead as today and sets `pool_dirty_`, so the next token re-sends the tables
  (ours `:1492–1502`). `reserve_qs4cx_wh_expert_slots(capacity)` stays theirs.
* **Decode-row hooks in their MoE layer**: the resident check (`htpDecodeRouter`, theirs
  `:1655`) moves ahead of `normRows(in_norm)` / `normRows(router_norm)` / `routerLogitsOnAccelerator`
  inside `total_tokens == 1` only — prefill never reaches it. Their other hooks stay; the
  `b1f984049` relaxations (qkv returns after the input norm when resident and before its
  RoPE; a fused `residual_add` row skipped when resident; the tied head's LM_HEAD hook fires
  with `in_norm`/`softcap` when the E2E graph is set) are re-applied — each is inside the
  `to - from == 1 && htpDecodeRowResident(from)` branch.
* **E2E list on their weight names** (`htp_graph_gemma_build` unchanged; `load_weight` keys
  every weight as `<layer>:<weight>` and under the owning layer, `67bd698a7`): RMSNORM
  `_qkv:in_norm_gamma`, `_post_attention_norm:gamma`, **`_sparse_moe:in_norm_gamma`,
  `_sparse_moe:out_norm_gamma`**, `_ffn:in_norm_gamma`, `_ffn:out_norm_gamma`,
  `_post_ffn_norm:gamma`; tail `output_of_causallm:gamma`; QK_NORM `_qkv:q_norm_gamma |
  k_norm_gamma`; ADD scalar `_post_ffn_norm:scalar_multiplier`; **ROUTER_TOPK: W =
  `_sparse_moe:gate` [H×E], BIAS = `_sparse_moe:router_norm_gamma` | `_sparse_moe:expert_bias`
  — the gamma as is (scale and H^-0.5 already folded in their file; our `rs[f] * hs` goes)**;
  FC `_qkv:qweight/kweight(/vweight)`, `_attention_out:weight`; DENSE_FFN `_ffn:up, :gate, :down`
  by name (gate-first file, up-first list); LM_HEAD `embedding0:Embedding` (tied).
* **FC / DENSE_FFN at M = 1**: QS4CX weights bind the WH handles their load-time
  `register_qs4cx_weight` (`transformer.cpp:570`) made — `add_decode_graph_qs4cx` →
  `get_or_register_fc/dense(key, K, N, scale)` (`bd544ead6`), read by `hexkl_mm_u8i4_fc_m1_run`
  in the token. Their own M = 1 answer is the hybrid's: `accelerates_qs4cx_at_m1() = true`
  sends a QS4CX FC row through `gemm_q4_0_batch_norm_fp32` (64-row pad, one call per FC) —
  that is A's decode, not E's. One image set; no Q4M1 for QS4CX (contract §2). #258's u8
  per-row activation stays the known accuracy term of this path.
* **LM_HEAD**: Q4M1 from the Q4_0 embedding (`add_decode_graph_q4_0(tied)`), 396 MiB. Their
  `lm_head_q4_0_prepare` would place a second copy → **E runs `lmhead_engine: cpu`** at step 1
  (last row on the CPU, ≈ 18 ms a prefill, their §9.12); one lookup (§3 rule 6) later.
* **ATTN_M1 seed**: their `htpDecodeAttention` already seeds from the CPU-side fp16 cache
  (`mha_core.cpp:696`, `htpDecodeKvSeed`); their rpcmem allocator keeps it host-readable
  (`causal_lm.cpp:153,185`, runs for the subclass since `load_weight` calls the base). Only
  with `kv_cache_quant` unset: **q8 is an A-only control cell**, E never runs on q8.
* **Softcap**: `shape.softcap = TIE_WORD_EMBEDDINGS ? FINAL_LOGIT_SOFTCAPPING : 0` (the DSP
  caps; their tied head folds it) — `b1f984049`'s line; the §3 rule 7 ponytail closes.

### 7.3 (C) Gates, fixtures, config, sitting

* **LFM lines unchanged**: `run_inproc_e2e.sh` lfm25 / hd64 lines (`tokens off==cpu 8/8`,
  `e3==off`, `2bit lfm25 == palette-twin bit_identical=1`, `pool C=1/C=2`) and the 7
  `*Lfm2Moe*` gtests on `lfm2_moe_pool`, compared line for line with the pre-merge run
  captured on `origin/htp_decode` (the implementer has it). See (D) for the qkv fold.
* **Gemma hd64 fixture re-written in their layout** (`router_norm` folded, between
  `_pre_ffn_norm_2` and `_router`): our `generate_gemma4_moe_reference.py` keeps its shape
  flags (`--head-dim 64 --global-head-dim 128 …`, `run_inproc_e2e.sh:266`) and writes
  `router_norm = router_scale · H^-0.5` in place of `router_scale` (one block); their
  `unittest_causallm_gemma4_moe*.cpp` take it as the order check. FCs of the fixture as
  `QS4CX` (`--fc_dtype QS4CX`, their writer) so the gemma64 lines exercise the QS4CX bind;
  the `2bit gemma64` lines are dropped (Gemma's file is 4-bit; the 2-bit path is gated on
  lfm25). Gate lines: `E2E gemma64 tokens off==cpu 8/8`, `E2E fwd gemma64 e3
  calls/token=1.00 attn_caches=2`, `E2E tokens gemma64 e3==off 8/8`, `E2E e3 pool C=2
  gemma64 bit_identical=1`, plus their five (`RMSNORM/ROUTER/ROPE ROWS OK`, `SOFTCAP OK`,
  `tile_f16_host_check`). A 26B-only host load line is not a substitute (no x86 load of
  12.83 GB).
* **Rung 2**: `test/htp/build.sh` v79 + `HEX_ARCH=v81`, `UNDEFINED SYMBOLS OK`, `ARCH OK`;
  md5 recorded; stub regenerated (IDL changed). Rung 3: `build_android.sh --htp` without
  `--cache`.
* **Config for the sitting** (`nntr_gemma4_qs4cx_fc_arm.bin` + its `nntr_config.json`):
  theirs' keys — `fc_layer_dtype QS4CX`, `moe_layer_dtype QS4CX_WH`, `moe_engine htp`,
  `moe_cache_experts 16`, `attention_engine / attn_proj_engine / dense_ffn_engine htp`,
  `lmhead_engine htp` for A, `cpu` for E (above), `init_seq_len 1024`, no `attention_kv_dtype`.
  No `fc_wh_file_name` (rule 3: no conversion; QS4CX → WH is `htp_qs4cx_from_packed`, a
  lossless repack).
* **Sitting** (S25 `R3CY205ZMND`, `.sitting.lock`, reboot, cool start per G block, md5 ==):
  **A** = pure #4415 hybrid (their LRU decode; the implementer's step-1 numbers in
  `docs/measurements/260-step1-4415-pure.md` are A's first reading, re-run in E's sitting);
  **E** = their prefill + our one-PD decode (`NNTR_HTP_E2E=1`); prompts 447 / 1023; G 64 /
  512 / 1024; **q8** = A with `attention_kv_dtype q8`, 447 and 1023 at G 64, prefill column
  only. 12 + 2 runs. Gates: E prefill MoE dumps == A `bit_identical=1`; E tokens == A under
  plan 130 §3.5 (near-tie flips; A's decode FCs are their 64-row QS4CX call, E's the WH M=1
  — not bit-identical by construction); E prefill ≥ −5 % of A's (same code); per-kind DSP
  lines, `calls/token = 1.00`, peak RSS, `mapped`, `heap_used_kib`, S1 ceiling, zone0.
  Expected: A decode 3–5 tok/s (their §9.13 / §9.16, miss-bound), E ≈ 15–16 at G 512 if P4's
  F16 carries over (same token path, same 789 MiB of WH FC bytes).

### 7.4 (D) The LFM2 side effect

Their `qkv_layer.cpp:389–392` takes the fused DSP call whenever `(in_norm || feature_size)`
and `rows > 1`: LFM2.5's per-head q / k norm (`feature_size`, no `in_norm`) moves from the
CPU into the call at prefill, which is the logprob move the audit saw. **Keep LFM on the old
path**: one opt-in property on `qkv_layer` (`norm_in_call`, default `true`; LFM2's
`lfm2_causallm.cpp:100` sets `false`) — 6 lines, default-preserving, so their Gemma prefill
is untouched and the LFM lines stay byte-identical. Accepting the move would re-baseline
every LFM host line and leave the frozen LFM device rows (cycle 36) on a different prefill
numerics than their reference run.

### 7.5 (E) Risks

* **Two expert caches**: not two by design (§7.2); the proof is `mapped` ≈ pool 1380 + FC 789
  + head 396 (+ `attn_m1` 440) per cell, and the token driver's "the ARM's pool and S1's
  table must agree" throw if the LRU and S1 diverge. Address space (rule 71, 3 840 MiB on
  this unit): ≈ 3.0 GB + their attention tiles on the DSP heap per call + DSP heap ≈ 0.6 GB
  — C 16 sits at the edge; ladder C 12 → 8 (S5-0: flat within 3 %), `lmhead_engine cpu` for
  E keeps the head placed once. An int8 KV copy is never beside it (q8 is A-only).
* **Their §9.10 regressions carried**: #1 dense registration by name (`transformer.cpp`,
  taken), #3 lm_head placement at load, #4 KV in rpcmem (`installKVCacheSharedAllocator`
  — verify once on the host that the subclass reaches `causal_lm.cpp:185`), #5 / #6 router /
  RoPE rows via ION, #7 in_norm on the CPU (theirs). Their §9.9 "broken output" was never
  attributed; the dummy cannot show it — the hd64 fixture and MoE dumps are the only reading.
* **Their head still moving**: pinned at `86bb496b6`; 14 commits past plan v1 (int8 KV /
  A8W8 opt-in, `9b6d59a09` fp16 staging skip). Later commits are folded at ⑧, not here.
* **Stale skel** (rule 3): IDL = theirs + 3; `AEE_EBADPARM` on the first fused call = the
  device's skel predates the merge; md5 line in the handoff. **DVFS / thermal**: A first and
  last, cool start per G; **DMA rate** per unit (rule 34: `R3CY205ZMND` 31.2 GB/s) — no
  cross-unit comparison with their `R3CY10WM83Y` numbers beyond the order of magnitude.
* **Host-vs-device**: hd 256 / 512 fused RoPE, hd-512 `attn_f16` multi-block and the WH M=1 FC
  on 26B shapes are device-only; the host gemma64 lines cover hd 64 / 128.

### 7.6 Steps and size

1. **Base** (≈ 1.5 d): branch off `refs/pr/4415`, merge `htp_decode` per the §7.1 table
   (theirs-wins on the prefill path; `lfm2_moe_pool` rename + `isMoeLayer`; token driver,
   `add_decode_graph_qs4cx`, DSP decode sources, IDL +3, `norm_in_call`, GeGLU theirs,
   `hvx_softcap_m1_f32`). Gate: rung 0, `ninja -C build`, `*Lfm2Moe*` 7/7, their + our
   `unittest_causallm_models`, `run_host_checks.sh` `ALL CHECKS PASS` with their five OK
   lines and `geglu_host_check` on their order, **every LFM inproc line == the captured
   pre-merge run**.
2. **Gemma decode graft** (≈ 1.5 d): §7.2 (hand-over + resident check in their layer, the
   name map, the hooks, softcap, `lmhead_engine cpu` for E), the fixture in their layout
   (§7.3). Gate: the four gemma64 lines + rung 2 (v79 + v81) + rung 3, md5s staged.
3. **First deliverable — device (unavoidable), ≈ 0.5 d: E at 447, G 512, with A 447 G 512
   in the same sitting** (cool start, md5 ==): the gates of §7.3; the number goes to #260 and
   `docs/measurements/260-r2-e2e.md` as a lever cell labelled "QS4CX file, WH M=1 FC (#258)".
4. **The grid** (≈ 0.5 d sitting + 0.5 d docs): the remaining 12 runs (A / E × 447 / 1023 ×
   G 64 / 512 / 1024, the two q8 prefill controls), BENCHMARK Gemma row (prefill cells for
   A and E at 447 / 1023, E decode re-read on the 4-bit QS4CX file), LEDGER cycle entry
   (rule: "two MoE layer classes — their `lfm2_moe` for Gemma prefill, our `lfm2_moe_pool`
   for LFM; the pool is the LRU's slot set"), §2 verdict row, open items (lm_head once,
   their DSP router as a lever cell, ⑧ reconcile list, #258), contract §1 References
   (merge commit). **≈ 4 days to the first deliverable, ≈ 5 to the grid.** If step 2
   overruns by a day: ship steps 1 + the A sitting (their prefill numbers on our base) and
   file E as the follow-up with the failing gemma64 line named.
