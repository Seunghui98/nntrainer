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
