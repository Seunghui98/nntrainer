# 234 — `htp_decode` stays the Gemma base: what to bring over from upstream #4408 and from `htp_first_version`, in what order, and what each port costs

Issue: dlwlzzero/nntrainer#234 (p1, `state:in-progress`). **Revision 2,
2026-10-07.** Base `htp_decode` @ `9323ad52d` (P1 = PR #247 and P2 = PR #248
merged). Sources: upstream nntrainer/nntrainer#4408 (`refs/pr/4408`, head
`28771a928`, unchanged) and `origin/htp_first_version` @ `77a01902f`, which
the user declared **finished** on 2026-10-07 (last merge PR #246): 51
commits over `htp_decode` (13 merges, 38 own; 15 touch code). Plan 229
(`229-ternary-lut-decode.md`) is the format plan this one feeds; plan 201
the structure.

**User, 2026-10-06:** the goal is the ternary 26B-A4B decoding end to end
on the one-PD path; no tok/s target; the 2 GB peak-memory constraint is
considered **last** (plan 229 S7). **User, 2026-10-07:** `htp_first_version`
is closed; re-survey all 51 commits and port what the Gemma work needs.

Evidence tags: **[C]** read on `htp_decode` @ `9323ad52d` at `path:line`;
**[A]** `git apply --check` of the commit against that tree, run in this
session (dry run only — nothing built or run here); **[D]** read in
`git diff origin/htp_decode origin/htp_first_version` (the real delta — the
#237 sync carried `htp_decode` code back, so per-commit stats overstate it);
**[M-4408]** / **[M-LFM]** as before; **[G]** arithmetic.

## 1. Goal and gate

Acceptance (issue): "which commits from each source come to `htp_decode`,
in what order, expected conflicts with the S4 code, and the host gate (LFM
and Gemma E2E lines unchanged, `INPROC E2E PASS`)". Measurable form, per
port PR (every PR runs the whole set; the "new" column is what that PR adds):

| gate | where | pass |
|---|---|---|
| LFM lines unchanged | `test/htp/host/run_inproc_e2e.sh` | `INPROC E2E PASS`; every `bit_identical=1` line of the header block (`run_inproc_e2e.sh:19-166` [C]) still prints; `E2E e3 pool C=1 lfm25 / C=2 … == e3 bit_identical=1` (`:544`) with the same `misses=` |
| Gemma lines unchanged | `:391-425`, `:575` [C] | `E2E gemma64 tokens off==cpu 8/8`, `E2E fwd gemma64 e3 calls/token=1.00 attn_caches=2 timeouts=0 ok`, `gemma64-e3 … min_snr_db` ≥ 20 (27.63 at cycle 34), `E2E e3 pool C=2 gemma64 == e3 bit_identical=1` |
| 2-bit lines unchanged (#238 / #239) | `:593`, `:607` [C] | `E2E 2bit lfm25 / gemma64 e3 == palette-twin bit_identical=1`, `E2E 2bit pool C=1 / C=2 … bit_identical=1` with the same `misses=` |
| keys line unchanged (P1) | `:641` [C] | `E2E keys lfm25 prefill-moved=1 htp_fc_rows=4 e3==e1 bit_identical=1 pool C=2 bit_identical=1 misses=15 … q4m1_handles=23 cpu-fc-skipped=70 per_token=10 ok` — same numbers |
| HF differential | `test/unittest/models/unittest_causallm_gemma4_reference.cpp` [C] | `unittest_causallm_models` all pass (114 / 114 at P2), 0 skipped |
| host checks | `test/htp/host/run_host_checks.sh`, `tools/htp_syntax_check.sh`, `*Lfm2Moe*`, `*qs4cx*` | `ALL CHECKS PASS`, `WORKER POOL LANES OK`, exit 0, 6 / 6, 2 / 2 |
| standing | prefill ≥ −5 % of the same sitting's A; text identical to the CPU run; `QS4CX_WH` has no CPU fallback | no device cell in this issue — the ports are host-gated; the first device reading of all of them is plan 201 S5 / plan 229 S4 |

Host state at the start of this revision: `INPROC E2E PASS` on `htp_decode`
@ `9323ad52d` per PR #248's comment (477 / 478 lines identical to the #247
run, the odd one a wake-split timing line). Not re-run in this session.

## 2. Where it lives

### 2.1 The 51 commits of `htp_first_version`, classified

By PR, oldest first. "already-in" = present on `htp_decode` by patch-id
(`git cherry`) or as a rebased port; "skip" = not needed by the Gemma work
or a docs hunk the supervisor mirrors (§6). The real code delta [D] is 15
commits in six files that matter: `quantize_stream.cpp`,
`htp_compute_ops.cpp` (+954 / −… net), `hexkl_graph.{c,h}`,
`hexkl_mm_u8i4_moe.{c,h}`, `hexkl_conv_block.{c,h}`, `htp_wh_layout.h`,
`htp_graph_desc.h`, `transformer.{cpp,h}`, `compute_ops.h`,
`lfm2_moe_layer.cpp`, `nntr_hvx_mm_u8i4.c`, the host checks and
`run_inproc_e2e.sh`. **No IDL change** (`test/htp/nntr_hvx.idl` differs
only by the 2-bit entries `htp_decode` has and `htp_first_version` lacks [D]).

| PR (there) | commits | verdict | Gemma relevance [C] |
|---|---|---|---|
| #223 (#222 config of record) | `c0fa4e419`, `d541bda3a` | **already-in** (patch-id: `f3db7c60b`, `6e7544d6b`) | — |
| | `f92ec3596` | **already-in** (rebased as `36f9e05d6`, PR #247) | — |
| | `ddb7d9ad8` (config json + 222 measurement files) | skip → supervisor mirror | — |
| #230 (#225 PR 1, FC WH sidecar, prefill half) | `db2c27a11` quantizer `--fc_wh_sidecar` | **port (P3)**, with a Gemma extension: the flag is gated `is_lfm2_moe && fc_dtype == Q4_0` and only `writeLfm2Moe`'s `writeFc` calls pass `fc_wh = true` (the commit's hunks at its `:354` and `:209-246`); `writeGemma4Moe` (`quantize_stream.cpp:1247` [C]) must flag its `_wq / _wk / _wv / _attention_out / _ffn_up / _ffn_gate / _ffn_down` the same way | **needed**: without a Gemma sidecar the one-PD FC / DENSE_FFN ops stay Q4M1 (plan 229 §3.2 row A, not A′), and plan 229 S2's 2-bit FCs have no file to extend |
| | `7b622617a` loader: `set_fc_wh_file`, `registerFcWh`, `fcwhFind`, `whUnpack`, `FcWhEntry` + `FCWH_FORMAT` in `htp_wh_layout.h`, `fc_wh_file_name` in `transformer.cpp`, `--repack` in `htp_e2e_test` | **port (P3)**; **it carries `whUnpack`**, so #4408's `7dbd876ed` is dropped from P5 | **shared**: `get_or_register_fc` / `_dense` (`htp_compute_ops.cpp:5015`, `:5130` [C]) are what P4's one-PD bind calls for Gemma's FC / DENSE_FFN ops. One Gemma gap: `transformer.cpp` opens the sidecar only for backends with a pending **keyed** FC / dense / conv (its hunk at `repack_weight`), and Gemma sets no engine keys (`FFN_ENGINE` and the #222 keys are read by `lfm2_causallm.cpp:363-377` only [C]) — P4 fixes that |
| | `75f6f007d` prefill in 512-row chunks (`prefillRows()`, `invokeMoeLayer` chunking, conv-block history `hist`, `conv_wLen == 5 C` in `nntr_hvx_mm_u8i4.c`) | **port (P3)** | **half shared**: `invokeMoeLayer` (`:4599` [C]) is Gemma's MoE prefill too (`tryMoeLayerOnAccelerator` → `gemm_qs4cx_moe_layer_fp32`, `lfm2_moe_layer.cpp:771` [C], the layer Gemma hands over to under `moe_engine=htp`), so a Gemma P1024 prefill on the HTP gets the chunking; the conv half is LFM-only and harmless (DSP source change, no IDL change — `conv_wLen` 3 C or 5 C) |
| | `2c60e5215`, `69928f26b` plan 225; `4617545f5` guide / Artifacts | plan file **carried with P3** (the code cites it); guide → supervisor | — |
| #224 (#219, the ARM tier) | `32b1bc6de` the tier, `dbda0fc25` preload names the misses, `cc9b20d0c` E2E tier lines + preload at load | **later, conditional (P6)** | **shared path, Gemma-unsized**: `tier_qs4cx_wh_experts` is called from `Lfm2MoELayer::preloadExperts` (`:578` [C]) for every expert that did not fit, with no cap — on the 26B that is 3 840 − C experts × 2.87 MiB ≈ 10.8 GiB of anon memory at 4 bits (5.4 at 2) [G, plan 229 §7.1]; `posix_memalign` would fail or swap before the first token. Needs a byte cap before it can even start on Gemma; plan 229 §7 says it cannot hold under 2 GB at all |
| | `69c05f517`, `469ecf007`, `149481ec4`, `0761995ff` (219 measurement + cycle 35b) | skip → supervisor mirror | — |
| #233 (#225 PR 2, one-PD decode FC / DENSE_FFN on WH) | `b73e01b1a` `hexkl_mm_u8i4_fc_m1_run` + `fc_wh_det.h` + host check | **port (P4)** | **needed**: the M=1 FC kernel on WH tiles that plan 229 S2's u8i2 FC variant extends (`hvx_gemm_u8i4_wh_col` per column tile; S2 adds the `bits == 2` → `hvx_gemm_u8i2_wh_col` branch, the per-handle `bits` the 2-bit registry already carries, `hexkl_mm_u8i4_moe.c:158, :667` [C]) |
| | `7efd22adf` `HTP_GRAPH_FEED_WH`, `bindQ4m1` WH branch, `graph_wh_flags`, `fcwhLeft` chunk | **port (P4)** | **shared**: Gemma's graph binds FC (q \| k \| v, o) and DENSE_FFN (up, gate, down) from `q4_pending_` (`gemma4_moe_causallm.cpp:227-262` [C]); the WH DENSE_FFN runs the MoE kernel with `env->moe_flags`, so the session's `HEXKL_MOE_FLAG_GEGLU` (`hexkl_graph.c:475` [C]) applies — Gemma's GeGLU dense FFN comes for free; dense inter 2 112 → `denseChunkCols` = 1 056, 2 chunks ≤ `HTP_GRAPH_WH_DENSE_MAX_CHUNKS` 16 [G]; q \| k \| v parts ≤ `HTP_GRAPH_MAX_PARTS` 32 [G] |
| | `c8a0a4792` config of record names the sidecar; `51b2b4477`, `f08cccfeb`, `fc1902ecf`, `5f0a5994f` | skip → supervisor mirror (the `docs/measurements/config/*.json` are LFM artifacts) | — |
| #237 (sync of `htp_decode` into `htp_first_version`) | `da5e2d7d8`, `f11372b9c`, `78dace26f` | **already-in by construction** (they carry `htp_decode`'s S4 back; nothing to port) | — |
| #243 (#236, qkv FC chunks at P1024) | `a3d966164` `fcRowStep(K) = min(fcMaxRows, prefillRows)` on the four FC entries + `fc_layer_host_check` cell | **port (P3)** — clean [A], 57 lines | **LFM hybrid only today**: the four entries (`gemm_q4_0_accel_fp32 :1236`, `_batch :1317`, `gemm_qs4cx_fp32`, `_batch :1381` [C]) run only for a `fully_connected` with engine `htp`, i.e. the #222 keys, which Gemma does not set; Gemma's FC prefill is on the CPU. Ported for parity (the same source on both lineages, one less future conflict) and because it is the P1024 hybrid's fix of record (rule 64 / 67); it becomes Gemma-relevant the day Gemma gains keys |
| | `366c06c57` E2E `lfm25-p2x … fc_calls=` line; `abcd0db26` plan 236; `445b4e987`, `3d61c1aff` (236 handoff / measurement) | E2E line + plan file **with P3**; measurement → supervisor | — |
| #245 (#219 default) | `cdbf66f20` `NNTR_MOE_TIER` unset = 2 under `NNTR_HTP_E2E=1`; `0235d9d79` its E2E line | **not as-is (P6)**: the blanket default would make every Gemma E2E build the tier — see #224 above | LFM-only unless capped |
| | `20711f83a`, `0c991b009`, `fa6db699a`, `6923850d0` | skip → supervisor mirror | — |
| #246 (guide close-out) | `21266c9a5`, `e3ee91cc1` (cycle 36 that side), `d39c30f85` (contract §4.1 S25 adb) | skip → supervisor mirror (`htp_decode`'s cycle-36 row already names the S25 `R3CY205ZMND` as the attached unit, contract §12 [C]) | — |
| #230/#233 era docs | `06ed17b7b` single-base rewrite | **not carried** (issue) | — |

### 2.2 The #4408 side (unchanged verdicts, two done)

| candidate | verdict |
|---|---|
| `9dd2c76de`, `f9c09eaa2` quantizer `.safetensors` + order guard | **done** (PR #248: `4322bd291`, `89cccb137`, `cf1c97b4f`, `25416ed5b`) |
| `1cffe5e25` 26B `nntr_config.json` (file only, rewritten) | **done** (PR #247: `1d434b250`) |
| `7dbd876ed` `whUnpack` | **dropped**: `7b622617a` brings the same function (P3) |
| `a7c527ac2` `NNTR_MOE_DIFF` / `NNTR_MOE_SHADOW` | **port, rebased (P5)**; still fails at `lfm2_moe_layer.cpp:20` [A] |
| `93c2a6ff6` `f33ed210a` `e8d2604ec` `tools/prefill_timeline.py` | **port, optional (P5)**; first clean [A] |
| everything model-side, the QS4CX-FC route, the fixtures / gtests, docs 55–57 | **skip** as in revision 1 (the lineage argument of §3 stands); §10.x findings → LEDGER rules (§6) |

### 2.3 Consumers that move with a changed contract

* **IDL / stub / skel**: no IDL change in any port. DSP sources change in
  P3 (`hexkl_conv_block.{c,h}`, `nntr_hvx_mm_u8i4.c`'s `conv_wLen`) and P4
  (`hexkl_mm_u8i4_moe.{c,h}`, `hexkl_graph.{c,h}`, `htp_graph_desc.h`), so
  rung 2 runs on both, both arches, md5s in the PR.
* **`HtpComputeOps`** (`compute_ops.h`): P3 adds `set_fc_wh_file`; P6 adds
  `tier_qs4cx_wh_experts`. Both are additive virtuals with a `false` / no-op
  default (the CPU backend answers them).
* **The quantizer's format tag**: P3 adds a second output (`<stem>_fcwh.bin`,
  `fc_wh_file_name` / `fc_wh_format = "QS4CX_WH/1"` in `nntr_config.json`);
  the loader check is `FCWH_MAGIC` / `FCWH_VERSION` + `Transformer`'s
  `FC_WH_FORMAT` compare. Plan 229 S2 bumps `FCWH_FORMAT` once for bits.
* **`NNTR_HTP_PROFILE` stage tables / `tools/htp_fc_report.py`**: untouched
  by P3–P6. P4's WH FC / DENSE_FFN ops keep their kind names (`FC`,
  `DENSE_FFN`) in the per-kind lines, so the #225 `prof_Q` breakdown reads as
  before; P5 adds a tool beside them.
* **The loader check for Gemma**: `Gemma4MoECausalLM::setupParameters`'s
  `QS4CX_WH | QS2CX_WH` gate (`:67` [C]) is untouched; the sidecar is keyed
  by the Q4_0 bytes (`fcWhKey`), not by names, so #4296's names need no
  mapping.

## 3. Design

**Chosen: finish the FC WH path first (P3 prefill half → P4 decode half),
each a PR into `htp_decode` gated by the unchanged E2E lines, with the two
Gemma-specific additions inside those PRs (the Gemma writer's `fc_wh` flag
in P3; opening the sidecar for an E2E graph with no keyed FC in P4). Then
the #4408 diagnostics (P5), and the tier only after S5's first miss numbers
and with a cap (P6).** The reason is plan 229 §3.2: the 26B floor crosses
50 tok/s only with the FC set at 2 bits, and 2-bit FCs sit on exactly this
path — the sidecar file (S2 extends its writer), `registerFcWh` (S2 sizes it
by `bits`), `hexkl_mm_u8i4_fc_m1_run` (S2 adds the u8i2 column branch).
Porting P3 and P4 before S2 means S2 extends one tree instead of porting a
diverged one, and the Gemma-specific gaps (§2.1) are found on the host now,
on the tiny fixture, not on the device with the 26B files.

**Rejected: merge `htp_first_version` into `htp_decode` wholesale** (one
merge commit). It would carry the tier and its blanket default (which the
Gemma E2E cannot run), the docs rewrite the issue excludes, and it would
fold the real conflict — `htp_first_version` knows nothing of the 2-bit
`w_bits` API that #238 / #239 put on every path the sidecar and the tier
touch — into one unreviewable hunk. Cherry-picks keep each port's gate its
own, and each port is re-expressed on the `w_bits` API as it lands.

**Rejected: skip #236 (`a3d966164`) as LFM-only.** It is clean, 57 lines,
host-checked, and the P1024 hybrid's fix of record; leaving it out keeps a
known divergence in the four FC entries for no saving.

**The expected conflicts, by file** (what the implementer meets; [A] for
the failing hunks, [C] for the lines on `htp_decode`):

| file | with | what moved |
|---|---|---|
| `quantize_stream.cpp` | P2 (#248) + #238 | `TensorWriter` ctor takes P2's source-tensor argument and #238's `PaletteOptions` / `setPalette` (`:508`, `:558`, `:573` [C]); `writeFc` (`:660`) gains `db2c27a11`'s `bool fc_wh` beside P2's `expectSourceTensor`; `writeFcConcat` (`:720`) is unaffected (experts, not FCs); the help text and the `is_lfm2_moe` gate at main (`:1676`) |
| `htp_compute_ops.cpp` | #238 / #239 (`w_bits`), #211, P1's `56a9e3f3b` | `7b622617a`'s `registerFcWh` / `fcwhFind` go next to `get_or_register_fc` `:5015` / `_dense` `:5130` (4-bit only: the sidecar is `QS4CX_WH/1`); `7efd22adf`'s `bindQ4m1` branch at `:1944`; `75f6f007d`'s `invokeMoeLayer` `:4599`, `invokeConvBlock` `:4756`, `prefillRows()`; `a3d966164`'s `fcRowStep` beside `fcMaxRows` `:1285`. P6 (the tier) collides hardest: `readWeight(…, w_bits, …)` `:5783`, `expertStride(K, N, w_bits)` `:5431`, `codeBytes` `:5438`, `ArenaEntry::pal` `:5300`, `registerStaged`'s 2-bit branch `:5579` — the tier's `tierRead` / `readExpert` must size by `codeBytes(…, d.w_bits) + paletteBytes + 8 N`, not `whBytes` |
| `htp_wh_layout.h` | #238 | `7b622617a`'s hunk after `whPack` (`:99` [A]) lands beside `whBytes2` / `whPack2`; additive |
| `hexkl_graph.h` / `.c` | #239 | the `rebind_fn` carries `pal_gu` / `pal_dn` on `htp_decode` [D]; `7efd22adf`'s WH runner hunks apply clean [A] and are additive |
| `hexkl_mm_u8i4_moe.{c,h}` | #238 / #239 | `b73e01b1a`'s `fc_m1_run` applies clean [A]; the 2-bit `hexkl_moe_expand_chunk` and the `bits == 2` M=1 dispatch (`:158`, `:667`, `:1513` [C]) stay; `fc_m1_run` refuses a 2-bit handle until S2 (state it in the function comment) |
| `run_inproc_e2e.sh` | P1 (#247), #239, S4 | every test hunk fails [A]: the keys block is at `:610-641` [C], the 2-bit block `:580-610`, the Gemma block `:391-425`; the new lines go after the keys line, in the order the commits add them |
| `htp_e2e_test.cpp` | P1 | `--repack` (`7b622617a`) and the tier's preload (`cc9b20d0c`) near the flag parse `:63-100` [C] |
| `moe_layer_host_check.c`, `run_host_checks.sh`, `graph_host_check.c` | #238 (`hvx_expand_i2i4.c` in the link lines `:41-60`, the 2-bit cells), S4 (the Gemma mutants `:294`) | `b73e01b1a`'s FC WH cells and `7efd22adf`'s `GRAPH FC WH` mutant slot in after the existing ones |
| `transformer.cpp` / `lfm2_moe_layer.cpp` / `compute_ops.h` | #238 (`w_bits` instead of `weights_wh`) | `7b622617a`'s `transformer.cpp` hunk is the sidecar block at the end of `repack_weight`'s walk (additive); `dbda0fc25` applies clean [A]; `compute_ops.h`'s two new virtuals are additive |

Contract §2 and doc 45 §3: no wall moves; the arena budget moves with P3 /
P4 (the FC set as WH images in their own chunk, `fcwhLeft`, in place of the
Q4M1 set it replaces; the DSP heap loses the rule 63 copies — plan 225
§3.5's LFM sum 3 691 of 3 840; the Gemma sum is plan 229 §7.1's, re-stated
in P4's PR for the 26B shape: FC set ≈ 790 MiB + lm_head Q4M1 ≈ 415 + pool);
`QS4CX_WH` keeps no CPU fallback (the sidecar is read by the HTP only; the
CPU keeps Q4_0, which is also why the Gemma one-PD RSS carries the 888 MiB
of dead originals plan 229 §7.1 names — S7's business); `_det` before every
quantizer and bit-identity: `fc_wh_det.h` is `fc_m1_run`'s spec
(`FC WH BIT-IDENTICAL` + its mutant), the chunked prefill is proven
bit-identical to the whole call (`lfm25-p2x`), and the Gemma E2E lines are
the hand-over proof.

## 4. Steps

Each step is one PR into `htp_decode`, ending in rung 1 of
`.claude/skills/hexagon-gates` (host checks, the §1 table) and, where DSP
sources change, rung 2 (both arches, md5s in the PR); rung 3 once per PR.
No step needs a device. P1 and P2 are done (PRs #247, #248; their "as
built" notes are in revision 1 and the PR bodies).

* **P3. The FC WH sidecar, prefill half (#225 PR 1 + #236)** — in order:
  `db2c27a11` rebased onto the P2 / #238 `TensorWriter` (the `bool fc_wh`
  argument beside `expectSourceTensor`; lift the `is_lfm2_moe` gate to
  `is_lfm2_moe || is_gemma4_moe` and flag `writeGemma4Moe`'s seven FC
  `writeFc` calls — not the per-layer-input FCs, which are no graph op —
  the help text says "LFM2-MoE / Gemma4-MoE"); `7b622617a` rebased
  (`htp_wh_layout.h` after `whPack2`, `htp_e2e_test.cpp --repack` beside
  P1's flags, `registerFcWh` 4-bit only with a `ponytail:` naming S2);
  `75f6f007d` rebased (`run_inproc_e2e.sh` after the keys line);
  `a3d966164` (clean) + `366c06c57` rebased; plan files 225 and 236
  copied as they are. The commit authors are kept.
  Gate: rung 1 — every §1 line unchanged, plus the new lines as the
  commits print them: `E2E keys fcwh-lfm25 handles=28 arena_kib=… heap=0
  requant=0 main=same heap-path heap_kib=… e3 calls/token=1.00 ok`,
  `E2E eval fcwh==wh-lfm25 … bit_identical=1`, `E2E eval
  fcwh-nokeys==off-lfm25 bit_identical=1 sidecar=unopened ok`, `E2E keys
  lfm25-p2x prompt=1024 chunks=512 moe_calls=… whole=… fc_calls=… fc_rows=…
  logits bit_identical=1 ok`; `CONV BLOCK CHUNKED BIT-IDENTICAL` and the
  `fc_layer_host_check` chunk cell in `ALL CHECKS PASS`; **and one Gemma
  line this plan adds**: the gemma64 fixture quantized with
  `--fc_wh_sidecar` byte-identical in its main `.bin` to the run without
  the flag, the sidecar's index naming 7 × layers FC images (`E2E quant
  fcwh-gemma64 main=same images=<n> ok`). Rung 2 (conv block + `conv_wLen`).
  **As built (branch `htp/234-p3-fcwh-prefill`):** only `db2c27a11`
  (`quantize_stream.cpp`, `htp_wh_layout.h`, where `whUnpack` was already
  in from #249) and `7b622617a` (`htp_e2e_test.cpp`, beside the Gemma model
  switch) conflicted; `75f6f007d`, `a3d966164` and `366c06c57` applied
  without one, and the `run_inproc_e2e.sh` hunks merged after the keys
  block by themselves. "7 × layers" is 7 × layers minus the
  `attention_k_eq_v` full-attention layers (no `_wv`): gemma64 has
  `images=20` (2 sliding × 7 + 1 full × 6), and the gate builds the list
  of expected names from `config.json`, so the 26B's own count comes from
  its own config. `*Lfm2Moe*` is now 7 / 7 and `unittest_causallm_models`
  115 / 115 (with `FcWhSidecarMatchesWhQuantize`).
* **P4. One-PD decode FC / DENSE_FFN on the sidecar (#225 PR 2 = #233)** —
  `b73e01b1a` (kernel + `fc_wh_det.h` + host check; `run_host_checks.sh`
  hunk re-placed after the 2-bit cells), `7efd22adf` rebased (`bindQ4m1`
  at `:1944`, `graph_host_check.c` after the Gemma mutants,
  `run_inproc_e2e.sh` after P3's lines). **Gemma addition**: in
  `Transformer::repack_weight`'s sidecar block, also hand the file to
  `get_htp_ops()` when `NNTR_HTP_E2E` is requested and the backend holds
  pending Q4_0 graph weights (`finish_decode_graph_q4_0`'s `q4_pending_`),
  so a model with no keyed FC (Gemma) opens it for the bind. Gate: rung 1 —
  §1 unchanged, the commit's lines `E2E fwd lfm25 fcwh kinds=all
  calls/token=1.00 q4m1_handles=1 wh_handles=28 e3==e1 bit_identical=1 pool
  C=2 bit_identical=1 misses=… ok`, `E2E tokens e3fcwh==off-lfm25 8/8`,
  `E2E ppl-decode e3fcwh-lfm25 … top1=7/7`, `GRAPH FC WH OK` + `GRAPH FC WH
  MUTANT CAUGHT`, `FC WH BIT-IDENTICAL … ` + `FC WH MUTANT CAUGHT`; **the
  Gemma lines this plan adds**: `E2E fwd gemma64 fcwh kinds=all
  calls/token=1.00 attn_caches=2 q4m1_handles=1 wh_handles=<n>`, `E2E
  tokens gemma64-fcwh==off 8/8`, `gemma64-fcwh … min_snr_db` ≥ 20 against
  the CPU run, `E2E e3 pool C=2 gemma64-fcwh == e3 bit_identical=1` (the
  MoE dumps are the sidecar-less run's: the experts did not move). Rung 2
  (`hexkl_graph.c`, `hexkl_mm_u8i4_moe.c`, `htp_graph_desc.h`). The PR body
  re-states the 26B arena sum (§3) and says the LM_HEAD stays Q4M1.
* **P5. The #4408 diagnostics** — `a7c527ac2` rebased on the pool's
  `ExpertFileDesc` (`compute_ops.h:471-484` [C]; the diff runs on the
  layer's CPU-side call, the `ponytail:` says it does not reach inside the
  one-PD token — the token path's instrument is the E2E `--dump` + `$EVAL`
  SNR); `prefill_timeline.py` ×3 squashed, optional. Gate: rung 1 — hd64
  Gemma fixture with `NNTR_MOE_DIFF=8`: SNR ≥ 30 dB on every layer;
  `NNTR_MOE_SHADOW=1` tokens == the CPU run 8 / 8; both unset → every line
  unchanged; `python3 -I tools/prefill_timeline.py` on a saved `--profile`
  log parses. Independent of P3 / P4; may run in parallel with them, lands
  after P4 if it touches `htp_compute_ops.cpp`.
* **P6. The ARM tier (#219), conditional — after plan 201 S5's first miss
  numbers, not before.** `32b1bc6de` re-expressed on the `w_bits` API
  (§3's table), `dbda0fc25` (clean), `cc9b20d0c` rebased; **not**
  `cdbf66f20` / `0235d9d79` as they are: on `htp_decode` the knob stays
  env-only (`NNTR_MOE_TIER` unset = 0 on every path) **and** the tier takes
  a byte cap (`NNTR_MOE_TIER_MIB`, default the LFM complement's ≈ 480) above
  which it logs `tier: skipped complement_mib=… cap=…` and holds nothing —
  the `posix_memalign` of a 10.8 GiB complement must not be reachable from
  a config. The LFM default of record (`=2` under E2E, user 2026-10-06)
  can then be re-stated per model in `nntr_config.json`, not in code. Gate:
  rung 1 — §1 unchanged, `E2E e3 pool tier=1 C=2 hd64 / … == tier=0
  bit_identical=1 … tier_reads=0`, and a Gemma line: the gemma64 E2E with
  `NNTR_MOE_TIER=2` and a cap below its complement prints the skip line and
  every Gemma line unchanged. Whether P6 happens at all is decided by S5's
  misses / token at the pool C the 26B allows (plan 201 §3.4, plan 229
  §7.3): the tier pays only where the miss cost is the largest term, which
  on LFM was G = 64 (rule 65).
* **Not in this issue:** plan 229 S2 (2-bit FCs on the P3 / P4 path: the
  `FCWH_FORMAT` bump, `registerFcWh` by `bits`, `fc_m1_run`'s u8i2 column
  branch) — lands after P4; LEDGER ㉞ (sliding cache sizing, hook-less CPU
  layers) is orthogonal; the single-base docs.

**Device measurement**: none in P3–P6. The first device run of everything
above is plan 201 S5 (the 26B files; the attached unit is the S25 Ultra
`R3CY205ZMND` since cycle 36, the S26 `R5KL20NFRCK` when re-attached), whose
handoff variants are plan 229 S4's (A / B2 / B2-C / B2-S4) once the ternary
files exist. An LFM bridge sitting (one PD, Q28: A vs the P3 + P4 set on
`htp_decode`) would only re-read what #225's sitting already read on
`htp_first_version` (`fc wh: … arena_kib=221184 heap_kib=0 requant=0`,
`calls/token=1.00`, LEDGER §2 #225 row) and is not filed.

## 5. Risks

* **The Gemma sidecar is untested on the real 26B until S5.** P3 proves
  the writer on the gemma64 fixture (image count, main `.bin` byte-identical)
  and P4 the bind and the texts; the 26B's `fcWhKey` collisions (8 KiB of
  each weight, `htp_wh_layout.h` `ponytail:`) across 30 × 7 FC weights are
  arithmetic until the file exists — P4's `set_fc_wh_file` refuses a
  duplicate key at load, so a collision is a load failure, not a wrong
  matmul.
* **The `w_bits` re-expression of the tier (P6)** is the one port that
  rewrites, not re-places; its host gate (`tier_reads=0`, bit-identity
  against `tier=0`) covers the 4-bit path; the 2-bit tier path (a 2-bit
  model with the tier on) needs its own E2E line in P6 or is declared
  unsupported there.
* **Ternary ≠ #4408's 4-bit**: unchanged from revision 1 — doc 55's
  accuracy numbers do not transfer; plan 229 S0 (needs-user) decides the
  scale scope that the sidecar's `w_scale` per column either matches or not.
* **Stale skel / stub** on P3 and P4 (DSP sources moved, IDL not): md5s
  on both ends, one tree per sitting (rule 3). A skel older than P3's
  `conv_wLen` 5 C refuses the chunked conv block with `AEE_EBADPARM`; a
  skel older than P4 refuses `HTP_GRAPH_FEED_WH` at `graph_init` — both
  are load-time failures, named in the PRs.
* **Address space**: P4 moves the FC set into its own arena chunk sized by
  `fcwhLeft`; on the 26B the sum (pool + FC set ≈ 790 + lm_head Q4M1 ≈ 415
  + heap ≈ 150 + KV) against 3 840 fixes the pool C S5 can ask for — the PR
  re-states it; the ceiling cell after every run.
* **`htp_first_version` is closed**, so the stale-source risk of revision
  1 is gone; the residual is the supervisor's mirror of its docs (§6)
  disagreeing with this tree's code state — each port PR's body names the
  sha it ported and the mirror cites that.
* **Host-vs-device gap**: these ports prove bytes (P3), the bind and the
  texts on fixtures (P4), SNR (P5) and the tier's mechanics (P6) on the
  host; DMA rate, DVFS, thermal drift and the page cache enter only at S5,
  where the handoff's A / B pairs inside one sitting make them visible
  (rules 13, 52, 61).

## 6. Docs to update

* **Mirroring decision (this plan's answer to the user's question):** the
  `htp_first_version` docs — LEDGER cycles 33–36 of that side, BENCHMARK's
  LFM2.5 close-out table and Artifacts rows, the guide refresh
  (`21266c9a5`), the measurement files under `docs/measurements/` (219 /
  222 / 225 / 236 and `config/*.json`), the contract §4.1 hunk
  (`d39c30f85`) — are **the supervisor's mirror, not the port PRs'**
  (`htp_decode`'s cycle 36 already states "LEDGER / BENCHMARK mirrored with
  `htp_first_version`", `9c92d0ac8`). The port PRs carry only the two plan
  files their code cites (`225-fc-qs4cx-wh-model-file.md`,
  `236-qkv-fc-prefill-chunks.md`, with P3) and this plan's "as built"
  notes. `06ed17b7b` is not carried.
* **`docs/htp_moe/LEDGER.md`** (supervisor): §Upstream — `htp_first_version`
  closed at `77a01902f`, the classification of §2.1 (15 code commits: 3
  already-in, 7 ported in P3 / P4, 3 + 2 conditional in P6), #4408 watch
  unchanged; §1 — the rule candidates of revision 1 §3 (marked
  `[M-4408, 4-bit, hybrid, S25]`), and one new candidate from this survey:
  **a load-time structure sized by "everything that did not fit" needs a
  cap before it meets a second model** (the tier on the 26B, §2.1); open
  items — ㉜ gains the P6 condition, ㊳ unchanged, a new item for "Gemma
  engine keys" (whether the hybrid's FC prefill ever moves to the HTP for
  Gemma — until then #236 is LFM-only).
* **`docs/htp_moe/BENCHMARK.md`** (supervisor): Artifacts — the gemma64
  fixture's sidecar md5 once P3 lands is **not** an artifact (fixture);
  the 26B sidecar's md5 joins the `nntr_config.json` row when the files
  exist; the Gemma block keeps the "#4408 hybrid S25" context rows of
  revision 1.
* **Plan 229**: S2's prerequisite is now "P4 merged" (not "P5"); S2's
  expected conflicts gain `registerFcWh` and `fc_m1_run`'s per-handle
  `bits`; §7.1's "FC set on WH (P5 sidecar)" reads P3 / P4.
* **Plan 201**: the S5 paragraph points at P3 / P4 as the host
  prerequisites for the FC kinds on WH, and at P6 as a lever gated on S5's
  miss numbers.
