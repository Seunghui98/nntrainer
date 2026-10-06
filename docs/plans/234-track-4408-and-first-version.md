# 234 — `htp_decode` stays the Gemma base: what to bring over from upstream #4408 and from `htp_first_version`, in what order, and what each port costs

Issue: dlwlzzero/nntrainer#234 (p1). Base `htp_decode` @ `a19d4dd4e`.
Sources: upstream nntrainer/nntrainer#4408 (`refs/pr/4408`, head
`28771a928`, 47 own commits over merge-base `7a8e2609e`; its docs 55 / 56 /
57 read with `git show refs/pr/4408:docs/htp_attention/…`) and
`origin/htp_first_version` (13 commits over `htp_decode`: #222 / #225 /
#219, PR #233 open there). Plan 229 (`229-ternary-lut-decode.md`) is the
format plan this one feeds; plan 201 the structure.

**User, 2026-10-06:** the goal is the ternary 26B-A4B decoding end to end
on the one-PD path; no tok/s target; the 2 GB peak-memory constraint is
considered **last**, after decode works on the device (plan 229 S7). Nothing
in this plan's order depends on the memory limit.

Evidence tags: **[C]** read on `htp_decode` at `path:line`; **[A]** the
commit's `git apply --check` against this tree, run in this session
(dry run only — no port was built or tested here); **[M-4408]** measured by
#4408's author on the S25 `R3CY10WM83Y` (doc 55 §10.x), 4-bit `QS4CX_WH`
from a bf16 checkpoint, hybrid path (CPU attention); **[M-LFM]** this
project's LFM2.5 silicon record (LEDGER rule cited); **[G]** arithmetic.

## 1. Goal and gate

Acceptance (issue): "which commits from each source come to `htp_decode`,
in what order, expected conflicts with the S4 code, and the host gate (LFM
and Gemma E2E lines unchanged, `INPROC E2E PASS`)". Measurable form, per
port PR:

| gate | where | pass |
|---|---|---|
| LFM lines unchanged | `test/htp/host/run_inproc_e2e.sh` | `INPROC E2E PASS`; every `bit_identical=1` line of the header block (`run_inproc_e2e.sh:19-166` [C]) still prints; `E2E e3 pool C=1 lfm25 / C=2 … == e3 bit_identical=1` with the same `misses=` |
| Gemma lines unchanged | same script, the block at `:367-388` / `:491-520` [C] | `E2E gemma64 tokens off==cpu 8/8`, `E2E fwd gemma64 e3 calls/token=1.00 attn_caches=2 timeouts=0 ok`, `gemma64-e3 … min_snr_db` ≥ 20 (27.63 at cycle 34), `E2E tokens gemma64 e3==off 8/8`, `E2E e3 pool C=2 gemma64 == e3 bit_identical=1` |
| HF differential | `test/unittest/models/unittest_causallm_gemma4_reference.cpp:82-111` [C] — the base already holds an HF-reference test on `gemma4_moe_tiny` / `_hd64` (PR #232); #4408's `06862c37f` is **not** ported | `unittest_causallm_models` all pass, the Gemma cases included |
| host checks | `test/htp/host/run_host_checks.sh`, `tools/htp_syntax_check.sh`, `*Lfm2Moe*` | `ALL CHECKS PASS`, exit 0, 6 / 6 |
| the quantizer guard (port 2) | a new check in the quantizer's unit test | a `.safetensors` whose header permutes two same-sized tensors is **refused at the dry run** with both names and sizes (doc 55 §10.5's failure, reproduced on the tiny fixture); the unpermuted file quantizes byte-identical to the `.bin` path |
| standing | prefill ≥ −5 % of the same sitting's A; text identical to the CPU run; `QS4CX_WH` has no CPU fallback | no device cell in this issue — the ports are host-gated; the first device reading of all of them is plan 201 S5 / plan 229 S4 |

Host state at the start of this plan: `INPROC E2E PASS` on `htp_decode`
per LEDGER cycle 34 (`874938550` / `b6d2f3a0e`). **Not re-run in this
session**; the first port PR re-establishes it on its base commit before
changing anything.

## 2. Where it lives

The two lineages are not the same Gemma. `htp_decode`'s MoE Gemma is
upstream #4296's (`Applications/CausalLM/models/gemma4/gemma4_moe_causallm.cpp`,
`gemma4_moe_layer.cpp`, `res/gemma4/gemma4_moe_weight_converter.py`) with
plan 201 S4's hand-over to `lfm2_moe` when `moe_engine=htp`
(`gemma4_moe_causallm.cpp:58-69, :303-327` [C]) and the `QS4CX_WH` gate | up
writer in `quantize_stream.cpp:1041-1120` [C]. #4408's is Seunghui98's own:
`gemma4_causallm.cpp` grown an `enable_moe_block` path (`1cffe5e25`), the
`lfm2_moe` layer given a `router_type` (`11fefbb76`), its own converter
(`228a02641`, fused gate_up, router gamma folded) and its own fixture
(`06862c37f`). Every model-side commit of #4408 therefore collides with S4
by construction, not by line drift; the portable parts are the ones that
sit under the model: the quantizer's input path, the diagnostics, the tool.

| candidate | verdict | why | `apply --check` [A] | conflicts with S4 code |
|---|---|---|---|---|
| **#4408** `9dd2c76de` quantizer reads a converter-written `.safetensors` | **port** (P2) | `htp_decode`'s quantizer refuses anything but `.bin` (`quantize_stream.cpp:1498-1500` [C]); the base converter already writes safetensors (`weight_converter.py:81-89`, `gemma4_moe_weight_converter.py:104-107` [C]); the 100 GB FP32 intermediate is what the user's PC produces | clean | none |
| **#4408** `f9c09eaa2` "one tensor read is one tensor the converter wrote" | **port** (P2) | the §10.5 guard: byte totals per layer were equal while the order was wrong; the guard costs one comparison per tensor and names the first disagreement at the dry run. On `htp_decode` the converter is #4296's and the writer #231's — a different pair, same hazard | fails at `:1486` (the `is_gemma4_moe` dtype check moved) | `quantize_stream.cpp` only: `TensorWriter` ctor and `expectSourceTensor` around `:489`, `:743`, the Gemma dtype check `:1491` |
| **#4408** `7dbd876ed` `whUnpack` | **port** (P3) | ten lines, the inverse of `whPack` (`htp_wh_layout.h:60-99` [C]); the host reference of P3 needs it; also simplifies `wh_pack_unpacks_like_the_load_path` | clean | none |
| **#4408** `a7c527ac2` `NNTR_MOE_DIFF` / `NNTR_MOE_SHADOW` | **port, rebased** (P3) | the one instrument that splits a wrong device text between the MoE call and what feeds it, in one run (input stats + SNR per layer); the first 26B run on this tree will want it (#4408's first run was noise, §10.4) | fails at `lfm2_moe_layer.cpp:20` | `lfm2_moe_layer.cpp`: the hook that returns before the LRU on the token path (`:1199-1206`, plan 201 §2.3) means the diff runs only on the hybrid / `NNTR_HTP_FORWARD` call, not inside the one-PD token — state that in the ported comment; the pool's `ExpertFileDesc` (`compute_ops.h:435-441`) replaces its `getFd()` read |
| **#4408** `93c2a6ff6` `f33ed210a` `e8d2604ec` `tools/prefill_timeline.py` | **port** (P4, optional) | a profile-build fold per node / per layer / CPU-vs-NPU; nothing on this tree folds a `--profile` build that way (`tools/htp_fc_report.py` reads the HTP stage lines only) | first clean, next two need the file | none (new file) |
| **#4408** `1cffe5e25` `res/gemma4/gemma4-26b-a4b/nntr_config.json` | **port the file only**, rewritten (P1) | the base has no 26B config; #4408's is FP32 / `moe_engine: cpu` / `moe_cache_experts 16` / `skip_prefill: true` (the PPL trap of §10.9). Ours: `moe_engine: htp`, `moe_layer_dtype: QS4CX_WH` (the base's own guard `gemma4_moe_causallm.cpp:67` [C]), the #222 engine keys, `skip_prefill: false`, `init_seq_len 1024` | code part fails (other `gemma4_causallm.cpp`) | none for the file |
| **#4408** `228a02641` converter MoE tensors | **skip** | a second converter for a second model class; `gemma4_moe_weight_converter.py` is the base's and matches the #231 writer. One idea is kept as a P2 check, not code: its "lazy slice" exists here as `TensorSlice` (`:22-39` [C]) | fails | whole file |
| **#4408** `11fefbb76` `1cffe5e25` (code) `42d1ac61f` `3fde10415` router type / MoE block on `gemma4_causallm` / projections on engine keys / string `use_bidirectional_attention` | **skip** | the base's `Gemma4MoECausalLM` does each already (S4; the issue lists them as overlap). `42d1ac61f`'s engine keys are the #222 keys on this tree; §10.12's finding about them is a LEDGER rule (§3) | fail | — |
| **#4408** `7becfdc97` `69db273ae` `66c41a5ab` QS4CX FCs of an MoE model, registered at load, sliced like Q4_0 | **skip, superseded** | §10.15 measured it: prefill nll 4.510 (better) but decode text loops and peak RSS +0.8 GB (the ARM holds both the original and the KleidiAI pack). `htp_first_version`'s FC WH sidecar (`db2c27a11` + `7b622617a`, below) stores the FC once, in the DSP's format, with no load-time copy — rule 63's consequence, and the path plan 229 needs for 2-bit FCs | two clean, one fails | — |
| **#4408** `89b426f4c` QS4CX pack only where read | **skip** | the base's E2E already quantizes and runs a QS4CX CPU Gemma on x86 (`run_inproc_e2e.sh:370-385` [C]) | fails | — |
| **#4408** `06862c37f` `d02cd45d6` `0dcb2789f` `3bc74999e` `757e81902` fixture / HF test / device gtests / GeGLU | **skip** | S4 has each (`gemma4_moe_tiny{,_hd64}`, `unittest_causallm_gemma4_reference.cpp`, `HEXKL_MOE_FLAG_GEGLU`) | fail / "already exists" | — |
| **#4408** docs 55 / 56 / 57 | **not carried as files**; §10.5–10.15 become LEDGER rules (§3) | — | — | — |
| **hfv** `c0fa4e419` convert the host ISA's Q4_0 repack before the qs4cx re-quantization | **port** (P1) | the in-process E2E with the engine keys on x86 needs it (`f92ec3596` depends on it) | clean | none |
| **hfv** `d541bda3a` skip the dense_ffn CPU FFN on a row the HTP runs whole | **port** (P1) | Gemma has a dense FFN in every layer; the CPU must not redo the row | clean | none |
| **hfv** `f92ec3596` the config of record's engine keys in the in-process E2E | **port, rebased** (P1) | the E2E line that proves all kinds resident on the host | fails at `htp_e2e_test.cpp:119`, `run_inproc_e2e.sh:122` | `run_inproc_e2e.sh`: the header block and the lfm25 section moved for the Gemma block (`:367-388`); `htp_e2e_test.cpp` flag parsing |
| **hfv** `db2c27a11` `--fc_wh_sidecar` writes the FC weights as `QS4CX_WH` images | **port** (P5) | the FC-on-WH path plan 229 §3.2 needs (floor 45 → 61 only with 2-bit FCs on WH); LFM-only today — the Gemma writer gains it in plan 229 S2 | clean | none; `quantize_stream.cpp` additive |
| **hfv** `7b622617a` read the FC WH sidecar into the arena at load | **port, rebased** (P5) | the loader half of the same path; removes the load-time re-quantization rule 63 names | fails at `htp_compute_ops.cpp:97`, `htp_wh_layout.h:99` | `htp_compute_ops.cpp`: the FC placement `e2ePlaceFc` (plan 201 §2.2 `:4085`) and the arena chunk list moved with #201 S1 / #211; `htp_wh_layout.h`: `whUnpack` (P3) lands in the same spot — order P3 before P5 |
| **hfv** `75f6f007d` prefill in 512-row chunks; the conv block reads its history | **port, rebased** (P5) | P1024 on the HTP FCs fails `AEE_ENOMEMORY` without it (rule 63 b); Gemma prompts 512 / 1024 are the sitting lengths. The conv-block half is LFM-only and harmless | fails at `run_inproc_e2e.sh:131` | `run_inproc_e2e.sh` only |
| **hfv** `32b1bc6de` `dbda0fc25` `cc9b20d0c` `69c05f517` the ARM tier (#219, PR #224) | **later** | measured on nothing yet (#219 `state:needs-measurement`); its tier size is `n_layers × E − capacity` (plan 219 §3), which on Gemma is ≈ 3 300 experts — needs a cap before it can even start; and plan 229 S7 says it cannot fit under 2 GB. Re-read after S5's first miss numbers | first clean, third fails | `run_inproc_e2e.sh` |
| **hfv** `2c60e5215` `69928f26b` `4617545f5` plan 225 + its guide / Artifacts rows | **carry with P5** (docs only) | the plan the sidecar code cites; the Artifacts row is already mirrored (cycle 34) | — | — |
| **hfv** `06ed17b7b` single-base docs rewrite | **not carried** (issue) | — | — | — |
| **hfv** PR #233 (#225 PR 2: decode FC / DENSE_FFN on the FC WH set, `b73e01b1a` + `7efd22adf` on `htp/225-fcwh-e2e`) | **later, as a unit after it merges there** | it is the decode half of P5 and the file plan 229 S2's 2-bit FC GEMV sits on; porting an open PR twice is the stale-stub trap in another form | — | `htp_compute_ops.cpp` FC ops, `hexkl_mm_u8i4_fc_m1_run` |

Consumers that move with a changed contract: **none of the ports changes
the IDL** (`test/htp/nntr_hvx.idl`), `generate_stub.sh` or the skel — P5's
loader reads a file into the arena the DSP already maps. The quantizer's
format tag moves twice: P2 (input `.safetensors`, no output change) and P5
(`--fc_wh_sidecar`, a second output file; the loader check is the sidecar
header `7b622617a` adds to `htp_wh_layout.h`). `NNTR_HTP_PROFILE` stage
tables and `tools/htp_fc_report.py` are untouched; P4 adds a tool beside
them. `HtpComputeOps` moves only in P5.

## 3. Design

**Chosen: port by layer, lowest first — tooling and quantizer input (P1,
P2), then diagnostics (P3, P4), then the FC WH path (P5) — each a PR into
`htp_decode` gated by the unchanged E2E lines; nothing of #4408's model
code.** The reason is the lineage split above: #4408's model-side commits
are a parallel implementation of what S4 built on #4296, and the base's
tests (HF reference, E2E Gemma lines, pool bit-identity) already cover the
same ground. What #4408 has that the base lacks is below the model —
reading the PC's `.safetensors`, refusing a wrong tensor order before a byte
is written, and seeing inside a wrong device text — plus a record of
device findings. Those are cheap and conflict-free, so they go first and
are on the tree before the 26B files arrive. The FC WH path is larger and
touches `htp_compute_ops.cpp`, so it goes last among the ports, but still
before plan 229 S2 (whose 2-bit FC writer extends it).

**Rejected: merge `htp_first_version` into `htp_decode` wholesale** (13
commits, one merge commit). It would carry the tier (#219, unmeasured and
Gemma-unsized), the single-base docs rewrite the issue excludes, and the
#222 closing-sitting docs, and it would make the three real conflicts
(`run_inproc_e2e.sh`, `htp_compute_ops.cpp`, `htp_wh_layout.h`) one
unreviewable hunk. Cherry-picks keep each port's gate its own.

**What #4408's device findings become (LEDGER rules; worded for Gemma on
the one-PD path, and marked as 4-bit-from-bf16 findings).** #4408 ran
4-bit `QS4CX` from a bf16 checkpoint; the user's 26B is ternary, whose three
levels fit int4 (and the 2-bit palette) exactly, so the *accuracy* findings
about the weight format do not transfer as numbers — only as the mechanism
and the measurement method.

| finding [M-4408] | rule candidate |
|---|---|
| §10.5–10.6: a 100 GB FP32 file whose MoE tensor order differed from the quantizer's passed the byte-total dry run and gave noise on the device; fixed by the per-tensor guard | **A byte-total check cannot see tensor order; every quantizer read must map to exactly one source tensor, refused at the dry run** (P2's guard is the rule's code) |
| §10.7 / §10.10: the HTP kernel equals the CPU `QS4CX` kernel within 0.007 nat; FP16 attention 0.01; the per-column scale over K = 2816 costs 0.098–0.116 nat against Q4_0's block-32 scale (weight SNR 17.2 vs 21.2 dB) | **On a 4-bit-from-bf16 checkpoint the per-column `QS4CX` scale is the accuracy gap, not the kernel** (0.1 nat on the 26B); the decomposition method — x86 QS4CX vs device HTP vs device Q4_0 on one prompt with `NNTR_PPL=1` — is the way to read a Gemma gap. For ternary weights the scale question is plan 229 S0's (per column / per group), and this rule's number is not expected to apply |
| §10.9: a least-squares per-channel scale raised weight SNR 17.2 → 18.9 dB and **worsened** nll by 0.032; reverted (`221212159`) | **Weight MSE is not a proxy for nll on this model; a quantizer change is read on nll, never on SNR alone** |
| §10.12: the attention projections on the HTP via load-time Q4_0 → QS4CX re-quantization cost +0.46 nat; the dense MLP +0.13 | **the same mechanism as LFM's rule 63** (two stacked quantizations); confirms the rule 63 consequence (store the FC once, in the DSP's format: P5) on the 26B |
| §10.14: with every FC on the HTP, C = 32 fails `nntr_hvx_weight_register_u8i4` and C = 40 `swap_u8i4_arena` with `AEE_ENOMEMORY`; the C ceiling is 24–32 on the hybrid; decode fastest at C = 8 because ms/miss rises with C (0.76 / 1.58 / 2.27 at C = 8 / 16 / 24, 2.87 MiB experts) | **The pool's C ceiling falls as more kinds move to the HTP (one 4 GiB address space, rule 8); on the 26B the arena + DSP heap must be summed against 3840 per configuration before a sitting** (rule 63 d restated for Gemma). The ms/miss-vs-C reading is LFM's ㉜ again (page cache squeezed by the arena) and is read on `pgpgin` / refaults, not assumed |
| §10.14's miss counts: 81 348 / 62 198 / 41 955 misses over 512 generated tokens at C = 8 / 16 / 24 (hit 34 / 49 / 66 % of 240 routed uses a token) | **the first Gemma hit-rate points** — plan 229 S7 and plan 201 §3.4's "unknown for Gemma" use them until S5's own trace; they are hybrid-path LRU numbers on one prompt |
| §10.11: prefill 447 tokens at C = 5 vs 16: 40.9 vs 89.3 TPS (synchronous expert reads 2 893 vs 0); RoPE table built for 262 144 positions (≈ 800 ms, 1.5 GiB) | **C ≥ 5 keeps the prefill read-ahead alive (30 C ≥ 128 + slack); the RoPE table must be sized by `max_seq_len`** — check `MHACoreLayer::precompute_freqs` on this tree before S5 (not verified here) |
| §10.13 / §10.15: prefill 512 / 1024 on the hybrid 110.7 / 116.0 TPS; decode 2.8–4.5 TPS on the hybrid | not rules: hybrid numbers; the one-PD path is the structure (plan 201); they go to BENCHMARK's Gemma block as "upstream, hybrid, S25" context rows |

Contract §2 and doc 45 §3: no wall moves; the arena budget moves only
with P5 (the FC set as WH images, same bytes as the Q4M1 set it replaces,
the DSP heap loses the rule 63 copies); `QS4CX_WH` keeps no CPU fallback
(P5's sidecar is read by the HTP only, the CPU keeps its own Q4_0 until
PR #233 decides decode); `_det` and bit-identity: the ports change no
kernel; the Gemma E2E lines are the proof.

## 4. Steps

Each step is one PR into `htp_decode`, ending in rung 1 of
`.claude/skills/hexagon-gates` (host checks); rung 2 (skel) only where
noted; no step needs a device. Order is by dependency, not by size.

* **P1. Tooling for the 26B config and the engine-key E2E** — cherry-pick
  `c0fa4e419`, `d541bda3a`, then `f92ec3596` rebased onto the current
  `run_inproc_e2e.sh` (the lfm25 section after the Gemma block); add
  `res/gemma4/gemma4-26b-a4b/nntr_config.json` written for this tree
  (`moe_engine: htp`, `moe_layer_dtype: QS4CX_WH`, the #222 keys all `htp`,
  `init_seq_len 1024`, `max_seq_len 4096`, `skip_prefill: false`,
  `moe_cache_size` = the pool C the sitting sets, `sample_input` with the
  Gemma chat template). Gate: rung 1 — `INPROC E2E PASS` with the new
  engine-key line and every existing line unchanged; `*Lfm2Moe*` 6 / 6.
  *As built (PR for P1):* the 26B file leaves out the #222 keys and
  `moe_cache_size`. The keys are read only by `lfm2_causallm.cpp:363-377`,
  `moe_cache_size` only by the CPU `gemma4_moe` layer, and the HTP pool is
  `NNTR_MOE_CACHE_EXPERTS`. `d541bda3a`'s skip is LFM-only: Gemma builds no
  `DenseFfnLayer`, so whether the Gemma one-PD decode still runs its dense
  MLP on the CPU is open. The keys line exposed a hang from `ed2660ddc`
  (`exp_live` read out of bounds on 4-bit M>1 calls); P1 carries the fix.
* **P2. The quantizer reads the PC's file and refuses a wrong order** —
  `9dd2c76de` (clean), then `f9c09eaa2` rebased (the ctor / dtype-check
  hunks around `:489` / `:1491`); drop the "Gemma4 MoE FC/expert dtype must
  be FP32 or Q4_0" check only if the writer path proves it dead (it guards
  `writeFc` dtypes the Gemma writer passes through — keep it otherwise).
  Add the permuted-header check of §1 to the quantizer's unit test on the
  tiny fixture, and a `.bin` vs `.safetensors` byte-identity check on the
  same fixture (the commit's own check, re-done on #4296's converter).
  Gate: rung 1 — `unittest_causallm_models` incl. the two new cases; E2E
  lines unchanged (the E2E quantizes `.bin` fixtures, so this is a no-op
  there by construction — say so in the PR).
  *As built (PR for P2, stacked on P1):* both commits ported, authors
  kept. Conflict only in `quantize_stream.cpp`: `PaletteOptions` (#238)
  and `writer.setPalette` kept beside `SourceTensor` and the new ctor
  argument; `writeFcConcat` (#231, the Gemma `QS4CX_WH` / `QS2CX_WH`
  gate | up writer) checks its two source tensors one by one. The "FP32
  or Q4_0" Gemma dtype check (now `:1737`) is untouched and kept: it
  reads `--fc_dtype` only, the experts' `QS4CX_WH` / `QS2CX_WH` come in
  through `--moe_dtype`. The §1 gate as built differs in
  one point: the guard compares **sizes**, so a swap of two same-sized
  tensors is not detectable by it (the converter's keys and the
  quantizer's names are different spellings; upstream chose not to map
  them). The test therefore reproduces §10.5 as it happened — router
  scale before the router matrix, offsets consistent, byte total equal —
  which is refused at the dry run with both names and sizes and an empty
  output; the same bytes as a `.bin` quantize (control). A header whose
  keys are in another order is **reordered** by `data_offsets`, not
  refused, and quantizes byte-identical to the `.bin` (Q4_0 and
  `QS4CX_WH`). The fixture's `.safetensors` is the converter's output,
  written by `generate_gemma4_moe_reference.py` beside the `.bin`. One
  addition beyond #4408, from the port's review: the reader refuses a
  header it cannot follow positionally (a non-F32 tensor, offsets that do
  not tile the payload, no tensors, a wrapping header length).
* **P3. The diagnostic pair** — `7dbd876ed` (clean), then `a7c527ac2`
  rebased: the diff runs on the layer's CPU-side call (the hybrid /
  `NNTR_HTP_FORWARD` path), reads expert bytes through the pool's
  `ExpertFileDesc`, and prints the input stats + SNR per layer; the
  `ponytail:` says it does not reach inside the one-PD token (the router
  hook returns first) — the token path's instrument is the E2E `--dump` +
  `$EVAL` SNR (`run_inproc_e2e.sh:491-510`). Gate: rung 1 — on the hd64
  Gemma fixture with `NNTR_MOE_DIFF=8`, SNR ≥ 30 dB on every layer (the
  commit's own expectation 30–45); `NNTR_MOE_SHADOW=1` tokens == the CPU
  run 8 / 8; both unset → every E2E line unchanged.
* **P4. `tools/prefill_timeline.py`** — the three commits squashed to one
  (new file). Gate: `python3 -I tools/prefill_timeline.py` on a saved
  `--profile` log from `docs/measurements/` parses and prints; no other
  check (a tool). Optional; can ride with P3.
* **P5. The FC WH sidecar (prefill half)** — `db2c27a11` (clean),
  `7b622617a` rebased onto the post-#211 `htp_compute_ops.cpp` (the one-PD
  FC placement; `whUnpack` from P3 already in `htp_wh_layout.h`),
  `75f6f007d` rebased (its `run_inproc_e2e.sh` hunk), plus the plan 225
  docs. The Gemma writer does **not** gain `--fc_wh_sidecar` here — that
  is plan 229 S2's `bits = 2` writer, which extends this path once. Gate:
  rung 1 — the sidecar E2E lines `7b622617a` adds (lfm25 loads the sidecar,
  `bit_identical=1` against the re-quantized run is **not** expected —
  the commit's own line states the sidecar is the f32-quantized set; read
  its line as it is written there), every other line unchanged; rung 2 —
  the skel is rebuilt once for the md5 because DSP sources moved since the
  last staged set, `UNDEFINED SYMBOLS OK` (no IDL change expected; if
  `75f6f007d`'s `nntr_hvx_mm_u8i4.c` hunk changes a signature, the stub is
  regenerated and the PR says so).
* **Later, not in this issue:** PR #233's decode half after it merges on
  `htp_first_version`; the tier (#219) after S5's miss numbers and with a
  Gemma-sized cap; plan 229 S1 (the #4410 2-bit stack) is independent of
  P1–P4 and may run in parallel, but lands after P5 if it touches the FC
  path.

**Device measurement**: none in this plan. The first device run of
everything above is plan 201 S5 (Gemma on the attached S26,
`R5KL20NFRCK`), whose handoff variants are plan 229 S4's (A / B2 / B2-C /
B2-S4) once the ternary files exist; until then an LFM bridge sitting on
the farm S25 (one PD, Q28: A vs the P5 sidecar set) would read the sidecar
loader's effect on RSS, DSP heap and the P1024 `AEE_ENOMEMORY` — filed on
#225 / #233's side, not here.

## 5. Risks

* **The converter / writer pair on this tree is untested on the real
  checkpoint** (the base's HF test writes the fixture directly, as #4408's
  did — doc 55 §10.5's blind spot). P2's guard is the mitigation; the
  residual is a wrong-but-consistent pair (gate / up swapped on both
  sides), which only a device text or P3's SNR shows. The handoff table
  carries P3's per-layer SNR line next to the text.
* **Ternary ≠ #4408's 4-bit**: every accuracy number of doc 55 is from a
  bf16 → int4 per-column recipe; a ternary checkpoint with a per-group
  scale (plan 229 S0, needs-user) re-opens the format question with no
  number from #4408 to lean on.
* **Stale skel / stub** on P5 (DSP sources moved): md5s on both ends, one
  tree per sitting (rule 3). No IDL change expected; the PR checks.
* **Address space**: P5 moves the FC set from the Q4M1 arena to WH images
  of the same size and removes the rule 63 heap copies — the budget sum
  (pool + FC set + heap + scratch ≤ 3840) is re-stated in the PR for the
  Gemma shape; the ceiling cell after every run.
* **`htp_first_version` keeps moving** (PR #233, the tier's sitting): a
  port is pinned to a sha in its PR body; a later divergence is a new
  issue, not a silent re-port.
* **Host-vs-device gap**: these ports prove bytes (P2), SNR on a fixture
  (P3) and the loader's mechanics (P5) on the host; DMA rate, DVFS, thermal
  drift and the page cache enter only at S5, where the handoff's A / B
  pairs inside one sitting make them visible (rules 13, 52, 61).

## 6. Docs to update

* **`docs/htp_moe/LEDGER.md`**: §Upstream gains PR #4408 (`28771a928`,
  open, what is taken: P2 / P3 / P4; what is not: the model side, the
  QS4CX FC route) and `htp_first_version`'s state (13 commits, what is
  ported, #233 pending); §1 the rule candidates of §3 (as rules only
  where measured by #4408 — marked `[M-4408, 4-bit, hybrid, S25]`); open
  item: the tier's Gemma sizing; ㉜ gains the §10.14 ms/miss-vs-C
  reading.
* **`docs/htp_moe/BENCHMARK.md`**: the Gemma block gains an "upstream
  #4408, hybrid, S25" context row set (prefill 110.7 / 116.0 at 512 /
  1024, decode 2.8–4.5, nll 4.556 / 4.510, C sweep) marked not of record;
  Artifacts: the 26B `nntr_config.json` md5 once it exists.
* **Plan 229**: S1's "expected conflicts" gains P5 as a prerequisite for
  the FC path; S7 (the 2 GB budget) appended by this issue.
* **Plan 201**: the S5 paragraph points at P1–P5 as the host prerequisites
  and at #233 for decode FCs.
