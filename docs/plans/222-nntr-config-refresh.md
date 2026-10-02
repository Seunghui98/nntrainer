# Plan 222: refresh `nntr_config.json` of the NPU model (`q40-qs4cx-wh`)

Issue #222 (hexagon, p2). Tree `htp_decode` @ `31a2128b1`. Contract
`docs/plans/0001-htp-moe-decode-agent-system.md`; history docs 50 and 51
(`docs/htp_attention/`), both the PR author's own measurements.

## 1. Goal and gate

The issue says the config "looks outdated" and pastes a replacement. That is
not one change: of the eight keys that differ, three are harmless, one breaks
every sitting script, and four change what the model computes. The goal is
therefore: **a config of record for `q40-qs4cx-wh` that (a) runs unchanged on
the workstation, the S25 units and the farm S26, (b) is recorded with an md5
in BENCHMARK.md, and (c) adopts a behaviour change only after a same-sitting
A/B against the current config.**

Gates, measurable:

* Pure fixes: a run with the fixed config on the same binary prints the same
  `prefill:` / `generation:` lines within drift and the same text as the old
  config (they do not touch arithmetic); JSON parses; `tokenizer_file`
  resolves relative to the model dir (`Applications/CausalLM/main.cpp:85–95`).
* Behaviour changes: BENCHMARK `prefill tok/s, NPU, prompt 512` cell of the
  new config vs the old one in one sitting; standing gates: prefill ≥ −5 % of
  A, decode within A's drift; accuracy by plan 201 D2 (decode PPL forced on
  A's continuation pooled ≤ 1.02 × A over the 8 prompts, no new loop by
  `tools/htp/loop_check.py`, text approved by the user) **plus** prefill
  PPL (`NNTR_PPL=1`, `causal_lm.cpp:626–632`) on the same 8 prompts, since
  the engine keys change prefill arithmetic only. Byte-identical text cannot
  be the gate for an engine key (the prefill re-quantization moves the KV
  cache, LEDGER rule 39's reasoning); it is the gate for `init_seq_len`
  modulo §3.2's first-token artefact.
* Re-anchoring: the row of record (53.97 / 52.16 / 51.41, record sitting
  2026-09-30, `init_seq_len 512`, no engine key) moves to the new config
  only through a sitting that runs old-config A and new-config A side by
  side (BENCHMARK Method, cycle-23 paragraph: a mirrored cool sitting taken
  for that purpose).

## 2. Where it lives

The config is not in git. Copies:

| copy | path | who touches it |
|---|---|---|
| workstation (source of truth) | `/local/mnt/workspace/models/lfm2.5-8b-a1b/q40-qs4cx-wh/nntr_config.json` (674 B, 2026-09-21) | staging scripts copy it to the device |
| S25 units | `/data/local/tmp/nntrainer/causallm/models/q40-qs4cx-wh/nntr_config.json` | every runner `sed`s `num_to_generate` into it (`docs/measurements/201-s3-run.sh:54`, `204-s26-run.sh:66`) and `bad_word_ids` / `moe_engine` (`204-s26-run.sh:112–115`) |
| farm S26 (#208) | same relative layout (`204-s26-run.sh:31`) | user |
| the user's phone, older dir | `models/lfm2.5-8b-a1b-q40-qs4cx-wh/` with the PR author's config (`init_seq_len 1024`, FC groups on `htp`) — left untouched since #117 (BENCHMARK.md:738) | nobody; this is where the proposal comes from (its `sample_input` is the PR's 444-token prompt, docs 49–51) |
| checked in | none; contract `0001`:355 describes it, doc 51:240 and doc 52:143 print the PR author's | — |

Consumers of the keys that differ (all verified in this tree):

| key | read at | effect |
|---|---|---|
| `tokenizer_file` | `main.cpp:379` → `resolveNntrConfigPath` (`:85–95`: relative → model dir); `transformer.cpp:119–124` | pure fix: the current absolute `/data/local/tmp/...` path only works on a phone |
| `model_file_name` | `main.cpp:382,395` | **the proposed `nntr_lfm2.5_8b_a1b_q40_arm.bin` does not exist**: workstation has `nntr_lfm2_8b_a1b_q40_arm.bin` (md5 `7b7867fa…`, BENCHMARK.md:800), every runner hard-codes it (`201-s3-run.sh:25`, `204-s26-run.sh:32`), the device dir symlinks that name (BENCHMARK.md:738) |
| `use_embedding` | `lfm2_causallm.cpp:353–355`, default `false` | supported here (not only upstream); `false` is a no-op |
| `sample_input` | `main.cpp:436–445`, only when no prompt argument | no cell uses it (prompt file on the CLI); 444 tokens, fits `init_seq_len 512` |
| `conv_block_engine` | `lfm2_causallm.cpp:374–376` → `:154–176` (`conv_block` one-layer form) → `conv_block_layer.cpp:189–201` (`use_htp = rows > 1`) → `htp_compute_ops.cpp:5036–5075` (`get_or_register_conv` = two `get_or_register_fc`, `:4819–4860`, re-quantizing Q4_0 → qs4cx at `:4838`); load-time registration + one M=512 warm-up `transformer.cpp:640–660` | prefill of 18 conv blocks on the HTP; decode row stays the CPU fold "byte for byte" |
| `attn_proj_engine` | `lfm2_causallm.cpp:363–365` → `:91` (`qkv_layer` + o_proj engine) → `float_tensor.cpp:1037–1038` (`M > 1 \|\| accelerates_q4_0_at_m1()`, the latter `false` at `htp_compute_ops.cpp:1185`) → `gemm_q4_0_accel_fp32` `:1201` | prefill q/k/v/o on the HTP; decode CPU |
| `dense_ffn_engine` | `lfm2_causallm.cpp:377–378` → `transformer.cpp:834–847` (`dense_ffn` layer) → `get_or_register_dense` `:4934–4956` (three `htp_qs4cx_from_q4_0x4`) | prefill of layers 0–1's dense FFN on the HTP; decode CPU |
| `init_seq_len` | `transformer.cpp:137`; graph input `transformer.cpp:248`, `lfm2_causallm.cpp:404,413`; `causal_lm.cpp:715` | activation shapes; **first-token registration** (§3.2) |
| `moe_engine`, `moe_htp_layers`, `moe_layer_dtype`, `max_seq_len`, dtypes, `fsu*`, `num_to_generate` | unchanged | — |

Nothing in the IDL, stub, `HtpComputeOps`, quantizer tag, loader, profile
tables or `tools/htp_fc_report.py` changes: this plan edits a JSON file and
documents; no code.

## 3. Design

### 3.1 Classification of the proposal

| key | class | verdict |
|---|---|---|
| `tokenizer_file: "tokenizer.json"` | pure fix | adopt now |
| `use_embedding: false` | pure fix (explicit default) | adopt now |
| `sample_input` (PR's 444-token summarization prompt) | pure fix (unused by sittings) | adopt now; it documents the doc 49–51 prompt |
| `model_file_name: nntr_lfm2.5_…` | **break** | do not adopt; keep `nntr_lfm2_8b_a1b_q40_arm.bin`. Renaming the file instead is a user call: it touches BENCHMARK.md:800's row, two live runners, the device symlink and the farm copy for no measured gain |
| `conv_block_engine: htp` | behaviour | candidate; **needs the A/B sitting** (§3.3) |
| `attn_proj_engine: htp`, `dense_ffn_engine: htp` | behaviour | **not adopted**: fail D2 on the measured numbers (§3.3) |
| `init_seq_len: 1024` | behaviour | **needs the A/B sitting**, isolated from the engine key (§3.2) |

### 3.2 `init_seq_len` 512 → 1024

What it does on the `CausalLM::run` path the sittings use
(`lfm2_causallm.cpp:497–501` → `causal_lm.cpp:397`): prompt truncation is by
`max_seq_len − num_to_generate` (`causal_lm.cpp:545–560`), not by
`init_seq_len`, and prefill compute runs over `from..to` = the prompt
(`neuralnet.cpp:1603–`, `incremental_forwarding(from, to)`), which is why doc
51 saw M = 444 rows under the PR author's 1024 config. So 1024 does **not**
double prefill work. It does three things:

1. **Memory.** Every activation tensor is shaped `[1, 1, init_seq_len, …]`
   (`lfm2_causallm.cpp:404, 413`; `transformer.cpp:248`); the planner's
   pools double. Under #216's finding (kswapd evicting the model file's
   page cache during a run is what makes a pool miss 3.7–5 ms, LEDGER rule 61
   amended) extra resident memory is not free on the E2E path. The handoff's
   peak-RSS column and the `pgpgin` note read it.
2. **First token.** `causal_lm.cpp:715–716`: `if (init_len < INIT_SEQ_LEN)
   registerOutputs(...)`. With prompt 512 and `init_seq_len 512` the first
   generated token is **not** appended to the printed text (it is still fed
   to the loop, `:745`). With 1024 it is. Every text of record so far (prompt
   512, p01 of the 8-prompt set) lacks that token; under the new config texts
   gain one leading token. Text comparisons across the two configs drop the
   first token of the new one; comparisons inside one config are unaffected.
   `docs/measurements/prompts/README.md` ("every prompt ≤ 512 tokens,
   `init_seq_len: 512`") is updated with the fact.
3. **Capacity.** Prompts up to 1024 tokens become possible (the embedding
   path clamps at `init_seq_len`, `lfm2_causallm.cpp:522–525`). Contract
   §1.1 fixes prompt 512; no open plan needs 1024. Load-time warm-ups stay
   at M = 512 (`transformer.cpp:629, 651`), so a first 1024-token prefill
   would grow scratch once.

Expected: prefill and decode tok/s inside A's drift (compute unchanged),
RSS up, text = old + one leading token. Without a prompt-1024 need the
honest default is **keep 512** and record the first-token artefact; adopt
1024 only if the sitting reads RSS and tok/s flat and the user wants the
capacity. Both outcomes are fine; the sitting decides.

### 3.3 Engine keys

Decode: no change by construction. Hybrid A — every FC gate declines M == 1
(`float_tensor.cpp:818, 832, 1038, 1111` with `accelerates_q4_0_at_m1() ==
false`, `htp_compute_ops.cpp:1185`; `conv_block_layer.cpp:189–195` `rows >
1`), so the decode row runs the same CPU kernels whatever the key says. The
conv block's one-layer form folds the four layers "byte for byte"
(`conv_block_layer.cpp:205–212`); the hybrid decode A/B in the handoff
confirms the fold costs nothing. E2E Q28 — every kind is resident and
`decode_row_resident` (`htp_compute_ops.cpp:2124–2130`) skips the CPU FCs;
the keys are read only at prefill (`invokeMoeLayer`'s "these calls (the
prefill's) stay per layer", `:1426–1440`).

Prefill (M > 1) is where they act, and the measured basis is doc 51
§2.17–2.21 and §2.24–2.27 (PR author's tree, prompt 444, `NNTR_PPL`):

| key | prefill Δ (ms of ≈ 870) | prefill PPL vs A 57.00 | D2 (≤ 1.02 × A) |
|---|---|---|---|
| `conv_block_engine` alone | **−150** (conv call 4.0 ms × 19 vs the CPU's four layers) | 57.12 (+0.2 %) | pass |
| + `attn_proj_engine` | ≈ −88 (qkv one call since §2.23) | ≈ +4.5 % | **fail** |
| + `dense_ffn_engine` | ≈ −10 | +4.2 % (down's 224:1 re-quantization depth, §2.19) | **fail** |
| all three | 618 → 585 ms with §2.22–2.23 | 62.09 (+8.9 %) | fail |

Is the verdict stale? Checked against this tree: the prefill FC path still
re-quantizes Q4_0 → qs4cx at load (`htp_qs4cx_from_q4_0x4`, `:4838`,
`:4953–4956`), unchanged. What changed since (upstream sync −12.8 % MoE
`dsp`, the VTCM feed, dspqueue, the DMA bypass, the `_det` kernels, plan
201's exact `hvx_intrin` FC — LEDGER rule 49) is decode-side or MoE-side;
none touches the HMX FC's quantization points, so the PPL costs stand and
the −150 ms stands as an estimate (the conv block's CPU cost is the same
four Q4_0 GEMMs it was). On today's A (≈ 0.9 s for 512 tokens, 528–574
tok/s) **−150 ms ≈ +18–20 % prefill tok/s**; it must be read on this tree,
on this unit, in one sitting, because doc 51's A was the PR author's tree
and prompt.

Design choice: **`conv_block_engine: htp` is the only engine key the config
of record may gain, and only after the sitting.** `attn_proj_engine` and
`dense_ffn_engine` are rejected on D2 as measured (+4.5 / +4.2 % PPL for
−88 / −10 ms); their only path back is an offline qs4cx quantization of
those weights from f32 (doc 51 §2.19), a separate issue if ever wanted.

Rejected alternative: adopt the proposal wholesale now and re-anchor later.
Rejected because (1) `model_file_name` breaks every runner the same day,
(2) three behaviour changes land unmeasured, two of which fail D2 on
numbers already in hand, (3) the row of record becomes incomparable with
no side-by-side A (Method's rule for moving the "now").

### 3.4 Address space and registry under Q28

The engine keys register a second, qs4cx-WH copy of the Q4_0 FC weights
(`registerRm` `:4878` into arena free room, else the DSP heap) beside the
E2E FC set (Q4M1 bytes, 448 MiB, `registerQ4m1` `:1750–1775`). Conv only:
302 M params (LEDGER rule 42) ≈ 150 MiB + scales. Q28 maps the pool (28 ×
5.25 MiB) + 448 MiB FC set, far under the 3840 MiB wall, so conv-only fits;
the Q28 cell in the handoff is the proof (`resident N MiB`, the ceiling
check). Registry slot capacity was dropped in PR #212 (rule 60).

## 4. Steps

1. **Classify and stage the config (workstation, no device).** Write the
   candidate configs into a scratch dir, verify with a 20-line Python check
   (JSON parses; the engine keys are a subset of `{cpu, htp}`; `model_file_name`
   and `tokenizer_file` resolve relative to the model dir exactly as
   `main.cpp:85–95` does; `max_seq_len ≥ init_seq_len`; diff vs the current
   file prints only the intended keys). Gate: rung 1 of `hexagon-gates` is
   `ninja -C build` for code; a JSON change has no code rung, so the check
   above plus `python3 -m json.tool` is the host gate. Optional: run
   `build/Applications/CausalLM/nntrainer_causallm` on the x86 host against
   the `q40` CPU model with the fixed keys (tokenizer relative, `use_embedding`,
   `sample_input`) to prove the parse; the engine keys cannot be exercised on
   x86 (no HTP backend) and `QS4CX_WH` has no CPU fallback (contract §2).
2. **Check the config of record into the tree.** One file
   `docs/measurements/config/q40-qs4cx-wh.nntr_config.json` = current
   workstation config + the three pure fixes (`init_seq_len 512`, no engine
   key), md5 recorded in BENCHMARK Artifacts. The handoff pushes this file;
   the variants are sed-derived from it on the device (as the runners already
   do for `num_to_generate`). Gate: `md5sum` of the staged file equals the
   Artifacts row; `git diff --stat` shows only `docs/`.
3. **Handoff (device, user, device farm) — the unavoidable measurement.**
   `docs/measurements/222-config-ab.md`, filed as an issue comment with
   `needs-user` + `state:needs-measurement`. One binary set (the committed
   `htp_decode` head; skel md5 in every log), one unit, cool start per G
   block (zone0 ≤ 35 °C), prompt 512 (`77-prompt512.txt`), G 64 / 512, two
   runs per cell, A first. Four variants, config only:

   | variant | config | reads |
   |---|---|---|
   | **A** | step 2's file (`init_seq_len 512`, no engine key) = today's behaviour + pure fixes | reference; its equality with the old config is the pure-fix gate |
   | **C1** | A + `init_seq_len: 1024` | §3.2: tok/s flat, RSS up, text = A + one leading token |
   | **C2** | C1 + `conv_block_engine: htp` | §3.3: prefill tok/s (expect +15–20 %), decode flat, prefill PPL and D2 |
   | **Q28-C2** | C2 with `NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=28` | the keys load beside the pool; E2E prefill moves like the hybrid's; decode = Q28's (token unchanged by construction); `resident N MiB`, ceiling 3840 |

   Dropped on purpose: Q28 under A (its decode equals Q28-C2's by
   construction and its prefill delta is C2 − A; the 4-variant cap goes to
   isolating `init_seq_len` from the key) and an "all three keys" cell (fails
   D2 on doc 51's numbers; a user override reopens it as its own sitting).
   G = 1024 is skipped (prefill-side change; the decode row of record is
   re-anchored in a separate mirrored cool sitting once the config is
   chosen). Accuracy block: `NNTR_PPL=1` on the 8 prompts
   (`docs/measurements/prompts/`) for A, C1, C2; `NNTR_PPL_DECODE` forced on
   A's continuation at G = 256 for C2 and Q28-C2; `loop_check.py`; texts of
   G = 64 run 1 pasted; user `text approved: y/n`. Expected log lines: one
   `[HTP] dspq: on`, `applied=0x703e1 … dma_bypass=1 source=default` in every
   hybrid cell; C2 / Q28-C2 print a conv-block registration (`conv block HTP
   kernel warmed up at load`, `ml_logd`) and `NNTR_HTP_PROFILE=2` shows an
   `M>1 conv calls=18` row in one profile run. Estimated 40–50 min.
4. **Fold.** Read every cell against A of the same sitting (`hexagon-handoff`
   "Reading a filled one"). Decide per §3: adopt 1024 only if C1 is flat and
   wanted; adopt the conv key only if C2 passes prefill PPL ≤ 1.02 × A, D2 and
   approval. Update the checked-in file and the workstation copy to the
   chosen config; the device copies follow at the next staging.
5. **Re-anchor the row of record** only if the chosen config differs from
   A's: a mirrored cool sitting, old-config A and new-config A, G 64 / 512 /
   1024, twice each — the Method's rule for moving the "now". If the chosen
   config is A (pure fixes only), the row of record stands and no sitting is
   needed.

## 5. Risks

* **Thermal drift between cells** hides a −5 % prefill: C2's gain (+15–20 %)
  is far above the band, but C1 vs A ("flat") is inside it. Mirrored order
  per G (A C1 C2 Q28, cool, Q28 C2 C1 A) and adjacent-cell reading
  (BENCHMARK cycle-22 paragraph) make it visible.
* **Memory pressure** (`init_seq_len 1024`): more resident activations →
  more page-cache eviction → slower Q28 misses (rule 61). The peak-RSS
  column and `/proc/vmstat pgpgin` around the decode window, as #216 did,
  separate it from the key.
* **First-token artefact** (§3.2): a naive text diff flags C1 as "text
  differs". The handoff states the rule (drop the new config's first token)
  and compares `NNTR_PPL_DECODE` nll lines, which are position-indexed.
* **Stale skel / stub**: none expected (no code), but the conv-block IDL
  call must exist on the device skel (`htp_compute_ops.cpp:4605` names the
  error: "the device predates mm_u8i4_conv_block"). The md5 line in each log
  and the `0x8000040e` stop in the runner catch it.
* **Address space**: conv-only adds ≈ 150 MiB of WH bytes beside the pool
  (§3.4). The `resident N MiB` line and the ceiling check in Q28-C2 read it;
  a load failure there names `NNTR_MOE_CACHE_EXPERTS` (`:1766–1772`) and
  the cell is void, not "at A's speed".
* **Different unit** (farm S26 vs the S25 rows): every number is read only
  against A of the same sitting; the row of record is re-anchored per unit
  column (#208).

## 6. Docs to update

* `docs/htp_moe/BENCHMARK.md`: Method — a "config of record" paragraph
  (path, md5, `init_seq_len`, engine keys, the first-token note) next to
  the "now"; Artifacts — a row for the checked-in config (md5, commit);
  Results — the sitting's rows; Log — the cycle line. If the config of
  record changes: the Goals "now" row carries the new config and the
  re-anchoring sitting.
* `docs/htp_moe/LEDGER.md`: §3a note "engine keys act at prefill only;
  `attn_proj` / `dense_ffn` fail D2 on doc 51's numbers; `causal_lm.cpp:715`
  first-token behaviour at prompt == `init_seq_len`"; a numbered rule only if
  the device disagrees with §3's reasoning.
* `docs/measurements/prompts/README.md`: the `init_seq_len` sentence and
  the first-token rule for p01.
* `docs/measurements/222-config-ab.md`: the handoff (step 3), filled by the
  user.
* Issue #222: the classification table of §3.1 as the comment, so the
  proposal's author sees which keys were taken and why the file name was not.
