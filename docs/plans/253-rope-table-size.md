# Plan 253: size the RoPE cos/sin table by `max_timestep`, not `max_position_embeddings`

Issue: dlwlzzero/nntrainer#253 (p1). Found in #201 S5-0
(`docs/measurements/201-s5-0-gemma-dummy.md` §Step 1, PR #252). Tree:
`htp_decode` @ `91f701fda`. One-function fix on the CPU side; rung 1 only.

## 1. Goal and gate

From the issue: *size the table by `max_timestep` (the cache length) instead
of `max_position_embeddings`, keep `original_max_position_embeddings` for the
proportional rope's scaling only.*

Measurable:

* **Host load, original config.** The repacked 26B set at
  `/local/mnt/workspace/models/gemma4_26b/` (plan 250 step 3; its
  `config.json` is the user's original, `max_position_embeddings` 262144,
  **not edited**) loads and generates through `build_htp_host`'s
  `htp_e2e_test --moe-engine htp --prompt 512 --steps 8` under
  `/usr/bin/time -v`. Gate: exit 0, the `E2E gen` line, and the
  `Maximum resident set size` line recorded before and after the change;
  the after-number is lower by ≈ 1.5 GiB (§3 arithmetic). The
  `max_seq_len 2048` / 4096 workaround of the sitting is not used anywhere.
* **Bit identity.** `bash test/htp/host/run_inproc_e2e.sh` prints every
  gemma64 and hd64 line of the skill unchanged, including the printed
  `min_snr_db=` values digit for digit (`E2E eval gemma64-off-vs-cpu`,
  `gemma64-e3`, `fwd-hd64`, `ppl-decode hd64 … delta=`): the rows the fix
  keeps are the same floats, so nothing downstream may move.
* **Reference tests.** `./build/Applications/CausalLM/unittest_causallm_models
  --gtest_filter='*Gemma4*'` all `[  PASSED  ]`, none skipped (the tiny
  fixtures have `max_position_embeddings == max_seq_len`, 8 / 32, so the
  table has the same rows before and after).
* Standing gates: prefill ≥ −5 % of variant A (not touched by this change;
  the table is built once per layer kind, and the CPU prefill gets faster,
  doc 55 §10.11), text identical to the CPU run (the CPU run *is* what
  changes; identity is the `tokens … 8/8` lines above).

Device confirmation (the 26B loads on the S25 with the original
`config.json`, first decode token without the Scudo abort) rides the next
#201 S5 sitting — no handoff of its own.

## 2. Where it lives

All in `Applications/CausalLM/layers/mha_core.cpp`, verified on `91f701fda`:

* `:681` — `htpDecodeAttention::build_rope_table`:
  `precompute_freqs(head_dim, max_position_embeddings, theta, false)`, then
  `:686` checks `freqs_fp32->cos.size() < rows` with `rows = max_timestep`
  and copies exactly `rows` positions into `htp_rope_table_` (the DSP's
  table, `[max_timestep][cos hd/2 | sin hd/2]`).
* `:1551` — `apply_rotary_emb_tensor_v2`, fp32 branch: same call.
* `:1602` — fp16 branch (`ENABLE_FP16`, i.e. the Android prefill):
  `precompute_freqs(head_dim, max_position_embeddings, theta, true)`.
* `:1329` `precompute_freqs(head_dim, seq_len, theta, is_fp16)` —
  allocates `cos`/`sin` as `seq_len` rows of `std::vector<float|_FP16>(head_dim)`
  and fills them with `calc_trigonometric_vals_dup`; the entry is memoised in
  the static maps `rope_cache_fp32` / `rope_cache_fp16` under
  `getRopeCacheKey(head_dim, seq_len, theta)` (`:1394`, key also carries
  `rope_scaling_type | scale | partial_rotary_factor | original_max_position_embeddings`),
  so the 25 sliding layers share one table and the 5 full layers another.
* Comment `:1327` "`seq_len -> max_position_embeddings`" and the header
  doc of `precompute_freqs` (`mha_core.h:518-525`).

What stays: `max_position_embeddings` is still a property every model sets
(`gemma4_causallm.cpp:594/735`, `transformer.cpp`), so the member and the
property stay; after the change nothing in `mha_core.cpp` reads the member
(it becomes dead but harmless; deleting the property would touch every
model builder for no gain). `original_max_position_embeddings`
(`mha_core.h:487`, read at `:222` only when `rope_scaling_type == "yarn"`)
is used by `_compute_yarn_parameters` alone (`:1478`); **the proportional
type does not use it** — `_compute_proportional_parameters` (`:1417`)
divides the thetas by `scale` (`rope_scaling_factor`, never set by the
Gemma builder → 1.0) and zeroes the thetas past `partial_rotary_factor ×
head_dim / 2`. So nothing about the scaling needs to move.

Consumers that do not move: the IDL / stub, `HtpComputeOps`, the
quantizer format tag, the loader check, the `NNTR_HTP_PROFILE` tables and
`tools/htp_fc_report.py` are untouched (no DSP, format or wire change). The
DSP's own copy is built from this table at `:686-694` with `rows =
max_timestep` and bound at `htp_compute_ops.cpp:2155` against
`graph_words_[6] * head_dim` (`max_seq` of the graph description, set from
`MAX_SEQ_LEN` at `gemma4_moe_causallm.cpp:153`); it already holds
`max_timestep` rows, which is why the fix cannot desync CPU and DSP.

### Why `b2b84250a` does not prevent this

Upstream's lazy allocation moved the build from `finalize` to the first use
and fixed a shadowed `max_position_embeddings` in `finalize` — i.e. it made
sure the table is sized by the **config's** value, and only deferred the
allocation. The size is still `max_position_embeddings`; for LFM2 / Qwen
(≤ 128 K and head_dim 64–128) that is tens of MiB, for Gemma 4 (262 144 ×
head_dim 256 / 512) it is 1.5 GiB. `refs/pr/4408`'s `mha_core.cpp` differs
from ours only by the removed HTP hooks (`git diff 91f701fda refs/pr/4408
-- Applications/CausalLM/layers/mha_core.cpp`: −547 / +3, no RoPE sizing
change), and its doc 55 §10.11 measured the same table as ≈ 800 ms of the
first sliding / full layer's prefill (266 + 591 ms; 262 144 × 2 small
`std::vector` allocations per table) and left it "to measure after the fix".

## 3. Design

**Sizes for the 26B config** (`~/Downloads/gemma4_26b/config.json`:
sliding `head_dim` 256, `default`, theta 1e4; full `global_head_dim` 512,
`proportional`, theta 1e6, `partial_rotary_factor` 0.25;
`max_position_embeddings` 262 144; `nntr_config.json` `max_seq_len` 2048,
`init_seq_len` 1024, `num_to_generate` 512):

| table | rows today | fp32 bytes today | fp16 (Android prefill) | rows after | fp32 after | fp16 after |
|---|---|---|---|---|---|---|
| sliding, hd 256 | 262 144 | 2 × 262 144 × 256 × 4 = **512 MiB** | 256 MiB | 2048 | 4 MiB | 2 MiB |
| full, hd 512 | 262 144 | 2 × 262 144 × 512 × 4 = **1 GiB** | 512 MiB | 2048 | 8 MiB | 4 MiB |
| total | | **1.5 GiB** (+ ≈ 24 MiB of vector headers / malloc overhead) | 768 MiB | | 12 MiB | 6 MiB |

The fp16 tables are built during the CPU prefill (`:1602`), the fp32 ones
on the first decode token (`:681`, the DSP hook's request 3, and `:1551`);
the sitting's abort is the fp32 1.5 GiB landing on top of the fp16 768 MiB
and the ≈ 1.3 GiB FC arena.

**What the table must cover: `max_timestep`.** Pinned by the code, not by
taste: `incremental_forwarding` throws when `to > max_timestep`
(`:519-531`, "NYI: cache shift"), and `apply_rotary_emb_tensor_v2` only
indexes `cos[from + h]` with `from + h < to ≤ max_timestep` (`:1560`). No
position ≥ `max_timestep` is ever looked up, on the CPU or by the DSP
(whose copy is `max_timestep` rows already). On this branch `max_timestep`
is the `max_timestep` property = `MAX_SEQ_LEN` (`gemma4_causallm.cpp:593`,
`causal_lm.cpp:155`): `resetInputDimension` is commented out at
`causal_lm.cpp:640`, so `updateTensorsByInputDimensions` (`:1968`, which
would set `height + max_new_tokens`) never runs and the table is `max_seq_len`
rows, 2048 in the user's set, `--max-seq` (= prompt + steps unless given)
in `htp_e2e_test`. `init_seq_len + num_to_generate` is the prompt budget
inside that cache, never larger (`causal_lm.cpp:586`), so sizing by
`max_timestep` is the superset and the one the cache already pays for.

**Numerics unchanged.** Row `i` of the table depends on `i`, the thetas and
`attention_scaling` only — none of the three rope types reads the table
length (`default`: `:1403`; `proportional`: `:1417`; `yarn` reads
`original_max_position_embeddings`, a separate property). A table of 2048
rows is the first 2048 rows of the 262 144-row table, bit for bit, in both
fp32 and fp16. That is the bit-identity gate.

**The change (≈ 6 lines).** In the three call sites pass the layer's
`max_timestep` (`std::get<nntrainer::props::MaxTimestep>(mha_core_props).get()`,
already in scope at `:683` and `:1544`; read it once at `:1602` the same
way) instead of `max_position_embeddings`; fix the `:1327` comment and the
header doc. `getRopeCacheKey` already keys on `seq_len`, so two models with
different `max_seq_len` in one process (the inproc E2E runs several) get
their own entries, as today. The `:686` size check becomes an equality in
practice and stays as the guard.

**Rejected alternative:** compute cos/sin on the fly per position (no
table). It changes the CPU RoPE's arithmetic order (NEON
`calc_trigonometric_vals_dup` vs scalar per step), which is not
bit-identical to today's table and would move every `min_snr_db`; and the
DSP needs the table anyway (`HTP_GRAPH_PARAM_ROPE_TABLE`). Also rejected:
`min(max_position_embeddings, max_timestep)` — a config with
`max_position_embeddings < max_seq_len` would then read past the table
(the latent out-of-bounds today at `:1560`; `std::vector::operator[]`),
whereas `max_timestep` alone can never be exceeded by a legal `from`.

Contract §2 / doc 45 §3: no wall, arena, DMA or quantizer is touched; the
`_det` rule and the "bit-identical + text" gate apply as the gate above.

## 4. Steps

1. Edit the three call sites and the two comments in `mha_core.cpp` /
   `mha_core.h`; `clang-format-14 -i` on the changed lines (rung 0).
2. `ninja -C build`; `./build/Applications/CausalLM/unittest_causallm_models
   --gtest_filter='*Gemma4*:*Lfm2Moe*'` — all passed, none skipped (rung 1).
3. `source tools/htp/env.sh && bash test/htp/host/run_inproc_e2e.sh` —
   `INPROC E2E PASS`; diff the printed gemma64 / hd64 lines (`tokens 8/8`,
   `bit_identical=1`, every `min_snr_db=` value) against a run on
   `91f701fda`: identical (rung 1).
4. Host load of the 26B with the original config, before (on `91f701fda`)
   and after: `/usr/bin/time -v build_htp_host/…/htp_e2e_test --model
   /local/mnt/workspace/models/gemma4_26b --tokenizer …/tokenizer.json
   --moe-engine htp --prompt 512 --steps 8` with `NNTR_MOE_CACHE_EXPERTS=16`
   (plan 250 step 3). Record both `Maximum resident set size` lines and the
   `E2E gen` lines in the PR; the token line is a dummy's and is not a
   number. The host x86 has the RAM to build the 1.5 GiB table, so "before"
   loads too — the gate is the RSS delta, not a crash.
5. PR to `htp_decode` with the before/after RSS and the unchanged E2E lines
   in its body; no skel, no app build (nothing under `test/htp/` or
   `htp_backend/` changed; rungs 2–3 not needed).

**Device measurement:** none filed. The next #201 S5 sitting runs with the
device's `config.orig.json` restored as `config.json` and reports "load:
pass" with peak RSS in its load row; that line closes #253.

## 5. Risks

* **fp16 prefill table.** The Android build (`ENABLE_FP16`) takes the
  `:1602` branch, which the host x86 E2E never executes. The three call
  sites change identically; the PR diff shows the fp16 line. The device
  confirmation above is the only place it runs.
* **Lazy-alloc thread safety.** Unchanged: the double-checked `freqs_fp32 ==
  nullptr` under `rope_init_mtx` (`:678`, `:1548`, `:1600`) is upstream's
  pattern; the key now differs per `max_timestep` but a layer's first use
  is still serialised by the same mutex. Not made worse, not fixed.
* **Positions beyond the table at long generation.** Impossible by
  construction: a step with `to > max_timestep` throws before any lookup
  (`:519`). If `resetInputDimension` is ever re-enabled (`causal_lm.cpp:640`)
  and `max_timestep` grows after the table is built, `:686` throws on the
  DSP path and the CPU path would index past the vector — the same latent
  issue as today with `max_position_embeddings < max_seq_len`; note it as a
  `ponytail:` on the `:686` guard (upgrade path: rebuild when
  `cos.size() < max_timestep`).
* **Host-vs-device.** RSS on the host counts the x86 process (no ION, no
  fastrpc mappings); the device number comes from the sitting's load row.
  The stale-skel trap does not apply (no DSP change). Thermal / DVFS / DMA
  rate: not exposed — nothing per token changes.

## 6. Docs to update

* `docs/htp_moe/BENCHMARK.md`: the Gemma-4 26B goal row — the load now
  runs on the original `config.json`; strike the `max_position_embeddings:
  4096` device-copy caveat once the sitting confirms. Artifacts table: the
  device's `config.json` md5 returns to the original `b2062bfe…`.
* `docs/htp_moe/LEDGER.md`: rule — "the RoPE table is `max_timestep` rows;
  `max_position_embeddings` only reaches `yarn` scaling" (closes the
  measurement doc's "left as a follow-up"); §2 verdict row for #253 with
  the host before/after RSS; ㊶'s load caveat points here.
* `docs/measurements/201-s5-0-gemma-dummy.md` §Step 1: a one-line pointer
  to this plan (the workaround is superseded).
