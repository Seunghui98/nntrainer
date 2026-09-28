# 136 — B's text with all six kinds resident: the worker-pool race between back-to-back DSP calls, host-reproduced; fix, host-gate, confirm on silicon

Issue #136 (tracker #76; LEDGER ㉕, §2 #130 row, rules 36–38; plan 130 §3.1–3.3).
Contract `docs/plans/0001-htp-moe-decode-agent-system.md`. Cycle 20, `htp_moe` @ `c5dcb382`.

## 0. What the host found before this plan was written (2026-09-28, worktree `nntrainer-136`)

The planner built the in-process harness (`build_htp_host/`) and tried to reproduce
the device fault on a fixture at LFM2.5's per-layer shape — hidden 2048, 32 / 8
heads, head_dim 64, `max_seq 2048`, `rope_theta 5e6`, layers
`conv,conv,attention,conv,attention,conv`, 2 dense, `moe_intermediate 256`
(the generator needs a `--layer-types / --num-dense / --rope-theta / --moe-inter`
option set, added on a scratch copy; `moe_intermediate 32` is refused by the MoE
layout with `AEE_EUNSUPPORTED`, the down chunk count exceeds `MOE_MAX_CHUNKS`).

1. **Numerics do not reproduce it.** Prompt 512 + 8 steps, all six kinds vs the
   switch-off run: tokens 8/8, every stretch inside the band plan 130 calibrated
   (`RMSNORM` alone 130 dB, `CONV1D_GATE` alone 105 dB, the attention stretch
   31–45 dB at the MoE inputs — the CPU's fp16 cache rows vs f32, as on hd64);
   `KINDS=MOE` bit-identical. The one 0.94 dB reading (`moe_00010_out`) is a
   router top-k flip on random weights at a near-tie, not wiring. Prompt sweep
   16 / 32 / 33 / 64 / 256 / 384 / 448 / 480 / 496 / 511 / 512: same band everywhere.
   So candidates 1–5 of the issue (KV seed, RoPE at pos ≥ 512, conv state seed,
   `graph_set_param` binding, wire-v2 fields) are **cleared on the host at the real
   shape and prompt length**; the device gtests had already cleared the kernels
   and the bulk 1023-row seed at 8 / 4 / 64 (LEDGER rule 37).
2. **The issue's mask ladder is not executable.** `HtpComputeOps::set_decode_graph_desc`
   (`nntrainer/tensor/htp_backend/htp_compute_ops.cpp:1302-1310`) throws on a
   resident `QK_NORM` or `ROPE` without `ATTN_M1` (no CPU layer consumes their
   output). The executable intermediate masks are `MOE,RMSNORM,CONV1D_GATE`,
   `MOE,ROPE,ATTN_M1` and `MOE,QK_NORM,ROPE,ATTN_M1` (all verified on the host:
   17 / 8 / 6 / 6 calls per token on the fixture).
3. **The fault reproduces as a crash under load.** With all six kinds resident,
   prompt 16, the harness segfaults intermittently: 0/12 alone, 0/12 under gdb,
   **5/30 when six instances run at once**. Every core (`/var/lib/apport/coredump/`,
   three read) has the same shape: a pool worker thread is inside
   `moe_m1_down_worker` (`hmx/hexkl_mm_u8i4_moe.c:739`) while the main thread is
   already in the CPU's next `nntr_gemv_q4_0_8x8_q8_0` — the `hvx_worker_pool_run`
   that dispatched the down phase (`:1459` / `:1465`) **returned before one of its
   participants finished**. Subset masks (`MOE,RMSNORM`, `MOE,CONV1D_GATE`,
   `MOE,ROPE,ATTN_M1`, `MOE,QK_NORM,ROPE,ATTN_M1`) did not crash in the same
   loops (small samples). `MALLOC_PERTURB_` leaves every dump bit-identical, so it
   is not an uninitialised read.

   Mechanism, from `hvx/hvx_worker_pool.c:135-178` (the worker loop) and
   `:379-420` (`run`): a worker reads `fg_id`, and only afterwards reads
   `pool->async`, `pool->n_threads`, `pool->func`, `pool->ctx` — plain fields the
   caller rewrites for the next job. A worker that observed job J's `fg_id` and
   is descheduled before reading the fields (a loaded host; on the DSP a worker
   woken from `qurt_futex_wait` at a call boundary, which is where the per-token
   entry differs from the per-layer path: 95 wake-ups per token instead of 22 and
   a pool job in the very next FastRPC call — `ATTN_M1` and `CONV1D_GATE` use the
   same `s->quant_pool`, `hvx_attn_m1_f32.c:335`, `hexkl_graph.c:157`) resumes
   with J+1's fields, joins J+1 uncounted, decrements J+1's `barrier`, and — its
   `prev_fg` still J — runs J+1 a second time on the next loop. J+1's caller
   sees `barrier == 0` while a real participant is still writing: on the host a
   worker runs on a freed / reused job context (the segfault), on the device the
   caller copies `out` from the slot before the last unit wrote it (**silent wrong
   output**: an incomplete MoE down or attention row, every token) — the
   "plausible wrong text" class of doc 46 §48.7, and why C (`KINDS=MOE`, one pool
   job cluster per MoE call, ms of CPU work between calls) reads ≡ A.

   This is the named cause the plan builds on. It is not proven on silicon yet:
   step 4 is the confirmation sitting, and step 3 keeps the mask bisect as the
   fallback if B's text does not return to A's after the fix.

## 1. Goal and gate

Acceptance criterion (issue, made measurable):

* **Named op and cause, host-gated.** The cause is the worker-pool job pickup
  race above (§0.3); the "op" is every pool user in a stretch (MOE's M=1 down /
  requant / pair phases, ATTN_M1, CONV1D_GATE). Host gates, all in one PR:
  `run_host_checks.sh` `ALL CHECKS PASS` + `WORKER POOL LANES OK` with the new
  pool race check (`POOL RACE OK`, §3.1) green; `graph_host_check` unchanged;
  `run_inproc_e2e.sh` `INPROC E2E PASS` on both existing fixtures **and** the new
  real-shape case (`E2E fwd lfm25 kinds=<all six> calls/token=23.00`, `E2E eval
  fwd-lfm25 … min_snr_db=` ≥ 30, `E2E tokens fwd==off 8/8`); the load loop
  (`test/htp/host/run_inproc_load.sh`, §3.1) `LOAD LOOP 30/30 rc=0` where today
  it is 25/30; `unittest_causallm_models --gtest_filter='*Lfm2Moe*'` 6 passed;
  `ninja -C build`; `tools/htp_syntax_check.sh`; `clang-format-14`.
* **Device, same sitting as its A (step 4):** all six kinds resident at G = 64:
  **text ≡ A byte for byte, or `text approved: y`** (2026-09-28 rule; `PPL` n/a
  until #134); `calls/token=95.00`; decode within the −22 / −25 / −30 % already
  measured (the fix removes a race, not calls — if B moves *up* it is because the
  early return had been skipping work, note it, do not read it as a lever);
  **prefill ≥ −5 % of A** (BENCHMARK.md `#136` rows, prefill column); ride-along
  `HvxM1Ops.*` / `HvxAttnM1.*` results copied (they belong to #137).
* Rungs 2–3 on the workstation with md5s in the PR; the handoff's sanity block
  carries `strings libcausallm_core.so | grep -c NNTR_HTP_FORWARD_KINDS` ≥ 1
  (rule 36; the first device use of #135's committed export).

Standing gates: prefill ≥ −5 % of variant A; text identical to the CPU `q40` run is
n/a for `QS4CX_WH` (contract §2) — the text column is "= A" + approval.

## 2. Where it lives

Verified at `c5dcb382`.

| file | what changes |
|---|---|
| `nntrainer/tensor/htp_backend/hvx/hvx_worker_pool.c:135-178` (worker loop), `:272-296` (`submit`), `:379-420` (`run`) | **the fix** (§3.1): the job's fields are snapshotted under the same generation the worker acts on, and a worker that lost the race does not participate |
| `nntrainer/tensor/htp_backend/hvx/hvx_worker_pool.h:39` | no API change intended; a `hvx_worker_pool_job` struct only if the snapshot needs one (§3.1) |
| `test/htp/host/worker_pool_host_check.c` (the existing `WORKER POOL LANES OK` check; `run_host_checks.sh` compiles it) | **new case `POOL RACE OK`**: two jobs back to back with different `n_units` (4 then 8, the requant → attention shape) × 10 000 rounds on a 3-worker pool under a busy-loop load thread, each round asserting every unit's output written before `run` returned; must fail on today's pool (the check ships only once it has been seen red) |
| `test/htp/host/run_inproc_load.sh` (new, ≈ 30 lines) | six concurrent `htp_e2e_test` runs × 5 rounds, all six kinds, prompt 16, the real-shape fixture; prints `LOAD LOOP <ok>/30 rc=0`; the load is the reproducer (§0.3), so it is a gate, not a benchmark |
| `test/unittest/models/causallm_reference/generators/generate_lfm2_moe_reference.py:447-470` | `--layer-types`, `--num-dense`, `--rope-theta`, `--moe-inter` (defaults = today's; the hd8 / hd64 fixtures byte-identical) |
| `test/unittest/models/causallm_reference/lfm2_moe_tiny_lfm25/` (new: `config.json`, `nntr_config.json`, `tokenizer.json`, `meta.json`, reference JSONs; the 350 MB `.bin` gitignored and generated per checkout like the other two) | the real-shape fixture of §0 |
| `test/htp/host/run_inproc_e2e.sh` | the `lfm25` cells (switch off, all six kinds, prompt 512, `--max-seq 2048`, SNR floor 30, tokens policy, init lines `n_ops=58`, `attn_m1: registered layers=2 kv=8 gqa=4 head_dim=64 max_seq=2048`) and the golden for it; `README.md` under `golden/lfm2_moe_tiny_lfm25/` |
| `nntrainer/tensor/htp_backend/htp_compute_ops.cpp:2419-2421` (`invokeForward`'s `dumpMoeCall` is MoE-only) | **dump ride-along**: under `NNTR_HTP_DUMP_ALL=1` every stretch's `act_in` / `out` is written as `fwd_<n>_<kinds>_{in,out}.f32` with its own `forward_manifest.txt`, plus `kvseed_<ordinal>_{k,v}.f32` from `decode_kv_seed_fp32` (`:1542-1562`) and `convstate_<op>.f32` from the `CONV1D_GATE` case (`:1476-1489`). Separate manifest so the existing goldens stay byte-equal. The device handoff turns it on for one B run at G = 64 |
| `docs/measurements/136-forward-text.md` | the confirmation sitting (§4 step 4) |

Unchanged, on purpose: the IDL `test/htp/nntr_hvx.idl` and both stubs (no
method changes — the pool is inside the skel), `HtpComputeOps`' contract,
`nntr_quantize_stream`, the loader, `htp_graph_desc.h`, every kernel and `_det`
spec, `NNTR_HTP_PROFILE` tables and `tools/htp_fc_report.py`. The skel **is**
rebuilt (the pool is a skel source; `test/htp/build.sh:77-89` lists it) — rule 3.

Consumers of the pool that the fix must not slow: the prefill MoE path's
`submit` / `wait` / `submit_bg` chain (`hexkl_mm_u8i4_moe.c:1514-1948`), the M=1
GEMV's `run` calls (`:1419-1465`), `hvx_conv_gate_f32`, `hvx_attn_m1_forward`.
The prefill gate (−5 %) is what reads that.

## 3. Design

### 3.1 The fix: a worker acts only on the job it saw published (chosen)

Today the job is five plain fields plus `fg_id`. Make the worker's read a
snapshot that is *checked* against the generation it is about to serve: read
`fg_id` (acquire) → read `func / ctx / n_threads / async` → re-read `fg_id`; if
it moved, loop (the caller published a newer job; this worker never belonged to
the older one). The caller side already writes the fields before the release
increment, so with the re-check the fields a worker acts on are exactly those of
the generation it will decrement the barrier for. `prev_fg` is set to the
generation actually served, so no job is served twice. The `submit` path
(`:272-296`) shares the loop and gets the same guarantee; the background ring is
untouched (its jobs are caller-owned and never reused, `:75-80`).

Cost: one extra atomic load per job pickup on the worker — nothing on the caller,
nothing per unit. The spin / futex sleep logic (`:160-176`) is unchanged.

`ponytail:` the double-read is a seqlock without the write-side odd/even
generation; it is enough because the caller never publishes a job while one is
outstanding (`run` and `submit` both `wait` first, `:381`, `:283`). If a future
caller wants overlapping foreground jobs, the fields move into a per-generation
job slot (the `hvx_worker_pool_job` ring, like `bg_ring`).

*Rejected: making `run` wait until every worker has acknowledged the generation
(a second barrier).* It fixes the same race but adds a wake-up round trip per
`run` — the M=1 MoE path runs ~10 `run`s per call, the profile columns
`DEQUANT` / `REQUANT` already carry the wake cost (`:160-166` comment), and the
prefill gate would read it.

*Rejected: fixing it on the ARM side* (a delay or a dummy call between stretches).
The race is inside the pool; every DSP entry that runs two pool jobs in quick
succession is exposed, including #132's larger resident set.

### 3.2 Why the bisect stays in the plan (fallback, step 3)

§0.3 is reproduced and explained on the host but not yet seen on the phone. If
B's text after the fix is not ≡ A (or approved), the sitting of step 4 already
contains the two executable intermediate masks (`MOE,RMSNORM,CONV1D_GATE`,
`MOE,ROPE,ATTN_M1` — 4 variants with A and B, contract §4.2), and `NNTR_HTP_DUMP_ALL`
on B gives every stretch's device input / output; the workstation replays those
inputs through `graph_host_check`'s emulation (the same `_det` specs) and names
the first stretch whose device output leaves the emulation's — a DSP-side fault —
or finds them equal, which puts the fault in what the ARM fed (then the dumped
`kvseed` / `convstate` / stretch inputs are compared against the switch-off run
of the same token on the workstation). The issue's five-step ladder is replaced
by these two masks because of §0.2.

### 3.3 Rules kept

Contract §2 (three walls, arena budget, no CPU fallback for `QS4CX_WH`) —
untouched; nothing new lives on the DSP heap. Doc 45 §3.1 (activation handles)
and §3.2 (DMA behind compute) — untouched. §3.3 — no new op in front of a
quantizer. §3.4 — the gate is bit-identity where it can be (`POOL RACE OK`,
`graph_host_check`, the goldens) plus text on the device.

## 4. Steps

Each step ends in a gate of `.claude/skills/hexagon-gates`.

1. **Reproduce and pin (host, rung 1).** Add `POOL RACE OK` to
   `worker_pool_host_check.c` and `run_inproc_load.sh` (with the real-shape fixture
   from the generator options). Gate: the new pool check is **red** on
   `c5dcb382` and the load loop reads ≈ 25/30 (both recorded in the PR body);
   `run_host_checks.sh` otherwise unchanged. A TSan build of the harness
   (`-Db_sanitize=thread`) did not link in the planning session; an ASan build
   is optional — the load loop is the reproducer of record.
2. **Fix the pool (host, rung 1 → rung 2).** §3.1 in `hvx_worker_pool.c`. Gate:
   `POOL RACE OK` green over 10 000 rounds, `WORKER POOL LANES OK`,
   `ALL CHECKS PASS`; `run_inproc_load.sh` `30/30`; `run_inproc_e2e.sh`
   `INPROC E2E PASS` with the new `lfm25` cells; `*Lfm2Moe*` 6 passed;
   `ninja -C build`; syntax check; clang-format. Then `test/htp/build.sh`
   (`-Wall -Werror`, `UNDEFINED SYMBOLS OK`), md5 recorded.
3. **Dump ride-along + app (host, rung 3).** `NNTR_HTP_DUMP_ALL` in
   `invokeForward` / the seed paths; `build_android.sh --htp --cache` and the
   two gtest binaries; `strings` checks of rule 36; md5s. PR into `htp_moe`
   (`state:review`) and the handoff below (`state:needs-measurement`).
4. **Device (unavoidable): the confirmation sitting**, `docs/measurements/136-forward-text.md`,
   ≈ 30 min, one binary set:

   | variant | env | cells |
   |---|---|---|
   | A | none | G = 64 / 512 / 1024 × 1 |
   | B | `NNTR_HTP_FORWARD=1` (all six) | G = 64 / 512 / 1024 × 1, mirrored with A; plus one G = 64 run with `NNTR_HTP_DUMP_ALL=1 NNTR_HTP_DUMP=/data/local/tmp/htp_dump` (pulled, not read for tok/s) |
   | B1 | `NNTR_HTP_FORWARD_KINDS=MOE,RMSNORM,CONV1D_GATE` | G = 64 × 1 — read only if B's text ≠ A |
   | B2 | `NNTR_HTP_FORWARD_KINDS=MOE,ROPE,ATTN_M1` | G = 64 × 1 — same |

   Ride-along: the two gtests on this skel (for #137). Expected lines: B's
   `graph: init … resident=RMSNORM\|CONV1D_GATE\|QK_NORM\|ROPE\|ATTN_M1\|MOE moe_ops=22`,
   `attn_m1: registered layers=6 kv=8 gqa=4 head_dim=64 max_seq=2048`,
   `calls/token=95.00`; B1 `47.00`-class and B2 `40.00`-class counts are the
   stretch arithmetic on the real list (copied, not judged). Verdict: B text ≡ A
   (or approved) closes the issue after the supervisor folds the rows; otherwise
   §3.2's replay names the op and the issue goes back to `in-progress` with the
   dumps.

## 5. Risks

* **The race may not be the whole story on silicon.** The host shows a 5/30
  crash under load; the device shows a deterministic wrong text. QuRT's
  scheduling makes the window rarer per boundary but there are 95 boundaries per
  token and every one lands on a corrupted out row if hit; a second, DSP-only
  cause (heap / address space at 48 MiB + 3.84 GB mapped, FastRPC buffer
  handling) would survive the fix. The handoff sees it as B ≠ A with B1 / B2 and
  the dump replay (§3.2) in the same sitting — no second sitting to find out.
* **Thermal drift between A and B cells** — mirrored order per G, checkpoints in
  the therm log, decode read only inside the sitting (rule 20). The decode
  column is not this issue's gate beyond the −22 / −25 / −30 % envelope.
* **DVFS / DMA rate** — untouched; the prefill column is the sensor for a pool
  regression (the prefill MoE path runs the same pool), −5 % is the gate.
* **Stale skel** — the pool is skel code: `0x8000040e` would name it; the skel
  md5 is in every log (rule 3 / 21).
* **Address space** — nothing new on the DSP heap; the 48 MiB cache stays as in
  #130.
* **Host-only artefact** — the host stand-in's `qurt_futex_*` is a pthread
  emulation; if the load loop still crashes after the fix, the stand-in itself is
  the next suspect (`test/htp/host/inproc/qurt.h`), and the device sitting is
  the arbiter.

## 6. Docs to update

* `docs/htp_moe/BENCHMARK.md`: `#136` rows (A / B / B1 / B2, unit, md5s, text
  column, approval), the artifacts row for the first post-#135 app set, Method
  note that the "now" stays #120 A until a cool sitting reproduces it.
* `docs/htp_moe/LEDGER.md`: ㉕ → measured; a new rule (proposed 39): *a pool
  worker acts only on the job generation it saw published — back-to-back DSP
  calls (the per-token entry) expose a pickup race the per-layer path never did;
  the host reproduces it only under load; the load loop is a gate*; §2 verdict
  row for #136; ⑨ updated (the six-kind set's text); ㉖ / #137 unchanged.
* `docs/plans/130-per-token-wiring.md` — not edited (history); this plan
  corrects its mask assumption in §0.2.
