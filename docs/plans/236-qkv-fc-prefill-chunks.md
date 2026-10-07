# Plan 236: every M > 1 FC call obeys `prefillRows()` — the hybrid's qkv / o projections at P1024 in 512-row chunks

Issue #236 (hexagon, p1). Base `htp_first_version` @ `0e8888bde` (PR #233
merged). Contract `docs/plans/0001-htp-moe-decode-agent-system.md`; the
measured failure is the #225 sitting (`docs/measurements/225-fcwh-table.md`,
G3, farm `R3CY205ZMND` 2026-10-06); LEDGER rule 64 (the fact), rule 38
(`AEE_ERPC` = the call never reached the entry), rule 8 / doc 46 §41 (the
32-bit DSP address space: 3840 MiB arena + ≈ 182 MiB heap in one 4 GiB
space); PR #230 (`75f6f007d`) is the chunking this plan extends.

## 1. Goal and gate

The issue's acceptance, made measurable:

| check | where it is read | pass |
|---|---|---|
| Host, kernel level | `test/htp/host/run_host_checks.sh` → `fc_layer_host_check` | new line `chunked 512+512 : FC LAYER CHUNKED BIT-IDENTICAL` at M = 1024, K = 2048, three handles N = 2048 / 512 / 512 (the qkv shape), then `ALL CHECKS PASS`. Verified on this tree with a scratch copy of the harness: whole vs 2 × 512 `memcmp` equal, 9.5 s + 8.9 s on the scalar stand-ins |
| Host, the real chunker | `test/htp/host/run_inproc_e2e.sh` lines `E2E keys lfm25-p2x prompt=1024 chunks=512 … logits bit_identical=1 ok` (`:609–618`) | still `bit_identical=1` — after this change the P1024 run chunks the qkv **and** o\_proj FCs too while the `NNTR_HTP_PREFILL_ROWS=0` run (`:420`) does not, so the line now proves the FC chunker's row / block offsets; plus the new count `fc_calls=<2n> whole=<n>` from the `M>1 FC` profile rows (§4 step 2). Every other line unchanged (`calls/token=1.00`, every `bit_identical=1`, `INPROC E2E PASS`) |
| Device (handoff, S25 farm, the user runs) | `docs/measurements/236-qkv-chunks.md` filled | hybrid **B** on the config of record (`cfg_new.json` md5 `3f6808e3…`) **loads and runs P1024 × G64 / G512 / G1024**: `prefill:` and `generation:` lines, `[HTP] fc wh: … heap_kib=<n> requant=0` printed, no `AEE_ERPC` / `VOID` |
| Prefill gate | the same rows | P1024 prefill ≥ −5 % of Bfb's P1024 cells of the #225 sitting (663.2 / 686.3 r2, 650.6, 649.7 tok/s), i.e. ≥ 630 / 618 / 617; the in-sitting anchor B P512 G64 within −5 % of #225's 727.3 (chunking cannot touch P512: M = 512 ≤ step) |
| Decode | the same rows | ≥ Bfb's 52.03 / 51.51 / 48.74 read beside the anchor's drift (#225's B P512 G64 55.17); the CPU of record stays 47.58 / 48.82 / 47.41 (#225, A) |
| E2E Q unchanged | not re-run | Q's prefill goes through the same entries only when `attn_proj_engine` routes a Q4_0 FC — it does, and Q ran P1024 at `heap_kib=0`; the host lines above are the gate (`calls/token=1.00` is decode's, untouched by a prefill-only change). If the user wants a device read, Q P1024 G64 is one optional run (§4 step 5) |
| Standing gates | | bit-identity: the two host lines above; text identical to the CPU run is not achievable for an engine key (rule 39) — T2 applies: B's P1024 texts are **recorded**, the loop check against A's P1024 texts of the #225 sitting (same prompt `p1024.txt`) is printed, the user approves |

BENCHMARK.md cells that move: the LFM2.5 table of record's hybrid P1024
column (Bfb → B) in the `decode tok/s, NPU` and `prefill tok/s` rows, once
the user approves the texts.

## 2. Where it lives

Verified in this tree (`0e8888bde`):

| what | path:line | change |
|---|---|---|
| the FC entry, one weight | `nntrainer/tensor/htp_backend/htp_compute_ops.cpp:1238–1276` `gemm_q4_0_accel_fp32`; the loop `:1267–1275` steps by `fcMaxRows(K)` | step = `fcRowStep(K)` (§3) |
| the FC entry, several weights (the qkv `handles=3` call) | `:1319–1348` `gemm_q4_0_batch_fp32`; loop `:1340–1347`, per-handle `dsts` offsets `m0 * N[i]` | same step |
| the VTCM cap | `:1287–1294` `fcMaxRows(K)` = (7.75 MiB − 4 MiB) / K, 64-row floor → 1920 at K = 2048, 512 at K = 7168 | untouched; `fcRowStep` wraps it |
| the QS4CX twins | `:1362–1373` `gemm_qs4cx_accel_fp32` (one `invokeLayer`, **no chunking at all**), `:1383–1413` `gemm_qs4cx_batch_fp32` (one call into `out_cat`, then per-handle `memcpy`) | the same loop; the batch twin takes `blocks` / `dsts` like the Q4_0 batch and drops `out_cat` |
| the 512-row rule | `:5507–5514` `prefillRows()` (`NNTR_HTP_PREFILL_ROWS`, default 512, 0 = one call) | untouched; read by `fcRowStep` |
| the other chunked prefill entries (already obey it) | `:4636` `invokeMoeLayer` (MoE prefill and, through `:5416–5435` `gemm_q4_0_dense_ffn_fp32`, the dense FFN), `:5493` `gemm_q4_0_conv_block_fp32` | untouched |
| the one call per chunk | `:3425–3480` `invokeLayer` (stages act / out through `stage()` `:3359–3367`, power-of-two ION classes from 64 KiB; `copyOut` with `blocks` / `dsts`) | untouched |
| who sends M > 1 FCs here | `nntrainer/tensor/float_tensor.cpp:817–819` (batch, `M > 1 \|\| accelerates_q4_0_at_m1()`), `:1037–1039` (accel), `:831–833` / `:1110–1112` (QS4CX twins); the layers: `Applications/CausalLM/models/lfm2/lfm2_causallm.cpp:90–106` `qkv_layer` (3 weights, N = 2048 / 512 / 512 at K = 2048) and `:129–134` `*_attention_out` (o\_proj, 2048 × 2048) under `attn_proj_engine`; `conv_in_proj_engine` / `conv_out_proj_engine` (`:184`, `:223`, default `cpu`, not in the config of record) would route the conv projections as plain FCs | untouched. **Not reached by these entries:** the dense FFN (`dense_ffn_engine` → `gemm_q4_0_dense_ffn_fp32`, chunked), the conv block (`conv_block_engine`, chunked), the lm_head (no engine key; CPU on the hybrid, Q4M1 on the one PD), the MoE experts (`gemm_qs4cx_moe_layer_fp32` → `invokeMoeLayer`, chunked). So the complete M > 1 FC list is: qkv (batch), o\_proj (accel), and the two conv projections if their keys are ever set — all four go through the two Q4_0 entries |
| DSP side (read, not changed) | `nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_dma.c:376–490` `hexkl_mm_u8i4_layer_run`: VTCM = act `m_pad × K` u8 (`:429–434`) \| widest handle's WH twice (`:435–439`) \| result tiles; `AEE_ENOMEMORY` only at `:444` (VTCM) and `:483–489` (heap: `loc_scale` + `loc_zp` = 8 bytes × `m_pad`; `acc_scratch` only when the acc layout is unusable); `test/htp/nntr_hvx_mm_u8i4.c:727–756` `check_layer_args` (returns `AEE_EBADPARM` / `AEE_EUNSUPPORTED`, never `AEE_ERPC`) | no DSP source change → no skel rebuild; the #225 set's skel stays valid |
| host checks | `test/htp/host/fc_layer_host_check.c:55–125` (M = 150, K = 64, N 1056 / 512 / 512; 8 MiB VTCM stand-in `:61`); `conv_block_host_check.c:171–194` is the `CONV BLOCK CHUNKED BIT-IDENTICAL` template | new block in `fc_layer_host_check.c` |
| inproc E2E | `test/htp/host/run_inproc_e2e.sh:418–420` (`q25w-p2x` PROMPT=1024 chunked, `q25w-p1x` `NNTR_HTP_PREFILL_ROWS=0`), `:609–618` (the `lfm25-p2x` logits compare + `moe_calls` count) | `NNTR_HTP_PROFILE=1` on both runs; count the `M>1 FC` rows' `calls=` (`htp_compute_ops.cpp:703–725`, kind name `"M>1 FC"`) |
| runner / handoff | `docs/measurements/225-run.sh` (variants, `run()`, `gen()` with the banner strip, `hyb()`), `225-stage.sh` (set + md5.txt, refuses a skel of the wrong arch) | `236-run.sh` = `225-run.sh` cut to B's cells (§4 step 5); the stage script reused |
| profile / report | `NNTR_HTP_PROFILE` buckets are keyed (K, N, M == 1, kind) with `calls` and `rows` (`:703–725`); `tools/htp_fc_report.py` reads rows and times | no table-shape change; the `M>1 FC` row at K = 2048 N = 3072 shows `calls` 12 and `rows` 6144 at P1024 instead of 6 / 6144 — the handoff reads that as "chunked" |

Consumers that do **not** move: the IDL `test/htp/nntr_hvx.idl:83` and its
stub (`mm_u8i4_layer`'s signature is per call; a chunk is a smaller call),
`nntr_quantize_stream`'s format tag (bytes unchanged), the loader check,
the `QS4CX_WH` layout (no CPU fallback question arises: the FCs of the
config of record are Q4_0 on the hybrid).

## 3. Design

**Chosen: one row step for every HTP prefill entry a config key can route.**
A static `fcRowStep(K)` beside `fcMaxRows` returns `min(fcMaxRows(K),
prefillRows())` when `prefillRows() != 0`, else `fcMaxRows(K)` (the
`NNTR_HTP_PREFILL_ROWS=0` host control must still respect VTCM: the P1024
whole call needs 1024 ≤ 1920 rows at K = 2048). The two Q4_0 entries
replace `fcMaxRows(K)` with it (two lines). The two QS4CX twins get the
same loop as their Q4_0 siblings (≈ 15 lines; the batch twin hands
`blocks` / `dsts` to `invokeLayer` and loses its `out_cat` + `memcpy`),
because the ledger item names them and a `fc_dtype QS4CX` model would hit
the identical unchunked M = 1024 shape — stated plainly: no caller on the
config of record reaches them, so their gate is the build and the
existing lines, not a new bit-identity line. No IDL, DSP, quantizer or
loader change. Rows are independent in this kernel (per-row activation
quantization `hvx_quant_pack_u8_ah`, per-row dequant), so chunk outputs are
the whole call's bytes — proven by the host probe in §1 and the
`lfm25-p2x` logits line.

Cost: one more FastRPC call per 512 rows per FC. At P1024 on LFM2.5 that is
6 attention layers × (qkv + o\_proj) = 12 extra calls × ≈ 0.4 ms fixed ≈
5 ms on a ≈ 1.5 s prefill (1024 tokens at ≈ 680 tok/s) ≈ 0.3 %; P512 and
below: zero (M ≤ step). Each chunk re-quantizes only its own rows and
re-streams the weight double buffer once per chunk (2 × 2 MiB per FC per
extra chunk, ≈ 24 MiB more weight DMA at P1024 — under 1 ms at 30 GB/s).

**Why `AEE_ERPC` and not `AEE_ENOMEMORY` — what the host can and cannot
say.** Verifiable in code: (a) the entry and the kernel return only
`AEE_EBADPARM` / `AEE_EUNSUPPORTED` / `AEE_ENOMEMORY`; `AEE_ERPC` is
FastRPC's own code, so the call died in the transport, before
`check_layer_args` (rule 38). (b) The kernel's M-proportional scratch is
small: VTCM act at `m_pad` = 1024, K = 2048 is 2 MiB beside the 4 MiB double
buffer (fits the 7.75 MiB budget, which is exactly `fcMaxRows`'s
arithmetic), and the per-call heap is 8 KiB (`loc_scale` + `loc_zp`) — so
neither VTCM nor the DSP heap is asked for anything M = 1024-sized by the
kernel, consistent with no `ENOMEMORY`. (c) The host side *is*
M-proportional: `stage()` picks power-of-two ION classes — at M = 512 the
act / out pair is 4 MiB / 8 MiB (6 MiB rounds up), at M = 1024 it is
**8 MiB / 16 MiB** (12 MiB rounds up), two ION buffers the DSP PD has never
mapped. Every other P1024 call is chunked at 512 (conv block, dense, MoE:
≤ 4 MiB / 8 MiB classes), so the first attention layer's qkv projection is
the first call in the whole P1024 forward that asks FastRPC to map new,
larger buffers — exactly where B died. (d) B maps all 15 arena chunks
(`arena: chunks unmapped 15/15 mib=3840`) + the FC WH overflow on the heap
(`heap_kib=75776`) + the base heap; Q maps 3520 + 192 = 3712 MiB with
`heap_used_kib` 94 604. Against rule 8's one 4 GiB space (3840 arena + ≈ 182
heap), B sits ≈ 170 MiB closer to the ceiling than Q, and 24 MiB of new
staging mappings is the kind of step that crosses it. **Not verifiable on
the host:** the PD's actual VA ceiling and the driver's error mapping (a
failed `fastrpc` map of a user ION buffer surfacing as `AEE_ERPC`); the
host stub has no FastRPC layer. The plan therefore does not claim the
cause; it removes the M = 1024 staging classes from the hybrid's path
(back to the P512 shapes B already ran) and lets the handoff say whether
that was enough. The optional control run (step 5, run 0: the #225 set
at B P1024 G64, expected to die the same way) separates "fixed" from
"drifted".

**Rejected: find and fix the DSP-side / transport cause.** Nothing in our
DSP code allocates per M here; the suspect is FastRPC's mapping budget,
which we cannot probe from the host and can only move by freeing address
space (PR #233's `ponytail:` — the lm_head slices into the FC chunk's tail,
≈ −64 MiB — or a smaller heap overflow). That is a residency lever for
another issue, measured on its own; the chunk is the way of record for the
three other paths (user 2026-10-06), host-provable, and leaves the 4 GiB
budget where it is. Also rejected: lowering `fcMaxRows` itself to 512 — it
would silently cap the `NNTR_HTP_PREFILL_ROWS=0` control and bake a policy
number into a VTCM fact.

Contract §2 / doc 45 §3: no new DSP heap allocation (the review-list note
on the 32-bit budget is this section), weight DMA stays double-buffered
inside each chunk call, no quantizer is moved (`_det` rule untouched), the
gate is bit-identity on the host + text approval on the device.

## 4. Steps

Each step ends in a `.claude/skills/hexagon-gates` rung. One PR on
`htp_first_version` (`htp/236-fc-prefill-chunks`); the handoff is step 5.

1. **`fcRowStep` and the four entries** (`htp_compute_ops.cpp:1267`,
   `:1340`, `:1372`, `:1400–1412`; the comment at `:1262–1266` names the
   two caps). **Host check:** `fc_layer_host_check.c` gains a block after
   the accumulate case: M = 1024, K = 2048, N = {2048, 512, 512}, whole vs
   two 512-row calls landing each handle's block at `m0 * N[i]` (the
   chunker's `dsts` shape), `memcmp` → `chunked 512+512 : FC LAYER CHUNKED
   BIT-IDENTICAL` (≈ 19 s, measured). Gate: rung 0 (`clang-format-14` on
   the two files), rung 1 — `bash test/htp/host/run_host_checks.sh` prints
   the new line and `ALL CHECKS PASS`; `bash tools/htp_syntax_check.sh`
   exits 0 (`htp_compute_ops.cpp` compiles only in the HTP / inproc
   builds, so this and step 2 are its compile gate).
2. **Inproc: the real chunker.** `run_inproc_e2e.sh:418–420`: add
   `NNTR_HTP_PROFILE=1` to `q25w-p2x` and `q25w-p1x`; at `:609–618` read
   the `M>1 FC` rows' `calls=` from both logs (`fc_calls` = p2x, `whole`
   = p1x) and require `fc_calls = 2 × whole` with equal `rows`, printed on
   the `lfm25-p2x` line. Gate: rung 1 — `bash test/htp/host/run_inproc_e2e.sh`
   prints `E2E keys lfm25-p2x prompt=1024 chunks=512 moe_calls=<n2>
   whole=<n1> fc_calls=<2m> whole=<m> logits bit_identical=1 ok` and every
   existing line unchanged (`E2E keys fcwh-lfm25 … ok`, `calls/token=1.00`,
   `INPROC E2E PASS`). Rung 2 is **not** required (no DSP source or IDL
   change); record the #225 set's skel md5 (`libnntr_hvx_skel.v79.so`
   `58e3a85f…`, device `5508180c…` as pushed) in the handoff so a stale /
   mismatched skel is visible, not assumed.
3. **App build and PR.** Gate: rung 3 (`build_android.sh --htp --cache`,
   both `NEEDED` lines, `NNTR_HTP_FORWARD_KINDS` count ≥ 1, md5s of
   `libnntrainer.so` / `nntrainer_causallm` / `libcausallm_core.so`
   recorded). Open the PR against `htp_first_version` with the host lines
   pasted; DCO + `Co-authored-by`.
4. **Stage.** `225-stage.sh` on the PR's checkout (one binary set: new app
   libs, the same v79 skel, the same sidecar `71812a91…`, the same
   `cfg_new.json` `3f6808e3…`; `md5.txt` written). Gate: the stage
   script's own md5 / arch checks pass.
5. **Device handoff — the unavoidable measurement** (`needs-user` +
   `state:needs-measurement`; `docs/measurements/236-qkv-chunks.md` +
   `236-run.sh`, written by the `hexagon-handoff` skill's shape, derived
   from `225-run.sh`: same `run()`, `gen()` with the banner strip,
   `therm` / `cool`, `p1024.txt` md5 `2e47c5f4…`). Variants (≤ 4, one
   binary set each): **A′** = the #225 set as staged (`md5.txt`
   `23000b96…`, the unchanged reference), **B** = the step-4 set, hybrid
   on the config of record. Runs, in order, after the reboot + 5 min idle
   (rule 61), cool start ≤ 35 °C:
   * run 0 (control, optional but cheap, ≈ 1 min): A′ B P1024 G64 —
     expected to die with the same `nntr_hvx_mm_u8i4_layer failed:
     err=0x80000600 (M=1024 K=2048 N=3072 handles=3)`; recorded, not gated
     (it reproduces the failure in this sitting);
   * run 1: B P512 G64 (the drift anchor vs #225's 727.3 / 55.17; P512 is
     untouched by construction);
   * runs 2–4: B P1024 G64 r1, cool, G64 r2 (mirrored), G512, G1024;
   * optional run 5: B + `NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=28` P1024
     G64 (Q), only if the user wants the E2E cell re-read on this binary.
   Per run the runner checks the `prefill:` / `generation:` lines, `[HTP]
   fc wh: … requant=0`, grep for `0x80000600` / `AEE_ENOMEMORY`, and
   pastes B P1024 G64 r1's text with `loop_check.py` against A's P1024 G64
   text of the #225 sitting (`logs/A_P1024_G64_r1.log` on the measuring
   workstation; same prompt). One `NNTR_HTP_PROFILE=2` run of B P1024 G64
   reads the `M>1 FC` row at K = 2048 N = 3072: `calls` 12, `rows` 6144
   (chunked), and its `us/call`. **Estimate: 6–7 runs ≈ 10–15 min** device
   time plus the reboot / idle. Gate: §1's device rows.

Step 5 is the only device measurement; steps 1–4 are host and build gates.

## 5. Risks

* **The cause is not the staging mappings.** Then B P1024 dies again with
  `AEE_ERPC` at a 512-row call — the handoff shows it beside run 0's
  identical failure (same sitting, same farm), which says "not fixed", not
  "drifted". Next lever is address space (PR #233's lm_head-in-tail
  ponytail), filed as its own issue; Bfb stays the P1024 case of record.
* **Cross-sitting comparison.** Bfb's P1024 cells and the CPU column are
  from the #225 sitting (decode drifts ±5 % between sittings, rule 9;
  thermal drift inside a sitting). Run 1 (B P512 G64) is the in-sitting
  anchor: its ratio to #225's 727.3 / 55.17 is the drift the P1024 cells
  are read through; the table prints both raw and anchored.
* **DVFS / the first-call pad.** The first P1024 chunk call pays
  registration-free but DVFS-cold time; `NNTR_HTP_PROFILE` is read as
  `min`, and tok/s only from the non-profile runs.
* **Stale skel / wrong set.** No DSP change, but the handoff carries the
  skel md5 and `md5.txt` of both sets; `0x8000040e` anywhere is a STOP.
* **Address-space budget drifts with the overflow.** `heap_kib` is printed
  per run; a value above #225's 75 776 means the arena tails were filled
  differently and the reading is not like-for-like — recorded next to the
  cell.
* **Loops at P1024** (Bfb P1024 G64 / G512 looped where A did not). This
  sitting answers whether that was the fallback's (CPU attention FCs) or
  the length's; a loop in B is recorded under T2, not a fail of this plan.

## 6. Docs to update

* `docs/htp_moe/BENCHMARK.md`: the LFM2.5 table of record's hybrid P1024
  column (`decode tok/s, NPU` and `prefill tok/s` rows): Bfb → B with the
  new cells, once the user marks the text approved; the "now" paragraph's
  "at P1024 the hybrid is Bfb" sentence.
* `docs/htp_moe/LEDGER.md`: rule 64 gains its closing sentence (the FC
  entries obey `prefillRows()`; the staging-class reading and the handoff
  verdict); open item ㉞ closes with the handoff reference; the #225 row's
  G3 column points at the #236 row; a new §2 row for the #236 sitting; the
  cycle paragraph.
* `docs/measurements/236-qkv-chunks.md` + `236-run.sh` (step 5).
