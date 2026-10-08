# 258 — The one-PD decode FC on the WH sidecar: the u8 per-row activation quantization (cause b), separated from the dummy's double int4 (cause a)

Tree `htp_decode` @ `575766532`. Inputs: issue #258,
`docs/measurements/234-p4-fcwh-decode.md` (S25, dummy, G 512 F16 15.89
tok/s, FC 10.65 ms, rms(F − A) 1.284, forced ppl +12.6 %), plan 229 §8 (FC
source format **A**: `QS4CX` per-column int4, the sidecar a repack — cause
(a) is closed for the real file; this plan is cause (b)), plan 130 §3.5,
LEDGER rules 47 / 58 / 73, doc 45 §3. Tags: [M] measured, [G] config, [E]
estimate.

## 1. Goal and gate

Issue: *how much of the shift is (b), and does it break the +2 % PPL rule
on its own; if so, the cheapest lever and its decode cost.* Measurable:

* **Host, the split.** `run_inproc_e2e.sh` prints, per op kind, the WH
  token against the Q4M1 token on gemma64x (int4-exact FCs, the fixture's
  **own** q / k gammas — not 0.3): that is (b) alone. And on gemma64 the
  from-Q4_0 sidecar against the f32-sourced one: that is (a) alone.
* **Host, the fix.** `E2E eval gemma64x-fcwh-vs-q4m1 … min_snr_db ≥ 20`
  with the 0.3 override deleted (expect ≥ 40 [E], §3), `E2E tokens
  gemma64x-fcwh==q4m1 8/8 expected_mismatch=0`, a new `E2E ppl-decode
  gemma64x-fcwh q4m1=… wh=… delta=…` with |delta| ≤ 2 % (the real
  sitting's rule); `FC WH BIT-IDENTICAL` against the new `fc_wh_det.h`,
  both `FC WH MUTANT CAUGHT`; every LFM line unchanged (`e3fcwh==off-lfm25
  8/8`, `FcWhSidecarMatchesWhQuantize` 7 / 7).
* **Device (rides plan 229 S2.5, the real-file sitting).** F16 vs A16:
  decode PPL ≤ 1.02 × A (F forced on A's ids), tokens by plan 130 §3.5,
  rms(F − A) printed; FC ≤ 11.2 ms/token (P4's 10.65 + 5 %), tok/s in
  rule 52's band of P4's F16; prefill ≥ 0.95 × A same sitting; text
  identical to the CPU run where one exists (the dummy has none, rule 73).

What SNR the real decode needs: no dB converts to the PPL rule (E16's rms
0.130 read −0.01 %, F16's 1.284 read +4.3 %; the real file's margins are
unknown), so the gate is the PPL rule itself, on the fixture first and on
the real file last; 20 dB stays as the tripwire (a wrong part reads < 0).

## 2. Where it lives (verified)

* Kernel: `nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.c:2792`
  `hexkl_mm_u8i4_fc_m1_run`; the quantization `:2841-2846`
  (`hvx_quant_rows_u8_params` on **one row of K**, `hvx_quant_pack_u8_ah_rows`
  into a `k_tiles × 2048 B` AH block); the worker `:2732-2790`, its GEMV
  call `:2775-2783` (`hvx_gemm_u8i4_wh_col[_nopf]`) and dequant `:2784-2788`
  (`hvx_dequant_acc_tile_to_f32`: `(sum − zp·colsum)·scale·w_scale + bias`).
* The u8 quantizer: `hvx/hvx_quant_u8.h:20-37` (asymmetric, range includes
  0, one scale + zp **per row**). The reference path the fixture compares
  against: `hmx/hexkl_graph.c:418-425` `graph_prep` → `hvx_q4m1_prep`
  (`hvx/hvx_q4_gemv_f32.h:33-55`): the CPU's **Q8_0, per 32-block int8,
  symmetric** (`nntrainer/tensor/q4_gemv_cpu_det.h` `q8_0_quant_cpu_det`;
  bit-identity check `test/htp/host/q4_gemv_host_check.c:309-330`).
  Both paths quantize the activation; WH spends 8 bits over the row's
  range, Q4M1 8 bits per 32 values. The MoE M=1 path is the same per-row
  u8; its dumps are bit-identical because they are DSP-vs-DSP.
* GEMV: `hvx/hvx_gemm_u8i4_wh.c:82-87` (rows4), `:210-216` (row1): per
  k-tile 8 `vrmpyacc` (u8 × i8), one `vasr 4` at the end; 2-bit twin
  `hvx_gemm_u8i2_wh_col_nopf` (LUT → the same i8 operand). f32 helpers
  `hvx/hvx_q4_gemv_f32.c:380-397` (`sf_mpy` / `sf_add`: one qf32 op +
  conversion = one IEEE rounding).
* Graph: `hexkl_graph.c:457-470` `graph_op_fc_wh`; `graph_op_qk_norm`
  `:258-279` — q_norm / k_norm **after** the projection per head, so the FC
  error passes the norm in relative terms and the softmax amplifies it.
* Bind: `htp_backend/htp_compute_ops.cpp:2030-2070` (`FEED_WH` per op;
  q|k|v are **one** FC op); `fcwhFind` `:5271-5286` **refuses** a weight
  missing from the sidecar — no per-kind isolation by a partial sidecar.
* Spec and checks: `test/htp/host/fc_wh_det.h:36-85`;
  `moe_layer_host_check.c:1262-1428`; mutants `run_host_checks.sh:48-68`;
  `run_inproc_e2e.sh:451-486` (gemma64x; `w += [0.3] * hd` is the gamma
  override), `:737-738` (the gate), `:1250-1262` (lfm25 forced PPL);
  `tools/htp/fc_wh_sidecar_from_q4.py` (`--check`: 16.40 dB on gemma64).
* **Unchanged:** IDL / stub, `HtpComputeOps` register calls, `FCWH_FORMAT
  "QS4CX_WH/1"` (image bytes identical), loader check, `NNTR_HTP_PROFILE`
  tables, `tools/htp_fc_report.py`, `hvx_quant_u8.*` (MoE / HMX prefill).

## 3. Design

**Chosen: the M=1 WH FC takes the CPU's Q8_0 activation (per-32 int8,
symmetric) and drains the accumulator per k-tile.** `fc_m1_run` replaces
`:2841-2846` with `hvx_q4m1_prep(act_f32, K, &a)` into its scratch (K int8
+ K/32 f32 `df`; the 2 KB-per-tile AH block goes away). A new
`hvx_gemm_i8i4_wh_col_m1[_nopf](q, df, k_tiles, wh, n_col, nt, out_f32)`:
per k-tile the same 8 products with `Q6_Vw_vrmpyacc_VwVbVb` (signed ×
signed, in the SDK protos), `vasr 4`, `Q6_Vsf_equals_Vw`, `sf_mpy(df[kt])`,
`sf_add` into one f32 accumulator; the epilogue is `out = acc·w_scale +
bias` (no zp, no colsum). The feed, the double buffer, the scoreboard and
`fc_m1_push` are untouched. Spec: `fc_wh_quant_row_det` → the existing
`q8_0_quant_cpu_det`; `fc_wh_col_det` → per tile `s = Σ q·w` (int32),
`acc = RN(acc + RN((float)s · d[kt]))`, `RN(RN(acc · w_scale) + bias)`.

Why it passes without touching the gammas: on int4-exact weights the WH
and Q4M1 tokens then hold the **same activation bytes**; what remains is
the scale placement (per-column vs per-32-block, equal values) and the f32
summation order — the ≥ 60 dB class of `E2E eval cpu`, not 9.5 [E].
Smallest because the quantizer, its spec and its check exist; the diff is
one GEMV variant (≈ 60 lines mirroring `gemm_row1`) and a 3-op epilogue.
Cost on the 26B: +≈ 5 vector ops per k-tile per column on a loop that is
DDR-bound at M=1 (rule 26; plan 229 §3.1: 25 µs compute vs 340 µs DMA per
LFM call) — expected hidden [E], read as FC ≤ 11.2 ms. K ≤ 8192, K % 64
== 0 hold for every 26B / fixture FC [G]. S2's 2-bit LUT yields the same
i8 operand, so one `_m1` loop serves both widths.

Rejected:

* **Per-group u8 asymmetric (32 / 64 columns).** Needs `zp_g · colsum_g`
  per group: either per-tile colsums in the image (+25 % bytes at K 2816,
  a `FCWH_FORMAT` bump) or 8 extra `vrmpy` per tile; the symmetric form
  needs neither and is what the reference already computes.
* **i16 activations for q / k.** `vrmpy` is 8-bit; a 16-bit path is a new
  layout + multiply (×2–4 compute) and a second activation format in one
  op; buys nothing over per-32 int8 against a per-32 int8 reference.
* **q / k (the q|k|v op) back on Q4M1, o / dense WH.** Zero kernel work and
  the fixture passes trivially, but Δ = (14.33 − 10.65) / 1 110 M w =
  3.3 µs/M [M]; q+k = 562 M (50.7 %), q|k|v as one op = 706 M (63.6 %)
  [G] → **+1.9 / +2.3 ms/token**, 15.89 → ≈ 15.3 tok/s [E] — and Q4M1 has
  no 2-bit form, so it forfeits S2's lever 1 on 64 % of the FC bytes (≈ 5
  of its −8.0 ms). Kept only as §4's isolation knob and the device
  fallback.
* **Norm before quantization.** q_norm acts on the FC's output per head;
  the FC is linear, the norm is not: no reordering is allowed. Lowering
  the gammas changes the model, not the path.

Contract §2 / doc 45 §3: no CPU fallback (the knob picks between two NPU
paths), arena and heap unchanged, the DMA schedule is the measured one,
`_det` spec + mutant in the same PR, tokens + PPL against the reference.

## 4. Steps

1. **Split (a) / (b) on the host — loader knob + script lines, no DSP.**
   `NNTR_HTP_FC_WH_SKIP=qkv|o` at `htp_compute_ops.cpp:2049` (an FC op is
   "qkv" when the next op is `QK_NORM`, else "o"); the sidecar still loads
   whole, the skipped ops bind Q4M1 as before — this is also lever 3.
   In `run_inproc_e2e.sh`: (i) gemma64x at the fixture's own gammas as a
   second copy (`fixg_x0`, the `0.3` line skipped), WH vs Q4M1 with skip
   = none / qkv / o, printed: `E2E eval gemma64x0-fcwh-<skip>-vs-q4m1`
   (the P4 hand numbers 9.5 / 28–32 dB reproduced by script); (ii) on
   gemma64 the tool's from-Q4_0 sidecar of `g64htp` run as `g64e3wq`,
   `E2E eval gemma64-fcwh-q4src-vs-f32src` and `-vs-cpu` printed (= (a));
   (iii) `E2E ppl-decode gemma64x0-fcwh` (WH forced on the Q4M1 token's
   ids, the lfm25 pattern at `:1250`), printed now, gated in step 2; (iv)
   lfm25: the e3fcwh ppl line with skip = qkv beside it (its +5.35 % is
   (b) + the Q4_0-vs-per-column weight difference, A′ of plan 229 §8.2 —
   not (a)). Gate rung 1: the lines print, every existing gate unchanged.
   Decision line in the PR: (b) per kind in dB and the forced-PPL delta
   at the fixture's gammas. If `gemma64x0` forced PPL is already inside
   2 % with skip = none, stop after step 1 and say so (the issue's "if (a)
   dominates, nothing changes").
2. **Kernel + spec (DSP).** §3 as written; `fc_wh_det.h` rewritten on
   `q8_0_quant_cpu_det`; `moe_layer_host_check.c` cells unchanged in
   shape; the second mutant becomes `w->w_scale + c0` → `w->w_scale` (the
   colsum one has no target left); `run_inproc_e2e.sh`: the 0.3 override
   deleted, the gate on the fixture's gammas, the ppl line gated at 2 %,
   the step-1 skip lines kept (they now read ≥ 40 [E] for every kind).
   Gate rung 1 (`FC WH BIT-IDENTICAL`, 2 × `FC WH MUTANT CAUGHT`, `GRAPH
   FC WH OK`, `E2E eval gemma64x-fcwh-vs-q4m1 ≥ 20`, `8/8`, ppl |delta| ≤
   2 %, `e3 pool C=2 gemma64-fcwh == e3 bit_identical=1`,
   `moe_dumps==sidecar-less bit_identical=1`, lfm25 tokens 8/8), then
   rung 2 for v79 and v81 (`UNDEFINED SYMBOLS OK`, `ARCH OK`, stub
   unchanged — `md5sum` of the stub before/after).
3. **App build, rung 3**, md5 table; then **the device read is
   unavoidable** and rides plan 229 S2.5 (the real option-A files; this
   lands before S2.3 so T16 inherits it). Handoff variants (≤ 4, prompt =
   the config's `sample_input` 447, G 64 / 512 / 1024, cool start per G,
   A first, `.sitting.lock`, S1 ceiling after every run):

   | variant | what | reads |
   |---|---|---|
   | **A16** | hybrid, unchanged reference | tok/s, prefill, PPL / text |
   | **F16** | one PD, 4-bit sidecar, this kernel | PPL ≤ 1.02 × A, tokens, rms(F − A), FC ms, tok/s |
   | F16-skip | `NNTR_HTP_FC_WH_SKIP=qkv` | the fallback's cost (expect +2.3 ms [E]) — G 512 only |
   | T16 | 2-bit sidecar, if S2.3 landed | plan 229 §8.3's cell |

   Files late: a dummy speed pre-read only (new vs P4 skel), no PPL column.

## 5. Risks

* **Compute surfaces.** If the +5 ops per tile are not hidden, FC rises
  above 11.2 ms; the FC column vs P4's 10.65 and `F16-skip` show it, the
  knob is the stopgap. No ISS in this tree prices it (rule 58).
* **DMA rate / DVFS / thermal.** A/F/F-skip in one sitting, cool start per
  G, rule 52's band; F after A is warm as in P4 (prefill not a gate here).
* **Stale skel** refuses nothing (no IDL change) and silently runs the old
  u8 path: the handoff lists the skel md5; `unittest_hvx_fc` on the device
  is the cross-check. **Address space / heap:** unchanged; ㉟ columns stay.
* **S2 interaction:** S2.3 extends `fc_m1_run`; order this before it so
  the 2-bit loop is written on the i8 variant once. The HMX prefill FC
  (LFM, M > 1) keeps per-row u8; the gate is DSP-vs-spec per path.
* **Fixture blindness:** gemma64x's softmax is peaked by construction; the
  real-file PPL column is the only verdict (rule 73).

## 6. Docs to update

* **BENCHMARK.md** Gemma block: under the #234 P4 side tables, the F16 /
  F16-skip cells of the S2.5 sitting with FC ms, rms(F − A), forced PPL;
  the host split table (per kind dB, (a) vs (b)) as a host side table.
* **LEDGER.md**: rule candidate "the M=1 WH FC's activation is the CPU's
  Q8_0 per-32 int8; per-row u8 cost 9.5 dB on q / k under a peaked
  softmax; the fixture's gammas are never a lever"; rule 73's last
  sentence → this plan; ㊸ untouched.
* **Plan 229 §8.4** "Cause 2 of P4 stays" → resolved here; S2.3's loop on
  the i8 variant. **Plan 234 §3** P4 ponytail line retires.
