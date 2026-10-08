# 137 — The device gtest set: `HvxM1Ops` 2/6, `HvxAttnM1` / `HvxAttnM1Probe` 2 red

Issue #137 (p2). Base `htp_decode` @ `2c740606c`. Every `path:line` below
was read on that commit. Inputs: the issue and its cycle-20 comment,
LEDGER rules 24 / 37 / 38 (`docs/htp_moe/LEDGER.md:928`, `:1133`, `:1154`),
open item ㉖ (`:1750`), the #208 logs
`/local/mnt/workspace/htp_moe/204/s26/logs/gt_unittest_hvx_softmax.log` and
`gt_unittest_hvx_attn.log` (v81, 2026-10-01), which repeat #130's v79 set.

## 0. What the planner found (decides the design)

**The set is smaller than rule 37 says.** On the #208 logs `rope64` is
`bad=0` at every position (fp16 RoPE since #152) and `ATTN_M1` is `bad=0
bad_stats=0` at every L (the #146 / #170 rewrite). What remains red:

| test | red field | first mismatch (device vs spec) |
|---|---|---|
| `HvxM1Ops.RmsnormMatchesDetBitExact` | kind 2 (the ±1e-39 row) `bad_y=2048 of 2048`; kinds 0/1/3, the 41-row sweep: 0 | `i=0 dsp=0x1.910abap-122 ref=0x1.910aacp-122` (+7 ulp) |
| `HvxM1Ops.QkNormMatchesDetBitExact` | `bad_y=64` = head 1 = the kind-2 head; the 80-head sweep: 0 | `i=64`, the same pair of values |
| `HvxM1Ops.ConvGateM1MatchesDetBitExact` | `bad_out=229 bad_state=0` | `i=2 dsp=-0x1.54bfap-128 ref=-0x1.54bfa8p-128` (1 subnormal quantum) |
| `HvxM1Ops.RejectsBadShapes`, `HvxAttnM1.RejectsBadShapes` | every `AEE_EINVALIDFORMAT` expectation reads `0x80000600` | `AEE_EBADSTATE` expectations in the same test pass |
| `HvxAttnM1Probe.Semantics` | `no fma case file at attn_fma_cases.bin` | not a kernel result |

**(2) The subnormal rows are a hardware property of `Q6_Vsf_vmpy_VsfVsf`,
not a kernel or spec fault, and the product never reaches it.** Replaying
the gtest's rows against `m1_ops_det.h` on the host
(`m1_rmsnorm_chunk_det`, `m1_conv_gate_det`; scratchpad program, not
committed):

* rmsnorm lane 0: x = 1e-39 (subnormal, 20 significant bits), r =
  `0x1.3c3a5p+8`, gamma = `0x1.dd0bbp-1`. The kernel is `(x * r) * gamma`
  (`hvx_m1_ops_f32.c:133`), the spec the same two roundings
  (`m1_ops_det.h:283-289`). IEEE gives the spec's `0x1.910aacp-122`. The
  device's `0x1.910abap-122` is reproduced exactly by rounding the first
  product to **20 bits = 24 − (leading zeros of the subnormal mantissa)**
  and then multiplying by gamma in IEEE; no other width (16 … 24, RN or
  truncation) and no other operation order (`x * (r * gamma)`, fma, f64)
  gives it. Over the whole row that model predicts `bad_y=2048`, as logged.
  Reading: the vector multiplier does not normalise a subnormal operand,
  so the product carries the operand's reduced precision.
* conv lane 2 (chain step t=2, the ±1e-39 `abc` row): `g = a*c` underflows
  to 0 on both sides, `y = fma(w2, s0, 0)` is normal, `out = b * y` with b
  subnormal lands **below 2^-126**. The device value is the IEEE 24-bit
  product truncated (not rounded) onto the subnormal grid; that model
  matches the lane but predicts 80 of 2048 lanes, the log says 229, so the
  exact rule for subnormal *results* is not pinned by one lane. It is not
  modelled further: it would need a device probe loop.
* The HVX add / sub path keeps subnormals exactly (rule 24 stands). What
  rule 37 called "the emulation's rounding of tiny values" is the
  multiplier with a subnormal operand or result; `qf32`, `vrsqrt`,
  `vrecip` are not involved (the row scale is `bad_row_scale=0` and the
  kernel uses none of them, `hvx_m1_ops_f32.h:15`).
* Product path: 802 816 RMSNORM / QK_NORM input elements dumped from the
  real model (`/local/mnt/workspace/htp_moe/norm/d_f_R/`, #164) hold
  **0 subnormals** (1300 exact zeros, min non-zero |x| 4.4e-9); 884 736
  CONV1D_GATE inputs and 294 912 outputs (`152/dump/D0`, `D1`) hold 0
  (min non-zero 1.0e-6). The subnormal rows exist only in the synthetic
  fixture. Under the bit-preserving rule (contract §12, 2026-09-28/29)
  the gate for a resident kind is the same-input shadow on real rows,
  which these kernels passed (392/392, 1920/1920 heads, conv 1152/1152);
  a bit at 2^-128 cannot move it. So the issue is "the suites must be
  honest", not "the kernel must match IEEE on subnormals": no DSP source
  changes.

**(1) `AEE_ERPC` is not the marshalling.** Rule 38's reading ("a sequence
length that disagrees with the shape is refused before the entry") is
refuted by the log itself: `attn_m1_register(2, kv, gqa, hd, 100)` carries
five `uint32` and no sequence (`nntr_hvx.idl:573-575`) and still reads
`0x80000600`; the rmsnorm case sends consistent lengths (x 2048 / gamma 48
/ y 2048 / rs 1, `unittest_hvx_softmax.cpp:384-389`). The pattern is
exact: every entry `return AEE_EINVALIDFORMAT` (`nntr_hvx_small_ops.c:50`,
`:68`, `nntr_hvx_attn_m1.c:95`, `:124`) arrives as `AEE_ERPC`
(`AEEStdErr.h:114`, "error due to fastrpc implementation"), every
`AEE_EBADSTATE` from the same entries arrives intact. The FastRPC runtime
rewrites a method's `AEE_EINVALIDFORMAT` (0x11) on the way back — SDK
6.4.0.1 ships no source for it (`ipc/fastrpc/` = `incs qaic remote rpcmem
rtld`), so which side cannot be cited; the DSP code is right and the
expectation in the tests is what is wrong. The probe already took this
route (`38f80800`, `unittest_hvx_attn.cpp:1401-1412`). A missing
validator stays visible: the kernels refuse a bad shape silently and the
entry would then return `AEE_SUCCESS`, which neither accepted code is.

**(3) `attn_fma_cases.bin`** is not built; `tools/htp/attn_fma_cases.py
<dump_attn dir> <out.bin>` derives it from the #136 attention dumps
(`/local/mnt/workspace/htp_moe/136/dump_attn`, 460 files, still there;
md5 `4ab75655…`, also at `170/s1/s170p/attn_fma_cases.bin`). The gtest
opens `./attn_fma_cases.bin` relative to its cwd unless
`NNTR_ATTN_FMA_CASES` is set (`unittest_hvx_attn.cpp:1243-1247`). #170's
runner passed the env (`170-s1-run.sh:90`); `204-s26-stage.sh:17-25`
copies binaries only and `204-s26-run.sh:118-120` runs `cd $D && ./$t`, so
staging the file into `app/` is enough — no env, no runner change.

## 1. Goal and gate

Acceptance (issue, made measurable):

| # | gate | reads as |
|---|---|---|
| G1 | host | `bash test/htp/host/run_host_checks.sh` → `ALL CHECKS PASS` (`m1_ops_host_check`, `attn_m1_host_check` BIT-IDENTICAL, unchanged — the emulation stays IEEE and nothing under `test/htp/` or `htp_backend/` changes); `bash tools/htp_syntax_check.sh` exit 0; `clang-format-14` on the two gtest files |
| G2 | device, one run | `unittest_hvx_softmax --gtest_filter='HvxM1Ops.*'` **6/6**, `unittest_hvx_attn --gtest_filter='HvxAttnM1*'` **all green** (`RejectsBadShapes`, `MatchesDetSpecBitExact`, `AppendChainEqualsBulk`, `PerLayerCost`, `Probe.Semantics`, `Probe.Cost`), no `0x8000040e`; the new fields print `bad_y=0 bad_y_subnormal=2048` (rmsnorm kind 2), `bad_y=0 bad_y_subnormal=64` (qk_norm 8 and 32 heads), `bad_out=0 bad_out_subnormal=229` (conv), each with its first-mismatch line — the device confirmation of rule 62 |
| G3 | ledger | rule 37 and 38 rewritten, rule 62 added, ㉖ closed; the issue's "device-confirmed rule naming the op and the exempt rows, a test that prints those rows as a field and gates the rest" branch |

Standing gates: no tok/s path changes (no kernel, IDL, app or quantizer
line), so prefill ≥ −5 % of A and text ≡ CPU are inherited from the row of
record, not re-measured; the plan states this in the handoff note instead
of running E2E variants (the handoff skill's "always a full-model E2E"
applies to measurements, and this sitting measures nothing).

## 2. Where it lives

| file | lines | change |
|---|---|---|
| `test/unittest/unittest_hvx_softmax.cpp` | `m1_count_bad` `:336-351`; `m1_rmsnorm_case` `:354-374`; `RejectsBadShapes` `:383-410` (expectations `:389`, `:395`, `:405`); rmsnorm `:420-461`; qk_norm `:466-508`; conv `:550-580` (`:569`) | split the mismatch count into gated / subnormal-exempt; accept `AEE_ERPC` |
| `test/unittest/unittest_hvx_attn.cpp` | `RejectsBadShapes` `:903-950` (expectations `:913`, `:918`, `:931`, `:942`); probe precedent `:1401-1412` | accept `AEE_ERPC` through one helper shared with the probe's n=63 check |
| `test/htp/host/hvx_emu/hvx_hexagon_protos.h` | premise comment `:13-16`, `:88` | one paragraph: the premise holds for normal operands and results; `vmpy` with a subnormal operand or result is not IEEE on v79 / v81 (rule 62); the emulation stays IEEE on purpose and the gtests exempt those lanes |
| `docs/measurements/204-s26-stage.sh` | after `:25` | one line: `python3 "$R"/tools/htp/attn_fma_cases.py /local/mnt/workspace/htp_moe/136/dump_attn "$A"/attn_fma_cases.bin` (the stage script is what the next sitting copies) |
| `docs/measurements/137-gtest-run.sh` (new, ≈ 25 lines) | — | stage + run of the two gtest binaries only (§4 step 4) |
| `docs/htp_moe/LEDGER.md` | `:1133` (37), `:1154` (38), `:1750` (㉖), after `:1638` (62) | §6 |
| `docs/htp_moe/BENCHMARK.md` | Log, after the #208 entry | §6 |

Not touched, by design: `test/htp/nntr_hvx.idl` and the stub, `HtpComputeOps`,
`nntr_quantize_stream`, the loader, `NNTR_HTP_PROFILE` tables,
`tools/htp_fc_report.py`, `m1_ops_det.h`, `hvx_m1_ops_f32.c`,
`nntr_hvx_small_ops.c`, `nntr_hvx_attn_m1.c`, the host checks' `AEE_EINVALIDFORMAT`
assertions (`attn_m1_host_check.c:1191-1225`, `graph_host_check.c:437`) —
the entries' contract stays `AEE_EINVALIDFORMAT`; only the device-side
reading of it changes.

## 3. Design

**Subnormal rows — print and exempt, gate the rest (chosen).**
`m1_count_bad` gets the input lane and a second counter:

```cpp
/* A mismatch whose input lane or spec output is subnormal (non-zero,
   |v| < FLT_MIN) is counted in *bad_sub, not in the return value: the HVX
   multiplier is not IEEE there (LEDGER rule 62) and no real row of the
   model has one (plan 137 §0). Both classes print their first lane. */
int m1_count_bad(const std::vector<float> &dsp, const std::vector<float> &ref,
                 const char *what, const float *in /* lane-aligned input */,
                 int *bad_sub);
```

rmsnorm / qk_norm pass `x` (lane-aligned with `y`), conv passes `abc + C`
(the `b` lane whose product is the subnormal output; the spec output test
covers the rest). Fields become `bad_y=… bad_y_subnormal=…` and
`bad_out=… bad_out_subnormal=…`; `EXPECT_EQ` gates the first only; the
`row_scale`, `state` and sweep fields are unchanged (they were 0). The
kind-2 row stays in the fixture: it still proves rule 24 (nothing is
flushed — the device values are within 7 ulp of the unflushed result, not
0) and keeps printing the evidence lane every sitting. The `overflow_row`
stays ungated as today.

*Rejected:* teaching `hvx_emu` the device's subnormal multiply so the host
reproduces `0x1.910abap-122`. One lane pins the subnormal-operand case to
"round at 24 − lz bits", but the subnormal-result case (conv, 229 vs the
model's 80 lanes) is not pinned, so an emulation written now would be a
guess checked only by the next sitting; and it would model a path no
product row takes. Also rejected: dropping the row (loses the rule-24
reading) and loosening the spec (the spec is the CPU's bits; the CPU keeps
subnormals in IEEE).

**`RejectsBadShapes` — accept both refusal codes, keep the contract
(chosen).** One helper per file, mirroring the probe:

```cpp
/* The entry returns AEE_EINVALIDFORMAT; the FastRPC runtime hands the
   host AEE_ERPC (0x80000600) for it on v79 and v81 — a scalar-only call
   (attn_m1_register) reads the same, so it is not marshalling (LEDGER
   rule 38, rewritten by #137). AEE_EBADPARM stays the stale-skel code. */
bool m1_refused(int err) {
  return err == AEE_EINVALIDFORMAT + kDspOffset || err == AEE_ERPC + kDspOffset;
}
```

(`AEE_ERPC` is `0x200` in the Android compile, `AEEStdErr.h:38-41`.) The
`AEE_EBADSTATE` and `AEE_SUCCESS` expectations stay exact. *Rejected:*
changing the entries to a code that survives the runtime (e.g.
`AEE_EBADITEM`): it rewrites the IDL comments, two host checks and the
graph error mapping, needs a skel rebuild and a stub regeneration for a
test's benefit, and still documents nothing the helper's comment does not.

**Fixture — stage it (chosen).** One line in the stage script; the
generator is deterministic (`4ab75655…` across #170 and this plan's
rebuild — the implementer checks the md5). *Rejected:* committing the
29 KiB binary (the repo keeps generated fixtures gitignored, gates skill
rung 1) or bundling the generator into the gtest (it needs the dumps).

Contract §2 and doc 45 §3 are untouched: no kernel, arena, DMA or quantizer
line changes.

## 4. Steps

1. **Tests.** `unittest_hvx_softmax.cpp`: the `m1_count_bad` split, the
   three call sites, the field prints, `m1_refused` at `:389/:395/:405`.
   `unittest_hvx_attn.cpp`: `m1_refused` (same name) at `:913/:918/:931/
   :942` and in the probe's n=63 check (`:1411`, replacing the literal).
   Gate: rung 0 (`clang-format-14` on the two files) and rung 1
   (`ninja -C build`; `bash test/htp/host/run_host_checks.sh` →
   `ALL CHECKS PASS`; `bash tools/htp_syntax_check.sh`). The gtests
   themselves do not run on the host; the host gate here is "nothing else
   moved".
2. **Emulation comment + stage line.** `hvx_hexagon_protos.h:13-16`
   paragraph; `204-s26-stage.sh` line; the new `137-gtest-run.sh`
   (serial required, `adb -s`, pushes `unittest_hvx_softmax`,
   `unittest_hvx_attn`, `attn_fma_cases.bin`, `libc++_shared.so`, the skel;
   md5 both ends; runs the two filters; saves `logs/gt_*.log`; greps
   `0x8000040e` → stop). Gate: rung 1 again (`run_host_checks.sh`), and
   `python3 tools/htp/attn_fma_cases.py … ` printing `written=4878`, md5
   `4ab75655226064cc3dcba73639f07d7f`.
3. **Binaries.** Rung 3 for the two gtests only (`test/jni` ndk-build of
   `unittest_hvx_softmax unittest_hvx_attn`). Skel: if `git log -1 --
   test/htp/nntr_hvx.idl` is still `0f63576c5` (2026-09-30), the staged
   `204/s26/app/libnntr_hvx_skel.so` (built 2026-10-01, v81) pairs with
   the new stub and is reused — the cleanest A/B, same skel as the red
   logs; otherwise rung 2 (`HEX_ARCH=v81 ./test/htp/build.sh`,
   `UNDEFINED SYMBOLS OK`, `ARCH OK (V81)`). Record md5s.
4. **Device (unavoidable; the only device step).** One sitting, gtests
   only, on the S26 developer unit (the #208 unit, `adb -s` its serial;
   the S25 `R3CY10WM83Y` with a v79 skel is the fallback and is also a
   valid read — the set is identical on both). Not a `RegistryCapacity`
   run (rule 60). Variants: none — A/B is the #208 log vs this run on the
   same skel; no E2E, no tok/s cell (§1). Gate: G2. The filled note
   (`docs/measurements/137-gtest-set.md`, short: md5 table, the two
   `[  PASSED  ]` lines, the three `*_subnormal` fields with their
   first-mismatch lanes) is the record.
5. **Docs** (§6), PR into `htp_decode`, `[test]` subject, DCO.

## 5. Risks

* **Stale skel / stub pair.** The gtests' stub is regenerated by the
  ndk-build from the IDL at head; a skel from an older IDL answers
  `0x8000040e`. Step 3's IDL-commit check and the runner's grep make it
  visible; the fix is rung 2.
* **The device disagrees with the model** (e.g. a `*_subnormal` count
  other than 2048 / 64 / 229, or a gated lane red). The fields print
  either way; a red gated lane means a normal-range mismatch, which is a
  real regression and the reason this suite is kept (㉖) — the PR does
  not merge until read.
* **Address-space / DVFS / thermal** do not apply: no E2E, no timing gate
  (`PerLayerCost` prints, it does not gate).
* **Honesty gap.** The emulation stays IEEE on lanes the device is not;
  rule 62 and the emulation's comment say so, and the gtest fields keep
  the device's values in every log. Anyone wiring an op whose real rows
  can be subnormal must read rule 62 first.

## 6. Docs to update

* `LEDGER.md` rule 37 (`:1133`): strike the candidates list (`qf32`,
  `vrsqrt`, `vrecip`, input flush) and the rope / ATTN_M1 items (green
  since #152 / #146); point to rule 62. Rule 38 (`:1154`): strike the
  marshalling reading, state the runtime's `AEE_EINVALIDFORMAT → AEE_ERPC`
  rewrite with the scalar-only `register` call as the proof and
  `m1_refused` as the reading. New rule 62: "`Q6_Vsf_vmpy_VsfVsf` keeps
  subnormals but is not IEEE on them: a subnormal operand's product is
  rounded at 24 − lz bits (one lane matched, 2048/2048 predicted), a
  subnormal result is below IEEE RNE (1 quantum, rule not pinned); add /
  sub exact (rule 24). No real RMSNORM / QK_NORM / CONV1D_GATE row has
  one (counts in plan 137 §0). Gtests print those lanes as
  `*_subnormal` and gate the rest." ㉖ (`:1750`): closed with the G2
  line and the device/skel md5.
* `BENCHMARK.md` Log: one entry for the gtest sitting (unit, skel md5,
  6/6 and the attn line, the three fields); no Results row (no tok/s).
* `docs/measurements/204-s26-rebaseline.md:157-158`: a one-line "resolved
  by #137" pointer (optional, supervisor's call).
* Guide `01-run-it.html:238/240` rows say "0/5" / "2/4" — supervisor's
  refresh, not this PR.
