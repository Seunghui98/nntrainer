## Dependency of the PR

#4236

Branches off `hvx_mm_op`. The softmax kernels are independent of the u8i4
matmul path, but they share `test/htp/build.sh`, `test/htp/nntr_hvx.idl` and
`test/jni/Android.mk`, so this stacks rather than racing #4243 for those
three files.

## Commits to be reviewed in this PR


<details><summary>c3844f7d [test] Add FastRPC entries for the HVX softmax bring-up</summary><br />

Open two IDL entries and wire the skel and the device test around them.
Both return zeroed output for now; the kernels land in the next commits.

exp gets its own entry rather than being checked through softmax alone.
softmax normalizes by the sum, so an error that moves every term the same
way cancels and never shows up in the output -- the polynomial's real
accuracy is only visible when exp is probed directly.

exp_f32 requires a length that is a multiple of 32. hvx_exp_sf is a
whole-vector operation and this entry is a test probe, so handling a tail
here would be code with no consumer.

The EBADPARM assertions add kDspOffset (0x80000400) before comparing:
AEEStdErr.h bakes that offset into every AEE_* code when __hexagon__ is
defined, so a DSP-side skel function returning AEE_EBADPARM crosses the
FastRPC boundary as 0x8000040e, not the host's plain 14.

**Self evaluation:**
1. Build test: [X]Passed [ ]Failed [ ]Skipped
2. Run test: [X]Passed [ ]Failed [ ]Skipped

Signed-off-by: dlwlzzero <dlwlzzero@gmail.com>
Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Signed-off-by: SeungHui Lee <shsh1004.lee@samsung.com>

</details>



<details><summary>7cee5de6 [htp] Add a vector exp for f32 lanes on HVX</summary><br />

exp(x) = 2^k * exp(r): k goes into the exponent field via a single bit
add, so the polynomial only has to cover r, which round-to-nearest
reduction narrows to [-ln2/2, ln2/2] -- half the floor-based range,
letting a 7th-order polynomial reach f32 epsilon.

Both the range reduction and the polynomial's Horner recurrence run in
qf32 rather than plain Vsf. On real HVX hardware (not reproducible in a
host-side float32 simulation), chained plain-Vsf multiply-adds lose
precision that qf32 retains; on-device testing on the S25 Ultra showed
this costing up to 1.68e-4 worst-case relative error before the fix,
against a 1e-6 target.

ln2 is split into ln2_hi + ln2_lo (the standard Cephes/fdlibm/SLEEF/glibc
technique) because a single f32 constant does not carry enough mantissa
bits once |k| is large -- the rounding error in ln2 gets multiplied by k,
and no amount of qf32 arithmetic downstream can recover precision that
was never in the input. ln2_hi's trailing zero mantissa bits make k*ln2_hi
exact in f32 over the whole k range this domain uses.

The low clamp at -120 is not cosmetic. hvx_sf_to_w_rne is only exact for
|v| <= 2^22, and softmax feeds x - max, which can be -1e30; without the
clamp k comes out as garbage and the underflow guard can be fooled into
letting it through.

No overflow clamp. softmax feeds x - max <= 0, so above the top of the
range is unreachable; clamping it would spend instructions per vector on
a path that cannot be taken. The header states the domain instead.

**Self evaluation:**
1. Build test: [X]Passed [ ]Failed [ ]Skipped
2. Run test: [X]Passed [ ]Failed [ ]Skipped

Signed-off-by: dlwlzzero <dlwlzzero@gmail.com>
Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Signed-off-by: SeungHui Lee <shsh1004.lee@samsung.com>

</details>



<details><summary>91f36bc4 [htp] Add a row-wise f32 softmax on HVX with tail handling</summary><br />

Three passes per row: max, then exp with the sum accumulated alongside
in qf32, then normalize. The exp results go straight into the output
buffer and pass three scales them in place, so there is no scratch
buffer -- no VTCM reservation, no per-thread scratch offsets. Reading a
vector before writing the same index also makes an in-place call safe.

Row lengths that are not a multiple of 32 go through the same vector
path via an aligned stack buffer (load_tail_sf / store_tail_sf). Reading
a vector straight off the row end would run up to 124 bytes past the
buffer; staging through the stack also keeps whatever follows the row
out of the max and the sum. The tail's pad lanes hold exp(0 - max), not
zero, so they are masked off after exp with Q6_Q_vsetq2_R before
reaching the sum.

1/sum is a scalar divide. sum ends up replicated across every lane, so a
vector reciprocal with a Newton iteration would be more code for a less
exact answer. The maximum is taken over the scaled values rather than as
scale*max(x); a negative scale flips which element is the maximum.

The shift-invariance test uses 1e2 rather than 1e4: a 1e4 shift rounds
f32 inputs to ULP ~0.001 on the host before the DSP sees them, and no
kernel can recover those lost bits. exp(100) still overflows f32, so the
max subtraction the test exists to check is still required.

**Self evaluation:**
1. Build test: [X]Passed [ ]Failed [ ]Skipped
2. Run test: [X]Passed [ ]Failed [ ]Skipped

Signed-off-by: dlwlzzero <dlwlzzero@gmail.com>
Co-Authored-By: GLM 5.2 <noreply@z.ai>
Signed-off-by: SeungHui Lee <shsh1004.lee@samsung.com>

</details>



<details><summary>a6eaf48f [test] Cover multi-row, in-place, negative scale and row ranges for softmax</summary><br />

Four properties of hvx_softmax_rows_f32 that the single-row tests could
not see: rows stay independent, an in-place call gives the same answer as
an out-of-place one, a negative scale still picks the right maximum, and
a row range leaves the rows outside it alone.

The row range is the one that needed new surface: softmax_f32 now takes
m_first, so the range API that a worker pool will split on is exercised
rather than only ever called as (0, M).

The LeavesRowsBeforeTheRangeAlone test compares the range call's output
rows against a full (0, M) call's, rather than checking the untouched
rows' values directly. FastRPC zeroes the rout buffer before sending it
to the DSP, so the untouched rows come back 0 regardless of whether the
kernel wrote them -- observing them would need an in-place (inrout) IDL
entry, which is not worth widening the test surface for.

**Self evaluation:**
1. Build test: [X]Passed [ ]Failed [ ]Skipped
2. Run test: [X]Passed [ ]Failed [ ]Skipped

Signed-off-by: dlwlzzero <dlwlzzero@gmail.com>
Co-Authored-By: GLM 5.2 <noreply@z.ai>
Signed-off-by: SeungHui Lee <shsh1004.lee@samsung.com>

</details>



<details><summary>c74d478a [test] Unify LANES and put the softmax IDL comments in English</summary><br />

LANES was spelled three ways across three files -- all of them 32, but
with signedness that differed enough to need casts at the use sites.

The softmax IDL entries were the only Korean comments in a file that is
otherwise English, and this one is headed upstream.

No behaviour change.

**Self evaluation:**
1. Build test: [X]Passed [ ]Failed [ ]Skipped
2. Run test: [X]Passed [ ]Failed [ ]Skipped

Signed-off-by: dlwlzzero <dlwlzzero@gmail.com>
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Signed-off-by: SeungHui Lee <shsh1004.lee@samsung.com>

</details>



<details><summary>3813d85f [chore] Fix doxygen tag on multi-line comments in HVX softmax/exp/add</summary><br />

The doxygen-cncpp checker rejects any multi-line /* */ comment that
doesn't start with /**, same rule 5cf9969b fixed for hvx_quant_u8.c.
The exp kernel and the LANES rationale comment above both introduced
multi-line /* */ comments that CI's doxygen tag check flagged. Comment
text is unchanged; no behaviour change.

**Self evaluation:**
1. Build test: [X]Passed [ ]Failed [ ]Skipped
2. Run test: [X]Passed [ ]Failed [ ]Skipped

Signed-off-by: dlwlzzero <dlwlzzero@gmail.com>
Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Signed-off-by: SeungHui Lee <shsh1004.lee@samsung.com>

</details>

### Summary

- **Add FastRPC entries for the HVX softmax bring-up**: two IDL entries
  (`exp_f32`, `softmax_f32`) plus the skel and device test around them,
  returning zeroed output first so the RED failure proves the call reached
  the DSP separately from the kernel being wrong
- **Add a vector exp for f32 lanes on HVX**: `exp(x) = 2^k * exp(r)` with
  round-to-nearest range reduction, a 7th-order polynomial, and both the
  reduction and the Horner recurrence in qf32 -- plain Vsf measured up to
  1.68e-4 worst-case relative error on device against a 1e-6 target
- **Add a row-wise f32 softmax on HVX with tail handling**: three passes per
  row (max, exp with the sum accumulated alongside, normalize) with no
  scratch buffer, so an in-place call is safe and no VTCM is reserved. Row
  lengths that are not a multiple of 32 go through the same vector path via
  an aligned stack buffer rather than reading up to 124 bytes past the row
- **Cover multi-row, in-place, negative scale and row ranges for softmax**:
  four properties the single-row tests could not see, including the
  `m_first` row-range API a worker pool will later split on
- **Unify LANES and put the softmax IDL comments in English** and **fix the
  doxygen tags**: no behaviour change, both required for CI

Split out of #4245, which carried three independent subjects in one branch.
The u8i4 DMA/registry/session work from that branch is #4243; the two
commits that make the softmax entries take the session handle #4243
introduces are held back as a small follow-up, so this PR and #4243 can be
reviewed in parallel.

## Verified on device (Galaxy S25 Ultra, V79, unsigned-PD CDSP session)

    unittest_hvx_softmax    15/15 PASSED

Signed-off-by: dlwlzzero <dlwlzzero@gmail.com>
Signed-off-by: SeungHui Lee <shsh1004.lee@samsung.com>
