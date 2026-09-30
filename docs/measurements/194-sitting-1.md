# Measurement 194 sitting 1: E0 / E1 on silicon, the L0 split, the first PPL gate

Sitting 1: branch `htp/194-s1` @ `9b6f575d`, staged at
`/local/mnt/workspace/htp_moe/194/s1/` (read below). **Sitting 1b: run
2026-09-30 14:38–17:12 KST on `R3CY10WM83Y` from `htp/194-s3` @ `b0248a63`
(stage `/local/mnt/workspace/htp_moe/194/s1b/`), read in section "Sitting 1b
as read"; P1–P3 pass, P4 FAIL (p04), four user decisions listed there.** Track `htp_moe_ppl` (plan
`docs/plans/194-htp-moe-ppl.md`): no bit-preserving rule; the accuracy gate
is plan 194 section 1 (P1–P4), read against this sitting's A.

## Sitting 1b as read (2026-09-30 14:38–17:12 KST, R3CY10WM83Y)

Run by the orchestrator at the user's request, from the stage of `htp/194-s3`
@ `b0248a63`. The implementer read it from `/local/mnt/workspace/htp_moe/194/s1b/logs/`:
`sitting.out`, whose summary starts at its last `--- nulls` line;
`ppl.tsv`, `loops.txt`, `mc_A.txt` / `mc_E1.txt` and the per-run logs.
All 17 blocks ran. The last invocation printed `expectation mismatches (this
invocation): 0`, and no invocation printed a `BAD` line. The run hit nine
`STOP: LEAK` stops, each followed by a reboot and a resume (Notes). Numbers
are in Results below.

**Verdict as the gate reads (E1 = L1, mask 0x2): P1 pass, P2 pass, P3
pass, P4 FAIL** (1 new loop, on p04). Plan §4's stop rule says that when E1
fails P1–P4, L1's numerics go back to S1. This handoff does not soften the
fail. The reading below says why the flagged loop is a loop that A has too,
and the user decides whether that counts (see "User decisions").

* **Nulls hold.** `null A forced == A self` (17 digits); E0 nll ≡ A 8/8;
  E0 mc ≡ A q01–q08 8/8; E0 text ≡ A r1 at G 512 and 1024, both runs.
  Canary: 3 PASSED, `FC_Q4_FIELD total bad=0`, `FC_Q4_NATIVE total bad=0`
  (15 shape/feed cells). So E0 is A's bits, and the native FC equals its
  spec on silicon.
* **P1**: 2048 scored steps; pooled PPL A 1.210766, E1 1.213821; ratio
  **1.0025** (+0.0025 nats a token; the band is 1.02). The plan expected
  about +0.2 %, and the reading is +0.25 %.
* **P2**: every prompt is ≤ 1.05. The highest are p07 at 1.0110 and p04 at
  1.0096, the lowest p06 at 0.9973 (Results). The runner's per-prompt lines went
  to a file named `/local/mnt/workspace/htp_moe/194/s1b/ P2 FAIL` instead of the screen because of an awk
  `printf …, r > 1.05 ? …` that parses as an output redirect. The file name
  does not signal a fail: the P2 decision (`if (r > 1.05) p2++`) is
  separate and correct, and the file lists all eight ratios.
* **P3**: A 12/40 (Σnll 143.54), E1 12/40 (Σnll 140.99); `right=12/12`
  means E1's count over A's; nll ratio **0.9822**, so it passes. E1 and A pick
  the same letter on 36 of 40 questions. They differ on q05 (E1 right), q20 (E1
  wrong), q32 and q37 (both wrong). The paired Δnll per question is −0.064 ±
  0.254 (sd).
* **P4**: `LOOP lc_A_p04 L1run=1 L2=0.44 loop=0`, `LOOP lc_E1_p04
  L1run=1 L2=0.74 loop=1`, so **FAIL (1 new loop)**. A loops on p02, p03, p05,
  p06 and p08, and E1 loops on exactly those five plus p04. The line's
  `( unreadable logs)` has an empty count because awk's `e` is never set
  when no `error=` line exists: it means 0 unreadable logs.

**The p04 loop, read from the texts** (`lc_A_p04.log`, `lc_E1_p04.log`;
p04 is the Korean prompt, and G = 256 tokens gives only 100 / 110
whitespace words):

* A (L2 0.44): A restates the prompt's example, then prints its `**질문**`
  block (three numbered questions and the instruction line) **twice
  verbatim**. It then degenerates into `(各问答, 2文, 2文, …` with **`2文,`
  repeated 16 times** until the budget ends. That is a single-token
  repetition loop plus a repeated block, which is a loop by eye.
* E1 (L2 0.74): E1 prints its `질문:` block (the three questions, the
  instruction and the example sentence) **2.4 times verbatim**, separated by
  `</think>`. That is a real block repetition.
* `loop_check.py` flags `L1 ≥ 3 or L2 ≥ 0.5`. L1 cannot see either text
  because the repeated sentences are numbered questions, which are not
  consecutive identical sentences. L2 reads about 44 12-grams in the second
  half of a 100-word text, so each gram moves it by about 0.02. A misses
  the threshold by 0.06, i.e. three grams. The detector was calibrated on
  English (plan 152, prompt512 at 200–431 words).
* **Reading**: E1's p04 is a real repetition loop, but it is on a prompt
  where A also loops; the detector misses A's loop by three 12-grams. So
  "new loop" here is an artefact of the threshold on a short Korean text,
  not new behaviour. p04's PPL ratio (1.0096, the second highest) is
  within P2. The gate still reads FAIL.

**mc-40's power.** A scores 12/40 where chance is 10/40:
P(X ≥ 12 | n = 40, p = ¼) = 0.29, so A's score does not differ from
chance. A picks C 18 times and B once against a key of 10 each, which is a
letter prior, not knowledge. P3's `right` rule (E1 ≥ A − 2) therefore
tests two near-chance scores against each other and has almost no power
to detect a routing break smaller than "every answer changes". The useful
signals are the pick agreement (36/40) and the Σnll ratio (0.98). These
are what P1 already sees, on 40 single steps instead of 2048.

**Speed** (prompt 512; decode tok/s r1 / r2; ms a token from their mean):

| G | A | E0 | E1 | E1 − E0 |
|---|---|---|---|---|
| 512 | 56.21 / 54.89 | 32.19 / 30.94 (31.70 ms) | 33.83 / 33.10 (29.89 ms) | −1.81 ms |
| 1024 | 52.89 / 51.93 | 32.25 / 31.19 (31.54 ms) | 33.80 / 32.91 (29.99 ms) | −1.55 ms |

E1 is about 5 % faster than E0 and about 40 % below A; the goal of 50 is not
approached by E1 alone. E1's text differs from A at the same word as in sitting 1
(word 466 of the log: "along it stand" → "along it stands"), at both G.
The per-session line (G = 512 r1): the S2 FC + DENSE_FFN + LM_HEAD sum is
11.220 → 9.554 ms (FC 5.937 → 4.910, DENSE_FFN 2.154 → 1.486, LM_HEAD
3.129 → 3.157), and the MoE round is 0.456 / 0.471 ms.

**The L0 split, now with the wake split** (E0, `NNTR_OP_TIME=1`): wake =
rt − s2_wall = 2.88 / 3.37 / 3.20 ms at G 64 / 512 / 1024. Of that, **ret
s2 = 2.78 / 3.30 / 3.13 ms**, while the dispatch side is s1 0.06 and s2
0.06–0.09 ms, and `clk_resid` is −0.1 … −0.2 µs, so cntvct_el0 and the
QTimer agree. **The wake is almost entirely the return: from S2's answer
to the ARM's blocking read waking up.** arm_us is 1.51 / 1.55 / 1.59 ms (the
walk). In the `[OP-TIME]` per-kind table, lm_head is 30.5–30.7 ms,
88–95 % of the token. That is **not a kernel cost**: on the E path the
ARM's walk blocks inside the lm_head node's hook for the whole DSP token
(the round trip), so the row is the wait. The ARM's own rows are the
others (sampling 0.002, register 0.10, unattributed 0.54–0.59).

**e2 (informational; G = 64, one block, E1 → E2 = 0x1F → E2s = E2 with a
200 µs hop bound).** E2's own P1–P4 belong to sitting 2 and are **not
measured** here.

| | E1 | E2 | E2s |
|---|---|---|---|
| decode tok/s (ms a token) | 34.26 (29.19) | **38.95 (25.67)** | 38.95 (25.67) |
| rt / s2_wall / s1_wall µs | 25451 / 22595 / 19661 | 22652 / 22579 / 19389 | 22789 / 22716 / 19426 |
| wake µs (ret s2) | 2856 (2761) | **72.5 (29.1)** | 72.3 (22.3) |
| arm_us | 1559 | **420** | 396 |
| hop_us s1 / s2 | 363.6 / 323.6 | 157.6 / 218.9 | 57.2 / 103.1 |
| S1 router / MoE ms | 0.795 / 9.926 | 0.554 / 10.476 | 0.551 / 10.825 |
| S2 kernel sum ms | 10.738 | 10.614 | 10.543 |
| L0 spins | — | arm_hits 63/64, early_posts 63, ARM spin 982 µs/token | 63/64, 63, 980 |
| text | ref | DIFF from its first token | ≡ E2 |

* **L0**: it removes the wake (2.86 → 0.07 ms: the ARM's bounded spin
  catches S2's answer 63 times of 64) and 1.14 ms of arm_us (the walk now
  runs beside the token). That is **−3.9 ms** in total, inside plan §3.1's
  range (−3 … −7) but under its −5 mid. The cost is one ARM core spinning
  about 1 ms a token, plus S2's queue thread spinning ≤ 3 ms after each
  answer, when nothing computes on the DSP.
* **L4**: hop_us falls from 364 / 324 to 158 / 219 at bound 50, and to 57 / 103
  at bound 200. But s2_wall does not fall (22.60 → 22.58 → 22.72), and the
  MoE rises 9.93 → 10.48 → 10.83 ms (round 0.451 → 0.476 → 0.492) as the
  spin grows. That is consistent with rule 56 (the spin takes DSP time from
  S1's pool). Monotonic drift over three runs in a row cannot be excluded
  from one block. At bound 200 there is no net gain (tok/s identical).
* **L2 + L3**: −0.66 ms of kernel time (router −0.24, CONV1D_GATE −0.25,
  DENSE_FFN −0.10, RMSNORM −0.06, QK −0.01) against plan −1.35.
* **What is left of E2's 22.65 ms round trip**: S1 kernels 11.03 (MoE 10.48 +
  router 0.55), S2 kernels 10.61 (FC 4.84, LM_HEAD 3.10, DENSE_FFN 1.29,
  ATTN_M1 0.59, RMSNORM 0.31, CONV 0.24, ADD 0.12, QK 0.09, ROPE 0.03),
  in-DSP non-kernel 0.94 (hops 0.38 + 0.56 per-op / per-stretch
  overhead), and wake 0.07. So s2_wall 22.58 ≈ 21.64 of kernels + 0.94.
  s1_wall (19.39) overlaps S2's stretches and is not additive. Outside the
  round trip: arm_us 0.42, plus the first decode token's one-time of about
  2.6 ms a token at G = 64 (25.67 − 22.65 − 0.42, ≈ 165 ms a run; not
  moved in S3).
* **E2's text** leaves E1's at its first generated token (`first token
  id=2023` vs 4386) and continues as meta-commentary ("final answer should
  be a single sentence … The user may have wanted a more detailed
  description …", repeated). It is not gated here. It is the first thing
  sitting 2's P1 / P4 must read (L2's routing or L3's norms; plan §5's D1
  risk).

**Plan §3.4 re-stated against these readings** (ms a token; G ≥ 512
steady state, where the one-time is ≤ 0.3):

| stage | plan §3.4 (G = 512 / 1024) | reading or projection | tok/s |
|---|---|---|---|
| E1 (L1) | E5f 32.4 − 3.8 = 28.6 | **29.9 / 30.0 measured** | 33.5 / 33.4 |
| + L0 + L2 + L3 + L4 = E2 | 21.45 / 21.55 | ≈ 23.5 (E2's G = 64 parts: rt 22.65 + arm 0.42 = 23.07, + attention growth ≈ 0.2 (E1 ATTN_M1 0.57 → 0.76 at G = 512) + one-time ≈ 0.3 / 0.15; **not measured at G ≥ 512**) | ≈ 42.5 |
| + L5 (MoE round → 0.43, −1.0) | 20.45 / 20.55 | ≈ 22.5 | ≈ 44.5 |
| + L6 (−1.7) | 18.75 / 18.85 | ≈ 20.8 | **≈ 48** |
| + L7 (−0.2 / −0.4) | 18.55 / 18.45 | ≈ 20.5 | ≈ 49 |
| G = 64 | — | + ≈ 2.6 until the first token's one-time moves to the end of the prefill | ≈ 44 with it, ≈ 49.5 without |

So **E2 + L5 + L6 reaches about 48–49 tok/s at G ≥ 512 on this reading,
not ≥ 50**. It falls about 1.5–2 ms a token short of plan §3.4, from four
sources: L0 −3.9 instead of −5, L2 + L3 −0.66 instead of −1.35, L4 −0.31
instead of −0.8, and the MoE's +0.55 under the spins, which the plan does
not have. What could close the gap is the MoE's +0.55 if it is the spin,
the 0.56 of in-DSP per-op overhead, the remaining hops (0.38) and arm_us
(0.42). Plan §3.4's ceiling (18.5 ms, 54 tok/s, without overlap) stands.

**FC lane ladder** (E1, G = 64): 6,3 30.32 tok/s (S2 sum 10.04 ms), 6,6
30.32 (10.76), 8,8 31.71 (10.07). The lanes block ran after speed_G1024 in
the same boot. Its 6,3 run is the same configuration as e2's E1 (34.26), yet
its MoE is 11.13 vs 9.93 ms and its LM_HEAD 3.43 vs 2.96. So the
cross-block drift on the DSP side is about 13 %, larger than the ladder's
differences. **The ladder does not separate 6,3 from 8,8.** 6,6 is slower
on FC (5.47 vs 5.00 ms), as in #178.

**Standing checks.** `calls/token=1.00`, `hops/token=44.00`,
`timeouts=0/0 stale=0/0 id_mismatch=0` and `unmap_fail=0 detach_fail=0` held on
every E run (the runner's OK lines). An A sanity followed every E run. Prefill
vs A (means of r1 / r2): G = 512 E0 −0.9 %, E1 +3.3 %; G = 1024 E0
**−8.5 %** (451.1 / 477.6 vs A 530.0 / 484.8), E1 −1.6 %. The prefill code
is the same for all three variants, and A's own r1 / r2 differ by 9 %, so
E0's G = 1024 cell reads as run spread. It still misses the −5 % row on its
mean. **Ceiling: 3840 after every run except nine** (the LEAK stops,
Notes). The teardown fix (`42e332f4`, S1's arena released and unmapped
before the close) **did not remove the per-boot mapping loss**.

**PR into `htp_moe_ppl`: not opened.** Nothing becomes default: the levers
are off unless set, and `NNTR_HTP_E2E` is opt-in. But the branch changes
A's load and close path (the arena retry with a fresh fd, the MoE warm-up
after mapping, S1's arena unmapped before the close). The standing "no
leak" row fails in this sitting after A runs as well as after E runs:
four of the nine stops followed an A run. No boot ran A alone, so whether
the default path loses the 256 MiB without any E run is not measured.
The gate for what the PR makes default is therefore not shown to hold.
See "User decisions".

### User decisions (2026-09-30, from sitting 1b)

1. **P4 on p04.** Either (a) hold the gate as written: E1 fails, so L1's
   numerics go back to S1 (the per-block f32 sum fallback) before anything
   else is built. Or (b) accept the reading: A's p04 is a loop the detector
   misses by 0.06, so the flag is not a new loop and E1 / L1 is carried. If
   (b), also say whether `loop_check.py` changes for this track (for example, a
   token-run rule that catches `2文, 2文, …`, or a length-scaled L2 for
   short texts), so sitting 2 reads P4 by a stated rule.
2. **The mapping loss.** The teardown fix did not remove it (nine stops;
   11–63 app runs a boot, after A runs and after E runs). Either (a) accept
   reboot-and-resume (`loop_s1b.sh` plus the keep-a-finished-mc-run runner) as
   the method for sitting 2, or (b) make it a blocker on #132 before
   sitting 2. That includes one A-only boot (A runs with the ceiling after
   each) on this build against `htp_moe`'s, to learn whether the default
   path loses mapping room without E. That boot is about 25 min of device time.
3. **The PR** (S0 + S1 + S3 into `htp_moe_ppl`, #132 Part B and #178
   included; its base `htp/ppl-plan` is PR #196, still open). Either open
   it now, accepting the leak as a known opt-in E-path defect, or wait for
   decision 2's A-only reading.
4. **§3.4.** On this reading E2 + L5 + L6 lands at about 48–49 at G ≥ 512,
   not ≥ 50. Decide whether to build L5 / L6 as planned, or first re-plan
   the remaining 1.5–2 ms: the MoE under the spins, the in-DSP per-op
   overhead, and the first token's one-time for G = 64.

## State (resumed 2026-09-30, after the pause)

**Resumed** by the user the same day (the pause for FSU, below, did not
change the two-session layout: no FSU change has landed). Branches, all
linear on `origin/htp/ppl-plan` @ `40762adf` (= `htp_moe_ppl` @ `c34fc445`
+ the plan):

* `htp/194-s1` @ `ec860006`: S0 + S1 rebased onto the teardown fix of
  `htp/132-partb-e3` (its six commits after `69251151`, cherry-picked in
  order after `ce37cccd`: the arena retry with a fresh fd, the MoE warm-up
  after mapping, the stand-in cap, e5g's handoff, **`8bb8a3f8` the S1
  arena released on the DSP and unmapped before the close** and its test
  `ad8496e2`). Old → new: `33b7ce36` → `ec860006` (one conflict in
  `run_inproc_e2e.sh`, both sides kept). Rung 1 green on it.
* `htp/194-s3` @ `b0248a63` (supersedes `htp/194-s3-wip` @ `d8599531`,
  left on origin untouched): plan S3 on the host, every lever a bit of
  `NNTR_HTP_PPL_LEVERS`, all off unless set:

  | bit | lever | what | host gate (rung 1) |
  |---|---|---|---|
  | 0x2 | L1 | native FC (sitting 1) | as sitting 1 |
  | 0x4 | L2 | router on 32 f32 lanes, four partial sums, swiglu_det's exp / reciprocal as the sigmoid | kernel ≡ `m1_ops_vec_det.h` (3 shapes × 24 rows); spec vs CPU order: logits ≥ 117 dB, weights ≥ 120 dB; expert set identical on random and exact-tie rows, 0 flips in 4000 LFM2.5 rows; planted 1-ulp near ties flip 1 / 0 / 2 of 8 (printed: that is what P1 / P2 read) |
  | 0x8 | L3 | RMSNORM / QK_NORM sum of squares on 32 lanes + rotate tree, conv gate unfused, DENSE_FFN SwiGLU by `swiglu_det_one` | kernel ≡ spec on 64 rows each; spec vs CPU order 133–146 dB; graph ops ≡ spec with the bit, 5 graph mutants caught |
  | 0x10 | L4 | the hops' deadline wait: sleep to the last token's wait of that round − 40 µs, spin ≤ `NNTR_HTP_E2E_HOP_SPIN_US` (50) | 10 000 tokens bit-identical at bounds 0 / 50 / 200 µs, lead 0, timeouts 0, stale 0; 0x1E ≡ 0xE in-process |
  | 0x1 | L0 | op 0's hook posts a steady token so the layer walk runs beside it (the walk is overlapped, not skipped: plan wording deviates); S2's queue spins `NNTR_HTP_E2E_S2Q_SPIN_US` (3000) after each answer; the ARM polls S2's answer ± `NNTR_HTP_E2E_ARM_SPIN_US` (1000) around the last token's time | 0x1F ≡ 0x1E in-process, `early_posts` 6 of 7, `arm_hits` 4/7, timeouts 0 |

  In-process PPL / SNR with L1–L3 (0xE): hd64 −0.009 % vs E0, lfm25
  32.5 dB vs E0 (gate ≥ 30), 34.1 dB vs L1 alone (the bits are wired).
  **Host-only**: no device number exists for L2–L4 or L0's fixes; the
  vector kernels have no device gtest (their silicon reading is the
  sitting's PPL, plan P1 / P2); the host's µs are not device µs.
* Code review (`/code-review high`, `ec860006..`): one finding, fixed in
  the L0 commit — the ARM's window slept once to its deadline and then
  recorded the read time, so after one slow token the estimate fell by
  only a window a token; it now sleeps in 500 µs slices with a poll
  between. Nothing else found in the state machine, L4 or the kernels.
* Not done in S3: the first decode token's one-time ≈ 140 ms (moving the
  KV seed / E2E start to the end of the prefill); a DSP-side spin of S1's
  queue (not needed: S1 waits for S2's first stretch anyway).

## State at pause (2026-09-30, superseded by the section above)

**htp_moe_ppl is paused** (user decision 2026-09-30: FSU will shrink the
MoE's DRAM residency, which changes the two-session premise). No sitting
runs, no PR. Branches: `htp/194-s1` (this file; everything below is on it)
and `htp/194-s3-wip` @ `d8599531` (L2 / L3 kernels, compiles, unchecked).

**Measured on silicon** (sitting 1, partial, G = 64 unless stated; details
in the next section): canary 3 PASSED incl. `FC_Q4_NATIVE total bad=0`
(the native FC equals its spec on the phone); decode tok/s A 57.25 / 53.69,
E0 30.35 / 28.91, E1 31.73 / 31.70 (E1 ≈ +6 % over E0, −1.5 ms a token
against L1's projected −3.8: FC −0.95, DENSE_FFN −0.65, LM_HEAD 0; all three
now feed-bound at 32–45 GB/s); E0 30.83 at G = 512; E0 text = A, E1 text
one word apart at G = 64. **L0 split** of E0's 32.95 ms token: kernels
24.1 (S2 13.17 + S1 10.93), in-DSP non-kernel 1.15 (hops 0.64), wake
(ARM ↔ S2, outside both DSP walls) 3.0–3.8, ARM between tokens 1.67 (≈ 1.5
of it the layer walk), a one-time first decode token of ≈ 140 ms (2.2 ms a
token at G = 64). Not measured: any PPL (P1 / P2), mc-40 (P3), loops (P4),
G = 1024, the lane ladder, the wake's dispatch / return split.

**Host-verified only** (never on the phone): the native pair's SNR
against the CPU order (149.4 dB natural rows) and its 11.5-packet static
count; the in-process PPL delta (hd64 −0.009 %) and lfm25 32.7 dB; the
wake-split line (`1ba5ae06`: closes to 0.0 µs in-process; whether
cntvct_el0 and the DSP QTimer agree on the phone is its first reading);
`NNTR_PPL_DECODE_ALTS`, `mc_ids.py` / `mc_score.py` (self-test) and the
mc-40 ids; sitting 1b's resumable runner (dry-run with a fake adb only).
On `htp/194-s3-wip`: the L2 / L3 kernels and their spec compile and leave
the existing host checks green, but nothing holds them to the spec yet;
L4 and the L0 fixes (bounded spins, one-call step) are not started.

**If resumed:** rebase `htp/194-s1` onto the teardown fix and whatever
FSU changes in the two-session layout, re-stage 1b with
`/local/mnt/workspace/htp_moe/194/s1b/stage_s1b.sh`, and re-read plan
194 §3.4 against FSU's MoE bytes before building L5 / L6.

## Sitting 1 as read (2026-09-30 11:01–11:2x, R3CY10WM83Y, after a reboot)

Logs `/local/mnt/workspace/htp_moe/194/s1/logs/`. **Stopped** at
`sanity_after_E0_G512_r1`: `cannot register a 2048x3584 weight …
fastrpc_mmap(64 MiB) failed` — the per-boot 256 MiB mapping loss (it hits
about one app run in 10–15, A or E; the teardown fix is being built on the
htp_moe track). The ceiling read 3840 after every run before it. No
PPL / mc-40 / lanes cell ran; they move to sitting 1b below.

* **Canary: 3 PASSED.** `FC_Q4_FIELD total bad=0`, **`FC_Q4_NATIVE total
  bad=0`**: the native pair equals `q4_gemv_native_det.h` on silicon, all
  five shapes, three feeds, 9 rows each — `hvx_emu`'s model of the hf
  widening product and the qf32 → hf narrowing holds. Isolated rates at 6
  lanes (`gemv_us`): K=2048 shapes 68–70 GB/s on either feed; K=7168 N=2048
  121 µs VTCM (68 GB/s) but **251 µs L2 (33 GB/s)**; native quantizer 6 µs
  at K=2048, 21–27 µs at K=7168.
* **Speed, G = 64** (decode tok/s r1 / r2; prefill r1): A 57.25 / 53.69
  (579); **E0 30.35 / 28.91**; **E1 31.73 / 31.70**. G = 512 r1: A 53.00,
  E0 30.83. E1 − E0 ≈ +6 % (≈ −1.5 ms a token), well under L1's projected
  −3.8.
* **E0 text = A** at G = 64 (both runs); **E1 text differs by one word**
  at G = 64 ("along it stand" → "along it stands", sentence 4) — the first
  non-exact kernel is visible in greedy text, as expected; P1–P4 judge it.
* **Per kind, S2 (G = 64 r1, ms a token)**:

  | kind | E0 | E1 | Δ | bytes a token | E1 rate |
  |---|---|---|---|---|---|
  | FC | 5.983 | 5.031 | −0.95 | ≈ 205 MB | 41 GB/s |
  | DENSE_FFN | 2.220 | 1.567 | −0.65 | 49.5 MB | 32 GB/s |
  | LM_HEAD | 3.248 | 3.274 | +0.03 | 147.5 MB | 45 GB/s |
  | sum (the `s2 fc+dense_ffn+lm_head` line) | 11.452 | 9.873 | −1.58 | 402 MB | 41 GB/s |

  LM_HEAD did not move: it was already at its feed rate under the exact
  kernel (K = 2048, 6 lanes). FC and DENSE_FFN dropped by what compute was
  and now sit at 32–41 GB/s — **the feed, not compute, bounds all three
  now**. The same kernels in isolation read 68–70 GB/s except the L2 feed
  at K = 7168 (33 GB/s, and the graph runs that shape at 3 lanes): the
  DENSE_FFN down is L2-feed-bound, and the in-graph K = 2048 FCs run at
  ≈ 60 % of their isolated rate (candidates: per-part fork/join and DMA
  start on the small parts — q | k | v, N = 512 parts read 7 GB/s
  isolated —, the prep and the join per op, DDR shared with S1's
  prefetch-free idle; the lane ladder of 1b reads the K = 7168 part).
  Plan §3.1's −3.8 assumed ≈ 50 GB/s in-graph; the reading is 41.

### The L0 split (E0_G64_r1; ms a token)

The tok/s cell (2109 ms / 64 = **32.95**) decomposes as:

| part | ms | what | removable by |
|---|---|---|---|
| S2 kernels | 13.17 | the per-kind line's sum | L1 (done, −1.6), L3, L7 |
| S1 kernels | 10.93 | router 0.80 + MoE 10.13 | L2 (router −0.65), L5 |
| in-DSP non-kernel | 1.15 | s2_wall − both kernel sums: the hops' post → wake-up (hop_us 0.32 + 0.32) and 0.51 of per-op / per-stretch overhead (slot copies, pool fork / join / park, cache clean of the row) | L4 (deadline wait: the 0.64 of hops → ≈ 0.1) |
| **wake** | **3.82** (2.99 / 3.47 / 3.21 in the other runs) | rt − s2_wall: outside both DSP walls, i.e. the ARM post → S2's queue thread → … → S2's answer → the ARM's blocking read. Not S2's first-op latency (that is inside s2_wall, whose clock starts in `nntr_hvx_token_run`). Both DSP queue threads block (`NNTR_HTP_E2E_SPIN_US=0`) and so does the ARM's read, so this is the two wake-ups a token; whether it is the dispatch or the return side, 1b's new line `L0 wake us/token disp s1= s2= s2_pkt= ret s2= clk_resid=` says (the DSP stamps read / write on the QTimer, the ARM on cntvct_el0; `clk_resid` ≈ 0 proves they are one counter) | **L0**: the ARM spins on S2's answer (read_noblock, bounded by the token) and S2's queue thread spins between tokens — S2 only: nothing computes on the DSP between S2's answer and the next packet, so the spin steals no lane (rule 56's objection is to S1 spinning while S2 computes). Expected ≈ −3 ms |
| ARM between tokens | 1.67 | arm_us: the 228-node walk with the hooks, sampler 0.002, register (tokenizer + print) 0.11 (OP-TIME) — so ≈ 1.5 is the walk | L0: the one-call decode step (skip the resident walk) ≈ −1.2 |
| one-time, first decode token | ≈ 140 ms a run | `NNTR_OP_TIME`: layer2_attention max 57.8 ms, layer0 rms_norm 23.9, the other attention layers ≈ 10 each — the first decode token's KV seed / E2E start inside the walk. 2.2 ms a token at G = 64, 0.27 at 512, 0.14 at 1024 | move it to the end of the prefill (not per token) |

So of the "≈ 8 ms outside S2's wall" (34.0 − 26.0 at E5f): wake 3.0–3.8,
ARM walk 1.5–1.7, the first token's one-time 2.2 at G = 64 only; the
DSP-side 1.15 is inside the walls. L0 as planned (bounded spins + one-call
step) removes ≈ 4–4.5 ms a token at every G, ≈ 6.5 at G = 64 with the
one-time moved; plan §3.4's L0 row (−5, range −3…−7) holds.

### Runner fix

`s2 arena after S1 (3840), feed=l2` read BAD on every E run: the pattern
still expected E5f's trailing `s1_heap_kib=`, which the s2 line no longer
prints. The runs were fine; run_s1b.sh matches `… s1_arena_mib=3840`.

## Sitting 1b (staged from `htp/194-s3` @ `b0248a63`; run 2026-09-30, read above)

`bash /local/mnt/workspace/htp_moe/194/s1b/run_s1b.sh R3CY10WM83Y` (any
attached S25 Ultra serial) — **≈ 110 min, reboot first**. The cells
sitting 1 did not reach: canary; speed A E0 E1 E1 E0 A + an E0
`NNTR_OP_TIME=1` run at G = 512 and 1024; the lane ladder (E1 G = 64 at
6,3 / 6,6 / 8,8) + an E0 L0 run at G = 64 (the wake split); the 8 prompts
at G = 256 (A self, A forced p01, E0 / E1 forced, E1 free text); mc-40 (A
and E1 all 40, E0 q01–q08); **new, last and informational: block `e2`**
— E1, E2 (`NNTR_HTP_PPL_LEVERS=0x1F`: L0–L4) and E2s (E2 with
`NNTR_HTP_E2E_HOP_SPIN_US=200`) at G = 64, for the S3 levers' speed and
L0 split on silicon (≈ 8 min; not a gate — their P1–P4 is sitting 2's).
**Resumable**: each block leaves `logs/done/<block>`; after any STOP,
reboot and run the same command — it skips finished blocks and restarts
the unfinished one. The summary (nulls, P1–P4, speed, L0 with the wake
split, lanes, e2) prints after the 17th block. Dry-run on the workstation
with a fake `adb` (every block, a stop injected mid-block, the resume and
the summary), not on the phone.

The binary carries the teardown fix (the S1 arena released on the DSP and
unmapped before the close) and every S3 lever; E0 (mask 0) and E1 (mask
2) run the same code paths as sitting 1's apart from that fix, so their
cells read as sitting 1's. Expected lines, in addition to the ones listed
under Steps: every E2 / E2s run `[HTP] ppl levers=0x1f L1=native_fc` and
`[HTP] ppl levers L2=router_vec L3=norm_conv_swiglu_vec L4
hop_spin_us=50 L0 s2q_spin_us=3000 arm_spin_us=1000` (200 for E2s), a
`token driver: L0 spins … arm_hits=… early_posts=…` line at the close,
and the same clean close (`timeouts=0/0 stale=0/0 … id_mismatch=0`).

| file (under `/local/mnt/workspace/htp_moe/194/s1b/`) | md5 | built with |
|---|---|---|
| app/libnntr_hvx_skel.so | `f87aef1a429bd9c5a301fbb5792aa603` | `test/htp/build.sh` (`UNDEFINED SYMBOLS OK (58 runtime imports)`) |
| app/nntrainer_causallm | `f5afd05b2674d484ba637868ec5a5ffd` | `(cd builddir && ninja install)`, then `build_android.sh --htp --cache` |
| app/libcausallm_core.so | `ee0bc1d6fb83197845b00f0d098930b8` | 〃 (`NNTR_HTP_FORWARD_KINDS` ×2, `NNTR_PPL_DECODE_ALTS` ×2) |
| app/libnntrainer.so | `8e6127005cc2201b157109e93021bfdf` | 〃 (`jni/obj/local`; `NEEDED libsdkl.so, libcdsprpc.so`) |
| app/libccapi-nntrainer.so | `d4d3dbf4d82bfffd0812f0792d437823` | 〃 (`jni/obj/local`) |
| app/libc++_shared.so | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 sysroot |
| app/libsdkl.so | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL beta.2 `lib/6.4.0.1/armv8_android26` |
| app/unittest_hvx_softmax | `eab0334e10f40ac08b6b6c61f373995e` | `ndk-build` (canary) |
| app/unittest_hvx_two_sessions | `712a6a101678ad5344b4eb1974e496c2` | `ndk-build` (`TwoSessions.S1Ceiling`) |
| run_s1b.sh, the prompts, mc-40, tools | `md5.txt` | `stage_s1b.sh` |

The skel's md5 names one build, not its sources: `hexagon-link` stores
its command line, with randomly suffixed `/tmp/*.o` names, in the
binary, so two builds of the same commit differ. The staged copy and
`md5.txt` are what the runner checks on both ends.

## Why

Plan 194 step S2. Three readings decide what is built next: the **L0
split** (where the ARM's ≈ 8 ms outside S2's wall goes: the dspqueue
wake-up, the layer walk, the app's per-token path), **L1's Δ** (the native
FC / DENSE_FFN / LM_HEAD kernels: ≈ 85 → ≈ 12 packets a step; is the FC now
DMA-bound, and at which lane count), and **the first PPL of a non-exact
kernel** on the 8B (P1–P4). E0 is the null: the bit-preserving E5g path,
whose nll lines must equal A's to 17 digits.

## What changed since E5f/E5g (`origin/htp/132-partb-e3` @ `69251151`, rebased onto `htp_moe_ppl`)

* `NNTR_HTP_PPL_LEVERS=<mask>` (bit n = lever Ln; `2` = L1). Unset / `0`
  = E0, the CPU-exact kernels. Banner: `[HTP] ppl levers=0x2 L1=native_fc`.
* L1: `hvx_q4m1_prep_vec` + `hvx_q4m1_gemv_groups_native`, spec
  `nntrainer/tensor/q4_gemv_native_det.h`, under the op feed word's bit 16.
  Host: bit-exact to the spec 3028/3028; spec vs the CPU order 149.4 dB on
  natural rows (37.6 dB on rows built to sit on q rounding edges); 11.5
  packets a 32-column step on v79 -O3 (static count; the exact kernel's
  loop 48.5); in-process hd64 PPL −0.009 % vs E0, lfm25 logits 32.7 dB.
  **Deviation from plan §3.1's wording:** the running sum is an IEEE f32
  add per block (a `Q6_Vsf_*` pair), not a qf32 accumulator converted once
  a group: `hvx_emu` does not model a qf32 chain, so a qf32 sum could not
  be held to a spec bit for bit (the device canary). It costs ≈ 1
  conversion a block; the static count above includes it.
* L0: `[HTP] token driver: L0 us/token rt= s2_wall= s1_wall= wake= hop_us
  s1= s2= arm_fwd= arm_us= arm_n=` at every E run's close (rt the ARM's
  round trip, wake = rt − S2's wall, hop_us the post → wake-up time inside
  each wall, arm_fwd tokenForward on the ARM, arm_us the ARM between two
  tokens: layer walk + sampler + tokenizer + print).
* `NNTR_PPL_DECODE_ALTS=<ids>` and the mc-40 set
  (`docs/measurements/prompts/mc-40*`, `tools/htp/mc_ids.py`,
  `tools/htp/mc_score.py`).

## Artifacts (sitting 1; built on the workstation, SDK 6.4.0.1, HexKL beta.2 6.4.0.1, NDK r30, v79)

Sitting 1b's artifacts and command are in section "Sitting 1b"; the
fill-in tables under Results serve both sittings.

| file (under `/local/mnt/workspace/htp_moe/194/s1/`) | md5 | built with |
|---|---|---|
| app/libnntr_hvx_skel.so | `5f4980e72d2ea961d54c3b3c615f8302` | `test/htp/build.sh` |
| app/nntrainer_causallm | `f5afd05b2674d484ba637868ec5a5ffd` | `build_android.sh --htp --cache` |
| app/libcausallm_core.so | `a7cf87090d77aed9b89136c425dc0ebd` | 〃 (`NNTR_HTP_FORWARD_KINDS` ×2, `NNTR_PPL_DECODE_ALTS` ×2) |
| app/libnntrainer.so | `9e1a5660b4bbb379fafaf49306802270` | 〃 (`jni/obj/local`; `NEEDED libsdkl.so, libcdsprpc.so`) |
| app/libccapi-nntrainer.so | `d4d3dbf4d82bfffd0812f0792d437823` | 〃 (`jni/obj/local`) |
| app/libc++_shared.so | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 sysroot |
| app/libsdkl.so | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL beta.2 `lib/6.4.0.1/armv8_android26` |
| app/unittest_hvx_softmax | `d99abba7e1c4702e1e9363d8064bc501` | `ndk-build` (canary: `HvxFcQ4.MatchesSpecBitExact`, `NativeMatchesSpec`, `SmallOpsMatchSpec`) |
| app/unittest_hvx_two_sessions | `46d6dadd85dc1a05cfc3e873abe9cca1` | `ndk-build` (`TwoSessions.S1Ceiling`) |
| app/prompt512.txt + app/bitset-0{2..8}-*.txt | as `docs/measurements/prompts/README.md` | the 8-prompt set |
| app/mc/q01..q40.{txt,ids}, app/mc/alts.txt | `md5.txt` | `tools/htp/mc_ids.py` |
| run_s1.sh | `18788dfb172b3201ea6a1d26d16cca66` | this sitting's runner |
| NPU model `models/q40-qs4cx-wh` on the device | unchanged since #132 E5 | — |

`md5.txt` in the stage lists every file; the runner checks it on both ends
and stops on a mismatch.

## Steps (sitting 1; 1b: the same, with `run_s1b.sh` in step 2)

1. **Reboot the phone**, let it idle to zone0 ≤ 35 °C.
2. `bash /local/mnt/workspace/htp_moe/194/s1/run_s1.sh R3CY10WM83Y`
   (≈ 110 min). Everything goes to `logs/sitting.out`; one log per run in
   `logs/`. It stops by itself on `0x8000040e` (stale skel), `LEAK` (S1's
   ceiling < 3840 MiB after a run), a failed token (`AEE_EEXPIRED`) or
   `FATAL`.
3. Paste the summary blocks of `sitting.out` (from `--- nulls` to `=== done`)
   below, the G = 64 texts (`logs/texts_g64.txt`) into the text table, mark
   `text approved`, commit this file on the branch.

Order: md5 → config (greedy, `bad_word_ids [124900]`, `moe_engine htp`) →
ceiling → canary → warm-up A → per G in 64 / 512 / 1024: cool,
`A E0 E1 E1 E0 A`, `E0 NNTR_OP_TIME=1` (L0) → lanes `E1` G=64 at
`NNTR_HTP_FC_LANES` 6,3 / 6,6 / 8,8 → 8 prompts at G = 256: A self (writes
`cont_pXX.ids`), A forced p01 (null), E0 forced, E1 forced, E1 free text →
mc-40 at G = 1: A all 40, E0 q01–q08, E1 all 40. After every run the S1
ceiling; after every E run an A sanity (G = 8).

Expected lines: every E run `[HTP] ppl levers=0x0 L1=exact` (E0) or
`0x2 L1=native_fc` (E1), `s2: fc arena weights=67 handles=74 … feed=l2`,
`token driver: close tokens=… hops/token=44.00 … timeouts=0/0 stale=0/0 …
id_mismatch=0`, `calls/token=1.00`, `s2: close … unmap_fail=0
detach_fail=0`, `L0 us/token …`; every A run `[HTP] dspq: on`, no
`s2: open`; the canary `[  PASSED  ] 3 tests` with `FC_Q4_NATIVE total
bad=0`.

## Results (sitting 1b; G = 64 speed and texts from sitting 1)

### Nulls and gate (plan 194 section 1)

| check | pass | read |
|---|---|---|
| A forced p01 ≡ A self (17 digits) | identical | identical |
| E0 nll ≡ A, 8 prompts | 8/8 | 8/8 |
| E0 mc ≡ A, q01–q08 | 8/8 | 8/8 |
| E0 text ≡ A r1 at each G | 0 DIFF | 0 DIFF (G 512 / 1024, r1 and r2) |
| canary (exact FC, native FC, small ops) | 3 PASSED, native bad=0 | 3 PASSED, `FC_Q4_FIELD total bad=0`, `FC_Q4_NATIVE total bad=0` |
| **P1** pooled PPL, E1 / A (2048 steps) | ≤ 1.02 | A 1.210766, E1 1.213821, **1.0025: pass** |
| **P2** per prompt, max E1 / A | ≤ 1.05 each | p01 1.0002, p02 1.0015, p03 0.9981, p04 1.0096, p05 1.0015, p06 0.9973, p07 **1.0110**, p08 1.0011: **pass** |
| **P3** mc-40 right (A, E1), Σnll ratio | E1 ≥ A − 2, ≤ 1.05 | A 12/40 (143.54), E1 12/40 (140.99), 0.9822: **pass** (36/40 picks equal; A is at chance, see "mc-40's power") |
| **P4** new loops (E1 where A has none) | 0 | **1, p04 (A L2 0.44 loop=0, E1 L2 0.74 loop=1): FAIL** (0 unreadable logs; reading: A loops on p04 too, below the threshold) |
| ceiling after every run | 3840 | 3840 except 3584 after nine runs (the LEAK stops, Notes) |

### Speed (prompt 512; decode tok/s all / last 64; prefill tok/s r1 / r2)

| variant | G | r1 | r2 | prefill | text = A r1 |
|---|---|---|---|---|---|
| A | 64 | 57.25 (s1) | 53.69 (s1) | 579 (s1) | (ref) |
| E0 | 64 | 30.35 (s1) | 28.91 (s1) | | same (s1) |
| E1 | 64 | 31.73 (s1); 34.26 (1b e2); 30.32 (1b lanes 6,3) | 31.70 (s1) | 508.9 (e2) | one word (s1, e2) |
| A | 512 | 56.21 / 55.70 | 54.89 / 53.60 | 460.0 / 502.5 | (ref) |
| E0 | 512 | 32.19 / 32.10 | 30.94 / 31.17 | 453.5 / 500.5 | same |
| E1 | 512 | 33.83 / 34.10 | 33.10 / 33.21 | 479.4 / 515.1 | DIFF (word 466, "stand" → "stands") |
| A | 1024 | 52.89 / 52.85 | 51.93 / 45.81 | 530.0 / 484.8 | (ref) |
| E0 | 1024 | 32.25 / 31.89 | 31.19 / 31.22 | 451.1 / 477.6 | same |
| E1 | 1024 | 33.80 / 33.33 | 32.91 / 32.90 | 483.9 / 515.1 | DIFF (word 466, as at 512) |

Reference (E5f, same unit, 2026-09-30): A 53.96 / 53.57 / 51.49, E 29.38 /
30.88 / 30.80 tok/s at G 64 / 512 / 1024; S2 per kind FC 6.06, DENSE_FFN
2.24, LM_HEAD 3.49 ms (11.79 of the Q4M1 kinds), S2 wall 26.0, ARM token
29.8 ms. Goal ≥ 50 at every G (plan 194 section 1; §3.4 expects E1 at
≈ 39.7 / 42.2 only after L0 too). E2 (informational, G = 64): 38.95.

### L0 split (E0, `NNTR_OP_TIME=1`; µs a token)

| G | rt | s2_wall | s1_wall | wake (disp s1 / s2, ret s2, clk_resid) | hop_us s1 / s2 | arm_fwd | arm_us | op_time ms: token / sample / register / unattributed |
|---|---|---|---|---|---|---|---|---|
| 64 | 30651.8 | 27770.5 | 23731.0 | 2881.3 (59.6 / 94.2, 2783.6, −0.2) | 336.1 / 342.0 | 30663.9 | 1507.7 | 34.566 / 0.002 / 0.103 / 0.543 |
| 512 | 30418.7 | 27048.0 | 23210.0 | 3370.7 (59.7 / 63.4, 3304.0, −0.1) | 319.5 / 339.8 | 30427.8 | 1547.5 | 32.294 / 0.002 / 0.095 / 0.559 |
| 1024 | 30508.8 | 27305.3 | 23556.8 | 3203.5 (60.5 / 66.6, 3133.8, −0.1) | 332.3 / 339.6 | 30518.2 | 1588.3 | 32.278 / 0.003 / 0.097 / 0.588 |

`[OP-TIME]`'s lm_head row (30.45–30.69 ms, 88–95 %) is the ARM waiting in
the lm_head hook for the whole DSP token, not a kernel.

### Per session (G = 512 r1, `graph[S1|S2] per-kind` lines)

| variant | S1 router / MoE ms | S2 FC / DENSE_FFN / LM_HEAD ms | S2 wall | arm token_ms |
|---|---|---|---|---|
| E0 | 0.794 / 10.042 (round 0.456) | 5.937 / 2.154 / 3.129 (sum 11.220) | 25.099 | 29.121 |
| E1 | 0.822 / 10.362 (round 0.471) | 4.910 / 1.486 / 3.157 (sum 9.554) | 23.775 | 27.650 |

### FC lane ladder (E1, G = 64)

| `NNTR_HTP_FC_LANES` | decode tok/s | s2 fc+dense_ffn+lm_head ms/token |
|---|---|---|
| 6,3 (default) | 30.32 | 10.037 |
| 6,6 | 30.32 | 10.758 |
| 8,8 | 31.71 | 10.073 |

Cross-block drift (this block vs e2's E1, same config) is ≈ 13 %: the
ladder does not separate 6,3 from 8,8.

### e2 (informational; E1 / E2 / E2s at G = 64)

| variant | prefill | decode tok/s | text = E1 of the block | L0 us/token (rt / s2_wall / wake / arm_us) | hop_us s1 / s2 | L0 spins (arm_hits, early_posts) |
|---|---|---|---|---|---|---|
| E1 | 508.9 | 34.26 | (ref) | 25450.7 / 22595.0 / 2855.6 / 1559.3 | 363.6 / 323.6 | — |
| E2 | 479.4 | 38.95 | DIFF (first token) | 22651.7 / 22579.2 / 72.5 / 419.9 | 157.6 / 218.9 | 63/64, 63 |
| E2s | 459.6 | 38.95 | DIFF (≡ E2) | 22788.5 / 22716.2 / 72.3 / 395.5 | 57.2 / 103.1 | 63/64, 63 |

## Text approval

| variant | decode PPL pooled (G=256, forced on A) | mc-40 right / Σnll | generated text (G=64, r1; sitting 1 unless stated) | text approved (user: y/n) |
|---|---|---|---|---|
| A | 1.210766 | 12/40 / 143.54 | "town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum that opens only on summer weekends, and a lifeboat station" | (reference) |
| E0 | (≡ A, 8/8) | (≡ A, q01–q08) | identical to A | |
| E1 | 1.213821 | 12/40 / 140.99 | as A with "along it **stands**" (sitting 1 and 1b e2) | |
| E2 (1b e2, not gated) | not measured | not measured | "final answer should be a single sentence, but the description should be a long paragraph. However, the user may have wanted a more detailed description. The user may have wanted a more thorough description. The user may have wanted a more detailed description of the town. The user may have wanted a more detailed description of the head" | |

G = 256 p01 (A self, E1 free): `logs/texts_p01.txt`. The two share the
first sentence (E1 "stands"), then differ in which sentences of the
prompt they re-use; both re-use prompt sentences, neither loops (L2 0.14 /
0.10).

## Notes from the run

* Unit `R3CY10WM83Y`, rebooted before the first invocation (uptime 49 s at
  14:38). There were ten invocations, 14:38:48–17:12:14 KST,
  each after a reboot. Run by the orchestrator at the user's explicit
  request; the implementer read the logs and ran no adb.
* **Nine `STOP: LEAK` stops (S1 ceiling 3584 after the run).** The
  teardown fix did not remove the per-boot mapping loss. The list gives the run
  after which each stop fell, with the app runs of that boot (A sanities
  and the warm-up included; each is followed by an S1Ceiling gtest):
  A_G512_r2 (boot 1, 11 runs; an A run, no S2 in it), E0_G1024_r1 (14),
  text_E1_p01 (27), ppl_E0_p05 (31), mc_E0_q02 (35), mc_E1_q16 (63),
  mc_A_q20 (28), mc_A_q33 (39), sanity_after_mc_E1_q40 (23; an A run).
  The last boot ran the e2 block (3 runs + 3 sanities) and finished. Four stops followed an A run and
  five an E run, and every boot had E runs before its stop. The loss comes
  after 11–63 runs of a boot; the G = 1 mc runs went furthest, so the
  count is not fixed.
* After each stop: a reboot and a resume. A block cut by a stop was re-run
  from its first run (speed_G512, speed_G1024, p01, p05, mc01, mc02 in
  boot 7), except the mc blocks after the runner change below.
* **Runner changed mid-sitting (by the orchestrator).** `mcrun` in
  `run_s1b.sh` keeps an mc question run whose log already holds its `[PPL]
  decode` line instead of redoing it, because an mc block is 28 app runs and did
  not fit in one boot. `md5.txt`'s runner line was updated to match
  (`b80993c1…` → `77fc90bd…`); the originals were kept as `run_s1b.sh.bak` /
  `md5.txt.bak`. The diff is the `mcrun` function and its three call sites only.
  From boot 8, `loop_s1b.sh` (reboot, run, retry only on `STOP: LEAK`, ≤ 12
  passes) drove the boots; 44 mc runs were kept from earlier boots. So the
  mc-40 cells come from several boots (mc01 whole in boot 6; mc02 q11–q20
  across boots 7–8; mc04 across 8–9). The app and skel md5s were unchanged,
  since the runner re-checks them every invocation (`MD5 OK` each time).
* **speed_G512 ran twice in full**: boot 1's set is void because of the stop
  after its last run. For the record, prefill / decode tok/s: A_r1 577.2 / 57.48,
  E0_r1 570.8 / 31.72, E1_r1 544.1 / 32.89, E1_r2 529.5 / 32.81, E0_r2
  474.1 / 30.86, A_r2 515.1 / 56.19. Boot 2's set is in Results.
* Thermal: the runner's `cool()` gate held zone0 ≤ 35 °C before each block
  (30 s waits, 2–4 per block). `therm.log` checkpoints run 36.3–59.4 °C at
  block ends, and the battery went from 100 to 81 %.
* `[OP-TIME]` on the E path puts about 30.5 ms in lm_head: the ARM waits there
  for the whole DSP token. This is the round trip, not an lm_head kernel.
* Runner cosmetics, no effect on a verdict: the P2 per-prompt lines went
  to the file ` P2 FAIL` in the stage directory through an awk `printf`
  redirect, and P4's `( unreadable logs)` is an unset awk counter, i.e. 0.
  A stray ` P2 FAIL` also sits untracked in the main checkout (a dry-run
  artefact); fix `printf …, (r > 1.05 ? …)` before sitting 2's runner.

## What each outcome means

* E0 not ≡ A anywhere: a broken build or skel, not a lever — nothing of
  E1 is read; re-stage.
* `NativeMatchesSpec` bad > 0 with E0 fine: silicon's hf widening product or
  qf32 → hf narrowing is not what `hvx_emu` models; E1's PPL still counts
  (the gate is PPL, not the spec), the host spec is corrected before S3.
* E1 fails P1–P4: L1's numerics go back to S1 before anything else is
  built (the quantizer first: f16(d) from the true amax / 127 and the
  CPU's id; then an exact per-block product).
* The L0 row decides S3's L0 design (plan 194 §3.1: the one-call decode
  step if `arm_us` dominates, the ARM's bounded spin if `wake` does).
