# Measurement 201 S2 (two-PD half): the expert pool inside the per-token E2E entry

Branch `htp/201-pool-miss-path` (PR #203); artifacts built from `715dffb62`
(the later commits on the branch touch docs and the evictor tool only),
staged at `/local/mnt/workspace/htp_moe/201/s2/`. Run by the implementer on
`R3CY10WM83Y` (user, 2026-09-30: device measurements are run directly, S25
only). **Estimated device time: ≈ 60 min, reboot first.**

## Why

Plan 201 S2, the two-PD half: #203 put the pool inside `NNTR_HTP_E2E=1`
(S1's miss rounds served by the ARM pool server). This sitting reads what
that costs per pool size against A and the all-resident E0, and whether the
three things the host twin cannot show hold on silicon: the ARM reading the
uncached mailbox page, the DSP's cache on a rebind in the middle of a token,
and the pool server's spin. The one-PD variants (P1) wait for plan S1's
remaining item (the FC set on S1's arena).

## Variants (one binary set; environment only)

| variant | what |
|---|---|
| **A** | hybrid, nothing set (the unchanged reference, run first) |
| **E0** | `NNTR_HTP_E2E=1`, all experts resident (two PDs) |
| **P28 / P24 / P16** | `NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=C`, the model file pre-read (warm), its resident MiB printed before the run |
| **P16c** | P16 with `page_cache_evict` every 20 ms (cold), G = 64 once |
| **P32** | every expert in the pool: the no-miss control, G = 64 once |

Order: A warm-up, evict check; G = 64: cool, `A E0 P28 P24 P16`, cool,
`P16 P24 P28 E0 A`; Pbest = the fastest pool at G = 64 (P24 if that is P28);
G = 512 / 1024: cool, `A E0 Pbest P28`, cool, `P28 Pbest E0 A`; then P16c,
one `NNTR_HTP_PROFILE=2` run of P28 / P24 / P16, P32 (all G = 64). Ceiling
after every run. Cooling (zone0 ≤ 35 °C) before each half block, not only
each G (S0's r2 cells at G = 1024 ran at 65 °C).

Gates: every text = A r1 of its G (this path is bit-preserving at mask 0);
`calls/token=1.00`; the driver's close line `timeouts=0/0 stale=0/0 …
id_mismatch=0`; S2's clean close; ceiling 3840 after every run; no LEAK.

## Artifacts

| file (under `/local/mnt/workspace/htp_moe/201/s2/`) | md5 | built with |
|---|---|---|
| app/libnntr_hvx_skel.so | `6956a3956f4b99a9eac376f203591c9d` | `test/htp/build.sh` (`UNDEFINED SYMBOLS OK (62 runtime imports)`) |
| app/nntrainer_causallm | `c106135d7074b3bd1129a17e0c57cb01` | `(cd builddir && ninja install)`, `build_android.sh --htp --cache`, stale stub .o deleted |
| app/libcausallm_core.so | `8f532d8a143a383d6cdba46755cd67ab` | 〃 (`NNTR_HTP_FORWARD_KINDS` ×2) |
| app/libnntrainer.so | `7b4c5038efc53c5b278617405260de91` | 〃 (`NEEDED libsdkl.so, libcdsprpc.so`) |
| app/libccapi-nntrainer.so | `ad46760cde21a9617ada13e0f310092a` | 〃 |
| app/libc++_shared.so | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 sysroot |
| app/libsdkl.so | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL beta.2 |
| app/unittest_hvx_two_sessions | `172f55aede7c7c27b85de443df489de2` | `ndk-build` (`TwoSessions.S1Ceiling`) |
| app/page_cache_evict | `42595651ef514e155887eb31f421b2dc` | NDK clang of `tools/htp/page_cache_evict.c` |
| run_s2.sh | `8d8703c78d34d70b1b7e91cd58ec3275` | `docs/measurements/201-s2-run.sh` |

Staged by `bash docs/measurements/201-s2-stage.sh`; run by `bash
/local/mnt/workspace/htp_moe/201/s2/run_s2.sh` (serial defaults to the S25).

## Smoke run before the sitting (22:10 KST, not a table cell)

P16 at G = 64, profiled, page cache emptied just before: `calls/token=1.00`,
`timeouts=0/0 stale=0/0 … id_mismatch=0`, first token id 4386 (= E0's),
`pool misses=677 misses/token=10.58 miss_wait_us/token=8615.0 rounds=484
arm_ms/round=1.545`, 27.61 tok/s. The L0 wake fell from E0's ≈ 4 ms to
0.21 ms (`ret s2=172.2`): the pool server's spin keeps an ARM core awake
through the token, which is what #194's L0 did on purpose.

## Sitting as read (2026-09-30 22:18–22:47 KST, R3CY10WM83Y)

Run by the implementer from the stage above: rebooted (uptime 80 s), MD5 OK
on both ends, one `LEAK` stop (ceiling 3584 MiB after `E0_G512_r1`, the
known per-boot mapping loss of E0's 3840 MiB), rebooted and resumed; the
ceiling was 3840 MiB after every other run. Every half block started at
zone0 ≤ 35 °C (33.9–34.7); within a block zone0 reached 54–62 °C.

**Gates: all hold.** Every pool and E0 run: `levers=0x0`, `calls/token=1.00`,
the close line `hops/token=44.00 … timeouts=0/0 stale=0/0 … id_mismatch=0`,
S2's clean close (the runner's `OK` lines; no `BAD` other than the one
below). **Every text equals A r1 of its G, 28 of 28 cells.** The runner
first reported 15 `text=DIFF`: all were the pool runs whose command printed
the evictor's `resident … MiB` line into the log, which the text strip did
not remove (P16c, which prints no such line, read `same`). With the line
stripped the same logs compare equal in all 28 cells; the runner is fixed.
So on silicon the ARM reads the uncached page, S1 applies the answers and
rebinds mid-token, and the bits stay A's, at C = 16 / 24 / 28 / 32, warm and
cold.

**The pool inside the E2E entry is faster than all-resident E0**, at every G:

| G | A | E0 | P28 | P24 | P16 | P16c | P32 |
|---|---|---|---|---|---|---|---|
| 64 | 53.92 / 55.90 | 30.78 / 30.33 | 31.89 / 36.12 | 33.83 / 35.28 | 31.02 / 32.19 | 13.14 | 35.60 |
| 512 | 57.62 / 57.30 | 31.06 / 33.23 | 38.42 / 39.70 | 37.13 / 37.03 | | | |
| 1024 | 55.12 / 53.76 | 31.78 / 31.78 | 37.63 / 39.10 | 37.15 / 36.73 | | | |

(decode tok/s, r1 / r2; Pbest at G = 64 was P24, 34.55 against P28 34.00.)

**Why it is faster: the pool server's spin, not the pool.** P32 (every
expert in the pool, no decode miss) reads 35.6 at G = 64 against E0's
30.3–30.8. The server thread spins on an ARM core for the whole token, and
that removes the ARM's wake-up on S2's answer: E0's `wake` 3.4–3.7 ms (`ret
s2` 3.6 ms) against 0.19–0.39 ms in every pool run, and the walk between
tokens (`arm_us`) 1.6 ms against 0.36–0.56. That is #194's lever L0 by
accident (≈ −4.5 ms a token). The ponytail on the spin (one ARM core
busy for the whole token) is now a measured trade, not only a cost.

**What the misses cost** (the driver's close line; ms a token):

| cell | misses a token | S1's miss wait | server ms a round | read ms a miss (profiled G = 64) |
|---|---|---|---|---|
| P28 G 64 / 512 / 1024 | 1.89 / 0.34 / 0.17 | 4.20, 0.82 / 1.14, 0.90 / 0.10, 0.09 | 1.0–4.6 | 3.79 |
| P24 G 64 / 512 / 1024 | 3.58 / 1.15 / 0.96 | 2.47, 1.19 / 2.83, 1.22 / 0.58, 0.43 | 0.87–3.2 | 0.72 |
| P16 G 64 | 10.58 | 3.06, 2.91 | 0.80 | 0.53 |
| P16c G 64 | 10.58 | 46.5 | 6.5 | — |

* Misses a token fall with G (the first tokens after the prefill miss
  most): at G ≥ 512 P28 pays 0.1–1.1 ms a token, P24 0.4–2.8.
* A P28 miss costs more than a P24 / P16 miss (3.79 against 0.72 / 0.53
  ms read, profiled; r1 cells 2–4× their r2), as F28 did in S0. The model
  file was 95–100 % resident in the page cache before these runs (the
  runner's `resident` line: 3931–4116 of 4116 MiB), so **page-cache
  residency does not explain it**. Not explained; the bigger arena (3.3
  GB of ION pages) and the reader threads' placement are the next reads.
* Cold (P16c): 4.4 ms a miss of wait, S1 waiting 46.5 ms a token: 13.1
  tok/s, as S0's F16c (13.4).

**Where P28 stands against A at G = 512** (r1, ms a token): A 17.4; P28
26.0 = rt 25.2 + arm 0.5 + the rest. S2's wall 24.9 (FC + DENSE_FFN +
LM_HEAD 10.85 against 8.06 isolated: no VTCM on S2) and S1's MoE 10.3 +
router 0.8 run in turn across 44 hops (0.33 ms each side). One PD (plan S1's
last item) is the next structural read.

**Prefill** (tok/s r1 / r2, G = 512): A 552 / 448, E0 577 / 442, P28 500 /
545, P24 540 / 417; the spread is run to run, no variant below −5 % of A on
the mean of the two.

**Next read:** the one-PD half, `201-one-pd.md` (2026-10-01): one PD with
the pool at C = 28 reads 43.0–43.4 tok/s against this sitting's P28 ≈ 37–40
and E0 ≈ 31, text = A.
