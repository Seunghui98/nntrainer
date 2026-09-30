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

## Results

(filled from the sitting's logs below)
