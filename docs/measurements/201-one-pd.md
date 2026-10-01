# Measurement 201 S2 (one-PD half): one PD with the pool (P1) against two PDs (P2), E0 and A

Branch `htp/201-pool-miss-path` (PR #203); artifacts built from `f5e8b1648`,
staged at `/local/mnt/workspace/htp_moe/201/s3/`, run by the implementer on
`R3CY10WM83Y` (user, 2026-09-30: device sittings run directly, S25 only).
Companion of `201-fsu-e2e.md` (the two-PD half). **Estimated device time:
≈ 50 min, reboot first.**

## Why

Plan 201 §3.4: one PD removes S2's lack of VTCM (E0's FC + DENSE_FFN +
LM_HEAD at 10.9–12.7 ms against 8.06 isolated), the 44 hops and the second
session, at the price of a smaller pool (C = 28 or 29 of 32 beside the
448 MiB FC set). This sitting reads one PD against two in the same boot.

## Variants (one binary set; environment only)

| variant | what |
|---|---|
| **A** | hybrid, nothing set (the reference, first in each half block) |
| **E0** | `NNTR_HTP_E2E=1`, two PDs, all experts resident |
| **P28** | `NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=28`, two PDs (the S2 sitting's best at G ≥ 512) |
| **Q28** | P28 + `NNTR_HTP_E2E_PDS=1`: one PD, the FC set and lm_head on S1's arena beside the pool |
| **Q29** | the same at C = 29, the largest pool that loads beside the FC set (C = 30 fails: `S2 FC arena: fastrpc_mmap(32 MiB) failed`, 22:52 probe) |

Per G in 64 / 512 / 1024: cool, `A E0 P28 Q28 Q29`, cool, `Q29 Q28 P28 E0
A`; pools warm (file pre-read, resident MiB printed); ceiling after every
run. Gates: text = A r1 of its G; `calls/token=1.00`; close line
`timeouts=0/0 stale=0/0 … id_mismatch=0` with `hops/token=44.00` (E0, P) or
`0.00` and `pds=1` (Q); S2's close `unmap_fail=0 detach_fail=0`.

## Artifacts

| file (under `/local/mnt/workspace/htp_moe/201/s3/`) | md5 |
|---|---|
| app/libnntr_hvx_skel.so | `31c0a033a76fb68eb3a1137f3c23bb15` (`UNDEFINED SYMBOLS OK (62 runtime imports)`) |
| app/nntrainer_causallm | `c106135d7074b3bd1129a17e0c57cb01` |
| app/libcausallm_core.so | `8f532d8a143a383d6cdba46755cd67ab` |
| app/libnntrainer.so | `651b23dd87519fd37ba22b520d96215e` (`NEEDED libsdkl.so, libcdsprpc.so`) |
| app/libccapi-nntrainer.so | `ad46760cde21a9617ada13e0f310092a` |
| app/libc++_shared.so | `b1586b9b512712800fd36a24abac1c0a` |
| app/libsdkl.so | `0ad4e22a70e4f135bce38ad8fd1e001b` |
| app/unittest_hvx_two_sessions | `172f55aede7c7c27b85de443df489de2` |
| app/page_cache_evict | `42595651ef514e155887eb31f421b2dc` |
| run_s3.sh | `f16a7b8ce7692feb29ac4a22bf9605da` (`docs/measurements/201-s3-run.sh`) |

## Found on silicon before the sitting

* Smoke 2026-09-30 22:52 (`0faa98f01`): Q28 / Q29 at G = 64 run, text = A,
  `calls/token=1.00`, `hops/token=0.00`, `timeouts=0/0`, 43.3 / 43.8 tok/s,
  S1's FC 4.67 ms (S2 sitting's E0 5.97–6.60), `feed=vtcm`. But the close
  read `s2: close … unmap_fail=2 detach_fail=2`: the FC slots were released
  while S1's one graph still named them. The host twin showed the same
  (`unmap_fail=1 detach_fail=1`), unnoticed because no host line read it.
  Fixed in `f5e8b1648` (the one-PD close releases S1's graph first; the
  in-process one-PD lines now require `unmap_fail=0 detach_fail=0`).
  Re-smoked on silicon 2026-10-01: Q29 `s2: close … unmap_fail=0
  detach_fail=0`, `arena: chunks unmapped 14/14`, ceiling 3840.

## Results

(filled from the sitting's logs below)
