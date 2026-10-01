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

## Sitting as read (2026-10-01 09:43–09:58 KST, R3CY10WM83Y)

Rebooted (uptime 75 s), MD5 OK on both ends, no stop, no `BAD` line (77
`OK` checks), ceiling 3840 MiB after all 22 runs. Each half block started
at zone0 31.6–34.3 °C; inside a block it reached 55–60 °C. **G = 1024 was
not run**: the S25 is being replaced by an S26 (user, 2026-10-01), so the
runner was stopped after G = 512 r2, during the cooling wait before G =
1024 (no app was running; checked with `ps` on the phone).

**Every text equals A r1 of its G, 20 of 20 cells** (E0, P28, Q28, Q29 at
G = 64 and 512). Every E / P / Q run: `levers=0x0`, `calls/token=1.00`,
`timeouts=0/0 stale=0/0 … id_mismatch=0`, `hops/token=44.00` (E0, P28) or
`0.00` with `pds=1` (Q28, Q29), S2's close `unmap_fail=0 detach_fail=0`.

### Decode tok/s (prompt 512; r1 / r2; last 64 in brackets at G = 512)

| G | A | E0 (2 PDs, all resident) | P28 (2 PDs, pool 28) | **Q28 (1 PD, pool 28)** | **Q29 (1 PD, pool 29)** |
|---|---|---|---|---|---|
| 64 | 56.74 / 54.51 | 30.51 / 30.46 | 36.72 / 35.81 | **42.98 / 43.13** | **43.45 / 43.66** |
| 512 | 48.45 / 54.23 (53.78 / 52.20) | 30.79 / 30.66 | 36.67 / 34.13 | **43.40 / 42.44** (45.36 / 44.48) | **42.40 / 43.51** (43.16 / 44.91) |
| 1024 | not measured (S25 withdrawn) | not measured | not measured | not measured | not measured |

Prefill (G = 512, r1 / r2): A 538 / 514, E0 528 / 496, P28 537 / 446, Q28
501 / 525, Q29 413 / 577; means within −5 % of A's except Q29 r1 (413), a
single low cell its r2 (577) does not repeat.

**One PD wins**: +6–7 tok/s over two PDs with the same pool (Q28 against
P28) and +12–13 over E0, at the same bits.

### Where the time goes (G = 512 r1, ms a token, the close lines)

| | E0 | P28 | Q28 | Q29 |
|---|---|---|---|---|
| ARM round trip `rt` | 30.59 | 26.63 | **22.38** | 22.81 |
| wake (rt − the DSP side's wall) | 3.00 | 0.35 | 0.20 | 0.22 |
| ARM between tokens `arm_us` | 1.59 | 0.33 | 0.37 | 0.45 |
| FC + DENSE_FFN + LM_HEAD | 13.05 (S2, L2-fed) | 11.64 (S2) | **9.20** (S1, `feed=vtcm`) | 9.16 |
| MOE + ROUTER_TOPK | 10.65 + 0.81 | 10.88 + 0.81 | 10.47 + 0.69 | 10.87 + 0.69 |
| hops | 44 × ≈ 0.33 | 44 × ≈ 0.33 | 0 | 0 |
| misses a token / S1's miss wait | — | 0.34 / 1.10 | 0.34 / 1.11 | 0.23 / 1.09 |

* The FC set on S1's VTCM: FC 4.80 ms against 6.29–6.77 on S2, DENSE_FFN
  1.37 against 2.16–2.25; the three kinds 9.2 ms, ≈ 1.1 ms above #178's
  isolated 8.06.
* The 44 hops and the second wall are gone (one stretch a token).
* C = 29 against 28: fewer misses (0.23 against 0.34 a token at G = 512,
  1.34 against 1.89 at G = 64) but the same speed within run to run; C = 30
  does not load beside the FC set. **C = 28 is the one-PD pool.**
* Against A (≈ 18.4–20.6 ms a token at G = 512 in this sitting) Q28 is
  ≈ 23 ms. What is left: the MoE round (10.5 ms, 0.48 ms a layer), the FC
  set's 1.1 ms over isolated, the misses (≈ 1.1 ms a token at G = 512, less
  at longer G as the pool settles), the small ops (≈ 2.4 ms: attention 0.71,
  router 0.69, conv 0.46, norms 0.33, ADD 0.09).

## Phone-side state, for re-staging on the S26

Recorded from the S25 (`SM-S938N`, build `S938NKSS9BZCH`, 11.1 GB RAM)
into `/local/mnt/workspace/htp_moe/201/phone_state/` (`ls.txt`,
`nntr_config.json`, `generation_config.json`, `device.txt`):

* App root `/data/local/tmp/nntrainer/causallm/`; each sitting pushes its
  whole app set into its own dir there: `s201s0/` (S0), `s201s2/` (two
  PDs), `s201s3/` (this sitting). Nothing else is installed by them.
* Model dir `models/q40-qs4cx-wh/` (the runners use `../models/q40-qs4cx-wh`
  from the sitting dir): `config.json`, `tokenizer_config.json` and the two
  configs as files; the weights `nntr_lfm2_8b_a1b_q40_arm.bin` and
  `tokenizer.json` are symlinks into `../lfm2.5-8b-a1b-q40-qs4cx-wh/`
  (4116 MiB file). Install from the workstation's
  `/local/mnt/workspace/models/lfm2.5-8b-a1b/q40-qs4cx-wh/` with
  `install_android.sh --model=…`.
* Config edits the runners make (idempotent): `generation_config.json`
  `"do_sample": false`; `nntr_config.json` `"bad_word_ids": [124900]`,
  `"moe_engine": "htp"` added after it, `"init_seq_len": 512`, and
  `"num_to_generate"` set per run (last value 512).
* What must be rebuilt for the S26 (v81): the skel (`HEX_ARCH=v81
  ./test/htp/build.sh`; the v79 skel will not load there) and the HexKL
  micro library for v81 (`HEXKL_ROOT/lib/<sdk>/hexagon_toolv19_v81`); the
  ARM app set (`nntrainer_causallm`, `libcausallm_core.so`,
  `libnntrainer.so`, `libccapi-nntrainer.so`), `libsdkl.so`,
  `libc++_shared.so`, `unittest_hvx_two_sessions` and `page_cache_evict`
  are arm64 and stay, as do the model file and the prompt. Re-check on the
  S26 before any table: the ceiling cell's 3840 MiB (a different SoC may
  map a different PD space, which moves the one-PD pool size: C = 29 here),
  the VTCM size the FC feed takes, and the runner's serial (the runners
  default to `R3CY10WM83Y` and must be given the S26's).
