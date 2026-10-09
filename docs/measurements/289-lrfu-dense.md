# Measurement 289: LRFU expert-pool eviction and the dense FFN inside the miss wait (2-bit file, q8, S25)

Issue #289. Branch `htp/289-lrfu-dense` = `htp_decode` @ `6e57b2f2f` plus `eefbc05fd`
(LRFU, `NNTR_MOE_LRFU`) and `f1a161346` (dense branch in the miss round,
`NNTR_HTP_DENSE_EARLY`). S25 `R3CY205ZMND`, 2026-10-09 23:5x – 2026-10-10 00:3x KST.
One run per cell, no repeats. Battery 32.0–33.0 °C at every start. Runner
`282-run.sh one … E <p> 512 24 <tag>` with `NOCEIL=1`.

Every cell has the 282 E levers on: `NNTR_HTP_DROP_HOST_FC=1 NNTR_HTP_E2E_FREE_KVQ=1
NNTR_HTP_ATTN_M1_Q8=2`. Configs are `s282cfg/r2_E_p*_g512`, which sample with the app's
fixed seed. Install `causallm/s289`:

- skel `a6677225`
- `libnntrainer.so` `4d7a7258`
- `libcausallm_core.so` `c8e61036`

## Result (C = 24)

`token ms` is the ARM's `token_ms`. `fc+dense+head` is the DSP's per-kind sum; under DENSE_EARLY the dense part is inside the MOE op.

| prompt | cell | decode tok/s (all / last 64) | token ms | misses/token | miss wait ms | rounds | fc+dense+head ms | text md5 |
|---|---|---|---|---|---|---|---|---|
| 1024 | base | 8.87 / 12.90 | 81.7 | 73.18 | 24.6 | 2843 | 31.1 | `698a29f5` |
| 1024 | LRFU 32 | 8.95 / 13.57 | 79.6 | **68.19** | 22.9 | 2837 | 30.9 | `698a29f5` |
| 1024 | DENSE_EARLY | 9.21 / 13.96 | **74.9** | 73.18 | 17.3 | 2843 | 23.5 | `698a29f5` |
| 1024 | both | **9.46 / 14.30** | **74.9** | 68.19 | 17.6 | 2837 | 23.5 | `698a29f5` |
| 2048 | base | 6.80 / 11.50 | 92.7 | 76.25 | 33.3 | 2718 | 32.9 | `cb78bf61` |
| 2048 | both | 6.77 / **12.28** | **87.7** | 71.38 | 27.8 | 2747 | 25.2 | `cb78bf61` |

Generation stops at `<turn|>`: 107 tokens at p1024, 102 at p2048.

## Reading

- **Gates.**
  - The text is identical in every cell of a prompt.
  - DENSE_EARLY leaves misses/token exactly unchanged (73.18).
  - LRFU lowers misses/token.
  - Peak RSS is within ±6 MiB.
- **DENSE_EARLY saves 6.8 ms a token at p1024 (−8.3 %).**
  - The dense branch (≈ 7.5 ms on the per-kind line) now runs inside the miss wait.
  - The wait falls 24.6 → 17.3 ms, not to zero, because the reads still take longer than the
    dense branch in most rounds.
- **LRFU cuts misses by 6.8 % at p1024 and 6.4 % at p2048, not the 20 % the replay predicted.**
  - The replay ran 512 decode tokens from an empty pool. These cells stop after
    102–107 tokens, and the app's LRU starts from the prefill's experts (73.18 misses/token
    here against the replay's 83.7).
  - Alone, LRFU saves 2.1 ms at p1024.
  - On top of DENSE_EARLY it saves nothing measurable at p1024 (74.9 = 74.9 ms): the number of rounds
    does not change (2843 → 2837), and the per-round latency plus the dense work
    now set the wait.
  - At p2048 the two levers together save 5.0 ms.
- **Not measured here.**
  - LRFU at a longer generation (G ≥ 512 real tokens), where the replay's gap should show.
  - Repeats for noise.
  - LRFU alone at p2048.
