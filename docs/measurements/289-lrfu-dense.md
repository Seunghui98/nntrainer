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

## 512 real tokens (EOS off)

These cells use a staged copy of the configs, `s289cfg/r2_E_p*_g512`. Its
`generation_config.json` has `eos_token_id: [999999]`, so generation runs to G = 512. The user's
files are not edited.

After `<turn|>` at ≈ 107 the model writes a second `<turn|>` and then copies the source text
back. That tail is not judged as text (the text gate's window); it is there for speed only.
The whole text is identical between base and both in each prompt.

| prompt | cell | **decode avg tok/s** | first token ms | rest avg ms | last 64 tok/s | misses/token | miss wait ms | text md5 |
|---|---|---|---|---|---|---|---|---|
| 1024 | base | 10.48 | 3 082 | 89.6 | 11.42 | 85.83 | 26.8 | `4257c8ec` |
| 1024 | both | **11.81 (+12.7 %)** | 3 287 | **78.4** | 13.10 | **73.62 (−14 %)** | **15.4** | `4257c8ec` |
| 2048 | base | 9.59 | 5 691 | 93.3 | 10.07 | 77.71 | 26.7 | `b667afcb` |
| 2048 | both | **10.50 (+9.5 %)** | 5 659 | **84.4** | 10.86 | 75.29 (−3 %) | **17.6** | `b667afcb` |

- Over 512 tokens, LRFU lowers misses by 14 % at p1024, against 6.8 % over 107 tokens. That is
  closer to the replay's −20 %. At p2048 the gain is −3 %; the copied source text reuses experts
  differently there.
- The first token costs 3.1 s at p1024 and 5.7 s at p2048. Of that, the KV seed is 2.0 / 4.0 s.
  These are 6–11 % of the 512-token time.

## First token: the KV seed over the pools (`attn_m1` + `mha_core`)

The seed's DSP amax and append now run over the session's worker pool. The ARM's fp16 → f32
rows run over ThreadManager. The per-element code is unchanged.

Install `causallm/s289s`:

- skel `86f18820`
- `libcausallm_core.so` `76cc54fc`

The cells are the same 512-token EOS-off cells, both levers on, C 24. Host:
`ATTN M1 BIT-IDENTICAL` and `ATTN M1 GEMMA BIT-IDENTICAL`, workers {0, 3, 7}.

| prompt | build | **decode avg tok/s** | first token ms | kv seed rpc ms | ARM convert ms | rest avg ms | text md5 |
|---|---|---|---|---|---|---|---|
| 1024 | before | 11.81 | 3 287 | 2 038 | ≈ 510 | 78.4 | `4257c8ec` |
| 1024 | **pool seed** | **12.04** | **1 529** | **659** | **160** | 80.3 | `4257c8ec` |
| 2048 | before | 10.50 | 5 659 | 4 041 | ≈ 1 000 | 84.4 | `b667afcb` |
| 2048 | **pool seed** | **10.83** | **2 689** | **1 558** | **294** | 87.2 | `b667afcb` |

- The first token is 1.8 s shorter at p1024 and 3.0 s shorter at p2048. The text is identical.
- The rest-of-tokens average moved by +1.9 / +2.8 ms between sittings. That is
  run-to-run noise: the seed does not touch the per-token path.
- What is left of the seed (0.66 / 1.56 s) is the f32 transfer, about 16 MiB a sliding layer at
  p1024, plus the DSP's strided tile writes. The next step there is an fp16 seed: half the bytes,
  no ARM conversion, an IDL change.
