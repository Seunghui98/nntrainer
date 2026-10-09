# Measurement 261 PV single walk: ATTN_M1 PV in one V walk per row group, 2-bit Gemma-4 26B, S25 Ultra

Issue #261. This is the bit-identical candidate 1 that `docs/measurements/261-softcap-attn.md`
("Next") left open. Branch `htp/261-pv-single-walk` = `htp_decode` @ `e20e22b24` + `760bcaf30`
(kernel only). No IDL or app change.

## Summary

- **ATTN_M1 drops by about 40 %, and the ids do not change.** The per-kind µs line (qtimer wall at 2112 MHz):
  - p1024: **20 576 → 12 229 µs/token (−8.35 ms)**
  - p512: **15 151 → 9 147 µs/token (−6.00 ms)**
  - The pcycle line agrees: 25.07 → 15.54 ms and 17.98 → 11.25 ms.
- **Steady decode** (last 64): 11.41 → **12.40 tok/s** at p1024 and 11.53 → **12.41** at p512.
  Token wall: 95.5 → 91.8 ms and 83.6 → 81.0 ms.
- The token gains less than ATTN_M1 saves, because the E cells' MoE miss wait read higher
  (p1024 29.0 → 32.9 ms, p512 22.4 → 25.6 ms). Those cells started hot, and another agent's
  cells ran on the phone between them (see Conditions). Attention cannot cause that wait.
- Against 261's expected range (0 … −5.3 ms at p512, 0 … −10.9 ms at p1024), p1024 lands
  inside it. p512 is past the ceiling, because that ceiling was derived on the 4-bit file with
  fp16 KV and C 16. The re-walk was most of PV's stall; it was not the DDR stream itself.

## What changed

`pv_group` walked the window's V rows once per 64-wide chunk. A sliding row (hd 256) was
walked 4 times per head pair, and a full row (hd 512) 8 times per group of 4 heads.
`pv_walk` now keeps up to 8 (head, chunk) chains in registers and takes all of them from one
load of each V row:

- sliding (gqa 2): one walk instead of 4;
- full (gqa 8 in groups of 4): 4 walks instead of 8.

LFM (hd 64), odd gqa, and an nch that `PV_CG(ng)` does not divide still walk once per chunk,
as before (a `ponytail:` note in the code covers this).

Each (head, d) chain is the same fp16 fma sequence in p order as `attn_m1_det.h`. Two
codegen guards keep it that way on silicon:

- **No software pipelining on the position loop.** With 8 chains, hexagon-clang 19's
  pipeliner carried a product across the back edge as sf and back (`vadd(sf, 0)`). That is
  the sequence S1 never measured.
- **One noinline function per (ng, cg) shape.** When the shapes were inlined together into
  `run_p3`, the 8-chain loops spilled their accumulators to the stack, and which loop
  spilled moved with unrelated edits.

The objdump of both skels shows no `.sf` operand and no stack reference in any `pv_walk_*`
position loop.

## Host / build rungs

| rung | result |
|---|---|
| 1 attn_m1 host check (the `run_host_checks.sh` section: spec, real kernel on `hvx_emu`, pool 0/3/7) | `ATTN M1 BIT-IDENTICAL`, `ATTN M1 GEMMA BIT-IDENTICAL: hd256 window 1024 and hd512 full, scale 1.0, L = 1 .. 4096, lowest SNR vs f64 35.5 dB`, `ATTN M1 PHASES OK`; both kernel mutants `CAUGHT` (3 Gemma lines each) |
| 1 rest (`run_inproc_e2e.sh`, gtests, full `run_host_checks.sh`) | not run (coordinator: kernel spec check only) |
| 2 skel v79 | `UNDEFINED SYMBOLS OK (68 runtime imports)`, `ARCH OK (V79)`, md5 `ea3673705e4db9e9e8287bdf447438b9` (the device file) |
| 2 skel v81 | `UNDEFINED SYMBOLS OK (68 runtime imports)`, `ARCH OK (V81)`, md5 `7faebfd0d9e8ec30e1500a5dcb78a5e1` |
| 3 app | built (`build_android.sh --htp`): NEEDED `libsdkl.so` / `libcdsprpc.so`, `NNTR_HTP_FORWARD_KINDS` strings = 2; `nntrainer_causallm` `d10095a7…`, `libcausallm_core.so` `3d27e258…`, `libnntrainer.so` `f115905e…`, `libccapi-nntrainer.so` `92575f61…`. Not pushed: the device cells use the `s282d` app (same source), so only the skel differs between A and E |

Skel md5s change from one build to the next, so a md5 identifies a build, not its source.

## Device cells

Setup:

- S25 Ultra `R3CY205ZMND` (v79), 2026-10-09 19:33–19:39 KST.
- File `gemma4_26b_ternary_fcqs4cx` (2-bit experts, QS4CX FCs), q8 KV, C 32.
- Env `NNTR_HTP_E2E=1 NNTR_HTP_DROP_HOST_FC=1`.
- Runner: `docs/measurements/282-run.sh one <L> <bin> /data/local/tmp/nntrainer/s261ccfg E <p> 512 32 <tag> NNTR_HTP_DROP_HOST_FC=1`.

Binaries:

- **A** = `causallm/s282d`, the base build (`6a3ba09b1`, identical to `e20e22b24` in `nntrainer/`, `Applications/` and `test/`); skel `328843bf…`.
- **E** = `causallm/s261c`, a device-side copy of `s282d` with only `libnntr_hvx_skel.so` replaced (`ea367370…`).

Configs:

- `s261ccfg/r2_E_p1024_g512` is a byte copy of `s282cfg/r2_E_p1024_g512` (md5 `57f00c82…`).
- `s261ccfg/r2_E_p512_g512` (md5 `88a46496…`) is the same config with the 512-token prompt from `s260cfg/r2_E_p512_g512` and init_seq_len 1024 (prompt + G).

| cell | bin | start soc / bat / zone0 °C | prefill tok/s | decode tok/s (all / last 64) | **ATTN_M1 µs/token** (us line) | ATTN_M1 pcyc line | wall ms/token | misses/token | miss wait ms/token | peak RSS KiB | ids |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A p1024 G512 (282 B `E_p1024_g512_Ad_c32`) | s282d | 31.8 / 25.9 / 28.7 | 367.4 | 7.86 / 11.41 | 20 576 | 25.067 ms (mhz 1734) | 95.46 | 49.85 | 29.0 | 3 247 424 | `5d73a95e` (107 → `<turn\|>`) |
| **E p1024 G512** | s261c | 49.2 / 27.9 / 43.4 | 359.9 | **8.01 / 12.40** | **12 229** | 15.541 ms (mhz 1662) | 91.75 | 49.85 | 32.9 | 3 242 880 | **`5d73a95e`, raw stream identical** |
| A p512 G512 | s282d | 66.7 / 29.5 / 56.2 | 302.1 | 10.83 / 11.53 | 15 151 | 17.979 ms (mhz 1780) | 83.59 | 43.04 | 22.4 | 3 249 576 | `3625a355` (512 tokens, no `<turn\|>`) |
| **E p512 G512** | s261c | 105.0 / 30.4 / 65.9 | 283.0 | **11.17 / 12.41** | **9 147** | 11.245 ms (mhz 1718) | 80.97 | 43.04 | 25.6 | 3 249 480 | **`3625a355`, raw stream identical** |

How the ids column was checked:

- The text md5 is `260-turn106.py <tokenizer> <log> | tail -n +2 | md5sum`, the recipe behind
  282's `5d73a95e`.
- "Raw stream identical" means the app's printed generation, compared byte for byte between
  the A and E logs with the `[HTP]` lines removed. At p512 the first-token `[HTP]` line is
  interleaved at different points in the two logs; the text itself is identical.

## Conditions and caveats

- **The phone was shared.** The #282-d agent ran cells from `causallm/s282g` between ours
  (one ran up to 19:35, and a p4096 G64 cell started at 19:36:55). Every cell of ours started
  only after `ps -A` showed no `nntrainer_causallm` (the runner's `cool()`). Theirs
  appear to wait the same way: their p4096 cell started the second our p512 A cell ended.
  Overlap was not monitored during a cell, so it is excluded only that far. The
  `S1_CEILING` probe right after each of our cells read 3 584 MiB instead of 3 840, at least
  once while the other agent's model was loading (19:36:55). It is not read as a leak of
  this build.
- **Start temperatures are hot** (`COOL_QUICK` only gates the battery at 32 °C), up to 105 °C
  on the hottest cpu/nsp zone for E p512. The ATTN_M1 delta comes from per-kind lines inside
  the same E2E token and is far larger than the drift between cells: 282 B's five p1024
  cells on `s282d` read 24.86–25.21 ms on the pcycle line, and 26.24 at C 16. The tok/s
  and miss-wait columns are more exposed to the heat and the other agent's page cache.
- No `NNTR_HTP_PROFILE` cell was run, so PV's share after the change is not split from
  scores and softmax. One profiled cell would read it.

## Not run

- `run_inproc_e2e.sh`, the gtests and the full `run_host_checks.sh` (the coordinator scoped
  this change to the kernel's own spec check).
- v81 / S26 cells, and G64 / G1024.
- An A cell repeated in the same thermal state as the E cells.

Logs are in the session scratchpad (`r261c/L/`); the A p1024 log is in `r276/P/`.
