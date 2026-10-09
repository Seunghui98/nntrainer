# Measurement 267 sitting 1: L0 (OP_TIME split) and L1 (heap-image bypass), Gemma-4 26B QS4CX, S25 Ultra

Issue #267, plan `docs/plans/267-decode-dma-feed.md` (PR #269) §6 S0 / S1. Branch
`htp/267-dma-bypass`. A = the E rows of the 260 r2 sitting
(`260-step2-r2-3897963d8.md` on PR #265's branch, same unit, same configs, build `3897963d8`).
They are not re-run here (coordinator, 2026-10-09: minimum sitting).

## Artifacts

| item | value |
|---|---|
| B (L0) | `f11f8d994`; skel v79 `d8784376ac45926b5fa88436c0843fca`, v81 `920d882e6cd1fd08889b37b74528e3a0` |
| D (L0 + L1) | `60d4d241f` (L1 `f3d886f58` + review fix); skel v79 `f73608cd86829db5245a7cafb34e16d4`, v81 `8e681ca19935b9f4951f05ba9ae76ab0` |
| app (B, C, D; L1 is DSP-only) | `nntrainer_causallm` `d6a21df0ac746203266173b58bc7fb6c`, `libcausallm_core.so` `ce60d34c98683b91cc7b1c16fdd9cf1d`, `libnntrainer.so` `c41f2e9465e548be9f0ad5880f656912`, `libccapi-nntrainer.so` `dd686c57aa7541abd94bff707fbec260` (NEEDED `libsdkl.so`, `libcdsprpc.so`; `NNTR_HTP_FORWARD_KINDS` strings = 2) |
| A (260 r2 E) | `s260r2/`: app `5f77c1ed…`, skel `4c665097…` |
| device dirs | `/data/local/tmp/nntrainer/causallm/s267b/` (B, C), `s267d/` (D); device `md5sum` == local for every file. `libc++_shared.so`, `libsdkl.so`, `unittest_hvx_two_sessions` copied from `s260r2/` |
| configs | the 260 r2 `s260cfg/r2_E_p*_g*` (C 16, fp16 KV, `lmhead_engine cpu`) |
| device | S25 Ultra `R3CY205ZMND` (v79), lock `r267@…` from 2026-10-09 10:28 KST |
| runner | `267-run.sh one <log> <variant> <p> <G>` (260-e-run.sh's `cell`, sourced) |

**Cool start (relaxed, coordinator 2026-10-09):** no other run, then battery ≤ 32.0 °C or a
5 minute cap (the 260 sitting waited 180 s plus cpu/nsp ≤ 38, battery ≤ 30.0, zone0 ≤ 35,
20 min cap). Start temperatures are in the table.

## Host / build rungs

| rung | result |
|---|---|
| 1 `run_host_checks.sh` | `ALL CHECKS PASS`, `WORKER POOL LANES OK`; `TOKEN DRIVER BIT-IDENTICAL … kind_ns/wall=0.999`, `TOKEN POOL BIT-IDENTICAL … kind_ns/wall=0.999`; `M=37 clean heap : … bypassed 12288 B (want 12288) bit-identical`, `MOE HMX DMA BYPASS OK (weights only, arena slots and flushed heap images only, bit-identical)` |
| 1 `run_inproc_e2e.sh` | `INPROC E2E PASS` (`E2E eval self-test ok`, `e3==e1-hd64 bit_identical=1`, `dspq-* bit_identical=1`) |
| 1 `htp_syntax_check.sh` | exit 0 |
| 2 skel v79 / v81 (B and D) | `UNDEFINED SYMBOLS OK (68 runtime imports)`, `ARCH OK (V79)` / `ARCH OK (V81)` |
| 3 app | built (`build_android.sh --htp`); device gtests built, not run (minimum sitting) |

## Results (C 16, KV fp16)

Per-kind wall ms are the new `[HTP] graph per-kind us/token` line (QTimer, any clock);
pcycle ms are the old line at the run's mean `mhz`.

| cell | start soc / bat / zone0 | prefill tok/s (ms) | decode tok/s (last 64) | tokens | text == A | misses/token | miss_wait ms/token | mhz / compute_mhz | FC ms wall (pcyc) | DENSE_FFN ms wall (pcyc) | LM_HEAD wall | ATTN_M1 wall | ROUTER wall | MOE wall (net of wait) | ops/wall | mapped MiB | heap_used_kib | peak RSS KiB | S1 ceiling after |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A = 260 E p512 G512 | 32.5 / 28.7 / 30.6 | 143.8 (3561) | 3.33 (3.68) | 169, `<eos>` | ref | 94.96 | 208.5 | 1180 / – | – (34.48) | – (17.58) | – | – | – | – (188.3 incl. wait) | – | 449.12 | 1395553 | 2,894,412 | 3840 |
| **D p512 G512** | 38.0 / 32.3 / 34.5 | **157.0** (3262) | **3.36** (3.86) | 169, `<eos>` | **identical** | 94.96 | 215.4 | 1137 / **2112** | **12.19** (22.64) | **6.30** (11.71) | 8.27 | 12.09 | 3.18 | 239.46 (**24.11**) | 0.9988 | 449.12 | 1395573 | 2,899,312 | **3584** |
| **B p512 G512** (+ `NNTR_HTP_DUMP`: prefill speed void) | 36.8 / 31.9 / 34.1 | (129.0, dump I/O) | **3.38** (3.77) | 169, `<eos>` | **identical** | 94.96 | 203.7 | 1187 / **2112** | **19.61** (34.89) | **9.86** (17.55) | 8.28 | 12.04 | 3.14 | 227.66 (**24.00**) | 0.9988 | 449.12 | 1395557 | 2,901,372 | 3584 |
| **C p512 G512** (B + `NNTR_HTP_E2E_SPIN_US=12000`) | 36.8 / 32.0 / 34.1 | 152.8 (3350) | 3.30 (3.66) | 169, `<eos>` | **identical** | 94.96 | 210.2 | 2001 / **2112** | 19.69 (20.78) | 9.87 (10.42) | 8.28 | 12.07 | 3.17 | 234.41 (24.19) | 0.9988 | 449.12 | 1395557 | 2,898,840 | 3584 |
