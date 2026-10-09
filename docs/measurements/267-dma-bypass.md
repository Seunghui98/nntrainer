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
| A = 260 E p1024 G512 | 34.5 / 30.0 / 32.1 | 191.1 (5359) | 2.58 (2.42) | 512 | ref | 120.81 | 286.9 | 1133 / – | – (37.02) | – (18.50) | – | – | – | – (253.8 incl. wait) | – | 449.12 | 1424737 | 2,940,116 | 3584 |
| **D p1024 G512** | 36.0 / 32.0 / 33.7 | **183.8** (5572), 0.96 × A | **2.65** (2.37) | 512 | **identical** | 120.81 | 288.5 | 1101 / **2112** | **12.26** (23.52) | **6.35** (12.18) | 8.27 | 22.14 | 3.14 | 313.21 (**24.69**) | 0.9991 | 449.12 | 1424757 | 2,941,168 | 3584 |

Start temperatures: all cells started at a hottest cpu/nsp zone of 36.0–38.0 °C and zone0
33.7–36.0 °C, with battery 31.9–33.6 °C. The PPL cell reached the 5 minute cap at a battery
temperature of 33.6 °C.

**S1 ceiling.** It read 3584 MiB after every cell here. A's own p1024 cell in the 260 sitting
read 3584 as well, and B, which has no L1, reads 3584 too. So the reduced ceiling comes from
the phone, not from L1. `mapped_mib` is unchanged at 449.12 in every cell.

## nll and prefill bit identity

| cell | prompt nll/token | prefill MoE dumps |
|---|---|---|
| A = 260 E p512 G64 PPL | 4.57345 | – |
| **D p512 G64 PPL** (`NNTR_PPL=1 NNTR_HTP_DUMP`) | **4.57345** (equal to the digit) | **B == D `bit_identical=1`**: 124 files (`manifest.txt` + every `moe_*_{in,out}.f32`), md5 of the md5 list `77c0b8e63ece56d452cbb6775ce4185d` in both. B's dump comes from its p512 G512 cell, D's from this cell. The prefill MoE calls are the same in both, because PPL only scores the prompt rows through lm_head |

The PPL cell's prefill speed (66.0 tok/s) includes lm_head over every row, so it is not a
speed number. The p1024 PPL cell was not run (minimum sitting).

## Gates (plan 267 §1, against A = 260 r2 E)

| gate | result |
|---|---|
| ids identical | **pass**: D p512 G512 has the same 169 tokens ending in `<eos>`, D p1024 G512 the same 512 tokens. B and C p512 G512 are identical too (generated text byte-equal, so the ids are equal) |
| nll equal | **pass** at p512: 4.57345 == 4.57345. p1024 was not run |
| prefill ≥ 0.95 × A | **pass**: p512 157.0 vs 143.8 (1.09) and p1024 183.8 vs 191.1 (0.96). These compare one run against another sitting's run; the 260 sitting's own E spread was 144–158 / 180–191 |
| prefill MoE dumps | **pass**: B == D `bit_identical=1` at p512 |
| misses/token unchanged | **pass**: 94.96 / 120.81 in every cell |
| memory | **pass**: `mapped_mib` 449.12 (=). `heap_used_kib` +20 KiB (+4 KiB of `op_qt` in the graph struct, plus allocator rounding); peak RSS +2–5 MiB |
| L0 coverage (`ops/wall`) | 0.9988–0.9991: the per-kind wall sums to the token's DSP wall within 0.12 % (the plan asked for ±1 %) |

## Reading

* **OP_TIME split (L0)** for the 260 E, from B p512 G512:
  * The token's DSP wall is 284.5 ms. The miss wait is 203.7 ms of it.
  * **MOE net of the wait is 24.0 ms**. The plan's estimate was 20–30.
  * **compute_mhz is 2112 in every cell.** Outside the wait the core runs at its full
    clock. The 1180 MHz in the pcycle line is the sleeping wait diluting the average.
  * Wall per kind, non-MoE: FC 19.61, DENSE_FFN 9.86, LM_HEAD 8.28, ATTN_M1 12.04 (p512),
    ROUTER 3.14, norms + rope + add 3.52. That adds to 56.4 ms. With the MoE net, the DSP's
    own token is ≈ 80.5 ms.
* **L1** (D against B, same clock, p512):
  * FC wall 19.61 → **12.19 ms** (−7.42) and DENSE_FFN 9.86 → **6.30 ms** (−3.56), together
    **−11.0 ms/token**. At p1024: 12.26 / 6.35.
  * At 557.8 / 269.3 MB per token this is 45.8 / 42.7 GB/s, up from 28.4 / 27.3. P4 read
    10.65 / 5.33 ms on the arena sidecar.
  * D's FC of 12.2 ms meets the plan's "≤ 12 ms ⇒ default" within 2 %. It stays 1.5 ms above
    P4. One untested candidate for the remainder: the heap images are `malloc`-aligned, while
    P4's sidecar sits on 4 KiB arena offsets.
  * Decode tok/s does not move (3.36 vs 3.38 vs A 3.33) because the 204–215 ms miss wait is
    72 % of the token. L1's 11 ms shows up once #266 cuts the wait.
* **L2 (spin, C)** is **not a lever**. The mean clock rises from 1187 to 2001 MHz, but
  compute_mhz (2112) and every non-MoE wall time equal B's.
* **L4 trigger**: not read. The route log was not split into miss-free layers in this
  sitting.

## Not run (coordinator's minimum sitting, 2026-10-09)

The following were skipped, so the matching rows of the plan's §6 S0 grid are open:
* A re-run (the 260 r2 E rows stand in for A);
* G64 / G1024 speed cells, C at p1024, B at p1024;
* the p1024 PPL cell;
* the device gtests (`unittest_hvx_mm_u8i4`, `unittest_hvx_fc`: built, md5 `73d44942…` /
  `16d27a6d…`, not run).

Logs are in the session scratchpad (`r267/logs/`), not in the repo.
