# Record sitting 2026-09-30: cool-start control (A only) for the row of record

Purpose: the mirrored cool control sitting that LEDGER rule 52 asked for —
the G=1024 ≥ 50 question was temperature-bound after cycle 23 (cool A cells
50.9–52.4, hot 45.9–49.1, the row of record #158 B's 47.36 read as the lever
of a warm A/B). No lever, no variant: **A twice per G, cool start, nothing
set.** Run by the orchestrator on the workstation (agents never run adb).

## Method

| item | value |
|---|---|
| unit | `R3CY10WM83Y` (SM-S938N, Galaxy S25 Ultra) |
| when | 2026-09-30 02:48:21 – 02:53:32 KST |
| code | `htp_moe` @ `90d88e2b` (the head after PR #176; the committed default: `applied=0x703e1 … feed=vtcm dma_bypass=1 source=default`, `dspq: on` once per log) |
| set | device dir `s170r2q`, the #170 S3 staging (same binaries as `170-attn-m1-round2.md`'s A; the round-2 kernel is off by default and not run by A) |
| env | `NNTR_NUM_THREADS=8`, nothing else (dspq default, `dma_bypass=1` default) |
| prompt / gen | 512 tokens / G = 64, 512, 1024; two runs per G back to back |
| cool start | each G started at zone0 ≤ 35 °C: **34.7 / 34.3 / 34.3 °C** (a cool-down between the G blocks) |
| binary | non-profile (`NNTR_HTP_PROFILE` never set, rule 15) |
| logs | `/local/mnt/workspace/htp_moe/record-0930/` (`sitting.out`, `A_G{64,512,1024}_r{1,2}.log`) |

## md5 (device, `sitting.out`)

| file | md5 |
|---|---|
| `libnntr_hvx_skel.so` | `4f78badb45ccc2e40d17899d9cf614f6` |
| `nntrainer_causallm` | `bad89163ce46c4ab5cd38f7a14c91e69` |
| `libnntrainer.so` | `ad0ecbcb82119e031a17770d7dc02e3d` |

Every log prints exactly one `[HTP] moe m1 gemv: on (applied=0x703e1)
lead=192KB rows1=1 feed=vtcm dma_bypass=1 source=default` and one
`[HTP] dspq: on … buffers=2x65536 ion=y`; `dspq: close calls=N served=N
bad=0` with N = 22 × G (1408 / 11264 / 22528) in all six logs (rule 40).

## Results (A, `q40-qs4cx-wh`)

| G | run | prefill tok/s | decode tok/s (all) | decode tok/s (last 64) | zone0 after (°C) | text |
|---|---|---|---|---|---|---|
| 64 | r1 | 573.99 | 53.51 | 53.51 | 59.0 | r1 = r2 |
| 64 | r2 | 482.11 | 54.42 | 54.42 | 55.2 | |
| 512 | r1 | 542.37 | 54.75 | 52.07 | 62.1 | r1 = r2 |
| 512 | r2 | 526.21 | 49.56 | 38.14 | 62.9 | |
| 1024 | r1 | 579.84 | 51.54 | 50.59 | 64.5 | r1 = r2 |
| 1024 | r2 | 425.96 | 51.29 | 49.27 | 65.6 | |

Means (two runs): decode **53.97 / 52.16 / 51.41** at G 64 / 512 / 1024;
prefill 528.1 / 534.3 / 502.9. Against the previous row of record (#158 B,
warm mirrored A/B, 51.82 / 50.61 / 47.36): +4.1 / +3.1 / +8.6 %. Against
cycle 23's cool A band (52–55 / 51–54 / 50–52): inside it at every G.

Spread inside the sitting: G=64 0.91, G=512 5.19 (r2's last-64 window read
38.1 tok/s — the same kind of transient as #178 set3 r1's 28.9; its
whole-generation 49.56 is kept, not excluded), G=1024 0.25 tok/s.

## Thermal

Cool start per G block (34.7 / 34.3 / 34.3 °C), 55–66 °C after each run:
the ramp is inside each run and the second run of a G starts warm (the
prefill of r2 shows it: 482 / 526 / 426 vs r1's 574 / 542 / 580). The G=1024
pair reads 51.54 / 51.29 with the phone at 64.5–65.6 °C after each run, so
G=1024 ≥ 50 holds on a cool start even though the run itself ends hot; the
warm-start reading of the same default (#158 B's 47.36, #164's 45.86,
#170 S2's 48.26, #162 (b)'s 49.11) stays below 50 — rule 52's condition is
now the stated one, not a pending one.

## Verdict

The row of record (BENCHMARK "now") moves to **53.97 / 52.16 / 51.41
(cool start, zone0 ≤ 35 °C per G block)**; #158 B's 51.82 / 50.61 / 47.36
is kept as the previous "now" and as the warm-start reading. Decode ≥ 50 at
all three G on a cool start; G=1024 warm still 47–49 (rule 52). Text r1 =
r2 at every G; no CPU cell (the CPU row of record stays #94 s2 on
`R3CY205ZMND`). Not a lever cell: nothing to fold into a design verdict.
