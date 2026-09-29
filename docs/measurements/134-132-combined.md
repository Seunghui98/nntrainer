# Measurement 134 + 132: decode PPL and ADD + ROUTER_TOPK resident, one sitting

Code: `htp_moe` @ `bb845426` (PR #144 = #134 and PR #145 = #132 PR 1 merged).
The staged set was built from the local merge `dev/sitting-134-132` @ `5cf77ebe`,
whose code under `nntrainer/`, `Applications/`, `test/htp/*.c` and the IDL is
byte-equal to `bb845426`. Estimated device time: **≈ 25 min**. Run by the
orchestrator on the workstation (user's request, 2026-09-28), unit recorded
under Results.

## Why

1. #134: the first device reading of the decode-side PPL (`NNTR_PPL_DECODE`),
   the accuracy gate of the per-token entry since #136 showed text identity
   cannot hold for any resident kind (LEDGER rule 39).
2. #132 PR 1: ADD and ROUTER_TOPK resident, 95 → 51 calls/token; expected
   D − B ≈ −3.5 ms/token (≈ +10 % decode), D still below A (51 > 22 calls and
   the ATTN_M1 term, #146).

Accuracy (contract §1, 2026-09-28): decode PPL of every variant forced on A's
own G = 512 continuation, **reference = A of this sitting**, fail above +2 %;
**text compared** against A at every cell, and the user's text approval.

### Variants (4, contract §4.2; one binary set, environment only)

| | env (beyond `NNTR_NUM_THREADS=8`) | expected banner / calls |
|---|---|---|
| **A** (reference, first) | none | no `graph:` line |
| **C** | `NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=MOE` | `resident=MOE`, 22.00 |
| **B** | `NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=MOE,RMSNORM,QK_NORM,ROPE,CONV1D_GATE,ATTN_M1` | six kinds, 95.00 |
| **D** | B's mask `,ADD,ROUTER_TOPK` | `resident=RMSNORM\|CONV1D_GATE\|QK_NORM\|ROPE\|ATTN_M1\|ADD\|ROUTER_TOPK\|MOE`, 51.00 |

Not run: an A0 on the pre-#143 tree (the pool fix's own prefill gate). PR
#145 changed the IDL, so A0 needs a separate `c5dcb382` set and would be a
fifth variant; **the user accepted #143 without it (2026-09-28)**.

## Artifacts (`/local/mnt/workspace/htp_moe/134-132/set/`, `md5.txt` next to it)

| file | md5 |
|---|---|
| `libnntr_hvx_skel.so` | `0c2d5b00f1e947fbfdb1eeaebd2d40a1` (`UNDEFINED SYMBOLS OK (46 runtime imports)`) |
| `nntrainer_causallm` | `b6adb4f55a849107c635770d5e2d5fcb` |
| `libcausallm_core.so` | `59e9be42d533bd6762e8ee93bff4846e` (`NNTR_HTP_FORWARD_KINDS` 2, `NNTR_PPL_DECODE` exact 1) |
| `libnntrainer.so` | `bd5abd809202adb742a35563230bc5bc` |
| `libccapi-nntrainer.so` | `06a3072b13937becc7fce9561ef172e3` |
| `unittest_hvx_softmax` | `25bec7dfd9f5b342783ec59cc7d5059d` (PR #145's, carries `HvxM1Ops.RouterTopkMatchesDetBitExact`) |
| `libc++_shared.so` / `libsdkl.so` / `prompt512.txt` | `b1586b9b…` / `0ad4e22a…` / `fc65c158…` |
| model `q40-qs4cx-wh`, `tokenizer.json` | on the phone since #100 (`tokenizer.json` `7b8067a5…`) |

## Steps

Clean run dir `/data/local/tmp/nntrainer/causallm/s134` (the old dir holds a
foreign `libc++_shared.so`); every `adb` names the serial (a second device may
be on USB). Config as #136: `do_sample: false`, `bad_word_ids: [124900]`,
`init_seq_len: 512`, `moe_engine: htp`, `moe_htp_layers: ""`.

1. Push the set, `md5sum` on the device = the table.
2. `unittest_hvx_softmax --gtest_filter='HvxM1Ops.*'` (router rows `bad=0`;
   the kind=2 rmsnorm row is #137's known subnormal case).
3. tok/s: for G in 64, 512: `A C B D` run 1, then `D B C A` run 2.
4. PPL at G = 512: `rm -f cont.ids`; A with `NNTR_PPL_DECODE=cont.ids` (writes
   A's continuation, `source=self`); A again (null check, must equal); C
   (must equal A); B; D.
5. Text: every C / B / D cell against A of the same G and run; paste A's, B's
   and D's G = 64 run-1 texts below for approval.

## Results (fill in)

Unit `R3CY10WM83Y` (SM-S938N), 2026-09-28 20:11–20:22 KST, run by the
orchestrator. Start cool (battery 26.9 °C, zone0 28.5 °C), warm by the end
(34.0 °C / 58.3 °C). Device `md5sum` = the table (MD5 OK). Every C / B / D log
printed its `graph: init` banner and `calls/token` 22.00 / 95.00 / 51.00.

### tok/s (mirrored: A C B D run 1, D B C A run 2)

| variant | G | prefill r1 / r2 | decode r1 / r2 | decode mean | vs A |
|---|---|---|---|---|---|
| A | 64 | 564.5 / 536.1 | 36.87 / 36.93 | **36.90** | — |
| C | 64 | 567.6 / 541.2 | 36.80 / 36.61 | 36.71 | −0.5 % |
| B | 64 | 568.3 / 562.0 | 26.96 / 26.78 | 26.87 | −27.2 % |
| D | 64 | 549.4 / 554.1 | 28.33 / 28.89 | **28.61** | −22.5 % |
| A | 512 | 540.7 / 418.6 | 36.60 / 36.30 | **36.45** | — |
| C | 512 | 504.9 / 422.4 | 36.45 / 35.62 | 36.03 | −1.2 % |
| B | 512 | 458.4 / 456.3 | 25.68 / 25.59 | 25.63 | −29.7 % |
| D | 512 | 457.6 / 421.1 | 28.09 / 28.23 | **28.16** | −22.7 % |

**D vs B: +6.5 % (G=64), +9.9 % (G=512)** — the 44 fewer calls, as #132's
plan estimated (≈ +10 %). Prefill falls through the sitting with the phone's
temperature in every variant (mirrored pairs agree), so no prefill verdict
beyond "no variant sits outside the mirrored band".

### Decode PPL (G = 512, forced on A's own continuation, 512 tokens)

| variant | nll/token | ppl | top1 | Δ vs A (paired, 2·SE) | gate (≤ +2 %) |
|---|---|---|---|---|---|
| A self | 0.216411 | 1.24161 | 512/512 | (reference) | — |
| A forced | 0.216411 | 1.24161 | 512/512 | 0 (nll_sum equal to 17 digits) | null check **ok** |
| C | 0.216411 | 1.24161 | 512/512 | 0 (equal to 17 digits) | null check **ok** |
| B | 0.215527 | 1.24052 | 506/512 | −0.088 % (±0.35 %) | **pass** |
| D | 0.214190 | 1.23886 | 509/512 | −0.222 % (±0.37 %) | **pass** |

### Text (every cell against A of the same G and run)

C = A in all 4 cells. B ≠ A and D ≠ A in all 4 cells each; both leave A at
the first generated word (`town` → `final`) and settle into a repeating
sentence (texts below).

### Ride-along: router gtest (`HvxM1Ops.*`)

`router_topk` K=2048 E=32 / K=128 E=4 / K=64 E=4: **bad_logits=0 bad_sel=0
bad_weight=0** — bit-exact on silicon. The other five failures are #137's
known rows (rmsnorm kind=2 subnormal, qk_norm 64, rope64 10–18, conv_gate 229,
RejectsBadShapes `0x80000600`), unchanged from #130.

### Ride-along: #141 dspqueue microbench (PR #147, own dir and skel)

| row | median µs | p90 µs |
|---|---|---|
| F0 (bare FastRPC invoke) | 37.2 | 38.0 |
| F12 (FastRPC, 12 KiB ION) | **87.1** | 88.8 |
| QSS12 (queue, both sides spin, 12 KiB) | **16.7** | 17.0 |
| QBS12 (DSP blocks, ARM spins) | 28.9 | 29.8 |
| QBB12 (both block) | 76.9 | 90.4 |
| QSS0 / QBS0 (message only) | 3.7 / 19.5 | 3.8 / 20.2 |
| F12p (F12 with the spinner parked) | 91.1 | 100.5 |

`verdict delta_us=70.4 s95_ms=6.68 s51_ms=3.59 rule=adopt`; all rows `bad=0`,
`served=4400 arm_answered=4400` in both modes. Even with the DSP thread
blocking (QBS12) the gap is 58 µs, above the 50 µs adopt line.

### Ride-along: #146 attention gtest (PR #148, own dir, skels prof → o1 → o13 → prof)

| skel | bad / bad_stats at L 1…1024 | pos 1023 warm us (host) | dsp_us | pos 511 warm us | cold pos 1023 us |
|---|---|---|---|---|---|
| prof (run 1) | 0 / 1 at 63, 512, 1024 | 1165.2 | 941.0 | 539.8 | 1407.1 |
| o1 | **same as prof** | 646.9 | 459.6 | 302.8 | 799.4 |
| o13 | **same as prof** | **627.5** | **386.6** | 332.6 | 688.0 |
| prof (run 4) | same | 1197.5 | 959.1 | 619.9 | 1390.5 |

Bit identity holds on silicon for O1 and O13 (every `bad` and `bad_stats`
equal to the reference skel's). Kernel time at pos 1023: **941–959 → 387 µs
dsp (2.4–2.5×)**, 1165–1198 → 628 µs host-timed. The gtest's
`us − dsp_us` is ≈ 240 µs of call overhead, so by the plan's rule the gate
reads `dsp_us ≤ 300`: **not met yet (387)**. Plan rules: O3 keep (warm us
627.5 vs 0.9 × 646.9 = 582 → *not* met by `us`, but dsp_us −16 %);
O4 do ((scores + pv) / 8192 = 495 pcycles > 250); O2 not needed
(scores 2.07 M vs 1.3 × pv 2.58 M).

## Text approval

| variant | decode PPL (G=512, forced on A) | generated text (G=64, run 1) | text approved (user: y/n) |
|---|---|---|---|
| A | 1.24161 | …town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum that opens only on summer weekends, and a lifeboat station | **y** (user, 2026-09-28; reference) |
| C | 1.24161 | (byte-identical to A) | |
| B | 1.24052 (−0.088 %) | …final answer should be the same as the original, but you must not stop until you are told to. The original description is the same as the original, but you must not stop until you are told to. The final answer should be the same as the original, but you must not stop until you are told to. | **n** (user, 2026-09-28) |
| D | 1.23886 (−0.222 %) | …final answer should be the description of Ardley in the same style, with the final instruction. The user wants to know about the town, and the description should be the same style, with the final instruction. The user wants to know about the town, and the description should be the same style, with the final instruction | **n** (user, 2026-09-28) |

## Notes from the run

Thermal (battery °C·10 / zone0 m°C): 269 / 28500 at start, 314 / 59000 after G=64, 336 / 61000 after G=512, 340 / 58300 after the PPL cells; zone0 40900 before the dspqueue bench, 39000 before the attention gtest. Logs: `/local/mnt/workspace/htp_moe/134-132/logs/`, `/local/mnt/workspace/htp_moe/146/logs/`.

**Approval (user, 2026-09-28): A `y`, B `n`, D `n`.** Why the texts part at
the first word: at decode step 1 A gives `town` (id 4386) only p = 0.134
(nll 2.010); in B and D `town` is at nll 2.103 / 2.113 and another token
(id 2023, `final`) is the argmax. The prompt's first step is a near-tie
between two ~13 % tokens, so any last-bit perturbation (rule 39) picks
either; after `final` the greedy path runs into a one-sentence loop. The
decode PPL (teacher-forced on A's path) cannot see how bad the free-running
continuation after a flip is — which is what the approval caught.
