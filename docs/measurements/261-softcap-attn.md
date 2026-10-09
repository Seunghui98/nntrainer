# Measurement 261 sitting 1: levers 2 and 3a measured before code, Gemma-4 26B QS4CX, S25 Ultra

Issue #261, plan `docs/plans/261-decode-levers.md` §2.2 (lever 2), §2.3 (lever 3a), §4 S0 / S3.
Branch `htp/261-softcap-attn` on `htp_decode` @ `4c3953bb1` (the #262 base with #263 router,
#271 miss readers, #272 DMA bypass and the #267 OP_TIME split).

## Summary

Neither lever was built. In both cases the measured breakdown contradicts the plan's premise.

- **Lever 2** has nothing to skip on the current tree:
  - The real checkpoint has `tie_word_embeddings: true`. A tied head folds the final norm
    and the softcap into itself (`FOLD_OUTPUT_NORM`, `496a04979`, from the #4415 port).
  - `Gemma4CausalLM::constructModel` adds the `logit_softcapping` layer only for an untied
    head (`gemma4_causallm.cpp:942-951`).
  - At a resident row the tied head's hook returns 1 and `continue`s
    (`tie_word_embedding.cpp:609-617`), so the ARM computes no `tanh`. The DSP's LM_HEAD caps
    the logits before its argmax (`hexkl_graph.c:563` at `4c3953bb1`).
  - All of the ARM's time between token calls is `arm_us` ≈ 1.46–1.56 ms/token (table below).
    The plan's 1.27 ms came from P4: the dummy file, on a tree without the fold.
- **Lever 3a** (softmax rewrite, plan §2.3) is bounded at about 1 ms:
  - The silicon phase words put the whole softmax at **6.8 % (p512) and 5.4 % (p1024)** of
    the call's lane time.
  - **PV is 66–70 %** of the lane time. It runs at 64–71 lane pcycles per 64-lane FMA, where
    the v79 ISS reads 11.4. The cost is in memory, not arithmetic.
  - The plan's −4…−6 ms at G 512 cannot come from the softmax.
- What was built is the measurement channel: `7a2902817`, the ATTN_M1 phase words per KV cache
  in the one-PD token under `NNTR_HTP_PROFILE`. This is plan §4 S0's missing read, i.e.
  S3's first commit.

## Artifacts

| item | value |
|---|---|
| P (this sitting) | `7a2902817`; skel v79 `38820e8d83bef55419ce5f5ea20addb5`, v81 `458818b2d83617ea6ed6c42265a97608` (both `UNDEFINED SYMBOLS OK (68 runtime imports)`, `ARCH OK`) |
| app (P) | `nntrainer_causallm` `55c918f99d1976ecabf4dd61686c7932`, `libcausallm_core.so` `499776637b556ce2553b91bcfff6a745`, `libnntrainer.so` `2adf91641b2e20da2387c015459159a2`, `libccapi-nntrainer.so` `f7ac413f8b16be9cf9db14958a2e7201` (`jni/obj/local`; NEEDED `libsdkl.so`, `libcdsprpc.so`; `NNTR_HTP_FORWARD_KINDS` strings = 2) |
| device dir | `/data/local/tmp/nntrainer/causallm/s261p/`, device `md5sum` == local for all 5 files; `libc++_shared.so`, `libsdkl.so`, `unittest_hvx_two_sessions` copied from `s260r2/` |
| reference | 267 sitting D (`docs/measurements/267-dma-bypass.md`): its ids equal the 260 r2 E |
| configs | the 260 r2 `s260cfg/r2_E_p{512,1024}_g512` (C 16, fp16 KV, `lmhead_engine cpu`). Prompts are 512 / 1024, as in the 266 / 267 sittings, so the numbers compare before / after. The 1024 / 2048 / 4096 standard applies to new sittings |
| device | S25 Ultra `R3CY205ZMND` (v79), lock `r261b@…` 14:48:42–14:52:11 KST (free when polled) |
| runner | `docs/measurements/261-softcap-attn-run.sh` (`wait <stage>`, `one <log> <p> <G> prof NNTR_HTP_PROFILE=1`) |

Cool start: relaxed rule (battery ≤ 32.0 °C or a 5 min cap).

| cell | start soc / bat / zone0 | end soc / bat / zone0 |
|---|---|---|
| p512 | 31.8 / 24.4 / 27.1 | 51.2 / 25.6 / 43.4 |
| p1024 | 55.0 / 25.6 / 41.1 | 61.6 / 27.9 / 49.2 |

The p1024 cell started right after the p512 one, because the battery was under 32 °C. Its SoC
start was hot. compute_mhz read 2112 in both cells.

## Host / build rungs (on `7a2902817`)

| rung | result |
|---|---|
| 1 `run_host_checks.sh` | `ALL CHECKS PASS`, `WORKER POOL LANES OK`. New: `TOKEN ATTN PHASES OK: 3334 calls with words (every third token), logits unchanged …`. Also `TOKEN DRIVER BIT-IDENTICAL: tokens 10000/10000 … logits bit_identical=1 … kind_ns/wall=0.999` and `TOKEN POOL BIT-IDENTICAL … bit_identical=1` |
| 1 `run_inproc_e2e.sh` | `INPROC E2E PASS`. The 60 gate lines (`bit_identical=`, `tokens htp==cpu`, `self-test ok`, `ppl-decode`, `min_snr_db`) equal the base `4c3953bb1` run's, line for line |
| 1 gtests | `*qs4cx*` `[  PASSED  ] 3 tests`, `*Lfm2Moe*` `[  PASSED  ] 7 tests` |
| 1 `htp_syntax_check.sh` | exit 0 |
| 2 skel v79 / v81 | `UNDEFINED SYMBOLS OK (68 runtime imports)`, `ARCH OK (V79)` / `ARCH OK (V81)` |
| 3 app | built (`build_android.sh --htp`), NEEDED lines and strings count as above; device gtests not built (no kernel change) |

## Results (C 16, KV fp16, `NNTR_HTP_PROFILE=1`)

| cell | prefill tok/s | decode tok/s (last 64) | tokens | text == 267 D | misses/token | miss wait ms/token | ATTN_M1 wall ms (267 D) | FC / DENSE_FFN / LM_HEAD / ROUTER ms | MOE net ms | arm_us/token | heap_used_kib | peak RSS KiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P p512 G512 | 147.1 | **5.01** (5.50) | 169, `<eos>` | **identical** | 94.96 | 117.8 | **12.12** (12.09) | 12.26 / 6.27 / 8.30 / 3.15 | 24.06 | 1456.7 | 1395573 | 2,899,184 |
| P p1024 G512 | 183.8 | **4.10** (3.99) | 512 | **identical** | 120.81 | 155.6 | **22.22** (22.14) | 12.36 / 6.43 / 8.29 / 3.16 | 24.29 | 1544.5 | 1424757 | 2,948,528 |

- With the phase words on, ATTN_M1 reads within 0.4 % of 267 D, so the channel costs nothing
  measurable.
- Decode tok/s is above 267 D's (3.36 / 2.65) because this base carries #271's miss readers:
  the miss wait drops 215 → 118 and 289 → 156 ms. These are not lever numbers.
- `heap_used_kib` equals 267 D's at both prompts.

### ATTN_M1 phase words (silicon, k pcycles per token, lane-summed; wall at 2112 MHz)

| cell / cache | calls | scores | softmax (exp / et / div / max / sum) | pv | pool → ms | busy_max | call wall ms |
|---|---|---|---|---|---|---|---|
| p512 sliding (n_kv 8, gqa 2, hd 256) | 25 | 27 871 (28.8 %) | **7 696 (8.0 %)** (3 470 / 1 316 / 1 914 / 313 / 140) | **61 088 (63.2 %)** | 18 622 → 8.82 | 17 828 | 9.05 |
| p512 full (n_kv 2, gqa 8, hd 512) | 5 | 7 000 (19.8 %) | 1 287 (3.6 %) | **27 093 (76.6 %)** | 6 123 → 2.90 | 5 974 | 2.95 |
| p1024 sliding | 25 | 46 624 (26.9 %) | 11 348 (6.5 %) (5 356 / 2 100 / 2 566 / 326 / 253) | **115 476 (66.6 %)** | 32 872 → 15.56 | 31 992 | 15.80 |
| p1024 full | 5 | 14 614 (19.0 %) | 2 300 (3.0 %) | **60 187 (78.1 %)** | 13 172 → 6.24 | 12 965 | 6.29 |

- **Softmax share over both caches** is 6.8 % at p512 and 5.4 % at p1024. With 6 lanes
  busy (`lanes=6.00`, busy_max within 4 % of pool), removing the softmax entirely buys about
  0.8 ms at p512 and 1.2 ms at p1024. That is the ceiling for lever 3a, and 3a also changes
  bits (gate D2).
- **PV per 64-lane FMA.**
  - Sliding p512: 61.09 M / (25 × 597 × 16 heads × 4 chunks = 955 k) = **64 pcycles**.
  - Full p512: 27.09 M / 382 k = **71**.
  - The v79 ISS on the same kernel and shape reads 11.4 (sliding, L 640, warm; see below).
  - So PV is waiting on memory, and the ISS does not model it.
- **Where PV's bytes go.** `pv_group` walks the positions once per 64-wide chunk
  (`hvx_attn_m1_f32.c:541-582`, the `ponytail:` note there):
  - Sliding layers: 4 strided walks over 512 B V rows per group.
  - Full layers: 8 walks over 1 KiB rows, and with gqa 8 each kv head is walked by 2 groups
    of 4 heads.
  - Each walk issues its own `l2fetch_box` of the whole rows.
- **The bit-identical candidate this points at:** one walk per group with every chunk's
  chains live, 2 heads × 4 chunks = 8 accumulators for the sliding shape.
  - Every (head, d) chain keeps its own p order, so the bits are unchanged.
  - This replaces lever 3a. It is not built here (plan change, see Next).
- **Scores** run at 29 pcycles per FMA on silicon (sliding p512) against the ISS's 7.1: the
  same memory effect on the Kt tiles, at a smaller share.

### v79 ISS (exploration, not a device number)

Setup: `hexagon-sim -mv79 --timing`, the tree's `hvx_attn_m1_forward_prof`, pool NULL (one
thread), second call. Values are k pcycles.

| shape | L | scores | softmax | pv | pool |
|---|---|---|---|---|---|
| sliding | 640 | 291 | 111 (13 %) | 469 | 871 |
| sliding (window) | 1024 | 447 | 177 (13 %) | 767 | 1391 |
| full | 640 | 548 | 91 (7 %) | 694 | 1333 |
| full | 1280 | 1063 | 179 (5 %) | 2632 | 3875 |

The ISS already put the softmax at 13 % or less. The silicon split above confirms it.

## Lever 2: what the code and the logs say

- `Gemma4CausalLM::constructModel`: `if (!TIE_WORD_EMBEDDINGS && FINAL_LOGIT_SOFTCAPPING > 0)`
  is the only `logit_softcapping` layer, and the 26B config is tied.
- `TieWordEmbedding::incremental_forwarding`: a resident row goes through `htpDecodeLmHead`
  and `continue`s. `head_of`'s `tanh` loop runs only on the non-resident path and on the
  `NNTR_PPL` rows, which are scored uncapped.
- `arm_us`, the ARM's whole time between token calls (#194 L0), is 1456.7 / 1544.5 µs per
  token here and 1558.8 in 267 D.
- A DSP-side variant (skip `hvx_softcap_m1_f32` when no logits are wanted) is not
  bit-identical: float `tanh` can tie two raw logits, which moves the first-max. It would also
  save only a fraction of the 262 144 `tanh` inside LM_HEAD's 8.3 ms. Not taken.

## Gates

| gate | result |
|---|---|
| ids identical with the channel on | **pass**: p512 169 tokens `<eos>`, p1024 512 tokens, generated text byte-equal to 267 D (= 260 r2 E) |
| misses/token unchanged | **pass**: 94.96 / 120.81 |
| ATTN_M1 wall with the channel on | 12.12 / 22.22 ms vs 12.09 / 22.14 (+0.3 %) |
| memory | **pass**: `mapped_mib` 449.12, `heap_used_kib` equal to 267 D |
| D2 PPL / lever-3a gates | **not run**: lever 3a was not built (see Summary) |

## Not run

- Lever 2 and lever 3a cells and the PPL cell: there is no code for either lever.
- No A control: the channel changes no value, and the ids equal 267 D.
- No G64 / G1024 cells.
- The ARM-side `NNTR_OP_TIME` breakdown of `arm_us`: not needed once the layer was shown absent.

## Next (issue back to `state:needs-plan`)

Plan 261's lever 3 should be re-derived from this split. The candidates:

1. **PV in one walk** (bit-identical, no IDL change; `pv_group`'s ponytail). The expected gain
   is not modelled: it depends on how much of the 64 pcycles per FMA is the 4–8× re-walk and
   how much is DDR.
2. More q heads per V load on the full layers (gqa 8 in groups of 4 today).
3. Plan §2.3's 3b (int8 KV): this halves exactly the bytes the split shows PV waiting on.

Logs are in the session scratchpad (`r261b/logs/`), not in the repo.
