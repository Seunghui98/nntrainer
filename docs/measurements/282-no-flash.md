# Measurement 282: no-flash decode, the C ladder, #267 L3, the grid (2-bit file, q8 KV, S25)

Issue #282. Branch `htp/282-no-flash`: `htp/276-s1-2bit` @ `141b7a650` (= `htp_decode` @
`082c1b24d` + QS2CX_WH intake + untied-head hook) + `NNTR_MOE_PIN` (`56eb3b7e9`) + #267 L3
(`2260ac64a`, `9a6e3f5e8`, cherry-picked from `htp/267-miss-batch` without its stack-fix
duplicate). File `gemma4_26b_ternary_fcqs4cx` (md5 `426437624d…`). E = one-PD decode.
q8 KV, `lmhead_engine cpu`. S25 `R3CY205ZMND`, 2026-10-09 17:2x–17:54 KST. No PC gates
(user rule). Logs `scratchpad/r276/N/<cell>.log`, runner `282-run.sh`.

**Result.**

- **No-flash decode does not fit this phone.** At the first decode token the app leaves
  MemAvailable 1.6 GB at C = 16 and 1.2 GB at C = 32. The 2-bit expert set is 5.6 GB.
  - Anonymous pinning of all 30 layers (`NNTR_MOE_PIN=2`, before the floor guard)
    **rebooted the phone**.
  - With the guard (MemAvailable floor 2 048 MiB) it pins nothing.
  - Page-cache pinning (`NNTR_MOE_PIN=1`, MAP_POPULATE; mlock refused on all 30 ranges)
    populated 5 568 MiB in 3.5 s. The kernel evicted it again: 1.15 MiB `pgpgin` a miss
    during decode.
  - Every miss is a flash read: 1.4–1.5 MiB `pgpgin` a miss at C = 32.
- **The C ladder tops out at C = 32** (arena 1 408 MiB). C = 34 / 36 / 40 fail at load:
  `nntr_hvx_attn_m1_register failed: AEE_? (0x80000402)`. C = 48 was not tried.
  - C = 32 halves the misses (110.6 → 49.9 a token).
  - Steady decode (last 64 tokens) at p1024 G512 goes from 7.2–8.1 to **9.3–9.5 tok/s**.
- **#267 L3** cuts the MoE kernel calls 149.8 → 66.5 a token, with the ids unchanged. The
  decode effect is within this sitting's spread, because the token is flash-bound.
- **u8×i8 router** (`NNTR_HTP_ROUTER_U8I8=1`, prefill): +1.9 % prefill. It changes the
  routing and the text (114 tokens against 107), so it stays off for the grid.
- **Grid** (init_seq_len = prompt + G):
  - p1024 runs at C = 32. Prefill 359–365 tok/s; decode 5.8–6.9 tok/s overall, 9.5 steady.
  - p2048 now loads, but only at C = 16 (C = 32: `attn_m1_register 0x80000402`). Prefill
    378–390 tok/s; decode 4.1–4.6 tok/s.
  - p4096 does not load at C = 16 or 32 (`no room for the FC set`). It does not reboot.
- **Prefetch (the coordinator's S2 addition): not started.** Its premise was a miss
  costing a DDR copy, and the flash stayed. A mid-token prefetch also needs the DSP to post
  its guess with the miss round; today the guess comes back only at the token's end. Both
  are written up under Next.

## C × memory table (E p1024 G512, q8 KV)

| C | build | prefill tok/s | decode tok/s (all / last 64) | misses/token | miss wait ms | pgpgin MiB/miss | arena MiB | MemAvailable at token 1 MiB | peak RSS KiB | ids == C16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 (276 grid, `s276b`) | 276 head | 352.6 | 5.84 / 7.44 | 110.55 | 66.5 | 0.85 | 704 | — | 3 250 996 | (ref) |
| 16, `PIN=1` (page cache, 5 568 MiB populated, mlock 0/30) | +PIN | 349.6 | 5.12 / 8.14 | 110.55 | 58.7 | 1.15 (incl. the populate's reads) | 704 | — | 3 979 848 | yes |
| 16, `PIN=2` all layers (no guard) | +PIN | — | — | — | — | — | — | — | — | **phone rebooted** |
| 16, `PIN=2` + floor 2048 (pinned 0 layers) | +PIN | 353.0 | 5.67 / 7.18 | 110.55 | 72.3 | 0.94 | 704 | **1 634** | 3 248 740 | yes |
| **32** | +PIN | 340.8 | 6.66 / **9.27** | **49.85** | 48.5 | 1.49 | 1408 | **1 186** | 3 252 228 | yes |
| 32 + L3 | +L3 | 360.1 | 6.54 / 8.69 | 49.85 | 52.4 | 1.42 | 1408 | — | 3 242 580 | yes |
| 32 + L3 + router u8i8 | +L3 | 366.9 | 6.52 / 8.18 | 54.18 | 53.9 | 1.47 | 1408 | — | 3 251 160 | **no** (114 tokens, different text) |
| 32 + L3 (grid cell) | +L3 | 360.7 | 6.86 / **9.53** | 49.85 | 46.0 | 1.38 | 1408 | — | 3 248 876 | yes |
| 34 / 36 / 40 | +PIN | — | — | — | — | — | 1536 / 1600 / 1792 | — | — | load fails: `attn_m1_register 0x80000402` |

- "ids == C16": the generated text (`260-turn106.py`) has the same md5 as the 276-grid
  C16 cell's (`5d73a95e`).
- #267 L3 calls per token: `moe calls/token=66.50 (one-at-a-time 149.77)` at p1024 and
  71.1–71.6 (216–218) at p2048.
- **What bounds the miss.** At C = 32 the wait is 46–54 ms for 50 misses × 1.45 MiB. That
  is ≈ 1.4 GiB/s from flash, with `arm_ms/round` 2.5–2.7 over ≈ 22 rounds a token. The
  ARM-side levers of S2 (batched registration, a split memcpy) work on the copy, and the
  copy is not the bound while the bytes come from flash. **S2 was not implemented.**
  Under this memory budget, miss wait ≤ 15 ms needs ≈ 4× fewer flash bytes a token, not
  less ARM work.

## Grid (E, q8 KV, init_seq_len = prompt + G, L3 on, router u8i8 off; C = 32, else 16)

| prompt | G | C | prefill tok/s | decode tok/s (all / last 64) | text ok? |
|---|---|---|---|---|---|
| 1024 | 64 | 32 | 364.9 | 5.80 / 5.80 | yes |
| 1024 | 512 | 32 | 360.7 | 6.86 / 9.53 (`<turn\|>` at 107) | yes |
| 1024 | 1024 | 32 | 358.5 | 6.72 / 9.48 (at 107) | yes |
| 2048 | 64 | 16 (32 fails) | 379.5 | 4.17 / 4.17 | yes |
| 2048 | 512 | 16 | **390.3** | 4.64 / 5.98 (at 117) | yes |
| 2048 | 1024 | 16 | 378.5 | 4.14 / 5.19 (at 117) | yes |
| 4096 | 64 / 512 / 1024 | — | — | — | **does not load** at C 32 or 16: `NNTR_HTP_E2E=1: no room for the FC set beside the resident experts (mapped=704 MiB, fastrpc_mmap(32 MiB) failed: err=1 …)`. No reboot |
| A p1024 G512 (ref, C 32) | | | 346.8 | 6.81 / 7.68 (at 105) | yes |

Per cell:

| cell | prefill | misses/token | miss wait ms | pgpgin MiB/miss | arena | peak RSS KiB | start cpu / bat / zone0 °C |
|---|---|---|---|---|---|---|---|
| E p1024 G64 | 2 806 ms | 55.44 | 53.7 | 1.48 | 1408 | 3 250 992 | 43.0 / 28.4 / 36.0 |
| E p1024 G512 | 2 839 ms | 49.85 | 46.0 | 1.38 | 1408 | 3 248 876 | 54.3 / 28.6 / 46.9 |
| E p1024 G1024 | 2 856 ms | 49.85 | 47.0 | 1.45 | 1408 | 3 249 236 | 57.0 / 29.2 / 50.0 |
| E p2048 G64 | 5 396 ms | 117.08 | 80.5 | 0.94 | 704 | 3 248 888 | 59.7 / 30.0 / 50.8 |
| E p2048 G512 | 5 247 ms | 114.44 | 88.4 | 1.20 | 704 | 3 255 108 | 64.3 / 30.4 / 52.3 |
| E p2048 G1024 | 5 411 ms | 114.44 | 112.7 | 1.62 | 704 | 3 257 456 | 61.6 / 30.7 / 53.5 |
| A p1024 G512 | 2 953 ms | — | — | — | 1408 | 2 946 464 | 57.0 / 30.7 / 48.4 |

Texts (first 300 characters; judged to `<turn|>`):

- E p1024 (all G, all C, L3 on/off, PIN on/off: one text): `<|channel>thought <channel|>The small harbor town of Ardley, located at a river mouth and characterized by its fishing and farming communities, has a history dating back to the twelfth century. The town's infrastructure and economy, including its stone buildings and fishing industries, are heavily s`
- E p2048 (G512 / G1024 to `<turn|>` at 117): `<|channel>thought <channel|>The report describes a three-year study in a city of 400,000 inhabitants to address the persistent problem of water loss caused by hidden leaks in municipal networks. The project utilized a network of low-cost, acoustic and pressure sensors to identify the location and size`. The rest of the summary is accurate and ends cleanly.
- A p1024 G512: `<|channel>thought <channel|>The small harbor town of Ardley, located at a river mouth and much influenced by the sea and weather, has a long history of fishing, …`

## Gates

| gate | result |
|---|---|
| text sane over the whole generation | yes: every E cell ends on `<turn\|>` (107 / 117) with no loop |
| ids identical across C (same build) | yes: C 16, C 32, PIN 1 / 2, L3: one text md5. The router u8i8 changes it (opt-in, off) |
| misses/token falls with C | yes: 110.55 → 49.85 |
| no flash reads (pgpgin/miss ≈ 0) | **no**: 0.9–1.6 MiB a miss. The RAM budget, above |
| memory within the ceiling | C ≤ 32 at p1024, C 16 at p2048. p4096 does not load. No reboot in any guarded cell |

## Next (ranked by what the numbers say)

1. **RAM for the expert cache.** RSS is 3.25 GB, and the one-PD's ION is ≈ 3.7 GB
   (arena + FC 448 + heap ≈ 1.46 GB + attn_m1 440). The cheapest bytes back are the app's own
   CPU copies of what the DSP already holds: the FCs (`cpu fc skipped=210`), the untied
   head, the embedding. They are file-backed and reclaimable, but they compete with the
   expert pages. `madvise(DONTNEED)` on the CPU FC / head ranges after the hand-over could
   let the page cache keep ≈ 1–2 GB more experts. Measure MemAvailable at token 1 first.
2. **DSP address space for C > 32.** `attn_m1` registration is what fails (0x80000402). A
   q8 `attn_m1` cache (440 → 220 MiB) would free room for ≈ 150 more slots (estimate, C ≈ 37; not measured).
3. **Next-layer prefetch** (coordinator's S2 addition). It hides flash reads behind layer
   L's compute instead of removing them, and needs the guess posted mid-token (a DSP-side
   change). Plan 266 S2's measured guess quality is p ≈ 0.67 / 0.61 at +59–62 % bytes.
   Under flash, that is a bet on overlap, not on bytes.
4. p4096: the FC set's 32 MiB map refused at init 4160 with q8. The same address-space
   budget as 2.
