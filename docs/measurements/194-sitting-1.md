# Measurement 194 sitting 1: E0 / E1 on silicon, the L0 split, the first PPL gate

Branch `htp/194-s1` @ `9b6f575d` (code; this file is the commit after it),
staged at `/local/mnt/workspace/htp_moe/194/s1/` — estimated device time:
**≈ 110 min** (reboot first). Track `htp_moe_ppl` (plan
`docs/plans/194-htp-moe-ppl.md`): no bit-preserving rule; the accuracy gate
is plan 194 section 1 (P1–P4), read against this sitting's A.

## Sitting 1 as read (2026-09-30 11:01–11:2x, R3CY10WM83Y, after a reboot)

Logs `/local/mnt/workspace/htp_moe/194/s1/logs/`. **Stopped** at
`sanity_after_E0_G512_r1`: `cannot register a 2048x3584 weight …
fastrpc_mmap(64 MiB) failed` — the per-boot 256 MiB mapping loss (it hits
about one app run in 10–15, A or E; the teardown fix is being built on the
htp_moe track). The ceiling read 3840 after every run before it. No
PPL / mc-40 / lanes cell ran; they move to sitting 1b below.

* **Canary: 3 PASSED.** `FC_Q4_FIELD total bad=0`, **`FC_Q4_NATIVE total
  bad=0`**: the native pair equals `q4_gemv_native_det.h` on silicon, all
  five shapes, three feeds, 9 rows each — `hvx_emu`'s model of the hf
  widening product and the qf32 → hf narrowing holds. Isolated rates at 6
  lanes (`gemv_us`): K=2048 shapes 68–70 GB/s on either feed; K=7168 N=2048
  121 µs VTCM (68 GB/s) but **251 µs L2 (33 GB/s)**; native quantizer 6 µs
  at K=2048, 21–27 µs at K=7168.
* **Speed, G = 64** (decode tok/s r1 / r2; prefill r1): A 57.25 / 53.69
  (579); **E0 30.35 / 28.91**; **E1 31.73 / 31.70**. G = 512 r1: A 53.00,
  E0 30.83. E1 − E0 ≈ +6 % (≈ −1.5 ms a token), well under L1's projected
  −3.8.
* **E0 text = A** at G = 64 (both runs); **E1 text differs by one word**
  at G = 64 ("along it stand" → "along it stands", sentence 4) — the first
  non-exact kernel is visible in greedy text, as expected; P1–P4 judge it.
* **Per kind, S2 (G = 64 r1, ms a token)**:

  | kind | E0 | E1 | Δ | bytes a token | E1 rate |
  |---|---|---|---|---|---|
  | FC | 5.983 | 5.031 | −0.95 | ≈ 205 MB | 41 GB/s |
  | DENSE_FFN | 2.220 | 1.567 | −0.65 | 49.5 MB | 32 GB/s |
  | LM_HEAD | 3.248 | 3.274 | +0.03 | 147.5 MB | 45 GB/s |
  | sum (the `s2 fc+dense_ffn+lm_head` line) | 11.452 | 9.873 | −1.58 | 402 MB | 41 GB/s |

  LM_HEAD did not move: it was already at its feed rate under the exact
  kernel (K = 2048, 6 lanes). FC and DENSE_FFN dropped by what compute was
  and now sit at 32–41 GB/s — **the feed, not compute, bounds all three
  now**. The same kernels in isolation read 68–70 GB/s except the L2 feed
  at K = 7168 (33 GB/s, and the graph runs that shape at 3 lanes): the
  DENSE_FFN down is L2-feed-bound, and the in-graph K = 2048 FCs run at
  ≈ 60 % of their isolated rate (candidates: per-part fork/join and DMA
  start on the small parts — q | k | v, N = 512 parts read 7 GB/s
  isolated —, the prep and the join per op, DDR shared with S1's
  prefetch-free idle; the lane ladder of 1b reads the K = 7168 part).
  Plan §3.1's −3.8 assumed ≈ 50 GB/s in-graph; the reading is 41.

### The L0 split (E0_G64_r1; ms a token)

The tok/s cell (2109 ms / 64 = **32.95**) decomposes as:

| part | ms | what | removable by |
|---|---|---|---|
| S2 kernels | 13.17 | the per-kind line's sum | L1 (done, −1.6), L3, L7 |
| S1 kernels | 10.93 | router 0.80 + MoE 10.13 | L2 (router −0.65), L5 |
| in-DSP non-kernel | 1.15 | s2_wall − both kernel sums: the hops' post → wake-up (hop_us 0.32 + 0.32) and 0.51 of per-op / per-stretch overhead (slot copies, pool fork / join / park, cache clean of the row) | L4 (deadline wait: the 0.64 of hops → ≈ 0.1) |
| **wake** | **3.82** (2.99 / 3.47 / 3.21 in the other runs) | rt − s2_wall: outside both DSP walls, i.e. the ARM post → S2's queue thread → … → S2's answer → the ARM's blocking read. Not S2's first-op latency (that is inside s2_wall, whose clock starts in `nntr_hvx_token_run`). Both DSP queue threads block (`NNTR_HTP_E2E_SPIN_US=0`) and so does the ARM's read, so this is the two wake-ups a token; whether it is the dispatch or the return side, 1b's new line `L0 wake us/token disp s1= s2= s2_pkt= ret s2= clk_resid=` says (the DSP stamps read / write on the QTimer, the ARM on cntvct_el0; `clk_resid` ≈ 0 proves they are one counter) | **L0**: the ARM spins on S2's answer (read_noblock, bounded by the token) and S2's queue thread spins between tokens — S2 only: nothing computes on the DSP between S2's answer and the next packet, so the spin steals no lane (rule 56's objection is to S1 spinning while S2 computes). Expected ≈ −3 ms |
| ARM between tokens | 1.67 | arm_us: the 228-node walk with the hooks, sampler 0.002, register (tokenizer + print) 0.11 (OP-TIME) — so ≈ 1.5 is the walk | L0: the one-call decode step (skip the resident walk) ≈ −1.2 |
| one-time, first decode token | ≈ 140 ms a run | `NNTR_OP_TIME`: layer2_attention max 57.8 ms, layer0 rms_norm 23.9, the other attention layers ≈ 10 each — the first decode token's KV seed / E2E start inside the walk. 2.2 ms a token at G = 64, 0.27 at 512, 0.14 at 1024 | move it to the end of the prefill (not per token) |

So of the "≈ 8 ms outside S2's wall" (34.0 − 26.0 at E5f): wake 3.0–3.8,
ARM walk 1.5–1.7, the first token's one-time 2.2 at G = 64 only; the
DSP-side 1.15 is inside the walls. L0 as planned (bounded spins + one-call
step) removes ≈ 4–4.5 ms a token at every G, ≈ 6.5 at G = 64 with the
one-time moved; plan §3.4's L0 row (−5, range −3…−7) holds.

### Runner fix

`s2 arena after S1 (3840), feed=l2` read BAD on every E run: the pattern
still expected E5f's trailing `s1_heap_kib=`, which the s2 line no longer
prints. The runs were fine; run_s1b.sh matches `… s1_arena_mib=3840`.

## Sitting 1b (staged; run after the teardown fix)

`/local/mnt/workspace/htp_moe/194/s1b/run_s1b.sh [serial]` — the cells
sitting 1 did not reach: canary; speed A E0 E1 E1 E0 A + an E0
`NNTR_OP_TIME=1` run at G = 512 and 1024; the lane ladder (E1 G = 64 at
6,3 / 6,6 / 8,8) + an E0 L0 run at G = 64 (the wake split); the 8 prompts
at G = 256 (A self, A forced p01, E0 / E1 forced, E1 free text); mc-40 (A
and E1 all 40, E0 q01–q08). **Resumable**: each block leaves
`logs/done/<block>`; after any STOP, reboot and run the same command — it
skips finished blocks and restarts the unfinished one. ≈ 100 min in all;
the summary (nulls, P1–P4, speed, L0 with the wake split, lanes) prints
after the last block. Checked on the workstation with a fake `adb` (two
boots, a stop injected mid-block, the resume and the summary), not on the
phone. Staged from `1ba5ae06` (the wake split); **re-stage after rebasing
onto the teardown fix**: rebuild (rungs 2, 3; `ninja -C builddir install`
first — `--cache` skips nntrainer), then `bash
/local/mnt/workspace/htp_moe/194/s1b/stage_s1b.sh` rewrites `app/`,
`tools/`, `md5.txt` and the runner's commit.

## Why

Plan 194 step S2. Three readings decide what is built next: the **L0
split** (where the ARM's ≈ 8 ms outside S2's wall goes: the dspqueue
wake-up, the layer walk, the app's per-token path), **L1's Δ** (the native
FC / DENSE_FFN / LM_HEAD kernels: ≈ 85 → ≈ 12 packets a step; is the FC now
DMA-bound, and at which lane count), and **the first PPL of a non-exact
kernel** on the 8B (P1–P4). E0 is the null: the bit-preserving E5g path,
whose nll lines must equal A's to 17 digits.

## What changed since E5f/E5g (`origin/htp/132-partb-e3` @ `69251151`, rebased onto `htp_moe_ppl`)

* `NNTR_HTP_PPL_LEVERS=<mask>` (bit n = lever Ln; `2` = L1). Unset / `0`
  = E0, the CPU-exact kernels. Banner: `[HTP] ppl levers=0x2 L1=native_fc`.
* L1: `hvx_q4m1_prep_vec` + `hvx_q4m1_gemv_groups_native`, spec
  `nntrainer/tensor/q4_gemv_native_det.h`, under the op feed word's bit 16.
  Host: bit-exact to the spec 3028/3028; spec vs the CPU order 149.4 dB on
  natural rows (37.6 dB on rows built to sit on q rounding edges); 11.5
  packets a 32-column step on v79 -O3 (static count; the exact kernel's
  loop 48.5); in-process hd64 PPL −0.009 % vs E0, lfm25 logits 32.7 dB.
  **Deviation from plan §3.1's wording:** the running sum is an IEEE f32
  add per block (a `Q6_Vsf_*` pair), not a qf32 accumulator converted once
  a group: `hvx_emu` does not model a qf32 chain, so a qf32 sum could not
  be held to a spec bit for bit (the device canary). It costs ≈ 1
  conversion a block; the static count above includes it.
* L0: `[HTP] token driver: L0 us/token rt= s2_wall= s1_wall= wake= hop_us
  s1= s2= arm_fwd= arm_us= arm_n=` at every E run's close (rt the ARM's
  round trip, wake = rt − S2's wall, hop_us the post → wake-up time inside
  each wall, arm_fwd tokenForward on the ARM, arm_us the ARM between two
  tokens: layer walk + sampler + tokenizer + print).
* `NNTR_PPL_DECODE_ALTS=<ids>` and the mc-40 set
  (`docs/measurements/prompts/mc-40*`, `tools/htp/mc_ids.py`,
  `tools/htp/mc_score.py`).

## Artifacts (built on the workstation, SDK 6.4.0.1, HexKL beta.2 6.4.0.1, NDK r30, v79)

| file (under `/local/mnt/workspace/htp_moe/194/s1/`) | md5 | built with |
|---|---|---|
| app/libnntr_hvx_skel.so | `5f4980e72d2ea961d54c3b3c615f8302` | `test/htp/build.sh` |
| app/nntrainer_causallm | `f5afd05b2674d484ba637868ec5a5ffd` | `build_android.sh --htp --cache` |
| app/libcausallm_core.so | `a7cf87090d77aed9b89136c425dc0ebd` | 〃 (`NNTR_HTP_FORWARD_KINDS` ×2, `NNTR_PPL_DECODE_ALTS` ×2) |
| app/libnntrainer.so | `9e1a5660b4bbb379fafaf49306802270` | 〃 (`jni/obj/local`; `NEEDED libsdkl.so, libcdsprpc.so`) |
| app/libccapi-nntrainer.so | `d4d3dbf4d82bfffd0812f0792d437823` | 〃 (`jni/obj/local`) |
| app/libc++_shared.so | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 sysroot |
| app/libsdkl.so | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL beta.2 `lib/6.4.0.1/armv8_android26` |
| app/unittest_hvx_softmax | `d99abba7e1c4702e1e9363d8064bc501` | `ndk-build` (canary: `HvxFcQ4.MatchesSpecBitExact`, `NativeMatchesSpec`, `SmallOpsMatchSpec`) |
| app/unittest_hvx_two_sessions | `46d6dadd85dc1a05cfc3e873abe9cca1` | `ndk-build` (`TwoSessions.S1Ceiling`) |
| app/prompt512.txt + app/bitset-0{2..8}-*.txt | as `docs/measurements/prompts/README.md` | the 8-prompt set |
| app/mc/q01..q40.{txt,ids}, app/mc/alts.txt | `md5.txt` | `tools/htp/mc_ids.py` |
| run_s1.sh | `18788dfb172b3201ea6a1d26d16cca66` | this sitting's runner |
| NPU model `models/q40-qs4cx-wh` on the device | unchanged since #132 E5 | — |

`md5.txt` in the stage lists every file; the runner checks it on both ends
and stops on a mismatch.

## Steps (workstation, phone on USB)

1. **Reboot the phone**, let it idle to zone0 ≤ 35 °C.
2. `bash /local/mnt/workspace/htp_moe/194/s1/run_s1.sh R3CY10WM83Y`
   (≈ 110 min). Everything goes to `logs/sitting.out`; one log per run in
   `logs/`. It stops by itself on `0x8000040e` (stale skel), `LEAK` (S1's
   ceiling < 3840 MiB after a run), a failed token (`AEE_EEXPIRED`) or
   `FATAL`.
3. Paste the summary blocks of `sitting.out` (from `--- nulls` to `=== done`)
   below, the G = 64 texts (`logs/texts_g64.txt`) into the text table, mark
   `text approved`, commit this file on the branch.

Order: md5 → config (greedy, `bad_word_ids [124900]`, `moe_engine htp`) →
ceiling → canary → warm-up A → per G in 64 / 512 / 1024: cool,
`A E0 E1 E1 E0 A`, `E0 NNTR_OP_TIME=1` (L0) → lanes `E1` G=64 at
`NNTR_HTP_FC_LANES` 6,3 / 6,6 / 8,8 → 8 prompts at G = 256: A self (writes
`cont_pXX.ids`), A forced p01 (null), E0 forced, E1 forced, E1 free text →
mc-40 at G = 1: A all 40, E0 q01–q08, E1 all 40. After every run the S1
ceiling; after every E run an A sanity (G = 8).

Expected lines: every E run `[HTP] ppl levers=0x0 L1=exact` (E0) or
`0x2 L1=native_fc` (E1), `s2: fc arena weights=67 handles=74 … feed=l2`,
`token driver: close tokens=… hops/token=44.00 … timeouts=0/0 stale=0/0 …
id_mismatch=0`, `calls/token=1.00`, `s2: close … unmap_fail=0
detach_fail=0`, `L0 us/token …`; every A run `[HTP] dspq: on`, no
`s2: open`; the canary `[  PASSED  ] 3 tests` with `FC_Q4_NATIVE total
bad=0`.

## Results (fill in)

### Nulls and gate (plan 194 section 1)

| check | pass | read |
|---|---|---|
| A forced p01 ≡ A self (17 digits) | identical | |
| E0 nll ≡ A, 8 prompts | 8/8 | |
| E0 mc ≡ A, q01–q08 | 8/8 | |
| E0 text ≡ A r1 at each G | 0 DIFF | |
| canary (exact FC, native FC, small ops) | 3 PASSED, native bad=0 | |
| **P1** pooled PPL, E1 / A (2048 steps) | ≤ 1.02 | |
| **P2** per prompt, max E1 / A | ≤ 1.05 each | |
| **P3** mc-40 right (A, E1), Σnll ratio | E1 ≥ A − 2, ≤ 1.05 | |
| **P4** new loops (E1 where A has none) | 0 | |
| ceiling after every run | 3840 | |

### Speed (prompt 512; decode tok/s all / last 64; prefill tok/s)

| variant | G | r1 | r2 | prefill | text = A r1 |
|---|---|---|---|---|---|
| A | 64 | | | | (ref) |
| E0 | 64 | | | | |
| E1 | 64 | | | | |
| A | 512 | | | | (ref) |
| E0 | 512 | | | | |
| E1 | 512 | | | | |
| A | 1024 | | | | (ref) |
| E0 | 1024 | | | | |
| E1 | 1024 | | | | |

Reference (E5f, same unit, 2026-09-30): A 53.96 / 53.57 / 51.49, E 29.38 /
30.88 / 30.80 tok/s at G 64 / 512 / 1024; S2 per kind FC 6.06, DENSE_FFN
2.24, LM_HEAD 3.49 ms (11.79 of the Q4M1 kinds), S2 wall 26.0, ARM token
29.8 ms. Goal ≥ 50 at every G (plan 194 section 1; §3.4 expects E1 at
≈ 39.7 / 42.2 only after L0 too).

### L0 split (E0, `NNTR_OP_TIME=1`; µs a token)

| G | rt | s2_wall | s1_wall | wake | hop_us s1 / s2 | arm_fwd | arm_us | op_time: infer / sample / register / unattributed |
|---|---|---|---|---|---|---|---|---|
| 64 | | | | | | | | |
| 512 | | | | | | | | |
| 1024 | | | | | | | | |

### Per session (G = 512 r1, `graph[S1|S2] per-kind` lines)

| variant | S1 router / MoE ms | S2 FC / DENSE_FFN / LM_HEAD ms | S2 wall | arm token_ms |
|---|---|---|---|---|
| E0 | | | | |
| E1 | | | | |

### FC lane ladder (E1, G = 64)

| `NNTR_HTP_FC_LANES` | decode tok/s | s2 fc+dense_ffn+lm_head ms/token |
|---|---|---|
| 6,3 (default) | | |
| 6,6 | | |
| 8,8 | | |

## Text approval

| variant | decode PPL pooled (G=256, forced on A) | mc-40 right / Σnll | generated text (G=64, r1) | text approved (user: y/n) |
|---|---|---|---|---|
| A | | | <paste> | (reference) |
| E0 | (≡ A) | (≡ A, q01–q08) | <paste> | |
| E1 | | | <paste> | |

## Notes from the run

<serial, reboot/uptime, thermal waits, any STOP line and its run, FARF>

## What each outcome means

* E0 not ≡ A anywhere: a broken build or skel, not a lever — nothing of
  E1 is read; re-stage.
* `NativeMatchesSpec` bad > 0 with E0 fine: silicon's hf widening product or
  qf32 → hf narrowing is not what `hvx_emu` models; E1's PPL still counts
  (the gate is PPL, not the spec), the host spec is corrected before S3.
* E1 fails P1–P4: L1's numerics go back to S1 before anything else is
  built (the quantizer first: f16(d) from the true amax / 127 and the
  CPU's id; then an exact per-block product).
* The L0 row decides S3's L0 design (plan 194 §3.1: the one-call decode
  step if `arm_us` dominates, the ARM's bounded spin if `wake` does).
