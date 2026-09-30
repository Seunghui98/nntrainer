# Measurement 194 sitting 1: E0 / E1 on silicon, the L0 split, the first PPL gate

Branch `htp/194-s1` @ `9b6f575d` (code; this file is the commit after it),
staged at `/local/mnt/workspace/htp_moe/194/s1/` — estimated device time:
**≈ 110 min** (reboot first). Track `htp_moe_ppl` (plan
`docs/plans/194-htp-moe-ppl.md`): no bit-preserving rule; the accuracy gate
is plan 194 section 1 (P1–P4), read against this sitting's A.

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
