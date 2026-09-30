# Measurement 132 Part B E5: the two-session NPU end-to-end decode on silicon

**Current set: set_e5b** — branch `htp/132-partb-e3` @ `54a30c7f` (app set
**e3**) and `dev/e2e-shadow-132` @ `e91737a2` (app set **sh**, never merged),
staged at `/local/mnt/workspace/htp_moe/132/set_e5b/` — estimated device
time: **≈ 90 min** (phone rebooted first). set_e5 (below, `10bb91b7` /
`d77a4bf2`) was read on 2026-09-30 06:13.

## set_e5f as read (2026-09-30 09:26–10:2x), and the leak

* MoE dumps E == A `bit_identical=1` (46 files). Lanes ladder (G = 64):
  3,3 → 27.1, **6,3 → 29.5**, 8,3 → 29.4 tok/s (6,3 is the default).
* Speed, A E E A: E 29.38 / 29.25, 30.88 / 30.80, 30.80 / 30.95 tok/s at
  G 64 / 512 / 1024 against A 53.96 / 54.28, 53.57 / 53.49, 51.49 / 51.80.
  E is flat across G: #170 round 3's ATTN_M1 is 0.61 ms a token.
* Per kind (6,3): S1 router 0.80 + MoE 10.46 ms (0.475 a round), wall
  22.5; S2 RMSNORM 0.40, FC 6.06, CONV 0.52, QK 0.10, ROPE 0.03, ATTN 0.61,
  ADD 0.13, DENSE_FFN 2.24, LM_HEAD 3.49 (the Q4M1 kinds 11.8 ms against the
  isolated 8.06), wall 26.0; the ARM's token 29.8 ms. S1 waits 11.2 ms and
  S2 12.2 ms a token: the sessions strictly alternate.
* **Stopped by the mapping leak.** Every E run up to ppl_E_p04 registered
  S1 to 3840 (`s1_arena_mib=3840`), and so did the A runs between them;
  text_A_p04, the run right after ppl_E_p04, stopped at `mapped=3584 MiB in
  14 chunks` (`fastrpc_mmap(64 MiB) failed: err=1`). ppl_E_p04's logcat
  (and ppl_E_p01's, a clean one) show the same close sequence and no
  `remote_munmap64` / `apps_mem` failure; its own close lines are clean
  (timeouts 0, stale 0, both queues `bad=0`). So one two-session run in
  about 12 lost 256 MiB of the next process's room, and the device logs do
  not say how.
* Changes for set_e5g: the teardown stops both queues and both token
  drivers before any munmap and unmaps in the reverse of the mapping order;
  a `[HTP] s2: close mapped_mib / unmap_fail / detach_fail / heap_used_kib`
  line (both sessions' heap from `HAP_mem_get_stats`); a
  `TwoSessions.S1Ceiling` cell the runner reads after every run, so the
  run that loses room is named at once.

## set_e5g (staged)

`/local/mnt/workspace/htp_moe/132/set_e5g/run_e5g.sh` (reboot first,
≈ 100 min; skel `30a1ffd7…`, `libnntrainer.so` `1fee6b5c…`,
`unittest_hvx_two_sessions` `fec2960d…`, runner `fa3b5b9b…`): E5f's cells
(canary, warm-up, dumps, startup, lanes ladder, speed A E E A, 8 prompts)
with S1's ceiling read before the sitting and after every run (below 3840:
`STOP: LEAK … after <run>`), an A sanity after every E run, each E run's
close line and logcat kept.

## Follow-up after the leak: weight prefetch across the alternation

The data dependence leaves one overlap: while S1 runs a layer's MoE round,
S2 is idle, and DRAM has ≈ 12–15 GB/s of headroom beside the MoE feed. A
prefetch lane on S2 (l2fetch or DMA into its L2 scratch) that pulls the
next layer's FC weights during S1's round covers ≈ 130–150 MB of the
≈ 400 MB S2 reads a token — ≈ −3 ms; S1 can prefetch the next layer's
router weights (256 KiB) during S2's stretch. Cell: E at G = 64 / 512 with
the prefetch on and off, per-kind lines, text against A.

## set_e5e and set_e5d as read (2026-09-30 08:40–09:2x, after reboots)

* **E5e: bit-identical over 64 steps on both prompts** — S and Ev
  `first_differing_step=-`, `E2E TRACE first_diff=-`, every shadow record
  equal including conv 1152 / 1152; 0 mismatches. The conv fix holds; the
  bit identity E5b read only for the first 5–25 steps is now whole.
* **E5d: spin 0 is best** — E 20.79 / 20.20 / 16.51 tok/s at G = 64 with
  spin 0 / 20 / 1000 (text == A at all three and at G = 512, 19.55); A
  53.9 at G = 512 (the first A after a reboot read 30.8: set_e5f warms up).
  Per kind at spin 0 (G = 64, 2.1 GHz): S1 router 0.79 + MoE 9.67 ms
  (0.44 ms a round, the isolated rate), wall 35.4; S2 wall 41.3 — RMSNORM
  0.35, FC 8.75, CONV 0.52, QK 0.09, ROPE 0.03, **ATTN_M1 8.33**, ADD 0.12,
  **DENSE_FFN 5.60**, **LM_HEAD 5.97**; S2's Q4M1 kinds 20.3 ms against the
  isolated 8.06; the ARM's token 44.7 ms.
* Readings, and what changed for set_e5f:
  * ATTN_M1 at 2.9 M pcyc a layer is the pre-#170 kernel: the branch
    stacked on the old #175 base. Rebased linearly onto `htp_moe` @
    `e3aad2ce` (#175, #170 round 3, both plans): ≈ 314 k pcyc a layer at
    G = 1024, ≈ 1.2 ms for six layers.
  * Every graph FC ran the L2 feed at 3 lanes, the best at K = 7168 but
    half of K = 2048's 52 GB/s at 6 (#178's ladder): now 6 lanes for
    K ≤ 2048 and 3 above (`NNTR_HTP_FC_LANES` for a ladder).
  * DENSE_FFN's SwiGLU ran the spec's integer division on one thread: now
    the scalar IEEE divide over the pool (`HvxFcQ4.ScalarDivide` gains
    general pairs), and the lm_head's argmax is one vector pass. The
    activation quantizer was already Part A's vector one.
  * The spin default is 0.
* **The budget this design reaches**, re-read with E5d's per-kind line and
  the fixes above: S1 ≈ 10.5 ms (MoE 9.7 + router 0.8); S2 ≈ FC 6 + dense
  FFN 2 + lm_head 3 + ATTN_M1 1.2 + small ops 1.5 ≈ 13.7 ms; hops and the
  ARM ≈ 0.5–1 ms: **≈ 25 ms a token, ≈ 40 tok/s** — below A (≈ 54) and the
  goal (50), because the two sessions alternate (the MoE and the rest never
  overlap) and S2's FC reads its weights at 26–52 GB/s. What closes the rest
  is the FC loop's rate (plan §3.5's ≈ 1 ms toward the 57 GB/s floor) and
  overlap between the sessions, not the transport (44 hops cost ≈ 0.1 ms).

## set_e5f (staged)

`/local/mnt/workspace/htp_moe/132/set_e5f/run_e5f.sh` (≈ 60 min, reboot
first; skel `6632315f…`, `libnntrainer.so` `bd2c6ab9…`, `unittest_hvx_softmax`
`bd9965bb…`, runner `cb83e480…`): canary gtests (exact FC, the SwiGLU /
argmax / quantizer small ops, the scalar divide with the general pairs, the
conv gate); a warm-up A; MoE dumps and the startup cell; a lanes ladder at
G = 64 (3,3 / 6,3 / 8,3); speed A E E A at G 64 / 512 / 1024; 8 prompts at
G = 256 (nll forced on A's ids, text). Per-kind lines from every E run.

## set_e5c as read, and the cause (2026-09-30 08:05–08:12)

* **S** (`NNTR_HTP_FORWARD=1`, one session, CPU-driven) and **Ev** (two
  sessions) left A at the **same step**: prompt512 step 25 (trace
  `step24/pos536/L13.in`), korean step 5 (`step4/pos330/L17.in`), while
  every shadow record was equal over 64 steps (fc 4744, add 3072, router
  1408, swiglu 128, argmax 65, norm 3136 + q|k heads 15360, attention
  heads 12288). So the two-session mechanism is not the cause.
* L13 and L17 are **conv** layers, and CONV1D_GATE was the one resident op
  with no shadow. Its spec `m1_conv_gate_det` was written in the HVX
  kernel's order, `(w0*g + w1*s1) + w2*s0` with every op rounded, and
  never held against the CPU. The phone's decode
  (`neon::causal_depthwise_conv1d_k3_decode`) is `vmulq(w0, x)` and two
  `vfmaq`: `fma(w2, s0, fma(w1, s1, w0*g))`. On the host check's inputs
  the two differ by an ulp on 2075 and 2694 of 16 384 outputs; out_proj's
  Q8 quantizer hides almost all of it, so the divergence is rare and
  data-dependent.
* Fix `a0bf121b`: the spec and `hvx_conv_gate_m1_f32` in the CPU's fused
  order (vector `a*c` and `w0*g`, the two taps as scalar `sffma`, then
  `b*y`); `m1_ops_host_check` now also holds the kernel against an
  independent `fmaf` model of the CPU decode. Open item: the prefill
  kernel `hvx_conv_gate_f32` is unfused too, while the CPU's prefill
  also uses two `vfmaq` (+ bias) — only the HTP conv-block prefill engine
  uses it, not the decode path.

## set_e5e (staged): the conv fix on silicon, the E5c runner

`/local/mnt/workspace/htp_moe/132/set_e5e/run_e5e.sh` (≈ 20 min, reboot
first; e3 skel `554ca8cc…`; sh = `dev/e2e-shadow-132` @ `32fe486d`, skel
`6d592879…`, which adds the conv shadow, tag 8). Expected: S and Ev logits
== S0 for all 64 steps on both prompts, `E2E TRACE … first_diff=-`, every
shadow record equal including `conv=n/n`.

## set_e5d (staged): speed after the wait fix

`376279fc` / `a7917394`: a paused spin of `NNTR_HTP_E2E_SPIN_US` (default
20; also the token queues' DSP spin), then 20 µs sleeps, one read per wake;
pools parked before every post; the token response carries each side's
wall time and pcycles (the clock) and per-kind op cycles. Close lines:
`graph[S1|S2] per-kind pcyc/token: … | wall_ms/token= mhz=`, `graph[S1]
moe pcyc/round=… router pcyc/round=…; s2 fc+dense_ffn+lm_head
ms/token=… (isolated #178: 8.06)`. Runner
`/local/mnt/workspace/htp_moe/132/set_e5d/run_e5d.sh` (≈ 30 min, reboot
first): canary (conv gate + exact FC gtests), A G=64, E at spin 0 / 20 /
1000 G=64 (each with an A sanity and a text check against A), A and E at
G=512 with the best spin, A last.

## set_e5b as read (2026-09-30 06:55–07:32, R3CY10WM83Y after a reboot)

* **The two-session path loads and runs**: S1's arena first (3840), then S2
  (`open_ms` 24.0, `attach_mib=383.6`, `load_ms` 684.9); startup E 15 049 vs
  A 14 706 ms; every E run `calls/token=1.00`, `hops/token=44.00`, timeouts
  0, stale 0, `id_mismatch=0`; no leak (every sanity generated). MoE dumps E
  == A (prefill calls, 46 files) `bit_identical=1`. Shadows at G = 8: every
  record equal; Ev logits == S0 every step; nll S0 / S / Ev == A (8 steps).
* **Not bit-identical past 5–26 decode steps.** The 8-prompt block (G = 256,
  forced on A's ids): E's nll differs from A on all 8 prompts, first
  differing step p01 = 25, p04 = 5, p07 = 26 (targets equal: the logits
  move before the argmax does); top1 255/256 (p01), 245/256 (p04); text ==
  A on p02 p03 p05 p06 p08 only; the G = 512 free-run text differs too.
  E5b's bit identity holds only for the first 5–25 steps.
* **Host**: the in-process build does not reproduce it — lfm25 at prompt
  512, E3 (two sessions) == E1 (one session) == D (hybrid) bit for bit
  over **256** decode steps (every logit and every MoE call; x86
  quantizer order). So the difference is a device reading: an op whose DSP
  result leaves the Android CPU's on some input the 8-step shadows did not
  meet, or something only two PDs on silicon do. set_e5c names the first
  (step, layer, side).
* **Speed**: E 16.7 / 16.0 / 15.2 tok/s at G 64 / 512 / 1024 against A
  53.5–55.1 / 54.4–54.8 / 52.4–52.9. Per token (E_G64_r1): S1 46.7 M pcyc,
  wait 24.3 ms; S2 61.7 M pcyc, wait 23.2 ms; first token 55.4 ms. The two
  sessions strictly alternate, so their op cycles sum over the ≈ 59 ms
  token: 108.4 M / 59 ms ≈ **1.8 GHz** — the clock is not the loss. S1's
  22 MoE + router rounds take ≈ 2.7× the cycles of the isolated MoE
  (≈ 17 M at 0.43 ms × 22). Leading suspect, from the code: the hop wait
  (`tk_take`) spins for `spin_us` = 1000 µs **without a pause and with a
  cache flush-invalidate each iteration** before it sleeps, and each wait
  is ≈ 1.1 ms, so the waiting PD holds one of the six HW threads for almost
  the whole of the other PD's compute; S1's MoE runs six lanes on a pool
  barrier, so one starved lane stretches every call. The pools' own
  post-job spin (≈ 100 µs, `HVX_WORKER_POOL_SPIN`) does the same at each
  hop. The fix and its cells come after E5c (correctness first).
* The runner's `attn cache on S2` pattern expected #170's 24576 KiB; the
  banner is `layers=6 kv=8 gqa=4 head_dim=64 max_seq=2048 cache=49152 KiB`
  (fixed in the runner copy).

## set_e5c (staged): the first step, layer and side where E leaves A

`/local/mnt/workspace/htp_moe/132/set_e5c/` (`md5.txt`; runner
`run_e5c.sh` `05644a61…`, ≈ 20 min, reboot first): e3 = set_e5b's A set
(skel `293628eb…`); sh = `dev/e2e-shadow-132` @ `7d8bf780` with its own
skel `149cf7d4…` (the hop trace: `libnntrainer.so` `78b76969…`). Per
prompt (prompt512, bitset-04-korean), G = 64, forced on A's own ids: A
(self), S0 (the CPU's MoE rows per decode layer, `NNTR_MOE_ROW_TRACE`), S
(`NNTR_HTP_FORWARD=1` + every shadow, every step), Ev (`NNTR_HTP_E2E=1`,
S2's hop rows per step, `NNTR_E2E_TRACE`), then an A sanity. Summary lines:
`nll S0 == A`; `Ev: logits … first_differing_step=`; `E2E TRACE steps=64
first_diff=step<s>/pos<p>/L<l>.in|out` (an `.in` first names S2's ops since
the previous MoE, an `.out` with an equal `.in` names S1's router or MoE);
the FC / norm / attention shadows' `first_diff`.

## set_e5 as read (2026-09-30 06:13–06:39, R3CY10WM83Y after a reboot)

* Every E / Ev run died at load, before decode: S2 opened (`s2_session=1
  s2_effdom=7 open_ms=21.2 open_path=1 hmx=0 vtcm_kib=0`), its FC arena
  placed (`attach_mib=383.6 chunks=2 mapped_mib=448 load_ms=788.6`), then
  S1's MoE arena failed: `fastrpc_mmap(64 MiB) failed: err=1 … mapped=3584
  MiB in 14 chunks` (3696 needed). No logcat was kept.
* The rule, from this and the #178 probe: S1 mapped **before** S2 opened
  reaches 3840 MiB and S2 then maps 3584 beside it (probe Q1, set2 / set3);
  S1 mapping **after** S2 was opened and had mapped stops at 3584, exactly
  where the probe's S2 stopped. Whichever session maps while the other is
  open and mapped gets 14 × 256 MiB. The driver-side cause is not known.
  Fix (`ba486d16`, `54a30c7f`): S2 opens and places its FC set after
  `repack_weight` has mapped S1's arena; the banner gains `s1_arena_mib`
  (3840 expected) and each E run keeps its logcat's fastrpc lines.
* A: fine (A_first 43.5, A_G64 55.5 tok/s); every sanity after an E run
  generated (no leak).
* S (shadows, `NNTR_HTP_FORWARD=1`, `calls/token=95`): **every per-op record
  equal** — FC 600/600, ADD 384/384, router 176/176, SwiGLU 16/16, argmax
  9/9; norm tag0 392/392, q|k heads 1920/1920; attention heads 1536/1536
  (48 records, 6 layers, 8 positions); logits 8/8 == S0; nll S0 / S == A.
* The `graph: q4m1 … feed=vtcm` banner on S2 was wrong (S2 is L2-fed): fixed.
* The dump check compared E's dump against all of A's calls; set_e5b
  compares A's first N calls (N = E's prefill calls).

## Why

Plan `docs/plans/132-part-b-two-session-e2e.md` (on `htp/132-partb-plan`)
step E5: does decode run end to end on the NPU at one call per token, every
op bit-identical to the Android CPU path (B1–B3), and how far is it from A
(B4)? `NNTR_HTP_E2E=1` opens a second cDSP session (S2, lite, no VTCM) beside
the default one (S1: router + MoE), places the FC set + lm_head (74 handles,
≈ 383 MiB) in an S2 arena at load, and runs each token as one dspqueue packet
per session with the 22 MoE rounds (44 hops) over a shared mailbox page; only
S2's argmax id comes back (its argmax skips `bad_word_ids`, `LM_BAN`), the
logits only under `NNTR_PPL_DECODE`. The switch is off by default: A is this
build with nothing set.

## Artifacts (workstation, SDK 6.4.0.1, HexKL 6.4.0.1, NDK r30, v79)

set_e5b at `/local/mnt/workspace/htp_moe/132/set_e5b/` (`md5.txt` beside it):
e3 `nntrainer_causallm` `24fb37f84d5ac51ad084ff5ec891c920`, `libcausallm_core.so`
`0429142211308ca47bdcd98c839aff67`, `libnntrainer.so` `f180cd9c0c3247af488cabaf8f98bbd5`,
`libccapi-nntrainer.so` `8fb6c3598757b0127c02f49f51752c8b`; sh `nntrainer_causallm`
`dfab27bd92d233d9fa2ef216ca9b7862`, `libcausallm_core.so` `e8e61c3146f85504fdbc1e316d0e0e57`,
`libnntrainer.so` `128b283e95ed3a3432056cea3797e12a`, `libccapi-nntrainer.so`
`eaf54d283563597dafb1ab125d590356`; the skel, gtests, libraries, prompts and
tools are set_e5's files (no DSP source changed since `10bb91b7`; a rebuild of
the same sources gives a byte-different skel, so set_e5's device-proven one is
kept). The set_e5 table below is kept for reference.

| file | md5 | built with |
|---|---|---|
| `e3/libnntr_hvx_skel.so` = `sh/libnntr_hvx_skel.so` | `293628ebca9b7cca1c698754689d3d4a` | `test/htp/build.sh` @ `10bb91b7` (the shadow branch changes no DSP source) |
| `e3/nntrainer_causallm` | `216ca07a2e70bf0b78e0afb32b8e6554` | `build_android.sh --htp --cache` @ `10bb91b7` |
| `e3/libcausallm_core.so` | `c85627bd5500e0ff6059f6ed8b058ee4` | same |
| `e3/libnntrainer.so` | `4067eb6d136edeaff82a61608e7e10c9` | `jni/obj/local/arm64-v8a/`; NEEDED `libsdkl.so`, `libcdsprpc.so` |
| `e3/libccapi-nntrainer.so` | `8fb6c3598757b0127c02f49f51752c8b` | same |
| `sh/nntrainer_causallm` | `a9ac35d901654f1587908c82804b780e` | `build_android.sh --htp --cache` @ `d77a4bf2` |
| `sh/libcausallm_core.so` | `61692d7b069734df466e4fd00688bd43` | same |
| `sh/libnntrainer.so` | `9d04a5f168039a418480116d33c18e14` | same |
| `sh/libccapi-nntrainer.so` | `eaf54d283563597dafb1ab125d590356` | same |
| `libc++_shared.so` (both) | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 sysroot |
| `libsdkl.so` (both) | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL `lib/6.4.0.1/armv8_android26` |
| `e3/unittest_hvx_softmax` | `259f6dcd860963708ed8c48ac3325ce5` | `ndk-build` @ `10bb91b7` (the canary, `HvxFcQ4.MatchesSpecBitExact`) |
| `e3/unittest_hvx_two_sessions` | `037cb050b185d7c92eb03648c8789741` | same (built against the new IDL; not run in this sitting) |
| `prompt512.txt`, `bitset-0[2-8]-*.txt` | as in `md5.txt` (#170 S4's) | `docs/measurements/77-prompt512.txt` and the #164 set |
| `tools/{htp_dump_eval,fc_shadow_check,norm_shadow_check,attn_shadow_check}.py` | as in `md5.txt` | the two branches' `tools/htp/` |
| `/data/local/tmp/nntrainer/causallm/models/q40-qs4cx-wh/` | (on the device, unchanged) | NPU model |

Rebuild recipe (fresh worktree): `git submodule update --init --depth 1`;
copy `Applications/CausalLM/lib/libtokenizers_android_c.a`; `set +u; source
tools/htp/env.sh` with `HEXKL_ROOT=/home/j2z0-lee/Qualcomm/hexkl-1.0-beta.2/hexkl_addon
HEXKL_SDK_VER=6.4.0.1`; `generate_stub.sh`; `test/htp/build.sh`; one
`build_android.sh --htp`, then `cd builddir && meson configure
-Dprefix=$PWD/android_build_result && ninja install`, then `build_android.sh
--htp --cache` (re-run `ninja install` after any header change: `--cache`
builds the app against the installed headers).

## Steps (workstation, phone on USB)

1. **Reboot the phone** and let it settle (#178: a leaked cDSP mapping
   survives processes; only a reboot clears it).
2. `bash /local/mnt/workspace/htp_moe/132/set_e5b/run_e5.sh R3CY10WM83Y`
   (≈ 90 min; the copy on the branch is `docs/measurements/132-part-b-e2e-run.sh`).
   It checks the set's md5 on both ends, sets `do_sample false`,
   `bad_word_ids [124900]`, `moe_engine htp` (the #170 S4 config), waits for
   zone0 ≤ 35 °C before each block, and runs, in order:
   * the canary `HvxFcQ4.MatchesSpecBitExact` (`FC_Q4_FIELD total bad=0`);
   * **A first** (G = 8, must generate: no leak from an earlier process);
   * the MoE dumps (A, E at G = 8, `NNTR_HTP_DUMP`) and the **startup cell**
     (`[e2e time]` − `total:` for A and E; E's `open_ms`, `attach_mib`,
     `load_ms`);
   * the **shadow block** at G = 8, prompt 512, forced on A's own
     continuation: **S0** (sh, switch off, logits), **S** (sh,
     `NNTR_HTP_FORWARD=1`, the norm / attention / FC shadows), **Ev** (sh,
     `NNTR_HTP_E2E=1`, logits);
   * **speed**, G 64 / 512 / 1024, mirrored `A E E A`;
   * **text and nll**, 8 prompts at G = 256: A self, E forced on A's ids,
     then A and E text; A forced once more on prompt 1 (null check).
   Every E run is followed by an A run that must generate (a missing
   `generation:` line prints `LEAK` and stops: reboot, note it).
3. Paste `logs/sitting.out`'s summary below, commit this file on the branch,
   push, set #132 to `state:measured`.

Stop rules (the runner's): `0x8000040e` anywhere (stale skel); a device md5
that differs from `md5.txt`; `s2: open FAILED` or `open_ms` > 2000; a token
call that failed (`AEE_EEXPIRED`: a hop timed out, the FARF names the side);
an A run after E that does not generate (`LEAK`).

## Expected lines

* **A** (and S0): `[HTP] moe m1 gemv: on (applied=0x703e1)` once, `[HTP] dspq:
  on` once, `dspq: close calls=N served=N bad=0`, no `s2: open`, no
  `graph: init`.
* **E** (and Ev), each once:
  `[HTP] s2: open s1_effdom=3 s2_session=<n> s2_effdom=<d> open_ms=<ms> info_rc=0x0 open_path=1 hmx=0 vtcm_kib=0 …`;
  `[HTP] s2: fc arena weights=67 handles=74 attach_mib=<≈383> chunks=2 mapped_mib=<384 or 448: 256 MiB + what is left> feed=l2 load_ms=<ms> s1_arena_mib=3840`; `[HTP] graph: q4m1 weights=67 handles=74 feed=l2`;
  `[HTP] graph: init n_ops=228 resident=RMSNORM|FC|CONV1D_GATE|QK_NORM|ROPE|ATTN_M1|ADD|ROUTER_TOPK|MOE|DENSE_FFN|LM_HEAD moe_ops=22`;
  `… max_seq=2048 cache=24576 KiB`; `[HTP] dspq: on …`, `[HTP] dspq[S2]: on … domain=<d>`;
  `[HTP] token driver: on s1_effdom=3 s2_effdom=<d> mbox=65536 spin_us=1000 rounds=22 hops/token=44 …`;
  `[HTP] token driver: first token … hops=44 … logits=0` (`logits=1` under `NNTR_PPL_DECODE`);
  at exit `[HTP] graph[S1]: …`, `[HTP] graph[S2]: …` (pcyc and wait per
  token), `[HTP] token driver: close tokens=N hops/token=44.00 … timeouts=0/0 stale=0/0 … id_mismatch=0 stop_err=0x0/0x0`,
  `dspq[S2]: close calls=N served=N bad=0`, `[HTP] graph: forward calls=N tokens=N calls/token=1.00`.
* **S**: `graph: init n_ops=228 resident=RMSNORM|CONV1D_GATE|QK_NORM|ROPE|ATTN_M1|MOE moe_ops=22`.
* **B2**: `moe-dump-E==A … bit_identical=1` (E's dump holds the prefill MoE
  calls only: the decode MoE runs inside S1 with no ARM call);
  `FC SHADOW … fc=n/n add=n/n router=n/n swiglu=n/n argmax=n/n` (every
  record equal; fc, add, router, argmax each n > 0);
  `NORM SHADOW d_S tag0=n/n tag1_heads=n/n tag1_zero_records=0 tag2=0/0 logits_equal_steps=8/8`;
  `ATTN SHADOW d_S tag3_heads=n/n records=48 layers=6 positions=8 zero_records=0 logits_equal_steps=8/8`;
  `B2 Ev: the E2E token's logits == S0's, every step`.
* **B3**: nll of S0, S, Ev == A's (G = 8); `B3 nll 8/8`, `B3 text 8/8` (G = 256).
* `=== done … expectation mismatches: 0`.

What the shadows read: S runs the E build's DSP kernels as test entries on
the CPU's own input at each op the CPU computes (norm and q|k norm, the
attention, every FC and lm_head slice through S2's L2 feed at 3 lanes, the
residual ADD, the router, the dense FFN's SwiGLU, the greedy pick) — the
per-op B2 records. Under `NNTR_HTP_E2E=1` the CPU skips those layers, so Ev
carries only the whole-token record: its logits per step against S0's.

## Results (fill in)

| variant | gen | run | prefill tok/s | decode tok/s (all) | decode tok/s (last 64) | startup ms | text = A run 1? | calls/token |
|---|---|---|---|---|---|---|---|---|
| A | 64 | 1 | | | | | (ref) | — |
| E | 64 | 1 | | | | | | |
| E | 64 | 2 | | | | | | |
| A | 64 | 2 | | | | | | — |
| A | 512 | 1 | | | | | (ref) | — |
| E | 512 | 1 | | | | | | |
| E | 512 | 2 | | | | | | |
| A | 512 | 2 | | | | | | — |
| A | 1024 | 1 | | | | | (ref) | — |
| E | 1024 | 1 | | | | | | |
| E | 1024 | 2 | | | | | | |
| A | 1024 | 2 | | | | | | — |

| cell | value |
|---|---|
| s2 `open_ms` / `vtcm_kib` / `open_path` | |
| FC arena `attach_mib` / `chunks` / `load_ms` | |
| startup ms, A vs E (`[e2e time]` − `total:`) | |
| `graph[S1]` / `graph[S2]` pcyc/token, wait_us/token (G = 512) | |
| hops/token, timeouts, stale, id_mismatch | |
| MoE dumps E == A (prefill calls) | |
| FC shadow (fc / add / router / swiglu / argmax) | |
| norm shadow (tag0 / tag1 heads), attn shadow (tag3 heads) | |
| Ev logits == S0 (8 steps); nll S0 / S / Ev == A | |
| B3 nll 8/8, text 8/8 | |
| prefill E vs A (−5 % band) | |

Reference (the #178 sittings, this unit): A 51.3–54.4 tok/s at G = 64,
53.5 at 512, 51.2–52.0 at 1024; plan §3.5 projects E at 19.5–21.2 ms/token
(47–51 tok/s), i.e. behind A: B4 is read as the distance, not promised.

## Text approval (per-token-entry handoff)

| variant | decode nll (G = 256, forced on A) | generated text (G = 64, run 1) | text approved (user: y/n) |
|---|---|---|---|
| A | | <paste> | (reference) |
| E | | <paste> | |

## Notes from the run

<uptime at start (reboot), battery, thermal, LEAK / FARF / AEE lines>

## Not verified on the workstation

* Two PDs: the in-process harness runs S1 and S2 in one process (one
  address space, no cache maintenance, no transport time); the device is the
  first run of the ARM-side open / arena / token path across two PDs.
* The E / S cells' bit identity on aarch64 (the x86 host differs from the
  Android CPU order for the FC quantizer, the router, the norms, the
  attention and SwiGLU; `run_inproc_e2e.sh` holds E3 bit-identical to the
  one-session E1 run, not to the Android CPU).
* The runner itself has not been executed (no phone here); its checks were
  written against the in-process logs' formats and #170 S4's runner.
