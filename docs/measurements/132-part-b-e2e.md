# Measurement 132 Part B E5: the two-session NPU end-to-end decode on silicon

**Current set: set_e5b** — branch `htp/132-partb-e3` @ `54a30c7f` (app set
**e3**) and `dev/e2e-shadow-132` @ `e91737a2` (app set **sh**, never merged),
staged at `/local/mnt/workspace/htp_moe/132/set_e5b/` — estimated device
time: **≈ 90 min** (phone rebooted first). set_e5 (below, `10bb91b7` /
`d77a4bf2`) was read on 2026-09-30 06:13.

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
