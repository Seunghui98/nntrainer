# Measurement 178: a second cDSP session beside the loaded app (probe Q1–Q5)

Branch `htp/178-probe` @ `55e9a6ca` (stacked on `htp/132-exact-fc` @ `e5e579b3`,
PR #175; plan `docs/plans/178-second-dsp-session.md` on `htp/178-plan`, PR #180)
— estimated device time: **35 min**

## Why

Decision D of #132 chose option (a): keep the FC set + lm_head (383 MiB) on
the NPU in a second session, because the loaded app has 113 MiB of address
space left beside the MoE arena. Whether this device gives an unsigned app a
second PD with its own 4 GiB, what a DSP-to-DSP hop costs, and whether that
PD gets VTCM or must feed the exact FC through L2 decide between building the
two-session E2E plan, recommendation (c) of #132, or the §3.5 fallback.

## Artifacts (built on the workstation, SDK 6.4.0.1, HexKL 6.4.0.1, NDK r30, v79)

Staged in `/local/mnt/workspace/htp_moe/178/set/` with `md5.txt`; every file
is built from this branch's one tree (the IDL grew by two entries, rule 3).

| file | md5 | built with |
|---|---|---|
| `libnntr_hvx_skel.so` | `e987ddc0ad6c1c56c3f53229be78d251` | `test/htp/build.sh` (this branch; the build is not byte-reproducible, the staged md5 is the one) |
| `nntrainer_causallm` | `2c230e310d6bc0b57bb9f801adf43d5f` | `build_android.sh --htp` (A's app set) |
| `libcausallm_core.so` | `9f88c99e70618bc28ca9d494aebf771f` | `build_android.sh --htp` (2 × `NNTR_HTP_FORWARD_KINDS`, rule 36) |
| `libnntrainer.so` | `639f16e9ec9e439b5ef1a4016eb01c02` | `jni/obj/local/arm64-v8a/`; NEEDED `libsdkl.so`, `libcdsprpc.so` |
| `libccapi-nntrainer.so` | `89fcb0d9b2681b546f9905f34cdc9933` | `jni/obj/local/arm64-v8a/` |
| `libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 sysroot |
| `libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL `lib/6.4.0.1/armv8_android26` |
| `unittest_hvx_two_sessions` | `c47b2d6043296d85ab9f7503da505dbc` | `ndk-build … unittest_hvx_two_sessions` (P) |
| `unittest_hvx_softmax` | `5113932745bc900e3c5cd4fe8e6f0f27` | `ndk-build … unittest_hvx_softmax` (the canary) |
| `prompt512.txt` | `fc65c1588dc66dd764c7013fe96cbb75` | `docs/measurements/77-prompt512.txt` |
| `/local/mnt/workspace/models/lfm2.5-8b-a1b/q40-qs4cx-wh/` | (on the device, unchanged) | NPU model |

The app set is A's: the probe changes no product code (no default, no env
switch, no weight format); the skel's S1 path is the full open as before
(hw_init + HMX lock succeed in S1, so the lite branch is not taken).

## Steps (the orchestrator, phone on USB)

1. `bash /local/mnt/workspace/htp_moe/178/run_178.sh R3CY10WM83Y` (≈ 35 min;
   log and summary in `/local/mnt/workspace/htp_moe/178/logs/sitting.out`).
   It checks the set's md5 on both ends, waits for zone0 ≤ 35 °C before each
   block, and runs, in order:
   * the stale-skel canary `unittest_hvx_softmax --gtest_filter='HvxFcQ4.MatchesSpecBitExact'`
     (`FC_Q4_FIELD total bad=0`; it also covers this branch's change to the
     VTCM feed's DMA call);
   * **A** (nothing set), prompt 512 (`prompt512.txt` = `docs/measurements/77-prompt512.txt`),
     G = 64 and 512, twice each;
   * **P cold**: `unittest_hvx_two_sessions --gtest_filter='TwoSessions.*'`;
   * **A** at G = 1024 twice, then **P warm** at once.
2. Expected lines: every A log `[HTP] moe m1 gemv: on (applied=0x703e1) … dma_bypass=1`
   once, `dspq: on` once, `dspq: close calls=N served=N bad=0`, no
   `calls/token` (rule 36); every P log five `TwoSessions.Q[1-5]_` results
   and the `S2_FIELD` lines below. `0x8000040e` anywhere stops the run
   (stale skel). The plan's stop rules print `S2_STOP rule=…` and are
   reported, not enforced (the remaining cells still run where they can).
3. Paste the summary below, commit this file on the branch, push, set #178
   to `state:measured`.

Stop rules on the day (plan §4 step 4):
* `s2_reserve_rc=0x00000073` (AEE_ENOSESSION): no second session; Q2–Q4
  skip, Q5 checks S1. The §3.5 fallback is then the only option (a) cost.
* `S2_STOP rule=s2_open_hang`: the gtest killed S2's PD after 10 s
  (`FASTRPC_REMOTE_PROCESS_KILL` on S2's effective domain only; never
  `FASTRPC_SESSION_CLOSE`); read `P_*.logcat` for S2's `hexkl_micro_hw_init`.
* `S2_STOP rule=s2_mmap_lt_512`: the 4 GiB is per HLOS process, not per PD.
* `S2_STOP rule=fc_set_over_13ms`: the FC set on S2 cannot beat A.

## Results (fill in)

### A (the sitting's control)

| G | run | prefill tok/s | decode tok/s (all) | decode tok/s (last 64) | text = run 1 | banner (0x703e1, dma_bypass=1, dspq on) |
|---|---|---|---|---|---|---|
| 64 | 1 | | | | (ref) | |
| 64 | 2 | | | | | |
| 512 | 1 | | | | (ref) | |
| 512 | 2 | | | | | |
| 1024 | 1 | | | | (ref) | |
| 1024 | 2 | | | | | |

Reference (A of the #158 sitting, S25 Ultra): 50.6 decode tok/s at G = 512.

### P (Q1–Q5), cold / warm

| Q | cell | cold | warm | pass / stop |
|---|---|---|---|---|
| 1 | `s1_mmap_mib` (expect 3840), `s1_heap_mib` (≈ 113–182) | | | — |
| 1 | `s2_reserve_rc`, `s2_session_id`, `s2_effdom` | | | `0x73` stop |
| 1 | `s2_open_rc`, `s2_open_us`, S2 `open_path` / `hmx_locked` / `vtcm_size` (`session_info who=s2_open`) | | | rc 0, ≤ 2 s |
| 1 | `s2_hmx_call_rc` (lite: `0x80000414`) | | | — |
| 1 | `s2_mmap_mib` with S1 held, `s2_heap_mib` (maps held), `s2_heap_mib_nomap` | | | `s2_mmap_mib < 512` stop |
| 1 | `s2_q4m1_n` / `s2_q4m1_mib` (74 / ≈ 385 = all fit) | | | ≥ 383 + 48 + 64 |
| 2 | `hop_arm_spin_us_0b` / `_12k` (median, p90) | | | — |
| 2 | `hop_arm_block_us_0b` / `_12k` | | | — |
| 2 | `hop_mbox_us_0b` / `_8k` (timeouts 0, bad 0) | | | > 100 µs on both transports flagged |
| 2 | `hop_mbox_thread_cost_pct` (S2 FC with S1 spinning vs parked) | | | — |
| 3 | `ddr_s1_alone_gbs` (≈ 69), `ddr_s2_alone_gbs` (≈ 45) | | | — |
| 3 | `ddr_concurrent_aggregate_gbs` (and `_pinned_cpu0_cpu1`) | | | vs ≈ 70 ceiling |
| 3 | `ddr_sequential_gbs` | | | the design's number |
| 4 | `s2_vtcm_avail_kib`, `s2_vtcm_max_page_kib`, `s2_vtcm_size_kib` | | | — |
| 4 | `s2_fc_rate_vtcm_us` / `_l2_us` / `_direct_us` (K = 7168, N = 2048, best lanes) | | | — |
| 4 | `s2_fc_ms_per_token` per feed, `s2_fc_ms_per_token_best`, `s2_fc_bad_total` | | | > 13 ms stop; bad 0 |
| 5 | `s2_close_rc`, `s1_keeps_hmx`, `s1_keeps_vtcm`, `vtcm_avail_kib_start_end` | | | yes / yes |

### §3.4 projection with the blanks filled

| term | ms/token | source |
|---|---|---|
| MoE, 22 × 0.418–0.43 | 9.2–9.5 | rule 43 |
| FC + lm_head on S2, best feed | | Q4 |
| activation quantizer (HVX, assumed) | 0.3 | #132 ponytail |
| router (multi-chain, assumed) | 1.0 | #132 |
| ATTN_M1, 6 layers | 2.0 | #170 |
| small ops | 0.9 | contract §12 |
| hops, 44 × best transport + 2 ends | | Q2 |
| **total** | | |

## Recommendation (fill in after the table)

<build the two-session E2E plan as a new issue / stop at (c) of #132 / the
§3.5 fallback — with the session split, hops per token and the projected
ms/token>

## Notes from the run

<serial, battery, zone0 log, S2's FARF lines from `P_*.logcat`, anything stale>

## Rebuild recipe (fresh worktree)

```
git submodule update --init --depth 1
cp <another worktree>/Applications/CausalLM/lib/libtokenizers_android_c.a Applications/CausalLM/lib/
set +u; export HEXKL_ROOT=$HOME/Qualcomm/hexkl-1.0-beta.2/hexkl_addon HEXKL_SDK_VER=6.4.0.1
source tools/htp/env.sh; export PATH=$HOME/.local/bin:$ANDROID_NDK:$PATH
./test/htp/build.sh                                   # skel + test/htp/generated
(cd Applications/CausalLM && ./build_android.sh --htp)   # fails at install on a fresh builddir
(cd builddir && meson configure -Dprefix=$PWD/android_build_result && ninja install)
(cd Applications/CausalLM && ./build_android.sh --htp --cache)
ln -sfn $PWD/subprojects/googletest/googletest test/jni/googletest
(cd test/jni && $ANDROID_NDK/ndk-build NDK_PROJECT_PATH=. NDK_APPLICATION_MK=./Application.mk \
   APP_BUILD_SCRIPT=./Android.mk NNTRAINER_ROOT=$PWD/../.. HEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT \
   unittest_hvx_two_sessions unittest_hvx_softmax -j8)
```
`libc++_shared.so` from the NDK r30 sysroot (md5 `b1586b9b…`), `libsdkl.so`
from `$HEXKL_ROOT/lib/6.4.0.1/armv8_android26/`.
