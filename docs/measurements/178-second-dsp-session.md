# Measurement 178: a second cDSP session beside the loaded app (probe Q1–Q5)

Branch `htp/178-probe` @ `bc3c52a8` (stacked on `htp/132-exact-fc` @ `e5e579b3`,
PR #175; plan `docs/plans/178-second-dsp-session.md` on `htp/178-plan`, PR #180)
— estimated device time: **35 min**

## Why

Decision D of #132 chose option (a): keep the FC set + lm_head (383 MiB) on
the NPU in a second session, because the loaded app has 113 MiB of address
space left beside the MoE arena. Whether this device gives an unsigned app a
second PD with its own 4 GiB, what a DSP-to-DSP hop costs, and whether that
PD gets VTCM or must feed the exact FC through L2 decide between building the
two-session E2E plan, recommendation (c) of #132, or the §3.5 fallback.

## Sitting 1 (2026-09-29 23:20, R3CY10WM83Y, set) and set2

Sitting 1 stopped at P cold on the runner's `0x8000040e` rule. What it
answered: canary `HvxFcQ4.MatchesSpecBitExact` FAILED (total bad 1146880,
all in hvx_native / sffma*, #132's G1 finding; hvx_intrin 10/10 cells
bad=0); A G=64 53.07 / 53.07, G=512 54.36 / 54.39 tok/s, banner
`0x703e1` OK; `s1_mmap_mib=3840`, `s1_heap_mib=144`; `s2_reserve_rc=0`,
session 1, effdom 7; `s2_open_rc=0` in 22.4 ms, `open_path=1` (lite),
`vtcm_size=0`; S1 keeps `hmx_locked=1`; `s2_hmx_call_rc=0x80000414`;
**`s2_mmap_mib=3584` with S1 holding 3840 (the 4 GiB is per PD)**; S2 heap
beside its held ladder 334 MiB.

Then S2's shell crashed (`__wrap_malloc`, logcat) right after that heap
probe ran its address space dry; every later S2 call returned AEE_ENOSUCH
(`0x27` = 39), so `s2_q4m1_n=0` was the dead PD, not the registration.
`s1_hmx_smoke_rc=0x8000040e` was the test's own wrong `m_pad` (32, the
block is 64): the skel returns AEE_EBADPARM for a bad length. Fixed in
`706f1eed`; set2 = set with the new `unittest_hvx_two_sessions` (skel and
app set unchanged, same md5), runner `run_178_set2.sh`
(`docs/measurements/178-run-set2.sh`): the probe's rc never stops the
sitting, the canary expects hvx_intrin only (10 cells bad=0), logs in
`/local/mnt/workspace/htp_moe/178/logs2/`. The heap-beside-the-ladder cell
is now Q5's last S2 cell (`s2_heap_mib`), Q1 has `s2_heap_mib_nomap`
(cap 1024).

| file | set2 md5 | set3 md5 |
|---|---|---|
| `unittest_hvx_two_sessions` | `e8d5c98e8fc996d714726e951bdcb325` | `43e22f378062c1a88d6c5dd4a4c3dc9d` (`bc3c52a8`) |
| every other file | as in the table below | same |

Set3 (`/local/mnt/workspace/htp_moe/178/set3/`) = set2 with the probe
that no longer grows S2's heap to the end of its address space; see
"Finding" below.

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

## Steps (set3, as run; phone on USB, **rebooted first**)

1. `bash /local/mnt/workspace/htp_moe/178/run_178_set3.sh R3CY10WM83Y` (≈ 35 min;
   `docs/measurements/178-run-set3.sh`; logs in
   `/local/mnt/workspace/htp_moe/178/logs3/`, summary `sitting.out`). It
   checks set3's md5 on both ends, waits for zone0 ≤ 35 °C before each
   block, and runs, in order:
   * the canary `unittest_hvx_softmax --gtest_filter='HvxFcQ4.MatchesSpecBitExact'`:
     the hvx_intrin cells (5 shapes × 2 feeds, the kernel the probe runs)
     must read `bad=0`; the test itself fails on hvx_native / sffma*
     (#132's G1 finding) and that is not a stale skel;
   * **A** (nothing set), prompt 512 (`prompt512.txt` = `docs/measurements/77-prompt512.txt`),
     G = 64 / 512 / 1024, twice each — **all before any probe**;
   * **P cold** (`unittest_hvx_two_sessions --gtest_filter='TwoSessions.*'`),
     then **sanity_1** (A at G = 8);
   * **P warm** at once, then **sanity_2**.
2. Expected lines: every A log (sanity included)
   `[HTP] moe m1 gemv: on (applied=0x703e1) … dma_bypass=1` once, `dspq: on`
   once, `dspq: close calls=N served=N bad=0`, no `calls/token` (rule 36);
   each sanity also one `generation:` line — a sanity without it (the
   banner alone is printed even when the arena then fails, set2's A_G1024)
   means the probe run before it left the cDSP short and prints `LEAK:`.
   Every P log: five `TwoSessions.Q[1-5]_` results and the `S2_FIELD`
   lines; `S2_STOP rule=s1_ceiling_lost` means the phone was not rebooted
   after a leak. A probe-internal rc is reported, never a stop; `0x8000040e`
   in the canary or an A log stops the run (stale skel).
3. Paste the summary below, commit this file on the branch, push, set #178
   to `state:measured`.

Stop rules on the day (plan §4 step 4), reported by the gtest as `S2_STOP`:
`s2_reserve_rc=0x00000073` (no second session), `s2_open_hang` (killed on
S2's effective domain only), `s2_mmap_lt_512` (the 4 GiB would be per
process), `fc_set_over_13ms`, `s2_dead`, `s1_ceiling_lost`.

## Results — set3 (2026-09-30 00:26–00:38, R3CY10WM83Y after a reboot, 90 % battery, zone0 ≤ 35 °C before each block)

`expectation mismatches: 0`, `plan stop rules hit: 0`. Set3 md5s as in
the table above (`unittest_hvx_two_sessions` `43e22f37…`); canary
hvx_intrin 10 / 10 cells bad=0.

### A (before any probe)

| G | run | prefill tok/s | decode tok/s (all) | decode tok/s (last 64) | text = run 1 | banner (0x703e1, dma_bypass=1, dspq on) |
|---|---|---|---|---|---|---|
| 64 | 1 | 581.8 | 54.38 | 54.38 | (ref) | y |
| 64 | 2 | 459.6 | 51.32 | 51.32 | y | y |
| 512 | 1 | 579.8 | 35.22 | 28.87 | (ref) | y — a transient dip (last 64 at 28.9), not read |
| 512 | 2 | 459.2 | 53.52 | 53.20 | y | y |
| 1024 | 1 | 487.2 | 51.96 | 49.84 | (ref) | y |
| 1024 | 2 | 463.8 | 51.21 | 48.86 | y | y |

### Sanity after each probe run (A, G = 8; not a tok/s cell)

| run | after | arena registered, banner, `dspq: on`, `close … bad=0` | `generation:` line |
|---|---|---|---|
| sanity_1 | P cold | y | y (8 tokens) |
| sanity_2 | P warm | y | y (8 tokens) |

### P (Q1–Q5): all five `[OK]` cold and warm, the set2 numbers again

| cell | cold | warm |
|---|---|---|
| `s1_mmap_mib` / `s1_heap_mib` | 3840 / 140 | **3840** / 140 (set2 warm: 3584) |
| `s2_open_us` (lite, `open_path=1`) | 22 356 | 22 441 |
| `s2_mmap_mib` beside S1's 3840; `s2_heap_mib_nomap` | 3584; ≥ 512 (cap) | same |
| `s2_q4m1_n` / `_mib` | 74 / 383 | 74 / 383 |
| hop dspq spin 0 B / 12 KiB; block 0 B / 12 KiB (µs) | 2.9 / 12.5; 35.7 / 44.2 | 3.1 / 13.5; 62.8 / 71.7 |
| hop mailbox 0 B / 8 KiB (µs; timeouts, bad) | 0.33 / 2.42 (0, 0) | 0.34 / 2.42 (0, 0) |
| `hop_mbox_thread_cost_pct` | 0.1 | −6.5 (noise) |
| DDR S1 / S2 alone; concurrent S1 / S2 own (GB/s) | 70 / 20; 59.4 / 16.5 | 70 / 20; 59.7 / 16.9 |
| `ddr_sequential_gbs` (DSP-only) | 26.0 (42.7) | 25.9 (43.5) |
| S2 VTCM | 0 KiB | 0 KiB |
| FC L2 / direct ms/token; `bad_total` | **8.07** / 33.0; 0 | 8.08 / 33.0; 0 |
| `s2_close_rc`, S1 keeps HMX / VTCM | 0, yes / yes | 0, yes / yes |

The probe logcats show no `apps_mem` / `munmap` error (only the reverse
module's open / close lines).

## Results — set2 (Q1–Q5 repeated by set3; its A G = 1024 void) (2026-09-29 23:28–23:32, R3CY10WM83Y, 100 % battery, zone0 29.7 → 65.6 °C)

`expectation mismatches: 5` — all from the leak below (A_G1024 r1 / r2
have no `dspq: on` / `close` because the app died at arena registration;
their text differs). `plan stop rules hit: 0`.

### A (the sitting's control)

| G | run | prefill tok/s | decode tok/s (all) | decode tok/s (last 64) | text = run 1 | banner (0x703e1, dma_bypass=1, dspq on) |
|---|---|---|---|---|---|---|
| 64 | 1 | 559.6 | 52.59 | 52.59 | (ref) | y |
| 64 | 2 | 559.0 | 52.33 | 52.33 | y | y |
| 512 | 1 | 531.7 | 54.64 | 54.01 | (ref) | y |
| 512 | 2 | 518.7 | 53.89 | 53.60 | y | y |
| 1024 | 1 | — | — | — | void: after P cold, `HTP arena: … fastrpc_mmap(64 MiB) failed: err=1 … mapped=3584 MiB in 14 chunks` | banner only |
| 1024 | 2 | — | — | — | void, same | banner only |

Sitting 1 (same unit, 23:20): A G = 64 53.07 / 53.07, G = 512 54.36 / 54.39.
Reference (#158 B, S25 Ultra): 50.6 decode tok/s at G = 512.

### P (Q1–Q5), cold / warm — all five `[OK]` in both

| Q | cell | cold | warm | pass / stop |
|---|---|---|---|---|
| 1 | `s1_mmap_mib`, `s1_heap_mib` | 3840, 140 | **3584** (the leak), 334 | — |
| 1 | `s1_hmx_smoke_rc` | 0 | 0 | — |
| 1 | `s2_reserve_rc`, `s2_session_id`, `s2_effdom`, URI | 0, 1, 7, `…&_dom=cdsp&_session=1` | same | **pass** |
| 1 | `s2_open_rc`, `s2_open_us`, S2 `open_path` / `hmx_locked` / `vtcm_size` | 0, 22 359, 1 / 0 / 0 | 0, 22 291, same | **pass** (lite, 22 ms) |
| 1 | `s2_hmx_call_rc`; S1 after S2's open | `0x80000414`; `hmx_locked=1`, VTCM 8 MiB | same | as designed |
| 1 | `s2_mmap_mib` with S1 at 3840 / 3584 held; `s2_heap_mib_nomap` | **3584**; ≥ 1024 (cap) | 3584; ≥ 1024 | **pass: the 4 GiB is per PD** |
| 1 | `s2_q4m1_n` / `s2_q4m1_mib` | **74 / 383, all fit** | 74 / 383 | **pass** (≥ 383 + 48 + 64 with the heap) |
| 2 | `hop_arm_spin_us_0b` / `_12k` (median; p90) | 2.8 (3.0) / 12.7 (12.9) | 2.9 / 13.1 | — |
| 2 | `hop_arm_block_us_0b` / `_12k` | 35.6 (68.8) / 46.6 (57.2) | 36.8 / 48.2 | — |
| 2 | `hop_mbox_us_0b` / `_8k` | **0.33 / 2.42**, 10 000 each, timeouts 0, bad 0 | 0.34 / 2.43 | none > 100 µs |
| 2 | `hop_mbox_thread_cost_pct` (S2 FC L2-fed, 402 µs) | −1.2 | +2.1 | noise |
| 3 | `ddr_s1_alone_gbs` / `ddr_s2_alone_gbs` (S2 L2-fed, 6 lanes) | 70 / 20 | 70 / 20 | — |
| 3 | concurrent: S1 / S2 own rates (ARM-window aggregate) | **58.2 / 15.4** (37.3) | 57.8 / 16.0 (38.5) | S1 −17 %; sum 73.6 ≈ the 70 GB/s ceiling |
| 3 | same, ARM threads pinned cpu0 / cpu1 | 58.6 / 14.8 (34.2) | 57.8 / 16.1 (38.5) | pinning changes nothing |
| 3 | `ddr_sequential_gbs` (ARM window; DSP-only) | 27.2 (42.4) | 26.5 (41.6) | — |
| 4 | S2 `vtcm_avail_kib` / `max_page` / `size` | **0 / 0 / 0** | 0 / 0 / 0 | S1 holds all 8 MiB |
| 4 | K 7168 N 2048: VTCM / L2 / direct (best lanes) | refused (rc `0x80000600`) / **316 µs, 26.1 GB/s @ 3 lanes** (17.5 @ 6) / 700 µs, 11.8 @ 6 | same | — |
| 4 | K 2048 N 6144: L2 / direct | **136 µs, 51.9 GB/s @ 6** / 575 µs, 12.3 | same | — |
| 4 | `s2_fc_ms_per_token` L2 / direct; `bad_total` | **8.06** / 32.7; 0 | 8.06 / 34.7; 0 | ≤ 13 → no stop; vs CPU 7.4 |
| 5 | `s2_mmap_mib_q5`, `s2_heap_mib` beside it (cell removed in set3) | 3072, 764 | 3072, 764 | — |
| 5 | `s2_close_rc`, `s1_keeps_hmx`, `s1_keeps_vtcm` (HMX smoke before == after) | 0, yes, yes | 0, yes, yes | **pass** |

The VTCM feed on a session with 0 bytes of VTCM returned `AEE_ERPC`
(`0x80000600`) instead of the skel's `AEE_EINVALIDFORMAT`; not followed
up (the feed cannot run there either way).

### Finding: set2 leaked cDSP mapping room across processes (cause confirmed by set3)

After P cold every later process lost ≈ 192–256 MiB of fastrpc mapping
room on the cDSP: A_G1024 r1 / r2 (and a fresh sanity run by the
orchestrator) died at `fastrpc_mmap(64 MiB) … mapped=3584 MiB in 14
chunks` (3696 needed), and P warm's own S1 ladder read 3584 instead of
3840. It persists across processes (kernel / DSP side), so only a reboot
clears it. The one driver error on the way out (P_cold.logcat,
23:31:39.224, S2's close): `apps_mem_imp.c:203 remote_munmap64 failed for
size 2097152 (vaddrout 0x50a000000) … 0x80000441` — the HLOS-side unmap of
a 2 MiB page of S2's **grown DSP heap** failed. Q5 had just grown S2's
heap to the end of its address space (764 MiB beside a 3072 MiB mapping
ladder); sitting 1's S2 had crashed after the same kind of probe. A 2 MiB
map stuck in the session's address space would block one 256 MiB chunk,
which fits the loss. Every buffer the test maps itself (both ladders, the
mailbox page, the dspqueue payloads) is `fastrpc_munmap`'d before
`nntr_hvx_close`; the Q4M1 weights are DSP heap and `q4m1_register` keeps
no mapping (it copies the FastRPC argument into `memalign`, freed by
`q4m1_release` / close). **Confirmed by set3**: with that cell removed
(`bc3c52a8`, Q1's S2 heap probe capped at 512 MiB) and nothing else
changed, both A sanities after the probe runs register the whole arena
and generate, P warm's S1 ladder reads 3840 again, and the probe
logcats carry no unmap error. The exact driver path from the failed
2 MiB unmap to the lost room is not known. For the E2E design
this is a rule to carry: a session must never grow its DSP heap to the
end of its address space, and a second session's teardown is checked
with an app run after it.

### §3.4 projection with the blanks filled

| term | ms/token | source |
|---|---|---|
| MoE, 22 × 0.418–0.43 | 9.2–9.5 | rule 43 |
| FC + lm_head on S2, L2-fed (VTCM: 0 KiB) | **8.06–8.08** | Q4, set2 / set3 (CPU does it in 7.4) |
| activation quantizer (HVX) | 0.3 **pending** (assumed; 4.8 scalar today) | #132 ponytail |
| router (multi-chain) | 1.0 **pending** (assumed; 4.4 scalar today) | #132 |
| ATTN_M1, 6 layers | 2.0 | #170 |
| norms and small ops | 0.9 | contract §12 |
| hops, 44 DSP↔DSP + 2 ends | **0.1–0.6** (dspq spin 2.8–12.7 µs; mailbox 0.33–2.4 µs ≈ 0.02–0.1) | Q2 |
| **total** | **≈ 20.3–21.1 without the pending two; ≈ 21.6–22.4 with them → ≈ 45–46 tok/s** | |
| A, hybrid (#158 B / set2 G = 512 / set3 G = 512 r2) | 19.7 → 50.6 / 18.4 → 54.3 / 18.7 → 53.5 | |

Overlap does not rescue it: concurrent S1 + S2 reads already sum to the
≈ 70 GB/s ceiling and cost S1's MoE feed 17 %.

## Recommendation

* **The mechanism works.** An unsigned app gets a second cDSP session:
  it opens lite in 22 ms and has its own 4 GiB (3584 MiB mapped beside
  S1's 3840). The whole FC set + lm_head (74 weights, 383 MiB) sits on
  its heap. A hop costs 2.9–13 µs through two dspqueues or 0.3–2.4 µs
  through a shared page. S1 keeps its HMX lock and VTCM through S2's
  lifetime, and with the heap rule above teardown leaves the app intact.
* **S2 has no VTCM.** S1's M=1 feed holds all 8 MiB. The exact FC is
  L2-fed at **8.06–8.08 ms/token**, against the CPU's 7.4; bit-identical
  to the spec (bad 0). Fed directly from DDR it takes 33 ms.
* **Projected end to end ≈ 21–22 ms/token (≈ 46 tok/s)**, against the
  hybrid A's 19.7 ms (50.6 tok/s, #158) — slower. The activation
  quantizer and the router are still assumed (HVX versions pending).
  Concurrent reads from both sessions hit the ≈ 70 GB/s ceiling (S1
  −17 %), so overlap does not close the gap.
* **The decision is the user's:** build the two-session E2E plan, stop at
  (c) of #132, or take the §3.5 fallback.

## Notes from the run

Serial `R3CY10WM83Y` in all three sittings. Sitting 1 (set, 23:20) and
set2 (23:28) as above; the phone was rebooted before set3 (00:26). Logs:
`/local/mnt/workspace/htp_moe/178/logs{,2,3}/`; each probe run's logcat
is kept next to it.

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
