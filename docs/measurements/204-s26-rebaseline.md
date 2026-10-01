# Measurement 204: the S26 (v81) re-baseline on htp_decode, A / E0 / P28 / Q28 at 4 and 1 DMA queues

Tree: `htp_decode` + PR #206 (`htp/204-s26-port` @ `dc1898239`, the S26 stack)
+ PR #202 / #203 rebased on it (`htp/201-pool-miss-path` @ `4c0c20dca`; the
pool inside the E2E entry and one PD, which only #203 has). Artifacts built
from `4c0c20dca` with the **v81** skel, staged at
`/local/mnt/workspace/htp_moe/204/s26/`. Run by the agent on the S26 Ultra
(user, 2026-09-30: agents run the sittings; contract §4.1 adb row).
**Estimated device time: ≈ 60 min, reboot first.** Plan
`docs/plans/204-s26-port.md` §4 step 7 (G4, G5, G6).

## Why

The S26 replaces the S25 for #201. `htp_decode` carried none of the S26
stack, and on the v81 engine one DMA queue reads ≈ 28–34 GB/s against ≈ 62
on four (#177): the user's "the S26 runs without the DMA optimization". This
sitting gives the S26 its own A / E0 / P28 / Q28 rows on the new tree, with
the queue lever read inside the sitting (`NNTR_MOE_DMA_QUEUES=4` against
unset = 1, same binary). The S25 rows of `201-pool-baseline.md` /
`201-one-pd.md` stay that device's column; nothing here is compared with
them except as context.

## Variants (one binary set; environment only; the queue count is a column)

| variant | what |
|---|---|
| **A** | hybrid, nothing set (the unchanged reference, first in each half block) |
| **E0** | `NNTR_HTP_E2E=1`, two PDs, all experts resident |
| **P28** | E0 + `NNTR_MOE_CACHE_EXPERTS=28`, two PDs, the model file pre-read (warm) |
| **Q28** | P28 + `NNTR_HTP_E2E_PDS=1`: one PD, the FC set and lm_head beside the pool in S1 |
| q4 / q1 | `NNTR_MOE_DMA_QUEUES=4` prefixed / unset (the default, 1) |

Order (the runner): the five device gtests (G4), S1's ceiling, then

| block | runs |
|---|---|
| G 64, q4: cool, A E0 P28 Q28, cool, Q28 P28 E0 A | 8 |
| G 64, q1: cool, A E0 P28 Q28, cool, Q28 P28 E0 A | 8 |
| G 512, q4: cool, A E0 P28 Q28 | 4 |
| G 64 `NNTR_HTP_PROFILE=2`: Q28 at q4 and q1 (`dmaq=`, `DMA_FIRST`) | 2 |
| CPU `q40` G 64 (text control, read, not gated); ceiling at the end | 1 + 1 |

Over budget: drop the q1 r2 half block first, then G 512 to A + Q28 (plan
§4 step 7). G 1024 is not in this sitting (temperature-bound, rule 52); the
4-vs-1 delta at G 512 (G6) is therefore read on A only through #177's
G-flatness, not measured here — a follow-up cell if wanted.

Gates per row (the runner prints `OK` / `BAD`): text ≡ A q4 r1 of its G
(E0 / P / Q / q1 are A's bits); banner `moe m1 gemv: on (applied=0x1f03e1)
… dma_q=4` on q4 rows, `(applied=0x703e1) … dma_q=1` on q1 rows; A:
`dspq: on` once, no `s2: open`, `dspq: close … bad=0`; E / P / Q:
`levers=0x0`, `calls/token=1.00`, close `timeouts=0/0 stale=0/0 …
id_mismatch=0` with `hops/token=44.00` (E0, P28) or `0.00` and `pds=1`
(Q28), S2's close `unmap_fail=0 detach_fail=0`; S1's ceiling after every
run ≥ the start's (the S25 read 3840 MiB; the S26's value is read here,
not assumed). Profile rows: `dmaq=4.00` / `1.00`, `DMA_FIRST` ≈ 60 / ≈ 130
µs (#177 on the previous S26 unit). Device gtests: one `[  PASSED  ]`, no
`[  FAILED  ]`, no `0x8000040e`.

If Q28 does not load (`fastrpc_mmap(32 MiB)` on this unit's PD space), rerun
with pool C 24 (`run_s26.sh <serial> 24`, log names become P24 / Q24) and
say so in the table.

## Artifacts

| file (under `/local/mnt/workspace/htp_moe/204/s26/`) | md5 | built |
|---|---|---|
| app/libc++_shared.so | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 |
| app/libcausallm_core.so | `8f532d8a143a383d6cdba46755cd67ab` | `build_android.sh --htp --cache`; `NNTR_HTP_FORWARD_KINDS` ×2 |
| app/libccapi-nntrainer.so | `ad46760cde21a9617ada13e0f310092a` | `build_android.sh --htp --cache` |
| app/libnntr_hvx_skel.so | `26fdf25a7457b1b30226c079d83f0e56` | **v81**, `HEX_ARCH=v81 ./test/htp/build.sh`: `UNDEFINED SYMBOLS OK (62 runtime imports)`, `ARCH OK (V81)`, ELF flags 0x81 (the v79 build of the same tree: `addffa398075812ca7c848de94b841fe`, not staged) |
| app/libnntrainer.so | `407c5821c998c5851e9510870c696143` | `build_android.sh --htp --cache`; `NEEDED libsdkl.so, libcdsprpc.so` |
| app/libsdkl.so | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL `lib/6.4.0.1/armv8_android26` |
| app/nntrainer_causallm | `c106135d7074b3bd1129a17e0c57cb01` | `build_android.sh --htp --cache` |
| app/page_cache_evict | `42595651ef514e155887eb31f421b2dc` | NDK clang, `tools/htp/page_cache_evict.c` |
| app/prompt512.txt | `fc65c1588dc66dd764c7013fe96cbb75` | `docs/measurements/77-prompt512.txt` |
| app/unittest_hvx_attn | `9acf65ca558fa03cbf763f414b13b70e` | ndk-build (test/jni) |
| app/unittest_hvx_fc | `0842eb907cc5feb648e5fff71e8e679d` | ndk-build (test/jni) |
| app/unittest_hvx_mm_u8i4 | `b068d8230c45fc735a22dfd50f581292` | ndk-build (test/jni) |
| app/unittest_hvx_softmax | `a3ee7b62789ce933405b3fc3216398d1` | ndk-build (test/jni) |
| app/unittest_hvx_two_sessions | `172f55aede7c7c27b85de443df489de2` | ndk-build (test/jni) |
| run_s26.sh | `1cc9ea4cc5babde3951dcb3eb300bf75` | `docs/measurements/204-s26-run.sh` |

Models on the phone (unchanged; md5 printed by the runner into
`logs/md5_models.log`): NPU `q40-qs4cx-wh` `7b7867fab51845664c0050c0a837073e`,
CPU `q40` `d28f55c5bd7adeb8bf73b02de582eb88`. Only the skel differs from a
v79 set (same file name); its md5 and `ARCH OK (V81)` are the tell.

Rebuild recipe (workstation, from the checkout at `4c0c20dca`):

```
export HEX_ARCH=v81; source tools/htp/env.sh
bash nntrainer/tensor/htp_backend/generate_stub.sh      # the IDL changed under #203
./test/htp/build.sh                                      # UNDEFINED SYMBOLS OK (62 …), ARCH OK (V81)
(cd builddir && ninja install) && (cd Applications/CausalLM && ./build_android.sh --htp --cache)
(cd test/jni && $ANDROID_NDK/ndk-build NDK_PROJECT_PATH=. NDK_APPLICATION_MK=./Application.mk \
   APP_BUILD_SCRIPT=./Android.mk NNTRAINER_ROOT=$PWD/../.. HEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT \
   unittest_hvx_mm_u8i4 unittest_hvx_softmax unittest_hvx_attn unittest_hvx_fc unittest_hvx_two_sessions -j8)
bash docs/measurements/204-s26-stage.sh                 # refuses a non-v81 skel; writes md5.txt
```

## Steps

1. Reboot the S26, wait for the lock screen, `adb devices` → its serial.
2. `bash /local/mnt/workspace/htp_moe/204/s26/run_s26.sh <serial>` — the
   serial is required (no default; the 201 runners defaulted to the S25).
   It checks `md5.txt` on both ends (`MD5 OK`), sets `do_sample=false`,
   `bad_word_ids=[124900]` and `moe_engine=htp` on the phone's model
   configs, runs the gtests, then the blocks above. RESUMABLE: after a
   `STOP`, reboot and run the same command again.
3. Fill the tables below from `logs/speed.txt`, `logs/prof.txt`,
   `logs/pool.txt`, `logs/ceiling.txt`, `logs/therm.log`; commit on the
   branch.

## Results (fill in)

Device: serial `…`, model `…`, SoC `…`, uptime at start `…`, zone0 per block `…`.

### G4 device gtests (v81 skel)

| gtest | passed | failed | note |
|---|---|---|---|
| unittest_hvx_mm_u8i4 | | | |
| unittest_hvx_softmax (`HvxM1Ops`) | | | |
| unittest_hvx_attn | | | |
| unittest_hvx_fc | | | |
| unittest_hvx_two_sessions | | | |

### Decode tok/s (prompt 512; r1 / r2)

| G | queues | A | E0 | P28 | Q28 |
|---|---|---|---|---|---|
| 64 | 4 | | | | |
| 64 | 1 | | | | |
| 512 | 4 | | | | |

4-vs-1 at G 64 per variant (G6; #177 A path: +24.8 % at G 64 on the
previous unit): A … / E0 … / P28 … / Q28 …

Prefill medians per variant (gate ≥ −5 % of A; S26 prefill means carry
outliers, #185 G5): …

Reference on the S25 (other device, context only; `201-one-pd.md`, G 64):
A 56.74 / 54.51, E0 30.51 / 30.46, P28 36.72 / 35.81, Q28 42.98 / 43.13.
Previous S26 unit (`R5KL20NFRCK`, `htp_moe` tree, #185): A (= Q4t) 50.00,
DQ 52.09 at G 64; one queue ≈ 40–42 (#168).

### Profiles (G 64, Q28)

| queues | `dmaq=` | `DMA_FIRST` µs | M==1 `dsp` µs/call |
|---|---|---|---|
| 4 | | | |
| 1 | | | |

### Ceiling, text, temperatures

S1 ceiling at start / min after a run / end: … MiB. Texts ≡ A q4 r1: … of
19. CPU `q40` G 64 vs A: first differing word … (not gated). zone0 pre /
post per block: ….

## Text approval

Every NPU variant is gated on bit-equal text with A of the same sitting;
the user's reading is only needed if a row shows `text=DIFF`.

## Notes from the run

…
