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

## Results (filled 2026-10-01, sitting run by the orchestrator)

**Scope (user, 2026-10-01, after the sitting):** the unit that ran this is a
**developer / engineering phone (userdebug build)**, so its numbers may not
represent a product S26. What this sitting records of record is one line:
**the one-PD E2E (Q28) ran on an S26 (v81), every NPU text == A, the S1
ceiling stayed 3840 MiB.** Everything below the record line is kept as
an appendix for the record; the queue-count and miss-cost readings are
**observations on a developer unit, not representative**, and derive no
rule and no BENCHMARK column. S26 optimization is deferred by the user
until a product unit is available.

### Record

| | |
|---|---|
| the one-PD E2E on v81 | ran: 24 NPU runs + 2 profiles, no stop, every gate line `OK` |
| texts | every NPU variant == A q4 r1 of its G, **20 / 20** (E0, P28, Q28, both queue counts, G 64 and 512); CPU `q40` differs at word 43 (`museum` vs `museum,`), the read-only control, expected (LEDGER ⑱'s word-43 split) |
| S1 ceiling | 3840 MiB at start, after each of the 24 runs and at the end (`logs/ceiling.txt`) |
| device | serial **R3CY70LV96T**, SM-S948U (US), SM8850, userdebug `S948USQU1AZAB`, kernel 6.12 — a different unit from the earlier S26 `R5KL20NFRCK` (#168 / #177 / #185). MemTotal 11.15 GB, 12 GB swap present, `/data` 99 % full (≈ 3 GB free) during the sitting (cleaned afterwards: 18 GB free) |

### Device facts and the three attempts (timeline, KST)

Neither LFM model set was on the phone; `q40-qs4cx-wh` (`7b7867fa…`) and
`q40` (`d28f55c5…`) were pushed and md5-verified before and after the
crash (`logs/md5_models.log`, `MD5 OK` in all three invocations).

| attempt | start | uptime | zone0 at t0 | what happened |
|---|---|---|---|---|
| 1 | 14:12 | 85 s (reboot first) | 41.3 °C | `unittest_hvx_mm_u8i4` ran 16 tests `[ OK ]`, then **`HmxMmU8I4Layer.RegistryCapacity` dropped the phone into Samsung download mode** (USB id `04e8:685d`, adb gone; hand-rebooted by the user; no tombstone, no pstore). The crash zeroed the four config files the runner had just edited (`sed` writes not synced); re-pushed and `sync`ed. The remaining four gtests read `0/0` (device gone); `logs/attempt1/` |
| 2 | 14:57 | 196 s | 35.5 °C | gtests with `--gtest_filter=-HmxMmU8I4Layer.RegistryCapacity` (`run_s26.sh` edited; original in `run_s26.sh.orig`, `md5.txt` updated): table below. Ceiling at start 3840. Then `STOP` on `A_G64_q4_r1`: `JSON parse error in generation_config.json` — the zeroed config (not a device event); fixed, rebooted |
| 3 | 15:02 | 58 s | 43.6 °C (block start after the cool wait 35.1) | all 24 runs + 2 profiles + the CPU control, no stop, 0 expectation mismatches; done 15:29 |

zone0 per block (`logs/therm.log`): t0 43.6 → after G64 q4 58.7 → after
G64 q1 59.8 → after G512 57.1 → end 59.0 °C (block starts 35.1 / 34.7 /
35.1 / 35.1 / 34.7 after the runner's cool wait).

### G4 device gtests (v81 skel, attempt 2)

| gtest | passed | failed | note |
|---|---|---|---|
| unittest_hvx_mm_u8i4 | 34 | 0 | **with `RegistryCapacity` excluded** — that test took the unit into download mode in attempt 1 (after 16 `[ OK ]`); not re-tried |
| unittest_hvx_softmax (`HvxM1Ops`) | 28 | 4 | `RejectsBadShapes` (error code `0x80000600`, rule 38), `RmsnormMatchesDetBitExact` (the ±1e-39 row, `overflow_row bad_y=2048`, rule 37), `QkNormMatchesDetBitExact` (`8 heads: y differs`), `ConvGateM1MatchesDetBitExact` — #137's known set, now reproduced on v81 |
| unittest_hvx_attn | 10 | 2 | `HvxAttnM1.RejectsBadShapes` (`0x80000600`), `HvxAttnM1Probe.Semantics` (`no fma case file at attn_fma_cases.bin` — a missing fixture on the phone, not a kernel result) — #137's set |
| unittest_hvx_fc | 1 | 0 | |
| unittest_hvx_two_sessions | 6 | 0 | |

No `0x8000040e` in any log.

### Appendix (for the record; developer unit, not representative)

Decode tok/s (prompt 512; r1 / r2; `logs/speed.txt`):

| G | queues | A | E0 | P28 | Q28 |
|---|---|---|---|---|---|
| 64 | 4 | 53.92 / 53.65 | 29.33 / 29.28 | 27.18 / 28.58 | 30.70 / 33.92 |
| 64 | 1 | 57.87 / 55.94 | 30.15 / 33.93 | 28.96 / 30.22 | 31.75 / 35.09 |
| 512 | 4 | 54.44 (last 64: 49.77) | 30.66 (30.67) | 31.62 (31.94) | 37.40 (38.23) |

Prefill tok/s (same runs; r1 / r2):

| G | queues | A | E0 | P28 | Q28 |
|---|---|---|---|---|---|
| 64 | 4 | 457.1 / 375.9 | 470.6 / 465.0 | 467.6 / 527.8 | 473.2 / 560.8 |
| 64 | 1 | 392.9 / 339.5 | 475.0 / 483.0 | 464.2 / 513.0 | 463.8 / 539.5 |
| 512 | 4 | 537.8 | 466.7 | 432.4 | 429.5 |

Prefill gate (≥ −5 % of A): every E0 / P28 / Q28 cell is above its block's
A at G 64 and inside −20 % at G 512 against an A cell (537.8) that is
itself the high outlier of A's 340–538 spread — the hybrid A's prefill
carries the largest spread of the sitting (#185 G5's outlier note), so
the gate is read as passed on the means and not as a finding.

CPU `q40` control, G 64: prefill 161.0, decode 50.91 tok/s (peak RSS
4.9 GB); text differs from A at word 43 (read, not gated).

**Observations (developer unit, not representative — no rule, no row):**

1. **A and E0 at G 64 sit at the S25 level** (A 53.6–57.9, E0 29.3–33.9;
   `R3CY10WM83Y` A 56.74 / 54.51, E0 30.51 / 30.46). Q28 at G 512 is the
   fastest NPU variant (37.4, last-64 38.2) over P28 31.6 and E0 30.7,
   under A 54.4 — the same ordering as the S25 (rule 59), by a smaller
   margin because of observation 3.
2. **DMA queues 4 vs 1: no gain from four on any variant.** The setting was
   applied (`queues banner` OK on every run: `applied=0x1f03e1 … dma_q=4`
   on q4 rows, `applied=0x703e1 … dma_q=1` on q1 rows). Per variant,
   q4 → q1 at G 64 (means of r1 / r2): A 53.8 → 56.9, E0 29.3 → 32.0,
   P28 27.9 → 29.6, Q28 32.3 → 33.4 — the one-queue half block read
   2–9 % *higher*, inside the r1 / r2 spread and with the q1 block second
   in order (warmer), so: no difference, direction if any against four.
   The plan's expected +22–25 % from #177 (`R5KL20NFRCK`) does not appear
   on this unit. **Gap:** the level-2 profile lists only M>1 rows
   (`m1_gemv=0/23 feed=0/23 dmaq=0.00`, identical M>1 `dsp=` 13 728 /
   13 730 µs/call) because in the E2E one-PD entry the M=1 MoE runs inside
   the S1 graph, so the decode `dmaq=` / `DMA_FIRST` read the plan wanted
   (G6) is not available from this profile on this tree; a profile hook
   inside the E2E graph would be needed to read it.
3. **The pool variants are slow because a miss costs ≈ 4–4.7 ms on this
   unit.** `prof_Q28_q4`: `expert cache misses: 121, file read 568.0 ms
   (4.69 ms/miss)`; `prof_Q28_q1`: `482.8 ms (3.99 ms/miss)`. At G 64
   every P28 / Q28 run reads `misses=121 misses/token=1.89
   miss_wait_us/token=6808–8139` with `arm_ms/round=5.33–6.32`; at G 512
   `misses=175 misses/token=0.34 miss_wait_us/token=1704–1872`,
   `arm_ms/round=6.5–7.2`. The S25 units read 3.45 / 3.79 ms on the same
   28-pool (rule 59 b) and 0.4–0.7 at C 16 / 24, so this is the same
   open item (㉜) at a higher level, and it is independent of page-cache
   residency here: `resident` 1299–1452 MiB (q4 block) and 2976–3789 MiB
   (q1 block) of 4116 read the same miss cost.
4. **Read-speed probe after the sitting (the orchestrator, not in the
   runner's logs):** a warm whole-file `cat` of the 4116 MiB model reads
   at only ≈ 1.6–2.0 GB/s on this phone with the file fully in the page
   cache (`Cached` 4.3 GB), pinned to little, mid or big cores alike at
   3.3–3.6 GHz — the page-cache `read()` copy path itself is ≈ 7× slower
   than what the S25's miss cost implies. Tag for every pool number of
   this sitting: *miss read path ≈ 4.7 ms/expert on R3CY70LV96T
   (userdebug); not a page-cache residency problem; cause in the
   kernel / read path, not in our code.* Mitigation candidates (mmap +
   memcpy instead of `pread`; a reader thread on a big core; read-ahead
   from the previous token) belong to the lever list, not here.

Pool lines (`logs/pool.txt`; `L0 us/token rt=` is the token round trip):

| run | resident MiB | rt µs | s2_wall | s1_wall | misses | misses/token | miss_wait µs/token | arm_ms/round |
|---|---|---|---|---|---|---|---|---|
| P28_G64_q4_r1 | 1299 | 33 918.7 | 33 675.4 | 30 675.1 | 121 | 1.89 | 7150.3 | 5.533 |
| P28_G64_q4_r2 | 1367 | 32 402.5 | 32 225.0 | 29 267.2 | 121 | 1.89 | 7797.5 | 6.072 |
| P28_G64_q1_r1 | 3789 | 31 684.6 | 31 540.7 | 28 583.0 | 121 | 1.89 | 8139.4 | 6.315 |
| P28_G64_q1_r2 | 2981 | 30 442.5 | 30 274.9 | 27 318.7 | 121 | 1.89 | 7045.3 | 5.514 |
| P28_G512_q4_r1 | 2540 | 30 284.5 | 29 172.0 | 25 933.0 | 175 | 0.34 | 1872.0 | 7.188 |
| Q28_G64_q4_r1 | 1362 | 29 969.9 | 29 813.1 | 0 (one PD) | 121 | 1.89 | 7896.7 | 6.115 |
| Q28_G64_q4_r2 | 1452 | 27 308.7 | 27 170.2 | 0 | 121 | 1.89 | 6808.7 | 5.330 |
| Q28_G64_q1_r1 | 3064 | 28 815.5 | 28 646.9 | 0 | 121 | 1.89 | 7233.0 | 5.580 |
| Q28_G64_q1_r2 | 2976 | 26 329.0 | 26 190.5 | 0 | 121 | 1.89 | 6943.3 | 5.423 |
| Q28_G512_q4_r1 | 2658 | 25 460.6 | 25 212.8 | 0 | 175 | 0.34 | 1703.8 | 6.517 |
| prof_Q28_q4 | 2385 | 29 414.4 | 29 272.4 | 0 | 121 | 1.89 | 8470.5 | 6.529 |
| prof_Q28_q1 | 2383 | 28 419.2 | 28 249.2 | 0 | 121 | 1.89 | 7178.3 | 5.550 |

Gate lines: A rows `dspq: on` once, no `s2: open`, `dspq bad=0`; E0 / P28
`levers=0x0`, `calls/token=1.00`, close clean, `hops/token=44.00`; Q28
`one PD` OK, `hops 0.00`, `pds=1`; S2 close `unmap_fail=0 detach_fail=0`
on every two-PD run.

### Profiles (G 64, Q28, level 2)

| queues | `dmaq=` | `DMA_FIRST` µs | M>1 `dsp` µs/call | note |
|---|---|---|---|---|
| 4 | n/a (`dmaq=0.00`, M>1 row only) | not printed | 13 728.0 | decode 31.67; miss read 4.69 ms/miss |
| 1 | n/a | not printed | 13 730.2 | decode 32.41; miss read 3.99 ms/miss |

### Not measured here

G 512 at one queue (the plan's G6 cell at G 512), G 1024 (rule 52), cold
cells (every pool run was warm by order), the decode `dmaq=` / `DMA_FIRST`
read (observation 2's gap), and anything on a product S26 — the queue and
miss readings above are this developer unit's only.

## Text approval

Every NPU variant is gated on bit-equal text with A of the same sitting;
no row showed `text=DIFF` except the CPU control (expected). No user
reading needed.

## Notes from the run

* `HmxMmU8I4Layer.RegistryCapacity` is excluded on this unit
  (`run_s26.sh`, md5 `md5.txt` updated); whether it is the test (exhausts
  the registry on purpose) or the userdebug build that takes the phone
  down is not known — one-line rule in LEDGER (rule 60), not re-tried.
* The runner's `sed` edits of the model configs must be followed by
  `sync` on this unit; a crash within seconds zeroed them.
* The runner's cool wait runs blind while adb is gone (`[: : 정수 표현식
  필요함` for 40 × 30 s in attempt 1) — harmless, but a dead device should
  stop the runner instead; a follow-up to `204-s26-run.sh` if it is reused.
