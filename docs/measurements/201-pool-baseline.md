# Measurement 201 S0: the expert pool on the path that already runs it (A, E0, F28, F16 warm / cold)

Branch `htp/201-pool-in-token` @ the commit that adds this file; artifacts
built from `htp_decode` @ `53f38aabf` (the branch adds only this handoff,
its runner and the evictor tool, no code the app or skel is built from).
Staged at `/local/mnt/workspace/htp_moe/201/s0/`. **Estimated device time:
≈ 55 min, reboot first** (+ ≈ 1 min a LEAK stop and its reboot).

## Why

Plan `docs/plans/201-htp-decode-e2e-review-gemma-moe.md` §4 S0. Before the
pool moves inside the per-token entry (S1), read on this tree what a miss
costs (warm page cache against flash), how many misses a decode token pays
at C = 28 (the one-PD pool size) and C = 16, and the hit rate for every C
from one routing trace. S2's pool-size sweep and the one-PD verdict are
read against these numbers; §3.4 / §3.6's LFM column is re-read from them.

## Variants (one binary set; environment only)

| variant | what | reads |
|---|---|---|
| **A** | hybrid, nothing set (the unchanged reference, run first) | tok/s ×2 per G; plus one run at G = 1024 with `NNTR_MOE_TRACE=moe_trace.txt` (pulled; `tools/moe_expert_cache_sim.py` gives the hit rate for C = 8 … 32) |
| **E0** | `NNTR_HTP_E2E=1` (two sessions, all experts resident, levers 0) | tok/s ×2 per G; the per-kind and L0 lines at close — the E2E base on this tree |
| **F28** | A + `NNTR_MOE_CACHE_EXPERTS=28`; model file pre-read into the page cache before the run (warm) | tok/s ×2 per G; one `NNTR_HTP_PROFILE=2` run at G = 64: misses, ms a miss, swap RPC, mapped arena MiB, RSS; text against A |
| **F16w / F16c** | A + `NNTR_MOE_CACHE_EXPERTS=16`, warm (pre-read) and **cold** | as F28; F16c is the miss price from flash |

**How F16c is cold without root.** The phone refuses `drop_caches` (doc 52:
cold was never measured). `app/page_cache_evict` (`tools/htp/page_cache_evict.c`,
built with the NDK) calls `posix_fadvise(POSIX_FADV_DONTNEED)` on the model
file, which any process that can open the file may do; the kernel drops the
file's clean, unmapped pages. F16c starts it with a 20 ms interval before
the app and kills it after, so the page cache never holds the experts
during the run: every miss read (prefill and decode) comes from flash
except a re-read of one expert within 20 ms. This is colder than plan
§4 S0's wording ("evicted before the decode"; a one-time eviction would let
decode's own reads warm the cache again) — it measures the pure flash
price, the case Gemma's 11.4 GB expert section will be in. F16c's prefill
is therefore slow and is not read against A's. The runner checks the tool
once on the device (`evict check: Cached a -> b MB`, ≥ 2000 MB dropped
after the warm-up A run, which read the whole 4.1 GB file); verified on the
workstation host only (500 MB file, `Cached` fell 488 MB), **not yet on
the phone**.

Every text is expected to equal A's r1 of the same G: E0 is A's bits
(194 sitting 1b: nll 8/8 identical), and the pool only compacts the
expert arrays, which changes nothing numerically (the lfm25 host fixture
generates the same tokens at C = 1, 2 and unset). A `text=DIFF` is a
mechanism bug, not a wording drift; that row is void and reported. For that
reason no decode PPL column is run in this sitting: identity to A is the
stronger check, and no variant here changes DSP arithmetic.

## Artifacts (built on the workstation, SDK 6.4.0.1, HexKL beta.2 6.4.0.1, NDK r30, v79)

| file (under `/local/mnt/workspace/htp_moe/201/s0/`) | md5 | built with |
|---|---|---|
| app/libnntr_hvx_skel.so | `74d36befb63f0017eb1cbc4000fa1fce` | `test/htp/build.sh` (`UNDEFINED SYMBOLS OK (62 runtime imports)`) |
| app/nntrainer_causallm | `c106135d7074b3bd1129a17e0c57cb01` | `generate_stub.sh`, stale `nntr_hvx_stub.o` deleted, `(cd builddir && ninja install)`, `build_android.sh --htp --cache` |
| app/libcausallm_core.so | `974c0a80beef7884a793535b0e71aff9` | 〃 (`NNTR_HTP_FORWARD_KINDS` ×2, `NNTR_MOE_TRACE` ×1) |
| app/libnntrainer.so | `a757576ef15819f19669717f7828afad` | 〃 (`jni/obj/local`; `NEEDED libsdkl.so, libcdsprpc.so`) |
| app/libccapi-nntrainer.so | `ad46760cde21a9617ada13e0f310092a` | 〃 (`jni/obj/local`) |
| app/libc++_shared.so | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 sysroot |
| app/libsdkl.so | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL beta.2 `lib/6.4.0.1/armv8_android26` |
| app/unittest_hvx_two_sessions | `172f55aede7c7c27b85de443df489de2` | `ndk-build` (`TwoSessions.S1Ceiling`) |
| app/page_cache_evict | `04c111b5e8b0f7ff5347b12cfc1341e6` | `aarch64-linux-android26-clang -O2` of `tools/htp/page_cache_evict.c` |
| app/prompt512.txt | `fc65c1588dc66dd764c7013fe96cbb75` | `docs/measurements/77-prompt512.txt` |
| run_s0.sh | `7e22db5de21aef1209282115953799a6` | `docs/measurements/201-s0-run.sh` |
| NPU model `models/q40-qs4cx-wh` on the device | unchanged since #132 E5 | — |

`md5.txt` in the stage lists every file; the runner checks it on both ends
and stops on a mismatch. The skel's md5 names one build, not its sources
(`hexagon-link` stores randomly suffixed `/tmp/*.o` names).

**Rebuild recipe** (only if the stage is lost; from a checkout of this
branch, `source tools/htp/env.sh`, `export HEXKL_ROOT=$HOME/Qualcomm/hexkl-1.0-beta.2/hexkl_addon`):
`git submodule update --init --depth 1`; `libtokenizers_android_c.a` copied
from another checkout or `build_tokenizer_android.sh`; `libc++_shared.so`
from the NDK sysroot; then `./test/htp/build.sh`,
`bash nntrainer/tensor/htp_backend/generate_stub.sh`, `rm -f
builddir/obj/local/arm64-v8a/objs/nntrainer/**/generated/nntr_hvx_stub.o*`
(`--cache` does not recompile it: `transport failed: err=0xe` on the E
path), `(cd builddir && ninja install)` (`--cache` does not rebuild
nntrainer; without it the app fails to compile on `decode_row_resident`),
`(cd Applications/CausalLM && ./build_android.sh --htp --cache)`, the
`ndk-build … unittest_hvx_two_sessions`, then
`bash docs/measurements/201-s0-stage.sh`.

## Steps (workstation, phone on USB)

1. **Reboot the phone**; leave it idle until it is cool (the runner waits
   for zone0 ≤ 35 °C before each G).
2. `bash /local/mnt/workspace/htp_moe/201/s0/run_s0.sh [serial]` (any
   attached S25 Ultra; with no serial it takes the first `adb devices`
   entry). Everything goes to `logs/sitting.out`, one log per run in
   `logs/`. **Resumable per run**: a finished run leaves
   `logs/done/<run>`. It stops by itself on `0x8000040e` (stale skel /
   stub), `LEAK` (S1's ceiling < 3840 MiB after a run), a failed token
   (`AEE_EEXPIRED`), `FATAL`, or a run with no `generation:` line; then
   reboot and run the same command again.
3. Paste the summary (from `--- speed` to `=== done`) into Results, the
   G = 64 r1 texts into the text table, commit this file on the branch,
   push, set #201 to `state:measured`.

Order: md5 → config (greedy, `bad_word_ids [124900]`, `moe_engine htp`) →
ceiling → warm-up A (G = 64) → evict check → per G in 64 / 512 / 1024:
cool, `A E0 F28 F16w F16c F16c F16w F28 E0 A` → profiles at G = 64
(`NNTR_HTP_PROFILE=2`: F28, F16w, F16c) → A at G = 1024 with the trace →
summary (simulator replay on the workstation). S1's ceiling after every
run.

Expected lines: every A / F run `[HTP] dspq: on …` and no `s2: open`;
every E0 run `[HTP] ppl levers=0x0 L1=exact`, `s2: fc arena weights=67
handles=74 … s1_arena_mib=3840`, `token driver: close tokens=… hops/token=44.00
… timeouts=0/0 stale=0/0 … id_mismatch=0`, `calls/token=1.00`, `s2: close
… unmap_fail=0 detach_fail=0`; every profile run `[HTP-PROFILE] expert
cache misses: <n>, file read <ms> (<ms>/miss), swap rpc …` and `[HTP] arena
chunk … mapped total <MiB>`; the summary `every text == A r1 of its G … = 0`
and `expectation mismatches (this invocation): 0`.

## Results (fill in)

### Speed (prompt 512; prefill / decode all / decode last 64 TPS; peak RSS; text vs A r1)

| G | run | A | E0 | F28 | F16w | F16c |
|---|---|---|---|---|---|---|
| 64 | r1 | | | | | |
| 64 | r2 | | | | | |
| 512 | r1 | | | | | |
| 512 | r2 | | | | | |
| 1024 | r1 | | | | | |
| 1024 | r2 | | | | | |

Reference (not this tree): #194 sitting 1b A 56.21 / 54.89 (G = 512), E0
32.19 / 30.94; doc 52 C = 16 warm on another unit 57 / 73 / 85 % hits at
C = 8 / 12 / 16. Goal ≥ 50 decode; prefill never below −5 % of this
sitting's A (F16c excepted, its prefill is cold by construction).

### Pool (G = 64 profile runs)

| variant | misses (all / per decode call) | file read ms a miss | swap rpc ms a miss | mapped arena MiB | peak RSS | decode tok/s |
|---|---|---|---|---|---|---|
| F28 | | | | | | |
| F16w | | | | | | |
| F16c | | | | | | |

### Hit rate per C (simulator on the A G = 1024 trace, policy `ours` / `belady`)

| C | 8 | 12 | 16 | 20 | 24 | 28 | 32 |
|---|---|---|---|---|---|---|---|
| miss / decode call | | | | | | | |
| hit % | | | | | | | |

### E0 per session and L0 (G = 512 r1)

<paste the `graph[S1|S2] per-kind`, `moe pcyc/round`, `L0` and `close` lines>

### Evict check, ceiling, thermal

<paste>

## Text approval

Not a PPL sitting (see Why): every text must be A's. The user reads A's
G = 64 r1 text once for sanity; any `text=DIFF` row is void.

| variant | text = A r1 (G 64 / 512 / 1024) | generated text (G = 64, r1) |
|---|---|---|
| A | (reference) | <paste> |
| E0 | | |
| F28 | | |
| F16w | | |
| F16c | | |

## Notes from the run

<unit serial, battery, thermal, LEAK stops and reboots, anything stale>
