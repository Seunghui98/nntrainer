# Measurement 201 S0: the expert pool on the path that already runs it (A, E0, F28, F16 warm / cold)

Branch `htp/201-pool-in-token` @ the commit that adds this file; artifacts
built from `htp_decode` @ `53f38aabf` (the branch adds only this handoff,
its runner and the evictor tool, no code the app or skel is built from).
Staged at `/local/mnt/workspace/htp_moe/201/s0/`. **Estimated device time:
≈ 55 min, reboot first** (+ ≈ 1 min a LEAK stop and its reboot).

**Run 2026-09-30 21:42–22:08 KST on `R3CY10WM83Y` (S25 Ultra), no stop,
read from `/local/mnt/workspace/htp_moe/201/s0/logs/` (`sitting.out`,
`speed.txt`, `sim.txt`, the per-run logs). Section "Sitting as read" below.**

## Sitting as read

Uptime 65 s at the start (rebooted), battery 100 %. MD5 OK on both ends.
Ceiling 3840 MiB after every one of the 36 runs; no `LEAK`, no
`0x8000040e`, no failed token. `expectation mismatches: 1`, the evict check
(below), which was a wrong check, not a wrong run.

* **Every text equals A r1 of its G** (`text=DIFF` count 0 over 30 cells):
  E0 and the pool at C = 28 / 16, warm and cold, are A's bits on silicon.
* **The misses explain most of the pool's cost.** Decode ms a token from
  the G = 64 profile runs: A ≈ 17.5, F28 20.7, F16w 23.4, F16c 74.4. Misses
  × (read + swap) account for 3.1 of F28's +3.2 ms (104 misses = 1.6 a
  token × 1.94 ms), 4.6 of F16w's +5.9 (648 = 10.1 a token × 0.46) and 48.8
  of F16c's +56.9 (× 4.83); the rest is not split. The simulator on A's G =
  1024 trace gives 0.48 misses a decode call at C = 16 (10.6 a token),
  matching F16's 10.1.
* **A warm miss is 0.40 ms (F16w) but 1.85 ms at F28.** Not explained.
  `page_cache_evict` (report mode, run after the sitting) found **2613 of
  the file's 4116 MiB** resident after the last run: the phone never holds
  the whole expert section, so "warm" is only partly warm, and a bigger
  arena (F28 3.3 GB against F16 1.9 GB) may leave less of it. That is a
  hypothesis for S2's warm cells to read (the S2 runner prints the file's
  resident MiB before each pool run), not a finding.
* **F16c is cold**: 4.76 ms a miss against F16w's 0.40 on the same 648
  misses. That, not the evict check, is the evidence. The check itself was
  wrong twice: `stat` read the symlink (`the file is 0 MB`) and the 2000 MB
  threshold on `/proc/meminfo` Cached assumed the whole file was cached
  (the drop was 558 MB). The runner now asks the tool for the file's own
  resident pages (mincore) before and after (`resident a -> b MiB of c`,
  pass when b < 1 % of c; on this phone after the sitting: `resident 2613
  -> 0 MiB of 4116 MiB`).
* **E0 is 12 ms a token behind A** (G = 512: 32.1 / 32.5 against 55.4 /
  55.4). Where it goes (E0 G = 512 r1): rt 29.2 ms = S2's wall 25.2 + the
  wake 4.1 (almost all `ret s2` 4.0: S2's answer to the ARM's blocking
  read); S2's FC + DENSE_FFN + LM_HEAD 11.07 ms against 8.06 isolated
  (#178: S2 has no VTCM); S1's MoE 10.2 ms (0.464 ms a round) + router 0.82.
* **Thermal**: each G block started at zone0 ≤ 35 °C, but within a block
  zone0 climbed to 52 / 62 / 65 °C (G 64 / 512 / 1024). The G = 1024 r2
  cells ran hot and read low for every variant (A 49.4 against r1 54.9);
  read G = 1024 from r1. S2's runner also cools between r1 and r2.
* **Prefill** (G = 512, r1 / r2): A 512 / 504, E0 548 / 392, F28 529 / 393,
  F16w 446 / 449. The r2 spread is thermal. F16c's prefill (344 / 354) is
  cold by construction.

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

## Results

### Speed (prompt 512; decode tok/s all, r1 / r2; last 64 in brackets where it differs)

| G | A | E0 | F28 | F16w | F16c |
|---|---|---|---|---|---|
| 64 | 57.30 / 55.90 | 30.70 / 31.83 | 54.94 / 52.89 | 44.82 / 45.42 | 12.81 / 13.42 |
| 512 | 55.36 / 55.42 (54.42 / 53.78) | 32.11 / 32.54 | 53.47 / 54.35 (55.41 / 54.51) | 44.38 / 45.90 | 12.72 / 12.77 |
| 1024 | 54.90 / 49.45 (52.33 / 45.04) | 31.36 / 31.62 | 54.44 / 47.81 (51.91 / 45.85) | 43.09 / 41.82 (40.74 / 35.22) | 11.72 / 11.38 |

Text = A r1 of the same G in all 30 cells. Peak RSS: A / E0 4.7–5.3 GB
(the resident experts), F28 / F16 0.83–0.91 GB.

### Pool (G = 64 profile runs, `NNTR_HTP_PROFILE=2`)

| variant | misses (decode calls 1408) | a token | file read ms a miss | swap rpc ms a miss | mapped arena MiB | peak RSS MB | decode tok/s (profiled) |
|---|---|---|---|---|---|---|---|
| F28 | 104 (0.07 / call) | 1.6 | 1.85 | 0.09 | 3328 | 843 | 48.37 |
| F16w | 648 (0.46 / call) | 10.1 | 0.40 | 0.06 | 1920 | 843 | 42.72 |
| F16c | 648 (0.46 / call) | 10.1 | 4.76 | 0.07 | 1920 | 814 | 13.44 |

### Hit rate per C (simulator on the A G = 1024 trace, 22 528 decode calls)

| C | 8 | 12 | 16 | 20 | 24 | 28 | 32 |
|---|---|---|---|---|---|---|---|
| miss / decode call, ours | 1.59 | 0.90 | 0.48 | 0.21 | 0.05 | 0.00 | 0.00 |
| hit %, ours | 60.2 | 77.6 | 88.0 | 94.7 | 98.8 | 99.9 | 100.0 |
| hit %, belady | 81.7 | 90.5 | 95.7 | 98.4 | 99.7 | 100.0 | 100.0 |

Read-ahead ceiling (previous token's top-(4+m) of the same layer): top-4
44.8 %, top-8 63.6 % covered.

### E0 per session and L0 (G = 512 r1)

```
graph[S1] per-kind pcyc/token: ROUTER_TOPK=1724753(0.822ms) MOE=21408265(10.206ms) | wall_ms/token=22.059 mhz=2098
graph[S2] per-kind pcyc/token: RMSNORM=813267(0.387ms) FC=12536569(5.972ms) CONV1D_GATE=1073936(0.512ms) QK_NORM=213166(0.102ms) ROPE=51525(0.025ms) ATTN_M1=1617489(0.770ms) ADD=275599(0.131ms) DENSE_FFN=4229761(2.015ms) LM_HEAD=6467801(3.081ms) | wall_ms/token=25.151 mhz=2099
graph[S1] moe pcyc/round=973103 (0.464 ms) router pcyc/round=78398; s2 fc+dense_ffn+lm_head ms/token=11.068 (isolated #178: 8.06); arm token_ms=29.233
L0 wake us/token disp s1=57.4 s2=63.3 s2_pkt=3.0 ret s2=4015.5 clk_resid=-0.2
L0 us/token rt=29232.7 s2_wall=25151.2 s1_wall=22059.5 wake=4081.5 hop_us s1=305.0 s2=327.5 arm_fwd=29242.7 arm_us=1624.7 arm_n=511
close tokens=512 hops/token=44.00 s1_served=512 s2_served=512 timeouts=0/0 stale=0/0 id_checked=0 id_mismatch=0
```

### Evict check, ceiling, thermal

`evict check: Cached 2007 -> 1449 MB (dropped 558 MB; the file is 0 MB)`
(the check was wrong, see "Sitting as read"). Ceiling 3840 MiB after all
36 runs. Therm: t0 21:42 zone0 34.3 °C; after G 64 / 512 / 1024 55.2 /
62.5 / 65.6 °C; end 61.4 °C; battery 100 % throughout.

## Text approval

Not a PPL sitting: every text equals A's (`every text == A r1 of its G = 0`
differing), so nothing needs approval. A's G = 64 r1 text, for the record:
"town has a single main street that climbs from the harbour to a stone
church at the top of the hill, and along it stand a bakery, a hardware
shop, two pubs, a post office that also sells fishing line, a small museum
that opens only on summer weekends, and a lifeboat station".

## Notes from the run

Run by the orchestrator on `R3CY10WM83Y`; a Note20 (`R3CN80CW3FY`) was also
attached. The runner then took the first `adb devices` entry when given no
serial; it now defaults to the S25's serial. Cooling waits of 30 s–3.5 min
before each G block.
