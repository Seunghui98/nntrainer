# Measurement 234 P4: Gemma-4 26B-A4B (dummy) one-PD decode with FC / DENSE_FFN on the FC WH sidecar

Branch `htp/234-p4-fcwh-decode` @ `a7fe77748`, PR #257. The agent ran
the sitting itself on `R3CY205ZMND` (SM-S938N, v79) on 2026-10-08
between 10:26 and 11:00 KST, about 34 min of device time including a
reboot and its 5-minute wait. `.sitting.lock` was taken at 01:26:23Z and
released at 01:59:49Z.

## Why

P4 binds the one-PD decode token's FC and DENSE_FFN ops to the FC WH
sidecar's handles. The LM_HEAD stays Q4M1. This sitting answers four
questions:

* Does Gemma-4 26B load with the sidecar under the **original**
  `config.json`? That is also #253's device confirmation.
* Are its tokens the hybrid's?
* What does it do to decode tok/s, the per-kind DSP time and the address
  space, against the sidecar-less one-PD token (S5-0's E16) read again
  in the same sitting?
* Whether S2 (plan 229, 2-bit FCs on this path) has a fast base to
  extend depends on these numbers.

**The weights are a DUMMY.** The text is meaningless and is not
reported. The misses/token column is the dummy's routing, and F and E
generate different streams, so their routing differs too.

## Artifacts (workstation, SDK 6.4.0.1, HexKL 6.4.0.1, NDK r30, v79)

All are built from `a7fe77748`. The app set is staged at
`/local/mnt/workspace/htp_moe/234/p4/app`. Its `md5.txt` matched the
device's `md5sum` before the sitting and again after the reboot.

| file | md5 | built with |
|---|---|---|
| libnntr_hvx_skel.so | `5cdd307e57fca0618f33a24da74dca48` | `test/htp/build.sh` (`UNDEFINED SYMBOLS OK (62 runtime imports)`, `ARCH OK (V79)`), stub unchanged |
| nntrainer_causallm | `d7e3da43d76a4817df5df9bbd699f1ad` | `build_android.sh --htp`, then `(cd builddir && ninja install)` + `--cache` |
| libcausallm_core.so | `4ae8676b8afb62911bcbed47a1999eeb` | same; `NNTR_HTP_FORWARD_KINDS` count 2 |
| libnntrainer.so (`jni/obj/local`) | `294aa73a76f55fcfd01c86267fee1326` | same; `NEEDED` libsdkl.so + libcdsprpc.so |
| libccapi-nntrainer.so (`jni/obj/local`) | `6f077d581b80326622b81ace2db8973e` | same |
| libc++_shared.so | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 |
| libsdkl.so | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL `lib/6.4.0.1/armv8_android26` |
| unittest_hvx_two_sessions (S1 ceiling) | `4d20ba3db0bc96f4fd06bc9ab340fdd7` | `test/jni` ndk-build |
| nntr_gemma4_qs2cx_wh.bin (device) | `6ac7df4c7a3a062ff80ec57c8cfa289a` | PR #251's repack of the dummy, 7,226,142,840 B. Its md5 is the same before and after the sidecar tool (workstation), and on the device |
| **nntr_gemma4_qs2cx_wh_fcwh.bin** (sidecar) | `aeab72fabb7302bc1a849e5f5f2b121f` | `python3 -I tools/htp/fc_wh_sidecar_from_q4.py /local/mnt/workspace/models/gemma4_26b <out> --lib build/nntrainer`: **images=205, 827,119,616 B (788.8 MiB), keys_unique=1**, 15.6 s. Device md5 matched |
| config.json (device) | `b2062bfe53fe2e805633f3e261cf1cdb` | **the original** (restored from `config.orig.json`; S5-0's 4096 copy is kept as `config.4096.json`) |
| nntr_config.json (`gemma4_26b`) | `2de9fd6a…` | user's file; only `num_to_generate` is edited per G |
| nntr_config.json (`gemma4_26b_fcwh`) | `a9789728…` (G = 64) | the same plus `"fc_wh_file_name": "nntr_gemma4_qs2cx_wh_fcwh.bin"`, `"fc_wh_format": "QS4CX_WH/1"`. The other files are symlinks to `../gemma4_26b/` |

**How the sidecar was made.** P3's path (`nntr_quantize_stream
--fc_wh_sidecar`) quantizes from the f32 source, and the 26B dummy has
none: its FCs exist only as the external converter's ARM Q4_0
(q4_0x4). The tool reads those bytes and re-quantizes each FC per column
with `htp_qs4cx_from_q4_0x4`, the backend's own load-time requant. So
the sidecar holds **a second quantization of already-4-bit weights**. A
sidecar written from a real f32 or ternary source would hold only one
(see the accuracy reading below). The image count is 25 sliding × 7 +
5 full × 6 (`attention_k_eq_v`, no `_wv`) = 205.

The runner is `docs/measurements/234-p4-run.sh` (this branch). The raw
logs are in `/local/mnt/workspace/htp_moe/234/p4/logs/`: `load_F16_G64`,
`ref_*`, `margin_*`, `optime_F16_G64`, `ceil_load`, and `sweep/` with its
`sweep.out`.

## Commands

From `/data/local/tmp/nntrainer/causallm/s234p4`:

```
# A16 (hybrid, the reference)
NNTR_MOE_CACHE_EXPERTS=16 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ../models/gemma4_26b
# E16 (one PD, no sidecar: S5-0's cell)
NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=16 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ../models/gemma4_26b
# F16 (one PD, the sidecar: "E16-fcwh")
NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=16 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ../models/gemma4_26b_fcwh
```

The prompt is the config's `sample_input`, which is **447 tokens** with
this tokenizer. Every cell is prompt 447. `NNTR_HTP_E2E_PDS` is not set.
The phone was rebooted before the sitting, and the work started 5 min
after boot (uptime 300 s).

## Step 1: Load with the original config.json (F16, G 64)

**It loads and runs.** No `max_position_embeddings` edit was needed, so
#253's fix holds on the device. A16 and E16 below also ran on the
original file. `load_F16_G64.log`:

```
[HTP] graph: description n_ops=542 resident=RMSNORM|FC|QK_NORM|ROPE|ATTN_M1|ADD|ROUTER_TOPK|MOE|DENSE_FFN|LM_HEAD moe_ops=30
[HTP] gemma: list n_ops=542 layers=30 params=60 by name
[HTP] fc wh: file=../models/gemma4_26b_fcwh/nntr_gemma4_qs2cx_wh_fcwh.bin handles=0 arena_kib=0 heap_kib=0 requant=0
[HTP] graph: q4m1 weights=206 handles=16 feed=vtcm wh_handles=435
[HTP] e2e: fc arena weights=206 handles=16 attach_mib=396.0 chunks=2 mapped_mib=448 feed=vtcm load_ms=2426.8 lanes=6,3 s1_arena_mib=704 s1_heap_kib=81619
[HTP] attn_m1: registered layers=25 kv=8 gqa=2 head_dim=256 max_seq=2048 cache=409600 KiB
[HTP] attn_m1: registered layers=5 kv=2 gqa=8 head_dim=512 max_seq=2048 cache=40960 KiB
prefill: 447 tokens, 3494 ms, 127.934 TPS
generation: 64 tokens, 5240 ms, 12.2137 TPS
peak memory: 2192964 KB
[HTP] token driver: pool misses=331 misses/token=5.17 miss_wait_us/token=6191.8 rounds=98 arm_ms/round=4.269 pgpgin_mib=395.7
[HTP] token driver: close tokens=64 hops/token=0.00 served=64 timeouts=0 stale=0 id_checked=0 id_mismatch=0 stop_err=0x0
[HTP] e2e: close mapped_mib=449.12 unmap_fail=0 detach_fail=0 heap_used_kib=603370 (info rc 0x0)
[HTP] arena: chunks unmapped 7/7 mib=1536 (release rc=0x0 put=7)
[HTP] graph: forward calls=64 tokens=64 calls/token=1.00
```

* `q4m1_handles` = **16**: the tied LM_HEAD's slices, the only Q4M1
  weight left. E16 has 221 handles for all 206 weights.
* `wh_handles` = **435**: every FC part and dense chunk pair on the
  sidecar's handles.
* `fc wh: … handles=0` is the banner at `finish_decode_graph_q4_0`,
  printed before the bind registers the images.
* **Address space.** The pool and the WH images share S1's arena:
  1536 MiB = pool 704 + one 832 MiB chunk for the FC images, which is
  `fcwhLeft`'s 788.8 MiB plus pads, rounded. The FC arena holds only the
  LM_HEAD: 448 MiB mapped (attach 396). Total ≈ **1 984 MiB** mapped,
  against E16's 704 + 1 344 = 2 048. The DSP heap at close is
  589–603 MiB (E16: 582). The S1 ceiling read 3840 after the run.
* Peak RSS 2142 MiB. E16 in this sitting reads the same, 2156–2165 MiB
  in every cell. The sidecar goes to rpcmem, and the CPU still holds its
  Q4_0 FCs for the prefill.
* C = 8 was not needed.

## Step 2: Reference gate (A16 vs F16, G 64)

Both runs had `NNTR_HTP_DUMP` and `NNTR_PPL_DECODE` (self) set, so
neither is a tok/s run (`ref_off_G64.log`, `ref_f_G64.log`).

| check | result |
|---|---|
| prefill MoE dumps, F vs off: every call F makes on ARM (warm-up + 30 prefill calls) | `E2E eval gemma26-prefill-f-vs-off files=62 bit_identical=1 min_snr_db=inf first_diff=-` |
| tokens over the whole run (65 ids, the prefill token included) | **60 / 65**. They differ at decode steps 1, 16, 33, 40 and 57; off has 126049 at each, F has 82982. So F's first 8 / 16 do **not** match: step 1 differs |
| off's margin top1 − top2 at those 5 steps, from off forced on its own ids with `NNTR_PPL_DECODE_ALTS=82982,126049` | 0.04095 / 0.00026 / 0.00762 / 0.01438 / 0.01225. rms(F − off) over the two logits is 1.284, so 2·rms = 2.568. **All 5 are expected mismatches by the plan 130 §3.5 policy, 0 unexpected.** The largest off margin anywhere in the run is 0.156 |
| null check: off forced on its own ids | `top1=64/64 nll_sum=177.49488952694784`. That is identical to the self run, and to S5-0's value to the last digit, so the hybrid is unchanged by this PR |
| F forced on off's ids | `top1=59/64 nll_sum=185.06547` (ppl 18.023 against off's 16.012) |

**Reading.**

* The prefill is bit-identical, as the plan expects: a Gemma prefill FC
  stays on the CPU, and the experts did not move.
* Every token difference is a near-tie flip under the repo's policy.
* **But the policy only passes because F's logits moved a lot.**
  rms(F − off) is 1.284 here. S5-0 read 0.130 for E. Forced on off's
  continuation, F's nll is +4.3 % (ppl +12.6 %), where E's was
  −0.01 %.
* At every differing step F prefers 82982 by 1.08–1.37 logits, where
  off had the two ids tied.
* On a dummy this is not an accuracy verdict. It is a flag.
* Two causes are known and could not be separated in this sitting:
  1. The sidecar here is a requantization of the dummy's Q4_0 (two
     4-bit quantizations stacked).
  2. The WH GEMV's u8 per-row activation quantization. The host
     fixture showed it on q / k feeding a peaked softmax (PR #257:
     sliding qkv alone 9.5 dB where o-proj, the dense FFN and full-layer
     qkv read 28–32 dB on exact weights).
* The real checkpoint's sidecar, written once from its source, removes
  cause 1. Cause 2 stays, and the S5 PPL handoff on the real files is
  where it is read.
* **If the per-token-entry PPL rule (+2 % of A) is applied literally,
  F fails it.** I did not apply it as a gate here because the weights
  are a dummy. That is my judgment, recorded so you can overrule it.

## Step 3: Speed cells (C = 16)

Sweep: `234-p4-run.sh R3CY205ZMND …/logs/sweep "512 64 1024" r1 16`.
Each G block starts cool (zone0 ≤ 35 °C) with A16, then runs F16 and
E16 back to back. So F and E start warm (57–64 °C, column below), and
A is the cool-start reference. To check the order, the G 512 block was
followed by a second cool block: E16 first (r2, cool), then F16 (r3,
warm). The S1 ceiling read **3840 after all 13 runs**.

Units:

* decode tok/s: over the whole generation.
* ms/miss = `miss_wait_us/token` / `misses/token`.
* decode pgpgin: the token driver's `pgpgin_mib`.
* run pgpgin / refaults: the `/proc/vmstat` delta around the whole
  process, load included.
* DSP wall: `wall_ms/token` of the per-kind line.
* q4m1 / wh handles: from the bind banner.

| run | prefill tok/s | decode tok/s | last 64 | peak RSS MiB | pool arena MiB | WH chunk MiB | FC arena MiB | DSP heap used MiB | misses/token (dummy routing) | ms/miss | decode pgpgin MiB | run pgpgin MiB | run refaults | zone0 start °C | DSP wall ms/token | calls/token | q4m1 / wh handles |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A16_G64_r1 | 134.3 | 19.24 | 19.24 | 2146 | 704 | – | – | – | – | – | – | 6794 | 1739155 | 34.9 | – | – | – |
| **F16_G64_r1** | 132.1 | **12.36** | 12.36 | 2159 | 704 | 832 | 448 | 589 | 5.17 | 0.90 | 256.7 | 7684 | 1966865 | 58.5 | 55.47 | 1.00 | 16 / 435 |
| E16_G64_r1 | 126.9 | 11.34 | 11.34 | 2159 | 704 | – | 1344 | 582 | 5.84 | 1.00 | 335.6 | 6962 | 1781957 | 58.1 | 62.15 | 1.00 | 221 / – |
| A16_G512_r1 | 126.2 | 18.37 | 17.21 | 2188 | 704 | – | – | – | – | – | – | 6842 | 1746350 | 34.5 | – | – | – |
| **F16_G512_r1** | 121.4 | **15.89** | 15.69 | 2165 | 704 | 832 | 448 | 589 | 0.71 | 1.10 | 315.9 | 7769 | 1988007 | 59.3 | 56.45 | 1.00 | 16 / 435 |
| E16_G512_r1 | 121.2 | 14.61 | 14.44 | 2165 | 704 | – | 1344 | 582 | 0.80 | 1.23 | 402.1 | 7059 | 1802091 | 57.4 | 61.77 | 1.00 | 221 / – |
| **F16_G512_r2** | 107.2 | **15.82** | 15.62 | 2159 | 704 | 832 | 448 | 589 | 0.71 | 1.06 | 291.0 | 7678 | 1965548 | 59.7 | 56.19 | 1.00 | 16 / 435 |
| E16_G512_r2 (cool, first) | 128.7 | 14.68 | 14.51 | 2156 | 704 | – | 1344 | 582 | 0.80 | 1.29 | 438.9 | 7082 | 1810387 | 34.9 | 61.87 | 1.00 | 221 / – |
| F16_G512_r3 (after E r2) | 108.4 | 15.90 | 15.71 | 2156 | 704 | 832 | 448 | 589 | 0.71 | 1.20 | 350.5 | 7827 | 2003465 | 57.0 | 56.11 | 1.00 | 16 / 435 |
| A16_G1024_r1 | 121.5 | 17.40 | 14.57 | 2295 | 704 | – | – | – | – | – | – | 7031 | 1787933 | 34.9 | – | – | – |
| **F16_G1024_r1** | 99.0 | **14.86** | 13.80 | 2159 | 704 | 832 | 448 | 589 | 0.37 | 1.18 | 339.7 | 7817 | 1997464 | 64.3 | 61.48 | 1.00 | 16 / 435 |
| E16_G1024_r1 | 114.3 | 13.65 | 12.97 | 2159 | 704 | – | 1344 | 582 | 0.41 | 1.23 | 417.1 | 7051 | 1804475 | 60.1 | 67.46 | 1.00 | 221 / – |

A16's pool arena is the hybrid's MoE pool, 3 chunks = 704 MiB, from its
`arena: chunks unmapped` line.

**Headline (one PD, C = 16, G 512): F16 15.89 / 15.82 / 15.90 tok/s
(three runs, spread 0.5 %), against E16 14.61 / 14.68: +8.4 %.** It
holds at every G: G 64 +9.0 %, G 1024 +8.9 %. The order check holds too:
E16 started cool and first gives 14.68, the same as warm (14.61). **The
hybrid A16 is still faster: 18.37 at G 512, and F16 is −13.5 % below
it** (G 64 −35.8 %, G 1024 −14.6 %).

Prefill is not a gate here (㊶). F and E start warm, after A. The F
cells after an E run (r2, r3) read 107–108 at zone0 57–60 °C; E16 r2,
started cool, reads 128.7. The prefill path is the same code in F and E
(CPU FCs, HTP MoE), so the gap is thermal order, not the sidecar.

Per kind on the DSP, ms/token, from the `graph per-kind` lines:

| run | RMSNORM | FC | QK_NORM | ROPE | ATTN_M1 | ADD | ROUTER_TOPK | MOE | DENSE_FFN | LM_HEAD | DSP wall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F16_G64_r1 | 1.751 | **11.033** | 0.659 | 0.225 | 9.562 | 0.355 | 4.242 | 13.341 | **5.456** | 8.576 | 55.469 |
| E16_G64_r1 | 1.769 | 15.008 | 0.518 | 0.228 | 9.573 | 0.222 | 4.255 | 14.082 | 7.658 | 8.562 | 62.151 |
| F16_G512_r1 | 1.665 | **10.649** | 0.630 | 0.216 | 13.736 | 0.350 | 4.038 | 11.394 | **5.346** | 8.184 | 56.447 |
| E16_G512_r1 | 1.713 | 14.327 | 0.493 | 0.219 | 13.701 | 0.217 | 4.055 | 11.295 | 7.323 | 8.165 | 61.770 |
| F16_G512_r2 | 1.675 | 10.670 | 0.632 | 0.216 | 13.719 | 0.351 | 4.030 | 11.073 | 5.400 | 8.180 | 56.193 |
| E16_G512_r2 | 1.709 | 14.324 | 0.491 | 0.218 | 13.713 | 0.217 | 4.072 | 11.388 | 7.308 | 8.174 | 61.873 |
| F16_G512_r3 | 1.684 | 10.638 | 0.632 | 0.215 | 13.771 | 0.350 | 4.034 | 11.104 | 5.240 | 8.184 | 56.108 |
| F16_G1024_r1 | 1.843 | **11.111** | 0.641 | 0.234 | 17.728 | 0.369 | 4.225 | 11.314 | **5.521** | 8.207 | 61.484 |
| E16_G1024_r1 | 1.909 | 14.521 | 0.503 | 0.243 | 18.011 | 0.244 | 4.349 | 11.765 | 7.382 | 8.218 | 67.459 |

**Where the 5.5 ms/token went** (G 512, mean of the F runs against the
mean of the E runs):

| kind | E16 ms | F16 ms | change |
|---|---|---|---|
| FC | 14.33 | 10.65 | **−3.67** |
| DENSE_FFN | 7.32 | 5.33 | **−1.99** |
| QK_NORM | 0.49 | 0.63 | +0.14 |
| ADD | 0.22 | 0.35 | +0.13 |
| LM_HEAD | 8.17 | 8.18 | unchanged (still Q4M1) |
| MOE | 11.34 | 11.19 | within noise |
| ATTN_M1 | 13.71 | 13.74 | within noise |
| DSP wall | 61.82 | 56.25 | −5.57 |

QK_NORM and ADD rise by +0.13–0.14 ms in every F cell. They are not
weight kinds, so this is probably a cache or VTCM-residency side effect
of the WH GEMV. That is unmeasured; only the per-kind lines show it.

The ARM side, from one `NNTR_OP_TIME=1` run (F16 G64,
`optime_F16_G64.log`, 12.26 tok/s; not a table cell):

* `output_of_causallm` (the token call) is 4139 ms over 64 calls. That
  is 64.7 ms/token with the first, or 61.7 ms/token without its 253 ms.
* `logit_softcapping` adds 1.13 ms/token.
* The `mha_core` max-equals-sum pattern of S5-0 is unchanged: it is the
  first decode token's one-time KV seeding.

## What did not run, and why

* **The CPU-only run.** As in S5-0, the dummy's FC Q4_0 is ARM
  interleaved, and the brief takes the hybrid A as the reference.
* **Prompt 512.** The config's `sample_input` is 447 tokens.
* **Second runs at G 64 and G 1024**, and a second A16 at G 512.
  Budget; the headline cell has 3 F runs and 2 E runs.
* **The PPL / dump gate at G 512.** It ran at G 64 only, as in S5-0.
* **C = 8.** C = 16 loaded, so the fallback was not needed.
* **The 4096 config copy.** The original loaded, so the fallback was not
  needed.
* **Text column.** The weights are a dummy.
* **Isolating the accuracy shift** (requant sidecar vs u8 activation
  quantization). Separating them needs a sidecar written from a source,
  which the dummy does not have.

## Notes from the run

* The phone was rebooted before the sitting (uptime was 17 h on the
  boot of S5-0's reboot). The work started 5 min after boot. Battery
  100 %, USB powered, zone0 26.7 °C before the reboot.
* `NNTR_HTP_DUMP` needs its directory to exist. The first two reference
  runs stopped with `NNTR_HTP_DUMP: cannot write …` before any decode,
  and were re-run after `mkdir`.
* `/data` had 191 GB free. The device keeps
  `models/gemma4_26b/config.json` as the **original** (b2062bfe), with
  `config.4096.json` and `config.orig.json` beside it, and
  `models/gemma4_26b_fcwh/` (sidecar + symlinks) for the next sitting.
