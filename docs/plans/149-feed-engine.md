# Plan 149: MoE feed-engine gap (LEDGER ㉓), bit-preserving

Issue #149 (p1), part of #76. Base `htp_moe` @ `8860933c` (code = `bb845426`).
This is lever (2) of the 2026-09-28 direction change: only when and in what
order the expert weights are fetched may change, never the arithmetic.

**Short version.** The archived #120 A trace already shows that the ring
never idles (§3.1). So the issue's three candidates are worth about 0–3 µs
per call, and two of them are already in the tree. The ≈ 86 µs gap is the
engine's per-byte rate inside the app, not the issue schedule. Step 0 is a
device read with no code change. It decides whether anything here is worth
building (§4).

## 1. Goal and gate

Acceptance (issue), made measurable:

| gate | read from | pass |
|---|---|---|
| `mm` at M==1 | level-2 `[HTP-PROFILE] K=2048 N=2048 M==1` row, G=64 | **≤ 630.0 µs**, only in a sitting whose anchor (`DMA_REPLAY workers=1 load=0 pace=0 fresh=0 gap_us=0`) reads **≥ 36.0 GB/s**. Below 36, the `mm` half is not decided and only the decode half is read (rule 34 (c)). |
| bit identity, kernel | device gtest `MoeLayerM1GemvMatchesHmx`, extended ×sched | `bad_elems` 0, `bit_identical yes` |
| bit identity, model | `NNTR_HTP_DUMP` of A and of each variant, same prompt, G=64; `tools/htp/htp_dump_eval.py <A> <V>` | `E2E eval … bit_identical=1` for every variant |
| text | the multi-prompt set (§4 step 5): `prompt512.txt` at G 64/512/1024 × 2 runs, plus three short prompts at G=64 | the generated text is byte-identical to A's in every cell |
| decode | E2E mean of two mirrored runs, per G | variant ≥ A at G 64, 512 **and** 1024 |
| prefill (standing) | E2E prefill tok/s; tie-breaker M>1 `dsp` (rule 27) | ≥ −5 % of A |
| host (standing) | §4 host gate | all pass lines, unchanged counts where noted |

The contract's standing text gate ("identical to the CPU run") is read here
as the 2026-09-28 decision: identical to A, the switch-off run of the same
NPU model. Bit-identical dumps make that true for any prompt.

## 2. Where it lives

How the ring line is produced (verified on this tree):

* DSP: `hexkl_dma_trace.c` records, only while `hexkl_probe_on` (the
  `_timed` entry, level ≥ 2), each push (issue tick, depth before it, a
  done bracket `[t_done_lo, t_done_hi]`) and each wait. The M=1 feed calls
  it from `moe_m1_push` / `moe_m1_wait` (`hmx/hexkl_mm_u8i4_moe.c:838`,
  `:856`). The summary fields are defined in `hmx/hexkl_dma_trace.h:127-138`.
  `busy` is the union of `[t_issue, t_done]` over the call's descriptors.
  `first_ready` is expert 0's gate_up `t_done_hi`. `last_issue` is the
  last weight push's `t_issue`. Every time is measured from
  `hexkl_dma_trace_reset` (`:1285`).
* ARM: `HtpProfile` sums them per bucket and prints the means per call.
  The code is at `nntrainer/tensor/htp_backend/htp_compute_ops.cpp:723-761`.
  `engine lo..hi = KB·1.024 / busy_hi .. / busy_lo`. The `[HTP-DMA]`
  per-descriptor lines (`:407-456`) are printed for the first
  `NNTR_HTP_DMA_TRACE` (default 3) calls of each bucket. They are read
  back by `nntr_hvx_moe_dma_trace_read` (`:2712-2730`). At level 3 the
  printed trace is the last of five repeats on the same input, i.e. a
  warm call, and the ring row keeps the fastest repeat (`:2633-2672`).

What the #117 / #120 signatures mean:

* `desc=10`: copy-in (8 KiB), gate_up e0..e3 (3.5 MiB each, one 2D
  descriptor, 64 × 56 KiB), down e0..e3 (1.75 MiB each, 56 × 32 KiB),
  copy-out.
* `first expert ready at 122 µs`: gu0 is pushed at t ≈ 3–6 µs, after the
  copy-in. Stage A of expert 0 cannot start before the whole slab lands,
  but the engine goes straight on with gu1. So this is latency, not
  engine idle.
* `depth max=4`: dn0..dn3 are outstanding together after A(3) pushes
  dn2/dn3.
* `last issue at 500 of 712`: dn2/dn3 go out after A(3). The engine is
  still busy with dn0/dn1 at that point.
* `busy=639..682` vs `mm` 677: the engine is busy for all of `mm`.

What changes (all behind one switch, §3):

| file | change |
|---|---|
| `hmx/hexkl_mm_u8i4_moe.h:290-363` | `HEXKL_MOE_FLAG_FEED_SCHED_SET 0x10u`; field bits 18 `TAIL`, 19 `TOUCH`, 20 `CLOCK`; `HEXKL_MOE_FLAGS_KNOWN` extended; `hexkl_moe_flags_feed_sched()` |
| `hmx/hexkl_mm_u8i4_moe.c:1308-1321` (first pushes), `:1454-1472` (stage C), `:838-863` (push/wait helpers) | EARLY + TAIL (§3.2), TOUCH in the wait loop |
| `htp_moe_opts.h:23-124` | `HTP_MOE_FLAG_FEED_SCHED_*`, `NNTR_MOE_HTP_FEED_SCHED` parsed into the word, `HTP_MOE_FEED_SCHED_DEFAULT 0u` |
| `htp_compute_ops.cpp:1745-1795` | read the env; banner gains ` sched=<n>` |
| `test/htp/nntr_hvx_mm_u8i4.c:1164-1178` | set_opts: on `CLOCK`, one extra `HAP_power_set` (min corner TURBO); `#ifdef NNTR_HVX_HAVE_HAP_POWER`. The open-time vote is at `test/htp/hvx_add_f32.c:126-180`. |
| `test/htp/host/moe_layer_host_check.c:552-560, 670-691, 798-860` | scoreboard knows column-half pushes; sched loop; TOUCH bounds |
| `test/htp/host/moe_opts_host_check.c` | the words below |
| `test/htp/host/run_inproc_e2e.sh:150, 219` | one `sched` run + `E2E eval feed-sched … bit_identical=1` vs golden |
| `test/unittest/unittest_hvx_mm_u8i4.cpp:2374, 2439, 2556` | `MoeGemvOpts(…, sched)`; ×sched in the bit-identity test; `sched=` on the `vtcm` bench cell |

Contract consumers that do **not** move, checked:

* The IDL and stub: `moe_set_opts(in uint32 flags, rout uint32 applied)`
  (`nntr_hvx.idl:478`) already carries the word.
* `HTP_MOE_N_STAGES` and the profile stage tables: no new slot.
* `tools/htp_fc_report.py` does not parse `DMA ring` or `applied`.
* The `QS4CX_WH` layout, the quantizer's format tag and the loader check:
  no byte moves.
* The HMX path's `IN-SITU CHUNK PLAN MATCHES KERNEL (46 descriptors)`:
  flags 0, untouched.

One visible change: the unset word becomes `0x303f1` (`0x303e1 | SCHED_SET`).
B = `0x343f1`, C = `0x3c3f1`, D = `0x1343f1`. A skel without the bits echoes
a different word, and the ARM side throws (rule 21). Old handoff greps for
`0x303e1` do not apply to sittings after this lands.

## 3. Design

### 3.1 What the archived data already say

Per-descriptor rates were read from the `[HTP-DMA]` lines of
`/local/mnt/workspace/htp_moe/120/logs/prof_A.log` (#120 A, `R3CY10WM83Y`,
first three M=1 calls). Each descriptor's start is taken as max(issue,
predecessor's `t_done_hi`). Every wait is `blocked=y`, so `t_done_hi` is
tight.

* Every descriptor starts the moment its predecessor ends. From gu0 to
  dn3 the queue is never empty.
* gu0 (depth 0) runs at 26.2 / 27.1 / 27.5 GB/s. The other seven run at
  29.5–32.0 GB/s (median 30.1). #117 s2 B reads the same (27.0–27.6 /
  29.7–33.2).
* The same unit's anchor read 37.3. The `f2` replay read 37.4, and
  `traced_f` (fresh=1, 32 regions over 168 MiB) read 37.5
  (`117/logs2/dma_probe.log`). Rotating source regions inside the harness
  therefore costs nothing on this unit, while #100's other unit showed
  0.90.
* The ring row (mean over 1408 calls) reads 32.3..34.4 GB/s. The first
  calls are slower than the average.

Candidates, ordered, with the expected saving per call at `mm` 677:

| # | candidate | state in the tree | expected | arithmetic |
|---|---|---|---|---|
| 1 | kick expert 0's gate_up before the activation quant | **already done**: gu0 and gu1 go out at `:1314-1321`, before `hvx_quant_rows_u8_params` at `:1328`. Still ahead of them: the copy-in wait and the `memset` (trace t = 1.0–2.7 µs). | ≤ 3 µs (EARLY: push gu0/gu1 after the copy-in *push* and before its *wait*; the FIFO keeps the 8 KiB first) | same pushes, same bytes |
| 2 | issue expert e+1's gate_up before expert e's down | **already the ring order**: gu0, gu1, gu2, gu3, dn0..dn3 (trace k = 1..8) | 0 µs | — |
| 3 | deeper issue | depth is bounded by the two 3.5 MiB slabs (`:1106-1108`), and the queue already holds ≥ 1 descriptor behind the running one at every push | 0 µs while the engine never idles. Not built. #100 found the descriptor count (8–46) irrelevant. | — |
| 4 | (new) TAIL: split the last down into two column halves (two descriptors, two pool runs over unit ranges `[j·dn, j·dn+h)` and `[j·dn+h, (j+1)·dn)`) | the tail from dn3's landing to the copy-out push is 24.5–26 µs (C(3) ≈ 16 + scatter 7.5 + wait) | ≈ 6–8 µs | each unit is one output column with its fixed k order. The same `moe_m1_down_worker`, the same bytes at the same VTCM offsets. Only the run boundary moves. |
| 5 | (new, only on branch D1) TOUCH: while the caller waits in `moe_m1_wait`, it loads one word per 4 KiB of the matrices ≥ 2 ahead of the running descriptor (volatile sink), then goes back to `dmpoll` | untested | 0 to ≈ 60–80 µs (the whole gap) | read-only touches of arena bytes the ring will copy. No output depends on them. |
| 6 | (new, only on branch D1, supervisor's yes) CLOCK: set_opts votes DCVS min corner TURBO for the session | untested | 0 to the whole gap | no kernel code; a power vote |

**The chosen approach** has three parts:

* Step 0 measures cold (level 2) against warm (level 3) in the app, next
  to the anchor, on today's tree.
* Only if the warm calls reach the anchor while the cold ones do not
  (branch D1) are 1 + 4 + 5 (+6) built behind one env switch,
  `NNTR_MOE_HTP_FEED_SCHED` (bit 0 EARLY + TAIL, bit 1 TOUCH, bit 2
  CLOCK). They are then measured as B / C / D against A.
* On any other branch nothing is built. Candidate 4 alone is ≈ 0.2 ms per
  token (≈ 0.6 % of decode), which is inside A's own spread. It cannot
  pass "decode up", and rule 33 does not cover it.

**Rejected alternative: two DMA queues** (a pool lane issuing half of each
slab on its own queue). It is the only schedule change that could raise
the engine's rate beyond the single-queue bound. But the harness measured
workers 1/2/4 at 37.3 / 38.2 / 36.2 GB/s (`117/logs2/dma_probe.log`). That
is +2 % at best. It would also put ring traffic on lanes that the
scoreboard proves only for the caller. The cost is not justified before
step 0 names the cause.

The contract (§2) is respected:

* Nothing is added to the three walls.
* No VTCM, heap or address-space growth: TAIL writes the same slab
  offsets, TOUCH reads mapped arena, CLOCK allocates nothing.
* `QS4CX_WH` keeps no CPU fallback.

Doc 45 §3 is respected:

* DMA stays hidden behind compute. It is the other way round here: the
  compute (≈ 240 µs) is hidden behind the DMA.
* No `_det` or quantizer is touched.
* The bit-identical and text gates are §1.

### 3.2 Switch semantics

The switch uses one word and one skel for every variant; the variant is
chosen by env. The echo check proves which cell ran.

* sched=0 reproduces today's schedule exactly. That is 10 descriptors,
  and the scoreboard counts are unchanged.
* sched & 1 gives 11 descriptors: dn(last) becomes two halves,
  `row_size = h·512`, `src_stride = dst_stride = dn_ntiles·512`,
  `nrows = inter_ktiles`, with `h = dn_ntiles/2`. The `DMA_KB` total is
  unchanged.
* TOUCH bounds: every touched address must lie inside
  `[wh_bytes, wh_bytes + bytes)` of a matrix this call pushes. A read past
  a registered weight is a fault, so this is checked on the host.

`ponytail:` the touch stride is a fixed 4 KiB, which assumes that the
page granule is what the engine misses on. If C shows any effect, the
upgrade is to take the stride from the arena mapping's page size.

## 4. Steps

**Step 0 — device read, no build (unavoidable; the orchestrator can run it
now, ≈ 12 min).** This step decides everything that follows.

The phone should hold the #134/#132 set. It is code-equal to `8860933c`:
`md5sum` on the device must match
`/local/mnt/workspace/htp_moe/134-132/md5.txt` (skel `0c2d5b00…`,
`nntrainer_causallm` `b6adb4f5…`, `libnntrainer.so` `bd5abd80…`). If it
does not, push that set.

The two gtests are built from this worktree. It needs
`git submodule update --init --depth 1` first, and then:

```
(cd test/jni && $ANDROID_NDK/ndk-build … unittest_hvx_dma_probe unittest_hvx_mm_u8i4 -j8)
```

They run against the same skel, pushed into `$T`.

```
W=/local/mnt/workspace/htp_moe/149; mkdir -p $W/logs
D=/data/local/tmp/nntrainer/causallm; T=/data/local/tmp/htp_u8i4_layer_test
therm() { adb shell dumpsys battery | grep -E 'level|temperature'; adb shell cat /sys/class/thermal/thermal_zone0/temp; }
adb devices; therm                                                      # serial + checkpoint 0
adb shell "cd $D && md5sum libnntr_hvx_skel.so nntrainer_causallm libnntrainer.so libcausallm_core.so && ls prompt512.txt && grep -H -E '_engine|_htp_layers' models/q40-qs4cx-wh/nntr_config.json"
adb shell "cd $T && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_dma_probe --gtest_filter='*MoeChunkReplay*'" 2>&1 \
  | tee $W/logs/dma_probe.log | grep -E '^DMA_REPLAY workers=1 load=0 pace=0 fresh=0 gap_us=0|name=(traced|traced_f|f2|f2_load) '
adb shell "cd $T && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_mm_u8i4 --gtest_filter='*MoeM1GemvFeedVsCompute*'" 2>&1 \
  | tee $W/logs/m1_bench.log | grep 'cell=vtcm'
prof() { adb shell "cd $D && sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": 64/' models/q40-qs4cx-wh/nntr_config.json && \
  NNTR_HTP_PROFILE=$1 NNTR_HTP_DMA_TRACE=66 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  ./nntrainer_causallm ./models/q40-qs4cx-wh \"\$(cat prompt512.txt)\"" 2>&1 | tee $W/logs/prof_L$1.log \
  | grep -E 'moe m1 gemv|level=|K=2048 N=2048|weight DMA|DMA ring' | cut -c1-480; }
prof 2; prof 3; therm                                                  # level 2 = cold, level 3 = warm (min of 5)
```

Per-descriptor rates (mawk-safe; the numbers in §3.1 came from this):

```
rate() { awk '/^\[HTP-DMA\] call=[0-9]+ M=/{m1=($3=="M=1");prev=-1;next}
  m1&&/^\[HTP-DMA\] call=[0-9]+ push/{for(i=1;i<=NF;i++){split($i,kv,"=");v[kv[1]]=kv[2]}
  if(v["kind"]=="copy")next; split(v["done"],d,"\\.\\."); t=v["t"]+0; hi=d[2]+0; s=(prev>t)?prev:t;
  print ($3=="k=1"?"first":"rest"), v["bytes"]/(hi-s)/1000; prev=hi}' $1 | sort -k1,1 -k2,2n |
  awk '{a[$1]=a[$1]" "$2} END{for(k in a){n=split(a[k],x," ");printf "%s median %.1f GB/s n=%d\n",k,x[int((n+1)/2)],n}}'; }
rate $W/logs/prof_L2.log; rate $W/logs/prof_L3.log
```

Expected lines:

* `applied=0x303e1 … feed=vtcm source=default` and `qos_mode=2`.
* The M==1 row with `m1_gemv=1408/1408 feed=1408/1408`, `DMA ring:
  desc=10/call`.
* 66 `[HTP-DMA] call=<k> M=1` headers per level, with `rep=1/1` at L2
  and `rep=5/5` at L3.

Decide, with `a` = the anchor and `c` / `w` = the "rest" medians at
L2 / L3 (the ring rows' `engine` ranges are pasted beside them):

* **D0**, `c ≥ 0.95a`: the gap is gone on today's tree. Close ㉓ as
  measured and build nothing.
* **D1**, `w ≥ 0.95a` and `w − c ≥ 0.05a`: the loss belongs to the cold
  call (state carried across the inter-call idle). Go to step 1.
  Variant D (CLOCK) needs the supervisor's yes, because it is a power
  vote and not a schedule change. Without it the handoff is A/B/C.
* **D2**, anything else: the in-app per-byte rate is the bound, and
  there is no fetch-schedule lever. Close ㉓ with the table and a ledger
  rule. Build nothing (§3.1).

Steps 1–6 run only on D1.

1. **Switch plumbing.** Add the flag defines, the parse, the banner, and
   `moe_opts_host_check` (`unset=0x303f1`, 1 → `0x343f1`, 3 → `0x3c3f1`,
   5 → `0x1343f1`, invalid → 0). *Gate:* `ninja -C build`;
   `bash test/htp/host/run_host_checks.sh` prints
   `MOE M1 GEMV OPTS: unset=on(0x303f1) …` and `ALL CHECKS PASS`.
2. **EARLY + TAIL, and TOUCH, in the kernel. Extend the host scoreboard.**
   *Gate:* `run_host_checks.sh` must print all of these:
   * `M1 GEMV PATH BIT-IDENTICAL TO HMX PATH (…, feed=arena,vtcm, sched=0,1,3)`
   * `M1 GEMV VTCM FEED SCHEDULE OK (…)`, where the push/wait counts
     rise by one per case with sched & 1 and are unchanged at sched=0
     (1232 / 1232 today)
   * a `FEED TOUCH IN BOUNDS` line
   * `IN-SITU CHUNK PLAN MATCHES KERNEL (46 descriptors)`, unchanged
   * `HVX GEMV NATIVE BIT-IDENTICAL`, unchanged, with both mutants caught

   `bash tools/htp_syntax_check.sh` must exit 0. `clang-format-14` is run
   on the changed lines.
3. **Full model on the host.** Add `NNTR_MOE_HTP_FEED_SCHED=3 run_e2e sched …`
   and `$EVAL --label feed-sched "$OUT/dump_htp" "$OUT/dump_sched"`.
   *Gate:* `bash test/htp/host/run_inproc_e2e.sh` prints every existing
   pass line unchanged, plus `E2E eval feed-sched … bit_identical=1`,
   then `INPROC E2E PASS`. The `*Lfm2Moe*` gtests show 6 PASSED, and
   `*qs4cx*` passes.
4. **Skel.** Run `./test/htp/build.sh` (`UNDEFINED SYMBOLS OK`) and record
   the md5. CLOCK's `HAP_power_set` must compile clean under `-Werror`.
5. **App + gtests.** Run `build_android.sh --htp`. Check the NEEDED lines
   and `strings libnntrainer.so | grep -c 'sched=%u'` ≥ 1. Build
   `ndk-build unittest_hvx_mm_u8i4 unittest_hvx_dma_probe` and record the
   md5s. Write the three short prompts to
   `docs/measurements/149-prompts/p{1,2,3}.txt` (English prose, code,
   Korean; 20–60 tokens each). *Gate:* rung 3 of `hexagon-gates`.
6. **Handoff `docs/measurements/149-feed-engine.md`** (device,
   unavoidable). One app set and one skel; the variant is chosen by
   `NNTR_MOE_HTP_FEED_SCHED`:
   * A = unset (the unchanged schedule; `applied=0x303f1`, `desc=10/call`),
     run first
   * B = 1
   * C = 3
   * D = 5, only with the supervisor's yes

   The handoff runs these, in order:
   1. Anchor cell (`MoeChunkReplay`).
   2. `MoeLayerM1GemvMatchesHmx` ×sched and the `vtcm` bench ×sched.
   3. Level-2 profiles at G=64, in the order A, B, C, D, A.
   4. E2E, prompt 512, G 64 / 512 / 1024 × 2 runs, mirrored per G:
      A B C D, then D C B A, with a 60 s cool-down before G=1024.
   5. The three short prompts at G=64, once per variant.
   6. `NNTR_HTP_DUMP` per variant on p1 at G=64, pulled and read with
      `htp_dump_eval.py`.
   7. `therm` at each block.

   ≈ 50 min. The empty table carries A's reference: #120 A `mm` 677.2,
   `dsp` 711.7, 35.97 / 35.96 / 35.17 tok/s. The ring line is expected to
   read `desc=11/call` for B/C/D. Set `state:needs-measurement`.
7. **Read it.** Apply the gates in §1. If a variant passes, the same PR
   flips `HTP_MOE_FEED_SCHED_DEFAULT` to it, and `=0` stays as the
   opt-out, which becomes the next sitting's A0. Otherwise the switch
   leaves with the PR closed unmerged and the table goes to the ledger.

## 5. Risks

**DMA rate drift between sittings** (rules 30/32/34). The three anchor
values are 40.2, 31.2 and 37.3 GB/s. So the `mm ≤ 630` half is read only
next to the same sitting's anchor, and only at ≥ 36. The handoff table
puts the anchor line in its first row.

**Cold vs warm confounds** (step 0). Level 3's warm repeat can gain in
three ways:

* from translation state, which TOUCH can reach;
* from the clock, which only CLOCK reaches;
* from cache residency of the same 22 MB, which no schedule reaches
  across calls.

That is why D1 needs `w` near the anchor (a no-reuse replay rate, since
`traced_f` equals `traced` on this unit) and not merely `w > c`. C and D
are separate variants so that the handoff says which of the two it is. A
null C together with a positive D is a result: the clock, not the fetch.

**The trace perturbs what it measures.** Each traced call is followed by
a read-back RPC. 66 of the 1408 calls get a longer idle before the next
call. The ring row covers all calls and is pasted next to the per-call
medians. A disagreement is noted, not averaged.

**DVFS and thermal.** CLOCK heats the DSP and can move later cells. The
mirrored order and the `therm` checkpoints make this visible. A D whose
prefill or later-G cells degrade against its own first run is read as
thermal. The M>1 `dsp` column is the prefill tie-breaker.

**Stale skel.** The new bits change the echo, so an old skel throws with
"predates moe_set_opts". A has to run on the new skel too (sched=0 is
today's schedule, byte for byte). Its `desc=10/call` and ring signature
must match step 0's, or the sitting is void.

**Host vs device.** The host ring stub copies synchronously, so a
descriptor-order race cannot show on the host. The scoreboard proves the
order by construction; the device gtest ×sched and the model dumps prove
it on silicon. TOUCH's bounds are a host check because an out-of-range
touch would be a fault, not a wrong number.

**Address space.** No new heap and no VTCM. The 3840 MiB arena +
≈ 182 MiB heap budget is untouched.

## 6. Docs to update

* **`docs/htp_moe/BENCHMARK.md`**:
  * Results: step 0's rows (anchor, `f2`, `traced_f`, the L2/L3 ring rows
    and per-descriptor medians, tagged with the unit and `8860933c`).
    After the sitting, the A/B/C(/D) rows.
  * Method: one line saying that the anchor and `f2` re-read the same 4
    experts, while the app reads 4 fresh ones per call.
  * Artifacts: the handoff set row.
* **`docs/htp_moe/LEDGER.md`**:
  * ㉓ (feed-engine row): candidates 1–3 are already in the tree or worth
    0 µs by the #120 A trace. The ring never idles, and the gap is
    per-byte rate. Add the step-0 branch taken.
  * If D0/D2: close ㉓ with a rule (the in-app engine rate and what bounds
    it).
  * If D1: a §2 verdict row after the sitting, plus a rule naming which
    of TOUCH / CLOCK moved the rate.
