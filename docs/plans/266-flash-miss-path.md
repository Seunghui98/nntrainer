# Plan 266: the decode E2E flash miss read path (C = 16, bit-preserving)

Issue #266 (p0, hexagon). Base `htp_decode` @ `3897963d8` (= the E sitting's
build, `docs/measurements/260-step2-r2-3897963d8.md` on PR #265). Inputs:
that sitting's `E_p512_g512.log` / `E_p1024_g512.log` / `*_optime.log`
(token driver lines), plan 260 §7 (the one-PD graft), plan 261 (ranking
format), plan 216 + LEDGER rule 62 (fadvise on this kernel), doc 52 §10.7 /
10.9 / 10.11 / 10.13 / 10.19 / 10.32 (the LFM flash-expert measurements).
Tags: [M] measured in the sitting, [G] arithmetic from measured numbers,
[E] estimate, [P] the probe of step S0 decides.

## 0. Three readings the issue asked for first

### 0.1 Expert reuse between tokens (what the logs can and cannot say)

**The sitting logs do not carry the routed ids.** The DSP returns every
token's routed sets to the ARM (`htp_dspq_token_resp.route`, 320 B =
30 × (1 + 8), `htp_dspq_wire.h:76,96`; filled from `g->route_log`,
`hexkl_graph.c:214-219`), but the driver only feeds them to `poolRefresh`
(`htp_compute_ops.cpp:3052-3077`) and prints nothing. What the logs do give
is the aggregate [M]:

| cell | routed / token | misses / token | **hit rate at C = 16 (LRU)** | rounds / token | misses / round | uniform-routing expectation |
|---|---|---|---|---|---|---|
| E p512 G512 (169 tok) | 240 (30 × 8) | 94.96 (16 049 / 169) | **60.4 %** | 28.6 | 3.32 of 8 | 12.5 % (16 of 128) |
| E p512 G64 | 240 | 99.4 | 58.6 % | 28.9 | 3.44 | 12.5 % |
| E p1024 G512 (512 tok) | 240 | 120.8 | **49.7 %** | 29.6 | 4.08 | 12.5 % |
| E p1024 G1024 | 240 | 118.6 | 50.6 % | 29.4 | 4.03 | 12.5 % |

So the routing has strong temporal locality (5× the uniform rate with a
pool that holds 12.5 % of the experts), 1.4 of 30 layers per token miss
nothing, and the p1024 prompt (a different text) reuses less. The two
prompts cannot be compared expert by expert without the ids. The miss
count is deterministic (16 049 in both p512 G512 runs).

**Instrumentation (one line, lands with S1's PR):** in `tokenForward`
after `r.route_n` is clamped (`htp_compute_ops.cpp:4987`), when
`NNTR_HTP_ROUTE_LOG=<path>` is set, append one line per MoE op in
`NNTR_MOE_TRACE`'s format — `"<layer> 1 | <ids> |"` — so
`tools/moe_expert_cache_sim.py` (doc 52 §10.10, `:246-260`) replays it
unchanged. Check: the sim's `lru` policy at `--cache 16` must reproduce
16 049 misses over 169 tokens; then it prints the per-layer hit rate, the
Belady ceiling, and (a 30-line addition to the sim) the previous-token
same-layer overlap curve — the "hit-rate curve" the issue wants. A
zero-code first reading exists today on the A path: `NNTR_MOE_TRACE` is
written in the layer's routing stage, decode included
(`lfm2_moe_layer.cpp:222-233`, `:1712-1721`); A's routing differs from E's
only through the FC path (plan 260 §7.3), so its statistics stand in until
E's own log exists.

### 0.2 The byte floor

Per miss the slot is 2.93 MiB (1408 MiB / 480; file bytes 2.87 MiB =
gate_up 1 993 728 + down 1 013 760 B incl. scales and sums; the sitting's
`pgpgin` says 2.46 MiB of it came from flash, the rest were page-cache
hits). Floor = misses / token × 2.93 MiB ÷ rate [G]:

| cell | misses / token | MiB / token | **1.0 GiB/s (today)** | 2.5 GiB/s | 2.79 GiB/s (= doc 52 §10.32's 3.0 GB/s dd) | 3.5 GiB/s | measured wait |
|---|---|---|---|---|---|---|---|
| p512 G512 | 94.96 | 278 | 278 ms | **111 ms** | 100 ms | 79 ms | 208.5 ms (= 1.12 GiB/s on the 2.46 MiB actually read) |
| p1024 G512 | 120.8 | 354 | 354 ms | **142 ms** | 127 ms | 101 ms | 286.9 ms |

Two consequences. (a) **No read-rate lever reaches the issue's ≤ 60 ms at
C = 16**: even at 3.5 GiB/s the floor is 79 / 101 ms. Rate levers take
208 → ≈ 100–115 ms; the rest can only come from overlapping reads with the
80.6 / 92.4 ms of DSP compute per token, which needs the reads to start
before the layer's router runs (§3, lever 3). (b) Today's rate is 2.5–3×
below the same storage's sequential rate — the request shape, not the
UFS, is the first lever (§3, lever 1).

### 0.3 What `prefetch_readers_` / `addPrefetch` already do, and reachability

`prefetch_qs4cx_wh_experts_begin/_end` (`htp_compute_ops.cpp:3412-3481`),
`startPrefetchReaders` / `prefetchReaderLoop` (`:6587-6631`, 4 threads
pinned off the caller's core, `NNTR_MOE_PREFETCH_READERS/_CPUS`,
`:7362-7380`) and `HtpProfile::addPrefetch` (`:315`) are doc 52 §10.10 /
10.20's **prefill** read-ahead: the layer queues layer L + k's non-resident
experts while L's call runs, `_end` registers them through the FastRPC
swap call between calls. They are called only from the layer's
`total_tokens > 1` branch (`lfm2_moe_layer.cpp:1324-1334`, `:1372`,
depth `expertPrefetchDepth()`), so **on the Gemma one-PD token they are
unreachable**: the token bypasses the layer at the decode row (plan 260
§7.2 resident check) and the only read path is the mailbox round
(`poolAnswer`). Their registration step (`_end` → `registerStagedBatch`)
is also the wrong one for the token, where S1 rebinds in place from the
answer (`hexkl_token.c:185-196`). What is reusable: the reader-thread
start / pin code and `readExpert(st, use_pool=false)` (`:6314-6331`,
lock-free, one whole-range `pread` per weight), which lever 1 drives from
a job queue of its own. Doc 52 §10.13 / 10.19's finding carries over:
readers on the caller's core slow the caller; pin them elsewhere.

## 1. Goal and gate

Issue: *"miss wait 208 → ≤ 60 ms/token at C = 16, bit-preserving, no
weight-file dtype change; gate: token ids identical to the sitting's E,
nll equal, misses/token unchanged (only wait time drops), memory within
the one-PD ceiling."* Measurable, per device variant (§4):

* `[HTP] token driver: pool … miss_wait_us/token=` on E p512 G512 and
  p1024 G512 (the two sitting cells), read with `arm_ms/round`,
  `misses/token`, `pgpgin_mib` and `rounds=` beside it; the §0.2 floor row
  is the bound each rate lever is read against.
* Bit identity: the generated token ids byte-equal to the sitting's E of
  the same prompt and G (`260-e-run.sh`'s text `cmp`; E's p512 G512 ends
  at token 169 on `<eos>`, so equality is over 169 tokens); `NNTR_PPL=1`
  nll equal to the digit (4.57345 / 3.61786). Read levers change bytes'
  *timing* only; lever 3 changes which experts are resident when, which
  `graph_moe_miss` (`hexkl_graph.c:101-166`) makes bit-independent of the
  miss pattern by construction — the gate still reads it.
* `misses/token` == 94.96 (p512) / 120.8 (p1024) for levers 1, 2, 4; for
  lever 3 the DSP still posts every miss (§3), so the count stays and the
  wasted bytes show in `pgpgin_mib` (gate: ≤ +25 % bytes/token).
* Memory: `s1_arena_mib=1408`, `mapped_mib` 449, peak RSS within +50 MiB
  of the sitting's (bounce buffers are anon); S1 ceiling 3840 after each cell.
* Host: `run_inproc_e2e.sh` pool lines (`E2E e3 pool C=1/C=2 …
  bit_identical=1`, the gemma64 C = 2 line) with the same `misses=`;
  `run_host_checks.sh` `ALL CHECKS PASS`; `*Lfm2Moe*` 7/7.
* Standing: prefill of every variant ≥ −5 % of the same sitting's A (the
  readers must not run during prefill — they idle outside `poolArm`); text
  identical to E (the CPU-run text gate is read as "== the sitting's E",
  which the user passed).

## 2. Where it lives (`htp_decode` @ `3897963d8`)

**ARM, `nntrainer/tensor/htp_backend/htp_compute_ops.cpp`** (all levers):
`poolAnswer :2883-2982` — the miss handler: `pool_fn_` (the LRU) names the
loads `:2903-2915`, the agreement check `:2917-2926`, then **each miss is
staged and read in turn under `handle_mutex_` `:2930-2940`** (`stageExpert
:6236-6240`, `readExpert :6314-6331` → `readWeight :6640-6697`: with
`use_pool=true` the nibble half goes through `ThreadManager::parallel_for`
in `kExpertReadSlicesMax = 8` page-aligned slices `:6664-6673`, `:7330`,
then one tail `pread` `:6681-6682`; `preadAll :6192-6208`); the answer is
written `:2963-2975`. `poolServe :2984-3017` (the server thread, a yield
spin; created in `poolArm :3022-3030` from the token thread, so it inherits
that thread's core — `ThreadManager` pins the main thread to the fastest
core, `thread_manager.cpp:101`); `poolDisarm :3031-3050`, `poolHarvest
:2857-2881` (files the previous answer's loads), `poolSync :2803-2855`,
`poolRefresh :3052-3077`. `tokenForward`: the dspqueue write / blocking
read `:4969-4988`, stats fold `:5040-5058`, the pool stats line
`:4835-4846` (`pool_read_us` `:2963`, `pool_rounds`). `PoolServer
:7556-7568`, `E2eState` pool fields `:7547-7548`, `fadviseKnob :6337-6343`,
`fadviseExpert :6349-6372`, `adviseLater :6383-6420`. The model fd is the
loader's long-lived `O_RDONLY` fd in `ExpertFileDesc` (plan 216 §2).

**DSP** (lever 3 only): `hexkl_graph.c:113-166` `graph_moe_miss` (post →
resident experts → wait → missed experts → row sums), `:168-232`
`graph_op_moe` (route log `:214-219`); `hexkl_token.c:120-141`
`tk_miss_post`, `:142-196` `tk_miss_wait` (20 µs sleep poll,
`HEXKL_TOKEN_POLL_US`, `hexkl_token.h:61`); the router op
(`hexkl_graph.c:377-392`, `hvx_m1_ops_f32.c:362-374`). Mailbox structs
`htp_dspq_wire.h:104-150`: `htp_miss_req` is 148 of its 256 B
(`HTP_MBOX_MISS_REQ_BYTES`), `htp_miss_ans` ≤ 1024 — a second 16-id list
fits the request without moving the answer.

**Layer**: `lfm2_moe_layer.cpp:1178-1184` hands the experts and
`g_expert_lru.acquire` to the driver (`set_decode_moe_experts`, plan 260
§7.2); `expert_lru.h:117-168` `acquire` (one LRU over all 30 layers,
pinned = the call's routed set), `:83-106` `makeRoom`, `:67-74`
`hold/unhold`, `:170-177` `refresh`; `:933-942` `moeReadAt` is the layer's
own helper (diff / shadow), not on the token path; `:222-233` /
`:1712-1721` `NNTR_MOE_TRACE`.

**Tools**: `tools/moe_expert_cache_sim.py`; `tools/htp/page_cache_evict.c`
(the NDK one-file build pattern, `:19`); `htp_rpcmem.h:65,125-127` (dlsym
`rpcmem_alloc`, heap 25, `RPCMEM_FLAG_UNCACHED` — the probe allocates its
ION target the same way). New: `tools/htp/pread_probe.c` (S0),
`docs/measurements/266-pread-probe.md`.

**Consumers that do not move** (levers 1, 2, 4): no IDL
(`test/htp/nntr_hvx.idl`) / `generate_stub.sh` / skel change — the device
skel stays `4c665097…`; no quantizer tag, no loader check; the
`NNTR_HTP_PROFILE` stage tables are untouched (the pool line gains
`readers=` and, for lever 3, `spec_mib=` / `pred_hit=` fields; plan 216
§2: `tools/htp_fc_report.py` and the runners `grep -o` up to
`arm_ms/round=`, so appended fields are harmless). **Lever 3 moves**:
`htp_dspq_wire.h` (shared header, no IDL), `hexkl_graph.c`,
`hexkl_token.c` → rung 2 both arches, `mailbox_host_check` /
`token_host_check` / `graph_host_check` lines, and `HtpComputeOps` only in
`poolAnswer`'s agreement check (`:2917-2926`).

## 3. Design: levers ranked by ms per day

Baseline E p512 G512 [M]: token 289.3 ms = miss wait **208.5** + DSP
non-wait 80.6 + ARM 1.3; 28.6 rounds, 7.71 ms ARM per round for 3.32
misses → **2.32 ms per 2.93 MiB miss**. p1024 G512: wait 286.9 of 379.5.

| # | lever | wait ms before → after (p512 / p1024) | gain | days | **ms / day** | gate |
|---|---|---|---|---|---|---|
| 1 | **Round-parallel miss reads**: every miss of a round read at once by dedicated readers (one expert per reader, two whole-range `pread`s of 1.99 / 1.01 MiB; `FADV_RANDOM` on the fd if [P] says so), slots staged under the lock, reads outside it | 208.5 → ≈ 100–115 / 286.9 → ≈ 130–150 at 2.5–2.8 GiB/s [G, rate P] | **−95…−110** | 2 | **≈ 50** | ids ==, nll ==, misses/token ==, `pgpgin/miss` ≤ today's 2.46 MiB |
| 2 | **`O_DIRECT` through a 4 KiB-aligned bounce** (no page allocation / reclaim per page, no refault bookkeeping), the reader `memcpy`s into the ION slot (8 threads reach 14 GB/s into uncached ION, doc 52 §10.11: ≈ 0.2 ms an expert, inside the reader) | lever 1's remainder − (10…30) [E, P decides] | −10…−30 | 1 (inside S1 if [P] ≥ 1.2× cached) | ≈ 10–30 | as 1; `pgpgin_mib` → ≈ 0 by construction |
| 3 | **Next-layer route prediction prefetch**: at layer L's MoE the DSP also runs layer L + 1's router on the current residual stream and posts the non-resident predicted ids as a second list; the ARM reads L's misses first, then the predicted ones into LRU-tail slots in the background; when L + 1's real round comes, a predicted hit costs the rebind only | after 1: ≈ 110 → ≈ 60 / 150 → ≈ 85 at p = 0.8 [E]; the hideable budget is the 80.6 / 92.4 ms of compute | −40…−50 | 1 (accuracy reading, S2) + 4 | ≈ 9 | ids ==, nll ==, misses/token == (the DSP still posts them), `pgpgin` ≤ +25 %, router pcyc ≤ 2× |
| 4 | **Page-cache hygiene** that is cheap on this kernel: `POSIX_FADV_RANDOM` once on the fd (readahead off: a slice is exactly its I/O), one `DONTNEED` pass at load end over the 480 preloaded experts (dead in cache: 1.4 GiB back to the complement; off the token path) | −5…−15 [E] (today 14 % of a miss's pages are cache hits) | −5…−15 | 0.5 | ≈ 10–30 (wide) | as 1; a `pgpgin/miss` column |

**Order: S0 probe → 1 (+2 / +4 as the probe says) → 3 after its accuracy
reading.** 1 + 2 + 4 reach the floor (§0.2): token ≈ 185 ms ≈ 5.4 tok/s
at p512 [G]; 3 on top ≈ 140 ms ≈ 7 tok/s and the ≤ 60 ms wait, at p ≥ 0.8
and ≥ 2.5 GiB/s. **At C = 16 this issue's ceiling is ≈ 7–8 tok/s** (the
floor plus plan 261's compute levers); 40 tok/s is not in this list — it
is the bytes-per-token's (hit rate × expert size), stated so the user
reads the ranking against it.

### 3.1 Lever 1 — why today's path reads at 1.1 GiB/s on a 2.8 GiB/s device

`readWeight` splits one 1.99 MiB weight into 8 contiguous ≈ 250 KiB
slices read by 8 `ThreadManager` workers, then the next weight, then the
next expert — serial across experts, parallel only inside one. On the S25's
6.6 kernel the file readahead window is 1 MiB (`fadviseExpert`'s comment,
`:6351-6353`): slice 0's `pread` starts a 1 MiB readahead that covers
slices 0–3, whose threads then block on the same locked pages; the I/O
depth the UFS sees is ≈ 2 requests of 1 MiB, not 8, and every page it
fills must first be reclaimed (refault ≈ pgpgin, the sitting's vmstat
table). That is the 2.2 ms per expert. Doc 52 §10.9's "8 slices bought
18 %" was measured warm (page cache → ION copy); doc 52 §10.32 read 3.0
GB/s cold from the same class of device with 4–8 *independent* streams.
Lever 1 makes the misses of a round independent streams: the pool server
stages the 3–8 slots (under `handle_mutex_`, as today), hands one
`StagedExpert` per job to N miss readers (`startPrefetchReaders`'s
pattern, pinned off the server's core, `NNTR_MOE_MISS_READERS`, default
= the probe's best N), waits on a done counter, writes the answer. A
round of 3.3 misses then has 3–7 outstanding ≥ 1 MiB reads; the tail
(scales + sums, 11 KiB) rides each reader. The slice `parallel_for` path
stays for the prefill batch (`register_qs4cx_wh_expert_files`,
`:3367-3393`) untouched.

Rejected: raising `kExpertReadSlicesMax` or re-slicing (plan 216 rev. 1's
lever, refuted there: slices inside one expert share the readahead window
and the page locks — the probe's `threads × request size` cells show it
either way).

### 3.2 Lever 2 — `O_DIRECT`

Direct into the ION mapping is expected to fail (`remap_pfn_range`
mapping, no page pinning — doc 52 §10.32 rejected it on that ground
without a reading); the probe records the errno. Through a bounce it is
legal (F2FS: 4 KiB-aligned offset / length / buffer; expert offsets in the
file are not 4 KiB-aligned — 1 993 728 mod 4096 ≠ 0 — so the range is
rounded out and the exact bytes copied), costs one `memcpy` per weight and
removes the page allocation, reclaim and refault accounting from the miss
path. Whether that is worth 10 or 30 ms is the probe's cell `direct` vs
`cached` under the app's memory pressure. Rejected until the probe says
otherwise: a user-space second-level cache (plan 216 §3's rejected row;
no free DRAM at 11.1 GB − 12.8 GB file).

### 3.3 Lever 3 — prediction prefetch, and why the previous token is not it

"Previous-token / previous-layer reuse" is already what the pool holds:
after a token `poolRefresh` makes every routed expert most recent, so the
LRU at 16 / layer keeps ≈ the last two tokens' sets of each layer — that
is the 60 % hit rate; prefetching "the previous token's experts" adds
nothing. New information inside the token is the residual stream: layer
L + 1's router applied to the stream *before* layer L's MoE adds its
output (the pre-gated-MoE idea) predicts L + 1's top-8 with an accuracy p
this plan does not know for Gemma-4 — **S2 measures it before a line of
the lever is written** (DSP side: with a graph param set, `graph_op_moe`
evaluates the next op's ROUTER_TOPK on its own input and writes the 8
predicted ids beside the route log; the ARM prints predicted vs actual per
layer; the sim scores p per layer and the net bytes). Mechanism if p
passes: `tk_miss_post` gains `n_pred` + `pred[16]` (the request has 108 B
of room); `poolAnswer` reads L's misses first and answers, then its
readers load the predicted non-resident ids into LRU-tail slots
(`g_expert_lru.acquire` with those keys, so the ARM's pool and the LRU
agree; `ExpertLru::hold` keeps them from being re-taken before L + 1's
round) as staged-unbound entries in `p.pending`; at L + 1's round the DSP
still sees `NO_HANDLE` and posts them — the agreement check accepts "in
`p.pending`" as a load, and the answer carries the already-read slots: the
round costs the rebind and the mailbox hop. No IDL change; `misses/token`
unchanged by construction; a wrong prediction costs 2.93 MiB of flash and
one LRU-tail eviction, which `pgpgin_mib` shows. The router runs twice per
layer (+3–6 ms/token at today's clock, plan 261 lever 1 brings it down).
Rejected: splitting the answer per expert so S1 computes each missed
expert as it lands — per-expert compute is ≈ 0.09 ms (2.93 MiB at 35
GB/s), nothing to hide.

### 3.4 Lever 4 — what rule 62 leaves of "page-cache hygiene"

LEDGER rule 62 (#216's B cell): on this kernel `WILLNEED` reads only the
1 MiB readahead window and costs 3–8 ms per expert, `DONTNEED` 2–10 ms;
per-miss advice is off the table (`fadvise=0` stays). `FADV_RANDOM` is a
flag on the file, costs nothing per read, and is the right setting for
3 MiB reads whose neighbours are never wanted next. A single `DONTNEED`
pass at load end (≈ 480 × 2–10 ms ≈ 1–5 s once, before the first token;
the user's call whether the load time is acceptable — default off) is the
only cheap way to give the complement 1.4 GiB more cache. Readahead
sysfs and `O_DIRECT` on the whole file need root — not available.

Design rules kept: no DSP change for 1 / 2 / 4; lever 3 touches no
quantizer input, no `_det` spec, no DMA ring; contract §2's arena budget
(1408 MiB pool, 449 FC/head) and the one-PD ceiling unchanged; no CPU
fallback for `QS4CX_WH`; every knob's unset value is today's path until
the A/B flips it (#115 / #151 precedent).

## 4. Steps (each ends in a `.claude/skills/hexagon-gates` rung)

* **S0 — device probe (unavoidable, 0.5 d; agent adb on `R3CY205ZMND`,
  `sitting_lock.sh take`)**. `tools/htp/pread_probe.c` (one file, built
  as `page_cache_evict.c` is): samples 256 expert ranges from the real
  file's expert region, evicts them (`page_cache_evict <file> -1`), applies
  `--pressure <MiB>` (anon, touched: 4 300 = the app's RSS + arena) and
  reads them under the matrix **destination** {malloc, ION uncached via
  `rpcmem_alloc` as `htp_rpcmem.h:125-127`} × **mode** {`pread`,
  `pread` + `FADV_RANDOM`, `O_DIRECT` → bounce → `memcpy`, `O_DIRECT` →
  ION direct (expect `EFAULT`, record)} × **threads** {1, 2, 4, 8} ×
  **request** {256 KiB slices of one expert (today's shape), 1 MiB, one
  weight 1.99 MiB, one expert 2.93 MiB}; per cell GiB/s, ms per expert,
  Δ`pgpgin`, `/proc/pressure/io` avg10. Also the §0.1 reading: A p512
  G512 with `NNTR_MOE_TRACE` → sim at `--cache 16` (hit rate per layer,
  Belady, prev-token overlap). Gate: the table in
  `docs/measurements/266-pread-probe.md` and a #266 comment with the
  decision cells: N and request shape for lever 1; `direct ≥ 1.2 ×
  cached` → lever 2 in; `RANDOM` in / out. No rung beyond 0 (a tool, not
  the app).
* **S1 — levers 1 (+2, +4 as decided) + the route log** (2 d,
  `htp_compute_ops.cpp` only). Rung 0; rung 1: `ninja -C build`, the pool
  lines of `run_inproc_e2e.sh` `bit_identical=1` with the same `misses=`
  under `NNTR_MOE_MISS_READERS=1,4` and the route log written and replayed
  by the sim to the same miss count; `run_host_checks.sh`; `*Lfm2Moe*`
  7/7. Rung 3 (`--cache`, skel md5 unchanged `4c665097…`). **Device
  (unavoidable, with S1)**: variants ≤ 4 — **A** = E of the sitting,
  unchanged (`3897963d8` set, nothing set); **B** = S1, default knobs;
  **D** = S1 with `NNTR_MOE_MISS_READERS=1` (isolates the request shape
  from the parallelism); prompts 512 / 1024, G 64 / 512 / 1024, cool start
  per block, A first and last (`260-e-run.sh`, same configs, same md5
  table), one `NNTR_PPL=1` cell per prompt. Gate §1.
* **S2 — prediction accuracy reading** (1 d; DSP instrumentation behind a
  graph param, no IDL): rung 1 (`graph_host_check` / `token_host_check`
  lines unchanged, the hd64 inproc E2E lines), rung 2 v79 + v81
  (`UNDEFINED SYMBOLS OK`, `ARCH OK`), rung 3; device: one E p512 G512
  cell with the log; the sim scores p per layer. **Decision rule:** p ≥
  0.7 on ≥ 20 of 30 layers and net bytes ≤ +25 % → S3; else the lever is
  closed with its number and this plan ends at S1's result.
* **S3 — lever 3** (4 d): `htp_dspq_wire.h` + `hexkl_token.c` +
  `hexkl_graph.c` + `poolAnswer`; rung 1 (`mailbox_host_check` with the
  second list, the pool inproc lines `bit_identical=1`), rung 2 both
  arches, rung 3; device **C** = S3 against A (and B from S1's sitting if
  the same day), the §1 gate plus `pgpgin ≤ +25 %`, `pred_hit=` on the
  pool line.
* **S4 — docs** (§6), PRs into `htp_decode` (S1: one PR; S3: its own).

## 5. Risks

* **The probe's regime ≠ the app's**: UFS rate depends on memory pressure
  (reclaim per page), the file's cache state and the UFS clock; doc 52
  §10.32's 3.0 GB/s was read without pressure. The probe runs with
  `--pressure` and after eviction; the handoff table carries `pgpgin/miss`,
  PSI io and ms/miss beside every tok/s so a rate that does not transfer
  to the app is visible as a floor gap, not as a tok/s surprise.
* **Thermal / DVFS**: flash and the ARM readers both throttle; `mhz` on the
  per-kind line, zone0 at start / end, A first and last (plan 260 §7.3);
  cool start per G block. The wait is read in ms per token, never as a
  share of a clock-diluted total.
* **Stale skel**: S1 changes no DSP source — a device run whose skel md5
  differs from `4c665097…` is a wrong set; S2 / S3 change it (md5 in the
  table, `AEE_EBADPARM` = stale).
* **Address space / memory**: no new ION (bounce buffers anon, ≤ 8 × 3
  MiB); lever 3 uses the pool's own slots; `mapped_mib` / `s1_arena_mib` /
  peak RSS per cell. A C = 20 point is *not* tried (user: C stays 16).
* **Host-vs-device gap is total for the rate** (the workstation's
  readahead is 128 KiB and it has no pressure); the host proves bit
  identity and the miss count only.
* **The sim approximates E with A's routing** until S1's route log exists;
  p (lever 3) is a Gemma-specific number no literature replaces — S2's
  decision rule stops the lever if it is low.
* **Reader threads vs the app's own ARM work**: the token's ARM side is
  1.3 ms, so contention is the readers' own; doc 52 §10.19's pin-off-the-
  caller rule is kept and `arm_ms/round` reads any regression.

## 6. Docs to update

* `docs/htp_moe/BENCHMARK.md`: Gemma goal row — the E "now" (3.33 / 2.58
  at p512 / p1024 G512) gains lever cells B (and D) from S1's sitting and
  C from S3's, each with `miss_wait`, `misses/token`, `pgpgin/miss`
  columns beside tok/s; Method: "a flash-bound cell carries its floor row
  (§0.2)".
* `docs/htp_moe/LEDGER.md`: §2 verdict rows for #266 (S1, S3); rule
  candidate if the probe confirms §3.1 — "slices inside one expert share
  the 1 MiB readahead window: parallelism is across experts, not inside
  one" — beside rule 62; a rule for the floor arithmetic (misses × 2.93
  MiB ÷ rate is the bound of every read lever; overlap needs the next
  layer's ids); the C = 16 ceiling (≈ 7–8 tok/s) written next to the
  Gemma goal so the 40 tok/s gap is attributed to bytes per token.
* `docs/plans/261-decode-levers.md` §2: two rows above lever 1 (this
  plan's 1 and 3) so one table ranks compute and flash levers together.
* Contract §1 Speed: the E reading and the flash floor as the current
  "now"; §2 "Facts": the route log and the probe table as things not to
  re-derive.
