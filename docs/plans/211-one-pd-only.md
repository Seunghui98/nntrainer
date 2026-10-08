# 211 — Remove the two-session (two-PD) E2E path; one PD is the only E2E entry

Issue: dlwlzzero/nntrainer#211 (p0). Base: `origin/htp_decode` @ `84142c0b9`
(after #202 / #203 / #206 / #205). Every `path:line` below was read on that
revision. Decision (user, 2026-10-01): the two-PD NPU E2E path goes; the
one-PD path (`NNTR_HTP_E2E=1`, today `+ NNTR_HTP_E2E_PDS=1`: the FC set and
the expert pool in S1, one packet a token) is the only E2E entry and the one
that is developed further.

## 1. Goal and gate

From the issue, made measurable:

* **Host.** `bash test/htp/host/run_host_checks.sh` → `ALL CHECKS PASS` and
  `WORKER POOL LANES OK`; `bash test/htp/host/run_inproc_e2e.sh` →
  `INPROC E2E PASS` with the one-PD lines of §4 step 5 (the pool lines
  `E2E e3 pool C=… == e3 bit_identical=1` kept, `hops/token=0.00`);
  `unittest_causallm_models --gtest_filter='*Lfm2Moe*'` 6/6, none skipped;
  `tools/htp_syntax_check.sh` exit 0.
* **Skel.** `./test/htp/build.sh` → `UNDEFINED SYMBOLS OK`, `ARCH OK (V79)`;
  `HEX_ARCH=v81 ./test/htp/build.sh` → `ARCH OK (V81)`.
* **App.** `build_android.sh --htp` and the device gtests build (rung 3).
* **Device (one short sitting, S26).** Q28 (`NNTR_HTP_E2E=1
  NNTR_MOE_CACHE_EXPERTS=28`, no `_PDS`) G 64 × 2 and G 512 × 1: text == A r1
  of its G, `calls/token=1.00`, `token driver: close … hops/token=0.00
  timeouts=0/0 stale=0/0 … id_mismatch=0`, decode tok/s inside the Q28 band
  of `204-s26-rebaseline.md` (31–37); `NNTR_HTP_E2E_PDS=2` refused at load
  with the #211 message; the S1 ceiling cell unchanged start → end.
* **Standing gates.** Prefill ≥ −5 % of A in the same sitting (the deletion
  touches no prefill code, so this is a null check); A's text identical to
  the CPU `q40` run; the A path (hybrid: CPU + DSP MoE, FSU) is not entered
  by any changed line — verified by the `*Lfm2Moe*` 6/6 and A's banner
  (`dspq: on`, no `token driver:` line).

No BENCHMARK cell moves: this issue removes code, it adds no lever.

## 2. Where it lives (inventory)

Legend: **D** = two-PD-only, deleted; **S** = shared with one PD, stays
(possibly re-commented); **R** = rewritten for one PD.

### 2.1 ARM side

| file | lines | what | verdict |
|---|---|---|---|
| `nntrainer/tensor/htp_backend/htp_backend.cpp:64-154` | `openSecond`, `openSecondNow` (reserve, effective domain, URI, unsigned PD, `nntr_hvx_open` of S2, the `[HTP] s2: open …` banner) | **D** — the only caller is `e2ePlaceFc` (`htp_compute_ops.cpp:4500`) |
| `htp_backend.cpp:269-273` | destructor closes `handle2_` | **D** |
| `htp_backend.cpp:44-50`, `htp_backend.h:119-125` | `e2eRequested()` | **S** |
| `htp_backend.h:128-148, 166-171` | `openSecond`, `enabled2`, `handle2`, `effDomain2`, `vtcm2Bytes`, `s2Error`, the six members | **D** |
| `htp_compute_ops.cpp:1505-1513` | `sessionFor(kind)` | **D** — every caller gets the one session (`ensureGraphInit:2489`) |
| `htp_compute_ops.cpp:1600-1633` | the E2E validation: `mask != present` check **S**; the `graph_inited_` one-description rule **S**; the per-mask loop over `{KINDS_S1, KINDS_S2}` **D** |
| `htp_compute_ops.cpp:1645-1659` | `E2eState` creation, `NNTR_HTP_E2E_PDS` parse | **R** (§3: accept 1 / unset, refuse the rest) |
| `htp_compute_ops.cpp:1772-1800` | `registerQ4m1`'s E2E branch (`placeOn(e.h2, e.dom2, e.arena …)`, `nntr_hvx_q4m1_attach`) | **S** — the one-PD path uses exactly this branch with `h2 = h1`, `dom2 = CDSP_DOMAIN_ID` (`e2ePlaceFc:4496-4498`); becomes `placeOn(e.h1, CDSP_DOMAIN_ID, …)`. `placeOn` itself (`:5846`) is the arena primitive the MoE arena uses too (`:5840`) — **S** |
| `htp_compute_ops.cpp:1812-1825` | `releaseQ4m1`'s E2E slots | **S** (handle renamed) |
| `htp_compute_ops.cpp:1830-1836` | `bindQ4m1` sizing `q4m1_left_` | **S** |
| `htp_compute_ops.cpp:2483-2555` | `ensureGraphInit`: the two-graph loop `:2501-2522` **D**; the single `graph_init` branch **S**; the EXPERTS-table loop **S**; the catch's `graph_release(h2)` **D** |
| `htp_compute_ops.cpp:2383-2440` | `poolServe` / `poolArm` / `poolDisarm` / `poolRefresh` / `poolHarvest` (the miss protocol on the page) | **S** |
| `htp_compute_ops.cpp:3559-3568` | `invokeForward`'s E2E branch → `tokenForward` | **S** |
| `htp_compute_ops.cpp:3976-4052` | `e2eStart`: S1's queue, the page, `token_driver_start(role 0)` **S**; `q2 = dspqMake(h2, dom2, "dspq[S2]")` `:3996-4003`, the second `mmap` `:4015-4019`, the second `token_driver_start` `:4031-4034`, `rounds` `:4040-4043` **D**; banner `:4045-4050` **R** (`s2_effdom`, `rounds`, `hops/token`, `pds=` go) |
| `htp_compute_ops.cpp:4057-4240` | `e2eTeardown`: queue stop, driver stop, page unmap, graph release, slots, arena detach **S**; everything indexed `2` (`q2`, `drv2`, `mbox2`, `graph2`, `r2`, `info2`, `graph[S2]` lines, the `moe pcyc/round … s2 fc+dense_ffn+lm_head` line, the per-side loop) **D**; the two L0 lines `:4149-4173` and the pool line `:4174-4183` **R** (one side; §3) |
| `htp_compute_ops.cpp:4267-4455` | `tokenForward`: the one-packet write / read, pool arm / disarm, LM_BAN, logits / id, counters **S**; `r1` / `q1.api->write` `:4329-4332`, the second read `:4345-4347`, the `s1r` synthesis `:4348-4356`, `rnb1 / len1` checks, `S1 … S2` error text, the `2`-suffixed accumulators `:4420-4440` **D** |
| `htp_compute_ops.cpp:4478-4484` | `finish_decode_graph_q4_0` | **S** |
| `htp_compute_ops.cpp:4490-4530` | `e2ePlaceFc`: the `openSecond` branch `:4499-4506` **D**; the one-PD branch and the `[HTP] s2: fc arena …` banner **S** → renamed `[HTP] e2e: fc arena …` (the only consumer is `run_inproc_e2e.sh:574`) |
| `htp_compute_ops.cpp:4534-4541` | `q4m1FeedName`'s `vtcm2Bytes() == 0` clause | **D** |
| `htp_compute_ops.cpp:6380-6412` | `E2eState`: `h2`, `dom2`, `mbox2`, `drv2`, `graph2`, `q2`, `wait2_us`, `pcyc2`, `wall2_us`, `wall2_pcyc`, `kind2`, `hop2_us`, `disp2_us`, `ret2_us`, `inout2_us`, `rounds`, `one_pd` **D**; the rest **S** (`ret2_us` / `inout2_us` become `ret_us` / `inout_us`: they are the one DSP side's) |
| `htp_compute_ops.cpp:6435` | `ban_sent_` comment "S2's LM_BAN" | **S** (comment) |
| `nntrainer/tensor/htp_backend/htp_graph_desc.h:83-91` | `HTP_GRAPH_KINDS_S1 / _S2` | **D** — users after this issue: none (`htp_compute_ops.cpp:1510, 1618, 2507`, `graph_host_check.c:443-478`, `token_host_check.c:381-649`, all D or R) |
| `nntrainer/tensor/htp_backend/htp_dspq_wire.h` | `htp_dspq_token_req / _resp`, `HTP_MBOX_MISS_*` | **S, byte-for-byte** (§3: the wire does not move; `hops`, `wait_us`, `hop_us` read 0) |

### 2.2 DSP side (`test/htp/`, `hmx/`)

| file | lines | what | verdict |
|---|---|---|---|
| `nntrainer/tensor/htp_backend/hmx/hexkl_token.c:104-120` | `tk_post` | **D** |
| `hexkl_token.c:122-165` | `tk_take` | **D** |
| `hexkl_token.c:41-101, 167-200` | `tk_now_us`, `tk_clean`, `tk_refresh`, `tk_pause`, `tk_sleep`, `tk_pcycles`, `tk_stretch_end`, `hexkl_token_rounds` | **S** (the miss round uses the first five; `tk_row_bytes` / `tk_row` go with the slots) |
| `hexkl_token.c:205-286` | `tk_miss`, `tk_miss_post`, `tk_miss_wait` (protocol P-A) | **S** |
| `hexkl_token.c:288-370` | `hexkl_token_main`: the miss env, `route_log_n = 0`, the `end == n_ops` arm (forward, park, pcycles, `lm_id`) **S**; the hop arm (`tk_post` / `tk_take`, `round`, `start = h.op`) **D** → the function becomes "forward the whole list once, with the miss env" (`AEE_EBADSTATE` when any op is not resident) |
| `hexkl_token.c:372-429` | `hexkl_token_serve` | **D** |
| `hexkl_token.h:27-56, 64-77, 84-92, 129-160` | the page's slot layout prose, `HEXKL_MBOX_PING/PONG/LINE/ROW_MAX/SLOT/S2_SLOT/S1_SLOT`, `hexkl_mbox_hdr`, `HEXKL_TOKEN_MAX_ROUNDS`, the two prototypes | **R**: the miss-line offsets (`HEXKL_MBOX_MISS_REQ/ANS`, `HEXKL_MBOX_BYTES`, `HEXKL_TOKEN_TIMEOUT_US`, `HEXKL_TOKEN_POLL_US`, `HEXKL_TOKEN_E_STALE`, `hexkl_token_stats`, `hexkl_token_seq`) **S**; the slot macros go **except** that `HTP_MBOX_MISS_REQ` stays at 17152 (`htp_dspq_wire.h:102`) — the static assert `hexkl_miss_at` (`:98-100`) is replaced by one that pins 17152 directly |
| `test/htp/nntr_hvx_token.c:42-93` | `token_driver_start`: `role > 1` check | **R**: `role != 0` → `AEE_EINVALIDFORMAT` (the wire keeps the argument) |
| `nntr_hvx_token.c:131-176` | `nntr_hvx_token_run`: the `role == 0` arm **S**, the `else` (`hexkl_token_serve`) **D**; the FARF's `"S1" : "S2"` → `"token"` |
| `test/htp/nntr_hvx_dspq.c:17-19, 132-135` | comments | **S** (comment) |
| `test/htp/nntr_hvx_graph.c:53-60` | "resident MOE on a session without HMX" guard | **S** (still a valid guard on a lite open; comment loses "S2") |
| `nntr_hvx_graph.c:158` | `env->miss.post = NULL /* (hexkl_token_serve) */` | **S** (comment) |
| `test/htp/nntr_hvx_session.h:114-117, 139-150` | `token` member, `nntr_hvx_token_run` doc | **S** (comment) |
| `test/htp/nntr_hvx.idl:740-756` | `token_driver_start / _stop` | **S, signature unchanged**; comment edited in the last commit (§3, §4 step 8) |
| `test/htp/nntr_hvx.idl:721-731`, `test/htp/nntr_hvx_mailbox.c` | `mailbox_run` (the #178 probe) | **S** — it belongs to `unittest_hvx_two_sessions` (§2.4), not to the E2E path |
| `test/htp/build.sh:96-113`, `nntrainer/tensor/htp_backend/meson.build:82, 91` | source lists | **S** (no file is added or removed) |

### 2.3 Host checks

| file | what | verdict |
|---|---|---|
| `test/htp/host/token_host_check.c:315-334` `serve_thread` | **D** |
| `:342-443` `check_bit_identical` (S1 + S2 pthreads vs the one-session reference) | **R**: one session with `HTP_GRAPH_KINDS_ALL` through `hexkl_token_main` vs `hexkl_graph_forward`, 10 000 tokens, `hops == 0`, `tokens == TOKENS`; the line stays `TOKEN DRIVER BIT-IDENTICAL: … (hd64 C A C, one session, vs the one-session forward)` |
| `:543-634` `check_pool` (owner thread + S1 serve thread + S2 main) | **R**: owner thread + one `hexkl_token_main` session; `TOKEN POOL BIT-IDENTICAL` stays |
| `:636-747` `check_failures` (lost post both ways, stale header / trailer, S1's failure reaches S2) | **R**: the three miss-round failure paths, which today have no check — no owner → `AEE_EEXPIRED` after the window; an answer whose `seq2` is stale → `HEXKL_TOKEN_E_STALE`; an owner posting `rc != 0` → that code at once. `TOKEN DRIVER FAILURE PATHS OK: no owner -> AEE_EEXPIRED, stale answer refused, the owner's code reaches the token` |
| `test/htp/host/run_host_checks.sh:240-275` | the token check's prose and the three mutants: `in = tk_row(theirs)` and the trailer check **D** (their lines no longer exist → `TOKEN MUTATION DID NOT APPLY` would exit 1), the eviction mutant **S**, plus one new mutant `a->seq2 != seq` → `0` (the stale-answer path must catch it) |
| `test/htp/host/graph_host_check.c:436-481` | `GRAPH SESSION MASKS OK` block | **D** (the `limits` and every other block **S**) |
| `test/htp/host/run_inproc_e2e.sh:90-127, 309-350, 437-463, 561-578, 591-602, 604-624, 772-802` | see §4 step 5 for the line-by-line | **R** |
| `test/htp/host/mailbox_host_check.c`, `run_host_checks.sh:304-311` | the #178 probe's `MAILBOX OK` | **S** (goes with `mailbox_run`, §2.4) |

### 2.4 Device gtest

`test/unittest/unittest_hvx_two_sessions.cpp` (966 lines, `test/jni/Android.mk:1054`)
is the #178 **platform** probe (address space, hop cost, DDR sharing,
teardown), not the E2E path: it reserves its own S2 (`:392`), never uses
`HtpBackend::openSecond`, and `S1Ceiling` (`:359-369`) opens S1 only. The
runners read `CEILING s1_mmap_mib=` from it before and after every run
(`201-s3-run.sh:35`, `204-s26-run.sh:44`). **Keep the test whole, in place,
with `mailbox_run` and `nntr_hvx_mailbox.c`.** Reason: moving the probe
means an IDL entry removed (stub + skel + app rebuild for nothing) and a new
gtest binary in every staging script; the probe's Q1–Q5 lines are what LEDGER
rule 59 and ㉝ cite. Its `TwoSessions` name stays a probe's name, not a
path's.

### 2.5 Docs and runners

* Historical runners `docs/measurements/132-part-b-e2e-run.sh`,
  `201-s0/s2/s3-run.sh`, `204-s26-run.sh` and their `-stage.sh`: **untouched**
  (they describe sittings that happened; `201-s3-run.sh:14`, `204-s26-run.sh:17`
  name E0 / P / Q as they were).
* `docs/plans/132-part-b-two-session-e2e.md`, `178-second-dsp-session.md`,
  `201-htp-decode-e2e-review-gemma-moe.md:405`, `204-s26-port.md:217, 245`:
  untouched (plans are dated); a one-line "superseded by #211" note at the
  top of plan 132 only.
* `docs/htp_moe/guide/*.html`: no mention of two sessions (grep), nothing to do.
* `.claude/skills/hexagon-handoff/SKILL.md`: no E0 / P variant text (grep);
  the next runner template is written by the next sitting's plan, with the
  variant table of §4 step 9.
* LEDGER, BENCHMARK, contract §12: §6.

### 2.6 All-resident E2E ("E0") after this issue

It ceases to exist on LFM2.5-8B-A1B, and nothing replaces it. Numbers: S1's
MoE arena needs 3696 MiB for every expert (`BENCHMARK.md:481, 592`), the
one-PD FC set + lm_head takes 448 MiB beside the pool (LEDGER rule 59 a), the
PD's ceiling is 3840 MiB (`S1Ceiling`); 3696 + 448 > 3840. On the host
fixtures (hd64, lfm25-tiny) the all-resident one-PD run loads and stays as
the inproc reference (`q64-e3`, `q25-e3`, §4 step 5). On the device,
`NNTR_HTP_E2E=1` without `NNTR_MOE_CACHE_EXPERTS` fails at load in
`e2ePlaceFc` → `registerQ4m1` with `"S2 FC arena: …"`; the message becomes
`"NNTR_HTP_E2E=1: no room for the FC set beside the resident experts
(mapped=… MiB); set NNTR_MOE_CACHE_EXPERTS=<C> (28 on the S25 / S26)"`. No
code path keeps E0 alive; the BENCHMARK rows that read it stay as history.

## 3. Design

**Chosen: delete the ARM / DSP two-PD logic; keep every wire unchanged.**

* `NNTR_HTP_E2E=1` alone is the one-PD E2E. **`NNTR_HTP_E2E_PDS` is kept as
  a guard, not a switch**: unset or `1` → proceed; anything else → `throw
  std::invalid_argument("NNTR_HTP_E2E_PDS=<v>: the two-PD path was removed
  (#211); one PD is the only E2E entry, unset the variable")`. Reason: the
  runners and the user's shell history pass `_PDS=1`, and a `_PDS=2` that
  silently ran one PD would label a row with a variant that no longer exists
  (rule 36's "void, never read as A"). Five lines; removed when the next
  runner generation stops passing it (a LEDGER 3a note).
* **IDL unchanged in every signature.** `token_driver_start(fd, bytes, role,
  spin_us)` keeps `role`; the DSP accepts only 0 (`AEE_EINVALIDFORMAT`
  otherwise). `htp_dspq_token_req / _resp` keep every field (`hops`,
  `wait_us`, `hop_us` read 0 on one PD). `HTP_MBOX_MISS_REQ` stays at 17152:
  the ARM's `poolServe` (`htp_compute_ops.cpp:2392`) and the DSP's
  `tk_miss_post` read the same header, but a skel older than the lib would
  read a different page if the offset moved — the stale-skel failure is a
  silent miss-round timeout, not a `0x8000040E`. Keeping the wire means
  deleting only logic: the stub's generated code does not change, so the
  stale-stub gotcha (`htp-first-version-is-base` memory: rebuild stub and
  skel, delete `nntr_hvx_stub.o` under `builddir`) cannot bite this issue.
* **The page stays** (`kMboxBytes` = 64 KiB, mapped into S1 once): the miss
  round needs it. Only the ping / pong words and the two row slots lose their
  readers and writers.
* **`hexkl_token_main` stays as the DSP's token entry**: strip the hop loop,
  keep the miss env and the stats. The one-session all-resident forward is
  then `hexkl_graph_forward(0, n_ops)` + park + `lm_id` + counters — the
  same kernels on the same bytes, so bit-identity against today's one-PD run
  is by construction and the host check (`TOKEN DRIVER BIT-IDENTICAL`) keeps
  proving it against `hexkl_graph_forward`.
* **L0 lines stay, one side.** `token driver: L0 us/token rt= dsp_wall=
  wake= hop_us= arm_fwd= arm_us= arm_n=` and `L0 wake us/token disp= pkt=
  ret= clk_resid=` (the `s1= / s2=` pairs collapse; `disp` is the ARM post →
  the DSP read, `pkt` the DSP's handling outside its wall, `ret` the DSP
  write → the ARM read). `graph[S1]` lines become `graph:`; the
  `per-kind pcyc/token` line stays (one side); the `moe pcyc/round` line
  becomes `moe pcyc/op=… fc+dense_ffn+lm_head ms/token=…` from the same
  counters. Consumers: `run_inproc_e2e.sh:789-796` only (updated in the same
  commit); the device runners are historical.
* **Contract §2 and doc 45 §3**: untouched — no kernel, no quantizer, no DMA
  schedule, no arena budget changes; the FC set stays on S1's arena chunks,
  the pool / miss protocol stays; no CPU fallback is added for any resident
  kind.

**Rejected: keep the two-PD code behind `NNTR_HTP_E2E_PDS=2` "for
measurement".** It is 29–32 (E0) / 36–39 (P28) against one PD's 43–47 on both
S25 units and 27–32 against 31–37 on the S26, bit-identical in every sitting
(LEDGER rule 59 a, `201-one-pd.md`, `204-s26-rebaseline.md`); its one
structural advantage (the second PD's address space) is what the pool
replaced. Keeping it costs a second `graph_init`, a second queue, a second
driver, the hop machinery and three host-check variants on every future
change to `tokenForward` / `hexkl_token.c` — the Gemma 4 work (#201 S4–S6)
touches exactly those. The user decided; this plan only records why the
code does not deserve a flag.

**Also rejected: removing `role` from `token_driver_start` and the hop fields
from `htp_dspq_token_resp` now.** It is cleaner, but it is an IDL + wire
change (stub, skel, app, `nntr_hvx_stub.o`), so it would make this deletion
the one PR whose device failure mode is the stale-stub `transport failed:
err=0xe`. Filed as a 3a note; done in a later PR together with the next
genuine IDL change.

## 4. Steps (one PR, `htp/211-one-pd-only` on `htp_decode`; each commit green)

Order chosen so every commit builds and passes rung 1 on its own: tests
first widen to accept both shapes, then the code shrinks, then the tests
narrow.

1. **Host checks become one-PD (tests first).** `token_host_check.c` as
   §2.3; `run_host_checks.sh:240-275` (prose, the mutant list: drop the two
   hop mutants, add the `seq2` mutant); `graph_host_check.c:436-481` removed.
   The checks run against the unchanged `hexkl_token.c` (its `_main` with
   every op resident already takes zero hops). Gate: rung 1 `ALL CHECKS
   PASS` with `TOKEN DRIVER BIT-IDENTICAL`, `TOKEN POOL BIT-IDENTICAL`,
   `TOKEN DRIVER FAILURE PATHS OK`, `TOKEN MUTANT CAUGHT` × 2, and no
   `GRAPH SESSION MASKS OK` line.
2. **DSP: `hexkl_token.c/.h`, `nntr_hvx_token.c`.** Delete `tk_post`,
   `tk_take`, `tk_row*`, `hexkl_token_serve`, the hop arm of
   `hexkl_token_main`, the slot macros and `hexkl_mbox_hdr`; pin
   `HEXKL_MBOX_MISS_REQ == 17152`; `role != 0` refused; comments in
   `nntr_hvx_dspq.c`, `nntr_hvx_graph.c`, `nntr_hvx_session.h`. Gate: rung 1
   (step 1's lines unchanged) **and rung 2 both arches** (`UNDEFINED SYMBOLS
   OK`, `ARCH OK (V79)` / `(V81)`).
3. **ARM: `htp_compute_ops.cpp`.** `E2eState` loses the `2` members and
   `one_pd`; `sessionFor`, the mask loop, the two-graph `graph_init`, `q2`,
   the second `mmap` / driver, the `r1` packet and second read, the `s1r`
   synthesis, the `S2` teardown go; `e2ePlaceFc` keeps only the S1 branch;
   the banners and L0 lines as §3; the `_PDS` guard as §3; the no-room
   message as §2.6. `HTP_GRAPH_KINDS_S1/_S2` deleted from `htp_graph_desc.h`.
   Gate: `ninja -C build`, `*Lfm2Moe*` 6/6, `tools/htp_syntax_check.sh`.
4. **ARM: `HtpBackend`.** `openSecond`, `openSecondNow`, the six members,
   the destructor's second close. Gate: `ninja -C build`, `*Lfm2Moe*` 6/6.
5. **`run_inproc_e2e.sh` narrows to one PD.** Line by line:
   * runs `q64-pd1`, `q25-pd1`, `q25-pd1pool2` (`:346-351`) deleted — they
     are now identical to `q64-e3`, `q25-e3`, `q25-e3pool2`;
   * `E2E e3 pds=1 … == pds=2` (`:451-463`) deleted; its `hops/token=0.00`,
     `unmap_fail=0` checks move into the `e3` block (`:566-578`), whose
     expected hops become `0.00` for both fixtures and whose banner grep
     (`:574`) reads `[HTP] e2e: fc arena weights=`;
   * the pool loop (`:437-449`) loses its `pds=1` row;
   * `E2E L0 wake split closes:` (`:789-796`) greps the new labels
     (`disp= pkt= ret= clk_resid=`);
   * `E2E teardown 25e3` (`:591-602`), `E2E arena retry cap=100` (`:580-586`),
     `E2E tokens e3-run==e1-run` (`:604-614`), `E2E ppl-decode e3==e1`
     (`:617-624`), the L1 lever lines (`:772-786`) stay, on the one PD;
   * the header comment (`:90-127`) rewritten: "the one-PD token: every
     kind, the FC set on S1's arena chunks, the pool's miss rounds on the
     page; one dspqueue packet a token".
   Gate: `INPROC E2E PASS` with `E2E fwd hd64 / lfm25 e3 calls/token=1.00
   hops/token=0.00 timeouts=0/0 id_mismatch=0 ok`, `E2E e3 pool C=2 hd64 /
   C=1 lfm25 / C=2 lfm25 == e3 bit_identical=1`, `E2E eval e3==e1-hd64 /
   -lfm25 … bit_identical=1`, and the step-3 guard proven once: a run with
   `NNTR_HTP_E2E_PDS=2` must print the #211 refusal (`E2E e2e pds=2 refused
   ok`, one new line).
6. **Rung 2 + rung 3 once.** Both skels, `build_android.sh --htp`, the
   device gtests incl. `unittest_hvx_two_sessions`; md5s into the handoff.
   Gate: the skill's pass lines; `strings … | grep -c NNTR_HTP_FORWARD_KINDS
   ≥ 1`.
7. **Docs commit** (§6) + the comment-only edit of `nntr_hvx.idl:740-756`
   (the prose names one role; signatures untouched → `build.sh` regenerates
   a stub whose code is byte-identical; confirm with `md5sum
   test/htp/generated/nntr_hvx_stub.c` before / after). Gate: rung 0,
   `tools/htp_syntax_check.sh`.
8. **Device — unavoidable, short (S26, agent-run adb).** Variants (≤ 4):
   **A** (unchanged reference, hybrid, nothing set), **Q28** (`NNTR_HTP_E2E=1
   NNTR_MOE_CACHE_EXPERTS=28`), **Q28p** (Q28 + `NNTR_HTP_E2E_PDS=1`, must be
   byte-identical in log shape to Q28 — the guard's accept side), and one
   `NNTR_HTP_E2E_PDS=2` launch that must refuse at load (not a timed run).
   Prompt 512; G 64 r1 / r2, G 512 r1; cool start per block; ceiling before
   and after. Pass: §1's device line. Gen 1024 is not run (the deletion is
   G-independent; the Q28 band at 1024 is `201-one-pd.md` sitting 2's).
9. **The next runner template** (for the sitting after this one, written by
   its plan, not here): variants A and Q<C> only; no E0 / P<C>; the
   `one PD` grep line (`204-s26-run.sh:91`) drops.

## 5. Risks

* **A behavioural difference hidden by the host.** The one DSP-side change
  with a runtime effect is `hexkl_token_main` losing its hop arm; the
  `TOKEN DRIVER BIT-IDENTICAL` check (10 000 tokens vs `hexkl_graph_forward`)
  and `E2E eval e3==e1 … bit_identical=1` cover it on scalar kernels; the
  S26 sitting's text == A covers the HVX build. Made visible by: the text
  column and `id_mismatch=0` in the handoff table.
* **Stale skel / stub on the device.** No signature moves, so a stale skel
  would still answer — which is exactly why the sitting pushes the md5s of
  both skels and the lib and prints `md5sum` on the device before the first
  run (`MD5 OK`), and why `role != 0` is refused: an old lib passing role 1
  to a new skel fails loudly at `token_driver_start`, not with a silent
  timeout.
* **DVFS / thermal drift.** Q28's tok/s on the S26 reads 31–37 across
  queues and blocks (`204-s26-rebaseline.md`); the sitting's A is the
  reference for prefill, Q28 is compared only to the band. Made visible by:
  `therm.log` per block and the A-first order.
* **Address-space budget unchanged.** The FC set's 448 MiB and the 28-pool
  are the same mappings as today's Q28; the ceiling cell start → end is the
  LEAK check (rule 59 c). A `fastrpc_mmap` refusal at C = 28 on the S26 is
  pre-existing (plan 204 §5: use 24 and say so), not this issue's.
* **DMA rate**: not touched (no DMA code changes). Listed only because the
  handoff table carries `dmaq=` per row anyway.
* **Deleting too much.** `placeOn`, `releaseQ4m1`, the page, the pool
  server and `mailbox_run` are shared or probe code (§2); the `*Lfm2Moe*`
  6/6 and the A path's untouched banners are the guard that the hybrid is
  not entered by changed code.

## 6. Docs to update

* **`docs/htp_moe/LEDGER.md`**: §1 rule 59 gains one sentence ("the two-PD
  path was removed by #211 on 2026-10-01; its numbers stay as the reason");
  §3a note: `NNTR_HTP_E2E_PDS` is a guard that accepts 1 and refuses the
  rest, to be dropped with the next IDL change together with `role` in
  `token_driver_start` and the hop fields of `htp_dspq_token_resp`; §3a
  note: `unittest_hvx_two_sessions` is the #178 platform probe and the
  runners' ceiling cell, kept whole.
* **`docs/htp_moe/BENCHMARK.md`**: Method, cycle paragraph: "E0 / P<C>
  variants retired with #211; from here E2E = one PD, variant letter Q<C>";
  Log: the deletion sitting's row (A / Q28 / Q28p, texts, ceiling), no "now"
  move.
* **`docs/plans/0001-htp-moe-decode-agent-system.md` §12**: new row
  `2026-10-01 | **Two-PD E2E path removed (user).** One PD (`NNTR_HTP_E2E=1`
  + `NNTR_MOE_CACHE_EXPERTS=<C>`) is the only NPU E2E entry on `htp_decode`;
  E0 / P<C> cease to exist (3696 + 448 > 3840 MiB on one PD); the wire (IDL,
  `htp_dspq_wire.h`, page offsets) unchanged; `unittest_hvx_two_sessions`
  kept as the #178 probe and the ceiling cell. Numbers: S25 43–47 vs 36–39
  vs E0 29–32; S26 31–37 vs 27–32 vs 29–34 (#211)`.
* **`docs/plans/132-part-b-two-session-e2e.md`**: one line at the top,
  "superseded by #211 (one PD only); kept as the record of the design and
  its measurements".
* **`docs/measurements/211-one-pd-only.md`**: the sitting's handoff
  (variants A / Q28 / Q28p + the refusal check, md5s, texts, ceiling).

## 7. Expected diff

| area | removed | added / changed |
|---|---|---|
| `htp_compute_ops.cpp` | ≈ 260 (`sessionFor` 10, mask loop 16, two-graph init 25, `e2eStart` 25, `e2eTeardown` 95, `tokenForward` 45, `e2ePlaceFc` 12, `E2eState` 18, misc 14) | ≈ 50 (one-side banners, the guard, the no-room message) |
| `htp_backend.cpp/.h` | ≈ 135 | ≈ 2 |
| `htp_graph_desc.h` | 12 | 0 |
| `hexkl_token.c/.h` | ≈ 250 (`tk_post/tk_take/tk_row*` 70, `_serve` 58, the hop arm 45, header prose + macros + prototypes ≈ 75) | ≈ 25 |
| `nntr_hvx_token.c`, `nntr_hvx_dspq.c`, `nntr_hvx_graph.c`, `nntr_hvx_session.h`, `nntr_hvx.idl` | ≈ 25 (comments, the `else`) | ≈ 15 |
| `token_host_check.c` | ≈ 300 | ≈ 150 |
| `graph_host_check.c` | 47 | 0 |
| `run_host_checks.sh`, `run_inproc_e2e.sh` | ≈ 70 | ≈ 35 |
| docs | — | ≈ 40 |
| **total** | **≈ 1 100** | **≈ 320** |

No file added or removed except `docs/measurements/211-one-pd-only.md`;
no build-list change; no IDL signature change; no `generated/` code change.
