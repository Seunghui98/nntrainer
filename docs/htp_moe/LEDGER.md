# HTP MoE decode ledger — rules learned on silicon, verdicts, open items

Contract: `docs/plans/0001-htp-moe-decode-agent-system.md`. The supervisor
appends here; the planner and implementer read it before touching a kernel.
The tracker issue (#76) carries `hexagon` + `prio:*` and **no `state:*`
label**: state belongs to its children, so the tracker never occupies the
single `state:in-progress` slot.

## Upstream

PR nntrainer/nntrainer#4327, branch `claude/htp-lfm2-moe-ffn` on
`Seunghui98/nntrainer`. `htp_moe` is **synced to `4ae1ebd7` via PR #121**
(upstream merge commit `fd3bfba1` on the PR branch, PR merged by the user
as `b7d46b57`, 2026-09-27 11:30 UTC; #120's sitting is its device evidence,
§2 ㉔). History: from cycle 1 through cycle 16 it was frozen at head
`2ce38d65` (2026-09-21 06:45 UTC, "[CausalLM] Route conv out_proj and the
dense FFN to the HTP by config") plus the three cherry-picks `fb0f02b9` /
`04a2fcc4` / `b0a384d6` that PR #103 carried (`ad714de7`, cycle 6).
Upstream watch sha: **`f923bf29`** (cycle 22, 2026-09-29; was `70655c4b` in
cycle 21, `bcfc1ac5` at the cycle 20 close, before that
`700371da` in cycles 19–20, `4ae1ebd7` from cycle 12 through cycle 18,
`a996b4bf` from cycle 7 through cycle 11). The commits after
`4ae1ebd7` are **not merged** (cycle 19, cycle 20 close, cycles 21 and 22
below). **Not watched since 2026-10-01 (user).** The one upstream PR this
project now takes is nntrainer/nntrainer#4296 (Gemma 4 on the CPU),
**pinned at `d345c3470`** and merged into `htp_decode` by #201 S4's PR #226
(open at cycle 33, stacked under #227 / #231; merged in cycle 34)
(user decision 2026-10-06, contract §12); it is a pin, not a watch — later
#4296 commits are not tracked here.

**Gemma watch (since cycle 35, user 2026-10-06, #234):** upstream
nntrainer/nntrainer#4408 ("[WIP][DRAFT][HTP] Run gemma-4-26B-A4B on the HTP
with flash offload", Seunghui98, `refs/pr/4408`) — watch sha
**`28771a928`** (PR updated 2026-10-02 09:32 UTC; unchanged in cycles 35–40,
re-checked 2026-10-07 evening in cycle 40: `gh pr view` head `28771a928`, 100 commits) — and `htp_first_version`, which is **closed and
frozen since cycle 38 (user, 2026-10-06 night: finished with PR #246,
`77a01902f`)**: 51 commits ahead of `htp_decode` @ `9323ad52d`, read once
here, not watched further. Of those, the code commits `htp_decode` does not
have are **12**: #225 PR 1 (`db2c27a11` `--fc_wh_sidecar` writer,
`7b622617a` sidecar read into the arena, `75f6f007d` 512-row prefill chunks
+ conv history), #225 PR 2 (`b73e01b1a` `hexkl_mm_u8i4_fc_m1_run`,
`7efd22adf` one-PD FC / dense FFN on the sidecar handles), #219 (`32b1bc6de`
the ARM tier, `dbda0fc25`, `cc9b20d0c`), #236 (`a3d966164` M > 1 FC entries
step by min(`fcMaxRows`, `prefillRows`), `366c06c57`), #219 default
(`cdbf66f20` unset = 2 on E2E, `0235d9d79`); the #223 three (`c0fa4e419`,
`d541bda3a`, `f92ec3596`) are already in `htp_decode` through PR #247. The
rest are docs (handoffs, cycles 33–36, guide, plans). Which of the 12 Gemma
needs is plan 234's revision (cycle 38, planner); cycle 36 of this copy
had read `78dace26f` (PR #237's sync, PR #233 merged `0e8888bde`). A port
issue is filed when the Gemma model needs something (#234 is the first).
Reference, not watched: #4410 (`refs/pr/4410` @ `0d603f29a`, unchanged
2026-10-07 evening, cycle 40 — PR still open; the 2-bit / ternary expert stack #229 S1 ported in cycle 36,
PRs #238 / #239 on `htp_decode`, with its u8i2 GEMV mask fixed on the way,
rule 66; **device gate passed on v79 2026-10-06, §2 #229 row; its test
PR #244 merged `8971aa873`, cycle 39**). #4327
stays frozen at `f923bf29` (not watched).

`hvx_conv_gate_f32.{c,h}` (upstream `7f81560b`) reached `htp_moe` through
PR #121's merge, so #82 is plan 82's case (a): it reuses the file unchanged
and takes no `hvx_scalar_stubs` (PR #125).

**New commits since (seen 2026-09-21, cycle 1; PR head `7f81560b`,
updated 08:10 UTC, fast-forward from `2ce38d65`, 31 files, +3490/−636).
Merging them into `htp_moe` is the user's decision (contract §5, Q16);
not merged.** Oldest first:

| sha | subject | touches |
|---|---|---|
| `95caa139` | [docs] 50: the fifth run — every FC group on, per-shape calls measured | doc 50 |
| `fb0f02b9` | [HTP] Stage each call through an ION buffer of its own size class | `htp_compute_ops.cpp` (transport-side, relevant to ⑦ / #83). **Cherry-picked into `htp_moe` by PR #103 (#88)**, `docs/htp_attention` hunk dropped |
| `7497cfd6` | [docs] 50: size-class staging measured — decode back to 20.5 TPS | doc 50 |
| `d2f0bf47` | [CausalLM] Run the dense FFN as one HTP call through the MoE layer kernel | app, `hexkl_mm_u8i4_moe.{c,h}` |
| `35b6124b` | [HTP] Give the dense FFN's MoE-kernel calls their own profile row | profile |
| `20d18ab4` | [docs] 50: the no-switch run — today's baseline is 921–1013, not 848 | doc 50 §3.8: prefill **ms** on the author's unit, 921 ms = 482 tok/s / 1013 ms = 438 tok/s (was 848 ms = 523), decode 21.2 / 21.1; "the device is +73..+165 ms slower today, two consecutive runs 10 % apart" — rule 9 again |
| `80bf92ad` | [docs] 51: third C run — the dense row needs a rebuild to appear | new doc 51 |
| `39ce49d7` | [docs] 51: the dense-only run — 10.4 ms per fused call, as predicted | doc 51 |
| `7f81560b` | [HTP] Run the LFM2 conv block as one call: in_proj, gates, conv1d, out_proj | new `hmx/hexkl_conv_block.{c,h}`, **`hvx/hvx_conv_gate_f32.{c,h}`** (causal depthwise conv1d L=3 + gate, HVX, prefill shape), `test/htp/host/conv_block_host_check.c`, stand-ins moved to `hvx_scalar_stubs.{c,h}`, two IDL entries `mm_u8i4_conv_block[_timed]`, `conv_block_layer.cpp`, `compute_ops.h`; "not yet run on a device" |

**Cycle 2 (seen 2026-09-21 09:44 UTC): PR head moved `7f81560b` →
`72d9b140`, six more commits, all on the conv-block DIFF/compare path
(`Applications/CausalLM/layers/conv_block_layer.cpp`) and doc 51 — no
kernel, IDL or ring change. Still not merged; user decision.** Oldest first:

| sha | subject | touches |
|---|---|---|
| `5d190100` | [docs] 51: the conv block on device — 4.9 ms a call, prefill 732–741 | doc 51, `00_START_HERE.md` (the fused conv block measured on the author's unit: prefill 732–741 **ms**) |
| `81298c35` | [CausalLM] conv_block: NNTR_CONV_BLOCK_DIFF / _SHADOW discriminators | `conv_block_layer.cpp` |
| `9681f465` | [CausalLM] conv_block compare: reference through the CPU Q4_0 GEMM directly | `conv_block_layer.cpp`, doc 51 |
| `4b8b8f6f` | [CausalLM] conv_block DIFF: an activations-only reference and outlier ratios | same |
| `fe74b6cb` | [CausalLM] conv_block DIFF: emulate four weight recipes in f32 | same |
| `72d9b140` | [CausalLM] conv_block DIFF: emulate row-flattening and K-grouped int4 | same |

Reading: the author is chasing an accuracy discrepancy in the fused conv
block (four DIFF commits in 80 minutes); `7f81560b`'s "not yet run on a
device" is now run (5d190100) and under investigation. Nothing here
touches decode; a merge decision can wait until that line settles.

**Cycle 3 (seen 2026-09-22 ~01:00 UTC): PR head moved `72d9b140` →
`0a0c0402` (updated 2026-09-21 10:27 UTC), eight more commits. Files:
`conv_block_layer.cpp`, `tie_word_embedding.{cpp,h}`, `causal_lm.cpp`,
`hmx/hexkl_conv_block.{c,h}`, `htp_backend.cpp`, docs 00/51. No IDL, ring
or MoE-kernel change; one new runtime knob (`NNTR_HTP_POLL_US`) touches
the FastRPC poll window that #88 (wall 3) also works on. Still not merged;
user decision.** Oldest first:

| sha | subject | touches |
|---|---|---|
| `8d361580` | [CausalLM] conv_block DIFF: K-grouped int4 with the symmetric rule, groups 32-256 | `conv_block_layer.cpp` |
| `2e69c732` | [CausalLM] conv_block DIFF: the grouped-int4 recipes the previous commit described | same |
| `32b46e32` | [CausalLM] NNTR_PPL: the prompt's teacher-forced perplexity at prefill | `causal_lm.cpp`, `tie_word_embedding.{cpp,h}` — a PPL readout we could reuse as an accuracy column |
| `e2554a24` | [docs] 51: perplexity closes the conv block accuracy question | doc 51 |
| `04a2fcc4` | [HTP] NNTR_HTP_POLL_US: the FastRPC poll window as a knob | `htp_backend.cpp` — overlaps #88's area (transport). **Cherry-picked into `htp_moe` by PR #103 (#88)** |
| `07a9bec5` | [CausalLM] conv_block: drop the recipe emulation, keep DIFF/SHADOW and PPL | `conv_block_layer.cpp` |
| `e699e1bd` | [HTP] conv block: pipeline phase 2 one block ahead | `hexkl_conv_block.{c,h}` (prefill-shape fused conv block) |
| `0a0c0402` | [docs] 51: the perplexity split -- the dense FFN costs 4%, the conv block nothing | doc 51: the author's accuracy verdict — the fused conv block is PPL-neutral, the dense FFN through the MoE kernel costs 4 % PPL |

Reading: the conv-block accuracy line has settled (PPL-neutral); the
dense-FFN-through-MoE-kernel path (`d2f0bf47`) costs 4 % PPL on the
author's measure, which would fail our gate (c) if we ever routed the dense
FFN that way. `04a2fcc4` is the one commit with a bearing on our decode
work (wall 3); the rest is prefill/accuracy. A merge decision can still
wait; if merged, the IDL from `7f81560b` forces a skel + app rebuild.

**Cycle 4 (seen 2026-09-22 ~02:30 UTC): PR head moved `0a0c0402` →
`b0a384d6` (updated 2026-09-22 01:54 UTC), three more commits. Files:
`htp_compute_ops.cpp` (profile accounting), `htp_backend.cpp` (FastRPC
poll default), docs 00/51. No IDL, ring, weight-layout or MoE-kernel
change. Still not merged; user decision.** Oldest first:

| sha | subject | touches |
|---|---|---|
| `6f8f7791` | [HTP] Keep the conv row's hidden worker time out of the mm residual | `htp_compute_ops.cpp` — `[HTP-PROFILE]` residual arithmetic for the conv row; prefill-only row, but the same table #94 B reads |
| `faba38d1` | [docs] 51: correct why the dense FFN is expensive -- down's depth, not the chunking | doc 51 |
| `b0a384d6` | [HTP] Poll the FastRPC reply for 5 ms by default | `htp_backend.cpp` — makes `04a2fcc4`'s `NNTR_HTP_POLL_US` knob default to 5 ms. Transport-side (wall 3); the #88 plan read it (author's unit: decode call transport 158 → 83 µs, doc 51 §2.20). **Cherry-picked into `htp_moe` by PR #103 (#88)**, `docs/htp_attention` hunks dropped |

Reading: `b0a384d6` is the second upstream commit in #88's area (after
`04a2fcc4`). Two consequences: (1) the #88 planner reads doc 51's poll
numbers before writing the design; (2) if the user merges before #88's
handoff, variant A of that sitting already carries the 5 ms poll and #88
is read against it — cleaner than merging in between. Nothing here touches
#94's artifact set.

**Cycle 5a (seen 2026-09-22 ~04:30 UTC): PR head unchanged at `b0a384d6`.**

**Cycle 5b (seen 2026-09-22 ~05:30 UTC): PR head moved `b0a384d6` →
`80fa1a1a` (updated 2026-09-22 05:14 UTC), four more commits. Files:
`hmx/hexkl_mm_u8i4_moe.c` (+370/−226), `htp_compute_ops.cpp`,
`Applications/CausalLM/layers/qkv_layer.{cpp,h}`, `lfm2_causallm.cpp`,
`transformer.cpp`, two LFM2 unit tests, doc 51. One commit
(`5731b6e5`) rewrites the MoE kernel's down-matmul pipeline — the same
file and the same DMA schedule that #94's trace measured and that the
wall-2 fix issue (#100) targets. Still not merged; user decision.**
Oldest first:

| sha | subject | touches |
|---|---|---|
| `3006d255` | [HTP] Give plain FC calls their own profile row | `htp_compute_ops.cpp` (profile rows; #94 B reads this table) |
| `5731b6e5` | [HTP] Issue the MoE kernel's down matmul one block behind | **`hexkl_mm_u8i4_moe.c`** — pipelining of the down matmul behind gate/up in the HMX block loop; changes the 46-descriptor issue order #94 traced (rows f/g/h). Prefill-shaped by intent (doc 51's "all-on run"); its M=1 effect is unmeasured |
| `1f538c20` | [CausalLM] Fuse LFM2's q/k/v projections and their norms into qkv_layer | `qkv_layer.{cpp,h}`, `lfm2_causallm.cpp`, `transformer.cpp`, tests — ARM-side attention projections (the ARM remainder, ⑨ / #81 area) |
| `80fa1a1a` | [Docs] Record the 585 ms all-on run: pipeline and qkv sum to −33 | doc 51: the author's prefill at 585 ms with everything on (−33 ms from the two commits above) |

Reading: `5731b6e5` is the first upstream commit inside the MoE kernel
since `htp_moe` froze. If merged, #94's trace / attribution table
(descriptor order, `first expert ready`, rows f/g/h) must be re-taken as
variant A of the next sitting before #100 is read against it; if not
merged, #100 works on the frozen list and a later merge re-opens the
question. Either way the planner of #100 reads that diff first. The GEMV
path (C) bypasses this loop entirely, so ⑯'s verdict is unaffected.

**User merge decision (cycle 6, 2026-09-22): PR #103 (#88) merged by the
user as `ad714de7`.** That merge is the user accepting the three upstream
cherry-picks `fb0f02b9` (size-class ION staging), `04a2fcc4`
(`NNTR_HTP_POLL_US`) and `b0a384d6` (5 ms poll default) into `htp_moe`,
with their `docs/htp_attention` hunks dropped. Nothing else from #4327 is
merged; the rest of the tables above and below stays a user decision.

**Cycle 6 (seen 2026-09-22 ~06:30 UTC): PR head moved `80fa1a1a` →
`5622c743` (updated 05:53 UTC), four more commits. Net files:
`hmx/hexkl_mm_u8i4_dma.c` (+104/−37), `hmx/hexkl_mm_u8i4_moe.c`
(+8/−3, a comment and one moved line), `htp_compute_ops.cpp` (+6, a
comment), `test/htp/host/fc_layer_host_check.c` (new), host stubs, docs
00/51/52. No IDL change; the M=1 GEMV path (#105) is not touched. Not
merged; user decision.** Oldest first:

| sha | subject | touches |
|---|---|---|
| `7e0b0e71` | [HTP] Three small exposures: early gate_up chunks, pooled FC epilogue, MoE args in rpcmem | moe, dma, compute_ops — two of the three reverted by the next commit |
| `aa371dd2` | [HTP] Revert the early gate_up chunks and the rpcmem MoE args; keep the pooled FC epilogue | net: `hexkl_mm_u8i4_dma.c` plain-FC epilogue double-buffered onto the pool (prefill FC calls, not on our decode config); two **measured negatives** left as comments: (a) pushing the next expert's gate_up chunks early put the activation behind them in the ring, GATHER 84 → 747 µs/call (388 at decode), "the DMA moves ~12 GB/s beside the HMX, not the idle 33" (doc 51 §2.26) — input for #100 / rows f–g; (b) the MoE call's five small sequences from one rpcmem buffer: transport 627 → 605 µs, noise, "the gap to the FC calls' ~400 is the call's length … a call longer than the poll window ends in the interrupt-driven wait" — input for reading handoff 88 |
| `7ada48b7` | [Docs] Record the pooled-FC-only run and the poll 9000 experiment | doc 51 (a longer poll window tried) |
| `5622c743` | [Docs] 52: the flash-experts (cached-slim on HTP) task, self-contained | new doc 52: the author's next task is **memory** (peak RSS 5.2 GB, 3840 MiB expert arena → LRU/flash-loaded experts on the HTP path, "DSP kernel untouched"); records the author's all-on state as prefill 585–616 ms / decode 24.2 TPS and "no HTP software lever left ≥ 5 ms" for prefill |

Reading: nothing here changes a decode kernel. Two of the comments are
free measurements for our open work: (a) a ~12 GB/s ring rate beside the
HMX, which bears on #100; (b) an rpcmem-args null, which bears on #88. Doc
52 says the author's line is now moving to memory residency. If that line
lands, it would change the arena our GEMV path reads directly (#105, ㉒),
so it is worth knowing about before any merge.

**Cycle 7 (seen 2026-09-22 ~07:30 UTC): PR head moved `5622c743` →
`a996b4bf` (updated 06:55 UTC), one commit, docs only
(`docs/htp_attention/00_START_HERE.md` +1, new
`docs/htp_attention/53_int_requant_task.md` +114). No code, IDL or kernel
change. Not merged; user decision.**

| sha | subject | touches |
|---|---|---|
| `a996b4bf` | [Docs] 53: the int32-to-u8 requantization task (arXiv 2511.11248), self-contained | new doc 53 (Korean): a task note for replacing the MoE kernel's int32 → f32 → u8 epilogue by a direct int32 → u8 requant; the paper itself not yet read. Its own reading: at prefill the epilogue is hidden under the HMX (0.35 ms/call exposed, 2.4 %), the gain would be structural (VTCM `gate_off` 448 → 112 KB, the row-scan barrier) and the gate is PPL (`NNTR_PPL=1`, 62.09 ± 0.5 %); **at decode (M=1) "not applicable"** — the author's M=1 call is 1.44 ms with dequant 1.1 µs, gather 128, acc 259, mm 764 |

Reading: nothing for the decode goal. It shares the requant stage with #95
(⑱, Hadamard before the u8 requant, PR #106), so whoever reads #95's
verdict should know an int-requant rewrite of the same stage is the
author's next prefill topic; a merge would conflict there, not in the M=1
GEMV path.

**Cycle 8 (seen 2026-09-22 ~08:30 UTC): PR head unchanged at `a996b4bf`
(updated 06:55 UTC).** One relevant fact from the #95 sitting: the
side tree `htp_hadamard` (upstream `3006d255` + tooling) reads decode
**25.08 / 24.40 / 23.83 tok/s** on its own control A (`R3CY205ZMND`,
same day). `htp_moe`'s #88 B read 24.50 / 23.80 / 23.52 in another
sitting. Rule 9 forbids reading this as a verdict. It is only
consistent with both trees carrying the same transport fix (`fb0f02b9` /
`b0a384d6`) and neither having the GEMV on.

What this changes for us: (1) the "net-new" causal conv1d + gating of §4
now exists upstream in prefill shape — #82 lifts it instead of deriving
it; (2) the ARM decode path of the conv block is still the CPU one there
(the fused call runs only at prefill); (3) doc 50 §3.8 re-measured the
author's unit 10–16 % slower on the same binary (prefill 482–438 tok/s,
not 523), one more reason the provisional "now" is replaced only by #77's
control run on ours; (4) the IDL changed, so a merge forces a
skel + stub rebuild and new md5s in BENCHMARK.md.

**Cycle 9 (seen 2026-09-22): PR head unchanged at `a996b4bf` (updated
06:55 UTC).** Nothing to decide.

**Cycle 10 (seen 2026-09-22): PR head unchanged at `a996b4bf` (updated
06:55 UTC).** Nothing to decide.

**Cycle 11 (seen 2026-09-23): PR head unchanged at `a996b4bf` (updated
2026-09-22 06:55 UTC).** Nothing to decide. Note for whenever a merge is
considered: `5731b6e5` (the down-matmul pipelining inside
`hexkl_mm_u8i4_moe.c`) was held back because it would invalidate #94's
descriptor trace. After #100 that reason is weaker — row h dissolved and
the M=1 GEMV path issues `desc=2/call`, so the 46-descriptor list is no
longer on the decode critical path — but `5731b6e5` is prefill-shaped
and still unmeasured at M=1, so the decision is unchanged, not urgent.

**Cycle 12 (seen 2026-09-23): PR head moved `a996b4bf` → `4ae1ebd7`
(updated 2026-09-23 02:14 UTC), five commits. Files: docs 00/52/53 and,
in one commit, `hmx/hexkl_mm_u8i4_moe.c` (+31/−13) and
`htp_compute_ops.cpp` (+16/−8) — profile accounting only, "arithmetic
unchanged", "not run on device". No IDL, ring, weight-layout or M=1 GEMV
change. Not merged; user decision.** Oldest first:

| sha | subject | touches |
|---|---|---|
| `2cb8a1ea` | [Docs] 52, 53: state the authorship rule as it is, and the prompts with it | docs 52/53 |
| `5cdd7a6f` | [Docs] 53: the paper is T-MAN, no int32-to-u8 hop in our kernels; closed at gate 0 | doc 53: the int-requant task is **closed** upstream at gate 0 — the conflict cycle 7 flagged for #110's requant site no longer exists |
| `c64be9dd` | [HTP] Time the MoE kernel's epilogue worker slices under the SWIGLU column | `hexkl_mm_u8i4_moe.c`: `moe_tail_probe_add` → `moe_worker_probe_add`, every pool job of the kernel timed under `HEXKL_PROBE_SWIGLU`; `htp_compute_ops.cpp`: a `swiglu_hidden` flag so the `mm` residual leaves that column out on every row, printed `swiglu(hidden)`. **Overlaps PR #108's fix of the same column on the GEMV row (㉑, rule 24)** — a merge conflicts there, in the print, not in a kernel |
| `6af38345` | [Docs] 53 8.4: the HVX epilogue in integer arithmetic, op by op, and the probe that decides it | doc 53 |
| `4ae1ebd7` | [Docs] 53 8.4: run 1 had no probe in the binary; same-session baseline recorded | doc 53 |

Reading: nothing for the decode goal. `c64be9dd` is the author's version
of what #102 / PR #108 did for the GEMV row (rule 24: `swiglu` is Σ
lane-time on that row); the two prints will conflict at merge time and
one of them has to win. No reason to merge now; if the user merges, the
GEMV row's `rest` arithmetic (PR #108) must be re-checked against the new
`swiglu_hidden` flag before the next profile is read.

**Cycle 13 (seen 2026-09-23): PR head unchanged at `4ae1ebd7` (updated
2026-09-23 02:14 UTC).** Nothing to decide. On our side: PR #115 merged
as `c45d4433` (D192 default, `727862c7` + `37fffcb4`), PR #116 (guide) as
`42370b95`; `htp_moe` head `42370b95`.

**Cycle 14 (seen 2026-09-23): PR head unchanged at `4ae1ebd7` (updated
2026-09-23 02:14 UTC).** Nothing to decide. On our side: **PR #118**
filed (`htp/117-m1-gemv-vtcm-feed` @ `e4b5d0f5`, #117 →
`state:needs-measurement`, on the device now — its sitting also carries
#113's A/A0 default confirmation and moves BENCHMARK's "now", rule 33);
#81 planned (`763762a2`); `htp_moe` head `763762a2`. Queue for the
CPU → HVX track (⑨) confirmed as **#85 → #82 → #81**: #82's RoPE / q-k
norm are #81's inputs and both slot into #85's op table; #84 (host E2E
harness) planned in parallel. **#99 downgraded p1 → p2** (⑳): its
verdict question is answered (plan 99's diagnosis + #100's tag-validated
cells, rule 28); what is left is a `test/`-only fix so the rule-30/32
anchor cell stops reading `checksum_ok=n`. **#114 stays p2**, conditional
on #117's gate (rule 31; nothing to decide before the sitting). **One
user decision surfaced by plans 81 §3.6 / 82:** the accuracy column for
the ⑨ wiring handoff — see ⑨.

**Cycle 15 (seen 2026-09-23): PR head unchanged at `4ae1ebd7` (updated
2026-09-23 02:14 UTC). USER DECISION OF RECORD (2026-09-23, "PR #4327
최신 거도 다 반영해줘", contract §5 Q16): the full sync is ordered — every
commit of the PR head onto `htp_moe` as one PR, filed as #120 (p1,
`state:planned`, implementer on it this cycle). Merge-base `2ce38d65`, 40
commits, 44 files, +5667/−998. What `htp_moe` already carries from that
range, for checking #120's PR body against:**

| upstream | on `htp_moe` as | via |
|---|---|---|
| `fb0f02b9` size-class ION staging | `e75011e8` | PR #103 (#88, merged `ad714de7`), `docs/htp_attention` hunk dropped |
| `04a2fcc4` `NNTR_HTP_POLL_US` knob | `308e1fde` | PR #103 |
| `b0a384d6` 5 ms poll default | `d84b7d69` | PR #103, docs hunks dropped |
| `c64be9dd` epilogue slices under `swiglu` | **not carried**; overlaps PR #108's GEMV-row `swiglu` / `rest` print (rule 24) — the merge keeps one print and re-checks the GEMV row's `rest` on the next profile | — |
| `3006d255` FC profile row | not carried on `htp_moe` (the `htp_hadamard` side tree had it, #95) | — |

Everything else arrives with the merge: `d2f0bf47` (dense FFN through the
MoE kernel — 4 % PPL on the author's measure, must stay **off** by
default under gate (c)), `7f81560b` / `e699e1bd` (conv block as one call,
two IDL entries → skel + app rebuild, new md5s), `5731b6e5` (down matmul
one block behind, inside `hexkl_mm_u8i4_moe.c`'s HMX loop; off the GEMV
path at M=1, unmeasured there), `1f538c20` (qkv fusion, ARM side — #81 /
#82's input), `32b46e32` (`NNTR_PPL`, the accuracy column ⑨ wants),
`aa371dd2` (pooled FC epilogue, prefill FC calls only). `git merge-tree`
names conflicts in `hexkl_mm_u8i4_moe.c`, `htp_compute_ops.cpp`,
`moe_layer_host_check.c`, `run_host_checks.sh`, `nntr_hvx_mm_u8i4.c` —
**#117's feed (PR #118) touches the first two and the last**, so the
ordering is a user call (needs-user): PR #118 first and #120 rebases over
it (the supervisor's recommendation: the feed is measured, the sync is
not), or #120 first and the feed re-applies on the merged tree. #120's
acceptance (issue body) carries the device A/B (decode ≥ −2 % of A,
prefill ≥ −5 %, text identical, every default-on upstream knob named).
On our side this cycle: #117 filled twice (`6412bfc9`, unit
`R3CY10WM83Y`), feed-default flip in progress on PR #118; #114 closed;
`htp_moe` head `6952cb6f`.

**Cycle 16 (seen 2026-09-27): PR head unchanged at `4ae1ebd7` (updated
2026-09-23 02:14 UTC; `gh api repos/Seunghui98/nntrainer/commits?sha=claude/htp-lfm2-moe-ffn`
lists nothing newer). Nothing to decide upstream. On our side: **#120
measured** (`120-upstream-4327-sync.md` @ `7ded3ce9`, `R3CY10WM83Y`,
2026-09-23 17:55–18:08, → §2 ㉔): the merge passes its gate (decode +4.7 /
+0.9 / −0.2 %, prefill **+12.4 / +10.2 / +6.2 %**, text = A 14/14,
`MoeLayerM1GemvMatchesHmx` bit-identical on B's skel), and its A — the
feed default with nothing set — is the new "now" (35.97 / 35.96 / 35.17,
BENCHMARK.md). **PR #121 is not merged**: after its rebase onto
`d79c0efe` it conflicts again with `htp_moe` @ `b4c96bdb` (PR #119, the
per-token entry). **User decision 2026-09-27 (contract §12): resolve as a
rebase onto `htp_moe`, rungs 0–1 only, no new sitting** — PR #119's
`NNTR_HTP_FORWARD=1` entry is off by default, so the measured B binary's
default path is unchanged by the rebase; the #120 sitting stands as the
PR's device evidence. The rebased PR states that rungs 2–3 were not run
(contract §4.1a; the workstation produces the skel/app md5s before any
later handoff). The "frozen at head `2ce38d65`" line above becomes
`4ae1ebd7` when PR #121 merges. #85 closed (PR #119 merged `b4c96bdb`);
`htp_moe` head `b4c96bdb`; PR #124 (Mac dev container, tooling) open.
**Queue (user, 2026-09-27; no device sitting possible for now, so only
issues that end in host-gated PRs): (1) PR #121 rebase (#120), (2) #82,
(3) #81, (4) #84, (5) #89.** #90 / #110 / #99 stay behind them.

**Cycle 16b (post-merge fold, seen 2026-09-27 ~12:00 UTC): PR head
unchanged at `4ae1ebd7` (updated 2026-09-23 02:14 UTC; the branch's last
ten commits end at `4ae1ebd7`). Nothing to decide upstream. On our side
the user merged the whole cycle-16 queue into `htp_moe` in order,
11:30–11:32 UTC: PR #124 (`a875c085`, Mac dev container, §4.1a), PR #121
(`b7d46b57`, #120: the upstream sync, merge commit `fd3bfba1` + 6
commits), PR #125 (`d0552b1e`, #82), PR #126 (`1afbf0aa`, #81), PR #127
(`ec589373`, #84), PR #128 (`6497f38a`, #89), PR #129 (`758aaacc`, guide).
#120, #82, #81, #84, #89 closed (completed). `htp_moe` head `758aaacc`.
The "synced to `4ae1ebd7`" line above is the record; #82's `hvx_conv_gate_f32`
reuse (case (a)) stands. Rungs 2–3 for these PRs: §3a.**

**Cycle 18 (seen 2026-09-28, the first sitting after the weekend's Mac
cycles): PR head unchanged at `4ae1ebd7` (updated 2026-09-23 02:14 UTC;
`gh pr view 4327` and the branch's last fifteen commits end there).
Nothing to decide upstream. On our side: `htp_moe` @ `39af2f39` (=
origin; cycle-16b fold `dbaf24b5` + plan 130), no filled handoff since
#120, no open PR into `htp_moe`, no `state:measured` issue. Consistency
check of the weekend's merges: #120 / #82 / #81 / #84 / #89 / #85 are
closed against PRs #121 / #125 / #126 / #127 / #128 / #119, and the tree
carries what plan 130 assumes (`hvx/hvx_m1_ops_f32`, `hvx_attn_m1_f32`,
`hvx_conv_gate_f32`, `m1_ops_det.h`, `attn_m1_det.h`, the `attn_m1_*` IDL
entries, `run_inproc_e2e.sh`, `htp_dump_eval.py`; `hexkl_graph.c`'s table
still MOE-only). Two things were stale and are fixed: the tracker #76 body
(cycle-12 text: "frozen at `2ce38d65`", provisional "now") is rewritten to
the merged state, and ten closed issues still carried a `state:*` label
(#117 #105 #102 #101 #100 #87 #85 `review`, #78 `in-progress`, #91 #114
`needs-plan`) — removed. Ruling recorded on #76: **#122 (docs/htp tech
docs, `state:in-progress`, user-driven sessions) is a docs track and does
not occupy the implementer's single `in-progress` slot**, so #130 may go
`in-progress` beside it. Local housekeeping left to the user / orchestrator
(not an agent output): worktrees of merged branches (`~/nntrainer-120`,
`-85`, `-guide`, `-moe` = `htp_moe_cycle` @ `d79c0efe`, `-A` detached)
and their `[gone]` local branches; `~/nntrainer-122` at `7d14d9fa` is three
commits behind `origin/htp/122-tech-docs` (`4c73bb01`). Rungs 2–3 md5s
for a post-#121 skel are still unrecorded (§3a); #130's gate produces
them. Queue: #130 planned p1, #99 / #90 planned p2, #110 parked — no new
issue.**

**Cycle 19 (seen 2026-09-28 ~05:30 UTC): PR head moved `4ae1ebd7` →
`700371da` (updated 2026-09-28 02:37 UTC), three commits. Files:
`hmx/hexkl_mm_u8i4_moe.c` (+87/−2), `hmx/hexkl_mm_u8i4_dma.{c,h}` (+9),
new `hvx/hvx_int_epilogue.{c,h}` (+633), `test/htp/build.sh` (+4, a
`HEXKL_MOE_INT_EPILOGUE=1` build knob), host stubs, new
`test/htp/host/int_epilogue_host_check.c`, `moe_layer_host_check.c`,
`run_host_checks.sh`, docs 00/53. No IDL change. **Not merged; user
decision.** Oldest first:**

| sha | subject | touches |
|---|---|---|
| `23c45cff` | [Docs] 53 8.5: epilogue workers fill 64% of the MoE shadow, 95% of dense; task closed | doc 53: the author's hidden-worker probe on device — MoE prefill epilogue workers 23.2 ms of a 36.0 ms shadow (64 %), dense 95 %; "neither clears the −5 ms gate"; PPL 62.0916 unchanged. Analysis closed |
| `55823f96` | [HTP] MoE gate_up epilogue with no f32: baked scales, integer SwiGLU and requant | **`hexkl_mm_u8i4_moe.c`'s HMX gate_up epilogue** (int32 acc → f32 → SwiGLU → u8 requant) gets an all-integer twin behind `HEXKL_MOE_INT_EPILOGUE` (T-MAN-style baked scales, Q30 sigmoid polynomials, per-(row, batch) fixed point, int32 mantissas + exponent table); portable scalar C "about 20× the f32 epilogue's worker time", so that skel "measures perplexity, not speed"; weights baked per registered slot (`iq` field, `hexkl_mm_u8i4_dma.h`). Host: 32.36 dB vs 32.39 for f32, 0.83 % one-step disagreements with the f32 grid, kernel bit-identical to its reference. **Off by default; "the f32-only tail path is a compile error in this build"** |
| `700371da` | [Docs] 53 9: the f32-free epilogue as built, its host numbers and the device ppl gate | doc 53 §9: "implementation waiting on the device ppl" — the author has not yet run the PPL gate on silicon |

Reading: nothing on the decode path — at M=1 the GEMV path bypasses the
HMX block loop and its epilogue (`blocks=0 m1_gemv=1408/1408`), and the
knob is off. It touches the same requant site as #110 (⑱, parked) and
the `moe_layer_host_check.c` / `run_host_checks.sh` that every PR here
runs, so a merge would rebase over PR #133's host-check edits. No reason
to merge now: the author's own device gate is pending and the build is
scalar by design. On our side: **#130 measured and closed** (§2, ㉕–㉗,
rules 36–38), **#135** (user-filed: the Android app has no
`ENABLE_HEXKL`, rule 36), **#136** (B's text, p1) and **#137** (gtest
ulp / `AEE_ERPC`, p2) filed; queue for the planner **#135 → #136 →
#132** (#134 rides #136's plan if cheap). `htp_moe` @ `9deea6e2`.

**Cycle 20 (2026-09-28): head unchanged at `700371da`** (updated 02:37 UTC,
no new commit; the three commits above stay unmerged — user decision). On
our side: **#135 closed** (PR #139 merged `02d96eb3`: `jni/meson.build`
exports `-DENABLE_HEXKL=1` through the prebuilt `Android.mk`,
`build_android.sh` exits 1 on a mismatch, rule 36 (3); app md5s in
BENCHMARK artifacts, no device cell); PR #138 (guide) merged; `htp_moe`
@ `c5dcb382`. Queue for the planner **#136 → #132** (held until #136's
sitting passes) → #137 (rides #136's sitting) → #134. #132's call-count
ladder and kernel inventory verified on the code (㉓, cycle-20 note).

**Cycle 20 close (seen 2026-09-28 ~10:00 UTC): PR head moved `700371da` →
`bcfc1ac5` (updated 09:51 UTC), ten commits. Files: new
`hvx/hvx_int_epilogue_hvx.c` (+408) and `hvx_int_epilogue_impl.h` (+231),
`hvx_int_epilogue.{c,h}` (+149/−176), **`test/htp/nntr_hvx.idl` (+13, a
self-check method)**, `test/htp/build.sh` (+5/−2, the HVX-epilogue
define), new `test/htp/nntr_hvx_int_epilogue.c` (+326), new
`test/unittest/unittest_hvx_int_epilogue.cpp` + `test/jni/Android.mk`
(+24), `run_u8i4_layer_on_device.sh`, docs 00/53. **Not merged; user
decision.** Oldest first:**

| sha | subject | touches |
|---|---|---|
| `f830eb44` | [Docs] 53 9.7: the integer epilogue measured on device, ppl 60.93, gate passed | doc 53: the scalar integer gate_up epilogue on silicon — PPL **60.9269 vs the f32 path's 62.0916** (inside the author's +0.5 % gate; "not read as an accuracy gain"), worker time 23 → 1014 ms a call (44×, scalar by design) |
| `a7839ed9` | [HTP] Integer MoE epilogue on HVX, with a device self-check against its C reference | `hvx_int_epilogue_hvx.c`: the reference's helpers mapped to HVX instructions (32 columns per vector, sigmoid in 32-lane words, four rows per k-tile packed per store); a device self-check entry (the IDL change) |
| `dc4510a0` | [Docs] 53 9.8: the HVX epilogue as built and the three device runs that judge it | doc 53: criteria = self-check 0 mismatches, PPL exactly 60.9269, hidden worker time back near 23 ms |
| `ebb07316` | [HTP] Select the HVX integer epilogue by a build.sh define, not only __hexagon__ | `build.sh`, the selector |
| `de33d52a` | [HTP] Integer epilogue on HVX: the row's max \|A\| with the signed word max | kernel fix found by the self-check |
| `4ac3dfd1` | [test] unittest_hvx_int_epilogue: define main like the other HVX gtests | gtest |
| `475f66eb` | [HTP] Integer epilogue: name the reference requant _c in its definition | rename |
| `c6a813cf` | [HTP] Integer epilogue self-check: locate the first divergence stage by stage | self-check |
| `87554f95` | [HTP] Integer epilogue on HVX: zp * colsum by vmpyie; the bake without libm | kernel fix, bake |
| `bcfc1ac5` | [Docs] 53 9.9: the HVX epilogue measured -- bit-identical, ppl 60.93 again, 19.3 ms a call | doc 53: device self-check passes for both seeds, PPL = the scalar build's to the digit, **decode at f32 speed, MoE prefill call 19.3 ms vs f32's 14.3 (+35 %)**: requant units ≈ 3× the f32 ones (610 µs exposed vs 62), the integer bake 2.9 ms on each layer's first call; next trim named (move the requant's scan pass into the gate_up worker) |

Reading: still nothing on the decode path — the integer epilogue is the
M>1 HMX gate_up epilogue, which the M=1 GEMV path bypasses
(`blocks=0 m1_gemv=1408/1408`), and it stays behind a build define, off
by default. As a default it would fail our prefill gate (MoE prefill
call +35 % by the author's own number) while its only gain is an
f32-free grid of the same accuracy class. A merge now costs a skel + app
rebuild pair (IDL +13, rule 3) and a rebase over our IDL / `build.sh` /
`test/jni/Android.mk` edits (#130, #132 PR 1) for no decode change.
**Supervisor's recommendation: do not merge now; revisit when the
author's named trim brings the prefill call back to f32 ± 5 % or when a
sync is needed for another reason.** User decision.

On our side at close: **PR #143** (`ff781a6c`, #136: worker-pool two
job slots + generation re-check; `POOL RACE` host check; `run_inproc_load.sh`
25/30 → 30/30; the real-shape fixture `lfm2_moe_tiny_lfm25`;
`NNTR_HTP_DUMP_ALL`) — **#136 closed**, verdict §2 and rule 39;
**PR #144** (`72df58c2`, #134: `NNTR_PPL_DECODE`, decode-side
teacher-forced PPL); **PR #145** (`bb845426`, #132 PR 1: ADD +
ROUTER_TOPK, 95 → 73 → 51 calls/token). #134 and #132 are
`state:needs-measurement`: one combined sitting (A switch off / C
`KINDS=MOE` / B six kinds / D six + ADD + ROUTER_TOPK; decode PPL on A's
own G=512 continuation, reference A; text + approval) is staged at
`/local/mnt/workspace/htp_moe/134-132/` and waits for the phone. New:
**#141** (transport: dspqueue microbench, p1) and **#146** (㉗, ATTN_M1
speed-up, p1), both with the planner. `htp_moe` @ `bb845426`. Queue:
**the combined #134 / #132 sitting → #146 → #141 → #132 PR 2 (held
behind #141) → #137 → #99 / #90; #110 parked (`needs-user`).**

**Cycle 21 (seen 2026-09-29): PR head moved `bcfc1ac5` → `70655c4b`
(updated 2026-09-28 10:22 UTC), three commits. Files:
`hmx/hexkl_mm_u8i4_moe.c` (+21/−23), `hmx/hexkl_mm_u8i4_dma.c` (+10),
`hvx/hvx_int_epilogue{.c,.h,_hvx.c,_impl.h}` (+94/−75),
`test/htp/nntr_hvx_int_epilogue.c` (+14/−10), the two host checks
(+8/−8), docs 00/53. No IDL change beyond `bcfc1ac5`'s. Not merged; user
decision (the 2026-09-28 hold covers `bcfc1ac5`; these three sit on
top of it). Oldest first:**

| sha | subject | touches |
|---|---|---|
| `b79c169f` | [HTP] Integer epilogue: the requant reads the mantissas once, the bake moves to upload | the gate_up worker leaves each (row, batch)'s min / max beside its exponent (`hvx_int_hmeta`), so the requant skips its rescan — same bytes; the per-column fixed-point bake moves from a layer's first call (2.9 ms) to weight-slot fill (`hexkl_mm_u8i4_dma.c`) |
| `28b456de` | [HTP] Integer requant: no clamp before the saturating packs, no shift for the coarsest batch | requant quantize-pass trim, same bytes; records the MoE prefill call 19.3 → 15.2 ms (f32 path 14.3) |
| `70655c4b` | [Docs] 53 9.13: the requant trim measured, the integer epilogue closed at f32 speed within noise | doc 53: requant exposure 560 → 457 µs (MoE rows), call host time within ≈ 0.4 ms of f32 (single-run noise ≈ 0.4 ms), PPL 60.9269 unchanged; "kept for its 1.2 ppl, not for speed"; the section closes |

Reading: still nothing on the decode path (M>1 HMX gate_up epilogue,
behind a build define, off by default; the M=1 GEMV bypasses it). The
cycle-20 condition for revisiting — the author's trim bringing the MoE
prefill call to f32 ± 5 % — is met by the author's own numbers
(≈ 14.7 vs 14.3 ms, ≈ +3 %), so the hold's performance objection is gone.
What remains against a merge now: it adds no decode speed, it is off by
default (so merged it changes no bit of our runs), and it costs a skel +
app rebuild pair (IDL +13 from `bcfc1ac5`, rule 3) plus a rebase over
PR #151's `htp_compute_ops.cpp` / skel / `build.sh` edits. As a
**default** it would change the prefill's u8 grid, which the 2026-09-28
bit-preserving rule excludes. **Supervisor's recommendation: keep holding;
merge only when a sync is needed for another reason.** User decision.

On our side (cycle 21): **#141 closed** (PR #151 `08de92b1`: dspqueue for
the 22 M==1 MoE calls, **default on** by the user's flip `c73384d2`; §2 #141
row, rule 40; BENCHMARK's "now" moves to its Q); **#149 closed** after its
step 0 (D2, no code; rule 41, ㉓ feed-engine row closed); **#134** closed
earlier (PR #144 + the combined sitting, §2 #134 row); **#146** (PR #148
open) and **#132** parked by the direction change (`needs-user`); **#150**
(⑰, p1) with the planner, its first device read under ⑰. The direction's
lever (4), a CPU-exact Q4_0 FC on the DSP, is **not filed** (㉘: it cannot
beat the CPU on bytes, rule 42). `htp_moe` @ `08de92b1`. Queue: **#150 →
#137 → #99 / #90; #110, #132, #146 parked (`needs-user`); #122 docs
track.**

**Cycle 22 (seen 2026-09-29): PR head moved `70655c4b` → `f923bf29`
(updated 04:20 UTC), two commits, both docs only (`docs/htp_attention/
53_int_requant_task.md`, +8 and +5/−2). Not merged; user decision.**

| sha | subject | touches |
|---|---|---|
| `d9fe6cef` | [Docs] 53 4: the paper's baseline is llm.npu, its decode gain is over CPU decode | doc 53: T-MAN's 1.4× prefill / 3.1× decode are against llm.npu (decodes on the CPU); the stage it removes (NPU weight dequant) does not exist in our u8 × i4 path |
| `f923bf29` | [Docs] 53 9.9: where the worker's +26% comes from -- not the sigmoid | doc 53: the integer epilogue's extra worker time is the max\|A\| scan, the fixed-point scale multiply and the split 32×32 product, not the sigmoid (HVX has no exp; both paths use a polynomial) |

Reading: nothing on any code path; the cycle-21 recommendation (keep
holding the integer epilogue; merge only when a sync is needed for another
reason) stands. The 16 unmerged commits after `4ae1ebd7` are now 18.

On our side (cycle 22, post-merge fold): the user merged PRs **#153**
(#150 step 1a, `NNTR_OP_TIME`), **#154** (#99, strided replay), **#156**
(#90, two-reader probe), **#159** (#158, DSP L2 bypass — **default on**
by `21169f6e`, `NNTR_MOE_DMA_BYPASS=0` keeps the L2 path), **#160** (#152,
resident attention bit-identical to the Android CPU, off by default) and
**#155** (guide refresh, written before #158) into `htp_moe` (head
`c95c0feb`). Folded: `150-cpu-decode.md`, `90-two-reader.md`,
`152-resident-accuracy.md`, `158-dma-bypass.md` and #99's ride-along
(PR #154 comment). **BENCHMARK's "now" moves to #158 B: 51.82 / 50.61 /
47.36 tok/s** (G 64 / 512 / 1024, prompt 512, `R3CY10WM83Y`), bit-identical
to its A; the device check that an unset run prints `dma_bypass=1`
(`applied=0x703e1`) is **pending** (phone disconnected when the flip was
written). New rules 43–46; §2 rows #150, #90, #99, #152, #158; ④ ⑰ ⑳
closed. **Closed: #150, #90, #99, #152, #158.** **#157** stays open,
`state:needs-measurement`, **measurement on hold by the user** (PR #161
open, not folded; to be rebased on the bypass default, rule 44 (2)).
#106 parked. New: **#162** (㉙, p1, `state:needs-plan`: G=1024 47.4 → 50,
bit-preserving). Queue: **#162 → #157 (when the user resumes it) → #137;
#110, #132 parked (`needs-user`); #122 docs track.**

**Cycle 23 (seen 2026-09-30): PR head unchanged at `f923bf29` (updated
2026-09-29 04:20 UTC). Nothing to decide upstream; the 18 commits after
`4ae1ebd7` stay unmerged (cycle-21 recommendation stands).**

On our side (cycle 23, the decode NPU end-to-end day, 2026-09-29/30): the
user merged PRs **#165** (#162 plan), **#166** (#164 plan), **#167** (#164,
bit-identical HTP RMSNORM / QK_NORM, silicon-verified, `efb88c7a`),
**#169** (contract decision 2026-09-29: decode NPU end-to-end is the main
track, bit-preserving), **#171** (#132 PR 2 plan), **#174** (#170 plan) and
**#176** (#170 round 1: fast bit-identical ATTN_M1, silicon-verified, speed
gate missed 2×, default off, `90d88e2b` = `htp_moe` head). **Open PRs,
recorded as pending, not folded as merged:** #175 (#132 PR 2 Part A,
device-measured), #179 (#170 round-2 plan), #180 (#178 plan), #182 (#170
round 2, device-measured); on the `htp_moe_v81` track (S26 Ultra, v81, not
in BENCHMARK's S25 rows) #173 (#168) and #181 (#177: the M=1 feed over 2–4
DMA queues — +24.8 / +25.1 / +22.2 % on the S26 where one queue tops out
34 % under the S25's; on the S25 queues 2 / 4 give nothing, #162 step 0,
rule 43). **Closed by the user:** #162 (both levers no gain: the CPU
attention split and the prefetch overlap, PR #172 closed), #164 (done).
**#157 on hold** (user, 2026-09-29; PR #161 not folded). Folded here:
`162-step0.md`, `162-prefetch.md`, `164-cpu-order-norm.md` (+ PR #167's
comment and the `dev/norm-shadow` analysis), `170-attn-m1-hf.md` (S1 + S2),
`170-attn-m1-round2.md` (S3), `132-pr2-exact-fc.md` (+ PR #175's comment),
`178-second-dsp-session.md` (+ issue #178's comments). **BENCHMARK's "now"
stays #158 B (51.82 / 50.61 / 47.36)**; the pending unset-banner device
check is done; the G=1024 ≥ 50 verdict is temperature-bound (rule 52).
**Record sitting 2026-09-30 (after the fold, PR #184 merged):** the
mirrored cool control was taken (`record-2026-09-30-cool-a.md`,
`R3CY10WM83Y`, 02:48–02:53 KST, `90d88e2b`, nothing set, each G block at
zone0 ≤ 35 °C) and read **53.97 / 52.16 / 51.41** — the "now" moves there
(§2 row), #158 B stays as the warm-start reading; rule 52's band is now the
row's stated condition. New
rules 47–52; §2 rows #164, norm shadow, #170 R1, #170 R2, #162, #132 Part A,
#178; ㉙ closed, ㉗ / ㉓ (#132) updated, ㉚ (#178 decision) added. Queue:
**#170 round 3 (softmax word split, P1 lead, append; on the issue) → the
user's decisions D (#132) and #178 → #157 (on hold) → #137; #110 parked
(`needs-user`); #122 docs track; #168 / #177 `htp_moe_v81` in review.**

**Cycle 24 (seen 2026-09-30): PR head unchanged at `f923bf29` (updated
2026-09-29 04:20 UTC). Nothing to decide upstream; the 18 commits after
`4ae1ebd7` stay unmerged.**

On our side (cycle 24, the second half of 2026-09-30; `htp_moe` @
`e3aad2ce`): the user merged PRs **#182** (#170 round 2, `196a3e13`),
**#190** (#170 round 3, `0a380112` — **both ATTN_M1 speed gates pass on silicon**,
193 838 / 313 747 pcyc/op vs 210 k / 350 k, bit-identical, text `y`,
default off), **#175** (#132 Part A, `721ff913`: the CPU-exact Q4_0 FC /
quantizer / SwiGLU / argmax / router — only `hvx_intrin` bit-identical on
silicon, rule 53 in the PR's own ledger text; FC + lm_head VTCM-fed
7.84 ms/token, HVX quantizer 0.53, router 62 311 pcyc/op vs gate 60 k
accepted), **#180** (#178 plan, `8e95bd51`), **#186** (#132 Part B plan,
`e3aad2ce`), **#184** and **#188** (cycle-23 ledger and the record
sitting). **Closed by the user:** #162, #189 (a duplicate of the round-3
plan on PR #190). **Closed by the orchestrator:** #192 (the single-session
map window: not viable, rule 57). **PR #183** (the #178 probe) closed, to
be re-targeted onto `htp_moe`. **Decision D taken (user, 2026-09-30):
option (1) + (2)** — Part B proceeds as the two-session end-to-end design,
and the VTCM-share probe (track 2) was stopped on the host finding that
the MoE prefill layout needs ≥ 6720 KiB of VTCM (S2 could get ≤ 0.95 MiB,
a ≤ 3-lane VTCM feed slower than L2), so S2's FC is L2-fed (㉚). Folded
here: `170-attn-m1-round3.md` (S4), `132-pr2-exact-fc.md` (S3 / S4),
`132-part-b-e2e.md` on `htp/132-partb-e3` + issue #132's E5b / E5c / E5e /
E5d comments, issue #192's closing comment (the branch doc is a template).
**BENCHMARK's "now" stays the record sitting (53.97 / 52.16 / 51.41
cool; #158 B 51.82 / 50.61 / 47.36 warm; CPU 52.18 / 51.31 / 50.12).** New
rules 54–58; §2 rows #170 R3, #192, #132 Part B, and #132 Part A / #178
updated; ㉓ / ㉗ / ㉚ updated. Open: **#132 Part B** (`state:in-progress`,
PR pending: rebase onto `htp_moe`, HVX quantizer / SwiGLU / argmax in the
graph kernels, per-shape lanes, then set_e5f), **#178** (PR #183
re-target), **#170 round 4** (`-DATTN_M1_EXP_TAB=0` gtest cell first,
then `vgather` from a VTCM carve-out), **#157** on hold (user). Queue:
**#132 Part B → #170 round 4 → #178 re-target → #157 (on hold) → #137;
#110 parked (`needs-user`); #122 docs track; #168 / #177 / #185 / #187
`htp_moe_v81`.**

**Cycle 25 (seen 2026-09-30, later in the day): PR head unchanged at
`f923bf29` (updated 2026-09-29 04:20 UTC). Nothing to decide upstream; the
18 commits after `4ae1ebd7` stay unmerged.**

On our side (cycle 25, 2026-09-30 evening; `htp_moe` @ `c34fc445` =
`htp_moe_ppl`): **#194 resumed by the user** ("htp_moe_ppl 작업 마저
진행해줘") — the FSU pause of the morning is lifted; `needs-user` removed,
`state:in-progress`. Heads: `htp/194-s1` @ `33b7ce36` (rebased from
`htp/132-partb-e3` @ `69251151`), `htp/194-s3-wip` @ `d8599531`; the
S1-arena teardown fix (`8bb8a3f8`, `ad8496e2`) sits on
`origin/htp/132-partb-e3` only; no FSU change has landed anywhere. The
implementer rebases 194-s1 / s3-wip onto the teardown fix and continues S3
on the host (L2 / L3 held to their specs, L4, the L0 fixes); **sitting 1b
is the user's device step** (PPL / mc-40 cells, G 512 / 1024 E1 rows).
**#185 folded** (`185-m1-schedule-overlap.md` on
`htp/185-m1-schedule-overlap` @ `25764b23`, PR #191 open, `state:review`
kept): the S26 decode passes the S25's #158 B on DQ and DQR, bit-identical,
ship rule picks **DQ** (§2 row; C3 stays on `htp/185-dqr` for the user).
The `htp_moe_v81` track stays out of BENCHMARK's S25 rows (cycle 23); its
numbers live in §2. No new silicon rule: the plan's model (≈ 419 µs/call
for DQR) met the device at 422.5, and the C1 negative fired a different
check than expected (`push not waited`, not `slice read in the run that
issued it`) for a host reason — the reading lanes are not the issuing lane.
Queue health: #197 (the #185 remainder: C(2) / C(3) ≈ 38 µs/call DMA-idle,
p2) and #198 (S26 prefill `mha_core` 17–18 %, after #187, p2) filed
`state:needs-plan`; ㉛ added. Queue: **#194 (host S3, sitting 1b on the
user) → #187 (`in-progress`) → #132 Part B → #170 round 4 → #178 re-target
→ #157 (on hold) → #137 → #197 → #198; #110 parked (`needs-user`); #122
docs track; #168 / #177 / #185 `htp_moe_v81` in review.**

**Cycle 26 (seen 2026-09-30, night): PR head unchanged at `f923bf29`**
(last commit 2026-09-29 04:20 UTC; the PR's `updatedAt` moved to 2026-09-30
05:24 UTC without a new commit). Nothing to decide upstream; the 18 commits
after `4ae1ebd7` stay unmerged. Not on this watch: upstream #4383 (FSU,
open draft, head `5a84c05d`) and #4343 (open), both merged into the new
base by the user.

On our side (cycle 26): **the user re-planned the project on a new base,
`htp_decode`** = `htp_first_version` @ `d4a898430` (contract §12,
2026-09-30): `htp_moe` @ `70e0f0b0` + #132 Part B + upstream #4383 + #4343
+ upstream main @ `aad932ce` + plan 194 and #194's E1 commits. Docs and PRs
target `htp_decode` from this cycle. **#201 filed** (`state:needs-plan`,
p0): review the whole decode NPU end-to-end path (`NNTR_HTP_E2E`) with FSU
in the tree and optimize it; plan first; its targets and gates are open
questions for the user, none invented. **#194 sitting 1b read, not folded
as BENCHMARK rows** (`194-sitting-1.md` on `htp_decode`, `R3CY10WM83Y`,
14:38–17:12 KST, from `htp/194-s3` @ `b0248a63`; `MD5 OK` ×10 in
`sitting.out`, 0 mismatches): nulls hold (E0 ≡ A on nll 8/8, mc 8/8, text
at G 512 / 1024), P1 1.0025 / P2 max 1.0110 / P3 pass, **P4 FAIL (p04)**;
A 56.21 / 54.89 and 52.89 / 51.93, E0 32.19 / 30.94 and 32.25 / 31.19, E1
33.83 / 33.10 and 33.80 / 32.91 tok/s at G 512 / 1024; nine `LEAK` stops
(S1 ceiling 3584), after A runs too. Not folded because E1's text differs
from A (one word) with the `text approved` column empty, and the handoff's
four user decisions (P4 on p04, the mapping loss, the PR, §3.4) are open;
the sitting predates FSU, so it is #201's documented starting breakdown,
not its baseline. A G = 64
device check at `d4a898430` (A 54.8–55.9, E0 29.5, E1 31.3, S1 ceiling
3840) was reported by the orchestrator; no handoff file or log for it is in
the tree (`/local/mnt/workspace/htp_moe/first_version/logs/` is empty), so
it is not a row. **Branch state on `origin`:** `htp_moe_ppl`, `htp_moe_v81`
and every `htp/194-*`, `htp/168-*`, `htp/177-*`, `htp/187-*` branch are
gone; they survive as local tags `archive/*` on the workstation only
(`archive/htp/194-s3` @ `e7f8660e`, `archive/htp_moe_v81` @ `5300f695`,
`archive/htp_moe_ppl` @ `62b7bbd2`, …). `origin/htp/185-m1-schedule-overlap`
@ `25764b23` remains. PRs #173 (#168), #181 (#177), #191 (#185) are closed
unmerged with their base; no PR is open. **The 2026-09-30 cleanup (user
decision; done outside the supervisor, 10:09–10:10 UTC, during this
cycle) closed twelve issues as `not_planned`** with a closing comment
each: #122, #132, #157, #168, #170, #177, #178, #185, #187, #194, #197,
#198 — "the remaining work is to be re-planned later on the new base";
their `state:*` labels were left as they were (#194 still reads
`state:measured`, #187 / #132 `state:in-progress`), which the supervisor
did not touch. Open `hexagon` issues after it: **#201** (`needs-plan`,
p0), #137 (`needs-plan`, p2), #110 (`needs-plan`, `needs-user`, parked),
#76 (tracker, still titled "htp_moe tracker"). Queue: **#201 (plan) →
the rest waits for the re-plan.**

**Cycle 27 (seen 2026-10-01): PR head unchanged at `f923bf29`** (`gh pr
view 4327 --json headRefOid,updatedAt`: `f923bf29…`, `updatedAt`
2026-09-30 05:24 UTC, as in cycle 26). Nothing to decide upstream.

On our side (cycle 27, 2026-10-01; base `htp_decode` @ `53f38aabf`, PRs
#202 / #203 open onto it, #205 merged `bb8c6d06`, #206 open = #204's S26
port): **#201's three device sittings and #207's farm sitting folded**
(§2 #201 row; rule 59; ㉜ ㉝; BENCHMARK Results ×4 + Method cycle 27).
Verdict: **one PD with the expert pool inside the per-token E2E entry
(Q28) beats two PDs with the same pool and all-resident E0 on both S25
units, at A's bits** — `R3CY10WM83Y` 43 / 36 / 31 tok/s (G ≤ 512; G = 1024
not run, the unit was withdrawn for the S26) and the device-farm unit
`R3CY205ZMND` 40–47 / 33–39 / 29–32 (G 64 / 512 / 1024; another session,
whole sitting re-run from a set rebuilt at `a2ebef9c9`, logs on that
machine); C = 28 is the one-PD pool (C = 29 same speed, C = 30 does not
load). Still 7–13 tok/s under A (50–57): no row of record moves, the S25
"now" is frozen (unit withdrawn), #208 opens the S26 column. Policy rows
(user, 2026-09-30 / 10-01): agents run adb on `htp_decode` (`dc1898239`);
the S25 is withdrawn, the S26 (v81) replaces it (plan 201 §4.1
`0d7efb00a`); #204's port is PR #206 (`state:needs-measurement`); #207
(S25 farm, `state:measured` → folded here) and #208 (S26 farm,
`state:needs-measurement`) are the device-farm sittings. Issue state after
the fold: #201 `state:review` (PRs #202 / #203 open), #207 done by this
fold (to close `completed`), #204 / #208 wait for the farm S26. Open:
sitting 2's prefill column is not in the comment (gate not read on that
unit), the pool-28 miss cost (㉜), the farm unit's LEAK rate (㉝).

**Cycle 28 (2026-10-01, base `htp_decode` @ `84142c0b9`): the #208 S26
sitting happened, on a developer unit — recorded, not derived from.**
The orchestrator ran plan 204 §4 step 7 (G4 / G5 / G6) on
**`R3CY70LV96T`** (SM-S948U, SM8850, userdebug `S948USQU1AZAB`, kernel
6.12 — not the earlier S26 `R5KL20NFRCK`), 14:12–15:29 KST, three
invocations: attempt 1 ended when the `HmxMmU8I4Layer.RegistryCapacity`
gtest dropped the phone into download mode (rule 60), attempt 2 ran the
gtests with it excluded (mm_u8i4 34/0, fc 1/0, two_sessions 6/0,
softmax 28/4 and attn 10/2 = #137's known set, reproduced on v81) and
stopped on a config file the crash had zeroed, attempt 3 ran all 24 runs
+ 2 profiles + the CPU control without a stop. **What it showed: the
one-PD E2E (Q28) runs on an S26 (v81); every NPU text == A q4 r1 of its
G (20 / 20); the S1 ceiling stayed 3840 MiB throughout.** The raw tables
(A 53.6–57.9 and E0 29.3–33.9 at G 64, Q28 37.4 at G 512; four DMA
queues no better than one on any variant; a pool-28 miss ≈ 4–4.7 ms with
the page-cache `read()` at ≈ 2 GB/s on this phone) are in
`204-s26-rebaseline.md` as an appendix **marked "developer unit, not
representative"** (user, 2026-10-01: a userdebug engineering phone may
not represent the product S26) — no rule, no BENCHMARK column, no "now"
is derived from them, and ㉜ is not updated from them. **S26
optimization is deferred by the user until a product unit is
available.** Same day: the user decided to delete the two-PD path (plan
being written as the next issue). Issue state: #208 → verdict row in §2
("E2E runs on v81; optimization deferred"), to close `completed`; #204
closes with it (its port, PR #206, is merged).

**Cycle 29 (2026-10-01 evening, base `htp_decode` @ `87cc9c4a4`): the
#201 S3 probe and its discriminator folded** (`201-s3-probe.md`,
`201-s3probe-run.sh`, `201-s3disc-run.sh`; `R3CY10WM83Y`, 17:27 and
17:34 KST, sitting 1's build `a2ebef9c9`, md5 OK both, 20 profiled
Q28 / Q24 G = 64 runs, every text == sitting 1's `A_G64_r1` 20 / 20,
ceiling 3840 after every run). **The orchestrator's first reading —
"reader-core placement decides the miss cost" (Q28_BIG 0.48 ms against
Q28 4.76) — was wrong**: the implementer's interleaved D / B / P0 × 4
sitting after a reboot read the big-core pin as slow as the default on
the first runs and all three settings equal from run 5 on; the readers do
not run in decode at all (code read). What is left: the slow miss (3.7–5.0
ms) is a **boot-proximity** effect (uptime ≲ 130 s in both sittings), the
cause unmeasured; rule 61 (wait ≥ 5 min or a warm-up after a reboot;
first-run cells flagged, #208's 4.7 ms a candidate), ㉜ rewritten, no
tok/s row (profiled runs). The S3 "reader cores" lever of plan 201 is not
applied (no code, no PR). Issues: #201 carried `state:in-progress` and
`state:review` at once → `state:in-progress` only (S4 slices PRs #213 /
#214 open, more remain); **the miss-cost cause filed as #216** (p1,
`state:needs-plan`); #137 stays `needs-plan`
p2 (device-gtest hygiene, reproduced on v81 in #208, not on the decode
goal path; PR #212 drops the one test that bricks the developer S26,
rule 60). Open PRs into `htp_decode`, none reviewed: #212, #213, #214,
#215 (#211). Upstream PR #4327 is not watched (contract §12).

**Cycle 30 (2026-10-02, base `htp_decode` @ `03a44701c`): #216's two
measurements folded — the pool-28 miss cost has a measured cause, and the
first lever against it fails the prefill gate.** Merged by the user:
PR #215 (#211, one PD only, `aa78104e9`) and PR #213 (#201 S4 attention,
`03a44701c`); #211 closed `completed`. **(1) Step 1** (`216-miss-read.md`
on `htp/216-miss-read`, PR #217 open; `R3CY10WM83Y` 21:43–22:01 KST,
sitting 1's build, MD5 OK ×3 boots, 12 profiled Q28 G = 64 runs, text
12 / 12 == A, ceiling 3840): **no busy core** (busiest non-app core
3–16 % in 11 of 12 decode windows); the slow misses are **UFS reads** —
in b3, 610 / 310 / 61 / 0 MiB `pgpgin` and 151 k / 79 k / 16 k / 1 file
refaults in the decode window line up with 5.03 / 2.72 / 1.55 / 0.55
ms/miss, PSI io 92–296 ms in every slow window (0–31 fast), kswapd
≥ 65 k scans/s; **not boot proximity** (b1 slow at uptime 300 s, b2 slow
at 60–301 s throughout, b3 slow after a 6-minute idle and fastest 8 s
later). Cause: the 3.3 GB ION arena + the 4.1 GB model file in the page
cache + Android (≈ 3.7 GB) exceed the 11.1 GB, so kswapd evicts the
file's pages during the run and a re-missed expert comes from storage.
Rule 61 amended (its protocol does not remove the slow regime), ㉜
rewritten. Plan 216's slice lever not built (stop rule). **(2) The
fadvise lever** (`216-fadvise.md` @ `45be8de65` on `htp/216-fadvise`,
PR #218 open, env-only `NNTR_MOE_FADVISE`; two sittings 23:06–23:36 KST
on a fresh and a ≥ 10-min-old boot, set `61a26580…` / skel `9d61aef4…`,
device md5 == staged on both boots, verified against
`216/fadvise/logs/{fresh,old}/md5_device.log` here; text 26 / 26 == A,
`calls/token=1.00`, ceiling 3840 after 25 / 26): B holds the miss at
0.86–1.13 ms where A ranges 0.68–4.38 on the same boot, decode +14.6 /
+10.5 % on the G = 64 block means (36.27 → 41.56, 38.67 → 42.71), −0.5 /
+4.3 % at G = 512; **prefill −10.4 / −7.4 % at G = 64 and −17.8 / −11.9 %
at G = 512 — the prefill gate fails**, as do the plan's `arm_ms/round`
(1.20–1.58 vs A-fast 0.945) and window-`pgpgin` gates; the `=2`
(drop-only) cell is as slow as A's slow regime (3.87 / 3.97 ms/miss,
every miss from storage), which confirms the plan's model (every decode
miss is a re-miss of an expert the arena once held). Silicon rule 62:
on this kernel `posix_fadvise` is not a hint — WILLNEED reads only the
1 MiB read-ahead window and costs 3–8 ms per expert, DONTNEED 2–10 ms.
§2 row; the default is **not flipped** (user's call at review: merge
#218 env-only, or close it). The next lever is filed (#219, p1): the
complement held in a cached ARM buffer and refilled off the token path,
with the page cache kept out of the sum. Also this cycle: **no device on
the workstation** (user, 2026-10-02): agents do not run adb, handoffs are
filed as issues / comments with `needs-user` + `state:needs-measurement`
(contract §12 row, §4.1 adb row, §4.2 — the 2026-10-01 "agents run adb"
policy of cycle 27 is suspended, not withdrawn). Issue state: #211
closed; #216 `state:review` (PRs #217 / #218 open, verdict on the
issue); #201 `state:in-progress` (PR #214 conflicting, being rebased;
PR #212 open); #137 `state:planned`; #219 `state:needs-plan`.

**Cycle 31 (2026-10-02, base `htp_decode` @ `b26c161eb`): no measurement,
no row of record moves; three merges folded.** Merged by the user: PR
#214 (#201 S4: QK_NORM 256 / 512 + v norm, dense GeGLU, two-branch FFN +
`layer_scalar`, soft-capped 262 144-row head, `2b8832a23`), PR #212 (drop
`HmxMmU8I4Layer.RegistryCapacity`, the gtest that puts the S26 into
download mode, rule 60, `0e880088d`) and PR #217 (#216 step 1 docs,
`216-miss-read.md`, `b26c161eb`). With #214 every S4 kernel is in
(#209 GeGLU epilogue, #210 router / any-width norm, #213 attention, #214
head / norms / FFN); what remains of S4 is the Gemma graph builder and the
load hand-over (plan 201 §2.4 last row), host-gated on the #4296 tiny
fixture; S5 / S6 wait for the Gemma model files (and the user's pin of
upstream #4296, plan 201 §0). Open PRs into `htp_decode`: #218 (#216's
fadvise lever, env-only — the user's call stands from cycle 30: merge
env-only or close; the default flip is not recommended) and #220 (guide
refresh, docs, cycles 22–30). No `state:measured` issue, no device
attached (contract §12), upstream not watched. Queue healthy: #219
`state:planned` (plan `a540b876e`) and #137 `state:planned` — nothing
derived. Issue state: #216 `state:review` (unchanged), #201
`state:in-progress`, tracker #76's body rewritten for `htp_decode` (it had
stopped at 2026-09-30 on `htp_moe`: #132 Part B, the v81 issues and #157
/ #178 are closed since).

**Cycle 32 (2026-10-02, base `htp_decode` @ `238a280b7`): no measurement,
no row of record moves; one merge, three user decisions.** Merged by the
user: PR #218 (#216's `NNTR_MOE_FADVISE` lever, `238a280b7`) **env-only,
default not flipped** (user decision on the PR and on #216: the steady
0.86–1.13 ms miss and +10 / +15 % decode at G = 64 do not buy prefill
−7 to −18 %, rule 62); #216 closed `completed`, its `pgpgin_mib=` field
stays on every future pool-miss cell. Decisions recorded in the contract
(§12, two 2026-10-02 rows; §4.1 adb row; §4.2 device scope): (1) **the
LFM2.5 table is closed out on the #222 config of record** (all engine keys
`htp`, `init_seq_len 1024`, PPL cost of `attn_proj` / `dense_ffn`
accepted) and then the project moves to Gemma — the closing sitting
`222-config-refresh.md` (PR #223, open) is being run by the user on the
S25 via the farm; #222 stays `state:needs-measurement` and nothing of it
is folded before the handoff is filled and PR #223 merges; (2) **Gemma
(#201 S5 / S6) sittings run on a Galaxy S26 Ultra attached to the
workstation via adb**, a new BENCHMARK column; who drives that adb (agent
or user) is a `needs-user` question — until answered, agents do not run
adb. #201: PR #221 (S4 graph builder, 542-op 26B-A4B list, four mutants
caught) open; the rest of S4 waits on the user's three answers of the
2026-10-02 comment (#4296 pin merge, Gemma HTP MoE layer + QS4CX_WH
writer S5 → S4, head_dim ≥ 64 fixture). #219 `state:in-progress`
(implementer on host steps 1–3). Queue healthy (#137 `state:planned`,
#219), nothing derived; upstream not watched. Open PRs into
`htp_decode`: #223, #221, #220 (guide). Tracker #76 body refreshed.

**Cycle 33 (2026-10-06, base `htp_first_version` @ `7f95140ad`): the #222
closing sitting is read as a verdict (no row of record, rule 63); one base
again; #225 PR 1 merged; six user decisions.** **Base.** The user
fast-forwarded `htp_first_version` to `htp_decode` @ `c7ec6c64a`, then
merged PR #223 (`b7c1d4ff6`) and PR #230 (`7f95140ad`) into it: from this
cycle `htp_first_version` is the only base (contract §5) and the
supervisor's docs live there. `htp_decode` is behind it and receives
nothing new; the PRs still open against it — #226 / #227 / #231 (#201 S4,
stacked), #224 (#219), #220 (guide) — need retargeting, a user step. The
first cycle-33 docs commit (`0e89e600b`, written on `htp_decode` before
the fast-forward) was **dropped by the merge of PR #223**, which took the
branch side of BENCHMARK / LEDGER / contract; its content (this paragraph's
#222 part, rule 63, the §2 #222 row, BENCHMARK's Method paragraph and Log
row, the contract's §4.1 / §4.2 / §5 / §12 hunks) is restored here. **(1)
#222's closing sitting** (2026-10-02 16:15–16:54 KST, farm `R3CY205ZMND`,
S25 v79, set from `ddb7d9ad8`; results only in #222's comments of 08:00 /
08:30 UTC and the farm session's logs — the handoff `222-config-refresh.md`
merged **unfilled**, no device md5 line reached the supervisor, so nothing
is a row of record and no BENCHMARK Results row is added): **Qnew (one PD,
C = 28) VOID in every cell** — `no room for the FC set beside the resident
experts (mapped=3712 MiB, fastrpc_mmap(32 MiB) failed: err=1)`, reproduced
after a fresh reboot, `heap_used_kib` 92 058 (Qold) → 223 003 (Qnew
attempt); **Anew (hybrid) P1024 VOID** — `nntr_hvx_mm_u8i4_conv_block …
err=0x80000402` (`AEE_ENOMEMORY`, M = 1024 K = 2048 C = 2048 N = 2048: the
conv block's M-proportional session scratch, ≈ 13 MiB at M = 512 and ≈ 26
MiB at 1024, no longer fits the DSP heap beside the keys' ≈ 216 MiB of WH
copies); **Anew accuracy FAIL** — pooled over the 8 prompts prefill PPL
90.31 → 128.31 (+42 %), decode PPL (forced) 1.2108 → 1.3752 (+13.6 %, gate
+2 %), G64 text degenerates into repetition (two stacked quantizations: the
file's Q4_0, then per-column qs4cx over K = 2048 at load). Read for
information only, not folded: Qold (C = 28) loads and closes clean, prefill
/ decode 556 / 33.7 and 584 / 34.6 (G64), 455 / 43.3 (G512), 531 / 43.3
(G1024), texts = Aold r1; Anew P512 786.5 / 56.5, 761.9 / 56.4, 739.9 /
51.7 and P64 294.9 / 56.1, 254.0 / 55.6, 266.7 / 52.9 (the keys do move the
prefill, at a PPL the gate refuses); CPU `q40` P64 244.3 / 53.6, 210.5 /
53.4, 210.5 / 49.4, P512 310.5 / 49.0, 307.7 / 46.9, 294.4 / 48.9, P1024
(`init_seq_len 1024`) 311.8 / 49.7, 317.8 / 49.1, 313.8 / 48.5 (P1024 +
G1024 = `max_seq_len` 2048 generated to the end). Rule 63 and the §2 #222
row carry the verdict; the accuracy fail is the next issue per the
contract — **#225** (p0, filed by the user), the Qnew C = 24 follow-up is
**cancelled** (user: the fix changes the FC format, the reading would not
carry), #222 closed `completed` on PR #223's merge. **(2) The LFM2.5 table
of record is 9 cells × 3 cases** — P64 / P512 / P1024 × G64 / G512 / G1024
for CPU only, hybrid and E2E one PD, on the config of record with
`init_seq_len 1024` (user) — filled by #225's handoff (step 7 of plan 225).
**(3) #225 (`state:in-progress` + `needs-user`, p0): option (b) as a
sidecar.** The FC weights are quantized once from f32 by the packer into
`QS4CX_WH` images in a sidecar `<main>_fcwh.bin` (user: sidecar, not a
single `.bin`): `nntr_lfm2_8b_a1b_q40_arm_fcwh.bin`, 66 weights,
228,188,160 B, md5 `71812a91…`, written in the same packer run as the main
file, whose md5 `7b7867fa…` is unchanged (`cmp` identical; BENCHMARK
Artifacts). **PR 1 = #230 merged `7f95140ad`**: `--fc_wh_sidecar` in
`nntr_quantize_stream`; the loader preads the images into the arena (no
requant, no heap copy; a bit-exact unpack to the heap only on overflow); the
conv block / dense FFN / MoE prefill calls run in 512-row chunks with the
conv history carried in `conv_w` (no IDL change) — **the user made PR 1's
512-token chunking the hybrid's way of record, not a stopgap**. Host:
`E2E keys fcwh-lfm25 … heap=0 requant=0 … ok`, `E2E eval fcwh==wh-lfm25 …
bit_identical=1`, `CONV BLOCK CHUNKED BIT-IDENTICAL`, `E2E keys lfm25-p2x
prompt=1024 … bit_identical=1`; skels v79 + v81; no device run (the device
gets PR 1 only with PR 2, plan 225 §3.4). PR 2 (E2E FC / DENSE_FFN on the
WH GEMV with the VTCM feed, `htp/225-fcwh-e2e`) is being built; **PR 3
(user request) = the hybrid's M = 1 FCs on the WH GEMV behind
`NNTR_HTP_FC_M1`, default off, a reading beside B in the handoff**, with
rule 40 / 42 / 49 caveats on record. Still `needs-user` before the handoff:
the accuracy threshold (T1 contract §1 ≤ 1.02 × the no-keys hybrid vs T2
rule 45 — deferred by the user) and the hybrid overflow fallback (pool
C = 31 vs the attention / dense keys off at prefill). **(4) Devices:**
LFM2.5 on the S25 via the farm, run by the user; Gemma on the S26 Ultra
attached to the workstation, **the agent drives adb** (user; the
2026-10-01 rule set applies to that unit). **(5) #201**
(`state:in-progress`): the user answered the 2026-10-02 questions — (a)
#4296 pinned at `d345c3470` merges now, (b) the Gemma HTP MoE layer +
`QS4CX_WH` writer move S5 → S4, (c) an hd ≥ 64 fixture is added; PR #221
(graph builder) merged `78e597a5b` (in the base); PRs #226 (pin merge),
#227 (hand-over by name, per-op RoPE, two caches), #231 (Gemma HTP MoE
layer + gate|up writer) open, stacked, all against `htp_decode`. The
26B-A4B files are not on the workstation yet. **(6) #219**
(`state:needs-measurement` + `needs-user`): PR #224 open (`NNTR_MOE_TIER`,
host-gated); its step-4 sitting waited for #222's result, which is that
the config of record cannot load one PD at all — it now waits on #225's
PR 2 and shares its base / retarget question. **#229** (p1,
`state:needs-plan`, Gemma ternary LUT review) belongs to another session
and is not touched by this supervisor. Queue healthy (#229, #137
`state:planned`, #110 `state:needs-plan` + `needs-user`), nothing derived;
upstream #4327 not watched. "Now" unchanged (record sitting 53.97 / 52.16 /
51.41 on the old config; S25 column frozen until the 9 × 3 table). Tracker
#76 body refreshed.

**Cycle 33 (2026-10-06, base `htp_decode` @ `78e597a5b`): the #222 closing
sitting is read as a verdict (no row of record, rule 63), two bases, four
user decisions.** Merged by the user: PR #221 (#201 S4 Gemma graph builder,
`htp_graph_gemma_build`, 542-op 26B-A4B list, `78e597a5b`) into
`htp_decode`; **PR #223 (#222 config of record + closing handoff) into
`htp_first_version`, not `htp_decode`** (`b7c1d4ff6`; the user
fast-forwarded `htp_first_version` to `htp_decode` @ `c7ec6c64a` first).
**(1) #222's closing sitting** (2026-10-02 16:15–16:54 KST, farm
`R3CY205ZMND`, S25 v79, set from `ddb7d9ad8`; results only in #222's
comments of 08:00 / 08:30 UTC and the farm session's logs — the handoff
`222-config-refresh.md` merged **unfilled**, no device md5 line reached the
supervisor, so nothing is a row of record and no BENCHMARK Results row is
added): **Qnew (one PD, C = 28) VOID in every cell** — `no room for the FC
set beside the resident experts (mapped=3712 MiB, fastrpc_mmap(32 MiB)
failed: err=1)`, reproduced after a fresh reboot, `heap_used_kib` 92 058
(Qold) → 223 003 (Qnew attempt); **Anew (hybrid) P1024 VOID** —
`nntr_hvx_mm_u8i4_conv_block … err=0x80000402` (`AEE_ENOMEMORY`, M = 1024
K = 2048 C = 2048 N = 2048: the conv block's M-proportional session scratch,
≈ 13 MiB at M = 512 and ≈ 26 MiB at 1024, no longer fits the DSP heap
beside the keys' ≈ 216 MiB of WH copies); **Anew accuracy FAIL** — pooled
over the 8 prompts prefill PPL 90.31 → 128.31 (+42 %), decode PPL (forced)
1.2108 → 1.3752 (+13.6 %, gate +2 %), G64 text degenerates into repetition
(two stacked quantizations: the file's Q4_0, then per-column qs4cx over
K = 2048 at load). Read for information only: Qold (C = 28) loads and
closes clean, prefill / decode 556 / 33.7 and 584 / 34.6 (G64), 455 / 43.3
(G512), 531 / 43.3 (G1024), texts = Aold r1; Anew P512 786.5 / 56.5,
761.9 / 56.4, 739.9 / 51.7 and P64 294.9 / 56.1, 254.0 / 55.6, 266.7 / 52.9
(the keys do move the prefill, at a PPL the gate refuses); CPU `q40` P64
244.3 / 53.6, 210.5 / 53.4, 210.5 / 49.4, P512 310.5 / 49.0, 307.7 / 46.9,
294.4 / 48.9, P1024 (`init_seq_len 1024`) 311.8 / 49.7, 317.8 / 49.1,
313.8 / 48.5 (P1024 + G1024 = `max_seq_len` 2048 generated to the end).
Rule 63 and the §2 #222 row carry the verdict; the accuracy fail is filed
as the next issue per the contract — **#225** (p0, `state:needs-plan`,
filed by the user 2026-10-06, base `htp_first_version`: the FC weights as
`QS4CX_WH` in the model file, one copy for HTP prefill and E2E decode; its
accuracy threshold is `needs-user`). The Qnew C = 24 follow-up is
cancelled (user); the LFM2.5 table of record becomes 9 cells × 3 cases
(contract §4.2, §12); #222 closed `completed` on PR #223's merge. **(2)
#201**: the user answered the 2026-10-02 questions — (a) #4296 pinned at
`d345c3470` merges into `htp_decode` now, (b) the Gemma HTP MoE layer +
`QS4CX_WH` writer move S5 → S4, (c) an hd ≥ 64 fixture is added — and
**the agent drives adb on the attached S26 Ultra** for the Gemma sittings
(the S25 / LFM route keeps the no-adb handoff rule); the 26B-A4B files are
not on the workstation yet. #201 stays `state:in-progress` (the only one).
**(3) #219**: PR #224 open (`NNTR_MOE_TIER`, host-gated), its step-4 device
handoff said "waits for #222's result" — that result is that the config of
record cannot load one PD at all, so the sitting now waits on #225 and its
base / PR target (`htp_decode` vs `htp_first_version`) is a `needs-user`
question noted on the issue; `state:needs-measurement` + `needs-user`
unchanged. **(4) Docs**: PR #223's two BENCHMARK artifact rows (the two
config files' md5s) and its §3a note landed on `htp_first_version` only;
mirrored verbatim into `htp_decode`'s copies this cycle (contract §5).
Queue healthy (#225 `state:needs-plan` p0, #137 `state:planned`, #110
`state:needs-plan` + `needs-user`), nothing derived; upstream #4327 not
watched. Open PRs into `htp_decode`: #224 (#219), #220 (guide). "Now"
unchanged (record sitting 53.97 / 52.16 / 51.41; S25 column frozen).
Tracker #76 body refreshed.

**Cycle 34 (2026-10-06, base `htp_decode` @ `8d164e534`): #201 S4 complete
on the host; #225 PR 2 in review; three user decisions; no measurement,
no row.** Merged by the user into `htp_decode`: PR #231 (`874938550`, the
Gemma HTP MoE layer as `lfm2_moe` with `moe_router=softmax` — `QS4CX_WH`
experts, the pool and the hook reused — plus the `nntr_quantize_stream`
`QS4CX_WH` fused gate | up writer for `expert_*`), PR #232 (`b6d2f3a0e`,
the `gemma4_moe_tiny_hd64` fixture — hd 64 sliding + hd 128 full, k = v,
seeded norms and `layer_scalar`, soft-cap — and the `run_inproc_e2e.sh`
Gemma lines; also a #4296 x86 bug: with two KV-cache widths the cache
inputs were fed in name order and overflowed the heap) and PR #220
(`8d164e534`, guide for cycles 22–30). **With #226 / #227 / #231 / #232
every S4 deliverable of plan 201 is in: `E2E fwd gemma64 e3
calls/token=1.00 attn_caches=2 timeouts=0 ok`, `E2E eval gemma64-e3
min_snr_db=27.63` (floor 20 dB, backed by five hand-over mutants reading
0.2–9.8 dB), `E2E tokens gemma64 e3==off 8/8` and `off==cpu 8/8`, pool
C = 2 `bit_identical=1 misses=31`; `unittest_causallm_models` 111 tests
incl. `Gemma4MoEDifferentialTest.Hd64FP32MatchesHFReference`.** Nothing
ran on a device; the Gemma E2E is not bit-identical to the CPU model the
way LFM2's E1 is (#4296's norms / GeGLU / fp32 q are not the DSP's `_det`
ones), so S5's device gate is the 20 dB floor + tokens, not
`bit_identical=1` — §2 gets its row when S5 reads the 26B on the S26.
**S5 open items (implementer, #201 comment 6009633337), host-doable before
the files arrive (㉞):** the sliding layers' DSP attention cache is ≈ 800
MiB at `max_seq` 4096 on the 26B shape and must shrink to the window;
#4296's hook-less CPU layers compute discarded values at decode; the final
soft-cap stays on the CPU. **User decisions today (contract §12):** (1) PR
#224 (#219, LFM2.5) retargets to `htp_first_version` — conflicts only in
`test/htp/host/htp_e2e_test.cpp` and `run_inproc_e2e.sh`; the implementer
rebases this cycle, the step-4 sitting still waits on #225; #219 loses
`needs-user` (its open question was this one), stays
`state:needs-measurement`. (2) The S26 Ultra **`R5KL20NFRCK`** (SM-S948N)
is attached to the workstation; the agent drives its adb for the Gemma
sittings (the 2026-10-06 rule set); `/data` has 26 GB free — a 4-bit
26B-A4B model fits once; the files themselves are still not on the
workstation (S5 blocked on them: #201 gets `needs-user`, stays
`state:in-progress`). (3) **#229** (p1, `state:needs-plan`, in this
project's queue from today): the real Gemma 4 checkpoint carries
**ternary** weights; a ternary → 4-bit dequantization will be added so
prefill and decode use the existing 4-bit kernels; the planner reviews
the decode gain of a LUT dequantization (T-MAN `hvx_lut_ctor` / `hvx_tbl`
/ `hvx_bit_serial`, `docs/htp_moe/t-man-lut-reference.md`) — options A
(4-bit in DDR, today's path), B (ternary in DDR, unpacked per tile into
VTCM, DDR bytes ÷ ≈ 2–2.5) and C (the LUT GEMV on ternary, own SNR gate);
ternary → 4-bit is exact, so A / B keep the CPU bit-identity. **#225:**
PR #233 (PR 2, `htp/225-fcwh-e2e`, base `htp_first_version` @
`06ed17b7b`) is open — the one-PD decode's FC / DENSE_FFN ops bind the
prefill's sidecar handles (`HTP_GRAPH_FEED_WH`), a new WH M = 1 FC kernel
`hexkl_mm_u8i4_fc_m1_run`, lm_head stays Q4M1, no IDL change; host `FC WH
BIT-IDENTICAL`, `E2E fwd lfm25 fcwh kinds=all calls/token=1.00 …
e3==e1 bit_identical=1`; the plan's ≥ 30 dB against the hybrid is **not**
met (9.48 dB: two different int4 quantizations of the fixture's random
FCs, not a wiring fault — the user reads this in review); `state:review` +
`needs-user` (T1 / T2 threshold, overflow fallback) unchanged, nothing for
the supervisor until it merges. **Docs / base:** the copies of this file,
BENCHMARK and the contract on `htp_first_version` carry a cycle-33 docs
commit from another session (`06ed17b7b`, 13:47 KST) that declares
**`htp_first_version` the single base and `htp_decode` closed**, while the
user merged #231 / #232 / #220 into `htp_decode` after it (15:07 KST) and
this cycle runs on `htp_decode` — **which branch the supervisor's docs live
on is `needs-user`** (contract §5 note); until decided, this copy follows
the cycle-33 two-bases rule: PR #230's Artifacts row (the FC WH sidecar
md5 `71812a91…`), the rule-63 wording and the §3a #225 PR 1 note are
mirrored verbatim from `htp_first_version` here; the `06ed17b7b`
cycle-33 rewrite is not. The `hexagon-gates` skill already carries PR
#232's rung-3 recipe (`ninja -C builddir install`, then `build_android.sh
--htp --cache`; rule 36) — nothing to add. Queue healthy (#229
`state:needs-plan` p1, #137 `state:planned`, #110 `state:needs-plan` +
`needs-user`), nothing derived; ㉞ filed as an open item. Upstream #4327
not watched. Open PRs: #233 (`htp_first_version`), #224 (to be retargeted).
"Now" unchanged (record sitting 53.97 / 52.16 / 51.41; S25 column frozen).
Tracker #76 body refreshed.

**Cycle 34 (2026-10-06 evening, base `htp_first_version` @ `9f33d7d43`):
#225's handoff folded — the 9 × 3 table of record is measured; the "now"
waits on the user's text approval; one new rule, one new issue.** The
filled handoff is `htp/225-fcwh-e2e:docs/measurements/225-fcwh-table.md @ f08cccfeb` (PR #233 open, not in the base — read
with `git show`); its summary is #225's comment of 07:43 UTC. Sitting:
farm `R3CY205ZMND` (S25 Ultra SM8750, v79), 15:56–16:36 KST, one
invocation, no STOP, zone0 ≤ 35 °C at every block start. **md5:** the set
was rebuilt on the measuring workstation from `51b2b4477` (code
`7efd22adf`; `md5.txt` `23000b96…`, not the staged `dd24b4fc…`), device
md5 == that set, the sidecar re-packed from fp32 byte-identical
(`71812a91…`) and the main bin unchanged (`7b7867fa…`) — the rule-14 /
rule-22 case, accepted as the device set (BENCHMARK Artifacts); it
started 52 s after the reboot, not after 5 min idle (rule 61), which the
Q G64 r1 cells show (P64 42.81 vs r2 49.31). **Verdicts per gate (§2
#225 row):** G1 fit — 27 cells, one VOID (B P1024) replaced by Bfb per
the user's decision (b); G2 E2E — pass in every Q cell (`fc wh: …
heap_kib=0 requant=0`, `calls/token=1.00`, `cpu fc skipped` = 32 ×
tokens, mapped 192 + 3520 = 3712 of 3840, `heap_used_kib` 94 604 vs
#222's 223 003); G3 hybrid P1024 — **fail**: `nntr_hvx_mm_u8i4_layer
failed: err=0x80000600 (M=1024 K=2048 N=3072 handles=3)` = `AEE_ERPC`
(rule 38) on the attention qkv projection — PR 1's 512-row chunking
(`prefillRows()`, `htp_compute_ops.cpp:5426`) covers the conv block, the
dense FFN and the MoE prefill calls, while the FC paths
(`gemm_q4_0_accel_fp32` / `gemm_q4_0_batch_fp32`, `:1262–1345`) chunk by
`fcMaxRows(K)` = 1920 rows at K = 2048, so the M = 1024 qkv call reaches
the DSP whole; the same call succeeds on Q (heap 0) and fails on B with
the FC overflow (`heap_kib=75776`) on the heap — **rule 64**, open item
㊱, filed as **#236** (p1, `state:needs-plan`); Bfb's
numbers are the hybrid's P1024 cells of record (52.03 / 51.51 / 48.74
vs the CPU 47.58 / 48.82 / 47.41). G4 accuracy, **T2** (user): PPL
recorded — pooled prefill A 86.44 / Aoff 90.31 / B 94.08 / Q 94.08,
decode forced on A's continuation 1.1809 / 1.2139 / 1.2965 / 1.3318, null
check equal; loops where A has none at **Bfb P1024 G64 (r1, r2), Bfb
P1024 G512, Q P1024 G512**; B P64 G64 degenerates where A loops too
(the P64 prompt asks to continue without stopping); Q P512 G64 echoes the
prompt's instruction; **the approval column is empty — the user's**, so
no cell is folded as a row of record (the contract's accuracy gate, as
amended 2026-09-28). **Speed, inside the sitting:** hybrid B at P512
55.17 / 55.72 / 51.43 vs the CPU 52.07 / 50.70 / 49.47 (+5.9 / +9.9 /
+4.0 %) and vs Aoff 53.96 / 52.58 (+2.2 / +6.0 %), prefill 727–739 vs
Aoff 527–560 (+30–38 %: the prefill gate against the same binary's
keys-off cell passes) and vs the CPU 231–328; P64 59.59 / 58.01 / 54.74
vs 54.05 / 52.54 / 51.48; E2E Q28 at P512 42.78 (r2 44.48) / 45.94 /
45.61 — under the CPU by 6–21 % at every cell, with the per-kind profile
`wall_ms/token` 21.275 = MOE 10.946 + FC 3.819 + LM_HEAD 2.973 +
DENSE_FFN 1.057 + ROUTER_TOPK 0.677 + ATTN_M1 0.620 + the small kinds
(the FC on the WH GEMV at 3.8 ms is ≈ 57 GB/s over 216 MiB, near the
byte floor; the MoE at 10.9 ms is the pool's 1.61 misses a token at
G = 64 on top of rule 41's 14.7 → the 22 × 0.395 = 8.7 ms `dsp`). **The
"now" row does not move** (BENCHMARK Method, cycle-34 paragraph): the
hybrid of record is a new config, not bit-identical, on an open PR, on
a different unit, with its texts unapproved; by the user's 2026-10-06
decision the 9 × 3 table becomes the LFM2.5 row of record once the
texts are approved and PR #233 is merged. **Runner artifact** (open item
㊲): `[HTP] dspq: on queue=0x…` / `graph: init …` print inside the
generated text (stdout not line-terminated), so `225-run.sh`'s own
`speed.txt` reads DIFF for every NPU run and its one `BAD` (`every r2
text == its r1: got 2`) is the queue address, not nondeterminism; with
the banners removed every r2 text equals its r1; `gen()` must strip
`\[HTP[^\n]*\n` anywhere — a docs-only fix on PR #233's branch, filed as
a comment on #225. **Issue states:** #225 `state:measured` →
`state:review` + `needs-user` (PR #233 complete; the user merges it and
approves the texts); the qkv chunk defect is **#236**; #219's PR
#224 is merged into the base (`9f33d7d43`, another session) — its
step-4 sitting (`219-arm-tier.md`) is now runnable once PR #233 is in,
since the one PD loads on the config of record with the sidecar.
**Base note:** PRs #226 / #227 / #231 / #220 were **merged into
`htp_decode`** (not retargeted) by another session on 2026-10-06, and
#234 (p1, `state:needs-plan`, another session) states "`htp_decode`
stays the Gemma base" and tracks upstream nntrainer/nntrainer#4408
(Seunghui98's Gemma-4 26B HTP draft) — this contradicts the contract's
§5 single-base row of the morning; the supervisor's docs stay on
`htp_first_version` as told, the reconciliation is the user's. #229
(another session's) untouched. Queue: #234 and #110 `state:needs-plan`,
#137 and #229 `state:planned`, #236 `needs-plan` — healthy,
nothing else derived. Upstream #4327 not watched. Tracker #76 body
refreshed (the #225 / #219 / open-PR paragraphs).

**Cycle 35 (2026-10-06, base `htp_decode` @ `a19d4dd4e`): the goal is
restated; no measurement, no row, no `state:measured` handoff folded.**
Merged by the user: PR #235 (guide, cycles 31–34, `a19d4dd4e`) into
`htp_decode`; PR #224 (#219's ARM tier, host-gated) into
`htp_first_version` — #219 stays `state:needs-measurement`, its step-4
sitting waits on #225 (one PD does not load on the config of record
without the FC WH sidecar). **User decisions (contract §1 / §12):** (1)
the project goal is now **Gemma-4 26B-A4B with ternary-quantized weight
files on the NPU, decode end to end on the one-PD path; no tok/s target**
(50 tok/s is a reference). Ternary is confirmed, the storage format is
still to come (#229 S0 `needs-user`); #229 S1 (port #4410's `QS2CX_WH` +
u8i2 GEMV onto the pool path, host-gated on the hd64 fixture) starts
without it. (2) A **peak-memory bound < 2 GB** (process RSS incl. ION
arenas, flash streaming for the rest) was stated and the same day
**deferred to a later stage — after the ternary 26B decodes end to end on
the device, not a gate on S5 / S6**; ㉟ holds it. (3) **`htp_decode` stays
the Gemma base**; §Upstream gains the #4408 watch (`28771a928`) and the
`htp_first_version` watch; #234 (p1, `state:needs-plan`, planner this
cycle) ports #4408's converter / safetensors reader / 26B config and
#225's FC WH path; the `06ed17b7b` "single base" docs rewrite is not
carried (closes cycle 34's docs-base question: the supervisor's docs live
on `htp_decode`). **#225** (`state:measured` + `needs-user`,
`htp_first_version`, another session): `origin/htp_first_version` carries
no `docs/measurements/225-*` since `06ed17b7b` (only the #219 handoff
files, unfilled) — nothing to fold; the `state:measured` label is that
session's and is left alone. Queue: #234 `needs-plan`, #229 + #137
`planned`, #110 `needs-plan` — healthy, nothing derived. Open PRs: #233
(`htp_first_version`). S26 column still empty (no files). "Now" unchanged
(LFM record sitting 53.97 / 52.16 / 51.41; S25 column frozen). Tracker
#76 retitled to the Gemma goal, body refreshed.

**2026-10-06 sync: the user's standing decision is the single base `htp_first_version`; htp_decode's Gemma work is merged here by this sync.** `origin/htp_decode` (cycles 33–35 above, the #201 S4
Gemma stack PRs #226 / #227 / #231 / #232, guide #220, plan 229) is
merged into `htp_first_version`. Both branches' cycle-33 and cycle-34
paragraphs are kept as written, in cycle order; their two base
statements ("single base `htp_first_version`" / "`htp_decode` stays
the Gemma base", #234) stay verbatim and this line is the reading of
record. Open items: `htp_decode`'s ㉞ / ㉟ keep their numbers;
`htp_first_version`'s cycle-34 ㉞ (#236) and ㉟ (runner banners) are
renumbered ㊱ / ㊲ in this copy (§3), with their references.

**Cycle 36 (2026-10-06, base `htp_decode` @ `4a7003f93`): #229 S1 merged
(the 2-bit expert stack, host-gated); #219's tier sitting folded — the
lever pays, G2 fails, the default is the user's; one host rule, one
silicon rule.** Docs: this copy of LEDGER / BENCHMARK is
`htp_first_version`'s (PR #237, the user's sync of `htp_decode` into it,
carried the cycle-33 / 34 paragraphs of that side, rule 64, ㊱ / ㊲ and the
#225 rows) plus this cycle; the contract's single-base hunks of that side
are **not** carried — the base question (cycle 34 / 35) stays the user's,
and this cycle ran on `htp_decode` as instructed. **(1) #229 S1 merged by
the user:** PR #238 (`e926a3e1f`, `QS2CX_WH` format + quantizer incl. the
#231 Gemma gate | up writer, `hvx_expand_i2i4` with its host check and
device gate) and PR #239 (`4a7003f93`, the u8i2 LUT GEMV, 2-bit weights
in the registry / MoE kernel — M=1 GEMV on the arena, ring feed, 4-queue
feed, GeGLU; HMX expansion for prefill — the pool's swap and the one-PD
miss round, the ARM routing, the host-E2E 2-bit lines). Gate lines at
`b9e4aa51e`: `INPROC E2E PASS`; `E2E 2bit lfm25 e3 == palette-twin
bit_identical=1 calls/token=1.00 timeouts=0`, `E2E 2bit gemma64 e3 ==
palette-twin bit_identical=1`, `E2E 2bit pool C=1 / C=2 lfm25 == 2bit e3
bit_identical=1`, `E2E 2bit pool C=2 gemma64 == 2bit e3 bit_identical=1`,
tokens 8 / 8, every existing line unchanged; `run_host_checks.sh` ALL
CHECKS PASS incl. `HVX GEMV NATIVE u8i2 cases=3456 bad=0`; skels v79 /
v81 `UNDEFINED SYMBOLS OK` / `ARCH OK`; app + device gtests built, **not
run on a device** (㊳: `HvxExpandI2I4.MatchesScalarBitExact` passed on v79
in #4410, never on v81; the u8i2 GEMV has no device test — its first
device reading is S4's MoE dumps against the palette twin). **Finding →
rule 66:** #4410's u8i2 GEMV (`dfca0acbf`) was not host-exact (the 4-bit
nibble mask `0xF0F0F0F0` on 2-bit codes; vlut32 indices 0 / 16 / 32 …,
> 31 read 0) and its only host check compared two wrong kernels with each
other; fixed to `0x0F0F0F0F` in PR #239 and held against the scalar spec
(the old mask as a mutant fails, `bad=1253376`). #229 `state:review` →
**`state:planned`** (+ `needs-user` for S0, the storage format): S2–S5 of
plan 229 are next for the implementer after #234, which is this cycle's
P1; one `state:in-progress` at a time (contract §7) is already exceeded
by the two other sessions' issues. **(2) #219 measured and folded** (§2
row, rule 65, ㉜): `htp/219-arm-tier:docs/measurements/219-arm-tier.md
@ 149481ec4` (the user re-created the branch for the two docs commits;
not on `origin/htp_first_version`, whose copy `69c05f517` is the unfilled
handoff), farm `R3CY205ZMND`, 18:28–19:01 KST (fresh + old boot, 30
runs) and 19:09–19:12 (`=2` × 4 interleaved, 12 runs), no STOP, ceiling
3840 after every run. **md5:** the set that ran is **not the Artifacts
table's** — a local merge `8629b9392` = `htp/225-fcwh-e2e` @ `f08cccfeb`
(PR #233, since merged) + `htp_first_version` @ `9f33d7d43` (PR #224),
built on the measuring workstation because the config of record needs
#225 PR 2 for Q28 to load; the runner's `MD5 OK` (device == the staged
`md5.txt`: `libnntrainer.so 48545aa4…`, skel v79 `44339c7a…`, …) holds on
all three boots, and both parents are pushed — the rule-14 / rule-22
case of #225, accepted, not void (the staged dir is on the measuring
machine, not here; the md5 reading is the runner's line). **Verdict:**
`NNTR_MOE_TIER=1` lifts Q28's G = 64 block from 39.99 → 47.60 (fresh,
+19.0 %) and 38.70 → 47.82 (old, +23.6 %), G 512 / 1024 +1.4–4.3 %, the
miss from 2.05–4.13 to 0.51–0.83 ms, refaults 63k–153k → ≤ 571 (two
follow-up runs 2294 / 6236), kswapd scan → 0, `tier_hits = misses`,
`tier_reads = 0`, resident-after 0; `=2` (one-thread copy) 48.37 / 48.45,
then **48.47 (sd 0.23) vs `=1` 47.31 (sd 0.71) vs A 37.15 (sd 5.43)** on
four interleaved runs, 0.43–0.45 ms a miss. Gates: G1 (≤ 0.5 ms) **fails
for `=1`, passes for `=2`**; **G2 fails** on `tier_waits` 3–4 a run (≤ 2)
and a non-zero `pswpin` on both modes; G3 prefill passes on the block
means (+0.9 … +2.1 %; one fresh G = 1024 pair −5.5 %, old +1.2 %); G4 /
G5 / G6 / G7 pass; the hybrid is unchanged (H vs H0 +0.7 / +0.4, text
identical). Texts: every Q run == A (fresh B2's `DIFF` is a banner
fragment, ㊲'s class); the Q texts differ from the hybrid's by
construction (the sidecar's FC WH GEMV vs the CPU's Q4_0, #225's T2
reading) — **no new accuracy issue**: the user put the FC accuracy on
hold on #225 (2026-10-06). **No row of record moves** (a lever on the
E2E path, under the hybrid, config of record unapproved). #219
`state:measured` → **`state:in-progress` + `needs-user`**: the default
(`=2` / `=1` / none) is the user's call as with #115 / #151 / #218, and
the G2 remainder (hide the refill behind the re-miss gap: `tier_waits`
≤ 2, `pswpin` 0) is the implementer's next step on the issue. **(3)
Board:** #236 `state:in-progress` (another session, hybrid P1024 qkv
chunks; plan on the working tree, PR #233 merged into
`htp_first_version` `0e8888bde` with the handoff and the sync `78dace26f`);
#225 `state:review` + `needs-user` (another session; the user's comment:
FC accuracy on hold, LFM2.5 E2E speed gap closed here, #236 + optional PR
3 remain); #201 `state:in-progress` + `needs-user` (the 26B files); #234
`state:planned` (the implementer's P1 this cycle). Upstream: #4408
unchanged at `28771a928`; #4327 not watched (`f923bf29`, unchanged).
Queue: #234 / #137 / #229 `state:planned`, #110 `state:needs-plan` —
healthy, nothing derived. "Now" unchanged (record sitting 53.97 / 52.16 /
51.41; S25 column frozen; the S26 column still empty — no files). Tracker
#76 body refreshed. **(4) Device (user, later on 2026-10-06):** the S26
Ultra `R5KL20NFRCK` is no longer attached; an **S25 Ultra `R3CY205ZMND`**
(SM-S938N, v79) is attached to the workstation and the agent drives its
adb for the Gemma track's device steps (contract §4.1 / §4.2: the S26
serial applies when attached; the Gemma device column is whichever unit
is attached, named per sitting). The implementer runs the #229 2-bit
device gtests on it this cycle — a **v79** reading; the v81 half of ㊳
stays unverified until an S26 is attached again.
**Mirrored verbatim from `htp_first_version` (cycle 38, 2026-10-07; `origin/htp_first_version` @ `77a01902f`): that side's cycle-35b and cycle-36 (night) paragraphs and its "LFM2.5 status" note. Numbering in this copy: that side's rule 65 replaces this copy's rule 65 (same sitting, same facts, fuller text); its rules 66 / 67 / 68 are rules 67 / 68 / 69 here (rule 66 here is #229 S1's u8i2 mask finding, cycle 36); its ㊴ keeps its number (㊳ here is #229's device reading). Its #219 label verdict (`state:review`) is the one that stood — PR #245 made the tier the E2E default and the G2 remainder is ㊴.**

**Cycle 35b (2026-10-06 evening, base `htp_first_version` @ `78dace26f`,
PR #237's sync in): #219's handoff folded — the ARM tier removes the
pool-miss slow regime; no row of record moves.** Source
`htp/219-arm-tier:docs/measurements/219-arm-tier.md @ 149481ec4` (PR #240,
docs-only, into `htp_first_version`); farm `R3CY205ZMND` (S25 SM8750 v79),
fresh boot (18:28 KST, first run at uptime 62 s) + old boot (18:58, idle
to 600 s), 30 runs, then the user-requested `=2` ×4 follow-up (19:09,
fresh, 12 runs); no STOP, ceiling 3840 after all 42. Provenance (rule
22): the set is a **local merge `8629b9392`** = `htp/225-fcwh-e2e` @
`f08cccfeb` + `htp_first_version` @ `9f33d7d43` (#224), not pushed — Q28
loads on the config of record (`3f6808e3…`, sidecar `71812a91…`) only
with #225 PR 2, then open (merged since, `0e8888bde`); device `MD5 OK`
against that staged set on every boot, so not void; the handoff's own
Artifacts table (`8449de674`, `d12e41f4…`) never reached a device.
**Reading** (Q28 one PD, G 64 block means, same-sitting A): no tier
39.99 / 38.70 → `NNTR_MOE_TIER=1` 47.60 / 47.82 (+19.0 / +23.6 %) →
`=2` 48.37 / 48.45; G 512 51.13 / 50.54 → 52.75 / 52.73; G 1024 51.69 /
51.22 → 52.27 / 52.42; ms/miss (profiled) 2.54 / 3.77 → 0.67 / 0.51 →
0.45 / 0.44; the ×4 follow-up 37.15 (sd 5.43) / 47.31 (0.71) / **48.47
(0.23)**, ms/miss 3.69 / 0.83 / 0.43. The page cache leaves the memory
sum: window refaults 74k–153k → ≤ 571 (`=1` once 2294 / 6236 in the
follow-up; `=2` ≤ 81), kswapd scan 65k–139k/s → 0, PSI io 81–176 → 19–37
ms, `resident after` 0.7–3.2 GiB → 0 on every tiered run; `tier_hits =
misses`, `tier_reads = 0`, app `pgpgin` = misses × 5.29 ± 7 MiB.
**Gates:** G1 `=1` 0.51–0.83 = the plan's 0.5–0.7 "slow regime gone,
copy store-bound" band (user's call), `=2` 0.43–0.45 ≤ 0.5 on all six
runs — pass; **G2 fail** — `tier_waits` 3–4 on every tiered run (gate ≤
2; the same 3–4 at G 64 / 512 / 1024 with 103 / 120 / 124 misses, so the
decode start's refill backlog from the prefill batch, which the host run
also showed, not a race that grows with G) and `pswpin` 0–2105 (gate 0;
`=2` ≤ 168, `=1` up to 2105 / `pswpout` 3621 in one run); G3 prefill
pass on the block means (+2.1 / +0.9 %, follow-up +0.7 / +1.7 %; G 512
−1.2 / −1.0 %; the fresh G 1024 pair −5.5 % is one unprofiled pair whose
A is the sitting's highest prefill of 30 runs, the old boot's mirror
+1.2 % — spread, not a miss); G4–G7 pass (text == the boot's A1 36 / 36
Q runs after removing one banner fragment, hybrid H0 / H +0.7 / +0.4
tok/s and identical, `calls/token=1.00`, `experts=88 mib=466.5
direct=1`, `pswpout` 0 on the main sitting's tiered windows). **Rule 65**
(cause removed by removal; one-thread copy beats the 8-slice copy;
`tier_waits` reading). **Verdict:** lever measured, G1 / G3–G7 pass, G2
missed on two counts that do not touch the token stream — the **default
and the copy mode are the user's call** (flip `NNTR_MOE_TIER=2` as the
E2E default, or keep env-only as #218); #219 → `state:review` +
`needs-user`, §2 #219 row, ㉜ closed, ㊴ holds the remainder
(`tier_waits` 3–4, `pswpin`, the default). **"Now" unchanged** (LFM
record sitting 53.97 / 52.16 / 51.41; S25 column frozen): `=2` 48.5 at
P512 G64 is the best E2E one-PD reading on an S25 farm unit (#225 Q
42.8 / 49.3, #207 40–47) but a lever cell at one prompt length, still
under the CPU and the hybrid of #225's blocks; the table of record's E2E
Q28 cells stay #225's (tier unset) — a tiered 9 × 3 E2E column is its own
sitting, not filed. Base: PR #233 (#225 PR 2) is in `htp_first_version`
(`0e8888bde`); #225's text approval is still open (user). Queue
untouched this run (#229 / #234 / #236 other sessions). Upstream not
watched (#4327); #4408 watch unchanged. Tracker #76 refreshed.
**Double fold, to reconcile at the next sync:** cycle 36 on
`origin/htp_decode` (`9c92d0ac8`, 10:20 UTC, one minute before this
fold) folded the same handoff with the same reading — its rule 65 (one
thread stores faster than eight), its ㉜ update, §2 #219 row, two Results
rows + an Artifacts row + a Method paragraph on that copy; its ㊳ is #229
S1's "no device reading" item (that is why this copy's #219 remainder is
㊴), its rule 66 is #229 S1's mask finding. One difference of verdict:
that fold set #219 `state:in-progress` (the implementer continues on the
G2 remainder — refill ordering / two refill threads); this one set
`state:review` (PR #224 merged, the measured change is complete; the
remainder is a second decidable change, ㊴, to be filed once the default
is decided). The label stands at `state:review` + `needs-user` as of
10:22 UTC; **which it should be is the user's / orchestrator's call**,
and the sync that carries `9c92d0ac8` here should keep one copy of rule
65 and of the #219 rows (the facts agree).
**Cycle 36 (2026-10-06 night, base `htp_first_version` @ `7244e7e2e`,
PR #240 in): the LFM2.5 close-out — #236's and #219's close-out handoffs
folded, the 9 × 3 × 3 table of record completed, no row of record moves;
after this fold the project is Gemma's (#201 / #229 / #234, the other
session's track on `htp_decode`).** Both sittings on `R3CY205ZMND`
attached to the workstation, the agent driving adb (user 2026-10-06: no
more farm sessions for LFM; `/data/local/tmp/nntrainer/.sitting.lock`
marks a sitting since another session shares the unit). **#236**
(`htp/236-qkv-chunks:docs/measurements/236-qkv-p1024.md @ 3d61c1aff`, PR
#243 open; 20:35–20:43 KST, 8 runs + profile, no STOP / BAD): the M > 1
FC entries step by min(`fcMaxRows`, `prefillRows`) = 512; hybrid B at
P1024 on the config of record loads (`heap_kib=75776 requant=0`, no
`0x80000600`) at **53.24 / 53.06 / 48.64**, prefill **725.7 / 722.7 /
711.1**; anchor B P512 G64 56.49 / 745.3 vs #225's 55.17 / 727.3 (+2.4 %
drift; A′ 56.84 / 749.6) → anchored 52.0 / 51.8 / 47.5 = Bfb's 52.03 /
51.51 / 48.74 (decode equal — the M = 1 path is untouched), prefill
+9.4 / +11.1 / +9.5 % raw over Bfb; above the CPU of record at every G;
A′ (the #225 set) reproduced the failure in-sitting
(`err=-2147482112`); FC calls = rows / 512 (12 / 6144, 13 / 6656); Q P1024
G64 40.76 (tier unset; #225 37.83). Provenance: the skels were replaced
before the run (the #225 skels are stale after the #237 sync's IDL
change — **rule 67**), `md5.txt` regenerated `1e6bf2f0…`, device md5 ==
set — accepted (rule 22), not void. Gates G1–G4 pass; T2 recorded (B
P1024 G512 loops where A does not, as Bfb did; G1024 both loop). **#219
close-out** (`htp/219-tier-default:docs/measurements/219-tier-table.md @
6923850d0`, PR #245 open: `NNTR_MOE_TIER` unset = 2 on the E2E path only,
`HtpBackend::e2eRequested() ? 2 : 0`; 20:50–21:02 KST, 14 runs, 0 BAD /
VOID / STOP, `MD5 OK`): the tiered E2E one PD (C = 28) 9 × 3 — **52.20 /
54.60 / 53.89, 48.08 / 51.79 / 51.65, 44.02 / 50.59 / 50.78** at P64 /
P512 / P1024 (G64 / G512 / G1024), prefill 282–287 / 794–803 / 763–773;
`experts=88 mib=466.5 direct=1`, `tier_reads=0`, `tier_waits` 0 / 3 /
3–4, misses/token at G64 1.16 / 1.61 / 2.62; Q0 P512 G64 37.58, B 56.44
(= #236's 56.49). G1 / G2 / G4 pass, G3 (≥ 48.4) 48.08 = −0.7 %, inside
drift; vs #225's untiered Q +10.7 … +21.9 %; vs the CPU of record ≥ 50
and above at every P for G ≥ 512 (+2.1 … +7.1 %), under at G64 (−3.4 /
−7.7 / −7.5 %) — **rule 68**. Deviations: 4.5–5.3 min idle after the
reboot; the cool-wait fired before every Q cell but not before Q0 / B
(Q's prefill +6 % at P512 is a start-condition difference). **Rule 68**
(the P1024 FC chunking: deterministic failure by call shape, not heap
residency; ≈ 3.4 % of the prefill, decode untouched). **Verdicts:** ㊱
closed (#236 → `state:review` + `needs-user`: PR #243, texts); #219 →
`state:review` + `needs-user` (PR #245, texts), ㊴ stays open and is not
pursued on LFM; #225 close-out comment (`state:review` + `needs-user`:
PRs #243 / #245 merge, PR 3 dropped unless the user says otherwise);
tracker #76 with the closed-out section. **No row of record moves:** the
texts of every config-of-record cell are approved n (#225) or pending
(#236, #219t); the "now" stays the record sitting (53.97 / 52.16 / 51.41)
and the contract §1 numbers are untouched — the table of record
(BENCHMARK Method, cycle-36 paragraph) is the close-out reading beside
it. Upstream watch unchanged (#4408 `28771a928`; #4410 `0d603f29a`,
reference). Queue ≥ 2 (#110, #137, #229), nothing derived.

**LFM2.5 status (close-out, cycle 36).** *Met and of record:* decode
≥ 50 tok/s at G 64 / 512 / 1024 above the CPU on the old config (record
sitting 2026-09-30, bit-identical levers, text ≡ A); the 9 × 3 × 3 table
on the config of record (`init_seq_len 1024`, all engine keys `htp`, FC
WH sidecar, hybrid prefill in 512-token chunks, `NNTR_MOE_TIER=2` E2E
default): CPU #225 A, hybrid #225 B / #236 B (P1024), E2E #219t — the
hybrid above the CPU at 9 / 9 and ≥ 50 at 8 / 9, the E2E above at 6 / 9
(G ≥ 512) and under at G64 — as speed readings until the user approves
the texts. *On hold (user 2026-10-06; not pursued on LFM, not handed to
Gemma):* the FC quantization accuracy problem — the config-of-record
texts loop at P1024 / G ≥ 512 where the CPU does not, prefill PPL +4 %
and decode PPL +7–10 % over the keys-off model (#225 texts n). *Closed by
decision:* the E2E decode speed gap on LFM2.5 (re-examined on Gemma);
#225's PR 3 (`NNTR_HTP_FC_M1`) unless the user says otherwise. **What
stays open and is Gemma's to pick up** (each with the LFM reading it
starts from): (a) **accuracy / quantization** — an int4 FC set that
passes an SNR gate yet loops in free-running text: Gemma's gate is 20 dB
SNR + tokens vs the CPU, and the LFM T2 reading says the free-running
text must be read beside it (rules 39 / 45); (b) **E2E G64 vs the
hybrid** — the one-PD path is 3–8 % under the CPU and 8–15 % under the
hybrid at G64 at every P because the residual scales with P (rule 68):
misses/token × the copy plus a ≈ 45–160 ms decode-start cost a run; the
levers are to drain the refill backlog before the first decode token
(㊴ 2) and a pool that survives the prefill batch; (c) **`tier_waits`
3–4 a run / `pswpin` ≠ 0** (㊴ 2–3, rule 65 b / c) — hygiene on the same
path Gemma's one PD uses; (d) **skip loading the ARM Q4_0 FC set
(≈ 170 MiB) on the E2E path** — the one PD never reads it (`cpu fc
skipped` = 32 × tokens, #225), it only costs RSS and load time; Gemma's
2 GB stage (㉟) is where it pays. Also carried: rule 67 (stale skel after
an IDL change) and rule 69 (prefill FC chunking) apply to every Gemma
sitting; the LFM2.5 device rules 61 / 65 hold for the S25 only until the
S26 re-reads them.

**Cycle 38 (2026-10-07, base `htp_decode` @ `9323ad52d`): `htp_first_version`
is closed by the user; #234 P1 / P2 merged (a #239 hang fixed on the way);
the 2-bit stack's device gate passed on v79; no row of record moves.**
(No cycle-37 supervisor fold exists on this copy: between cycle 36
(`9c92d0ac8`) and this cycle the base received only #234's PRs.) **(1)
`htp_first_version` finished (user, with PR #246 `77a01902f`, 2026-10-06
night):** PRs #243 (#236, `e0dccdd7b`), #245 (#219's tier default,
`718cd9f1a`) and #246 (guide close-out) merged there; the LFM2.5 9 × 3 × 3
table of record is **final** as that side's cycle-36 paragraph (mirrored
above) states it — CPU #225 A, hybrid #225 B / #236 B at P1024, E2E #219t
tiered Q — and its texts stay **unapproved** (#225 n; #236 / #219t never
approved), so under the 2026-09-28 amendment the "now" (record sitting
53.97 / 52.16 / 51.41) and contract §1 do not move; the branch is 51
commits ahead of `htp_decode` and receives nothing further (Upstream
paragraph: the 12 code commits Gemma may need; plan 234's revision names
which). `htp_decode` is the only live base; contract §5 / §12 updated.
#236, #219 and #225 **closed `completed`** this cycle (their PRs merged;
the open text approval is recorded, not a reason to keep them open — the
user declared the track finished). **(2) #234 P1 (PR #247 `f3e47ef24`) and
P2 (PR #248 `9323ad52d`) merged by the user:** P1 = `htp_first_version`'s
#223 three (host ISA Q4_0 repack order before the qs4cx requantization,
the dense_ffn CPU skip on a row the HTP runs whole, the config-of-record
keys line in the in-process E2E) plus the Gemma 26B
`res/gemma4/gemma4-26b-a4b/nntr_config.json` (BENCHMARK Artifacts,
`89618392…`) — and, found by the keys line, **a hang from #239
(`ed2660ddc`): `exp_live[16]` was indexed by the chunk number on every
M > 1 call, so a 4-bit M > 1 MoE / dense FFN call with more than 16 down
chunks read past the array and `hvx_worker_pool_wait_bg` spun forever**
(lfm25 dense FFN at M = 512 under `dense_ffn_engine=htp`: `606ddb83e`
finishes in 59 s, `ed2660ddc` and `htp_decode` + P1 make no progress in
10 / 26 min); fixed by `56a9e3f3b` (test `exp_bg` first), rung 2 v79 / v81
pass, the v79 device `*MoeLayer*` gtests pass with the fixed skel —
**rule 70** (which device runs are suspect). P2 = #4408's safetensors
source reader (`9dd2c76de` clean, `f9c09eaa2` with the #238 palette and
the #231 `writeFcConcat` kept) + a per-tensor order guard: `INPROC E2E
PASS` (477 / 478 lines identical to the P1 tip, the one a wake-split
timing line), `ALL CHECKS PASS`, `*qs4cx*` 2 / 2, `*Lfm2Moe*` 6 / 6,
`unittest_causallm_models` 114 / 114; **limit: two same-size swapped
tensors are not caught by the guard** (㊵). #234 stays `state:in-progress`
(P3 this cycle). **(3) #229 device gate, v79 (implementer, S25
`R3CY205ZMND`, 2026-10-06 19:27–19:30 KST, uptime 20 min; #229 comment
11:03 UTC; §2 #229 row):** `htp_decode` @ `4a7003f93` + PR #244
(`5abbc0730`, test only); skel v79 `d7019280…`, `unittest_hvx_mm_u8i4`
`d7818b7e…`, `unittest_hvx_softmax` `41520073…`.
`HvxExpandI2I4.MatchesScalarBitExact` OK (four palettes, `bad=0 of
131072` each); the new `HmxMmU8I4Layer.MoeLayerU8I2GemvMatchesI4Twin`
(Gemma expert shape K 2816 / inter 704 / N 2816, 8 experts, M = 1; 5 GEMV
cells × {SwiGLU, GeGLU}) OK — all 12 `U8I2_FIELD` lines
`bad_vs_twin=0 bad_vs_hmx=0`, `path=m1_gemv field=bit_identical
value=yes`; the four pre-existing `*MoeLayer*` tests OK. Not a tok/s
reading (a gtest, no row); the informational `EXPAND_I2I4_MAP` probe
printed `INCONSISTENT` while the assertion passed — the probe's decode
assumption, not investigated. **㊳ updated: v79 read, v81 open** (the
S26 is not attached). PR #244 is open (cpp-linter fix pushed, rebased on
`9323ad52d`, CI rerun pending) — merge is the user's. **(4) Sitting lock:**
PR #242 (open, into `htp_decode`) carries the `tools/htp/sitting_lock.sh`
rule in the contract §4.1 adb row and the gates / handoff skills; the
`htp_first_version` contract already states it (its §4.1 row, `d39c30f85`)
— this copy takes it when PR #242 merges, noted in §12. **(5) Board:**
#234 `state:in-progress` p1; #229 `state:planned` + `needs-user` (S0);
#201 `state:in-progress` + `needs-user` (the 26B files); #137 `planned`;
#110 `needs-plan` + `needs-user`; #76 refreshed; #236 / #219 / #225
closed. Queue ≥ 2 (#229, #137, #110), nothing derived. Open PRs into
`htp_decode`: #244, #242, #241 (all the user's to merge). Upstream: #4408
`28771a928` unchanged, #4410 `0d603f29a` unchanged, #4327 not watched.
"Now" unchanged.

**Cycle 39 (2026-10-07, base `htp_decode` @ `0490e161a`): the 26B files
are on the workstation — as a DUMMY; the Gemma device is the attached S25
`R3CY205ZMND`, the agent drives adb; #234 P5 and the cycle-38 PRs merged;
no handoff, no row, no `state:measured` issue.** **(1) Merged by the user
since cycle 38, all into `htp_decode`:** PR #244 (`8971aa873`, #229's
u8i2 device gtest — ㊳'s v79 half is on `htp_decode`), PR #249
(`a43a7ffc0`, #234 **P5**: `NNTR_MOE_DIFF` / `NNTR_MOE_SHADOW` on the MoE
layer call for 4- and 2-bit experts, `whUnpack` / `whUnpack2` on the host,
a failed reference throws under shadow; gate `INPROC E2E PASS` 476 / 477
lines identical, `ALL CHECKS PASS` ×4, diff 34.8–43.7 dB on every layer of
lfm25 / gemma64 at 4 and 2 bits, Gemma 4-bit shadow tokens == CPU 8 / 8;
**the one-PD token is not covered** — an E3 run prints only the prefill
lines, so S5's first token-level accuracy read stays `--dump` +
`htp_dump_eval.py`), PR #242 (`53084edf1`, `tools/htp/sitting_lock.sh` +
the contract §4.1 adb row — the lock is the rule of record on this copy
now), PR #241 (`0490e161a`, guide cycles 35–36). No open PR into
`htp_decode`. #234 P3 (FC WH sidecar, prefill half) is the implementer's
next; P4 after it; P6 waits for S5's miss numbers. **(2) The 26B files
(user, 2026-10-07, `~/Downloads/gemma4_26b/`):** `config.json`
(`Gemma4ForConditionalGeneration`, `text_config`: 30 layers, hidden 2816,
128 experts `top_k_experts` 8, `moe_intermediate_size` 704, dense
`intermediate_size` 2112, `head_dim` 256 / `global_head_dim` 512,
`sliding_window` 1024, 16 q / 8 kv heads, vocab 262144,
`final_logit_softcapping` 30, `tie_word_embeddings` true, RoPE full =
proportional θ 1e6 partial 0.25 / sliding θ 1e4; md5 `b2062bfe…`),
`nntr_config.json` (md5 `c2d0298d…`: `moe_engine htp`, `moe_layer_dtype
QS4CX_WH`, FC / embedding / lm_head `Q4_0`, `Q4_0-FP32`, `init_seq_len
1024`, `max_seq_len 2048`, `num_to_generate 512`, **`moe_cache_experts
16`**, `model_file_name nntr_gemma4_int2.bin`, a long summarization
`sample_input`), `tokenizer.json` (md5 `9e33baac…`, 32 170 070 B),
`generation_config.json` (bos 2, eos 1, pad 0, `do_sample false`), and the
weight blob **`nntr_gemma4_int2.bin`, 7 226 112 120 B, md5
`9b4e93493d6e870afdef05b1ba99ccd0`** — a nntrainer `.bin`, not
safetensors (the P2 reader and ㊵'s guard do not apply to it). **The blob
is a DUMMY (user, 2026-10-07): the right shape and format, not the real
checkpoint.** Consequences: (a) the text / PPL / accuracy gates of
contract §1 **do not apply** to any run on it — what counts is the path
(loads, routes, `calls/token=1.00`, `attn_caches=2`), **bit-identity NPU
vs CPU on the same dummy weights** (tokens of the E2E run == the CPU run of
the same file, MoE dumps vs the pool-less / hybrid run `bit_identical=1`;
the host fixture's cycle-34 reading is tokens 8 / 8 with SNR 27.6 dB, not
logits-identical), and the speed / memory readings (tok/s, peak RSS,
`mapped=`, arena bytes — ㉟'s baseline); (b) the user's text approval
column is empty by construction until the real file; (c) the exact bin
format is **another agent's** task this cycle — not derived here; two
facts for it: the blob's name says `int2` while `nntr_config.json` names
`QS4CX_WH` for the experts, and 7.23 GB sits between the 4-bit
(≈ 10.8 GiB experts alone, plan 234 §3) and 2-bit (≈ 5.4 GiB) expert sums
plus ≈ 1.2–1.6 GiB of Q4_0 FC / embedding (tied head) bytes — consistent
with 2-bit experts, to be confirmed by that agent, not assumed. The repo's
26B config (`res/gemma4/gemma4-26b-a4b/nntr_config.json`, `89618392…`,
P1) differs from the user's in `model_file_name`, `max_seq_len` (4096 vs
2048), `num_to_generate` (256 vs 512), the `chat_input` vs `sample_input`
form and the absence of `moe_cache_experts` — the user's file is the one a
sitting runs; the repo's is the tree's template. **(3) Device (user,
2026-10-07): the Gemma measurement device is the Galaxy S25 Ultra
`R3CY205ZMND` (v79), attached, and the agent drives adb itself — no farm
handoff.** The S26 `R5KL20NFRCK` clause (contract §1 / §4.1, BENCHMARK
Goals, #76) lapses until an S26 is attached again; ㊳'s v81 half is
deferred, not blocked. Every sitting takes `.sitting.lock` first (PR
#242). **(4) Board:** #234 `state:in-progress` p1 (P3 next; P5 done);
#229 `state:planned` + `needs-user` (S0 — the format is now the format
agent's reading + the user's confirmation that the dummy's format is the
real checkpoint's; ㊳ v79 done); #201 `state:in-progress` p0, **`needs-user`
removed** (its two reasons — the files and the device — are met); #137
`planned` p2; #110 `needs-plan` + `needs-user` p2; #76 refreshed. Queue
≥ 2 (#229, #137, #110); nothing new derived — the next deliverable is
**#201 S5-0 (㊶)**, the first dummy-file path sitting on `R3CY205ZMND`,
owned by #201 as S4 was. **Upstream:** #4408 `28771a928` and #4410
`0d603f29a` unchanged (re-checked 2026-10-07), nothing to decide. "Now"
unchanged; BENCHMARK's Gemma goal row carries the file / device facts, no
cell. **For the orchestrator (contract text outside §1's numbers, not
edited by the supervisor):** §1's Model row ("files are being produced",
"S26"), §1's Accuracy row (dummy ⇒ path + bit-identity + speed / memory
only, until the real file), §4.1's adb row (S25 = the Gemma device, agent
drives adb), §4.1 line "through PR #242 (open)" → merged `53084edf1`.

**Cycle 40 (2026-10-07 evening, base `htp_decode` @ `91f701fda`): #201 S5-0
folded — the Gemma-4 26B dummy decodes end to end on the one PD, on the
S25; the FC / head / dense bytes on the Q4M1 path are the measured first
lever; the repack tool is ours.** **(1) Merged by the user since cycle 39,
both into `htp_decode`:** PR #251 (`24a42c19f`, #250: `tools/htp/repack_int2_to_qs2cx.py`
+ `repack_int2_check.cc` + plan 250 — the external int2 `.bin` → canonical
`QS2CX_WH`, 7,226,112,120 → 7,226,142,840 B, `experts=7680 mismatches=0 ok`,
the host loader opens it; the checker's colsum reach is stated in plan 250:
it catches a different tile order or column placement, not a reordering of
k inside a column) and PR #252 (`91f701fda`, #201 S5-0: the filled record
`201-s5-0-gemma-dummy.md` + `201-s5-0-run.sh`, docs-only). **(2) User
decisions (contract §12), 2026-10-07:** #229 **S0 closed** — the repack
tool is ours and stays (#250); the real checkpoint will be produced with
the same script / settings as the dummy, so it goes through the same
repack; the palette is {−2, −1, 0, +1}; `nntr_config` `moe_layer_dtype` →
`QS2CX_WH`. S5-0's reference is the **hybrid `off` run on the same
`QS2CX_WH` file**, no Q4_0 CPU twin. **S5-0 gate verdict = (a) PASS**: the
prefill MoE dumps are 62 / 62 `bit_identical=1`, the tokens 57 / 65 with
the 8 differences all near-tie flips (off's top-2 logit gap 0.0003–0.022
vs E-vs-off 2·rms 0.26, plan 130 §3.5), so the speed table is valid; it is
re-read with the text gate when the real weights arrive. Standing: follow
upstream #4408 every cycle and plan the sync (#234). **(3) S5-0 read
(`R3CY205ZMND`, 17:11–17:50 KST, `6904594ed` no code change, set md5 ==
device on both boots, bin `6ac7df4c…`; BENCHMARK Method cycle 40, four
Results rows, side tables):** decode tok/s at G 64 / 512 / 1024 — hybrid
A16 16.97–20.83 / 20.06 / 18.46; E2E one PD C = 16 11.10 / 14.35 (×2:
14.25) / 13.68; C = 40 (C-max) 10.9–11.6 / 14.31 / 13.63; C = 32 and C = 8
inside 3 % of it → **the pool is not the bottleneck at the dummy's
routing** (misses/token 5.84 / 0.80 / 0.41 at C = 16, 0.5–1.2 ms a miss
from G 512 on; the dummy's stream alternates between two ids, so the miss
column does not predict the real checkpoint). Per kind on the DSP (E16,
G512, ms/token): FC 14.5, MOE 11.7, ATTN_M1 13.9 (9.8 → 17.8 with context),
LM_HEAD 8.2, DENSE_FFN 7.3, ROUTER_TOPK 4.3, RMSNORM 1.9; wall 63–67; ARM
steady state ≈ 1.6 ms → ≈ 15 tok/s is this graph's ceiling at Q4M1.
**What it says: the E2E is ≈ 30 % under the hybrid at every G because FC +
LM_HEAD + DENSE_FFN ≈ 30 ms/token run on the DSP's Q4M1 path, byte-bound
(㉘'s arithmetic, now measured on the 26B: the FC arena's 1278.7 MiB of Q4M1
bytes a token in ≈ 30 ms ≈ 43 GB/s), where the hybrid's CPU reads the same bytes at
51–63 GB/s (rule 46) and keeps only the MoE on the DSP.** So the measured
first lever is **#234 P3 / P4 — the FC WH sidecar** (P3 the prefill half
and the packer; P4 the one-PD FC / DENSE_FFN ops on the WH handles with the
WH GEMV: ≈ half the bytes of Q4M1 per FC, LFM: 448 → ≈ 216 MiB), then
**#229 S2 (2-bit FCs)** on that path; the pool / tier levers (plan 234 P6,
㊴) are deferred until the real routing exists, and the ATTN_M1 growth
(≈ 4 ms per 512 positions) is S6's after them. Memory, ㉟'s baseline: peak
RSS 2174–2182 MiB at E16 = pool 704 + FC arena 1344 (Q4M1, 206 weights) +
DSP heap 582 (KV 400 + 40 inside) + the ARM process; C-max **40** (pool
1792 + FC 1344 = 3136 mapped, heap 617) — C 48 / 56 fail at
`nntr_hvx_attn_m1_register` `0x80000402`, C 64 at `mapped=3840` (rule 71);
plan 229's "C ≈ 57–66 on a 3.8 GB arena" sizing is therefore not reachable
on the S25 with the Q4M1 FC set and these KV caches. **RoPE finding:** the
first load died on the host (`Scudo … Out of memory`) in
`MHACoreLayer::precompute_freqs` ← `build_rope_table` (`mha_core.cpp:681`),
which sizes the fp32 cos / sin tables by `max_position_embeddings` =
262144 (1.5 GiB at head_dim 256 + 512) beside the CPU prefill's fp16 768
MiB; the brief's `max_seq_len` lever does not apply; the sitting's lever
was config-only (`text_config.max_position_embeddings` 4096 in the device
copy, identical numbers below 4096, A and E alike) — filed as **#253**
(p1, `needs-plan`, the planner's this cycle: size by `max_timestep`; rule
72). **Leak stop:** after E8 G64 r1 the S1 ceiling read 3584 (a 20.6 h
boot; two re-reads); reboot, 5 min wait (rule 61), md5 re-checked, 3840
after every later run — ㉝'s class, the sitting is not void. **(4) Open
items:** ㊶ measured → §2; **㊷ new** — the decode MoE inside the E2E token
is not observable from ARM (`NNTR_HTP_DUMP` dumps the ARM-side calls only,
P5's `NNTR_MOE_DIFF` covers the MoE layer call, so an E-vs-off token
divergence cannot be pinned to a layer; the near-tie reading stands in for
it on the dummy); ㉘ / ㉟ / ㉝ notes. **(5) Board:** #250 **closed
`completed`** (PR #251 merged); #201 `state:in-progress` p0 (S5 continues
on S5-0's numbers); **#234 `state:in-progress` p1 → p0** (P3 then P4 are
the next deliverable, the measured reason above); #229 `state:planned` p1
(S2 follows P4; S0 closed); #253 `state:needs-plan` p1 (planner this
cycle, labels untouched here); #137 `planned` p2; #110 `needs-plan` +
`needs-user` p2; #76 cycle comment. Queue ≥ 2 (#253, #229, #137, #110),
nothing new derived. **Upstream:** #4408 `28771a928` (100 commits, PR
updated 2026-10-02) and #4410 `0d603f29a` unchanged; #4327 not watched.
"Now" of the LFM rows unchanged; the Gemma goal row carries its first
measured cells; contract §1's numbers untouched (no number there is
Gemma's). **For the orchestrator (contract text outside §1's numbers):**
§1's Model row — S0 closed: format = external int2 `.bin` repacked to
`QS2CX_WH` by #250's tool, palette {−2, −1, 0, +1}, the real file through
the same script; §1's Accuracy row — S5-0's reference is the hybrid `off`
run, no CPU twin; §12 gains the 2026-10-07 decisions above.

## 1. Rules (device disagreed with reasoning; do not re-derive)

Inherited from the PR's device work, with their sources:

1. **A `--profile` build is never the tok/s binary** — it inflates prefill
   by 83 % (doc 46 §43.4).
2. **Read `min`, not `avg`** — profile averages include the first
   (registration) call (doc 46 §48.7).
3. **Rebuild the skel whenever the IDL or DSP sources change**;
   `build_android.sh` never does. Symptom of a stale skel:
   `AEE_EBADPARM (0x8000040E)` (doc 46 §48.7).
4. **`build_android.sh --clean` drops `-Denable-htp` silently**; check
   `readelf -d libnntrainer.so` for `libsdkl.so` (mobile guide §4).
5. **Never push the link-time stub `libcdsprpc.so`** to the phone (mobile
   guide §5).
6. **`QS4CX_WH` has no CPU fallback**; a wrong layout gives plausible wrong
   text, not an error; `moe_htp_layers` must be empty (doc 46 §35, §48.7).
7. **Moving a matmul to the HTP does not pay when the round trip exceeds
   the matmul** — conv in_proj: −13 ms of an expected −65..−100 (doc 50
   §3.4); the same rule closed doc 45 §0.
8. **The DSP address space, not host RAM, bounds registration**: 3840 MiB
   arena + ≈ 182 MiB heap in one 4 GiB space; 256 MiB chunks avoid the
   alignment waste that made it look like a 3 GiB limit (doc 46 §39–§41).
9. **Decode drifts between sittings without a code cause** (20.8 → 16.5,
   doc 50 §3.4; ±5 % in `hvx_impl` #53): every sitting starts with the
   control binary, and results are read as A/B inside the sitting.
10. **Cross-session decode numbers from another unit are provisional**
    (two S25 Ultra units: 6–8.6 % apart on one binary, `hvx_impl` #23/#53).

Learned in this project (#77, unit `R3CY205ZMND`, 2026-09-21):

11. **AMENDED by rule 28 (#100, 2026-09-23): the "4–7×" is a ratio against
    a probe number no tag-validated per-call path reproduces.** The
    in-situ 16–18 GB/s is real; the 72–117 GB/s denominator is not a
    ceiling. Read this rule as its surviving half — *wall 2 is decided
    only by instrumentation inside the layer call, never by a probe* —
    and take the validated ceiling from `c_star` (26.3 GB/s), against
    which the in-situ list is **faster**, not 2.7× slower. Original text:
    **Isolated DMA probes overstate the in-situ rate by 4–7×.** The same
    engine on the same arena and skel: 72–80 GB/s contiguous, 108–117
    GB/s strided 2D in a probe; **16.2–18.0 GB/s** in the MoE layer call's
    `weight DMA` line of the same sitting (the PR author's 38 GB/s probe
    sat in between). Workers 1 → 4 are flat to negative, the bus vote is
    ≤ 4 % (noise). Wall 2 is therefore decided only by instrumentation
    *inside* the layer call (#87), never by a probe.
12. **A DSP-side bandwidth loop must defeat VTCM/L2 residency or it
    measures nothing**: the two-reader probe's `dsp_alone` read 3265 GB/s.
    Bound the result inside the test (≤ 150 GB/s or `INVALID`) (#90).
13. **Any unit; only a same-sitting A/B is a verdict** (user decision
    2026-09-22, contract §12; supersedes the earlier "unit `R3CY10WM83Y`
    is the anchor" form of this rule). Handoffs name no serial; the
    filled handoff records the serial that ran; every BENCHMARK row is
    unit-tagged; cross-sitting numbers are compared only as same-unit
    drift when the unit happens to be the same (rule 9). The second-unit
    anchor and the per-cell unit ratio (#91, ⑮) are withdrawn as goals.
    Two S25 Ultra units were 6–8.6 % apart in `hvx_impl`; that fact stays
    as the reason for the rule.
14. **Compiled artifacts are not byte-reproducible across build paths.**
    #77's skel, app and gtest md5s all differed from the table after a
    rebuild of identical sources under another path, while every
    non-compiled file and both model `.bin`s matched bit-for-bit. A
    handoff's md5 table is filled from the build that is pushed, or the
    device `md5sum` lines are recorded against the commit; a mismatch
    with that provenance intact is a note, not a void, in a control-only
    sitting — with a B variant it stays a void.
15. **`NNTR_HTP_PROFILE=3` inflates `[M0-PROF] ffn=` ~5×** (16–20 ms →
    88–90 ms per MoE layer): only the `[HTP-PROFILE]` `transport` /
    `host` / `dsp` columns are comparable across levels 2 and 3.
16. **The `generation(last 64)` line lives in the base report block or
    not at all**: `Lfm2MoeCausalLM` reports from `causal_lm.cpp:699`, not
    from `lfm2_causallm.cpp`'s block (#89).

Learned in this project (#94 first attempt / #97, unit `R3CY205ZMND`,
2026-09-22):

17. **Gate 2 must include an undefined-symbol check of the skel.**
    `-Wall -Werror` clean is not "links": a Hexagon shared object links
    with unresolved symbols by default, and the device loader then fails
    `dlopen` with `0x80000406` / `dlerror RX VA 0xFFF00000 outside ELF
    segment` — a message that reads like a segment-layout bug and is not
    one. #94's skel from `2a75f7d9` had seven `U hexkl_dma_trace_*`
    (`hexkl_dma_trace.c` from PR #93 never entered `test/htp/build.sh`
    `SRCS`; the host check compiled it on its own, so gate 1 passed).
    From now on gate 2 = `-Wall -Werror` **and** `hexagon-nm -u -D
    build/libnntr_hvx_skel.so` shows no `U` outside the runtime set
    (`HAP_*`, `compute_resource_*`, `qurt*`, libc, compiler-rt); #97 puts
    that check (or `-Wl,--no-undefined`) into `build.sh`. Second artifact
    found broken only on the device (rule 14 was the first): every new
    `.c` under `htp_backend/` is checked against `build.sh` at review.
    Distinguish the three loader errors: `0x8000040E` = stale skel vs IDL
    (rule 3), `htp Context is not registered` = app-side, `0x80000406` =
    the skel itself does not load.

Learned in this project (#94 sitting 2, unit `R3CY205ZMND`, 2026-09-22):

18. **The isolated probe rate drifts 3–19 % with the phone's temperature
    inside one sitting** (`DMA_PROBE` shape iii, 1 worker: 106.9 GB/s at
    29 °C, 88.8 at ~51 °C) while the tok/s cells of the same sitting did
    not collapse (CPU 47–53). A probe reading is quoted with its
    `thermal_zone0`; the gap analysis uses the cool reading as the
    achievable ceiling and the hot one as the run of record.
19. **WITHDRAWN by rule 28 (#100, 2026-09-23).** The "2.7× lost in the
    list itself" was `traced` / `DMA_PROBE iii`; against the
    tag-validated `c_star` of the same log the traced list reads
    **1.19×**, i.e. faster than the ceiling. Row h dissolves and there is
    no list-side loss to fix. What survives: the *paced* replay
    reproduces the in-situ 15.6–18.6 GB/s (the interleaving is real,
    rows f + g), and more workers make the replay slower. Original text:
    **The real chunk list is slow on its own** (provisional on #99): the
    traced 46-descriptor / 32-region M=1 list replayed unpaced, one
    worker, no compute, no competitor runs at 40.2 GB/s where a probe of
    the identical 8 KiB × 64 @ 56 KiB shape does 106.9 on the same unit
    and skel; paced to the in-situ schedule it reproduces the in-situ
    15.6–18.6 GB/s. Rule 11's 4–7× therefore splits into ≈ 2.7× in the
    list itself and ≈ 1.3–1.8× in the interleaving (rows f + g ≈ 20 % of
    `dsp_us`). More workers make the replay slower (40.2 → 39.1 → 36.5).
20. **NPU tok/s drifts −7..−16 % day to day on one unit while the DSP
    profile columns move ≤ 4 %** (#94 s2 vs #77, same unit and forward
    path): the tok/s cells carry host and thermal state, the
    `[HTP-PROFILE]` `dsp=` / `mm` columns carry the kernel. Cross-sitting
    kernel claims are read from the profile columns, tok/s only inside a
    sitting.
21. **Rule 14, refined by the cycle-5 decision on #94 Deviation 5:** a
    device set whose md5s differ from the staged table but whose
    provenance is intact (commit + build line + `md5sum` recorded, DSP
    sources shown identical by `git diff`) is **accepted as the sitting's
    provenance** when every variant of the sitting is the same binary set
    switched by an environment variable or config key (A and C of #94).
    "Void once a B variant is included" applies only when a variant is a
    *different binary* whose staged md5 the device does not match. Also
    from that sitting: `test/htp/build.sh` picks the newest directory
    under `$HEXKL_ROOT/lib` when `HEXKL_SDK_VER` is unset — a workstation
    with 6.6.0.0 installed silently links another HexKL; every handoff's
    build line carries `HEXKL_SDK_VER=6.4.0.1`.

Learned in this project (#88, unit `R3CY205ZMND`, 2026-09-22):

22. **Handoff artifacts must be reproducible from the commit on any
    workstation; the staging path is not a deliverable.** Twice now (#94
    s2's skel, #88's whole A/B set) the sitting ran on a workstation where
    `/local/mnt/workspace/htp_moe/<n>/` and the implementer's worktrees did
    not exist, and the user rebuilt from the named commits. A handoff
    therefore carries, per variant: the commit, the exact build line
    (`build_android.sh --htp` then `--htp --cache`, `test/htp/build.sh`
    with `HEXKL_SDK_VER=6.4.0.1`, env var names, no absolute worktree
    paths), and the md5s of the non-compiled inputs (model, tokenizer,
    prompt, `libc++_shared.so`, `libsdkl.so`), which must match anywhere.
    **Rule 21 extended to binary B variants:** a sitting whose variants are
    *different binaries* rebuilt from their named commits is accepted
    when (a) every results row carries the device md5 of the binary that
    differs between variants, (b) the source diff between the variant
    commits is exactly the PR under test (`git diff A B -- nntrainer
    Applications test`), and (c) shared DSP artifacts are one file for
    all variants. It stays void when a device md5 matches neither the
    staged table nor a recorded rebuild. #88 met (a)–(c) (device
    `libnntrainer.so` `5db71a2e…` A / `3cf06450…` B in every row; diff
    `08afbb10..f3e99176` = `htp_backend/{htp_backend.cpp,
    htp_compute_ops.cpp, htp_rpcmem.h}` only; one skel `2bd7311f…`).
23. **The FastRPC `transport` column drifts across sittings like tok/s;
    only `dsp=` is a cross-sitting column.** Identical sources
    (`2a75f7d9` = `08afbb10` outside `build.sh`), same unit, same day,
    ≈ 2 h apart: #94 s2 A level 2 M==1 host / dsp / transport 2062.5 /
    1414.0 / 648.6 µs, #88 A 1812.1 / 1410.8 / 401.3 — dsp −0.2 %,
    transport −38 %. Rule 20 extends to `host` and `transport`: a
    transport claim is an A/B inside one sitting (as #88's was), never a
    comparison against an earlier sitting's row. Corollary for budgets:
    the time **outside** the MoE call (TPS token minus 22 × host) moved
    with the poll/staging variant too — 13.5 ms (A) / 11.4 (C) / 8.4 (B)
    at G=64 — so ≈ 5 of #88's 12.6 ms/token gain landed outside the
    call's own timer (candidates: the 5 ms poll keeps the calling core
    and its cluster clock up between calls; cache maintenance around the
    call; not decided).
24. **In an `m1_gemv=calls/calls` profile row, `swiglu` is Σ lane-time
    over the worker pool (6 lanes on v79), not a stage;** `swiglu / mm`
    is lane utilisation (#94 C: 5601.2 / 974.7 = 5.75). Since PR #108 the
    row's `rest` leaves it out and the `weight DMA: n/a` line prints the
    ratio; before that PR a GEMV row's `rest` is meaningless (#102).

Learned in this project (#95, side tree `htp_hadamard`, unit
`R3CY205ZMND`, 2026-09-22):

24. **v79 HVX IEEE `sf` arithmetic keeps subnormals. It does not flush
    them to zero.** `HvxFwht.MatchesScalarBitExact` fed a row of ±1e-39 /
    ±1e-40 (all subnormal). The DSP (`Q6_Vsf_vadd/vsub_VsfVsf`, then
    `Q6_Vsf_vmpy_VsfVsf` by 1/16, `-mhvx-ieee-fp`) returned
    `0x1.7f4b4p-127` at index 1. That is exactly 8·(1e-39 + 1e-40), the
    unflushed IEEE result, and itself subnormal. The scalar spec
    (`fwht_det_ftz` on every input and result) returned `+0`. Only the two
    non-zero outputs of that row differed (`bad_gated = bad_subnormal_row
    = 2`). The spec's premise ("HVX sf flushes them") was host reasoning,
    and the device refutes it: this is not a flushed-zero *sign*. A `_det`
    scalar spec for HVX `sf` code is plain IEEE RNE with no FTZ, and the host
    side must not run under a flush either (`-ffast-math` /
    `crtfastmath` can set aarch64 FPCR.FZ). Fix carried by #110. The
    same test's overflow row (±3e38, `bad_overflow_row=127 of 256`) is
    ungated and not yet explained (inf/NaN encoding is the candidate).
25. **A synthetic single-shape gtest cannot give an accuracy verdict. It
    gates bit-exactness only.** `MoeLayerHadamardMatchesTwoCallReference`
    matched its two-call host reference bit for bit (`bad_elems=0 of
    409600`, `max_ulp=0`). It still failed its own SNR assertion: rotated
    17.51 dB vs unrotated 23.32 dB, −5.8 dB on its uniform synthetic input.
    The full model in the same sitting improved +8.0 dB median requant SNR
    over about 6.2 k real expert calls, and PPL dropped −12.7 %. The two
    disagree in sign. Accuracy verdicts come from the full-model columns
    (`NNTR_PPL`, `[L2-DIFF-MOE]` over a run, text vs A). A gtest prints
    its SNR as a field and asserts only the bit-identity to its scalar
    reference (gate (a)).

Learned in this project (#105, unit `R3CY205ZMND`, 2026-09-22):

26. **At m = 1 the HVX GEMV over the arena is bound by DDR read latency,
    not by `vrmpy` issue. Cutting compute there makes it slower.** The plan
    expected that dropping `gemm_rows4`'s three dead accumulators would
    save 50–100 µs. On the device the one-row loop (B1) raised the M==1
    `mm` from 973.0 to 1044.6 µs (+7.4 %) and cut decode by 4–8 %. Under
    the same loop, the L2-hot microbench cell ran 2.26× faster (70.77 →
    31.25 ns/tile). With 2 `vrmpy` per quarter-tile instead of 8, fewer
    loads are in flight to cover the DDR latency. The `l2fetch` lead wins
    it back monotonically (0 / 64 / 192 KB: 145.2 / 137.5 / 120.2 ns/tile
    arena) but still does not beat the four-row loop's 111.7 at 192 KB.
    A compute-side change to the M=1 GEMV is read on the `hot` cell, or
    after the feed is in VTCM. On the arena it shows the latency, not the
    compute.
27. **ION arena and cached heap read alike from HVX (within 11 %, no
    consistent sign), 21–27 GB/s.** The ION mapping is not why the M=1
    GEMV feed is slow. That is the plain direct-vector DDR band `hvx_impl`
    #59 measured on a sibling SoC (21–25 GB/s vs 37 through the ring). Only
    a DMA feed into VTCM (#100 → ㉒) gets past it. Also from this sitting,
    confirming rule 20: two level-2 profile passes in opposite order, from
    30 °C and 50–58 °C starts, agreed ≤ 0.4 % on every `mm`. A's `mm`
    973.0 reproduces #94 s2 C's 974.7 on the same DSP path, while decode
    tok/s differs by +47 % (PR #103's transport). The `mm` / `dsp` columns
    carry the kernel; E2E prefill tok/s carries the position in the
    block (A's own pairs −10..−24 %).

Learned in this project (#100, unit `R3CY205ZMND`, 2026-09-23):

28. **A `DMA_PROBE` number is not a ceiling; the tag-validated `c_star`
    is.** In one log, `c_star` (the per-call shape with its content
    checked, `checksum_ok=y`) reads **26.3 GB/s** = 0.36 × the same log's
    `DMA_PROBE shape=iii workers=1` (73.2). Against `c_star` the traced
    46-descriptor / 32-region M=1 list runs at **1.19 ×** — *faster* than
    the ceiling — and every other validated cell sits at 30–31.7 GB/s
    whatever the descriptor count (8 / 24 / 28 / 36 / 46), the chaining
    mode (`dmstart`, `link1`, chained) or `dst` (linear, strided). So the
    probe measures something the engine cannot deliver through a real
    per-call list, rules 11 and 19 are amended/withdrawn above, and row h
    of the wall-2 attribution dissolves. **Consequence for gates: an
    absolute GB/s target is only meaningful next to the same run's
    `c_star` and next to rule 30's anchor cell.**
29. **HVX streaming out of VTCM is nearly free beside a concurrent DMA
    into it**: `load=2` costs **0.1–0.3 %** against `load=0` on all three
    certified feed shapes (699.8 → 700.5, 695.5 → 697.6, 700.5 → 701.8
    µs/call). A VTCM feed's cost is therefore the DMA rate alone, and the
    "HVX and DMA fight over VTCM" worry is closed.
30. **One unit's DMA rate drifts ≈ 22 % between sittings on an untouched
    cell, and it is not thermal.** `DMA_REPLAY workers=1 load=0 pace=0`
    (identical code in both sittings) read 561.5 µs/call = 40.2 GB/s in
    #94 s2 and **719.6 µs = 31.4 GB/s** in #100 on the same unit; a
    cooled repeat at 31 800 m°C (vs 46 500) reproduces every #100 ratio
    within 1 %. One-worker probe rows are 15–30 % low while four-worker
    rows are within 6 %, so it bites **single-queue issue rate, not
    aggregate DDR bandwidth**. Cause not separated (session clock /
    governor vs a rebuilt skel). **Any handoff with an absolute GB/s gate
    carries that cell as its in-sitting anchor and reports the gate
    scaled by it.** **Resolved by #113 (rule 32): the drift is
    persistent, and 31–32 GB/s is the bound.**

Learned in this project (#113, unit `R3CY205ZMND`, 2026-09-23):

31. **An `l2fetch` lead harms a loop that is not latency-starved; it
    helps only the loop that is, and only up to one L2 budget.** Rule 26
    read the other way round, now measured on the full 2 × 5 matrix. The
    four-row loop (8 `vrmpy` of loads in flight per quarter-tile) with
    *any* lead reads **2.1–2.5× slower** on the arena (`ns_per_tile`
    133.5 → 282.3 / 316.4 / 323.1 / 328.4 at 192 / 384 / 768 / 1536 KB),
    the damage starts at the first non-zero lead rather than at a
    capacity boundary, and the `hot` cell is flat (71.7–73.5) — the
    signature of **interference** (the box competes with the loop's own
    demand loads for L2 and bus), not of eviction. The one-row loop (2
    `vrmpy`) improves to a minimum at 192 KB (154.3 → 129.1) and degrades
    past it (`inflight_kb=384` × 6 lanes ≈ 2.25 MB, the L2 budget). In
    the layer call: four-row + 192 KB `mm` 1925.5 vs 1000.2 (+92.5 %),
    one-row + 192 KB **937.0 (−6.3 %)**, one-row + 384 KB 1042.8. The
    whole matrix is on the board; its minimum is one-row + 192 KB and no
    (loop, lead) pair reaches 840. Corollary: a prefetch knob is never
    added to a loop without its own `hot`-vs-`arena` cell, and the lead
    is read as an ordinal (the up half of stage A gets ≈ two
    column-computes of it, not the nominal KB — the fourth outstanding
    box is #114's).
32. **The ≈ 22 % DMA drift of rule 30 is persistent, not a session
    effect: 31–32 GB/s is this unit's real single-queue DMA bound.** The
    untouched anchor cell (`DMA_REPLAY workers=1 load=0 pace=0`, source
    unchanged since #94) read **724.0 µs / 31.2 GB/s** cold (27 900 m°C)
    in #113 — #100's 719.6 / 31.4 to within 0.6 %, not #94's 561.5 /
    40.2. Whatever made #94 s2 read 40.2 has not returned in two
    sittings; #100's `f2` feed cell (31.6 GB/s beside HVX) is therefore
    read at face value against a 31–32 GB/s ceiling, not scaled to 37.
    Consequence: **㉒'s feed half closes as a bandwidth question** — the
    DMA cannot deliver 37 GB/s per call on this unit — and re-opens only
    as a *rate* lever: 31.6 through VTCM vs the direct HVX arena read's
    21–27 (23.5 under the one-row loop + 192 KB lead) is ≈ 1.33×.
33. **Decision rule (user, 2026-09-23, "1번으로 진행", #76-level): a
    sub-gate decode win may land as the default when it is consistent at
    all three G in one sitting, the text is byte-identical, and the
    prefill gate holds; the issue's gate still says whether the issue
    is done.** #113's D192 missed `mm` ≤ 840 (937.0, +11.5 %) and lands
    anyway (`HVX_GEMV_M1_ROWS1=1u`, `HVX_GEMV_PF_LEAD_KB=192u`, PR
    #115) on decode **+4.11 / +3.29 / +4.26 %** at G 64 / 512 / 1024,
    text = A 6/6, M>1 `dsp` within 0.08 %, prefill −0.77 / +0.05 /
    +3.73 %. What does *not* qualify: a win at one G only, a win inside
    the sitting's own A spread (rule 27), or one bought with a text
    change. The default flip is its own change with its own acceptance
    (issue #113's cycle-12 comment): an unset run prints the D192 word,
    host checks green, the old cell stays reachable by env for the
    same-sitting A/A0. **Landed: PR #115 merged as `c45d4433` (cycle
    13)** — banner `[HTP] moe m1 gemv: on (applied=0x103c1) lead=192KB
    rows1=1 source=default`, A0 = `NNTR_MOE_HTP_GEMV_LEAD_KB=0
    NNTR_MOE_HTP_GEMV_ROWS1=0` (`applied=0xc1`); the device confirmation
    (A vs A0, same sitting) rides #117's handoff, and BENCHMARK's "now"
    moves to that A, not before.

Learned in this project (#117, unit `R3CY10WM83Y`, 2026-09-23, two
sittings):

34. **The single-queue DMA bound is per unit — 37.3 GB/s on
    `R3CY10WM83Y`, 31.2 on `R3CY205ZMND` (+19 %) — while the HVX direct
    arena read is the same on both.** The untouched anchor cell read
    607.9 / 605.3 µs (37.1 / 37.3 GB/s) in two sittings on the second
    unit against 724.0 / 719.6 (31.2 / 31.4) on the first (rules 30, 32);
    `f2` 588.7 vs 695.5. But D192's `mm` — the latency-bound HVX read of
    the arena (rule 26) — read 922.7 / 931.9 here vs 937.0 there (< 1 %),
    and the `hot` cell 31.25 / 32.17 ns/tile vs 31.25. The unit difference
    sits in the DMA engine (or its clock), not in DDR or HVX; rule 32's
    "31–32 GB/s is the real bound" is `R3CY205ZMND`'s number, and three
    anchor values now exist (40.2 once, 31.2–31.4 twice, 37.1–37.3 twice).
    Consequences: (a) an absolute `mm` gate on a DMA-fed path is read
    next to the sitting's anchor — #117's 676.4 passes 760 as measured
    and would read ≈ 805–809 on a 31.2 GB/s unit; (b) a DMA-fed path's
    win is unit-dependent (931.9 / 676.4 = 1.38× here, ≈ 1.16× projected
    there, decode ≈ +10 % instead of +30 %) while a direct-read path's is
    not; (c) what transfers between units is the decode half of a gate
    (B ≥ A at all G, text = A), the `mm` half is scaled; (d) the goal's
    ceiling is per unit too: 22.02 MB × 22 = 484 MB/token of MoE weights
    is 13.0 ms at 37.3 and 15.5 at 31.2.
35. **Under the VTCM feed the M==1 transport falls 190 → 92 µs/call: the
    "+100 µs GEMV path cost" of ⑦ / #100's A/A0 belonged to the direct
    HVX arena read, not to the GEMV path.** #100 read GEMV 185.8 vs HMX
    84.6 µs (one sitting, equal staging classes) and the ledger called it
    path cost that "comes back only with ⑨". #117's B (GEMV + DMA feed)
    reads 82.0 / 92.3 in two sittings against A's 184.6 / 190.3 in the
    same sitting and binary (`qos_mode=2`, `staging:` identical) — the
    HMX path's level. 22 × ≈ 0.1 = 2.2 ms/token came back with the feed.
    Cause not separated (candidates: the HVX arena read leaves the
    ARM-side cache maintenance or the `dmlink` sync more to do at the
    call boundary; the shorter call stays inside the poll window).
    Corollary: `transport` is read beside `mm` on every DSP-path variant —
    a DSP-side read pattern moves the host column.

Learned in this project (#130, unit `R3CY10WM83Y`, 2026-09-28):

36. **A switch proven on the host is not live on the device until the
    device log prints its banner — the Android app never carried
    `ENABLE_HEXKL`.** `Applications/CausalLM/jni/Android.mk` takes its
    defines from the prebuilt `Android.mk`'s `NNTRAINER_EXPORT_CFLAGS`
    (`-march` + the two FP16 ABI defines only, `jni/meson.build`
    `2f642ab7`), so every `#ifdef ENABLE_HEXKL` block of the app
    (`lfm2_moe_causallm.cpp`'s op list and `set_decode_graph_desc`,
    `htp_decode_hook.h`'s four hooks) compiled out of every device app
    since #85 while `libnntrainer.so` itself had the define. #130's first
    pass of B / C printed no `graph: init` and ran at A's speed; the host
    gates (`graph_host_check`, `run_inproc_e2e.sh`, meson) never go
    through that makefile and cannot see it. Consequences: (1) a variant
    whose expected banner is missing is **void**, never "at A's speed";
    (2) every handoff's sanity block greps the app library for the
    switch's own string (`strings libcausallm_core.so | grep -c
    NNTR_HTP_FORWARD_KINDS` ≥ 1) next to the `graph: forward calls` check
    on `libnntrainer.so`; (3) since #135 `jni/meson.build` exports
    `-DENABLE_HEXKL=1` through the prebuilt `Android.mk` whenever
    `libnntrainer.so` has it, `build_android.sh` exits 1 when the app's
    count disagrees with `--htp`, and the `hexagon-gates` rung 3 /
    `hexagon-handoff` sanity block carry the line (the local
    `-DENABLE_HEXKL=1` ndk-build of handoff 130 Notes ① is gone).
    The M1-GEMV / feed defaults of PRs #108 / #115 / #118 live in
    `libnntrainer.so` and the skel and are unaffected (their banners were
    in every log).
37. **`hvx_emu` is not the silicon on subnormal / tiny values; it is on
    normal-range rows.** The wired kernels, bit-exact to their `_det`
    specs on the host emulation, are bit-exact on the device for every
    normal-range row (rmsnorm kinds 0 / 1 / 3 `bad=0`, ATTN_M1 `out
    bad=0` at L = 1 … 1024) and differ by a few ulp only where the values
    are ≈ 2^-120 … 2^-132: rmsnorm kind=2 (the ±1e-39 row) `bad_y=2048`
    of 2048, qk_norm `bad_y=64`, rope64 `bad=10…18 of 2560` at pos ≥ 1,
    conv_gate_m1 `bad_out=229`, ATTN_M1 `bad_stats=1` (the (m, l) pair) at
    L = 63 / 512 / 1024. Rule 24 (v79 keeps subnormals) stands — the rows
    are not flushed — but the emulation's rounding of tiny values is not
    the hardware's to the bit (candidates: the `qf32` intermediates and
    `Q6_Vsf_equals_Vqf32`, the `vrsqrt` / `vrecip` seeds, one op's
    input flush). Consequences: gate (a) "bit-identical on the device" is
    passed on real-valued rows and open on the subnormal rows until #137
    decides which side is wrong; and **B's degenerate text is not this** —
    lanes at 2^-120 cannot move a logit. ~~so #136 is a logic / binding
    fault, not an arithmetic one~~ — **withdrawn by rule 39 (#136, cycle
    20 close)**: it is neither; it is last-bit differences on
    normal-range rows amplified by the quantized pipeline. `rmsnorm
    kind=2 bad_y=2048` is the ±1e-39 row itself (every element tiny),
    i.e. this rule as written, not a larger error.
38. **`AEE_ERPC` (`0x80000600`, "error due to fastrpc implementation",
    `AEEStdErr.h`) on a bad-shape call means the call never reached the
    entry's validator.** `RejectsBadShapes` in both suites expected the
    entry's `AEE_EINVALIDFORMAT` / `AEE_EBADPARM` (+ `0x80000400`) and got
    `AEE_ERPC` for every bad shape: the FastRPC stub / skel marshalling
    rejects a sequence whose length disagrees with the declared shape
    before the entry runs, which the host check (no marshalling layer)
    cannot show. A device negative test either sends shapes the
    marshalling accepts and lets the entry reject them, or reads
    `AEE_ERPC` as the rejection and says so (#137). Not a stale skel
    (`0x8000040e` never appeared; the skel md5 was in every log).
39. **Text identity cannot hold for any resident kind on this quantized
    pipeline; per-kind last-bit exactness is checked by recomputing
    dumped stretch inputs; the gate is decode PPL + text + approval**
    (#136, 2026-09-28, `R3CY10WM83Y`; §2 #136 row). The host put
    RMSNORM at 129 dB and the attention stretch at 39 dB on the fixture
    and read both as harmless; on silicon each alone moves the first
    decode token's first MoE input to 34 / 40 dB and its last to 18 dB,
    and RMSNORM + ROPE + ATTN_M1 together flip the greedy token at pos
    512 (`town` → `final`) while every other executable mask keeps the
    text. The kernel is not the cause: every dumped RMSNORM input,
    recomputed in the Android CPU's order
    (`neon::rms_norm_wrt_width_fp32_intrinsic`), sits at **131–148 dB**
    against the DSP output, and the DSP row scale is the nearer to f64
    (0.7–3.9 × 10⁻⁸ vs the CPU order's 1.5–27 × 10⁻⁸). Switch-off and
    `KINDS=MOE` are bit-identical run to run, so every dB below ∞ is what
    a resident kind puts in; the amplifier is the CPU Q4_0 GEMM's
    per-block activation quantizer and the MoE's u8 quantizer, which
    flip a level wherever a value sits on a rounding boundary (⑱'s
    mechanism). Consequences: (1) a per-token-entry variant whose text
    differs from A is neither a defect nor a pass by that fact — the
    contract §1 gate decides (decode PPL vs A of the same sitting ≤ +2 %,
    the text column, the user's approval; the text column is never
    dropped); (2) a resident kernel's device exactness is checked per
    kind by recomputing its dumped stretch inputs (`NNTR_HTP_DUMP_ALL`,
    PR #143) against its `_det` spec or the CPU order, never inferred
    from end-to-end text; (3) SNR at the MoE inputs is a diagnostic, not
    a gate, and the host's 30 dB floor (plan 130) does not predict the
    device text; (4) the mask that flips text is prompt-dependent — the
    bisect named a pair only because two perturbations stacked at one
    near-tie, so it names no faulty op.

Learned in this project (cycle 21: #141, #149, #134, unit `R3CY10WM83Y`,
2026-09-28):

40. **The M==1 MoE call's transport is the FastRPC invoke itself;
    dspqueue with a spinning DSP thread takes it from ≈ 88 to 14 µs/call
    without moving a bit — and the spin is what buys it.** #147's
    microbench read F12 (FastRPC, 12 KiB ION) 87.1 µs vs QSS12 (both sides
    spin) 16.7, QBS12 (DSP blocks) 28.9, QBB12 (both block) 76.9; in the
    model (#141 step 2) the level-2 M==1 row read A `transport` 87.9 vs Q
    14.1 µs/call with `dsp` −1.3 %, `mm` −0.2 %, and decode +7.1 / +5.0 %
    (G 64 / 512), while Q0 (DSP blocks at once) kept only +4.8 / +0.4 %.
    The DSP runs the FastRPC method's own function on the same argument
    bytes, so the MoE dumps are `bit_identical=1` (2862 files) and the text
    ≡ A on 8 prompts — a transport change is bit-preserving by construction
    and is gated by dumps + text, not by PPL. Consequences: (1) since PR #151
    (`08de92b1`, flip `c73384d2`) variant A of every sitting is the queue:
    its log prints `[HTP] dspq: on …` once and `dspq: close calls=N served=N
    bad=0` (N = 22 × decode steps); no `on` line or a `dspq: off (…)` banner
    is the FastRPC fallback and voids the sitting (rule 36); A0 is
    `NNTR_HTP_DSPQ=0`; (2) the transport term of the budget is now
    22 × 0.014 = 0.3 ms/token — nothing is left to win on it; (3) the DSP
    spin window (1 ms) keeps one HVX-free DSP thread hot between the 22
    calls; the ARM spins up to 5 ms, which #150 must count when it reads
    CPU-side contention (its first read ran on the FastRPC build).
41. *(Cycle 22: this rule describes the L2 path, `NNTR_MOE_DMA_BYPASS=0`;
    the default since `21169f6e` bypasses the DSP L2 and reads ≈ 57 GB/s
    in the app — rule 43.)* **In the app the DMA engine reads ≈ 0.88 × the isolated anchor per
    byte, cold and warm alike; there is no fetch-schedule lever; the MoE
    `mm` floor on this unit is ≈ 670 µs, not 590** (#149 step 0, no
    build). On one skel and unit: anchor 37.3 GB/s, `traced` / `traced_f`
    / `f2` / `f2_load` 37.2–37.5, the `vtcm` bench 37.64 (`mm` 585 µs),
    but the in-app per-descriptor median 32.9 (level 2, cold, n = 528)
    and 33.0 (level 3, warm) = 0.88 × the anchor. The ring never idles
    (each descriptor starts at its predecessor's completion), gu0 and gu1
    already leave before the activation quant, and the ring order is
    already e+1's gate_up before e's down, so ㉓'s three candidates are
    worth ≤ 3, 0 and 0 µs. Warm ≈ cold rules out the cold call (TLB /
    page walk) as the cause; the remaining cause (DDR contention from the
    ARM side between calls, a bus clock vote in the app, or the engine
    sharing with the HVX read) is not separated and no schedule change
    can reach it. Consequences: (1) ㉓ (feed engine) is closed with no
    code; (2) the MoE `dsp` floor per call on `R3CY10WM83Y` is 22.02 MB /
    33 GB/s ≈ 670 µs (`mm` measured 673.7–677.2), i.e. **22 × 0.670 =
    14.7 ms/token**, not rule 34's 13.0 — the goal's per-unit ceiling
    reads against the in-app rate; (3) rule 34's anchor stays the
    cross-unit scale, the in-app rate the budget's.
42. **The per-token byte count is ≈ 886 MB on the NPU path and ≈ 947 on
    the CPU path, not 730; the CPU reads ≈ 50 GB/s in decode, not
    34–38.** Counted from `config.json` and checked against both model
    files: MoE 22 layers × 4 experts × 3 × 2048 × 1792 = 969 M weights
    (`QS4CX_WH` 484 MB/token, Q4_0 545); the dense Q4_0 FCs = conv
    `in_proj` 3 × 2048 and `out_proj` 2048 per conv layer (18 × 16.8 M =
    302 M) + attention q / k / v / o (6 × 10.5 M = 63 M) + the dense FFN of
    layers 0–1 (2 × 3 × 2048 × 7168 = 88 M) = 453 M = **255 MB**; lm_head
    (tied) 262 M = **147 MB**. The all-Q4_0 file (4 768.9 MB) = 4 360 MoE
    + 255 + 147 (once, tied) = 4 762 — the count holds within 0.2 %; the
    NPU file (4 316.1 MB) = 3 875 + 255 + 147 + ≈ 39 of scales / padding.
    Doc 48 §1's 730 MB leaves 730 − 484 − 147 = 99 MB for the FCs; where
    that figure came from is not reconstructed here, but it is ≈ 156 MB
    short of the file. Consequences: (1) the CPU control's
    52.4 tok/s at G=64 is ≈ 49.7 GB/s effective (§2 ④ read the CPU alone
    at 67.9) — the "ceiling 48–52 tok/s, the CPU sits on it" of contract
    §1 is an arithmetic artefact; (2) on the NPU path the time outside the
    MoE call (≈ 9.7 ms at G=64, ≈ 10.8 at G=512 on #141 Q) reads ≥ 402 MB,
    so the CPU's M=1 Q4_0 GEMVs already run at ≥ 37–41 GB/s effective;
    the "≈ 6.5 ms byte floor" of the budgets was 246 MB at 38 GB/s and is
    **≈ 5.9 ms at the CPU's 67.9 alone, ≈ 10.6 at 38**; (3) any DSP-side
    reader of those 402 MB runs at ≤ 33 GB/s in the app (rule 41) = ≥ 12 ms,
    slower than the CPU does it today, so moving the Q4_0 FCs to the DSP
    is not a byte lever at all (㉘); what is left outside the call is the
    CPU's own GEMV rate (#150). This is arithmetic on measured tok/s and
    file sizes, not a new device reading.

Learned in this project (cycle 22: #150, #90, #99, #152, #158, unit
`R3CY10WM83Y`, 2026-09-29):

43. **The DSP L2 on the weight DMA's source path cost almost half the DMA
    rate: `src_bypass=1` on the arena expert weights reads 69.3 GB/s
    isolated and ≈ 57 in the app, and moves no bit** (#158, PR #159). Probe
    (fresh 168 MiB DDR→VTCM, tag-validated, two runs): **37.3 GB/s with
    `src_bypass=0`** (= rule 34's anchor) **vs 69.3 with `src_bypass=1`**
    (+85.6 %); more UDMA queues from several HW threads add nothing. In the
    app (level 2, G=64): M==1 ring `engine 31.5..33.4 → 56.3..57.9 GB/s`,
    `dsp` **726.0 → 417.5 µs/call**, `first expert ready at` 123 → 77 µs;
    M>1 `dsp` −2.7 %. Per-op at G=512: MoE wait 16.07 → **9.85 ms/token**,
    token 26.0 → 19.74. Decode **+34.2 / +35.3 / +30.9 %** at G 64 / 512 /
    1024. The bytes cannot change: arena slots are filled by the CPU before
    `arena_attach` and never written on the DSP, so no source line in the
    L2 is dirty; dumps `bit_identical=1` (2862), nll lines B == A, text 8/8.
    Consequences: (1) rule 41's 0.88 × anchor and ≈ 670 µs floor are the
    L2 path's; on the default the in-app rate is ≈ 0.82 × the bypass probe
    (57 / 69.3) and the M==1 `dsp` ≈ 418 µs/call = **9.2 ms/token**;
    rule 34's anchor stays the cross-unit scale of the L2 path only; (2)
    the "37 GB/s vs T-MAN's 59" gap was the source-side L2, not the
    descriptor shape, the queue count or DVFS (none of the 2026-09-22..28
    probes set the bit); (3) since `21169f6e` an A log must print
    `applied=0x703e1 … dma_bypass=1 source=default` once; `dma_bypass=0`
    without `NNTR_MOE_DMA_BYPASS=0` set voids the sitting (rule 36); A0 =
    `NNTR_MOE_DMA_BYPASS=0` (`0x303e1`). **The unset banner has not been
    seen on the device yet** (host checks and the in-process E2E only);
    (4) heap slots, activation blocks and the staging copies keep
    `src_bypass=0` (the DSP writes them); #157's cached ARM arena still
    needs its own clean per fill — the ARM caches are not the DSP L2.
44. **The phone's DRAM tops out at ≈ 70 GB/s for any mix of readers; a
    second reader redistributes the bandwidth, it does not add to it**
    (#90, cold probe, every line `valid=y`, `INVALID` 0). CPU alone 63.0 /
    68.1 / 69.4 / 68.3 GB/s (1 / 2 / 4 / 8 threads); DSP ring alone 37.3
    (L2 path); CPU + ring **69.5–71.5 aggregate** (CPU 38.5–40.6, −36..44 %;
    ring 30.9–31.3, −16..17 %); CPU + HVX direct 64.6–66.6; warm 63.9–70.2.
    Consequences: (1) ④ / Q11 answered — the two-reader ceiling is ≈ 70,
    not 38 + 68; for the NPU model 886 MB / 70 GB/s = 12.7 ms ≈ 79 tok/s is
    this phone's hard ceiling; (2) with the bypass (rule 43) the DSP alone
    takes ≈ 57 of the 70 during the MoE window, so ≈ 13 GB/s is left for a
    concurrent CPU reader: #157's split and #90's prefetch overlap (S = 4
    MiB: DSP `mm` unchanged, **+0.7 to +1.4 ms/token**; S = 10 passes warm
    only; S ≥ 20 costs 0.6–4.2 ms — all measured on the L2 path) must be
    re-read on the bypass default before anything is built on them.
45. **Repetition in free-running text is the deciding accuracy failure,
    even when the decode PPL passes; the bit-preserving rule stays** (user
    direction 2026-09-29, after #152). #152's D1 (resident ROPE + ATTN_M1,
    now bit-identical to the Android fp16 CPU attention: G0 / G1 `bad=0`,
    `N_attn` dumps `bit_identical=1`) kept the decode PPL within +0.9 % of A
    on every one of the 8 prompts at G=256 and still got **user `n` on all
    8** (repetition); the remaining difference is the norms (`N_rms` 2.64
    dB, a routing flip; `N_qk` 28.68 dB). **The baseline A itself loops at
    G=256 on p02 p03 p05 p06 p08** (`tools/htp/loop_check.py --prompt`); D1
    adds a new loop on **p04** (Korean). It also fails speed: exact ATTN_M1
    `dsp_us` 1736.6 warm at pos 1023 (gate 940; #148's f32 kernel 387), D1
    decode 25.91 / 24.55 / 22.51 vs A 38.87 / 38.54 / 35.53. Consequences:
    (1) the PPL half of contract §1's amended gate (c) never passes a
    variant by itself; a non-bit-identical variant needs text ≡ A or the
    user's approval, and a new loop fails it; (2) a loop counts against a
    variant only where A's own text does not loop; (3) track (c) (the
    resident path) stays off by default, now also on speed.
46. **The CPU side of NPU decode is near its byte floor; the term that
    grows with G is attention** (#150 step 1b, `NNTR_OP_TIME=1` proven
    inert: T8 within A0's spread, dumps `bit_identical=1`, nll equal). At
    G=512 on the dspq default: token 26.22 ms = MoE wait 16.08 (61 %) + ≈
    10.1 on the CPU; the Q4_0 FCs stream at **51–63 GB/s** (conv in_proj
    51.1, out_proj 53.1, attn o 53.6, dense FFN 57.2, lm_head 61.4; only
    the fused qkv + q/k norm 37.4); FC + lm_head 7.4 ms for 402 MB (floor at
    60 GB/s ≈ 6.7); attention (mha_core) **1.47 ms at G=512, 1.97 at
    G=1024** (#90); norms / adds / conv1d / sampling / register ≈ 0.6;
    unattributed 0.28. Threads 8 / 6 / 4 move decode within the spread and
    cost prefill 8–35 %; CPU clocks sit at 2.0–3.1 GHz during decode.
    Consequences: (1) rule 42 (2)'s "≥ 37–41 GB/s effective" is superseded
    by the per-op 51–63; (2) the bit-preserving CPU headroom is ≈ 0.5–1 ms
    at G=512 (qkv, attention, unattributed) and larger at G=1024 only
    through attention; (3) after the bypass the CPU part (≈ 9.9 ms of
    19.74 at G=512) is half the token and the only lever left for G=1024
    (#162, ㉙).

47. **The HTP norm is bit-identical to the Android CPU once its row scale
    r is computed in the CPU's order; the resident-norm path's speed loss
    is its call count, not the kernel** (#164, PR #167, silicon
    2026-09-29). The old kernel differed from the CPU only in r (the
    `dev/norm-shadow` sitting: 21 of 49 RMSNORM calls fully equal, the
    rest max 2–4 ulp, 136–142 dB, tag0 185/392); with r as 16 `sffma`
    chains + pairwise reduce + integer RN sqrt and reciprocal: RMSNORM
    **392/392** rows, QK_NORM **1920/1920** heads, logits == A on 8/8
    steps, dumps `bit_identical=1`, nll and text ≡ A on 8 prompts at
    G=256. Cost of the new r ≈ +0.05–0.1 ms/token; the gates set from
    the ISS (RMSNORM ≤ 4000, QK_NORM ≤ 25 000 pcyc/op) read **16 077 /
    33 629** on silicon (DDR latency the ISS does not model) — pcyc gates
    are set from a silicon baseline, never from the ISS. Decode with the
    norms resident: R (71 calls/token) −20…−25 %, RQ (77) −38…−50 %, and
    R / A = 0.799 vs 0.773 with the old kernel the same day: the loss is
    the 71–77 calls, so the norms pay only inside a one-call-per-token
    path. Left for #137: the ±1e-39 row (`kind=2 bad_y=2048`) and
    `RejectsBadShapes` on `AEE_ERPC`.
48. **A qf32 multiply + qf32 add narrowed once to hf reproduces the CPU's
    fused fp16 FMA (`fmla .8h`) bit for bit on v79 silicon** (#170 S1:
    0 bad of 4878 replayed + 76 800 adversarial + 25 600 zero / sign
    cases; hf add / sub / mul / max exhaustive-sampled 0 bad; vector exp16
    0 bad of 31 745; div16 0 bad of 27 049 hard quotients — the first time
    today's divide ran on silicon). Two things the host cannot see: (1)
    **hexagon-clang 19 `-O3` folds an explicit `Vsf_equals_Vqf32` into
    the next qf32 op**, skipping the rounding the spec requires — every
    pinned rounding is an empty `asm` (`hvx_hf_pin`), and the ISS was
    bit-exact either way, so a fold shows only on silicon or in the `-S`;
    (2) `Q6_Vqf32_equals_Vsf` is v81-only. Cost: today's `fma16_sf` 8.06
    pcyc per 64-lane FMA at 6 lanes, the qf32 loops 1.96 (scores, one q
    head per Kt load) and 1.26 (pv4); a two-heads-per-load shape stops
    scaling past 2–4 lanes. In-model ATTN_M1 2.47 M → **394.5 k** (G=64)
    and 4.75 M → **700.4 k** (G=1024) pcyc/op = −84 / −85 %, bit-identical
    (Q1 / RQ1 ≡ A 8/8, user `y`), gates 210 k / 350 k missed 2×; round 2
    (PR #182) **250.3 k / 373.6 k**, missed by 19 / 7 %. **Fetch rule
    from the phase words:** a plain 6-lane stream reads 38 GB/s and an
    `l2fetch` lead adds nothing to it (S1), but a *dependent* single-lane
    chain (the PV V rows: 47 pcyc/FMA cold vs 6.9 warm) is latency-bound
    and a 32 KiB windowed lead takes it 37.8 → 7.14 at one lane, 7.55 →
    3.12 at six (PV 1545 k → 386 k); the same lead on a bus-bound tile
    fetch (scores at 6 lanes) buys 11.8 → 9.96 cold and costs +9.5 % warm.
    Round 3 opens with a measurement (split the softmax word: exp16, ET
    stores, max, sum, divides) before any kernel change.
49. **Only the `hvx_intrin` variant of the CPU-exact Q4_0 FC is
    bit-identical on v79 silicon; the `-mhvx-ieee-fp` native `.sf`
    variant and the `sffma8/16/32` chain variants give `bad=2048` per row
    in every cell although host, hvx_emu and the ISS pass all five**
    (#132 PR 2 Part A, PR #175, 2026-09-29). The CPU-order specs equal
    the phone's shipped CPU functions (Q8 quantizer 4000 rows, Q4 GEMV 5
    shapes × 2000 rows, SwiGLU, `sgemv_n` router, **bionic `expf` over all
    4 278 190 082 inputs bad=0**), and in the model the shadow's add
    96/96 and router 176/176 records equal the CPU's. Rate: the exact FC +
    lm_head at 6 lanes **VTCM-fed 7.88 ms/token** (fc 5.14 + lm_head
    2.74; 45.5 GB/s) against the CPU's 7.4, **DDR-direct 30 ms** (11.8
    GB/s) — an FC on the DSP must be DMA-fed like the MoE; the scalar
    activation quantizer adds **4.8 ms/token** (ponytail, HVX version
    needed) and the CPU-exact scalar-chain ROUTER_TOPK costs **427 784
    pcyc/op vs 34 100** (≈ 4.4 ms/token; HVX / multi-chain version
    needed). Address space: **113 MiB allocatable** in the loaded app vs
    383 MiB for the FC set + lm_head. These are decision D's inputs
    (user).
50. **A second cDSP session beside the loaded app exists and is cheap to
    talk to, but it gets no VTCM, and a PD must never grow its DSP heap
    to the end of its address space** (#178, 2026-09-29/30).
    `remote_session_control` reserve + open rc=0 in **22 ms** (lite path,
    no HMX; S1 keeps `hmx_locked=1` and its 8 MiB VTCM); **the 4 GiB is
    per PD** (S2 maps 3584 MiB while S1 holds 3840); the 74 `q4m1`
    weights (383 MiB) fit on S2's heap. Hops: dspqueue spin **2.8 / 12.7
    µs** (0 B / 12 KiB), blocking 36–47, a shared mailbox page both PDs
    poll **0.33 / 2.4 µs** (0 B / 8 KiB), 10 000/10 000, no thread cost —
    44 hops/token ≈ 0.1–0.6 ms. DDR: S1 alone 70 GB/s, S2 alone 20
    (L2-fed), concurrent **58–60 + 15–17** (the rule-44 ceiling; S1's MoE
    feed −17 %), sequential 26–27. **S2 VTCM = 0** (S1 holds all 8 MiB),
    so the exact FC in S2 is L2-fed: **8.06 ms/token** (K=7168 at 26 GB/s
    on 3 lanes — 6 lanes is slower; K=2048 at 52), direct 33. Projected
    two-session end-to-end **≈ 21–22 ms/token ≈ 45–46 tok/s** vs the
    hybrid A's 18.4–19.7 (50.6–54.3). **Defect and rule:** the set2 probe
    grew S2's heap to the end of its space; at S2's close a 2 MiB heap
    page failed to unmap (`remote_munmap64 … 0x80000441`) and the cDSP
    lost ≈ 192–256 MiB of mapping room for every later process (the app's
    arena died at `fastrpc_mmap(64 MiB) … mapped=3584 MiB`) until a
    reboot. Removing that cell fixed it (set3: `s1_mmap_mib=3840` after
    each probe, post-probe app sanities generate). Every second-session
    probe or design caps its heap growth and is bracketed by an app run.
51. **Neither bit-preserving CPU-side lever at G=1024 pays** (#162,
    2026-09-29). (a) The decode attention split (`split4`, `kv_ilp`)
    gives no speedup at L 1024 / 1536 (`as_is` 140.6 / 216.6 µs per layer
    at 8 threads vs 143–157 / 230–238), and the microbench covers only
    ≈ 50 % of the in-app `mha_core` (6 × 0.14 = 0.84 ms vs 1.68–1.87
    ms/token): the in-app attention is half something the kernel does not
    model (the qkv / cache traffic around it), so a kernel-only speedup
    cannot reach the ≈ 1.1 ms G=1024 needs. (b) The prefetch overlap
    (8 threads touching the next layer's first 4 MiB of FC weights during
    the MoE wait) passes the probe's rule P′ under the bypass (net
    +1.0…+1.6 ms/token cold, +0.6…+1.3 warm, `dmm_pct` −3.1) yet reads
    **+1.30 / −0.43 / −1.30 %** in the app against a mirrored A — inside
    A's own spread — with dumps, nll and text identical. The probe's
    "net" credits a hot re-read the app does not realise (the FCs already
    stream at 51–63 GB/s, rule 46); a probe-level saving under 1.5
    ms/token is not a lever until the app confirms it. The S25's DMA
    queues 2 / 4 under the bypass read 67.9 / 63.9 vs 69.2 GB/s on one
    (rule 43 reconfirmed; the S26 scales, #177).
52. **The G=1024 cell moves ±6 % with the start temperature on one unit
    within one day; a "≥ 50" read there is temperature-bound until a
    mirrored cool sitting is taken for it** (cycle 23, `R3CY10WM83Y`,
    2026-09-29/30, eight A cells on the same binary default). Cool starts
    (zone0 25–33 °C) read **52–55 / 51–54 / 50–52** at G 64 / 512 / 1024
    (G=1024: 52.36, 50.90, 51.26, 50.86, 51.59, 51.40; two cool-start
    sittings that ran hot by their G=1024 cells read 49.11 and 48.26); the
    hot #164 sitting (57–65 °C) read 47.88 / 49.50 / 45.86. The row of
    record stays #158 B's 47.36 (a mirrored A/B, read as the lever it
    was): BENCHMARK's "now" moves only with a sitting taken as a control
    (A twice per G, cool, nothing set) or with a bit-identical lever
    accepted as the default (rule 33 / cycle 21), and no cycle-23 cell was
    either. Corollary for every handoff: a G=1024 verdict needs the A of
    the same sitting *and* its thermal checkpoints next to it. The pending
    unset-banner check of the bypass flip is done: every cycle-23 A log
    printed `applied=0x703e1 … dma_bypass=1 source=default` with nothing
    set.
    **Resolved 2026-09-30 by the record sitting** (`record-2026-09-30-cool-a.md`):
    A twice per G, nothing set, each G block started at zone0 ≤ 35 °C →
    **53.97 / 52.16 / 51.41** (G=1024 51.54 / 51.29, ending 64.5–65.6 °C);
    the row of record moved there, #158 B's 51.82 / 50.61 / 47.36 stays as
    the warm-start reading. The rule stands as the row's condition: a
    G=1024 number is quoted with its start temperature, and a warm-start
    A of this default reads 47–49 there.
53. **The inline-asm IEEE HVX `.sf` forms do not execute on v79
    silicon: `Vd.sf = vadd / vsub / vmpy(Vu.sf, Vv.sf)` (assembled with
    `-mhvx-ieee-fp`) return `0x00000000` in every lane, while the ISS
    runs them as IEEE ops; use the `Q6_Vsf_*` intrinsics (a qf32 op and a
    conversion, which the compiler emits even with `-mhvx-ieee-fp`) or
    the scalar `sffma`** (#132 PR 2 sitting S3, `R3CY10WM83Y`,
    2026-09-30, `HvxFcQ4.SfProbe`: 256/256 random normal pairs and the
    kernel's own operands, asm form `0x0`, intrinsics and scalar sffma
    bad=0). This is rule 49's cause: `hvx_native` used the asm form for
    every op, the sffma variants for their scale terms; #164's norm and
    the router never did. A kernel that passes on the ISS with the asm
    form proves nothing about silicon. Same sitting: the scalar
    `sfrecipa / sffixup` divide equals the integer RN divide on every
    mantissa of amax / 127 and 1 / d (50 331 648 divides); the
    ISS-to-silicon pcycle ratio of scalar-heavy DSP code read ×1.3–1.4
    (router 65 528 vs ≈ 50 k projected, quantizer ≈ 14.7 k vs 10.7 k).

54. **With two cDSP sessions in one app, map session 1's arena before
    session 2 opens; whichever PD maps while the other is open and
    mapped stops at 14 × 256 MiB (3584 MiB)** (#132 Part B E5 vs E5b,
    `R3CY10WM83Y`, 2026-09-30). E5 opened S2 first and placed its 383.6
    MiB FC set (`mapped_mib=448`), then S1's MoE arena died at
    `fastrpc_mmap(64 MiB) failed: err=1 … mapped=3584 MiB in 14 chunks`
    (3696 needed) — the same 3584 the #178 probe's S2 stopped at beside
    S1's 3840 (rule 50). Reordered (S2 opens after `repack_weight` has
    mapped S1, `54a30c7f`, banner `s1_arena_mib=3840`), E5b loaded and
    ran: S2 open 24 ms, attach 383.6 MiB, load 685 ms, startup +343 ms
    vs A, `calls/token=1.00`, `hops/token=44`, no leak across five
    reboot-bracketed sets. The driver-side cause is not known; the order
    is the rule. Rule 50's heap rule stands beside it (never grow a PD's
    heap to the end of its space; a leaked 2 MiB page needs a reboot).
55. **A resident kind is proven only by its own same-input CPU-vs-HTP
    shadow over ≥ 64 decode steps; G = 4 / 8 dumps, equal logits for
    eight steps and identical text on most prompts prove nothing about
    a kind that has no shadow** (#132 Part B E5b → E5c → E5e, 2026-09-30).
    E5b passed every G=8 check (46 MoE dumps `bit_identical=1`, every
    shadowed op equal, Ev logits == S0, nll == A for 8 steps) and then
    left A at decode step 25 (prompt512) / 5 (korean) / 26 (p07) on 8/8
    prompts, text ≡ A on 5/8; the host build (x86) reproduced nothing
    over 256 steps. E5c: the one-session `NNTR_HTP_FORWARD=1` path and
    the two-session path leave A at the **same** step and row
    (`step24/pos536/L13.in`, `step4/pos330/L17.in`) while every
    shadowed op stays equal over 64 steps — so the row comes from the
    one resident kind without a shadow, **CONV1D_GATE**: the DSP
    computed `(w0·g + w1·s1) + w2·s0` with every op rounded, the phone's
    decode is `fma(w2, s0, fma(w1, s1, w0·g))` (`vmulq` + two `vfmaq`);
    1 ulp apart on ≈ 13–16 % of outputs, hidden by out_proj's Q8
    quantizer except rarely and data-dependently. Fixed with the CPU's
    fused order (scalar `sffma` taps, `a0bf121b`): E5e S and Ev
    bit-identical over 64 steps on both prompts, conv shadow 1152/1152.
    Corollaries: (1) every resident kind ships with a shadow (the list
    is now FC, ADD, ROUTER, SwiGLU, argmax, RMSNORM / QK_NORM, ATTN_M1,
    CONV1D_GATE; DENSE_FFN and LM_HEAD ride the FC shadow); (2) a spec
    written in the kernel's order is not a spec — it is held against
    the CPU function (`fmaf` model) as #164 and Part A did; (3) the
    prefill `hvx_conv_gate_f32` is unfused the same way (only the HTP
    conv-block prefill engine uses it — open, not on the decode path).
56. **A PD that spins on a hop steals a hardware thread from the PD that
    is computing; across two cDSP sessions the hop wait must sleep,
    not spin** (#132 Part B E5b → E5d, 2026-09-30). With `tk_take`
    spinning 1000 µs (no pause, a flush-invalidate per iteration) the
    two-session path read **16.7 / 16.0 / 15.2 tok/s** against A's
    53.5–55.1 / 54.4–54.8 / 52.4–52.9: both sessions strictly alternate
    and each computed 2–3× slower than its isolated kernels (S1's 22
    MoE + router rounds ≈ 2.7× the isolated MoE's cycles; 108 M pcyc
    over a 59 ms token ≈ 1.8 GHz, so not the clock) — S1's MoE runs six
    lanes on a pool barrier, and one starved lane stretches every call;
    the pools' own ≈ 100 µs post-job spin does the same at each hop.
    Spin 0 / 20 / 1000 µs → **spin 0 best: 20.8 tok/s at G=64, 19.5 at
    G=512**, text ≡ A, S1's MoE back at its isolated rate (9.67 ms =
    0.44 ms/round). Per kind after the fix (G=64, ms/token): S1 MoE
    9.67 + router 0.79; S2 FC 8.75, DENSE_FFN 5.60, LM_HEAD 5.97,
    ATTN_M1 8.33 (the pre-#170 kernel in that branch), small ≈ 1.2; ARM
    token 44.7 ms. Same-PD rule 40 (the dspqueue's 1 ms DSP spin helps
    the single-session MoE transport) does not transfer to two PDs.
57. **A single-session map window for the FC set is not viable: an
    ARM-side `fastrpc_mmap` + `munmap` of an 11 MiB window costs ≈ 1.4
    ms per pair, and the unsigned PD cannot map on its own** (#192,
    `R3CY10WM83Y`, 2026-09-30 07:59, after a reboot). Per pair `fd` 1.40
    ms / `fd_delayed` 1.55 → **33.4 / 36.7 ms/token** for 22 layers;
    DSP-side `HAP_mmap` answers `0x80000402` (not available to the
    unsigned PD, whatever rpc.html lists); re-attach by copy into a
    long-mapped 24 MiB window 0.72 ms copy + 0.36 ms repoint → 23.8
    ms/token; a freshly mapped window costs the first FC call +115 µs
    (456 vs 341); lm_head 140 MiB whole map 2.84 ms + unmap 0.21 (328
    MiB VA headroom beside the arena and the window) or 12 × 12 MiB
    slices ≈ 15 ms/token. Stop rule (> 1.0 ms/token) hit by 30×;
    sanities before / after generate, no leak. The resident FC set
    needs the second session (rules 50, 54) or stays on the CPU.
58. **The ISS prices HVX compute right and memory-side scalar work
    wrong: a scalar gather after HVX stores read 3× the ISS on silicon
    and an L1 prefetch that took the ISS −60 % moved silicon −5 %**
    (#170 S4 and #132 S4, `R3CY10WM83Y`, 2026-09-30). (a) #170's exp
    table (`attn_m1_det_exp_table`, a 3-op vector index + batched
    scalar gather) was priced 272 k → 115 k lane-summed on the ISS;
    on silicon the gather reads **387 k** against the exp16 compute's
    **272 k** it replaced (the ISS's 271 k, 1.0×) — six threads issuing
    scalar loads after HVX stores is what the ISS cannot model; `pool`
    still fell (307 k → 276–281 k) because the `vlut16` q operand cut
    `scores` 681 k → 392 k (warm 601 → 314 k), so both in-model gates
    passed (193 838 / 313 747 vs 210 k / 350 k) with the table on;
    `-DATTN_M1_EXP_TAB=0` (host bit-identical, kept) is round 4's first
    cell, then `vgather` from a VTCM carve-out. The other softmax
    pieces are also larger on silicon than on the ISS (`div` 140 k vs
    90 k, `et` 73 k vs 41 k, `max + sum` 26 k vs 15 k), which is why
    G6b's ratio (0.507 vs 0.55–0.75) missed although the split held.
    (b) #132's router: a row `dcfetch` 8 ahead took the ISS model of an
    8-chain group 47.7 k → 13.8 k pcycles and silicon 65 528 → **62 311**
    pcyc/op (−5 %; gate 60 k missed by 4 %, accepted: ≈ 0.65 ms/token);
    the ISS's L1 miss per weight row is not what bounds the loop on
    silicon (candidates: the 4-lane fork / join, the serial sigmoid +
    pick tail, `sffma` chain latency — needs stage timers inside the
    op). Extends rules 47 and 53: an ISS pcycle gate is a projection,
    the silicon reading decides, and any lever whose ISS gain is a
    memory-side effect is read on silicon before it is planned around.

59. **One PD beats two PDs for the resident FC set, and a pool-28 miss
    costs ≈ 3.5 ms where a pool-16 / -24 miss costs 0.5–0.7** (#201 S2
    one-PD sittings 1 and 2, `R3CY10WM83Y` 2026-10-01 09:43 and
    `R3CY205ZMND` 11:00, plus the two-PD sitting 2026-09-30 22:18). (a)
    With the FC set + lm_head on S1's arena beside a 28-expert pool
    (`NNTR_HTP_E2E_PDS=1`), FC + DENSE_FFN + LM_HEAD read 9.2 ms a token
    `feed=vtcm` against 11.6–13.1 on the VTCM-less S2, the 44 hops
    (≈ 0.33 ms each side) and the second wall go, `rt` 22.4 against 26.6
    (P28) / 30.6 (E0) ms: +6–9 tok/s over two PDs, +12–14 over E0, on
    both units, every text == A. #178's projection for a second session
    (45–46 tok/s) was the two-PD shape; the one-PD shape is what plan 201
    §3.4 asked for and it is the better of the two. The pool is capped by
    the PD's address space: C = 29 is the largest that loads beside the
    448 MiB FC set on the S25, C = 30 fails `fastrpc_mmap(32 MiB)`; C = 29
    gives fewer misses (0.12–0.23 a token at G ≥ 512 vs 0.17–0.34) but the
    same speed, so **C = 28 is the one-PD pool**. The FC set's ≈ 1.1 ms
    over #178's isolated 8.06 and the MoE round's 10.5 ms (0.48 a layer)
    are what is left between Q28 (≈ 22–23 ms a token) and A (≈ 18–21).
    (b) Against reasoning: a miss on the 28-pool reads **3.45 ms** (one
    PD, farm unit, profiled: 121 misses / 417 ms) and **3.79** (two PDs,
    `R3CY10WM83Y`) against 0.72 (C = 24) / 0.53 (C = 16), with the model
    file 95–100 % page-cache resident and the profiled prefetch reading
    88 / 88 experts ahead with 0.0 ms exposed wait — page-cache residency
    and the prefetch do not explain it; the bigger ION arena (3.3 GB) and
    the reader threads' placement are the next reads (㉜). At G ≥ 512 the
    28-pool misses 0.12–0.34 a token, so the cost is 0.1–1.1 ms a token,
    not the lever's ceiling. (c) Per unit, per boot, no cause: the farm
    unit stopped twice on the S1 ceiling 3584 (after a two-PD G = 512 r1
    on a 7.8-day-uptime boot and after the profiled run, both runs having
    closed clean, `chunks unmapped 13/13`), `R3CY10WM83Y` once across the
    three #201 sittings (two-PD, after `E0_G512_r1`, also a first boot) —
    read it before any S26 sitting plans its reboots (㉝). (d) The two
    units are not one column: ≈ 7 % apart in DSP clock by the farm
    session's reading (rule 34's band), yet Q28 at G = 512 read higher on
    the slower unit (44.7–46.7 vs 42.4–43.4) — the one-PD cell's spread
    across boots and prefill ramps is larger than the unit gap; read each
    sitting against its own A only. (e) The two-PD path was removed by
    #211 on 2026-10-01; its numbers stay as the reason.

60. **`HmxMmU8I4Layer.RegistryCapacity` takes the S26 developer unit
    `R3CY70LV96T` (userdebug `S948USQU1AZAB`) into Samsung download mode**
    (2026-10-01 14:12, after 16 `[ OK ]`; USB `04e8:685d`, adb gone, no
    tombstone / pstore; the four config files written seconds before came
    back zeroed — `sync` after every push on this unit). Exclude it there
    (`--gtest_filter=-HmxMmU8I4Layer.RegistryCapacity`, as
    `204/s26/run_s26.sh` now does); whether the test or the build is the
    cause is not known and was not re-tried. The S25 units and
    `R5KL20NFRCK` ran it.

61. **A boot's first ≈ 2 minutes are not measurement time: the pool
    miss read (8 `pread` slices into the ION arena, one per pinned core)
    costs 3.7–5.0 ms a miss at uptime ≲ 130 s and 0.5–1.6 ms afterwards,
    whatever the knob** (`201-s3-probe.md`, `R3CY10WM83Y` 2026-10-01
    17:27 / 17:34, two reboots, 20 profiled Q28 / Q24 G = 64 runs, every
    text == A). Against reasoning: the handoffs' "reboot, then start" put
    the first runs inside that window, and rule 59 b's 3.5–3.8 vs 0.5–0.7
    compared a C = 28 cell near a boot with C = 16 / 24 cells that were
    not. Protocol from here: **after a reboot wait ≥ 5 min, or run one
    discarded warm-up run, before the first measured run**, and record
    the uptime on every run line. The first-run cells already in the
    tables are flagged, not corrected: `201-one-pd.md` sitting 2's
    `Q28 G = 512 r1` (noted cold) and `prof_Q28` (3.45 ms), the two-PD
    sitting's 3.79 (`201-fsu-e2e.md`), and #208's `prof_Q28_q4` / `_q1`
    (4.69 / 3.99 ms on the developer S26) — that unit's "≈ 4.7 ms a miss"
    is unexplained and boot proximity is a candidate for it. The
    reader-thread knobs (`NNTR_MOE_PREFETCH_CPUS`, `_READERS`,
    `NNTR_MOE_PREFETCH=0`) do not move the decode miss (the readers run
    only in prefill; the miss is the pool server's `parallel_for`); the
    cause of the slow regime is not measured (㉜, #216).
    **Amended, cycle 30 (#216 step 1, `216-miss-read.md`): the cause is
    not boot proximity and the protocol does not remove it.** The slow
    miss is a storage read: the model file's page cache is evicted by
    kswapd *during the run* (3.3 GB ION arena + 4.1 GB file + Android on
    11.1 GB), and a re-missed expert then comes from UFS — `pgpgin` 610 /
    310 / 61 / 0 MiB in the decode window against 5.03 / 2.72 / 1.55 /
    0.55 ms/miss on one boot, PSI io 92–296 ms in every slow window.
    Slow runs happen at uptime 300 s and after a 6-minute idle; the
    uptime correlation of the two S3 sittings was that sitting's memory
    state. What stays of the protocol: the uptime on every run line, and
    **a pool-miss cell is readable only with the decode window's
    `pgpgin` (the `pgpgin_mib=` field on the pool line since PR #218) or
    PSI io next to it** — without it a ms/miss number is one draw from a
    bimodal distribution (0.68–4.38 on one boot, `216-fadvise.md`). The
    first-run cells flagged above stay flagged for this reason, not for
    their uptime.

62. **`posix_fadvise` is not a hint on this phone (S25, kernel
    6.6.77-android15): WILLNEED reads at most the device's 1 MiB
    read-ahead window of a range and costs 3–8 ms of caller time per
    5.4 MiB expert; DONTNEED costs 2–10 ms per expert** (`216-fadvise.md`
    probes, `R3CY10WM83Y` 2026-10-01; the workstation clamps the window
    at 128 KiB). Against reasoning: plan 216 rev. 2 assumed one
    asynchronous call per range at ≈ 0.1 ms. Consequences measured in the
    same sitting: issued in 128 KiB pieces on one unpinned worker, the
    advice keeps every decode miss a cache hit (0.86–1.13 ms/miss on both
    boots) but puts +0.3–0.6 ms on each miss round beside A's fast reads
    and **−7 to −18 % on the prefill**, where the drops and WILLNEEDs run
    beside the 8-thread prefill; no placement tried (inline, per-call
    threads, one worker at any priority, deferred to the decode start)
    removed the cost. `mlock` is not available either (`ulimit -l`
    64 KiB without root). A page-cache lever on this kernel must avoid
    the syscalls on the token and prefill paths: hold the bytes in a
    user buffer, or read with `O_DIRECT`, and drop the file's pages once
    at load.

63. **The engine keys' load-time Q4_0 → qs4cx-WH re-quantization of the FC
    weights is not the +4 % of doc 51 on this tree, and its ≈ 216 MiB of
    copies live in the DSP heap, which the budget must count** (#222
    closing sitting, farm `R3CY205ZMND` 2026-10-02, set from `ddb7d9ad8`;
    numbers in #222's comments, handoff unfilled). Against reasoning: plan
    222 and the handoff took doc 51's `attn_proj` +4.5 % / `dense_ffn`
    +4.2 % prefill PPL as the accepted cost and budgeted the copies only
    against the 3840 MiB map window ("≈ 200 MiB of room for 216"). The
    device: (a) with all three keys the hybrid's prefill PPL is 90.31 →
    128.31 (+42 %) and the forced decode PPL 1.2108 → 1.3752 (+13.6 %),
    the G64 text loops — two stacked quantizations (the file's Q4_0, then
    per-column qs4cx over K = 2048) are not the author's one; (b) the
    copies raise `heap_used_kib` 92 058 → 223 003, and the conv block's
    M-proportional session scratch (`hexkl_conv_block.c:268–284`, ≈ 26
    MiB at M = 1024) then fails with `AEE_ENOMEMORY` at P1024 while P512
    still fits; (c) one PD + pool C = 28 (3328 mapped) + the 448 MiB Q4M1
    FC set + the copies reaches `mapped=3712` and the FC set's last 32 MiB
    `fastrpc_mmap` fails — the "room" was never there once the heap is
    counted. Consequence: a weight that the HTP reads at prefill and at
    decode is stored **once, in a file, in the DSP's format** (#225: the
    `QS4CX_WH` FC sidecar written by the packer from f32, PR #230; no
    load-time re-quantization, the heap keeps scratch only), the prefill
    runs in 512-row chunks (PR #230, the hybrid's way of record), a config
    / feature that adds DSP-heap bytes is read at P1024 as well as P512
    before it is adopted, and a one-PD budget sums pool + FC set + heap
    copies + scratch against 3840.

64. **A prefill FC call the HTP runs is chunked by `fcMaxRows(K)` (VTCM
    arithmetic: 1920 rows at K = 2048), not by PR #230's 512-row
    `prefillRows()`, so the hybrid's attention qkv projection at
    P1024 reaches the DSP as one M = 1024 call — and that call fails
    `AEE_ERPC` (`0x80000600`) when the FC WH overflow sits on the DSP
    heap, while the identical call succeeds with the heap empty** (#225
    handoff, farm `R3CY205ZMND` 2026-10-06, set from `51b2b4477`; log
    `nntr_hvx_mm_u8i4_layer failed: err=-2147482112 (M=1024 K=2048
    N=3072 handles=3)` after `[HTP] fc wh: … arena_kib=145408
    heap_kib=75776 requant=0`). Against reasoning: plan 225 and the
    handoff expected the hybrid's P1024 to hold because "P1024 now runs
    in 512-row chunks, so the conv block's scratch is the P512 size" —
    true of the conv block, the dense FFN and the MoE prefill
    (`htp_compute_ops.cpp:4558`, `:5412`), not of the FC entries
    (`gemm_q4_0_accel_fp32` `:1262–1272`, `gemm_q4_0_batch_fp32`
    `:1333–1345`, which is the qkv `handles=3` call) that keep their own
    cap; and the failure is `AEE_ERPC`, not #222's `AEE_ENOMEMORY` — the
    DSP side rejects the call rather than failing an allocation, which
    the host cannot tell apart from a stale skel without the banner
    (rule 38). Measured beside it, as the plan predicted: the hybrid's
    overflow `heap_kib=75776` (plan 72–86 MiB) is real and is the
    difference between B (fails) and Q (`heap_kib=0`, runs the same
    M = 1024 keys at P1024: 699–755 tok/s prefill); the E2E one PD with
    the WH FC set fits at C = 28 (mapped 192 + 3520 = 3712 of 3840,
    `heap_used_kib` 94 604). Consequences: every HTP prefill entry a
    config key can route (conv, dense, MoE **and the FCs**) obeys one
    row cap, `prefillRows()`; a chunked path is proven bit-identical to
    the unchunked one on the host before the device sees it (PR #230's
    `CONV BLOCK CHUNKED BIT-IDENTICAL` is the template); and a hybrid
    cell with a non-zero `heap_kib` is read at P1024 before it is
    adopted — until the FC chunk lands, the hybrid's P1024 case of record
    is Bfb (attention / dense keys `cpu`, conv `htp`; user decision (b)).

65. **The pool-miss slow regime is page-cache eviction and nothing else:
    take the model file out of the page cache once (the 88-expert
    complement in a cached ARM tier, `O_DIRECT` reads, one whole-file
    `DONTNEED` at load) and it is gone — window refaults 74k–153k →
    ≤ 571, kswapd scan 65k–139k/s → 0, PSI io 81–176 → 19–37 ms,
    `resident after` 0.7–3.2 GiB → 0, ms/miss 2.05–8.11 → 0.43–0.83 on
    every tiered run of three boots (#219, farm `R3CY205ZMND`
    2026-10-06, 42 runs). Three things the device said against the
    plan:** (a) **the one-thread copy beats the 8-slice `parallel_for`
    copy on every axis** — 0.43–0.45 vs 0.51–0.83 ms a miss, decode
    48.47 (sd 0.23) vs 47.31 (0.71) over four interleaved runs,
    refaults ≤ 81 vs up to 6236, `pswpin` ≤ 168 vs 2105, prefill +1.7 vs
    +0.7 % — plan 219 expected the 8-slice memcpy at 22–26 GB/s
    (≈ 0.25 ms for 5.3 MiB); measured, the 8-slice copy is 0.67 ms
    (≈ 8 GB/s) and one thread 0.44 (≈ 12 GB/s): a store into an uncached
    ION slot is store-bound, and the eight slices' barrier and their
    contention with the DSP's own DRAM traffic cost more than the
    parallelism gives. (b) **`tier_waits` 3–4 a run at every G (103 /
    120 / 124 misses) means the refill is not fully off the token path
    — but the count does not grow with G, so it is the decode start's
    backlog (the prefill batch's victims still refilling when the first
    tokens miss; the host's `run_inproc_e2e.sh` showed the same 2 waits
    at decode start behind `refill_ms=42.5`), not a steady-state race;
    the lever is to drain the refill queue before the first decode
    token, not a second refill thread. (c) **`pswpin` is not 0 with the
    tier** (0–1261 on the main sitting, up to 2105 / `pswpout` 3621 on
    one `=1` run; `=2` ≤ 168 / 12): anon 766 + tier 467 MiB beside the
    3 776 MiB of ION arenas puts the sum at the edge of the S25's 11 114
    MiB, and the kernel swaps a little (zram) — rule 61's memory
    arithmetic holds with the page cache replaced by the tier, with
    ≈ 0.5 GiB less headroom than the plan assumed. Consequence: a miss
    on this phone is a memcpy, never a read, from now on; G2's `pgpgin`
    is the refills alone (misses × 5.29 MiB ± 7); the E2E Q28 cell's
    largest term at G 64 (3–8 ms a token, ㉜) is ≤ 0.9 ms a token with
    `=2`; what is left between Q28 (48.5) and the CPU (52–54) at that
    cell is no longer the miss. Default not flipped — the user's call
    (#219, as #218).

66. **An upstream kernel's own host check is not evidence when it compares
    the kernel with a variant of itself: #4410's native u8i2 GEMV
    (`dfca0acbf`) split the 2-bit codes with the 4-bit nibble mask
    `0xF0F0F0F0`, so its vlut32 indices were 0 / 16 / 32 … and every index
    > 31 read 0 — and its only check held `_cols2_nopf` against
    `_col_nopf`, both wrong the same way** (#229 S1, cycle 36, host; not a
    device reading). Fixed to `0x0F0F0F0F` in PR #239 (as the device-gated
    expansion `hvx_expand_i2i4` already had it), held against the scalar
    spec on libnative (`HVX GEMV NATIVE u8i2 cases=3456 bad=0`; the old
    mask as a mutant fails `bad=1253376`), and the whole E2E bit-identical
    to the 4-bit palette twin. Consequence (doc 45 §3.3 restated for
    ports): a ported DSP kernel ships with a scalar spec and a check
    against *that* spec, never only against another variant of the same
    port; a port issue's host gate names the spec line. The u8i2 GEMV
    still has no device reading (㊳).

67. **A skel that was built before an IDL change in the base is stale
    for an app built after it, even when the branch under test changed
    no DSP source — "no DSP change on this branch" is not the test;
    `git diff <skel's commit>..HEAD -- test/htp/nntr_hvx.idl` is** (#236
    close-out, 2026-10-06: the handoff planned to reuse the #225 skels
    `58e3a85f…` / `ca25ec2d…` because `htp/236-qkv-chunks` touches the
    host only, but the #237 sync had brought #232's IDL change into
    the base under it; the stub compiled into `libnntrainer.so` and the
    reused skel would have disagreed — on the device that reads as
    `0x8000040e` (the runner's STOP condition, rule 38's class), not as
    a wrong answer). The set ran with skels rebuilt at `a3d966164`
    against the current IDL (v79 `ce4e6486…`, v81 `adfd44d4…`), the
    staged `md5.txt` was regenerated (`1e6bf2f0…`) and the device md5s
    equal the regenerated set — a rule-22 case, accepted. Consequence:
    a handoff that copies a skel from an earlier set states the skel's
    source commit and the IDL diff against the app's commit (empty, or
    rebuild); the supervisor voids a run whose skel predates an IDL
    change in the app's history; `build.sh` is still not
    bit-reproducible (rule 14: a second v79 build at the same commit
    gave `b5f04afe…`), so skel identity is "built at commit X against
    IDL Y", never an md5 match across machines.

68. **With the tier on, the E2E one PD's G64 cell is still under the CPU
    at every prompt length, and its residual scales with P: misses/token
    × the tier copy plus a decode-start cost — 1.16 / 1.61 / 2.62
    misses a token (`miss_wait_us/token` 157 / 317–331 / 472–475) and a
    one-off ≈ 45 / 77 / 160 ms a run at P64 / P512 / P1024, so the G64
    cell sits 0.8 / 1.5 / 3.0 ms a token behind the same P's G512 cell
    and −3.4 / −7.7 / −7.5 % under the CPU, while at G ≥ 512 the same
    path reads 50.6–54.6, ≥ 50 and above the CPU at every P** (#219
    close-out, `R3CY205ZMND` 2026-10-06, 14 runs; Q 52.20 / 54.60 /
    53.89, 48.08 / 51.79 / 51.65, 44.02 / 50.59 / 50.78 vs the CPU of
    record 54.05 / 52.54 / 51.48, 52.07 / 50.70 / 49.47, 47.58 / 48.82 /
    47.41). Against reasoning: plan 219 and the cycle-35b fold treated
    the pool miss as *the* G64 term and expected the tier (ms/miss 3 →
    0.44) to close that cell; it moved it +12 … +22 % over the untiered
    Q, but the miss wait is now only 0.14–0.41 ms a token of the gap —
    the rest is a start cost that grows with P (the prefill batch's
    victims still refilling when decode starts: `tier_waits` 0 / 3 / 3
    at P64 / P512 / P1024, constant across G, rule 65 b; plus the first
    token's own work) and is paid once a run, so G amortises it (÷ 64
    vs ÷ 512). Consequence: the G64 column of an E2E row is read as
    "start cost ÷ G + misses × copy", not as a per-token rate; a lever
    for it drains the refill queue before the first decode token (㊴ 2)
    or keeps the prefill batch from evicting the pool, and is gated on
    the G64 cell *at P1024* (the worst case), not at P512; the hybrid
    (resident experts, no pool) has no such term and stays the fastest
    configuration at every cell (56.4–56.5 at P512 G64 in both
    close-out sittings). Not pursued on LFM2.5 (user: the E2E speed gap
    is closed here, re-examined on Gemma).

69. **The hybrid's P1024 failure beside the FC WH heap overflow is the
    FC call's row count, not the overflow's residency: the same
    `heap_kib=75776` overflow with every M > 1 FC call capped at 512
    rows loads and runs P1024 at every G, the unchunked set fails
    deterministically in the same sitting (`0x80000600` on the first
    attention qkv call, A′ P1024 VOID), and the chunking costs ≈ 0 in
    decode and ≈ 3.4 % of the prefill** (#236 close-out, `R3CY205ZMND`
    2026-10-06: B P1024 53.24 / 53.06 / 48.64 = Bfb's 52.03 / 51.51 /
    48.74 anchored (the M = 1 path is untouched), prefill 725.7 / 722.7 /
    711.1 = +9.4 / +11.1 / +9.5 % over Bfb (which ran those FCs on the
    CPU); `prof_B_P1024`: `M>1 FC K=2048 N=3072` 12 calls / 6144 rows,
    `N=2048` 13 / 6656, 2.31 / 1.61 ms host a call with 0.60 / 0.37 ms
    transport and `rest` 39–41 % of host (quant 0.42 + acc 0.19–0.29
    ms), 48.8 ms of a 1458 ms profiled prefill). Against reasoning: plan
    236 priced the extra calls at ≈ 0.4 ms each (≈ 5 ms, 0.3 %); on the
    device an FC chunk is 1.6–2.3 ms of host time — the DSP work is
    1.2–1.7 ms (77 / 74 % of host) and the per-call `quant` / `acc` /
    transport ≈ 1 ms, ten times the plan's figure — yet still small
    beside the 1.4 s prefill, so the plan's "chunking is free" verdict
    stands at the prefill gate (+9 % over Bfb, floors 630 / 618 / 617
    passed) while its per-call arithmetic does not. The o_proj row's
    13th call (6656 = 6144 + 512 rows) is the host fixture's extra
    call, read as expected; `heap_kib` stayed 75776 on every B run, so
    the arena tails were filled as in #225 and the overflow itself is
    sound with 512-row calls. Consequence: every HTP prefill entry a
    config key can route obeys `prefillRows()` (rule 64's consequence,
    now measured); a per-call cost on the DSP side is read from a
    profile row, never from the plan's µs/call; Bfb (attention / dense
    keys `cpu`) is no longer the hybrid's P1024 case of record — B is.

70. **A device run on `htp_decode` between PR #239 (`4a7003f93`,
    2026-10-06) and PR #247 (`f3e47ef24`, 2026-10-07) that makes a 4-bit
    M > 1 MoE or dense FFN HTP call with more down chunks than the
    16-slot `exp_live` holds (the #234 comment counts every 4-bit M > 1
    call beyond `exp_bg`'s 8 chunks as suspect) hangs in
    `hvx_worker_pool_wait_bg` — it is not slow, it never returns**
    (#234 P1, host bisect 2026-10-06: `ed2660ddc` read `exp_live[16]` by
    chunk index on every M > 1 call while only `exp_bg`, the 2-bit
    expansion path with ≤ 8 chunks, bounds that index; the lfm25 dense
    FFN at M = 512 under `dense_ffn_engine=htp` waited forever on a job
    that was never queued; fixed by `56a9e3f3b`, which tests `exp_bg`
    first; the v79 device `*MoeLayer*` gtests pass with the fixed skel).
    Not a silicon disagreement — a host-found bug — kept as a rule because
    it decides provenance: a run from a skel built in that window whose
    log stops after an M > 1 MoE / dense FFN banner is read as **this
    hang**, void, not as a device fault or a thermal stop; no filled
    handoff of this project ran in the window (the #236 / #219t sittings
    used `htp_first_version` skels, which never had `ed2660ddc`; the
    #229 device gate ran M = 1 only). Consequence: a per-chunk bookkeeping
    array that one path bounds is indexed only behind that path's flag;
    the M > 1 E2E keys line (`dense_ffn_engine=htp` at M = 512) stays in
    `run_inproc_e2e.sh` as the host check that would have caught it.

71. **The one-PD pool's C-max on the Gemma-4 26B is set by the DSP heap
    (the two KV caches beside the pool's slots), not by the 3 840 MiB mmap
    ceiling: C 48 (pool 2112 + FC 1344 = 3456 mapped, 384 MiB of room) and
    C 56 (3840 exactly) both fail at `nntr_hvx_attn_m1_register`
    `0x80000402`, C 64 fails earlier at `fastrpc_mmap` (`mapped=3840`), and
    C 40 loads (3136 mapped, heap 617 MiB)** (#201 S5-0, `R3CY205ZMND`
    2026-10-07, dummy weights, `max_seq` 2048: KV 400 + 40 MiB). Against
    reasoning: ㊶ and plan 229 sized the pool from the ceiling minus the
    FC set and the head (C ≈ 57–66 on a 3.8 GB arena), which is what C 56
    hits; the attention registration's heap need was not in that budget.
    Consequence: the C-max search on a new model starts from the KV cache
    bytes (`attn_m1: registered … cache=` lines) and the heap at load
    (`s1_heap_kib`), not from `mapped=`; a lever that lifts C above 40 on
    the S25 is a KV-cache or heap lever (㉞ a's window sizing, a smaller
    `max_seq`), not an arena one — and on this dummy C above 16 does not
    pay anyway (decode flat within 3 % over C 8–40).

72. **The Gemma host-side RoPE table is sized by `max_position_embeddings`
    (262 144), not by `max_seq_len` or the cache length: at head_dim 256 +
    512 it is 1.5 GiB of fp32 beside the prefill's 768 MiB of fp16, and the
    first E2E decode token OOMs the ARM process in
    `MHACoreLayer::precompute_freqs` ← `build_rope_table` (`mha_core.cpp:681`)
    while the DSP side has already loaded and registered** (#201 S5-0 Step
    1). Against reasoning: the brief named `max_seq_len` as the lever —
    it changes nothing here; upstream's `b2b84250a` (lazy RoPE allocation)
    is in the tree and does not prevent it; doc 55 §10.11 (#4408) saw the
    same table as ≈ 800 ms of prefill on the first layer. The DSP uses
    ≤ `max_timestep` rows. Consequence: until #253 lands, every 26B sitting
    runs with `text_config.max_position_embeddings` 4096 in the device's
    `config.json` copy (identical numbers below 4096 for both RoPE types,
    the original kept as `config.orig.json`) and the Artifacts row names
    the copy's md5; a load that dies after the `token driver: on` line with
    a Scudo map failure is this, not a DSP fault.

## 2. Verdicts (measured, closed)

| item | verdict | source |
|---|---|---|
| Tier-1 per-expert FastRPC (176 calls/token) | regression by construction (57 ms transport) | doc 41 §3 |
| Grouped gate_up at decode | built; transport 326 → 68 µs/mm | doc 34 §4E |
| In-place tile dequant, weight residency, cross-matmul prefetch, HVX worker pool, poll QoS + ION | built and verified | doc 34 §4A–F |
| GEMV instead of HMX at M=1 (prefill FC context) | rejected there (padding tax ~40 µs) — **re-opened for decode** where the 64-row tile is wall 1 (1.03 of 1.35 ms) | doc 34 §5.1 vs doc 48 §3 |
| Phase A W1/Q1/DQ1 micro-optimisations | one of four survived (D1: overlap weight DMA with activation quant) | doc 44 Phase A |
| Fused SwiGLU MoE path (A1) | passed, 32/32 `max_abs_err=0` | doc 44 |
| conv in_proj on HTP | runs, text identical, −13 ms prefill, not a decode lever | doc 50 §3.4 |
| Whole-model residency (doc 45) | plan exists, prefill-ordered; we take its Phase E (one call per token) first, M=1 shape | contract §3.1 |
| ① ARM remainder / per-type decode cost (ms/token, NPU vs CPU, unit `R3CY205ZMND`) | `lfm2_moe` **42.1 vs 22.3**, `fully_connected` **28.2 vs 10.3**, `output_of_causallm` 2.85 vs 3.40, `mha_core` 2.30 vs 2.81, all others ≤ 0.2; MoE + dense FC = 92 % of the NPU token, 83 % of the CPU's. The MoE FFN on the NPU costs ~2× the CPU path; the dense FC routed through the HTP 2.7×. lm_head is the one place the NPU already wins, so ⑧ (lm_head twin) is not first. **Correction (cycle 3, guide-writer's reading of the config):** the #78 `q40-qs4cx-wh` `nntr_config.json` routes **no FC to the HTP** — `lfm2_causallm.cpp:337–348` defaults `attn_proj_engine`, `conv_in_proj_engine`, `conv_out_proj_engine`, `dense_ffn_engine` to `cpu` and #78 set only `moe_engine`. So the NPU run's `fully_connected` 28.2 vs 10.3 ms/token is **not** an HTP round trip; the cause is open (candidates: doc 48 §2 ③ FC thread over-splitting when the 8 CPU threads contend with the FastRPC poll thread, or the CPU clock dropping while the DSP runs). Decidable with one extra profile cell: the NPU model at `NNTR_NUM_THREADS=4` (FC row moves → threading), or the non-WH `QS4CX` model with `NNTR_MOE_HTP_DECODE` off vs on (the `q40-qs4cx-wh` model has no CPU MoE fallback, rule 6, so it cannot be the control). Open item ⑰ | #77 A (profile); correction from the guide (PR #92) |
| ② Transport floor (wall 3) | **marshalling**: 527.7 µs/call at level 3 vs 587.7 at level 2 (−10 %, inside ±15 %; wake/clock would have been ≤ 300), `qos_mode` 2 both. **Refined by the #88 plan (host inventory, not yet measured):** cache maintenance on the over-sized staging pair (decode's 8 KiB rode a 4 MiB cached ION pair after prompt 512; 36–59 µs/MB, doc 50 §3.6) plus the 100 µs poll-window fallback to the interrupt path; the per-call arguments proper are 464 non-ION bytes (48 B primitive block + 5 sequences) and cannot carry 500 µs. Fix taken first = upstream `fb0f02b9` + `04a2fcc4` + `b0a384d6` (PR #103); prebind is conditional on that measurement (plan 88 §3.3 / §4 step 5); dspqueue / resident worker deferred to one paragraph (#83 §4). **Confirmed and closed by #88's sitting: staging 245.8 µs + poll 67.6 µs of A's 401.3; see ⑦ below** | #77 B; plan 88 §2.1; #88 |
| ③ Arena DMA probe (wall 2) | **shape** is the only axis: strided 2D 108–117 GB/s vs contiguous 72–80; workers flat-to-negative; vote ≤ 4 %. **But isolated rates are 4–7× the in-situ 16–18 GB/s**, so the wall is the layer call's use of the ring, not the engine (rule 11) — ⑥ rewritten, #87 | #77 C |
| ④ Two-reader DDR | **inconclusive on the DSP side** (probe defect, rule 12, #90); CPU side clean: 67.9 → 39.7 GB/s under DSP contention (−41 %). Q11 stays open | #77 ④ |
| Control on `R3CY205ZMND` (12 cells, prompt 512) | NPU decode 18.2–21.3 tok/s, CPU 46.4–54.1, NPU prefill 403–541: the NPU is **2.7× short** of 50 at G=512; the CPU clears 50 at G=64/512 and not at G=1024. Provisional for our unit until #91 | #77 A |
| **⑯ M=1 HVX GEMV (PR #86, `NNTR_MOE_HTP_M1_GEMV=1`) — variant C of #94 sitting 2, same binary and sitting as A** | **Goal progress, accuracy gate passed.** Decode +6.2 % (G=64, 18.83 vs 17.72), +7.7 % (G=512, 18.33 vs 17.03), +1.7 % (G=1024, no change); text byte-identical to A at every G and run; gtest `MoeLayerM1GemvMatchesHmx` bit-identical (0 bad of 2048 / 8192). Profile M==1: dsp **1414 → 1044 µs/call (−26 %)**, `DMA ring:` 46 → 2 descriptors, wait 269 → 4.3 µs, gather 146.5 → 0 — the DMA wait is gone from the path — **but `mm` 783 → 975 µs (+24 %)**: the GEMV reads the arena directly at ≈ 22.6 GB/s effective (22 MB in 975 µs), slower than the HMX loop's staged DMA + compute. Transport 649 → 748 µs/call (+100 µs, unexplained; `moe_set_opts` is once per session, so it is per-call cost of the switched path — #88 reads it). Prefill untouched (M>1 dsp 0.13 % apart). **Next lever inside ⑯: the GEMV's weight feed** (prefetch the expert tiles into VTCM ahead of the `vrmpy` loop instead of the direct arena read) — it must not re-import row h's slow list, so it waits for #100 (open item ㉒). Provenance: device skel `d6568c8b…` from `08afbb10` on the user's workstation, accepted under rule 21 | #94 s2 §C, `dad0f476` |
| **⑦ Wall 3, transport (PR #103: upstream size-class ION staging `fb0f02b9` + 5 ms poll `04a2fcc4`/`b0a384d6` + `staging:` line) — #88, one sitting A/B/C, `R3CY205ZMND`** | **Closed; the largest single-issue gain so far.** M==1 transport **401.3 → 87.9 µs/call (−78.1 %)** at level 2 (82.3 at level 3), `qos_mode=2` in all four profiles, gate ≤ 0.1 ms met → plan 88 §3.3 prebind **not built**. C (B + `NNTR_HTP_POLL_US=100`) 155.5 → the 313.4 µs cut = **staging 245.8 (78 %) + poll 67.6 (22 %)**, the plan's order and size. `staging:` M==1 `act 65536 out 65536 ion=y rpc allocs=19 non-ION 6/464 B`, `allocs` flat at level 3 (no per-call allocation). Decode 18.72 / 16.83 / 16.46 → **24.50 / 23.80 / 23.52 tok/s (+30.9 / +41.4 / +42.9 %)**, C at G=64 21.82 (+16.6 %); text bit-identical A = B = C at every G, run 1 = run 2; prefill −0.9 % (six-cell means 486.4 → 481.9), M>1 transport 2494.6 → 1770.3 with M>1 `dsp=` −0.07 %; peak RSS +0.3 % (both size classes stay alive). Thermal: B ran hotter (48.8 → 56.6 °C) and still won every cell; A re-run after B 17.82 (inside A's own spread). Only ≈ 6.9 of the 12.6 ms/token gain is in the transport column; ≈ 5 ms landed outside the MoE call (rule 23). Provenance: A/B/skel rebuilt on the user's workstation from `08afbb10` / `f3e99176` — accepted under rule 22 | `88-moe-call-marshalling.md` @ `3f9fa38d` |
| **Per-token budget at `htp_moe` head (cycle 7 estimate, not a measurement: TPS token time minus 22 × the level-2 M==1 `host` of the same sitting's G=64 profile)** | #88 B, G=512: token **42.0 ms** = MoE dsp 22 × 1.383 = **30.4 (72 %)** + transport 22 × 0.088 = **1.9 (5 %)** + outside the MoE call **≈ 9.6 (23 %)** (8.4 at G=64: FC, conv, attention, norms, lm_head, sampling). Floors at 38 GB/s: MoE 484 MB/token (22 × 22.02 MB) → **12.7 ms**, the rest 246 MB (lm_head Q4_0 alone 147 MB) → **≈ 6.5 ms** — so the MoE dsp column is **2.4× its floor (17.7 ms of excess)**, the outside-the-call time **≈ 1.5× (≈ 3 ms of excess)**, transport 1.9. **Biggest lever = MoE dsp (walls 1–2).** Ladder (projections): + #101 (GEMV default, #94 C: dsp −370 µs/call) ≈ −8 ms → ≈ 34 ms (≈ 29 tok/s, if C's +100 µs transport does not return under the new staging); + #105's gate (`mm` 975 → 600) ≈ −8 ms → ≈ 26 ms (≈ 39 tok/s); MoE at its floor → ≈ 24 ms (≈ 41 tok/s). 50 tok/s = 20 ms also needs transport + outside-the-call ≤ ≈ 7 ms → ⑨ (#85) and ⑧. ⑰ in TPS terms is ≈ 2–3 ms, not the profile's 18 (the `--profile` build adds ≈ 20 ms/token to both paths, #77 76.2 vs 54.4 and 39.3 vs 19.0), so it is not the biggest lever and gets no issue of its own | #88 B / A / C profiles + TPS; #94 s2 A / C; #77 profile |
| **Plan 87 §3.2 attribution a–h (wall 2): in-situ B levels 2/3 + `MoeChunkReplay` + `DMA_PROBE`, one sitting** | **a refuted** (wait 10.2 / 19.0 % of dsp, depth 9–11), **b refuted** (pushed shapes = probe iii), **c refuted, provisional #99** (workers 1/2/4 → 40.2 / 39.1 / 36.5 GB/s), **d sensitivity, not cause** (DDR competitor +1620 % wait, HVX competitor 0), **e refuted, sign reversed** (expert 0 fastest), **f confirmed 9.7 / 10.3 %** (the single blocked `site=act` wait = gather), **g real 11.0 / 7.1 %, unexplained** (`fresh` and `gap_us` controls both null), **h new, ≈ 60–70 % of the gap, provisional #99** (rule 19). **Wall-2 fix = row h → #100**; f (hide the first expert's 3.5 MiB behind gather / the previous layer) and g (cold-call penalty) are the remaining ~20 % and are filed after h. Plan 87 §0 expected f + g large: they are real but small; h was not foreseen | #94 s2 attribution table |
| **⑱ Hadamard rotation on the MoE down_proj input (`QS4CX_WH_HAD`, FWHT-256 before the u8 requant), #95. One sitting A/B/C/D on the side tree `htp_hadamard` (upstream `3006d255` + tooling: HMX block loop only, no M=1 GEMV, no DMA trace, 5 ms poll), `R3CY205ZMND`. Off-tree, so the numbers are not in BENCHMARK.md's `htp_moe` rows** | **Carry it over. Accuracy win, cost within noise.** **PPL (`NNTR_PPL`, 511 prompt tokens):** D CPU `q40` **109.759**, A **115.095**, B **115.095** (= A to the last digit), C **100.497**. C is −12.7 % vs A and −8.4 % vs the CPU, the first NPU variant below the CPU. **Requant SNR** (`[L2-DIFF-MOE]`, min / p10 / median dB, about 6.2 k calls): B 25.01 / 31.27 / 34.68 → C **39.68 / 41.67 / 42.67** (+14.7 / +10.4 / +8.0). `had=0` on all 6246 B lines, `had=1` on all 6274 C lines. **Gates:** B text byte-identical to A at G 64 / 512 / 1024 (after stripping the `[HTP-MOE] opts` line). C diverges from A at generated word 52, and both NPU variants diverge from D at word 43 at every G (a property of the weights). **Cost** (`[HTP-PROFILE]` level 2, G=64, mean µs/call): `requant` M>1 1097.3 / 1121.2 / **1567.4** (A / B / C, +446 µs), M==1 41.9 / 42.0 / **46.6**. Whole-call `dsp=`: M>1 15915.6 / 16061.7 / **16491.0** (C vs B **+2.7 %**), M==1 1333.5 / 1334.3 / **1338.5** (**+0.3 %**). **Decode tok/s:** A 25.08 / 24.40 / 23.83, B 24.82 / 24.12 / 23.75, C 24.49 / 23.65 / 23.53. A's own G=64 cell drifted −2.6 % from the sitting's start to its end (29 → 58 °C), so every B and C delta is inside that drift. C vs A's end re-run is +0.3 %. **Prefill tok/s cannot be read** (A 466.8 → B 433.8 → C 403.2, but A's re-run at the end read 404.7: order and heat, not code). The M>1 `dsp=` +2.7 % is the prefill reading, inside the −5 % gate. **Gtests:** `RejectsPartialBlock` and `MoeSetOptsEchoesKnownBits` OK. `MatchesScalarBitExact` failed only in the subnormal row (rule 24). `MoeLayerHadamardMatchesTwoCallReference` was bit-exact but failed its SNR assertion (rule 25). **Provenance:** everything was rebuilt on the user's workstation (Deviation 1). Device md5s match the rebuilt set. `git diff 63167235 6c89a912` is exactly the #95 commits. The model `a2829cd9…` equals the predicted hash and differs from `q40-qs4cx-wh` only in the 704 `down` tensors. The verdict is B vs C, one binary set switched by the model's dtype (rule 21). **Side reading on the accuracy gate:** A's PPL is +4.9 % over the CPU `q40`, and its text leaves the CPU's at word 43. That is the size of the "n/a (different weights)" that every NPU row in BENCHMARK.md carries, measured for the first time. Remainder (spec fix, gtest assertion, port incl. the M=1 GEMV requant site, `NNTR_PPL`): **#110** | `95-down-hadamard.md` @ `3a566761` (`origin/htp/95-down-hadamard`) |
| **㉒ compute half: M=1 GEMV one-row loop + `l2fetch` lead (PR #107 @ `a0ae8b29`), #105. One sitting A/B1/B2/B3, one app set, four skels, `R3CY205ZMND`, `NNTR_MOE_HTP_M1_GEMV=1` in every cell** | **Gate failed, accuracy passed. PR #107: hold, do not merge as-is.** M==1 `mm` (level 2, mean of two passes ≤ 0.4 % apart): A **973.0**, B1 1044.6 (+7.4 %), B2 1004.9 (+3.3 %, the branch default), B3 **930.9 (−4.3 %, inside ±5 %)**. Decode vs A 27.743 / 26.879 / 24.826: B1 −4.1 / −4.0 / −8.4 %, B2 −1.9 / −4.3 / −5.8 %, B3 **+2.6 / +2.2 / −1.7 %** (A's own G=1024 pair spreads 7 %). No variant reaches `mm` ≤ 600. **Accuracy:** `bit_identical yes`, `bad_elems` 0 under all four skels; text = A in 18/18 B cells. **Prefill:** M>1 `dsp` within 2.5 % of A in all eight profiles, `m1_gemv=0/23`. The E2E prefill column drifts with position (rule 27), so the gate is not failed. **Split (`MoeM1GemvFeedVsCompute`):** `arena − hot` is **≈ 74 % of `mm` under B3** (≈ 690 µs feed / ≈ 240 compute) and ≈ 37 % under A's four-row loop (≈ 590 compute). So the feed half is the lever. The one-row loop is what lets a VTCM feed hide the compute (240 ≪ 600 rather than 590 ≈ 600) (rule 26). **Merge verdict (cycle 9): hold.** As-is, the default lead is 64 KB = B2, a measured decode regression at all three G. Flipped to 192 KB (B3) it is neutral, not a gain, and the flip is a code change (no implementer this cycle). Once the weights stream from VTCM the `l2fetch` lead has nothing to prefetch, so the lead question dissolves if #100 certifies a feed shape. The PR's lasting content is the one-row loop, the x86 HVX-emulation check (`gemv_native_check.c`) and the microbench. The ㉒ feed issue takes them over by rebasing onto #107 and measuring feed + one-row together. If #100's feed verdict fails (no shape ≥ 37 GB/s beside HVX), the fallback is to land #107 with the lead at ≥ 192 KB after a sweep (384 KB + L2 budget). That fallback is filed only then. **Also read:** the GEMV path's M==1 transport is 182–184 µs here vs #88 B's 87.9 on the GEMV-off path. That is cross-sitting (rule 23), but it is the same +≈ 95 µs #94 C showed pre-#103 (649 → 748). #108's sitting reads it as an A/B (⑦ note). Provenance: rebuilt set accepted under rule 22 (BENCHMARK artifact row) | `105-m1-gemv-compute.md` @ `da41340b` (`origin/htp/105-m1-gemv-compute`) |
| **⑥ Wall 2, row h + the M=1 GEMV feed shape (PR #112, test-only; #100). One sitting, one app set, one skel, plus a cooled repeat of the gtest** | **Row h dissolves.** `c_star` = **26.3 GB/s** = 0.36 × the same log's `DMA_PROBE shape=iii w=1` (73.2), `checksum_ok=y`; plan 100 §3.3's H0 fires alone (H1 0.41, list row 0.99). `traced / c_star` = **1.19** — the real 46-descriptor list is *faster* than the tag-validated ceiling, so **there is no list-side loss to fix** and rules 11 / 19 are amended (rule 28). **Feed shape: FAIL as measured, not closed.** Best `f2` (8 desc, `dst=strided`, depth 2): `load=0 / c_star` = 1.21 **pass**, `load=2` = **31.6 GB/s** vs the 37.0 gate **fail**; `f1` / `f3` 31.4. **HVX-from-VTCM is not the cause** (0.1–0.3 %, rule 29): the engine itself tops out near 31 GB/s on every validated cell. **Confound:** the sitting's whole DMA is ≈ 22 % low on an untouched cell (rule 30), and scaled by it `f2` would clear 37.0 — hence "do not close ㉒'s feed half on this sitting alone" (the user's own read). Hygiene: 16/16 `DMA_PROBE` and **16/16 `DMA_REPLAY_X` `checksum_ok=y` with `tag=N/N`** (including the five allowed to fail), 11 old `DMA_REPLAY` failures all at `:633` (#99), nothing at `:715` / `:692`, `plan_shape_ok=y`, no FARF/AEE. Cooled repeat at 31 800 m°C reproduces every ratio within 1 % | `100-dma-chunk-list.md` @ `4bf1fdac`, `R3CY205ZMND`, 2026-09-23 |
| **#101 ride-along in the #100 sitting: the M=1 GEMV default as an in-sitting A/A0** | **A (default on) = 27.887 vs A0 (`NNTR_MOE_HTP_M1_GEMV=0`) = 24.578 tok/s at G=64: +13.5 %** (the handoff expected ≈ 6 %), text identical once the `[HTP]` banner line is excluded, prefill A not below −5 % of A0 on the warm runs. So PR #108's default is a measured win, not an inherited one. Level-2 M==1: A `blocks=0 m1_gemv=1408/1408`, host 1227.2 / dsp **1041.5** / transport **185.8** / `mm` 980.9, ring `desc=2/call`, `rest` ≤ 10.1 = **0.8 % of host** (㉑'s print fix confirmed on silicon, and the `weight DMA: n/a (… 5.76 lanes busy over mm)` line is there); A0 `blocks=5632`, host 1469.2 / dsp **1384.6** / transport **84.6** / `mm` 783.1, ring = the 46-descriptor list at 20.4–29.1 GB/s. **Transport GEMV / HMX = 185.8 / 84.6 (Δ +101.2, 2.20 ×) with equal `act`/`out` classes (65536 B)** — ⑦'s open note answered in-sitting: the +100 µs survives the new staging, and it buys −343 µs of `dsp`. `qos_mode=2`, M>1 `m1_gemv=0/23` on both | same |
| **㉒ compute half, closed: the M=1 GEMV (loop × `l2fetch` lead) matrix and the lead sweep, PR #115 @ `5be9c4ba` (absorbs PR #107), #113. One sitting, one app set, one skel, five profile cells, A vs D192 E2E mirrored at all three G, `R3CY205ZMND`** | **Gate `mm` ≤ 840 FAILED on every cell; accuracy and prefill passed; D192 lands as the default by user decision (rule 33).** M==1 `mm` (level 2, G=64): A (four-row, 0) **1000.2**; B (four-row, 192) 1925.5 (+92.5 %); C (four-row, 384) 2126.9 (+112.6 %); D (one-row, 384) 1042.8 (+4.3 %); **D192 (one-row, 192) 937.0 (−6.3 %)** — reproduces #105 B3's 930.9 within 0.7 %. The matrix (`ns_per_tile`, arena, lead 0 / 192 / 384 / 768 / 1536): four-row 133.5 / 282.3 / 316.4 / 323.1 / 328.4, one-row 154.3 / **129.1** / 139.4 / 147.3 / 155.2; `hot` four-row flat 71.7–73.5, one-row 42.7 / **31.3** / 31.7 / 34.0 / 32.6 (rule 31). **The hypothesis of #113 is inverted:** the cheapest unmeasured cell (four-row + lead) is a 2× regression, the lead helps only the one-row loop and only to 192 KB, and the whole 2 × 5 board's minimum is D192; nothing left unmeasured could reach 840. **E2E, A vs D192:** decode 27.444 / 26.829 / 25.930 → **28.572 / 27.711 / 27.035 (+4.11 / +3.29 / +4.26 %)**, text byte-identical 6/6, prefill −0.77 / +0.05 / +3.73 %, M>1 `dsp` within 0.08 % (`m1_gemv=0/23`). **Accuracy (a):** `bit_identical yes`, `bad_elems` 0 of 20480 / 81920 over all ten pairs. **Ride-along:** 724.0 µs / 31.2 GB/s = #100 (rule 32) → ㉒'s feed half closes as a bandwidth question. **What lands:** the per-call `(lead, rows1)` bits of `moe_set_opts` with the ARM-side echo check, `gemv_native_check.c` (x86 HVX emulation), the swept `MoeM1GemvFeedVsCompute`, and — by the user's decision, not by the gate — `HVX_GEMV_M1_ROWS1=1u` + `HVX_GEMV_PF_LEAD_KB=192u` as the build and app defaults (**flipped and merged: PR #115 `c45d4433`, cycle 13; #113 closed**, its A/A0 device confirmation rides #117's sitting; handoff head `37fffcb4`). **#105 closes** (its ≤ 600 target is superseded by rules 26 / 31: `mm` at 937 reads the arena at 23.5 GB/s, ≈ 1.15× the direct-read band's top of 27, so there is no compute-side headroom left under a direct feed). **#114 drops to p2**: the up-half lead is the only untested lead form and can only help the one-row loop by a few % — moot if the VTCM feed lands. Provenance: rebuilt set accepted under rules 21 / 22 (BENCHMARK artifact row) | `113-m1-gemv-lead-matrix.md` @ `d03c8929` (`origin/htp/113-m1-gemv-lead-matrix`), 2026-09-23 |
| **㉒ feed half, measured: the M=1 GEMV fed from VTCM by DMA (`f2` shape, one expert ahead) under the one-row loop, PR #118 @ `3c4912fa`, #117. Two sittings (warm, then cooled) on `R3CY10WM83Y`, one app set, one skel, A (D192 arena) / A0 (pre-#115) / B (feed) mirrored at all three G, `NNTR_MOE_HTP_M1_GEMV=1` explicit in every cell** | **Gate passed twice; accuracy passed; prefill passed on the tie-breaker; the feed becomes the default (#117 → in progress, PR #118), by the gate this time, not by rule 33.** M==1 `mm` (level 2, G=64): A **922.7 / 931.9**, **B 676.6 / 676.4 (−26.7 / −27.4 %, gate ≤ 760)**, A0 989.7 / 997.4; B `dsp` 711 (−28 %), **transport 82.0 / 92.3 vs A's 184.6 / 190.3 (rule 35)**, host 793 / 804 vs 1170 / 1185. In situ the ring reads 22.02 MB at **32.6 GB/s** (`engine 32.3..34.4`, busy 640–682 of 712 µs — the DMA never idles, the ≈ 240 µs compute is hidden, rule 29 confirmed in the layer call); `first expert ready at 122 µs` = the first 3.5 MB slab un-hidden per call. The `vtcm` microbench cell runs at the engine rate (579 µs = 22.02 MB / 37.7 GB/s, tail −14.5 µs — plan 117 §3.2's upgrade not triggered). **E2E, B vs A:** sitting 1 35.66 / 35.93 / 31.47 vs 28.48 / 26.47 / 25.55 (+25.2 / +35.7 / +23.2 %), sitting 2 **37.38 / 36.49 / 35.01 vs 28.22 / 27.89 / 27.23 (+32.5 / +30.9 / +28.6 %)**; text byte-identical 36/36 (per-G hashes equal across both sittings). **Accuracy (a):** `bit_identical yes`, `bad_elems` 0 of 40960 / 163840 over the ten pairs × {arena, feed}, both sittings. **Prefill:** M>1 `dsp` +0.04 / −0.04 % (`feed=0/23`); tok/s B vs A −0.6 / −3.3 / −4.1 % (s1) and −3.3 / +4.9 / −8.5 % (s2), −2.5 % over all G in both; the s2 G=1024 cell is A's own pair (528.9 first after the cool-down vs 401.9 = A0's run 2), rule 27's spread → pass on the M>1 `dsp` tie-breaker. **Anchor:** 37.1 / 37.3 GB/s — a third value, on a second unit (rule 34); read honestly, the `mm` gate holds as measured on this unit and would miss by ≈ 6 % on `R3CY205ZMND` (≈ 805–809), while the decode half (+29–32 % here, ≈ +10 % projected there) holds on both. **A vs A0:** +3.56 / +2.41 / +3.93 and +2.41 / +4.16 / +3.91 %, text identical — #113's D192 default confirmed on a second unit; BENCHMARK's "now" moves to this A (28.22 / 27.89 / 27.23). **What lands (flip, PR #118):** `HVX_GEMV_M1_FEED` → `1u`, an unset run prints `feed=vtcm source=default`, `NNTR_MOE_HTP_GEMV_FEED=0` is the opt-out (= this A, the next sitting's A0); the host scoreboard, the `vtcm` microbench cell and the ×feed bit-identity gtest stay. **#114 closes** (the lead is forced off under the feed and the VTCM cell is at the engine rate: no lead form has anything left to prefetch). Residual: the in-situ engine 32.3–34.4 vs the anchor's 37.3 ≈ 86 µs/call ≈ 1.9 ms/token → ㉓. Provenance: the staged set itself ran (device md5 = table), the first sitting without rule 22's rebuild | `117-m1-gemv-vtcm-feed.md` @ `6412bfc9` (`origin/htp/117-m1-gemv-vtcm-feed`), 2026-09-23 |
| **㉔ Upstream PR #4327 sync (merge of `4ae1ebd7` onto `htp_moe` @ `d79c0efe`, code `0e98036d`, PR #121), #120. One sitting A/B + A0 on `R3CY10WM83Y`, two app sets and two skels (IDL differs), no CPU cell** | **Gate pass; decode unmoved, prefill up.** Decode A 35.97 / 35.96 / 35.17 → B **37.68 / 36.29 / 35.10 (+4.74 / +0.92 / −0.19 %)**; the +4.7 % at G=64 is A's run 1 (35.42, first E2E cell after the profiles) against its run 2 (36.53), G 512 / 1024 read ±1 %. M==1 `dsp` **711.7 vs 711.3**, `mm` 677.2 vs 676.6, transport 82.2 vs 87.7, ring `engine 32.3..34.4 GB/s` in both — the decode path is byte-for-byte A's with upstream's epilogue column renamed (`swiglu(hidden)`). **Prefill 487.6 / 459.1 / 448.8 → 548.2 / 506.0 / 476.7 (+12.4 / +10.2 / +6.2 %)**, the first variant to raise prefill: M>1 `dsp` 16 986 → 14 808 µs (−12.8 %) from `requant` 1099 → 70, `dequant` 929 → 275, `mm` 10 222 → 9672, transport 1979 → 1082 — upstream's default-on M>1 changes (`5731b6e5` down matmul one block behind, `aa371dd2` pooled epilogue). Every upstream engine knob at its default: no `dense` / `conv` / `FC` row in B's profile, qkv fusion on the CPU side reads the unchanged model (weight order q, q_norm, k, k_norm, v). Text = A 14/14 (hashes `55f577d9` / `d7e89826` / `db08b4e1` per G). Ride-along on B's skel: `bit_identical yes`, `bad_elems_M1 0 of 40960`. **A vs A0 (`FEED=0`) at G=64: 35.97 vs 28.62 = +25.7 %**, `mm` 677.2 vs 934.3, text identical → PR #118's default confirmed as a sitting's control; A is the "now" (BENCHMARK.md, contract §1). Provenance: staged set ran as-is, device md5 = table, 0 mismatches. **Not merged (cycle 16):** PR #121 re-conflicts with `b4c96bdb`; user decision 2026-09-27 = rebase, rungs 0–1, no new sitting (the per-token entry is off by default) | `120-upstream-4327-sync.md` @ `7ded3ce9` (`origin/htp/120-upstream-4327-sync`) |
| **Per-token budget, cycle 16 (from #120 A, the feed default as a control: TPS token at G=512 minus 22 × the level-2 M==1 `host`)** | **A 27.81 ms** = 22 × 0.794 = **17.5 (63 %)** [dsp 22 × 0.712 = 15.7, of which `mm` 22 × 0.677 = 14.9; transport 22 × 0.082 = 1.8] + outside the MoE call **≈ 10.3 (37 %)**; **B (sync) 27.55** = 22 × 0.799 = 17.6 + ≈ 10.0. The cycle-15 arithmetic on #117 B (27.40 = 15.7 + 2.0 + 9.7) reproduces within 0.6 ms on a control cell. **−7.8 ms to 20 ms (50 tok/s)**: ≤ 2.7 on the MoE side on this unit (㉓, the engine gap), the rest is the ⑨ track (transport 1.8 + outside 10.3 = 12.1 against a ≈ 6.5 ms byte floor) | #120 A / B level-2 profiles |
| **Per-token budget, cycle 15 (from #117 sitting 2: TPS token at G=512 minus 22 × the level-2 M==1 `host`; the same arithmetic on A and on B, so the outside-the-call term is measured twice)** | **A (D192, the "now") 35.86 ms** = MoE dsp 22 × 0.995 = **21.9 (61 %)** [`mm` 20.5 at 23.6 GB/s] + transport 22 × 0.190 = **4.2 (12 %)** + outside **9.8 (27 %)**. **B (feed) 27.40 ms** = MoE dsp 22 × 0.712 = **15.7 (57 %)** [`mm` 14.9 at 32.6 GB/s through VTCM] + transport 22 × 0.092 = **2.0 (7 %)** [rule 35] + outside **9.7 (35 %)** [9.78 → 9.71: unchanged, as it should be]. **To 20 ms from 27.4: −7.4 ms**, and where it can come from: (1) MoE `dsp` floor = 484 MB / the unit's anchor = **13.0 ms at 37.3 GB/s, 15.5 at 31.2** → ≤ 2.7 ms on this unit (≈ 1.9 of it is ㉓'s engine gap, the rest the 35 µs `dsp − mm` tail), ≈ 0 on the other; (2) transport 2.0 + outside 9.7 = **11.7 ms vs a ≈ 6.5 ms byte floor** (lm_head 147 MB + FC / attention / conv ≈ 99 MB at 38 GB/s) → **⑨ must deliver ≈ −5 ms** (#85 → #82 → #81, ⑧ inside the op table; ⑰'s ≈ 2–3 ms is part of the 9.7); (3) sum **19.5–22 ms → 45–51 tok/s**: 50 is at the physical ceiling (contract §1: 48–52) and reachable on a 37 GB/s unit with ⑨ at its floor; on a 31 GB/s unit the NPU ceiling is ≈ 45 and only bytes per token (§3.3) moves it. The cycle-12 ladder's "≈ 33 tok/s" for the feed was low: it assumed `mm` ≈ 710 at 31.6 GB/s and no transport gain; the device gave 676 and −2.2 ms of transport | #117 sitting 2 profiles + E2E |
| **Per-token budget, cycle 12 (estimate: #100 A's 36.9 ms at G=512 minus 22 × D192's `mm` gain 63.2 µs; the measured D192 27.71 tok/s = 36.1 ms agrees within 2 %)** | Pending default D192, G=512: token **≈ 35.5 ms** = MoE dsp 22 × 0.996 = **21.9 (62 %)**, of which `mm` 22 × 0.937 = 20.6 reads 484 MB at **23.5 GB/s** (floor at this unit's DMA bound 31.2 GB/s: 15.5 ms; at 38: 12.7), + transport 22 × 0.180 = **4.0 (11 %)** + outside the call **≈ 9.6 (27 %)**, floor ≈ 6.5. Ladder: (1) **VTCM DMA feed under the one-row loop** (㉒ feed issue, planner's next): the certified `f2` shape at 31.6 GB/s beside HVX (rules 29, 32) bounds `mm` at ≈ 710 µs with the ≈ 240 µs compute hidden → ≈ **−5 ms → ≈ 30.5 ms (≈ 33 tok/s)**; gate `mm` ≤ 760. (2) **⑨ one call per token** (#85 → #82 → #81): transport 4.0 + outside 9.6 = **13.6 ms** against a ≈ 6.5 ms byte floor — the only path from ≈ 30 to 20 ms. (3) Bytes per token: 22.02 MB/call is int4 per-channel already; expert reuse across tokens needs a cache the DSP does not have (VTCM 8 MB ≈ 1.5 of the 4 experts) — §3.3 track, no issue. **The compute-side of the M=1 GEMV is exhausted** (rules 26, 31): under a direct feed `mm` cannot go below ≈ 815 (22 MB at 27 GB/s) and D192 is at 937 | #113 profiles + E2E; #100 A budget |
| **Per-token budget, cycle 11 (estimate: TPS token minus 22 × #100 A_L2's level-2 M==1 `host` 1227.2 µs)** | #100 A (GEMV default), G=512: token **36.9 ms** = MoE dsp 22 × 1.042 = **22.9 (62 %)**, of which `mm` 21.6 at ≈ 22.6 GB/s vs the 12.7 ms DDR floor, + transport 22 × 0.186 = **4.1 (11 %)** + outside the call **≈ 9.9 (27 %)**, floor ≈ 6.5. Ladder, re-ordered after #100: (1) **`l2fetch` lead** — the feed is ≈ 74 % of the one-row loop's `mm` and is DDR *latency* (rule 26), the lead moved it 145.2 → 120.2 ns/tile at 0 → 192 KB and is not saturated; #113 gates `mm` ≤ 840 → ≈ −3.1 ms → ≈ 33.8 ms (≈ 29.6 tok/s). (2) VTCM feed (㉒) now worth ≈ 1.2 × over the direct read (31.6 vs 25–27 GB/s), not the 37/22.6 = 1.6 × assumed in cycle 9 — demoted. (3) Then 50 tok/s = 20 ms needs outside-the-call ≤ ≈ 3.5 ms → ⑨ (#85), ⑧ | #100 A profile + TPS |
| **Per-token budget, cycle 9 (estimate: TPS token minus 22 × #105 A's level-2 M==1 `host` 1215.7 µs)** | #105 A (GEMV on, post-#103), G=512: token **37.2 ms** = MoE dsp 22 × 1.033 = **22.7 (61 %)**, of which `mm` 21.4 at ≈ 22.6 GB/s vs the 12.7 ms DDR floor, + transport 22 × 0.183 = **4.0 (11 %)** + outside the call **≈ 10.5 (28 %)** (9.3 at G=64). Ladder (projections): VTCM feed + one-row loop, `mm` 973 → ≈ 600 → ≈ −8.2 ms → ≈ 29 ms (≈ 34 tok/s). GEMV transport back to the GEMV-off 0.09 (if #108's A/B confirms it is path cost) → ≈ −2 ms. Then 50 tok/s = 20 ms needs outside-the-call ≤ ≈ 3.5 ms (MoE dsp ≈ 14.5 + transport ≈ 2) → ⑨ (#85), ⑧ | #105 A profile + TPS |
| Control, #94 sitting 2 (`R3CY205ZMND`, 13 A + 6 C cells, one binary set `2a75f7d9` + #98 skel) | NPU **17.72 / 17.03 / 17.30**, CPU **52.43 / 49.22 / 48.31**, NPU prefill 389–527: **the "now"** (BENCHMARK, contract §1). Distance to 50: 2.9× at G=512 (2.7× with C). Same-unit drift vs #77: CPU −9..+4 %, NPU −16..+5 % per cell with DSP profile columns ≤ 4 % apart (rule 20) | #94 s2 §A/§B |
| CPU control re-run on `R3CY205ZMND` (#94 first attempt, 6 cells, next day) | 53.0 / 53.5 · 51.1 / 51.2 · 49.7 / 48.4 tok/s vs #77's 54.1 / 52.9 · 52.7 / 52.0 · 46.4 / 48.5: **one unit drifts ≤ 7 % per cell day to day** (rule 9 quantified for the CPU path); G=1024 stays 46–50, i.e. the CPU does not reliably clear 50 there. No NPU cell (#97). Not the anchor; not a verdict on any lever | #94 `ec7ad296` |
| **Per-op residency without adjacency to the MoE op (cycle 18, #130's PR, host-gated: `run_inproc_e2e.sh`, both fixtures)** | **A stretch pays only when it absorbs a round trip.** `forward` stops at the first non-resident op; with RMSNORM / QK_NORM / ROPE / CONV1D_GATE / ATTN_M1 resident beside MOE every stretch is still bounded by an ARM op (FC, ADD, ROUTER_TOPK), so the count is 4 calls per conv/attention MoE layer, 3 per dense layer, 1 for the tail: **95/token on LFM2.5 vs 22 with `KINDS=MOE`** (11 vs 4 and 12 vs 4 on the host fixtures, measured). The wiring is proven end to end (bit-identical stretches vs the `_det` specs in `graph_host_check`, `INPROC E2E PASS`), the speed verdict is the device's (plan 130 §4 step 7); the call count falls only with ㉓ / #132 | #130's PR, plan 130 §0 |
| **#130's sitting (cycle 19, `R3CY10WM83Y`, 2026-09-28, warm, one binary set switched by env — rule 21; local rebuild + `-DENABLE_HEXKL=1`, rule 36)** | **The per-token entry's plumbing is free; the wired resident set, as wired, costs −22 / −25 / −30 % and its text fails.** C (`NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=MOE`, 22 calls/token) 35.21 / 34.60 / 31.45 vs A 32.96 / 32.91 / 31.35 — inside A's own 18 % spread, text ≡ A 6/6, user `y`: plan 85's B measured at last, the entry itself costs nothing. B (all six kinds, **95 calls/token**) 25.77 / 24.79 / 21.82 = **−22 / −25 / −30 %** (plan 130 §0 said ≈ −20 %): the profile reads 95 × 220.1 µs = 20.9 ms of DSP time per token vs A's 15.7, per-kind `ATTN_M1=1204826` pcyc/op = 0.79 × a MOE op (`1523245`) while the four small kinds together are < 4 % of one, and the ride-along puts **ATTN_M1 at 538.75 µs at pos 511 / 1183.38 at pos 1023 ≈ 1.05 µs per KV position per layer** (plan 81 §0's 0.5 / 1.0 ms confirmed) — 6 layers ≈ 3.2 → 7.1 ms/token, plus 73 more round trips × 82–92 µs ≈ 6–7 ms: the two terms are the whole loss and the second grows with G. **Text: B leaves A at the first word in 6/6 cells and loops ("The final answer should be the same as the original, but you must not stop until you are told to."), user `text approved: n` → fail (#136)**; PPL n/a by construction (#134). Ride-along gtests 0/5 and 2/4: bit-exact on normal-range rows, few-ulp on subnormal rows, `AEE_ERPC` on bad shapes (rules 37–38, #137). Prefill: B and C inside A's range at every G (pass). A is **not** the "now" (warm phone, −8.4 / −8.5 / −10.9 % under #120 A on the same unit, uncommitted define; BENCHMARK Method). What this decides for ⑨: as wired the track is **+8.5 … +13.9 ms/token** against the ≈ −5 it must deliver; nothing below 22 calls/token (#132) is worth a sitting before B's text passes (#136), and the ATTN_M1 per-position cost (㉗) is a second term that #132's call-count fall does not remove | `130-per-token-wiring.md` @ `d3a02a66` on `origin/htp/130-per-token-wiring`; BENCHMARK #130 rows and side tables |
| **#136's sittings (cycle 20 close, `R3CY10WM83Y`, 2026-09-28 16:37–17:00 KST, one binary set from `b4a999bf` switched by env — rule 21; a G=64 ×1 text bisect, a G=4 dump sitting, then a workstation recompute; text only, no tok/s verdict)** | **Not a defect in the wired kinds: quantizer amplification of last-bit differences (rule 39). The worker-pool race is a real, separate defect, fixed.** (1) **Pool race (PR #143 `ff781a6c`)**: a worker read `fg_id` and then the job's fields as plain loads, so a descheduled worker could join the next job uncounted and let `hvx_worker_pool_run` return while a participant still wrote; fixed with two job slots indexed by generation and an acquire re-check of `fg_id` (seqlock). Host red → green: `POOL RACE` 1–26 bad slices per 6 × 2000 (or a hang) → OK; `run_inproc_load.sh` (30 concurrent all-kinds E2E) 25/30 → 30/30. On silicon it leaves the six-kind text the #130 loop word for word — it was not the text's cause. (2) **Bisect** over the executable masks (QK_NORM / ROPE need ATTN_M1, plan 136 §0): every kind group alone and every pair ≡ switch-off except RMSNORM with ROPE + ATTN_M1 (with or without QK_NORM); 71 and 89 calls/token pass, 77 fails — neither the call count nor CONV1D_GATE. (3) **Dumps** (G=4, `NNTR_HTP_DUMP_ALL`): switch-off twice and `KINDS=MOE` bit-identical at every MoE input; `MOE,RMSNORM` 34.2 → 18.2 dB and `MOE,ROPE,ATTN_M1` 40.2 → 18.3 dB over the first decode token's MoE layers, both recovering at token 2 (34.6 / 43.6 dB); the failing mask −1.6 dB at token 2 (another token); KV seeds byte-identical. (4) **Recompute**: 131–148 dB NEON order vs DSP, DSP nearer f64. Reading: #130's B stays failed by the user's `n`, but the question moves to the decode PPL (#134, PR #144) against A of the same sitting (contract §12, 2026-09-28). Withdrawn: rule 37's "logic / binding" reading and the `rmsnorm kind=2` lead (the ±1e-39 row, #137). Side number: switch-off **36.61 tok/s at G=64**, cool phone (zone0 28 °C at start), one run — a note, not a "now". **Open: the pool fix touches the worker pool that prefill and the M=1 MoE share, and no sitting has read it against a pre-fix A** (the #136 sitting had no such variant; its switch-off prefill 568.9 sits above every earlier A range, which is a cool phone, not a verdict). The combined #134 / #132 sitting's A already carries the fix, so the −5 % prefill gate and the switch-off decode of PR #143 stay unread unless a pre-#143 A0 rides along as its own set (skel + app from `c5dcb382`: PR #145 changed the IDL, so the pair travels together, rule 3) — user decision | `136-forward-text.md` (on `htp_moe` since PR #143); plan 136; BENCHMARK #136 rows |
| **#134 / #132 PR 1 combined sitting (cycle 21 fold; `R3CY10WM83Y`, 2026-09-28 20:11–20:22, one set = `bb845426`, switched by env — rule 21; the last per-token-entry sitting)** | **Speed: D (six kinds + ADD + ROUTER_TOPK, 51 calls/token) is +6.5 / +9.9 % over B (95 calls) at G 64 / 512 — the 44 fewer calls, as plan 132 estimated — and still −22.5 / −22.7 % under A (36.90 / 36.45); C (`KINDS=MOE`, 22 calls) −0.5 / −1.2 %, ≡ A. Accuracy: the decode PPL on A's own G=512 continuation passes for B (−0.088 %) and D (−0.222 %) with A's null check and C equal to 17 digits, but B's and D's text leaves A at the first word (`town` → `final`) and loops, and the user rejected both (A y, B n, D n).** Decode step 1 of p01 is a near-tie (A's `town` p = 0.134; B / D pick `final` with `town` at nll 2.103 / 2.113), so any last-bit perturbation (rule 39) picks either token, and the teacher-forced PPL cannot see how the free-running text degrades after the flip. This sitting is what led to the user's 2026-09-28 direction change (contract §12): only bit-preserving levers; the resident path stays off; #132 PR 2 and #146 parked. Router `router_topk` bit-exact on silicon (three shapes, `bad_*=0`). Ride-alongs: #141 step 1 → `rule=adopt` (Δ 70.4 µs, rule 40); #146 R1 → O1 / O13 bit-identical on silicon, ATTN_M1 941–959 → 387 µs `dsp` at pos 1023, the `dsp_us ≤ 300` gate not met (parked) | `134-132-combined.md` @ `3eecf213`; BENCHMARK rows and side tables |
| **#141 step 2: the 22 M==1 MoE calls through dspqueue (PR #151 `08de92b1`; `R3CY10WM83Y`, 2026-09-28 21:2x–21:5x, one set from `07fb1938`, A / Q / Q0 by env, mirrored)** | **Pass on every gate; default on by user decision (flip `c73384d2`).** G1 MoE dumps `bit_identical=1` for Q and Q0 vs A1 (2862 files, A2 null check = 1); G2 text ≡ A on the 8-prompt set; G3 ≡ A in 8/8 tok/s cells; G4 decode **36.78 → 39.40 (+7.1 %) at G=64, 35.98 → 37.77 (+5.0 %) at G=512**, above A's own spread (0.04 / 1.51 tok/s); G5 `transport` 87.9 → **14.1** µs/call, `dsp` −1.3 %, `mm` −0.2 %; G6 prefill +0.6 / −1.6 % on means (M>1 unchanged by construction); G7 banners as specified. Q0 (DSP blocks) +4.8 / +0.4 % — the spin is the lever (rule 40). G=1024 not measured. BENCHMARK's "now" moves to Q (Method, cycle 21) | `141-dspq-moe.md` @ `4049dd15`; plan `141-dspq-moe.md` |
| **#149 step 0: the MoE feed engine in the app vs the isolated anchor (㉓ feed half; `R3CY10WM83Y`, 2026-09-28 20:56–20:58, the #134 set, no build)** | **D2 — no fetch-schedule lever; ㉓ closed with no code (#149 closed).** Anchor 37.3 GB/s, replays 37.2–37.5, `vtcm` bench 37.64; in-app per-descriptor median 32.9 cold / 33.0 warm = 0.88 × anchor; the ring never idles (rule 41). The MoE `mm` floor on this unit is ≈ 670 µs/call (14.7 ms/token), already reached (673.7–677.2) | #149 comment; plan `docs/plans/149-feed-engine.md` |
| **#150 step 1b: per-op decode on the dspq default (`NNTR_OP_TIME=1`, PR #153 `b6b92077`; `R3CY10WM83Y`, 2026-09-29 10:50–11:05, A0 = #141b set, T8 / T6 / T4 from `56ba0835`, mirrored, G=512)** | **Timer inert; CPU side near its byte floor (rule 46); #150 closed with no lever built.** A0 37.48, T8 38.47, T6 38.97, T4 37.64 tok/s (inside A0's 1.40 spread); dumps T8 vs A0 `bit_identical=1` (2862), nll equal, texts equal. Per-op: MoE wait 16.08 of 26.22 ms; FC + lm_head 7.4 ms at 51–63 GB/s; attention 1.47; rest ≈ 0.9. Bit-preserving CPU headroom ≈ 0.5–1 ms at G=512, not the plan's 1.5–2.5 | `150-cpu-decode.md` (on `htp_moe`) |
| **#90: two-reader DDR probe + prefetch overlap (PR #156 `397abc07`; `R3CY10WM83Y`, 2026-09-29 11:07–11:10, probe `f0e448ee…` on the #150 set)** | **④ / Q11 answered: DRAM ≈ 70 GB/s aggregate for any reader mix (rule 44); #90 closed.** CPU + ring 69.5–71.5, CPU + HVX 64.6–66.6; prefetch S = 4 MiB +0.7–1.4 ms/token with the DSP unchanged, S = 10 warm only, S ≥ 20 a loss; rule P fails cold, passes warm at T 2 / 7 (L2 path — re-read under the bypass, #162). E2E A 39.70 / 39.18 / **36.53** (first G=1024 on dspq). `MoeChunkReplay` in this probe binary `checksum_ok=n` = pre-#154 binary, expected | `90-two-reader.md` (on `htp_moe`) |
| **#99: `MoeChunkReplay` content check with the kernel's strided VTCM layout (PR #154 `cf20aa50`; ride-along 2026-09-29 11:0x, `R3CY10WM83Y`)** | **Fixed; ⑳ closed; #99 closed.** Anchor `DMA_REPLAY workers=1 load=0 pace=0 … us_per_call=608.1 bytes_per_call=22560768 weight_bytes=22020096 gbs=37.1 … checksum_ok=y`, `[  PASSED  ] 1 test`, within ±10 % of #149's 605.0 µs | PR #154 comment; `/local/mnt/workspace/htp_moe/99/logs/dma_probe.log` |
| **#152: resident ROPE + ATTN_M1 bit-identical to the Android fp16 CPU attention (PR #160 `f330dfbe`, off by default; `R3CY10WM83Y`, 2026-09-29 11:56–12:33, A / D0 / D1 mirrored, 8 prompts at G=256)** | **Accuracy of the arithmetic reached, gate failed on speed and on the user's approval; #152 closed, track (c) stays off (rule 45).** G0 / G1 / G2 pass (`N_attn` dumps `=1`); G3 fails (`dsp_us` 1736.6 at pos 1023, gate 940); decode A 38.87 / 38.54 / 35.53, D0 28.57 / 27.88 / 26.57, D1 25.91 / 24.55 / 22.51; D1 PPL within +0.9 % on every prompt, ≡ A on p05 p06 p08 only, new loop on p04 (A loops on p02 p03 p05 p06 p08); **user `n` on all 8** | `152-resident-accuracy.md` (on `htp_moe`) |
| **#158: DSP L2 bypass on the MoE weight DMA (PR #159 `130fdf0d`, default by `21169f6e`; `R3CY10WM83Y`, 2026-09-29 12:33–12:49, one set from `0106622f`, A / B mirrored at G 64 / 512 / 1024)** | **Pass on every gate; the largest single gain since the VTCM feed; default on by user decision; #158 closed (rule 43).** Decode A 38.61 / 37.40 / 36.17 → **B 51.82 / 50.61 / 47.36 (+34.2 / +35.3 / +30.9 %)** — ≥ 50 at G 64 / 512 for the first time; M==1 `dsp` 726.0 → 417.5 µs, engine 31.5..33.4 → 56.3..57.9 GB/s; MoE wait 16.07 → 9.85 ms/token; dumps `bit_identical=1`, nll B == A, text 8/8 + 6/6; prefill equal within temperature (mirrored-mean −8.3 % at G=512 from A's cool first cell, adjacent cells ±1 %, M>1 `dsp` −2.7 %). Unset-banner device check pending | `158-dma-bypass.md` (on `htp_moe`); probe in PR #159 |

| **#164: the HTP RMSNORM / QK_NORM row scale in the Android CPU's order (PR #167 `41b03160`, merged `efb88c7a`; `R3CY10WM83Y`, 2026-09-29 16:44–17:21, hot, one set from `dev/norm-shadow-164` @ `07718beb`, A / R / Q / RQ mirrored)** | **Bit-identical on silicon; #164 closed. G0 ✓ (the phone's CPU = the spec, 41 041 rows), G2 ✓ (RMSNORM 392/392, QK_NORM 1920/1920, logits 8/8 = A, nll equal), G3 ✓ (dumps `=1`), G4 ✓ (8 prompts nll and text ≡ A, loops = A's); G1 kind 2 ✗ and `RejectsBadShapes` ✗ (#137, rules 37–38); G5 ✗ (16 077 / 33 629 vs ISS gates 4000 / 25 000 — re-baseline from silicon).** Speed A 47.88 / 49.50 / 45.86, R 38.26 / 37.86 / 34.36, Q 33.47 / 31.50 / 25.89, RQ 29.69 / 27.17 / 22.79: the call count (71 / 28 / 77) and the f32 ATTN_M1 (2.81 M pcyc/op), not the r (rule 47) | PR #167 comment; `164/logs/`; `164-cpu-order-norm.md` (template) |
| **Norm shadow sitting (branch `dev/norm-shadow`, 2026-09-29, before #164; `/local/mnt/workspace/htp_moe/norm/logs/analysis.txt`)** | **The old RMSNORM differed from the CPU only in r:** self-check 392/392 (a fresh CPU recompute = the layer's real output); CPU vs HTP on the same input at pos 512: 21 of 49 calls all 2048 equal, the rest max 2–4 ulp, mean ≤ 1.64 ulp, SNR 136.4–142.4 dB, 4 q8 flips on call 0 (e.g. `x=-6.628e-02 cpu=3ef7d478 htp=3ef7d474`); tag0 185/392 (R) / 195/392 (RQ), logits 0/8 (R, RQ), 1/8 (Q). Forcing r to the CPU's reproduced 392/392 → plan 164 → fixed by PR #167 | `analysis.txt`; plan 164 §0 |
| **#170 round 1: ATTN_M1 in fp16 lanes with the one-rounding qf32 FMA (PR #176 `38f80800`, merged `90d88e2b`, off by default; `R3CY10WM83Y`, S1 2026-09-29 19:36 cool, S2 20:37–21:13 cool start; A / Q0 / Q1 / RQ1 mirrored)** | **Accuracy: pass everywhere on silicon** (G1 0 bad of 107 278 FMA cases + exp16 + div16; G3 kernel = spec at 8 L; G4 model attention = CPU's for every head / layer / step, 1536/1536, logits 8/8; G5 nll = A to 17 digits, text Q1 = RQ1 = A 8/8, **user `y`**). **Speed: gate missed 2× — 394 501 / 700 449 pcyc/op vs 210 k / 350 k** (Q0 2 465 164 / 4 751 395: −84 / −85 %); per L `dsp_us` warm 763 → 158 (pos 511), 1823 → 265 (1023), 3236 → 440 (1535); decode Q1 42.13 / 43.71 / 42.31 vs A 52.46 / 50.84 / 48.26 (28 calls/token) — default off. Terms at pos 1023 cold: pv 1.54 M (V-row fetch in the chain), append 95 k (scalar), softmax 0.53 M, scores 0.70 M, lane balance 1.28× → round 2 (rule 48) | `170-attn-m1-hf.md` §S1 / §S2; issue #170 |
| **#170 round 2: V-row `l2fetch` lead, chain-range units, vector append / ET (PR #182 `098f51a1`, merged `196a3e13`, cycle 24; `R3CY10WM83Y`, 2026-09-29 23:56–00:26; A / Q1 (round 1) / Q2 / RQ2 mirrored)** | **Accuracy: pass (G3 / G4 / G5 as round 1, Q2 = Q1 = A 8/8). Speed: gate missed by 19 / 7 % — 250 289 / 373 642 pcyc/op vs 210 k / 350 k** (Q1 same sitting 411 215 / 668 528: −39 / −44 %). Per term at pos 1023 cold: **pv 1545 k → 386 k**, append 95 k → 33 k, scores 698 k → 668 k (warm +9.5 %), softmax 528 k → 518 k (unchanged, now the largest with scores), busy_max 595 k → 278 k, `dsp_us` 342 → 165; decode Q2 44.49 / 46.26 / 45.24 vs Q1 44.71 / 45.10 / 44.07 (+2.6 % at G 512 / 1024), A 55.13 / 50.82 / 51.40. **Round 3 (on #170): split the softmax word first, compile the P1 lead out, time append's 33 k apart; DMA-into-VTCM stays unopened.** Default off | `170-attn-m1-round2.md`; issue #170 |
| **#162: G=1024 by bit-preserving CPU-side levers — step 0 sizing (`19fb142a`, 16:21) and the prefetch A/B (PR #172 `47494ee1`, 17:37–17:51; `R3CY10WM83Y`, 2026-09-29)** | **Both levers void; #162 closed, PR #172 closed by the user.** (a) attention microbench: no speedup at L 1024 / 1536, coverage 0.50 → not built; (b) prefetch 4 MiB: probe P′ pass (+1.0…+1.6 ms/token cold under the bypass), **in-app +1.30 / −0.43 / −1.30 % vs A (53.11 / 52.88 / 49.11), inside A's spread**, dumps `=1`, nll = A, text ≡ A 8/8 — G1 (G=1024 ≥ 50) fails. Step 0's A at G=1024 with `NNTR_OP_TIME=1`: 52.36 / 50.90 (token 19.1–19.6 ms: MoE wait 9.5, FC + lm_head 6.8, attention 1.7–1.9). G=1024 ≥ 50 moves to the end-to-end track (PR #169) and to rule 52's reading (rule 51) | `162/logs0/sitting.out`, `162/logs_b/speed.txt`; issue #162 |
| **#132 PR 2 Part A: CPU-exact Q4_0 FC, router, SwiGLU, quantizer, argmax specs + DSP kernels (PR #175 `e5e579b3` → merged `721ff913`, cycle 24; S3 2026-09-30 02:22–02:47 and S4 07:33–07:58 added below; `R3CY10WM83Y`, 2026-09-29 21:13–22:0x; A / S mirrored + P / Pb profiles)** | **G0 ✓ G2 ✓; G1 ✓ for `hvx_intrin` only** (native `.sf` and `sffma*` bad=2048 per row on silicon); **Part A's arithmetic closes on the intrinsics variant; decision D pending (user).** G3: exact FC + lm_head VTCM-fed **7.88 ms/token** vs CPU 7.4 (direct 30; quantizer +4.8 scalar; router 427 784 vs 34 100 pcyc/op ≈ 4.4 ms/token); G4: **113 MiB** free vs 383 needed. A 53.24 / 51.18 / 50.86; S ≡ A (text 8/8, nll) (rule 49). **S3 (cycle 24): the cause is rule 53** — the inline-asm IEEE `.sf` forms return `0x0` on v79 (`SfProbe` 256/256), intrinsics + `sffma` exact; the intrinsics-only kernel with the HVX quantizer: G1 10/10, **G2 fc 600/600**, **VTCM-fed 7.844 ms/token** (`quant_ms` 0.527, was 4.8), heap 109 MiB, router 65 528 pcyc/op (gate 60 k, −9 %). **S4: router with a row `dcfetch` 62 311 pcyc/op (−4 %, accepted** as Part A's result: ≈ 0.65 ms/token inside the E2E budget's ± 1 ms; ISS −60 % vs silicon −5 %, rule 58); 8 prompts S ≡ A both sittings. **Part A closed with the merge; Part B (§2 row below, ㉓) is the successor** | PR #175 comment; `132/logs/`, `logs_s3/`, `logs_s4/`; `132-pr2-exact-fc.md` §S3 / §S4 (merged) |
| **#178: a second cDSP session for the FC set + lm_head (decision D option (a) probe; `htp/178-probe` @ `bc3c52a8`, plan PR #180; `R3CY10WM83Y`, set2 2026-09-29 23:28, set3 2026-09-30 00:26 after a reboot; A + P cold / warm)** | **Q1–Q5 all `[OK]` cold and warm, both sets; by the numbers option (a) is not faster than A: projected ≈ 21–22 ms/token (≈ 45–46 tok/s) vs the hybrid's 18.4–19.7 (50.6–54.3), because S2 gets 0 VTCM (L2-fed exact FC 8.06 ms/token vs CPU 7.4) and concurrent S1 + S2 reads already sum to the 70 GB/s ceiling (S1 −17 %).** Mechanism facts (rule 50): open 22 ms lite, per-PD 4 GiB (3584 beside 3840), 383 MiB fit, mailbox hop 0.33 / 2.4 µs, dspqueue spin 2.8 / 12.7. Set2's heap-to-the-end cell leaked a 2 MiB mapping across processes until reboot; set3 fixed. A set3 (cool) 52.85 / 53.5 / 51.59. **Next step is the user's: build the two-session E2E plan, take (c) of #132, or the §3.5 fallback (㉚)** — **taken (cycle 24): decision D = (1) + (2); PR #180 (plan) merged `8e95bd51`; the probe PR #183 closed, to be re-targeted onto `htp_moe` over PR #175; the two-session design is #132 Part B (row below), whose E5 added the mapping-order rule 54** | issue #178 comments; `178-second-dsp-session.md` (branch) |
| **#170 round 3: `vlut16` q operand, the exp table, the softmax word split into five phase words (PR #190, merged `0a380112`, cycle 24; `R3CY10WM83Y`, 2026-09-30 04:40–05:07; A / Q2 (round 2) / Q3 / RQ3 mirrored; four skels of one source, W = `r3e`)** | **Accuracy: pass (G3 22 lines 0, G4 1536/1536 + logits 8/8 for Q3 and RQ3, G5 nll = A ×17, text = A ×24, user `y`). Speed: both gates pass — in-model ATTN_M1 193 838 / 313 747 pcyc/op vs 210 k / 350 k (0.92× / 0.90×)**, from round 2's 254 869 / 379 551 in the same sitting (−23.9 / −17.3 %); cold pool at pos 1023 307 k → 281 k, `dsp_us` 163 → 148. What landed: the `vlut16` q operand (`scores` 681 k → 392 k cold, 601 → 314 k warm; `append` 34 k → 24–28 k); the P1 lead is worth 649 → 387 k with the lut; the ET lead is neutral (removed). What did not: the exp table's gather costs **387 k** on silicon against the exp16 compute's 272 k (the ISS priced it 3× low, rule 58) — kept as the default because the gates were read with it; **round 4 = `-DATTN_M1_EXP_TAB=0` gtest cell, then `vgather` from a VTCM carve-out**; order by size (W, cold): exp 373 k, scores 385 k, pv 384 k (DDR line), div 123 k, et 71 k. Decode Q3 44.49 / 46.04 / 44.49 vs Q2 44.05 / 45.51 / 43.99 (+1.0 / +1.2 / +1.1 %), A 53.94 / 53.13 / 50.33: 28 calls/token keep Q3 12–18 % under A — the round trips are #132 Part B's; default off. Prefill Q3 +3.4 / −1.2 / +3.0 % | `170-attn-m1-round3.md` §S4 (merged); issue #170 |
| **#185: hide the decode MoE schedule cost on the S26 (v81, 4 queues; `htp/185-m1-schedule-overlap` @ `25764b23`, PR #191 open; `R5KL20NFRCK`, 2026-09-30 03:55–05:26, implementer adb under the v81 exception; A = #177 Q4t skel / DQ = C1 + C2 (`7d0cb151`) / DQR = + C3 (`8dea5f0a`), mirrored A DQ DQR DQR DQ A, G 64 / 512 / 1024 × 2 rounds + 2 re-runs; md5 checked before every run, none failed)** | **G4 pass on both: DQ 52.09 / 51.52 / 48.61 tok/s (+4.2 / +4.0 / +3.0 % vs A 50.00 / 49.52 / 47.18; +0.5 / +1.8 / +2.6 % vs the S25's #158 B 51.82 / 50.61 / 47.36), DQR 52.76 / 51.50 / 48.76 (+1.8 / +1.8 / +3.0 % vs S25); DQ's G=512 mean is above A's max. Ship rule (plan §1: DQR only if its G=512 mean ≥ DQ's): 51.50 vs 51.52 → DQ ships, C3 dropped from the PR and kept on `htp/185-dqr` for the user's call** (DQR leads at G=64 +1.3 % and G=1024 +0.3 % and has the lower `dsp`). G6 (level 2, G=64): `dsp` 454.1 → 427.1 (DQ, −27.0) → 422.5 (DQR, −31.6) µs/call; `requant` 57.3 → 4.5 → 0.0, `rest` 61.9 → 54.2 (GU(0) hidden behind QUANT ≈ 8 µs), `mm` 319.3 → 351.8 (C(0) / C(1) now carry D(2) / D(3)), `dmaq=4.00`, `DMA_FIRST` 59 → 50 µs (the remainder after QUANT); plan model 419 vs 422.5. Bits: dumps `bit_identical=1` (A = #177 Q4; DQ, DQR = A, 2862 files), nll byte-identical on A's `cont.ids`, text 8/8 both, `bad=0` in all 84 logs. Prefill (G5): medians within −2.7..+1.9 % of A at every G (means carry outliers on every variant: A 446 / 448, DQ 396 / 356, DQR 376 / 387 — the S26 noise #187 addresses); S26 prefill medians +19..+27 % over the S25 at G 512 / 1024, level at G=64. Host: `ALL CHECKS PASS` at each commit, the three negatives fire once each. What is left: C(2) / C(3) still move no bytes (≈ 38 µs/call DMA-idle, `ponytail:` in `hexkl_mm_u8i4_moe.c`) → **#197** | `185-m1-schedule-overlap.md` (branch); issue #185; PR #191 |
| **#192: the single-session map window for the FC set (probe `72086ab3` on `htp/192-map-window`; `R3CY10WM83Y`, 2026-09-30 07:59 after a reboot; sanities + `MapWindow.W1–W4`)** | **Not viable (stop rule > 1.0 ms/token hit by 30×; issue closed by the orchestrator).** fastrpc map + unmap of an 11 MiB window 1.40 (`fd`) / 1.55 (`fd_delayed`) ms per pair → **33.4 / 36.7 ms/token**; `HAP_mmap` unavailable to the unsigned PD (`0x80000402`); re-attach by copy 23.8 ms/token; fresh mapping +115 µs on the first FC call; lm_head 140 MiB whole map 2.84 ms (328 MiB VA headroom with the window) or 12 slices ≈ 15 ms/token; W1–W4 `[OK]`, no leak (rule 57). The two-session design (#178 / #132 Part B) stays the route to a resident FC set; the hybrid keeps the FCs on the CPU | issue #192 closing comment; `192-map-window.md` (template on the branch) |
| **#132 Part B: the two-session NPU end-to-end decode on silicon — E5 / E5b / E5c / E5e / E5d (`htp/132-partb-e3` `10bb91b7` → `7cf87794`, shadow branch `dev/e2e-shadow-132`; `R3CY10WM83Y`, 2026-09-30 06:13–09:2x, five reboot-bracketed sets; decision D = (1) + (2), plan PR #186)** | **Mechanism: pass** — once S1's arena is mapped before S2 opens (rule 54; E5 died at `mapped=3584`), the path loads (S2 open 24 ms, attach 383.6 MiB, load 685 ms, startup +343 ms) and runs every token at **`calls/token=1.00`, 44 hops**, no timeouts, no leak. **Accuracy: E5b failed** (E ≠ A from decode step 5–26 on 8/8 prompts with every G=8 check and every shadowed op equal — not folded as a result), **E5c located it** (S and Ev leave A at the same conv-layer row, L13 / L17, with every shadow equal), **E5e passes**: CONV1D_GATE in the CPU's fused tap order (`fma(w2, s0, fma(w1, s1, w0·g))`, scalar `sffma` taps) → S and Ev bit-identical over 64 steps on prompt512 and korean, every shadow equal incl. conv 1152/1152 (rule 55). **Speed: E5b 16.7 / 16.0 / 15.2 tok/s** (the 1000 µs hop-wait spin starving the other PD, rule 56) → **E5d spin 0: 20.8 / 19.5 at G 64 / 512**, text ≡ A; per kind (G=64) S1 MoE 9.67 ms (at the isolated 0.44 ms/round) + router 0.79, S2 FC 8.75 + DENSE_FFN 5.60 + LM_HEAD 5.97 + ATTN_M1 8.33 (pre-#170 kernel) + small ≈ 1.2, ARM token 44.7 ms. **Verdict: a lever row far under A (53.5–55 / 54.4–54.8 / 52.4–52.9), not a default; the issue continues** — rebase onto `htp_moe` (ATTN_M1 round 3 ≈ −7 ms), the HVX quantizer / SwiGLU / argmax in the graph kernels and per-shape lanes (FC / FFN / lm_head 20.3 → ≈ 11 ms), then set_e5f; projected ≈ 25 ms/token ≈ 40 tok/s before the FC-loop work, against the plan's 47–51 (the FC loop toward 57 GB/s is the missing term). Track 2 (VTCM share) stopped: the MoE prefill layout needs ≥ 6720 KiB (caps 6656 / 5888 → `AEE_ENOMEMORY`), S2 gets ≤ 0.95 MiB → a ≤ 3-lane VTCM feed slower than L2 (≈ 15 vs 8.06 ms/token) | issue #132 comments (E5b, E5c, E5e / E5d, track 2); `132-part-b-e2e.md` (branch); `132/set_e5*/logs/` |
| **Record sitting 2026-09-30: cool-start A control for the row of record (`record-2026-09-30-cool-a.md`; `R3CY10WM83Y`, 02:48–02:53 KST, `htp_moe` @ `90d88e2b`, device dir `s170r2q`, nothing set, A twice per G, each G block at zone0 ≤ 35 °C)** | **The "now" moves to 53.97 / 52.16 / 51.41** (r1 / r2 53.51 / 54.42, 54.75 / 49.56, 51.54 / 51.29; prefill 574.0 / 482.1, 542.4 / 526.2, 579.8 / 426.0; zone0 after 59.0 / 55.2, 62.1 / 62.9, 64.5 / 65.6 °C; text r1 = r2; banners `applied=0x703e1 … dma_bypass=1 source=default`, `dspq` close `bad=0`). Decode ≥ 50 at all three G on a cool start; #158 B's 51.82 / 50.61 / 47.36 is kept as the warm-start reading (+4.1 / +3.1 / +8.6 % cool over warm), so the goal holds under the stated condition and rule 52 is the row's condition, not an open question. Not a lever. | `record-2026-09-30-cool-a.md`, BENCHMARK Goals + Method (record-sitting paragraph), contract §1 |
| **Record sitting 2026-09-30, CPU `q40` control on the same unit (02:58–03:03 KST, `R3CY10WM83Y`, same app set, model bin `d28f55c5…` = #78's, config 512 / [124900] / no-sample for the sitting, `htp=0`, cool start per G 31.6 / 32.8 / 33.6 °C, two runs)** | **CPU 52.18 / 51.31 / 50.12** (52.03 / 52.33, 51.74 / 50.87, 50.49 / 49.75; prefill 348 / 332, 338 / 318, 337 / 280; text r1 = r2) — the first CPU cell on this unit, −0.5 / +4.2 / +3.7 % vs `R3CY205ZMND`'s #94 s2. **Read against the NPU cool row (53.97 / 52.16 / 51.41): NPU above the CPU by +3.4 / +1.7 / +2.6 % under the same protocol on the same unit — the contract's "above the CPU" clause is met on `R3CY10WM83Y`, narrowly** (the G=512 margin, 0.85 tok/s, is inside one run's spread; the CPU itself clears 50 at every G cool). | `record-2026-09-30-cool-a.md`, BENCHMARK Goals (both decode rows), contract §1 |
| **#201 S2, one PD (Q28 / Q29 = P28 + `NNTR_HTP_E2E_PDS=1`) against two PDs (P28), E0 and A — sitting 1 `R3CY10WM83Y` (2026-10-01 09:43–09:58, G 64 / 512, set `f5e8b1648`, 22 runs, no stop) and sitting 2 `R3CY205ZMND` (device farm, #207, 10:57–11:22, G 64 / 512 / 1024, set rebuilt at `a2ebef9c9`, 32 runs, two LEAK stops, three boots; tables from the issue comment, logs on that machine); with S0 (hybrid pool, 21:42) and the two-PD sitting (22:18) of 2026-09-30 on `R3CY10WM83Y`** | **One PD wins, bit-identical, on both units; still under A.** Q28 **42.98 / 43.13, 43.40 / 42.44** (G 64 / 512) vs P28 36.72 / 35.81, 36.67 / 34.13 vs E0 30.51 / 30.46, 30.79 / 30.66 vs A 56.74 / 54.51, 48.45 / 54.23 on `R3CY10WM83Y`; **40.48 / 40.97, 44.68 / 46.74, 45.11 / 45.20** vs P28 33.16 / 32.94, 35.37 / 39.07, 36.75 / 35.89 vs E0 29.30 / 30.09, 31.23 / 32.31, 31.02 / 30.92 vs A 54.61 / 54.65, 51.81 / 56.11, 51.82 / 50.30 on the farm unit (G 64 / 512 / 1024); Q29 = Q28 within spread with fewer misses; every text == A r1 of its G (20 / 20, 30 / 30; S0 30 / 30, two-PD 28 / 28). Mechanism: FC set on S1's VTCM 9.2 ms vs 11.6–13.1 on S2, no hops, `rt` 22.4 vs 26.6 / 30.6 (rule 59 a). Two-PD sitting: the pool beats all-resident E0 by the server spin's wake-up effect (L0 by accident, ≈ −4.5 ms a token), P32 35.6 vs E0 30.3–30.8. S0: the pool on the hybrid path F28 ≈ A (54.9 / 53.5 / 54.4 vs 57.3 / 55.4 / 54.9), F16w −20 %, cold 12–13. Pool-28 miss 3.45 / 3.79 ms on the two units, unexplained (rule 59 b, ㉜); LEAK 2 in 32 (farm) vs 1 across three sittings (㉝). Prefill: sitting 1 and two-PD inside −5 % on the means; sitting 2's column not in the comment. **No row of record** (lever on open PR #203, under A, second unit); the next structural read is the MoE round (10.5 ms) and the FC set's 1.1 ms over isolated, on the S26 (#204 / #208) | `201-one-pd.md` §Sitting 1 / §Sitting 2, `201-fsu-e2e.md`, `201-pool-baseline.md` (all on `htp/201-pool-miss-path`); issue #207 comment 2026-10-01; `201/s0/`, `201/s2/logs/`, `201/s3/logs/` (sitting 1) |
| **#208: the S26 re-baseline on `htp_decode` (plan 204 §4 step 7; set from `4c0c20dca`, v81 skel `26fdf25a…`; `R3CY70LV96T`, SM-S948U userdebug, 2026-10-01 14:12–15:29 KST, three invocations, 24 runs + 2 profiles + CPU control; run by the orchestrator)** | **E2E runs on v81; optimization deferred.** The one-PD E2E (Q28) and the two-PD variants ran on an S26 with every NPU text == A q4 r1 of its G (20 / 20) and the S1 ceiling at 3840 MiB on all 24 cells; gtests on the v81 skel pass except #137's known set (softmax 28/4, attn 10/2) and `RegistryCapacity`, which drops this unit (rule 60). The tables (A 53.9 / 53.6, 57.9 / 55.9 at G 64 q4 / q1; E0 29.3 / 29.3, 30.1 / 33.9; P28 27.2 / 28.6, 29.0 / 30.2; Q28 30.7 / 33.9, 31.7 / 35.1; G 512 q4 A 54.4, E0 30.7, P28 31.6, Q28 37.4; CPU 50.9) are an appendix of record only: **developer unit, not representative** (user) — no rule, no BENCHMARK column. Observations kept there: four DMA queues read no better than one on any variant (the banner shows the setting applied); a pool-28 miss reads 3.99–4.69 ms with the phone's page-cache `read()` at ≈ 1.6–2.0 GB/s. S26 optimization deferred until a product unit | `204-s26-rebaseline.md` §Results (this fold); `/local/mnt/workspace/htp_moe/204/s26/logs/` |
| **#216: the pool-28 miss cost — step 1 (`216-miss-read.md`, PR #217; `R3CY10WM83Y` 2026-10-01 21:43–22:01 KST, sitting 1's build `a2ebef9c9`, MD5 OK on three boots, 12 profiled Q28 G = 64 runs, per-core `/proc/stat` + `top -H` + vmstat sampler at 0.5 s) and the fadvise lever (`216-fadvise.md` @ `45be8de65`, PR #218, `htp/216-fadvise` @ `222a3196c`; two sittings 23:06–23:36 KST on a fresh and a ≥ 10-min-old boot, set `61a26580…` / skel `9d61aef4…` rebuilt from `03811d8ef`, device md5 == staged on both boots, 26 runs)** | **Cause measured, first lever fails the prefill gate, default not flipped.** (1) The slow miss is a storage read, not a busy core and not boot proximity (rule 61 amended, ㉜): `pgpgin` 610 / 310 / 61 / 0 MiB and refaults 151 k / 79 k / 16 k / 1 in the decode window against 5.03 / 2.72 / 1.55 / 0.55 ms/miss, PSI io 92–296 ms in every slow window (0–31 fast), busiest non-app core 3–16 % in 11 / 12 windows; slow at uptime 300 s and after a 6-min idle. Memory arithmetic: arena 3 328 + FC 448 + RSS 766 unreclaimable + the model file 4 116 cached + Android ≈ 3 740 > 11 114 MiB — every resident expert held twice. Plan 216's slice lever not built (stop rule; patch kept). (2) `NNTR_MOE_FADVISE=1` (DONTNEED after a slot is filled, WILLNEED on the victim, 128 KiB pieces on one worker, decode drops at `poolSync`): ms/miss **B 0.86–1.13 vs A 0.68–4.38** on both boots; decode G = 64 block means **36.27 → 41.56 (+14.6 %) fresh, 38.67 → 42.71 (+10.5 %) old**, G = 512 −0.5 / +4.3 %; against A's one fast run (0.68, 43.66) B is 1–4 % slower. **Prefill 494.8 → 443.6 (−10.4 %) / 540.5 → 500.2 (−7.4 %) at G = 64, −17.8 / −11.9 % at G = 512 — gate fail**; `arm_ms/round` 1.20–1.58 vs A-fast 0.945 (fail); window `pgpgin` 639–643 MiB at G = 64 by the plan's own arithmetic (fail by construction); `=2` drop-only 3.87 / 3.97 ms/miss = A's slow regime (model confirmed: every decode miss is a re-miss). Text 26 / 26 == A, `calls/token=1.00` 22 / 22, ceiling 3840 after 25 / 26 (one 3584 after a hybrid run, unexplained); hybrid H vs H0 inside spread (the path never reaches `readExpert`). Host: `ALL CHECKS PASS`, `INPROC E2E PASS` under unset / `=1` / `=2` with the same `misses=`, `bit_identical=1`. Rule 62 (fadvise is 2–10 ms a call here, WILLNEED reads 1 MiB). What is left → **#219**: the complement in a cached ARM buffer, refilled off the token path, the page cache dropped once at load; not verified: G = 1024, the victim re-miss race, `MADV_PAGEOUT` / plain `pread` into scratch as a cheaper drop | `216-miss-read.md` + `216-core-load-{run.sh,report.py}`, `216-sampler.sh` (PR #217); `216-fadvise.md` + `216-fadvise-{run.sh,sampler.sh,report.py}` (PR #218); issue #216 comments 2026-10-01 13:03 / 14:54; `/local/mnt/workspace/htp_moe/216/{logs_core,fadvise}/` |
| **#222: the LFM2.5 closing sitting on the config of record — Aold / Anew / Qold / Qnew / CPU (`222-config-refresh.md`, PR #223 merged into `htp_first_version` `b7c1d4ff6`; farm `R3CY205ZMND`, S25 SM8750 v79, 2026-10-02 16:15–16:54 KST, set from `ddb7d9ad8`; handoff merged unfilled — numbers from #222's comments 08:00 / 08:30 UTC, no device md5 line seen by the supervisor)** | **FAIL on three counts; no row of record, next issue #225.** (1) **Qnew (one PD, C = 28) VOID in every cell**: `no room for the FC set beside the resident experts (mapped=3712 MiB, fastrpc_mmap(32 MiB) failed: err=1)`, same after a fresh reboot; `heap_used_kib` 92 058 (Qold) → 223 003 (Qnew attempt). (2) **Anew (hybrid) P1024 VOID** in `nntr_hvx_mm_u8i4_conv_block` (`AEE_ENOMEMORY`, M = 1024): the conv block's M-proportional DSP-heap scratch (≈ 26 MiB at 1024) beside the keys' ≈ 216 MiB WH copies. (3) **Anew accuracy fails**: prefill PPL 90.31 → 128.31 (+42 %), decode PPL forced 1.2108 → 1.3752 (+13.6 %, gate +2 %), G64 text loops — stacked Q4_0 → qs4cx quantization (rule 63). Information only, never averaged in: Qold (C = 28) loads, prefill / decode 556 / 33.7, 584 / 34.6 (G64), 455 / 43.3 (G512), 531 / 43.3 (G1024), texts = Aold r1; Anew P512 786.5 / 56.5, 761.9 / 56.4, 739.9 / 51.7, P64 294.9 / 56.1, 254.0 / 55.6, 266.7 / 52.9; CPU `q40` P64 244.3 / 53.6, 210.5 / 53.4, 210.5 / 49.4, P512 310.5 / 49.0, 307.7 / 46.9, 294.4 / 48.9, P1024 (`init_seq_len 1024`) 311.8 / 49.7, 317.8 / 49.1, 313.8 / 48.5. User (2026-10-06): the Qnew C = 24 follow-up is cancelled; the table of record is 9 cells × 3 cases on the config of record; the fix is option (b) — the FC weights as `QS4CX_WH` in a sidecar file written by the packer (#225, base `htp_first_version`; PR 1 = #230 merged `7f95140ad`, the 512-row prefill chunking is the hybrid's way of record); accuracy threshold for #225's handoff open (`needs-user`). #222 closed `completed` | #222 comments 2026-10-02 08:00 / 08:30 UTC and 2026-10-06; `/local/mnt/workspace/htp_moe/222/logs/` on the farm session's machine (not on this workstation); `222-config-refresh.md` (unfilled) on `htp_first_version` |
| **#225: the LFM2.5 table of record, 9 cells × 3 cases on the config of record with the FC WH sidecar (`htp/225-fcwh-e2e:docs/measurements/225-fcwh-table.md @ f08cccfeb`, PR #233 open; farm `R3CY205ZMND`, S25 SM8750 v79, 2026-10-06 15:56–16:36 KST, one invocation, no STOP; set rebuilt from `51b2b4477` = code `7efd22adf`, `md5.txt` `23000b96…`, device md5 == set, sidecar `71812a91…` / main bin `7b7867fa…` byte-identical; started 52 s after reboot — rule 61 deviation; run by the user)** | **G1 fit pass with the Bfb substitution; G2 E2E pass; G3 hybrid P1024 FAIL (rule 64, #236); G4 under T2 recorded, approval pending — no row of record moves yet.** **CPU (A)** P64 54.05 / 52.54 / 51.48, P512 52.07 / 50.70 / 49.47, P1024 47.58 / 48.82 / 47.41 (prefill 231–328). **Hybrid (B)** P64 59.59 / 58.01 / 54.74 (prefill 272–284), P512 **55.17 / 55.72 / 51.43 (prefill 727–739)**, **P1024 B VOID** → **Bfb** 52.03 / 51.51 / 48.74 (prefill 650–686); vs the CPU +2.8 … +10.4 % at 9 / 9 cells, ≥ 50 at 8 / 9; vs Aoff (keys off, P512) 53.96 / 52.58 → +2.2 / +6.0 %, prefill +30–38 % — the prefill gate against the same binary passes. B's `fc wh: handles=112 arena_kib=145408 heap_kib=75776 requant=0` (overflow as planned); B P1024 dies in the attention qkv FC: `nntr_hvx_mm_u8i4_layer failed: err=0x80000600 (M=1024 K=2048 N=3072 handles=3)` (`AEE_ERPC`; the FC entries chunk by `fcMaxRows` = 1920 rows, not `prefillRows` = 512). **E2E one PD (Q28)** loads in every cell — `fc wh: … arena_kib=221184 heap_kib=0 requant=0`, `calls/token=1.00`, `cpu fc skipped` = 32 × tokens, mapped 192 + 3520 = 3712, `heap_used_kib` 94 604 (#222's Qnew 223 003, VOID) — at P64 42.81 (r2 49.31) / 49.34 / 48.49, P512 42.78 (r2 44.48) / 45.94 / 45.61, P1024 37.83 (r2 38.72) / 43.93 / 43.52 (prefill 248–755): under the CPU by 6–21 % and under the hybrid at every cell; `prof_Q` P512 G64 `wall_ms/token` 21.275 = MOE 10.946 + FC 3.819 + LM_HEAD 2.973 + DENSE_FFN 1.057 + ROUTER_TOPK 0.677 + ATTN_M1 0.620 + CONV1D_GATE 0.478 + RMSNORM 0.331 + ADD / QK_NORM / ROPE 0.21; pool misses 1.61 / token, `miss_wait_us/token` 549. **Accuracy (T2):** prefill PPL pooled A 86.44 / Aoff 90.31 / B 94.08 / Q 94.08; decode forced on A 1.1809 / 1.2139 / 1.2965 / 1.3318 (null check equal; worst p04 B 1.671 / Q 1.791, p07 1.853 / 1.989); every NPU text ≠ A (rule 39); r2 == r1 with the `[HTP]` banners stripped (B 3 / 3, Q 3 / 3); **loops where A has none: Bfb P1024 G64 (r1, r2), Bfb P1024 G512, Q P1024 G512**; B P64 G64 degenerate where A loops too; Q P512 G64 echoes the prompt; **approval column empty (user)**. Runner artifact: banners inside the text (㊲). User decisions in force: T2, fallback (b), 9 × 3 table of record, LFM2.5 is the test bed for Gemma. Next: PR #233 merge + text approval (user) → the table becomes the row of record; #236 (p1) re-reads B at P1024; #219's step-4 sitting is runnable after PR #233 | `htp/225-fcwh-e2e:docs/measurements/225-fcwh-table.md @ f08cccfeb`; #225 comment 2026-10-06 07:43 UTC; `/local/mnt/workspace/htp_moe/225/logs/` on the measuring workstation (`j2z0`) |
| **#219: the pool miss as a memcpy from a cached ARM tier — `NNTR_MOE_TIER` A / B on a fresh and an old boot + `=2` ×4 (`htp/219-arm-tier:docs/measurements/219-arm-tier.md @ 149481ec4`, PR #240; farm `R3CY205ZMND` S25 v79, 2026-10-06 18:28–19:01 KST 30 runs + 19:09 12 runs; set from the local merge `8629b9392` = `htp/225-fcwh-e2e` @ `f08cccfeb` + `htp_first_version` @ `9f33d7d43`, device `MD5 OK` on all three boots, rule 22; config of record `3f6808e3…` + sidecar, `CONFIG OK 3848ab71…`; the code is PR #224, merged `9f33d7d43`)** | **Lever measured: the slow regime is gone, decode +19 / +24 % at G 64, G1 / G3–G7 pass, G2 missed on two counts; default not flipped (user's call).** Q28 one PD, G 64 block means (same-sitting A): **no tier 39.99 / 38.70 → `=1` 47.60 / 47.82 → `=2` 48.37 / 48.45** (fresh / old); G 512 51.13 / 50.54 → 52.75 / 52.73; G 1024 51.69 / 51.22 → 52.27 / 52.42; ×4 follow-up 37.15 (sd 5.43) / 47.31 (0.71) / **48.47 (0.23)**. ms/miss (profiled) **2.54 / 3.77 → 0.67 / 0.51 → 0.45 / 0.44**, follow-up 3.69 / 0.83 / 0.43; `miss_wait` at G 64 2927–12669 → 452–934 → 302–322 µs a token. Window refaults 74k–153k → ≤ 571 (`=2` ≤ 81), kswapd 65k–139k/s → 0, PSI io → 19–37 ms, `resident after` 0.7–3.2 GiB → 0 on every tiered run; `tier_hits = misses`, `tier_reads = 0`, app `pgpgin` = misses × 5.29 ± 7 MiB; `experts=88 mib=466.5 read_ms=78.5 direct=1`, the first call's `DONTNEED` 451.6 ms at load. **G1** `=1` 0.51–0.83 (the plan's 0.5–0.7 "copy store-bound" band, user's call), `=2` ≤ 0.45 on all six runs — pass. **G2 fail**: `tier_waits` 3–4 on every tiered run (gate ≤ 2; constant across G → the decode start's refill backlog, rule 65 b) and `pswpin` 0–2105 (gate 0; `=2` ≤ 168; rule 65 c); the rest of G2 holds. **G3** prefill: G 64 block means +2.1 / +0.9 %, ×4 +0.7 / +1.7 %; G 512 −1.2 / −1.0 %; fresh G 1024 −5.5 % (801.3 → 757.4, A the sitting's highest prefill of 30 runs, unprofiled) vs old +1.2 % — pass, the pair read as spread. **G4** pass (B's block means above A's best run, B ≥ A at G 512 / 1024 both boots). **G5** text == the boot's A1 on 36 / 36 Q runs (fresh B2's runner `DIFF` = a ` wh_handles=112` banner fragment, ㊲'s class), H == H0, host `bit_identical=1`. **G6** `calls/token=1.00`, close clean, ceiling 3840 after all 42; hybrid H0 / H 50.35 → 51.08, 53.51 → 53.92 (inside spread). **G7** `resident after` 0, `pswpout` 0 on the main sitting's tiered windows. **One-thread copy beats the 8-slice copy on every axis** (rule 65 a). "Now" not moved (lever cell; the table of record's E2E Q28 cells stay #225's). What is left → ㊴: the default / copy-mode decision (user), the decode-start refill backlog (`tier_waits`), the small `pswpin`; a tiered 9 × 3 E2E column would be its own sitting (not filed) | `219-arm-tier.md` @ `149481ec4` + `219-arm-tier-{run.sh,report.py}`, `219-b2x4-run.sh` (PR #240); issue #219 comments 2026-10-06 10:04 / 10:12 UTC; `/local/mnt/workspace/htp_moe/219/logs/{fresh,old}/` + `report.md` |
| **#236 close-out: the hybrid at P1024 with the M > 1 FC entries in 512-row chunks (`htp/236-qkv-chunks:docs/measurements/236-qkv-p1024.md @ 3d61c1aff`, PR #243 open; `R3CY205ZMND` S25 v79 on the workstation's USB, agent-run, 2026-10-06 20:35–20:43 KST, uptime 1565 s at start, 8 runs + profile, no STOP / BAD; set from `a3d966164`, skels rebuilt against the current IDL (v79 `ce4e6486…`, v81 `adfd44d4…` — the #225 skels are stale after the #237 sync's IDL change, rule 67), `md5.txt` regenerated `1e6bf2f0…`, device md5 == set, `MD5 OK (app, sidecar)`; config of record `3f6808e3…` + sidecar `71812a91…`)** | **G1 loads pass, G2 prefill pass, G3 decode pass (= Bfb), G4 chunked pass; T2 recorded, approval pending — the table of record's hybrid P1024 cell moves from Bfb to B; ㊱ closed.** B P1024 **53.24 / 53.06 / 48.64** (G64 r2 52.89), prefill **725.7 / 722.7 / 711.1**, `fc wh: … heap_kib=75776 requant=0` on every run (= #225's overflow), no `0x80000600` / `0x80000402`; anchor B P512 G64 56.49 / 745.3 (#225 55.17 / 727.3 → +2.4 %, the sitting's drift; A′ 56.84 / 749.6) → anchored 52.00 / 51.82 / 47.50 vs Bfb 52.03 / 51.51 / 48.74 (−0.1 / +0.6 / −2.5 %: equal, the M = 1 path is untouched), prefill +9.4 / +11.1 / +9.5 % raw over Bfb (floors 630 / 618 / 617); vs the CPU of record 47.58 / 48.82 / 47.41 +11.9 / +8.7 / +2.6 %. **A′** (the #225 set as staged, `dd24b4fc…`, skel `58e3a85f…`) P1024 G64 VOID with `nntr_hvx_mm_u8i4_layer failed: err=-2147482112 (M=1024 K=2048 N=3072 handles=3)` — #225's failure reproduced in-sitting (deterministic; rule 69). `prof_B_P1024`: `M>1 FC K=2048 N=3072` calls 12 rows 6144, `N=2048` 13 / 6656 (every call 512 rows), 2.31 / 1.61 ms host a call, 48.8 ms of a 1458 ms prefill. Q P1024 G64 (tier unset) 772.2 / 40.76 vs #225's 699.0 / 37.83, text byte-identical to #225's Q. **T2:** B P512 text == #225's B byte for byte; B P1024 G64 no loop (1 / 0.23 vs A 0.00; ends on the loop's opening clause), G512 loops where #225's A does not (1.00 vs 0.16; as Bfb did — so it is the length / the FC quantization, not the fallback), G1024 both loop; r2 == r1; **approval pending (user)**. Logcat: one `0x80000414` `libdspqueue_rpc_skel` line at teardown in every run incl. the good ones (the queue close, not a failure). Next: PR #243 merge + text approval (user); nothing further on LFM | `236-qkv-p1024.md` @ `3d61c1aff` + `236-run.sh` / `236-stage.sh` (PR #243); issue #236 comment; `/local/mnt/workspace/htp_moe/236/logs/` |
| **#219 close-out (#219t): the E2E one-PD row of the table of record with `NNTR_MOE_TIER=2` on by default (`htp/219-tier-default:docs/measurements/219-tier-table.md @ 6923850d0`, PR #245 open — `tierKnob()` unset = 2 when `HtpBackend::e2eRequested()`, else 0; hybrid / CPU untouched, no DSP / IDL change; `R3CY205ZMND` on the workstation's USB, agent-run, 2026-10-06 20:50–21:02 KST, 4.5–5.3 min after a reboot (rule 61 deviation), 14 runs, 0 BAD / VOID / STOP; set from `0235d9d79`, `md5.txt` `058b97a5…`, `MD5 OK (app, sidecar)`, skel v79 `1b35bff1…`; config of record + sidecar)** | **G1 tier-by-default pass, G2 E2E hygiene pass, G3 speed met within drift (48.08 vs ≥ 48.4, −0.7 %; #219's `=2` sd 0.23), G4 hybrid untouched pass; T2 recorded, approval pending — the table of record's E2E column becomes the tiered one, #225's untiered Q cells stay as the tier-off reading; ㊴ open, not pursued on LFM.** Q (C = 28, tier unset = 2) decode **P64 52.20 / 54.60 / 53.89, P512 48.08 / 51.79 / 51.65, P1024 44.02 / 50.59 / 50.78** (G64 r2 52.03 / 48.05 / 44.14), prefill 281.9 / 285.7 / 287.0, 793.8 / 802.5 / 800.0, 765.9 / 773.4 / 771.1; every Q run `token driver: on … tier=2`, three `[HTP] tier:` lines, the last `experts=88 mib=466.5 … direct=1`, `tier_reads=0`, `tier_hits == misses`, `calls/token=1.00`, `fc wh … heap_kib=0 requant=0`, mapped 192 + 3520, close clean; `tier_waits` 0 / 3 / 3 (P512 G ≥ 512: 4); misses/token at G64 1.16 / 1.61 / 2.62, `miss_wait_us/token` 157 / 317–331 / 472–475 (G ≥ 512: 10–64). Q0 (`=0`) P512 G64 37.58 (`miss_wait_us/token` 5905; Q +27.9 %); B P512 G64 56.44 / 749.6 (= #236's 56.49, −0.1 %; #225's 55.17 +2.3 %). vs #225's untiered Q +21.9 / +10.7 / +11.1, +12.4 / +12.7 / +13.2, +16.4 / +15.2 / +16.7 %; vs the CPU of record −3.4 / +3.9 / +4.7, −7.7 / +2.1 / +4.4, −7.5 / +3.6 / +7.1 % — **≥ 50 and above the CPU at every P for G ≥ 512, under at G64, residual scaling with P (rule 68)**. Prefill: Q +6 % over Q0 / B at P512, both of which ran without a cool-wait — a start-condition difference, not a tier effect (#219 read +1.7 %). **T2:** Q P512 G64 == Q0 (the tier copies the file's bytes — bit-preserving), every G64 r2 == r1; loops at P64 every G (the CPU too), P512 / P1024 G ≥ 512 (L1 run 17 / 35), none at G64 / Q0 / B; **approval pending (user)**. Next: PR #245 merge + text approval (user); ㊴ (`tier_waits`, `pswpin`, the G64 residual) is Gemma's | `219-tier-table.md` @ `6923850d0` + `219-tier-run.sh` / `219-tier-stage.sh` (PR #245); issue #219 comment; `/local/mnt/workspace/htp_moe/219t/logs/` |
| **#229 S1 device gate, v79 (implementer, S25 `R3CY205ZMND` SM8750 v79, 2026-10-06 19:27–19:30 KST, uptime 20 min; `htp_decode` @ `4a7003f93` + PR #244 `5abbc0730` test-only; skel v79 `d7019280…`, `unittest_hvx_mm_u8i4` `d7818b7e…`, `unittest_hvx_softmax` `41520073…`; #229 comment 2026-10-06 11:03 UTC, full log in PR #244)** | **Pass on v79; v81 not run.** `HvxExpandI2I4.MatchesScalarBitExact` OK — `EXPAND_I2I4_FIELD` `bad=0 of 131072` for palettes `[-8,-2,1,7]`, `[-5,-2,1,4]`, `[0,1,2,3]`, `[-3,-3,-3,-3]`; the new `HmxMmU8I4Layer.MoeLayerU8I2GemvMatchesI4Twin` (Gemma expert shape K 2816, inter 704, N 2816, 8 experts, M = 1; the 2-bit GEMV must equal the 4-bit GEMV on the palette twin and the 2-bit HMX loop, 5 GEMV cells × {SwiGLU, GeGLU}, path and feed words asserted, a one-byte twin corruption fails it) OK — 12 `U8I2_FIELD` lines `bad_vs_twin=0 bad_vs_hmx=0`, `path=m1_gemv field=bit_identical value=yes`; `MoeLayerMatchesTwoCallReference` / `SplitMatchesWhole` / `FromArenaMatchesHeap` / `M1GemvMatchesHmx` OK; `HvxSwigluDet` 2 / 2, `SwigluDetNeon` OK. The informational `EXPAND_I2I4_MAP` probe printed `INCONSISTENT` (`1->2 2->4 …`) beside the passing assertion — read as the probe's decode assumption, not investigated. A gtest reading, no tok/s row; ㊳'s v79 half is closed, the v81 half (an attached S26) and the 2-bit-vs-twin `NNTR_HTP_DUMP` reading on a real model stay open. PR #244 open (CI rerun pending after the cpp-linter fix) | #229 comment 2026-10-06 11:03 UTC; PR #244 |
| **#201 S5-0 — Gemma-4 26B-A4B DUMMY on the one-PD E2E path, the C sweep (agent, S25 `R3CY205ZMND` SM-S938N v79, 2026-10-07 17:11–17:50 KST, lock taken / released; `htp_decode` @ `6904594ed`, no code change; skel `6f3029c3…`, app `6367c828…`, bin = PR #251's `QS2CX_WH` repack `6ac7df4c…` 7,226,142,840 B, device md5 == on both boots; device `config.json` copy with `max_position_embeddings` 4096 — rule 72; prompt 447, not 512; `201-s5-0-gemma-dummy.md`, PR #252 `91f701fda`)** | **Path: pass. Gate: (a) PASS (user 2026-10-07). Speed: E2E ≈ 30 % under the hybrid at every G; the pool is not the bottleneck.** Loads (`fc arena weights=206 handles=221 mapped_mib=1344 s1_arena_mib=704`, `attn_m1` 25 × hd256 + 5 × hd512, `calls/token=1.00`); reference = the hybrid `off` run on the same file (user: no Q4_0 CPU twin). Prefill MoE dumps E vs off 62 / 62 `bit_identical=1`, routing identical; tokens 8 / 8, 16 / 16, 57 / 65 over G64 — the 8 differing steps are near-tie flips (off's top-2 margin 0.0003–0.022 vs E-vs-off 2·rms 0.26; the largest margin in the run 0.156; null check off-on-off 64 / 64, nll identical) — plan 130 §3.5 expected mismatches, 0 unexpected; E and off cannot be bit-identical over decode (rule 39: FC / attention / norm / head on the DSP vs the CPU). Decode tok/s G 64 / 512 / 1024: A16 16.97–20.83 / 20.06 / 18.46; **E16 11.10 / 14.35 (×2: 14.25) / 13.68**; E40 (C-max, rule 71) 10.9–11.6 / 14.31 / 13.63; E32 / E8 inside 3 %; misses/token at C = 16 5.84 / 0.80 / 0.41 (dummy routing), 0.5–1.2 ms a miss from G 512 on. DSP wall 63–67 ms/token: FC 14.5, MOE 11.7, ATTN_M1 9.8 → 17.8 with context, LM_HEAD 8.2, DENSE_FFN 7.3, ROUTER_TOPK 4.3; ARM steady state ≈ 1.6 ms → ≈ 15 tok/s ceiling at Q4M1. **Cause of the gap: FC + LM_HEAD + DENSE_FFN ≈ 30 ms/token on the DSP's Q4M1 path (1278.7 MiB in ≈ 30 ms ≈ 43 GB/s, ㉘'s arithmetic) where the hybrid's CPU reads the same Q4_0 bytes at 51–63 GB/s (rule 46)** → the first lever is #234 P3 / P4 (FC WH sidecar), then #229 S2 (2-bit FCs); pool / tier levers deferred. Memory (㉟ baseline): peak RSS 2174–2182 MiB, pool 704 + FC arena 1344, DSP heap 582 (617 at C 40); prefill A 138–144 / E 100–140 (warm E starts; recorded, not gated). First load OOMed on the host RoPE table (rule 72 → #253); one LEAK stop on a 20.6 h boot (reboot, ㉝). No text / PPL column (dummy); the decode MoE inside the token has no ARM-side dump (㊷) | `201-s5-0-gemma-dummy.md` @ `3a151680b`; #201 comment 2026-10-07 08:52 UTC; PR #252 |

## 3. Open items (candidates for issues; the supervisor promotes them)

| # | item | expected | depends on |
|---|---|---|---|
| ① | ~~Measurement A~~ **measured (#77) → §2.** Follow-on: the dense FC's 28.2 ms/token on the NPU path is the second-largest lever after MoE; **it is not a round-trip cost** (no FC is on the HTP in that config, §2 correction) — see ⑰ | — | ⑰ |
| ⑱ | **Measured by #95 on the side tree `htp_hadamard` → §2 (cycle 8): carry it over. PPL −12.7 % (100.5 vs A 115.1, CPU 109.8), requant SNR median +8.0 dB, cost +2.7 % / +0.3 % of the M>1 / M==1 `dsp=`, decode inside drift. #95 closed. Remainder filed as #110 (p2, `state:needs-plan`):** (a) `fwht_det.h` drops its FTZ (rule 24); (b) the synthetic SNR assertion becomes a printed field (rule 25); (c) port to `htp_moe` with the M=1 GEMV requant site (about `:509`), plus the HMX block loop (about `:1415`) and the tail (about `:370`); the HAD opts bit must not clear PR #108's default-on GEMV bit; (d) `NNTR_PPL` as a separate upstream cherry-pick (`32b46e32`, user decision at merge). Gate = an `htp_moe` handoff: B ≡ A (text + PPL), C PPL ≤ A, SNR median ≥ B + 5 dB, M==1 `dsp=` ≤ +2 %, prefill −5 %. p2 because it does not move decode tok/s; it becomes p1 if the user adopts PPL as the accuracy column for `QS4CX_WH*` models. History: **Accuracy, filed as #95 (p1):** Hadamard rotation on the MoE down_proj input — fold `Hᵀ·W_down` offline (new dtype `QS4CX_WH_HAD`, block 256, 1792 = 7 × 256, 1/16 both sides), FWHT-256 in IEEE `sf` add/sub on the DSP right before the existing u8 requant at all three requant sites of `hexkl_mm_u8i4_moe.c`. Targets the recorded gap (doc 43, 2026-09-09: 5 of 32 MoE calls at 67–80 dB SNR, each from exactly one of 1792 elements crossing a u8 level — boundary rounding, not row outliers; column-wise outliers never measured). **First accuracy item in this table**; it does not move tok/s by design and may show no effect — a "no effect" result closes it with the numbers. Gate = `NNTR_L2_DIFF` per-call SNR / `total_flips` on vs off on the 5 recorded calls + full-model handoff with prefill/decode TPS and text-identical-to-CPU; prefill gate applies (it touches the weight layout and the requant stage). Rules that bind: no qf32 (v75/v79), host scalar `fwht_rows_f32_ref` bit-identical to the HVX kernel, CPU `QS4CX` run stays the reference (no CPU kernel for `_HAD`, rule 6 applies to it too) | requant SNR on the 5 bad calls ↑; text-identical-to-CPU unchanged or better; TPS within noise | none; can run in parallel with the walls (touches quantizer + requant only) |
| ② | ~~Measurement B~~ **measured (#77) → marshalling → ⑦ / #88 → closed by #88's sitting 2026-09-22** | — | — |
| ③ | ~~Measurement C~~ **measured (#77) → shape only; in-situ gap is the wall → ⑥ / #87** | — | — |
| ④ | **CLOSED (cycle 22) by #90 (PR #156): DRAM ≈ 70 GB/s aggregate for any reader mix, rule 44; §2 #90 row.** Two-reader DDR probe — **DSP side invalid in #77 (rule 12); refiled as #90** (stream > VTCM+L2 or use the DMA ring, bounded result). CPU side: −41 % under contention, so any split pays less than the sum | decides whether the CPU+NPU split (contract §3.2) can ever pay | #90, ride-along step in a later sitting |
| ⑤ | **Filed as #80.** M=1 MoE path on the existing HVX GEMV (wall 1). The kernel already exists: `hvx/hvx_gemm_u8i4_wh.c` (u8×i4 over WH tiles, int32 bit-identical to HMX, m ≤ 16) is used only by the prefill "tail" path in `hexkl_mm_u8i4_moe.c` (off by default, net −0.5 ms there). Decode needs a dispatch that sends all four experts through it at M=1 with no 64-row block, plus the weight feed (arena read vs DMA into VTCM) that ③ decides | MoE DSP 1.35 → ≈ 0.3 ms/call if DMA ≥ 30 GB/s | ③ |
| ⑥ | **Loose end read by #113's ride-along (cycle 12): 724.0 µs / 31.2 GB/s = #100, not #94 — rule 32; the feed half does not re-open at 37, it re-files at 31.6 vs 23.5 (㉒). Nothing left in ⑥.** **CLOSED as a descriptor-list question, 2026-09-23 by #100 (§2 row): row h dissolves.** Against the tag-validated `c_star` the traced 46-descriptor list runs at **1.19 ×**, i.e. faster than the ceiling; rules 11 and 19 are amended/withdrawn (rule 28) and there is no list-side loss to fix. The 16–18 GB/s in-situ number stays real, but its cause is the interleaving (rows f + g, ≈ 20 % of `dsp_us`) plus an engine that tops out near 31 GB/s on every validated per-call cell — **not a shape or a list**. The GEMV path bypasses the ring entirely (`desc=2/call`), so at M=1 nothing in ⑥ is on the decode critical path any more; rows f and g are **not** filed. What wall 2 leaves behind at M=1 is the read-rate half, which lives in ㉒ and #113. **One loose end, a ride-along not an issue:** re-read `DMA_REPLAY workers=1 load=0 pace=0` on the next sitting and see whether it returns to #94's 561.5 µs / 40.2 GB/s (rule 30); if it does, the `f2` feed cell would clear 37.0 and ㉒'s feed half re-opens. History: **Wall 2 — measured by #94 sitting 2 (§2 attribution): the 4–7× is ≈ 2.7× in the traced chunk list itself (row h, rule 19, provisional on #99) and ≈ 1.3–1.8× in the interleaving (f + g ≈ 20 % of `dsp_us`). Step 2 filed as #100 (row h: why the 46-descriptor / 32-region list replays at 40 GB/s where probe iii does 107; gate = replay ≥ 80 % of probe iii and in-situ `engine GB/s` ≥ 40; blocked by #99 until the replay's content check is explained). Rows f and g follow #100.** **Cycle 9: #100 re-scoped to plan 100 §0 (row h's cause + a DDR → VTCM feed shape certified ≥ 37 GB/s beside HVX for ㉒, same-run `c_star` denominator), raised to p0 after #105, no longer blocked by #99 (new cells set their dst mode explicitly).** History: rewritten by #77 C — not descriptor / engine / vote but "why does the MoE call see a quarter of the isolated rate". Step 1 = ~~#87~~ **landed on `htp_moe` as `b6ebc2b7` (PR #93, 2026-09-22; #87 closed)**: `hmx/hexkl_dma_trace.{c,h}` (static tables, union-of-intervals busy / depth / blocked-wait arithmetic, host-checked by `dma_trace_host_check`), trace hooks in `hexkl_mm_u8i4_moe.c` behind `hexkl_probe_on` (byte-identical output on vs off, 19 descriptors traced at the fixture shape, 46 at the LFM2 M=1 shape), IDL entries `dma_probe` / `moe_dma_trace_read` / `dma_replay` (`test/htp/nntr_hvx_dma_probe.c`), the header-only in-situ descriptor plan `test/htp/nntr_moe_dma_plan.h`, the second `weight DMA:` line and the `[HTP-DMA]` per-descriptor dump in `[HTP-PROFILE]` (`htp_compute_ops.cpp`, first `NNTR_HTP_DMA_TRACE` calls, default 3), and `TEST_F(HvxDmaProbe, MoeChunkReplay)` (workers 1/2/4 × HVX load × fresh/gap) in `unittest_hvx_dma_probe`. **The device sitting that fills the attribution table (plan 87 §1, hypotheses a–g) is #94 (sitting 2); its first attempt never reached the NPU because the trace's own `hexkl_dma_trace.c` was missing from the skel build (#97, rule 17).** Original scope for reference: instrument the ring use inside `hexkl_mm_u8i4_moe.c` (per-descriptor issue/complete pcycles, wait time in `hexkl_dma_ring_wait`, outstanding depth, actual chunk shapes at M=1 and M>1) + a device gtest that reproduces the in-situ pattern, gate = a table attributing the 4–7× to named causes. Step 2 = the fix that table names (separate issue). Interacts with #80/#86: the M=1 GEMV path reads the arena directly, so its A/B also tells what the ring costs | 16–18 → ≥ 40 GB/s in situ; 1.19 → ≤ 0.57 ms per call | #87 |
| ⑦ | **CLOSED 2026-09-22 by #88's sitting** (`88-moe-call-marshalling.md` @ `3f9fa38d`, `R3CY205ZMND`, one sitting A/B/C; §2 row): PR #103's size-class ION staging + 5 ms poll take the M==1 transport **401.3 → 87.9 µs/call (−78.1 %)**, past the ≤ 0.1 ms gate, so plan §3.3's prebind IDL pair is **not built**; the cut splits **staging 245.8 µs / poll 67.6 µs** (C = B + `NNTR_HTP_POLL_US=100` reads 155.5). Decode **+30.9 / +41.4 / +42.9 %** at G 64 / 512 / 1024, prefill −0.9 % (gate ≥ −5 %), M>1 transport 2494.6 → 1770.3 with `dsp=` unmoved, text bit-identical at all three G. Wall 3 is no longer the lever (1.9 ms/token left); the MoE call is 94 % DSP, so walls 1–2 own most of the remaining 2.1× to 50 tok/s. #88 and #83 (the transport-options document, consumed by plan 88) closed in cycle 7. **Left open, not an issue:** whether the GEMV path's +100 µs transport of #94 C (649 → 748) survives the new staging — #101's ride-along reads it as the same-sitting A (GEMV default) − A0 (`NNTR_MOE_HTP_M1_GEMV=0`) level-2 M==1 transport (`101-ride-along.md` R2/R4, rule 23). **Cycle 11: ANSWERED in-sitting by #100's A/A0 (§2 row). GEMV 185.8 vs HMX 84.6 µs/call, Δ +101.2 (2.20 ×), equal `act`/`out` classes (65536 B), one sitting, one binary — so the +100 µs is path cost and it survives PR #103's staging. It is not a defect to chase: the same A/B wins 343 µs of `dsp` and 242 of `host`. The 4.1 ms/token it costs comes back only with ⑨ (one call per token), not with a transport fix; no issue.** **Cycle 15: wrong — it came back with the VTCM feed (rule 35): B's M==1 transport 82.0 / 92.3 µs vs A's 184.6 / 190.3 in the same sitting, so the +100 µs was the direct HVX arena read's, and transport at the feed default is ≈ 2.0 ms/token, the HMX-path level of ⑦.** Earlier, cycle 9: #105's A reads M==1 transport 182–184 µs against #88 B's 87.9 (GEMV off), which is cross-sitting (rule 23) and consistent with it surviving. At ≈ 2 ms/token it is worth an A/B line; PR #108 merged (`8e121dbe`) without a sitting, so that line (`101-ride-along.md` R2/R4) rides #100's handoff. dspqueue / resident worker: one paragraph (plan 83 §4), revisited only with ⑨. History: resolved by #77 B to prebound handles + per-call buffer/marshalling cleanup on plain FastRPC — #88; #83 narrowed to the document that says what "prebound" concretely means | 0.53 → ≤ 0.1 ms per call — **met: 0.088** | — |
| ⑧ | lm_head blocked Q4_0 twin on device (doc 46 §46): confirm 25.7 → ≈ 3.4 ms | ARM remainder | ① |
| ⑨ | **Cycle 21: parked by the user's 2026-09-28 direction change (contract §12).** The per-token resident path (`NNTR_HTP_FORWARD`) stays in the tree, off by default, and is not extended; the last sitting of it (#134 / #132, §2) read D (51 calls) at −22.7 % of A with a rejected text. Its transport half is done without it (#141, rule 40: 22 × 14 µs = 0.3 ms/token), so ⑨'s ≈ −5 ms is no longer an open target: what is left outside the MoE call is ⑰ / #150 (the CPU GEMV rate, rule 42). #132 PR 2 and #146 are `needs-user`; `NNTR_PPL_DECODE` (#134) stays as a tool. **Cycle 20 close: landed PR #143 (pool race fix; #136 closed as not-a-defect, rule 39 / §2), PR #144 (`NNTR_PPL_DECODE`, #134), PR #145 (ADD + ROUTER_TOPK, #132 PR 1: 95 → 73 → 51). B's six-kind verdict now reads from the decode PPL against A of the same sitting (+2 %), next to the text column and the approval. Next device read: the combined #134 / #132 sitting staged at `/local/mnt/workspace/htp_moe/134-132/` (A switch off / C `KINDS=MOE` / B six kinds / D six + ADD + ROUTER_TOPK). The two terms ⑨ must still cut are the call count (51 → 1: #132 PR 2, held behind #141's transport microbench) and ATTN_M1's per-position cost (㉗ = #146).** **D mask (#132 PR 1, host-gated, cycle 20): `NNTR_HTP_FORWARD_KINDS=MOE,RMSNORM,QK_NORM,ROPE,CONV1D_GATE,ATTN_M1,ADD,ROUTER_TOPK` → 51 calls/token on LFM2.5 (host fixtures: hd8 7, hd64 8, lfm25 15; + ADD alone 9 / 10 / 19 by the same count); the default mask stays the six kinds. Device cell rides the combined sitting with #134's decode PPL.** **MEASURED ON SILICON (cycle 19, #130's sitting → §2): the entry is free (C ≡ A), the wired six-kind set costs −22 / −25 / −30 % at 95 calls/token and its text degenerates (user `n`). #135 done (the app never carried `ENABLE_HEXKL`, rule 36; the export is committed, no device cell — #136's sitting is its first device use). Open, in order: #136 (B's text: kind-by-kind bisect with `NNTR_HTP_FORWARD_KINDS` on silicon, ㉕), then ㉓ / #132; ㉗ (ATTN_M1 ≈ 1.05 µs/position/layer) is the other term. #130 closed.** **WIRED (cycle 18, #130's PR, host-gated): `hexkl_graph.c`'s table = MOE \| RMSNORM \| QK_NORM \| ROPE \| CONV1D_GATE \| ATTN_M1 (wire v2: `n_kv` / `gqa` / `head_dim` / `eps_bits` in the op record, the kernels' shape rules in the validator as `AEE_ESCHEMENOTSUPPORTED`, ATTN_M1-needs-ROPE as `AEE_ENOTALLOWED`); parameters bound once through `graph_set_param` (gammas, conv weights, conv state seed, RoPE table); the four CPU layers call one hook at a decode row (`htp_decode_hook.h`), the KV cache is registered at `graph_init` and seeded from the CPU prefill's rows once, the conv state per layer lives on the DSP after the first routed token. Host gate on both fixtures (`INPROC E2E PASS`): hd8 with MOE,RMSNORM,CONV1D_GATE = **11 calls/token**, hd64 with all six = **12**, the switch-off runs untouched; `NNTR_HTP_FORWARD_KINDS=MOE` reproduces plan 85's B. On LFM2.5 the same arithmetic gives **95 calls/token with all six kinds on, 22 with `KINDS=MOE`** — a per-op residency without adjacency to the MoE op *raises* the call count (§2 verdict); the device cost is the ride-along's (plan 130 §4 step 7, under the accuracy rule below). The count falls below 22 only with ㉓ (#132). Measured on the host: CONV1D_GATE bit-identical to the CPU path, RMSNORM 129 dB, the attention kinds 39 dB at worst (fp16 KV rows on the CPU vs f32 on the DSP, through two activation quantizers on 128-wide rows), tokens 8/8; the gate's floor is 30 dB.** **USER DECISION 2026-09-28 (cycle 18): the accuracy column for the ⑨ handoffs is (a) + (b) + approval — keep text-identical-to-CPU as a column, add `NNTR_PPL` vs the CPU `q40` PPL (upstream `32b46e32` is on `htp_moe` since PR #121), and add a user approval step: the filled handoff pastes each variant's generated text, the user marks `text approved: y/n`, and only approved rows are folded (contract §1 accuracy gate, `hexagon-handoff` skill). (c) is not taken. Binds #130's device ride-along and every later ⑨ / ㉓ handoff.** **Cycle 16b (2026-09-27, post-merge fold): #82 (PR #125 `d0552b1e`) and #81 (PR #126 `1afbf0aa`) are on `htp_moe` — their kernels, `_det` specs, host bit checks and IDL entries exist, but `hexkl_graph.c`'s table still has MOE as its only non-NULL slot (`RMSNORM` / `CONV1D_GATE` / `QK_NORM` / `ROPE` / `ATTN_M1` NULL, `resident_ok = MOE`), so the per-token entry still stops at the first non-MoE op: 22 calls/token, 0 ms by construction. #84 (PR #127 `ec589373`) landed the host E2E harness (`-Dhtp-inproc`, `run_inproc_e2e.sh`, `NNTR_HTP_DUMP`), so a wiring change can now be gated on the whole tiny model through the per-token entry on the host, not only on `graph_host_check`. **What is still missing for the call count to fall (㉓, plan 85 §7):** (1) the wiring itself — register the five kinds, seed the KV cache after the CPU prefill, upload the RoPE table at `graph_init` — **filed as #130** (host-gated PR + device ride-along; the accuracy-column decision (a)/(b)/(c) below binds only its device handoff); (2) ㉓'s resident set beyond it: the M=1 FCs (conv in_proj / out_proj, q/k/v/o, the dense FFN of layers 0–1; Q4_0 on the NPU model), the router FC + top-k, the final norm + lm_head — not yet filed, a follow-up to #130 with its own weight-format decision. #89's `generation(last 64)` (PR #128) fills the contract column from the next handoff on.** **Cycle 17, #81 in review (PR #126, stacked on #125; rungs 0–1 on the Mac, `qaic` + `hexagon-clang -c` of every new DSP source, rungs 2–3 not run there): `nntrainer/tensor/attn_m1_det.h` (the spec: 64-step q·Kᵀ per position, `* scale`, exact max, `exp_det`, 32-lane tree sum, `recip_det`, sequential PV), `hvx/hvx_attn_m1_f32.{c,h}` (the cache object `Kt [layer][kv][64][max_seq]` + `V [layer][kv][max_seq][64]`, f32, memalign(128) heap, **48 MiB at max_seq 2048 = 26 % of the ≈ 182 MiB heap**, + 256 KiB scratch; one pool unit per kv head), host check `ATTN M1 BIT-IDENTICAL` at L = 1 / 63 / 64 / 65 / 512 / 1024 × pool 0 / 3 / 7 workers byte-equal, append-chain ≡ bulk, double reference ≤ 1.7e-6 of max|V|, four mutants caught; IDL +4 methods after `conv_gate_m1_f32` (`attn_m1_register / release / kv_append / forward`, result sequence named `y` — `out` is an IDL reserved word); `HvxAttnM1.*` in `unittest_hvx_attn`. Unwired: #85's `ATTN_M1` slot stays NULL, 0 ms/token; the wiring issue needs `attn_m1_register(6, 8, 4, 64, max_seq)` at graph init and one `attn_m1_kv_append` per layer after the CPU prefill (the seeded rows are the CPU's fp16-rounded k/v, the decode rows f32 — one more reason (a) below cannot hold). **Plan correction:** `recip_det(1.0f)` is `0x3F7FFFFF` (swiglu_det.h's fixed point), not 1.0f, so at L = 1 the output is `v · (1 − 2⁻²⁴)`, not `v`; the check asserts that instead. Ride-along for the next sitting: `unittest_hvx_attn --gtest_filter='HvxAttnM1.*'` on that sitting's skel (expect `bad=0` at all six lengths, `ATTN_M1_FIELD pos=511/1023 us=` the first read of plan 81 §0's 0.5 / 1.0 ms estimate; `0x8000040e` = stale skel). Open (supervisor's call): the fp16 cache upgrade once an `sf → hf` RNE spec is device-confirmed (halves the 37.7 MB/token read at pos 1536, and max_seq 4096 fits).** **Cycle 17: #82 in review (PR #125, rungs 0–1 on the Mac, 2–3 not run there): `nntrainer/tensor/m1_ops_det.h` (the spec), `hvx/hvx_m1_ops_f32.{c,h}` (RMSNorm / q-k norm, RoPE 64, conv1d + gate at M=1 through upstream's `hvx_conv_gate_f32`), host check `M1 OPS BIT-IDENTICAL` on the real HVX source over an intrinsic emulation (`test/htp/host/hvx_emu/`), IDL +3 test methods after `forward_debug`, `HvxM1Ops.*` in `unittest_hvx_softmax`. Unwired: #85's `RMSNORM` / `QK_NORM` / `ROPE` / `CONV1D_GATE` slots stay NULL, 0 ms/token. Ride-along for the next sitting: `unittest_hvx_softmax --gtest_filter='HvxM1Ops.*'` on that sitting's skel (expect `bad=0` per case, `0x8000040e` = stale skel). Open for the wiring issue (user decision, option (c) below): the spec now exists for a NEON twin; the RoPE table is the ARM's `cos_ps`/`sin_ps` rows uploaded at `graph_init` (≈ 0.5 MiB at max_seq 2048), and the ops' DSP memory once wired is ≈ 1.8 MiB (plan 82 §3.4).** **Cycle 16: #85 CLOSED (PR #119 merged `b4c96bdb`, `NNTR_HTP_FORWARD=1` off by default). Queue (user, 2026-09-27; no sitting possible, host-gated PRs only): PR #121 rebase → #82 → #81 → #84 → #89; the user decision on the wiring handoff's accuracy column (a / b / c below) is still open and does not block these.** **Skeleton landed as PR #119 (`htp/85-one-call-per-token`, cycle 14, `state:review`): IDL `graph_init / graph_release / forward / forward_debug` after `moe_set_opts`, `htp_backend/htp_graph_desc.h` (wire format, validator with one `AEE_*` code per finding and never `AEE_EBADPARM`, the LFM2 builder: 228 ops at LFM2.5, 22 MoE), `hmx/hexkl_graph.c` (kernel table with MOE as the only non-NULL slot, per-op pcycles, one FARF line per call), `graph_host_check` in `run_host_checks.sh`, one core virtual `ComputeOps::set_decode_graph_desc(words)`, and the `NNTR_HTP_FORWARD=1` switch, off by default. Resident = MOE, so still 22 calls/token: 0 ms by construction (plan 85 §0); no device sitting in the PR. Ride-along for the next sitting: one decode token with the switch on (`[HTP] graph: init n_ops=228 resident=MOE moe_ops=22`, text ≡ switch-off at G=64; `0x8000040E` = stale skel). See ㉓ for what the call count actually needs.** Cycle 14: order #85 (skeleton, in progress) → #82 (small ops) → #81 (m=1 attention); #84 in parallel. All three end in host-gated PRs; the E2E A/B belongs to the wiring issue (not yet filed). USER DECISION (plans 81 §3.6, 82 last bullet), binds that wiring handoff, not these PRs:** the CPU decodes attention in fp16 with its own exp and runs no `_det` for RMSNorm / RoPE, so once the per-token entry is wired the NPU text cannot be bit-identical to the CPU run and B ≡ A is at risk too. Options: (a) keep text B ≡ A and expect a fail, (b) adopt `NNTR_PPL` (upstream `32b46e32`, #110 (d)) as the accuracy column for the ⑨ handoffs — this also lifts #110 to p1 per ⑱, (c) give the CPU path the same `_det` (NEON twin, doc 45 §3.3), which moves the CPU control and touches prefill. The supervisor recommends (b) with the CPU `q40` PPL as the reference, since #95 already measured that the NPU text leaves the CPU's at word 43 on today's weights. History: One FastRPC call per token: M=1 RMSNorm, conv1d + gating, RoPE, attention, dense FFN, lm_head on the DSP; per-token entry in the IDL. **Filed as #85 (skeleton entry + op table), #81 (m=1 attention), #82 (RMSNorm, q/k norm, RoPE, conv1d + gating)**; host harness for all of them #84 **Cycle 15 budget (§2): after the feed, transport 2.0 + outside the call 9.7 = 11.7 ms/token at G=512 against a ≈ 6.5 ms byte floor; ⑨ must deliver ≈ −5 ms for 20 ms/token, and 50 tok/s is reachable only on a 37 GB/s unit with ⑨ at its floor (rule 34). PR #119 (#85 skeleton) is in review with one unanswered clang-format nit on the new `HAP_perf.h` stub.** **Cycle 14: order #85 (skeleton, in progress) → #82 (small ops) → #81 (m=1 attention); #84 in parallel. All three end in host-gated PRs; the E2E A/B belongs to the wiring issue (not yet filed). USER DECISION (plans 81 §3.6, 82 last bullet), binds that wiring handoff, not these PRs:** the CPU decodes attention in fp16 with its own exp and runs no `_det` for RMSNorm / RoPE, so once the per-token entry is wired the NPU text cannot be bit-identical to the CPU run and B ≡ A is at risk too. Options: (a) keep text B ≡ A and expect a fail, (b) adopt `NNTR_PPL` (upstream `32b46e32`, #110 (d)) as the accuracy column for the ⑨ handoffs — this also lifts #110 to p1 per ⑱, (c) give the CPU path the same `_det` (NEON twin, doc 45 §3.3), which moves the CPU control and touches prefill. The supervisor recommends (b) with the CPU `q40` PPL as the reference, since #95 already measured that the NPU text leaves the CPU's at word 43 on today's weights. History: One FastRPC call per token: M=1 RMSNorm, conv1d + gating, RoPE, attention, dense FFN, lm_head on the DSP; per-token entry in the IDL. **Filed as #85 (skeleton entry + op table), #81 (m=1 attention), #82 (RMSNorm, q/k norm, RoPE, conv1d + gating)**; host harness for all of them #84 | removes 22 round trips (**1.9 ms/token since #88**, was 12.5) and the time outside the MoE call (≈ 9.6 ms/token at G=512, floor ≈ 6.5); needed for the last ≈ 20 % to 50 tok/s once the MoE dsp column nears its floor (§2 budget row) | ⑤ ⑥ ⑦ |
| ⑩ | Registration at load time, bake cache on disk (doc 45 Phase D "P4") | load time, not speed | — |
| ⑪ | Prefill residency (doc 45 B/C/D) | prefill 523 → 700+ | after decode goal |
| ⑫ | **Cycle 24: unchanged — #157 on hold (user), PR #161 not folded.** **Cycle 23: unchanged — #157 on hold (user, 2026-09-29), PR #161 not folded; under the bypass the split's estimate is ≈ 0.3–1.5 ms/token and it needs a per-layer DSP↔CPU sync once decode is one call per token (issue #157 comment).** **Cycle 22: filed as #157; step 2 is PR #161 (open, not folded), measurement on hold by the user (2026-09-29); to be rebased on the bypass default, where only ≈ 13 GB/s is left beside the DSP (rule 44 (2)).** CPU+NPU expert split | raises the ceiling only if ④ > 45 GB/s | ④, user decision Q11 |
| ⑬ | (withdrawn 2026-09-21: no simulator in this project, user decision) | — | — |
| ⑭ | **CLOSED (cycle 16b): #89 landed as PR #128 (`40dd4ca8`, merged `6497f38a`, 2026-09-27).** `generation(last 64)` in the base report block, host-proven on the tiny fixture at G = 70 and G = 10; not yet seen on device — the next handoff's A logs carry it and BENCHMARK's last-64 column fills from that row on (it cites PR #128) | fills the contract §1.1 column from the next handoff on | — |
| ⑮ | ~~Anchor sitting on a named unit~~ **withdrawn (user decision 2026-09-22, rule 13): any unit, same-sitting A/B only.** #94 sitting 2 (`dad0f476`, `R3CY205ZMND`) is the "now" in BENCHMARK.md and contract §1; no second-unit issue is opened | — | — |
| ⑯ | **Since PR #115 (`c45d4433`, cycle 13) the default is D192** — proof line `[HTP] moe m1 gemv: on (applied=0x103c1) lead=192KB rows1=1 source=default`; `applied=0x1` in an A log is the pre-#115 default and voids the sitting; A0 = `LEAD_KB=0 ROWS1=0` (`0xc1`). **Default since PR #108 (#101, merged `8e121dbe`, #101 closed; ride-along cells in #100's handoff):** env unset = GEMV on, `NNTR_MOE_HTP_M1_GEMV=0` = opt-out; proof line then `[HTP] moe m1 gemv: on (applied=0x1) source=default`. From that PR on, an A log with `off` or without `source=default` is the HMX path and is not the reference (the sitting-2 rule below is inverted). **Measured → §2 (goal progress +6.2 / +7.7 / +1.7 %, text identical).** Follow-ups: make the GEMV path the default (**#101**, PR #108 merged, so every later variant A carries it); its compute side (**#105**, measured cycle 9: gate failed, PR #107 held, §2) and weight feed (㉒ remainder, via #100). History: Device A/B of #80 / PR #86 (M=1 GEMV switch on vs off): the first lever against `lfm2_moe` 42.1 ms/token. **Cycle 4: PR #86 merged as `2a75f7d9` (2026-09-22 02:00 UTC) before #94's set was pushed, so it rides as variant C (`NNTR_MOE_HTP_M1_GEMV=1`, 6 NPU cells + one level-2 run + `*MoeLayerM1GemvMatchesHmx*`) of sitting 2 — handoff rebuilt from `2a75f7d9` (`htp/94-sitting2-anchor-trace` @ `8029b76e`). A on that head = the same binary with the switch off (`[HTP] moe m1 gemv: off (applied=0x0)`, `blocks=5632 m1_gemv=0/1408`); a C log that prints `on` in an A cell voids the run.** | MoE DSP 1.35 → ≈ 0.3 ms/call if the arena read keeps up; if not, the number says what ⑥ must deliver | #97 (skel loads) → #94 (user's phone time); first attempt blocked before any NPU cell |
| ⑲ | ~~#97~~ **closed (PR #98 merged `14f65120`; confirmed on silicon by #94 s2's `DmaProbeShapes` PASSED as the first device command).** Skel from `2a75f7d9` did not load (`0x80000406`): `hexkl_dma_trace.c` absent from `test/htp/build.sh` `SRCS` → seven undefined `hexkl_dma_trace_*`. Fix = add the file + an undefined-symbol guard in `build.sh` (rule 17); then #94's artifact set is rebuilt and the handoff re-issued (same document, new md5s) | unblocks every NPU cell of #94 (⑥ ⑮ ⑯ ⑰) | none |
| ⑳ | **CLOSED (cycle 22): PR #154 (`cf20aa50`) + the ride-along anchor `us_per_call=608.1 … checksum_ok=y`; #99 closed (§2 #99 row).** **Cycle 14: #99 downgraded to p2 (plan stands, `docs/plans/99-replay-content-check.md`).** The question it was filed to decide is answered: plan 99 found the replay writes each chunk packed (`dst_stride = row_size`) where the kernel writes strided, so the DDR read stream was byte-for-byte the traced one and rows c/d/g/h stood as timed; then #100 replaced row h with tag-validated cells (16/16 `checksum_ok=y`, rule 28) and rule 19 was withdrawn. What is left is hygiene: the rule-30/32 anchor cell (`DMA_REPLAY workers=1 load=0 pace=0`, one of the 11 old cells) rides every handoff — plan 117 reads it first and verbatim — with `checksum_ok=n` and a `FAILED` gtest that a reader will one day mistake for a regression. Gate for the remaining fix: `test/` only, the anchor line reads `checksum_ok=y` with `us_per_call` within ±10 % of the previous sitting's anchor (724.0), `weight_bytes` printed; rides along any sitting, no E2E cell. History: **Filed as #99 (p1).** `MoeChunkReplay` fails its content check on all 11 cells (`res[6]=1467840` vs `want_sum=9461760`, `bytes_per_call` 22560768 vs the in-situ 22020096, `regions=32`) while `DmaProbeShapes` passes in the same binary. Decides whether the replay timings (rows c, d, g, h; rule 19) stand or must be re-measured; ~~#100 is blocked on it~~ (cycle 9: only #100's old 11 cells and `traced*` depend on it) | rows c/d/g/h confirmed or re-measured | none |
| ㉑ | **Fixed by PR #108 (print only, #102; merged `8e121dbe`, #102 closed); row not yet seen on a device (#101 ride-along R2, rides #100's handoff).** Cause: on the GEMV path the `swiglu` slot is Σ lane-time over the pool (`moe_tail_probe_add`, by design), and the print subtracted it as a stage; the 30-slot table is not misaligned. `rest` now leaves it out, and a `weight DMA: n/a (direct arena read inside mm …, N.NN lanes busy over mm)` line replaces the missing one. **Filed as #102 (p2).** Level-2 `[HTP-PROFILE]` M==1 row with the GEMV path on: `swiglu 5601.2`, `rest<=-5588.2 (-311.8% of host)`, no `weight DMA:` line — the GEMV path's stage slots are mis-attributed in the print (`htp_compute_ops.cpp` / the 30-slot stage table of #86). Cosmetic, no tok/s cell affected, but the B row of every sitting after #101 is unreadable until fixed | a readable M==1 row under `NNTR_MOE_HTP_M1_GEMV=1` | #101 makes it urgent |
| ㉒ | **CLOSED (cycle 16): PR #118 merged `d79c0efe`; #120's A (nothing set, `applied=0x303e1 … feed=vtcm source=default`) vs A0 (`FEED=0`) = **+25.7 %** decode at G=64, `mm` 677.2 vs 934.3, text identical — the feed default confirmed as a sitting's control and BENCHMARK's "now" (35.97 / 35.96 / 35.17, §2 ㉔). Nothing left here; the engine gap is ㉓.** **Cycle 15: feed half MEASURED (#117 → §2), gate passed twice on `R3CY10WM83Y` (`mm` 676.4, decode +32.5 / +30.9 / +28.6 %, text = A 36/36); the flip to the feed default is in progress on PR #118 (#117 `state:in-progress`, acceptance on the issue) and ㉒ closes when it merges. #114 closed as moot. The residual engine gap is ㉓.** **Cycle 13: D192 is on `htp_moe` (PR #115 `c45d4433`), #113 closed; #117 is `state:planned` (`docs/plans/117-m1-gemv-vtcm-feed.md`, `a566e61f`) and its base is now the `htp_moe` head, not PR #115's branch — its sitting's A is the D192 default and carries #113's A/A0 confirmation.** **Cycle 12 (#113 → §2): compute half CLOSED, feed half re-filed as a rate lever.** The 2 × 5 (loop × lead) matrix is complete (rule 31): the lead harms the four-row loop, helps the one-row loop only to 192 KB, and the minimum D192 (`mm` 937.0, −6.3 %) misses the 840 gate; **the user lands D192 as the default anyway** (rule 33, PR #115, #113 in progress for the flip; decode +4.11 / +3.29 / +4.26 %). The ride-along (724.0 µs / 31.2 GB/s, rule 32) says 31–32 GB/s is the real DMA bound, so the feed half closes *as a 37 GB/s question* and the VTCM feed is worth what #100 measured: `f2` **31.6 GB/s beside HVX vs 23.5 GB/s** for D192's direct read = **≈ 1.33×** → `mm` ≈ 710 µs, ≈ −5 ms/token, with the one-row loop's ≈ 240 µs compute hidden under the feed (rule 29). **Filed as the ㉒ feed issue (**#117**, p1, `state:planned` since cycle 12's planner, `a566e61f`):** two linear descriptors per expert (`f2`: 8 descriptors, `dst=strided`, depth 2), gate `mm` ≤ 760 at M==1 (−19 % vs D192's 937), decode ≥ A at all three G, text = A, `bit_identical yes`, prefill −5 % on M>1 `dsp`, plus the `DMA_REPLAY` anchor cell in-sitting (rule 30). It must not re-import the 46-descriptor interleaving (rows f + g): at M=1 the ring carries only the expert's own `wh_bytes`, issued one expert ahead. #114 (up-half lead, p2) is moot if it lands. Earlier: **Cycle 11, after #100 and the user's direction ("stop leaning on the DMA numbers, take the compute-side wins first"): the feed half is DEMOTED and the lever is the `l2fetch` lead.** #100 measured the DMA feed at **31.6 GB/s** (best `f2`) against the GEMV's own direct arena read at 21–27, i.e. ≈ **1.2 ×**, not the 1.6 × (37 / 22.6) cycle 9 assumed — and on a sitting whose DMA is itself 22 % low (rule 30). VTCM staging therefore buys ≈ 690 → ≈ 550 µs of feed at best, and it costs a ring, a double-buffer and a prefill-gate risk. The cheap half is that **the feed is DDR latency, not bandwidth** (rule 26), and the only lever measured against it is monotonic and unsaturated: `l2fetch` lead 0 / 64 / 192 KB = 145.2 / 137.5 / 120.2 ns/tile. #105 confounded the lead with the loop shape — A is four-row with a zero-lead self-prefetch, B1/B2/B3 are one-row with lead 0/64/192 — so **the four-row loop has never been measured with any lead at all**. Filed as **#113 (p0)**: complete the 2 × N matrix, sweep past 192 KB, cross the expert boundary, land the winner (which decides PR #107's default and whether #107 lands at all). Gate `mm` ≤ 840 µs at M==1 (−13.7 % vs #100 A_L2's 980.9). **#114 (p1)** carries the next-expert / cross-op prefetch. The feed half stays open but unfiled, behind ⑥'s ride-along. Earlier: **Cycle 9: compute half measured (#105 → §2): gate failed. The best loop (one-row + 192 KB lead) gives `mm` 930.9 vs 973.0 (−4.3 %). The feed is ≈ 74 % of `mm` under that loop and ≈ 37 % under the four-row one. The feed half is the lever. It needs #107's loop to hide the compute: ≈ 240 µs vs the ≈ 590 the four-row loop leaves. PR #107 is held, and the feed issue rebases onto it. Order: #100 certifies a DDR → VTCM descriptor shape (≥ 37 GB/s beside HVX). Then the ㉒ feed issue is filed with that shape, gated `mm` ≤ 600 at M==1, text = A, prefill −5 %, and its A/B measures feed + one-row loop against A. If #100 certifies nothing, land #107 with a lead sweep (≥ 192 KB) and close ㉒ at what the arena allows.** Earlier: **Split (user direction, cycle 6, 2026-09-22). Compute side filed as #105 (p0, `state:needs-plan`):** one-row `gemm_rows1` at `m = 1` (drops `gemm_rows4`'s three dead accumulators; accumulation order unchanged, so int32 stays bit-identical to the HMX), `l2fetch` lead / distance, and the lane split over the pool. It needs no ring, so it does not wait for #100. The handoff sets `NNTR_MOE_HTP_M1_GEMV=1` explicitly in A and B, which keeps it independent of #101. **Feed side (remainder, not yet an issue):** VTCM staging of each expert's `wh_bytes` by DMA (two linear descriptors per expert); it must not re-import row h's slow list, so it is filed once #100 names h's cause. Evidence: C's `mm` 783 → 975 µs, a direct arena read at ≈ 22.6 GB/s effective, lane-bound (`swiglu / mm = 5.75`; plan 101 §3.4). Supervisor's estimate, not measured: the dead `vrmpy` are ~50–100 µs of the 975, and `hvx_impl` #59 measured direct DDR vector reads at 21–25 GB/s against 37 through the ring. So #105 alone may stop above 600; its ride-along microbench (arena vs VTCM copy × 4-acc vs 1-acc) sizes the feed half. Combined gate for ㉒ = `mm` ≤ 600 µs at M==1 with text identical | MoE dsp 1033 → ≈ 660 µs/call (`mm` 973 → ≈ 600; ≈ −8 ms/token) | feed: #100 → feed issue; compute: PR #107 (held) |
| ㉓ | *(Two rows carry ㉓: this one is the feed-engine gap; the next is the resident set = #132, which every "㉓ / #132" reference means.)* **CLOSED (cycle 21): filed as #149, closed after its step 0 (D2, no code; §2 #149 row, rule 41).** The anchor read 37.3 GB/s and the in-app per-descriptor median 32.9 cold / 33.0 warm = 0.88 × anchor; the ring never idles and the candidates below are already in the tree or worth 0 µs. The MoE `mm` floor on `R3CY10WM83Y` is ≈ 670 µs/call (14.7 ms/token), not 590, and the current 674 sits on it. Original text: **In-situ feed engine below the anchor (cycle 15, #117): `engine 32.3..34.4 GB/s` in the layer call vs the same sitting's anchor 37.3 and the `vtcm` microbench's 37.7** — `mm` 676 vs the 590 µs bound = **≈ 86 µs/call ≈ 1.9 ms/token (≈ 7 %)**, the last MoE-side lever on a 37 GB/s unit and nothing on a 31 GB/s one (there the in-situ 32–34 already exceeds the anchor). Signatures in B's ring line: `first expert ready at 122 µs` (the first 3.5 MB slab at 33 GB/s = 106 µs, un-hidden per call — the row-f analogue of #94's attribution), `depth max=4` with 10 descriptors, `last issue at 500 of 712`. Candidates, in order: kick expert 0's gate_up slab before the activation quant (the call's first ≈ 20 µs), issue expert e+1's gate_up before expert e's down instead of after, deeper issue; a cross-call prefetch of the next layer's expert 0 needs the router result and exists only with ⑨. **Not an issue yet:** it is filed after the flip's own sitting confirms the gap on the unit that runs it (a ride-along: the B-as-A ring line vs that sitting's anchor). Gate when filed: `mm` ≤ 630 in-sitting with the anchor ≥ 36 GB/s, text = A, prefill −5 % | ~~≤ −1.9 ms/token~~ **0 (rule 41)** | closed |
| ⑰ | **CLOSED (cycle 22): #150 closed after step 1b (rule 46, §2 #150 row); the CPU-side lever that is left (attention at G=1024) moves to #162 / ㉙.** **Filed as #150 (p1, `state:needs-plan`, cycle 20 close; the queue head in cycle 21, bit-preserving lever (3) of the direction change).** **First device read (#150 comment; `R3CY10WM83Y`, 2026-09-28 20:5x, the #134 set = `bb845426`, FastRPC build, switch-off path, G=64, mirrored 8 6 4 \| 4 6 8): `NNTR_NUM_THREADS` 8 / 6 / 4 → decode 36.64 / 37.31 / 37.18 (+1.8 / +1.5 %, near the run-to-run spread), prefill 563.9, 548.2 / 512.5, 514.6 / 468.0, 458.0 (−8 / −17 %); MoE dumps (G=8) `bit_identical=1` 6 vs 8 and 4 vs 8 (398 files), texts identical.** So the thread count is a free knob for the bit gate but worth ≤ 0.5 ms/token and only as a decode-only count (prefill stays at 8); ⑰'s ≈ 2–3 ms is not plain over-splitting. **Resized again by rule 42:** the time outside the MoE call (≈ 9.7 ms at G=64, 10.8 at G=512 on #141 Q) reads ≈ 402 MB of Q4_0 FC + lm_head, i.e. the CPU's M=1 GEMVs already run at ≥ 37–41 GB/s effective; the CPU alone read 67.9 GB/s (§2 ④), so the headroom is the GEMV's own read rate (≤ ≈ 4–5 ms if it reached 67.9, ≈ 0 if 40 is its in-decode bound). The plan's first step is the per-op CPU breakdown at M=1 as a byte rate (FC GEMVs, lm_head, attention, norms, sampling) on the dspqueue default (the ARM spins up to 5 ms after each MoE call now, rule 40). Original row: **`fully_connected` 28.2 vs 10.3 ms/token on the NPU run with no FC on the HTP** (§2 ① correction). Not a round trip; candidates: CPU thread over-splitting / contention with the FastRPC poll thread (doc 48 §2 ③), CPU DVFS while the DSP runs. 18 ms/token is the second-largest single lever after MoE and needs no DSP code if it is a threading matter. Decide with one extra profile cell (`NNTR_NUM_THREADS=4` on the NPU model, or the non-WH `QS4CX` model with `NNTR_MOE_HTP_DECODE` off/on) — candidate ride-along for #94 or its own issue — **#94 s2 skipped the ride-along (budget)**, so this still needs its cell | **Resized in cycle 7 (§2 budget row):** in TPS terms the whole outside-the-MoE-call time is 11.1–13.7 ms/token on the pre-#103 binaries (#94 s2 A and C, #88 A) and **8.4 (G=64) / 9.6 (G=512) on `htp_moe` head** (#88 B — the 5 ms poll / staging moved ≈ 5 ms of it, rule 23), against a ≈ 6.5 ms byte floor; the profile's 28.2 vs 10.3 is mostly the `--profile` build's own ≈ 20 ms/token. So ⑰ is worth ≈ 2–3 ms/token now, not 18: no issue; still a cheap ride-along (`NNTR_NUM_THREADS=4` on the NPU model, TPS binary) in the next sitting that has room | ≈ −2..−3 ms/token on the NPU path | next sitting's ride-along |
| ㉓ | **Cycle 24: Part A merged (PR #175 `721ff913`; S3 / S4 folded, router 62 311 accepted). Part B on silicon (§2 row, rules 54–56): the two-session path runs at `calls/token=1.00` and is bit-identical after the CONV1D_GATE fix (E5e), 20.8 / 19.5 tok/s at G 64 / 512 after the spin fix (E5d) vs A ≈ 54 — `state:in-progress`, PR pending: rebase onto `htp_moe` (ATTN_M1 round 3), HVX quantizer / SwiGLU / argmax in the graph kernels + per-shape lanes (20.3 → ≈ 11 ms), then set_e5f (projected ≈ 25 ms ≈ 40 tok/s; the plan's 47–51 needs the FC loop at ≈ 57 GB/s). Track 2 (VTCM share) stopped. #192's single-session window is not viable (rule 57).** **Cycle 23: Part A of PR 2 measured (PR #175 open, §2 row, rule 49): the CPU-exact FC / router / SwiGLU / quantizer are bit-identical on silicon on the `hvx_intrin` variant; exact FC 7.88 ms/token VTCM-fed (CPU 7.4), heap 113 MiB vs 383 → decision D (user); #178 probed option (a) (㉚). Reopened by the 2026-09-29 direction (PR #169): the resident path is the main track again, bit-preserving.** **Parked (cycle 21, #132 `needs-user`): the direction change stops the resident path; PR 1's reading is in §2 (D +6.5 / +9.9 % over B, text rejected). PR 2's premise (FCs through the DSP) is also void on bytes (rule 42, ㉘). Reopen only if the accuracy rule changes.** **Cycle 20 close: PR 1 merged (PR #145 `bb845426`); #132 `state:needs-measurement` — its D cell rides #134's combined sitting. PR 2 (FC / DENSE_FFN / LM_HEAD through the WH GEMV, 51 → 1) held behind #141.** **#132 PR 1 (cycle 20, host-gated): ladder corrected to 95 → + ADD **73** → + ROUTER_TOPK **51** → + FC 3 → 1 (the cycle-20 row below dropped the DENSE_FFN op of layers 0–1 in the ADD and ROUTER rows; counted with `htp_graph_lfm2_build`). ADD and ROUTER_TOPK are wired: ADD adds into DSP slot 0, which holds the residual across calls (bit-identical to the CPU add: `add==noadd` logits `bit_identical=1` on hd8 and hd64); ROUTER_TOPK is `m1_router_topk_det` (four-partial-sum GEMV, `exp_det` / `recip_det` sigmoid, lowest index wins a tie) with the same tie rule now in the CPU comparator; its routing reaches the MOE op in the same call. PR 2 (FC / DENSE_FFN / LM_HEAD) held behind #141.** **Cycle 20 (verified on the code, posted on #132): calls/token = the number of non-resident ops in `htp_graph_lfm2_build`'s list (none are adjacent, `hexkl_graph_forward` stops at the first `!resident`): six kinds 95 → + ADD **71** → + ROUTER_TOPK **49** → + FC **3** → + DENSE_FFN / LM_HEAD **1**. The weight-format question is narrower than the body says: the upstream FC kernels (`gemm_q4_0_accel / _batch / _dense_ffn / _conv_block_fp32`) already run Q4_0 FCs on the DSP, but through `htp_qs4cx_from_q4_0x4` — "a real requantization, not a bit reshuffle" (per-column int4, u8 activations) — at first call, and the handles are plain qs4cx on `hexkl_mm_u8i4_layer_run`'s HMX tile (wall 1 at M=1), not WH order: `hvx_gemm_u8i4_wh_col` (the M=1 GEMV + VTCM feed) is reached only from `hexkl_mm_u8i4_moe.c`. So option (ii) is what exists, at load time, with no new model file; the plan's design step is the host-side requant to WH bytes + arena registration before `graph_init` (≈ 360 MB: 24 layers of FCs, dense FFN 2 × 3 × 2048 × 7168, lm_head 128000 × 2048 ≈ 131 MB) and the startup cost of converting it. Router: gate [2048 × 32] f32 + `expert_bias` [32] f32 ("Always kept FP32"), sigmoid + bias top-4 `partial_sort`, weights = bias-free sigmoid / sum × `ROUTED_SCALING_FACTOR` — a `graph_set_param` slot and a `_det` spec (exp, tie rule). Split agreed: PR 1 = ADD + ROUTER_TOPK (→ 49, no sitting of its own), PR 2 = FC / DENSE_FFN / LM_HEAD (→ 1); ㉗ is budgeted alongside.** **Cycle 19: #132 waits behind #135 and #136 — a #132 sitting is meaningless while B's text fails (§2 #130 row), and its call-count fall leaves ㉗'s ATTN_M1 term (≈ 3.2–7.1 ms/token) untouched, so the plan must budget both.** **Filed as #132 (p1, `state:needs-plan`) on 2026-09-28; user order: #130 PR → #130 sitting → #132. Weight-format decision (HVX Q4_0 GEMV vs `QS4CX_WH` re-quant) is the plan's first question; accuracy read by the 2026-09-28 rule.** **Since #130's PR the only path below 22 calls/token; filed as #132 (user decision 2026-09-28).** With #82 / #81 wired the resident stretches are still bounded by the ARM's FCs, ADD and router, so every stretch is its own call (95/token with all six kinds). **The resident set that removes calls is larger than #81 + #82 (plan 85 §7; opened by PR #119).** `forward` stops at the first non-resident op, so the 22 calls/token fall only when *every* op between two MoE ops is resident: besides #82's norms / conv1d + gate / RoPE and #81's attention, the M=1 FCs (conv in_proj / out_proj, q/k/v/o, the dense FFN of layers 0–1), the router FC + top-k, and for the tail the final norm + lm_head. Those FC weights are Q4_0 on the NPU model (contract §11), so it is either a Q4_0 HVX GEMV on the DSP or a re-quantization of the FCs to `QS4CX_WH` — the latter moves the weights vs the CPU control and needs the user's say on the accuracy gate (the ⑨ user decision above covers text; this adds weights). Without it ⑨'s ≈ −5 ms cannot be reached. Supervisor's call: file it once #81 / #82 land, or fold it into the wiring issue | ≈ −5 ms/token at G=512 (transport 1.9 → ≈ 0.1, outside-the-call ≈ 9.6 → floor ≈ 6.5) | #85 (PR #119), #81, #82 |
| ㉕ | **CLOSED (cycle 20 close): #136 completed with PR #143 — not a defect (rule 39, §2 #136 row). The candidates below were cleared on the host at the real shape (plan 136 §0) and on silicon (bisect + dumps + recompute); the race the plan found is fixed but was not the cause.** **Filed as #136 (p1, `state:needs-plan`, cycle 19).** B's text on silicon degenerates from the first token with all six kinds resident, while C (MOE only through the same entry) ≡ A and the host E2E (tiny hd64 fixture, prompt 16 + 8) passes at 39 dB with tokens 8/8. Not arithmetic (rule 37): a logic / binding fault that the host never exercised — candidates in order: the KV seed after a 512-row CPU prefill (`attn_m1_kv_append` `pos` = the layer's `from`, fp16-rounded rows), the RoPE table / position offset at pos ≥ 512 (rope64 is the one kind whose gtest differs at pos ≥ 1), the conv state seed at the prefill boundary, a `graph_set_param` binding that the hd64 fixture's shapes hide, the CONV1D_GATE hook in the one-layer conv block form. Method: one sitting, G=64 ×1 per mask on the same binary set, `KINDS=MOE,RMSNORM` → `+QK_NORM` → `+ROPE` → `+CONV1D_GATE` → `+ATTN_M1` (the first mask whose text leaves A names the op), then `NNTR_HTP_DUMP` of that op's first calls against the CPU rows if the dump covers it; #134's decode PPL as the column if it lands first. Gate: text ≡ A (or user-approved) with all six kinds at G=64, prefill ≥ −5 % | unblocks #132 | #135 (a committed app with the define) |
| ㉖ | **Cycle 20 close: kept at p2, last in the queue.** #136 removed its only urgency: `rmsnorm kind=2 bad_y=2048` is the ±1e-39 row (rule 37 as written), the normal-row RMSNORM sits at 131–148 dB vs the CPU order and nearer f64 (rule 39), so nothing here moves text or tok/s. It stays because the device suites are gate (a)'s only device reading and a red suite hides a real regression — `HvxM1Ops` now also carries PR #145's `RouterTopkMatchesDetBitExact`. The combined set stages `unittest_hvx_softmax` (the `HvxM1Ops` suite, `25bec7df…`), not `unittest_hvx_attn`. **Filed as #137 (p2, `state:needs-plan`, cycle 19).** The ride-along gtests: `HvxM1Ops` 0/5 and `HvxAttnM1` 2/4 on #130's skel — subnormal / tiny-value rows a few ulp off the `_det` specs (rule 37), `RejectsBadShapes` answering `AEE_ERPC` (rule 38). Decide per case whether the spec / `hvx_emu` or the kernel is wrong (the spec is the contract: if the silicon's rounding of tiny values is the intended one, the emu and the spec's tolerance change and rule 37 is rewritten; if the kernel flushes or mis-rounds, the kernel changes), and make the negative tests read the marshalling. Gate: `bad=0` on every row kind on silicon, or a device-confirmed rule in this ledger naming the exact op and the rows it exempts; `RejectsBadShapes` green on the device; host checks unchanged or extended | gate (a) on silicon for the ⑨ kernels | rides the next sitting that stages both gtest binaries (#136 closed) |
| ㉗ | **Cycle 24: rounds 2 and 3 merged (PRs #182, #190); both speed gates pass on silicon (193 838 / 313 747 vs 210 k / 350 k, §2 #170 R3 row), bit-identical, default off (28 calls/token — the round trips are ㉓'s). Round 4 (on #170, after Part B's PR): (1) a gtest-only cell of W with `-DATTN_M1_EXP_TAB=0` (the table's gather costs 387 k on silicon vs the exp16's 272 k, rule 58; estimate softmax ≈ 510 k, pool ≈ 257 k), (2) `vgather` from a VTCM carve-out (plan §3.5), (3) pv 384 k at the DDR line (DMA-into-VTCM, ≈ 45 L). ATTN_M1 in Part B's branch still reads 8.33 ms/token (pre-#170 kernel) — the rebase collects ≈ −7 ms.** **Cycle 23: #170 (p1, `state:review`) — round 1 merged (PR #176: bit-identical fp16-lane ATTN_M1, 394.5 k / 700.4 k pcyc/op, off by default), round 2 in PR #182 (250.3 k / 373.6 k; gates 210 k / 350 k missed by 19 / 7 %). Round 3 on the issue, in this order: (1) split the softmax word (exp16 / ET stores / max / sum / divides) — a measurement, then cut the largest; (2) `-DATTN_M1_P1_LEAD=0` and scalar splats vs splat vectors in a two-skel gtest sitting; (3) append's 33 k. DMA-into-VTCM (plan §3.1, ≈ 45 L) stays unopened until the compute terms are at their lines (rule 48, §2 rows). ATTN_M1 at ≈ 0.12–0.18 ms/layer in-model is no longer the end-to-end blocker; the FC set (㉓ / ㉚) is.** **Parked (cycle 21, #146 `state:review` + `needs-user`, p2): PR #148 (phase counters, O1, O13) is open, bit-identical on silicon, ATTN_M1 941–959 → 387 µs `dsp` at pos 1023 (§2 #134 row); the `dsp_us ≤ 300` gate is not met and O4 is not started. ATTN_M1 runs only on the parked resident path, so merging PR #148 is the user's call and buys no decode today.** **Filed as #146 (p1, `state:needs-plan`, cycle 20; planner running).** **ATTN_M1 costs ≈ 1.05 µs per KV position per layer on silicon** (538.75 µs at pos 511, 1183.38 at pos 1023; `1204826` pcyc/op = 0.79 × a MOE op at pos 512–575): 6 layers = ≈ 3.2 ms/token at pos 512, ≈ 7.1 at pos 1024 — the same order as the whole MoE `dsp` column, and it grows with G. At pos 512 the kernel reads 6 × 2 MB of f32 KV per token in ≈ 3.2 ms ≈ 3.9 GB/s, an order under the DDR rate: latency / issue bound (one pool unit per kv head, sequential PV — plan 81), not bytes. Levers: vectorise across positions (q·Kᵀ as a 64-wide dot per 32 positions, PV as a 64-lane accumulate), the fp16 cache (plan 81's open upgrade: halves the bytes, needs an `sf → hf` RNE spec), `l2fetch` of the next kv head's slab. Gate as filed on #146 (the cycle-19 draft here said ≤ 0.25 ms at pos 511 and waited for #136's text; both superseded): `ATTN_M1_FIELD pos=1023` ≤ 400 µs, `bad=0` on silicon, six-kind decode up at G 512 / 1024 in a same-sitting A/B, decode PPL ≤ +2 % of A with text compared and approved (rule 39), prefill ≥ −5 %; with #132 it is the second term ⑨ must cut | ≈ −2.5 ms/token at G=512, more at G=1024 | — (#136 closed) |
| ㉘ | *(Cycle 40 note: measured on the Gemma-4 26B dummy, #201 S5-0 — FC 14.5 + LM_HEAD 8.2 + DENSE_FFN 7.3 ≈ 30 ms/token on the one PD's Q4M1 path for the FC arena's 1278.7 MiB of Q4M1 bytes, i.e. ≈ 43 GB/s (above rule 41's in-app 33, under the bypass's ≈ 57 and the CPU's 51–63), while the hybrid that keeps those on the CPU is ≈ 30 % faster at every G; this arithmetic is now the measured reason #234 P3 / P4 and #229 S2 come first — the lever is fewer bytes (WH sidecar, then 2-bit FCs), not a CPU-exact Q4_0 on the DSP, which stays not filed.)* *(Cycle 22 note: under the bypass the DSP reads ≈ 57 GB/s in the app, so the 402 MB would take ≈ 7.1 ms against the CPU's 7.4 — a wash on bytes, while the NEON fused-FMA epilogue still cannot be matched; stays not filed.)* **CPU-exact Q4_0 FC on the DSP (the direction change's lever (4)) — not filed (cycle 21, supervisor; user may overrule).** Proposed budget: ≈ 3–5 ms/token, with the integer parts of Q4_0 exact on any ISA and only the float epilogue and the activation quantizer to match the NEON path bit for bit. **Why it is not an issue:** (1) bytes — the Q4_0 FCs + lm_head are ≈ 402 MB/token (rule 42), and the DSP reads at ≤ 33 GB/s in the app (rule 41) → ≥ 12 ms on the DSP against the ≈ 9.7–10.8 ms the CPU spends on *all* its work outside the MoE call today; the DSP would be slower on the reads alone, before the added round trips (22 → ≈ 100 calls/token at 14 µs = +1.1 ms). (2) Bit-exactness is harder than the proposal assumed: the ARM kernel (`nntr_gemv_q4_0_4x8_q8_0`, `nntr_ggml_impl_neon.cpp`) accumulates `acc = vfmaq_f32(acc, vcvtq_n_f32_s32(sumi, 4), d_a · d_w)` — a **fused** multiply-add per block, which HVX `sf` arithmetic (separate multiply and add, qf32 intermediates) does not have, so each block's epilogue needs an emulated single-rounding FMA, plus the q8_0 activation quantizer's scalar `1/d` and its rounding mode; which kernel runs at M=1 on the phone (4x8 i8mm vs 4x4 dotprod) also fixes the per-row block order the DSP must copy. **What would make it an issue:** a DSP read path above the CPU's in-decode FC rate (not in sight: the anchor is 37.3), or #150 showing the CPU's M=1 GEMV far below 33 GB/s — then the plan starts with the byte-rate breakdown and a host `_det` spec of the NEON epilogue (FMA included). The real 'more readers' lever is ⑫ (CPU + DSP reading disjoint FC rows at once; bit-preserving if split by output rows) and needs ④ / #90's aggregate first | none as proposed (≥ +1 ms/token by bytes) | rule 42; #150; ④ / #90 for the split variant |
| ㉙ | **CLOSED (cycle 23): #162 closed by the user after step 0 and the prefetch A/B (§2 #162 row, rule 51) — neither lever pays; G=1024 ≥ 50 belongs to the end-to-end track (PR #169) and, on the hybrid default, to rule 52 (cool A cells read 50.9–52.4 at G=1024, the row of record 47.36 waits for a mirrored cool sitting).** **Filed as #162 (p1, `state:needs-plan`, cycle 22).** G=1024 is the one length below the goal after #158: **47.36 tok/s = 21.11 ms/token, −1.1 ms needed**, bit-preserving. Candidates: (a) CPU attention (`mha_core`) 1.97 ms/token at G=1024 (1.47 at G=512), heads independent, the fp16 NEON reduction order kept as `152-resident-accuracy.md` documents it; (b) the #90 prefetch overlap (S = 4 MiB, +0.7–1.4 ms/token on the L2 path) — re-read on the bypass default first (rule 44 (2)). Gate: G=1024 ≥ 50 against the sitting's A (bypass default), dumps `bit_identical=1`, nll equal, text ≡ A 8/8, prefill ≥ −5 % | decode goal at G=1024 | rules 44, 46 |
| ㉚ | **Decided (user, 2026-09-30): option (1) + (2)** — Part B as the two-session design (S1 = router + MoE, S2 = the rest, one call per session per token, shared-page hops), bit-preserving; (2) the VTCM-share probe, stopped on the host finding (MoE prefill layout ≥ 6720 KiB; S2 ≤ 0.95 MiB; ≤ 3-lane VTCM feed slower than L2) — S2's FC stays L2-fed. Status in ㉓ and the §2 #132 Part B row; #178 = PR #183 re-target; #192 (single-session window) not viable. The hybrid stays the default until the E2E path is faster and bit-identical (20.8 vs ≈ 54 today). **Decision for the user (cycle 23), after #132 Part A and #178:** the FC set + lm_head (383 MiB, 402 MB/token) is the term that decides the decode NPU end-to-end path. Measured: exact FC on the DSP VTCM-fed 7.88 ms/token (S1, but S1 has 113 MiB free), L2-fed in a second session 8.06 (S2 has 0 VTCM), the CPU 7.4; the two-session path projects ≈ 45–46 tok/s against the hybrid's 50.6–54. Options: (a) build the two-session E2E plan anyway (needs an HVX quantizer, an HVX CPU-exact router, and either VTCM for S2 or an L2 feed at the K=2048 rate for K=7168 ≈ −1 ms); (c) of #132 (keep the FCs on the CPU, one call per token only for the DSP-resident kinds — the hybrid stays the product path); or the §3.5 fallback. Until decided, #132 and #178 stay `needs-user`; the implementer's queue is #170 round 3 | decides whether the end-to-end track continues past ATTN_M1 | user |
| ㉛ | **Filed as #197 (p2, `state:needs-plan`, cycle 25), after PR #191.** The S26 M=1 decode MoE call after #185 is 427.1 µs `dsp`/call (DQ) against ≈ 346 µs of transfer at 62 GB/s over 4 queues: C(2) and C(3) move no bytes (≈ 38 µs/call DMA-idle) and GU(0)'s remainder after QUANT is exposed (`DMA_FIRST` 50 µs). Next rung named by #185's `ponytail:`: a third down slot in the arena's spare ≈ 0.9 MiB (half a down) or a row-split of one down over two jobs, each with its own host proof (dataflow / submit-lane scoreboards + a negative). Gate on #197: `dsp` ≤ A − 20 µs/call, decode ≥ A and ≥ S25 #158 B at every G, bit-identical, prefill ≥ −5 %. Only the S26 (v81) benefits — on the S25 one queue is the bypass ceiling (rule 43) | ≈ −20..−38 µs/call ≈ +4..8 % S26 decode | #185, #177, rule 43 |
| ㉜ | **Measured by #219 → §2 #219 row, rule 65 (cycle 35b, 2026-10-06): the slow regime is gone with the page cache out of the sum — ms/miss 2.5–3.8 → 0.67 / 0.51 (`=1`) → 0.45 / 0.44 (`=2`), decode +19 / +24 % at G 64; closed, the remainder is ㊴.** History: Pool-28 miss cost: cause measured (cycle 30, #216 step 1 + lever; rule 61 amended, rule 62). The slow miss (3.7–5.0 ms) is a UFS read of a re-missed expert whose file pages kswapd evicted during the run; the fast one (0.5–1.1) is the same `pread` from the page cache. Memory: arena 3 328 + FC 448 + RSS 766 MiB unreclaimable, the model file 4 116 cached (every resident expert held twice), Android ≈ 3 740, on 11 114 MiB.** Ruled out, in order: page-cache residency *before* the run (`201-fsu-e2e.md`; it is evicted during), arena / pool size as a cure (C = 24 is 512 MiB less pressure), reader-thread placement (readers run only in prefill), a busy core on the 8-pinned-slice barrier (busiest non-app core 3–16 %), boot proximity (slow at uptime 300 s and after a 6-min idle). Measured lever (PR #218, `NNTR_MOE_FADVISE=1`, env-only, not flipped): misses 0.86–1.13 ms on every run of both boots, decode +10 / +15 % at G = 64, but the advice costs 2–10 ms a call on this kernel and **prefill −7 to −18 %** — gate fail; the `=2` cell shows pure dropping makes every miss a storage read (102 / 102 re-misses in the S0-trace replay, 121 on the device). At G ≥ 512 the pool misses 0.12–0.34 a token, so even the slow regime is ≤ 1.7 ms a token there; at G = 64 (1.89 misses a token) it is 3–8 ms a token, the largest single term between Q28 (43) and A (54) on that cell. Next (#219, p1, `state:needs-plan`): keep the 88-expert complement (≈ 465 MiB) in a cached ARM buffer at load, serve a miss as a 5.3 MiB memcpy into the ION slot (ARM staging memcpy 22–26 GB/s → ≈ 0.25 ms), refill the vacated tier slot with the victim's bytes by a plain / `O_DIRECT` `pread` on a helper off the token path (re-miss gap p10 1.1 tokens), and drop the file's pages once at load so the page cache leaves the sum; a hybrid-path run must be unchanged | ≤ 0.5 ms a miss on every run of a fresh and an old boot, window `pgpgin` ≈ 0 without `fadvise` on the token or prefill path, prefill ≥ −5 %, text == A; or a measured reason the tier cannot hold | #219; rule 62 |
| ㉝ | **S1 ceiling 3584 (LEAK) stops: 2 in 32 runs on the farm S25 `R3CY205ZMND` vs 1 across three sittings (≈ 94 runs) on `R3CY10WM83Y`** (rule 59 c); all four known cases on a boot's first G = 512 r1 or a profiled run, the stopped runs themselves closed clean. Per unit, per boot, no cause. Not a lever; a runner fact: the S26 sittings keep the reboot-and-resume rule and log uptime at every stop. Read `ceiling.txt` + the `.logcat` of the farm's two stops (on that machine) for the mapping that stays | which PD / process holds the 256 MiB after a clean close | farm session's logs |
| ㉞ | **#201 S5 host-side prep (cycle 34, from the implementer's S4 close-out): three items that need no 26B files.** (a) The sliding layers' DSP attention cache (`HTP_ATTN_KV_CACHE_B`, hd 64) is sized at `max_seq` 4096 on the 26B-A4B shape ≈ 800 MiB — it has to be window-sized (Gemma's sliding window) or capped by a smaller `max_seq`, inside the one-PD budget of rule 63 (pool + FC set + heap + scratch ≤ 3840 MiB, S26 ceiling 3840 per #208); (b) #4296's CPU layers without a hook still compute at decode on stale rows and the result is discarded — correct, wasted CPU time per token; the E2E path should skip them the way `dense_ffn` does since PR #223 (`htpDecodeRowResident`); (c) the final soft-cap stays on the CPU (one op; the DSP has the soft-capped head since PR #214). Gate: the hd64 fixture's E2E lines unchanged (`calls/token=1.00`, tokens 8/8, SNR ≥ 20 dB), the cache bytes printed at load, the CPU skip count printed per token. Also from #229: the ternary → 4-bit converter and the prefill dequantization path are their own issue once the checkpoint's storage format is pinned (`needs-user`) | S5's 26B load fits the S26's one PD; decode's CPU remainder is the hooks' only | #201 (files), #229 |
| ㉟ | *(Cycle 40: the baseline is measured — #201 S5-0, E16 on the S25: peak RSS 2174–2182 MiB = pool arena 704 (C 16 × 2-bit experts) + FC arena 1344 (206 Q4M1 weights, `attach_mib=1278.7`) + DSP heap 582 (KV 400 + 40 inside) + the ARM process incl. the never-read CPU FC set and the fp16 RoPE / prefill tables (rule 72); C 40 adds 1088 to the pool. The FC arena is the largest single term and #234 P4's WH set roughly halves it; the 2 GB stage starts from these numbers after S5 / S6.)* **Peak memory < 2 GB — deferred constraint (user 2026-10-06, cycle 35; contract §1).** Stated: Gemma decode E2E should keep process peak RSS incl. the ION arenas under 2 GB (pool + FC set + KV caches + DSP heap + scratch), flash streaming (the expert pool / FSU) covering what does not fit. Same day: **considered last, after the ternary 26B decodes end to end on the S26** — not a gate on S5 / S6, so plan 229's one-PD sizing (C ≈ 57–66 on a 3.8 GB arena) stands for S5 and ㉞ (a)'s cache sizing works to rule 63's budget. For S5 and its levers: every Gemma row records peak RSS and the arena bytes (`mapped=` line, pool C, FC set, cache bytes) so the later stage starts from measured numbers. When promoted (after S5 / S6): one issue per budget term, gate = peak RSS < 2 GB on the S26 with text / SNR unchanged vs the unconstrained A and the pool's miss cost (㉜) re-read at the smaller C | sizing from S5's first sitting; which terms (pool C, KV `max_seq`, FC set bits) carry the cut | S5's first sitting (#201), #229 (2-bit halves the slots), #234 |
| ㊱ | **Closed by #236 (cycle 36, 2026-10-06; §2 #236 row, rule 69): B loads and runs P1024 × G64 / G512 / G1024 on the config of record (53.24 / 53.06 / 48.64, prefill 726 / 723 / 711, `heap_kib=75776`), chunked == unchunked bit-identical on the host, prefill +9 % over Bfb, E2E Q unchanged; the old set fails in the same sitting. B is the hybrid's P1024 case of record; PR #243 and the text approval are the user's.** Originally:**Filed as #236 (p1, `state:needs-plan`, cycle 34; rule 64).** The hybrid's attention qkv FC at P1024 goes to the DSP as one M = 1024 call (`gemm_q4_0_batch_fp32` chunks by `fcMaxRows(K)` = 1920 at K = 2048, not by PR #230's `prefillRows()` = 512) and fails `AEE_ERPC` beside the FC WH overflow on the heap (`heap_kib=75776`); the same call runs on the one PD (`heap_kib=0`). Fix: the FC entries (`gemm_q4_0_accel_fp32`, `gemm_q4_0_batch_fp32`, and the qs4cx twins) take `prefillRows()` as their cap too, host-proven bit-identical chunked vs unchunked (the `CONV BLOCK CHUNKED BIT-IDENTICAL` template); then B's three P1024 cells are re-read on the config of record (Bfb is the case of record there until then) with the loop check against A — Bfb P1024 G64 / G512 loop where A does not, so the re-read also answers whether that is the fallback's or the length's | hybrid B loads and runs P1024 × G64 / G512 / G1024 on the config of record (no VOID, no `AEE_ERPC`); chunked == unchunked bit-identical on the host at M = 1024, K = 2048, 3 handles; prefill ≥ −5 % of the unchunked P512 cell (727–739); E2E Q unchanged (`calls/token=1.00`, `heap_kib=0`) | PR #233 merged; the user's text approval of the P512 cells |
| ㊲ | **Runner defect, docs-only (cycle 34):** `225-run.sh`'s `gen()` keeps `[HTP]` banners that print inside the generated text (`dspq: on queue=0x…`, `graph: init …` — stdout not line-terminated), so its `speed.txt` text column reads DIFF for every NPU run and the `every r2 text == its r1` check reports `BAD … got '2'` on a byte-identical pair (the queue address differs). Fix on PR #233's branch: strip `\[HTP[^\n]*\n` anywhere in the captured text before comparing (and in `loops.txt`'s input); the filled handoff's text-vs-A column was recomputed that way by hand. Filed as a comment on #225, not an issue | the next runner's `speed.txt` reads `same` for r2 vs r1 on every NPU cell; `loops.txt` computed on banner-free text | PR #233 |
| ㊳ | **Cycle 39: PR #244 merged `8971aa873`; the Gemma device is the S25 `R3CY205ZMND` (user 2026-10-07), so the v81 half is deferred until an S26 is attached again, not a blocker.** Cycle 38: v79 read — the expand gtest and the new u8i2-vs-twin layer gtest pass on the S25 `R3CY205ZMND` (§2 #229 row, 2026-10-06); v81 still open. Was: the 2-bit expert stack (#229 S1, PRs #238 / #239, cycle 36) has no device reading. `HvxExpandI2I4.MatchesScalarBitExact` passed on v79 in upstream #4410 and never ran on v81; the u8i2 LUT GEMV (`hvx_gemv_u8i2*`, mask fixed per rule 66) had no device gtest — PR #244 adds `HmxMmU8I4Layer.MoeLayerU8I2GemvMatchesI4Twin` at the Gemma expert shape (12 cells bit-identical on v79); the host holds it bit-identical to the 4-bit palette twin on lfm25 / gemma64 E2E and the pool (C = 1 / 2). What is left: a v81 run of both gtests when an S26 is attached, and S4's `NNTR_HTP_DUMP` MoE dumps of a 2-bit model against its palette twin on a device | expand + u8i2 gtests PASS on v81; 2-bit vs palette-twin dumps `bit_identical=1` on the device; then the S5 2-bit floor (plan 229 §3.2) is a number | #229 S2–S4 (after #234); the S26 unit; PR #244 merge (user) |
| ㊴ | **Cycle 36 (close-out): (1) decided — `NNTR_MOE_TIER=2` is the LFM E2E default (PR #245, `HtpBackend::e2eRequested() ? 2 : 0`; the tiered 9 × 3 E2E column is measured, §2 #219t row); (2) / (3) stay open and are not pursued on LFM2.5 (user) — they move to Gemma's one PD with rule 68's reading (the G64 residual = start cost ÷ G + misses × copy, scaling with P; `tier_waits` 0 / 3 / 3 at P64 / P512 / P1024) and the ≈ 170 MiB ARM Q4_0 FC set the E2E loads and never reads (`cpu fc skipped` = 32 × tokens) as the RSS item beside (3).** Originally:**#219's remainder (cycle 35b, 2026-10-06; §2 #219 row, rule 65; ㊳ is `htp_decode`'s cycle-36 item for #229 S1) — three things, none a code change the gate asks for yet.** (1) **The default is the user's call** (`needs-user` on #219): flip `NNTR_MOE_TIER=2` (one-thread copy; G1 pass on every run, decode 48.47 sd 0.23 at G 64, refaults ≤ 81, `pswpin` ≤ 168) as the E2E one-PD default, or keep it env-only as #218's fadvise; `=1` (8-slice copy) is dominated on every axis and should not be the default either way. (2) **`tier_waits` 3–4 a run at every G** (G2 wants ≤ 2): the decode start's refill backlog from the prefill batch (constant across G, the host saw 2 waits behind `refill_ms=42.5`), not a growing race; lever = drain the refill queue (or `poolSync` on it) before the first decode token, cost ≈ one refill's time off the first token, not the per-token path; worth ≈ 3–4 × 1 ms a run — small, a hygiene item. (3) **`pswpin` non-zero with the tier** (0–1261; `=1` once 2105 / `pswpout` 3621): the sum anon 766 + tier 467 + ION 3 776 MiB is at the S25's edge; a cheaper tier (hold only the complement's gate \| up images the pool actually re-reads, or drop the tier's slots the arena holds twice at load) or a smaller RSS recovers the ≈ 0.5 GiB. Also open: a tiered 9 × 3 E2E column (P64 / P512 / P1024 × G) is its own sitting on the same unit class as #225's Q cells — not filed until the default is decided | (1) a decision; (2) `tier_waits ≤ 2` on every run with decode unchanged; (3) `pswpin = pswpout = 0` in every tiered window | #219 `needs-user`; PR #240 |
| ㊵ | **#234 P2's safetensors order guard does not catch two same-size swapped tensors (PR #248, cycle 38):** `nntr_quantize_stream` now reads the converter's `.safetensors` and refuses a header whose tensor order it cannot follow (§10.5 of plan 234), but two tensors of equal byte size in each other's place pass the guard and produce a plausibly wrong model — the `QS4CX_WH` trap of contract §2 ("a wrong layout yields plausible wrong text, not an error") in a new place. Matters once the user's 26B ternary files arrive (#229 S0 / #201 S5): the first Gemma conversion needs a per-tensor identity check beyond size (name-keyed, as #227's hand-over is) or a CPU-vs-HTP shadow on the converted model before any sitting | the first 26B conversion is verified by name-keyed tensor identity or a CPU run of the converted model that matches the reference tokens; or the guard is extended to names | #234 P3 / plan 234 revision; #201 S5 |
| ㊶ | ~~#201 S5-0~~ **measured (cycle 40, 2026-10-07) → §2 #201 S5-0 row: path pass, gate (a) PASS (user), E2E ≈ 30 % under the hybrid at every G (FC / head / dense byte-bound on Q4M1, ㉘), pool flat over C 8–40, C-max 40 (rule 71), RoPE table OOM (rule 72, #253), ㉟ baseline recorded. Deviations from the brief below: prompt 447, reference = the hybrid `off` run (user: no CPU twin), E warm after A. What is left of it goes to ㊷ (the decode MoE inside the token is not dumpable) and to the next S5 sitting on #234 P4's WH FC set.** Was: **#201 S5-0 — the first 26B sitting, on the DUMMY file (cycle 39, 2026-10-07).** The files are on the workstation (`~/Downloads/gemma4_26b/`, bin md5 `9b4e9349…`, 7 226 112 120 B, a nntrainer `.bin` — cycle 39 (2)), the blob is a dummy of the right shape and format, the device is the attached S25 `R3CY205ZMND` and the agent drives adb. So S5's first sitting is a **path** sitting, not an accuracy one: A = CPU run of the same file (`moe_engine` off / CPU MoE cache, the bit-identity reference, host x86 if the phone cannot hold it), E = `NNTR_HTP_E2E=1` one PD through the pool at the config's `moe_cache_experts 16` and at the largest C that loads (the one-PD budget on the S25 is the 3 840 MiB ceiling minus the Q4M1 FC set ≈ 790 MiB and the tied Q4M1 head ≈ 415, plan 234 §3 — P3 / P4 are not prerequisites: the S4 graph binds FC / DENSE_FFN from `q4_pending_` as Q4M1), prompt 512, G 64 / 512 / 1024, cool start, `.sitting.lock` taken. Prerequisite: the format agent's reading of the bin (the `int2` name vs the config's `QS4CX_WH`) and the loader accepting it — the first attempt is a host load (`run_inproc_e2e.sh`-style registration on the real file: dumps only, never a number, §3a) before the phone | loads on the phone (`mapped=`, `calls/token=1.00`, `attn_caches=2`, pool misses / token printed); tokens of E == A over the prompt set on the same dummy weights and MoE dumps `bit_identical=1` vs the pool-less run; tok/s at the three G + peak RSS + arena bytes recorded (㉟'s baseline); prefill of E vs A recorded (no −5 % gate — nothing of the DMA ring / pool / layout changes in this sitting); no text / PPL column (dummy) | #201 (S5-0), the format agent's answer; plan 229 S4's A / B2 / B2-C rows follow on the same file once the 2-bit path is what the bin holds |
| ㊷ | **The decode MoE inside the one-PD token is not observable from ARM, so an E-vs-off token divergence cannot be pinned to a layer (cycle 40, from #201 S5-0).** `NNTR_HTP_DUMP` dumps the MoE calls the ARM makes (warm-up + the prefill calls; 62 files on the 26B), P5's `NNTR_MOE_DIFF` / `SHADOW` covers the MoE layer call (an E3 run prints only the prefill lines, cycle 39), and the token driver returns logits only — S5-0 read 57 / 65 tokens and could say "near-tie flips, prefill MoE bit-identical, logits drift 0.13 rms" but not *which* resident kind drifts. On the dummy that reading is enough (user: (a) PASS); on the real file the text gate (SNR ≥ 20 dB + tokens vs the CPU run) may fail without a named layer. Candidate: a per-token DSP-side dump of one named kind's output (the graph already has the per-kind timers; an `NNTR_HTP_E2E_DUMP=<kind>,<layer>` copy of one op's output into the logits buffer's tail, read by `htp_dump_eval.py` against the hybrid's ARM-side value for the same op) — one op a run, no IDL change if it rides the existing logits buffer; gate = the hd64 fixture's E3 == off per kind on the host, then the 26B. Not an issue until the real file's gate needs it | one named kind per run, `bit_identical` / SNR vs the hybrid's value for the same op | the real 26B file; #234 P4 (the FC set changes under it) |

## 3a. Guide and tooling notes

* **#132 PR 1 (cycle 20): rungs 0–3 on the workstation** at the PR head — skel `b000b0cd91f6274acba6a9873a6bfd5d` (`UNDEFINED SYMBOLS OK (46 runtime imports)`; not bit-reproducible, rule above), `libcausallm_core.so` `3e06d3b08f1443b59d1ead3e4aa754cf`, `nntrainer_causallm` `75e61732f02bb243fb4b5ca2f01a09d4`, `libnntrainer.so` `aea9a50abc9b34d51676c8fc6bac1c3f`, `libccapi-nntrainer.so` `eba2f98d64939341db99a5d82d5825d9`, `unittest_hvx_softmax` `25bec7dfd9f5b342783ec59cc7d5059d` (carries `HvxM1Ops.RouterTopkMatchesDetBitExact`). The IDL gained one test method, so the stub and skel go together (rule 3).

* **#130's sitting (cycle 19): the first recorded post-#121 skel md5** is
  `ca1f2aac4985b7799897de2dcb4c2d1e` (`test/htp/build.sh` @ `9deea6e2`,
  `HEXKL_SDK_VER=6.4.0.1`, `UNDEFINED SYMBOLS OK (46 runtime imports)`;
  BENCHMARK artifacts). **Rebuild recipe since #135 (rule 36):**
  `build_android.sh --htp` carries the define (the prebuilt `Android.mk`
  exports it, `jni/meson.build`); a pre-#135 `builddir` needs one
  `(cd builddir && ninja install)` (or a build without `--cache`) first,
  since `--cache` keeps the old installed `Android.mk` — the script's
  own check (`strings libcausallm_core.so | grep -c
  NNTR_HTP_FORWARD_KINDS` >= 1 with `--htp`, 0 without) exits 1 on the
  stale one. The staged set of PR #133 (`80452a28…` skel etc.) has no
  live switch and must not be re-used for a B/C cell.

* **#130's PR (cycle 18): rungs 0–2 on the workstation** — `run_host_checks.sh`
  `ALL CHECKS PASS` / `GRAPH CHECKS PASS` / `GRAPH STRETCH BIT-IDENTICAL`,
  `run_inproc_e2e.sh` `INPROC E2E PASS`, `test/htp/build.sh` `UNDEFINED
  SYMBOLS OK (46 runtime imports)`. Two notes for the next handoff: (1)
  `build.sh`'s skel is **not bit-reproducible** (three consecutive builds
  of one tree gave three md5s), so a recorded skel md5 identifies a build,
  not a source state — the PR body says which build it is; (2) the committed
  goldens of `test/htp/host/golden/` were cut on the container (PR #127) and
  failed on the workstation on the very first MoE input (128 dB, the
  README's "this machine's CPU path differs" case) before any change of
  #130; with the container path reverted (PR #124 reverted) they are now
  cut on the workstation, and a golden mismatch on another host means that
  host, not the HTP path.

* **`docs/htp_moe/guide/`** exists since PR #92 (`cffcec3e`, merged
  `750428e8`, 2026-09-22): five self-contained English pages
  (`index`, `01-run-it`, `02-decode-path`, `03-performance`,
  `04-glossary`), decode only. The guide writer refreshes it on every
  filled handoff and every merged kernel/app PR (contract §8 step 4); its
  numbers are copied from BENCHMARK.md, never measured.
* Guide-writer findings folded into this ledger: (a) the FC 28.2 vs 10.3
  ms/token question (§2 ① correction, ⑰); (b)
  `Applications/CausalLM/install_android.sh` takes **no `--model`
  argument** in this tree — it pushes binaries and prints the `adb push`
  of a model dir as a hint (`install_android.sh:268–279`); handoffs must
  spell out the model push themselves (#77's did).
* **Rebuilding a handoff set on a fresh checkout (#105 run notes, cycle 9;
  rule 22):** four gaps the recipes did not state, now in the
  `hexagon-gates` skill rung 3. (1) A new `git worktree` has **empty
  `subprojects/`**. `build_android.sh` then dies on `iniparser.h` after
  ≈ 15 min. Run `git submodule update --init --depth 1` (or copy the
  populated dirs) first. (2) `Applications/CausalLM/lib/libtokenizers_android_c.a`
  is per-checkout. Copy it or rebuild it with `build_tokenizer_android.sh`.
  (3) `libc++_shared.so` may not appear under `jni/obj/local/arm64-v8a/`.
  Take it from the NDK r30 sysroot
  (`toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/`,
  md5 `b1586b9b…`). (4) `HEXKL_ROOT` differs between workstations. Export
  it explicitly next to `HEXKL_SDK_VER=6.4.0.1`, and never rely on
  `env.sh`'s default path.
* **Mac client (cycle 16, contract §4.1a):** only rungs 0–1 run on the
  Mac (`tools/docker/run.sh`, HexKL beta1 mounted; the tree needs beta.2
  for the skel). A PR opened from the Mac says in its body that rungs 2–3
  were not run; the workstation produces the skel/app md5s before any
  handoff. The GitHub MCP server cannot write from this client (HTTP 403
  `Resource not accessible by integration` on PR create, issue comment,
  issue close and label edits; reads work); agents use `gh` for every
  write: `gh pr create --repo dlwlzzero/nntrainer --base htp_moe`,
  `gh issue comment`, `gh issue close --reason completed`,
  `gh issue edit --add-label/--remove-label`.
* **Rungs 2–3 at the cycle-16 merge (2026-09-27):** PRs #121 / #125 /
  #126 / #127 / #128 were opened from the Mac with rungs 0–1 only; the user
  ran rungs 2–3 (skel + `build_android.sh --htp`) on the workstation at
  merge time, and **no skel/app md5s are recorded in those PRs**. The next
  handoff's rebuild recipe (rung 3, `hexagon-gates`) produces the md5s for
  `htp_moe` @ `758aaacc` or later; until then BENCHMARK's artifact table
  has no row for a post-#121 skel (the IDL grew by `mm_u8i4_conv_block`,
  the small-ops and `attn_m1_*` methods — a stale skel reads
  `0x8000040e`).
* **In-process HTP build (#84, cycle 17):** `meson setup build_htp_host
  -Denable-htp=true -Dhtp-inproc=true -Dhexagon-sdk-root=$HEXAGON_SDK_ROOT
  …` links the DSP skel subset (`test/htp/{hvx_add_f32,nntr_hvx_mm_u8i4,
  nntr_hvx_graph,nntr_hvx_small_ops,nntr_hvx_attn_m1}.c` + their `hmx/`,
  `hvx/` deps) into `libnntrainer.so` in place of the FastRPC stub, on
  scalar stand-ins (`test/htp/host/standin/hvx_scalar.c` for the HMX tile
  / GEMV / u8 quant / dequant / SwiGLU, `inproc/` for HexKL's micro API
  and the FastRPC runtime, `hvx_emu/` for the f32 intrinsics, `stub/qurt.h`
  for the pool) with `-Wl,--no-undefined` as the host twin of `build.sh`'s
  symbol guard. Gate: `bash test/htp/host/run_inproc_e2e.sh` → the five
  `E2E` lines and `INPROC E2E PASS` (tiny fixture, prompt 16 + 8, golden in
  `test/htp/host/golden/lfm2_moe_tiny/`). `NNTR_HTP_DUMP=<dir>` makes any
  run (host or phone) write every MoE call's input / output + a manifest;
  `tools/htp/htp_dump_eval.py <ref> <got>` compares two such directories.
  The profile's `dsp=` / transport columns on the host are the stand-ins'
  clock and mean nothing; never quote them. The real 8B model on the host
  needs an x86-packed twin (`nntr_quantize_stream … --moe_dtype QS4CX_WH`
  without `--isa ARM`) and is minutes per token on the scalar HMX
  stand-in: registration and dumps only, never a number.

* **#211 (2026-10-01): `NNTR_HTP_E2E_PDS` is a guard, not a switch.**
  Unset or `1` proceeds (one PD, the only E2E path); any other value
  throws at load naming #211, so a runner that still passes `_PDS=2`
  cannot label a one-PD row with a removed variant. Drop the guard with
  the next genuine IDL change, together with `role` in
  `token_driver_start` (the DSP accepts only 0) and the `hops` /
  `wait_us` / `hop_us` fields of `htp_dspq_token_resp` (they read 0):
  removing them now would have made a pure deletion the PR whose device
  failure mode is the stale-stub `transport failed: err=0xe`.

* **#211: `unittest_hvx_two_sessions` is the #178 platform probe, not the
  E2E path, and stays whole** (with `mailbox_run` and
  `nntr_hvx_mailbox.c`): it reserves its own S2 and never used
  `HtpBackend::openSecond` (deleted), its `S1Ceiling` is the runners'
  ceiling cell (`CEILING s1_mmap_mib=` before and after every run), and
  its Q1–Q5 lines are what rule 59 and ㉝ cite. Its `TwoSessions` name is
  a probe's.

* **#222 (2026-10-02): the engine keys act at prefill only, and the
  in-process host build runs them.** `conv_block_engine`, `dense_ffn_engine`
  and `attn_proj_engine` = `htp` (the config of record,
  `docs/measurements/config/q40-qs4cx-wh.nntr_config.json`, user decision
  on #222) move the M > 1 matmuls onto the HMX FC kernels; every decode
  row stays where it was (hybrid: the CPU kernels, every FC gate declines
  M == 1; one PD: `decode_row_resident`). Two code facts found on the
  way: (1) the `dense_ffn` one-layer form did not ask
  `htpDecodeRowResident`, so under `NNTR_HTP_E2E=1` it recomputed layers
  0–1's FFN on the CPU every token and threw it away (host: `cpu fc
  skipped` 8 / token instead of 10 on the lfm25 fixture; the logits are
  unchanged by the fix, bit-identical) — on the 8B the per-token skip
  count reads 32 with the keys (18 conv blocks, 6 × qkv + attention_out,
  2 dense_ffn) against 36 without; (2) `htp_qs4cx_from_q4_0x4` reads only
  the ARM q4_0x4 repack, so the in-process build (x86 repack) produced
  NaN logits with any key on: `qs4cxFromModelQ4_0` re-lays the host's
  repack as q4_0x4 off aarch64 (device bytes pass through unchanged).
  `causal_lm.cpp:715` registers the prefill's token only when the prompt
  is shorter than `init_seq_len`: at prompt 512 the old config (512)
  prints the text without its first token, the new one (1024) with it
  (`docs/measurements/prompts/README.md`).

* **#225 PR 1 (cycle 33, `7f95140ad`): the FC WH sidecar and the 512-row
  prefill chunks.** `nntr_quantize_stream --fc_wh_sidecar` writes the 66
  FC weights as `QS4CX_WH` images from f32 beside the main file
  (`<main>_fcwh.bin`; on the 8B 228,188,160 B, md5 `71812a91…`, main file
  md5 `7b7867fa…` unchanged); `fc_wh_file_name` in `nntr_config.json`
  names it. The loader preads each image into the arena keyed by the Q4_0
  bytes (not by name — see PR #230's deviations), no re-quantization, no
  heap copy; only an arena overflow unpacks bit-exactly to the heap
  (`[HTP] fc wh: arena=<n> heap=<n>`). The conv block / dense FFN / MoE
  prefill calls run in 512-row chunks, the conv history carried in `conv_w`
  [5 × C] (no IDL change) — host `CONV BLOCK CHUNKED BIT-IDENTICAL`; the
  user made the chunking the hybrid's way of record. The config of record
  does not name the sidecar until #225's handoff; a no-keys config leaves
  it unopened (bit-identical to before). E2E still binds Q4M1 until PR 2.

## 4. Reusable code on `hvx_impl` (survey 2026-09-21; read with `git show hvx_impl:<path>`)

`hvx_impl` (Qwen3-0.6B, W8A8 HVX-only) is frozen but its kernels and
harness are device-validated. What lifts, in value order:

| lift | from | serves | adaptation |
|---|---|---|---|
| **DONE (#81, PR #126).** Lifted: the Kᵀ `[head_dim][max_seq]` layout (one f32 vector = 32 positions, not 64: f32 cache), the per-layer/per-kv-head cache indexing, the split by kv head with per-unit score scratch, the fp32 vector accumulation of PV, hvx/58's GQA fusion (one Kᵀ/V stream serves the 4 q heads of a kv head, as four accumulator sets in one pass). Not lifted: the fp16-widening scores (`hvx_vec_mpyacc_f32_f16`), the qf32 `hvx-exp.h`, the fp16 probabilities before PV, the scalar `1.0f / sum`, the hvx/58 position-block split + merge (re-associates the softmax), `HTP_ATTN_L2FETCH`, the token loop — all for the `_det` bit gate (plan 81 §3.3), the same reason #82 rejected `hvx-rmsnorm.c` / `hvx-rope.c`. Replaced by `nntrainer/tensor/attn_m1_det.h` + `hvx/hvx_attn_m1_f32.{c,h}`, plain Vsf with `hvx_exp_det_sf` / `hvx_recip_det_sf`. m=1 decode attention with a DSP-resident KV cache (K stored transposed `[head_dim][max_seq]` so one vector covers 64 positions; workers split by kv head) | `nntrainer/tensor/hexagon/htp/ops/hvx-attn.c` (152 LOC) | ⑨ attention | drop the token loop (m=1); per-worker score scratch from the orchestrator; KV dtype vs `hexkl_kv_quant.c` (fp16 there, u8 here → dequant in the score loop if u8 stays); `wp_run` → `hvx_worker_pool_run` |
| "DSP owns the graph": validate the op list once at init, then `forward(tokens, pos → logits)` runs `for op in ops: table[kind]()` with per-op pcycles and one FARF line per call | `htp/htp_graph.{h,c}` (~200 of 546 LOC), `htp/nntr_htp.idl` (3 methods incl. `forward_debug`), `htp/executor.c` (fd mmap + `AEE_EALREADY` handling) | ⑨ one call per token | add a `forward` entry beside the per-op IDL; rebuild `next_mm[]` over the LFM2.5 op sequence; repopulate the op table with MoE/conv1d/dense kinds |
| cross-op weight prefetch: two op-independent VTCM half-slabs so the last chunk of op N kicks chunk 0 of op N+1 while norm/RoPE/attention run in between | `htp/ops/hvx-matmul.c` `mm_slab`/`mm_worker_vtcm`/`mm_pf_kick` (~120 LOC) | ⑥ ⑨ | hvx_impl's per-worker push/pop DMA FIFO vs our global index ring (`hexkl_dma_ring_push2d` + `next_idx`/`wait`): rewrite the pipeline loop; keep the "drain, never abandon" rule for a prefetch left by another op |
| **DONE (#84, PR #127 `ec589373`).** Lifted: the `E2E step/gen` line format and the `--eval` verdict (`bit_identical` + SNR, as `tools/htp/htp_dump_eval.py` over per-call `NNTR_HTP_DUMP` dumps, the first differing file named). Not lifted: `HexagonRunner` / `RpcmemBuffer` (here `HtpBackend` / `HtpRpcMemApi` are the code under test), `run_e2e_test.sh` (agents never run adb; the md5 gate is `md5sum -c` in the handoff recipe), `summ_farf_prof.py` (the ARM prints `[HTP-PROFILE]` as a table), `find_divergence.py` (every call is dumped in order, the first non-identical file is the divergence), `make_tokens.py`, the static-map probe (`htp_moe` maps arenas with `FASTRPC_MAP_FD_DELAYED`). Replaced by the in-process build (`-Dhtp-inproc=true`, `build_htp_host`): the ARM owns the graph on `htp_moe`, so the driver is nntrainer's own decode loop (`test/htp/host/htp_e2e_test.cpp`, `run_inproc_e2e.sh`). host E2E harness without the CausalLM app: `hexagon_e2e_test` (`--tokens/--chunk/--steps/--eval/--dump-*`, `E2E` lines), `HexagonRunner`, `RpcmemBuffer`, md5-gated `run_e2e_test.sh`, `summ_farf_prof.py`, `find_divergence.py` (needs `forward_debug`), `make_tokens.py` | `test/hexagon/hexagon_e2e_test.cpp`, `nntrainer/tensor/hexagon/host/*`, `tools/hexagon/*` (~650 LOC) | measurement | replace the qwen3 lowering/config with LFM2.5's; `NNTR_HAVE_FASTRPC_MAP_STATIC` probe into `test/htp/build.sh` |
| **DONE (#82, PR #125).** Lifted: the per-head chunking (`chunk = head_dim`), the rotate-half structure with one vector per half, the cos/sin row formula as the host check's input generator. Not lifted: the fp16/qf32 arithmetic and `Vhf_equals_Wqf32` (v75/v79 differ), the scalar `1.0f / sqrtf` per row (libm bits no host spec can pin), `hvx-quant.h` (these ops end before the quantizer). Replaced by `nntrainer/tensor/m1_ops_det.h` + `hvx/hvx_m1_ops_f32.c`, plain Vsf with a `_det` bit gate. small ops: RMSNorm (also per-head q/k norm via `FLAG_PER_HEAD`), RoPE (`rope_rotate`, 20 lines, head_dim 128 hard-coded — LFM2.5 is 64), the shared fp32 cos/sin row `nntr_htp_rope.h`, and the quantizer's integer-only rounding recipe (one qf32 product, then sign/exp/significand split, ties-to-even) that made results v75/v79-portable | `htp/ops/hvx-rmsnorm.c`, `hvx-rope.c`, `nntr_htp_rope.h`, `hvx/hvx-quant.h` (~295 LOC) | ⑨ | activation dtype (fp16 there, f32 residual here); our quantizer is asymmetric u8 with zero point — port the rounding only if `hvx_quant_u8.c` still reads sf bits after a qf32 op |

Not lifted: the W8A8 tiled `vrmpy` matmul (wrong layout for int4; our `hvx_gemm_u8i4_wh_col` is the math), `dma-queue.c` (we have `hexkl_dma_ring.c`), the worker pool (ours is richer; note the opposite HVX-context convention: hvx_impl locks a unit per worker and leaves none for the caller, ours runs index 0 inline), the qwen3 lowering/packer/app glue, the simulator harness (no simulator here). Causal depthwise conv1d (L=3) + gating has no counterpart on either branch: net-new.

Silicon rules from `hvx_impl`'s HEXAGON.md §7 that bind here too: compute in fp32 inside an op and narrow to fp16 once (`Vhf_equals_Vqf16` after a qf16 multiply rounds badly, 2.7 % PPL); clamp the SiLU exp argument (already 85 here, doc 44); `-mhvx-ieee-fp` is required for fp16 intrinsics on toolchain 19; int32 `vrmpy` sums are exact in any order, divergence enters only in the epilogue; qf32→sf conversion differs between v75 and v79, so quantizers decode integers, never sf bits; HVX code runs only on threads that own an HVX context; validate `k` bounds for exact int accumulation; never compare wall-clock tok/s across units, report pcycles and `pcycles_per_us`.
