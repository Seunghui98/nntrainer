# 134 — Decode-side teacher-forced perplexity (`NNTR_PPL_DECODE`)

Issue #134 (p1, cycle 20, 2026-09-28). Base `htp_moe` @ `b12b6fe0`
(contains `16cc864d`). Evidence that makes this the gate for the per-token
entry: `docs/measurements/136-forward-text.md` on
`origin/htp/136-forward-text` (§ Dump sitting, § Kernel-level check). Each
resident kind is correct to the last bit on silicon (RMSNORM 131–148 dB
against the NEON order). The quantized pipeline turns those last-bit
differences into 34 dB at the first MoE input and 18 dB at the last MoE
input. Two such perturbations flip the greedy token at pos 512. So
"text identical to the CPU run" cannot hold for any resident kind, and
`NNTR_PPL` cannot help either: it scores prefill rows only, so A and B print
the same line whatever the DSP does at decode.

## 1. Goal and gate

**Goal.** Add an app mode, off by default and never set in a tok/s cell.
After the prompt's prefill it decodes token by token, feeding a given
continuation instead of the argmax. It sums −log p(target) over each decode
step's logits and prints

```
[PPL] decode tokens=<N> nll/token=<x> ppl=<e^x> top1=<M>/<N> source=<self|file> nll_sum=<%.17g>
```

It also prints one `[PPL] decode step=<k> target=<t> nll=<%.17g> top1=<id>`
line per scored step, for paired statistics.

**Gates of the PR (host, all must print):**

1. `E2E ppl-decode self==forced tokens=7 identical=1`. This is the run()
   path on the hd8 fixture. The free run writes its greedy continuation.
   The forced run reads it back. Every `[PPL] decode step=` line is equal
   to 17 digits, and the two runs' MoE dumps are `bit_identical=1` in
   `htp_dump_eval.py`.
2. `E2E tokens run==adapter 8/8`. The run() path's generated ids equal
   the `E2E gen` line of today's adapter path, so the app loop under test
   is the loop the E2E already gates.
3. `E2E ppl-decode index |dec-(S24-S17)|=<d> ok`, with `d < 1e-2` nats.
   This is forced decoding of a **non-greedy** continuation (ids 16..23 of
   the deterministic prompt sequence: `23 30 7 14 21 28 5 12`, all
   distinct) checked against prefill `NNTR_PPL` on prompts of 24 and 17
   tokens. The difference of the two prefill sums scores exactly the same
   seven targets. The greedy continuation of both fixtures is `16` at
   every step, so gate 1 alone cannot see an off-by-one. This check can:
   a shifted target reads O(0.1–1) nats per token on this fixture.
4. `E2E ppl-decode hd64 off=<ppl> on=<ppl> delta=<%> top1=7/7`. The hd64
   fixture: the six-kind entry forced on the switch-off continuation. The
   PPLs are printed; the gate checks only that both are finite and that
   top1 is 7/7. That follows from the existing `fwd==off 8/8` policy line.
5. Every existing `run_inproc_e2e.sh` line still passes (`INPROC E2E
   PASS`). `*Lfm2Moe*` gtests 6/6 pass. `run_host_checks.sh` prints `ALL
   CHECKS PASS`.

**Gate of the device handoff (§4 step 5, read in §3.3):**

* Decode PPL of **B (six kinds)** vs **A (switch off)**, both forced on
  A's own greedy continuation from the same sitting, at **G = 512**:
  `PPL_B / PPL_A − 1 ≤ +2 %`. Otherwise B fails, whatever the text approval
  says.
* The reading is valid only if all three null checks pass, all to 17
  digits:
  * A-forced ≡ A-self.
  * C (`KINDS=MOE`) ≡ A. The dump sitting found MOE-only bit-identical.
  * Each variant's G=64 step lines ≡ the first 64 of its G=512 lines.
    With teacher forcing the G=64 run is a prefix of the G=512 run.
* The sitting is also valid only if the paired noise band satisfies
  `2·SE(Δnll) < 0.01` nats/token (≈ 1 % PPL).

**Standing gates:**

* Prefill ≥ −5 % of A. The mode does not touch prefill, and a PPL run is
  never read for tok/s.
* A's generated text with nothing set is identical to the switch-off text
  of #136's sitting (the same app arithmetic, off by default). That proves
  the tok/s path is untouched.
* On the host, `E2E tokens htp==cpu 8/8` stays.
* The contract's "text identical to the CPU run" is read, for decode-only
  changes, through the approval table plus this decode PPL (§3.3).

## 2. Where it lives

| file:line (verified at `b12b6fe0`) | change |
|---|---|
| `Applications/CausalLM/layers/tie_word_embedding.cpp:383` `static double nllOf(...)` | Becomes the public static member `TieWordEmbedding::nllOf`. The body is unchanged. Its one caller at `:536` compiles unchanged. |
| `Applications/CausalLM/layers/tie_word_embedding.h:182` (next to `takePpl`) | Add `WIN_EXPORT static double nllOf(const float *logits, unsigned int vocab, unsigned int target);` with a one-line brief. |
| `Applications/CausalLM/models/causal_lm.cpp:532` (`ppl_on`) | Read `NNTR_PPL_DECODE` once, next to `NNTR_PPL`. Load the id file if it exists (source=file), otherwise remember the path (source=self). Validate each id `< NUM_VOCAB` and require a non-empty file; this is a trust boundary, so it throws. Refuse with one `[PPL] decode unsupported …` line, and run normally, when `BATCH_SIZE != 1`, `SKIP_PREFILL` (`:584`) or `logits_processor != nullptr`. |
| `causal_lm.cpp:609` (prefill `id_list = generate(...)`) and `:640–643` (first decode input) | source=file: if `id_list[0] != cont[0]`, print `[PPL] decode warning: prefill top1 != continuation[0]` (a stale or wrong-prompt file), then feed `cont[0]`. source=self: record `id_list[0]`. |
| `causal_lm.cpp:653–707` (generation loop) | Before `generate()` at `:663`, which applies the bad-word penalty in place: `nll += TieWordEmbedding::nllOf(output_interval[0], NUM_VOCAB, target)`, where target = `cont[step]` (file) or the argmax taken after `generate()` (self). Print the step line. Count `top1` (argmax == target). In file mode, feed the target at `:669` instead of `ids_list` and register it as the output. Stop when the targets are used up. In file mode, ignore the EOS check (`:684–700`). |
| `causal_lm.cpp:709` (after the loop) | Print the summary line. source=self: write `[id_list[0], t1 … tG]` whitespace-separated to the path (`<fstream>` is already included). |
| `test/htp/host/htp_e2e_test.cpp:74` (options), `:101` `run()`, `:135` | New `--run` flag. It builds the prompt text from the same deterministic ids (1→`hello`, 2→`world`, k→`tok<k>`; the fixture tokenizer is WordLevel + Whitespace with no post-processor, so the text maps 1:1 back to ids). It calls `model.run(text)` with `num_to_generate = steps − 1` (run() emits prefill + G tokens), then prints `E2E gen` from `tokenAt(prompt + k)`. Without `--run`, behaviour is unchanged. |
| `test/htp/host/run_inproc_e2e.sh` after `:190` (end of (f)); header `:20–35` | New block (g) with gates 1–4; header lines for them. |

**Consumers that do not move:** no contract changes, and that is
deliberate. The IDL `test/htp/nntr_hvx.idl` and `generate_stub.sh` output,
`HtpComputeOps` (`nntrainer/tensor/htp_backend/htp_compute_ops.cpp`), the
quantizer tag (`nntr_quantize_stream`), the loader check, the
`NNTR_HTP_PROFILE` stage tables and `tools/htp_fc_report.py` are all
untouched. No skel rebuild is needed for correctness. The handoff reuses
the `htp_moe` skel, built once and recorded by md5.

**Collision:** `origin/htp/136-forward-text` also edits
`run_inproc_e2e.sh` (+57 lines) and `htp_compute_ops.cpp`. Block (g) goes
after (f) so a rebase is mechanical. Whichever PR lands second rebases.

## 3. Design

### 3.1 Chosen: score the loop's own logits in `causal_lm.cpp`

`output_interval[0]` at `:663` holds the exact logits the greedy pick is
taken from. Scoring them is model-agnostic (tied or untied lm_head, any
weight type). The cost is one `nllOf` pass per step: 128 000 `exp` in
double, ≈ 1 ms on the A-core, and only when the variable is set. It covers
every DSP-resident decode op, because each op sits upstream of those
logits.

One variable does both jobs. When the file is missing, the run is a normal
greedy run that scores its own path and writes it. When the file exists,
the run is forced. So A creates the continuation and B/C read it, with no
second knob. The prefill's last row (target `cont[0]`) is **not** scored.
It is a prefill row, identical across variants, and `NNTR_PPL` already
covers it. That keeps the number decode-only.

**Rejected: scoring inside `TieWordEmbedding` at the decode row**, the way
the prefill branch at `tie_word_embedding.cpp:525` does it. That is less
code in the app, but the layer would need the next target pushed into a
static before every step. It works only for the tied lm_head, and it
scores a scratch recomputation rather than the logits the loop actually
reads. Also rejected as the default: a fixed held-out text as the
continuation. It is still allowed (any id file works), but on the model's
own path the `top1` agreement column reads directly as "how many greedy
picks B would flip". That is what the text approval needs.

Contract §2 and doc 45 §3 are not engaged: no DSP code, arena, walls or
`QS4CX_WH` path change. The `_det`, bit-identity rules and the IDL are
untouched. The host gates keep doc 45 §3.4's spirit where it can hold:
gate 1 is bit-identity of forced vs free, gate 3 is exact index
arithmetic.

### 3.2 Reference

The reference is **A = the switch-off run of the same binary, same model
(`q40-qs4cx-wh`), same sitting, forced on A's own G=512 greedy
continuation** (513 ids). The CPU `q40` run is **not** the reference for
decode PPL. It has different weights: #95 (LEDGER ⑱) measured the NPU
model's prompt PPL at +4.9 % over the CPU's, and its text leaves the CPU's
at word 43. A +2 % bound against the CPU would fail every NPU variant on
the weights alone. The contract's §1 wording ("`NNTR_PPL` … vs the CPU
`q40` PPL") stays for the prompt-PPL column; §6 asks the supervisor to
amend it for decode PPL.

### 3.3 Threshold: keep +2 %, read at G = 512

Δln PPL equals Δ(nll/token), so +2 % PPL means **+0.0198 nats/token**
whatever the base PPL (a greedy path's PPL is near 1–3, the prompt's near
60–115). Two classes have been measured:

* Arithmetic-neutral changes: the fused conv block moved prompt PPL +0.2 %
  (doc 51, 57.00 → 57.12). Doc 53 used ±0.5 % as its tolerance.
* Real accuracy losses: the dense FFN through the u8 MoE kernel +4.2 %,
  attn proj ≈ +4.5 % (doc 51 §2.21), the WH weights vs CPU +4.9 %
  (LEDGER ⑱).

**Does last-bit amplification argue for a different number? No, for two
reasons.**

1. A resident kind's last-bit difference does not add quantization noise.
   It picks different rounding levels at the same quantizers (Q4_0
   activation, MoE u8). B carries the same noise power as A, in another
   realization. On a neutral text its expected Δ is 0.
2. Teacher forcing removes the runaway that makes text identity fail.
   After the pos-512 flip, B keeps being fed A's tokens, so a flip costs
   one token's NLL instead of the rest of the text.

What remains is **one-sided**. On A's own greedy path any perturbed model
reads worse in expectation: A's picks are A's modes. That is a
second-order self-preference term. It is why the bound should not be
tightened to doc 53's 0.5 %. And +2 % still sits halfway between the
neutral class and the loss class.

**Stability, and why G = 512.** With paired forcing, the noise of Δ is the
per-token spread of `nll_B − nll_A` over √N, not the spread of the NLL
itself. It is unknown on silicon. The host fixture's hd64 per-step
difference is ≈ 5·10⁻⁴ nats at 39 dB, but silicon runs at 18 dB. So the
sitting measures it:

```
paste <(grep -o 'decode step=.*' A.log) <(grep -o 'decode step=.*' B.log) |
  awk '{split($4,a,"=");split($9,b,"=");d=b[2]-a[2];s+=d;q+=d*d;n++}
       END{m=s/n;sd=sqrt((q-n*m*m)/(n-1));printf "dnll=%.5f se=%.5f n=%d\n",m,sd/sqrt(n),n}'
```

The verdict needs `2·SE < 0.01`. At N = 512 that holds for a per-token
spread up to 0.11 nats. At N = 64 it would need 0.04, so G = 64 is
reported but never decides. If `2·SE ≥ 0.01` at 512, the remedy is a
second prompt (a second, independent continuation), not a longer G. The
attention stretch's error grows with position, so a longer G mixes a
second effect in.

G = 64 still earns its runs. It is the window where #136's flip sat, and
its step lines must equal the first 64 of G = 512's. That is a
cross-process determinism check on B. The #136 worker-pool race would
show up exactly there.

## 4. Steps

1. **Expose `nllOf`** (`tie_word_embedding.{h,cpp}`).
   Gate rung 1:
   * `ninja -C build`
   * the `*Lfm2Moe*` 6/6
   * `clang-format-14` on the changed lines

2. **The mode in `causal_lm.cpp`** as in §2. Keep all state in locals of
   `run()`: one `std::vector<unsigned>`, one path string, one double, two
   counters. Every new branch hangs off one hoisted `bool`, so the unset
   path costs one predictable branch per token. Add a
   `ponytail: ids file only (no text), batch 1, no SKIP_PREFILL; upgrade = tokenize a .txt through the loaded tokenizer`
   comment.
   Gate rung 1:
   * the host build
   * `*Lfm2Moe*` 6/6
   * `run_host_checks.sh` `ALL CHECKS PASS`

3. **`htp_e2e_test --run` and block (g)** of `run_inproc_e2e.sh` (gates
   1–4 of §1):
   * g1: hd8, htp engine, `--run --dump`, with `NNTR_PPL_DECODE=$OUT/g.ids`
     twice. The first run writes the file, the second reads it. Diff the
     step lines, then `htp_dump_eval.py` the two dump dirs.
   * g2: compare the first run's `E2E gen` with `htp.log`'s.
   * g3: cpu engine on the QS4CX model (it is the index logic under test,
     not the engine):
     * `--run --prompt 16 --steps 8` with the id file `23 30 7 14 21 28 5 12`
     * `NNTR_PPL=1 --run --prompt 17 --steps 1`
     * `NNTR_PPL=1 --run --prompt 24 --steps 1`

     Then S = 23·(nll/token₂₄) − 16·(nll/token₁₇), compared with the
     forced `nll_sum`. The 6-digit prefill print adds ≤ 3·10⁻⁴ of error,
     inside the 10⁻² bound.
   * g4: hd64 `--max-seq 32`: switch off self, then `NNTR_HTP_FORWARD=1`
     forced on that file.

   First check that run() works in the in-process build at all. The
   device app uses it, but the host E2E never has; its decode call passes
   `input_len` where the adapter passes 1.
   Gate rung 1: `run_inproc_e2e.sh` prints the four new lines and `INPROC
   E2E PASS`.

4. **App build for the handoff** (rung 3, `build_android.sh --htp`,
   `strings … NNTR_HTP_FORWARD_KINDS` ≥ 1, and `strings libcausallm_core.so
   | grep -c NNTR_PPL_DECODE` = 1).
   * Build the skel once with `test/htp/build.sh` (rung 2) from the same
     tree; the pass line is `UNDEFINED SYMBOLS OK`. Record its md5.
   * If #136's pool fix (`8d3aa1d9`) has not merged into `htp_moe` by
     then, build the **device** set from this branch merged onto
     `b4a999bf`. Do not put that merge in the PR. Record both SHAs, and
     the rule 22 provenance line.

5. **Device measurement, unavoidable:** the silicon magnitude of the
   amplification (18 dB) cannot be reproduced on the host (39 dB on the
   fixture).

   Handoff `docs/measurements/134-decode-ppl.md`, one sitting, ≈ 20 min.
   Setup:
   * one app set and one skel, variants switched by environment (rule 21)
   * model `q40-qs4cx-wh`, prompt 512 (`prompt512.txt`)
   * `NNTR_NUM_THREADS=8`, `do_sample: false`, `bad_word_ids: [124900]`
   * a clean run dir with `rm -f cont.ids` first

   | variant | env |
   |---|---|
   | **A** switch off (reference) | none |
   | **B** six kinds | `NNTR_HTP_FORWARD=1` |
   | **C** MOE only (null check) | `NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=MOE` |

   Order, A first:
   1. **Normal cells** (tok/s + text), G = 64 then 512, ×1 each, A B C.
      Tok/s is read only against A in-sitting and is not a verdict here
      (#130 measured it). The G=64 texts fill the approval table.
   2. **PPL cells** (never read for tok/s), `num_to_generate: 512`:
      * A with `NNTR_PPL_DECODE=cont.ids` (self; writes 513 ids)
      * A again (forced; null check 1)
      * B forced
      * C forced (null check 2)
   3. The same PPL cells at `num_to_generate: 64`: A, B, C forced (prefix
      check).
   4. Void conditions: any variant log missing its banner
      (`[HTP] graph: init … resident=<mask>` for B/C), or missing its
      `[PPL] decode tokens=` line.

   Results table:
   * variant | G | prefill | decode (all / last 64) | `ppl` | `nll/token` | `top1` | Δ vs A | `2·SE` | null checks
   * then the text-approval table: A / B / C G=64 text, `text approved: y/n`
     per variant, and the decode-PPL column next to it.

   Verdict: §1's device gate. The supervisor folds a B row only if it is
   approved and inside +2 %.

## 5. Risks

* **Host cannot size the effect.** The fixture reads 39 dB where silicon
  reads 18 dB, and the greedy path there is one repeated token. The host
  gates prove the mechanism (indexing, forcing, determinism), never the
  magnitude. The handoff's `2·SE` column says whether the sitting's
  number is meaningful.
* **Device nondeterminism** (the #136 pool race or anything like it)
  would make Δ noise masquerade as accuracy. The three null checks (A≡A,
  C≡A, G64 ⊂ G512, all to 17 digits) catch it. Any failure voids the
  sitting's PPL, not just the row.
* **DVFS / thermal drift** does not move the arithmetic. The PPL cells are
  immune to it; only the tok/s cells carry it, and they are not a verdict
  here.
* **Stale skel / stale app:** the banners (rule 36), the `strings` counts
  and the md5 table. A stale continuation file or a wrong prompt shows up
  as the `prefill top1 != continuation[0]` warning and as `top1` near 0.
* **Address-space budget:** no DSP allocation changes. On the ARM side,
  the mode adds 513 ids and nothing else.
* **Self-preference bias** is larger than expected (B > +2 % while its
  text is approved). The table shows it as `top1` flips concentrated at
  low-margin steps. Then the fix is a symmetric reading (B-self's
  continuation scored by A), filed as a follow-up. It is not a threshold
  change made in the sitting.

## 6. Docs to update

* `docs/htp_moe/BENCHMARK.md` (supervisor, from the filled handoff):
  * the Method paragraph's accuracy sentence and the Goals `accuracy` row
    gain "decode PPL (`NNTR_PPL_DECODE`, A's G=512 continuation, +2 % vs
    A)"
  * the #134 rows A / B / C with a decode-PPL column
* `docs/htp_moe/LEDGER.md`:
  * ⑨: the accuracy gate for the per-token entry is decode PPL + approval
  * ㉕: #136's reading, no defect in the kinds, amplification
  * a silicon-rule candidate: "text identity cannot hold for any resident
    kind on the quantized pipeline; a last-bit change re-draws the
    quantizer levels"
  * the measured `dnll` / `2·SE` of B as the first calibration of the +2 %
* `docs/plans/0001-htp-moe-decode-agent-system.md` §1 accuracy gate: the
  decode PPL's reference is A, not the CPU `q40`. This needs a
  supervisor/user amendment, and the plan does not make it.
* `.claude/skills/hexagon-handoff/SKILL.md`: the per-token-entry bullet's
  PPL column becomes the §4 step 5 PPL cells. `.claude/skills/hexagon-gates/SKILL.md`
  rung 1: add the four new `E2E ppl-decode` / `run==adapter` pass lines.
* `test/htp/host/run_inproc_e2e.sh` header (in the PR).
* #132's handoff reuses §4 step 5's PPL cells, per the issue's ordering
  ("lands before #132's sitting").
