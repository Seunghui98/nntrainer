# 260 — standard prefill prompts (512 / 1024 / 2048 / 4096 tokens)

Contract §1 Speed row (2026-10-09): prefill prompts 1024 / 2048 / 4096 from the
next sitting (512 retired once #266 / #267 finish). This note adds the 2048 and
4096 files beside the 512 / 1024 ones used by the #260 r2 sitting.

| file | tokens (app count) | md5 | body |
|---|---|---|---|
| `260-prompt512.txt` | 512 | `c31a365fbaecd148602c09c8a2523a1d` | Thyme abstract (research register) — in use |
| `260-prompt1024.txt` | 1024 | `93bd76497aeb9c9d9ea7074d93e2bff8` | Ardley harbour town — in use |
| `260-prompt2048.txt` | 2048 | `193cd0918df99aa2a9e06794b4847e30` | municipal water-network leak study (research report register, 1760 words) — **not yet device-validated** (checklist below) |
| `260-prompt4096.txt` | 4096 | `a157af12c759e5ca00056d0d40dd38e7` | Halstrand, a mountain mining / observatory town (3520 words) — **not yet device-validated** |

All four share the same wrapper byte for byte: `<bos><|turn>user\n` + the
3-sentence summarisation instruction + `<Text>\n` … `\n</Text><turn|>\n<|turn>model\n`.
The body is continuous original prose in several paragraphs, no lists, no
repeated paragraphs (`sort | uniq -d` empty), ASCII only. The 2048 / 4096
texts are on subjects the 512 / 1024 texts do not touch, so a generation is
not a continuation of them.

## How the count is taken

The app feeds `sample_input` verbatim to `tokenizer->Encode(prompt_)`, which is
`Encode(text, /*add_special_tokens=*/false)` (`Applications/CausalLM/models/causal_lm.cpp`
`run()`, `huggingface_tokenizer.cpp`); the model's `nntr_config.json` sets no
`system_prompt`, so nothing is prepended. The literal `<bos>` in the file is
parsed by the tokenizer as id 2. `prefill: N tokens` prints `init_len`, the
length of that encoding (no truncation while N ≤ `max_seq_len − num_to_generate`).

Host-side count, HF `tokenizers` 0.22.2 against
`/local/mnt/workspace/models/gemma4_26b_qs4cx/tokenizer.json`:

```python
tok.encode(open(f).read(), add_special_tokens=False).ids   # == the app's count
```

Calibration on the files in use: 512 → 512, 1024 → 1024, i.e. **offset 0**
between this host count and the app's `prefill: 512 tokens` / `1024 tokens`
lines of the r2 sitting (`260-step2-r2-3897963d8.md`). With
`add_special_tokens=True` the count is one higher (a second BOS) — not what the
app does. The new files count 2048 / 4096 exactly under the same call; the last
sentence of each body was adjusted (reworded, not padded) until the count landed.

## Runner

`260-e-run.sh` (2026-10-09) stages all four prompts and runs `PROMPTS`
(default `1024 2048 4096`; `PROMPTS="512 1024"` for before/after lever
comparisons). Per prompt, doc 57 §7.5 rule (init_seq_len strictly above the
prompt) and `max_seq_len ≥ init_seq_len` so p + G always fits (the base config's
`max_seq_len` 2048 would truncate a 2048 prompt's tail):

| prompt | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| `init_seq_len` | 1024 | 2048 | 4096 | 8192 |
| `max_seq_len` | 2048 | 2048 | 4096 | 8192 |

The 512 / 1024 configs are byte-identical to the r2 sitting's (checked on the
host against the old `stage`). p4096 sizes the KV cache and the RoPE table at
8192 positions and the prefill graph at 8192 rows; it may not fit the one-PD
3840 MiB ceiling, at G1024 least of all. Such a cell keeps
`<cell>.C<c>.fail.log`, the first failure line goes to the sweep output and
to `sum`'s `FAIL` list, and the runner continues.

## First device run (validation before adoption)

The phone is held by the #266 / #267 sittings; nothing below has run yet.

```bash
cd docs/measurements
./260-e-run.sh stage                               # under the lock if a sitting may run
PPL=1 ./260-e-run.sh prefill <bin dir> <log dir>   # E and A, p1024 / 2048 / 4096 x G64, + nll cells
```

Check per prompt, for E and A:

- [ ] `prefill: N tokens` reads exactly 1024 / 2048 / 4096 (`sum` marks a
      mismatch `**!= p<N>**`); no `[CausalLM] WARNING: prompt (…) exceeds`
      line in the log.
- [ ] Prefill tok/s and ms recorded, E / A ratio (prefill gate E ≥ 0.95 × A).
- [ ] G64 text judged up to the first `<turn|>`: a coherent summary of that
      prompt's body (water-network leak study for 2048, Halstrand for 4096),
      not a continuation of the prose.
- [ ] nll/token from the `_ppl` cell recorded beside the p512 / p1024 `_ppl`
      cells of the r2 sitting (`260-step2-r2-3897963d8.md`, fp16 KV).
- [ ] Peak RSS and S1 ceiling recorded; any `FAIL` line (expected candidate:
      p4096) copied verbatim into the doc with the C it ran at.
- [ ] Then mark the 2048 / 4096 rows above as validated.
