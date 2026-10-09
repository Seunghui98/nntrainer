# 260 — standard prefill prompts (512 / 1024 / 2048 / 4096 tokens)

Contract §1 Speed row (2026-10-09): prefill prompts 1024 / 2048 / 4096 from the
next sitting (512 retired once #266 / #267 finish). This note adds the 2048 and
4096 files beside the 512 / 1024 ones used by the #260 r2 sitting.

| file | tokens (app count) | md5 | body |
|---|---|---|---|
| `260-prompt512.txt` | 512 | `c31a365fbaecd148602c09c8a2523a1d` | Thyme abstract (research register) — in use |
| `260-prompt1024.txt` | 1024 | `93bd76497aeb9c9d9ea7074d93e2bff8` | Ardley harbour town — in use |
| `260-prompt2048.txt` | 2048 | `193cd0918df99aa2a9e06794b4847e30` | municipal water-network leak study (research report register, 1760 words) — **not yet device-validated** |
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

## What remains before adoption

- **Device validation, G 64, not done** (the phone is held by the #266 / #267
  sittings; no adb from this task): one run per file to confirm the log reads
  `prefill: 2048 tokens` / `prefill: 4096 tokens`, and the user's read of the
  text (a coherent summary up to the first `<turn|>`) plus a prompt nll for the
  record, as for 512 / 1024.
- `260-e-run.sh stage` hard-codes `prompts = {512, 1024}` and
  `init_seq_len = 2048 if p == 1024 else 1024`; the next sitting's runner must
  add the two files and keep the doc 57 §7.5 rule (init_seq_len strictly
  above the prompt length, so 4096 → 8192).
