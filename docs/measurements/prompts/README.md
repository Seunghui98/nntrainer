# The bit-preserving prompt set (plan 141 step 6)

The reusable accuracy set for every bit-preserving lever of the
2026-09-28 direction change (contract `0001` decisions table): a variant
passes when its generated text is byte-identical to A's on all eight
prompts at G = 64. First used by `docs/measurements/141-dspq-moe.md`.

Token counts: the model's own `tokenizer.json`
(`/local/mnt/workspace/models/lfm2.5-8b-a1b/q40-qs4cx-wh/tokenizer.json`,
md5 `7b8067a580173d3eb1697afae3b456f5`, `tokenizers` 0.22.2, no special
tokens added; the device passes the file as `"$(cat <file>)"`, which drops
the final newline and does not change the count). Every prompt is at most
512 tokens (`init_seq_len: 512`).

| id | file | domain | tokens | md5 |
|---|---|---|---|---|
| p01 | `../77-prompt512.txt` | narrative (the harbour town); a near-tie at decode step 1 (`town` p = 0.134, #134), the most sensitive prompt known | 512 | `fc65c1588dc66dd764c7013fe96cbb75` |
| p02 | `bitset-02-code.txt` | a buggy Python `median`, explain and fix | 207 | `7d2a17d3cc097d0567c9463c5eaeff88` |
| p03 | `bitset-03-math.txt` | multi-step arithmetic word problem, step by step | 117 | `c804b037d3bb87bf09259504fcae1c79` |
| p04 | `bitset-04-korean.txt` | a Korean paragraph (Joseon Hanyang) and three questions | 326 | `c5d4595b859a4e6abc3a035353aac180` |
| p05 | `bitset-05-json.txt` | extract an order note as JSON | 207 | `a21ecdeea9d43d1b1366bde523347a62` |
| p06 | `bitset-06-dialogue.txt` | multi-turn chat, the assistant's next turn | 276 | `6efd387197d6078366152f83a9b72a69` |
| p07 | `bitset-07-facts.txt` | encyclopedic article (the transistor) | 402 | `db0251e1bc57f072f815802cd93c260d` |
| p08 | `bitset-08-short.txt` | "List ten fruits", a short prefill | 24 | `67b657c1261c2c12022907c2117a9a54` |
