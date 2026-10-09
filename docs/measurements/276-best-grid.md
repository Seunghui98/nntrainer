# Measurement 276: the 2-bit file on the best build, prompt 1024 / 2048 / 4096 × G 64 / 512 / 1024

Issue #276. Build = `htp_decode` @ `082c1b24d` (revision 3: #4415 @ `2db6cad59` prefill,
the FastRPC 64 KiB stack fix `aea5762c3`, #278) + this branch's
QS2CX_WH `expertDesc` (`9784d6508`) and untied-head hook (`2c98d5dd5`). E = our one-PD decode
(`NNTR_HTP_E2E=1`), C = 16, `lmhead_engine cpu`, **KV `q8` (the user's config of
record)**. The file is variant C (`gemma4_26b_ternary_fcqs4cx`, md5 `426437624d…`; see
`276-2bit-s1.md`). S25 `R3CY205ZMND`, 2026-10-09 16:47–17:07 KST.

## Table of record (E, q8 KV, C 16)

| prompt | G | prefill tok/s | decode tok/s (all / last 64) | text ok? |
|---|---|---|---|---|
| 1024 | 64 | **354.9** | 5.13 / 5.13 | yes |
| 1024 | 512 | **352.6** | 5.84 / 7.44 (ends `<turn\|>` at 107) | yes |
| 1024 | 1024 | **355.1** | 5.75 / 7.06 (ends at 107) | yes |
| 2048 | 64 / 512 / 1024 | — | — | **does not load**: `no room for the FC set beside the resident experts (mapped=704 MiB, fastrpc_mmap(32 MiB) failed: err=1 …)`. The same at C = 8 (`mapped=384`) |
| 4096 | 64 | — | — | **does not load**, same line. G 512 / 1024 skipped after it |

Reference, A (#4415's hybrid decode, same build, same file), p1024 G512 q8: prefill
**344.7**, decode **6.04** (last 64: 6.83). Ends at 105. Text ok.

**fp16 KV (labelled reference, not the record; the r2 E config without `attention_kv_dtype`):**

| prompt | G | prefill tok/s | decode tok/s (all / last 64) | text ok? |
|---|---|---|---|---|
| 1024 | 64 | 245.3 | 5.42 / 5.42 | yes |
| 1024 | 512 | 250.9 | 6.83 / 8.12 (ends at 121) | yes |
| 1024 | 1024 | 248.5 | 6.79 / 8.10 (ends at 121) | yes |
| 2048 | 64 | 247.8 | 3.48 / 3.48 | yes (on topic) |
| 2048 | 512 | 245.9 | 4.84 / 4.91 (ends at 368) | yes to `<turn\|>`, loop onset at 260 |
| 2048 | 1024 | 246.2 | 4.84 / 4.88 (ends at 368) | same text as G512 |
| 4096 | 64 / 512 / 1024 | — | — | **the phone rebooted**, all three times. Prefill finished (the first token `<\|channel>` printed); the reboot came at the first decode token's `graph: init`. adb lost the device; uptime reset |
| A p1024 G512 | | 234.6 | 4.50 / 5.16 (ends at 123) | yes |

## Reading

- **q8 is the prefill lever.** At p1024 prefill rises from 245–251 to 353–355 tok/s
  (4.1 → 2.9 s). The int8 row-blocked attention runs on every prefill layer: logcat
  `mha_core: attention over the int8 quantized KV cache on the accelerator (row-blocked,
  fixed scales)`.
- **Decode under q8.** The one-PD decode runs (`calls/token=1.00`, `timeouts=0`,
  `id_mismatch=0`) and its text stays coherent. Its attention cache is the fp16 `attn_m1`
  one (`registered layers=25 … cache=409600 KiB` + 40960 KiB).
  - That E seeds from the fp16 master is **not shown by any log line**. The evidence is
    the coherent text. The E texts at q8 and fp16 diverge within the first sentence:
    the prefill attention differs, so the first decode row does too.
  - Decode is 5.1–5.8 tok/s under q8 against 5.4–6.8 under fp16. The q8 cells read more
    flash per miss (0.80–0.85 MiB against 0.48–0.69). Cell-to-cell, the page cache moves
    these numbers as much as the KV format does (`276-2bit-s1.md`, Reading).
- **E vs A at p1024 G512.** Prefill: E 352.6 / A 344.7 = 1.02 under q8; 250.9 / 234.6 = 1.07
  under fp16. Gate ≥ 0.95: pass. Decode: E is **slower** than A under q8 (5.84 against
  6.04 all, 7.44 against 6.83 last-64) and faster under fp16 (6.83 against 4.50).
- **Prompts ≥ 2048 do not fit the one-PD address space.**
  - With fp16 KV, p2048 loads: `heap_used_kib` 1 992 693 against 1 458 549 at p1024, from
    `init_seq_len` 4096 and `max_seq_len` 4096. p4096 (init / max 8192) takes the phone
    down at the first decode token. The cause is not read: the device rebooted before
    logcat was saved.
  - With q8 KV, p2048 no longer loads at all: the FC set's 32 MiB map is refused on the
    DSP side, at C = 16 and at C = 8 alike. The arena is not what fills the space.
  - **A p4096 cell must not run again on this build until the reboot is understood.**
- **2-bit file, p1024 text.** E, A, q8 and fp16 all give a coherent, on-topic Ardley
  summary that ends on `<turn|>`.

## Per-cell details

`pgpgin/miss` = the driver's `pgpgin_mib` / misses. Temperatures are cpu-nsp max / battery /
zone0 °C at the cell's start (`COOL_QUICK`). Text: the first 300 characters, judged to the
first `<turn|>`.

| cell | prefill | prefill tok/s | decode tok/s | last 64 | tokens | misses/token | miss wait ms | pgpgin/miss MiB | arena MiB | peak RSS KiB | start °C | `<turn\|>` at | loop onset |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E p1024 G64 q8 | 2 885 ms | 354.9 | 5.13 | 5.13 | 64 | 114.05 | 73.2 | 0.81 | 704 | 3 250 840 | 43.0 / 30.3 / 40.7 | — | none |
| E p1024 G512 q8 | 2 904 ms | 352.6 | 5.84 | 7.44 | 107 | 110.55 | 66.5 | 0.85 | 704 | 3 250 996 | 48.8 / 30.1 / 44.9 | 107 | none |
| E p1024 G1024 q8 | 2 884 ms | 355.1 | 5.75 | 7.06 | 107 | 110.55 | 69.2 | 0.80 | 704 | 3 243 668 | 59.7 / 30.2 / 51.9 | 107 | none |
| A p1024 G512 q8 | 2 971 ms | 344.7 | 6.04 | 6.83 | 105 | — | — | — | 704 | 2 952 656 | 56.6 / 30.5 / 49.2 | 105 | none |
| E p1024 G64 fp16 | 4 175 ms | 245.3 | 5.42 | 5.42 | 64 | 111.47 | 63.3 | 0.69 | 704 | 3 242 976 | 35.2 / 27.3 / 31.8 | — | none |
| E p1024 G512 fp16 | 4 081 ms | 250.9 | 6.83 | 8.12 | 121 | 110.37 | 46.5 | 0.48 | 704 | 3 252 088 | 51.9 / 27.5 / 43.8 | 121 | none |
| E p1024 G1024 fp16 | 4 121 ms | 248.5 | 6.79 | 8.10 | 121 | 110.37 | 47.4 | 0.48 | 704 | 3 251 240 | 55.8 / 28.3 / 47.7 | 121 | none |
| E p2048 G64 fp16 | 8 264 ms | 247.8 | 3.48 | 3.48 | 64 | 122.50 | 115.1 | 1.51 | 704 | 3 263 348 | 57.4 / 28.7 / 48.8 | — | none |
| E p2048 G512 fp16 | 8 329 ms | 245.9 | 4.84 | 4.91 | 368 | 127.67 | 107.7 | 1.39 | 704 | 3 283 316 | 55.0 / 29.1 / 48.0 | 368 | 260 |
| E p2048 G1024 fp16 | 8 318 ms | 246.2 | 4.84 | 4.88 | 368 | 127.67 | 107.8 | 1.37 | 704 | 3 280 092 | 57.4 / 29.7 / 49.2 | 368 | 260 |
| A p1024 G512 fp16 | 4 365 ms | 234.6 | 4.50 | 5.16 | 123 | — | — | — | 704 | 2 932 512 | 84.1 / 30.6 / 63.6 | 123 | none |

Texts (first 300 characters):

- E p1024 q8 (all three G): `<|channel>thought <channel|>The small harbor town of Ardley, located at a river mouth and characterized by its fishing and farming communities, has a history dating back to the twelfth century. The town's infrastructure and economy, including its stone buildings and fishing industries, are heavily s`
- A p1024 G512 q8: `<|channel>thought <channel|>The small harbor town of Ardley, located at a river mouth and much influenced by the sea and weather, has a long history of fishing, farming, and maritime services. The town's infrastructure and economy, including a single main street, a stone church, a small museum, and `
- E p1024 fp16 (all three G): `<|channel>thought <channel|>The small harbor town of Ardley, located at a river mouth near the northern sea, is a community shaped by the shifting tides and weather, with local life centered on fishing, farming, and a small museum. The town has a long history dating back to the twelfth century, with`
- E p2048 fp16 (all three G): `<|channel>thought ing { "summarize": "The text describes a three-year study on a city's water utility, which used a dense network of low-cost acoustic and pressure sensors to locate and estimate the volume of hidden leaks. By combining night-flow analysis and acoustic correlation, the project moved `
- A p1024 G512 fp16: `<|channel>thought <channel|>The small harbor town of Ardley, located near a river and sea, is a community defined by its history of fishing and farming, with much of the infrastructure and social life shaped by the changing seasons. The town has its origins in the twelfth century and developed into `

## Artifacts

| item | value |
|---|---|
| branch | `htp/276-s1-2bit` rebased on `082c1b24d`: `9784d6508`, `5ed0fee22`, `2c98d5dd5`, `988881e09`, and this doc's commit. Old → new hashes: `404862b0c` → `9784d6508`, `b67fd1200` → `5ed0fee22`, `28f5d96ef` → `2c98d5dd5`, `8e1adf119` → `988881e09`. One conflict: the `lfm2_moe_layer.cpp` layer call takes revision 3's new arguments (`pre_gamma`, the fused add) and our `w_bits` |
| skel v79 | `28d1c411f803972ae2512327076596dc`, `UNDEFINED SYMBOLS OK (68 runtime imports)`, `ARCH OK (V79)` |
| `nntrainer_causallm` / `libcausallm_core.so` | `273fbd5e82e52f72f1edef1c4914c990` / `df9fe4ddb7972235a0349c5f2c7f33ea` (FORWARD_KINDS strings 2) |
| `libnntrainer.so` / `libccapi-nntrainer.so` | `37c8e1c9b75e21dbe8c3bdae6f500a32` (NEEDED `libsdkl.so`, `libcdsprpc.so`) / `c4d7c66cd1bcccb590cdbd8b047337fe` |
| install | `/data/local/tmp/nntrainer/causallm/s276b/`. Device md5 == local for all 8 files. `libc++_shared`, `libsdkl` and `unittest_hvx_two_sessions` copied from `s266p/` |
| configs | q8: `/data/local/tmp/nntrainer/s276q8/r2_{E,A}_p{1024,2048,4096}_g{64,512,1024}`. fp16: `…/s276g/`. Both staged by `KV276=q8 DC276=… P276="1024 2048 4096" G276="64 512 1024" 276-run.sh stage <user nntr_config.json>`: `init_seq_len = 2 × prompt`, `max_seq_len = max(2048, init)`, `bad_word_ids []` |
| runner | `DC276=… B276=… [TAG276=q8 SKIP4096=1] 276-run.sh grid <log dir>`. Logs: `scratchpad/r276/G/<cell>[_q8].log` |
| PC gates | not run (user rule for this task) |
