# Measurement 136: short confirmation of the pool fix, then a kind bisect of B's text

Branch `htp/136-forward-text` @ `b4a999bf` (code) — run 2026-09-28 16:37–16:40
KST by the orchestrator on the user's request, unit `R3CY10WM83Y` (SM-S938N),
**shortened sitting**: the phone was available for ≈ 30 minutes, so every cell
is G = 64 × 1 and the question was text only. No tok/s verdict is drawn here.

## Why

Plan 136 host-reproduced a worker-pool race at job boundaries (`POOL RACE`
fails before the fix, 25/30 load runs before, pass after) and expected it to
be the cause of #130's degenerate six-kind text. This sitting asks one thing:
does the six-kind text return to the switch-off text with the fix?

## Artifacts (built on the workstation from `b4a999bf`, SDK 6.4.0.1, HexKL 6.4.0.1, NDK r30, v79)

Staged at `/local/mnt/workspace/htp_moe/136/`, pushed to a clean run dir
`/data/local/tmp/nntrainer/causallm/s136/` (the old run dir holds a foreign
`libc++_shared.so` `50ca2c7b…`; its `libcdsprpc.so` equals `/vendor/lib64`'s,
not the stub). Device `md5sum` equal to the staged files for every row.

| file | md5 |
|---|---|
| `libnntr_hvx_skel.so` | `ae53dfca29ad85f859f5f6dd18ff0ffa` (`UNDEFINED SYMBOLS OK (46 runtime imports)`) |
| `nntrainer_causallm` | `0c93401780a4ed7223b72eb05517f0c6` |
| `libcausallm_core.so` | `05cdca2b567b148dc631cfdc9c3e06d5` (`NNTR_HTP_FORWARD_KINDS` strings: 2) |
| `libnntrainer.so` | `16757c4303ad6851788428b7abab3709` (NEEDED `libsdkl.so`, `libcdsprpc.so`) |
| `libccapi-nntrainer.so` | `86c63c385ad812e2ad6d1c5fb0608445` |
| `libc++_shared.so` / `libsdkl.so` / `prompt512.txt` | `b1586b9b…` / `0ad4e22a…` / `fc65c158…` |
| model `q40-qs4cx-wh`, `tokenizer.json` | on the phone since #100 (`tokenizer.json` `7b8067a5…`) |

Config: `do_sample: false`, `bad_word_ids: [124900]`, `init_seq_len: 512`,
`moe_engine: htp`, `moe_htp_layers: ""`, `num_to_generate: 64`.

## Results (G = 64, one run each, text vs the switch-off run)

| cell | env | prefill tok/s | decode tok/s | calls/token | text = switch-off? |
|---|---|---|---|---|---|
| switch off | none | 568.9 | 36.61 | (n/a) | (reference) |
| all six kinds | `NNTR_HTP_FORWARD=1` | 563.9 | 26.46 | 95.00 | **no** (`town` → `final`, the same loop as #130) |
| MOE,RMSNORM | `…KINDS=MOE,RMSNORM` | 576.6 | 31.92 | 71.00 | yes |
| MOE,CONV1D_GATE | `…KINDS=MOE,CONV1D_GATE` | 560.2 | 33.54 | 40.00 | yes |
| MOE,ROPE,ATTN_M1 | `…KINDS=MOE,ROPE,ATTN_M1` | 532.8 | 31.20 | 28.00 | yes |
| MOE,QK_NORM,ROPE,ATTN_M1 | `…` | 526.7 | 31.27 | 28.00 | yes |
| MOE,RMSNORM,CONV1D_GATE | `…` | 532.2 | 29.99 | 89.00 | yes |
| MOE,CONV1D_GATE,QK_NORM,ROPE,ATTN_M1 | `…` | 503.9 | 29.21 | 46.00 | yes |
| **MOE,RMSNORM,QK_NORM,ROPE,ATTN_M1** | `…` | 524.6 | 27.83 | 77.00 | **no** |
| **MOE,RMSNORM,ROPE,ATTN_M1** | `…` | 437.6 | 26.82 | 77.00 | **no** |

Every switch-on log printed the `graph: init … resident=<mask>` banner with its
mask and the exit `calls/token=` line (rule 36 satisfied: the committed app
carries the switch).

**Reading.**
1. **The pool fix does not restore the text.** The six-kind text is the #130
   loop word for word. The race is real on the host (red/green recorded in the
   branch's commits) but it is not what breaks the text on silicon.
2. **Every kind group alone is text-identical, and so is every pair except one:
   RMSNORM together with the attention stretch (ROPE + ATTN_M1) fails,
   with or without QK_NORM.** CONV1D_GATE is not involved. The fault is an
   interaction between the RMSNORM hook and the attention hook, not a kernel's
   arithmetic (each kernel alone is fine on silicon) and not the call count
   (71 and 89 calls pass, 77 fails).
3. The host E2E (all six kinds on the real-shape `lfm2_moe_tiny_lfm25`
   fixture) passes, so the interaction depends on something only the device
   path has; leading candidates for the plan: the per-call rpcmem staging
   (`stage(act_pool_, …)` / `stage(out_pool_, …)` by size class in
   `invokeForward`; the in-process host build has no FastRPC staging), the
   DSP-side slot routing between the RMSNORM stretch (slot 0 → 1) and the
   attention stretch (slot 2 → 1) of the same layer, and the attention layer's
   own RMSNORM feeding the qkv FC on the CPU. A `NNTR_HTP_DUMP_ALL` run of
   `KINDS=MOE,RMSNORM,ROPE,ATTN_M1` at G = 4 against the switch-off dump names
   the first differing stretch.

Thermal (battery °C·10, thermal_zone0 m°C): start 276 / 28100, after the
six-kind run 287 / 59400, after four bisect cells 322 / 56000, end 342 / 55600.

## Text approval

| cell | generated text (G = 64) | text approved (user: y/n) |
|---|---|---|
| switch off | …town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum that opens only on summer weekends, and a lifeboat station | (reference) |
| all six kinds | …final answer should be the same as the original, but you must not stop until you are told to. The original description is the same as the original, but you must not stop until you are told to. The final answer should be the same as the original, but you must not stop until you are told to. | **n** (user, 2026-09-28: fail) |

Logs: `/local/mnt/workspace/htp_moe/136/logs/` (`off_G64_r1`, `fwd_G64_r1`,
`k_<mask>`, `md5.log`, `therm.log`).

## Dump sitting (same day 16:50–17:00, same unit and artifacts, G = 4)

`NNTR_HTP_DUMP=<dir>` (+ `NNTR_HTP_DUMP_ALL=1` for switch-on runs) on six
runs: switch off (twice), `KINDS=MOE`, `MOE,RMSNORM`, `MOE,ROPE,ATTN_M1`, and
the failing `MOE,RMSNORM,ROPE,ATTN_M1`. Dumps under
`/local/mnt/workspace/htp_moe/136/dump_<run>/`. SNR of each MoE call's input
against the first switch-off run (prefill = calls 0–22, decode token 1 =
calls 23–44, token 2 = 45–):

| run | prefill | decode token 1 (first → last MoE layer) | decode token 2 | text (G = 4) |
|---|---|---|---|---|
| switch off, second run | 23/23 bit-identical | all bit-identical | bit-identical | same |
| `KINDS=MOE` | 23/23 bit-identical | all bit-identical | bit-identical | same |
| `MOE,RMSNORM` | bit-identical | 34.2 → … → 18.2 dB | 34.6 dB (recovers) | same |
| `MOE,ROPE,ATTN_M1` | bit-identical | 40.2 → … → 18.3 dB | 43.6 dB (recovers) | same |
| **`MOE,RMSNORM,ROPE,ATTN_M1`** | bit-identical | 33.7 → … → 18.3 dB | **−1.6 dB** (another token) | **differs** (`town` → `final`) |

Per stretch, the failing run at pos 512 matches the `MOE,RMSNORM` run's
RMSNORM stretches and the `MOE,ROPE,ATTN_M1` run's attention stretches at
28–46 dB (layer 0's norm bit-identical); at pos 513 the very first RMSNORM
input (layer 0, the new token's embedding) is at 0.2 dB: the token chosen at
pos 512 differs. The KV seeds are byte-identical across runs.

**Reading.** The CPU switch-off path and the MoE-only entry are bit-exact run
to run, so every dB below ∞ is what a resident kind puts in. **On silicon a
resident RMSNORM alone moves the first decode token's first MoE input to
34 dB (≈ 2 % relative) and the last to 18 dB; the attention stretch alone does
the same (40 → 18 dB).** Each alone leaves the greedy token unchanged at this
prompt; together the two deviations flip the token decided at pos 512, and
the text runs away from there. So #136 is **not a logic / binding / race
fault**: it is two per-kind numeric deviations far larger than the host
measured (RMSNORM 129 dB, attention 39 dB on the fixture), stacked. Leads
for the re-plan: the silicon `HvxM1Ops` gtest failure `rmsnorm kind=2
bad_y=2048` (every element of one norm kind differs, #137) — larger than the
"subnormal ulp" reading of rule 37 — and whatever the Android CPU path does
differently from the host's in the norm (fp16 activations, NEON rsqrt).
