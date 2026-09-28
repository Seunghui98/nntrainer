# Measurement 134 + 132: decode PPL and ADD + ROUTER_TOPK resident, one sitting

Code: `htp_moe` @ `bb845426` (PR #144 = #134 and PR #145 = #132 PR 1 merged).
The staged set was built from the local merge `dev/sitting-134-132` @ `5cf77ebe`,
whose code under `nntrainer/`, `Applications/`, `test/htp/*.c` and the IDL is
byte-equal to `bb845426`. Estimated device time: **≈ 25 min**. Run by the
orchestrator on the workstation (user's request, 2026-09-28), unit recorded
under Results.

## Why

1. #134: the first device reading of the decode-side PPL (`NNTR_PPL_DECODE`),
   the accuracy gate of the per-token entry since #136 showed text identity
   cannot hold for any resident kind (LEDGER rule 39).
2. #132 PR 1: ADD and ROUTER_TOPK resident, 95 → 51 calls/token; expected
   D − B ≈ −3.5 ms/token (≈ +10 % decode), D still below A (51 > 22 calls and
   the ATTN_M1 term, #146).

Accuracy (contract §1, 2026-09-28): decode PPL of every variant forced on A's
own G = 512 continuation, **reference = A of this sitting**, fail above +2 %;
**text compared** against A at every cell, and the user's text approval.

### Variants (4, contract §4.2; one binary set, environment only)

| | env (beyond `NNTR_NUM_THREADS=8`) | expected banner / calls |
|---|---|---|
| **A** (reference, first) | none | no `graph:` line |
| **C** | `NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=MOE` | `resident=MOE`, 22.00 |
| **B** | `NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=MOE,RMSNORM,QK_NORM,ROPE,CONV1D_GATE,ATTN_M1` | six kinds, 95.00 |
| **D** | B's mask `,ADD,ROUTER_TOPK` | `resident=RMSNORM\|CONV1D_GATE\|QK_NORM\|ROPE\|ATTN_M1\|ADD\|ROUTER_TOPK\|MOE`, 51.00 |

Not run: an A0 on the pre-#143 tree (the pool fix's own prefill gate). PR
#145 changed the IDL, so A0 needs a separate `c5dcb382` set and would be a
fifth variant; **the user accepted #143 without it (2026-09-28)**.

## Artifacts (`/local/mnt/workspace/htp_moe/134-132/set/`, `md5.txt` next to it)

| file | md5 |
|---|---|
| `libnntr_hvx_skel.so` | `0c2d5b00f1e947fbfdb1eeaebd2d40a1` (`UNDEFINED SYMBOLS OK (46 runtime imports)`) |
| `nntrainer_causallm` | `b6adb4f55a849107c635770d5e2d5fcb` |
| `libcausallm_core.so` | `59e9be42d533bd6762e8ee93bff4846e` (`NNTR_HTP_FORWARD_KINDS` 2, `NNTR_PPL_DECODE` exact 1) |
| `libnntrainer.so` | `bd5abd809202adb742a35563230bc5bc` |
| `libccapi-nntrainer.so` | `06a3072b13937becc7fce9561ef172e3` |
| `unittest_hvx_softmax` | `25bec7dfd9f5b342783ec59cc7d5059d` (PR #145's, carries `HvxM1Ops.RouterTopkMatchesDetBitExact`) |
| `libc++_shared.so` / `libsdkl.so` / `prompt512.txt` | `b1586b9b…` / `0ad4e22a…` / `fc65c158…` |
| model `q40-qs4cx-wh`, `tokenizer.json` | on the phone since #100 (`tokenizer.json` `7b8067a5…`) |

## Steps

Clean run dir `/data/local/tmp/nntrainer/causallm/s134` (the old dir holds a
foreign `libc++_shared.so`); every `adb` names the serial (a second device may
be on USB). Config as #136: `do_sample: false`, `bad_word_ids: [124900]`,
`init_seq_len: 512`, `moe_engine: htp`, `moe_htp_layers: ""`.

1. Push the set, `md5sum` on the device = the table.
2. `unittest_hvx_softmax --gtest_filter='HvxM1Ops.*'` (router rows `bad=0`;
   the kind=2 rmsnorm row is #137's known subnormal case).
3. tok/s: for G in 64, 512: `A C B D` run 1, then `D B C A` run 2.
4. PPL at G = 512: `rm -f cont.ids`; A with `NNTR_PPL_DECODE=cont.ids` (writes
   A's continuation, `source=self`); A again (null check, must equal); C
   (must equal A); B; D.
5. Text: every C / B / D cell against A of the same G and run; paste A's, B's
   and D's G = 64 run-1 texts below for approval.

## Results (fill in)

## Text approval

| variant | decode PPL (G=512, forced on A) | generated text (G=64, run 1) | text approved (user: y/n) |
|---|---|---|---|
| A | | | (reference) |
| C | | | |
| B | | | |
| D | | | |

## Notes from the run
