# Plan 225: FC weights as QS4CX_WH beside the model file — one copy for HTP prefill and E2E decode

Issue #225 (hexagon, p0). Base `htp_first_version` @ `b7c1d4ff6` (PR #223
merged). Contract `docs/plans/0001-htp-moe-decode-agent-system.md`; the
measured failures are in #222's comments (sitting 2026-10-02, farm
`R3CY205ZMND`); the quantization analysis is doc 51 §2.17 / §2.19 / §2.21;
the budget facts are doc 46 §41 and LEDGER rules 49, 59.

## 1. Goal and gate

The user's decision (b): the FC weights (conv `in_proj` / `out_proj` of the
18 conv layers, q / k / v / o of the 6 attention layers, up / gate / down of
the dense FFN of layers 0–1 — 67 weights, ≈ 216 MiB as WH) are quantized
**once, from the f32 checkpoint, by the packer** into the HMX tile layout, so
that the load-time Q4_0 → qs4cx re-quantization (`htp_q4_0_convert.cpp:42`
`htp_qs4cx_from_q4_0x4`, called from `htp_compute_ops.cpp:4898` /
`:5012–5015`) and its second copy on the DSP heap (`registerRm`
`:4935–4984`) disappear, and HTP prefill (HMX tile kernels) and HTP E2E
decode (the WH GEMV `hvx_gemm_u8i4_wh_col`, `hexkl_mm_u8i4_moe.c:472`,
`:741`) read the same arena bytes. **The lm_head stays Q4_0** (§3.5).

Gate, as the issue states it, made measurable:

| check | where it is read | pass |
|---|---|---|
| G1 fit: the three cases load and run all 9 cells (P64 / P512 / P1024 × G64 / G512 / G1024) on the config of record (`docs/measurements/config/q40-qs4cx-wh.nntr_config.json`, `init_seq_len 1024`) | the handoff's 27 result rows; no `VOID` | every cell has `prefill:` and `generation:` lines |
| G2 E2E: `calls/token = 1.00`, `resident` / ceiling lines | `[HTP] graph: forward calls=… calls/token=1.00`, `token driver: pool …`, `s1_arena_mib`, `heap_used_kib` | `calls/token=1.00` in every E2E cell; `heap_used_kib` below #222's 223 003 and no FC bytes on the heap (`[HTP] fc wh: arena=<n> heap=0`, the new banner of §3.3) |
| G3 hybrid P1024 | `Anew_P1024_G*` cells | no `AEE_ENOMEMORY` (`0x80000402`) in `nntr_hvx_mm_u8i4_conv_block` |
| G4 accuracy, prefill PPL (`NNTR_PPL=1`) and decode PPL (`NNTR_PPL_DECODE`, forced on the CPU `q40` continuation) over the 8 prompts at G=256, text pasted for approval | the handoff's PPL block | **threshold open, needs-user** — the plan carries both candidates (§3.6): T1 contract §1 (pooled decode PPL ≤ 1.02 × A, A = the no-keys hybrid of the same sitting, LEDGER ⑱'s amendment), T2 #201's rule (drift tolerated, a new loop by `tools/htp/loop_check.py` or off-context text fails; text approved by the user). Either way the user approves the texts |
| Standing gates | | bit-identity: the host checks of §4 print their pass lines (WH-from-file registration gives the same handles' bytes as today's `get_or_register_wh` on the fixture, `bit_identical=1`); the E2E one-PD logits of the WH FC ops equal their host spec; text ≡ CPU is not achievable for an engine key (rule 39) and is replaced by G4 as the issue says. Prefill ≥ −5 % of variant A: **not a gate here** (the issue: the old config has no HTP prefill FC); the comparison cell is old config vs new, same sitting |

BENCHMARK.md cells that move: the LFM2.5 table of record (9 cells × 3 cases)
the user asked for on 2026-10-06; the NPU rows of record stay the record
sitting 2026-09-30 until a mirrored cool sitting replaces them (BENCHMARK
Method).

## 2. Where it lives

Verified in this tree (`b7c1d4ff6`):

| what | path:line | change |
|---|---|---|
| packer, FC write | `Applications/CausalLM/quantize_stream.cpp:581–631` `writeFc` — already accepts `QS4CX_WH` for every FC; `:608–613` refuses only a source over `MAX_TENSOR_BUFFER_BYTES` = 64 MiB (`:47`), and the largest FC (dense up / down, 2048 × 7168 f32 = 56 MiB) is under it; `:550–553` refuses `QS4CX_WH` for the **embedding** (= the tied lm_head) | a second output stream for the FC WH images (§3.2); nothing in the quantizer arithmetic (`:776–820` `writeQuantized` WH branch = `quant_qs4cx_f32` + `whPack`) |
| packer, LFM2 walk | `:1120–1200` `writeLfm2Moe` (`fc_dtype` at `:1142–1169`), `:1260–1276` `writeOutputConfig` (writes `fc_layer_dtype`, `moe_layer_dtype`), `:1330–1440` options | `--fc_wh_sidecar` option; config keys `fc_wh_file_name`, `fc_wh_format` |
| converter | `nntrainer/tensor/htp_q4_0_convert.cpp:42` (`htp_qs4cx_from_q4_0x4`), `:99` (`htp_qs4cx_from_packed`) | untouched; the first stops being called on the config of record (stays for a model without the sidecar) |
| HTP ops, prefill registration | `nntrainer/tensor/htp_backend/htp_compute_ops.cpp:4876–4915` `get_or_register_fc` (`fcSliceCols` `:4866`, 2 MiB slices = 2048 columns at K = 2048), `:4935–4984` `registerRm` (arena tail first, else DSP heap), `:4991–5056` `get_or_register_dense` (`denseChunkCols` 1792, gate\|up pairs + down row chunks), `:5094–5121` `get_or_register_conv` (in_proj thirds a, b, c), `:5707–5800` `get_or_register_wh` (arena, no heap, no bake), `:5603–5650` `readWeight` (pread of a WH tensor into the arena), `:6146–6175` `registerFromArena`, `:5882–5900` `placeExisting`, `:5925` `kArenaChunkMax` 256 MiB | new: a sidecar reader that fills the same `FcHandles` / `DenseHandles` / `ConvHandles` from WH images (§3.3); `registerRm` keeps the heap only as the hybrid's overflow with a **bit-exact unpack** instead of a requant |
| HTP ops, E2E FC set | `:1773–1821` `registerQ4m1` (Q4M1 into the FC arena chunks, "no room" error `:1785–1795`), `:1842–1915` `bindQ4m1`, `:4408–4430` `e2ePlaceFc`, `:1672–1690` `NNTR_HTP_PPL_LEVERS` (0x2 = native Q4M1 GEMV, still Q4M1 bytes) | PR 2: FC / DENSE_FFN ops bind arena WH handles; LM_HEAD stays Q4M1 (§3.4) |
| graph descriptor | `nntrainer/tensor/htp_backend/htp_graph_desc.h:93–95` `HTP_GRAPH_KINDS_Q4M1`, `:274` `feed`, `:281–282` `h_gu` / `h_dn`, `:286–287` feed bits | PR 2: `HTP_GRAPH_FEED_WH` bit; FC / DENSE_FFN parts become u8i4 registry handles when set (shared header, host and DSP; no IDL signature change) |
| DSP graph runner | `nntrainer/tensor/htp_backend/hmx/hexkl_graph.h:71` `hexkl_graph_fc_fn`, `:129`; `test/htp/nntr_hvx_graph.c:162` (`env->fc = nntr_hvx_fc_q4m1_graph`); `test/htp/nntr_hvx_fc_q4.c:376–388` (feeds); `test/htp/nntr_hvx_session.h:172–186` | PR 2: `nntr_hvx_fc_wh_graph` (u8 row quant → `hvx_gemm_u8i4_wh_col` → dequant), DENSE_FFN through the MoE M=1 pair path with fixed routing |
| conv block kernel | `nntrainer/tensor/htp_backend/hmx/hexkl_conv_block.c:268–284` (session scratch ∝ `m_pad`: act f32 + AH + g + out ≈ 26 MiB at M = 1024), `:461–468` (state is **written** from g's last two rows, never read as history) | read `state_f32` as the two history rows when the host chunks M (§3.3 c) |
| app layers | `Applications/CausalLM/layers/conv_block_layer.cpp:189–201` (`use_htp = rows > 1 && … == Q4_0`), `dense_ffn_layer.cpp:148–157`, `qkv_layer.cpp:256`; `nntrainer/tensor/float_tensor.cpp:765–775` (QS4CX_WH on a CPU dot throws), `:810–820`, `:1037–1039` (`M > 1 \|\| accelerates_q4_0_at_m1()`, the latter false at `htp_compute_ops.cpp:1208`) | untouched: the layers keep passing the Q4_0 pointer as the key; the decode row stays CPU Q4_0 (hybrid) or resident (E2E) |
| load-time registration | `Applications/CausalLM/models/transformer.cpp:146` (`FC_LAYER_DTYPE`), `:470–491` (`fc_pending` for Q4_0 FCs under an HTP engine), `:600–660` (warm-ups via `register_q4_0_weight` / `register_q4_0_dense_ffn` / `register_q4_0_conv_block`, `compute_ops.h:421`, `:539`, `:578`) | the pending entries carry the packer's tensor name so the sidecar entry is found by name (§3.3 a); `main.cpp:380` style `resolveNntrConfigPath` for `fc_wh_file_name` |
| E2E loader | `Applications/CausalLM/models/lfm2_moe/lfm2_moe_causallm.cpp:154–222` `load_weight` (collects Q4_0 tensors by layer name, `add_decode_graph_q4_0`), `:224–231` `repack_weight` → `finish_decode_graph_q4_0` | PR 2: the FC kinds hand no Q4_0 weight when the sidecar is present; the tied lm_head still does |
| config of record | `docs/measurements/config/q40-qs4cx-wh.nntr_config.json` (md5 `110cb8bc…`) | gains `fc_wh_file_name` (+ `fc_wh_format`); new md5 row |
| runners / md5 | `docs/measurements/222-run.sh`, `222-stage.sh` (templates; one binary set, `M=../models/q40-qs4cx-wh`, `MF` = the main bin) | new `225-run.sh` / `225-stage.sh`: push the sidecar, md5 it on both ends, 9 cells × 3 cases |
| IDL / stub | `test/htp/nntr_hvx.idl`, `nntrainer/tensor/htp_backend/generate_stub.sh`, `test/htp/build.sh` | **PR 1: no IDL change** (the overflow path reuses `weight_register_u8i4`, `nntr_hvx.idl:34`). PR 2: no new entry either — the graph descriptor word carries the WH bit; skel rebuild for both PRs (conv kernel in PR 1, graph runner in PR 2) |
| profile / report | `NNTR_HTP_PROFILE` registration row (`addRegister(convert_us, …)`), `tools/htp_fc_report.py` | the `convert` column reads 0 on the config of record; PR 2 adds the per-kind FC-WH row under the existing kind names; no table shape change |
| host fixtures | `test/htp/host/run_inproc_e2e.sh:215–219` (the lfm25 fixture quantized twice: `--moe_dtype QS4CX` CPU control, `--moe_dtype QS4CX_WH` HTP) | the HTP quantization adds `--fc_wh_sidecar`; new lines (§4) |

Consumers that must move with the contract: the packer's config writer
(`fc_wh_format` tag), the loader check (`HtpComputeOps` refuses a sidecar
whose tag or shapes disagree), the IDL stub only in PR 2 if the descriptor
grows (it does not; `HTP_GRAPH_OP_WORDS` keeps 16 + 2 × 32 words), the
profile row and `htp_fc_report.py` (unchanged columns), BENCHMARK Artifacts
(one new md5 row), the handoff skill's artifact table (the sidecar line).

## 3. Design

### 3.1 Coexistence with the CPU (the issue's open question)

Facts: the CPU has no QS4CX_WH kernel (`float_tensor.cpp:765` throws);
the hybrid's decode row runs every FC on the CPU at M == 1 (`conv_block_layer.cpp:189`,
`dense_ffn_layer.cpp:151`, `float_tensor.cpp:1037`), so the **hybrid needs
Q4_0 FC bytes on the host** whatever prefill reads; the CPU-only case
already uses its own file (`q40`, md5 `d28f55c5…`, experts Q4_0) and never
reads the NPU file.

| option | serves | cost | verdict |
|---|---|---|---|
| (i) two full model files: `q40` (CPU) and a new NPU file whose FCs are WH only | CPU-only, E2E | the hybrid breaks: no Q4_0 FC for its decode row; routing the M = 1 FCs to the DSP costs ≈ 54 FastRPC calls a token (≈ 0.4 ms each, doc 50 §3.4) | rejected |
| (ii) both tensors in one file (Q4_0 then WH per FC) | all three | the app layers (`conv_block`, `dense_ffn`, `qkv_layer`) and the **core** `fully_connected` (`attention_out`) must declare a second weight; the graph builder, the nntrainer weight reader and the one-layer / split forms all change; the main bin's md5 (`7b7867fa…`, BENCHMARK:800) moves; the WH copy sits in host RAM until `releaseArmSource` | works, widest diff |
| **(ii′) WH sidecar beside the main file** — `nntr_lfm2_8b_a1b_fcwh.bin` (≈ 216 MiB, the 67 FC tensors as today's `QS4CX_WH` tensors in packer order with a 4 KiB name index), named by `fc_wh_file_name` in `nntr_config.json` like `embedding_file_name` (`transformer.cpp:149`, `main.cpp:380`) | all three | packer: one extra output stream in the same walk; loader: `HtpComputeOps` preads each image straight into the arena (`readWeight` `:5603` does this for experts today), **zero host RAM, main bin and its md5 untouched, no layer or graph change**; the CPU-only `q40` dir has no sidecar and is unaffected; a model without the key falls back to today's requant path (kept for other models) | **recommended** |
| (iii) a CPU WH GEMV (doc 48 §7) | one file for all | a NEON kernel reading 32 × 32 tiles + a `_det` spec + a new CPU decode accuracy point; contract §2 states "no CPU fallback" as a design fact | rejected for this issue; a separate track if ever wanted |

(ii′) is a variant of the user's (b): the WH bytes are packed once from f32
by the same quantizer, live next to the model file and are the one DSP
copy. **If the user wants a literal single `.bin`, (ii) is the fallback and
the rest of this plan is unchanged** (only §3.3 a's source of bytes moves
from a pread to a tensor pointer). Flagged in the issue comment.

The hybrid then holds Q4_0 FCs on the host (255 MiB, as today) for its
decode row, and the DSP holds WH once. E2E holds the same Q4_0 bytes on the
host unused (255 MiB of a 12 GB phone; `MADV_DONTNEED` is a one-line
follow-up, not in this plan).

### 3.2 Packer

* `nntr_quantize_stream <fp32 dir> --fc_dtype Q4_0 --moe_dtype QS4CX_WH --fc_wh_sidecar --isa ARM`:
  `writeFc` quantizes the already-read, transposed f32 `source` a second time
  with `DType::QS4CX_WH` into the sidecar stream (`writeQuantized` `:776` +
  `flushQs4cxScales` `:838`), so the Q4_0 and the WH of one weight come from
  the same f32 pass — one packer run, as `q40-qs4cx-wh` itself was
  (BENCHMARK:800: `nntr_quantize_stream` from the fp32 bin, `--moe_dtype QS4CX_WH`;
  the fp32 bin is `res/lfm2_moe/.../weight_converter.py`'s output of the HF
  checkpoint, `/local/mnt/workspace/models/lfm2.5-8b-a1b/fp32/`). The main
  bin's bytes are unchanged: re-run the packer, `cmp` against `7b7867fa…`
  as a host check.
* Sidecar layout: header (magic `NNTRFCWH`, version 1, count 67, per entry
  `name[64]`, `K`, `N`, `offset`, `bytes`), then the images = `QS4CX_WH_Tensor`
  bytes (`[WH nibbles][N f32 scales][N f32 colsums]`, `qs4cx_tensor.h:299`).
  The packer writes `fc_wh_file_name` and `fc_wh_format: "QS4CX_WH/1"`
  into the output config (`writeOutputConfig` `:1260`), the loader refuses
  any other tag (gates skill: a layout change bumps the tag in the same PR).
* Dense FFN down (K = 7168): per-column scale over 7168 values chosen by the
  quantizer on f32 — doc 51 §2.19's 224:1 double-quantization loss is gone
  by construction (the experts' `htp_qs4cx_from_packed` case).
* Dry run and `requireEndOfFile` unchanged; the sidecar's own byte total is
  checked against its header.

### 3.3 Prefill consumer (PR 1): WH images from the sidecar into the arena, no heap copy

(a) **Names.** `Transformer::repack_weight`'s `fc_pending` / `dense_pending`
/ `conv_pending` (`transformer.cpp:470–491`, `:600–660`) carry the packer's
tensor name (`layer<l>_conv_in_proj`, `_wq` …; the one-layer forms map their
weights to those names in `lfm2_causallm.cpp`). The hooks
`register_q4_0_weight / _dense_ffn / _conv_block` get a name overload; the
old overload stays for other backends.

(b) **Registration.** `get_or_register_fc(key, K, N, name)`: if the sidecar
has `name` with matching (K, N), pread the image (6 MiB for in_proj) into
a host staging buffer, then place each handle's tiles in the arena and
`registerFromArena` with the sliced scales / colsums:
* contiguous handles (out_proj, q, k, v, o; dense down row chunks = k-tile
  ranges): pread straight into the arena at the slice offset, as
  `readWeight` does;
* column slices (in_proj thirds a, b, c; dense gate\|up pairs): a tile
  gather (512 B tiles, k-major) into the arena — cheap, tile granularity 32
  columns; the down chunk's per-chunk colsum is summed from its tiles
  (the int4 values, no requant).
  No `htp_qs4cx_from_q4_0x4`, no `whPack`, no heap. Keyed by the Q4_0
  pointer as today, so `gemm_q4_0_*` callers do not change.
  `[HTP] fc wh: file=<name> handles=<n> arena=<MiB> heap=<MiB>` is printed
  once.

(c) **Address budget — the hybrid.** Doc 46 §41: a PD maps 3840 MiB in
256 MiB chunks and the heap shares the same 4 GB. The hybrid at full
residency holds 3696 MiB of experts (15 chunks, ≈ 144 MiB of tails) and
needs 216 MiB of FC WH: **≈ 72–86 MiB do not fit in the arena, whichever
format.** Today those bytes go to the heap through `registerRm` (requant +
bake) and fit at P512; P1024 fails because the conv kernel's scratch grows
with M (failure 2). PR 1 handles both without an IDL change:
* overflow: the slices `placeExisting` cannot place are **unpacked bit-exactly**
  from WH tiles to row-major int8 on the host (the inverse of `whPack`,
  the `htp_qs4cx_from_packed` pattern) and registered through the existing
  `weight_register_u8i4` (`nntr_hvx.idl:34`, which bakes them back to WH
  on the heap: same bytes, no requant). The banner's `heap=<MiB>` makes the
  overflow visible; on the E2E path (pool C = 28, 13 chunks) nothing
  overflows (3328 + 216 + 147 lm_head Q4M1 = 3691 < 3840, §3.4);
* P1024: the host chunks the conv block, the dense FFN and the MoE prefill
  calls at 512 rows (as `fcMaxRows` already does for the FCs,
  `:1260–1270`), so the scratch stays at the P512 size that fits today.
  The conv kernel must read the incoming `state_f32` as the two history
  rows of g for the second chunk (`hexkl_conv_block.c:461–468` only writes
  it); zero history for the first chunk stays the default. A host check
  proves chunked == unchunked bit for bit (int32 tiles; the f32 epilogue
  is per row).
  Cost: one FastRPC call per extra chunk (≈ 0.4 ms × ≈ 40 at P1024 on a
  ≈ 1.3 s prefill).
  Fallback if the hybrid still fails to load on the device: the runner
  re-runs it with `NNTR_MOE_CACHE_EXPERTS=31` (the pool exists on the
  hybrid, `lfm2_moe_layer.cpp:464`) and records that as a deviation —
  it changes the hybrid's decode (misses) and is **not** the case of
  record; the user decides.

Rejected for PR 1: a new IDL entry that stores WH bytes on the heap as-is
(saves one bake, costs a stub regeneration and a skel for 72 MiB of bytes
that the hybrid should not keep on the heap at all); fewer keyed layers via
`*_htp_layers` (not the config of record).

### 3.4 E2E decode consumer (PR 2): FC / DENSE_FFN ops on the WH GEMV, LM_HEAD stays Q4M1

What exists: the FC set is Q4M1 (`registerQ4m1` `:1773`, CPU-exact, LEDGER
rule 49, 448 MiB with the lm_head 147); the native lever (`NNTR_HTP_PPL_LEVERS=0x2`,
`:1672–1690`, plan 194 L1) is a *different accumulate on the same Q4M1
bytes*, not a WH reader; the WH GEMV is the MoE M = 1 path
(`hexkl_mm_u8i4_moe.c:717–760` `moe_m1_pair_worker`: u8 row quant →
`hvx_gemm_u8i4_wh_col` → dequant / SwiGLU → down GEMV, VTCM-fed one expert
ahead at ≈ 33 GB/s, rule 35). Plan 132 rejected "(ii) requant to WH" only
under the bit-preserving rule, which #201 / the issue's gate replaced by
PPL + text approval for the keys.

What is missing:
* `HTP_GRAPH_FEED_WH` on FC / DENSE_FFN ops; their `h_gu` / `h_dn` words
  then name u8i4 registry handles (`weight_register_u8i4_arena`) instead of
  Q4M1 slots; `bindQ4m1` binds LM_HEAD only; `Lfm2MoeCausalLM::load_weight`
  hands only the tied lm_head when the sidecar is present.
* `nntr_hvx_fc_wh_graph` (DSP): per-op u8 row quantization of the 2048-wide
  input (`hvx_quant_rows_u8_params`, one row), one `hvx_gemm_u8i4_wh_col`
  per part, the dequant epilogue (int32 − zp·colsum, × scale), parts
  concatenated as the Q4M1 runner does; DENSE_FFN = the MoE M = 1 pair path
  with 4 fixed "experts" of weight 1 (`invokeMoeLayer(kind=1)`'s meaning,
  `:5063–5075`). The DMA feed: the FC set is 216 MiB a token; direct arena
  reads run at 21–27 GB/s (rule 26) ≈ 8–10 ms, the MoE's VTCM feed at ≈ 33
  ≈ 6.5 ms, today's Q4M1 VTCM feed at 45.5 GB/s reads its 255 MiB in 5.1 ms
  (rule 49). **So the WH FC ops must reuse the MoE feed (stage the next
  part into VTCM during the current GEMV) to break even; the gain is the
  byte cut (4.0 vs 4.5 bits a weight, ≈ −0.5 ms) and the fit, not speed.**
  The handoff reads `dsp_us` per kind against the Q28 cell of #222 (FC +
  DENSE_FFN + LM_HEAD 9.2 ms, rule 59a).
* Host spec `fc_wh_det.h` (the u8 quantizer + int32 GEMV + dequant in one
  scalar order; the MoE's `_det` pieces reused) and the graph host check's
  FC / DENSE_FFN cases under the WH bit; `run_inproc_e2e.sh` lines (§4).

Can E2E keep Q4M1 as a step-1 fallback? On the host yes (no address limit),
and PR 1 keeps it: with the sidecar present but PR 2 absent, E2E runs
prefill on WH and decode on Q4M1. **On the S25 it does not fit**: pool
C = 28 (3328) + Q4M1 set 448 + WH 216 = 3992 > 3840 (that is #222's
`mapped=3712` failure plus 216). C = 24 fits (2772 + 448 + 216 = 3436) but
is a different cell. Hence the device handoff is after PR 2 (§4), and PR 1's
E2E is a host gate only.

### 3.5 lm_head

Stays Q4_0 in the main file (tied embedding; the packer refuses WH for the
embedding, `:550`; the embedding lookup needs the Q4_0 table anyway) and
Q4M1 on the E2E arena (147 MiB, CPU-exact; a WH lm_head would be a second
125 MiB copy of the same table and would put the per-column int4 grid on
the logits, where PPL is most sensitive). The E2E arena after PR 2: 3328 +
216 + 147 = 3691 MiB in 15 chunks, ≈ 150 MiB of slack.

### 3.6 Accuracy — two candidate thresholds, needs-user

* **T1 (contract §1, amended 2026-09-28):** pooled decode PPL forced on the
  CPU `q40` continuation ≤ 1.02 × A, A = the **no-keys hybrid on the same
  binary and model in the same sitting** (the NPU "switch-off"; the CPU
  `q40` PPL is recorded as information, LEDGER ⑱: the NPU model sits ≈ 4.9 %
  from it by its weights alone); prefill PPL read the same way; text
  approved.
* **T2 (#201 / LEDGER rule 45):** slight drift is fine; a loop
  (`tools/htp/loop_check.py --prompt`) where A does not loop, or off-context
  text, fails; text approved by the user.

Expectation (doc 51 §2.15 / §2.18, single quantization at K = 2048: conv on
HTP read −1.5 %; §2.19: the dense loss was the 224:1 double quantization,
gone here): the keys should land near A, far from #222's +42 % / +13.6 %.
The host SNR line of §4 (`E2E eval fcwh-lfm25 min_snr_db`) against the CPU
`q40`-style control is the first read; if it is not well above #222's
behaviour the device sitting is not booked.

Doc 45 §3 compliance: no new op before a quantizer that is not `_det` (the
u8 row quantizer is the MoE's); weight DMA hidden behind compute (the feed
reuse is the requirement, not an option); the prefill kernels and their
double buffers are untouched.

## 4. Steps

Each step ends in a `.claude/skills/hexagon-gates` rung. Two PRs on
`htp_first_version`: PR 1 = steps 1–4 (`htp/225-fcwh-prefill`), PR 2 =
steps 5–6 (`htp/225-fcwh-e2e`); the handoff is step 7.

1. **Packer: the sidecar.** `--fc_wh_sidecar`, header + images, config
   keys, `cmp` of the main bin against `7b7867fa…` on the 8B
   (`nntr_quantize_stream /local/mnt/workspace/models/lfm2.5-8b-a1b/fp32 …`,
   ≈ 10 min, writes `q40-qs4cx-wh/nntr_lfm2_8b_a1b_fcwh.bin`); on the
   fixtures the sidecar's images are byte-equal to a `--fc_dtype QS4CX_WH`
   run's tensors (a new `*Lfm2Moe*` gtest case). Gate: rung 1
   (`unittest_causallm_models --gtest_filter='*Lfm2Moe*'` 6 + 1 PASSED,
   `run_host_checks.sh` ALL CHECKS PASS, `tools/htp_syntax_check.sh`).
2. **Loader: WH from the sidecar into the arena.** §3.3 a–b, the banner,
   the format-tag check, the bit-exact unpack overflow in `registerRm`, the
   requant path kept for a model without the key. Gate: rung 1 +
   `run_inproc_e2e.sh` with the lfm25 fixture quantized with the sidecar:
   new lines `E2E keys fcwh-lfm25 handles=<n> arena=<n> heap=0 requant=0 ok`
   and `E2E eval fcwh==wh-lfm25 bit_identical=1` (the sidecar handles'
   bytes equal `get_or_register_wh` of the same fixture's WH tensors), plus
   every existing line (`E2E keys lfm25 … bit_identical=1`, `INPROC E2E PASS`).
3. **P1024: chunk the prefill calls at 512 rows; the conv kernel reads the
   state.** Host check in `test/htp/host/` (conv block chunked == whole,
   dense / MoE chunked == whole, bit-identical). Gate: rung 1, then rung 2
   (`test/htp/build.sh` for v79 **and** v81: `UNDEFINED SYMBOLS OK`,
   `ARCH OK`), then `run_inproc_e2e.sh` with the fixture's `init_seq_len`
   raised so one prefill is two chunks (`E2E keys lfm25-p2x … bit_identical=1`).
4. **PR 1 app build.** Gate: rung 3 (`build_android.sh --htp`, NEEDED lines,
   `NNTR_HTP_FORWARD_KINDS` count ≥ 1, md5s recorded). Open PR 1 (target
   `htp_first_version`); the E2E path on PR 1 still binds Q4M1 and passes
   the host E2E lines; the device does not get PR 1 alone (§3.4).
5. **E2E: FC / DENSE_FFN on WH handles.** §3.4: feed bit, binding,
   `nntr_hvx_fc_wh_graph`, DENSE_FFN through the pair path with the VTCM
   feed, `fc_wh_det.h` spec, graph host check cases. Gate: rung 1
   (`run_host_checks.sh` prints the new `FC WH BIT-IDENTICAL` line;
   `run_inproc_e2e.sh` prints `E2E fwd lfm25 kinds=all calls/token=1.00
   q4m1_handles=<lm_head slices only> wh_handles=<n>`, `E2E eval
   e3fcwh-lfm25 … min_snr_db=<x>` ≥ 30 against the switch-off run, `E2E
   tokens e3fcwh==e1-lfm25 8/8`, `E2E ppl-decode …` lines, `INPROC E2E PASS`),
   rung 2 (v79 + v81).
6. **PR 2 app build.** Rung 3 as step 4; stage the set with `225-stage.sh`
   (app, both skels, the evictor, prompts, the config of record with the
   new key, md5.txt including the sidecar). Open PR 2.
7. **Device handoff (user, S25 farm, `needs-user` + `state:needs-measurement`)**
   — `docs/measurements/225-fcwh-table.md` + `225-run.sh`. One binary set;
   variants ≤ 4: **A** = CPU `q40` (its config with `init_seq_len 1024` for
   P1024, as #222 did); **Aoff** = the NPU model, config of record minus
   the three engine keys (no HTP prefill FC; the PPL reference for T1);
   **B** = hybrid on the config of record; **Q** = B + `NNTR_HTP_E2E=1
   NNTR_MOE_CACHE_EXPERTS=28`. Cells: the 9 (P64 / P512 / P1024 × G64 /
   G512 / G1024) for A, B, Q (27 runs; G64 twice mirrored for B and Q,
   cool start per block ≤ 35 °C), Aoff at P512 × G64 / G512 only (the PPL
   reference; 2 runs); two profile runs (`NNTR_HTP_PROFILE=2`, B and Q at
   P512 G64: the `M>1` FC rows' `convert` = 0, Q's per-kind `dsp_us`); the
   PPL block: 8 prompts at G=256, `NNTR_PPL=1` + `NNTR_PPL_DECODE` (A
   writes its continuation; Aoff, B, Q forced on it); texts pasted (G64
   run 1) for approval. The runner records a case that does not load as
   VOID with its arena / heap lines and goes on; the hybrid VOID triggers
   the C=31 deviation run (§3.3 c). Per run it checks `calls/token=1.00`,
   `cpu fc skipped = 32 × tokens`, `fc wh: … heap=0` (Q) / `heap=<MiB>` (B),
   the device md5 of the sidecar. **Estimate: ≈ 90 min device time** (27 +
   2 + 4 + 2 + 24 PPL runs ≈ 59 runs at ≈ 1–1.5 min each incl. load, 9 cool
   waits, reboot + 5 min idle) plus ≈ 15 min of workstation steps.

Step 7 is the only unavoidable device measurement; steps 1–6 are host
and build gates.

## 5. Risks

| risk | how it shows in the handoff |
|---|---|
| Address space (host cannot check the 32-bit PD budget): the hybrid's overflow onto the heap (≈ 72–86 MiB) plus scratch; the E2E 15th chunk | the `fc wh: arena= heap=` banner, `heap_used_kib`, `s1_arena_mib`, `fastrpc_mmap` lines in `void_<v>`; the C=31 deviation cell if B is VOID |
| DMA rate of the WH FC ops at M = 1: without the VTCM feed the FC set reads at 21–27 GB/s and E2E decode loses 2–4 ms a token vs Q4M1's 45 GB/s | Q's per-kind `dsp_us` profile against #222's Q28 (`FC + DENSE_FFN + LM_HEAD` 9.2 ms); decode tok/s Q vs #222's Qold cells on the same unit class |
| Accuracy: the u8 row quantizer on every FC input at decode (E2E) and the per-column int4 grid on K = 7168 downs | the PPL block (T1 / T2), `loop_check.py`, the texts; the host SNR line first |
| Thermal drift between sittings and units (rules 9, 20, 34, 52, 59d) | every verdict is A/B inside this sitting; `therm.log` per block; cool start per G block; G64 mirrored |
| Stale skel / stub (doc 46 §48.7): PR 1 changes the conv kernel, PR 2 the graph runner | the runner stops on `0x8000040e`; skel md5 per arch in md5.txt; `strings … NNTR_HTP_FORWARD_KINDS` ≥ 1 |
| Prefill chunking at 512 rows adds ≈ 40 FastRPC calls at P1024 | the P1024 prefill tok/s cells against P512's; the profile's call counts |
| The sidecar and the main bin drift apart (a re-packed main bin with an old sidecar) | the format tag + per-entry (K, N) check at load; md5 rows for both files; `cmp` in step 1 |
| `init_seq_len 1024`'s first-token artefact (plan 222 §3.2) | unchanged from #222: the runner's `same+1st` compare |

## 6. Docs to update

* BENCHMARK.md: Artifacts — one row for the sidecar
  (`q40-qs4cx-wh/nntr_lfm2_8b_a1b_fcwh.bin`, md5, packer commit, "FC WH from
  f32, one run with the main bin"), one row for the config of record's new
  md5 (`fc_wh_file_name`), the staged set rows; Results — the 27 cells of
  the table of record (CPU only / hybrid / E2E one PD) tagged with the unit,
  the Aoff PPL reference, the profile rows; Goals — the LFM2.5 table of
  record closed on the config of record if G1–G4 pass.
* LEDGER.md: a rule for the address-budget fact (full-residency hybrid +
  216 MiB of FC WH exceeds 3840 by ≈ 72–86 MiB; the overflow goes to the
  heap bit-exactly), one for the measured WH-FC decode rate vs Q4M1
  (whatever the sitting reads), one for the prefill-PPL effect of single vs
  double quantization on the same keys (#222's +42 % against this sitting);
  the §2 row for #225 and the cycle entry.
* Contract 0001 §2 "two weight formats, two model files": add the sidecar
  (three files: `q40`, `q40-qs4cx-wh` main, its FC WH sidecar) and that
  `QS4CX_WH` still has no CPU fallback.
* `docs/htp_moe/guide` (the model-file / config section): the new key and
  packer flag.
