# A8W8 / A8W4 flash attention on HMX over a quantized KV cache

Status: 2026-09-28, Phase Q1 delivered on device (SM8850 / v81):
`unittest_hexkl_kv_q` 5/5 host, `unittest_hvx_attn_q` 4/4 device -- the DSP
quantizer matches the same C on the ARM side bit for bit for both kinds, HMX
over the baked int8 and int4 tiles equals a plain int matmul over the masters
exactly, and the int32 accumulator readout probe found `usable=1 base=0
row_stride=32`: the tile lands as a plain row-major 64x32 int32 matrix, so
every int kernel reads it as 64 HVX vectors with no permutation table.
Branch: `htp/quant-dequant-hvx-opt`. Builds on
`20_hmx_flash_attention_plan.md` (fp16 attention, delivered through Phase 4d).

Goal: the same attention layer (`MHACoreLayer` -> `ComputeOps` -> FastRPC ->
HMX/HVX) with the KV cache held as int8 (A8W8) or int4 (A8W4) on the DSP and
the matmuls run on HMX's 8-bit port through the HexKL kernels this tree
already uses for its A8W4/A8W8 fully-connected layers
(`hexkl_micro_hmx_mm_u8i8` / `_u8i4`, `rm_to_wh_i8` / `_i4`, the int32
accumulator), with `hvx_quant_u8` / `hvx_dequant_i32` and the fp16 attention's
HVX softmax reused as they are.

"A8" is the activation port: Q and the probabilities P enter HMX as uint8.
"W8" / "W4" is what plays the weight: K^T in Q.K^T and V in P.V, i.e. the
cache.

## 1. Where things stand

### 1.1 Reusable, on this branch
- fp16 attention (`hexkl_attn_f16.c`): FA-2 pipeline (softmax on the pool
  overlapping the next block's DMA + Q.K^T), causal geometry in
  `hexkl_attn_f16_plan.h`, fp16 HVX softmax (`hvx_attn_softmax_f16.h`),
  the pure-HVX decode kernel, the resident tile registry pattern
  (`hexkl_kv_tiles_f16`: register / append / attention by handle), and the
  device gtest with SNR gates. Device numbers: 62-78 dB, 27/27.
- A8W4 / A8W8 matmul: `hexkl_mm_u8i4{,_dma}.c`, `hexkl_mm_u8i8_dma.c` --
  WH bake once + resident weights, `hvx_quant_u8` (per-row asymmetric u8,
  writes AH tiles directly: a u8 activation tile is flat row-major 64x32),
  `hvx_dequant_i32` (`(acc - zp*colsum) * s_act * s_w + bias`).
- Session: int32 accumulator config is already the session's permanent
  one (`config_off`); the fp16 kernel writes its own f16 config per call.
- rpcmem-shared host cache (Phase 4d) -- not needed here: the quantized
  cache lives on the DSP, only the new rows cross FastRPC per step.

### 1.2 Prior u8 SDPA (`seunghui/htp/rope-u8-qnn-parity`, fa80ac82a; not here)
What it settled, and this plan takes as given:
- **Never call `hexkl_micro_hmx_copy_32b_to_submatrix` on the hot path**:
  52.8 us per 8 KiB tile, 80% of a prefill layer and 97% of a decode one.
  `hexkl_acc_tile.{c,h}` derives the int32 readout permutation at runtime
  by pushing a ramp through the vendor copy once; on every part measured
  the tile is row-major with a constant row stride, so a 64x32 int32 tile
  is 64 HVX vectors and the dequant reads it in place. This file is
  ported as is.
- **Quantization axes that work on HMX.** Dequant constants can only live
  on the non-reduced axes: for S = Q.K^T that is per Q row (activation)
  and per cache row (weight column); for O = P.V it is per Q row and per
  head dim. A per-token V scale therefore cannot be a "weight scale" --
  it sits on the reduced axis.
- **Accuracy**: (K i8, V i8) 5e-3 max rel err; (K i4, V i8) 2e-2;
  V at i4 with per-channel-over-block scales 5e-2 and red on rows that
  attend few positions. "V to i4 costs ~15x, K to i4 costs ~3x." Its V
  scale axis (per head dim over a 32-64 row block) is the reason: a
  symmetric 15-level grid over a block's column range.
- FastRPC fixed cost ~404 us per call; the transport spreads 90 -> 3900 us
  without a HAP power vote. Fork/join costs more than a decode-band softmax.
- It quantized on the host and kept fp16 shadows; no `ComputeOps` /
  `MHACoreLayer` seam existed. It used a blocked (two-pass) f32 softmax
  with S round-tripping DDR.

### 1.3 Contract (unchanged from the fp16 kernel)
Q f32 post-RoPE, out f32, causal by geometry, sliding window, softcap,
sinks, GQA rows packed as `q*G + g`. Reference is `MHACoreLayer`'s f32 math;
the gate is SNR against it. The int paths are expected to land lower than
fp16's 62-78 dB; the gates below are set from the prior branch's numbers
and tightened once measured.

## 2. Decisions

1. **Cache scales are per token, computed on the DSP at append.** K: one
   symmetric scale per (row, kv head) over head_dim, plus `colsum` (the
   int sum over head_dim) for the u8 zero-point correction. V: one
   symmetric scale per (row, kv head, 32-dim group). Every appended row is
   quantized once and never revisited, so the cache grows token by token
   without re-quantization and without calibration data. No host-side
   quantizer, no fp16 shadow.
2. **The per-token V scale is folded into P, per 32-dim group.** P.V on
   HMX produces output tile (row tile, d) from activation P and weight
   tile column d. Since the V scale of group d is per cache row k -- the
   reduction axis -- it is multiplied into P before P is quantized:
   `P'_d[q][k] = P[q][k] * s_v[k][d]`, then `P'_d` gets its own per-row u8
   scale. There are head_dim/32 activation buffers for P instead of one;
   HMX runs exactly the same number of passes (output tile (i, d) always
   used activation i and weight column d), and V gets llama.cpp's Q8_0 /
   Q4_0 granularity (32-value groups) for free. This is the answer to the
   prior branch's V-at-i4 result; it is verified on device before A8W4
   is called done.
3. **Q is per-row asymmetric u8** (`hvx_quant_rows_u8_params` +
   `hvx_quant_pack_u8_ah`, unchanged; the pool parallelism they already
   have). The score dequant is exactly `hvx_dequant_i32`'s formula with
   `act = Q rows`, `w_scale = s_k[k]`, `colsum_w = colsum_k[k]`, `bias = 0`,
   with `log2e/sqrt(hd)` folded into the per-row activation scale. P is
   u8 with zero point 0 (P' >= 0), per-row scale = row max / 255 -- the
   prior branch's finding that a fixed 1/255 is wrong for blocks that do
   not hold the row max.
4. **S and O leave the accumulator through HVX, in VTCM, never DDR.**
   acc(int32, 64x32) -> dequant per row -> f16 S tiles in the 2-row
   interleaved layout the fp16 softmax consumes, so `softmax_row_pair` runs
   unmodified. O is a f32 HVX-resident block `[g_br][hd]`: per cache
   block `O = a * O + s_p[q][d] * acc` (the online rescale is an HVX FMA,
   not the fp16 kernel's diagonal HMX matmul -- with int accumulators O
   cannot stay on HMX across blocks). Final `out = O / l`.
5. **Row tile is 64.** The u8 activation tile is 64x32, so `g_br` aligns
   to 64 (the fp16 kernel's 32). The geometry helpers in
   `hexkl_attn_f16_plan.h` are parametric in `g_br` already; only the
   layout differs, in a new `hexkl_attn_q_plan.h`.
6. **A8W4 is A8W8 with the weight kind swapped**: `mm_u8i4` for `mm_u8i8`,
   512 B tiles for 1024 B, `rm_to_wh_i4` for `_i8`, values in [-8, 7] for
   [-127, 127]. One kernel, a `kind` struct with tile bytes, matmul and
   bake function pointers and the int range. Masters (row-major copies the
   decode path reads) stay int8 containers for both kinds in this phase;
   packing the i4 master is listed as follow-up.
7. **Decode (n_q < 5, hd <= 128) is pure HVX on `vrmpy`**: 4 MACs per
   lane per instruction on 8-bit data, against fp16's 1. The registry
   keeps the masters in the layout that makes this one instruction per
   32 cache rows x 4 dims (K^T `[hd/4][rows][4]`) and one per 4 rows x 32
   dims (V `[rows/4][hd][4]`), and bakes the HMX WH tiles from a 32-row
   row-major staging of the column tile being filled. Q is symmetric i8
   for this path (`vrmpy(Vb, Rb)`); softmax and the f32 O accumulator are
   the fp16 decode kernel's.
8. **Seam**: the cache is a DSP handle. `ComputeOps` gains
   `kv_cache_register / append / release` and `sdpa_q_kvcache(handle, ...)`,
   `HtpComputeOps` forwards over FastRPC, `MHACoreLayer` appends this
   step's fp16 rows right where it writes them into the host cache and
   calls attention by handle. The host fp16 cache stays the source of
   truth in this phase (save/load, rollback, CPU fallback all keep
   working), so host memory does not shrink yet; dropping the host copy
   is a `KVCacheManager` follow-up. Selection: `"attention_kv_dtype":
   "q8" | "q4"` in `nntr_config.json` next to `"attention_engine"`.
9. **Validation as before**: every pure-arithmetic part host-unit-tested
   (plan/layout, quantizer references, dequant formulas); kernels gated on
   device by SNR against the f32 reference (A8W8 >= 40 dB, A8W4 >= 30 dB
   to start), resident-vs-raw bit-identity where a raw path exists;
   `-Wall -Werror` on hexagon-clang.

## 3. Design

### 3.1 Registry `hexkl_kv_q` (per attention layer, per kind)
```
masters (DSP heap, int8 containers, zero until written):
  kT4 [n_head_kv][hd/4][max_rows][4]         decode K^T, 4-dim interleaved
  v4  [n_head_kv][max_rows/4][hd][4]          decode V, 4-row interleaved
  s_k [n_head_kv][max_rows] f32, colsum_k [n_head_kv][max_rows] i32
  s_v [n_head_kv][max_rows][hd/32] f32
tiles (DSP heap, WH layout, tile = 1024 B (i8) | 512 B (i4)):
  kt  [n_head_kv][max_rows/32][hd/32]         K^T weight tiles (rows=dims, cols=cache rows)
  v   [n_head_kv][max_rows/32][hd/32]         V weight tiles (rows=cache rows, cols=dims)
staging: [n_head_kv] x (32 rows x hd) int8 for K^T (as [hd][32]) and V (as [32][hd])
```
`append(handle, row0, n_rows, k_f16_rows, v_f16_rows)`: per row, per head:
HVX absmax over hd -> s_k, round-to-nearest to the kind's range, colsum;
per 32-dim group of V -> s_v; scatter into the masters and the staging;
for every column tile touched, re-bake its hd/32 K^T tiles and hd/32 V
tiles with `rm_to_wh_*` from the staging (source in DDR: the fast regime
for HexKL's extraction helpers) through a VTCM scratch tile into the heap
tile array. Cost per token: 2*(hd/32) tile bakes per head, a few us.

### 3.2 Prefill kernel `hexkl_attn_q_prefill`
VTCM regions (`hexkl_attn_q_plan.h`), rt = g_br/64, ct = bc/32, dt = hd/32:
```
q_ah    [rt][dt]         u8 AH tiles (2 KiB), zp/scale per row in q_meta
kt_wh[2][ct][dt]         K^T tiles (kind bytes)      <- DMA from registry
v_wh [2][ct][dt]         V tiles                     <- DMA from registry
acc     [2]              8 KiB int32 readout tiles
s_hf [2][2*rt][ct]       f16 32x32 tiles for the softmax (as fp16 kernel)
p_ah [dt][rt][ct]        u8 AH tiles of P'_d
o_f32   [g_br][hd]       f32
meta    q_scale/q_zp/m/l/a/p_scale[dt] per row
```
Per (kv head, q block): qprep = `hvx_quant_rows_u8_params` +
`hvx_quant_pack_u8_ah` over the gathered f32 Q rows (rows `q*G+g`, scale
fold `log2e/sqrt(hd)` into `q_scale`). Per cache block, the fp16 kernel's
pipeline with three inserted HVX passes:
- **acc -> S**: per output tile (i64, col): dt x `mm_u8i8|i4`, one
  `acc_read_int32` into `acc[..]`, then 64 rows x
  `(acc - zp*colsum_k) * q_scale * s_k` -> f32 -> f16, written as two
  interleaved 32x32 tiles into `s_hf`. Runs on the calling thread right
  after the read (the pool is in the softmax of the previous block).
- **softmax**: unchanged (`softmax_row_pair` over `s_hf`), yields P (f16)
  in place, `a` and `l` per row.
- **P -> P'_d**: per d, per row: `P * s_v[k][d]` (s_v splat per lane from
  the block's cache rows), row max -> `p_scale[r][d]`, u8 RNE, flat 64x32
  store. Pool workers, one (row tile, d) per unit -- it replaces the fp16
  kernel's D-tile build.
- **P.V and O update**: per output tile (i64, d): ct x `mm_*(p_ah[d][i][col],
  v_wh[col][d])`, `acc_read_int32`, then 64 rows x
  `O[r][32d..] = a[r]*O[r][32d..] + p_scale[r][d] * acc[r]` (f32 FMA).
Store: `out = O / l` per real row.

Phase counters as in `hexkl_attn_f16_stats` plus `us_dequant` and
`us_pquant`, so the device test reports where int time goes.

### 3.3 Decode kernel `hvx_attn_decode_q`
Per (query row, head): Q -> i8 symmetric (one scale per unit). Scores in
blocks of 128 cache rows: `acc[k] = sum_g vrmpy(kT4[g][k..k+127], q4[g])`
(hd/4 instructions per 128 rows), dequant `s_q*s_k[k]*acc[k]` to f32 (two
64-lane halves), then the fp16 decode kernel's online softmax per 64 rows
(f16) and f32 O accumulate with `P'` = `P*s_v[k][d]` folded per 32-dim group:
`o[32d..] += vrmpy(v4[k/4][32d..][..], P'4)` over 4 rows at a time.
Gate: A8W8 >= 40 dB vs the f32 reference on 1x1024 and 1x4096.

### 3.4 IDL / skel
```
kv_register_q(kind, max_rows, n_head_kv, head_dim, rout handle)
kv_release_q(handle)
kv_append_q(handle, row0, in seq<uint16> k_rows, in seq<uint16> v_rows)
attn_q_prefill(handle, n_q, cache_from, cache_to, n_head_q, window, br, bc,
               softcap, q_f32, sinks, rout out_f32, rout stats_us)
attn_q_decode (same, no br/bc)
probe_acc_i32_layout(rout base, rout stride)          -- device test only
```
kind: 0 = A8W8, 1 = A8W4.

### 3.5 Host seam
`ComputeOps`: `supports_sdpa_q_kvcache()`, `kv_cache_register(kind, ...)`,
`kv_cache_append(handle, row0, n, k_f16, v_f16)`, `kv_cache_release`,
`sdpa_q_kvcache(handle, q, ...)`. `HtpComputeOps` forwards.
`MHACoreLayer`: on the first step with `attention_kv_dtype` set, registers
one handle per layer (`max_timestep`); in `one_batch_incremental_forwarding`
right after `b_cache_key_step` / `b_cache_value_step` are written, appends
those rows; `try_accelerated_attention` calls by handle; any failure ->
CPU path over the fp16 cache, exactly as today. Batch > 1: one handle per
(layer, batch).

## 4. Phases

| # | Deliverable | Gate |
|---|---|---|
| Q1 | `hexkl_acc_tile` port; `hexkl_kv_q` registry + on-DSP quantizer; IDL register/append/release + a `kv_q_dump` debug entry; host references for the quantizer | host: quantizer/dequant round trip; device: dump == host reference bit-exact, acc layout probe usable |
| Q2 | `hexkl_attn_q_plan.h` + host test; A8W8 prefill kernel; `attn_q_prefill` | device SNR >= 40 dB on the fp16 suite's shapes (softcap, sink, window, GQA) |
| Q3 | A8W4 kind (tiles, bake, range) | device SNR >= 30 dB; report K-i4 vs V-i4 contribution separately (mixed kinds in the test) |
| Q4 | `hvx_attn_decode_q` (A8W8, then A8W4 via unpack) | SNR >= 40 / 30 dB; time vs fp16 decode at 1x1024, 1x4096 |
| Q5 | `ComputeOps` seam, `HtpComputeOps`, `MHACoreLayer` append/by-handle, config key | CausalLM runs with `attention_kv_dtype` on device; output tokens match fp16 run on a short prompt |
| Q6 | Device timing table (prefill 128x1024, 32x4096; decode 1x1024, 1x4096) for f16 / q8 / q4; doc update | -- |

## 5. Risks
- **Accumulator layout**: the probe may find a non-affine layout on v81;
  then `acc -> S` needs the permutation table (still in VTCM, still no
  vendor copy). The probe result is printed by the device test first.
- **P at 8 bits.** Probabilities below 1/510 of the row max vanish. With
  the per-(row, block) scale this is the prior branch's measured regime
  (5e-3 for A8W8). If the gate fails on long rows, the fallback is P in
  two u8 planes (hi/lo) -- doubling P.V HMX passes -- before giving up on
  the int path for P.V.
- **HVX work per block grows**: two dequant passes and a P quant pass that
  the fp16 kernel does not have. Int8 HMX runs at twice the fp16 MAC rate,
  so prefill should land near fp16; the memory and decode wins are the
  point. Numbers decide, in Q6.
- **DSP heap**: master + tiles is 1.5 B/value (A8W8) or 1.5 B (A8W4 with
  int8 masters) against the host's fp16 2 B/value, with the host copy
  still present. Packing the i4 master (0.5 + 0.5 B) and dropping the host
  copy are the follow-ups that turn this into a memory win.
