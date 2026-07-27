# VTCM weight residency (HexKL micro API)

How to cut the NPU-side cost of an fc_layer matmul by keeping the WH-packed
weight resident in VTCM instead of re-deriving it per tile.

> **Status: §3.1 done, §3.2 in progress, §3.3 unbuilt.** The SDK's micro
> sample builds and passes on the V79 simulator, and VTCM is 8 MiB. The kernel
> change in §5 is measured by `vtcm_probe/`, not yet by anything shipping.
> Nothing here has run on a device: there is no IDL, no skel build, and
> `libhexkl_skel.so` comes prebuilt from the SDK. Treat §5 as a starting point
> to compile and verify, in the staged order in §3, not as working code.

## 1. Where the time goes

Measured on a Galaxy S25 Ultra, q_proj (M=64, N=2048, K=1024), u8i4:

| | μs |
| :--- | ---: |
| HexKL per fc_layer call | 545 |
| ↳ host quantize + dequantize | ~250 |
| ↳ **NPU matmul (`sdkl_npu_mm_u8i4_i32`)** | **298** |
| QNN NetRun (host-observed execute) | 513 |
| ↳ **QNN FC (device-timeline sub-phase)** | **69.9** |

The host side is settled: keeping the weight resident in NPU memory took the
call from 1214 μs to 545 μs, and threading what is left made throughput worse
(see the table in `hexkl_mm.cpp`). What remains is the 298 μs on the NPU, where
QNN's comparable phase costs 69.9 μs.

A `--sweep` on device (`hexkl_fc_compare --sweep`) put the fixed per-call cost
at ~76 μs, so roughly 217 μs of the 298 is work that scales with the shape.

## 2. The hypothesis, and why it is only a hypothesis

The SDK ships a micro-API sample, `hexkl_micro_matmul_u8i4_i32`, whose inner
loop looks like this:

```c
for (row = 0; row < A_rows; row += HEXKL_HMX_INT8_BLOCK_N_ROW) {
  /* load this row-band of A into VTCM */
  for (col = 0; col < W_cols; col += HEXKL_HMX_INT8_BLOCK_N_COL) {
    hexkl_micro_hmx_acc_clear_int32();
    for (i = 0; i < row_tiles_in_A; i++) {
      /* converts one weight tile from DDR into VTCM, every iteration,
         always to the same single-tile scratch offset */
      hexkl_micro_hmx_rm_to_wh_i4(vtcm_base, weight_offset, matW, i, col / 32,
                                  W_cols);
      hexkl_micro_hmx_mm_u8i4(vtcm_base, HEXKL_HMX_ACTIVATION_ALIGNMENT * i,
                              weight_offset);
    }
    hexkl_micro_hmx_acc_read_int32(...);
  }
}
```

Two things stand out. The RM→WH conversion sits in the innermost loop, and the
result goes to one fixed `weight_offset` — a single 512-byte scratch tile that
is overwritten every iteration, so nothing is ever reused. For q_proj that is
`(N/32) × (K/32)` = 2048 conversions and 1 MiB re-read from DDR per call, and
for `M > 64` the whole weight is re-converted once per row band.

**This is sample code, not the shipped implementation.** `sdkl_npu_mm_u8i4_i32`
is a compiled library function; whether it has the same structure is unknown.
§3.1 tests that before anything is built on top of it.

## 3. Staged plan

Each stage answers a question before the next one costs anything.

### 3.1 Does the toolchain work, and how big is VTCM?

Build and run the SDK's micro sample unmodified. It targets the Hexagon
simulator, so this needs no device.

```bash
source $HEXAGON_SDK_ROOT/setup_sdk_env.source
export HEXKL_SDK_ROOT=$HEXAGON_SDK_ROOT/addons/hexkl_addon

cd $HEXKL_SDK_ROOT/examples/hexkl_micro_hmx_mm_u8i4_i32
./build.sh --hex-arch v79
./run_simulator.sh --hex-arch v79
```

Its `main()` already prints what is needed, and on a V79 simulator it gives:

```
[HEXKL_MICRO] VTCM base = 0xd9000000  VTCM size = 8388608 bytes
[HEXKL_MICRO] Test Passed
```

**8 MiB, confirmed** — §4's budget was written against an assumed ~8 MiB and
that assumption holds. If the sample does not build or the test does not pass,
stop here; nothing below is reachable until it does.

Two setup problems worth knowing about, both hit on first contact:

- `build.sh` fails with `HEXAGON_SDK_ROOT is not set` unless
  `setup_sdk_env.source` is sourced. Exporting the variable by hand is not
  enough — the toolchain paths come from the same script.
- `hexagon-sim` wants `libncurses.so.5`, which Ubuntu 24.04 no longer ships.
  Symlinking `libncurses.so.6`/`libtinfo.so.6` to the `.5` names works;
  `ldconfig` afterwards is the part that is easy to miss, since the links can
  exist on disk while the loader cache still does not know them.

### 3.2 Does hoisting the conversion actually help?

`vtcm_probe/hexkl_vtcm_probe.c` is that experiment, written as a drop-in
replacement for the sample above: same includes, same `main()` shape, so it
builds with the same two scripts. It runs the sample's per-tile structure and
the §5 structure over identical data, times both in pcycles, and checks both
against a scalar reference. See [vtcm_probe/README.md](vtcm_probe/README.md).

Being a simulator, it is deterministic — none of the jitter that made the
host-side threading numbers unreadable — but its absolute cycle counts are not
device times. Read the ratio, not the magnitude.

This measures the hypothesis directly:

- **Large win** (approaching QNN's 69.9 μs): the conversion was the cost, and a
  custom skel is worth building.
- **Little or no win**: the shipped `sdkl_npu_mm_u8i4_i32` already does
  something smarter than the sample, and the 217 μs is elsewhere. Stop, and
  re-open the question with the sweep data.

### 3.3 Only if 3.2 pays: wrap it for the host

Wrapping the kernel so nntrainer can call it means, from scratch:

1. an IDL describing the call, compiled with `qaic` into stub (host) and skel
   (DSP) halves;
2. a DSP skel built with `hexagon-clang` for V79, linking the micro library;
3. deploying that skel next to `libhexkl_skel.so` and putting it on
   `ADSP_LIBRARY_PATH`;
4. calling the generated stub from `shgemm_u8i4_i32` in
   `nntrainer/tensor/htp_backend/hmx_ops/hexkl_mm.cpp`, in place of
   `sdkl_npu_mm_u8i4_i32`.

The weight upload becomes a one-off: the resident-weight cache added in Phase 1
already keeps the WH bytes in NPU memory, so the skel receives a pointer it can
DMA into VTCM once and reuse across calls.

## 4. VTCM budget

Tile geometry, from `hexkl_micro.h`:

| constant | value |
| :--- | ---: |
| `HEXKL_HMX_INT8_BLOCK_N_ROW` | 64 |
| `HEXKL_HMX_INT8_BLOCK_N_COL` | 32 |
| `HEXKL_HMX_INT8_BLOCK_N_INNER` | 32 |
| `HEXKL_HMX_ACTIVATION_ALIGNMENT` | 2048 |
| `HEXKL_HMX_WEIGHTS_ALIGNMENT` | 128 |

So an activation tile is 64×32 = 2048 B (exactly the alignment), a packed int4
weight tile is 32×32/2 = 512 B, and an int32 accumulator tile is 64×32×4 =
8192 B. 512 is a multiple of 128, so weight tiles pack back to back with no
padding — worth checking, since a 2048-byte alignment would have cost 4×.

For q_proj (M=64, N=2048, K=1024):

| region | tiles | bytes |
| :--- | ---: | ---: |
| weight, all tiles resident | (K/32)·(N/32) = 2048 | **1.00 MiB** |
| activation | (M/64)·(K/32) = 32 | 64 KiB |
| output | (M/64)·(N/32) = 64 | 512 KiB |
| | | **≈1.58 MiB** |

Per-layer weight totals for qwen3-0.6b (u8i4, hidden 1024, intermediate 3072):

| q | k | v | o | gate | up | down | layer |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 0.5 | 0.5 | 0.5 | 1.5 | 1.5 | 1.5 | **7.0 MiB** |

VTCM measures 8 MiB (§3.1), so q_proj's 1.58 MiB is comfortable.
One layer's weights nearly fill it, so plan on holding the
current projection (and prefetching the next) rather than a whole layer. All 28
layers is 196 MiB — never resident.

## 5. The change

Convert every weight tile once, into its own VTCM slot, before the matmul
loops; then index into them.

```c
/* --- VTCM layout ---------------------------------------------------------
 * [0]                     activations : k_tiles * 2048 B (2048-aligned)
 * [weight_base]           weights     : k_tiles * n_tiles * 512 B (128-aligned)
 * [result_offset]         accumulator : 64*32*4 = 8192 B
 * [hmx_config_offset]     HMX config  : hexkl_micro_hmx_config_size()
 */
const uint32_t k_tiles = A_cols / HEXKL_HMX_INT8_BLOCK_N_INNER; /* K/32 */
const uint32_t n_tiles = W_cols / HEXKL_HMX_INT8_BLOCK_N_COL;   /* N/32 */
const uint32_t WT_TILE_BYTES = 512;                             /* 32x32 int4 */

const uint32_t weight_base = k_tiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;

/* Prologue: one conversion per tile, each to its own slot. Hoisting this out
   of the matmul is the whole point -- the sample redid it per (row, col, k). */
for (uint32_t kt = 0; kt < k_tiles; kt++)
  for (uint32_t nt = 0; nt < n_tiles; nt++)
    hexkl_micro_hmx_rm_to_wh_i4(vtcm_base,
                                weight_base + (kt * n_tiles + nt) * WT_TILE_BYTES,
                                matW, kt, nt, W_cols);

hexkl_micro_hmx_setup_acc_read_int32(vtcm_base, hmx_config_offset);

for (uint32_t row = 0; row < A_rows; row += HEXKL_HMX_INT8_BLOCK_N_ROW) {
  for (uint32_t kt = 0; kt < k_tiles; kt++)
    hexkl_micro_hmx_copy_submatrix_to_8b_activation(
      vtcm_base, HEXKL_HMX_ACTIVATION_ALIGNMENT * kt, matX,
      row / HEXKL_HMX_INT8_BLOCK_N_ROW, kt, A_rows, A_cols);

  for (uint32_t col = 0; col < W_cols; col += HEXKL_HMX_INT8_BLOCK_N_COL) {
    const uint32_t nt = col / HEXKL_HMX_INT8_BLOCK_N_COL;
    hexkl_micro_hmx_acc_clear_int32();

    /* Both operands are already in VTCM: no conversion, no DDR read. */
    for (uint32_t kt = 0; kt < k_tiles; kt++)
      hexkl_micro_hmx_mm_u8i4(
        vtcm_base, HEXKL_HMX_ACTIVATION_ALIGNMENT * kt,
        weight_base + (kt * n_tiles + nt) * WT_TILE_BYTES);

    hexkl_micro_hmx_acc_read_int32(vtcm_base, hmx_config_offset, result_offset);
    hexkl_micro_hmx_copy_32b_to_submatrix(
      vtcm_base, result_offset, matA, row / HEXKL_HMX_INT8_BLOCK_N_ROW, nt,
      X_rows, X_cols);
  }
}
```

Conversions drop from `(M/64)·(N/32)·(K/32)` to `(N/32)·(K/32)` — for q_proj at
M=64 that is 2048 → 2048, i.e. unchanged for a single call, but each tile is now
converted to a distinct slot instead of one scratch tile, so **a second call on
the same weight needs none at all**. That is the case that matters: an fc_layer
is called once per token, and for `M > 64` the saving applies within a single
call too.

Two further wins available once the kernel is ours:

- `hexkl_micro_hmx_lock()` can wrap a whole forward pass rather than one matmul.
- The accumulator can stay in VTCM across output tiles and be copied out once,
  instead of per tile.

## 6. What has to hold

- ~~**VTCM is large enough**~~ — settled: §3.1 measured 8 MiB, which is what
  §4 budgeted against.
- **`weight_offset` is a free parameter.** `hexkl_micro_hmx_rm_to_wh_i4` takes
  it as an argument and `hexkl_micro_hmx_mm_u8i4` requires only 128-byte
  alignment, so laying tiles out contiguously is allowed. This is what makes
  the hoist possible at all.
- **The shipped `sdkl_npu_mm_u8i4_i32` really does re-convert per tile.**
  Unverified — §3.2 exists to test it.

`hexkl_micro_hmx_copy_psubmatrix_to_8b_weight` looks like it would skip the
conversion entirely, but it is int8-only (no int4 variant in the header) and
its own docs call it "primarily intended for testing and debugging purposes. In
operational mode, DMA is typically used." So the route for int4 is either the
per-tile convert above, or a DMA of pre-baked WH bytes — and nntrainer already
bakes those offline in `quantize_qint4_weight`.
