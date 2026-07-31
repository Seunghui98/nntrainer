# DSP-side probes (HexKL micro API)

The probes here run **on the Hexagon DSP**, not on the ARM side, so they are
not part of `../Android.mk`. nntrainer has no Hexagon toolchain path — these
link `libhexkl_micro.a` and are built inside the HexKL addon's own examples
tree, which already ships the cross-compile scripts.

## hexkl_layout_probe.c

Reads the RM, AH and WH layouts off the hardware instead of inferring them.

A 32×32 fp16 tile is filled with `0, 1, …, 1023`, so every element carries its
own row-major index; fp16 holds integers exactly to 2048, so the round trip is
lossless. After a conversion, reading the destination back raw says where each
element went, and that is the permutation.

What it reports:

| # | question | primitive |
| :-- | :-- | :-- |
| 1 | is "flat" plain row-major? | `copy_submatrix_to_f16` |
| 2 | the RM → AH permutation | `rm_to_ah_f16` |
| 3 | does AH → RM invert it exactly? | `ah_to_rm_f16` |
| 4 | the RM → WH permutation | `rm_to_wh_f16` |
| 5 | **is the accumulator's AH the same AH?** | `acc_read_f16` |
| 6 | which axis do `tile_row` / `tile_col` select? | 64×64 matrix |

Each observed permutation is matched against four candidate formulas
(row-major, column-major, row-pair interleave, column-pair interleave) and the
match is named; when none fits, the `(r,c) → slot` map is printed instead.

**(5) is the one that decides the attention design.** If the tile
`acc_read_f16` leaves in VTCM is byte-identical to what `rm_to_ah_f16`
produces, a matmul result can feed the next matmul as an activation with no
conversion — which is what would let an attention score matrix stay in VTCM
between `Q·Kᵀ` and `P·V`.

**(6) settles `[K][N]` vs `[N][K]`.** A square tile cannot tell rows from
columns, so it asks for tile (0,1) and (1,0) of a 64×64 matrix and reports
which sub-block came back. `rm_to_wh_f16` takes `wt_cols` and no row count, so
this is where the weight convention is decided — and therefore whether K or V
is the attention operand that needs transposing.

It also prints `hmx_fp16_rate` from `hexkl_micro_hw_init`. **A zero there means
the device has no HMX fp16**, which would invalidate any `mm_f16`-based plan
before anything else is worth measuring.

### Building

The addon examples carry the toolchain setup, so the least-effort route is to
borrow one:

```bash
cd $HEXKL_ADDON_ROOT/examples
cp -r hexkl_micro_hmx_mm_f16 layout_probe
cd layout_probe
rm src/*.c
cp <nntrainer>/test/unittest/jni_htp/hexagon/hexkl_layout_probe.c src/
# point build.sh at the new source name
./build.sh
./run_simulator.sh      # or run_android.sh for on-device
```

Optional: `-DPROBE_PRINT_GRID` dumps the full 1024-entry map per layout.

### Notes

- `hexkl_micro_hw_init` takes **three** arguments in 1.0.0-beta2
  (`vtcm_base`, `vtcm_size`, `hmx_fp16_rate`). Probes written against beta1
  pass two and will not compile.
- The weight-alignment macro is spelled differently across addon revisions, so
  it is printed only when defined. Only the activation alignment is
  load-bearing here.
