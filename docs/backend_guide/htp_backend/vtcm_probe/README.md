# VTCM weight-residency probe

`hexkl_vtcm_probe.c` answers one question before any skel work is committed:
**does keeping the WH-packed weight resident in VTCM make the u8i4 matmul
faster, and by how much?**

It runs on the Hexagon DSP, standalone — no FastRPC, no nntrainer.

> **Not built or run yet.** The nntrainer tree carries no Hexagon toolchain, so
> this has never been compiled. Expect to fix build details on first contact.

## What it measures

For each shape it runs two kernels over identical data and times both in DSP
pcycles:

| kernel | weight handling |
| :--- | :--- |
| `baseline` | the SDK sample's structure — each tile re-derived from DDR inside the innermost loop, always into one 512-byte scratch slot |
| `resident` | every tile converted once into its own VTCM slot up front; the matmul then reads VTCM only |

`resident` is timed twice. The first call pays the conversion; the second
reuses what is in VTCM. **The second is the number that matters** — an fc_layer
is called once per token with the same weight, so every call after the first
looks like that one.

Both are checked against a plain C reference, so a faster wrong answer fails
rather than reading as a win.

It also prints the VTCM size, which the budget in
[07_vtcm_weight_residency.md](../07_vtcm_weight_residency.md) assumes is ~8 MiB
without ever having confirmed it.

## Build

The file is a drop-in replacement for the addon's own micro-API sample — same
includes, same `main()` shape — so the least fragile route is to build it with
that example's existing setup rather than writing a new one:

```bash
export HEXAGON_SDK_ROOT=<Hexagon_SDK>/6.4.0.2
export HEXKL_SDK_ROOT=$HEXAGON_SDK_ROOT/addons/hexkl_addon

# find the micro sample (the one with hexkl_micro_matmul_u8i4_i32 in it)
grep -rl "hexkl_micro_hw_init" $HEXKL_SDK_ROOT --include=*.c

# back up that sample, drop this file in its place, and build as the addon
# documents (hexagon-clang, V79 target)
cp docs/backend_guide/htp_backend/vtcm_probe/hexkl_vtcm_probe.c \
   <sample_dir>/<sample_name>.c
```

It needs `HAP_perf.h` for `HAP_perf_get_pcycles()`; if the sample's build does
not already pull in the SDK's `incs/`, add it.

## Run

Same way the addon runs its own micro sample — it is a DSP-side binary, so it
goes through whatever launcher the example uses, not `adb shell` directly.

## Reading the output

```
[VTCM PROBE] VTCM base = 0x...  size = 8388608 bytes (8.00 MiB)
[q_proj     ] M=64 N=2048 K=1024 | tiles k=32 n=64 | VTCM need=1638400 (act=65536 wt=1048576) avail=...
  [OK] both match the reference exactly
  pcycles  baseline=...  load=...  resident_1st=...  resident_2nd=...
  baseline / resident_2nd = ...x
```

| observation | meaning | next step |
| :--- | :--- | :--- |
| `resident_2nd` ≪ `baseline` | the per-tile conversion was the cost | build the skel — §3.3 of [07](../07_vtcm_weight_residency.md) |
| `resident_2nd` ≈ `baseline` | the shipped `sdkl_npu_mm_u8i4_i32` already avoids this; the 217 µs is elsewhere | stop, and re-open with the `--sweep` data |
| `DOES NOT FIT` on q_proj | VTCM is smaller than the budget assumed | re-plan around the printed size before anything else |
| a `[FAIL]` line | the kernel is wrong, times are meaningless | fix before reading any timing |

To convert pcycles to time, divide by the DSP clock. What matters here is the
**ratio**, which needs no clock: on device the host sees ~298 µs per
`sdkl_npu_mm_u8i4_i32` against QNN's 69.9 µs, so roughly a 4x gap to close.

## If it does not fit

`plan_vtcm` lays VTCM out as activations, then every weight tile, then the
accumulator, with the HMX config at the top. If q_proj does not fit, the
weight is what to give up first: hold a slice of the output columns (`n_tiles`)
at a time and loop over slices, so resident bytes scale with the slice instead
of the whole weight.
