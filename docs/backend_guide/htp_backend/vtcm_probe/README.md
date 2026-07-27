# VTCM weight-residency probe

`hexkl_vtcm_probe.c` answers one question before any skel work is committed:
**does keeping the WH-packed weight resident in VTCM make the u8i4 matmul
faster, and by how much?**

It is a drop-in replacement for the HexKL addon's own micro-API sample, so it
builds and runs with that sample's scripts and needs nothing else — no
FastRPC, no nntrainer, no device.

> **Never built or run.** The nntrainer tree carries no Hexagon toolchain, so
> this has only been syntax-checked against stub headers. Expect to fix build
> details on first contact.

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

It also prints the VTCM size, which the budget in
[07_vtcm_weight_residency.md](../07_vtcm_weight_residency.md) assumes is ~8 MiB
without ever having confirmed it.

## Build and run

The addon sample lives at:

```
$HEXAGON_SDK_ROOT/addons/hexkl_addon/examples/hexkl_micro_hmx_mm_u8i4_i32/
├── README.md
├── build.sh
├── run_simulator.sh
└── src/test_hexkl_micro_hmx_mm_u8i4_i32.c
```

### 1. Build and run it unmodified, first

Do not skip this. If the probe fails after a straight swap, this is what tells
you whether the toolchain or the probe is at fault.

```bash
source $HEXAGON_SDK_ROOT/setup_sdk_env.source

export HEXKL_SDK_ROOT=$HEXAGON_SDK_ROOT/addons/hexkl_addon
cd $HEXKL_SDK_ROOT/examples/hexkl_micro_hmx_mm_u8i4_i32

./build.sh --hex-arch v79
./run_simulator.sh --hex-arch v79
```

Expect `Test Passed`. Stop here if you do not get it.

### 2. Swap in the probe

```bash
EX=$HEXKL_SDK_ROOT/examples/hexkl_micro_hmx_mm_u8i4_i32
SRC=$EX/src/test_hexkl_micro_hmx_mm_u8i4_i32.c

cp $SRC $SRC.orig                       # keep the original
cp <nntrainer>/docs/backend_guide/htp_backend/vtcm_probe/hexkl_vtcm_probe.c $SRC

./build.sh --hex-arch v79
./run_simulator.sh --hex-arch v79
```

`cp $SRC.orig $SRC` puts it back.

The probe keeps the sample's includes and `main()` shape deliberately, so the
build file should need no edits. If `build.sh` compiles by filename rather than
by wildcard, the swap above is what keeps that working.

### Knobs

Pass through whatever `build.sh` uses for extra defines (`CFLAGS`, or edit its
compile line):

| define | effect |
| :--- | :--- |
| `-DPROBE_SHAPES=N` | run only the first N of the 5 shapes |
| `-DPROBE_VERIFY_ALL` | run the scalar reference on the large shapes too |
| `-DUSE_HAP_PERF` | time with `HAP_perf_get_pcycles()` instead of the simulator timer |

## This is a simulator, which cuts both ways

- **Deterministic.** No thermal throttling, no scheduler, no other tenants —
  exactly the jitter that made the host-side threading numbers unreadable. The
  `baseline / resident_2nd` ratio comes out clean.
- **Absolute numbers are not device numbers.** Do not compare the printed
  pcycles against the 298 µs measured on the S25 Ultra. Read the ratio.
- **Slow, and cycle-accurate about it.** The scalar C reference is the
  expensive part: q_proj alone is 134 M multiply-accumulates. It is therefore
  **off by default for `medium`, `q_proj` and `ffn_up`** — those shapes only
  check that the two kernels agree with each other, having been proven against
  the reference at `sample` and `small`. If a run still drags, use
  `-DPROBE_SHAPES=3`. Output is flushed per shape, so stopping early still
  leaves usable results.

## Reading the output

```
[VTCM PROBE] VTCM base = 0x...  size = 8388608 bytes (8.00 MiB)
[q_proj     ] M=64 N=2048 K=1024 | tiles k=32 n=64 | VTCM need=1638400 (act=65536 wt=1048576) avail=...
  [OK] baseline and resident agree (reference skipped)
  pcycles  baseline=...  load=...  resident_1st=...  resident_2nd=...
  baseline / resident_2nd = ...x
```

| observation | meaning | next step |
| :--- | :--- | :--- |
| `resident_2nd` ≪ `baseline` | the per-tile conversion was the cost | build the skel — §3.3 of [07](../07_vtcm_weight_residency.md) |
| `resident_2nd` ≈ `baseline` | the shipped `sdkl_npu_mm_u8i4_i32` already avoids this; the 217 µs is elsewhere | stop, and re-open with the `--sweep` data |
| `DOES NOT FIT` on q_proj | VTCM is smaller than the budget assumed | re-plan around the printed size before anything else |
| a `[FAIL]` line | the kernel is wrong, times are meaningless | fix before reading any timing |

The two numbers worth reporting back are the **VTCM size** and the
**`baseline / resident_2nd` ratio at q_proj**. On device the host sees ~298 µs
per `sdkl_npu_mm_u8i4_i32` against QNN's 69.9 µs, so there is roughly a 4x gap
to close — a ratio well under that means residency alone will not close it.

## If it does not fit

`plan_vtcm` lays VTCM out as activations, then every weight tile, then the
accumulator, with the HMX config at the top. If q_proj does not fit, the
weight is what to give up first: hold a slice of the output columns (`n_tiles`)
at a time and loop over slices, so resident bytes scale with the slice instead
of the whole weight.
