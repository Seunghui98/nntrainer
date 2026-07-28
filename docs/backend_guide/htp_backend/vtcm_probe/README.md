# VTCM weight-residency probe

`hexkl_vtcm_probe.c` answers one question before any skel work is committed:
**does keeping the WH-packed weight resident in VTCM make the u8i4 matmul
faster, and by how much?**

It is a drop-in replacement for the HexKL addon's own micro-API sample, so it
builds and runs with that sample's scripts and needs nothing else — no
FastRPC, no nntrainer, no device.

> **Run, on the V79 simulator. Results and verdict below.** Short version:
> 15.3x at q_proj, and it does not help us — nntrainer bakes WH offline, so it
> never pays the cost this removes. The useful finding is the cost breakdown.

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

It also prints the VTCM size, which on a V79 simulator comes out at 8 MiB —
matching what the budget in
[07_vtcm_weight_residency.md](../07_vtcm_weight_residency.md) assumed.

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

Expect `Test Passed`, preceded by `VTCM size = 8388608 bytes`. Stop here if you
do not get it.

Two things bite on a fresh machine:

- `HEXAGON_SDK_ROOT is not set` — `setup_sdk_env.source` has to be sourced;
  exporting the variable by hand leaves the toolchain paths unset.
- `hexagon-sim: libncurses.so.5: cannot open shared object file` — Ubuntu 24.04
  does not ship ncurses 5 and has no `libncurses5` package. Symlink the `.6`
  libraries to the `.5` names and **run `ldconfig`**; the links can already
  exist while the loader cache still does not know them.

  ```bash
  sudo ln -s /lib/x86_64-linux-gnu/libncurses.so.6 /lib/x86_64-linux-gnu/libncurses.so.5
  sudo ln -s /lib/x86_64-linux-gnu/libtinfo.so.6   /lib/x86_64-linux-gnu/libtinfo.so.5
  sudo ldconfig
  ```

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

## What it actually returned

```
[VTCM PROBE] VTCM base = d9000000  size = 8388608 bytes (8.00 MiB)
[VTCM PROBE] hmx config = 16384 bytes | tile: act=2048 wt=512 acc=8192
[q_proj     ] M=64 N=2048 K=1024 | tiles k=32 n=64 | VTCM need=1122304 (act=65536 wt=1048576) avail=8372224
  [OK] baseline and resident agree (reference skipped)
  pcycles  baseline=27504216  load=25740000  resident_1st=1781076  resident_2nd=1797104
  baseline / resident_2nd = 15.30x
```


V79 simulator, pcycles. VTCM = 8388608 bytes (8 MiB).

| shape | baseline | load | resident_1st | resident_2nd | ratio |
| :--- | ---: | ---: | ---: | ---: | ---: |
| sample 64×128×128 | 344,784 | 201,264 | 143,140 | 143,280 | 2.41x |
| small 64×256×256 | 1,089,432 | 804,608 | 285,556 | 286,064 | 3.81x |
| medium 64×1024×1024 | 14,028,504 | 12,870,368 | 1,166,932 | 1,175,024 | 11.94x |
| **q_proj 64×2048×1024** | 27,504,216 | 25,740,000 | 1,781,076 | **1,797,104** | **15.30x** |
| ffn_up 64×3072×1024 | 40,979,928 | 38,609,632 | 2,395,220 | 2,419,184 | 16.94x |

Both kernels matched the reference exactly at the two verified shapes and
agreed with each other at the rest. `load + resident_1st` reproduces `baseline`
to within 0.06% everywhere, and the conversion is a flat 12,568 pcycles per
512-byte tile — so the split between conversion and matmul is real.

**The 15.3x does not transfer to nntrainer.** The sample converts row-major
weights on the DSP; `sdkl_npu_mm_u8i4_i32` is handed a weight already in WH
layout, baked once by `quantize_qint4_weight` and passed through untouched by
`shgemm_u8i4_i32`. Our per-call conversion count is zero. The probe measured a
large saving on a cost we do not pay.

### The finding that is worth something

The five shapes fit `A + B·(mm ops) + C·(output tiles) + D·(activation tiles)`
exactly, with `ffn_up` held out of the fit and predicted to the cycle:

| term | cycles | what it is |
| :--- | ---: | :--- |
| `B` | 48 | one `hexkl_micro_hmx_mm_u8i4` |
| `C` | 17,904 | `acc_read_int32` + `copy_32b_to_submatrix`, per 8 KiB output tile |
| `D` | 17,216 | `copy_submatrix_to_8b_activation`, per 2 KiB activation tile |
| `A` | 2,032 | setup |

At q_proj: HMX 98,304 (**5.5%**), output copies 1,145,856 (64%), activation
copies 550,912 (31%). With every weight already resident, the kernel still
spends 95% of its time shuffling bytes — which is what `hexkl_micro.h` warns
about when it says the copy helpers are "primarily intended for testing and
debugging purposes. In operational mode, DMA is typically used."

So the lever is DMA for staging, not where the weight lives. See §7 of
[07_vtcm_weight_residency.md](../07_vtcm_weight_residency.md).

## Reading the output on a re-run

| observation | meaning |
| :--- | :--- |
| `DOES NOT FIT` on q_proj | VTCM is smaller than 8 MiB on this target; re-plan around the printed size |
| a `[FAIL]` line | the kernel is wrong and the times mean nothing; fix before reading any timing |

## If it does not fit

`plan_vtcm` lays VTCM out as activations, then every weight tile, then the
accumulator, with the HMX config at the top. If q_proj does not fit, the
weight is what to give up first: hold a slice of the output columns (`n_tiles`)
at a time and loop over slices, so resident bytes scale with the slice instead
of the whole weight.
