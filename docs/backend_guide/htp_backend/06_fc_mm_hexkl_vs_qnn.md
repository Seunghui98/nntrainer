# fc_layer mm-level: HexKL vs QNN comparison

This describes how the HTP u8i4 / u8i8 HexKL kernels are compared against
Qualcomm QNN (AI Engine Direct) at the **single fully-connected matmul** level
— one `[M, K] × [K, N]` GEMM as it appears inside an fc_layer projection — not
at the whole-model level. The point of comparing at the mm level is to isolate
the matmul engine (HMX via HexKL vs QNN's HTP op) from the rest of the model.

## Matched quantization ("QNN done identically to HexKL")

Both backends receive the **same** operands with the **same** quantization, so
any difference is engine behaviour, not a different numeric recipe:

| operand | quantization |
| :--- | :--- |
| activation `A [M,K]` | per-tensor UINT8, `act_scale = max_abs(A)/127`, zero-point `128`, `X_u8 = clamp(round(A/act_scale)+128, 0, 255)` |
| weight `W [N,K]` (INT8 / u8i8) | per-output-channel symmetric INT8 `[-127,127]`, `scale[n] = max_abs(W[n,:])/127` |
| weight `W [N,K]` (INT4 / u8i4) | per-output-channel symmetric INT4 `[-7,7]`, `scale[n] = max_abs(W[n,:])/7` |
| accumulator | INT32 `C_i32[m,n] = Σ_k X_u8[m,k]·W_q[n,k]` |
| dequant | `C[m,n] = act_scale · scale[n] · (C_i32[m,n] − zp_corr[n])`, `zp_corr[n] = 128·Σ_k W_q[n,k]` |

The `zp_corr` term cancels the activation zero-point exactly:
`C_i32 − zp_corr[n] = Σ_k (X_u8−128)·W_q[n,k]`.

This is the recipe `quantize_qint4_weight` / `quantize_qint8_weight` and
`shgemm_u8i{4,8}_i32` implement. QNN is given the identical `X_u8`, `W_q`,
`scale[]` and `zp_corr[]`, so its INT4/INT8 matmul must reproduce the same
`C[m,n]`.

## What is verified where

- **Host (no NPU, no SDK):** `unittest_nntrainer_htp_kernel_math` validates the
  quantization + zp-correction + dequant math against an FP32 reference and an
  exact int32 GEMM (the NPU op is emulated on CPU). Runs in any CI. This is the
  numeric contract QNN must also satisfy.
- **Device (Hexagon + HexKL SDK):** `unittest_nntrainer_htp_kernels` runs the
  real `sdkl_npu_mm_u8i{4,8}_i32` kernels:
  - `Accuracy_u8i4_full_pipeline` / `Accuracy_u8i8_full_pipeline` — relErr vs
    FP32 at qwen3-0.6b fc shapes.
  - `FcMm_Compare_HexKL_u8i4_u8i8` — kernel latency + relErr for u8i4 and u8i8
    side by side (the HexKL half of the comparison table).
- **Device (Hexagon + HexKL SDK + QNN SDK):** the QNN column — build QNN a
  single-MatMul/FullyConnected graph fed the identical `X_u8` / `W_q` /
  `scale` / `zp_corr`, time its execute, and compare relErr + latency against
  the HexKL rows.

## Measured: q_proj (M=64, N=2048, K=1024), Galaxy S25 Ultra

One fc_layer matmul, u8i4. HexKL numbers from
`hexkl_fc_compare --proj q_proj --bits 4`; QNN from a `qnn-net-run` profile of
the same shape.

| | μs | note |
| :--- | ---: | :--- |
| **HexKL per call (steady)** | **351** | `--engine nntr`: what an fc_layer pays |
| ↳ NPU matmul | 270 | `sdkl_npu_mm_u8i4_i32`, from the kernel's own timers |
| ↳ host quantize + dequantize | 77 | NEON; scan 9.3 + quantize 23 + dequantize 45 |
| ↳ buffer staging | 0.1 | scratch pool + resident weight, both hits |
| HexKL first call | 1060 | uploads the weight; also the pre-residency cost |
| **QNN NetRun** | **513** | host-observed execute |
| QNN device total | 347 | Convert 44.5 + FC 69.9 + format 151.1 + writeback 72.9 |
| QNN FC alone | 69.9 | device-timeline sub-phase |

Measured with `HEXKL_FC_ITERS=100` on a build-stamped binary, reproduced across
four runs (351 / 362 / 351 / 360). Earlier revisions of this file quoted 495 μs
per call and 197 μs of host time; those predate the NEON quantize/dequantize.

Reading these:

- **351 vs 513** is the closest like-for-like pair — both are what the host
  pays per matmul — but it flatters us twice over, and neither is the whole
  story. See the phase-by-phase comparison below.
- **270 vs 69.9** is the compute gap, and it is real: QNN's FC sub-phase is
  ~4x cheaper. A `--sweep` puts our fixed per-call cost at ~76 μs, so roughly
  194 μs of the 270 is shape-dependent work against QNN's 69.9.
- **77 μs of host time**, down from ~197 before NEON. The three loops are now
  near memory bandwidth: dequantize moves 1 MB in 45 μs (~23 GB/s), the scan
  256 KB in 9.3 μs (~27 GB/s). Threading them was tried and reverted (see the
  table in `hexkl_mm.cpp`); SIMD was the axis that held up.

### Where QNN's 513 μs goes, and why the comparison is not clean

| | μs | what it is |
| :--- | ---: | :--- |
| Host / qnn-net-run overhead | 128 | the benchmark tool: arg parsing, reading the input from a file, tensor setup, profiling |
| Non-accelerate RPC overhead | 39 | FastRPC marshalling, domain switch, cache maintenance |
| Input slice / load (uint8) | 44.5 | input into the device's layout |
| **FC (weight load + compute)** | **69.9** | |
| Output format / layout (uint8) | 151.1 | accumulator → output tensor layout |
| Output writeback | 72.9 | into memory the host can read |
| Idle / misc | 5.2 | |

Only 13.6% of QNN's 513 μs is the matmul. Two of those rows do not belong in a
comparison against an integrated app: the 128 μs is `qnn-net-run`'s own cost,
which we pay none of, and QNN's graph takes uint8 in and returns uint8 out, so
its 513 excludes the FP32 conversion our 351 includes. Adjusting for the first
alone gives 351 vs 385.

**The per-call win does not survive scaling to a model.** QNN's overheads are
per graph execution; ours are per matmul. A qwen3-0.6b token is 196 fc_layer
calls (7 projections × 28 layers), so we pay ~76 μs of RPC and ~77 μs of
conversion 196 times — about 30 ms — where QNN pays its equivalent once. A
rough projection puts a token at ~14 ms for QNN against ~60-70 ms for us.

Which is to say: we are ahead on an isolated fc_layer and behind on the model,
and the two levers that would close it (keeping activations u8 between layers,
batching calls into one RPC) are structural rather than kernel work. See §7 of
[07_vtcm_weight_residency.md](07_vtcm_weight_residency.md).

## Fc shapes (qwen3-0.6b, hidden 1024, intermediate 3072, GQA q=2048/kv=1024)

| projection | N | K |
| :--- | :---: | :---: |
| wq (q_proj) | 2048 | 1024 |
| wk/wv/wo | 1024 | 1024 |
| ffn gate/up | 3072 | 1024 |
| ffn down | 1024 | 3072 |

`M` = 64 for a prefill tile, `M` = 1 for decode. `N % 32 == 0` and (INT4)
`K` even are required by the HMX tiling.

## Build & run

### Host numeric tests (any x86/arm host)

```bash
meson setup build -Denable-htp=false
ninja -C build test/unittest/unittest_nntrainer_htp_kernel_math
./build/test/unittest/unittest_nntrainer_htp_kernel_math
```

### Device HexKL kernel tests (Hexagon + HexKL SDK)

```bash
meson setup build_htp \
  -Denable-htp=true \
  -Dhexkl-sdk-root=<Hexagon_SDK>/addons/hexkl_addon \
  -Denable-fp16=true
ninja -C build_htp test/unittest/unittest_nntrainer_htp_kernels
# push the binary + libsdkl.so to the device and run under adb:
adb shell /data/local/tmp/unittest_nntrainer_htp_kernels \
  --gtest_filter='HtpKernelTest.FcMm_Compare_HexKL_u8i4_u8i8:HtpKernelTest.Accuracy_u8i*_full_pipeline'
```

The `FcMm_Compare_HexKL_u8i4_u8i8` test prints a markdown table of kernel
latency (ms) and relErr for each shape.

### Adding the QNN column (Hexagon + HexKL SDK + QNN SDK)

```bash
meson setup build_qnn \
  -Denable-htp=true -Dhexkl-sdk-root=<Hexagon_SDK>/addons/hexkl_addon \
  -Denable-npu=true  -Dqnn-sdk-root=<QNN_SDK>
```

Feed the QNN single-MatMul graph the identical quantized operands (`X_u8`,
`W_q`, per-channel `scale`, `zp_corr`) from the shared reference above, run its
execute over the same fc shapes, and record latency + relErr next to the HexKL
rows. Because the quantization is identical, relErr differences isolate the
engine; latency differences isolate the matmul implementation.
