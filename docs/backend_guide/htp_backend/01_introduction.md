# Introduction to the HTP Backend

The Qualcomm Hexagon Tensor Processor (HTP) backend is a dedicated execution backend that accelerates the matrix multiplication (MatMul/GEMM) operations of transformer models on-device via the Qualcomm HexKL SDK.

## 1. Motivation and Goals
- **On-device acceleration**: Reduces bandwidth consumption compared to CPU (NEON) compute, and maximizes power and compute performance by leveraging the Snapdragon-specific HMX (Hexagon Matrix eXtension) tiles.
- **Robust fallback**: Aims for an architecture that accelerates high-performance prefill computation while transparently falling back to the CPU without interruption if the NPU fails to initialize.

## 2. Supported Operations and Hardware Alignment Constraints
Due to the structural characteristics of the NPU tiles' parallel execution units, each operation below is subject to matrix dimension alignment constraints. However, the framework's internal physical-op wrapper layer provides automatic padding, which relaxes these constraints for higher-level operations.

- **FP16 GEMM**: `shgemm_f32f16_f32` (FP32 activation × FP16 weight $\rightarrow$ FP32 C)
  - **M**: $M \% 32 == 0$ (auto-padded internally, so any $M$ works from the caller's side)
  - **N**: $N \% 32 == 0$, required by SDKL's WH memory layout; violating this forces CPU fallback
  - **K**: no alignment requirement; passed through to SDKL unchecked
- **QINT8 GEMM**: `shgemm_u8i8_i32` (U8 activation × I8 weight $\rightarrow$ I32 accumulation $\rightarrow$ FP32 C)
  - **M**: $M \% 64 == 0$ (auto-padded internally, so any $M$ works from the caller's side)
  - **N**: $N \% 32 == 0$, required by SDKL's WH memory layout; violating this forces CPU fallback
  - **K**: no alignment requirement; passed through to SDKL unchecked

## 3. Document Guide
- [02. Build and Run](02_build_and_run.md): End-to-end flow for building the Android binary with the HTP backend enabled and running actual on-device qwen3-0.6b (WH-baked FP16) inference.
- [03. Backend, Memory, and Kernels](03_backend_internals.md): Architecture and lifecycle of the `HtpBackend` singleton, the NPU DMA pool allocation strategy, and the low-level matrix multiplication kernels and `WHCache` buffer management strategy.
- [04. Unittest Guide and Results](04_unittest_guide.md): Procedure for deploying cross-compiled unittest binaries to a target device (e.g. Galaxy S25 Ultra), running remote verification, and the table for recording results.
- [05. E2E Performance Results](05_e2e_performance_results.md): On-device end-to-end performance measurements for the prefill-stage matrix operations of core transformer projection layers.
- [08. Attention on HMX — Design Record](08_attention_hmx_design.md): What was decided about putting `mha_core`'s attention on HMX, the RM/AH/WH layouts, the HexKL 1.0.0-beta2 changes, and what the work is blocked on. Analysis only; not implemented.
- [09. lm_head on the NPU as u8i4](09_lmhead_u8i4_plan.md): The active plan for the last remaining CPU matmul in qwen3-0.6b — the residency gate that decides it, the model preparation, and the order of work.
- [10. Probe Runbook](10_probe_runbook.md): How to build, deploy and read the HTP probes on device, and what each outcome does to the plans in 09 and 11.
- [12. Handoff](12_handoff.md): Where this line of work stands — what is measured, what is implemented but unrun, which branches are in play, and what to do next. **Start here if you are picking this up.**
- [11. Where a Decode Token Goes](11_decode_time_budget.md): 39.3 ms of a 102 ms token is a weight-staging `memcpy`. What removes it, and what that does to the ordering in 09.
