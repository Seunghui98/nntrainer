# HTP E2E Performance Results

Template for recording on-device end-to-end inference performance when running the HTP backend.

## Measurement Setup

- **Device:** Galaxy S25 Ultra (SM-S938N, Snapdragon 8 Elite / V79 HTP)
- **App:** `nntrainer_causallm` (see [02. Build and Run](02_build_and_run.md))
- **Method:** run the prompt via `run_causallm.sh` (§6 of [02](02_build_and_run.md)) and record the prefill/generation timing and TPS reported on stdout.

## E2E Results

| Model | Kernel | Input Len (tok) | Output Len (tok) | Prefill Time (ms) | Prefill TPS | Generation Time (ms) | Generation TPS | E2E Time (ms) | Peak Mem (KB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-0.6b(baked) | f32f16_f16 | 32 | 32 | 105 | 304.762 | 3105 | 10.366 | 4416 | 2537828 |
| Qwen3-0.6b(baked) | f32f16_f16 | 512 | 32 | 803 | 637.609 | 3255 | 9.83103 | 5327 | 2566516 |
| Qwen3-0.6b(baked) | f32f16_f16 | 1024 | 32 | 2086 | 490.892 | 3091 | 10.3526 | 6536 | 2670812 |

"HTP Version" is the commit hash or build tag of the `libnntrainer.so` under test. Add one row per run.
