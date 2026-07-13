# HTP E2E Performance Results

Template for recording on-device end-to-end inference performance when running the HTP backend.

## Measurement Setup

- **Device:** Galaxy S25 Ultra (SM-S938N, Snapdragon 8 Elite / V79 HTP)
- **App:** `nntrainer_causallm` (see [02. Build and Run](02_build_and_run.md))
- **Method:** run the prompt via `run_causallm.sh` (§6 of [02](02_build_and_run.md)) and record the prefill/generation timing and TPS reported on stdout.

## E2E Results

| HTP Version | Input Prompt | Prefill Time (ms) | Prefill TPS | Generation Time (ms) | Generation TPS | E2E Time (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| | | | | | | |

"HTP Version" is the commit hash or build tag of the `libnntrainer.so` under test. Add one row per run.
