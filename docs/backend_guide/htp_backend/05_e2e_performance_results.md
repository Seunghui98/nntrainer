# HTP E2E Performance Results

Template for recording on-device end-to-end inference performance when running the HTP backend.

## Measurement Setup

- **Device:** Galaxy S25 Ultra (SM-S938N, Snapdragon 8 Elite / V79 HTP)
- **App:** `nntrainer_causallm` (see [02. Build and Run](02_build_and_run.md))
- **Method:** run the prompt via `run_causallm.sh` (§6 of [02](02_build_and_run.md)) and record the prefill/generation timing and TPS reported on stdout.

## Example Prompts (32 / 512 / 1024 tokens)

[`e2e_prompts/`](e2e_prompts/) ships three example prompts with an exact qwen3-0.6b tokenizer count (`gen_prompts.py` regenerates them): [`prompt_32.txt`](e2e_prompts/prompt_32.txt), [`prompt_512.txt`](e2e_prompts/prompt_512.txt), [`prompt_1024.txt`](e2e_prompts/prompt_1024.txt).

Push a file and run it on-device:

```bash
adb push docs/backend_guide/htp_backend/e2e_prompts/prompt_512.txt /data/local/tmp/nntrainer/causallm/

adb shell \
  "/data/local/tmp/nntrainer/causallm/run_causallm.sh \
   /data/local/tmp/nntrainer/causallm/models/qwen3-0.6b \
   \"\$(cat /data/local/tmp/nntrainer/causallm/prompt_512.txt)\""
```

Repeat with `prompt_32.txt` / `prompt_1024.txt` for the other prefill lengths. For a synthetic (non-text) sweep across arbitrary prompt-token counts, use `Applications/CausalLM/benchmarks/benchmark_android.py -p 32,512,1024 -n 0` instead (see its README).

## E2E Results

| Model | Kernel | Input Len (tok) | Output Len (tok) | Prefill Time (ms) | Prefill TPS | Generation Time (ms) | Generation TPS | E2E Time (ms) | Peak Mem (KB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-0.6b | f32f16_f16 | 32 | 32 | 108 | 296.296 | 11424 | 2.80112 | 12829 | 2521764 |
| Qwen3-0.6b | f32f16_f16 | 512 | 32 | 854 | 599.532 | 12140 | 2.63591 | 14242 | 2565204 |
| Qwen3-0.6b | f32f16_f16 | 1024 | 32 | 2297 | 445.799 | 15543 | 2.0588 | 19142 | 2659980 |

"HTP Version" is the commit hash or build tag of the `libnntrainer.so` under test. Add one row per run.

- **Model:** qwen3-0.6b (WH-baked FP16, `nntr_qwen3_0.6b_fp16_wh.bin`).
- **`E2E Time`** is the app's `[e2e time]` (includes model load/setup); the LLM summary's `total` (prefill + generation only) was **50262 ms**. Peak memory ≈ **2.53 GB**.
- Row `d7deba9d`, 2026-07-13, Galaxy S25 Ultra (ADB `R3CY205ZMND`, V79 skel).

## u8i4 (QINT4_HTP) Results

Model: qwen3-0.6b quantized to **u8i4** FC weights (`nntr_quantize --fc_dtype QINT4_HTP`, embedding/LM-head kept FP32). Emitted config `model_tensor_type: QINT4_HTP-FP32`, `compute_engine: htp`. UINT8 activation × INT4 weight on HMX; only the FC projections (wq/wk/wv/wo, ffn_*) run on the NPU.

Prefill throughput (input-length sweep, prefill only):

| Input Len (tok) | Prefill Time (ms) | Prefill TPS |
| :---: | :---: | :---: |
| 64 | 554 | 115.523 |
| 128 | 966 | 132.505 |
| 512 | 1732 | 295.612 |
| 1024 | 3269 | 313.246 |

Generation + memory (512-token generation run):

| Kernel | Output Len (tok) | Generation TPS | E2E Time (ms) | Peak Mem (KB) |
| :---: | :---: | :---: | :---: | :---: |
| u8i4 | 512 | 1.96321 | 264836 | 1451140 |

- **Peak memory ≈ 1.45 GB** vs f32f16 ≈ 2.53 GB → **~42% reduction** from INT4 FC weights.
- Prefill throughput scales up with input length (115 → 313 TPS across 64 → 1024 tokens) as the HMX matmul amortizes per-call overhead.
- Generation (decode, M=1) is 1.96 TPS: decode is memory-bandwidth bound and currently runs the padded NPU u8i4 path; a CPU decode fallback is future work.
- `Prefill Time (ms)` derived as tokens / Prefill TPS.
- Row `3cff75f` (u8i4 kernel), 2026-07-14, Galaxy S25 Ultra (ADB `R3CY205ZMND`, V79 skel).
