# NPU (Hexagon HTP / HMX) decoder-block offload

Runs the matmul-heavy part of every transformer decoder layer on the
Qualcomm HTP matrix unit (HMX) as int8 x int8 -> int32 GEMMs, with **two
fastRPC round trips per layer** instead of one per projection:

```
attention_norm (rms_norm, out_quant -> uint8)
  -> qnn_graph  layer<i>_attn_in        fused q/k/v projection          [HMX]
  -> [q_norm / k_norm] -> mha_core                                       [CPU]
  -> qnn_graph  layer<i>_attn_out_ffn   o_proj + residual + ffn_norm
                                        + gate/up + SiLU*up + down
                                        + residual                       [HMX]
```

Design rules carried over from the per-channel W8A8 conv path (PR #4138):

| rule | here |
|---|---|
| per-channel int8 weight, per-tensor activation, int32 accumulate, one FP32 scale at the end | QNN `FullyConnected` with `AXIS_SCALE_OFFSET` weights; the int8 bytes are produced by the **same** quantizer as `__ggml_quantize_q8_0_per_channel` (fp16-rounded `amax/127`, round-half-away) so device output is checkable against the CPU kernel |
| affine activation with a fixed offset, no per-tensor zero point storage | static (calibrated) `scale:offset`, QNN `scaleOffsetEncoding` convention `real = (q + offset) * scale` |
| producer writes the layout the kernel wants (taps-last / identity gather) | `rms_norm out_quant` quantizes into a uint8 tensor in the same pass as gamma; no float transfer, no HTP quantize op for graph A |
| allocate once, never per forward | `QNNGraph` builds graph handles / IO tensors / rpcmem staging buffers on first use and only calls `graphExecute` afterwards |
| small work stays serial | M buckets: `_m1` for decode, `_m32/_m128` for prefill; pick `npu_layers` to leave layers on the CPU |

## 1. Build the graphs (host)

```bash
cd Applications/quick_ai/tools/npu
pip install numpy onnx onnxruntime            # onnxruntime only for selftest
python3 selftest.py                           # must print PASS

# calibration ranges (optional but strongly recommended for real models):
#   dump activations as raw fp16 rows named layer<i>_<edge>_*.raw, then
python3 calib_from_dumps.py --dir act_dumps --out calib.json

python3 build_decoder_graphs.py \
    --hf-dir  /path/to/Qwen3-0.6B \            # config.json + safetensors
    --out     npu_qwen3 \
    --buckets 1,32,128 \                        # token-count buckets
    --calib   calib.json \
    --a16     ffn_act                           # down_proj input in uint16
```

Outputs in `npu_qwen3/`:

* `onnx/layer<i>_{attn_in,attn_out_ffn}_m<M>.onnx` + `.encodings.json`
  (QDQ graphs; the encodings file carries the per-channel weight and static
  activation params for `qnn-onnx-converter --quantization_overrides`)
* `weights/layer<i>.npz` – int8 weights, scales, activation params
  (input of `verify_htp.py`)
* `npu_graphs.json` – what the runtime binds (dims, dtypes, quant params)
* `convert.sh`, `htp_backend_ext.json`, `htp_config.json`

Edges (`--calib` keys / `--a16` names): `attn_in` (attention_norm output),
`attn` (mha_core output), `ffn_in` (ffn_norm output), `ffn_act`
(SiLU(gate)*up, the down_proj input – the outlier-prone one).

## 2. Convert to context binaries (host, QNN SDK)

```bash
export QNN_SDK_ROOT=/opt/qcom/aistack/qnn-2.xx
HTP_ARCH=v79 ./npu_qwen3/convert.sh        # v75 = 8 Gen 3, v79 = 8 Elite
```

One `bin/layer<i>.bin` per layer; each carries all its graphs
(`layer<i>_attn_in_m1`, `..._m32`, `layer<i>_attn_out_ffn_m1`, ...).
`QNNGraph` discovers the `_m<N>` suffixes at runtime, so buckets can be
changed without touching the model code.

The converter flags in `convert.sh` (`--float_bitwidth 16`,
`--preserve_io datatype`, `--quantization_overrides`) are the ones that give
FP16 float IO + int8 FCs on QNN 2.2x; older/newer SDKs may spell them
differently – the script is meant to be edited.

## 3. Run on the device

`nntr_config.json`:

```json
{
  "model_tensor_type": "FP16-FP16",
  "npu_graph_dir": "/data/local/tmp/npu_qwen3",
  "npu_layers": "0-27"
}
```

* `model_tensor_type` activation half must match the graphs' float IO
  (`--fp-bits 16` -> `FP16-FP16`). A mismatch is caught at first forward
  (`QNN bytes != nntrainer rows*row_bytes`).
* `npu_layers` is optional; default = every layer in the manifest. Use it to
  bisect accuracy or to keep the first/last layers on the CPU.
* Push `npu_graphs.json` and `bin/` to the device (the ONNX/npz are not
  needed there).

Measure:

```bash
NNTR_QNN_PROFILE=1 ./nntrainer_quick_ai ...     # per layer: copy_in / exec / copy_out us
NNTR_QNN_DUMP=/data/local/tmp/dump ./nntrainer_quick_ai ...   # first 4 executes' raw IO
```

Read the profile as: `exec` = fastRPC round trip + HTP time, `copy_in/out` =
staging copies (rows x row bytes, ~1 KB per token in decode). Compare prefill
(bucket 32/128) and decode (bucket 1) separately; if decode `exec` is
dominated by the round trip, that layer belongs on the CPU (`npu_layers`).

## 4. Verify numerics against the CPU reference

```bash
python3 verify_htp.py --weights npu_qwen3/weights/layer0.npz --graph attn_in \
    --M 1 --x dump/layer0_attn_in_m1.in0.0.raw \
    --out dump/layer0_attn_in_m1.out0.0.raw --out-k ...out1.0.raw --out-v ...out2.0.raw \
    --out-dtype fp16
```

The reference accumulates in int32 with the same int8 weights, so graph A
should match to FP16 output rounding. A large `rel` means a wrong
scale/offset, a transposed weight, or a bucket/dtype mismatch – not
quantization noise.

## Files

| file | role |
|---|---|
| `npu_quant.py` | per-channel int8 weight quantizer (bit-matches the C++), affine act params, int32 GEMM reference |
| `reference.py` | NumPy reference of graphs A/B, HF safetensors loader, calibration ranges |
| `build_decoder_graphs.py` | ONNX QDQ graph builder, manifest + convert.sh emitter |
| `calib_from_dumps.py` | activation dumps -> calibration JSON (percentile clipping) |
| `selftest.py` | onnxruntime vs reference on a synthetic model |
| `verify_htp.py` | device dump vs reference |

Runtime side: `nntrainer/qnn/jni/QNNGraph.{h,cpp}` (bucketed execution,
one-time setup, profiling/dump), `Applications/quick_ai/layers/rms_norm.*`
(`out_quant`), `Applications/quick_ai/models/transformer.*` and
`qwen3_causallm.*` (`createTransformerDecoderBlockNPU`).

## Status / caveats

* The host toolchain is tested (`selftest.py`); the runtime C++ compiles
  (syntax-checked, FP16 and FP32) but the QNN layer could not be built here
  without the QNN SDK – first device run should be with `NNTR_QNN_PROFILE=1`
  and one layer (`"npu_layers": "0"`).
* Graph B takes `attn` as float (mha_core has no quantize epilogue) and
  quantizes it on the HTP; graph A takes uint8 from `rms_norm out_quant`.
* `mha_core` (QK^T / PV, KV cache) stays on the CPU by design; `lm_head` is
  not offloaded yet.
* Static activation ranges: calibrate. The built-in defaults only make the
  pipeline run.
