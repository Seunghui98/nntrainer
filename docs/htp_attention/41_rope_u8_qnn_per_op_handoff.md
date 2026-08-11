# QNN uint8 RoPE per-op quantization handoff

## Purpose

This document is the handoff for extending the current HVX uint8 RoPE path so
that it reproduces QNN intermediate `mul`/`add` quantization, not only the final
uint8 output. It is intended for another AI agent to review and turn into an
implementation plan.

## Branch and baseline

Review/plan branch:

```
claude/hvx-rope-qnn-per-op-plan
```

Implementation baseline:

```
f21d0ad42 [HVX] Add uint8 RoPE kernel and tests
339041c56 [HVX] Add cached QNN uint8 RoPE tables
```

Read these files first:

```
docs/htp_attention/40_rope_u8_task.md
nntrainer/tensor/htp_backend/hvx/hvx_rope_u8.c
nntrainer/tensor/htp_backend/hvx/hvx_rope_u8.h
test/htp/nntr_hvx_rope.c
test/htp/nntr_hvx.idl
test/htp/nntr_hvx_session.h
test/unittest/unittest_hvx_rope.cpp
test/unittest/mha_htp_host_model.cpp
```

Trace:

```
/home/leeseunghui/workspace/hxkl-beta2/Qwen3-0.6B_attn_quant_w8_1024_chromeTrace_opTrace.json
```

Reference implementation:

```
/home/leeseunghui/workspace/llama.cpp/ggml/src/ggml-hexagon
```

## Current implementation

The old endpoint is:

```
rope_u8(x, s_in, zp_in, cos_q15, sin_q15, s_out, zp_out, y)
```

- input/output activations are uint8;
- cos/sin are Q15 int16 half-tables;
- HVX performs the rotation arithmetic with f32 vectors;
- only the final result is requantized to uint8.

The cache endpoint is:

```
rope_cache_init(n_positions, dim, theta, generation)
rope_u8_cached(n_rows, width, dim, position_start, ...)
rope_cache_clear()
```

- cache storage is QNN-style uint8;
- current table quantization is scale `1/127`, zero point `128`;
- sinf/cosf are evaluated once during cache initialization;
- an internal Q15 view is created once and reused by the current HVX kernel;
- repeated initialization with the same `(n_positions, dim, theta)` is a cache hit.

## Facts observed in the QNN trace

The relevant-looking `q::mul_op` and `q::Add.tcm` operations are
`ElementWiseBinary`. Their representative tensors have:

```
Data Type  : QUInt8
Shape      : [1, 1, 1024, 32] (representative)
Step Size  : 0.00784314 ~= 1/127
Zero Offset: 128
```

`QUInt8` proves the graph tensor storage/boundary type. It does not, by itself,
prove that the HTP implementation uses pure integer arithmetic internally.
The complete per-op input/output scales, zero points, broadcasts, and order
must be extracted by following tensor IDs, or obtained from a QNN graph/tensor
dump. Do not infer missing values from dtype alone.

## Gap between current fused HVX and QNN graph

Current fused path:

```
uint8 input -> zero-point centering -> HVX f32 rotation -> final uint8 requant
```

Target QNN-like path:

```
uint8 q0/q1 and uint8 cos/sin
  -> quantized mul -> uint8 intermediate
  -> quantized mul -> uint8 intermediate
  -> quantized add/sub -> uint8 output
```

The current implementation therefore matches the uint8 table and final output
contract, but does not yet reproduce QNN intermediate `mul`/`add` requantization
bit-for-bit.

## Required implementation work

### 1. Extract the QNN operator contract

Build a small trace/graph metadata extractor. For every RoPE operator, record:

```
operator name/type
execution order
input tensor IDs
output tensor ID
shape and broadcast dimensions
dtype
scale/step size
zero point/zero offset
```

If the trace lacks a field, mark it unresolved and obtain a graph or tensor
dump before implementing the arithmetic.

### 2. Add a CPU QNN per-op reference

Keep the existing `rope_u8_ref()` and add a separate
`rope_u8_qnn_ref()` that can return intermediate stages.

Quantized multiply contract:

```
a_real = (qa - za) * sa
b_real = (qb - zb) * sb
q_product = RNE(a_real * b_real / s_product) + z_product
q_product = clamp(q_product, 0, 255)
```

Quantized add/sub contract:

```
a_real = (qa - za) * sa
b_real = (qb - zb) * sb
q_sum = RNE((a_real +/- b_real) / s_out) + z_out
q_sum = clamp(q_sum, 0, 255)
```

Confirm whether QNN uses round-to-nearest-even. Make the rounding mode
explicit; do not silently mix `round`, `nearbyint`, and `nearbyintf`.

### 3. Define the complete cache key

The cache key currently covers only positions, dimension, and theta. Decide
whether it must also include:

```
max_position
head_dim
rotary_dim
partial_rotary_factor
rope_scaling_type
rope_scaling_factor
YaRN parameters
position offset policy
```

The table contract remains:

```
dtype      : uint8
scale      : QNN metadata
zero_point : QNN metadata
layout     : [position][rotary_dim/2]
```

### 4. Implement a uint8-domain HVX kernel

Keep the Q15 compatibility path separate. Add helpers such as:

```c
hvx_qnn_mul_u8(...)
hvx_qnn_add_u8(...)
hvx_qnn_requant_i32_to_u8(...)
hvx_rope_u8_qnn_rows(...)
```

Choose the arithmetic only after the QNN contract is known:

1. int64 reference plus HVX requantization;
2. int32/int64 fixed-point multiplier and shift;
3. f32 emulation that still rounds/clamps to uint8 after every QNN op.

Accuracy must be fixed before performance tuning. Precompute scale ratios or
fixed-point multipliers during cache/init rather than recomputing them for every
element.

### 5. Extend FastRPC without breaking the old endpoint

Add separate APIs, for example:

```
rope_u8_qnn_cache_init(...)
rope_u8_qnn_cached(...)
rope_u8_qnn_cache_clear(...)
```

Debug builds should optionally return:

```
mul_q0_cos
mul_q1_sin
sub_result
mul_q0_sin
mul_q1_cos
add_result
final_output
```

### 6. Host tests

Use at least:

```
dim      : 32, 64, 128
rows     : 1, 3, 64
position : 0, 1, 2, 127, 1023
theta    : 1e4, 1e6
```

Test broadcast/per-row scales and zero points, RNE half-way cases, positive and
negative saturation, uint8 boundaries (`cos=1 -> 255`, `sin=0 -> 128`), aliasing,
cache hit/miss/reinitialize/clear, invalid ranges, and overflow validation.

### 7. Device tests

Compare intermediate stages, not only final output:

```cpp
EXPECT_EQ(cpu.stage.mul_q0_cos, dsp.stage.mul_q0_cos);
EXPECT_EQ(cpu.stage.sub_result, dsp.stage.sub_result);
EXPECT_EQ(cpu.output, dsp.output);
EXPECT_EQ(cpu.saturation, dsp.saturation);
```

Use exact comparison where the contract is integer-exact; otherwise state the
per-stage LSB tolerance explicitly.

### 8. Compare against captured QNN tensors

Dump QNN cos/sin, mul outputs, add/sub outputs, and final RoPE output. Compare:

```
QNN captured tensor
CPU QNN reference
HVX result
```

Report the first stage that differs. This is the required proof that the fused
kernel matches QNN, rather than only matching an internally defined reference.

### 9. Integrate with CausalLM MHA

After the standalone endpoint is correct, verify query/key position mapping,
prefill/decode offsets, K-cache storage timing, Q/K scale differences, partial
rotary dimension, and rope scaling/YaRN behavior.

## Accuracy baseline

An independent FP32-vs-QNN-table numerical check over positions 0..1023 and
dim=128 gave:

```
cos/sin max absolute error : 0.003937
RoPE result RMSE           : 0.2116 activation-code units
relative RMS error         : 0.2872%
```

The existing Android cache tests report:

```
rows=1 width=64  dim=64  max_lsb=0
rows=3 width=96  dim=64  max_lsb=0
rows=2 width=128 dim=128 max_lsb=0
[  PASSED  ] 5 tests.
```

Those tests prove cached HVX equals the current quantized reference. They do
not yet prove equality with every QNN intermediate tensor.

## Build and test commands

```bash
export HEXAGON_SDK_ROOT=/home/leeseunghui/workspace/Hexagon_SDK/6.4.0.2
export HEXKL_ROOT=/home/leeseunghui/workspace/hxkl-beta2/hexkl_addon
export HEXKL_SDK_VER=6.4.0.2
export ANDROID_NDK=/home/leeseunghui/workspace/android-ndk-r26d

cd /home/leeseunghui/workspace/nntrainer/test/htp
bash run_u8i4_layer_on_device.sh
```

RoPE-only rerun:

```bash
adb shell "cd /data/local/tmp/htp_u8i4_layer_test && \
  LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_rope"
```

Log:

```
/tmp/hvx_rope_device_run.log
```

## Prompt for another AI agent

```text
You are reviewing and planning the next implementation stage for nntrainer's
Hexagon HVX RoPE path.

Repository: /home/leeseunghui/workspace/nntrainer
Branch: claude/hvx-rope-qnn-per-op-plan

Read first:
  docs/htp_attention/40_rope_u8_task.md
  docs/htp_attention/41_rope_u8_qnn_per_op_handoff.md
  nntrainer/tensor/htp_backend/hvx/hvx_rope_u8.c
  test/htp/nntr_hvx_rope.c
  test/htp/nntr_hvx.idl
  test/unittest/unittest_hvx_rope.cpp

Use this local trace as evidence:
  /home/leeseunghui/workspace/hxkl-beta2/Qwen3-0.6B_attn_quant_w8_1024_chromeTrace_opTrace.json

Current code already provides uint8 input/output, a Q15 compatibility path,
QNN-style uint8 cos/sin cache (scale ~= 1/127, zero point 128), cache reuse,
and passing Android HVX tests. The missing work is QNN per-op quantization:
the current fused kernel uses HVX f32 rotation and only final requantization,
while the trace shows QUInt8 elementwise mul/add tensors.

Do not assume that QUInt8 tensor dtype proves pure integer internal arithmetic.
First extract every RoPE operator's input/output scale, zero point, shape,
broadcast, and execution order. Clearly separate facts proven by the trace from
facts that require a QNN graph or tensor dump.

Produce a concrete implementation plan, not code yet, covering:
1. trace/graph metadata extraction;
2. CPU reference with per-op requantization and RNE semantics;
3. cache key and uint8 table contract;
4. HVX kernel arithmetic and overflow strategy;
5. FastRPC IDL changes;
6. host/device tests including intermediate tensors;
7. captured-QNN comparison;
8. prefill/decode integration;
9. performance and memory risks;
10. explicit acceptance criteria and unresolved assumptions.
```

## Review completion criteria

The reviewing agent must answer:

1. What is the exact QNN RoPE operator order?
2. What are the scale/zero-point values for every intermediate tensor?
3. How will internal fixed-point versus float-assisted arithmetic be verified?
4. At which stages are CPU and HVX compared?
5. What is the complete cache key?
6. How are prefill/decode positions passed?
7. What are the exact accuracy and performance gates?

Unknown answers must be called out as blockers before implementation.

