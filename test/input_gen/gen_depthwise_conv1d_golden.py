#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate golden test data for depthwise conv1d layer.

Tensor layout used by this DepthwiseConv1D layer:

Input  layout (nntrainer): (batch, 1, H, W)
  - H: sequence length
  - W: logical channels

Weight layout (nntrainer): (1, 1, K, W)
  - K: kernel size along H
  - W: per-channel independent kernel values

Output layout (nntrainer): (batch, 1, out_H, W)

Golden file format:
  For each tensor: [uint32 num_elements] [float32 data...]
  Order: initial_weights, inputs, outputs, gradients, weights, derivatives
"""

import numpy as np
import struct
import os

SEED = 1234
np.random.seed(SEED)


def write_tensor(f, data):
    """Write a tensor to file: uint32 size followed by float32 data."""
    flat = data.flatten().astype(np.float32)
    f.write(struct.pack('I', len(flat)))
    flat.tofile(f)


def depthwise_conv1d_forward(input_data, weight, bias, stride, padding,
                             dilation):
    """
    Compute depthwise conv1d forward pass.

    Args:
        input_data: (batch, seq_len, channels)
        weight: (kernel_size, channels)
        bias: (channels,) or None
        stride: int
        padding: (pad_left, pad_right)
        dilation: int

    Returns:
        output: (batch, out_seq_len, channels)
    """
    batch, in_height, channels = input_data.shape
    kernel_size = weight.shape[0]
    pad_left, pad_right = padding

    eff_k = (kernel_size - 1) * dilation + 1
    padded_height = in_height + pad_left + pad_right
    out_height = (padded_height - eff_k) // stride + 1

    output = np.zeros((batch, out_height, channels), dtype=np.float32)

    for b in range(batch):
        for oh in range(out_height):
            base_h = oh * stride - pad_left
            for c in range(channels):
                val = 0.0
                for k in range(kernel_size):
                    ih = base_h + k * dilation
                    if 0 <= ih < in_height:
                        val += input_data[b, ih, c] * weight[k, c]
                output[b, oh, c] = val

    if bias is not None:
        output += bias[np.newaxis, np.newaxis, :]

    return output


def depthwise_conv1d_calc_derivative(incoming_deriv, weight, stride, padding,
                                     dilation, in_height):
    """
    Compute input derivative (backprop through depthwise conv1d).

    Args:
        incoming_deriv: (batch, out_height, channels)
        weight: (kernel_size, channels)
    Returns:
        input_deriv: (batch, in_height, channels)
    """
    batch, out_height, channels = incoming_deriv.shape
    kernel_size = weight.shape[0]
    pad_left, _ = padding

    input_deriv = np.zeros((batch, in_height, channels), dtype=np.float32)

    for b in range(batch):
        for oh in range(out_height):
            base_h = oh * stride - pad_left
            for c in range(channels):
                grad_out = incoming_deriv[b, oh, c]
                for k in range(kernel_size):
                    ih = base_h + k * dilation
                    if 0 <= ih < in_height:
                        input_deriv[b, ih, c] += grad_out * weight[k, c]

    return input_deriv


def depthwise_conv1d_calc_gradient(incoming_deriv, input_data, kernel_size,
                                   stride, padding, dilation):
    """
    Compute weight and bias gradients.

    Args:
        incoming_deriv: (batch, out_height, channels)
        input_data: (batch, in_height, channels)

    Returns:
        weight_grad: (kernel_size, channels)
        bias_grad: (channels,)
    """
    batch, out_height, channels = incoming_deriv.shape
    in_height = input_data.shape[1]
    pad_left, _ = padding

    weight_grad = np.zeros((kernel_size, channels), dtype=np.float32)

    for b in range(batch):
        for oh in range(out_height):
            base_h = oh * stride - pad_left
            for c in range(channels):
                grad_out = incoming_deriv[b, oh, c]
                for k in range(kernel_size):
                    ih = base_h + k * dilation
                    if 0 <= ih < in_height:
                        weight_grad[k, c] += grad_out * input_data[b, ih, c]

    # bias gradient: sum over batch and sequence axis
    bias_grad = incoming_deriv.sum(axis=(0, 1))

    return weight_grad, bias_grad


def compute_padding_same(in_height, kernel_size, stride, dilation):
    """Compute 'same' padding (output_height = ceil(in_height / stride))."""
    eff_k = (kernel_size - 1) * dilation + 1
    out_height = (in_height + stride - 1) // stride
    total_pad = max(0, (out_height - 1) * stride + eff_k - in_height)
    pad_left = total_pad // 2
    pad_right = total_pad - pad_left
    return (pad_left, pad_right)


def compute_padding_causal(kernel_size, dilation):
    """Compute causal padding (only left padding)."""
    eff_k = (kernel_size - 1) * dilation + 1
    return (eff_k - 1, 0)


def gen_rand_input(shape):
    return np.random.uniform(-1, 1, shape).astype(np.float32)


def generate_golden(name, batch, seq_len, channels, kernel_size, stride=1,
                    padding_mode="valid", dilation=1, disable_bias=False,
                    output_dir="."):
    """Generate a single golden test file."""

    if padding_mode == "valid":
        padding = (0, 0)
    elif padding_mode == "same":
        padding = compute_padding_same(seq_len, kernel_size, stride, dilation)
    elif padding_mode == "causal":
        padding = compute_padding_causal(kernel_size, dilation)
    elif isinstance(padding_mode, tuple):
        padding = padding_mode
    else:
        raise ValueError(f"Unknown padding mode: {padding_mode}")

    # Internal compute layout: (batch, H, W)
    input_3d = gen_rand_input((batch, seq_len, channels))

    # Weight layout for compute: (K, W)
    weight = gen_rand_input((kernel_size, channels))

    # Bias: (W,)
    bias = gen_rand_input((channels,)) if not disable_bias else None

    initial_weight = weight.copy()
    initial_bias = bias.copy() if bias is not None else None

    output_3d = depthwise_conv1d_forward(input_3d, weight, bias, stride,
                                         padding, dilation)

    incoming_deriv = np.full_like(output_3d, 2.0)

    weight_grad, bias_grad = depthwise_conv1d_calc_gradient(
        incoming_deriv, input_3d, kernel_size, stride, padding, dilation)

    input_deriv = depthwise_conv1d_calc_derivative(
        incoming_deriv, weight, stride, padding, dilation, seq_len)

    # Convert to nntrainer layout: (B, 1, H, W)
    input_4d = input_3d.reshape(batch, 1, seq_len, channels)
    out_height = output_3d.shape[1]
    output_4d = output_3d.reshape(batch, 1, out_height, channels)
    input_deriv_4d = input_deriv.reshape(batch, 1, seq_len, channels)

    # Weight to nntrainer layout: (1, 1, K, W)
    weight_4d = weight.reshape(1, 1, kernel_size, channels)
    initial_weight_4d = initial_weight.reshape(1, 1, kernel_size, channels)
    weight_grad_4d = weight_grad.reshape(1, 1, kernel_size, channels)

    # Bias to nntrainer layout: (1, 1, 1, W)
    bias_4d = bias.reshape(1, 1, 1, channels) if bias is not None else None
    initial_bias_4d = (
        initial_bias.reshape(1, 1, 1, channels)
        if initial_bias is not None else None
    )
    bias_grad_4d = (
        bias_grad.reshape(1, 1, 1, channels)
        if bias is not None else None
    )

    filepath = os.path.join(output_dir, name + ".nnlayergolden")
    with open(filepath, "wb") as f:
        # 1. initial_weights
        write_tensor(f, initial_weight_4d)
        if initial_bias_4d is not None:
            write_tensor(f, initial_bias_4d)

        # 2. inputs
        write_tensor(f, input_4d)

        # 3. outputs
        write_tensor(f, output_4d)

        # 4. gradients
        write_tensor(f, weight_grad_4d)
        if bias_grad_4d is not None:
            write_tensor(f, bias_grad_4d)

        # 5. weights
        write_tensor(f, weight_4d)
        if bias_4d is not None:
            write_tensor(f, bias_4d)

        # 6. derivatives
        write_tensor(f, input_deriv_4d)

    print(f"Generated: {filepath}")
    print(f"  input:  ({batch}, 1, {seq_len}, {channels})")
    print(f"  weight: (1, 1, {kernel_size}, {channels})")
    print(f"  output: ({batch}, 1, {out_height}, {channels})")
    print(f"  padding: {padding}, stride: {stride}, dilation: {dilation}")


def main():
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(output_dir, exist_ok=True)

    # Basic tests - single batch
    generate_golden("depthwise_conv1d_sb_minimum", batch=1, seq_len=8,
                    channels=3, kernel_size=3, output_dir=output_dir)

    # Multi-batch
    generate_golden("depthwise_conv1d_mb_minimum", batch=3, seq_len=8,
                    channels=3, kernel_size=3, output_dir=output_dir)

    # Same padding
    generate_golden("depthwise_conv1d_sb_same", batch=1, seq_len=8,
                    channels=4, kernel_size=3, padding_mode="same",
                    output_dir=output_dir)

    generate_golden("depthwise_conv1d_mb_same", batch=3, seq_len=8,
                    channels=4, kernel_size=3, padding_mode="same",
                    output_dir=output_dir)

    # Stride
    generate_golden("depthwise_conv1d_sb_stride", batch=1, seq_len=8,
                    channels=3, kernel_size=3, stride=2, output_dir=output_dir)

    generate_golden("depthwise_conv1d_mb_stride", batch=3, seq_len=8,
                    channels=3, kernel_size=3, stride=2, output_dir=output_dir)

    # Dilation
    generate_golden("depthwise_conv1d_sb_dilation", batch=1, seq_len=11,
                    channels=3, kernel_size=3, dilation=2,
                    output_dir=output_dir)

    generate_golden("depthwise_conv1d_mb_dilation", batch=3, seq_len=11,
                    channels=3, kernel_size=3, dilation=2,
                    output_dir=output_dir)

    # Causal padding
    generate_golden("depthwise_conv1d_sb_causal", batch=1, seq_len=8,
                    channels=4, kernel_size=4, padding_mode="causal",
                    output_dir=output_dir)

    generate_golden("depthwise_conv1d_mb_causal", batch=3, seq_len=8,
                    channels=4, kernel_size=4, padding_mode="causal",
                    output_dir=output_dir)

    # No bias
    generate_golden("depthwise_conv1d_sb_no_bias", batch=1, seq_len=8,
                    channels=3, kernel_size=3, disable_bias=True,
                    output_dir=output_dir)

    generate_golden("depthwise_conv1d_mb_no_bias", batch=3, seq_len=8,
                    channels=3, kernel_size=3, disable_bias=True,
                    output_dir=output_dir)

    # FP16
    generate_golden("depthwise_conv1d_sb_fp16_causal_w3", batch=1, seq_len=8,
                    channels=4, kernel_size=3, disable_bias=True,
                    output_dir=output_dir)

    generate_golden("depthwise_conv1d_mb_fp16_causal_w3", batch=3, seq_len=8,
                    channels=4, kernel_size=3, disable_bias=True,
                    output_dir=output_dir)

if __name__ == "__main__":
    main()