#!/usr/bin/env python3
"""Rewrite ONNX STFT nodes into Conv1D DFT-basis subgraphs.

This targets TensorRT compatibility for real-valued, onesided STFT nodes with
constant frame_step, frame_length, and window inputs. The replacement computes
the same windowed DFT with a bank of real/imag convolution kernels:

    [B, L] -> Unsqueeze(axis=1) -> Conv(stride=frame_step, kernel=frame_length)
           -> Reshape([B, F, 2, frames]) -> Transpose([0, 3, 1, 2])

The resulting output layout is ONNX STFT's real-valued layout:
    [batch, frames, frequencies, 2]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _const_scalar(initializers: dict[str, onnx.TensorProto], name: str, dtype: type) -> int | float:
    if name not in initializers:
        raise ValueError(f"STFT input must be a constant initializer: {name}")
    arr = numpy_helper.to_array(initializers[name])
    if arr.shape != ():
        raise ValueError(f"STFT scalar input expected for {name}, got shape {arr.shape}")
    return dtype(arr.item())


def _const_vector(initializers: dict[str, onnx.TensorProto], name: str) -> np.ndarray:
    if name not in initializers:
        raise ValueError(f"STFT window input must be a constant initializer: {name}")
    arr = numpy_helper.to_array(initializers[name]).astype(np.float32)
    if arr.ndim != 1:
        raise ValueError(f"STFT window must be rank-1 for {name}, got shape {arr.shape}")
    return arr


def _attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(helper.get_attribute_value(attr))
    return default


def _make_basis(window: np.ndarray, frame_length: int, onesided: bool, imag_sign: float, layout: str) -> np.ndarray:
    if len(window) != frame_length:
        raise ValueError(f"window length {len(window)} != frame_length {frame_length}")
    freq_bins = frame_length // 2 + 1 if onesided else frame_length
    kernels = np.empty((freq_bins * 2, 1, frame_length), dtype=np.float32)
    n = np.arange(frame_length, dtype=np.float32)
    for freq in range(freq_bins):
        angle = 2.0 * math.pi * freq * n / frame_length
        if layout == "interleaved":
            real_idx = freq * 2
            imag_idx = freq * 2 + 1
        elif layout == "blocked":
            real_idx = freq
            imag_idx = freq_bins + freq
        else:
            raise ValueError(f"Unsupported channel layout: {layout}")
        kernels[real_idx, 0, :] = window * np.cos(angle)
        kernels[imag_idx, 0, :] = imag_sign * window * np.sin(angle)
    return kernels


def rewrite(model: onnx.ModelProto, imag_sign: float, layout: str) -> int:
    graph = model.graph
    initializers = {item.name: item for item in graph.initializer}
    new_nodes: list[onnx.NodeProto] = []
    rewritten = 0

    for node in graph.node:
        if node.op_type != "STFT":
            new_nodes.append(node)
            continue
        if len(node.input) < 4:
            raise ValueError(f"Only STFT with signal, frame_step, window, frame_length is supported: {node.name}")
        onesided = bool(_attr_int(node, "onesided", 1))
        frame_step = _const_scalar(initializers, node.input[1], int)
        window = _const_vector(initializers, node.input[2])
        frame_length = _const_scalar(initializers, node.input[3], int)
        basis = _make_basis(window, frame_length, onesided, imag_sign, layout)
        freq_bins = basis.shape[0] // 2

        prefix = (node.name or node.output[0]).replace("/", "_").strip("_")
        axes_name = f"{prefix}_convstft_unsqueeze_axes"
        weight_name = f"{prefix}_convstft_weight"
        shape_name = f"{prefix}_convstft_reshape_shape"
        unsqueezed = f"{prefix}_convstft_unsqueezed"
        conv_out = f"{prefix}_convstft_conv"
        reshaped = f"{prefix}_convstft_reshaped"

        graph.initializer.extend(
            [
                numpy_helper.from_array(np.array([1], dtype=np.int64), axes_name),
                numpy_helper.from_array(basis, weight_name),
                numpy_helper.from_array(
                    np.array(
                        [0, freq_bins, 2, -1] if layout == "interleaved" else [0, 2, freq_bins, -1],
                        dtype=np.int64,
                    ),
                    shape_name,
                ),
            ]
        )
        new_nodes.extend(
            [
                helper.make_node("Unsqueeze", [node.input[0], axes_name], [unsqueezed], name=f"{prefix}/ConvSTFT_Unsqueeze"),
                helper.make_node(
                    "Conv",
                    [unsqueezed, weight_name],
                    [conv_out],
                    name=f"{prefix}/ConvSTFT_Conv",
                    strides=[frame_step],
                ),
                helper.make_node(
                    "Reshape",
                    [conv_out, shape_name],
                    [reshaped],
                    name=f"{prefix}/ConvSTFT_Reshape",
                    allowzero=0,
                ),
                helper.make_node(
                    "Transpose",
                    [reshaped],
                    list(node.output),
                    name=f"{prefix}/ConvSTFT_Transpose",
                    perm=[0, 3, 1, 2] if layout == "interleaved" else [0, 3, 2, 1],
                ),
            ]
        )
        rewritten += 1

    del graph.node[:]
    graph.node.extend(new_nodes)
    return rewritten


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--imag-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    ap.add_argument("--channel-layout", choices=("interleaved", "blocked"), default="interleaved")
    args = ap.parse_args()

    model = onnx.load(Path(args.input))
    count = rewrite(model, args.imag_sign, args.channel_layout)
    if count == 0:
        raise RuntimeError("No STFT nodes were rewritten")
    if args.check:
        onnx.checker.check_model(model)
    onnx.save(model, Path(args.output))
    print(f"rewritten_stft={count}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
