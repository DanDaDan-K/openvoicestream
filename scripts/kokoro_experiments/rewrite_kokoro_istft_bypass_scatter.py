#!/usr/bin/env python3
"""Bypass Kokoro ISTFT edge ScatterND correction for TRT experiments.

The exported ISTFT subgraph performs ConvTranspose with an inverse DFT basis,
then uses NonZero/Gather/ScatterND to correct edge samples before a center
slice. This experimental rewrite turns the ScatterND node into Identity from
the ConvTranspose squeeze output. It is intended to measure TensorRT build and
runtime impact; audio equivalence must be checked before product use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


SCATTER_NAME = "/decoder/decoder/generator/istft/stft/ScatterND"
SOURCE = "/decoder/decoder/generator/istft/stft/Squeeze_1_output_0"
OUTPUT = "/decoder/decoder/generator/istft/stft/ScatterND_output_0"


def rewrite(model: onnx.ModelProto) -> int:
    count = 0
    for node in model.graph.node:
        if node.name == SCATTER_NAME and node.op_type == "ScatterND":
            del node.input[:]
            del node.output[:]
            node.op_type = "Identity"
            node.name = "/decoder/decoder/generator/istft/stft/ScatterND_bypass"
            node.input.extend([SOURCE])
            node.output.extend([OUTPUT])
            count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    model = onnx.load(Path(args.input))
    count = rewrite(model)
    if count != 1:
        raise RuntimeError(f"Expected one ScatterND rewrite, got {count}")
    if args.infer:
        model = onnx.shape_inference.infer_shapes(model)
    if args.check:
        onnx.checker.check_model(model)
    onnx.save(model, Path(args.output))
    print(f"rewritten_scatter={count}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
