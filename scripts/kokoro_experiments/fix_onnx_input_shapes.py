#!/usr/bin/env python3
"""Set selected ONNX graph input dimensions to fixed values."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def _set_shape(value_info: onnx.ValueInfoProto, dims: list[int]) -> None:
    shape = value_info.type.tensor_type.shape
    del shape.dim[:]
    for dim in dims:
        shape.dim.add().dim_value = int(dim)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--shape", action="append", required=True, help="name:dimxdim, e.g. tokens:1x64")
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    model = onnx.load(Path(args.input))
    graph_inputs = {item.name: item for item in model.graph.input}
    for spec in args.shape:
        name, raw_dims = spec.split(":", 1)
        if name not in graph_inputs:
            raise ValueError(f"Input not found: {name}")
        dims = [int(item) for item in raw_dims.lower().split("x")]
        _set_shape(graph_inputs[name], dims)
    if args.infer:
        model = onnx.shape_inference.infer_shapes(model)
    if args.check:
        onnx.checker.check_model(model)
    onnx.save(model, Path(args.output))
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
