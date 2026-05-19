#!/usr/bin/env python3
"""Add intermediate tensors as ONNX graph outputs for debugging."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tensor", action="append", required=True)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    model = onnx.load(Path(args.input))
    existing = {item.name for item in model.graph.output}
    infos = {
        vi.name: vi
        for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    }
    for name in args.tensor:
        if name in existing:
            continue
        vi = infos.get(name)
        if vi is None:
            vi = helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
        model.graph.output.extend([vi])
        existing.add(name)
    if args.check:
        onnx.checker.check_model(model)
    onnx.save(model, Path(args.output))
    print(f"output={args.output}")
    for name in args.tensor:
        print(f"exposed={name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
