#!/usr/bin/env python3
"""Extract an ONNX subgraph with explicit input and output tensor names."""

from __future__ import annotations

import argparse
from pathlib import Path

from onnx import utils


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-model", required=True)
    ap.add_argument("--input", action="append", required=True)
    ap.add_argument("--output", action="append", required=True)
    args = ap.parse_args()

    utils.extract_model(
        args.model,
        args.output_model,
        input_names=args.input,
        output_names=args.output,
    )
    print(f"output_model={args.output_model}")
    for name in args.input:
        print(f"input={name}")
    for name in args.output:
        print(f"output={name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
