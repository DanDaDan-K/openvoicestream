#!/usr/bin/env python3
"""Make Kokoro length-regulator Range static for TensorRT experiments.

The sherpa Kokoro ONNX expands token-level features to frame-level features via
duration -> CumSum -> Range(total_frames) -> mask -> MatMul. TensorRT 10.3 does
not allow the data-dependent CumSum/Gather result to drive Range/shape tensors.

This experimental rewrite replaces `/encoder/Range` with a constant
`0..max_frames-1` vector. The model is no longer output-equivalent by itself
because downstream tensors become fixed-length; keep the original total frame
count as a boundary if a CPU suffix needs to trim.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def rewrite(model: onnx.ModelProto, max_frames: int) -> int:
    graph = model.graph
    rewritten = 0
    const_name = "kokoro_static_frame_range"
    graph.initializer.extend([numpy_helper.from_array(np.arange(max_frames, dtype=np.int64), const_name)])
    for node in graph.node:
        if node.name == "/encoder/Range" and node.op_type == "Range":
            del node.input[:]
            del node.output[:]
            node.op_type = "Identity"
            node.name = "/encoder/Range_static"
            node.input.extend([const_name])
            node.output.extend(["/encoder/Range_output_0"])
            rewritten += 1
    return rewritten


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-frames", type=int, default=2048)
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    model = onnx.load(Path(args.input))
    count = rewrite(model, args.max_frames)
    if count != 1:
        raise RuntimeError(f"Expected to rewrite exactly one /encoder/Range, got {count}")
    if args.infer:
        model = onnx.shape_inference.infer_shapes(model)
    if args.check:
        onnx.checker.check_model(model)
    onnx.save(model, Path(args.output))
    print(f"rewritten_range={count}")
    print(f"max_frames={args.max_frames}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
