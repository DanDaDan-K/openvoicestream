#!/usr/bin/env python3
"""Trace ancestor dependencies of ONNX tensors, optionally stopping at tensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", action="append", required=True)
    ap.add_argument("--stop", action="append", default=[])
    ap.add_argument("--show-nodes", action="store_true")
    args = ap.parse_args()

    model = onnx.load(Path(args.model))
    graph = model.graph
    producers = {out: idx for idx, node in enumerate(graph.node) for out in node.output if out}
    graph_inputs = {item.name for item in graph.input}
    initializers = {item.name for item in graph.initializer}
    stops = set(args.stop)
    needed_inputs: set[str] = set()
    used_nodes: set[int] = set()

    def visit_tensor(name: str) -> None:
        if not name:
            return
        if name in stops:
            needed_inputs.add(name)
            return
        if name in initializers:
            return
        idx = producers.get(name)
        if idx is None:
            if name in graph_inputs or name not in initializers:
                needed_inputs.add(name)
            return
        if idx in used_nodes:
            return
        used_nodes.add(idx)
        for inp in graph.node[idx].input:
            visit_tensor(inp)

    for out in args.output:
        visit_tensor(out)

    print(f"output_count={len(args.output)}")
    for out in args.output:
        print(f"output={out}")
    print(f"node_count={len(used_nodes)}")
    print(f"input_count={len(needed_inputs)}")
    for name in sorted(needed_inputs):
        print(f"input={name}")
    if args.show_nodes:
        for idx in sorted(used_nodes):
            node = graph.node[idx]
            print(f"node={idx} {node.op_type} {node.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
