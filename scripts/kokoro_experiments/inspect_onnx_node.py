#!/usr/bin/env python3
"""Print producers/consumers around ONNX nodes matching an op type or name."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper


def _attr_pairs(node: onnx.NodeProto) -> list[tuple[str, object]]:
    return [(attr.name, helper.get_attribute_value(attr)) for attr in node.attribute]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--op-type", default="")
    ap.add_argument("--name-contains", default="")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    model = onnx.load(Path(args.model))
    graph = model.graph
    producers = {out: idx for idx, node in enumerate(graph.node) for out in node.output if out}
    consumers: dict[str, list[tuple[int, onnx.NodeProto]]] = {}
    for idx, node in enumerate(graph.node):
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append((idx, node))
    value_infos = {
        vi.name: vi
        for vi in list(graph.input) + list(graph.value_info) + list(graph.output)
    }
    initializers = {item.name: item for item in graph.initializer}

    hits = []
    for idx, node in enumerate(graph.node):
        if args.op_type and node.op_type != args.op_type:
            continue
        if args.name_contains and args.name_contains not in node.name:
            continue
        hits.append((idx, node))

    for idx, node in hits[: args.limit]:
        print(f"NODE {idx} {node.op_type} name={node.name}")
        print(f"  inputs={list(node.input)}")
        print(f"  outputs={list(node.output)}")
        print(f"  attrs={_attr_pairs(node)}")
        for inp in node.input:
            print(f"  IN {inp}")
            if inp in initializers:
                init = initializers[inp]
                print(f"    initializer dtype={init.data_type} dims={list(init.dims)}")
            if inp in value_infos:
                print(f"    value_info={value_infos[inp]}")
            prod_idx = producers.get(inp)
            if prod_idx is not None:
                prod = graph.node[prod_idx]
                print(f"    producer {prod_idx} {prod.op_type} name={prod.name}")
                print(f"      inputs={list(prod.input)}")
                print(f"      outputs={list(prod.output)}")
                print(f"      attrs={_attr_pairs(prod)}")
        for out in node.output:
            print(f"  OUT {out}")
            for cidx, consumer in consumers.get(out, [])[: args.limit]:
                print(f"    consumer {cidx} {consumer.op_type} name={consumer.name}")
                print(f"      inputs={list(consumer.input)}")
                print(f"      outputs={list(consumer.output)}")
                print(f"      attrs={_attr_pairs(consumer)}")
    print(f"matches={len(hits)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
