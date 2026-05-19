#!/usr/bin/env python3
"""List ONNX nodes in a compact form."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--op-type", default="")
    ap.add_argument("--name-contains", default="")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    model = onnx.load(Path(args.model))
    count = 0
    for idx, node in enumerate(model.graph.node):
        if args.op_type and node.op_type != args.op_type:
            continue
        if args.name_contains and args.name_contains not in node.name:
            continue
        print(f"{idx}\t{node.op_type}\t{node.name}\t{list(node.output)}")
        count += 1
        if count >= args.limit:
            break
    print(f"matches_printed={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
