#!/usr/bin/env python3
"""Split Kokoro ONNX into a TensorRT prefix and CPU suffix.

This is a graph-surgery helper for the sherpa Kokoro v1.0 ONNX. The full graph
does not build in TensorRT 10.3 because later generator/STFT/dynamic-shape
regions hit parser/Myelin failures. A prefix/suffix split lets us compile the
TRT-friendly front of the graph while keeping the problematic tail on CPU.

Preferred mode is --cut-output: all ancestors of that tensor become the prefix,
and the suffix receives that tensor as an explicit graph input. The validated
initial cut point is `/encoder/Cast_2_output_0`, which builds in TensorRT 10.3
and keeps the hybrid boundary to one tensor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper, utils


def _value_info_map(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    inferred = onnx.shape_inference.infer_shapes(model)
    out: dict[str, onnx.ValueInfoProto] = {}
    for vi in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        out[vi.name] = vi
    return out


def _clone_value_info(name: str, value_infos: dict[str, onnx.ValueInfoProto]) -> onnx.ValueInfoProto:
    vi = value_infos.get(name)
    if vi is not None:
        return onnx.ValueInfoProto().CopyFrom(vi) or vi
    return helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)


def _external_inputs(nodes: list[onnx.NodeProto], produced: set[str], graph_inputs: set[str]) -> set[str]:
    needed: set[str] = set()
    local_produced = set(produced)
    initializer_names = set()
    for node in nodes:
        for inp in node.input:
            if inp and inp not in local_produced:
                needed.add(inp)
        local_produced.update(o for o in node.output if o)
    return {n for n in needed if n in graph_inputs or n not in initializer_names}


def split_model(src: Path, out_dir: Path, cut_index: int) -> tuple[Path, Path, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(src)
    graph = model.graph
    nodes = list(graph.node)
    if cut_index <= 0 or cut_index >= len(nodes):
        raise ValueError(f"cut-index must be in 1..{len(nodes)-1}, got {cut_index}")

    prefix_nodes = nodes[:cut_index]
    suffix_nodes = nodes[cut_index:]
    prefix_produced = {o for n in prefix_nodes for o in n.output if o}
    suffix_inputs = {i for n in suffix_nodes for i in n.input if i}
    boundary = sorted(prefix_produced & suffix_inputs)
    if not boundary:
        raise RuntimeError(f"cut-index {cut_index} produced no boundary tensors")

    original_inputs = [i.name for i in graph.input]
    prefix_path = out_dir / f"kokoro_prefix_{cut_index}.onnx"
    utils.extract_model(str(src), str(prefix_path), original_inputs, boundary)

    value_infos = _value_info_map(model)
    graph_input_names = {i.name for i in graph.input}
    initializer_names = {i.name for i in graph.initializer}
    suffix_external = set(boundary)
    for node in suffix_nodes:
        for inp in node.input:
            if inp and inp in graph_input_names:
                suffix_external.add(inp)

    # Keep original inputs first for readability, then boundary tensors.
    suffix_inputs_vi = []
    for name in original_inputs:
        if name in suffix_external:
            suffix_inputs_vi.append(_clone_value_info(name, value_infos))
    for name in boundary:
        suffix_inputs_vi.append(_clone_value_info(name, value_infos))

    suffix_initializers = list(graph.initializer)
    suffix_outputs = [onnx.ValueInfoProto().CopyFrom(o) or o for o in graph.output]
    suffix_graph = helper.make_graph(
        suffix_nodes,
        name=f"{graph.name or 'kokoro'}_suffix_{cut_index}",
        inputs=suffix_inputs_vi,
        outputs=suffix_outputs,
        initializer=suffix_initializers,
    )
    suffix_model = helper.make_model(
        suffix_graph,
        producer_name="split_kokoro_hybrid",
        opset_imports=list(model.opset_import),
        ir_version=model.ir_version,
    )
    suffix_path = out_dir / f"kokoro_suffix_{cut_index}.onnx"
    onnx.checker.check_model(suffix_model)
    onnx.save(suffix_model, suffix_path)
    return prefix_path, suffix_path, boundary


def split_model_by_output(src: Path, out_dir: Path, cut_output: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(src)
    graph = model.graph
    nodes = list(graph.node)
    producers = {out: idx for idx, node in enumerate(nodes) for out in node.output if out}
    if cut_output not in producers:
        raise ValueError(f"cut-output is not produced by any node: {cut_output}")

    ancestors: set[int] = set()

    def visit_tensor(name: str) -> None:
        idx = producers.get(name)
        if idx is None or idx in ancestors:
            return
        ancestors.add(idx)
        for inp in nodes[idx].input:
            if inp:
                visit_tensor(inp)

    visit_tensor(cut_output)

    value_infos = _value_info_map(model)
    graph_input_names = {i.name for i in graph.input}
    original_inputs = [i.name for i in graph.input]
    suffix_nodes = [node for idx, node in enumerate(nodes) if idx not in ancestors]
    ancestor_outputs = {out for idx in ancestors for out in nodes[idx].output if out}
    suffix_consumed = {inp for node in suffix_nodes for inp in node.input if inp}
    boundary = sorted(ancestor_outputs & suffix_consumed)
    if cut_output not in boundary:
        boundary.append(cut_output)

    prefix_path = out_dir / "kokoro_prefix_encoder.onnx"
    utils.extract_model(str(src), str(prefix_path), original_inputs, boundary)

    suffix_external = set(boundary)
    for node in suffix_nodes:
        for inp in node.input:
            if inp and inp in graph_input_names:
                suffix_external.add(inp)

    suffix_inputs_vi = []
    for name in original_inputs:
        if name in suffix_external:
            suffix_inputs_vi.append(_clone_value_info(name, value_infos))
    for name in boundary:
        suffix_inputs_vi.append(_clone_value_info(name, value_infos))

    suffix_graph = helper.make_graph(
        suffix_nodes,
        name=f"{graph.name or 'kokoro'}_suffix_encoder",
        inputs=suffix_inputs_vi,
        outputs=[o for o in graph.output],
        initializer=list(graph.initializer),
    )
    suffix_model = helper.make_model(
        suffix_graph,
        producer_name="split_kokoro_hybrid",
        opset_imports=list(model.opset_import),
        ir_version=model.ir_version,
    )
    onnx.checker.check_model(suffix_model)
    suffix_path = out_dir / "kokoro_suffix_encoder.onnx"
    onnx.save(suffix_model, suffix_path)
    return prefix_path, suffix_path, boundary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cut-index", type=int, default=1321)
    ap.add_argument("--cut-output", default="")
    args = ap.parse_args()
    if args.cut_output:
        prefix, suffix, boundary = split_model_by_output(Path(args.model), Path(args.out_dir), args.cut_output)
        print(f"prefix={prefix}")
        print(f"suffix={suffix}")
        print(f"boundary_count={len(boundary)}")
        for name in boundary:
            print(f"boundary={name}")
    else:
        prefix, suffix, boundary = split_model(Path(args.model), Path(args.out_dir), args.cut_index)
        print(f"prefix={prefix}")
        print(f"suffix={suffix}")
        print(f"boundary_count={len(boundary)}")
        for name in boundary:
            print(f"boundary={name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
