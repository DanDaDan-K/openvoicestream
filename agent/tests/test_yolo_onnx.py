"""Tests for the torch-free YOLOE-seg segmenter (perception/yolo_onnx.py).

Two layers:
  1. Real-inference (skipped if the probe ONNX / image is absent): run the
     exported yoloe-26s-seg.onnx on bus.jpg, assert the YoloResult field
     structure + non-empty masks + that de-letterbox-padding makes the mask
     area land near the ultralytics baseline (NOT ~25% low).
  2. Synthetic post-process (always runs, no ONNX): feed hand-built
     output0/output1 tensors through ``_postprocess`` and assert conf
     filtering, mask assembly, the padding strip, and the resize-to-orig.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ovs_agent.apps.voice_rebot_arm.perception.yolo_onnx import (
    YoloOnnxSegmenter,
    YoloResult,
    _Boxes,
    _letterbox,
)

PROBE_DIR = Path("/Users/harvest/project/_scratch/yolo-onnx-probe")
ONNX_PATH = PROBE_DIR / "yoloe-26s-seg.onnx"
IMG_PATH = PROBE_DIR / "bus.jpg"
BASELINE = PROBE_DIR / "baseline.json"
CLASSES = ["person", "bus"]


# ── layer 1: real inference (opportunistic) ──────────────────────────────────
@pytest.mark.skipif(
    not (ONNX_PATH.exists() and IMG_PATH.exists()),
    reason="probe ONNX / image not present",
)
def test_real_inference_fields_and_depadding():
    import cv2

    img = cv2.imread(str(IMG_PATH))
    assert img is not None
    h0, w0 = img.shape[:2]

    seg = YoloOnnxSegmenter(
        str(ONNX_PATH), CLASSES, providers=("CPUExecutionProvider",)
    )
    results = seg.predict(img, conf=0.25)

    # ultralytics-parity result shape.
    assert isinstance(results, list) and len(results) == 1
    r = results[0]
    assert isinstance(r, YoloResult)
    assert r.orig_shape == (h0, w0)
    assert isinstance(r.names, dict) and r.names[1] == "bus"
    n = len(r.boxes)
    assert n >= 1, "expected at least one detection"

    # boxes expose ultralytics-style accessors (numpy, no .cpu()).
    b0 = r.boxes[0]
    assert b0.xyxy.shape == (1, 4)
    assert float(b0.conf[0]) > 0.25

    # masks present, shaped to ORIG image, non-empty.
    assert r.masks is not None
    assert r.masks.data.shape[0] == n
    assert r.masks.data.shape[1:] == (h0, w0)
    assert int((r.masks.data[0] > 0.5).sum()) > 0

    # de-padding check: the produced mask area must be near the ultralytics
    # baseline (which already strips letterbox padding). Without the strip,
    # area is systematically ~25% low — assert we are within 15%.
    if BASELINE.exists():
        base = json.loads(BASELINE.read_text())
        bdets = {d["name"]: d for d in base["dets"]}
        names_by_idx = {i: r.names[int(r.boxes[i].cls[0])] for i in range(n)}
        areas = {
            names_by_idx[i]: int((r.masks.data[i] > 0.5).sum()) for i in range(n)
        }
        for name, bd in bdets.items():
            if name not in areas:
                continue
            base_area = bd.get("mask_area_orig") or 0
            if base_area <= 0:
                continue
            rel = abs(areas[name] - base_area) / base_area
            assert rel < 0.15, (
                f"{name}: mask area {areas[name]} vs baseline {base_area} "
                f"(rel {rel:.1%}) — de-padding likely broken"
            )


# ── layer 2: synthetic post-process (always runs) ────────────────────────────
def _make_segmenter() -> YoloOnnxSegmenter:
    # No session created until predict(); _postprocess is callable directly.
    return YoloOnnxSegmenter("unused.onnx", CLASSES, providers=("CPUExecutionProvider",))


def test_letterbox_geometry():
    img = np.zeros((480, 640, 3), dtype=np.uint8)  # 4:3
    padded, ratio, dw, dh = _letterbox(img, 640)
    assert padded.shape == (640, 640, 3)
    assert ratio == pytest.approx(1.0)        # 640/640 width-bound
    assert dw == pytest.approx(0.0)
    assert dh == pytest.approx(80.0)          # (640-480)/2


def test_postprocess_conf_filter_and_fields():
    seg = _make_segmenter()
    net = 640
    # Two rows: one above conf, one below. Square 640x640 input → no padding.
    # det row = [x1,y1,x2,y2, conf, cls, *32coeffs]
    det = np.zeros((2, 38), dtype=np.float32)
    det[0, :6] = [100, 100, 300, 300, 0.9, 1]  # bus, kept
    det[0, 6:] = 0.0
    det[0, 6] = 5.0  # weight on proto channel 0
    det[1, :6] = [50, 50, 80, 80, 0.10, 0]     # person, filtered (conf<=0.25)

    proto = np.zeros((32, 160, 160), dtype=np.float32)
    # channel 0 positive inside a central box → mask present for det0.
    proto[0, 30:130, 30:130] = 1.0

    out = seg._postprocess(
        [det[None], proto[None]], conf=0.25, orig_shape=(640, 640),
        ratio=1.0, dw=0.0, dh=0.0, net=net,
    )
    assert isinstance(out, YoloResult)
    assert len(out.boxes) == 1                  # low-conf row filtered
    assert int(out.boxes[0].cls[0]) == 1
    assert out.masks is not None
    assert out.masks.data.shape == (1, 640, 640)
    assert int((out.masks.data[0] > 0.5).sum()) > 0


def test_postprocess_empty_when_all_below_conf():
    seg = _make_segmenter()
    det = np.zeros((1, 38), dtype=np.float32)
    det[0, :6] = [10, 10, 20, 20, 0.05, 0]
    proto = np.zeros((32, 160, 160), dtype=np.float32)
    out = seg._postprocess(
        [det[None], proto[None]], conf=0.25, orig_shape=(480, 640),
        ratio=1.0, dw=0.0, dh=0.0, net=640,
    )
    assert len(out.boxes) == 0
    assert out.masks is None


def test_postprocess_depadding_resizes_content_to_orig():
    """A 4:3 orig (480x640) letterboxes with dh=80 vertical padding. The
    post-process must strip those 80px bands before resizing to orig, so the
    output mask is exactly (480, 640) and its area scales to orig, not 640²."""
    seg = _make_segmenter()
    net = 640
    h0, w0 = 480, 640
    ratio = min(net / h0, net / w0)  # 1.0
    dw, dh = 0.0, 80.0

    det = np.zeros((1, 38), dtype=np.float32)
    # box in 640 space, inside the content region (padding band is y<80, y>560).
    det[0, :6] = [200, 200, 440, 440, 0.8, 1]
    det[0, 6] = 5.0
    proto = np.zeros((32, 160, 160), dtype=np.float32)
    # proto channel 0 positive across a central region.
    proto[0, 40:120, 40:120] = 1.0

    out = seg._postprocess(
        [det[None], proto[None]], conf=0.25, orig_shape=(h0, w0),
        ratio=ratio, dw=dw, dh=dh, net=net,
    )
    assert out.masks is not None
    # The mask is in ORIG image shape, proving the strip+resize ran.
    assert out.masks.data.shape == (1, h0, w0)
    assert int((out.masks.data[0] > 0.5).sum()) > 0
    # box mapped back to orig: x unchanged (dw=0,ratio=1), y shifted up by 80.
    bx = np.asarray(out.boxes[0].xyxy[0])
    assert bx[1] == pytest.approx(120.0, abs=1.0)   # 200 - 80
    assert bx[3] == pytest.approx(360.0, abs=1.0)   # 440 - 80


def test_boxes_container_len_and_index():
    from ovs_agent.apps.voice_rebot_arm.perception.yolo_onnx import _Box

    boxes = _Boxes([_Box([0, 0, 10, 10], 0, 0.5), _Box([1, 1, 5, 5], 1, 0.9)])
    assert len(boxes) == 2
    assert int(boxes[1].cls[0]) == 1
