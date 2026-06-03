"""Tests for the torch-free grasp pipeline (grasp_service.run_grasp_once).

All dependencies are mocked — no torch, no real camera/SDK, no onnxruntime.
We build a fake YoloResult by hand and a recording fake arm to assert:
  * full pipeline ordering (open → pregrasp → grasp_move → grasp → lift)
  * cancel_event mid-pipeline stops at the current stage and SAFE-PARKS the
    gripper (open_gripper called, no clamp left)
  * target filtering keeps only the requested class
  * no-detection / no-hand-eye early returns
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from ovs_agent.apps.voice_rebot_arm.grasp_service import run_grasp_once
from ovs_agent.apps.voice_rebot_arm.perception.yolo_onnx import (
    YoloResult,
    _Box,
    _Boxes,
    _Masks,
)


# ── fakes ─────────────────────────────────────────────────────────────────
class FakeArm:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self._holding = True

    def open_gripper(self, distance_m: float = 0.09) -> None:
        self.calls.append(("open_gripper",))

    def move_to(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0, duration=2.0) -> bool:
        self.calls.append(("move_to", round(float(z), 4)))
        return True

    def wait_motion(self, duration: float, extra: float = 0.6) -> None:
        self.calls.append(("wait_motion",))

    def get_tcp_pose(self) -> np.ndarray:
        self.calls.append(("get_tcp_pose",))
        return np.eye(4, dtype=np.float64)

    def grasp(self, force=None, timeout: float = 5.0) -> bool:
        self.calls.append(("grasp", force))
        return True

    def release_gripper(self, timeout: float = 4.0) -> None:
        self.calls.append(("release_gripper",))

    def gripper_is_holding(self) -> bool:
        return self._holding

    def names_of_calls(self) -> list[str]:
        return [c[0] for c in self.calls]


class FakeCamera:
    def __init__(self, color, depth, K) -> None:
        self._color = color
        self._depth = depth
        self.K = K
        self.warmed = 0

    def warm_up(self, n: int) -> None:
        self.warmed += n

    def get_frame(self):
        return self._color, self._depth


class FakeSegmenter:
    """Returns a fixed YoloResult; records predict() conf."""

    def __init__(self, result) -> None:
        self._result = result
        self.predict_calls = 0

    def predict(self, image_bgr, conf=0.25, iou=0.45):
        self.predict_calls += 1
        return [self._result]


def _make_result(h=480, w=640, label="banana", cls_id=0, extra=None):
    names = {0: "banana", 1: "bottle"}
    # a solid central rectangular mask → min-area-rect succeeds, depth valid.
    mask = np.zeros((h, w), dtype=np.float32)
    mask[180:300, 260:380] = 1.0
    boxes = [_Box([260, 180, 380, 300], cls_id, 0.88)]
    masks = [mask]
    if extra is not None:
        # add a distractor detection of a different class.
        m2 = np.zeros((h, w), dtype=np.float32)
        m2[50:120, 50:160] = 1.0
        boxes.append(_Box([50, 50, 160, 120], extra, 0.80))
        masks.append(m2)
    return YoloResult(
        names=names,
        boxes=_Boxes(boxes),
        masks=_Masks(np.stack(masks, axis=0)),
        orig_shape=(h, w),
    )


def _scene():
    h, w = 480, 640
    color = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), 400, dtype=np.uint16)  # 400mm everywhere
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
    return color, depth, K


# ── tests ─────────────────────────────────────────────────────────────────
def test_full_pipeline_order_and_success():
    color, depth, K = _scene()
    arm = FakeArm()
    cam = FakeCamera(color, depth, K)
    seg = FakeSegmenter(_make_result())

    res = run_grasp_once(
        "banana",
        arm=arm,
        segmenter=seg,
        camera=cam,
        K=K,
        T_hand_eye=np.eye(4),
        warm_up_frames=3,
        grasp_force=1.2,
    )

    assert res["success"] is True
    assert res["cancelled"] is False
    assert res["stage"] == "done"
    assert res["grasp_class"] == "banana"
    assert "grasp_pose" in res and len(res["grasp_pose"]) == 6
    assert cam.warmed == 3
    assert seg.predict_calls == 1

    names = arm.names_of_calls()
    # ordering: open before any move; grasp after both moves; force threaded.
    assert names.index("open_gripper") < names.index("move_to")
    assert names.count("move_to") >= 3   # pregrasp, grasp, lift
    assert "grasp" in names
    assert names.index("grasp") < len(names)
    # grasp received the configured force.
    grasp_call = next(c for c in arm.calls if c[0] == "grasp")
    assert grasp_call[1] == 1.2


def test_cancel_before_motion_safe_parks_gripper():
    color, depth, K = _scene()
    arm = FakeArm()
    cam = FakeCamera(color, depth, K)
    seg = FakeSegmenter(_make_result())

    cancel = threading.Event()
    cancel.set()  # cancelled from the start

    res = run_grasp_once(
        "banana", arm=arm, segmenter=seg, camera=cam, K=K,
        T_hand_eye=np.eye(4), cancel_event=cancel,
    )

    assert res["success"] is False
    assert res["cancelled"] is True
    # safe-park = gripper opened; NO grasp (clamp) issued.
    assert "open_gripper" in arm.names_of_calls()
    assert "grasp" not in arm.names_of_calls()


def test_cancel_midway_stops_after_pregrasp():
    color, depth, K = _scene()
    cam = FakeCamera(color, depth, K)
    seg = FakeSegmenter(_make_result())

    # Arm that trips the cancel event right after the pregrasp move, so the
    # next _check_cancel (grasp_move stage) aborts.
    cancel = threading.Event()

    class TrippingArm(FakeArm):
        def __init__(self, ev) -> None:
            super().__init__()
            self._ev = ev
            self._moves = 0

        def move_to(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0, duration=2.0) -> bool:
            self._moves += 1
            if self._moves == 1:  # pregrasp done → request stop
                self._ev.set()
            return super().move_to(x, y, z, roll, pitch, yaw, duration)

    arm = TrippingArm(cancel)
    res = run_grasp_once(
        "banana", arm=arm, segmenter=seg, camera=cam, K=K,
        T_hand_eye=np.eye(4), cancel_event=cancel,
    )

    assert res["cancelled"] is True
    assert res["stage"] == "grasp_move"
    names = arm.names_of_calls()
    assert names.count("move_to") == 1     # only the pregrasp move ran
    assert "grasp" not in names            # never clamped
    assert names[-1] == "open_gripper"     # safe-parked last


def test_target_filter_keeps_only_requested_class():
    color, depth, K = _scene()
    arm = FakeArm()
    cam = FakeCamera(color, depth, K)
    # scene has banana(0) + bottle(1); we request bottle.
    seg = FakeSegmenter(_make_result(label="banana", cls_id=0, extra=1))

    res = run_grasp_once(
        "bottle", arm=arm, segmenter=seg, camera=cam, K=K, T_hand_eye=np.eye(4)
    )
    # grasp executed and the chosen class is the bottle (filtered correctly).
    assert res.get("grasp_class") == "bottle"


def test_no_detection_returns_error():
    color, depth, K = _scene()
    arm = FakeArm()
    cam = FakeCamera(color, depth, K)
    seg = FakeSegmenter(_make_result(cls_id=0))  # only banana present

    res = run_grasp_once(
        "wrench", arm=arm, segmenter=seg, camera=cam, K=K, T_hand_eye=np.eye(4)
    )
    assert res["success"] is False
    assert res["stage"] == "detect"
    assert "no valid grasp" in res["error"]
    assert "grasp" not in arm.names_of_calls()


def test_missing_hand_eye_returns_error_before_motion():
    color, depth, K = _scene()
    arm = FakeArm()
    cam = FakeCamera(color, depth, K)
    seg = FakeSegmenter(_make_result())

    res = run_grasp_once(
        "banana", arm=arm, segmenter=seg, camera=cam, K=K, T_hand_eye=None
    )
    assert res["success"] is False
    assert res["stage"] == "transform"
    assert "hand-eye" in res["error"]
    # no arm motion issued.
    assert "move_to" not in arm.names_of_calls()


def test_missing_camera_returns_error():
    arm = FakeArm()
    seg = FakeSegmenter(_make_result())
    res = run_grasp_once("banana", arm=arm, segmenter=seg, camera=None)
    assert res["success"] is False
    assert "camera" in res["error"]
