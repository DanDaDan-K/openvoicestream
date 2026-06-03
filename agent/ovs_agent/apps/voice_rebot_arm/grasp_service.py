"""Torch-free vision-grasp pipeline for the reBot B601-DM voice app.

``run_grasp_once(target, *, ...)`` runs one cancellable grasp attempt:

    camera frame  →  YoloOnnxSegmenter.predict (target class only)
                  →  estimate_grasps (short-axis OBB)
                  →  select_best_grasp
                  →  T_cam2base = arm.get_tcp_pose() @ T_hand_eye
                  →  transform_grasp_pose_to_base
                  →  open → pregrasp → grasp → arm.grasp(force) → lift
                  →  (optional release)

The ``cancel_event`` (``threading.Event``) is polled before every arm motion;
on cancel the gripper is driven to a safe (open) state and the pipeline stops
at the current stage, returning ``{"success": False, "cancelled": True, ...}``.

Heavy / device-only deps (onnxruntime via the segmenter, camera SDK) are
imported lazily so this module imports on a Mac without them. There is no
torch / ultralytics anywhere in this path.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class GraspCancelled(Exception):
    """Raised internally when ``cancel_event`` fires mid-pipeline."""


def _check_cancel(cancel_event: Optional[threading.Event], arm: Any) -> None:
    """Raise :class:`GraspCancelled` (after parking the gripper safe) if the
    cancel event is set. Safe-park = open the gripper so we never leave it
    clamped on a half-finished grasp."""
    if cancel_event is not None and cancel_event.is_set():
        try:
            arm.open_gripper()
        except Exception:
            logger.exception("grasp cancel: open_gripper (safe-park) failed")
        raise GraspCancelled()


def run_grasp_once(
    target: str,
    *,
    arm: Any,
    segmenter: Any = None,
    camera: Any = None,
    K: Optional[np.ndarray] = None,
    T_hand_eye: Optional[np.ndarray] = None,
    cancel_event: Optional[threading.Event] = None,
    conf: float = 0.25,
    iou: float = 0.45,
    depth_quantile: float = 0.5,
    pregrasp_offset_m: float = 0.08,
    insertion_depth_m: float = 0.015,
    lift_height_m: float = 0.12,
    grasp_force: Optional[float] = None,
    move_duration: float = 2.0,
    warm_up_frames: int = 5,
    release_after: bool = False,
) -> dict:
    """Run a single cancellable grasp attempt for ``target``.

    Args:
        target: object label to grasp (must be in the segmenter vocabulary).
        arm: a connected ``RebotArm`` (move_to / get_tcp_pose / grasp /
            open_gripper / release_gripper).
        segmenter: a :class:`..perception.yolo_onnx.YoloOnnxSegmenter`. If
            ``None`` the caller must have configured one elsewhere (raises).
        camera: an opened camera driver exposing ``get_frame()`` ->
            ``(color_bgr, depth_mm)`` and ``warm_up(n)`` / ``K``.
        K: camera intrinsics (3x3). Falls back to ``camera.K`` when ``None``.
        T_hand_eye: eye-in-hand transform (4x4, camera←TCP). Required to
            transform the camera-frame grasp into the base frame.
        cancel_event: polled before each motion; set → safe-park + abort.
        conf/iou: detection thresholds.
        depth_quantile/pregrasp_offset_m/insertion_depth_m: grasp geometry.
        lift_height_m: how far up to retreat after the compliant grasp.
        grasp_force: compliant-close force (Nm); ``None`` → arm default.
        move_duration: per-waypoint duration (s).
        warm_up_frames: frames to discard for exposure/AWB stability.
        release_after: open the gripper at the end (drop the object).

    Returns:
        ``{"success": bool, "target": str, ...}``. On cancel:
        ``{"success": False, "cancelled": True, "stage": <str>, ...}``.
    """
    result: dict[str, Any] = {"success": False, "target": target, "cancelled": False}
    stage = "init"
    try:
        if segmenter is None:
            return {**result, "error": "no segmenter configured"}
        if camera is None:
            return {**result, "error": "no camera configured"}

        # ── 1. acquire a stable frame ───────────────────────────────────
        stage = "capture"
        _check_cancel(cancel_event, arm)
        if warm_up_frames > 0:
            try:
                camera.warm_up(warm_up_frames)
            except Exception:
                logger.debug("camera.warm_up failed (continuing)", exc_info=True)
        color_bgr, depth_mm = camera.get_frame()
        if color_bgr is None or depth_mm is None:
            return {**result, "stage": stage, "error": "camera returned no frame"}
        if K is None:
            K = np.asarray(camera.K, dtype=np.float32)
        K = np.asarray(K, dtype=np.float32)

        # ── 2. detect + grasp estimation (target class only) ────────────
        stage = "detect"
        _check_cancel(cancel_event, arm)
        from .perception.ordinary_grasp import estimate_grasps, select_best_grasp

        results = segmenter.predict(color_bgr, conf=conf, iou=iou)
        results = _filter_results_to_target(results, target)
        grasps = estimate_grasps(results, depth_mm, K, depth_quantile=depth_quantile)
        best = select_best_grasp(grasps)
        if best is None:
            return {
                **result,
                "stage": stage,
                "error": f"no valid grasp for target {target!r}",
                "num_detections": sum(len(getattr(r, "boxes", []) or []) for r in results),
            }
        result["grasp_class"] = best.class_name
        result["grasp_conf"] = float(best.conf)
        result["center_px"] = list(best.center_px)
        result["jaw_width_m"] = float(best.jaw_width_m)

        # ── 3. camera → base transform ──────────────────────────────────
        stage = "transform"
        _check_cancel(cancel_event, arm)
        if T_hand_eye is None:
            return {**result, "stage": stage, "error": "no hand-eye calibration"}
        from .perception.transforms import transform_grasp_pose_to_base

        T_cam2base = np.asarray(arm.get_tcp_pose(), dtype=np.float64) @ np.asarray(
            T_hand_eye, dtype=np.float64
        )
        grasp6d, pre6d = transform_grasp_pose_to_base(
            best.position,
            best.tcp_rotation,
            T_cam2base,
            pregrasp_offset_m,
            insertion_depth_m=insertion_depth_m,
        )
        result["grasp_pose"] = [float(v) for v in grasp6d]
        result["pregrasp_pose"] = [float(v) for v in pre6d]

        # ── 4. execute: open → pregrasp → grasp pos → compliant grasp ───
        stage = "open"
        _check_cancel(cancel_event, arm)
        arm.open_gripper()

        stage = "pregrasp"
        _check_cancel(cancel_event, arm)
        xp, yp, zp, rxp, ryp, rzp = pre6d
        if not arm.move_to(xp, yp, zp, rxp, ryp, rzp, duration=move_duration):
            return {**result, "stage": stage, "error": "pregrasp IK failed"}
        arm.wait_motion(move_duration)

        stage = "grasp_move"
        _check_cancel(cancel_event, arm)
        xg, yg, zg, rxg, ryg, rzg = grasp6d
        if not arm.move_to(xg, yg, zg, rxg, ryg, rzg, duration=max(1.0, move_duration * 0.75)):
            return {**result, "stage": stage, "error": "grasp-pose IK failed"}
        arm.wait_motion(max(1.0, move_duration * 0.75))

        stage = "grasp"
        _check_cancel(cancel_event, arm)
        held = bool(arm.grasp(force=grasp_force))
        result["grasp_closed"] = held

        # ── 5. lift (retreat straight up along base Z) ──────────────────
        stage = "lift"
        _check_cancel(cancel_event, arm)
        zl = zg + float(lift_height_m)
        if arm.move_to(xg, yg, zl, rxg, ryg, rzg, duration=move_duration):
            arm.wait_motion(move_duration)
        else:
            logger.warning("lift IK failed; leaving arm at grasp height")

        # ── 6. optional release ─────────────────────────────────────────
        if release_after:
            stage = "release"
            _check_cancel(cancel_event, arm)
            arm.release_gripper()

        # holding check (best-effort).
        try:
            result["holding"] = bool(arm.gripper_is_holding())
        except Exception:
            result["holding"] = held

        result["success"] = held
        result["stage"] = "done"
        return result

    except GraspCancelled:
        logger.info("grasp cancelled at stage=%s", stage)
        return {**result, "success": False, "cancelled": True, "stage": stage}
    except Exception as exc:  # pragma: no cover — defensive
        logger.exception("grasp pipeline failed at stage=%s", stage)
        # Best-effort safe-park on any unexpected failure.
        try:
            arm.open_gripper()
        except Exception:
            pass
        return {**result, "stage": stage, "error": str(exc)}


def _filter_results_to_target(results: list[Any], target: str) -> list[Any]:
    """Drop detections whose label != ``target`` so the grasp estimator only
    considers the requested object. Rebuilds each result's ``boxes`` / ``masks``
    in place via the same numpy containers the segmenter uses.

    Comparison is case-insensitive and also matches when the requested target
    is a substring of the detected label (e.g. ``"bottle"`` matches
    ``"water bottle"``).
    """
    from .perception.yolo_onnx import _Boxes, _Masks, YoloResult  # local import

    want = target.strip().lower()
    out: list[Any] = []
    for r in results:
        names = getattr(r, "names", {}) or {}
        boxes = getattr(r, "boxes", None)
        masks = getattr(r, "masks", None)
        if boxes is None:
            out.append(r)
            continue
        keep_idx: list[int] = []
        for i in range(len(boxes)):
            cls_id = int(np.asarray(boxes[i].cls[0]).reshape(-1)[0])
            label = (names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id))
            label_l = str(label).lower()
            if label_l == want or want in label_l or label_l in want:
                keep_idx.append(i)
        kept_boxes = _Boxes([boxes[i] for i in keep_idx])
        kept_masks = None
        if masks is not None and getattr(masks, "data", None) is not None and keep_idx:
            data = np.asarray(masks.data)
            kept_masks = _Masks(data[keep_idx])
        out.append(
            YoloResult(
                names=names,
                boxes=kept_boxes,
                masks=kept_masks,
                orig_shape=getattr(r, "orig_shape", (0, 0)),
            )
        )
    return out


__all__ = ["run_grasp_once", "GraspCancelled"]
