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

import contextlib
import logging
import math
import threading
from typing import Any, Optional

import numpy as np

from .rebot_actuator import sleep_cancellable

logger = logging.getLogger(__name__)


class GraspCancelled(Exception):
    """Raised internally when ``cancel_event`` fires mid-pipeline."""


# SDK mechanical max jaw opening (m) and the clearance added over the detected
# object width so the open jaw clears the object before the approach.
_GRIPPER_MAX_M = 0.09
_OPEN_MARGIN_M = 0.012


def _motion_lock(actuator: Any):
    """Return a context manager that holds the actuator's bus lock for one
    arm motion, or a null context when no actuator is supplied.

    SAFETY: the grasp pipeline drives the raw arm directly, bypassing
    ``execute_sequence``. Wrapping each discrete bus op (one move / one
    gripper command) in the actuator's lock keeps it mutually exclusive with
    a concurrent action sequence, the 500Hz gripper thread and observation
    reads. Like execute_sequence, the lock is per-op — the blocking
    ``wait_motion`` poll happens OUTSIDE the lock so torque-off / cache reads
    are not starved during the multi-second move.
    """
    acq = getattr(actuator, "acquire_motion_lock", None)
    if callable(acq):
        return acq()
    return contextlib.nullcontext()


def _safe_open_distance(value: float) -> float:
    dist = float(value)
    if not math.isfinite(dist) or dist < 0.0:
        raise ValueError(
            "open_distance_m must be a non-negative finite number; "
            f"got {value!r}"
        )
    return dist


def _open_gripper_safe(arm: Any, open_distance_m: float) -> None:
    arm.open_gripper(open_distance_m)


def _check_cancel(
    cancel_event: Optional[threading.Event],
    arm: Any,
    open_distance_m: float,
) -> None:
    """Raise :class:`GraspCancelled` (after parking the gripper safe) if the
    cancel event is set. Safe-park = open the gripper so we never leave it
    clamped on a half-finished grasp."""
    if cancel_event is not None and cancel_event.is_set():
        try:
            _open_gripper_safe(arm, open_distance_m)
        except Exception:
            logger.exception("grasp cancel: open_gripper (safe-park) failed")
        raise GraspCancelled()


def _wait_motion_cancellable(
    arm: Any,
    duration: float,
    cancel_event: Optional[threading.Event],
) -> bool:
    """Wait for a move to settle in small steps, polling ``cancel_event`` every
    ~0.1s. Returns True if it completed, False if cancelled (caller safe-parks
    and must NOT issue further motion). The arm's own ``wait_motion`` blocks
    uninterruptibly, so we poll instead and only consult it for the final
    settle when not cancelled."""
    if cancel_event is None:
        arm.wait_motion(duration)
        return True
    return sleep_cancellable(max(0.0, float(duration)), cancel_event)


def run_grasp_once(
    target: str,
    *,
    arm: Any,
    actuator: Any = None,
    segmenter: Any = None,
    camera: Any = None,
    K: Optional[np.ndarray] = None,
    T_hand_eye: Optional[np.ndarray] = None,
    scan_poses: Optional[list] = None,
    cancel_event: Optional[threading.Event] = None,
    conf: float = 0.25,
    iou: float = 0.45,
    depth_quantile: float = 0.5,
    pregrasp_offset_m: float = 0.08,
    insertion_depth_m: float = 0.015,
    lift_height_m: float = 0.12,
    grasp_force: Optional[float] = None,
    open_distance_m: float = 0.06,
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
        open_distance_m: safe jaw opening distance (m) for pre-grasp and
            cancellation safe-park. The SDK mechanical max is 0.09m; this
            default intentionally stays at the validated 0.06m action width.
        move_duration: per-waypoint duration (s).
        warm_up_frames: frames to discard for exposure/AWB stability.
        release_after: open the gripper at the end (drop the object).

    Returns:
        ``{"success": bool, "target": str, ...}``. On cancel:
        ``{"success": False, "cancelled": True, "stage": <str>, ...}``.
    """
    result: dict[str, Any] = {"success": False, "target": target, "cancelled": False}
    stage = "init"
    safe_open_m = _safe_open_distance(open_distance_m)
    try:
        if segmenter is None:
            return {**result, "error": "no segmenter configured"}
        if camera is None:
            return {**result, "error": "no camera configured"}

        # ── 1+2. detect from the current view; if nothing is seen AND
        # scan_poses are configured, sweep the camera across them until the
        # target appears (auto-search), so grasp_object does not silently fail
        # just because the object is not perfectly centered. ────────────
        from .perception.ordinary_grasp import estimate_grasps, select_best_grasp

        def _capture_and_detect():
            if warm_up_frames > 0:
                try:
                    camera.warm_up(warm_up_frames)
                except Exception:
                    logger.debug("camera.warm_up failed (continuing)", exc_info=True)
            color_bgr, depth_mm = camera.get_frame()
            if color_bgr is None or depth_mm is None:
                return None, None, 0
            Kl = np.asarray(K if K is not None else camera.K, dtype=np.float32)
            results = segmenter.predict(color_bgr, conf=conf, iou=iou, only_names={target})
            results = _filter_results_to_target(results, target)
            nd = sum(len(getattr(r, "boxes", []) or []) for r in results)
            b = select_best_grasp(
                estimate_grasps(results, depth_mm, Kl, depth_quantile=depth_quantile)
            )
            return b, Kl, nd

        stage = "capture"
        _check_cancel(cancel_event, arm, safe_open_m)
        best, K_local, num_det = _capture_and_detect()
        if best is None and scan_poses:
            stage = "scan"
            for pose in scan_poses:
                _check_cancel(cancel_event, arm, safe_open_m)
                with _motion_lock(actuator):
                    mok = arm.move_to(*pose, duration=move_duration)
                if not mok:
                    continue
                if not _wait_motion_cancellable(arm, move_duration, cancel_event):
                    _check_cancel(cancel_event, arm, safe_open_m)
                best, K_local, num_det = _capture_and_detect()
                if best is not None:
                    logger.info("grasp: target found while scanning at pose %r", pose)
                    break

        stage = "detect"
        if best is None:
            return {
                **result,
                "stage": stage,
                "error": f"no valid grasp for target {target!r}",
                "num_detections": int(num_det),
            }
        K = K_local
        result["grasp_class"] = best.class_name
        result["grasp_conf"] = float(best.conf)
        result["center_px"] = list(best.center_px)
        result["jaw_width_m"] = float(best.jaw_width_m)

        # Auto-size the pre-grasp open width to the detected object: a fixed
        # safe-open (e.g. 0.06m) is NARROWER than a wide box (e.g. 0.077m) and
        # the jaw would collide instead of going around it. Widen to
        # object_width + margin, clamped to the mechanical max. Never shrink
        # below the configured safe-open. The safe-park (cancel) open uses the
        # same widened value so a release always clears the object.
        widened = float(best.jaw_width_m) + _OPEN_MARGIN_M
        safe_open_m = min(_GRIPPER_MAX_M, max(safe_open_m, widened))
        result["open_distance_m"] = safe_open_m

        # ── 3. camera → base transform ──────────────────────────────────
        stage = "transform"
        _check_cancel(cancel_event, arm, safe_open_m)
        if T_hand_eye is None:
            return {**result, "stage": stage, "error": "no hand-eye calibration"}
        from .perception.transforms import transform_grasp_pose_to_base

        with _motion_lock(actuator):
            tcp_pose = np.asarray(arm.get_tcp_pose(), dtype=np.float64)
        T_cam2base = tcp_pose @ np.asarray(
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
        # SAFETY: each arm bus op is wrapped in the actuator lock (atomic vs a
        # concurrent action / gripper thread / cache read); the blocking
        # settle wait runs OUTSIDE the lock and is cancellable.
        stage = "open"
        _check_cancel(cancel_event, arm, safe_open_m)
        with _motion_lock(actuator):
            _open_gripper_safe(arm, safe_open_m)

        stage = "pregrasp"
        _check_cancel(cancel_event, arm, safe_open_m)
        xp, yp, zp, rxp, ryp, rzp = pre6d
        with _motion_lock(actuator):
            pregrasp_ok = arm.move_to(xp, yp, zp, rxp, ryp, rzp, duration=move_duration)
        if not pregrasp_ok:
            return {**result, "stage": stage, "error": "pregrasp IK failed"}
        if not _wait_motion_cancellable(arm, move_duration, cancel_event):
            _check_cancel(cancel_event, arm, safe_open_m)  # safe-park + raise

        stage = "grasp_move"
        _check_cancel(cancel_event, arm, safe_open_m)
        xg, yg, zg, rxg, ryg, rzg = grasp6d
        grasp_dur = max(1.0, move_duration * 0.75)
        with _motion_lock(actuator):
            grasp_move_ok = arm.move_to(xg, yg, zg, rxg, ryg, rzg, duration=grasp_dur)
        if not grasp_move_ok:
            return {**result, "stage": stage, "error": "grasp-pose IK failed"}
        if not _wait_motion_cancellable(arm, grasp_dur, cancel_event):
            _check_cancel(cancel_event, arm, safe_open_m)  # safe-park + raise

        stage = "grasp"
        _check_cancel(cancel_event, arm, safe_open_m)
        with _motion_lock(actuator):
            held = bool(arm.grasp(force=grasp_force))
        result["grasp_closed"] = held

        # ── 5. lift — actually pick the object UP ───────────────────────
        # First try a straight-up lift at the grasp x/y. If the object was
        # grasped near the reach limit (large x), "far + up" is often IK-
        # unreachable, so a naive straight-up lift silently fails and the arm
        # just stays clamped at table height ("只夹住不拿起来"). Fall back to a
        # guaranteed-reachable holding pose that pulls the object back + up to a
        # visible carry position, so it is genuinely lifted regardless of where
        # it was grasped.
        stage = "lift"
        _check_cancel(cancel_event, arm, safe_open_m)
        lifted = False
        zl = zg + float(lift_height_m)
        with _motion_lock(actuator):
            lift_ok = arm.move_to(xg, yg, zl, rxg, ryg, rzg, duration=move_duration)
        if lift_ok:
            if not _wait_motion_cancellable(arm, move_duration, cancel_event):
                _check_cancel(cancel_event, arm, safe_open_m)  # safe-park + raise
            lifted = True
        else:
            # Fallback: a safe holding pose well inside the reachable box.
            hx = min(0.30, max(0.22, xg))
            hy = min(0.12, max(-0.12, yg))
            hz = 0.26
            hold_ok = False
            try:
                hold_ok, _ = arm.check_ik(hx, hy, hz, 0.0, 0.0, 0.0)
            except Exception:
                hold_ok = False
            if hold_ok:
                logger.info("lift: straight-up IK failed; lifting to holding pose "
                            "(%.2f,%.2f,%.2f)", hx, hy, hz)
                with _motion_lock(actuator):
                    moved = arm.move_to(hx, hy, hz, 0.0, 0.0, 0.0, duration=move_duration)
                if moved:
                    if not _wait_motion_cancellable(arm, move_duration, cancel_event):
                        _check_cancel(cancel_event, arm, safe_open_m)
                    lifted = True
            if not lifted:
                logger.warning("lift IK failed (both straight-up and holding pose); "
                               "leaving arm at grasp height")
        result["lifted"] = lifted

        # ── 6. optional release ─────────────────────────────────────────
        if release_after:
            stage = "release"
            _check_cancel(cancel_event, arm, safe_open_m)
            with _motion_lock(actuator):
                _open_gripper_safe(arm, safe_open_m)

        # holding check (best-effort). On the real RebotArm this is a
        # PROPERTY (not a method); only call it when it's actually callable so
        # we don't TypeError on a bool.
        try:
            holding_attr = getattr(arm, "gripper_is_holding", None)
            holding = holding_attr() if callable(holding_attr) else holding_attr
            result["holding"] = bool(holding) if holding is not None else held
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
            _open_gripper_safe(arm, safe_open_m)
        except Exception:
            pass
        return {**result, "stage": stage, "error": str(exc)}


def run_search_once(
    target: str,
    *,
    arm: Any,
    actuator: Any = None,
    segmenter: Any = None,
    camera: Any = None,
    T_hand_eye: Optional[np.ndarray] = None,
    scan_poses: Optional[list] = None,
    home_pose: tuple = (0.27, 0.0, 0.24, 0.0, 0.0, 0.0),
    cancel_event: Optional[threading.Event] = None,
    conf: float = 0.20,
    move_duration: float = 2.0,
    warm_up_frames: int = 3,
    frames: int = 6,
    indicate: bool = True,
) -> dict:
    """Sweep the eye-in-hand camera across ``scan_poses`` to find ``target``.

    Unlike :func:`run_grasp_once` (which only looks from the current pose), this
    moves the arm through a list of observation poses, runs multi-frame
    detection at each, and stops at the first pose where the target is found —
    optionally reaching toward it ("pointing") without grasping. If no pose sees
    the target, the arm returns home. The gripper is never closed.

    Returns ``{"found": bool, "target": str, "scan_index": int,
    "position_base": [x,y,z], "conf": float, "indicated": bool, ...}``.
    """
    result: dict[str, Any] = {"found": False, "target": target, "cancelled": False}
    try:
        if segmenter is None or camera is None:
            return {**result, "error": "perception not configured"}
        poses = list(scan_poses) if scan_poses else [home_pose]
        K = np.asarray(camera.K, dtype=np.float32)
        scanned = 0
        for idx, pose in enumerate(poses):
            if cancel_event is not None and cancel_event.is_set():
                return {**result, "cancelled": True, "scan_index": idx}
            # move to the observation pose
            with _motion_lock(actuator):
                ok = arm.move_to(*pose, duration=move_duration)
            if not ok:
                logger.info("search: scan pose %d IK failed; skipping", idx)
                continue
            if not _wait_motion_cancellable(arm, move_duration, cancel_event):
                return {**result, "cancelled": True, "scan_index": idx}
            scanned += 1
            # multi-frame detection at this view
            try:
                camera.warm_up(warm_up_frames)
            except Exception:
                logger.debug("camera.warm_up failed (continuing)", exc_info=True)
            best = None
            for _ in range(max(1, frames)):
                color_bgr, depth_mm = camera.get_frame()
                if color_bgr is None or depth_mm is None:
                    continue
                from .perception.ordinary_grasp import estimate_grasps, select_best_grasp
                results = segmenter.predict(color_bgr, conf=conf, only_names={target})
                results = _filter_results_to_target(results, target)
                cand = select_best_grasp(estimate_grasps(results, depth_mm, K))
                if cand is not None and (best is None or cand.conf > best.conf):
                    best = cand
            if best is None:
                continue
            # found — compute the target's base-frame position for reporting
            result["found"] = True
            result["scan_index"] = idx
            result["conf"] = float(best.conf)
            result["center_px"] = list(best.center_px)
            if T_hand_eye is not None:
                from .perception.transforms import transform_grasp_pose_to_base
                with _motion_lock(actuator):
                    tcp_pose = np.asarray(arm.get_tcp_pose(), dtype=np.float64)
                T_cam2base = tcp_pose @ np.asarray(T_hand_eye, dtype=np.float64)
                # Use the SAME transform the (validated) grasp path uses so the
                # reported/pointed position matches reality. grasp6d[:3] is the
                # box's base-frame xyz; we only point near it (no grasp).
                grasp6d, _pre = transform_grasp_pose_to_base(
                    best.position, best.tcp_rotation, T_cam2base,
                    pregrasp_offset_m=0.08, insertion_depth_m=0.0,
                )
                p_base = [float(v) for v in grasp6d[:3]]
                result["position_base"] = p_base
                # A single depth pixel from a far/high scan pose can yield a
                # physically impossible position (e.g. z below the base plane),
                # so only POINT at it when the computed location is plausible for
                # a box on the table AND IK-reachable. Otherwise we simply stay
                # at this scan pose — the camera is already centered on the box,
                # which is itself a natural "found it" indication. Never grasps.
                bx, by, bz = p_base
                plausible = (0.15 <= bx <= 0.50 and -0.25 <= by <= 0.25 and 0.0 <= bz <= 0.30)
                result["position_plausible"] = plausible
                if (indicate and plausible
                        and not (cancel_event is not None and cancel_event.is_set())):
                    px = min(0.34, max(0.20, bx))
                    py = min(0.14, max(-0.14, by))
                    pz = 0.20
                    try:
                        ik_ok, _ = arm.check_ik(px, py, pz, 0.0, 0.0, 0.0)
                    except Exception:
                        ik_ok = False
                    if ik_ok:
                        with _motion_lock(actuator):
                            arm.move_to(px, py, pz, 0.0, 0.0, 0.0, duration=move_duration)
                        _wait_motion_cancellable(arm, move_duration, cancel_event)
                        result["indicated"] = True
            result["scanned_poses"] = scanned
            return result
        # nothing found anywhere → return home
        if cancel_event is None or not cancel_event.is_set():
            with _motion_lock(actuator):
                arm.move_to(*home_pose, duration=move_duration)
        result["scanned_poses"] = scanned
        return result
    except Exception as exc:  # pragma: no cover — defensive
        logger.exception("search pipeline failed")
        return {**result, "error": str(exc)}


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


__all__ = ["run_grasp_once", "run_search_once", "GraspCancelled"]
