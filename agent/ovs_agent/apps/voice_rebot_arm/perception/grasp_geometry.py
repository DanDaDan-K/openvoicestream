"""Pure grasp-pose geometry — the SINGLE SOURCE OF TRUTH for the executed
base-frame pose, shared by the live pipeline (grasp_service.run_grasp_once) and
the offline real-frame replay harness (tools/grasp_replay.py).

Leaf module: imports only numpy + transforms, NO app/voxedge deps, so the
geometry can be unit-tested and replayed on real captured depth frames without
booting the agent.
"""
from __future__ import annotations

import logging
import os

import numpy as np

from .transforms import transform_grasp_pose_to_base

logger = logging.getLogger(__name__)


def _side_insertion_depth_m(default: float) -> float:
    """Deeper insertion for SIDE grasps so the jaw wraps the box body instead of
    catching its front edge (real machine 2026-06-17: shallow grip → slip).
    Env-tunable REBOT_SIDE_INSERT (m); default 0.04."""
    try:
        v = float(os.environ.get("REBOT_SIDE_INSERT", "") or 0.0)
    except (TypeError, ValueError):
        v = 0.0
    return max(default, v if v > 0 else 0.035)


def finalize_grasp_pose(best, T_cam2base, pregrasp_offset_m, insertion_depth_m):
    """Camera-frame grasp candidate → executed base-frame ``(grasp6d, pre6d)``.

    Pure — no arm, no camera, no I/O. Steps:

      1. camera→base transform with the insertion/pregrasp offset along the
         camera→object ray (``offset_axis_cam=best.position``) — decouples the
         insertion translation from the box-facing approach re-aim so the
         landing point does not swing when the gripper turns to face the box.
      2. TOP grasps: CENTRE on the top-face centroid + vertical bite. The
         insertion along the forward-tilted ray adds ~14mm useful downward bite
         but also ~21mm of HORIZONTAL shift toward the box's far edge (real
         machine 2026-06-17: jaw closed ~2cm off-centre → could not hold). The
         horizontal shift is a pure artifact, so pin x,y to the centroid and
         keep the bite via z (≥28mm below the measured top surface, ≥25mm above
         the table — the 2026-06-12 "抓得深" floor). The pregrasp tracks the same
         x,y shift so the approach path stays consistent.
    """
    method = getattr(best, "method", "legacy")
    # SIDE grasps: advance the insertion along the FACE NORMAL into the box (set
    # on the GraspPose) and deeper, so the jaw wraps the body rather than the
    # front edge. TOP/other grasps keep the camera→object ray (centring path).
    side_axis = getattr(best, "insertion_axis_cam", None)
    if method == "side_face" and side_axis is not None:
        offset_axis = side_axis
        insertion = _side_insertion_depth_m(insertion_depth_m)
    else:
        offset_axis = best.position
        insertion = insertion_depth_m
    grasp6d, pre6d = transform_grasp_pose_to_base(
        best.position,
        best.tcp_rotation,
        T_cam2base,
        pregrasp_offset_m,
        insertion_depth_m=insertion,
        offset_axis_cam=offset_axis,
    )
    if method == "top_face":
        surf = (
            np.asarray(T_cam2base, dtype=np.float64)
            @ np.append(np.asarray(best.position, dtype=np.float64), 1.0)
        )[:3]
        dx = float(surf[0]) - float(grasp6d[0])
        dy = float(surf[1]) - float(grasp6d[1])
        z_bite = max(float(surf[2]) - 0.028, 0.025)
        gz = min(float(grasp6d[2]), z_bite)
        logger.info(
            "grasp: centred top-grasp on centroid (dx %.3f dy %.3f) "
            "bite z %.3f→%.3f (surface %.3f)",
            dx, dy, float(grasp6d[2]), gz, float(surf[2]),
        )
        grasp6d = [float(surf[0]), float(surf[1]), gz,
                   *(float(v) for v in grasp6d[3:])]
        pre6d = [float(pre6d[0]) + dx, float(pre6d[1]) + dy,
                 *(float(v) for v in pre6d[2:])]
    return [float(v) for v in grasp6d], [float(v) for v in pre6d]
