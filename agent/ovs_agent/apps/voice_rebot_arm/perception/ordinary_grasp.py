"""Short-axis (min-area-rect) grasp estimation — torch-free vendoring.

Vendored from ``reBot-DevArm-Grasp/utils/ordinary_grasp.py`` with the torch
runtime dependency removed. The upstream consumed ultralytics ``Results`` and
called ``.cpu().numpy()`` on box/mask tensors; here the input is a numpy-only
:class:`..perception.yolo_onnx.YoloResult`, so every tensor access is a plain
``np.asarray`` and there is no torch import, no ``.cpu()``, no ultralytics.

The geometry (OBB short-edge → grip/open/approach axes → 6-DoF camera-frame
pose) is byte-for-byte the upstream algorithm; only the result-field plumbing
changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from .transforms import grasp_axes_to_rebot_tcp_rotation


# ── torch-free helpers (inlined from upstream common_utils) ──────────────────
def detection_count(result: Any) -> int:
    for attr in ("obb", "boxes"):
        container = getattr(result, attr, None)
        if container is None:
            continue
        try:
            count = len(container)
        except Exception:
            continue
        if count > 0:
            return count
    return 0


@dataclass
class GraspPose:
    class_name: str
    conf: float
    bbox_xyxy: tuple[int, int, int, int]
    center_px: tuple[int, int]
    position: Optional[np.ndarray]
    rotation: Optional[np.ndarray]
    tcp_rotation: Optional[np.ndarray]
    jaw_width_m: float
    object_length_m: float
    angle_deg: float
    rect_points: np.ndarray
    short_edge_points: np.ndarray
    valid_depth_pixels: int
    rejected_reason: Optional[str] = None
    # which estimator produced this pose: "top_face" (3D plane fit — only
    # possible when the camera can actually SEE the top), "side_face", or
    # "legacy" (silhouette short-axis). grasp_service uses this to pick the
    # re-observation strategy: no top face visible → go HIGH and tilt down.
    method: str = "legacy"
    # GG-CNN second-opinion vote (None = refiner off/unavailable): False
    # makes grasp_service trigger a re-observation before committing.
    ggcnn_agree: "Optional[bool]" = None

    @property
    def is_valid(self) -> bool:
        return (
            self.rejected_reason is None
            and self.position is not None
            and self.rotation is not None
        )


def get_depth_mm(depth_map: np.ndarray, u: int, v: int, roi_size: int = 5) -> float:
    """Sample the median valid depth from a small window."""
    h, w = depth_map.shape
    half = roi_size // 2
    x1, x2 = max(0, u - half), min(w, u + half + 1)
    y1, y2 = max(0, v - half), min(h, v + half + 1)
    roi = depth_map[y1:y2, x1:x2]
    valid = roi[roi > 0]
    return float(np.median(valid)) if len(valid) > 0 else 0.0


def estimate_grasps(
    results: list[Any],
    depth_mm: np.ndarray,
    K: np.ndarray,
    depth_quantile: float = 0.75,
    up_hint_cam: Optional[np.ndarray] = None,
    ggcnn: Any = None,
) -> list[GraspPose]:
    grasps: list[GraspPose] = []
    for result in results:
        for index in range(detection_count(result)):
            grasps.append(
                estimate_grasp(
                    result, index, depth_mm, K,
                    depth_quantile=depth_quantile,
                    up_hint_cam=up_hint_cam,
                    ggcnn=ggcnn,
                )
            )
    return grasps


def select_best_grasp(grasps: list[GraspPose]) -> Optional[GraspPose]:
    valid = [grasp for grasp in grasps if grasp.is_valid]
    if not valid:
        return None
    return max(valid, key=lambda grasp: grasp.conf)


def estimate_grasp(
    result: Any,
    index: int,
    depth_mm: np.ndarray,
    K: np.ndarray,
    depth_quantile: float = 0.75,
    up_hint_cam: Optional[np.ndarray] = None,
    ggcnn: Any = None,
) -> GraspPose:
    class_name, conf, bbox_xyxy = _detection_meta(result, index)
    rect_points = _rect_points(result, index, depth_mm.shape, bbox_xyxy)
    center = rect_points.mean(axis=0).astype(np.float32)

    mask = _depth_mask(result, index, depth_mm.shape, rect_points)

    # ── TOP-FACE plane grasp (preferred when an up-hint is available) ──
    # The 2D-silhouette family (min-area-rect below) merges the box's top and
    # side faces under oblique views, so the "short axis" can measure the
    # wrong physical dimension entirely. Fitting the TOP plane in 3D (RANSAC,
    # candidate whose normal best matches "up") and doing PCA inside that
    # plane measures the real graspable face: metric width, true angle, and
    # a face-normal approach. Falls back to the legacy estimators whenever
    # the fit is not confident (curved objects, sparse depth, no hint).
    if up_hint_cam is not None:
        side_cands: list = []
        top = _top_face_grasp(
            mask, depth_mm, K, np.asarray(up_hint_cam, dtype=np.float64),
            side_out=side_cands,
        )
        # Arbitration: TOP grasp when the top face is visible AND its width
        # fits the jaw; otherwise a SIDE grasp on a camera-facing vertical
        # face whose HORIZONTAL extent fits (tall objects whose top is out of
        # view / too wide). Each construction keeps the legacy camera-ray
        # approach (transform sign convention + the measured IK envelope:
        # pitch 0.2-0.9 is the 93-99% feasible band, pure down-press is not).
        if top is not None and top[3] > 0.085 and side_cands:
            top = None  # top face too wide for the jaw — try the side path
        if top is None and side_cands:
            best_side = min(
                (c for c in side_cands if c[3] <= 0.085),
                key=lambda c: c[3],
                default=None,
            )
            if best_side is not None:
                c_pos, horiz, _n_cam, h_width, v_len, n_in = best_side
                approach_s = _normalize(-c_pos)
                if approach_s is None:
                    approach_s = np.array([0.0, 0.0, -1.0], dtype=np.float64)
                grip_s = _normalize(np.cross(horiz, approach_s))
                open_s = _normalize(np.cross(approach_s, grip_s))
                if grip_s is not None and open_s is not None:
                    rotation = np.column_stack([grip_s, open_s, approach_s]).astype(np.float32)
                    tcp_rotation = grasp_axes_to_rebot_tcp_rotation(
                        rotation[:, 0], rotation[:, 1], rotation[:, 2]
                    ).astype(np.float32)
                    u, v = _project(c_pos, K)
                    return GraspPose(
                        class_name=class_name,
                        conf=conf,
                        bbox_xyxy=bbox_xyxy,
                        center_px=(int(round(u)), int(round(v))),
                        position=c_pos.astype(np.float32),
                        rotation=rotation,
                        tcp_rotation=tcp_rotation,
                        jaw_width_m=float(h_width),
                        object_length_m=float(v_len),
                        angle_deg=float(np.degrees(np.arctan2(open_s[1], open_s[0]))),
                        rect_points=rect_points,
                        short_edge_points=_line_from_center(
                            np.array([u, v], dtype=np.float32),
                            np.array([open_s[0], open_s[1]], dtype=np.float32) * 40.0,
                        ),
                        valid_depth_pixels=int(n_in),
                        method="side_face",
                    )
        if top is not None:
            position_t, open_axis_t, _face_normal_t, width_t, length_t, n_in = top
            # APPROACH stays the legacy camera-ray (pointing TOWARD the
            # camera — grasp_axes_to_rebot_tcp_rotation negates it into the
            # tool-forward). Two reasons: (1) sign convention — the transform
            # assumes a toward-camera approach; (2) kinematics — the real
            # B601-DM's validated grasps all use the forward-tilted camera-ray
            # approach (pitch 0.3-0.6), a pure face-normal down-press lands
            # outside the comfortable IK envelope. The TOP-FACE fit therefore
            # contributes only what the silhouette could not measure: the true
            # metric width/length and the in-plane open-axis direction.
            approach_t = _normalize(-position_t)
            if approach_t is None:
                approach_t = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            grip_t = _normalize(np.cross(open_axis_t, approach_t))
            open_t = _normalize(np.cross(approach_t, grip_t))
            if grip_t is not None and open_t is not None:
                rotation = np.column_stack([grip_t, open_t, approach_t]).astype(np.float32)
                tcp_rotation = grasp_axes_to_rebot_tcp_rotation(
                    rotation[:, 0], rotation[:, 1], rotation[:, 2]
                ).astype(np.float32)
                u, v = _project(position_t, K)
                short_uv = _line_from_center(
                    np.array([u, v], dtype=np.float32),
                    np.array([open_t[0], open_t[1]], dtype=np.float32) * 40.0,
                )
                agree = _ggcnn_vote(
                    ggcnn, depth_mm, mask, K,
                    float(np.degrees(np.arctan2(open_t[1], open_t[0]))),
                    float(width_t), (int(round(u)), int(round(v))),
                )
                return GraspPose(
                    class_name=class_name,
                    conf=conf,
                    bbox_xyxy=bbox_xyxy,
                    center_px=(int(round(u)), int(round(v))),
                    position=position_t.astype(np.float32),
                    rotation=rotation,
                    tcp_rotation=tcp_rotation,
                    jaw_width_m=float(width_t),
                    object_length_m=float(length_t),
                    angle_deg=float(np.degrees(np.arctan2(open_t[1], open_t[0]))),
                    rect_points=rect_points,
                    short_edge_points=short_uv,
                    valid_depth_pixels=int(n_in),
                    method="top_face",
                    ggcnn_agree=agree,
                )
    # ── GG-CNN PRIMARY (curved/irregular objects) ──────────────────────
    # Reaching here means NO plane fit held (the plane-failure itself is the
    # curved-object detector). When the refiner is enabled, its per-pixel
    # quality map replaces the weak silhouette geometry: grasp point + angle
    # + width come from the network, the axis construction below is reused.
    _gg_primary = None
    if ggcnn is not None and up_hint_cam is not None:
        try:
            _gg_primary = ggcnn.predict(depth_mm, mask, K)
        except Exception:
            _gg_primary = None

    # One pass over the 4 rect edges: gather vectors/norms, then derive the
    # longest edge (object length) and the shortest edge (grasp short-axis).
    edge_vecs = [rect_points[(i + 1) % 4] - rect_points[i] for i in range(4)]
    edge_lengths = [float(np.linalg.norm(vec)) for vec in edge_vecs]
    long_len_px = max(edge_lengths)
    short_i = min(range(4), key=lambda i: edge_lengths[i])
    short_vec_uv = edge_vecs[short_i].astype(np.float32)
    short_len_px = edge_lengths[short_i]
    short_dir_uv = _normalize(short_vec_uv)
    grasp_span_px = short_len_px
    short_edge_points = _line_from_center(center, short_vec_uv)

    if short_dir_uv is not None:
        refined = _refine_grasp_line_from_mask(mask, center, short_dir_uv, long_len_px)
        if refined is not None:
            center, short_edge_points, grasp_span_px = refined

    if _gg_primary is not None and _gg_primary.quality > 0.1:
        # Override the silhouette's center/axis with the network's best
        # in-mask grasp; the geometric construction below stays identical.
        center = np.array(_gg_primary.center_px, dtype=np.float32)
        ga = float(_gg_primary.angle_rad)
        short_dir_uv = np.array([np.cos(ga), np.sin(ga)], dtype=np.float32)
        fxl = max(float(K[0, 0]), 1e-6)
        grasp_span_px = float(_gg_primary.width_m) * fxl / max(_gg_primary.depth_m, 1e-6)
        short_edge_points = _line_from_center(center, short_dir_uv * grasp_span_px)

    center_px = (int(round(float(center[0]))), int(round(float(center[1]))))
    depth_values = depth_mm[mask > 0]
    depth_values = depth_values[depth_values > 0]
    if len(depth_values) == 0:
        center_depth = get_depth_mm(depth_mm, center_px[0], center_px[1], 5)
        if center_depth > 0:
            depth_values = np.array([center_depth], dtype=np.float32)

    if len(depth_values) == 0 or short_dir_uv is None:
        return _rejected(
            class_name, conf, bbox_xyxy, center_px, rect_points, short_edge_points,
            len(depth_values), "no_valid_depth_or_rect",
        )

    depth_quantile = float(np.clip(depth_quantile, 0.0, 1.0))
    z_m = float(np.quantile(depth_values, depth_quantile) / 1000.0)
    position = _backproject(float(center[0]), float(center[1]), z_m, K)
    approach = _normalize(-position)
    if approach is None:
        approach = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    # 3D short-axis measurement (2026-06-12): the legacy path turned the 2D
    # pixel direction into 3D with a z=0 assumption (_pixel_vec_to_3d), i.e.
    # "the grasped face is fronto-parallel to the camera". For an OBLIQUE box
    # that skews the open-axis DIRECTION and inflates the WIDTH — the real
    # machine then gripped across the wrong edge ("不对着短边夹"). Sample the
    # depth at two points INSIDE the object along the short axis and
    # back-project each with its own depth: their difference is the true 3D
    # open axis + width. Falls back to the legacy estimate when the depth at
    # either sample is invalid.
    open_vec_3d = _short_axis_3d(
        depth_mm, K, center, short_dir_uv, grasp_span_px
    )
    jaw_width_3d: Optional[float] = None
    if open_vec_3d is not None:
        jaw_width_3d = float(np.linalg.norm(open_vec_3d))
        open_axis = open_vec_3d
    else:
        open_axis = _pixel_vec_to_3d(short_dir_uv, z_m, K)
    open_axis = open_axis - float(np.dot(open_axis, approach)) * approach
    open_axis = _normalize(open_axis)
    if open_axis is None:
        return _rejected(
            class_name, conf, bbox_xyxy, center_px, rect_points, short_edge_points,
            len(depth_values), "open_axis_failed",
        )

    if open_axis[0] < 0:
        open_axis = -open_axis
    grip_axis = _normalize(np.cross(open_axis, approach))
    open_axis = _normalize(np.cross(approach, grip_axis))
    if grip_axis is None or open_axis is None:
        return _rejected(
            class_name, conf, bbox_xyxy, center_px, rect_points, short_edge_points,
            len(depth_values), "grasp_axis_failed",
        )

    rotation = np.column_stack([grip_axis, open_axis, approach]).astype(np.float32)
    tcp_rotation = grasp_axes_to_rebot_tcp_rotation(
        rotation[:, 0], rotation[:, 1], rotation[:, 2]
    ).astype(np.float32)

    if jaw_width_3d is not None:
        jaw_width_m = jaw_width_3d
    else:
        jaw_width_m = float(
            np.linalg.norm(_pixel_vec_to_3d(short_dir_uv * grasp_span_px, z_m, K))
        )
    object_length_m = float(np.linalg.norm(_pixel_vec_to_3d(short_dir_uv * long_len_px, z_m, K)))
    angle_deg = float(np.degrees(np.arctan2(short_dir_uv[1], short_dir_uv[0])))

    return GraspPose(
        class_name=class_name,
        conf=conf,
        bbox_xyxy=bbox_xyxy,
        center_px=center_px,
        position=position,
        rotation=rotation,
        tcp_rotation=tcp_rotation,
        jaw_width_m=jaw_width_m,
        object_length_m=object_length_m,
        angle_deg=angle_deg,
        rect_points=rect_points,
        short_edge_points=short_edge_points,
        valid_depth_pixels=int(len(depth_values)),
    )


def _rejected(
    class_name: str,
    conf: float,
    bbox_xyxy: tuple[int, int, int, int],
    center_px: tuple[int, int],
    rect_points: np.ndarray,
    short_edge_points: np.ndarray,
    n_depth: int,
    reason: str,
) -> GraspPose:
    return GraspPose(
        class_name=class_name,
        conf=conf,
        bbox_xyxy=bbox_xyxy,
        center_px=center_px,
        position=None,
        rotation=None,
        tcp_rotation=None,
        jaw_width_m=0.0,
        object_length_m=0.0,
        angle_deg=0.0,
        rect_points=rect_points,
        short_edge_points=short_edge_points,
        valid_depth_pixels=int(n_depth),
        rejected_reason=reason,
    )


def _normalize(vec: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return None
    return (vec / norm).astype(np.float32)


def _line_from_center(center: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return np.stack([center - 0.5 * vec, center + 0.5 * vec], axis=0).astype(np.float32)


def _refine_grasp_line_from_mask(
    mask: np.ndarray,
    center: np.ndarray,
    short_dir_uv: np.ndarray,
    long_len_px: float,
) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    """Use the mask's central cross-section to refine the short-axis grasp.

    The short-axis direction still comes from the OBB/min-area-rect. We only
    replace the grasp center with the midpoint of the mask's actual thickness
    around the object's median longitudinal slice, more reliable for curved /
    asymmetric shapes such as bananas.
    """
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 32:
        return None

    points = np.column_stack([xs, ys]).astype(np.float32)
    grip_dir_uv = np.array([-short_dir_uv[1], short_dir_uv[0]], dtype=np.float32)
    rel = points - center.reshape(1, 2)
    grip_coord = rel @ grip_dir_uv
    open_coord = rel @ short_dir_uv

    grip_center = float(np.median(grip_coord))
    band_half_width_px = float(np.clip(long_len_px * 0.04, 2.0, 12.0))
    band_mask = np.abs(grip_coord - grip_center) <= band_half_width_px
    if int(np.count_nonzero(band_mask)) < 24:
        band_half_width_px = float(np.clip(long_len_px * 0.08, 4.0, 18.0))
        band_mask = np.abs(grip_coord - grip_center) <= band_half_width_px
    if int(np.count_nonzero(band_mask)) < 24:
        return None

    band_open = open_coord[band_mask]
    open_min = float(np.percentile(band_open, 5.0))
    open_max = float(np.percentile(band_open, 95.0))
    grasp_span_px = open_max - open_min
    if grasp_span_px < 2.0:
        return None

    open_center = 0.5 * (open_min + open_max)
    refined_center = center + grip_center * grip_dir_uv + open_center * short_dir_uv
    short_edge_points = _line_from_center(refined_center, short_dir_uv * grasp_span_px)
    return refined_center.astype(np.float32), short_edge_points, float(grasp_span_px)


def _detection_meta(result: Any, index: int) -> tuple[str, float, tuple[int, int, int, int]]:
    names = getattr(result, "names", {})
    box = result.boxes[index]
    # YoloResult boxes are already numpy → no .cpu(); plain asarray.
    x1, y1, x2, y2 = [int(v) for v in np.asarray(box.xyxy[0])[:4]]
    cls_id = int(np.asarray(box.cls[0]).reshape(-1)[0])
    conf = float(np.asarray(box.conf[0]).reshape(-1)[0])
    label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
    return label, conf, (x1, y1, x2, y2)


def _rect_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    rect = cv2.minAreaRect(contour.astype(np.float32))
    return cv2.boxPoints(rect).astype(np.float32)


def _instance_mask(
    result: Any, index: int, image_shape: tuple[int, int]
) -> Optional[np.ndarray]:
    """Return the resized binary instance mask, or ``None`` if unavailable.

    Shared prefix of :func:`_rect_points` / :func:`_depth_mask`: pull the
    seg mask out of the result, resize to the image grid, threshold at 0.5.
    """
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is None or boxes is None or len(masks.data) != len(boxes):
        return None
    mask = np.asarray(masks.data[index])
    if mask.shape != tuple(image_shape):
        mask = cv2.resize(
            mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return (mask > 0.5).astype(np.uint8)


def _rect_points(
    result: Any,
    index: int,
    image_shape: tuple[int, int],
    bbox_xyxy: tuple[int, int, int, int],
) -> np.ndarray:
    mask = _instance_mask(result, index, image_shape)
    if mask is not None:
        rect = _rect_from_mask(mask)
        if rect is not None:
            return rect

    x1, y1, x2, y2 = bbox_xyxy
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _depth_mask(
    result: Any, index: int, image_shape: tuple[int, int], rect_points: np.ndarray
) -> np.ndarray:
    mask = _instance_mask(result, index, image_shape)
    if mask is not None:
        return mask

    polygon = np.round(rect_points).astype(np.int32)
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask


def _ggcnn_vote(ggcnn, depth_mm, mask, K, plane_angle_deg, plane_width_m, hint_px):
    """Optional consistency vote: None when the refiner is off/silent."""
    if ggcnn is None:
        return None
    try:
        gg = ggcnn.predict(depth_mm, mask, K, center_hint_px=hint_px)
        if gg is None:
            return None
        from .ggcnn_refiner import consistent
        return bool(consistent(plane_angle_deg, plane_width_m, gg))
    except Exception:
        return None


def _project(p_cam: np.ndarray, K: np.ndarray) -> tuple[float, float]:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    z = max(float(p_cam[2]), 1e-6)
    return float(p_cam[0]) * fx / z + cx, float(p_cam[1]) * fy / z + cy


def _top_face_grasp(
    mask: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    up_cam: np.ndarray,
    max_points: int = 1500,
    ransac_iters: int = 80,
    plane_thresh_m: float = 0.008,
    min_inliers: int = 120,
    min_up_alignment: float = 0.85,
    side_out: Optional[list] = None,
) -> Optional[tuple]:
    """Fit the object's TOP face and derive the grasp inside it.

    Returns ``(center_cam, open_axis_cam, approach_cam, width_m, length_m,
    n_inliers)`` or ``None`` (caller falls back to the silhouette path).

    Steps: lift masked valid-depth pixels to a camera-frame cloud (sampled);
    RANSAC up to two candidate planes (largest face first, then the rest);
    keep the candidate whose normal aligns with ``up_cam`` (gravity-up
    expressed in the camera frame — supplied by the caller from the current
    TCP pose and hand-eye, so SIDE faces are rejected by construction); PCA
    of the inliers projected into the plane → minor axis = open (grasp)
    axis, extents (5–95 pct) = width/length; approach = -normal (i.e. press
    onto the face). All numpy, no Open3D — slim-container friendly.
    """
    # MEASUREMENT HYGIENE (real-machine 2026-06-12): the seg mask bleeds a
    # few px onto the background at the silhouette edge; lifted to 3D those
    # points pulled the RANSAC plane into a ~28°-slanted compromise through
    # box-top + bled table pixels and inflated the in-plane extents to
    # 0.11-0.17m on a 0.077m box. Erode the mask (kill edge bleed) and
    # band-pass the depths around the object's median (±0.12m) before any
    # fitting.
    mask_in = cv2.erode(
        (mask > 0).astype(np.uint8), np.ones((7, 7), dtype=np.uint8)
    )
    ys, xs = np.nonzero(mask_in > 0)
    if len(xs) < min_inliers:
        return None
    z = depth_mm[ys, xs].astype(np.float64)
    ok = z > 0
    xs, ys, z = xs[ok], ys[ok], z[ok] / 1000.0
    if len(xs) < min_inliers:
        return None
    z_med = float(np.median(z))
    band = np.abs(z - z_med) <= 0.12
    xs, ys, z = xs[band], ys[band], z[band]
    if len(xs) < min_inliers:
        return None
    if len(xs) > max_points:
        sel = np.random.default_rng(0).choice(len(xs), size=max_points, replace=False)
        xs, ys, z = xs[sel], ys[sel], z[sel]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    pts = np.column_stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z])

    up = up_cam / max(float(np.linalg.norm(up_cam)), 1e-9)
    rng = np.random.default_rng(1)
    remaining = np.ones(len(pts), dtype=bool)
    for _round in range(2):
        idx_pool = np.nonzero(remaining)[0]
        if len(idx_pool) < min_inliers:
            return None
        best_inliers: Optional[np.ndarray] = None
        sub = pts[idx_pool]
        for _ in range(ransac_iters):
            tri = sub[rng.choice(len(sub), size=3, replace=False)]
            n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            nn = float(np.linalg.norm(n))
            if nn < 1e-9:
                continue
            n = n / nn
            d = np.abs((sub - tri[0]) @ n)
            inl = d < plane_thresh_m
            if best_inliers is None or inl.sum() > best_inliers.sum():
                best_inliers = inl
        if best_inliers is None or int(best_inliers.sum()) < min_inliers:
            return None
        inlier_pts = sub[best_inliers]
        # Refined normal via SVD of the inlier covariance.
        centroid = inlier_pts.mean(axis=0)
        _u, _s, vt = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
        normal = vt[2]
        if float(np.dot(normal, up)) < 0:
            normal = -normal
        if float(np.dot(normal, up)) >= min_up_alignment:
            # Top face found: PCA inside the plane.
            in_plane = (inlier_pts - centroid) - np.outer(
                (inlier_pts - centroid) @ normal, normal
            )
            _u2, _s2, vt2 = np.linalg.svd(in_plane, full_matrices=False)
            major, minor = vt2[0], vt2[1]
            major_c = in_plane @ major
            minor_c = in_plane @ minor
            length = float(np.percentile(major_c, 95) - np.percentile(major_c, 5))
            width = float(np.percentile(minor_c, 95) - np.percentile(minor_c, 5))
            if width < 0.005 or length < 0.005:
                return None
            approach = -normal  # press onto the face
            return (
                centroid,
                minor / max(float(np.linalg.norm(minor)), 1e-9),
                approach / max(float(np.linalg.norm(approach)), 1e-9),
                width,
                length,
                int(best_inliers.sum()),
            )
        # Largest plane was a SIDE face (oblique view): remember it as a
        # SIDE-GRASP candidate (jaw closes across its HORIZONTAL in-plane
        # axis) before moving on to the next-largest plane. Only faces that
        # actually FACE the camera are graspable this way.
        if abs(float(np.dot(normal, up))) <= 0.35:
            view_dir = -centroid / max(float(np.linalg.norm(centroid)), 1e-9)
            facing = float(np.dot(normal, view_dir))
            n_cam = normal if facing >= 0 else -normal
            if abs(facing) >= 0.3:
                in_plane2 = (inlier_pts - centroid) - np.outer(
                    (inlier_pts - centroid) @ n_cam, n_cam
                )
                # horizontal in-plane axis = in-plane direction ⊥ up
                horiz = np.cross(n_cam, up)
                hn = float(np.linalg.norm(horiz))
                if hn > 1e-6:
                    horiz = horiz / hn
                    h_coord = in_plane2 @ horiz
                    v_axis = np.cross(n_cam, horiz)
                    v_coord = in_plane2 @ v_axis
                    h_width = float(np.percentile(h_coord, 95) - np.percentile(h_coord, 5))
                    v_len = float(np.percentile(v_coord, 95) - np.percentile(v_coord, 5))
                    if h_width >= 0.005 and side_out is not None:
                        side_out.append((centroid, horiz, n_cam, h_width, v_len,
                                         int(best_inliers.sum())))
        # remove this plane's inliers and try the next-largest candidate.
        remaining[idx_pool[best_inliers]] = False
    return None


def _short_axis_3d(
    depth_mm: np.ndarray,
    K: np.ndarray,
    center: np.ndarray,
    short_dir_uv: np.ndarray,
    grasp_span_px: float,
    inner_frac: float = 0.35,
) -> Optional[np.ndarray]:
    """True 3D open-axis vector across the object's short side, full width.

    Samples median depth at ``center ± inner_frac·span`` along the short axis
    (INSIDE the object — the exact edge pixels often land on background),
    back-projects each with its own depth, and scales the inner separation
    back up to the full span. Returns the full-width 3D vector (length =
    jaw width in metres), or ``None`` when either depth sample is invalid —
    callers then fall back to the legacy fronto-parallel estimate.
    """
    if grasp_span_px < 4.0:
        return None
    h, w = depth_mm.shape
    offsets = (-inner_frac * grasp_span_px, inner_frac * grasp_span_px)
    pts3d = []
    for off in offsets:
        u = float(center[0] + short_dir_uv[0] * off)
        v = float(center[1] + short_dir_uv[1] * off)
        if not (0 <= u < w and 0 <= v < h):
            return None
        z = get_depth_mm(depth_mm, int(round(u)), int(round(v)), 5)
        if z <= 0:
            return None
        pts3d.append(_backproject(u, v, z / 1000.0, K))
    inner_vec = pts3d[1] - pts3d[0]
    inner_norm = float(np.linalg.norm(inner_vec))
    if inner_norm < 1e-6:
        return None
    # inner separation covers 2·inner_frac of the span → scale to full width.
    return (inner_vec / (2.0 * inner_frac)).astype(np.float32)


def _backproject(u: float, v: float, z_m: float, K: np.ndarray) -> np.ndarray:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (u - cx) * z_m / fx
    y = (v - cy) * z_m / fy
    return np.array([x, y, z_m], dtype=np.float32)


def _pixel_vec_to_3d(vec_uv: np.ndarray, z_m: float, K: np.ndarray) -> np.ndarray:
    fx, fy = max(float(K[0, 0]), 1e-6), max(float(K[1, 1]), 1e-6)
    return np.array(
        [float(vec_uv[0]) * z_m / fx, float(vec_uv[1]) * z_m / fy, 0.0], dtype=np.float32
    )


__all__ = [
    "GraspPose",
    "estimate_grasp",
    "estimate_grasps",
    "select_best_grasp",
    "get_depth_mm",
    "detection_count",
]
