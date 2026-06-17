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

    # ── TOP-FACE plane fit FIRST (the box detector) ────────────────────
    # Real-machine regression 2026-06-16: the shape-general descriptor (full-cloud
    # PCA) is NOT a reliable box detector under D405 depth noise. A box seen
    # obliquely backprojects to a TWO/THREE-FACE shell (top + visible sides), and
    # the PCA of that shell reads the *box dimensions* as a rod or a disc:
    #   - a 0.10×0.06×0.08 box at yaw=0 reads elongation≈4.2 → "elongated" (jaw
    #     across the wrong axis);
    #   - the same box at yaw=90 reads elongation≈1.1, roundness≈0.94 → "round"
    #     (jaw blows up to ~0.083 m, near the 0.088 limit → nothing held);
    # and the descriptor's angle is noise-driven (clean 98° vs noisy −62°).
    # The OLD ``planarity<=0.04`` gate never fired for these (the full shell is
    # non-planar), so a perfectly graspable box fell through to the descriptor's
    # non-box routes. Clean Tier-A passed only because the noise-free shell read
    # cleaner.
    #
    # The fix: a box HAS a flat top → fit the TOP plane in 3D (RANSAC, normal
    # aligned with up) BEFORE consulting the descriptor. When that fit is good
    # (enough up-aligned inliers, in-plane width within the jaw), the object is
    # planar-topped → KEEP the top/side path REGARDLESS of the full-cloud
    # descriptor's elongation/roundness. The descriptor's elongated/round/
    # cylinder/near-square routes fire ONLY when NO good top plane exists
    # (genuinely curved bodies: banana/orange present no planar top, so this
    # leaves them on the descriptor route, byte-identical to before). The
    # top-plane PCA also gives the box grasp angle from a robust in-plane fit
    # (the existing top/side path), so the angle is stable across noise —
    # unlike the noisy full-cloud descriptor.
    side_cands: list = []
    top = None
    if up_hint_cam is not None:
        top = _top_face_grasp(
            mask, depth_mm, K, np.asarray(up_hint_cam, dtype=np.float64),
            side_out=side_cands,
        )

    # ── SHAPE ARBITER (shape-general 3D descriptor) ────────────────────
    # One backprojected, depth-band-filtered cloud → PCA descriptor that decides
    # the grasp strategy for the NON-box families (banana / orange / bottle), not
    # the brittle 2D min-area-rect short axis. ``desc`` is None when the cloud is
    # too sparse (<200 pts) → the legacy 2D silhouette path below is the fallback.
    desc = None
    if up_hint_cam is not None:
        desc = _shape_descriptor(
            mask, depth_mm, K, np.asarray(up_hint_cam, dtype=np.float64)
        )
    # PLANAR-TOPPED gate (the box guard): a confident, up-aligned top plane (the
    # RANSAC fit in ``_top_face_grasp`` only returns non-None when it finds a
    # plane with >= min_inliers inliers whose normal aligns with up >= 0.85)
    # means a FLAT-TOPPED object — a box. Leave it to the top/side path below
    # REGARDLESS of the full-cloud descriptor's elongation/roundness (that PCA
    # reads the multi-face shell as a rod/disc and misroutes the box).
    #
    # NOTE: the gate does NOT condition on the top-plane WIDTH fitting the jaw.
    # A box too wide to grasp from the top is STILL a planar-topped box, not a
    # banana/orange; its width is handled by the top/side arbitration + the
    # re-observation path below (an over-wide flat plate must trigger reobserve,
    # not be re-interpreted by the descriptor's near-square route — which on a
    # zero-thickness fronto-parallel plane degenerates to a 0 m jaw). This is the
    # behaviour the OLD ``planarity<=0.04 AND top_align>=0.85`` gate produced for
    # a flat plate (planarity≈0, top_align≈1 ⇒ suppressed); the new gate keys off
    # the actual RANSAC top-plane fit, which is the reliable planar-top signal
    # under a multi-face box shell where the full-cloud planarity is high.
    _planar_topped = top is not None
    # A side candidate with a sane (fit-jaw) horizontal extent is the tall-box
    # case where the top is out of view but a vertical face is graspable — also a
    # box, handled by the side path below ⇒ also suppress the descriptor.
    _side_box = any(c[3] <= 0.088 for c in side_cands)
    # A TALL upright box reads as "elongated" (its major axis is the vertical
    # extent) but it is NOT a lying banana/bottle — the existing top/side path
    # owns it (IK-aware camera-ray approach, 8fb88ac over-wide guard). Defer to
    # that path whenever the major axis aligns with gravity-up; the descriptor
    # routes (banana/bottle lying on the table, orange) all have a major axis
    # roughly PERPENDICULAR to up.
    # Only ELONGATED objects have a meaningful major axis; for a round/near-
    # square blob the major axis is arbitrary, so the vertical-major deferral
    # must not apply (it would wrongly bounce an orange to the legacy path).
    _major_is_vertical = False
    if desc is not None and up_hint_cam is not None and desc.elongation >= 1.8:
        up = np.asarray(up_hint_cam, dtype=np.float64)
        up = up / max(float(np.linalg.norm(up)), 1e-9)
        _major_is_vertical = abs(float(np.dot(desc.axes[:, 0], up))) >= 0.70
    if (
        desc is not None
        and not _planar_topped
        and not _side_box
        and not _major_is_vertical
    ):
        g = _descriptor_grasp(
            desc, class_name, conf, bbox_xyxy, rect_points, K
        )
        if g is not None:
            return g

    # ── TOP-FACE plane grasp (preferred when an up-hint is available) ──
    # The 2D-silhouette family (min-area-rect below) merges the box's top and
    # side faces under oblique views, so the "short axis" can measure the
    # wrong physical dimension entirely. Fitting the TOP plane in 3D (RANSAC,
    # candidate whose normal best matches "up") and doing PCA inside that
    # plane measures the real graspable face: metric width, true angle, and
    # a face-normal approach. Falls back to the legacy estimators whenever
    # the fit is not confident (curved objects, sparse depth, no hint).
    # ``top`` / ``side_cands`` were computed ABOVE (the box guard); reuse them.
    if up_hint_cam is not None:
        # Arbitration: TOP grasp when the top face is visible AND its width
        # fits the jaw; otherwise a SIDE grasp on a camera-facing vertical
        # face whose HORIZONTAL extent fits (tall objects whose top is out of
        # view / too wide). Each construction keeps the legacy camera-ray
        # approach (transform sign convention + the measured IK envelope:
        # pitch 0.2-0.9 is the 93-99% feasible band, pure down-press is not).
        # Reject an over-wide TOP fit even when there is NO side candidate to
        # fall back to. A tall box's compromise plane (box-top + upper side
        # fused under the ±0.12m depth band) aligns with up >0.85 and RETURNS
        # as a "top face" with a hugely inflated width (real machine
        # 2026-06-13: 0.270m on a ~0.06m box) at _top_face_grasp's first
        # accepted plane — BEFORE the side-candidate collector runs — so
        # side_cands is empty and the old `and side_cands` guard let that bogus
        # width through to the plausibility gate (rejected → grasp lost). Drop
        # the top fit on width alone; the code below then takes a side
        # candidate if one exists, else falls through to the legacy silhouette
        # / GG-CNN path. Normal flat boxes (top width < the 0.085 jaw limit)
        # never trip this — their path stays byte-identical.
        if top is not None and top[3] > 0.085:
            top = None
        # TALL UPRIGHT objects → force the SIDE grasp. The shape descriptor's
        # verticality (major axis ∥ gravity, elongated, ≥~12cm tall) is STABLE
        # across frames, unlike the per-frame top/legacy routing that
        # intermittently grabs the small/high top ("gripper closed but nothing
        # held") or reads the fused-face silhouette as an over-wide jaw
        # (rejected at plausibility). Real machine 2026-06-17 cycle on a 17cm
        # standing box: side_face HELD through lift+carry; top_face grabbed air
        # and legacy read 0.156m. Drop any top fit so the side path below owns
        # it (a tall box always presents a large camera-facing side face, which
        # _top_face_grasp collects as a side candidate). Normal flat boxes have a
        # horizontal major axis → this never fires → byte-identical.
        _tall_upright = False
        if desc is not None and up_hint_cam is not None and float(desc.elongation) >= 2.5:
            _up = np.asarray(up_hint_cam, dtype=np.float64)
            _up = _up / max(float(np.linalg.norm(_up)), 1e-9)
            _tall_upright = (
                abs(float(np.asarray(desc.axes[:, 0], dtype=np.float64) @ _up)) >= 0.85
                and float(desc.extent_major) >= 0.12
            )
        if _tall_upright and any(c[3] <= 0.085 for c in side_cands):
            top = None
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
                # Same azimuth re-aim as the top path: keep the jaw open-axis on
                # the true horizontal face extent instead of its tilt-biased
                # projection.
                approach_s = _approach_aligned_to_short_axis(
                    approach_s, horiz, np.asarray(up_hint_cam, dtype=np.float64)
                )
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
            # Re-aim the approach AZIMUTH along the box long axis (same pitch)
            # so the jaw aligns with the true horizontal short axis — otherwise
            # the forward camera-ray tilt rotates the jaw off by up to ~36° and
            # the gripper "won't turn its head to face an angled box".
            # GATE: only re-aim when the top face has a CLEARLY determined long
            # axis (length distinctly > width). For a near-square projected
            # footprint (e.g. a box at yaw≈90 viewed far/steep, where
            # foreshortening compresses the long side) the PCA major/minor split
            # is ambiguous and flips under noise — re-aiming along that unstable
            # long axis would amplify the flip. There the old ⊥-approach
            # projection (roll-encoded) is the more stable choice.
            if length_t >= width_t * 1.15:
                approach_t = _approach_aligned_to_short_axis(
                    approach_t, open_axis_t, np.asarray(up_hint_cam, dtype=np.float64)
                )
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

    if _gg_primary is not None and _gg_primary.quality >= 0.25:
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


def _approach_aligned_to_short_axis(
    approach: np.ndarray,
    short_axis: np.ndarray,
    up_cam: np.ndarray,
    max_swing_rad: float = 0.43,
    skip_swing_rad: float = 0.95,
) -> np.ndarray:
    """Rotate ``approach`` in AZIMUTH (capped) so it is perpendicular to the
    (horizontal) short axis, KEEPING its downward steepness (pitch). Camera
    frame; ``up_cam`` is gravity-up in the camera frame.

    Why (sim-verified 2026-06-16): the gripper jaw open-axis is forced into the
    plane ⊥ approach (both here via the grip/open cross-products AND again in
    ``grasp_axes_to_rebot_tcp_rotation``). Keeping the raw camera-ray approach
    (azimuth pointing at the camera, ~50° forward tilt) therefore projects the
    true horizontal short axis off by up to ~36° — worst at intermediate box
    yaw, ~0 at yaw 0/90 — so on the real machine "the gripper won't turn its
    head to face an angled box". Re-aiming the approach azimuth along the box
    LONG axis (⊥ short axis) at the SAME pitch puts the short axis exactly in
    the plane ⊥ approach, so the jaw aligns with an unchanged approach
    steepness. The gripper now yaws to face the box.

    REACHABILITY CAP (sim-verified 2026-06-16): re-aiming the azimuth shows up
    as base YAW of the grasp pose, and the B601-DM's measured IK envelope only
    admits base yaw within ≈±0.6 rad. A full re-aim needs up to ±1.54 rad at
    box-yaw 90 → unreachable. So the swing is capped at ``max_swing_rad`` (the
    residual stays as a roll-projected partial alignment, exactly the old
    behaviour). For the common moderate-angle case (box yaw ≤ ~30°) the cap is
    not hit and the jaw aligns fully; beyond it the gripper aligns as far as the
    arm can physically reach. (Unlike a wrist-roll, swinging the azimuth is the
    ONLY way to beat the ⊥-approach projection limit.)
    """
    up = _normalize(up_cam)
    a = _normalize(approach)
    if up is None or a is None:
        return approach
    up = up.astype(np.float64)
    a = a.astype(np.float64)
    # horizontalise the short axis (drop any up-component from PCA noise)
    short = _normalize(
        np.asarray(short_axis, dtype=np.float64)
        - float(np.dot(np.asarray(short_axis, dtype=np.float64), up)) * up
    )
    if short is None:
        return a.astype(np.float32)
    short = short.astype(np.float64)
    vert = float(np.dot(a, -up))                      # downward steepness component
    hmag = float(np.sqrt(max(0.0, 1.0 - vert * vert)))
    long_axis = _normalize(np.cross(up, short))       # horizontal, ⊥ short axis
    if long_axis is None:
        return a.astype(np.float32)
    long_axis = long_axis.astype(np.float64)
    if float(np.dot(long_axis, a)) < 0.0:
        long_axis = -long_axis                        # keep pointing toward the camera side
    target = _normalize(vert * (-up) + hmag * long_axis)
    if target is None:
        return a.astype(np.float32)
    target = target.astype(np.float64)
    # Cap the azimuth swing (SLERP toward the target by at most max_swing_rad).
    cos_sw = float(np.clip(np.dot(a, target), -1.0, 1.0))
    swing = float(np.arccos(cos_sw))
    # SKIP when full alignment needs a swing far past the reachable cap: the box
    # short axis is then near-parallel to the view azimuth (box yaw ≈ 90°, short
    # side pointing at the arm). A capped re-aim there barely improves the jaw
    # (still tens of degrees off) yet sits near the azimuth singularity where the
    # long-axis direction is noise-sensitive — destabilising angle_deg. Leave the
    # raw camera-ray approach (stable roll-projected alignment, == old behaviour).
    if swing >= skip_swing_rad:
        return a.astype(np.float32)
    if swing <= max_swing_rad or swing < 1e-6:
        return target.astype(np.float32)
    t = max_swing_rad / swing
    sin_sw = np.sin(swing)
    capped = (np.sin((1.0 - t) * swing) * a + np.sin(t * swing) * target) / sin_sw
    capped = _normalize(capped)
    return capped.astype(np.float32) if capped is not None else target.astype(np.float32)


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


@dataclass
class ShapeDescriptor:
    """Shape-general 3D descriptor of the masked object, derived ONCE from a
    backprojected, depth-band-filtered point cloud (same intrinsics / depth-band
    style as :func:`_top_face_grasp`). All grasp-axis routing keys off this so
    the strategy is robust across boxes, elongated (banana), round (orange) and
    cylinder (bottle) — not the brittle 2D min-area-rect short axis.

    Fields (PCA eigenvalues λ1 ≥ λ2 ≥ λ3 of the cloud covariance):
      * ``elongation = λ1/λ2`` — how rod-like (>= 2.2 → elongated).
      * ``planarity  = λ3/(λ1+λ2+λ3)`` — how flat (<= 0.04 → planar/box-top).
      * ``roundness  = λ2/λ1`` — how isotropic in the major plane (round when
        ``elongation`` is small AND ``planarity`` is large).
      * ``axes`` — the three eigenvectors (columns major→minor), camera frame.
      * ``centroid`` — cloud centroid (camera frame, metres).
      * ``extent_major/mid/minor`` — 5–95 % metric extents along each axis.
      * ``top_align`` — |major-plane-normal · up_hint| (1 = face-on top plane).
      * ``spine_bend`` — straightness of the object's spine: the peak-to-peak
        lateral wander of per-axial-slice centroids (in the mid direction),
        normalised by the major extent. ~0 for a straight rod/cylinder
        (bottle), large for a curved body (banana). A single oblique depth view
        sees only a planar shell, so a cross-section-radius CV cannot separate
        cylinder from box/banana — the SPINE CURVATURE is the reliable
        discriminator, so the cylinder route keys off ``spine_bend`` being small.
      * ``n_points`` — cloud size (the legacy 2D fallback fires when < 200).
    """

    elongation: float
    planarity: float
    roundness: float
    axes: np.ndarray
    centroid: np.ndarray
    extent_major: float
    extent_mid: float
    extent_minor: float
    top_align: float
    spine_bend: float
    n_points: int
    # ── table-floor hygiene (z below table) ──
    #  ``up_cam`` — gravity-up expressed in the camera frame (normalised), or
    #    None when no up-hint was supplied (legacy path, no floor clamp).
    #  ``table_proj`` — the cloud's MINIMUM projection onto ``up_cam`` (the
    #    object's footprint on the table). Because base_z = dot(pos, up_cam) +
    #    const (up_cam = R_cam2base.T @ +Z), any grasp point whose up-projection
    #    drops below this sits BELOW the table surface. The descriptor routes
    #    push the grasp point INTO the object (``recenter_depth_m``) which, for a
    #    flat object lying on the table, can shove it under the plane — so
    #    :func:`_pose_from_axes` floors the up-projection at ``table_proj`` (the
    #    same z hygiene the box top/side paths enforce).
    up_cam: Optional[np.ndarray] = None
    table_proj: float = 0.0


def _mask_cloud(
    mask: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    band_m: float = 0.12,
    min_points: int = 200,
    max_points: int = 3000,
    erode_k: int = 7,
    seed: int = 0,
) -> Optional[np.ndarray]:
    """Backproject the eroded mask to a camera-frame 3D cloud (N,3) metres.

    Mirrors :func:`_top_face_grasp`'s measurement hygiene: erode the mask (kill
    seg edge-bleed), keep depth>0, band-pass the depths around the median
    (±``band_m``), subsample to ``max_points``. Returns ``None`` when fewer than
    ``min_points`` survive (caller routes to the legacy 2D fallback).
    """
    mask_in = cv2.erode(
        (mask > 0).astype(np.uint8), np.ones((erode_k, erode_k), dtype=np.uint8)
    )
    ys, xs = np.nonzero(mask_in > 0)
    if len(xs) < min_points:
        return None
    z = depth_mm[ys, xs].astype(np.float64)
    ok = z > 0
    xs, ys, z = xs[ok], ys[ok], z[ok] / 1000.0
    if len(xs) < min_points:
        return None
    z_med = float(np.median(z))
    band = np.abs(z - z_med) <= band_m
    xs, ys, z = xs[band], ys[band], z[band]
    if len(xs) < min_points:
        return None
    if len(xs) > max_points:
        sel = np.random.default_rng(seed).choice(
            len(xs), size=max_points, replace=False
        )
        xs, ys, z = xs[sel], ys[sel], z[sel]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    pts = np.column_stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z])
    return pts


def _shape_descriptor(
    mask: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    up_cam: Optional[np.ndarray],
) -> Optional[ShapeDescriptor]:
    """Compute the shared :class:`ShapeDescriptor`, or ``None`` if the cloud is
    too sparse (< 200 pts) — the caller then uses the legacy 2D path."""
    pts = _mask_cloud(mask, depth_mm, K)
    if pts is None or len(pts) < 200:
        return None
    # ── OUTLIER-ROBUST PCA (noise hardening, 2026-06-16) ───────────────────
    # D405 axial noise + edge "flying pixels" put a heavy tail on the cloud:
    # those few far-from-body points lever the principal axes (the covariance is
    # a SUM of squared deviations, so a handful of outliers at the cloud edge
    # disproportionately inflate λ1 and the elongation/roundness verdict). Trim
    # the extreme points before the PCA so the descriptor reflects the body, not
    # the noise tail: compute the centroid, drop the points whose distance to it
    # exceeds the 95th percentile (a robust radius), then PCA the trimmed set.
    # This is the "require the verdict to be robust to dropping the extreme
    # points" the noise suite checks; it never flips a genuine rod/disc (its
    # extent is intrinsic, not tail-driven) but stops a noisy near-box from
    # reading as elongated. Kept above the 200-pt floor (we only ever drop ~5%).
    c0 = pts.mean(axis=0)
    r = np.linalg.norm(pts - c0, axis=1)
    keep = r <= np.percentile(r, 95.0)
    if int(keep.sum()) >= 200:
        pts = pts[keep]
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    # covariance PCA; eigenvalues ascending → reorder major→minor.
    cov = (centered.T @ centered) / max(len(centered) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    l1, l2, l3 = (float(max(e, 1e-12)) for e in evals)
    elongation = l1 / l2
    planarity = l3 / (l1 + l2 + l3)
    roundness = l2 / l1

    major, mid, minor = evecs[:, 0], evecs[:, 1], evecs[:, 2]
    proj_major = centered @ major
    proj_mid = centered @ mid
    proj_minor = centered @ minor

    def _extent(c: np.ndarray) -> float:
        return float(np.percentile(c, 95) - np.percentile(c, 5))

    extent_major = _extent(proj_major)
    extent_mid = _extent(proj_mid)
    extent_minor = _extent(proj_minor)

    # plane normal of the dominant (major×mid) plane = minor axis; top alignment.
    top_align = 0.0
    up_unit: Optional[np.ndarray] = None
    table_proj = 0.0
    if up_cam is not None:
        up = np.asarray(up_cam, dtype=np.float64)
        up = up / max(float(np.linalg.norm(up)), 1e-9)
        up_unit = up
        top_align = abs(float(np.dot(minor, up)))
        # Object footprint on the table = the cloud's MINIMUM up-projection.
        # base_z = dot(pt, up) + const, so this is the lowest (table) surface;
        # the grasp point's up-projection must never fall below it.
        table_proj = float((pts @ up).min())

    # SPINE STRAIGHTNESS: bin the cloud along the major axis and track the
    # per-slice centroid offset in the MID direction. A straight rod/cylinder
    # keeps that offset ~constant (small peak-to-peak); a banana's spine bends,
    # giving a large wander. Normalised by the major extent so it is scale-free.
    spine_bend = 0.0
    span = float(proj_major.max() - proj_major.min())
    if span > 1e-6:
        edges = np.linspace(proj_major.min(), proj_major.max(), 13)
        slice_mids = []
        for i in range(len(edges) - 1):
            sel = (proj_major >= edges[i]) & (proj_major < edges[i + 1])
            if int(sel.sum()) < 10:
                continue
            slice_mids.append(float(proj_mid[sel].mean()))
        if len(slice_mids) >= 3:
            sm = np.asarray(slice_mids)
            spine_bend = float((sm.max() - sm.min()) / span)

    return ShapeDescriptor(
        elongation=elongation,
        planarity=planarity,
        roundness=roundness,
        axes=evecs.astype(np.float64),
        centroid=centroid.astype(np.float64),
        extent_major=extent_major,
        extent_mid=extent_mid,
        extent_minor=extent_minor,
        top_align=top_align,
        spine_bend=spine_bend,
        n_points=int(len(pts)),
        up_cam=up_unit,
        table_proj=table_proj,
    )


def _pose_from_axes(
    class_name: str,
    conf: float,
    bbox_xyxy: tuple[int, int, int, int],
    position: np.ndarray,
    open_axis_cam: np.ndarray,
    width_m: float,
    length_m: float,
    n_points: int,
    rect_points: np.ndarray,
    K: np.ndarray,
    method: str,
    recenter_depth_m: float = 0.0,
    up_cam: Optional[np.ndarray] = None,
    table_proj: float = 0.0,
) -> Optional[GraspPose]:
    """Build a :class:`GraspPose` from a 3D grasp point + jaw-closing (open)
    axis, using the legacy camera-ray approach convention (approach = toward the
    camera; ``grasp_axes_to_rebot_tcp_rotation`` negates it into tool-forward).

    ``recenter_depth_m`` pushes the grasp point AWAY from the camera (into the
    object) by that distance along the view ray. A single depth view only sees
    the near SHELL, so the cloud centroid sits ~half a diameter in front of the
    true body axis; pushing it back by half the cross-section diameter recovers
    the body-centred grasp point. Zero for routes whose point is already on the
    body (box top-face path never calls this).

    Returns ``None`` if the axis construction degenerates OR the jaw width is
    over the 0.088 m physical limit (same over-wide rejection as every route).
    """
    if width_m > 0.088:
        return None
    position = np.asarray(position, dtype=np.float64)
    approach = _normalize(-position)
    if approach is None:
        approach = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    if recenter_depth_m > 0.0:
        # approach points TOWARD the camera; -approach goes into the object.
        position = position - approach * float(recenter_depth_m)
    # ── Z-FLOOR (table hygiene) ──────────────────────────────────────────
    # base_z = dot(pos, up_cam) + const, so the grasp point's up-projection
    # must stay at/above the object's footprint on the table (``table_proj``).
    # The recenter push (along -approach, into the object) can drop a flat
    # object's grasp point UNDER the table plane; floor the up-projection so
    # the emitted point never sits below the table surface — the same z
    # hygiene the box top-face (z-bite floor) and side-face (gz gate) enforce.
    # The margin matches the box top-face z-bite floor (≈25mm above the
    # table): the downstream pick (``transform_grasp_pose_to_base`` with
    # ``insertion_depth_m``≈0.025) pushes the committed point ANOTHER ~25mm
    # along the approach INTO the object, so a grasp floored only a few mm
    # above the footprint still lands below the plane after insertion. Flooring
    # the grasp point a full insertion-depth (25mm) above the object footprint
    # keeps the committed point at/above the table for flat objects.
    if up_cam is not None:
        up = np.asarray(up_cam, dtype=np.float64)
        up = up / max(float(np.linalg.norm(up)), 1e-9)
        floor_proj = float(table_proj) + 0.025  # 25mm above the footprint
        cur_proj = float(np.dot(position, up))
        if cur_proj < floor_proj:
            position = position + up * (floor_proj - cur_proj)
    # project the open axis to be ⊥ the approach (jaw plane).
    open_axis = np.asarray(open_axis_cam, dtype=np.float64)
    open_axis = open_axis - float(np.dot(open_axis, approach)) * approach
    open_axis = _normalize(open_axis)
    if open_axis is None:
        return None
    if open_axis[0] < 0:
        open_axis = -open_axis
    grip_axis = _normalize(np.cross(open_axis, approach))
    open_axis = _normalize(np.cross(approach, grip_axis))
    if grip_axis is None or open_axis is None:
        return None
    rotation = np.column_stack([grip_axis, open_axis, approach]).astype(np.float32)
    tcp_rotation = grasp_axes_to_rebot_tcp_rotation(
        rotation[:, 0], rotation[:, 1], rotation[:, 2]
    ).astype(np.float32)
    u, v = _project(position, K)
    short_uv = _line_from_center(
        np.array([u, v], dtype=np.float32),
        np.array([open_axis[0], open_axis[1]], dtype=np.float32) * 40.0,
    )
    return GraspPose(
        class_name=class_name,
        conf=conf,
        bbox_xyxy=bbox_xyxy,
        center_px=(int(round(u)), int(round(v))),
        position=position.astype(np.float32),
        rotation=rotation,
        tcp_rotation=tcp_rotation,
        jaw_width_m=float(width_m),
        object_length_m=float(length_m),
        angle_deg=float(np.degrees(np.arctan2(open_axis[1], open_axis[0]))),
        rect_points=rect_points,
        short_edge_points=short_uv,
        valid_depth_pixels=int(n_points),
        method=method,
    )


def _descriptor_grasp(
    desc: ShapeDescriptor,
    class_name: str,
    conf: float,
    bbox_xyxy: tuple[int, int, int, int],
    rect_points: np.ndarray,
    K: np.ndarray,
) -> Optional[GraspPose]:
    """Route the shape descriptor to a grasp axis for the non-planar shapes:
    elongated/curved, round, cylinder, and the near-square disambiguation.

    Returns ``None`` to fall through (planar shapes — handled by the existing
    top/side path before this is ever called — or a degenerate construction).
    The jaw always closes across the object's narrowest graspable dimension; the
    width is taken from the corresponding 3D extent and is hard-capped at
    0.088 m by :func:`_pose_from_axes`.
    """
    major = desc.axes[:, 0]
    mid = desc.axes[:, 1]
    minor = desc.axes[:, 2]
    pos = desc.centroid

    elong = desc.elongation
    planar = desc.planarity

    # ── CYLINDER (bottle): rod-like with a STRAIGHT spine → grasp axis closes
    # the jaw across the DIAMETER (the visible cross-section width = the mid
    # extent). Keyed off ``spine_bend`` being small: a single oblique depth view
    # collapses a cylinder to a planar shell, so a cross-section-radius CV cannot
    # tell cylinder from box; the straight (vs banana-curved) spine is the
    # reliable cue. Checked BEFORE the elongated route so a bottle gets the
    # cylinder label (the grasp axis is identical — jaw across the body width).
    if elong >= 1.8 and desc.spine_bend < 0.06:
        open_axis, diameter = mid, desc.extent_mid
        return _pose_from_axes(
            class_name, conf, bbox_xyxy, pos, open_axis,
            width_m=diameter, length_m=desc.extent_major,
            n_points=desc.n_points, rect_points=rect_points, K=K,
            method="cylinder", recenter_depth_m=0.5 * desc.extent_mid,
            up_cam=desc.up_cam, table_proj=desc.table_proj,
        )

    # ── ELONGATED / CURVED (banana): jaw closes ACROSS the minor cross-section
    # of the body — i.e. the grasp (open) axis is the MID axis (the wider of the
    # two short axes) so the jaw spans the body thickness, gripping perpendicular
    # to the long (major) axis.
    # Evidence bar raised 2.2 → 2.6 (noise hardening, 2026-06-16): a box's
    # multi-face shell under D405 noise read elongation up to ~1.9 on a 1.67
    # aspect box, and the OUTLIER-TRIMMED descriptor + planar-topped gate already
    # keep boxes off this route entirely (the descriptor fires 0× for boxes in
    # the 840-case noise suite). 2.6 is belt-and-suspenders: it still fires for
    # genuine rods (banana ≈ 18, bottle ≈ 11 — enormous headroom) but refuses a
    # borderline near-box should it ever reach here (top-plane RANSAC narrowly
    # failing). Tuned against the noise suite, not clean depth.
    if elong >= 2.6:
        if desc.extent_mid >= desc.extent_minor:
            open_axis, width = mid, desc.extent_mid
        else:
            open_axis, width = minor, desc.extent_minor
        return _pose_from_axes(
            class_name, conf, bbox_xyxy, pos, open_axis,
            width_m=width, length_m=desc.extent_major,
            n_points=desc.n_points, rect_points=rect_points, K=K,
            method="elongated", recenter_depth_m=0.5 * width,
            up_cam=desc.up_cam, table_proj=desc.table_proj,
        )

    # ── ROUND (orange): isotropic blob → physically symmetric, no preferred
    # jaw angle. Grasp the centroid; jaw axis is the smallest in-plane extent
    # (reachability/symmetry makes the exact angle immaterial). Width = the
    # representative diameter (the mid extent — robust to the planar squash).
    # NOTE: the spec's ``planarity>0.06`` was tuned for a fuller cloud; a single
    # oblique depth view sees only the sphere's CAP, which reads ~0.05, so the
    # gate is relaxed to 0.045. ``elongation<1.35`` already separates round from
    # the rods; this only distinguishes a round blob from a flat near-square
    # plate (lower planarity) for the method LABEL — both grasp the centroid.
    if elong < 1.35 and planar > 0.045:
        width = max(desc.extent_mid, desc.extent_minor)
        return _pose_from_axes(
            class_name, conf, bbox_xyxy, pos, mid,
            width_m=width, length_m=desc.extent_major,
            n_points=desc.n_points, rect_points=rect_points, K=K,
            method="round", recenter_depth_m=0.5 * width,
            up_cam=desc.up_cam, table_proj=desc.table_proj,
        )

    # ── NEAR-SQUARE (ambiguous, not round): don't trust the 2D short axis —
    # close the jaw across the genuinely smaller 3D extent of the two short axes.
    if elong < 1.35:
        if desc.extent_mid <= desc.extent_minor:
            open_axis, width = mid, desc.extent_mid
        else:
            open_axis, width = minor, desc.extent_minor
        return _pose_from_axes(
            class_name, conf, bbox_xyxy, pos, open_axis,
            width_m=width, length_m=desc.extent_major,
            n_points=desc.n_points, rect_points=rect_points, K=K,
            method="near_square", recenter_depth_m=0.5 * width,
            up_cam=desc.up_cam, table_proj=desc.table_proj,
        )

    return None


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
