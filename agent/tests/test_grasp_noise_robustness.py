"""Tier-A NOISE-ROBUSTNESS regression for BOX grasping (Mac, no device).

This is the deterministic regression that reproduced — and now guards — the
real-machine box-grasp regression of 2026-06-16: under D405-class depth noise
the shape-general arbiter MIS-ROUTED boxes to the non-box descriptor routes
(elongated / cylinder / round), gripping across the wrong jaw axis with a noise-
unstable angle and, at yaw=90, a ~0.083 m "round" jaw blow-up (jaw closed but
nothing held). Clean Tier-A passed because the noise-free shell read cleaner;
only the D405 noise model (axial Gaussian + edge flying-pixels + dropout) inflated
the full-cloud PCA's elongation/roundness enough to trip the wrong route.

The fix (``perception/ordinary_grasp.py``): a box HAS a flat top, so the top-
plane RANSAC fit runs FIRST and, when it holds (the planar-topped gate), the
top/side path OWNS the grasp regardless of the full-cloud descriptor — the
descriptor's non-box routes fire only for genuinely non-planar bodies. The box
grasp angle comes from the robust in-plane top-face PCA, so it is stable across
noise; the descriptor cloud is also outlier-trimmed before its PCA.

What this suite asserts (these DEFINE "fixed"), over
  dims  ∈ {square 0.06³, rect 0.10×0.06×0.08, flat 0.12×0.08×0.05,
           tall 0.06×0.06×0.12}
  yaw   ∈ {0,15,30,45,60,75,90}
  pos   ∈ a couple of (x∈[0.35,0.55], y) work-zone points
  noise × N=5 ``NoiseModel()`` samples (seeds 0..4) per (dims,yaw,pos):

  * ROUTING — every BOX routes to ``top_face`` OR ``side_face``, NEVER
    elongated/cylinder/round/near_square (a 3D box is not a thin/round object).
  * ANGLE STABILITY — for a fixed (dims,yaw,pos) the grasp angle across the 5
    noise samples has circular-MAD ≤ 20° (mod 180° — the jaw axis is a line, not
    a vector). SQUARE-faced boxes are rotationally ambiguous, so for them we
    assert the JAW closes across a face (jaw width ≈ the box face dim) instead of
    a specific angle.
  * JAW WIDTH — ≤ 0.088 m always, and ≈ the box's short graspable face within
    tolerance (no 0.083 "round" blow-ups).
  * No Z_BELOW_TABLE, no over-wide.

Run with: ``uv run --extra rebot --extra dev pytest
tests/test_grasp_noise_robustness.py -q -s`` (``-s`` prints the summary table).
"""

from __future__ import annotations

import numpy as np
import pytest

from ovs_agent.apps.voice_rebot_arm.tools.synthetic_grasp_harness import (
    NoiseModel,
    default_K,
    default_T_cam2base,
    make_detection,
    render_box_depth,
    reachable,
    up_hint_from_extrinsic,
)
from ovs_agent.apps.voice_rebot_arm.perception.ordinary_grasp import (
    estimate_grasps,
    select_best_grasp,
)
from ovs_agent.apps.voice_rebot_arm.perception.transforms import (
    transform_grasp_pose_to_base,
)

JAW_LIMIT = 0.088          # physical max jaw opening (m)
TABLE_Z = 0.05             # base-frame table surface (matches the IK grid /
#                            test_shape_general_grasp.py: an in-envelope height)
Z_EPS = 0.006              # table-floor epsilon (m)
N_NOISE = 5                # noise samples per (dims, yaw, pos)
ANGLE_MAD_MAX = 20.0       # circular-MAD ceiling for non-square boxes (deg)
BOX_ROUTES = {"top_face", "side_face"}

# (name, (Lx, Ly, Lz), short graspable face dim in metres). The graspable face
# is the smaller horizontal footprint extent for a top grasp (or the horizontal
# extent of the visible side face for the tall side grasp) — all here ≤ jaw.
BOXES = [
    ("square", (0.06, 0.06, 0.06), 0.06),
    ("rect", (0.10, 0.06, 0.08), 0.06),
    ("flat", (0.12, 0.08, 0.05), 0.08),
    ("tall", (0.06, 0.06, 0.12), 0.06),
]
YAWS = [0, 15, 30, 45, 60, 75, 90]
POSITIONS = [(0.35, 0.0), (0.45, -0.05), (0.55, 0.0)]


def _is_square_faced(dims: tuple[float, float, float]) -> bool:
    """Square footprint (Lx≈Ly) → rotationally ambiguous top grasp angle."""
    return abs(dims[0] - dims[1]) < 1e-6


def _circular_mad_deg(angles_deg: list[float], period_deg: float = 180.0) -> float:
    """Circular median-absolute-deviation of axis angles, modulo ``period_deg``.

    The grasp/open axis is a LINE (180°-periodic), so 1° and 179° are ~2° apart,
    not 178°. Map the period onto the full circle, take the circular mean as the
    centre, then the median of the wrapped absolute deviations, mapped back.
    """
    a = np.radians(np.asarray(angles_deg, dtype=np.float64) * (360.0 / period_deg))
    centre = np.arctan2(float(np.mean(np.sin(a))), float(np.mean(np.cos(a))))
    dev = np.angle(np.exp(1j * (a - centre)))  # wrapped to (-π, π]
    return float(np.degrees(np.median(np.abs(dev))) * (period_deg / 360.0))


def _base_z(g, T: np.ndarray, insertion: float) -> float:
    grasp6, _pre = transform_grasp_pose_to_base(
        np.asarray(g.position, dtype=np.float64),
        np.asarray(g.tcp_rotation, dtype=np.float64),
        np.asarray(T, dtype=np.float64),
        pregrasp_offset_m=0.08,
        insertion_depth_m=insertion,
    )
    return float(grasp6[2])


def _grasp(dims, yaw_deg, pos, seed, T, K, up):
    pose = (pos[0], pos[1], TABLE_Z, np.radians(yaw_deg))
    depth_mm, mask = render_box_depth(dims, pose, T, K, noise=NoiseModel(seed=seed))
    result = make_detection(mask, K, class_name="box")
    return select_best_grasp(
        estimate_grasps([result], depth_mm, K, depth_quantile=0.5, up_hint_cam=up)
    )


@pytest.fixture(scope="module")
def _rig():
    return default_T_cam2base(), default_K()


def test_box_noise_robustness_sweep(_rig):
    T, K = _rig
    up = up_hint_from_extrinsic(T)

    rows: list[str] = []
    failures: list[str] = []

    for name, dims, face_dim in BOXES:
        square = _is_square_faced(dims)
        for yaw in YAWS:
            for pos in POSITIONS:
                angles: list[float] = []
                jaws: list[float] = []
                methods: list[str] = []
                for seed in range(N_NOISE):
                    g = _grasp(dims, yaw, pos, seed, T, K, up)
                    tag = f"{name} yaw={yaw:>2} pos={pos} seed={seed}"

                    # ── valid grasp ──
                    if g is None or not g.is_valid:
                        failures.append(
                            f"{tag}: INVALID grasp "
                            f"({getattr(g, 'rejected_reason', 'None')})"
                        )
                        continue

                    # ── ROUTING: box must be top/side, never a non-box route ──
                    if g.method not in BOX_ROUTES:
                        failures.append(
                            f"{tag}: MIS-ROUTED to {g.method!r} "
                            f"(jaw {g.jaw_width_m:.4f}, angle {g.angle_deg:.1f}) "
                            f"— a box must be top_face/side_face"
                        )

                    # ── JAW WIDTH: ≤ limit and ≈ the box face dim ──
                    if g.jaw_width_m > JAW_LIMIT:
                        failures.append(
                            f"{tag}: OVER-WIDE jaw {g.jaw_width_m:.4f} > {JAW_LIMIT}"
                        )
                    # the 5-95 pct in-plane extent underreads the true face by a
                    # few mm; allow [face-0.03, face+0.02] — catches a "round"
                    # blow-up (~0.083) on a 0.06 face without being brittle.
                    if not (face_dim - 0.030 <= g.jaw_width_m <= face_dim + 0.020):
                        failures.append(
                            f"{tag}: jaw {g.jaw_width_m:.4f} not ≈ face {face_dim:.3f} "
                            f"(allowed [{face_dim-0.030:.3f},{face_dim+0.020:.3f}])"
                        )

                    # ── Z hygiene: never below the table (bare + committed) ──
                    z_bare = _base_z(g, T, 0.0)
                    z_commit = _base_z(g, T, 0.025)
                    if z_bare < TABLE_Z - Z_EPS or z_commit < TABLE_Z - Z_EPS:
                        failures.append(
                            f"{tag}: Z_BELOW_TABLE bare={z_bare:.4f} "
                            f"commit={z_commit:.4f} (table {TABLE_Z})"
                        )

                    angles.append(float(g.angle_deg))
                    jaws.append(float(g.jaw_width_m))
                    methods.append(g.method)

                if not jaws:
                    continue  # all-invalid already recorded as failures

                mad = _circular_mad_deg(angles) if len(angles) >= 2 else 0.0
                # ── ANGLE STABILITY (non-square only) ──
                if not square and len(angles) >= 2 and mad > ANGLE_MAD_MAX:
                    failures.append(
                        f"{name} yaw={yaw} pos={pos}: angle circular-MAD {mad:.1f}° "
                        f"> {ANGLE_MAD_MAX}° across {len(angles)} noise samples "
                        f"(angles={[round(a,1) for a in angles]})"
                    )
                # ── SQUARE: assert the jaw closes across a face instead ──
                if square:
                    jw = np.asarray(jaws)
                    if not np.all(
                        (jw >= face_dim - 0.030) & (jw <= face_dim + 0.020)
                    ):
                        failures.append(
                            f"{name} yaw={yaw} pos={pos}: square jaw not ≈ face "
                            f"{face_dim:.3f} (jaws={[round(j,4) for j in jaws]})"
                        )

                rows.append(
                    f"{name:6} yaw={yaw:>2} pos=({pos[0]:.2f},{pos[1]:+.2f}) | "
                    f"method={'/'.join(sorted(set(methods))):20} "
                    f"jaw[{min(jaws):.4f},{max(jaws):.4f}] "
                    f"angle-MAD={mad:5.1f}°"
                    + ("  [square: angle immaterial]" if square else "")
                )

    # ── summary table ──
    print("\n=== BOX NOISE-ROBUSTNESS SWEEP (N=5 noise samples/cell) ===")
    print(f"{len(rows)} cells × {N_NOISE} samples = {len(rows) * N_NOISE} grasps")
    for r in rows:
        print("  " + r)
    print("=" * 70)

    assert not failures, (
        f"{len(failures)} box noise-robustness failure(s):\n  "
        + "\n  ".join(failures[:40])
        + ("" if len(failures) <= 40 else f"\n  ... +{len(failures)-40} more")
    )


def test_in_envelope_boxes_reachable(_rig):
    """A representative in-envelope box stays reachable under noise (the fix must
    not produce a geometrically valid but unreachable pose)."""
    T, K = _rig
    up = up_hint_from_extrinsic(T)
    n_reach = 0
    n_total = 0
    for name, dims, _face in BOXES:
        for yaw in (0, 45, 90):
            for seed in range(N_NOISE):
                g = _grasp(dims, yaw, (0.45, 0.0), seed, T, K, up)
                if g is None or not g.is_valid:
                    continue
                n_total += 1
                if reachable(g, T)[0]:
                    n_reach += 1
    # the synthetic observation pose frames these boxes on the IK grid; require
    # the vast majority reachable (a few yaw/face combos can sit at the grid edge).
    assert n_total > 0
    assert n_reach >= int(0.8 * n_total), (
        f"only {n_reach}/{n_total} in-envelope noisy boxes reachable"
    )
