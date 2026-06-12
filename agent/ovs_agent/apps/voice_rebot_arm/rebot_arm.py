"""rebot_arm — vendored RebotArm driver for the B601-DM arm.

Vendored from reBot-DevArm-Grasp ``drivers/robot/rebot_arm.py``. The only
substantive change vs the upstream copy is :func:`find_rebot_repo_root`,
which no longer carries hard-coded developer paths (``/home/chlorine/seeed``
etc.). Instead it resolves the ``reBotArm_control_py`` SDK from, in order:

  1. an explicit ``repo_root`` hint (wired from agent config
     ``metadata.actuator.config.repo_root``),
  2. the ``REBOT_REPO_ROOT`` environment variable,
  3. the container's canonical install location
     ``/opt/rebot/reBotArm_control_py``.

Like ``apps/voice_arm/so_arm.py``, the heavy SDK import (``motorbridge`` /
``reBotArm_control_py`` C-extensions) is **deferred to connect time** so this
module imports cleanly on a developer Mac that has no SDK installed. The
:class:`RebotArm` constructor performs the first SDK touch, so a Mac unit
test can instantiate the *actuator* (which holds an unconnected RebotArm)
without tripping ImportError — the SDK is only needed once ``connect()``
runs against real hardware.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


_REBOT_REPO_NAME = "reBotArm_control_py"

# Canonical container install location (see Dockerfile.rebot-arm: the SDK is
# cloned here and pip-installed editable).
_CONTAINER_SDK_ROOT = Path("/opt/rebot")


def _is_rebot_repo_root(path: Path) -> bool:
    return path.is_dir() and (path / _REBOT_REPO_NAME).is_dir()


def find_rebot_repo_root(hint: Optional[str] = None) -> Path:
    """Locate the directory that *contains* ``reBotArm_control_py``.

    Resolution order (first match wins):
      1. ``hint`` (agent config ``metadata.actuator.config.repo_root``),
      2. ``$REBOT_REPO_ROOT`` env var,
      3. the container default ``/opt/rebot``.

    Raises ``FileNotFoundError`` with an actionable message when none match.
    NOTE: the upstream developer-machine fallbacks (``~/seeed``,
    ``/home/chlorine/seeed``, ``<cameraws>/sdk``) are intentionally removed —
    this driver only ever runs inside the rebot-arm container or against a
    config-supplied path.
    """
    candidates: list[Path] = []
    if hint:
        candidates.append(Path(hint).expanduser().resolve())
    env_hint = os.environ.get("REBOT_REPO_ROOT")
    if env_hint:
        candidates.append(Path(env_hint).expanduser().resolve())
    candidates.append(_CONTAINER_SDK_ROOT)

    for p in candidates:
        if _is_rebot_repo_root(p):
            return p

    tried = ", ".join(str(p) for p in candidates) or "(none)"
    raise FileNotFoundError(
        f"Cannot locate {_REBOT_REPO_NAME!r} SDK. Tried: {tried}. "
        "Set metadata.actuator.config.repo_root in the agent YAML, the "
        "REBOT_REPO_ROOT env var, or install the SDK to "
        f"{_CONTAINER_SDK_ROOT / _REBOT_REPO_NAME}."
    )


def ensure_rebot_sdk_in_syspath(hint: Optional[str] = None) -> Path:
    repo = find_rebot_repo_root(hint)
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo


def normalize_channel(channel: str) -> str:
    """Resolve ``channel`` to a serial realpath and validate it.

    The SDK's ``_make_controller`` decides serial-vs-SocketCAN with
    ``channel.startswith("/dev/tty")``. A ``/dev/serial/by-id/...`` symlink
    would be misread as a SocketCAN interface, so we ``realpath`` it first
    (resolving by-id symlinks to their ``/dev/ttyACM*`` target) and then
    require the result to start with ``/dev/tty``. Phase A only supports the
    DM-serial transport — non-tty paths (e.g. SocketCAN ``can0``) are rejected
    with an actionable error.
    """
    resolved = os.path.realpath(str(channel))
    if not resolved.startswith("/dev/tty"):
        raise ValueError(
            f"rebot_arm channel must resolve to a serial device "
            f"(/dev/ttyACM*); got {channel!r} → realpath {resolved!r}. "
            "Pass the realpath /dev/ttyACM* (NOT a /dev/serial/by-id/... "
            "symlink that resolves elsewhere, and NOT a SocketCAN path — "
            "Phase A only supports DM-serial)."
        )
    return resolved


def _write_channel_override_yaml(src_cfg_path: str, channel: str) -> str:
    """Load ``src_cfg_path``, override the top-level ``channel`` field, and
    write the result to a process-level temp file. Returns the temp path.

    Only the ``channel`` field is touched — all other fields (motor configs,
    joints, gains, gripper config) are passed through unchanged. The SDK has
    no ``channel`` kwarg; it only reads it from the yaml, so this override is
    the only way to retarget the bus port.
    """
    with open(src_cfg_path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"rebot_arm cfg yaml {src_cfg_path!r} did not parse to a mapping"
        )
    data["channel"] = channel
    fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="rebot_chan_")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return tmp_path


# ── 夹爪状态机常量 ────────────────────────────────────────────────────────────
_G_MAX_DIST_M      = 0.09
_G_ANGLE_OPEN      = -5.0
_G_OPEN_SOFT_LIMIT = -4.9
_G_ARRIVE_TOL      = 0.12
_G_HARD_STOP_ANGLE = -0.05
_G_TAU_MAX         = 1.5
_G_KP_MOVE         = 5.0
_G_KD_MOVE         = 1.0
_G_OPEN_RATE       = 4.0
_G_CLOSE_TORQUE    = 1.0
_G_KD_CLOSE        = 0.5
_G_STALL_VEL       = 0.05
_G_STARTUP_DIST    = 0.30
_G_KP_HOLD         = 5.0
_G_KD_HOLD         = 1.0
_G_DEFAULT_FORCE   = 0.30
_G_CTRL_RATE       = 500.0


class _GS:
    IDLE    = 0
    OPENING = 1
    CLOSING = 2
    CONTACT = 3
    HOLDING = 4
    HOMING  = 5


class RebotArm:
    """B601-DM arm ↔ high-level interface, with built-in gripper force-control
    state machine.

    Args:
        config_path: source arm.yaml path; None = SDK default
                     (``<repo_root>/config/arm.yaml``). When ``channel`` is
                     given, this yaml is the *source* whose ``channel`` field
                     is overridden into a temp file passed to the SDK.
        urdf_path:   URDF path; None = SDK default
        repo_root:   reBotArm_control_py parent dir; None = auto-search
        channel:     serial port realpath (e.g. ``/dev/ttyACM1``). The SDK has
                     no ``channel`` kwarg — it only reads ``channel`` from the
                     yaml (defaulting to ``/dev/ttyACM0``, the SO-ARM's port).
                     When provided, we override the source arm.yaml's
                     ``channel`` into a temp cfg so the B601-DM connects to the
                     correct bus. ``None`` → use the source yaml's channel
                     verbatim (SDK default ttyACM0).
    """

    # Probe the two layouts the SDK config files can live under, relative to
    # the located repo root: the packaged ``reBotArm_control_py/config/`` first,
    # then the SDK-relative ``config/`` fallback the upstream driver used.
    def _sdk_cfg_path(self, filename: str) -> Optional[str]:
        cand = self._repo_root / _REBOT_REPO_NAME / "config" / filename
        if cand.exists():
            return str(cand)
        cand = self._repo_root / "config" / filename
        if cand.exists():
            return str(cand)
        return None

    # Where the SDK ships its default arm.yaml relative to the located repo
    # root. ``init_gripper`` probes the same two layouts for gripper.yaml.
    def _default_arm_cfg_path(self) -> Optional[str]:
        return self._sdk_cfg_path("arm.yaml")

    def __init__(
        self,
        config_path: Optional[str] = None,
        urdf_path:   Optional[str] = None,
        repo_root:   Optional[str] = None,
        channel:     Optional[str] = None,
    ) -> None:
        # The SDK import lives here (constructor), not at module scope, so the
        # module imports on a Mac without the SDK. Constructing a RebotArm
        # still requires the SDK; the actuator defers construction to
        # connect() to keep the import-only smoke test SDK-free.
        repo = ensure_rebot_sdk_in_syspath(repo_root)
        self._repo_root = repo

        # Channel (if any) is normalized to a serial realpath up front so an
        # invalid path fails fast before any SDK / bus touch.
        self._channel = normalize_channel(channel) if channel else None
        # Track temp cfg files created for channel override so disconnect()
        # can clean them up.
        self._tmp_cfg_paths: list[str] = []

        from reBotArm_control_py.actuator import RobotArm
        from reBotArm_control_py.controllers import ArmEndPos
        from reBotArm_control_py.kinematics import (
            IKSolverParams,
            compute_fk,
            get_end_effector_frame_id,
            load_robot_model,
            pos_rot_to_se3,
        )
        from reBotArm_control_py.kinematics.inverse_kinematics import solve_ik

        # Resolve the source arm.yaml: caller-supplied config_path wins,
        # otherwise the SDK-shipped default. When a channel override is
        # requested we MUST have a concrete source yaml to copy+patch (the SDK
        # accepts no channel kwarg), so fail clearly if it cannot be located.
        cfg = str(config_path) if config_path else None
        if self._channel is not None:
            src_cfg = cfg or self._default_arm_cfg_path()
            if src_cfg is None:
                raise FileNotFoundError(
                    "rebot_arm channel override requires a source arm.yaml: "
                    "pass config_path or install the SDK default at "
                    f"{self._repo_root / _REBOT_REPO_NAME / 'config' / 'arm.yaml'}"
                )
            cfg = _write_channel_override_yaml(src_cfg, self._channel)
            self._tmp_cfg_paths.append(cfg)
        # From here on a failure must NOT leak the channel-override temp file:
        # disconnect() won't run (no object is returned), so clean up inline.
        try:
            self._arm = RobotArm(cfg_path=cfg)

            if urdf_path:
                self._model = load_robot_model(urdf_path=str(urdf_path))
            else:
                self._model = load_robot_model()

            self._data = self._model.createData()
            self._ee_frame_id = get_end_effector_frame_id(self._model)
            self._compute_fk = compute_fk
            self._pos_rot_to_se3 = pos_rot_to_se3
            self._solve_ik = solve_ik
            self._ik_check_params = IKSolverParams(
                max_iter=200, tolerance=1e-4, step_size=0.5, damping=1e-6
            )
        except Exception:
            self._cleanup_tmp_cfgs()
            raise

        self._endpos_ctrl = None
        self._ArmEndPos = ArmEndPos

        self._connected = False

        # Gripper motor (registered onto the arm's existing CAN bus).
        self._gripper_mot  = None
        self._gripper_kp   = _G_KP_HOLD
        self._gripper_kd   = _G_KD_HOLD
        self._gripper_ctrl = None

        # Gripper state machine.
        self._g_state            = _GS.IDLE
        self._g_lock             = threading.Lock()
        self._g_pos              = 0.0
        self._g_vel              = 0.0
        self._g_torq             = 0.0
        self._g_pos_start        = 0.0
        self._g_q_contact        = 0.0
        self._g_contact_elapsed  = 0.0
        self._g_open_q_des       = _G_OPEN_SOFT_LIMIT
        self._g_open_target      = _G_OPEN_SOFT_LIMIT
        self._g_target_force     = _G_DEFAULT_FORCE
        self._g_loop_thread: Optional[threading.Thread] = None
        self._g_loop_running     = False
        self._g_loop_stop        = threading.Event()

    # ── lifecycle ────────────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self, enable: bool = True) -> None:
        self._arm.connect()
        if enable:
            self._arm.enable()
            # SAFETY: once enable() succeeds the motors are energised. If any
            # subsequent step (settle / ArmEndPos start) raises, the outer
            # caller will NOT call disconnect() (it never got a constructed
            # object back from connect), so the arm would be left torqued with
            # no controller. Best-effort disable() then re-raise so the arm
            # ends in a safe, de-energised state.
            try:
                time.sleep(0.5)
                self._endpos_ctrl = self._ArmEndPos(self._arm)
                self._endpos_ctrl.start()
            except Exception:
                self._endpos_ctrl = None
                disable_fn = getattr(self._arm, "disable", None)
                if callable(disable_fn):
                    try:
                        disable_fn()
                    except Exception:
                        pass
                raise
            print("[RebotArm] connected, motors enabled")
        else:
            self._arm._request_and_poll()
            print("[RebotArm] connected, motors stay disabled (read-only)")
        self._connected = True

    def disconnect(self) -> None:
        self._g_stop_loop()
        if self._endpos_ctrl is not None:
            try:
                self._endpos_ctrl.end()
            except Exception:
                pass
            self._endpos_ctrl = None
        # SAFETY: explicitly de-energise the joints before tearing down the
        # bus. The SDK's disconnect() may disable internally, but we add an
        # explicit best-effort disable() so the motors are guaranteed to drop
        # torque even if the SDK path changes.
        disable_fn = getattr(self._arm, "disable", None)
        if callable(disable_fn):
            try:
                disable_fn()
            except Exception:
                pass
        try:
            self._arm.disconnect()
        except Exception:
            pass
        self._cleanup_tmp_cfgs()
        self._connected = False
        print("[RebotArm] disconnected")

    def _cleanup_tmp_cfgs(self) -> None:
        """Unlink any channel-override temp cfg files this instance created."""
        for p in self._tmp_cfg_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass
        self._tmp_cfg_paths = []

    # ── gripper init ───────────────────────────────────────────────────────────

    def init_gripper(self, cfg_path: Optional[str] = None) -> None:
        """Register the gripper motor onto the arm's CAN bus and start the
        force-control state machine."""
        from motorbridge import CallError, Mode
        from reBotArm_control_py.actuator.gripper import load_cfg as load_gripper_cfg

        if cfg_path is None:
            cfg_path = self._sdk_cfg_path("gripper.yaml")
            if cfg_path is None:
                # Neither layout has the file; fall back to the SDK-relative
                # path string so load_cfg surfaces a clear missing-file error.
                cfg_path = str(self._repo_root / "config" / "gripper.yaml")

        gcfg = load_gripper_cfg(cfg_path)
        gc = gcfg["gripper"]

        vendor = gc.vendor
        if vendor not in self._arm._ctrl_map:
            raise RuntimeError(
                f"gripper vendor={vendor!r} differs from arm vendor; cannot "
                "share Controller"
            )
        ctrl = self._arm._ctrl_map[vendor]

        if vendor == "damiao":
            self._gripper_mot = ctrl.add_damiao_motor(gc.motor_id, gc.feedback_id, gc.model)
        elif vendor == "myactuator":
            self._gripper_mot = ctrl.add_myactuator_motor(gc.motor_id, gc.feedback_id, gc.model)
        elif vendor == "robstride":
            self._gripper_mot = ctrl.add_robstride_motor(gc.motor_id, gc.feedback_id, gc.model)
        else:
            raise ValueError(f"unsupported gripper vendor: {vendor!r}")

        self._gripper_kp   = gc.kp
        self._gripper_kd   = gc.kd
        self._gripper_ctrl = ctrl

        # One RLock serializes arm-loop and gripper-loop bus ops.
        if not hasattr(ctrl, "_bus_lock"):
            ctrl._bus_lock = threading.RLock()
        lock = ctrl._bus_lock

        def _wrap(fn, _lock=lock):
            def _locked(*a, **kw):
                with _lock:
                    return fn(*a, **kw)
            return _locked

        if not hasattr(ctrl, "_bus_lock_patched"):
            ctrl.poll_feedback_once = _wrap(ctrl.poll_feedback_once)
            ctrl._bus_lock_patched = True

        if not hasattr(self._arm, "_bus_lock_patched"):
            for jc in self._arm._joints:
                mot = self._arm._motor_map[jc.name]
                for _mattr in ("send_pos_vel", "send_mit", "request_feedback"):
                    if hasattr(mot, _mattr):
                        setattr(mot, _mattr, _wrap(getattr(mot, _mattr)))
            self._arm._bus_lock_patched = True

        try:
            ctrl.enable_all()
            time.sleep(0.3)
        except CallError as e:
            print(f"[RebotArm] gripper enable warning: {e}")
        try:
            self._gripper_mot.ensure_mode(Mode.MIT, 1000)
        except CallError as e:
            raise RuntimeError(f"gripper MIT mode switch failed: {e}") from e

        self._g_start_loop()
        print("[RebotArm] gripper registered on CAN bus, force-control loop started")

    @property
    def has_gripper(self) -> bool:
        return self._gripper_mot is not None

    @property
    def gripper_is_holding(self) -> bool:
        with self._g_lock:
            return self._g_state == _GS.HOLDING

    # ── gripper state-machine internals ──────────────────────────────────────

    def _g_safe_mit(self, pos: float, vel: float, kp: float, kd: float, tau_ff: float = 0.0) -> None:
        pos_cmd  = float(np.clip(pos, _G_OPEN_SOFT_LIMIT, 0.0))
        pos_term = kp * (pos_cmd - self._g_pos) + kd * (-self._g_vel)
        tau_safe = float(np.clip(pos_term + tau_ff, -_G_TAU_MAX, _G_TAU_MAX)) - pos_term
        lock = getattr(self._gripper_ctrl, "_bus_lock", None)
        try:
            with (lock or contextlib.nullcontext()):
                self._gripper_mot.send_mit(pos_cmd, vel, kp, kd, tau_safe)
                self._gripper_mot.request_feedback()
                self._gripper_ctrl.poll_feedback_once()
        except Exception:
            pass

    def _g_tick(self, dt: float) -> None:
        try:
            st = self._gripper_mot.get_state()
            if st is not None:
                self._g_pos  = float(st.pos)
                self._g_vel  = float(st.vel)
                self._g_torq = float(st.torq)
        except Exception:
            pass

        pos = self._g_pos
        vel = self._g_vel

        with self._g_lock:
            s  = self._g_state
            tf = self._g_target_force

        if s == _GS.OPENING:
            with self._g_lock:
                target = self._g_open_target
                self._g_open_q_des = max(self._g_open_q_des - _G_OPEN_RATE * dt, target)
                q = self._g_open_q_des
            self._g_safe_mit(q, 0.0, _G_KP_MOVE, _G_KD_MOVE)
            if abs(pos - target) < _G_ARRIVE_TOL:
                with self._g_lock:
                    self._g_state = _GS.IDLE

        elif s == _GS.CLOSING:
            self._g_safe_mit(0.0, 0.0, 0.0, _G_KD_CLOSE, _G_CLOSE_TORQUE)
            with self._g_lock:
                ps = self._g_pos_start
            if abs(pos - ps) >= _G_STARTUP_DIST:
                if pos > _G_HARD_STOP_ANGLE:
                    with self._g_lock:
                        self._g_state = _GS.IDLE
                elif abs(vel) < _G_STALL_VEL:
                    with self._g_lock:
                        self._g_q_contact       = pos
                        self._g_contact_elapsed = 0.0
                        self._g_state           = _GS.CONTACT

        elif s == _GS.CONTACT:
            with self._g_lock:
                qc = self._g_q_contact
            self._g_safe_mit(qc, 0.0, _G_KP_HOLD, _G_KD_HOLD)
            with self._g_lock:
                self._g_contact_elapsed += dt
                if self._g_contact_elapsed >= 0.02:
                    self._g_state = _GS.HOLDING

        elif s == _GS.HOLDING:
            with self._g_lock:
                qc = self._g_q_contact
            self._g_safe_mit(qc, 0.0, _G_KP_HOLD, _G_KD_HOLD, tf)

        elif s == _GS.HOMING:
            self._g_safe_mit(0.0, 0.0, _G_KP_MOVE, _G_KD_MOVE)
            if abs(pos) < _G_ARRIVE_TOL:
                with self._g_lock:
                    self._g_state = _GS.IDLE

    def _g_ctrl_loop(self) -> None:
        dt = 1.0 / _G_CTRL_RATE
        last = time.perf_counter()
        while not self._g_loop_stop.is_set():
            now = time.perf_counter()
            elapsed = now - last
            if elapsed >= dt:
                last += dt
                self._g_tick(elapsed)
            else:
                time.sleep(1e-4)

    def _g_start_loop(self) -> None:
        if self._g_loop_running:
            return
        self._g_loop_stop.clear()
        self._g_loop_thread = threading.Thread(target=self._g_ctrl_loop, daemon=True)
        self._g_loop_thread.start()
        self._g_loop_running = True

    def _g_stop_loop(self) -> None:
        if not self._g_loop_running:
            return
        self._g_loop_stop.set()
        thread_alive = False
        if self._g_loop_thread is not None:
            self._g_loop_thread.join(timeout=1.0)
            if self._g_loop_thread.is_alive():
                # The 500Hz control thread did NOT exit within the timeout —
                # it may still be touching the gripper motor / CAN bus. It is
                # UNSAFE to send a follow-up soft-stop frame (we'd race the
                # still-running tick on the same bus), so mark the loop as
                # unavailable, log, and leave the shared resources alone.
                thread_alive = True
                print(
                    "[RebotArm] ERROR: gripper control thread did not stop "
                    "within 1.0s; skipping soft-stop frame to avoid racing "
                    "the live thread. Gripper marked unavailable."
                )
                self._g_loop_running = False
                self._gripper_mot = None
                return
            self._g_loop_thread = None
        self._g_loop_running = False
        if not thread_alive and self._gripper_mot is not None:
            try:
                self._gripper_mot.send_mit(self._g_pos, 0.0, 0.0, _G_KD_MOVE, 0.0)
                self._gripper_mot.request_feedback()
                self._gripper_ctrl.poll_feedback_once()
            except Exception:
                pass

    def _g_wait_idle(self, timeout: float = 3.0) -> bool:
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            with self._g_lock:
                if self._g_state == _GS.IDLE:
                    return True
            time.sleep(0.01)
        return False

    # ── gripper public API ──────────────────────────────────────────────────

    def open_gripper(self, distance_m: float = _G_MAX_DIST_M) -> None:
        """Open the gripper (blocking, up to 3s)."""
        if self._gripper_mot is None:
            return
        d = float(np.clip(distance_m, 0.0, _G_MAX_DIST_M))
        target = max((d / _G_MAX_DIST_M) * _G_ANGLE_OPEN, _G_OPEN_SOFT_LIMIT)
        with self._g_lock:
            self._g_open_target = target
            self._g_open_q_des  = self._g_pos
            self._g_state = _GS.OPENING
        if not self._g_wait_idle(3.0):
            # Target not reached — the jaw is blocked (e.g. an open command
            # NARROWER than a held object drives the jaws inward into it).
            # Without this give-up the state machine stays in OPENING and the
            # 500Hz loop keeps pushing into the obstacle at the clamped
            # ~_G_TAU_MAX forever — a motor-overheat hazard. Mirror grasp()'s
            # timeout: force IDLE, then park at the CURRENT position with
            # damping only so the motor stops regulating into the obstacle.
            print(
                "[RebotArm] open_gripper: target not reached within 3s "
                "(jaw blocked?); giving up and parking at current position"
            )
            with self._g_lock:
                self._g_state = _GS.IDLE
            self._g_safe_mit(self._g_pos, 0.0, 0.0, _G_KD_MOVE, 0.0)

    def close_gripper(self) -> None:
        """Pure-torque close (non-blocking)."""
        if self._gripper_mot is None:
            return
        with self._g_lock:
            self._g_pos_start = self._g_pos
            self._g_state = _GS.CLOSING

    def grasp(self, force: Optional[float] = None, timeout: float = 5.0) -> bool:
        """Compliant grasp: close → contact detect → force-hold (blocking)."""
        if self._gripper_mot is None:
            return False
        if force is not None:
            with self._g_lock:
                self._g_target_force = float(np.clip(force, 0.05, _G_TAU_MAX))
        with self._g_lock:
            self._g_pos_start = self._g_pos
            self._g_state = _GS.CLOSING
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            with self._g_lock:
                s = self._g_state
            if s == _GS.HOLDING:
                return True
            if s == _GS.IDLE:
                return False
            time.sleep(0.01)
        with self._g_lock:
            self._g_state = _GS.IDLE
        return False

    def release_gripper(self, timeout: float = 4.0) -> None:
        """Open the gripper and home it (blocking)."""
        if self._gripper_mot is None:
            return
        with self._g_lock:
            self._g_open_q_des = self._g_pos
            self._g_state = _GS.OPENING
        self._g_wait_idle(2.0)
        with self._g_lock:
            self._g_state = _GS.HOMING
        self._g_wait_idle(timeout)

    def get_gripper_state(self) -> tuple:
        """Return (pos_rad, vel_rad_s, torq_nm)."""
        return (self._g_pos, self._g_vel, self._g_torq)

    def set_gripper_zero(self) -> bool:
        """Set the current position as the zero point (pauses the ctrl loop)."""
        if self._gripper_mot is None:
            return False
        self._g_stop_loop()
        # If _g_stop_loop could not stop the thread it nulls _gripper_mot and
        # marks the gripper unavailable; bail out (cannot zero a live/absent
        # motor).
        if self._gripper_mot is None:
            print("[RebotArm] gripper zero aborted: gripper unavailable")
            return False
        from motorbridge import CallError
        ok = False
        try:
            self._gripper_mot.set_zero_position()
            print("[RebotArm] gripper zero set")
            ok = True
        except CallError as e:
            print(f"[RebotArm] gripper zero set failed: {e}")
            ok = False
        finally:
            # ALWAYS restart the control loop — a raise from set_zero_position
            # (e.g. an unexpected non-CallError) must not leave the gripper
            # permanently un-controlled.
            self._g_start_loop()
            with self._g_lock:
                self._g_state = _GS.IDLE
        return ok

    # ── state read ────────────────────────────────────────────────────────────

    def get_tcp_pose(self) -> np.ndarray:
        """Read current TCP pose via FK; returns a (4, 4) homogeneous transform."""
        self._arm._request_and_poll()
        q, _, _ = self._arm.get_state()
        position, rotation, _ = self._compute_fk(self._model, q)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rotation
        T[:3,  3] = position
        return T

    # ── motion control ─────────────────────────────────────────────────────────

    def check_ik(
        self,
        x: float, y: float, z: float,
        roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0,
    ) -> tuple[bool, float]:
        """Solve IK only; send no motion command."""
        self._arm._request_and_poll()
        q_curr, _, _ = self._arm.get_state()
        target = self._pos_rot_to_se3(
            np.array([x, y, z], dtype=np.float64),
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        )
        result = self._solve_ik(
            self._model,
            self._data,
            self._ee_frame_id,
            target,
            q_curr,
            self._ik_check_params,
        )
        return bool(result.success), float(result.error)

    def move_to(
        self,
        x: float, y: float, z: float,
        roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0,
        duration: float = 2.0,
    ) -> bool:
        if self._endpos_ctrl is None:
            raise RuntimeError("arm not connected; call connect() first")
        return bool(self._endpos_ctrl.move_to_traj(
            x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, duration=duration,
        ))

    def wait_motion(self, duration: float, extra: float = 0.6) -> None:
        """Wait for the current TCP-trajectory send thread to finish."""
        if self._endpos_ctrl is None:
            return
        thread = getattr(self._endpos_ctrl, "_send_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=duration + extra + 2.0)
        else:
            time.sleep(duration + extra)

    def safe_home(self, duration: float = 3.0) -> None:
        """Home the arm (all joints to zero)."""
        if self._endpos_ctrl is None:
            raise RuntimeError("arm not connected; call connect() first")
        self._endpos_ctrl.safe_home()

    # ── context manager ─────────────────────────────────────────────────────────

    def __enter__(self) -> "RebotArm":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()


__all__ = [
    "RebotArm",
    "find_rebot_repo_root",
    "ensure_rebot_sdk_in_syspath",
]
