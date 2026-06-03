"""Unit tests for RebotArmActuator (reBot B601-DM, Phase A).

Runs without the reBotArm_control_py SDK / motorbridge — we inject a
``_FakeRebotArm`` in place of the real RebotArm so frame→waypoint
translation, cancellation, the torque gate and factory registration can be
exercised on a developer Mac. Style mirrors test_actuator_actions.py.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Tuple

import pytest

from ovs_agent.actuators.factory import create_actuator
from ovs_agent.apps.voice_rebot_arm.rebot_actuator import (
    RebotArmActuator,
    _make_rebot_arm,
)


# ── fakes ───────────────────────────────────────────────────────────


class _FakeRebotArm:
    """Records move_to / gripper calls; no real bus."""

    def __init__(self) -> None:
        self.moves: List[Dict[str, float]] = []
        self.gripper_calls: List[str] = []
        self.gripper_args: List[Tuple[str, float]] = []  # (call, magnitude)
        self.connected = False
        self.gripper_inited = False
        self.disconnected = False
        self._arm = _FakeLowLevelArm()
        self._holding = False

    # lifecycle
    def connect(self, enable: bool = True) -> None:
        self.connected = True

    def init_gripper(self, cfg_path=None) -> None:
        self.gripper_inited = True

    def disconnect(self) -> None:
        self.disconnected = True
        self.connected = False

    # motion
    def move_to(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0, duration=2.0) -> bool:
        self.moves.append(
            {"x": x, "y": y, "z": z, "roll": roll, "pitch": pitch,
             "yaw": yaw, "duration": duration}
        )
        return True

    def safe_home(self, duration: float = 3.0) -> None:
        self.gripper_calls.append("safe_home")

    # gripper
    def open_gripper(self, distance_m: float = 0.09) -> None:
        self.gripper_calls.append("open")
        self.gripper_args.append(("open", float(distance_m)))

    def close_gripper(self) -> None:
        self.gripper_calls.append("close")
        self.gripper_args.append(("close", 0.0))

    def grasp(self, force=None, timeout: float = 5.0) -> bool:
        self.gripper_calls.append("grasp")
        self.gripper_args.append(("grasp", float(force) if force is not None else 0.0))
        self._holding = True
        return True

    def release_gripper(self, timeout: float = 4.0) -> None:
        self.gripper_calls.append("release")

    @property
    def gripper_is_holding(self) -> bool:
        return self._holding

    def get_gripper_state(self) -> Tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    # observation
    def get_tcp_pose(self):
        import numpy as np
        T = np.eye(4, dtype=float)
        T[0, 3] = 0.30
        T[1, 3] = 0.00
        T[2, 3] = 0.30
        return T


class _FakeLowLevelArm:
    def __init__(self) -> None:
        self.enabled = False

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False


def _make_connected_actuator(**overrides) -> Tuple[RebotArmActuator, _FakeRebotArm]:
    """Build an actuator with a fake arm already injected + torque on."""
    act = RebotArmActuator(channel="/dev/ttyACM1", **overrides)
    fake = _FakeRebotArm()
    act._robot = fake  # noqa: SLF001
    act._torque_state = "on"  # noqa: SLF001
    return act, fake


def _wp(delay: float = 0.05, **fields: float) -> Dict[str, Any]:
    """Build a {joints: {...}, delay} cartesian-waypoint frame."""
    joints: Dict[str, float] = {
        "x": 0.30, "y": 0.00, "z": 0.30,
        "roll": 0.0, "pitch": 0.7, "yaw": 0.0, "gripper": 0.0,
    }
    joints.update({k: float(v) for k, v in fields.items()})
    return {"joints": joints, "delay": delay}


# ── import / construction smoke ─────────────────────────────────────


def test_import_without_sdk() -> None:
    """The module imports + the actuator constructs on a Mac with no SDK."""
    act = RebotArmActuator(channel="/dev/ttyACM1")
    assert act.observation_features().keys() >= {
        "x", "y", "z", "roll", "pitch", "yaw", "gripper"
    }
    # Not connected → torque off, empty obs.
    assert act.torque_enabled is False
    assert act.get_cached_observation() == {}


# ── frame → waypoint translation ────────────────────────────────────


def test_execute_sequence_translates_pose_to_move_to() -> None:
    act, fake = _make_connected_actuator()
    ok = act.execute_sequence([
        _wp(x=0.30, y=0.10, z=0.35),
        _wp(x=0.30, y=-0.10, z=0.35),
    ])
    assert ok is True
    assert len(fake.moves) == 2
    assert fake.moves[0]["y"] == pytest.approx(0.10)
    assert fake.moves[1]["y"] == pytest.approx(-0.10)


def test_execute_sequence_uses_frame_duration_override() -> None:
    act, fake = _make_connected_actuator(move_duration=2.0)
    joints = _wp()["joints"]
    joints["duration"] = 1.25
    act.execute_sequence([{"joints": joints, "delay": 0.05}])
    assert fake.moves[0]["duration"] == pytest.approx(1.25)


def test_gripper_signed_magnitude_open_and_grasp() -> None:
    # +gripper → open to that width (m); -gripper → grasp with |g| force (N·m).
    act, fake = _make_connected_actuator()
    act.execute_sequence([_wp(gripper=0.06)])   # open 6 cm
    act.execute_sequence([_wp(gripper=-0.2)])   # grasp 0.2 N·m
    assert fake.gripper_args == [("open", 0.06), ("grasp", 0.2)]


def test_gripper_open_width_clamped_to_max() -> None:
    # open_distance_m is the max-open clamp; a wider request is clamped.
    act, fake = _make_connected_actuator(open_distance_m=0.09)
    act.execute_sequence([_wp(gripper=0.5)])  # request 0.5 m → clamp to 0.09
    assert fake.gripper_args == [("open", 0.09)]


def test_gripper_grasp_force_clamped_when_configured() -> None:
    # grasp_force is the max-force clamp; a larger |g| is clamped down.
    act, fake = _make_connected_actuator(grasp_force=0.3)
    act.execute_sequence([_wp(gripper=-1.5)])  # request 1.5 → clamp to 0.3
    assert fake.gripper_args == [("grasp", 0.3)]


def test_gripper_zero_holds() -> None:
    act, fake = _make_connected_actuator()
    act.execute_sequence([_wp(gripper=0.0)])
    assert fake.gripper_calls == []


def test_pose_only_frame_skips_gripper_and_gripper_only_skips_move() -> None:
    act, fake = _make_connected_actuator()
    # Gripper-only frame: no x/y/z/roll/pitch/yaw keys → no move_to.
    act.execute_sequence([{"joints": {"gripper": 1.0}, "delay": 0.05}])
    assert fake.moves == []
    assert fake.gripper_calls == ["open"]


# ── cancellation ────────────────────────────────────────────────────


def test_cancel_event_stops_sequence_midway() -> None:
    act, fake = _make_connected_actuator()
    cancel = threading.Event()

    frames = [_wp(y=0.1, delay=0.5), _wp(y=-0.1, delay=0.5), _wp(y=0.0, delay=0.5)]

    # Fire cancel shortly after start so only the first frame's move runs.
    def _trip() -> None:
        time.sleep(0.1)
        cancel.set()

    t = threading.Thread(target=_trip)
    t.start()
    ok = act.execute_sequence(frames, cancel_event=cancel)
    t.join()

    assert ok is True
    # First frame's move_to runs, then the long settle is interrupted, and
    # the loop breaks before frame 2's move.
    assert len(fake.moves) == 1


def test_cancel_before_any_frame_sends_nothing() -> None:
    act, fake = _make_connected_actuator()
    cancel = threading.Event()
    cancel.set()
    ok = act.execute_sequence([_wp(), _wp()], cancel_event=cancel)
    assert ok is True
    assert fake.moves == []


# ── torque gate ─────────────────────────────────────────────────────


def test_execute_sequence_refused_when_torque_off() -> None:
    act, fake = _make_connected_actuator()
    act._torque_state = "off"  # noqa: SLF001
    ok = act.execute_sequence([_wp(y=0.1)])
    assert ok is False
    assert fake.moves == []


def test_set_torque_toggles_state_and_low_level_arm() -> None:
    act, fake = _make_connected_actuator()
    act.set_torque(False)
    assert act.torque_enabled is False
    assert fake._arm.enabled is False  # noqa: SLF001
    act.set_torque(True)
    assert act.torque_enabled is True
    assert fake._arm.enabled is True  # noqa: SLF001


def test_set_torque_raises_when_not_connected() -> None:
    act = RebotArmActuator(channel="/dev/ttyACM1")
    with pytest.raises(RuntimeError):
        act.set_torque(True)


# ── empty / disconnected ────────────────────────────────────────────


def test_execute_sequence_empty_returns_false() -> None:
    act, _fake = _make_connected_actuator()
    assert act.execute_sequence([]) is False


def test_execute_sequence_not_connected_returns_false() -> None:
    act = RebotArmActuator(channel="/dev/ttyACM1")
    assert act.execute_sequence([_wp()]) is False


def test_disconnect_best_effort() -> None:
    act, fake = _make_connected_actuator()
    act.disconnect()
    assert fake.disconnected is True
    assert act._robot is None  # noqa: SLF001
    assert act.torque_enabled is False
    # Idempotent: a second disconnect on an already-cleared actuator is a no-op.
    act.disconnect()


# ── observation cache ───────────────────────────────────────────────


def test_update_cache_reads_tcp_pose() -> None:
    act, _fake = _make_connected_actuator()
    obs = act.update_cache()
    assert obs["x"] == pytest.approx(0.30)
    assert obs["z"] == pytest.approx(0.30)
    assert "gripper" in obs
    # Cached read returns the same without touching the bus.
    assert act.get_cached_observation()["x"] == pytest.approx(0.30)


# ── factory registration ────────────────────────────────────────────


def test_registered_in_factory() -> None:
    # Importing rebot_actuator (done at module top) self-registers the
    # builder. create_actuator must resolve it.
    act = create_actuator("rebot_arm", {"channel": "/dev/ttyACM1"})
    assert isinstance(act, RebotArmActuator)


def test_builder_requires_channel() -> None:
    with pytest.raises(ValueError):
        _make_rebot_arm({})


def test_builder_threads_config_through() -> None:
    act = _make_rebot_arm({
        "channel": "/dev/ttyACM1",
        "repo_root": "/opt/rebot",
        "move_duration": 1.5,
        "grasp_force": 0.4,
        "open_distance_m": 0.05,
    })
    assert act._channel == "/dev/ttyACM1"  # noqa: SLF001
    assert act._repo_root == "/opt/rebot"  # noqa: SLF001


def test_builder_tolerates_empty_string_env_substitution() -> None:
    # "${VAR:-}" with VAR unset substitutes to "" — the builder must treat
    # empty/whitespace as "unset" for every optional field (the deploy bug:
    # grasp_force="" → float("") ValueError disabled the whole arm).
    act = _make_rebot_arm({
        "channel": "/dev/ttyACM1",
        "repo_root": "",
        "config_path": "  ",
        "urdf_path": "",
        "gripper_cfg_path": "",
        "move_duration": "",
        "grasp_force": "",
        "open_distance_m": "",
    })
    assert act._grasp_force is None       # noqa: SLF001
    assert act._repo_root is None         # noqa: SLF001
    assert act._config_path is None       # noqa: SLF001
    assert act._move_duration == 2.0      # noqa: SLF001  (default)
    assert act._open_distance_m == 0.09   # noqa: SLF001  (default)


# ── channel actually reaches the SDK (regression guard) ─────────────


def test_connect_passes_channel_to_rebotarm(monkeypatch) -> None:
    """connect() MUST forward channel to RebotArm, else the SDK falls back to
    its arm.yaml default /dev/ttyACM0 — the SO-ARM's port — and we would drive
    the wrong arm. This is the safety regression that motivated the fix."""
    import ovs_agent.apps.voice_rebot_arm.rebot_actuator as mod

    captured: Dict[str, Any] = {}

    def _spy(**kwargs):
        captured.update(kwargs)
        return _FakeRebotArm()

    monkeypatch.setattr(mod, "RebotArm", _spy)

    act = RebotArmActuator(channel="/dev/ttyACM1", repo_root="/opt/rebot")
    act.connect()

    assert captured.get("channel") == "/dev/ttyACM1", (
        f"connect() did not forward channel to RebotArm; got {captured!r}"
    )
    assert captured.get("repo_root") == "/opt/rebot"
