"""Tests for GraspPlugin dispatch guards (Phase B safety).

SDK-free: we stub the ArmPlugin / actuator / arm and the perception init so we
can exercise the torque gate, the re-entry guard, and the stop() lifecycle on
a developer Mac. We never touch onnxruntime / camera / CAN bus.
"""
from __future__ import annotations

import asyncio
import threading

import pytest

from ovs_agent.apps.voice_rebot_arm.grasp_plugin import GraspPlugin
from ovs_agent.tools import ToolRegistry


# ── fakes ───────────────────────────────────────────────────────────


class _FakeActuator:
    def __init__(self, torque: bool = True) -> None:
        self.torque_enabled = torque
        self.robot = object()  # non-None → "connected"


class _FakeArmPlugin:
    def __init__(self, actuator) -> None:
        self.arm = actuator


class _FakeApp:
    def __init__(self) -> None:
        self.tool_registry = None
        self.session = None
        self.plugins = []


def _make_plugin(torque: bool = True) -> tuple[GraspPlugin, _FakeActuator]:
    app = _FakeApp()
    plugin = GraspPlugin(app)
    actuator = _FakeActuator(torque=torque)
    plugin._arm_plugin = _FakeArmPlugin(actuator)  # noqa: SLF001
    # Skip the real perception init (onnxruntime / camera).
    plugin._ensure_perception = lambda: None  # type: ignore[assignment]
    return plugin, actuator


# ── torque gate (item 2) ────────────────────────────────────────────


def test_dispatch_refused_when_torque_off() -> None:
    plugin, _ = _make_plugin(torque=False)
    res = asyncio.run(plugin._dispatch_grasp("banana"))  # noqa: SLF001
    assert res["started"] is False
    assert res["error"] == "torque disabled"


# ── re-entry guard (item 8) ─────────────────────────────────────────


def test_dispatch_rejects_reentry_while_running() -> None:
    plugin, _ = _make_plugin(torque=True)

    # First dispatch: stub the runner so the task stays "in flight".
    gate = threading.Event()

    async def _slow_runner() -> dict:
        # Block until the test releases it.
        while not gate.is_set():
            await asyncio.sleep(0.01)
        return {"success": True}

    async def _drive() -> tuple[dict, dict]:
        # Patch run via monkeypatching the _runner path: easiest is to replace
        # _dispatch_grasp's worker by pre-seeding a live task.
        plugin._grasp_task = asyncio.create_task(_slow_runner())  # noqa: SLF001
        # Second dispatch must be rejected because one is in flight.
        second = await plugin._dispatch_grasp("bottle")  # noqa: SLF001
        gate.set()
        await plugin._grasp_task  # noqa: SLF001
        return second, {}

    second, _ = asyncio.run(_drive())
    assert second["started"] is False
    assert second["error"].startswith("already_running")


def test_dispatch_allows_after_previous_done() -> None:
    # A COMPLETED previous task must NOT trigger the re-entry guard. We make
    # _ensure_perception raise a sentinel so dispatch returns right AFTER the
    # guard — proving the guard let us through (error is the perception error,
    # not "already_running").
    plugin, _ = _make_plugin(torque=True)

    async def _already_done() -> dict:
        return {"success": True}

    def _boom_perception() -> None:
        raise RuntimeError("perception-sentinel")

    plugin._ensure_perception = _boom_perception  # type: ignore[assignment]

    async def _drive() -> dict:
        done = asyncio.create_task(_already_done())
        await done
        plugin._grasp_task = done  # noqa: SLF001 — completed task
        return await plugin._dispatch_grasp("banana")  # noqa: SLF001

    res = asyncio.run(_drive())
    assert res["started"] is False
    assert res["error"] == "perception-sentinel"   # passed the re-entry guard


# ── empty / unavailable arm ─────────────────────────────────────────


def test_dispatch_empty_object_rejected() -> None:
    plugin, _ = _make_plugin(torque=True)
    res = asyncio.run(plugin._dispatch_grasp("   "))  # noqa: SLF001
    assert res["started"] is False
    assert "empty" in res["error"]


def test_dispatch_arm_not_connected() -> None:
    plugin, actuator = _make_plugin(torque=True)
    actuator.robot = None  # not connected
    res = asyncio.run(plugin._dispatch_grasp("banana"))  # noqa: SLF001
    assert res["started"] is False
    assert res["error"] == "arm not connected"


def test_setup_does_not_register_tool_when_disabled() -> None:
    app = _FakeApp()
    app.tool_registry = ToolRegistry()
    plugin = GraspPlugin(app, {"enabled": False})

    assert plugin.setup() is True
    assert not app.tool_registry.has("grasp_object")


# ── stop() lifecycle (item 9) ───────────────────────────────────────


def test_stop_sets_cancel_waits_and_closes_camera() -> None:
    plugin, _ = _make_plugin(torque=True)

    closed = {"n": 0}

    class _FakeCam:
        def close(self) -> None:
            closed["n"] += 1

    plugin._camera = _FakeCam()  # noqa: SLF001

    observed_cancel = {"set": False}

    async def _worker() -> dict:
        # Simulate a grasp that watches the cancel event and returns promptly.
        for _ in range(200):
            if plugin._cancel_event.is_set():  # noqa: SLF001
                observed_cancel["set"] = True
                return {"cancelled": True}
            await asyncio.sleep(0.01)
        return {"cancelled": False}

    async def _drive() -> None:
        plugin._grasp_task = asyncio.create_task(_worker())  # noqa: SLF001
        await asyncio.sleep(0.05)
        await plugin.stop()

    asyncio.run(_drive())
    assert observed_cancel["set"] is True       # cancel event was set
    assert plugin._grasp_task is None           # noqa: SLF001 — cleared
    assert closed["n"] == 1                      # camera closed exactly once
    assert plugin._camera is None                # noqa: SLF001


# ── put_down dispatch (place back where picked up) ──────────────────


class _HoldingRobot:
    """Stub robot whose gripper_is_holding is a PROPERTY (like the real arm)."""

    def __init__(self, holding) -> None:
        self._holding = holding

    @property
    def gripper_is_holding(self):
        return self._holding


def test_put_down_rejected_when_nothing_held() -> None:
    plugin, actuator = _make_plugin(torque=True)
    actuator.robot = _HoldingRobot(holding=False)
    res = asyncio.run(plugin._dispatch_put_down())  # noqa: SLF001
    assert res["started"] is False
    assert res["error"] == "nothing held"


def test_put_down_proceeds_with_recorded_grasp_even_if_not_holding() -> None:
    """A recorded grasp admits put_down regardless of the holding flag — the
    flag can be momentarily False mid-transition (or after a misheard gripper
    command), and replaying the place-back is harmless even when empty."""
    plugin, actuator = _make_plugin(torque=True)
    actuator.robot = _HoldingRobot(holding=False)
    plugin._last_grasp = {  # noqa: SLF001
        "success": True,
        "grasp_pose": [0.40, 0.0, 0.08, 0.0, 0.0, 0.0],
        "pregrasp_pose": [0.38, 0.0, 0.16, 0.0, 0.0, 0.0],
        "open_distance_m": 0.089,
    }

    import ovs_agent.apps.voice_rebot_arm.grasp_service as gs

    orig = gs.run_put_down_once
    gs.run_put_down_once = lambda **kwargs: {"success": True, "released": True}
    try:
        async def _drive() -> dict:
            res = await plugin._dispatch_put_down()  # noqa: SLF001
            await plugin._grasp_task  # noqa: SLF001
            return res

        res = asyncio.run(_drive())
    finally:
        gs.run_put_down_once = orig

    assert res["started"] is True
    assert res["used_recorded_pose"] is True


def test_put_down_rejected_when_torque_off() -> None:
    plugin, _ = _make_plugin(torque=False)
    res = asyncio.run(plugin._dispatch_put_down())  # noqa: SLF001
    assert res["started"] is False
    assert res["error"] == "torque disabled"


def test_on_grasp_done_records_last_grasp_for_put_down() -> None:
    plugin, _ = _make_plugin(torque=True)

    async def _grasp_result() -> dict:
        return {
            "success": True,
            "grasp_pose": [0.40, 0.0, 0.08, 0.0, 0.0, 0.0],
            "pregrasp_pose": [0.38, 0.0, 0.16, 0.0, 0.0, 0.0],
            "open_distance_m": 0.089,
        }

    async def _drive() -> None:
        task = asyncio.create_task(_grasp_result())
        await task
        plugin._on_grasp_done(task)  # noqa: SLF001

    asyncio.run(_drive())
    assert plugin._last_grasp is not None  # noqa: SLF001
    assert plugin._last_grasp["grasp_pose"][0] == 0.40  # noqa: SLF001


def test_on_grasp_done_ignores_failed_and_search_results() -> None:
    plugin, _ = _make_plugin(torque=True)

    async def _failed() -> dict:
        return {"success": False, "error": "no valid grasp"}

    async def _search() -> dict:
        return {"found": True, "position_base": [0.4, 0.0, 0.05]}

    async def _drive() -> None:
        for coro in (_failed(), _search()):
            task = asyncio.create_task(coro)
            await task
            plugin._on_grasp_done(task)  # noqa: SLF001

    asyncio.run(_drive())
    assert plugin._last_grasp is None  # noqa: SLF001


def test_put_down_uses_recorded_pose_and_clears_it_on_success() -> None:
    plugin, actuator = _make_plugin(torque=True)
    actuator.robot = _HoldingRobot(holding=True)
    plugin._last_grasp = {  # noqa: SLF001
        "success": True,
        "grasp_pose": [0.40, 0.0, 0.08, 0.0, 0.0, 0.0],
        "pregrasp_pose": [0.38, 0.0, 0.16, 0.0, 0.0, 0.0],
        "open_distance_m": 0.089,
    }

    seen = {}

    def _fake_run_put_down_once(**kwargs):
        seen.update(kwargs)
        return {"success": True, "released": True}

    import ovs_agent.apps.voice_rebot_arm.grasp_service as gs

    orig = gs.run_put_down_once
    gs.run_put_down_once = _fake_run_put_down_once
    try:
        async def _drive() -> dict:
            res = await plugin._dispatch_put_down()  # noqa: SLF001
            await plugin._grasp_task  # noqa: SLF001
            return res

        res = asyncio.run(_drive())
    finally:
        gs.run_put_down_once = orig

    assert res["started"] is True
    assert res["used_recorded_pose"] is True
    assert seen["grasp_pose"] == [0.40, 0.0, 0.08, 0.0, 0.0, 0.0]
    assert seen["pregrasp_pose"] == [0.38, 0.0, 0.16, 0.0, 0.0, 0.0]
    assert seen["open_distance_m"] == 0.089
    # consumed on success → next put_down falls back to place_pose.
    assert plugin._last_grasp is None  # noqa: SLF001


# ── per-class force policy (Level 1) + adaptive flag (Level 2) ──────


def _capture_run_kwargs(plugin, actuator, target="box"):
    """Dispatch a grasp with run_grasp_once stubbed; return its kwargs."""
    import ovs_agent.apps.voice_rebot_arm.grasp_service as gs

    seen = {}

    def _fake_run(tgt, **kwargs):
        seen["target"] = tgt
        seen.update(kwargs)
        return {"success": True, "grasp_pose": [0.4, 0, 0.1, 0, 0, 0]}

    orig = gs.run_grasp_once
    gs.run_grasp_once = _fake_run
    try:
        async def _drive():
            res = await plugin._dispatch_grasp(target)  # noqa: SLF001
            await plugin._grasp_task  # noqa: SLF001
            return res

        res = asyncio.run(_drive())
    finally:
        gs.run_grasp_once = orig
    return res, seen


def test_configured_class_uses_fixed_force_no_adaptive() -> None:
    app = _FakeApp()
    plugin = GraspPlugin(app, {
        "yolo_classes": ["box", "apple"],
        "grasp_force": 0.8,
        "grasp_force_by_class": {"box": 0.65},
    })
    actuator = _FakeActuator(torque=True)
    plugin._arm_plugin = _FakeArmPlugin(actuator)  # noqa: SLF001
    plugin._ensure_perception = lambda: None  # type: ignore[assignment]

    res, seen = _capture_run_kwargs(plugin, actuator, target="box")
    assert res["started"] is True
    assert seen["grasp_force"] == 0.65
    assert seen["adaptive_force"] is False


def test_unconfigured_class_gets_adaptive_with_global_cap() -> None:
    app = _FakeApp()
    plugin = GraspPlugin(app, {
        "yolo_classes": ["box", "apple"],
        "grasp_force": 0.8,
        "grasp_force_by_class": {"box": 0.65},
    })
    actuator = _FakeActuator(torque=True)
    plugin._arm_plugin = _FakeArmPlugin(actuator)  # noqa: SLF001
    plugin._ensure_perception = lambda: None  # type: ignore[assignment]

    res, seen = _capture_run_kwargs(plugin, actuator, target="apple")
    assert res["started"] is True
    assert seen["adaptive_force"] is True
    # ceiling comes from the global grasp_force config
    assert seen["grasp_force"] == 0.8


def test_no_by_class_config_means_adaptive_for_everything() -> None:
    app = _FakeApp()
    plugin = GraspPlugin(app, {"yolo_classes": ["box"], "grasp_force": 0.7})
    actuator = _FakeActuator(torque=True)
    plugin._arm_plugin = _FakeArmPlugin(actuator)  # noqa: SLF001
    plugin._ensure_perception = lambda: None  # type: ignore[assignment]

    _, seen = _capture_run_kwargs(plugin, actuator, target="box")
    assert seen["adaptive_force"] is True
