"""Wake-time SLV reconnect policy (2026-06-13).

``reconnect_on_wake: true`` forced a fresh SLV WebSocket on EVERY wake, and the
mic audio during that 6s reconnect window was dropped — the first command right
after waking was lost ("第一次没听到，要再说一次"). Turning it off restores the
health/idle-gated policy: reconnect only when the WS is unhealthy OR idle >30s
(server may have recycled the ASR session), never on a hot, healthy turn.

These lock that decision matrix so it can't silently regress back to
always-reconnect.
"""
from __future__ import annotations

import pytest

from ovs_agent import Config
from ovs_agent.app_base import BaseApp
from ovs_agent.state import ConvState


def _wake_app(*, reconnect_on_wake: bool, healthy: bool, idle_s: float):
    app = BaseApp.__new__(BaseApp)
    app.config = Config(
        system_prompt="SYS",
        pipeline_mode="wake_word",
        reconnect_on_wake=reconnect_on_wake,
    )
    app._state = ConvState.SLEEPING

    class _SLV:
        def __init__(self):
            self.reconnect_count = 0

        def is_healthy(self) -> bool:
            return healthy

        def seconds_since_activity(self) -> float:
            return idle_s

        def is_reconnecting(self) -> bool:
            return False

        async def reconnect(self):
            self.reconnect_count += 1

    app.slv = _SLV()
    app._slv_reconnect_count = 0

    async def _noop_async(*a, **k):
        return None

    app._broadcast = _noop_async  # type: ignore[assignment]
    app._readvertise_after_reconnect = _noop_async  # type: ignore[assignment]
    app._set_state = lambda s: setattr(app, "_state", s)  # type: ignore[assignment]
    app._reset_sleep_timer = lambda: None  # type: ignore[assignment]
    app._arm_wake_command_timeout = lambda: None  # type: ignore[assignment]
    app._play_wake_tone = lambda: None  # type: ignore[assignment]

    class _Audio:
        def arm_for_next_turn(self):
            return None

    app.audio = _Audio()
    return app


@pytest.mark.asyncio
async def test_hot_healthy_wake_does_not_reconnect_when_off():
    """The fix: a healthy, recently-active WS is NOT rebuilt on wake, so the
    first post-wake command isn't dropped in a reconnect window."""
    app = _wake_app(reconnect_on_wake=False, healthy=True, idle_s=5.0)
    await app.wake(source="openwakeword")
    assert app.slv.reconnect_count == 0
    assert app._state == ConvState.IDLE  # still wakes, just no reconnect


@pytest.mark.asyncio
async def test_long_idle_wake_still_reconnects_when_off():
    """Empty-final recovery preserved: after >30s idle the server may have
    recycled the ASR session, so wake still forces a fresh WS."""
    app = _wake_app(reconnect_on_wake=False, healthy=True, idle_s=35.0)
    await app.wake(source="openwakeword")
    assert app.slv.reconnect_count == 1


@pytest.mark.asyncio
async def test_unhealthy_wake_still_reconnects_when_off():
    """A dead/unhealthy WS is always rebuilt regardless of the flag."""
    app = _wake_app(reconnect_on_wake=False, healthy=False, idle_s=5.0)
    await app.wake(source="openwakeword")
    assert app.slv.reconnect_count == 1


@pytest.mark.asyncio
async def test_flag_true_restores_always_reconnect():
    """Locks the old behaviour for anyone who explicitly opts back in."""
    app = _wake_app(reconnect_on_wake=True, healthy=True, idle_s=5.0)
    await app.wake(source="openwakeword")
    assert app.slv.reconnect_count == 1
