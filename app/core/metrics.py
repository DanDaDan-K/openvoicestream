"""Week 1 in-process metrics stub.

Lightweight counters/gauges that support the session limiter and readiness
probe. No Prometheus dependency; Week 2 will swap the backing store while
keeping the public API stable.

Naming follows ``ovs_<noun>_<verb>[_total]`` (see
``docs/specs/prod-hardening-week1.md`` §Metrics Naming Convention).
"""

from __future__ import annotations

import threading
from typing import Dict


_lock = threading.Lock()

# Gauges
_sessions_active: int = 0

# Counters
_sessions_rejected_total: Dict[str, int] = {}
_auth_rejected_total: Dict[str, int] = {}


def inc_sessions_active() -> int:
    """Increment the live-session gauge. Returns the post-increment value."""
    global _sessions_active
    with _lock:
        _sessions_active += 1
        return _sessions_active


def dec_sessions_active() -> int:
    """Decrement the live-session gauge, clamped at 0.

    Returns the post-decrement value. Never goes negative even on double-release.
    """
    global _sessions_active
    with _lock:
        if _sessions_active > 0:
            _sessions_active -= 1
        return _sessions_active


def get_sessions_active() -> int:
    with _lock:
        return _sessions_active


def inc_sessions_rejected(reason: str) -> int:
    """Bump ``ovs_sessions_rejected_total{reason=...}``.

    Week 1 reasons: ``"http"`` and ``"ws"``.
    """
    with _lock:
        _sessions_rejected_total[reason] = _sessions_rejected_total.get(reason, 0) + 1
        return _sessions_rejected_total[reason]


def get_sessions_rejected(reason: str | None = None) -> int | Dict[str, int]:
    with _lock:
        if reason is None:
            return dict(_sessions_rejected_total)
        return _sessions_rejected_total.get(reason, 0)


def inc_auth_rejected(endpoint: str) -> int:
    with _lock:
        _auth_rejected_total[endpoint] = _auth_rejected_total.get(endpoint, 0) + 1
        return _auth_rejected_total[endpoint]


def get_auth_rejected(endpoint: str | None = None) -> int | Dict[str, int]:
    with _lock:
        if endpoint is None:
            return dict(_auth_rejected_total)
        return _auth_rejected_total.get(endpoint, 0)


def snapshot() -> dict:
    """Read-only snapshot for tests/readiness."""
    with _lock:
        return {
            "ovs_sessions_active": _sessions_active,
            "ovs_sessions_rejected_total": dict(_sessions_rejected_total),
            "ovs_auth_rejected_total": dict(_auth_rejected_total),
        }


def _reset_for_tests() -> None:
    """Test-only hook; never call from production code."""
    global _sessions_active
    with _lock:
        _sessions_active = 0
        _sessions_rejected_total.clear()
        _auth_rejected_total.clear()
