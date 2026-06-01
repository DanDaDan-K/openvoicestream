"""factory.py — build a concrete :class:`Actuator` from config.

The plugin layer never imports a concrete driver directly; it asks the
factory for one by name. Registering a new motor is a one-line addition
to ``_REGISTRY`` plus the driver module.

Config shape (lives under ``metadata.actuator`` in the agent YAML):

    metadata:
      actuator:
        backend: so_arm
        config:
          port: /dev/ttyACM0
          arm_id: voice_arm
          move_delay: 1.5
          gesture_delay: 0.4

``create_actuator("so_arm", {...})`` returns a connected-on-demand
``SOArmActuator`` — the actual serial ``connect()`` is the caller's
responsibility (ArmPlugin runs it in ``asyncio.to_thread`` during
``start()`` so the sync setup path doesn't block).
"""
from __future__ import annotations

from typing import Any, Callable, Dict

from .base import Actuator


def _make_so_arm(config: Dict[str, Any]) -> Actuator:
    from .so_arm import SOArmActuator

    # Accept both the new key (``port``) and the historical ``arm_port``
    # alias so existing config keeps working through the metadata.arm
    # compat window.
    port = config.get("port", config.get("arm_port"))
    if port is None:
        raise ValueError("so_arm actuator requires a 'port' in config")
    return SOArmActuator(
        port=port,
        arm_id=config.get("arm_id", "voice_arm"),
        move_delay=float(config.get("move_delay", 1.5)),
        gesture_delay=float(config.get("gesture_delay", 0.4)),
    )


_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Actuator]] = {
    "so_arm": _make_so_arm,
}


def create_actuator(name: str, config: Dict[str, Any]) -> Actuator:
    """Build an :class:`Actuator` by registry name.

    Raises ``ValueError`` for an unknown backend name.
    """
    builder = _REGISTRY.get(name)
    if builder is None:
        raise ValueError(
            f"unknown actuator backend {name!r}; "
            f"known: {sorted(_REGISTRY)}"
        )
    return builder(dict(config or {}))


__all__ = ["create_actuator"]
