"""Pluggable actuator abstraction for the OpenVoiceStream Agent.

An ``Actuator`` owns a physical motion device (a robot arm, a pan/tilt
head, a gripper, …) plus its observation cache. The voice pipeline and
the observation HTTP server talk to the actuator only through the
``Actuator`` ABC, so swapping in a new motor is a matter of writing one
concrete driver and registering it in ``factory.create_actuator``.

The first concrete driver is :class:`~.so_arm.SOArmActuator` (the SO-ARM
Feetech follower). Future motors implement the same ABC.
"""
from .base import Actuator
from .factory import create_actuator

__all__ = ["Actuator", "create_actuator"]
