"""voice_arm — actuator (SO-ARM) voice agent built on MultiModeApp.

Loaded by the ovs-agent CLI as ``apps.voice_arm.app:App``. The actuator
itself is pluggable: ``metadata.actuator.backend`` selects the concrete
driver (currently ``so_arm``); future motors implement the same
``openvoicestream_agent.actuators.base.Actuator`` ABC.
"""
from .app import App, VoiceArmApp

__all__ = ["App", "VoiceArmApp"]
