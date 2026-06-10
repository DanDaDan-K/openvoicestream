"""VoiceArmApp — actuator (SO-ARM) voice agent built on MultiModeApp.

Pipeline: wake_word (OpenWakeWordSource) → SLV streaming ASR → LLM with
actuator tool-calling → SLV streaming TTS.

Key wiring decisions (preserved verbatim from the original voice_arm app):

  * BaseApp's ``__init__`` directly assigns ``self.audio = AudioIO(...)``;
    there is NO ``_make_audio_io`` factory hook. We therefore replace the
    instance attribute immediately after ``super().__init__`` so that any
    subsequent code reading ``self.audio`` (capture taps, plugin start,
    run()'s mic_pump) sees the TappedAudioIO.

  * Plugin.setup() is SYNCHRONOUS per the framework's contract. ArmPlugin
    keeps the serial connect in start() (async, wrapped in
    asyncio.to_thread) so the sync setup path doesn't block.

  * The CLI loader resolves ``apps.<name>.app:App`` — we expose ``App``
    as a module-level alias for ovs-agent compatibility.

Config (see config.yaml): actuator settings live under
``metadata.actuator: {backend, config: {...}, actions_yaml_path}``. The
old ``metadata.arm`` shape is still accepted for one release as a
read-compat alias (see ``_resolve_actuator_cfg``).
"""
from __future__ import annotations

import logging

from ovs_agent.apps.multi_mode.app import MultiModeApp

from ovs_agent.audio.profiles import resolve_mic_profile
from ovs_agent.audio.tapped_audio_io import TappedAudioIO
from ovs_agent.plugins.actuator_actions import ArmPlugin
from ovs_agent.wake_sources.openwakeword import OpenWakeWordSource

logger = logging.getLogger(__name__)


def _resolve_actuator_cfg(meta: dict) -> dict:
    """Build the ArmPlugin config from ``metadata.actuator`` (preferred)
    with a ``metadata.arm`` read-compat alias for one release.

    New shape (``metadata.actuator``):
        backend: so_arm
        config: {port, arm_id, move_delay, gesture_delay, ...}
        actions_yaml_path: /path/to/actions.yaml
        # optional pass-through: observation_port, required_fields,
        #   clear_history_on_turn_end, clear_history_on_tool_change

    Legacy shape (``metadata.arm``):
        actions_yaml_path, arm_port, arm_id, move_delay, gesture_delay,
        observation_port, clear_history_on_turn_end, ...

    Returns the dict ArmPlugin expects:
        {backend, actuator_config, actions_yaml_path, observation_port?,
         required_fields?, clear_history_on_turn_end?, ...}
    """
    actuator = dict(meta.get("actuator", {}) or {})
    arm = dict(meta.get("arm", {}) or {})

    if actuator:
        backend = actuator.get("backend", "so_arm")
        actuator_config = dict(actuator.get("config", {}) or {})
        plugin_cfg: dict = {
            "backend": backend,
            "actuator_config": actuator_config,
        }
        # Plugin-level keys may live either at the actuator top level or
        # inside its config block; accept both for convenience.
        for key in (
            "actions_yaml_path",
            "observation_port",
            "required_fields",
            "clear_history_on_turn_end",
            "clear_history_on_tool_change",
        ):
            if key in actuator:
                plugin_cfg[key] = actuator[key]
            elif key in actuator_config:
                plugin_cfg[key] = actuator_config[key]
        return plugin_cfg

    # ── legacy metadata.arm compat alias ───────────────────────────
    logger.warning(
        "metadata.arm is deprecated; move actuator settings under "
        "metadata.actuator: {backend, config: {...}}. Reading the legacy "
        "shape for this release."
    )
    actuator_config = {
        "port": arm.get("arm_port"),
        "arm_id": arm.get("arm_id", "voice_arm"),
        "move_delay": float(arm.get("move_delay", 1.5)),
        "gesture_delay": float(arm.get("gesture_delay", 0.4)),
    }
    plugin_cfg = {
        "backend": "so_arm",
        "actuator_config": actuator_config,
    }
    for key in (
        "actions_yaml_path",
        "observation_port",
        "required_fields",
        "clear_history_on_turn_end",
        "clear_history_on_tool_change",
    ):
        if key in arm:
            plugin_cfg[key] = arm[key]
    return plugin_cfg


class VoiceArmApp(MultiModeApp):
    def __init__(self, config) -> None:  # noqa: ANN001
        super().__init__(config)

        # F1: BaseApp built a plain AudioIO. Swap to our tap-capable
        # variant before run() opens any audio streams. Reuse the same
        # device / sample-rate config the framework just resolved.
        logger.info("VoiceArmApp: replacing AudioIO with TappedAudioIO")
        self.audio = TappedAudioIO(
            input_device=config.audio_input_device,
            output_device=config.audio_output_device,
            input_sr=config.audio_input_sample_rate,
            output_sr=config.audio_output_sample_rate,
        )

        # Pull our subblocks out of Config.metadata — Config is a dataclass
        # so anything not in the schema lives under ``metadata`` from YAML.
        meta = getattr(config, "metadata", {}) or {}
        wake_cfg = dict(meta.get("wakeword", {}) or {})
        arm_cfg = _resolve_actuator_cfg(meta)

        # Override AudioIO again, now that we know the mic channel count
        # from the wakeword config (reSpeaker = 6 channels, exclusive USB
        # device that rejects channels=1 → PaErrorCode -9998).
        # Resolve the mic channel count. An explicit numeric ``mic_channels``
        # in the YAML/env wins (back-compat: pinned deployments unchanged).
        # ``auto`` (the default) runs reSpeaker profile auto-detection, so
        # swapping a firmware variant that changes the USB-UAC channel count
        # (6ch Flex ↔ 2ch 4-Mic) no longer needs a config edit. See
        # ovs_agent.audio.profiles.
        mic_channels_cfg = wake_cfg.get("mic_channels", "auto")
        mic_channel_select_raw = wake_cfg.get("mic_channel_select")
        prof = None
        if str(mic_channels_cfg).strip().lower() in ("", "auto", "none"):
            prof = resolve_mic_profile(config.audio_input_device)
            mic_channels = prof.mic_channels
            mic_channel_select = (
                prof.mic_channel_select
                if mic_channel_select_raw in (None, "", "auto", "mean")
                else int(mic_channel_select_raw)
            )
        else:
            mic_channels = int(mic_channels_cfg)
            if mic_channel_select_raw in (None, "", "auto"):
                mic_channel_select = 0
            elif mic_channel_select_raw == "mean":
                mic_channel_select = None
            else:
                mic_channel_select = int(mic_channel_select_raw)
        if mic_channels > 1:
            logger.info(
                "Re-opening AudioIO with mic_channels=%d select=%r",
                mic_channels, mic_channel_select,
            )
            self.audio = TappedAudioIO(
                input_device=config.audio_input_device,
                output_device=config.audio_output_device,
                input_sr=config.audio_input_sample_rate,
                output_sr=config.audio_output_sample_rate,
                mic_channels=mic_channels,
                mic_channel_select=mic_channel_select,
            )

        # Mic makeup gain follows the firmware profile too. The 6ch and 2ch
        # reSpeaker variants need very different gains (quiet ch0 needs ~12x;
        # the louder 2ch firmware clips at 12x → garbled ASR, wants ~2x). When
        # the config leaves makeup at the 1.0 no-op default, fill it from the
        # resolved profile; an explicit non-1.0 in the YAML still wins.
        if abs(float(getattr(config, "mic_makeup_gain", 1.0)) - 1.0) < 1e-9:
            if prof is None:
                prof = resolve_mic_profile(config.audio_input_device)
            if prof is not None and prof.mic_makeup_gain is not None:
                logger.info(
                    "mic makeup gain from profile '%s': %.1f (config left at default)",
                    prof.name, prof.mic_makeup_gain,
                )
                config.mic_makeup_gain = float(prof.mic_makeup_gain)

        # ArmPlugin owns the serial port + obs HTTP server. Register the
        # plugin before the wake source so tools are available the moment
        # the first wake fires.
        self.register(ArmPlugin(self, arm_cfg))

        # Local wake-word detection. Replaces the default WakeSource set
        # that MultiModeApp registers when pipeline_mode != always_on
        # (those entries are HTTP / MQTT / serial / local_keyword — none
        # of them speak openwakeword). To avoid double-registering wake
        # sources, set ``wake_sources: []`` in the YAML config.
        self.register(OpenWakeWordSource(
            self,
            model_name=wake_cfg.get("model", "hey jarvis"),
            threshold=float(wake_cfg.get("threshold", 0.5)),
            cooldown_s=float(wake_cfg.get("cooldown_s", 2.0)),
            vad_threshold=float(wake_cfg.get("vad_threshold", 0.0)),
        ))


# CLI loader expects an ``App`` symbol at module top level.
App = VoiceArmApp


__all__ = ["VoiceArmApp", "App"]
