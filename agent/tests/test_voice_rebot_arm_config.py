from __future__ import annotations

from pathlib import Path

from ovs_agent.config import load_config


def test_voice_rebot_arm_uses_direct_wake_command_turns():
    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "ovs_agent"
        / "apps"
        / "voice_rebot_arm"
        / "config.yaml"
    )

    cfg = load_config(cfg_path)

    assert cfg.pipeline_mode == "wake_word"
    assert cfg.wake_command_single_turn is True
    assert cfg.gate_drive_eos is True
    assert float(cfg.asr_final_timeout_s) <= 1.0
    assert cfg.tool_trigger_guard is True
    prompt = " ".join(cfg.system_prompt.split())
    assert "FIRST emit a brief" not in prompt
    assert "Do not speak a separate acknowledgement before the tool call" in prompt
    assert "Only say \"没听清" in prompt
    assert "I heard" in prompt
