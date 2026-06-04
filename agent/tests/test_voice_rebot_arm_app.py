from types import SimpleNamespace


def test_voice_rebot_arm_uses_resolved_audio_indices(monkeypatch):
    from ovs_agent.apps.voice_rebot_arm import app as rebot_app

    audio_calls = []

    class FakeTappedAudioIO:
        def __init__(self, **kwargs):
            audio_calls.append(kwargs)

    def fake_multimode_init(self, config):
        self.config = config
        self.plugins = []
        self.tool_registry = object()

    def fake_register(self, plugin):
        self.plugins.append(plugin)

    monkeypatch.setattr(rebot_app.MultiModeApp, "__init__", fake_multimode_init)
    monkeypatch.setattr(rebot_app.VoiceRebotArmApp, "register", fake_register)
    monkeypatch.setattr(rebot_app, "TappedAudioIO", FakeTappedAudioIO)
    monkeypatch.setattr(rebot_app, "ArmPlugin", lambda app, cfg: ("arm", cfg))
    monkeypatch.setattr(rebot_app, "GraspPlugin", lambda app, cfg: ("grasp", cfg))
    monkeypatch.setattr(rebot_app, "OpenWakeWordSource", lambda *args, **kwargs: ("wake", kwargs))

    input_values = []
    output_values = []
    monkeypatch.setattr(
        rebot_app,
        "resolve_input_index",
        lambda value: input_values.append(value) or 11,
        raising=False,
    )
    monkeypatch.setattr(
        rebot_app,
        "resolve_output_index",
        lambda value: output_values.append(value) or 24,
        raising=False,
    )

    cfg = SimpleNamespace(
        audio_input_device="reSpeaker",
        audio_output_device="reSpeaker",
        audio_input_sample_rate=16000,
        audio_output_sample_rate=16000,
        metadata={
            "wakeword": {
                "mic_channels": 6,
                "mic_channel_select": 4,
                "threshold": 0.5,
                "cooldown_s": 2.0,
                "vad_threshold": 0.0,
            },
            "actuator": {"backend": "rebot_arm", "config": {}},
        },
    )

    rebot_app.VoiceRebotArmApp(cfg)

    assert input_values == ["reSpeaker"]
    assert output_values == ["reSpeaker"]
    assert audio_calls[0]["input_device"] == 11
    assert audio_calls[0]["output_device"] == 24
    assert audio_calls[1]["input_device"] == 11
    assert audio_calls[1]["output_device"] == 24
