from ovs_agent.audio import devices


def test_resolve_output_index_accepts_respeaker_name(monkeypatch):
    monkeypatch.setattr(devices, "_enumerate_outputs", lambda: [
        (2, "NVIDIA Jetson APE"),
        (24, "reSpeaker XVF3800"),
    ])
    monkeypatch.setattr(devices.time, "sleep", lambda _seconds: None)

    assert devices.resolve_output_index("reSpeaker") == 24


def test_resolve_input_index_falls_back_to_system_default(monkeypatch):
    monkeypatch.setattr(devices, "_enumerate_inputs", lambda: [(5, "Jetson APE Input")])
    monkeypatch.setattr(devices, "_default_input_index", lambda: 3)
    monkeypatch.setattr(devices.time, "sleep", lambda _seconds: None)

    assert devices.resolve_input_index("reSpeaker") == 3
