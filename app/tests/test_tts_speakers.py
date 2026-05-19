import base64


def test_tts_speaker_registry_supports_preset_and_embedding(monkeypatch):
    from app.core import tts_speakers

    emb = b"\x00\x00\x80?" * 1024
    monkeypatch.setenv(
        "OVS_TTS_SPEAKERS_JSON",
        '{"2301":{"type":"preset","speaker":"2301"},'
        '"10001":{"type":"embedding","speaker_embedding_b64":"%s"}}'
        % base64.b64encode(emb).decode("ascii"),
    )

    assert tts_speakers.speaker_kwargs_for_id(2301) == {
        "speaker_id": 2301,
        "speaker": "2301",
    }
    assert tts_speakers.speaker_kwargs_for_id(10001) == {
        "speaker_id": 10001,
        "speaker_embedding": emb,
    }

    speakers = tts_speakers.available_speakers()
    assert {"id": 2301, "type": "preset", "speaker": "2301"} in speakers
    assert {"id": 10001, "type": "embedding", "speaker_embedding_b64": "<configured>"} in speakers


def test_tts_speaker_registry_rejects_unknown_when_configured(monkeypatch):
    from app.core import tts_speakers

    monkeypatch.setenv("OVS_TTS_ALLOW_UNMAPPED_SPEAKER_ID", "0")

    try:
        tts_speakers.speaker_kwargs_for_id(9999)
    except ValueError as exc:
        assert "Unknown TTS speaker_id" in str(exc)
    else:
        raise AssertionError("expected unknown speaker_id to be rejected")
