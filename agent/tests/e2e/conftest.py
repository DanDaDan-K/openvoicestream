"""E2E pytest fixtures + WAV fixture auto-generation."""
from __future__ import annotations

import asyncio
import os
import socket
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import pytest_asyncio

# Force NO_PROXY to bypass system proxy for orin-nx Tailscale IP — proxy
# would otherwise corrupt the WS upgrade handshake.
os.environ.setdefault("NO_PROXY", "100.82.225.102,localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "100.82.225.102,localhost,127.0.0.1")

from .fake_audio import ScriptedAudioIO  # noqa: E402
from .probe import AgentProbe  # noqa: E402

ORIN = "100.82.225.102"
SLV_URL = f"ws://{ORIN}:8621/v2v/stream"
LLM_BASE = f"http://{ORIN}:8000/v1"
LLM_MODEL = "Qwen/Qwen3-4B-AWQ"

WAV_DIR = Path(__file__).parent / "fixtures" / "wav"
# (filename, text) — text fed to `say -v Tingting` if file is missing.
# Tingting handles both Chinese and English (English speech sounds odd but
# ASR still picks up the characters / phonemes well enough for these tests).
WAV_SCRIPT = [
    ("hello.wav",         "你好"),
    ("weather.wav",       "今天天气怎么样"),
    ("story_request.wav", "请详细讲一个五百字的小故事"),
    # Use "别说了" (not "停下来"): it is an EXACT entry in the default
    # stop_words and macOS `say -v Tingting` enunciates it cleanly, so it
    # matches reliably across ASR engines. ("停下来" was historically
    # mis-transcribed as "听下来" by the streaming ASR, failing the match.)
    ("stop_zh.wav",       "别说了"),
    ("stop_en.wav",       "stop please"),
    ("stopwatch.wav",     "stopwatch"),
    ("barge_in.wav",      "等一下我要换个话题"),
    ("silence_5s.wav",    None),  # generated as 5s pure silence
]


def _gen_wav(name: str, text: str | None) -> Path:
    out = WAV_DIR / name
    if out.exists() and out.stat().st_size > 0:
        return out
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    if text is None:
        # 5s silence at 16kHz mono int16
        import wave

        with wave.open(str(out), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * (16000 * 5))
        return out
    # Use macOS `say -v Tingting`. If on Linux CI, skip generation and let
    # tests fail with a clear message.
    if subprocess.run(["which", "say"], capture_output=True).returncode != 0:
        raise RuntimeError(
            f"WAV {name} missing and `say` unavailable (non-macOS). Commit fixtures."
        )
    voice_arg = ["-v", "Tingting"] if text and any("一" <= c <= "鿿" for c in text) else []
    cmd = [
        "say",
        *voice_arg,
        text,
        "-o",
        str(out),
        "--data-format=LEI16@16000",
        "--file-format=WAVE",
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"say failed for {name}: {r.stderr.decode(errors='ignore')}")
    return out


def _ensure_wavs() -> None:
    for name, text in WAV_SCRIPT:
        _gen_wav(name, text)


# Generate fixtures at import time (cheap; cached after first run).
_ensure_wavs()


@pytest.fixture
def wav_dir() -> Path:
    return WAV_DIR


@pytest.fixture
def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture
def test_config(free_port: int):
    from ovs_agent.config import Config, _default_slv_config

    slv_cfg = _default_slv_config()
    # Disable server VAD; let client VAD drive endpoints. asr_language=auto
    # so English & Chinese both work without per-test reconfig.
    slv_cfg.update({"vad": "none", "asr_language": "auto"})
    return Config(
        slv_url=SLV_URL,
        slv_config=slv_cfg,
        llm_backend="edge_llm",
        llm_base_url=LLM_BASE,
        llm_api_key="EMPTY",
        llm_model=LLM_MODEL,
        system_prompt="你是简洁的语音助手，每次回答两三句话以内。",
        audio_input_sample_rate=16000,
        audio_output_sample_rate=24000,
        client_vad_backend="energy",
        client_vad_threshold=0.005,
        client_vad_speech_min_ms=200,
        client_vad_silence_ms=600,
        client_vad_drive_eos=True,
        metadata={"dashboard_port": free_port},
    )


@asynccontextmanager
async def run_agent(config, audio: ScriptedAudioIO):
    """Combined helper: spawn MultiModeApp, attach scripted audio, connect probe.

    Yields (app, probe). Tears everything down on exit.
    """
    from ovs_agent.apps.multi_mode.app import MultiModeApp

    app = MultiModeApp(config)
    app.audio = audio
    # Startup-race gate (#38): the mic pump drops audio until
    # ``app._advertise_ready`` is set. Wire a ready event into the scripted
    # audio so the first WAV only streams once the pump will forward it.
    ready = asyncio.Event()
    audio.ready_event = ready

    async def _wait_advertise_ready() -> None:
        # ``_advertise_ready`` is created inside app.run(); poll until it
        # exists and is set, then release the scripted audio.
        while True:
            ev = getattr(app, "_advertise_ready", None)
            if ev is not None:
                await ev.wait()
                break
            await asyncio.sleep(0.02)
        ready.set()

    run_task = asyncio.create_task(app.run(), name="multi-mode-run")
    ready_task = asyncio.create_task(_wait_advertise_ready(), name="advertise-ready-gate")
    probe = AgentProbe(port=config.metadata["dashboard_port"])
    try:
        await probe.connect()
        yield app, probe
    finally:
        ready_task.cancel()
        try:
            await ready_task
        except (asyncio.CancelledError, Exception):
            pass
        try:
            app.request_shutdown()
        except Exception:
            pass
        try:
            await audio.close()
        except Exception:
            pass
        try:
            await probe.close()
        except Exception:
            pass
        try:
            await asyncio.wait_for(run_task, timeout=8)
        except (asyncio.TimeoutError, Exception):
            run_task.cancel()
            try:
                await run_task
            except (asyncio.CancelledError, Exception):
                pass


@pytest_asyncio.fixture
async def agent_factory(test_config):
    """Returns the run_agent context manager bound to test_config.

    Usage in tests:
        async with agent_factory(audio) as (app, probe): ...
    """
    def _factory(audio: ScriptedAudioIO):
        return run_agent(test_config, audio)
    return _factory


def pytest_collection_modifyitems(config, items):
    """Tolerate the known remote-ASR flake (the live streaming ASR can drop
    `asr_final` under back-to-back multi-turn load). This is an environmental
    flake on the live remote engine, NOT a code bug — so mark every collected
    e2e item flaky to auto-retry. Scoped to this e2e conftest only; the unit
    suite is unaffected.
    """
    for item in items:
        item.add_marker(pytest.mark.flaky(reruns=2, reruns_delay=3))


__all__ = ["run_agent", "ScriptedAudioIO", "AgentProbe", "WAV_DIR"]
