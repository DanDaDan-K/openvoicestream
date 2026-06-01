"""Regression guard for the barge-in mic-forwarding fix (2026-06-01).

Barge-in regressed because `mic_drop_while_speaking=true` dropped mic audio
while the agent was SPEAKING → SLV never received the user's interrupting
speech → never emitted ASRPartial → the barge-in trigger never fired. The
fix set it false (reSpeaker XVF3800 has hardware AEC). This locks the
mic-pump behaviour both ways so it can't silently regress.
"""
from __future__ import annotations

import pytest

from ovs_agent import Config, Session
from ovs_agent.app_base import BaseApp
from ovs_agent.state import ConvState


class _FakeAudio:
    chunk_ms = 100

    def __init__(self, chunks):
        self._chunks = chunks

    async def start_capture(self):
        for c in self._chunks:
            yield c


class _FakeSLV:
    def is_reconnecting(self) -> bool:
        return False


def _mic_app(*, drop_while_speaking: bool, state: ConvState):
    app = BaseApp.__new__(BaseApp)
    app.config = Config(
        system_prompt="SYS",
        mic_drop_while_speaking=drop_while_speaking,
        energy_gate_enabled=False,   # keep the forward path simple
        mic_makeup_gain=1.0,
    )
    # 3 non-silent 100ms@16k chunks (3200 bytes each).
    app.audio = _FakeAudio([b"\x10\x00" * 1600 for _ in range(3)])
    app.slv = _FakeSLV()
    app._client_vad = None          # server-VAD mode (production server-loop)
    app._state = state
    app._advertise_ready = None     # skip the advertise barrier
    app._vad_state = "idle"
    app._last_mic_chunk_ts = 0.0

    sent: list[bytes] = []

    async def _send(pcm):
        sent.append(pcm)

    app._send_audio_nonblocking = _send          # type: ignore[assignment]
    app._schedule_mic_rms_broadcast = lambda d: False  # type: ignore[assignment]
    return app, sent


@pytest.mark.asyncio
async def test_mic_forwards_during_speaking_when_drop_false():
    """barge-in enabler: with drop=false, mic audio is forwarded to SLV even
    while SPEAKING, so the server can detect an interrupt and emit ASRPartial."""
    app, sent = _mic_app(drop_while_speaking=False, state=ConvState.SPEAKING)
    await app._mic_pump()
    assert len(sent) == 3, "mic must keep forwarding during SPEAKING for barge-in"


@pytest.mark.asyncio
async def test_mic_dropped_during_speaking_when_drop_true():
    """The old (barge-in-breaking) behaviour: drop=true silences the mic while
    SPEAKING. Locked so the trade-off is explicit, not accidental."""
    app, sent = _mic_app(drop_while_speaking=True, state=ConvState.SPEAKING)
    await app._mic_pump()
    assert len(sent) == 0, "drop=true must drop mic during SPEAKING"


@pytest.mark.asyncio
async def test_mic_forwards_when_idle_regardless():
    """Sanity: in a listening/idle-ish state the mic forwards either way
    (the drop gate only applies to SPEAKING/THINKING)."""
    app, sent = _mic_app(drop_while_speaking=True, state=ConvState.LISTENING)
    await app._mic_pump()
    assert len(sent) == 3, "mic must forward when not SPEAKING/THINKING"
