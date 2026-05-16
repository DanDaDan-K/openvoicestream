"""SLVClient against a mock /v2v/stream WS server."""
from __future__ import annotations

import asyncio
import json
import struct

import pytest
import websockets
from websockets.asyncio.server import serve

from openvoicestream_agent.slv_client import (
    ASREndpoint,
    ASRFinal,
    ASRPartial,
    SLVClient,
    TTSAudio,
    TTSDone,
    TTSSentenceDone,
    TTSStarted,
)


async def _mock_server(received: list, ready: asyncio.Event):
    async def handler(ws):
        ready.set()
        # 1. read config frame
        cfg_msg = await ws.recv()
        received.append(("config", cfg_msg))
        # 2. push canonical event sequence
        await ws.send(json.dumps({"type": "asr_partial", "text": "你", "is_stable": False}))
        await ws.send(json.dumps({"type": "asr_endpoint"}))
        await ws.send(json.dumps({
            "type": "asr_final", "text": "你好", "session_complete": False,
        }))
        await ws.send(json.dumps({"type": "tts_started", "sentence": "嗨"}))
        # First binary: 4-byte LE sample-rate header + dummy PCM
        sr = 24000
        pcm = b"\x01\x00" * 8
        await ws.send(struct.pack("<I", sr) + pcm)
        # Second binary: pure PCM
        await ws.send(b"\x02\x00" * 4)
        await ws.send(json.dumps({"type": "tts_sentence_done", "sentence": "嗨"}))
        await ws.send(json.dumps({"type": "tts_done"}))
        # 3. wait for any client frames that arrive during the test
        try:
            async for msg in ws:
                received.append(("from_client", msg))
        except websockets.ConnectionClosed:
            return

    server = await serve(handler, "127.0.0.1", 0)
    return server


@pytest.mark.asyncio
async def test_slv_client_decodes_event_sequence():
    received: list = []
    ready = asyncio.Event()
    server = await _mock_server(received, ready)
    host, port = server.sockets[0].getsockname()[:2]

    client = SLVClient(f"ws://{host}:{port}", {"asr_language": "zh", "tts_language": "zh"})
    await client.connect()

    collected = []

    async def collect():
        async for evt in client.events():
            collected.append(evt)
            if isinstance(evt, TTSDone):
                return

    try:
        await asyncio.wait_for(collect(), timeout=3.0)
    finally:
        await client.close()
        server.close()
        await server.wait_closed()

    # Config frame received by server
    assert received[0][0] == "config"
    cfg = json.loads(received[0][1])
    assert cfg["type"] == "config"
    assert cfg["multi_utterance"] is True  # invariant 1 enforced

    types = [type(e) for e in collected]
    assert ASRPartial in types
    assert ASREndpoint in types
    assert ASRFinal in types
    assert TTSStarted in types
    assert TTSAudio in types
    assert TTSSentenceDone in types
    assert TTSDone in types

    # First TTSAudio: 4-byte SR stripped, sample_rate = 24000
    audios = [e for e in collected if isinstance(e, TTSAudio)]
    assert len(audios) == 2
    assert audios[0].sample_rate == 24000
    assert audios[0].pcm == b"\x01\x00" * 8
    assert audios[1].sample_rate == 24000  # cached for subsequent frames
    assert audios[1].pcm == b"\x02\x00" * 4

    # asr_final session_complete=False survived
    final = next(e for e in collected if isinstance(e, ASRFinal))
    assert final.text == "你好"
    assert final.session_complete is False


@pytest.mark.asyncio
async def test_slv_client_send_methods_emit_correct_payloads():
    received: list = []
    ready = asyncio.Event()
    server = await _mock_server(received, ready)
    host, port = server.sockets[0].getsockname()[:2]

    client = SLVClient(f"ws://{host}:{port}", {"asr_language": "zh"})
    await client.connect()

    await client.send_audio(b"\xab\xcd" * 16)
    await client.send_text("hello")
    await client.flush_tts()
    await client.abort()
    await client.asr_eos()

    # Give the server a moment to receive
    await asyncio.sleep(0.2)
    await client.close()
    server.close()
    await server.wait_closed()

    client_frames = [m for tag, m in received if tag == "from_client"]
    # binary frame
    assert any(isinstance(m, (bytes, bytearray)) and bytes(m) == b"\xab\xcd" * 16 for m in client_frames)
    # JSON frames
    json_frames = [json.loads(m) for m in client_frames if isinstance(m, str)]
    types = [j["type"] for j in json_frames]
    assert "text" in types
    assert "tts_flush" in types
    assert "abort" in types
    assert "asr_eos" in types
    text_frame = next(j for j in json_frames if j["type"] == "text")
    assert text_frame["text"] == "hello"
