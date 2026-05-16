"""DialogueApp: tokens stream directly to SLV; no client-side batching."""
from __future__ import annotations

from typing import Any

import pytest

from openvoicestream_agent import Config, Session
from openvoicestream_agent.apps_dialogue_shim import DialogueApp  # type: ignore  # see conftest
from openvoicestream_agent.llm.base import LLMBackend


class FakeSLV:
    def __init__(self) -> None:
        self.text_frames: list[str] = []
        self.flushed: int = 0
        self.aborted: int = 0

    async def send_text(self, text: str) -> None:
        self.text_frames.append(text)

    async def flush_tts(self) -> None:
        self.flushed += 1

    async def abort(self) -> None:
        self.aborted += 1


class FakeLLM(LLMBackend):
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.last_messages: list[dict[str, str]] | None = None
        self.last_session: Any = None

    async def stream(self, messages, **kw):  # type: ignore[override]
        self.last_messages = list(messages)
        self.last_session = kw.get("session")
        for t in self.tokens:
            yield t


@pytest.mark.asyncio
async def test_dialogue_app_streams_tokens_directly_to_slv():
    cfg = Config(system_prompt="SYS")
    app = DialogueApp.__new__(DialogueApp)
    app.config = cfg
    app.slv = FakeSLV()
    app.llm = FakeLLM(["你", "好", "，", "世界。"])
    app.session = Session()
    app.events = type("E", (), {"emit": lambda *a, **k: None})()

    await app.on_user_utterance("hi")

    # Every LLM token forwarded individually (no batching/joining).
    assert app.slv.text_frames == ["你", "好", "，", "世界。"]
    # flush_tts called exactly once after stream ends.
    assert app.slv.flushed == 1
    # History has user + assistant entries.
    assert app.session.history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "你好，世界。"},
    ]
    # LLM saw full messages including system prompt + user turn.
    assert app.llm.last_messages[0] == {"role": "system", "content": "SYS"}
    assert app.llm.last_messages[-1] == {"role": "user", "content": "hi"}
    # session was passed through to LLM (for prefix-cache control).
    assert app.llm.last_session is app.session
