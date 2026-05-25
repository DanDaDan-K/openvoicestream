"""Default dialogue turn: tokens stream directly to SLV; no client-side batching.

Historically this was DialogueApp.on_user_utterance; the same logic now
lives in ModeContext.run_default_dialogue_turn (invoked by ChatMode).
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from openvoicestream_agent import Config, Session
from openvoicestream_agent.app_mode import ModeContext, ModeManager
from openvoicestream_agent.apps_dialogue_shim import DialogueApp  # back-compat alias
from openvoicestream_agent.llm.base import LLMBackend
from openvoicestream_agent.llm import LLMStreamError
from openvoicestream_agent.modes import ChatMode


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


class FakeAudio:
    def __init__(self) -> None:
        self.stopped = 0

    async def stop_playback(self) -> None:
        self.stopped += 1


async def _noop_broadcast(*args, **kwargs):
    return None


@pytest.mark.asyncio
async def test_default_dialogue_turn_streams_tokens_directly_to_slv():
    cfg = Config(system_prompt="SYS")
    slv = FakeSLV()
    llm = FakeLLM(["你", "好", "，", "世界。"])
    session = Session()
    events = type("E", (), {"emit": lambda *a, **k: None})()

    ctx = ModeContext(
        config=cfg, slv=slv, llm=llm, session=session, audio=None,
        events=events, broadcast=_noop_broadcast,
    )
    mgr = ModeManager(lambda: ctx)
    mgr.register(ChatMode())
    await mgr.start("chat")

    await mgr.current.on_user_utterance(ctx, "hi")

    # Every LLM token forwarded individually (no batching/joining).
    assert slv.text_frames == ["你", "好", "，", "世界。"]
    # flush_tts called exactly once after stream ends.
    assert slv.flushed == 1
    # History has user + assistant entries.
    assert session.history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "你好，世界。"},
    ]
    # LLM saw full messages including the configured system prompt.
    assert llm.last_messages[0] == {"role": "system", "content": "SYS"}
    assert llm.last_messages[-1] == {"role": "user", "content": "hi"}
    # session was passed through to LLM (for prefix-cache control).
    assert llm.last_session is session


@pytest.mark.asyncio
async def test_cancelled_dialogue_turn_closes_llm_stream_without_tts_flush():
    """Barge-in cancels the dialogue task. That must close the upstream LLM
    stream (edge-llm maps client disconnect to channel.cancel()), but must
    not flush partial old tokens into TTS."""
    cfg = Config(system_prompt="SYS")
    slv = FakeSLV()
    session = Session()
    events = type("E", (), {"emit": lambda *a, **k: None})()
    first_token_sent = asyncio.Event()
    stream_closed = asyncio.Event()

    class CancellableLLM(LLMBackend):
        async def stream(self, messages, **kw):  # type: ignore[override]
            try:
                yield "old"
                first_token_sent.set()
                await asyncio.sleep(60)
                yield "tail"  # pragma: no cover
            finally:
                stream_closed.set()

    ctx = ModeContext(
        config=cfg, slv=slv, llm=CancellableLLM(), session=session, audio=None,
        events=events, broadcast=_noop_broadcast,
    )

    task = asyncio.create_task(ctx.run_default_dialogue_turn("hi"))
    await asyncio.wait_for(first_token_sent.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stream_closed.is_set()
    assert slv.text_frames == ["old"]
    assert slv.flushed == 0
    assert session.history == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_midstream_llm_error_aborts_partial_tts_without_history_pollution():
    cfg = Config(system_prompt="SYS")
    slv = FakeSLV()
    audio = FakeAudio()
    session = Session()
    events = type("E", (), {"emit": lambda *a, **k: None})()

    class PartialThenErrorLLM(LLMBackend):
        async def stream(self, messages, **kw):  # type: ignore[override]
            yield "你"
            yield "可能"
            raise LLMStreamError("finish_reason=error")

    ctx = ModeContext(
        config=cfg, slv=slv, llm=PartialThenErrorLLM(), session=session,
        audio=audio, events=events, broadcast=_noop_broadcast,
    )

    with pytest.raises(LLMStreamError):
        await ctx.run_default_dialogue_turn("hi")

    assert slv.text_frames == ["你", "可能"]
    assert slv.flushed == 0
    assert slv.aborted == 1
    assert audio.stopped == 1
    assert session.history == [{"role": "user", "content": "hi"}]


# ── Batch 2: tool-runner integration into run_default_dialogue_turn ──


class _ScriptedEventsLLM(LLMBackend):
    """LLM that yields a scripted list of LLMEvent objects per call,
    via the richer ``stream_events`` channel. ``stream`` is left as the
    base-class default (text-only filter) so a smoke test through the
    new path is realistic."""

    def __init__(self, scripts):
        from openvoicestream_agent.llm.base import LLMEvent  # noqa
        self._scripts = list(scripts)
        self.calls: list[dict[str, Any]] = []
        self._i = 0

    async def stream(self, messages, **kw):  # pragma: no cover - unused
        if False:
            yield ""
        raise RuntimeError("stream() should not be called")

    async def stream_events(self, messages, **kw):  # type: ignore[override]
        self.calls.append({"messages": list(messages), "kwargs": dict(kw)})
        if self._i >= len(self._scripts):
            raise RuntimeError("script exhausted")
        evs = self._scripts[self._i]
        self._i += 1
        for ev in evs:
            yield ev


@pytest.mark.asyncio
async def test_tools_enabled_one_tool_round_trip_via_app_mode():
    """End-to-end via ModeContext.run_default_dialogue_turn with tools
    enabled: the assistant emits text → tool_call → final text. Each
    text token still streams to SLV; dashboard events fire."""
    from openvoicestream_agent.llm.base import LLMEvent
    from openvoicestream_agent.tools import ToolRegistry

    cfg = Config(
        system_prompt="SYS",
        tools_enabled=True,
        tools_default_allowlist=["time_now"],
    )
    slv = FakeSLV()
    session = Session()
    events_emitted: list[tuple[str, Any]] = []
    events = type(
        "E", (), {"emit": lambda self, n, p=None: events_emitted.append((n, p))}
    )()

    # Per-test registry so we don't depend on global builtins.
    reg = ToolRegistry()

    @reg.tool()
    def time_now() -> dict:
        return {"now": "2026-01-01T12:00:00"}

    llm = _ScriptedEventsLLM([
        # iter 0: small preamble + a tool call
        [
            LLMEvent(kind="text", text="好的，"),
            LLMEvent(
                kind="tool_call_delta",
                tool_call_index=0,
                tool_call_id="c1",
                name="time_now",
                arguments="{}",
            ),
            LLMEvent(kind="finish", finish_reason="tool_calls"),
        ],
        # iter 1: final answer
        [
            LLMEvent(kind="text", text="现在是中午十二点。"),
            LLMEvent(kind="finish", finish_reason="stop"),
        ],
    ])

    bcast_calls: list[tuple[str, tuple]] = []

    # Make the broadcast callable look like a bound method so app_mode's
    # `__self__` lookup finds an "app" with a tool_registry attribute.
    class _FakeApp:
        def __init__(self):
            self.tool_registry = reg

        async def broadcast(self, name, *args):
            bcast_calls.append((name, args))

    fake_app = _FakeApp()

    ctx = ModeContext(
        config=cfg, slv=slv, llm=llm, session=session, audio=None,
        events=events, broadcast=fake_app.broadcast,
    )
    mgr = ModeManager(lambda: ctx)
    mgr.register(ChatMode())
    await mgr.start("chat")

    await mgr.current.on_user_utterance(ctx, "几点了")

    # Every text token streamed to TTS (preamble + post-tool answer).
    assert slv.text_frames == ["好的，", "现在是中午十二点。"]
    assert slv.flushed == 1

    # Session history: user, assistant(tc), tool_result, assistant(text).
    assert len(session.history) == 4
    assert session.history[0] == {"role": "user", "content": "几点了"}
    assert session.history[1]["role"] == "assistant"
    assert session.history[1]["content"] == "好的，"
    assert session.history[1]["tool_calls"][0]["function"]["name"] == "time_now"
    assert session.history[2]["role"] == "tool"
    assert session.history[2]["tool_call_id"] == "c1"
    assert session.history[3] == {
        "role": "assistant",
        "content": "现在是中午十二点。",
    }

    # Dashboard events emitted on the event bus.
    names = [n for n, _ in events_emitted]
    assert "tool_call_started" in names
    assert "tool_call_completed" in names
    # And broadcast to plugins.
    bnames = [n for n, _ in bcast_calls]
    assert "on_tool_call_started" in bnames
    assert "on_tool_call_completed" in bnames


@pytest.mark.asyncio
async def test_tools_disabled_default_path_unchanged():
    """Regression: tools_enabled=False (default) must produce the exact
    same observable behaviour as before batch 1 — every text token
    streams to SLV, history grows by user+assistant, no tool events."""
    from openvoicestream_agent.llm.base import LLMEvent

    cfg = Config(system_prompt="SYS")  # tools_enabled defaults to False
    slv = FakeSLV()
    session = Session()
    events_emitted: list[str] = []
    events = type(
        "E", (), {"emit": lambda self, n, p=None: events_emitted.append(n)}
    )()

    llm = _ScriptedEventsLLM([
        [
            LLMEvent(kind="text", text="你"),
            LLMEvent(kind="text", text="好"),
            LLMEvent(kind="finish", finish_reason="stop"),
        ],
    ])
    ctx = ModeContext(
        config=cfg, slv=slv, llm=llm, session=session, audio=None,
        events=events, broadcast=_noop_broadcast,
    )
    mgr = ModeManager(lambda: ctx)
    mgr.register(ChatMode())
    await mgr.start("chat")

    await mgr.current.on_user_utterance(ctx, "hi")

    assert slv.text_frames == ["你", "好"]
    assert slv.flushed == 1
    assert session.history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "你好"},
    ]
    # No tool dashboard events fired.
    assert "tool_call_started" not in events_emitted
    assert "tool_call_completed" not in events_emitted
    # No tools schema sent to the LLM (runner converts empty allowlist → None).
    assert llm.calls[0]["kwargs"].get("tools") is None


@pytest.mark.asyncio
async def test_app_mode_first_token_timeout_raises_llm_timeout_error():
    """Mocked LLM hangs without ever emitting an event → app_mode
    surfaces LLMTimeoutError(kind='first_token')."""
    from openvoicestream_agent.app_mode import LLMTimeoutError

    cfg = Config(system_prompt="SYS")
    # Use a tiny first-token timeout so the test runs fast.
    cfg.llm_first_token_timeout_s = 0.05
    slv = FakeSLV()
    session = Session()
    events = type("E", (), {"emit": lambda *a, **k: None})()

    class HangLLM(LLMBackend):
        async def stream(self, messages, **kw):
            await asyncio.sleep(10)
            if False:  # pragma: no cover
                yield ""

    ctx = ModeContext(
        config=cfg, slv=slv, llm=HangLLM(), session=session, audio=None,
        events=events, broadcast=_noop_broadcast,
    )
    with pytest.raises(LLMTimeoutError) as exc_info:
        await ctx.run_default_dialogue_turn("hi")
    assert exc_info.value.kind == "first_token"


@pytest.mark.asyncio
async def test_per_mode_tools_enabled_false_overrides_global_true():
    """Codex review HIGH #1 regression: a mode declaring
    ``tools_enabled=False`` must override a global ``tools_enabled=True``.
    The earlier truthy-fallback logic treated False as "not set" and
    inherited the global, leaking tools into modes that opted out."""
    from openvoicestream_agent.llm.base import LLMEvent
    from openvoicestream_agent.tools import ToolRegistry

    cfg = Config(
        system_prompt="SYS",
        tools_enabled=True,
        tools_default_allowlist=["time_now"],
        mode_overrides={"chat": {"tools_enabled": False}},
    )
    slv = FakeSLV()
    session = Session()
    events = type("E", (), {"emit": lambda *a, **kw: None})()
    reg = ToolRegistry()

    @reg.tool()
    def time_now() -> dict:  # pragma: no cover - should never fire
        return {"now": "nope"}

    class _FakeApp:
        def __init__(self):
            self.tool_registry = reg

        async def broadcast(self, name, *args):
            pass

    fake_app = _FakeApp()
    llm = _ScriptedEventsLLM([
        [LLMEvent(kind="text", text="hi"), LLMEvent(kind="finish", finish_reason="stop")],
    ])
    ctx = ModeContext(
        config=cfg, slv=slv, llm=llm, session=session, audio=None,
        events=events, broadcast=fake_app.broadcast,
    )
    mgr = ModeManager(lambda: ctx)
    mgr.register(ChatMode())
    await mgr.start("chat")
    await mgr.current.on_user_utterance(ctx, "say hi")

    # No tools sent to LLM since the mode opted out.
    assert llm.calls[0]["kwargs"].get("tools") is None


@pytest.mark.asyncio
async def test_exception_mid_tool_loop_rolls_back_history():
    """Codex review HIGH #2 regression: an exception that escapes after
    we've already committed assistant(tool_calls) + tool result messages
    must roll the session.history back to the pre-turn anchor, otherwise
    the orphan assistant_tool_calls pins forever and breaks subsequent
    trim and prefix-cache invariants."""
    from openvoicestream_agent.llm.base import LLMEvent
    from openvoicestream_agent.tools import ToolRegistry

    cfg = Config(
        system_prompt="SYS",
        tools_enabled=True,
        tools_default_allowlist=["explode"],
    )
    slv = FakeSLV()
    session = Session()
    events = type("E", (), {"emit": lambda *a, **kw: None})()
    reg = ToolRegistry()

    @reg.tool()
    def explode() -> dict:
        return {"ok": True}

    class _FakeApp:
        def __init__(self):
            self.tool_registry = reg

        async def broadcast(self, name, *args):
            pass

    fake_app = _FakeApp()

    # First iteration commits assistant(tool_calls) + tool result;
    # second iteration's stream raises mid-flight. Rollback should
    # remove everything added in this turn (the user msg too).
    class _BoomLLM(_ScriptedEventsLLM):
        async def stream_events(self, messages, **kw):  # type: ignore[override]
            self.calls.append({"messages": list(messages), "kwargs": dict(kw)})
            if self._i == 0:
                self._i += 1
                for ev in [
                    LLMEvent(
                        kind="tool_call_delta", tool_call_index=0,
                        tool_call_id="c1", name="explode", arguments="{}",
                    ),
                    LLMEvent(kind="finish", finish_reason="tool_calls"),
                ]:
                    yield ev
                return
            raise RuntimeError("boom")

    llm = _BoomLLM([])
    ctx = ModeContext(
        config=cfg, slv=slv, llm=llm, session=session, audio=None,
        events=events, broadcast=fake_app.broadcast,
    )
    mgr = ModeManager(lambda: ctx)
    mgr.register(ChatMode())
    await mgr.start("chat")
    with pytest.raises(RuntimeError, match="boom"):
        await mgr.current.on_user_utterance(ctx, "do it")

    # Rollback semantics: the runner's anchor was captured AFTER
    # add_user(text), so user message stays; assistant_tool_calls
    # and tool_result added inside the runner are dropped.
    roles = [m["role"] for m in session.history]
    assert "tool" not in roles, f"orphan tool message survived: {session.history}"
    assistant_with_tc = [
        m for m in session.history
        if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    assert not assistant_with_tc, (
        f"orphan assistant(tool_calls) survived: {session.history}"
    )


@pytest.mark.asyncio
async def test_multi_mode_app_class_is_back_compat_dialogue_shim():
    """The legacy `DialogueApp` import path now resolves to MultiModeApp."""
    from apps.multi_mode.app import MultiModeApp

    assert DialogueApp is MultiModeApp
