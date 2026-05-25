"""Tests for the multi-turn LLM ↔ tool runner."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from openvoicestream_agent.llm import LLMEvent
from openvoicestream_agent.session import Session
from openvoicestream_agent.tools import ToolCallCtx, ToolRegistry, stream_with_tools


class _FakeLLM:
    """Stub backend that returns a scripted sequence of LLMEvent lists,
    one per stream_events() call. Records the kwargs of each call."""

    def __init__(self, script: list[list[LLMEvent]]):
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []
        self._call_idx = 0

    async def stream_events(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ):
        # Snapshot the kwargs the runner passed in.
        self.calls.append({
            "messages": [dict(m) for m in messages],
            "kwargs": dict(kwargs),
        })
        if self._call_idx >= len(self._script):
            raise RuntimeError("fake LLM script exhausted")
        events = self._script[self._call_idx]
        self._call_idx += 1
        for ev in events:
            yield ev


def _text(t: str) -> LLMEvent:
    return LLMEvent(kind="text", text=t)


def _tc(idx: int, *, id: str | None = None, name: str | None = None,
        arguments: str | None = None) -> LLMEvent:
    return LLMEvent(
        kind="tool_call_delta",
        tool_call_index=idx,
        tool_call_id=id,
        name=name,
        arguments=arguments,
    )


def _finish(reason: str) -> LLMEvent:
    return LLMEvent(kind="finish", finish_reason=reason)


def _make_ctx(session: Session) -> ToolCallCtx:
    return ToolCallCtx(session=session)


# ── (a) text-only, no tools called ────────────────────────────────────


@pytest.mark.asyncio
async def test_text_only_no_tools():
    session = Session()
    registry = ToolRegistry()
    llm = _FakeLLM([[_text("hello"), _text(" world"), _finish("stop")]])

    tokens: list[str] = []

    async def on_tok(t):
        tokens.append(t)

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    final = await stream_with_tools(
        llm, msgs,
        session=session, registry=registry, allowed_tools=None,
        ctx=_make_ctx(session), on_assistant_token=on_tok,
    )
    assert final == "hello world"
    assert tokens == ["hello", " world"]
    assert session.history == [
        {"role": "assistant", "content": "hello world"},
    ]
    assert len(llm.calls) == 1


# ── (b) one tool round-trip ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_one_tool_round_trip():
    session = Session()
    registry = ToolRegistry()

    @registry.tool()
    def time_now() -> dict:
        return {"now": "2026-01-01T00:00:00"}

    llm = _FakeLLM([
        # Iteration 1: emit a tool_call
        [
            _tc(0, id="c1", name="time_now", arguments=""),
            _tc(0, arguments="{}"),
            _finish("tool_calls"),
        ],
        # Iteration 2: emit the final text
        [_text("it is morning"), _finish("stop")],
    ])

    tokens: list[str] = []

    async def on_tok(t):
        tokens.append(t)

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    final = await stream_with_tools(
        llm, msgs,
        session=session, registry=registry, allowed_tools={"time_now"},
        ctx=_make_ctx(session), on_assistant_token=on_tok,
    )
    assert final == "it is morning"
    # Session history: assistant_tc + tool_result + assistant_text
    assert len(session.history) == 3
    assert session.history[0]["role"] == "assistant"
    assert session.history[0]["content"] is None
    assert session.history[0]["tool_calls"][0]["function"]["name"] == "time_now"
    assert session.history[1]["role"] == "tool"
    assert session.history[1]["tool_call_id"] == "c1"
    assert "2026-01-01" in session.history[1]["content"]
    assert session.history[2] == {"role": "assistant", "content": "it is morning"}
    # messages list mirrored:
    assert msgs[0]["role"] == "system"
    assert len(msgs) == 1 + 3  # system + same 3 history entries


# ── (c) two consecutive tool rounds ───────────────────────────────────


@pytest.mark.asyncio
async def test_two_tool_rounds():
    session = Session()
    registry = ToolRegistry()

    @registry.tool()
    def step(n: int) -> dict:
        return {"n": n}

    llm = _FakeLLM([
        [_tc(0, id="c1", name="step", arguments='{"n":1}'), _finish("tool_calls")],
        [_tc(0, id="c2", name="step", arguments='{"n":2}'), _finish("tool_calls")],
        [_text("done"), _finish("stop")],
    ])

    async def on_tok(t):
        pass

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    final = await stream_with_tools(
        llm, msgs,
        session=session, registry=registry, allowed_tools={"step"},
        ctx=_make_ctx(session), on_assistant_token=on_tok,
    )
    assert final == "done"
    # 2 tc + 2 tool + 1 text = 5 entries
    assert len(session.history) == 5
    assert session.history[-1]["content"] == "done"


# ── (d) cancel during tool dispatch rolls back BOTH lists ─────────────


@pytest.mark.asyncio
async def test_cancel_during_tool_dispatch_rolls_back():
    session = Session()
    session.add_user("pre-existing user msg")
    anchor_history = list(session.history)
    registry = ToolRegistry()

    @registry.tool(timeout_s=5.0)
    async def slow() -> dict:
        await asyncio.sleep(10)
        return {}

    llm = _FakeLLM([
        [_tc(0, id="c1", name="slow", arguments="{}"), _finish("tool_calls")],
    ])

    async def on_tok(t):
        pass

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}, *session.history]

    async def runner_coro():
        return await stream_with_tools(
            llm, msgs,
            session=session, registry=registry, allowed_tools={"slow"},
            ctx=_make_ctx(session), on_assistant_token=on_tok,
        )

    task = asyncio.create_task(runner_coro())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # Session rolled back to pre-runner state.
    assert session.history == anchor_history
    # Messages list mirrored — only system + pre-existing user msg.
    assert len(msgs) == 1 + len(anchor_history)
    assert msgs[0]["role"] == "system"


# ── (e) iteration cap rolls back ──────────────────────────────────────


@pytest.mark.asyncio
async def test_iteration_cap_rolls_back():
    session = Session()
    session.add_user("u1")
    anchor_history = list(session.history)
    registry = ToolRegistry()

    @registry.tool()
    def loop() -> dict:
        return {"again": True}

    # Always issue a tool_call → trigger cap after max_iterations.
    def iter_script():
        return [
            _tc(0, id="c", name="loop", arguments="{}"),
            _finish("tool_calls"),
        ]

    llm = _FakeLLM([iter_script() for _ in range(5)])

    bus_events: list[tuple[str, dict]] = []

    class _Bus:
        def emit(self, name, data):
            bus_events.append((name, data))

    ctx = ToolCallCtx(session=session, event_bus=_Bus())

    async def on_tok(t):
        pass

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}, *session.history]
    final = await stream_with_tools(
        llm, msgs,
        session=session, registry=registry, allowed_tools={"loop"},
        ctx=ctx, on_assistant_token=on_tok, max_iterations=3,
    )
    assert final == ""
    # Session rolled back to anchor.
    assert session.history == anchor_history
    # Messages list mirrored.
    assert len(msgs) == 1 + len(anchor_history)
    # iteration_limit event emitted.
    names = [n for n, _ in bus_events]
    assert "on_tool_iteration_limit" in names


# ── (f) invalid args JSON → error result, loop continues ──────────────


@pytest.mark.asyncio
async def test_invalid_args_json_continues_loop():
    session = Session()
    registry = ToolRegistry()

    @registry.tool()
    def f(x: int) -> dict:
        return {"x": x}

    llm = _FakeLLM([
        [_tc(0, id="c1", name="f", arguments="not-json"), _finish("tool_calls")],
        [_text("recovered"), _finish("stop")],
    ])

    async def on_tok(t):
        pass

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    final = await stream_with_tools(
        llm, msgs,
        session=session, registry=registry, allowed_tools={"f"},
        ctx=_make_ctx(session), on_assistant_token=on_tok,
    )
    assert final == "recovered"
    # Tool result should carry the JSON-decode error.
    tool_msg = next(m for m in session.history if m.get("role") == "tool")
    import json
    body = json.loads(tool_msg["content"])
    assert body["success"] is False
    assert "invalid arguments JSON" in body["error"]


# ── (g) iter >0 sets extra_body.prefix_cache=False (must-fix #1) ──────


@pytest.mark.asyncio
async def test_iter_gt_zero_disables_prefix_cache():
    session = Session()
    registry = ToolRegistry()

    @registry.tool()
    def f() -> dict:
        return {"ok": True}

    llm = _FakeLLM([
        [_tc(0, id="c1", name="f", arguments="{}"), _finish("tool_calls")],
        [_text("done"), _finish("stop")],
    ])

    async def on_tok(t):
        pass

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    _ = await stream_with_tools(
        llm, msgs,
        session=session, registry=registry, allowed_tools={"f"},
        ctx=_make_ctx(session), on_assistant_token=on_tok,
    )
    # First call: no extra_body forced; caller's llm_kwargs was empty.
    first_kw = llm.calls[0]["kwargs"]
    assert "extra_body" not in first_kw or "prefix_cache" not in (
        first_kw.get("extra_body") or {}
    )
    # Second call (iter >0): extra_body.prefix_cache must be False.
    second_kw = llm.calls[1]["kwargs"]
    assert second_kw["extra_body"]["prefix_cache"] is False


@pytest.mark.asyncio
async def test_iter_gt_zero_preserves_caller_extra_body():
    """A caller-supplied extra_body must survive the prefix_cache=False
    injection on iter >0."""
    session = Session()
    registry = ToolRegistry()

    @registry.tool()
    def f() -> dict:
        return {}

    llm = _FakeLLM([
        [_tc(0, id="c1", name="f", arguments="{}"), _finish("tool_calls")],
        [_text("ok"), _finish("stop")],
    ])

    async def on_tok(t):
        pass

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    await stream_with_tools(
        llm, msgs,
        session=session, registry=registry, allowed_tools={"f"},
        ctx=_make_ctx(session), on_assistant_token=on_tok,
        llm_kwargs={"extra_body": {"custom_flag": "keep_me"}},
    )
    second_kw = llm.calls[1]["kwargs"]
    assert second_kw["extra_body"]["custom_flag"] == "keep_me"
    assert second_kw["extra_body"]["prefix_cache"] is False
