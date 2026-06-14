"""Server-loop tool chain on the LIVE dev orin-nx SLV (TC-001/002 + MT-008).

Closes the production blind spot: prod reBot runs **server-loop** (the SLV's LLM
selects tools and proxies execution back via SERVER_TOOL_CALL), but every other
e2e is client-loop, so the full tool path had zero real-SLV coverage. These
drive it end to end against the real SLV + real edge-llm LLM:

    in-process agent advertises tools (CLIENT_TOOL_ADVERTISE)
      → real SLV /v2v server-loop runs the real edge-llm LLM
      → the LLM picks the advertised tool
      → SLV sends SERVER_TOOL_CALL back
      → agent._handle_server_tool_call dispatches it against the local registry
         (== the stubs below record the call)

The assertion face is the stub tool being invoked — the dashboard /ws does NOT
expose SERVER_TOOL_CALL, but the in-process agent's own registry does. Fresh
registries holding only stubs are swapped in so the advertise payload is
controlled and the command has an unambiguous target.

Scenarios:
  * TC-001  test_server_loop_tool_chain            — single tool advertised, a
            command dispatches it.
  * TC-002  test_server_loop_selects_correct_tool  — TWO tools advertised; the
            command must select the right one and NOT the other.
  * MT-008  test_server_loop_chat_then_command     — a chitchat turn fires NO
            tool (false-trigger guard); a following command turn fires it, on
            one persistent server-loop session.

────────────────────────────────────────────────────────────────────────────
GATED: skipped unless ``OVS_E2E_SERVER_LOOP=1``. The dev orin-nx SLV runs
client-loop by default; these only pass when it has been flipped to server-loop.
Setup (verified 2026-06-14):

  1. On orin-nx, clone the relaunch script and add the server-loop env. The SLV
     is on a bridge network, so the edge-llm LLM must be reached via the host
     gateway, NOT localhost (edge_llm_base_url() defaults to the container's own
     127.0.0.1:8000 — server/core/edge_llm_backend.py:51):
       cp ~/relaunch_seeed_voice.sh ~/relaunch_serverloop.sh
       # in the edgellm branch's `docker run`, add:
       #   -e OVS_V2V_ENGINE=voxedge -e OVS_V2V_SERVER_LOOP=1 \
       #   -e EDGE_LLM_BASE_URL=http://172.17.0.1:8000/v1
       bash ~/relaunch_serverloop.sh edgellm jetson-qwen3asr-matcha-nx
  2. Run:
       OVS_E2E_SERVER_LOOP=1 \
       env -u http_proxy -u https_proxy NO_PROXY='100.82.225.102,localhost,127.0.0.1' \
         uv run pytest tests/e2e/test_server_loop_tool_e2e.py -v -s
  3. ALWAYS restore client-loop afterwards (shared dev box):
       bash ~/relaunch_seeed_voice.sh edgellm jetson-qwen3asr-matcha-nx
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import replace

import aiohttp
import pytest

from .conftest import WAV_DIR
from .fake_audio import ScriptedAudioIO
from .probe import AgentProbe
from .test_rebot_voice_capture import _rebot_voice_config

pytestmark = pytest.mark.skipif(
    os.environ.get("OVS_E2E_SERVER_LOOP") != "1",
    reason=(
        "requires the dev orin-nx SLV flipped to server-loop "
        "(OVS_V2V_ENGINE=voxedge + OVS_V2V_SERVER_LOOP=1 + EDGE_LLM_BASE_URL "
        "→ host gateway); see module docstring. Set OVS_E2E_SERVER_LOOP=1 to run."
    ),
)


def _stub_registry(sink: list[dict], tools: tuple[str, ...] = ("grasp_object",)):
    """Build a registry of recording stubs. Each dispatch appends
    ``{"tool": name, "args": {...}}`` to ``sink`` (the assertion surface)."""
    from ovs_agent.tools.registry import ToolRegistry

    reg = ToolRegistry()

    if "grasp_object" in tools:
        @reg.tool(
            name="grasp_object",
            description=(
                "抓取/拿起指定的物体。当用户要求抓、拿、抓起某个东西时调用。"
                "Grab/pick up a named object."
            ),
            preamble_text="好的，正在抓取。",
        )
        def grasp_object(object_name: str) -> dict:  # noqa: ANN001
            sink.append({"tool": "grasp_object", "args": {"object_name": object_name}})
            return {"started": True, "object_name": object_name}

    if "go_home" in tools:
        @reg.tool(
            name="go_home",
            description="让机械臂回到原位/home 姿态。当用户要求回家、回原位时调用。",
            preamble_text="好的，正在回原位。",
        )
        def go_home() -> dict:
            sink.append({"tool": "go_home", "args": {}})
            return {"started": True, "action": "go_home"}

    return reg


async def _wake(port: int) -> None:
    async with aiohttp.ClientSession() as s:
        async with s.post(f"http://127.0.0.1:{port}/api/control/wake") as r:
            assert r.status == 200


async def _wait_for(pred, timeout: float) -> bool:
    """Poll ``pred()`` until truthy or timeout. Returns whether it fired."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if pred():
            return True
        await asyncio.sleep(0.3)
    return False


@asynccontextmanager
async def _server_loop_agent(test_config, registry):
    """Spawn an in-process reBot agent in server-loop mode with ``registry``
    advertised, wake it, and yield ``(app, audio, probe)``. Mirrors the proven
    single-turn harness; handles teardown."""
    cfg = replace(_rebot_voice_config(test_config), server_loop=True)

    from ovs_agent.apps.multi_mode.app import MultiModeApp

    app = MultiModeApp(cfg)
    app.tool_registry = registry  # advertise our stubs at boot

    audio = ScriptedAudioIO([])
    app.audio = audio
    ready = asyncio.Event()
    audio.ready_event = ready

    async def _wait_advertise_ready() -> None:
        while True:
            ev = getattr(app, "_advertise_ready", None)
            if ev is not None:
                await ev.wait()
                break
            await asyncio.sleep(0.02)
        ready.set()

    run_task = asyncio.create_task(app.run(), name="server-loop-e2e-run")
    ready_task = asyncio.create_task(_wait_advertise_ready())
    probe = AgentProbe(port=cfg.metadata["dashboard_port"])
    try:
        await probe.connect()
        await _wake(cfg.metadata["dashboard_port"])
        await probe.wait_event("on_wake", timeout=10)
        await asyncio.sleep(0.6)  # clear wake-tone mic suppression
        yield app, audio, probe
    finally:
        ready_task.cancel()
        try:
            await ready_task
        except BaseException:
            pass
        try:
            app.request_shutdown()
        except Exception:
            pass
        for closer in (audio.close, probe.close):
            try:
                await closer()
            except Exception:
                pass
        try:
            await asyncio.wait_for(run_task, timeout=8)
        except BaseException:
            run_task.cancel()
            try:
                await run_task
            except BaseException:
                pass


def _assistant_done_count(probe) -> int:
    return sum(1 for e in probe.events if e.get("event") == "on_assistant_done")


# ── TC-001: single advertised tool → command dispatches it ────────────


@pytest.mark.asyncio
async def test_server_loop_tool_chain(test_config):
    calls: list[dict] = []
    reg = _stub_registry(calls, tools=("grasp_object",))
    async with _server_loop_agent(test_config, reg) as (app, audio, probe):
        audio.inject(WAV_DIR / "tts_q_grab_box.wav")  # "抓盒子"
        fired = await _wait_for(lambda: bool(calls), timeout=45)
        evts = [e.get("event") for e in probe.events][-30:]
        assert fired, f"server-loop did not dispatch the tool. calls={calls} events={evts}"
        assert calls[0]["tool"] == "grasp_object", calls


# ── TC-002: two tools advertised → the RIGHT one is selected ──────────


@pytest.mark.asyncio
async def test_server_loop_selects_correct_tool(test_config):
    """TC-002: with both ``grasp_object`` and ``go_home`` advertised, "抓盒子"
    must select grasp_object and NOT go_home. Guards against the LLM firing the
    wrong tool when several are available."""
    calls: list[dict] = []
    reg = _stub_registry(calls, tools=("grasp_object", "go_home"))
    async with _server_loop_agent(test_config, reg) as (app, audio, probe):
        audio.inject(WAV_DIR / "tts_q_grab_box.wav")  # "抓盒子" → grasp, not home
        fired = await _wait_for(lambda: bool(calls), timeout=45)
        assert fired, f"no tool dispatched. calls={calls}"
        names = [c["tool"] for c in calls]
        assert "grasp_object" in names, f"expected grasp_object, got {names}"
        assert "go_home" not in names, f"go_home wrongly fired: {names}"


# ── MT-008: chitchat fires no tool; a following command does ──────────


@pytest.mark.asyncio
async def test_server_loop_chat_then_command(test_config):
    """MT-008: on one persistent server-loop session, a chitchat turn ("你好")
    must NOT fire a tool (false-trigger guard), while a following command turn
    ("抓盒子") must. Both the dialogue path and the tool path share the WS."""
    calls: list[dict] = []
    reg = _stub_registry(calls, tools=("grasp_object",))
    async with _server_loop_agent(test_config, reg) as (app, audio, probe):
        # Turn 1 — chitchat. Expect an assistant reply, NO tool call.
        audio.inject(WAV_DIR / "hello.wav")  # "你好"
        got_reply = await _wait_for(lambda: _assistant_done_count(probe) >= 1, timeout=45)
        assert got_reply, "chitchat turn produced no assistant reply"
        assert calls == [], f"chitchat must not fire a tool, but did: {calls}"

        # Turn 2 — command. Expect grasp_object to fire.
        audio.inject(WAV_DIR / "tts_q_grab_box.wav")  # "抓盒子"
        fired = await _wait_for(lambda: bool(calls), timeout=45)
        evts = [e.get("event") for e in probe.events][-30:]
        assert fired, f"command turn did not dispatch the tool. calls={calls} events={evts}"
        assert calls[0]["tool"] == "grasp_object", calls
