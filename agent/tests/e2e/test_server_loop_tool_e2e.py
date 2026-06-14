"""Server-loop tool chain on the LIVE dev orin-nx SLV (TC-001 real-SLV).

Closes the production blind spot: prod reBot runs **server-loop** (the SLV's LLM
selects tools and proxies execution back via SERVER_TOOL_CALL), but every other
e2e is client-loop, so the full tool path had zero real-SLV coverage. This
verifies it end to end against the real SLV + real edge-llm LLM:

    in-process agent advertises tools (CLIENT_TOOL_ADVERTISE)
      → real SLV /v2v server-loop runs the real edge-llm LLM
      → the LLM picks the advertised tool
      → SLV sends SERVER_TOOL_CALL back
      → agent._handle_server_tool_call dispatches it against the local registry
         (== the stub below records the call)

The assertion face is the stub tool being invoked — the dashboard /ws does NOT
expose SERVER_TOOL_CALL, but the in-process agent's own registry does. A FRESH
registry holding ONLY a `grasp_object` stub is swapped in, so the advertise
payload carries exactly one tool and "抓盒子" has an unambiguous target.

────────────────────────────────────────────────────────────────────────────
GATED: skipped unless ``OVS_E2E_SERVER_LOOP=1``. The dev orin-nx SLV runs
client-loop by default; this test only passes when it has been flipped to
server-loop. Setup (verified 2026-06-14):

  1. On orin-nx, clone the relaunch script and add the server-loop env. The
     SLV is on a bridge network, so the edge-llm LLM must be reached via the
     host gateway, NOT localhost:
       cp ~/relaunch_seeed_voice.sh ~/relaunch_serverloop.sh
       # in the edgellm branch's `docker run`, add:
       #   -e OVS_V2V_ENGINE=voxedge -e OVS_V2V_SERVER_LOOP=1 \
       #   -e EDGE_LLM_BASE_URL=http://172.17.0.1:8000/v1
       bash ~/relaunch_serverloop.sh edgellm jetson-qwen3asr-matcha-nx
     (edge_llm_base_url() defaults to http://127.0.0.1:8000/v1 = the container
     itself, which is NOT the LLM — server/core/edge_llm_backend.py:51.)
  2. Run:
       OVS_E2E_SERVER_LOOP=1 \
       env -u http_proxy -u https_proxy NO_PROXY='100.82.225.102,localhost,127.0.0.1' \
         uv run pytest tests/e2e/test_server_loop_tool_e2e.py -v -s
  3. ALWAYS restore client-loop afterwards (shared dev box):
       bash ~/relaunch_seeed_voice.sh edgellm jetson-qwen3asr-matcha-nx

Live-verified run produced SLV-side proof:
  voxedge.engine.conversation: tool_advertise: registered 1 remote tool(s)
    ['grasp_object']
  voxedge tool loop: round=0 ... finish=tool_calls n_tools=1
  voxedge tool loop: tool=grasp_object dispatch=0.009s
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import asyncio
import os
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


# Records every dispatch of the stub tool. Module-level so the closure and the
# assertion share state regardless of how the registry copies the fn.
CALLS: list[dict] = []


def _build_stub_registry():
    from ovs_agent.tools.registry import ToolRegistry

    reg = ToolRegistry()

    @reg.tool(
        name="grasp_object",
        description=(
            "抓取/拿起指定的物体。当用户要求抓、拿、抓起某个东西时调用。"
            "Grab/pick up a named object."
        ),
        preamble_text="好的，正在抓取。",
    )
    def grasp_object(object_name: str) -> dict:  # noqa: ANN001
        CALLS.append({"object_name": object_name})
        # parallel-mode ack shape so the server-loop reports ok=True.
        return {"started": True, "object_name": object_name}

    return reg


async def _wake(port: int) -> None:
    async with aiohttp.ClientSession() as s:
        async with s.post(f"http://127.0.0.1:{port}/api/control/wake") as r:
            assert r.status == 200


@pytest.mark.asyncio
async def test_server_loop_tool_chain(test_config):
    CALLS.clear()
    # Production reBot gate path + force server-loop ON. The advertise payload
    # then carries our stub; the SLV runs the real LLM + emits SERVER_TOOL_CALL.
    cfg = replace(_rebot_voice_config(test_config), server_loop=True)

    from ovs_agent.apps.multi_mode.app import MultiModeApp

    app = MultiModeApp(cfg)
    # Swap in the stub registry BEFORE run() so boot-time advertise uses it.
    app.tool_registry = _build_stub_registry()

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
        # Wake, clear wake-tone suppression, then feed the grasp command.
        await _wake(cfg.metadata["dashboard_port"])
        await probe.wait_event("on_wake", timeout=10)
        await asyncio.sleep(0.6)
        audio.inject(WAV_DIR / "tts_q_grab_box.wav")  # "抓盒子"

        # Wait up to 40s for the stub to be dispatched (ASR + LLM round-trip).
        deadline = asyncio.get_event_loop().time() + 40
        while asyncio.get_event_loop().time() < deadline:
            if CALLS:
                break
            await asyncio.sleep(0.5)

        # Diagnostics regardless of pass/fail.
        utt = [e.get("data") for e in probe.events if e.get("event") == "on_user_utterance"]
        evt_names = [e.get("event") for e in probe.events][-40:]
        print(f"\n[server-loop-e2e] stub CALLS = {CALLS}")
        print(f"[server-loop-e2e] utterances = {utt}")
        print(f"[server-loop-e2e] recent events = {evt_names}")
        print(f"[server-loop-e2e] errors = {probe.errors}")

        assert CALLS, (
            "server-loop did NOT dispatch the advertised tool. "
            f"utterances={utt} events={evt_names} errors={probe.errors}"
        )
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
        except BaseException:
            run_task.cancel()
            try:
                await run_task
            except BaseException:
                pass
