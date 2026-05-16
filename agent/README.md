# openvoicestream-agent

Agent-layer client on top of [Seeed Local Voice (SLV)](../README.md).
SLV provides `/v2v/stream` (ASR + TTS + VAD + barge-in) — this package
adds the LLM, the session, the plugin system, and the audio I/O.

## Architecture

```
+----------+   PCM       +-----------+   text     +---------+
|   mic    | -----------> |    SLV    | <--------- |  LLM    |
+----------+              |  /v2v/    |  tokens    | (edge-  |
                          |  stream   | ---------> |  llm)   |
+----------+   PCM        |           |            +---------+
| speaker  | <----------- |           |
+----------+              +-----------+
```

ONE persistent WebSocket to SLV per App lifetime. SLV does the
ASR / VAD / sentence splitting / TTS server-side. The agent only
orchestrates LLM streaming and barge-in.

## HARD invariants (do not violate)

1. **Single persistent WS** to `/v2v/stream`, opened with
   `multi_utterance: true`. NOT a new connection per turn.
2. **LLM tokens go DIRECTLY to SLV** as `text` frames. The agent does
   NOT do client-side sentence buffering — SLV's `SentenceBuffer` runs
   server-side and is the single source of truth.
3. **Session history is sent FULL** to the LLM, no client-side
   trimming or summarization. Edge-LLM's prefix cache is the
   optimization.
4. **Barge-in**: on `asr_partial` while TTS is playing, send
   `{"type":"abort"}` to SLV and drain the local playback queue.
5. **Plugin hooks are observer broadcasts**, not routers.
   `BaseApp.on_user_utterance` is the single router.
6. **Protocol constants come from `app.core.v2v`** (SLV's module).
   Never redeclare them in the agent.

## Layout

```
agent/
├── pyproject.toml
├── openvoicestream_agent/
│   ├── __init__.py        # sys.path shim so `from app.core.v2v ...` resolves
│   ├── app_base.py        # BaseApp orchestrator
│   ├── audio_io.py        # sounddevice mic + speaker
│   ├── cli.py             # `ovs-agent run <app>`
│   ├── config.py          # YAML loader, env var substitution
│   ├── event_bus.py       # pub/sub
│   ├── plugin.py          # Plugin ABC with 7 observer hooks
│   ├── session.py         # OpenAI-format history (no trimming)
│   ├── slv_client.py      # persistent WS to /v2v/stream
│   └── llm/
│       ├── base.py
│       ├── openai_compat.py
│       └── edge_llm.py    # adds save_system_prompt_kv_cache / prefix_cache
└── apps/
    └── dialogue/
        ├── app.py         # DialogueApp -- voice chat
        └── config.yaml
```

## Quick start

Prereqs:

- SLV running locally with `/v2v/stream` exposed (default `ws://localhost:8621`)
- edge-llm-chat-service running at `http://localhost:8000/v1` (OpenAI compatible)

```bash
cd agent
uv sync
uv run pytest tests/ -v
uv run ovs-agent run dialogue
```

Env overrides:

```bash
export OVS_SLV_URL="ws://192.168.1.100:8621/v2v/stream"
export OVS_LLM_URL="http://192.168.1.100:8000/v1"
export OVS_LLM_MODEL="qwen2.5-3b-instruct"
uv run ovs-agent run dialogue
```

## Why the `sys.path` shim?

SLV has no `pyproject.toml`, so we can't `pip install` it.
`openvoicestream_agent/__init__.py` prepends the SLV repo root to
`sys.path` at import time so `from app.core.v2v import CLIENT_TEXT, ...`
works without restructuring SLV. The Docker image accomplishes the same
thing by copying SLV's `app/` directory next to the agent and setting
`PYTHONPATH=/opt/slv`.

## Writing a plugin

```python
from openvoicestream_agent import Plugin

class LoggerPlugin(Plugin):
    name = "logger"

    async def on_user_utterance(self, text: str) -> None:
        print(f"user said: {text}")

    async def on_assistant_sentence(self, sentence: str) -> None:
        print(f"assistant said: {sentence}")
```

Register before `app.run()`:

```python
app = DialogueApp(config)
app.register(LoggerPlugin(app))
await app.run()
```

## Writing a new App

Subclass `BaseApp` and implement `on_user_utterance`:

```python
from openvoicestream_agent import BaseApp

class MyApp(BaseApp):
    async def on_user_utterance(self, text: str) -> None:
        # Decide what to say, then stream to SLV.
        async for token in self.llm.stream(
            self.session.messages(self.config.system_prompt),
            session=self.session,
        ):
            await self.slv.send_text(token)
        await self.slv.flush_tts()
```

Drop a `config.yaml` next to `app.py` under `apps/<name>/` and run with
`ovs-agent run <name>`.
