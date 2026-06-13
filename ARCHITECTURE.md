# Architecture

A map of how the pieces fit, written for someone who just cloned the repo and
wants to understand, run, and extend it. For feature/endpoint detail see
[README.md](README.md); for a dev box see [DEVELOP.md](DEVELOP.md); for
configuration see [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## The three repositories

This product is **not** self-contained — it spans three sibling repos plus two
git submodules. Knowing the split is the single most important onboarding fact.

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │  seeed-local-voice   (THIS repo — the product / deployment)          │
 │                                                                       │
 │   server/   FastAPI service: stable HTTP+WS API, backend registry,   │
 │             hot-reload, admission control, profiles                   │
 │   agent/    ovs_agent: a SEPARATE package + container. Mic→speaker    │
 │             apps (voice_arm, voice_rebot_arm, multi_mode, …) that     │
 │             talk to the server over /v2v/stream                       │
 │   configs/  device profiles (JSON) + leaf composition (YAML)          │
 │   deploy/   per-device Dockerfiles + compose + the voxedge wheel      │
 └───────────────┬──────────────────────────────────────┬──────────────┘
                 │ imports voxedge.*                     │ submodules
                 ▼                                       ▼
 ┌──────────────────────────────────────┐   ┌──────────────────────────────┐
 │  voxedge   (../voxedge)              │   │ third_party/                 │
 │  "Pipecat for the edge"             │   │  rkvoice-stream  (RK runtime)│
 │  PURE-PYTHON library, numpy-only    │   │  qwen3-edgellm-jetson        │
 │  core, NO env reads, NO CUDA/torch  │   └──────────────────────────────┘
 │  at import time.                    │
 │   engine/      conversation loop    │
 │   backends/    ASR/TTS/VAD/LLM ABCs │
 │                + jetson/ rk/ sherpa/│
 │   transport/   InProcess + WebSocket│
 │   capabilities/ punctuation, speaker│
 └───────────────┬─────────────────────┘
                 │ heavy backends (jetson/) shell out to the native engine
                 ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │  voxedge-engine   (../voxedge-engine)                                  │
 │  Thin OVERLAY over NVIDIA TensorRT-Edge-LLM (Apache-2.0).             │
 │  UPSTREAM_PIN + addon/ (40 new files) + patches/ (8 themed) +         │
 │  DIVERGENCE.md ledger. Full source is reconstructed at build time;    │
 │  produces the prebuilt TRT worker the jetson backends drive.          │
 └──────────────────────────────────────────────────────────────────────┘
```

**Why the split.** voxedge is the reusable, open-core library — it must stay
`pip install`-able on any laptop with no GPU. The product (this repo) is the
edge-device-specific shell: deployment, profiles, the agent apps, and the wiring
that turns env/profile config into voxedge backend instances. The engine repo
isolates the heavyweight NVIDIA fork so its build/divergence is governed
separately from product code.

## Layer responsibilities (who owns what)

| Concern | Lives in | Notes |
|---|---|---|
| HTTP/WS API contract | `server/main.py` | `/asr/stream`, `/tts/stream`, `/v2v/stream`, admin, health |
| Backend registry + hot-reload | `server/core/{asr,tts}_backend.py`, `backend_manager.py` | wraps voxedge backends; state machine for live swaps |
| env/profile → backend config | `server/core/voxedge_backend_config.py` | voxedge itself is env-free; this is the adapter |
| Conversation orchestration (VAD→ASR→LLM→TTS) | `voxedge/engine/conversation.py` | the engine; used by `/v2v` when `OVS_V2V_ENGINE=voxedge` |
| Backend interfaces (ABCs) | `voxedge/backends/base.py` | `ASRBackend`, `TTSBackend`, `VADBackend`, `LLMBackend` |
| Concrete inference | `voxedge/backends/{jetson,rk,sherpa}/` | lazy-import heavy runtimes; never at module load |
| Native TRT engine | `voxedge-engine` | prebuilt worker; product/voxedge only drive it |
| Mic/speaker + app logic | `agent/ovs_agent/apps/*` | each app is a plugin-based client of `/v2v/stream` |
| Device deployment | `deploy/` + `configs/` | Dockerfiles, compose, profiles |

### The two processes

There are exactly **two production processes**, and they are decoupled by the
`/v2v/stream` WebSocket:

1. **server** (`uvicorn server.main:app`, port 8000 in-container) — owns the
   models. Stateless across turns; the LLM+tool loop runs here when
   `OVS_V2V_ENGINE=voxedge` ("server-loop" architecture).
2. **agent** (`ovs-agent run <app>`) — owns the device I/O (mic, speaker,
   wakeword, robot arm). It advertises tools and executes tool calls the server
   dispatches back; it never orchestrates the LLM itself.

This is why the same server serves a dumb captioning client and a robot arm: the
intelligence is server-side, the device-specific actuation is agent-side.

## How voxedge reaches a device (the wheel flow)

voxedge is **not vendored** into this repo and **not on PyPI**. It flows in as a
built wheel:

```
../voxedge (source)  ──scripts/build_voxedge_wheel.sh──▶  deploy/wheels/voxedge-0.0.1a0-py3-none-any.whl
                                                          + deploy/wheels/voxedge.BUILD.txt (source SHA, date)
                                                                    │
                                       Dockerfile.{jetson,rk,rpi}   │ pip install
                                                                    ▼
                                                          device image  ──▶  orin-nano / rk3576 / rpi
```

The wheel (a ~200KB pure-Python artifact) **is committed** to git, so a fresh
checkout can `docker build` with no "rebuild the wheel first" step (the images
have no `git` for a `git+https` install, and we don't publish to PyPI). The rest
of `deploy/wheels/` is git-ignored; rebuild + commit the wheel when voxedge
changes (`scripts/build_voxedge_wheel.sh`, see DEVELOP.md). The `.BUILD.txt`
sidecar records which voxedge commit a given wheel came from. The
`Dockerfile.*.voxedge-patch` images
`--force-reinstall` just this wheel onto a running base image for fast,
Python-only iteration without a full rebuild. See [DEVELOP.md](DEVELOP.md).

## Run it locally (no GPU)

The whole conversation engine runs on a laptop with mock backends — this is the
fastest way to understand the dataflow:

```bash
scripts/dev-setup.sh                                  # editable voxedge + deps
( cd ../voxedge && pytest voxedge/tests/test_engine_inprocess.py -q )
```

That test wires the real engine with fakes and asserts the V2V event contract:

```python
from voxedge.backends.mock import MockASR, MockLLM, MockTTS, MockVAD
from voxedge.engine import ConversationEngine
from voxedge.transport import InProcessTransport

engine = ConversationEngine(backends={
    "asr": MockASR(transcript="hello world"),
    "vad": MockVAD(), "tts": MockTTS(), "llm": MockLLM(),
})
# drive it through InProcessTransport — no CUDA, no sockets, no models
```

To exercise the HTTP API instead, start the server and use the clients in
`examples/` (`stream_tts_to_wav.py`, `v2v_tts_only.py`).

## Extending it

- **New ASR/TTS backend** → implement the ABC in `voxedge/backends/<platform>/`
  (config dataclass + lazy heavy import), declare it in `server/core/{asr,tts}_backend.py`.
  See `backends/sherpa/asr.py` as the simplest template.
- **New device profile** → add `configs/profiles/<name>.json`; see
  [docs/CONFIGURATION.md](docs/CONFIGURATION.md).
- **New agent app** → add `agent/ovs_agent/apps/<name>/` with an `App` subclass
  and plugins; `multi_mode` is the reference.
- **New tool** → register with the engine's `ToolRegistry`; type hints become the
  JSON schema automatically.

## Known structural debt (tracked, not hidden)

- **`server/core/coordinator.py` & `capability_resolver.py` are migration twins**
  of `voxedge/engine/{coordinator,capability_resolver}.py`. The server copies are
  profile-driven and currently wired into the HTTP path; the voxedge copies are
  the env-free target the product is converging onto. They share a spec
  (`docs/specs/concurrency-capability-framework.md`). Don't add features to the
  server copies — see their header comments.
- **`voxedge/engine/conversation.py` is ~1600 LOC** — the orchestration is
  correct and well-commented but concentrated. A staged split is designed in
  [docs/plans/conversation-split.md](docs/plans/conversation-split.md); it is
  deliberately **not** done yet because that file is shared with the production
  robot-arm stack and needs the full perf/behavioral bench gate, not a smoke test.
- **Config has two layers** — mature flat JSON profiles, plus an optional newer
  "leaf composition" YAML system gated behind a profile key. See
  [docs/CONFIGURATION.md](docs/CONFIGURATION.md).
